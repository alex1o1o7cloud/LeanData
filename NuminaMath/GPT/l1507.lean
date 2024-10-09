import Mathlib

namespace min_value_expression_l1507_150758

theorem min_value_expression (x y : ℝ) (hx : |x| < 1) (hy : |y| < 2) (hxy : x * y = 1) : 
  ∃ k, k = 4 ∧ (∀ z, z = (1 / (1 - x^2) + 4 / (4 - y^2)) → z ≥ k) :=
sorry

end min_value_expression_l1507_150758


namespace min_value_of_a_b_c_l1507_150793

variable (a b c : ℕ)
variable (x1 x2 : ℝ)

axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a * x1^2 + b * x1 + c = 0
axiom h5 : a * x2^2 + b * x2 + c = 0
axiom h6 : |x1| < 1/3
axiom h7 : |x2| < 1/3

theorem min_value_of_a_b_c : a + b + c = 25 :=
by
  sorry

end min_value_of_a_b_c_l1507_150793


namespace person_age_l1507_150738

theorem person_age (x : ℕ) (h : 4 * (x + 3) - 4 * (x - 3) = x) : x = 24 :=
by {
  sorry
}

end person_age_l1507_150738


namespace minimize_total_price_l1507_150772

noncomputable def total_price (a : ℝ) (m x : ℝ) : ℝ :=
  a * ((m / 2 + x)^2 + (m / 2 - x)^2)

theorem minimize_total_price (a m : ℝ) : 
  ∃ y : ℝ, (∀ x, total_price a m x ≥ y) ∧ y = total_price a m 0 :=
by
  sorry

end minimize_total_price_l1507_150772


namespace citizens_own_a_cat_l1507_150783

theorem citizens_own_a_cat (p d : ℝ) (n : ℕ) (h1 : p = 0.60) (h2 : d = 0.50) (h3 : n = 100) : 
  (p * n - d * p * n) = 30 := 
by 
  sorry

end citizens_own_a_cat_l1507_150783


namespace least_positive_multiple_of_primes_l1507_150703

theorem least_positive_multiple_of_primes :
  11 * 13 * 17 * 19 = 46189 :=
by
  sorry

end least_positive_multiple_of_primes_l1507_150703


namespace unique_solution_system_l1507_150792

noncomputable def f (x : ℝ) := 4 * x ^ 3 + x - 4

theorem unique_solution_system :
  (∃ x y z : ℝ, y^2 = 4*x^3 + x - 4 ∧ z^2 = 4*y^3 + y - 4 ∧ x^2 = 4*z^3 + z - 4) ↔
  (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  sorry

end unique_solution_system_l1507_150792


namespace symmetric_point_coordinates_l1507_150736

def point_symmetric_to_x_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (x, -y, -z)

theorem symmetric_point_coordinates :
  point_symmetric_to_x_axis (-2, 1, 4) = (-2, -1, -4) := by
  sorry

end symmetric_point_coordinates_l1507_150736


namespace second_interest_rate_l1507_150746

theorem second_interest_rate (P1 P2 : ℝ) (r : ℝ) (total_amount total_income: ℝ) (h1 : total_amount = 2500)
  (h2 : P1 = 1500.0000000000007) (h3 : total_income = 135) :
  P2 = total_amount - P1 →
  P1 * 0.05 = 75 →
  P2 * r = 60 →
  r = 0.06 :=
sorry

end second_interest_rate_l1507_150746


namespace mean_score_classes_is_82_l1507_150744

theorem mean_score_classes_is_82
  (F S : ℕ)
  (f s : ℕ)
  (hF : F = 90)
  (hS : S = 75)
  (hf_ratio : f * 6 = s * 5)
  (hf_total : f + s = 66) :
  ((F * f + S * s) / (f + s) : ℚ) = 82 :=
by
  sorry

end mean_score_classes_is_82_l1507_150744


namespace intersection_of_A_and_B_l1507_150796

open Set

noncomputable def A : Set ℤ := {1, 3, 5, 7}
noncomputable def B : Set ℤ := {x | 2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by
  sorry

end intersection_of_A_and_B_l1507_150796


namespace find_like_term_l1507_150722

-- Definition of the problem conditions
def monomials : List (String × String) := 
  [("A", "-2a^2b"), 
   ("B", "a^2b^2"), 
   ("C", "ab^2"), 
   ("D", "3ab")]

-- A function to check if two terms can be combined (like terms)
def like_terms(a b : String) : Prop :=
  a = "a^2b" ∧ b = "-2a^2b"

-- The theorem we need to prove
theorem find_like_term : ∃ x, x ∈ monomials ∧ like_terms "a^2b" (x.2) ∧ x.2 = "-2a^2b" :=
  sorry

end find_like_term_l1507_150722


namespace sophie_total_spend_l1507_150741

def total_cost_with_discount_and_tax : ℝ :=
  let cupcakes_price := 5 * 2
  let doughnuts_price := 6 * 1
  let apple_pie_price := 4 * 2
  let cookies_price := 15 * 0.60
  let chocolate_bars_price := 8 * 1.50
  let soda_price := 12 * 1.20
  let gum_price := 3 * 0.80
  let chips_price := 10 * 1.10
  let total_before_discount := cupcakes_price + doughnuts_price + apple_pie_price + cookies_price + chocolate_bars_price + soda_price + gum_price + chips_price
  let discount := 0.10 * total_before_discount
  let subtotal_after_discount := total_before_discount - discount
  let sales_tax := 0.06 * subtotal_after_discount
  let total_cost := subtotal_after_discount + sales_tax
  total_cost

theorem sophie_total_spend :
  total_cost_with_discount_and_tax = 69.45 :=
sorry

end sophie_total_spend_l1507_150741


namespace triangle_area_transform_l1507_150785

-- Define the concept of a triangle with integer coordinates
structure Triangle :=
  (A : ℤ × ℤ)
  (B : ℤ × ℤ)
  (C : ℤ × ℤ)

-- Define the area of a triangle using determinant
def triangle_area (T : Triangle) : ℤ :=
  let ⟨(x1, y1), (x2, y2), (x3, y3)⟩ := (T.A, T.B, T.C)
  abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

-- Define a legal transformation for triangles
def legal_transform (T : Triangle) : Set Triangle :=
  { T' : Triangle |
    (∃ c : ℤ, 
      (T'.A = (T.A.1 + c * (T.B.1 - T.C.1), T.A.2 + c * (T.B.2 - T.C.2)) ∧ T'.B = T.B ∧ T'.C = T.C) ∨
      (T'.A = T.A ∧ T'.B = (T.B.1 + c * (T.A.1 - T.C.1), T.B.2 + c * (T.A.2 - T.C.2)) ∧ T'.C = T.C) ∨
      (T'.A = T.A ∧ T'.B = T.B ∧ T'.C = (T.C.1 + c * (T.A.1 - T.B.1), T.C.2 + c * (T.A.2 - T.B.2)))) }

-- Proposition that any two triangles with equal area can be legally transformed into each other
theorem triangle_area_transform (T1 T2 : Triangle) (h : triangle_area T1 = triangle_area T2) :
  ∃ (T' : Triangle), T' ∈ legal_transform T1 ∧ triangle_area T' = triangle_area T2 :=
sorry

end triangle_area_transform_l1507_150785


namespace find_larger_number_l1507_150775

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 :=
sorry

end find_larger_number_l1507_150775


namespace interval_of_decrease_for_f_x_plus_1_l1507_150731

def f_prime (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem interval_of_decrease_for_f_x_plus_1 : 
  ∀ x, (f_prime (x + 1) < 0 ↔ 0 < x ∧ x < 2) :=
by 
  intro x
  sorry

end interval_of_decrease_for_f_x_plus_1_l1507_150731


namespace value_of_a5_l1507_150716

theorem value_of_a5 {a_1 a_3 a_5 : ℤ} (n : ℕ) (hn : n = 8) (h1 : (1 - x)^n = 1 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8) (h_ratio : a_1 / a_3 = 1 / 7) :
  a_5 = -56 := 
sorry

end value_of_a5_l1507_150716


namespace max_plus_min_eq_four_l1507_150729

theorem max_plus_min_eq_four {g : ℝ → ℝ} (h_odd_function : ∀ x, g (-x) = -g x)
  (M m : ℝ) (h_f : ∀ x, 2 + g x ≤ M) (h_f' : ∀ x, m ≤ 2 + g x) :
  M + m = 4 :=
by
  sorry

end max_plus_min_eq_four_l1507_150729


namespace arithmetic_progression_sum_at_least_66_l1507_150757

-- Define the sum of the first n terms of an arithmetic progression
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

-- Define the conditions for the arithmetic progression
def arithmetic_prog_conditions (a1 d : ℤ) (n : ℕ) :=
  sum_first_n_terms a1 d n ≥ 66

-- The main theorem to prove
theorem arithmetic_progression_sum_at_least_66 (n : ℕ) :
  (n >= 3 ∧ n <= 14) → arithmetic_prog_conditions 25 (-3) n :=
by
  sorry

end arithmetic_progression_sum_at_least_66_l1507_150757


namespace volunteer_hours_per_year_l1507_150779

def volunteers_per_month : ℕ := 2
def hours_per_session : ℕ := 3
def months_per_year : ℕ := 12

theorem volunteer_hours_per_year :
  volunteers_per_month * months_per_year * hours_per_session = 72 :=
by
  -- Proof is omitted
  sorry

end volunteer_hours_per_year_l1507_150779


namespace tip_percentage_l1507_150787

theorem tip_percentage 
  (total_bill : ℕ) 
  (silas_payment : ℕ) 
  (remaining_friend_payment_with_tip : ℕ) 
  (num_remaining_friends : ℕ) 
  (num_friends : ℕ)
  (h1 : total_bill = 150) 
  (h2 : silas_payment = total_bill / 2) 
  (h3 : num_remaining_friends = 5)
  (h4 : remaining_friend_payment_with_tip = 18)
  : (remaining_friend_payment_with_tip - (total_bill / 2 / num_remaining_friends) * num_remaining_friends) / total_bill * 100 = 10 :=
by
  sorry

end tip_percentage_l1507_150787


namespace proof_problem_l1507_150743

def operation1 (x : ℝ) := 9 - x
def operation2 (x : ℝ) := x - 9

theorem proof_problem : operation2 (operation1 15) = -15 := 
by
  sorry

end proof_problem_l1507_150743


namespace suzanna_distance_ridden_l1507_150763

theorem suzanna_distance_ridden (rate_per_5minutes : ℝ) (time_minutes : ℕ) (total_distance : ℝ) (units_per_interval : ℕ) (interval_distance : ℝ) :
  rate_per_5minutes = 0.75 → time_minutes = 45 → units_per_interval = 5 → interval_distance = 0.75 → total_distance = (time_minutes / units_per_interval) * interval_distance → total_distance = 6.75 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end suzanna_distance_ridden_l1507_150763


namespace age_difference_l1507_150752

-- defining the conditions
variable (A B : ℕ)
variable (h1 : B = 35)
variable (h2 : A + 10 = 2 * (B - 10))

-- the proof statement
theorem age_difference : A - B = 5 :=
by
  sorry

end age_difference_l1507_150752


namespace consecutive_numbers_difference_l1507_150766

theorem consecutive_numbers_difference :
  ∃ (n : ℕ), (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 105) → (n + 5 - n = 5) :=
by {
  sorry
}

end consecutive_numbers_difference_l1507_150766


namespace cubic_roots_l1507_150776

variable (p q : ℝ)

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem cubic_roots (y z : ℂ) (h1 : -3 * y * z = p) (h2 : y^3 + z^3 = q) :
  ∃ (x1 x2 x3 : ℂ),
    (x^3 + p * x + q = 0) ∧
    (x1 = -(y + z)) ∧
    (x2 = -(ω * y + ω^2 * z)) ∧
    (x3 = -(ω^2 * y + ω * z)) :=
by
  sorry

end cubic_roots_l1507_150776


namespace chocolate_bars_in_large_box_l1507_150710

theorem chocolate_bars_in_large_box
  (number_of_small_boxes : ℕ)
  (chocolate_bars_per_box : ℕ)
  (h1 : number_of_small_boxes = 21)
  (h2 : chocolate_bars_per_box = 25) :
  number_of_small_boxes * chocolate_bars_per_box = 525 :=
by {
  sorry
}

end chocolate_bars_in_large_box_l1507_150710


namespace find_m_l1507_150756

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem find_m (y b : ℝ) (m : ℕ) 
  (h5 : binomial m 4 * y^(m-4) * b^4 = 210) 
  (h6 : binomial m 5 * y^(m-5) * b^5 = 462) 
  (h7 : binomial m 6 * y^(m-6) * b^6 = 792) : 
  m = 7 := 
sorry

end find_m_l1507_150756


namespace inequality_abc_l1507_150730

variable {a b c : ℝ}

theorem inequality_abc
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := 
by
  sorry

end inequality_abc_l1507_150730


namespace ab_cd_not_prime_l1507_150753

theorem ab_cd_not_prime (a b c d : ℕ) (ha : a > b) (hb : b > c) (hc : c > d) (hd : d > 0)
  (h : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : ¬ Nat.Prime (a * b + c * d) := 
sorry

end ab_cd_not_prime_l1507_150753


namespace additional_money_spent_on_dvds_correct_l1507_150706

def initial_money : ℕ := 320
def spent_on_books : ℕ := initial_money / 4 + 10
def remaining_after_books : ℕ := initial_money - spent_on_books
def spent_on_dvds_portion : ℕ := 2 * remaining_after_books / 5
def remaining_after_dvds : ℕ := 130
def total_spent_on_dvds : ℕ := remaining_after_books - remaining_after_dvds
def additional_spent_on_dvds : ℕ := total_spent_on_dvds - spent_on_dvds_portion

theorem additional_money_spent_on_dvds_correct : additional_spent_on_dvds = 8 :=
by
  sorry

end additional_money_spent_on_dvds_correct_l1507_150706


namespace train_speed_160m_6sec_l1507_150713

noncomputable def train_speed (distance time : ℕ) : ℚ :=
(distance : ℚ) / (time : ℚ)

theorem train_speed_160m_6sec : train_speed 160 6 = 26.67 :=
by
  simp [train_speed]
  norm_num
  sorry

end train_speed_160m_6sec_l1507_150713


namespace sum_of_two_equal_sides_is_4_l1507_150750

noncomputable def isosceles_right_triangle (a c : ℝ) : Prop :=
  c = 2.8284271247461903 ∧ c ^ 2 = 2 * (a ^ 2)

theorem sum_of_two_equal_sides_is_4 :
  ∃ a : ℝ, isosceles_right_triangle a 2.8284271247461903 ∧ 2 * a = 4 :=
by
  sorry

end sum_of_two_equal_sides_is_4_l1507_150750


namespace solution_set_inequalities_l1507_150789

theorem solution_set_inequalities (x : ℝ) :
  (2 * x + 3 ≥ -1) ∧ (7 - 3 * x > 1) ↔ (-2 ≤ x ∧ x < 2) :=
by
  sorry

end solution_set_inequalities_l1507_150789


namespace sum_of_triangles_l1507_150781

def triangle (a b c : ℕ) : ℕ := a * b + c

theorem sum_of_triangles :
  triangle 3 2 5 + triangle 4 1 7 = 22 :=
by
  sorry

end sum_of_triangles_l1507_150781


namespace track_circumference_l1507_150794

def same_start_point (A B : ℕ) : Prop := A = B

def opposite_direction (a_speed b_speed : ℕ) : Prop := a_speed > 0 ∧ b_speed > 0

def first_meet_after (A B : ℕ) (a_distance b_distance : ℕ) : Prop := a_distance = 150 ∧ b_distance = 150

def second_meet_near_full_lap (B : ℕ) (lap_length short_distance : ℕ) : Prop := short_distance = 90

theorem track_circumference
    (A B : ℕ) (a_speed b_speed lap_length : ℕ)
    (h1 : same_start_point A B)
    (h2 : opposite_direction a_speed b_speed)
    (h3 : first_meet_after A B 150 150)
    (h4 : second_meet_near_full_lap B lap_length 90) :
    lap_length = 300 :=
sorry

end track_circumference_l1507_150794


namespace regular_tiles_area_l1507_150723

theorem regular_tiles_area (L W : ℝ) (T : ℝ) (h₁ : 1/3 * T * (3 * L * W) + 2/3 * T * (L * W) = 385) : 
  (2/3 * T * (L * W) = 154) :=
by
  sorry

end regular_tiles_area_l1507_150723


namespace percentage_cut_in_magazine_budget_l1507_150715

noncomputable def magazine_budget_cut (original_budget : ℕ) (cut_amount : ℕ) : ℕ :=
  (cut_amount * 100) / original_budget

theorem percentage_cut_in_magazine_budget : 
  magazine_budget_cut 940 282 = 30 :=
by
  sorry

end percentage_cut_in_magazine_budget_l1507_150715


namespace minimum_points_to_determine_polynomial_l1507_150791

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

def different_at (f g : ℝ → ℝ) (x : ℝ) : Prop :=
  f x ≠ g x

theorem minimum_points_to_determine_polynomial :
  ∀ (f g : ℝ → ℝ), is_quadratic f → is_quadratic g → 
  (∀ t, t < 8 → (different_at f g t → ∃ t₁ t₂ t₃, different_at f g t₁ ∧ different_at f g t₂ ∧ different_at f g t₃)) → False :=
by {
  sorry
}

end minimum_points_to_determine_polynomial_l1507_150791


namespace total_contribution_is_1040_l1507_150707

-- Definitions of contributions based on conditions.
def Niraj_contribution : ℕ := 80
def Brittany_contribution : ℕ := 3 * Niraj_contribution
def Angela_contribution : ℕ := 3 * Brittany_contribution

-- Statement to prove that total contribution is $1040.
theorem total_contribution_is_1040 : Niraj_contribution + Brittany_contribution + Angela_contribution = 1040 := by
  sorry

end total_contribution_is_1040_l1507_150707


namespace find_percentage_l1507_150782

theorem find_percentage (P : ℝ) (h: (20 / 100) * 580 = (P / 100) * 120 + 80) : P = 30 := 
by
  sorry

end find_percentage_l1507_150782


namespace current_speed_l1507_150765

theorem current_speed (c : ℝ) :
  (∀ d1 t1 u v, d1 = 20 ∧ t1 = 2 ∧ u = 6 ∧ v = c → d1 = t1 * (u + v))
  ∧ (∀ d2 t2 u w, d2 = 4 ∧ t2 = 2 ∧ u = 6 ∧ w = c → d2 = t2 * (u - w)) 
  → c = 4 :=
by 
  intros
  sorry

end current_speed_l1507_150765


namespace abs_fraction_inequality_l1507_150755

theorem abs_fraction_inequality (x : ℝ) :
  (abs ((3 * x - 4) / (x - 2)) > 3) ↔
  (x ∈ Set.Iio (5 / 3) ∪ Set.Ioo (5 / 3) 2 ∪ Set.Ioi 2) :=
by 
  sorry

end abs_fraction_inequality_l1507_150755


namespace probability_angie_carlos_two_seats_apart_l1507_150740

theorem probability_angie_carlos_two_seats_apart :
  let people := ["Angie", "Bridget", "Carlos", "Diego", "Edwin"]
  let table_size := people.length
  let total_arrangements := (Nat.factorial (table_size - 1))
  let favorable_arrangements := 2 * (Nat.factorial (table_size - 2))
  total_arrangements > 0 ∧
  (favorable_arrangements / total_arrangements : ℚ) = 1 / 2 :=
by {
  sorry
}

end probability_angie_carlos_two_seats_apart_l1507_150740


namespace f_5times_8_eq_l1507_150718

def f (x : ℚ) : ℚ := 1 / x ^ 2

theorem f_5times_8_eq :
  f (f (f (f (f (8 : ℚ))))) = 1 / 79228162514264337593543950336 := 
  by
    sorry

end f_5times_8_eq_l1507_150718


namespace count_multiples_of_4_l1507_150732

/-- 
Prove that the number of multiples of 4 between 100 and 300 inclusive is 49.
-/
theorem count_multiples_of_4 : 
  ∃ n : ℕ, (∀ k : ℕ, 100 ≤ 4 * k ∧ 4 * k ≤ 300 ↔ k = 26 + n) ∧ n = 48 :=
by
  sorry

end count_multiples_of_4_l1507_150732


namespace least_value_expr_l1507_150701

   variable {x y : ℝ}

   theorem least_value_expr : ∃ x y : ℝ, (x^3 * y - 1)^2 + (x + y)^2 = 1 :=
   by
     sorry
   
end least_value_expr_l1507_150701


namespace symmetric_point_l1507_150773

-- Define the given point M
def point_M : ℝ × ℝ × ℝ := (1, 0, -1)

-- Define the line in parametric form
def line (t : ℝ) : ℝ × ℝ × ℝ :=
  (3.5 + 2 * t, 1.5 + 2 * t, 0)

-- Define the symmetric point M'
def point_M' : ℝ × ℝ × ℝ := (2, -1, 1)

-- Statement: Prove that M' is the symmetric point to M with respect to the given line
theorem symmetric_point (M M' : ℝ × ℝ × ℝ) (line : ℝ → ℝ × ℝ × ℝ) :
  M = (1, 0, -1) →
  line (t) = (3.5 + 2 * t, 1.5 + 2 * t, 0) →
  M' = (2, -1, 1) :=
sorry

end symmetric_point_l1507_150773


namespace annieka_free_throws_l1507_150774

theorem annieka_free_throws (deshawn_throws : ℕ) (kayla_factor : ℝ) (annieka_diff : ℕ) (ht1 : deshawn_throws = 12) (ht2 : kayla_factor = 1.5) (ht3 : annieka_diff = 4) :
  ∃ (annieka_throws : ℕ), annieka_throws = (⌊deshawn_throws * kayla_factor⌋.toNat - annieka_diff) :=
by
  sorry

end annieka_free_throws_l1507_150774


namespace no_such_function_exists_l1507_150762

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (f x) = x^2 - 1996 :=
by
  sorry

end no_such_function_exists_l1507_150762


namespace shortTreesPlanted_l1507_150749

-- Definitions based on conditions
def currentShortTrees : ℕ := 31
def tallTrees : ℕ := 32
def futureShortTrees : ℕ := 95

-- The proposition to be proved
theorem shortTreesPlanted :
  futureShortTrees - currentShortTrees = 64 :=
by
  sorry

end shortTreesPlanted_l1507_150749


namespace maria_carrots_l1507_150725

theorem maria_carrots :
  ∀ (picked initially thrownOut moreCarrots totalLeft : ℕ),
    initially = 48 →
    thrownOut = 11 →
    totalLeft = 52 →
    moreCarrots = totalLeft - (initially - thrownOut) →
    moreCarrots = 15 :=
by
  intros
  sorry

end maria_carrots_l1507_150725


namespace maximize_area_center_coordinates_l1507_150751

theorem maximize_area_center_coordinates (k : ℝ) :
  (∃ r : ℝ, r^2 = 1 - (3/4) * k^2 ∧ r ≥ 0) →
  ((k = 0) → ∃ a b : ℝ, (a = 0 ∧ b = -1)) :=
by
  sorry

end maximize_area_center_coordinates_l1507_150751


namespace inequality_preserves_neg_half_l1507_150727

variable (a b : ℝ)

theorem inequality_preserves_neg_half (h : a ≤ b) : -a / 2 ≥ -b / 2 := by
  sorry

end inequality_preserves_neg_half_l1507_150727


namespace katie_remaining_juice_l1507_150768

-- Define the initial condition: Katie initially has 5 gallons of juice
def initial_gallons : ℚ := 5

-- Define the amount of juice given to Mark
def juice_given : ℚ := 18 / 7

-- Define the expected remaining fraction of juice
def expected_remaining_gallons : ℚ := 17 / 7

-- The theorem statement that Katie should have 17/7 gallons of juice left
theorem katie_remaining_juice : initial_gallons - juice_given = expected_remaining_gallons := 
by
  -- proof would go here
  sorry

end katie_remaining_juice_l1507_150768


namespace newsletter_cost_l1507_150798

theorem newsletter_cost (x : ℝ) (h1 : 14 * x < 16) (h2 : 19 * x > 21) : x = 1.11 :=
by
  sorry

end newsletter_cost_l1507_150798


namespace least_multiple_72_112_199_is_310_l1507_150745

theorem least_multiple_72_112_199_is_310 :
  ∃ k : ℕ, (112 ∣ k * 72) ∧ (199 ∣ k * 72) ∧ k = 310 := 
by
  sorry

end least_multiple_72_112_199_is_310_l1507_150745


namespace max_digits_in_product_l1507_150714

theorem max_digits_in_product :
  let n := (99999 : Nat)
  let m := (999 : Nat)
  let product := n * m
  ∃ d : Nat, product < 10^d ∧ 10^(d-1) ≤ product :=
by
  sorry

end max_digits_in_product_l1507_150714


namespace solve_2019_gon_l1507_150754

noncomputable def problem_2019_gon (x : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, (x i + x (i+1) + x (i+2) + x (i+3) + x (i+4) + x (i+5) + x (i+6) + x (i+7) + x (i+8) = 300))
  ∧ (x 18 = 19)
  ∧ (x 19 = 20)

theorem solve_2019_gon :
  ∀ x : ℕ → ℕ,
  problem_2019_gon x →
  x 2018 = 61 :=
by sorry

end solve_2019_gon_l1507_150754


namespace problem_solution_l1507_150764

theorem problem_solution : (90 + 5) * (12 / (180 / (3^2))) = 57 :=
by
  sorry

end problem_solution_l1507_150764


namespace tunnel_length_l1507_150780

noncomputable def train_length : Real := 2 -- miles
noncomputable def time_to_exit_tunnel : Real := 4 -- minutes
noncomputable def train_speed : Real := 120 -- miles per hour

theorem tunnel_length : ∃ tunnel_length : Real, tunnel_length = 6 :=
  by
  -- We use the conditions given:
  let speed_in_miles_per_minute := train_speed / 60 -- converting speed from miles per hour to miles per minute
  let distance_travelled_by_front_in_4_min := speed_in_miles_per_minute * time_to_exit_tunnel
  let tunnel_length := distance_travelled_by_front_in_4_min - train_length
  have h : tunnel_length = 6 := by sorry
  exact ⟨tunnel_length, h⟩

end tunnel_length_l1507_150780


namespace valid_points_region_equivalence_l1507_150702

def valid_point (x y : ℝ) : Prop :=
  |x - 1| + |x + 1| + |2 * y| ≤ 4

def region1 (x y : ℝ) : Prop :=
  x ≤ -1 ∧ y ≤ x + 2 ∧ y ≥ -x - 2

def region2 (x y : ℝ) : Prop :=
  -1 < x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1

def region3 (x y : ℝ) : Prop :=
  1 < x ∧ y ≤ 2 - x ∧ y ≥ x - 2

def solution_region (x y : ℝ) : Prop :=
  region1 x y ∨ region2 x y ∨ region3 x y

theorem valid_points_region_equivalence : 
  ∀ x y : ℝ, valid_point x y ↔ solution_region x y :=
sorry

end valid_points_region_equivalence_l1507_150702


namespace voldemort_spending_l1507_150771

theorem voldemort_spending :
  let book_price_paid := 8
  let original_book_price := 64
  let journal_price := 2 * book_price_paid
  let total_spent := book_price_paid + journal_price
  (book_price_paid = (original_book_price / 8)) ∧ (total_spent = 24) :=
by
  let book_price_paid := 8
  let original_book_price := 64
  let journal_price := 2 * book_price_paid
  let total_spent := book_price_paid + journal_price
  have h1 : book_price_paid = (original_book_price / 8) := by
    sorry
  have h2 : total_spent = 24 := by
    sorry
  exact ⟨h1, h2⟩

end voldemort_spending_l1507_150771


namespace find_c_l1507_150708

def f (x : ℤ) : ℤ := x - 2

def F (x y : ℤ) : ℤ := y^2 + x

theorem find_c : ∃ c, c = F 3 (f 16) ∧ c = 199 :=
by
  use F 3 (f 16)
  sorry

end find_c_l1507_150708


namespace max_winners_at_least_three_matches_l1507_150709

theorem max_winners_at_least_three_matches (n : ℕ) (h : n = 200) :
  (∃ k : ℕ, k ≤ n ∧ ∀ m : ℕ, ((m ≥ 3) → ∃ x : ℕ, x = k → k = 66)) := 
sorry

end max_winners_at_least_three_matches_l1507_150709


namespace simplify_logarithmic_expression_l1507_150747

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem simplify_logarithmic_expression :
  simplify_expression = 4 / 3 :=
by
  sorry

end simplify_logarithmic_expression_l1507_150747


namespace number_of_classes_l1507_150735

theorem number_of_classes
  (s : ℕ)    -- s: number of students in each class
  (bpm : ℕ) -- bpm: books per month per student
  (months : ℕ) -- months: number of months in a year
  (total_books : ℕ) -- total_books: total books read by the entire student body in a year
  (H1 : bpm = 5)
  (H2 : months = 12)
  (H3 : total_books = 60)
  (H4 : total_books = s * bpm * months)
: s = 1 :=
by
  sorry

end number_of_classes_l1507_150735


namespace cube_volume_given_face_perimeter_l1507_150733

-- Define the perimeter condition
def is_face_perimeter (perimeter : ℝ) (side_length : ℝ) : Prop :=
  4 * side_length = perimeter

-- Define volume computation
def cube_volume (side_length : ℝ) : ℝ :=
  side_length^3

-- Theorem stating the relationship between face perimeter and cube volume
theorem cube_volume_given_face_perimeter : 
  ∀ (side_length perimeter : ℝ), is_face_perimeter 40 side_length → cube_volume side_length = 1000 :=
by
  intros side_length perimeter h
  sorry

end cube_volume_given_face_perimeter_l1507_150733


namespace find_a_and_union_set_l1507_150797

theorem find_a_and_union_set (a : ℝ) 
  (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-3, a + 1}) 
  (hB : B = {2 * a - 1, a ^ 2 + 1}) 
  (h_inter : A ∩ B = {3}) : 
  a = 2 ∧ A ∪ B = {-3, 3, 5} :=
by
  sorry

end find_a_and_union_set_l1507_150797


namespace downstream_speed_l1507_150777

-- Definitions based on the conditions
def V_m : ℝ := 50 -- speed of the man in still water
def V_upstream : ℝ := 45 -- speed of the man when rowing upstream

-- The statement to prove
theorem downstream_speed : ∃ (V_s V_downstream : ℝ), V_upstream = V_m - V_s ∧ V_downstream = V_m + V_s ∧ V_downstream = 55 := 
by
  sorry

end downstream_speed_l1507_150777


namespace new_average_of_subtracted_elements_l1507_150739

theorem new_average_of_subtracted_elements (a b c d e : ℝ) 
  (h_average : (a + b + c + d + e) / 5 = 5) 
  (new_a : ℝ := a - 2) 
  (new_b : ℝ := b - 2) 
  (new_c : ℝ := c - 2) 
  (new_d : ℝ := d - 2) :
  (new_a + new_b + new_c + new_d + e) / 5 = 3.4 := 
by 
  sorry

end new_average_of_subtracted_elements_l1507_150739


namespace sum_of_samples_is_six_l1507_150719

-- Defining the conditions
def grains_varieties : ℕ := 40
def vegetable_oil_varieties : ℕ := 10
def animal_products_varieties : ℕ := 30
def fruits_and_vegetables_varieties : ℕ := 20
def sample_size : ℕ := 20
def total_varieties : ℕ := grains_varieties + vegetable_oil_varieties + animal_products_varieties + fruits_and_vegetables_varieties

def proportion_sample := (sample_size : ℚ) / total_varieties

-- Definitions for the problem
def vegetable_oil_sampled := (vegetable_oil_varieties : ℚ) * proportion_sample
def fruits_and_vegetables_sampled := (fruits_and_vegetables_varieties : ℚ) * proportion_sample

-- Lean 4 statement for the proof problem
theorem sum_of_samples_is_six :
  vegetable_oil_sampled + fruits_and_vegetables_sampled = 6 := by
  sorry

end sum_of_samples_is_six_l1507_150719


namespace selection_of_representatives_l1507_150761

theorem selection_of_representatives 
  (females : ℕ) (males : ℕ)
  (h_females : females = 3) (h_males : males = 4) :
  (females ≥ 1 ∧ males ≥ 1) →
  (females * (males * (males - 1) / 2) + (females * (females - 1) / 2 * males) = 30) := 
by
  sorry

end selection_of_representatives_l1507_150761


namespace number_of_non_congruent_triangles_with_perimeter_20_l1507_150704

theorem number_of_non_congruent_triangles_with_perimeter_20 :
  ∃ T : Finset (Finset ℕ), 
    (∀ t ∈ T, ∃ a b c : ℕ, t = {a, b, c} ∧ a + b + c = 20 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧
    T.card = 14 :=
by
  sorry

end number_of_non_congruent_triangles_with_perimeter_20_l1507_150704


namespace inverse_proportion_inequality_l1507_150711

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_inequality
  (h1 : y1 = 6 / x1)
  (h2 : y2 = 6 / x2)
  (hx : x1 < 0 ∧ 0 < x2) :
  y1 < y2 :=
by
  sorry

end inverse_proportion_inequality_l1507_150711


namespace ratio_Bipin_Alok_l1507_150770

-- Definitions based on conditions
def Alok_age : Nat := 5
def Chandan_age : Nat := 10
def Bipin_age : Nat := 30
def Bipin_age_condition (B C : Nat) : Prop := B + 10 = 2 * (C + 10)

-- Statement to prove
theorem ratio_Bipin_Alok : 
  Bipin_age_condition Bipin_age Chandan_age -> 
  Alok_age = 5 -> 
  Chandan_age = 10 -> 
  Bipin_age / Alok_age = 6 :=
by
  sorry

end ratio_Bipin_Alok_l1507_150770


namespace floor_alpha_six_eq_three_l1507_150742

noncomputable def floor_of_alpha_six (α : ℝ) (h : α^5 - α^3 + α - 2 = 0) : ℤ :=
  Int.floor (α^6)

theorem floor_alpha_six_eq_three (α : ℝ) (h : α^5 - α^3 + α - 2 = 0) : floor_of_alpha_six α h = 3 :=
sorry

end floor_alpha_six_eq_three_l1507_150742


namespace product_mn_l1507_150700

-- Λet θ1 be the angle L1 makes with the positive x-axis.
-- Λet θ2 be the angle L2 makes with the positive x-axis.
-- Given that θ1 = 3 * θ2 and m = 6 * n.
-- Using the tangent triple angle formula: tan(3θ) = (3 * tan(θ) - tan^3(θ)) / (1 - 3 * tan^2(θ))
-- We need to prove mn = 9/17.

noncomputable def mn_product_condition (θ1 θ2 : ℝ) (m n : ℝ) : Prop :=
θ1 = 3 * θ2 ∧ m = 6 * n ∧ m = Real.tan θ1 ∧ n = Real.tan θ2

theorem product_mn (θ1 θ2 : ℝ) (m n : ℝ) (h : mn_product_condition θ1 θ2 m n) :
  m * n = 9 / 17 :=
sorry

end product_mn_l1507_150700


namespace veranda_width_l1507_150726

theorem veranda_width (l w : ℝ) (room_area veranda_area : ℝ) (h1 : l = 20) (h2 : w = 12) (h3 : veranda_area = 144) : 
  ∃ w_v : ℝ, (l + 2 * w_v) * (w + 2 * w_v) - l * w = veranda_area ∧ w_v = 2 := 
by
  sorry

end veranda_width_l1507_150726


namespace dean_ordered_two_pizzas_l1507_150795

variable (P : ℕ)

-- Each large pizza is cut into 12 slices
def slices_per_pizza := 12

-- Dean ate half of the Hawaiian pizza
def dean_slices := slices_per_pizza / 2

-- Frank ate 3 slices of Hawaiian pizza
def frank_slices := 3

-- Sammy ate a third of the cheese pizza
def sammy_slices := slices_per_pizza / 3

-- Total slices eaten plus slices left over equals total slices from pizzas
def total_slices_eaten := dean_slices + frank_slices + sammy_slices
def slices_left_over := 11
def total_pizza_slices := total_slices_eaten + slices_left_over

-- Total pizzas ordered is the total slices divided by slices per pizza
def pizzas_ordered := total_pizza_slices / slices_per_pizza

-- Prove that Dean ordered 2 large pizzas
theorem dean_ordered_two_pizzas : pizzas_ordered = 2 := by
  -- Proof omitted, add your proof here
  sorry

end dean_ordered_two_pizzas_l1507_150795


namespace robin_has_43_packages_of_gum_l1507_150767

theorem robin_has_43_packages_of_gum (P : ℕ) (h1 : 23 * P + 8 = 997) : P = 43 :=
by
  sorry

end robin_has_43_packages_of_gum_l1507_150767


namespace calculate_tough_week_sales_l1507_150769

-- Define the conditions
variables (G T : ℝ)
def condition1 := T = G / 2
def condition2 := 5 * G + 3 * T = 10400

-- By substituting and proving
theorem calculate_tough_week_sales (G T : ℝ) (h1 : condition1 G T) (h2 : condition2 G T) : T = 800 := 
by {
  sorry 
}

end calculate_tough_week_sales_l1507_150769


namespace problem_statement_l1507_150720

def assoc_number (x : ℚ) : ℚ :=
  if x >= 0 then 2 * x - 1 else -2 * x + 1

theorem problem_statement (a b : ℚ) (ha : a > 0) (hb : b < 0) (hab : assoc_number a = assoc_number b) :
  (a + b)^2 - 2 * a - 2 * b = -1 :=
sorry

end problem_statement_l1507_150720


namespace clive_change_l1507_150712

theorem clive_change (money : ℝ) (olives_needed : ℕ) (olives_per_jar : ℕ) (price_per_jar : ℝ) : 
  (money = 10) → 
  (olives_needed = 80) → 
  (olives_per_jar = 20) →
  (price_per_jar = 1.5) →
  money - (olives_needed / olives_per_jar) * price_per_jar = 4 := by
  sorry

end clive_change_l1507_150712


namespace judy_shopping_total_l1507_150737

noncomputable def carrot_price := 1
noncomputable def milk_price := 3
noncomputable def pineapple_price := 4 / 2 -- half price
noncomputable def flour_price := 5
noncomputable def ice_cream_price := 7

noncomputable def carrot_quantity := 5
noncomputable def milk_quantity := 3
noncomputable def pineapple_quantity := 2
noncomputable def flour_quantity := 2
noncomputable def ice_cream_quantity := 1

noncomputable def initial_cost : ℝ := 
  carrot_quantity * carrot_price 
  + milk_quantity * milk_price 
  + pineapple_quantity * pineapple_price 
  + flour_quantity * flour_price 
  + ice_cream_quantity * ice_cream_price

noncomputable def final_cost (initial_cost: ℝ) := if initial_cost ≥ 25 then initial_cost - 5 else initial_cost

theorem judy_shopping_total : final_cost initial_cost = 30 := by
  sorry

end judy_shopping_total_l1507_150737


namespace find_admission_score_l1507_150786

noncomputable def admission_score : ℝ := 87

theorem find_admission_score :
  ∀ (total_students admitted_students not_admitted_students : ℝ) 
    (admission_score admitted_avg not_admitted_avg overall_avg : ℝ),
    admitted_students = total_students / 4 →
    not_admitted_students = 3 * admitted_students →
    admitted_avg = admission_score + 10 →
    not_admitted_avg = admission_score - 26 →
    overall_avg = 70 →
    total_students * overall_avg = 
    (admitted_students * admitted_avg + not_admitted_students * not_admitted_avg) →
    admission_score = 87 :=
by
  intros total_students admitted_students not_admitted_students 
         admission_score admitted_avg not_admitted_avg overall_avg
         h1 h2 h3 h4 h5 h6
  sorry

end find_admission_score_l1507_150786


namespace negative_x_is_positive_l1507_150760

theorem negative_x_is_positive (x : ℝ) (hx : x < 0) : -x > 0 :=
sorry

end negative_x_is_positive_l1507_150760


namespace min_equilateral_triangles_l1507_150734

theorem min_equilateral_triangles (s : ℝ) (S : ℝ) :
  s = 1 → S = 15 → 
  225 = (S / s) ^ 2 :=
by
  intros hs hS
  rw [hs, hS]
  simp
  sorry

end min_equilateral_triangles_l1507_150734


namespace domain_of_f_l1507_150759

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x^2)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ Real.log (x + 1) ≠ 0 ∧ 4 - x^2 ≥ 0} =
  {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_f_l1507_150759


namespace range_of_a_l1507_150717

theorem range_of_a (a : ℝ) (h1 : a ≤ 1)
(h2 : ∃ n₁ n₂ n₃ : ℤ, a ≤ n₁ ∧ n₁ < n₂ ∧ n₂ < n₃ ∧ n₃ ≤ 2 - a
  ∧ (∀ x : ℤ, a ≤ x ∧ x ≤ 2 - a → x = n₁ ∨ x = n₂ ∨ x = n₃)) :
  -1 < a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l1507_150717


namespace max_n_arithmetic_sequences_l1507_150728

theorem max_n_arithmetic_sequences (a b : ℕ → ℤ) 
  (ha : ∀ n, a n = 1 + (n - 1) * 1)  -- Assuming x = 1 for simplicity, as per solution x = y = 1
  (hb : ∀ n, b n = 1 + (n - 1) * 1)  -- Assuming y = 1
  (a1 : a 1 = 1)
  (b1 : b 1 = 1)
  (a2_leq_b2 : a 2 ≤ b 2)
  (hn : ∃ n, a n * b n = 1764) :
  ∃ n, n = 44 ∧ a n * b n = 1764 :=
by
  sorry

end max_n_arithmetic_sequences_l1507_150728


namespace sixty_percent_of_number_l1507_150778

theorem sixty_percent_of_number (N : ℚ) (h : ((1 / 6) * (2 / 3) * (3 / 4) * (5 / 7) * N = 25)) :
  0.60 * N = 252 := sorry

end sixty_percent_of_number_l1507_150778


namespace average_of_remaining_numbers_l1507_150748

theorem average_of_remaining_numbers 
    (nums : List ℝ) 
    (h_length : nums.length = 12) 
    (h_avg_90 : (nums.sum) / 12 = 90) 
    (h_contains_65_85 : 65 ∈ nums ∧ 85 ∈ nums) 
    (nums' := nums.erase 65)
    (nums'' := nums'.erase 85) : 
   nums''.length = 10 ∧ nums''.sum / 10 = 93 :=
by
  sorry

end average_of_remaining_numbers_l1507_150748


namespace bar_charts_as_line_charts_l1507_150799

-- Given that line charts help to visualize trends of increase and decrease
axiom trends_visualization (L : Type) : Prop

-- Bar charts can be drawn as line charts, which helps in visualizing trends
theorem bar_charts_as_line_charts (L B : Type) (h : trends_visualization L) : trends_visualization B := sorry

end bar_charts_as_line_charts_l1507_150799


namespace pearJuicePercentageCorrect_l1507_150705

-- Define the conditions
def dozen : ℕ := 12
def pears := dozen
def oranges := dozen
def pearJuiceFrom3Pears : ℚ := 8
def orangeJuiceFrom2Oranges : ℚ := 10
def juiceBlendPears : ℕ := 4
def juiceBlendOranges : ℕ := 4
def pearJuicePerPear : ℚ := pearJuiceFrom3Pears / 3
def orangeJuicePerOrange : ℚ := orangeJuiceFrom2Oranges / 2
def totalPearJuice : ℚ := juiceBlendPears * pearJuicePerPear
def totalOrangeJuice : ℚ := juiceBlendOranges * orangeJuicePerOrange
def totalJuice : ℚ := totalPearJuice + totalOrangeJuice

-- Prove that the percentage of pear juice in the blend is 34.78%
theorem pearJuicePercentageCorrect : 
  (totalPearJuice / totalJuice) * 100 = 34.78 := by
  sorry

end pearJuicePercentageCorrect_l1507_150705


namespace prob_A_is_15_16_prob_B_is_3_4_prob_C_is_5_9_prob_exactly_two_good_ratings_is_77_576_l1507_150724

-- Define the probability of success for student A, B, and C on a single jump
def p_A1 := 3 / 4
def p_B1 := 1 / 2
def p_C1 := 1 / 3

-- Calculate the total probability of excellence for A, B, and C
def P_A := p_A1 + (1 - p_A1) * p_A1
def P_B := p_B1 + (1 - p_B1) * p_B1
def P_C := p_C1 + (1 - p_C1) * p_C1

-- Statement to prove probabilities
theorem prob_A_is_15_16 : P_A = 15 / 16 := sorry
theorem prob_B_is_3_4 : P_B = 3 / 4 := sorry
theorem prob_C_is_5_9 : P_C = 5 / 9 := sorry

-- Definition for P(Good_Ratings) - exactly two students get a good rating
def P_Good_Ratings := 
  P_A * (1 - P_B) * (1 - P_C) + 
  (1 - P_A) * P_B * (1 - P_C) + 
  (1 - P_A) * (1 - P_B) * P_C

-- Statement to prove the given condition about good ratings
theorem prob_exactly_two_good_ratings_is_77_576 : P_Good_Ratings = 77 / 576 := sorry

end prob_A_is_15_16_prob_B_is_3_4_prob_C_is_5_9_prob_exactly_two_good_ratings_is_77_576_l1507_150724


namespace prime_factorization_count_l1507_150721

theorem prime_factorization_count :
  (∃ (S : Finset ℕ), S = {97, 101, 2, 13, 107, 109} ∧ S.card = 6) :=
by
  sorry

end prime_factorization_count_l1507_150721


namespace percentage_increase_twice_eq_16_64_l1507_150784

theorem percentage_increase_twice_eq_16_64 (x : ℝ) (hx : (1 + x)^2 = 1 + 0.1664) : x = 0.08 :=
by
  sorry -- This is the placeholder for the proof.

end percentage_increase_twice_eq_16_64_l1507_150784


namespace andrei_club_visits_l1507_150788

theorem andrei_club_visits (d c : ℕ) (h : 15 * d + 11 * c = 115) : d + c = 9 :=
by
  sorry

end andrei_club_visits_l1507_150788


namespace calculation_is_correct_l1507_150790

theorem calculation_is_correct : -1^6 + 8 / (-2)^2 - abs (-4 * 3) = -9 := by
  sorry

end calculation_is_correct_l1507_150790
