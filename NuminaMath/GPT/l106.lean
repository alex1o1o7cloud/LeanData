import Mathlib

namespace optimalBananaBuys_l106_10668

noncomputable def bananaPrices : List ℕ := [1, 5, 1, 6, 7, 8, 1, 8, 7, 2, 7, 8, 1, 9, 2, 8, 7, 1]

def days := List.range 18

def computeOptimalBuys : List ℕ :=
  sorry -- Implement the logic to compute the optimal number of bananas to buy each day.

theorem optimalBananaBuys :
  computeOptimalBuys = [4, 0, 0, 3, 0, 0, 7, 0, 0, 1, 0, 0, 4, 0, 0, 3, 0, 1] :=
sorry

end optimalBananaBuys_l106_10668


namespace four_disjoint_subsets_with_equal_sums_l106_10634

theorem four_disjoint_subsets_with_equal_sums :
  ∀ (S : Finset ℕ), 
  (∀ x ∈ S, 100 ≤ x ∧ x ≤ 999) ∧ S.card = 117 → 
  ∃ A B C D : Finset ℕ, 
    (A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧ D ⊆ S) ∧ 
    (A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ A ∩ D = ∅ ∧ B ∩ C = ∅ ∧ B ∩ D = ∅ ∧ C ∩ D = ∅) ∧ 
    (A.sum id = B.sum id ∧ B.sum id = C.sum id ∧ C.sum id = D.sum id) := by
  sorry

end four_disjoint_subsets_with_equal_sums_l106_10634


namespace cards_problem_l106_10639

theorem cards_problem : 
  ∀ (cards people : ℕ),
  cards = 60 →
  people = 8 →
  ∃ fewer_people : ℕ,
  (∀ p: ℕ, p < people → (p < fewer_people → cards/people < 8)) ∧ 
  fewer_people = 4 := 
by 
  intros cards people h_cards h_people
  use 4
  sorry

end cards_problem_l106_10639


namespace find_divisor_l106_10645

theorem find_divisor (remainder quotient dividend divisor : ℕ) 
  (h_rem : remainder = 8)
  (h_quot : quotient = 43)
  (h_div : dividend = 997)
  (h_eq : dividend = divisor * quotient + remainder) : 
  divisor = 23 :=
by
  sorry

end find_divisor_l106_10645


namespace wade_average_points_per_game_l106_10658

variable (W : ℝ)

def teammates_average_points_per_game : ℝ := 40

def total_team_points_after_5_games : ℝ := 300

theorem wade_average_points_per_game :
  teammates_average_points_per_game * 5 + W * 5 = total_team_points_after_5_games →
  W = 20 :=
by
  intro h
  sorry

end wade_average_points_per_game_l106_10658


namespace trigonometric_identity_l106_10607

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin θ * Real.sin (π / 2 - θ)) / (Real.sin θ ^ 2 + Real.cos (2 * θ) + Real.cos θ ^ 2) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l106_10607


namespace twenty_cows_twenty_days_l106_10623

-- Defining the initial conditions as constants
def num_cows : ℕ := 20
def days_one_cow_eats_one_bag : ℕ := 20
def bags_eaten_by_one_cow_in_days (d : ℕ) : ℕ := if d = days_one_cow_eats_one_bag then 1 else 0

-- Defining the total bags eaten by all cows
def total_bags_eaten_by_cows (cows : ℕ) (days : ℕ) : ℕ :=
  cows * (days / days_one_cow_eats_one_bag)

-- Statement to be proved: In 20 days, 20 cows will eat 20 bags of husk
theorem twenty_cows_twenty_days :
  total_bags_eaten_by_cows num_cows days_one_cow_eats_one_bag = 20 := sorry

end twenty_cows_twenty_days_l106_10623


namespace sum_of_coordinates_of_C_parallelogram_l106_10618

-- Definitions that encapsulate the given conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨-1, 0⟩
def D : Point := ⟨5, -4⟩

-- The theorem we need to prove
theorem sum_of_coordinates_of_C_parallelogram :
  ∃ C : Point, C.x + C.y = 7 ∧
  ∃ M : Point, M = ⟨(A.x + D.x) / 2, (A.y + D.y) / 2⟩ ∧
  (M = ⟨(B.x + C.x) / 2, (B.y + C.y) / 2⟩) :=
sorry

end sum_of_coordinates_of_C_parallelogram_l106_10618


namespace divisible_by_six_l106_10644

theorem divisible_by_six (n a b : ℕ) (h1 : 2^n = 10 * a + b) (h2 : n > 3) (h3 : b > 0) (h4 : b < 10) : 6 ∣ (a * b) := 
sorry

end divisible_by_six_l106_10644


namespace complex_modulus_inequality_l106_10678

theorem complex_modulus_inequality (z : ℂ) : (‖z‖ ^ 2 + 2 * ‖z - 1‖) ≥ 1 :=
by
  sorry

end complex_modulus_inequality_l106_10678


namespace Carlos_candy_share_l106_10636

theorem Carlos_candy_share (total_candy : ℚ) (num_piles : ℕ) (piles_for_Carlos : ℕ)
  (h_total_candy : total_candy = 75 / 7)
  (h_num_piles : num_piles = 5)
  (h_piles_for_Carlos : piles_for_Carlos = 2) :
  (piles_for_Carlos * (total_candy / num_piles) = 30 / 7) :=
by
  sorry

end Carlos_candy_share_l106_10636


namespace exists_natural_sum_of_squares_l106_10652

theorem exists_natural_sum_of_squares : ∃ n : ℕ, n^2 = 0^2 + 7^2 + 24^2 + 312^2 + 48984^2 :=
by {
  sorry
}

end exists_natural_sum_of_squares_l106_10652


namespace ratio_to_percent_l106_10616

theorem ratio_to_percent :
  (9 / 5 * 100) = 180 :=
by
  sorry

end ratio_to_percent_l106_10616


namespace triangle_cosine_l106_10667

theorem triangle_cosine {A : ℝ} (h : 0 < A ∧ A < π / 2) (tan_A : Real.tan A = -2) :
  Real.cos A = - (Real.sqrt 5) / 5 :=
sorry

end triangle_cosine_l106_10667


namespace polynomial_coeff_sum_l106_10641

theorem polynomial_coeff_sum {a_0 a_1 a_2 a_3 a_4 a_5 : ℝ} :
  (2 * (x : ℝ) - 3)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  intro h
  sorry

end polynomial_coeff_sum_l106_10641


namespace area_of_inscribed_square_l106_10611

theorem area_of_inscribed_square
    (r : ℝ)
    (h : ∀ A : ℝ × ℝ, (A.1 = r - 1 ∨ A.1 = -(r - 1)) ∧ (A.2 = r - 2 ∨ A.2 = -(r - 2)) → A.1^2 + A.2^2 = r^2) :
    4 * r^2 = 100 := by
  -- proof would go here
  sorry

end area_of_inscribed_square_l106_10611


namespace calculate_T1_T2_l106_10648

def triangle (a b c : ℤ) : ℤ := a + b - 2 * c

def T1 := triangle 3 4 5
def T2 := triangle 6 8 2

theorem calculate_T1_T2 : 2 * T1 + 3 * T2 = 24 :=
  by
    sorry

end calculate_T1_T2_l106_10648


namespace stop_signs_per_mile_l106_10689

-- Define the conditions
def miles_traveled := 5 + 2
def stop_signs_encountered := 17 - 3

-- Define the proof statement
theorem stop_signs_per_mile : (stop_signs_encountered / miles_traveled) = 2 := by
  -- Proof goes here
  sorry

end stop_signs_per_mile_l106_10689


namespace expression_parity_l106_10632

theorem expression_parity (p m : ℤ) (hp : Odd p) : (Odd (p^3 + m * p)) ↔ Even m := by
  sorry

end expression_parity_l106_10632


namespace first_discount_percentage_l106_10665

theorem first_discount_percentage (x : ℝ) (h : 450 * (1 - x / 100) * 0.85 = 306) : x = 20 :=
sorry

end first_discount_percentage_l106_10665


namespace tan_negative_angle_l106_10620

theorem tan_negative_angle (m : ℝ) (h1 : m = Real.cos (80 * Real.pi / 180)) (h2 : m = Real.sin (10 * Real.pi / 180)) :
  Real.tan (-440 * Real.pi / 180) = - (Real.sqrt (1 - m^2)) / m :=
by
  sorry

end tan_negative_angle_l106_10620


namespace possible_values_l106_10625

theorem possible_values (a : ℝ) (h : a > 1) : ∃ (v : ℝ), (v = 5 ∨ v = 6 ∨ v = 7) ∧ (a + 4 / (a - 1) = v) :=
sorry

end possible_values_l106_10625


namespace solve_wire_cut_problem_l106_10676

def wire_cut_problem : Prop :=
  ∃ x y : ℝ, x + y = 35 ∧ y = (2/5) * x ∧ x = 25

theorem solve_wire_cut_problem : wire_cut_problem := by
  sorry

end solve_wire_cut_problem_l106_10676


namespace numberOfBags_l106_10604

-- Define the given conditions
def totalCookies : Nat := 33
def cookiesPerBag : Nat := 11

-- Define the statement to prove
theorem numberOfBags : totalCookies / cookiesPerBag = 3 := by
  sorry

end numberOfBags_l106_10604


namespace older_brother_stamps_l106_10670

variable (y o : ℕ)

def condition1 : Prop := o = 2 * y + 1
def condition2 : Prop := o + y = 25

theorem older_brother_stamps (h1 : condition1 y o) (h2 : condition2 y o) : o = 17 :=
by
  sorry

end older_brother_stamps_l106_10670


namespace age_of_15th_student_l106_10656

theorem age_of_15th_student (avg_age_15 : ℕ) (avg_age_6 : ℕ) (avg_age_8 : ℕ) (num_students_15 : ℕ) (num_students_6 : ℕ) (num_students_8 : ℕ) 
  (h_avg_15 : avg_age_15 = 15) 
  (h_avg_6 : avg_age_6 = 14) 
  (h_avg_8 : avg_age_8 = 16) 
  (h_num_15 : num_students_15 = 15) 
  (h_num_6 : num_students_6 = 6) 
  (h_num_8 : num_students_8 = 8) : 
  ∃ age_15th_student : ℕ, age_15th_student = 13 := 
by
  sorry


end age_of_15th_student_l106_10656


namespace curve_touches_x_axis_at_most_three_times_l106_10610

theorem curve_touches_x_axis_at_most_three_times
  (a b c d : ℝ) :
  ∃ (x : ℝ), (x^4 - x^5 + a * x^3 + b * x^2 + c * x + d = 0) → ∃ (y : ℝ), (y = 0) → 
  ∃(n : ℕ), (n ≤ 3) :=
by sorry

end curve_touches_x_axis_at_most_three_times_l106_10610


namespace product_ineq_l106_10650

-- Define the relevant elements and conditions
variables (a b : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ)

-- Assumptions based on the conditions provided
variables (h₀ : a > 0) (h₁ : b > 0)
variables (h₂ : a + b = 1)
variables (h₃ : x₁ > 0) (h₄ : x₂ > 0) (h₅ : x₃ > 0) (h₆ : x₄ > 0) (h₇ : x₅ > 0)
variables (h₈ : x₁ * x₂ * x₃ * x₄ * x₅ = 1)

-- The theorem statement to be proved
theorem product_ineq : (a * x₁ + b) * (a * x₂ + b) * (a * x₃ + b) * (a * x₄ + b) * (a * x₅ + b) ≥ 1 :=
sorry

end product_ineq_l106_10650


namespace mike_scored_212_l106_10646

variable {M : ℕ}

def passing_marks (max_marks : ℕ) : ℕ := (30 * max_marks) / 100

def mike_marks (passing_marks shortfall : ℕ) : ℕ := passing_marks - shortfall

theorem mike_scored_212 (max_marks : ℕ) (shortfall : ℕ)
  (h1 : max_marks = 790)
  (h2 : shortfall = 25)
  (h3 : M = mike_marks (passing_marks max_marks) shortfall) : 
  M = 212 := 
by 
  sorry

end mike_scored_212_l106_10646


namespace hyperbola_real_axis_length_l106_10631

theorem hyperbola_real_axis_length :
  (∃ (a b : ℝ), (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) ∧ a = 3) →
  2 * 3 = 6 :=
by
  sorry

end hyperbola_real_axis_length_l106_10631


namespace sum_of_angles_l106_10685

theorem sum_of_angles (α β : ℝ) (hα: 0 < α ∧ α < π) (hβ: 0 < β ∧ β < π) (h_tan_α: Real.tan α = 1 / 2) (h_tan_β: Real.tan β = 1 / 3) : α + β = π / 4 := 
by 
  sorry

end sum_of_angles_l106_10685


namespace compound_interest_second_year_l106_10649

theorem compound_interest_second_year
  (P : ℝ) (r : ℝ) (CI_3 : ℝ) (CI_2 : ℝ) 
  (h1 : r = 0.08) 
  (h2 : CI_3 = 1512)
  (h3 : CI_3 = CI_2 * (1 + r)) :
  CI_2 = 1400 :=
by
  rw [h1, h2] at h3
  sorry

end compound_interest_second_year_l106_10649


namespace polygon_sides_l106_10608

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 :=
sorry

end polygon_sides_l106_10608


namespace sum_of_fractions_irreducible_l106_10683

noncomputable def is_irreducible (num denom : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ num ∧ d ∣ denom → d = 1

theorem sum_of_fractions_irreducible (a b : ℕ) (h_coprime : Nat.gcd a b = 1) :
  is_irreducible (2 * a + b) (a * (a + b)) :=
by
  sorry

end sum_of_fractions_irreducible_l106_10683


namespace proof_problem_l106_10612

def diamond (a b : ℚ) := a - (1 / b)

theorem proof_problem :
  ((diamond (diamond 2 4) 5) - (diamond 2 (diamond 4 5))) = (-71 / 380) := by
  sorry

end proof_problem_l106_10612


namespace original_price_l106_10663

variable (q r : ℝ)

theorem original_price (x : ℝ) (h : x * (1 + q / 100) * (1 - r / 100) = 1) :
  x = 1 / ((1 + q / 100) * (1 - r / 100)) :=
sorry

end original_price_l106_10663


namespace twenty_second_entry_l106_10629

-- Definition of r_9 which is the remainder left when n is divided by 9
def r_9 (n : ℕ) : ℕ := n % 9

-- Statement to prove that the 22nd entry in the ordered list of all nonnegative integers
-- that satisfy r_9(5n) ≤ 4 is 38
theorem twenty_second_entry (n : ℕ) (hn : 5 * n % 9 ≤ 4) :
  ∃ m : ℕ, m = 22 ∧ n = 38 :=
sorry

end twenty_second_entry_l106_10629


namespace andre_total_payment_l106_10655

def treadmill_initial_price : ℝ := 1350
def treadmill_discount : ℝ := 0.30
def plate_initial_price : ℝ := 60
def plate_discount : ℝ := 0.15
def plate_quantity : ℝ := 2

theorem andre_total_payment :
  let treadmill_discounted_price := treadmill_initial_price * (1 - treadmill_discount)
  let plates_total_initial_price := plate_quantity * plate_initial_price
  let plates_discounted_price := plates_total_initial_price * (1 - plate_discount)
  treadmill_discounted_price + plates_discounted_price = 1047 := 
by
  sorry

end andre_total_payment_l106_10655


namespace compute_g_five_times_l106_10635

def g (x : ℤ) : ℤ :=
  if x ≥ 0 then -x^3 else x + 6

theorem compute_g_five_times :
  g (g (g (g (g 1)))) = -113 :=
  by sorry

end compute_g_five_times_l106_10635


namespace ab_bc_ca_max_le_l106_10614

theorem ab_bc_ca_max_le (a b c : ℝ) :
  ab + bc + ca + max (abs (a - b)) (max (abs (b - c)) (abs (c - a))) ≤
  1 + (1 / 3) * (a + b + c)^2 :=
sorry

end ab_bc_ca_max_le_l106_10614


namespace inequality_holds_l106_10677

theorem inequality_holds (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (y * z)) + (y^3 / (z * x)) + (z^3 / (x * y)) ≥ x + y + z :=
by
  sorry

end inequality_holds_l106_10677


namespace rate_of_interest_l106_10619

/-
Let P be the principal amount, SI be the simple interest paid, R be the rate of interest, and N be the number of years. 
The problem states:
- P = 1200
- SI = 432
- R = N

We need to prove that R = 6.
-/

theorem rate_of_interest (P SI R N : ℝ) (h1 : P = 1200) (h2 : SI = 432) (h3 : R = N) :
  R = 6 :=
  sorry

end rate_of_interest_l106_10619


namespace prime_between_30_and_40_has_remainder_7_l106_10662

theorem prime_between_30_and_40_has_remainder_7 (p : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_interval : 30 < p ∧ p < 40) 
  (h_mod : p % 9 = 7) : 
  p = 34 := 
sorry

end prime_between_30_and_40_has_remainder_7_l106_10662


namespace positive_integer_count_l106_10651

/-
  Prove that the number of positive integers \( n \) for which \( \frac{n(n+1)}{2} \) divides \( 30n \) is 11.
-/

theorem positive_integer_count (n : ℕ) :
  (∃ k : ℕ, k > 0 ∧ k ≤ 11 ∧ (2 * 30 * n) % (n * (n + 1)) = 0) :=
sorry

end positive_integer_count_l106_10651


namespace max_n_l106_10640

noncomputable def seq_a (n : ℕ) : ℤ := 3 * n - 1

noncomputable def seq_b (n : ℕ) : ℤ := 2 * n - 3

noncomputable def sum_T (n : ℕ) : ℤ := n * (3 * n + 1) / 2

noncomputable def sum_S (n : ℕ) : ℤ := n^2 - 2 * n

theorem max_n (n : ℕ) :
  ∃ n_max : ℕ, T_n < 20 * seq_b n ∧ (∀ m : ℕ, m > n_max → T_n ≥ 20 * seq_b n) :=
  sorry

end max_n_l106_10640


namespace largest_unattainable_sum_l106_10664

noncomputable def largestUnattainableSum (n : ℕ) : ℕ :=
  12 * n^2 + 8 * n - 1

theorem largest_unattainable_sum (n : ℕ) :
  ∀ s, (¬∃ a b c d, s = (a * (6 * n + 1) + b * (6 * n + 3) + c * (6 * n + 5) + d * (6 * n + 7)))
  ↔ s > largestUnattainableSum n := by
  sorry

end largest_unattainable_sum_l106_10664


namespace sixteen_a_four_plus_one_div_a_four_l106_10693

theorem sixteen_a_four_plus_one_div_a_four (a : ℝ) (h : 2 * a - 1 / a = 3) :
  16 * a^4 + (1 / a^4) = 161 :=
sorry

end sixteen_a_four_plus_one_div_a_four_l106_10693


namespace borrowed_nickels_l106_10671

-- Define the initial and remaining number of nickels
def initial_nickels : ℕ := 87
def remaining_nickels : ℕ := 12

-- Prove that the number of nickels borrowed is 75
theorem borrowed_nickels : initial_nickels - remaining_nickels = 75 := by
  sorry

end borrowed_nickels_l106_10671


namespace merchant_gross_profit_l106_10696

noncomputable def purchase_price : ℝ := 48
noncomputable def markup_rate : ℝ := 0.40
noncomputable def discount_rate : ℝ := 0.20

theorem merchant_gross_profit :
  ∃ S : ℝ, S = purchase_price + markup_rate * S ∧ 
  ((S - discount_rate * S) - purchase_price = 16) :=
by
  sorry

end merchant_gross_profit_l106_10696


namespace problem_1_problem_2_l106_10601

-- Definitions of conditions
variables {a b : ℝ}
axiom h_pos_a : a > 0
axiom h_pos_b : b > 0
axiom h_sum : a + b = 1

-- The statements to prove
theorem problem_1 : 
  (1 / (a^2)) + (1 / (b^2)) ≥ 8 := 
sorry

theorem problem_2 : 
  (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 := 
sorry

end problem_1_problem_2_l106_10601


namespace number_of_sets_l106_10627

theorem number_of_sets (weight_per_rep reps total_weight : ℕ) 
  (h_weight_per_rep : weight_per_rep = 15)
  (h_reps : reps = 10)
  (h_total_weight : total_weight = 450) :
  (total_weight / (weight_per_rep * reps)) = 3 :=
by
  sorry

end number_of_sets_l106_10627


namespace days_from_friday_l106_10647

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l106_10647


namespace point_on_line_l106_10609

theorem point_on_line :
  ∃ a b : ℝ, (a ≠ 0) ∧
  (∀ x y : ℝ, (x = 4 ∧ y = 5) ∨ (x = 8 ∧ y = 17) ∨ (x = 12 ∧ y = 29) → y = a * x + b) →
  (∃ t : ℝ, (15, t) ∈ {(x, y) | y = a * x + b} ∧ t = 38) :=
by
  sorry

end point_on_line_l106_10609


namespace common_chord_length_l106_10698

theorem common_chord_length (r : ℝ) (h : r = 12) 
  (condition : ∀ (C₁ C₂ : Set (ℝ × ℝ)), 
      ((C₁ = {p : ℝ × ℝ | dist p (0, 0) = r}) ∧ 
       (C₂ = {p : ℝ × ℝ | dist p (12, 0) = r}) ∧
       (C₂ ∩ C₁ ≠ ∅))) : 
  ∃ chord_len : ℝ, chord_len = 12 * Real.sqrt 3 :=
by
  sorry

end common_chord_length_l106_10698


namespace kevin_ends_with_cards_l106_10666

def cards_found : ℝ := 47.0
def cards_lost : ℝ := 7.0

theorem kevin_ends_with_cards : cards_found - cards_lost = 40.0 := by
  sorry

end kevin_ends_with_cards_l106_10666


namespace expansion_coefficients_sum_l106_10690

theorem expansion_coefficients_sum : 
  ∀ (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ), 
    (x - 2)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 → 
    a_0 + a_2 + a_4 = -122 := 
by 
  intros x a_0 a_1 a_2 a_3 a_4 a_5 h_eq
  sorry

end expansion_coefficients_sum_l106_10690


namespace product_zero_probability_l106_10691

noncomputable def probability_product_is_zero : ℚ :=
  let S := [-3, -1, 0, 0, 2, 5]
  let total_ways := 15 -- Calculated as 6 choose 2 taking into account repetition
  let favorable_ways := 8 -- Calculated as (2 choose 1) * (4 choose 1)
  favorable_ways / total_ways

theorem product_zero_probability : probability_product_is_zero = 8 / 15 := by
  sorry

end product_zero_probability_l106_10691


namespace sum_of_squares_l106_10659

def satisfies_conditions (x y z : ℕ) : Prop :=
  x + y + z = 24 ∧
  Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10

theorem sum_of_squares (x y z : ℕ) (h : satisfies_conditions x y z) :
  ∀ (x y z : ℕ), x + y + z = 24 ∧ Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10 →
  x^2 + y^2 + z^2 = 216 :=
sorry

end sum_of_squares_l106_10659


namespace simplify_fraction_l106_10679

theorem simplify_fraction (x y z : ℝ) (h : x + 2 * y + z ≠ 0) :
  (x^2 + y^2 - 4 * z^2 + 2 * x * y) / (x^2 + 4 * y^2 - z^2 + 2 * x * z) = (x + y - 2 * z) / (x + z - 2 * y) :=
by
  sorry

end simplify_fraction_l106_10679


namespace luncheon_cost_l106_10687

theorem luncheon_cost (s c p : ℝ)
  (h1 : 2 * s + 5 * c + 2 * p = 3.50)
  (h2 : 3 * s + 7 * c + 2 * p = 4.90) :
  s + c + p = 1.00 :=
  sorry

end luncheon_cost_l106_10687


namespace min_ab_l106_10654

theorem min_ab {a b : ℝ} (h1 : (a^2) * (-b) + (a^2 + 1) = 0) : |a * b| = 2 :=
sorry

end min_ab_l106_10654


namespace trigonometric_identity_l106_10692

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -3) : 
  (Real.cos α - Real.sin α) / (Real.cos α + Real.sin α) = -2 :=
by 
  sorry

end trigonometric_identity_l106_10692


namespace find_numbers_l106_10624

theorem find_numbers (a b : ℕ) 
  (h1 : a / b * 6 = 10)
  (h2 : a - b + 4 = 10) :
  a = 15 ∧ b = 9 := by
  sorry

end find_numbers_l106_10624


namespace num_four_digit_integers_divisible_by_7_l106_10605

theorem num_four_digit_integers_divisible_by_7 :
  ∃ n : ℕ, n = 1286 ∧ ∀ k : ℕ, (1000 ≤ k ∧ k ≤ 9999) → (k % 7 = 0 ↔ ∃ m : ℕ, k = m * 7) :=
by {
  sorry
}

end num_four_digit_integers_divisible_by_7_l106_10605


namespace sum_of_values_l106_10603

theorem sum_of_values (N : ℝ) (R : ℝ) (h : N ≠ 0) (h_eq : N + 5 / N = R) : N = R := 
sorry

end sum_of_values_l106_10603


namespace correct_operation_l106_10699

variable (a b : ℝ)

theorem correct_operation : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l106_10699


namespace product_mk_through_point_l106_10633

theorem product_mk_through_point (k m : ℝ) (h : (2 : ℝ) ^ m * k = (1/4 : ℝ)) : m * k = -2 := 
sorry

end product_mk_through_point_l106_10633


namespace problem_ABC_sum_l106_10606

-- Let A, B, and C be positive integers such that A and C, B and C, and A and B
-- have no common factor greater than 1.
-- If they satisfy the equation A * log_100 5 + B * log_100 4 = C,
-- then we need to prove that A + B + C = 4.

theorem problem_ABC_sum (A B C : ℕ) (h1 : 1 < A ∧ 1 < B ∧ 1 < C)
    (h2 : A.gcd B = 1 ∧ B.gcd C = 1 ∧ A.gcd C = 1)
    (h3 : A * Real.log 5 / Real.log 100 + B * Real.log 4 / Real.log 100 = C) :
    A + B + C = 4 :=
sorry

end problem_ABC_sum_l106_10606


namespace problem_statement_l106_10680

variable {f : ℝ → ℝ}

-- Condition 1: The function f satisfies (x - 1)f'(x) ≤ 0
def cond1 (f : ℝ → ℝ) : Prop := ∀ x, (x - 1) * (deriv f x) ≤ 0

-- Condition 2: The function f satisfies f(-x) = f(2 + x)
def cond2 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f (2 + x)

theorem problem_statement (f : ℝ → ℝ) (x₁ x₂ : ℝ)
  (h_cond1 : cond1 f)
  (h_cond2 : cond2 f)
  (h_dist : abs (x₁ - 1) < abs (x₂ - 1)) :
  f (2 - x₁) > f (2 - x₂) :=
sorry

end problem_statement_l106_10680


namespace pizza_slices_count_l106_10695

/-
  We ordered 21 pizzas. Each pizza has 8 slices. 
  Prove that the total number of slices of pizza is 168.
-/

theorem pizza_slices_count :
  (21 * 8) = 168 :=
by
  sorry

end pizza_slices_count_l106_10695


namespace total_pies_l106_10674

theorem total_pies {team1 team2 team3 total_pies : ℕ} 
  (h1 : team1 = 235) 
  (h2 : team2 = 275) 
  (h3 : team3 = 240) 
  (h4 : total_pies = team1 + team2 + team3) : 
  total_pies = 750 := by 
  sorry

end total_pies_l106_10674


namespace correct_81st_in_set_s_l106_10643

def is_in_set_s (x : ℕ) : Prop :=
  ∃ n : ℕ, x = 8 * n + 5

noncomputable def find_81st_in_set_s : ℕ :=
  8 * 80 + 5

theorem correct_81st_in_set_s : find_81st_in_set_s = 645 := by
  sorry

end correct_81st_in_set_s_l106_10643


namespace arrow_sequence_correct_l106_10626

variable (A B C D E F G : ℕ)
variable (square : ℕ → ℕ)

-- Definitions based on given conditions
def conditions : Prop :=
  square 1 = 1 ∧ square 9 = 9 ∧
  square A = 6 ∧ square B = 2 ∧ square C = 4 ∧
  square D = 5 ∧ square E = 3 ∧ square F = 8 ∧ square G = 7 ∧
  (∀ x, (x = 1 → square 2 = B) ∧ (x = 2 → square 3 = E) ∧
       (x = 3 → square 4 = C) ∧ (x = 4 → square 5 = D) ∧
       (x = 5 → square 6 = A) ∧ (x = 6 → square 7 = G) ∧
       (x = 7 → square 8 = F) ∧ (x = 8 → square 9 = 9))

theorem arrow_sequence_correct :
  conditions A B C D E F G square → 
  ∀ x, square (x + 1) = 1 + x :=
by sorry

end arrow_sequence_correct_l106_10626


namespace probability_chord_length_not_less_than_radius_l106_10673

theorem probability_chord_length_not_less_than_radius
  (R : ℝ) (M N : ℝ) (h_circle : N = 2 * π * R) : 
  (∃ P : ℝ, P = 2 / 3) :=
sorry

end probability_chord_length_not_less_than_radius_l106_10673


namespace xiao_wang_ways_to_make_8_cents_l106_10615

theorem xiao_wang_ways_to_make_8_cents :
  let one_cent_coins := 8
  let two_cent_coins := 4
  let five_cent_coin := 1
  ∃ ways, ways = 7 ∧ (
       (ways = 8 ∧ one_cent_coins >= 8) ∨
       (ways = 4 ∧ two_cent_coins >= 4) ∨
       (ways = 2 ∧ one_cent_coins >= 2 ∧ two_cent_coins >= 3) ∨
       (ways = 4 ∧ one_cent_coins >= 4 ∧ two_cent_coins >= 2) ∨
       (ways = 6 ∧ one_cent_coins >= 6 ∧ two_cent_coins >= 1) ∨
       (ways = 3 ∧ one_cent_coins >= 3 ∧ five_cent_coin >= 1) ∨
       (ways = 1 ∧ one_cent_coins >= 1 ∧ two_cent_coins >= 1 ∧ five_cent_coin >= 1)
   ) :=
  sorry

end xiao_wang_ways_to_make_8_cents_l106_10615


namespace max_sum_a_b_c_d_l106_10621

theorem max_sum_a_b_c_d (a c d : ℤ) (b : ℕ) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  a + b + c + d = -5 := 
sorry

end max_sum_a_b_c_d_l106_10621


namespace percentage_of_Muscovy_ducks_l106_10622

theorem percentage_of_Muscovy_ducks
  (N : ℕ) (M : ℝ) (female_percentage : ℝ) (female_Muscovy : ℕ)
  (hN : N = 40)
  (hfemale_percentage : female_percentage = 0.30)
  (hfemale_Muscovy : female_Muscovy = 6)
  (hcondition : female_percentage * M * N = female_Muscovy) 
  : M = 0.5 := 
sorry

end percentage_of_Muscovy_ducks_l106_10622


namespace grandmother_ratio_l106_10686

noncomputable def Grace_Age := 60
noncomputable def Mother_Age := 80

theorem grandmother_ratio :
  ∃ GM, Grace_Age = (3 / 8 : Rat) * GM ∧ GM / Mother_Age = 2 :=
by
  sorry

end grandmother_ratio_l106_10686


namespace find_total_shaded_area_l106_10613

/-- Definition of the rectangles' dimensions and overlap conditions -/
def rect1_length : ℕ := 4
def rect1_width : ℕ := 15
def rect2_length : ℕ := 5
def rect2_width : ℕ := 10
def rect3_length : ℕ := 3
def rect3_width : ℕ := 18
def shared_side_length : ℕ := 4
def trip_overlap_width : ℕ := 3

/-- Calculation of the rectangular overlap using given conditions -/
theorem find_total_shaded_area : (rect1_length * rect1_width + rect2_length * rect2_width + rect3_length * rect3_width - shared_side_length * shared_side_length - trip_overlap_width * shared_side_length) = 136 :=
    by sorry

end find_total_shaded_area_l106_10613


namespace last_digits_nn_periodic_l106_10637

theorem last_digits_nn_periodic (n : ℕ) : 
  ∃ p > 0, ∀ k, (n + k * p)^(n + k * p) % 10 = n^n % 10 := 
sorry

end last_digits_nn_periodic_l106_10637


namespace units_digit_of_3_pow_2009_l106_10642

noncomputable def units_digit (n : ℕ) : ℕ :=
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 9
  else if n % 4 = 3 then 7
  else 1

theorem units_digit_of_3_pow_2009 : units_digit (2009) = 3 :=
by
  -- Skipping the proof as instructed
  sorry

end units_digit_of_3_pow_2009_l106_10642


namespace number_subtract_four_l106_10660

theorem number_subtract_four (x : ℤ) (h : 2 * x = 18) : x - 4 = 5 :=
sorry

end number_subtract_four_l106_10660


namespace solve_quadratic_equation_l106_10688

theorem solve_quadratic_equation : 
  ∃ (a b c : ℤ), (0 < a) ∧ (64 * x^2 + 48 * x - 36 = 0) ∧ ((a * x + b)^2 = c) ∧ (a + b + c = 56) := 
by
  sorry

end solve_quadratic_equation_l106_10688


namespace gcd_282_470_l106_10617

theorem gcd_282_470 : Int.gcd 282 470 = 94 := by
  sorry

end gcd_282_470_l106_10617


namespace band_weight_correct_l106_10653

universe u

structure InstrumentGroup where
  count : ℕ
  weight_per_instrument : ℕ

def total_weight (ig : InstrumentGroup) : ℕ :=
  ig.count * ig.weight_per_instrument

def total_band_weight : ℕ :=
  (total_weight ⟨6, 5⟩) + (total_weight ⟨9, 5⟩) +
  (total_weight ⟨8, 10⟩) + (total_weight ⟨3, 20⟩) + (total_weight ⟨2, 15⟩)

theorem band_weight_correct : total_band_weight = 245 := by
  rfl

end band_weight_correct_l106_10653


namespace bob_stickers_l106_10669

variables {B T D : ℕ}

theorem bob_stickers (h1 : D = 72) (h2 : T = 3 * B) (h3 : D = 2 * T) : B = 12 :=
by
  sorry

end bob_stickers_l106_10669


namespace gcd_gx_x_l106_10684

theorem gcd_gx_x (x : ℕ) (h : 2520 ∣ x) : 
  Nat.gcd ((4*x + 5) * (5*x + 2) * (11*x + 8) * (3*x + 7)) x = 280 := 
sorry

end gcd_gx_x_l106_10684


namespace lucas_1500th_day_is_sunday_l106_10682

def days_in_week : ℕ := 7

def start_day : ℕ := 5  -- 0: Monday, 1: Tuesday, ..., 5: Friday

def nth_day_of_life (n : ℕ) : ℕ :=
  (n - 1 + start_day) % days_in_week

theorem lucas_1500th_day_is_sunday : nth_day_of_life 1500 = 0 :=
by
  sorry

end lucas_1500th_day_is_sunday_l106_10682


namespace range_of_f_l106_10657

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem range_of_f 
  (x : ℝ) : f (x - 1) + f (x + 1) > 0 ↔ x ∈ Set.Ioi 0 :=
by
  sorry

end range_of_f_l106_10657


namespace min_value_a2b3c_l106_10638

theorem min_value_a2b3c {m : ℝ} (hm : m > 0)
  (hineq : ∀ x : ℝ, |x + 1| + |2 * x - 1| ≥ m)
  {a b c : ℝ} (habc : a^2 + 2 * b^2 + 3 * c^2 = m) :
  a + 2 * b + 3 * c ≥ -3 :=
sorry

end min_value_a2b3c_l106_10638


namespace soccer_lineup_count_l106_10628

theorem soccer_lineup_count :
  let total_players : ℕ := 16
  let total_starters : ℕ := 7
  let m_j_players : ℕ := 2 -- Michael and John
  let other_players := total_players - m_j_players
  let total_ways : ℕ :=
    2 * Nat.choose other_players (total_starters - 1) + Nat.choose other_players (total_starters - 2)
  total_ways = 8008
:= sorry

end soccer_lineup_count_l106_10628


namespace nat_number_of_the_form_l106_10630

theorem nat_number_of_the_form (a b : ℕ) (h : ∃ (a b : ℕ), a * a * 3 + b * b * 32 = n) :
  ∃ (a' b' : ℕ), a' * a' * 3 + b' * b' * 32 = 97 * n  :=
  sorry

end nat_number_of_the_form_l106_10630


namespace number_of_students_l106_10602

theorem number_of_students (total_students : ℕ) :
  (total_students = 19 * 6 + 4) ∧ 
  (∃ (x y : ℕ), x + y = 22 ∧ x > 7 ∧ total_students = x * 6 + y * 5) →
  total_students = 118 :=
by
  sorry

end number_of_students_l106_10602


namespace last_score_is_80_l106_10694

-- Define the list of scores
def scores : List ℕ := [71, 76, 80, 82, 91]

-- Define the total sum of the scores
def total_sum : ℕ := 400

-- Define the condition that the average after each score is an integer
def average_integer_condition (scores : List ℕ) (total_sum : ℕ) : Prop :=
  ∀ (sublist : List ℕ), sublist ≠ [] → sublist ⊆ scores → 
  (sublist.sum / sublist.length : ℕ) * sublist.length = sublist.sum

-- Define the proposition to prove that the last score entered must be 80
theorem last_score_is_80 : ∃ (last_score : ℕ), (last_score = 80) ∧
  average_integer_condition scores total_sum :=
sorry

end last_score_is_80_l106_10694


namespace part_1_part_2_l106_10661

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

-- (Part 1): Prove the value of a
theorem part_1 (a : ℝ) (P : ℝ × ℝ) (hP : P = (a, -4)) :
  (∃ t : ℝ, ∃ t₂ : ℝ, t ≠ t₂ ∧ P.2 = (2 * t^3 - 3 * t^2 + 1) + (6 * t^2 - 6 * t) * (a - t)) →
  a = -1 ∨ a = 7 / 2 :=
sorry

-- (Part 2): Prove the range of k
noncomputable def g (x k : ℝ) : ℝ := k * x + 1 - Real.log x

noncomputable def h (x k : ℝ) : ℝ := min (f x) (g x k)

theorem part_2 (k : ℝ) :
  (∀ x > 0, h x k = 0 → (x = 1 ∨ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 k = 0 ∧ h x2 k = 0)) →
  0 < k ∧ k < 1 / Real.exp 2 :=
sorry

end part_1_part_2_l106_10661


namespace no_such_set_exists_l106_10675

theorem no_such_set_exists :
  ¬ ∃ (A : Finset ℕ), A.card = 11 ∧
  (∀ (s : Finset ℕ), s ⊆ A → s.card = 6 → ¬ 6 ∣ s.sum id) :=
sorry

end no_such_set_exists_l106_10675


namespace stamp_distribution_correct_l106_10600

variables {W : ℕ} -- We use ℕ (natural numbers) for simplicity but this can be any type representing weight.

-- Number of envelopes that weigh less than W and need 2 stamps each
def envelopes_lt_W : ℕ := 6

-- Number of stamps per envelope if the envelope weighs less than W
def stamps_lt_W : ℕ := 2

-- Number of envelopes in total
def total_envelopes : ℕ := 14

-- Number of stamps for the envelopes that weigh less
def total_stamps_lt_W : ℕ := envelopes_lt_W * stamps_lt_W

-- Total stamps bought by Micah
def total_stamps_bought : ℕ := 52

-- Stamps left for envelopes that weigh more than W
def stamps_remaining : ℕ := total_stamps_bought - total_stamps_lt_W

-- Remaining envelopes that need stamps (those that weigh more than W)
def envelopes_gt_W : ℕ := total_envelopes - envelopes_lt_W

-- Number of stamps required per envelope that weighs more than W
def stamps_gt_W : ℕ := 5

-- Total stamps needed for the envelopes that weigh more than W
def total_stamps_needed_gt_W : ℕ := envelopes_gt_W * stamps_gt_W

theorem stamp_distribution_correct :
  total_stamps_bought = (total_stamps_lt_W + total_stamps_needed_gt_W) :=
by
  sorry

end stamp_distribution_correct_l106_10600


namespace pyramid_total_surface_area_l106_10697

theorem pyramid_total_surface_area :
  ∀ (s h : ℝ), s = 8 → h = 10 →
  6 * (1/2 * s * (Real.sqrt (h^2 - (s/2)^2))) = 48 * Real.sqrt 21 :=
by
  intros s h s_eq h_eq
  rw [s_eq, h_eq]
  sorry

end pyramid_total_surface_area_l106_10697


namespace pages_left_after_all_projects_l106_10681

-- Definitions based on conditions
def initial_pages : ℕ := 120
def pages_for_science : ℕ := (initial_pages * 25) / 100
def pages_for_math : ℕ := 10
def pages_after_science_and_math : ℕ := initial_pages - pages_for_science - pages_for_math
def pages_for_history : ℕ := (initial_pages * 15) / 100
def pages_after_history : ℕ := pages_after_science_and_math - pages_for_history
def remaining_pages : ℕ := pages_after_history / 2

theorem pages_left_after_all_projects :
  remaining_pages = 31 :=
  by
  sorry

end pages_left_after_all_projects_l106_10681


namespace ellipse_equation_correct_l106_10672

noncomputable def ellipse_equation_proof : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ), (x - 2 * y + 4 = 0) ∧ (∃ (f : ℝ × ℝ), f = (-4, 0)) ∧ (∃ (v : ℝ × ℝ), v = (0, 2)) → 
    (x^2 / (a^2) + y^2 / (b^2) = 1 → x^2 / 20 + y^2 / 4 = 1))

theorem ellipse_equation_correct : ellipse_equation_proof :=
  sorry

end ellipse_equation_correct_l106_10672
