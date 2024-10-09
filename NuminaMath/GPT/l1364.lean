import Mathlib

namespace value_of_expression_l1364_136418

theorem value_of_expression {a b : ℝ} (h1 : 2 * a^2 + 6 * a - 14 = 0) (h2 : 2 * b^2 + 6 * b - 14 = 0) :
  (2 * a - 3) * (4 * b - 6) = -2 :=
by
  sorry

end value_of_expression_l1364_136418


namespace solve_quadratic_eq_l1364_136438

theorem solve_quadratic_eq (x : ℝ) : (x^2 - 2*x + 1 = 9) → (x = 4 ∨ x = -2) :=
by
  intro h
  sorry

end solve_quadratic_eq_l1364_136438


namespace batsman_average_l1364_136490

theorem batsman_average (A : ℝ) (h1 : 24 * A < 95) 
                        (h2 : 24 * A + 95 = 25 * (A + 3.5)) : A + 3.5 = 11 :=
by
  sorry

end batsman_average_l1364_136490


namespace minimum_value_of_expression_l1364_136422

theorem minimum_value_of_expression :
  ∀ x y : ℝ, x^2 - x * y + y^2 ≥ 0 :=
by
  sorry

end minimum_value_of_expression_l1364_136422


namespace first_point_x_coord_l1364_136453

variables (m n : ℝ)

theorem first_point_x_coord (h1 : m = 2 * n + 5) (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  m = 2 * n + 5 :=
by 
  sorry

end first_point_x_coord_l1364_136453


namespace sum_of_coordinates_l1364_136469

-- Define the points C and D and the conditions
def point_C : ℝ × ℝ := (0, 0)

def point_D (x : ℝ) : ℝ × ℝ := (x, 5)

def slope_CD (x : ℝ) : Prop :=
  (5 - 0) / (x - 0) = 3 / 4

-- The required theorem to be proved
theorem sum_of_coordinates (D : ℝ × ℝ)
  (hD : D.snd = 5)
  (h_slope : slope_CD D.fst) :
  D.fst + D.snd = 35 / 3 :=
sorry

end sum_of_coordinates_l1364_136469


namespace tan_alpha_expression_l1364_136497

theorem tan_alpha_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 :=
by
  sorry

end tan_alpha_expression_l1364_136497


namespace annual_interest_rate_equivalent_l1364_136423

noncomputable def quarterly_compound_rate : ℝ := 1 + 0.02
noncomputable def annual_compound_amount : ℝ := quarterly_compound_rate ^ 4

theorem annual_interest_rate_equivalent : 
  (annual_compound_amount - 1) * 100 = 8.24 := 
by
  sorry

end annual_interest_rate_equivalent_l1364_136423


namespace sum_of_squares_of_rates_equals_536_l1364_136443

-- Define the biking, jogging, and swimming rates as integers.
variables (b j s : ℤ)

-- Condition: Ed's total distance equation.
def ed_distance_eq : Prop := 3 * b + 2 * j + 4 * s = 80

-- Condition: Sue's total distance equation.
def sue_distance_eq : Prop := 4 * b + 3 * j + 2 * s = 98

-- The main statement to prove.
theorem sum_of_squares_of_rates_equals_536 (hb : b ≥ 0) (hj : j ≥ 0) (hs : s ≥ 0) 
  (h1 : ed_distance_eq b j s) (h2 : sue_distance_eq b j s) :
  b^2 + j^2 + s^2 = 536 :=
by sorry

end sum_of_squares_of_rates_equals_536_l1364_136443


namespace diamonds_in_G_15_l1364_136478

/-- Define the number of diamonds in G_n -/
def diamonds_in_G (n : ℕ) : ℕ :=
  if n < 3 then 1
  else 3 * n ^ 2 - 3 * n + 1

/-- Theorem to prove the number of diamonds in G_15 is 631 -/
theorem diamonds_in_G_15 : diamonds_in_G 15 = 631 :=
by
  -- The proof is omitted
  sorry

end diamonds_in_G_15_l1364_136478


namespace show_revenue_l1364_136404

variable (tickets_first_show : Nat) (tickets_cost : Nat) (multiplicator : Nat)
variable (tickets_second_show : Nat := multiplicator * tickets_first_show)
variable (total_tickets : Nat := tickets_first_show + tickets_second_show)
variable (total_revenue : Nat := total_tickets * tickets_cost)

theorem show_revenue :
    tickets_first_show = 200 ∧ tickets_cost = 25 ∧ multiplicator = 3 →
    total_revenue = 20000 := 
by
    intros h
    sorry

end show_revenue_l1364_136404


namespace roots_of_polynomial_inequality_l1364_136440

theorem roots_of_polynomial_inequality :
  (∃ (p q r s : ℂ), (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) ∧
  (p * q * r * s = 3) ∧ (p*q + p*r + p*s + q*r + q*s + r*s = 11)) →
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3) :=
by
  sorry

end roots_of_polynomial_inequality_l1364_136440


namespace possible_values_of_K_l1364_136491

theorem possible_values_of_K (K N : ℕ) (h1 : K * (K + 1) = 2 * N^2) (h2 : N < 100) :
  K = 1 ∨ K = 8 ∨ K = 49 :=
sorry

end possible_values_of_K_l1364_136491


namespace seven_nat_sum_divisible_by_5_l1364_136493

theorem seven_nat_sum_divisible_by_5 
  (a b c d e f g : ℕ)
  (h1 : (b + c + d + e + f + g) % 5 = 0)
  (h2 : (a + c + d + e + f + g) % 5 = 0)
  (h3 : (a + b + d + e + f + g) % 5 = 0)
  (h4 : (a + b + c + e + f + g) % 5 = 0)
  (h5 : (a + b + c + d + f + g) % 5 = 0)
  (h6 : (a + b + c + d + e + g) % 5 = 0)
  (h7 : (a + b + c + d + e + f) % 5 = 0)
  : 
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0 ∧ f % 5 = 0 ∧ g % 5 = 0 :=
sorry

end seven_nat_sum_divisible_by_5_l1364_136493


namespace net_effect_on_sale_l1364_136465

variable (P S : ℝ) (orig_revenue : ℝ := P * S) (new_revenue : ℝ := 0.7 * P * 1.8 * S)

theorem net_effect_on_sale : new_revenue = orig_revenue * 1.26 := by
  sorry

end net_effect_on_sale_l1364_136465


namespace second_pipe_fill_time_l1364_136467

theorem second_pipe_fill_time (x : ℝ) :
  (1 / 18) + (1 / x) - (1 / 45) = (1 / 15) → x = 30 :=
by
  intro h
  sorry

end second_pipe_fill_time_l1364_136467


namespace rational_roots_iff_a_eq_b_l1364_136468

theorem rational_roots_iff_a_eq_b (a b : ℤ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x : ℚ, x^2 + (a + b)^2 * x + 4 * a * b = 1) ↔ a = b :=
by
  sorry

end rational_roots_iff_a_eq_b_l1364_136468


namespace total_loss_is_correct_l1364_136472

variable (A P : ℝ)
variable (Ashok_loss Pyarelal_loss : ℝ)

-- Condition 1: Ashok's capital is 1/9 of Pyarelal's capital
def ashokCapital (A P : ℝ) : Prop :=
  A = (1 / 9) * P

-- Condition 2: Pyarelal's loss was Rs 1800
def pyarelalLoss (Pyarelal_loss : ℝ) : Prop :=
  Pyarelal_loss = 1800

-- Question: What was the total loss in the business?
def totalLoss (Ashok_loss Pyarelal_loss : ℝ) : ℝ :=
  Ashok_loss + Pyarelal_loss

-- The mathematically equivalent proof problem statement
theorem total_loss_is_correct (P A : ℝ) (Ashok_loss Pyarelal_loss : ℝ)
  (h1 : ashokCapital A P)
  (h2 : pyarelalLoss Pyarelal_loss)
  (h3 : Ashok_loss = (1 / 9) * Pyarelal_loss) :
  totalLoss Ashok_loss Pyarelal_loss = 2000 := by
  sorry

end total_loss_is_correct_l1364_136472


namespace parabola_c_value_l1364_136461

theorem parabola_c_value {b c : ℝ} :
  (1:ℝ)^2 + b * (1:ℝ) + c = 2 → 
  (4:ℝ)^2 + b * (4:ℝ) + c = 5 → 
  (7:ℝ)^2 + b * (7:ℝ) + c = 2 →
  c = 9 :=
by
  intros h1 h2 h3
  sorry

end parabola_c_value_l1364_136461


namespace bryden_payment_l1364_136407

theorem bryden_payment :
  (let face_value := 0.25
   let quarters := 6
   let collector_multiplier := 16
   let discount := 0.10
   let initial_payment := collector_multiplier * (quarters * face_value)
   let final_payment := initial_payment - (initial_payment * discount)
   final_payment = 21.6) :=
by
  sorry

end bryden_payment_l1364_136407


namespace mark_total_votes_l1364_136494

-- Definitions for the problem conditions
def first_area_registered_voters : ℕ := 100000
def first_area_undecided_percentage : ℕ := 5
def first_area_mark_votes_percentage : ℕ := 70

def remaining_area_increase_percentage : ℕ := 20
def remaining_area_undecided_percentage : ℕ := 7
def multiplier_for_remaining_area_votes : ℕ := 2

-- The Lean statement
theorem mark_total_votes : 
  let first_area_undecided_voters := first_area_registered_voters * first_area_undecided_percentage / 100
  let first_area_votes_cast := first_area_registered_voters - first_area_undecided_voters
  let first_area_mark_votes := first_area_votes_cast * first_area_mark_votes_percentage / 100

  let remaining_area_registered_voters := first_area_registered_voters * (1 + remaining_area_increase_percentage / 100)
  let remaining_area_undecided_voters := remaining_area_registered_voters * remaining_area_undecided_percentage / 100
  let remaining_area_votes_cast := remaining_area_registered_voters - remaining_area_undecided_voters
  let remaining_area_mark_votes := first_area_mark_votes * multiplier_for_remaining_area_votes

  let total_mark_votes := first_area_mark_votes + remaining_area_mark_votes
  total_mark_votes = 199500 := 
by
  -- We skipped the proof (it's not required as per instructions)
  sorry

end mark_total_votes_l1364_136494


namespace part_I_part_II_l1364_136411

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem part_I (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1) :
  ∃ x ∈ Set.Ioo m (m + 1), ∀ y ∈ Set.Ioo m (m + 1), f y ≤ f x := sorry

theorem part_II (x : ℝ) (h : 1 < x) :
  (x + 1) * (x + Real.exp (-x)) * f x > 2 * (1 + 1 / Real.exp 1) := sorry

end part_I_part_II_l1364_136411


namespace gp_condition_necessity_l1364_136499

theorem gp_condition_necessity {a b c : ℝ} 
    (h_gp: ∃ r: ℝ, b = a * r ∧ c = a * r^2 ) : b^2 = a * c :=
by
  sorry

end gp_condition_necessity_l1364_136499


namespace length_of_field_l1364_136417

-- Define the problem conditions
variables (width length : ℕ)
  (pond_area field_area : ℕ)
  (h1 : length = 2 * width)
  (h2 : pond_area = 64)
  (h3 : pond_area = field_area / 8)

-- Define the proof problem
theorem length_of_field : length = 32 :=
by
  -- We'll provide the proof later
  sorry

end length_of_field_l1364_136417


namespace fraction_exponent_evaluation_l1364_136429

theorem fraction_exponent_evaluation : 
  (3 ^ 10 + 3 ^ 8) / (3 ^ 10 - 3 ^ 8) = 5 / 4 :=
by sorry

end fraction_exponent_evaluation_l1364_136429


namespace pyramid_base_is_octagon_l1364_136484
-- Import necessary library

-- Declare the problem
theorem pyramid_base_is_octagon (A : Nat) (h : A = 8) : A = 8 :=
by
  -- Proof goes here
  sorry

end pyramid_base_is_octagon_l1364_136484


namespace no_pairs_exist_l1364_136444

theorem no_pairs_exist (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : (1/a + 1/b = 2/(a+b)) → False :=
by
  sorry

end no_pairs_exist_l1364_136444


namespace find_constants_l1364_136441

theorem find_constants (A B C : ℝ) (hA : A = 7) (hB : B = -9) (hC : C = 5) :
  (∀ (x : ℝ), x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → 
    ( -2 * x ^ 2 + 5 * x - 7) / (x ^ 3 - x) = A / x + (B * x + C) / (x ^ 2 - 1) ) :=
by
  intros x hx
  rw [hA, hB, hC]
  sorry

end find_constants_l1364_136441


namespace simplify_expression_l1364_136462

variable (x : ℝ)

theorem simplify_expression :
  3 * x^3 + 4 * x + 5 * x^2 + 2 - (7 - 3 * x^3 - 4 * x - 5 * x^2) =
  6 * x^3 + 10 * x^2 + 8 * x - 5 :=
by
  sorry

end simplify_expression_l1364_136462


namespace most_probable_light_l1364_136488

theorem most_probable_light (red_duration : ℕ) (yellow_duration : ℕ) (green_duration : ℕ) :
  red_duration = 30 ∧ yellow_duration = 5 ∧ green_duration = 40 →
  (green_duration / (red_duration + yellow_duration + green_duration) > red_duration / (red_duration + yellow_duration + green_duration)) ∧
  (green_duration / (red_duration + yellow_duration + green_duration) > yellow_duration / (red_duration + yellow_duration + green_duration)) :=
by
  sorry

end most_probable_light_l1364_136488


namespace find_certain_number_l1364_136427

theorem find_certain_number (x : ℝ) (h : 0.80 * x = (4 / 5 * 20) + 16) : x = 40 :=
by sorry

end find_certain_number_l1364_136427


namespace product_of_x_y_l1364_136450

-- Assume the given conditions
variables (EF GH FG HE : ℝ)
variables (x y : ℝ)
variable (EFGH : Type)

-- Conditions given
axiom h1 : EF = 58
axiom h2 : GH = 3 * x + 1
axiom h3 : FG = 2 * y^2
axiom h4 : HE = 36
-- It is given that EFGH forms a parallelogram
axiom h5 : EF = GH
axiom h6 : FG = HE

-- The product of x and y is determined by the conditions
theorem product_of_x_y : x * y = 57 * Real.sqrt 2 :=
by
  sorry

end product_of_x_y_l1364_136450


namespace box_surface_area_l1364_136464

theorem box_surface_area
  (a b c : ℕ)
  (h1 : a < 10)
  (h2 : b < 10)
  (h3 : c < 10)
  (h4 : a * b * c = 280) : 2 * (a * b + b * c + c * a) = 262 := 
sorry

end box_surface_area_l1364_136464


namespace point_p_locus_equation_l1364_136460

noncomputable def locus_point_p (x y : ℝ) : Prop :=
  ∀ (k b x1 y1 x2 y2 : ℝ), 
  (x1^2 + y1^2 = 1) ∧ 
  (x2^2 + y2^2 = 1) ∧ 
  (3 * x1 * x + 4 * y1 * y = 12) ∧ 
  (3 * x2 * x + 4 * y2 * y = 12) ∧ 
  (1 + k^2 = b^2) ∧ 
  (y = 3 / b) ∧ 
  (x = -4 * k / (3 * b)) → 
  x^2 / 16 + y^2 / 9 = 1

theorem point_p_locus_equation :
  ∀ (x y : ℝ), locus_point_p x y → (x^2 / 16 + y^2 / 9 = 1) :=
by
  intros x y h
  sorry

end point_p_locus_equation_l1364_136460


namespace fair_coin_flip_difference_l1364_136498

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end fair_coin_flip_difference_l1364_136498


namespace range_of_x_l1364_136420

noncomputable def range_of_independent_variable (x : ℝ) : Prop :=
  1 - x > 0

theorem range_of_x (x : ℝ) : range_of_independent_variable x → x < 1 :=
by sorry

end range_of_x_l1364_136420


namespace best_fitting_model_l1364_136408

-- Define the \(R^2\) values for each model
def R2_Model1 : ℝ := 0.75
def R2_Model2 : ℝ := 0.90
def R2_Model3 : ℝ := 0.25
def R2_Model4 : ℝ := 0.55

-- State that Model 2 is the best fitting model
theorem best_fitting_model : R2_Model2 = max (max R2_Model1 R2_Model2) (max R2_Model3 R2_Model4) :=
by -- Proof skipped
  sorry

end best_fitting_model_l1364_136408


namespace maria_final_bottle_count_l1364_136486

-- Define the initial conditions
def initial_bottles : ℕ := 14
def bottles_drunk : ℕ := 8
def bottles_bought : ℕ := 45

-- State the theorem to prove
theorem maria_final_bottle_count : initial_bottles - bottles_drunk + bottles_bought = 51 :=
by
  sorry

end maria_final_bottle_count_l1364_136486


namespace find_ks_l1364_136492

theorem find_ks (n : ℕ) (h_pos : 0 < n) :
  ∀ k, k ∈ (Finset.range (2 * n * n + 1)).erase 0 ↔ (n^2 - n + 1 ≤ k ∧ k ≤ n^2) ∨ (2*n ∣ k ∧ k ≥ n^2 - n + 1) :=
sorry

end find_ks_l1364_136492


namespace solve_arithmetic_sequence_l1364_136406

theorem solve_arithmetic_sequence (y : ℝ) (h1 : y ^ 2 = (4 + 25) / 2) (h2 : y > 0) :
  y = Real.sqrt 14.5 :=
sorry

end solve_arithmetic_sequence_l1364_136406


namespace average_marks_of_class_l1364_136434

theorem average_marks_of_class :
  (∀ (students total_students: ℕ) (marks95 marks0: ℕ) (avg_remaining: ℕ),
    total_students = 25 →
    students = 3 →
    marks95 = 95 →
    students = 5 →
    marks0 = 0 →
    (total_students - students - students) = 17 →
    avg_remaining = 45 →
    ((students * marks95 + students * marks0 + (total_students - students - students) * avg_remaining) / total_students) = 42)
:= sorry

end average_marks_of_class_l1364_136434


namespace find_angle_B_l1364_136413

-- Define the triangle with the given conditions
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a * Real.cos B - b * Real.cos A = c ∧ C = Real.pi / 5

-- Prove angle B given the conditions
theorem find_angle_B {A B C a b c : ℝ} (h : triangle_condition A B C a b c) : 
  B = 3 * Real.pi / 10 :=
sorry

end find_angle_B_l1364_136413


namespace sum_of_first_six_primes_l1364_136405

theorem sum_of_first_six_primes : (2 + 3 + 5 + 7 + 11 + 13) = 41 :=
by
  sorry

end sum_of_first_six_primes_l1364_136405


namespace aeroplane_speed_l1364_136481

theorem aeroplane_speed (D : ℝ) (S : ℝ) (h1 : D = S * 6) (h2 : D = 540 * (14 / 3)) :
  S = 420 := by
  sorry

end aeroplane_speed_l1364_136481


namespace simplify_expression_l1364_136476

theorem simplify_expression (x : ℝ) : 3 * x^2 - 4 * x^2 = -x^2 :=
by
  sorry

end simplify_expression_l1364_136476


namespace Jeff_has_20_trucks_l1364_136439

theorem Jeff_has_20_trucks
  (T C : ℕ)
  (h1 : C = 2 * T)
  (h2 : T + C = 60) :
  T = 20 :=
sorry

end Jeff_has_20_trucks_l1364_136439


namespace count_three_digit_multiples_13_and_5_l1364_136475

theorem count_three_digit_multiples_13_and_5 : 
  ∃ count : ℕ, count = 14 ∧ 
  ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 65 = 0) → 
  (∃ k : ℕ, n = k * 65 ∧ 2 ≤ k ∧ k ≤ 15) → count = 14 :=
by
  sorry

end count_three_digit_multiples_13_and_5_l1364_136475


namespace range_of_m_l1364_136421

noncomputable def range_m (a b : ℝ) (m : ℝ) : Prop :=
  (3 * a + 4 / b = 1) ∧ a > 0 ∧ b > 0 → (1 / a + 3 * b > m)

theorem range_of_m (m : ℝ) : (∀ a b : ℝ, (range_m a b m)) ↔ m < 27 :=
by
  sorry

end range_of_m_l1364_136421


namespace least_value_xy_l1364_136448

theorem least_value_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/9) : x*y = 108 :=
sorry

end least_value_xy_l1364_136448


namespace sum_of_100th_group_is_1010100_l1364_136454

theorem sum_of_100th_group_is_1010100 : (100 + 100^2 + 100^3) = 1010100 :=
by
  sorry

end sum_of_100th_group_is_1010100_l1364_136454


namespace octagon_area_l1364_136410

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end octagon_area_l1364_136410


namespace total_games_l1364_136424

theorem total_games (N : ℕ) (p : ℕ)
  (hPetya : 2 ∣ N)
  (hKolya : 3 ∣ N)
  (hVasya : 5 ∣ N)
  (hGamesNotInvolving : 2 ≤ N - (N / 2 + N / 3 + N / 5)) :
  N = 30 :=
by
  sorry

end total_games_l1364_136424


namespace fraction_condition_l1364_136425

theorem fraction_condition (x : ℚ) :
  (3 + 2 * x) / (4 + 3 * x) = 5 / 9 ↔ x = -7 / 3 :=
by
  sorry

end fraction_condition_l1364_136425


namespace find_divisor_l1364_136400

-- Definitions based on the conditions
def is_divisor (d : ℕ) (a b k : ℕ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ (b - a) / n = k ∧ k = d

-- Problem statement
theorem find_divisor (a b k : ℕ) (H : b = 43 ∧ a = 10 ∧ k = 11) : ∃ d, d = 3 :=
by
  sorry

end find_divisor_l1364_136400


namespace cut_out_area_l1364_136455

theorem cut_out_area (x : ℝ) (h1 : x * (x - 10) = 1575) : 10 * x - 10 * 10 = 450 := by
  -- Proof to be filled in here
  sorry

end cut_out_area_l1364_136455


namespace minimum_questions_to_find_number_l1364_136496

theorem minimum_questions_to_find_number (n : ℕ) (h : n ≤ 2020) :
  ∃ m, m = 64 ∧ (∀ (strategy : ℕ → ℕ), ∃ questions : ℕ, questions ≤ m ∧ (strategy questions = n)) :=
sorry

end minimum_questions_to_find_number_l1364_136496


namespace error_in_area_l1364_136437

theorem error_in_area (s : ℝ) (h : s > 0) :
  let s_measured := 1.02 * s
  let A_actual := s^2
  let A_measured := s_measured^2
  let error := (A_measured - A_actual) / A_actual * 100
  error = 4.04 := by
  sorry

end error_in_area_l1364_136437


namespace monotonic_increasing_interval_l1364_136451

noncomputable def f (a : ℝ) (h : 0 < a ∧ a < 1) (x : ℝ) := a ^ (-x^2 + 3 * x + 2)

theorem monotonic_increasing_interval (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∀ x1 x2 : ℝ, (3 / 2 < x1 ∧ x1 < x2) → f a h x1 < f a h x2 :=
sorry

end monotonic_increasing_interval_l1364_136451


namespace bacterium_descendants_l1364_136415

theorem bacterium_descendants (n a : ℕ) (h : a ≤ n / 2) :
  ∃ k, a ≤ k ∧ k ≤ 2 * a - 1 := 
sorry

end bacterium_descendants_l1364_136415


namespace bell_rings_count_l1364_136446

-- Defining the conditions
def bell_rings_per_class : ℕ := 2
def total_classes_before_music : ℕ := 4
def bell_rings_during_music_start : ℕ := 1

-- The main proof statement
def total_bell_rings : ℕ :=
  total_classes_before_music * bell_rings_per_class + bell_rings_during_music_start

theorem bell_rings_count : total_bell_rings = 9 := by
  sorry

end bell_rings_count_l1364_136446


namespace chess_club_girls_l1364_136431

theorem chess_club_girls (B G : ℕ) (h1 : B + G = 32) (h2 : (1 / 2 : ℝ) * G + B = 20) : G = 24 :=
by
  -- proof
  sorry

end chess_club_girls_l1364_136431


namespace cos_value_of_angle_l1364_136403

theorem cos_value_of_angle (α : ℝ) (h : Real.sin (α + Real.pi / 6) = 1 / 3) :
  Real.cos (2 * α - 2 * Real.pi / 3) = -7 / 9 :=
by
  sorry

end cos_value_of_angle_l1364_136403


namespace luke_good_games_l1364_136456

-- Definitions
def bought_from_friend : ℕ := 2
def bought_from_garage_sale : ℕ := 2
def defective_games : ℕ := 2

-- The theorem we want to prove
theorem luke_good_games :
  bought_from_friend + bought_from_garage_sale - defective_games = 2 := 
by 
  sorry

end luke_good_games_l1364_136456


namespace Oscar_height_correct_l1364_136401

-- Definitions of the given conditions
def Tobias_height : ℕ := 184
def avg_height : ℕ := 178

def heights_valid (Victor Peter Oscar Tobias : ℕ) : Prop :=
  Tobias = 184 ∧ (Tobias + Victor + Peter + Oscar) / 4 = 178 ∧ 
  Victor = Tobias + (Tobias - Peter) ∧ 
  Oscar = Peter - (Tobias - Peter)

theorem Oscar_height_correct :
  ∃ (k : ℕ), ∀ (Victor Peter Oscar : ℕ), heights_valid Victor Peter Oscar Tobias_height →
  Oscar = 160 :=
by
  sorry

end Oscar_height_correct_l1364_136401


namespace total_profit_percentage_l1364_136459

theorem total_profit_percentage (total_apples : ℕ) (percent_sold_10 : ℝ) (percent_sold_30 : ℝ) (profit_10 : ℝ) (profit_30 : ℝ) : 
  total_apples = 280 → 
  percent_sold_10 = 0.40 → 
  percent_sold_30 = 0.60 → 
  profit_10 = 0.10 → 
  profit_30 = 0.30 → 
  ((percent_sold_10 * total_apples * (1 + profit_10) + percent_sold_30 * total_apples * (1 + profit_30) - total_apples) / total_apples * 100) = 22 := 
by 
  intros; sorry

end total_profit_percentage_l1364_136459


namespace total_number_of_girls_l1364_136474

-- Define the given initial number of girls and the number of girls joining the school
def initial_girls : Nat := 732
def girls_joined : Nat := 682
def total_girls : Nat := 1414

-- Formalize the problem
theorem total_number_of_girls :
  initial_girls + girls_joined = total_girls :=
by
  -- placeholder for proof
  sorry

end total_number_of_girls_l1364_136474


namespace monthly_installment_amount_l1364_136485

variable (cashPrice : ℕ) (deposit : ℕ) (monthlyInstallments : ℕ) (savingsIfCash : ℕ)

-- Defining the conditions
def conditions := 
  cashPrice = 8000 ∧ 
  deposit = 3000 ∧ 
  monthlyInstallments = 30 ∧ 
  savingsIfCash = 4000

-- Proving the amount of each monthly installment
theorem monthly_installment_amount (h : conditions cashPrice deposit monthlyInstallments savingsIfCash) : 
  (12000 - deposit) / monthlyInstallments = 300 :=
sorry

end monthly_installment_amount_l1364_136485


namespace hollis_student_loan_l1364_136466

theorem hollis_student_loan
  (interest_loan1 : ℝ)
  (interest_loan2 : ℝ)
  (total_loan1 : ℝ)
  (total_loan2 : ℝ)
  (additional_amount : ℝ)
  (total_interest_paid : ℝ) :
  interest_loan1 = 0.07 →
  total_loan1 = total_loan2 + additional_amount →
  additional_amount = 1500 →
  total_interest_paid = 617 →
  total_loan2 = 4700 →
  total_loan1 * interest_loan1 + total_loan2 * interest_loan2 = total_interest_paid →
  total_loan2 = 4700 :=
by
  sorry

end hollis_student_loan_l1364_136466


namespace even_function_implies_a_zero_l1364_136419

theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, (x^2 - |x + a|) = (x^2 - |x - a|)) → a = 0 :=
by
  sorry

end even_function_implies_a_zero_l1364_136419


namespace leftover_value_is_correct_l1364_136470

def value_of_leftover_coins (total_quarters total_dimes quarters_per_roll dimes_per_roll : ℕ) : ℝ :=
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters * 0.25) + (leftover_dimes * 0.10)

def michael_quarters : ℕ := 95
def michael_dimes : ℕ := 172
def anna_quarters : ℕ := 140
def anna_dimes : ℕ := 287
def quarters_per_roll : ℕ := 50
def dimes_per_roll : ℕ := 40

def total_quarters : ℕ := michael_quarters + anna_quarters
def total_dimes : ℕ := michael_dimes + anna_dimes

theorem leftover_value_is_correct : 
  value_of_leftover_coins total_quarters total_dimes quarters_per_roll dimes_per_roll = 10.65 :=
by
  sorry

end leftover_value_is_correct_l1364_136470


namespace nancy_packs_l1364_136480

theorem nancy_packs (total_bars packs_bars : ℕ) (h_total : total_bars = 30) (h_packs : packs_bars = 5) :
  total_bars / packs_bars = 6 :=
by
  sorry

end nancy_packs_l1364_136480


namespace transform_equation_to_square_form_l1364_136412

theorem transform_equation_to_square_form : 
  ∀ x : ℝ, (x^2 - 6 * x = 0) → ∃ m n : ℝ, (x + m) ^ 2 = n ∧ m = -3 ∧ n = 9 := 
sorry

end transform_equation_to_square_form_l1364_136412


namespace arithmetic_sequence_value_l1364_136487

theorem arithmetic_sequence_value (a_1 d : ℤ) (h : (a_1 + 2 * d) + (a_1 + 7 * d) = 10) : 
  3 * (a_1 + 4 * d) + (a_1 + 6 * d) = 20 :=
by
  sorry

end arithmetic_sequence_value_l1364_136487


namespace song_book_cost_correct_l1364_136416

noncomputable def cost_of_trumpet : ℝ := 145.16
noncomputable def total_spent : ℝ := 151.00
noncomputable def cost_of_song_book : ℝ := total_spent - cost_of_trumpet

theorem song_book_cost_correct : cost_of_song_book = 5.84 :=
  by
    sorry

end song_book_cost_correct_l1364_136416


namespace only_zero_function_satisfies_conditions_l1364_136482

def is_increasing (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n > m → f n ≥ f m

def satisfies_functional_equation (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, f (n * m) = f n + f m

theorem only_zero_function_satisfies_conditions :
  ∀ f : ℕ → ℕ, 
  (is_increasing f) ∧ (satisfies_functional_equation f) → (∀ n : ℕ, f n = 0) :=
by
  sorry

end only_zero_function_satisfies_conditions_l1364_136482


namespace matrix_multiplication_correct_l1364_136414

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, 3, -1], ![1, -2, 5], ![0, 6, 1]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 0, 4], ![3, 2, -1], ![0, 4, -2]]

def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![11, 2, 7], ![-5, 16, -4], ![18, 16, -8]]

theorem matrix_multiplication_correct :
  A * B = C :=
by
  sorry

end matrix_multiplication_correct_l1364_136414


namespace intersection_A_B_l1364_136436

def A := {x : ℝ | 2 ≤ x ∧ x ≤ 8}
def B := {x : ℝ | x^2 - 3 * x - 4 < 0}
def expected := {x : ℝ | 2 ≤ x ∧ x < 4 }

theorem intersection_A_B : (A ∩ B) = expected := 
by 
  sorry

end intersection_A_B_l1364_136436


namespace cabbage_price_is_4_02_l1364_136447

noncomputable def price_of_cabbage (broccoli_price_per_pound: ℝ) (broccoli_pounds: ℝ) 
                                    (orange_price_each: ℝ) (oranges: ℝ) 
                                    (bacon_price_per_pound: ℝ) (bacon_pounds: ℝ) 
                                    (chicken_price_per_pound: ℝ) (chicken_pounds: ℝ) 
                                    (budget_percentage_for_meat: ℝ) 
                                    (meat_price: ℝ) : ℝ := 
  let broccoli_total := broccoli_pounds * broccoli_price_per_pound
  let oranges_total := oranges * orange_price_each
  let bacon_total := bacon_pounds * bacon_price_per_pound
  let chicken_total := chicken_pounds * chicken_price_per_pound
  let subtotal := broccoli_total + oranges_total + bacon_total + chicken_total
  let total_budget := meat_price / budget_percentage_for_meat
  total_budget - subtotal

theorem cabbage_price_is_4_02 : 
  price_of_cabbage 4 3 0.75 3 3 1 3 2 0.33 9 = 4.02 := 
by 
  sorry

end cabbage_price_is_4_02_l1364_136447


namespace mountain_hill_school_absent_percentage_l1364_136473

theorem mountain_hill_school_absent_percentage :
  let total_students := 120
  let boys := 70
  let girls := 50
  let absent_boys := (1 / 7) * boys
  let absent_girls := (1 / 5) * girls
  let absent_students := absent_boys + absent_girls
  let absent_percentage := (absent_students / total_students) * 100
  absent_percentage = 16.67 := sorry

end mountain_hill_school_absent_percentage_l1364_136473


namespace smallest_possible_stamps_l1364_136402

theorem smallest_possible_stamps (M : ℕ) : 
  ((M % 5 = 2) ∧ (M % 7 = 2) ∧ (M % 9 = 2) ∧ (M > 2)) → M = 317 := 
by 
  sorry

end smallest_possible_stamps_l1364_136402


namespace annual_percentage_increase_20_l1364_136426

variable (P0 P1 : ℕ) (r : ℚ)

-- Population initial condition
def initial_population : Prop := P0 = 10000

-- Population after 1 year condition
def population_after_one_year : Prop := P1 = 12000

-- Define the annual percentage increase formula
def percentage_increase (P0 P1 : ℕ) : ℚ := ((P1 - P0 : ℚ) / P0) * 100

-- State the theorem
theorem annual_percentage_increase_20
  (h1 : initial_population P0)
  (h2 : population_after_one_year P1) :
  percentage_increase P0 P1 = 20 := by
  sorry

end annual_percentage_increase_20_l1364_136426


namespace tan_double_angle_l1364_136483

theorem tan_double_angle (α : ℝ) (h1 : Real.sin (5 * Real.pi / 6) = 1 / 2)
  (h2 : Real.cos (5 * Real.pi / 6) = -Real.sqrt 3 / 2) : 
  Real.tan (2 * α) = Real.sqrt 3 := 
sorry

end tan_double_angle_l1364_136483


namespace investment_months_l1364_136458

theorem investment_months (i_a i_b i_c a_gain total_gain : ℝ) (m : ℝ) :
  i_a = 1 ∧ i_b = 2 * i_a ∧ i_c = 3 * i_a ∧ a_gain = 6100 ∧ total_gain = 18300 ∧ m * i_b * (12 - m) + i_c * 3 * 4 = 12200 →
  a_gain / total_gain = i_a * 12 / (i_a * 12 + i_b * (12 - m) + i_c * 4) → m = 6 :=
by
  intros h1 h2
  obtain ⟨ha, hb, hc, hag, htg, h⟩ := h1
  -- proof omitted
  sorry

end investment_months_l1364_136458


namespace empty_set_negation_l1364_136435

open Set

theorem empty_set_negation (α : Type) : ¬ (∀ s : Set α, ∅ ⊆ s) ↔ (∃ s : Set α, ¬(∅ ⊆ s)) :=
by
  sorry

end empty_set_negation_l1364_136435


namespace ratio_of_width_to_length_l1364_136489

theorem ratio_of_width_to_length (w l : ℕ) (h1 : w * l = 800) (h2 : l - w = 20) : w / l = 1 / 2 :=
by sorry

end ratio_of_width_to_length_l1364_136489


namespace total_steps_to_times_square_l1364_136495

-- Define the conditions
def steps_to_rockefeller : ℕ := 354
def steps_to_times_square_from_rockefeller : ℕ := 228

-- State the theorem using the conditions
theorem total_steps_to_times_square : 
  steps_to_rockefeller + steps_to_times_square_from_rockefeller = 582 := 
  by 
    -- We skip the proof for now
    sorry

end total_steps_to_times_square_l1364_136495


namespace candle_cost_correct_l1364_136409

-- Variables and conditions
def candles_per_cake : Nat := 8
def num_cakes : Nat := 3
def candles_needed : Nat := candles_per_cake * num_cakes

def candles_per_box : Nat := 12
def boxes_needed : Nat := candles_needed / candles_per_box

def cost_per_box : ℝ := 2.5
def total_cost : ℝ := boxes_needed * cost_per_box

-- Proof statement
theorem candle_cost_correct :
  total_cost = 5 := by
  sorry

end candle_cost_correct_l1364_136409


namespace both_selected_probability_l1364_136457

-- Define the probabilities of selection for X and Y
def P_X := 1 / 7
def P_Y := 2 / 9

-- Statement to prove that the probability of both being selected is 2 / 63
theorem both_selected_probability :
  (P_X * P_Y) = (2 / 63) :=
by
  -- Proof skipped
  sorry

end both_selected_probability_l1364_136457


namespace series_sum_eq_l1364_136471

noncomputable def sum_series (k : ℝ) : ℝ :=
  (∑' n : ℕ, (4 * (n + 1) + k) / 3^(n + 1))

theorem series_sum_eq (k : ℝ) : sum_series k = 3 + k / 2 := 
  sorry

end series_sum_eq_l1364_136471


namespace find_diff_eq_l1364_136428

noncomputable def general_solution (y : ℝ → ℝ) : Prop :=
∃ (C1 C2 : ℝ), ∀ x : ℝ, y x = C1 * x + C2

theorem find_diff_eq (y : ℝ → ℝ) (C1 C2 : ℝ) (h : ∀ x : ℝ, y x = C1 * x + C2) :
  ∀ x : ℝ, (deriv (deriv y)) x = 0 :=
by
  sorry

end find_diff_eq_l1364_136428


namespace pencils_needed_l1364_136432

theorem pencils_needed (pencilsA : ℕ) (pencilsB : ℕ) (classroomsA : ℕ) (classroomsB : ℕ) (total_shortage : ℕ)
  (hA : pencilsA = 480)
  (hB : pencilsB = 735)
  (hClassA : classroomsA = 6)
  (hClassB : classroomsB = 9)
  (hShortage : total_shortage = 85) 
  : 90 = 6 + 5 * ((total_shortage / (classroomsA + classroomsB)) + 1) * classroomsB :=
by {
  sorry
}

end pencils_needed_l1364_136432


namespace num_four_digit_snappy_numbers_divisible_by_25_l1364_136445

def is_snappy (n : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 = d4 ∧ d2 = d3

def is_divisible_by_25 (n : ℕ) : Prop :=
  let last_two_digits := n % 100
  last_two_digits = 0 ∨ last_two_digits = 25 ∨ last_two_digits = 50 ∨ last_two_digits = 75

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem num_four_digit_snappy_numbers_divisible_by_25 : 
  ∃ n, n = 3 ∧ (∀ x, is_four_digit x ∧ is_snappy x ∧ is_divisible_by_25 x ↔ x = 5225 ∨ x = 0550 ∨ x = 5775)
:=
sorry

end num_four_digit_snappy_numbers_divisible_by_25_l1364_136445


namespace range_of_m_max_value_of_t_l1364_136479

-- Define the conditions for the quadratic equation problem
def quadratic_eq_has_real_roots (m n : ℝ) := 
  m^2 - 4 * n ≥ 0

def roots_are_negative (m : ℝ) := 
  2 ≤ m ∧ m < 3

-- Question 1: Prove range of m
theorem range_of_m (m : ℝ) (h1 : quadratic_eq_has_real_roots m (3 - m)) : 
  roots_are_negative m :=
sorry

-- Define the conditions for the inequality problem
def quadratic_inequality (m n : ℝ) (t : ℝ) := 
  t ≤ (m-1)^2 + (n-1)^2 + (m-n)^2

-- Question 2: Prove maximum value of t
theorem max_value_of_t (m n t : ℝ) (h1 : quadratic_eq_has_real_roots m n) : 
  quadratic_inequality m n t -> t ≤ 9/8 :=
sorry

end range_of_m_max_value_of_t_l1364_136479


namespace ott_fraction_part_l1364_136452

noncomputable def fractional_part_of_group_money (x : ℝ) (M L N P : ℝ) :=
  let total_initial := M + L + N + P + 2
  let money_received_by_ott := 4 * x
  let ott_final_money := 2 + money_received_by_ott
  let total_final := total_initial + money_received_by_ott
  (ott_final_money / total_final) = (3 / 14)

theorem ott_fraction_part (x : ℝ) (M L N P : ℝ)
    (hM : M = 6 * x) (hL : L = 5 * x) (hN : N = 4 * x) (hP : P = 7 * x) :
    fractional_part_of_group_money x M L N P :=
by
  sorry

end ott_fraction_part_l1364_136452


namespace common_divisors_greatest_l1364_136433

theorem common_divisors_greatest (n : ℕ) (h₁ : ∀ d, d ∣ 120 ∧ d ∣ n ↔ d = 1 ∨ d = 3 ∨ d = 9) : 9 = Nat.gcd 120 n := by
  sorry

end common_divisors_greatest_l1364_136433


namespace calculate_a3_b3_l1364_136449

theorem calculate_a3_b3 (a b : ℝ) (h₁ : a + b = 12) (h₂ : a * b = 20) : a^3 + b^3 = 1008 := 
by
  sorry

end calculate_a3_b3_l1364_136449


namespace find_number_l1364_136463

theorem find_number (x : ℤ) (h : 42 + 3 * x - 10 = 65) : x = 11 := 
by 
  sorry 

end find_number_l1364_136463


namespace complex_identity_l1364_136442

theorem complex_identity (a b : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 - 2 * i) * i = a + b * i) : a * b = 2 :=
by
  sorry

end complex_identity_l1364_136442


namespace cubic_expression_value_l1364_136477

theorem cubic_expression_value (m : ℝ) (h : m^2 + 3 * m - 2023 = 0) :
  m^3 + 2 * m^2 - 2026 * m - 2023 = -4046 :=
by
  sorry

end cubic_expression_value_l1364_136477


namespace smallest_c_inv_l1364_136430

def f (x : ℝ) : ℝ := (x + 3)^2 - 7

theorem smallest_c_inv (c : ℝ) : (∀ x1 x2 : ℝ, c ≤ x1 → c ≤ x2 → f x1 = f x2 → x1 = x2) →
  c = -3 :=
sorry

end smallest_c_inv_l1364_136430
