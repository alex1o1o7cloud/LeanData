import Mathlib

namespace find_number_l905_90558

def exceeding_condition (x : ℝ) : Prop :=
  x = 0.16 * x + 84

theorem find_number : ∃ x : ℝ, exceeding_condition x ∧ x = 100 :=
by
  -- Proof goes here, currently omitted.
  sorry

end find_number_l905_90558


namespace bathroom_length_l905_90554

theorem bathroom_length (A L W : ℝ) (h₁ : A = 8) (h₂ : W = 2) (h₃ : A = L * W) : L = 4 :=
by
  -- Skip the proof with sorry
  sorry

end bathroom_length_l905_90554


namespace find_value_of_z_l905_90596

open Complex

-- Define the given complex number z and imaginary unit i
def z : ℂ := sorry
def i : ℂ := Complex.I

-- Given condition
axiom condition : z / (1 - i) = i ^ 2019

-- Proof that z equals -1 - i
theorem find_value_of_z : z = -1 - i :=
by
  sorry

end find_value_of_z_l905_90596


namespace arcsin_sqrt3_div_2_eq_pi_div_3_l905_90509

theorem arcsin_sqrt3_div_2_eq_pi_div_3 : 
  Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 := 
by 
    sorry

end arcsin_sqrt3_div_2_eq_pi_div_3_l905_90509


namespace solve_for_a_l905_90542

def g (x : ℝ) : ℝ := 5 * x - 6

theorem solve_for_a (a : ℝ) : g a = 4 → a = 2 := by
  sorry

end solve_for_a_l905_90542


namespace ratio_expression_value_l905_90506

theorem ratio_expression_value (a b : ℝ) (h : a / b = 4 / 1) : 
  (a - 3 * b) / (2 * a - b) = 1 / 7 := 
by 
  sorry

end ratio_expression_value_l905_90506


namespace stream_speed_is_one_l905_90561

noncomputable def speed_of_stream (downstream_speed upstream_speed : ℝ) : ℝ :=
  (downstream_speed - upstream_speed) / 2

theorem stream_speed_is_one : speed_of_stream 10 8 = 1 := by
  sorry

end stream_speed_is_one_l905_90561


namespace total_sample_needed_l905_90521

-- Given constants
def elementary_students : ℕ := 270
def junior_high_students : ℕ := 360
def senior_high_students : ℕ := 300
def junior_high_sample : ℕ := 12

-- Calculate the total number of students in the school
def total_students : ℕ := elementary_students + junior_high_students + senior_high_students

-- Define the sampling ratio based on junior high section
def sampling_ratio : ℚ := junior_high_sample / junior_high_students

-- Apply the sampling ratio to the total number of students to get the total sample size
def total_sample : ℚ := sampling_ratio * total_students

-- Prove that the total number of students that need to be sampled is 31
theorem total_sample_needed : total_sample = 31 := sorry

end total_sample_needed_l905_90521


namespace find_x_l905_90508

theorem find_x (t : ℤ) : 
∃ x : ℤ, (x % 7 = 3) ∧ (x^2 % 49 = 44) ∧ (x^3 % 343 = 111) ∧ (x = 343 * t + 17) :=
sorry

end find_x_l905_90508


namespace infinitely_many_not_representable_l905_90513

def can_be_represented_as_p_n_2k (c : ℕ) : Prop :=
  ∃ (p n k : ℕ), Prime p ∧ c = p + n^(2 * k)

theorem infinitely_many_not_representable :
  ∃ᶠ m in at_top, ¬ can_be_represented_as_p_n_2k (2^m + 1) := 
sorry

end infinitely_many_not_representable_l905_90513


namespace price_increase_needed_l905_90598

theorem price_increase_needed (P : ℝ) (hP : P > 0) : (100 * ((P / (0.85 * P)) - 1)) = 17.65 :=
by
  sorry

end price_increase_needed_l905_90598


namespace quadratic_roots_prime_distinct_l905_90572

theorem quadratic_roots_prime_distinct (a α β m : ℕ) (h1: α ≠ β) (h2: Nat.Prime α) (h3: Nat.Prime β) (h4: α + β = m / a) (h5: α * β = 1996 / a) :
    a = 2 := by
  sorry

end quadratic_roots_prime_distinct_l905_90572


namespace complement_intersection_l905_90586

def U : Set ℝ := Set.univ
def M : Set ℝ := {y | 0 ≤ y ∧ y ≤ 2}
def N : Set ℝ := {x | (x < -3) ∨ (x > 0)}

theorem complement_intersection :
  (Set.univ \ M) ∩ N = {x | x < -3 ∨ x > 2} :=
by
  sorry

end complement_intersection_l905_90586


namespace remainder_q_x_plus_2_l905_90545

def q (x : ℝ) (D E F : ℝ) : ℝ := D * x ^ 6 + E * x ^ 4 + F * x ^ 2 + 5

theorem remainder_q_x_plus_2 (D E F : ℝ) (h : q 2 D E F = 13) : q (-2) D E F = 13 :=
by
  sorry

end remainder_q_x_plus_2_l905_90545


namespace total_surface_area_correct_l905_90504

def six_cubes_surface_area : ℕ :=
  let cube_edge := 1
  let cubes := 6
  let initial_surface_area := 6 * cubes -- six faces per cube, total initial surface area
  let hidden_faces := 10 -- determined by counting connections
  initial_surface_area - hidden_faces

theorem total_surface_area_correct : six_cubes_surface_area = 26 := by
  sorry

end total_surface_area_correct_l905_90504


namespace number_of_lucky_tickets_l905_90566

def is_leningrad_lucky (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₁ + a₂ + a₃ = a₄ + a₅ + a₆

def is_moscow_lucky (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  a₂ + a₄ + a₆ = a₁ + a₃ + a₅

def is_symmetric (a₂ a₅ : ℕ) : Prop :=
  a₂ = a₅

def is_valid_ticket (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : Prop :=
  is_leningrad_lucky a₁ a₂ a₃ a₄ a₅ a₆ ∧
  is_moscow_lucky a₁ a₂ a₃ a₄ a₅ a₆ ∧
  is_symmetric a₂ a₅

theorem number_of_lucky_tickets : 
  ∃ n : ℕ, n = 6700 ∧ 
  (∀ a₁ a₂ a₃ a₄ a₅ a₆ : ℕ, 
    0 ≤ a₁ ∧ a₁ ≤ 9 ∧
    0 ≤ a₂ ∧ a₂ ≤ 9 ∧
    0 ≤ a₃ ∧ a₃ ≤ 9 ∧
    0 ≤ a₄ ∧ a₄ ≤ 9 ∧
    0 ≤ a₅ ∧ a₅ ≤ 9 ∧
    0 ≤ a₆ ∧ a₆ ≤ 9 →
    is_valid_ticket a₁ a₂ a₃ a₄ a₅ a₆ →
    n = 6700) := sorry

end number_of_lucky_tickets_l905_90566


namespace base_conversion_problem_l905_90503

theorem base_conversion_problem 
  (b x y z : ℕ)
  (h1 : 1987 = x * b^2 + y * b + z)
  (h2 : x + y + z = 25) :
  b = 19 ∧ x = 5 ∧ y = 9 ∧ z = 11 := 
by
  sorry

end base_conversion_problem_l905_90503


namespace gym_membership_total_cost_l905_90559

-- Definitions for the conditions stated in the problem
def first_gym_monthly_fee : ℕ := 10
def first_gym_signup_fee : ℕ := 50
def first_gym_discount_rate : ℕ := 10
def first_gym_personal_training_cost : ℕ := 25
def first_gym_sessions_per_year : ℕ := 52

def second_gym_multiplier : ℕ := 3
def second_gym_monthly_fee : ℕ := 3 * first_gym_monthly_fee
def second_gym_signup_fee_multiplier : ℕ := 4
def second_gym_discount_rate : ℕ := 10
def second_gym_personal_training_cost : ℕ := 45
def second_gym_sessions_per_year : ℕ := 52

-- Proof of the total amount John paid in the first year
theorem gym_membership_total_cost:
  let first_gym_annual_cost := (first_gym_monthly_fee * 12) +
                                (first_gym_signup_fee * (100 - first_gym_discount_rate) / 100) +
                                (first_gym_personal_training_cost * first_gym_sessions_per_year)
  let second_gym_annual_cost := (second_gym_monthly_fee * 12) +
                                (second_gym_monthly_fee * second_gym_signup_fee_multiplier * (100 - second_gym_discount_rate) / 100) +
                                (second_gym_personal_training_cost * second_gym_sessions_per_year)
  let total_annual_cost := first_gym_annual_cost + second_gym_annual_cost
  total_annual_cost = 4273 := by
  -- Declaration of the variables used in the problem
  let first_gym_annual_cost := 1465
  let second_gym_annual_cost := 2808
  let total_annual_cost := first_gym_annual_cost + second_gym_annual_cost
  -- Simplify and verify the total cost
  sorry

end gym_membership_total_cost_l905_90559


namespace total_books_l905_90563

def number_of_zoology_books : ℕ := 16
def number_of_botany_books : ℕ := 4 * number_of_zoology_books

theorem total_books : number_of_zoology_books + number_of_botany_books = 80 := by
  sorry

end total_books_l905_90563


namespace find_a_plus_b_l905_90549

theorem find_a_plus_b (x a b : ℝ) (ha : a > 0) (hb : b > 0) (h : x = a + Real.sqrt b) 
  (hx : x^2 + 3 * x + ↑(3) / x + 1 / x^2 = 30) : 
  a + b = 5 := 
sorry

end find_a_plus_b_l905_90549


namespace willie_gave_emily_7_stickers_l905_90520

theorem willie_gave_emily_7_stickers (initial_stickers : ℕ) (final_stickers : ℕ) (given_stickers : ℕ) 
  (h1 : initial_stickers = 36) (h2 : final_stickers = 29) (h3 : given_stickers = initial_stickers - final_stickers) : 
  given_stickers = 7 :=
by
  rw [h1, h2] at h3 -- Replace initial_stickers with 36 and final_stickers with 29 in h3
  exact h3  -- given_stickers = 36 - 29 which is equal to 7.


end willie_gave_emily_7_stickers_l905_90520


namespace square_perimeter_eq_16_l905_90540

theorem square_perimeter_eq_16 (s : ℕ) (h : s^2 = 4 * s) : 4 * s = 16 :=
by {
  sorry
}

end square_perimeter_eq_16_l905_90540


namespace solve_eq1_solve_eq2_l905_90527

variable (x : ℝ)

theorem solve_eq1 : (2 * x - 3 * (2 * x - 3) = x + 4) → (x = 1) :=
by
  intro h
  sorry

theorem solve_eq2 : ((3 / 4 * x - 1 / 4) - 1 = (5 / 6 * x - 7 / 6)) → (x = -1) :=
by
  intro h
  sorry

end solve_eq1_solve_eq2_l905_90527


namespace standard_equation_of_tangent_circle_l905_90583

theorem standard_equation_of_tangent_circle (r h k : ℝ)
  (h_r : r = 1) 
  (h_k : k = 1) 
  (h_center_quadrant : h > 0 ∧ k > 0)
  (h_tangent_x_axis : k = r) 
  (h_tangent_line : r = abs (4 * h - 3) / 5)
  : (x - 2)^2 + (y - 1)^2 = 1 := 
by {
  sorry
}

end standard_equation_of_tangent_circle_l905_90583


namespace dormitory_problem_l905_90516

theorem dormitory_problem (x : ℕ) :
  9 < x ∧ x < 12
  → (x = 10 ∧ 4 * x + 18 = 58)
  ∨ (x = 11 ∧ 4 * x + 18 = 62) :=
by
  intros h
  sorry

end dormitory_problem_l905_90516


namespace no_solutions_in_domain_l905_90588

-- Define the function g
def g (x : ℝ) : ℝ := -0.5 * x^2 + x + 3

-- Define the condition on the domain of g
def in_domain (x : ℝ) : Prop := x ≥ -3 ∧ x ≤ 3

-- State the theorem to be proved
theorem no_solutions_in_domain :
  ∀ x : ℝ, in_domain x → ¬ (g (g x) = 3) :=
by
  -- Provide a placeholder for the proof
  sorry

end no_solutions_in_domain_l905_90588


namespace compare_fractions_l905_90589

variable {a b : ℝ}

theorem compare_fractions (h1 : 3 * a > b) (h2 : b > 0) :
  (a / b) > ((a + 1) / (b + 3)) :=
by
  sorry

end compare_fractions_l905_90589


namespace quadratic_completion_l905_90560

theorem quadratic_completion (x : ℝ) :
  2 * x^2 + 3 * x + 1 = 0 ↔ 2 * (x + 3 / 4)^2 - 1 / 8 = 0 :=
by
  sorry

end quadratic_completion_l905_90560


namespace solve_for_x_l905_90522

theorem solve_for_x (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 :=
by
  -- Proof will go here
  sorry

end solve_for_x_l905_90522


namespace num_customers_did_not_tip_l905_90528

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

end num_customers_did_not_tip_l905_90528


namespace commission_rate_correct_l905_90538

variables (weekly_earnings : ℕ) (commission : ℕ) (total_earnings : ℕ) (sales : ℕ) (commission_rate : ℕ)

-- Base earnings per week without commission
def base_earnings : ℕ := 190

-- Total earnings target
def earnings_goal : ℕ := 500

-- Minimum sales required to meet the earnings goal
def sales_needed : ℕ := 7750

-- Definition of the commission as needed to meet the goal
def needed_commission : ℕ := earnings_goal - base_earnings

-- Definition of the actual commission rate
def commission_rate_per_sale : ℕ := (needed_commission * 100) / sales_needed

-- Proof goal: Show that commission_rate_per_sale is 4
theorem commission_rate_correct : commission_rate_per_sale = 4 :=
by
  sorry

end commission_rate_correct_l905_90538


namespace angle_B_in_parallelogram_l905_90543

variable (A B : ℝ)

theorem angle_B_in_parallelogram (h_parallelogram : ∀ {A B C D : ℝ}, A + B = 180 ↔ A = B) 
  (h_A : A = 50) : B = 130 := by
  sorry

end angle_B_in_parallelogram_l905_90543


namespace algebraic_expression_value_l905_90599

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 - x - 1 = 5) : 6 * x^2 - 3 * x - 9 = 9 := 
by 
  sorry

end algebraic_expression_value_l905_90599


namespace find_stream_speed_l905_90597

-- Define the conditions
def boat_speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def upstream_time : ℝ := 1.5
def speed_of_stream (v : ℝ) : Prop :=
  let downstream_speed := boat_speed_in_still_water + v
  let upstream_speed := boat_speed_in_still_water - v
  (downstream_speed * downstream_time) = (upstream_speed * upstream_time)

-- Define the theorem to prove
theorem find_stream_speed : ∃ v, speed_of_stream v ∧ v = 3 :=
by {
  sorry
}

end find_stream_speed_l905_90597


namespace find_a_l905_90505

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem find_a (a : ℝ) : f' a 1 = 6 → a = 1 :=
by
  intro h
  have h_f_prime : 3 * (1 : ℝ) ^ 2 + 2 * a * (1 : ℝ) + 1 = 6 := h
  sorry

end find_a_l905_90505


namespace order_of_6_with_respect_to_f_is_undefined_l905_90577

noncomputable def f (x : ℕ) : ℕ := x ^ 2 % 13

def order_of_6_undefined : Prop :=
  ∀ m : ℕ, m > 0 → f^[m] 6 ≠ 6

theorem order_of_6_with_respect_to_f_is_undefined : order_of_6_undefined :=
by
  sorry

end order_of_6_with_respect_to_f_is_undefined_l905_90577


namespace cleaning_time_together_l905_90562

theorem cleaning_time_together (lisa_time kay_time ben_time sarah_time : ℕ)
  (h_lisa : lisa_time = 8) (h_kay : kay_time = 12) 
  (h_ben : ben_time = 16) (h_sarah : sarah_time = 24) :
  1 / ((1 / (lisa_time:ℚ)) + (1 / (kay_time:ℚ)) + (1 / (ben_time:ℚ)) + (1 / (sarah_time:ℚ))) = (16 / 5 : ℚ) :=
by
  sorry

end cleaning_time_together_l905_90562


namespace tickets_sold_total_l905_90525

-- Define the conditions
variables (A : ℕ) (S : ℕ) (total_amount : ℝ := 222.50) (adult_ticket_price : ℝ := 4) (student_ticket_price : ℝ := 2.50)
variables (student_tickets_sold : ℕ := 9)

-- Define the total money equation and the question
theorem tickets_sold_total :
  4 * (A : ℝ) + 2.5 * (9 : ℝ) = 222.50 → A + 9 = 59 :=
by sorry

end tickets_sold_total_l905_90525


namespace part1_part2_l905_90507

open Set

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1 / a) * x + 1

theorem part1 (x : ℝ) : f 2 (2^x) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

theorem part2 (a x : ℝ) (h : a > 2) : f a x ≥ 0 ↔ x ∈ (Iic (1/a) ∪ Ici a) :=
by sorry

end part1_part2_l905_90507


namespace count_valid_words_l905_90534

def total_words (n : ℕ) : ℕ := 25 ^ n

def words_with_no_A (n : ℕ) : ℕ := 24 ^ n

def words_with_one_A (n : ℕ) : ℕ := n * 24 ^ (n - 1)

def words_with_less_than_two_As : ℕ :=
  (words_with_no_A 2) + (2 * 24) +
  (words_with_no_A 3) + (3 * 24 ^ 2) +
  (words_with_no_A 4) + (4 * 24 ^ 3) +
  (words_with_no_A 5) + (5 * 24 ^ 4)

def valid_words : ℕ :=
  (total_words 1 + total_words 2 + total_words 3 + total_words 4 + total_words 5) -
  words_with_less_than_two_As

theorem count_valid_words : valid_words = sorry :=
by sorry

end count_valid_words_l905_90534


namespace best_coupon1_price_l905_90550

theorem best_coupon1_price (x : ℝ) 
    (h1 : 60 ≤ x ∨ x = 60)
    (h2_1 : 25 < 0.12 * x) 
    (h2_2 : 0.12 * x > 0.2 * x - 30) :
    x = 209.95 ∨ x = 229.95 ∨ x = 249.95 :=
by sorry

end best_coupon1_price_l905_90550


namespace player_A_always_wins_l905_90532

theorem player_A_always_wins (a b c : ℤ) :
  ∃ (x1 x2 x3 : ℤ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (x - x1) * (x - x2) * (x - x3) = x^3 + a*x^2 + b*x + c :=
sorry

end player_A_always_wins_l905_90532


namespace part_I_part_II_l905_90580

-- Let the volume V of the tetrahedron ABCD be given
def V : ℝ := sorry

-- Areas of the faces opposite vertices A, B, C, D
def S_A : ℝ := sorry
def S_B : ℝ := sorry
def S_C : ℝ := sorry
def S_D : ℝ := sorry

-- Definitions of the edge lengths and angles
def a : ℝ := sorry -- BC
def a' : ℝ := sorry -- DA
def b : ℝ := sorry -- CA
def b' : ℝ := sorry -- DB
def c : ℝ := sorry -- AB
def c' : ℝ := sorry -- DC
def alpha : ℝ := sorry -- Angle between BC and DA
def beta : ℝ := sorry -- Angle between CA and DB
def gamma : ℝ := sorry -- Angle between AB and DC

theorem part_I : 
  S_A^2 + S_B^2 + S_C^2 + S_D^2 = 
  (1 / 4) * ((a * a' * Real.sin alpha)^2 + (b * b' * Real.sin beta)^2 + (c * c' * Real.sin gamma)^2) := 
  sorry

theorem part_II : 
  S_A^2 + S_B^2 + S_C^2 + S_D^2 ≥ 9 * (3 * V^4)^(1/3) :=
  sorry

end part_I_part_II_l905_90580


namespace sqrt_sum_eq_five_sqrt_three_l905_90541

theorem sqrt_sum_eq_five_sqrt_three : Real.sqrt 12 + Real.sqrt 27 = 5 * Real.sqrt 3 := by
  sorry

end sqrt_sum_eq_five_sqrt_three_l905_90541


namespace bryden_receives_10_dollars_l905_90510

theorem bryden_receives_10_dollars 
  (collector_rate : ℝ := 5)
  (num_quarters : ℝ := 4)
  (face_value_per_quarter : ℝ := 0.50) :
  collector_rate * num_quarters * face_value_per_quarter = 10 :=
by
  sorry

end bryden_receives_10_dollars_l905_90510


namespace g_neg_one_l905_90539

variables (f : ℝ → ℝ) (g : ℝ → ℝ)
variables (h₀ : ∀ x : ℝ, f (-x) + x^2 = -(f x + x^2))
variables (h₁ : f 1 = 1)
variables (h₂ : ∀ x : ℝ, g x = f x + 2)

theorem g_neg_one : g (-1) = -1 :=
by
  sorry

end g_neg_one_l905_90539


namespace sophie_perceived_height_in_mirror_l905_90573

noncomputable def inch_to_cm : ℝ := 2.5

noncomputable def sophie_height_in_inches : ℝ := 50

noncomputable def sophie_height_in_cm := sophie_height_in_inches * inch_to_cm

noncomputable def perceived_height := sophie_height_in_cm * 2

theorem sophie_perceived_height_in_mirror : perceived_height = 250 :=
by
  unfold perceived_height
  unfold sophie_height_in_cm
  unfold sophie_height_in_inches
  unfold inch_to_cm
  sorry

end sophie_perceived_height_in_mirror_l905_90573


namespace total_selling_price_16800_l905_90514

noncomputable def total_selling_price (CP_per_toy : ℕ) : ℕ :=
  let CP_18 := 18 * CP_per_toy
  let Gain := 3 * CP_per_toy
  CP_18 + Gain

theorem total_selling_price_16800 :
  total_selling_price 800 = 16800 :=
by
  sorry

end total_selling_price_16800_l905_90514


namespace julia_total_watches_l905_90515

namespace JuliaWatches

-- Given conditions
def silver_watches : ℕ := 20
def bronze_watches : ℕ := 3 * silver_watches
def platinum_watches : ℕ := 2 * bronze_watches
def gold_watches : ℕ := (20 * (silver_watches + platinum_watches)) / 100  -- 20 is 20% and division by 100 to get the percentage

-- Proving the total watches Julia owns after the purchase
theorem julia_total_watches : silver_watches + bronze_watches + platinum_watches + gold_watches = 228 := by
  sorry

end JuliaWatches

end julia_total_watches_l905_90515


namespace digits_sum_is_31_l905_90555

noncomputable def digits_sum_proof (A B C D E F G : ℕ) : Prop :=
  (1000 * A + 100 * B + 10 * C + D + 100 * E + 10 * F + G = 2020) ∧ 
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧ (A ≠ G) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧ (B ≠ G) ∧
  (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧ (C ≠ G) ∧
  (D ≠ E) ∧ (D ≠ F) ∧ (D ≠ G) ∧
  (E ≠ F) ∧ (E ≠ G) ∧
  (F ≠ G)

theorem digits_sum_is_31 (A B C D E F G : ℕ) (h : digits_sum_proof A B C D E F G) : 
  A + B + C + D + E + F + G = 31 :=
sorry

end digits_sum_is_31_l905_90555


namespace trains_cross_time_l905_90552

noncomputable def time_to_cross : ℝ := 
  let length_train1 := 110 -- length of the first train in meters
  let length_train2 := 150 -- length of the second train in meters
  let speed_train1 := 60 * 1000 / 3600 -- speed of the first train in meters per second
  let speed_train2 := 45 * 1000 / 3600 -- speed of the second train in meters per second
  let bridge_length := 340 -- length of the bridge in meters
  let total_distance := length_train1 + length_train2 + bridge_length -- total distance to be covered
  let relative_speed := speed_train1 + speed_train2 -- relative speed in meters per second
  total_distance / relative_speed

theorem trains_cross_time :
  abs (time_to_cross - 20.57) < 0.01 :=
sorry

end trains_cross_time_l905_90552


namespace length_of_AB_l905_90524

/-- A triangle ABC lies between two parallel lines where AC = 5 cm. Prove that AB = 10 cm. -/
noncomputable def triangle_is_between_two_parallel_lines : Prop := sorry

noncomputable def segmentAC : ℝ := 5

theorem length_of_AB :
  ∃ (AB : ℝ), triangle_is_between_two_parallel_lines ∧ segmentAC = 5 ∧ AB = 10 :=
sorry

end length_of_AB_l905_90524


namespace radius_of_inscribed_circle_l905_90565

theorem radius_of_inscribed_circle (a b x : ℝ) (hx : 0 < x) 
  (h_side_length : a > 20) 
  (h_TM : a = x + 8) 
  (h_OM : b = x + 9) 
  (h_Pythagorean : (a - 8)^2 + (b - 9)^2 = x^2) :
  x = 29 :=
by
  -- Assume all conditions and continue to the proof part.
  sorry

end radius_of_inscribed_circle_l905_90565


namespace induction_example_l905_90531

theorem induction_example (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 :=
sorry

end induction_example_l905_90531


namespace total_sum_spent_l905_90536

theorem total_sum_spent (b gift : ℝ) (friends tanya : ℕ) (extra_payment : ℝ)
  (h1 : friends = 10)
  (h2 : tanya = 1)
  (h3 : extra_payment = 3)
  (h4 : gift = 15)
  (h5 : b = 270)
  : (b + gift) = 285 :=
by {
  -- Given:
  -- friends = 10 (number of dinner friends),
  -- tanya = 1 (Tanya who forgot to pay),
  -- extra_payment = 3 (extra payment by each of the remaining 9 friends),
  -- gift = 15 (cost of the gift),
  -- b = 270 (total bill for the dinner excluding the gift),

  -- We need to prove:
  -- total sum spent by the group is $285, i.e., (b + gift) = 285

  sorry 
}

end total_sum_spent_l905_90536


namespace distinct_solutions_for_quadratic_l905_90593

theorem distinct_solutions_for_quadratic (n : ℕ) : ∃ (xs : Finset ℤ), xs.card = n ∧ ∀ x ∈ xs, ∃ y : ℤ, x^2 + 2^(n + 1) = y^2 :=
by sorry

end distinct_solutions_for_quadratic_l905_90593


namespace electric_blankets_sold_l905_90523

theorem electric_blankets_sold (T H E : ℕ)
  (h1 : 2 * T + 6 * H + 10 * E = 1800)
  (h2 : T = 7 * H)
  (h3 : H = 2 * E) : 
  E = 36 :=
by {
  sorry
}

end electric_blankets_sold_l905_90523


namespace maddie_watched_8_episodes_l905_90500

def minutes_per_episode : ℕ := 44
def minutes_monday : ℕ := 138
def minutes_tuesday_wednesday : ℕ := 0
def minutes_thursday : ℕ := 21
def episodes_friday : ℕ := 2
def minutes_per_episode_friday := episodes_friday * minutes_per_episode
def minutes_weekend : ℕ := 105
def total_minutes := minutes_monday + minutes_tuesday_wednesday + minutes_thursday + minutes_per_episode_friday + minutes_weekend
def answer := total_minutes / minutes_per_episode

theorem maddie_watched_8_episodes : answer = 8 := by
  sorry

end maddie_watched_8_episodes_l905_90500


namespace maximum_value_of_transformed_function_l905_90535

theorem maximum_value_of_transformed_function (a b : ℝ) (h_max : ∀ x : ℝ, a * (Real.cos x) + b ≤ 1)
  (h_min : ∀ x : ℝ, a * (Real.cos x) + b ≥ -7) : 
  ∃ ab : ℝ, (ab = 3 + a * b * (Real.sin x)) ∧ (∀ x : ℝ, ab ≤ 15) :=
by
  sorry

end maximum_value_of_transformed_function_l905_90535


namespace initial_men_in_camp_l905_90576

theorem initial_men_in_camp (days_initial men_initial : ℕ) (days_plus_thirty men_plus_thirty : ℕ)
(h1 : days_initial = 20)
(h2 : men_plus_thirty = men_initial + 30)
(h3 : days_plus_thirty = 5)
(h4 : (men_initial * days_initial) = (men_plus_thirty * days_plus_thirty)) :
  men_initial = 10 :=
by sorry

end initial_men_in_camp_l905_90576


namespace seating_arrangements_exactly_two_adjacent_empty_l905_90533

theorem seating_arrangements_exactly_two_adjacent_empty :
  let seats := 6
  let people := 3
  let arrangements := (seats.factorial / (seats - people).factorial)
  let non_adj_non_empty := ((seats - people).choose people * people.factorial)
  let all_adj_empty := ((seats - (people + 1)).choose 1 * people.factorial)
  arrangements - non_adj_non_empty - all_adj_empty = 72 := by
  sorry

end seating_arrangements_exactly_two_adjacent_empty_l905_90533


namespace production_cost_per_performance_l905_90585

theorem production_cost_per_performance
  (overhead : ℕ)
  (revenue_per_performance : ℕ)
  (num_performances : ℕ)
  (production_cost : ℕ)
  (break_even : num_performances * revenue_per_performance = overhead + num_performances * production_cost) :
  production_cost = 7000 :=
by
  have : num_performances = 9 := by sorry
  have : revenue_per_performance = 16000 := by sorry
  have : overhead = 81000 := by sorry
  exact sorry

end production_cost_per_performance_l905_90585


namespace solve_equation_l905_90590

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by
sorry

end solve_equation_l905_90590


namespace find_m_plus_n_l905_90544

theorem find_m_plus_n
  (m n : ℝ)
  (l1 : ∀ x y : ℝ, 2 * x + m * y + 2 = 0)
  (l2 : ∀ x y : ℝ, 2 * x + y - 1 = 0)
  (l3 : ∀ x y : ℝ, x + n * y + 1 = 0)
  (parallel_l1_l2 : ∀ x y : ℝ, (2 * x + m * y + 2 = 0) → (2 * x + y - 1 = 0))
  (perpendicular_l1_l3 : ∀ x y : ℝ, (2 * x + m * y + 2 = 0) ∧ (x + n * y + 1 = 0) → true) :
  m + n = -1 :=
by
  sorry

end find_m_plus_n_l905_90544


namespace inconsistent_fractions_l905_90553

theorem inconsistent_fractions : (3 / 5 : ℚ) + (17 / 20 : ℚ) > 1 := by
  sorry

end inconsistent_fractions_l905_90553


namespace value_of_a_minus_b_l905_90581

theorem value_of_a_minus_b (a b : ℝ) (h1 : (a + b)^2 = 49) (h2 : ab = 6) : a - b = 5 ∨ a - b = -5 := 
by
  sorry

end value_of_a_minus_b_l905_90581


namespace systematic_sampling_40th_number_l905_90591

theorem systematic_sampling_40th_number
  (total_students sample_size : ℕ)
  (first_group_start first_group_end selected_first_group_number steps : ℕ)
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_group_start = 1)
  (h4 : first_group_end = 20)
  (h5 : selected_first_group_number = 15)
  (h6 : steps = total_students / sample_size)
  (h7 : first_group_end - first_group_start + 1 = steps)
  : (selected_first_group_number + steps * (40 - 1)) = 795 :=
sorry

end systematic_sampling_40th_number_l905_90591


namespace certain_number_is_negative_425_l905_90595

theorem certain_number_is_negative_425 (x : ℝ) :
  (3 - (1/5) * x = 88) ∧ (4 - (1/7) * 210 = -26) → x = -425 :=
by
  sorry

end certain_number_is_negative_425_l905_90595


namespace find_B_l905_90592

variable (A B : Set ℤ)
variable (U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 6})

theorem find_B (hU : U = {x | 0 ≤ x ∧ x ≤ 6})
               (hA_complement_B : A ∩ (U \ B) = {1, 3, 5}) :
  B = {0, 2, 4, 6} :=
sorry

end find_B_l905_90592


namespace inequality_proof_l905_90584

theorem inequality_proof (a b c d : ℝ) (hnonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) (hsum : a + b + c + d = 1) :
  abcd + bcda + cdab + dabc ≤ 1/27 + (176/27) * abcd :=
by
  sorry

end inequality_proof_l905_90584


namespace B_spends_85_percent_salary_l905_90575

theorem B_spends_85_percent_salary (A_s B_s : ℝ) (A_savings : ℝ) :
  A_s + B_s = 2000 →
  A_s = 1500 →
  A_savings = 0.05 * A_s →
  (B_s - (B_s * (1 - 0.05))) = A_savings →
  (1 - 0.85) * B_s = 0.15 * B_s := 
by
  intros h1 h2 h3 h4
  sorry

end B_spends_85_percent_salary_l905_90575


namespace Kylie_uses_3_towels_in_one_month_l905_90570

-- Define the necessary variables and conditions
variable (daughters_towels : Nat) (husband_towels : Nat) (loads : Nat) (towels_per_load : Nat)
variable (K : Nat) -- number of bath towels Kylie uses

-- Given conditions
axiom h1 : daughters_towels = 6
axiom h2 : husband_towels = 3
axiom h3 : loads = 3
axiom h4 : towels_per_load = 4
axiom h5 : (K + daughters_towels + husband_towels) = (loads * towels_per_load)

-- Prove that K = 3
theorem Kylie_uses_3_towels_in_one_month : K = 3 :=
by
  sorry

end Kylie_uses_3_towels_in_one_month_l905_90570


namespace num_cows_correct_l905_90574

-- Definitions from the problem's conditions
def total_animals : ℕ := 500
def percentage_chickens : ℤ := 10
def remaining_animals := total_animals - (percentage_chickens * total_animals / 100)
def goats (cows: ℕ) : ℕ := 2 * cows

-- Statement to prove
theorem num_cows_correct : ∃ cows, remaining_animals = cows + goats cows ∧ 3 * cows = 450 :=
by
  sorry

end num_cows_correct_l905_90574


namespace top_leftmost_rectangle_is_B_l905_90512

-- Define the sides of the rectangles
structure Rectangle :=
  (w : ℕ)
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)

-- Define the specific rectangles with their side values
noncomputable def rectA : Rectangle := ⟨2, 7, 4, 7⟩
noncomputable def rectB : Rectangle := ⟨0, 6, 8, 5⟩
noncomputable def rectC : Rectangle := ⟨6, 3, 1, 1⟩
noncomputable def rectD : Rectangle := ⟨8, 4, 0, 2⟩
noncomputable def rectE : Rectangle := ⟨5, 9, 3, 6⟩
noncomputable def rectF : Rectangle := ⟨7, 5, 9, 0⟩

-- Prove that Rectangle B is the top leftmost rectangle
theorem top_leftmost_rectangle_is_B :
  (rectB.w = 0 ∧ rectB.x = 6 ∧ rectB.y = 8 ∧ rectB.z = 5) :=
by {
  sorry
}

end top_leftmost_rectangle_is_B_l905_90512


namespace maximum_rectangle_area_l905_90501

-- Define the perimeter condition
def perimeter (rectangle : ℝ × ℝ) : ℝ :=
  2 * rectangle.fst + 2 * rectangle.snd

-- Define the area function
def area (rectangle : ℝ × ℝ) : ℝ :=
  rectangle.fst * rectangle.snd

-- Define the question statement in terms of Lean
theorem maximum_rectangle_area (length_width : ℝ × ℝ) (h : perimeter length_width = 32) : 
  area length_width ≤ 64 :=
sorry

end maximum_rectangle_area_l905_90501


namespace number_of_students_not_enrolled_in_biology_l905_90571

noncomputable def total_students : ℕ := 880

noncomputable def biology_enrollment_percent : ℕ := 40

noncomputable def students_not_enrolled_in_biology : ℕ :=
  (100 - biology_enrollment_percent) * total_students / 100

theorem number_of_students_not_enrolled_in_biology :
  students_not_enrolled_in_biology = 528 :=
by
  -- Proof goes here.
  -- Use sorry to skip the proof for this placeholder:
  sorry

end number_of_students_not_enrolled_in_biology_l905_90571


namespace min_rectangle_perimeter_l905_90517

theorem min_rectangle_perimeter (x y : ℤ) (h1 : x * y = 50) (hx : 0 < x) (hy : 0 < y) : 
  (∀ x y, x * y = 50 → 2 * (x + y) ≥ 30) ∧ 
  ∃ x y, x * y = 50 ∧ 2 * (x + y) = 30 := 
by sorry

end min_rectangle_perimeter_l905_90517


namespace proof_problem_l905_90519

def g : ℕ → ℕ := sorry
def g_inv : ℕ → ℕ := sorry

axiom g_inv_is_inverse : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y
axiom g_4_eq_6 : g 4 = 6
axiom g_6_eq_2 : g 6 = 2
axiom g_3_eq_7 : g 3 = 7

theorem proof_problem :
  g_inv (g_inv 7 + g_inv 6) = 3 :=
by
  sorry

end proof_problem_l905_90519


namespace odd_function_inequality_solution_l905_90557

noncomputable def f (x : ℝ) : ℝ := if x > 0 then x - 2 else -(x - 2)

theorem odd_function_inequality_solution :
  {x : ℝ | f x < 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by
  -- A placeholder for the actual proof
  sorry

end odd_function_inequality_solution_l905_90557


namespace cubic_roots_expression_l905_90526

noncomputable def polynomial : Polynomial ℂ :=
  Polynomial.X^3 - 3 * Polynomial.X - 2

theorem cubic_roots_expression (α β γ : ℂ)
  (h1 : (Polynomial.X - Polynomial.C α) * 
        (Polynomial.X - Polynomial.C β) * 
        (Polynomial.X - Polynomial.C γ) = polynomial) :
  α * (β - γ)^2 + β * (γ - α)^2 + γ * (α - β)^2 = -18 :=
by
  sorry

end cubic_roots_expression_l905_90526


namespace least_possible_square_area_l905_90537

theorem least_possible_square_area (measured_length : ℝ) (h : measured_length = 7) : 
  ∃ (actual_length : ℝ), 6.5 ≤ actual_length ∧ actual_length < 7.5 ∧ 
  (∀ (side : ℝ), 6.5 ≤ side ∧ side < 7.5 → side * side ≥ actual_length * actual_length) ∧ 
  actual_length * actual_length = 42.25 :=
by
  sorry

end least_possible_square_area_l905_90537


namespace smaller_square_area_percentage_is_zero_l905_90502

noncomputable def area_smaller_square_percentage (r : ℝ) : ℝ :=
  let side_length_larger_square := 2 * r
  let x := 0  -- Solution from the Pythagorean step
  let area_larger_square := side_length_larger_square ^ 2
  let area_smaller_square := x ^ 2
  100 * area_smaller_square / area_larger_square

theorem smaller_square_area_percentage_is_zero (r : ℝ) :
    area_smaller_square_percentage r = 0 :=
  sorry

end smaller_square_area_percentage_is_zero_l905_90502


namespace icosahedron_probability_div_by_three_at_least_one_fourth_l905_90548
open ProbabilityTheory

theorem icosahedron_probability_div_by_three_at_least_one_fourth (a b c : ℕ) (h : a + b + c = 20) :
  (a^3 + b^3 + c^3 + 6 * a * b * c : ℚ) / (a + b + c)^3 ≥ 1 / 4 :=
sorry

end icosahedron_probability_div_by_three_at_least_one_fourth_l905_90548


namespace simplify_complex_fraction_l905_90547

theorem simplify_complex_fraction :
  let numerator := (5 : ℂ) + 7 * I
  let denominator := (2 : ℂ) + 3 * I
  numerator / denominator = (31 / 13 : ℂ) - (1 / 13) * I :=
by
  let numerator := (5 : ℂ) + 7 * I
  let denominator := (2 : ℂ) + 3 * I
  sorry

end simplify_complex_fraction_l905_90547


namespace number_of_truthful_dwarfs_l905_90564

def num_dwarfs : Nat := 10

def likes_vanilla : Nat := num_dwarfs

def likes_chocolate : Nat := num_dwarfs / 2

def likes_fruit : Nat := 1

theorem number_of_truthful_dwarfs : 
  ∃ t l : Nat, 
  t + l = num_dwarfs ∧  -- total number of dwarfs
  t + 2 * l = likes_vanilla + likes_chocolate + likes_fruit ∧  -- total number of hand raises
  t = 4 :=  -- number of truthful dwarfs
  sorry

end number_of_truthful_dwarfs_l905_90564


namespace intersection_of_sets_l905_90556

def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | 2 < x ∧ x < 4 }

theorem intersection_of_sets : A ∩ B = { x | 2 < x ∧ x ≤ 3 } := 
by 
  sorry

end intersection_of_sets_l905_90556


namespace sara_initial_savings_l905_90587

-- Given conditions as definitions
def save_rate_sara : ℕ := 10
def save_rate_jim : ℕ := 15
def weeks : ℕ := 820

-- Prove that the initial savings of Sara is 4100 dollars given the conditions
theorem sara_initial_savings : 
  ∃ S : ℕ, S + save_rate_sara * weeks = save_rate_jim * weeks → S = 4100 := 
sorry

end sara_initial_savings_l905_90587


namespace range_of_m_l905_90546

theorem range_of_m (m x : ℝ) (h₁ : (x / (x - 3) - 2 = m / (x - 3))) (h₂ : x ≠ 3) : x > 0 ↔ m < 6 ∧ m ≠ 3 :=
by
  sorry

end range_of_m_l905_90546


namespace number_of_C_animals_l905_90579

-- Define the conditions
def A : ℕ := 45
def B : ℕ := 32
def C : ℕ := 5

-- Define the theorem that we need to prove
theorem number_of_C_animals : B + C = A - 8 :=
by
  -- placeholder to complete the proof (not part of the problem's requirement)
  sorry

end number_of_C_animals_l905_90579


namespace circle_intersection_l905_90582

theorem circle_intersection (a : ℝ) :
  ((-3 * Real.sqrt 2 / 2 < a ∧ a < -Real.sqrt 2 / 2) ∨ (Real.sqrt 2 / 2 < a ∧ a < 3 * Real.sqrt 2 / 2)) ↔
  (∃ x y : ℝ, (x - a)^2 + (y - a)^2 = 4 ∧ x^2 + y^2 = 1) :=
sorry

end circle_intersection_l905_90582


namespace probability_closer_to_6_than_0_is_0_6_l905_90578

noncomputable def probability_closer_to_6_than_0 : ℝ :=
  let total_length := 7
  let segment_length_closer_to_6 := 4
  let probability := (segment_length_closer_to_6 : ℝ) / total_length
  probability

theorem probability_closer_to_6_than_0_is_0_6 :
  probability_closer_to_6_than_0 = 0.6 := by
  sorry

end probability_closer_to_6_than_0_is_0_6_l905_90578


namespace minimum_area_for_rectangle_l905_90569

theorem minimum_area_for_rectangle 
(length width : ℝ) 
(h_length_min : length = 4 - 0.5) 
(h_width_min : width = 5 - 1) :
length * width = 14 := 
by 
  simp [h_length_min, h_width_min]
  sorry

end minimum_area_for_rectangle_l905_90569


namespace donny_money_left_l905_90518

-- Definitions based on Conditions
def initial_amount : ℝ := 78
def cost_kite : ℝ := 8
def cost_frisbee : ℝ := 9

-- Discounted cost of roller skates
def original_cost_roller_skates : ℝ := 15
def discount_rate_roller_skates : ℝ := 0.10
def discounted_cost_roller_skates : ℝ :=
  original_cost_roller_skates * (1 - discount_rate_roller_skates)

-- Cost of LEGO set with coupon
def original_cost_lego_set : ℝ := 25
def coupon_lego_set : ℝ := 5
def discounted_cost_lego_set : ℝ :=
  original_cost_lego_set - coupon_lego_set

-- Cost of puzzle with tax
def original_cost_puzzle : ℝ := 12
def tax_rate_puzzle : ℝ := 0.05
def taxed_cost_puzzle : ℝ :=
  original_cost_puzzle * (1 + tax_rate_puzzle)

-- Total cost calculated from item costs
def total_cost : ℝ :=
  cost_kite + cost_frisbee + discounted_cost_roller_skates + discounted_cost_lego_set + taxed_cost_puzzle

def money_left_after_shopping : ℝ :=
  initial_amount - total_cost

-- Prove the main statement
theorem donny_money_left : money_left_after_shopping = 14.90 := by
  sorry

end donny_money_left_l905_90518


namespace inequality_range_l905_90530

theorem inequality_range (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 3| > a) → a < 5 :=
  sorry

end inequality_range_l905_90530


namespace exterior_angle_hexagon_l905_90568

theorem exterior_angle_hexagon (θ : ℝ) (hθ : θ = 60) (h_sum : θ * 6 = 360) : n = 6 :=
sorry

end exterior_angle_hexagon_l905_90568


namespace filling_time_with_ab_l905_90511

theorem filling_time_with_ab (a b c l : ℝ) (h1 : a + b + c - l = 5 / 6) (h2 : a + c - l = 1 / 2) (h3 : b + c - l = 1 / 3) : 
  1 / (a + b) = 1.2 :=
by
  sorry

end filling_time_with_ab_l905_90511


namespace average_marks_of_all_students_l905_90594

theorem average_marks_of_all_students :
  (22 * 40 + 28 * 60) / (22 + 28) = 51.2 :=
by
  sorry

end average_marks_of_all_students_l905_90594


namespace find_value_of_reciprocal_cubic_sum_l905_90551

theorem find_value_of_reciprocal_cubic_sum
  (a b c r s : ℝ)
  (h₁ : a + b + c = 0)
  (h₂ : a ≠ 0)
  (h₃ : b^2 - 4 * a * c ≥ 0)
  (h₄ : r ≠ 0)
  (h₅ : s ≠ 0)
  (h₆ : a * r^2 + b * r + c = 0)
  (h₇ : a * s^2 + b * s + c = 0)
  (h₈ : r + s = -b / a)
  (h₉ : r * s = -c / a) :
  1 / r^3 + 1 / s^3 = -b * (b^2 + 3 * a^2 + 3 * a * b) / (a + b)^3 :=
by
  sorry

end find_value_of_reciprocal_cubic_sum_l905_90551


namespace carolyn_marbles_l905_90567

theorem carolyn_marbles (initial_marbles : ℕ) (shared_items : ℕ) (end_marbles: ℕ) : 
  initial_marbles = 47 → shared_items = 42 → end_marbles = initial_marbles - shared_items → end_marbles = 5 :=
by
  intros h₀ h₁ h₂
  rw [h₀, h₁] at h₂
  exact h₂

end carolyn_marbles_l905_90567


namespace minimum_value_of_expression_l905_90529

theorem minimum_value_of_expression (x y z : ℝ) (h : 2 * x - 3 * y + z = 3) :
  ∃ (x y z : ℝ), (x^2 + (y - 1)^2 + z^2) = 18 / 7 ∧ y = -2 / 7 :=
sorry

end minimum_value_of_expression_l905_90529
