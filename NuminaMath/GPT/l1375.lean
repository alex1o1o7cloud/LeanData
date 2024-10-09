import Mathlib

namespace more_candidates_selected_l1375_137500

theorem more_candidates_selected (total_a total_b selected_a selected_b : ℕ)
  (h1 : total_a = 8000)
  (h2 : total_b = 8000)
  (h3 : selected_a = 6 * total_a / 100)
  (h4 : selected_b = 7 * total_b / 100) :
  selected_b - selected_a = 80 :=
  sorry

end more_candidates_selected_l1375_137500


namespace find_a_l1375_137545

theorem find_a (a b x : ℝ) (h1 : a ≠ b)
  (h2 : a^3 + b^3 = 35 * x^3)
  (h3 : a^2 - b^2 = 4 * x^2) : a = 2 * x ∨ a = -2 * x :=
by
  sorry

end find_a_l1375_137545


namespace option_b_option_c_option_d_l1375_137542

theorem option_b (x : ℝ) (h : x > 1) : (∀ y, y = 2*x + 4 / (x - 1) - 1 → y ≥ 4*Real.sqrt 2 + 1) :=
by
  sorry

theorem option_c (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3 * x * y) : 2*x + y ≥ 3 :=
by
  sorry

theorem option_d (x y : ℝ) (h : 9*x^2 + y^2 + x*y = 1) : 3*x + y ≤ 2*Real.sqrt 21 / 7 :=
by
  sorry

end option_b_option_c_option_d_l1375_137542


namespace polynomial_roots_l1375_137595

theorem polynomial_roots :
  (∀ x : ℝ, (x^3 - 2*x^2 - 5*x + 6 = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3)) :=
by
  sorry

end polynomial_roots_l1375_137595


namespace least_f_e_l1375_137547

theorem least_f_e (e : ℝ) (he : e > 0) : 
  ∃ f, (∀ (a b c d : ℝ), a^3 + b^3 + c^3 + d^3 ≤ e^2 * (a^2 + b^2 + c^2 + d^2) + f * (a^4 + b^4 + c^4 + d^4)) ∧ f = 1 / (4 * e^2) :=
sorry

end least_f_e_l1375_137547


namespace luke_bought_stickers_l1375_137586

theorem luke_bought_stickers :
  ∀ (original birthday given_to_sister used_on_card left total_before_buying stickers_bought : ℕ),
  original = 20 →
  birthday = 20 →
  given_to_sister = 5 →
  used_on_card = 8 →
  left = 39 →
  total_before_buying = original + birthday →
  stickers_bought = (left + given_to_sister + used_on_card) - total_before_buying →
  stickers_bought = 12 :=
by
  intros
  sorry

end luke_bought_stickers_l1375_137586


namespace no_even_integers_of_form_3k_plus_4_and_5m_plus_2_l1375_137565

theorem no_even_integers_of_form_3k_plus_4_and_5m_plus_2 (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 1000) (h2 : ∃ k : ℕ, n = 3 * k + 4) (h3 : ∃ m : ℕ, n = 5 * m + 2) (h4 : n % 2 = 0) : false :=
sorry

end no_even_integers_of_form_3k_plus_4_and_5m_plus_2_l1375_137565


namespace at_least_one_heart_or_king_l1375_137521

-- Define the conditions
def total_cards := 52
def hearts := 13
def kings := 4
def king_of_hearts := 1
def cards_hearts_or_kings := hearts + kings - king_of_hearts

-- Calculate probabilities based on the above conditions
def probability_not_heart_or_king := 
  1 - (cards_hearts_or_kings / total_cards)

def probability_neither_heart_nor_king :=
  (probability_not_heart_or_king) ^ 2

def probability_at_least_one_heart_or_king :=
  1 - probability_neither_heart_nor_king

-- State the theorem to be proved
theorem at_least_one_heart_or_king : 
  probability_at_least_one_heart_or_king = (88 / 169) :=
by
  sorry

end at_least_one_heart_or_king_l1375_137521


namespace find_normal_price_l1375_137509

theorem find_normal_price (P : ℝ) (S : ℝ) (d1 d2 d3 : ℝ) : 
  (P * (1 - d1) * (1 - d2) * (1 - d3) = S) → S = 144 → d1 = 0.12 → d2 = 0.22 → d3 = 0.15 → P = 246.81 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_normal_price_l1375_137509


namespace palindrome_probability_divisible_by_7_l1375_137510

-- Define the conditions
def is_four_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 1001 * a + 110 * b

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

-- Define the proof problem
theorem palindrome_probability_divisible_by_7 : 
  (∃ (n : ℕ), is_four_digit_palindrome n ∧ is_divisible_by_7 n) →
  ∃ p : ℚ, p = 1/5 :=
sorry

end palindrome_probability_divisible_by_7_l1375_137510


namespace equation_parallel_equation_perpendicular_l1375_137501

variables {x y : ℝ}

def l1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) := 2 * x - 5 * y + 14 = 0
def l3 (x y : ℝ) := 2 * x - y + 7 = 0

theorem equation_parallel {x y : ℝ} (hx : l1 x y) (hy : l2 x y) : 2 * x - y + 6 = 0 :=
sorry

theorem equation_perpendicular {x y : ℝ} (hx : l1 x y) (hy : l2 x y) : x + 2 * y - 2 = 0 :=
sorry

end equation_parallel_equation_perpendicular_l1375_137501


namespace tank_capacity_percentage_l1375_137522

noncomputable def radius (C : ℝ) := C / (2 * Real.pi)
noncomputable def volume (r h : ℝ) := Real.pi * r^2 * h

theorem tank_capacity_percentage :
  let r_M := radius 8
  let r_B := radius 10
  let V_M := volume r_M 10
  let V_B := volume r_B 8
  (V_M / V_B * 100) = 80 :=
by
  sorry

end tank_capacity_percentage_l1375_137522


namespace correct_product_of_a_b_l1375_137567

theorem correct_product_of_a_b (a b : ℕ) (h1 : (a - (10 * (a / 10 % 10) + 1)) * b = 255)
                              (h2 : (a - (10 * (a / 100 % 10 * 10 + a % 10 - (a / 100 % 10 * 10 + 5 * 10)))) * b = 335) :
  a * b = 285 := sorry

end correct_product_of_a_b_l1375_137567


namespace probability_of_top_card_heart_l1375_137582

-- Define the total number of cards in the deck.
def total_cards : ℕ := 39

-- Define the number of hearts in the deck.
def hearts : ℕ := 13

-- Define the probability that the top card is a heart.
def probability_top_card_heart : ℚ := hearts / total_cards

-- State the theorem to prove.
theorem probability_of_top_card_heart : probability_top_card_heart = 1 / 3 :=
by
  sorry

end probability_of_top_card_heart_l1375_137582


namespace fraction_product_eq_l1375_137572

theorem fraction_product_eq : (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 2 / 7 := by
  sorry

end fraction_product_eq_l1375_137572


namespace sum_of_consecutive_integers_with_product_506_l1375_137534

theorem sum_of_consecutive_integers_with_product_506 :
  ∃ x : ℕ, (x * (x + 1) = 506) → (x + (x + 1) = 45) :=
by
  sorry

end sum_of_consecutive_integers_with_product_506_l1375_137534


namespace intersection_of_M_and_N_l1375_137590

-- Definitions from conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof problem statement
theorem intersection_of_M_and_N : M ∩ N = {2, 3} :=
sorry

end intersection_of_M_and_N_l1375_137590


namespace cubics_sum_l1375_137564

noncomputable def roots_cubic (a b c d p q r : ℝ) : Prop :=
  (p + q + r = b) ∧ (p*q + p*r + q*r = c) ∧ (p*q*r = d)

noncomputable def root_values (p q r : ℝ) : Prop :=
  p^3 = 2*p^2 - 3*p + 4 ∧
  q^3 = 2*q^2 - 3*q + 4 ∧
  r^3 = 2*r^2 - 3*r + 4

theorem cubics_sum (p q r : ℝ) (h1 : p + q + r = 2) (h2 : p*q + q*r + p*r = 3)  (h3 : p*q*r = 4)
  (h4 : root_values p q r) : p^3 + q^3 + r^3 = 2 :=
by
  sorry

end cubics_sum_l1375_137564


namespace max_area_of_rectangle_l1375_137585

theorem max_area_of_rectangle (x y : ℝ) (h1 : 2 * (x + y) = 40) : 
  (x * y) ≤ 100 :=
by
  sorry

end max_area_of_rectangle_l1375_137585


namespace algebraic_expression_value_l1375_137593

theorem algebraic_expression_value (x : ℝ) (h : 3 / (x^2 + x) - x^2 = 2 + x) :
  2 * x^2 + 2 * x = 2 :=
sorry

end algebraic_expression_value_l1375_137593


namespace needed_adjustment_l1375_137557

def price_adjustment (P : ℝ) : ℝ :=
  let P_reduced := P - 0.20 * P
  let P_raised := P_reduced + 0.10 * P_reduced
  let P_target := P - 0.10 * P
  P_target - P_raised

theorem needed_adjustment (P : ℝ) : price_adjustment P = 2 * (P / 100) := sorry

end needed_adjustment_l1375_137557


namespace average_greater_median_l1375_137551

theorem average_greater_median :
  let h : ℝ := 120
  let s1 : ℝ := 4
  let s2 : ℝ := 4
  let s3 : ℝ := 5
  let s4 : ℝ := 7
  let s5 : ℝ := 9
  let median : ℝ := (s3 + s4) / 2
  let average : ℝ := (h + s1 + s2 + s3 + s4 + s5) / 6
  average - median = 18.8333 := by
    sorry

end average_greater_median_l1375_137551


namespace inequality_solution_l1375_137543

theorem inequality_solution (x : ℝ) : x^2 + x - 12 ≤ 0 ↔ -4 ≤ x ∧ x ≤ 3 := sorry

end inequality_solution_l1375_137543


namespace real_number_representation_l1375_137517

theorem real_number_representation (x : ℝ) 
  (h₀ : 0 < x) (h₁ : x ≤ 1) :
  ∃ (n : ℕ → ℕ), (∀ k, n k > 0) ∧ (∀ k, n (k + 1) = n k * 2 ∨ n (k + 1) = n k * 3 ∨ n (k + 1) = n k * 4) ∧ 
  (x = ∑' k, 1 / (n k)) :=
sorry

end real_number_representation_l1375_137517


namespace share_ratio_l1375_137599

theorem share_ratio (A B C x : ℝ)
  (h1 : A = 280)
  (h2 : A + B + C = 700)
  (h3 : A = x * (B + C))
  (h4 : B = (6 / 9) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end share_ratio_l1375_137599


namespace linear_function_properties_l1375_137503

def linear_function (x : ℝ) : ℝ := -2 * x + 1

theorem linear_function_properties :
  (∀ x, linear_function x = -2 * x + 1) ∧
  (∀ x₁ x₂, x₁ < x₂ → linear_function x₁ > linear_function x₂) ∧
  (linear_function 0 = 1) ∧
  ((∃ x, x > 0 ∧ linear_function x > 0) ∧ (∃ x, x < 0 ∧ linear_function x > 0) ∧ (∃ x, x > 0 ∧ linear_function x < 0))
  :=
by
  sorry

end linear_function_properties_l1375_137503


namespace age_of_new_person_l1375_137502

theorem age_of_new_person (T : ℝ) (A : ℝ) (h : T / 20 - 4 = (T - 60 + A) / 20) : A = 40 :=
sorry

end age_of_new_person_l1375_137502


namespace range_of_function_l1375_137507

theorem range_of_function (y : ℝ) (t: ℝ) (x : ℝ) (h_t : t = x^2 - 1) (h_domain : t ∈ Set.Ici (-1)) :
  ∃ (y_set : Set ℝ), ∀ y ∈ y_set, y = (1/3)^t ∧ y_set = Set.Ioo 0 3 ∨ y_set = Set.Icc 0 3 := by
  sorry

end range_of_function_l1375_137507


namespace div_eq_implies_eq_l1375_137569

theorem div_eq_implies_eq (a b : ℕ) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b :=
sorry

end div_eq_implies_eq_l1375_137569


namespace total_photos_in_gallery_l1375_137583

def initial_photos : ℕ := 800
def photos_first_day : ℕ := (2 * initial_photos) / 3
def photos_second_day : ℕ := photos_first_day + 180

theorem total_photos_in_gallery : initial_photos + photos_first_day + photos_second_day = 2046 := by
  -- the proof can be provided here
  sorry

end total_photos_in_gallery_l1375_137583


namespace total_number_of_games_in_season_l1375_137575

def number_of_games_per_month : ℕ := 13
def number_of_months_in_season : ℕ := 14

theorem total_number_of_games_in_season :
  number_of_games_per_month * number_of_months_in_season = 182 := by
  sorry

end total_number_of_games_in_season_l1375_137575


namespace isosceles_triangle_angle_sum_l1375_137584

theorem isosceles_triangle_angle_sum (x : ℝ) (h1 : x = 50 ∨ x = 65 ∨ x = 80) : (50 + 65 + 80 = 195) :=
by sorry

end isosceles_triangle_angle_sum_l1375_137584


namespace find_x_l1375_137516

theorem find_x (x : ℝ) (h : 2 * x - 1 = -( -x + 5 )) : x = -6 :=
by
  sorry

end find_x_l1375_137516


namespace value_of_q_l1375_137588

-- Define the problem in Lean 4

variable (a d q : ℝ) (h0 : a ≠ 0)
variables (M P : Set ℝ)
variable (hM : M = {a, a + d, a + 2 * d})
variable (hP : P = {a, a * q, a * q * q})
variable (hMP : M = P)

theorem value_of_q : q = -1 :=
by
  sorry

end value_of_q_l1375_137588


namespace trigonometric_identity_l1375_137580

theorem trigonometric_identity :
  3 * Real.arcsin (Real.sqrt 3 / 2) - Real.arctan (-1) - Real.arccos 0 = (3 * Real.pi) / 4 := 
by
  sorry

end trigonometric_identity_l1375_137580


namespace remainder_of_sum_l1375_137553

theorem remainder_of_sum (c d : ℤ) (p q : ℤ) (h1 : c = 60 * p + 53) (h2 : d = 45 * q + 28) : 
  (c + d) % 15 = 6 := 
by
  sorry

end remainder_of_sum_l1375_137553


namespace complex_exponential_to_rectangular_form_l1375_137531

theorem complex_exponential_to_rectangular_form :
  Real.sqrt 2 * Complex.exp (13 * Real.pi * Complex.I / 4) = -1 - Complex.I := by
  -- Proof will go here
  sorry

end complex_exponential_to_rectangular_form_l1375_137531


namespace mixed_groups_count_l1375_137528

/-- Define the initial conditions --/
def number_of_children : ℕ := 300
def number_of_groups : ℕ := 100
def group_size : ℕ := 3
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56

/-- Define the proof problem -/
theorem mixed_groups_count : 
    (number_of_children = 300) →
    (number_of_groups = 100) →
    (group_size = 3) →
    (boy_boy_photos = 100) →
    (girl_girl_photos = 56) →
    (∀ total_photos, total_photos = number_of_groups * group_size) →
    (∃ mixed_groups, mixed_groups = (total_photos - boy_boy_photos - girl_girl_photos) / 2) →
    mixed_groups = 72 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end mixed_groups_count_l1375_137528


namespace equal_focal_distances_l1375_137591

def ellipse1 (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1
def ellipse2 (k x y : ℝ) (hk : k < 9) : Prop := x^2 / (25 - k) + y^2 / (9 - k) = 1

theorem equal_focal_distances (k : ℝ) (hk : k < 9) : 
  let f1 := 8
  let f2 := 8 
  f1 = f2 :=
by 
  sorry

end equal_focal_distances_l1375_137591


namespace find_multiple_l1375_137513

-- Definitions of the conditions
def is_positive (x : ℝ) : Prop := x > 0

-- Main statement
theorem find_multiple (x : ℝ) (h : is_positive x) (hx : x = 8) : ∃ k : ℝ, x + 8 = k * (1 / x) ∧ k = 128 :=
by
  use 128
  sorry

end find_multiple_l1375_137513


namespace minimal_sum_of_squares_l1375_137529

theorem minimal_sum_of_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∃ p q r : ℕ, a + b = p^2 ∧ b + c = q^2 ∧ a + c = r^2) ∧
  a + b + c = 55 := 
by sorry

end minimal_sum_of_squares_l1375_137529


namespace p_minus_q_l1375_137597

theorem p_minus_q (p q : ℚ) (h1 : 3 / p = 6) (h2 : 3 / q = 18) : p - q = 1 / 3 := by
  sorry

end p_minus_q_l1375_137597


namespace intersection_points_l1375_137506

variables {α β : Type*} [DecidableEq α] {f : α → β} {x m : α}

theorem intersection_points (dom : α → Prop) (h : dom x → ∃! y, f x = y) : 
  (∃ y, f m = y) ∨ ¬ ∃ y, f m = y :=
by
  sorry

end intersection_points_l1375_137506


namespace star_three_and_four_l1375_137514

def star (a b : ℝ) : ℝ := 4 * a + 5 * b - 2 * a * b

theorem star_three_and_four : star 3 4 = 8 :=
by
  sorry

end star_three_and_four_l1375_137514


namespace image_relative_velocity_l1375_137523

-- Definitions of the constants
def f : ℝ := 0.2
def x : ℝ := 0.5
def vt : ℝ := 3

-- Lens equation
def lens_equation (f x y : ℝ) : Prop :=
  (1 / x) + (1 / y) = 1 / f

-- Image distance
noncomputable def y (f x : ℝ) : ℝ :=
  1 / (1 / f - 1 / x)

-- Derivative of y with respect to x
noncomputable def dy_dx (f x : ℝ) : ℝ :=
  (f^2) / (x - f)^2

-- Image velocity
noncomputable def vk (vt dy_dx : ℝ) : ℝ :=
  vt * dy_dx

-- Relative velocity
noncomputable def v_rel (vt vk : ℝ) : ℝ :=
  vk - vt

-- Theorem to prove the relative velocity
theorem image_relative_velocity : v_rel vt (vk vt (dy_dx f x)) = -5 / 3 := 
by
  sorry

end image_relative_velocity_l1375_137523


namespace find_a_l1375_137549

noncomputable def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x * a = 1}
axiom A_is_B (a : ℝ) : A ∩ B a = B a → (a = 0) ∨ (a = 1/3) ∨ (a = 1/5)

-- statement to prove
theorem find_a (a : ℝ) (h : A ∩ B a = B a) : (a = 0) ∨ (a = 1/3) ∨ (a = 1/5) :=
by 
  apply A_is_B
  assumption

end find_a_l1375_137549


namespace sculptures_not_on_display_count_l1375_137554

noncomputable def total_art_pieces : ℕ := 1800
noncomputable def pieces_on_display : ℕ := total_art_pieces / 3
noncomputable def pieces_not_on_display : ℕ := total_art_pieces - pieces_on_display
noncomputable def sculptures_on_display : ℕ := pieces_on_display / 6
noncomputable def sculptures_not_on_display : ℕ := pieces_not_on_display * 2 / 3

theorem sculptures_not_on_display_count : sculptures_not_on_display = 800 :=
by {
  -- Since this is a statement only as requested, we use sorry to skip the proof
  sorry
}

end sculptures_not_on_display_count_l1375_137554


namespace milk_for_flour_l1375_137581

theorem milk_for_flour (milk flour use_flour : ℕ) (h1 : milk = 75) (h2 : flour = 300) (h3 : use_flour = 900) : (use_flour/flour * milk) = 225 :=
by sorry

end milk_for_flour_l1375_137581


namespace cost_of_each_book_l1375_137571

noncomputable def cost_of_book (money_given money_left notebook_cost notebook_count book_count : ℕ) : ℕ :=
  (money_given - money_left - (notebook_count * notebook_cost)) / book_count

-- Conditions
def money_given : ℕ := 56
def money_left : ℕ := 14
def notebook_cost : ℕ := 4
def notebook_count : ℕ := 7
def book_count : ℕ := 2

-- Theorem stating that the cost of each book is $7 under given conditions
theorem cost_of_each_book : cost_of_book money_given money_left notebook_cost notebook_count book_count = 7 := by
  sorry

end cost_of_each_book_l1375_137571


namespace intersection_of_A_and_B_l1375_137532

-- Define the sets A and B
def setA : Set ℝ := { x | -1 < x ∧ x ≤ 4 }
def setB : Set ℝ := { x | 2 < x ∧ x ≤ 5 }

-- The intersection of sets A and B
def intersectAB : Set ℝ := { x | 2 < x ∧ x ≤ 4 }

-- The theorem statement to be proved
theorem intersection_of_A_and_B : ∀ x, x ∈ setA ∩ setB ↔ x ∈ intersectAB := by
  sorry

end intersection_of_A_and_B_l1375_137532


namespace algebraic_expression_value_l1375_137504

theorem algebraic_expression_value (x : ℝ) (h : x^2 - x - 1 = 0) : x^3 - 2*x + 1 = 2 :=
sorry

end algebraic_expression_value_l1375_137504


namespace exists_group_of_four_l1375_137526

-- Assuming 21 students, and any three have done homework together exactly once in either mathematics or Russian.
-- We aim to prove there exists a group of four students such that any three of them have done homework together in the same subject.
noncomputable def students : Type := Fin 21

-- Define a predicate to show that three students have done homework together.
-- We use "math" and "russian" to denote the subjects.
inductive Subject
| math
| russian

-- Define a relation expressing that any three students have done exactly one subject homework together.
axiom homework_done (s1 s2 s3 : students) : Subject 

theorem exists_group_of_four :
  ∃ (a b c d : students), 
    (homework_done a b c = homework_done a b d) ∧
    (homework_done a b c = homework_done a c d) ∧
    (homework_done a b c = homework_done b c d) ∧
    (homework_done a b d = homework_done a c d) ∧
    (homework_done a b d = homework_done b c d) ∧
    (homework_done a c d = homework_done b c d) :=
sorry

end exists_group_of_four_l1375_137526


namespace parallelogram_area_l1375_137533

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) 
  (h_b : b = 7) (h_h : h = 2 * b) (h_A : A = b * h) : A = 98 :=
by {
  sorry
}

end parallelogram_area_l1375_137533


namespace cos_pi_plus_2alpha_value_l1375_137568

theorem cos_pi_plus_2alpha_value (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : 
    Real.cos (π + 2 * α) = 7 / 9 := sorry

end cos_pi_plus_2alpha_value_l1375_137568


namespace opposite_event_of_hitting_target_at_least_once_is_both_shots_miss_l1375_137577

/-- A person is shooting at a target, firing twice in succession. 
    The opposite event of "hitting the target at least once" is "both shots miss". -/
theorem opposite_event_of_hitting_target_at_least_once_is_both_shots_miss :
  ∀ (A B : Prop) (hits_target_at_least_once both_shots_miss : Prop), 
    (hits_target_at_least_once → (A ∨ B)) → (both_shots_miss ↔ ¬hits_target_at_least_once) ∧ 
    (¬(A ∧ B) → both_shots_miss) :=
by
  sorry

end opposite_event_of_hitting_target_at_least_once_is_both_shots_miss_l1375_137577


namespace correct_system_of_equations_l1375_137520

theorem correct_system_of_equations :
  ∃ (x y : ℕ), 
    x + y = 38 
    ∧ 26 * x + 20 * y = 952 := 
by
  sorry

end correct_system_of_equations_l1375_137520


namespace length_of_DE_l1375_137578

theorem length_of_DE (base : ℝ) (area_ratio : ℝ) (height_ratio : ℝ) :
  base = 18 → area_ratio = 0.09 → height_ratio = 0.3 → DE = 2 :=
by
  sorry

end length_of_DE_l1375_137578


namespace max_n_value_is_9_l1375_137546

variable (a b c d n : ℝ)
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : c > d)
variable (h : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d)))

theorem max_n_value_is_9 (h1 : a > b) (h2 : b > c) (h3 : c > d)
    (h : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (n / (a - d))) : n ≤ 9 :=
sorry

end max_n_value_is_9_l1375_137546


namespace anna_initial_stamps_l1375_137592

theorem anna_initial_stamps (final_stamps : ℕ) (alison_stamps : ℕ) (alison_to_anna : ℕ) : 
  final_stamps = 50 ∧ alison_stamps = 28 ∧ alison_to_anna = 14 → (final_stamps - alison_to_anna = 36) :=
by
  sorry

end anna_initial_stamps_l1375_137592


namespace line_intersects_semicircle_at_two_points_l1375_137512

theorem line_intersects_semicircle_at_two_points
  (m : ℝ) :
  (3 ≤ m ∧ m < 3 * Real.sqrt 2) ↔ 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ (y₁ = -x₁ + m ∧ y₁ = Real.sqrt (9 - x₁^2)) ∧ (y₂ = -x₂ + m ∧ y₂ = Real.sqrt (9 - x₂^2))) :=
by
  -- The proof goes here
  sorry

end line_intersects_semicircle_at_two_points_l1375_137512


namespace hexagon_sum_balanced_assignment_exists_l1375_137548

-- Definitions based on the conditions
def is_valid_assignment (a b c d e f g : ℕ) : Prop :=
a + b + g = a + c + g ∧ a + b + g = a + d + g ∧ a + b + g = a + e + g ∧
a + b + g = b + c + g ∧ a + b + g = b + d + g ∧ a + b + g = b + e + g ∧
a + b + g = c + d + g ∧ a + b + g = c + e + g ∧ a + b + g = d + e + g

-- The theorem we want to prove
theorem hexagon_sum_balanced_assignment_exists :
  ∃ (a b c d e f g : ℕ), 
  (a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 2 ∨ b = 3 ∨ b = 5) ∧ 
  (c = 2 ∨ c = 3 ∨ c = 5) ∧ 
  (d = 2 ∨ d = 3 ∨ d = 5) ∧ 
  (e = 2 ∨ e = 3 ∨ e = 5) ∧
  (f = 2 ∨ f = 3 ∨ f = 5) ∧
  (g = 2 ∨ g = 3 ∨ g = 5) ∧
  is_valid_assignment a b c d e f g :=
sorry

end hexagon_sum_balanced_assignment_exists_l1375_137548


namespace find_k_for_maximum_value_l1375_137576

theorem find_k_for_maximum_value (k : ℝ) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 2 → k * x^2 + 2 * k * x + 1 ≤ 5) ∧
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ k * x^2 + 2 * k * x + 1 = 5) ↔
  k = 1 / 2 ∨ k = -4 :=
by
  sorry

end find_k_for_maximum_value_l1375_137576


namespace James_balloons_l1375_137537

theorem James_balloons (A J : ℕ) (h1 : A = 513) (h2 : J = A + 208) : J = 721 :=
by {
  sorry
}

end James_balloons_l1375_137537


namespace middle_card_is_four_l1375_137519

theorem middle_card_is_four (a b c : ℕ) (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
                            (h2 : a + b + c = 15)
                            (h3 : a < b ∧ b < c)
                            (h_casey : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            (h_tracy : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            (h_stacy : true) -- Dummy condition, more detailed conditions can be derived from solution steps
                            : b = 4 := 
sorry

end middle_card_is_four_l1375_137519


namespace jellybean_total_l1375_137518

theorem jellybean_total (large_jellybeans_per_glass : ℕ) 
  (small_jellybeans_per_glass : ℕ) 
  (num_large_glasses : ℕ) 
  (num_small_glasses : ℕ) 
  (h1 : large_jellybeans_per_glass = 50) 
  (h2 : small_jellybeans_per_glass = large_jellybeans_per_glass / 2) 
  (h3 : num_large_glasses = 5) 
  (h4 : num_small_glasses = 3) : 
  (num_large_glasses * large_jellybeans_per_glass + num_small_glasses * small_jellybeans_per_glass) = 325 :=
by
  sorry

end jellybean_total_l1375_137518


namespace largest_val_is_E_l1375_137535

noncomputable def A : ℚ := 4 / (2 - 1/4)
noncomputable def B : ℚ := 4 / (2 + 1/4)
noncomputable def C : ℚ := 4 / (2 - 1/3)
noncomputable def D : ℚ := 4 / (2 + 1/3)
noncomputable def E : ℚ := 4 / (2 - 1/2)

theorem largest_val_is_E : E > A ∧ E > B ∧ E > C ∧ E > D := 
by sorry

end largest_val_is_E_l1375_137535


namespace y_square_range_l1375_137539

theorem y_square_range (y : ℝ) (h : (y + 16) ^ (1/3) - (y - 16) ^ (1/3) = 4) : 
  230 ≤ y^2 ∧ y^2 < 240 :=
sorry

end y_square_range_l1375_137539


namespace factorize_polynomial_l1375_137563

theorem factorize_polynomial (a b : ℝ) : a^2 - 9 * b^2 = (a + 3 * b) * (a - 3 * b) := by
  sorry

end factorize_polynomial_l1375_137563


namespace range_of_a_l1375_137594

noncomputable def prop_p (a x : ℝ) : Prop := 3 * a < x ∧ x < a

noncomputable def prop_q (x : ℝ) : Prop := x^2 - x - 6 < 0

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, ¬ prop_p a x) ∧ ¬ (∃ x : ℝ, ¬ prop_p a x) → ¬ (∃ x : ℝ, ¬ prop_q x) → -2/3 ≤ a ∧ a < 0 := 
by
  sorry

end range_of_a_l1375_137594


namespace function_passes_through_fixed_point_l1375_137550

noncomputable def f (a x : ℝ) := a^(x+1) - 1

theorem function_passes_through_fixed_point (a : ℝ) (h_pos : 0 < a) (h_not_one : a ≠ 1) :
  f a (-1) = 0 := by
  sorry

end function_passes_through_fixed_point_l1375_137550


namespace cos_alpha_value_l1375_137596

theorem cos_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos (α + π / 4) = 4 / 5) :
  Real.cos α = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end cos_alpha_value_l1375_137596


namespace lowest_test_score_dropped_l1375_137598

theorem lowest_test_score_dropped (A B C D : ℝ) 
  (h1: A + B + C + D = 280)
  (h2: A + B + C = 225) : D = 55 := 
by 
  sorry

end lowest_test_score_dropped_l1375_137598


namespace prism_volume_l1375_137579

theorem prism_volume (a b c : ℝ) (h1 : a * b = 18) (h2 : b * c = 12) (h3 : a * c = 8) : a * b * c = 12 :=
by sorry

end prism_volume_l1375_137579


namespace sin_330_eq_neg_half_l1375_137536

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_eq_neg_half_l1375_137536


namespace circumference_of_tank_B_l1375_137555

noncomputable def radius_of_tank (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def volume_of_tank (r h : ℝ) : ℝ := Real.pi * r^2 * h

theorem circumference_of_tank_B 
  (h_A : ℝ) (C_A : ℝ) (h_B : ℝ) (volume_ratio : ℝ)
  (hA_pos : 0 < h_A) (CA_pos : 0 < C_A) (hB_pos : 0 < h_B) (vr_pos : 0 < volume_ratio) :
  2 * Real.pi * (radius_of_tank (volume_of_tank (radius_of_tank C_A) h_A / (volume_ratio * Real.pi * h_B))) = 17.7245 :=
by 
  sorry

end circumference_of_tank_B_l1375_137555


namespace real_b_values_for_non_real_roots_l1375_137541

theorem real_b_values_for_non_real_roots (b : ℝ) :
  let discriminant := b^2 - 4 * 1 * 16
  discriminant < 0 ↔ -8 < b ∧ b < 8 := 
sorry

end real_b_values_for_non_real_roots_l1375_137541


namespace factorization_correct_l1375_137558

theorem factorization_correct :
  ∀ (m a b x y : ℝ), 
    (m^2 - 4 = (m + 2) * (m - 2)) ∧
    ((a + 3) * (a - 3) = a^2 - 9) ∧
    (a^2 - b^2 + 1 = (a + b) * (a - b) + 1) ∧
    (6 * x^2 * y^3 = 2 * x^2 * 3 * y^3) →
    (m^2 - 4 = (m + 2) * (m - 2)) :=
by
  intros m a b x y h
  have ⟨hA, hB, hC, hD⟩ := h
  exact hA

end factorization_correct_l1375_137558


namespace option_B_can_be_factored_l1375_137559

theorem option_B_can_be_factored (a b : ℝ) : 
  (-a^2 + b^2) = (b+a)*(b-a) := 
by
  sorry

end option_B_can_be_factored_l1375_137559


namespace x_intercept_of_line_l1375_137552

theorem x_intercept_of_line (x y : ℚ) (h_eq : 4 * x + 7 * y = 28) (h_y : y = 0) : (x, y) = (7, 0) := 
by 
  sorry

end x_intercept_of_line_l1375_137552


namespace coconut_grove_average_yield_l1375_137562

theorem coconut_grove_average_yield :
  ∀ (x : ℕ),
  40 * (x + 2) + 120 * x + 180 * (x - 2) = 100 * 3 * x →
  x = 7 :=
by
  intro x
  intro h
  /- sorry proof -/
  sorry

end coconut_grove_average_yield_l1375_137562


namespace evaluate_expression_l1375_137540

theorem evaluate_expression :
  2003^3 - 2002 * 2003^2 - 2002^2 * 2003 + 2002^3 = 4005 :=
by
  sorry

end evaluate_expression_l1375_137540


namespace only_number_smaller_than_zero_l1375_137527

theorem only_number_smaller_than_zero : ∀ (x : ℝ), (x = 5 ∨ x = 2 ∨ x = 0 ∨ x = -Real.sqrt 2) → x < 0 → x = -Real.sqrt 2 :=
by
  intro x hx h
  sorry

end only_number_smaller_than_zero_l1375_137527


namespace handshake_count_l1375_137505

def total_handshakes (men women : ℕ) := 
  (men * (men - 1)) / 2 + men * (women - 1)

theorem handshake_count :
  let men := 13
  let women := 13
  total_handshakes men women = 234 :=
by
  sorry

end handshake_count_l1375_137505


namespace month_length_l1375_137574

def treats_per_day : ℕ := 2
def cost_per_treat : ℝ := 0.1
def total_cost : ℝ := 6

theorem month_length : (total_cost / cost_per_treat) / treats_per_day = 30 := by
  sorry

end month_length_l1375_137574


namespace cannot_determine_remaining_pictures_l1375_137530

theorem cannot_determine_remaining_pictures (taken_pics : ℕ) (dolphin_show_pics : ℕ) (total_pics : ℕ) :
  taken_pics = 28 → dolphin_show_pics = 16 → total_pics = 44 → 
  (∀ capacity : ℕ, ¬ (total_pics + x = capacity)) → 
  ¬ ∃ remaining_pics : ℕ, remaining_pics = capacity - total_pics :=
by {
  sorry
}

end cannot_determine_remaining_pictures_l1375_137530


namespace radius_of_congruent_spheres_in_cone_l1375_137587

noncomputable def radius_of_congruent_spheres (base_radius height : ℝ) : ℝ := 
  let slant_height := Real.sqrt (height^2 + base_radius^2)
  let r := (4 : ℝ) / (10 + 4) * slant_height
  r

theorem radius_of_congruent_spheres_in_cone :
  radius_of_congruent_spheres 4 10 = 4 * Real.sqrt 29 / 7 := by
  sorry

end radius_of_congruent_spheres_in_cone_l1375_137587


namespace problem_statement_l1375_137511

-- Define the variables
variables (S T Tie : ℝ)

-- Define the given conditions
def condition1 : Prop := 6 * S + 4 * T + 2 * Tie = 80
def condition2 : Prop := 5 * S + 3 * T + 2 * Tie = 110

-- Define the question to be proved
def target : Prop := 4 * S + 2 * T + 2 * Tie = 50

-- Lean theorem statement
theorem problem_statement (h1 : condition1 S T Tie) (h2 : condition2 S T Tie) : target S T Tie :=
  sorry

end problem_statement_l1375_137511


namespace remainder_when_divided_l1375_137544

theorem remainder_when_divided (x : ℤ) (k : ℤ) (h: x = 82 * k + 5) : 
  ((x + 17) % 41) = 22 := by
  sorry

end remainder_when_divided_l1375_137544


namespace convert_yahs_to_bahs_l1375_137573

noncomputable section

def bahs_to_rahs (bahs : ℕ) : ℕ := bahs * (36/24)
def rahs_to_bahs (rahs : ℕ) : ℕ := rahs * (24/36)
def rahs_to_yahs (rahs : ℕ) : ℕ := rahs * (18/12)
def yahs_to_rahs (yahs : ℕ) : ℕ := yahs * (12/18)
def yahs_to_bahs (yahs : ℕ) : ℕ := rahs_to_bahs (yahs_to_rahs yahs)

theorem convert_yahs_to_bahs :
  yahs_to_bahs 1500 = 667 :=
sorry

end convert_yahs_to_bahs_l1375_137573


namespace unique_solution_exists_q_l1375_137556

theorem unique_solution_exists_q :
  (∃ q : ℝ, q ≠ 0 ∧ (∀ x y : ℝ, (2 * q * x^2 - 20 * x + 5 = 0) ∧ (2 * q * y^2 - 20 * y + 5 = 0) → x = y)) ↔ q = 10 := 
sorry

end unique_solution_exists_q_l1375_137556


namespace maynard_filled_percentage_l1375_137589

theorem maynard_filled_percentage (total_holes : ℕ) (unfilled_holes : ℕ) (filled_holes : ℕ) (p : ℚ) :
  total_holes = 8 →
  unfilled_holes = 2 →
  filled_holes = total_holes - unfilled_holes →
  p = (filled_holes : ℚ) / (total_holes : ℚ) * 100 →
  p = 75 := 
by {
  -- proofs and calculations would go here
  sorry
}

end maynard_filled_percentage_l1375_137589


namespace triangles_intersection_area_is_zero_l1375_137538

-- Define the vertices of the two triangles
def vertex_triangle_1 : Fin 3 → (ℝ × ℝ)
| ⟨0, _⟩ => (0, 2)
| ⟨1, _⟩ => (2, 1)
| ⟨2, _⟩ => (0, 0)

def vertex_triangle_2 : Fin 3 → (ℝ × ℝ)
| ⟨0, _⟩ => (2, 2)
| ⟨1, _⟩ => (0, 1)
| ⟨2, _⟩ => (2, 0)

-- The area of the intersection of the two triangles
def area_intersection (v1 v2 : Fin 3 → (ℝ × ℝ)) : ℝ :=
  0

-- The theorem to prove
theorem triangles_intersection_area_is_zero :
  area_intersection vertex_triangle_1 vertex_triangle_2 = 0 :=
by
  -- Proof is omitted here.
  sorry

end triangles_intersection_area_is_zero_l1375_137538


namespace equation_B_no_solution_l1375_137525

theorem equation_B_no_solution : ¬ ∃ x : ℝ, |-2 * x| + 6 = 0 :=
by
  sorry

end equation_B_no_solution_l1375_137525


namespace geese_survived_first_year_l1375_137570

-- Definitions based on the conditions
def total_eggs := 900
def hatch_rate := 2 / 3
def survive_first_month_rate := 3 / 4
def survive_first_year_rate := 2 / 5

-- Definitions derived from the conditions
def hatched_geese := total_eggs * hatch_rate
def survived_first_month := hatched_geese * survive_first_month_rate
def survived_first_year := survived_first_month * survive_first_year_rate

-- Target proof statement
theorem geese_survived_first_year : survived_first_year = 180 := by
  sorry

end geese_survived_first_year_l1375_137570


namespace river_trip_longer_than_lake_trip_l1375_137524

theorem river_trip_longer_than_lake_trip (v w : ℝ) (h1 : v > w) : 
  (20 * v) / (v^2 - w^2) > 20 / v :=
by {
  sorry
}

end river_trip_longer_than_lake_trip_l1375_137524


namespace find_triples_l1375_137515

-- Defining the conditions
def divides (x y : ℕ) : Prop := ∃ k, y = k * x

-- The main Lean statement
theorem find_triples (a b c : ℕ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  divides a (b * c - 1) → divides b (a * c - 1) → divides c (a * b - 1) →
  (a = 2 ∧ b = 3 ∧ c = 5) ∨ (a = 2 ∧ b = 5 ∧ c = 3) ∨
  (a = 3 ∧ b = 2 ∧ c = 5) ∨ (a = 3 ∧ b = 5 ∧ c = 2) ∨
  (a = 5 ∧ b = 2 ∧ c = 3) ∨ (a = 5 ∧ b = 3 ∧ c = 2) :=
sorry

end find_triples_l1375_137515


namespace quadratic_extreme_values_l1375_137508

theorem quadratic_extreme_values (y1 y2 y3 y4 : ℝ) 
  (h1 : y2 < y3) 
  (h2 : y3 = y4) 
  (h3 : ∀ x, ∃ (a b c : ℝ), ∀ y, y = a * x * x + b * x + c) :
  (y1 < y2) ∧ (y2 < y3) :=
by
  sorry

end quadratic_extreme_values_l1375_137508


namespace principal_amount_l1375_137561

theorem principal_amount (SI : ℝ) (T : ℝ) (R : ℝ) (P : ℝ) (h1 : SI = 140) (h2 : T = 2) (h3 : R = 17.5) :
  P = 400 :=
by
  -- Formal proof would go here
  sorry

end principal_amount_l1375_137561


namespace find_b_value_l1375_137566

-- Definitions based on given conditions
def original_line (x : ℝ) (b : ℝ) : ℝ := 2 * x + b
def shifted_line (x : ℝ) (b : ℝ) : ℝ := 2 * (x - 2) + b
def passes_through_origin (b : ℝ) := shifted_line 0 b = 0

-- Main proof statement
theorem find_b_value (b : ℝ) (h : passes_through_origin b) : b = 4 := by
  sorry

end find_b_value_l1375_137566


namespace solve_equation_l1375_137560

theorem solve_equation {x : ℂ} : (x - 2)^4 + (x - 6)^4 = 272 →
  x = 6 ∨ x = 2 ∨ x = 4 + 2 * Complex.I ∨ x = 4 - 2 * Complex.I :=
by
  intro h
  sorry

end solve_equation_l1375_137560
