import Mathlib

namespace geometric_sequence_178th_term_l410_41054

-- Conditions of the problem as definitions
def first_term : ℤ := 5
def second_term : ℤ := -20
def common_ratio : ℤ := second_term / first_term
def nth_term (a : ℤ) (r : ℤ) (n : ℕ) : ℤ := a * r^(n-1)

-- The translated problem statement in Lean 4
theorem geometric_sequence_178th_term :
  nth_term first_term common_ratio 178 = -5 * 4^177 :=
by
  repeat { sorry }

end geometric_sequence_178th_term_l410_41054


namespace condition_necessary_but_not_sufficient_l410_41014

-- Definitions based on given conditions
variables {a b c : ℝ}

-- The condition that needs to be qualified
def condition (a b c : ℝ) := a > 0 ∧ b^2 - 4 * a * c < 0

-- The statement to be verified
def statement (a b c : ℝ) := ∀ x : ℝ, a * x^2 + b * x + c > 0

-- Prove that the condition is a necessary but not sufficient condition for the statement
theorem condition_necessary_but_not_sufficient :
  condition a b c → (¬ (condition a b c ↔ statement a b c)) :=
by
  sorry

end condition_necessary_but_not_sufficient_l410_41014


namespace max_value_exp_l410_41059

theorem max_value_exp (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_constraint : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 8 + 7 * y * z + 5 * x * z ≤ 23.0219 :=
sorry

end max_value_exp_l410_41059


namespace marked_price_is_300_max_discount_is_50_l410_41031

-- Definition of the conditions given in the problem:
def loss_condition (x : ℝ) : Prop := 0.4 * x - 30 = 0.7 * x - 60
def profit_condition (x : ℝ) : Prop := 0.7 * x - 60 - (0.4 * x - 30) = 90

-- Statement for the first problem: Prove the marked price is 300 yuan.
theorem marked_price_is_300 : ∃ x : ℝ, loss_condition x ∧ profit_condition x ∧ x = 300 := by
  exists 300
  simp [loss_condition, profit_condition]
  sorry

noncomputable def max_discount (x : ℝ) : ℝ := 100 - (30 + 0.4 * x) / x * 100

def no_loss_max_discount (d : ℝ) : Prop := d = 50

-- Statement for the second problem: Prove the maximum discount is 50%.
theorem max_discount_is_50 (x : ℝ) (h_loss : loss_condition x) (h_profit : profit_condition x) : no_loss_max_discount (max_discount x) := by
  simp [max_discount, no_loss_max_discount]
  sorry

end marked_price_is_300_max_discount_is_50_l410_41031


namespace sin_value_proof_l410_41084

theorem sin_value_proof (θ : ℝ) (h : Real.cos (5 * Real.pi / 12 - θ) = 1 / 3) :
  Real.sin (Real.pi / 12 + θ) = 1 / 3 := by
  sorry

end sin_value_proof_l410_41084


namespace fraction_equivalence_l410_41076

theorem fraction_equivalence (x y : ℝ) (h : x ≠ y) :
  (x - y)^2 / (x^2 - y^2) = (x - y) / (x + y) :=
by
  sorry

end fraction_equivalence_l410_41076


namespace gcd_45_75_105_l410_41074

theorem gcd_45_75_105 : Nat.gcd (45 : ℕ) (Nat.gcd 75 105) = 15 := 
by
  sorry

end gcd_45_75_105_l410_41074


namespace find_some_expression_l410_41053

noncomputable def problem_statement : Prop :=
  ∃ (some_expression : ℝ), 
    (5 + 7 / 12 = 6 - some_expression) ∧ 
    (some_expression = 0.4167)

theorem find_some_expression : problem_statement := 
  sorry

end find_some_expression_l410_41053


namespace hyperbola_with_foci_on_y_axis_l410_41028

variable (m n : ℝ)

-- condition stating that mn < 0
def mn_neg : Prop := m * n < 0

-- the main theorem statement
theorem hyperbola_with_foci_on_y_axis (h : mn_neg m n) : 
  (∃ a : ℝ, a > 0 ∧ ∀ x y : ℝ, m * x^2 - m * y^2 = n ↔ y^2 - x^2 = a) :=
sorry

end hyperbola_with_foci_on_y_axis_l410_41028


namespace henrietta_has_three_bedrooms_l410_41066

theorem henrietta_has_three_bedrooms
  (living_room_walls_sqft : ℕ)
  (bedroom_walls_sqft : ℕ)
  (num_bedrooms : ℕ)
  (gallon_coverage_sqft : ℕ)
  (h1 : living_room_walls_sqft = 600)
  (h2 : bedroom_walls_sqft = 400)
  (h3 : gallon_coverage_sqft = 600)
  (h4 : num_bedrooms = 3) : 
  num_bedrooms = 3 :=
by
  exact h4

end henrietta_has_three_bedrooms_l410_41066


namespace total_books_left_l410_41062

def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def benny_lost_books : ℕ := 24

theorem total_books_left : sandy_books + tim_books - benny_lost_books = 19 :=
by
  sorry

end total_books_left_l410_41062


namespace degree_product_l410_41071

-- Define the degrees of the polynomials p and q
def degree_p : ℕ := 3
def degree_q : ℕ := 4

-- Define the functions p(x) and q(x) as polynomials and their respective degrees
axiom degree_p_definition (p : Polynomial ℝ) : p.degree = degree_p
axiom degree_q_definition (q : Polynomial ℝ) : q.degree = degree_q

-- Define the degree of the product p(x^2) * q(x^4)
noncomputable def degree_p_x2_q_x4 (p q : Polynomial ℝ) : ℕ :=
  2 * degree_p + 4 * degree_q

-- Prove that the degree of p(x^2) * q(x^4) is 22
theorem degree_product (p q : Polynomial ℝ) (hp : p.degree = degree_p) (hq : q.degree = degree_q) :
  degree_p_x2_q_x4 p q = 22 :=
by
  sorry

end degree_product_l410_41071


namespace number_halfway_l410_41012

theorem number_halfway (a b : ℚ) (h1 : a = 1/12) (h2 : b = 1/10) : (a + b) / 2 = 11 / 120 := by
  sorry

end number_halfway_l410_41012


namespace largest_multiple_of_15_less_than_400_l410_41023

theorem largest_multiple_of_15_less_than_400 (x : ℕ) (k : ℕ) (h : x = 15 * k) (h1 : x < 400) (h2 : ∀ m : ℕ, (15 * m < 400) → m ≤ k) : x = 390 :=
by
  sorry

end largest_multiple_of_15_less_than_400_l410_41023


namespace units_digit_of_p_is_6_l410_41078

theorem units_digit_of_p_is_6 (p : ℤ) (h1 : p % 10 > 0) 
                             (h2 : ((p^3) % 10 - (p^2) % 10) = 0) 
                             (h3 : (p + 1) % 10 = 7) : 
                             p % 10 = 6 :=
by sorry

end units_digit_of_p_is_6_l410_41078


namespace ratio_of_men_to_women_l410_41092

theorem ratio_of_men_to_women 
  (M W : ℕ) 
  (h1 : W = M + 5) 
  (h2 : M + W = 15): M = 5 ∧ W = 10 ∧ (M + W) / Nat.gcd M W = 1 ∧ (W + M) / Nat.gcd M W = 2 :=
by 
  sorry

end ratio_of_men_to_women_l410_41092


namespace pizza_slices_with_both_toppings_l410_41025

theorem pizza_slices_with_both_toppings (total_slices ham_slices pineapple_slices slices_with_both : ℕ)
  (h_total: total_slices = 15)
  (h_ham: ham_slices = 8)
  (h_pineapple: pineapple_slices = 12)
  (h_slices_with_both: slices_with_both + (ham_slices - slices_with_both) + (pineapple_slices - slices_with_both) = total_slices)
  : slices_with_both = 5 :=
by
  -- the proof would go here, but we use sorry to skip it
  sorry

end pizza_slices_with_both_toppings_l410_41025


namespace arcsin_sqrt_3_div_2_is_pi_div_3_l410_41011

noncomputable def arcsin_sqrt_3_div_2 : ℝ := Real.arcsin (Real.sqrt 3 / 2)

theorem arcsin_sqrt_3_div_2_is_pi_div_3 : arcsin_sqrt_3_div_2 = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_3_div_2_is_pi_div_3_l410_41011


namespace first_term_of_geometric_sequence_l410_41037

-- Define a geometric sequence
def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r^n

-- Initialize conditions
variable (a r : ℝ)

-- Provided that the 3rd term and the 6th term
def third_term : Prop := geometric_sequence a r 2 = 5
def sixth_term : Prop := geometric_sequence a r 5 = 40

-- The theorem to prove that a == 5/4 given the conditions
theorem first_term_of_geometric_sequence : third_term a r ∧ sixth_term a r → a = 5 / 4 :=
by 
  sorry

end first_term_of_geometric_sequence_l410_41037


namespace avg_cost_is_12_cents_l410_41027

noncomputable def avg_cost_per_pencil 
    (price_per_package : ℝ)
    (num_pencils : ℕ)
    (shipping_cost : ℝ)
    (discount_rate : ℝ) : ℝ :=
  let price_after_discount := price_per_package - (discount_rate * price_per_package)
  let total_cost := price_after_discount + shipping_cost
  let total_cost_cents := total_cost * 100
  total_cost_cents / num_pencils

theorem avg_cost_is_12_cents :
  avg_cost_per_pencil 29.70 300 8.50 0.10 = 12 := 
by {
  sorry
}

end avg_cost_is_12_cents_l410_41027


namespace cheese_pizzas_l410_41046

theorem cheese_pizzas (p b c total : ℕ) (h1 : p = 2) (h2 : b = 6) (h3 : total = 14) (ht : p + b + c = total) : c = 6 := 
by
  sorry

end cheese_pizzas_l410_41046


namespace trigonometric_product_l410_41098

theorem trigonometric_product :
  (1 - Real.sin (Real.pi / 12)) * 
  (1 - Real.sin (5 * Real.pi / 12)) * 
  (1 - Real.sin (7 * Real.pi / 12)) * 
  (1 - Real.sin (11 * Real.pi / 12)) = 1 / 4 :=
by sorry

end trigonometric_product_l410_41098


namespace distinct_real_roots_c_l410_41013

theorem distinct_real_roots_c (c : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + c = 0 ∧ x₂^2 - 4*x₂ + c = 0) ↔ c < 4 := by
  sorry

end distinct_real_roots_c_l410_41013


namespace problem_eq_995_l410_41073

theorem problem_eq_995 :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) /
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400))
  = 995 := sorry

end problem_eq_995_l410_41073


namespace min_photographs_42_tourists_3_monuments_l410_41029

noncomputable def min_photos_taken (num_tourists : ℕ) (num_monuments : ℕ) : ℕ :=
  if num_tourists = 42 ∧ num_monuments = 3 then 123 else 0

-- Main statement:
theorem min_photographs_42_tourists_3_monuments : 
  (∀ (num_tourists num_monuments : ℕ), 
    num_tourists = 42 ∧ num_monuments = 3 → min_photos_taken num_tourists num_monuments = 123)
  := by
    sorry

end min_photographs_42_tourists_3_monuments_l410_41029


namespace ratio_matt_fem_4_1_l410_41010

-- Define Fem's current age
def FemCurrentAge : ℕ := 11

-- Define the condition about the sum of their ages in two years
def AgeSumInTwoYears (MattCurrentAge : ℕ) : Prop :=
  (FemCurrentAge + 2) + (MattCurrentAge + 2) = 59

-- Define the desired ratio as a property
def DesiredRatio (MattCurrentAge : ℕ) : Prop :=
  MattCurrentAge / FemCurrentAge = 4

-- Create the theorem statement
theorem ratio_matt_fem_4_1 (M : ℕ) (h : AgeSumInTwoYears M) : DesiredRatio M :=
  sorry

end ratio_matt_fem_4_1_l410_41010


namespace range_of_a_l410_41047

-- Definitions for propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, ¬(x^2 + (a-1)*x + 1 ≤ 0)

def q (a : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (a - 1)^x₁ < (a - 1)^x₂

-- The final theorem to prove
theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) → (-1 < a ∧ a ≤ 2) ∨ (a ≥ 3) :=
by
  sorry

end range_of_a_l410_41047


namespace son_l410_41017

theorem son's_age (S M : ℕ) (h₁ : M = S + 25) (h₂ : M + 2 = 2 * (S + 2)) : S = 23 := by
  sorry

end son_l410_41017


namespace find_g_neg_2_l410_41067

-- Definitions
variable {R : Type*} [CommRing R] [Inhabited R]
variable (f g : R → R)

-- Conditions
axiom odd_y (x : R) : f (-x) + 2 * x^2 = -(f x + 2 * x^2)
axiom definition_g (x : R) : g x = f x + 1
axiom value_f_2 : f 2 = 2

-- Goal
theorem find_g_neg_2 : g (-2) = -17 :=
by
  sorry

end find_g_neg_2_l410_41067


namespace domain_eq_l410_41064

theorem domain_eq (f : ℝ → ℝ) : 
  (∀ x : ℝ, -1 ≤ 3 - 2 * x ∧ 3 - 2 * x ≤ 2) ↔ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 :=
by sorry

end domain_eq_l410_41064


namespace cars_in_north_america_correct_l410_41008

def total_cars_produced : ℕ := 6755
def cars_produced_in_europe : ℕ := 2871

def cars_produced_in_north_america : ℕ := total_cars_produced - cars_produced_in_europe

theorem cars_in_north_america_correct : cars_produced_in_north_america = 3884 :=
by sorry

end cars_in_north_america_correct_l410_41008


namespace emily_age_proof_l410_41009

theorem emily_age_proof (e m : ℕ) (h1 : e = m - 18) (h2 : e + m = 54) : e = 18 :=
by
  sorry

end emily_age_proof_l410_41009


namespace complement_intersection_l410_41003

-- Define sets P and Q.
def P : Set ℝ := {x | x ≥ 2}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Define the complement of P.
def complement_P : Set ℝ := {x | x < 2}

-- The theorem we need to prove.
theorem complement_intersection : complement_P ∩ Q = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end complement_intersection_l410_41003


namespace units_digit_in_base_7_l410_41085

theorem units_digit_in_base_7 (n m : ℕ) (h1 : n = 312) (h2 : m = 57) : (n * m) % 7 = 4 :=
by
  sorry

end units_digit_in_base_7_l410_41085


namespace cannot_make_120_cents_with_6_coins_l410_41042

def Coin := ℕ → ℕ -- represents a number of each type of coin

noncomputable def coin_value (c : Coin) : ℕ :=
  c 0 * 1 + c 1 * 5 + c 2 * 10 + c 3 * 25

def total_coins (c : Coin) : ℕ :=
  c 0 + c 1 + c 2 + c 3

theorem cannot_make_120_cents_with_6_coins (c : Coin) (h1 : total_coins c = 6) :
  coin_value c ≠ 120 :=
sorry

end cannot_make_120_cents_with_6_coins_l410_41042


namespace solve_for_F_l410_41056

theorem solve_for_F (C F : ℝ) (h1 : C = 5 / 9 * (F - 32)) (h2 : C = 40) : F = 104 :=
by
  sorry

end solve_for_F_l410_41056


namespace present_age_of_A_l410_41016

theorem present_age_of_A {x : ℕ} (h₁ : ∃ (x : ℕ), 5 * x = A ∧ 3 * x = B)
                         (h₂ : ∀ (A B : ℕ), (A + 6) / (B + 6) = 7 / 5) : A = 15 :=
by sorry

end present_age_of_A_l410_41016


namespace second_divisor_l410_41032

theorem second_divisor (N k D m : ℤ) (h1 : N = 35 * k + 25) (h2 : N = D * m + 4) : D = 17 := by
  -- Follow conditions from problem
  sorry

end second_divisor_l410_41032


namespace perpendicular_to_plane_l410_41058

theorem perpendicular_to_plane (Line : Type) (Plane : Type) (triangle : Plane) (circle : Plane)
  (perpendicular1 : Line → Plane → Prop)
  (perpendicular2 : Line → Plane → Prop) :
  (∀ l, ∃ t, perpendicular1 l t ∧ t = triangle) ∧ (∀ l, ∃ c, perpendicular2 l c ∧ c = circle) →
  (∀ l, ∃ p, (perpendicular1 l p ∨ perpendicular2 l p) ∧ (p = triangle ∨ p = circle)) :=
by
  sorry

end perpendicular_to_plane_l410_41058


namespace find_x2_y2_l410_41077

theorem find_x2_y2 (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = (10344 / 169) := by
  sorry

end find_x2_y2_l410_41077


namespace problem_180_180_minus_12_l410_41090

namespace MathProof

theorem problem_180_180_minus_12 :
  180 * (180 - 12) - (180 * 180 - 12) = -2148 := 
by
  -- Placeholders for computation steps
  sorry

end MathProof

end problem_180_180_minus_12_l410_41090


namespace imaginary_part_of_z_l410_41052

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : (3 - 4 * I) * z = abs (4 + 3 * I)) : im z = 4 / 5 :=
sorry

end imaginary_part_of_z_l410_41052


namespace andy_wrong_questions_l410_41001

theorem andy_wrong_questions (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 6) (h3 : c = 3) : a = 6 := by
  sorry

end andy_wrong_questions_l410_41001


namespace range_of_a_l410_41061

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x + 1

def is_monotonic_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) a, 0 ≤ (deriv f) x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) a, 0 ≤ (deriv (f a)) x) → 1 ≤ a := 
sorry

end range_of_a_l410_41061


namespace find_f_neg_one_l410_41091

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x >= 0 then 2^x + 2 * x + m else -(2^(-x) + 2 * (-x) + m)

theorem find_f_neg_one (m : ℝ) (h_m : f 0 m = 0) : f (-1) (-1) = -3 :=
by
  sorry

end find_f_neg_one_l410_41091


namespace impossible_to_use_up_all_parts_l410_41069

theorem impossible_to_use_up_all_parts (p q r : ℕ) :
  (∃ p q r : ℕ,
    2 * p + 2 * r + 2 = A ∧
    2 * p + q + 1 = B ∧
    q + r = C) → false :=
by {
  sorry
}

end impossible_to_use_up_all_parts_l410_41069


namespace one_inch_represents_feet_l410_41065

def height_statue : ℕ := 80 -- Height of the statue in feet

def height_model : ℕ := 5 -- Height of the model in inches

theorem one_inch_represents_feet : (height_statue / height_model) = 16 := 
by
  sorry

end one_inch_represents_feet_l410_41065


namespace one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero_l410_41050

theorem one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / a < 1 / b) ↔ ((a * b) / (a^3 - b^3) > 0) := 
by
  sorry

end one_over_a_lt_one_over_b_iff_ab_over_a3_minus_b3_gt_zero_l410_41050


namespace forum_posting_total_l410_41041

theorem forum_posting_total (num_members : ℕ) (num_answers_per_question : ℕ) (num_questions_per_hour : ℕ) (hours_per_day : ℕ) :
  num_members = 1000 ->
  num_answers_per_question = 5 ->
  num_questions_per_hour = 7 ->
  hours_per_day = 24 ->
  ((num_questions_per_hour * hours_per_day * num_members) + (num_answers_per_question * num_questions_per_hour * hours_per_day * num_members)) = 1008000 :=
by
  intros
  sorry

end forum_posting_total_l410_41041


namespace salem_size_comparison_l410_41002

theorem salem_size_comparison (S L : ℕ) (hL: L = 58940)
  (hSalem: S - 130000 = 2 * 377050) :
  (S / L = 15) :=
sorry

end salem_size_comparison_l410_41002


namespace difference_between_numbers_l410_41004

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 27630) (h2 : a = 5 * b + 5) : a - b = 18421 :=
  sorry

end difference_between_numbers_l410_41004


namespace alex_mother_age_proof_l410_41089

-- Define the initial conditions
def alex_age_2004 : ℕ := 7
def mother_age_2004 : ℕ := 35
def initial_year : ℕ := 2004

-- Define the time variable and the relationship conditions
def years_after_2004 (x : ℕ) : Prop :=
  let alex_age := alex_age_2004 + x
  let mother_age := mother_age_2004 + x
  mother_age = 2 * alex_age

-- State the theorem to be proved
theorem alex_mother_age_proof : ∃ x : ℕ, years_after_2004 x ∧ initial_year + x = 2025 :=
by
  sorry

end alex_mother_age_proof_l410_41089


namespace sum_infinite_partial_fraction_l410_41044

theorem sum_infinite_partial_fraction :
  ∑' n : ℕ, n > 0 → (3 * n - 2) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by
  sorry

end sum_infinite_partial_fraction_l410_41044


namespace frank_spends_more_l410_41079

def cost_computer_table : ℕ := 140
def cost_computer_chair : ℕ := 100
def cost_joystick : ℕ := 20
def frank_share_joystick : ℕ := cost_joystick / 4
def eman_share_joystick : ℕ := cost_joystick * 3 / 4

def total_spent_frank : ℕ := cost_computer_table + frank_share_joystick
def total_spent_eman : ℕ := cost_computer_chair + eman_share_joystick

theorem frank_spends_more : total_spent_frank - total_spent_eman = 30 :=
by
  sorry

end frank_spends_more_l410_41079


namespace green_notebook_cost_each_l410_41097

-- Definitions for conditions:
def num_notebooks := 4
def num_green_notebooks := 2
def num_black_notebooks := 1
def num_pink_notebooks := 1
def total_cost := 45
def black_notebook_cost := 15
def pink_notebook_cost := 10

-- Define the problem statement:
theorem green_notebook_cost_each : 
  (2 * g + black_notebook_cost + pink_notebook_cost = total_cost) → 
  g = 10 := 
by 
  intros h
  sorry

end green_notebook_cost_each_l410_41097


namespace base_amount_calculation_l410_41087

theorem base_amount_calculation (tax_amount : ℝ) (tax_rate : ℝ) (base_amount : ℝ) 
  (h1 : tax_amount = 82) (h2 : tax_rate = 82) : base_amount = 100 :=
by
  -- Proof will be provided here.
  sorry

end base_amount_calculation_l410_41087


namespace solve_system1_solve_system2_l410_41086

-- Definitions for the first system of equations
def system1_equation1 (x y : ℚ) := 3 * x - 6 * y = 4
def system1_equation2 (x y : ℚ) := x + 5 * y = 6

-- Definitions for the second system of equations
def system2_equation1 (x y : ℚ) := x / 4 + y / 3 = 3
def system2_equation2 (x y : ℚ) := 3 * (x - 4) - 2 * (y - 1) = -1

-- Lean statement for proving the solution to the first system
theorem solve_system1 :
  ∃ (x y : ℚ), system1_equation1 x y ∧ system1_equation2 x y ∧ x = 8 / 3 ∧ y = 2 / 3 :=
by
  sorry

-- Lean statement for proving the solution to the second system
theorem solve_system2 :
  ∃ (x y : ℚ), system2_equation1 x y ∧ system2_equation2 x y ∧ x = 6 ∧ y = 9 / 2 :=
by
  sorry

end solve_system1_solve_system2_l410_41086


namespace sheets_in_set_l410_41018

-- Definitions of the conditions
def John_sheets_left (S E : ℕ) : Prop := S - E = 80
def Mary_sheets_used (S E : ℕ) : Prop := S = 4 * E

-- Theorems to prove the number of sheets
theorem sheets_in_set (S E : ℕ) (hJohn : John_sheets_left S E) (hMary : Mary_sheets_used S E) : S = 320 :=
by { 
  sorry 
}

end sheets_in_set_l410_41018


namespace sample_size_calculation_l410_41068

-- Definitions based on the conditions
def num_classes : ℕ := 40
def num_representatives_per_class : ℕ := 3

-- Theorem statement we aim to prove
theorem sample_size_calculation : num_classes * num_representatives_per_class = 120 :=
by
  sorry

end sample_size_calculation_l410_41068


namespace suraj_innings_l410_41021

theorem suraj_innings (n A : ℕ) (h1 : A + 6 = 16) (h2 : (n * A + 112) / (n + 1) = 16) : n = 16 :=
by
  sorry

end suraj_innings_l410_41021


namespace initial_pens_eq_42_l410_41055

-- Definitions based on the conditions
def initial_books : ℕ := 143
def remaining_books : ℕ := 113
def remaining_pens : ℕ := 19
def sold_pens : ℕ := 23

-- Theorem to prove that the initial number of pens was 42
theorem initial_pens_eq_42 (b_init b_remain p_remain p_sold : ℕ) 
    (H_b_init : b_init = initial_books)
    (H_b_remain : b_remain = remaining_books)
    (H_p_remain : p_remain = remaining_pens)
    (H_p_sold : p_sold = sold_pens) : 
    (p_sold + p_remain = 42) := 
by {
    -- Provide proof later
    sorry
}

end initial_pens_eq_42_l410_41055


namespace olivia_pays_in_dollars_l410_41030

theorem olivia_pays_in_dollars (q_chips q_soda : ℕ) 
  (h_chips : q_chips = 4) (h_soda : q_soda = 12) : (q_chips + q_soda) / 4 = 4 := by
  sorry

end olivia_pays_in_dollars_l410_41030


namespace abc_relationship_l410_41019

noncomputable def a : ℝ := Real.log 5 - Real.log 3
noncomputable def b : ℝ := (2/5) * Real.exp (2/3)
noncomputable def c : ℝ := 2/3

theorem abc_relationship : b > c ∧ c > a :=
by
  sorry

end abc_relationship_l410_41019


namespace special_op_eight_four_l410_41057

def special_op (a b : ℕ) : ℕ := 2 * a + a / b

theorem special_op_eight_four : special_op 8 4 = 18 := by
  sorry

end special_op_eight_four_l410_41057


namespace isosceles_triangle_bisector_properties_l410_41020

theorem isosceles_triangle_bisector_properties:
  ∀ (T : Type) (triangle : T)
  (is_isosceles : Prop) (vertex_angle_bisector_bisects_base : Prop) (vertex_angle_bisector_perpendicular_to_base : Prop),
  is_isosceles 
  → (vertex_angle_bisector_bisects_base ∧ vertex_angle_bisector_perpendicular_to_base) :=
sorry

end isosceles_triangle_bisector_properties_l410_41020


namespace solution_set_of_inequality_l410_41088

theorem solution_set_of_inequality : 
  {x : ℝ | x * (x + 3) ≥ 0} = {x : ℝ | x ≥ 0 ∨ x ≤ -3} := 
by sorry

end solution_set_of_inequality_l410_41088


namespace overall_profit_percentage_is_30_l410_41045

noncomputable def overall_profit_percentage (n_A n_B : ℕ) (price_A price_B profit_A profit_B : ℝ) : ℝ :=
  (n_A * profit_A + n_B * profit_B) / (n_A * price_A + n_B * price_B) * 100

theorem overall_profit_percentage_is_30 :
  overall_profit_percentage 5 10 850 950 225 300 = 30 :=
by
  sorry

end overall_profit_percentage_is_30_l410_41045


namespace marble_probability_is_correct_l410_41043

def marbles_probability
  (total_marbles: ℕ) 
  (red_marbles: ℕ) 
  (blue_marbles: ℕ) 
  (green_marbles: ℕ)
  (choose_marbles: ℕ) 
  (required_red: ℕ) 
  (required_blue: ℕ) 
  (required_green: ℕ): ℚ := sorry

-- Define conditions
def total_marbles := 7
def red_marbles := 3
def blue_marbles := 2
def green_marbles := 2
def choose_marbles := 4
def required_red := 2
def required_blue := 1
def required_green := 1

-- Proof statement
theorem marble_probability_is_correct : 
  marbles_probability total_marbles red_marbles blue_marbles green_marbles choose_marbles required_red required_blue required_green = (12 / 35 : ℚ) :=
sorry

end marble_probability_is_correct_l410_41043


namespace perpendicular_slope_l410_41022

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 4 * y = 20) : 
  ∃ m : ℝ, m = -4 / 5 :=
sorry

end perpendicular_slope_l410_41022


namespace abs_sum_less_b_l410_41015

theorem abs_sum_less_b (x : ℝ) (b : ℝ) (h : |2 * x - 8| + |2 * x - 6| < b) (hb : b > 0) : b > 2 :=
by
  sorry

end abs_sum_less_b_l410_41015


namespace find_unknown_rate_l410_41036

-- Define the known quantities
def num_blankets1 := 4
def price1 := 100

def num_blankets2 := 5
def price2 := 150

def num_blankets3 := 3
def price3 := 200

def num_blankets4 := 6
def price4 := 75

def num_blankets_unknown := 2

def avg_price := 150
def total_blankets := num_blankets1 + num_blankets2 + num_blankets3 + num_blankets4 + num_blankets_unknown -- 20 blankets in total

-- Hypotheses
def total_known_cost := num_blankets1 * price1 + num_blankets2 * price2 + num_blankets3 * price3 + num_blankets4 * price4
-- 2200 Rs.

def total_cost := total_blankets * avg_price -- 3000 Rs.

theorem find_unknown_rate :
  (total_cost - total_known_cost) / num_blankets_unknown = 400 :=
by sorry

end find_unknown_rate_l410_41036


namespace tim_income_less_juan_l410_41096

variable {T M J : ℝ}

theorem tim_income_less_juan :
  (M = 1.60 * T) → (M = 0.6400000000000001 * J) → T = 0.4 * J :=
by
  sorry

end tim_income_less_juan_l410_41096


namespace gemma_amount_given_l410_41034

theorem gemma_amount_given
  (cost_per_pizza : ℕ)
  (number_of_pizzas : ℕ)
  (tip : ℕ)
  (change_back : ℕ)
  (h1 : cost_per_pizza = 10)
  (h2 : number_of_pizzas = 4)
  (h3 : tip = 5)
  (h4 : change_back = 5) :
  number_of_pizzas * cost_per_pizza + tip + change_back = 50 := sorry

end gemma_amount_given_l410_41034


namespace find_876_last_three_digits_l410_41035

noncomputable def has_same_last_three_digits (N : ℕ) : Prop :=
  (N^2 - N) % 1000 = 0

theorem find_876_last_three_digits (N : ℕ) (h1 : has_same_last_three_digits N) (h2 : N > 99) (h3 : N < 1000) : 
  N % 1000 = 876 :=
sorry

end find_876_last_three_digits_l410_41035


namespace executed_is_9_l410_41070

-- Define the conditions based on given problem
variables (x K I : ℕ)

-- Condition 1: Number of killed
def number_killed (x : ℕ) : ℕ := 2 * x + 4

-- Condition 2: Number of injured
def number_injured (x : ℕ) : ℕ := (16 * x) / 3 + 8

-- Condition 3: Total of killed, injured, and executed is less than 98
def total_less_than_98 (x : ℕ) (k : ℕ) (i : ℕ) : Prop := k + i + x < 98

-- Condition 4: Relation between killed and executed
def killed_relation (x : ℕ) (k : ℕ) : Prop := k - 4 = 2 * x

-- The final theorem statement to prove
theorem executed_is_9 : ∃ x, number_killed x = 2 * x + 4 ∧
                       number_injured x = (16 * x) / 3 + 8 ∧
                       total_less_than_98 x (number_killed x) (number_injured x) ∧
                       killed_relation x (number_killed x) ∧
                       x = 9 :=
by
  sorry

end executed_is_9_l410_41070


namespace george_total_socks_l410_41005

-- Define the initial number of socks George had
def initial_socks : ℝ := 28.0

-- Define the number of socks he bought
def bought_socks : ℝ := 36.0

-- Define the number of socks his Dad gave him
def given_socks : ℝ := 4.0

-- Define the number of total socks
def total_socks : ℝ := initial_socks + bought_socks + given_socks

-- State the theorem we want to prove
theorem george_total_socks : total_socks = 68.0 :=
by
  sorry

end george_total_socks_l410_41005


namespace Hulk_jump_l410_41099

theorem Hulk_jump :
  ∃ n : ℕ, 2^n > 500 ∧ ∀ m : ℕ, m < n → 2^m ≤ 500 :=
by
  sorry

end Hulk_jump_l410_41099


namespace original_slices_proof_l410_41083

def original_slices (andy_consumption toast_slices leftover_slice: ℕ) : ℕ :=
  andy_consumption + toast_slices + leftover_slice

theorem original_slices_proof :
  original_slices (3 * 2) (10 * 2) 1 = 27 :=
by
  sorry

end original_slices_proof_l410_41083


namespace leftover_cents_l410_41049

noncomputable def total_cents (pennies nickels dimes quarters : Nat) : Nat :=
  (pennies * 1) + (nickels * 5) + (dimes * 10) + (quarters * 25)

noncomputable def total_cost (num_people : Nat) (cost_per_person : Nat) : Nat :=
  num_people * cost_per_person

theorem leftover_cents (h₁ : total_cents 123 85 35 26 = 1548)
                       (h₂ : total_cost 5 300 = 1500) :
  1548 - 1500 = 48 :=
sorry

end leftover_cents_l410_41049


namespace proportion_of_bike_riders_is_correct_l410_41024

-- Define the given conditions as constants
def total_students : ℕ := 92
def bus_riders : ℕ := 20
def walkers : ℕ := 27

-- Define the remaining students after bus riders and after walkers
def remaining_after_bus_riders : ℕ := total_students - bus_riders
def bike_riders : ℕ := remaining_after_bus_riders - walkers

-- Define the expected proportion
def expected_proportion : ℚ := 45 / 72

-- State the theorem to be proved
theorem proportion_of_bike_riders_is_correct :
  (↑bike_riders / ↑remaining_after_bus_riders : ℚ) = expected_proportion := 
by
  sorry

end proportion_of_bike_riders_is_correct_l410_41024


namespace proof_expr_is_neg_four_ninths_l410_41080

noncomputable def example_expr : ℚ := (-3 / 2) ^ 2021 * (2 / 3) ^ 2023

theorem proof_expr_is_neg_four_ninths : example_expr = (-4 / 9) := 
by 
  -- Here the proof would be placed
  sorry

end proof_expr_is_neg_four_ninths_l410_41080


namespace rectangle_perimeter_equal_area_l410_41093

theorem rectangle_perimeter_equal_area (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 2 * a + 2 * b) : 2 * (a + b) = 18 := 
by 
  sorry

end rectangle_perimeter_equal_area_l410_41093


namespace beyonce_total_songs_l410_41033

theorem beyonce_total_songs (s a b t : ℕ) (h_s : s = 5) (h_a : a = 2 * 15) (h_b : b = 20) (h_t : t = s + a + b) : t = 55 := by
  rw [h_s, h_a, h_b] at h_t
  exact h_t

end beyonce_total_songs_l410_41033


namespace translate_line_up_l410_41007

-- Define the original line equation as a function
def original_line (x : ℝ) : ℝ := 2 * x - 4

-- Define the new line equation after translating upwards by 5 units
def new_line (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement to prove the translation result
theorem translate_line_up (x : ℝ) : original_line x + 5 = new_line x :=
by
  -- This would normally be where the proof goes, but we'll insert a placeholder
  sorry

end translate_line_up_l410_41007


namespace triangle_perimeter_l410_41040

def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

variable (a b c : ℝ)

theorem triangle_perimeter
  (h1 : 90 = (1/2) * 18 * b)
  (h2 : right_triangle 18 b c) :
  18 + b + c = 28 + 2 * Real.sqrt 106 :=
by
  sorry

end triangle_perimeter_l410_41040


namespace compute_five_fold_application_l410_41063

def f (x : ℤ) : ℤ :=
  if x ≥ 0 then -2 * x^2 else x^2 + 4 * x + 12

theorem compute_five_fold_application :
  f (f (f (f (f 2)))) = -449183247763232 :=
  by
    sorry

end compute_five_fold_application_l410_41063


namespace contractor_job_completion_l410_41000

theorem contractor_job_completion 
  (total_days : ℕ := 100) 
  (initial_workers : ℕ := 10) 
  (days_worked_initial : ℕ := 20) 
  (fraction_completed_initial : ℚ := 1/4) 
  (fired_workers : ℕ := 2) 
  : ∀ (remaining_days : ℕ), remaining_days = 75 → (remaining_days + days_worked_initial = 95) :=
by
  sorry

end contractor_job_completion_l410_41000


namespace total_pages_in_book_l410_41026

-- Conditions
def hours_reading := 5
def pages_read := 2323
def increase_per_hour := 10
def extra_pages_read := 90

-- Main statement to prove
theorem total_pages_in_book (T : ℕ) :
  (∃ P : ℕ, P + (P + increase_per_hour) + (P + 2 * increase_per_hour) + 
   (P + 3 * increase_per_hour) + (P + 4 * increase_per_hour) = pages_read) ∧
  (pages_read = T - pages_read + extra_pages_read) →
  T = 4556 :=
by { sorry }

end total_pages_in_book_l410_41026


namespace one_third_of_four_l410_41048

theorem one_third_of_four : (1/3) * 4 = 2 :=
by
  sorry

end one_third_of_four_l410_41048


namespace mixed_oil_rate_l410_41051

theorem mixed_oil_rate :
  let oil1 := (10, 50)
  let oil2 := (5, 68)
  let oil3 := (8, 42)
  let oil4 := (7, 62)
  let oil5 := (12, 55)
  let oil6 := (6, 75)
  let total_cost := oil1.1 * oil1.2 + oil2.1 * oil2.2 + oil3.1 * oil3.2 + oil4.1 * oil4.2 + oil5.1 * oil5.2 + oil6.1 * oil6.2
  let total_volume := oil1.1 + oil2.1 + oil3.1 + oil4.1 + oil5.1 + oil6.1
  (total_cost / total_volume : ℝ) = 56.67 :=
by
  sorry

end mixed_oil_rate_l410_41051


namespace min_value_of_A_div_B_l410_41075

noncomputable def A (g1 : Finset ℕ) : ℕ :=
  g1.prod id

noncomputable def B (g2 : Finset ℕ) : ℕ :=
  g2.prod id

theorem min_value_of_A_div_B : ∃ (g1 g2 : Finset ℕ), 
  g1 ∪ g2 = (Finset.range 31).erase 0 ∧ g1 ∩ g2 = ∅ ∧ A g1 % B g2 = 0 ∧ A g1 / B g2 = 1077205 :=
by
  sorry

end min_value_of_A_div_B_l410_41075


namespace difference_is_24_l410_41072

namespace BuffaloesAndDucks

def numLegs (B D : ℕ) : ℕ := 4 * B + 2 * D

def numHeads (B D : ℕ) : ℕ := B + D

def diffLegsAndHeads (B D : ℕ) : ℕ := numLegs B D - 2 * numHeads B D

theorem difference_is_24 (D : ℕ) : diffLegsAndHeads 12 D = 24 := by
  sorry

end BuffaloesAndDucks

end difference_is_24_l410_41072


namespace a_1_value_l410_41095

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ)

axiom a_n_def : ∀ n ≥ 2, a n + 2 * (S n) * (S (n - 1)) = 0
axiom S_5_value : S 5 = 1/11
axiom summation_def : ∀ k ≥ 1, S k = S (k - 1) + a k

theorem a_1_value : a 1 = 1/3 := by
  sorry

end a_1_value_l410_41095


namespace minimum_selling_price_l410_41060

theorem minimum_selling_price (total_cost : ℝ) (total_fruit : ℝ) (spoilage : ℝ) (min_price : ℝ) :
  total_cost = 760 ∧ total_fruit = 80 ∧ spoilage = 0.05 ∧ min_price = 10 → 
  ∀ price : ℝ, (price * total_fruit * (1 - spoilage) >= total_cost) → price >= min_price :=
by
  intros h price hp
  rcases h with ⟨hc, hf, hs, hm⟩
  sorry

end minimum_selling_price_l410_41060


namespace find_a_l410_41039

theorem find_a : 
  (∃ (a : ℝ), ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 4 * y = 0 → (3 * x + y + a = 0))) → a = 1 :=
sorry

end find_a_l410_41039


namespace find_product_l410_41082

theorem find_product (a b c d : ℝ) 
  (h_avg : (a + b + c + d) / 4 = 7.1)
  (h_rel : 2.5 * a = b - 1.2 ∧ b - 1.2 = c + 4.8 ∧ c + 4.8 = 0.25 * d) :
  a * b * c * d = 49.6 := 
sorry

end find_product_l410_41082


namespace comprehensive_score_l410_41081

theorem comprehensive_score :
  let w_c := 0.4
  let w_u := 0.6
  let s_c := 80
  let s_u := 90
  s_c * w_c + s_u * w_u = 86 :=
by
  sorry

end comprehensive_score_l410_41081


namespace line_intersects_y_axis_at_eight_l410_41006

theorem line_intersects_y_axis_at_eight :
  ∃ b : ℝ, ∃ f : ℝ → ℝ, (∀ x, f x = 2 * x + b) ∧ f 1 = 10 ∧ f (-9) = -10 ∧ f 0 = 8 :=
by
  -- Definitions and calculations leading to verify the theorem
  sorry

end line_intersects_y_axis_at_eight_l410_41006


namespace smallest_number_of_ducks_l410_41038

theorem smallest_number_of_ducks (n_ducks n_cranes : ℕ) (h1 : n_ducks = n_cranes) : 
  ∃ n, n_ducks = n ∧ n_cranes = n ∧ n = Nat.lcm 13 17 := by
  use 221
  sorry

end smallest_number_of_ducks_l410_41038


namespace find_x_of_orthogonal_vectors_l410_41094

theorem find_x_of_orthogonal_vectors (x : ℝ) : 
  (⟨3, -4, 1⟩ : ℝ × ℝ × ℝ) • (⟨x, 2, -7⟩ : ℝ × ℝ × ℝ) = 0 → x = 5 := 
by
  sorry

end find_x_of_orthogonal_vectors_l410_41094
