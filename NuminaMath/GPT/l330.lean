import Mathlib

namespace NUMINAMATH_GPT_south_walk_correct_representation_l330_33052

theorem south_walk_correct_representation {north south : ℤ} (h_north : north = 3) (h_representation : south = -north) : south = -5 :=
by
  have h1 : -north = -3 := by rw [h_north]
  have h2 : -3 = -5 := by sorry
  rw [h_representation, h1]
  exact h2

end NUMINAMATH_GPT_south_walk_correct_representation_l330_33052


namespace NUMINAMATH_GPT_correct_distribution_l330_33092

-- Define the conditions
def num_students : ℕ := 40
def ratio_A_to_B : ℚ := 0.8
def ratio_C_to_B : ℚ := 1.2

-- Definitions for the number of students earning each grade
def num_B (x : ℕ) : ℕ := x
def num_A (x : ℕ) : ℕ := Nat.floor (ratio_A_to_B * x)
def num_C (x : ℕ) : ℕ := Nat.ceil (ratio_C_to_B * x)

-- Prove the distribution is correct
theorem correct_distribution :
  ∃ x : ℕ, num_A x + num_B x + num_C x = num_students ∧ 
           num_A x = 10 ∧ num_B x = 14 ∧ num_C x = 16 :=
by
  sorry

end NUMINAMATH_GPT_correct_distribution_l330_33092


namespace NUMINAMATH_GPT_evaluate_s_squared_plus_c_squared_l330_33034

variable {x y : ℝ}

theorem evaluate_s_squared_plus_c_squared (r : ℝ) (h_r_def : r = Real.sqrt (x^2 + y^2))
                                          (s : ℝ) (h_s_def : s = y / r)
                                          (c : ℝ) (h_c_def : c = x / r) :
  s^2 + c^2 = 1 :=
sorry

end NUMINAMATH_GPT_evaluate_s_squared_plus_c_squared_l330_33034


namespace NUMINAMATH_GPT_percentage_palm_oil_in_cheese_l330_33001

theorem percentage_palm_oil_in_cheese
  (initial_cheese_price: ℝ := 100)
  (cheese_price_increase: ℝ := 3)
  (palm_oil_price_increase_percentage: ℝ := 0.10)
  (expected_palm_oil_percentage : ℝ := 30):
  ∃ (palm_oil_initial_price: ℝ),
  cheese_price_increase = palm_oil_initial_price * palm_oil_price_increase_percentage ∧
  expected_palm_oil_percentage = 100 * (palm_oil_initial_price / initial_cheese_price) := by
  sorry

end NUMINAMATH_GPT_percentage_palm_oil_in_cheese_l330_33001


namespace NUMINAMATH_GPT_simplify_expression_l330_33046

theorem simplify_expression (x y : ℝ) : 3 * y - 5 * x + 2 * y + 4 * x = 5 * y - x :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l330_33046


namespace NUMINAMATH_GPT_avg_of_all_5_is_8_l330_33049

-- Let a1, a2, a3 be three quantities such that their average is 4.
def is_avg_4 (a1 a2 a3 : ℝ) : Prop :=
  (a1 + a2 + a3) / 3 = 4

-- Let a4, a5 be the remaining two quantities such that their average is 14.
def is_avg_14 (a4 a5 : ℝ) : Prop :=
  (a4 + a5) / 2 = 14

-- Prove that the average of all 5 quantities is 8.
theorem avg_of_all_5_is_8 (a1 a2 a3 a4 a5 : ℝ) :
  is_avg_4 a1 a2 a3 ∧ is_avg_14 a4 a5 → 
  ((a1 + a2 + a3 + a4 + a5) / 5 = 8) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_avg_of_all_5_is_8_l330_33049


namespace NUMINAMATH_GPT_expression_equals_thirteen_l330_33067

-- Define the expression
def expression : ℤ :=
    8 + 15 / 3 - 4 * 2 + Nat.pow 2 3

-- State the theorem that proves the value of the expression
theorem expression_equals_thirteen : expression = 13 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_thirteen_l330_33067


namespace NUMINAMATH_GPT_cylinder_volume_eq_l330_33068

variable (α β l : ℝ)

theorem cylinder_volume_eq (hα_pos : 0 < α ∧ α < π/2) (hβ_pos : 0 < β ∧ β < π/2) (hl_pos : 0 < l) :
  let V := (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2)
  V = (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2) :=
by 
  sorry

end NUMINAMATH_GPT_cylinder_volume_eq_l330_33068


namespace NUMINAMATH_GPT_value_of_sum_l330_33091

theorem value_of_sum (x y z : ℝ) 
    (h1 : x + 2*y + 3*z = 10) 
    (h2 : 4*x + 3*y + 2*z = 15) : 
    x + y + z = 5 :=
by
    sorry

end NUMINAMATH_GPT_value_of_sum_l330_33091


namespace NUMINAMATH_GPT_minimum_value_expression_l330_33007

theorem minimum_value_expression (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    3 * a^2 + 2 * b^2 + 4 * c^2 - a * b - 3 * b * c - 5 * c * a ≥ 6 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l330_33007


namespace NUMINAMATH_GPT_max_y_value_l330_33013

theorem max_y_value (x : ℝ) : ∃ y : ℝ, y = -x^2 + 4 * x + 3 ∧ y ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_max_y_value_l330_33013


namespace NUMINAMATH_GPT_part_one_part_two_l330_33058

def f (x a : ℝ) : ℝ :=
  x^2 + a * (abs x) + x 

theorem part_one (x1 x2 a : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) :
  (1 / 2) * (f x1 a + f x2 a) ≥ f ((x1 + x2) / 2) a :=
sorry

theorem part_two (a : ℝ) (ha : 0 ≤ a) (x1 x2 : ℝ) :
  (1 / 2) * (f x1 a + f x2 a) ≥ f ((x1 + x2) / 2) a :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l330_33058


namespace NUMINAMATH_GPT_total_dresses_l330_33027

theorem total_dresses (D M E : ℕ) (h1 : E = 16) (h2 : M = E / 2) (h3 : D = M + 12) : D + M + E = 44 :=
by
  sorry

end NUMINAMATH_GPT_total_dresses_l330_33027


namespace NUMINAMATH_GPT_radius_of_smaller_circle_l330_33088

theorem radius_of_smaller_circle (R r : ℝ) (h1 : R = 6)
  (h2 : 2 * R = 3 * 2 * r) : r = 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_smaller_circle_l330_33088


namespace NUMINAMATH_GPT_correct_option_l330_33029

-- Definitions of the options as Lean statements
def optionA : Prop := (-1 : ℝ) / 6 > (-1 : ℝ) / 7
def optionB : Prop := (-4 : ℝ) / 3 < (-3 : ℝ) / 2
def optionC : Prop := (-2 : ℝ)^3 = -2^3
def optionD : Prop := -(-4.5 : ℝ) > abs (-4.6 : ℝ)

-- Theorem stating that optionC is the correct statement among the provided options
theorem correct_option : optionC :=
by
  unfold optionC
  rw [neg_pow, neg_pow, pow_succ, pow_succ]
  sorry  -- The proof is omitted as per instructions

end NUMINAMATH_GPT_correct_option_l330_33029


namespace NUMINAMATH_GPT_solve_m_n_l330_33051

theorem solve_m_n (m n : ℝ) (h : m^2 + 2 * m + n^2 - 6 * n + 10 = 0) :
  m = -1 ∧ n = 3 :=
sorry

end NUMINAMATH_GPT_solve_m_n_l330_33051


namespace NUMINAMATH_GPT_initial_tiger_sharks_l330_33082

open Nat

theorem initial_tiger_sharks (initial_guppies : ℕ) (initial_angelfish : ℕ) (initial_oscar_fish : ℕ)
  (sold_guppies : ℕ) (sold_angelfish : ℕ) (sold_tiger_sharks : ℕ) (sold_oscar_fish : ℕ)
  (remaining_fish : ℕ) (initial_total_fish : ℕ) (total_guppies_angelfish_oscar : ℕ) (initial_tiger_sharks : ℕ) :
  initial_guppies = 94 → initial_angelfish = 76 → initial_oscar_fish = 58 →
  sold_guppies = 30 → sold_angelfish = 48 → sold_tiger_sharks = 17 → sold_oscar_fish = 24 →
  remaining_fish = 198 →
  initial_total_fish = (sold_guppies + sold_angelfish + sold_tiger_sharks + sold_oscar_fish + remaining_fish) →
  total_guppies_angelfish_oscar = (initial_guppies + initial_angelfish + initial_oscar_fish) →
  initial_tiger_sharks = (initial_total_fish - total_guppies_angelfish_oscar) →
  initial_tiger_sharks = 89 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end NUMINAMATH_GPT_initial_tiger_sharks_l330_33082


namespace NUMINAMATH_GPT_max_length_is_3sqrt2_l330_33084

noncomputable def max_vector_length (θ : ℝ) (h : 0 ≤ θ ∧ θ < 2 * Real.pi) : ℝ :=
  let OP₁ := (Real.cos θ, Real.sin θ)
  let OP₂ := (2 + Real.sin θ, 2 - Real.cos θ)
  let P₁P₂ := (OP₂.1 - OP₁.1, OP₂.2 - OP₁.2)
  Real.sqrt ((P₁P₂.1)^2 + (P₁P₂.2)^2)

theorem max_length_is_3sqrt2 : ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → max_vector_length θ sorry = 3 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_max_length_is_3sqrt2_l330_33084


namespace NUMINAMATH_GPT_greatest_possible_sum_of_two_consecutive_integers_lt_500_l330_33008

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end NUMINAMATH_GPT_greatest_possible_sum_of_two_consecutive_integers_lt_500_l330_33008


namespace NUMINAMATH_GPT_sum_of_roots_l330_33018

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem sum_of_roots (m : ℝ) (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 < 2 * Real.pi)
  (h3 : 0 ≤ x2) (h4 : x2 < 2 * Real.pi) (h_distinct : x1 ≠ x2)
  (h_eq1 : f x1 = m) (h_eq2 : f x2 = m) : x1 + x2 = Real.pi / 2 ∨ x1 + x2 = 5 * Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l330_33018


namespace NUMINAMATH_GPT_thirtieth_entry_satisfies_l330_33076

def r_9 (n : ℕ) : ℕ := n % 9

theorem thirtieth_entry_satisfies (n : ℕ) (h : ∃ k : ℕ, k < 30 ∧ ∀ m < 30, k ≠ m → 
    (r_9 (7 * n + 3) ≤ 4) ∧ 
    ((r_9 (7 * n + 3) ≤ 4) ↔ 
    (r_9 (7 * m + 3) > 4))) :
  n = 37 :=
sorry

end NUMINAMATH_GPT_thirtieth_entry_satisfies_l330_33076


namespace NUMINAMATH_GPT_shaniqua_earnings_l330_33032

noncomputable def shaniqua_total_earnings : ℕ :=
  let haircut_rate := 12
  let style_rate := 25
  let coloring_rate := 35
  let treatment_rate := 50
  let haircuts := 8
  let styles := 5
  let colorings := 10
  let treatments := 6
  (haircuts * haircut_rate) +
  (styles * style_rate) +
  (colorings * coloring_rate) +
  (treatments * treatment_rate)

theorem shaniqua_earnings : shaniqua_total_earnings = 871 := by
  sorry

end NUMINAMATH_GPT_shaniqua_earnings_l330_33032


namespace NUMINAMATH_GPT_maximum_positive_numbers_l330_33079

theorem maximum_positive_numbers (a : ℕ → ℝ) (n : ℕ) (h₀ : n = 100)
  (h₁ : ∀ i : ℕ, 0 < a i) 
  (h₂ : ∀ i : ℕ, a i > a ((i + 1) % n) * a ((i + 2) % n)) : 
  ∃ m : ℕ, m ≤ 50 ∧ (∀ k : ℕ, k < m → (a k) > 0) :=
by sorry

end NUMINAMATH_GPT_maximum_positive_numbers_l330_33079


namespace NUMINAMATH_GPT_equation_three_no_real_roots_l330_33057

theorem equation_three_no_real_roots
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁^2 - 4 ≥ 0)
  (h₂ : a₂^2 - 8 < 0)
  (h₃ : a₂^2 = a₁ * a₃) :
  a₃^2 - 16 < 0 :=
sorry

end NUMINAMATH_GPT_equation_three_no_real_roots_l330_33057


namespace NUMINAMATH_GPT_find_divisor_l330_33066

theorem find_divisor (n k : ℤ) (h1 : n % 30 = 16) : (2 * n) % 30 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l330_33066


namespace NUMINAMATH_GPT_probability_more_than_60000_l330_33031

def boxes : List ℕ := [8, 800, 8000, 40000, 80000]

def probability_keys (keys : ℕ) : ℚ :=
  1 / keys

def probability_winning (n : ℕ) : ℚ :=
  if n = 4 then probability_keys 5 + probability_keys 5 * probability_keys 4 else 0

theorem probability_more_than_60000 : 
  probability_winning 4 = 1/4 := sorry

end NUMINAMATH_GPT_probability_more_than_60000_l330_33031


namespace NUMINAMATH_GPT_cards_value_1_count_l330_33015

/-- There are 4 different suits in a deck of cards containing a total of 52 cards.
  Each suit has 13 cards numbered from 1 to 13.
  Feifei draws 2 hearts, 3 spades, 4 diamonds, and 5 clubs.
  The sum of the face values of these 14 cards is exactly 35.
  Prove that 4 of these cards have a face value of 1. -/
theorem cards_value_1_count :
  ∃ (hearts spades diamonds clubs : List ℕ),
  hearts.length = 2 ∧ spades.length = 3 ∧ diamonds.length = 4 ∧ clubs.length = 5 ∧
  (∀ v, v ∈ hearts → v ∈ List.range 13) ∧ 
  (∀ v, v ∈ spades → v ∈ List.range 13) ∧
  (∀ v, v ∈ diamonds → v ∈ List.range 13) ∧
  (∀ v, v ∈ clubs → v ∈ List.range 13) ∧
  (hearts.sum + spades.sum + diamonds.sum + clubs.sum = 35) ∧
  ((hearts ++ spades ++ diamonds ++ clubs).count 1 = 4) := sorry

end NUMINAMATH_GPT_cards_value_1_count_l330_33015


namespace NUMINAMATH_GPT_find_f_prime_zero_l330_33072

noncomputable def f (a : ℝ) (fd0 : ℝ) (x : ℝ) : ℝ :=
  (a * x^2 + x - 1) * Real.exp x + fd0

theorem find_f_prime_zero (a fd0 : ℝ) : (deriv (f a fd0) 0 = 0) :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_find_f_prime_zero_l330_33072


namespace NUMINAMATH_GPT_romance_movie_tickets_l330_33056

-- Define the given conditions.
def horror_movie_tickets := 93
def relationship (R : ℕ) := 3 * R + 18 = horror_movie_tickets

-- The theorem we need to prove
theorem romance_movie_tickets (R : ℕ) (h : relationship R) : R = 25 :=
by sorry

end NUMINAMATH_GPT_romance_movie_tickets_l330_33056


namespace NUMINAMATH_GPT_number_of_letters_l330_33083

-- Definitions and Conditions, based on the given problem
variables (n : ℕ) -- n is the number of different letters in the local language

-- Given: The people have lost 129 words due to the prohibition of the seventh letter
def words_lost_due_to_prohibition (n : ℕ) : ℕ := 2 * n

-- The main theorem to prove
theorem number_of_letters (h : 129 = words_lost_due_to_prohibition n) : n = 65 :=
by sorry

end NUMINAMATH_GPT_number_of_letters_l330_33083


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l330_33098

-- Problem 1
theorem problem1 (a : ℝ) : -2 * a^3 * 3 * a^2 = -6 * a^5 := 
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) : m^4 * (m^2)^3 / m^8 = m^2 := 
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (-2 * x - 1) * (2 * x - 1) = 1 - 4 * x^2 := 
by
  sorry

-- Problem 4
theorem problem4 (x : ℝ) : (-3 * x + 2)^2 = 9 * x^2 - 12 * x + 4 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l330_33098


namespace NUMINAMATH_GPT_find_a_solution_l330_33025

open Complex

noncomputable def find_a : Prop := 
  ∃ a : ℂ, ((1 + a * I) / (2 + I) = 1 + 2 * I) ∧ (a = 5 + I)

theorem find_a_solution : find_a := 
  by
    sorry

end NUMINAMATH_GPT_find_a_solution_l330_33025


namespace NUMINAMATH_GPT_thirteen_pow_2011_mod_100_l330_33095

theorem thirteen_pow_2011_mod_100 : (13^2011) % 100 = 37 := by
  sorry

end NUMINAMATH_GPT_thirteen_pow_2011_mod_100_l330_33095


namespace NUMINAMATH_GPT_total_number_of_posters_l330_33019

theorem total_number_of_posters : 
  ∀ (P : ℕ), 
  (2 / 5 : ℚ) * P + (1 / 2 : ℚ) * P + 5 = P → 
  P = 50 :=
by
  intro P
  intro h
  sorry

end NUMINAMATH_GPT_total_number_of_posters_l330_33019


namespace NUMINAMATH_GPT_integral_sign_l330_33086

noncomputable def I : ℝ := ∫ x in -Real.pi..0, Real.sin x

theorem integral_sign : I < 0 := sorry

end NUMINAMATH_GPT_integral_sign_l330_33086


namespace NUMINAMATH_GPT_distance_between_sasha_and_kolya_is_19_meters_l330_33006

theorem distance_between_sasha_and_kolya_is_19_meters
  (v_S v_L v_K : ℝ)
  (h1 : v_L = 0.9 * v_S)
  (h2 : v_K = 0.81 * v_S)
  (h3 : ∀ t_S : ℝ, t_S = 100 / v_S) :
  (∀ t_S : ℝ, 100 - v_K * t_S = 19) :=
by
  intros t_S
  have vL_defined : v_L = 0.9 * v_S := h1
  have vK_defined : v_K = 0.81 * v_S := h2
  have time_S : t_S = 100 / v_S := h3 t_S
  sorry

end NUMINAMATH_GPT_distance_between_sasha_and_kolya_is_19_meters_l330_33006


namespace NUMINAMATH_GPT_value_of_y_l330_33080

theorem value_of_y : exists y : ℝ, (∀ k : ℝ, (∀ x y : ℝ, x = k / y^2 → (x = 1 → y = 2 → k = 4)) ∧ (x = 0.1111111111111111 → k = 4 → y = 6)) := by
  sorry

end NUMINAMATH_GPT_value_of_y_l330_33080


namespace NUMINAMATH_GPT_least_value_of_x_l330_33097

theorem least_value_of_x 
  (x : ℕ) 
  (p : ℕ) 
  (hx : 0 < x) 
  (hp : Prime p) 
  (h : x = 2 * 11 * p) : x = 44 := 
by
  sorry

end NUMINAMATH_GPT_least_value_of_x_l330_33097


namespace NUMINAMATH_GPT_linear_equation_a_ne_1_l330_33041

theorem linear_equation_a_ne_1 (a : ℝ) : (∀ x : ℝ, (a - 1) * x - 6 = 0 → a ≠ 1) :=
sorry

end NUMINAMATH_GPT_linear_equation_a_ne_1_l330_33041


namespace NUMINAMATH_GPT_coin_flip_probability_l330_33085

def total_outcomes := 2^6
def favorable_outcomes := 2^3
def probability := favorable_outcomes / total_outcomes

theorem coin_flip_probability :
  probability = 1 / 8 :=
by
  unfold probability total_outcomes favorable_outcomes
  sorry

end NUMINAMATH_GPT_coin_flip_probability_l330_33085


namespace NUMINAMATH_GPT_probability_of_B_given_A_l330_33024

noncomputable def balls_in_box : Prop :=
  let total_balls := 12
  let yellow_balls := 5
  let blue_balls := 4
  let green_balls := 3
  let event_A := (yellow_balls * green_balls + yellow_balls * blue_balls + green_balls * blue_balls) / (total_balls * (total_balls - 1) / 2)
  let event_B := (yellow_balls * blue_balls) / (total_balls * (total_balls - 1) / 2)
  (event_B / event_A) = 20 / 47

theorem probability_of_B_given_A : balls_in_box := sorry

end NUMINAMATH_GPT_probability_of_B_given_A_l330_33024


namespace NUMINAMATH_GPT_area_of_rectangle_A_is_88_l330_33038

theorem area_of_rectangle_A_is_88 
  (lA lB lC w wC : ℝ)
  (h1 : lB = lA + 2)
  (h2 : lB * w = lA * w + 22)
  (h3 : wC = w - 4)
  (AreaB : ℝ := lB * w)
  (AreaC : ℝ := lB * wC)
  (h4 : AreaC = AreaB - 40) : 
  (lA * w = 88) :=
sorry

end NUMINAMATH_GPT_area_of_rectangle_A_is_88_l330_33038


namespace NUMINAMATH_GPT_value_of_polynomial_l330_33071

theorem value_of_polynomial :
  98^3 + 3 * (98^2) + 3 * 98 + 1 = 970299 :=
by sorry

end NUMINAMATH_GPT_value_of_polynomial_l330_33071


namespace NUMINAMATH_GPT_a_n_is_perfect_square_l330_33028

def sequence_c (n : ℕ) : ℤ :=
  if n = 0 then 1
  else if n = 1 then 0
  else if n = 2 then 2005
  else -3 * sequence_c (n - 2) - 4 * sequence_c (n - 3) + 2008

def sequence_a (n : ℕ) :=
  if n < 2 then 0
  else 5 * (sequence_c (n + 2) - sequence_c n) * (502 - sequence_c (n - 1) - sequence_c (n - 2)) + (4 ^ n) * 2004 * 501

theorem a_n_is_perfect_square (n : ℕ) (h : n > 2) : ∃ k : ℤ, sequence_a n = k^2 :=
sorry

end NUMINAMATH_GPT_a_n_is_perfect_square_l330_33028


namespace NUMINAMATH_GPT_sum_of_highest_powers_of_10_and_6_dividing_20_factorial_l330_33094

def legendre (n p : Nat) : Nat :=
  if p > 1 then (Nat.div n p + Nat.div n (p * p) + Nat.div n (p * p * p) + Nat.div n (p * p * p * p)) else 0

theorem sum_of_highest_powers_of_10_and_6_dividing_20_factorial :
  let highest_power_5 := legendre 20 5
  let highest_power_2 := legendre 20 2
  let highest_power_3 := legendre 20 3
  let highest_power_10 := min highest_power_2 highest_power_5
  let highest_power_6 := min highest_power_2 highest_power_3
  highest_power_10 + highest_power_6 = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_highest_powers_of_10_and_6_dividing_20_factorial_l330_33094


namespace NUMINAMATH_GPT_right_triangle_shorter_leg_l330_33012

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_shorter_leg_l330_33012


namespace NUMINAMATH_GPT_number_subtracted_l330_33036

theorem number_subtracted (x : ℝ) : 3 + 2 * (8 - x) = 24.16 → x = -2.58 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_subtracted_l330_33036


namespace NUMINAMATH_GPT_pooh_piglet_cake_sharing_l330_33077

theorem pooh_piglet_cake_sharing (a b : ℚ) (h1 : a + b = 1) (h2 : b + a/3 = 3*b) : 
  a = 6/7 ∧ b = 1/7 :=
by
  sorry

end NUMINAMATH_GPT_pooh_piglet_cake_sharing_l330_33077


namespace NUMINAMATH_GPT_sum_of_altitudes_of_triangle_l330_33021

theorem sum_of_altitudes_of_triangle (a b c : ℝ) (h_line : ∀ x y, 8 * x + 10 * y = 80 → x = 10 ∨ y = 8) :
  (8 + 10 + 40/Real.sqrt 41) = 18 + 40/Real.sqrt 41 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_altitudes_of_triangle_l330_33021


namespace NUMINAMATH_GPT_reflect_over_x_axis_reflect_over_y_axis_l330_33065

-- Mathematical Definitions
def Point := (ℝ × ℝ)

-- Reflect a point over the x-axis
def reflectOverX (M : Point) : Point :=
  (M.1, -M.2)

-- Reflect a point over the y-axis
def reflectOverY (M : Point) : Point :=
  (-M.1, M.2)

-- Theorem statements
theorem reflect_over_x_axis (M : Point) : reflectOverX M = (M.1, -M.2) :=
by
  sorry

theorem reflect_over_y_axis (M : Point) : reflectOverY M = (-M.1, M.2) :=
by
  sorry

end NUMINAMATH_GPT_reflect_over_x_axis_reflect_over_y_axis_l330_33065


namespace NUMINAMATH_GPT_calculate_total_area_l330_33033

theorem calculate_total_area :
  let height1 := 7
  let width1 := 6
  let width2 := 4
  let height2 := 5
  let height3 := 1
  let width3 := 2
  let width4 := 5
  let height4 := 6
  let area1 := width1 * height1
  let area2 := width2 * height2
  let area3 := height3 * width3
  let area4 := width4 * height4
  area1 + area2 + area3 + area4 = 94 := by
  sorry

end NUMINAMATH_GPT_calculate_total_area_l330_33033


namespace NUMINAMATH_GPT_part_1_part_2_1_part_2_2_l330_33047

variable {k x : ℝ}
def y (k : ℝ) (x : ℝ) := k * x^2 - 2 * k * x + 2 * k - 1

theorem part_1 (k : ℝ) : (∀ x, y k x ≥ 4 * k - 2) ↔ (0 ≤ k ∧ k ≤ 1 / 3) := by
  sorry

theorem part_2_1 (k : ℝ) : ¬∃ x1 x2 : ℝ, y k x = 0 ∧ y k x = 0 ∧ x1^2 + x2^2 = 3 * x1 * x2 - 4 := by
  sorry

theorem part_2_2 (k : ℝ) : (∀ x1 x2 : ℝ, y k x = 0 ∧ y k x = 0 ∧ x1 > 0 ∧ x2 > 0) ↔ (1 / 2 < k ∧ k < 1) := by
  sorry

end NUMINAMATH_GPT_part_1_part_2_1_part_2_2_l330_33047


namespace NUMINAMATH_GPT_sum_of_edges_of_geometric_progression_solid_l330_33090

theorem sum_of_edges_of_geometric_progression_solid
  (a : ℝ)
  (r : ℝ)
  (volume_eq : a^3 = 512)
  (surface_eq : 2 * (64 / r + 64 * r + 64) = 352)
  (r_value : r = 1.25 ∨ r = 0.8) :
  4 * (8 / r + 8 + 8 * r) = 97.6 := by
  sorry

end NUMINAMATH_GPT_sum_of_edges_of_geometric_progression_solid_l330_33090


namespace NUMINAMATH_GPT_line_perp_to_plane_imp_perp_to_line_l330_33063

def Line := Type
def Plane := Type

variables (m n : Line) (α : Plane)

def is_parallel (l : Line) (p : Plane) : Prop := sorry
def is_perpendicular (l1 l2 : Line) : Prop := sorry
def is_contained (l : Line) (p : Plane) : Prop := sorry

theorem line_perp_to_plane_imp_perp_to_line :
  (is_perpendicular m α) ∧ (is_contained n α) → (is_perpendicular m n) :=
sorry

end NUMINAMATH_GPT_line_perp_to_plane_imp_perp_to_line_l330_33063


namespace NUMINAMATH_GPT_min_value_expr_l330_33069

theorem min_value_expr (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) : 
  (∃ m : ℝ, (∀ x y : ℝ, x^2 + y^2 = 2 ∧ |x| ≠ |y| → m ≤ (1 / (x + y)^2 + 1 / (x - y)^2)) ∧ m = 1) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l330_33069


namespace NUMINAMATH_GPT_least_n_l330_33042

theorem least_n (n : ℕ) (h_pos : n > 0) (h_ineq : 1 / n - 1 / (n + 1) < 1 / 15) : n = 4 :=
sorry

end NUMINAMATH_GPT_least_n_l330_33042


namespace NUMINAMATH_GPT_largest_divisor_l330_33043

theorem largest_divisor (n : ℕ) (h1 : 0 < n) (h2 : 450 ∣ n ^ 2) : 30 ∣ n :=
sorry

end NUMINAMATH_GPT_largest_divisor_l330_33043


namespace NUMINAMATH_GPT_aladdin_can_find_heavy_coins_l330_33096

theorem aladdin_can_find_heavy_coins :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20 ∧ x ≠ y ∧ (x + y ≥ 28) :=
by
  sorry

end NUMINAMATH_GPT_aladdin_can_find_heavy_coins_l330_33096


namespace NUMINAMATH_GPT_orange_gumdrops_after_replacement_l330_33045

noncomputable def total_gumdrops : ℕ :=
  100

noncomputable def initial_orange_gumdrops : ℕ :=
  10

noncomputable def initial_blue_gumdrops : ℕ :=
  40

noncomputable def replaced_blue_gumdrops : ℕ :=
  initial_blue_gumdrops / 3

theorem orange_gumdrops_after_replacement : 
  (initial_orange_gumdrops + replaced_blue_gumdrops) = 23 :=
by
  sorry

end NUMINAMATH_GPT_orange_gumdrops_after_replacement_l330_33045


namespace NUMINAMATH_GPT_smallest_value_of_sum_l330_33000

theorem smallest_value_of_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : 3 * a = 4 * b ∧ 4 * b = 7 * c) : a + b + c = 61 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_sum_l330_33000


namespace NUMINAMATH_GPT_intersection_point_of_lines_l330_33014

theorem intersection_point_of_lines (n : ℕ) (x y : ℤ) :
  15 * x + 18 * y = 1005 ∧ y = n * x + 2 → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l330_33014


namespace NUMINAMATH_GPT_number_of_two_point_safeties_l330_33040

variables (f g s : ℕ)

theorem number_of_two_point_safeties (h1 : 4 * f = 6 * g) 
                                    (h2 : s = g + 2) 
                                    (h3 : 4 * f + 3 * g + 2 * s = 50) : 
                                    s = 6 := 
by sorry

end NUMINAMATH_GPT_number_of_two_point_safeties_l330_33040


namespace NUMINAMATH_GPT_train_length_l330_33064

theorem train_length (V L : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 600.0000000000001 = V * 54) : 
  L = 300.00000000000005 :=
by 
  sorry

end NUMINAMATH_GPT_train_length_l330_33064


namespace NUMINAMATH_GPT_find_range_m_l330_33087

noncomputable def p (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, m * x₀^2 + 1 < 1

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

theorem find_range_m (m : ℝ) : ¬ (p m ∨ ¬ q m) ↔ -2 ≤ m ∧ m ≤ 2 :=
  sorry

end NUMINAMATH_GPT_find_range_m_l330_33087


namespace NUMINAMATH_GPT_points_A_B_D_collinear_l330_33022

variable (a b : ℝ)

theorem points_A_B_D_collinear
  (AB : ℝ × ℝ := (a, 5 * b))
  (BC : ℝ × ℝ := (-2 * a, 8 * b))
  (CD : ℝ × ℝ := (3 * a, -3 * b)) :
  AB = (BC.1 + CD.1, BC.2 + CD.2) := 
by
  sorry

end NUMINAMATH_GPT_points_A_B_D_collinear_l330_33022


namespace NUMINAMATH_GPT_evaluate_fraction_l330_33039

theorem evaluate_fraction (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - b * (1 / a) ≠ 0) :
  (a^2 - 1 / b^2) / (b^2 - 1 / a^2) = a^2 / b^2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l330_33039


namespace NUMINAMATH_GPT_incorrect_ac_bc_impl_a_b_l330_33026

theorem incorrect_ac_bc_impl_a_b : ∀ (a b c : ℝ), (ac = bc → a = b) ↔ c ≠ 0 :=
by sorry

end NUMINAMATH_GPT_incorrect_ac_bc_impl_a_b_l330_33026


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l330_33020

theorem no_positive_integer_solutions : ¬∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ^ 4004 + y ^ 4004 = z ^ 2002 :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l330_33020


namespace NUMINAMATH_GPT_number_of_A_items_number_of_A_proof_l330_33054

def total_items : ℕ := 600
def ratio_A_B_C := (1, 2, 3)
def selected_items : ℕ := 120

theorem number_of_A_items (total_items : ℕ) (selected_items : ℕ) (rA rB rC : ℕ) (ratio_proof : rA + rB + rC = 6) : ℕ :=
  let total_ratio := rA + rB + rC
  let A_ratio := rA
  (selected_items * A_ratio) / total_ratio

theorem number_of_A_proof : number_of_A_items total_items selected_items 1 2 3 (rfl) = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_A_items_number_of_A_proof_l330_33054


namespace NUMINAMATH_GPT_kendra_change_is_correct_l330_33017

-- Define the initial conditions
def price_wooden_toy : ℕ := 20
def price_hat : ℕ := 10
def kendra_initial_money : ℕ := 100
def num_wooden_toys : ℕ := 2
def num_hats : ℕ := 3

-- Calculate the total costs
def total_wooden_toys_cost : ℕ := price_wooden_toy * num_wooden_toys
def total_hats_cost : ℕ := price_hat * num_hats
def total_cost : ℕ := total_wooden_toys_cost + total_hats_cost

-- Calculate the change Kendra received
def kendra_change : ℕ := kendra_initial_money - total_cost

theorem kendra_change_is_correct : kendra_change = 30 := by
  sorry

end NUMINAMATH_GPT_kendra_change_is_correct_l330_33017


namespace NUMINAMATH_GPT_inscribed_circle_radius_l330_33075

theorem inscribed_circle_radius (R r x : ℝ) (hR : R = 18) (hr : r = 9) :
  x = 8 :=
sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l330_33075


namespace NUMINAMATH_GPT_train_speed_is_252_144_l330_33073

/-- Train and pedestrian problem setup -/
noncomputable def train_speed (train_length : ℕ) (cross_time : ℕ) (man_speed_kmph : ℕ) : ℝ :=
  let man_speed_mps := (man_speed_kmph : ℝ) * 1000 / 3600
  let relative_speed_mps := (train_length : ℝ) / (cross_time : ℝ)
  let train_speed_mps := relative_speed_mps - man_speed_mps
  train_speed_mps * 3600 / 1000

theorem train_speed_is_252_144 :
  train_speed 500 7 5 = 252.144 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_252_144_l330_33073


namespace NUMINAMATH_GPT_coopers_age_l330_33010

theorem coopers_age (C D M E : ℝ) 
  (h1 : D = 2 * C) 
  (h2 : M = 2 * C + 1) 
  (h3 : E = 3 * C)
  (h4 : C + D + M + E = 62) : 
  C = 61 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_coopers_age_l330_33010


namespace NUMINAMATH_GPT_find_y_l330_33081

theorem find_y (x y : ℝ) (h1 : 9823 + x = 13200) (h2 : x = y / 3 + 37.5) : y = 10018.5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l330_33081


namespace NUMINAMATH_GPT_even_function_value_sum_l330_33099

noncomputable def g (x : ℝ) (d e f : ℝ) : ℝ :=
  d * x^8 - e * x^6 + f * x^2 + 5

theorem even_function_value_sum (d e f : ℝ) (h : g 15 d e f = 7) :
  g 15 d e f + g (-15) d e f = 14 := by
  sorry

end NUMINAMATH_GPT_even_function_value_sum_l330_33099


namespace NUMINAMATH_GPT_production_equation_l330_33059

-- Definitions based on the problem conditions
def original_production_rate (x : ℕ) := x
def additional_parts_per_day := 4
def original_days := 20
def actual_days := 15
def extra_parts := 10

-- Prove the equation
theorem production_equation (x : ℕ) :
  original_days * original_production_rate x = actual_days * (original_production_rate x + additional_parts_per_day) - extra_parts :=
by
  simp [original_production_rate, additional_parts_per_day, original_days, actual_days, extra_parts]
  sorry

end NUMINAMATH_GPT_production_equation_l330_33059


namespace NUMINAMATH_GPT_robin_candy_consumption_l330_33002

theorem robin_candy_consumption (x : ℕ) : 23 - x + 21 = 37 → x = 7 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_robin_candy_consumption_l330_33002


namespace NUMINAMATH_GPT_total_bananas_in_collection_l330_33023

-- Definitions based on the conditions
def group_size : ℕ := 18
def number_of_groups : ℕ := 10

-- The proof problem statement
theorem total_bananas_in_collection : group_size * number_of_groups = 180 := by
  sorry

end NUMINAMATH_GPT_total_bananas_in_collection_l330_33023


namespace NUMINAMATH_GPT_circumradius_relation_l330_33005

-- Definitions of the geometric constructs from the problem
open EuclideanGeometry

noncomputable def circumradius (A B C : Point) : Real := sorry

-- Given conditions
def angle_bisectors_intersect_at_point (A B C B1 C1 I : Point) : Prop := sorry
def line_intersects_circumcircle_at_points (B1 C1 : Point) (circumcircle : Circle) (M N : Point) : Prop := sorry

-- Main statement to prove
theorem circumradius_relation
  (A B C B1 C1 I M N : Point)
  (circumcircle : Circle)
  (h1 : angle_bisectors_intersect_at_point A B C B1 C1 I)
  (h2 : line_intersects_circumcircle_at_points B1 C1 circumcircle M N) :
  circumradius M I N = 2 * circumradius A B C :=
sorry

end NUMINAMATH_GPT_circumradius_relation_l330_33005


namespace NUMINAMATH_GPT_average_percent_increase_in_profit_per_car_l330_33074

theorem average_percent_increase_in_profit_per_car
  (N P : ℝ) -- N: Number of cars sold last year, P: Profit per car last year
  (HP1 : N > 0) -- Non-zero number of cars
  (HP2 : P > 0) -- Non-zero profit
  (HProfitIncrease : 1.3 * (N * P) = 1.3 * N * P) -- Total profit increased by 30%
  (HCarDecrease : 0.7 * N = 0.7 * N) -- Number of cars decreased by 30%
  : ((1.3 / 0.7) - 1) * 100 = 85.7 := sorry

end NUMINAMATH_GPT_average_percent_increase_in_profit_per_car_l330_33074


namespace NUMINAMATH_GPT_percentage_of_8thgraders_correct_l330_33089

def total_students_oakwood : ℕ := 150
def total_students_pinecrest : ℕ := 250

def percent_8thgraders_oakwood : ℕ := 60
def percent_8thgraders_pinecrest : ℕ := 55

def number_of_8thgraders_oakwood : ℚ := (percent_8thgraders_oakwood * total_students_oakwood) / 100
def number_of_8thgraders_pinecrest : ℚ := (percent_8thgraders_pinecrest * total_students_pinecrest) / 100

def total_number_of_8thgraders : ℚ := number_of_8thgraders_oakwood + number_of_8thgraders_pinecrest
def total_number_of_students : ℕ := total_students_oakwood + total_students_pinecrest

def percent_8thgraders_combined : ℚ := (total_number_of_8thgraders / total_number_of_students) * 100

theorem percentage_of_8thgraders_correct : percent_8thgraders_combined = 57 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_8thgraders_correct_l330_33089


namespace NUMINAMATH_GPT_range_of_expression_positive_range_of_expression_negative_l330_33060

theorem range_of_expression_positive (x : ℝ) : 
  (2 * x ^ 2 - 5 * x - 12 > 0) ↔ (x < -3/2 ∨ x > 4) :=
sorry

theorem range_of_expression_negative (x : ℝ) : 
  (2 * x ^ 2 - 5 * x - 12 < 0) ↔ ( -3/2 < x ∧ x < 4) :=
sorry

end NUMINAMATH_GPT_range_of_expression_positive_range_of_expression_negative_l330_33060


namespace NUMINAMATH_GPT_matching_charge_and_minutes_l330_33035

def charge_at_time (x : ℕ) : ℕ :=
  100 - x / 6

def minutes_past_midnight (x : ℕ) : ℕ :=
  x % 60

theorem matching_charge_and_minutes :
  ∃ x, (x = 292 ∨ x = 343 ∨ x = 395 ∨ x = 446 ∨ x = 549) ∧ 
       charge_at_time x = minutes_past_midnight x :=
by {
  sorry
}

end NUMINAMATH_GPT_matching_charge_and_minutes_l330_33035


namespace NUMINAMATH_GPT_smallest_EF_minus_DE_l330_33004

theorem smallest_EF_minus_DE (x y z : ℕ) (h1 : x < y) (h2 : y ≤ z) (h3 : x + y + z = 2050)
  (h4 : x + y > z) (h5 : y + z > x) (h6 : z + x > y) : y - x = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_EF_minus_DE_l330_33004


namespace NUMINAMATH_GPT_proposition_A_iff_proposition_B_l330_33016

-- Define propositions
def Proposition_A (A B C : ℕ) : Prop := (A = 60 ∨ B = 60 ∨ C = 60)
def Proposition_B (A B C : ℕ) : Prop :=
  (A + B + C = 180) ∧ 
  (2 * B = A + C)

-- The theorem stating the relationship between Proposition_A and Proposition_B
theorem proposition_A_iff_proposition_B (A B C : ℕ) :
  Proposition_A A B C ↔ Proposition_B A B C :=
sorry

end NUMINAMATH_GPT_proposition_A_iff_proposition_B_l330_33016


namespace NUMINAMATH_GPT_arithmetic_mean_calc_l330_33003

theorem arithmetic_mean_calc (x a : ℝ) (hx : x ≠ 0) (ha : a ≠ 0) :
  ( ( (x + a)^2 / x ) + ( (x - a)^2 / x ) ) / 2 = x + (a^2 / x) :=
sorry

end NUMINAMATH_GPT_arithmetic_mean_calc_l330_33003


namespace NUMINAMATH_GPT_car_travel_distance_l330_33062

-- Definitions of conditions
def speed_kmph : ℝ := 27 -- 27 kilometers per hour
def time_sec : ℝ := 50 -- 50 seconds

-- Equivalent in Lean 4 for car moving distance in meters
theorem car_travel_distance : (speed_kmph * 1000 / 3600) * time_sec = 375 := by
  sorry

end NUMINAMATH_GPT_car_travel_distance_l330_33062


namespace NUMINAMATH_GPT_max_mn_on_parabola_l330_33053

theorem max_mn_on_parabola :
  ∀ m n : ℝ, (n = -m^2 + 3) → (m + n ≤ 13 / 4) :=
by
  sorry

end NUMINAMATH_GPT_max_mn_on_parabola_l330_33053


namespace NUMINAMATH_GPT_multiplication_of_exponents_l330_33050

theorem multiplication_of_exponents (x : ℝ) : (x ^ 4) * (x ^ 2) = x ^ 6 := 
by
  sorry

end NUMINAMATH_GPT_multiplication_of_exponents_l330_33050


namespace NUMINAMATH_GPT_recliner_price_drop_l330_33011

theorem recliner_price_drop
  (P : ℝ) (N : ℝ)
  (N' : ℝ := 1.8 * N)
  (G : ℝ := P * N)
  (G' : ℝ := 1.44 * G) :
  (P' : ℝ) → P' = 0.8 * P → (P - P') / P * 100 = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_recliner_price_drop_l330_33011


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l330_33078

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 1) (h2 : x * y = -4) : x^2 + y^2 = 9 :=
sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l330_33078


namespace NUMINAMATH_GPT_B_pow_5_eq_r_B_add_s_I_l330_33044

def B : Matrix (Fin 2) (Fin 2) ℤ := ![![ -2,  3 ], 
                                      ![  4,  5 ]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem B_pow_5_eq_r_B_add_s_I :
  ∃ r s : ℤ, (r = 425) ∧ (s = 780) ∧ (B^5 = r • B + s • I) :=
by
  sorry

end NUMINAMATH_GPT_B_pow_5_eq_r_B_add_s_I_l330_33044


namespace NUMINAMATH_GPT_point_in_plane_region_l330_33037

theorem point_in_plane_region :
  (2 * 0 + 1 - 6 < 0) ∧ ¬(2 * 5 + 0 - 6 < 0) ∧ ¬(2 * 0 + 7 - 6 < 0) ∧ ¬(2 * 2 + 3 - 6 < 0) :=
by
  -- Proof detail goes here.
  sorry

end NUMINAMATH_GPT_point_in_plane_region_l330_33037


namespace NUMINAMATH_GPT_hyperbola_sufficient_condition_l330_33061

-- Define the condition for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  (3 - k) * (k - 1) < 0

-- Lean 4 statement to prove that k > 3 is a sufficient condition for the given equation
theorem hyperbola_sufficient_condition (k : ℝ) (h : k > 3) :
  represents_hyperbola k :=
sorry

end NUMINAMATH_GPT_hyperbola_sufficient_condition_l330_33061


namespace NUMINAMATH_GPT_value_expression_eq_zero_l330_33093

theorem value_expression_eq_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
    (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
    a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_expression_eq_zero_l330_33093


namespace NUMINAMATH_GPT_total_earnings_l330_33070

def oil_change_cost : ℕ := 20
def repair_cost : ℕ := 30
def car_wash_cost : ℕ := 5

def num_oil_changes : ℕ := 5
def num_repairs : ℕ := 10
def num_car_washes : ℕ := 15

theorem total_earnings :
  (num_oil_changes * oil_change_cost) +
  (num_repairs * repair_cost) +
  (num_car_washes * car_wash_cost) = 475 :=
by
  sorry

end NUMINAMATH_GPT_total_earnings_l330_33070


namespace NUMINAMATH_GPT_yella_computer_usage_difference_l330_33009

-- Define the given conditions
def last_week_usage : ℕ := 91
def this_week_daily_usage : ℕ := 8
def days_in_week : ℕ := 7

-- Compute this week's total usage
def this_week_total_usage := this_week_daily_usage * days_in_week

-- Statement to prove
theorem yella_computer_usage_difference :
  last_week_usage - this_week_total_usage = 35 := 
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_yella_computer_usage_difference_l330_33009


namespace NUMINAMATH_GPT_ticket_value_unique_l330_33048

theorem ticket_value_unique (x : ℕ) (h₁ : ∃ n, n > 0 ∧ x * n = 60)
  (h₂ : ∃ m, m > 0 ∧ x * m = 90)
  (h₃ : ∃ p, p > 0 ∧ x * p = 49) : 
  ∃! x, x = 1 :=
by
  sorry

end NUMINAMATH_GPT_ticket_value_unique_l330_33048


namespace NUMINAMATH_GPT_regression_analysis_incorrect_statement_l330_33055

theorem regression_analysis_incorrect_statement
  (y : ℕ → ℝ) (x : ℕ → ℝ) (b a : ℝ)
  (r : ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ)
  (H1 : ∀ i, y i = b * x i + a)
  (H2 : abs r = 1 → ∀ x1 x2, l x1 = l x2 → x1 = x2)
  (H3 : ∃ m k, ∀ x, l x = m * x + k)
  (H4 : P.1 = b → l P.1 = P.2)
  (cond_A : ∀ i, y i ≠ b * x i + a) : false := 
sorry

end NUMINAMATH_GPT_regression_analysis_incorrect_statement_l330_33055


namespace NUMINAMATH_GPT_combination_identity_l330_33030

theorem combination_identity : (Nat.choose 5 3 + Nat.choose 5 4 = Nat.choose 6 4) := 
by 
  sorry

end NUMINAMATH_GPT_combination_identity_l330_33030
