import Mathlib

namespace NUMINAMATH_GPT_base6_addition_sum_l1103_110374

theorem base6_addition_sum 
  (P Q R : ℕ) 
  (h1 : P ≠ Q) 
  (h2 : Q ≠ R) 
  (h3 : P ≠ R) 
  (h4 : P < 6) 
  (h5 : Q < 6) 
  (h6 : R < 6) 
  (h7 : 2*R % 6 = P) 
  (h8 : 2*Q % 6 = R)
  : P + Q + R = 7 := 
  sorry

end NUMINAMATH_GPT_base6_addition_sum_l1103_110374


namespace NUMINAMATH_GPT_virginia_initial_eggs_l1103_110383

theorem virginia_initial_eggs (final_eggs : ℕ) (taken_eggs : ℕ) (H : final_eggs = 93) (G : taken_eggs = 3) : final_eggs + taken_eggs = 96 := 
by
  -- proof part could go here
  sorry

end NUMINAMATH_GPT_virginia_initial_eggs_l1103_110383


namespace NUMINAMATH_GPT_general_rule_equation_l1103_110303

theorem general_rule_equation (n : ℕ) (hn : n > 0) : (n + 1) / n + (n + 1) = (n + 2) + 1 / n :=
by
  sorry

end NUMINAMATH_GPT_general_rule_equation_l1103_110303


namespace NUMINAMATH_GPT_find_k_l1103_110381

def line_p (x y : ℝ) : Prop := y = -2 * x + 3
def line_q (x y k : ℝ) : Prop := y = k * x + 4
def intersection (x y k : ℝ) : Prop := line_p x y ∧ line_q x y k

theorem find_k (k : ℝ) (h_inter : intersection 1 1 k) : k = -3 :=
sorry

end NUMINAMATH_GPT_find_k_l1103_110381


namespace NUMINAMATH_GPT_det_of_matrix_l1103_110389

def determinant_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem det_of_matrix :
  determinant_2x2 5 (-2) 3 1 = 11 := by
  sorry

end NUMINAMATH_GPT_det_of_matrix_l1103_110389


namespace NUMINAMATH_GPT_rational_root_theorem_l1103_110310

theorem rational_root_theorem :
  (∃ x : ℚ, 3 * x^4 - 4 * x^3 - 10 * x^2 + 8 * x + 3 = 0)
  → (x = 1 ∨ x = 1/3) := by
  sorry

end NUMINAMATH_GPT_rational_root_theorem_l1103_110310


namespace NUMINAMATH_GPT_sum_of_xy_is_1289_l1103_110305

-- Define the variables and conditions
def internal_angle1 (x y : ℕ) : ℕ := 5 * x + 3 * y
def internal_angle2 (x y : ℕ) : ℕ := 3 * x + 20
def internal_angle3 (x y : ℕ) : ℕ := 10 * y + 30

-- Definition of the sum of angles of a triangle
def sum_of_angles (x y : ℕ) : ℕ := internal_angle1 x y + internal_angle2 x y + internal_angle3 x y

-- Define the theorem statement
theorem sum_of_xy_is_1289 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h : sum_of_angles x y = 180) : x + y = 1289 :=
by sorry

end NUMINAMATH_GPT_sum_of_xy_is_1289_l1103_110305


namespace NUMINAMATH_GPT_polygon_interior_angle_sum_360_l1103_110371

theorem polygon_interior_angle_sum_360 (n : ℕ) (h : (n-2) * 180 = 360) : n = 4 :=
sorry

end NUMINAMATH_GPT_polygon_interior_angle_sum_360_l1103_110371


namespace NUMINAMATH_GPT_table_tennis_matches_l1103_110332

def num_players : ℕ := 8

def total_matches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem table_tennis_matches : total_matches num_players = 28 := by
  sorry

end NUMINAMATH_GPT_table_tennis_matches_l1103_110332


namespace NUMINAMATH_GPT_problem_l1103_110362

noncomputable def f (A B x : ℝ) : ℝ := A * x^2 + B
noncomputable def g (A B x : ℝ) : ℝ := B * x^2 + A

theorem problem (A B x : ℝ) (h : A ≠ B) 
  (h1 : f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A + B = 0 := 
  sorry

end NUMINAMATH_GPT_problem_l1103_110362


namespace NUMINAMATH_GPT_triangle_area_l1103_110359

theorem triangle_area :
  ∀ (x y : ℝ), (3 * x + 2 * y = 12 ∧ x ≥ 0 ∧ y ≥ 0) →
  (1 / 2) * 4 * 6 = 12 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l1103_110359


namespace NUMINAMATH_GPT_power_of_i_l1103_110311

theorem power_of_i (i : ℂ) 
  (h1: i^1 = i) 
  (h2: i^2 = -1) 
  (h3: i^3 = -i) 
  (h4: i^4 = 1)
  (h5: i^5 = i) 
  : i^2016 = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_power_of_i_l1103_110311


namespace NUMINAMATH_GPT_fifth_term_is_67_l1103_110329

noncomputable def satisfies_sequence (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :=
  (a = 3) ∧ (d = 27) ∧ 
  (a = (1/3 : ℚ) * (3 + b)) ∧
  (b = (1/3 : ℚ) * (a + 27)) ∧
  (27 = (1/3 : ℚ) * (b + e))

theorem fifth_term_is_67 :
  ∃ (e : ℕ), satisfies_sequence 3 a b 27 e ∧ e = 67 :=
sorry

end NUMINAMATH_GPT_fifth_term_is_67_l1103_110329


namespace NUMINAMATH_GPT_inequality_min_m_l1103_110355

theorem inequality_min_m (m : ℝ) (x : ℝ) (hx : 1 < x) : 
  x + m * Real.log x + 1 / Real.exp x ≥ Real.exp (m * Real.log x) :=
sorry

end NUMINAMATH_GPT_inequality_min_m_l1103_110355


namespace NUMINAMATH_GPT_sin_sub_pi_over_3_eq_neg_one_third_l1103_110351

theorem sin_sub_pi_over_3_eq_neg_one_third {x : ℝ} (h : Real.cos (x + (π / 6)) = 1 / 3) :
  Real.sin (x - (π / 3)) = -1 / 3 := 
  sorry

end NUMINAMATH_GPT_sin_sub_pi_over_3_eq_neg_one_third_l1103_110351


namespace NUMINAMATH_GPT_min_moves_to_emit_all_colors_l1103_110328

theorem min_moves_to_emit_all_colors :
  ∀ (colors : Fin 7 → Prop) (room : Fin 4 → Fin 7)
  (h : ∀ i j, i ≠ j → room i ≠ room j) (moves : ℕ),
  (∀ (n : ℕ) (i : Fin 4), n < moves → ∃ c : Fin 7, colors c ∧ room i = c ∧
    (∀ j, j ≠ i → room j ≠ c)) →
  (∃ n, n = 8) :=
by
  sorry

end NUMINAMATH_GPT_min_moves_to_emit_all_colors_l1103_110328


namespace NUMINAMATH_GPT_smallest_value_of_Q_l1103_110331

def Q (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

noncomputable def A := Q (-1)
noncomputable def B := Q (0)
noncomputable def C := (2 : ℝ)^2
def D := 1 - 2 + 3 - 4 + 5
def E := 2 -- assuming all zeros are real

theorem smallest_value_of_Q :
  min (min (min (min A B) C) D) E = 2 :=
by sorry

end NUMINAMATH_GPT_smallest_value_of_Q_l1103_110331


namespace NUMINAMATH_GPT_length_to_width_ratio_is_three_l1103_110375

def rectangle_ratio (x : ℝ) : Prop :=
  let side_length_large_square := 4 * x
  let length_rectangle := 4 * x
  let width_rectangle := x
  length_rectangle / width_rectangle = 3

-- We state the theorem to be proved
theorem length_to_width_ratio_is_three (x : ℝ) (h : 0 < x) :
  rectangle_ratio x :=
sorry

end NUMINAMATH_GPT_length_to_width_ratio_is_three_l1103_110375


namespace NUMINAMATH_GPT_greatest_monthly_drop_is_march_l1103_110322

-- Define the price changes for each month
def price_change_january : ℝ := -0.75
def price_change_february : ℝ := 1.50
def price_change_march : ℝ := -3.00
def price_change_april : ℝ := 2.50
def price_change_may : ℝ := -1.00
def price_change_june : ℝ := 0.50
def price_change_july : ℝ := -2.50

-- Prove that the month with the greatest drop in price is March
theorem greatest_monthly_drop_is_march :
  (price_change_march = -3.00) →
  (∀ m, m ≠ price_change_march → m ≥ price_change_march) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_greatest_monthly_drop_is_march_l1103_110322


namespace NUMINAMATH_GPT_percentage_of_music_students_l1103_110317

theorem percentage_of_music_students 
  (total_students : ℕ) 
  (dance_students : ℕ) 
  (art_students : ℕ) 
  (drama_students : ℕ)
  (h_total : total_students = 2000) 
  (h_dance : dance_students = 450) 
  (h_art : art_students = 680) 
  (h_drama : drama_students = 370) 
  : (total_students - (dance_students + art_students + drama_students)) / total_students * 100 = 25 
:= by 
  sorry

end NUMINAMATH_GPT_percentage_of_music_students_l1103_110317


namespace NUMINAMATH_GPT_steve_pencils_left_l1103_110320

def initial_pencils : ℕ := 2 * 12
def pencils_given_to_lauren : ℕ := 6
def pencils_given_to_matt : ℕ := pencils_given_to_lauren + 3

def pencils_left (initial_pencils given_lauren given_matt : ℕ) : ℕ :=
  initial_pencils - given_lauren - given_matt

theorem steve_pencils_left : pencils_left initial_pencils pencils_given_to_lauren pencils_given_to_matt = 9 := by
  sorry

end NUMINAMATH_GPT_steve_pencils_left_l1103_110320


namespace NUMINAMATH_GPT_mul_scientific_notation_l1103_110366

theorem mul_scientific_notation (a b : ℝ) (c d : ℝ) (h1 : a = 7 * 10⁻¹) (h2 : b = 8 * 10⁻¹) :
  (a * b = 0.56) :=
by
  sorry

end NUMINAMATH_GPT_mul_scientific_notation_l1103_110366


namespace NUMINAMATH_GPT_definite_integral_sin_cos_l1103_110349

open Real

theorem definite_integral_sin_cos :
  ∫ x in - (π / 2)..(π / 2), (sin x + cos x) = 2 :=
sorry

end NUMINAMATH_GPT_definite_integral_sin_cos_l1103_110349


namespace NUMINAMATH_GPT_dan_initial_money_l1103_110324

theorem dan_initial_money (cost_candy : ℕ) (cost_chocolate : ℕ) (total_spent: ℕ) (hc : cost_candy = 7) (hch : cost_chocolate = 6) (hs : total_spent = 13) 
  (h : total_spent = cost_candy + cost_chocolate) : total_spent = 13 := by
  sorry

end NUMINAMATH_GPT_dan_initial_money_l1103_110324


namespace NUMINAMATH_GPT_least_possible_value_of_m_plus_n_l1103_110312

noncomputable def least_possible_sum (m n : ℕ) : ℕ :=
m + n

theorem least_possible_value_of_m_plus_n (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0)
  (h3 : Nat.gcd (m + n) 330 = 1)
  (h4 : m^m % n^n = 0)
  (h5 : m % n ≠ 0) : 
  least_possible_sum m n = 98 := 
sorry

end NUMINAMATH_GPT_least_possible_value_of_m_plus_n_l1103_110312


namespace NUMINAMATH_GPT_correct_statements_l1103_110353

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem correct_statements :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (f (Real.log 3 / Real.log 2) ≠ 2) ∧
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (|x|) ≥ 0 ∧ f 0 = 0) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l1103_110353


namespace NUMINAMATH_GPT_bob_sheep_and_ratio_l1103_110384

-- Define the initial conditions
def mary_initial_sheep : ℕ := 300
def additional_sheep_bob_has : ℕ := 35
def sheep_mary_buys : ℕ := 266
def fewer_sheep_than_bob : ℕ := 69

-- Define the number of sheep Bob has
def bob_sheep (mary_initial_sheep : ℕ) (additional_sheep_bob_has : ℕ) : ℕ := 
  mary_initial_sheep + additional_sheep_bob_has

-- Define the number of sheep Mary has after buying more sheep
def mary_new_sheep (mary_initial_sheep : ℕ) (sheep_mary_buys : ℕ) : ℕ := 
  mary_initial_sheep + sheep_mary_buys

-- Define the relation between Mary's and Bob's sheep (after Mary buys sheep)
def mary_bob_relation (mary_new_sheep : ℕ) (fewer_sheep_than_bob : ℕ) : Prop :=
  mary_new_sheep + fewer_sheep_than_bob = bob_sheep mary_initial_sheep additional_sheep_bob_has

-- Define the proof problem
theorem bob_sheep_and_ratio : 
  bob_sheep mary_initial_sheep additional_sheep_bob_has = 635 ∧ 
  (bob_sheep mary_initial_sheep additional_sheep_bob_has) * 300 = 635 * mary_initial_sheep := 
by 
  sorry

end NUMINAMATH_GPT_bob_sheep_and_ratio_l1103_110384


namespace NUMINAMATH_GPT_trucks_transport_l1103_110330

variables {x y : ℝ}

theorem trucks_transport (h1 : 2 * x + 3 * y = 15.5)
                         (h2 : 5 * x + 6 * y = 35) :
  3 * x + 2 * y = 17 :=
sorry

end NUMINAMATH_GPT_trucks_transport_l1103_110330


namespace NUMINAMATH_GPT_fraction_increases_by_3_l1103_110340

-- Define initial fraction
def initial_fraction (x y : ℕ) : ℕ :=
  2 * x * y / (3 * x - y)

-- Define modified fraction
def modified_fraction (x y : ℕ) (m : ℕ) : ℕ :=
  2 * (m * x) * (m * y) / (m * (3 * x) - (m * y))

-- State the theorem to prove the value of modified fraction is 3 times the initial fraction
theorem fraction_increases_by_3 (x y : ℕ) : modified_fraction x y 3 = 3 * initial_fraction x y :=
by sorry

end NUMINAMATH_GPT_fraction_increases_by_3_l1103_110340


namespace NUMINAMATH_GPT_tangent_line_of_ellipse_l1103_110358

theorem tangent_line_of_ellipse 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (x₀ y₀ : ℝ) (hx₀ : x₀^2 / a^2 + y₀^2 / b^2 = 1) :
  ∀ x y : ℝ, x₀ * x / a^2 + y₀ * y / b^2 = 1 := 
sorry

end NUMINAMATH_GPT_tangent_line_of_ellipse_l1103_110358


namespace NUMINAMATH_GPT_hamza_bucket_problem_l1103_110365

-- Definitions reflecting the problem conditions
def bucket_2_5_capacity : ℝ := 2.5
def bucket_3_0_capacity : ℝ := 3.0
def bucket_5_6_capacity : ℝ := 5.6
def bucket_6_5_capacity : ℝ := 6.5

def initial_fill_in_5_6 : ℝ := bucket_5_6_capacity
def pour_5_6_to_3_0_remaining : ℝ := 5.6 - 3.0
def remaining_in_5_6_after_second_fill : ℝ := bucket_5_6_capacity - 0.5

-- Main problem statement
theorem hamza_bucket_problem : (bucket_6_5_capacity - 2.6 = 3.9) :=
by sorry

end NUMINAMATH_GPT_hamza_bucket_problem_l1103_110365


namespace NUMINAMATH_GPT_sequence_to_one_l1103_110357

def nextStep (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n - 1

theorem sequence_to_one (n : ℕ) (h : n > 0) :
  ∃ seq : ℕ → ℕ, seq 0 = n ∧ (∀ i, seq (i + 1) = nextStep (seq i)) ∧ (∃ j, seq j = 1) := by
  sorry

end NUMINAMATH_GPT_sequence_to_one_l1103_110357


namespace NUMINAMATH_GPT_joe_lifting_problem_l1103_110391

theorem joe_lifting_problem (x y : ℝ) (h1 : x + y = 900) (h2 : 2 * x = y + 300) : x = 400 :=
sorry

end NUMINAMATH_GPT_joe_lifting_problem_l1103_110391


namespace NUMINAMATH_GPT_find_number_l1103_110380

-- Define the condition: a number exceeds by 40 from its 3/8 part.
def exceeds_by_40_from_its_fraction (x : ℝ) := x = (3/8) * x + 40

-- The theorem: prove that the number is 64 given the condition.
theorem find_number (x : ℝ) (h : exceeds_by_40_from_its_fraction x) : x = 64 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1103_110380


namespace NUMINAMATH_GPT_find_c_l1103_110348

-- Define the functions p and q as given in the conditions
def p (x : ℝ) : ℝ := 3 * x - 9
def q (x : ℝ) (c : ℝ) : ℝ := 4 * x - c

-- State the main theorem with conditions and goal
theorem find_c (c : ℝ) (h : p (q 3 c) = 15) : c = 4 := by
  sorry -- Proof is not required

end NUMINAMATH_GPT_find_c_l1103_110348


namespace NUMINAMATH_GPT_percentage_green_shirts_correct_l1103_110387

variable (total_students blue_percentage red_percentage other_students : ℕ)

noncomputable def percentage_green_shirts (total_students blue_percentage red_percentage other_students : ℕ) : ℕ :=
  let total_blue_shirts := blue_percentage * total_students / 100
  let total_red_shirts := red_percentage * total_students / 100
  let total_blue_red_other_shirts := total_blue_shirts + total_red_shirts + other_students
  let green_shirts := total_students - total_blue_red_other_shirts
  (green_shirts * 100) / total_students

theorem percentage_green_shirts_correct
  (h1 : total_students = 800) 
  (h2 : blue_percentage = 45)
  (h3 : red_percentage = 23)
  (h4 : other_students = 136) : 
  percentage_green_shirts total_students blue_percentage red_percentage other_students = 15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_green_shirts_correct_l1103_110387


namespace NUMINAMATH_GPT_smallest_number_is_C_l1103_110369

def A : ℕ := 36
def B : ℕ := 27 + 5
def C : ℕ := 3 * 10
def D : ℕ := 40 - 3

theorem smallest_number_is_C :
  min (min A B) (min C D) = C :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_smallest_number_is_C_l1103_110369


namespace NUMINAMATH_GPT_div_1_eq_17_div_2_eq_2_11_mul_1_eq_1_4_l1103_110386

-- Define the values provided in the problem
def div_1 := (8 : ℚ) / (8 / 17 : ℚ)
def div_2 := (6 / 11 : ℚ) / 3
def mul_1 := (5 / 4 : ℚ) * (1 / 5 : ℚ)

-- Prove the equivalences
theorem div_1_eq_17 : div_1 = 17 := by
  sorry

theorem div_2_eq_2_11 : div_2 = 2 / 11 := by
  sorry

theorem mul_1_eq_1_4 : mul_1 = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_div_1_eq_17_div_2_eq_2_11_mul_1_eq_1_4_l1103_110386


namespace NUMINAMATH_GPT_hexagon_perimeter_of_intersecting_triangles_l1103_110308

/-- Given two equilateral triangles with parallel sides, where the perimeter of the blue triangle 
    is 4 and the perimeter of the green triangle is 5, prove that the perimeter of the hexagon 
    formed by their intersection is 3. -/
theorem hexagon_perimeter_of_intersecting_triangles 
    (P_blue P_green P_hexagon : ℝ)
    (h_blue : P_blue = 4)
    (h_green : P_green = 5) :
    P_hexagon = 3 := 
sorry

end NUMINAMATH_GPT_hexagon_perimeter_of_intersecting_triangles_l1103_110308


namespace NUMINAMATH_GPT_find_m_l1103_110347

-- Definitions of the given vectors and their properties
def a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Condition that vectors a and b are parallel
def are_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.2 - v₁.2 * v₂.1 = 0

-- Goal: Find the value of m such that vectors a and b are parallel
theorem find_m (m : ℝ) : 
  are_parallel a (b m) → m = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1103_110347


namespace NUMINAMATH_GPT_hyperbola_intersection_l1103_110338

variable (a b c : ℝ) -- positive constants
variables (F1 F2 : (ℝ × ℝ)) -- foci of the hyperbola

-- The positive constants a and b
axiom a_pos : a > 0
axiom b_pos : b > 0

-- The foci are at (-c, 0) and (c, 0)
axiom F1_def : F1 = (-c, 0)
axiom F2_def : F2 = (c, 0)

-- We want to prove that the points (-c, b^2 / a) and (-c, -b^2 / a) are on the hyperbola
theorem hyperbola_intersection :
  (F1 = (-c, 0) ∧ F2 = (c, 0) ∧ a > 0 ∧ b > 0) →
  ∀ y : ℝ, ∃ y1 y2 : ℝ, (y1 = b^2 / a ∧ y2 = -b^2 / a ∧ 
  ( ( (-c)^2 / a^2) - (y1^2 / b^2) = 1 ∧  (-c)^2 / a^2 - y2^2 / b^2 = 1 ) ) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_hyperbola_intersection_l1103_110338


namespace NUMINAMATH_GPT_problem_1_problem_2_l1103_110390

noncomputable def f (x : ℝ) : ℝ := (1 / (9 * (Real.sin x)^2)) + (4 / (9 * (Real.cos x)^2))

theorem problem_1 (x : ℝ) (hx : 0 < x ∧ x < Real.pi / 2) : f x ≥ 1 := 
sorry

theorem problem_2 (x : ℝ) : x^2 + |x-2| + 1 ≥ 3 ↔ (x ≤ 0 ∨ x ≥ 1) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1103_110390


namespace NUMINAMATH_GPT_angle_at_7_20_is_100_degrees_l1103_110363

def angle_between_hands_at_7_20 : ℝ := 100

theorem angle_at_7_20_is_100_degrees
    (hour_hand_pos : ℝ := 210) -- 7 * 30 degrees
    (minute_hand_pos : ℝ := 120) -- 4 * 30 degrees
    (hour_hand_move_per_minute : ℝ := 0.5) -- 0.5 degrees per minute
    (time_past_7_clock : ℝ := 20) -- 20 minutes
    (adjacent_angle : ℝ := 30) -- angle between adjacent numbers
    : angle_between_hands_at_7_20 = 
      (hour_hand_pos - (minute_hand_pos - hour_hand_move_per_minute * time_past_7_clock)) :=
sorry

end NUMINAMATH_GPT_angle_at_7_20_is_100_degrees_l1103_110363


namespace NUMINAMATH_GPT_max_chord_length_of_parabola_l1103_110339

-- Definitions based on the problem conditions
def parabola (x y : ℝ) : Prop := x^2 = 8 * y
def y_midpoint_condition (y1 y2 : ℝ) : Prop := (y1 + y2) / 2 = 4

-- The theorem to prove that the maximum length of the chord AB is 12
theorem max_chord_length_of_parabola (x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h_mid : y_midpoint_condition y1 y2) : 
  abs ((y1 + y2) + 2 * 2) = 12 :=
sorry

end NUMINAMATH_GPT_max_chord_length_of_parabola_l1103_110339


namespace NUMINAMATH_GPT_jason_commute_with_detour_l1103_110306

theorem jason_commute_with_detour (d1 d2 d3 d4 d5 : ℝ) 
  (h1 : d1 = 4)     -- Distance from house to first store
  (h2 : d2 = 6)     -- Distance between first and second store
  (h3 : d3 = d2 + (2/3) * d2) -- Distance between second and third store without detour
  (h4 : d4 = 3)     -- Additional distance due to detour
  (h5 : d5 = d1)    -- Distance from third store to work
  : d1 + d2 + (d3 + d4) + d5 = 27 :=
by
  sorry

end NUMINAMATH_GPT_jason_commute_with_detour_l1103_110306


namespace NUMINAMATH_GPT_photos_in_gallery_l1103_110315

theorem photos_in_gallery (P : ℕ) 
  (h1 : P / 2 + (P / 2 + 120) + P = 920) : P = 400 :=
by
  sorry

end NUMINAMATH_GPT_photos_in_gallery_l1103_110315


namespace NUMINAMATH_GPT_quadrilateral_inequality_l1103_110302

theorem quadrilateral_inequality
  (AB AC BD CD: ℝ)
  (h1 : AB + BD ≤ AC + CD)
  (h2 : AB + CD < AC + BD) :
  AB < AC := by
  sorry

end NUMINAMATH_GPT_quadrilateral_inequality_l1103_110302


namespace NUMINAMATH_GPT_gcd_2750_9450_l1103_110393

theorem gcd_2750_9450 : Nat.gcd 2750 9450 = 50 := by
  sorry

end NUMINAMATH_GPT_gcd_2750_9450_l1103_110393


namespace NUMINAMATH_GPT_david_overall_average_l1103_110367

open Real

noncomputable def english_weighted_average := (74 * 0.20) + (80 * 0.25) + (77 * 0.55)
noncomputable def english_modified := english_weighted_average * 1.5

noncomputable def math_weighted_average := (65 * 0.15) + (75 * 0.25) + (90 * 0.60)
noncomputable def math_modified := math_weighted_average * 2.0

noncomputable def physics_weighted_average := (82 * 0.40) + (85 * 0.60)
noncomputable def physics_modified := physics_weighted_average * 1.2

noncomputable def chemistry_weighted_average := (67 * 0.35) + (89 * 0.65)
noncomputable def chemistry_modified := chemistry_weighted_average * 1.0

noncomputable def biology_weighted_average := (90 * 0.30) + (95 * 0.70)
noncomputable def biology_modified := biology_weighted_average * 1.5

noncomputable def overall_average := (english_modified + math_modified + physics_modified + chemistry_modified + biology_modified) / 5

theorem david_overall_average :
  overall_average = 120.567 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_david_overall_average_l1103_110367


namespace NUMINAMATH_GPT_bryan_travel_hours_per_year_l1103_110368

-- Definitions based on the conditions
def minutes_walk_to_bus_station := 5
def minutes_ride_bus := 20
def minutes_walk_to_job := 5
def days_per_year := 365

-- Total time for one-way travel in minutes
def one_way_travel_minutes := minutes_walk_to_bus_station + minutes_ride_bus + minutes_walk_to_job

-- Total daily travel time in minutes
def daily_travel_minutes := one_way_travel_minutes * 2

-- Convert daily travel time from minutes to hours
def daily_travel_hours := daily_travel_minutes / 60

-- Total yearly travel time in hours
def yearly_travel_hours := daily_travel_hours * days_per_year

-- The theorem to prove
theorem bryan_travel_hours_per_year : yearly_travel_hours = 365 :=
by {
  -- The preliminary arithmetic is not the core of the theorem
  sorry
}

end NUMINAMATH_GPT_bryan_travel_hours_per_year_l1103_110368


namespace NUMINAMATH_GPT_rod_length_is_38_point_25_l1103_110341

noncomputable def length_of_rod (n : ℕ) (l : ℕ) (conversion_factor : ℕ) : ℝ :=
  (n * l : ℝ) / conversion_factor

theorem rod_length_is_38_point_25 :
  length_of_rod 45 85 100 = 38.25 :=
by
  sorry

end NUMINAMATH_GPT_rod_length_is_38_point_25_l1103_110341


namespace NUMINAMATH_GPT_motorist_gas_problem_l1103_110397

noncomputable def original_price_per_gallon (P : ℝ) : Prop :=
  12 * P = 10 * (P + 0.30)

def fuel_efficiency := 25

def new_distance_travelled (P : ℝ) : ℝ :=
  10 * fuel_efficiency

theorem motorist_gas_problem :
  ∃ P : ℝ, original_price_per_gallon P ∧ P = 1.5 ∧ new_distance_travelled P = 250 :=
by
  use 1.5
  sorry

end NUMINAMATH_GPT_motorist_gas_problem_l1103_110397


namespace NUMINAMATH_GPT_max_value_of_N_l1103_110300

def I_k (k : Nat) : Nat :=
  10^(k + 1) + 32

def N (k : Nat) : Nat :=
  (Nat.factors (I_k k)).count 2

theorem max_value_of_N :
  ∃ k : Nat, N k = 6 ∧ (∀ m : Nat, N m ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_N_l1103_110300


namespace NUMINAMATH_GPT_no_consecutive_integers_square_difference_2000_l1103_110354

theorem no_consecutive_integers_square_difference_2000 :
  ¬ ∃ a : ℤ, (a + 1) ^ 2 - a ^ 2 = 2000 :=
by {
  -- some detailed steps might go here in a full proof
  sorry
}

end NUMINAMATH_GPT_no_consecutive_integers_square_difference_2000_l1103_110354


namespace NUMINAMATH_GPT_first_customer_bought_5_l1103_110378

variables 
  (x : ℕ) -- Number of boxes the first customer bought
  (x2 : ℕ) -- Number of boxes the second customer bought
  (x3 : ℕ) -- Number of boxes the third customer bought
  (x4 : ℕ) -- Number of boxes the fourth customer bought
  (x5 : ℕ) -- Number of boxes the fifth customer bought

def goal : ℕ := 150
def remaining_boxes : ℕ := 75
def sold_boxes := x + x2 + x3 + x4 + x5

axiom second_customer (hx2 : x2 = 4 * x) : True
axiom third_customer (hx3 : x3 = (x2 / 2)) : True
axiom fourth_customer (hx4 : x4 = 3 * x3) : True
axiom fifth_customer (hx5 : x5 = 10) : True
axiom sales_goal (hgoal : sold_boxes = goal - remaining_boxes) : True

theorem first_customer_bought_5 (hx2 : x2 = 4 * x) 
                                (hx3 : x3 = (x2 / 2)) 
                                (hx4 : x4 = 3 * x3) 
                                (hx5 : x5 = 10) 
                                (hgoal : sold_boxes = goal - remaining_boxes) : 
                                x = 5 :=
by
  -- Here, we would perform the proof steps
  sorry

end NUMINAMATH_GPT_first_customer_bought_5_l1103_110378


namespace NUMINAMATH_GPT_speed_of_stream_l1103_110336

-- Definitions based on the conditions provided
def speed_still_water : ℝ := 15
def upstream_time_ratio := 2

-- Proof statement
theorem speed_of_stream (v : ℝ) 
  (h1 : ∀ d t_up t_down, (15 - v) * t_up = d ∧ (15 + v) * t_down = d ∧ t_up = upstream_time_ratio * t_down) : 
  v = 5 :=
sorry

end NUMINAMATH_GPT_speed_of_stream_l1103_110336


namespace NUMINAMATH_GPT_mona_unique_players_l1103_110392

-- Define the conditions
def groups (mona: String) : ℕ := 9
def players_per_group : ℕ := 4
def repeat_players_group1 : ℕ := 2
def repeat_players_group2 : ℕ := 1

-- Statement of the proof problem
theorem mona_unique_players
  (total_groups : ℕ := groups "Mona")
  (players_each_group : ℕ := players_per_group)
  (repeats_group1 : ℕ := repeat_players_group1)
  (repeats_group2 : ℕ := repeat_players_group2) :
  (total_groups * players_each_group) - (repeats_group1 + repeats_group2) = 33 := by
  sorry

end NUMINAMATH_GPT_mona_unique_players_l1103_110392


namespace NUMINAMATH_GPT_avg_eq_pos_diff_l1103_110388

theorem avg_eq_pos_diff (y : ℝ) (h : (35 + y) / 2 = 42) : |35 - y| = 14 := 
sorry

end NUMINAMATH_GPT_avg_eq_pos_diff_l1103_110388


namespace NUMINAMATH_GPT_findInitialVolume_l1103_110377

def initialVolume (V : ℝ) : Prop :=
  let newVolume := V + 18
  let initialSugar := 0.27 * V
  let addedSugar := 3.2
  let totalSugar := initialSugar + addedSugar
  let finalSugarPercentage := 0.26536312849162012
  finalSugarPercentage * newVolume = totalSugar 

theorem findInitialVolume : ∃ (V : ℝ), initialVolume V ∧ V = 340 := by
  use 340
  unfold initialVolume
  sorry

end NUMINAMATH_GPT_findInitialVolume_l1103_110377


namespace NUMINAMATH_GPT_tank_capacity_l1103_110325

variable (c : ℕ) -- Total capacity of the tank in liters.
variable (w_0 : ℕ := c / 3) -- Initial volume of water in the tank in liters.

theorem tank_capacity (h1 : w_0 = c / 3) (h2 : (w_0 + 5) / c = 2 / 5) : c = 75 :=
by
  -- Proof steps would be here.
  sorry

end NUMINAMATH_GPT_tank_capacity_l1103_110325


namespace NUMINAMATH_GPT_range_of_m_l1103_110370

noncomputable def A := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
noncomputable def B (m : ℝ) := {x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1}

theorem range_of_m (m : ℝ) (h : B m ⊆ A) : -2 ≤ m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1103_110370


namespace NUMINAMATH_GPT_sum_of_cubes_l1103_110385

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -3) : x^3 + y^3 = 26 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_l1103_110385


namespace NUMINAMATH_GPT_product_sum_of_roots_l1103_110360

theorem product_sum_of_roots (p q r : ℂ)
  (h_eq : ∀ x : ℂ, (2 : ℂ) * x^3 + (1 : ℂ) * x^2 + (-7 : ℂ) * x + (2 : ℂ) = 0 → (x = p ∨ x = q ∨ x = r)) 
  : p * q + q * r + r * p = -7 / 2 := 
sorry

end NUMINAMATH_GPT_product_sum_of_roots_l1103_110360


namespace NUMINAMATH_GPT_overall_average_score_l1103_110346

-- Definitions based on given conditions
def n_m : ℕ := 8   -- number of male students
def avg_m : ℚ := 87  -- average score of male students
def n_f : ℕ := 12  -- number of female students
def avg_f : ℚ := 92  -- average score of female students

-- The target statement to prove
theorem overall_average_score (n_m : ℕ) (avg_m : ℚ) (n_f : ℕ) (avg_f : ℚ) (overall_avg : ℚ) :
  n_m = 8 ∧ avg_m = 87 ∧ n_f = 12 ∧ avg_f = 92 → overall_avg = 90 :=
by
  sorry

end NUMINAMATH_GPT_overall_average_score_l1103_110346


namespace NUMINAMATH_GPT_find_range_of_m_l1103_110364

-- Statements of the conditions given in the problem
axiom positive_real_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : (1 / x + 4 / y = 1)

-- Main statement of the proof problem
theorem find_range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 / x + 4 / y = 1) :
  (∃ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (1 / x + 4 / y = 1) ∧ (x + y / 4 < m^2 - 3 * m)) ↔ (m < -1 ∨ m > 4) := 
sorry

end NUMINAMATH_GPT_find_range_of_m_l1103_110364


namespace NUMINAMATH_GPT_cheenu_time_difference_l1103_110321

theorem cheenu_time_difference :
  let boy_distance : ℝ := 18
  let boy_time_hours : ℝ := 4
  let old_man_distance : ℝ := 12
  let old_man_time_hours : ℝ := 5
  let hour_to_minute : ℝ := 60
  
  let boy_time_minutes := boy_time_hours * hour_to_minute
  let old_man_time_minutes := old_man_time_hours * hour_to_minute

  let boy_time_per_mile := boy_time_minutes / boy_distance
  let old_man_time_per_mile := old_man_time_minutes / old_man_distance
  
  old_man_time_per_mile - boy_time_per_mile = 12 :=
by sorry

end NUMINAMATH_GPT_cheenu_time_difference_l1103_110321


namespace NUMINAMATH_GPT_evaluate_expression_l1103_110309

theorem evaluate_expression : 8^3 + 4 * 8^2 + 6 * 8 + 3 = 1000 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1103_110309


namespace NUMINAMATH_GPT_number_of_students_who_bought_2_pencils_l1103_110352

variable (a b c : ℕ)     -- a is the number of students buying 1 pencil, b is the number of students buying 2 pencils, c is the number of students buying 3 pencils.
variable (total_students total_pencils : ℕ) -- total_students is 36, total_pencils is 50
variable (students_condition1 students_condition2 : ℕ) -- conditions: students_condition1 for the sum of the students, students_condition2 for the sum of the pencils

theorem number_of_students_who_bought_2_pencils :
  total_students = 36 ∧
  total_pencils = 50 ∧
  total_students = a + b + c ∧
  total_pencils = a * 1 + b * 2 + c * 3 ∧
  a = 2 * (b + c) → 
  b = 10 :=
by sorry

end NUMINAMATH_GPT_number_of_students_who_bought_2_pencils_l1103_110352


namespace NUMINAMATH_GPT_ratio_of_guests_l1103_110350

def bridgette_guests : Nat := 84
def alex_guests : Nat := sorry -- This will be inferred in the theorem
def extra_plates : Nat := 10
def total_asparagus_spears : Nat := 1200
def asparagus_per_plate : Nat := 8

theorem ratio_of_guests (A : Nat) (h1 : total_asparagus_spears / asparagus_per_plate = 150) (h2 : 150 - extra_plates = 140) (h3 : 140 - bridgette_guests = A) : A / bridgette_guests = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_guests_l1103_110350


namespace NUMINAMATH_GPT_general_term_l1103_110396

def S (n : ℕ) : ℕ := n^2 + 3 * n

def a (n : ℕ) : ℕ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = 2 * n + 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_general_term_l1103_110396


namespace NUMINAMATH_GPT_hexagon_arrangements_eq_144_l1103_110373

def is_valid_arrangement (arr : (Fin 7 → ℕ)) : Prop :=
  ∀ (i j k : Fin 7),
    (i.val + j.val + k.val = 18) → -- 18 being a derived constant factor (since 3x = 28 + 2G where G ∈ {1, 4, 7} and hence x = 30,34,38/3 respectively make it divisible by 3 sum is 18 always)
    arr i + arr j + arr k = arr ⟨3, sorry⟩ -- arr[3] is the position of G

noncomputable def count_valid_arrangements : ℕ :=
  sorry -- Calculation of 3*48 goes here and respective pairing and permutations.

theorem hexagon_arrangements_eq_144 :
  count_valid_arrangements = 144 :=
sorry

end NUMINAMATH_GPT_hexagon_arrangements_eq_144_l1103_110373


namespace NUMINAMATH_GPT_weeks_to_save_l1103_110319

-- Define the conditions as given in the problem
def cost_of_bike : ℕ := 600
def gift_from_parents : ℕ := 60
def gift_from_uncle : ℕ := 40
def gift_from_sister : ℕ := 20
def gift_from_friend : ℕ := 30
def weekly_earnings : ℕ := 18

-- Total gift money
def total_gift_money : ℕ := gift_from_parents + gift_from_uncle + gift_from_sister + gift_from_friend

-- Total money after x weeks
def total_money_after_weeks (x : ℕ) : ℕ := total_gift_money + weekly_earnings * x

-- Main theorem statement
theorem weeks_to_save (x : ℕ) : total_money_after_weeks x = cost_of_bike → x = 25 := by
  sorry

end NUMINAMATH_GPT_weeks_to_save_l1103_110319


namespace NUMINAMATH_GPT_fraction_value_l1103_110344

theorem fraction_value : (5 - Real.sqrt 4) / (5 + Real.sqrt 4) = 3 / 7 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l1103_110344


namespace NUMINAMATH_GPT_mark_deposit_amount_l1103_110316

-- Define the conditions
def bryans_deposit (M : ℝ) : ℝ := 5 * M - 40
def total_deposit (M : ℝ) : ℝ := M + bryans_deposit M

-- State the theorem
theorem mark_deposit_amount (M : ℝ) (h1: total_deposit M = 400) : M = 73.33 :=
by
  sorry

end NUMINAMATH_GPT_mark_deposit_amount_l1103_110316


namespace NUMINAMATH_GPT_problem_m_n_l1103_110399

theorem problem_m_n (m n : ℝ) (h1 : m * n = 1) (h2 : m^2 + n^2 = 3) (h3 : m^3 + n^3 = 44 + n^4) (h4 : m^5 + 5 = 11) : m^9 + n = -29 :=
sorry

end NUMINAMATH_GPT_problem_m_n_l1103_110399


namespace NUMINAMATH_GPT_seats_per_bus_correct_l1103_110313

-- Define the conditions given in the problem
def students : ℕ := 28
def buses : ℕ := 4

-- Define the number of seats per bus
def seats_per_bus : ℕ := students / buses

-- State the theorem that proves the number of seats per bus
theorem seats_per_bus_correct : seats_per_bus = 7 := by
  -- conditions are used as definitions, the goal is to prove seats_per_bus == 7
  sorry

end NUMINAMATH_GPT_seats_per_bus_correct_l1103_110313


namespace NUMINAMATH_GPT_calculation_A_B_l1103_110327

theorem calculation_A_B :
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  A - B = 4397 :=
by
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  sorry

end NUMINAMATH_GPT_calculation_A_B_l1103_110327


namespace NUMINAMATH_GPT_angle_BCM_in_pentagon_l1103_110307

-- Definitions of the conditions
structure Pentagon (A B C D E : Type) :=
  (is_regular : ∀ (x y : Type), ∃ (angle : ℝ), angle = 108)

structure EquilateralTriangle (A B M : Type) :=
  (is_equilateral : ∀ (x y : Type), ∃ (angle : ℝ), angle = 60)

-- Problem statement
theorem angle_BCM_in_pentagon (A B C D E M : Type) (P : Pentagon A B C D E) (T : EquilateralTriangle A B M) :
  ∃ (angle : ℝ), angle = 66 :=
by
  sorry

end NUMINAMATH_GPT_angle_BCM_in_pentagon_l1103_110307


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_x_gt_5_x_gt_3_l1103_110314

theorem sufficient_but_not_necessary_condition_x_gt_5_x_gt_3 :
  ∀ x : ℝ, (x > 5 → x > 3) ∧ (∃ x : ℝ, x > 3 ∧ x ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_x_gt_5_x_gt_3_l1103_110314


namespace NUMINAMATH_GPT_triangle_incircle_ratio_l1103_110343

theorem triangle_incircle_ratio (r s q : ℝ) (h1 : r + s = 8) (h2 : r < s) (h3 : r + q = 13) (h4 : s + q = 17) (h5 : 8 + 13 > 17 ∧ 8 + 17 > 13 ∧ 13 + 17 > 8):
  r / s = 1 / 3 := by sorry

end NUMINAMATH_GPT_triangle_incircle_ratio_l1103_110343


namespace NUMINAMATH_GPT_fraction_of_time_at_15_mph_l1103_110395

theorem fraction_of_time_at_15_mph
  (t1 t2 : ℝ)
  (h : (5 * t1 + 15 * t2) / (t1 + t2) = 10) :
  t2 / (t1 + t2) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_time_at_15_mph_l1103_110395


namespace NUMINAMATH_GPT_ratio_copper_zinc_l1103_110334

theorem ratio_copper_zinc (total_mass zinc_mass : ℕ) (h1 : total_mass = 100) (h2 : zinc_mass = 35) : 
  ∃ (copper_mass : ℕ), 
    copper_mass = total_mass - zinc_mass ∧ (copper_mass / 5, zinc_mass / 5) = (13, 7) :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_copper_zinc_l1103_110334


namespace NUMINAMATH_GPT_o_hara_triple_example_l1103_110361

-- definitions
def is_OHara_triple (a b x : ℕ) : Prop :=
  (Real.sqrt a) + (Real.sqrt b) = x

-- conditions
def a : ℕ := 81
def b : ℕ := 49
def x : ℕ := 16

-- statement
theorem o_hara_triple_example : is_OHara_triple a b x :=
by
  sorry

end NUMINAMATH_GPT_o_hara_triple_example_l1103_110361


namespace NUMINAMATH_GPT_number_of_subsets_of_set_l1103_110379

theorem number_of_subsets_of_set (x y : ℝ) 
  (z : ℂ) (hz : z = (2 - (1 : ℂ) * Complex.I) / (1 + (2 : ℂ) * Complex.I))
  (hx : z.re = x) (hy : z.im = y) : 
  (Finset.powerset ({x, 2^x, y} : Finset ℝ)).card = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_subsets_of_set_l1103_110379


namespace NUMINAMATH_GPT_equation_of_line_l1103_110398

theorem equation_of_line (A B : ℝ × ℝ) (M : ℝ × ℝ) (hM : M = (-1, 2)) (hA : A.2 = 0) (hB : B.1 = 0) (hMid : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  ∃ (a b c : ℝ), (a = 2 ∧ b = -1 ∧ c = 4) ∧ ∀ (x y : ℝ), y = a * x + b * y + c → 2 * x - y + 4 = 0 := 
  sorry

end NUMINAMATH_GPT_equation_of_line_l1103_110398


namespace NUMINAMATH_GPT_find_building_block_width_l1103_110301

noncomputable def building_block_width
  (box_height box_width box_length building_block_height building_block_length : ℕ)
  (num_building_blocks : ℕ)
  (box_height_eq : box_height = 8)
  (box_width_eq : box_width = 10)
  (box_length_eq : box_length = 12)
  (building_block_height_eq : building_block_height = 3)
  (building_block_length_eq : building_block_length = 4)
  (num_building_blocks_eq : num_building_blocks = 40)
: ℕ :=
(8 * 10 * 12) / 40 / (3 * 4)

theorem find_building_block_width
  (box_height box_width box_length building_block_height building_block_length : ℕ)
  (num_building_blocks : ℕ)
  (box_height_eq : box_height = 8)
  (box_width_eq : box_width = 10)
  (box_length_eq : box_length = 12)
  (building_block_height_eq : building_block_height = 3)
  (building_block_length_eq : building_block_length = 4)
  (num_building_blocks_eq : num_building_blocks = 40) :
  building_block_width box_height box_width box_length building_block_height building_block_length num_building_blocks box_height_eq box_width_eq box_length_eq building_block_height_eq building_block_length_eq num_building_blocks_eq = 2 := 
sorry

end NUMINAMATH_GPT_find_building_block_width_l1103_110301


namespace NUMINAMATH_GPT_functional_expression_y_l1103_110372

theorem functional_expression_y (x y : ℝ) (k : ℝ) 
  (h1 : ∀ x, y + 2 = k * x) 
  (h2 : y = 7) 
  (h3 : x = 3) : 
  y = 3 * x - 2 := 
by 
  sorry

end NUMINAMATH_GPT_functional_expression_y_l1103_110372


namespace NUMINAMATH_GPT_min_max_f_l1103_110394

theorem min_max_f (a b x y z t : ℝ) (ha : 0 < a) (hb : 0 < b)
  (hxz : x + z = 1) (hyt : y + t = 1) (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hz : 0 ≤ z) (ht : 0 ≤ t) :
  1 ≤ ( (a * x^2 + b * y^2) / (a * x + b * y) + (a * z^2 + b * t^2) / (a * z + b * t) ) ∧
  ( (a * x^2 + b * y^2) / (a * x + b * y) + (a * z^2 + b * t^2) / (a * z + b * t) ) ≤ 2 :=
sorry

end NUMINAMATH_GPT_min_max_f_l1103_110394


namespace NUMINAMATH_GPT_y_coordinate_of_C_l1103_110376

def Point : Type := (ℤ × ℤ)

def A : Point := (0, 0)
def B : Point := (0, 4)
def D : Point := (4, 4)
def E : Point := (4, 0)

def PentagonArea (C : Point) : ℚ :=
  let triangleArea : ℚ := (1/2 : ℚ) * 4 * ((C.2 : ℚ) - 4)
  let squareArea : ℚ := 4 * 4
  triangleArea + squareArea

theorem y_coordinate_of_C (h : ℤ) (C : Point := (2, h)) : PentagonArea C = 40 → C.2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_y_coordinate_of_C_l1103_110376


namespace NUMINAMATH_GPT_average_marks_five_subjects_l1103_110345

theorem average_marks_five_subjects 
  (P total_marks : ℕ)
  (h1 : total_marks = P + 350) :
  (total_marks - P) / 5 = 70 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_five_subjects_l1103_110345


namespace NUMINAMATH_GPT_r_cube_plus_inv_r_cube_eq_zero_l1103_110382

theorem r_cube_plus_inv_r_cube_eq_zero {r : ℝ} (h : (r + 1/r)^2 = 3) : r^3 + 1/r^3 = 0 := 
sorry

end NUMINAMATH_GPT_r_cube_plus_inv_r_cube_eq_zero_l1103_110382


namespace NUMINAMATH_GPT_total_handshakes_at_convention_l1103_110318

def number_of_gremlins := 30
def number_of_imps := 20
def disagreeing_imps := 5
def specific_gremlins := 10

theorem total_handshakes_at_convention : 
  (number_of_gremlins * (number_of_gremlins - 1) / 2) +
  ((number_of_imps - disagreeing_imps) * number_of_gremlins) + 
  (disagreeing_imps * (number_of_gremlins - specific_gremlins)) = 985 :=
by 
  sorry

end NUMINAMATH_GPT_total_handshakes_at_convention_l1103_110318


namespace NUMINAMATH_GPT_prime_pairs_satisfying_conditions_l1103_110304

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def satisfies_conditions (p q : ℕ) : Prop :=
  (7 * p + 1) % q = 0 ∧ (7 * q + 1) % p = 0

theorem prime_pairs_satisfying_conditions :
  { (p, q) | is_prime p ∧ is_prime q ∧ satisfies_conditions p q } = {(2, 3), (2, 5), (3, 11)} := 
sorry

end NUMINAMATH_GPT_prime_pairs_satisfying_conditions_l1103_110304


namespace NUMINAMATH_GPT_find_blue_balls_l1103_110333

theorem find_blue_balls 
  (B : ℕ)
  (red_balls : ℕ := 7)
  (green_balls : ℕ := 4)
  (prob_red_red : ℚ := 7 / 40) -- 0.175 represented as a rational number
  (h : (21 / ((11 + B) * (10 + B) / 2 : ℚ)) = prob_red_red) :
  B = 5 :=
sorry

end NUMINAMATH_GPT_find_blue_balls_l1103_110333


namespace NUMINAMATH_GPT_trapezoid_lower_side_length_l1103_110323

variable (U L : ℝ) (height area : ℝ)

theorem trapezoid_lower_side_length
  (h1 : L = U - 3.4)
  (h2 : height = 5.2)
  (h3 : area = 100.62)
  (h4 : area = (1 / 2) * (U + L) * height) :
  L = 17.65 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_lower_side_length_l1103_110323


namespace NUMINAMATH_GPT_polynomial_divisibility_l1103_110335

noncomputable def polynomial_with_positive_int_coeffs : Type :=
{ f : ℕ → ℕ // ∀ m n : ℕ, f m < f n ↔ m < n }

theorem polynomial_divisibility
  (f : polynomial_with_positive_int_coeffs)
  (n : ℕ) (hn : n > 0) :
  f.1 n ∣ f.1 (f.1 n + 1) ↔ n = 1 :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1103_110335


namespace NUMINAMATH_GPT_age_of_other_replaced_man_l1103_110326

theorem age_of_other_replaced_man (A B C D : ℕ) (h1 : A = 23) (h2 : ((52 + C + D) / 4 > (A + B + C + D) / 4)) :
  B < 29 := 
by
  sorry

end NUMINAMATH_GPT_age_of_other_replaced_man_l1103_110326


namespace NUMINAMATH_GPT_pipe_A_filling_time_l1103_110356

theorem pipe_A_filling_time :
  ∃ (t : ℚ), 
  (∀ (t : ℚ), (t > 0) → (1 / t + 5 / t = 1 / 4.571428571428571) ↔ t = 27.42857142857143) := 
by
  -- definition of t and the corresponding conditions are directly derived from the problem
  sorry

end NUMINAMATH_GPT_pipe_A_filling_time_l1103_110356


namespace NUMINAMATH_GPT_add_three_to_both_sides_l1103_110342

variable {a b : ℝ}

theorem add_three_to_both_sides (h : a < b) : 3 + a < 3 + b :=
by
  sorry

end NUMINAMATH_GPT_add_three_to_both_sides_l1103_110342


namespace NUMINAMATH_GPT_twenty_eight_is_seventy_percent_of_what_number_l1103_110337

theorem twenty_eight_is_seventy_percent_of_what_number (x : ℝ) (h : 28 / x = 70 / 100) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_twenty_eight_is_seventy_percent_of_what_number_l1103_110337
