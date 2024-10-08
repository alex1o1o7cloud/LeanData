import Mathlib

namespace scientific_notation_of_1040000000_l152_152633

theorem scientific_notation_of_1040000000 : (1.04 * 10^9 = 1040000000) :=
by
  -- Math proof steps can be added here
  sorry

end scientific_notation_of_1040000000_l152_152633


namespace find_number_l152_152116

theorem find_number (x : ℤ) (h : ((x * 2) - 37 + 25) / 8 = 5) : x = 26 :=
sorry  -- Proof placeholder

end find_number_l152_152116


namespace stockholm_to_uppsala_distance_l152_152793

-- Definitions based on conditions
def map_distance_cm : ℝ := 3
def scale_cm_to_km : ℝ := 80

-- Theorem statement based on the question and correct answer
theorem stockholm_to_uppsala_distance : 
  (map_distance_cm * scale_cm_to_km = 240) :=
by 
  -- This is where the proof would go
  sorry

end stockholm_to_uppsala_distance_l152_152793


namespace bridgette_has_4_birds_l152_152327

/-
Conditions:
1. Bridgette has 2 dogs.
2. Bridgette has 3 cats.
3. Bridgette has some birds.
4. She gives the dogs a bath twice a month.
5. She gives the cats a bath once a month.
6. She gives the birds a bath once every 4 months.
7. In a year, she gives a total of 96 baths.
-/

def num_birds (num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year : ℕ) : ℕ :=
  let yearly_dog_baths := num_dogs * dog_baths_per_month * 12
  let yearly_cat_baths := num_cats * cat_baths_per_month * 12
  let birds_baths := total_baths_per_year - (yearly_dog_baths + yearly_cat_baths)
  let baths_per_bird_per_year := 12 / bird_baths_per_4_months
  birds_baths / baths_per_bird_per_year

theorem bridgette_has_4_birds :
  ∀ (num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year : ℕ),
    num_dogs = 2 →
    num_cats = 3 →
    dog_baths_per_month = 2 →
    cat_baths_per_month = 1 →
    bird_baths_per_4_months = 4 →
    total_baths_per_year = 96 →
    num_birds num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year = 4 :=
by
  intros
  sorry


end bridgette_has_4_birds_l152_152327


namespace calculate_green_paint_l152_152805

theorem calculate_green_paint {green white : ℕ} (ratio_white_to_green : 5 * green = 3 * white) (use_white_paint : white = 15) : green = 9 :=
by
  sorry

end calculate_green_paint_l152_152805


namespace find_sample_size_l152_152185

def ratio_A : ℕ := 2
def ratio_B : ℕ := 3
def ratio_C : ℕ := 5
def total_ratio : ℕ := ratio_A + ratio_B + ratio_C
def num_B_selected : ℕ := 24

theorem find_sample_size : ∃ n : ℕ, num_B_selected * total_ratio = ratio_B * n :=
by
  sorry

end find_sample_size_l152_152185


namespace Daniel_correct_answers_l152_152621

theorem Daniel_correct_answers
  (c w : ℕ)
  (h1 : c + w = 12)
  (h2 : 4 * c - 3 * w = 21) :
  c = 9 :=
sorry

end Daniel_correct_answers_l152_152621


namespace probability_product_positive_correct_l152_152795

noncomputable def probability_product_positive : ℚ :=
  let length_total := 45
  let length_negative := 30
  let length_positive := 15
  let prob_negative := (length_negative : ℚ) / length_total
  let prob_positive := (length_positive : ℚ) / length_total
  let prob_product_positive := prob_negative^2 + prob_positive^2
  prob_product_positive

theorem probability_product_positive_correct :
  probability_product_positive = 5 / 9 :=
by
  sorry

end probability_product_positive_correct_l152_152795


namespace broccoli_difference_l152_152265

theorem broccoli_difference (A : ℕ) (s : ℕ) (s' : ℕ)
  (h1 : A = 1600)
  (h2 : s = Nat.sqrt A)
  (h3 : s' < s)
  (h4 : (s')^2 < A)
  (h5 : A - (s')^2 = 79) :
  (1600 - (s')^2) = 79 :=
by
  sorry

end broccoli_difference_l152_152265


namespace at_least_two_equal_l152_152583

theorem at_least_two_equal (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : b + a^2 + c^2 = c + a^2 + b^2) : 
  (a = b) ∨ (a = c) ∨ (b = c) :=
sorry

end at_least_two_equal_l152_152583


namespace problem_statement_l152_152729

theorem problem_statement :
  ∀ x a k n : ℤ, 
  (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n → a - n + k = 3 :=
by  
  sorry

end problem_statement_l152_152729


namespace trig_identity_l152_152672

theorem trig_identity (a : ℝ) (h : (1 + Real.sin a) / Real.cos a = -1 / 2) : 
  (Real.cos a / (Real.sin a - 1)) = 1 / 2 := by
  -- Proof goes here
  sorry

end trig_identity_l152_152672


namespace smallest_angle_in_triangle_l152_152985

theorem smallest_angle_in_triangle (a b c x : ℝ) 
  (h1 : a + b + c = 180)
  (h2 : a = 5 * x)
  (h3 : b = 3 * x) :
  x = 20 :=
by
  sorry

end smallest_angle_in_triangle_l152_152985


namespace mmobile_additional_line_cost_l152_152130

noncomputable def cost_tmobile (n : ℕ) : ℕ :=
  if n ≤ 2 then 50 else 50 + (n - 2) * 16

noncomputable def cost_mmobile (x : ℕ) (n : ℕ) : ℕ :=
  if n ≤ 2 then 45 else 45 + (n - 2) * x

theorem mmobile_additional_line_cost
  (x : ℕ)
  (ht : cost_tmobile 5 = 98)
  (hm : cost_tmobile 5 - cost_mmobile x 5 = 11) :
  x = 14 :=
by
  sorry

end mmobile_additional_line_cost_l152_152130


namespace find_quadratic_function_l152_152103

theorem find_quadratic_function (g : ℝ → ℝ) 
  (h1 : g 0 = 0) 
  (h2 : g 1 = 1) 
  (h3 : g (-1) = 5) 
  (h_quadratic : ∃ a b, ∀ x, g x = a * x^2 + b * x) : 
  g = fun x => 3 * x^2 - 2 * x := 
by
  sorry

end find_quadratic_function_l152_152103


namespace ratio_of_x_y_l152_152646

theorem ratio_of_x_y (x y : ℝ) (h₁ : 3 < (x - y) / (x + y)) (h₂ : (x - y) / (x + y) < 4) (h₃ : ∃ a b : ℤ, x = a * y / b ) (h₄ : x + y = 10) :
  x / y = -2 := sorry

end ratio_of_x_y_l152_152646


namespace collinear_vectors_x_value_l152_152175

theorem collinear_vectors_x_value (x : ℝ) (a b : ℝ × ℝ) (h₁: a = (2, x)) (h₂: b = (1, 2))
  (h₃: ∃ k : ℝ, a = k • b) : x = 4 :=
by
  sorry

end collinear_vectors_x_value_l152_152175


namespace jacks_walking_rate_l152_152739

theorem jacks_walking_rate :
  let distance := 8
  let time_in_minutes := 1 * 60 + 15
  let time := time_in_minutes / 60.0
  let rate := distance / time
  rate = 6.4 :=
by
  sorry

end jacks_walking_rate_l152_152739


namespace sequence_remainder_prime_l152_152217

theorem sequence_remainder_prime (p : ℕ) (hp : Nat.Prime p) (x : ℕ → ℕ)
  (h1 : ∀ i, 0 ≤ i ∧ i < p → x i = i)
  (h2 : ∀ n, n ≥ p → x n = x (n-1) + x (n-p)) :
  (x (p^3) % p) = p - 1 :=
sorry

end sequence_remainder_prime_l152_152217


namespace minimum_bailing_rate_l152_152300

-- Conditions as formal definitions.
def distance_from_shore : ℝ := 3
def intake_rate : ℝ := 20 -- gallons per minute
def sinking_threshold : ℝ := 120 -- gallons
def speed_first_half : ℝ := 6 -- miles per hour
def speed_second_half : ℝ := 3 -- miles per hour

-- Formal translation of the problem using definitions.
theorem minimum_bailing_rate : (distance_from_shore = 3) →
                             (intake_rate = 20) →
                             (sinking_threshold = 120) →
                             (speed_first_half = 6) →
                             (speed_second_half = 3) →
                             (∃ r : ℝ, 18 ≤ r) :=
by
  sorry

end minimum_bailing_rate_l152_152300


namespace sum_of_areas_of_circles_l152_152302

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end sum_of_areas_of_circles_l152_152302


namespace exists_negative_number_satisfying_inequality_l152_152693

theorem exists_negative_number_satisfying_inequality :
  ∃ x : ℝ, x < 0 ∧ (1 + x) * (1 - 9 * x) > 0 :=
sorry

end exists_negative_number_satisfying_inequality_l152_152693


namespace find_y_l152_152195

theorem find_y (x y : ℝ) (h1 : 0.5 * x = 0.25 * y - 30) (h2 : x = 690) : y = 1500 :=
by
  sorry

end find_y_l152_152195


namespace right_triangle_least_side_l152_152146

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end right_triangle_least_side_l152_152146


namespace vector_coordinates_l152_152862

theorem vector_coordinates :
  let a : ℝ × ℝ := (3, -1)
  let b : ℝ × ℝ := (-1, 2)
  (-3 : ℝ) • a + (-2 : ℝ) • b = (-7, -1) :=
by
  let a : ℝ × ℝ := (3, -1)
  let b : ℝ × ℝ := (-1, 2)
  show (-3 : ℝ) • a + (-2 : ℝ) • b = (-7, -1)
  sorry

end vector_coordinates_l152_152862


namespace opposite_vertices_equal_l152_152108

-- Define the angles of a regular convex hexagon
variables {α β γ δ ε ζ : ℝ}

-- Regular hexagon condition: The sum of the alternating angles
axiom angle_sum_condition :
  α + γ + ε = β + δ + ε

-- Define the final theorem to prove that the opposite vertices have equal angles
theorem opposite_vertices_equal (h : α + γ + ε = β + δ + ε) :
  α = δ ∧ β = ε ∧ γ = ζ :=
sorry

end opposite_vertices_equal_l152_152108


namespace find_N_l152_152340

variable (a b c N : ℕ)

theorem find_N (h1 : a + b + c = 90) (h2 : a - 7 = N) (h3 : b + 7 = N) (h4 : 5 * c = N) : N = 41 := 
by
  sorry

end find_N_l152_152340


namespace test_scores_ordering_l152_152933

variable (M Q S Z K : ℕ)
variable (M_thinks_lowest : M > K)
variable (Q_thinks_same : Q = K)
variable (S_thinks_not_highest : S < K)
variable (Z_thinks_not_middle : (Z < S ∨ Z > M))

theorem test_scores_ordering : (Z < S) ∧ (S < Q) ∧ (Q < M) := by
  -- proof
  sorry

end test_scores_ordering_l152_152933


namespace find_explicit_formula_l152_152535

variable (f : ℝ → ℝ)

theorem find_explicit_formula 
  (h : ∀ x : ℝ, f (x - 1) = 2 * x^2 - 8 * x + 11) :
  ∀ x : ℝ, f x = 2 * x^2 - 4 * x + 5 :=
by
  sorry

end find_explicit_formula_l152_152535


namespace impossible_coins_l152_152802

theorem impossible_coins (p1 p2 : ℝ) : 
  ¬ ((1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * p2 = p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end impossible_coins_l152_152802


namespace hazel_salmon_caught_l152_152334

-- Define the conditions
def father_salmon_caught : Nat := 27
def total_salmon_caught : Nat := 51

-- Define the main statement to be proved
theorem hazel_salmon_caught : total_salmon_caught - father_salmon_caught = 24 := by
  sorry

end hazel_salmon_caught_l152_152334


namespace white_black_ratio_l152_152859

theorem white_black_ratio (W B : ℕ) (h1 : W + B = 78) (h2 : (2 / 3 : ℚ) * (B - W) = 4) : W / B = 6 / 7 := by
  sorry

end white_black_ratio_l152_152859


namespace total_number_of_animals_l152_152524

-- Definitions for the number of each type of animal
def cats : ℕ := 645
def dogs : ℕ := 567
def rabbits : ℕ := 316
def reptiles : ℕ := 120

-- The statement to prove
theorem total_number_of_animals :
  cats + dogs + rabbits + reptiles = 1648 := by
  sorry

end total_number_of_animals_l152_152524


namespace infinite_series_sum_eq_1_div_432_l152_152655

theorem infinite_series_sum_eq_1_div_432 :
  (∑' n : ℕ, (4 * (n + 1) + 1) / ((4 * (n + 1) - 1)^3 * (4 * (n + 1) + 3)^3)) = (1 / 432) :=
  sorry

end infinite_series_sum_eq_1_div_432_l152_152655


namespace total_trip_length_is570_l152_152507

theorem total_trip_length_is570 (v D : ℝ) (h1 : (2:ℝ) + (2/3) + (6 * (D - 2 * v) / (5 * v)) = 2.75)
(h2 : (2:ℝ) + (50 / v) + (2/3) + (6 * (D - 2 * v - 50) / (5 * v)) = 2.33) :
D = 570 :=
sorry

end total_trip_length_is570_l152_152507


namespace num_integers_in_set_x_l152_152360

-- Definition and conditions
variable (x y : Finset ℤ)
variable (h1 : y.card = 10)
variable (h2 : (x ∩ y).card = 6)
variable (h3 : (x.symmDiff y).card = 6)

-- Proof statement
theorem num_integers_in_set_x : x.card = 8 := by
  sorry

end num_integers_in_set_x_l152_152360


namespace original_price_of_sarees_l152_152625

theorem original_price_of_sarees (P : ℝ) (h : 0.72 * P = 108) : P = 150 := 
by 
  sorry

end original_price_of_sarees_l152_152625


namespace deductive_reasoning_correctness_l152_152020

theorem deductive_reasoning_correctness (major_premise minor_premise form_of_reasoning correct : Prop) 
  (h : major_premise ∧ minor_premise ∧ form_of_reasoning) : correct :=
  sorry

end deductive_reasoning_correctness_l152_152020


namespace product_of_last_two_digits_l152_152048

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 14) (h2 : B = 0 ∨ B = 5) : A * B = 45 :=
sorry

end product_of_last_two_digits_l152_152048


namespace students_not_take_test_l152_152479

theorem students_not_take_test
  (total_students : ℕ)
  (q1_correct : ℕ)
  (q2_correct : ℕ)
  (both_correct : ℕ)
  (h_total : total_students = 29)
  (h_q1 : q1_correct = 19)
  (h_q2 : q2_correct = 24)
  (h_both : both_correct = 19)
  : (total_students - (q1_correct + q2_correct - both_correct) = 5) :=
by
  sorry

end students_not_take_test_l152_152479


namespace students_in_ms_delmont_class_l152_152749

-- Let us define the necessary conditions

def total_cupcakes : Nat := 40
def students_mrs_donnelly_class : Nat := 16
def adults_count : Nat := 4 -- Ms. Delmont, Mrs. Donnelly, the school nurse, and the school principal
def leftover_cupcakes : Nat := 2

-- Define the number of students in Ms. Delmont's class
def students_ms_delmont_class : Nat := 18

-- The statement to prove
theorem students_in_ms_delmont_class :
  total_cupcakes - adults_count - students_mrs_donnelly_class - leftover_cupcakes = students_ms_delmont_class :=
by
  sorry

end students_in_ms_delmont_class_l152_152749


namespace zamena_inequalities_l152_152038

theorem zamena_inequalities :
  ∃ (M E H A : ℕ), 
    M ≠ E ∧ M ≠ H ∧ M ≠ A ∧ 
    E ≠ H ∧ E ≠ A ∧ H ≠ A ∧
    (∃ Z, Z = 5 ∧ -- Assume Z is 5 based on the conclusion
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ A ∧ A ≤ 5 ∧
    3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
    -- ZAMENA evaluates to 541234
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) :=
sorry

end zamena_inequalities_l152_152038


namespace even_marked_squares_9x9_l152_152007

open Nat

theorem even_marked_squares_9x9 :
  let n := 9
  let total_squares := n * n
  let odd_rows_columns := [1, 3, 5, 7, 9]
  let odd_squares := odd_rows_columns.length * odd_rows_columns.length
  total_squares - odd_squares = 56 :=
by
  sorry

end even_marked_squares_9x9_l152_152007


namespace Kat_training_hours_l152_152953

theorem Kat_training_hours
  (h_strength_times : ℕ)
  (h_strength_hours : ℝ)
  (h_boxing_times : ℕ)
  (h_boxing_hours : ℝ)
  (h_times : h_strength_times = 3)
  (h_strength : h_strength_hours = 1)
  (b_times : h_boxing_times = 4)
  (b_hours : h_boxing_hours = 1.5) :
  h_strength_times * h_strength_hours + h_boxing_times * h_boxing_hours = 9 :=
by
  sorry

end Kat_training_hours_l152_152953


namespace solve_quadratic_1_solve_quadratic_2_l152_152565

-- Define the first problem
theorem solve_quadratic_1 (x : ℝ) : 3 * x^2 - 4 * x = 2 * x → x = 0 ∨ x = 2 := by
  -- Proof step will go here
  sorry

-- Define the second problem
theorem solve_quadratic_2 (x : ℝ) : x * (x + 8) = 16 → x = -4 + 4 * Real.sqrt 2 ∨ x = -4 - 4 * Real.sqrt 2 := by
  -- Proof step will go here
  sorry

end solve_quadratic_1_solve_quadratic_2_l152_152565


namespace intersecting_lines_l152_152603

theorem intersecting_lines (c d : ℝ)
  (h1 : 16 = 2 * 4 + c)
  (h2 : 16 = 5 * 4 + d) :
  c + d = 4 :=
sorry

end intersecting_lines_l152_152603


namespace pears_equivalence_l152_152345

theorem pears_equivalence :
  (3 / 4 : ℚ) * 16 * (5 / 6) = 10 → 
  (2 / 5 : ℚ) * 20 * (5 / 6) = 20 / 3 := 
by
  intros h
  sorry

end pears_equivalence_l152_152345


namespace d_not_unique_minimum_l152_152904

noncomputable def d (n : ℕ) (x : Fin n → ℝ) (t : ℝ) : ℝ :=
  (Finset.min' (Finset.univ.image (λ i => abs (x i - t))) sorry + 
  Finset.max' (Finset.univ.image (λ i => abs (x i - t))) sorry) / 2

theorem d_not_unique_minimum (n : ℕ) (x : Fin n → ℝ) :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ d n x t1 = d n x t2 := sorry

end d_not_unique_minimum_l152_152904


namespace range_of_s_triangle_l152_152683

theorem range_of_s_triangle (inequalities_form_triangle : Prop) : 
  (0 < s ∧ s ≤ 2) ∨ (s ≥ 4) ↔ inequalities_form_triangle := 
sorry

end range_of_s_triangle_l152_152683


namespace number_of_juniors_l152_152109

theorem number_of_juniors
  (T : ℕ := 28)
  (hT : T = 28)
  (x y : ℕ)
  (hxy : x = y)
  (J S : ℕ)
  (hx : x = J / 4)
  (hy : y = S / 10)
  (hJS : J + S = T) :
  J = 8 :=
by sorry

end number_of_juniors_l152_152109


namespace range_of_m_if_forall_x_gt_0_l152_152430

open Real

theorem range_of_m_if_forall_x_gt_0 (m : ℝ) :
  (∀ x : ℝ, 0 < x → x + 1/x - m > 0) ↔ m < 2 :=
by
  -- Placeholder proof
  sorry

end range_of_m_if_forall_x_gt_0_l152_152430


namespace complex_square_l152_152388

theorem complex_square (i : ℂ) (h : i^2 = -1) : (1 - i)^2 = -2 * i :=
by
  sorry

end complex_square_l152_152388


namespace polynomial_root_sum_nonnegative_l152_152709

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c
noncomputable def g (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem polynomial_root_sum_nonnegative 
  (m1 m2 k1 k2 b c p q : ℝ)
  (h1 : f m1 b c = 0) (h2 : f m2 b c = 0)
  (h3 : g k1 p q = 0) (h4 : g k2 p q = 0) :
  f k1 b c + f k2 b c + g m1 p q + g m2 p q ≥ 0 := 
by
  sorry  -- Proof placeholders

end polynomial_root_sum_nonnegative_l152_152709


namespace country_albums_count_l152_152989

-- Definitions based on conditions
def pop_albums : Nat := 8
def songs_per_album : Nat := 7
def total_songs : Nat := 70

-- Theorem to prove the number of country albums
theorem country_albums_count : (total_songs - pop_albums * songs_per_album) / songs_per_album = 2 := by
  sorry

end country_albums_count_l152_152989


namespace average_daily_sales_l152_152875

def pens_sold_day_one : ℕ := 96
def pens_sold_next_days : ℕ := 44
def total_days : ℕ := 13

theorem average_daily_sales : (pens_sold_day_one + 12 * pens_sold_next_days) / total_days = 48 := 
by 
  sorry

end average_daily_sales_l152_152875


namespace intersection_of_lines_l152_152956

theorem intersection_of_lines : ∃ (x y : ℝ), 9 * x - 4 * y = 6 ∧ 7 * x + y = 17 ∧ (x, y) = (2, 3) := 
by
  sorry

end intersection_of_lines_l152_152956


namespace B_contribution_l152_152027

theorem B_contribution (A_capital : ℝ) (A_time : ℝ) (B_time : ℝ) (total_profit : ℝ) (A_profit_share : ℝ) (B_contributed : ℝ) :
  A_capital * A_time / (A_capital * A_time + B_contributed * B_time) = A_profit_share / total_profit →
  B_contributed = 6000 :=
by
  intro h
  sorry

end B_contribution_l152_152027


namespace vasya_max_pencils_l152_152341

theorem vasya_max_pencils (money_for_pencils : ℕ) (rebate_20 : ℕ) (rebate_5 : ℕ) :
  money_for_pencils = 30 → rebate_20 = 25 → rebate_5 = 10 → ∃ max_pencils, max_pencils = 36 :=
by
  intros h_money h_r20 h_r5
  sorry

end vasya_max_pencils_l152_152341


namespace problem_statement_l152_152914

noncomputable def pi : ℝ := Real.pi

theorem problem_statement :
  (pi - 1) ^ 0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end problem_statement_l152_152914


namespace intersection_complement_P_Q_l152_152320

def P (x : ℝ) : Prop := x - 1 ≤ 0
def Q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

def complement_P (x : ℝ) : Prop := ¬ P x

theorem intersection_complement_P_Q :
  {x : ℝ | complement_P x} ∩ {x : ℝ | Q x} = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_complement_P_Q_l152_152320


namespace squares_sum_l152_152182

theorem squares_sum {r s : ℝ} (h1 : r * s = 16) (h2 : r + s = 8) : r^2 + s^2 = 32 :=
by
  sorry

end squares_sum_l152_152182


namespace f_zero_add_f_neg_three_l152_152995

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_add (x y : ℝ) : f x + f y = f (x + y)

axiom f_three : f 3 = 4

theorem f_zero_add_f_neg_three : f 0 + f (-3) = -4 :=
by
  sorry

end f_zero_add_f_neg_three_l152_152995


namespace total_bill_cost_l152_152323

-- Definitions of costs and conditions
def curtis_meal_cost : ℝ := 16.00
def rob_meal_cost : ℝ := 18.00
def total_cost_before_discount : ℝ := curtis_meal_cost + rob_meal_cost
def discount_rate : ℝ := 0.5
def time_of_meal : ℝ := 3.0

-- Condition for discount applicability
def discount_applicable : Prop := 2.0 ≤ time_of_meal ∧ time_of_meal ≤ 4.0

-- Total cost with discount applied
def cost_with_discount (total_cost : ℝ) (rate : ℝ) : ℝ := total_cost * rate

-- Theorem statement we need to prove
theorem total_bill_cost :
  discount_applicable →
  cost_with_discount total_cost_before_discount discount_rate = 17.00 :=
by
  sorry

end total_bill_cost_l152_152323


namespace find_family_ages_l152_152381

theorem find_family_ages :
  ∃ (a b father_age mother_age : ℕ), 
    (a < 21) ∧
    (b < 21) ∧
    (a^3 + b^2 > 1900) ∧
    (a^3 + b^2 < 1978) ∧
    (father_age = 1978 - (a^3 + b^2)) ∧
    (mother_age = father_age - 8) ∧
    (a = 12) ∧
    (b = 14) ∧
    (father_age = 54) ∧
    (mother_age = 46) := 
by 
  use 12, 14, 54, 46
  sorry

end find_family_ages_l152_152381


namespace total_carrots_l152_152753

theorem total_carrots (sandy_carrots: Nat) (sam_carrots: Nat) (h1: sandy_carrots = 6) (h2: sam_carrots = 3) : sandy_carrots + sam_carrots = 9 :=
by
  sorry

end total_carrots_l152_152753


namespace sum_first_8_terms_eq_8_l152_152419

noncomputable def arithmetic_sequence_sum (n : ℕ) (a₁ d : ℤ) : ℤ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

theorem sum_first_8_terms_eq_8
  (a : ℕ → ℤ)
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a 1 + ↑n * d)
  (h_a1 : a 1 = 8)
  (h_a4_a6 : a 4 + a 6 = 0) :
  arithmetic_sequence_sum 8 8 (-2) = 8 := 
by
  sorry

end sum_first_8_terms_eq_8_l152_152419


namespace hens_on_farm_l152_152092

theorem hens_on_farm (H R : ℕ) (h1 : H = 9 * R - 5) (h2 : H + R = 75) : H = 67 :=
by
  sorry

end hens_on_farm_l152_152092


namespace collinear_points_l152_152930

theorem collinear_points (k : ℝ) (OA OB OC : ℝ × ℝ) 
  (hOA : OA = (1, -3)) 
  (hOB : OB = (2, -1))
  (hOC : OC = (k + 1, k - 2))
  (h_collinear : ∃ t : ℝ, OC - OA = t • (OB - OA)) : 
  k = 1 :=
by
  have := h_collinear
  sorry

end collinear_points_l152_152930


namespace sample_size_stratified_sampling_l152_152252

theorem sample_size_stratified_sampling 
  (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (n : ℕ) (females_drawn : ℕ) 
  (total_people : ℕ := teachers + male_students + female_students) 
  (females_total : ℕ := female_students) 
  (proportion_drawn : ℚ := (females_drawn : ℚ) / females_total) :
  teachers = 200 → 
  male_students = 1200 → 
  female_students = 1000 → 
  females_drawn = 80 → 
  proportion_drawn = ((n : ℚ) / total_people) → 
  n = 192 :=
by
  sorry

end sample_size_stratified_sampling_l152_152252


namespace simple_interest_l152_152018

/-- Given:
    - Principal (P) = Rs. 80325
    - Rate (R) = 1% per annum
    - Time (T) = 5 years
    Prove that the total simple interest earned (SI) is Rs. 4016.25.
-/
theorem simple_interest (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ)
  (hP : P = 80325)
  (hR : R = 1)
  (hT : T = 5)
  (hSI : SI = P * R * T / 100) :
  SI = 4016.25 :=
by
  sorry

end simple_interest_l152_152018


namespace power_function_condition_direct_proportionality_condition_inverse_proportionality_condition_l152_152008

theorem power_function_condition (m : ℝ) : m^2 + 2 * m = 1 ↔ m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2 :=
by sorry

theorem direct_proportionality_condition (m : ℝ) : (m^2 + m - 1 = 1 ∧ m^2 + 3 * m ≠ 0) ↔ m = 1 :=
by sorry

theorem inverse_proportionality_condition (m : ℝ) : (m^2 + m - 1 = -1 ∧ m^2 + 3 * m ≠ 0) ↔ m = -1 :=
by sorry

end power_function_condition_direct_proportionality_condition_inverse_proportionality_condition_l152_152008


namespace average_of_four_l152_152031

variable {r s t u : ℝ}

theorem average_of_four (h : (5 / 2) * (r + s + t + u) = 20) : (r + s + t + u) / 4 = 2 := 
by 
  sorry

end average_of_four_l152_152031


namespace sock_problem_l152_152324

def sock_pair_count (total_socks : Nat) (socks_distribution : List (String × Nat)) (target_color : String) (different_color : String) : Nat :=
  if target_color = different_color then 0
  else match socks_distribution with
    | [] => 0
    | (color, count) :: tail =>
        if color = target_color then count * socks_distribution.foldl (λ acc (col_count : String × Nat) =>
          if col_count.fst ≠ target_color then acc + col_count.snd else acc) 0
        else sock_pair_count total_socks tail target_color different_color

theorem sock_problem : sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "white" +
                        sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "brown" +
                        sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "blue" =
                        48 :=
by sorry

end sock_problem_l152_152324


namespace max_coconuts_needed_l152_152774

theorem max_coconuts_needed (goats : ℕ) (coconuts_per_crab : ℕ) (crabs_per_goat : ℕ) 
  (final_goats : ℕ) : 
  goats = 19 ∧ coconuts_per_crab = 3 ∧ crabs_per_goat = 6 →
  ∃ coconuts, coconuts = 342 :=
by
  sorry

end max_coconuts_needed_l152_152774


namespace monomial_exponents_l152_152786

theorem monomial_exponents (m n : ℕ) 
  (h1 : m + 1 = 3)
  (h2 : n - 1 = 3) : 
  m^n = 16 := by
  sorry

end monomial_exponents_l152_152786


namespace abs_eq_sum_condition_l152_152454

theorem abs_eq_sum_condition (x y : ℝ) (h : |x - y^2| = x + y^2) : x = 0 ∧ y = 0 :=
  sorry

end abs_eq_sum_condition_l152_152454


namespace complement_union_eq_l152_152731

namespace SetComplementUnion

-- Defining the universal set U, set M and set N.
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

-- Proving the desired equality
theorem complement_union_eq :
  (U \ M) ∪ N = {x | x > -1} :=
sorry

end SetComplementUnion

end complement_union_eq_l152_152731


namespace crayons_per_pack_l152_152083

theorem crayons_per_pack (total_crayons : ℕ) (num_packs : ℕ) (crayons_per_pack : ℕ) 
  (h1 : total_crayons = 615) (h2 : num_packs = 41) : crayons_per_pack = 15 := by
sorry

end crayons_per_pack_l152_152083


namespace man_alone_days_l152_152591

-- Conditions from the problem
variables (M : ℕ) (h1 : (1 / (↑M : ℝ)) + (1 / 12) = 1 / 3)  -- Combined work rate condition

-- The proof statement we need to show
theorem man_alone_days : M = 4 :=
by {
  sorry
}

end man_alone_days_l152_152591


namespace sum_of_cubes_divisible_by_9_l152_152406

theorem sum_of_cubes_divisible_by_9 (n : ℕ) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) := 
  sorry

end sum_of_cubes_divisible_by_9_l152_152406


namespace triangle_area_l152_152965

theorem triangle_area (AB CD : ℝ) (h₁ : 0 < AB) (h₂ : 0 < CD) (h₃ : CD = 3 * AB) :
    let trapezoid_area := 18
    let triangle_ABC_area := trapezoid_area / 4
    triangle_ABC_area = 4.5 := by
  sorry

end triangle_area_l152_152965


namespace sum_of_powers_twice_square_l152_152046

theorem sum_of_powers_twice_square (x y : ℤ) : 
  ∃ z : ℤ, x^4 + y^4 + (x + y)^4 = 2 * z^2 := by
  let z := x^2 + x * y + y^2
  use z
  sorry

end sum_of_powers_twice_square_l152_152046


namespace g_five_l152_152721

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_multiplicative : ∀ x y : ℝ, g (x * y) = g x * g y
axiom g_zero : g 0 = 0
axiom g_one : g 1 = 1

theorem g_five : g 5 = 1 := by
  sorry

end g_five_l152_152721


namespace arithmetic_geometric_sequence_l152_152734

theorem arithmetic_geometric_sequence (x y z : ℤ) :
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  ((x + y + z = 6) ∧ (y - x = z - y) ∧ (y^2 = x * z)) →
  (x = -4 ∧ y = 2 ∧ z = 8 ∨ x = 8 ∧ y = 2 ∧ z = -4) :=
by
  intros h
  sorry

end arithmetic_geometric_sequence_l152_152734


namespace scientific_notation_of_86_million_l152_152455

theorem scientific_notation_of_86_million :
  86000000 = 8.6 * 10^7 :=
sorry

end scientific_notation_of_86_million_l152_152455


namespace solve_equation_l152_152946

theorem solve_equation (x : ℝ) :
  (1 / (x^2 + 17 * x - 8) + 1 / (x^2 + 4 * x - 8) + 1 / (x^2 - 9 * x - 8) = 0) →
  (x = 1 ∨ x = -8 ∨ x = 2 ∨ x = -4) :=
by
  sorry

end solve_equation_l152_152946


namespace cost_per_bag_l152_152180

theorem cost_per_bag (total_bags : ℕ) (sale_price_per_bag : ℕ) (desired_profit : ℕ) (total_revenue : ℕ)
  (total_cost : ℕ) (cost_per_bag : ℕ) :
  total_bags = 100 → sale_price_per_bag = 10 → desired_profit = 300 →
  total_revenue = total_bags * sale_price_per_bag →
  total_cost = total_revenue - desired_profit →
  cost_per_bag = total_cost / total_bags →
  cost_per_bag = 7 := by
  sorry

end cost_per_bag_l152_152180


namespace no_solution_condition_l152_152380

theorem no_solution_condition (m : ℝ) : (∀ x : ℝ, (3 * x - m) / (x - 2) ≠ 1) → m = 6 :=
by
  sorry

end no_solution_condition_l152_152380


namespace find_x_value_l152_152378

theorem find_x_value (x y z k: ℚ)
  (h1 : x = k * (z^3) / (y^2))
  (h2 : y = 2) (h3 : z = 3)
  (h4 : x = 1)
  : x = (4 / 27) * (4^3) / (6^2) := by
  sorry

end find_x_value_l152_152378


namespace second_machine_time_l152_152169

theorem second_machine_time
  (machine1_rate : ℕ)
  (machine2_rate : ℕ)
  (combined_rate12 : ℕ)
  (combined_rate123 : ℕ)
  (rate3 : ℕ)
  (time3 : ℚ) :
  machine1_rate = 60 →
  machine2_rate = 120 →
  combined_rate12 = 200 →
  combined_rate123 = 600 →
  rate3 = 420 →
  time3 = 10 / 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_machine_time_l152_152169


namespace sandy_grew_watermelons_l152_152818

-- Definitions for the conditions
def jason_grew_watermelons : ℕ := 37
def total_watermelons : ℕ := 48

-- Define what we want to prove
theorem sandy_grew_watermelons : total_watermelons - jason_grew_watermelons = 11 := by
  sorry

end sandy_grew_watermelons_l152_152818


namespace det_scaled_matrix_l152_152137

variable (a b c d : ℝ)
variable (h : Matrix.det ![![a, b], ![c, d]] = 5)

theorem det_scaled_matrix : Matrix.det ![![3 * a, 3 * b], ![4 * c, 4 * d]] = 60 := by
  sorry

end det_scaled_matrix_l152_152137


namespace photos_in_each_album_l152_152439

theorem photos_in_each_album (total_photos : ℕ) (number_of_albums : ℕ) (photos_per_album : ℕ) 
    (h1 : total_photos = 2560) 
    (h2 : number_of_albums = 32) 
    (h3 : total_photos = number_of_albums * photos_per_album) : 
    photos_per_album = 80 := 
by 
    sorry

end photos_in_each_album_l152_152439


namespace find_x1_l152_152941

theorem find_x1 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1/4) 
  : x1 = 3/4 := 
sorry

end find_x1_l152_152941


namespace taller_tree_height_l152_152258

variable (T S : ℝ)

theorem taller_tree_height (h1 : T - S = 20)
  (h2 : T - 10 = 3 * (S - 10)) : T = 40 :=
sorry

end taller_tree_height_l152_152258


namespace range_of_m_l152_152927

theorem range_of_m (m : ℝ) (x : ℝ) (h_eq : m / (x - 2) = 3) (h_pos : x > 0) : m > -6 ∧ m ≠ 0 := 
sorry

end range_of_m_l152_152927


namespace system1_l152_152939

theorem system1 {x y : ℝ} 
  (h1 : x + y = 3) 
  (h2 : x - y = 1) : 
  x = 2 ∧ y = 1 :=
by
  sorry

end system1_l152_152939


namespace barber_total_loss_is_120_l152_152872

-- Definitions for the conditions
def haircut_cost : ℕ := 25
def initial_payment_by_customer : ℕ := 50
def flower_shop_change : ℕ := 50
def bakery_change : ℕ := 10
def customer_received_change : ℕ := 25
def counterfeit_50_replacement : ℕ := 50
def counterfeit_10_replacement : ℕ := 10

-- Calculate total loss for the barber
def total_loss : ℕ :=
  let loss_haircut := haircut_cost
  let loss_change_to_customer := customer_received_change
  let loss_given_to_flower_shop := counterfeit_50_replacement
  let loss_given_to_bakery := counterfeit_10_replacement
  let total_loss_before_offset := loss_haircut + loss_change_to_customer + loss_given_to_flower_shop + loss_given_to_bakery
  let real_currency_received := flower_shop_change
  total_loss_before_offset - real_currency_received

-- Proof statement
theorem barber_total_loss_is_120 : total_loss = 120 := by {
  sorry
}

end barber_total_loss_is_120_l152_152872


namespace first_pipe_time_l152_152740

noncomputable def pool_filling_time (T : ℝ) : Prop :=
  (1 / T + 1 / 12 = 1 / 4.8) → (T = 8)

theorem first_pipe_time :
  ∃ T : ℝ, pool_filling_time T := by
  use 8
  sorry

end first_pipe_time_l152_152740


namespace haley_initial_music_files_l152_152477

theorem haley_initial_music_files (M : ℕ) 
  (h1 : M + 42 - 11 = 58) : M = 27 := 
by
  sorry

end haley_initial_music_files_l152_152477


namespace blue_marbles_difference_l152_152279

-- Definitions of the conditions
def total_green_marbles := 95

-- Ratios for Jar 1 and Jar 2
def ratio_blue_green_jar1 := (9, 1)
def ratio_blue_green_jar2 := (8, 1)

-- Total number of green marbles in each jar
def green_marbles_jar1 (a : ℕ) := a
def green_marbles_jar2 (b : ℕ) := b

-- Total number of marbles in each jar
def total_marbles_jar1 (a : ℕ) := 10 * a
def total_marbles_jar2 (b : ℕ) := 9 * b

-- Number of blue marbles in each jar
def blue_marbles_jar1 (a : ℕ) := 9 * a
def blue_marbles_jar2 (b : ℕ) := 8 * b

-- Conditions in terms of Lean definitions
theorem blue_marbles_difference:
  ∀ (a b : ℕ), green_marbles_jar1 a + green_marbles_jar2 b = total_green_marbles →
  total_marbles_jar1 a = total_marbles_jar2 b →
  blue_marbles_jar1 a - blue_marbles_jar2 b = 5 :=
by sorry

end blue_marbles_difference_l152_152279


namespace scientists_nobel_greater_than_not_nobel_by_three_l152_152776

-- Definitions of the given conditions
def total_scientists := 50
def wolf_prize_laureates := 31
def nobel_prize_laureates := 25
def wolf_and_nobel_laureates := 14

-- Derived quantities
def no_wolf_prize := total_scientists - wolf_prize_laureates
def only_wolf_prize := wolf_prize_laureates - wolf_and_nobel_laureates
def only_nobel_prize := nobel_prize_laureates - wolf_and_nobel_laureates
def nobel_no_wolf := only_nobel_prize
def no_wolf_no_nobel := no_wolf_prize - nobel_no_wolf
def difference := nobel_no_wolf - no_wolf_no_nobel

-- The theorem to be proved
theorem scientists_nobel_greater_than_not_nobel_by_three :
  difference = 3 := 
sorry

end scientists_nobel_greater_than_not_nobel_by_three_l152_152776


namespace probability_A_not_lose_l152_152070

theorem probability_A_not_lose (p_win p_draw : ℝ) (h_win : p_win = 0.3) (h_draw : p_draw = 0.5) :
  (p_win + p_draw = 0.8) :=
by
  rw [h_win, h_draw]
  norm_num

end probability_A_not_lose_l152_152070


namespace pi_approx_by_jews_l152_152570

theorem pi_approx_by_jews (S D C : ℝ) (h1 : 4 * S = (5 / 4) * C) (h2 : D = S) (h3 : C = π * D) : π = 3 := by
  sorry

end pi_approx_by_jews_l152_152570


namespace prize_distribution_l152_152845

theorem prize_distribution 
  (total_winners : ℕ)
  (score1 score2 score3 : ℕ)
  (total_points : ℕ) 
  (winners1 winners2 winners3 : ℕ) :
  total_winners = 5 →
  score1 = 20 →
  score2 = 19 →
  score3 = 18 →
  total_points = 94 →
  score1 * winners1 + score2 * winners2 + score3 * winners3 = total_points →
  winners1 + winners2 + winners3 = total_winners →
  winners1 = 1 ∧ winners2 = 2 ∧ winners3 = 2 :=
by
  intros
  sorry

end prize_distribution_l152_152845


namespace same_type_sqrt_l152_152936

theorem same_type_sqrt (x : ℝ) : (x = 2 * Real.sqrt 3) ↔
  (x = Real.sqrt (1/3)) ∨
  (¬(x = Real.sqrt 8) ∧ ¬(x = Real.sqrt 18) ∧ ¬(x = Real.sqrt 9)) :=
by
  sorry

end same_type_sqrt_l152_152936


namespace lars_total_breads_per_day_l152_152413

def loaves_per_hour : ℕ := 10
def baguettes_per_two_hours : ℕ := 30
def hours_per_day : ℕ := 6

theorem lars_total_breads_per_day :
  (loaves_per_hour * hours_per_day) + ((hours_per_day / 2) * baguettes_per_two_hours) = 150 :=
  by 
  sorry

end lars_total_breads_per_day_l152_152413


namespace marissa_sunflower_height_l152_152486

def height_sister_in_inches : ℚ := 4 * 12 + 3
def height_difference_in_inches : ℚ := 21
def inches_to_cm (inches : ℚ) : ℚ := inches * 2.54
def cm_to_m (cm : ℚ) : ℚ := cm / 100

theorem marissa_sunflower_height :
  cm_to_m (inches_to_cm (height_sister_in_inches + height_difference_in_inches)) = 1.8288 :=
by sorry

end marissa_sunflower_height_l152_152486


namespace diamond_value_l152_152049

def diamond (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem diamond_value : diamond 3 4 = 36 := by
  -- Given condition: x ♢ y = 4x + 6y
  -- To prove: (diamond 3 4) = 36
  sorry

end diamond_value_l152_152049


namespace orange_ribbons_count_l152_152016

variable (total_ribbons : ℕ)
variable (orange_ribbons : ℚ)

-- Definitions of the given conditions
def yellow_fraction := (1 : ℚ) / 4
def purple_fraction := (1 : ℚ) / 3
def orange_fraction := (1 : ℚ) / 6
def black_ribbons := 40
def black_fraction := (1 : ℚ) / 4

-- Using the given and derived conditions
theorem orange_ribbons_count
  (hy : yellow_fraction = 1 / 4)
  (hp : purple_fraction = 1 / 3)
  (ho : orange_fraction = 1 / 6)
  (hb : black_ribbons = 40)
  (hbf : black_fraction = 1 / 4)
  (total_eq : total_ribbons = black_ribbons * 4) :
  orange_ribbons = total_ribbons * orange_fraction := by
  -- Proof omitted
  sorry

end orange_ribbons_count_l152_152016


namespace find_m_l152_152871

theorem find_m (m : ℕ) (h1 : m > 0) (h2 : lcm 40 m = 120) (h3 : lcm m 45 = 180) : m = 60 :=
sorry

end find_m_l152_152871


namespace math_problem_l152_152682

variables {a b c d e : ℤ}

theorem math_problem 
(h1 : a - b + c - e = 7)
(h2 : b - c + d + e = 9)
(h3 : c - d + a - e = 5)
(h4 : d - a + b + e = 1)
: a + b + c + d + e = 11 := 
by 
  sorry

end math_problem_l152_152682


namespace range_of_a_l152_152847

noncomputable def proposition_p (a : ℝ) : Prop :=
∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

noncomputable def proposition_q (a : ℝ) : Prop :=
∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0

theorem range_of_a (a : ℝ) (h : proposition_p a ∧ proposition_q a) : a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l152_152847


namespace square_side_length_l152_152867

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
sorry

end square_side_length_l152_152867


namespace range_of_a_l152_152261

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (-2 < x - 1 ∧ x - 1 < 3) ∧ (x - a > 0)) ↔ (a ≤ -1) :=
sorry

end range_of_a_l152_152261


namespace square_in_semicircle_l152_152202

theorem square_in_semicircle (Q : ℝ) (h1 : ∃ Q : ℝ, (Q^2 / 4) + Q^2 = 4) : Q = 4 * Real.sqrt 5 / 5 := sorry

end square_in_semicircle_l152_152202


namespace sum_of_possible_values_of_N_l152_152404

theorem sum_of_possible_values_of_N (N : ℤ) : 
  (N * (N - 8) = 16) -> (∃ a b, N^2 - 8 * N - 16 = 0 ∧ (a + b = 8)) :=
sorry

end sum_of_possible_values_of_N_l152_152404


namespace smallest_sum_of_integers_on_square_vertices_l152_152518

theorem smallest_sum_of_integers_on_square_vertices :
  ∃ (a b c d : ℕ), 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  (a % b = 0 ∨ b % a = 0) ∧ (c % a = 0 ∨ a % c = 0) ∧ 
  (d % b = 0 ∨ b % d = 0) ∧ (d % c = 0 ∨ c % d = 0) ∧ 
  a % c ≠ 0 ∧ a % d ≠ 0 ∧ b % c ≠ 0 ∧ b % d ≠ 0 ∧ 
  (a + b + c + d = 35) := sorry

end smallest_sum_of_integers_on_square_vertices_l152_152518


namespace trigonometric_identity_l152_152998

-- Define the main theorem
theorem trigonometric_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 :=
by
  sorry

end trigonometric_identity_l152_152998


namespace A_is_guilty_l152_152132

-- Define the conditions
variables (A B C : Prop)  -- A, B, C are the propositions that represent the guilt of the individuals A, B, and C
variable  (car : Prop)    -- car represents the fact that the crime involved a car
variable  (C_never_alone : C → A)  -- C never commits a crime without A

-- Facts:
variables (crime_committed : A ∨ B ∨ C) -- the crime was committed by A, B, or C (or a combination)
variable  (B_knows_drive : B → car)     -- B knows how to drive

-- The proof goal: Show that A is guilty.
theorem A_is_guilty : A :=
sorry

end A_is_guilty_l152_152132


namespace power_function_expression_l152_152517

theorem power_function_expression (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x ^ α) (h_point : f 2 = 4) :
  α = 2 ∧ (∀ x, f x = x ^ 2) :=
by
  sorry

end power_function_expression_l152_152517


namespace vasya_numbers_l152_152512

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l152_152512


namespace sarah_bottle_caps_total_l152_152773

def initial_caps : ℕ := 450
def first_day_caps : ℕ := 175
def second_day_caps : ℕ := 95
def third_day_caps : ℕ := 220
def total_caps : ℕ := 940

theorem sarah_bottle_caps_total : 
    initial_caps + first_day_caps + second_day_caps + third_day_caps = total_caps :=
by
  sorry

end sarah_bottle_caps_total_l152_152773


namespace determinant_example_l152_152590

noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)
noncomputable def sin_deg (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

-- Define the determinant of a 2x2 matrix in terms of its entries
def determinant_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Proposed theorem statement in Lean 4
theorem determinant_example : 
  determinant_2x2 (cos_deg 45) (sin_deg 75) (sin_deg 135) (cos_deg 105) = - (Real.sqrt 3 / 2) := 
by sorry

end determinant_example_l152_152590


namespace intersection_A_B_l152_152858

def setA : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem intersection_A_B :
  (setA ∩ setB = {x | 0 ≤ x ∧ x ≤ 2}) :=
by
  sorry

end intersection_A_B_l152_152858


namespace resistor_value_l152_152798

/-- Two resistors with resistance R are connected in series to a DC voltage source U.
    An ideal voltmeter connected in parallel to one resistor shows a reading of 10V.
    The voltmeter is then replaced by an ideal ammeter, which shows a reading of 10A.
    Prove that the resistance R of each resistor is 2Ω. -/
theorem resistor_value (R U U_v I_A : ℝ)
  (hU_v : U_v = 10)
  (hI_A : I_A = 10)
  (hU : U = 2 * U_v)
  (hU_total : U = R * I_A) : R = 2 :=
by
  sorry

end resistor_value_l152_152798


namespace janet_daily_search_time_l152_152364

-- Define the conditions
def minutes_looking_for_keys_per_day (x : ℕ) := 
  let total_time_per_day := x + 3
  let total_time_per_week := 7 * total_time_per_day
  total_time_per_week = 77

-- State the theorem
theorem janet_daily_search_time : 
  ∃ x : ℕ, minutes_looking_for_keys_per_day x ∧ x = 8 := by
  sorry

end janet_daily_search_time_l152_152364


namespace solution_a_l152_152638

noncomputable def problem_a (a b c y : ℕ) : Prop :=
  a + b + c = 30 ∧ b + c + y = 30 ∧ a = 2 ∧ y = 3

theorem solution_a (a b c y x : ℕ)
  (h : problem_a a b c y)
  : x = 25 :=
by sorry

end solution_a_l152_152638


namespace samuel_faster_l152_152309

theorem samuel_faster (S T_h : ℝ) (hT_h : T_h = 1.3) (hS : S = 30) :
  (T_h * 60) - S = 48 :=
by
  sorry

end samuel_faster_l152_152309


namespace totalCandies_l152_152184

def bobCandies : Nat := 10
def maryCandies : Nat := 5
def sueCandies : Nat := 20
def johnCandies : Nat := 5
def samCandies : Nat := 10

theorem totalCandies : bobCandies + maryCandies + sueCandies + johnCandies + samCandies = 50 := 
by
  sorry

end totalCandies_l152_152184


namespace goods_amount_decreased_initial_goods_amount_total_fees_l152_152918

-- Define the conditions as variables
def tonnages : List Int := [31, -31, -16, 34, -38, -20]
def final_goods : Int := 430
def fee_per_ton : Int := 5

-- Prove that the amount of goods in the warehouse has decreased
theorem goods_amount_decreased : (tonnages.sum < 0) := by
  sorry

-- Prove the initial amount of goods in the warehouse
theorem initial_goods_amount : (final_goods + tonnages.sum = 470) := by
  sorry

-- Prove the total loading and unloading fees
theorem total_fees : (tonnages.map Int.natAbs).sum * fee_per_ton = 850 := by
  sorry

end goods_amount_decreased_initial_goods_amount_total_fees_l152_152918


namespace simplify_sum_of_squares_roots_l152_152333

theorem simplify_sum_of_squares_roots :
  Real.sqrt 12 + Real.sqrt 27 + Real.sqrt 48 = 9 * Real.sqrt 3 :=
by
  sorry

end simplify_sum_of_squares_roots_l152_152333


namespace binary_representation_of_38_l152_152561

theorem binary_representation_of_38 : ∃ binary : ℕ, binary = 0b100110 ∧ binary = 38 :=
by
  sorry

end binary_representation_of_38_l152_152561


namespace perimeter_equals_interior_tiles_l152_152434

theorem perimeter_equals_interior_tiles (m n : ℕ) (h : m ≤ n) :
  (2 * m + 2 * n - 4 = 2 * (m * n) - (2 * m + 2 * n - 4)) ↔ (m = 5 ∧ n = 12 ∨ m = 6 ∧ n = 8) :=
by sorry

end perimeter_equals_interior_tiles_l152_152434


namespace solve_equation_l152_152269

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 1) :
  (3 / (x - 2) = 2 / (x - 1)) ↔ (x = -1) :=
sorry

end solve_equation_l152_152269


namespace find_m_l152_152095

def vector (α : Type) := α × α

noncomputable def dot_product {α} [Add α] [Mul α] (a b : vector α) : α :=
a.1 * b.1 + a.2 * b.2

theorem find_m (m : ℝ) (a : vector ℝ) (b : vector ℝ) (h₁ : a = (1, 2)) (h₂ : b = (m, 1)) (h₃ : dot_product a b = 0) : 
m = -2 :=
by
  sorry

end find_m_l152_152095


namespace unique_solution_a_eq_sqrt_three_l152_152408

theorem unique_solution_a_eq_sqrt_three {a : ℝ} (h1 : ∀ x y : ℝ, x^2 + a * abs x + a^2 - 3 = 0 ∧ y^2 + a * abs y + a^2 - 3 = 0 → x = y)
  (h2 : a > 0) : a = Real.sqrt 3 := by
  sorry

end unique_solution_a_eq_sqrt_three_l152_152408


namespace value_of_expression_l152_152770

def delta (a b : ℕ) : ℕ := a * a - b

theorem value_of_expression :
  delta (5 ^ (delta 6 17)) (2 ^ (delta 7 11)) = 5 ^ 38 - 2 ^ 38 :=
by
  sorry

end value_of_expression_l152_152770


namespace order_of_a_add_b_sub_b_l152_152935

variable (a b : ℚ)

theorem order_of_a_add_b_sub_b (hb : b < 0) : a + b < a ∧ a < a - b := by
  sorry

end order_of_a_add_b_sub_b_l152_152935


namespace highest_digit_a_divisible_by_eight_l152_152538

theorem highest_digit_a_divisible_by_eight :
  ∃ a : ℕ, a ≤ 9 ∧ 8 ∣ (100 * a + 16) ∧ ∀ b : ℕ, b > a → b ≤ 9 → ¬ (8 ∣ (100 * b + 16)) := by
  sorry

end highest_digit_a_divisible_by_eight_l152_152538


namespace compute_fraction_l152_152287

theorem compute_fraction : (2015 : ℝ) / ((2015 : ℝ)^2 - (2016 : ℝ) * (2014 : ℝ)) = 2015 :=
by {
  sorry
}

end compute_fraction_l152_152287


namespace sin_double_angle_l152_152669

theorem sin_double_angle {θ : ℝ} (h : Real.tan θ = 1 / 3) : 
  Real.sin (2 * θ) = 3 / 5 := 
  sorry

end sin_double_angle_l152_152669


namespace concert_cost_l152_152215

-- Definitions of the given conditions
def ticket_price : ℝ := 50.00
def num_tickets : ℕ := 2
def processing_fee_rate : ℝ := 0.15
def parking_fee : ℝ := 10.00
def entrance_fee_per_person : ℝ := 5.00
def num_people : ℕ := 2

-- Function to compute the total cost
def total_cost : ℝ :=
  let ticket_total := num_tickets * ticket_price
  let processing_fee := processing_fee_rate * ticket_total
  let total_with_processing := ticket_total + processing_fee
  let total_with_parking := total_with_processing + parking_fee
  let entrance_fee_total := num_people * entrance_fee_per_person
  total_with_parking + entrance_fee_total

-- The proof statement
theorem concert_cost :
  total_cost = 135.00 :=
by
  -- Using the assumptions defined
  let ticket_total := num_tickets * ticket_price
  let processing_fee := processing_fee_rate * ticket_total
  let total_with_processing := ticket_total + processing_fee
  let total_with_parking := total_with_processing + parking_fee
  let entrance_fee_total := num_people * entrance_fee_per_person
  let final_total := total_with_parking + entrance_fee_total
  
  -- Proving the final total
  show final_total = 135.00
  sorry

end concert_cost_l152_152215


namespace Vanya_433_sum_l152_152003

theorem Vanya_433_sum : 
  ∃ (A B : ℕ), 
  A + B = 91 
  ∧ (3 * A + 7 * B = 433) 
  ∧ (∃ (subsetA subsetB : Finset ℕ),
      (∀ x ∈ subsetA, x ∈ Finset.range (13 + 1))
      ∧ (∀ x ∈ subsetB, x ∈ Finset.range (13 + 1))
      ∧ subsetA ∩ subsetB = ∅
      ∧ subsetA ∪ subsetB = Finset.range (13 + 1)
      ∧ subsetA.card = 5
      ∧ subsetA.sum id = A
      ∧ subsetB.sum id = B) :=
by
  sorry

end Vanya_433_sum_l152_152003


namespace paul_erasers_l152_152059

theorem paul_erasers (E : ℕ) (E_crayons : E + 353 = 391) : E = 38 := 
by
  sorry

end paul_erasers_l152_152059


namespace range_of_a1_of_arithmetic_sequence_l152_152652

theorem range_of_a1_of_arithmetic_sequence
  {a : ℕ → ℝ} (S : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum: ∀ n, S n = (n + 1) * (a 0 + a n) / 2)
  (h_min: ∀ n > 0, S n ≥ S 0)
  (h_S1: S 0 = 10) :
  -30 < a 0 ∧ a 0 < -27 := 
sorry

end range_of_a1_of_arithmetic_sequence_l152_152652


namespace unique_prime_sum_diff_l152_152586

-- Define that p is a prime number that satisfies both conditions
def sum_two_primes (p a b : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime a ∧ Nat.Prime b ∧ p = a + b

def diff_two_primes (p c d : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime c ∧ Nat.Prime d ∧ p = c - d

-- Main theorem to prove: The only prime p that satisfies both conditions is 5
theorem unique_prime_sum_diff (p : ℕ) :
  (∃ a b, sum_two_primes p a b) ∧ (∃ c d, diff_two_primes p c d) → p = 5 :=
by
  sorry

end unique_prime_sum_diff_l152_152586


namespace total_students_count_l152_152969

variable (T : ℕ)
variable (J : ℕ) (S : ℕ) (F : ℕ) (Sn : ℕ)

-- Given conditions:
-- 1. 26 percent are juniors.
def percentage_juniors (T J : ℕ) : Prop := J = 26 * T / 100
-- 2. 75 percent are not sophomores.
def percentage_sophomores (T S : ℕ) : Prop := S = 25 * T / 100
-- 3. There are 160 seniors.
def seniors_count (Sn : ℕ) : Prop := Sn = 160
-- 4. There are 32 more freshmen than sophomores.
def freshmen_sophomore_relationship (F S : ℕ) : Prop := F = S + 32

-- Question: Prove the total number of students is 800.
theorem total_students_count
  (hJ : percentage_juniors T J)
  (hS : percentage_sophomores T S)
  (hSn : seniors_count Sn)
  (hF : freshmen_sophomore_relationship F S) :
  F + S + J + Sn = T → T = 800 := by
  sorry

end total_students_count_l152_152969


namespace greatest_possible_y_l152_152062

theorem greatest_possible_y (y : ℕ) (h1 : (y^4 / y^2) < 18) : y ≤ 4 := 
  sorry -- Proof to be filled in later

end greatest_possible_y_l152_152062


namespace area_of_ABCD_l152_152019

theorem area_of_ABCD (x : ℕ) (h1 : 0 < x)
  (h2 : 10 * x = 160) : 4 * x ^ 2 = 1024 := by
  sorry

end area_of_ABCD_l152_152019


namespace range_of_a_l152_152476

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - (a + 1) * x + a ≤ 0 → -4 ≤ x ∧ x ≤ 3) ↔ (-4 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l152_152476


namespace fraction_value_l152_152349

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end fraction_value_l152_152349


namespace interval_length_condition_l152_152159

theorem interval_length_condition (c : ℝ) (x : ℝ) (H1 : 3 ≤ 5 * x - 4) (H2 : 5 * x - 4 ≤ c) 
                                  (H3 : (c + 4) / 5 - 7 / 5 = 15) : c - 3 = 75 := 
sorry

end interval_length_condition_l152_152159


namespace intersection_eq_l152_152551

-- Define the sets M and N
def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := { x | -1 ≤ x ∧ x ≤ 1 }

-- The statement to prove
theorem intersection_eq : M ∩ N = {0, 1} := by
  sorry

end intersection_eq_l152_152551


namespace tim_grew_cantaloupes_l152_152515

theorem tim_grew_cantaloupes (fred_cantaloupes : ℕ) (total_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : total_cantaloupes = 82) :
  ∃ tim_cantaloupes : ℕ, tim_cantaloupes = total_cantaloupes - fred_cantaloupes ∧ tim_cantaloupes = 44 :=
by
  sorry

end tim_grew_cantaloupes_l152_152515


namespace original_recipe_calls_for_4_tablespoons_l152_152458

def key_limes := 8
def juice_per_lime := 1 -- in tablespoons
def juice_doubled := key_limes * juice_per_lime
def original_juice_amount := juice_doubled / 2

theorem original_recipe_calls_for_4_tablespoons :
  original_juice_amount = 4 :=
by
  sorry

end original_recipe_calls_for_4_tablespoons_l152_152458


namespace intersection_of_M_and_N_l152_152417

open Set

def M : Set ℕ := {0, 1, 2, 3}
def N : Set ℕ := {2, 3}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by 
  sorry

end intersection_of_M_and_N_l152_152417


namespace rhombus_area_l152_152917

noncomputable def sqrt125 : ℝ := Real.sqrt 125

theorem rhombus_area 
  (p q : ℝ) 
  (h1 : p < q) 
  (h2 : p + 8 = q) 
  (h3 : ∀ a b : ℝ, a^2 + b^2 = 125 ↔ 2*a = p ∧ 2*b = q) : 
  p*q/2 = 60.5 :=
by
  sorry

end rhombus_area_l152_152917


namespace maximum_value_at_2001_l152_152395
noncomputable def a_n (n : ℕ) : ℝ := n^2 / (1.001^n)

theorem maximum_value_at_2001 : ∃ n : ℕ, n = 2001 ∧ ∀ k : ℕ, a_n k ≤ a_n 2001 := by
  sorry

end maximum_value_at_2001_l152_152395


namespace trip_time_difference_l152_152402

-- Define the speed of the motorcycle
def speed : ℤ := 60

-- Define the distances for the two trips
def distance1 : ℤ := 360
def distance2 : ℤ := 420

-- Define the time calculation function
def time (distance speed : ℤ) : ℤ := distance / speed

-- Prove the problem statement
theorem trip_time_difference : (time distance2 speed - time distance1 speed) * 60 = 60 := by
  -- Provide the proof here
  sorry

end trip_time_difference_l152_152402


namespace cost_of_pencil_and_pens_l152_152547

variable (p q : ℝ)

def equation1 := 3 * p + 4 * q = 3.20
def equation2 := 2 * p + 3 * q = 2.50

theorem cost_of_pencil_and_pens (h1 : equation1 p q) (h2 : equation2 p q) : p + 2 * q = 1.80 := 
by 
  sorry

end cost_of_pencil_and_pens_l152_152547


namespace remainder_when_s_div_6_is_5_l152_152766

theorem remainder_when_s_div_6_is_5 (s t : ℕ) (h1 : s > t) (Rs Rt : ℕ) (h2 : s % 6 = Rs) (h3 : t % 6 = Rt) (h4 : (s - t) % 6 = 5) : Rs = 5 := 
by
  sorry

end remainder_when_s_div_6_is_5_l152_152766


namespace union_of_A_B_l152_152932

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x > 0}

theorem union_of_A_B :
  A ∪ B = {x | x ≥ -1} := by
  sorry

end union_of_A_B_l152_152932


namespace find_m_from_equation_l152_152718

theorem find_m_from_equation :
  ∀ (x m : ℝ), (x^2 + 2 * x - 1 = 0) → ((x + m)^2 = 2) → m = 1 :=
by
  intros x m h1 h2
  sorry

end find_m_from_equation_l152_152718


namespace probability_white_given_red_l152_152924

-- Define the total number of balls initially
def total_balls := 10

-- Define the number of red balls, white balls, and black balls
def red_balls := 3
def white_balls := 2
def black_balls := 5

-- Define the event A: Picking a red ball on the first draw
def event_A := red_balls

-- Define the event B: Picking a white ball on the second draw
-- Number of balls left after picking one red ball
def remaining_balls_after_A := total_balls - 1

-- Define the event AB: Picking a red ball first and then a white ball
def event_AB := red_balls * white_balls

-- Calculate the probability P(B|A)
def P_B_given_A := event_AB / (event_A * remaining_balls_after_A)

-- Prove the probability of picking a white ball on the second draw given that the first ball picked is a red ball
theorem probability_white_given_red : P_B_given_A = (2 / 9) := by
  sorry

end probability_white_given_red_l152_152924


namespace minimum_value_function_l152_152186

theorem minimum_value_function (x : ℝ) (h : x > 1) : 
  ∃ y, y = (16 - 2 * Real.sqrt 7) / 3 ∧ ∀ x > 1, (4*x^2 + 2*x + 5) / (x^2 + x + 1) ≥ y :=
sorry

end minimum_value_function_l152_152186


namespace speed_of_current_l152_152351

variables (b c : ℝ)

theorem speed_of_current (h1 : b + c = 12) (h2 : b - c = 4) : c = 4 :=
sorry

end speed_of_current_l152_152351


namespace sum_fractions_l152_152172

theorem sum_fractions:
  (Finset.range 16).sum (λ k => (k + 1) / 7) = 136 / 7 := by
  sorry

end sum_fractions_l152_152172


namespace cristina_running_pace_4point2_l152_152243

theorem cristina_running_pace_4point2 :
  ∀ (nicky_pace head_start time_after_start cristina_pace : ℝ),
    nicky_pace = 3 →
    head_start = 12 →
    time_after_start = 30 →
    cristina_pace = 4.2 →
    (time_after_start = head_start + 30 →
    cristina_pace * time_after_start = nicky_pace * (head_start + 30)) :=
by
  sorry

end cristina_running_pace_4point2_l152_152243


namespace percentage_reduction_correct_l152_152959

-- Define the initial conditions
def initial_conditions (P S : ℝ) (new_sales_increase_percentage net_sale_value_increase_percentage: ℝ) :=
  new_sales_increase_percentage = 0.72 ∧ net_sale_value_increase_percentage = 0.4104

-- Define the statement for the required percentage reduction
theorem percentage_reduction_correct (P S : ℝ) (x : ℝ) 
  (h : initial_conditions P S 0.72 0.4104) : 
  (S:ℝ) * (1 - x / 100) = 1.4104 * S := 
sorry

end percentage_reduction_correct_l152_152959


namespace arcsin_one_half_eq_pi_six_l152_152134

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := 
by
  sorry

end arcsin_one_half_eq_pi_six_l152_152134


namespace molecular_weight_l152_152167

noncomputable def molecular_weight_of_one_mole : ℕ → ℝ :=
  fun n => if n = 1 then 78 else n * 78

theorem molecular_weight (n: ℕ) (hn: n > 0) (condition: ∃ k: ℕ, k = 4 ∧ 312 = k * 78) :
  molecular_weight_of_one_mole n = 78 * n :=
by
  sorry

end molecular_weight_l152_152167


namespace part_I_part_II_l152_152806

def f (x : ℝ) (m : ℕ) : ℝ := |x - m| + |x|

theorem part_I (m : ℕ) (hm : m = 1) : ∃ x : ℝ, f x m < 2 :=
by sorry

theorem part_II (α β : ℝ) (hα : 1 < α) (hβ : 1 < β) (h : f α 1 + f β 1 = 2) :
  (4 / α) + (1 / β) ≥ 9 / 2 :=
by sorry

end part_I_part_II_l152_152806


namespace original_price_of_item_l152_152410

theorem original_price_of_item (P : ℝ) 
(selling_price : ℝ) 
(h1 : 0.9 * P = selling_price) 
(h2 : selling_price = 675) : 
P = 750 := sorry

end original_price_of_item_l152_152410


namespace percentage_difference_l152_152293

-- Define the quantities involved
def milk_in_A : ℕ := 1264
def transferred_milk : ℕ := 158

-- Define the quantities of milk in container B and C after transfer
noncomputable def quantity_in_B : ℕ := milk_in_A / 2
noncomputable def quantity_in_C : ℕ := quantity_in_B

-- Prove that the percentage difference between the quantity of milk in container B
-- and the capacity of container A is 50%
theorem percentage_difference :
  ((milk_in_A - quantity_in_B) * 100 / milk_in_A) = 50 := sorry

end percentage_difference_l152_152293


namespace distance_gracie_joe_l152_152905

noncomputable def distance_between_points := Real.sqrt (5^2 + (-1)^2)
noncomputable def joe_point := Complex.mk 3 (-4)
noncomputable def gracie_point := Complex.mk (-2) (-3)

theorem distance_gracie_joe : Complex.abs (joe_point - gracie_point) = distance_between_points := by 
  sorry

end distance_gracie_joe_l152_152905


namespace price_of_food_before_tax_and_tip_l152_152385

noncomputable def actual_price_of_food (total_paid : ℝ) (tip_rate tax_rate : ℝ) : ℝ :=
  total_paid / (1 + tip_rate) / (1 + tax_rate)

theorem price_of_food_before_tax_and_tip :
  actual_price_of_food 211.20 0.20 0.10 = 160 :=
by
  sorry

end price_of_food_before_tax_and_tip_l152_152385


namespace triangle_inequality_l152_152198

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  2 < (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ∧
  (a + b) / c + (b + c) / a + (c + a) / b - (a^3 + b^3 + c^3) / (a * b * c) ≤ 3 :=
sorry

end triangle_inequality_l152_152198


namespace constant_sequence_from_conditions_l152_152620

variable (k b : ℝ) [Nontrivial ℝ]
variable (a_n : ℕ → ℝ)

-- Define the conditions function
def cond1 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (q : ℝ), ∀ n, a_n (n + 1) = q * a_n n) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b = m * (k * a_n n + b))

def cond2 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (d : ℝ), ∀ n, a_n (n + 1) = a_n n + d) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b = m * (k * a_n n + b))

def cond3 (k b : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∃ (q : ℝ), ∀ n, a_n (n + 1) = q * a_n n) ∧ 
  (∃ (m : ℝ), ∀ n, k * a_n (n + 1) + b - (k * a_n n + b) = m)

-- Lean statement to prove the problem
theorem constant_sequence_from_conditions (k b : ℝ) [Nontrivial ℝ] (a_n : ℕ → ℝ) :
  (cond1 k b a_n ∨ cond2 k b a_n ∨ cond3 k b a_n) → 
  ∃ c : ℝ, ∀ n, a_n n = c :=
by
  -- To be proven
  intros
  sorry

end constant_sequence_from_conditions_l152_152620


namespace ant_weight_statement_l152_152970

variable (R : ℝ) -- Rupert's weight
variable (A : ℝ) -- Antoinette's weight
variable (C : ℝ) -- Charles's weight

-- Conditions
def condition1 : Prop := A = 2 * R - 7
def condition2 : Prop := C = (A + R) / 2 + 5
def condition3 : Prop := A + R + C = 145

-- Question: Prove Antoinette's weight
def ant_weight_proof : Prop :=
  ∃ R A C, condition1 R A ∧ condition2 R A C ∧ condition3 R A C ∧ A = 79

theorem ant_weight_statement : ant_weight_proof :=
sorry

end ant_weight_statement_l152_152970


namespace minimum_sum_of_reciprocals_l152_152398

open BigOperators

theorem minimum_sum_of_reciprocals (b : Fin 15 → ℝ) (h_pos : ∀ i, 0 < b i)
    (h_sum : ∑ i, b i = 1) :
    ∑ i, 1 / (b i) ≥ 225 := sorry

end minimum_sum_of_reciprocals_l152_152398


namespace min_sum_of_squares_l152_152649

theorem min_sum_of_squares (y1 y2 y3 : ℝ) (h1 : y1 > 0) (h2 : y2 > 0) (h3 : y3 > 0) (h4 : y1 + 3 * y2 + 4 * y3 = 72) : 
  y1^2 + y2^2 + y3^2 ≥ 2592 / 13 ∧ (∃ k, y1 = k ∧ y2 = 3 * k ∧ y3 = 4 * k ∧ k = 36 / 13) :=
sorry

end min_sum_of_squares_l152_152649


namespace find_three_digit_integers_mod_l152_152328

theorem find_three_digit_integers_mod (n : ℕ) :
  (n % 7 = 3) ∧ (n % 8 = 6) ∧ (n % 5 = 2) ∧ (100 ≤ n) ∧ (n < 1000) :=
sorry

end find_three_digit_integers_mod_l152_152328


namespace max_band_members_l152_152313

theorem max_band_members (n : ℤ) (h1 : 20 * n % 31 = 11) (h2 : 20 * n < 1200) : 20 * n = 1100 :=
sorry

end max_band_members_l152_152313


namespace part1_part2_part3_l152_152391

-- Defining the quadratic function
def quadratic (t : ℝ) (x : ℝ) : ℝ := x^2 - 2 * t * x + 3

-- Part (1)
theorem part1 (t : ℝ) (h : quadratic t 2 = 1) : t = 3 / 2 :=
by sorry

-- Part (2)
theorem part2 (t : ℝ) (h : ∀x, 0 ≤ x → x ≤ 3 → (quadratic t x) ≥ -2) : t = Real.sqrt 5 :=
by sorry

-- Part (3)
theorem part3 (m a b : ℝ) (hA : quadratic t (m - 2) = a) (hB : quadratic t 4 = b) 
              (hC : quadratic t m = a) (ha : a < b) (hb : b < 3) (ht : t > 0) : 
              (3 < m ∧ m < 4) ∨ (m > 6) :=
by sorry

end part1_part2_part3_l152_152391


namespace find_a3_plus_a5_l152_152981

variable {a : ℕ → ℝ}

-- Condition 1: The sequence {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Condition 2: All terms in the sequence are negative
def all_negative (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < 0

-- Condition 3: The given equation
def given_equation (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25

-- The problem statement
theorem find_a3_plus_a5 (h_geo : is_geometric_sequence a) (h_neg : all_negative a) (h_eq : given_equation a) :
  a 3 + a 5 = -5 :=
sorry

end find_a3_plus_a5_l152_152981


namespace non_degenerate_ellipse_condition_l152_152262

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ x y : ℝ, 9 * x^2 + y^2 - 18 * x - 2 * y = k) ↔ k > -10 :=
sorry

end non_degenerate_ellipse_condition_l152_152262


namespace third_term_of_sequence_l152_152221

theorem third_term_of_sequence :
  (3 - (1 / 3) = 8 / 3) :=
by
  sorry

end third_term_of_sequence_l152_152221


namespace max_true_statements_l152_152363

theorem max_true_statements (a b : ℝ) :
  ((a < b) → (b < 0) → (a < 0) → ¬(1 / a < 1 / b)) ∧
  ((a < b) → (b < 0) → (a < 0) → ¬(a^2 < b^2)) →
  3 = 3
:=
by
  intros
  sorry

end max_true_statements_l152_152363


namespace smallest_integer_k_no_real_roots_l152_152389

def quadratic_no_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c < 0

theorem smallest_integer_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, quadratic_no_real_roots (2 * k - 1) (-8) 6) ∧ (k = 2) :=
by
  sorry

end smallest_integer_k_no_real_roots_l152_152389


namespace period_of_f_l152_152629

noncomputable def f : ℝ → ℝ := sorry

def functional_equation (f : ℝ → ℝ) := ∀ x y : ℝ, f (2 * x) + f (2 * y) = f (x + y) * f (x - y)

def f_pi_zero (f : ℝ → ℝ) := f (Real.pi) = 0

def f_not_identically_zero (f : ℝ → ℝ) := ∃ x : ℝ, f x ≠ 0

theorem period_of_f (f : ℝ → ℝ)
  (hf_eq : functional_equation f)
  (hf_pi_zero : f_pi_zero f)
  (hf_not_zero : f_not_identically_zero f) : 
  ∀ x : ℝ, f (x + 4 * Real.pi) = f x := sorry

end period_of_f_l152_152629


namespace triangle_is_obtuse_l152_152659

theorem triangle_is_obtuse
  (A B C : ℝ)
  (h1 : 3 * A > 5 * B)
  (h2 : 3 * C < 2 * B)
  (h3 : A + B + C = 180) :
  A > 90 :=
sorry

end triangle_is_obtuse_l152_152659


namespace complex_number_solution_l152_152147

theorem complex_number_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) (h_i : z * i = 2 + i) : z = 1 - 2 * i := by
  sorry

end complex_number_solution_l152_152147


namespace zeros_of_f_l152_152767

noncomputable def f (a : ℝ) (x : ℝ) :=
if x ≤ 1 then a + 2^x else (1/2) * x + a

theorem zeros_of_f (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ a ∈ Set.Ico (-2) (-1/2) :=
sorry

end zeros_of_f_l152_152767


namespace mod_81256_eq_16_l152_152420

theorem mod_81256_eq_16 : ∃ n : ℤ, 0 ≤ n ∧ n < 31 ∧ 81256 % 31 = n := by
  use 16
  sorry

end mod_81256_eq_16_l152_152420


namespace total_plant_count_l152_152491

-- Definitions for conditions.
def total_rows : ℕ := 96
def columns_per_row : ℕ := 24
def divided_rows : ℕ := total_rows / 3
def undivided_rows : ℕ := total_rows - divided_rows
def beans_in_undivided_row : ℕ := columns_per_row
def corn_in_divided_row : ℕ := columns_per_row / 2
def tomatoes_in_divided_row : ℕ := columns_per_row / 2

-- Total number of plants calculation.
def total_bean_plants : ℕ := undivided_rows * beans_in_undivided_row
def total_corn_plants : ℕ := divided_rows * corn_in_divided_row
def total_tomato_plants : ℕ := divided_rows * tomatoes_in_divided_row

def total_plants : ℕ := total_bean_plants + total_corn_plants + total_tomato_plants

-- Proof statement.
theorem total_plant_count : total_plants = 2304 :=
by
  sorry

end total_plant_count_l152_152491


namespace bus_speed_excluding_stoppages_l152_152860

theorem bus_speed_excluding_stoppages (s_including_stops : ℕ) (stop_time_minutes : ℕ) (s_excluding_stops : ℕ) (v : ℕ) : 
  (s_including_stops = 45) ∧ (stop_time_minutes = 24) ∧ (v = s_including_stops * 5 / 3) → s_excluding_stops = 75 := 
by {
  sorry
}

end bus_speed_excluding_stoppages_l152_152860


namespace gas_tank_size_l152_152801

-- Conditions from part a)
def advertised_mileage : ℕ := 35
def actual_mileage : ℕ := 31
def total_miles_driven : ℕ := 372

-- Question and the correct answer in the context of conditions
theorem gas_tank_size (h1 : actual_mileage = advertised_mileage - 4) 
                      (h2 : total_miles_driven = 372) 
                      : total_miles_driven / actual_mileage = 12 := 
by sorry

end gas_tank_size_l152_152801


namespace integer_points_count_l152_152861

theorem integer_points_count :
  ∃ (n : ℤ), n = 9 ∧
  ∀ a b : ℝ, (1 < a) → (1 < b) → (ab + a - b - 10 = 0) →
  (a + b = 6) → 
  ∃ (x y : ℤ), (3 * x^2 + 2 * y^2 ≤ 6) :=
by
  sorry

end integer_points_count_l152_152861


namespace age_difference_l152_152362

theorem age_difference (x : ℕ) (older_age younger_age : ℕ) 
  (h1 : 3 * x = older_age)
  (h2 : 2 * x = younger_age)
  (h3 : older_age + younger_age = 60) : 
  older_age - younger_age = 12 := 
by
  sorry

end age_difference_l152_152362


namespace factorize_x4_minus_16_factorize_trinomial_l152_152916

-- For problem 1: Factorization of \( x^4 - 16 \)
theorem factorize_x4_minus_16 (x : ℝ) : 
  x^4 - 16 = (x - 2) * (x + 2) * (x^2 + 4) := 
sorry

-- For problem 2: Factorization of \( -9x^2y + 12xy^2 - 4y^3 \)
theorem factorize_trinomial (x y : ℝ) : 
  -9 * x^2 * y + 12 * x * y^2 - 4 * y^3 = -y * (3 * x - 2 * y)^2 := 
sorry

end factorize_x4_minus_16_factorize_trinomial_l152_152916


namespace maximum_value_l152_152703

noncomputable def conditions (m n t : ℝ) : Prop :=
  -- m, n, t are positive real numbers
  (0 < m) ∧ (0 < n) ∧ (0 < t) ∧
  -- Equation condition
  (m^2 - 3 * m * n + 4 * n^2 - t = 0)

noncomputable def minimum_u (m n t : ℝ) : Prop :=
  -- Minimum value condition for t / mn
  (t / (m * n) = 1)

theorem maximum_value (m n t : ℝ) (h1 : conditions m n t) (h2 : minimum_u m n t) :
  -- Proving the maximum value of m + 2n - t
  (m + 2 * n - t) = 2 :=
sorry

end maximum_value_l152_152703


namespace number_of_five_digit_numbers_with_at_least_one_zero_l152_152568

-- Definitions for the conditions
def total_five_digit_numbers : ℕ := 90000
def five_digit_numbers_with_no_zeros : ℕ := 59049

-- Theorem stating that the number of 5-digit numbers with at least one zero is 30,951
theorem number_of_five_digit_numbers_with_at_least_one_zero : 
    total_five_digit_numbers - five_digit_numbers_with_no_zeros = 30951 :=
by
  sorry

end number_of_five_digit_numbers_with_at_least_one_zero_l152_152568


namespace no_real_roots_x_squared_minus_x_plus_nine_l152_152181

theorem no_real_roots_x_squared_minus_x_plus_nine :
  ∀ x : ℝ, ¬ (x^2 - x + 9 = 0) :=
by 
  intro x 
  sorry

end no_real_roots_x_squared_minus_x_plus_nine_l152_152181


namespace piastres_in_6th_purse_l152_152894

theorem piastres_in_6th_purse (x : ℕ) (sum : ℕ := 10) (total : ℕ := 150)
  (h1 : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) = 150)
  (h2 : x * 2 ≥ x + 9)
  (n : ℕ := 5):
  x + n = 15 :=
  sorry

end piastres_in_6th_purse_l152_152894


namespace number_of_poison_frogs_l152_152251

theorem number_of_poison_frogs
  (total_frogs : ℕ) (tree_frogs : ℕ) (wood_frogs : ℕ) (poison_frogs : ℕ)
  (h₁ : total_frogs = 78)
  (h₂ : tree_frogs = 55)
  (h₃ : wood_frogs = 13)
  (h₄ : total_frogs = tree_frogs + wood_frogs + poison_frogs) :
  poison_frogs = 10 :=
by sorry

end number_of_poison_frogs_l152_152251


namespace total_hotdogs_sold_l152_152294

-- Define the number of small and large hotdogs
def small_hotdogs : ℕ := 58
def large_hotdogs : ℕ := 21

-- Define the total hotdogs
def total_hotdogs : ℕ := small_hotdogs + large_hotdogs

-- The Main Statement to prove the total number of hotdogs sold
theorem total_hotdogs_sold : total_hotdogs = 79 :=
by
  -- Proof is skipped using sorry
  sorry

end total_hotdogs_sold_l152_152294


namespace correct_proportion_l152_152799

theorem correct_proportion {a b c x y : ℝ} 
  (h1 : x + y = b)
  (h2 : x * c = y * a) :
  y / a = b / (a + c) :=
sorry

end correct_proportion_l152_152799


namespace arithmetic_sequence_sixtieth_term_l152_152775

theorem arithmetic_sequence_sixtieth_term (a₁ a₂₁ a₆₀ d : ℕ) 
  (h1 : a₁ = 7)
  (h2 : a₂₁ = 47)
  (h3 : a₂₁ = a₁ + 20 * d) : 
  a₆₀ = a₁ + 59 * d := 
  by
  have HD : d = 2 := by 
    rw [h1] at h3
    rw [h2] at h3
    linarith
  rw [HD]
  rw [h1]
  sorry

end arithmetic_sequence_sixtieth_term_l152_152775


namespace renata_lottery_winnings_l152_152100

def initial_money : ℕ := 10
def donation : ℕ := 4
def prize_won : ℕ := 90
def water_cost : ℕ := 1
def lottery_ticket_cost : ℕ := 1
def final_money : ℕ := 94

theorem renata_lottery_winnings :
  ∃ (lottery_winnings : ℕ), 
  initial_money - donation + prize_won 
  - water_cost - lottery_ticket_cost 
  = final_money ∧ 
  lottery_winnings = 2 :=
by
  -- Proof steps will go here
  sorry

end renata_lottery_winnings_l152_152100


namespace original_time_40_l152_152227

theorem original_time_40
  (S T : ℝ)
  (h1 : ∀ D : ℝ, D = S * T)
  (h2 : ∀ D : ℝ, D = 0.8 * S * (T + 10)) :
  T = 40 :=
by
  sorry

end original_time_40_l152_152227


namespace smallest_m_for_integral_solutions_l152_152061

theorem smallest_m_for_integral_solutions : ∃ (m : ℕ), (∀ (p q : ℤ), (15 * (p * p) - m * p + 630 = 0 ∧ 15 * (q * q) - m * q + 630 = 0) → (m = 195)) :=
sorry

end smallest_m_for_integral_solutions_l152_152061


namespace Jose_got_5_questions_wrong_l152_152888

def Jose_questions_wrong (M J A : ℕ) : Prop :=
  M = J - 20 ∧
  J = A + 40 ∧
  M + J + A = 210 ∧
  (50 * 2 = 100) ∧
  (100 - J) / 2 = 5

theorem Jose_got_5_questions_wrong (M J A : ℕ) (h1 : M = J - 20) (h2 : J = A + 40) (h3 : M + J + A = 210) : 
  Jose_questions_wrong M J A :=
by
  sorry

end Jose_got_5_questions_wrong_l152_152888


namespace lines_per_page_l152_152058

theorem lines_per_page
  (total_words : ℕ)
  (words_per_line : ℕ)
  (words_left : ℕ)
  (pages_filled : ℚ) :
  total_words = 400 →
  words_per_line = 10 →
  words_left = 100 →
  pages_filled = 1.5 →
  (total_words - words_left) / words_per_line / pages_filled = 20 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end lines_per_page_l152_152058


namespace manufacturer_A_more_cost_effective_manufacturer_B_more_cost_effective_l152_152608

-- Definitions of costs and the problem conditions.
def cost_manufacturer_A (desks chairs : ℕ) : ℝ :=
  200 * desks + 50 * (chairs - desks)

def cost_manufacturer_B (desks chairs : ℕ) : ℝ :=
  0.9 * (200 * desks + 50 * chairs)

-- Given condition: School needs 60 desks.
def desks : ℕ := 60

-- (1) Prove manufacturer A is more cost-effective when x < 360.
theorem manufacturer_A_more_cost_effective (chairs : ℕ) (h : chairs ≥ 60) :
  chairs < 360 → cost_manufacturer_A desks chairs < cost_manufacturer_B desks chairs :=
by sorry

-- (2) Prove manufacturer B is more cost-effective when x > 360.
theorem manufacturer_B_more_cost_effective (chairs : ℕ) (h : chairs ≥ 60) :
  chairs > 360 → cost_manufacturer_A desks chairs > cost_manufacturer_B desks chairs :=
by sorry

end manufacturer_A_more_cost_effective_manufacturer_B_more_cost_effective_l152_152608


namespace total_wait_days_l152_152277

-- Definitions based on the conditions
def days_first_appointment := 4
def days_second_appointment := 20
def days_vaccine_effective := 2 * 7  -- 2 weeks converted to days

-- Theorem stating the total wait time
theorem total_wait_days : days_first_appointment + days_second_appointment + days_vaccine_effective = 38 := by
  sorry

end total_wait_days_l152_152277


namespace find_natural_pairs_l152_152216

theorem find_natural_pairs (m n : ℕ) :
  (n * (n - 1) * (n - 2) * (n - 3) = m * (m - 1)) ↔ (n = 1 ∧ m = 1) ∨ (n = 2 ∧ m = 1) ∨ (n = 3 ∧ m = 1) :=
by sorry

end find_natural_pairs_l152_152216


namespace smallest_prime_factor_of_1917_l152_152091

theorem smallest_prime_factor_of_1917 : ∃ p : ℕ, Prime p ∧ (p ∣ 1917) ∧ (∀ q : ℕ, Prime q ∧ (q ∣ 1917) → q ≥ p) :=
by
  sorry

end smallest_prime_factor_of_1917_l152_152091


namespace special_hash_calculation_l152_152117

-- Definition of the operation #
def special_hash (a b : ℤ) : ℚ := 2 * a + (a / b) + 3

-- Statement of the proof problem
theorem special_hash_calculation : special_hash 7 3 = 19 + 1/3 := 
by 
  sorry

end special_hash_calculation_l152_152117


namespace bathroom_area_l152_152436

def tile_size : ℝ := 0.5 -- Each tile is 0.5 feet

structure Section :=
  (width : ℕ)
  (length : ℕ)

def longer_section : Section := ⟨15, 25⟩
def alcove : Section := ⟨10, 8⟩

def area (s : Section) : ℝ := (s.width * tile_size) * (s.length * tile_size)

theorem bathroom_area :
  area longer_section + area alcove = 113.75 := by
  sorry

end bathroom_area_l152_152436


namespace tan_ratio_is_7_over_3_l152_152949

open Real

theorem tan_ratio_is_7_over_3 (a b : ℝ) (h1 : sin (a + b) = 5 / 8) (h2 : sin (a - b) = 1 / 4) : (tan a / tan b) = 7 / 3 :=
by
  sorry

end tan_ratio_is_7_over_3_l152_152949


namespace daily_wage_of_a_man_l152_152838

theorem daily_wage_of_a_man (M W : ℝ) 
  (h1 : 24 * M + 16 * W = 11600) 
  (h2 : 12 * M + 37 * W = 11600) : 
  M = 350 :=
by
  sorry

end daily_wage_of_a_man_l152_152838


namespace interval_sum_l152_152080

theorem interval_sum (m n : ℚ) (h : ∀ x : ℚ, m < x ∧ x < n ↔ (mx - 1) / (x + 3) > 0) :
  m + n = -10 / 3 :=
sorry

end interval_sum_l152_152080


namespace was_not_speeding_l152_152940

theorem was_not_speeding (x s : ℝ) (s_obs : ℝ := 26.5) (x_limit : ℝ := 120)
  (brake_dist_eq : s = 0.01 * x + 0.002 * x^2) : s_obs < 30 → x ≤ x_limit :=
sorry

end was_not_speeding_l152_152940


namespace avg_salary_officers_correct_l152_152796

def total_employees := 465
def avg_salary_employees := 120
def non_officers := 450
def avg_salary_non_officers := 110
def officers := 15

theorem avg_salary_officers_correct : (15 * 420) = ((total_employees * avg_salary_employees) - (non_officers * avg_salary_non_officers)) := by
  sorry

end avg_salary_officers_correct_l152_152796


namespace paint_per_large_canvas_l152_152206

-- Define the conditions
variables (L : ℕ) (paint_large paint_small total_paint : ℕ)

-- Given conditions
def large_canvas_paint := 3 * L
def small_canvas_paint := 4 * 2
def total_paint_used := large_canvas_paint + small_canvas_paint

-- Statement that needs to be proven
theorem paint_per_large_canvas :
  total_paint_used = 17 → L = 3 :=
by
  intro h
  sorry

end paint_per_large_canvas_l152_152206


namespace jessa_needs_470_cupcakes_l152_152852

def total_cupcakes_needed (fourth_grade_classes : ℕ) (students_per_fourth_grade_class : ℕ) (pe_class_students : ℕ) (afterschool_clubs : ℕ) (students_per_afterschool_club : ℕ) : ℕ :=
  (fourth_grade_classes * students_per_fourth_grade_class) + pe_class_students + (afterschool_clubs * students_per_afterschool_club)

theorem jessa_needs_470_cupcakes :
  total_cupcakes_needed 8 40 80 2 35 = 470 :=
by
  sorry

end jessa_needs_470_cupcakes_l152_152852


namespace find_f_neg_9_over_2_l152_152376

noncomputable def f : ℝ → ℝ
| x => if 0 ≤ x ∧ x ≤ 1 then 2^x else sorry

theorem find_f_neg_9_over_2
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hf_periodic : ∀ x : ℝ, f (x + 2) = f x)
  (hf_definition : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = 2^x) :
  f (-9 / 2) = Real.sqrt 2 := by
  sorry

end find_f_neg_9_over_2_l152_152376


namespace abs_neg_one_div_three_l152_152742

open Real

theorem abs_neg_one_div_three : abs (-1 / 3) = 1 / 3 :=
by
  sorry

end abs_neg_one_div_three_l152_152742


namespace all_positive_l152_152322

theorem all_positive (a1 a2 a3 a4 a5 a6 a7 : ℝ)
  (h1 : a1 + a2 + a3 + a4 > a5 + a6 + a7)
  (h2 : a1 + a2 + a3 + a5 > a4 + a6 + a7)
  (h3 : a1 + a2 + a3 + a6 > a4 + a5 + a7)
  (h4 : a1 + a2 + a3 + a7 > a4 + a5 + a6)
  (h5 : a1 + a2 + a4 + a5 > a3 + a6 + a7)
  (h6 : a1 + a2 + a4 + a6 > a3 + a5 + a7)
  (h7 : a1 + a2 + a4 + a7 > a3 + a5 + a6)
  (h8 : a1 + a2 + a5 + a6 > a3 + a4 + a7)
  (h9 : a1 + a2 + a5 + a7 > a3 + a4 + a6)
  (h10 : a1 + a2 + a6 + a7 > a3 + a4 + a5)
  (h11 : a1 + a3 + a4 + a5 > a2 + a6 + a7)
  (h12 : a1 + a3 + a4 + a6 > a2 + a5 + a7)
  (h13 : a1 + a3 + a4 + a7 > a2 + a5 + a6)
  (h14 : a1 + a3 + a5 + a6 > a2 + a4 + a7)
  (h15 : a1 + a3 + a5 + a7 > a2 + a4 + a6)
  (h16 : a1 + a3 + a6 + a7 > a2 + a4 + a5)
  (h17 : a1 + a4 + a5 + a6 > a2 + a3 + a7)
  (h18 : a1 + a4 + a5 + a7 > a2 + a3 + a6)
  (h19 : a1 + a4 + a6 + a7 > a2 + a3 + a5)
  (h20 : a1 + a5 + a6 + a7 > a2 + a3 + a4)
  (h21 : a2 + a3 + a4 + a5 > a1 + a6 + a7)
  (h22 : a2 + a3 + a4 + a6 > a1 + a5 + a7)
  (h23 : a2 + a3 + a4 + a7 > a1 + a5 + a6)
  (h24 : a2 + a3 + a5 + a6 > a1 + a4 + a7)
  (h25 : a2 + a3 + a5 + a7 > a1 + a4 + a6)
  (h26 : a2 + a3 + a6 + a7 > a1 + a4 + a5)
  (h27 : a2 + a4 + a5 + a6 > a1 + a3 + a7)
  (h28 : a2 + a4 + a5 + a7 > a1 + a3 + a6)
  (h29 : a2 + a4 + a6 + a7 > a1 + a3 + a5)
  (h30 : a2 + a5 + a6 + a7 > a1 + a3 + a4)
  (h31 : a3 + a4 + a5 + a6 > a1 + a2 + a7)
  (h32 : a3 + a4 + a5 + a7 > a1 + a2 + a6)
  (h33 : a3 + a4 + a6 + a7 > a1 + a2 + a5)
  (h34 : a3 + a5 + a6 + a7 > a1 + a2 + a4)
  (h35 : a4 + a5 + a6 + a7 > a1 + a2 + a3)
: a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0 ∧ a5 > 0 ∧ a6 > 0 ∧ a7 > 0 := 
sorry

end all_positive_l152_152322


namespace John_pays_2400_per_year_l152_152717

theorem John_pays_2400_per_year
  (hours_per_month : ℕ)
  (average_length : ℕ)
  (cost_per_song : ℕ)
  (h1 : hours_per_month = 20)
  (h2 : average_length = 3)
  (h3 : cost_per_song = 50) :
  (hours_per_month * 60 / average_length * cost_per_song * 12 = 2400) :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end John_pays_2400_per_year_l152_152717


namespace parallel_vectors_sum_is_six_l152_152618

theorem parallel_vectors_sum_is_six (x y : ℝ) :
  let a := (4, -1, 1)
  let b := (x, y, 2)
  (x / 4 = 2) ∧ (y / -1 = 2) →
  x + y = 6 :=
by
  intros
  sorry

end parallel_vectors_sum_is_six_l152_152618


namespace tech_gadget_cost_inr_l152_152325

def conversion_ratio (a b : ℝ) : Prop := a = b

theorem tech_gadget_cost_inr :
  (forall a b c : ℝ, conversion_ratio (a / b) c) →
  (forall a b c d : ℝ, conversion_ratio (a / b) c → conversion_ratio (a / d) c) →
  ∀ (n_usd : ℝ) (n_inr : ℝ) (cost_n : ℝ), 
    n_usd = 8 →
    n_inr = 5 →
    cost_n = 160 →
    cost_n / n_usd * n_inr = 100 :=
by
  sorry

end tech_gadget_cost_inr_l152_152325


namespace no_a_satisfies_condition_l152_152733

noncomputable def M : Set ℝ := {0, 1}
noncomputable def N (a : ℝ) : Set ℝ := {11 - a, Real.log a / Real.log 1, 2^a, a}

theorem no_a_satisfies_condition :
  ¬ ∃ a : ℝ, M ∩ N a = {1} :=
by
  sorry

end no_a_satisfies_condition_l152_152733


namespace sum_of_powers_2017_l152_152616

theorem sum_of_powers_2017 (n : ℕ) (x : Fin n → ℤ) (h : ∀ i, x i = 0 ∨ x i = 1 ∨ x i = -1) (h_sum : (Finset.univ : Finset (Fin n)).sum x = 1000) :
  (Finset.univ : Finset (Fin n)).sum (λ i => (x i)^2017) = 1000 :=
by
  sorry

end sum_of_powers_2017_l152_152616


namespace max_value_l152_152764

noncomputable def max_expression (x : ℝ) : ℝ :=
  3^x - 2 * 9^x

theorem max_value : ∃ x : ℝ, max_expression x = 1 / 8 :=
sorry

end max_value_l152_152764


namespace convert_to_general_form_l152_152304

theorem convert_to_general_form (x : ℝ) :
  5 * x^2 - 2 * x = 3 * (x + 1) ↔ 5 * x^2 - 5 * x - 3 = 0 :=
by
  sorry

end convert_to_general_form_l152_152304


namespace max_area_parallelogram_l152_152836

theorem max_area_parallelogram
    (P : ℝ)
    (a b : ℝ)
    (h1 : P = 60)
    (h2 : a = 3 * b)
    (h3 : P = 2 * a + 2 * b) :
    (a * b ≤ 168.75) :=
by
  -- We prove that given the conditions, the maximum area is 168.75 square units.
  sorry

end max_area_parallelogram_l152_152836


namespace find_a_for_symmetric_and_parallel_lines_l152_152052

theorem find_a_for_symmetric_and_parallel_lines :
  ∃ (a : ℝ), (∀ (x y : ℝ), y = a * x + 3 ↔ x = a * y + 3) ∧ (∀ (x y : ℝ), x + 2 * y - 1 = 0 ↔ x = a * y + 3) ∧ ∃ (a : ℝ), a = -2 := 
sorry

end find_a_for_symmetric_and_parallel_lines_l152_152052


namespace factor_expression_l152_152232

theorem factor_expression (x : ℝ) :
  (12 * x ^ 5 + 33 * x ^ 3 + 10) - (3 * x ^ 5 - 4 * x ^ 3 - 1) = x ^ 3 * (9 * x ^ 2 + 37) + 11 :=
by {
  -- Provide the skeleton for the proof using simplification
  sorry
}

end factor_expression_l152_152232


namespace find_pq_l152_152212

noncomputable def find_k_squared (x y : ℝ) : ℝ :=
  let u1 := x^2 + y^2 - 12 * x + 16 * y - 160
  let u2 := x^2 + y^2 + 12 * x + 16 * y - 36
  let k_sq := 741 / 324
  k_sq

theorem find_pq : (741 + 324) = 1065 := by
  sorry

end find_pq_l152_152212


namespace area_of_stripe_l152_152811

def cylindrical_tank.diameter : ℝ := 40
def cylindrical_tank.height : ℝ := 100
def green_stripe.width : ℝ := 4
def green_stripe.revolutions : ℝ := 3

theorem area_of_stripe :
  let diameter := cylindrical_tank.diameter
  let height := cylindrical_tank.height
  let width := green_stripe.width
  let revolutions := green_stripe.revolutions
  let circumference := Real.pi * diameter
  let length := revolutions * circumference
  let area := length * width
  area = 480 * Real.pi := by
  sorry

end area_of_stripe_l152_152811


namespace total_population_after_births_l152_152647

theorem total_population_after_births:
  let initial_population := 300000
  let immigrants := 50000
  let emigrants := 30000
  let pregnancies_fraction := 1 / 8
  let twins_fraction := 1 / 4
  let net_population := initial_population + immigrants - emigrants
  let pregnancies := net_population * pregnancies_fraction
  let twin_pregnancies := pregnancies * twins_fraction
  let twin_children := twin_pregnancies * 2
  let single_births := pregnancies - twin_pregnancies
  net_population + single_births + twin_children = 370000 := by
  sorry

end total_population_after_births_l152_152647


namespace calculate_expression_l152_152937

theorem calculate_expression : -4^2 * (-1)^2022 = -16 :=
by
  sorry

end calculate_expression_l152_152937


namespace product_ge_one_l152_152821

variable (a b : ℝ)
variable (x1 x2 x3 x4 x5 : ℝ)

theorem product_ge_one
  (ha : 0 < a)
  (hb : 0 < b)
  (h_ab : a + b = 1)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (hx3 : 0 < x3)
  (hx4 : 0 < x4)
  (hx5 : 0 < x5)
  (h_prod_xs : x1 * x2 * x3 * x4 * x5 = 1) :
  (a * x1 + b) * (a * x2 + b) * (a * x3 + b) * (a * x4 + b) * (a * x5 + b) ≥ 1 :=
by
  sorry

end product_ge_one_l152_152821


namespace max_value_of_b_l152_152233

theorem max_value_of_b {m b : ℚ} (x : ℤ) 
  (line_eq : ∀ x : ℤ, 0 < x ∧ x ≤ 200 → 
    ¬ ∃ (y : ℤ), y = m * x + 3)
  (m_range : 1/3 < m ∧ m < b) :
  b = 69/208 :=
by
  sorry

end max_value_of_b_l152_152233


namespace purchasing_plans_count_l152_152289

theorem purchasing_plans_count :
  (∃ (x y : ℕ), 15 * x + 20 * y = 360) ∧ ∀ (x y : ℕ), 15 * x + 20 * y = 360 → (x % 4 = 0) ∧ (y = 18 - (3 / 4) * x) := sorry

end purchasing_plans_count_l152_152289


namespace roots_cubic_identity_l152_152958

theorem roots_cubic_identity (r s : ℚ) (h1 : 3 * r^2 + 5 * r + 2 = 0) (h2 : 3 * s^2 + 5 * s + 2 = 0) :
  (1 / r^3) + (1 / s^3) = -27 / 35 :=
sorry

end roots_cubic_identity_l152_152958


namespace joel_average_speed_l152_152010

theorem joel_average_speed :
  let start_time := (8, 50)
  let end_time := (14, 35)
  let total_distance := 234
  let total_time := (14 - 8) + (35 - 50) / 60
  ∀ start_time end_time total_distance,
    (start_time = (8, 50)) →
    (end_time = (14, 35)) →
    total_distance = 234 →
    (total_time = (14 - 8) + (35 - 50) / 60) →
    total_distance / total_time = 41 :=
by
  sorry

end joel_average_speed_l152_152010


namespace convert_base_10_to_base_5_l152_152534

theorem convert_base_10_to_base_5 :
  (256 : ℕ) = 2 * 5^3 + 0 * 5^2 + 1 * 5^1 + 1 * 5^0 :=
by
  sorry

end convert_base_10_to_base_5_l152_152534


namespace min_value_reciprocals_l152_152589

theorem min_value_reciprocals (a b : ℝ) 
  (h1 : 2 * a + 2 * b = 2) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  (1 / a + 1 / b) ≥ 4 :=
sorry

end min_value_reciprocals_l152_152589


namespace number_of_possible_winning_scores_l152_152168

noncomputable def sum_of_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem number_of_possible_winning_scores : 
  let total_sum := sum_of_first_n_integers 12
  let max_possible_score := total_sum / 2
  let min_possible_score := sum_of_first_n_integers 6
  39 - 21 + 1 = 19 := 
by
  sorry

end number_of_possible_winning_scores_l152_152168


namespace train_speed_in_kmph_l152_152407

theorem train_speed_in_kmph
  (train_length : ℝ) (bridge_length : ℝ) (time_seconds : ℝ)
  (H1: train_length = 200) (H2: bridge_length = 150) (H3: time_seconds = 34.997200223982084) :
  train_length + bridge_length = 200 + 150 →
  (train_length + bridge_length) / time_seconds * 3.6 = 36 :=
sorry

end train_speed_in_kmph_l152_152407


namespace system1_solution_system2_solution_l152_152387

-- Part 1: Substitution Method
theorem system1_solution (x y : ℤ) :
  2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ↔ x = 2 ∧ y = 1 :=
by
  sorry

-- Part 2: Elimination Method
theorem system2_solution (x y : ℚ) :
  2 * x + y = 2 ∧ 8 * x + 3 * y = 9 ↔ x = 3 / 2 ∧ y = -1 :=
by
  sorry

end system1_solution_system2_solution_l152_152387


namespace three_digit_numbers_without_579_l152_152849

def count_valid_digits (exclusions : List Nat) (range : List Nat) : Nat :=
  (range.filter (λ n => n ∉ exclusions)).length

def count_valid_three_digit_numbers : Nat :=
  let hundreds := count_valid_digits [5, 7, 9] [1, 2, 3, 4, 6, 8]
  let tens_units := count_valid_digits [5, 7, 9] [0, 1, 2, 3, 4, 6, 8]
  hundreds * tens_units * tens_units

theorem three_digit_numbers_without_579 : 
  count_valid_three_digit_numbers = 294 :=
by
  unfold count_valid_three_digit_numbers
  /- 
  Here you can add intermediate steps if necessary, 
  but for now we assert the final goal since this is 
  just the problem statement with the proof omitted.
  -/
  sorry

end three_digit_numbers_without_579_l152_152849


namespace tammy_avg_speed_l152_152483

theorem tammy_avg_speed 
  (v t : ℝ) 
  (h1 : t + (t - 2) = 14)
  (h2 : v * t + (v + 0.5) * (t - 2) = 52) : 
  v + 0.5 = 4 :=
by
  sorry

end tammy_avg_speed_l152_152483


namespace product_of_solutions_l152_152692

theorem product_of_solutions :
  (∃ x y : ℝ, (|x^2 - 6 * x| + 5 = 41) ∧ (|y^2 - 6 * y| + 5 = 41) ∧ x ≠ y ∧ x * y = -36) :=
by
  sorry

end product_of_solutions_l152_152692


namespace length_of_each_train_l152_152383

theorem length_of_each_train (L : ℝ) 
  (speed_faster : ℝ := 45 * 5 / 18) -- converting 45 km/hr to m/s
  (speed_slower : ℝ := 36 * 5 / 18) -- converting 36 km/hr to m/s
  (time : ℝ := 36) 
  (relative_speed : ℝ := speed_faster - speed_slower) 
  (total_distance : ℝ := relative_speed * time) 
  (length_each_train : ℝ := total_distance / 2) 
  : length_each_train = 45 := 
by 
  sorry

end length_of_each_train_l152_152383


namespace total_students_l152_152051

theorem total_students (m d : ℕ) 
  (H1: 30 < m + d ∧ m + d < 40)
  (H2: ∃ r, r = 3 * m ∧ r = 5 * d) : 
  m + d = 32 := 
by
  sorry

end total_students_l152_152051


namespace horner_evaluation_at_2_l152_152242

noncomputable def f : ℕ → ℕ :=
  fun x => (((2 * x + 3) * x + 0) * x + 5) * x - 4

theorem horner_evaluation_at_2 : f 2 = 14 :=
  by
    sorry

end horner_evaluation_at_2_l152_152242


namespace line_equation_l152_152193

theorem line_equation (a b : ℝ) 
  (h1 : -4 = (a + 0) / 2)
  (h2 : 6 = (0 + b) / 2) :
  (∀ x y : ℝ, y = (3 / 2) * (x + 4) → 3 * x - 2 * y + 24 = 0) :=
by
  sorry

end line_equation_l152_152193


namespace paving_stone_proof_l152_152343

noncomputable def paving_stone_width (length_court : ℝ) (width_court : ℝ) 
                                      (num_stones: ℕ) (stone_length: ℝ) : ℝ :=
  let area_court := length_court * width_court
  let area_stone := stone_length * (area_court / (num_stones * stone_length))
  area_court / area_stone

theorem paving_stone_proof :
  paving_stone_width 50 16.5 165 2.5 = 2 :=
sorry

end paving_stone_proof_l152_152343


namespace rahul_matches_played_l152_152681

theorem rahul_matches_played
  (current_avg : ℕ)
  (runs_today : ℕ)
  (new_avg : ℕ)
  (m: ℕ)
  (h1 : current_avg = 51)
  (h2 : runs_today = 78)
  (h3 : new_avg = 54)
  (h4 : (51 * m + runs_today) / (m + 1) = new_avg) :
  m = 8 :=
by
  sorry

end rahul_matches_played_l152_152681


namespace non_powers_of_a_meet_condition_l152_152754

-- Definitions used directly from the conditions detailed in the problem:
def Sa (a x : ℕ) : ℕ := sorry -- S_{a}(x): sum of the digits of x in base a
def Fa (a x : ℕ) : ℕ := sorry -- F_{a}(x): number of digits of x in base a
def fa (a x : ℕ) : ℕ := sorry -- f_{a}(x): position of the first non-zero digit from the right in base a

theorem non_powers_of_a_meet_condition (a M : ℕ) (h₁: a > 1) (h₂ : M ≥ 2020) :
  ∀ n : ℕ, (n > 0) → (∀ k : ℕ, (k > 0) → (Sa a (k * n) = Sa a n ∧ Fa a (k * n) - fa a (k * n) > M)) ↔ (∃ α : ℕ, n = a ^ α) :=
sorry

end non_powers_of_a_meet_condition_l152_152754


namespace solve_for_x_l152_152077

def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem solve_for_x : {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by
  sorry

end solve_for_x_l152_152077


namespace find_two_digit_number_l152_152452

theorem find_two_digit_number (N : ℕ) (h1 : 10 ≤ N ∧ N < 100) 
                              (h2 : N % 2 = 0) (h3 : N % 11 = 0) 
                              (h4 : ∃ k : ℕ, (N / 10) * (N % 10) = k^3) :
  N = 88 :=
by {
  sorry
}

end find_two_digit_number_l152_152452


namespace bricks_lay_calculation_l152_152068

theorem bricks_lay_calculation (b c d : ℕ) (h1 : 0 < c) (h2 : 0 < d) : 
  ∃ y : ℕ, y = (b * (b + d) * (c + d))/(c * d) :=
sorry

end bricks_lay_calculation_l152_152068


namespace Clea_ride_time_l152_152566

theorem Clea_ride_time
  (c s d t : ℝ)
  (h1 : d = 80 * c)
  (h2 : d = 30 * (c + s))
  (h3 : s = 5 / 3 * c)
  (h4 : t = d / s) :
  t = 48 := by sorry

end Clea_ride_time_l152_152566


namespace geometric_sequence_product_l152_152249

theorem geometric_sequence_product :
  ∃ a : ℕ → ℝ, 
    a 1 = 1 ∧ 
    a 5 = 16 ∧ 
    (∀ n, a (n + 1) = a n * r) ∧
    ∃ r : ℝ, 
      a 2 * a 3 * a 4 = 64 :=
by
  sorry

end geometric_sequence_product_l152_152249


namespace evaluate_expression_l152_152056

theorem evaluate_expression (a : ℚ) (h : a = 4 / 3) : (6 * a^2 - 8 * a + 3) * (3 * a - 4) = 0 :=
by
  rw [h]
  sorry

end evaluate_expression_l152_152056


namespace only_polynomial_is_identity_l152_152489

-- Define the number composed only of digits 1
def Ones (k : ℕ) : ℕ := (10^k - 1) / 9

theorem only_polynomial_is_identity (P : ℕ → ℕ) :
  (∀ k : ℕ, P (Ones k) = Ones k) → (∀ x : ℕ, P x = x) :=
by
  intro h
  sorry

end only_polynomial_is_identity_l152_152489


namespace ratio_of_population_l152_152550

theorem ratio_of_population (Z : ℕ) :
  let Y := 2 * Z
  let X := 3 * Y
  let W := X + Y
  X / (Z + W) = 2 / 3 :=
by
  sorry

end ratio_of_population_l152_152550


namespace angle_2016_in_third_quadrant_l152_152542

def quadrant (θ : ℤ) : ℤ :=
  let angle := θ % 360
  if 0 ≤ angle ∧ angle < 90 then 1
  else if 90 ≤ angle ∧ angle < 180 then 2
  else if 180 ≤ angle ∧ angle < 270 then 3
  else 4

theorem angle_2016_in_third_quadrant : 
  quadrant 2016 = 3 := 
by
  sorry

end angle_2016_in_third_quadrant_l152_152542


namespace range_of_a_l152_152411

theorem range_of_a (a b c : ℝ) (h₁ : a + b + c = 2) (h₂ : a^2 + b^2 + c^2 = 4) (h₃ : a > b ∧ b > c) :
  (2 / 3 < a ∧ a < 2) :=
sorry

end range_of_a_l152_152411


namespace rectangle_area_l152_152759

theorem rectangle_area (a b c: ℝ) (h₁ : a = 7.1) (h₂ : b = 8.9) (h₃ : c = 10.0) (L W: ℝ)
  (h₄ : L = 2 * W) (h₅ : 2 * (L + W) = a + b + c) : L * W = 37.54 :=
by
  sorry

end rectangle_area_l152_152759


namespace percentage_of_two_is_point_eight_l152_152104

theorem percentage_of_two_is_point_eight (p : ℝ) : (p / 100) * 2 = 0.8 ↔ p = 40 := 
by
  sorry

end percentage_of_two_is_point_eight_l152_152104


namespace height_of_cone_l152_152728

theorem height_of_cone (e : ℝ) (bA : ℝ) (v : ℝ) :
  e = 6 ∧ bA = 54 ∧ v = e^3 → ∃ h : ℝ, (1/3) * bA * h = v ∧ h = 12 := by
  sorry

end height_of_cone_l152_152728


namespace new_persons_joined_l152_152569

theorem new_persons_joined (initial_avg_age new_avg_age initial_total new_avg_age_total final_avg_age final_total : ℝ) 
  (n_initial n_new : ℕ) 
  (h1 : initial_avg_age = 16)
  (h2 : n_initial = 20)
  (h3 : new_avg_age = 15)
  (h4 : final_avg_age = 15.5)
  (h5 : initial_total = initial_avg_age * n_initial)
  (h6 : new_avg_age_total = new_avg_age * (n_new : ℝ))
  (h7 : final_total = initial_total + new_avg_age_total)
  (h8 : final_total = final_avg_age * (n_initial + n_new)) 
  : n_new = 20 :=
by
  sorry

end new_persons_joined_l152_152569


namespace probability_manu_wins_l152_152338

theorem probability_manu_wins :
  ∑' (n : ℕ), (1 / 2)^(4 * (n + 1)) = 1 / 15 :=
by
  sorry

end probability_manu_wins_l152_152338


namespace equivalent_terminal_angle_l152_152972

theorem equivalent_terminal_angle :
  ∃ n : ℤ, 660 = n * 360 - 420 := 
by
  sorry

end equivalent_terminal_angle_l152_152972


namespace equalize_costs_l152_152257

theorem equalize_costs (X Y Z : ℝ) (hXY : X < Y) (hYZ : Y < Z) : (Y + Z - 2 * X) / 3 = (X + Y + Z) / 3 - X := by
  sorry

end equalize_costs_l152_152257


namespace lighter_shopping_bag_weight_l152_152501

theorem lighter_shopping_bag_weight :
  ∀ (G : ℕ), (G + 7 = 10) → (G = 3) := by
  intros G h
  sorry

end lighter_shopping_bag_weight_l152_152501


namespace cost_difference_of_dolls_proof_l152_152823

-- Define constants
def cost_large_doll : ℝ := 7
def total_spent : ℝ := 350
def additional_dolls : ℝ := 20

-- Define the function for the cost of small dolls
def cost_small_doll (S : ℝ) : Prop :=
  total_spent / S = total_spent / cost_large_doll + additional_dolls

-- The statement given the conditions and solving for the difference in cost
theorem cost_difference_of_dolls_proof : 
  ∃ S, cost_small_doll S ∧ (cost_large_doll - S = 2) :=
by
  sorry

end cost_difference_of_dolls_proof_l152_152823


namespace second_trial_addition_amount_l152_152035

variable (optimal_min optimal_max: ℝ) (phi: ℝ)

def method_618 (optimal_min optimal_max phi: ℝ) :=
  let x1 := optimal_min + (optimal_max - optimal_min) * phi
  let x2 := optimal_max + optimal_min - x1
  x2

theorem second_trial_addition_amount:
  optimal_min = 10 ∧ optimal_max = 110 ∧ phi = 0.618 →
  method_618 10 110 0.618 = 48.2 :=
by
  intro h
  simp [method_618, h]
  sorry

end second_trial_addition_amount_l152_152035


namespace problem_l152_152828

variable (g : ℝ → ℝ)
variables (x y : ℝ)

noncomputable def cond1 : Prop := ∀ x y : ℝ, 0 < x → 0 < y → g (x^2 * y) = g x / y^2
noncomputable def cond2 : Prop := g 800 = 4

-- The statement to be proved
theorem problem (h1 : cond1 g) (h2 : cond2 g) : g 7200 = 4 / 81 :=
by
  sorry

end problem_l152_152828


namespace ratio_sandra_amy_ruth_l152_152344

/-- Given the amounts received by Sandra and Amy, and an unknown amount received by Ruth,
    the ratio of the money shared between Sandra, Amy, and Ruth is 2:1:R/50. -/
theorem ratio_sandra_amy_ruth (R : ℝ) (hAmy : 50 > 0) (hSandra : 100 > 0) :
  (100 : ℝ) / 50 = 2 ∧ (50 : ℝ) / 50 = 1 ∧ ∃ (R : ℝ), (100/50 : ℝ) = 2 ∧ (50/50 : ℝ) = 1 ∧ (R / 50 : ℝ) = (R / 50 : ℝ) :=
by
  sorry

end ratio_sandra_amy_ruth_l152_152344


namespace tips_fraction_to_salary_l152_152865

theorem tips_fraction_to_salary (S T I : ℝ)
  (h1 : I = S + T)
  (h2 : T / I = 0.6923076923076923) :
  T / S = 2.25 := by
  sorry

end tips_fraction_to_salary_l152_152865


namespace simplify_expression_l152_152192

theorem simplify_expression : (Real.sqrt (9 / 4) - Real.sqrt (4 / 9)) = 5 / 6 :=
by
  sorry

end simplify_expression_l152_152192


namespace lemon_juice_calculation_l152_152460

noncomputable def lemon_juice_per_lemon (table_per_dozen : ℕ) (dozens : ℕ) (lemons : ℕ) : ℕ :=
  (table_per_dozen * dozens) / lemons

theorem lemon_juice_calculation :
  lemon_juice_per_lemon 12 3 9 = 4 :=
by
  -- proof would be here
  sorry

end lemon_juice_calculation_l152_152460


namespace relationship_S_T_l152_152911

def S (n : ℕ) : ℤ := 2^n
def T (n : ℕ) : ℤ := 2^n - (-1)^n

theorem relationship_S_T (n : ℕ) (h : n > 0) : 
  (n % 2 = 1 → S n < T n) ∧ (n % 2 = 0 → S n > T n) :=
by
  sorry

end relationship_S_T_l152_152911


namespace three_digit_division_l152_152685

theorem three_digit_division (abc : ℕ) (a b c : ℕ) (h1 : 100 ≤ abc ∧ abc < 1000) (h2 : abc = 100 * a + 10 * b + c) (h3 : a ≠ 0) :
  (1001 * abc) / 7 / 11 / 13 = abc :=
by
  sorry

end three_digit_division_l152_152685


namespace mary_final_books_l152_152549

def mary_initial_books := 5
def mary_first_return := 3
def mary_first_checkout := 5
def mary_second_return := 2
def mary_second_checkout := 7

theorem mary_final_books :
  (mary_initial_books - mary_first_return + mary_first_checkout - mary_second_return + mary_second_checkout) = 12 := 
by 
  sorry

end mary_final_books_l152_152549


namespace max_perimeter_of_polygons_l152_152482

theorem max_perimeter_of_polygons 
  (t s : ℕ) 
  (hts : t + s = 7) 
  (hsum_angles : 60 * t + 90 * s = 360) 
  (max_squares : s ≤ 4) 
  (side_length : ℕ := 2) 
  (tri_perimeter : ℕ := 3 * side_length) 
  (square_perimeter : ℕ := 4 * side_length) :
  2 * (t * tri_perimeter + s * square_perimeter) = 68 := 
sorry

end max_perimeter_of_polygons_l152_152482


namespace simplified_expression_at_one_l152_152271

noncomputable def original_expression (a : ℚ) : ℚ :=
  (2 * a + 2) / a / (4 / (a ^ 2)) - a / (a + 1)

theorem simplified_expression_at_one : original_expression 1 = 1 / 2 := by
  sorry

end simplified_expression_at_one_l152_152271


namespace oil_bill_for_January_l152_152601

variable {F J : ℕ}

theorem oil_bill_for_January (h1 : 2 * F = 3 * J) (h2 : 3 * (F + 20) = 5 * J) : J = 120 := by
  sorry

end oil_bill_for_January_l152_152601


namespace elastic_band_radius_increase_l152_152029

theorem elastic_band_radius_increase 
  (C1 C2 : ℝ) 
  (hC1 : C1 = 40) 
  (hC2 : C2 = 80) 
  (hC1_def : C1 = 2 * π * r1) 
  (hC2_def : C2 = 2 * π * r2) :
  r2 - r1 = 20 / π :=
by
  sorry

end elastic_band_radius_increase_l152_152029


namespace common_tangents_l152_152445

noncomputable def radius1 := 8
noncomputable def radius2 := 6
noncomputable def distance := 2

theorem common_tangents (r1 r2 d : ℕ) 
  (h1 : r1 = radius1) 
  (h2 : r2 = radius2) 
  (h3 : d = distance) :
  (d = r1 - r2) → 1 = 1 := by 
  sorry

end common_tangents_l152_152445


namespace Donny_change_l152_152371

theorem Donny_change (tank_capacity : ℕ) (initial_fuel : ℕ) (money_available : ℕ) (fuel_cost_per_liter : ℕ) 
  (h1 : tank_capacity = 150) 
  (h2 : initial_fuel = 38) 
  (h3 : money_available = 350) 
  (h4 : fuel_cost_per_liter = 3) : 
  money_available - (tank_capacity - initial_fuel) * fuel_cost_per_liter = 14 := 
by 
  sorry

end Donny_change_l152_152371


namespace abs_difference_equality_l152_152072

theorem abs_difference_equality : (abs (3 - Real.sqrt 2) - abs (Real.sqrt 2 - 2) = 1) :=
  by
    -- Define our conditions as hypotheses
    have h1 : 3 > Real.sqrt 2 := sorry
    have h2 : Real.sqrt 2 < 2 := sorry
    -- The proof itself is skipped in this step
    sorry

end abs_difference_equality_l152_152072


namespace even_times_odd_is_even_l152_152835

theorem even_times_odd_is_even {a b : ℤ} (h₁ : ∃ k, a = 2 * k) (h₂ : ∃ j, b = 2 * j + 1) : ∃ m, a * b = 2 * m :=
by
  sorry

end even_times_odd_is_even_l152_152835


namespace units_digit_of_sum_is_three_l152_152039

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_of_factorials : ℕ :=
  (List.range 10).map factorial |>.sum

def power_of_ten (n : ℕ) : ℕ :=
  10^n

theorem units_digit_of_sum_is_three : 
  units_digit (sum_of_factorials + power_of_ten 3) = 3 := by
  sorry

end units_digit_of_sum_is_three_l152_152039


namespace concrete_pillars_correct_l152_152599

-- Definitions based on conditions
def concrete_for_roadway := 1600
def concrete_for_one_anchor := 700
def total_concrete_for_bridge := 4800

-- Total concrete for both anchors
def concrete_for_both_anchors := 2 * concrete_for_one_anchor

-- Total concrete needed for the roadway and anchors
def concrete_for_roadway_and_anchors := concrete_for_roadway + concrete_for_both_anchors

-- Concrete needed for the supporting pillars
def concrete_for_pillars := total_concrete_for_bridge - concrete_for_roadway_and_anchors

-- Proof problem statement, verify that the concrete for the supporting pillars is 1800 tons
theorem concrete_pillars_correct : concrete_for_pillars = 1800 := by
  sorry

end concrete_pillars_correct_l152_152599


namespace domain_of_function_l152_152085

noncomputable def domain_is_valid (x z : ℝ) : Prop :=
  1 < x ∧ x < 2 ∧ (|x| - z) ≠ 0

theorem domain_of_function (x z : ℝ) : domain_is_valid x z :=
by
  sorry

end domain_of_function_l152_152085


namespace find_a_l152_152579

theorem find_a
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 2 * (b * Real.cos A + a * Real.cos B) = c^2)
  (h2 : b = 3)
  (h3 : 3 * Real.cos A = 1) :
  a = 3 :=
sorry

end find_a_l152_152579


namespace eccentricities_ellipse_hyperbola_l152_152467

theorem eccentricities_ellipse_hyperbola :
  let a := 2
  let b := -5
  let c := 2
  let delta := b^2 - 4 * a * c
  let x1 := (-b + Real.sqrt delta) / (2 * a)
  let x2 := (-b - Real.sqrt delta) / (2 * a)
  (x1 > 1) ∧ (0 < x2) ∧ (x2 < 1) :=
sorry

end eccentricities_ellipse_hyperbola_l152_152467


namespace karlson_expenditure_exceeds_2000_l152_152069

theorem karlson_expenditure_exceeds_2000 :
  ∃ n m : ℕ, 25 * n + 340 * m > 2000 :=
by {
  -- proof must go here
  sorry
}

end karlson_expenditure_exceeds_2000_l152_152069


namespace hypotenuse_length_l152_152605

theorem hypotenuse_length {a b c : ℕ} (ha : a = 8) (hb : b = 15) (hc : c = (8^2 + 15^2).sqrt) : c = 17 :=
by
  sorry

end hypotenuse_length_l152_152605


namespace tan_of_cos_alpha_l152_152043

open Real

theorem tan_of_cos_alpha (α : ℝ) (h1 : cos α = 3 / 5) (h2 : -π < α ∧ α < 0) : tan α = -4 / 3 :=
sorry

end tan_of_cos_alpha_l152_152043


namespace proof_problem_l152_152521

-- definitions of the given conditions
variable (a b c : ℝ)
variables (h₁ : 6 < a) (h₂ : a < 10) 
variable (h₃ : (a / 2) ≤ b ∧ b ≤ 2 * a) 
variable (h₄ : c = a + b)

-- statement to be proved
theorem proof_problem (h₁ : 6 < a) (h₂ : a < 10) (h₃ : (a / 2) ≤ b ∧ b ≤ 2 * a) (h₄ : c = a + b) : 9 < c ∧ c < 30 := 
sorry

end proof_problem_l152_152521


namespace parallel_lines_m_eq_minus_seven_l152_152884

theorem parallel_lines_m_eq_minus_seven
  (m : ℝ)
  (l₁ : ∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m)
  (l₂ : ∀ x y : ℝ, 2 * x + (5 + m) * y = 8)
  (parallel : ∀ x y : ℝ, (3 + m) * 4 = 2 * (5 + m)) :
  m = -7 :=
sorry

end parallel_lines_m_eq_minus_seven_l152_152884


namespace possible_value_of_2n_plus_m_l152_152945

variable (n m : ℤ)

theorem possible_value_of_2n_plus_m : (3 * n - m < 5) → (n + m > 26) → (3 * m - 2 * n < 46) → (2 * n + m = 36) :=
by
  sorry

end possible_value_of_2n_plus_m_l152_152945


namespace problem_exists_integers_a_b_c_d_l152_152829

theorem problem_exists_integers_a_b_c_d :
  ∃ (a b c d : ℤ), 
  |a| > 1000000 ∧ |b| > 1000000 ∧ |c| > 1000000 ∧ |d| > 1000000 ∧
  (1 / (a:ℚ) + 1 / (b:ℚ) + 1 / (c:ℚ) + 1 / (d:ℚ) = 1 / (a * b * c * d : ℚ)) :=
sorry

end problem_exists_integers_a_b_c_d_l152_152829


namespace option_C_represents_same_function_l152_152702

-- Definitions of the functions from option C
def f (x : ℝ) := x^2 - 1
def g (t : ℝ) := t^2 - 1

-- The proof statement that needs to be proven
theorem option_C_represents_same_function :
  f = g :=
sorry

end option_C_represents_same_function_l152_152702


namespace least_pos_int_with_ten_factors_l152_152286

theorem least_pos_int_with_ten_factors : ∃ (n : ℕ), n > 0 ∧ (∀ m, (m > 0 ∧ ∃ d : ℕ, d∣n → d = 1 ∨ d = n) → m < n) ∧ ( ∃! n, ∃ d : ℕ, d∣n ) := sorry

end least_pos_int_with_ten_factors_l152_152286


namespace min_value_fraction_l152_152938

theorem min_value_fraction (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∃ x₀, (2 * x₀ - 2) * (-2 * x₀ + a) = -1) : 
  ∃ a b, a + b = 5 / 2 → a > 0 → b > 0 → 
  (∀ a b, (1 / a + 4 / b) ≥ 18 / 5) :=
by
  sorry

end min_value_fraction_l152_152938


namespace remainder_sum_of_squares_mod_13_l152_152115

-- Define the sum of squares of the first n natural numbers
def sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

-- Prove that the remainder when the sum of squares of the first 20 natural numbers
-- is divided by 13 is 10
theorem remainder_sum_of_squares_mod_13 : sum_of_squares 20 % 13 = 10 := 
by
  -- Here you can imagine the relevant steps or intermediate computations might go, if needed.
  sorry -- Placeholder for the proof.

end remainder_sum_of_squares_mod_13_l152_152115


namespace men_work_days_l152_152473

theorem men_work_days (M : ℕ) (W : ℕ) (h : W / (M * 40) = W / ((M - 5) * 50)) : M = 25 :=
by
  -- Will add the proof later
  sorry

end men_work_days_l152_152473


namespace cone_volume_proof_l152_152690

noncomputable def cone_volume (l h : ℕ) : ℝ :=
  let r := Real.sqrt (l^2 - h^2)
  1 / 3 * Real.pi * r^2 * h

theorem cone_volume_proof :
  cone_volume 13 12 = 100 * Real.pi :=
by
  sorry

end cone_volume_proof_l152_152690


namespace proof_no_solution_l152_152743

noncomputable def no_solution (a b : ℕ) : Prop :=
  2 * a^2 + 1 ≠ 4 * b^2

theorem proof_no_solution (a b : ℕ) : no_solution a b := by
  sorry

end proof_no_solution_l152_152743


namespace DanAgeIs12_l152_152021

def DanPresentAge (x : ℕ) : Prop :=
  (x + 18 = 5 * (x - 6))

theorem DanAgeIs12 : ∃ x : ℕ, DanPresentAge x ∧ x = 12 :=
by
  use 12
  unfold DanPresentAge
  sorry

end DanAgeIs12_l152_152021


namespace polygon_sides_l152_152480

theorem polygon_sides (x : ℕ) (h1 : 180 * (x - 2) = 3 * 360) : x = 8 := by
  sorry

end polygon_sides_l152_152480


namespace tiling_problem_l152_152960

theorem tiling_problem (b c f : ℕ) (h : b * c = f) : c * (b^2 / f) = b :=
by 
  sorry

end tiling_problem_l152_152960


namespace axis_of_symmetry_eq_l152_152748

theorem axis_of_symmetry_eq : 
  ∃ k : ℤ, (λ x => 2 * Real.cos (2 * x)) = (λ x => 2 * Real.sin (2 * (x + π / 3) - π / 6)) ∧
            x = (1/2) * k * π ∧ x = -π / 2 := 
by
  sorry

end axis_of_symmetry_eq_l152_152748


namespace trig_identity_t_half_l152_152335

theorem trig_identity_t_half (a t : ℝ) (ht : t = Real.tan (a / 2)) :
  Real.sin a = (2 * t) / (1 + t^2) ∧
  Real.cos a = (1 - t^2) / (1 + t^2) ∧
  Real.tan a = (2 * t) / (1 - t^2) := 
sorry

end trig_identity_t_half_l152_152335


namespace average_of_all_5_numbers_is_20_l152_152716

def average_of_all_5_numbers
  (sum_3_numbers : ℕ)
  (avg_2_numbers : ℕ) : ℕ :=
(sum_3_numbers + 2 * avg_2_numbers) / 5

theorem average_of_all_5_numbers_is_20 :
  average_of_all_5_numbers 48 26 = 20 :=
by
  unfold average_of_all_5_numbers -- Expand the definition
  -- Sum of 5 numbers is 48 (sum of 3) + (2 * 26) (sum of other 2)
  -- Total sum is 48 + 52 = 100
  -- Average is 100 / 5 = 20
  norm_num -- Check the numeric calculation
  -- sorry

end average_of_all_5_numbers_is_20_l152_152716


namespace volleyball_team_ways_l152_152164

def num_ways_choose_starers : ℕ :=
  3 * (Nat.choose 12 6 + Nat.choose 12 5)

theorem volleyball_team_ways :
  num_ways_choose_starers = 5148 := by
  sorry

end volleyball_team_ways_l152_152164


namespace herring_invariant_l152_152047

/--
A circle is divided into six sectors. Each sector contains one herring. 
In one move, you can move any two herrings in adjacent sectors moving them in opposite directions.
Prove that it is impossible to gather all herrings into one sector using these operations.
-/
theorem herring_invariant (herring : Fin 6 → Bool) :
  ¬ ∃ i : Fin 6, ∀ j : Fin 6, herring j = herring i := 
sorry

end herring_invariant_l152_152047


namespace triangle_inequality_at_vertex_l152_152889

-- Define the edge lengths of the tetrahedron and the common vertex label
variables {a b c d e f S : ℝ}

-- Conditions for the edge lengths and vertex label
axiom edge_lengths :
  a + b + c = S ∧
  a + d + e = S ∧
  b + d + f = S ∧
  c + e + f = S

-- The theorem to be proven
theorem triangle_inequality_at_vertex :
  a + b + c = S →
  a + d + e = S →
  b + d + f = S →
  c + e + f = S →
  (a ≤ b + c) ∧
  (b ≤ c + a) ∧
  (c ≤ a + b) ∧
  (a ≤ d + e) ∧
  (d ≤ e + a) ∧
  (e ≤ a + d) ∧
  (b ≤ d + f) ∧
  (d ≤ f + b) ∧
  (f ≤ b + d) ∧
  (c ≤ e + f) ∧
  (e ≤ f + c) ∧
  (f ≤ c + e) :=
sorry

end triangle_inequality_at_vertex_l152_152889


namespace arithmetic_seq_8th_term_l152_152696

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end arithmetic_seq_8th_term_l152_152696


namespace upper_limit_of_range_l152_152678

theorem upper_limit_of_range (n : ℕ) (h : (10 + 10 * n) / 2 = 255) : 10 * n = 500 :=
by 
  sorry

end upper_limit_of_range_l152_152678


namespace part1_part2_l152_152543

namespace Problem

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem part1 (m : ℝ) : (B m ⊆ A) → (m ≤ 3) :=
by
  intro h
  sorry

theorem part2 (m : ℝ) : (A ∩ B m = ∅) → (m < 2 ∨ 4 < m) :=
by
  intro h
  sorry

end Problem

end part1_part2_l152_152543


namespace sum_of_consecutive_odds_mod_16_l152_152738

theorem sum_of_consecutive_odds_mod_16 :
  (12001 + 12003 + 12005 + 12007 + 12009 + 12011 + 12013) % 16 = 1 :=
by
  sorry

end sum_of_consecutive_odds_mod_16_l152_152738


namespace find_y_when_x_eq_4_l152_152275

theorem find_y_when_x_eq_4 (x y : ℝ) (k : ℝ) :
  (8 * y = k / x^3) →
  (y = 25) →
  (x = 2) →
  (exists y', x = 4 → y' = 25/8) :=
by
  sorry

end find_y_when_x_eq_4_l152_152275


namespace arithmetic_mean_after_removal_l152_152427

theorem arithmetic_mean_after_removal 
  (mean_original : ℝ) (num_original : ℕ) 
  (nums_removed : List ℝ) (mean_new : ℝ)
  (h1 : mean_original = 50) 
  (h2 : num_original = 60) 
  (h3 : nums_removed = [60, 65, 70, 40]) 
  (h4 : mean_new = 49.38) :
  let sum_original := mean_original * num_original
  let num_remaining := num_original - nums_removed.length
  let sum_removed := List.sum nums_removed
  let sum_new := sum_original - sum_removed
  
  mean_new = sum_new / num_remaining :=
sorry

end arithmetic_mean_after_removal_l152_152427


namespace stickers_initial_count_l152_152553

theorem stickers_initial_count (S : ℕ) 
  (h1 : (3 / 5 : ℝ) * (2 / 3 : ℝ) * S = 54) : S = 135 := 
by
  sorry

end stickers_initial_count_l152_152553


namespace total_area_of_strips_l152_152777

def strip1_length := 12
def strip1_width := 1
def strip2_length := 8
def strip2_width := 2
def num_strips1 := 2
def num_strips2 := 2
def overlap_area_per_strip := 2
def num_overlaps := 4
def total_area_covered := 48

theorem total_area_of_strips : 
  num_strips1 * (strip1_length * strip1_width) + 
  num_strips2 * (strip2_length * strip2_width) - 
  num_overlaps * overlap_area_per_strip = total_area_covered := sorry

end total_area_of_strips_l152_152777


namespace multiply_fractions_l152_152559

theorem multiply_fractions :
  (2 / 3) * (5 / 7) * (8 / 9) = 80 / 189 :=
by sorry

end multiply_fractions_l152_152559


namespace arithmetic_sequence_third_term_l152_152012

theorem arithmetic_sequence_third_term
  (a d : ℤ)
  (h_fifteenth_term : a + 14 * d = 15)
  (h_sixteenth_term : a + 15 * d = 21) :
  a + 2 * d = -57 :=
by
  sorry

end arithmetic_sequence_third_term_l152_152012


namespace ice_cream_amount_l152_152028

/-- Given: 
    Amount of ice cream eaten on Friday night: 3.25 pints
    Total amount of ice cream eaten over both nights: 3.5 pints
    Prove: 
    Amount of ice cream eaten on Saturday night = 0.25 pints -/
theorem ice_cream_amount (friday_night saturday_night total : ℝ) (h_friday : friday_night = 3.25) (h_total : total = 3.5) : 
  saturday_night = total - friday_night → saturday_night = 0.25 :=
by
  intro h
  rw [h_total, h_friday] at h
  simp [h]
  sorry

end ice_cream_amount_l152_152028


namespace payment_amount_l152_152057

/-- 
A certain debt will be paid in 52 installments from January 1 to December 31 of a certain year.
Each of the first 25 payments is to be a certain amount; each of the remaining payments is to be $100 more than each of the first payments.
The average (arithmetic mean) payment that will be made on the debt for the year is $551.9230769230769.
Prove that the amount of each of the first 25 payments is $500.
-/
theorem payment_amount (X : ℝ) 
  (h1 : 25 * X + 27 * (X + 100) = 52 * 551.9230769230769) :
  X = 500 :=
sorry

end payment_amount_l152_152057


namespace inequality_cubed_l152_152899

theorem inequality_cubed (a b : ℝ) (h : a < b ∧ b < 0) : a^3 ≤ b^3 :=
sorry

end inequality_cubed_l152_152899


namespace determinant_matrix_3x3_l152_152036

theorem determinant_matrix_3x3 :
  Matrix.det ![![3, 1, -2], ![8, 5, -4], ![1, 3, 6]] = 140 :=
by
  sorry

end determinant_matrix_3x3_l152_152036


namespace evaluate_expression_l152_152396

theorem evaluate_expression :
  (2 * 4 * 6) * (1 / 2 + 1 / 4 + 1 / 6) = 44 :=
by
  sorry

end evaluate_expression_l152_152396


namespace sample_capacity_l152_152648

theorem sample_capacity (freq : ℕ) (freq_rate : ℚ) (H_freq : freq = 36) (H_freq_rate : freq_rate = 0.25) : 
  ∃ n : ℕ, n = 144 :=
by
  sorry

end sample_capacity_l152_152648


namespace number_of_cows_l152_152934

theorem number_of_cows (D C : ℕ) (h1 : 2 * D + 4 * C = 40 + 2 * (D + C)) : C = 20 :=
by
  sorry

end number_of_cows_l152_152934


namespace find_m_l152_152975

theorem find_m (m : ℝ) (x : ℝ) (h : 2*x + m = 1) (hx : x = -1) : m = 3 := 
by
  rw [hx] at h
  linarith

end find_m_l152_152975


namespace total_people_on_hike_l152_152522

def cars : Nat := 3
def people_per_car : Nat := 4
def taxis : Nat := 6
def people_per_taxi : Nat := 6
def vans : Nat := 2
def people_per_van : Nat := 5

theorem total_people_on_hike :
  cars * people_per_car + taxis * people_per_taxi + vans * people_per_van = 58 := by
  sorry

end total_people_on_hike_l152_152522


namespace max_value_t_min_value_y_l152_152336

open Real

-- Maximum value of t for ∀ x ∈ ℝ, |3x + 2| + |3x - 1| ≥ t
theorem max_value_t :
  ∃ t, (∀ x : ℝ, |3 * x + 2| + |3 * x - 1| ≥ t) ∧ t = 3 :=
by
  sorry

-- Minimum value of y for 4m + 5n = 3
theorem min_value_y (m n: ℝ) (hm : m > 0) (hn: n > 0) (h: 4 * m + 5 * n = 3) :
  ∃ y, (y = (1 / (m + 2 * n)) + (4 / (3 * m + 3 * n))) ∧ y = 3 :=
by
  sorry

end max_value_t_min_value_y_l152_152336


namespace hangar_length_l152_152375

-- Define the conditions
def num_planes := 7
def length_per_plane := 40 -- in feet

-- Define the main theorem to be proven
theorem hangar_length : num_planes * length_per_plane = 280 := by
  -- Proof omitted with sorry
  sorry

end hangar_length_l152_152375


namespace cosine_expression_value_l152_152440

noncomputable def c : ℝ := 2 * Real.pi / 7

theorem cosine_expression_value :
  (Real.cos (3 * c) * Real.cos (5 * c) * Real.cos (6 * c)) / 
  (Real.cos c * Real.cos (2 * c) * Real.cos (3 * c)) = 1 :=
by
  sorry

end cosine_expression_value_l152_152440


namespace min_value_of_quadratic_l152_152880

theorem min_value_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ x : ℝ, (x = -p / 2) ∧ ∀ y : ℝ, (y^2 + p * y + q) ≥ ((-p/2)^2 + p * (-p/2) + q) :=
sorry

end min_value_of_quadratic_l152_152880


namespace xiaoMing_xiaoHong_diff_university_l152_152943

-- Definitions based on problem conditions
inductive Student
| XiaoMing
| XiaoHong
| StudentC
| StudentD
deriving DecidableEq

inductive University
| A
| B
deriving DecidableEq

-- Definition for the problem
def num_ways_diff_university : Nat :=
  4 -- The correct answer based on the solution steps

-- Problem statement
theorem xiaoMing_xiaoHong_diff_university :
  let students := [Student.XiaoMing, Student.XiaoHong, Student.StudentC, Student.StudentD]
  let universities := [University.A, University.B]
  (∃ (assign : Student → University),
    assign Student.XiaoMing ≠ assign Student.XiaoHong ∧
    (assign Student.StudentC ≠ assign Student.StudentD ∨
     assign Student.XiaoMing ≠ assign Student.StudentD ∨
     assign Student.XiaoHong ≠ assign Student.StudentC ∨
     assign Student.XiaoMing ≠ assign Student.StudentC)) →
  num_ways_diff_university = 4 :=
by
  sorry

end xiaoMing_xiaoHong_diff_university_l152_152943


namespace third_butcher_delivered_8_packages_l152_152444

variables (x y z t1 t2 t3 : ℕ)

-- Given Conditions
axiom h1 : x = 10
axiom h2 : y = 7
axiom h3 : 4 * x + 4 * y + 4 * z = 100
axiom t1_time : t1 = 8
axiom t2_time : t2 = 10
axiom t3_time : t3 = 18

-- Proof Problem
theorem third_butcher_delivered_8_packages :
  z = 8 :=
by
  -- proof to be filled
  sorry

end third_butcher_delivered_8_packages_l152_152444


namespace find_a_l152_152974

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - Real.exp (1 - x) - a * x
noncomputable def g (a x : ℝ) : ℝ := Real.exp x + Real.exp (1 - x) - a

theorem find_a (x₁ x₂ a : ℝ) (h₁ : g a x₁ = 0) (h₂ : g a x₂ = 0) (hf : f a x₁ + f a x₂ = -4) : a = 4 :=
sorry

end find_a_l152_152974


namespace capacitor_capacitance_l152_152668

theorem capacitor_capacitance 
  (U ε Q : ℝ) 
  (hQ : Q = (U^2 * (ε - 1)^2 * C) /  (2 * ε * (ε + 1)))
  : C = (2 * ε * (ε + 1) * Q) / (U^2 * (ε - 1)^2) :=
by
  sorry

end capacitor_capacitance_l152_152668


namespace degree_to_radian_conversion_l152_152474

theorem degree_to_radian_conversion : (-330 : ℝ) * (π / 180) = -(11 * π / 6) :=
by 
  sorry

end degree_to_radian_conversion_l152_152474


namespace compare_f_values_l152_152228

variable (a : ℝ) (f : ℝ → ℝ) (m n : ℝ)

theorem compare_f_values (h_a : 0 < a ∧ a < 1)
    (h_f : ∀ x > 0, f (Real.logb a x) = a * (x^2 - 1) / (x * (a^2 - 1)))
    (h_mn : m > n ∧ n > 0 ∧ m > 0) :
    f (1 / n) > f (1 / m) := by 
  sorry

end compare_f_values_l152_152228


namespace length_of_AB_l152_152017

theorem length_of_AB
  (height h : ℝ)
  (AB CD : ℝ)
  (ratio_AB_ADC : (1/2 * AB * h) / (1/2 * CD * h) = 5/4)
  (sum_AB_CD : AB + CD = 300) :
  AB = 166.67 :=
by
  -- The proof goes here.
  sorry

end length_of_AB_l152_152017


namespace polynomial_multiplication_l152_152637

theorem polynomial_multiplication (x z : ℝ) :
  (3*x^5 - 7*z^3) * (9*x^10 + 21*x^5*z^3 + 49*z^6) = 27*x^15 - 343*z^9 :=
by
  sorry

end polynomial_multiplication_l152_152637


namespace election_proof_l152_152554

noncomputable def election_problem : Prop :=
  ∃ (V : ℝ) (votesA votesB votesC : ℝ),
  (votesA = 0.35 * V) ∧
  (votesB = votesA + 1800) ∧
  (votesC = 0.5 * votesA) ∧
  (V = votesA + votesB + votesC) ∧
  (V = 14400) ∧
  ((votesA / V) * 100 = 35) ∧
  ((votesB / V) * 100 = 47.5) ∧
  ((votesC / V) * 100 = 17.5)

theorem election_proof : election_problem := sorry

end election_proof_l152_152554


namespace first_player_always_wins_l152_152558

theorem first_player_always_wins :
  ∃ A B : ℤ, A ≠ 0 ∧ B ≠ 0 ∧
  (A = 1998 ∧ B = -2 * 1998) ∧
  (∀ a b c : ℤ, (a = A ∨ a = B ∨ a = 1998) ∧ 
                (b = A ∨ b = B ∨ b = 1998) ∧ 
                (c = A ∨ c = B ∨ c = 1998) ∧ 
                a ≠ b ∧ b ≠ c ∧ a ≠ c → 
                ∃ x1 x2 : ℚ, x1 ≠ x2 ∧ 
                (a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0)) :=
by
  sorry

end first_player_always_wins_l152_152558


namespace simplify_division_l152_152804

theorem simplify_division (x : ℝ) : 2 * x^8 / x^4 = 2 * x^4 := 
by sorry

end simplify_division_l152_152804


namespace round_155_628_l152_152079

theorem round_155_628 :
  round (155.628 : Real) = 156 := by
  sorry

end round_155_628_l152_152079


namespace largest_multiple_of_15_less_than_500_l152_152866

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), n < 500 ∧ 15 ∣ n ∧ ∀ (m : ℕ), m < 500 ∧ 15 ∣ m -> m ≤ n :=
sorry

end largest_multiple_of_15_less_than_500_l152_152866


namespace team_members_run_distance_l152_152405

-- Define the given conditions
def total_distance : ℕ := 150
def members : ℕ := 5

-- Prove the question == answer given the conditions
theorem team_members_run_distance :
  total_distance / members = 30 :=
by
  sorry

end team_members_run_distance_l152_152405


namespace age_difference_is_18_l152_152895

def difference_in_ages (X Y Z : ℕ) : ℕ := (X + Y) - (Y + Z)
def younger_by_eighteen (X Z : ℕ) : Prop := Z = X - 18

theorem age_difference_is_18 (X Y Z : ℕ) (h : younger_by_eighteen X Z) : difference_in_ages X Y Z = 18 := by
  sorry

end age_difference_is_18_l152_152895


namespace nth_term_sequence_l152_152278

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (2 ^ n) - 1

theorem nth_term_sequence (n : ℕ) : 
  sequence n = 2 ^ n - 1 :=
by
  sorry

end nth_term_sequence_l152_152278


namespace cone_radius_l152_152562

theorem cone_radius
    (l : ℝ) (n : ℝ) (r : ℝ)
    (h1 : l = 2 * Real.pi)
    (h2 : n = 120)
    (h3 : l = (n * Real.pi * r) / 180 ) :
    r = 3 :=
sorry

end cone_radius_l152_152562


namespace tangent_line_at_point_l152_152661

noncomputable def tangent_line_eq (x y : ℝ) : Prop := x^3 - y = 0

theorem tangent_line_at_point :
  tangent_line_eq (-2) (-8) →
  ∃ (k : ℝ), (k = 12) ∧ (12 * x - y + 16 = 0) :=
sorry

end tangent_line_at_point_l152_152661


namespace solve_fraction_l152_152220

variable (x y : ℝ)
variable (h1 : y > x)
variable (h2 : x > 0)
variable (h3 : x / y + y / x = 8)

theorem solve_fraction : (x + y) / (x - y) = Real.sqrt (5 / 3) :=
by
  sorry

end solve_fraction_l152_152220


namespace Djibo_sister_age_l152_152194

variable (d s : ℕ)
variable (h1 : d = 17)
variable (h2 : d - 5 + (s - 5) = 35)

theorem Djibo_sister_age : s = 28 :=
by sorry

end Djibo_sister_age_l152_152194


namespace percentage_increase_numerator_l152_152361

variable (N D : ℝ) (P : ℝ)
variable (h1 : N / D = 0.75)
variable (h2 : (N * (1 + P / 100)) / (D * 0.92) = 15 / 16)

theorem percentage_increase_numerator :
  P = 15 :=
by
  sorry

end percentage_increase_numerator_l152_152361


namespace fresh_grape_weight_l152_152697

variable (D : ℝ) (F : ℝ)

axiom dry_grape_weight : D = 66.67
axiom fresh_grape_water_content : F * 0.25 = D * 0.75

theorem fresh_grape_weight : F = 200.01 :=
by sorry

end fresh_grape_weight_l152_152697


namespace greatest_integer_sum_l152_152255

def floor (x : ℚ) : ℤ := ⌊x⌋

theorem greatest_integer_sum :
  floor (2017 * 3 / 11) + 
  floor (2017 * 4 / 11) + 
  floor (2017 * 5 / 11) + 
  floor (2017 * 6 / 11) + 
  floor (2017 * 7 / 11) + 
  floor (2017 * 8 / 11) = 6048 :=
  by sorry

end greatest_integer_sum_l152_152255


namespace cow_count_l152_152431

theorem cow_count
  (initial_cows : ℕ) (cows_died : ℕ) (cows_sold : ℕ)
  (increase_cows : ℕ) (gift_cows : ℕ) (final_cows : ℕ) (bought_cows : ℕ) :
  initial_cows = 39 ∧ cows_died = 25 ∧ cows_sold = 6 ∧
  increase_cows = 24 ∧ gift_cows = 8 ∧ final_cows = 83 →
  bought_cows = 43 :=
by
  sorry

end cow_count_l152_152431


namespace find_f_2010_l152_152808

open Nat

variable (f : ℕ → ℕ)

axiom strictly_increasing : ∀ m n : ℕ, m < n → f m < f n

axiom function_condition : ∀ n : ℕ, f (f n) = 3 * n

theorem find_f_2010 : f 2010 = 3015 := sorry

end find_f_2010_l152_152808


namespace small_cubes_with_two_faces_painted_l152_152230

-- Statement of the problem
theorem small_cubes_with_two_faces_painted
  (remaining_cubes : ℕ)
  (edges_with_two_painted_faces : ℕ)
  (number_of_edges : ℕ) :
  remaining_cubes = 60 → edges_with_two_painted_faces = 2 → number_of_edges = 12 →
  (remaining_cubes - (4 * (edges_with_two_painted_faces - 1) * (number_of_edges))) = 28 :=
by
  sorry

end small_cubes_with_two_faces_painted_l152_152230


namespace total_wheels_l152_152634

def cars := 2
def car_wheels := 4
def bikes_with_one_wheel := 1
def bikes_with_two_wheels := 2
def trash_can_wheels := 2
def tricycle_wheels := 3
def roller_skate_wheels := 3 -- since one is missing a wheel
def wheelchair_wheels := 6 -- 4 large + 2 small wheels
def wagon_wheels := 4

theorem total_wheels : cars * car_wheels + 
                        bikes_with_one_wheel * 1 + 
                        bikes_with_two_wheels * 2 + 
                        trash_can_wheels + 
                        tricycle_wheels + 
                        roller_skate_wheels + 
                        wheelchair_wheels + 
                        wagon_wheels = 31 :=
by
  sorry

end total_wheels_l152_152634


namespace football_games_total_l152_152295

def total_football_games_per_season (games_per_month : ℝ) (num_months : ℝ) : ℝ :=
  games_per_month * num_months

theorem football_games_total (games_per_month : ℝ) (num_months : ℝ) (total_games : ℝ) :
  games_per_month = 323.0 ∧ num_months = 17.0 ∧ total_games = 5491.0 →
  total_football_games_per_season games_per_month num_months = total_games :=
by
  intros h
  have h1 : games_per_month = 323.0 := h.1
  have h2 : num_months = 17.0 := h.2.1
  have h3 : total_games = 5491.0 := h.2.2
  rw [h1, h2, h3]
  sorry

end football_games_total_l152_152295


namespace days_B_to_finish_work_l152_152944

-- Definition of work rates based on the conditions
def work_rate_A (A_days: ℕ) : ℚ := 1 / A_days
def work_rate_B (B_days: ℕ) : ℚ := 1 / B_days

-- Theorem that encapsulates the problem statement
theorem days_B_to_finish_work (A_days B_days together_days : ℕ) (work_rate_A_eq : work_rate_A 4 = 1/4) (work_rate_B_eq : work_rate_B 12 = 1/12) : 
  ∀ (remaining_work: ℚ), remaining_work = 1 - together_days * (work_rate_A 4 + work_rate_B 12) → 
  (remaining_work / (work_rate_B 12)) = 4 :=
by
  sorry

end days_B_to_finish_work_l152_152944


namespace positive_difference_of_two_numbers_l152_152800

theorem positive_difference_of_two_numbers :
  ∃ x y : ℚ, x + y = 40 ∧ 3 * y - 4 * x = 20 ∧ y - x = 80 / 7 :=
by
  sorry

end positive_difference_of_two_numbers_l152_152800


namespace required_run_rate_l152_152156

theorem required_run_rate (target : ℝ) (initial_run_rate : ℝ) (initial_overs : ℕ) (remaining_overs : ℕ) :
  target = 282 → initial_run_rate = 3.8 → initial_overs = 10 → remaining_overs = 40 →
  (target - initial_run_rate * initial_overs) / remaining_overs = 6.1 :=
by
  intros
  sorry

end required_run_rate_l152_152156


namespace choose_socks_l152_152711

open Nat

theorem choose_socks :
  (Nat.choose 8 4) = 70 :=
by 
  sorry

end choose_socks_l152_152711


namespace minimum_value_is_six_l152_152152

noncomputable def minimum_value_expression (x y z : ℝ) : ℝ :=
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z)

theorem minimum_value_is_six
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + y + z = 9) (h2 : y = 2 * x) :
  minimum_value_expression x y z = 6 :=
by
  sorry

end minimum_value_is_six_l152_152152


namespace gift_wrapping_combinations_l152_152347

theorem gift_wrapping_combinations 
  (wrapping_varieties : ℕ)
  (ribbon_colors : ℕ)
  (gift_card_types : ℕ)
  (H_wrapping_varieties : wrapping_varieties = 8)
  (H_ribbon_colors : ribbon_colors = 3)
  (H_gift_card_types : gift_card_types = 4) : 
  wrapping_varieties * ribbon_colors * gift_card_types = 96 := 
by
  sorry

end gift_wrapping_combinations_l152_152347


namespace product_sum_correct_l152_152055

def product_sum_eq : Prop :=
  let a := 4 * 10^6
  let b := 8 * 10^6
  (a * b + 2 * 10^13) = 5.2 * 10^13

theorem product_sum_correct : product_sum_eq :=
by
  sorry

end product_sum_correct_l152_152055


namespace quadrilateral_area_l152_152783

theorem quadrilateral_area {d o1 o2 : ℝ} (hd : d = 15) (ho1 : o1 = 6) (ho2 : o2 = 4) :
  (d * (o1 + o2)) / 2 = 75 := by
  sorry

end quadrilateral_area_l152_152783


namespace simplify_fraction_l152_152600

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l152_152600


namespace comparison_abc_l152_152123

noncomputable def a : ℝ := 0.98 + Real.sin 0.01
noncomputable def b : ℝ := Real.exp (-0.01)
noncomputable def c : ℝ := 0.5 * (Real.log 2023 / Real.log 2022 + Real.log 2022 / Real.log 2023)

theorem comparison_abc : c > b ∧ b > a := by
  sorry

end comparison_abc_l152_152123


namespace train_meeting_distance_l152_152977

theorem train_meeting_distance
  (d : ℝ) (tx ty: ℝ) (dx dy: ℝ)
  (hx : dx = 140) 
  (hy : dy = 140)
  (hx_speed : dx / tx = 35) 
  (hy_speed : dy / ty = 46.67) 
  (meet : tx = ty) :
  d = 60 := 
sorry

end train_meeting_distance_l152_152977


namespace sqrt_factorial_product_l152_152000

theorem sqrt_factorial_product:
  (Int.sqrt (Nat.factorial 4 * Nat.factorial 4)).toNat = 24 :=
by
  sorry

end sqrt_factorial_product_l152_152000


namespace length_of_second_platform_l152_152494

-- Definitions
def length_train : ℝ := 230
def time_first_platform : ℝ := 15
def length_first_platform : ℝ := 130
def total_distance_first_platform : ℝ := length_train + length_first_platform
def time_second_platform : ℝ := 20

-- Statement to prove
theorem length_of_second_platform : 
  ∃ L : ℝ, (total_distance_first_platform / time_first_platform) = ((length_train + L) / time_second_platform) ∧ L = 250 :=
by
  sorry

end length_of_second_platform_l152_152494


namespace ratio_adults_children_l152_152311

-- Definitions based on conditions
def children := 45
def total_adults (A : ℕ) : Prop := (2 / 3 : ℚ) * A = 10

-- The theorem stating the problem
theorem ratio_adults_children :
  ∃ A, total_adults A ∧ (A : ℚ) / children = (1 / 3 : ℚ) :=
by {
  sorry
}

end ratio_adults_children_l152_152311


namespace arithmetic_sequence_value_l152_152497

theorem arithmetic_sequence_value (a : ℕ) (h : 2 * a = 12) : a = 6 :=
by
  sorry

end arithmetic_sequence_value_l152_152497


namespace number_line_steps_l152_152281

theorem number_line_steps (total_steps : ℕ) (total_distance : ℕ) (steps_taken : ℕ) (result_distance : ℕ) 
  (h1 : total_distance = 36) (h2 : total_steps = 9) (h3 : steps_taken = 6) : 
  result_distance = (steps_taken * (total_distance / total_steps)) → result_distance = 24 :=
by
  intros H
  sorry

end number_line_steps_l152_152281


namespace not_net_of_cuboid_l152_152513

noncomputable def cuboid_closed_path (c : Type) (f : c → c) :=
∀ (x1 x2 : c), ∃ (y : c), f x1 = y ∧ f x2 = y

theorem not_net_of_cuboid (c : Type) [Nonempty c] [DecidableEq c] (net : c → Set c) (f : c → c) :
  cuboid_closed_path c f → ¬ (∀ x, net x = {x}) :=
by
  sorry

end not_net_of_cuboid_l152_152513


namespace connie_correct_answer_l152_152789

theorem connie_correct_answer 
  (x : ℝ) 
  (h1 : 2 * x = 80) 
  (correct_ans : ℝ := x / 3) :
  correct_ans = 40 / 3 :=
by
  sorry

end connie_correct_answer_l152_152789


namespace problem_proof_l152_152318

-- Problem statement
variable (f : ℕ → ℕ)

-- Condition: if f(k) ≥ k^2 then f(k+1) ≥ (k+1)^2
variable (h : ∀ k, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2)

-- Additional condition: f(4) ≥ 25
variable (h₀ : f 4 ≥ 25)

-- To prove: ∀ k ≥ 4, f(k) ≥ k^2
theorem problem_proof : ∀ k ≥ 4, f k ≥ k^2 :=
by
  sorry

end problem_proof_l152_152318


namespace middle_number_is_14_5_l152_152133

theorem middle_number_is_14_5 (x y z : ℝ) (h1 : x + y = 24) (h2 : x + z = 29) (h3 : y + z = 34) : y = 14.5 :=
sorry

end middle_number_is_14_5_l152_152133


namespace largest_integer_less_100_leaves_remainder_4_l152_152712

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l152_152712


namespace prove_temperature_on_Thursday_l152_152098

def temperature_on_Thursday 
  (temps : List ℝ)   -- List of temperatures for 6 days.
  (avg : ℝ)          -- Average temperature for the week.
  (sum_six_days : ℝ) -- Sum of temperature readings for 6 days.
  (days : ℕ := 7)    -- Number of days in the week.
  (missing_day : ℕ := 1)  -- One missing day (Thursday).
  (thurs_temp : ℝ := 99.8) -- Temperature on Thursday to be proved.
: Prop := (avg * days) - sum_six_days = thurs_temp

theorem prove_temperature_on_Thursday 
  : temperature_on_Thursday [99.1, 98.2, 98.7, 99.3, 99, 98.9] 99 593.2 :=
by
  sorry

end prove_temperature_on_Thursday_l152_152098


namespace cyclic_quadrilateral_condition_l152_152885

-- Definitions of the points and sides of the triangle
variables (A B C S E F : Type) 

-- Assume S is the centroid of triangle ABC
def is_centroid (A B C S : Type) : Prop := 
  -- actual centralized definition here (omitted)
  sorry

-- Assume E is the midpoint of side AB
def is_midpoint (A B E : Type) : Prop := 
  -- actual midpoint definition here (omitted)
  sorry 

-- Assume F is the midpoint of side AC
def is_midpoint_AC (A C F : Type) : Prop := 
  -- actual midpoint definition here (omitted)
  sorry 

-- Assume a quadrilateral AESF
def is_cyclic (A E S F : Type) : Prop :=
  -- actual cyclic definition here (omitted)
  sorry 

theorem cyclic_quadrilateral_condition 
  (A B C S E F : Type)
  (a b c : ℝ) 
  (h1 : is_centroid A B C S)
  (h2 : is_midpoint A B E) 
  (h3 : is_midpoint_AC A C F) :
  is_cyclic A E S F ↔ (c^2 + b^2 = 2 * a^2) :=
sorry

end cyclic_quadrilateral_condition_l152_152885


namespace spending_after_drink_l152_152555

variable (X : ℝ)
variable (Y : ℝ)

theorem spending_after_drink (h : X - 1.75 - Y = 6) : Y = X - 7.75 :=
by sorry

end spending_after_drink_l152_152555


namespace bianca_points_per_bag_l152_152813

theorem bianca_points_per_bag (total_bags : ℕ) (not_recycled : ℕ) (total_points : ℕ) 
  (h1 : total_bags = 17) 
  (h2 : not_recycled = 8) 
  (h3 : total_points = 45) : 
  total_points / (total_bags - not_recycled) = 5 :=
by
  sorry 

end bianca_points_per_bag_l152_152813


namespace sufficient_but_not_necessary_condition_l152_152730

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 5) :=
by
  sorry

end sufficient_but_not_necessary_condition_l152_152730


namespace molecular_weight_of_NH4I_correct_l152_152266

-- Define the atomic weights as given conditions
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_I : ℝ := 126.90

-- Define the calculation of the molecular weight of NH4I
def molecular_weight_NH4I : ℝ :=
  atomic_weight_N + 4 * atomic_weight_H + atomic_weight_I

-- Theorem stating the molecular weight of NH4I is 144.95 g/mol
theorem molecular_weight_of_NH4I_correct : molecular_weight_NH4I = 144.95 :=
by
  sorry

end molecular_weight_of_NH4I_correct_l152_152266


namespace polyhedron_equation_l152_152013

variables (V E F H T : ℕ)

-- Euler's formula for convex polyhedra
axiom euler_formula : V - E + F = 2
-- Number of faces is 50, and each face is either a triangle or a hexagon
axiom faces_count : F = 50
-- At each vertex, 3 triangles and 2 hexagons meet
axiom triangles_meeting : T = 3
axiom hexagons_meeting : H = 2

-- Prove that 100H + 10T + V = 230
theorem polyhedron_equation : 100 * H + 10 * T + V = 230 :=
  sorry

end polyhedron_equation_l152_152013


namespace root_equation_l152_152627

theorem root_equation (p q : ℝ) (hp : 3 * p^2 - 5 * p - 7 = 0)
                                  (hq : 3 * q^2 - 5 * q - 7 = 0) :
            (3 * p^2 - 3 * q^2) * (p - q)⁻¹ = 5 := 
by sorry

end root_equation_l152_152627


namespace sum_of_reciprocals_of_root_products_eq_4_l152_152978

theorem sum_of_reciprocals_of_root_products_eq_4
  (p q r s t : ℂ)
  (h_poly : ∀ x : ℂ, x^5 + 10*x^4 + 20*x^3 + 15*x^2 + 8*x + 5 = 0 ∨ (x - p)*(x - q)*(x - r)*(x - s)*(x - t) = 0)
  (h_vieta_2 : p*q + p*r + p*s + p*t + q*r + q*s + q*t + r*s + r*t + s*t = 20)
  (h_vieta_all : p*q*r*s*t = 5) :
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 4 := 
sorry

end sum_of_reciprocals_of_root_products_eq_4_l152_152978


namespace sales_tax_is_8_percent_l152_152688

-- Define the conditions
def total_before_tax : ℝ := 150
def total_with_tax : ℝ := 162

-- Define the relationship to find the sales tax percentage
noncomputable def sales_tax_percent (before_tax after_tax : ℝ) : ℝ :=
  ((after_tax - before_tax) / before_tax) * 100

-- State the theorem to prove the sales tax percentage is 8%
theorem sales_tax_is_8_percent :
  sales_tax_percent total_before_tax total_with_tax = 8 :=
by
  -- skipping the proof
  sorry

end sales_tax_is_8_percent_l152_152688


namespace average_age_of_5_people_l152_152229

theorem average_age_of_5_people (avg_age_18 : ℕ) (avg_age_9 : ℕ) (age_15th : ℕ) (total_persons: ℕ) (persons_9: ℕ) (remaining_persons: ℕ) : 
  avg_age_18 = 15 ∧ 
  avg_age_9 = 16 ∧ 
  age_15th = 56 ∧ 
  total_persons = 18 ∧ 
  persons_9 = 9 ∧ 
  remaining_persons = 5 → 
  (avg_age_18 * total_persons - avg_age_9 * persons_9 - age_15th) / remaining_persons = 14 := 
sorry

end average_age_of_5_people_l152_152229


namespace max_val_neg_5000_l152_152426

noncomputable def max_val_expression (x y : ℝ) : ℝ :=
  (x^2 + (1 / y^2)) * (x^2 + (1 / y^2) - 100) + (y^2 + (1 / x^2)) * (y^2 + (1 / x^2) - 100)

theorem max_val_neg_5000 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ x y, x > 0 ∧ y > 0 ∧ max_val_expression x y = -5000 :=
by
  sorry

end max_val_neg_5000_l152_152426


namespace Bryce_grapes_l152_152532

theorem Bryce_grapes : 
  ∃ x : ℝ, (∀ y : ℝ, y = (1/3) * x → y = x - 7) → x = 21 / 2 :=
by
  sorry

end Bryce_grapes_l152_152532


namespace quadratic_function_l152_152502

theorem quadratic_function :
  ∃ a : ℝ, ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = a * (x - 1) * (x - 5)) ∧ f 3 = 10 ∧ 
  f = fun x => -2.5 * x^2 + 15 * x - 12.5 :=
by
  sorry

end quadratic_function_l152_152502


namespace integer_pairs_count_l152_152877

theorem integer_pairs_count : ∃ (pairs : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), (x ≥ y ∧ (x, y) ∈ pairs → (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 211))
  ∧ pairs.card = 3 :=
by
  sorry

end integer_pairs_count_l152_152877


namespace express_a_in_terms_of_b_l152_152710

noncomputable def a : ℝ := Real.log 1250 / Real.log 6
noncomputable def b : ℝ := Real.log 50 / Real.log 3

theorem express_a_in_terms_of_b : a = (b + 0.6826) / 1.2619 :=
by
  sorry

end express_a_in_terms_of_b_l152_152710


namespace stratified_sample_sum_l152_152533

theorem stratified_sample_sum :
  let grains := 40
  let veg_oils := 10
  let animal_foods := 30
  let fruits_veggies := 20
  let total_varieties := grains + veg_oils + animal_foods + fruits_veggies
  let sample_size := 20
  let veg_oils_proportion := (veg_oils:ℚ) / total_varieties
  let fruits_veggies_proportion := (fruits_veggies:ℚ) / total_varieties
  let veg_oils_sample := sample_size * veg_oils_proportion
  let fruits_veggies_sample := sample_size * fruits_veggies_proportion
  veg_oils_sample + fruits_veggies_sample = 6 := sorry

end stratified_sample_sum_l152_152533


namespace expression_is_composite_l152_152887

theorem expression_is_composite (a b : ℕ) : ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ 4 * a^2 + 4 * a * b + 4 * a + 2 * b + 1 = m * n := 
by 
  sorry

end expression_is_composite_l152_152887


namespace sum_of_consecutive_odds_l152_152377

theorem sum_of_consecutive_odds (a : ℤ) (h : (a - 2) * a * (a + 2) = 9177) : (a - 2) + a + (a + 2) = 63 := 
sorry

end sum_of_consecutive_odds_l152_152377


namespace compute_ab_l152_152931

theorem compute_ab (a b : ℝ) 
  (h1 : b^2 - a^2 = 25) 
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = Real.sqrt 444 := 
by 
  sorry

end compute_ab_l152_152931


namespace find_a_l152_152780

noncomputable def unique_quad_solution (a : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1^2 - a * x1 + a = 1 → x2^2 - a * x2 + a = 1 → x1 = x2

theorem find_a (a : ℝ) (h : unique_quad_solution a) : a = 2 :=
sorry

end find_a_l152_152780


namespace positive_solution_in_interval_l152_152765

def quadratic (x : ℝ) := x^2 + 3 * x - 5

theorem positive_solution_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ quadratic x = 0 :=
sorry

end positive_solution_in_interval_l152_152765


namespace math_problem_l152_152429

variable (x Q : ℝ)

theorem math_problem (h : 5 * (6 * x - 3 * Real.pi) = Q) :
  15 * (18 * x - 9 * Real.pi) = 9 * Q := 
by
  sorry

end math_problem_l152_152429


namespace download_time_l152_152961

theorem download_time (avg_speed : ℤ) (size_A size_B size_C : ℤ) (gb_to_mb : ℤ) (secs_in_min : ℤ) :
  avg_speed = 30 →
  size_A = 450 →
  size_B = 240 →
  size_C = 120 →
  gb_to_mb = 1000 →
  secs_in_min = 60 →
  ( (size_A * gb_to_mb + size_B * gb_to_mb + size_C * gb_to_mb ) / avg_speed ) / secs_in_min = 450 := by
  intros h_avg h_A h_B h_C h_gb h_secs
  sorry

end download_time_l152_152961


namespace rationalize_denominator_div_l152_152118

theorem rationalize_denominator_div (h : 343 = 7 ^ 3) : 7 / Real.sqrt 343 = Real.sqrt 7 / 7 := 
by 
  sorry

end rationalize_denominator_div_l152_152118


namespace shaded_triangle_area_l152_152506

/--
The large equilateral triangle shown consists of 36 smaller equilateral triangles.
Each of the smaller equilateral triangles has an area of 10 cm². 
The area of the shaded triangle is K cm².
Prove that K = 110 cm².
-/
theorem shaded_triangle_area 
  (n : ℕ) (area_small : ℕ) (area_total : ℕ) (K : ℕ)
  (H1 : n = 36)
  (H2 : area_small = 10)
  (H3 : area_total = n * area_small)
  (H4 : K = 110)
: K = 110 :=
by
  -- Adding 'sorry' indicating missing proof steps.
  sorry

end shaded_triangle_area_l152_152506


namespace number_of_lattice_points_in_triangle_l152_152129

theorem number_of_lattice_points_in_triangle (N L S : ℕ) (A B O : (ℕ × ℕ)) :
  (A = (0, 30)) →
  (B = (20, 10)) →
  (O = (0, 0)) →
  (S = 300) →
  (L = 60) →
  S = N + L / 2 - 1 →
  N = 271 :=
by
  intros hA hB hO hS hL hPick
  sorry

end number_of_lattice_points_in_triangle_l152_152129


namespace runners_never_meet_l152_152451

theorem runners_never_meet
    (x : ℕ)  -- Speed of first runner
    (a : ℕ)  -- 1/3 of the circumference of the track
    (C : ℕ)  -- Circumference of the track
    (hC : C = 3 * a)  -- Given that C = 3 * a
    (h_speeds : 1 * x = x ∧ 2 * x = 2 * x ∧ 4 * x = 4 * x)  -- Speed ratios: 1:2:4
    (t : ℕ)  -- Time variable
: ¬(∃ t, (x * t % C = 2 * x * t % C ∧ 2 * x * t % C = 4 * x * t % C)) :=
by sorry

end runners_never_meet_l152_152451


namespace josh_marbles_earlier_l152_152654

-- Define the conditions
def marbles_lost : ℕ := 11
def marbles_now : ℕ := 8

-- Define the problem statement
theorem josh_marbles_earlier : marbles_lost + marbles_now = 19 :=
by
  sorry

end josh_marbles_earlier_l152_152654


namespace next_correct_time_l152_152564

def clock_shows_correct_time (start_date : String) (start_time : String) (time_lost_per_hour : Int) : String :=
  if start_date = "March 21" ∧ start_time = "12:00 PM" ∧ time_lost_per_hour = 25 then
    "June 1, 12:00 PM"
  else
    "unknown"

theorem next_correct_time :
  clock_shows_correct_time "March 21" "12:00 PM" 25 = "June 1, 12:00 PM" :=
by sorry

end next_correct_time_l152_152564


namespace triangles_with_perimeter_20_l152_152607

theorem triangles_with_perimeter_20 (sides : Finset (Finset ℕ)) : 
  (∀ {a b c : ℕ}, (a + b + c = 20) → (a > 0) → (b > 0) → (c > 0) 
  → (a + b > c) → (a + c > b) → (b + c > a) → ({a, b, c} ∈ sides)) 
  → sides.card = 8 := 
by
  sorry

end triangles_with_perimeter_20_l152_152607


namespace number_modulo_conditions_l152_152066

theorem number_modulo_conditions : 
  ∃ n : ℕ, 
  (n % 10 = 9) ∧ 
  (n % 9 = 8) ∧ 
  (n % 8 = 7) ∧ 
  (n % 7 = 6) ∧ 
  (n % 6 = 5) ∧ 
  (n % 5 = 4) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (n % 2 = 1) ∧ 
  (n = 2519) :=
by
  sorry

end number_modulo_conditions_l152_152066


namespace union_of_S_and_T_l152_152723

def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem union_of_S_and_T : S ∪ T = {1, 3, 5, 6} := 
by
  sorry

end union_of_S_and_T_l152_152723


namespace b_gives_c_start_l152_152545

variable (Va Vb Vc : ℝ)

-- Conditions given in the problem
def condition1 : Prop := Va / Vb = 1000 / 930
def condition2 : Prop := Va / Vc = 1000 / 800
def race_distance : ℝ := 1000

-- Proposition to prove
theorem b_gives_c_start (h1 : condition1 Va Vb) (h2 : condition2 Va Vc) :
  ∃ x : ℝ, (1000 - x) / 1000 = (930 / 800) :=
sorry

end b_gives_c_start_l152_152545


namespace inequality_ge_one_l152_152751

theorem inequality_ge_one {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by sorry

end inequality_ge_one_l152_152751


namespace unique_function_and_sum_calculate_n_times_s_l152_152421

def f : ℝ → ℝ := sorry

theorem unique_function_and_sum :
  (∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z) →
  (∃! g : ℝ → ℝ, ∀ x, f x = g x) ∧ f 3 = 0 :=
sorry

theorem calculate_n_times_s :
  ∃ n s : ℕ, (∃! f : ℝ → ℝ, ∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z) ∧ n = 1 ∧ s = (0 : ℝ) ∧ n * s = 0 :=
sorry

end unique_function_and_sum_calculate_n_times_s_l152_152421


namespace evaluate_f_at_t_plus_one_l152_152498

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 1

-- Define the proposition to be proved
theorem evaluate_f_at_t_plus_one (t : ℝ) : f (t + 1) = 3 * t + 2 := by
  sorry

end evaluate_f_at_t_plus_one_l152_152498


namespace root_exists_in_interval_l152_152121

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_exists_in_interval : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := 
by
  sorry

end root_exists_in_interval_l152_152121


namespace simplify_expression_l152_152741

noncomputable def term1 : ℝ := 3 / (Real.sqrt 2 + 2)
noncomputable def term2 : ℝ := 4 / (Real.sqrt 5 - 2)
noncomputable def simplifiedExpression : ℝ := 1 / (term1 + term2)
noncomputable def finalExpression : ℝ := 1 / (11 + 4 * Real.sqrt 5 - 3 * Real.sqrt 2 / 2)

theorem simplify_expression : simplifiedExpression = finalExpression := by
  sorry

end simplify_expression_l152_152741


namespace sum_of_distinct_integers_l152_152901

theorem sum_of_distinct_integers 
  (a b c d e : ℤ)
  (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 60)
  (h2 : (7 - a) ≠ (7 - b) ∧ (7 - a) ≠ (7 - c) ∧ (7 - a) ≠ (7 - d) ∧ (7 - a) ≠ (7 - e))
  (h3 : (7 - b) ≠ (7 - c) ∧ (7 - b) ≠ (7 - d) ∧ (7 - b) ≠ (7 - e))
  (h4 : (7 - c) ≠ (7 - d) ∧ (7 - c) ≠ (7 - e))
  (h5 : (7 - d) ≠ (7 - e)) : 
  a + b + c + d + e = 24 := 
sorry

end sum_of_distinct_integers_l152_152901


namespace contractor_absent_days_l152_152687

noncomputable def solve_contractor_problem : Prop :=
  ∃ (x y : ℕ), 
    x + y = 30 ∧ 
    25 * x - 750 / 100 * y = 555 ∧
    y = 6

theorem contractor_absent_days : solve_contractor_problem :=
  sorry

end contractor_absent_days_l152_152687


namespace quadratic_distinct_real_roots_l152_152536

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ a b c : ℝ, a = 1 ∧ b = m ∧ c = 9 ∧ a * c * 4 < b^2) ↔ (m < -6 ∨ m > 6) :=
by
  sorry

end quadratic_distinct_real_roots_l152_152536


namespace a6_value_l152_152308

variable (a_n : ℕ → ℤ)

/-- Given conditions in the arithmetic sequence -/
def arithmetic_sequence_property (a_n : ℕ → ℤ) :=
  ∀ n, a_n n = a_n 0 + n * (a_n 1 - a_n 0)

/-- Given sum condition a_4 + a_5 + a_6 + a_7 + a_8 = 150 -/
def sum_condition :=
  a_n 4 + a_n 5 + a_n 6 + a_n 7 + a_n 8 = 150

theorem a6_value (h : arithmetic_sequence_property a_n) (hsum : sum_condition a_n) :
  a_n 6 = 30 := 
by
  sorry

end a6_value_l152_152308


namespace fixed_point_1_3_l152_152822

noncomputable def fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Prop :=
  (f (1) = 3) where f x := a^(x-1) + 2

theorem fixed_point_1_3 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : fixed_point a h1 h2 :=
by
  unfold fixed_point
  sorry

end fixed_point_1_3_l152_152822


namespace quadrilateral_trapezium_l152_152090

theorem quadrilateral_trapezium (a b c d : ℝ) 
  (h1 : a / 6 = b / 7) 
  (h2 : b / 7 = c / 8) 
  (h3 : c / 8 = d / 9) 
  (h4 : a + b + c + d = 360) : 
  ((a + c = 180) ∨ (b + d = 180)) :=
by
  sorry

end quadrilateral_trapezium_l152_152090


namespace mcgregor_books_finished_l152_152280

def total_books := 89
def floyd_books := 32
def books_left := 23

theorem mcgregor_books_finished : ∀ mg_books : Nat, mg_books = total_books - floyd_books - books_left → mg_books = 34 := 
by
  intro mg_books
  sorry

end mcgregor_books_finished_l152_152280


namespace mean_yoga_practice_days_l152_152106

noncomputable def mean_number_of_days (counts : List ℕ) (days : List ℕ) : ℚ :=
  let total_days := List.zipWith (λ c d => c * d) counts days |>.sum
  let total_students := counts.sum
  total_days / total_students

def counts : List ℕ := [2, 4, 5, 3, 2, 1, 3]
def days : List ℕ := [1, 2, 3, 4, 5, 6, 7]

theorem mean_yoga_practice_days : mean_number_of_days counts days = 37 / 10 := 
by 
  sorry

end mean_yoga_practice_days_l152_152106


namespace average_speed_with_stoppages_l152_152425

theorem average_speed_with_stoppages
  (avg_speed_without_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ)
  (moving_time_per_hour : ℝ)
  (total_distance_moved : ℝ)
  (total_time_with_stoppages : ℝ) :
  avg_speed_without_stoppages = 60 → 
  stoppage_time_per_hour = 45 / 60 →
  moving_time_per_hour = 15 / 60 →
  total_distance_moved = avg_speed_without_stoppages * moving_time_per_hour →
  total_time_with_stoppages = 1 →
  (total_distance_moved / total_time_with_stoppages) = 15 :=
by
  intros
  sorry

end average_speed_with_stoppages_l152_152425


namespace smallest_N_satisfying_conditions_l152_152992

def is_divisible (n m : ℕ) : Prop :=
  m ∣ n

def satisfies_conditions (N : ℕ) : Prop :=
  (is_divisible N 10) ∧
  (is_divisible N 5) ∧
  (N > 15)

theorem smallest_N_satisfying_conditions : ∃ N, satisfies_conditions N ∧ N = 20 := 
  sorry

end smallest_N_satisfying_conditions_l152_152992


namespace triangle_interior_angle_leq_60_l152_152817

theorem triangle_interior_angle_leq_60 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (angle_sum : A + B + C = 180)
  (all_gt_60 : A > 60 ∧ B > 60 ∧ C > 60) :
  false :=
by
  sorry

end triangle_interior_angle_leq_60_l152_152817


namespace product_of_two_even_numbers_is_even_product_of_two_odd_numbers_is_odd_product_of_even_and_odd_number_is_even_product_of_odd_and_even_number_is_even_l152_152201

-- Definition of even and odd numbers
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Theorem statements for each condition

-- Prove that the product of two even numbers is even
theorem product_of_two_even_numbers_is_even (a b : ℤ) :
  is_even a → is_even b → is_even (a * b) :=
by sorry

-- Prove that the product of two odd numbers is odd
theorem product_of_two_odd_numbers_is_odd (c d : ℤ) :
  is_odd c → is_odd d → is_odd (c * d) :=
by sorry

-- Prove that the product of one even and one odd number is even
theorem product_of_even_and_odd_number_is_even (e f : ℤ) :
  is_even e → is_odd f → is_even (e * f) :=
by sorry

-- Prove that the product of one odd and one even number is even
theorem product_of_odd_and_even_number_is_even (g h : ℤ) :
  is_odd g → is_even h → is_even (g * h) :=
by sorry

end product_of_two_even_numbers_is_even_product_of_two_odd_numbers_is_odd_product_of_even_and_odd_number_is_even_product_of_odd_and_even_number_is_even_l152_152201


namespace tan_alpha_parallel_vectors_l152_152983

theorem tan_alpha_parallel_vectors
    (α : ℝ)
    (a : ℝ × ℝ := (6, 8))
    (b : ℝ × ℝ := (Real.sin α, Real.cos α))
    (h : a.fst * b.snd = a.snd * b.fst) :
    Real.tan α = 3 / 4 := 
sorry

end tan_alpha_parallel_vectors_l152_152983


namespace find_pencils_l152_152552

theorem find_pencils :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (6 ∣ n) ∧ (9 ∣ n) ∧ n % 7 = 1 ∧ n = 36 :=
by
  sorry

end find_pencils_l152_152552


namespace problem1_problem2_problem3_problem4_l152_152285

section
  variable (a b c d : Int)

  theorem problem1 : -27 + (-32) + (-8) + 72 = 5 := by
    sorry

  theorem problem2 : -4 - 2 * 32 + (-2 * 32) = -132 := by
    sorry

  theorem problem3 : (-48 : Int) / (-2 : Int)^3 - (-25 : Int) * (-4 : Int) + (-2 : Int)^3 = -102 := by
    sorry

  theorem problem4 : (-3 : Int)^2 - (3 / 2)^3 * (2 / 9) - 6 / (-(2 / 3))^3 = -12 := by
    sorry
end

end problem1_problem2_problem3_problem4_l152_152285


namespace roots_of_star_equation_l152_152176

def star (m n : ℝ) : ℝ := m * n^2 - m * n - 1

theorem roots_of_star_equation :
  ∀ x : ℝ, (star 1 x = 0) → (∃ a b : ℝ, a ≠ b ∧ x = a ∨ x = b) := 
by
  sorry

end roots_of_star_equation_l152_152176


namespace seating_arrangement_l152_152997

def valid_arrangements := 6

def Alice_refusal (A B C : Prop) := (¬ (A ∧ B)) ∧ (¬ (A ∧ C))
def Derek_refusal (D E C : Prop) := (¬ (D ∧ E)) ∧ (¬ (D ∧ C))

theorem seating_arrangement (A B C D E : Prop) : 
  Alice_refusal A B C ∧ Derek_refusal D E C → valid_arrangements = 6 := 
  sorry

end seating_arrangement_l152_152997


namespace movie_ticket_notation_l152_152701

-- Definition of movie ticket notation
def ticket_notation (row : ℕ) (seat : ℕ) : (ℕ × ℕ) :=
  (row, seat)

-- Given condition: "row 10, seat 3" is denoted as (10, 3)
def given := ticket_notation 10 3 = (10, 3)

-- Proof statement: "row 6, seat 16" is denoted as (6, 16)
theorem movie_ticket_notation : ticket_notation 6 16 = (6, 16) :=
by
  -- Proof omitted, since the theorem statement is the focus
  sorry

end movie_ticket_notation_l152_152701


namespace exist_two_numbers_with_GCD_and_LCM_l152_152803

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem exist_two_numbers_with_GCD_and_LCM :
  ∃ A B : ℕ, GCD A B = 21 ∧ LCM A B = 3969 ∧ ((A = 21 ∧ B = 3969) ∨ (A = 147 ∧ B = 567)) :=
by
  sorry

end exist_two_numbers_with_GCD_and_LCM_l152_152803


namespace cube_surface_area_equals_353_l152_152596

noncomputable def volume_of_prism : ℝ := 5 * 3 * 30
noncomputable def edge_length_of_cube (volume : ℝ) : ℝ := (volume)^(1/3)
noncomputable def surface_area_of_cube (edge_length : ℝ) : ℝ := 6 * edge_length^2

theorem cube_surface_area_equals_353 :
  surface_area_of_cube (edge_length_of_cube volume_of_prism) = 353 := by
sorry

end cube_surface_area_equals_353_l152_152596


namespace fraction_relationships_l152_152694

variables (a b c d : ℚ)

theorem fraction_relationships (h1 : a / b = 3) (h2 : b / c = 2 / 3) (h3 : c / d = 5) :
  d / a = 1 / 10 :=
by
  sorry

end fraction_relationships_l152_152694


namespace card_total_l152_152178

theorem card_total (Brenda Janet Mara : ℕ)
  (h1 : Janet = Brenda + 9)
  (h2 : Mara = 2 * Janet)
  (h3 : Mara = 150 - 40) :
  Brenda + Janet + Mara = 211 := by
  sorry

end card_total_l152_152178


namespace incorrect_conclusion_l152_152218

-- Define the given parabola.
def parabola (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Define the conditions for the parabola.
def parabola_opens_upwards : Prop := ∀ x y : ℝ, parabola (x + y) = (x + y - 2)^2 + 1
def axis_of_symmetry : Prop := ∀ x : ℝ, parabola x = parabola (4 - x)
def vertex_coordinates : Prop := parabola 2 = 1 ∧ (parabola 2, 2) = (1, 2)
def behavior_when_x_less_than_2 : Prop := ∀ x : ℝ, x < 2 → parabola x < parabola (x + 1)

-- The statement that needs to be proven in Lean 4.
theorem incorrect_conclusion : ¬ behavior_when_x_less_than_2 :=
  by
  sorry

end incorrect_conclusion_l152_152218


namespace find_z_l152_152270

theorem find_z (y z : ℝ) (k : ℝ) 
  (h1 : y = 3) (h2 : z = 16) (h3 : y ^ 2 * (z ^ (1 / 4)) = k)
  (h4 : k = 18) (h5 : y = 6) : z = 1 / 16 := by
  sorry

end find_z_l152_152270


namespace calum_spend_per_disco_ball_l152_152490

def calum_budget := 330
def food_cost_per_box := 25
def number_of_food_boxes := 10
def number_of_disco_balls := 4

theorem calum_spend_per_disco_ball : (calum_budget - food_cost_per_box * number_of_food_boxes) / number_of_disco_balls = 20 :=
by
  sorry

end calum_spend_per_disco_ball_l152_152490


namespace optionD_is_quad_eq_in_one_var_l152_152459

/-- Define a predicate for being a quadratic equation in one variable --/
def is_quad_eq_in_one_var (eq : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ ∀ x : ℕ, eq a b c

/-- Options as given predicates --/
def optionA (a b c : ℕ) : Prop := 3 * a^2 - 6 * b + 2 = 0
def optionB (a b c : ℕ) : Prop := a * a^2 - b * a + c = 0
def optionC (a b c : ℕ) : Prop := (1 / a^2) + b = c
def optionD (a b c : ℕ) : Prop := a^2 = 0

/-- Prove that Option D is a quadratic equation in one variable --/
theorem optionD_is_quad_eq_in_one_var : is_quad_eq_in_one_var optionD :=
sorry

end optionD_is_quad_eq_in_one_var_l152_152459


namespace obtuse_triangle_has_two_acute_angles_l152_152253

-- Definition of an obtuse triangle
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90)

-- A theorem to prove that an obtuse triangle has exactly 2 acute angles 
theorem obtuse_triangle_has_two_acute_angles (A B C : ℝ) (h : is_obtuse_triangle A B C) : 
  (A > 0 ∧ A < 90 → B > 0 ∧ B < 90 → C > 0 ∧ C < 90) ∧
  (A > 0 ∧ A < 90 ∧ B > 0 ∧ B < 90) ∨
  (A > 0 ∧ A < 90 ∧ C > 0 ∧ C < 90) ∨
  (B > 0 ∧ B < 90 ∧ C > 0 ∧ C < 90) :=
sorry

end obtuse_triangle_has_two_acute_angles_l152_152253


namespace value_of_M_l152_152593

theorem value_of_M (M : ℝ) :
  (20 / 100) * M = (60 / 100) * 1500 → M = 4500 :=
by
  intro h
  sorry

end value_of_M_l152_152593


namespace average_speed_l152_152883

theorem average_speed (v1 v2 : ℝ) (hv1 : v1 ≠ 0) (hv2 : v2 ≠ 0) : 
  2 / (1 / v1 + 1 / v2) = 2 * v1 * v2 / (v1 + v2) :=
by sorry

end average_speed_l152_152883


namespace average_of_numbers_not_1380_l152_152002

def numbers : List ℤ := [1200, 1300, 1400, 1520, 1530, 1200]

theorem average_of_numbers_not_1380 :
  let s := numbers.sum
  let n := numbers.length
  n > 0 → (s / n : ℚ) ≠ 1380 := by
  sorry

end average_of_numbers_not_1380_l152_152002


namespace circle_center_l152_152955

theorem circle_center (x y : ℝ) :
  x^2 + y^2 - 2*x + 4*y + 1 = 0 → (1, -2) = (1, -2) :=
by
  sorry

end circle_center_l152_152955


namespace total_weight_of_peppers_l152_152715

def green_peppers := 0.3333333333333333
def red_peppers := 0.4444444444444444
def yellow_peppers := 0.2222222222222222
def orange_peppers := 0.7777777777777778

theorem total_weight_of_peppers :
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 1.7777777777777777 :=
by
  sorry

end total_weight_of_peppers_l152_152715


namespace value_of_a8_l152_152044

variable (a : ℕ → ℝ) (a_1 : a 1 = 2) (common_sum : ℝ) (h_sum : common_sum = 5)
variable (equal_sum_sequence : ∀ n, a (n + 1) + a n = common_sum)

theorem value_of_a8 : a 8 = 3 :=
sorry

end value_of_a8_l152_152044


namespace graph_passes_through_fixed_point_l152_152673

theorem graph_passes_through_fixed_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
    ∃ (x y : ℝ), (x = -3) ∧ (y = -1) ∧ (y = a^(x + 3) - 2) :=
by
  sorry

end graph_passes_through_fixed_point_l152_152673


namespace concatenated_natural_irrational_l152_152857

def concatenated_natural_decimal : ℝ := 0.1234567891011121314151617181920 -- and so on

theorem concatenated_natural_irrational :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ concatenated_natural_decimal = p / q :=
sorry

end concatenated_natural_irrational_l152_152857


namespace number_square_l152_152298

-- Define conditions.
def valid_digit (d : ℕ) : Prop := d ≠ 0 ∧ d * d ≤ 9

-- Main statement.
theorem number_square (n : ℕ) (valid_digits : ∀ d, d ∈ [n / 100, (n / 10) % 10, n % 10] → valid_digit d) : 
  n = 233 :=
by
  -- Proof goes here
  sorry

end number_square_l152_152298


namespace solve_for_x_l152_152200

theorem solve_for_x : ∀ (x : ℂ) (i : ℂ), i^2 = -1 → 3 - 2 * i * x = 6 + i * x → x = i :=
by
  intros x i hI2 hEq
  sorry

end solve_for_x_l152_152200


namespace point_a_coordinates_l152_152142

open Set

theorem point_a_coordinates (A B : ℝ × ℝ) :
  B = (2, 4) →
  (A.1 = B.1 + 3 ∨ A.1 = B.1 - 3) ∧ A.2 = B.2 →
  dist A B = 3 →
  A = (5, 4) ∨ A = (-1, 4) :=
by
  intros hB hA hDist
  sorry

end point_a_coordinates_l152_152142


namespace net_change_in_salary_l152_152572

variable (S : ℝ)

theorem net_change_in_salary : 
  let increased_salary := S + (0.1 * S)
  let final_salary := increased_salary - (0.1 * increased_salary)
  final_salary - S = -0.01 * S :=
by
  sorry

end net_change_in_salary_l152_152572


namespace max_band_members_l152_152984

theorem max_band_members 
  (m : ℤ)
  (h1 : 30 * m % 31 = 7)
  (h2 : 30 * m < 1500) : 
  30 * m = 720 :=
sorry

end max_band_members_l152_152984


namespace range_of_2x_plus_y_l152_152465

theorem range_of_2x_plus_y {x y: ℝ} (h: x^2 / 4 + y^2 = 1) : -Real.sqrt 17 ≤ 2 * x + y ∧ 2 * x + y ≤ Real.sqrt 17 :=
sorry

end range_of_2x_plus_y_l152_152465


namespace transition_term_l152_152809

theorem transition_term (k : ℕ) : (2 * k + 2) + (2 * k + 3) = (2 * (k + 1) + 1) + (2 * k + 2) :=
by
  sorry

end transition_term_l152_152809


namespace fraction_problem_l152_152898

theorem fraction_problem (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := 
by
  sorry

end fraction_problem_l152_152898


namespace small_poster_ratio_l152_152612

theorem small_poster_ratio (total_posters : ℕ) (medium_posters large_posters small_posters : ℕ)
  (h1 : total_posters = 50)
  (h2 : medium_posters = 50 / 2)
  (h3 : large_posters = 5)
  (h4 : small_posters = total_posters - medium_posters - large_posters)
  (h5 : total_posters ≠ 0) :
  small_posters = 20 ∧ (small_posters : ℚ) / total_posters = 2 / 5 := 
sorry

end small_poster_ratio_l152_152612


namespace find_original_number_l152_152433

theorem find_original_number (x : ℕ) (h : 3 * (2 * x + 9) = 57) : x = 5 := 
by sorry

end find_original_number_l152_152433


namespace total_plants_in_garden_l152_152372

-- Definitions based on conditions
def basil_plants : ℕ := 5
def oregano_plants : ℕ := 2 + 2 * basil_plants

-- Theorem statement
theorem total_plants_in_garden : basil_plants + oregano_plants = 17 := by
  -- Skipping the proof with sorry
  sorry

end total_plants_in_garden_l152_152372


namespace train_cross_pole_time_l152_152160

noncomputable def train_time_to_cross_pole (length : ℕ) (speed_km_per_hr : ℕ) : ℕ :=
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  length / speed_m_per_s

theorem train_cross_pole_time :
  train_time_to_cross_pole 100 72 = 5 :=
by
  unfold train_time_to_cross_pole
  sorry

end train_cross_pole_time_l152_152160


namespace increasing_interval_l152_152893

noncomputable def y (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

def is_monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x1 x2, a < x1 ∧ x1 < x2 ∧ x2 < b → f x1 < f x2

theorem increasing_interval :
  is_monotonic_increasing y π (2 * π) :=
by
  -- Proof would go here
  sorry

end increasing_interval_l152_152893


namespace cyrus_pages_proof_l152_152675

def pages_remaining (total_pages: ℝ) (day1: ℝ) (day2: ℝ) (day3: ℝ) (day4: ℝ) (day5: ℝ) : ℝ :=
  total_pages - (day1 + day2 + day3 + day4 + day5)

theorem cyrus_pages_proof :
  let total_pages := 750
  let day1 := 30
  let day2 := 1.5 * day1
  let day3 := day2 / 2
  let day4 := 2.5 * day3
  let day5 := 15
  pages_remaining total_pages day1 day2 day3 day4 day5 = 581.25 :=
by 
  sorry

end cyrus_pages_proof_l152_152675


namespace max_b_c_value_l152_152088

theorem max_b_c_value (a b c : ℕ) (h1 : a > b) (h2 : a + b = 18) (h3 : c - b = 2) : b + c = 18 :=
sorry

end max_b_c_value_l152_152088


namespace count_multiples_of_15_l152_152457

theorem count_multiples_of_15 : ∃ n : ℕ, ∀ k, 12 < k ∧ k < 202 ∧ k % 15 = 0 ↔ k = 15 * n ∧ n = 13 := sorry

end count_multiples_of_15_l152_152457


namespace boys_from_school_a_not_study_science_l152_152199

theorem boys_from_school_a_not_study_science (total_boys : ℕ) (boys_from_school_a_percentage : ℝ) (science_study_percentage : ℝ)
  (total_boys_in_camp : total_boys = 250) (school_a_percent : boys_from_school_a_percentage = 0.20) 
  (science_percent : science_study_percentage = 0.30) :
  ∃ (boys_from_school_a_not_science : ℕ), boys_from_school_a_not_science = 35 :=
by
  sorry

end boys_from_school_a_not_study_science_l152_152199


namespace determine_f_2014_l152_152986

open Function

noncomputable def f : ℕ → ℕ :=
  sorry

theorem determine_f_2014
  (h1 : f 2 = 0)
  (h2 : f 3 > 0)
  (h3 : f 6042 = 2014)
  (h4 : ∀ m n : ℕ, f (m + n) - f m - f n ∈ ({0, 1} : Set ℕ)) :
  f 2014 = 671 :=
sorry

end determine_f_2014_l152_152986


namespace one_fourth_div_one_eighth_l152_152449

theorem one_fourth_div_one_eighth : (1 / 4) / (1 / 8) = 2 := by
  sorry

end one_fourth_div_one_eighth_l152_152449


namespace bowling_average_l152_152143

theorem bowling_average (A : ℝ) (W : ℕ) (hW : W = 145) (hW7 : W + 7 ≠ 0)
  (h : ( A * W + 26 ) / ( W + 7 ) = A - 0.4) : A = 12.4 := 
by 
  sorry

end bowling_average_l152_152143


namespace composite_19_8n_plus_17_l152_152256

theorem composite_19_8n_plus_17 (n : ℕ) (h : n > 0) : ¬ Nat.Prime (19 * 8^n + 17) := 
by 
  sorry

end composite_19_8n_plus_17_l152_152256


namespace arithmetic_sequence_a7_l152_152848

noncomputable def a_n (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1) * d

theorem arithmetic_sequence_a7 
  (a : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 4 - a 1) / 3)
  (h_a1 : a 1 = 3)
  (h_a4 : a 4 = 5) : 
  a 7 = 7 :=
by
  sorry

end arithmetic_sequence_a7_l152_152848


namespace apples_distribution_l152_152850

theorem apples_distribution (total_apples : ℝ) (apples_per_person : ℝ) (number_of_people : ℝ) 
    (h1 : total_apples = 45) (h2 : apples_per_person = 15.0) : number_of_people = 3 :=
by
  sorry

end apples_distribution_l152_152850


namespace sufficient_and_necessary_condition_l152_152724

theorem sufficient_and_necessary_condition {a : ℝ} :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a ≥ 4 :=
sorry

end sufficient_and_necessary_condition_l152_152724


namespace solve_for_x_l152_152074

theorem solve_for_x (x : ℝ) : 
  2.5 * ((3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5)) = 2000.0000000000002 → 
  x = 2.5 :=
by 
  sorry

end solve_for_x_l152_152074


namespace MeatMarket_sales_l152_152622

theorem MeatMarket_sales :
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  total_sales - planned_sales = 325 :=
by
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  show total_sales - planned_sales = 325
  sorry

end MeatMarket_sales_l152_152622


namespace area_of_region_l152_152122

noncomputable def circle_radius : ℝ := 3

noncomputable def segment_length : ℝ := 4

theorem area_of_region : ∃ (area : ℝ), area = 4 * Real.pi :=
by
  sorry

end area_of_region_l152_152122


namespace blue_balls_in_JarB_l152_152719

-- Defining the conditions
def ratio_white_blue (white blue : ℕ) : Prop := white / gcd white blue = 5 ∧ blue / gcd white blue = 3

def white_balls_in_B := 15

-- Proof statement
theorem blue_balls_in_JarB :
  ∃ (blue : ℕ), ratio_white_blue 15 blue ∧ blue = 9 :=
by {
  -- Proof outline (not required, thus just using sorry)
  sorry
}


end blue_balls_in_JarB_l152_152719


namespace games_planned_to_attend_this_month_l152_152760

theorem games_planned_to_attend_this_month (T A_l P_l M_l P_m : ℕ) 
  (h1 : T = 12) 
  (h2 : P_l = 17) 
  (h3 : M_l = 16) 
  (h4 : A_l = P_l - M_l) 
  (h5 : T = A_l + P_m) : P_m = 11 :=
by 
  sorry

end games_planned_to_attend_this_month_l152_152760


namespace faster_current_takes_more_time_l152_152086

theorem faster_current_takes_more_time (v v1 v2 S : ℝ) (h_v1_gt_v2 : v1 > v2) :
  let t1 := (2 * S * v) / (v^2 - v1^2)
  let t2 := (2 * S * v) / (v^2 - v2^2)
  t1 > t2 :=
by
  sorry

end faster_current_takes_more_time_l152_152086


namespace rationalize_denominator_sum_l152_152750

theorem rationalize_denominator_sum :
  let A := -4
  let B := 7
  let C := 3
  let D := 13
  let E := 1
  A + B + C + D + E = 20 := by
    sorry

end rationalize_denominator_sum_l152_152750


namespace parabola_one_intersection_l152_152528

theorem parabola_one_intersection (k : ℝ) :
  (∀ x : ℝ, x^2 - x + k = 0 → x = 0) → k = 1 / 4 :=
sorry

end parabola_one_intersection_l152_152528


namespace lollipop_count_l152_152469

theorem lollipop_count (total_cost : ℝ) (cost_per_lollipop : ℝ) (h1 : total_cost = 90) (h2 : cost_per_lollipop = 0.75) : 
  total_cost / cost_per_lollipop = 120 :=
by 
  sorry

end lollipop_count_l152_152469


namespace y_value_l152_152254

theorem y_value (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -6) : y = 29 :=
by
  sorry

end y_value_l152_152254


namespace candy_division_l152_152210

def pieces_per_bag (total_candies : ℕ) (bags : ℕ) : ℕ :=
total_candies / bags

theorem candy_division : pieces_per_bag 42 2 = 21 :=
by
  sorry

end candy_division_l152_152210


namespace exponential_function_passes_through_01_l152_152771

theorem exponential_function_passes_through_01 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : (a^0 = 1) :=
by
  sorry

end exponential_function_passes_through_01_l152_152771


namespace measure_angle_BCQ_l152_152527

/-- Given:
  - Segment AB has a length of 12 units.
  - Segment AC is 9 units long.
  - Segment AC : CB = 3 : 1.
  - A semi-circle is constructed with diameter AB.
  - Another smaller semi-circle is constructed with diameter CB.
  - A line segment CQ divides the combined area of the two semi-circles into two equal areas.

  Prove: The degree measure of angle BCQ is 11.25°.
-/ 
theorem measure_angle_BCQ (AB AC CB : ℝ) (hAB : AB = 12) (hAC : AC = 9) (hRatio : AC / CB = 3) :
  ∃ θ : ℝ, θ = 11.25 :=
by
  sorry

end measure_angle_BCQ_l152_152527


namespace num_different_pairs_l152_152864

theorem num_different_pairs :
  (∃ (A B : Finset ℕ), A ∪ B = {1, 2, 3, 4} ∧ A ≠ B ∧ (A, B) ≠ (B, A)) ∧
  (∃ n : ℕ, n = 81) :=
by
  -- Proof would go here, but it's skipped per instructions
  sorry

end num_different_pairs_l152_152864


namespace binary_to_base5_1101_l152_152584

-- Definition of the binary to decimal conversion for the given number
def binary_to_decimal (b: Nat): Nat :=
  match b with
  | 0    => 0
  | 1101 => 1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3
  | _    => 0  -- This is a specific case for the given problem

-- Definition of the decimal to base-5 conversion method
def decimal_to_base5 (d: Nat): Nat :=
  match d with
  | 0    => 0
  | 13   =>
    let rem1 := 13 % 5
    let div1 := 13 / 5
    let rem2 := div1 % 5
    let div2 := div1 / 5
    rem2 * 10 + rem1  -- Assemble the base-5 number from remainders
  | _    => 0  -- This is a specific case for the given problem

-- Proof statement: conversion of 1101 in binary to base-5 yields 23
theorem binary_to_base5_1101 : decimal_to_base5 (binary_to_decimal 1101) = 23 := by
  sorry

end binary_to_base5_1101_l152_152584


namespace delta_f_l152_152292

open BigOperators

def f (n : ℕ) : ℕ := ∑ i in Finset.range n, (i + 1) * (n - i)

theorem delta_f (k : ℕ) : f (k + 1) - f k = ∑ i in Finset.range (k + 1), (i + 1) :=
by
  sorry

end delta_f_l152_152292


namespace birthday_guests_l152_152492

theorem birthday_guests (total_guests : ℕ) (women men children guests_left men_left children_left : ℕ)
  (h_total : total_guests = 60)
  (h_women : women = total_guests / 2)
  (h_men : men = 15)
  (h_children : children = total_guests - (women + men))
  (h_men_left : men_left = men / 3)
  (h_children_left : children_left = 5)
  (h_guests_left : guests_left = men_left + children_left) :
  (total_guests - guests_left) = 50 :=
by sorry

end birthday_guests_l152_152492


namespace find_AX_length_l152_152356

noncomputable def AX_length (AC BC BX : ℕ) : ℚ :=
AC * (BX / BC)

theorem find_AX_length :
  let AC := 25
  let BC := 35
  let BX := 30
  AX_length AC BC BX = 150 / 7 :=
by
  -- proof is omitted using 'sorry'
  sorry

end find_AX_length_l152_152356


namespace m_le_n_l152_152947

def polygon : Type := sorry  -- A placeholder definition for polygon.

variables (M : polygon) -- The polygon \( M \)
def max_non_overlapping_circles (M : polygon) : ℕ := sorry -- The maximum number of non-overlapping circles with diameter 1 inside \( M \).
def min_covering_circles (M : polygon) : ℕ := sorry -- The minimum number of circles with radius 1 required to cover \( M \).

theorem m_le_n (M : polygon) : min_covering_circles M ≤ max_non_overlapping_circles M :=
sorry

end m_le_n_l152_152947


namespace find_f_60_l152_152296

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition.

axiom functional_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom f_48 : f 48 = 36

theorem find_f_60 : f 60 = 28.8 := by 
  sorry

end find_f_60_l152_152296


namespace min_value_reciprocal_l152_152923

variable {a b : ℝ}

theorem min_value_reciprocal (h1 : a * b > 0) (h2 : a + 4 * b = 1) : 
  ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ ((1/a) + (1/b) = 9) := 
by
  sorry

end min_value_reciprocal_l152_152923


namespace kyler_wins_zero_l152_152284

-- Definitions based on conditions provided
def peter_games_won : ℕ := 5
def peter_games_lost : ℕ := 3
def emma_games_won : ℕ := 4
def emma_games_lost : ℕ := 4
def kyler_games_lost : ℕ := 4

-- Number of games each player played
def peter_total_games : ℕ := peter_games_won + peter_games_lost
def emma_total_games : ℕ := emma_games_won + emma_games_lost
def kyler_total_games (k : ℕ) : ℕ := k + kyler_games_lost

-- Step 1: total number of games in the tournament
def total_games (k : ℕ) : ℕ := (peter_total_games + emma_total_games + kyler_total_games k) / 2

-- Step 2: Total games equation
def games_equation (k : ℕ) : Prop := 
  (peter_games_won + emma_games_won + k = total_games k)

-- The proof problem, we need to prove Kyler's wins
theorem kyler_wins_zero : games_equation 0 := by
  -- proof omitted
  sorry

end kyler_wins_zero_l152_152284


namespace imaginary_part_of_z_l152_152441

-- Define the problem conditions and what to prove
theorem imaginary_part_of_z (z : ℂ) (h : (1 - I) * z = I) : z.im = 1 / 2 :=
sorry

end imaginary_part_of_z_l152_152441


namespace positive_even_representation_l152_152150

theorem positive_even_representation (k : ℕ) (h : k > 0) :
  ∃ (a b : ℤ), (2 * k : ℤ) = a * b ∧ a + b = 0 := 
by
  sorry

end positive_even_representation_l152_152150


namespace problem_statement_l152_152424

theorem problem_statement : 
  (∀ x y : ℤ, y = 2 * x^2 - 3 * x + 4 ∧ y = 6 ∧ x = 2) → (2 * 2 - 3 * (-3) + 4 * 4 = 29) :=
by
  intro h
  sorry

end problem_statement_l152_152424


namespace triangle_is_isosceles_l152_152722

variable (a b m_a m_b : ℝ)

-- Conditions: 
-- A circle touches two sides of a triangle (denoted as a and b).
-- The circle also touches the medians m_a and m_b drawn to these sides.
-- Given equations:
axiom Eq1 : (1/2) * a + (1/3) * m_b = (1/2) * b + (1/3) * m_a
axiom Eq3 : (1/2) * a + m_b = (1/2) * b + m_a

-- Question: Prove that the triangle is isosceles, i.e., a = b
theorem triangle_is_isosceles : a = b :=
by
  sorry

end triangle_is_isosceles_l152_152722


namespace corrected_mean_is_correct_l152_152448

-- Define the initial conditions
def initial_mean : ℝ := 36
def n_obs : ℝ := 50
def incorrect_obs : ℝ := 23
def correct_obs : ℝ := 45

-- Calculate the incorrect total sum
def incorrect_total_sum : ℝ := initial_mean * n_obs

-- Define the corrected total sum
def corrected_total_sum : ℝ := incorrect_total_sum - incorrect_obs + correct_obs

-- State the main theorem to be proved
theorem corrected_mean_is_correct : corrected_total_sum / n_obs = 36.44 := by
  sorry

end corrected_mean_is_correct_l152_152448


namespace gcd_ab_conditions_l152_152114

theorem gcd_ab_conditions 
  (a b : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = 1) : 
  Nat.gcd (a + b) (a - b) = 1 ∨ Nat.gcd (a + b) (a - b) = 2 := 
sorry

end gcd_ab_conditions_l152_152114


namespace squirrel_walnut_count_l152_152790

-- Lean 4 statement
theorem squirrel_walnut_count :
  let initial_boy_walnuts := 12
  let gathered_walnuts := 6
  let dropped_walnuts := 1
  let initial_girl_walnuts := 0
  let brought_walnuts := 5
  let eaten_walnuts := 2
  (initial_boy_walnuts + gathered_walnuts - dropped_walnuts + initial_girl_walnuts + brought_walnuts - eaten_walnuts) = 20 :=
by
  -- Proof goes here
  sorry

end squirrel_walnut_count_l152_152790


namespace people_who_own_neither_l152_152814

theorem people_who_own_neither (total_people cat_owners cat_and_dog_owners dog_owners non_cat_dog_owners: ℕ)
        (h1: total_people = 522)
        (h2: 20 * cat_and_dog_owners = cat_owners)
        (h3: 7 * dog_owners = 10 * (dog_owners + cat_and_dog_owners))
        (h4: 2 * non_cat_dog_owners = (non_cat_dog_owners + dog_owners)):
    non_cat_dog_owners = 126 := 
by
  sorry

end people_who_own_neither_l152_152814


namespace taxi_charge_l152_152073

theorem taxi_charge :
  ∀ (initial_fee additional_charge_per_segment total_distance total_charge : ℝ),
  initial_fee = 2.05 →
  total_distance = 3.6 →
  total_charge = 5.20 →
  (total_charge - initial_fee) / (5/2 * total_distance) = 0.35 :=
by
  intros initial_fee additional_charge_per_segment total_distance total_charge
  intros h_initial_fee h_total_distance h_total_charge
  -- Proof here
  sorry

end taxi_charge_l152_152073


namespace graveling_cost_l152_152443

theorem graveling_cost
  (length_lawn : ℝ) (width_lawn : ℝ)
  (width_road : ℝ)
  (cost_per_sq_m : ℝ)
  (h1: length_lawn = 80) (h2: width_lawn = 40) (h3: width_road = 10) (h4: cost_per_sq_m = 3) :
  (length_lawn * width_road + width_lawn * width_road - width_road * width_road) * cost_per_sq_m = 3900 := 
by
  sorry

end graveling_cost_l152_152443


namespace lcm_of_two_numbers_l152_152853

theorem lcm_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 9) (h2 : a * b = 1800) : Nat.lcm a b = 200 :=
by
  sorry

end lcm_of_two_numbers_l152_152853


namespace valid_5_digit_numbers_l152_152282

noncomputable def num_valid_numbers (d : ℕ) (h : d ≠ 7) (h_valid : d < 10) (h_pos : d ≠ 0) : ℕ :=
  let choices_first_place := 7   -- choices for the first digit (1-9, excluding d and 7)
  let choices_other_places := 8  -- choices for other digits (0-9, excluding d and 7)
  choices_first_place * choices_other_places ^ 4

theorem valid_5_digit_numbers (d : ℕ) (h_d_ne_7 : d ≠ 7) (h_d_valid : d < 10) (h_d_pos : d ≠ 0) :
  num_valid_numbers d h_d_ne_7 h_d_valid h_d_pos = 28672 := sorry

end valid_5_digit_numbers_l152_152282


namespace find_a1_l152_152869

-- Definitions used in the conditions
variables {a : ℕ → ℝ} -- Sequence a(n)
variable (n : ℕ) -- Number of terms
noncomputable def arithmeticSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))

noncomputable def arithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a m + (n - m) * (a 2 - a 1)

theorem find_a1 (h_seq : arithmeticSeq a)
  (h_sum_first_100 : arithmeticSum a 100 = 100)
  (h_sum_last_100 : arithmeticSum (λ i => a (i + 900)) 100 = 1000) :
  a 1 = 101 / 200 :=
  sorry

end find_a1_l152_152869


namespace calculation_correct_l152_152267

noncomputable def calc_expression : Float :=
  20.17 * 69 + 201.7 * 1.3 - 8.2 * 1.7

theorem calculation_correct : calc_expression = 1640 := 
  by 
    sorry

end calculation_correct_l152_152267


namespace complex_division_l152_152170

theorem complex_division (z : ℂ) (hz : (3 + 4 * I) * z = 25) : z = 3 - 4 * I :=
sorry

end complex_division_l152_152170


namespace teacher_proctor_arrangements_l152_152537

theorem teacher_proctor_arrangements {f m : ℕ} (hf : f = 2) (hm : m = 5) :
  (∃ moving_teachers : ℕ, moving_teachers = 1 ∧ (f - moving_teachers) + m = 7 
   ∧ (f - moving_teachers).choose 2 = 21)
  ∧ 2 * 21 = 42 :=
by
    sorry

end teacher_proctor_arrangements_l152_152537


namespace solution_set_of_inequality_l152_152393

theorem solution_set_of_inequality : 
  { x : ℝ | (1 : ℝ) * (2 * x + 1) < (x + 1) } = { x | -1 < x ∧ x < 0 } :=
by
  sorry

end solution_set_of_inequality_l152_152393


namespace reciprocal_sum_hcf_lcm_l152_152679

variables (m n : ℕ)

def HCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem reciprocal_sum_hcf_lcm (h₁ : HCF m n = 6) (h₂ : LCM m n = 210) (h₃ : m + n = 60) :
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 21 :=
by
  -- The proof will be inserted here.
  sorry

end reciprocal_sum_hcf_lcm_l152_152679


namespace geometric_sequence_strictly_increasing_iff_l152_152788

noncomputable def geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q^(n-1)

theorem geometric_sequence_strictly_increasing_iff (a_1 q : ℝ) :
  (∀ n : ℕ, geometric_sequence a_1 q (n+2) > geometric_sequence a_1 q n) ↔ 
  (∀ n : ℕ, geometric_sequence a_1 q (n+1) > geometric_sequence a_1 q n) := 
by
  sorry

end geometric_sequence_strictly_increasing_iff_l152_152788


namespace printer_fraction_l152_152787

noncomputable def basic_computer_price : ℝ := 2000
noncomputable def total_basic_price : ℝ := 2500
noncomputable def printer_price : ℝ := total_basic_price - basic_computer_price -- inferred as 500

noncomputable def enhanced_computer_price : ℝ := basic_computer_price + 500
noncomputable def total_enhanced_price : ℝ := enhanced_computer_price + printer_price -- inferred as 3000

theorem printer_fraction  (h1 : basic_computer_price + printer_price = total_basic_price)
                          (h2 : basic_computer_price = 2000)
                          (h3 : enhanced_computer_price = basic_computer_price + 500) :
  printer_price / total_enhanced_price = 1 / 6 :=
  sorry

end printer_fraction_l152_152787


namespace airplane_average_speed_l152_152785

theorem airplane_average_speed :
  ∃ v : ℝ, 
  (1140 = 12 * (0.9 * v) + 26 * (1.2 * v)) ∧ 
  v = 27.14 := 
by
  sorry

end airplane_average_speed_l152_152785


namespace intersection_is_correct_l152_152854

-- Define the sets A and B based on the given conditions
def setA : Set ℝ := { x | x > 1/3 }
def setB : Set ℝ := { y | -3 ≤ y ∧ y ≤ 3 }

-- Prove that the intersection of A and B is (1/3, 3]
theorem intersection_is_correct : setA ∩ setB = { x | 1/3 < x ∧ x ≤ 3 } := 
by
  sorry

end intersection_is_correct_l152_152854


namespace widgets_per_shipping_box_l152_152136

theorem widgets_per_shipping_box 
  (widgets_per_carton : ℕ := 3)
  (carton_width : ℕ := 4)
  (carton_length : ℕ := 4)
  (carton_height : ℕ := 5)
  (box_width : ℕ := 20)
  (box_length : ℕ := 20)
  (box_height : ℕ := 20) :
  (widgets_per_carton * ((box_width * box_length * box_height) / (carton_width * carton_length * carton_height))) = 300 :=
by
  sorry

end widgets_per_shipping_box_l152_152136


namespace dice_probability_l152_152209

noncomputable def probability_same_face (throws : ℕ) (dice : ℕ) : ℚ :=
  1 - (1 - (1 / 6) ^ dice) ^ throws

theorem dice_probability : 
  probability_same_face 5 10 = 1 - (1 - (1 / 6) ^ 10) ^ 5 :=
by 
  sorry

end dice_probability_l152_152209


namespace sum_of_all_possible_values_of_z_l152_152582

noncomputable def sum_of_z_values (w x y z : ℚ) : ℚ :=
if h : w < x ∧ x < y ∧ y < z ∧ 
       (w + x = 1 ∧ w + y = 2 ∧ w + z = 3 ∧ x + y = 4 ∨ 
        w + x = 1 ∧ w + y = 2 ∧ w + z = 4 ∧ x + y = 3) ∧ 
       ((w + x) ≠ (w + y) ∧ (w + x) ≠ (w + z) ∧ (w + x) ≠ (x + y) ∧ (w + x) ≠ (x + z) ∧ (w + x) ≠ (y + z)) ∧ 
       ((w + y) ≠ (w + z) ∧ (w + y) ≠ (x + y) ∧ (w + y) ≠ (x + z) ∧ (w + y) ≠ (y + z)) ∧ 
       ((w + z) ≠ (x + y) ∧ (w + z) ≠ (x + z) ∧ (w + z) ≠ (y + z)) ∧ 
       ((x + y) ≠ (x + z) ∧ (x + y) ≠ (y + z)) ∧ 
       ((x + z) ≠ (y + z)) then
  if w + z = 4 then
    4 + 7/2
  else 0
else
  0

theorem sum_of_all_possible_values_of_z : sum_of_z_values w x y z = 15 / 2 :=
by sorry

end sum_of_all_possible_values_of_z_l152_152582


namespace intersection_point_divides_chord_l152_152707

theorem intersection_point_divides_chord (R AB PO : ℝ)
    (hR: R = 11) (hAB: AB = 18) (hPO: PO = 7) :
    ∃ (AP PB : ℝ), (AP / PB = 2 ∨ AP / PB = 1 / 2) ∧ (AP + PB = AB) := by
  sorry

end intersection_point_divides_chord_l152_152707


namespace smallest_possible_value_l152_152548

theorem smallest_possible_value 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ 9 / 2 :=
sorry

end smallest_possible_value_l152_152548


namespace goldfinch_percentage_l152_152196

def number_of_goldfinches := 6
def number_of_sparrows := 9
def number_of_grackles := 5
def total_birds := number_of_goldfinches + number_of_sparrows + number_of_grackles
def goldfinch_fraction := (number_of_goldfinches : ℚ) / total_birds

theorem goldfinch_percentage : goldfinch_fraction * 100 = 30 := 
by
  sorry

end goldfinch_percentage_l152_152196


namespace wheels_motion_is_rotation_l152_152365

def motion_wheel_car := "rotation"
def question_wheels_motion := "What is the type of motion exhibited by the wheels of a moving car?"

theorem wheels_motion_is_rotation :
  (question_wheels_motion = "What is the type of motion exhibited by the wheels of a moving car?" ∧ 
   motion_wheel_car = "rotation") → motion_wheel_car = "rotation" :=
by
  sorry

end wheels_motion_is_rotation_l152_152365


namespace problem1_problem2_l152_152307

noncomputable def f (x a b : ℝ) : ℝ := 2 * x ^ 2 - 2 * a * x + b

noncomputable def set_A (a b : ℝ) : Set ℝ := {x | f x a b > 0 }

noncomputable def set_B (t : ℝ) : Set ℝ := {x | |x - t| ≤ 1 }

theorem problem1 (a b : ℝ) (h : f (-1) a b = -8) :
  (∀ x, x ∈ (set_A a b)ᶜ ∪ set_B 1 ↔ -3 ≤ x ∧ x ≤ 2) :=
  sorry

theorem problem2 (a b : ℝ) (t : ℝ) (h : f (-1) a b = -8) (h_not_P : (set_A a b) ∩ (set_B t) = ∅) :
  -2 ≤ t ∧ t ≤ 0 :=
  sorry

end problem1_problem2_l152_152307


namespace find_f_value_l152_152982

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f (x : ℝ) : f (-x) = -f x
axiom even_f_shift (x : ℝ) : f (-x + 1) = f (x + 1)
axiom f_interval (x : ℝ) (h : 2 < x ∧ x < 4) : f x = |x - 3|

theorem find_f_value : f 1 + f 2 + f 3 + f 4 = 0 :=
by
  sorry

end find_f_value_l152_152982


namespace arithmetic_sequence_range_of_m_l152_152139

-- Conditions
variable {a : ℕ+ → ℝ} -- Sequence of positive terms
variable {S : ℕ+ → ℝ} -- Sum of the first n terms
variable (h : ∀ n, 2 * Real.sqrt (S n) = a n + 1) -- Relationship condition

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence (n : ℕ+)
    (h1 : ∀ n, 2 * Real.sqrt (S n) = a n + 1)
    (h2 : S 1 = 1 / 4 * (a 1 + 1)^2) :
    ∃ d : ℝ, ∀ n, a (n + 1) = a n + d :=
sorry

-- Part 2: Find range of m
theorem range_of_m (T : ℕ+ → ℝ)
    (hT : ∀ n, T n = 1 / 4 * n + 1 / 8 * (1 - 1 / (2 * n + 1))) :
    ∃ m : ℝ, (6 / 7 : ℝ) < m ∧ m ≤ 10 / 9 ∧
    (∃ n₁ n₂ n₃ : ℕ+, (n₁ < n₂ ∧ n₂ < n₃) ∧ (∀ n, T n < m ↔ n₁ ≤ n ∧ n ≤ n₃)) :=
sorry

end arithmetic_sequence_range_of_m_l152_152139


namespace anna_money_left_eur_l152_152435

noncomputable def total_cost_usd : ℝ := 4 * 1.50 + 7 * 2.25 + 3 * 0.75 + 3.00 * 0.80
def sales_tax_rate : ℝ := 0.075
def exchange_rate : ℝ := 0.85
def initial_amount_usd : ℝ := 50

noncomputable def total_cost_with_tax_usd : ℝ := total_cost_usd * (1 + sales_tax_rate)
noncomputable def total_cost_eur : ℝ := total_cost_with_tax_usd * exchange_rate
noncomputable def initial_amount_eur : ℝ := initial_amount_usd * exchange_rate

noncomputable def money_left_eur : ℝ := initial_amount_eur - total_cost_eur

theorem anna_money_left_eur : abs (money_left_eur - 18.38) < 0.01 := by
  -- Add proof steps here
  sorry

end anna_money_left_eur_l152_152435


namespace books_more_than_figures_l152_152358

-- Definitions of initial conditions
def initial_action_figures := 2
def initial_books := 10
def added_action_figures := 4

-- Problem statement to prove
theorem books_more_than_figures :
  initial_books - (initial_action_figures + added_action_figures) = 4 :=
by
  -- Proof goes here
  sorry

end books_more_than_figures_l152_152358


namespace maria_drank_8_bottles_l152_152874

def initial_bottles : ℕ := 14
def bought_bottles : ℕ := 45
def remaining_bottles : ℕ := 51

theorem maria_drank_8_bottles :
  let total_bottles := initial_bottles + bought_bottles
  let drank_bottles := total_bottles - remaining_bottles
  drank_bottles = 8 :=
by
  let total_bottles := 14 + 45
  let drank_bottles := total_bottles - 51
  show drank_bottles = 8
  sorry

end maria_drank_8_bottles_l152_152874


namespace range_of_a_l152_152120

theorem range_of_a 
  (e : ℝ) (h_e_pos : 0 < e) 
  (a : ℝ) 
  (h_equation : ∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ ≠ x₂ ∧ (1 / e ^ x₁ - a / x₁ = 0) ∧ (1 / e ^ x₂ - a / x₂ = 0)) :
  0 < a ∧ a < 1 / e :=
by
  sorry

end range_of_a_l152_152120


namespace min_value_of_sum_l152_152807

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 4 / x + 1 / y = 1) : x + y = 9 :=
by
  -- sorry used to skip the proof
  sorry

end min_value_of_sum_l152_152807


namespace distance_between_city_centers_l152_152171

def distance_on_map : ℝ := 45  -- Distance on the map in cm
def scale_factor : ℝ := 20     -- Scale factor (1 cm : 20 km)

theorem distance_between_city_centers : distance_on_map * scale_factor = 900 := by
  sorry

end distance_between_city_centers_l152_152171


namespace evaluate_expression_l152_152412

theorem evaluate_expression :
  |7 - (8^2) * (3 - 12)| - |(5^3) - (Real.sqrt 11)^4| = 579 := 
by 
  sorry

end evaluate_expression_l152_152412


namespace intersecting_points_radius_squared_l152_152968

noncomputable def parabola1 (x : ℝ) : ℝ := (x - 2) ^ 2
noncomputable def parabola2 (y : ℝ) : ℝ := (y - 5) ^ 2 - 1

theorem intersecting_points_radius_squared :
  ∃ (x y : ℝ), (y = parabola1 x ∧ x = parabola2 y) → (x - 2) ^ 2 + (y - 5) ^ 2 = 16 := by
sorry

end intersecting_points_radius_squared_l152_152968


namespace x_intercept_of_line_l152_152597

open Real

theorem x_intercept_of_line : 
  ∃ x : ℝ, 
  (∃ m : ℝ, m = (3 - -5) / (10 - -6) ∧ (∀ y : ℝ, y = m * (x - 10) + 3)) ∧ 
  (∀ y : ℝ, y = 0 → x = 4) :=
sorry

end x_intercept_of_line_l152_152597


namespace solution1_solution2_l152_152060

open Complex

noncomputable def problem1 : Prop := 
  ((3 - I) / (1 + I)) ^ 2 = -3 - 4 * I

noncomputable def problem2 (z : ℂ) : Prop := 
  z = 1 + I → (2 / z - z = -2 * I)

theorem solution1 : problem1 := 
  by sorry

theorem solution2 : problem2 (1 + I) :=
  by sorry

end solution1_solution2_l152_152060


namespace range_of_a_l152_152639

section
variables (a : ℝ)
def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a = 1 ∨ a ≤ -2 :=
sorry
end

end range_of_a_l152_152639


namespace largest_rectangle_area_l152_152519

noncomputable def max_rectangle_area_with_perimeter (p : ℕ) : ℕ := sorry

theorem largest_rectangle_area (p : ℕ) (h : p = 60) : max_rectangle_area_with_perimeter p = 225 :=
sorry

end largest_rectangle_area_l152_152519


namespace units_digit_Fermat_5_l152_152244

def Fermat_number (n: ℕ) : ℕ :=
  2 ^ (2 ^ n) + 1

theorem units_digit_Fermat_5 : (Fermat_number 5) % 10 = 7 := by
  sorry

end units_digit_Fermat_5_l152_152244


namespace distribution_plans_equiv_210_l152_152595

noncomputable def number_of_distribution_plans : ℕ := sorry -- we will skip the proof

theorem distribution_plans_equiv_210 :
  number_of_distribution_plans = 210 := by
  sorry

end distribution_plans_equiv_210_l152_152595


namespace new_average_weight_l152_152151

def num_people := 6
def avg_weight1 := 154
def weight_seventh := 133

theorem new_average_weight :
  (num_people * avg_weight1 + weight_seventh) / (num_people + 1) = 151 := by
  sorry

end new_average_weight_l152_152151


namespace batsman_running_percentage_l152_152747

theorem batsman_running_percentage (total_runs boundary_runs six_runs : ℕ) 
  (h1 : total_runs = 120) (h2 : boundary_runs = 3 * 4) (h3 : six_runs = 8 * 6) : 
  (total_runs - (boundary_runs + six_runs)) * 100 / total_runs = 50 := 
sorry

end batsman_running_percentage_l152_152747


namespace matrix_sum_correct_l152_152418

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, 3], ![-2, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![-1, 5], ![8, -3]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 8], ![6, -2]]

theorem matrix_sum_correct : A + B = C := by
  sorry

end matrix_sum_correct_l152_152418


namespace max_k_inequality_l152_152379

open Real

theorem max_k_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b) * (a * b + 1) * (b + 1) ≥ (27 / 4) * a * b^2 :=
by
  sorry

end max_k_inequality_l152_152379


namespace find_difference_of_squares_l152_152488

variable (x y : ℝ)
variable (h1 : (x + y) ^ 2 = 81)
variable (h2 : x * y = 18)

theorem find_difference_of_squares : (x - y) ^ 2 = 9 := by
  sorry

end find_difference_of_squares_l152_152488


namespace simplification_problem_l152_152720

theorem simplification_problem (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h_sum : p + q + r = 1) :
  (1 / (q^2 + r^2 - p^2) + 1 / (p^2 + r^2 - q^2) + 1 / (p^2 + q^2 - r^2) = 3 / (1 - 2 * q * r)) :=
by
  sorry

end simplification_problem_l152_152720


namespace geometric_sequence_cannot_determine_a3_l152_152481

/--
Suppose we have a geometric sequence {a_n} such that 
the product of the first five terms a_1 * a_2 * a_3 * a_4 * a_5 = 32.
We aim to show that the value of a_3 cannot be determined with the given information.
-/
theorem geometric_sequence_cannot_determine_a3 (a : ℕ → ℝ) (r : ℝ) (h : a 0 * a 1 * a 2 * a 3 * a 4 = 32) : 
  ¬ ∃ x : ℝ, a 2 = x :=
sorry

end geometric_sequence_cannot_determine_a3_l152_152481


namespace total_snowfall_l152_152891

variable (morning_snowfall : ℝ) (afternoon_snowfall : ℝ)

theorem total_snowfall {morning_snowfall afternoon_snowfall : ℝ} (h_morning : morning_snowfall = 0.12) (h_afternoon : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.62 :=
sorry

end total_snowfall_l152_152891


namespace find_x_y_z_l152_152560

theorem find_x_y_z (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x * y = x + y) (h2 : y * z = 3 * (y + z)) (h3 : z * x = 2 * (z + x)) : 
  x + y + z = 12 :=
sorry

end find_x_y_z_l152_152560


namespace largest_4_digit_div_by_5_smallest_primes_l152_152686

noncomputable def LCM_5_smallest_primes : ℕ := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))

theorem largest_4_digit_div_by_5_smallest_primes :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 9240 := by
  sorry

end largest_4_digit_div_by_5_smallest_primes_l152_152686


namespace percentage_of_masters_is_76_l152_152588

variable (x y : ℕ)  -- Let x be the number of junior players, y be the number of master players
variable (junior_avg master_avg team_avg : ℚ)

-- The conditions given in the problem
def juniors_avg_points : Prop := junior_avg = 22
def masters_avg_points : Prop := master_avg = 47
def team_avg_points (x y : ℕ) (junior_avg master_avg team_avg : ℚ) : Prop :=
  (22 * x + 47 * y) / (x + y) = 41

def proportion_of_masters (x y : ℕ) : ℚ := (y : ℚ) / (x + y)

-- The theorem to be proved
theorem percentage_of_masters_is_76 (x y : ℕ) (junior_avg master_avg team_avg : ℚ) :
  juniors_avg_points junior_avg →
  masters_avg_points master_avg →
  team_avg_points x y junior_avg master_avg team_avg →
  proportion_of_masters x y = 19 / 25 := 
sorry

end percentage_of_masters_is_76_l152_152588


namespace find_number_l152_152148

-- We define n, x, y as real numbers
variables (n x y : ℝ)

-- Define the conditions as hypotheses
def condition1 : Prop := n * (x - y) = 4
def condition2 : Prop := 6 * x - 3 * y = 12

-- Define the theorem we need to prove: If the conditions hold, then n = 2
theorem find_number (h1 : condition1 n x y) (h2 : condition2 x y) : n = 2 := 
sorry

end find_number_l152_152148


namespace alphabet_letters_l152_152353

theorem alphabet_letters (DS S_only Total D_only : ℕ) 
  (h_DS : DS = 9) 
  (h_S_only : S_only = 24) 
  (h_Total : Total = 40) 
  (h_eq : Total = D_only + S_only + DS) 
  : D_only = 7 := 
by
  sorry

end alphabet_letters_l152_152353


namespace percent_decrease_area_pentagon_l152_152314

open Real

noncomputable def area_hexagon (s : ℝ) : ℝ :=
  (3 * sqrt 3 / 2) * s ^ 2

noncomputable def area_pentagon (s : ℝ) : ℝ :=
  (sqrt (5 * (5 + 2 * sqrt 5)) / 4) * s ^ 2

noncomputable def diagonal_pentagon (s : ℝ) : ℝ :=
  (1 + sqrt 5) / 2 * s

theorem percent_decrease_area_pentagon :
  let s_p := sqrt (400 / sqrt (5 * (5 + 2 * sqrt 5)))
  let d := diagonal_pentagon s_p
  let new_d := 0.9 * d
  let new_s := new_d / ((1 + sqrt 5) / 2)
  let new_area := area_pentagon new_s
  (100 - new_area) / 100 * 100 = 20 :=
by
  sorry

end percent_decrease_area_pentagon_l152_152314


namespace annual_subscription_cost_l152_152097

-- Definitions based on the conditions

def monthly_cost : ℝ := 10
def months_per_year : ℕ := 12
def discount_rate : ℝ := 0.20

-- The statement based on the correct answer
theorem annual_subscription_cost : 
  (monthly_cost * months_per_year) * (1 - discount_rate) = 96 := 
by
  sorry

end annual_subscription_cost_l152_152097


namespace josh_payment_correct_l152_152432

/-- Josh's purchase calculation -/
def josh_total_payment : ℝ :=
  let string_cheese_cost := 0.10
  let number_of_cheeses_per_pack := 20
  let packs_bought := 3
  let sales_tax_rate := 0.12
  let cost_before_tax := packs_bought * number_of_cheeses_per_pack * string_cheese_cost
  let sales_tax := sales_tax_rate * cost_before_tax
  cost_before_tax + sales_tax

theorem josh_payment_correct :
  josh_total_payment = 6.72 := by
  sorry

end josh_payment_correct_l152_152432


namespace sum_max_min_ratio_ellipse_l152_152878

theorem sum_max_min_ratio_ellipse :
  ∃ (a b : ℝ), (∀ (x y : ℝ), 3*x^2 + 2*x*y + 4*y^2 - 18*x - 28*y + 50 = 0 → (y/x = a ∨ y/x = b)) ∧ a + b = 13 :=
by
  sorry

end sum_max_min_ratio_ellipse_l152_152878


namespace polygon_expected_value_l152_152135

def polygon_expected_sides (area_square : ℝ) (flower_prob : ℝ) (area_flower : ℝ) (hex_sides : ℝ) (pent_sides : ℝ) : ℝ :=
  hex_sides * flower_prob + pent_sides * (area_square - flower_prob)

theorem polygon_expected_value :
  polygon_expected_sides 1 (π - 1) (π - 1) 6 5 = π + 4 :=
by
  -- Proof is skipped
  sorry

end polygon_expected_value_l152_152135


namespace perpendicular_lines_m_l152_152032

theorem perpendicular_lines_m (m : ℝ) : 
    (∀ x y : ℝ, x - 2 * y + 5 = 0 → 
                2 * x + m * y - 6 = 0 → 
                (1 / 2) * (-2 / m) = -1) → 
    m = 1 :=
by
  intros
  -- proof goes here
  sorry

end perpendicular_lines_m_l152_152032


namespace product_mod_25_l152_152006

theorem product_mod_25 (m : ℕ) (h : 0 ≤ m ∧ m < 25) : 
  43 * 67 * 92 % 25 = 2 :=
by
  sorry

end product_mod_25_l152_152006


namespace compute_expression_l152_152761

theorem compute_expression : 9 * (1 / 13) * 26 = 18 :=
by
  sorry

end compute_expression_l152_152761


namespace difference_between_local_and_face_value_l152_152706

def numeral := 657903

def local_value (n : ℕ) : ℕ :=
  if n = 7 then 70000 else 0

def face_value (n : ℕ) : ℕ :=
  n

theorem difference_between_local_and_face_value :
  local_value 7 - face_value 7 = 69993 :=
by
  sorry

end difference_between_local_and_face_value_l152_152706


namespace scientific_notation_of_216000_l152_152264

theorem scientific_notation_of_216000 :
  216000 = 2.16 * 10^5 :=
sorry

end scientific_notation_of_216000_l152_152264


namespace factor_expression_l152_152890

theorem factor_expression (x y : ℝ) : x * y^2 - 4 * x = x * (y + 2) * (y - 2) := 
by
  sorry

end factor_expression_l152_152890


namespace sum_of_coefficients_of_expansion_l152_152571

theorem sum_of_coefficients_of_expansion (x y : ℝ) :
  (3*x - 4*y) ^ 20 = 1 :=
by 
  sorry

end sum_of_coefficients_of_expansion_l152_152571


namespace number_is_165_l152_152111

def is_between (n a b : ℕ) : Prop := a ≤ n ∧ n ≤ b
def is_odd (n : ℕ) : Prop := n % 2 = 1
def contains_digit_5 (n : ℕ) : Prop := ∃ k : ℕ, 10^k * 5 ≤ n ∧ n < 10^(k+1) * 5
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem number_is_165 : 
  (is_between 165 144 169) ∧ 
  (is_odd 165) ∧ 
  (contains_digit_5 165) ∧ 
  (is_divisible_by_3 165) :=
by 
  sorry 

end number_is_165_l152_152111


namespace arithmetic_seq_a4_l152_152592

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions and the goal to prove
theorem arithmetic_seq_a4 (h₁ : is_arithmetic_sequence a d) (h₂ : a 2 + a 6 = 10) : 
  a 4 = 5 :=
by
  sorry

end arithmetic_seq_a4_l152_152592


namespace minimum_n_for_80_intersections_l152_152224

-- Define what an n-sided polygon is and define the intersection condition
def n_sided_polygon (n : ℕ) : Type := sorry -- definition of n-sided polygon

-- Define the condition when boundaries of two polygons intersect at exactly 80 points
def boundaries_intersect_at (P Q : n_sided_polygon n) (k : ℕ) : Prop := sorry -- definition of boundaries intersecting at exactly k points

theorem minimum_n_for_80_intersections (n : ℕ) :
  (∃ (P Q : n_sided_polygon n), boundaries_intersect_at P Q 80) → (n ≥ 10) :=
sorry

end minimum_n_for_80_intersections_l152_152224


namespace avg_difference_l152_152952

theorem avg_difference : 
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 80 + 15) / 3
  avg1 - avg2 = 5 :=
by
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 80 + 15) / 3
  show avg1 - avg2 = 5
  sorry

end avg_difference_l152_152952


namespace find_value_of_a_plus_b_l152_152161

variables (a b : ℝ)

theorem find_value_of_a_plus_b
  (h1 : a^3 - 3 * a^2 + 5 * a = 1)
  (h2 : b^3 - 3 * b^2 + 5 * b = 5) :
  a + b = 2 := 
sorry

end find_value_of_a_plus_b_l152_152161


namespace solution_set_inequality_l152_152662

theorem solution_set_inequality (m : ℝ) (x : ℝ) 
  (h : 3 - m < 0) : (2 - m) * x + m > 2 ↔ x < 1 :=
by
  sorry

end solution_set_inequality_l152_152662


namespace correct_mms_packs_used_l152_152140

variable (num_sundaes_monday : ℕ) (mms_per_sundae_monday : ℕ)
variable (num_sundaes_tuesday : ℕ) (mms_per_sundae_tuesday : ℕ)
variable (mms_per_pack : ℕ)

-- Conditions
def conditions : Prop := 
  num_sundaes_monday = 40 ∧ 
  mms_per_sundae_monday = 6 ∧ 
  num_sundaes_tuesday = 20 ∧
  mms_per_sundae_tuesday = 10 ∧ 
  mms_per_pack = 40

-- Question: How many m&m packs does Kekai use?
def number_of_mms_packs (num_sundaes_monday mms_per_sundae_monday 
                         num_sundaes_tuesday mms_per_sundae_tuesday 
                         mms_per_pack : ℕ) : ℕ := 
  (num_sundaes_monday * mms_per_sundae_monday + num_sundaes_tuesday * mms_per_sundae_tuesday) / mms_per_pack

-- Theorem to prove the correct number of m&m packs used
theorem correct_mms_packs_used (h : conditions num_sundaes_monday mms_per_sundae_monday 
                                              num_sundaes_tuesday mms_per_sundae_tuesday 
                                              mms_per_pack) : 
  number_of_mms_packs num_sundaes_monday mms_per_sundae_monday 
                      num_sundaes_tuesday mms_per_sundae_tuesday 
                      mms_per_pack = 11 := by {
  -- Proof goes here
  sorry
}

end correct_mms_packs_used_l152_152140


namespace largest_possible_median_l152_152745

theorem largest_possible_median (l : List ℕ) (h1 : l.length = 10) 
  (h2 : ∀ x ∈ l, 0 < x) (exists6l : ∃ l1 : List ℕ, l1 = [3, 4, 5, 7, 8, 9]) :
  ∃ median_val : ℝ, median_val = 8.5 := 
sorry

end largest_possible_median_l152_152745


namespace renu_suma_combined_work_days_l152_152826

theorem renu_suma_combined_work_days :
  (1 / (1 / 8 + 1 / 4.8)) = 3 :=
by
  sorry

end renu_suma_combined_work_days_l152_152826


namespace remainder_is_zero_l152_152820

theorem remainder_is_zero :
  (86 * 87 * 88 * 89 * 90 * 91 * 92) % 7 = 0 := 
by 
  sorry

end remainder_is_zero_l152_152820


namespace tan_add_pi_over_four_sin_cos_ratio_l152_152967

-- Definition of angle α with the condition that tanα = 2
def α : ℝ := sorry -- Define α such that tan α = 2

-- The first Lean statement for proving tan(α + π/4) = -3
theorem tan_add_pi_over_four (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
sorry

-- The second Lean statement for proving (sinα + cosα) / (2sinα - cosα) = 1
theorem sin_cos_ratio (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 1 :=
sorry

end tan_add_pi_over_four_sin_cos_ratio_l152_152967


namespace find_primes_l152_152942

-- Definition of being a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0)

-- Lean 4 statement of the problem
theorem find_primes (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  3 * p^4 - 5 * q^4 - 4 * r^2 = 26 → p = 5 ∧ q = 3 ∧ r = 19 := 
by
  sorry

end find_primes_l152_152942


namespace father_present_age_l152_152493

theorem father_present_age (S F : ℕ) 
  (h1 : F = 3 * S + 3) 
  (h2 : F + 3 = 2 * (S + 3) + 10) : 
  F = 33 :=
by
  sorry

end father_present_age_l152_152493


namespace banana_permutations_l152_152374

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l152_152374


namespace sum_of_prime_factors_2310_l152_152781

def is_prime (n : Nat) : Prop :=
  2 ≤ n ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def prime_factors_sum (n : Nat) : Nat :=
  (List.filter Nat.Prime (Nat.factors n)).sum

theorem sum_of_prime_factors_2310 :
  prime_factors_sum 2310 = 28 :=
by
  sorry

end sum_of_prime_factors_2310_l152_152781


namespace greatest_negative_root_l152_152030

noncomputable def sine (x : ℝ) : ℝ := Real.sin (Real.pi * x)
noncomputable def cosine (x : ℝ) : ℝ := Real.cos (2 * Real.pi * x)

theorem greatest_negative_root :
  ∀ (x : ℝ), (x < 0 ∧ (sine x - cosine x) / ((sine x + 1)^2 + (Real.cos (Real.pi * x))^2) = 0) → 
    x ≤ -7/6 :=
by
  sorry

end greatest_negative_root_l152_152030


namespace proof_correct_chemical_information_l152_152403

def chemical_formula_starch : String := "(C_{6}H_{10}O_{5})_{n}"
def structural_formula_glycine : String := "H_{2}N-CH_{2}-COOH"
def element_in_glass_ceramics_cement : String := "Si"
def elements_cause_red_tide : List String := ["N", "P"]

theorem proof_correct_chemical_information :
  chemical_formula_starch = "(C_{6}H_{10}O_{5})_{n}" ∧
  structural_formula_glycine = "H_{2}N-CH_{2}-COOH" ∧
  element_in_glass_ceramics_cement = "Si" ∧
  elements_cause_red_tide = ["N", "P"] :=
by
  sorry

end proof_correct_chemical_information_l152_152403


namespace trigonometric_identity_l152_152241

theorem trigonometric_identity :
  (Real.cos (12 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.sin (12 * Real.pi / 180) * Real.sin (18 * Real.pi / 180) = 
   Real.cos (30 * Real.pi / 180)) :=
by
  sorry

end trigonometric_identity_l152_152241


namespace estimate_white_balls_l152_152906

theorem estimate_white_balls
  (total_balls : ℕ)
  (trials : ℕ)
  (white_draws : ℕ)
  (proportion_white : ℚ)
  (hw : total_balls = 10)
  (ht : trials = 400)
  (hd : white_draws = 240)
  (hprop : proportion_white = 0.6) :
  ∃ x : ℕ, x = 6 :=
by
  sorry

end estimate_white_balls_l152_152906


namespace right_triangle_side_length_l152_152784

theorem right_triangle_side_length (r f : ℝ) (h : f < 2 * r) :
  let c := 2 * r
  let a := (f / (4 * c)) * (f + Real.sqrt (f^2 + 8 * c^2))
  a = (f / (4 * (2 * r))) * (f + Real.sqrt (f^2 + 8 * (2 * r)^2)) :=
by
  let c := 2 * r
  let a := (f / (4 * c)) * (f + Real.sqrt (f^2 + 8 * c^2))
  have acalc : a = (f / (4 * (2 * r))) * (f + Real.sqrt (f^2 + 8 * (2 * r)^2)) := by sorry
  exact acalc

end right_triangle_side_length_l152_152784


namespace contractor_laborers_l152_152641

theorem contractor_laborers (x : ℕ) (h1 : 15 * x = 20 * (x - 5)) : x = 20 :=
by sorry

end contractor_laborers_l152_152641


namespace find_x_l152_152189

-- define initial quantities of apples and oranges
def initial_apples (x : ℕ) : ℕ := 3 * x + 1
def initial_oranges (x : ℕ) : ℕ := 4 * x + 12

-- define the condition that the number of oranges is twice the number of apples
def condition (x : ℕ) : Prop := initial_oranges x = 2 * initial_apples x

-- define the final state
def final_apples : ℕ := 1
def final_oranges : ℕ := 12

-- theorem to prove that the number of times is 5
theorem find_x : ∃ x : ℕ, condition x ∧ final_apples = 1 ∧ final_oranges = 12 :=
by
  use 5
  sorry

end find_x_l152_152189


namespace range_of_a_l152_152990

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ Real.exp 1) := by
  sorry

end range_of_a_l152_152990


namespace fraction_equality_l152_152991

theorem fraction_equality (a b : ℚ) (h₁ : a = 1/2) (h₂ : b = 2/3) : 
    (6 * a + 18 * b) / (12 * a + 6 * b) = 3 / 2 := by
  sorry

end fraction_equality_l152_152991


namespace ratio_correct_l152_152657

-- Definitions based on the problem conditions
def initial_cards_before_eating (X : ℤ) : ℤ := X
def cards_bought_new : ℤ := 4
def cards_left_after_eating : ℤ := 34

-- Definition of the number of cards eaten by the dog
def cards_eaten_by_dog (X : ℤ) : ℤ := X + cards_bought_new - cards_left_after_eating

-- Definition of the ratio of the number of cards eaten to the total number of cards before being eaten
def ratio_cards_eaten_to_total (X : ℤ) : ℚ := (cards_eaten_by_dog X : ℚ) / (X + cards_bought_new : ℚ)

-- Statement to prove
theorem ratio_correct (X : ℤ) : ratio_cards_eaten_to_total X = (X - 30) / (X + 4) := by
  sorry

end ratio_correct_l152_152657


namespace sample_size_stratified_sampling_l152_152119

theorem sample_size_stratified_sampling (n : ℕ) 
  (total_employees : ℕ) 
  (middle_aged_employees : ℕ) 
  (middle_aged_sample : ℕ)
  (stratified_sampling : n * middle_aged_employees = middle_aged_sample * total_employees)
  (total_employees_pos : total_employees = 750)
  (middle_aged_employees_pos : middle_aged_employees = 250) :
  n = 15 := 
by
  rw [total_employees_pos, middle_aged_employees_pos] at stratified_sampling
  sorry

end sample_size_stratified_sampling_l152_152119


namespace volume_of_convex_solid_l152_152714

variables {m V t6 T t3 : ℝ} 

-- Definition of the distance between the two parallel planes
def distance_between_planes (m : ℝ) : Prop := m > 0

-- Areas of the two parallel faces
def area_hexagon_face (t6 : ℝ) : Prop := t6 > 0
def area_triangle_face (t3 : ℝ) : Prop := t3 > 0

-- Area of the cross-section of the solid with a plane perpendicular to the height at its midpoint
def area_cross_section (T : ℝ) : Prop := T > 0

-- Volume of the convex solid
def volume_formula_holds (V m t6 T t3 : ℝ) : Prop :=
  V = (m / 6) * (t6 + 4 * T + t3)

-- Formal statement of the problem
theorem volume_of_convex_solid
  (m t6 t3 T V : ℝ)
  (h₁ : distance_between_planes m)
  (h₂ : area_hexagon_face t6)
  (h₃ : area_triangle_face t3)
  (h₄ : area_cross_section T) :
  volume_formula_holds V m t6 T t3 :=
by
  sorry

end volume_of_convex_solid_l152_152714


namespace sector_angle_l152_152617

theorem sector_angle (r l θ : ℝ) (h : 2 * r + l = π * r) : θ = π - 2 :=
sorry

end sector_angle_l152_152617


namespace journey_time_l152_152505

theorem journey_time
  (speed1 speed2 : ℝ)
  (distance total_time : ℝ)
  (h1 : speed1 = 40)
  (h2 : speed2 = 60)
  (h3 : distance = 240)
  (h4 : total_time = 5) :
  ∃ (t1 t2 : ℝ), (t1 + t2 = total_time) ∧ (speed1 * t1 + speed2 * t2 = distance) ∧ (t1 = 3) := 
by
  use (3 : ℝ), (2 : ℝ)
  simp [h1, h2, h3, h4]
  norm_num
  -- Additional steps to finish the proof would go here, but are omitted as per the requirements
  -- sorry

end journey_time_l152_152505


namespace problem_statement_l152_152471

/-
Definitions of the given conditions:
- Circle P: (x-1)^2 + y^2 = 8, center C.
- Point M(-1,0).
- Line y = kx + m intersects trajectory at points A and B.
- k_{OA} \cdot k_{OB} = -1/2.
-/

noncomputable def Circle_P : Set (ℝ × ℝ) :=
  { p | (p.1 - 1)^2 + p.2^2 = 8 }

def Point_M : (ℝ × ℝ) := (-1, 0)

def Trajectory_C : Set (ℝ × ℝ) :=
  { p | p.1^2 / 2 + p.2^2 = 1 }

def Line_kx_m (k m : ℝ) : Set (ℝ × ℝ) :=
  { p | p.2 = k * p.1 + m }

def k_OA_OB (k_OA k_OB : ℝ) : Prop :=
  k_OA * k_OB = -1/2

/-
Mathematical equivalence proof problem:
- Prove the trajectory of center C is an ellipse with equation x^2/2 + y^2 = 1.
- Prove that if line y=kx+m intersects with the trajectory, the area of the triangle AOB is a fixed value.
-/

theorem problem_statement (k m : ℝ)
    (h_intersects : ∃ A B : ℝ × ℝ, A ∈ (Trajectory_C ∩ Line_kx_m k m) ∧ B ∈ (Trajectory_C ∩ Line_kx_m k m))
    (k_OA k_OB : ℝ) (h_k_OA_k_OB : k_OA_OB k_OA k_OB) :
  ∃ (C_center_trajectory : Trajectory_C),
  ∃ (area_AOB : ℝ), area_AOB = (3 * Real.sqrt 2) / 2 :=
sorry

end problem_statement_l152_152471


namespace proof_problem_l152_152022

-- Define the operation
def star (a b : ℝ) : ℝ := (a - b) ^ 2

-- The proof problem as a Lean statement
theorem proof_problem (x y : ℝ) : star ((x - y) ^ 2 + 1) ((y - x) ^ 2 + 1) = 0 := by
  sorry

end proof_problem_l152_152022


namespace allan_balloons_l152_152756

def jak_balloons : ℕ := 11
def diff_balloons : ℕ := 6

theorem allan_balloons (jake_allan_diff : jak_balloons = diff_balloons + 5) : jak_balloons - diff_balloons = 5 :=
by
  sorry

end allan_balloons_l152_152756


namespace last_two_digits_of_large_exponent_l152_152350

theorem last_two_digits_of_large_exponent :
  (9 ^ (8 ^ (7 ^ (6 ^ (5 ^ (4 ^ (3 ^ 2))))))) % 100 = 21 :=
by
  sorry

end last_two_digits_of_large_exponent_l152_152350


namespace A_and_C_together_2_hours_l152_152011

theorem A_and_C_together_2_hours (A_rate B_rate C_rate : ℝ) (hA : A_rate = 1 / 5)
  (hBC : B_rate + C_rate = 1 / 3) (hB : B_rate = 1 / 30) : A_rate + C_rate = 1 / 2 := 
by
  sorry

end A_and_C_together_2_hours_l152_152011


namespace inequality_proof_l152_152778

theorem inequality_proof (x y : ℝ) (hx : x > -1) (hy : y > -1) (hxy : x + y = 1) : 
    (x / (y + 1) + y / (x + 1) ≥ 2 / 3) := 
  sorry

end inequality_proof_l152_152778


namespace annual_average_growth_rate_l152_152628

theorem annual_average_growth_rate (x : ℝ) (h : x > 0): 
  100 * (1 + x)^2 = 169 :=
sorry

end annual_average_growth_rate_l152_152628


namespace g_13_equals_236_l152_152087

def g (n : ℕ) : ℕ := n^2 + 2 * n + 41

theorem g_13_equals_236 : g 13 = 236 := sorry

end g_13_equals_236_l152_152087


namespace min_value_of_function_l152_152630

noncomputable def y (x : ℝ) : ℝ := (Real.cos x) * (Real.sin (2 * x))

theorem min_value_of_function :
  ∃ x ∈ Set.Icc (-Real.pi) Real.pi, y x = -4 * Real.sqrt 3 / 9 :=
sorry

end min_value_of_function_l152_152630


namespace sally_out_of_pocket_l152_152954

-- Definitions based on conditions
def g : ℕ := 320 -- Amount given by the school
def c : ℕ := 12  -- Cost per book
def n : ℕ := 30  -- Number of students

-- Definition derived from conditions
def total_cost : ℕ := n * c
def out_of_pocket : ℕ := total_cost - g

-- Proof statement
theorem sally_out_of_pocket : out_of_pocket = 40 := by
  -- The proof steps would go here
  sorry

end sally_out_of_pocket_l152_152954


namespace brian_cards_after_waine_takes_l152_152574

-- Define the conditions
def brian_initial_cards : ℕ := 76
def wayne_takes_away : ℕ := 59

-- Define the expected result
def brian_remaining_cards : ℕ := 17

-- The statement of the proof problem
theorem brian_cards_after_waine_takes : brian_initial_cards - wayne_takes_away = brian_remaining_cards := 
by 
-- the proof would be provided here 
sorry

end brian_cards_after_waine_takes_l152_152574


namespace mango_selling_price_l152_152863

theorem mango_selling_price
  (CP SP_loss SP_profit : ℝ)
  (h1 : SP_loss = 0.8 * CP)
  (h2 : SP_profit = 1.05 * CP)
  (h3 : SP_profit = 6.5625) :
  SP_loss = 5.00 :=
by
  sorry

end mango_selling_price_l152_152863


namespace smallest_counterexample_is_14_l152_152988

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_not_prime (n : ℕ) : Prop := ¬Prime n
def smallest_counterexample (n : ℕ) : Prop :=
  is_even n ∧ is_not_prime n ∧ is_not_prime (n + 2) ∧ ∀ m, is_even m ∧ is_not_prime m ∧ is_not_prime (m + 2) → n ≤ m

theorem smallest_counterexample_is_14 : smallest_counterexample 14 :=
by
  sorry

end smallest_counterexample_is_14_l152_152988


namespace find_q_l152_152197

theorem find_q (p q : ℝ) (h : (-2)^3 - 2*(-2)^2 + p*(-2) + q = 0) : 
  q = 16 + 2 * p :=
sorry

end find_q_l152_152197


namespace sufficient_but_not_necessary_l152_152979

theorem sufficient_but_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2 → x^2 + y^2 ≥ 4) ∧ ∃ x y : ℝ, x^2 + y^2 ≥ 4 ∧ ¬(x ≥ 2 ∧ y ≥ 2) :=
by
  sorry

end sufficient_but_not_necessary_l152_152979


namespace func_symmetry_monotonicity_range_of_m_l152_152415

open Real

theorem func_symmetry_monotonicity (f : ℝ → ℝ)
  (h1 : ∀ x, f (3 + x) = f (1 - x))
  (h2 : ∀ x1 x2, 2 < x1 → 2 < x2 → (f x1 - f x2) / (x1 - x2) > 0) :
  (∀ x, f (2 + x) = f (2 - x)) ∧
  (∀ x1 x2, (x1 > 2 ∧ x2 > 2 → f x1 < f x2 → x1 < x2) ∧
            (x2 > 2 ∧ x1 > x2 → f x2 < f x1 → x2 < x1)) := 
sorry

theorem range_of_m (f : ℝ → ℝ)
  (h : ∀ θ : ℝ, f (cos θ ^ 2 + 2 * (m : ℝ) ^ 2 + 2) < f (sin θ + m ^ 2 - 3 * m - 2)) :
  ∀ m, (3 - sqrt 42) / 6 < m ∧ m < (3 + sqrt 42) / 6 :=
sorry

end func_symmetry_monotonicity_range_of_m_l152_152415


namespace quadratic_roots_bounds_l152_152484

theorem quadratic_roots_bounds (a b c : ℤ) (p1 p2 : ℝ) (h_a_pos : a > 0) 
  (h_int_coeff : ∀ x : ℤ, x = a ∨ x = b ∨ x = c) 
  (h_distinct_roots : p1 ≠ p2) 
  (h_roots : a * p1^2 + b * p1 + c = 0 ∧ a * p2^2 + b * p2 + c = 0) 
  (h_roots_bounds : 0 < p1 ∧ p1 < 1 ∧ 0 < p2 ∧ p2 < 1) : 
     a ≥ 5 := 
sorry

end quadratic_roots_bounds_l152_152484


namespace simplify_fraction_l152_152966

theorem simplify_fraction (a b : ℝ) (h : a ≠ b): 
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) :=
by
  sorry

end simplify_fraction_l152_152966


namespace value_of_m_if_f_is_power_function_l152_152546

theorem value_of_m_if_f_is_power_function (m : ℤ) :
  (2 * m + 3 = 1) → m = -1 :=
by
  sorry

end value_of_m_if_f_is_power_function_l152_152546


namespace simplify_expr_l152_152250

-- Define variables and conditions
variables (x y a b c : ℝ)

-- State the theorem
theorem simplify_expr : 
  (2 - y) * 24 * (x - y + 2 * (a - 2 - 3 * c) * a - 2 * b + c) = 
  2 + 4 * b^2 - a * b - c^2 :=
sorry

end simplify_expr_l152_152250


namespace solve_for_x_l152_152237

theorem solve_for_x (x : ℚ) (h : 5 * x + 9 * x = 420 - 10 * (x - 4)) : 
  x = 115 / 6 :=
by
  sorry

end solve_for_x_l152_152237


namespace time_after_2023_hours_l152_152851

theorem time_after_2023_hours (current_time : ℕ) (hours_later : ℕ) (modulus : ℕ) : 
    (current_time = 3) → 
    (hours_later = 2023) → 
    (modulus = 12) → 
    ((current_time + (hours_later % modulus)) % modulus = 2) :=
by 
    intros h1 h2 h3
    rw [h1, h2, h3]
    sorry

end time_after_2023_hours_l152_152851


namespace isosceles_triangle_perimeter_l152_152276

-- Definitions and conditions
-- Define the lengths of the three sides of the triangle
def a : ℕ := 3
def b : ℕ := 8

-- Define that the triangle is isosceles
def is_isosceles_triangle := 
  (a = a) ∨ (b = b) ∨ (a = b)

-- Perimeter of the triangle
def perimeter (x y z : ℕ) := x + y + z

-- The theorem we need to prove
theorem isosceles_triangle_perimeter : is_isosceles_triangle → (a + b + b = 19) :=
by
  intro h
  sorry

end isosceles_triangle_perimeter_l152_152276


namespace participation_increase_closest_to_10_l152_152892

def percentage_increase (old new : ℕ) : ℚ := ((new - old) / old) * 100

theorem participation_increase_closest_to_10 :
  (percentage_increase 80 88 = 10) ∧ 
  (percentage_increase 90 99 = 10) := by
  sorry

end participation_increase_closest_to_10_l152_152892


namespace budget_circle_salaries_degrees_l152_152071

theorem budget_circle_salaries_degrees :
  let transportation := 20
  let research_development := 9
  let utilities := 5
  let equipment := 4
  let supplies := 2
  let total_percent := 100
  let full_circle_degrees := 360
  let total_allocated_percent := transportation + research_development + utilities + equipment + supplies
  let salaries_percent := total_percent - total_allocated_percent
  let salaries_degrees := (salaries_percent * full_circle_degrees) / total_percent
  salaries_degrees = 216 :=
by
  sorry

end budget_circle_salaries_degrees_l152_152071


namespace gwen_did_not_recycle_2_bags_l152_152882

def points_per_bag : ℕ := 8
def total_bags : ℕ := 4
def points_earned : ℕ := 16

theorem gwen_did_not_recycle_2_bags : total_bags - points_earned / points_per_bag = 2 := by
  sorry

end gwen_did_not_recycle_2_bags_l152_152882


namespace major_premise_incorrect_l152_152886

theorem major_premise_incorrect (a b : ℝ) (h : a > b) : ¬ (a^2 > b^2) :=
by {
  sorry
}

end major_premise_incorrect_l152_152886


namespace reduced_price_per_kg_l152_152514

-- Assume the constants in the conditions
variables (P R : ℝ)
variables (h1 : R = P - 0.40 * P) -- R = 0.60P
variables (h2 : 2000 / P + 10 = 2000 / R) -- extra 10 kg for the same 2000 rs

-- State the target we want to prove
theorem reduced_price_per_kg : R = 80 :=
by
  -- The steps and details of the proof
  sorry

end reduced_price_per_kg_l152_152514


namespace problem_1_problem_2_l152_152357

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * a * x^2 + 2 * x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x - g x a
noncomputable def h' (x : ℝ) (a : ℝ) : ℝ := (1 / x) - a * x - 2
noncomputable def G (x : ℝ) : ℝ := ((1 / x) - 1) ^ 2 - 1

theorem problem_1 (a : ℝ): 
  (∃ x : ℝ, 0 < x ∧ h' x a < 0) ↔ a > -1 :=
by sorry

theorem problem_2 (a : ℝ):
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → h' x a ≤ 0) ↔ a ≥ -(7 / 16) :=
by sorry

end problem_1_problem_2_l152_152357


namespace total_games_played_is_53_l152_152576

theorem total_games_played_is_53 :
  ∃ (ken_wins dave_wins jerry_wins larry_wins total_ties total_games_played : ℕ),
  jerry_wins = 7 ∧
  dave_wins = jerry_wins + 3 ∧
  ken_wins = dave_wins + 5 ∧
  larry_wins = 2 * jerry_wins ∧
  5 ≤ ken_wins ∧ 5 ≤ dave_wins ∧ 5 ≤ jerry_wins ∧ 5 ≤ larry_wins ∧
  total_ties = jerry_wins ∧
  total_games_played = ken_wins + dave_wins + jerry_wins + larry_wins + total_ties ∧
  total_games_played = 53 :=
by
  sorry

end total_games_played_is_53_l152_152576


namespace rectangle_enclosed_by_lines_l152_152698

theorem rectangle_enclosed_by_lines : 
  ∃ (ways : ℕ), 
  (ways = (Nat.choose 5 2) * (Nat.choose 4 2)) ∧ 
  ways = 60 := 
by
  sorry

end rectangle_enclosed_by_lines_l152_152698


namespace sqrt_expression_eval_l152_152881

theorem sqrt_expression_eval :
  (Real.sqrt 48 / Real.sqrt 3) - (Real.sqrt (1 / 6) * Real.sqrt 12) + Real.sqrt 24 = 4 - Real.sqrt 2 + 2 * Real.sqrt 6 :=
by
  sorry

end sqrt_expression_eval_l152_152881


namespace number_of_classes_l152_152925

theorem number_of_classes (x : ℕ) (h : x * (x - 1) = 20) : x = 5 :=
by
  sorry

end number_of_classes_l152_152925


namespace monotonic_range_of_a_l152_152509

theorem monotonic_range_of_a (a : ℝ) :
  (a ≥ 9 ∨ a ≤ 3) → 
  ∀ x y : ℝ, (1 ≤ x ∧ x ≤ 4) → (1 ≤ y ∧ y ≤ 4) → x ≤ y → 
  (x^2 + (1-a)*x + 3) ≤ (y^2 + (1-a)*y + 3) :=
by
  intro ha x y hx hy hxy
  sorry

end monotonic_range_of_a_l152_152509


namespace even_function_has_specific_a_l152_152658

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x ^ 2 + (2 * a ^ 2 - a) * x + 1

-- State the proof problem
theorem even_function_has_specific_a (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 1 / 2 :=
by
  intros h
  sorry

end even_function_has_specific_a_l152_152658


namespace square_field_area_l152_152870

theorem square_field_area (s : ℕ) (area cost_per_meter total_cost gate_width : ℕ):
  area = s^2 →
  cost_per_meter = 2 →
  total_cost = 1332 →
  gate_width = 1 →
  (4 * s - 2 * gate_width) * cost_per_meter = total_cost →
  area = 27889 :=
by
  intros h_area h_cost_per_meter h_total_cost h_gate_width h_equation
  sorry

end square_field_area_l152_152870


namespace quadratic_polynomial_solution_is_zero_l152_152604

-- Definitions based on given conditions
variables (a b c r s : ℝ)
variables (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
variables (h2 : a ≠ b ∧ a ≠ c ∧ b ≠ c)
variables (h3 : r + s = -b / a)
variables (h4 : r * s = c / a)

-- Proposition matching the equivalent proof problem
theorem quadratic_polynomial_solution_is_zero :
  ¬ ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  (∃ r s : ℝ, (r + s = -b / a) ∧ (r * s = c / a) ∧ (c = r * s ∨ b = r * s ∨ a = r * s) ∧
  (a = r ∨ a = s)) :=
sorry

end quadratic_polynomial_solution_is_zero_l152_152604


namespace sum_of_integers_square_greater_272_l152_152666

theorem sum_of_integers_square_greater_272 (x : ℤ) (h : x^2 = x + 272) :
  ∃ (roots : List ℤ), (roots = [17, -16]) ∧ (roots.sum = 1) :=
sorry

end sum_of_integers_square_greater_272_l152_152666


namespace koala_fiber_l152_152779

theorem koala_fiber (absorption_percent: ℝ) (absorbed_fiber: ℝ) (total_fiber: ℝ) 
  (h1: absorption_percent = 0.25) 
  (h2: absorbed_fiber = 10.5) 
  (h3: absorbed_fiber = absorption_percent * total_fiber) : 
  total_fiber = 42 :=
by
  rw [h1, h2] at h3
  have h : 10.5 = 0.25 * total_fiber := h3
  sorry

end koala_fiber_l152_152779


namespace pennies_to_quarters_ratio_l152_152725

-- Define the given conditions as assumptions
variables (pennies dimes nickels quarters: ℕ)

-- Given conditions
axiom cond1 : dimes = pennies + 10
axiom cond2 : nickels = 2 * dimes
axiom cond3 : quarters = 4
axiom cond4 : nickels = 100

-- Theorem stating the final result should be a certain ratio
theorem pennies_to_quarters_ratio (hpn : pennies = 40) : pennies / quarters = 10 := 
by sorry

end pennies_to_quarters_ratio_l152_152725


namespace probability_even_toys_l152_152602

theorem probability_even_toys:
  let total_toys := 21
  let even_toys := 10
  let probability_first_even := (even_toys : ℚ) / total_toys
  let probability_second_even := (even_toys - 1 : ℚ) / (total_toys - 1)
  let probability_both_even := probability_first_even * probability_second_even
  probability_both_even = 3 / 14 :=
by
  sorry

end probability_even_toys_l152_152602


namespace largest_pies_without_ingredients_l152_152624

variable (total_pies : ℕ) (chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
variable (b : total_pies = 36)
variable (c : chocolate_pies = total_pies / 2)
variable (m : marshmallow_pies = 2 * total_pies / 3)
variable (k : cayenne_pies = 3 * total_pies / 4)
variable (s : soy_nut_pies = total_pies / 6)

theorem largest_pies_without_ingredients (total_pies chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
  (b : total_pies = 36)
  (c : chocolate_pies = total_pies / 2)
  (m : marshmallow_pies = 2 * total_pies / 3)
  (k : cayenne_pies = 3 * total_pies / 4)
  (s : soy_nut_pies = total_pies / 6) :
  9 = total_pies - chocolate_pies - marshmallow_pies - cayenne_pies - soy_nut_pies + 3 * cayenne_pies := 
by
  sorry

end largest_pies_without_ingredients_l152_152624


namespace maximize_tables_eqn_l152_152556

theorem maximize_tables_eqn :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 12 → 400 * x = 20 * (12 - x) * 4 :=
by
  sorry

end maximize_tables_eqn_l152_152556


namespace stuffed_animal_ratio_l152_152544

theorem stuffed_animal_ratio
  (K : ℕ)
  (h1 : 34 + K + (K + 5) = 175) :
  K / 34 = 2 :=
by sorry

end stuffed_animal_ratio_l152_152544


namespace evaluate_expression_l152_152126

def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 7

theorem evaluate_expression : 3 * g 2 + 4 * g (-2) = 113 := by
  sorry

end evaluate_expression_l152_152126


namespace sin_270_eq_neg_one_l152_152758

theorem sin_270_eq_neg_one : Real.sin (270 * Real.pi / 180) = -1 := 
by
  sorry

end sin_270_eq_neg_one_l152_152758


namespace intersection_of_A_and_B_l152_152416

def A : Set ℝ := { x | 0 < x ∧ x < 2 }
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_of_A_and_B_l152_152416


namespace problem_l152_152422

variable {f : ℝ → ℝ}

-- Condition: f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Condition: f is monotonically decreasing on (0, +∞)
def monotone_decreasing_on_pos (f : ℝ → ℝ) : Prop := 
  ∀ ⦃x y : ℝ⦄, 0 < x → 0 < y → x < y → f y < f x

theorem problem (h_even : even_function f) (h_mon_dec : monotone_decreasing_on_pos f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end problem_l152_152422


namespace algebraic_expression_value_l152_152145

-- Define the conditions given
variables {a b : ℝ}
axiom h1 : a ≠ b
axiom h2 : a^2 - 8 * a + 5 = 0
axiom h3 : b^2 - 8 * b + 5 = 0

-- Main theorem to prove the expression equals -20
theorem algebraic_expression_value:
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 :=
sorry

end algebraic_expression_value_l152_152145


namespace complementary_angle_l152_152179

theorem complementary_angle (angle_deg : ℕ) (angle_min : ℕ) 
  (h1 : angle_deg = 37) (h2 : angle_min = 38) : 
  exists (comp_deg : ℕ) (comp_min : ℕ), comp_deg = 52 ∧ comp_min = 22 :=
by
  sorry

end complementary_angle_l152_152179


namespace sequence_sum_l152_152373

theorem sequence_sum (S : ℕ → ℕ) (a : ℕ → ℕ) : 
  (∀ n, S n = 2^n) →
  (a 1 = S 1) ∧ (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  a 3 + a 4 = 12 :=
by
  sorry

end sequence_sum_l152_152373


namespace smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd_l152_152339

theorem smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ 
             (n % 9 = 0) ∧ 
             (∃ d1 d2 d3 d4 : ℕ, 
               d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n ∧ 
               d1 % 2 = 1 ∧ 
               d2 % 2 = 0 ∧ 
               d3 % 2 = 0 ∧ 
               d4 % 2 = 0) ∧ 
             (∀ m : ℕ, (1000 ≤ m ∧ m < 10000 ∧ m % 9 = 0 ∧ 
               ∃ e1 e2 e3 e4 : ℕ, 
                 e1 * 1000 + e2 * 100 + e3 * 10 + e4 = m ∧ 
                 e1 % 2 = 1 ∧ 
                 e2 % 2 = 0 ∧ 
                 e3 % 2 = 0 ∧ 
                 e4 % 2 = 0) → n ≤ m) ∧ 
             n = 1026 :=
sorry

end smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd_l152_152339


namespace measure_of_side_XY_l152_152456

theorem measure_of_side_XY 
  (a b c : ℝ) 
  (Area : ℝ)
  (h1 : a = 30)
  (h2 : b = 60)
  (h3 : c = 90)
  (h4 : a + b + c = 180)
  (h_area : Area = 36)
  : (∀ (XY YZ XZ : ℝ), XY = 4.56) :=
by
  sorry

end measure_of_side_XY_l152_152456


namespace sum_not_prime_if_product_equality_l152_152400

theorem sum_not_prime_if_product_equality 
  (a b c d : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a * b = c * d) : ¬Nat.Prime (a + b + c + d) := 
by
  sorry

end sum_not_prime_if_product_equality_l152_152400


namespace inequality_holds_l152_152231

theorem inequality_holds (m : ℝ) (h : 0 ≤ m ∧ m < 12) :
  ∀ x : ℝ, 3 * m * x ^ 2 + m * x + 1 > 0 :=
sorry

end inequality_holds_l152_152231


namespace smallest_value_l152_152520

theorem smallest_value (a b c : ℕ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
    (h1 : a = 2 * b) (h2 : b = 2 * c) (h3 : 4 * c = a) :
    (Int.floor ((a + b : ℚ) / c) + Int.floor ((b + c : ℚ) / a) + Int.floor ((c + a : ℚ) / b)) = 8 := 
sorry

end smallest_value_l152_152520


namespace hari_contribution_correct_l152_152138

-- Translate the conditions into definitions
def praveen_investment : ℝ := 3360
def praveen_duration : ℝ := 12
def hari_duration : ℝ := 7
def profit_ratio_praveen : ℝ := 2
def profit_ratio_hari : ℝ := 3

-- The target Hari's contribution that we need to prove
def hari_contribution : ℝ := 2160

-- Problem statement: prove Hari's contribution given the conditions
theorem hari_contribution_correct :
  (praveen_investment * praveen_duration) / (hari_contribution * hari_duration) = profit_ratio_praveen / profit_ratio_hari :=
by {
  -- The statement is set up to prove equality of the ratios as given in the problem
  sorry
}

end hari_contribution_correct_l152_152138


namespace kareem_has_largest_final_number_l152_152651

def jose_final : ℕ := (15 - 2) * 4 + 5
def thuy_final : ℕ := (15 * 3 - 3) - 4
def kareem_final : ℕ := ((20 - 3) + 4) * 3

theorem kareem_has_largest_final_number :
  kareem_final > jose_final ∧ kareem_final > thuy_final := 
by 
  sorry

end kareem_has_largest_final_number_l152_152651


namespace not_make_all_numbers_equal_l152_152064

theorem not_make_all_numbers_equal (n : ℕ) (h : n ≥ 3)
  (a : Fin n → ℕ) (h1 : ∃ (i : Fin n), a i = 1 ∧ (∀ (j : Fin n), j ≠ i → a j = 0)) :
  ¬ ∃ x, ∀ i : Fin n, a i = x :=
by
  sorry

end not_make_all_numbers_equal_l152_152064


namespace smallest_number_divide_perfect_cube_l152_152504

theorem smallest_number_divide_perfect_cube (n : ℕ):
  n = 450 → (∃ m : ℕ, n * m = k ∧ ∃ k : ℕ, k ^ 3 = n * m) ∧ (∀ m₂ : ℕ, (n * m₂ = l ∧ ∃ l : ℕ, l ^ 3 = n * m₂) → m ≤ m₂) → m = 60 :=
by
  sorry

end smallest_number_divide_perfect_cube_l152_152504


namespace find_x_l152_152154

def hash (a b : ℕ) : ℕ := a * b - b + b^2

theorem find_x (x : ℕ) : hash x 7 = 63 → x = 3 :=
by
  sorry

end find_x_l152_152154


namespace min_value_expression_l152_152581

/-- Prove that for integers a, b, c satisfying 1 ≤ a ≤ b ≤ c ≤ 5, the minimum value of the expression 
  (a - 2)^2 + ((b + 1) / a - 1)^2 + ((c + 1) / b - 1)^2 + (5 / c - 1)^2 is 1.2595. -/
theorem min_value_expression (a b c : ℤ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  ∃ (min_val : ℝ), min_val = ((a - 2)^2 + ((b + 1) / a - 1)^2 + ((c + 1) / b - 1)^2 + (5 / c - 1)^2) ∧ min_val = 1.2595 :=
by
  sorry

end min_value_expression_l152_152581


namespace time_solution_l152_152128

-- Define the condition as a hypothesis
theorem time_solution (x : ℝ) (h : x / 4 + (24 - x) / 2 = x) : x = 9.6 :=
by
  -- Proof skipped
  sorry

end time_solution_l152_152128


namespace sequence_sum_l152_152222

theorem sequence_sum {A B C D E F G H I J : ℤ} (hD : D = 8)
    (h_sum1 : A + B + C + D = 45)
    (h_sum2 : B + C + D + E = 45)
    (h_sum3 : C + D + E + F = 45)
    (h_sum4 : D + E + F + G = 45)
    (h_sum5 : E + F + G + H = 45)
    (h_sum6 : F + G + H + I = 45)
    (h_sum7 : G + H + I + J = 45)
    (h_sum8 : H + I + J + A = 45)
    (h_sum9 : I + J + A + B = 45)
    (h_sum10 : J + A + B + C = 45) :
  A + J = 0 := 
sorry

end sequence_sum_l152_152222


namespace borrowed_nickels_l152_152291

def n_original : ℕ := 87
def n_left : ℕ := 12
def n_borrowed : ℕ := n_original - n_left

theorem borrowed_nickels : n_borrowed = 75 := by
  sorry

end borrowed_nickels_l152_152291


namespace petya_wrong_l152_152082

theorem petya_wrong : ∃ (a b : ℕ), b^2 ∣ a^5 ∧ ¬ (b ∣ a^2) :=
by
  use 4
  use 32
  sorry

end petya_wrong_l152_152082


namespace present_age_of_B_l152_152236

theorem present_age_of_B (A B : ℕ) (h1 : A + 20 = 2 * (B - 20)) (h2 : A = B + 10) : B = 70 :=
by
  sorry

end present_age_of_B_l152_152236


namespace find_num_alligators_l152_152665

-- We define the conditions as given in the problem
def journey_to_delta_hours : ℕ := 4
def extra_hours : ℕ := 2
def combined_time_alligators_walked : ℕ := 46

-- We define the hypothesis in terms of Lean variables
def num_alligators_traveled_with_Paul (A : ℕ) : Prop :=
  (journey_to_delta_hours + (journey_to_delta_hours + extra_hours) * A) = combined_time_alligators_walked

-- Now the theorem statement where we prove that the number of alligators (A) is 7
theorem find_num_alligators :
  ∃ A : ℕ, num_alligators_traveled_with_Paul A ∧ A = 7 :=
by
  existsi 7
  unfold num_alligators_traveled_with_Paul
  simp
  sorry -- this is where the actual proof would go

end find_num_alligators_l152_152665


namespace collinear_points_sum_l152_152948

theorem collinear_points_sum (x y : ℝ) : 
  (∃ a b : ℝ, a * x + b * 3 + (1 - a - b) * 2 = a * x + b * y + (1 - a - b) * y ∧ 
               a * y + b * 4 + (1 - a - b) * y = a * x + b * y + (1 - a - b) * x) → 
  x = 2 → y = 4 → x + y = 6 :=
by sorry

end collinear_points_sum_l152_152948


namespace plates_not_adj_l152_152054

def num_ways_arrange_plates (blue red green orange : ℕ) (no_adj : Bool) : ℕ :=
  -- assuming this function calculates the desired number of arrangements
  sorry

theorem plates_not_adj (h : num_ways_arrange_plates 6 2 2 1 true = 1568) : 
  num_ways_arrange_plates 6 2 2 1 true = 1568 :=
  by exact h -- using the hypothesis directly for the theorem statement

end plates_not_adj_l152_152054


namespace sum_of_monomials_is_monomial_l152_152830

variable (a b : ℕ)

theorem sum_of_monomials_is_monomial (m n : ℕ) (h : ∃ k : ℕ, 2 * a^m * b^n + a * b^3 = k * a^1 * b^3) :
  m = 1 ∧ n = 3 :=
sorry

end sum_of_monomials_is_monomial_l152_152830


namespace min_value_expression_l152_152268

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 64 :=
by
  sorry

end min_value_expression_l152_152268


namespace game_result_2013_game_result_2014_l152_152153

inductive Player
| Barbara
| Jenna

def winning_player (n : ℕ) : Option Player :=
  if n % 5 = 3 then some Player.Jenna
  else if n % 5 = 4 then some Player.Barbara
  else none

theorem game_result_2013 : winning_player 2013 = some Player.Jenna := 
by sorry

theorem game_result_2014 : (winning_player 2014 = some Player.Barbara) ∨ (winning_player 2014 = some Player.Jenna) :=
by sorry

end game_result_2013_game_result_2014_l152_152153


namespace arithmetic_sequence_n_l152_152636

theorem arithmetic_sequence_n {a : ℕ → ℕ} (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) :
  (∃ n : ℕ, a n = 2005) → (∃ n : ℕ, n = 669) :=
by
  sorry

end arithmetic_sequence_n_l152_152636


namespace solve_remainder_problem_l152_152173

def remainder_problem : Prop :=
  ∃ (n : ℕ), 
    (n % 481 = 179) ∧ 
    (n % 752 = 231) ∧ 
    (n % 1063 = 359) ∧ 
    (((179 + 231 - 359) % 37) = 14)

theorem solve_remainder_problem : remainder_problem :=
by
  sorry

end solve_remainder_problem_l152_152173


namespace total_teachers_l152_152831

theorem total_teachers (total_individuals sample_size sampled_students : ℕ)
  (H1 : total_individuals = 2400)
  (H2 : sample_size = 160)
  (H3 : sampled_students = 150) :
  ∃ total_teachers, total_teachers * (sample_size / (sample_size - sampled_students)) = 2400 / (sample_size / (sample_size - sampled_students)) ∧ total_teachers = 150 := 
  sorry

end total_teachers_l152_152831


namespace four_digit_number_divisible_by_36_l152_152219

theorem four_digit_number_divisible_by_36 (n : ℕ) (h₁ : ∃ k : ℕ, 6130 + n = 36 * k) 
  (h₂ : ∃ k : ℕ, 130 + n = 4 * k) 
  (h₃ : ∃ k : ℕ, (10 + n) = 9 * k) : n = 6 :=
sorry

end four_digit_number_divisible_by_36_l152_152219


namespace ap_80th_term_l152_152812

/--
If the sum of the first 20 terms of an arithmetic progression is 200,
and the sum of the first 60 terms is 180, then the 80th term is -573/40.
-/
theorem ap_80th_term (S : ℤ → ℚ) (a d : ℚ)
  (h1 : S 20 = 200)
  (h2 : S 60 = 180)
  (hS : ∀ n, S n = n / 2 * (2 * a + (n - 1) * d)) :
  a + 79 * d = -573 / 40 :=
by {
  sorry
}

end ap_80th_term_l152_152812


namespace range_of_m_l152_152466

open Real

-- Defining conditions as propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 1 ≠ 0
def q (m : ℝ) : Prop := m > 1
def p_or_q (m : ℝ) : Prop := p m ∨ q m
def p_and_q (m : ℝ) : Prop := p m ∧ q m

-- Mathematically equivalent proof problem
theorem range_of_m (m : ℝ) (H1 : p_or_q m) (H2 : ¬p_and_q m) : -2 < m ∧ m ≤ 1 ∨ 2 ≤ m :=
by
  sorry

end range_of_m_l152_152466


namespace value_of_p_l152_152166

noncomputable def third_term (x y : ℝ) := 45 * x^8 * y^2
noncomputable def fourth_term (x y : ℝ) := 120 * x^7 * y^3

theorem value_of_p (p q : ℝ) (h1 : third_term p q = fourth_term p q) (h2 : p + 2 * q = 1) (h3 : 0 < p) (h4 : 0 < q) : p = 4 / 7 :=
by
  have h : third_term p q = 45 * p^8 * q^2 := rfl
  have h' : fourth_term p q = 120 * p^7 * q^3 := rfl
  rw [h, h'] at h1
  sorry

end value_of_p_l152_152166


namespace sum_of_cubes_l152_152900

-- Definitions
noncomputable def p : ℂ := sorry
noncomputable def q : ℂ := sorry
noncomputable def r : ℂ := sorry

-- Roots conditions
axiom h_root_p : p^3 - 2 * p^2 + 3 * p - 4 = 0
axiom h_root_q : q^3 - 2 * q^2 + 3 * q - 4 = 0
axiom h_root_r : r^3 - 2 * r^2 + 3 * r - 4 = 0

-- Vieta's conditions
axiom h_sum : p + q + r = 2
axiom h_product_pairs : p * q + q * r + r * p = 3
axiom h_product : p * q * r = 4

-- Goal
theorem sum_of_cubes : p^3 + q^3 + r^3 = 2 :=
  sorry

end sum_of_cubes_l152_152900


namespace intersection_of_sets_l152_152626

def setA : Set ℝ := {x | (x^2 - x - 2 < 0)}
def setB : Set ℝ := {y | ∃ x ≤ 0, y = 3^x}

theorem intersection_of_sets : (setA ∩ setB) = {z | 0 < z ∧ z ≤ 1} :=
sorry

end intersection_of_sets_l152_152626


namespace solve_equation_l152_152158

theorem solve_equation (x : ℝ) (h : x ≠ 2) : -x^2 = (4 * x + 2) / (x - 2) ↔ x = -2 :=
by sorry

end solve_equation_l152_152158


namespace simplify_expression_l152_152213

variable {R : Type*} [CommRing R] (x y : R)

theorem simplify_expression :
  (x - 2 * y) * (x + 2 * y) - x * (x - y) = -4 * y ^ 2 + x * y :=
by
  sorry

end simplify_expression_l152_152213


namespace find_m_l152_152078

-- Definitions for the system of equations and the condition
def system_of_equations (x y m : ℝ) :=
  2 * x + 6 * y = 25 ∧ 6 * x + 2 * y = -11 ∧ x - y = m - 1

-- Statement to prove
theorem find_m (x y m : ℝ) (h : system_of_equations x y m) : m = -8 :=
  sorry

end find_m_l152_152078


namespace length_of_CD_l152_152653

theorem length_of_CD (x y u v : ℝ) (R S C D : ℝ → ℝ)
  (h1 : 5 * x = 3 * y)
  (h2 : 7 * u = 4 * v)
  (h3 : u = x + 3)
  (h4 : v = y - 3)
  (h5 : C x + D y = 1) : 
  x + y = 264 :=
by
  sorry

end length_of_CD_l152_152653


namespace max_value_x_plus_2y_l152_152526

theorem max_value_x_plus_2y (x y : ℝ) (h : |x| + |y| ≤ 1) : x + 2 * y ≤ 2 :=
sorry

end max_value_x_plus_2y_l152_152526


namespace infinite_geometric_sum_example_l152_152503

noncomputable def infinite_geometric_sum (a₁ q : ℝ) : ℝ :=
a₁ / (1 - q)

theorem infinite_geometric_sum_example :
  infinite_geometric_sum 18 (-1/2) = 12 := by
  sorry

end infinite_geometric_sum_example_l152_152503


namespace martha_bottles_l152_152987

def total_bottles_left (a b c d : ℕ) : ℕ :=
  a + b + c - d

theorem martha_bottles : total_bottles_left 4 4 5 3 = 10 :=
by
  sorry

end martha_bottles_l152_152987


namespace disproving_rearranged_sum_l152_152463

noncomputable section

open scoped BigOperators

variable {a : ℕ → ℝ} {f : ℕ → ℕ}

-- Conditions
def summable_a (a : ℕ → ℝ) : Prop :=
  ∑' i, a i = 1

def strictly_decreasing_abs (a : ℕ → ℝ) : Prop :=
  ∀ n m, n < m → abs (a n) > abs (a m)

def bijection (f : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m, f m = n

def limit_condition (a : ℕ → ℝ) (f : ℕ → ℕ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs ((f n : ℤ) - (n : ℤ)) * abs (a n) < ε

-- Statement
theorem disproving_rearranged_sum :
  summable_a a ∧
  strictly_decreasing_abs a ∧
  bijection f ∧
  limit_condition a f →
  ∑' i, a (f i) ≠ 1 :=
sorry

end disproving_rearranged_sum_l152_152463


namespace max_min_P_l152_152337

theorem max_min_P (a b c : ℝ) (h : |a + b| + |b + c| + |c + a| = 8) :
  (a^2 + b^2 + c^2 = 48) ∨ (a^2 + b^2 + c^2 = 16 / 3) :=
sorry

end max_min_P_l152_152337


namespace simplify_expression_l152_152382

theorem simplify_expression (y : ℝ) : (3 * y^4)^4 = 81 * y^16 :=
by
  sorry

end simplify_expression_l152_152382


namespace jamesOreos_count_l152_152664

noncomputable def jamesOreos (jordanOreos : ℕ) : ℕ := 4 * jordanOreos + 7

theorem jamesOreos_count (J : ℕ) (h1 : J + jamesOreos J = 52) : jamesOreos J = 43 :=
by
  sorry

end jamesOreos_count_l152_152664


namespace find_the_number_l152_152238

theorem find_the_number :
  ∃ x : ℕ, 72519 * x = 724827405 ∧ x = 10005 :=
by
  sorry

end find_the_number_l152_152238


namespace geometric_sequence_a3_l152_152015

theorem geometric_sequence_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 1 = 1)
  (h5 : a 5 = 4)
  (geo_seq : ∀ n, a n = a 1 * r ^ (n - 1)) :
  a 3 = 2 :=
by
  sorry

end geometric_sequence_a3_l152_152015


namespace problem1_domain_valid_problem2_domain_valid_l152_152464

-- Definition of the domains as sets.

def domain1 (x : ℝ) : Prop := ∃ k : ℤ, x = 2 * k * Real.pi

def domain2 (x : ℝ) : Prop := (-3 ≤ x ∧ x < -Real.pi / 2) ∨ (0 < x ∧ x < Real.pi / 2)

-- Theorem statements for the domains.

theorem problem1_domain_valid (x : ℝ) : (∀ y : ℝ, y = Real.log (Real.cos x) → y ≥ 0) ↔ domain1 x := sorry

theorem problem2_domain_valid (x : ℝ) : 
  (∀ y : ℝ, y = Real.log (Real.sin (2 * x)) + Real.sqrt (9 - x ^ 2) → y ∈ Set.Icc (-3) 3) ↔ domain2 x := sorry

end problem1_domain_valid_problem2_domain_valid_l152_152464


namespace original_cost_of_dolls_l152_152695

theorem original_cost_of_dolls 
  (x : ℝ) -- original cost of each Russian doll
  (savings : ℝ) -- total savings of Daniel
  (h1 : savings = 15 * x) -- Daniel saves enough to buy 15 dolls at original price
  (h2 : savings = 20 * 3) -- with discounted price, he can buy 20 dolls
  : x = 4 :=
by
  sorry

end original_cost_of_dolls_l152_152695


namespace correct_factorization_l152_152234

theorem correct_factorization :
  ∀ (x : ℝ), -x^2 + 2*x - 1 = - (x - 1)^2 :=
by
  intro x
  sorry

end correct_factorization_l152_152234


namespace banana_cream_pie_correct_slice_l152_152840

def total_students := 45
def strawberry_pie_preference := 15
def pecan_pie_preference := 10
def pumpkin_pie_preference := 9

noncomputable def banana_cream_pie_slice_degrees : ℝ :=
  let remaining_students := total_students - strawberry_pie_preference - pecan_pie_preference - pumpkin_pie_preference
  let students_per_preference := remaining_students / 2
  (students_per_preference / total_students) * 360

theorem banana_cream_pie_correct_slice :
  banana_cream_pie_slice_degrees = 44 := by
  sorry

end banana_cream_pie_correct_slice_l152_152840


namespace combination_sum_l152_152076

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Given conditions
axiom combinatorial_identity (n r : ℕ) : combination n r + combination n (r + 1) = combination (n + 1) (r + 1)

-- The theorem we aim to prove
theorem combination_sum : combination 8 2 + combination 8 3 + combination 9 2 = 120 := 
by
  sorry

end combination_sum_l152_152076


namespace nth_equation_l152_152611

theorem nth_equation (n : ℕ) : 
  1 + 6 * n = (3 * n + 1) ^ 2 - 9 * n ^ 2 := 
by 
  sorry

end nth_equation_l152_152611


namespace Nina_total_problems_l152_152330

def Ruby_math_problems := 12
def Ruby_reading_problems := 4
def Ruby_science_problems := 5

def Nina_math_problems := 5 * Ruby_math_problems
def Nina_reading_problems := 9 * Ruby_reading_problems
def Nina_science_problems := 3 * Ruby_science_problems

def total_problems := Nina_math_problems + Nina_reading_problems + Nina_science_problems

theorem Nina_total_problems : total_problems = 111 :=
by
  sorry

end Nina_total_problems_l152_152330


namespace number_of_apartment_complexes_l152_152640

theorem number_of_apartment_complexes (width_land length_land side_complex : ℕ)
    (h_width : width_land = 262) (h_length : length_land = 185) 
    (h_side : side_complex = 18) :
    width_land / side_complex * length_land / side_complex = 140 := by
  -- given conditions
  rw [h_width, h_length, h_side]
  -- apply calculation steps for clarity (not necessary for final theorem)
  -- calculate number of complexes along width
  have h1 : 262 / 18 = 14 := sorry
  -- calculate number of complexes along length
  have h2 : 185 / 18 = 10 := sorry
  -- final product calculation
  sorry

end number_of_apartment_complexes_l152_152640


namespace triangle_third_side_l152_152897

theorem triangle_third_side (DE DF : ℝ) (E F : ℝ) (EF : ℝ) 
    (h₁ : DE = 7) 
    (h₂ : DF = 21) 
    (h₃ : E = 3 * F) : EF = 14 * Real.sqrt 2 :=
sorry

end triangle_third_side_l152_152897


namespace product_of_solutions_l152_152744

theorem product_of_solutions (x : ℝ) (hx : |x - 5| - 5 = 0) :
  ∃ a b : ℝ, (|a - 5| - 5 = 0 ∧ |b - 5| - 5 = 0) ∧ a * b = 0 := by
  sorry

end product_of_solutions_l152_152744


namespace carla_order_cost_l152_152177

theorem carla_order_cost (base_cost : ℝ) (coupon : ℝ) (senior_discount_rate : ℝ)
  (additional_charge : ℝ) (tax_rate : ℝ) (conversion_rate : ℝ) :
  base_cost = 7.50 →
  coupon = 2.50 →
  senior_discount_rate = 0.20 →
  additional_charge = 1.00 →
  tax_rate = 0.08 →
  conversion_rate = 0.85 →
  (2 * (base_cost - coupon) * (1 - senior_discount_rate) + additional_charge) * (1 + tax_rate) * conversion_rate = 4.59 :=
by
  sorry

end carla_order_cost_l152_152177


namespace solution_form_l152_152819

noncomputable def required_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) ≤ (x * f y + y * f x) / 2

theorem solution_form (f : ℝ → ℝ) (h : ∀ x : ℝ, 0 < x → 0 < f x) : required_function f → ∃ a : ℝ, 0 < a ∧ ∀ x : ℝ, 0 < x → f x = a * x :=
by
  intros
  sorry

end solution_form_l152_152819


namespace bakery_new_cakes_count_l152_152368

def cakes_sold := 91
def more_cakes_bought := 63

theorem bakery_new_cakes_count : (91 + 63) = 154 :=
by
  sorry

end bakery_new_cakes_count_l152_152368


namespace matrix_addition_is_correct_l152_152839

-- Definitions of matrices A and B according to given conditions
def A : Matrix (Fin 4) (Fin 4) ℤ :=  
  ![![ 3,  0,  1,  4],
    ![ 1,  2,  0,  0],
    ![ 5, -3,  2,  1],
    ![ 0,  0, -1,  3]]

def B : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![-5, -7,  3,  2],
    ![ 4, -9,  5, -2],
    ![ 8,  2, -3,  0],
    ![ 1,  1, -2, -4]]

-- The expected result matrix from the addition of A and B
def C : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![-2, -7,  4,  6],
    ![ 5, -7,  5, -2],
    ![13, -1, -1,  1],
    ![ 1,  1, -3, -1]]

-- The statement that A + B equals C
theorem matrix_addition_is_correct : A + B = C :=
by 
  -- Here we would provide the proof steps.
  sorry

end matrix_addition_is_correct_l152_152839


namespace Homer_first_try_points_l152_152297

variable (x : ℕ)
variable (h1 : x + (x - 70) + 2 * (x - 70) = 1390)

theorem Homer_first_try_points : x = 400 := by
  sorry

end Homer_first_try_points_l152_152297


namespace area_of_circle_l152_152667

theorem area_of_circle (x y : ℝ) :
  x^2 + y^2 + 8 * x + 10 * y = -9 → 
  ∃ a : ℝ, a = 32 * Real.pi :=
by
  sorry

end area_of_circle_l152_152667


namespace total_whipped_cream_l152_152331

theorem total_whipped_cream (cream_from_farm : ℕ) (cream_to_buy : ℕ) (total_cream : ℕ) 
  (h1 : cream_from_farm = 149) 
  (h2 : cream_to_buy = 151) 
  (h3 : total_cream = cream_from_farm + cream_to_buy) : 
  total_cream = 300 :=
sorry

end total_whipped_cream_l152_152331


namespace jorge_spent_amount_l152_152499

theorem jorge_spent_amount
  (num_tickets : ℕ)
  (price_per_ticket : ℕ)
  (discount_percentage : ℚ)
  (h1 : num_tickets = 24)
  (h2 : price_per_ticket = 7)
  (h3 : discount_percentage = 0.5) :
  num_tickets * price_per_ticket * (1 - discount_percentage) = 84 := 
by
  simp [h1, h2, h3]
  sorry

end jorge_spent_amount_l152_152499


namespace B_fraction_l152_152370

theorem B_fraction (A_s B_s C_s : ℕ) (h1 : A_s = 600) (h2 : A_s = (2 / 5) * (B_s + C_s))
  (h3 : A_s + B_s + C_s = 1800) :
  B_s / (A_s + C_s) = 1 / 6 :=
by
  sorry

end B_fraction_l152_152370


namespace average_monthly_balance_l152_152468

def january_balance : ℕ := 100
def february_balance : ℕ := 200
def march_balance : ℕ := 150
def april_balance : ℕ := 150
def may_balance : ℕ := 180
def number_of_months : ℕ := 5
def total_balance : ℕ := january_balance + february_balance + march_balance + april_balance + may_balance

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance + may_balance) / number_of_months = 156 := by
  sorry

end average_monthly_balance_l152_152468


namespace min_mn_sum_l152_152409

theorem min_mn_sum :
  ∃ (m n : ℕ), n > m ∧ m ≥ 1 ∧ 
  (1978^n % 1000 = 1978^m % 1000) ∧ (m + n = 106) :=
sorry

end min_mn_sum_l152_152409


namespace min_value_l152_152094

variables (a b c : ℝ)
variable (hpos : a > 0 ∧ b > 0 ∧ c > 0)
variable (hsum : a + b + c = 1)

theorem min_value (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) :
  9 * a^2 + 4 * b^2 + (1/4) * c^2 = 36 / 157 := 
sorry

end min_value_l152_152094


namespace gcd_lcm_identity_l152_152844

variables {n m k : ℕ}

/-- Given positive integers n, m, and k such that n divides lcm(m, k) 
    and m divides lcm(n, k), we prove that n * gcd(m, k) = m * gcd(n, k). -/
theorem gcd_lcm_identity (n_pos : 0 < n) (m_pos : 0 < m) (k_pos : 0 < k) 
  (h1 : n ∣ Nat.lcm m k) (h2 : m ∣ Nat.lcm n k) :
  n * Nat.gcd m k = m * Nat.gcd n k :=
sorry

end gcd_lcm_identity_l152_152844


namespace probability_five_cards_one_from_each_suit_and_extra_l152_152746

/--
Given five cards chosen with replacement from a standard 52-card deck, 
the probability of having exactly one card from each suit, plus one 
additional card from any suit, is 3/32.
-/
theorem probability_five_cards_one_from_each_suit_and_extra 
  (cards : ℕ) (total_suits : ℕ)
  (prob_first_diff_suit : ℚ) 
  (prob_second_diff_suit : ℚ) 
  (prob_third_diff_suit : ℚ) 
  (prob_fourth_diff_suit : ℚ) 
  (prob_any_suit : ℚ) 
  (total_prob : ℚ) :
  cards = 5 ∧ total_suits = 4 ∧ 
  prob_first_diff_suit = 3 / 4 ∧ 
  prob_second_diff_suit = 1 / 2 ∧ 
  prob_third_diff_suit = 1 / 4 ∧ 
  prob_fourth_diff_suit = 1 ∧ 
  prob_any_suit = 1 →
  total_prob = 3 / 32 :=
by {
  sorry
}

end probability_five_cards_one_from_each_suit_and_extra_l152_152746


namespace solve_inequality_l152_152096

theorem solve_inequality (x : ℝ) :
  (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 6) ↔ (2 < x) ∧ (x < 3) := by
  sorry

end solve_inequality_l152_152096


namespace find_initial_population_l152_152671

-- Define the conditions that the population increases annually by 20%
-- and that the population after 2 years is 14400.
def initial_population (P : ℝ) : Prop :=
  1.44 * P = 14400

-- The theorem states that given the conditions, the initial population is 10000.
theorem find_initial_population (P : ℝ) (h : initial_population P) : P = 10000 :=
  sorry

end find_initial_population_l152_152671


namespace arrangement_books_l152_152755

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem arrangement_books : combination 9 4 = 126 := by
  sorry

end arrangement_books_l152_152755


namespace ellipse_tangency_construction_l152_152162

theorem ellipse_tangency_construction
  (a : ℝ)
  (e1 e2 : ℝ → Prop)  -- Representing the parallel lines as propositions
  (F1 F2 : ℝ × ℝ)  -- Foci represented as points in the plane
  (d : ℝ)  -- Distance between the parallel lines
  (angle_condition : ℝ)
  (conditions : 2 * a > d ∧ angle_condition = 1 / 3) : 
  ∃ O : ℝ × ℝ,  -- Midpoint O
    ∃ (T1 T1' T2 T2' : ℝ × ℝ),  -- Points of tangency
      (∃ E1 E2 : ℝ, e1 E1 ∧ e2 E2) ∧  -- Intersection points on the lines
      (F1.1 * (T1.1 - F1.1) + F1.2 * (T1.2 - F1.2)) / 
      (F2.1 * (T2.1 - F2.1) + F2.2 * (T2.2 - F2.2)) = 1 / 3 :=
sorry

end ellipse_tangency_construction_l152_152162


namespace sqrt6_special_op_l152_152976

-- Define the binary operation (¤) as given in the problem.
def special_op (x y : ℝ) : ℝ := (x + y) ^ 2 - (x - y) ^ 2

-- States that √6 ¤ √6 is equal to 24.
theorem sqrt6_special_op : special_op (Real.sqrt 6) (Real.sqrt 6) = 24 :=
by
  sorry

end sqrt6_special_op_l152_152976


namespace machine_fill_time_l152_152846

theorem machine_fill_time (filled_cans : ℕ) (time_per_batch : ℕ) (total_cans : ℕ) (expected_time : ℕ)
  (h1 : filled_cans = 150)
  (h2 : time_per_batch = 8)
  (h3 : total_cans = 675)
  (h4 : expected_time = 36) :
  (total_cans / filled_cans) * time_per_batch = expected_time :=
by 
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end machine_fill_time_l152_152846


namespace calculate_expression_l152_152575

theorem calculate_expression : 
  (3^2 - 2 * 3) - (5^2 - 2 * 5) + (7^2 - 2 * 7) = 23 := 
by sorry

end calculate_expression_l152_152575


namespace apples_left_l152_152973

theorem apples_left (initial_apples : ℕ) (difference_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 46) 
  (h2 : difference_apples = 32) 
  (h3 : final_apples = initial_apples - difference_apples) : 
  final_apples = 14 := 
by
  rw [h1, h2] at h3
  exact h3

end apples_left_l152_152973


namespace fractional_part_of_students_who_walk_home_l152_152124

theorem fractional_part_of_students_who_walk_home 
  (students_by_bus : ℚ)
  (students_by_car : ℚ)
  (students_by_bike : ℚ)
  (students_by_skateboard : ℚ)
  (h_bus : students_by_bus = 1/3)
  (h_car : students_by_car = 1/5)
  (h_bike : students_by_bike = 1/8)
  (h_skateboard : students_by_skateboard = 1/15)
  : 1 - (students_by_bus + students_by_car + students_by_bike + students_by_skateboard) = 11/40 := 
by
  sorry

end fractional_part_of_students_who_walk_home_l152_152124


namespace find_c_k_l152_152635

noncomputable def common_difference (a : ℕ → ℕ) : ℕ := sorry
noncomputable def common_ratio (b : ℕ → ℕ) : ℕ := sorry
noncomputable def arith_seq (d : ℕ) (n : ℕ) : ℕ := 1 + (n - 1) * d
noncomputable def geom_seq (r : ℕ) (n : ℕ) : ℕ := r^(n - 1)
noncomputable def combined_seq (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ := a n + b n

variable (k : ℕ) (d : ℕ) (r : ℕ)

-- Conditions
axiom arith_condition : common_difference (arith_seq d) = d
axiom geom_condition : common_ratio (geom_seq r) = r
axiom combined_k_minus_1 : combined_seq (arith_seq d) (geom_seq r) (k - 1) = 50
axiom combined_k_plus_1 : combined_seq (arith_seq d) (geom_seq r) (k + 1) = 1500

-- Prove that c_k = 2406
theorem find_c_k : combined_seq (arith_seq d) (geom_seq r) k = 2406 := by
  sorry

end find_c_k_l152_152635


namespace polynomial_problem_l152_152510

noncomputable def F (x : ℝ) : ℝ := sorry

theorem polynomial_problem
  (F : ℝ → ℝ)
  (h1 : F 4 = 22)
  (h2 : ∀ x : ℝ, (F (2 * x) / F (x + 2) = 4 - (16 * x + 8) / (x^2 + x + 1))) :
  F 8 = 1078 / 9 := sorry

end polynomial_problem_l152_152510


namespace nth_equation_l152_152713

theorem nth_equation (n : ℕ) (hn: n ≥ 1) : 
  (n+1) / ((n+1)^2 - 1) - 1 / (n * (n+1) * (n+2)) = 1 / (n+1) :=
by
  sorry

end nth_equation_l152_152713


namespace pam_age_l152_152573

-- Given conditions:
-- 1. Pam is currently twice as young as Rena.
-- 2. In 10 years, Rena will be 5 years older than Pam.

variable (Pam Rena : ℕ)

theorem pam_age
  (h1 : 2 * Pam = Rena)
  (h2 : Rena + 10 = Pam + 15)
  : Pam = 5 := 
sorry

end pam_age_l152_152573


namespace directrix_parabola_l152_152105

theorem directrix_parabola (x y : ℝ) :
  (x^2 = (1/4 : ℝ) * y) → (y = -1/16) :=
sorry

end directrix_parabola_l152_152105


namespace slope_of_parallel_line_l152_152093

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l152_152093


namespace complement_union_eq_l152_152437

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l152_152437


namespace instantaneous_velocity_at_t3_l152_152305

open Real

noncomputable def displacement (t : ℝ) : ℝ := 4 - 2 * t + t ^ 2

theorem instantaneous_velocity_at_t3 : deriv displacement 3 = 4 := 
by
  sorry

end instantaneous_velocity_at_t3_l152_152305


namespace calum_disco_ball_budget_l152_152290

-- Defining the conditions
def n_d : ℕ := 4  -- Number of disco balls
def n_f : ℕ := 10  -- Number of food boxes
def p_f : ℕ := 25  -- Price per food box in dollars
def B : ℕ := 330  -- Total budget in dollars

-- Defining the expected result
def p_d : ℕ := 20  -- Cost per disco ball in dollars

-- Proof statement (no proof, just the statement)
theorem calum_disco_ball_budget :
  (10 * p_f + 4 * p_d = B) → (p_d = 20) :=
by
  sorry

end calum_disco_ball_budget_l152_152290


namespace curved_surface_area_cone_l152_152205

-- Define the necessary values
def r := 8  -- radius of the base of the cone in centimeters
def l := 18 -- slant height of the cone in centimeters

-- Prove the curved surface area of the cone
theorem curved_surface_area_cone :
  (π * r * l = 144 * π) :=
by sorry

end curved_surface_area_cone_l152_152205


namespace find_three_digit_number_l152_152926

/-- 
  Define the three-digit number abc and show that for some digit d in the range of 1 to 9,
  the conditions are satisfied.
-/
theorem find_three_digit_number
  (a b c d : ℕ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : 1 ≤ d ∧ d ≤ 9)
  (h_abc : 100 * a + 10 * b + c = 627)
  (h_bcd : 100 * b + 10 * c + d = 627 * a)
  (h_1a4d : 1040 + 100 * a + d = 627 * a)
  : 100 * a + 10 * b + c = 627 := 
sorry

end find_three_digit_number_l152_152926


namespace intersection_A_B_l152_152067

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x | x^2 < 10}
def intersection_of_A_and_B : Set ℕ := {0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = intersection_of_A_and_B :=
by
  sorry

end intersection_A_B_l152_152067


namespace cos_half_diff_proof_l152_152677

noncomputable def cos_half_diff (A B C : ℝ) (h_triangle : A + B + C = 180)
  (h_relation : A + C = 2 * B)
  (h_equation : (1 / Real.cos A) + (1 / Real.cos C) = - (Real.sqrt 2 / Real.cos B)) : Real :=
  Real.cos ((A - C) / 2)

theorem cos_half_diff_proof (A B C : ℝ)
  (h_triangle : A + B + C = 180)
  (h_relation : A + C = 2 * B)
  (h_equation : (1 / Real.cos A) + (1 / Real.cos C) = - (Real.sqrt 2 / Real.cos B)) :
  cos_half_diff A B C h_triangle h_relation h_equation = -Real.sqrt 2 / 2 :=
sorry

end cos_half_diff_proof_l152_152677


namespace factorize_problem1_factorize_problem2_l152_152023

-- Problem 1: Prove that 6p^3q - 10p^2 == 2p^2 * (3pq - 5)
theorem factorize_problem1 (p q : ℝ) : 
    6 * p^3 * q - 10 * p^2 = 2 * p^2 * (3 * p * q - 5) := 
by 
    sorry

-- Problem 2: Prove that a^4 - 8a^2 + 16 == (a-2)^2 * (a+2)^2
theorem factorize_problem2 (a : ℝ) : 
    a^4 - 8 * a^2 + 16 = (a - 2)^2 * (a + 2)^2 := 
by 
    sorry

end factorize_problem1_factorize_problem2_l152_152023


namespace find_value_of_expression_l152_152598

variable (a b c : ℝ)

def parabola_symmetry (a b c : ℝ) :=
  (36 * a + 6 * b + c = 2) ∧ 
  (25 * a + 5 * b + c = 6) ∧ 
  (49 * a + 7 * b + c = -4)

theorem find_value_of_expression :
  (∃ a b c : ℝ, parabola_symmetry a b c) →
  3 * a + 3 * c + b = -8 :=  sorry

end find_value_of_expression_l152_152598


namespace mark_gig_schedule_l152_152259

theorem mark_gig_schedule 
  (every_other_day : ∀ weeks, ∃ gigs, gigs = weeks * 7 / 2) 
  (songs_per_gig : 2 * 5 + 10 = 20) 
  (total_minutes : ∃ gigs, 280 = gigs * 20) : 
  ∃ weeks, weeks = 4 := 
by 
  sorry

end mark_gig_schedule_l152_152259


namespace all_n_eq_one_l152_152446

theorem all_n_eq_one (k : ℕ) (n : ℕ → ℕ)
  (h₁ : k ≥ 2)
  (h₂ : ∀ i, 1 ≤ i ∧ i < k → (n (i + 1)) ∣ 2^(n i) - 1)
  (h₃ : (n 1) ∣ 2^(n k) - 1) :
  ∀ i, 1 ≤ i ∧ i ≤ k → n i = 1 := 
sorry

end all_n_eq_one_l152_152446


namespace minimum_value_of_k_l152_152299

theorem minimum_value_of_k (x y : ℝ) (h : x * (x - 1) ≤ y * (1 - y)) : x^2 + y^2 ≤ 2 :=
sorry

end minimum_value_of_k_l152_152299


namespace sum_series_eq_260_l152_152075

theorem sum_series_eq_260 : (2 + 12 + 22 + 32 + 42) + (10 + 20 + 30 + 40 + 50) = 260 := by
  sorry

end sum_series_eq_260_l152_152075


namespace determine_exponent_l152_152084

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x ^ a

theorem determine_exponent (a : ℝ) (hf : power_function a 4 = 8) : power_function (3/2) = power_function a := by
  sorry

end determine_exponent_l152_152084


namespace initial_apples_l152_152587

-- Definitions based on the given conditions
def apples_given_away : ℕ := 88
def apples_left : ℕ := 39

-- Statement to prove
theorem initial_apples : apples_given_away + apples_left = 127 :=
by {
  -- Proof steps would go here
  sorry
}

end initial_apples_l152_152587


namespace find_positive_value_of_A_l152_152225

variable (A : ℝ)

-- Given conditions
def relation (A B : ℝ) : ℝ := A^2 + B^2

-- The proof statement
theorem find_positive_value_of_A (h : relation A 7 = 200) : A = Real.sqrt 151 := sorry

end find_positive_value_of_A_l152_152225


namespace area_triangle_COD_l152_152656

noncomputable def area_of_triangle (t s : ℝ) : ℝ := 
  1 / 2 * abs (5 + 2 * s + 7 * t)

theorem area_triangle_COD (t s : ℝ) : 
  ∃ (C : ℝ × ℝ) (D : ℝ × ℝ), 
    C = (3 + 5 * t, 2 + 4 * t) ∧ 
    D = (2 + 5 * s, 3 + 4 * s) ∧ 
    area_of_triangle t s = 1 / 2 * abs (5 + 2 * s + 7 * t) :=
by
  sorry

end area_triangle_COD_l152_152656


namespace sum_of_first_20_primes_l152_152288

theorem sum_of_first_20_primes :
  ( [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71].sum = 639 ) :=
by
  sorry

end sum_of_first_20_primes_l152_152288


namespace ginger_total_water_l152_152495

def hours_worked : Nat := 8
def cups_per_bottle : Nat := 2
def bottles_drank_per_hour : Nat := 1
def bottles_for_plants : Nat := 5

theorem ginger_total_water : 
  (hours_worked * cups_per_bottle * bottles_drank_per_hour) + (bottles_for_plants * cups_per_bottle) = 26 :=
by
  sorry

end ginger_total_water_l152_152495


namespace ab_range_l152_152843

theorem ab_range (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x, f x = |2 - x^2|)
  (h_a_lt_b : 0 < a ∧ a < b) (h_fa_eq_fb : f a = f b) :
  0 < a * b ∧ a * b < 2 := 
by
  sorry

end ab_range_l152_152843


namespace intersection_of_M_and_N_l152_152971

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}
def intersection_M_N : Set ℕ := {0, 1}

theorem intersection_of_M_and_N : M ∩ N = intersection_M_N := by
  sorry

end intersection_of_M_and_N_l152_152971


namespace box_weight_l152_152996

theorem box_weight (total_weight : ℕ) (number_of_boxes : ℕ) (box_weight : ℕ) 
  (h1 : total_weight = 267) 
  (h2 : number_of_boxes = 3) 
  (h3 : box_weight = total_weight / number_of_boxes) : 
  box_weight = 89 := 
by 
  sorry

end box_weight_l152_152996


namespace area_of_PDCE_l152_152394

/-- A theorem to prove the area of quadrilateral PDCE given conditions in triangle ABC. -/
theorem area_of_PDCE
  (ABC_area : ℝ)
  (BD_to_CD_ratio : ℝ)
  (E_is_midpoint : Prop)
  (AD_intersects_BE : Prop)
  (P : Prop)
  (area_PDCE : ℝ) :
  (ABC_area = 1) →
  (BD_to_CD_ratio = 2 / 1) →
  E_is_midpoint →
  AD_intersects_BE →
  ∃ P, P →
    area_PDCE = 7 / 30 :=
by sorry

end area_of_PDCE_l152_152394


namespace min_value_of_quadratic_expression_l152_152508

variable (x y z : ℝ)

theorem min_value_of_quadratic_expression 
  (h1 : 2 * x + 2 * y + z + 8 = 0) : 
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 = 9 :=
sorry

end min_value_of_quadratic_expression_l152_152508


namespace inverse_44_mod_53_l152_152063

theorem inverse_44_mod_53 : (44 * 22) % 53 = 1 :=
by
-- Given condition: 19's inverse modulo 53 is 31
have h: (19 * 31) % 53 = 1 := by sorry
-- We should prove the required statement using the given condition.
sorry

end inverse_44_mod_53_l152_152063


namespace find_k_l152_152475

theorem find_k (k : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-1, k)
  (a.1 * (2 * a.1 - b.1) + a.2 * (2 * a.2 - b.2)) = 0 → k = 12 := sorry

end find_k_l152_152475


namespace rect_solution_proof_l152_152980

noncomputable def rect_solution_exists : Prop :=
  ∃ (l2 w2 : ℝ), 2 * (l2 + w2) = 12 ∧ l2 * w2 = 4 ∧
               l2 = 3 + Real.sqrt 5 ∧ w2 = 3 - Real.sqrt 5

theorem rect_solution_proof : rect_solution_exists :=
  by
    sorry

end rect_solution_proof_l152_152980


namespace average_xyz_l152_152631

theorem average_xyz (x y z : ℝ) 
  (h1 : 2003 * z - 4006 * x = 1002) 
  (h2 : 2003 * y + 6009 * x = 4004) : (x + y + z) / 3 = 5 / 6 :=
by
  sorry

end average_xyz_l152_152631


namespace maximum_n_for_positive_sum_l152_152910

noncomputable def max_n_for_positive_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :=
  S n > 0

-- Definition of the arithmetic sequence properties
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d
  
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)

-- Given conditions
variable (h₁ : a 1 > 0)
variable (h₅ : a 2016 + a 2017 > 0)
variable (h₆ : a 2016 * a 2017 < 0)

-- Add the definition of the sum of the first n terms of the arithmetic sequence
noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * (a 0 + a n) / 2

-- Prove the final statement
theorem maximum_n_for_positive_sum : max_n_for_positive_sum a S 4032 :=
by
  -- conditions to use in the proof
  have h₁ : a 1 > 0 := sorry
  have h₅ : a 2016 + a 2017 > 0 := sorry
  have h₆ : a 2016 * a 2017 < 0 := sorry
  -- positively bounded sum
  let Sn := sum_of_first_n_terms a
  -- proof to utilize Lean's capabilities, replace with actual proof later
  sorry

end maximum_n_for_positive_sum_l152_152910


namespace modified_prism_surface_area_l152_152190

theorem modified_prism_surface_area :
  let original_surface_area := 2 * (2 * 4 + 2 * 5 + 4 * 5)
  let modified_surface_area := original_surface_area + 5
  modified_surface_area = original_surface_area + 5 :=
by
  -- set the original dimensions
  let l := 2
  let w := 4
  let h := 5
  -- calculate original surface area
  let SA_original := 2 * (l * w + l * h + w * h)
  -- calculate modified surface area
  let SA_new := SA_original + 5
  -- assert the relationship
  have : SA_new = SA_original + 5 := rfl
  exact this

end modified_prism_surface_area_l152_152190


namespace adult_tickets_count_l152_152033

theorem adult_tickets_count (A C : ℕ) (h1 : A + C = 7) (h2 : 21 * A + 14 * C = 119) : A = 3 :=
sorry

end adult_tickets_count_l152_152033


namespace total_carriages_l152_152708

theorem total_carriages (Euston Norfolk Norwich FlyingScotsman : ℕ) 
  (h1 : Euston = 130)
  (h2 : Norfolk = Euston - 20)
  (h3 : Norwich = 100)
  (h4 : FlyingScotsman = Norwich + 20) :
  Euston + Norfolk + Norwich + FlyingScotsman = 460 :=
by 
  sorry

end total_carriages_l152_152708


namespace non_union_employees_women_percent_l152_152540

-- Define the conditions
variables (total_employees men_percent women_percent unionized_percent unionized_men_percent : ℕ)
variables (total_men total_women total_unionized total_non_unionized unionized_men non_unionized_men non_unionized_women : ℕ)

axiom condition1 : men_percent = 52
axiom condition2 : unionized_percent = 60
axiom condition3 : unionized_men_percent = 70

axiom calc1 : total_employees = 100
axiom calc2 : total_men = total_employees * men_percent / 100
axiom calc3 : total_women = total_employees - total_men
axiom calc4 : total_unionized = total_employees * unionized_percent / 100
axiom calc5 : unionized_men = total_unionized * unionized_men_percent / 100
axiom calc6 : non_unionized_men = total_men - unionized_men
axiom calc7 : total_non_unionized = total_employees - total_unionized
axiom calc8 : non_unionized_women = total_non_unionized - non_unionized_men

-- Define the proof statement
theorem non_union_employees_women_percent : 
  (non_unionized_women / total_non_unionized) * 100 = 75 :=
by 
  sorry

end non_union_employees_women_percent_l152_152540


namespace robin_piano_highest_before_lowest_l152_152319

def probability_reach_highest_from_middle_C : ℚ :=
  let p_k (k : ℕ) (p_prev : ℚ) (p_next : ℚ) : ℚ := (1/2 : ℚ) * p_prev + (1/2 : ℚ) * p_next
  let p_1 := 0
  let p_88 := 1
  let A := -1/87
  let B := 1/87
  A + B * 40

theorem robin_piano_highest_before_lowest :
  probability_reach_highest_from_middle_C = 13 / 29 :=
by
  sorry

end robin_piano_highest_before_lowest_l152_152319


namespace sales_first_month_l152_152487

theorem sales_first_month (S1 S2 S3 S4 S5 S6 : ℝ) 
  (h2 : S2 = 7000) (h3 : S3 = 6800) (h4 : S4 = 7200) (h5 : S5 = 6500) (h6 : S6 = 5100)
  (avg : (S1 + S2 + S3 + S4 + S5 + S6) / 6 = 6500) : S1 = 6400 := by
  sorry

end sales_first_month_l152_152487


namespace train_passing_time_correct_l152_152841

-- Definitions of the conditions
def length_of_train : ℕ := 180  -- Length of the train in meters
def speed_of_train_km_hr : ℕ := 54  -- Speed of the train in kilometers per hour

-- Known conversion factors
def km_per_hour_to_m_per_sec (v : ℕ) : ℚ := (v * 1000) / 3600

-- Define the speed of the train in meters per second
def speed_of_train_m_per_sec : ℚ := km_per_hour_to_m_per_sec speed_of_train_km_hr

-- Define the time to pass the oak tree
def time_to_pass_oak_tree (d : ℕ) (v : ℚ) : ℚ := d / v

-- The statement to prove
theorem train_passing_time_correct :
  time_to_pass_oak_tree length_of_train speed_of_train_m_per_sec = 12 := 
by
  sorry

end train_passing_time_correct_l152_152841


namespace find_w_value_l152_152606

theorem find_w_value : 
  (2^5 * 9^2) / (8^2 * 243) = 0.16666666666666666 := 
by
  sorry

end find_w_value_l152_152606


namespace total_salaries_l152_152557

variable (A_salary B_salary : ℝ)

def A_saves : ℝ := 0.05 * A_salary
def B_saves : ℝ := 0.15 * B_salary

theorem total_salaries (h1 : A_salary = 5250) 
                       (h2 : A_saves = B_saves) : 
    A_salary + B_salary = 7000 := by
  sorry

end total_salaries_l152_152557


namespace find_x_squared_minus_y_squared_l152_152674

variable (x y : ℝ)

theorem find_x_squared_minus_y_squared 
(h1 : y + 6 = (x - 3)^2)
(h2 : x + 6 = (y - 3)^2)
(h3 : x ≠ y) :
x^2 - y^2 = 27 := by
  sorry

end find_x_squared_minus_y_squared_l152_152674


namespace polynomial_equality_l152_152951

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 5 * x + 7
noncomputable def g (x : ℝ) : ℝ := 12 * x^2 - 19 * x + 25

theorem polynomial_equality :
  f 3 = g 3 ∧ f (3 - Real.sqrt 3) = g (3 - Real.sqrt 3) ∧ f (3 + Real.sqrt 3) = g (3 + Real.sqrt 3) :=
by
  sorry

end polynomial_equality_l152_152951


namespace bin_expected_value_l152_152191

theorem bin_expected_value (m : ℕ) (h : (21 - 4 * m) / (7 + m) = 1) : m = 3 := 
by {
  sorry
}

end bin_expected_value_l152_152191


namespace train_length_is_180_l152_152837

noncomputable def train_length (time_seconds : ℕ) (speed_kmh : ℕ) : ℕ := 
  (speed_kmh * 5 / 18) * time_seconds

theorem train_length_is_180 : train_length 9 72 = 180 :=
by
  sorry

end train_length_is_180_l152_152837


namespace cosine_sum_formula_l152_152248

theorem cosine_sum_formula
  (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 4 / 5) 
  (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.cos (α + Real.pi / 4) = -Real.sqrt 2 / 10 := 
by 
  sorry

end cosine_sum_formula_l152_152248


namespace conference_handshakes_l152_152207

-- Define the number of attendees at the conference
def attendees : ℕ := 10

-- Define the number of ways to choose 2 people from the attendees
-- This is equivalent to the combination formula C(10, 2)
def handshakes (n : ℕ) : ℕ := n.choose 2

-- Prove that the number of handshakes at the conference is 45
theorem conference_handshakes : handshakes attendees = 45 := by
  sorry

end conference_handshakes_l152_152207


namespace least_n_l152_152246

theorem least_n (n : ℕ) (h : 0 < n ∧ (1 / n - 1 / (n + 1) < 1 / 15)) : n ≥ 4 :=
by sorry

end least_n_l152_152246


namespace arithmetic_expression_l152_152794

theorem arithmetic_expression :
  4 * 6 * 8 + 24 / 4 - 2^3 = 190 := by
  sorry

end arithmetic_expression_l152_152794


namespace a_is_minus_one_l152_152577

theorem a_is_minus_one (a : ℤ) (h1 : 2 * a + 1 < 0) (h2 : 2 + a > 0) : a = -1 := 
by
  sorry

end a_is_minus_one_l152_152577


namespace joanna_marbles_l152_152929

theorem joanna_marbles (m n : ℕ) (h1 : m * n = 720) (h2 : m > 1) (h3 : n > 1) :
  ∃ (count : ℕ), count = 28 :=
by
  -- Use the properties of divisors and conditions to show that there are 28 valid pairs (m, n).
  sorry

end joanna_marbles_l152_152929


namespace rectangle_ratio_l152_152306

theorem rectangle_ratio (A L : ℝ) (hA : A = 100) (hL : L = 20) :
  ∃ W : ℝ, A = L * W ∧ (L / W) = 4 :=
by
  sorry

end rectangle_ratio_l152_152306


namespace turtle_distance_in_six_minutes_l152_152645

theorem turtle_distance_in_six_minutes 
  (observers : ℕ)
  (time_interval : ℕ)
  (distance_seen : ℕ)
  (total_time : ℕ)
  (total_distance : ℕ)
  (observation_per_minute : ∀ t ≤ total_time, ∃ n : ℕ, n ≤ observers ∧ (∃ interval : ℕ, interval ≤ time_interval ∧ distance_seen = 1)) :
  total_distance = 10 :=
sorry

end turtle_distance_in_six_minutes_l152_152645


namespace necessary_condition_not_sufficient_condition_l152_152042

variable (a b : ℝ)
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0
def proposition_p (a : ℝ) : Prop := a = 0

theorem necessary_condition (a b : ℝ) (z : ℂ) (h : z = ⟨a, b⟩) : isPureImaginary z → proposition_p a := sorry

theorem not_sufficient_condition (a b : ℝ) (z : ℂ) (h : z = ⟨a, b⟩) : proposition_p a → ¬isPureImaginary z := sorry

end necessary_condition_not_sufficient_condition_l152_152042


namespace solve_quadratic_and_cubic_eqns_l152_152619

-- Define the conditions as predicates
def eq1 (x : ℝ) : Prop := (x - 1)^2 = 4
def eq2 (x : ℝ) : Prop := (x - 2)^3 = -125

-- State the theorem
theorem solve_quadratic_and_cubic_eqns : 
  (∃ x : ℝ, eq1 x ∧ (x = 3 ∨ x = -1)) ∧ (∃ x : ℝ, eq2 x ∧ x = -3) :=
by
  sorry

end solve_quadratic_and_cubic_eqns_l152_152619


namespace tan_sin_cos_identity_l152_152470

theorem tan_sin_cos_identity {x : ℝ} (htan : Real.tan x = 1 / 3) : Real.sin x * Real.cos x + 1 = 13 / 10 :=
by
  sorry

end tan_sin_cos_identity_l152_152470


namespace problem1_problem2_l152_152110

-- Problem 1 Statement
theorem problem1 (a : ℝ) (h : a ≠ 1) : (a^2 / (a - 1) - a - 1) = (1 / (a - 1)) :=
by
  sorry

-- Problem 2 Statement
theorem problem2 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ -y) : 
  (2 * x * y / (x^2 - y^2)) / ((1 / (x - y)) + (1 / (x + y))) = y :=
by
  sorry

end problem1_problem2_l152_152110


namespace nuts_in_tree_l152_152245

def num_squirrels := 4
def num_nuts := 2

theorem nuts_in_tree :
  ∀ (S N : ℕ), S = num_squirrels → S - N = 2 → N = num_nuts :=
by
  intros S N hS hDiff
  sorry

end nuts_in_tree_l152_152245


namespace smallest_denominator_fraction_l152_152005

theorem smallest_denominator_fraction 
  (p q : ℕ) (hp : 0 < p) (hq : 0 < q) 
  (h1 : 99 / 100 < p / q) 
  (h2 : p / q < 100 / 101) :
  p = 199 ∧ q = 201 := 
by 
  sorry

end smallest_denominator_fraction_l152_152005


namespace p_sufficient_not_necessary_for_q_l152_152660

def p (x1 x2 : ℝ) : Prop := x1 > 1 ∧ x2 > 1
def q (x1 x2 : ℝ) : Prop := x1 + x2 > 2 ∧ x1 * x2 > 1

theorem p_sufficient_not_necessary_for_q : 
  (∀ x1 x2 : ℝ, p x1 x2 → q x1 x2) ∧ ¬ (∀ x1 x2 : ℝ, q x1 x2 → p x1 x2) :=
by 
  sorry

end p_sufficient_not_necessary_for_q_l152_152660


namespace daps_equiv_dirps_l152_152354

noncomputable def dops_equiv_daps : ℝ := 5 / 4
noncomputable def dips_equiv_dops : ℝ := 3 / 10
noncomputable def dirps_equiv_dips : ℝ := 2

theorem daps_equiv_dirps (n : ℝ) : 20 = (dops_equiv_daps * dips_equiv_dops * dirps_equiv_dips) * n → n = 15 :=
by sorry

end daps_equiv_dirps_l152_152354


namespace complex_number_quadrant_l152_152824

noncomputable def complex_quadrant : ℂ → String
| z => if z.re > 0 ∧ z.im > 0 then "First quadrant"
      else if z.re < 0 ∧ z.im > 0 then "Second quadrant"
      else if z.re < 0 ∧ z.im < 0 then "Third quadrant"
      else if z.re > 0 ∧ z.im < 0 then "Fourth quadrant"
      else "On the axis"

theorem complex_number_quadrant (z : ℂ) (h : z = (5 : ℂ) / (2 + I)) : complex_quadrant z = "Fourth quadrant" :=
by
  sorry

end complex_number_quadrant_l152_152824


namespace selling_price_per_book_l152_152567

noncomputable def fixed_costs : ℝ := 35630
noncomputable def variable_cost_per_book : ℝ := 11.50
noncomputable def num_books : ℕ := 4072
noncomputable def total_production_costs : ℝ := fixed_costs + variable_cost_per_book * num_books

theorem selling_price_per_book :
  (total_production_costs / num_books : ℝ) = 20.25 := by
  sorry

end selling_price_per_book_l152_152567


namespace number_of_elements_l152_152610

theorem number_of_elements (n : ℕ) (S : ℕ) (sum_first_six : ℕ) (sum_last_six : ℕ) (sixth_number : ℕ)
    (h1 : S = 22 * n) 
    (h2 : sum_first_six = 6 * 19) 
    (h3 : sum_last_six = 6 * 27) 
    (h4 : sixth_number = 34) 
    (h5 : S = sum_first_six + sum_last_six - sixth_number) : 
    n = 11 := 
by
  sorry

end number_of_elements_l152_152610


namespace nearest_edge_of_picture_l152_152040

theorem nearest_edge_of_picture
    (wall_width : ℝ) (picture_width : ℝ) (offset : ℝ) (x : ℝ)
    (hw : wall_width = 25) (hp : picture_width = 5) (ho : offset = 2) :
    x + (picture_width / 2) + offset = wall_width / 2 →
    x = 8 :=
by
  intros h
  sorry

end nearest_edge_of_picture_l152_152040


namespace multiple_of_5_multiple_of_10_not_multiple_of_20_not_multiple_of_40_l152_152009

def x : ℤ := 50 + 100 + 140 + 180 + 320 + 400 + 5000

theorem multiple_of_5 : x % 5 = 0 := by 
  sorry

theorem multiple_of_10 : x % 10 = 0 := by 
  sorry

theorem not_multiple_of_20 : x % 20 ≠ 0 := by 
  sorry

theorem not_multiple_of_40 : x % 40 ≠ 0 := by 
  sorry

end multiple_of_5_multiple_of_10_not_multiple_of_20_not_multiple_of_40_l152_152009


namespace simplify_expression1_simplify_expression2_simplify_expression3_simplify_expression4_l152_152301

-- Problem 1
theorem simplify_expression1 (a b : ℝ) : 
  4 * a^3 + 2 * b - 2 * a^3 + b = 2 * a^3 + 3 * b := 
sorry

-- Problem 2
theorem simplify_expression2 (x : ℝ) : 
  2 * x^2 + 6 * x - 6 - (-2 * x^2 + 4 * x + 1) = 4 * x^2 + 2 * x - 7 := 
sorry

-- Problem 3
theorem simplify_expression3 (a b : ℝ) : 
  3 * (3 * a^2 - 2 * a * b) - 2 * (4 * a^2 - a * b) = a^2 - 4 * a * b := 
sorry

-- Problem 4
theorem simplify_expression4 (x y : ℝ) : 
  6 * x * y^2 - (2 * x - (1 / 2) * (2 * x - 4 * x * y^2) - x * y^2) = 5 * x * y^2 - x := 
sorry

end simplify_expression1_simplify_expression2_simplify_expression3_simplify_expression4_l152_152301


namespace prob_A_not_losing_is_correct_l152_152993

def prob_A_wins := 0.4
def prob_draw := 0.2
def prob_A_not_losing := 0.6

theorem prob_A_not_losing_is_correct : prob_A_wins + prob_draw = prob_A_not_losing :=
by sorry

end prob_A_not_losing_is_correct_l152_152993


namespace money_left_after_expenses_l152_152461

theorem money_left_after_expenses :
  let salary := 8123.08
  let food_expense := (1:ℝ) / 3 * salary
  let rent_expense := (1:ℝ) / 4 * salary
  let clothes_expense := (1:ℝ) / 5 * salary
  let total_expense := food_expense + rent_expense + clothes_expense
  let money_left := salary - total_expense
  money_left = 1759.00 :=
sorry

end money_left_after_expenses_l152_152461


namespace binom_28_7_l152_152050

theorem binom_28_7 (h1 : Nat.choose 26 3 = 2600) (h2 : Nat.choose 26 4 = 14950) (h3 : Nat.choose 26 5 = 65780) : 
  Nat.choose 28 7 = 197340 :=
by
  sorry

end binom_28_7_l152_152050


namespace arithmetic_identity_l152_152273

theorem arithmetic_identity : Real.sqrt 16 + ((1/2) ^ (-2:ℤ)) = 8 := 
by 
  sorry

end arithmetic_identity_l152_152273


namespace max_diff_units_digit_l152_152149

theorem max_diff_units_digit (n : ℕ) (h1 : n = 850 ∨ n = 855) : ∃ d, d = 5 :=
by 
  sorry

end max_diff_units_digit_l152_152149


namespace find_zebras_last_year_l152_152842

def zebras_last_year (current : ℕ) (born : ℕ) (died : ℕ) : ℕ :=
  current - born + died

theorem find_zebras_last_year :
  zebras_last_year 725 419 263 = 569 :=
by
  sorry

end find_zebras_last_year_l152_152842


namespace leak_empty_tank_time_l152_152615

theorem leak_empty_tank_time (fill_time_A : ℝ) (fill_time_A_with_leak : ℝ) (leak_empty_time : ℝ) :
  fill_time_A = 6 → fill_time_A_with_leak = 9 → leak_empty_time = 18 :=
by
  intros hA hL
  -- Here follows the proof we skip
  sorry

end leak_empty_tank_time_l152_152615


namespace exponentiation_addition_l152_152223

theorem exponentiation_addition : (3^3)^2 + 1 = 730 := by
  sorry

end exponentiation_addition_l152_152223


namespace distinct_integers_sum_l152_152125

theorem distinct_integers_sum (m n p q : ℕ) (h1 : m ≠ n) (h2 : m ≠ p) (h3 : m ≠ q) (h4 : n ≠ p)
  (h5 : n ≠ q) (h6 : p ≠ q) (h71 : m > 0) (h72 : n > 0) (h73 : p > 0) (h74 : q > 0)
  (h_eq : (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4) : m + n + p + q = 28 := by
  sorry

end distinct_integers_sum_l152_152125


namespace sum_of_remainders_eq_11_mod_13_l152_152922

theorem sum_of_remainders_eq_11_mod_13 
  (a b c d : ℤ)
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5) 
  (hc : c % 13 = 7) 
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_eq_11_mod_13_l152_152922


namespace value_of_r_when_n_is_2_l152_152102

-- Define the given conditions
def s : ℕ := 2 ^ 2 + 1
def r : ℤ := 3 ^ s - s

-- Prove that r equals 238 when n = 2
theorem value_of_r_when_n_is_2 : r = 238 := by
  sorry

end value_of_r_when_n_is_2_l152_152102


namespace green_apples_more_than_red_apples_l152_152014

theorem green_apples_more_than_red_apples 
    (total_apples : ℕ)
    (red_apples : ℕ)
    (total_apples_eq : total_apples = 44)
    (red_apples_eq : red_apples = 16) :
    (total_apples - red_apples) - red_apples = 12 :=
by
  sorry

end green_apples_more_than_red_apples_l152_152014


namespace expression_value_l152_152594

theorem expression_value (x y z w : ℝ) (h1 : 4 * x * z + y * w = 3) (h2 : x * w + y * z = 6) :
  (2 * x + y) * (2 * z + w) = 15 := 
sorry

end expression_value_l152_152594


namespace car_speed_on_local_roads_l152_152316

theorem car_speed_on_local_roads
    (v : ℝ) -- Speed of the car on local roads
    (h1 : v > 0) -- The speed is positive
    (h2 : 40 / v + 3 = 5) -- Given equation based on travel times and distances
    : v = 20 := 
sorry

end car_speed_on_local_roads_l152_152316


namespace arithmetic_mean_of_fractions_l152_152541

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 := 
by 
  sorry

end arithmetic_mean_of_fractions_l152_152541


namespace proof_problem_l152_152053

def a : ℕ := 5^2
def b : ℕ := a^4

theorem proof_problem : b = 390625 := 
by 
  sorry

end proof_problem_l152_152053


namespace tina_savings_l152_152705

theorem tina_savings :
  let june_savings : ℕ := 27
  let july_savings : ℕ := 14
  let august_savings : ℕ := 21
  let books_spending : ℕ := 5
  let shoes_spending : ℕ := 17
  let total_savings := june_savings + july_savings + august_savings
  let total_spending := books_spending + shoes_spending
  let remaining_money := total_savings - total_spending
  remaining_money = 40 :=
by
  sorry

end tina_savings_l152_152705


namespace garden_roller_length_l152_152155

/-- The length of a garden roller with diameter 1.4m,
covering 52.8m² in 6 revolutions, and using π = 22/7,
is 2 meters. -/
theorem garden_roller_length
  (diameter : ℝ)
  (total_area_covered : ℝ)
  (revolutions : ℕ)
  (approx_pi : ℝ)
  (circumference : ℝ := approx_pi * diameter)
  (area_per_revolution : ℝ := total_area_covered / (revolutions : ℝ))
  (length : ℝ := area_per_revolution / circumference) :
  diameter = 1.4 ∧ total_area_covered = 52.8 ∧ revolutions = 6 ∧ approx_pi = (22 / 7) → length = 2 :=
by
  sorry

end garden_roller_length_l152_152155


namespace height_percentage_increase_l152_152797

theorem height_percentage_increase (B A : ℝ) (h : A = B - 0.3 * B) : 
  ((B - A) / A) * 100 = 42.857 :=
by
  sorry

end height_percentage_increase_l152_152797


namespace is_not_innovative_54_l152_152144

def is_innovative (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < b ∧ b < a ∧ n = a^2 - b^2

theorem is_not_innovative_54 : ¬ is_innovative 54 :=
sorry

end is_not_innovative_54_l152_152144


namespace gcd_779_209_589_eq_19_l152_152359

theorem gcd_779_209_589_eq_19 : Nat.gcd 779 (Nat.gcd 209 589) = 19 := by
  sorry

end gcd_779_209_589_eq_19_l152_152359


namespace find_daily_wage_of_c_l152_152453

noncomputable def daily_wage_c (a b c : ℕ) (days_a days_b days_c total_earning : ℕ) : ℕ :=
  if 3 * b = 4 * a ∧ 3 * c = 5 * a ∧ 
    total_earning = 6 * a + 9 * b + 4 * c then c else 0

theorem find_daily_wage_of_c (a b c : ℕ)
  (days_a days_b days_c total_earning : ℕ)
  (h1 : days_a = 6)
  (h2 : days_b = 9)
  (h3 : days_c = 4)
  (h4 : 3 * b = 4 * a)
  (h5 : 3 * c = 5 * a)
  (h6 : total_earning = 1554)
  (h7 : total_earning = 6 * a + 9 * b + 4 * c) : 
  daily_wage_c a b c days_a days_b days_c total_earning = 105 := 
by sorry

end find_daily_wage_of_c_l152_152453


namespace values_of_x_l152_152352

def f (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x (x : ℝ) : f (f x) = f x → x = 0 ∨ x = -2 ∨ x = 5 ∨ x = 6 :=
by {
  sorry
}

end values_of_x_l152_152352


namespace math_bonanza_2016_8_l152_152650

def f (x : ℕ) := x^2 + x + 1

theorem math_bonanza_2016_8 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : f p = f q + 242) (hpq : p > q) :
  (p, q) = (61, 59) :=
by sorry

end math_bonanza_2016_8_l152_152650


namespace cassie_nail_cutting_l152_152450

structure AnimalCounts where
  dogs : ℕ
  dog_nails_per_foot : ℕ
  dog_feet : ℕ
  parrots : ℕ
  parrot_claws_per_leg : ℕ
  parrot_legs : ℕ
  extra_toe_parrots : ℕ
  extra_toe_claws : ℕ

def total_nails (counts : AnimalCounts) : ℕ :=
  counts.dogs * counts.dog_nails_per_foot * counts.dog_feet +
  counts.parrots * counts.parrot_claws_per_leg * counts.parrot_legs +
  counts.extra_toe_parrots * counts.extra_toe_claws

theorem cassie_nail_cutting :
  total_nails {
    dogs := 4,
    dog_nails_per_foot := 4,
    dog_feet := 4,
    parrots := 8,
    parrot_claws_per_leg := 3,
    parrot_legs := 2,
    extra_toe_parrots := 1,
    extra_toe_claws := 1
  } = 113 :=
by sorry

end cassie_nail_cutting_l152_152450


namespace sum_of_nonneg_numbers_ineq_l152_152397

theorem sum_of_nonneg_numbers_ineq
  (a b c d : ℝ)
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0)
  (h_sum : a + b + c + d = 4) :
  (a * b + c * d) * (a * c + b * d) * (a * d + b * c) ≤ 8 := sorry

end sum_of_nonneg_numbers_ineq_l152_152397


namespace range_xf_ge_0_l152_152643

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 2 else - (-x) - 2

theorem range_xf_ge_0 :
  { x : ℝ | x * f x ≥ 0 } = { x : ℝ | -2 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end range_xf_ge_0_l152_152643


namespace linear_equation_solution_l152_152348

theorem linear_equation_solution (x : ℝ) (h : 1 - x = -3) : x = 4 :=
by
  sorry

end linear_equation_solution_l152_152348


namespace find_pos_int_l152_152127

theorem find_pos_int (n p : ℕ) (h_prime : Nat.Prime p) (h_pos_n : 0 < n) (h_pos_p : 0 < p) : 
  n^8 - p^5 = n^2 + p^2 → (n = 2 ∧ p = 3) :=
by
  sorry

end find_pos_int_l152_152127


namespace find_f_log2_5_l152_152623

variable {f g : ℝ → ℝ}

-- f is an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- g is an odd function
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- Given conditions
axiom f_even : is_even f
axiom g_odd : is_odd g
axiom f_g_equation : ∀ x, f x + g x = (2:ℝ)^x + x

-- Proof goal: Compute f(log_2 5) and show it equals 13/5
theorem find_f_log2_5 : f (Real.log 5 / Real.log 2) = (13:ℝ) / 5 := by
  sorry

end find_f_log2_5_l152_152623


namespace div_condition_l152_152366

theorem div_condition (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  4 * (m * n + 1) % (m + n)^2 = 0 ↔ m = n := 
sorry

end div_condition_l152_152366


namespace total_cost_of_two_rackets_l152_152247

axiom racket_full_price : ℕ
axiom price_of_first_racket : racket_full_price = 60
axiom price_of_second_racket : racket_full_price / 2 = 30

theorem total_cost_of_two_rackets : 60 + 30 = 90 :=
sorry

end total_cost_of_two_rackets_l152_152247


namespace part_I_solution_set_part_II_range_of_a_l152_152037

-- Definitions
def f (x : ℝ) (a : ℝ) := |x - 1| + |a * x + 1|
def g (x : ℝ) := |x + 1| + 2

-- Part I: Prove the solution set of the inequality f(x) < 2 when a = 1/2
theorem part_I_solution_set (x : ℝ) : f x (1/2 : ℝ) < 2 ↔ 0 < x ∧ x < (4/3 : ℝ) :=
sorry
  
-- Part II: Prove the range of a such that (0, 1] ⊆ {x | f x a ≤ g x}
theorem part_II_range_of_a (a : ℝ) : (∀ x, 0 < x ∧ x ≤ 1 → f x a ≤ g x) ↔ -5 ≤ a ∧ a ≤ 3 :=
sorry

end part_I_solution_set_part_II_range_of_a_l152_152037


namespace min_books_borrowed_l152_152912

theorem min_books_borrowed
  (total_students : ℕ)
  (students_no_books : ℕ)
  (students_one_book : ℕ)
  (students_two_books : ℕ)
  (avg_books_per_student : ℝ)
  (total_students_eq : total_students = 40)
  (students_no_books_eq : students_no_books = 2)
  (students_one_book_eq : students_one_book = 12)
  (students_two_books_eq : students_two_books = 13)
  (avg_books_per_student_eq : avg_books_per_student = 2) :
  ∀ min_books_borrowed : ℕ, 
    (total_students * avg_books_per_student = 80) → 
    (students_one_book * 1 + students_two_books * 2 ≤ 38) → 
    (total_students - students_no_books - students_one_book - students_two_books = 13) →
    min_books_borrowed * 13 = 42 → 
    min_books_borrowed = 4 :=
by
  intros min_books_borrowed total_books_eq books_count_eq remaining_students_eq total_min_books_eq
  sorry

end min_books_borrowed_l152_152912


namespace tic_tac_toe_lines_l152_152920

theorem tic_tac_toe_lines (n : ℕ) (h_pos : 0 < n) : 
  ∃ lines : ℕ, lines = (5^n - 3^n) / 2 :=
sorry

end tic_tac_toe_lines_l152_152920


namespace probability_log_interval_l152_152399

open Set Real

noncomputable def probability_in_interval (a b c d : ℝ) (I J : Set ℝ) := 
  (b - a) / (d - c)

theorem probability_log_interval : 
  probability_in_interval 2 4 0 6 (Icc 0 6) (Ioo 2 4) = 1 / 3 := 
sorry

end probability_log_interval_l152_152399


namespace bert_toy_phones_l152_152188

theorem bert_toy_phones (P : ℕ) (berts_price_per_phone : ℕ) (berts_earning : ℕ)
                        (torys_price_per_gun : ℕ) (torys_earning : ℕ) (tory_guns : ℕ)
                        (earnings_difference : ℕ)
                        (h1 : berts_price_per_phone = 18)
                        (h2 : torys_price_per_gun = 20)
                        (h3 : tory_guns = 7)
                        (h4 : torys_earning = tory_guns * torys_price_per_gun)
                        (h5 : berts_earning = torys_earning + earnings_difference)
                        (h6 : earnings_difference = 4)
                        (h7 : P = berts_earning / berts_price_per_phone) :
  P = 8 := by sorry

end bert_toy_phones_l152_152188


namespace sufficient_not_necessary_condition_l152_152283

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ) (h_pos : x > 0) :
  (a = 4 → x + a / x ≥ 4) ∧ (∃ b : ℝ, b ≠ 4 ∧ ∃ x : ℝ, x > 0 ∧ x + b / x ≥ 4) :=
by
  sorry

end sufficient_not_necessary_condition_l152_152283


namespace minimize_std_deviation_l152_152963

theorem minimize_std_deviation (m n : ℝ) (h1 : m + n = 32) 
    (h2 : 11 ≤ 12 ∧ 12 ≤ m ∧ m ≤ n ∧ n ≤ 20 ∧ 20 ≤ 27) : 
    m = 16 :=
by {
  -- No proof required, only the theorem statement as per instructions
  sorry
}

end minimize_std_deviation_l152_152963


namespace length_of_road_l152_152680

-- Definitions based on conditions
def trees : Nat := 10
def interval : Nat := 10

-- Statement of the theorem
theorem length_of_road 
  (trees : Nat) (interval : Nat) (beginning_planting : Bool) (h_trees : trees = 10) (h_interval : interval = 10) (h_beginning : beginning_planting = true) 
  : (trees - 1) * interval = 90 := 
by 
  sorry

end length_of_road_l152_152680


namespace number_of_m_gons_proof_l152_152950

noncomputable def number_of_m_gons_with_two_acute_angles (m n : ℕ) (h1 : 4 < m) (h2 : m < n) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem number_of_m_gons_proof {m n : ℕ} (h1 : 4 < m) (h2 : m < n) :
  number_of_m_gons_with_two_acute_angles m n h1 h2 =
  (2 * n + 1) * ((Nat.choose (n + 1) (m - 1)) + (Nat.choose n (m - 1))) :=
sorry

end number_of_m_gons_proof_l152_152950


namespace three_equal_mass_piles_l152_152208

theorem three_equal_mass_piles (n : ℕ) (h : n > 3) : 
  (∃ (A B C : Finset ℕ), 
    (A ∪ B ∪ C = Finset.range (n + 1)) ∧ 
    (A ∩ B = ∅) ∧ 
    (A ∩ C = ∅) ∧ 
    (B ∩ C = ∅) ∧ 
    (A.sum id = B.sum id) ∧ 
    (B.sum id = C.sum id)) 
  ↔ (n % 3 = 0 ∨ n % 3 = 2) :=
sorry

end three_equal_mass_piles_l152_152208


namespace solve_inequality_l152_152041

theorem solve_inequality (x : ℝ) :
  (x - 5) / (x - 3)^2 < 0 ↔ x ∈ (Set.Iio 3 ∪ Set.Ioo 3 5) :=
by
  sorry

end solve_inequality_l152_152041


namespace solve_for_x_l152_152141

theorem solve_for_x (x : ℝ) (h : 3 / (x + 10) = 1 / (2 * x)) : x = 2 :=
sorry

end solve_for_x_l152_152141


namespace number_of_terms_arithmetic_sequence_l152_152782

theorem number_of_terms_arithmetic_sequence :
  ∀ (a d l : ℤ), a = -36 → d = 6 → l = 66 → ∃ n, l = a + (n-1) * d ∧ n = 18 :=
by
  intros a d l ha hd hl
  exists 18
  rw [ha, hd, hl]
  sorry

end number_of_terms_arithmetic_sequence_l152_152782


namespace dealership_sales_prediction_l152_152462

theorem dealership_sales_prediction (sports_cars_sold sedans SUVs : ℕ) 
    (ratio_sc_sedans : 3 * sedans = 5 * sports_cars_sold) 
    (ratio_sc_SUVs : sports_cars_sold = 2 * SUVs) 
    (sports_cars_sold_next_month : sports_cars_sold = 36) :
    (sedans = 60 ∧ SUVs = 72) :=
sorry

end dealership_sales_prediction_l152_152462


namespace greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l152_152414

theorem greatest_integer_less_than_150_with_gcd_30_eq_5_is_145 :
  ∃ n : ℕ, n < 150 ∧ Nat.gcd n 30 = 5 ∧ (∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ n) :=
sorry

end greatest_integer_less_than_150_with_gcd_30_eq_5_is_145_l152_152414


namespace unique_sequence_l152_152321

theorem unique_sequence (a : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 1) ^ 2 = 1 + (n + 2021) * a n) →
  (∀ n : ℕ, a n = n + 2019) :=
by
  sorry

end unique_sequence_l152_152321


namespace reduced_cost_per_meter_l152_152855

theorem reduced_cost_per_meter (original_cost total_cost new_length original_length : ℝ) :
  original_cost = total_cost / original_length →
  new_length = original_length + 4 →
  total_cost = total_cost →
  original_cost - (total_cost / new_length) = 1 :=
by sorry

end reduced_cost_per_meter_l152_152855


namespace mass_fraction_K2SO4_l152_152239

theorem mass_fraction_K2SO4 :
  (2.61 * 100 / 160) = 1.63 :=
by
  -- Proof details are not required as per instructions
  sorry

end mass_fraction_K2SO4_l152_152239


namespace extra_men_needed_approx_is_60_l152_152065

noncomputable def extra_men_needed : ℝ :=
  let total_distance := 15.0   -- km
  let total_days := 300.0      -- days
  let initial_workforce := 40.0 -- men
  let completed_distance := 2.5 -- km
  let elapsed_days := 100.0    -- days

  let remaining_distance := total_distance - completed_distance -- km
  let remaining_days := total_days - elapsed_days               -- days

  let current_rate := completed_distance / elapsed_days -- km/day
  let required_rate := remaining_distance / remaining_days -- km/day

  let required_factor := required_rate / current_rate
  let new_workforce := initial_workforce * required_factor
  let extra_men := new_workforce - initial_workforce

  extra_men

theorem extra_men_needed_approx_is_60 :
  abs (extra_men_needed - 60) < 1 :=
sorry

end extra_men_needed_approx_is_60_l152_152065


namespace solve_for_y_l152_152876

theorem solve_for_y (x y : ℝ) (h1 : x ^ (2 * y) = 16) (h2 : x = 2) : y = 2 :=
by {
  sorry
}

end solve_for_y_l152_152876


namespace bill_painting_hours_l152_152832

theorem bill_painting_hours (B J : ℝ) (hB : 0 < B) (hJ : 0 < J) : 
  ∃ t : ℝ, t = (B-1)/(B+J) ∧ (t + 1 = (B * (J + 1)) / (B + J)) :=
by
  sorry

end bill_painting_hours_l152_152832


namespace circle_center_radius_l152_152827

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 4 * x + 2 * y - 4 = 0 ↔ (x - 2)^2 + (y + 1)^2 = 3 :=
by
  sorry

end circle_center_radius_l152_152827


namespace equation1_solution_equation2_solution_l152_152964

theorem equation1_solution (x : ℚ) : 2 * (x - 3) = 1 - 3 * (x + 1) → x = 4 / 5 :=
by sorry

theorem equation2_solution (x : ℚ) : 3 * x + (x - 1) / 2 = 3 - (x - 1) / 3 → x = 1 :=
by sorry

end equation1_solution_equation2_solution_l152_152964


namespace least_value_sum_l152_152903

theorem least_value_sum (x y z : ℤ) (h : (x - 10) * (y - 5) * (z - 2) = 1000) : x + y + z = 92 :=
sorry

end least_value_sum_l152_152903


namespace add_like_terms_l152_152768

variable (a : ℝ)

theorem add_like_terms : a^2 + 2 * a^2 = 3 * a^2 := 
by sorry

end add_like_terms_l152_152768


namespace afb_leq_bfa_l152_152962

open Real

variable {f : ℝ → ℝ}

theorem afb_leq_bfa
  (h_nonneg : ∀ x > 0, f x ≥ 0)
  (h_diff : ∀ x > 0, DifferentiableAt ℝ f x)
  (h_cond : ∀ x > 0, x * (deriv (deriv f) x) - f x ≤ 0)
  (a b : ℝ)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_a_lt_b : a < b) :
  a * f b ≤ b * f a := 
sorry

end afb_leq_bfa_l152_152962


namespace sum_of_square_roots_l152_152691

theorem sum_of_square_roots : 
  (Real.sqrt 1) + (Real.sqrt (1 + 3)) + (Real.sqrt (1 + 3 + 5)) + (Real.sqrt (1 + 3 + 5 + 7)) = 10 := 
by 
  sorry

end sum_of_square_roots_l152_152691


namespace journey_length_25_km_l152_152369

theorem journey_length_25_km:
  ∀ (D T : ℝ),
  (D = 100 * T) →
  (D = 50 * (T + 15/60)) →
  D = 25 :=
by
  intros D T h1 h2
  sorry

end journey_length_25_km_l152_152369


namespace mean_correct_and_no_seven_l152_152203

-- Define the set of numbers.
def numbers : List ℕ := 
  [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

-- Define the arithmetic mean of the numbers in the set.
def arithmetic_mean (l : List ℕ) : ℕ := (l.sum / l.length)

-- Specify the mean value
def mean_value : ℕ := 109629012

-- State the theorem that the mean value is correct and does not contain the digit 7.
theorem mean_correct_and_no_seven : arithmetic_mean numbers = mean_value ∧ ¬ 7 ∈ (mean_value.digits 10) :=
  sorry

end mean_correct_and_no_seven_l152_152203


namespace find_f_half_l152_152879

variable {α : Type} [DivisionRing α]

theorem find_f_half {f : α → α} (h : ∀ x, f (1 - 2 * x) = 1 / (x^2)) : f (1 / 2) = 16 :=
by
  sorry

end find_f_half_l152_152879


namespace evaluate_expression_l152_152303

theorem evaluate_expression : 
  - (16 / 2 * 8 - 72 + 4^2) = -8 :=
by 
  -- here, the proof would typically go
  sorry

end evaluate_expression_l152_152303


namespace solution_l152_152531

-- Define the linear equations and their solutions
def system_of_equations (x y : ℕ) :=
  3 * x + y = 500 ∧ x + 2 * y = 250

-- Define the budget constraint
def budget_constraint (m : ℕ) :=
  150 * m + 50 * (25 - m) ≤ 2700

-- Define the purchasing plans and costs
def purchasing_plans (m n : ℕ) :=
  (m = 12 ∧ n = 13 ∧ 150 * m + 50 * n = 2450) ∨ 
  (m = 13 ∧ n = 12 ∧ 150 * m + 50 * n = 2550) ∨ 
  (m = 14 ∧ n = 11 ∧ 150 * m + 50 * n = 2650)

-- Define the Lean statement
theorem solution :
  (∃ x y, system_of_equations x y ∧ x = 150 ∧ y = 50) ∧
  (∃ m, budget_constraint m ∧ m ≤ 14) ∧
  (∃ m n, 12 ≤ m ∧ m ≤ 14 ∧ m + n = 25 ∧ purchasing_plans m n ∧ 150 * m + 50 * n = 2450) :=
sorry

end solution_l152_152531


namespace quadrilateral_areas_product_l152_152342

noncomputable def areas_product_property (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) : Prop :=
  (S_ADP * S_BCP * S_ABP * S_CDP) % 10000 ≠ 1988
  
theorem quadrilateral_areas_product (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) :
  areas_product_property S_ADP S_ABP S_CDP S_BCP h1 :=
by
  sorry

end quadrilateral_areas_product_l152_152342


namespace king_paid_after_tip_l152_152689

-- Define the cost of the crown and the tip percentage
def cost_of_crown : ℝ := 20000
def tip_percentage : ℝ := 0.10

-- Define the total amount paid after the tip
def total_amount_paid (C : ℝ) (tip_pct : ℝ) : ℝ :=
  C + (tip_pct * C)

-- Theorem statement: The total amount paid after the tip is $22,000
theorem king_paid_after_tip : total_amount_paid cost_of_crown tip_percentage = 22000 := by
  sorry

end king_paid_after_tip_l152_152689


namespace runner_distance_l152_152999

theorem runner_distance :
  ∃ x t d : ℕ,
    d = x * t ∧
    d = (x + 1) * (2 * t / 3) ∧
    d = (x - 1) * (t + 3) ∧
    d = 6 :=
by
  sorry

end runner_distance_l152_152999


namespace weight_of_new_person_l152_152791

-- Definition of the problem
def average_weight_increases (W : ℝ) (N : ℝ) : Prop :=
  let increase := 2.5
  W - 45 + N = W + 8 * increase

-- The main statement we need to prove
theorem weight_of_new_person (W : ℝ) : ∃ N, average_weight_increases W N ∧ N = 65 := 
by
  use 65
  unfold average_weight_increases
  sorry

end weight_of_new_person_l152_152791


namespace smallest_positive_debt_l152_152274

noncomputable def pigs_value : ℤ := 300
noncomputable def goats_value : ℤ := 210

theorem smallest_positive_debt : ∃ D p g : ℤ, (D = pigs_value * p + goats_value * g) ∧ D > 0 ∧ ∀ D' p' g' : ℤ, (D' = pigs_value * p' + goats_value * g' ∧ D' > 0) → D ≤ D' :=
by
  sorry

end smallest_positive_debt_l152_152274


namespace monotonicity_tangent_intersection_points_l152_152915

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l152_152915


namespace problem_solution_l152_152113

theorem problem_solution (x y m : ℝ) (hx : x > 0) (hy : y > 0) : 
  (∀ x y, (2 * y / x) + (8 * x / y) > m^2 + 2 * m) → -4 < m ∧ m < 2 :=
by
  intros h
  sorry

end problem_solution_l152_152113


namespace Eval_trig_exp_l152_152909

theorem Eval_trig_exp :
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 / 3 :=
by
  sorry

end Eval_trig_exp_l152_152909


namespace digit_difference_is_7_l152_152500

def local_value (d : Nat) (place : Nat) : Nat :=
  d * (10^place)

def face_value (d : Nat) : Nat :=
  d

def difference (d : Nat) (place : Nat) : Nat :=
  local_value d place - face_value d

def numeral : Nat := 65793

theorem digit_difference_is_7 :
  ∃ d place, 0 ≤ d ∧ d < 10 ∧ difference d place = 693 ∧ d ∈ [6, 5, 7, 9, 3] ∧ numeral = 65793 ∧
  (local_value 6 4 = 60000 ∧ local_value 5 3 = 5000 ∧ local_value 7 2 = 700 ∧ local_value 9 1 = 90 ∧ local_value 3 0 = 3 ∧
   face_value 6 = 6 ∧ face_value 5 = 5 ∧ face_value 7 = 7 ∧ face_value 9 = 9 ∧ face_value 3 = 3) ∧ 
  d = 7 :=
sorry

end digit_difference_is_7_l152_152500


namespace circle_and_tangent_lines_l152_152530

-- Define the problem conditions
def passes_through (a b r : ℝ) : Prop :=
  (a - (-2))^2 + (b - 2)^2 = r^2 ∧
  (a - (-5))^2 + (b - 5)^2 = r^2

def lies_on_line (a b : ℝ) : Prop :=
  a + b + 3 = 0

-- Define the standard equation of the circle
def is_circle_eq (a b r : ℝ) : Prop := ∀ x y : ℝ, 
  (x - a)^2 + (y - b)^2 = r^2 ↔ (x + 5)^2 + (y - 2)^2 = 9

-- Define the tangent lines
def is_tangent_lines (x y k : ℝ) : Prop :=
  (k = (20 / 21) ∨ x = -2) → (20 * x - 21 * y + 229 = 0 ∨ x = -2)

-- The theorem statement in Lean 4
theorem circle_and_tangent_lines (a b r : ℝ) (x y k : ℝ) :
  passes_through a b r →
  lies_on_line a b →
  is_circle_eq a b r →
  is_tangent_lines x y k :=
by {
  sorry
}

end circle_and_tangent_lines_l152_152530


namespace butanoic_acid_molecular_weight_l152_152833

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_weight_butanoic_acid : ℝ :=
  4 * atomic_weight_C + 8 * atomic_weight_H + 2 * atomic_weight_O

theorem butanoic_acid_molecular_weight :
  molecular_weight_butanoic_acid = 88.104 :=
by
  -- proof not required
  sorry

end butanoic_acid_molecular_weight_l152_152833


namespace find_a_l152_152001

def operation (a b : ℤ) : ℤ := 2 * a - b * b

theorem find_a (a : ℤ) : operation a 3 = 15 → a = 12 := by
  sorry

end find_a_l152_152001


namespace minimize_fence_perimeter_l152_152187

-- Define the area of the pen
def area (L W : ℝ) : ℝ := L * W

-- Define that only three sides of the fence need to be fenced
def perimeter (L W : ℝ) : ℝ := 2 * W + L

-- Given conditions
def A : ℝ := 54450  -- Area in square meters

-- The proof statement
theorem minimize_fence_perimeter :
  ∃ (L W : ℝ), 
  area L W = A ∧ 
  ∀ (L' W' : ℝ), area L' W' = A → perimeter L W ≤ perimeter L' W' ∧ L = 330 ∧ W = 165 :=
sorry

end minimize_fence_perimeter_l152_152187


namespace compute_expression_l152_152034

section
variable (a : ℝ)

theorem compute_expression :
  (-a^2)^3 * a^3 = -a^9 :=
sorry
end

end compute_expression_l152_152034


namespace geometric_sequence_a3_eq_sqrt_5_l152_152699

theorem geometric_sequence_a3_eq_sqrt_5 (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * r)
  (h_a1 : a 1 = 1) (h_a5 : a 5 = 5) :
  a 3 = Real.sqrt 5 :=
sorry

end geometric_sequence_a3_eq_sqrt_5_l152_152699


namespace inequality_system_solution_l152_152663

theorem inequality_system_solution (a b : ℝ) (h : ∀ x : ℝ, x > -a → x > -b) : a ≥ b :=
by
  sorry

end inequality_system_solution_l152_152663


namespace trader_profit_l152_152994

-- Definitions and conditions
def original_price (P : ℝ) := P
def discounted_price (P : ℝ) := 0.70 * P
def marked_up_price (P : ℝ) := 0.84 * P
def sale_price (P : ℝ) := 0.714 * P
def final_price (P : ℝ) := 1.2138 * P

-- Proof statement
theorem trader_profit (P : ℝ) : ((final_price P - original_price P) / original_price P) * 100 = 21.38 := by
  sorry

end trader_profit_l152_152994


namespace train_speed_in_kmh_l152_152737

-- Definitions of conditions
def time_to_cross_platform := 30  -- in seconds
def time_to_cross_man := 17  -- in seconds
def length_of_platform := 260  -- in meters

-- Conversion factor from m/s to km/h
def meters_per_second_to_kilometers_per_hour (v : ℕ) : ℕ :=
  v * 36 / 10

-- The theorem statement
theorem train_speed_in_kmh :
  (∃ (L V : ℕ),
    L = V * time_to_cross_man ∧
    L + length_of_platform = V * time_to_cross_platform ∧
    meters_per_second_to_kilometers_per_hour V = 72) :=
sorry

end train_speed_in_kmh_l152_152737


namespace solve_quadratic_equation_l152_152516

theorem solve_quadratic_equation (x : ℝ) : 4 * (2 * x + 1) ^ 2 = 9 * (x - 3) ^ 2 ↔ x = -11 ∨ x = 1 := 
by sorry

end solve_quadratic_equation_l152_152516


namespace prime_numbers_satisfying_condition_l152_152919

theorem prime_numbers_satisfying_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ x : ℕ, 1 + p * 2^p = x^2) ↔ p = 2 ∨ p = 3 :=
by
  sorry

end prime_numbers_satisfying_condition_l152_152919


namespace smallest_N_value_l152_152392

theorem smallest_N_value (a b c d : ℕ)
  (h1 : gcd a b = 1 ∧ gcd a c = 2 ∧ gcd a d = 4 ∧ gcd b c = 5 ∧ gcd b d = 3 ∧ gcd c d = N)
  (h2 : N > 5) : N = 14 := sorry

end smallest_N_value_l152_152392


namespace sin_135_eq_sqrt2_div_2_l152_152726

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l152_152726


namespace total_points_correct_l152_152856

def points_from_two_pointers (t : ℕ) : ℕ := 2 * t
def points_from_three_pointers (th : ℕ) : ℕ := 3 * th
def points_from_free_throws (f : ℕ) : ℕ := f

def total_points (two_points three_points free_throws : ℕ) : ℕ :=
  points_from_two_pointers two_points + points_from_three_pointers three_points + points_from_free_throws free_throws

def sam_points : ℕ := total_points 20 5 10
def alex_points : ℕ := total_points 15 6 8
def jake_points : ℕ := total_points 10 8 5
def lily_points : ℕ := total_points 12 3 16

def game_total_points : ℕ := sam_points + alex_points + jake_points + lily_points

theorem total_points_correct : game_total_points = 219 :=
by
  sorry

end total_points_correct_l152_152856


namespace problem_statement_l152_152165

def T : Set ℤ :=
  {n^2 + (n+2)^2 + (n+4)^2 | n : ℤ }

theorem problem_statement :
  (∀ x ∈ T, ¬ (4 ∣ x)) ∧ (∃ x ∈ T, 13 ∣ x) :=
by
  sorry

end problem_statement_l152_152165


namespace sam_seashell_count_l152_152386

/-!
# Problem statement:
-/
def initialSeashells := 35
def seashellsGivenToJoan := 18
def seashellsFoundToday := 20
def seashellsGivenToTom := 5

/-!
# Proof goal: Prove that the current number of seashells Sam has is 32.
-/
theorem sam_seashell_count :
  initialSeashells - seashellsGivenToJoan + seashellsFoundToday - seashellsGivenToTom = 32 :=
  sorry

end sam_seashell_count_l152_152386


namespace find_number_l152_152704

theorem find_number (n : ℝ) (h : (1 / 3) * n = 6) : n = 18 :=
sorry

end find_number_l152_152704


namespace parabola_vertex_x_coordinate_l152_152312

theorem parabola_vertex_x_coordinate (a b c : ℝ) (h1 : c = 0) (h2 : 16 * a + 4 * b = 0) (h3 : 9 * a + 3 * b = 9) : 
    -b / (2 * a) = 2 :=
by 
  -- You can start by adding a proof here
  sorry

end parabola_vertex_x_coordinate_l152_152312


namespace probability_prime_or_odd_ball_l152_152025

def isPrime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def isOdd (n : ℕ) : Prop :=
  n % 2 = 1

def isPrimeOrOdd (n : ℕ) : Prop :=
  isPrime n ∨ isOdd n

theorem probability_prime_or_odd_ball :
  (1+2+3+5+7)/8 = 5/8 := by
  sorry

end probability_prime_or_odd_ball_l152_152025


namespace find_a1_l152_152928

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a1 (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a →
  a 6 = 9 →
  a 3 = 3 * a 2 →
  a 1 = -1 :=
by
  sorry

end find_a1_l152_152928


namespace total_birds_in_tree_l152_152263

theorem total_birds_in_tree (bluebirds cardinals swallows : ℕ) 
  (h1 : swallows = 2) 
  (h2 : swallows = bluebirds / 2) 
  (h3 : cardinals = 3 * bluebirds) : 
  swallows + bluebirds + cardinals = 18 := 
by 
  sorry

end total_birds_in_tree_l152_152263


namespace jonathan_weekly_deficit_correct_l152_152442

def daily_intake_non_saturday : ℕ := 2500
def daily_intake_saturday : ℕ := 3500
def daily_burn : ℕ := 3000
def weekly_caloric_deficit : ℕ :=
  (7 * daily_burn) - ((6 * daily_intake_non_saturday) + daily_intake_saturday)

theorem jonathan_weekly_deficit_correct :
  weekly_caloric_deficit = 2500 :=
by
  unfold weekly_caloric_deficit daily_intake_non_saturday daily_intake_saturday daily_burn
  sorry

end jonathan_weekly_deficit_correct_l152_152442


namespace lucy_total_fish_l152_152401

theorem lucy_total_fish (current fish_needed : ℕ) (h1 : current = 212) (h2 : fish_needed = 68) : 
  current + fish_needed = 280 := 
by
  sorry

end lucy_total_fish_l152_152401


namespace tailoring_business_days_l152_152026

theorem tailoring_business_days
  (shirts_per_day : ℕ)
  (fabric_per_shirt : ℕ)
  (pants_per_day : ℕ)
  (fabric_per_pant : ℕ)
  (total_fabric : ℕ)
  (h1 : shirts_per_day = 3)
  (h2 : fabric_per_shirt = 2)
  (h3 : pants_per_day = 5)
  (h4 : fabric_per_pant = 5)
  (h5 : total_fabric = 93) :
  (total_fabric / (shirts_per_day * fabric_per_shirt + pants_per_day * fabric_per_pant)) = 3 :=
by {
  sorry
}

end tailoring_business_days_l152_152026


namespace gcd_840_1764_evaluate_polynomial_at_2_l152_152107

-- Define the Euclidean algorithm steps and prove the gcd result
theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

-- Define the polynomial and evaluate it using Horner's method
def polynomial := λ x : ℕ => 2 * (x ^ 4) + 3 * (x ^ 3) + 5 * x - 4

theorem evaluate_polynomial_at_2 : polynomial 2 = 62 := by
  sorry

end gcd_840_1764_evaluate_polynomial_at_2_l152_152107


namespace right_triangle_third_side_l152_152472

theorem right_triangle_third_side (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) : c = Real.sqrt (b^2 - a^2) :=
by
  rw [h1, h2]
  sorry

end right_triangle_third_side_l152_152472


namespace average_wage_per_day_l152_152810

theorem average_wage_per_day :
  let num_male := 20
  let num_female := 15
  let num_child := 5
  let wage_male := 35
  let wage_female := 20
  let wage_child := 8
  let total_wages := (num_male * wage_male) + (num_female * wage_female) + (num_child * wage_child)
  let total_workers := num_male + num_female + num_child
  total_wages / total_workers = 26 := by
  sorry

end average_wage_per_day_l152_152810


namespace semesters_needed_l152_152613

def total_credits : ℕ := 120
def credits_per_class : ℕ := 3
def classes_per_semester : ℕ := 5

theorem semesters_needed (h1 : total_credits = 120)
                         (h2 : credits_per_class = 3)
                         (h3 : classes_per_semester = 5) :
  total_credits / (credits_per_class * classes_per_semester) = 8 := 
by {
  sorry
}

end semesters_needed_l152_152613


namespace max_value_of_a_squared_b_squared_c_squared_l152_152174

theorem max_value_of_a_squared_b_squared_c_squared
  (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_constraint : a + 2 * b + 3 * c = 1) : a^2 + b^2 + c^2 ≤ 1 :=
sorry

end max_value_of_a_squared_b_squared_c_squared_l152_152174


namespace inequality_solution_set_empty_l152_152585

theorem inequality_solution_set_empty (a : ℝ) :
  (∀ x : ℝ, ¬ (|x - 2| + |x + 3| < a)) → a ≤ 5 :=
by sorry

end inequality_solution_set_empty_l152_152585


namespace infinite_squares_in_arithmetic_sequence_l152_152428

open Nat Int

theorem infinite_squares_in_arithmetic_sequence
  (a d : ℤ) (h_d_nonneg : d ≥ 0) (x : ℤ) 
  (hx_square : ∃ n : ℕ, a + n * d = x * x) :
  ∃ (infinitely_many_n : ℕ → Prop), 
    (∀ k : ℕ, ∃ n : ℕ, infinitely_many_n n ∧ a + n * d = (x + k * d) * (x + k * d)) :=
sorry

end infinite_squares_in_arithmetic_sequence_l152_152428


namespace set_intersection_complement_l152_152792

-- Define the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 3, 6, 7}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ ¬ x ∈ A}

-- Define the intersection of B and complement_U_A
def B_inter_complement_U_A : Set ℕ := B ∩ complement_U_A

-- The statement to prove: B ∩ complement_U_A = {6, 7}
theorem set_intersection_complement :
  B_inter_complement_U_A = {6, 7} := by sorry

end set_intersection_complement_l152_152792


namespace discount_percentage_l152_152632

theorem discount_percentage (CP SP SP_no_discount discount : ℝ)
  (h1 : SP = CP * (1 + 0.44))
  (h2 : SP_no_discount = CP * (1 + 0.50))
  (h3 : discount = SP_no_discount - SP) :
  (discount / SP_no_discount) * 100 = 4 :=
by
  sorry

end discount_percentage_l152_152632


namespace number_of_bananas_l152_152355

-- Define costs as constants
def cost_per_banana := 1
def cost_per_apple := 2
def cost_per_twelve_strawberries := 4
def cost_per_avocado := 3
def cost_per_half_bunch_grapes := 2
def total_cost := 28

-- Define quantities as constants
def number_of_apples := 3
def number_of_strawberries := 24
def number_of_avocados := 2
def number_of_half_bunches_grapes := 2

-- Define calculated costs
def cost_of_apples := number_of_apples * cost_per_apple
def cost_of_strawberries := (number_of_strawberries / 12) * cost_per_twelve_strawberries
def cost_of_avocados := number_of_avocados * cost_per_avocado
def cost_of_grapes := number_of_half_bunches_grapes * cost_per_half_bunch_grapes

-- Define total cost of other fruits
def total_cost_of_other_fruits := cost_of_apples + cost_of_strawberries + cost_of_avocados + cost_of_grapes

-- Define the remaining cost for bananas
def remaining_cost := total_cost - total_cost_of_other_fruits

-- Prove the number of bananas
theorem number_of_bananas : remaining_cost / cost_per_banana = 4 :=
by
  -- This is a placeholder to indicate a non-implemented proof
  sorry

end number_of_bananas_l152_152355


namespace min_value_f_l152_152346

def f (x : ℝ) : ℝ := |2 * x - 1| + |3 * x - 2| + |4 * x - 3| + |5 * x - 4|

theorem min_value_f : (∃ x : ℝ, ∀ y : ℝ, f y ≥ f x) := 
sorry

end min_value_f_l152_152346


namespace sam_pennies_total_l152_152447

def initial_pennies : ℕ := 980
def found_pennies : ℕ := 930
def exchanged_pennies : ℕ := 725
def gifted_pennies : ℕ := 250

theorem sam_pennies_total :
  initial_pennies + found_pennies - exchanged_pennies + gifted_pennies = 1435 := 
sorry

end sam_pennies_total_l152_152447


namespace sphere_surface_area_l152_152907

theorem sphere_surface_area (R : ℝ) (h : (4 / 3) * π * R^3 = (32 / 3) * π) : 4 * π * R^2 = 16 * π :=
sorry

end sphere_surface_area_l152_152907


namespace total_saplings_l152_152752

theorem total_saplings (a_efficiency b_efficiency : ℝ) (A B T n : ℝ) 
  (h1 : a_efficiency = (3/4))
  (h2 : b_efficiency = 1)
  (h3 : B = n + 36)
  (h4 : T = 2 * n + 36)
  (h5 : n * (4/3) = n + 36)
  : T = 252 :=
by {
  sorry
}

end total_saplings_l152_152752


namespace range_of_a_for_circle_l152_152700

theorem range_of_a_for_circle (a : ℝ) : 
  -2 < a ∧ a < 2/3 ↔ 
  ∃ (x y : ℝ), (x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1) = 0 :=
sorry

end range_of_a_for_circle_l152_152700


namespace polygon_sides_l152_152825

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l152_152825


namespace b_share_220_l152_152204

theorem b_share_220 (A B C : ℝ) (h1 : A = B + 40) (h2 : C = A + 30) (h3 : A + B + C = 770) : 
  B = 220 :=
by
  sorry

end b_share_220_l152_152204


namespace max_sum_of_factors_l152_152367

theorem max_sum_of_factors (a b : ℕ) (h : a * b = 42) : a + b ≤ 43 :=
by
  -- sorry to skip the proof
  sorry

end max_sum_of_factors_l152_152367


namespace min_Box_value_l152_152272

/-- The conditions are given as:
  1. (ax + b)(bx + a) = 24x^2 + Box * x + 24
  2. a, b, Box are distinct integers
  The task is to find the minimum possible value of Box.
-/
theorem min_Box_value :
  ∃ (a b Box : ℤ), a ≠ b ∧ a ≠ Box ∧ b ≠ Box ∧ (∀ x : ℤ, (a * x + b) * (b * x + a) = 24 * x^2 + Box * x + 24) ∧ Box = 52 := sorry

end min_Box_value_l152_152272


namespace percentage_relationships_l152_152644

variable (a b c d e f g : ℝ)

theorem percentage_relationships (h1 : d = 0.22 * b) (h2 : d = 0.35 * f)
                                 (h3 : e = 0.27 * a) (h4 : e = 0.60 * f)
                                 (h5 : c = 0.14 * a) (h6 : c = 0.40 * b)
                                 (h7 : d = 2 * c) (h8 : g = 3 * e):
    b = 0.7 * a ∧ f = 0.45 * a ∧ g = 0.81 * a :=
sorry

end percentage_relationships_l152_152644


namespace count_100_digit_numbers_divisible_by_3_l152_152769

def num_100_digit_numbers_divisible_by_3 : ℕ := (4^50 + 2) / 3

theorem count_100_digit_numbers_divisible_by_3 :
  ∃ n : ℕ, n = num_100_digit_numbers_divisible_by_3 :=
by
  use (4^50 + 2) / 3
  sorry

end count_100_digit_numbers_divisible_by_3_l152_152769


namespace find_d_l152_152727

noncomputable def area_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem find_d (d : ℝ) (h₁ : 0 ≤ d ∧ d ≤ 2) (h₂ : 6 - ((1 / 2) * (2 - d) * 2) = 2 * ((1 / 2) * (2 - d) * 2)) : 
  d = 0 :=
sorry

end find_d_l152_152727


namespace line_tangent_to_parabola_l152_152131

theorem line_tangent_to_parabola (d : ℝ) :
  (∀ x y: ℝ, y = 3 * x + d ↔ y^2 = 12 * x) → d = 1 :=
by
  sorry

end line_tangent_to_parabola_l152_152131


namespace arithmetic_sequence_initial_term_l152_152732

theorem arithmetic_sequence_initial_term (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : ∀ n, S n = n * (a 1 + n * d / 2))
  (h_product : a 2 * a 3 = a 4 * a 5)
  (h_sum_9 : S 9 = 27)
  (h_d_nonzero : d ≠ 0) :
  a 1 = -5 :=
sorry

end arithmetic_sequence_initial_term_l152_152732


namespace redistributed_gnomes_l152_152332

def WestervilleWoods : ℕ := 20
def RavenswoodForest := 4 * WestervilleWoods
def GreenwoodGrove := (5 * RavenswoodForest) / 4
def OwnerTakes (f: ℕ) (p: ℚ) := p * f

def RemainingGnomes (initial: ℕ) (p: ℚ) := initial - (OwnerTakes initial p)

def TotalRemainingGnomes := 
  (RemainingGnomes RavenswoodForest (40 / 100)) + 
  (RemainingGnomes WestervilleWoods (30 / 100)) + 
  (RemainingGnomes GreenwoodGrove (50 / 100))

def GnomesPerForest := TotalRemainingGnomes / 3

theorem redistributed_gnomes : 
  2 * 37 + 38 = TotalRemainingGnomes := by
  sorry

end redistributed_gnomes_l152_152332


namespace determine_x_value_l152_152670

theorem determine_x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y ^ 3) (h2 : x / 9 = 9 * y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 := by 
  sorry

end determine_x_value_l152_152670


namespace time_for_nth_mile_l152_152112

noncomputable def speed (k : ℝ) (d : ℝ) : ℝ := k / (d * d)

noncomputable def time_for_mile (n : ℕ) : ℝ :=
  if n = 1 then 1
  else if n = 2 then 2
  else 2 * (n - 1) * (n - 1)

theorem time_for_nth_mile (n : ℕ) (h₁ : ∀ d : ℝ, d ≥ 1 → speed (1/2) d = 1 / (2 * d * d))
  (h₂ : time_for_mile 1 = 1)
  (h₃ : time_for_mile 2 = 2) :
  time_for_mile n = 2 * (n - 1) * (n - 1) := sorry

end time_for_nth_mile_l152_152112


namespace find_m_l152_152024

theorem find_m (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 ^ a = m) (h4 : 3 ^ b = m) (h5 : 2 * a * b = a + b) : m = Real.sqrt 6 :=
sorry

end find_m_l152_152024


namespace polynomial_evaluation_l152_152211

theorem polynomial_evaluation (n : ℕ) (p : ℕ → ℝ) 
  (h_poly : ∀ k, k ≤ n → p k = 1 / (Nat.choose (n + 1) k)) :
  p (n + 1) = if n % 2 = 0 then 1 else 0 :=
by
  sorry

end polynomial_evaluation_l152_152211


namespace morning_routine_time_l152_152868

section

def time_for_teeth_and_face : ℕ := 3
def time_for_cooking : ℕ := 14
def time_for_reading_while_cooking : ℕ := time_for_cooking - time_for_teeth_and_face
def additional_time_for_reading : ℕ := 1
def total_time_for_reading : ℕ := time_for_reading_while_cooking + additional_time_for_reading
def time_for_eating : ℕ := 6

def total_time_to_school : ℕ := time_for_cooking + time_for_eating

theorem morning_routine_time :
  total_time_to_school = 21 := sorry

end

end morning_routine_time_l152_152868


namespace mass_percentage_O_in_N2O3_l152_152815

variable (m_N : ℝ := 14.01)  -- Molar mass of nitrogen (N) in g/mol
variable (m_O : ℝ := 16.00)  -- Molar mass of oxygen (O) in g/mol
variable (n_N : ℕ := 2)      -- Number of nitrogen (N) atoms in N2O3
variable (n_O : ℕ := 3)      -- Number of oxygen (O) atoms in N2O3

theorem mass_percentage_O_in_N2O3 :
  let molar_mass_N2O3 := (n_N * m_N) + (n_O * m_O)
  let mass_O_in_N2O3 := n_O * m_O
  let percentage_O := (mass_O_in_N2O3 / molar_mass_N2O3) * 100
  percentage_O = 63.15 :=
by
  -- Formal proof here
  sorry

end mass_percentage_O_in_N2O3_l152_152815


namespace students_passed_in_both_subjects_l152_152614

theorem students_passed_in_both_subjects:
  ∀ (F_H F_E F_HE : ℝ), F_H = 0.30 → F_E = 0.42 → F_HE = 0.28 → (1 - (F_H + F_E - F_HE)) = 0.56 :=
by
  intros F_H F_E F_HE h1 h2 h3
  sorry

end students_passed_in_both_subjects_l152_152614


namespace khalil_dogs_l152_152816

theorem khalil_dogs (D : ℕ) (cost_dog cost_cat : ℕ) (num_cats total_cost : ℕ) 
  (h1 : cost_dog = 60)
  (h2 : cost_cat = 40)
  (h3 : num_cats = 60)
  (h4 : total_cost = 3600) :
  (num_cats * cost_cat + D * cost_dog = total_cost) → D = 20 :=
by
  intros h
  sorry

end khalil_dogs_l152_152816


namespace part1_part2_l152_152762

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x + 3|

theorem part1 (x : ℝ) : f x ≥ 6 ↔ x ≥ 1 ∨ x ≤ -2 := by
  sorry

theorem part2 (a b : ℝ) (m : ℝ) 
  (a_pos : a > 0) (b_pos : b > 0) 
  (fmin : m = 4) 
  (condition : 2 * a * b + a + 2 * b = m) : 
  a + 2 * b = 2 * Real.sqrt 5 - 2 := by
  sorry

end part1_part2_l152_152762


namespace rod_sliding_friction_l152_152896

noncomputable def coefficient_of_friction (mg : ℝ) (F : ℝ) (α : ℝ) := 
  (F * Real.cos α - 6 * mg * Real.sin α) / (6 * mg)

theorem rod_sliding_friction
  (α : ℝ)
  (hα : α = 85 * Real.pi / 180)
  (mg : ℝ)
  (hmg_pos : 0 < mg)
  (F : ℝ)
  (hF : F = (mg - 6 * mg * Real.cos 85) / Real.sin 85) :
  coefficient_of_friction mg F α = 0.08 := 
by
  simp [coefficient_of_friction, hα, hF, Real.cos, Real.sin]
  sorry

end rod_sliding_friction_l152_152896


namespace quadratic_distinct_real_roots_l152_152214

theorem quadratic_distinct_real_roots (m : ℝ) : 
  (m ≠ 0 ∧ m < 1 / 5) ↔ ∃ (x y : ℝ), x ≠ y ∧ m * x^2 - 2 * x + 5 = 0 ∧ m * y^2 - 2 * y + 5 = 0 :=
sorry

end quadratic_distinct_real_roots_l152_152214


namespace square_side_length_l152_152163

theorem square_side_length (s : ℝ) (h : s^2 = 1/9) : s = 1/3 :=
sorry

end square_side_length_l152_152163


namespace cats_left_l152_152757

def initial_siamese_cats : ℕ := 12
def initial_house_cats : ℕ := 20
def cats_sold : ℕ := 20

theorem cats_left : (initial_siamese_cats + initial_house_cats - cats_sold) = 12 :=
by
sorry

end cats_left_l152_152757


namespace pencil_eraser_cost_l152_152578

theorem pencil_eraser_cost (p e : ℕ) (hp : p > e) (he : e > 0)
  (h : 20 * p + 4 * e = 160) : p + e = 12 :=
sorry

end pencil_eraser_cost_l152_152578


namespace shadow_length_building_l152_152329

theorem shadow_length_building:
  let height_flagstaff := 17.5
  let shadow_flagstaff := 40.25
  let height_building := 12.5
  let expected_shadow_building := 28.75
  (height_flagstaff / shadow_flagstaff = height_building / expected_shadow_building) := by
  let height_flagstaff := 17.5
  let shadow_flagstaff := 40.25
  let height_building := 12.5
  let expected_shadow_building := 28.75
  sorry

end shadow_length_building_l152_152329


namespace sin_alpha_third_quadrant_l152_152485

theorem sin_alpha_third_quadrant 
  (α : ℝ) 
  (hcos : Real.cos α = -3 / 5) 
  (hquad : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.sin α = -4 / 5 := 
sorry

end sin_alpha_third_quadrant_l152_152485


namespace greatest_integer_less_PS_l152_152240

theorem greatest_integer_less_PS 
  (PQ PS T : ℝ)
  (midpoint_TPS : T = PS / 2)
  (perpendicular_PT_QT : (PQ ^ 2 = (PS / 2) ^ 2 + (PS / 2) ^ 2))
  (PQ_value : PQ = 150) :
  ⌊ PS ⌋ = 212 :=
by
  sorry

end greatest_integer_less_PS_l152_152240


namespace find_taller_tree_height_l152_152772

-- Define the known variables and conditions
variables (H : ℕ) (ratio : ℚ) (difference : ℕ)

-- Specify the conditions from the problem
def taller_tree_height (H difference : ℕ) := H
def shorter_tree_height (H difference : ℕ) := H - difference
def height_ratio (H : ℕ) (ratio : ℚ) (difference : ℕ) :=
  (shorter_tree_height H difference : ℚ) / (taller_tree_height H difference : ℚ) = ratio

-- Prove the height of the taller tree given the conditions
theorem find_taller_tree_height (H : ℕ) (h_ratio : height_ratio H (2/3) 20) : 
  taller_tree_height H 20 = 60 :=
  sorry

end find_taller_tree_height_l152_152772


namespace unit_digit_G_1000_l152_152913

def G (n : ℕ) : ℕ := 3^(3^n) + 1

theorem unit_digit_G_1000 : (G 1000) % 10 = 2 :=
by
  sorry

end unit_digit_G_1000_l152_152913


namespace solve_abs_equation_l152_152563

theorem solve_abs_equation (y : ℤ) : (|y - 8| + 3 * y = 12) ↔ (y = 2) :=
by
  sorry  -- skip the proof steps.

end solve_abs_equation_l152_152563


namespace simplify_and_evaluate_expression_l152_152523

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end simplify_and_evaluate_expression_l152_152523


namespace inequalities_count_three_l152_152157

theorem inequalities_count_three
  (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : x^2 < a^2) (h2 : y^3 < b^3) :
  (x^2 + y^2 < a^2 + b^2) ∧ ¬(x^2 - y^2 < a^2 - b^2) ∧ (x^2 * y^3 < a^2 * b^3) ∧ (x^2 / y^3 < a^2 / b^3) := 
sorry

end inequalities_count_three_l152_152157


namespace coloring_count_l152_152676

theorem coloring_count (n : ℕ) (h : 0 < n) :
  ∃ (num_colorings : ℕ), num_colorings = 2 :=
sorry

end coloring_count_l152_152676


namespace find_z_plus_inverse_y_l152_152496

theorem find_z_plus_inverse_y
  (x y z : ℝ)
  (h1 : x * y * z = 1)
  (h2 : x + 1/z = 10)
  (h3 : y + 1/x = 5) :
  z + 1/y = 17 / 49 :=
by
  sorry

end find_z_plus_inverse_y_l152_152496


namespace cats_remaining_proof_l152_152384

def initial_siamese : ℕ := 38
def initial_house : ℕ := 25
def sold_cats : ℕ := 45

def total_cats (s : ℕ) (h : ℕ) : ℕ := s + h
def remaining_cats (total : ℕ) (sold : ℕ) : ℕ := total - sold

theorem cats_remaining_proof : remaining_cats (total_cats initial_siamese initial_house) sold_cats = 18 :=
by
  sorry

end cats_remaining_proof_l152_152384


namespace find_n_for_geometric_series_l152_152921

theorem find_n_for_geometric_series
  (n : ℝ)
  (a1 : ℝ := 12)
  (a2 : ℝ := 4)
  (r1 : ℝ)
  (S1 : ℝ)
  (b1 : ℝ := 12)
  (b2 : ℝ := 4 + n)
  (r2 : ℝ)
  (S2 : ℝ) :
  (r1 = a2 / a1) →
  (S1 = a1 / (1 - r1)) →
  (S2 = 4 * S1) →
  (r2 = b2 / b1) →
  (S2 = b1 / (1 - r2)) →
  n = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_n_for_geometric_series_l152_152921


namespace divisor_increase_by_10_5_l152_152317

def condition_one (n t : ℕ) : Prop :=
  n * (t + 7) = t * (n + 2)

def condition_two (n t z : ℕ) : Prop :=
  n * (t + z) = t * (n + 3)

theorem divisor_increase_by_10_5 (n t : ℕ) (hz : ℕ) (nz : n ≠ 0) (tz : t ≠ 0)
  (h1 : condition_one n t) (h2 : condition_two n t hz) : hz = 21 / 2 :=
by {
  sorry
}

end divisor_increase_by_10_5_l152_152317


namespace female_voters_percentage_is_correct_l152_152089

def percentage_of_population_that_are_female_voters
  (female_percentage : ℝ)
  (voter_percentage_of_females : ℝ) : ℝ :=
  female_percentage * voter_percentage_of_females * 100

theorem female_voters_percentage_is_correct :
  percentage_of_population_that_are_female_voters 0.52 0.4 = 20.8 := by
  sorry

end female_voters_percentage_is_correct_l152_152089


namespace product_of_consecutive_integers_l152_152908

theorem product_of_consecutive_integers (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_less : a < b) :
  ∃ (x y : ℕ), x ≠ y ∧ x * y % (a * b) = 0 :=
by
  sorry

end product_of_consecutive_integers_l152_152908


namespace minimum_x_plus_3y_l152_152873

theorem minimum_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 3 * x + y) : x + 3 * y ≥ 16 :=
sorry

end minimum_x_plus_3y_l152_152873


namespace final_answer_l152_152326

theorem final_answer : (848 / 8) - 100 = 6 := 
by
  sorry

end final_answer_l152_152326


namespace find_ratio_PS_SR_l152_152736

variable {P Q R S : Type}
variable [MetricSpace P]
variable [MetricSpace Q]
variable [MetricSpace R]
variable [MetricSpace S]

-- Given conditions
variable (PQ QR PR : ℝ)
variable (hPQ : PQ = 6)
variable (hQR : QR = 8)
variable (hPR : PR = 10)
variable (QS : ℝ)
variable (hQS : QS = 6)

-- Points on the segments
variable (PS : ℝ)
variable (SR : ℝ)

-- The theorem to be proven: the ratio PS : SR = 0 : 1
theorem find_ratio_PS_SR (hPQ : PQ = 6) (hQR : QR = 8) (hPR : PR = 10) (hQS : QS = 6) :
    PS = 0 ∧ SR = 10 → PS / SR = 0 :=
by
  sorry

end find_ratio_PS_SR_l152_152736


namespace numbers_in_circle_are_zero_l152_152609

theorem numbers_in_circle_are_zero (a : Fin 55 → ℤ) 
  (h : ∀ i, a i = a ((i + 54) % 55) + a ((i + 1) % 55)) : 
  ∀ i, a i = 0 := 
by
  sorry

end numbers_in_circle_are_zero_l152_152609


namespace water_left_after_operations_l152_152539

theorem water_left_after_operations :
  let initial_water := (3 : ℚ)
  let water_used := (4 / 3 : ℚ)
  let extra_water := (1 / 2 : ℚ)
  initial_water - water_used + extra_water = (13 / 6 : ℚ) := 
by
  -- Skips the proof, as the focus is on the problem statement
  sorry

end water_left_after_operations_l152_152539


namespace parabola_vertex_x_coord_l152_152310

theorem parabola_vertex_x_coord (a b c : ℝ)
  (h1 : 5 = a * 2^2 + b * 2 + c)
  (h2 : 5 = a * 8^2 + b * 8 + c)
  (h3 : 11 = a * 9^2 + b * 9 + c) :
  5 = (2 + 8) / 2 := 
sorry

end parabola_vertex_x_coord_l152_152310


namespace Lenora_scored_30_points_l152_152390

variable (x y : ℕ)
variable (hx : x + y = 40)
variable (three_point_success_rate : ℚ := 25 / 100)
variable (free_throw_success_rate : ℚ := 50 / 100)
variable (points_three_point : ℚ := 3)
variable (points_free_throw : ℚ := 1)
variable (three_point_contribution : ℚ := three_point_success_rate * points_three_point * x)
variable (free_throw_contribution : ℚ := free_throw_success_rate * points_free_throw * y)
variable (total_points : ℚ := three_point_contribution + free_throw_contribution)

theorem Lenora_scored_30_points : total_points = 30 :=
by
  sorry

end Lenora_scored_30_points_l152_152390


namespace range_of_a_l152_152525

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^3 - a) → (∀ x : ℝ, f 0 ≤ 0) → (0 ≤ a) :=
by
  intro h1 h2
  suffices h : -a ≤ 0 by
    simpa using h
  have : f 0 = -a
  simp [h1]
  sorry -- Proof steps are omitted

end range_of_a_l152_152525


namespace numberOfAntiPalindromes_l152_152315

-- Define what it means for a number to be an anti-palindrome in base 3
def isAntiPalindrome (n : ℕ) : Prop :=
  ∀ (a b : ℕ), a + b = 2 → a ≠ b

-- Define the constraint of no two consecutive digits being the same
def noConsecutiveDigits (digits : List ℕ) : Prop :=
  ∀ (i : ℕ), i < digits.length - 1 → digits.nthLe i sorry ≠ digits.nthLe (i + 1) sorry

-- We want to find the number of anti-palindromes less than 3^12 fulfilling both conditions
def countAntiPalindromes (m : ℕ) (base : ℕ) : ℕ :=
  sorry -- Placeholder definition for the count, to be implemented

-- The main theorem to prove
theorem numberOfAntiPalindromes : countAntiPalindromes (3^12) 3 = 126 :=
  sorry -- Proof to be filled

end numberOfAntiPalindromes_l152_152315


namespace possible_scenario_l152_152081

variable {a b c d : ℝ}

-- Conditions
def abcd_positive : a * b * c * d > 0 := sorry
def a_less_than_c : a < c := sorry
def bcd_negative : b * c * d < 0 := sorry

-- Statement
theorem possible_scenario :
  (a < 0) ∧ (b > 0) ∧ (c < 0) ∧ (d > 0) :=
sorry

end possible_scenario_l152_152081


namespace train_platform_length_l152_152423

noncomputable def kmph_to_mps (v : ℕ) : ℕ := v * 1000 / 3600

theorem train_platform_length :
  ∀ (train_length speed_kmph time_sec : ℕ),
    speed_kmph = 36 →
    train_length = 175 →
    time_sec = 40 →
    let speed_mps := kmph_to_mps speed_kmph
    let total_distance := speed_mps * time_sec
    let platform_length := total_distance - train_length
    platform_length = 225 :=
by
  intros train_length speed_kmph time_sec h_speed h_train h_time
  let speed_mps := kmph_to_mps speed_kmph
  let total_distance := speed_mps * time_sec
  let platform_length := total_distance - train_length
  sorry

end train_platform_length_l152_152423


namespace recurring_decimal_as_fraction_l152_152004

theorem recurring_decimal_as_fraction :
  0.53 + (247 / 999) * 0.001 = 53171 / 99900 :=
by
  sorry

end recurring_decimal_as_fraction_l152_152004


namespace discriminant_of_quadratic_eq_l152_152834

-- Define the coefficients of the quadratic equation
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 1

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

-- State the theorem that we want to prove
theorem discriminant_of_quadratic_eq : discriminant a b c = 61 := by
  sorry

end discriminant_of_quadratic_eq_l152_152834


namespace union_of_M_N_is_real_set_l152_152438

-- Define the set M
def M : Set ℝ := { x | x^2 + 3 * x + 2 > 0 }

-- Define the set N
def N : Set ℝ := { x | (1 / 2 : ℝ) ^ x ≤ 4 }

-- The goal is to prove that the union of M and N is the set of all real numbers
theorem union_of_M_N_is_real_set : M ∪ N = Set.univ :=
by
  sorry

end union_of_M_N_is_real_set_l152_152438


namespace base_six_to_base_ten_equivalent_l152_152735

theorem base_six_to_base_ten_equivalent :
  let n := 12345
  (5 * 6^0 + 4 * 6^1 + 3 * 6^2 + 2 * 6^3 + 1 * 6^4) = 1865 :=
by
  sorry

end base_six_to_base_ten_equivalent_l152_152735


namespace number_of_sophomores_l152_152235

-- Definition of the conditions
variables (J S P j s p : ℕ)

-- Condition: Equal number of students in debate team
def DebateTeam_Equal : Prop := j = s ∧ s = p

-- Condition: Total number of students
def TotalStudents : Prop := J + S + P = 45

-- Condition: Percentage relationships
def PercentRelations_J : Prop := j = J / 5
def PercentRelations_S : Prop := s = 3 * S / 20
def PercentRelations_P : Prop := p = P / 10

-- The main theorem to prove
theorem number_of_sophomores : DebateTeam_Equal j s p 
                               → TotalStudents J S P 
                               → PercentRelations_J J j 
                               → PercentRelations_S S s 
                               → PercentRelations_P P p 
                               → P = 21 :=
by 
  sorry

end number_of_sophomores_l152_152235


namespace log_mul_l152_152183

theorem log_mul (a M N : ℝ) (ha_pos : 0 < a) (hM_pos : 0 < M) (hN_pos : 0 < N) (ha_ne_one : a ≠ 1) :
    Real.log (M * N) / Real.log a = Real.log M / Real.log a + Real.log N / Real.log a := by
  sorry

end log_mul_l152_152183


namespace no_adjacent_performers_probability_l152_152529

-- A definition to model the probability of non-adjacent performers in a circle of 6 people.
def probability_no_adjacent_performers : ℚ :=
  -- Given conditions: fair coin tosses by six people, modeling permutations
  -- and specific valid configurations derived from the problem.
  9 / 32

-- Proving the final probability calculation is correct
theorem no_adjacent_performers_probability :
  probability_no_adjacent_performers = 9 / 32 :=
by
  -- Using sorry to indicate the proof needs to be filled in, acknowledging the correct answer.
  sorry

end no_adjacent_performers_probability_l152_152529


namespace distance_from_center_to_tangent_chord_l152_152260

theorem distance_from_center_to_tangent_chord
  (R a m x : ℝ)
  (h1 : m^2 = 4 * R^2)
  (h2 : 16 * R^2 * x^4 - 16 * R^2 * x^2 * (a^2 + R^2) + 16 * a^4 * R^4 - a^2 * (4 * R^2 - m^2)^2 = 0) :
  x = R :=
sorry

end distance_from_center_to_tangent_chord_l152_152260


namespace dot_product_parallel_a_b_l152_152099

noncomputable def a : ℝ × ℝ := (-1, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Definition of parallel vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v2 = (k * v1.1, k * v1.2)

-- Given conditions and result to prove
theorem dot_product_parallel_a_b : ∀ (x : ℝ), parallel a (b x) → x = -2 → (a.1 * (b x).1 + a.2 * (b x).2) = -4 := 
by
  intros x h_parallel h_x
  subst h_x
  sorry

end dot_product_parallel_a_b_l152_152099


namespace exists_t_for_f_inequality_l152_152902

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * (x + 1) ^ 2

theorem exists_t_for_f_inequality :
  ∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → f (x + t) ≤ x := by
  sorry

end exists_t_for_f_inequality_l152_152902


namespace find_radius_k_l152_152478

/-- Mathematical conditions for the given geometry problem -/
structure problem_conditions where
  radius_F : ℝ := 15
  radius_G : ℝ := 4
  radius_H : ℝ := 3
  radius_I : ℝ := 3
  radius_J : ℝ := 1

/-- Proof problem statement defining the required theorem -/
theorem find_radius_k (conditions : problem_conditions) :
  let r := (137:ℝ) / 8
  20 * r = (342.5 : ℝ) :=
by
  sorry

end find_radius_k_l152_152478


namespace lcm_36_98_is_1764_l152_152101

theorem lcm_36_98_is_1764 : Nat.lcm 36 98 = 1764 := by
  sorry

end lcm_36_98_is_1764_l152_152101


namespace boy_usual_time_reach_school_l152_152763

theorem boy_usual_time_reach_school (R T : ℝ) (h : (7 / 6) * R * (T - 3) = R * T) : T = 21 := by
  sorry

end boy_usual_time_reach_school_l152_152763


namespace expected_number_of_heads_after_flips_l152_152045

theorem expected_number_of_heads_after_flips :
  let p_heads_after_tosses : ℚ := (1/3) + (2/9) + (4/27) + (8/81)
  let expected_heads : ℚ := 100 * p_heads_after_tosses
  expected_heads = 6500 / 81 :=
by
  let p_heads_after_tosses : ℚ := (1/3) + (2/9) + (4/27) + (8/81)
  let expected_heads : ℚ := 100 * p_heads_after_tosses
  show expected_heads = (6500 / 81)
  sorry

end expected_number_of_heads_after_flips_l152_152045


namespace first_character_more_lines_than_second_l152_152511

theorem first_character_more_lines_than_second :
  let x := 2
  let second_character_lines := 3 * x + 6
  20 - second_character_lines = 8 := by
  sorry

end first_character_more_lines_than_second_l152_152511


namespace frood_points_l152_152580

theorem frood_points (n : ℕ) (h : n > 29) : (n * (n + 1) / 2) > 15 * n := by
  sorry

end frood_points_l152_152580


namespace card_area_l152_152642

theorem card_area (length width : ℕ) (h_length : length = 5) (h_width : width = 7)
  (h_area_after_shortening : (length - 1) * width = 24 ∨ length * (width - 1) = 24) :
  length * (width - 1) = 18 :=
by
  sorry

end card_area_l152_152642


namespace radius_of_circle_B_l152_152957

theorem radius_of_circle_B (diam_A : ℝ) (factor : ℝ) (r_A r_B : ℝ) 
  (h1 : diam_A = 80) 
  (h2 : r_A = diam_A / 2) 
  (h3 : r_A = factor * r_B) 
  (h4 : factor = 4) : r_B = 10 := 
by 
  sorry

end radius_of_circle_B_l152_152957


namespace minute_hand_angle_45min_l152_152684

theorem minute_hand_angle_45min
  (duration : ℝ)
  (h1 : duration = 45) :
  (-(3 / 4) * 2 * Real.pi = - (3 * Real.pi / 2)) :=
by
  sorry

end minute_hand_angle_45min_l152_152684


namespace final_price_after_discounts_l152_152226

noncomputable def initial_price : ℝ := 9795.3216374269
noncomputable def discount_20 (p : ℝ) : ℝ := p * 0.80
noncomputable def discount_10 (p : ℝ) : ℝ := p * 0.90
noncomputable def discount_5 (p : ℝ) : ℝ := p * 0.95

theorem final_price_after_discounts : discount_5 (discount_10 (discount_20 initial_price)) = 6700 := 
by
  sorry

end final_price_after_discounts_l152_152226
