import Mathlib

namespace symmetric_point_line_l253_25303

theorem symmetric_point_line (a b : ℝ) :
  (∀ (x y : ℝ), (y - 2) / (x - 1) = -2 → (x + 1)/2 + 2 * (y + 2)/2 - 10 = 0) →
  a = 3 ∧ b = 6 := by
  intro h
  sorry

end symmetric_point_line_l253_25303


namespace isosceles_triangle_base_length_l253_25329

theorem isosceles_triangle_base_length
  (a b c: ℕ) 
  (h_iso: a = b ∨ a = c ∨ b = c)
  (h_perimeter: a + b + c = 21)
  (h_side: a = 5 ∨ b = 5 ∨ c = 5) :
  c = 5 :=
by
  sorry

end isosceles_triangle_base_length_l253_25329


namespace opposite_of_neg_two_is_two_l253_25379

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l253_25379


namespace arithmetic_sequence_general_formula_l253_25356

noncomputable def a_n (n : ℕ) : ℝ :=
sorry

theorem arithmetic_sequence_general_formula (h1 : (a_n 2 + a_n 6) / 2 = 5)
                                            (h2 : (a_n 3 + a_n 7) / 2 = 7) :
  a_n n = 2 * (n : ℝ) - 3 :=
sorry

end arithmetic_sequence_general_formula_l253_25356


namespace volume_of_increased_box_l253_25394

theorem volume_of_increased_box {l w h : ℝ} (vol : l * w * h = 4860) (sa : l * w + w * h + l * h = 930) (sum_dim : l + w + h = 56) :
  (l + 2) * (w + 3) * (h + 1) = 5964 :=
by
  sorry

end volume_of_increased_box_l253_25394


namespace total_pets_remaining_l253_25388

def initial_counts := (7, 6, 4, 5, 3)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def morning_sales := (1, 2, 1, 0, 0)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def afternoon_sales := (1, 1, 2, 3, 0)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)
def returns := (0, 1, 0, 1, 1)  -- (puppies, kittens, rabbits, guinea pigs, chameleons)

def calculate_remaining (initial_counts morning_sales afternoon_sales returns : Nat × Nat × Nat × Nat × Nat) : Nat :=
  let (p0, k0, r0, g0, c0) := initial_counts
  let (p1, k1, r1, g1, c1) := morning_sales
  let (p2, k2, r2, g2, c2) := afternoon_sales
  let (p3, k3, r3, g3, c3) := returns
  let remaining_puppies := p0 - p1 - p2 + p3
  let remaining_kittens := k0 - k1 - k2 + k3
  let remaining_rabbits := r0 - r1 - r2 + r3
  let remaining_guinea_pigs := g0 - g1 - g2 + g3
  let remaining_chameleons := c0 - c1 - c2 + c3
  remaining_puppies + remaining_kittens + remaining_rabbits + remaining_guinea_pigs + remaining_chameleons

theorem total_pets_remaining : calculate_remaining initial_counts morning_sales afternoon_sales returns = 15 := 
by
  simp [initial_counts, morning_sales, afternoon_sales, returns, calculate_remaining]
  sorry

end total_pets_remaining_l253_25388


namespace pen_defect_probability_l253_25319

theorem pen_defect_probability :
  ∀ (n m : ℕ) (k : ℚ), n = 12 → m = 4 → k = 2 → 
  (8 / 12) * (7 / 11) = 141 / 330 := 
by
  intros n m k h1 h2 h3
  sorry

end pen_defect_probability_l253_25319


namespace solve_for_x_l253_25346

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 15 / (x / 3)) : x = 15 * Real.sqrt 3 ∨ x = -15 * Real.sqrt 3 :=
by
  sorry

end solve_for_x_l253_25346


namespace comic_books_exclusive_count_l253_25393

theorem comic_books_exclusive_count 
  (shared_comics : ℕ) 
  (total_andrew_comics : ℕ) 
  (john_exclusive_comics : ℕ) 
  (h_shared_comics : shared_comics = 15) 
  (h_total_andrew_comics : total_andrew_comics = 22) 
  (h_john_exclusive_comics : john_exclusive_comics = 10) : 
  (total_andrew_comics - shared_comics + john_exclusive_comics = 17) := by 
  sorry

end comic_books_exclusive_count_l253_25393


namespace domain_translation_l253_25381

theorem domain_translation (f : ℝ → ℝ) :
  (∀ x : ℝ, 0 < 3 * x + 2 ∧ 3 * x + 2 < 1 → (∃ y : ℝ, f (3 * x + 2) = y)) →
  (∀ x : ℝ, ∃ y : ℝ, f (2 * x - 1) = y ↔ (3 / 2) < x ∧ x < 3) :=
sorry

end domain_translation_l253_25381


namespace complex_mul_im_unit_l253_25386

theorem complex_mul_im_unit (i : ℂ) (h : i^2 = -1) : i * (1 - i) = 1 + i := by
  sorry

end complex_mul_im_unit_l253_25386


namespace inequality_8xyz_l253_25396

theorem inequality_8xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) : 
  (1 - x) * (1 - y) * (1 - z) > 8 * x * y * z := 
  by sorry

end inequality_8xyz_l253_25396


namespace fraction_sum_lt_one_l253_25322

theorem fraction_sum_lt_one (n : ℕ) (h_pos : n > 0) : 
  (1 / 2 + 1 / 3 + 1 / 10 + 1 / n < 1) ↔ (n > 15) :=
sorry

end fraction_sum_lt_one_l253_25322


namespace find_k_l253_25339

theorem find_k (Z K : ℤ) (h1 : 2000 < Z) (h2 : Z < 3000) (h3 : K > 1) (h4 : Z = K * K^2) (h5 : ∃ n : ℤ, n^3 = Z) : K = 13 :=
by
-- Solution omitted
sorry

end find_k_l253_25339


namespace steven_card_count_l253_25309

theorem steven_card_count (num_groups : ℕ) (cards_per_group : ℕ) (h_groups : num_groups = 5) (h_cards : cards_per_group = 6) : num_groups * cards_per_group = 30 := by
  sorry

end steven_card_count_l253_25309


namespace ada_original_seat_l253_25365

theorem ada_original_seat (seats: Fin 6 → Option String)
  (Bea_init Ceci_init Dee_init Edie_init Fran_init: Fin 6) 
  (Bea_fin Ceci_fin Fran_fin: Fin 6) 
  (Ada_fin: Fin 6)
  (Bea_moves_right: Bea_fin = Bea_init + 3)
  (Ceci_stays: Ceci_fin = Ceci_init)
  (Dee_switches_with_Edie: ∃ Dee_fin Edie_fin: Fin 6, Dee_fin = Edie_init ∧ Edie_fin = Dee_init)
  (Fran_moves_left: Fran_fin = Fran_init - 1)
  (Ada_end_seat: Ada_fin = 0 ∨ Ada_fin = 5):
  ∃ Ada_init: Fin 6, Ada_init = 2 + Ada_fin + 1 → Ada_init = 3 := 
by 
  sorry

end ada_original_seat_l253_25365


namespace problem_solution_l253_25352

theorem problem_solution :
  ∃ (b₂ b₃ b₄ b₅ b₆ b₇ : ℤ),
    (0 ≤ b₂ ∧ b₂ < 2) ∧
    (0 ≤ b₃ ∧ b₃ < 3) ∧
    (0 ≤ b₄ ∧ b₄ < 4) ∧
    (0 ≤ b₅ ∧ b₅ < 5) ∧
    (0 ≤ b₆ ∧ b₆ < 6) ∧
    (0 ≤ b₇ ∧ b₇ < 8) ∧
    (6 / 7 = b₂ / 2 + b₃ / 6 + b₄ / 24 + b₅ / 120 + b₆ / 720 + b₇ / 5040) ∧
    (b₂ + b₃ + b₄ + b₅ + b₆ + b₇ = 11) :=
sorry

end problem_solution_l253_25352


namespace find_a_in_triangle_l253_25377

theorem find_a_in_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : c = 3)
  (h2 : C = Real.pi / 3)
  (h3 : Real.sin B = 2 * Real.sin A)
  (h4 : a = 3) :
  a = Real.sqrt 3 := by
  sorry

end find_a_in_triangle_l253_25377


namespace place_two_after_three_digit_number_l253_25349

theorem place_two_after_three_digit_number (h t u : ℕ) 
  (Hh : h < 10) (Ht : t < 10) (Hu : u < 10) : 
  (100 * h + 10 * t + u) * 10 + 2 = 1000 * h + 100 * t + 10 * u + 2 := 
by
  sorry

end place_two_after_three_digit_number_l253_25349


namespace parabola_centroid_locus_l253_25350

/-- Let P_0 be a parabola defined by the equation y = m * x^2. 
    Let A and B be points on P_0 such that the tangents at A and B are perpendicular. 
    Let G be the centroid of the triangle formed by A, B, and the vertex of P_0.
    Let P_n be the nth derived parabola.
    Prove that the equation of P_n is y = 3^n * m * x^2 + (1 / (4 * m)) * (1 - (1 / 3)^n). -/
theorem parabola_centroid_locus (n : ℕ) (m : ℝ) 
  (h_pos_m : 0 < m) :
  ∃ P_n : ℝ → ℝ, 
    ∀ x : ℝ, P_n x = 3^n * m * x^2 + (1 / (4 * m)) * (1 - (1 / 3)^n) :=
sorry

end parabola_centroid_locus_l253_25350


namespace Greg_harvested_acres_l253_25351

-- Defining the conditions
def Sharon_harvested : ℝ := 0.1
def Greg_harvested (additional: ℝ) (Sharon: ℝ) : ℝ := Sharon + additional

-- Proving the statement
theorem Greg_harvested_acres : Greg_harvested 0.3 Sharon_harvested = 0.4 :=
by
  sorry

end Greg_harvested_acres_l253_25351


namespace matt_total_vibrations_l253_25360

noncomputable def vibrations_lowest : ℕ := 1600
noncomputable def vibrations_highest : ℕ := vibrations_lowest + (6 * vibrations_lowest / 10)
noncomputable def time_seconds : ℕ := 300
noncomputable def total_vibrations : ℕ := vibrations_highest * time_seconds

theorem matt_total_vibrations :
  total_vibrations = 768000 := by
  sorry

end matt_total_vibrations_l253_25360


namespace tayzia_tip_l253_25323

theorem tayzia_tip (haircut_women : ℕ) (haircut_children : ℕ) (num_women : ℕ) (num_children : ℕ) (tip_percentage : ℕ) :
  ((num_women * haircut_women + num_children * haircut_children) * tip_percentage / 100) = 24 :=
by
  -- Given conditions
  let haircut_women := 48
  let haircut_children := 36
  let num_women := 1
  let num_children := 2
  let tip_percentage := 20
  -- Perform the calculations as shown in the solution steps
  sorry

end tayzia_tip_l253_25323


namespace min_a2_plus_b2_quartic_eq_l253_25325

theorem min_a2_plus_b2_quartic_eq (a b : ℝ) (x : ℝ) 
  (h : x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  a^2 + b^2 ≥ 4/5 := 
sorry

end min_a2_plus_b2_quartic_eq_l253_25325


namespace seq_general_formula_l253_25347

def seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ a n ^ 2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0

theorem seq_general_formula {a : ℕ → ℝ} (h1 : a 1 = 1) (h2 : seq a) :
  ∀ n, a n = 1 / 2 ^ (n - 1) :=
by
  sorry

end seq_general_formula_l253_25347


namespace geometric_sequence_common_ratio_l253_25359

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 3 = 2 * S 2 + 1) (h2 : a 4 = 2 * S 3 + 1) :
  ∃ q : ℝ, (q = 3) :=
by
  -- Proof will go here.
  sorry

end geometric_sequence_common_ratio_l253_25359


namespace distinct_arrangements_of_pebbles_in_octagon_l253_25301

noncomputable def number_of_distinct_arrangements : ℕ :=
  (Nat.factorial 8) / 16

theorem distinct_arrangements_of_pebbles_in_octagon : 
  number_of_distinct_arrangements = 2520 :=
by
  sorry

end distinct_arrangements_of_pebbles_in_octagon_l253_25301


namespace slope_of_perpendicular_line_l253_25300

theorem slope_of_perpendicular_line (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  ∃ m : ℝ, a * x - b * y = c → m = - (b / a) :=
by
  -- Here we state the definition and conditions provided in the problem
  -- And indicate what we want to prove (that the slope is -b/a in this case)
  sorry

end slope_of_perpendicular_line_l253_25300


namespace stratified_sampling_sample_size_l253_25382

-- Definitions based on conditions
def total_employees : ℕ := 120
def male_employees : ℕ := 90
def female_employees_in_sample : ℕ := 3

-- Proof statement
theorem stratified_sampling_sample_size : total_employees = 120 ∧ male_employees = 90 ∧ female_employees_in_sample = 3 → 
  (female_employees_in_sample + female_employees_in_sample * (male_employees / (total_employees - male_employees))) = 12 :=
sorry

end stratified_sampling_sample_size_l253_25382


namespace asha_wins_probability_l253_25392

variable (p_lose p_tie p_win : ℚ)

theorem asha_wins_probability 
  (h_lose : p_lose = 3 / 7) 
  (h_tie : p_tie = 1 / 7) 
  (h_total : p_win + p_lose + p_tie = 1) : 
  p_win = 3 / 7 := by
  sorry

end asha_wins_probability_l253_25392


namespace Frank_seeds_per_orange_l253_25383

noncomputable def Betty_oranges := 15
noncomputable def Bill_oranges := 12
noncomputable def total_oranges := Betty_oranges + Bill_oranges
noncomputable def Frank_oranges := 3 * total_oranges
noncomputable def oranges_per_tree := 5
noncomputable def Philip_oranges := 810
noncomputable def number_of_trees := Philip_oranges / oranges_per_tree
noncomputable def seeds_per_orange := number_of_trees / Frank_oranges

theorem Frank_seeds_per_orange :
  seeds_per_orange = 2 :=
by
  sorry

end Frank_seeds_per_orange_l253_25383


namespace H_double_prime_coordinates_l253_25342

/-- Define the points of the parallelogram EFGH and their reflections. --/
structure Point := (x : ℝ) (y : ℝ)

def E : Point := ⟨3, 4⟩
def F : Point := ⟨5, 7⟩
def G : Point := ⟨7, 4⟩
def H : Point := ⟨5, 1⟩

/-- Reflection of a point across the x-axis changes the y-coordinate sign. --/
def reflect_x (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Reflection of a point across y=x-1 involves translation and reflection across y=x. --/
def reflect_y_x_minus_1 (p : Point) : Point :=
  let translated := Point.mk p.x (p.y + 1)
  let reflected := Point.mk translated.y translated.x
  Point.mk reflected.x (reflected.y - 1)

def H' : Point := reflect_x H
def H'' : Point := reflect_y_x_minus_1 H'

theorem H_double_prime_coordinates : H'' = ⟨0, 4⟩ :=
by
  sorry

end H_double_prime_coordinates_l253_25342


namespace right_triangle_leg_square_l253_25320

theorem right_triangle_leg_square (a b c : ℝ) 
  (h1 : c = a + 2) 
  (h2 : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 := 
by
  sorry

end right_triangle_leg_square_l253_25320


namespace original_price_of_good_l253_25374

theorem original_price_of_good (P : ℝ) (h1 : 0.684 * P = 6840) : P = 10000 :=
sorry

end original_price_of_good_l253_25374


namespace sales_value_minimum_l253_25370

theorem sales_value_minimum (V : ℝ) (base_salary new_salary : ℝ) (commission_rate sales_needed old_salary : ℝ)
    (h_base_salary : base_salary = 45000 )
    (h_new_salary : new_salary = base_salary + 0.15 * V * sales_needed)
    (h_sales_needed : sales_needed = 266.67)
    (h_old_salary : old_salary = 75000) :
    new_salary ≥ old_salary ↔ V ≥ 750 := 
by
  sorry

end sales_value_minimum_l253_25370


namespace solution_exists_l253_25372

noncomputable def verifySolution (x y z : ℝ) : Prop := 
  x^2 - y = (z - 1)^2 ∧
  y^2 - z = (x - 1)^2 ∧
  z^2 - x = (y- 1)^2 

theorem solution_exists (x y z : ℝ) (h : verifySolution x y z) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x, y, z) = (-2.93122, 2.21124, 0.71998) ∨ 
  (x, y, z) = (2.21124, 0.71998, -2.93122) ∨ 
  (x, y, z) = (0.71998, -2.93122, 2.21124) :=
sorry

end solution_exists_l253_25372


namespace min_value_x_2y_l253_25369

theorem min_value_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y + 2 * x * y = 8) : x + 2 * y ≥ 4 :=
sorry

end min_value_x_2y_l253_25369


namespace fraction_same_ratio_l253_25321

theorem fraction_same_ratio (x : ℚ) : 
  (x / (2 / 5)) = (3 / 7) / (6 / 5) ↔ x = 1 / 7 :=
by
  sorry

end fraction_same_ratio_l253_25321


namespace brown_gumdrops_after_replacement_l253_25348

theorem brown_gumdrops_after_replacement
  (total_gumdrops : ℕ)
  (percent_blue : ℚ)
  (percent_brown : ℚ)
  (percent_red : ℚ)
  (percent_yellow : ℚ)
  (num_green : ℕ)
  (replace_half_blue_with_brown : ℕ) :
  total_gumdrops = 120 →
  percent_blue = 0.30 →
  percent_brown = 0.20 →
  percent_red = 0.15 →
  percent_yellow = 0.10 →
  num_green = 30 →
  replace_half_blue_with_brown = 18 →
  ((percent_brown * ↑total_gumdrops) + replace_half_blue_with_brown) = 42 :=
by sorry

end brown_gumdrops_after_replacement_l253_25348


namespace total_books_l253_25331

def sam_books : ℕ := 110
def joan_books : ℕ := 102

theorem total_books : sam_books + joan_books = 212 := by
  sorry

end total_books_l253_25331


namespace total_distance_traveled_l253_25345

theorem total_distance_traveled:
  let speed1 := 30
  let time1 := 4
  let speed2 := 35
  let time2 := 5
  let speed3 := 25
  let time3 := 6
  let total_time := 20
  let time1_3 := time1 + time2 + time3
  let time4 := total_time - time1_3
  let speed4 := 40

  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4

  let total_distance := distance1 + distance2 + distance3 + distance4

  total_distance = 645 :=
  sorry

end total_distance_traveled_l253_25345


namespace solve_equation_l253_25315

theorem solve_equation (x : ℝ) (floor : ℝ → ℤ) 
  (h_floor : ∀ y, floor y ≤ y ∧ y < floor y + 1) :
  (floor (20 * x + 23) = 20 + 23 * x) ↔ 
  (∃ n : ℤ, 20 ≤ n ∧ n ≤ 43 ∧ x = (n - 23) / 20) := 
by
  sorry

end solve_equation_l253_25315


namespace rectangular_solid_surface_area_l253_25327

theorem rectangular_solid_surface_area (a b c : ℕ) (h₁ : Prime a ∨ ∃ p : ℕ, Prime p ∧ a = p + (p + 1))
                                         (h₂ : Prime b ∨ ∃ q : ℕ, Prime q ∧ b = q + (q + 1))
                                         (h₃ : Prime c ∨ ∃ r : ℕ, Prime r ∧ c = r + (r + 1))
                                         (h₄ : a * b * c = 399) :
  2 * (a * b + b * c + c * a) = 422 := 
sorry

end rectangular_solid_surface_area_l253_25327


namespace gcd_max_possible_value_l253_25390

theorem gcd_max_possible_value (x y : ℤ) (h_coprime : Int.gcd x y = 1) : 
  ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
by
  sorry

end gcd_max_possible_value_l253_25390


namespace seq_positive_integers_no_m_exists_l253_25389

-- Definition of the sequence
def seq (n : ℕ) : ℕ :=
  Nat.recOn n
    1
    (λ n a_n => 3 * a_n + 2 * (2 * a_n * a_n - 1).sqrt)

-- Axiomatize the properties involved in the recurrence relation
axiom rec_sqrt_property (n : ℕ) : ∃ k : ℕ, (2 * seq n * seq n - 1) = k * k

-- Proof statement for the sequence of positive integers
theorem seq_positive_integers (n : ℕ) : seq n > 0 := sorry

-- Proof statement for non-existence of m such that 2015 divides seq(m)
theorem no_m_exists (m : ℕ) : ¬ (2015 ∣ seq m) := sorry

end seq_positive_integers_no_m_exists_l253_25389


namespace length_of_first_train_l253_25334

noncomputable def first_train_length 
  (speed_first_train_km_h : ℕ) 
  (speed_second_train_km_h : ℕ) 
  (length_second_train_m : ℕ) 
  (time_seconds : ℝ) 
  (relative_speed_m_s : ℝ) : ℝ :=
  let relative_speed_mps := (speed_first_train_km_h + speed_second_train_km_h) * (5 / 18)
  let distance_covered := relative_speed_mps * time_seconds
  let length_first_train := distance_covered - length_second_train_m
  length_first_train

theorem length_of_first_train : 
  first_train_length 40 50 165 11.039116870650348 25 = 110.9779217662587 :=
by 
  rw [first_train_length]
  sorry

end length_of_first_train_l253_25334


namespace probability_of_exactly_three_heads_l253_25366

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := (n.choose k)
noncomputable def probability_three_heads_in_eight_tosses : ℚ :=
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := binomial 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_exactly_three_heads :
  probability_three_heads_in_eight_tosses = 7 / 32 :=
by 
  sorry

end probability_of_exactly_three_heads_l253_25366


namespace subset_single_element_l253_25355

-- Define the set X
def X : Set ℝ := { x | x > -1 }

-- The proof statement
-- We need to prove that {0} ⊆ X
theorem subset_single_element : {0} ⊆ X :=
sorry

end subset_single_element_l253_25355


namespace eggs_total_l253_25384

-- Definitions of the conditions in Lean
def num_people : ℕ := 3
def omelets_per_person : ℕ := 3
def eggs_per_omelet : ℕ := 4

-- The claim we need to prove
theorem eggs_total : (num_people * omelets_per_person) * eggs_per_omelet = 36 :=
by
  sorry

end eggs_total_l253_25384


namespace part1_part2_l253_25317

def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) := 
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) := 
sorry

end part1_part2_l253_25317


namespace bricks_required_to_pave_courtyard_l253_25316

theorem bricks_required_to_pave_courtyard :
  let courtyard_length_m := 24
  let courtyard_width_m := 14
  let brick_length_cm := 25
  let brick_width_cm := 15
  let courtyard_area_m2 := courtyard_length_m * courtyard_width_m
  let courtyard_area_cm2 := courtyard_area_m2 * 10000
  let brick_area_cm2 := brick_length_cm * brick_width_cm
  let num_bricks := courtyard_area_cm2 / brick_area_cm2
  num_bricks = 8960 := by
  {
    -- Additional context not needed for theorem statement, mock proof omitted
    sorry
  }

end bricks_required_to_pave_courtyard_l253_25316


namespace inequality_solution_set_l253_25354

theorem inequality_solution_set (x : ℝ) : (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end inequality_solution_set_l253_25354


namespace proposition_verification_l253_25344

-- Definitions and Propositions
def prop1 : Prop := (∀ x, x = 1 → x^2 - 3 * x + 2 = 0) ∧ (∃ x, x ≠ 1 ∧ x^2 - 3 * x + 2 = 0)
def prop2 : Prop := (∀ x, ¬ (x^2 - 3 * x + 2 = 0 → x = 1) → (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0))
def prop3 : Prop := ¬ (∃ x > 0, x^2 + x + 1 < 0) → (∀ x ≤ 0, x^2 + x + 1 ≥ 0)
def prop4 : Prop := ¬ (∃ p q : Prop, (p ∨ q) → ¬p ∧ ¬q)

-- Final theorem statement
theorem proposition_verification : prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4 := by 
  sorry

end proposition_verification_l253_25344


namespace average_speed_l253_25368

def initial_odometer_reading : ℕ := 20
def final_odometer_reading : ℕ := 200
def travel_duration : ℕ := 6

theorem average_speed :
  (final_odometer_reading - initial_odometer_reading) / travel_duration = 30 := by
  sorry

end average_speed_l253_25368


namespace Rachel_books_total_l253_25397

-- Define the conditions
def mystery_shelves := 6
def picture_shelves := 2
def scifi_shelves := 3
def bio_shelves := 4
def books_per_shelf := 9

-- Define the total number of books
def total_books := 
  mystery_shelves * books_per_shelf + 
  picture_shelves * books_per_shelf + 
  scifi_shelves * books_per_shelf + 
  bio_shelves * books_per_shelf

-- Statement of the problem
theorem Rachel_books_total : total_books = 135 := 
by
  -- Proof can be added here
  sorry

end Rachel_books_total_l253_25397


namespace find_x_l253_25399

theorem find_x (x : ℕ) (h : 2^x - 2^(x-2) = 3 * 2^(12)) : x = 14 :=
sorry

end find_x_l253_25399


namespace set_B_roster_method_l253_25376

def A : Set ℤ := {-2, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem set_B_roster_method : B = {4, 9, 16} :=
by
  sorry

end set_B_roster_method_l253_25376


namespace corveus_lack_of_sleep_l253_25312

def daily_sleep_actual : ℕ := 4
def daily_sleep_recommended : ℕ := 6
def days_in_week : ℕ := 7

theorem corveus_lack_of_sleep : (daily_sleep_recommended - daily_sleep_actual) * days_in_week = 14 := 
by 
  sorry

end corveus_lack_of_sleep_l253_25312


namespace distinct_L_shapes_l253_25364

-- Definitions of conditions
def num_convex_shapes : Nat := 10
def L_shapes_per_convex : Nat := 2
def corner_L_shapes : Nat := 4

-- Total number of distinct "L" shapes
def total_L_shapes : Nat :=
  num_convex_shapes * L_shapes_per_convex + corner_L_shapes

theorem distinct_L_shapes :
  total_L_shapes = 24 :=
by
  -- Proof is omitted
  sorry

end distinct_L_shapes_l253_25364


namespace scientific_notation_l253_25306

theorem scientific_notation : (20160 : ℝ) = 2.016 * 10^4 := 
  sorry

end scientific_notation_l253_25306


namespace binary_to_decimal_110011_l253_25371

theorem binary_to_decimal_110011 : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 0 * 2^3 + 1 * 2^4 + 1 * 2^5) = 51 :=
by
  sorry

end binary_to_decimal_110011_l253_25371


namespace value_of_ak_l253_25328

noncomputable def Sn (n : ℕ) : ℤ := n^2 - 9 * n
noncomputable def a (n : ℕ) : ℤ := Sn n - Sn (n - 1)

theorem value_of_ak (k : ℕ) (hk : 5 < a k ∧ a k < 8) : a k = 6 := by
  sorry

end value_of_ak_l253_25328


namespace teaching_arrangements_l253_25357

theorem teaching_arrangements : 
  let teachers := ["A", "B", "C", "D", "E", "F"]
  let lessons := ["L1", "L2", "L3", "L4"]
  let valid_first_lesson := ["A", "B"]
  let valid_fourth_lesson := ["A", "C"]
  ∃ arrangements : ℕ, 
    (arrangements = 36) ∧
    (∀ (l1 l2 l3 l4 : String), (l1 ∈ valid_first_lesson) → (l4 ∈ valid_fourth_lesson) → 
      (l2 ≠ l1 ∧ l2 ≠ l4 ∧ l3 ≠ l1 ∧ l3 ≠ l4) ∧ 
      (List.length teachers - (if (l1 == "A") then 1 else 0) - (if (l4 == "A") then 1 else 0) = 4)) :=
by {
  -- This is just the theorem statement; no proof is required.
  sorry
}

end teaching_arrangements_l253_25357


namespace find_f3_l253_25385

theorem find_f3 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 5) : f 3 = -2 :=
by
  sorry

end find_f3_l253_25385


namespace original_water_depth_in_larger_vase_l253_25305

-- Definitions based on the conditions
noncomputable def largerVaseDiameter := 20 -- in cm
noncomputable def smallerVaseDiameter := 10 -- in cm
noncomputable def smallerVaseHeight := 16 -- in cm

-- Proving the original depth of the water in the larger vase
theorem original_water_depth_in_larger_vase :
  ∃ depth : ℝ, depth = 14 :=
by
  sorry

end original_water_depth_in_larger_vase_l253_25305


namespace april_plant_arrangement_l253_25308

theorem april_plant_arrangement :
    let nBasil := 5
    let nTomato := 4
    let nPairs := nTomato / 2
    let nUnits := nBasil + nPairs
    let totalWays := (Nat.factorial nUnits) * (Nat.factorial nPairs) * (Nat.factorial (nPairs - 1))
    totalWays = 20160 := by
{
  let nBasil := 5
  let nTomato := 4
  let nPairs := nTomato / 2
  let nUnits := nBasil + nPairs
  let totalWays := (Nat.factorial nUnits) * (Nat.factorial nPairs) * (Nat.factorial (nPairs - 1))
  sorry
}

end april_plant_arrangement_l253_25308


namespace tom_dimes_now_l253_25326

-- Define the initial number of dimes and the number of dimes given by dad
def initial_dimes : ℕ := 15
def dimes_given_by_dad : ℕ := 33

-- Define the final count of dimes Tom has now
def final_dimes (initial_dimes dimes_given_by_dad : ℕ) : ℕ :=
  initial_dimes + dimes_given_by_dad

-- The main theorem to prove "how many dimes Tom has now"
theorem tom_dimes_now : initial_dimes + dimes_given_by_dad = 48 :=
by
  -- The proof can be skipped using sorry
  sorry

end tom_dimes_now_l253_25326


namespace michael_hours_worked_l253_25375

def michael_hourly_rate := 7
def michael_overtime_rate := 2 * michael_hourly_rate
def work_hours := 40
def total_earnings := 320

theorem michael_hours_worked :
  (total_earnings = michael_hourly_rate * work_hours + michael_overtime_rate * (42 - work_hours)) :=
sorry

end michael_hours_worked_l253_25375


namespace sum_and_product_of_roots_l253_25336

theorem sum_and_product_of_roots (m n : ℝ) (h1 : (m / 3) = 9) (h2 : (n / 3) = 20) : m + n = 87 :=
by
  sorry

end sum_and_product_of_roots_l253_25336


namespace greatest_prime_factor_3_8_plus_6_7_l253_25335

theorem greatest_prime_factor_3_8_plus_6_7 : ∃ p, p = 131 ∧ Prime p ∧ ∀ q, Prime q ∧ q ∣ (3^8 + 6^7) → q ≤ 131 :=
by
  sorry


end greatest_prime_factor_3_8_plus_6_7_l253_25335


namespace solve_quadratic_l253_25395

theorem solve_quadratic : 
  ∀ x : ℝ, (x - 1) ^ 2 = 64 → (x = 9 ∨ x = -7) :=
by
  sorry

end solve_quadratic_l253_25395


namespace find_a11_l253_25378

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
axiom cond1 : ∀ n : ℕ, n > 0 → 4 * S n = 2 * a n - n^2 + 7 * n

-- Theorem stating the proof problem
theorem find_a11 :
  a 11 = -2 :=
sorry

end find_a11_l253_25378


namespace white_area_of_painting_l253_25343

theorem white_area_of_painting (s : ℝ) (total_gray_area : ℝ) (gray_area_squares : ℕ)
  (h1 : ∀ t, t = 3 * s) -- The frame is 3 times the smaller square's side length.
  (h2 : total_gray_area = 62) -- The gray area is 62 cm^2.
  (h3 : gray_area_squares = 31) -- The gray area is composed of 31 smaller squares.
  : ∃ white_area, white_area = 10 := 
  sorry

end white_area_of_painting_l253_25343


namespace haruto_ratio_is_1_to_2_l253_25398

def haruto_tomatoes_ratio (total_tomatoes : ℕ) (eaten_by_birds : ℕ) (remaining_tomatoes : ℕ) : ℚ :=
  let picked_tomatoes := total_tomatoes - eaten_by_birds
  let given_to_friend := picked_tomatoes - remaining_tomatoes
  given_to_friend / picked_tomatoes

theorem haruto_ratio_is_1_to_2 : haruto_tomatoes_ratio 127 19 54 = 1 / 2 :=
by
  -- We'll skip the proof details as instructed
  sorry

end haruto_ratio_is_1_to_2_l253_25398


namespace sqrt_meaningful_range_l253_25337

theorem sqrt_meaningful_range (x : ℝ) (h : 0 ≤ x - 1) : 1 ≤ x :=
by
sorry

end sqrt_meaningful_range_l253_25337


namespace perpendicular_lines_l253_25338

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, (a * x - y + 2 * a = 0) → ((2 * a - 1) * x + a * y + a = 0) -> 
  (a ≠ 0 → ∃ k : ℝ, k = (a * ((1 - 2 * a) / a)) ∧ k = -1) -> a * ((1 - 2 * a) / a) = -1) →
  a = 0 ∨ a = 1 := by sorry

end perpendicular_lines_l253_25338


namespace min_club_members_l253_25311

theorem min_club_members (n : ℕ) :
  (∀ k : ℕ, k = 8 ∨ k = 9 ∨ k = 11 → n % k = 0) ∧ (n ≥ 300) → n = 792 :=
sorry

end min_club_members_l253_25311


namespace focus_of_parabola_l253_25330

theorem focus_of_parabola (a : ℝ) (h : ℝ) (k : ℝ) (x y : ℝ) :
  (∀ x, y = a * (x - h) ^ 2 + k) →
  a = -2 ∧ h = 0 ∧ k = 4 →
  (0, y - (1 / (4 * a))) = (0, 31 / 8) := by
  sorry

end focus_of_parabola_l253_25330


namespace x_minus_y_eq_2_l253_25391

theorem x_minus_y_eq_2 (x y : ℝ) (h1 : 2 * x + 3 * y = 9) (h2 : 3 * x + 2 * y = 11) : x - y = 2 :=
sorry

end x_minus_y_eq_2_l253_25391


namespace max_height_of_projectile_l253_25304

def projectile_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height_of_projectile : 
  ∃ t : ℝ, projectile_height t = 161 :=
sorry

end max_height_of_projectile_l253_25304


namespace solve_fraction_equation_l253_25387

theorem solve_fraction_equation :
  ∀ x : ℝ, (3 / (2 * x - 2) + 1 / (1 - x) = 3) → x = 7 / 6 :=
by
  sorry

end solve_fraction_equation_l253_25387


namespace inlet_pipe_rate_l253_25313

-- Conditions definitions
def tank_capacity : ℕ := 4320
def leak_empty_time : ℕ := 6
def full_empty_time_with_inlet : ℕ := 8

-- Question translated into a theorem
theorem inlet_pipe_rate : 
  (tank_capacity / leak_empty_time) = 720 →
  (tank_capacity / full_empty_time_with_inlet) = 540 →
  ∀ R : ℕ, 
    R - 720 = 540 →
    (R / 60) = 21 :=
by
  intros h_leak h_net R h_R
  sorry

end inlet_pipe_rate_l253_25313


namespace trigonometric_identity_proof_l253_25353

theorem trigonometric_identity_proof :
  ( (Real.cos (40 * Real.pi / 180) + Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)))
  / (Real.sin (70 * Real.pi / 180) * Real.sqrt (1 + Real.cos (40 * Real.pi / 180))) ) =
  Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_proof_l253_25353


namespace factor_expression_zero_l253_25314

theorem factor_expression_zero (a b c : ℝ) (h : a + b + c ≠ 0) :
  (a^3 - b^3)^2 + (b^3 - c^3)^2 + (c^3 - a^3)^2 = 0 :=
sorry

end factor_expression_zero_l253_25314


namespace solve_for_x_l253_25358

theorem solve_for_x (x : ℕ) (hx : 1000^4 = 10^x) : x = 12 := 
by
  sorry

end solve_for_x_l253_25358


namespace earnings_per_widget_l253_25341

theorem earnings_per_widget (W_h : ℝ) (H_w : ℕ) (W_t : ℕ) (E_w : ℝ) (E : ℝ) :
  W_h = 12.50 ∧ H_w = 40 ∧ W_t = 1000 ∧ E_w = 660 →
  E = 0.16 :=
by
  sorry

end earnings_per_widget_l253_25341


namespace value_of_m_div_x_l253_25310

variable (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ratio : a / b = 4 / 5)

def x := a + 0.25 * a
def m := b - 0.40 * b

theorem value_of_m_div_x (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ratio : a / b = 4 / 5) :
    m / x = 3 / 5 :=
by
  sorry

end value_of_m_div_x_l253_25310


namespace number_of_girls_l253_25333

theorem number_of_girls (sections : ℕ) (boys_per_section : ℕ) (total_boys : ℕ) (total_sections : ℕ) (boys_sections girls : ℕ) :
  total_boys = 408 → 
  total_sections = 27 → 
  total_boys / total_sections = boys_per_section → 
  boys_sections = total_boys / boys_per_section → 
  total_sections - boys_sections = girls / boys_per_section → 
  girls = 324 :=
by sorry

end number_of_girls_l253_25333


namespace trapezoid_area_l253_25340

theorem trapezoid_area (A B : ℝ) (n : ℕ) (hA : A = 36) (hB : B = 4) (hn : n = 6) :
    (A - B) / n = 5.33 := 
by 
  -- Given conditions and the goal
  sorry

end trapezoid_area_l253_25340


namespace width_of_bottom_trapezium_l253_25332

theorem width_of_bottom_trapezium (top_width : ℝ) (area : ℝ) (depth : ℝ) (bottom_width : ℝ) 
  (h_top_width : top_width = 10)
  (h_area : area = 640)
  (h_depth : depth = 80) :
  bottom_width = 6 :=
by
  -- Problem description: calculating the width of the bottom of the trapezium given the conditions.
  sorry

end width_of_bottom_trapezium_l253_25332


namespace cars_meet_cars_apart_l253_25361

section CarsProblem

variable (distance : ℕ) (speedA speedB : ℕ) (distanceToMeet distanceApart : ℕ)

def meetTime := distance / (speedA + speedB)
def apartTime1 := (distance - distanceApart) / (speedA + speedB)
def apartTime2 := (distance + distanceApart) / (speedA + speedB)

theorem cars_meet (h1: distance = 450) (h2: speedA = 115) (h3: speedB = 85):
  meetTime distance speedA speedB = 9 / 4 := by
  sorry

theorem cars_apart (h1: distance = 450) (h2: speedA = 115) (h3: speedB = 85) (h4: distanceApart = 50):
  apartTime1 distance speedA speedB distanceApart = 2 ∧ apartTime2 distance speedA speedB distanceApart = 5 / 2 := by
  sorry

end CarsProblem

end cars_meet_cars_apart_l253_25361


namespace dilation_image_l253_25362

open Complex

noncomputable def dilation_center := (1 : ℂ) + (3 : ℂ) * I
noncomputable def scale_factor := -3
noncomputable def initial_point := -I
noncomputable def target_point := (4 : ℂ) + (15 : ℂ) * I

theorem dilation_image :
  let c := dilation_center
  let k := scale_factor
  let z := initial_point
  let z_prime := target_point
  z_prime = c + k * (z - c) := 
  by
    sorry

end dilation_image_l253_25362


namespace customers_total_l253_25318

theorem customers_total 
  (initial : ℝ) 
  (added_lunch_rush : ℝ) 
  (added_after_lunch_rush : ℝ) :
  initial = 29.0 →
  added_lunch_rush = 20.0 →
  added_after_lunch_rush = 34.0 →
  initial + added_lunch_rush + added_after_lunch_rush = 83.0 :=
by
  intros h1 h2 h3
  sorry

end customers_total_l253_25318


namespace angle_acb_after_rotations_is_30_l253_25373

noncomputable def initial_angle : ℝ := 60
noncomputable def rotation_clockwise_540 : ℝ := -540
noncomputable def rotation_counterclockwise_90 : ℝ := 90
noncomputable def final_angle : ℝ := 30

theorem angle_acb_after_rotations_is_30 
  (initial_angle : ℝ)
  (rotation_clockwise_540 : ℝ)
  (rotation_counterclockwise_90 : ℝ) :
  final_angle = 30 :=
sorry

end angle_acb_after_rotations_is_30_l253_25373


namespace amount_given_to_second_set_of_families_l253_25363

theorem amount_given_to_second_set_of_families
  (total_spent : ℝ) (amount_first_set : ℝ) (amount_last_set : ℝ)
  (h_total_spent : total_spent = 900)
  (h_amount_first_set : amount_first_set = 325)
  (h_amount_last_set : amount_last_set = 315) :
  total_spent - amount_first_set - amount_last_set = 260 :=
by
  -- sorry is placed to skip the proof
  sorry

end amount_given_to_second_set_of_families_l253_25363


namespace radius_of_circle_l253_25307

variables (O P A B : Type) [MetricSpace O] [MetricSpace P] [MetricSpace A] [MetricSpace B]
variables (circle_radius : ℝ) (PA PB OP : ℝ)

theorem radius_of_circle
  (h1 : PA * PB = 24)
  (h2 : OP = 5)
  (circle_radius : ℝ)
  : circle_radius = 7 :=
by sorry

end radius_of_circle_l253_25307


namespace add_to_fraction_eq_l253_25324

theorem add_to_fraction_eq (n : ℤ) (h : (4 + n) / (7 + n) = 3 / 4) : n = 5 :=
by sorry

end add_to_fraction_eq_l253_25324


namespace height_of_building_l253_25302

def flagpole_height : ℝ := 18
def flagpole_shadow_length : ℝ := 45

def building_shadow_length : ℝ := 65
def building_height : ℝ := 26

theorem height_of_building
  (hflagpole : flagpole_height / flagpole_shadow_length = building_height / building_shadow_length) :
  building_height = 26 :=
sorry

end height_of_building_l253_25302


namespace count_implications_l253_25380

def r : Prop := sorry
def s : Prop := sorry

def statement_1 := ¬r ∧ ¬s
def statement_2 := ¬r ∧ s
def statement_3 := r ∧ ¬s
def statement_4 := r ∧ s

def neg_rs : Prop := r ∨ s

theorem count_implications : (statement_2 → neg_rs) ∧ 
                             (statement_3 → neg_rs) ∧ 
                             (statement_4 → neg_rs) ∧ 
                             (¬(statement_1 → neg_rs)) -> 
                             3 = 3 := by
  sorry

end count_implications_l253_25380


namespace solve_for_s_l253_25367

-- Definition of the given problem conditions
def parallelogram_sides_60_angle_sqrt_area (s : ℝ) :=
  ∃ (area : ℝ), (area = 27 * Real.sqrt 3) ∧
  (3 * s * s * Real.sqrt 3 = area)

-- Proof statement to demonstrate the equivalence of the theoretical and computed value of s
theorem solve_for_s (s : ℝ) : parallelogram_sides_60_angle_sqrt_area s → s = 3 :=
by
  intro h
  sorry

end solve_for_s_l253_25367
