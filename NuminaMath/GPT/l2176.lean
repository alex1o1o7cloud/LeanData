import Mathlib

namespace NUMINAMATH_GPT_no_solution_for_triples_l2176_217602

theorem no_solution_for_triples :
  ¬ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ (a * b + b * c = 66) ∧ (a * c + b * c = 35) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_solution_for_triples_l2176_217602


namespace NUMINAMATH_GPT_area_is_300_l2176_217649

variable (l w : ℝ) -- Length and Width of the playground

-- Conditions
def condition1 : Prop := 2 * l + 2 * w = 80
def condition2 : Prop := l = 3 * w

-- Question and Answer
def area_of_playground : ℝ := l * w

theorem area_is_300 (h1 : condition1 l w) (h2 : condition2 l w) : area_of_playground l w = 300 := 
by
  sorry

end NUMINAMATH_GPT_area_is_300_l2176_217649


namespace NUMINAMATH_GPT_Jina_has_51_mascots_l2176_217623

def teddies := 5
def bunnies := 3 * teddies
def koala_bear := 1
def additional_teddies := 2 * bunnies
def total_mascots := teddies + bunnies + koala_bear + additional_teddies

theorem Jina_has_51_mascots : total_mascots = 51 := by
  sorry

end NUMINAMATH_GPT_Jina_has_51_mascots_l2176_217623


namespace NUMINAMATH_GPT_problem1_problem2_l2176_217622

-- Definitions
variables {a b z : ℝ}

-- Problem 1 translated to Lean
theorem problem1 (h1 : a + 2 * b = 9) (h2 : |9 - 2 * b| + |a + 1| < 3) : -2 < a ∧ a < 1 := 
sorry

-- Problem 2 translated to Lean
theorem problem2 (h1 : a + 2 * b = 9) (ha_pos : 0 < a) (hb_pos : 0 < b) : 
  ∃ z : ℝ, z = a * b^2 ∧ ∀ w : ℝ, (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + 2 * b = 9 ∧ w = a * b^2) → w ≤ 27 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2176_217622


namespace NUMINAMATH_GPT_total_flowers_in_vases_l2176_217643

theorem total_flowers_in_vases :
  let vase_count := 5
  let flowers_per_vase_4 := 5
  let flowers_per_vase_1 := 6
  let vases_with_5_flowers := 4
  let vases_with_6_flowers := 1
  (4 * 5 + 1 * 6 = 26) := by
  let total_flowers := 4 * 5 + 1 * 6
  show total_flowers = 26
  sorry

end NUMINAMATH_GPT_total_flowers_in_vases_l2176_217643


namespace NUMINAMATH_GPT_total_people_clean_city_l2176_217672

-- Define the conditions
def lizzie_group : Nat := 54
def group_difference : Nat := 17
def other_group := lizzie_group - group_difference

-- State the theorem
theorem total_people_clean_city : lizzie_group + other_group = 91 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_total_people_clean_city_l2176_217672


namespace NUMINAMATH_GPT_question_1_question_2_l2176_217645

noncomputable def f (x m : ℝ) : ℝ := abs (x + m) - abs (2 * x - 2 * m)

theorem question_1 (x : ℝ) (m : ℝ) (h : m = 1/2) (h_pos : m > 0) : 
  (f x m ≥ 1/2) ↔ (1/3 ≤ x ∧ x < 1) :=
sorry

theorem question_2 (m : ℝ) (h_pos : m > 0) : 
  (∀ x : ℝ, ∃ t : ℝ, f x m + abs (t - 3) < abs (t + 4)) ↔ (0 < m ∧ m < 7/2) :=
sorry

end NUMINAMATH_GPT_question_1_question_2_l2176_217645


namespace NUMINAMATH_GPT_percent_increase_quarter_l2176_217644

-- Define the profit changes over each month
def profit_march (P : ℝ) := P
def profit_april (P : ℝ) := 1.40 * P
def profit_may (P : ℝ) := 1.12 * P
def profit_june (P : ℝ) := 1.68 * P

-- Starting Lean theorem statement
theorem percent_increase_quarter (P : ℝ) (hP : P > 0) :
  ((profit_june P - profit_march P) / profit_march P) * 100 = 68 :=
  sorry

end NUMINAMATH_GPT_percent_increase_quarter_l2176_217644


namespace NUMINAMATH_GPT_celer_tanks_dimensions_l2176_217638

theorem celer_tanks_dimensions :
  ∃ (a v : ℕ), 
    (a * a * v = 200) ∧
    (2 * a ^ 3 + 50 = 300) ∧
    (a = 5) ∧
    (v = 8) :=
sorry

end NUMINAMATH_GPT_celer_tanks_dimensions_l2176_217638


namespace NUMINAMATH_GPT_number_added_to_x_l2176_217699

theorem number_added_to_x (x : ℕ) (some_number : ℕ) (h1 : x = 3) (h2 : x + some_number = 4) : some_number = 1 := 
by
  -- Given hypotheses can be used here
  sorry

end NUMINAMATH_GPT_number_added_to_x_l2176_217699


namespace NUMINAMATH_GPT_sphere_surface_area_l2176_217624

theorem sphere_surface_area (a b c : ℝ) (r : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : r = (Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2)) / 2):
    4 * Real.pi * r ^ 2 = 50 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_l2176_217624


namespace NUMINAMATH_GPT_weight_of_8_moles_of_AlI3_l2176_217641

noncomputable def atomic_weight_Al : ℝ := 26.98
noncomputable def atomic_weight_I : ℝ := 126.90
noncomputable def molecular_weight_AlI3 : ℝ := atomic_weight_Al + 3 * atomic_weight_I

theorem weight_of_8_moles_of_AlI3 : 
  (8 * molecular_weight_AlI3) = 3261.44 := by
sorry

end NUMINAMATH_GPT_weight_of_8_moles_of_AlI3_l2176_217641


namespace NUMINAMATH_GPT_total_fishes_l2176_217650

theorem total_fishes (Will_catfish : ℕ) (Will_eels : ℕ) (Henry_multiplier : ℕ) (Henry_return_fraction : ℚ) :
  Will_catfish = 16 → Will_eels = 10 → Henry_multiplier = 3 → Henry_return_fraction = 1 / 2 →
  (Will_catfish + Will_eels) + (Henry_multiplier * Will_catfish - (Henry_multiplier * Will_catfish / 2)) = 50 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_fishes_l2176_217650


namespace NUMINAMATH_GPT_perfect_square_polynomial_l2176_217652

theorem perfect_square_polynomial (m : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x : ℝ, x^2 + 2*(m-3)*x + 25 = f x * f x) ↔ (m = 8 ∨ m = -2) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_polynomial_l2176_217652


namespace NUMINAMATH_GPT_first_term_geometric_series_l2176_217614

theorem first_term_geometric_series (a r S : ℝ) (h1 : r = -1/3) (h2 : S = 9)
  (h3 : S = a / (1 - r)) : a = 12 :=
sorry

end NUMINAMATH_GPT_first_term_geometric_series_l2176_217614


namespace NUMINAMATH_GPT_nutmeg_amount_l2176_217642

def amount_of_cinnamon : ℝ := 0.6666666666666666
def difference_cinnamon_nutmeg : ℝ := 0.16666666666666666

theorem nutmeg_amount (x : ℝ) 
  (h1 : amount_of_cinnamon = x + difference_cinnamon_nutmeg) : 
  x = 0.5 :=
by 
  sorry

end NUMINAMATH_GPT_nutmeg_amount_l2176_217642


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2176_217687

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 3 * x + 2 < 0 ↔ 1 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2176_217687


namespace NUMINAMATH_GPT_subset_ratio_l2176_217605

theorem subset_ratio (S T : ℕ) (hS : S = 256) (hT : T = 56) :
  (T / S : ℚ) = 7 / 32 := by
sorry

end NUMINAMATH_GPT_subset_ratio_l2176_217605


namespace NUMINAMATH_GPT_inequality_solution_set_result_l2176_217658

theorem inequality_solution_set_result (a b x : ℝ) :
  (∀ x, a ≤ (3/4) * x^2 - 3 * x + 4 ∧ (3/4) * x^2 - 3 * x + 4 ≤ b) ∧ 
  (∀ x, x ∈ Set.Icc a b ↔ a ≤ x ∧ x ≤ b) →
  a + b = 4 := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_result_l2176_217658


namespace NUMINAMATH_GPT_directrix_of_parabola_l2176_217679

theorem directrix_of_parabola (p : ℝ) (y x : ℝ) :
  y = x^2 → x^2 = 4 * p * y → 4 * y + 1 = 0 :=
by
  intros hyp1 hyp2
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l2176_217679


namespace NUMINAMATH_GPT_number_of_packages_l2176_217662

theorem number_of_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 56) (h2 : tshirts_per_package = 2) : 
  (total_tshirts / tshirts_per_package) = 28 := 
  by
    sorry

end NUMINAMATH_GPT_number_of_packages_l2176_217662


namespace NUMINAMATH_GPT_initial_students_l2176_217695

variable (x : ℕ) -- let x be the initial number of students

-- each condition defined as a function
def first_round_rem (x : ℕ) : ℕ := (40 * x) / 100
def second_round_rem (x : ℕ) : ℕ := first_round_rem x / 2
def third_round_rem (x : ℕ) : ℕ := second_round_rem x / 4

theorem initial_students (x : ℕ) (h : third_round_rem x = 15) : x = 300 := 
by sorry  -- proof will be inserted here

end NUMINAMATH_GPT_initial_students_l2176_217695


namespace NUMINAMATH_GPT_plane_intersects_unit_cubes_l2176_217601

-- Definitions:
def isLargeCube (cube : ℕ × ℕ × ℕ) : Prop := cube = (4, 4, 4)
def isUnitCube (size : ℕ) : Prop := size = 1

-- The main theorem we want to prove:
theorem plane_intersects_unit_cubes :
  ∀ (cube : ℕ × ℕ × ℕ) (plane : (ℝ × ℝ × ℝ) → ℝ),
  isLargeCube cube →
  (∀ point : ℝ × ℝ × ℝ, plane point = 0 → 
       ∃ (x y z : ℕ), x < 4 ∧ y < 4 ∧ z < 4 ∧ 
                     (x, y, z) ∈ { coords : ℕ × ℕ × ℕ | true }) →
  (∃ intersects : ℕ, intersects = 16) :=
by
  intros cube plane Hcube Hplane
  sorry

end NUMINAMATH_GPT_plane_intersects_unit_cubes_l2176_217601


namespace NUMINAMATH_GPT_subset_implies_bound_l2176_217625

def setA := {x : ℝ | x < 2}
def setB (m : ℝ) := {x : ℝ | x < m}

theorem subset_implies_bound (m : ℝ) (h : setB m ⊆ setA) : m ≤ 2 :=
by 
  sorry

end NUMINAMATH_GPT_subset_implies_bound_l2176_217625


namespace NUMINAMATH_GPT_problem_solution_l2176_217666

theorem problem_solution :
  ∃ n : ℕ, 50 < n ∧ n < 70 ∧ n % 5 = 3 ∧ n % 7 = 2 ∧ n = 58 :=
by
  -- Lean code to prove the statement
  sorry

end NUMINAMATH_GPT_problem_solution_l2176_217666


namespace NUMINAMATH_GPT_solve_for_a_minus_b_l2176_217659

theorem solve_for_a_minus_b (a b : ℚ) 
  (h1 : 2020 * a + 2024 * b = 2030) 
  (h2 : 2022 * a + 2026 * b = 2032) : 
  a - b = -4 := 
sorry

end NUMINAMATH_GPT_solve_for_a_minus_b_l2176_217659


namespace NUMINAMATH_GPT_sqrt_operations_correctness_l2176_217612

open Real

theorem sqrt_operations_correctness :
  (sqrt 2 + sqrt 3 ≠ sqrt 5) ∧
  (sqrt (2/3) * sqrt 6 = 2) ∧
  (sqrt 9 = 3) ∧
  (sqrt ((-6) ^ 2) = 6) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_operations_correctness_l2176_217612


namespace NUMINAMATH_GPT_emmalyn_earnings_l2176_217665

theorem emmalyn_earnings :
  let rate_per_meter := 0.20
  let number_of_fences := 50
  let length_of_each_fence := 500
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * rate_per_meter
  total_income = 5000 :=
by
  sorry

end NUMINAMATH_GPT_emmalyn_earnings_l2176_217665


namespace NUMINAMATH_GPT_annes_score_l2176_217617

theorem annes_score (a b : ℕ) (h1 : a = b + 50) (h2 : (a + b) / 2 = 150) : a = 175 := 
by
  sorry

end NUMINAMATH_GPT_annes_score_l2176_217617


namespace NUMINAMATH_GPT_probability_pink_second_marble_l2176_217604

def bagA := (5, 5)  -- (red, green)
def bagB := (8, 2)  -- (pink, purple)
def bagC := (3, 7)  -- (pink, purple)

def P (success total : ℕ) := success / total

def probability_red := P 5 10
def probability_green := P 5 10

def probability_pink_given_red := P 8 10
def probability_pink_given_green := P 3 10

theorem probability_pink_second_marble :
  probability_red * probability_pink_given_red +
  probability_green * probability_pink_given_green = 11 / 20 :=
sorry

end NUMINAMATH_GPT_probability_pink_second_marble_l2176_217604


namespace NUMINAMATH_GPT_bird_cages_count_l2176_217618

/-- 
If each bird cage contains 2 parrots and 2 parakeets,
and the total number of birds is 36,
then the number of bird cages is 9.
-/
theorem bird_cages_count (parrots_per_cage parakeets_per_cage total_birds cages : ℕ)
  (h1 : parrots_per_cage = 2)
  (h2 : parakeets_per_cage = 2)
  (h3 : total_birds = 36)
  (h4 : total_birds = (parrots_per_cage + parakeets_per_cage) * cages) :
  cages = 9 := 
by 
  sorry

end NUMINAMATH_GPT_bird_cages_count_l2176_217618


namespace NUMINAMATH_GPT_y_value_when_x_is_3_l2176_217675

theorem y_value_when_x_is_3 :
  (x + y = 30) → (x - y = 12) → (x * y = 189) → (x = 3) → y = 63 :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_y_value_when_x_is_3_l2176_217675


namespace NUMINAMATH_GPT_prob_board_251_l2176_217683

noncomputable def probability_boarding_bus_251 (r1 r2 : ℕ) : ℚ :=
  let interval_152 := r1
  let interval_251 := r2
  let total_area := interval_152 * interval_251
  let triangle_area := 1 / 2 * interval_152 * interval_152
  triangle_area / total_area

theorem prob_board_251 : probability_boarding_bus_251 5 7 = 5 / 14 := by
  sorry

end NUMINAMATH_GPT_prob_board_251_l2176_217683


namespace NUMINAMATH_GPT_quadratic_roots_l2176_217648

theorem quadratic_roots (x : ℝ) : 
  (x^2 + 3 * x - 1 = 0) ↔ (x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l2176_217648


namespace NUMINAMATH_GPT_students_appeared_l2176_217619

def passed (T : ℝ) : ℝ := 0.35 * T
def B_grade_range (T : ℝ) : ℝ := 0.25 * T
def failed (T : ℝ) : ℝ := T - passed T

theorem students_appeared (T : ℝ) (hp : passed T = 0.35 * T)
    (hb : B_grade_range T = 0.25 * T) (hf : failed T = 481) :
    T = 740 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_students_appeared_l2176_217619


namespace NUMINAMATH_GPT_tan_sum_of_angles_eq_neg_sqrt_three_l2176_217600

theorem tan_sum_of_angles_eq_neg_sqrt_three 
  (A B C : ℝ)
  (h1 : B - A = C - B)
  (h2 : A + B + C = Real.pi) :
  Real.tan (A + C) = -Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_sum_of_angles_eq_neg_sqrt_three_l2176_217600


namespace NUMINAMATH_GPT_min_percentage_both_physics_chemistry_l2176_217616

/--
Given:
- A certain school conducted a survey.
- 68% of the students like physics.
- 72% of the students like chemistry.

Prove that the minimum percentage of students who like both physics and chemistry is 40%.
-/
theorem min_percentage_both_physics_chemistry (P C : ℝ)
(hP : P = 0.68) (hC : C = 0.72) :
  ∃ B, B = P + C - 1 ∧ B = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_min_percentage_both_physics_chemistry_l2176_217616


namespace NUMINAMATH_GPT_cat_food_sufficiency_l2176_217670

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end NUMINAMATH_GPT_cat_food_sufficiency_l2176_217670


namespace NUMINAMATH_GPT_benny_gave_sandy_books_l2176_217661

theorem benny_gave_sandy_books :
  ∀ (Benny_initial Tim_books total_books Benny_after_giving : ℕ), 
    Benny_initial = 24 → 
    Tim_books = 33 →
    total_books = 47 → 
    total_books - Tim_books = Benny_after_giving →
    Benny_initial - Benny_after_giving = 10 :=
by
  intros Benny_initial Tim_books total_books Benny_after_giving
  intros hBenny_initial hTim_books htotal_books hBooks_after
  simp [hBenny_initial, hTim_books, htotal_books, hBooks_after]
  sorry


end NUMINAMATH_GPT_benny_gave_sandy_books_l2176_217661


namespace NUMINAMATH_GPT_find_a_plus_b_l2176_217647

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 2^(a * x + b)

theorem find_a_plus_b
  (a b : ℝ)
  (h1 : f a b 2 = 1 / 2)
  (h2 : f a b (1 / 2) = 2) :
  a + b = 1 / 3 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l2176_217647


namespace NUMINAMATH_GPT_digital_earth_concept_wrong_l2176_217680

theorem digital_earth_concept_wrong :
  ∀ (A C D : Prop),
  (A → true) →
  (C → true) →
  (D → true) →
  ¬(B → true) :=
by
  sorry

end NUMINAMATH_GPT_digital_earth_concept_wrong_l2176_217680


namespace NUMINAMATH_GPT_num_tuples_abc_l2176_217663

theorem num_tuples_abc (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : 2019 ≥ 10 * a) (h5 : 10 * a ≥ 100 * b) (h6 : 100 * b ≥ 1000 * c) : 
  ∃ n, n = 574 := sorry

end NUMINAMATH_GPT_num_tuples_abc_l2176_217663


namespace NUMINAMATH_GPT_number_of_solution_pairs_l2176_217684

theorem number_of_solution_pairs : 
  ∃ n, (∀ x y : ℕ, 4 * x + 7 * y = 548 → (x > 0 ∧ y > 0) → n = 19) :=
sorry

end NUMINAMATH_GPT_number_of_solution_pairs_l2176_217684


namespace NUMINAMATH_GPT_square_plot_area_l2176_217607

theorem square_plot_area
  (cost_per_foot : ℕ)
  (total_cost : ℕ)
  (s : ℕ)
  (area : ℕ)
  (h1 : cost_per_foot = 55)
  (h3 : total_cost = 3740)
  (h4 : total_cost = 4 * s * cost_per_foot)
  (h5 : area = s * s) :
  area = 289 := sorry

end NUMINAMATH_GPT_square_plot_area_l2176_217607


namespace NUMINAMATH_GPT_calculate_total_shaded_area_l2176_217689

theorem calculate_total_shaded_area
(smaller_square_side larger_square_side smaller_circle_radius larger_circle_radius : ℝ)
(h1 : smaller_square_side = 6)
(h2 : larger_square_side = 12)
(h3 : smaller_circle_radius = 3)
(h4 : larger_circle_radius = 6) :
  (smaller_square_side^2 - π * smaller_circle_radius^2) + 
  (larger_square_side^2 - π * larger_circle_radius^2) = 180 - 45 * π :=
by
  sorry

end NUMINAMATH_GPT_calculate_total_shaded_area_l2176_217689


namespace NUMINAMATH_GPT_age_difference_l2176_217656

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 16) : A - C = 16 :=
sorry

end NUMINAMATH_GPT_age_difference_l2176_217656


namespace NUMINAMATH_GPT_max_value_x_sub_2z_l2176_217688

theorem max_value_x_sub_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 16) :
  ∃ m, m = 4 * Real.sqrt 5 ∧ ∀ x y z, x^2 + y^2 + z^2 = 16 → x - 2 * z ≤ m :=
sorry

end NUMINAMATH_GPT_max_value_x_sub_2z_l2176_217688


namespace NUMINAMATH_GPT_harry_book_pages_correct_l2176_217609

-- Define the total pages in Selena's book.
def selena_book_pages : ℕ := 400

-- Define Harry's book pages as 20 fewer than half of Selena's book pages.
def harry_book_pages : ℕ := (selena_book_pages / 2) - 20

-- The theorem to prove the number of pages in Harry's book.
theorem harry_book_pages_correct : harry_book_pages = 180 := by
  sorry

end NUMINAMATH_GPT_harry_book_pages_correct_l2176_217609


namespace NUMINAMATH_GPT_minimize_x_l2176_217690

theorem minimize_x (x y : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : x + y^2 = x * y) : x ≥ 3 :=
sorry

end NUMINAMATH_GPT_minimize_x_l2176_217690


namespace NUMINAMATH_GPT_distinct_triangles_count_l2176_217646

def num_points : ℕ := 8
def num_rows : ℕ := 2
def num_cols : ℕ := 4

-- Define the number of ways to choose 3 points from the 8 available points.
def combinations (n k : ℕ) := Nat.choose n k
def total_combinations := combinations num_points 3

-- Define the number of degenerate cases of collinear points in columns.
def degenerate_cases_per_column := combinations num_cols 3
def total_degenerate_cases := num_cols * degenerate_cases_per_column

-- The number of distinct triangles is the total combinations minus the degenerate cases.
def distinct_triangles := total_combinations - total_degenerate_cases

theorem distinct_triangles_count : distinct_triangles = 40 := by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_distinct_triangles_count_l2176_217646


namespace NUMINAMATH_GPT_polynomial_function_value_l2176_217674

theorem polynomial_function_value 
  (p q r s : ℝ) 
  (h : p - q + r - s = 4) : 
  2 * p + q - 3 * r + 2 * s = -8 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_function_value_l2176_217674


namespace NUMINAMATH_GPT_value_of_g_at_2_l2176_217692

def g (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

theorem value_of_g_at_2 : g 2 = 11 := 
by
  sorry

end NUMINAMATH_GPT_value_of_g_at_2_l2176_217692


namespace NUMINAMATH_GPT_evaluate_expression_l2176_217639

theorem evaluate_expression (x : ℝ) :
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2176_217639


namespace NUMINAMATH_GPT_find_a_plus_b_l2176_217698

theorem find_a_plus_b (a b : ℤ) (h : 2*x^3 - a*x^2 - 5*x + 5 = (2*x^2 + a*x - 1)*(x - b) + 3) : a + b = 4 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_find_a_plus_b_l2176_217698


namespace NUMINAMATH_GPT_gum_sharing_l2176_217631

theorem gum_sharing (john cole aubrey : ℕ) (sharing_people : ℕ) 
  (hj : john = 54) (hc : cole = 45) (ha : aubrey = 0) 
  (hs : sharing_people = 3) : 
  john + cole + aubrey = 99 ∧ (john + cole + aubrey) / sharing_people = 33 := 
by
  sorry

end NUMINAMATH_GPT_gum_sharing_l2176_217631


namespace NUMINAMATH_GPT_range_of_a_l2176_217626

variable (a : ℝ)
def proposition_p := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def proposition_q := ∃ x₀ : ℝ, x₀^2 - x₀ + a = 0

theorem range_of_a (h1 : proposition_p a ∨ proposition_q a)
    (h2 : ¬ (proposition_p a ∧ proposition_q a)) :
    a < 0 ∨ (1 / 4) < a ∧ a < 4 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l2176_217626


namespace NUMINAMATH_GPT_work_days_together_l2176_217657

theorem work_days_together (d : ℕ) (h : d * (17 / 140) = 6 / 7) : d = 17 := by
  sorry

end NUMINAMATH_GPT_work_days_together_l2176_217657


namespace NUMINAMATH_GPT_initial_legos_500_l2176_217686

-- Definitions and conditions from the problem
def initial_legos (x : ℕ) : Prop :=
  let used_pieces := x / 2
  let remaining_pieces := x - used_pieces
  let boxed_pieces := remaining_pieces - 5
  boxed_pieces = 245

-- Statement to be proven
theorem initial_legos_500 : initial_legos 500 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_initial_legos_500_l2176_217686


namespace NUMINAMATH_GPT_inv_geom_seq_prod_next_geom_seq_l2176_217603

variable {a : Nat → ℝ} (q : ℝ) (h_q : q ≠ 0)
variable (h_geom : ∀ n, a (n + 1) = q * a n)

theorem inv_geom_seq :
  ∀ n, ∃ c q_inv, (q_inv ≠ 0) ∧ (1 / a n = c * q_inv ^ n) :=
sorry

theorem prod_next_geom_seq :
  ∀ n, ∃ c q_sq, (q_sq ≠ 0) ∧ (a n * a (n + 1) = c * q_sq ^ n) :=
sorry

end NUMINAMATH_GPT_inv_geom_seq_prod_next_geom_seq_l2176_217603


namespace NUMINAMATH_GPT_sin_double_angle_sub_pi_over_4_l2176_217681

open Real

theorem sin_double_angle_sub_pi_over_4 (x : ℝ) (h : sin x = (sqrt 5 - 1) / 2) : 
  sin (2 * (x - π / 4)) = 2 - sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_sub_pi_over_4_l2176_217681


namespace NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l2176_217615

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) :=
  (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2

theorem arithmetic_sequence_ninth_term
  (a: ℕ → ℕ)
  (h_arith: is_arithmetic_sequence a)
  (h_sum_5: sum_of_first_n_terms a 5 = 75)
  (h_a4: a 4 = 2 * a 2) :
  a 9 = 45 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l2176_217615


namespace NUMINAMATH_GPT_age_product_difference_l2176_217629

theorem age_product_difference (age_today : ℕ) (product_today : ℕ) (product_next_year : ℕ) :
  age_today = 7 →
  product_today = age_today * age_today →
  product_next_year = (age_today + 1) * (age_today + 1) →
  product_next_year - product_today = 15 :=
by
  sorry

end NUMINAMATH_GPT_age_product_difference_l2176_217629


namespace NUMINAMATH_GPT_expense_and_income_calculations_l2176_217611

def alexander_salary : ℕ := 125000
def natalia_salary : ℕ := 61000
def utilities_transport_household : ℕ := 17000
def loan_repayment : ℕ := 15000
def theater_cost : ℕ := 5000
def cinema_cost_per_person : ℕ := 1000
def savings_crimea : ℕ := 20000
def dining_weekday_cost : ℕ := 1500
def dining_weekend_cost : ℕ := 3000
def weekdays : ℕ := 20
def weekends : ℕ := 10
def phone_A_cost : ℕ := 57000
def phone_B_cost : ℕ := 37000

def total_expenses : ℕ :=
  utilities_transport_household +
  loan_repayment +
  theater_cost + 2 * cinema_cost_per_person +
  savings_crimea +
  weekdays * dining_weekday_cost +
  weekends * dining_weekend_cost

def net_income : ℕ :=
  alexander_salary + natalia_salary

def can_buy_phones : Prop :=
  net_income - total_expenses < phone_A_cost + phone_B_cost

theorem expense_and_income_calculations :
  total_expenses = 119000 ∧
  net_income = 186000 ∧
  can_buy_phones :=
by
  sorry

end NUMINAMATH_GPT_expense_and_income_calculations_l2176_217611


namespace NUMINAMATH_GPT_pull_ups_per_time_l2176_217682

theorem pull_ups_per_time (pull_ups_week : ℕ) (times_day : ℕ) (days_week : ℕ)
  (h1 : pull_ups_week = 70) (h2 : times_day = 5) (h3 : days_week = 7) :
  pull_ups_week / (times_day * days_week) = 2 := by
  sorry

end NUMINAMATH_GPT_pull_ups_per_time_l2176_217682


namespace NUMINAMATH_GPT_product_of_last_two_digits_l2176_217696

theorem product_of_last_two_digits (A B : ℕ) (h1 : B = 0 ∨ B = 5) (h2 : A + B = 12) : A * B = 35 :=
by {
  -- proof omitted
  sorry
}

end NUMINAMATH_GPT_product_of_last_two_digits_l2176_217696


namespace NUMINAMATH_GPT_range_of_m_l2176_217637

theorem range_of_m (x m : ℝ) (h₀ : -2 ≤ x ∧ x ≤ 11)
  (h₁ : 1 - 3 * m ≤ x ∧ x ≤ 3 + m)
  (h₂ : ¬ (-2 ≤ x ∧ x ≤ 11) → ¬ (1 - 3 * m ≤ x ∧ x ≤ 3 + m)) :
  m ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2176_217637


namespace NUMINAMATH_GPT_no_cell_with_sum_2018_l2176_217655

theorem no_cell_with_sum_2018 : ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 4900 → (5 * x = 2018 → false) := 
by
  intros x hx
  have h_bound : 1 ≤ x ∧ x ≤ 4900 := hx
  sorry

end NUMINAMATH_GPT_no_cell_with_sum_2018_l2176_217655


namespace NUMINAMATH_GPT_total_balloons_is_72_l2176_217635

-- Definitions for the conditions from the problem
def fred_balloons : Nat := 10
def sam_balloons : Nat := 46
def dan_balloons : Nat := 16

-- The total number of red balloons is the sum of Fred's, Sam's, and Dan's balloons
def total_balloons (f s d : Nat) : Nat := f + s + d

-- The theorem stating the problem to be proved
theorem total_balloons_is_72 : total_balloons fred_balloons sam_balloons dan_balloons = 72 := by
  sorry

end NUMINAMATH_GPT_total_balloons_is_72_l2176_217635


namespace NUMINAMATH_GPT_solve_fraction_equation_l2176_217606

theorem solve_fraction_equation (x : ℚ) :
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 := 
by
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l2176_217606


namespace NUMINAMATH_GPT_line_through_fixed_point_l2176_217676

-- Define the arithmetic sequence condition
def arithmetic_sequence (k b : ℝ) : Prop :=
  k + b = -2

-- Define the line passing through a fixed point
def line_passes_through (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ (x = 1 ∧ y = -2)

-- The theorem stating the main problem
theorem line_through_fixed_point (k b : ℝ) (h : arithmetic_sequence k b) : line_passes_through k b :=
  sorry

end NUMINAMATH_GPT_line_through_fixed_point_l2176_217676


namespace NUMINAMATH_GPT_books_total_l2176_217630

def stuBooks : ℕ := 9
def albertBooks : ℕ := 4 * stuBooks
def totalBooks : ℕ := stuBooks + albertBooks

theorem books_total : totalBooks = 45 := by
  sorry

end NUMINAMATH_GPT_books_total_l2176_217630


namespace NUMINAMATH_GPT_option_C_correct_l2176_217640

-- Define the base a and natural numbers m and n for exponents
variables {a : ℕ} {m n : ℕ}

-- Lean statement to prove (a^5)^3 = a^(5 * 3)
theorem option_C_correct : (a^5)^3 = a^(5 * 3) := 
by sorry

end NUMINAMATH_GPT_option_C_correct_l2176_217640


namespace NUMINAMATH_GPT_prob1_prob2_l2176_217671

-- Definition and theorems related to the calculations of the given problem.
theorem prob1 : ((-12) - 5 + (-14) - (-39)) = 8 := by 
  sorry

theorem prob2 : (-2^2 * 5 - (-12) / 4 - 4) = -21 := by
  sorry

end NUMINAMATH_GPT_prob1_prob2_l2176_217671


namespace NUMINAMATH_GPT_books_difference_l2176_217697

theorem books_difference (bobby_books : ℕ) (kristi_books : ℕ) (h1 : bobby_books = 142) (h2 : kristi_books = 78) : bobby_books - kristi_books = 64 :=
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_books_difference_l2176_217697


namespace NUMINAMATH_GPT_determine_x_l2176_217610

theorem determine_x
  (total_area : ℝ)
  (side_length_square1 : ℝ)
  (side_length_square2 : ℝ)
  (h1 : total_area = 1300)
  (h2 : side_length_square1 = 3 * x)
  (h3 : side_length_square2 = 7 * x) :
    x = Real.sqrt (2600 / 137) :=
by
  sorry

end NUMINAMATH_GPT_determine_x_l2176_217610


namespace NUMINAMATH_GPT_exists_arith_prog_5_primes_exists_arith_prog_6_primes_l2176_217633

-- Define the condition of being an arithmetic progression
def is_arith_prog (seq : List ℕ) : Prop :=
  ∀ (i : ℕ), i < seq.length - 1 → seq.get! (i + 1) - seq.get! i = seq.get! 1 - seq.get! 0

-- Define the condition of being prime
def all_prime (seq : List ℕ) : Prop :=
  ∀ (n : ℕ), n ∈ seq → Nat.Prime n

-- The main statements
theorem exists_arith_prog_5_primes :
  ∃ (seq : List ℕ), seq.length = 5 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

theorem exists_arith_prog_6_primes :
  ∃ (seq : List ℕ), seq.length = 6 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

end NUMINAMATH_GPT_exists_arith_prog_5_primes_exists_arith_prog_6_primes_l2176_217633


namespace NUMINAMATH_GPT_cubic_root_identity_l2176_217677

theorem cubic_root_identity (r : ℝ) (h : (r^(1/3)) - (1/(r^(1/3))) = 2) : r^3 - (1/r^3) = 14 := 
by 
  sorry

end NUMINAMATH_GPT_cubic_root_identity_l2176_217677


namespace NUMINAMATH_GPT_cost_of_each_burger_l2176_217651

theorem cost_of_each_burger (purchases_per_day : ℕ) (total_days : ℕ) (total_amount_spent : ℕ)
  (h1 : purchases_per_day = 4) (h2 : total_days = 30) (h3 : total_amount_spent = 1560) : 
  total_amount_spent / (purchases_per_day * total_days) = 13 :=
by
  subst h1
  subst h2
  subst h3
  sorry

end NUMINAMATH_GPT_cost_of_each_burger_l2176_217651


namespace NUMINAMATH_GPT_sun_xing_zhe_problem_l2176_217691

theorem sun_xing_zhe_problem (S X Z : ℕ) (h : S < 10 ∧ X < 10 ∧ Z < 10)
  (hprod : (100 * S + 10 * X + Z) * (100 * Z + 10 * X + S) = 78445) :
  (100 * S + 10 * X + Z) + (100 * Z + 10 * X + S) = 1372 := 
by
  sorry

end NUMINAMATH_GPT_sun_xing_zhe_problem_l2176_217691


namespace NUMINAMATH_GPT_base_of_third_term_l2176_217653

theorem base_of_third_term (x : ℝ) (some_number : ℝ) :
  625^(-x) + 25^(-2 * x) + some_number^(-4 * x) = 14 → x = 0.25 → some_number = 125 / 1744 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_base_of_third_term_l2176_217653


namespace NUMINAMATH_GPT_correct_mark_proof_l2176_217608

-- Define the conditions
def wrong_mark := 85
def increase_in_average : ℝ := 0.5
def number_of_pupils : ℕ := 104

-- Define the correct mark to be proven
noncomputable def correct_mark : ℕ := 33

-- Statement to be proven
theorem correct_mark_proof (x : ℝ) :
  (wrong_mark - x) / number_of_pupils = increase_in_average → x = correct_mark :=
by
  sorry

end NUMINAMATH_GPT_correct_mark_proof_l2176_217608


namespace NUMINAMATH_GPT_product_evaluation_l2176_217664

theorem product_evaluation : 
  (7 - 5) * (7 - 4) * (7 - 3) * (7 - 2) * (7- 1) * 7 = 5040 := 
by 
  sorry

end NUMINAMATH_GPT_product_evaluation_l2176_217664


namespace NUMINAMATH_GPT_solve_graph_equation_l2176_217669

/- Problem:
Solve for the graph of the equation x^2(x+y+2)=y^2(x+y+2)
Given condition: equation x^2(x+y+2)=y^2(x+y+2)
Conclusion: Three lines that do not all pass through a common point
The final answer should be formally proven.
-/

theorem solve_graph_equation (x y : ℝ) :
  (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (∃ a b c d : ℝ,  (a = -x - 2 ∧ b = -x ∧ c = x ∧ (a ≠ b ∧ a ≠ c ∧ b ≠ c)) ∧
   (d = 0) ∧ ¬ ∀ p q r : ℝ, p = q ∧ q = r ∧ r = p) :=
by
  sorry

end NUMINAMATH_GPT_solve_graph_equation_l2176_217669


namespace NUMINAMATH_GPT_determine_k_circle_l2176_217685

theorem determine_k_circle (k : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 14*y - k = 0) ∧ ((∀ x y : ℝ, (x + 4)^2 + (y + 7)^2 = 25) ↔ k = -40) :=
by
  sorry

end NUMINAMATH_GPT_determine_k_circle_l2176_217685


namespace NUMINAMATH_GPT_no_integer_roots_l2176_217620

theorem no_integer_roots (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) : 
  ¬ ∃ x : ℤ, a * x^2 + b * x + c = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_no_integer_roots_l2176_217620


namespace NUMINAMATH_GPT_cristina_running_pace_l2176_217621

theorem cristina_running_pace
  (nicky_pace : ℝ) (nicky_headstart : ℝ) (time_nicky_run : ℝ) 
  (distance_nicky_run : ℝ) (time_cristina_catch : ℝ) :
  (nicky_pace = 3) →
  (nicky_headstart = 12) →
  (time_nicky_run = 30) →
  (distance_nicky_run = nicky_pace * time_nicky_run) →
  (time_cristina_catch = time_nicky_run - nicky_headstart) →
  (cristina_pace : ℝ) →
  (cristina_pace = distance_nicky_run / time_cristina_catch) →
  cristina_pace = 5 :=
by
  sorry

end NUMINAMATH_GPT_cristina_running_pace_l2176_217621


namespace NUMINAMATH_GPT_radius_of_sphere_is_two_sqrt_46_l2176_217678

theorem radius_of_sphere_is_two_sqrt_46
  (a b c : ℝ)
  (s : ℝ)
  (h1 : 4 * (a + b + c) = 160)
  (h2 : 2 * (a * b + b * c + c * a) = 864)
  (h3 : s = Real.sqrt ((a^2 + b^2 + c^2) / 4)) :
  s = 2 * Real.sqrt 46 :=
by
  -- proof placeholder
  sorry

end NUMINAMATH_GPT_radius_of_sphere_is_two_sqrt_46_l2176_217678


namespace NUMINAMATH_GPT_remainder_67_pow_67_plus_67_mod_68_l2176_217660

theorem remainder_67_pow_67_plus_67_mod_68 :
  (67 ^ 67 + 67) % 68 = 66 :=
by
  -- Skip the proof for now
  sorry

end NUMINAMATH_GPT_remainder_67_pow_67_plus_67_mod_68_l2176_217660


namespace NUMINAMATH_GPT_donna_total_episodes_per_week_l2176_217636

-- Defining the conditions
def episodes_per_weekday : ℕ := 8
def weekday_count : ℕ := 5
def weekend_factor : ℕ := 3
def weekend_count : ℕ := 2

-- Theorem statement
theorem donna_total_episodes_per_week :
  (episodes_per_weekday * weekday_count) + ((episodes_per_weekday * weekend_factor) * weekend_count) = 88 := 
  by sorry

end NUMINAMATH_GPT_donna_total_episodes_per_week_l2176_217636


namespace NUMINAMATH_GPT_range_of_a_l2176_217693

theorem range_of_a :
  (∃ a : ℝ, (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0)
    ↔ a ≤ -2 ∨ a = 1) := 
sorry

end NUMINAMATH_GPT_range_of_a_l2176_217693


namespace NUMINAMATH_GPT_value_subtracted_l2176_217667

theorem value_subtracted (x y : ℤ) (h1 : (x - 5) / 7 = 7) (h2 : (x - y) / 13 = 4) : y = 2 :=
sorry

end NUMINAMATH_GPT_value_subtracted_l2176_217667


namespace NUMINAMATH_GPT_calories_difference_l2176_217634

theorem calories_difference
  (calories_squirrel : ℕ := 300)
  (squirrels_per_hour : ℕ := 6)
  (calories_rabbit : ℕ := 800)
  (rabbits_per_hour : ℕ := 2) :
  ((squirrels_per_hour * calories_squirrel) - (rabbits_per_hour * calories_rabbit)) = 200 :=
by
  sorry

end NUMINAMATH_GPT_calories_difference_l2176_217634


namespace NUMINAMATH_GPT_ratio_of_ages_ten_years_ago_l2176_217668

theorem ratio_of_ages_ten_years_ago (A T : ℕ) 
    (h1: A = 30) 
    (h2: T = A - 15) : 
    (A - 10) / (T - 10) = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_ten_years_ago_l2176_217668


namespace NUMINAMATH_GPT_simplify_tan_product_l2176_217654

-- Mathematical Conditions
def tan_inv (x : ℝ) : ℝ := sorry
noncomputable def tan (θ : ℝ) : ℝ := sorry

-- Problem statement to be proven
theorem simplify_tan_product (x y : ℝ) (hx : tan_inv x = 10) (hy : tan_inv y = 35) :
  (1 + x) * (1 + y) = 2 :=
sorry

end NUMINAMATH_GPT_simplify_tan_product_l2176_217654


namespace NUMINAMATH_GPT_find_y_eq_l2176_217673

theorem find_y_eq (y : ℝ) : (10 - y)^2 = 4 * y^2 → (y = 10 / 3 ∨ y = -10) :=
by
  intro h
  -- The detailed proof will be provided here
  sorry

end NUMINAMATH_GPT_find_y_eq_l2176_217673


namespace NUMINAMATH_GPT_pencils_per_child_l2176_217632

theorem pencils_per_child (children : ℕ) (total_pencils : ℕ) (h1 : children = 2) (h2 : total_pencils = 12) :
  total_pencils / children = 6 :=
by 
  sorry

end NUMINAMATH_GPT_pencils_per_child_l2176_217632


namespace NUMINAMATH_GPT_power_product_to_seventh_power_l2176_217613

theorem power_product_to_seventh_power :
  (2 ^ 14) * (2 ^ 21) = (32 ^ 7) :=
by
  sorry

end NUMINAMATH_GPT_power_product_to_seventh_power_l2176_217613


namespace NUMINAMATH_GPT_emails_in_morning_and_evening_l2176_217628

def morning_emails : ℕ := 3
def afternoon_emails : ℕ := 4
def evening_emails : ℕ := 8

theorem emails_in_morning_and_evening : morning_emails + evening_emails = 11 :=
by
  sorry

end NUMINAMATH_GPT_emails_in_morning_and_evening_l2176_217628


namespace NUMINAMATH_GPT_coffee_cost_per_week_l2176_217694

theorem coffee_cost_per_week 
  (people_in_house : ℕ) 
  (drinks_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (num_days_in_week : ℕ) 
  (h1 : people_in_house = 4) 
  (h2 : drinks_per_person_per_day = 2)
  (h3 : ounces_per_cup = 0.5)
  (h4 : cost_per_ounce = 1.25)
  (h5 : num_days_in_week = 7) :
  people_in_house * drinks_per_person_per_day * ounces_per_cup * cost_per_ounce * num_days_in_week = 35 := 
by
  sorry

end NUMINAMATH_GPT_coffee_cost_per_week_l2176_217694


namespace NUMINAMATH_GPT_half_angle_quadrant_l2176_217627

theorem half_angle_quadrant (α : ℝ) (k : ℤ) (h1 : k * 360 + 180 < α) (h2 : α < k * 360 + 270) :
    (∃ n : ℤ, (n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ (n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315)) :=
sorry

end NUMINAMATH_GPT_half_angle_quadrant_l2176_217627
