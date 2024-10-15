import Mathlib

namespace NUMINAMATH_GPT_find_positive_k_l1367_136716

noncomputable def polynomial_with_equal_roots (k: ℚ) : Prop := 
  ∃ a b : ℚ, a ≠ b ∧ 2 * a + b = -3 ∧ 2 * a * b + a^2 = -50 ∧ k = -2 * a^2 * b

theorem find_positive_k : ∃ k : ℚ, polynomial_with_equal_roots k ∧ 0 < k ∧ k = 950 / 27 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_k_l1367_136716


namespace NUMINAMATH_GPT_part1_part2_l1367_136760

noncomputable def f (a x : ℝ) : ℝ := (Real.exp x) - a * x ^ 2 - x

theorem part1 {a : ℝ} : (∀ x y: ℝ, x < y → f a x ≤ f a y) ↔ (a = 1 / 2) :=
sorry

theorem part2 {a : ℝ} (h1 : a > 1 / 2):
  ∃ (x1 x2 : ℝ), (x1 < x2) ∧ (f a x2 < 1 + (Real.sin x2 - x2) / 2) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1367_136760


namespace NUMINAMATH_GPT_sum_of_undefined_fractions_l1367_136794

theorem sum_of_undefined_fractions (x₁ x₂ : ℝ) (h₁ : x₁^2 - 7*x₁ + 12 = 0) (h₂ : x₂^2 - 7*x₂ + 12 = 0) :
  x₁ + x₂ = 7 :=
sorry

end NUMINAMATH_GPT_sum_of_undefined_fractions_l1367_136794


namespace NUMINAMATH_GPT_vasya_can_construct_polyhedron_l1367_136750

-- Definition of a polyhedron using given set of shapes
-- where the original set of shapes can form a polyhedron
def original_set_can_form_polyhedron (squares triangles : ℕ) : Prop :=
  squares = 1 ∧ triangles = 4

-- Transformation condition: replacing 2 triangles with 2 squares
def replacement_condition (initial_squares initial_triangles replaced_squares replaced_triangles : ℕ) : Prop :=
  initial_squares + 2 = replaced_squares ∧ initial_triangles - 2 = replaced_triangles

-- Proving that new set of shapes can form a polyhedron
theorem vasya_can_construct_polyhedron :
  ∃ (new_squares new_triangles : ℕ),
    (original_set_can_form_polyhedron 1 4)
    ∧ (replacement_condition 1 4 new_squares new_triangles)
    ∧ (new_squares = 3 ∧ new_triangles = 2) :=
by
  sorry

end NUMINAMATH_GPT_vasya_can_construct_polyhedron_l1367_136750


namespace NUMINAMATH_GPT_upstream_distance_calc_l1367_136728

noncomputable def speed_in_still_water : ℝ := 10.5
noncomputable def downstream_distance : ℝ := 45
noncomputable def downstream_time : ℝ := 3
noncomputable def upstream_time : ℝ := 3

theorem upstream_distance_calc : 
  ∃ (d v : ℝ), (10.5 + v) * downstream_time = downstream_distance ∧ 
               v = 4.5 ∧ 
               d = (10.5 - v) * upstream_time ∧ 
               d = 18 :=
by
  sorry

end NUMINAMATH_GPT_upstream_distance_calc_l1367_136728


namespace NUMINAMATH_GPT_shaded_areas_sum_l1367_136700

theorem shaded_areas_sum (triangle_area : ℕ) (parts : ℕ)
  (h1 : triangle_area = 18)
  (h2 : parts = 9) :
  3 * (triangle_area / parts) = 6 :=
by
  sorry

end NUMINAMATH_GPT_shaded_areas_sum_l1367_136700


namespace NUMINAMATH_GPT_solve_for_x_l1367_136705

variable (x : ℝ)

theorem solve_for_x (h : 5 * x - 3 = 17) : x = 4 := sorry

end NUMINAMATH_GPT_solve_for_x_l1367_136705


namespace NUMINAMATH_GPT_ways_to_divide_day_l1367_136754

theorem ways_to_divide_day (n m : ℕ+) : n * m = 86400 → 96 = 96 :=
by
  sorry

end NUMINAMATH_GPT_ways_to_divide_day_l1367_136754


namespace NUMINAMATH_GPT_cos_alpha_minus_pi_over_4_l1367_136736

theorem cos_alpha_minus_pi_over_4 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (h_tan : Real.tan α = 2) :
  Real.cos (α - π / 4) = (3 * Real.sqrt 10) / 10 := 
  sorry

end NUMINAMATH_GPT_cos_alpha_minus_pi_over_4_l1367_136736


namespace NUMINAMATH_GPT_mans_rate_in_still_water_l1367_136783

theorem mans_rate_in_still_water (R S : ℝ) (h1 : R + S = 18) (h2 : R - S = 4) : R = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_mans_rate_in_still_water_l1367_136783


namespace NUMINAMATH_GPT_Jake_weight_l1367_136701

variables (J S : ℝ)

theorem Jake_weight (h1 : 0.8 * J = 2 * S) (h2 : J + S = 168) : J = 120 :=
  sorry

end NUMINAMATH_GPT_Jake_weight_l1367_136701


namespace NUMINAMATH_GPT_sandy_comic_books_l1367_136782

-- Problem definition
def initial_comic_books := 14
def sold_comic_books := initial_comic_books / 2
def remaining_comic_books := initial_comic_books - sold_comic_books
def bought_comic_books := 6
def final_comic_books := remaining_comic_books + bought_comic_books

-- Proof statement
theorem sandy_comic_books : final_comic_books = 13 := by
  sorry

end NUMINAMATH_GPT_sandy_comic_books_l1367_136782


namespace NUMINAMATH_GPT_inscribed_sphere_radius_base_height_l1367_136755

noncomputable def radius_of_inscribed_sphere (r base_radius height : ℝ) := 
  r = (30 / (Real.sqrt 5 + 1)) * (Real.sqrt 5 - 1) 

theorem inscribed_sphere_radius_base_height (r : ℝ) (b d : ℝ) (base_radius height : ℝ) 
  (h_base: base_radius = 15) (h_height: height = 30) 
  (h_radius: radius_of_inscribed_sphere r base_radius height) 
  (h_expr: r = b * (Real.sqrt d) - b) : 
  b + d = 12.5 :=
sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_base_height_l1367_136755


namespace NUMINAMATH_GPT_min_value_abs_expression_l1367_136732

theorem min_value_abs_expression {p x : ℝ} (hp1 : 0 < p) (hp2 : p < 15) (hx1 : p ≤ x) (hx2 : x ≤ 15) :
  |x - p| + |x - 15| + |x - p - 15| = 15 :=
sorry

end NUMINAMATH_GPT_min_value_abs_expression_l1367_136732


namespace NUMINAMATH_GPT_rational_product_sum_l1367_136726

theorem rational_product_sum (x y : ℚ) 
  (h1 : x * y < 0) 
  (h2 : x + y < 0) : 
  |y| < |x| ∧ y < 0 ∧ x > 0 ∨ |x| < |y| ∧ x < 0 ∧ y > 0 :=
by
  sorry

end NUMINAMATH_GPT_rational_product_sum_l1367_136726


namespace NUMINAMATH_GPT_ratio_of_professionals_l1367_136763

-- Define the variables and conditions as stated in the problem.
variables (e d l : ℕ)

-- The condition about the average ages leading to the given equation.
def avg_age_condition : Prop := (40 * e + 50 * d + 60 * l) / (e + d + l) = 45

-- The statement to prove that given the average age condition, the ratio is 1:1:3.
theorem ratio_of_professionals (h : avg_age_condition e d l) : e = d + 3 * l :=
sorry

end NUMINAMATH_GPT_ratio_of_professionals_l1367_136763


namespace NUMINAMATH_GPT_part1_solution_part2_no_solution_l1367_136796

theorem part1_solution (x y : ℚ) :
  x + y = 5 ∧ 3 * x + 10 * y = 30 ↔ x = 20 / 7 ∧ y = 15 / 7 :=
by
  sorry

theorem part2_no_solution (x : ℚ) :
  (x + 7) / 2 < 4 ∧ (3 * x - 1) / 2 ≤ 2 * x - 3 ↔ False :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_part2_no_solution_l1367_136796


namespace NUMINAMATH_GPT_abs_linear_combination_l1367_136711

theorem abs_linear_combination (a b : ℝ) :
  (∀ x y : ℝ, |a * x + b * y| + |b * x + a * y| = |x| + |y|) →
  (a = 1 ∧ b = 0) ∨ (a = 0 ∧ b = 1) ∨ (a = 0 ∧ b = -1) ∨ (a = -1 ∧ b = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_abs_linear_combination_l1367_136711


namespace NUMINAMATH_GPT_invertible_elements_mod_8_l1367_136765

theorem invertible_elements_mod_8 :
  {x : ℤ | (x * x) % 8 = 1} = {1, 3, 5, 7} :=
by
  sorry

end NUMINAMATH_GPT_invertible_elements_mod_8_l1367_136765


namespace NUMINAMATH_GPT_smallest_solution_x4_minus_50x2_plus_625_eq_0_l1367_136771

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 :
  ∃ (x : ℝ), (x * x * x * x - 50 * x * x + 625 = 0) ∧ (∀ y, (y * y * y * y - 50 * y * y + 625 = 0) → x ≤ y) :=
sorry

end NUMINAMATH_GPT_smallest_solution_x4_minus_50x2_plus_625_eq_0_l1367_136771


namespace NUMINAMATH_GPT_shanghai_expo_visitors_l1367_136769

theorem shanghai_expo_visitors :
  505000 = 5.05 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_shanghai_expo_visitors_l1367_136769


namespace NUMINAMATH_GPT_single_reduction_equivalent_l1367_136770

theorem single_reduction_equivalent (P : ℝ) (h1 : P > 0) :
  let final_price := 0.75 * P - 0.7 * (0.75 * P)
  let single_reduction := (P - final_price) / P
  single_reduction * 100 = 77.5 := 
by
  sorry

end NUMINAMATH_GPT_single_reduction_equivalent_l1367_136770


namespace NUMINAMATH_GPT_intersection_of_curves_l1367_136742

theorem intersection_of_curves (x : ℝ) (y : ℝ) (h₁ : y = 9 / (x^2 + 3)) (h₂ : x + y = 3) : x = 0 :=
sorry

end NUMINAMATH_GPT_intersection_of_curves_l1367_136742


namespace NUMINAMATH_GPT_centroid_value_l1367_136720

-- Define the points P, Q, R
def P : ℝ × ℝ := (4, 3)
def Q : ℝ × ℝ := (-1, 6)
def R : ℝ × ℝ := (7, -2)

-- Define the coordinates of the centroid S
noncomputable def S : ℝ × ℝ := 
  ( (4 + (-1) + 7) / 3, (3 + 6 + (-2)) / 3 )

-- Statement to prove
theorem centroid_value : 
  let x := (4 + (-1) + 7) / 3
  let y := (3 + 6 + (-2)) / 3
  8 * x + 3 * y = 101 / 3 :=
by
  let x := (4 + (-1) + 7) / 3
  let y := (3 + 6 + (-2)) / 3
  have h: 8 * x + 3 * y = 101 / 3 := sorry
  exact h

end NUMINAMATH_GPT_centroid_value_l1367_136720


namespace NUMINAMATH_GPT_bob_bakes_pie_in_6_minutes_l1367_136799

theorem bob_bakes_pie_in_6_minutes (x : ℕ) (h_alice : 60 / 5 = 12)
  (h_condition : 12 - 2 = 60 / x) : x = 6 :=
sorry

end NUMINAMATH_GPT_bob_bakes_pie_in_6_minutes_l1367_136799


namespace NUMINAMATH_GPT_no_strategy_for_vasya_tolya_l1367_136721

-- This definition encapsulates the conditions and question
def players_game (coins : ℕ) : Prop :=
  ∀ p v t : ℕ, 
    (1 ≤ p ∧ p ≤ 4) ∧ (1 ≤ v ∧ v ≤ 2) ∧ (1 ≤ t ∧ t ≤ 2) →
    (∃ (n : ℕ), coins = 5 * n)

-- Theorem formalizing the problem's conclusion
theorem no_strategy_for_vasya_tolya (n : ℕ) (h : n = 300) : 
  ¬ ∀ (v t : ℕ), 
     (1 ≤ v ∧ v ≤ 2) ∧ (1 ≤ t ∧ t ≤ 2) →
     players_game (n - v - t) :=
by
  intro h
  sorry -- Skip the proof, as it is not required

end NUMINAMATH_GPT_no_strategy_for_vasya_tolya_l1367_136721


namespace NUMINAMATH_GPT_problem1_problem2_l1367_136789

-- Problem 1
theorem problem1 (a b : ℝ) : 
  a^2 * (2 * a * b - 1) + (a - 3 * b) * (a + b) = 2 * a^3 * b - 2 * a * b - 3 * b^2 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (2 * x - 3)^2 - (x + 2)^2 = 3 * x^2 - 16 * x + 5 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1367_136789


namespace NUMINAMATH_GPT_cannot_achieve_1970_minuses_l1367_136733

theorem cannot_achieve_1970_minuses :
  ∃ (x y : ℕ), x ≤ 100 ∧ y ≤ 100 ∧ (x - 50) * (y - 50) = 1515 → false :=
by
  sorry

end NUMINAMATH_GPT_cannot_achieve_1970_minuses_l1367_136733


namespace NUMINAMATH_GPT_cage_cost_correct_l1367_136752

def cost_of_cat_toy : Real := 10.22
def total_cost_of_purchases : Real := 21.95
def cost_of_cage : Real := total_cost_of_purchases - cost_of_cat_toy

theorem cage_cost_correct : cost_of_cage = 11.73 := by
  sorry

end NUMINAMATH_GPT_cage_cost_correct_l1367_136752


namespace NUMINAMATH_GPT_parabola_directrix_l1367_136768

variable (a : ℝ)

theorem parabola_directrix (h1 : ∀ x : ℝ, y = a * x^2) (h2 : y = -1/4) : a = 1 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l1367_136768


namespace NUMINAMATH_GPT_molecular_weight_of_one_mole_l1367_136739

-- Definitions as Conditions
def total_molecular_weight := 960
def number_of_moles := 5

-- The theorem statement
theorem molecular_weight_of_one_mole :
  total_molecular_weight / number_of_moles = 192 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_one_mole_l1367_136739


namespace NUMINAMATH_GPT_pipes_fill_tank_in_10_hours_l1367_136793

noncomputable def R_A := 1 / 70
noncomputable def R_B := 2 * R_A
noncomputable def R_C := 2 * R_B
noncomputable def R_total := R_A + R_B + R_C
noncomputable def T := 1 / R_total

theorem pipes_fill_tank_in_10_hours :
  T = 10 := 
sorry

end NUMINAMATH_GPT_pipes_fill_tank_in_10_hours_l1367_136793


namespace NUMINAMATH_GPT_additional_pass_combinations_l1367_136719

def original_combinations : ℕ := 4 * 2 * 3 * 3
def new_combinations : ℕ := 6 * 2 * 4 * 3
def additional_combinations : ℕ := new_combinations - original_combinations

theorem additional_pass_combinations : additional_combinations = 72 := by
  sorry

end NUMINAMATH_GPT_additional_pass_combinations_l1367_136719


namespace NUMINAMATH_GPT_cameron_list_count_l1367_136767

theorem cameron_list_count : 
  (∃ (n m : ℕ), n = 900 ∧ m = 27000 ∧ (∀ k : ℕ, (30 * k) ≥ n ∧ (30 * k) ≤ m → ∃ count : ℕ, count = 871)) :=
by
  sorry

end NUMINAMATH_GPT_cameron_list_count_l1367_136767


namespace NUMINAMATH_GPT_dice_probability_l1367_136738

def prob_at_least_one_one : ℚ :=
  let total_outcomes := 36
  let no_1_outcomes := 25
  let favorable_outcomes := total_outcomes - no_1_outcomes
  let probability := favorable_outcomes / total_outcomes
  probability

theorem dice_probability :
  prob_at_least_one_one = 11 / 36 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_l1367_136738


namespace NUMINAMATH_GPT_functional_equation_solution_l1367_136714

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (f (f x + f y)) = f x + y) : ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1367_136714


namespace NUMINAMATH_GPT_pasha_game_solvable_l1367_136781

def pasha_game : Prop :=
∃ (a : Fin 2017 → ℕ), 
  (∀ i, a i > 0) ∧
  (∃ (moves : ℕ), moves = 43 ∧
   (∀ (box_contents : Fin 2017 → ℕ), 
    (∀ j, box_contents j = 0) →
    (∃ (equal_count : ℕ),
      (∀ j, box_contents j = equal_count)
      ∧
      (∀ m < 43,
        ∃ j, box_contents j ≠ equal_count))))

theorem pasha_game_solvable : pasha_game :=
by
  sorry

end NUMINAMATH_GPT_pasha_game_solvable_l1367_136781


namespace NUMINAMATH_GPT_new_person_weight_l1367_136775

theorem new_person_weight (W : ℝ) (old_weight : ℝ) (increase_per_person : ℝ) (num_persons : ℕ)
  (h1 : old_weight = 68)
  (h2 : increase_per_person = 5.5)
  (h3 : num_persons = 5)
  (h4 : W = old_weight + increase_per_person * num_persons) :
  W = 95.5 :=
by
  sorry

end NUMINAMATH_GPT_new_person_weight_l1367_136775


namespace NUMINAMATH_GPT_simplify_abs_expression_l1367_136715

theorem simplify_abs_expression (a b c : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : c = 0) :
  |a - c| + |c - b| - |a - b| = 0 := 
by
  sorry

end NUMINAMATH_GPT_simplify_abs_expression_l1367_136715


namespace NUMINAMATH_GPT_sally_took_home_pens_l1367_136737

theorem sally_took_home_pens
    (initial_pens : ℕ)
    (students : ℕ)
    (pens_per_student : ℕ)
    (locker_fraction : ℕ)
    (total_pens_given : ℕ)
    (remainder : ℕ)
    (locker_pens : ℕ)
    (home_pens : ℕ) :
    initial_pens = 5230 →
    students = 89 →
    pens_per_student = 58 →
    locker_fraction = 2 →
    total_pens_given = students * pens_per_student →
    remainder = initial_pens - total_pens_given →
    locker_pens = remainder / locker_fraction →
    home_pens = locker_pens →
    home_pens = 34 :=
by {
  sorry
}

end NUMINAMATH_GPT_sally_took_home_pens_l1367_136737


namespace NUMINAMATH_GPT_number_of_ways_to_choose_one_top_and_one_bottom_l1367_136707

theorem number_of_ways_to_choose_one_top_and_one_bottom :
  let number_of_hoodies := 5
  let number_of_sweatshirts := 4
  let number_of_jeans := 3
  let number_of_slacks := 5
  let total_tops := number_of_hoodies + number_of_sweatshirts
  let total_bottoms := number_of_jeans + number_of_slacks
  total_tops * total_bottoms = 72 := 
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_one_top_and_one_bottom_l1367_136707


namespace NUMINAMATH_GPT_find_constant_a_l1367_136741

theorem find_constant_a (S : ℕ → ℝ) (a : ℝ) :
  (∀ n, S n = (1/2) * 3^(n+1) - a) →
  a = 3/2 :=
sorry

end NUMINAMATH_GPT_find_constant_a_l1367_136741


namespace NUMINAMATH_GPT_smallest_perfect_square_divisible_by_5_and_7_l1367_136798

theorem smallest_perfect_square_divisible_by_5_and_7 
  (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = k^2)
  (h3 : 5 ∣ n)
  (h4 : 7 ∣ n) : 
  n = 1225 :=
sorry

end NUMINAMATH_GPT_smallest_perfect_square_divisible_by_5_and_7_l1367_136798


namespace NUMINAMATH_GPT_company_pays_each_man_per_hour_l1367_136773

theorem company_pays_each_man_per_hour
  (men : ℕ) (hours_per_job : ℕ) (jobs : ℕ) (total_pay : ℕ)
  (completion_time : men * hours_per_job = 1)
  (total_jobs_time : jobs * hours_per_job = 5)
  (total_earning : total_pay = 150) :
  (total_pay / (jobs * men * hours_per_job)) = 10 :=
sorry

end NUMINAMATH_GPT_company_pays_each_man_per_hour_l1367_136773


namespace NUMINAMATH_GPT_division_remainder_is_7_l1367_136722

theorem division_remainder_is_7 (d q D r : ℕ) (hd : d = 21) (hq : q = 14) (hD : D = 301) (h_eq : D = d * q + r) : r = 7 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_is_7_l1367_136722


namespace NUMINAMATH_GPT_animath_workshop_lists_l1367_136751

/-- The 79 trainees of the Animath workshop each choose an activity for the free afternoon 
among 5 offered activities. It is known that:
- The swimming pool was at least as popular as soccer.
- The students went shopping in groups of 5.
- No more than 4 students played cards.
- At most one student stayed in their room.
We write down the number of students who participated in each activity.
How many different lists could we have written? --/
theorem animath_workshop_lists :
  ∃ (l : ℕ), l = Nat.choose 81 2 := 
sorry

end NUMINAMATH_GPT_animath_workshop_lists_l1367_136751


namespace NUMINAMATH_GPT_correct_calculation_l1367_136724

-- Definitions of the conditions
def condition1 : Prop := 3 + Real.sqrt 3 ≠ 3 * Real.sqrt 3
def condition2 : Prop := 2 * Real.sqrt 3 + Real.sqrt 3 = 3 * Real.sqrt 3
def condition3 : Prop := 2 * Real.sqrt 3 - Real.sqrt 3 ≠ 2
def condition4 : Prop := Real.sqrt 3 + Real.sqrt 2 ≠ Real.sqrt 5

-- Proposition using the conditions to state the correct calculation
theorem correct_calculation (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  2 * Real.sqrt 3 + Real.sqrt 3 = 3 * Real.sqrt 3 :=
by
  exact h2

end NUMINAMATH_GPT_correct_calculation_l1367_136724


namespace NUMINAMATH_GPT_geoff_total_spending_l1367_136764

def price_day1 : ℕ := 60
def pairs_day1 : ℕ := 2
def price_per_pair_day1 : ℕ := price_day1 / pairs_day1

def multiplier_day2 : ℕ := 3
def price_per_pair_day2 : ℕ := price_per_pair_day1 * 3 / 2
def discount_day2 : Real := 0.10
def cost_before_discount_day2 : ℕ := multiplier_day2 * price_per_pair_day2
def cost_after_discount_day2 : Real := cost_before_discount_day2 * (1 - discount_day2)

def multiplier_day3 : ℕ := 5
def price_per_pair_day3 : ℕ := price_per_pair_day1 * 2
def sales_tax_day3 : Real := 0.08
def cost_before_tax_day3 : ℕ := multiplier_day3 * price_per_pair_day3
def cost_after_tax_day3 : Real := cost_before_tax_day3 * (1 + sales_tax_day3)

def total_cost : Real := price_day1 + cost_after_discount_day2 + cost_after_tax_day3

theorem geoff_total_spending : total_cost = 505.50 := by
  sorry

end NUMINAMATH_GPT_geoff_total_spending_l1367_136764


namespace NUMINAMATH_GPT_expIConjugate_l1367_136747

open Complex

-- Define the given condition
def expICondition (θ φ : ℝ) : Prop :=
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I

-- The theorem we want to prove
theorem expIConjugate (θ φ : ℝ) (h : expICondition θ φ) : 
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I :=
sorry

end NUMINAMATH_GPT_expIConjugate_l1367_136747


namespace NUMINAMATH_GPT_prob_TeamA_wins_2_1_proof_prob_TeamB_wins_proof_best_of_five_increases_prob_l1367_136786

noncomputable def prob_TeamA_wins_game : ℝ := 0.6
noncomputable def prob_TeamB_wins_game : ℝ := 0.4

-- Probability of Team A winning 2-1 in a best-of-three
noncomputable def prob_TeamA_wins_2_1 : ℝ := 2 * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game 

-- Probability of Team B winning in a best-of-three
noncomputable def prob_TeamB_wins_2_0 : ℝ := prob_TeamB_wins_game * prob_TeamB_wins_game
noncomputable def prob_TeamB_wins_2_1 : ℝ := 2 * prob_TeamB_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game
noncomputable def prob_TeamB_wins : ℝ := prob_TeamB_wins_2_0 + prob_TeamB_wins_2_1

-- Probability of Team A winning in a best-of-three
noncomputable def prob_TeamA_wins_best_of_three : ℝ := 1 - prob_TeamB_wins

-- Probability of Team A winning in a best-of-five
noncomputable def prob_TeamA_wins_3_0 : ℝ := prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamA_wins_game
noncomputable def prob_TeamA_wins_3_1 : ℝ := 3 * (prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game)
noncomputable def prob_TeamA_wins_3_2 : ℝ := 6 * (prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game)

noncomputable def prob_TeamA_wins_best_of_five : ℝ := prob_TeamA_wins_3_0 + prob_TeamA_wins_3_1 + prob_TeamA_wins_3_2

theorem prob_TeamA_wins_2_1_proof :
  prob_TeamA_wins_2_1 = 0.288 :=
sorry

theorem prob_TeamB_wins_proof :
  prob_TeamB_wins = 0.352 :=
sorry

theorem best_of_five_increases_prob :
  prob_TeamA_wins_best_of_three < prob_TeamA_wins_best_of_five :=
sorry

end NUMINAMATH_GPT_prob_TeamA_wins_2_1_proof_prob_TeamB_wins_proof_best_of_five_increases_prob_l1367_136786


namespace NUMINAMATH_GPT_compute_expression_l1367_136792

theorem compute_expression : 12 * (1 / 17) * 34 = 24 :=
by sorry

end NUMINAMATH_GPT_compute_expression_l1367_136792


namespace NUMINAMATH_GPT_arithmetic_sequence_length_l1367_136784

theorem arithmetic_sequence_length :
  ∃ n : ℕ, ∀ (a d l : ℕ), a = 2 → d = 5 → l = 3007 → l = a + (n-1) * d → n = 602 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_length_l1367_136784


namespace NUMINAMATH_GPT_infinite_series_sum_l1367_136779

theorem infinite_series_sum :
  (∑' n : Nat, (4 * n + 1) / ((4 * n - 1)^2 * (4 * n + 3)^2)) = 1 / 72 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l1367_136779


namespace NUMINAMATH_GPT_emily_sixth_quiz_score_l1367_136790

theorem emily_sixth_quiz_score (a1 a2 a3 a4 a5 : ℕ) (target_mean : ℕ) (sixth_score : ℕ) :
  a1 = 94 ∧ a2 = 97 ∧ a3 = 88 ∧ a4 = 90 ∧ a5 = 102 ∧ target_mean = 95 →
  sixth_score = (target_mean * 6 - (a1 + a2 + a3 + a4 + a5)) →
  sixth_score = 99 :=
by
  sorry

end NUMINAMATH_GPT_emily_sixth_quiz_score_l1367_136790


namespace NUMINAMATH_GPT_tan_alpha_eq_one_l1367_136785

open Real

theorem tan_alpha_eq_one (α : ℝ) (h : (sin α + cos α) / (2 * sin α - cos α) = 2) : tan α = 1 := 
by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_one_l1367_136785


namespace NUMINAMATH_GPT_minimum_time_to_replace_shades_l1367_136774

theorem minimum_time_to_replace_shades :
  ∀ (C : ℕ) (S : ℕ) (T : ℕ) (E : ℕ),
  ((C = 60) ∧ (S = 4) ∧ (T = 5) ∧ (E = 48)) →
  ((C * S * T) / E = 25) :=
by
  intros C S T E h
  rcases h with ⟨hC, hS, hT, hE⟩
  sorry

end NUMINAMATH_GPT_minimum_time_to_replace_shades_l1367_136774


namespace NUMINAMATH_GPT_first_number_is_210_l1367_136787

theorem first_number_is_210 (A B hcf lcm : ℕ) (h1 : lcm = 2310) (h2: hcf = 47) (h3 : B = 517) :
  A * B = lcm * hcf → A = 210 :=
by
  sorry

end NUMINAMATH_GPT_first_number_is_210_l1367_136787


namespace NUMINAMATH_GPT_middle_part_division_l1367_136703

theorem middle_part_division 
  (x : ℝ) 
  (x_pos : x > 0) 
  (H : x + (1 / 4) * x + (1 / 8) * x = 96) :
  (1 / 4) * x = 17 + 21 / 44 :=
by
  sorry

end NUMINAMATH_GPT_middle_part_division_l1367_136703


namespace NUMINAMATH_GPT_net_income_difference_l1367_136777

-- Define Terry's and Jordan's daily income and working days
def terryDailyIncome : ℝ := 24
def terryWorkDays : ℝ := 7
def jordanDailyIncome : ℝ := 30
def jordanWorkDays : ℝ := 6

-- Define the tax rate
def taxRate : ℝ := 0.10

-- Calculate weekly gross incomes
def terryGrossWeeklyIncome : ℝ := terryDailyIncome * terryWorkDays
def jordanGrossWeeklyIncome : ℝ := jordanDailyIncome * jordanWorkDays

-- Calculate tax deductions
def terryTaxDeduction : ℝ := taxRate * terryGrossWeeklyIncome
def jordanTaxDeduction : ℝ := taxRate * jordanGrossWeeklyIncome

-- Calculate net weekly incomes
def terryNetWeeklyIncome : ℝ := terryGrossWeeklyIncome - terryTaxDeduction
def jordanNetWeeklyIncome : ℝ := jordanGrossWeeklyIncome - jordanTaxDeduction

-- Calculate the difference
def incomeDifference : ℝ := jordanNetWeeklyIncome - terryNetWeeklyIncome

-- The theorem to be proven
theorem net_income_difference :
  incomeDifference = 10.80 :=
by
  sorry

end NUMINAMATH_GPT_net_income_difference_l1367_136777


namespace NUMINAMATH_GPT_range_of_m_l1367_136713

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (m / (2*x - 1) + 3 = 0) ∧ (x > 0)) ↔ (m < 3 ∧ m ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1367_136713


namespace NUMINAMATH_GPT_lines_through_origin_l1367_136702

theorem lines_through_origin (n : ℕ) (h : 0 < n) :
    ∃ S : Finset (ℤ × ℤ), 
    (∀ xy : ℤ × ℤ, xy ∈ S ↔ (0 ≤ xy.1 ∧ xy.1 ≤ n ∧ 0 ≤ xy.2 ∧ xy.2 ≤ n ∧ Int.gcd xy.1 xy.2 = 1)) ∧
    S.card ≥ n^2 / 4 := 
sorry

end NUMINAMATH_GPT_lines_through_origin_l1367_136702


namespace NUMINAMATH_GPT_find_xy_l1367_136795

theorem find_xy (x y : ℝ) (π_ne_zero : Real.pi ≠ 0) (h1 : 4 * (x + 2) = 6 * x) (h2 : 6 * x = 2 * Real.pi * y) : x = 4 ∧ y = 12 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l1367_136795


namespace NUMINAMATH_GPT_total_marks_by_category_l1367_136758

theorem total_marks_by_category 
  (num_candidates_A : ℕ) (num_candidates_B : ℕ) (num_candidates_C : ℕ)
  (avg_marks_A : ℕ) (avg_marks_B : ℕ) (avg_marks_C : ℕ) 
  (hA : num_candidates_A = 30) (hB : num_candidates_B = 25) (hC : num_candidates_C = 25)
  (h_avg_A : avg_marks_A = 35) (h_avg_B : avg_marks_B = 42) (h_avg_C : avg_marks_C = 46) :
  (num_candidates_A * avg_marks_A = 1050) ∧
  (num_candidates_B * avg_marks_B = 1050) ∧
  (num_candidates_C * avg_marks_C = 1150) := 
by
  sorry

end NUMINAMATH_GPT_total_marks_by_category_l1367_136758


namespace NUMINAMATH_GPT_complex_number_quadrant_l1367_136740

open Complex

theorem complex_number_quadrant (z : ℂ) (h : (1 + 2 * Complex.I) / z = Complex.I) : 
  (0 < z.re) ∧ (0 < z.im) :=
by
  -- sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_complex_number_quadrant_l1367_136740


namespace NUMINAMATH_GPT_opposite_of_neg_2023_l1367_136757

theorem opposite_of_neg_2023 : -( -2023 ) = 2023 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2023_l1367_136757


namespace NUMINAMATH_GPT_triangle_side_a_l1367_136743

theorem triangle_side_a (a : ℝ) : 2 < a ∧ a < 8 → a = 7 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_a_l1367_136743


namespace NUMINAMATH_GPT_possible_values_y_l1367_136744

theorem possible_values_y (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y : ℝ, (y = 0 ∨ y = 41 ∨ y = 144) ∧ y = (x - 3)^2 * (x + 4) / (2 * x - 5) :=
by sorry

end NUMINAMATH_GPT_possible_values_y_l1367_136744


namespace NUMINAMATH_GPT_ticket_cost_difference_l1367_136708

theorem ticket_cost_difference
  (num_adults : ℕ) (num_children : ℕ)
  (cost_adult_ticket : ℕ) (cost_child_ticket : ℕ)
  (h1 : num_adults = 9)
  (h2 : num_children = 7)
  (h3 : cost_adult_ticket = 11)
  (h4 : cost_child_ticket = 7) :
  num_adults * cost_adult_ticket - num_children * cost_child_ticket = 50 := 
by
  sorry

end NUMINAMATH_GPT_ticket_cost_difference_l1367_136708


namespace NUMINAMATH_GPT_scalene_triangle_process_l1367_136712

theorem scalene_triangle_process (a b c : ℝ) 
  (h1: a > 0) (h2: b > 0) (h3: c > 0) 
  (h4: a + b > c) (h5: b + c > a) (h6: a + c > b) : 
  ¬(∃ k : ℝ, (k > 0) ∧ 
    ((k * a = a + b - c) ∧ 
     (k * b = b + c - a) ∧ 
     (k * c = a + c - b))) ∧ 
  (∀ n: ℕ, n > 0 → (a + b - c)^n + (b + c - a)^n + (a + c - b)^n < 1) :=
by
  sorry

end NUMINAMATH_GPT_scalene_triangle_process_l1367_136712


namespace NUMINAMATH_GPT_number_with_1_before_and_after_l1367_136797

theorem number_with_1_before_and_after (n : ℕ) (hn : n < 10) : 100 * 1 + 10 * n + 1 = 101 + 10 * n := by
    sorry

end NUMINAMATH_GPT_number_with_1_before_and_after_l1367_136797


namespace NUMINAMATH_GPT_percent_increase_second_half_century_l1367_136710

variable (P : ℝ) -- Initial population
variable (x : ℝ) -- Percentage increase in the second half of the century

noncomputable def population_first_half_century := 3 * P
noncomputable def population_end_century := P + 11 * P

theorem percent_increase_second_half_century :
  3 * P + (x / 100) * (3 * P) = 12 * P → x = 300 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_percent_increase_second_half_century_l1367_136710


namespace NUMINAMATH_GPT_solution_set_l1367_136704

noncomputable def truncated_interval (x : ℝ) (n : ℤ) : Prop :=
n ≤ x ∧ x < n + 1

theorem solution_set (x : ℝ) (hx : ∃ n : ℤ, n > 0 ∧ truncated_interval x n) :
  2 ≤ x ∧ x < 8 :=
sorry

end NUMINAMATH_GPT_solution_set_l1367_136704


namespace NUMINAMATH_GPT_find_coefficients_l1367_136718

-- Define the polynomial
def poly (a b : ℤ) (x : ℚ) : ℚ := a * x^4 + b * x^3 + 40 * x^2 - 20 * x + 8

-- Define the factor
def factor (x : ℚ) : ℚ := 3 * x^2 - 2 * x + 2

-- States that for a given polynomial and factor, the resulting (a, b) pair is (-51, 25)
theorem find_coefficients :
  ∃ a b c d : ℤ, 
  (∀ x, poly a b x = (factor x) * (c * x^2 + d * x + 4)) ∧ 
  a = -51 ∧ 
  b = 25 :=
by sorry

end NUMINAMATH_GPT_find_coefficients_l1367_136718


namespace NUMINAMATH_GPT_intersection_A_B_eq_B_l1367_136756

variable (a : ℝ) (A : Set ℝ) (B : Set ℝ)

def satisfies_quadratic (a : ℝ) (x : ℝ) : Prop := x^2 - a*x + 1 = 0

def set_A : Set ℝ := {1, 2, 3}

def set_B (a : ℝ) : Set ℝ := {x | satisfies_quadratic a x}

theorem intersection_A_B_eq_B (a : ℝ) (h : a ∈ set_A) : 
  (∀ x, x ∈ set_B a → x ∈ set_A) → (∃ x, x ∈ set_A ∧ satisfies_quadratic a x) →
  a = 2 :=
sorry

end NUMINAMATH_GPT_intersection_A_B_eq_B_l1367_136756


namespace NUMINAMATH_GPT_compute_fraction_power_l1367_136709

theorem compute_fraction_power :
  8 * (2 / 7)^4 = 128 / 2401 :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_power_l1367_136709


namespace NUMINAMATH_GPT_total_limes_picked_l1367_136749

def Fred_limes : ℕ := 36
def Alyssa_limes : ℕ := 32
def Nancy_limes : ℕ := 35
def David_limes : ℕ := 42
def Eileen_limes : ℕ := 50

theorem total_limes_picked :
  Fred_limes + Alyssa_limes + Nancy_limes + David_limes + Eileen_limes = 195 :=
by
  sorry

end NUMINAMATH_GPT_total_limes_picked_l1367_136749


namespace NUMINAMATH_GPT_book_arrangement_count_l1367_136723

-- Define the conditions
def total_books : ℕ := 6
def identical_books : ℕ := 3
def different_books : ℕ := total_books - identical_books

-- Prove the number of arrangements
theorem book_arrangement_count : (total_books.factorial / identical_books.factorial) = 120 := by
  sorry

end NUMINAMATH_GPT_book_arrangement_count_l1367_136723


namespace NUMINAMATH_GPT_part1_part2_l1367_136725

def z1 (a : ℝ) : Complex := Complex.mk 2 a
def z2 : Complex := Complex.mk 3 (-4)

-- Part 1: Prove that the product of z1 and z2 equals 10 - 5i when a = 1.
theorem part1 : z1 1 * z2 = Complex.mk 10 (-5) :=
by
  -- proof to be filled in
  sorry

-- Part 2: Prove that a = 4 when z1 + z2 is a real number.
theorem part2 (a : ℝ) (h : (z1 a + z2).im = 0) : a = 4 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_part1_part2_l1367_136725


namespace NUMINAMATH_GPT_ned_initial_lives_l1367_136729

variable (lost_lives : ℕ) (current_lives : ℕ) 
variable (initial_lives : ℕ)

theorem ned_initial_lives (h_lost: lost_lives = 13) (h_current: current_lives = 70) :
  initial_lives = current_lives + lost_lives := by
  sorry

end NUMINAMATH_GPT_ned_initial_lives_l1367_136729


namespace NUMINAMATH_GPT_actual_average_height_l1367_136727

theorem actual_average_height 
  (incorrect_avg_height : ℝ)
  (num_students : ℕ)
  (incorrect_height : ℝ)
  (correct_height : ℝ)
  (actual_avg_height : ℝ) :
  incorrect_avg_height = 175 →
  num_students = 20 →
  incorrect_height = 151 →
  correct_height = 111 →
  actual_avg_height = 173 :=
by
  sorry

end NUMINAMATH_GPT_actual_average_height_l1367_136727


namespace NUMINAMATH_GPT_x_squared_minus_y_squared_l1367_136778

theorem x_squared_minus_y_squared (x y : ℚ) (h₁ : x + y = 9 / 17) (h₂ : x - y = 1 / 51) : x^2 - y^2 = 1 / 289 :=
by
  sorry

end NUMINAMATH_GPT_x_squared_minus_y_squared_l1367_136778


namespace NUMINAMATH_GPT_min_value_a_plus_one_over_a_minus_one_l1367_136776

theorem min_value_a_plus_one_over_a_minus_one (a : ℝ) (h : a > 1) : 
  a + 1 / (a - 1) ≥ 3 ∧ (a = 2 → a + 1 / (a - 1) = 3) :=
by
  -- Translate the mathematical proof problem into a Lean 4 theorem statement.
  sorry

end NUMINAMATH_GPT_min_value_a_plus_one_over_a_minus_one_l1367_136776


namespace NUMINAMATH_GPT_expression_divisible_by_1897_l1367_136753

theorem expression_divisible_by_1897 (n : ℕ) :
  1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end NUMINAMATH_GPT_expression_divisible_by_1897_l1367_136753


namespace NUMINAMATH_GPT_eliza_ironing_hours_l1367_136772

theorem eliza_ironing_hours (h : ℕ) 
  (blouse_minutes : ℕ := 15) 
  (dress_minutes : ℕ := 20) 
  (hours_ironing_blouses : ℕ := h)
  (hours_ironing_dresses : ℕ := 3)
  (total_clothes : ℕ := 17) :
  ((60 / blouse_minutes) * hours_ironing_blouses) + ((60 / dress_minutes) * hours_ironing_dresses) = total_clothes →
  hours_ironing_blouses = 2 := 
sorry

end NUMINAMATH_GPT_eliza_ironing_hours_l1367_136772


namespace NUMINAMATH_GPT_polygon_has_twelve_sides_l1367_136734

theorem polygon_has_twelve_sides
  (sum_exterior_angles : ℝ)
  (sum_interior_angles : ℝ → ℝ)
  (n : ℝ)
  (h1 : sum_exterior_angles = 360)
  (h2 : ∀ n, sum_interior_angles n = 180 * (n - 2))
  (h3 : ∀ n, sum_interior_angles n = 5 * sum_exterior_angles) :
  n = 12 :=
by
  sorry

end NUMINAMATH_GPT_polygon_has_twelve_sides_l1367_136734


namespace NUMINAMATH_GPT_students_walk_home_fraction_l1367_136759

theorem students_walk_home_fraction :
  (1 - (3 / 8 + 2 / 5 + 1 / 8 + 5 / 100)) = (1 / 20) :=
by 
  -- The detailed proof is complex and would require converting these fractions to a common denominator,
  -- performing the arithmetic operations carefully and using Lean's rational number properties. Thus,
  -- the full detailed proof can be written with further steps, but here we insert 'sorry' to focus on the statement.
  sorry

end NUMINAMATH_GPT_students_walk_home_fraction_l1367_136759


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l1367_136788

theorem arithmetic_expression_evaluation : 
  -6 * 3 - (-8 * -2) + (-7 * -5) - 10 = -9 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l1367_136788


namespace NUMINAMATH_GPT_equidistant_point_quadrants_l1367_136706

theorem equidistant_point_quadrants :
  ∀ (x y : ℝ), 3 * x + 5 * y = 15 → (|x| = |y| → (x > 0 → y > 0 ∧ x = y ∧ y = x) ∧ (x < 0 → y > 0 ∧ x = -y ∧ -x = y)) := 
by
  sorry

end NUMINAMATH_GPT_equidistant_point_quadrants_l1367_136706


namespace NUMINAMATH_GPT_simple_interest_rate_l1367_136762

theorem simple_interest_rate (P A : ℝ) (T : ℕ) (R : ℝ) 
  (P_pos : P = 800) (A_pos : A = 950) (T_pos : T = 5) :
  R = 3.75 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1367_136762


namespace NUMINAMATH_GPT_power_division_identity_l1367_136766

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end NUMINAMATH_GPT_power_division_identity_l1367_136766


namespace NUMINAMATH_GPT_distance_from_tangency_to_tangent_l1367_136780

theorem distance_from_tangency_to_tangent 
  (R r : ℝ)
  (hR : R = 3)
  (hr : r = 1)
  (externally_tangent : true) :
  ∃ d : ℝ, (d = 0 ∨ d = 7/3) :=
by
  sorry

end NUMINAMATH_GPT_distance_from_tangency_to_tangent_l1367_136780


namespace NUMINAMATH_GPT_fraction_is_one_fifth_l1367_136746

theorem fraction_is_one_fifth (f : ℚ) (h1 : f * 50 - 4 = 6) : f = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_one_fifth_l1367_136746


namespace NUMINAMATH_GPT_michael_lap_time_l1367_136748

theorem michael_lap_time (T : ℝ) :
  (∀ (lap_time_donovan : ℝ), lap_time_donovan = 45 → (9 * T) / lap_time_donovan + 1 = 9 → T = 40) :=
by
  intro lap_time_donovan
  intro h1
  intro h2
  sorry

end NUMINAMATH_GPT_michael_lap_time_l1367_136748


namespace NUMINAMATH_GPT_water_leaving_rate_l1367_136735

-- Definitions: Volume of water and time taken
def volume_of_water : ℕ := 300
def time_taken : ℕ := 25

-- Theorem statement: Rate of water leaving the tank
theorem water_leaving_rate : (volume_of_water / time_taken) = 12 := 
by sorry

end NUMINAMATH_GPT_water_leaving_rate_l1367_136735


namespace NUMINAMATH_GPT_min_reciprocal_sum_l1367_136791

theorem min_reciprocal_sum (m n a b : ℝ) (h1 : m = 5) (h2 : n = 5) 
  (h3 : m * a + n * b = 1) (h4 : 0 < a) (h5 : 0 < b) : 
  (1 / a + 1 / b) = 20 :=
by 
  sorry

end NUMINAMATH_GPT_min_reciprocal_sum_l1367_136791


namespace NUMINAMATH_GPT_stamp_collection_cost_l1367_136730

def cost_brazil_per_stamp : ℝ := 0.08
def cost_peru_per_stamp : ℝ := 0.05
def num_brazil_stamps_60s : ℕ := 7
def num_peru_stamps_60s : ℕ := 4
def num_brazil_stamps_70s : ℕ := 12
def num_peru_stamps_70s : ℕ := 6

theorem stamp_collection_cost :
  num_brazil_stamps_60s * cost_brazil_per_stamp +
  num_peru_stamps_60s * cost_peru_per_stamp +
  num_brazil_stamps_70s * cost_brazil_per_stamp +
  num_peru_stamps_70s * cost_peru_per_stamp =
  2.02 :=
by
  -- Skipping proof steps.
  sorry

end NUMINAMATH_GPT_stamp_collection_cost_l1367_136730


namespace NUMINAMATH_GPT_gina_expenditure_l1367_136761

noncomputable def gina_total_cost : ℝ :=
  let regular_classes_cost := 12 * 450
  let lab_classes_cost := 6 * 550
  let textbooks_cost := 3 * 150
  let online_resources_cost := 4 * 95
  let facilities_fee := 200
  let lab_fee := 6 * 75
  let total_cost := regular_classes_cost + lab_classes_cost + textbooks_cost + online_resources_cost + facilities_fee + lab_fee
  let scholarship_amount := 0.5 * regular_classes_cost
  let discount_amount := 0.25 * lab_classes_cost
  let adjusted_cost := total_cost - scholarship_amount - discount_amount
  let interest := 0.04 * adjusted_cost
  adjusted_cost + interest

theorem gina_expenditure : gina_total_cost = 5881.20 :=
by
  sorry

end NUMINAMATH_GPT_gina_expenditure_l1367_136761


namespace NUMINAMATH_GPT_base7_to_base10_div_l1367_136731

theorem base7_to_base10_div (x y : ℕ) (h : 546 = x * 10^2 + y * 10 + 9) : (x + y + 9) / 21 = 6 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_base7_to_base10_div_l1367_136731


namespace NUMINAMATH_GPT_milford_age_in_3_years_l1367_136717

theorem milford_age_in_3_years (current_age_eustace : ℕ) (current_age_milford : ℕ) :
  (current_age_eustace = 2 * current_age_milford) → 
  (current_age_eustace + 3 = 39) → 
  current_age_milford + 3 = 21 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_milford_age_in_3_years_l1367_136717


namespace NUMINAMATH_GPT_each_cut_piece_weight_l1367_136745

theorem each_cut_piece_weight (L : ℕ) (W : ℕ) (c : ℕ) 
  (hL : L = 20) (hW : W = 150) (hc : c = 2) : (L / c) * W = 1500 := by
  sorry

end NUMINAMATH_GPT_each_cut_piece_weight_l1367_136745
