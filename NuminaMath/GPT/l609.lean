import Mathlib

namespace circle_problem_is_solved_l609_60959

def circle_problem_pqr : ℕ :=
  let n := 3 / 2;
  let p := 3;
  let q := 1;
  let r := 4;
  p + q + r

theorem circle_problem_is_solved : circle_problem_pqr = 8 :=
by {
  -- Additional context of conditions can be added here if necessary
  sorry
}

end circle_problem_is_solved_l609_60959


namespace ratio_rounded_to_nearest_tenth_l609_60965

theorem ratio_rounded_to_nearest_tenth : 
  (Float.round (11 / 16 : Float) * 10) / 10 = 0.7 :=
by
  -- sorry is used because the proof steps are not required in this task.
  sorry

end ratio_rounded_to_nearest_tenth_l609_60965


namespace hannah_mugs_problem_l609_60968

theorem hannah_mugs_problem :
  ∀ (total_mugs red_mugs yellow_mugs blue_mugs : ℕ),
    total_mugs = 40 →
    yellow_mugs = 12 →
    red_mugs * 2 = yellow_mugs →
    blue_mugs = 3 * red_mugs →
    total_mugs - (red_mugs + yellow_mugs + blue_mugs) = 4 :=
by
  intros total_mugs red_mugs yellow_mugs blue_mugs Htotal Hyellow Hred Hblue
  sorry

end hannah_mugs_problem_l609_60968


namespace max_friendly_groups_19_max_friendly_groups_20_l609_60936

def friendly_group {Team : Type} (beat : Team → Team → Prop) (A B C : Team) : Prop :=
  beat A B ∧ beat B C ∧ beat C A

def max_friendly_groups_19_teams : ℕ := 285
def max_friendly_groups_20_teams : ℕ := 330

theorem max_friendly_groups_19 {Team : Type} (n : ℕ) (h : n = 19) (beat : Team → Team → Prop) :
  ∃ (G : ℕ), G = max_friendly_groups_19_teams := sorry

theorem max_friendly_groups_20 {Team : Type} (n : ℕ) (h : n = 20) (beat : Team → Team → Prop) :
  ∃ (G : ℕ), G = max_friendly_groups_20_teams := sorry

end max_friendly_groups_19_max_friendly_groups_20_l609_60936


namespace proof_sum_of_ab_l609_60922

theorem proof_sum_of_ab :
  ∃ (a b : ℕ), a ≤ b ∧ 0 < a ∧ 0 < b ∧ a ^ 2 + b ^ 2 + 8 * a * b = 2010 ∧ a + b = 42 :=
sorry

end proof_sum_of_ab_l609_60922


namespace range_of_a_l609_60996

theorem range_of_a 
  (a b x1 x2 x3 x4 : ℝ)
  (h1 : a ≠ 0)
  (h2 : a^2 ≠ 0)
  (hx1 : a * x1^2 + b * x1 + 1 = 0) 
  (hx2 : a * x2^2 + b * x2 + 1 = 0) 
  (hx3 : a^2 * x3^2 + b * x3 + 1 = 0) 
  (hx4 : a^2 * x4^2 + b * x4 + 1 = 0)
  (h_order : x3 < x1 ∧ x1 < x2 ∧ x2 < x4) :
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l609_60996


namespace rakesh_salary_l609_60904

variable (S : ℝ) -- The salary S is a real number
variable (h : 0.595 * S = 2380) -- Condition derived from the problem

theorem rakesh_salary : S = 4000 :=
by
  sorry

end rakesh_salary_l609_60904


namespace teacher_age_l609_60951

theorem teacher_age (avg_student_age : ℕ) (num_students : ℕ) (new_avg_age : ℕ) (num_total : ℕ) (total_student_age : ℕ) (total_age_with_teacher : ℕ) :
  avg_student_age = 22 → 
  num_students = 23 → 
  new_avg_age = 23 → 
  num_total = 24 → 
  total_student_age = avg_student_age * num_students → 
  total_age_with_teacher = new_avg_age * num_total → 
  total_age_with_teacher - total_student_age = 46 :=
by
  intros
  sorry

end teacher_age_l609_60951


namespace no_empty_boxes_prob_l609_60937

def P (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem no_empty_boxes_prob :
  let num_balls := 3
  let num_boxes := 3
  let total_outcomes := num_boxes ^ num_balls
  let favorable_outcomes := P num_balls num_boxes
  let probability := favorable_outcomes / total_outcomes
  probability = 2 / 9 :=
by
  sorry

end no_empty_boxes_prob_l609_60937


namespace frac_eq_l609_60932

theorem frac_eq (x : ℝ) (h : 3 - 9 / x + 6 / x^2 = 0) : 2 / x = 1 ∨ 2 / x = 2 := 
by 
  sorry

end frac_eq_l609_60932


namespace range_of_a_l609_60949

def p (x : ℝ) : Prop := abs (2 * x - 1) ≤ 3

def q (x a : ℝ) : Prop := x^2 - (2*a + 1) * x + a*(a + 1) ≤ 0

theorem range_of_a : 
  (∀ x a, (¬ q x a) → (¬ p x))
  ∧ (∃ x a, (¬ q x a) ∧ (¬ p x))
  → (-1 : ℝ) ≤ a ∧ a ≤ (1 : ℝ) :=
sorry

end range_of_a_l609_60949


namespace simplify_and_evaluate_l609_60946

def my_expression (x : ℝ) := (x + 2) * (x - 2) + 3 * (1 - x)

theorem simplify_and_evaluate : 
  my_expression (Real.sqrt 2) = 1 - 3 * Real.sqrt 2 := by
    sorry

end simplify_and_evaluate_l609_60946


namespace max_value_of_z_l609_60990

open Real

theorem max_value_of_z (x y : ℝ) (h₁ : x + y ≥ 1) (h₂ : 2 * x - y ≤ 0) (h₃ : 3 * x - 2 * y + 2 ≥ 0) : 
  ∃ x y, 3 * x - y = 2 :=
sorry

end max_value_of_z_l609_60990


namespace james_marbles_left_l609_60954

def marbles_remain (total_marbles : ℕ) (bags : ℕ) (given_away : ℕ) : ℕ :=
  (total_marbles / bags) * (bags - given_away)

theorem james_marbles_left :
  marbles_remain 28 4 1 = 21 := 
by
  sorry

end james_marbles_left_l609_60954


namespace sum_of_digits_l609_60963

theorem sum_of_digits (A T M : ℕ) (h1 : T = A + 3) (h2 : M = 3)
    (h3 : (∃ k : ℕ, T = k^2 * M) ∧ (∃ l : ℕ, T = 33)) : 
    ∃ x : ℕ, ∃ dsum : ℕ, (A + x) % (M + x) = 0 ∧ dsum = 12 :=
by
  sorry

end sum_of_digits_l609_60963


namespace largest_five_digit_congruent_to_31_modulo_26_l609_60976

theorem largest_five_digit_congruent_to_31_modulo_26 :
  ∃ x : ℕ, (10000 ≤ x ∧ x < 100000) ∧ x % 26 = 31 ∧ x = 99975 :=
by
  sorry

end largest_five_digit_congruent_to_31_modulo_26_l609_60976


namespace distinct_real_numbers_eq_l609_60974

theorem distinct_real_numbers_eq (x : ℝ) :
  (x^2 - 7)^2 + 2 * x^2 = 33 → 
  (∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                    {a, b, c, d} = {x | (x^2 - 7)^2 + 2 * x^2 = 33}) :=
sorry

end distinct_real_numbers_eq_l609_60974


namespace smallest_tournament_with_ordered_group_l609_60915

-- Define the concept of a tennis tournament with n players
def tennis_tournament (n : ℕ) := 
  ∀ (i j : ℕ), (i < n) → (j < n) → (i ≠ j) → (i < j) ∨ (j < i)

-- Define what it means for a group of four players to be "ordered"
def ordered_group (p1 p2 p3 p4 : ℕ) : Prop := 
  ∃ (winner : ℕ), ∃ (loser : ℕ), 
    (winner ≠ loser) ∧ (winner = p1 ∨ winner = p2 ∨ winner = p3 ∨ winner = p4) ∧ 
    (loser = p1 ∨ loser = p2 ∨ loser = p3 ∨ loser = p4)

-- Prove that any tennis tournament with 8 players has an ordered group
theorem smallest_tournament_with_ordered_group : 
  ∀ (n : ℕ), ∀ (tournament : tennis_tournament n), 
    (n ≥ 8) → 
    (∃ (p1 p2 p3 p4 : ℕ), ordered_group p1 p2 p3 p4) :=
  by
  -- proof omitted
  sorry

end smallest_tournament_with_ordered_group_l609_60915


namespace find_square_side_length_l609_60956

noncomputable def side_length_PQRS (x : ℝ) : Prop :=
  let PT := 1
  let QU := 2
  let RV := 3
  let SW := 4
  let PQRS_area := x^2
  let TUVW_area := 1 / 2 * x^2
  let triangle_area (base height : ℝ) : ℝ := 1 / 2 * base * height
  PQRS_area = x^2 ∧ TUVW_area = 1 / 2 * x^2 ∧
  triangle_area 1 (x - 4) + (x - 1) + 
  triangle_area 3 (x - 2) + 2 * (x - 3) = 1 / 2 * x^2

theorem find_square_side_length : ∃ x : ℝ, side_length_PQRS x ∧ x = 6 := 
  sorry

end find_square_side_length_l609_60956


namespace income_before_taxes_l609_60925

/-- Define given conditions -/
def net_income (x : ℝ) : ℝ := x - 0.10 * (x - 3000)

/-- Prove that the income before taxes must have been 13000 given the conditions. -/
theorem income_before_taxes (x : ℝ) (hx : net_income x = 12000) : x = 13000 :=
by sorry

end income_before_taxes_l609_60925


namespace minimum_value_of_expression_l609_60926

theorem minimum_value_of_expression (a b : ℝ) (h : 1 / a + 2 / b = 1) : 4 * a^2 + b^2 ≥ 32 :=
by sorry

end minimum_value_of_expression_l609_60926


namespace remainder_of_polynomial_l609_60960

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

-- Define the main theorem stating the remainder when f(x) is divided by (x - 1) is 6
theorem remainder_of_polynomial : f 1 = 6 := 
by 
  sorry

end remainder_of_polynomial_l609_60960


namespace min_correct_answers_l609_60930

/-- 
Given:
1. There are 25 questions in the preliminary round.
2. Scoring rules: 
   - 4 points for each correct answer,
   - -1 point for each incorrect or unanswered question.
3. A score of at least 60 points is required to advance to the next round.

Prove that the minimum number of correct answers needed to advance is 17.
-/
theorem min_correct_answers (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 25) (h3 : 4 * x - (25 - x) ≥ 60) : x ≥ 17 :=
sorry

end min_correct_answers_l609_60930


namespace solve_integer_pairs_l609_60927

-- Definition of the predicate that (m, n) satisfies the given equation
def satisfies_equation (m n : ℤ) : Prop :=
  m * n^2 = 2009 * (n + 1)

-- Theorem stating that the only solutions are (4018, 1) and (0, -1)
theorem solve_integer_pairs :
  ∀ (m n : ℤ), satisfies_equation m n ↔ (m = 4018 ∧ n = 1) ∨ (m = 0 ∧ n = -1) :=
by
  sorry

end solve_integer_pairs_l609_60927


namespace lattice_points_in_bounded_region_l609_60945

def isLatticePoint (p : ℤ × ℤ) : Prop :=
  true  -- All (n, m) ∈ ℤ × ℤ are lattice points

def boundedRegion (x y : ℤ) : Prop :=
  y = x ^ 2 ∨ y = 8 - x ^ 2
  
theorem lattice_points_in_bounded_region :
  ∃ S : Finset (ℤ × ℤ), 
    (∀ p ∈ S, isLatticePoint p ∧ boundedRegion p.1 p.2) ∧ S.card = 17 :=
by
  sorry

end lattice_points_in_bounded_region_l609_60945


namespace volume_remaining_cube_l609_60983

theorem volume_remaining_cube (a : ℝ) (original_volume vertex_cube_volume : ℝ) (number_of_vertices : ℕ) :
  original_volume = a^3 → 
  vertex_cube_volume = 1 → 
  number_of_vertices = 8 → 
  a = 3 →
  original_volume - (number_of_vertices * vertex_cube_volume) = 19 := 
by
  sorry

end volume_remaining_cube_l609_60983


namespace ryan_bread_slices_l609_60969

theorem ryan_bread_slices 
  (num_pb_people : ℕ)
  (pb_sandwiches_per_person : ℕ)
  (num_tuna_people : ℕ)
  (tuna_sandwiches_per_person : ℕ)
  (num_turkey_people : ℕ)
  (turkey_sandwiches_per_person : ℕ)
  (slices_per_pb_sandwich : ℕ)
  (slices_per_tuna_sandwich : ℕ)
  (slices_per_turkey_sandwich : ℝ)
  (h1 : num_pb_people = 4)
  (h2 : pb_sandwiches_per_person = 2)
  (h3 : num_tuna_people = 3)
  (h4 : tuna_sandwiches_per_person = 3)
  (h5 : num_turkey_people = 2)
  (h6 : turkey_sandwiches_per_person = 1)
  (h7 : slices_per_pb_sandwich = 2)
  (h8 : slices_per_tuna_sandwich = 3)
  (h9 : slices_per_turkey_sandwich = 1.5) : 
  (num_pb_people * pb_sandwiches_per_person * slices_per_pb_sandwich 
  + num_tuna_people * tuna_sandwiches_per_person * slices_per_tuna_sandwich 
  + (num_turkey_people * turkey_sandwiches_per_person : ℝ) * slices_per_turkey_sandwich) = 46 :=
by
  sorry

end ryan_bread_slices_l609_60969


namespace sozopolian_ineq_find_p_l609_60952

noncomputable def is_sozopolian (p a b c : ℕ) : Prop :=
  p % 2 = 1 ∧
  Nat.Prime p ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a * b + 1) % p = 0 ∧
  (b * c + 1) % p = 0 ∧
  (c * a + 1) % p = 0

theorem sozopolian_ineq (p a b c : ℕ) (hp : is_sozopolian p a b c) :
  p + 2 ≤ (a + b + c) / 3 :=
sorry

theorem find_p (p : ℕ) :
  (∃ a b c : ℕ, is_sozopolian p a b c ∧ (a + b + c) / 3 = p + 2) ↔ p = 5 :=
sorry

end sozopolian_ineq_find_p_l609_60952


namespace mean_of_remaining_two_numbers_l609_60914

theorem mean_of_remaining_two_numbers :
  let n1 := 1871
  let n2 := 1997
  let n3 := 2023
  let n4 := 2029
  let n5 := 2113
  let n6 := 2125
  let n7 := 2137
  let total_sum := n1 + n2 + n3 + n4 + n5 + n6 + n7
  let known_mean := 2100
  let mean_of_other_two := 1397.5
  total_sum = 13295 →
  5 * known_mean = 10500 →
  total_sum - 10500 = 2795 →
  2795 / 2 = mean_of_other_two :=
by
  intros
  sorry

end mean_of_remaining_two_numbers_l609_60914


namespace find_E_l609_60961

variables (E F G H : ℕ)

noncomputable def conditions := 
  (E * F = 120) ∧ 
  (G * H = 120) ∧ 
  (E - F = G + H - 2) ∧ 
  (E ≠ F) ∧
  (E ≠ G) ∧ 
  (E ≠ H) ∧
  (F ≠ G) ∧
  (F ≠ H) ∧
  (G ≠ H)

theorem find_E (E F G H : ℕ) (h : conditions E F G H) : E = 30 :=
sorry

end find_E_l609_60961


namespace solve_for_r_l609_60957

variable (k r : ℝ)

theorem solve_for_r (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := sorry

end solve_for_r_l609_60957


namespace no_valid_arrangement_in_7x7_grid_l609_60989

theorem no_valid_arrangement_in_7x7_grid :
  ¬ (∃ (f : Fin 7 → Fin 7 → ℕ),
    (∀ (i j : Fin 6),
      (f i j + f i (j + 1) + f (i + 1) j + f (i + 1) (j + 1)) % 2 = 1) ∧
    (∀ (i j : Fin 5),
      (f i j + f i (j + 1) + f i (j + 2) + f (i + 1) j + f (i + 1) (j + 1) + f (i + 1) (j + 2) +
       f (i + 2) j + f (i + 2) (j + 1) + f (i + 2) (j + 2)) % 2 = 1)) := by
  sorry

end no_valid_arrangement_in_7x7_grid_l609_60989


namespace inequality_solution_set_l609_60975

theorem inequality_solution_set (x : ℝ) : (x - 1 < 7) ∧ (3 * x + 1 ≥ -2) ↔ -1 ≤ x ∧ x < 8 :=
by
  sorry

end inequality_solution_set_l609_60975


namespace max_sum_of_radii_in_prism_l609_60931

noncomputable def sum_of_radii (AB AD AA1 : ℝ) : ℝ :=
  let r (t : ℝ) := 2 - 2 * t
  let R (t : ℝ) := 3 * t / (1 + t)
  let f (t : ℝ) := R t + r t
  let t_max := 1 / 2
  f t_max

theorem max_sum_of_radii_in_prism :
  let AB := 5
  let AD := 3
  let AA1 := 4
  sum_of_radii AB AD AA1 = 21 / 10 := by
sorry

end max_sum_of_radii_in_prism_l609_60931


namespace fruits_in_good_condition_l609_60958

def percentage_good_fruits (num_oranges num_bananas pct_rotten_oranges pct_rotten_bananas : ℕ) : ℚ :=
  let total_fruits := num_oranges + num_bananas
  let rotten_oranges := (pct_rotten_oranges * num_oranges) / 100
  let rotten_bananas := (pct_rotten_bananas * num_bananas) / 100
  let good_fruits := total_fruits - (rotten_oranges + rotten_bananas)
  (good_fruits * 100) / total_fruits

theorem fruits_in_good_condition :
  percentage_good_fruits 600 400 15 8 = 87.8 := sorry

end fruits_in_good_condition_l609_60958


namespace azalea_paid_shearer_l609_60928

noncomputable def amount_paid_to_shearer (number_of_sheep wool_per_sheep price_per_pound profit : ℕ) : ℕ :=
  let total_wool := number_of_sheep * wool_per_sheep
  let total_revenue := total_wool * price_per_pound
  total_revenue - profit

theorem azalea_paid_shearer :
  let number_of_sheep := 200
  let wool_per_sheep := 10
  let price_per_pound := 20
  let profit := 38000
  amount_paid_to_shearer number_of_sheep wool_per_sheep price_per_pound profit = 2000 := 
by
  sorry

end azalea_paid_shearer_l609_60928


namespace ladder_base_distance_l609_60940

theorem ladder_base_distance
  (c : ℕ) (b : ℕ) (hypotenuse : c = 13) (wall_height : b = 12) :
  ∃ x : ℕ, x^2 + b^2 = c^2 ∧ x = 5 := by
  sorry

end ladder_base_distance_l609_60940


namespace parabola_tangents_intersection_y_coord_l609_60948

theorem parabola_tangents_intersection_y_coord
  (a b : ℝ)
  (ha : A = (a, a^2 + 1))
  (hb : B = (b, b^2 + 1))
  (tangent_perpendicular : ∀ t1 t2 : ℝ, t1 * t2 = -1):
  ∃ y : ℝ, y = 3 / 4 :=
by
  sorry

end parabola_tangents_intersection_y_coord_l609_60948


namespace solve_for_x_l609_60985

noncomputable def x_solution (x : ℚ) : Prop :=
  x > 1 ∧ 3 * x^2 + 11 * x - 20 = 0

theorem solve_for_x :
  ∃ x : ℚ, x_solution x ∧ x = 4 / 3 :=
by
  sorry

end solve_for_x_l609_60985


namespace cubic_yards_to_cubic_feet_l609_60992

theorem cubic_yards_to_cubic_feet :
  (1 : ℝ) * 3^3 * 5 = 135 := by
sorry

end cubic_yards_to_cubic_feet_l609_60992


namespace closed_path_even_length_l609_60919

def is_closed_path (steps : List Char) : Bool :=
  let net_vertical := steps.count 'U' - steps.count 'D'
  let net_horizontal := steps.count 'R' - steps.count 'L'
  net_vertical = 0 ∧ net_horizontal = 0

def move_length (steps : List Char) : Nat :=
  steps.length

theorem closed_path_even_length (steps : List Char) :
  is_closed_path steps = true → move_length steps % 2 = 0 :=
by
  -- Conditions extracted as definitions
  intros h
  -- The proof will handle showing that the length of the closed path is even
  sorry

end closed_path_even_length_l609_60919


namespace repeating_decimal_fraction_l609_60950

theorem repeating_decimal_fraction : (0.363636363636 : ℚ) = 4 / 11 := 
sorry

end repeating_decimal_fraction_l609_60950


namespace speed_conversion_l609_60970

-- Define the given condition
def kmph_to_mps (v : ℕ) : ℕ := v * 5 / 18

-- Speed in kmph
def speed_kmph : ℕ := 216

-- The proof statement
theorem speed_conversion : kmph_to_mps speed_kmph = 60 :=
by
  sorry

end speed_conversion_l609_60970


namespace circumference_of_circle_x_l609_60987

theorem circumference_of_circle_x (A_x A_y : ℝ) (r_x r_y C_x : ℝ)
  (h_area: A_x = A_y) (h_half_radius_y: r_y = 2 * 5)
  (h_area_y: A_y = Real.pi * r_y^2)
  (h_area_x: A_x = Real.pi * r_x^2)
  (h_circumference_x: C_x = 2 * Real.pi * r_x) :
  C_x = 20 * Real.pi :=
by
  sorry

end circumference_of_circle_x_l609_60987


namespace moles_of_C2H6_are_1_l609_60924

def moles_of_C2H6_reacted (n_C2H6: ℕ) (n_Cl2: ℕ) (n_C2Cl6: ℕ): Prop :=
  n_Cl2 = 6 ∧ n_C2Cl6 = 1 ∧ (n_C2H6 + 6 * (n_Cl2 - 1) = n_C2Cl6 + 6 * (n_Cl2 - 1))

theorem moles_of_C2H6_are_1:
  ∀ (n_C2H6 n_Cl2 n_C2Cl6: ℕ), moles_of_C2H6_reacted n_C2H6 n_Cl2 n_C2Cl6 → n_C2H6 = 1 :=
by
  intros n_C2H6 n_Cl2 n_C2Cl6 h
  sorry

end moles_of_C2H6_are_1_l609_60924


namespace arcsin_one_half_l609_60934

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l609_60934


namespace points_per_win_is_5_l609_60905

-- Definitions based on conditions
def rounds_played : ℕ := 30
def vlad_points : ℕ := 64
def taro_points (T : ℕ) : ℕ := (3 * T) / 5 - 4
def total_points (T : ℕ) : ℕ := taro_points T + vlad_points

-- Theorem statement to prove the number of points per win
theorem points_per_win_is_5 (T : ℕ) (H : total_points T = T) : T / rounds_played = 5 := sorry

end points_per_win_is_5_l609_60905


namespace janet_dresses_total_pockets_l609_60906

theorem janet_dresses_total_pockets :
  let dresses := 24
  let with_pockets := dresses / 2
  let with_two_pockets := with_pockets / 3
  let with_three_pockets := with_pockets - with_two_pockets
  let total_two_pockets := with_two_pockets * 2
  let total_three_pockets := with_three_pockets * 3
  total_two_pockets + total_three_pockets = 32 := by
  sorry

end janet_dresses_total_pockets_l609_60906


namespace find_f_2023_l609_60903

def is_odd_function (g : ℝ → ℝ) := ∀ x, g x = -g (-x)

def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (f (x + 1)) - 2 = -((f (1 - x)) - 2)

def condition2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 - x) = f (3 + x)

theorem find_f_2023 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, (f (x + 1)) - 2 = -((f (1 - x)) - 2)) 
  (h2 : ∀ x : ℝ, f (1 - x) = f (3 + x)) : 
  f 2023 = 2 :=
sorry

end find_f_2023_l609_60903


namespace system_of_equations_solution_l609_60967

theorem system_of_equations_solution :
  ∃ x y : ℚ, x = 2 * y ∧ 2 * x - y = 5 ∧ x = 10 / 3 ∧ y = 5 / 3 :=
by
  sorry

end system_of_equations_solution_l609_60967


namespace james_total_time_l609_60912

def time_to_play_main_game : ℕ := 
  let download_time := 10
  let install_time := download_time / 2
  let update_time := download_time * 2
  let account_time := 5
  let internet_issues_time := 15
  let before_tutorial_time := download_time + install_time + update_time + account_time + internet_issues_time
  let tutorial_time := before_tutorial_time * 3
  before_tutorial_time + tutorial_time

theorem james_total_time : time_to_play_main_game = 220 := by
  sorry

end james_total_time_l609_60912


namespace polynomial_quotient_correct_l609_60962

noncomputable def polynomial_division_quotient : Polynomial ℝ :=
  (Polynomial.C 1 * Polynomial.X^6 + Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 8) / (Polynomial.X - Polynomial.C 1)

-- Math proof statement
theorem polynomial_quotient_correct :
  polynomial_division_quotient = Polynomial.C 1 * Polynomial.X^5 + Polynomial.C 1 * Polynomial.X^4 
                                 + Polynomial.C 1 * Polynomial.X^3 + Polynomial.C 1 * Polynomial.X^2 
                                 + Polynomial.C 3 * Polynomial.X + Polynomial.C 3 :=
by
  sorry

end polynomial_quotient_correct_l609_60962


namespace regular_hexagon_has_greatest_lines_of_symmetry_l609_60972

-- Definitions for the various shapes and their lines of symmetry.
def regular_pentagon_lines_of_symmetry : ℕ := 5
def parallelogram_lines_of_symmetry : ℕ := 0
def oval_ellipse_lines_of_symmetry : ℕ := 2
def right_triangle_lines_of_symmetry : ℕ := 0
def regular_hexagon_lines_of_symmetry : ℕ := 6

-- Theorem stating that the regular hexagon has the greatest number of lines of symmetry.
theorem regular_hexagon_has_greatest_lines_of_symmetry :
  regular_hexagon_lines_of_symmetry > regular_pentagon_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > parallelogram_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > oval_ellipse_lines_of_symmetry ∧
  regular_hexagon_lines_of_symmetry > right_triangle_lines_of_symmetry :=
by
  sorry

end regular_hexagon_has_greatest_lines_of_symmetry_l609_60972


namespace trisha_initial_money_l609_60979

-- Definitions based on conditions
def spent_on_meat : ℕ := 17
def spent_on_chicken : ℕ := 22
def spent_on_veggies : ℕ := 43
def spent_on_eggs : ℕ := 5
def spent_on_dog_food : ℕ := 45
def spent_on_cat_food : ℕ := 18
def money_left : ℕ := 35

-- Total amount spent
def total_spent : ℕ :=
  spent_on_meat + spent_on_chicken + spent_on_veggies + spent_on_eggs + spent_on_dog_food + spent_on_cat_food

-- The target amount she brought with her at the beginning
def total_money_brought : ℕ :=
  total_spent + money_left

-- The theorem to be proved
theorem trisha_initial_money :
  total_money_brought = 185 :=
by
  sorry

end trisha_initial_money_l609_60979


namespace min_value_of_quadratic_l609_60902

theorem min_value_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, 3 * x^2 - 18 * x + 2000 ≤ 3 * y^2 - 18 * y + 2000) ∧ (3 * x^2 - 18 * x + 2000 = 1973) :=
by
  sorry

end min_value_of_quadratic_l609_60902


namespace boat_distance_downstream_l609_60907

-- Definitions
def boat_speed_in_still_water : ℝ := 24
def stream_speed : ℝ := 4
def time_downstream : ℝ := 3

-- Effective speed downstream
def speed_downstream := boat_speed_in_still_water + stream_speed

-- Distance calculation
def distance_downstream := speed_downstream * time_downstream

-- Proof statement
theorem boat_distance_downstream : distance_downstream = 84 := 
by
  -- This is where the proof would go, but we use sorry for now
  sorry

end boat_distance_downstream_l609_60907


namespace yaya_bike_walk_l609_60984

theorem yaya_bike_walk (x y : ℝ) : 
  (x + y = 1.5 ∧ 15 * x + 5 * y = 20) ↔ (x + y = 1.5 ∧ 15 * x + 5 * y = 20) :=
by 
  sorry

end yaya_bike_walk_l609_60984


namespace part_a_l609_60941

theorem part_a (x α : ℝ) (hα : 0 < α ∧ α < 1) (hx : x ≥ 0) : x^α - α * x ≤ 1 - α :=
sorry

end part_a_l609_60941


namespace gcd_9157_2695_eq_1_l609_60993

theorem gcd_9157_2695_eq_1 : Int.gcd 9157 2695 = 1 := 
by
  sorry

end gcd_9157_2695_eq_1_l609_60993


namespace prime_factors_power_l609_60997

-- Given conditions
def a_b_c_factors (a b c : ℕ) : Prop :=
  (∀ x, x = a ∨ x = b ∨ x = c → Prime x) ∧
  a < b ∧ b < c ∧ a * b * c ∣ 1998

-- Proof problem
theorem prime_factors_power (a b c : ℕ) (h : a_b_c_factors a b c) : (b + c) ^ a = 1600 := 
sorry

end prime_factors_power_l609_60997


namespace find_function_f_l609_60908

-- Define the problem in Lean 4
theorem find_function_f (f : ℝ → ℝ) : 
  (f 0 = 1) → 
  ((∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2)) → 
  (∀ x : ℝ, f x = x + 1) :=
  by
    intros h₁ h₂
    sorry

end find_function_f_l609_60908


namespace triangle_area_l609_60982

theorem triangle_area (base height : ℕ) (h_base : base = 10) (h_height : height = 5) :
  (base * height) / 2 = 25 := by
  -- Proof is not required as per instructions.
  sorry

end triangle_area_l609_60982


namespace josephs_total_cards_l609_60980

def number_of_decks : ℕ := 4
def cards_per_deck : ℕ := 52
def total_cards : ℕ := number_of_decks * cards_per_deck

theorem josephs_total_cards : total_cards = 208 := by
  sorry

end josephs_total_cards_l609_60980


namespace camila_weeks_to_goal_l609_60939

open Nat

noncomputable def camila_hikes : ℕ := 7
noncomputable def amanda_hikes : ℕ := 8 * camila_hikes
noncomputable def steven_hikes : ℕ := amanda_hikes + 15
noncomputable def additional_hikes_needed : ℕ := steven_hikes - camila_hikes
noncomputable def hikes_per_week : ℕ := 4
noncomputable def weeks_to_goal : ℕ := additional_hikes_needed / hikes_per_week

theorem camila_weeks_to_goal : weeks_to_goal = 16 :=
  by sorry

end camila_weeks_to_goal_l609_60939


namespace molecular_weight_CO_l609_60947

theorem molecular_weight_CO :
  let atomic_weight_C := 12.01
  let atomic_weight_O := 16.00
  let molecular_weight := atomic_weight_C + atomic_weight_O
  molecular_weight = 28.01 := 
by
  sorry

end molecular_weight_CO_l609_60947


namespace least_x_for_inequality_l609_60900

theorem least_x_for_inequality : 
  ∃ (x : ℝ), (-x^2 + 9 * x - 20 ≤ 0) ∧ ∀ y, (-y^2 + 9 * y - 20 ≤ 0) → x ≤ y ∧ x = 4 := 
by
  sorry

end least_x_for_inequality_l609_60900


namespace problem_statement_l609_60991

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (Real.sin x)^2 - Real.tan x else Real.exp (-2 * x)

theorem problem_statement : f (f (-25 * Real.pi / 4)) = Real.exp (-3) :=
by
  sorry

end problem_statement_l609_60991


namespace minimum_value_of_quadratic_l609_60921

theorem minimum_value_of_quadratic : ∀ x : ℝ, (∃ y : ℝ, y = (x-2)^2 - 3) → ∃ m : ℝ, (∀ x : ℝ, (x-2)^2 - 3 ≥ m) ∧ m = -3 :=
by
  sorry

end minimum_value_of_quadratic_l609_60921


namespace min_value_ineq_l609_60988

noncomputable def min_value (x y z : ℝ) := (1/x) + (1/y) + (1/z)

theorem min_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z ≥ 4.5 :=
sorry

end min_value_ineq_l609_60988


namespace quadratic_two_equal_real_roots_c_l609_60994

theorem quadratic_two_equal_real_roots_c (c : ℝ) : 
  (∃ x : ℝ, (2*x^2 - x + c = 0) ∧ (∃ y : ℝ, y ≠ x ∧ 2*y^2 - y + c = 0)) →
  c = 1/8 :=
sorry

end quadratic_two_equal_real_roots_c_l609_60994


namespace infinite_squares_in_ap_l609_60943

theorem infinite_squares_in_ap
    (a d : ℤ)
    (h : ∃ n : ℤ, a^2 = a + n * d) :
    ∀ N : ℕ, ∃ m : ℤ, ∃ k : ℕ, k > N ∧ m^2 = a + k * d :=
by
  sorry

end infinite_squares_in_ap_l609_60943


namespace hundreds_digit_of_8_pow_2048_l609_60955

theorem hundreds_digit_of_8_pow_2048 : 
  (8^2048 % 1000) / 100 = 0 := 
by
  sorry

end hundreds_digit_of_8_pow_2048_l609_60955


namespace average_weight_l609_60923

variable (A B C : ℝ) 

theorem average_weight (h1 : (A + B) / 2 = 48) (h2 : (B + C) / 2 = 42) (h3 : B = 51) :
  (A + B + C) / 3 = 43 := by
  sorry

end average_weight_l609_60923


namespace cost_difference_l609_60935

def TMobile_cost (num_lines : ℕ) : ℝ :=
  let base_cost := 50
  let additional_line_cost := 16
  let discount := 0.1
  let data_charge := 3
  let monthly_cost_before_discount := base_cost + (additional_line_cost * (num_lines - 2))
  let total_monthly_cost := monthly_cost_before_discount + (data_charge * num_lines)
  (total_monthly_cost * (1 - discount)) * 12

def MMobile_cost (num_lines : ℕ) : ℝ :=
  let base_cost := 45
  let additional_line_cost := 14
  let activation_fee := 20
  let monthly_cost := base_cost + (additional_line_cost * (num_lines - 2))
  (monthly_cost * 12) + (activation_fee * num_lines)

theorem cost_difference (num_lines : ℕ) (h : num_lines = 5) :
  TMobile_cost num_lines - MMobile_cost num_lines = 76.40 :=
  sorry

end cost_difference_l609_60935


namespace red_black_ball_ratio_l609_60933

theorem red_black_ball_ratio (R B x : ℕ) (h1 : 3 * R = B + x) (h2 : 2 * R + x = B) :
  R / B = 2 / 5 := by
  sorry

end red_black_ball_ratio_l609_60933


namespace intersection_of_sets_l609_60911

-- Conditions as Lean definitions
def A : Set Int := {-2, -1}
def B : Set Int := {-1, 2, 3}

-- Stating the proof problem in Lean 4
theorem intersection_of_sets : A ∩ B = {-1} :=
by
  sorry

end intersection_of_sets_l609_60911


namespace find_parabola_l609_60909

variable (P : ℝ × ℝ)
variable (a b : ℝ)

def parabola1 (P : ℝ × ℝ) (a : ℝ) := P.2^2 = 4 * a * P.1
def parabola2 (P : ℝ × ℝ) (b : ℝ) := P.1^2 = 4 * b * P.2

theorem find_parabola (hP : P = (-2, 4)) :
  (∃ a, parabola1 P a ∧ P.2^2 = -8 * P.1) ∨ 
  (∃ b, parabola2 P b ∧ P.1^2 = P.2) := by
  sorry

end find_parabola_l609_60909


namespace exponential_function_f1_l609_60977

theorem exponential_function_f1 (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) (h3 : a^3 = 8) : a^1 = 2 := by
  sorry

end exponential_function_f1_l609_60977


namespace no_rational_points_on_sqrt3_circle_l609_60966

theorem no_rational_points_on_sqrt3_circle (x y : ℚ) : x^2 + y^2 ≠ 3 :=
sorry

end no_rational_points_on_sqrt3_circle_l609_60966


namespace james_paid_with_l609_60973

variable (candy_packs : ℕ) (cost_per_pack : ℕ) (change_received : ℕ)

theorem james_paid_with (h1 : candy_packs = 3) (h2 : cost_per_pack = 3) (h3 : change_received = 11) :
  let total_cost := candy_packs * cost_per_pack
  let amount_paid := total_cost + change_received
  amount_paid = 20 :=
by
  sorry

end james_paid_with_l609_60973


namespace no_simultaneous_negative_values_l609_60942

theorem no_simultaneous_negative_values (m n : ℝ) :
  ¬ ((3*m^2 + 4*m*n - 2*n^2 < 0) ∧ (-m^2 - 4*m*n + 3*n^2 < 0)) :=
by
  sorry

end no_simultaneous_negative_values_l609_60942


namespace num_type_A_cubes_internal_diagonal_l609_60999

theorem num_type_A_cubes_internal_diagonal :
  let L := 120
  let W := 350
  let H := 400
  -- Total cubes traversed calculation
  let GCD := Nat.gcd
  let total_cubes_traversed := L + W + H - (GCD L W + GCD W H + GCD H L) + GCD L (GCD W H)
  -- Type A cubes calculation
  total_cubes_traversed / 2 = 390 := by sorry

end num_type_A_cubes_internal_diagonal_l609_60999


namespace value_of_nested_custom_div_l609_60944

def custom_div (x y z : ℕ) (hz : z ≠ 0) : ℕ :=
  (x + y) / z

theorem value_of_nested_custom_div : custom_div (custom_div 45 15 60 (by decide)) (custom_div 3 3 6 (by decide)) (custom_div 20 10 30 (by decide)) (by decide) = 2 :=
sorry

end value_of_nested_custom_div_l609_60944


namespace interest_calculation_l609_60986

/-- Define the initial deposit in thousands of yuan (50,000 yuan = 5 x 10,000 yuan) -/
def principal : ℕ := 5

/-- Define the annual interest rate as a percentage in decimal form -/
def annual_interest_rate : ℝ := 0.04

/-- Define the number of years for the deposit -/
def years : ℕ := 3

/-- Calculate the total amount after 3 years using compound interest -/
def total_amount_after_3_years : ℝ :=
  principal * (1 + annual_interest_rate) ^ years

/-- Calculate the interest earned after 3 years -/
def interest_earned : ℝ :=
  total_amount_after_3_years - principal

theorem interest_calculation :
  interest_earned = 5 * (1 + 0.04) ^ 3 - 5 :=
by 
  sorry

end interest_calculation_l609_60986


namespace original_earnings_l609_60916

variable (x : ℝ) -- John's original weekly earnings

theorem original_earnings:
  (1.20 * x = 72) → 
  (x = 60) :=
by
  intro h
  sorry

end original_earnings_l609_60916


namespace maximize_a2_b2_c2_d2_l609_60971

theorem maximize_a2_b2_c2_d2 
  (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 85)
  (h3 : ad + bc = 187)
  (h4 : cd = 110) :
  a^2 + b^2 + c^2 + d^2 ≤ 120 :=
sorry

end maximize_a2_b2_c2_d2_l609_60971


namespace xy_relationship_l609_60913

theorem xy_relationship (x y : ℤ) (h1 : 2 * x - y > x + 1) (h2 : x + 2 * y < 2 * y - 3) :
  x < -3 ∧ y < -4 ∧ x > y + 1 :=
sorry

end xy_relationship_l609_60913


namespace base_conversion_subtraction_l609_60929

def base6_to_base10 (n : Nat) : Nat :=
  n / 100000 * 6^5 +
  (n / 10000 % 10) * 6^4 +
  (n / 1000 % 10) * 6^3 +
  (n / 100 % 10) * 6^2 +
  (n / 10 % 10) * 6^1 +
  (n % 10) * 6^0

def base7_to_base10 (n : Nat) : Nat :=
  n / 10000 * 7^4 +
  (n / 1000 % 10) * 7^3 +
  (n / 100 % 10) * 7^2 +
  (n / 10 % 10) * 7^1 +
  (n % 10) * 7^0

theorem base_conversion_subtraction :
  base6_to_base10 543210 - base7_to_base10 43210 = 34052 := by
  sorry

end base_conversion_subtraction_l609_60929


namespace prime_square_pairs_l609_60978

theorem prime_square_pairs (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
    ∃ n : Nat, p^2 + 5 * p * q + 4 * q^2 = n^2 ↔ (p = 13 ∧ q = 3) ∨ (p = 7 ∧ q = 5) ∨ (p = 5 ∧ q = 11) ∨ (p = 3 ∧ q = 13) ∨ (p = 5 ∧ q = 7) ∨ (p = 11 ∧ q = 5) :=
by
  sorry

end prime_square_pairs_l609_60978


namespace downstream_speed_l609_60917

-- Define the given conditions as constants
def V_u : ℝ := 25 -- upstream speed in kmph
def V_m : ℝ := 40 -- speed of the man in still water in kmph

-- Define the speed of the stream
def V_s := V_m - V_u

-- Define the downstream speed
def V_d := V_m + V_s

-- Assertion we need to prove
theorem downstream_speed : V_d = 55 := by
  sorry

end downstream_speed_l609_60917


namespace third_roll_six_probability_l609_60910

noncomputable def Die_A_six_prob : ℚ := 1 / 6
noncomputable def Die_B_six_prob : ℚ := 1 / 2
noncomputable def Die_C_one_prob : ℚ := 3 / 5
noncomputable def Die_B_not_six_prob : ℚ := 1 / 10
noncomputable def Die_C_not_one_prob : ℚ := 1 / 15

noncomputable def prob_two_sixes_die_A : ℚ := Die_A_six_prob ^ 2
noncomputable def prob_two_sixes_die_B : ℚ := Die_B_six_prob ^ 2
noncomputable def prob_two_sixes_die_C : ℚ := Die_C_not_one_prob ^ 2

noncomputable def total_prob_two_sixes : ℚ := 
  (1 / 3) * (prob_two_sixes_die_A + prob_two_sixes_die_B + prob_two_sixes_die_C)

noncomputable def cond_prob_die_A_given_two_sixes : ℚ := prob_two_sixes_die_A / total_prob_two_sixes
noncomputable def cond_prob_die_B_given_two_sixes : ℚ := prob_two_sixes_die_B / total_prob_two_sixes
noncomputable def cond_prob_die_C_given_two_sixes : ℚ := prob_two_sixes_die_C / total_prob_two_sixes

noncomputable def prob_third_six : ℚ := 
  cond_prob_die_A_given_two_sixes * Die_A_six_prob + 
  cond_prob_die_B_given_two_sixes * Die_B_six_prob + 
  cond_prob_die_C_given_two_sixes * Die_C_not_one_prob

theorem third_roll_six_probability : 
  prob_third_six = sorry := 
  sorry

end third_roll_six_probability_l609_60910


namespace chef_sold_12_meals_l609_60901

theorem chef_sold_12_meals
  (initial_meals_lunch : ℕ)
  (additional_meals_dinner : ℕ)
  (meals_left_after_lunch : ℕ)
  (meals_for_dinner : ℕ)
  (H1 : initial_meals_lunch = 17)
  (H2 : additional_meals_dinner = 5)
  (H3 : meals_for_dinner = 10) :
  ∃ (meals_sold_lunch : ℕ), meals_sold_lunch = 12 := by
  sorry

end chef_sold_12_meals_l609_60901


namespace altitude_segment_product_eq_half_side_diff_square_l609_60953

noncomputable def altitude_product (a b c t m m_1: ℝ) :=
  m * m_1 = (b^2 + c^2 - a^2) / 2

theorem altitude_segment_product_eq_half_side_diff_square {a b c t m m_1: ℝ}
  (hm : m = 2 * t / a)
  (hm_1 : m_1 = a * (b^2 + c^2 - a^2) / (4 * t)) :
  altitude_product a b c t m m_1 :=
by sorry

end altitude_segment_product_eq_half_side_diff_square_l609_60953


namespace find_a_l609_60998

def star (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a (a : ℝ) (h : star a 4 = 17) : a = 49 / 3 :=
by sorry

end find_a_l609_60998


namespace rosa_calls_pages_l609_60920

theorem rosa_calls_pages (pages_last_week : ℝ) (pages_this_week : ℝ) (h_last_week : pages_last_week = 10.2) (h_this_week : pages_this_week = 8.6) : pages_last_week + pages_this_week = 18.8 :=
by sorry

end rosa_calls_pages_l609_60920


namespace factorization_of_polynomial_l609_60995

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l609_60995


namespace matrix_self_inverse_pairs_l609_60981

theorem matrix_self_inverse_pairs :
  ∃ p : Finset (ℝ × ℝ), (∀ a d, (a, d) ∈ p ↔ (∃ (m : Matrix (Fin 2) (Fin 2) ℝ), 
    m = !![a, 4; -9, d] ∧ m * m = 1)) ∧ p.card = 2 :=
by {
  sorry
}

end matrix_self_inverse_pairs_l609_60981


namespace min_value_expression_l609_60964

theorem min_value_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 48) :
  x^2 + 6 * x * y + 9 * y^2 + 4 * z^2 ≥ 128 := 
sorry

end min_value_expression_l609_60964


namespace smallest_number_is_42_l609_60918

theorem smallest_number_is_42 (x : ℤ) 
  (h1 : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 225)
  (h2 : x % 7 = 0) : 
  x = 42 := 
sorry

end smallest_number_is_42_l609_60918


namespace alcohol_quantity_in_mixture_l609_60938

theorem alcohol_quantity_in_mixture 
  (A W : ℝ)
  (h1 : A / W = 4 / 3)
  (h2 : A / (W + 4) = 4 / 5)
  : A = 8 :=
sorry

end alcohol_quantity_in_mixture_l609_60938
