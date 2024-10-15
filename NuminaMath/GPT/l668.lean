import Mathlib

namespace NUMINAMATH_GPT_courses_choice_l668_66859

theorem courses_choice (total_courses : ℕ) (chosen_courses : ℕ)
  (h_total_courses : total_courses = 5)
  (h_chosen_courses : chosen_courses = 2) :
  ∃ (ways : ℕ), ways = 60 ∧
    (ways = ((Nat.choose total_courses chosen_courses)^2) - 
            (Nat.choose total_courses chosen_courses) - 
            ((Nat.choose total_courses chosen_courses) * 
             (Nat.choose (total_courses - chosen_courses) chosen_courses))) :=
by
  sorry

end NUMINAMATH_GPT_courses_choice_l668_66859


namespace NUMINAMATH_GPT_joe_total_paint_used_l668_66888

-- Define the initial amount of paint Joe buys.
def initial_paint : ℕ := 360

-- Define the fraction of paint used during the first week.
def first_week_fraction := 1 / 4

-- Define the fraction of remaining paint used during the second week.
def second_week_fraction := 1 / 2

-- Define the total paint used by Joe in the first week.
def paint_used_first_week := first_week_fraction * initial_paint

-- Define the remaining paint after the first week.
def remaining_paint_after_first_week := initial_paint - paint_used_first_week

-- Define the total paint used by Joe in the second week.
def paint_used_second_week := second_week_fraction * remaining_paint_after_first_week

-- Define the total paint used by Joe.
def total_paint_used := paint_used_first_week + paint_used_second_week

-- The theorem to be proven: the total amount of paint Joe has used is 225 gallons.
theorem joe_total_paint_used : total_paint_used = 225 := by
  sorry

end NUMINAMATH_GPT_joe_total_paint_used_l668_66888


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l668_66803

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + x - 12 > 0) → (x > 3 ∨ x < -4) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l668_66803


namespace NUMINAMATH_GPT_number_of_boys_in_biology_class_l668_66806

variable (B G : ℕ) (PhysicsClass BiologyClass : ℕ)

theorem number_of_boys_in_biology_class
  (h1 : G = 3 * B)
  (h2 : PhysicsClass = 200)
  (h3 : BiologyClass = PhysicsClass / 2)
  (h4 : BiologyClass = B + G) :
  B = 25 := by
  sorry

end NUMINAMATH_GPT_number_of_boys_in_biology_class_l668_66806


namespace NUMINAMATH_GPT_simplify_and_evaluate_l668_66866

theorem simplify_and_evaluate (a : ℚ) (h : a = -1/6) : 
  2 * (a + 1) * (a - 1) - a * (2 * a - 3) = -5 / 2 := by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l668_66866


namespace NUMINAMATH_GPT_number_of_committees_correct_l668_66846

noncomputable def number_of_committees (teams members host_selection non_host_selection : ℕ) : ℕ :=
  have ways_to_choose_host := teams
  have ways_to_choose_four_from_seven := Nat.choose members host_selection
  have ways_to_choose_two_from_seven := Nat.choose members non_host_selection
  have total_non_host_combinations := ways_to_choose_two_from_seven ^ (teams - 1)
  ways_to_choose_host * ways_to_choose_four_from_seven * total_non_host_combinations

theorem number_of_committees_correct :
  number_of_committees 5 7 4 2 = 34134175 := by
  sorry

end NUMINAMATH_GPT_number_of_committees_correct_l668_66846


namespace NUMINAMATH_GPT_man_speed_in_still_water_l668_66889

theorem man_speed_in_still_water
  (vm vs : ℝ)
  (h1 : vm + vs = 6)  -- effective speed downstream
  (h2 : vm - vs = 4)  -- effective speed upstream
  : vm = 5 := 
by
  sorry

end NUMINAMATH_GPT_man_speed_in_still_water_l668_66889


namespace NUMINAMATH_GPT_area_of_circle_l668_66852

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end NUMINAMATH_GPT_area_of_circle_l668_66852


namespace NUMINAMATH_GPT_wall_length_l668_66895

theorem wall_length (mirror_side length width : ℝ) (h_mirror : mirror_side = 21) (h_width : width = 28) 
  (h_area_relation : (mirror_side * mirror_side) * 2 = width * length) : length = 31.5 :=
by
  -- here you start the proof, but it's not required for the statement
  sorry

end NUMINAMATH_GPT_wall_length_l668_66895


namespace NUMINAMATH_GPT_perfect_square_solutions_l668_66853

theorem perfect_square_solutions :
  {n : ℕ | ∃ m : ℕ, n^2 + 77 * n = m^2} = {4, 99, 175, 1444} :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_solutions_l668_66853


namespace NUMINAMATH_GPT_red_stars_eq_35_l668_66875

-- Define the conditions
noncomputable def number_of_total_stars (x : ℕ) : ℕ := x + 20 + 15
noncomputable def red_star_frequency (x : ℕ) : ℚ := x / (number_of_total_stars x : ℚ)

-- Define the theorem statement
theorem red_stars_eq_35 : ∃ x : ℕ, red_star_frequency x = 0.5 ↔ x = 35 := sorry

end NUMINAMATH_GPT_red_stars_eq_35_l668_66875


namespace NUMINAMATH_GPT_avg_weight_b_c_43_l668_66835

noncomputable def weights_are_correct (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧ (A + B) / 2 = 40 ∧ B = 31

theorem avg_weight_b_c_43 (A B C : ℝ) (h : weights_are_correct A B C) : (B + C) / 2 = 43 :=
by sorry

end NUMINAMATH_GPT_avg_weight_b_c_43_l668_66835


namespace NUMINAMATH_GPT_factorize_expression_l668_66840

theorem factorize_expression (x : ℝ) : x^3 - 2 * x^2 + x = x * (x - 1)^2 :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l668_66840


namespace NUMINAMATH_GPT_find_x_l668_66874

-- Define the conditions
def cherryGum := 25
def grapeGum := 35
def packs (x : ℚ) := x -- Each pack contains exactly x pieces of gum

-- Define the ratios after losing one pack of cherry gum and finding 6 packs of grape gum
def ratioAfterLosingCherryPack (x : ℚ) := (cherryGum - packs x) / grapeGum
def ratioAfterFindingGrapePacks (x : ℚ) := cherryGum / (grapeGum + 6 * packs x)

-- State the theorem to be proved
theorem find_x (x : ℚ) (h : ratioAfterLosingCherryPack x = ratioAfterFindingGrapePacks x) : x = 115 / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l668_66874


namespace NUMINAMATH_GPT_Tony_fever_l668_66891

theorem Tony_fever :
  ∀ (normal_temp sickness_increase fever_threshold : ℕ),
    normal_temp = 95 →
    sickness_increase = 10 →
    fever_threshold = 100 →
    (normal_temp + sickness_increase) - fever_threshold = 5 :=
by
  intros normal_temp sickness_increase fever_threshold h1 h2 h3
  sorry

end NUMINAMATH_GPT_Tony_fever_l668_66891


namespace NUMINAMATH_GPT_watch_A_accurate_l668_66880

variable (T : ℕ) -- Standard time, represented as natural numbers for simplicity
variable (A B : ℕ) -- Watches A and B, also represented as natural numbers
variable (h1 : A = B + 2) -- Watch A is 2 minutes faster than Watch B
variable (h2 : B = T - 2) -- Watch B is 2 minutes slower than the standard time

theorem watch_A_accurate : A = T :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_watch_A_accurate_l668_66880


namespace NUMINAMATH_GPT_find_result_of_adding_8_l668_66820

theorem find_result_of_adding_8 (x : ℕ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end NUMINAMATH_GPT_find_result_of_adding_8_l668_66820


namespace NUMINAMATH_GPT_solve_system_l668_66887

variable (x y z : ℝ)

theorem solve_system :
  (y + z = 20 - 4 * x) →
  (x + z = -18 - 4 * y) →
  (x + y = 10 - 4 * z) →
  (2 * x + 2 * y + 2 * z = 4) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_solve_system_l668_66887


namespace NUMINAMATH_GPT_max_k_pos_l668_66851

-- Define the sequences {a_n} and {b_n}
def sequence_a (n k : ℤ) : ℤ := 2 * n + k - 1
def sequence_b (n : ℤ) : ℤ := 3 * n + 2

-- Conditions and given values
def S (n k : ℤ) : ℤ := n + k
def sum_first_9_b : ℤ := 153
def b_3 : ℤ := 11

-- Given the sequence {c_n}
def sequence_c (n k : ℤ) : ℤ := sequence_a n k - k * sequence_b n

-- Define the sum of the first n terms of the sequence {c_n}
def T (n k : ℤ) : ℤ := (n * (2 * sequence_c 1 k + (n - 1) * (2 - 3 * k))) / 2

-- Proof problem statement
theorem max_k_pos (k : ℤ) : (∀ n : ℤ, n > 0 → T n k > 0) → k ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_k_pos_l668_66851


namespace NUMINAMATH_GPT_g_difference_l668_66821

def g (n : ℕ) : ℚ :=
  1/4 * n * (n + 1) * (n + 2) * (n + 3)

theorem g_difference (r : ℕ) : g r - g (r - 1) = r * (r + 1) * (r + 2) :=
  sorry

end NUMINAMATH_GPT_g_difference_l668_66821


namespace NUMINAMATH_GPT_train_length_l668_66848

noncomputable def relative_speed_kmh (vA vB : ℝ) : ℝ :=
  vA - vB

noncomputable def relative_speed_mps (relative_speed_kmh : ℝ) : ℝ :=
  relative_speed_kmh * (5 / 18)

noncomputable def distance_covered (relative_speed_mps : ℝ) (time_s : ℝ) : ℝ :=
  relative_speed_mps * time_s

theorem train_length (vA_kmh : ℝ) (vB_kmh : ℝ) (time_s : ℝ) (L : ℝ) 
  (h1 : vA_kmh = 42) (h2 : vB_kmh = 36) (h3 : time_s = 36) 
  (h4 : distance_covered (relative_speed_mps (relative_speed_kmh vA_kmh vB_kmh)) time_s = 2 * L) :
  L = 30 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l668_66848


namespace NUMINAMATH_GPT_dora_rate_correct_l668_66867

noncomputable def betty_rate : ℕ := 10
noncomputable def dora_rate : ℕ := 8
noncomputable def total_time : ℕ := 5
noncomputable def betty_break_time : ℕ := 2
noncomputable def cupcakes_difference : ℕ := 10

theorem dora_rate_correct :
  ∃ D : ℕ, 
  (D = dora_rate) ∧ 
  ((total_time - betty_break_time) * betty_rate = 30) ∧ 
  (total_time * D - 30 = cupcakes_difference) :=
sorry

end NUMINAMATH_GPT_dora_rate_correct_l668_66867


namespace NUMINAMATH_GPT_minimum_familiar_pairs_l668_66844

theorem minimum_familiar_pairs (n : ℕ) (students : Finset (Fin n)) 
  (familiar : Finset (Fin n × Fin n))
  (h_n : n = 175)
  (h_condition : ∀ (s : Finset (Fin n)), s.card = 6 → 
    ∃ (s1 s2 : Finset (Fin n)), s1 ∪ s2 = s ∧ s1.card = 3 ∧ s2.card = 3 ∧ 
    ∀ x ∈ s1, ∀ y ∈ s1, (x ≠ y → (x, y) ∈ familiar) ∧
    ∀ x ∈ s2, ∀ y ∈ s2, (x ≠ y → (x, y) ∈ familiar)) :
  ∃ m : ℕ, m = 15050 ∧ ∀ p : ℕ, (∃ g : Finset (Fin n × Fin n), g.card = p) → p ≥ m := 
sorry

end NUMINAMATH_GPT_minimum_familiar_pairs_l668_66844


namespace NUMINAMATH_GPT_circle_equation_l668_66871

theorem circle_equation 
  (x y : ℝ)
  (passes_origin : (x, y) = (0, 0))
  (intersects_line : ∃ (x y : ℝ), 2 * x - y + 1 = 0)
  (intersects_circle : ∃ (x y :ℝ), x^2 + y^2 - 2 * x - 15 = 0) : 
  x^2 + y^2 + 28 * x - 15 * y = 0 :=
sorry

end NUMINAMATH_GPT_circle_equation_l668_66871


namespace NUMINAMATH_GPT_fourth_child_sweets_l668_66802

theorem fourth_child_sweets (total_sweets : ℕ) (mother_sweets : ℕ) (child_sweets : ℕ) 
  (Y E T F: ℕ) (h1 : total_sweets = 120) (h2 : mother_sweets = total_sweets / 4) 
  (h3 : child_sweets = total_sweets - mother_sweets) 
  (h4 : E = 2 * Y) (h5 : T = F - 8) 
  (h6 : Y = (8 * (T + 6)) / 10) 
  (h7 : Y + E + (T + 6) + (F - 8) + F = child_sweets) : 
  F = 24 :=
by
  sorry

end NUMINAMATH_GPT_fourth_child_sweets_l668_66802


namespace NUMINAMATH_GPT_expression_value_l668_66892

theorem expression_value : (1 * 3 * 5 * 7) / (1^2 + 2^2 + 3^2 + 4^2) = 7 / 2 := by
  sorry

end NUMINAMATH_GPT_expression_value_l668_66892


namespace NUMINAMATH_GPT_min_value_of_expression_l668_66858

theorem min_value_of_expression (x y : ℝ) (h : x^2 + y^2 + x * y = 315) :
  ∃ m : ℝ, m = x^2 + y^2 - x * y ∧ m ≥ 105 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l668_66858


namespace NUMINAMATH_GPT_num_valid_lists_l668_66834

-- Define a predicate for a list to satisfy the given constraints
def valid_list (l : List ℕ) : Prop :=
  l = List.range' 1 12 ∧ ∀ i, 1 < i ∧ i ≤ 12 → (l.indexOf (l.get! (i - 1) + 1) < i - 1 ∨ l.indexOf (l.get! (i - 1) - 1) < i - 1) ∧ ¬(l.indexOf (l.get! (i - 1) + 1) < i - 1 ∧ l.indexOf (l.get! (i - 1) - 1) < i - 1)

-- Prove that there is exactly one valid list of such nature
theorem num_valid_lists : ∃! l : List ℕ, valid_list l :=
  sorry

end NUMINAMATH_GPT_num_valid_lists_l668_66834


namespace NUMINAMATH_GPT_inequality_proof_l668_66826

open Real

theorem inequality_proof
  (a b c x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hx_cond : 1 / x + 1 / y + 1 / z = 1) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a ^ x + b ^ y + c ^ z ≥ (4 * a * b * c * x * y * z) / (x + y + z - 3) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l668_66826


namespace NUMINAMATH_GPT_molecular_weight_3_moles_ascorbic_acid_l668_66842

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_formula_ascorbic_acid : List (ℝ × ℕ) :=
  [(atomic_weight_C, 6), (atomic_weight_H, 8), (atomic_weight_O, 6)]

def molecular_weight (formula : List (ℝ × ℕ)) : ℝ :=
  formula.foldl (λ acc (aw, count) => acc + aw * count) 0.0

def weight_of_moles (mw : ℝ) (moles : ℕ) : ℝ :=
  mw * moles

theorem molecular_weight_3_moles_ascorbic_acid :
  weight_of_moles (molecular_weight molecular_formula_ascorbic_acid) 3 = 528.372 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_3_moles_ascorbic_acid_l668_66842


namespace NUMINAMATH_GPT_paco_initial_cookies_l668_66823

theorem paco_initial_cookies :
  ∀ (total_cookies initially_ate initially_gave : ℕ),
    initially_ate = 14 →
    initially_gave = 13 →
    initially_ate = initially_gave + 1 →
    total_cookies = initially_ate + initially_gave →
    total_cookies = 27 :=
by
  intros total_cookies initially_ate initially_gave h_ate h_gave h_diff h_sum
  sorry

end NUMINAMATH_GPT_paco_initial_cookies_l668_66823


namespace NUMINAMATH_GPT_aladdin_no_profit_l668_66882

theorem aladdin_no_profit (x : ℕ) :
  (x + 1023000) / 1024 <= x :=
by
  sorry

end NUMINAMATH_GPT_aladdin_no_profit_l668_66882


namespace NUMINAMATH_GPT_area_of_triangles_equal_l668_66861

theorem area_of_triangles_equal {a b c d : ℝ} (h_hyperbola_a : a ≠ 0) (h_hyperbola_b : b ≠ 0) 
    (h_hyperbola_c : c ≠ 0) (h_hyperbola_d : d ≠ 0) (h_parallel : a * b = c * d) :
  (1 / 2) * ((a + c) * (a + c) / (a * c)) = (1 / 2) * ((b + d) * (b + d) / (b * d)) :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangles_equal_l668_66861


namespace NUMINAMATH_GPT_extreme_values_f_a4_no_zeros_f_on_1e_l668_66810

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (a + 2) * Real.log x - 2 / x + 2

theorem extreme_values_f_a4 :
  f 4 (1 / 2) = 6 * Real.log 2 ∧ f 4 1 = 4 := sorry

theorem no_zeros_f_on_1e (a : ℝ) :
  (a ≤ 0 ∨ a ≥ 2 / (Real.exp 1 * (Real.exp 1 - 1))) →
  ∀ x, 1 < x → x < Real.exp 1 → f a x ≠ 0 := sorry

end NUMINAMATH_GPT_extreme_values_f_a4_no_zeros_f_on_1e_l668_66810


namespace NUMINAMATH_GPT_xyz_range_l668_66824

theorem xyz_range (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x^2 + y^2 + z^2 = 3) : 
  -1 ≤ x * y * z ∧ x * y * z ≤ 5 / 27 := 
sorry

end NUMINAMATH_GPT_xyz_range_l668_66824


namespace NUMINAMATH_GPT_op_five_two_is_twentyfour_l668_66825

def op (x y : Int) : Int :=
  (x + y + 1) * (x - y)

theorem op_five_two_is_twentyfour : op 5 2 = 24 := by
  unfold op
  sorry

end NUMINAMATH_GPT_op_five_two_is_twentyfour_l668_66825


namespace NUMINAMATH_GPT_slower_train_speed_l668_66830

-- Conditions
variables (L : ℕ) -- Length of each train (in meters)
variables (v_f : ℕ) -- Speed of the faster train (in km/hr)
variables (t : ℕ) -- Time taken by the faster train to pass the slower one (in seconds)
variables (v_s : ℕ) -- Speed of the slower train (in km/hr)

-- Assumptions based on conditions of the problem
axiom length_eq : L = 30
axiom fast_speed : v_f = 42
axiom passing_time : t = 36

-- Conversion for km/hr to m/s
def km_per_hr_to_m_per_s (v : ℕ) : ℕ := (v * 5) / 18

-- Problem statement
theorem slower_train_speed : v_s = 36 :=
by
  let rel_speed := km_per_hr_to_m_per_s (v_f - v_s)
  have rel_speed_def : rel_speed = (42 - v_s) * 5 / 18 := by sorry
  have distance : 60 = rel_speed * t := by sorry
  have equation : 60 = (42 - v_s) * 10 := by sorry
  have solve_v_s : v_s = 36 := by sorry
  exact solve_v_s

end NUMINAMATH_GPT_slower_train_speed_l668_66830


namespace NUMINAMATH_GPT_bananas_per_friend_l668_66850

-- Define the conditions
def total_bananas : ℕ := 40
def number_of_friends : ℕ := 40

-- Define the theorem to be proved
theorem bananas_per_friend : 
  (total_bananas / number_of_friends) = 1 :=
by
  sorry

end NUMINAMATH_GPT_bananas_per_friend_l668_66850


namespace NUMINAMATH_GPT_marbles_given_l668_66804

theorem marbles_given (initial remaining given : ℕ) (h_initial : initial = 143) (h_remaining : remaining = 70) :
    given = initial - remaining → given = 73 :=
by
  intros
  sorry

end NUMINAMATH_GPT_marbles_given_l668_66804


namespace NUMINAMATH_GPT_find_x_l668_66818

theorem find_x :
  ∃ X : ℝ, 0.25 * X + 0.20 * 40 = 23 ∧ X = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l668_66818


namespace NUMINAMATH_GPT_plane_through_line_and_point_l668_66890

-- Definitions from the conditions
def line (x y z : ℝ) : Prop :=
  (x - 1) / 2 = (y - 3) / 4 ∧ (x - 1) / 2 = z / (-1)

def pointP1 : ℝ × ℝ × ℝ := (1, 5, 2)

-- Correct answer
def plane_eqn (x y z : ℝ) : Prop :=
  5 * x - 2 * y + 2 * z + 1 = 0

-- The theorem to prove
theorem plane_through_line_and_point (x y z : ℝ) :
  line x y z → plane_eqn x y z := by
  sorry

end NUMINAMATH_GPT_plane_through_line_and_point_l668_66890


namespace NUMINAMATH_GPT_rewrite_expression_l668_66886

theorem rewrite_expression : ∀ x : ℝ, x^2 + 4 * x + 1 = (x + 2)^2 - 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rewrite_expression_l668_66886


namespace NUMINAMATH_GPT_part1_quantity_of_vegetables_part2_functional_relationship_part3_min_vegetable_a_l668_66845

/-- Part 1: Quantities of vegetables A and B wholesaled. -/
theorem part1_quantity_of_vegetables (x y : ℝ) 
  (h1 : x + y = 40) 
  (h2 : 4.8 * x + 4 * y = 180) : 
  x = 25 ∧ y = 15 :=
sorry

/-- Part 2: Functional relationship between m and n. -/
theorem part2_functional_relationship (n m : ℝ) 
  (h : n ≤ 80) 
  (h2 : m = 4.8 * n + 4 * (80 - n)) : 
  m = 0.8 * n + 320 :=
sorry

/-- Part 3: Minimum amount of vegetable A to ensure profit of at least 176 yuan -/
theorem part3_min_vegetable_a (n : ℝ) 
  (h : 0.8 * n + 128 ≥ 176) : 
  n ≥ 60 :=
sorry

end NUMINAMATH_GPT_part1_quantity_of_vegetables_part2_functional_relationship_part3_min_vegetable_a_l668_66845


namespace NUMINAMATH_GPT_number_of_zeros_f_l668_66811

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.log x

theorem number_of_zeros_f : ∃! n : ℕ, n = 2 ∧ ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_GPT_number_of_zeros_f_l668_66811


namespace NUMINAMATH_GPT_transformed_triangle_area_l668_66807

-- Define the function g and its properties
variable {R : Type*} [LinearOrderedField R]
variable (g : R → R)
variable (a b c : R)
variable (area_original : R)

-- Given conditions
-- The function g is defined such that the area of the triangle formed by 
-- points (a, g(a)), (b, g(b)), and (c, g(c)) is 24
axiom h₀ : {x | x = a ∨ x = b ∨ x = c} ⊆ Set.univ
axiom h₁ : area_original = 24

-- Define a function that computes the area of a triangle given three points
noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : R) : R := 
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem transformed_triangle_area (h₀ : {x | x = a ∨ x = b ∨ x = c} ⊆ Set.univ)
  (h₁ : area_triangle a (g a) b (g b) c (g c) = 24) :
  area_triangle (a / 3) (3 * g a) (b / 3) (3 * g b) (c / 3) (3 * g c) = 24 :=
sorry

end NUMINAMATH_GPT_transformed_triangle_area_l668_66807


namespace NUMINAMATH_GPT_GCF_of_LCMs_l668_66864

def GCF : ℕ → ℕ → ℕ := Nat.gcd
def LCM : ℕ → ℕ → ℕ := Nat.lcm

theorem GCF_of_LCMs :
  GCF (LCM 9 21) (LCM 10 15) = 3 :=
by
  sorry

end NUMINAMATH_GPT_GCF_of_LCMs_l668_66864


namespace NUMINAMATH_GPT_cylinder_volume_l668_66863

theorem cylinder_volume (r h : ℝ) (π : ℝ) 
  (h_pos : 0 < π) 
  (cond1 : 2 * π * r * h = 100 * π) 
  (cond2 : 4 * r^2 + h^2 = 200) : 
  (π * r^2 * h = 250 * π) := 
by 
  sorry

end NUMINAMATH_GPT_cylinder_volume_l668_66863


namespace NUMINAMATH_GPT_toys_produced_per_week_l668_66800

theorem toys_produced_per_week (daily_production : ℕ) (work_days_per_week : ℕ) (total_production : ℕ) :
  daily_production = 680 ∧ work_days_per_week = 5 → total_production = 3400 := by
  sorry

end NUMINAMATH_GPT_toys_produced_per_week_l668_66800


namespace NUMINAMATH_GPT_find_x_l668_66857

theorem find_x (x : ℝ) (h : 0.65 * x = 0.20 * 682.50) : x = 210 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l668_66857


namespace NUMINAMATH_GPT_exists_ints_a_b_l668_66896

theorem exists_ints_a_b (n : ℤ) (h : n % 4 ≠ 2) : ∃ a b : ℤ, n + a^2 = b^2 :=
by
  sorry

end NUMINAMATH_GPT_exists_ints_a_b_l668_66896


namespace NUMINAMATH_GPT_apple_distribution_l668_66833

theorem apple_distribution (x : ℕ) (h₁ : 1430 % x = 0) (h₂ : 1430 % (x + 45) = 0) (h₃ : 1430 / x - 1430 / (x + 45) = 9) : 
  1430 / x = 22 :=
by
  sorry

end NUMINAMATH_GPT_apple_distribution_l668_66833


namespace NUMINAMATH_GPT_fold_string_twice_l668_66873

theorem fold_string_twice (initial_length : ℕ) (half_folds : ℕ) (result_length : ℕ) 
  (h1 : initial_length = 12)
  (h2 : half_folds = 2)
  (h3 : result_length = initial_length / (2 ^ half_folds)) :
  result_length = 3 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_fold_string_twice_l668_66873


namespace NUMINAMATH_GPT_product_4_7_25_l668_66884

theorem product_4_7_25 : 4 * 7 * 25 = 700 :=
by sorry

end NUMINAMATH_GPT_product_4_7_25_l668_66884


namespace NUMINAMATH_GPT_number_of_girls_l668_66847

theorem number_of_girls
  (B : ℕ) (k : ℕ) (G : ℕ)
  (hB : B = 10) 
  (hk : k = 5)
  (h1 : B / k = 2)
  (h2 : G % k = 0) :
  G = 5 := 
sorry

end NUMINAMATH_GPT_number_of_girls_l668_66847


namespace NUMINAMATH_GPT_acute_angle_at_315_equals_7_5_l668_66894

/-- The degrees in a full circle -/
def fullCircle := 360

/-- The number of hours on a clock -/
def hoursOnClock := 12

/-- The measure in degrees of the acute angle formed by the minute hand and the hour hand at 3:15 -/
def acuteAngleAt315 : ℝ :=
  let degreesPerHour := fullCircle / hoursOnClock
  let hourHandAt3 := degreesPerHour * 3
  let additionalDegrees := (15 / 60) * degreesPerHour
  let hourHandPosition := hourHandAt3 + additionalDegrees
  let minuteHandPosition := (15 / 60) * fullCircle
  abs (hourHandPosition - minuteHandPosition)

theorem acute_angle_at_315_equals_7_5 : acuteAngleAt315 = 7.5 := by
  sorry

end NUMINAMATH_GPT_acute_angle_at_315_equals_7_5_l668_66894


namespace NUMINAMATH_GPT_radius_B_eq_8_div_9_l668_66819

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Given conditions
variable (A B C D : Circle)
variable (h1 : A.radius = 1)
variable (h2 : A.radius + A.radius = D.radius)
variable (h3 : B.radius = C.radius)
variable (h4 : (A.center.1 - B.center.1)^2 + (A.center.2 - B.center.2)^2 = (A.radius + B.radius)^2)
variable (h5 : (A.center.1 - C.center.1)^2 + (A.center.2 - C.center.2)^2 = (A.radius + C.radius)^2)
variable (h6 : (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 = (B.radius + C.radius)^2)
variable (h7 : (D.center.1 - A.center.1)^2 + (D.center.2 - A.center.2)^2 = D.radius^2)

-- Prove the radius of circle B is 8/9
theorem radius_B_eq_8_div_9 : B.radius = 8 / 9 := 
by
  sorry

end NUMINAMATH_GPT_radius_B_eq_8_div_9_l668_66819


namespace NUMINAMATH_GPT_quadratic_coefficients_l668_66839

theorem quadratic_coefficients :
  ∀ x : ℝ, 3 * x^2 = 5 * x - 1 → (∃ a b c : ℝ, a = 3 ∧ b = -5 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x h
  use 3, -5, 1
  sorry

end NUMINAMATH_GPT_quadratic_coefficients_l668_66839


namespace NUMINAMATH_GPT_Cedar_school_earnings_l668_66856

noncomputable def total_earnings_Cedar_school : ℝ :=
  let total_payment := 774
  let total_student_days := 6 * 4 + 5 * 6 + 3 * 10
  let daily_wage := total_payment / total_student_days
  let Cedar_student_days := 3 * 10
  daily_wage * Cedar_student_days

theorem Cedar_school_earnings :
  total_earnings_Cedar_school = 276.43 :=
by
  sorry

end NUMINAMATH_GPT_Cedar_school_earnings_l668_66856


namespace NUMINAMATH_GPT_probability_all_operating_probability_shutdown_l668_66876

-- Define the events and their probabilities
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.85

-- Prove that the probability of all three machines operating without supervision is 0.612
theorem probability_all_operating : P_A * P_B * P_C = 0.612 := 
by sorry

-- Prove that the probability of a shutdown is 0.059
theorem probability_shutdown :
    P_A * (1 - P_B) * (1 - P_C) +
    (1 - P_A) * P_B * (1 - P_C) +
    (1 - P_A) * (1 - P_B) * P_C +
    (1 - P_A) * (1 - P_B) * (1 - P_C) = 0.059 :=
by sorry

end NUMINAMATH_GPT_probability_all_operating_probability_shutdown_l668_66876


namespace NUMINAMATH_GPT_sum_of_two_numbers_l668_66885

theorem sum_of_two_numbers (x y : ℕ) (hxy : x > y) (h1 : x - y = 4) (h2 : x * y = 156) : x + y = 28 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_two_numbers_l668_66885


namespace NUMINAMATH_GPT_pear_juice_percentage_l668_66812

/--
Miki has a dozen oranges and pears. She extracts juice as follows:
5 pears -> 10 ounces of pear juice
3 oranges -> 12 ounces of orange juice
She uses 10 pears and 10 oranges to make a blend.
Prove that the percent of the blend that is pear juice is 33.33%.
-/
theorem pear_juice_percentage :
  let pear_juice_per_pear := 10 / 5
  let orange_juice_per_orange := 12 / 3
  let total_pear_juice := 10 * pear_juice_per_pear
  let total_orange_juice := 10 * orange_juice_per_orange
  let total_juice := total_pear_juice + total_orange_juice
  total_pear_juice / total_juice = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_pear_juice_percentage_l668_66812


namespace NUMINAMATH_GPT_intersection_A_B_l668_66831

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l668_66831


namespace NUMINAMATH_GPT_man_completes_in_9_days_l668_66881

-- Definitions of the work rates and the conditions given
def M : ℚ := sorry
def W : ℚ := 1 / 6
def B : ℚ := 1 / 18
def combined_rate : ℚ := 1 / 3

-- Statement that the man alone can complete the work in 9 days
theorem man_completes_in_9_days
  (h_combined : M + W + B = combined_rate) : 1 / M = 9 :=
  sorry

end NUMINAMATH_GPT_man_completes_in_9_days_l668_66881


namespace NUMINAMATH_GPT_sum_of_reciprocals_l668_66805

variable (x y : ℝ)

theorem sum_of_reciprocals (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  (1 / x) + (1 / y) = 3 := 
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l668_66805


namespace NUMINAMATH_GPT_equation_of_line_AB_l668_66815

noncomputable def center_of_circle : (ℝ × ℝ) := (-4, -1)

noncomputable def point_P : (ℝ × ℝ) := (2, 3)

noncomputable def slope_OP : ℝ :=
  let (x₁, y₁) := center_of_circle
  let (x₂, y₂) := point_P
  (y₂ - y₁) / (x₂ - x₁)

noncomputable def slope_AB : ℝ :=
  -1 / slope_OP

theorem equation_of_line_AB : (6 * x + 4 * y + 19 = 0) :=
  sorry

end NUMINAMATH_GPT_equation_of_line_AB_l668_66815


namespace NUMINAMATH_GPT_find_real_pairs_l668_66893

theorem find_real_pairs (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
  (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) :=
sorry

end NUMINAMATH_GPT_find_real_pairs_l668_66893


namespace NUMINAMATH_GPT_larger_number_is_28_l668_66883

theorem larger_number_is_28
  (x y : ℕ)
  (h1 : 4 * y = 7 * x)
  (h2 : y - x = 12) : y = 28 :=
sorry

end NUMINAMATH_GPT_larger_number_is_28_l668_66883


namespace NUMINAMATH_GPT_closure_property_of_A_l668_66837

theorem closure_property_of_A 
  (a b c d k1 k2 : ℤ) 
  (x y : ℤ) 
  (Hx : x = a^2 + k1 * a * b + b^2) 
  (Hy : y = c^2 + k2 * c * d + d^2) : 
  ∃ m k : ℤ, x * y = m * (a^2 + k * a * b + b^2) := 
  by 
    -- this is where the proof would go
    sorry

end NUMINAMATH_GPT_closure_property_of_A_l668_66837


namespace NUMINAMATH_GPT_sum_eq_24_of_greatest_power_l668_66816

theorem sum_eq_24_of_greatest_power (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_b_gt_1 : b > 1) (h_a_pow_b_lt_500 : a^b < 500)
  (h_greatest : ∀ (x y : ℕ), (0 < x) → (0 < y) → (y > 1) → (x^y < 500) → (x^y ≤ a^b)) : a + b = 24 :=
  sorry

end NUMINAMATH_GPT_sum_eq_24_of_greatest_power_l668_66816


namespace NUMINAMATH_GPT_one_circle_equiv_three_squares_l668_66841

-- Define the weights of circles and squares symbolically
variables {w_circle w_square : ℝ}

-- Equations based on the conditions in the problem
-- 3 circles balance 5 squares
axiom eq1 : 3 * w_circle = 5 * w_square

-- 2 circles balance 3 squares and 1 circle
axiom eq2 : 2 * w_circle = 3 * w_square + w_circle

-- We need to prove that 1 circle is equivalent to 3 squares
theorem one_circle_equiv_three_squares : w_circle = 3 * w_square := 
by sorry

end NUMINAMATH_GPT_one_circle_equiv_three_squares_l668_66841


namespace NUMINAMATH_GPT_solve_max_eq_l668_66836

theorem solve_max_eq (x : ℚ) (h : max x (-x) = 2 * x + 1) : x = -1 / 3 := by
  sorry

end NUMINAMATH_GPT_solve_max_eq_l668_66836


namespace NUMINAMATH_GPT_range_of_a_squared_plus_b_l668_66829

variable (a b : ℝ)

theorem range_of_a_squared_plus_b (h1 : a < -2) (h2 : b > 4) : ∃ y, y = a^2 + b ∧ 8 < y :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_squared_plus_b_l668_66829


namespace NUMINAMATH_GPT_probability_three_consecutive_cards_l668_66879

-- Definitions of the conditions
def total_ways_to_draw_three : ℕ := Nat.choose 52 3

def sets_of_consecutive_ranks : ℕ := 10

def ways_to_choose_three_consecutive : ℕ := 64

def favorable_outcomes : ℕ := sets_of_consecutive_ranks * ways_to_choose_three_consecutive

def probability_consecutive_ranks : ℚ := favorable_outcomes / total_ways_to_draw_three

-- The main statement to prove
theorem probability_three_consecutive_cards :
  probability_consecutive_ranks = 32 / 1105 := 
sorry

end NUMINAMATH_GPT_probability_three_consecutive_cards_l668_66879


namespace NUMINAMATH_GPT_find_interest_rate_l668_66898

noncomputable def interest_rate (total_investment remaining_investment interest_earned part_interest : ℝ) : ℝ :=
  (interest_earned - part_interest) / remaining_investment

theorem find_interest_rate :
  let total_investment := 9000
  let invested_at_8_percent := 4000
  let total_interest := 770
  let interest_at_8_percent := invested_at_8_percent * 0.08
  let remaining_investment := total_investment - invested_at_8_percent
  let interest_from_remaining := total_interest - interest_at_8_percent
  interest_rate total_investment remaining_investment total_interest interest_at_8_percent = 0.09 :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l668_66898


namespace NUMINAMATH_GPT_sequence_number_theorem_l668_66801

def seq_count (n k : ℕ) : ℕ :=
  -- Sequence count function definition given the conditions.
  sorry -- placeholder, as we are only defining the statement, not the function itself.

theorem sequence_number_theorem (n k : ℕ) : seq_count n k = Nat.choose (n-1) k :=
by
  -- This is where the proof would go, currently omitted.
  sorry

end NUMINAMATH_GPT_sequence_number_theorem_l668_66801


namespace NUMINAMATH_GPT_inequality_always_holds_l668_66862

theorem inequality_always_holds
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_def : ∀ x, f x = (1 - 2^x) / (1 + 2^x))
  (h_odd : ∀ x, f (-x) = -f x)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_ineq : f (2 * a + b) + f (4 - 3 * b) > 0)
  : b - a > 2 :=
sorry

end NUMINAMATH_GPT_inequality_always_holds_l668_66862


namespace NUMINAMATH_GPT_dustin_reads_more_pages_l668_66872

theorem dustin_reads_more_pages (dustin_rate_per_hour : ℕ) (sam_rate_per_hour : ℕ) : 
  (dustin_rate_per_hour = 75) → (sam_rate_per_hour = 24) → 
  (dustin_rate_per_hour * 40 / 60 - sam_rate_per_hour * 40 / 60 = 34) :=
by
  sorry

end NUMINAMATH_GPT_dustin_reads_more_pages_l668_66872


namespace NUMINAMATH_GPT_sum_series_a_sum_series_b_sum_series_c_l668_66878

-- Part (a)
theorem sum_series_a : (∑' n : ℕ, (1 / 2) ^ (n + 1)) = 1 := by
  --skip proof
  sorry

-- Part (b)
theorem sum_series_b : (∑' n : ℕ, (1 / 3) ^ (n + 1)) = 1/2 := by
  --skip proof
  sorry

-- Part (c)
theorem sum_series_c : (∑' n : ℕ, (1 / 4) ^ (n + 1)) = 1/3 := by
  --skip proof
  sorry

end NUMINAMATH_GPT_sum_series_a_sum_series_b_sum_series_c_l668_66878


namespace NUMINAMATH_GPT_percentage_markup_l668_66868

theorem percentage_markup (selling_price cost_price : ℝ) (h_selling : selling_price = 2000) (h_cost : cost_price = 1250) :
  ((selling_price - cost_price) / cost_price) * 100 = 60 := by
  sorry

end NUMINAMATH_GPT_percentage_markup_l668_66868


namespace NUMINAMATH_GPT_place_pawns_distinct_5x5_l668_66855

noncomputable def number_of_ways_place_pawns : ℕ :=
  5 * 4 * 3 * 2 * 1 * 120

theorem place_pawns_distinct_5x5 : number_of_ways_place_pawns = 14400 := by
  sorry

end NUMINAMATH_GPT_place_pawns_distinct_5x5_l668_66855


namespace NUMINAMATH_GPT_paul_spending_l668_66849

theorem paul_spending :
  let cost_of_dress_shirts := 4 * 15
  let cost_of_pants := 2 * 40
  let cost_of_suit := 150
  let cost_of_sweaters := 2 * 30
  let total_cost := cost_of_dress_shirts + cost_of_pants + cost_of_suit + cost_of_sweaters
  let store_discount := 0.2 * total_cost
  let after_store_discount := total_cost - store_discount
  let coupon_discount := 0.1 * after_store_discount
  let final_amount := after_store_discount - coupon_discount
  final_amount = 252 :=
by
  -- Mathematically equivalent proof problem.
  sorry

end NUMINAMATH_GPT_paul_spending_l668_66849


namespace NUMINAMATH_GPT_john_total_cost_l668_66877

def base_cost : ℤ := 25
def text_cost_per_message : ℤ := 8
def extra_minute_cost_per_minute : ℤ := 15
def international_minute_cost : ℤ := 100

def texts_sent : ℤ := 200
def total_hours : ℤ := 42
def international_minutes : ℤ := 10

-- Calculate the number of extra minutes
def extra_minutes : ℤ := (total_hours - 40) * 60

noncomputable def total_cost : ℤ :=
  base_cost +
  (texts_sent * text_cost_per_message) / 100 +
  (extra_minutes * extra_minute_cost_per_minute) / 100 +
  international_minutes * (international_minute_cost / 100)

theorem john_total_cost :
  total_cost = 69 := by
    sorry

end NUMINAMATH_GPT_john_total_cost_l668_66877


namespace NUMINAMATH_GPT_parabola_coefficients_sum_l668_66808

theorem parabola_coefficients_sum (a b c : ℝ)
  (h_eqn : ∀ y, (-1) = a * y^2 + b * y + c)
  (h_vertex : (-1, -10) = (-a/(2*a), (4*a*c - b^2)/(4*a)))
  (h_pass_point : 0 = a * (-9)^2 + b * (-9) + c) 
  : a + b + c = 120 := 
sorry

end NUMINAMATH_GPT_parabola_coefficients_sum_l668_66808


namespace NUMINAMATH_GPT_shelves_needed_l668_66843

def books_in_stock : Nat := 27
def books_sold : Nat := 6
def books_per_shelf : Nat := 7

theorem shelves_needed :
  let remaining_books := books_in_stock - books_sold
  let shelves := remaining_books / books_per_shelf
  shelves = 3 :=
by
  sorry

end NUMINAMATH_GPT_shelves_needed_l668_66843


namespace NUMINAMATH_GPT_range_of_m_l668_66865

theorem range_of_m (a : ℝ) (h : a ≠ 0) (x1 x2 y1 y2 : ℝ) (m : ℝ)
  (hx1 : -2 < x1 ∧ x1 < 0) (hx2 : m < x2 ∧ x2 < m + 1)
  (h_on_parabola_A : y1 = a * x1^2 - 2 * a * x1 - 3)
  (h_on_parabola_B : y2 = a * x2^2 - 2 * a * x2 - 3)
  (h_diff_y : y1 ≠ y2) :
  (0 < m ∧ m ≤ 1) ∨ m ≥ 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l668_66865


namespace NUMINAMATH_GPT_negation_of_proposition_is_false_l668_66827

theorem negation_of_proposition_is_false :
  (¬ ∀ (x : ℝ), x < 0 → x^2 > 0) = true :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_is_false_l668_66827


namespace NUMINAMATH_GPT_rachel_math_homework_pages_l668_66822

-- Define the number of pages of math homework and reading homework
def pagesReadingHomework : ℕ := 4

theorem rachel_math_homework_pages (M : ℕ) (h1 : M + 1 = pagesReadingHomework) : M = 3 :=
by
  sorry

end NUMINAMATH_GPT_rachel_math_homework_pages_l668_66822


namespace NUMINAMATH_GPT_math_competition_correct_answers_l668_66899

theorem math_competition_correct_answers (qA qB cA cB : ℕ) 
  (h_total_questions : qA + qB = 10)
  (h_score_A : cA * 5 - (qA - cA) * 2 = 36)
  (h_score_B : cB * 5 - (qB - cB) * 2 = 22) 
  (h_combined_score : cA * 5 - (qA - cA) * 2 + cB * 5 - (qB - cB) * 2 = 58)
  (h_score_difference : cA * 5 - (qA - cA) * 2 - (cB * 5 - (qB - cB) * 2) = 14) : 
  cA = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_math_competition_correct_answers_l668_66899


namespace NUMINAMATH_GPT_eval_expression_l668_66814

theorem eval_expression : (-2 ^ 3) ^ (1/3 : ℝ) - (-1 : ℝ) ^ 0 = -3 := by 
  sorry

end NUMINAMATH_GPT_eval_expression_l668_66814


namespace NUMINAMATH_GPT_problem_solution_l668_66860

theorem problem_solution (a : ℝ) (h : a = Real.sqrt 5 - 1) :
  2 * a^3 + 7 * a^2 - 2 * a - 12 = 0 :=
by 
  sorry  -- Proof placeholder

end NUMINAMATH_GPT_problem_solution_l668_66860


namespace NUMINAMATH_GPT_inverse_proportion_inequality_l668_66838

theorem inverse_proportion_inequality {x1 x2 : ℝ} (h1 : x1 > x2) (h2 : x2 > 0) : 
    -3 / x1 > -3 / x2 := 
by 
  sorry

end NUMINAMATH_GPT_inverse_proportion_inequality_l668_66838


namespace NUMINAMATH_GPT_parabola_condition_max_area_triangle_l668_66870

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_GPT_parabola_condition_max_area_triangle_l668_66870


namespace NUMINAMATH_GPT_cost_of_building_fence_eq_3944_l668_66869

def area_square : ℕ := 289
def price_per_foot : ℕ := 58

theorem cost_of_building_fence_eq_3944 : 
  let side_length := (area_square : ℝ) ^ (1/2)
  let perimeter := 4 * side_length
  let cost := perimeter * (price_per_foot : ℝ)
  cost = 3944 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_building_fence_eq_3944_l668_66869


namespace NUMINAMATH_GPT_sin_beta_value_l668_66854

theorem sin_beta_value (a β : ℝ) (ha : 0 < a ∧ a < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (hcos_a : Real.cos a = 4 / 5)
  (hcos_a_plus_beta : Real.cos (a + β) = 5 / 13) :
  Real.sin β = 63 / 65 :=
sorry

end NUMINAMATH_GPT_sin_beta_value_l668_66854


namespace NUMINAMATH_GPT_total_trash_cans_paid_for_l668_66828

-- Definitions based on conditions
def trash_cans_on_streets : ℕ := 14
def trash_cans_back_of_stores : ℕ := 2 * trash_cans_on_streets

-- Theorem to prove
theorem total_trash_cans_paid_for : trash_cans_on_streets + trash_cans_back_of_stores = 42 := 
by
  -- proof would go here, but we use sorry since proof is not required
  sorry

end NUMINAMATH_GPT_total_trash_cans_paid_for_l668_66828


namespace NUMINAMATH_GPT_scientific_notation_of_909_000_000_000_l668_66817

theorem scientific_notation_of_909_000_000_000 :
    ∃ (a : ℝ) (n : ℤ), 909000000000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 9.09 ∧ n = 11 := 
sorry

end NUMINAMATH_GPT_scientific_notation_of_909_000_000_000_l668_66817


namespace NUMINAMATH_GPT_cube_root_of_64_l668_66897

theorem cube_root_of_64 : ∃ x : ℝ, x^3 = 64 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_cube_root_of_64_l668_66897


namespace NUMINAMATH_GPT_total_hamburgers_for_lunch_l668_66832

theorem total_hamburgers_for_lunch 
  (initial_hamburgers: ℕ) 
  (additional_hamburgers: ℕ)
  (h1: initial_hamburgers = 9)
  (h2: additional_hamburgers = 3)
  : initial_hamburgers + additional_hamburgers = 12 := 
by
  sorry

end NUMINAMATH_GPT_total_hamburgers_for_lunch_l668_66832


namespace NUMINAMATH_GPT_find_smaller_number_l668_66813

theorem find_smaller_number (a b : ℤ) (h1 : a + b = 18) (h2 : a - b = 24) : b = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l668_66813


namespace NUMINAMATH_GPT_solve_quadratic_l668_66809

theorem solve_quadratic {x : ℝ} : x^2 = 2 * x ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l668_66809
