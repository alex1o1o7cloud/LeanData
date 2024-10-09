import Mathlib

namespace below_zero_notation_l914_91419

def celsius_above (x : ℤ) : String := "+" ++ toString x ++ "°C"
def celsius_below (x : ℤ) : String := "-" ++ toString x ++ "°C"

theorem below_zero_notation (h₁ : celsius_above 5 = "+5°C")
  (h₂ : ∀ x : ℤ, x > 0 → celsius_above x = "+" ++ toString x ++ "°C")
  (h₃ : ∀ x : ℤ, x > 0 → celsius_below x = "-" ++ toString x ++ "°C") :
  celsius_below 3 = "-3°C" :=
sorry

end below_zero_notation_l914_91419


namespace sequence_2018_value_l914_91433

theorem sequence_2018_value (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) - a n = (-1 / 2) ^ n) :
  a 2018 = (2 * (1 - (1 / 2) ^ 2018)) / 3 :=
by sorry

end sequence_2018_value_l914_91433


namespace ratio_equal_one_of_log_conditions_l914_91410

noncomputable def logBase (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem ratio_equal_one_of_log_conditions
  (p q : ℝ)
  (hp : 0 < p)
  (hq : 0 < q)
  (h : logBase 8 p = logBase 18 q ∧ logBase 18 q = logBase 24 (p + 2 * q)) :
  q / p = 1 :=
by
  sorry

end ratio_equal_one_of_log_conditions_l914_91410


namespace intersection_m_n_l914_91488

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := { -1, 0, 1, 2, 3 }

theorem intersection_m_n : M ∩ N = {0, 1, 2} := 
sorry

end intersection_m_n_l914_91488


namespace integer_solution_system_l914_91489

theorem integer_solution_system (n : ℕ) (H : n ≥ 2) : 
  ∃ (x : ℕ → ℤ), (
    ∀ i : ℕ, x ((i % n) + 1)^2 + x (((i + 1) % n) + 1)^2 + 50 = 16 * x ((i % n) + 1) + 12 * x (((i + 1) % n) + 1)
  ) ↔ n % 3 = 0 :=
by
  sorry

end integer_solution_system_l914_91489


namespace intersection_M_N_eq_M_l914_91472

-- Definition of M
def M := {y : ℝ | ∃ x : ℝ, y = 3^x}

-- Definition of N
def N := {y : ℝ | ∃ x : ℝ, y = x^2 - 1}

-- Theorem statement
theorem intersection_M_N_eq_M : (M ∩ N) = M :=
  sorry

end intersection_M_N_eq_M_l914_91472


namespace algebraic_expression_value_l914_91481

theorem algebraic_expression_value 
  (p q r s : ℝ) 
  (hpq3 : p^2 / q^3 = 4 / 5) 
  (hrs2 : r^3 / s^2 = 7 / 9) : 
  11 / (7 - r^3 / s^2) + (2 * q^3 - p^2) / (2 * q^3 + p^2) = 123 / 56 := 
by 
  sorry

end algebraic_expression_value_l914_91481


namespace find_a_plus_b_l914_91417

theorem find_a_plus_b (a b : ℝ) 
  (h1 : 2 = a - b / 2) 
  (h2 : 6 = a - b / 3) : 
  a + b = 38 := by
  sorry

end find_a_plus_b_l914_91417


namespace not_possible_to_obtain_target_triple_l914_91498

def is_target_triple_achievable (a1 a2 a3 b1 b2 b3 : ℝ) : Prop :=
  ∀ x y : ℝ, (x, y) = (0.6 * x - 0.8 * y, 0.8 * x + 0.6 * y) →
    (b1^2 + b2^2 + b3^2 = 169 → False)

theorem not_possible_to_obtain_target_triple :
  ¬ is_target_triple_achievable 3 4 12 2 8 10 :=
by sorry

end not_possible_to_obtain_target_triple_l914_91498


namespace Fedya_age_statement_l914_91400

theorem Fedya_age_statement (d a : ℕ) (today : ℕ) (birthday : ℕ) 
    (H1 : d + 2 = a) 
    (H2 : a + 2 = birthday + 3) 
    (H3 : birthday = today + 1) :
    ∃ sameYear y, (birthday < today + 2 ∨ today < birthday) ∧ ((sameYear ∧ y - today = 1) ∨ (¬ sameYear ∧ y - today = 0)) :=
by
  sorry

end Fedya_age_statement_l914_91400


namespace molecular_weight_of_aluminum_part_in_Al2_CO3_3_l914_91438

def total_molecular_weight_Al2_CO3_3 : ℝ := 234
def atomic_weight_Al : ℝ := 26.98
def num_atoms_Al_in_Al2_CO3_3 : ℕ := 2

theorem molecular_weight_of_aluminum_part_in_Al2_CO3_3 :
  num_atoms_Al_in_Al2_CO3_3 * atomic_weight_Al = 53.96 :=
by
  sorry

end molecular_weight_of_aluminum_part_in_Al2_CO3_3_l914_91438


namespace rounding_strategy_correct_l914_91467

-- Definitions of rounding functions
def round_down (n : ℕ) : ℕ := n - 1  -- Assuming n is a large integer, for simplicity
def round_up (n : ℕ) : ℕ := n + 1

-- Definitions for conditions
def cond1 (p q r : ℕ) : ℕ := round_down p / round_down q + round_down r
def cond2 (p q r : ℕ) : ℕ := round_up p / round_down q + round_down r
def cond3 (p q r : ℕ) : ℕ := round_down p / round_up q + round_down r
def cond4 (p q r : ℕ) : ℕ := round_down p / round_down q + round_up r
def cond5 (p q r : ℕ) : ℕ := round_up p / round_up q + round_down r

-- Theorem stating the correct condition
theorem rounding_strategy_correct (p q r : ℕ) (hp : 1 ≤ p) (hq : 1 ≤ q) (hr : 1 ≤ r) :
  cond3 p q r < p / q + r :=
sorry

end rounding_strategy_correct_l914_91467


namespace ten_sided_polygon_diagonals_l914_91494

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem ten_sided_polygon_diagonals :
  number_of_diagonals 10 = 35 :=
by sorry

end ten_sided_polygon_diagonals_l914_91494


namespace solve_system_of_equations_solve_system_of_inequalities_l914_91423

-- Proof for the system of equations
theorem solve_system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 32) 
  (h2 : 2 * x - y = 0) :
  x = 8 ∧ y = 16 :=
by
  sorry

-- Proof for the system of inequalities
theorem solve_system_of_inequalities (x : ℝ)
  (h3 : 3 * x - 1 < 5 - 2 * x)
  (h4 : 5 * x + 1 ≥ 2 * x + 3) :
  (2 / 3 : ℝ) ≤ x ∧ x < (6 / 5 : ℝ) :=
by
  sorry

end solve_system_of_equations_solve_system_of_inequalities_l914_91423


namespace angle_C_of_triangle_l914_91413

theorem angle_C_of_triangle (A B C : ℝ) (h1 : A + B = 110) (h2 : A + B + C = 180) : C = 70 := 
by
  sorry

end angle_C_of_triangle_l914_91413


namespace max_value_ineq_l914_91406

theorem max_value_ineq (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : a + b + c = 1) :
  (a + 3 * b + 5 * c) * (a + b / 3 + c / 5) ≤ 9 / 5 :=
sorry

end max_value_ineq_l914_91406


namespace find_b_of_parabola_axis_of_symmetry_l914_91442

theorem find_b_of_parabola_axis_of_symmetry (b : ℝ) :
  (∀ (x : ℝ), (x = 1) ↔ (x = - (b / (2 * 2))) ) → b = 4 :=
by
  intro h
  sorry

end find_b_of_parabola_axis_of_symmetry_l914_91442


namespace find_b_value_l914_91470

theorem find_b_value :
  (∀ x : ℝ, (x < 0 ∨ x > 4) → -x^2 + 4*x - 4 < 0) ↔ b = 4 := by
sorry

end find_b_value_l914_91470


namespace correct_order_of_operations_l914_91456

def order_of_operations (e : String) : String :=
  if e = "38 * 50 - 25 / 5" then
    "multiplication, division, subtraction"
  else
    "unknown"

theorem correct_order_of_operations :
  order_of_operations "38 * 50 - 25 / 5" = "multiplication, division, subtraction" :=
by
  sorry

end correct_order_of_operations_l914_91456


namespace normal_vector_to_line_l914_91414

theorem normal_vector_to_line : 
  ∀ (x y : ℝ), x - 3 * y + 6 = 0 → (1, -3) = (1, -3) :=
by
  intros x y h_line
  sorry

end normal_vector_to_line_l914_91414


namespace line_equation_passing_through_P_and_equal_intercepts_l914_91422

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition: line passes through point P(1, 3)
def passes_through_P (P : Point) (line_eq : ℝ → ℝ → ℝ) : Prop :=
  line_eq 1 3 = 0

-- Define the condition: equal intercepts on the x-axis and y-axis
def has_equal_intercepts (line_eq : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ (∀ x y, line_eq x y = 0 ↔ x / a + y / a = 1)

-- Define the specific lines x + y - 4 = 0 and 3x - y = 0
def specific_line1 (x y : ℝ) : ℝ := x + y - 4
def specific_line2 (x y : ℝ) : ℝ := 3 * x - y

-- Define the point P(1, 3)
def P := Point.mk 1 3

theorem line_equation_passing_through_P_and_equal_intercepts :
  (passes_through_P P specific_line1 ∧ has_equal_intercepts specific_line1) ∨
  (passes_through_P P specific_line2 ∧ has_equal_intercepts specific_line2) :=
by
  sorry

end line_equation_passing_through_P_and_equal_intercepts_l914_91422


namespace sum_of_digits_is_2640_l914_91429

theorem sum_of_digits_is_2640 (x : ℕ) (h_cond : (1 + 3 + 4 + 6 + x) * (Nat.factorial 5) = 2640) : x = 8 := by
  sorry

end sum_of_digits_is_2640_l914_91429


namespace likelihood_of_white_crows_at_birch_unchanged_l914_91499

theorem likelihood_of_white_crows_at_birch_unchanged 
  (a b c d : ℕ) 
  (h1 : a + b = 50) 
  (h2 : c + d = 50) 
  (h3 : b ≥ a) 
  (h4 : d ≥ c - 1) : 
  (bd + ac + a + b : ℝ) / 2550 > (bc + ad : ℝ) / 2550 := by 
  sorry

end likelihood_of_white_crows_at_birch_unchanged_l914_91499


namespace geometric_mean_l914_91412

theorem geometric_mean (a b c : ℝ) (h1 : a = 5 + 2 * Real.sqrt 6) (h2 : c = 5 - 2 * Real.sqrt 6) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h6 : b^2 = a * c) : b = 1 :=
sorry

end geometric_mean_l914_91412


namespace naturals_less_than_10_l914_91491

theorem naturals_less_than_10 :
  {n : ℕ | n < 10} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end naturals_less_than_10_l914_91491


namespace Alchemerion_is_3_times_older_than_his_son_l914_91415

-- Definitions of Alchemerion's age, his father's age and the sum condition
def Alchemerion_age : ℕ := 360
def Father_age (A : ℕ) := 2 * A + 40
def age_sum (A S F : ℕ) := A + S + F

-- Main theorem statement
theorem Alchemerion_is_3_times_older_than_his_son (S : ℕ) (h1 : Alchemerion_age = 360)
    (h2 : Father_age Alchemerion_age = 2 * Alchemerion_age + 40)
    (h3 : age_sum Alchemerion_age S (Father_age Alchemerion_age) = 1240) :
    Alchemerion_age / S = 3 :=
sorry

end Alchemerion_is_3_times_older_than_his_son_l914_91415


namespace prime_sum_and_difference_l914_91448

theorem prime_sum_and_difference (m n p : ℕ) (hmprime : Nat.Prime m) (hnprime : Nat.Prime n) (hpprime: Nat.Prime p)
  (h1: m > n)
  (h2: n > p)
  (h3 : m + n + p = 74) 
  (h4 : m - n - p = 44) : 
  m = 59 ∧ n = 13 ∧ p = 2 :=
by
  sorry

end prime_sum_and_difference_l914_91448


namespace distance_between_points_l914_91444

theorem distance_between_points : ∀ (A B : ℤ), A = 5 → B = -3 → |A - B| = 8 :=
by
  intros A B hA hB
  rw [hA, hB]
  norm_num

end distance_between_points_l914_91444


namespace integer_values_count_l914_91435

theorem integer_values_count (x : ℕ) (h1 : 5 < Real.sqrt x) (h2 : Real.sqrt x < 6) : 
  ∃ count : ℕ, count = 10 := 
by 
  sorry

end integer_values_count_l914_91435


namespace find_valid_m_l914_91450

noncomputable def g (m x : ℝ) : ℝ := (3 * x + 4) / (m * x - 3)

theorem find_valid_m (m : ℝ) : (∀ x, ∃ y, g m x = y ∧ g m y = x) ↔ (m ∈ Set.Iio (-9 / 4) ∪ Set.Ioi (-9 / 4)) :=
by
  sorry

end find_valid_m_l914_91450


namespace perpendicular_vectors_m_eq_half_l914_91447

theorem perpendicular_vectors_m_eq_half (m : ℝ) (a b : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (-1, m)) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : m = 1 / 2 :=
sorry

end perpendicular_vectors_m_eq_half_l914_91447


namespace tom_remaining_balloons_l914_91460

def original_balloons : ℕ := 30
def given_balloons : ℕ := 16
def remaining_balloons (original_balloons given_balloons : ℕ) : ℕ := original_balloons - given_balloons

theorem tom_remaining_balloons : remaining_balloons original_balloons given_balloons = 14 :=
by
  -- proof omitted for clarity
  sorry

end tom_remaining_balloons_l914_91460


namespace last_number_of_ratio_l914_91495

theorem last_number_of_ratio (A B C : ℕ) (h1 : 5 * B = A) (h2 : 4 * B = C) (h3 : A + B + C = 1000) : C = 400 :=
by
  sorry

end last_number_of_ratio_l914_91495


namespace number_of_sheets_l914_91426

theorem number_of_sheets (S E : ℕ) 
  (h1 : S - E = 40)
  (h2 : 5 * E = S) : 
  S = 50 := by 
  sorry

end number_of_sheets_l914_91426


namespace find_first_offset_l914_91493

theorem find_first_offset 
  (area : ℝ) (diagonal : ℝ) (offset2 : ℝ) (offset1 : ℝ) 
  (h_area : area = 210) 
  (h_diagonal : diagonal = 28)
  (h_offset2 : offset2 = 6) :
  offset1 = 9 :=
by
  sorry

end find_first_offset_l914_91493


namespace functions_are_same_l914_91431

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem functions_are_same : ∀ x : ℝ, f x = g x := by
  sorry

end functions_are_same_l914_91431


namespace polynomial_binomial_square_l914_91411

theorem polynomial_binomial_square (b : ℝ) : 
  (∃ c : ℝ, (3*X + c)^2 = 9*X^2 - 24*X + b) → b = 16 :=
by
  sorry

end polynomial_binomial_square_l914_91411


namespace K_set_I_K_set_III_K_set_IV_K_set_V_l914_91457

-- Definitions for the problem conditions
def K (x y z : ℤ) : ℤ :=
  (x + 2 * y + 3 * z) * (2 * x - y - z) * (y + 2 * z + 3 * x) +
  (y + 2 * z + 3 * x) * (2 * y - z - x) * (z + 2 * x + 3 * y) +
  (z + 2 * x + 3 * y) * (2 * z - x - y) * (x + 2 * y + 3 * z)

-- The equivalent form as a product of terms
def K_equiv (x y z : ℤ) : ℤ :=
  (y + z - 2 * x) * (z + x - 2 * y) * (x + y - 2 * z)

-- Proof statements for each set of numbers
theorem K_set_I : K 1 4 9 = K_equiv 1 4 9 := by
  sorry

theorem K_set_III : K 4 9 1 = K_equiv 4 9 1 := by
  sorry

theorem K_set_IV : K 1 8 11 = K_equiv 1 8 11 := by
  sorry

theorem K_set_V : K 5 8 (-2) = K_equiv 5 8 (-2) := by
  sorry

end K_set_I_K_set_III_K_set_IV_K_set_V_l914_91457


namespace part1_part2_l914_91490

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | 4 ≤ x ∧ x < 8 }
def B : Set ℝ := { x | 3 < x ∧ x < 7 }

theorem part1 :
  (A ∩ B = { x | 4 ≤ x ∧ x < 7 }) ∧
  ((U \ A) ∪ B = { x | x < 7 ∨ x ≥ 8 }) :=
by
  sorry
  
def C (t : ℝ) : Set ℝ := { x | x < t + 1 }

theorem part2 (t : ℝ) :
  (A ∩ C t = ∅) → (t ≤ 3 ∨ t ≥ 7) :=
by
  sorry

end part1_part2_l914_91490


namespace Marcy_sips_interval_l914_91407

theorem Marcy_sips_interval:
  ∀ (total_volume_ml sip_volume_ml total_time min_per_sip: ℕ),
  total_volume_ml = 2000 →
  sip_volume_ml = 40 →
  total_time = 250 →
  min_per_sip = total_time / (total_volume_ml / sip_volume_ml) →
  min_per_sip = 5 :=
by
  intros total_volume_ml sip_volume_ml total_time min_per_sip hv hs ht hm
  rw [hv, hs, ht] at hm
  simp at hm
  exact hm

end Marcy_sips_interval_l914_91407


namespace astroid_arc_length_l914_91449

theorem astroid_arc_length (a : ℝ) (h_a : a > 0) :
  ∃ l : ℝ, (l = 6 * a) ∧ 
  ((a = 1 → l = 6) ∧ (a = 2/3 → l = 4)) := 
by
  sorry

end astroid_arc_length_l914_91449


namespace ratio_length_breadth_l914_91462

theorem ratio_length_breadth
  (b : ℝ) (A : ℝ) (h_b : b = 11) (h_A : A = 363) :
  (∃ l : ℝ, A = l * b ∧ l / b = 3) :=
by
  sorry

end ratio_length_breadth_l914_91462


namespace total_dogs_is_28_l914_91463

def number_of_boxes : ℕ := 7
def dogs_per_box : ℕ := 4
def total_dogs (boxes : ℕ) (dogs_in_each : ℕ) : ℕ := boxes * dogs_in_each

theorem total_dogs_is_28 : total_dogs number_of_boxes dogs_per_box = 28 :=
by
  sorry

end total_dogs_is_28_l914_91463


namespace number_of_males_in_village_l914_91441

-- Given the total population is 800 and it is divided into four equal groups.
def total_population : ℕ := 800
def num_groups : ℕ := 4

-- Proof statement
theorem number_of_males_in_village : (total_population / num_groups) = 200 := 
by sorry

end number_of_males_in_village_l914_91441


namespace evaluation_expression_l914_91469

theorem evaluation_expression (a b c d : ℝ) 
  (h1 : a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h2 : b = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h3 : c = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h4 : d = -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6) :
  (1/a + 1/b + 1/c + 1/d)^2 = (16 * (11 + 2 * Real.sqrt 30)) / ((11 + 2 * Real.sqrt 30 - 3 * Real.sqrt 6)^2) :=
sorry

end evaluation_expression_l914_91469


namespace soccer_tournament_eq_l914_91466

theorem soccer_tournament_eq (x : ℕ) (h : (x * (x - 1)) / 2 = 28) : (1 / 2 : ℚ) * x * (x - 1) = 28 := by
  sorry

end soccer_tournament_eq_l914_91466


namespace anand_income_l914_91464

theorem anand_income (x y : ℕ)
  (income_A : ℕ := 5 * x)
  (income_B : ℕ := 4 * x)
  (expenditure_A : ℕ := 3 * y)
  (expenditure_B : ℕ := 2 * y)
  (savings_A : ℕ := 800)
  (savings_B : ℕ := 800)
  (hA : income_A - expenditure_A = savings_A)
  (hB : income_B - expenditure_B = savings_B) :
  income_A = 2000 := by
  sorry

end anand_income_l914_91464


namespace find_adult_ticket_cost_l914_91474

noncomputable def adult_ticket_cost (A : ℝ) : Prop :=
  let num_adults := 152
  let num_children := num_adults / 2
  let children_ticket_cost := 2.50
  let total_receipts := 1026
  total_receipts = num_adults * A + num_children * children_ticket_cost

theorem find_adult_ticket_cost : adult_ticket_cost 5.50 :=
by
  sorry

end find_adult_ticket_cost_l914_91474


namespace trapezoid_area_calculation_l914_91458

noncomputable def trapezoid_area : ℝ :=
  let y1 := 20
  let y2 := 10
  let x1 := y1 / 2
  let x2 := y2 / 2
  let base1 := x1
  let base2 := x2
  let height := y1 - y2
  (base1 + base2) * height / 2

theorem trapezoid_area_calculation :
  let y1 := 20
  let y2 := 10
  let x1 := y1 / 2
  let x2 := y2 / 2
  let base1 := x1
  let base2 := x2
  let height := y1 - y2
  (base1 + base2) * height / 2 = 75 := 
by
  -- Validation of the translation to Lean 4. Proof steps are omitted.
  sorry

end trapezoid_area_calculation_l914_91458


namespace upstream_distance_18_l914_91468

theorem upstream_distance_18 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (still_water_speed : ℝ) : 
  upstream_distance = 18 :=
by
  have v := (downstream_distance / downstream_time) - still_water_speed
  have upstream_distance := (still_water_speed - v) * upstream_time
  sorry

end upstream_distance_18_l914_91468


namespace value_of_f_nine_halves_l914_91486

noncomputable def f : ℝ → ℝ := sorry  -- Define f with noncomputable since it's not explicitly given

axiom even_function (x : ℝ) : f x = f (-x)  -- Define the even function property
axiom not_identically_zero : ∃ x : ℝ, f x ≠ 0 -- Define the property that f is not identically zero
axiom functional_equation (x : ℝ) : x * f (x + 1) = (x + 1) * f x -- Define the given functional equation

theorem value_of_f_nine_halves : f (9 / 2) = 0 := by
  sorry

end value_of_f_nine_halves_l914_91486


namespace radio_show_play_song_duration_l914_91420

theorem radio_show_play_song_duration :
  ∀ (total_show_time talking_time ad_break_time : ℕ),
  total_show_time = 180 →
  talking_time = 3 * 10 →
  ad_break_time = 5 * 5 →
  total_show_time - (talking_time + ad_break_time) = 125 :=
by
  intros total_show_time talking_time ad_break_time h1 h2 h3
  sorry

end radio_show_play_song_duration_l914_91420


namespace discount_is_20_percent_l914_91471

noncomputable def discount_percentage 
  (puppy_cost : ℝ := 20.0)
  (dog_food_cost : ℝ := 20.0)
  (treat_cost : ℝ := 2.5)
  (num_treats : ℕ := 2)
  (toy_cost : ℝ := 15.0)
  (crate_cost : ℝ := 20.0)
  (bed_cost : ℝ := 20.0)
  (collar_leash_cost : ℝ := 15.0)
  (total_spent : ℝ := 96.0) : ℝ := 
  let total_cost_before_discount := dog_food_cost + (num_treats * treat_cost) + toy_cost + crate_cost + bed_cost + collar_leash_cost
  let spend_at_store := total_spent - puppy_cost
  let discount_amount := total_cost_before_discount - spend_at_store
  (discount_amount / total_cost_before_discount) * 100

theorem discount_is_20_percent : discount_percentage = 20 := sorry

end discount_is_20_percent_l914_91471


namespace value_of_y_l914_91479

theorem value_of_y :
  ∃ y : ℝ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end value_of_y_l914_91479


namespace initially_calculated_average_l914_91401

open List

theorem initially_calculated_average (numbers : List ℝ) (h_len : numbers.length = 10) 
  (h_wrong_reading : ∃ (n : ℝ), n ∈ numbers ∧ n ≠ 26 ∧ (numbers.erase n).sum + 26 = numbers.sum - 36 + 26) 
  (h_correct_avg : numbers.sum / 10 = 16) : 
  ((numbers.sum - 10) / 10 = 15) := 
sorry

end initially_calculated_average_l914_91401


namespace total_games_played_l914_91436

def games_lost : ℕ := 4
def games_won : ℕ := 8

theorem total_games_played : games_lost + games_won = 12 :=
by
  -- Proof is omitted
  sorry

end total_games_played_l914_91436


namespace partial_fraction_product_l914_91497

theorem partial_fraction_product (A B C : ℚ)
  (h_eq : ∀ x, (x^2 - 13) / ((x-2) * (x+2) * (x-3)) = A / (x-2) + B / (x+2) + C / (x-3))
  (h_A : A = 9 / 4)
  (h_B : B = -9 / 20)
  (h_C : C = -4 / 5) :
  A * B * C = 81 / 100 := 
by
  sorry

end partial_fraction_product_l914_91497


namespace slope_parallel_to_original_line_l914_91424

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l914_91424


namespace abs_inequality_solution_set_l914_91496

theorem abs_inequality_solution_set (x : ℝ) : |x - 1| > 2 ↔ x > 3 ∨ x < -1 :=
by
  sorry

end abs_inequality_solution_set_l914_91496


namespace find_a_l914_91485

theorem find_a (a : ℝ)
  (hl : ∀ x y : ℝ, ax + 2 * y - a - 2 = 0)
  (hm : ∀ x y : ℝ, 2 * x - y = 0)
  (perpendicular : ∀ x y : ℝ, (2 * - (a / 2)) = -1) : 
  a = 1 := sorry

end find_a_l914_91485


namespace price_per_strawberry_basket_is_9_l914_91475

-- Define the conditions
def strawberry_plants := 5
def tomato_plants := 7
def strawberries_per_plant := 14
def tomatoes_per_plant := 16
def items_per_basket := 7
def price_per_tomato_basket := 6
def total_revenue := 186

-- Define the total number of strawberries and tomatoes harvested
def total_strawberries := strawberry_plants * strawberries_per_plant
def total_tomatoes := tomato_plants * tomatoes_per_plant

-- Define the number of baskets of strawberries and tomatoes
def strawberry_baskets := total_strawberries / items_per_basket
def tomato_baskets := total_tomatoes / items_per_basket

-- Define the revenue from tomato baskets
def revenue_tomatoes := tomato_baskets * price_per_tomato_basket

-- Define the revenue from strawberry baskets
def revenue_strawberries := total_revenue - revenue_tomatoes

-- Calculate the price per basket of strawberries (which should be $9)
def price_per_strawberry_basket := revenue_strawberries / strawberry_baskets

theorem price_per_strawberry_basket_is_9 : 
  price_per_strawberry_basket = 9 := by
    sorry

end price_per_strawberry_basket_is_9_l914_91475


namespace solve_for_x_l914_91452

theorem solve_for_x (x : ℝ) (h : 144 / 0.144 = 14.4 / x) : x = 0.0144 := 
by
  sorry

end solve_for_x_l914_91452


namespace power_of_2_l914_91427

theorem power_of_2 (n : ℕ) (h1 : n ≥ 1) (h2 : ∃ m : ℕ, m ≥ 1 ∧ (2^n - 1) ∣ (m^2 + 9)) : ∃ k : ℕ, n = 2^k := 
sorry

end power_of_2_l914_91427


namespace total_toys_l914_91453

variable (B H : ℕ)

theorem total_toys (h1 : B = 60) (h2 : H = 9 + (B / 2)) : B + H = 99 := by
  sorry

end total_toys_l914_91453


namespace six_digit_number_condition_l914_91402

theorem six_digit_number_condition :
  ∃ A B : ℕ, 100 ≤ A ∧ A < 1000 ∧ 100 ≤ B ∧ B < 1000 ∧
            1000 * B + A = 6 * (1000 * A + B) :=
by
  sorry

end six_digit_number_condition_l914_91402


namespace days_to_finish_by_b_l914_91416

theorem days_to_finish_by_b (A B C : ℚ) 
  (h1 : A + B + C = 1 / 5) 
  (h2 : A = 1 / 9) 
  (h3 : A + C = 1 / 7) : 
  1 / B = 12.115 :=
by
  sorry

end days_to_finish_by_b_l914_91416


namespace jinho_initial_money_l914_91425

variable (M : ℝ)

theorem jinho_initial_money :
  (M / 2 + 300) + (((M / 2 - 300) / 2) + 400) = M :=
by
  -- This proof is yet to be completed.
  sorry

end jinho_initial_money_l914_91425


namespace Josiah_spent_on_cookies_l914_91478

theorem Josiah_spent_on_cookies :
  let cookies_per_day := 2
  let cost_per_cookie := 16
  let days_in_march := 31
  2 * days_in_march * cost_per_cookie = 992 := 
by
  sorry

end Josiah_spent_on_cookies_l914_91478


namespace sum_distinct_vars_eq_1716_l914_91440

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l914_91440


namespace sphere_surface_area_l914_91439

theorem sphere_surface_area (edge_length : ℝ) (diameter_eq_edge_length : (diameter : ℝ) = edge_length) :
  (edge_length = 2) → (diameter = 2) → (surface_area : ℝ) = 8 * Real.pi :=
by
  sorry

end sphere_surface_area_l914_91439


namespace find_c8_l914_91445

-- Definitions of arithmetic sequences and their products
def arithmetic_seq (a d : ℤ) (n : ℕ) := a + n * d

def c_n (a d1 b d2 : ℤ) (n : ℕ) := arithmetic_seq a d1 n * arithmetic_seq b d2 n

-- Given conditions
variables (a1 d1 a2 d2 : ℤ)
variables (c1 c2 c3 : ℤ)
variables (h1 : c_n a1 d1 a2 d2 1 = 1440)
variables (h2 : c_n a1 d1 a2 d2 2 = 1716)
variables (h3 : c_n a1 d1 a2 d2 3 = 1848)

-- The goal is to prove c_8 = 348
theorem find_c8 : c_n a1 d1 a2 d2 8 = 348 :=
sorry

end find_c8_l914_91445


namespace racetrack_circumference_diff_l914_91482

theorem racetrack_circumference_diff (d_inner d_outer width : ℝ) 
(h1 : d_inner = 55) (h2 : width = 15) (h3 : d_outer = d_inner + 2 * width) : 
  (π * d_outer - π * d_inner) = 30 * π :=
by
  sorry

end racetrack_circumference_diff_l914_91482


namespace arithmetic_sum_2015_l914_91492

-- Definitions based on problem conditions
def a1 : ℤ := -2015
def S (n : ℕ) (d : ℤ) : ℤ := n * a1 + n * (n - 1) / 2 * d
def arithmetic_sequence (n : ℕ) (d : ℤ) : ℤ := a1 + (n - 1) * d

-- Proof problem
theorem arithmetic_sum_2015 (d : ℤ) :
  2 * S 6 d - 3 * S 4 d = 24 →
  S 2015 d = -2015 :=
by
  sorry

end arithmetic_sum_2015_l914_91492


namespace potato_difference_l914_91418

def x := 8 * 13
def k := (67 - 13) / 2
def z := 20 * k
def d := z - x

theorem potato_difference : d = 436 :=
by
  sorry

end potato_difference_l914_91418


namespace perpendicular_line_x_intercept_l914_91446

theorem perpendicular_line_x_intercept :
  (∃ x : ℝ, ∃ y : ℝ, 4 * x + 5 * y = 10) →
  (∃ y : ℝ, y = (5/4) * x - 3) →
  (∃ x : ℝ, y = 0) →
  x = 12 / 5 :=
by
  sorry

end perpendicular_line_x_intercept_l914_91446


namespace abc_divisible_by_6_l914_91403

theorem abc_divisible_by_6 (a b c : ℤ) (h : 18 ∣ (a^3 + b^3 + c^3)) : 6 ∣ (a * b * c) :=
by
  sorry

end abc_divisible_by_6_l914_91403


namespace solve_fractional_equation_l914_91477

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 3) : (2 / (x - 3) = 3 / x) → x = 9 :=
by
  sorry

end solve_fractional_equation_l914_91477


namespace infinite_solutions_distinct_natural_numbers_l914_91404

theorem infinite_solutions_distinct_natural_numbers :
  ∃ (x y z : ℕ), (x ≠ y) ∧ (x ≠ z) ∧ (y ≠ z) ∧ (x ^ 2015 + y ^ 2015 = z ^ 2016) :=
by
  sorry

end infinite_solutions_distinct_natural_numbers_l914_91404


namespace perpendicular_line_eq_l914_91451

theorem perpendicular_line_eq (x y : ℝ) :
  (∃ (p : ℝ × ℝ), p = (-2, 3) ∧ 
    ∀ y₀ x₀, 3 * x - y = 6 ∧ y₀ = 3 ∧ x₀ = -2 → y = -1 / 3 * x + 7 / 3) :=
sorry

end perpendicular_line_eq_l914_91451


namespace football_team_lineup_ways_l914_91428

theorem football_team_lineup_ways :
  let members := 12
  let offensive_lineman_options := 4
  let remaining_after_linemen := members - offensive_lineman_options
  let quarterback_options := remaining_after_linemen
  let remaining_after_qb := remaining_after_linemen - 1
  let wide_receiver_options := remaining_after_qb
  let remaining_after_wr := remaining_after_qb - 1
  let tight_end_options := remaining_after_wr
  let lineup_ways := offensive_lineman_options * quarterback_options * wide_receiver_options * tight_end_options
  lineup_ways = 3960 :=
by
  sorry

end football_team_lineup_ways_l914_91428


namespace meaningful_expression_range_l914_91465

theorem meaningful_expression_range (a : ℝ) : (a + 1 ≥ 0) ∧ (a ≠ 2) ↔ (a ≥ -1) ∧ (a ≠ 2) :=
by
  sorry

end meaningful_expression_range_l914_91465


namespace stable_performance_l914_91421

theorem stable_performance 
  (X_A_mean : ℝ) (X_B_mean : ℝ) (S_A_var : ℝ) (S_B_var : ℝ)
  (h1 : X_A_mean = 82) (h2 : X_B_mean = 82)
  (h3 : S_A_var = 245) (h4 : S_B_var = 190) : S_B_var < S_A_var :=
by {
  sorry
}

end stable_performance_l914_91421


namespace div_by_eleven_l914_91455

theorem div_by_eleven (n : ℤ) : 11 ∣ ((n + 11)^2 - n^2) :=
by
  sorry

end div_by_eleven_l914_91455


namespace functional_relationship_profit_maximized_at_sufficient_profit_range_verified_l914_91473

noncomputable def daily_sales_profit (x : ℝ) : ℝ :=
  -5 * x^2 + 800 * x - 27500

def profit_maximized (x : ℝ) : Prop :=
  daily_sales_profit x = -5 * (80 - x)^2 + 4500

def sufficient_profit_range (x : ℝ) : Prop :=
  daily_sales_profit x >= 4000 ∧ (x - 50) * (500 - 5 * x) <= 7000

theorem functional_relationship (x : ℝ) : daily_sales_profit x = -5 * x^2 + 800 * x - 27500 :=
  sorry

theorem profit_maximized_at (x : ℝ) : profit_maximized x → x = 80 ∧ daily_sales_profit x = 4500 :=
  sorry

theorem sufficient_profit_range_verified (x : ℝ) : sufficient_profit_range x → 82 ≤ x ∧ x ≤ 90 :=
  sorry

end functional_relationship_profit_maximized_at_sufficient_profit_range_verified_l914_91473


namespace quadratic_root_conditions_l914_91408

theorem quadratic_root_conditions : ∃ p q : ℝ, (p - 1)^2 - 4 * q > 0 ∧ (p + 1)^2 - 4 * q > 0 ∧ p^2 - 4 * q < 0 := 
sorry

end quadratic_root_conditions_l914_91408


namespace tangent_line_parabola_l914_91484

theorem tangent_line_parabola (d : ℝ) :
  (∃ (f g : ℝ → ℝ), (∀ x y, y = f x ↔ y = 3 * x + d) ∧ (∀ x y, y = g x ↔ y ^ 2 = 12 * x)
  ∧ (∀ x y, y = f x ∧ y = g x → y = 3 * x + d ∧ y ^ 2 = 12 * x )) →
  d = 1 :=
sorry

end tangent_line_parabola_l914_91484


namespace coins_donated_l914_91459

theorem coins_donated (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (coins_left : ℕ) : 
  pennies = 42 ∧ nickels = 36 ∧ dimes = 15 ∧ coins_left = 27 → (pennies + nickels + dimes - coins_left) = 66 :=
by
  intros h
  sorry

end coins_donated_l914_91459


namespace max_value_ab_bc_cd_l914_91443

theorem max_value_ab_bc_cd (a b c d : ℝ) (h1 : 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) (h4: 0 ≤ d) (h_sum : a + b + c + d = 200) :
  ab + bc + cd ≤ 2500 :=
by
  sorry

end max_value_ab_bc_cd_l914_91443


namespace football_club_balance_l914_91461

def initial_balance : ℕ := 100
def income := 2 * 10
def cost := 4 * 15
def final_balance := initial_balance + income - cost

theorem football_club_balance : final_balance = 60 := by
  sorry

end football_club_balance_l914_91461


namespace polynomial_at_most_one_integer_root_l914_91432

theorem polynomial_at_most_one_integer_root (n : ℤ) :
  ∀ x1 x2 : ℤ, (x1 ≠ x2) → 
  (x1 ^ 4 - 1993 * x1 ^ 3 + (1993 + n) * x1 ^ 2 - 11 * x1 + n = 0) → 
  (x2 ^ 4 - 1993 * x2 ^ 3 + (1993 + n) * x2 ^ 2 - 11 * x2 + n = 0) → 
  false :=
by
  sorry

end polynomial_at_most_one_integer_root_l914_91432


namespace probability_same_color_probability_different_color_l914_91487

def count_combinations {α : Type*} (s : Finset α) (k : ℕ) : ℕ :=
  Nat.choose s.card k

noncomputable def count_ways_same_color : ℕ :=
  (count_combinations (Finset.range 3) 2) * 2

noncomputable def count_ways_diff_color : ℕ :=
  (Finset.range 3).card * (Finset.range 3).card

noncomputable def total_ways : ℕ :=
  count_combinations (Finset.range 6) 2

noncomputable def prob_same_color : ℚ :=
  count_ways_same_color / total_ways

noncomputable def prob_diff_color : ℚ :=
  count_ways_diff_color / total_ways

theorem probability_same_color :
  prob_same_color = 2 / 5 := by
  sorry

theorem probability_different_color :
  prob_diff_color = 3 / 5 := by
  sorry

end probability_same_color_probability_different_color_l914_91487


namespace value_of_f_is_negative_l914_91476

theorem value_of_f_is_negative {a b c : ℝ} (h1 : a + b < 0) (h2 : b + c < 0) (h3 : c + a < 0) :
  2 * a ^ 3 + 4 * a + 2 * b ^ 3 + 4 * b + 2 * c ^ 3 + 4 * c < 0 := by
sorry

end value_of_f_is_negative_l914_91476


namespace sum_coords_B_l914_91483

def point (A B : ℝ × ℝ) : Prop :=
A = (0, 0) ∧ B.snd = 5 ∧ (B.snd - A.snd) / (B.fst - A.fst) = 3 / 4

theorem sum_coords_B (x : ℝ) :
  point (0, 0) (x, 5) → x + 5 = 35 / 3 :=
by
  intro h
  sorry

end sum_coords_B_l914_91483


namespace jenny_original_amount_half_l914_91434

-- Definitions based on conditions
def original_amount (x : ℝ) := x
def spent_fraction := 3 / 7
def left_after_spending (x : ℝ) := x * (1 - spent_fraction)

theorem jenny_original_amount_half (x : ℝ) (h : left_after_spending x = 24) : original_amount x / 2 = 21 :=
by
  -- Indicate the intention to prove the statement by sorry
  sorry

end jenny_original_amount_half_l914_91434


namespace geometric_sequence_term_l914_91454

theorem geometric_sequence_term :
  ∃ (a_n : ℕ → ℕ),
    -- common ratio condition
    (∀ n, a_n (n + 1) = 2 * a_n n) ∧
    -- sum of first 4 terms condition
    (a_n 1 + a_n 2 + a_n 3 + a_n 4 = 60) ∧
    -- conclusion: value of the third term
    (a_n 3 = 16) :=
by
  sorry

end geometric_sequence_term_l914_91454


namespace solve_for_x_l914_91480

theorem solve_for_x (x : ℤ) (h : 3 * x + 20 = (1/3 : ℚ) * (7 * x + 60)) : x = 0 :=
sorry

end solve_for_x_l914_91480


namespace min_value_of_2x_plus_y_l914_91405

theorem min_value_of_2x_plus_y {x y : ℝ} (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 1) + 8 / y = 2) : 2 * x + y ≥ 7 :=
sorry

end min_value_of_2x_plus_y_l914_91405


namespace no_real_solution_l914_91430

theorem no_real_solution (x : ℝ) : 
  x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 → 
  ¬ (
    (1 / ((x - 1) * (x - 3))) + (1 / ((x - 3) * (x - 5))) + (1 / ((x - 5) * (x - 7))) = 1 / 4
  ) :=
by sorry

end no_real_solution_l914_91430


namespace jim_less_than_anthony_l914_91409

-- Definitions for the conditions
def scott_shoes : ℕ := 7

def anthony_shoes : ℕ := 3 * scott_shoes

def jim_shoes : ℕ := anthony_shoes - 2

-- Lean statement to prove the problem
theorem jim_less_than_anthony : anthony_shoes - jim_shoes = 2 := by
  sorry

end jim_less_than_anthony_l914_91409


namespace number_subtracted_from_10000_l914_91437

theorem number_subtracted_from_10000 (x : ℕ) (h : 10000 - x = 9001) : x = 999 := by
  sorry

end number_subtracted_from_10000_l914_91437
