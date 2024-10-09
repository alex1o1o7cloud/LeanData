import Mathlib

namespace pool_ratio_three_to_one_l276_27615

theorem pool_ratio_three_to_one (P : ℕ) (B B' : ℕ) (k : ℕ) :
  (P = 5 * B + 2) → (k * P = 5 * B' + 1) → k = 3 :=
by
  intros h1 h2
  sorry

end pool_ratio_three_to_one_l276_27615


namespace consecutive_integers_equation_l276_27696

theorem consecutive_integers_equation
  (X Y : ℕ)
  (h_consecutive : Y = X + 1)
  (h_equation : 2 * X^2 + 4 * X + 5 * Y + 3 = (X + Y)^2 + 9 * (X + Y) + 4) :
  X + Y = 15 := by
  sorry

end consecutive_integers_equation_l276_27696


namespace a_2n_perfect_square_l276_27613

-- Define the sequence a_n following the described recurrence relation.
def a (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 4
  else a (n-1) + a (n-3) + a (n-4)

-- Define the main theorem to prove
theorem a_2n_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k * k := by
  sorry

end a_2n_perfect_square_l276_27613


namespace find_value_l276_27636

theorem find_value (
  a b c d e f : ℝ) 
  (h1 : a * b * c = 65) 
  (h2 : b * c * d = 65) 
  (h3 : c * d * e = 1000) 
  (h4 : (a * f) / (c * d) = 0.25) :
  d * e * f = 250 := 
sorry

end find_value_l276_27636


namespace max_value_of_f_l276_27600

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x + 15 * Real.cos x

theorem max_value_of_f : ∃ x : ℝ, f x = 17 :=
sorry

end max_value_of_f_l276_27600


namespace ranking_of_anna_bella_carol_l276_27678

-- Define three people and their scores
variables (Anna Bella Carol : ℕ)

-- Define conditions based on problem statements
axiom Anna_not_highest : ∃ x : ℕ, x > Anna
axiom Bella_not_lowest : ∃ x : ℕ, x < Bella
axiom Bella_higher_than_Carol : Bella > Carol

-- The theorem to be proven
theorem ranking_of_anna_bella_carol (h : Anna < Bella ∧ Carol < Anna) :
  (Bella > Anna ∧ Anna > Carol) :=
by sorry

end ranking_of_anna_bella_carol_l276_27678


namespace symmetric_circle_l276_27603

theorem symmetric_circle :
  ∀ (C D : Type) (hD : ∀ x y : ℝ, (x + 2)^2 + (y - 6)^2 = 1) (hline : ∀ x y : ℝ, x - y + 5 = 0), 
  (∀ x y : ℝ, (x - 1)^2 + (y - 3)^2 = 1) := 
by sorry

end symmetric_circle_l276_27603


namespace fraction_of_painted_surface_area_l276_27606

def total_surface_area_of_smaller_prisms : ℕ := 
  let num_smaller_prisms := 27
  let num_square_faces := num_smaller_prisms * 3
  let num_triangular_faces := num_smaller_prisms * 2
  num_square_faces + num_triangular_faces

def painted_surface_area_of_larger_prism : ℕ :=
  let painted_square_faces := 3 * 9
  let painted_triangular_faces := 2 * 9
  painted_square_faces + painted_triangular_faces

theorem fraction_of_painted_surface_area : 
  (painted_surface_area_of_larger_prism : ℚ) / (total_surface_area_of_smaller_prisms : ℚ) = 1 / 3 :=
by sorry

end fraction_of_painted_surface_area_l276_27606


namespace time_to_fill_partial_bucket_l276_27676

-- Definitions for the conditions
def time_to_fill_full_bucket : ℝ := 135
def r := 2 / 3

-- The time to fill 2/3 of the bucket should be proven as 90
theorem time_to_fill_partial_bucket : time_to_fill_full_bucket * r = 90 := 
by 
  -- Prove that 90 is the correct time to fill two-thirds of the bucket
  sorry

end time_to_fill_partial_bucket_l276_27676


namespace circle_graph_to_bar_graph_correct_l276_27665

theorem circle_graph_to_bar_graph_correct :
  ∀ (white black gray blue : ℚ) (w_proportion b_proportion g_proportion blu_proportion : ℚ),
    white = 1/2 →
    black = 1/4 →
    gray = 1/8 →
    blue = 1/8 →
    w_proportion = 1/2 →
    b_proportion = 1/4 →
    g_proportion = 1/8 →
    blu_proportion = 1/8 →
    white = w_proportion ∧ black = b_proportion ∧ gray = g_proportion ∧ blue = blu_proportion :=
by
sorry

end circle_graph_to_bar_graph_correct_l276_27665


namespace johns_weekly_earnings_percentage_increase_l276_27694

theorem johns_weekly_earnings_percentage_increase (initial final : ℝ) :
  initial = 30 →
  final = 50 →
  ((final - initial) / initial) * 100 = 66.67 :=
by
  intros h_initial h_final
  rw [h_initial, h_final]
  norm_num
  sorry

end johns_weekly_earnings_percentage_increase_l276_27694


namespace side_of_larger_square_l276_27626

theorem side_of_larger_square (s S : ℕ) (h₁ : s = 5) (h₂ : S^2 = 4 * s^2) : S = 10 := 
by sorry

end side_of_larger_square_l276_27626


namespace parallel_lines_slope_l276_27635

theorem parallel_lines_slope (a : ℝ) : 
  let m1 := - (a / 2)
  let m2 := 3
  ax + 2 * y + 2 = 0 ∧ 3 * x - y - 2 = 0 → m1 = m2 → a = -6 := 
by
  intros
  sorry

end parallel_lines_slope_l276_27635


namespace arc_length_TQ_l276_27647

-- Definitions from the conditions
def center (O : Type) : Prop := true
def inscribedAngle (T I Q : Type) (angle : ℝ) := angle = 45
def radius (T : Type) (len : ℝ) := len = 12

-- Theorem to be proved
theorem arc_length_TQ (O T I Q : Type) (r : ℝ) (angle : ℝ) 
  (h_center : center O) 
  (h_angle : inscribedAngle T I Q angle)
  (h_radius : radius T r) :
  ∃ l : ℝ, l = 6 * Real.pi := 
sorry

end arc_length_TQ_l276_27647


namespace part1_part2_l276_27614

noncomputable def f (x a : ℝ) : ℝ := |x + a|
noncomputable def g (x : ℝ) : ℝ := |x + 3| - x

theorem part1 (x : ℝ) : f x 1 < g x → x < 2 :=
sorry

theorem part2 (a : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x a < g x) → -2 < a ∧ a < 2 :=
sorry

end part1_part2_l276_27614


namespace equivalent_product_lists_l276_27672

-- Definitions of the value assigned to each letter.
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1
  | 'B' => 2
  | 'C' => 3
  | 'D' => 4
  | 'E' => 5
  | 'F' => 6
  | 'G' => 7
  | 'H' => 8
  | 'I' => 9
  | 'J' => 10
  | 'K' => 11
  | 'L' => 12
  | 'M' => 13
  | 'N' => 14
  | 'O' => 15
  | 'P' => 16
  | 'Q' => 17
  | 'R' => 18
  | 'S' => 19
  | 'T' => 20
  | 'U' => 21
  | 'V' => 22
  | 'W' => 23
  | 'X' => 24
  | 'Y' => 25
  | 'Z' => 26
  | _ => 0  -- We only care about uppercase letters A-Z

def list_product (l : List Char) : ℕ :=
  l.foldl (λ acc c => acc * (letter_value c)) 1

-- Given the list MNOP with their products equals letter values.
def MNOP := ['M', 'N', 'O', 'P']
def BJUZ := ['B', 'J', 'U', 'Z']

-- Lean statement to assert the equivalence of the products.
theorem equivalent_product_lists :
  list_product MNOP = list_product BJUZ :=
by
  sorry

end equivalent_product_lists_l276_27672


namespace total_words_in_poem_l276_27669

theorem total_words_in_poem 
  (stanzas : ℕ) 
  (lines_per_stanza : ℕ) 
  (words_per_line : ℕ) 
  (h_stanzas : stanzas = 20) 
  (h_lines_per_stanza : lines_per_stanza = 10) 
  (h_words_per_line : words_per_line = 8) : 
  stanzas * lines_per_stanza * words_per_line = 1600 := 
sorry

end total_words_in_poem_l276_27669


namespace cost_of_three_tshirts_l276_27663

-- Defining the conditions
def saving_per_tshirt : ℝ := 5.50
def full_price_per_tshirt : ℝ := 16.50
def number_of_tshirts : ℕ := 3
def number_of_paid_tshirts : ℕ := 2

-- Statement of the problem
theorem cost_of_three_tshirts :
  (number_of_paid_tshirts * full_price_per_tshirt) = 33 := 
by
  -- Proof steps go here (using sorry as a placeholder)
  sorry

end cost_of_three_tshirts_l276_27663


namespace A_beats_B_by_7_seconds_l276_27619

noncomputable def speed_A : ℝ := 200 / 33
noncomputable def distance_A : ℝ := 200
noncomputable def time_A : ℝ := 33

noncomputable def distance_B : ℝ := 200
noncomputable def distance_B_at_time_A : ℝ := 165

-- B's speed is calculated at the moment A finishes the race
noncomputable def speed_B : ℝ := distance_B_at_time_A / time_A
noncomputable def time_B : ℝ := distance_B / speed_B

-- Prove that A beats B by 7 seconds
theorem A_beats_B_by_7_seconds : time_B - time_A = 7 := 
by 
  -- Proof goes here, assume all definitions and variables are correct.
  sorry

end A_beats_B_by_7_seconds_l276_27619


namespace sum_of_diagonals_l276_27699

noncomputable def length_AB : ℝ := 31
noncomputable def length_sides : ℝ := 81

def hexagon_inscribed_in_circle (A B C D E F : Type) : Prop :=
-- Assuming A, B, C, D, E, F are suitable points on a circle
-- Definitions to be added as per detailed proof needs
sorry

theorem sum_of_diagonals (A B C D E F : Type) :
    hexagon_inscribed_in_circle A B C D E F →
    (length_AB + length_sides + length_sides + length_sides + length_sides + length_sides = 384) := 
by
  sorry

end sum_of_diagonals_l276_27699


namespace union_A_B_inter_A_compl_B_range_of_a_l276_27609

-- Define the sets A, B, and C
def A := {x : ℝ | -1 ≤ x ∧ x < 3}
def B := {x : ℝ | 2 * x - 4 ≥ x - 2}
def C (a : ℝ) := {x : ℝ | x ≥ a - 1}

-- Prove A ∪ B = {x | -1 ≤ x}
theorem union_A_B : A ∪ B = {x : ℝ | -1 ≤ x} :=
by sorry

-- Prove A ∩ (complement B) = {x | -1 ≤ x < 2}
theorem inter_A_compl_B : A ∩ (compl B) = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by sorry

-- Prove the range of a given B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 :=
by sorry

end union_A_B_inter_A_compl_B_range_of_a_l276_27609


namespace sum_of_prime_factors_of_143_l276_27683

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l276_27683


namespace profit_percentage_calculation_l276_27633

def selling_price : ℝ := 120
def cost_price : ℝ := 96

theorem profit_percentage_calculation (sp cp : ℝ) (hsp : sp = selling_price) (hcp : cp = cost_price) : 
  ((sp - cp) / cp) * 100 = 25 := 
 by
  sorry

end profit_percentage_calculation_l276_27633


namespace num_zeros_in_expansion_l276_27667

noncomputable def bigNum := (10^11 - 2) ^ 2

theorem num_zeros_in_expansion : ∀ n : ℕ, bigNum = n ↔ (n = 9999999999900000000004) := sorry

end num_zeros_in_expansion_l276_27667


namespace recurring_decimal_to_fraction_l276_27604

theorem recurring_decimal_to_fraction :
  ∃ (frac : ℚ), frac = 1045 / 1998 ∧ 0.5 + (23 / 999) = frac :=
by
  sorry

end recurring_decimal_to_fraction_l276_27604


namespace range_of_a_l276_27675

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) → -1 < a ∧ a < 3 :=
sorry

end range_of_a_l276_27675


namespace elvie_age_l276_27673

variable (E : ℕ) (A : ℕ)

theorem elvie_age (hA : A = 11) (h : E + A + (E * A) = 131) : E = 10 :=
by
  sorry

end elvie_age_l276_27673


namespace triangle_is_equilateral_l276_27625

-- Define a triangle with angles A, B, and C
variables (A B C : ℝ)

-- The conditions of the problem
def log_sin_arithmetic_sequence : Prop :=
  Real.log (Real.sin A) + Real.log (Real.sin C) = 2 * Real.log (Real.sin B)

def angles_arithmetic_sequence : Prop :=
  2 * B = A + C

-- The theorem that the triangle is equilateral given these conditions
theorem triangle_is_equilateral :
  log_sin_arithmetic_sequence A B C → angles_arithmetic_sequence A B C → 
  A = 60 ∧ B = 60 ∧ C = 60 :=
by
  sorry

end triangle_is_equilateral_l276_27625


namespace temperature_difference_l276_27646

def lowest_temp : ℝ := -15
def highest_temp : ℝ := 3

theorem temperature_difference :
  highest_temp - lowest_temp = 18 :=
by
  sorry

end temperature_difference_l276_27646


namespace money_saved_l276_27662

noncomputable def total_savings :=
  let fox_price := 15
  let pony_price := 18
  let num_fox_pairs := 3
  let num_pony_pairs := 2
  let total_discount_rate := 0.22
  let pony_discount_rate := 0.10999999999999996
  let fox_discount_rate := total_discount_rate - pony_discount_rate
  let fox_savings := fox_price * fox_discount_rate * num_fox_pairs
  let pony_savings := pony_price * pony_discount_rate * num_pony_pairs
  fox_savings + pony_savings

theorem money_saved :
  total_savings = 8.91 :=
by
  -- We assume the savings calculations are correct as per the problem statement
  sorry

end money_saved_l276_27662


namespace regular_polygon_sides_l276_27690

theorem regular_polygon_sides (interior_angle exterior_angle : ℕ)
  (h1 : interior_angle = exterior_angle + 60)
  (h2 : interior_angle + exterior_angle = 180) :
  ∃ n : ℕ, n = 6 :=
by
  have ext_angle_eq : exterior_angle = 60 := sorry
  have ext_angles_sum : exterior_angle * 6 = 360 := sorry
  exact ⟨6, by linarith⟩

end regular_polygon_sides_l276_27690


namespace initial_matches_l276_27644

theorem initial_matches (x : ℕ) (h1 : (34 * x + 89) / (x + 1) = 39) : x = 10 := by
  sorry

end initial_matches_l276_27644


namespace complement_of_A_in_U_l276_27616

namespace SetTheory

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {-1, 1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {0} := by
  sorry

end SetTheory

end complement_of_A_in_U_l276_27616


namespace find_x_value_l276_27634

theorem find_x_value (x : ℝ) (a b c : ℝ × ℝ × ℝ) 
  (h_a : a = (1, 1, x)) 
  (h_b : b = (1, 2, 1)) 
  (h_c : c = (1, 1, 1)) 
  (h_cond : (c - a) • (2 • b) = -2) : 
  x = 2 := 
by 
  -- the proof goes here
  sorry

end find_x_value_l276_27634


namespace compute_expression_l276_27691

theorem compute_expression : (88 * 707 - 38 * 707) / 1414 = 25 :=
by
  sorry

end compute_expression_l276_27691


namespace find_solutions_l276_27638

noncomputable
def is_solution (a b c d : ℝ) : Prop :=
  a + b + c = d ∧ (1 / a + 1 / b + 1 / c = 1 / d)

theorem find_solutions (a b c d : ℝ) :
  is_solution a b c d ↔ (c = -a ∧ d = b) ∨ (c = -b ∧ d = a) :=
by
  sorry

end find_solutions_l276_27638


namespace inclination_angle_of_line_3x_sqrt3y_minus1_l276_27664

noncomputable def inclination_angle_of_line (A B C : ℝ) (h : A ≠ 0 ∧ B ≠ 0) : ℝ :=
  let m := -A / B 
  if m = Real.tan (120 * Real.pi / 180) then 120
  else 0 -- This will return 0 if the slope m does not match, for simplifying purposes

theorem inclination_angle_of_line_3x_sqrt3y_minus1 :
  inclination_angle_of_line 3 (Real.sqrt 3) (-1) (by sorry) = 120 := 
sorry

end inclination_angle_of_line_3x_sqrt3y_minus1_l276_27664


namespace max_area_equilateral_triangle_in_rectangle_l276_27686

-- Define the problem parameters
def rect_width : ℝ := 12
def rect_height : ℝ := 15

-- State the theorem to be proved
theorem max_area_equilateral_triangle_in_rectangle 
  (width height : ℝ) (h_width : width = rect_width) (h_height : height = rect_height) :
  ∃ area : ℝ, area = 369 * Real.sqrt 3 - 540 := 
sorry

end max_area_equilateral_triangle_in_rectangle_l276_27686


namespace ratio_Q_P_l276_27623

theorem ratio_Q_P : 
  ∀ (P Q : ℚ), (∀ x : ℚ, x ≠ -3 → x ≠ 0 → x ≠ 5 → 
    (P / (x + 3) + Q / (x * (x - 5)) = (x^2 - 3*x + 12) / (x^3 + x^2 - 15*x))) →
    (Q / P) = 20 / 9 :=
by
  intros P Q h
  sorry

end ratio_Q_P_l276_27623


namespace box_volume_l276_27682

theorem box_volume (x y : ℝ) (hx : 0 < x ∧ x < 6) (hy : 0 < y ∧ y < 8) :
  (16 - 2 * x) * (12 - 2 * y) * y = 192 * y - 32 * y^2 - 24 * x * y + 4 * x * y^2 :=
by
  sorry

end box_volume_l276_27682


namespace smallest_model_length_l276_27629

theorem smallest_model_length (full_size : ℕ) (mid_size_factor smallest_size_factor : ℚ) :
  full_size = 240 →
  mid_size_factor = 1 / 10 →
  smallest_size_factor = 1 / 2 →
  (full_size * mid_size_factor) * smallest_size_factor = 12 :=
by
  intros h_full_size h_mid_size_factor h_smallest_size_factor
  sorry

end smallest_model_length_l276_27629


namespace problem1_problem2_l276_27668

noncomputable def circle_ast (a b : ℕ) : ℕ := sorry

axiom circle_ast_self (a : ℕ) : circle_ast a a = a
axiom circle_ast_zero (a : ℕ) : circle_ast a 0 = 2 * a
axiom circle_ast_add (a b c d : ℕ) : circle_ast a b + circle_ast c d = circle_ast (a + c) (b + d)

theorem problem1 : circle_ast (2 + 3) (0 + 3) = 7 := sorry

theorem problem2 : circle_ast 1024 48 = 2000 := sorry

end problem1_problem2_l276_27668


namespace day_after_7k_days_is_friday_day_before_7k_days_is_thursday_day_after_100_days_is_sunday_l276_27628

-- Definitions based on problem conditions and questions
def day_of_week_after (n : ℤ) (current_day : String) : String :=
  if n % 7 = 0 then current_day else
    if n % 7 = 1 then "Saturday" else
    if n % 7 = 2 then "Sunday" else
    if n % 7 = 3 then "Monday" else
    if n % 7 = 4 then "Tuesday" else
    if n % 7 = 5 then "Wednesday" else
    "Thursday"

def day_of_week_before (n : ℤ) (current_day : String) : String :=
  day_of_week_after (-n) current_day

-- Conditions
def today : String := "Friday"

-- Prove the following
theorem day_after_7k_days_is_friday (k : ℤ) : day_of_week_after (7 * k) today = "Friday" :=
by sorry

theorem day_before_7k_days_is_thursday (k : ℤ) : day_of_week_before (7 * k) today = "Thursday" :=
by sorry

theorem day_after_100_days_is_sunday : day_of_week_after 100 today = "Sunday" :=
by sorry

end day_after_7k_days_is_friday_day_before_7k_days_is_thursday_day_after_100_days_is_sunday_l276_27628


namespace inequality_abc_distinct_positive_l276_27618

theorem inequality_abc_distinct_positive
  (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  (a^2 / b + b^2 / c + c^2 / d + d^2 / a > a + b + c + d) := 
by
  sorry

end inequality_abc_distinct_positive_l276_27618


namespace number_of_students_l276_27642

theorem number_of_students (x : ℕ) (total_cards : ℕ) (h : x * (x - 1) = total_cards) (h_total : total_cards = 182) : x = 14 :=
by
  sorry

end number_of_students_l276_27642


namespace inequality_holds_for_positive_vars_l276_27679

theorem inequality_holds_for_positive_vars (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
    x^2 + y^2 + 1 ≥ x * y + x + y :=
sorry

end inequality_holds_for_positive_vars_l276_27679


namespace find_value_of_p_l276_27651

theorem find_value_of_p (p q : ℚ) (h1 : p + q = 3 / 4)
    (h2 : 45 * p^8 * q^2 = 120 * p^7 * q^3) : p = 6 / 11 :=
by
    sorry

end find_value_of_p_l276_27651


namespace similar_right_triangles_l276_27639

open Real

theorem similar_right_triangles (x : ℝ) (h : ℝ)
  (h₁: 12^2 + 9^2 = (12^2 + 9^2))
  (similarity : (12 / x) = (9 / 6))
  (p : hypotenuse = 12*12) :
  x = 8 ∧ h = 10 := by
  sorry

end similar_right_triangles_l276_27639


namespace probability_of_exactly_three_positives_l276_27680

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_exactly_three_positives :
  let p := 2/5
  let n := 7
  let k := 3
  let positive_prob := p^k
  let negative_prob := (1 - p)^(n - k)
  let binomial_coefficient := choose n k
  binomial_coefficient * positive_prob * negative_prob = 22680/78125 := 
by
  sorry

end probability_of_exactly_three_positives_l276_27680


namespace four_numbers_sum_divisible_by_2016_l276_27661

theorem four_numbers_sum_divisible_by_2016 {x : Fin 65 → ℕ} (h_distinct: Function.Injective x) (h_range: ∀ i, x i ≤ 2016) :
  ∃ a b c d : Fin 65, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ (x a + x b - x c - x d) % 2016 = 0 :=
by
  -- Proof omitted
  sorry

end four_numbers_sum_divisible_by_2016_l276_27661


namespace james_total_toys_l276_27617

-- Definition for the number of toy cars
def numToyCars : ℕ := 20

-- Definition for the number of toy soldiers
def numToySoldiers : ℕ := 2 * numToyCars

-- The total number of toys is the sum of toy cars and toy soldiers
def totalToys : ℕ := numToyCars + numToySoldiers

-- Statement to prove: James buys a total of 60 toys
theorem james_total_toys : totalToys = 60 := by
  -- Insert proof here
  sorry

end james_total_toys_l276_27617


namespace edward_skee_ball_tickets_l276_27605

theorem edward_skee_ball_tickets (w_tickets : Nat) (candy_cost : Nat) (num_candies : Nat) (total_tickets : Nat) (skee_ball_tickets : Nat) :
  w_tickets = 3 ∧ candy_cost = 4 ∧ num_candies = 2 ∧ total_tickets = num_candies * candy_cost ∧ total_tickets - w_tickets = skee_ball_tickets → 
  skee_ball_tickets = 5 :=
by
  sorry

end edward_skee_ball_tickets_l276_27605


namespace mike_toys_l276_27697

theorem mike_toys (M A T : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : T = A + 2)
  (h3 : M + A + T = 56) 
  : M = 6 := 
by 
  sorry

end mike_toys_l276_27697


namespace quarterback_passes_left_l276_27622

noncomputable def number_of_passes (L : ℕ) : Prop :=
  let R := 2 * L
  let C := L + 2
  L + R + C = 50

theorem quarterback_passes_left : ∃ L, number_of_passes L ∧ L = 12 := by
  sorry

end quarterback_passes_left_l276_27622


namespace balls_distribution_ways_l276_27630

theorem balls_distribution_ways : 
  ∃ (ways : ℕ), ways = 15 := by
  sorry

end balls_distribution_ways_l276_27630


namespace all_defective_is_impossible_l276_27643

def total_products : ℕ := 10
def defective_products : ℕ := 2
def selected_products : ℕ := 3

theorem all_defective_is_impossible :
  ∀ (products : Finset ℕ),
  products.card = selected_products →
  ∀ (product_ids : Finset ℕ),
  product_ids.card = defective_products →
  products ⊆ product_ids → False :=
by
  sorry

end all_defective_is_impossible_l276_27643


namespace trigonometric_identity_l276_27624

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -1 / 2) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1 / 3 := 
by 
  sorry

end trigonometric_identity_l276_27624


namespace sequence_bound_l276_27649

theorem sequence_bound
  (a : ℕ → ℕ)
  (h_base0 : a 0 < a 1)
  (h_base1 : 0 < a 0 ∧ 0 < a 1)
  (h_recur : ∀ n, 2 ≤ n → a n = 3 * a (n - 1) - 2 * a (n - 2)) :
  a 100 > 2^99 :=
by
  sorry

end sequence_bound_l276_27649


namespace Molly_swam_on_Saturday_l276_27637

variable (total_meters : ℕ) (sunday_meters : ℕ)

def saturday_meters := total_meters - sunday_meters

theorem Molly_swam_on_Saturday : 
  total_meters = 73 ∧ sunday_meters = 28 → saturday_meters total_meters sunday_meters = 45 := by
sorry

end Molly_swam_on_Saturday_l276_27637


namespace smallest_side_length_1008_l276_27608

def smallest_side_length_original_square :=
  let n := Nat.lcm 7 8
  let n := Nat.lcm n 9
  let lcm := Nat.lcm n 10
  2 * lcm

theorem smallest_side_length_1008 :
  smallest_side_length_original_square = 1008 := by
  sorry

end smallest_side_length_1008_l276_27608


namespace math_problem_l276_27666

open Real -- Open the real number namespace

theorem math_problem (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) :=
by
  sorry

end math_problem_l276_27666


namespace evaluate_expression_l276_27693

theorem evaluate_expression : (24^36 / 72^18) = 8^18 := by
  sorry

end evaluate_expression_l276_27693


namespace soccer_balls_with_holes_l276_27645

-- Define the total number of soccer balls
def total_soccer_balls : ℕ := 40

-- Define the total number of basketballs
def total_basketballs : ℕ := 15

-- Define the number of basketballs with holes
def basketballs_with_holes : ℕ := 7

-- Define the total number of balls without holes
def total_balls_without_holes : ℕ := 18

-- Prove the number of soccer balls with holes given the conditions
theorem soccer_balls_with_holes : (total_soccer_balls - (total_balls_without_holes - (total_basketballs - basketballs_with_holes))) = 30 := by
  sorry

end soccer_balls_with_holes_l276_27645


namespace correct_statement_l276_27688

def angle_terminal_side (a b : ℝ) : Prop :=
∃ k : ℤ, a = b + k * 360

def obtuse_angle (θ : ℝ) : Prop :=
90 < θ ∧ θ < 180

def third_quadrant_angle (θ : ℝ) : Prop :=
180 < θ ∧ θ < 270

def first_quadrant_angle (θ : ℝ) : Prop :=
0 < θ ∧ θ < 90

def acute_angle (θ : ℝ) : Prop :=
0 < θ ∧ θ < 90

theorem correct_statement :
  ¬∀ a b, angle_terminal_side a b → a = b ∧
  ¬∀ θ, obtuse_angle θ → θ < θ - 360 ∧
  ¬∀ θ, first_quadrant_angle θ → acute_angle θ ∧
  ∀ θ, acute_angle θ → first_quadrant_angle θ :=
by
  sorry

end correct_statement_l276_27688


namespace sum_of_powers_mod7_l276_27653

theorem sum_of_powers_mod7 (k : ℕ) : (2^k + 3^k) % 7 = 0 ↔ k % 6 = 3 := by
  sorry

end sum_of_powers_mod7_l276_27653


namespace fruit_store_problem_l276_27674

-- Define the conditions
def total_weight : Nat := 140
def total_cost : Nat := 1000

def purchase_price_A : Nat := 5
def purchase_price_B : Nat := 9

def selling_price_A : Nat := 8
def selling_price_B : Nat := 13

-- Define the total purchase price equation
def purchase_cost (x : Nat) : Nat := purchase_price_A * x + purchase_price_B * (total_weight - x)

-- Define the profit calculation
def profit (x : Nat) (y : Nat) : Nat := (selling_price_A - purchase_price_A) * x + (selling_price_B - purchase_price_B) * y

-- State the problem
theorem fruit_store_problem :
  ∃ x y : Nat, x + y = total_weight ∧ purchase_cost x = total_cost ∧ profit x y = 495 :=
by
  sorry

end fruit_store_problem_l276_27674


namespace range_of_x_l276_27687

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

theorem range_of_x (x : ℝ) (h : f (x^2) < f (3*x - 2)) : 1 < x ∧ x < 2 :=
by
  sorry

end range_of_x_l276_27687


namespace Faye_apps_left_l276_27677

theorem Faye_apps_left (total_apps gaming_apps utility_apps deleted_gaming_apps deleted_utility_apps remaining_apps : ℕ)
  (h1 : total_apps = 12) 
  (h2 : gaming_apps = 5) 
  (h3 : utility_apps = total_apps - gaming_apps) 
  (h4 : remaining_apps = total_apps - (deleted_gaming_apps + deleted_utility_apps))
  (h5 : deleted_gaming_apps = gaming_apps) 
  (h6 : deleted_utility_apps = 3) : 
  remaining_apps = 4 :=
by
  sorry

end Faye_apps_left_l276_27677


namespace man_speed_down_l276_27648

variable (d : ℝ) (v : ℝ)

theorem man_speed_down (h1 : 32 > 0) (h2 : 38.4 > 0) (h3 : d > 0) (h4 : v > 0) 
  (avg_speed : 38.4 = (2 * d) / ((d / 32) + (d / v))) : v = 48 :=
sorry

end man_speed_down_l276_27648


namespace initial_violet_marbles_eq_l276_27610

variable {initial_violet_marbles : Nat}
variable (red_marbles : Nat := 14)
variable (total_marbles : Nat := 78)

theorem initial_violet_marbles_eq :
  initial_violet_marbles = total_marbles - red_marbles := by
  sorry

end initial_violet_marbles_eq_l276_27610


namespace inequality_always_true_l276_27601

-- Definitions from the conditions
variables {a b c : ℝ}
variable (h1 : a < b)
variable (h2 : b < c)
variable (h3 : a + b + c = 0)

-- The statement to prove
theorem inequality_always_true : c * a < c * b :=
by
  -- Proof steps go here.
  sorry

end inequality_always_true_l276_27601


namespace sum_of_fractions_l276_27658

-- Definition of the fractions
def frac1 : ℚ := 3/5
def frac2 : ℚ := 5/11
def frac3 : ℚ := 1/3

-- Main theorem stating that the sum of the fractions equals 229/165
theorem sum_of_fractions : frac1 + frac2 + frac3 = 229 / 165 := sorry

end sum_of_fractions_l276_27658


namespace increase_number_correct_l276_27660

-- Definitions for the problem
def originalNumber : ℕ := 110
def increasePercent : ℝ := 0.5

-- Statement to be proved
theorem increase_number_correct : originalNumber + (originalNumber * increasePercent) = 165 := by
  sorry

end increase_number_correct_l276_27660


namespace negation_exists_to_forall_l276_27656

theorem negation_exists_to_forall (P : ℝ → Prop) (h : ∃ x : ℝ, x^2 + 3 * x + 2 < 0) :
  (¬ (∃ x : ℝ, x^2 + 3 * x + 2 < 0)) ↔ (∀ x : ℝ, x^2 + 3 * x + 2 ≥ 0) := by
sorry

end negation_exists_to_forall_l276_27656


namespace range_of_a_exists_x_ax2_ax_1_lt_0_l276_27632

theorem range_of_a_exists_x_ax2_ax_1_lt_0 :
  {a : ℝ | ∃ x : ℝ, a * x^2 + a * x + 1 < 0} = {a : ℝ | a < 0 ∨ a > 4} :=
sorry

end range_of_a_exists_x_ax2_ax_1_lt_0_l276_27632


namespace asymptotes_of_hyperbola_eq_m_l276_27659

theorem asymptotes_of_hyperbola_eq_m :
  ∀ (m : ℝ), (∀ (x y : ℝ), (x^2 / 16 - y^2 / 25 = 1) → (y = m * x ∨ y = -m * x)) → m = 5 / 4 :=
by 
  sorry

end asymptotes_of_hyperbola_eq_m_l276_27659


namespace problem_statement_l276_27602

-- Define the binary operation "*"
def custom_mul (a b : ℤ) : ℤ := a^2 + a * b - b^2

-- State the problem with the conditions
theorem problem_statement : custom_mul 5 (-3) = 1 := by
  sorry

end problem_statement_l276_27602


namespace stream_current_rate_l276_27621

theorem stream_current_rate (r w : ℝ) : (
  (18 / (r + w) + 6 = 18 / (r - w)) ∧ 
  (18 / (3 * r + w) + 2 = 18 / (3 * r - w))
) → w = 6 := 
by {
  sorry
}

end stream_current_rate_l276_27621


namespace binary_to_decimal_l276_27607

theorem binary_to_decimal :
  1 * 2^8 + 0 * 2^7 + 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 379 :=
by
  sorry

end binary_to_decimal_l276_27607


namespace solve_inequality_l276_27611

theorem solve_inequality :
  { x : ℝ | (x - 5) / (x - 3)^2 < 0 } = { x : ℝ | x < 3 } ∪ { x : ℝ | 3 < x ∧ x < 5 } :=
by
  sorry

end solve_inequality_l276_27611


namespace parallel_lines_l276_27689

-- Definitions for the equations of the lines
def l1 (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 10 = 0
def l2 (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- Theorem stating the conditions under which the lines l1 and l2 are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y) = (∀ x y : ℝ, l2 a x y) → a = -1 ∨ a = 2 :=
by sorry

end parallel_lines_l276_27689


namespace maximal_cardinality_set_l276_27641

theorem maximal_cardinality_set (n : ℕ) (h_n : n ≥ 2) :
  ∃ M : Finset (ℕ × ℕ), ∀ (j k : ℕ), (1 ≤ j ∧ j < k ∧ k ≤ n) → 
  ((j, k) ∈ M → ∀ m, (k, m) ∉ M) ∧ 
  M.card = ⌊(n * n / 4 : ℝ)⌋ :=
by
  sorry

end maximal_cardinality_set_l276_27641


namespace polynomial_root_interval_l276_27695

open Real

theorem polynomial_root_interval (b : ℝ) (x : ℝ) :
  (x^4 + b*x^3 + x^2 + b*x - 1 = 0) → (b ≤ -2 * sqrt 3 ∨ b ≥ 0) :=
sorry

end polynomial_root_interval_l276_27695


namespace dummies_remainder_l276_27692

/-
  Prove that if the number of Dummies in one bag is such that when divided among 10 kids, 3 pieces are left over,
  then the number of Dummies in four bags when divided among 10 kids leaves 2 pieces.
-/
theorem dummies_remainder (n : ℕ) (h : n % 10 = 3) : (4 * n) % 10 = 2 := 
by {
  sorry
}

end dummies_remainder_l276_27692


namespace number_of_possible_values_for_b_l276_27671

theorem number_of_possible_values_for_b : 
  ∃ (n : ℕ), n = 2 ∧ ∀ b : ℕ, (b ≥ 2 ∧ b^3 ≤ 197 ∧ 197 < b^4) → b = 4 ∨ b = 5 :=
sorry

end number_of_possible_values_for_b_l276_27671


namespace sum_of_ages_is_60_l276_27657

theorem sum_of_ages_is_60 (A B : ℕ) (h1 : A = 2 * B) (h2 : (A + 3) + (B + 3) = 66) : A + B = 60 :=
by sorry

end sum_of_ages_is_60_l276_27657


namespace cylinder_projections_tangency_l276_27620

def plane1 : Type := sorry
def plane2 : Type := sorry
def projection_axis : Type := sorry
def is_tangent_to (cylinder : Type) (plane : Type) : Prop := sorry
def is_base_tangent_to (cylinder : Type) (axis : Type) : Prop := sorry
def cylinder : Type := sorry

theorem cylinder_projections_tangency (P1 P2 : Type) (axis : Type)
  (h1 : is_tangent_to cylinder P1) 
  (h2 : is_tangent_to cylinder P2) 
  (h3 : is_base_tangent_to cylinder axis) : 
  ∃ (solutions : ℕ), solutions = 4 :=
sorry

end cylinder_projections_tangency_l276_27620


namespace avg_percentage_students_l276_27640

-- Define the function that calculates the average percentage of all students
def average_percent (n1 n2 : ℕ) (p1 p2 : ℕ) : ℕ :=
  (n1 * p1 + n2 * p2) / (n1 + n2)

-- Define the properties of the numbers of students and their respective percentages
def students_avg : Prop :=
  average_percent 15 10 70 90 = 78

-- The main theorem: Prove that given the conditions, the average percentage is 78%
theorem avg_percentage_students : students_avg :=
  by
    -- The proof will be provided here.
    sorry

end avg_percentage_students_l276_27640


namespace volume_ratio_l276_27652

theorem volume_ratio (a : ℕ) (b : ℕ) (ft_to_inch : ℕ) (h1 : a = 4) (h2 : b = 2 * ft_to_inch) (ft_to_inch_value : ft_to_inch = 12) :
  (a^3) / (b^3) = 1 / 216 :=
by
  sorry

end volume_ratio_l276_27652


namespace only_n_is_zero_l276_27631

theorem only_n_is_zero (n : ℕ) (h : (n^2 + 1) ∣ n) : n = 0 := 
by sorry

end only_n_is_zero_l276_27631


namespace find_square_l276_27655

-- Define the conditions as hypotheses
theorem find_square (p : ℕ) (sq : ℕ)
  (h1 : sq + p = 75)
  (h2 : (sq + p) + p = 142) :
  sq = 8 := by
  sorry

end find_square_l276_27655


namespace ladder_leaning_distance_l276_27650

variable (m f h : ℝ)
variable (f_pos : f > 0) (h_pos : h > 0)

def distance_to_wall_upper_bound : ℝ := 12.46
def distance_to_wall_lower_bound : ℝ := 8.35

theorem ladder_leaning_distance (m f h : ℝ) (f_pos : f > 0) (h_pos : h > 0) :
  ∃ x : ℝ, x = 12.46 ∨ x = 8.35 := 
sorry

end ladder_leaning_distance_l276_27650


namespace isabel_uploaded_pictures_l276_27681

theorem isabel_uploaded_pictures :
  let album1 := 10
  let total_other_pictures := 5 * 3
  let total_pictures := album1 + total_other_pictures
  total_pictures = 25 :=
by
  let album1 := 10
  let total_other_pictures := 5 * 3
  let total_pictures := album1 + total_other_pictures
  show total_pictures = 25
  sorry

end isabel_uploaded_pictures_l276_27681


namespace chandler_bike_purchase_l276_27627

theorem chandler_bike_purchase : 
  ∀ (x : ℕ), (120 + 20 * x = 640) → x = 26 := 
by
  sorry

end chandler_bike_purchase_l276_27627


namespace poodle_barks_proof_l276_27654

-- Definitions based on our conditions
def terrier_barks (hushes : Nat) : Nat := hushes * 2
def poodle_barks (terrier_barks : Nat) : Nat := terrier_barks * 2

-- Given that the terrier's owner says "hush" six times
def hushes : Nat := 6
def terrier_barks_total : Nat := terrier_barks hushes

-- The final statement that we need to prove
theorem poodle_barks_proof : 
    ∃ P, P = poodle_barks terrier_barks_total ∧ P = 24 := 
by
  -- The proof goes here
  sorry

end poodle_barks_proof_l276_27654


namespace determine_x_l276_27670

noncomputable def x_candidates := { x : ℝ | x = (3 + Real.sqrt 105) / 24 ∨ x = (3 - Real.sqrt 105) / 24 }

theorem determine_x (x y : ℝ) (h_y : y = 3 * x) 
  (h_eq : 4 * y ^ 2 + 2 * y + 7 = 3 * (8 * x ^ 2 + y + 3)) :
  x ∈ x_candidates :=
by
  sorry

end determine_x_l276_27670


namespace solution_set_ineq_l276_27685

open Set

theorem solution_set_ineq (a x : ℝ) (h : 0 < a ∧ a < 1) : 
 (a < x ∧ x < 1/a) ↔ ((x - a) * (x - 1/a) > 0) :=
by
  sorry

end solution_set_ineq_l276_27685


namespace prime_divisor_property_l276_27698

-- Given conditions
variable (p k : ℕ)
variable (prime_p : Nat.Prime p)
variable (divisor_p : p ∣ (2 ^ (2 ^ k)) + 1)

-- The theorem we need to prove
theorem prime_divisor_property (p k : ℕ) (prime_p : Nat.Prime p) (divisor_p : p ∣ (2 ^ (2 ^ k)) + 1) : (2 ^ (k + 1)) ∣ (p - 1) := 
by 
  sorry

end prime_divisor_property_l276_27698


namespace ratio_of_areas_l276_27612

structure Triangle :=
  (AB BC AC AD AE : ℝ)
  (AB_pos : 0 < AB)
  (BC_pos : 0 < BC)
  (AC_pos : 0 < AC)
  (AD_pos : 0 < AD)
  (AE_pos : 0 < AE)

theorem ratio_of_areas (t : Triangle)
  (hAB : t.AB = 30)
  (hBC : t.BC = 45)
  (hAC : t.AC = 54)
  (hAD : t.AD = 24)
  (hAE : t.AE = 18) :
  (t.AD / t.AB) * (t.AE / t.AC) / (1 - (t.AD / t.AB) * (t.AE / t.AC)) = 4 / 11 :=
by
  sorry

end ratio_of_areas_l276_27612


namespace binary_101110_to_octal_l276_27684

-- Definition: binary number 101110 represents some decimal number
def binary_101110 : ℕ := 0 * 2^0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5

-- Definition: decimal number 46 represents some octal number
def decimal_46 := 46

-- A utility function to convert decimal to octal (returns the digits as a list)
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else decimal_to_octal (n / 8) ++ [n % 8]

-- Hypothesis: the binary 101110 equals the decimal 46
lemma binary_101110_eq_46 : binary_101110 = decimal_46 := by sorry

-- Hypothesis: the decimal 46 converts to the octal number 56 (in list form)
def octal_56 := [5, 6]

-- Theorem: binary 101110 converts to the octal number 56
theorem binary_101110_to_octal :
  decimal_to_octal binary_101110 = octal_56 := by
  rw [binary_101110_eq_46]
  sorry

end binary_101110_to_octal_l276_27684
