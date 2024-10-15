import Mathlib

namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l755_75561

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h1 : a + b + c = 60) 
  (h2 : 0.5 * a * b = 120) 
  (h3 : a^2 + b^2 = c^2) : 
  c = 26 :=
by {
  sorry
}

end NUMINAMATH_GPT_right_triangle_hypotenuse_l755_75561


namespace NUMINAMATH_GPT_volume_of_fifth_section_l755_75597

theorem volume_of_fifth_section (a : ℕ → ℚ) (d : ℚ) :
  (a 1 + a 2 + a 3 + a 4) = 3 ∧ (a 9 + a 8 + a 7) = 4 ∧
  (∀ n, a n = a 1 + (n - 1) * d) →
  a 5 = 67 / 66 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_fifth_section_l755_75597


namespace NUMINAMATH_GPT_slant_height_l755_75588

-- Define the variables and conditions
variables (r A : ℝ)
-- Assume the given conditions
def radius := r = 5
def area := A = 60 * Real.pi

-- Statement of the theorem to prove the slant height
theorem slant_height (r A l : ℝ) (h_r : r = 5) (h_A : A = 60 * Real.pi) : l = 12 :=
sorry

end NUMINAMATH_GPT_slant_height_l755_75588


namespace NUMINAMATH_GPT_probability_adjacent_difference_l755_75595

noncomputable def probability_no_adjacent_same_rolls : ℚ :=
  (7 / 8) ^ 6

theorem probability_adjacent_difference :
  let num_people := 6
  let sides_of_die := 8
  ( ∀ i : ℕ, 0 ≤ i ∧ i < num_people -> (∃ x : ℕ, 1 ≤ x ∧ x ≤ sides_of_die)) →
  probability_no_adjacent_same_rolls = 117649 / 262144 := 
by 
  sorry

end NUMINAMATH_GPT_probability_adjacent_difference_l755_75595


namespace NUMINAMATH_GPT_undefined_expression_values_l755_75579

theorem undefined_expression_values : 
    ∃ x : ℝ, x^2 - 9 = 0 ↔ (x = -3 ∨ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_undefined_expression_values_l755_75579


namespace NUMINAMATH_GPT_find_A_l755_75510

def is_divisible (n : ℕ) (d : ℕ) : Prop := d ∣ n

noncomputable def valid_digit (A : ℕ) : Prop :=
  A < 10

noncomputable def digit_7_number := 653802 * 10

theorem find_A (A : ℕ) (h : valid_digit A) :
  is_divisible (digit_7_number + A) 2 ∧
  is_divisible (digit_7_number + A) 3 ∧
  is_divisible (digit_7_number + A) 4 ∧
  is_divisible (digit_7_number + A) 6 ∧
  is_divisible (digit_7_number + A) 8 ∧
  is_divisible (digit_7_number + A) 9 ∧
  is_divisible (digit_7_number + A) 25 →
  A = 0 :=
sorry

end NUMINAMATH_GPT_find_A_l755_75510


namespace NUMINAMATH_GPT_geometric_sequence_S6_l755_75505

variable (a : ℕ → ℝ) -- represents the geometric sequence

noncomputable def S (n : ℕ) : ℝ :=
if n = 0 then 0 else ((a 0) * (1 - (a 1 / a 0) ^ n)) / (1 - a 1 / a 0)

theorem geometric_sequence_S6 (h : ∀ n, a n = (a 0) * (a 1 / a 0) ^ n) :
  S a 2 = 6 ∧ S a 4 = 18 → S a 6 = 42 := 
by 
  intros h1
  sorry

end NUMINAMATH_GPT_geometric_sequence_S6_l755_75505


namespace NUMINAMATH_GPT_batsman_average_increase_l755_75565

theorem batsman_average_increase (A : ℕ) 
    (h1 : 15 * A + 64 = 19 * 16) 
    (h2 : 19 - A = 3) : 
    19 - A = 3 := 
sorry

end NUMINAMATH_GPT_batsman_average_increase_l755_75565


namespace NUMINAMATH_GPT_length_AB_l755_75583

theorem length_AB :
  ∀ (A B : ℝ × ℝ) (k : ℝ),
    (A.2 = k * A.1 - 2) ∧ (B.2 = k * B.1 - 2) ∧ (A.2^2 = 8 * A.1) ∧ (B.2^2 = 8 * B.1) ∧
    ((A.1 + B.1) / 2 = 2) →
  dist A B = 2 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_length_AB_l755_75583


namespace NUMINAMATH_GPT_not_mutually_exclusive_option_D_l755_75526

-- Definitions for mutually exclusive events
def mutually_exclusive (event1 event2 : Prop) : Prop := ¬ (event1 ∧ event2)

-- Conditions as given in the problem
def eventA1 : Prop := True -- Placeholder for "score is greater than 8"
def eventA2 : Prop := True -- Placeholder for "score is less than 6"

def eventB1 : Prop := True -- Placeholder for "90 seeds germinate"
def eventB2 : Prop := True -- Placeholder for "80 seeds germinate"

def eventC1 : Prop := True -- Placeholder for "pass rate is higher than 70%"
def eventC2 : Prop := True -- Placeholder for "pass rate is 70%"

def eventD1 : Prop := True -- Placeholder for "average score is not lower than 90"
def eventD2 : Prop := True -- Placeholder for "average score is not higher than 120"

-- Lean proof statement
theorem not_mutually_exclusive_option_D :
  mutually_exclusive eventA1 eventA2 ∧
  mutually_exclusive eventB1 eventB2 ∧
  mutually_exclusive eventC1 eventC2 ∧
  ¬ mutually_exclusive eventD1 eventD2 :=
sorry

end NUMINAMATH_GPT_not_mutually_exclusive_option_D_l755_75526


namespace NUMINAMATH_GPT_triangle_area_of_parabola_intersection_l755_75586

theorem triangle_area_of_parabola_intersection
  (line_passes_through : ∃ (p : ℝ × ℝ), p = (0, -2))
  (parabola_intersection : ∃ (x1 y1 x2 y2 : ℝ),
    (x1, y1) ∈ {p : ℝ × ℝ | p.snd ^ 2 = 16 * p.fst} ∧
    (x2, y2) ∈ {p : ℝ × ℝ | p.snd ^ 2 = 16 * p.fst})
  (y_cond : ∃ (y1 y2 : ℝ), y1 ^ 2 - y2 ^ 2 = 1) :
  ∃ (area : ℝ), area = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_of_parabola_intersection_l755_75586


namespace NUMINAMATH_GPT_unique_triple_satisfying_conditions_l755_75576

theorem unique_triple_satisfying_conditions :
  ∃! (x y z : ℝ), x + y = 4 ∧ xy - z^2 = 4 :=
sorry

end NUMINAMATH_GPT_unique_triple_satisfying_conditions_l755_75576


namespace NUMINAMATH_GPT_find_angle_MBA_l755_75599

-- Define the angles and the triangle
def triangle (A B C : Type) := true

-- Define the angles in degrees
def angle (deg : ℝ) := deg

-- Assume angles' degrees as given in the problem
variables {A B C M : Type}
variable {BAC ABC MAB MCA MBA : ℝ}

-- Given conditions
axiom angle_BAC : angle BAC = 30
axiom angle_ABC : angle ABC = 70
axiom angle_MAB : angle MAB = 20
axiom angle_MCA : angle MCA = 20

-- Prove that angle MBA is 30 degrees
theorem find_angle_MBA : angle MBA = 30 := 
by 
  sorry

end NUMINAMATH_GPT_find_angle_MBA_l755_75599


namespace NUMINAMATH_GPT_min_value_expression_l755_75525

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    let a := 2
    let b := 3
    let term1 := 2*x + 1/(3*y)
    let term2 := 3*y + 1/(2*x)
    (term1 * (term1 - 2023) + term2 * (term2 - 2023)) = -2050529.5 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l755_75525


namespace NUMINAMATH_GPT_hyperbola_focus_coordinates_l755_75540

theorem hyperbola_focus_coordinates:
  ∀ (x y : ℝ), 
    (x - 5)^2 / 7^2 - (y - 12)^2 / 10^2 = 1 → 
      ∃ (c : ℝ), c = 5 + Real.sqrt 149 ∧ (x, y) = (c, 12) :=
by
  intros x y h
  -- prove the coordinates of the focus with the larger x-coordinate are (5 + sqrt 149, 12)
  sorry

end NUMINAMATH_GPT_hyperbola_focus_coordinates_l755_75540


namespace NUMINAMATH_GPT_commercials_played_l755_75594

theorem commercials_played (M C : ℝ) (h1 : M / C = 9 / 5) (h2 : M + C = 112) : C = 40 :=
by
  sorry

end NUMINAMATH_GPT_commercials_played_l755_75594


namespace NUMINAMATH_GPT_solve_for_y_l755_75548

theorem solve_for_y (y : ℝ) (hy : y ≠ -2) : 
  (6 * y / (y + 2) - 2 / (y + 2) = 5 / (y + 2)) ↔ y = 7 / 6 :=
by sorry

end NUMINAMATH_GPT_solve_for_y_l755_75548


namespace NUMINAMATH_GPT_find_a_l755_75573

def A : Set ℤ := {-1, 1, 3}
def B (a : ℤ) : Set ℤ := {a + 1, a^2 + 4}
def intersection (a : ℤ) : Set ℤ := A ∩ B a

theorem find_a : ∃ a : ℤ, intersection a = {3} ∧ a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l755_75573


namespace NUMINAMATH_GPT_chromium_percentage_in_new_alloy_l755_75584

theorem chromium_percentage_in_new_alloy :
  ∀ (weight1 weight2 chromium1 chromium2: ℝ),
  weight1 = 15 → weight2 = 35 → chromium1 = 0.12 → chromium2 = 0.08 →
  (chromium1 * weight1 + chromium2 * weight2) / (weight1 + weight2) * 100 = 9.2 :=
by
  intros weight1 weight2 chromium1 chromium2 hweight1 hweight2 hchromium1 hchromium2
  sorry

end NUMINAMATH_GPT_chromium_percentage_in_new_alloy_l755_75584


namespace NUMINAMATH_GPT_triangle_area_l755_75556

theorem triangle_area (P : ℝ) (r : ℝ) (s : ℝ) (A : ℝ) :
  P = 42 → r = 5 → s = P / 2 → A = r * s → A = 105 :=
by
  intro hP hr hs hA
  sorry

end NUMINAMATH_GPT_triangle_area_l755_75556


namespace NUMINAMATH_GPT_a_sub_b_eq_2_l755_75582

theorem a_sub_b_eq_2 (a b : ℝ)
  (h : (a - 5) ^ 2 + |b ^ 3 - 27| = 0) : a - b = 2 :=
by
  sorry

end NUMINAMATH_GPT_a_sub_b_eq_2_l755_75582


namespace NUMINAMATH_GPT_count_complex_numbers_l755_75537

theorem count_complex_numbers (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_sum : a + b ≤ 5) : 
  ∃ n : ℕ, n = 10 :=
by
  sorry

end NUMINAMATH_GPT_count_complex_numbers_l755_75537


namespace NUMINAMATH_GPT_stacy_faster_than_heather_l755_75580

-- Definitions for the conditions
def distance : ℝ := 40
def heather_rate : ℝ := 5
def heather_distance : ℝ := 17.090909090909093
def heather_delay : ℝ := 0.4
def stacy_distance : ℝ := distance - heather_distance
def stacy_rate (S : ℝ) (T : ℝ) : Prop := S * T = stacy_distance
def heather_time (T : ℝ) : ℝ := T - heather_delay
def heather_walk_eq (T : ℝ) : Prop := heather_rate * heather_time T = heather_distance

-- The proof problem statement
theorem stacy_faster_than_heather :
  ∃ (S T : ℝ), stacy_rate S T ∧ heather_walk_eq T ∧ (S - heather_rate = 1) :=
by
  sorry

end NUMINAMATH_GPT_stacy_faster_than_heather_l755_75580


namespace NUMINAMATH_GPT_imaginary_part_of_complex_division_l755_75592

theorem imaginary_part_of_complex_division : 
  let i := Complex.I
  let z := (1 - 2 * i) / (2 - i)
  Complex.im z = -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_complex_division_l755_75592


namespace NUMINAMATH_GPT_chess_tournament_games_l755_75504

theorem chess_tournament_games (n : ℕ) (h : n = 16) :
  (n * (n - 1) * 2) / 2 = 480 :=
by
  rw [h]
  simp
  norm_num
  sorry

end NUMINAMATH_GPT_chess_tournament_games_l755_75504


namespace NUMINAMATH_GPT_lcm_36_90_eq_180_l755_75507

theorem lcm_36_90_eq_180 : Nat.lcm 36 90 = 180 := 
by 
  sorry

end NUMINAMATH_GPT_lcm_36_90_eq_180_l755_75507


namespace NUMINAMATH_GPT_seq_sum_eq_314_l755_75521

theorem seq_sum_eq_314 (d r : ℕ) (k : ℕ) (a_n b_n c_n : ℕ → ℕ)
  (h1 : ∀ n, a_n n = 1 + (n - 1) * d)
  (h2 : ∀ n, b_n n = r ^ (n - 1))
  (h3 : ∀ n, c_n n = a_n n + b_n n)
  (hk1 : c_n (k - 1) = 150)
  (hk2 : c_n (k + 1) = 900) :
  c_n k = 314 := by
  sorry

end NUMINAMATH_GPT_seq_sum_eq_314_l755_75521


namespace NUMINAMATH_GPT_find_a_l755_75502

theorem find_a (a : ℝ) (h : ∃ (b : ℝ), (16 * (x : ℝ) * x) + 40 * x + a = (4 * x + b) ^ 2) : a = 25 := sorry

end NUMINAMATH_GPT_find_a_l755_75502


namespace NUMINAMATH_GPT_find_sum_of_integers_l755_75551

theorem find_sum_of_integers (x y : ℕ) (h_diff : x - y = 8) (h_prod : x * y = 180) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : x + y = 28 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_integers_l755_75551


namespace NUMINAMATH_GPT_teams_face_each_other_l755_75530

theorem teams_face_each_other (n : ℕ) (total_games : ℕ) (k : ℕ)
  (h1 : n = 20)
  (h2 : total_games = 760)
  (h3 : total_games = n * (n - 1) * k / 2) :
  k = 4 :=
by
  sorry

end NUMINAMATH_GPT_teams_face_each_other_l755_75530


namespace NUMINAMATH_GPT_center_of_circle_l755_75568

theorem center_of_circle : ∀ (x y : ℝ), x^2 + y^2 = 4 * x - 6 * y + 9 → (x, y) = (2, -3) :=
by
sorry

end NUMINAMATH_GPT_center_of_circle_l755_75568


namespace NUMINAMATH_GPT_functional_eq_solution_l755_75528

theorem functional_eq_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) :
  ∀ x : ℝ, f x = x := 
sorry

end NUMINAMATH_GPT_functional_eq_solution_l755_75528


namespace NUMINAMATH_GPT_triangles_from_pentadecagon_l755_75570

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon
    is 455, given that there are 15 vertices and none of them are collinear. -/

theorem triangles_from_pentadecagon : (Nat.choose 15 3) = 455 := 
by
  sorry

end NUMINAMATH_GPT_triangles_from_pentadecagon_l755_75570


namespace NUMINAMATH_GPT_number_of_participants_l755_75500

theorem number_of_participants (n : ℕ) (h : n - 1 = 25) : n = 26 := 
by sorry

end NUMINAMATH_GPT_number_of_participants_l755_75500


namespace NUMINAMATH_GPT_find_m_of_ellipse_conditions_l755_75572

-- definition for isEllipseGivenFocus condition
def isEllipseGivenFocus (m : ℝ) : Prop :=
  ∃ (a : ℝ), a = 5 ∧ (-4)^2 = a^2 - m^2 ∧ 0 < m

-- statement to prove the described condition implies m = 3
theorem find_m_of_ellipse_conditions (m : ℝ) (h : isEllipseGivenFocus m) : m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_of_ellipse_conditions_l755_75572


namespace NUMINAMATH_GPT_bulb_cheaper_than_lamp_by_4_l755_75532

/-- Jim bought a $7 lamp and a bulb. The bulb cost a certain amount less than the lamp. 
    He bought 2 lamps and 6 bulbs and paid $32 in all. 
    The amount by which the bulb is cheaper than the lamp is $4. -/
theorem bulb_cheaper_than_lamp_by_4
  (lamp_price bulb_price : ℝ)
  (h1 : lamp_price = 7)
  (h2 : bulb_price = 7 - 4)
  (h3 : 2 * lamp_price + 6 * bulb_price = 32) :
  (7 - bulb_price = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_bulb_cheaper_than_lamp_by_4_l755_75532


namespace NUMINAMATH_GPT_geometric_product_seven_terms_l755_75563

theorem geometric_product_seven_terms (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 1) 
  (h2 : a 6 + a 4 = 2 * (a 3 + a 1)) 
  (h_geometric : ∀ n, a (n + 1) = q * a n) :
  (a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7) = 128 := 
by 
  -- Steps involving algebraic manipulation and properties of geometric sequences should be here
  sorry

end NUMINAMATH_GPT_geometric_product_seven_terms_l755_75563


namespace NUMINAMATH_GPT_olivia_earning_l755_75511

theorem olivia_earning
  (cost_per_bar : ℝ)
  (total_bars : ℕ)
  (unsold_bars : ℕ)
  (sold_bars : ℕ := total_bars - unsold_bars)
  (earnings : ℝ := sold_bars * cost_per_bar) :
  cost_per_bar = 3 → total_bars = 7 → unsold_bars = 4 → earnings = 9 :=
by
  sorry

end NUMINAMATH_GPT_olivia_earning_l755_75511


namespace NUMINAMATH_GPT_none_of_these_l755_75571

-- Problem Statement:
theorem none_of_these (r x y : ℝ) (h1 : r > 0) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : x^2 + y^2 > x^2 * y^2) :
  ¬ (-x > -y) ∧ ¬ (-x > y) ∧ ¬ (1 > -y / x) ∧ ¬ (1 < x / y) :=
by
  sorry

end NUMINAMATH_GPT_none_of_these_l755_75571


namespace NUMINAMATH_GPT_f_correct_l755_75522

noncomputable def f (n : ℕ) : ℕ :=
  if h : n ≥ 15 then (n - 1) / 2
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else if n = 6 then 4
  else if 7 ≤ n ∧ n ≤ 15 then 7
  else 0

theorem f_correct (n : ℕ) (hn : n ≥ 3) : 
  f n = if n ≥ 15 then (n - 1) / 2
        else if n = 3 then 1
        else if n = 4 then 1
        else if n = 5 then 2
        else if n = 6 then 4
        else if 7 ≤ n ∧ n ≤ 15 then 7
        else 0 := sorry

end NUMINAMATH_GPT_f_correct_l755_75522


namespace NUMINAMATH_GPT_sum_of_ages_is_18_l755_75544

-- Define the conditions
def product_of_ages (kiana twin : ℕ) := kiana * twin^2 = 128

-- Define the proof problem statement
theorem sum_of_ages_is_18 : ∃ (kiana twin : ℕ), product_of_ages kiana twin ∧ twin > kiana ∧ kiana + twin + twin = 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_18_l755_75544


namespace NUMINAMATH_GPT_solve_trigonometric_eqn_l755_75574

theorem solve_trigonometric_eqn (x : ℝ) : 
  (∃ k : ℤ, x = 3 * (π / 4 * (4 * k + 1))) ∨ (∃ n : ℤ, x = π * (3 * n + 1) ∨ x = π * (3 * n - 1)) :=
by 
  sorry

end NUMINAMATH_GPT_solve_trigonometric_eqn_l755_75574


namespace NUMINAMATH_GPT_three_numbers_equal_l755_75559

theorem three_numbers_equal {a b c d : ℕ} 
  (h : ∀ {x y z w : ℕ}, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
                  (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ (w = a ∨ w = b ∨ w = c ∨ w = d) → x * y + z * w = x * z + y * w) :
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end NUMINAMATH_GPT_three_numbers_equal_l755_75559


namespace NUMINAMATH_GPT_password_count_l755_75509

theorem password_count : ∃ s : Finset ℕ, s.card = 4 ∧ s.sum id = 27 ∧ 
  (s = {9, 8, 7, 3} ∨ s = {9, 8, 6, 4} ∨ s = {9, 7, 6, 5}) ∧ 
  (s.toList.permutations.length = 72) := sorry

end NUMINAMATH_GPT_password_count_l755_75509


namespace NUMINAMATH_GPT_barefoot_kids_l755_75545

theorem barefoot_kids (total_kids kids_socks kids_shoes kids_both : ℕ) 
  (h1 : total_kids = 22) 
  (h2 : kids_socks = 12) 
  (h3 : kids_shoes = 8) 
  (h4 : kids_both = 6) : 
  (total_kids - (kids_socks - kids_both + kids_shoes - kids_both + kids_both) = 8) :=
by
  -- following sorry to skip proof.
  sorry

end NUMINAMATH_GPT_barefoot_kids_l755_75545


namespace NUMINAMATH_GPT_all_initial_rectangles_are_squares_l755_75552

theorem all_initial_rectangles_are_squares (n : ℕ) (total_squares : ℕ) (h_prime : Nat.Prime total_squares) 
  (cut_rect_into_squares : ℕ → ℕ → ℕ → Prop) :
  ∀ (a b : ℕ), (∀ i, i < n → cut_rect_into_squares a b total_squares) → a = b :=
by 
  sorry

end NUMINAMATH_GPT_all_initial_rectangles_are_squares_l755_75552


namespace NUMINAMATH_GPT_sequence_n_value_l755_75541

theorem sequence_n_value (n : ℤ) : (2 * n^2 - 3 = 125) → (n = 8) := 
by {
    sorry
}

end NUMINAMATH_GPT_sequence_n_value_l755_75541


namespace NUMINAMATH_GPT_parabola_equation_l755_75503

-- Define the constants and the conditions
def parabola_focus : ℝ × ℝ := (3, 3)
def directrix : ℝ × ℝ × ℝ := (3, 7, -21)

theorem parabola_equation :
  ∃ a b c d e f : ℤ,
  a > 0 ∧
  Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd a b) c) d) e) f = 1 ∧
  (a : ℝ) * x^2 + (b : ℝ) * x * y + (c : ℝ) * y^2 + (d : ℝ) * x + (e : ℝ) * y + (f : ℝ) = 
  49 * x^2 - 42 * x * y + 9 * y^2 - 222 * x - 54 * y + 603 := sorry

end NUMINAMATH_GPT_parabola_equation_l755_75503


namespace NUMINAMATH_GPT_bullet_trains_crossing_time_l755_75569

theorem bullet_trains_crossing_time
  (length_train1 : ℝ) (length_train2 : ℝ)
  (speed_train1_km_hr : ℝ) (speed_train2_km_hr : ℝ)
  (opposite_directions : Prop)
  (h_length1 : length_train1 = 140)
  (h_length2 : length_train2 = 170)
  (h_speed1 : speed_train1_km_hr = 60)
  (h_speed2 : speed_train2_km_hr = 40)
  (h_opposite : opposite_directions = true) :
  ∃ t : ℝ, t = 11.16 :=
by
  sorry

end NUMINAMATH_GPT_bullet_trains_crossing_time_l755_75569


namespace NUMINAMATH_GPT_maximum_volume_of_pyramid_l755_75515

theorem maximum_volume_of_pyramid (a b : ℝ) (hb : b > 0) (ha : a > 0):
  ∃ V_max : ℝ, V_max = (a * (4 * b ^ 2 - a ^ 2)) / 12 := 
sorry

end NUMINAMATH_GPT_maximum_volume_of_pyramid_l755_75515


namespace NUMINAMATH_GPT_surface_area_increase_96_percent_l755_75523

variable (s : ℝ)

def original_surface_area : ℝ := 6 * s^2
def new_edge_length : ℝ := 1.4 * s
def new_surface_area : ℝ := 6 * (new_edge_length s)^2

theorem surface_area_increase_96_percent :
  (new_surface_area s - original_surface_area s) / (original_surface_area s) * 100 = 96 :=
by
  simp [original_surface_area, new_edge_length, new_surface_area]
  sorry

end NUMINAMATH_GPT_surface_area_increase_96_percent_l755_75523


namespace NUMINAMATH_GPT_average_score_of_remaining_students_correct_l755_75506

noncomputable def average_score_remaining_students (n : ℕ) (h_n : n > 15) (avg_all : ℚ) (avg_subgroup : ℚ) : ℚ :=
if h_avg_all : avg_all = 10 ∧ avg_subgroup = 16 then
  (10 * n - 240) / (n - 15)
else
  0

theorem average_score_of_remaining_students_correct (n : ℕ) (h_n : n > 15) :
  (average_score_remaining_students n h_n 10 16) = (10 * n - 240) / (n - 15) :=
by
  dsimp [average_score_remaining_students]
  split_ifs with h_avg
  · sorry
  · sorry

end NUMINAMATH_GPT_average_score_of_remaining_students_correct_l755_75506


namespace NUMINAMATH_GPT_joelle_initial_deposit_l755_75577

-- Definitions for the conditions
def annualInterestRate : ℝ := 0.05
def initialTimePeriod : ℕ := 2 -- in years
def numberOfCompoundsPerYear : ℕ := 1
def finalAmount : ℝ := 6615

-- Compound interest formula: A = P(1 + r/n)^(nt)
noncomputable def initialDeposit : ℝ :=
  finalAmount / ((1 + annualInterestRate / numberOfCompoundsPerYear)^(numberOfCompoundsPerYear * initialTimePeriod))

-- Theorem statement to prove the initial deposit
theorem joelle_initial_deposit : initialDeposit = 6000 := 
  sorry

end NUMINAMATH_GPT_joelle_initial_deposit_l755_75577


namespace NUMINAMATH_GPT_combined_future_value_l755_75590

noncomputable def future_value (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem combined_future_value :
  let A1 := future_value 3000 0.05 3
  let A2 := future_value 5000 0.06 4
  let A3 := future_value 7000 0.07 5
  A1 + A2 + A3 = 19603.119 :=
by
  sorry

end NUMINAMATH_GPT_combined_future_value_l755_75590


namespace NUMINAMATH_GPT_factor_polynomial_l755_75517

theorem factor_polynomial (x : ℝ) :
  3 * x^2 * (x - 5) + 5 * (x - 5) = (3 * x^2 + 5) * (x - 5) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l755_75517


namespace NUMINAMATH_GPT_line_equation_midpoint_ellipse_l755_75543

theorem line_equation_midpoint_ellipse (x1 y1 x2 y2 : ℝ) 
  (h_midpoint_x : x1 + x2 = 4) (h_midpoint_y : y1 + y2 = 2)
  (h_ellipse_x1_y1 : (x1^2) / 12 + (y1^2) / 4 = 1) (h_ellipse_x2_y2 : (x2^2) / 12 + (y2^2) / 4 = 1) :
  2 * (x1 - x2) + 3 * (y1 - y2) = 0 :=
sorry

end NUMINAMATH_GPT_line_equation_midpoint_ellipse_l755_75543


namespace NUMINAMATH_GPT_sally_took_out_5_onions_l755_75587

theorem sally_took_out_5_onions (X Y : ℕ) 
    (h1 : 4 + 9 - Y + X = X + 8) : Y = 5 := 
by
  sorry

end NUMINAMATH_GPT_sally_took_out_5_onions_l755_75587


namespace NUMINAMATH_GPT_length_of_rooms_l755_75514

-- Definitions based on conditions
def width : ℕ := 18
def num_rooms : ℕ := 20
def total_area : ℕ := 6840

-- Theorem stating the length of the rooms
theorem length_of_rooms : (total_area / num_rooms) / width = 19 := by
  sorry

end NUMINAMATH_GPT_length_of_rooms_l755_75514


namespace NUMINAMATH_GPT_extreme_points_sum_gt_two_l755_75531

noncomputable def f (x : ℝ) (b : ℝ) := x^2 / 2 + b * Real.exp x
noncomputable def f_prime (x : ℝ) (b : ℝ) := x + b * Real.exp x

theorem extreme_points_sum_gt_two
  (b : ℝ)
  (h_b : -1 / Real.exp 1 < b ∧ b < 0)
  (x₁ x₂ : ℝ)
  (h_x₁ : f_prime x₁ b = 0)
  (h_x₂ : f_prime x₂ b = 0)
  (h_x₁_lt_x₂ : x₁ < x₂) :
  x₁ + x₂ > 2 := by
  sorry

end NUMINAMATH_GPT_extreme_points_sum_gt_two_l755_75531


namespace NUMINAMATH_GPT_payment_to_C_l755_75501

/-- 
If A can complete a work in 6 days, B can complete the same work in 8 days, 
they signed to do the work for Rs. 2400 and completed the work in 3 days with 
the help of C, then the payment to C should be Rs. 300.
-/
theorem payment_to_C (total_payment : ℝ) (days_A : ℝ) (days_B : ℝ) (days_worked : ℝ) (portion_C : ℝ) :
   total_payment = 2400 ∧ days_A = 6 ∧ days_B = 8 ∧ days_worked = 3 ∧ portion_C = 1 / 8 →
   (portion_C * total_payment) = 300 := 
by 
  intros h
  cases h
  sorry

end NUMINAMATH_GPT_payment_to_C_l755_75501


namespace NUMINAMATH_GPT_rectangle_area_increase_l755_75533

theorem rectangle_area_increase (b : ℕ) (h1 : 2 * b = 40) (h2 : b = 20) : 
  let l := 2 * b
  let A_original := l * b
  let l_new := l - 5
  let b_new := b + 5
  let A_new := l_new * b_new
  A_new - A_original = 75 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_increase_l755_75533


namespace NUMINAMATH_GPT_sequence_solution_l755_75589

theorem sequence_solution (a : ℕ → ℝ)
  (h₁ : a 1 = 0)
  (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 4 * (Real.sqrt (a n + 1)) + 4) :
  ∀ n ≥ 1, a n = 4 * n^2 - 4 * n :=
by
  sorry

end NUMINAMATH_GPT_sequence_solution_l755_75589


namespace NUMINAMATH_GPT_net_gain_A_correct_l755_75585

-- Define initial values and transactions
def initial_cash_A : ℕ := 20000
def house_value : ℕ := 20000
def car_value : ℕ := 5000
def initial_cash_B : ℕ := 25000
def house_sale_price : ℕ := 21000
def car_sale_price : ℕ := 4500
def house_repurchase_price : ℕ := 19000
def car_depreciation : ℕ := 10
def car_repurchase_price : ℕ := 4050

-- Define the final cash calculations
def final_cash_A := initial_cash_A + house_sale_price + car_sale_price - house_repurchase_price - car_repurchase_price
def final_cash_B := initial_cash_B - house_sale_price - car_sale_price + house_repurchase_price + car_repurchase_price

-- Define the net gain calculations
def net_gain_A := final_cash_A - initial_cash_A
def net_gain_B := final_cash_B - initial_cash_B

-- Theorem to prove
theorem net_gain_A_correct : net_gain_A = 2000 :=
by 
  -- Definitions and calculations would go here
  sorry

end NUMINAMATH_GPT_net_gain_A_correct_l755_75585


namespace NUMINAMATH_GPT_people_in_room_l755_75539

variable (total_chairs occupied_chairs people_present : ℕ)
variable (h1 : total_chairs = 28)
variable (h2 : occupied_chairs = 14)
variable (h3 : (2 / 3 : ℚ) * people_present = 14)
variable (h4 : total_chairs = 2 * occupied_chairs)

theorem people_in_room : people_present = 21 := 
by 
  --proof will be here
  sorry

end NUMINAMATH_GPT_people_in_room_l755_75539


namespace NUMINAMATH_GPT_T_10_mod_5_eq_3_l755_75566

def a_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in A
sorry

def b_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in B
sorry

def c_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in C
sorry

def T (n : ℕ) : ℕ := -- Number of valid sequences of length n
  a_n n + b_n n

theorem T_10_mod_5_eq_3 :
  T 10 % 5 = 3 :=
sorry

end NUMINAMATH_GPT_T_10_mod_5_eq_3_l755_75566


namespace NUMINAMATH_GPT_part_a_l755_75513

theorem part_a (cities : Finset (ℝ × ℝ)) (h_cities : cities.card = 100) 
  (distances : Finset (ℝ × ℝ → ℝ)) (h_distances : distances.card = 4950) :
  ∃ (erased_distance : ℝ × ℝ → ℝ), ¬ ∃ (restored_distance : ℝ × ℝ → ℝ), 
    restored_distance = erased_distance :=
sorry

end NUMINAMATH_GPT_part_a_l755_75513


namespace NUMINAMATH_GPT_interval_between_segments_systematic_sampling_l755_75546

theorem interval_between_segments_systematic_sampling 
  (total_students : ℕ) (sample_size : ℕ) 
  (h_total_students : total_students = 1000) 
  (h_sample_size : sample_size = 40):
  total_students / sample_size = 25 :=
by
  sorry

end NUMINAMATH_GPT_interval_between_segments_systematic_sampling_l755_75546


namespace NUMINAMATH_GPT_initial_apps_count_l755_75557

theorem initial_apps_count (x A : ℕ) 
  (h₁ : A - 18 + x = 5) : A = 23 - x :=
by
  sorry

end NUMINAMATH_GPT_initial_apps_count_l755_75557


namespace NUMINAMATH_GPT_range_of_m_l755_75538

theorem range_of_m (m : ℝ) : 
  (¬(-2 ≤ 1 - (x - 1) / 3 ∧ (1 - (x - 1) / 3 ≤ 2)) → (∀ x, m > 0 → x^2 - 2*x + 1 - m^2 > 0)) → 
  (40 ≤ m ∧ m < 50) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l755_75538


namespace NUMINAMATH_GPT_min_value_inequality_l755_75564

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 3^x + 9^y ≥ 2 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l755_75564


namespace NUMINAMATH_GPT_part_a_part_b_l755_75581

-- Part (a)
theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (a b : ℝ), a ≠ b ∧ 0 < (a - b) / (1 + a * b) ∧ (a - b) / (1 + a * b) ≤ 1 := sorry

-- Part (b)
theorem part_b (x y z u : ℝ) :
  ∃ (a b : ℝ), a ≠ b ∧ 0 < (b - a) / (1 + a * b) ∧ (b - a) / (1 + a * b) ≤ 1 := sorry

end NUMINAMATH_GPT_part_a_part_b_l755_75581


namespace NUMINAMATH_GPT_FastFoodCost_l755_75578

theorem FastFoodCost :
  let sandwich_cost := 4
  let soda_cost := 1.5
  let fries_cost := 2.5
  let num_sandwiches := 4
  let num_sodas := 6
  let num_fries := 3
  let discount := 5
  let total_cost := (sandwich_cost * num_sandwiches) + (soda_cost * num_sodas) + (fries_cost * num_fries) - discount
  total_cost = 27.5 := 
by
  sorry

end NUMINAMATH_GPT_FastFoodCost_l755_75578


namespace NUMINAMATH_GPT_solution_set_inequality_l755_75535

theorem solution_set_inequality :
  {x : ℝ | (x^2 + 4) / (x - 4)^2 ≥ 0} = {x | x < 4} ∪ {x | x > 4} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l755_75535


namespace NUMINAMATH_GPT_find_a_plus_b_l755_75512

variables {a b : ℝ}

theorem find_a_plus_b (h1 : a - b = -3) (h2 : a * b = 2) : a + b = Real.sqrt 17 ∨ a + b = -Real.sqrt 17 := by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l755_75512


namespace NUMINAMATH_GPT_one_and_two_thirds_eq_36_l755_75596

theorem one_and_two_thirds_eq_36 (x : ℝ) (h : (5 / 3) * x = 36) : x = 21.6 :=
sorry

end NUMINAMATH_GPT_one_and_two_thirds_eq_36_l755_75596


namespace NUMINAMATH_GPT_poly_division_l755_75575

noncomputable def A := 1
noncomputable def B := 3
noncomputable def C := 2
noncomputable def D := -1

theorem poly_division :
  (∀ x : ℝ, x ≠ -1 → (x^3 + 4*x^2 + 5*x + 2) / (x+1) = x^2 + 3*x + 2) ∧
  (A + B + C + D = 5) :=
by
  sorry

end NUMINAMATH_GPT_poly_division_l755_75575


namespace NUMINAMATH_GPT_union_complement_A_when_a_eq_1_A_cap_B_eq_A_range_of_a_l755_75518

def setA (a : ℝ) : Set ℝ := { x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3 }
def setB : Set ℝ := { x | -1 / 2 < x ∧ x < 2 }
def complementB : Set ℝ := { x | x ≤ -1 / 2 ∨ x ≥ 2 }

theorem union_complement_A_when_a_eq_1 :
  (complementB ∪ setA 1) = { x | x ≤ 1 ∨ x ≥ 2 } :=
by
  sorry

theorem A_cap_B_eq_A_range_of_a (a : ℝ) :
  (setA a ∩ setB = setA a) → (-1 < a ∧ a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_union_complement_A_when_a_eq_1_A_cap_B_eq_A_range_of_a_l755_75518


namespace NUMINAMATH_GPT_remainder_polynomial_l755_75558

theorem remainder_polynomial (n : ℕ) (hn : n ≥ 2) : 
  ∃ Q R : Polynomial ℤ, (R.degree < 2) ∧ (X^n = Q * (X^2 - 4 * X + 3) + R) ∧ 
                       (R = (Polynomial.C ((3^n - 1) / 2) * X + Polynomial.C ((3 - 3^n) / 2))) :=
by
  sorry

end NUMINAMATH_GPT_remainder_polynomial_l755_75558


namespace NUMINAMATH_GPT_range_of_a_l755_75534

variable (f : ℝ → ℝ)

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f y < f x

theorem range_of_a 
  (decreasing_f : is_decreasing f)
  (hfdef : ∀ x, -1 ≤ x ∧ x ≤ 1 → f (2 * x - 3) < f (x - 2)) :
  ∃ a : ℝ, 1 < a ∧ a ≤ 2  :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l755_75534


namespace NUMINAMATH_GPT_ratio_volumes_l755_75560

noncomputable def V_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

noncomputable def V_cone (r : ℝ) : ℝ := (1 / 3) * Real.pi * r^3

theorem ratio_volumes (r : ℝ) (hr : r > 0) : 
  (V_cone r) / (V_sphere r) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_volumes_l755_75560


namespace NUMINAMATH_GPT_cost_of_first_shirt_l755_75547

theorem cost_of_first_shirt (x : ℝ) (h1 : x + (x + 6) = 24) : x + 6 = 15 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_first_shirt_l755_75547


namespace NUMINAMATH_GPT_percentage_of_50_l755_75555

theorem percentage_of_50 (P : ℝ) :
  (0.10 * 30) + (P / 100 * 50) = 10.5 → P = 15 := by
  sorry

end NUMINAMATH_GPT_percentage_of_50_l755_75555


namespace NUMINAMATH_GPT_radii_touching_circles_l755_75554

noncomputable def radius_of_circles_touching_unit_circles 
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (centerA centerB centerC : A) 
  (unit_radius : ℝ) (h1 : dist centerA centerB = 2 * unit_radius) 
  (h2 : dist centerB centerC = 2 * unit_radius) (h3 : dist centerC centerA = 2 * unit_radius) 
  : Prop :=
  ∃ r₁ r₂ : ℝ, r₁ = 1/3 ∧ r₂ = 7/3

theorem radii_touching_circles (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (centerA centerB centerC : A)
  (unit_radius : ℝ) (h1 : dist centerA centerB = 2 * unit_radius)
  (h2 : dist centerB centerC = 2 * unit_radius) (h3 : dist centerC centerA = 2 * unit_radius)
  : radius_of_circles_touching_unit_circles A B C centerA centerB centerC unit_radius h1 h2 h3 :=
sorry

end NUMINAMATH_GPT_radii_touching_circles_l755_75554


namespace NUMINAMATH_GPT_solution_pairs_correct_l755_75591

theorem solution_pairs_correct:
  { (n, m) : ℕ × ℕ | m^2 + 2 * 3^n = m * (2^(n+1) - 1) }
  = {(3, 6), (3, 9), (6, 54), (6, 27)} :=
by
  sorry -- no proof is required as per the instruction

end NUMINAMATH_GPT_solution_pairs_correct_l755_75591


namespace NUMINAMATH_GPT_find_b_l755_75536

def f (x : ℝ) : ℝ := 5 * x + 3

theorem find_b : ∃ b : ℝ, f b = -2 ∧ b = -1 := by
  have h : 5 * (-1 : ℝ) + 3 = -2 := by norm_num
  use -1
  simp [f, h]
  sorry

end NUMINAMATH_GPT_find_b_l755_75536


namespace NUMINAMATH_GPT_probability_vowel_probability_consonant_probability_ch_l755_75598

def word := "дифференцициал"
def total_letters := 12
def num_vowels := 5
def num_consonants := 7
def num_letter_ch := 0

theorem probability_vowel : (num_vowels : ℚ) / total_letters = 5 / 12 := by
  sorry

theorem probability_consonant : (num_consonants : ℚ) / total_letters = 7 / 12 := by
  sorry

theorem probability_ch : (num_letter_ch : ℚ) / total_letters = 0 := by
  sorry

end NUMINAMATH_GPT_probability_vowel_probability_consonant_probability_ch_l755_75598


namespace NUMINAMATH_GPT_remaining_area_l755_75567

theorem remaining_area (x : ℝ) :
  let A_large := (2 * x + 8) * (x + 6)
  let A_hole := (3 * x - 4) * (x + 1)
  A_large - A_hole = - x^2 + 22 * x + 52 := by
  let A_large := (2 * x + 8) * (x + 6)
  let A_hole := (3 * x - 4) * (x + 1)
  have hA_large : A_large = 2 * x^2 + 20 * x + 48 := by
    sorry
  have hA_hole : A_hole = 3 * x^2 - 2 * x - 4 := by
    sorry
  calc
    A_large - A_hole = (2 * x^2 + 20 * x + 48) - (3 * x^2 - 2 * x - 4) := by
      rw [hA_large, hA_hole]
    _ = -x^2 + 22 * x + 52 := by
      ring

end NUMINAMATH_GPT_remaining_area_l755_75567


namespace NUMINAMATH_GPT_workers_count_l755_75520

noncomputable def numberOfWorkers (W: ℕ) : Prop :=
  let old_supervisor_salary := 870
  let new_supervisor_salary := 690
  let avg_old := 430
  let avg_new := 410
  let total_after_old := (W + 1) * avg_old
  let total_after_new := 9 * avg_new
  total_after_old - old_supervisor_salary = total_after_new - new_supervisor_salary

theorem workers_count : numberOfWorkers 8 :=
by
  sorry

end NUMINAMATH_GPT_workers_count_l755_75520


namespace NUMINAMATH_GPT_dress_designs_possible_l755_75527

theorem dress_designs_possible (colors patterns fabric_types : Nat) (color_choices : colors = 5) (pattern_choices : patterns = 6) (fabric_type_choices : fabric_types = 2) : 
  colors * patterns * fabric_types = 60 := by 
  sorry

end NUMINAMATH_GPT_dress_designs_possible_l755_75527


namespace NUMINAMATH_GPT_find_fraction_l755_75529

theorem find_fraction {a b : ℕ} 
  (h1 : 32016 + (a / b) = 2016 * 3 + (a / b)) 
  (ha : a = 2016) 
  (hb : b = 2016^3 - 1) : 
  (b + 1) / a^2 = 2016 := 
by 
  sorry

end NUMINAMATH_GPT_find_fraction_l755_75529


namespace NUMINAMATH_GPT_constant_seq_arith_geo_l755_75524

def is_arithmetic_sequence (s : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, s (n + 1) = s n + d

def is_geometric_sequence (s : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, s (n + 1) = s n * r

theorem constant_seq_arith_geo (s : ℕ → ℝ) (d r : ℝ) :
  is_arithmetic_sequence s d →
  is_geometric_sequence s r →
  (∃ c : ℝ, ∀ n : ℕ, s n = c) ∧ r = 1 :=
by
  sorry

end NUMINAMATH_GPT_constant_seq_arith_geo_l755_75524


namespace NUMINAMATH_GPT_combined_yearly_return_percentage_l755_75516

-- Given conditions
def investment1 : ℝ := 500
def return_rate1 : ℝ := 0.07
def investment2 : ℝ := 1500
def return_rate2 : ℝ := 0.15

-- Question to prove
theorem combined_yearly_return_percentage :
  let yearly_return1 := investment1 * return_rate1
  let yearly_return2 := investment2 * return_rate2
  let total_yearly_return := yearly_return1 + yearly_return2
  let total_investment := investment1 + investment2
  ((total_yearly_return / total_investment) * 100) = 13 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_combined_yearly_return_percentage_l755_75516


namespace NUMINAMATH_GPT_rhombus_perimeter_52_l755_75562

-- Define the conditions of the rhombus
def isRhombus (a b c d : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = d

def rhombus_diagonals (p q : ℝ) : Prop :=
  p = 10 ∧ q = 24

-- Define the perimeter calculation
def rhombus_perimeter (s : ℝ) : ℝ :=
  4 * s

-- Main theorem statement
theorem rhombus_perimeter_52 (p q s : ℝ)
  (h_diagonals : rhombus_diagonals p q)
  (h_rhombus : isRhombus s s s s)
  (h_side_length : s = 13) :
  rhombus_perimeter s = 52 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_52_l755_75562


namespace NUMINAMATH_GPT_smallest_n_mod_equiv_l755_75508

theorem smallest_n_mod_equiv (n : ℕ) (h : 5 * n ≡ 4960 [MOD 31]) : n = 31 := by 
  sorry

end NUMINAMATH_GPT_smallest_n_mod_equiv_l755_75508


namespace NUMINAMATH_GPT_interest_difference_l755_75550

theorem interest_difference (P R T: ℝ) (hP: P = 2500) (hR: R = 8) (hT: T = 8) :
  let I := P * R * T / 100
  (P - I = 900) :=
by
  -- definition of I
  let I := P * R * T / 100
  -- proof goal
  sorry

end NUMINAMATH_GPT_interest_difference_l755_75550


namespace NUMINAMATH_GPT_num_of_B_sets_l755_75519

def A : Set ℕ := {1, 2}

theorem num_of_B_sets (S : Set ℕ) (A : Set ℕ) (h : A = {1, 2}) (h1 : ∀ B : Set ℕ, A ∪ B = S) : 
  ∃ n : ℕ, n = 4 ∧ (∀ B : Set ℕ, B ⊆ {1, 2} → S = {1, 2}) :=
by {
  sorry
}

end NUMINAMATH_GPT_num_of_B_sets_l755_75519


namespace NUMINAMATH_GPT_value_of_c_l755_75542

variables (a b c : ℝ)

theorem value_of_c :
  a + b = 3 ∧
  a * c + b = 18 ∧
  b * c + a = 6 →
  c = 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_c_l755_75542


namespace NUMINAMATH_GPT_units_digit_of_6_to_the_6_l755_75553

theorem units_digit_of_6_to_the_6 : (6^6) % 10 = 6 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_6_to_the_6_l755_75553


namespace NUMINAMATH_GPT_subset_a_eq_1_l755_75593

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_a_eq_1 (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_subset_a_eq_1_l755_75593


namespace NUMINAMATH_GPT_circle_equation_unique_l755_75549

theorem circle_equation_unique {F D E : ℝ} : 
  (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧ 
  (∀ (x y : ℝ), (x = 1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧ 
  (∀ (x y : ℝ), (x = 4 ∧ y = 2) → x^2 + y^2 + D * x + E * y + F = 0) → 
  (x^2 + y^2 - 8 * x + 6 * y = 0) :=
by 
  sorry

end NUMINAMATH_GPT_circle_equation_unique_l755_75549
