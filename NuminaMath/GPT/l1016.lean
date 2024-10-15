import Mathlib

namespace NUMINAMATH_GPT_find_t_value_l1016_101606

theorem find_t_value (t : ℝ) (a b : ℝ × ℝ) (h₁ : a = (t, 1)) (h₂ : b = (1, 2)) 
  (h₃ : (a.1 + b.1)^2 + (a.2 + b.2)^2 = a.1^2 + a.2^2 + b.1^2 + b.2^2) : 
  t = -2 :=
by 
  sorry

end NUMINAMATH_GPT_find_t_value_l1016_101606


namespace NUMINAMATH_GPT_solve_a1_solve_a2_l1016_101601

noncomputable def initial_volume := 1  -- in m^3
noncomputable def initial_pressure := 10^5  -- in Pa
noncomputable def initial_temperature := 300  -- in K

theorem solve_a1 (a1 : ℝ) : a1 = -10^5 :=
  sorry

theorem solve_a2 (a2 : ℝ) : a2 = -1.4 * 10^5 :=
  sorry

end NUMINAMATH_GPT_solve_a1_solve_a2_l1016_101601


namespace NUMINAMATH_GPT_rise_in_water_level_correct_l1016_101624

noncomputable def volume_of_rectangular_solid (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def area_of_circular_base (d : ℝ) : ℝ :=
  Real.pi * (d / 2) ^ 2

noncomputable def rise_in_water_level (solid_volume base_area : ℝ) : ℝ :=
  solid_volume / base_area

theorem rise_in_water_level_correct :
  let l := 10
  let w := 12
  let h := 15
  let d := 18
  let solid_volume := volume_of_rectangular_solid l w h
  let base_area := area_of_circular_base d
  let expected_rise := 7.07
  abs (rise_in_water_level solid_volume base_area - expected_rise) < 0.01 
:= 
by {
  sorry
}

end NUMINAMATH_GPT_rise_in_water_level_correct_l1016_101624


namespace NUMINAMATH_GPT_present_age_of_dan_l1016_101672

theorem present_age_of_dan (x : ℕ) : (x + 16 = 4 * (x - 8)) → x = 16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_present_age_of_dan_l1016_101672


namespace NUMINAMATH_GPT_incorrect_statement_C_l1016_101682

theorem incorrect_statement_C :
  (∀ r : ℚ, ∃ p : ℝ, p = r) ∧  -- Condition A: All rational numbers can be represented by points on the number line.
  (∀ x : ℝ, x = 1 / x → x = 1 ∨ x = -1) ∧  -- Condition B: The reciprocal of a number equal to itself is ±1.
  (∀ f : ℚ, ∃ q : ℝ, q = f) →  -- Condition C (negation of C as presented): Fractions cannot be represented by points on the number line.
  (∀ x : ℝ, abs x ≥ 0) ∧ (∀ x : ℝ, abs x = 0 ↔ x = 0) →  -- Condition D: The number with the smallest absolute value is 0.
  false :=                      -- Prove that statement C is incorrect
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_C_l1016_101682


namespace NUMINAMATH_GPT_inequality_holds_for_a_l1016_101658

theorem inequality_holds_for_a (a : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < -1 → (x + 1)^2 < Real.logb a (|x|)) ↔ 1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_a_l1016_101658


namespace NUMINAMATH_GPT_count_integer_length_chords_l1016_101623

/-- Point P is 9 units from the center of a circle with radius 15. -/
def point_distance_from_center : ℝ := 9

def circle_radius : ℝ := 15

/-- Correct answer to the number of different chords that contain P and have integer lengths. -/
def correct_answer : ℕ := 7

/-- Proving the number of chords containing P with integer lengths given the conditions. -/
theorem count_integer_length_chords : 
  ∀ (r_P : ℝ) (r_circle : ℝ), r_P = point_distance_from_center → r_circle = circle_radius → 
  (∃ n : ℕ, n = correct_answer) :=
by 
  intros r_P r_circle h1 h2
  use 7 
  sorry

end NUMINAMATH_GPT_count_integer_length_chords_l1016_101623


namespace NUMINAMATH_GPT_pollen_mass_in_scientific_notation_l1016_101645

theorem pollen_mass_in_scientific_notation : 
  ∃ c n : ℝ, 0.0000037 = c * 10^n ∧ 1 ≤ c ∧ c < 10 ∧ c = 3.7 ∧ n = -6 :=
sorry

end NUMINAMATH_GPT_pollen_mass_in_scientific_notation_l1016_101645


namespace NUMINAMATH_GPT_metallic_weight_problem_l1016_101614

variables {m1 m2 m3 m4 : ℝ}

theorem metallic_weight_problem
  (h_total : m1 + m2 + m3 + m4 = 35)
  (h1 : m1 = 1.5 * m2)
  (h2 : m2 = (3/4) * m3)
  (h3 : m3 = (5/6) * m4) :
  m4 = 105 / 13 :=
sorry

end NUMINAMATH_GPT_metallic_weight_problem_l1016_101614


namespace NUMINAMATH_GPT_triangle_with_angle_ratio_obtuse_l1016_101688

theorem triangle_with_angle_ratio_obtuse 
  (a b c : ℝ) 
  (h_sum : a + b + c = 180) 
  (h_ratio : a = 2 * d ∧ b = 2 * d ∧ c = 5 * d) : 
  90 < c :=
by
  sorry

end NUMINAMATH_GPT_triangle_with_angle_ratio_obtuse_l1016_101688


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1016_101666

noncomputable def Sn (a d n : ℕ) : ℕ :=
n * a + (n * (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a d : ℕ) (h1 : a = 3 * d) (h2 : Sn a d 5 = 50) : Sn a d 8 = 104 :=
by
/-
  From the given conditions:
  - \(a_4\) is the geometric mean of \(a_2\) and \(a_7\) implies \(a = 3d\)
  - Sum of first 5 terms is 50 implies \(S_5 = 50\)
  We need to prove \(S_8 = 104\)
-/
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1016_101666


namespace NUMINAMATH_GPT_greatest_possible_a_l1016_101634

theorem greatest_possible_a (a : ℤ) (x : ℤ) (h_pos : 0 < a) (h_eq : x^3 + a * x^2 = -30) : 
  a ≤ 29 :=
sorry

end NUMINAMATH_GPT_greatest_possible_a_l1016_101634


namespace NUMINAMATH_GPT_length_of_ae_l1016_101664

theorem length_of_ae
  (a b c d e : ℝ)
  (bc : ℝ)
  (cd : ℝ)
  (de : ℝ := 8)
  (ab : ℝ := 5)
  (ac : ℝ := 11)
  (h1 : bc = 2 * cd)
  (h2 : bc = ac - ab)
  : ab + bc + cd + de = 22 := 
by
  sorry

end NUMINAMATH_GPT_length_of_ae_l1016_101664


namespace NUMINAMATH_GPT_range_of_f_l1016_101607

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sqrt (5 + 4 * Real.cos x))

theorem range_of_f :
  Set.range f = Set.Icc (-1/2 : ℝ) (1/2 : ℝ) := 
sorry

end NUMINAMATH_GPT_range_of_f_l1016_101607


namespace NUMINAMATH_GPT_actors_per_group_l1016_101600

theorem actors_per_group (actors_per_hour : ℕ) (show_time_per_actor : ℕ) (total_show_time : ℕ)
  (h1 : show_time_per_actor = 15) (h2 : actors_per_hour = 20) (h3 : total_show_time = 60) :
  actors_per_hour * show_time_per_actor / total_show_time = 5 :=
by sorry

end NUMINAMATH_GPT_actors_per_group_l1016_101600


namespace NUMINAMATH_GPT_scientific_notation_104000000_l1016_101621

theorem scientific_notation_104000000 :
  104000000 = 1.04 * 10^8 :=
sorry

end NUMINAMATH_GPT_scientific_notation_104000000_l1016_101621


namespace NUMINAMATH_GPT_number_of_solutions_l1016_101641

open Real

theorem number_of_solutions :
  ∀ x : ℝ, (0 < x ∧ x < 3 * π) → (3 * cos x ^ 2 + 2 * sin x ^ 2 = 2) → 
  ∃ (L : Finset ℝ), L.card = 3 ∧ ∀ y ∈ L, 0 < y ∧ y < 3 * π ∧ 3 * cos y ^ 2 + 2 * sin y ^ 2 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_solutions_l1016_101641


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l1016_101622

theorem no_real_roots_of_quadratic 
  (a b c : ℝ) 
  (h1 : b - a + c > 0) 
  (h2 : b + a - c > 0) 
  (h3 : b - a - c < 0) 
  (h4 : b + a + c > 0) 
  (x : ℝ) : ¬ ∃ x : ℝ, a^2 * x^2 + (b^2 - a^2 - c^2) * x + c^2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l1016_101622


namespace NUMINAMATH_GPT_tan_rewrite_l1016_101693

open Real

theorem tan_rewrite (α β : ℝ) 
  (h1 : tan (α + β) = 2 / 5)
  (h2 : tan (β - π / 4) = 1 / 4) : 
  (1 + tan α) / (1 - tan α) = 3 / 22 := 
by
  sorry

end NUMINAMATH_GPT_tan_rewrite_l1016_101693


namespace NUMINAMATH_GPT_penny_makes_total_revenue_l1016_101638

def price_per_slice : ℕ := 7
def slices_per_pie : ℕ := 6
def pies_sold : ℕ := 7

theorem penny_makes_total_revenue :
  (pies_sold * slices_per_pie) * price_per_slice = 294 := by
  sorry

end NUMINAMATH_GPT_penny_makes_total_revenue_l1016_101638


namespace NUMINAMATH_GPT_chord_length_l1016_101667

theorem chord_length (x y t : ℝ) (h₁ : x = 1 + 2 * t) (h₂ : y = 2 + t) (h_circle : x^2 + y^2 = 9) : 
  ∃ l, l = 12 / 5 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_chord_length_l1016_101667


namespace NUMINAMATH_GPT_constant_seq_decreasing_implication_range_of_values_l1016_101616

noncomputable def sequences (a b : ℕ → ℝ) := 
  (∀ n, a (n+1) = (1/2) * a n + (1/2) * b n) ∧
  (∀ n, (1/b (n+1)) = (1/2) * (1/a n) + (1/2) * (1/b n))

theorem constant_seq (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 > 0) (h_b1 : b 1 > 0) :
  ∃ c, ∀ n, a n * b n = c :=
sorry

theorem decreasing_implication (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 > 0) (h_b1 : b 1 > 0) (h_dec : ∀ n, a (n+1) < a n) :
  a 1 > b 1 :=
sorry

theorem range_of_values (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 = 4) (h_b1 : b 1 = 1) :
  ∀ n ≥ 2, 2 < a n ∧ a n ≤ 5/2 :=
sorry

end NUMINAMATH_GPT_constant_seq_decreasing_implication_range_of_values_l1016_101616


namespace NUMINAMATH_GPT_total_chairs_agreed_proof_l1016_101633

/-
Conditions:
- Carey moved 28 chairs
- Pat moved 29 chairs
- They have 17 chairs left to move
Question:
- How many chairs did they agree to move in total?
Proof Problem:
- Prove that the total number of chairs they agreed to move is equal to 74.
-/

def carey_chairs : ℕ := 28
def pat_chairs : ℕ := 29
def chairs_left : ℕ := 17
def total_chairs_agreed : ℕ := carey_chairs + pat_chairs + chairs_left

theorem total_chairs_agreed_proof : total_chairs_agreed = 74 := 
by
  sorry

end NUMINAMATH_GPT_total_chairs_agreed_proof_l1016_101633


namespace NUMINAMATH_GPT_greatest_multiple_of_4_less_than_100_l1016_101695

theorem greatest_multiple_of_4_less_than_100 : ∃ n : ℕ, n % 4 = 0 ∧ n < 100 ∧ ∀ m : ℕ, (m % 4 = 0 ∧ m < 100) → m ≤ n 
:= by
  sorry

end NUMINAMATH_GPT_greatest_multiple_of_4_less_than_100_l1016_101695


namespace NUMINAMATH_GPT_find_second_number_l1016_101608

theorem find_second_number (a b c : ℕ) (h1 : a = 5 * x) (h2 : b = 3 * x) (h3 : c = 4 * x) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l1016_101608


namespace NUMINAMATH_GPT_find_first_term_l1016_101615

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

variable (a1 a3 a9 d : ℤ)

-- Given conditions
axiom h1 : arithmetic_seq a1 d 2 = 30
axiom h2 : arithmetic_seq a1 d 8 = 60

theorem find_first_term : a1 = 20 :=
by
  -- mathematical proof steps here
  sorry

end NUMINAMATH_GPT_find_first_term_l1016_101615


namespace NUMINAMATH_GPT_quadratic_roots_opposite_signs_l1016_101665

theorem quadratic_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, (a * x^2 - (a + 3) * x + 2 = 0) ∧ (a * y^2 - (a + 3) * y + 2 = 0) ∧ x * y < 0) ↔ (a < 0) :=
sorry

end NUMINAMATH_GPT_quadratic_roots_opposite_signs_l1016_101665


namespace NUMINAMATH_GPT_Kendra_weekly_words_not_determined_without_weeks_l1016_101630

def Kendra_goal : Nat := 60
def Kendra_already_learned : Nat := 36
def Kendra_needs_to_learn : Nat := 24

theorem Kendra_weekly_words_not_determined_without_weeks (weeks : Option Nat) : weeks = none → Kendra_needs_to_learn / weeks.getD 1 = 24 -> False := by
  sorry

end NUMINAMATH_GPT_Kendra_weekly_words_not_determined_without_weeks_l1016_101630


namespace NUMINAMATH_GPT_first_player_wins_l1016_101625

def wins (sum_rows sum_cols : ℕ) : Prop := sum_rows > sum_cols

theorem first_player_wins 
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ)
  (h : a_1 > a_2 ∧ a_2 > a_3 ∧ a_3 > a_4 ∧ a_4 > a_5 ∧ a_5 > a_6 ∧ a_6 > a_7 ∧ a_7 > a_8 ∧ a_8 > a_9) :
  ∃ sum_rows sum_cols, wins sum_rows sum_cols :=
sorry

end NUMINAMATH_GPT_first_player_wins_l1016_101625


namespace NUMINAMATH_GPT_triangle_inequality_l1016_101674

theorem triangle_inequality (a b c : ℝ) (habc_triangle : a + b > c ∧ b + c > a ∧ a + c > b) : 
  2 * (a^2 * b^2 + b^2 * c^2 + a^2 * c^2) > (a^4 + b^4 + c^4) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1016_101674


namespace NUMINAMATH_GPT_bishop_safe_squares_l1016_101611

def chessboard_size : ℕ := 64
def total_squares_removed_king : ℕ := chessboard_size - 1
def threat_squares : ℕ := 7

theorem bishop_safe_squares : total_squares_removed_king - threat_squares = 30 :=
by
  sorry

end NUMINAMATH_GPT_bishop_safe_squares_l1016_101611


namespace NUMINAMATH_GPT_pencils_in_drawer_l1016_101626

theorem pencils_in_drawer (P : ℕ) 
  (h1 : 19 + 16 = 35)
  (h2 : P + 35 = 78) : 
  P = 43 := 
by
  sorry

end NUMINAMATH_GPT_pencils_in_drawer_l1016_101626


namespace NUMINAMATH_GPT_multiply_powers_same_base_l1016_101647

theorem multiply_powers_same_base (a : ℝ) : a^4 * a^2 = a^6 := by
  sorry

end NUMINAMATH_GPT_multiply_powers_same_base_l1016_101647


namespace NUMINAMATH_GPT_smallest_b_for_composite_l1016_101640

theorem smallest_b_for_composite (x : ℤ) : 
  ∃ b : ℕ, b > 0 ∧ Even b ∧ (∀ x : ℤ, ¬ Prime (x^4 + b^2)) ∧ b = 16 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_b_for_composite_l1016_101640


namespace NUMINAMATH_GPT_fence_perimeter_l1016_101619

theorem fence_perimeter 
  (N : ℕ) (w : ℝ) (g : ℝ) 
  (square_posts : N = 36) 
  (post_width : w = 0.5) 
  (gap_length : g = 8) :
  4 * ((N / 4 - 1) * g + (N / 4) * w) = 274 :=
by
  sorry

end NUMINAMATH_GPT_fence_perimeter_l1016_101619


namespace NUMINAMATH_GPT_arthur_hot_dogs_first_day_l1016_101661

theorem arthur_hot_dogs_first_day (H D n : ℕ) (h₀ : D = 1)
(h₁ : 3 * H + n = 10)
(h₂ : 2 * H + 3 * D = 7) : n = 4 :=
by sorry

end NUMINAMATH_GPT_arthur_hot_dogs_first_day_l1016_101661


namespace NUMINAMATH_GPT_find_quadruple_l1016_101687

/-- Problem Statement:
Given distinct positive integers a, b, c, and d such that a + b = c * d and a * b = c + d,
find the quadruple (a, b, c, d) that meets these conditions.
-/

theorem find_quadruple :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
            0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
            (a + b = c * d) ∧ (a * b = c + d) ∧
            ((a, b, c, d) = (1, 5, 3, 2) ∨ (a, b, c, d) = (1, 5, 2, 3) ∨
             (a, b, c, d) = (5, 1, 3, 2) ∨ (a, b, c, d) = (5, 1, 2, 3) ∨
             (a, b, c, d) = (2, 3, 1, 5) ∨ (a, b, c, d) = (3, 2, 1, 5) ∨
             (a, b, c, d) = (2, 3, 5, 1) ∨ (a, b, c, d) = (3, 2, 5, 1)) :=
sorry

end NUMINAMATH_GPT_find_quadruple_l1016_101687


namespace NUMINAMATH_GPT_range_of_f_2x_le_1_l1016_101660

-- Given conditions
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

def cond_f_neg_2_eq_1 (f : ℝ → ℝ) : Prop :=
  f (-2) = 1

-- Main theorem
theorem range_of_f_2x_le_1 (f : ℝ → ℝ) 
  (h1 : is_odd f)
  (h2 : is_monotonically_decreasing f (Set.Iic 0))
  (h3 : cond_f_neg_2_eq_1 f) :
  Set.Icc (-1 : ℝ) 1 = { x | |f (2 * x)| ≤ 1 } :=
sorry

end NUMINAMATH_GPT_range_of_f_2x_le_1_l1016_101660


namespace NUMINAMATH_GPT_max_value_x_plus_2y_l1016_101696

variable (x y : ℝ)
variable (h1 : 4 * x + 3 * y ≤ 12)
variable (h2 : 3 * x + 6 * y ≤ 9)

theorem max_value_x_plus_2y : x + 2 * y ≤ 3 := by
  sorry

end NUMINAMATH_GPT_max_value_x_plus_2y_l1016_101696


namespace NUMINAMATH_GPT_find_varphi_l1016_101692

theorem find_varphi (ϕ : ℝ) (h1 : 0 < ϕ) (h2 : ϕ < π)
(h_symm : ∃ k : ℤ, ϕ = k * π + 2 * π / 3) :
ϕ = 2 * π / 3 :=
sorry

end NUMINAMATH_GPT_find_varphi_l1016_101692


namespace NUMINAMATH_GPT_isabella_more_than_giselle_l1016_101627

variables (I S G : ℕ)

def isabella_has_more_than_sam : Prop := I = S + 45
def giselle_amount : Prop := G = 120
def total_amount : Prop := I + S + G = 345

theorem isabella_more_than_giselle
  (h1 : isabella_has_more_than_sam I S)
  (h2 : giselle_amount G)
  (h3 : total_amount I S G) :
  I - G = 15 :=
by
  sorry

end NUMINAMATH_GPT_isabella_more_than_giselle_l1016_101627


namespace NUMINAMATH_GPT_two_point_question_count_l1016_101663

/-- Define the number of questions and points on the test,
    and prove that the number of 2-point questions is 30. -/
theorem two_point_question_count (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 := by
  sorry

end NUMINAMATH_GPT_two_point_question_count_l1016_101663


namespace NUMINAMATH_GPT_part_a_7_pieces_l1016_101686

theorem part_a_7_pieces (grid : Fin 4 × Fin 4 → Prop) (h : ∀ i j, ∃ n, grid (i, j) → n < 7)
  (hnoTwoInSameCell : ∀ (i₁ i₂ : Fin 4) (j₁ j₂ : Fin 4), (i₁, j₁) ≠ (i₂, j₂) → grid (i₁, j₁) ≠ grid (i₂, j₂))
  : ∀ (rowsRemoved colsRemoved : Finset (Fin 4)), rowsRemoved.card = 2 → colsRemoved.card = 2
    → ∃ i j, ¬ grid (i, j) := by sorry

end NUMINAMATH_GPT_part_a_7_pieces_l1016_101686


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1016_101685

noncomputable def hyperbola (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def foci_condition (a b : ℝ) (c : ℝ) : Prop :=
  c = Real.sqrt (a^2 + b^2)

noncomputable def trisection_condition (a b c : ℝ) : Prop :=
  2 * c = 6 * a^2 / c

theorem eccentricity_of_hyperbola (a b c e : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hc : c = Real.sqrt (a^2 + b^2)) (ht : 2 * c = 6 * a^2 / c) :
  e = Real.sqrt 3 :=
by
  apply sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1016_101685


namespace NUMINAMATH_GPT_pythagorean_triple_B_l1016_101670

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_B : isPythagoreanTriple 3 4 5 :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_triple_B_l1016_101670


namespace NUMINAMATH_GPT_correct_sample_size_l1016_101698

variable {StudentScore : Type} {scores : Finset StudentScore} (extract_sample : Finset StudentScore → Finset StudentScore)

noncomputable def is_correct_statement : Prop :=
  ∀ (total_scores : Finset StudentScore) (sample_scores : Finset StudentScore),
  (total_scores.card = 1000) →
  (extract_sample total_scores = sample_scores) →
  (sample_scores.card = 100) →
  sample_scores.card = 100

theorem correct_sample_size (total_scores sample_scores : Finset StudentScore)
  (H_total : total_scores.card = 1000)
  (H_sample : extract_sample total_scores = sample_scores)
  (H_card : sample_scores.card = 100) :
  sample_scores.card = 100 :=
sorry

end NUMINAMATH_GPT_correct_sample_size_l1016_101698


namespace NUMINAMATH_GPT_mario_haircut_price_l1016_101655

theorem mario_haircut_price (P : ℝ) 
  (weekend_multiplier : ℝ := 1.50)
  (sunday_price : ℝ := 27) 
  (weekend_price_eq : sunday_price = P * weekend_multiplier) : 
  P = 18 := 
by
  sorry

end NUMINAMATH_GPT_mario_haircut_price_l1016_101655


namespace NUMINAMATH_GPT_max_super_bishops_l1016_101681

/--
A "super-bishop" attacks another "super-bishop" if they are on the
same diagonal, there are no pieces between them, and the next cell
along the diagonal after the "super-bishop" B is empty. Given these
conditions, prove that the maximum number of "super-bishops" that can
be placed on a standard 8x8 chessboard such that each one attacks at
least one other is 32.
-/
theorem max_super_bishops (n : ℕ) (chessboard : ℕ → ℕ → Prop) (super_bishop : ℕ → ℕ → Prop)
  (attacks : ∀ {x₁ y₁ x₂ y₂}, super_bishop x₁ y₁ → super_bishop x₂ y₂ →
            (x₁ - x₂ = y₁ - y₂ ∨ x₁ + y₁ = x₂ + y₂) →
            (∀ x y, super_bishop x y → (x < min x₁ x₂ ∨ x > max x₁ x₂ ∨ y < min y₁ y₂ ∨ y > max y₁ y₂)) →
            chessboard (x₂ + (x₁ - x₂)) (y₂ + (y₁ - y₂))) :
  ∃ k, k = 32 ∧ (∀ x y, super_bishop x y → x < 8 ∧ y < 8) → k ≤ n :=
sorry

end NUMINAMATH_GPT_max_super_bishops_l1016_101681


namespace NUMINAMATH_GPT_range_of_m_l1016_101649

variable (m : ℝ)
def p : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h : p m ∧ q m) : -2 < m ∧ m < 0 := sorry

end NUMINAMATH_GPT_range_of_m_l1016_101649


namespace NUMINAMATH_GPT_pennies_thrown_total_l1016_101654

theorem pennies_thrown_total (rachelle_pennies gretchen_pennies rocky_pennies : ℕ) 
  (h1 : rachelle_pennies = 180)
  (h2 : gretchen_pennies = rachelle_pennies / 2)
  (h3 : rocky_pennies = gretchen_pennies / 3) : 
  rachelle_pennies + gretchen_pennies + rocky_pennies = 300 := 
by 
  sorry

end NUMINAMATH_GPT_pennies_thrown_total_l1016_101654


namespace NUMINAMATH_GPT_prime_add_eq_2001_l1016_101639

theorem prime_add_eq_2001 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^2 + b = 2003) : a + b = 2001 :=
sorry

end NUMINAMATH_GPT_prime_add_eq_2001_l1016_101639


namespace NUMINAMATH_GPT_ellipse_standard_equation_l1016_101677

theorem ellipse_standard_equation (c a : ℝ) (h1 : 2 * c = 8) (h2 : 2 * a = 10) : 
  (∃ b : ℝ, b^2 = a^2 - c^2 ∧ ( ( ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ) ∨ ( ∀ x y : ℝ, x^2 / b^2 + y^2 / a^2 = 1 ) )) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_standard_equation_l1016_101677


namespace NUMINAMATH_GPT_parallel_line_slope_l1016_101651

theorem parallel_line_slope {x y : ℝ} (h : 3 * x + 6 * y = -24) : 
  ∀ m b : ℝ, (y = m * x + b) → m = -1 / 2 :=
sorry

end NUMINAMATH_GPT_parallel_line_slope_l1016_101651


namespace NUMINAMATH_GPT_find_a4_l1016_101605

open Nat

def seq (a : ℕ → ℝ) := (a 1 = 1) ∧ (∀ n : ℕ, a (n + 1) = (2 * a n) / (a n + 2))

theorem find_a4 (a : ℕ → ℝ) (h : seq a) : a 4 = 2 / 5 :=
  sorry

end NUMINAMATH_GPT_find_a4_l1016_101605


namespace NUMINAMATH_GPT_unique_functional_equation_solution_l1016_101603

theorem unique_functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_GPT_unique_functional_equation_solution_l1016_101603


namespace NUMINAMATH_GPT_vans_hold_people_per_van_l1016_101646

theorem vans_hold_people_per_van (students adults vans total_people people_per_van : ℤ) 
    (h1: students = 12) 
    (h2: adults = 3) 
    (h3: vans = 3) 
    (h4: total_people = students + adults) 
    (h5: people_per_van = total_people / vans) :
    people_per_van = 5 := 
by
    -- Steps will go here
    sorry

end NUMINAMATH_GPT_vans_hold_people_per_van_l1016_101646


namespace NUMINAMATH_GPT_shaded_area_of_squares_l1016_101642

theorem shaded_area_of_squares :
  let s_s := 4
  let s_L := 9
  let area_L := s_L * s_L
  let area_s := s_s * s_s
  area_L - area_s = 65 := sorry

end NUMINAMATH_GPT_shaded_area_of_squares_l1016_101642


namespace NUMINAMATH_GPT_linear_regression_decrease_l1016_101684

theorem linear_regression_decrease (x : ℝ) (y : ℝ) :
  (h : ∃ c₀ c₁, (c₀ = 2) ∧ (c₁ = -1.5) ∧ y = c₀ - c₁ * x) →
  ( ∃ Δx, Δx = 1 → ∃ Δy, Δy = -1.5) :=
by 
  sorry

end NUMINAMATH_GPT_linear_regression_decrease_l1016_101684


namespace NUMINAMATH_GPT_solution_set_ln_inequality_l1016_101691

noncomputable def f (x : ℝ) := Real.cos x - 4 * x^2

theorem solution_set_ln_inequality :
  {x : ℝ | 0 < x ∧ x < Real.exp (-Real.pi / 2)} ∪ {x : ℝ | x > Real.exp (Real.pi / 2)} =
  {x : ℝ | f (Real.log x) + Real.pi^2 > 0} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_ln_inequality_l1016_101691


namespace NUMINAMATH_GPT_three_powers_in_two_digit_range_l1016_101659

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_three_powers_in_two_digit_range_l1016_101659


namespace NUMINAMATH_GPT_inequality_solution_set_inequality_proof_2_l1016_101628

theorem inequality_solution_set : 
  { x : ℝ | |x + 1| + |x + 3| < 4 } = { x : ℝ | -4 < x ∧ x < 0 } :=
sorry

theorem inequality_proof_2 (a b : ℝ) (ha : -4 < a) (ha' : a < 0) (hb : -4 < b) (hb' : b < 0) : 
  2 * |a - b| < |a * b + 2 * a + 2 * b| :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_inequality_proof_2_l1016_101628


namespace NUMINAMATH_GPT_even_and_odd_implies_zero_l1016_101679

theorem even_and_odd_implies_zero (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = -f x) (h2 : ∀ x : ℝ, f (-x) = f x) : ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_even_and_odd_implies_zero_l1016_101679


namespace NUMINAMATH_GPT_earliest_time_meet_l1016_101631

open Nat

def lap_time_anna := 5
def lap_time_bob := 8
def lap_time_carol := 10

def lcm_lap_times : ℕ :=
  Nat.lcm lap_time_anna (Nat.lcm lap_time_bob lap_time_carol)

theorem earliest_time_meet : lcm_lap_times = 40 := by
  sorry

end NUMINAMATH_GPT_earliest_time_meet_l1016_101631


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l1016_101644

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (Real.sqrt (2 * x - x ^ 2))

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → 1 ≤ x ∧ x < 2 → ∀ x1 x2, x1 < x2 → f x1 ≤ f x2 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l1016_101644


namespace NUMINAMATH_GPT_sally_oscillation_distance_l1016_101648

noncomputable def C : ℝ := 5 / 4
noncomputable def D : ℝ := 11 / 4

theorem sally_oscillation_distance :
  abs (C - D) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sally_oscillation_distance_l1016_101648


namespace NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l1016_101673

-- Definition for the conditions
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The theorem stating the proof problem
theorem smallest_prime_with_digit_sum_23 : ∃ p : ℕ, Prime p ∧ sum_of_digits p = 23 ∧ p = 1993 := 
by {
 sorry
}

end NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l1016_101673


namespace NUMINAMATH_GPT_value_range_of_quadratic_l1016_101676

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem value_range_of_quadratic :
  ∀ x, -1 ≤ x ∧ x ≤ 2 → (2 : ℝ) ≤ quadratic_function x ∧ quadratic_function x ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_value_range_of_quadratic_l1016_101676


namespace NUMINAMATH_GPT_minimum_value_2a_plus_3b_is_25_l1016_101669

noncomputable def minimum_value_2a_plus_3b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (2 / a) + (3 / b) = 1) : ℝ :=
2 * a + 3 * b

theorem minimum_value_2a_plus_3b_is_25
  (a b : ℝ)
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : (2 / a) + (3 / b) = 1) :
  minimum_value_2a_plus_3b a b h₁ h₂ h₃ = 25 :=
sorry

end NUMINAMATH_GPT_minimum_value_2a_plus_3b_is_25_l1016_101669


namespace NUMINAMATH_GPT_product_remainder_mod_7_l1016_101683

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end NUMINAMATH_GPT_product_remainder_mod_7_l1016_101683


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_subset_l1016_101671

variable {A B : Set ℕ}
variable {a : ℕ}

theorem sufficient_but_not_necessary_condition_for_subset (hA : A = {1, a}) (hB : B = {1, 2, 3}) :
  (a = 3 → A ⊆ B) ∧ (A ⊆ B → (a = 3 ∨ a = 2)) ∧ ¬(A ⊆ B → a = 3) := by
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_subset_l1016_101671


namespace NUMINAMATH_GPT_y_is_multiple_of_2_y_is_multiple_of_3_y_is_multiple_of_6_y_is_multiple_of_9_l1016_101629

def y : ℕ := 54 + 108 + 162 + 216 + 648 + 810 + 972

theorem y_is_multiple_of_2 : 2 ∣ y :=
sorry

theorem y_is_multiple_of_3 : 3 ∣ y :=
sorry

theorem y_is_multiple_of_6 : 6 ∣ y :=
sorry

theorem y_is_multiple_of_9 : 9 ∣ y :=
sorry

end NUMINAMATH_GPT_y_is_multiple_of_2_y_is_multiple_of_3_y_is_multiple_of_6_y_is_multiple_of_9_l1016_101629


namespace NUMINAMATH_GPT_problem_value_l1016_101662

theorem problem_value:
  3^(1+3+4) - (3^1 + 3^3 + 3^4) = 6450 :=
by sorry

end NUMINAMATH_GPT_problem_value_l1016_101662


namespace NUMINAMATH_GPT_value_of_2alpha_minus_beta_l1016_101690

theorem value_of_2alpha_minus_beta (a β : ℝ) (h1 : 3 * Real.sin a - Real.cos a = 0) 
    (h2 : 7 * Real.sin β + Real.cos β = 0) (h3 : 0 < a ∧ a < Real.pi / 2) 
    (h4 : Real.pi / 2 < β ∧ β < Real.pi) : 
    2 * a - β = -3 * Real.pi / 4 := 
sorry

end NUMINAMATH_GPT_value_of_2alpha_minus_beta_l1016_101690


namespace NUMINAMATH_GPT_dice_sum_eight_dice_l1016_101637

/--
  Given 8 fair 6-sided dice, prove that the number of ways to obtain
  a sum of 11 on the top faces of these dice, is 120.
-/
theorem dice_sum_eight_dice :
  (∃ n : ℕ, ∀ (dices : List ℕ), (dices.length = 8 ∧ (∀ d ∈ dices, 1 ≤ d ∧ d ≤ 6) 
   ∧ dices.sum = 11) → n = 120) :=
sorry

end NUMINAMATH_GPT_dice_sum_eight_dice_l1016_101637


namespace NUMINAMATH_GPT_annual_earning_difference_l1016_101617

def old_hourly_wage := 16
def old_weekly_hours := 25
def new_hourly_wage := 20
def new_weekly_hours := 40
def weeks_per_year := 52

def old_weekly_earnings := old_hourly_wage * old_weekly_hours
def new_weekly_earnings := new_hourly_wage * new_weekly_hours

def old_annual_earnings := old_weekly_earnings * weeks_per_year
def new_annual_earnings := new_weekly_earnings * weeks_per_year

theorem annual_earning_difference:
  new_annual_earnings - old_annual_earnings = 20800 := by
  sorry

end NUMINAMATH_GPT_annual_earning_difference_l1016_101617


namespace NUMINAMATH_GPT_mul_add_distrib_l1016_101657

theorem mul_add_distrib :
  15 * 36 + 15 * 24 = 900 := by
  sorry

end NUMINAMATH_GPT_mul_add_distrib_l1016_101657


namespace NUMINAMATH_GPT_probability_is_12_over_2907_l1016_101613

noncomputable def probability_drawing_red_red_green : ℚ :=
  (3 / 19) * (2 / 18) * (4 / 17)

theorem probability_is_12_over_2907 :
  probability_drawing_red_red_green = 12 / 2907 :=
sorry

end NUMINAMATH_GPT_probability_is_12_over_2907_l1016_101613


namespace NUMINAMATH_GPT_sqrt_polynomial_eq_l1016_101604

variable (a b c : ℝ)

def polynomial := 16 * a * c + 4 * a^2 - 12 * a * b + 9 * b^2 - 24 * b * c + 16 * c^2

theorem sqrt_polynomial_eq (a b c : ℝ) : 
  (polynomial a b c) ^ (1 / 2) = (2 * a - 3 * b + 4 * c) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_polynomial_eq_l1016_101604


namespace NUMINAMATH_GPT_bills_head_circumference_l1016_101699

/-- Jack is ordering custom baseball caps for him and his two best friends, and we need to prove the circumference of Bill's head. -/
theorem bills_head_circumference (Jack : ℝ) (Charlie : ℝ) (Bill : ℝ)
  (h1 : Jack = 12)
  (h2 : Charlie = (1 / 2) * Jack + 9)
  (h3 : Bill = (2 / 3) * Charlie) :
  Bill = 10 :=
by sorry

end NUMINAMATH_GPT_bills_head_circumference_l1016_101699


namespace NUMINAMATH_GPT_calculate_area_of_triangle_l1016_101636

theorem calculate_area_of_triangle :
  let p1 := (5, -2)
  let p2 := (5, 8)
  let p3 := (12, 8)
  let area := (1 / 2) * ((p2.2 - p1.2) * (p3.1 - p2.1))
  area = 35 := 
by
  sorry

end NUMINAMATH_GPT_calculate_area_of_triangle_l1016_101636


namespace NUMINAMATH_GPT_percent_larger_semicircles_l1016_101635

theorem percent_larger_semicircles (r1 r2 : ℝ) (d1 d2 : ℝ)
  (hr1 : r1 = d1 / 2) (hr2 : r2 = d2 / 2)
  (hd1 : d1 = 12) (hd2 : d2 = 8) : 
  (2 * (1/2) * Real.pi * r1^2) = (9/4 * (2 * (1/2) * Real.pi * r2^2)) :=
by
  sorry

end NUMINAMATH_GPT_percent_larger_semicircles_l1016_101635


namespace NUMINAMATH_GPT_tan_diff_sin_double_l1016_101678

theorem tan_diff (α : ℝ) (h : Real.tan α = 2) : 
  Real.tan (α - Real.pi / 4) = 1 / 3 := 
by 
  sorry

theorem sin_double (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_tan_diff_sin_double_l1016_101678


namespace NUMINAMATH_GPT_integer_pairs_solution_l1016_101618

theorem integer_pairs_solution (a b : ℤ) : 
  (a - b - 1 ∣ a^2 + b^2 ∧ (a^2 + b^2) * 19 = (2 * a * b - 1) * 20) ↔
  (a, b) = (22, 16) ∨ (a, b) = (-16, -22) ∨ (a, b) = (8, 6) ∨ (a, b) = (-6, -8) :=
by 
  sorry

end NUMINAMATH_GPT_integer_pairs_solution_l1016_101618


namespace NUMINAMATH_GPT_divisors_large_than_8_fact_count_l1016_101650

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem divisors_large_than_8_fact_count :
  let n := 9
  let factorial_n := factorial n
  let factorial_n_minus_1 := factorial (n - 1)
  ∃ (num_divisors : ℕ), num_divisors = 8 ∧
    (∀ d, d ∣ factorial_n → d > factorial_n_minus_1 ↔ ∃ k, k ∣ factorial_n ∧ k < 9) :=
by
  sorry

end NUMINAMATH_GPT_divisors_large_than_8_fact_count_l1016_101650


namespace NUMINAMATH_GPT_trajectory_of_moving_circle_l1016_101694

-- Definitions for the given circles C1 and C2
def Circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def Circle2 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Prove the trajectory of the center of the moving circle M
theorem trajectory_of_moving_circle (x y : ℝ) :
  ((∃ x_center y_center : ℝ, Circle1 x_center y_center ∧ Circle2 x_center y_center ∧ 
  -- Tangency conditions for Circle M
  (x - x_center)^2 + y^2 = (x_center - 2)^2 + y^2 ∧ (x - x_center)^2 + y^2 = (x_center + 2)^2 + y^2)) →
  (x = 0 ∨ x^2 - y^2 / 3 = 1) := 
sorry

end NUMINAMATH_GPT_trajectory_of_moving_circle_l1016_101694


namespace NUMINAMATH_GPT_scientific_notation_correct_l1016_101602

-- Define the given condition
def average_daily_users : ℝ := 2590000

-- The proof problem
theorem scientific_notation_correct :
  average_daily_users = 2.59 * 10^6 :=
sorry

end NUMINAMATH_GPT_scientific_notation_correct_l1016_101602


namespace NUMINAMATH_GPT_percent_increase_l1016_101612

/-- Problem statement: Given (1/2)x = 1, prove that the percentage increase from 1/2 to x is 300%. -/
theorem percent_increase (x : ℝ) (h : (1/2) * x = 1) : 
  ((x - (1/2)) / (1/2)) * 100 = 300 := 
by
  sorry

end NUMINAMATH_GPT_percent_increase_l1016_101612


namespace NUMINAMATH_GPT_teresa_ahmad_equation_l1016_101652

theorem teresa_ahmad_equation (b c : ℝ) :
  (∀ x : ℝ, |x - 4| = 3 ↔ x = 7 ∨ x = 1) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ x = 7 ∨ x = 1) →
  (b, c) = (-8, 7) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_teresa_ahmad_equation_l1016_101652


namespace NUMINAMATH_GPT_total_flowers_purchased_l1016_101675

-- Define the conditions
def sets : ℕ := 3
def pieces_per_set : ℕ := 90

-- State the proof problem
theorem total_flowers_purchased : sets * pieces_per_set = 270 :=
by
  sorry

end NUMINAMATH_GPT_total_flowers_purchased_l1016_101675


namespace NUMINAMATH_GPT_determine_other_number_l1016_101609

theorem determine_other_number (a b : ℤ) (h₁ : 3 * a + 4 * b = 161) (h₂ : a = 17 ∨ b = 17) : 
(a = 31 ∨ b = 31) :=
by
  sorry

end NUMINAMATH_GPT_determine_other_number_l1016_101609


namespace NUMINAMATH_GPT_max_area_rectangle_l1016_101668

theorem max_area_rectangle (p : ℝ) (a b : ℝ) (h : p = 2 * (a + b)) : 
  ∃ S : ℝ, S = a * b ∧ (∀ (a' b' : ℝ), p = 2 * (a' + b') → S ≥ a' * b') → a = b :=
by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_l1016_101668


namespace NUMINAMATH_GPT_number_of_sweet_potatoes_sold_to_mrs_adams_l1016_101643

def sweet_potatoes_harvested := 80
def sweet_potatoes_sold_to_mr_lenon := 15
def sweet_potatoes_unsold := 45

def sweet_potatoes_sold_to_mrs_adams :=
  sweet_potatoes_harvested - sweet_potatoes_sold_to_mr_lenon - sweet_potatoes_unsold

theorem number_of_sweet_potatoes_sold_to_mrs_adams :
  sweet_potatoes_sold_to_mrs_adams = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_sweet_potatoes_sold_to_mrs_adams_l1016_101643


namespace NUMINAMATH_GPT_find_k_value_l1016_101697

theorem find_k_value (S : ℕ → ℕ) (a : ℕ → ℕ) (k : ℤ) 
  (hS : ∀ n, S n = 5 * n^2 + k * n)
  (ha2 : a 2 = 18) :
  k = 3 := 
sorry

end NUMINAMATH_GPT_find_k_value_l1016_101697


namespace NUMINAMATH_GPT_ten_unique_positive_odd_integers_equality_l1016_101632

theorem ten_unique_positive_odd_integers_equality {x : ℕ} (h1: x = 3):
  ∃ S : Finset ℕ, S.card = 10 ∧ 
    (∀ n ∈ S, n < 100 ∧ n % 2 = 1 ∧ 
      ∃ k : ℕ, k % 2 = 1 ∧ n = k * x) :=
by
  sorry

end NUMINAMATH_GPT_ten_unique_positive_odd_integers_equality_l1016_101632


namespace NUMINAMATH_GPT_question1_solution_question2_solution_l1016_101689

noncomputable def f (x m : ℝ) : ℝ := x^2 - m * x + m - 1

theorem question1_solution (x : ℝ) :
  ∀ x, f x 3 ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2 :=
sorry

theorem question2_solution (m : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f x m ≥ -1) ↔ m ≤ 4 :=
sorry

end NUMINAMATH_GPT_question1_solution_question2_solution_l1016_101689


namespace NUMINAMATH_GPT_gcd_polynomial_example_l1016_101653

theorem gcd_polynomial_example (b : ℕ) (h : ∃ k : ℕ, b = 2 * 7784 * k) : 
  gcd (5 * b ^ 2 + 68 * b + 143) (3 * b + 14) = 25 :=
by 
  sorry

end NUMINAMATH_GPT_gcd_polynomial_example_l1016_101653


namespace NUMINAMATH_GPT_sqrt_225_eq_15_l1016_101610

theorem sqrt_225_eq_15 : Real.sqrt 225 = 15 :=
sorry

end NUMINAMATH_GPT_sqrt_225_eq_15_l1016_101610


namespace NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1016_101620

-- Problem 1
theorem simplify_expression1 (a : ℝ) : 
  (a^2)^3 + 3 * a^4 * a^2 - a^8 / a^2 = 3 * a^6 :=
by sorry

-- Problem 2
theorem simplify_expression2 (x : ℝ) : 
  (x - 3) * (x + 4) - x * (x + 3) = -2 * x - 12 :=
by sorry

end NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1016_101620


namespace NUMINAMATH_GPT_find_k_l1016_101680

theorem find_k 
  (x y: ℝ) 
  (h1: y = 5 * x + 3) 
  (h2: y = -2 * x - 25) 
  (h3: y = 3 * x + k) : 
  k = -5 :=
sorry

end NUMINAMATH_GPT_find_k_l1016_101680


namespace NUMINAMATH_GPT_number_of_kittens_l1016_101656

-- Definitions for the given conditions.
def total_animals : ℕ := 77
def hamsters : ℕ := 15
def birds : ℕ := 30

-- The proof problem statement.
theorem number_of_kittens : total_animals - hamsters - birds = 32 := by
  sorry

end NUMINAMATH_GPT_number_of_kittens_l1016_101656
