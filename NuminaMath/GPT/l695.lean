import Mathlib

namespace NUMINAMATH_GPT_elaine_earnings_increase_l695_69506

variable (E : ℝ) (P : ℝ)

theorem elaine_earnings_increase
  (h1 : E > 0) 
  (h2 : 0.30 * E * (1 + P / 100) = 1.80 * 0.20 * E) : 
  P = 20 :=
by
  sorry

end NUMINAMATH_GPT_elaine_earnings_increase_l695_69506


namespace NUMINAMATH_GPT_first_term_and_common_difference_l695_69511

theorem first_term_and_common_difference (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) :
  a 1 = 1 ∧ (a 2 - a 1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_first_term_and_common_difference_l695_69511


namespace NUMINAMATH_GPT_average_percentage_popped_average_percentage_kernels_l695_69544

theorem average_percentage_popped (
  pops1 total1 pops2 total2 pops3 total3 : ℕ
) (h1 : pops1 = 60) (h2 : total1 = 75) 
  (h3 : pops2 = 42) (h4 : total2 = 50) 
  (h5 : pops3 = 82) (h6 : total3 = 100) : 
  ((pops1 : ℝ) / total1) * 100 + ((pops2 : ℝ) / total2) * 100 + ((pops3 : ℝ) / total3) * 100 = 246 := 
by
  sorry

theorem average_percentage_kernels (pops1 total1 pops2 total2 pops3 total3 : ℕ)
  (h1 : pops1 = 60) (h2 : total1 = 75)
  (h3 : pops2 = 42) (h4 : total2 = 50)
  (h5 : pops3 = 82) (h6 : total3 = 100) :
  ((
      (((pops1 : ℝ) / total1) * 100) + 
       (((pops2 : ℝ) / total2) * 100) + 
       (((pops3 : ℝ) / total3) * 100)
    ) / 3 = 82) :=
by
  sorry

end NUMINAMATH_GPT_average_percentage_popped_average_percentage_kernels_l695_69544


namespace NUMINAMATH_GPT_sufficient_condition_for_beta_l695_69572

theorem sufficient_condition_for_beta (m : ℝ) : 
  (∀ x, (1 ≤ x ∧ x ≤ 3) → (x ≤ m)) → (3 ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_beta_l695_69572


namespace NUMINAMATH_GPT_find_total_salary_l695_69529

noncomputable def total_salary (salary_left : ℕ) : ℚ :=
  salary_left * (120 / 19)

theorem find_total_salary
  (food : ℚ) (house_rent : ℚ) (clothes : ℚ) (transport : ℚ) (remaining : ℕ) :
  food = 1 / 4 →
  house_rent = 1 / 8 →
  clothes = 3 / 10 →
  transport = 1 / 6 →
  remaining = 35000 →
  total_salary remaining = 210552.63 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_find_total_salary_l695_69529


namespace NUMINAMATH_GPT_line_through_points_on_parabola_l695_69510

theorem line_through_points_on_parabola
  (p q : ℝ)
  (hpq : p^2 - 4 * q > 0) :
  ∃ (A B : ℝ × ℝ),
    (exists (x₁ x₂ : ℝ), x₁^2 + p * x₁ + q = 0 ∧ x₂^2 + p * x₂ + q = 0 ∧
                         A = (x₁, x₁^2 / 3) ∧ B = (x₂, x₂^2 / 3) ∧
                         (∀ x y, (x, y) = A ∨ (x, y) = B → px + 3 * y + q = 0)) :=
sorry

end NUMINAMATH_GPT_line_through_points_on_parabola_l695_69510


namespace NUMINAMATH_GPT_maximum_marks_l695_69557

theorem maximum_marks (M : ℝ) (mark_obtained failed_by : ℝ) (pass_percentage : ℝ) 
  (h1 : pass_percentage = 0.6) (h2 : mark_obtained = 250) (h3 : failed_by = 50) :
  (pass_percentage * M = mark_obtained + failed_by) → M = 500 :=
by 
  sorry

end NUMINAMATH_GPT_maximum_marks_l695_69557


namespace NUMINAMATH_GPT_rosie_circles_track_24_l695_69531

-- Definition of the problem conditions
def lou_distance := 3 -- Lou's total distance in miles
def track_length := 1 / 4 -- Length of the circular track in miles
def rosie_speed_factor := 2 -- Rosie runs at twice the speed of Lou

-- Define the number of times Rosie circles the track as a result
def rosie_circles_the_track : Nat :=
  let lou_circles := lou_distance / track_length
  let rosie_distance := lou_distance * rosie_speed_factor
  let rosie_circles := rosie_distance / track_length
  rosie_circles

-- The theorem stating that Rosie circles the track 24 times
theorem rosie_circles_track_24 : rosie_circles_the_track = 24 := by
  sorry

end NUMINAMATH_GPT_rosie_circles_track_24_l695_69531


namespace NUMINAMATH_GPT_no_integer_n_for_fractions_l695_69555

theorem no_integer_n_for_fractions (n : ℤ) : ¬ (∃ n : ℤ, (n - 6) % 15 = 0 ∧ (n - 5) % 24 = 0) :=
by sorry

end NUMINAMATH_GPT_no_integer_n_for_fractions_l695_69555


namespace NUMINAMATH_GPT_cyclist_speed_ratio_is_4_l695_69512

noncomputable def ratio_of_speeds (v_a v_b v_c : ℝ) : ℝ :=
  if v_a ≤ v_b ∧ v_b ≤ v_c then v_c / v_a else 0

theorem cyclist_speed_ratio_is_4
  (v_a v_b v_c : ℝ)
  (h1 : v_a + v_b = d / 5)
  (h2 : v_b + v_c = 15)
  (h3 : 15 = (45 - d) / 3)
  (d : ℝ) : 
  ratio_of_speeds v_a v_b v_c = 4 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_speed_ratio_is_4_l695_69512


namespace NUMINAMATH_GPT_find_a_l695_69542

noncomputable def p (a : ℝ) : Prop := 3 < a ∧ a < 7/2
noncomputable def q (a : ℝ) : Prop := a > 3 ∧ a ≠ 7/2
theorem find_a (a : ℝ) (h1 : a > 3) (h2 : a ≠ 7/2) (hpq : (p a ∨ q a) ∧ ¬(p a ∧ q a)) : a > 7/2 :=
sorry

end NUMINAMATH_GPT_find_a_l695_69542


namespace NUMINAMATH_GPT_liking_songs_proof_l695_69585

def num_ways_liking_songs : ℕ :=
  let total_songs := 6
  let pair1 := 1
  let pair2 := 2
  let ways_to_choose_pair1 := Nat.choose total_songs pair1
  let remaining_songs := total_songs - pair1
  let ways_to_choose_pair2 := Nat.choose remaining_songs pair2 * Nat.choose (remaining_songs - pair2) pair2
  let final_song_choices := 4
  ways_to_choose_pair1 * ways_to_choose_pair2 * final_song_choices * 3 -- multiplied by 3 for the three possible pairs

theorem liking_songs_proof :
  num_ways_liking_songs = 2160 :=
  by sorry

end NUMINAMATH_GPT_liking_songs_proof_l695_69585


namespace NUMINAMATH_GPT_sufficient_condition_for_having_skin_l695_69562

theorem sufficient_condition_for_having_skin (H_no_skin_no_hair : ¬skin → ¬hair) :
  (hair → skin) :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_having_skin_l695_69562


namespace NUMINAMATH_GPT_common_internal_tangent_length_l695_69502

-- Definitions based on given conditions
def center_distance : ℝ := 50
def radius_small : ℝ := 7
def radius_large : ℝ := 10

-- Target theorem
theorem common_internal_tangent_length :
  let AB := center_distance
  let BE := radius_small + radius_large 
  let AE := Real.sqrt (AB^2 - BE^2)
  AE = Real.sqrt 2211 :=
by
  sorry

end NUMINAMATH_GPT_common_internal_tangent_length_l695_69502


namespace NUMINAMATH_GPT_cost_price_for_a_l695_69532

-- Definitions from the conditions
def selling_price_c : ℝ := 225
def profit_b : ℝ := 0.25
def profit_a : ℝ := 0.60

-- To prove: The cost price of the bicycle for A (cp_a) is 112.5
theorem cost_price_for_a : 
  ∃ (cp_a : ℝ), 
  (∃ (cp_b : ℝ), cp_b = (selling_price_c / (1 + profit_b)) ∧ 
   cp_a = (cp_b / (1 + profit_a))) ∧ 
   cp_a = 112.5 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_for_a_l695_69532


namespace NUMINAMATH_GPT_reflected_parabola_equation_l695_69516

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := x^2

-- Define the line of reflection
def reflection_line (x : ℝ) : ℝ := x + 2

-- The reflected equation statement to be proved
theorem reflected_parabola_equation (x y : ℝ) :
  (parabola x = y) ∧ (reflection_line x = y) →
  (∃ y' x', x = y'^2 - 4 * y' + 2 ∧ y = x' + 2 ∧ x' = y - 2) :=
sorry

end NUMINAMATH_GPT_reflected_parabola_equation_l695_69516


namespace NUMINAMATH_GPT_travel_time_equation_l695_69564

theorem travel_time_equation (x : ℝ) (h1 : ∀ d : ℝ, d > 0) :
  (x / 160) - (x / 200) = 2.5 :=
sorry

end NUMINAMATH_GPT_travel_time_equation_l695_69564


namespace NUMINAMATH_GPT_find_radius_l695_69556

theorem find_radius
  (sector_area : ℝ)
  (arc_length : ℝ)
  (sector_area_eq : sector_area = 11.25)
  (arc_length_eq : arc_length = 4.5) :
  ∃ r : ℝ, 11.25 = (1/2 : ℝ) * r * arc_length ∧ r = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_radius_l695_69556


namespace NUMINAMATH_GPT_real_coeffs_with_even_expression_are_integers_l695_69597

theorem real_coeffs_with_even_expression_are_integers
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h : ∀ x y : ℤ, (∃ k1 : ℤ, a1 * x + b1 * y + c1 = 2 * k1) ∨ (∃ k2 : ℤ, a2 * x + b2 * y + c2 = 2 * k2)) :
  (∃ (i1 j1 k1 : ℤ), a1 = i1 ∧ b1 = j1 ∧ c1 = k1) ∨
  (∃ (i2 j2 k2 : ℤ), a2 = i2 ∧ b2 = j2 ∧ c2 = k2) := by
  sorry

end NUMINAMATH_GPT_real_coeffs_with_even_expression_are_integers_l695_69597


namespace NUMINAMATH_GPT_highway_extension_completion_l695_69575

def current_length := 200
def final_length := 650
def built_first_day := 50
def built_second_day := 3 * built_first_day

theorem highway_extension_completion :
  (final_length - current_length - built_first_day - built_second_day) = 250 := by
  sorry

end NUMINAMATH_GPT_highway_extension_completion_l695_69575


namespace NUMINAMATH_GPT_domain_of_function_l695_69545

def function_domain : Set ℝ := { x : ℝ | x + 1 ≥ 0 ∧ 2 - x ≠ 0 }

theorem domain_of_function :
  function_domain = { x : ℝ | x ≥ -1 ∧ x ≠ 2 } :=
sorry

end NUMINAMATH_GPT_domain_of_function_l695_69545


namespace NUMINAMATH_GPT_find_m_l695_69554

theorem find_m
  (x y : ℝ)
  (h1 : 100 = 300 * x + 200 * y)
  (h2 : 120 = 240 * x + 300 * y)
  (h3 : ∃ m : ℝ, 50 * 3 = 150 * x + m * y):
  ∃ m : ℝ, m = 450 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l695_69554


namespace NUMINAMATH_GPT_fraction_neither_cell_phones_nor_pagers_l695_69509

theorem fraction_neither_cell_phones_nor_pagers
  (E : ℝ) -- total number of employees (E must be positive)
  (h1 : 0 < E)
  (frac_cell_phones : ℝ)
  (H1 : frac_cell_phones = (2 / 3))
  (frac_pagers : ℝ)
  (H2 : frac_pagers = (2 / 5))
  (frac_both : ℝ)
  (H3 : frac_both = 0.4) :
  (1 / 3) = (1 - frac_cell_phones - frac_pagers + frac_both) :=
by
  -- setup definitions, conditions and final proof
  sorry

end NUMINAMATH_GPT_fraction_neither_cell_phones_nor_pagers_l695_69509


namespace NUMINAMATH_GPT_find_k_l695_69599

theorem find_k (k : ℝ) :
  (∀ x, x ≠ 1 → (1 / (x^2 - x) + (k - 5) / (x^2 + x) = (k - 1) / (x^2 - 1))) →
  (1 / (1^2 - 1) + (k - 5) / (1^2 + 1) ≠ (k - 1) / (1^2 - 1)) →
  k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l695_69599


namespace NUMINAMATH_GPT_value_of_b_l695_69550

theorem value_of_b (b : ℝ) : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^3 - b*x1^2 + 1/2 = 0) ∧ (x2^3 - b*x2^2 + 1/2 = 0)) → b = 3/2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_l695_69550


namespace NUMINAMATH_GPT_larger_segment_of_triangle_l695_69537

theorem larger_segment_of_triangle (x y : ℝ) (h1 : 40^2 = x^2 + y^2) 
  (h2 : 90^2 = (100 - x)^2 + y^2) :
  100 - x = 82.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_larger_segment_of_triangle_l695_69537


namespace NUMINAMATH_GPT_solve_for_x_l695_69520

theorem solve_for_x (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) :
  x = 37 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l695_69520


namespace NUMINAMATH_GPT_evaluate_expression_l695_69596

noncomputable def repeating_to_fraction_06 : ℚ := 2 / 3
noncomputable def repeating_to_fraction_02 : ℚ := 2 / 9
noncomputable def repeating_to_fraction_04 : ℚ := 4 / 9

theorem evaluate_expression : 
  ((repeating_to_fraction_06 * repeating_to_fraction_02) - repeating_to_fraction_04) = -8 / 27 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l695_69596


namespace NUMINAMATH_GPT_characterize_solution_pairs_l695_69582

/-- Define a set S --/
def S : Set ℝ := { x : ℝ | x > 0 ∧ x ≠ 1 }

/-- log inequality --/
def log_inequality (a b : ℝ) : Prop :=
  Real.log b / Real.log a < Real.log (b + 1) / Real.log (a + 1)

/-- Define the solution sets --/
def sol1 : Set (ℝ × ℝ) := {p | p.2 = 1 ∧ p.1 > 0 ∧ p.1 ≠ 1}
def sol2 : Set (ℝ × ℝ) := {p | p.1 > p.2 ∧ p.2 > 1}
def sol3 : Set (ℝ × ℝ) := {p | p.2 > 1 ∧ p.2 > p.1}
def sol4 : Set (ℝ × ℝ) := {p | p.1 < p.2 ∧ p.2 < 1}
def sol5 : Set (ℝ × ℝ) := {p | p.2 < 1 ∧ p.2 < p.1}

/-- Prove the log inequality and characterize the solution pairs --/
theorem characterize_solution_pairs (a b : ℝ) (h1 : a ∈ S) (h2 : b > 0) :
  log_inequality a b ↔
  (a, b) ∈ sol1 ∨ (a, b) ∈ sol2 ∨ (a, b) ∈ sol3 ∨ (a, b) ∈ sol4 ∨ (a, b) ∈ sol5 :=
sorry

end NUMINAMATH_GPT_characterize_solution_pairs_l695_69582


namespace NUMINAMATH_GPT_multiplication_division_l695_69565

theorem multiplication_division:
  (213 * 16 = 3408) → (1.6 * 2.13 = 3.408) :=
by
  sorry

end NUMINAMATH_GPT_multiplication_division_l695_69565


namespace NUMINAMATH_GPT_solve_system_of_equations_l695_69553

theorem solve_system_of_equations :
  (∃ x y : ℚ, 2 * x + 4 * y = 9 ∧ 3 * x - 5 * y = 8) ↔ 
  (∃ x y : ℚ, x = 7 / 2 ∧ y = 1 / 2) := by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l695_69553


namespace NUMINAMATH_GPT_remaining_slices_correct_l695_69548

def pies : Nat := 2
def slices_per_pie : Nat := 8
def slices_total : Nat := pies * slices_per_pie
def slices_rebecca_initial : Nat := 1 * pies
def slices_remaining_after_rebecca : Nat := slices_total - slices_rebecca_initial
def slices_family_friends : Nat := 7
def slices_remaining_after_family_friends : Nat := slices_remaining_after_rebecca - slices_family_friends
def slices_rebecca_husband_last : Nat := 2
def slices_remaining : Nat := slices_remaining_after_family_friends - slices_rebecca_husband_last

theorem remaining_slices_correct : slices_remaining = 5 := 
by sorry

end NUMINAMATH_GPT_remaining_slices_correct_l695_69548


namespace NUMINAMATH_GPT_sum_of_x_values_l695_69580

noncomputable def arithmetic_angles_triangle (x : ℝ) : Prop :=
  let α := 30 * Real.pi / 180
  let β := (30 + 40) * Real.pi / 180
  let γ := (30 + 80) * Real.pi / 180
  (x = 6) ∨ (x = 8) ∨ (x = (7 + Real.sqrt 36 + Real.sqrt 83))

theorem sum_of_x_values : ∀ x : ℝ, 
  arithmetic_angles_triangle x → 
  (∃ p q r : ℝ, x = p + Real.sqrt q + Real.sqrt r ∧ p = 7 ∧ q = 36 ∧ r = 83) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_x_values_l695_69580


namespace NUMINAMATH_GPT_students_in_both_clubs_l695_69558

theorem students_in_both_clubs (total_students drama_club science_club : ℕ) 
  (students_either_or_both both_clubs : ℕ) 
  (h_total_students : total_students = 250)
  (h_drama_club : drama_club = 80)
  (h_science_club : science_club = 120)
  (h_students_either_or_both : students_either_or_both = 180)
  (h_inclusion_exclusion : students_either_or_both = drama_club + science_club - both_clubs) :
  both_clubs = 20 :=
  by sorry

end NUMINAMATH_GPT_students_in_both_clubs_l695_69558


namespace NUMINAMATH_GPT_find_m_l695_69583

theorem find_m (m : ℕ) (hm : 0 < m)
  (a : ℕ := Nat.choose (2 * m) m)
  (b : ℕ := Nat.choose (2 * m + 1) m)
  (h : 13 * a = 7 * b) : m = 6 := by
  sorry

end NUMINAMATH_GPT_find_m_l695_69583


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l695_69519

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem sufficient_but_not_necessary_condition {x y : ℝ} :
  (floor x = floor y) → (abs (x - y) < 1) ∧ (¬ (abs (x - y) < 1) → (floor x ≠ floor y)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l695_69519


namespace NUMINAMATH_GPT_expected_coins_basilio_20_l695_69518

noncomputable def binomialExpectation (n : ℕ) (p : ℚ) : ℚ :=
  n * p

noncomputable def expectedCoinsDifference : ℚ :=
  0.5

noncomputable def expectedCoinsBasilio (n : ℕ) (p : ℚ) : ℚ :=
  (binomialExpectation n p + expectedCoinsDifference) / 2

theorem expected_coins_basilio_20 :
  expectedCoinsBasilio 20 (1/2) = 5.25 :=
by
  -- here you would fill in the proof steps
  sorry

end NUMINAMATH_GPT_expected_coins_basilio_20_l695_69518


namespace NUMINAMATH_GPT_geometric_sequence_problem_l695_69517

theorem geometric_sequence_problem
  (a : ℕ → ℝ) (r : ℝ)
  (h₀ : ∀ n, a n > 0)
  (h₁ : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)
  (h₂ : ∀ n, a (n + 1) = a n * r) :
  a 3 + a 5 = 5 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l695_69517


namespace NUMINAMATH_GPT_correct_option_l695_69574

-- Definitions based on the conditions in step a
def option_a : Prop := (-3 - 1 = -2)
def option_b : Prop := (-2 * (-1 / 2) = 1)
def option_c : Prop := (16 / (-4 / 3) = 12)
def option_d : Prop := (- (3^2) / 4 = (9 / 4))

-- The proof problem statement asserting that only option B is correct.
theorem correct_option : option_b ∧ ¬ option_a ∧ ¬ option_c ∧ ¬ option_d :=
by sorry

end NUMINAMATH_GPT_correct_option_l695_69574


namespace NUMINAMATH_GPT_cross_section_area_of_truncated_pyramid_l695_69523

-- Given conditions
variables (a b : ℝ) (α : ℝ)
-- Constraints
variable (h : a > b ∧ b > 0 ∧ α > 0 ∧ α < Real.pi / 2)

-- Proposed theorem
theorem cross_section_area_of_truncated_pyramid (h : a > b ∧ b > 0 ∧ α > 0 ∧ α < Real.pi / 2) :
    ∃ area : ℝ, area = (7 * a + 3 * b) / (144 * Real.cos α) * Real.sqrt (3 * (a^2 + b^2 + 2 * a * b * Real.cos (2 * α))) :=
sorry

end NUMINAMATH_GPT_cross_section_area_of_truncated_pyramid_l695_69523


namespace NUMINAMATH_GPT_infinite_series_sum_l695_69543

theorem infinite_series_sum : 
  (∑' n : ℕ, (4 * n + 1 : ℝ) / ((4 * n - 1)^3 * (4 * n + 3)^3)) = 1 / 972 := 
by 
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l695_69543


namespace NUMINAMATH_GPT_nth_monomial_l695_69566

variable (a : ℝ)

def monomial_seq (n : ℕ) : ℝ :=
  (n + 1) * a ^ n

theorem nth_monomial (n : ℕ) : monomial_seq a n = (n + 1) * a ^ n :=
by
  sorry

end NUMINAMATH_GPT_nth_monomial_l695_69566


namespace NUMINAMATH_GPT_librarians_all_work_together_l695_69500

/-- Peter works every 5 days -/
def Peter_days := 5

/-- Quinn works every 8 days -/
def Quinn_days := 8

/-- Rachel works every 10 days -/
def Rachel_days := 10

/-- Sam works every 14 days -/
def Sam_days := 14

/-- Least common multiple of the intervals at which Peter, Quinn, Rachel, and Sam work -/
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem librarians_all_work_together : LCM (LCM (LCM Peter_days Quinn_days) Rachel_days) Sam_days = 280 :=
  by
  sorry

end NUMINAMATH_GPT_librarians_all_work_together_l695_69500


namespace NUMINAMATH_GPT_probability_passing_exam_l695_69503

-- Define probabilities for sets A, B, and C, and passing conditions
def P_A := 0.3
def P_B := 0.3
def P_C := 1 - P_A - P_B
def P_D_given_A := 0.8
def P_D_given_B := 0.6
def P_D_given_C := 0.8

-- Total probability of passing
def P_D := P_A * P_D_given_A + P_B * P_D_given_B + P_C * P_D_given_C

-- Proof that the total probability of passing is 0.74
theorem probability_passing_exam : P_D = 0.74 :=
by
  -- (skip the proof steps)
  sorry

end NUMINAMATH_GPT_probability_passing_exam_l695_69503


namespace NUMINAMATH_GPT_train_speed_l695_69594

noncomputable def train_length : ℝ := 120
noncomputable def crossing_time : ℝ := 2.699784017278618

theorem train_speed : (train_length / crossing_time) = 44.448 := by
  sorry

end NUMINAMATH_GPT_train_speed_l695_69594


namespace NUMINAMATH_GPT_positive_difference_of_squares_l695_69563

theorem positive_difference_of_squares (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 12) : x^2 - y^2 = 600 := by
  sorry

end NUMINAMATH_GPT_positive_difference_of_squares_l695_69563


namespace NUMINAMATH_GPT_more_seventh_graders_than_sixth_graders_l695_69577

theorem more_seventh_graders_than_sixth_graders 
  (n m : ℕ)
  (H1 : ∀ x : ℕ, x = n → 7 * n = 6 * m) : 
  m > n := 
by
  -- Proof is not required and will be skipped with sorry.
  sorry

end NUMINAMATH_GPT_more_seventh_graders_than_sixth_graders_l695_69577


namespace NUMINAMATH_GPT_tshirt_costs_more_than_jersey_l695_69592

-- Definitions based on the conditions
def cost_of_tshirt : ℕ := 192
def cost_of_jersey : ℕ := 34

-- Theorem statement
theorem tshirt_costs_more_than_jersey : cost_of_tshirt - cost_of_jersey = 158 := by
  sorry

end NUMINAMATH_GPT_tshirt_costs_more_than_jersey_l695_69592


namespace NUMINAMATH_GPT_number_of_arrangements_word_l695_69538

noncomputable def factorial (n : Nat) : Nat := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem number_of_arrangements_word (letters : List Char) (n : Nat) (r1 r2 r3 : Nat) 
  (h1 : letters = ['M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'S'])
  (h2 : 2 = r1) (h3 : 2 = r2) (h4 : 2 = r3) :
  n = 11 → 
  factorial n / (factorial r1 * factorial r2 * factorial r3) = 4989600 := 
by
  sorry

end NUMINAMATH_GPT_number_of_arrangements_word_l695_69538


namespace NUMINAMATH_GPT_number_of_terminating_decimals_l695_69552

theorem number_of_terminating_decimals :
  ∃ (count : ℕ), count = 64 ∧ ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 449 → (∃ k : ℕ, n = 7 * k) → (∃ k : ℕ, (∃ m : ℕ, 560 = 2^m * 5^k * n)) :=
sorry

end NUMINAMATH_GPT_number_of_terminating_decimals_l695_69552


namespace NUMINAMATH_GPT_lines_perpendicular_to_same_line_l695_69527

-- Definitions for lines and relationship types
structure Line := (name : String)
inductive RelType
| parallel 
| intersect
| skew

-- Definition stating two lines are perpendicular to the same line
def perpendicular_to_same_line (l1 l2 l3 : Line) : Prop :=
  -- (dot product or a similar condition could be specified, leaving abstract here)
  sorry

-- Theorem statement
theorem lines_perpendicular_to_same_line (l1 l2 l3 : Line) (h1 : perpendicular_to_same_line l1 l2 l3) : 
  RelType :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_lines_perpendicular_to_same_line_l695_69527


namespace NUMINAMATH_GPT_triangle_is_isosceles_l695_69590

theorem triangle_is_isosceles (a b c : ℝ) (h : 3 * a^3 + 6 * a^2 * b - 3 * a^2 * c - 6 * a * b * c = 0) 
  (habc : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) : 
  (a = c) := 
by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l695_69590


namespace NUMINAMATH_GPT_Jenny_has_6_cards_l695_69526

variable (J : ℕ)

noncomputable def Jenny_number := J
noncomputable def Orlando_number := J + 2
noncomputable def Richard_number := 3 * (J + 2)
noncomputable def Total_number := J + (J + 2) + 3 * (J + 2)

theorem Jenny_has_6_cards
  (h1 : Orlando_number J = J + 2)
  (h2 : Richard_number J = 3 * (J + 2))
  (h3 : Total_number J = 38) : J = 6 :=
by
  sorry

end NUMINAMATH_GPT_Jenny_has_6_cards_l695_69526


namespace NUMINAMATH_GPT_water_added_l695_69571

def container_capacity : ℕ := 80
def initial_fill_percentage : ℝ := 0.5
def final_fill_percentage : ℝ := 0.75
def initial_fill_amount (capacity : ℕ) (percentage : ℝ) : ℝ := percentage * capacity
def final_fill_amount (capacity : ℕ) (percentage : ℝ) : ℝ := percentage * capacity

theorem water_added (capacity : ℕ) (initial_percentage final_percentage : ℝ) :
  final_fill_amount capacity final_percentage - initial_fill_amount capacity initial_percentage = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_water_added_l695_69571


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l695_69578

open Real

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 1) → (a + b > 2 ∧ a * b > 1) ∧ ¬((a + b > 2 ∧ a * b > 1) → (a > 1 ∧ b > 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l695_69578


namespace NUMINAMATH_GPT_range_of_a_l695_69595

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + a * x + a = 0}

theorem range_of_a (a : ℝ) (h : set_A ∪ set_B a = set_A) : 0 ≤ a ∧ a < 4 := 
sorry

end NUMINAMATH_GPT_range_of_a_l695_69595


namespace NUMINAMATH_GPT_find_d_l695_69573

theorem find_d (a b c d : ℝ) (h : a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c - d + 3)) : 
  d = 13 / 4 :=
sorry

end NUMINAMATH_GPT_find_d_l695_69573


namespace NUMINAMATH_GPT_find_multiplier_l695_69508

theorem find_multiplier 
  (x : ℝ)
  (number : ℝ)
  (condition1 : 4 * number + x * number = 55)
  (condition2 : number = 5.0) :
  x = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_multiplier_l695_69508


namespace NUMINAMATH_GPT_difference_of_sums_l695_69513

noncomputable def sum_of_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

noncomputable def sum_of_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem difference_of_sums : 
  sum_of_first_n_even 2004 - sum_of_first_n_odd 2003 = 6017 := 
by sorry

end NUMINAMATH_GPT_difference_of_sums_l695_69513


namespace NUMINAMATH_GPT_reduced_price_per_dozen_l695_69568

theorem reduced_price_per_dozen 
  (P : ℝ) -- original price per apple
  (R : ℝ) -- reduced price per apple
  (A : ℝ) -- number of apples originally bought for Rs. 30
  (H1 : R = 0.7 * P) 
  (H2 : A * P = (A + 54) * R) :
  30 / (A + 54) * 12 = 2 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_per_dozen_l695_69568


namespace NUMINAMATH_GPT_tree_planting_l695_69507

/-- The city plans to plant 500 thousand trees. The original plan 
was to plant x thousand trees per day. Due to volunteers, the actual number 
of trees planted per day exceeds the original plan by 30%. As a result, 
the task is completed 2 days ahead of schedule. Prove the equation. -/
theorem tree_planting
    (x : ℝ) 
    (hx : x > 0) : 
    (500 / x) - (500 / ((1 + 0.3) * x)) = 2 :=
sorry

end NUMINAMATH_GPT_tree_planting_l695_69507


namespace NUMINAMATH_GPT_number_of_cows_l695_69588

theorem number_of_cows (C H : ℕ) (L : ℕ) (h1 : L = 4 * C + 2 * H) (h2 : L = 2 * (C + H) + 20) : C = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cows_l695_69588


namespace NUMINAMATH_GPT_min_value_x_squared_y_cubed_z_l695_69584

theorem min_value_x_squared_y_cubed_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
(h : 1 / x + 1 / y + 1 / z = 9) : x^2 * y^3 * z ≥ 729 / 6912 :=
sorry

end NUMINAMATH_GPT_min_value_x_squared_y_cubed_z_l695_69584


namespace NUMINAMATH_GPT_frog_jump_probability_is_one_fifth_l695_69581

noncomputable def frog_jump_probability : ℝ := sorry

theorem frog_jump_probability_is_one_fifth : frog_jump_probability = 1 / 5 := sorry

end NUMINAMATH_GPT_frog_jump_probability_is_one_fifth_l695_69581


namespace NUMINAMATH_GPT_largest_common_number_in_arithmetic_sequences_l695_69515

theorem largest_common_number_in_arithmetic_sequences (x : ℕ)
  (h1 : x ≡ 2 [MOD 8])
  (h2 : x ≡ 5 [MOD 9])
  (h3 : x < 200) : x = 194 :=
by sorry

end NUMINAMATH_GPT_largest_common_number_in_arithmetic_sequences_l695_69515


namespace NUMINAMATH_GPT_right_triangle_unique_perimeter_18_l695_69541

theorem right_triangle_unique_perimeter_18 :
  ∃! (a b c : ℤ), a^2 + b^2 = c^2 ∧ a + b + c = 18 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end NUMINAMATH_GPT_right_triangle_unique_perimeter_18_l695_69541


namespace NUMINAMATH_GPT_decagon_ratio_bisect_l695_69561

theorem decagon_ratio_bisect (area_decagon unit_square area_trapezoid : ℕ) 
  (h_area_decagon : area_decagon = 12) 
  (h_bisect : ∃ RS : ℕ, ∃ XR : ℕ, RS * 2 = area_decagon) 
  (below_RS : ∃ base1 base2 height : ℕ, base1 = 3 ∧ base2 = 3 ∧ base1 * height + 1 = 6) 
  : ∃ XR RS : ℕ, RS ≠ 0 ∧ XR / RS = 1 := 
sorry

end NUMINAMATH_GPT_decagon_ratio_bisect_l695_69561


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_eq_6_or_neg6_l695_69528

theorem perfect_square_trinomial_m_eq_6_or_neg6
  (m : ℤ) :
  (∃ a : ℤ, x * x + m * x + 9 = (x + a) * (x + a)) → (m = 6 ∨ m = -6) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_eq_6_or_neg6_l695_69528


namespace NUMINAMATH_GPT_Beth_and_Jan_total_money_l695_69536

theorem Beth_and_Jan_total_money (B J : ℝ) 
  (h1 : B + 35 = 105)
  (h2 : J - 10 = B) : 
  B + J = 150 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_Beth_and_Jan_total_money_l695_69536


namespace NUMINAMATH_GPT_faster_train_speed_is_45_l695_69593

noncomputable def speedOfFasterTrain (V_s : ℝ) (length_train : ℝ) (time : ℝ) : ℝ :=
  let V_r : ℝ := (length_train * 2) / (time / 3600)
  V_r - V_s

theorem faster_train_speed_is_45 
  (length_train : ℝ := 0.5)
  (V_s : ℝ := 30)
  (time : ℝ := 47.99616030717543) :
  speedOfFasterTrain V_s length_train time = 45 :=
sorry

end NUMINAMATH_GPT_faster_train_speed_is_45_l695_69593


namespace NUMINAMATH_GPT_no_sport_members_count_l695_69591

theorem no_sport_members_count (n B T B_and_T : ℕ) (h1 : n = 27) (h2 : B = 17) (h3 : T = 19) (h4 : B_and_T = 11) : 
  n - (B + T - B_and_T) = 2 :=
by
  sorry

end NUMINAMATH_GPT_no_sport_members_count_l695_69591


namespace NUMINAMATH_GPT_find_y_l695_69514

theorem find_y (x y : ℤ) 
  (h1 : x^2 + 4 = y - 2) 
  (h2 : x = 6) : 
  y = 42 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_l695_69514


namespace NUMINAMATH_GPT_find_BA_prime_l695_69533

theorem find_BA_prime (BA BC A_prime C_1 : ℝ) 
  (h1 : BA = 3)
  (h2 : BC = 2)
  (h3 : A_prime < BA)
  (h4 : A_prime * C_1 = 3) : A_prime = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_BA_prime_l695_69533


namespace NUMINAMATH_GPT_problem_l695_69589

variables {a b : ℝ}

theorem problem (h₁ : -1 < a) (h₂ : a < b) (h₃ : b < 0) : 
  (1/a > 1/b) ∧ (a^2 + b^2 > 2 * a * b) ∧ (a + (1/a) > b + (1/b)) :=
by
  sorry

end NUMINAMATH_GPT_problem_l695_69589


namespace NUMINAMATH_GPT_d_share_l695_69530

theorem d_share (x : ℝ) (d c : ℝ)
  (h1 : c = 3 * x + 500)
  (h2 : d = 3 * x)
  (h3 : c = 4 * x) :
  d = 1500 := 
by 
  sorry

end NUMINAMATH_GPT_d_share_l695_69530


namespace NUMINAMATH_GPT_pyramid_height_correct_l695_69525

noncomputable def pyramid_height (a α : ℝ) : ℝ :=
  a / (Real.sqrt (2 * (Real.tan (α / 2))^2 - 2))

theorem pyramid_height_correct (a α : ℝ) (hα : α ≠ 0 ∧ α ≠ π) :
  ∃ m : ℝ, m = pyramid_height a α := 
by
  use a / (Real.sqrt (2 * (Real.tan (α / 2))^2 - 2))
  sorry

end NUMINAMATH_GPT_pyramid_height_correct_l695_69525


namespace NUMINAMATH_GPT_letters_into_mailboxes_l695_69559

theorem letters_into_mailboxes (letters : ℕ) (mailboxes : ℕ) (h_letters: letters = 3) (h_mailboxes: mailboxes = 4) :
  (mailboxes ^ letters) = 64 := by
  sorry

end NUMINAMATH_GPT_letters_into_mailboxes_l695_69559


namespace NUMINAMATH_GPT_opposite_of_neg3_l695_69505

def opposite (a : Int) : Int := -a

theorem opposite_of_neg3 : opposite (-3) = 3 := by
  unfold opposite
  show (-(-3)) = 3
  sorry

end NUMINAMATH_GPT_opposite_of_neg3_l695_69505


namespace NUMINAMATH_GPT_school_year_length_l695_69521

theorem school_year_length
  (children : ℕ)
  (juice_boxes_per_child_per_day : ℕ)
  (days_per_week : ℕ)
  (total_juice_boxes : ℕ)
  (w : ℕ)
  (h1 : children = 3)
  (h2 : juice_boxes_per_child_per_day = 1)
  (h3 : days_per_week = 5)
  (h4 : total_juice_boxes = 375)
  (h5 : total_juice_boxes = children * juice_boxes_per_child_per_day * days_per_week * w)
  : w = 25 :=
by
  sorry

end NUMINAMATH_GPT_school_year_length_l695_69521


namespace NUMINAMATH_GPT_quadratic_rewrite_de_value_l695_69549

theorem quadratic_rewrite_de_value : 
  ∃ (d e f : ℤ), (d^2 * x^2 + 2 * d * e * x + e^2 + f = 4 * x^2 - 16 * x + 2) → (d * e = -8) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_rewrite_de_value_l695_69549


namespace NUMINAMATH_GPT_log_sum_equals_18084_l695_69560

theorem log_sum_equals_18084 : 
  (Finset.sum (Finset.range 2013) (λ x => (Int.floor (Real.log x / Real.log 2)))) = 18084 :=
by
  sorry

end NUMINAMATH_GPT_log_sum_equals_18084_l695_69560


namespace NUMINAMATH_GPT_range_of_a_l695_69576

theorem range_of_a (f : ℝ → ℝ) (h_increasing : ∀ x y, x < y → f x < f y) (a : ℝ) :
  f (a^2 - a) > f (2 * a^2 - 4 * a) → 0 < a ∧ a < 3 :=
by
  -- We translate the condition f(a^2 - a) > f(2a^2 - 4a) to the inequality
  intro h
  -- Apply the fact that f is increasing to deduce the inequality on a
  sorry

end NUMINAMATH_GPT_range_of_a_l695_69576


namespace NUMINAMATH_GPT_divisible_by_17_l695_69504

theorem divisible_by_17 (a b c d : ℕ) (h1 : a + b + c + d = 2023)
    (h2 : 2023 ∣ (a * b - c * d))
    (h3 : 2023 ∣ (a^2 + b^2 + c^2 + d^2))
    (h4 : ∀ x, x = a ∨ x = b ∨ x = c ∨ x = d → 7 ∣ x) :
    (∀ x, x = a ∨ x = b ∨ x = c ∨ x = d → 17 ∣ x) := 
sorry

end NUMINAMATH_GPT_divisible_by_17_l695_69504


namespace NUMINAMATH_GPT_marcus_scored_50_percent_l695_69522

variable (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total_points : ℕ)

def marcus_percentage_points (three_point_goals two_point_goals team_total_points : ℕ) : ℚ :=
  let marcus_points := three_point_goals * 3 + two_point_goals * 2
  (marcus_points : ℚ) / team_total_points * 100

theorem marcus_scored_50_percent (h1 : three_point_goals = 5) (h2 : two_point_goals = 10) (h3 : team_total_points = 70) :
  marcus_percentage_points three_point_goals two_point_goals team_total_points = 50 :=
by
  sorry

end NUMINAMATH_GPT_marcus_scored_50_percent_l695_69522


namespace NUMINAMATH_GPT_arabella_first_step_time_l695_69539

def time_first_step (x : ℝ) : Prop :=
  let time_second_step := x / 2
  let time_third_step := x + x / 2
  (x + time_second_step + time_third_step = 90)

theorem arabella_first_step_time (x : ℝ) (h : time_first_step x) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_arabella_first_step_time_l695_69539


namespace NUMINAMATH_GPT_case_b_conditions_l695_69546

-- Definition of the polynomial
def polynomial (p q x : ℝ) : ℝ := x^2 + p * x + q

-- Main theorem
theorem case_b_conditions (p q: ℝ) (x1 x2: ℝ) (hx1: x1 ≤ 0) (hx2: x2 ≥ 2) :
    q ≤ 0 ∧ 2 * p + q + 4 ≤ 0 :=
sorry

end NUMINAMATH_GPT_case_b_conditions_l695_69546


namespace NUMINAMATH_GPT_asian_games_volunteer_selection_l695_69570

-- Define the conditions.

def total_volunteers : ℕ := 5
def volunteer_A_cannot_serve_language_services : Prop := true

-- Define the main problem.
-- We are supposed to find the number of ways to assign three roles given the conditions.
def num_ways_to_assign_roles : ℕ :=
  let num_ways_language_services := 4 -- A cannot serve this role, so 4 choices
  let num_ways_other_roles := 4 * 3 -- We need to choose and arrange 2 volunteers out of remaining
  num_ways_language_services * num_ways_other_roles

-- The target theorem.
theorem asian_games_volunteer_selection : num_ways_to_assign_roles = 48 :=
by
  sorry

end NUMINAMATH_GPT_asian_games_volunteer_selection_l695_69570


namespace NUMINAMATH_GPT_dice_product_probability_l695_69586

def is_valid_die_value (n : ℕ) : Prop := n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)

theorem dice_product_probability :
  ∃ (a b c : ℕ), is_valid_die_value a ∧ is_valid_die_value b ∧ is_valid_die_value c ∧ 
  a * b * c = 8 ∧ 
  (1 / 6 : ℝ) * (1 / 6) * (1 / 6) * (6 + 1) = (7 / 216 : ℝ) :=
sorry

end NUMINAMATH_GPT_dice_product_probability_l695_69586


namespace NUMINAMATH_GPT_total_cans_collected_l695_69587

-- Definitions based on conditions
def bags_on_saturday : ℕ := 6
def bags_on_sunday : ℕ := 3
def cans_per_bag : ℕ := 8

-- The theorem statement
theorem total_cans_collected : bags_on_saturday + bags_on_sunday * cans_per_bag = 72 :=
by
  sorry

end NUMINAMATH_GPT_total_cans_collected_l695_69587


namespace NUMINAMATH_GPT_max_books_single_student_l695_69540

theorem max_books_single_student (total_students : ℕ) (students_0_books : ℕ) (students_1_book : ℕ) (students_2_books : ℕ) (avg_books_per_student : ℕ) :
  total_students = 20 →
  students_0_books = 3 →
  students_1_book = 9 →
  students_2_books = 4 →
  avg_books_per_student = 2 →
  ∃ max_books : ℕ, max_books = 14 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_max_books_single_student_l695_69540


namespace NUMINAMATH_GPT_trajectory_eq_of_midpoint_l695_69501

theorem trajectory_eq_of_midpoint (x y m n : ℝ) (hM_on_circle : m^2 + n^2 = 1)
  (hP_midpoint : (2*x = 3 + m) ∧ (2*y = n)) :
  (2*x - 3)^2 + 4*y^2 = 1 := 
sorry

end NUMINAMATH_GPT_trajectory_eq_of_midpoint_l695_69501


namespace NUMINAMATH_GPT_average_30_matches_is_25_l695_69524

noncomputable def average_runs_in_30_matches (average_20_matches average_10_matches : ℝ) (total_matches_20 total_matches_10 : ℕ) : ℝ :=
  let total_runs_20 := total_matches_20 * average_20_matches
  let total_runs_10 := total_matches_10 * average_10_matches
  (total_runs_20 + total_runs_10) / (total_matches_20 + total_matches_10)

theorem average_30_matches_is_25 (h1 : average_runs_in_30_matches 30 15 20 10 = 25) : 
  average_runs_in_30_matches 30 15 20 10 = 25 := 
  by
    exact h1

end NUMINAMATH_GPT_average_30_matches_is_25_l695_69524


namespace NUMINAMATH_GPT_train_speed_in_kph_l695_69569

-- Define the given conditions
def length_of_train : ℝ := 200 -- meters
def time_crossing_pole : ℝ := 16 -- seconds

-- Define conversion factor
def mps_to_kph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

-- Statement of the theorem
theorem train_speed_in_kph : mps_to_kph (length_of_train / time_crossing_pole) = 45 := 
sorry

end NUMINAMATH_GPT_train_speed_in_kph_l695_69569


namespace NUMINAMATH_GPT_geometric_sequence_problem_l695_69535

variable {a : ℕ → ℝ} -- Considering the sequence is a real number sequence
variable {q : ℝ} -- Common ratio

-- Conditions
axiom a2a6_eq_16 : a 2 * a 6 = 16
axiom a4_plus_a8_eq_8 : a 4 + a 8 = 8

-- Geometric sequence definition
axiom geometric_sequence : ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_problem : a 20 / a 10 = 1 :=
  by
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l695_69535


namespace NUMINAMATH_GPT_symmetric_points_sum_l695_69567

theorem symmetric_points_sum (a b : ℝ) (hA1 : A = (a, 1)) (hB1 : B = (5, b))
    (h_symmetric : (a, 1) = -(5, b)) : a + b = -6 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_points_sum_l695_69567


namespace NUMINAMATH_GPT_avg_books_per_student_l695_69551

theorem avg_books_per_student 
  (total_students : ℕ)
  (students_zero_books : ℕ)
  (students_one_book : ℕ)
  (students_two_books : ℕ)
  (max_books_per_student : ℕ) 
  (remaining_students_min_books : ℕ)
  (total_books : ℕ)
  (avg_books : ℚ)
  (h1 : total_students = 32)
  (h2 : students_zero_books = 2)
  (h3 : students_one_book = 12)
  (h4 : students_two_books = 10)
  (h5 : max_books_per_student = 11)
  (h6 : remaining_students_min_books = 8)
  (h7 : total_books = 0 * students_zero_books + 1 * students_one_book + 2 * students_two_books + 3 * remaining_students_min_books)
  (h8 : avg_books = total_books / total_students) :
  avg_books = 1.75 :=
by {
  -- Additional constraints and intermediate steps can be added here if necessary
  sorry
}

end NUMINAMATH_GPT_avg_books_per_student_l695_69551


namespace NUMINAMATH_GPT_even_function_periodic_odd_function_period_generalized_period_l695_69534

-- Problem 1
theorem even_function_periodic (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (-x) = f x) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 2 * a) = f x :=
by sorry

-- Problem 2
theorem odd_function_period (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 4 * a) = f x :=
by sorry

-- Problem 3
theorem generalized_period (f : ℝ → ℝ) (a m n : ℝ) (h₁ : ∀ x : ℝ, 2 * n - f x = f (2 * m - x)) (h₂ : ∀ x : ℝ, f (2 * a - x) = f x) :
  ∀ x : ℝ, f (x + 4 * (m - a)) = f x :=
by sorry

end NUMINAMATH_GPT_even_function_periodic_odd_function_period_generalized_period_l695_69534


namespace NUMINAMATH_GPT_smallest_value_wawbwcwd_l695_69598

noncomputable def g (x : ℝ) : ℝ := x^4 + 10 * x^3 + 35 * x^2 + 50 * x + 24

theorem smallest_value_wawbwcwd (w1 w2 w3 w4 : ℝ) : 
  (∀ x : ℝ, g x = 0 ↔ x = w1 ∨ x = w2 ∨ x = w3 ∨ x = w4) →
  |w1 * w2 + w3 * w4| = 12 ∨ |w1 * w3 + w2 * w4| = 12 ∨ |w1 * w4 + w2 * w3| = 12 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_value_wawbwcwd_l695_69598


namespace NUMINAMATH_GPT_matrix_det_zero_l695_69547

variables {α β γ : ℝ}

theorem matrix_det_zero (h : α + β + γ = π) :
  Matrix.det ![
    ![Real.cos β, Real.cos α, -1],
    ![Real.cos γ, -1, Real.cos α],
    ![-1, Real.cos γ, Real.cos β]
  ] = 0 :=
sorry

end NUMINAMATH_GPT_matrix_det_zero_l695_69547


namespace NUMINAMATH_GPT_simplify_and_evaluate_l695_69579

theorem simplify_and_evaluate (a b : ℝ) (h : a - 2 * b = -1) :
  -3 * a * (a - 2 * b)^5 + 6 * b * (a - 2 * b)^5 - 5 * (-a + 2 * b)^3 = -8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l695_69579
