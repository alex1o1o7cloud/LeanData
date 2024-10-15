import Mathlib

namespace NUMINAMATH_GPT_solve_for_x_l535_53566

theorem solve_for_x : ∀ x : ℝ, 3^(2 * x) = Real.sqrt 27 → x = 3 / 4 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_solve_for_x_l535_53566


namespace NUMINAMATH_GPT_at_least_one_corner_square_selected_l535_53517

theorem at_least_one_corner_square_selected :
  let total_squares := 16
  let total_corners := 4
  let total_non_corners := 12
  let ways_to_select_3_from_total := Nat.choose total_squares 3
  let ways_to_select_3_from_non_corners := Nat.choose total_non_corners 3
  let probability_no_corners := (ways_to_select_3_from_non_corners : ℚ) / ways_to_select_3_from_total
  let probability_at_least_one_corner := 1 - probability_no_corners
  probability_at_least_one_corner = (17 / 28 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_corner_square_selected_l535_53517


namespace NUMINAMATH_GPT_derivative_at_zero_l535_53545

def f (x : ℝ) : ℝ := (x + 1)^4

theorem derivative_at_zero : deriv f 0 = 4 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_zero_l535_53545


namespace NUMINAMATH_GPT_probability_at_least_one_hit_l535_53592

theorem probability_at_least_one_hit (pA pB pC : ℝ) (hA : pA = 0.7) (hB : pB = 0.5) (hC : pC = 0.4) : 
  (1 - ((1 - pA) * (1 - pB) * (1 - pC))) = 0.91 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_hit_l535_53592


namespace NUMINAMATH_GPT_fraction_of_female_participants_is_correct_l535_53540

-- defining conditions
def last_year_males : ℕ := 30
def male_increase_rate : ℚ := 1.1
def female_increase_rate : ℚ := 1.25
def overall_increase_rate : ℚ := 1.2

-- the statement to prove
theorem fraction_of_female_participants_is_correct :
  ∀ (y : ℕ), 
  let males_this_year := last_year_males * male_increase_rate
  let females_this_year := y * female_increase_rate
  let total_last_year := last_year_males + y
  let total_this_year := total_last_year * overall_increase_rate
  total_this_year = males_this_year + females_this_year →
  (females_this_year / total_this_year) = (25 / 36) :=
by
  intros y
  let males_this_year := last_year_males * male_increase_rate
  let females_this_year := y * female_increase_rate
  let total_last_year := last_year_males + y
  let total_this_year := total_last_year * overall_increase_rate
  intro h
  sorry

end NUMINAMATH_GPT_fraction_of_female_participants_is_correct_l535_53540


namespace NUMINAMATH_GPT_bees_process_2_77_kg_nectar_l535_53500

noncomputable def nectar_to_honey : ℝ :=
  let percent_other_in_nectar : ℝ := 0.30
  let other_mass_in_honey : ℝ := 0.83
  other_mass_in_honey / percent_other_in_nectar

theorem bees_process_2_77_kg_nectar :
  nectar_to_honey = 2.77 :=
by
  sorry

end NUMINAMATH_GPT_bees_process_2_77_kg_nectar_l535_53500


namespace NUMINAMATH_GPT_units_digit_uniform_l535_53535

-- Definitions
def domain : Finset ℕ := Finset.range 15

def pick : Type := { n // n ∈ domain }

def uniform_pick : pick := sorry

-- Statement of the theorem
theorem units_digit_uniform :
  ∀ (J1 J2 K : pick), 
  ∃ d : ℕ, d < 10 ∧ (J1.val + J2.val + K.val) % 10 = d
:= sorry

end NUMINAMATH_GPT_units_digit_uniform_l535_53535


namespace NUMINAMATH_GPT_max_distance_circle_ellipse_l535_53577

theorem max_distance_circle_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 2}
  let ellipse := {p : ℝ × ℝ | p.1^2 / 10 + p.2^2 = 1}
  ∀ (P Q : ℝ × ℝ), P ∈ circle → Q ∈ ellipse → 
  dist P Q ≤ 6 * Real.sqrt 2 :=
by
  intro circle ellipse P Q hP hQ
  sorry

end NUMINAMATH_GPT_max_distance_circle_ellipse_l535_53577


namespace NUMINAMATH_GPT_area_of_similar_rectangle_l535_53570

theorem area_of_similar_rectangle:
  ∀ (R1 : ℝ → ℝ → Prop) (R2 : ℝ → ℝ → Prop),
  (∀ a b, R1 a b → a = 3 ∧ a * b = 18) →
  (∀ a b c d, R1 a b → R2 c d → c / d = a / b) →
  (∀ a b, R2 a b → a^2 + b^2 = 400) →
  ∃ areaR2, (∀ a b, R2 a b → a * b = areaR2) ∧ areaR2 = 160 :=
by
  intros R1 R2 hR1 h_similar h_diagonal
  use 160
  sorry

end NUMINAMATH_GPT_area_of_similar_rectangle_l535_53570


namespace NUMINAMATH_GPT_bob_speed_before_construction_l535_53515

theorem bob_speed_before_construction:
  ∀ (v : ℝ),
    (1.5 * v + 2 * 45 = 180) →
    v = 60 :=
by
  intros v h
  sorry

end NUMINAMATH_GPT_bob_speed_before_construction_l535_53515


namespace NUMINAMATH_GPT_solution_set_of_inequality_l535_53567

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 - x + 2 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l535_53567


namespace NUMINAMATH_GPT_total_spent_l535_53581

theorem total_spent (jayda_spent : ℝ) (haitana_spent : ℝ) (jayda_spent_eq : jayda_spent = 400) (aitana_more_than_jayda : haitana_spent = jayda_spent + (2/5) * jayda_spent) :
  jayda_spent + haitana_spent = 960 :=
by
  rw [jayda_spent_eq, aitana_more_than_jayda]
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_total_spent_l535_53581


namespace NUMINAMATH_GPT_union_of_A_and_B_complement_of_A_intersect_B_intersection_of_A_and_C_l535_53505

open Set

def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | x^2 - 12*x + 20 < 0 }
def C (a : ℝ) : Set ℝ := { x | x < a }

theorem union_of_A_and_B :
  A ∪ B = { x : ℝ | 2 < x ∧ x < 10 } :=
sorry

theorem complement_of_A_intersect_B :
  ((univ \ A) ∩ B) = { x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } :=
sorry

theorem intersection_of_A_and_C (a : ℝ) (h : (A ∩ C a).Nonempty) :
  a > 3 :=
sorry

end NUMINAMATH_GPT_union_of_A_and_B_complement_of_A_intersect_B_intersection_of_A_and_C_l535_53505


namespace NUMINAMATH_GPT_ellipse_equation_l535_53549

theorem ellipse_equation (a b : ℝ) (x y : ℝ) (M : ℝ × ℝ)
  (h1 : 2 * a = 4)
  (h2 : 2 * b = 2 * a / 2)
  (h3 : M = (2, 1))
  (line_eq : ∀ k : ℝ, (y = 1 + k * (x - 2))) :
  (a = 2) ∧ (b = 1) ∧ (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → (x^2 / 4 + y^2 = 1)) ∧
  (∃ k : ℝ, (k = -1/2) ∧ (∀ x y : ℝ, (y - 1 = k * (x - 2)) → (x + 2*y - 4 = 0))) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l535_53549


namespace NUMINAMATH_GPT_max_A_l535_53501

theorem max_A (A : ℝ) : (∀ (x y : ℕ), 0 < x → 0 < y → 3 * x^2 + y^2 + 1 ≥ A * (x^2 + x * y + x)) ↔ A ≤ 5 / 3 := by
  sorry

end NUMINAMATH_GPT_max_A_l535_53501


namespace NUMINAMATH_GPT_find_n_l535_53511

theorem find_n (n : ℕ) (h : n > 2016) (h_not_divisible : ¬ (1^n + 2^n + 3^n + 4^n) % 10 = 0) : n = 2020 :=
sorry

end NUMINAMATH_GPT_find_n_l535_53511


namespace NUMINAMATH_GPT_geometric_sequence_a2_a6_l535_53527

theorem geometric_sequence_a2_a6 (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a (n + 1) = r * a n) (h₄ : a 4 = 4) :
  a 2 * a 6 = 16 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a2_a6_l535_53527


namespace NUMINAMATH_GPT_number_of_white_cats_l535_53561

theorem number_of_white_cats (total_cats : ℕ) (percent_black : ℤ) (grey_cats : ℕ) : 
  total_cats = 16 → 
  percent_black = 25 →
  grey_cats = 10 → 
  (total_cats - (total_cats * percent_black / 100 + grey_cats)) = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_white_cats_l535_53561


namespace NUMINAMATH_GPT_find_a_plus_b_l535_53547

-- Definitions for the conditions
variables {a b : ℝ} (i : ℂ)
def imaginary_unit : Prop := i * i = -1

-- Given condition
def given_equation (a b : ℝ) (i : ℂ) : Prop := (a + 2 * i) / i = b + i

-- Theorem statement
theorem find_a_plus_b (h1 : imaginary_unit i) (h2 : given_equation a b i) : a + b = 1 := 
sorry

end NUMINAMATH_GPT_find_a_plus_b_l535_53547


namespace NUMINAMATH_GPT_polygon_diagonals_regions_l535_53576

theorem polygon_diagonals_regions (n : ℕ) (hn : n ≥ 3) :
  let D := n * (n - 3) / 2
  let P := n * (n - 1) * (n - 2) * (n - 3) / 24
  let R := D + P + 1
  R = n * (n - 1) * (n - 2) * (n - 3) / 24 + n * (n - 3) / 2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_polygon_diagonals_regions_l535_53576


namespace NUMINAMATH_GPT_howard_items_l535_53538

theorem howard_items (a b c : ℕ) (h1 : a + b + c = 40) (h2 : 40 * a + 300 * b + 400 * c = 5000) : a = 20 :=
by
  sorry

end NUMINAMATH_GPT_howard_items_l535_53538


namespace NUMINAMATH_GPT_students_liking_both_l535_53528

theorem students_liking_both (total_students sports_enthusiasts music_enthusiasts neither : ℕ)
  (h1 : total_students = 55)
  (h2: sports_enthusiasts = 43)
  (h3: music_enthusiasts = 34)
  (h4: neither = 4) : 
  ∃ x, ((sports_enthusiasts - x) + x + (music_enthusiasts - x) = total_students - neither) ∧ (x = 22) :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_students_liking_both_l535_53528


namespace NUMINAMATH_GPT_height_of_Joaos_salary_in_kilometers_l535_53553

def real_to_cruzados (reais: ℕ) : ℕ := reais * 2750000000

def stacks (cruzados: ℕ) : ℕ := cruzados / 100

def height_in_cm (stacks: ℕ) : ℕ := stacks * 15

noncomputable def height_in_km (height_cm: ℕ) : ℕ := height_cm / 100000

theorem height_of_Joaos_salary_in_kilometers :
  height_in_km (height_in_cm (stacks (real_to_cruzados 640))) = 264000 :=
by
  sorry

end NUMINAMATH_GPT_height_of_Joaos_salary_in_kilometers_l535_53553


namespace NUMINAMATH_GPT_cole_avg_speed_back_home_l535_53529

noncomputable def avg_speed_back_home 
  (speed_to_work : ℚ) 
  (total_round_trip_time : ℚ) 
  (time_to_work : ℚ) 
  (time_in_minutes : ℚ) :=
  let time_to_work_hours := time_to_work / time_in_minutes
  let distance_to_work := speed_to_work * time_to_work_hours
  let time_back_home := total_round_trip_time - time_to_work_hours
  distance_to_work / time_back_home

theorem cole_avg_speed_back_home :
  avg_speed_back_home 75 1 (35/60) 60 = 105 := 
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_cole_avg_speed_back_home_l535_53529


namespace NUMINAMATH_GPT_initial_number_correct_l535_53555

-- Define the relevant values
def x : ℝ := 53.33
def initial_number : ℝ := 319.98

-- Define the conditions in Lean with appropriate constraints
def conditions (n : ℝ) (x : ℝ) : Prop :=
  x = n / 2 / 3

-- Theorem stating that 319.98 divided by 2 and then by 3 results in 53.33
theorem initial_number_correct : conditions initial_number x :=
by
  unfold conditions
  sorry

end NUMINAMATH_GPT_initial_number_correct_l535_53555


namespace NUMINAMATH_GPT_bamboo_sections_length_l535_53564

variable {n d : ℕ} (a : ℕ → ℕ)
variable (h_arith : ∀ k, a (k + 1) = a k + d)
variable (h_top : a 1 = 10)
variable (h_sum_last_three : a n + a (n - 1) + a (n - 2) = 114)
variable (h_geom_6 : (a 6) ^ 2 = a 1 * a n)

theorem bamboo_sections_length : n = 16 := 
by 
  sorry

end NUMINAMATH_GPT_bamboo_sections_length_l535_53564


namespace NUMINAMATH_GPT_wrapping_paper_per_present_l535_53509

theorem wrapping_paper_per_present :
  (1 / 2) / 5 = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_per_present_l535_53509


namespace NUMINAMATH_GPT_day_of_week_306_2003_l535_53534

-- Note: Definitions to support the conditions and the proof
def day_of_week (n : ℕ) : ℕ := n % 7

-- Theorem statement: Given conditions lead to the conclusion that the 306th day of the year 2003 falls on a Sunday
theorem day_of_week_306_2003 :
  (day_of_week (15) = 2) → (day_of_week (306) = 0) :=
by sorry

end NUMINAMATH_GPT_day_of_week_306_2003_l535_53534


namespace NUMINAMATH_GPT_count_valid_n_l535_53530

theorem count_valid_n : 
  ∃ (count : ℕ), count = 88 ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 2000 ∧ 
   (∃ (a b : ℤ), a + b = -2 ∧ a * b = -n) ↔ 
   ∃ m, 1 ≤ m ∧ m ≤ 2000 ∧ (∃ a, a * (a + 2) = m)) := 
sorry

end NUMINAMATH_GPT_count_valid_n_l535_53530


namespace NUMINAMATH_GPT_klinker_daughter_age_l535_53584

-- Define the conditions in Lean
variable (D : ℕ) -- ℕ is the natural number type in Lean

-- Define the theorem statement
theorem klinker_daughter_age (h1 : 35 + 15 = 50)
    (h2 : 50 = 2 * (D + 15)) : D = 10 := by
  sorry

end NUMINAMATH_GPT_klinker_daughter_age_l535_53584


namespace NUMINAMATH_GPT_students_with_all_three_pets_correct_l535_53573

noncomputable def students_with_all_three_pets (total_students dog_owners cat_owners bird_owners dog_and_cat_owners cat_and_bird_owners dog_and_bird_owners : ℕ) : ℕ :=
  total_students - (dog_owners + cat_owners + bird_owners - dog_and_cat_owners - cat_and_bird_owners - dog_and_bird_owners)

theorem students_with_all_three_pets_correct : 
  students_with_all_three_pets 50 30 35 10 8 5 3 = 7 :=
by
  rw [students_with_all_three_pets]
  norm_num
  sorry

end NUMINAMATH_GPT_students_with_all_three_pets_correct_l535_53573


namespace NUMINAMATH_GPT_solution_set_inequality_l535_53587

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom increasing_on_positive : ∀ {x y : ℝ}, 0 < x → x < y → f x < f y
axiom f_one : f 1 = 0

theorem solution_set_inequality :
  {x : ℝ | (f x) / x < 0} = {x : ℝ | x < -1} ∪ {x | 0 < x ∧ x < 1} := sorry

end NUMINAMATH_GPT_solution_set_inequality_l535_53587


namespace NUMINAMATH_GPT_clever_seven_year_count_l535_53520

def isCleverSevenYear (y : Nat) : Bool :=
  let d1 := y / 1000
  let d2 := (y % 1000) / 100
  let d3 := (y % 100) / 10
  let d4 := y % 10
  d1 + d2 + d3 + d4 = 7

theorem clever_seven_year_count : 
  ∃ n, n = 21 ∧ ∀ y, 2000 ≤ y ∧ y ≤ 2999 → isCleverSevenYear y = true ↔ n = 21 :=
by 
  sorry

end NUMINAMATH_GPT_clever_seven_year_count_l535_53520


namespace NUMINAMATH_GPT_therapy_charge_l535_53593

-- Let F be the charge for the first hour and A be the charge for each additional hour
-- Two conditions are:
-- 1. F = A + 40
-- 2. F + 4A = 375

-- We need to prove that the total charge for 2 hours of therapy is 174
theorem therapy_charge (A F : ℕ) (h1 : F = A + 40) (h2 : F + 4 * A = 375) :
  F + A = 174 :=
by
  sorry

end NUMINAMATH_GPT_therapy_charge_l535_53593


namespace NUMINAMATH_GPT_negative_subtraction_result_l535_53510

theorem negative_subtraction_result : -2 - 1 = -3 := 
by
  -- The proof is not required by the prompt, so we use "sorry" to indicate the unfinished proof.
  sorry

end NUMINAMATH_GPT_negative_subtraction_result_l535_53510


namespace NUMINAMATH_GPT_range_of_d_largest_S_n_l535_53552

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)
variable (d a_1 : ℝ)

-- Conditions
axiom a_3_eq_12 : a_n 3 = 12
axiom S_12_pos : S_n 12 > 0
axiom S_13_neg : S_n 13 < 0
axiom arithmetic_sequence : ∀ n, a_n n = a_1 + (n - 1) * d
axiom sum_of_terms : ∀ n, S_n n = n * a_1 + (n * (n - 1)) / 2 * d

-- Problems
theorem range_of_d : -24/7 < d ∧ d < -3 := sorry

theorem largest_S_n : (∀ m, m > 0 ∧ m < 13 → (S_n 6 >= S_n m)) := sorry

end NUMINAMATH_GPT_range_of_d_largest_S_n_l535_53552


namespace NUMINAMATH_GPT_remainder_problem_l535_53531

def rem (x y : ℚ) := x - y * (⌊x / y⌋ : ℤ)

theorem remainder_problem :
  let x := (5 : ℚ) / 9
  let y := -(3 : ℚ) / 7
  rem x y = (-19 : ℚ) / 63 :=
by
  let x := (5 : ℚ) / 9
  let y := -(3 : ℚ) / 7
  sorry

end NUMINAMATH_GPT_remainder_problem_l535_53531


namespace NUMINAMATH_GPT_red_flowers_needed_l535_53508

-- Define the number of white and red flowers
def white_flowers : ℕ := 555
def red_flowers : ℕ := 347

-- Define the problem statement.
theorem red_flowers_needed : red_flowers + 208 = white_flowers := by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_red_flowers_needed_l535_53508


namespace NUMINAMATH_GPT_f_9_over_2_l535_53556

noncomputable def f (x : ℝ) : ℝ := sorry -- The function f(x) is to be defined later according to conditions

theorem f_9_over_2 :
  (∀ x : ℝ, f (x + 1) = -f (-x + 1)) ∧ -- f(x+1) is odd
  (∀ x : ℝ, f (x + 2) = f (-x + 2)) ∧ -- f(x+2) is even
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = -2 * x^2 + 2) ∧ -- f(x) = ax^2 + b, where a = -2 and b = 2
  (f 0 + f 3 = 6) → -- Sum f(0) and f(3)
  f (9 / 2) = 5 / 2 := 
by {
  sorry -- The proof is omitted as per the instruction
}

end NUMINAMATH_GPT_f_9_over_2_l535_53556


namespace NUMINAMATH_GPT_exists_n_le_2500_perfect_square_l535_53559

noncomputable def sum_of_squares (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def sum_of_squares_segment (n : ℕ) : ℚ :=
  ((26 * n^3 + 12 * n^2 + n) / 3)

theorem exists_n_le_2500_perfect_square :
  ∃ (n : ℕ), n ≤ 2500 ∧ ∃ (k : ℚ), k^2 = (sum_of_squares n) * (sum_of_squares_segment n) :=
sorry

end NUMINAMATH_GPT_exists_n_le_2500_perfect_square_l535_53559


namespace NUMINAMATH_GPT_isosceles_triangle_side_length_l535_53536

theorem isosceles_triangle_side_length (n : ℕ) : 
  (∃ a b : ℕ, a ≠ 4 ∧ b ≠ 4 ∧ (a = b ∨ a = 4 ∨ b = 4) ∧ 
  a^2 - 6*a + n = 0 ∧ b^2 - 6*b + n = 0) → 
  (n = 8 ∨ n = 9) := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_side_length_l535_53536


namespace NUMINAMATH_GPT_problem1_problem2_l535_53551

noncomputable def vec (α : ℝ) (β : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (Real.cos α, Real.sin α, Real.cos β, -Real.sin β)

theorem problem1 (α β : ℝ) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : (Real.sqrt ((Real.cos α - Real.cos β) ^ 2 + (Real.sin α + Real.sin β) ^ 2)) = (Real.sqrt 10) / 5) :
  Real.cos (α + β) = 4 / 5 :=
by
  sorry

theorem problem2 (α β : ℝ) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.cos α = 3 / 5) (h4 : Real.cos (α + β) = 4 / 5) :
  Real.cos β = 24 / 25 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l535_53551


namespace NUMINAMATH_GPT_part1_part2_l535_53596

-- Part (1) prove maximum value of 4 - 2x - 1/x when x > 0 is 0.
theorem part1 (x : ℝ) (h : 0 < x) : 
  4 - 2 * x - (2 / x) ≤ 0 :=
sorry

-- Part (2) prove minimum value of 1/a + 1/b when a + 2b = 1 and a > 0, b > 0 is 3 + 2 * sqrt 2.
theorem part2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + 2 * b = 1) :
  (1 / a) + (1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l535_53596


namespace NUMINAMATH_GPT_grade3_trees_count_l535_53582

-- Declare the variables and types
variables (x y : ℕ)

-- Given conditions as definitions
def students_equation := (2 * x + y = 100)
def trees_equation := (9 * x + (13 / 2) * y = 566)
def avg_trees_grade3 := 4

-- Assert the problem statement
theorem grade3_trees_count (hx : students_equation x y) (hy : trees_equation x y) : 
  (avg_trees_grade3 * x = 84) :=
sorry

end NUMINAMATH_GPT_grade3_trees_count_l535_53582


namespace NUMINAMATH_GPT_trajectory_of_point_l535_53519

theorem trajectory_of_point (P : ℝ × ℝ) 
  (h1 : dist P (0, 3) = dist P (x1, -3)) :
  ∃ p > 0, (P.fst)^2 = 2 * p * P.snd ∧ p = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_trajectory_of_point_l535_53519


namespace NUMINAMATH_GPT_combined_cost_is_3490_l535_53560

-- Definitions for the quantities of gold each person has and their respective prices per gram
def Gary_gold_grams : ℕ := 30
def Gary_gold_price_per_gram : ℕ := 15

def Anna_gold_grams : ℕ := 50
def Anna_gold_price_per_gram : ℕ := 20

def Lisa_gold_grams : ℕ := 40
def Lisa_gold_price_per_gram : ℕ := 18

def John_gold_grams : ℕ := 60
def John_gold_price_per_gram : ℕ := 22

-- Combined cost
def combined_cost : ℕ :=
  Gary_gold_grams * Gary_gold_price_per_gram +
  Anna_gold_grams * Anna_gold_price_per_gram +
  Lisa_gold_grams * Lisa_gold_price_per_gram +
  John_gold_grams * John_gold_price_per_gram

-- Proof that the combined cost is equal to $3490
theorem combined_cost_is_3490 : combined_cost = 3490 :=
  by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_combined_cost_is_3490_l535_53560


namespace NUMINAMATH_GPT_integer_solutions_range_l535_53562

def operation (p q : ℝ) : ℝ := p + q - p * q

theorem integer_solutions_range (m : ℝ) :
  (∃ (x1 x2 : ℤ), (operation 2 x1 > 0) ∧ (operation x1 3 ≤ m) ∧ (operation 2 x2 > 0) ∧ (operation x2 3 ≤ m) ∧ (x1 ≠ x2)) ↔ (3 ≤ m ∧ m < 5) :=
by sorry

end NUMINAMATH_GPT_integer_solutions_range_l535_53562


namespace NUMINAMATH_GPT_probability_diff_colors_l535_53575

theorem probability_diff_colors (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  total_balls = 4 ∧ white_balls = 3 ∧ black_balls = 1 ∧ drawn_balls = 2 ∧ 
  total_outcomes = Nat.choose 4 2 ∧ favorable_outcomes = Nat.choose 3 1 * Nat.choose 1 1
  → favorable_outcomes / total_outcomes = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_diff_colors_l535_53575


namespace NUMINAMATH_GPT_candies_per_friend_l535_53597

theorem candies_per_friend (initial_candies : ℕ) (additional_candies : ℕ) (friends : ℕ) 
  (h_initial : initial_candies = 10)
  (h_additional : additional_candies = 4)
  (h_friends : friends = 7) : initial_candies + additional_candies = 14 ∧ 14 / friends = 2 :=
by
  sorry

end NUMINAMATH_GPT_candies_per_friend_l535_53597


namespace NUMINAMATH_GPT_arithmetic_progression_square_l535_53580

theorem arithmetic_progression_square (a b c : ℝ) (h : b - a = c - b) : a^2 + 8 * b * c = (2 * b + c)^2 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_square_l535_53580


namespace NUMINAMATH_GPT_calculate_solution_volume_l535_53525

theorem calculate_solution_volume (V : ℝ) (h : 0.35 * V = 1.4) : V = 4 :=
sorry

end NUMINAMATH_GPT_calculate_solution_volume_l535_53525


namespace NUMINAMATH_GPT_intersection_A_B_l535_53572

-- Define the conditions of set A and B using the given inequalities and constraints
def set_A : Set ℤ := {x | -2 < x ∧ x < 3}
def set_B : Set ℤ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the proof problem translating conditions and question to Lean
theorem intersection_A_B : (set_A ∩ set_B) = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l535_53572


namespace NUMINAMATH_GPT_determine_m_even_function_l535_53539

theorem determine_m_even_function (m : ℤ) :
  (∀ x : ℤ, (x^2 + (m-1)*x) = (x^2 - (m-1)*x)) → m = 1 :=
by
    sorry

end NUMINAMATH_GPT_determine_m_even_function_l535_53539


namespace NUMINAMATH_GPT_marcus_savings_l535_53574

theorem marcus_savings
  (running_shoes_price : ℝ)
  (running_shoes_discount : ℝ)
  (cashback : ℝ)
  (running_shoes_tax_rate : ℝ)
  (athletic_socks_price : ℝ)
  (athletic_socks_tax_rate : ℝ)
  (bogo : ℝ)
  (performance_tshirt_price : ℝ)
  (performance_tshirt_discount : ℝ)
  (performance_tshirt_tax_rate : ℝ)
  (total_budget : ℝ)
  (running_shoes_final_price : ℝ)
  (athletic_socks_final_price : ℝ)
  (performance_tshirt_final_price : ℝ) :
  running_shoes_price = 120 →
  running_shoes_discount = 30 / 100 →
  cashback = 10 →
  running_shoes_tax_rate = 8 / 100 →
  athletic_socks_price = 25 →
  athletic_socks_tax_rate = 6 / 100 →
  bogo = 2 →
  performance_tshirt_price = 55 →
  performance_tshirt_discount = 10 / 100 →
  performance_tshirt_tax_rate = 7 / 100 →
  total_budget = 250 →
  running_shoes_final_price = (running_shoes_price * (1 - running_shoes_discount) - cashback) * (1 + running_shoes_tax_rate) →
  athletic_socks_final_price = (athletic_socks_price * bogo) * (1 + athletic_socks_tax_rate) / bogo →
  performance_tshirt_final_price = (performance_tshirt_price * (1 - performance_tshirt_discount)) * (1 + performance_tshirt_tax_rate) →
  total_budget - (running_shoes_final_price + athletic_socks_final_price + performance_tshirt_final_price) = 103.86 :=
sorry

end NUMINAMATH_GPT_marcus_savings_l535_53574


namespace NUMINAMATH_GPT_minimum_expression_value_l535_53512

noncomputable def expr (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  (2 * (Real.sin x₁)^2 + 1 / (Real.sin x₁)^2) *
  (2 * (Real.sin x₂)^2 + 1 / (Real.sin x₂)^2) *
  (2 * (Real.sin x₃)^2 + 1 / (Real.sin x₃)^2) *
  (2 * (Real.sin x₄)^2 + 1 / (Real.sin x₄)^2)

theorem minimum_expression_value :
  ∀ (x₁ x₂ x₃ x₄ : ℝ),
  x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧
  x₁ + x₂ + x₃ + x₄ = Real.pi →
  expr x₁ x₂ x₃ x₄ ≥ 81 := sorry

end NUMINAMATH_GPT_minimum_expression_value_l535_53512


namespace NUMINAMATH_GPT_isosceles_triangle_l535_53513

theorem isosceles_triangle 
  (α β γ : ℝ) 
  (a b : ℝ) 
  (h_sum : a + b = (Real.tan (γ / 2)) * (a * (Real.tan α) + b * (Real.tan β)))
  (h_sum_angles : α + β + γ = π) 
  (zero_lt_γ : 0 < γ ∧ γ < π) 
  (zero_lt_α : 0 < α ∧ α < π / 2) 
  (zero_lt_β : 0 < β ∧ β < π / 2) : 
  α = β := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_l535_53513


namespace NUMINAMATH_GPT_probability_cheryl_same_color_l535_53502

theorem probability_cheryl_same_color :
  let total_marble_count := 12
  let marbles_per_color := 3
  let carol_draw := 3
  let claudia_draw := 3
  let cheryl_draw := total_marble_count - carol_draw - claudia_draw
  let num_colors := 4

  0 < marbles_per_color ∧ marbles_per_color * num_colors = total_marble_count ∧
  0 < carol_draw ∧ carol_draw <= total_marble_count ∧
  0 < claudia_draw ∧ claudia_draw <= total_marble_count - carol_draw ∧
  0 < cheryl_draw ∧ cheryl_draw <= total_marble_count - carol_draw - claudia_draw ∧
  num_colors * (num_colors - 1) > 0
  →
  ∃ (p : ℚ), p = 2 / 55 := 
sorry

end NUMINAMATH_GPT_probability_cheryl_same_color_l535_53502


namespace NUMINAMATH_GPT_total_money_l535_53557

variable (Sally Jolly Molly : ℕ)

-- Conditions
def condition1 (Sally : ℕ) : Prop := Sally - 20 = 80
def condition2 (Jolly : ℕ) : Prop := Jolly + 20 = 70
def condition3 (Molly : ℕ) : Prop := Molly + 30 = 100

-- The theorem to prove
theorem total_money (h1: condition1 Sally)
                    (h2: condition2 Jolly)
                    (h3: condition3 Molly) :
  Sally + Jolly + Molly = 220 :=
by
  sorry

end NUMINAMATH_GPT_total_money_l535_53557


namespace NUMINAMATH_GPT_amount_given_to_beggar_l535_53591

variable (X : ℕ)
variable (pennies_initial : ℕ := 42)
variable (pennies_to_farmer : ℕ := 22)
variable (pennies_after_farmer : ℕ := 20)

def amount_to_boy (X : ℕ) : ℕ :=
  (20 - X) / 2 + 3

theorem amount_given_to_beggar : 
  (X = 12) →  (pennies_initial - pennies_to_farmer - X) / 2 + 3 + 1 = pennies_initial - pennies_to_farmer - X :=
by
  intro h
  subst h
  sorry

end NUMINAMATH_GPT_amount_given_to_beggar_l535_53591


namespace NUMINAMATH_GPT_average_words_written_l535_53504

def total_words : ℕ := 50000
def total_hours : ℕ := 100
def average_words_per_hour : ℕ := total_words / total_hours

theorem average_words_written :
  average_words_per_hour = 500 := 
by
  sorry

end NUMINAMATH_GPT_average_words_written_l535_53504


namespace NUMINAMATH_GPT_exists_solution_real_l535_53518

theorem exists_solution_real (m : ℝ) :
  (∃ x y : ℝ, y = (m + 1) * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_exists_solution_real_l535_53518


namespace NUMINAMATH_GPT_graphs_intersection_count_l535_53589

theorem graphs_intersection_count (g : ℝ → ℝ) (hg : Function.Injective g) :
  ∃ S : Finset ℝ, (∀ x ∈ S, g (x^3) = g (x^5)) ∧ S.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_graphs_intersection_count_l535_53589


namespace NUMINAMATH_GPT_susan_strawberries_l535_53521

def strawberries_picked (total_in_basket : ℕ) (handful_size : ℕ) (eats_per_handful : ℕ) : ℕ :=
  let strawberries_per_handful := handful_size - eats_per_handful
  (total_in_basket / strawberries_per_handful) * handful_size

theorem susan_strawberries : strawberries_picked 60 5 1 = 75 := by
  sorry

end NUMINAMATH_GPT_susan_strawberries_l535_53521


namespace NUMINAMATH_GPT_shaded_area_is_110_l535_53523

-- Definitions based on conditions
def equilateral_triangle_area : ℕ := 10
def num_triangles_small : ℕ := 1
def num_triangles_medium : ℕ := 3
def num_triangles_large : ℕ := 7

-- Total area calculation
def total_area : ℕ := (num_triangles_small + num_triangles_medium + num_triangles_large) * equilateral_triangle_area

-- The theorem statement
theorem shaded_area_is_110 : total_area = 110 := 
by 
  sorry

end NUMINAMATH_GPT_shaded_area_is_110_l535_53523


namespace NUMINAMATH_GPT_water_jugs_problem_l535_53565

-- Definitions based on the conditions
variables (m n : ℕ) (relatively_prime_m_n : Nat.gcd m n = 1)
variables (k : ℕ) (hk : 1 ≤ k ∧ k ≤ m + n)

-- Statement of the theorem
theorem water_jugs_problem : 
    ∃ (x y z : ℕ), 
    (x = m ∨ x = n ∨ x = m + n) ∧ 
    (y = m ∨ y = n ∨ y = m + n) ∧ 
    (z = m ∨ z = n ∨ z = m + n) ∧ 
    (x ≤ m + n) ∧ 
    (y ≤ m + n) ∧ 
    (z ≤ m + n) ∧ 
    x + y + z = m + n ∧ 
    (x = k ∨ y = k ∨ z = k) :=
sorry

end NUMINAMATH_GPT_water_jugs_problem_l535_53565


namespace NUMINAMATH_GPT_train_speed_identification_l535_53585

-- Define the conditions
def train_length : ℕ := 300
def crossing_time : ℕ := 30

-- Define the speed calculation
def calculate_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The target theorem stating the speed of the train
theorem train_speed_identification : calculate_speed train_length crossing_time = 10 := 
by 
  sorry

end NUMINAMATH_GPT_train_speed_identification_l535_53585


namespace NUMINAMATH_GPT_find_g_1_l535_53586

noncomputable def g (x : ℝ) : ℝ := sorry -- express g(x) as a 4th degree polynomial with unknown coefficients

-- Conditions given in the problem
axiom cond1 : |g (-1)| = 15
axiom cond2 : |g (0)| = 15
axiom cond3 : |g (2)| = 15
axiom cond4 : |g (3)| = 15
axiom cond5 : |g (4)| = 15

-- The statement we need to prove
theorem find_g_1 : |g 1| = 11 :=
sorry

end NUMINAMATH_GPT_find_g_1_l535_53586


namespace NUMINAMATH_GPT_jumpy_implies_not_green_l535_53595

variables (Lizard : Type)
variables (IsJumpy IsGreen CanSing CanDance : Lizard → Prop)

-- Conditions given in the problem
axiom jumpy_implies_can_sing : ∀ l, IsJumpy l → CanSing l
axiom green_implies_cannot_dance : ∀ l, IsGreen l → ¬ CanDance l
axiom cannot_dance_implies_cannot_sing : ∀ l, ¬ CanDance l → ¬ CanSing l

theorem jumpy_implies_not_green (l : Lizard) : IsJumpy l → ¬ IsGreen l :=
by
  sorry

end NUMINAMATH_GPT_jumpy_implies_not_green_l535_53595


namespace NUMINAMATH_GPT_find_x_l535_53590

theorem find_x (x y : ℝ) (h₁ : x - y = 10) (h₂ : x + y = 14) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l535_53590


namespace NUMINAMATH_GPT_indeterminate_equation_solution_exists_l535_53554

theorem indeterminate_equation_solution_exists
  (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * c = b^2 + b + 1) :
  ∃ x y : ℤ, a * x^2 - (2 * b + 1) * x * y + c * y^2 = 1 := by
  sorry

end NUMINAMATH_GPT_indeterminate_equation_solution_exists_l535_53554


namespace NUMINAMATH_GPT_inv_eq_self_l535_53578

noncomputable def g (m x : ℝ) : ℝ := (3 * x + 4) / (m * x - 3)

theorem inv_eq_self (m : ℝ) :
  (∀ x : ℝ, g m x = g m (g m x)) ↔ m ∈ Set.Iic (-9 / 4) ∪ Set.Ici (-9 / 4) :=
by
  sorry

end NUMINAMATH_GPT_inv_eq_self_l535_53578


namespace NUMINAMATH_GPT_bisection_approximation_interval_l535_53598

noncomputable def bisection_accuracy (a b : ℝ) (n : ℕ) : ℝ := (b - a) / 2^n

theorem bisection_approximation_interval 
  (a b : ℝ) (n : ℕ) (accuracy : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : accuracy = 0.01) 
  (h4 : 2^n ≥ 100) : bisection_accuracy a b n ≤ accuracy :=
sorry

end NUMINAMATH_GPT_bisection_approximation_interval_l535_53598


namespace NUMINAMATH_GPT_infinitely_many_a_l535_53558

theorem infinitely_many_a (n : ℕ) : ∃ (a : ℕ), ∃ (k : ℕ), ∀ n : ℕ, n^6 + 3 * (3 * n^4 * k + 9 * n^2 * k^2 + 9 * k^3) = (n^2 + 3 * k)^3 :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_a_l535_53558


namespace NUMINAMATH_GPT_east_high_school_students_l535_53599

theorem east_high_school_students (S : ℝ) 
  (h1 : 0.52 * S * 0.125 = 26) :
  S = 400 :=
by
  -- The proof is omitted for this exercise
  sorry

end NUMINAMATH_GPT_east_high_school_students_l535_53599


namespace NUMINAMATH_GPT_sequence_general_formula_l535_53524

theorem sequence_general_formula (a : ℕ → ℤ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n > 0 → a (n + 1) > a n)
  (h3 : ∀ n : ℕ, n > 0 → (a (n + 1))^2 - 2 * a n * a (n + 1) + (a n)^2 = 1) :
  ∀ n : ℕ, n > 0 → a n = n :=
by 
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l535_53524


namespace NUMINAMATH_GPT_minimum_digits_for_divisibility_l535_53550

theorem minimum_digits_for_divisibility :
  ∃ n : ℕ, (10 * 2013 + n) % 2520 = 0 ∧ n < 1000 :=
sorry

end NUMINAMATH_GPT_minimum_digits_for_divisibility_l535_53550


namespace NUMINAMATH_GPT_divide_polynomials_l535_53541

theorem divide_polynomials (n : ℕ) (h : ∃ (k : ℤ), n^2 + 3*n + 51 = 13 * k) : 
  ∃ (m : ℤ), 21*n^2 + 89*n + 44 = 169 * m := by
  sorry

end NUMINAMATH_GPT_divide_polynomials_l535_53541


namespace NUMINAMATH_GPT_least_multiple_of_seven_not_lucky_is_14_l535_53583

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky_integer (n : ℕ) : Prop :=
  n > 0 ∧ n % sum_of_digits n = 0

def is_multiple_of_seven_not_lucky (n : ℕ) : Prop :=
  n % 7 = 0 ∧ ¬ is_lucky_integer n

theorem least_multiple_of_seven_not_lucky_is_14 : 
  ∃ n : ℕ, is_multiple_of_seven_not_lucky n ∧ ∀ m, (is_multiple_of_seven_not_lucky m → n ≤ m) :=
⟨ 14, 
  by {
    -- Proof is provided here, but for now, we use "sorry"
    sorry
  }⟩

end NUMINAMATH_GPT_least_multiple_of_seven_not_lucky_is_14_l535_53583


namespace NUMINAMATH_GPT_max_value_expression_l535_53563

theorem max_value_expression (a b c : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  2 * a * b * Real.sqrt 3 + 2 * b * c ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_value_expression_l535_53563


namespace NUMINAMATH_GPT_find_sqrt_abc_sum_l535_53542

theorem find_sqrt_abc_sum (a b c : ℝ)
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_GPT_find_sqrt_abc_sum_l535_53542


namespace NUMINAMATH_GPT_surface_area_of_sphere_given_cube_volume_8_l535_53568

theorem surface_area_of_sphere_given_cube_volume_8 
  (volume_of_cube : ℝ)
  (h₁ : volume_of_cube = 8) :
  ∃ (surface_area_of_sphere : ℝ), 
  surface_area_of_sphere = 12 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_sphere_given_cube_volume_8_l535_53568


namespace NUMINAMATH_GPT_area_at_stage_8_l535_53514

-- Defining the constants and initial settings
def first_term : ℕ := 1
def common_difference : ℕ := 1
def stage : ℕ := 8
def square_side_length : ℕ := 4

-- Calculating the number of squares at the given stage
def num_squares : ℕ := first_term + (stage - 1) * common_difference

--Calculating the area of one square
def area_one_square : ℕ := square_side_length * square_side_length

-- Calculating the total area at the given stage
def total_area : ℕ := num_squares * area_one_square

-- Proving the total area equals 128 at Stage 8
theorem area_at_stage_8 : total_area = 128 := 
by
  sorry

end NUMINAMATH_GPT_area_at_stage_8_l535_53514


namespace NUMINAMATH_GPT_projection_of_c_onto_b_l535_53569

open Real

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := sqrt (b.1^2 + b.2^2)
  let scalar := dot_product / magnitude_b
  (scalar * b.1 / magnitude_b, scalar * b.2 / magnitude_b)

theorem projection_of_c_onto_b :
  let a := (2, 3)
  let b := (-4, 7)
  let c := (-a.1, -a.2)
  vector_projection c b = (-sqrt 65 / 5, -sqrt 65 / 5) :=
by sorry

end NUMINAMATH_GPT_projection_of_c_onto_b_l535_53569


namespace NUMINAMATH_GPT_intersection_eq_l535_53544

def U : Set ℝ := {x : ℝ | True}
def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x ≥ 1}
def CU_N : Set ℝ := {x : ℝ | x < 1}

theorem intersection_eq : M ∩ CU_N = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l535_53544


namespace NUMINAMATH_GPT_find_k_l535_53543

theorem find_k (k : ℝ) (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0)
  (h3 : r / s = 3) (h4 : r + s = 4) (h5 : r * s = k) : k = 3 :=
sorry

end NUMINAMATH_GPT_find_k_l535_53543


namespace NUMINAMATH_GPT_problem_statement_l535_53533

theorem problem_statement (f : ℕ → ℕ) (h1 : f 1 = 4) (h2 : ∀ x y, f (x + y) = f x + f y + 7 * x * y + 4) :
  f 2 + f 5 = 125 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l535_53533


namespace NUMINAMATH_GPT_symmetric_points_y_axis_l535_53571

theorem symmetric_points_y_axis (a b : ℝ) (h₁ : ∃ M N : ℝ × ℝ, M = (a, 3) ∧ N = (4, b) ∧ M.1 = -N.1 ∧ M.2 = N.2) :
  (a + b) ^ 2012 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_symmetric_points_y_axis_l535_53571


namespace NUMINAMATH_GPT_square_free_condition_l535_53526

/-- Define square-free integer -/
def square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

/-- Define the problem in Lean -/
theorem square_free_condition (p : ℕ) (hp : p ≥ 3 ∧ Nat.Prime p) :
  (∀ q : ℕ, Nat.Prime q ∧ q < p → square_free (p - (p / q) * q)) ↔
  p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 13 := by
  sorry

end NUMINAMATH_GPT_square_free_condition_l535_53526


namespace NUMINAMATH_GPT_compute_modulo_l535_53588

theorem compute_modulo :
    (2015 % 7) = 3 ∧ (2016 % 7) = 4 ∧ (2017 % 7) = 5 ∧ (2018 % 7) = 6 →
    (2015 * 2016 * 2017 * 2018) % 7 = 3 :=
by
  intros h
  have h1 := h.left
  have h2 := h.right.left
  have h3 := h.right.right.left
  have h4 := h.right.right.right
  sorry

end NUMINAMATH_GPT_compute_modulo_l535_53588


namespace NUMINAMATH_GPT_k_h_neg3_l535_53506

-- Definitions of h and k
def h (x : ℝ) : ℝ := 4 * x^2 - 12

variable (k : ℝ → ℝ) -- function k with range an ℝ

-- Given k(h(3)) = 16
axiom k_h_3 : k (h 3) = 16

-- Prove that k(h(-3)) = 16
theorem k_h_neg3 : k (h (-3)) = 16 :=
sorry

end NUMINAMATH_GPT_k_h_neg3_l535_53506


namespace NUMINAMATH_GPT_ribbon_segment_length_l535_53507

theorem ribbon_segment_length :
  ∀ (ribbon_length : ℚ) (segments : ℕ), ribbon_length = 4/5 → segments = 3 → 
  (ribbon_length / segments) = 4/15 :=
by
  intros ribbon_length segments h1 h2
  sorry

end NUMINAMATH_GPT_ribbon_segment_length_l535_53507


namespace NUMINAMATH_GPT_distance_between_fourth_and_work_l535_53548

theorem distance_between_fourth_and_work (x : ℝ) (h₁ : x > 0) :
  let total_distance := x + 0.5 * x + 2 * x
  let to_fourth := (1 / 3) * total_distance
  let total_to_fourth := total_distance + to_fourth
  3 * total_to_fourth = 14 * x :=
by
  sorry

end NUMINAMATH_GPT_distance_between_fourth_and_work_l535_53548


namespace NUMINAMATH_GPT_acuteAnglesSum_l535_53579

theorem acuteAnglesSum (A B C : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) 
  (hC : 0 < C ∧ C < π / 2) (h : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 1) :
  π / 2 ≤ A + B + C ∧ A + B + C ≤ π :=
by
  sorry

end NUMINAMATH_GPT_acuteAnglesSum_l535_53579


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l535_53594

/-- Let {a_n} be an arithmetic sequence with a positive common difference d.
  Given that a_1 + a_2 + a_3 = 15 and a_1 * a_2 * a_3 = 80, we aim to show that
  a_11 + a_12 + a_13 = 105. -/
theorem arithmetic_sequence_sum
  (a : ℕ → ℚ)
  (d : ℚ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d > 0)
  (h3 : a 1 + a 2 + a 3 = 15)
  (h4 : a 1 * a 2 * a 3 = 80) :
  a 11 + a 12 + a 13 = 105 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l535_53594


namespace NUMINAMATH_GPT_complement_U_M_l535_53522

noncomputable def U : Set ℝ := {x : ℝ | x > 0}

noncomputable def M : Set ℝ := {x : ℝ | 2 * x - x^2 > 0}

theorem complement_U_M : (U \ M) = {x : ℝ | x ≥ 2} := 
by
  sorry

end NUMINAMATH_GPT_complement_U_M_l535_53522


namespace NUMINAMATH_GPT_odd_function_example_l535_53532

theorem odd_function_example (f : ℝ → ℝ)
    (h_odd : ∀ x, f (-x) = -f x)
    (h_neg : ∀ x, x < 0 → f x = x + 2) : f 0 + f 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_example_l535_53532


namespace NUMINAMATH_GPT_plums_in_basket_l535_53537

theorem plums_in_basket (initial : ℕ) (added : ℕ) (total : ℕ) (h_initial : initial = 17) (h_added : added = 4) : total = 21 := by
  sorry

end NUMINAMATH_GPT_plums_in_basket_l535_53537


namespace NUMINAMATH_GPT_music_player_and_concert_tickets_l535_53546

theorem music_player_and_concert_tickets (n : ℕ) (h1 : 35 % 5 = 0) (h2 : 35 % n = 0) (h3 : ∀ m : ℕ, m < 35 → (m % 5 ≠ 0 ∨ m % n ≠ 0)) : n = 7 :=
sorry

end NUMINAMATH_GPT_music_player_and_concert_tickets_l535_53546


namespace NUMINAMATH_GPT_sum_of_two_numbers_l535_53516

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 23 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l535_53516


namespace NUMINAMATH_GPT_total_money_divided_l535_53503

theorem total_money_divided (A B C : ℝ) (h1 : A = (1 / 2) * B) (h2 : B = (1 / 2) * C) (h3 : C = 208) :
  A + B + C = 364 := 
sorry

end NUMINAMATH_GPT_total_money_divided_l535_53503
