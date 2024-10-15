import Mathlib

namespace NUMINAMATH_GPT_no_intersect_M1_M2_l517_51784

theorem no_intersect_M1_M2 (A B : ℤ) : ∃ C : ℤ, 
  ∀ x y : ℤ, (x^2 + A * x + B) ≠ (2 * y^2 + 2 * y + C) := by
  sorry

end NUMINAMATH_GPT_no_intersect_M1_M2_l517_51784


namespace NUMINAMATH_GPT_solve_system_of_equations_l517_51716

theorem solve_system_of_equations
  (a b c d x y z u : ℝ)
  (h1 : a^3 * x + a^2 * y + a * z + u = 0)
  (h2 : b^3 * x + b^2 * y + b * z + u = 0)
  (h3 : c^3 * x + c^2 * y + c * z + u = 0)
  (h4 : d^3 * x + d^2 * y + d * z + u = 1) :
  x = 1 / ((d - a) * (d - b) * (d - c)) ∧
  y = -(a + b + c) / ((d - a) * (d - b) * (d - c)) ∧
  z = (a * b + b * c + c * a) / ((d - a) * (d - b) * (d - c)) ∧
  u = - (a * b * c) / ((d - a) * (d - b) * (d - c)) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l517_51716


namespace NUMINAMATH_GPT_subtraction_problem_solution_l517_51791

theorem subtraction_problem_solution :
  ∃ x : ℝ, (8 - x) / (9 - x) = 4 / 5 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_subtraction_problem_solution_l517_51791


namespace NUMINAMATH_GPT_z_in_fourth_quadrant_l517_51719

def complex_quadrant (re im : ℤ) : String :=
  if re > 0 ∧ im > 0 then "First Quadrant"
  else if re < 0 ∧ im > 0 then "Second Quadrant"
  else if re < 0 ∧ im < 0 then "Third Quadrant"
  else if re > 0 ∧ im < 0 then "Fourth Quadrant"
  else "Axis"

theorem z_in_fourth_quadrant : complex_quadrant 2 (-3) = "Fourth Quadrant" :=
by
  sorry

end NUMINAMATH_GPT_z_in_fourth_quadrant_l517_51719


namespace NUMINAMATH_GPT_pairs_of_polygons_with_angle_ratio_l517_51744

theorem pairs_of_polygons_with_angle_ratio :
  ∃ n, n = 2 ∧ (∀ {k r : ℕ}, (k > 2 ∧ r > 2) → 
  (4 * (180 * r - 360) = 3 * (180 * k - 360) →
  ((k = 3 ∧ r = 18) ∨ (k = 2 ∧ r = 6)))) :=
by
  -- The proof should be provided here, but we skip it
  sorry

end NUMINAMATH_GPT_pairs_of_polygons_with_angle_ratio_l517_51744


namespace NUMINAMATH_GPT_sequences_identity_l517_51712

variables {α β γ : ℤ}
variables {a b : ℕ → ℤ}

-- Define the recurrence relations conditions
def conditions (a b : ℕ → ℤ) (α β γ : ℤ) : Prop :=
  a 0 = 1 ∧ b 0 = 1 ∧
  (∀ n, a (n + 1) = α * a n + β * b n) ∧
  (∀ n, b (n + 1) = β * a n + γ * b n) ∧
  α < γ ∧ α * γ = β^2 + 1

-- Define the main statement
theorem sequences_identity (a b : ℕ → ℤ) 
  (h : conditions a b α β γ) (m n : ℕ) :
  a (m + n) + b (m + n) = a m * a n + b m * b n :=
sorry

end NUMINAMATH_GPT_sequences_identity_l517_51712


namespace NUMINAMATH_GPT_dice_probability_l517_51753

theorem dice_probability (p : ℚ) (h : p = (1 / 42)) : 
  p = 0.023809523809523808 := 
sorry

end NUMINAMATH_GPT_dice_probability_l517_51753


namespace NUMINAMATH_GPT_person_A_money_left_l517_51762

-- We define the conditions and question in terms of Lean types.
def initial_money_ratio : ℚ := 7 / 6
def money_spent_A : ℚ := 50
def money_spent_B : ℚ := 60
def final_money_ratio : ℚ := 3 / 2
def x : ℚ := 30

-- The theorem to prove the amount of money left by person A
theorem person_A_money_left 
  (init_ratio : initial_money_ratio = 7 / 6)
  (spend_A : money_spent_A = 50)
  (spend_B : money_spent_B = 60)
  (final_ratio : final_money_ratio = 3 / 2)
  (hx : x = 30) : 3 * x = 90 := by 
  sorry

end NUMINAMATH_GPT_person_A_money_left_l517_51762


namespace NUMINAMATH_GPT_exists_equal_sum_disjoint_subsets_l517_51755

-- Define the set and conditions
def is_valid_set (S : Finset ℕ) : Prop :=
  S.card = 15 ∧ ∀ x ∈ S, x ≤ 2020

-- Define the problem statement
theorem exists_equal_sum_disjoint_subsets (S : Finset ℕ) (h : is_valid_set S) :
  ∃ (A B : Finset ℕ), A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

end NUMINAMATH_GPT_exists_equal_sum_disjoint_subsets_l517_51755


namespace NUMINAMATH_GPT_find_a_l517_51766

noncomputable def f (x : ℝ) : ℝ := x^2 + 9
noncomputable def g (x : ℝ) : ℝ := x^2 - 5

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 9) : a = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l517_51766


namespace NUMINAMATH_GPT_number_of_possible_schedules_l517_51740

-- Define the six teams
inductive Team : Type
| A | B | C | D | E | F

open Team

-- Define the function to get the number of different schedules possible
noncomputable def number_of_schedules : ℕ := 70

-- Define the theorem statement
theorem number_of_possible_schedules (teams : Finset Team) (play_games : Team → Finset Team) (h : teams.card = 6) 
  (h2 : ∀ t ∈ teams, (play_games t).card = 3 ∧ ∀ t' ∈ (play_games t), t ≠ t') : 
  number_of_schedules = 70 :=
by sorry

end NUMINAMATH_GPT_number_of_possible_schedules_l517_51740


namespace NUMINAMATH_GPT_seq_equality_iff_initial_equality_l517_51722

variable {α : Type*} [AddGroup α]

-- Definition of sequences and their differences
def sequence_diff (u : ℕ → α) (v : ℕ → α) : Prop := ∀ n, (u (n+1) - u n) = (v (n+1) - v n)

-- Main theorem statement
theorem seq_equality_iff_initial_equality (u v : ℕ → α) :
  sequence_diff u v → (∀ n, u n = v n) ↔ (u 1 = v 1) :=
by
  sorry

end NUMINAMATH_GPT_seq_equality_iff_initial_equality_l517_51722


namespace NUMINAMATH_GPT_class3_qualifies_l517_51778

/-- Data structure representing a class's tardiness statistics. -/
structure ClassStats where
  mean : ℕ
  median : ℕ
  variance : ℕ
  mode : Option ℕ -- mode is optional because not all classes might have a unique mode.

def class1 : ClassStats := { mean := 3, median := 3, variance := 0, mode := none }
def class2 : ClassStats := { mean := 2, median := 0, variance := 1, mode := none }
def class3 : ClassStats := { mean := 2, median := 0, variance := 2, mode := none }
def class4 : ClassStats := { mean := 0, median := 2, variance := 0, mode := some 2 }

/-- Predicate to check if a class qualifies for the flag, meaning no more than 5 students tardy each day for 5 consecutive days. -/
def qualifies (cs : ClassStats) : Prop :=
  cs.mean = 2 ∧ cs.variance = 2

theorem class3_qualifies : qualifies class3 :=
by
  sorry

end NUMINAMATH_GPT_class3_qualifies_l517_51778


namespace NUMINAMATH_GPT_find_A_minus_B_l517_51731

variables (A B : ℝ)

-- Define the conditions
def condition1 : Prop := B + A + B = 814.8
def condition2 : Prop := B = A / 10

-- Statement to prove
theorem find_A_minus_B (h1 : condition1 A B) (h2 : condition2 A B) : A - B = 611.1 :=
sorry

end NUMINAMATH_GPT_find_A_minus_B_l517_51731


namespace NUMINAMATH_GPT_range_of_a_l517_51734

open Function

theorem range_of_a (f : ℝ → ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 0 → f x₁ > f x₂) (a : ℝ) (h_gt : f a > f 2) : a < -2 ∨ a > 2 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l517_51734


namespace NUMINAMATH_GPT_janice_purchase_l517_51796

theorem janice_purchase (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 30 * a + 200 * b + 300 * c = 3000) : a = 20 :=
sorry

end NUMINAMATH_GPT_janice_purchase_l517_51796


namespace NUMINAMATH_GPT_man_l517_51779

variable (V_m V_c : ℝ)

theorem man's_speed_against_current :
  (V_m + V_c = 21 ∧ V_c = 2.5) → (V_m - V_c = 16) :=
by
  sorry

end NUMINAMATH_GPT_man_l517_51779


namespace NUMINAMATH_GPT_quadratic_form_l517_51782

-- Define the constants b and c based on the problem conditions
def b : ℤ := 900
def c : ℤ := -807300

-- Create a statement that represents the proof goal
theorem quadratic_form (c_eq : c = -807300) (b_eq : b = 900) : c / b = -897 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_form_l517_51782


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_equals_3_over_22_l517_51787

theorem tan_alpha_plus_pi_over_4_equals_3_over_22
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 :=
sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_equals_3_over_22_l517_51787


namespace NUMINAMATH_GPT_temperature_difference_l517_51795

def highest_temperature : ℤ := 8
def lowest_temperature : ℤ := -2

theorem temperature_difference :
  highest_temperature - lowest_temperature = 10 := by
  sorry

end NUMINAMATH_GPT_temperature_difference_l517_51795


namespace NUMINAMATH_GPT_starting_elevation_l517_51730

variable (rate time final_elevation : ℝ)
variable (h_rate : rate = 10)
variable (h_time : time = 5)
variable (h_final_elevation : final_elevation = 350)

theorem starting_elevation (start_elevation : ℝ) :
  start_elevation = 400 :=
  by
    sorry

end NUMINAMATH_GPT_starting_elevation_l517_51730


namespace NUMINAMATH_GPT_B_starts_cycling_after_A_l517_51738

theorem B_starts_cycling_after_A (t : ℝ) : 10 * t + 20 * (2 - t) = 60 → t = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_B_starts_cycling_after_A_l517_51738


namespace NUMINAMATH_GPT_integer_solution_l517_51794

theorem integer_solution (a b : ℤ) (h : 6 * a * b = 9 * a - 10 * b + 303) : a + b = 15 :=
sorry

end NUMINAMATH_GPT_integer_solution_l517_51794


namespace NUMINAMATH_GPT_colored_shirts_count_l517_51710

theorem colored_shirts_count (n : ℕ) (h1 : 6 = 6) (h2 : (1 / (n : ℝ)) ^ 6 = 1 / 120) : n = 2 := 
sorry

end NUMINAMATH_GPT_colored_shirts_count_l517_51710


namespace NUMINAMATH_GPT_cyclist_speed_l517_51742

theorem cyclist_speed 
  (v : ℝ) 
  (hiker1_speed : ℝ := 4)
  (hiker2_speed : ℝ := 5)
  (cyclist_overtakes_hiker2_after_hiker1 : ∃ t1 t2 : ℝ, 
      t1 = 8 / (v - hiker1_speed) ∧ 
      t2 = 5 / (v - hiker2_speed) ∧ 
      t2 - t1 = 1/6)
: (v = 20 ∨ v = 7 ∨ abs (v - 6.5) < 0.1) :=
sorry

end NUMINAMATH_GPT_cyclist_speed_l517_51742


namespace NUMINAMATH_GPT_set_intersection_complement_eq_l517_51704

-- Definitions based on conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Complement of B in U
def complement_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- The theorem statement
theorem set_intersection_complement_eq :
  A ∩ complement_B = {x | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_eq_l517_51704


namespace NUMINAMATH_GPT_minimum_value_C2_minus_D2_l517_51711

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3)) + (Real.sqrt (y + 6)) + (Real.sqrt (z + 11))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2)) + (Real.sqrt (y + 4)) + (Real.sqrt (z + 9))

theorem minimum_value_C2_minus_D2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (C x y z)^2 - (D x y z)^2 ≥ 36 := by
  sorry

end NUMINAMATH_GPT_minimum_value_C2_minus_D2_l517_51711


namespace NUMINAMATH_GPT_percent_profit_is_25_percent_l517_51745

theorem percent_profit_is_25_percent
  (CP SP : ℝ)
  (h : 75 * (CP - 0.05 * CP) = 60 * SP) :
  let profit := SP - (0.95 * CP)
  let percent_profit := (profit / (0.95 * CP)) * 100
  percent_profit = 25 :=
by
  sorry

end NUMINAMATH_GPT_percent_profit_is_25_percent_l517_51745


namespace NUMINAMATH_GPT_tan_x_over_tan_y_plus_tan_y_over_tan_x_l517_51727

open Real

theorem tan_x_over_tan_y_plus_tan_y_over_tan_x (x y : ℝ) 
  (h1 : sin x / cos y + sin y / cos x = 2) 
  (h2 : cos x / sin y + cos y / sin x = 5) :
  tan x / tan y + tan y / tan x = 10 := 
by
  sorry

end NUMINAMATH_GPT_tan_x_over_tan_y_plus_tan_y_over_tan_x_l517_51727


namespace NUMINAMATH_GPT_find_positive_x_l517_51709

theorem find_positive_x (x y z : ℝ) 
  (h1 : x * y = 15 - 3 * x - 2 * y)
  (h2 : y * z = 8 - 2 * y - 4 * z)
  (h3 : x * z = 56 - 5 * x - 6 * z) : x = 8 := 
sorry

end NUMINAMATH_GPT_find_positive_x_l517_51709


namespace NUMINAMATH_GPT_series_sum_is_6_over_5_l517_51776

noncomputable def series_sum : ℝ := ∑' n : ℕ, if n % 4 == 0 then 1 / (4^(n/4)) else 
                                          if n % 4 == 1 then 1 / (2 * 4^(n/4)) else 
                                          if n % 4 == 2 then -1 / (4^(n/4) * 4^(1/2)) else 
                                          -1 / (2 * 4^(n/4 + 1/2))

theorem series_sum_is_6_over_5 : series_sum = 6 / 5 := 
  sorry

end NUMINAMATH_GPT_series_sum_is_6_over_5_l517_51776


namespace NUMINAMATH_GPT_sum_shade_length_l517_51700

-- Define the arithmetic sequence and the given conditions
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (is_arithmetic : ∀ n, a (n + 1) = a n + d)

-- Define the shadow lengths for each term using the arithmetic progression properties
def shade_length_seq (seq : ArithmeticSequence) : ℕ → ℝ := seq.a

variables (seq : ArithmeticSequence)

-- Given conditions
axiom sum_condition_1 : seq.a 1 + seq.a 4 + seq.a 7 = 31.5
axiom sum_condition_2 : seq.a 2 + seq.a 5 + seq.a 8 = 28.5

-- Question to prove
theorem sum_shade_length : seq.a 3 + seq.a 6 + seq.a 9 = 25.5 :=
by
  -- proof to be filled in later
  sorry

end NUMINAMATH_GPT_sum_shade_length_l517_51700


namespace NUMINAMATH_GPT_total_cost_to_plant_flowers_l517_51797

noncomputable def flower_cost : ℕ := 9
noncomputable def clay_pot_cost : ℕ := flower_cost + 20
noncomputable def soil_bag_cost : ℕ := flower_cost - 2
noncomputable def total_cost : ℕ := flower_cost + clay_pot_cost + soil_bag_cost

theorem total_cost_to_plant_flowers : total_cost = 45 := by
  sorry

end NUMINAMATH_GPT_total_cost_to_plant_flowers_l517_51797


namespace NUMINAMATH_GPT_triangle_segments_equivalence_l517_51748

variable {a b c p : ℝ}

theorem triangle_segments_equivalence (h_acute : a^2 + b^2 > c^2) 
  (h_perpendicular : ∃ h: ℝ, h^2 = c^2 - (a - p)^2 ∧ h^2 = b^2 - p^2) :
  a / (c + b) = (c - b) / (a - 2 * p) := by
sorry

end NUMINAMATH_GPT_triangle_segments_equivalence_l517_51748


namespace NUMINAMATH_GPT_diagonals_in_nine_sided_polygon_l517_51760

-- Define the conditions
def sides : ℕ := 9
def right_angles : ℕ := 2

-- The function to calculate the number of diagonals for a polygon
def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- The theorem to prove
theorem diagonals_in_nine_sided_polygon : number_of_diagonals sides = 27 := by
  sorry

end NUMINAMATH_GPT_diagonals_in_nine_sided_polygon_l517_51760


namespace NUMINAMATH_GPT_unique_even_odd_decomposition_l517_51769

def is_symmetric (s : Set ℝ) : Prop := ∀ x ∈ s, -x ∈ s

def is_even (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ x ∈ s, f (-x) = f x

def is_odd (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ x ∈ s, f (-x) = -f x

theorem unique_even_odd_decomposition (s : Set ℝ) (hs : is_symmetric s) (f : ℝ → ℝ) (hf : ∀ x ∈ s, True) :
  ∃! g h : ℝ → ℝ, (is_even g s) ∧ (is_odd h s) ∧ (∀ x ∈ s, f x = g x + h x) :=
sorry

end NUMINAMATH_GPT_unique_even_odd_decomposition_l517_51769


namespace NUMINAMATH_GPT_third_median_length_l517_51717

-- Proposition stating the problem with conditions and the conclusion
theorem third_median_length (m1 m2 : ℝ) (area : ℝ) (h1 : m1 = 4) (h2 : m2 = 5) (h_area : area = 10 * Real.sqrt 3) : 
  ∃ m3 : ℝ, m3 = 3 * Real.sqrt 10 :=
by
  sorry  -- proof is not included

end NUMINAMATH_GPT_third_median_length_l517_51717


namespace NUMINAMATH_GPT_students_failed_exam_l517_51726

def total_students : ℕ := 740
def percent_passed : ℝ := 0.35
def percent_failed : ℝ := 1 - percent_passed
def failed_students : ℝ := percent_failed * total_students

theorem students_failed_exam : failed_students = 481 := 
by sorry

end NUMINAMATH_GPT_students_failed_exam_l517_51726


namespace NUMINAMATH_GPT_large_pizza_cost_l517_51758

theorem large_pizza_cost
  (small_side : ℕ) (small_cost : ℝ) (large_side : ℕ) (friend_money : ℝ) (extra_square_inches : ℝ)
  (A_small : small_side * small_side = 196)
  (A_large : large_side * large_side = 441)
  (small_cost_per_sq_in : 196 / small_cost = 19.6)
  (individual_area : (30 / small_cost) * 196 = 588)
  (total_individual_area : 2 * 588 = 1176)
  (pool_area_eq : (60 / (441 / x)) = 1225)
  : (x = 21.6) := 
by
  sorry

end NUMINAMATH_GPT_large_pizza_cost_l517_51758


namespace NUMINAMATH_GPT_f_expr_for_nonneg_l517_51799

-- Define the function f piecewise as per the given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then
    Real.exp (-x) + 2 * x - 1
  else
    -Real.exp x + 2 * x + 1

-- Prove that for x > 0, f(x) = -e^x + 2x + 1 given the conditions
theorem f_expr_for_nonneg (x : ℝ) (h : x ≥ 0) : f x = -Real.exp x + 2 * x + 1 := by
  sorry

end NUMINAMATH_GPT_f_expr_for_nonneg_l517_51799


namespace NUMINAMATH_GPT_cab_speed_fraction_l517_51754

def usual_time := 30 -- The usual time of the journey in minutes
def delay_time := 6   -- The delay time in minutes
def usual_speed : ℝ := sorry -- Placeholder for the usual speed
def reduced_speed : ℝ := sorry -- Placeholder for the reduced speed

-- Given the conditions:
-- 1. The usual time for the cab to cover the journey is 30 minutes.
-- 2. The cab is 6 minutes late when walking at a reduced speed.
-- Prove that the fraction of the cab's usual speed it is walking at is 5/6

theorem cab_speed_fraction : (reduced_speed / usual_speed) = (5 / 6) :=
sorry

end NUMINAMATH_GPT_cab_speed_fraction_l517_51754


namespace NUMINAMATH_GPT_find_n_l517_51736

theorem find_n (n : ℕ) (h : n * Nat.factorial n + Nat.factorial n = 5040) : n = 6 :=
sorry

end NUMINAMATH_GPT_find_n_l517_51736


namespace NUMINAMATH_GPT_shape_of_triangle_l517_51718

-- Define the problem conditions
variable {a b : ℝ}
variable {A B C : ℝ}
variable (triangle_condition : (a^2 / b^2 = tan A / tan B))

-- Define the theorem to be proved
theorem shape_of_triangle ABC
  (h : triangle_condition):
  (A = B ∨ A + B = π / 2) :=
sorry

end NUMINAMATH_GPT_shape_of_triangle_l517_51718


namespace NUMINAMATH_GPT_correct_growth_equation_l517_51792

-- Define the parameters
def initial_income : ℝ := 2.36
def final_income : ℝ := 2.7
def growth_period : ℕ := 2

-- Define the growth rate x
variable (x : ℝ)

-- The theorem we want to prove
theorem correct_growth_equation : initial_income * (1 + x)^growth_period = final_income :=
sorry

end NUMINAMATH_GPT_correct_growth_equation_l517_51792


namespace NUMINAMATH_GPT_marion_score_is_correct_l517_51746

-- Definition of the problem conditions
def exam_total_items := 40
def ella_incorrect_answers := 4

-- Calculate Ella's score
def ella_score := exam_total_items - ella_incorrect_answers

-- Calculate half of Ella's score
def half_ella_score := ella_score / 2

-- Marion's score is 6 more than half of Ella's score
def marion_score := half_ella_score + 6

-- The theorem we need to prove
theorem marion_score_is_correct : marion_score = 24 := by
  sorry

end NUMINAMATH_GPT_marion_score_is_correct_l517_51746


namespace NUMINAMATH_GPT_smallest_prime_divisor_of_sum_l517_51721

theorem smallest_prime_divisor_of_sum (a b : ℕ) (h1 : a = 3^19) (h2 : b = 11^13) (h3 : Odd a) (h4 : Odd b) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (a + b) ∧ p = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_prime_divisor_of_sum_l517_51721


namespace NUMINAMATH_GPT_total_cost_of_pens_and_notebooks_l517_51772

theorem total_cost_of_pens_and_notebooks (a b : ℝ) : 5 * a + 8 * b = 5 * a + 8 * b := 
by 
  sorry

end NUMINAMATH_GPT_total_cost_of_pens_and_notebooks_l517_51772


namespace NUMINAMATH_GPT_Martin_correct_answers_l517_51749

theorem Martin_correct_answers (C K M : ℕ) 
  (h1 : C = 35)
  (h2 : K = C + 8)
  (h3 : M = K - 3) : 
  M = 40 :=
by
  sorry

end NUMINAMATH_GPT_Martin_correct_answers_l517_51749


namespace NUMINAMATH_GPT_set_intersection_l517_51777

def A := {x : ℝ | -5 < x ∧ x < 2}
def B := {x : ℝ | |x| < 3}

theorem set_intersection : {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | -3 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_l517_51777


namespace NUMINAMATH_GPT_geometric_progression_fifth_term_sum_l517_51771

def gp_sum_fifth_term
    (p q : ℝ)
    (hpq_sum : p + q = 3)
    (hpq_6th : p^5 + q^5 = 573) : ℝ :=
p^4 + q^4

theorem geometric_progression_fifth_term_sum :
    ∃ p q : ℝ, p + q = 3 ∧ p^5 + q^5 = 573 ∧ gp_sum_fifth_term p q (by sorry) (by sorry) = 161 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_fifth_term_sum_l517_51771


namespace NUMINAMATH_GPT_parabola_hyperbola_tangent_l517_51743

noncomputable def parabola (x : ℝ) : ℝ := x^2 + 5
noncomputable def hyperbola (x y : ℝ) (m : ℝ) : ℝ := y^2 - m * x^2 - 1

theorem parabola_hyperbola_tangent (m : ℝ) :
(∃ x y : ℝ, y = parabola x ∧ hyperbola x y m = 0) ↔ 
m = 10 + 2 * Real.sqrt 6 ∨ m = 10 - 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_tangent_l517_51743


namespace NUMINAMATH_GPT_dogs_neither_long_furred_nor_brown_l517_51765

theorem dogs_neither_long_furred_nor_brown :
  (∀ (total_dogs long_furred_dogs brown_dogs both_long_furred_and_brown neither_long_furred_nor_brown : ℕ),
     total_dogs = 45 →
     long_furred_dogs = 26 →
     brown_dogs = 22 →
     both_long_furred_and_brown = 11 →
     neither_long_furred_nor_brown = total_dogs - (long_furred_dogs + brown_dogs - both_long_furred_and_brown) → 
     neither_long_furred_nor_brown = 8) :=
by
  intros total_dogs long_furred_dogs brown_dogs both_long_furred_and_brown neither_long_furred_nor_brown
  sorry

end NUMINAMATH_GPT_dogs_neither_long_furred_nor_brown_l517_51765


namespace NUMINAMATH_GPT_singer_arrangements_l517_51763

-- Let's assume the 5 singers are represented by the indices 1 through 5

theorem singer_arrangements :
  ∀ (singers : List ℕ) (no_first : ℕ) (must_last : ℕ), 
  singers = [1, 2, 3, 4, 5] →
  no_first ∈ singers →
  must_last ∈ singers →
  no_first ≠ must_last →
  ∃ (arrangements : ℕ),
    arrangements = 18 :=
by
  sorry

end NUMINAMATH_GPT_singer_arrangements_l517_51763


namespace NUMINAMATH_GPT_alice_walking_speed_l517_51783

theorem alice_walking_speed:
  ∃ v : ℝ, 
  (∀ t : ℝ, t = 1 → ∀ d_a d_b : ℝ, d_a = 25 → d_b = 41 - d_a → 
  ∀ s_b : ℝ, s_b = 4 → 
  d_b / s_b + t = d_a / v) ∧ v = 5 :=
by
  sorry

end NUMINAMATH_GPT_alice_walking_speed_l517_51783


namespace NUMINAMATH_GPT_angles_between_plane_and_catheti_l517_51775

theorem angles_between_plane_and_catheti
  (α β : ℝ)
  (h_alpha : 0 < α ∧ α < π / 2)
  (h_beta : 0 < β ∧ β < π / 2) :
  ∃ γ θ : ℝ,
    γ = Real.arcsin (Real.sin β * Real.cos α) ∧
    θ = Real.arcsin (Real.sin β * Real.sin α) :=
by
  sorry

end NUMINAMATH_GPT_angles_between_plane_and_catheti_l517_51775


namespace NUMINAMATH_GPT_general_formula_a_n_sum_first_n_terms_T_n_l517_51708

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {T_n : ℕ → ℕ}

-- Condition: S_n = 2a_n - 3
axiom condition_S (n : ℕ) : S_n n = 2 * (a_n n) - 3

-- (I) General formula for a_n
theorem general_formula_a_n (n : ℕ) : a_n n = 3 * 2^(n - 1) := 
sorry

-- (II) General formula for T_n
theorem sum_first_n_terms_T_n (n : ℕ) : T_n n = 3 * (n - 1) * 2^n + 3 := 
sorry

end NUMINAMATH_GPT_general_formula_a_n_sum_first_n_terms_T_n_l517_51708


namespace NUMINAMATH_GPT_units_digit_of_product_of_first_four_composites_l517_51768

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_product_of_first_four_composites_l517_51768


namespace NUMINAMATH_GPT_max_non_intersecting_segments_l517_51701

theorem max_non_intersecting_segments (n m : ℕ) (hn: 1 < n) (hm: m ≥ 3): 
  ∃ L, L = 3 * n - m - 3 :=
by
  sorry

end NUMINAMATH_GPT_max_non_intersecting_segments_l517_51701


namespace NUMINAMATH_GPT_eleven_step_paths_l517_51781

def H : (ℕ × ℕ) := (0, 0)
def K : (ℕ × ℕ) := (4, 3)
def J : (ℕ × ℕ) := (6, 5)

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem eleven_step_paths (H K J : (ℕ × ℕ)) (H_coords : H = (0, 0)) (K_coords : K = (4, 3)) (J_coords : J = (6, 5)) : 
  (binomial 7 4) * (binomial 4 2) = 210 := by 
  sorry

end NUMINAMATH_GPT_eleven_step_paths_l517_51781


namespace NUMINAMATH_GPT_T_n_formula_l517_51747

def a_n (n : ℕ) : ℕ := 3 * n - 1
def b_n (n : ℕ) : ℕ := 2 ^ n
def T_n (n : ℕ) : ℕ := (Finset.range n).sum (λ k => a_n (k + 1) * b_n (k + 1))

theorem T_n_formula (n : ℕ) : T_n n = 8 - 8 * 2 ^ n + 3 * n * 2 ^ (n + 1) :=
by 
  sorry

end NUMINAMATH_GPT_T_n_formula_l517_51747


namespace NUMINAMATH_GPT_line_inclination_angle_l517_51723

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + (Real.sqrt 3) * y - 1 = 0

-- Define the condition of inclination angle in radians
def inclination_angle (θ : ℝ) : Prop := θ = Real.arctan (-1 / Real.sqrt 3) + Real.pi

-- The theorem to prove the inclination angle of the line
theorem line_inclination_angle (x y θ : ℝ) (h : line_eq x y) : inclination_angle θ :=
by
  sorry

end NUMINAMATH_GPT_line_inclination_angle_l517_51723


namespace NUMINAMATH_GPT_total_length_of_sticks_l517_51739

-- Definitions based on conditions
def num_sticks := 30
def length_per_stick := 25
def overlap := 6
def effective_length_per_stick := length_per_stick - overlap

-- Theorem statement
theorem total_length_of_sticks : num_sticks * effective_length_per_stick - effective_length_per_stick + length_per_stick = 576 := sorry

end NUMINAMATH_GPT_total_length_of_sticks_l517_51739


namespace NUMINAMATH_GPT_mangoes_in_shop_l517_51793

-- Define the conditions
def ratio_mango_to_apple := 10 / 3
def apples := 36

-- Problem statement to prove
theorem mangoes_in_shop : ∃ (m : ℕ), m = 120 ∧ m = apples * ratio_mango_to_apple :=
by
  sorry

end NUMINAMATH_GPT_mangoes_in_shop_l517_51793


namespace NUMINAMATH_GPT_emily_seeds_start_with_l517_51741

-- Define the conditions as hypotheses
variables (big_garden_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ)

-- Conditions: Emily planted 29 seeds in the big garden and 4 seeds in each of her 3 small gardens.
def emily_conditions := big_garden_seeds = 29 ∧ small_gardens = 3 ∧ seeds_per_small_garden = 4

-- Define the statement to prove the total number of seeds Emily started with
theorem emily_seeds_start_with (h : emily_conditions big_garden_seeds small_gardens seeds_per_small_garden) : 
(big_garden_seeds + small_gardens * seeds_per_small_garden) = 41 :=
by
  -- Assuming the proof follows logically from conditions
  sorry

end NUMINAMATH_GPT_emily_seeds_start_with_l517_51741


namespace NUMINAMATH_GPT_slope_of_line_through_points_l517_51705

theorem slope_of_line_through_points 
  (t : ℝ) 
  (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 12 * t + 6) 
  (h2 : 2 * x + 3 * y = 8 * t - 1) : 
  ∃ m b : ℝ, (∀ t : ℝ, y = m * x + b) ∧ m = 0 :=
by 
  sorry

end NUMINAMATH_GPT_slope_of_line_through_points_l517_51705


namespace NUMINAMATH_GPT_points_on_intersecting_lines_l517_51707

def clubsuit (a b : ℝ) := a^3 * b - a * b^3

theorem points_on_intersecting_lines (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = y ∨ x = -y) := 
by
  sorry

end NUMINAMATH_GPT_points_on_intersecting_lines_l517_51707


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l517_51725

-- Define the repeating decimal 4.25252525... as x
def repeating_decimal : ℚ := 4 + 25 / 99

-- Theorem statement to prove the equivalence
theorem repeating_decimal_as_fraction :
  repeating_decimal = 421 / 99 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l517_51725


namespace NUMINAMATH_GPT_prime_factorization_sum_l517_51785

theorem prime_factorization_sum (w x y z k : ℕ) (h : 2^w * 3^x * 5^y * 7^z * 11^k = 2310) :
  2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 28 :=
sorry

end NUMINAMATH_GPT_prime_factorization_sum_l517_51785


namespace NUMINAMATH_GPT_distance_between_A_and_B_l517_51720

def scale : ℕ := 20000
def map_distance : ℕ := 6
def actual_distance_cm : ℕ := scale * map_distance
def actual_distance_m : ℕ := actual_distance_cm / 100

theorem distance_between_A_and_B : actual_distance_m = 1200 := by
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l517_51720


namespace NUMINAMATH_GPT_distance_between_towns_l517_51728

theorem distance_between_towns (D S : ℝ) (h1 : D = S * 3) (h2 : 200 = S * 5) : D = 120 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_towns_l517_51728


namespace NUMINAMATH_GPT_C_share_of_profit_l517_51752

variable (A B C P Rs_36000 k : ℝ)

-- Definitions as per the conditions given in the problem statement.
def investment_A := 24000
def investment_B := 32000
def investment_C := 36000
def total_profit := 92000
def C_Share := 36000

-- The Lean statement without the proof as requested.
theorem C_share_of_profit 
  (h_A : investment_A = 24000)
  (h_B : investment_B = 32000)
  (h_C : investment_C = 36000)
  (h_P : total_profit = 92000)
  (h_C_share : C_Share = 36000)
  : C_Share = (investment_C / k) / ((investment_A / k) + (investment_B / k) + (investment_C / k)) * total_profit := 
sorry

end NUMINAMATH_GPT_C_share_of_profit_l517_51752


namespace NUMINAMATH_GPT_regular_polygon_sides_l517_51724

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i < n → (180 * (n - 2) / n) = 174) : n = 60 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l517_51724


namespace NUMINAMATH_GPT_power_function_is_odd_l517_51759

open Function

noncomputable def power_function (a : ℝ) (b : ℝ) : ℝ → ℝ := λ x => (a - 1) * x^b

theorem power_function_is_odd (a b : ℝ) (h : power_function a b a = 1 / 8)
  :  a = 2 ∧ b = -3 → (∀ x : ℝ, power_function a b (-x) = -power_function a b x) :=
by
  intro ha hb
  -- proofs can be filled later with details
  sorry

end NUMINAMATH_GPT_power_function_is_odd_l517_51759


namespace NUMINAMATH_GPT_lucas_150_mod_9_l517_51732

-- Define the Lucas sequence recursively
def lucas (n : ℕ) : ℕ :=
  match n with
  | 0 => 1 -- Since L_1 in the sequence provided is actually the first Lucas number (index starts from 1)
  | 1 => 3
  | (n + 2) => lucas n + lucas (n + 1)

-- Define the theorem for the remainder when the 150th term is divided by 9
theorem lucas_150_mod_9 : lucas 149 % 9 = 3 := by
  sorry

end NUMINAMATH_GPT_lucas_150_mod_9_l517_51732


namespace NUMINAMATH_GPT_symmetric_points_l517_51786

theorem symmetric_points (a b : ℝ) (h1 : 2 * a + 1 = -1) (h2 : 4 = -(3 * b - 1)) :
  2 * a + b = -3 := 
sorry

end NUMINAMATH_GPT_symmetric_points_l517_51786


namespace NUMINAMATH_GPT_problem_l517_51751

def p (x y : Int) : Int :=
  if x ≥ 0 ∧ y ≥ 0 then x * y
  else if x < 0 ∧ y < 0 then x - 2 * y
  else if x ≥ 0 ∧ y < 0 then 2 * x + 3 * y
  else if x < 0 ∧ y ≥ 0 then x + 3 * y
  else 3 * x + y

theorem problem : p (p 2 (-3)) (p (-1) 4) = 28 := by
  sorry

end NUMINAMATH_GPT_problem_l517_51751


namespace NUMINAMATH_GPT_first_group_hours_per_day_l517_51790

theorem first_group_hours_per_day :
  ∃ H : ℕ, 
    (39 * 12 * H = 30 * 26 * 3) ∧
    H = 5 :=
by sorry

end NUMINAMATH_GPT_first_group_hours_per_day_l517_51790


namespace NUMINAMATH_GPT_second_machine_time_l517_51750

/-- Given:
1. A first machine can address 600 envelopes in 10 minutes.
2. Both machines together can address 600 envelopes in 4 minutes.
We aim to prove that the second machine alone would take 20/3 minutes to address 600 envelopes. -/
theorem second_machine_time (x : ℝ) 
  (first_machine_rate : ℝ := 600 / 10)
  (combined_rate_needed : ℝ := 600 / 4)
  (second_machine_rate : ℝ := combined_rate_needed - first_machine_rate) 
  (secs_envelope_rate : ℝ := second_machine_rate) 
  (envelopes : ℝ := 600) : 
  x = envelopes / secs_envelope_rate :=
sorry

end NUMINAMATH_GPT_second_machine_time_l517_51750


namespace NUMINAMATH_GPT_hamburgers_made_l517_51789

theorem hamburgers_made (initial_hamburgers additional_hamburgers total_hamburgers : ℝ)
    (h_initial : initial_hamburgers = 9.0)
    (h_additional : additional_hamburgers = 3.0)
    (h_total : total_hamburgers = initial_hamburgers + additional_hamburgers) :
    total_hamburgers = 12.0 :=
by
    sorry

end NUMINAMATH_GPT_hamburgers_made_l517_51789


namespace NUMINAMATH_GPT_problem_statement_l517_51770

-- Given conditions
variables {p q r t n : ℕ}

axiom prime_p : Nat.Prime p
axiom prime_q : Nat.Prime q
axiom prime_r : Nat.Prime r

axiom nat_n : n ≥ 1
axiom nat_t : t ≥ 1

axiom eqn1 : p^2 + q * t = (p + t)^n
axiom eqn2 : p^2 + q * r = t^4

-- Statement to prove
theorem problem_statement : n < 3 ∧ (p = 2 ∧ q = 7 ∧ r = 11 ∧ t = 3 ∧ n = 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l517_51770


namespace NUMINAMATH_GPT_celine_smartphones_l517_51706

-- Definitions based on the conditions
def laptop_cost : ℕ := 600
def smartphone_cost : ℕ := 400
def num_laptops_bought : ℕ := 2
def initial_amount : ℕ := 3000
def change_received : ℕ := 200

-- The proof goal is to show that the number of smartphones bought is 4
theorem celine_smartphones (laptop_cost smartphone_cost num_laptops_bought initial_amount change_received : ℕ)
  (h1 : laptop_cost = 600)
  (h2 : smartphone_cost = 400)
  (h3 : num_laptops_bought = 2)
  (h4 : initial_amount = 3000)
  (h5 : change_received = 200) :
  (initial_amount - change_received - num_laptops_bought * laptop_cost) / smartphone_cost = 4 := 
by
  sorry

end NUMINAMATH_GPT_celine_smartphones_l517_51706


namespace NUMINAMATH_GPT_water_pressure_on_dam_l517_51757

theorem water_pressure_on_dam :
  let a := 10 -- length of upper base in meters
  let b := 20 -- length of lower base in meters
  let h := 3 -- height in meters
  let ρg := 9810 -- natural constant for water pressure in N/m^3
  let P := ρg * ((a + 2 * b) * h^2 / 6)
  P = 735750 :=
by
  sorry

end NUMINAMATH_GPT_water_pressure_on_dam_l517_51757


namespace NUMINAMATH_GPT_lisa_savings_l517_51774

-- Define the conditions
def originalPricePerNotebook : ℝ := 3
def numberOfNotebooks : ℕ := 8
def discountRate : ℝ := 0.30
def additionalDiscount : ℝ := 5

-- Define the total savings calculation
def calculateSavings (originalPricePerNotebook : ℝ) (numberOfNotebooks : ℕ) (discountRate : ℝ) (additionalDiscount : ℝ) : ℝ := 
  let totalPriceWithoutDiscount := originalPricePerNotebook * numberOfNotebooks
  let discountedPricePerNotebook := originalPricePerNotebook * (1 - discountRate)
  let totalPriceWith30PercentDiscount := discountedPricePerNotebook * numberOfNotebooks
  let totalPriceWithAllDiscounts := totalPriceWith30PercentDiscount - additionalDiscount
  totalPriceWithoutDiscount - totalPriceWithAllDiscounts

-- Theorem for the proof problem
theorem lisa_savings :
  calculateSavings originalPricePerNotebook numberOfNotebooks discountRate additionalDiscount = 12.20 :=
by
  -- Inserting the proof as sorry
  sorry

end NUMINAMATH_GPT_lisa_savings_l517_51774


namespace NUMINAMATH_GPT_wrapping_paper_area_correct_l517_51714

structure Box :=
  (l : ℝ)  -- length of the box
  (w : ℝ)  -- width of the box
  (h : ℝ)  -- height of the box
  (h_lw : l > w)  -- condition that length is greater than width

def wrapping_paper_area (b : Box) : ℝ :=
  3 * (b.l + b.w) * b.h

theorem wrapping_paper_area_correct (b : Box) : 
  wrapping_paper_area b = 3 * (b.l + b.w) * b.h :=
sorry

end NUMINAMATH_GPT_wrapping_paper_area_correct_l517_51714


namespace NUMINAMATH_GPT_parking_lot_problem_l517_51756

theorem parking_lot_problem :
  let total_spaces := 50
  let cars := 2
  let total_ways := total_spaces * (total_spaces - 1)
  let adjacent_ways := (total_spaces - 1) * 2
  let valid_ways := total_ways - adjacent_ways
  valid_ways = 2352 :=
by
  sorry

end NUMINAMATH_GPT_parking_lot_problem_l517_51756


namespace NUMINAMATH_GPT_sets_equal_l517_51737

def A : Set ℝ := {1, Real.sqrt 3, Real.pi}
def B : Set ℝ := {Real.pi, 1, abs (-(Real.sqrt 3))}

theorem sets_equal : A = B :=
by 
  sorry

end NUMINAMATH_GPT_sets_equal_l517_51737


namespace NUMINAMATH_GPT_complement_of_A_in_U_l517_51713

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def complement_U_A : Set ℝ := {x | x ≤ 1 ∨ x > 3}

theorem complement_of_A_in_U : (U \ A) = complement_U_A := by
  simp only [U, A, complement_U_A]
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l517_51713


namespace NUMINAMATH_GPT_solve_polynomial_equation_l517_51729

theorem solve_polynomial_equation :
  ∃ z, (z^5 + 40 * z^3 + 80 * z - 32 = 0) →
  ∃ x, (x = z + 4) ∧ ((x - 2)^5 + (x - 6)^5 = 32) :=
by
  sorry

end NUMINAMATH_GPT_solve_polynomial_equation_l517_51729


namespace NUMINAMATH_GPT_spring_length_at_9kg_l517_51773

theorem spring_length_at_9kg :
  (∃ (k b : ℝ), (∀ x : ℝ, y = k * x + b) ∧ 
                 (y = 10 ∧ x = 0) ∧ 
                 (y = 10.5 ∧ x = 1)) → 
  (∀ x : ℝ, x = 9 → y = 14.5) :=
sorry

end NUMINAMATH_GPT_spring_length_at_9kg_l517_51773


namespace NUMINAMATH_GPT_first_group_person_count_l517_51761

theorem first_group_person_count
  (P : ℕ)
  (h1 : P * 24 * 5 = 30 * 26 * 6) : 
  P = 39 :=
by
  sorry

end NUMINAMATH_GPT_first_group_person_count_l517_51761


namespace NUMINAMATH_GPT_fermats_little_theorem_l517_51733

theorem fermats_little_theorem (p : ℕ) (a : ℤ) (hp : Nat.Prime p) : 
  (a^p - a) % p = 0 :=
sorry

end NUMINAMATH_GPT_fermats_little_theorem_l517_51733


namespace NUMINAMATH_GPT_triangles_in_divided_square_l517_51702

theorem triangles_in_divided_square (V E F : ℕ) 
  (hV : V = 24) 
  (h1 : 3 * F + 1 = 2 * E) 
  (h2 : V - E + F = 2) : F = 43 ∧ (F - 1 = 42) := 
by 
  have hF : F = 43 := sorry
  have hTriangles : F - 1 = 42 := sorry
  exact ⟨hF, hTriangles⟩

end NUMINAMATH_GPT_triangles_in_divided_square_l517_51702


namespace NUMINAMATH_GPT_sum_between_100_and_500_ending_in_3_l517_51703

-- Definition for the sum of all integers between 100 and 500 that end in 3
def sumOfIntegersBetween100And500EndingIn3 : ℕ :=
  let a := 103
  let d := 10
  let n := (493 - a) / d + 1
  (n * (a + 493)) / 2

-- Statement to prove that the sum is 11920
theorem sum_between_100_and_500_ending_in_3 : sumOfIntegersBetween100And500EndingIn3 = 11920 := by
  sorry

end NUMINAMATH_GPT_sum_between_100_and_500_ending_in_3_l517_51703


namespace NUMINAMATH_GPT_irwin_basketball_l517_51735

theorem irwin_basketball (A B C D : ℕ) (h1 : C = 2) (h2 : 2^A * 5^B * 11^C * 13^D = 2420) : A = 2 :=
by
  sorry

end NUMINAMATH_GPT_irwin_basketball_l517_51735


namespace NUMINAMATH_GPT_company_salary_decrease_l517_51764

variables {E S : ℝ} -- Let the initial number of employees be E and the initial average salary be S

theorem company_salary_decrease :
  (0.8 * E * (1.15 * S)) / (E * S) = 0.92 := 
by
  -- The proof will go here, but we use sorry to skip it for now
  sorry

end NUMINAMATH_GPT_company_salary_decrease_l517_51764


namespace NUMINAMATH_GPT_sequence_sum_a1_a3_l517_51780

theorem sequence_sum_a1_a3 (S : ℕ → ℕ) (a : ℕ → ℤ) 
  (h1 : ∀ n, n ≥ 2 → S n + S (n - 1) = 2 * n - 1) 
  (h2 : S 2 = 3) : 
  a 1 + a 3 = -1 := by
  sorry

end NUMINAMATH_GPT_sequence_sum_a1_a3_l517_51780


namespace NUMINAMATH_GPT_black_spools_l517_51767

-- Define the given conditions
def spools_per_beret : ℕ := 3
def red_spools : ℕ := 12
def blue_spools : ℕ := 6
def berets_made : ℕ := 11

-- Define the statement to be proved using the defined conditions
theorem black_spools (spools_per_beret red_spools blue_spools berets_made : ℕ) : (spools_per_beret * berets_made) - (red_spools + blue_spools) = 15 :=
by sorry

end NUMINAMATH_GPT_black_spools_l517_51767


namespace NUMINAMATH_GPT_ratio_distance_l517_51715

-- Definitions based on conditions
def speed_ferry_P : ℕ := 6 -- speed of ferry P in km/h
def time_ferry_P : ℕ := 3 -- travel time of ferry P in hours
def speed_ferry_Q : ℕ := speed_ferry_P + 3 -- speed of ferry Q in km/h
def time_ferry_Q : ℕ := time_ferry_P + 1 -- travel time of ferry Q in hours

-- Calculating the distances
def distance_ferry_P : ℕ := speed_ferry_P * time_ferry_P -- distance covered by ferry P
def distance_ferry_Q : ℕ := speed_ferry_Q * time_ferry_Q -- distance covered by ferry Q

-- Main theorem to prove
theorem ratio_distance (d_P d_Q : ℕ) (h_dP : d_P = distance_ferry_P) (h_dQ : d_Q = distance_ferry_Q) : d_Q / d_P = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_distance_l517_51715


namespace NUMINAMATH_GPT_problem_statement_l517_51798

theorem problem_statement (x y : ℝ) (h₁ : x + y = 5) (h₂ : x * y = 3) : 
  x + (x^2 / y) + (y^2 / x) + y = 95 / 3 := 
sorry

end NUMINAMATH_GPT_problem_statement_l517_51798


namespace NUMINAMATH_GPT_initial_cookies_count_l517_51788

def cookies_left : ℕ := 9
def cookies_eaten : ℕ := 9

theorem initial_cookies_count : cookies_left + cookies_eaten = 18 :=
by sorry

end NUMINAMATH_GPT_initial_cookies_count_l517_51788
