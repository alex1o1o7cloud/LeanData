import Mathlib

namespace NUMINAMATH_GPT_other_asymptote_l1684_168491

/-- Problem Statement:
One of the asymptotes of a hyperbola is y = 2x. The foci have the same 
x-coordinate, which is 4. Prove that the equation of the other asymptote
of the hyperbola is y = -2x + 16.
-/
theorem other_asymptote (focus_x : ℝ) (asymptote1: ℝ → ℝ) (asymptote2 : ℝ → ℝ) :
  focus_x = 4 →
  (∀ x, asymptote1 x = 2 * x) →
  (asymptote2 4 = 8) → 
  (∀ x, asymptote2 x = -2 * x + 16) :=
sorry

end NUMINAMATH_GPT_other_asymptote_l1684_168491


namespace NUMINAMATH_GPT_six_points_within_circle_l1684_168402

/-- If six points are placed inside or on a circle with radius 1, then 
there always exist at least two points such that the distance between 
them is at most 1. -/
theorem six_points_within_circle : ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (points i).1^2 + (points i).2^2 ≤ 1) → 
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ 1 :=
by
  -- Condition: Circle of radius 1
  intro points h_points
  sorry

end NUMINAMATH_GPT_six_points_within_circle_l1684_168402


namespace NUMINAMATH_GPT_no_sum_of_19_l1684_168428

theorem no_sum_of_19 (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6)
  (hprod : a * b * c * d = 180) : a + b + c + d ≠ 19 :=
sorry

end NUMINAMATH_GPT_no_sum_of_19_l1684_168428


namespace NUMINAMATH_GPT_nth_term_arithmetic_sequence_l1684_168475

variable (n r : ℕ)

def S (n : ℕ) : ℕ := 4 * n + 5 * n^2

theorem nth_term_arithmetic_sequence :
  (S r) - (S (r-1)) = 10 * r - 1 :=
by
  sorry

end NUMINAMATH_GPT_nth_term_arithmetic_sequence_l1684_168475


namespace NUMINAMATH_GPT_remove_five_yields_average_10_5_l1684_168424

def numberList : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def averageRemaining (l : List ℕ) : ℚ :=
  (List.sum l : ℚ) / l.length

theorem remove_five_yields_average_10_5 :
  averageRemaining (numberList.erase 5) = 10.5 :=
sorry

end NUMINAMATH_GPT_remove_five_yields_average_10_5_l1684_168424


namespace NUMINAMATH_GPT_solve_for_x_l1684_168432

theorem solve_for_x (x : ℝ) (h : (x / 6) / 3 = 9 / (x / 3)) : x = 9 * Real.sqrt 6 ∨ x = - (9 * Real.sqrt 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1684_168432


namespace NUMINAMATH_GPT_central_angle_of_sector_in_unit_circle_with_area_1_is_2_l1684_168443

theorem central_angle_of_sector_in_unit_circle_with_area_1_is_2 :
  ∀ (θ : ℝ), (∀ (r : ℝ), (r = 1) → (1 / 2 * r^2 * θ = 1) → θ = 2) :=
by
  intros θ r hr h
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_in_unit_circle_with_area_1_is_2_l1684_168443


namespace NUMINAMATH_GPT_at_least_one_of_p_or_q_true_l1684_168469

variable (p q : Prop)

theorem at_least_one_of_p_or_q_true (h : ¬(p ∨ q) = false) : p ∨ q :=
by 
  sorry

end NUMINAMATH_GPT_at_least_one_of_p_or_q_true_l1684_168469


namespace NUMINAMATH_GPT_relationship_among_abc_l1684_168492

noncomputable def a := Real.sqrt 5 + 2
noncomputable def b := 2 - Real.sqrt 5
noncomputable def c := Real.sqrt 5 - 2

theorem relationship_among_abc : a > c ∧ c > b :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l1684_168492


namespace NUMINAMATH_GPT_age_of_B_l1684_168486

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 11) : B = 41 :=
by
  -- Proof not required as per instructions
  sorry

end NUMINAMATH_GPT_age_of_B_l1684_168486


namespace NUMINAMATH_GPT_inscribed_sphere_radius_eq_l1684_168448

noncomputable def inscribed_sphere_radius (b α : ℝ) : ℝ :=
  b * (Real.sin α) / (4 * (Real.cos (α / 4))^2)

theorem inscribed_sphere_radius_eq
  (b α : ℝ) 
  (h1 : 0 < b)
  (h2 : 0 < α ∧ α < Real.pi) 
  : inscribed_sphere_radius b α = b * (Real.sin α) / (4 * (Real.cos (α / 4))^2) :=
sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_eq_l1684_168448


namespace NUMINAMATH_GPT_arithmetic_sqrt_sqrt_16_eq_2_l1684_168476

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sqrt_sqrt_16_eq_2_l1684_168476


namespace NUMINAMATH_GPT_corrected_mean_l1684_168499

theorem corrected_mean (mean_incorrect : ℝ) (number_of_observations : ℕ) (wrong_observation correct_observation : ℝ) : 
  mean_incorrect = 36 → 
  number_of_observations = 50 → 
  wrong_observation = 23 → 
  correct_observation = 43 → 
  (mean_incorrect * number_of_observations + (correct_observation - wrong_observation)) / number_of_observations = 36.4 :=
by
  intros h_mean_incorrect h_number_of_observations h_wrong_observation h_correct_observation
  have S_incorrect : ℝ := mean_incorrect * number_of_observations
  have difference : ℝ := correct_observation - wrong_observation
  have S_correct : ℝ := S_incorrect + difference
  have mean_correct : ℝ := S_correct / number_of_observations
  sorry

end NUMINAMATH_GPT_corrected_mean_l1684_168499


namespace NUMINAMATH_GPT_karlson_max_eat_chocolates_l1684_168468

noncomputable def maximum_chocolates_eaten : ℕ :=
  34 * (34 - 1) / 2

theorem karlson_max_eat_chocolates : maximum_chocolates_eaten = 561 := by
  sorry

end NUMINAMATH_GPT_karlson_max_eat_chocolates_l1684_168468


namespace NUMINAMATH_GPT_find_a_l1684_168431

-- Define the slopes of the lines and the condition that they are perpendicular.
def slope1 (a : ℝ) : ℝ := a
def slope2 (a : ℝ) : ℝ := a + 2

-- The main statement of our problem.
theorem find_a (a : ℝ) (h : slope1 a * slope2 a = -1) : a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_l1684_168431


namespace NUMINAMATH_GPT_geometric_sequence_q_l1684_168481

theorem geometric_sequence_q (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 * a 6 = 16)
  (h3 : a 4 + a 8 = 8) :
  q = 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_q_l1684_168481


namespace NUMINAMATH_GPT_ratio_of_slices_l1684_168401

theorem ratio_of_slices
  (initial_slices : ℕ)
  (slices_eaten_for_lunch : ℕ)
  (remaining_slices_after_lunch : ℕ)
  (slices_left_for_tomorrow : ℕ)
  (slices_eaten_for_dinner : ℕ)
  (ratio : ℚ) :
  initial_slices = 12 → 
  slices_eaten_for_lunch = initial_slices / 2 →
  remaining_slices_after_lunch = initial_slices - slices_eaten_for_lunch →
  slices_left_for_tomorrow = 4 →
  slices_eaten_for_dinner = remaining_slices_after_lunch - slices_left_for_tomorrow →
  ratio = (slices_eaten_for_dinner : ℚ) / remaining_slices_after_lunch →
  ratio = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_slices_l1684_168401


namespace NUMINAMATH_GPT_sequence_a_5_l1684_168465

noncomputable section

-- Definition of the sequence
def a : ℕ → ℕ
| 0       => 1
| 1       => 2
| (n + 2) => a (n + 1) + a n

-- Statement to prove that a 4 = 8 (in Lean, the sequence is zero-indexed, so a 4 is a_5)
theorem sequence_a_5 : a 4 = 8 :=
  by
    sorry

end NUMINAMATH_GPT_sequence_a_5_l1684_168465


namespace NUMINAMATH_GPT_polygon_sides_l1684_168405

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 - 180 = 2190) : n = 15 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l1684_168405


namespace NUMINAMATH_GPT_minimum_n_for_candy_purchases_l1684_168478

theorem minimum_n_for_candy_purchases' {o s p : ℕ} (h1 : 9 * o = 10 * s) (h2 : 9 * o = 20 * p) : 
  ∃ n : ℕ, 30 * n = 180 ∧ ∀ m : ℕ, (30 * m = 9 * o) → n ≤ m :=
by sorry

end NUMINAMATH_GPT_minimum_n_for_candy_purchases_l1684_168478


namespace NUMINAMATH_GPT_g_at_5_l1684_168440

def g : ℝ → ℝ := sorry

axiom g_property : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 3 * x + 2

theorem g_at_5 : g 5 = -20 :=
by {
  apply sorry
}

end NUMINAMATH_GPT_g_at_5_l1684_168440


namespace NUMINAMATH_GPT_unique_polynomial_l1684_168484

-- Define the conditions
def valid_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ (p : Polynomial ℝ), Polynomial.degree p > 0 ∧ ∀ (z : ℝ), z ≠ 0 → P z = Polynomial.eval z p

-- The main theorem
theorem unique_polynomial (P : ℝ → ℝ) (hP : valid_polynomial P) :
  (∀ (z : ℝ), z ≠ 0 → P z ≠ 0 → P (1/z) ≠ 0 → 
  1 / P z + 1 / P (1 / z) = z + 1 / z) → ∀ x, P x = x :=
by
  sorry

end NUMINAMATH_GPT_unique_polynomial_l1684_168484


namespace NUMINAMATH_GPT_work_together_time_l1684_168473

theorem work_together_time (man_days : ℝ) (son_days : ℝ)
  (h_man : man_days = 5) (h_son : son_days = 7.5) :
  (1 / (1 / man_days + 1 / son_days)) = 3 :=
by
  -- Given the constraints, prove the result
  rw [h_man, h_son]
  sorry

end NUMINAMATH_GPT_work_together_time_l1684_168473


namespace NUMINAMATH_GPT_band_gigs_count_l1684_168410

-- Definitions of earnings per role and total earnings
def leadSingerEarnings := 30
def guitaristEarnings := 25
def bassistEarnings := 20
def drummerEarnings := 25
def keyboardistEarnings := 20
def backupSingerEarnings := 15
def totalEarnings := 2055

-- Calculate total per gig earnings
def totalPerGigEarnings :=
  leadSingerEarnings + guitaristEarnings + bassistEarnings + drummerEarnings + keyboardistEarnings + backupSingerEarnings

-- Statement to prove the number of gigs played is 15
theorem band_gigs_count :
  totalEarnings / totalPerGigEarnings = 15 := 
by { sorry }

end NUMINAMATH_GPT_band_gigs_count_l1684_168410


namespace NUMINAMATH_GPT_diff_of_squares_l1684_168414

theorem diff_of_squares (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) : a^2 - b^2 = -8 :=
by
  sorry

end NUMINAMATH_GPT_diff_of_squares_l1684_168414


namespace NUMINAMATH_GPT_quadrilateral_area_is_8_l1684_168466

noncomputable section
open Real

def f1 : ℝ × ℝ := (-2, 0)
def f2 : ℝ × ℝ := (2, 0)

def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

def origin_symmetric (P Q : ℝ × ℝ) : Prop := P.1 = -Q.1 ∧ P.2 = -Q.2

def distance (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def is_quadrilateral (P Q F1 F2 : ℝ × ℝ) : Prop :=
  ∃ a b c d, a = P ∧ b = F1 ∧ c = Q ∧ d = F2

def area_of_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1*B.2 + B.1*C.2 + C.1*D.2 + D.1*A.2 - (B.1*A.2 + C.1*B.2 + D.1*C.2 + A.1*D.2))

theorem quadrilateral_area_is_8 (P Q : ℝ × ℝ) :
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  origin_symmetric P Q →
  distance P Q = distance f1 f2 →
  is_quadrilateral P Q f1 f2 →
  area_of_quadrilateral P f1 Q f2 = 8 := 
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_is_8_l1684_168466


namespace NUMINAMATH_GPT_arrangement_count_l1684_168451

-- Define the problem conditions: 3 male students and 2 female students.
def male_students : ℕ := 3
def female_students : ℕ := 2
def total_students : ℕ := male_students + female_students

-- Define the condition that female students do not stand at either end.
def valid_positions_for_female : Finset ℕ := {1, 2, 3}
def valid_positions_for_male : Finset ℕ := {0, 4}

-- Theorem statement: the total number of valid arrangements is 36.
theorem arrangement_count : ∃ (n : ℕ), n = 36 := sorry

end NUMINAMATH_GPT_arrangement_count_l1684_168451


namespace NUMINAMATH_GPT_inequality_of_sum_of_squares_l1684_168490

theorem inequality_of_sum_of_squares (a b c : ℝ) (h : a * b + b * c + a * c = 1) : (a + b + c) ^ 2 ≥ 3 :=
sorry

end NUMINAMATH_GPT_inequality_of_sum_of_squares_l1684_168490


namespace NUMINAMATH_GPT_find_percentage_reduction_l1684_168435

-- Given the conditions of the problem.
def original_price : ℝ := 7500
def current_price: ℝ := 4800
def percentage_reduction (x : ℝ) : Prop := (original_price * (1 - x)^2 = current_price)

-- The statement we need to prove:
theorem find_percentage_reduction (x : ℝ) (h : percentage_reduction x) : x = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_reduction_l1684_168435


namespace NUMINAMATH_GPT_sqrt_fraction_identity_l1684_168429

theorem sqrt_fraction_identity (n : ℕ) (h : n > 0) : 
    Real.sqrt ((1 : ℝ) / n - (1 : ℝ) / (n * n)) = Real.sqrt (n - 1) / n :=
by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_identity_l1684_168429


namespace NUMINAMATH_GPT_find_x_positive_multiple_of_8_l1684_168445

theorem find_x_positive_multiple_of_8 (x : ℕ) 
  (h1 : ∃ k, x = 8 * k) 
  (h2 : x^2 > 100) 
  (h3 : x < 20) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_x_positive_multiple_of_8_l1684_168445


namespace NUMINAMATH_GPT_parabola_standard_eq_l1684_168434

theorem parabola_standard_eq (p p' : ℝ) (h₁ : p > 0) (h₂ : p' > 0) :
  (∀ (x y : ℝ), (x^2 = 2 * p * y ∨ y^2 = -2 * p' * x) → 
  (x = -2 ∧ y = 4 → (x^2 = y ∨ y^2 = -8 * x))) :=
by
  sorry

end NUMINAMATH_GPT_parabola_standard_eq_l1684_168434


namespace NUMINAMATH_GPT_temperature_increase_l1684_168433

variable (T_morning T_afternoon : ℝ)

theorem temperature_increase : 
  (T_morning = -3) → (T_afternoon = 5) → (T_afternoon - T_morning = 8) :=
by
intros h1 h2
rw [h1, h2]
sorry

end NUMINAMATH_GPT_temperature_increase_l1684_168433


namespace NUMINAMATH_GPT_sarah_age_ratio_l1684_168425

theorem sarah_age_ratio 
  (S M : ℕ) 
  (h1 : S = 3 * (S / 3))
  (h2 : S - M = 5 * (S / 3 - 2 * M)) : 
  S / M = 27 / 2 := 
sorry

end NUMINAMATH_GPT_sarah_age_ratio_l1684_168425


namespace NUMINAMATH_GPT_sqrt_meaningful_condition_l1684_168409

theorem sqrt_meaningful_condition (x : ℝ) : (2 * x + 6 >= 0) ↔ (x >= -3) := by
  sorry

end NUMINAMATH_GPT_sqrt_meaningful_condition_l1684_168409


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1684_168461

variables {a b : ℕ}

def HCF (m n : ℕ) : ℕ := m.gcd n
def LCM (m n : ℕ) : ℕ := m.lcm n

theorem sum_of_reciprocals (h_sum : a + b = 55)
                           (h_hcf : HCF a b = 5)
                           (h_lcm : LCM a b = 120) :
  (1 / a : ℚ) + (1 / b) = 11 / 120 :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1684_168461


namespace NUMINAMATH_GPT_angle_BDC_correct_l1684_168418

theorem angle_BDC_correct (A B C D : Type) 
  (angle_A : ℝ) (angle_B : ℝ) (angle_DBC : ℝ) : 
  angle_A = 60 ∧ angle_B = 70 ∧ angle_DBC = 40 → 
  ∃ angle_BDC : ℝ, angle_BDC = 100 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_angle_BDC_correct_l1684_168418


namespace NUMINAMATH_GPT_unique_k_value_l1684_168494

noncomputable def findK (k : ℝ) : Prop :=
  ∃ (x : ℝ), (x^2 - k) * (x + k + 1) = x^3 + k * (x^2 - x - 4) ∧ k ≠ 0 ∧ k = -3

theorem unique_k_value : ∀ (k : ℝ), findK k :=
by
  intro k
  sorry

end NUMINAMATH_GPT_unique_k_value_l1684_168494


namespace NUMINAMATH_GPT_current_time_l1684_168408

theorem current_time (t : ℝ) 
  (h1 : 6 * (t + 10) - (90 + 0.5 * (t - 5)) = 90 ∨ 6 * (t + 10) - (90 + 0.5 * (t - 5)) = -90) :
  t = 3 + 11 / 60 := sorry

end NUMINAMATH_GPT_current_time_l1684_168408


namespace NUMINAMATH_GPT_grassy_plot_width_l1684_168447

theorem grassy_plot_width (L : ℝ) (P : ℝ) (C : ℝ) (cost_per_sqm : ℝ) (W : ℝ) : 
  L = 110 →
  P = 2.5 →
  C = 510 →
  cost_per_sqm = 0.6 →
  (115 * (W + 5) - 110 * W = C / cost_per_sqm) →
  W = 55 :=
by
  intros hL hP hC hcost_per_sqm harea
  sorry

end NUMINAMATH_GPT_grassy_plot_width_l1684_168447


namespace NUMINAMATH_GPT_functional_equation_solution_l1684_168457

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ a : ℝ, ∀ x : ℝ, f x = x - a :=
by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1684_168457


namespace NUMINAMATH_GPT_area_of_rectangle_l1684_168449

theorem area_of_rectangle (length width : ℝ) (h1 : length = 15) (h2 : width = length * 0.9) : length * width = 202.5 := by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1684_168449


namespace NUMINAMATH_GPT_polynomial_identity_l1684_168459

open Polynomial

-- Definition of the non-zero polynomial of interest
noncomputable def p (a : ℝ) : Polynomial ℝ := Polynomial.C a * (Polynomial.X ^ 3 - Polynomial.X)

-- Theorem stating that, for all x, the given equation holds for the polynomial p
theorem polynomial_identity (a : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, (x - 1) * (p a).eval (x + 1) - (x + 2) * (p a).eval x = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_identity_l1684_168459


namespace NUMINAMATH_GPT_both_players_same_score_probability_l1684_168403

theorem both_players_same_score_probability :
  let p_A_score := 0.6
  let p_B_score := 0.8
  let p_A_miss := 1 - p_A_score
  let p_B_miss := 1 - p_B_score
  (p_A_score * p_B_score + p_A_miss * p_B_miss = 0.56) :=
by
  sorry

end NUMINAMATH_GPT_both_players_same_score_probability_l1684_168403


namespace NUMINAMATH_GPT_find_a_and_b_function_value_at_0_function_positive_x_less_than_7_over_6_l1684_168497

def linear_function (a b x : ℝ) : ℝ := a * x + b

theorem find_a_and_b : ∃ (a b : ℝ), 
  linear_function a b 1 = 1 ∧ 
  linear_function a b 2 = -5 ∧ 
  a = -6 ∧ 
  b = 7 :=
sorry

theorem function_value_at_0 : 
  ∀ a b, 
  a = -6 → b = 7 → 
  linear_function a b 0 = 7 :=
sorry

theorem function_positive_x_less_than_7_over_6 :
  ∀ a b x, 
  a = -6 → b = 7 → 
  x < 7 / 6 → 
  linear_function a b x > 0 :=
sorry

end NUMINAMATH_GPT_find_a_and_b_function_value_at_0_function_positive_x_less_than_7_over_6_l1684_168497


namespace NUMINAMATH_GPT_correct_equation_l1684_168471

theorem correct_equation : ∃a : ℝ, (-3 * a) ^ 2 = 9 * a ^ 2 :=
by
  use 1
  sorry

end NUMINAMATH_GPT_correct_equation_l1684_168471


namespace NUMINAMATH_GPT_find_H_over_G_l1684_168400

variable (G H : ℤ)
variable (x : ℝ)

-- Conditions
def condition (G H : ℤ) (x : ℝ) : Prop :=
  x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 ∧
  (↑G / (x + 7) + ↑H / (x * (x - 6)) = (x^2 - 3 * x + 15) / (x^3 + x^2 - 42 * x))

-- Theorem Statement
theorem find_H_over_G (G H : ℤ) (x : ℝ) (h : condition G H x) : (H : ℝ) / G = 15 / 7 :=
sorry

end NUMINAMATH_GPT_find_H_over_G_l1684_168400


namespace NUMINAMATH_GPT_calculate_expression_l1684_168496

theorem calculate_expression : 
  (1 - Real.sqrt 2)^0 + |(2 - Real.sqrt 5)| + (-1)^2022 - (1/3) * Real.sqrt 45 = 0 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1684_168496


namespace NUMINAMATH_GPT_derivative_at_zero_l1684_168404

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.log (1 + 2 * x^2 + x^3)) / x else 0

theorem derivative_at_zero : deriv f 0 = 2 := by
  sorry

end NUMINAMATH_GPT_derivative_at_zero_l1684_168404


namespace NUMINAMATH_GPT_problem_l1684_168456

theorem problem : (1 * (2 + 3) * 4 * 5) = 100 := by
  sorry

end NUMINAMATH_GPT_problem_l1684_168456


namespace NUMINAMATH_GPT_polynomial_inequality_l1684_168444

theorem polynomial_inequality
  (x1 x2 x3 a b c : ℝ)
  (h1 : x1 > 0) 
  (h2 : x2 > 0) 
  (h3 : x3 > 0)
  (h4 : x1 + x2 + x3 ≤ 1)
  (h5 : x1^3 + a * x1^2 + b * x1 + c = 0)
  (h6 : x2^3 + a * x2^2 + b * x2 + c = 0)
  (h7 : x3^3 + a * x3^2 + b * x3 + c = 0) :
  a^3 * (1 + a + b) - 9 * c * (3 + 3 * a + a^2) ≤ 0 :=
sorry

end NUMINAMATH_GPT_polynomial_inequality_l1684_168444


namespace NUMINAMATH_GPT_number_of_four_digit_numbers_l1684_168463

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end NUMINAMATH_GPT_number_of_four_digit_numbers_l1684_168463


namespace NUMINAMATH_GPT_max_correct_answers_l1684_168430

variables {a b c : ℕ} -- Define a, b, and c as natural numbers

theorem max_correct_answers : 
  ∀ a b c : ℕ, (a + b + c = 50) → (5 * a - 2 * c = 150) → a ≤ 35 :=
by
  -- Proof steps can be skipped by adding sorry
  sorry

end NUMINAMATH_GPT_max_correct_answers_l1684_168430


namespace NUMINAMATH_GPT_minimum_time_to_cook_3_pancakes_l1684_168411

theorem minimum_time_to_cook_3_pancakes (can_fry_two_pancakes_at_a_time : Prop) 
   (time_to_fully_cook_one_pancake : ℕ) (time_to_cook_one_side : ℕ) :
  can_fry_two_pancakes_at_a_time →
  time_to_fully_cook_one_pancake = 2 →
  time_to_cook_one_side = 1 →
  3 = 3 := 
by
  intros
  sorry

end NUMINAMATH_GPT_minimum_time_to_cook_3_pancakes_l1684_168411


namespace NUMINAMATH_GPT_max_x_real_nums_l1684_168437

theorem max_x_real_nums (x y z : ℝ) (h₁ : x + y + z = 6) (h₂ : x * y + x * z + y * z = 10) : x ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_x_real_nums_l1684_168437


namespace NUMINAMATH_GPT_total_nails_needed_l1684_168482

-- Define the conditions
def nails_already_have : ℕ := 247
def nails_found : ℕ := 144
def nails_to_buy : ℕ := 109

-- The statement to prove
theorem total_nails_needed : nails_already_have + nails_found + nails_to_buy = 500 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_total_nails_needed_l1684_168482


namespace NUMINAMATH_GPT_num_2_coins_l1684_168498

open Real

theorem num_2_coins (x y z : ℝ) (h1 : x + y + z = 900)
                     (h2 : x + 2 * y + 5 * z = 1950)
                     (h3 : z = 0.5 * x) : y = 450 :=
by sorry

end NUMINAMATH_GPT_num_2_coins_l1684_168498


namespace NUMINAMATH_GPT_hyperbola_equation_of_midpoint_l1684_168416

-- Define the hyperbola E
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Given conditions
variables (a b : ℝ) (hapos : a > 0) (hbpos : b > 0)
variables (F : ℝ × ℝ) (hF : F = (-2, 0))
variables (M : ℝ × ℝ) (hM : M = (-3, -1))

-- The statement requiring proof
theorem hyperbola_equation_of_midpoint (hE : hyperbola a b (-2) 0) 
(hFocus : a^2 + b^2 = 4) : 
  (∃ a' b', a' = 3 ∧ b' = 1 ∧ hyperbola a' b' (-3) (-1)) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_of_midpoint_l1684_168416


namespace NUMINAMATH_GPT_count_odd_perfect_squares_less_than_16000_l1684_168417

theorem count_odd_perfect_squares_less_than_16000 : 
  ∃ n : ℕ, n = 31 ∧ ∀ k < 16000, 
    ∃ b : ℕ, b = 2 * n + 1 ∧ k = (4 * n + 3) ^ 2 ∧ (∃ m : ℕ, m = b + 1 ∧ m % 2 = 0) := 
sorry

end NUMINAMATH_GPT_count_odd_perfect_squares_less_than_16000_l1684_168417


namespace NUMINAMATH_GPT_inradius_triangle_l1684_168470

theorem inradius_triangle (p A : ℝ) (h1 : p = 39) (h2 : A = 29.25) :
  ∃ r : ℝ, A = (1 / 2) * r * p ∧ r = 1.5 := by
  sorry

end NUMINAMATH_GPT_inradius_triangle_l1684_168470


namespace NUMINAMATH_GPT_each_parent_suitcases_l1684_168441

namespace SuitcaseProblem

-- Definitions based on conditions
def siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def total_suitcases : Nat := 14

-- Theorem statement corresponding to the question and correct answer
theorem each_parent_suitcases (suitcases_per_parent : Nat) :
  (siblings * suitcases_per_sibling + 2 * suitcases_per_parent = total_suitcases) →
  suitcases_per_parent = 3 := by
  intro h
  sorry

end SuitcaseProblem

end NUMINAMATH_GPT_each_parent_suitcases_l1684_168441


namespace NUMINAMATH_GPT_jaylen_has_2_cucumbers_l1684_168422

-- Definitions based on given conditions
def carrots_jaylen := 5
def bell_peppers_kristin := 2
def green_beans_kristin := 20
def total_vegetables_jaylen := 18

def bell_peppers_jaylen := 2 * bell_peppers_kristin
def green_beans_jaylen := (green_beans_kristin / 2) - 3

def known_vegetables_jaylen := carrots_jaylen + bell_peppers_jaylen + green_beans_jaylen
def cucumbers_jaylen := total_vegetables_jaylen - known_vegetables_jaylen

-- The theorem to prove
theorem jaylen_has_2_cucumbers : cucumbers_jaylen = 2 :=
by
  -- We'll place the proof here
  sorry

end NUMINAMATH_GPT_jaylen_has_2_cucumbers_l1684_168422


namespace NUMINAMATH_GPT_joan_paid_230_l1684_168412

theorem joan_paid_230 (J K : ℝ) (h1 : J + K = 600) (h2 : 2 * J = K + 90) : J = 230 :=
sorry

end NUMINAMATH_GPT_joan_paid_230_l1684_168412


namespace NUMINAMATH_GPT_total_seats_taken_l1684_168480

def students_per_bus : ℝ := 14.0
def number_of_buses : ℝ := 2.0

theorem total_seats_taken :
  students_per_bus * number_of_buses = 28.0 :=
by
  sorry

end NUMINAMATH_GPT_total_seats_taken_l1684_168480


namespace NUMINAMATH_GPT_circle_inscribed_in_square_area_l1684_168454

theorem circle_inscribed_in_square_area :
  ∀ (x y : ℝ) (h : 2 * x^2 + 2 * y^2 - 8 * x - 12 * y + 24 = 0),
  ∃ side : ℝ, 4 * (side^2) = 16 :=
by
  sorry

end NUMINAMATH_GPT_circle_inscribed_in_square_area_l1684_168454


namespace NUMINAMATH_GPT_range_of_m_l1684_168474

theorem range_of_m (m : ℝ) (P : ℝ × ℝ) (h : P = (m + 3, m - 5)) (quadrant4 : P.1 > 0 ∧ P.2 < 0) : -3 < m ∧ m < 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1684_168474


namespace NUMINAMATH_GPT_minimum_value_l1684_168420

theorem minimum_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) : 
  x^2 + 8 * x * y + 16 * y^2 + 4 * z^2 ≥ 192 := 
  sorry

end NUMINAMATH_GPT_minimum_value_l1684_168420


namespace NUMINAMATH_GPT_find_y_value_l1684_168487

theorem find_y_value
  (y z : ℝ)
  (h1 : y + z + 175 = 360)
  (h2 : z = y + 10) :
  y = 88 :=
by
  sorry

end NUMINAMATH_GPT_find_y_value_l1684_168487


namespace NUMINAMATH_GPT_point_between_lines_l1684_168483

theorem point_between_lines (b : ℝ) (h1 : 6 * 5 - 8 * b + 1 < 0) (h2 : 3 * 5 - 4 * b + 5 > 0) : b = 4 :=
  sorry

end NUMINAMATH_GPT_point_between_lines_l1684_168483


namespace NUMINAMATH_GPT_initial_group_machines_l1684_168452

-- Define the number of bags produced by n machines in one minute and 150 machines in one minute
def bags_produced (machines : ℕ) (bags_per_minute : ℕ) : Prop :=
  machines * bags_per_minute = 45

def bags_produced_150 (bags_produced_in_8_mins : ℕ) : Prop :=
  150 * (bags_produced_in_8_mins / 8) = 450

-- Given the conditions, prove that the number of machines in the initial group is 15
theorem initial_group_machines (n : ℕ) (bags_produced_in_8_mins : ℕ) :
  bags_produced n 45 → bags_produced_150 bags_produced_in_8_mins → n = 15 :=
by
  intro h1 h2
  -- use the conditions to derive the result
  sorry

end NUMINAMATH_GPT_initial_group_machines_l1684_168452


namespace NUMINAMATH_GPT_measure_of_angle_D_l1684_168467

-- Definitions of angles in pentagon ABCDE
variables (A B C D E : ℝ)

-- Conditions
def condition1 := D = A + 30
def condition2 := E = A + 50
def condition3 := B = C
def condition4 := A = B - 45
def condition5 := A + B + C + D + E = 540

-- Theorem to prove
theorem measure_of_angle_D (h1 : condition1 A D)
                           (h2 : condition2 A E)
                           (h3 : condition3 B C)
                           (h4 : condition4 A B)
                           (h5 : condition5 A B C D E) :
  D = 104 :=
sorry

end NUMINAMATH_GPT_measure_of_angle_D_l1684_168467


namespace NUMINAMATH_GPT_water_pumping_problem_l1684_168458

theorem water_pumping_problem :
  let pumpA_rate := 300 -- gallons per hour
  let pumpB_rate := 500 -- gallons per hour
  let combined_rate := pumpA_rate + pumpB_rate -- Combined rate per hour
  let time_duration := 1 / 2 -- Time in hours (30 minutes)
  combined_rate * time_duration = 400 := -- Total volume in gallons
by
  -- Lean proof would go here
  sorry

end NUMINAMATH_GPT_water_pumping_problem_l1684_168458


namespace NUMINAMATH_GPT_dentist_ratio_l1684_168493

-- Conditions
def cost_cleaning : ℕ := 70
def cost_filling : ℕ := 120
def cost_extraction : ℕ := 290

-- Theorem statement
theorem dentist_ratio : (cost_cleaning + 2 * cost_filling + cost_extraction) / cost_filling = 5 := 
by
  -- To be proven
  sorry

end NUMINAMATH_GPT_dentist_ratio_l1684_168493


namespace NUMINAMATH_GPT_union_of_sets_l1684_168485

def A := { x : ℝ | -1 ≤ x ∧ x ≤ 5 }
def B := { x : ℝ | 3 < x ∧ x < 9 }

theorem union_of_sets : (A ∪ B) = { x : ℝ | -1 ≤ x ∧ x < 9 } :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l1684_168485


namespace NUMINAMATH_GPT_vaclav_multiplication_correct_l1684_168415

-- Definitions of the involved numbers and their multiplication consistency.
def a : ℕ := 452
def b : ℕ := 125
def result : ℕ := 56500

-- The main theorem statement proving the correctness of the multiplication.
theorem vaclav_multiplication_correct : a * b = result :=
by sorry

end NUMINAMATH_GPT_vaclav_multiplication_correct_l1684_168415


namespace NUMINAMATH_GPT_purely_imaginary_complex_number_l1684_168462

theorem purely_imaginary_complex_number (a : ℝ) (h : (a^2 - 3 * a + 2) = 0 ∧ (a - 2) ≠ 0) : a = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_purely_imaginary_complex_number_l1684_168462


namespace NUMINAMATH_GPT_jane_cycling_time_difference_l1684_168477

theorem jane_cycling_time_difference :
  (3 * 5 / 6.5 - (5 / 10 + 5 / 5 + 5 / 8)) * 60 = 11 :=
by sorry

end NUMINAMATH_GPT_jane_cycling_time_difference_l1684_168477


namespace NUMINAMATH_GPT_inequality_is_linear_l1684_168479

theorem inequality_is_linear (k : ℝ) (h1 : (|k| - 1) = 1) (h2 : (k + 2) ≠ 0) : k = 2 :=
sorry

end NUMINAMATH_GPT_inequality_is_linear_l1684_168479


namespace NUMINAMATH_GPT_mary_max_earnings_l1684_168436

def max_hours : ℕ := 40
def regular_rate : ℝ := 8
def first_hours : ℕ := 20
def overtime_rate : ℝ := regular_rate + 0.25 * regular_rate

def earnings : ℝ := 
  (first_hours * regular_rate) +
  ((max_hours - first_hours) * overtime_rate)

theorem mary_max_earnings : earnings = 360 := by
  sorry

end NUMINAMATH_GPT_mary_max_earnings_l1684_168436


namespace NUMINAMATH_GPT_q1_q2_l1684_168464

variable (a b : ℝ)

-- Definition of the conditions
def conditions : Prop := a + b = 7 ∧ a * b = 6

-- Statement of the first question
theorem q1 (h : conditions a b) : a^2 + b^2 = 37 := sorry

-- Statement of the second question
theorem q2 (h : conditions a b) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = 150 := sorry

end NUMINAMATH_GPT_q1_q2_l1684_168464


namespace NUMINAMATH_GPT_arithmetic_mean_of_scores_l1684_168446

theorem arithmetic_mean_of_scores :
  let s1 := 85
  let s2 := 94
  let s3 := 87
  let s4 := 93
  let s5 := 95
  let s6 := 88
  let s7 := 90
  (s1 + s2 + s3 + s4 + s5 + s6 + s7) / 7 = 90.2857142857 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_scores_l1684_168446


namespace NUMINAMATH_GPT_infinitenat_not_sum_square_prime_l1684_168423

theorem infinitenat_not_sum_square_prime : ∀ k : ℕ, ¬ ∃ (n : ℕ) (p : ℕ), Prime p ∧ (3 * k + 2) ^ 2 = n ^ 2 + p :=
by
  intro k
  sorry

end NUMINAMATH_GPT_infinitenat_not_sum_square_prime_l1684_168423


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1684_168438

theorem problem_1 (avg_daily_production : ℕ) (deviation_wed : ℤ) :
  avg_daily_production = 3000 →
  deviation_wed = -15 →
  avg_daily_production + deviation_wed = 2985 :=
by intros; sorry

theorem problem_2 (avg_daily_production : ℕ) (deviation_sat : ℤ) (deviation_fri : ℤ) :
  avg_daily_production = 3000 →
  deviation_sat = 68 →
  deviation_fri = -20 →
  (avg_daily_production + deviation_sat) - (avg_daily_production + deviation_fri) = 88 :=
by intros; sorry

theorem problem_3 (planned_weekly_production : ℕ) (deviations : List ℤ) :
  planned_weekly_production = 21000 →
  deviations = [35, -12, -15, 30, -20, 68, -9] →
  planned_weekly_production + deviations.sum = 21077 :=
by intros; sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l1684_168438


namespace NUMINAMATH_GPT_exponent_simplification_l1684_168489

theorem exponent_simplification : (7^3 * (2^5)^3) / (7^2 * 2^(3*3)) = 448 := by
  sorry

end NUMINAMATH_GPT_exponent_simplification_l1684_168489


namespace NUMINAMATH_GPT_fill_tub_together_time_l1684_168455

theorem fill_tub_together_time :
  let rate1 := 1 / 4
  let rate2 := 1 / 4
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  combined_rate ≠ 0 → (1 / combined_rate = 12 / 7) :=
by
  let rate1 := 1 / 4
  let rate2 := 1 / 4
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  sorry

end NUMINAMATH_GPT_fill_tub_together_time_l1684_168455


namespace NUMINAMATH_GPT_Seth_bought_20_cartons_of_ice_cream_l1684_168407

-- Definitions from conditions
def ice_cream_cost_per_carton : ℕ := 6
def yogurt_cost_per_carton : ℕ := 1
def num_yogurt_cartons : ℕ := 2
def extra_amount_spent_on_ice_cream : ℕ := 118

-- Let x be the number of cartons of ice cream Seth bought
def num_ice_cream_cartons (x : ℕ) : Prop :=
  ice_cream_cost_per_carton * x = num_yogurt_cartons * yogurt_cost_per_carton + extra_amount_spent_on_ice_cream

-- The proof goal
theorem Seth_bought_20_cartons_of_ice_cream : num_ice_cream_cartons 20 :=
by
  unfold num_ice_cream_cartons
  unfold ice_cream_cost_per_carton yogurt_cost_per_carton num_yogurt_cartons extra_amount_spent_on_ice_cream
  sorry

end NUMINAMATH_GPT_Seth_bought_20_cartons_of_ice_cream_l1684_168407


namespace NUMINAMATH_GPT_midpoint_of_five_points_on_grid_l1684_168406

theorem midpoint_of_five_points_on_grid 
    (points : Fin 5 → ℤ × ℤ) :
    ∃ i j : Fin 5, i ≠ j ∧ ((points i).fst + (points j).fst) % 2 = 0 
    ∧ ((points i).snd + (points j).snd) % 2 = 0 :=
by sorry

end NUMINAMATH_GPT_midpoint_of_five_points_on_grid_l1684_168406


namespace NUMINAMATH_GPT_pre_image_of_f_l1684_168488

theorem pre_image_of_f (x y : ℝ) (f : ℝ × ℝ → ℝ × ℝ) 
  (h : f = λ p => (2 * p.1 + p.2, p.1 - 2 * p.2)) :
  f (1, 0) = (2, 1) := by
  sorry

end NUMINAMATH_GPT_pre_image_of_f_l1684_168488


namespace NUMINAMATH_GPT_positive_integer_solutions_l1684_168453

theorem positive_integer_solutions (n m : ℕ) (h : n > 0 ∧ m > 0) : 
  (n + 1) * m = n! + 1 ↔ (n = 1 ∧ m = 1) ∨ (n = 2 ∧ m = 1) ∨ (n = 4 ∧ m = 5) := by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_l1684_168453


namespace NUMINAMATH_GPT_length_of_longest_side_l1684_168419

theorem length_of_longest_side (l w : ℝ) (h_fencing : 2 * l + 2 * w = 240) (h_area : l * w = 8 * 240) : max l w = 96 :=
by sorry

end NUMINAMATH_GPT_length_of_longest_side_l1684_168419


namespace NUMINAMATH_GPT_false_prop_range_of_a_l1684_168427

theorem false_prop_range_of_a (a : ℝ) :
  (¬ ∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ (a < -2 * Real.sqrt 2 ∨ a > 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_false_prop_range_of_a_l1684_168427


namespace NUMINAMATH_GPT_complex_fraction_equivalence_l1684_168495

/-- The complex number 2 / (1 - i) is equal to 1 + i. -/
theorem complex_fraction_equivalence : (2 : ℂ) / (1 - (I : ℂ)) = 1 + (I : ℂ) := by
  sorry

end NUMINAMATH_GPT_complex_fraction_equivalence_l1684_168495


namespace NUMINAMATH_GPT_paula_twice_as_old_as_karl_6_years_later_l1684_168439

theorem paula_twice_as_old_as_karl_6_years_later
  (P K : ℕ)
  (h1 : P - 5 = 3 * (K - 5))
  (h2 : P + K = 54) :
  P + 6 = 2 * (K + 6) :=
sorry

end NUMINAMATH_GPT_paula_twice_as_old_as_karl_6_years_later_l1684_168439


namespace NUMINAMATH_GPT_smaller_inscribed_cube_volume_is_192_sqrt_3_l1684_168472

noncomputable def volume_of_smaller_inscribed_cube : ℝ :=
  let edge_length_of_larger_cube := 12
  let diameter_of_sphere := edge_length_of_larger_cube
  let side_length_of_smaller_cube := diameter_of_sphere / Real.sqrt 3
  let volume := side_length_of_smaller_cube ^ 3
  volume

theorem smaller_inscribed_cube_volume_is_192_sqrt_3 : 
  volume_of_smaller_inscribed_cube = 192 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_smaller_inscribed_cube_volume_is_192_sqrt_3_l1684_168472


namespace NUMINAMATH_GPT_triangle_perimeter_l1684_168413

theorem triangle_perimeter (a b c : ℕ) (ha : a = 14) (hb : b = 8) (hc : c = 9) : a + b + c = 31 := 
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1684_168413


namespace NUMINAMATH_GPT_acres_used_for_corn_l1684_168426

-- Define the conditions given in the problem
def total_land : ℕ := 1034
def ratio_beans : ℕ := 5
def ratio_wheat : ℕ := 2
def ratio_corn : ℕ := 4
def total_ratio_parts : ℕ := ratio_beans + ratio_wheat + ratio_corn
def part_size : ℕ := total_land / total_ratio_parts

-- State the theorem to prove that the land used for corn is 376 acres
theorem acres_used_for_corn : (part_size * ratio_corn = 376) :=
  sorry

end NUMINAMATH_GPT_acres_used_for_corn_l1684_168426


namespace NUMINAMATH_GPT_find_intersection_l1684_168442

noncomputable def A : Set ℝ := { x | -4 < x ∧ x < 3 }
noncomputable def B : Set ℝ := { x | x ≤ 2 }

theorem find_intersection : A ∩ B = { x | -4 < x ∧ x ≤ 2 } := sorry

end NUMINAMATH_GPT_find_intersection_l1684_168442


namespace NUMINAMATH_GPT_g_at_5_l1684_168460

def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ (x : ℝ), g x + 2 * g (1 - x) = x^2 + 2 * x

theorem g_at_5 : g 5 = -19 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_g_at_5_l1684_168460


namespace NUMINAMATH_GPT_fourth_intersection_point_of_curve_and_circle_l1684_168450

theorem fourth_intersection_point_of_curve_and_circle (h k R : ℝ)
  (h1 : (3 - h)^2 + (2 / 3 - k)^2 = R^2)
  (h2 : (-4 - h)^2 + (-1 / 2 - k)^2 = R^2)
  (h3 : (1 / 2 - h)^2 + (4 - k)^2 = R^2) :
  ∃ (x y : ℝ), xy = 2 ∧ (x, y) ≠ (3, 2 / 3) ∧ (x, y) ≠ (-4, -1 / 2) ∧ (x, y) ≠ (1 / 2, 4) ∧ 
    (x - h)^2 + (y - k)^2 = R^2 ∧ (x, y) = (2 / 3, 3) := 
sorry

end NUMINAMATH_GPT_fourth_intersection_point_of_curve_and_circle_l1684_168450


namespace NUMINAMATH_GPT_set_union_l1684_168421

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem set_union : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

end NUMINAMATH_GPT_set_union_l1684_168421
