import Mathlib

namespace NUMINAMATH_GPT_quadrilateral_equality_l734_73466

-- Variables definitions for points and necessary properties
variables {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assumptions based on given conditions
variables (AB : ℝ) (AD : ℝ) (BC : ℝ) (DC : ℝ) (beta : ℝ)
variables {angleB : ℝ} {angleD : ℝ}

-- Given conditions
axiom AB_eq_AD : AB = AD
axiom angleB_eq_angleD : angleB = angleD

-- The statement to be proven
theorem quadrilateral_equality (h1 : AB = AD) (h2 : angleB = angleD) : BC = DC :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_equality_l734_73466


namespace NUMINAMATH_GPT_value_divided_by_is_three_l734_73493

theorem value_divided_by_is_three (x : ℝ) (h : 72 / x = 24) : x = 3 := 
by
  sorry

end NUMINAMATH_GPT_value_divided_by_is_three_l734_73493


namespace NUMINAMATH_GPT_chloe_sold_strawberries_l734_73489

noncomputable section

def cost_per_dozen : ℕ := 50
def sale_price_per_half_dozen : ℕ := 30
def total_profit : ℕ := 500
def profit_per_half_dozen := sale_price_per_half_dozen - (cost_per_dozen / 2)
def half_dozens_sold := total_profit / profit_per_half_dozen

theorem chloe_sold_strawberries : half_dozens_sold / 2 = 50 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_chloe_sold_strawberries_l734_73489


namespace NUMINAMATH_GPT_circumcircle_radius_l734_73426

theorem circumcircle_radius (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) :
  let s₁ := a^2 + b^2
  let s₂ := c^2
  s₁ = s₂ → 
  (c / 2) = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_circumcircle_radius_l734_73426


namespace NUMINAMATH_GPT_count_solutions_congruence_l734_73401

theorem count_solutions_congruence (x : ℕ) (h1 : 0 < x ∧ x < 50) (h2 : x + 7 ≡ 45 [MOD 22]) : ∃ x1 x2, (x1 ≠ x2) ∧ (0 < x1 ∧ x1 < 50) ∧ (0 < x2 ∧ x2 < 50) ∧ (x1 + 7 ≡ 45 [MOD 22]) ∧ (x2 + 7 ≡ 45 [MOD 22]) ∧ (∀ y, (0 < y ∧ y < 50) ∧ (y + 7 ≡ 45 [MOD 22]) → (y = x1 ∨ y = x2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_count_solutions_congruence_l734_73401


namespace NUMINAMATH_GPT_solve_system_of_equations_l734_73452

theorem solve_system_of_equations (x y z : ℝ) :
  x + y + z = 2 →
  x * y * z = 2 * (x * y + y * z + z * x) →
  ((x = -y ∧ z = 2) ∨ (y = -z ∧ x = 2) ∨ (z = -x ∧ y = 2)) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l734_73452


namespace NUMINAMATH_GPT_evaluate_expression_l734_73446

theorem evaluate_expression :
  (4 * 6) / (12 * 16) * (8 * 12 * 16) / (4 * 6 * 8) = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l734_73446


namespace NUMINAMATH_GPT_fraction_zero_implies_x_neg1_l734_73418

theorem fraction_zero_implies_x_neg1 (x : ℝ) (h₁ : x^2 - 1 = 0) (h₂ : x - 1 ≠ 0) : x = -1 := by
  sorry

end NUMINAMATH_GPT_fraction_zero_implies_x_neg1_l734_73418


namespace NUMINAMATH_GPT_max_cos_product_l734_73465

open Real

theorem max_cos_product (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
                                       (hβ : 0 < β ∧ β < π / 2)
                                       (hγ : 0 < γ ∧ γ < π / 2)
                                       (h : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) : 
  cos α * cos β * cos γ ≤ 2 * Real.sqrt 6 / 9 := 
by sorry

end NUMINAMATH_GPT_max_cos_product_l734_73465


namespace NUMINAMATH_GPT_circle_eq_focus_tangent_directrix_l734_73479

theorem circle_eq_focus_tangent_directrix (x y : ℝ) :
  let focus := (0, 4)
  let directrix := -4
  let radius := 8
  ((x - focus.1)^2 + (y - focus.2)^2 = radius^2) :=
by
  let focus := (0, 4)
  let directrix := -4
  let radius := 8
  sorry

end NUMINAMATH_GPT_circle_eq_focus_tangent_directrix_l734_73479


namespace NUMINAMATH_GPT_work_rate_l734_73471

theorem work_rate (x : ℕ) (hx : 2 * x = 30) : x = 15 := by
  -- We assume the prerequisite 2 * x = 30
  sorry

end NUMINAMATH_GPT_work_rate_l734_73471


namespace NUMINAMATH_GPT_cone_lateral_area_l734_73404

-- Definitions from the conditions
def radius_base : ℝ := 1 -- in cm
def slant_height : ℝ := 2 -- in cm

-- Statement to be proved: The lateral area of the cone is 2π cm²
theorem cone_lateral_area : 
  1/2 * (2 * π * radius_base) * slant_height = 2 * π :=
by
  sorry

end NUMINAMATH_GPT_cone_lateral_area_l734_73404


namespace NUMINAMATH_GPT_number_of_glasses_l734_73400

theorem number_of_glasses (oranges_per_glass total_oranges : ℕ) 
  (h1 : oranges_per_glass = 2) 
  (h2 : total_oranges = 12) : 
  total_oranges / oranges_per_glass = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_glasses_l734_73400


namespace NUMINAMATH_GPT_total_investment_with_interest_l734_73464

def principal : ℝ := 1000
def part3Percent : ℝ := 199.99999999999983
def rate3Percent : ℝ := 0.03
def rate5Percent : ℝ := 0.05
def interest3Percent : ℝ := part3Percent * rate3Percent
def part5Percent : ℝ := principal - part3Percent
def interest5Percent : ℝ := part5Percent * rate5Percent
def totalWithInterest : ℝ := principal + interest3Percent + interest5Percent

theorem total_investment_with_interest :
  totalWithInterest = 1046.00 :=
by
  unfold totalWithInterest interest5Percent part5Percent interest3Percent
  sorry

end NUMINAMATH_GPT_total_investment_with_interest_l734_73464


namespace NUMINAMATH_GPT_Tino_jellybeans_l734_73487

variable (L T A : ℕ)
variable (h1 : T = L + 24)
variable (h2 : A = L / 2)
variable (h3 : A = 5)

theorem Tino_jellybeans : T = 34 :=
by
  sorry

end NUMINAMATH_GPT_Tino_jellybeans_l734_73487


namespace NUMINAMATH_GPT_dave_deleted_apps_l734_73470

def apps_initial : ℕ := 23
def apps_left : ℕ := 5
def apps_deleted : ℕ := apps_initial - apps_left

theorem dave_deleted_apps : apps_deleted = 18 := 
by
  sorry

end NUMINAMATH_GPT_dave_deleted_apps_l734_73470


namespace NUMINAMATH_GPT_find_number_l734_73440

-- Define the problem conditions
def problem_condition (x : ℝ) : Prop := 2 * x - x / 2 = 45

-- Main theorem statement
theorem find_number : ∃ (x : ℝ), problem_condition x ∧ x = 30 :=
by
  existsi 30
  -- Include the problem condition and the solution check
  unfold problem_condition
  -- We are skipping the proof using sorry to just provide the statement
  sorry

end NUMINAMATH_GPT_find_number_l734_73440


namespace NUMINAMATH_GPT_deriv_y1_deriv_y2_deriv_y3_l734_73412

variable (x : ℝ)

-- Prove the derivative of y = 3x^3 - 4x is 9x^2 - 4
theorem deriv_y1 : deriv (λ x => 3 * x^3 - 4 * x) x = 9 * x^2 - 4 := by
sorry

-- Prove the derivative of y = (2x - 1)(3x + 2) is 12x + 1
theorem deriv_y2 : deriv (λ x => (2 * x - 1) * (3 * x + 2)) x = 12 * x + 1 := by
sorry

-- Prove the derivative of y = x^2 (x^3 - 4) is 5x^4 - 8x
theorem deriv_y3 : deriv (λ x => x^2 * (x^3 - 4)) x = 5 * x^4 - 8 * x := by
sorry


end NUMINAMATH_GPT_deriv_y1_deriv_y2_deriv_y3_l734_73412


namespace NUMINAMATH_GPT_quadratic_negativity_cond_l734_73443

theorem quadratic_negativity_cond {x m k : ℝ} :
  (∀ x, x^2 - m * x - k + m < 0) ↔ k > m - (m^2 / 4) :=
sorry

end NUMINAMATH_GPT_quadratic_negativity_cond_l734_73443


namespace NUMINAMATH_GPT_part_I_part_II_l734_73422

-- Translate the conditions and questions to Lean definition statements.

-- First part of the problem: proving the value of a
theorem part_I (a : ℝ) (f : ℝ → ℝ) (Hf : ∀ x, f x = |a * x - 1|) 
(Hsol : ∀ x, f x ≤ 2 ↔ -6 ≤ x ∧ x ≤ 2) : a = -1 / 2 :=
sorry

-- Second part of the problem: proving the range of m
theorem part_II (m : ℝ) 
(H : ∃ x : ℝ, |4 * x + 1| - |2 * x - 3| ≤ 7 - 3 * m) : m ≤ 7 / 2 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l734_73422


namespace NUMINAMATH_GPT_max_min_AB_length_chord_length_at_angle_trajectory_midpoint_chord_l734_73444

noncomputable def point_in_circle : Prop :=
  let P := (-Real.sqrt 3, 2)
  ∃ (x y : ℝ), x^2 + y^2 = 12 ∧ x = -Real.sqrt 3 ∧ y = 2

theorem max_min_AB_length (α : ℝ) (h1 : -Real.sqrt 3 ≤ α ∧ α ≤ Real.pi / 2) :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  let R := Real.sqrt 12
  ∀ (A B : ℝ × ℝ), (A.1^2 + A.2^2 = 12 ∧ B.1^2 + B.2^2 = 12 ∧ (P.1, P.2) = (-Real.sqrt 3, 2)) →
    ((max (dist A B) (dist P P)) = 4 * Real.sqrt 3 ∧ (min (dist A B) (dist P P)) = 2 * Real.sqrt 5) :=
sorry

theorem chord_length_at_angle (α : ℝ) (h2 : α = 120 / 180 * Real.pi) :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  let A := (Real.sqrt 12, 0)
  let B := (-Real.sqrt 12, 0)
  let AB := (dist A B)
  AB = Real.sqrt 47 :=
sorry

theorem trajectory_midpoint_chord :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  ∀ (M : ℝ × ℝ), (∀ k : ℝ, P.2 - 2 = k * (P.1 + Real.sqrt 3) ∧ M.2 = - 1 / k * M.1) → 
  (M.1^2 + M.2^2 + Real.sqrt 3 * M.1 + 2 * M.2 = 0) :=
sorry

end NUMINAMATH_GPT_max_min_AB_length_chord_length_at_angle_trajectory_midpoint_chord_l734_73444


namespace NUMINAMATH_GPT_equal_sets_d_l734_73495

theorem equal_sets_d : 
  (let M := {x | x^2 - 3*x + 2 = 0}
   let N := {1, 2}
   M = N) :=
by 
  sorry

end NUMINAMATH_GPT_equal_sets_d_l734_73495


namespace NUMINAMATH_GPT_red_balls_count_l734_73492

theorem red_balls_count (y : ℕ) (p_yellow : ℚ) (h1 : y = 10)
  (h2 : p_yellow = 5/8) (total_balls_le : ∀ r : ℕ, y + r ≤ 32) :
  ∃ r : ℕ, 10 + r > 0 ∧ p_yellow = 10 / (10 + r) ∧ r = 6 :=
by
  sorry

end NUMINAMATH_GPT_red_balls_count_l734_73492


namespace NUMINAMATH_GPT_triangle_perimeter_l734_73402

theorem triangle_perimeter (x : ℕ) (a b c : ℕ) 
  (h1 : a = 3 * x) (h2 : b = 4 * x) (h3 : c = 5 * x)  
  (h4 : c - a = 6) : a + b + c = 36 := 
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l734_73402


namespace NUMINAMATH_GPT_water_in_pool_after_35_days_l734_73483

theorem water_in_pool_after_35_days :
  ∀ (initial_amount : ℕ) (evap_rate : ℕ) (cycle_days : ℕ) (add_amount : ℕ) (total_days : ℕ),
  initial_amount = 300 → evap_rate = 1 → cycle_days = 5 → add_amount = 5 → total_days = 35 →
  initial_amount - evap_rate * total_days + (total_days / cycle_days) * add_amount = 300 :=
by
  intros initial_amount evap_rate cycle_days add_amount total_days h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_water_in_pool_after_35_days_l734_73483


namespace NUMINAMATH_GPT_cd_player_percentage_l734_73475

-- Define the percentage variables
def powerWindowsAndAntiLock : ℝ := 0.10
def antiLockAndCdPlayer : ℝ := 0.15
def powerWindowsAndCdPlayer : ℝ := 0.22
def cdPlayerAlone : ℝ := 0.38

-- Define the problem statement
theorem cd_player_percentage : 
  powerWindowsAndAntiLock = 0.10 → 
  antiLockAndCdPlayer = 0.15 → 
  powerWindowsAndCdPlayer = 0.22 → 
  cdPlayerAlone = 0.38 → 
  (antiLockAndCdPlayer + powerWindowsAndCdPlayer + cdPlayerAlone) = 0.75 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cd_player_percentage_l734_73475


namespace NUMINAMATH_GPT_Dawn_has_10_CDs_l734_73451

-- Lean definition of the problem conditions
def Kristine_more_CDs (D K : ℕ) : Prop :=
  K = D + 7

def Total_CDs (D K : ℕ) : Prop :=
  D + K = 27

-- Lean statement of the proof
theorem Dawn_has_10_CDs (D K : ℕ) (h1 : Kristine_more_CDs D K) (h2 : Total_CDs D K) : D = 10 :=
by
  sorry

end NUMINAMATH_GPT_Dawn_has_10_CDs_l734_73451


namespace NUMINAMATH_GPT_triangle_is_right_angled_l734_73498

noncomputable def median (a b c : ℝ) : ℝ := (1 / 2) * (Real.sqrt (2 * b^2 + 2 * c^2 - a^2))

theorem triangle_is_right_angled (a b c : ℝ) (ha : median a b c = 5) (hb : median b c a = Real.sqrt 52) (hc : median c a b = Real.sqrt 73) :
  a^2 = b^2 + c^2 :=
sorry

end NUMINAMATH_GPT_triangle_is_right_angled_l734_73498


namespace NUMINAMATH_GPT_relationship_between_roses_and_total_flowers_l734_73485

variables (C V T R F : ℝ)
noncomputable def F_eq_64_42376521116678_percent_of_C := 
  C = 0.6442376521116678 * F

def V_eq_one_third_of_C := 
  V = (1 / 3) * C

def T_eq_one_ninth_of_C := 
  T = (1 / 9) * C

def F_eq_C_plus_V_plus_T_plus_R := 
  F = C + V + T + R

theorem relationship_between_roses_and_total_flowers (C V T R F : ℝ) 
    (h1 : C = 0.6442376521116678 * F)
    (h2 : V = 1 / 3 * C)
    (h3 : T = 1 / 9 * C)
    (h4 : F = C + V + T + R) :
    R = F - 13 / 9 * C := 
  by sorry

end NUMINAMATH_GPT_relationship_between_roses_and_total_flowers_l734_73485


namespace NUMINAMATH_GPT_MrsHiltTravelMiles_l734_73405

theorem MrsHiltTravelMiles
  (one_book_miles : ℕ)
  (finished_books : ℕ)
  (total_miles : ℕ)
  (h1 : one_book_miles = 450)
  (h2 : finished_books = 15)
  (h3 : total_miles = one_book_miles * finished_books) :
  total_miles = 6750 :=
by
  sorry

end NUMINAMATH_GPT_MrsHiltTravelMiles_l734_73405


namespace NUMINAMATH_GPT_initial_bottles_count_l734_73447

theorem initial_bottles_count
  (players : ℕ)
  (bottles_per_player_first_break : ℕ)
  (bottles_per_player_end_game : ℕ)
  (remaining_bottles : ℕ)
  (total_bottles_taken_first_break : bottles_per_player_first_break * players = 22)
  (total_bottles_taken_end_game : bottles_per_player_end_game * players = 11)
  (total_remaining_bottles : remaining_bottles = 15) :
  players * bottles_per_player_first_break + players * bottles_per_player_end_game + remaining_bottles = 48 :=
by 
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_initial_bottles_count_l734_73447


namespace NUMINAMATH_GPT_Angelina_drive_time_equation_l734_73430

theorem Angelina_drive_time_equation (t : ℝ) 
    (h_speed1 : ∀ t: ℝ, 70 * t = 70 * t)
    (h_stop : 0.5 = 0.5) 
    (h_speed2 : ∀ t: ℝ, 90 * t = 90 * t) 
    (h_total_distance : 300 = 300) 
    (h_total_time : 4 = 4) 
    : 70 * t + 90 * (3.5 - t) = 300 :=
by
  sorry

end NUMINAMATH_GPT_Angelina_drive_time_equation_l734_73430


namespace NUMINAMATH_GPT_union_A_B_l734_73449

-- Definitions for the sets A and B
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- The statement to be proven
theorem union_A_B :
  A ∪ B = {x | (-1 < x ∧ x ≤ 3) ∨ x = 4} :=
sorry

end NUMINAMATH_GPT_union_A_B_l734_73449


namespace NUMINAMATH_GPT_probability_average_is_five_l734_73469

-- Definitions and conditions
def numbers : List ℕ := [1, 3, 4, 6, 7, 9]

def average_is_five (a b : ℕ) : Prop := (a + b) / 2 = 5

-- Desired statement
theorem probability_average_is_five : 
  ∃ p : ℚ, p = 1 / 5 ∧ (∃ a b : ℕ, a ∈ numbers ∧ b ∈ numbers ∧ average_is_five a b) := 
sorry

end NUMINAMATH_GPT_probability_average_is_five_l734_73469


namespace NUMINAMATH_GPT_inequality_always_holds_l734_73459

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_always_holds_l734_73459


namespace NUMINAMATH_GPT_cattle_train_speed_is_56_l734_73455

variable (v : ℝ)

def cattle_train_speed :=
  let cattle_distance_until_diesel_starts := 6 * v
  let diesel_speed := v - 33
  let diesel_distance := 12 * diesel_speed
  let cattle_additional_distance := 12 * v
  let total_distance := cattle_distance_until_diesel_starts + diesel_distance + cattle_additional_distance
  total_distance = 1284

theorem cattle_train_speed_is_56 (h : cattle_train_speed v) : v = 56 :=
  sorry

end NUMINAMATH_GPT_cattle_train_speed_is_56_l734_73455


namespace NUMINAMATH_GPT_find_x_l734_73460

theorem find_x :
  (12^3 * 6^3) / x = 864 → x = 432 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l734_73460


namespace NUMINAMATH_GPT_fill_cistern_time_l734_73437

theorem fill_cistern_time (F E : ℝ) (hF : F = 1/2) (hE : E = 1/4) : 
  (1 / (F - E)) = 4 :=
by
  -- Definitions of F and E are used as hypotheses hF and hE
  -- Prove the actual theorem stating the time to fill the cistern is 4 hours
  sorry

end NUMINAMATH_GPT_fill_cistern_time_l734_73437


namespace NUMINAMATH_GPT_negation_of_proposition_l734_73435

theorem negation_of_proposition (a b : ℝ) : 
  (¬ (∀ (a b : ℝ), (ab > 0 → a > 0)) ↔ ∀ (a b : ℝ), (ab ≤ 0 → a ≤ 0)) := 
sorry

end NUMINAMATH_GPT_negation_of_proposition_l734_73435


namespace NUMINAMATH_GPT_value_of_y_l734_73463

theorem value_of_y (y : ℕ) (h : 9 / (y^2) = y / 81) : y = 9 :=
by
-- Since we are only required to state the theorem, we leave the proof out for now.
sorry

end NUMINAMATH_GPT_value_of_y_l734_73463


namespace NUMINAMATH_GPT_books_in_shipment_l734_73497

theorem books_in_shipment (B : ℕ) (h : 3 / 4 * B = 180) : B = 240 :=
sorry

end NUMINAMATH_GPT_books_in_shipment_l734_73497


namespace NUMINAMATH_GPT_rod_length_difference_l734_73408

theorem rod_length_difference (L₁ L₂ : ℝ) (h1 : L₁ + L₂ = 33)
    (h2 : (∀ x : ℝ, x = (2 / 3) * L₁ ∧ x = (4 / 5) * L₂)) :
    abs (L₁ - L₂) = 3 := by
  sorry

end NUMINAMATH_GPT_rod_length_difference_l734_73408


namespace NUMINAMATH_GPT_points_on_ellipse_l734_73474

-- Definitions of the conditions
def ellipse (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

def passes_through_point (a b : ℝ) : Prop :=
  ellipse a b 2 1

-- Target set of points
def target_set (x y : ℝ) : Prop :=
  x^2 + y^2 < 5 ∧ |y| > 1

-- Main theorem to prove
theorem points_on_ellipse (a b x y : ℝ) (h₁ : passes_through_point a b) (h₂ : |y| > 1) :
  ellipse a b x y → target_set x y :=
sorry

end NUMINAMATH_GPT_points_on_ellipse_l734_73474


namespace NUMINAMATH_GPT_solve_for_a_l734_73413

open Complex

theorem solve_for_a (a : ℝ) (h : (2 + a * I) * (a - 2 * I) = -4 * I) : a = 0 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l734_73413


namespace NUMINAMATH_GPT_work_completion_days_l734_73477

-- Definitions based on the conditions
def A_work_days : ℕ := 20
def B_work_days : ℕ := 30
def C_work_days : ℕ := 10  -- Twice as fast as A, and A can do it in 20 days, hence 10 days.
def together_work_days : ℕ := 12
def B_C_half_day_rate : ℚ := (1 / B_work_days) / 2 + (1 / C_work_days) / 2  -- rate per half day for both B and C
def A_full_day_rate : ℚ := 1 / A_work_days  -- rate per full day for A

-- Converting to rate per day when B and C work only half day daily
def combined_rate_per_day_with_BC_half : ℚ := A_full_day_rate + B_C_half_day_rate

-- The main theorem to prove
theorem work_completion_days 
  (A_work_days B_work_days C_work_days together_work_days : ℕ)
  (C_work_days_def : C_work_days = A_work_days / 2) 
  (total_days_def : 1 / combined_rate_per_day_with_BC_half = 60 / 7) :
  (1 / combined_rate_per_day_with_BC_half) = 60 / 7 :=
sorry

end NUMINAMATH_GPT_work_completion_days_l734_73477


namespace NUMINAMATH_GPT_prove_a_ge_neg_one_fourth_l734_73417

-- Lean 4 statement to reflect the problem
theorem prove_a_ge_neg_one_fourth
  (x y z a : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h1 : x * y - z = a)
  (h2 : y * z - x = a)
  (h3 : z * x - y = a) :
  a ≥ - (1 / 4) :=
sorry

end NUMINAMATH_GPT_prove_a_ge_neg_one_fourth_l734_73417


namespace NUMINAMATH_GPT_increasing_function_range_b_l734_73441

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then (b - 3 / 2) * x + b - 1 else -x^2 + (2 - b) * x

theorem increasing_function_range_b :
  (∀ x y, x < y → f b x ≤ f b y) ↔ (3 / 2 < b ∧ b ≤ 2 ) := 
by
  sorry

end NUMINAMATH_GPT_increasing_function_range_b_l734_73441


namespace NUMINAMATH_GPT_min_value_of_sequence_l734_73486

variable (b1 b2 b3 : ℝ)

def geometric_sequence (b1 b2 b3 : ℝ) : Prop :=
  ∃ s : ℝ, b2 = b1 * s ∧ b3 = b1 * s^2 

theorem min_value_of_sequence (h1 : b1 = 2) (h2 : geometric_sequence b1 b2 b3) :
  ∃ s : ℝ, 3 * b2 + 4 * b3 = -9 / 8 :=
sorry

end NUMINAMATH_GPT_min_value_of_sequence_l734_73486


namespace NUMINAMATH_GPT_find_r_l734_73403

theorem find_r (r : ℝ) (AB AD BD : ℝ) (circle_radius : ℝ) (main_circle_radius : ℝ) :
  main_circle_radius = 2 →
  circle_radius = r →
  AB = 2 * r →
  AD = 2 * r →
  BD = 4 + 2 * r →
  (2 * r)^2 + (2 * r)^2 = (4 + 2 * r)^2 →
  r = 4 :=
by 
  intros h_main_radius h_circle_radius h_AB h_AD h_BD h_pythagorean
  sorry

end NUMINAMATH_GPT_find_r_l734_73403


namespace NUMINAMATH_GPT_marbles_in_jar_l734_73482

theorem marbles_in_jar (M : ℕ) (h1 : M / 24 = 24 * 26 / 26) (h2 : M / 26 + 1 = M / 24) : M = 312 := by
  sorry

end NUMINAMATH_GPT_marbles_in_jar_l734_73482


namespace NUMINAMATH_GPT_ms_brown_expects_8100_tulips_l734_73416

def steps_length := 3
def width_steps := 18
def height_steps := 25
def tulips_per_sqft := 2

def width_feet := width_steps * steps_length
def height_feet := height_steps * steps_length
def area_feet := width_feet * height_feet
def expected_tulips := area_feet * tulips_per_sqft

theorem ms_brown_expects_8100_tulips :
  expected_tulips = 8100 := by
  sorry

end NUMINAMATH_GPT_ms_brown_expects_8100_tulips_l734_73416


namespace NUMINAMATH_GPT_slopes_of_line_intersecting_ellipse_l734_73424

theorem slopes_of_line_intersecting_ellipse (m : ℝ) : 
  (m ∈ Set.Iic (-1 / Real.sqrt 624) ∨ m ∈ Set.Ici (1 / Real.sqrt 624)) ↔
  ∃ x y, y = m * x + 10 ∧ 4 * x^2 + 25 * y^2 = 100 :=
by
  sorry

end NUMINAMATH_GPT_slopes_of_line_intersecting_ellipse_l734_73424


namespace NUMINAMATH_GPT_more_people_attended_l734_73436

def saturday_attendance := 80
def monday_attendance := saturday_attendance - 20
def wednesday_attendance := monday_attendance + 50
def friday_attendance := saturday_attendance + monday_attendance
def expected_audience := 350

theorem more_people_attended :
  saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance - expected_audience = 40 :=
by
  sorry

end NUMINAMATH_GPT_more_people_attended_l734_73436


namespace NUMINAMATH_GPT_field_length_l734_73445

theorem field_length (w l : ℕ) (Pond_Area : ℕ) (Pond_Field_Ratio : ℚ) (Field_Length_Ratio : ℕ) 
  (h1 : Length = 2 * Width)
  (h2 : Pond_Area = 8 * 8)
  (h3 : Pond_Field_Ratio = 1 / 50)
  (h4 : Pond_Area = Pond_Field_Ratio * Field_Area)
  : l = 80 := 
by
  -- begin solution
  sorry

end NUMINAMATH_GPT_field_length_l734_73445


namespace NUMINAMATH_GPT_evaluate_expression_l734_73423

theorem evaluate_expression : 
  (3 / 20 - 5 / 200 + 7 / 2000 : ℚ) = 0.1285 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l734_73423


namespace NUMINAMATH_GPT_folded_strip_fit_l734_73481

open Classical

noncomputable def canFitAfterFolding (r : ℝ) (strip : Set (ℝ × ℝ)) (folded_strip : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ folded_strip → (p.1^2 + p.2^2 ≤ r^2)

theorem folded_strip_fit {r : ℝ} {strip folded_strip : Set (ℝ × ℝ)} :
  (∀ p : ℝ × ℝ, p ∈ strip → (p.1^2 + p.2^2 ≤ r^2)) →
  (∀ q : ℝ × ℝ, q ∈ folded_strip → (∃ p : ℝ × ℝ, p ∈ strip ∧ q = p)) →
  canFitAfterFolding r strip folded_strip :=
by
  intros hs hf
  sorry

end NUMINAMATH_GPT_folded_strip_fit_l734_73481


namespace NUMINAMATH_GPT_initial_pieces_l734_73431

-- Definitions based on given conditions
variable (left : ℕ) (used : ℕ)
axiom cond1 : left = 93
axiom cond2 : used = 4

-- The mathematical proof problem statement
theorem initial_pieces (left used : ℕ) (cond1 : left = 93) (cond2 : used = 4) : left + used = 97 :=
by
  sorry

end NUMINAMATH_GPT_initial_pieces_l734_73431


namespace NUMINAMATH_GPT_find_xyz_l734_73425

theorem find_xyz (x y z : ℝ)
  (h1 : x > 4)
  (h2 : y > 4)
  (h3 : z > 4)
  (h4 : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 42) :
  (x, y, z) = (11, 9, 7) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_xyz_l734_73425


namespace NUMINAMATH_GPT_correct_total_distance_l734_73406

theorem correct_total_distance (km_to_m : 3.5 * 1000 = 3500) (add_m : 3500 + 200 = 3700) : 
  3.5 * 1000 + 200 = 3700 :=
by
  -- The proof would be filled here.
  sorry

end NUMINAMATH_GPT_correct_total_distance_l734_73406


namespace NUMINAMATH_GPT_find_positive_integers_satisfying_inequality_l734_73434

theorem find_positive_integers_satisfying_inequality :
  (∃ n : ℕ, (n - 1) * (n - 3) * (n - 5) * (n - 7) * (n - 9) * (n - 11) * (n - 13) * (n - 15) *
    (n - 17) * (n - 19) * (n - 21) * (n - 23) * (n - 25) * (n - 27) * (n - 29) * (n - 31) *
    (n - 33) * (n - 35) * (n - 37) * (n - 39) * (n - 41) * (n - 43) * (n - 45) * (n - 47) *
    (n - 49) * (n - 51) * (n - 53) * (n - 55) * (n - 57) * (n - 59) * (n - 61) * (n - 63) *
    (n - 65) * (n - 67) * (n - 69) * (n - 71) * (n - 73) * (n - 75) * (n - 77) * (n - 79) *
    (n - 81) * (n - 83) * (n - 85) * (n - 87) * (n - 89) * (n - 91) * (n - 93) * (n - 95) *
    (n - 97) * (n - 99) < 0 ∧ 1 ≤ n ∧ n ≤ 99) 
  → ∃ f : ℕ → ℕ, (∀ i, f i = 2 + 4 * i) ∧ (∀ i, 1 ≤ f i ∧ f i ≤ 24) :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integers_satisfying_inequality_l734_73434


namespace NUMINAMATH_GPT_password_lock_probability_l734_73488

theorem password_lock_probability :
  (∃ (A B C : ℕ), 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 
   (∃ password : ℕ × ℕ × ℕ, password = (A, B, C) ∧ 
   (∃ chosen_middle_digit : ℕ, 0 <= chosen_middle_digit ∧ chosen_middle_digit <= 9 ∧
   chosen_middle_digit = B))) →
  (1/10 : ℝ) = 0.1 :=
by 
  sorry

end NUMINAMATH_GPT_password_lock_probability_l734_73488


namespace NUMINAMATH_GPT_find_a_n_l734_73478

theorem find_a_n (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n, S n = 3^n + 2) :
  ∀ n, a n = if n = 1 then 5 else 2 * 3^(n - 1) := by
  sorry

end NUMINAMATH_GPT_find_a_n_l734_73478


namespace NUMINAMATH_GPT_coordinates_C_on_segment_AB_l734_73453

theorem coordinates_C_on_segment_AB :
  ∃ C : (ℝ × ℝ), 
  (C.1 = 2 ∧ C.2 = 6) ∧
  ∃ A B : (ℝ × ℝ), 
  (A = (-1, 0)) ∧ 
  (B = (3, 8)) ∧ 
  (∃ k : ℝ, (k = 3) ∧ dist (C) (A) = k * dist (C) (B)) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_C_on_segment_AB_l734_73453


namespace NUMINAMATH_GPT_income_remaining_percentage_l734_73415

theorem income_remaining_percentage :
  let initial_income := 100
  let food_percentage := 42
  let education_percentage := 18
  let transportation_percentage := 12
  let house_rent_percentage := 55
  let total_spent := food_percentage + education_percentage + transportation_percentage
  let remaining_after_expenses := initial_income - total_spent
  let house_rent_amount := (house_rent_percentage * remaining_after_expenses) / 100
  let final_remaining_income := remaining_after_expenses - house_rent_amount
  final_remaining_income = 12.6 :=
by
  sorry

end NUMINAMATH_GPT_income_remaining_percentage_l734_73415


namespace NUMINAMATH_GPT_reflected_curve_equation_l734_73407

-- Define the original curve equation
def original_curve (x y : ℝ) : Prop :=
  2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0

-- Define the line of reflection
def line_of_reflection (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

-- Define the equation of the reflected curve
def reflected_curve (x y : ℝ) : Prop :=
  146 * x^2 - 44 * x * y + 29 * y^2 + 152 * x - 64 * y - 494 = 0

-- Problem: Prove the equation of the reflected curve is as given
theorem reflected_curve_equation (x y : ℝ) :
  (∃ x1 y1 : ℝ, original_curve x1 y1 ∧ line_of_reflection x1 y1 ∧ (x, y) = (x1, y1)) →
  reflected_curve x y :=
by
  intros
  sorry

end NUMINAMATH_GPT_reflected_curve_equation_l734_73407


namespace NUMINAMATH_GPT_total_cost_of_hats_l734_73468

-- Definition of conditions
def weeks := 2
def days_per_week := 7
def cost_per_hat := 50

-- Definition of the number of hats
def num_hats := weeks * days_per_week

-- Statement of the problem
theorem total_cost_of_hats : num_hats * cost_per_hat = 700 := 
by sorry

end NUMINAMATH_GPT_total_cost_of_hats_l734_73468


namespace NUMINAMATH_GPT_range_of_x_l734_73461

def f (x a : ℝ) : ℝ := x^3 + 3 * a * x - 1

def g (x a : ℝ) : ℝ := 3 * x^2 - a * x + 3 * a - 5

def condition (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1

theorem range_of_x (x a : ℝ) (h : condition a) : g x a < 0 → -2/3 < x ∧ x < 1 := 
sorry

end NUMINAMATH_GPT_range_of_x_l734_73461


namespace NUMINAMATH_GPT_tangent_line_equation_range_of_k_l734_73491

noncomputable def f (x : ℝ) : ℝ := x^2 - x * Real.log x

-- Part (I): Tangent line equation
theorem tangent_line_equation :
  let f (x : ℝ) := x^2 - x * Real.log x
  let p := (1 : ℝ)
  let y := f p
  (∀ x, y = x) :=
sorry

-- Part (II): Range of k
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → (k / x + x / 2 - f x / x < 0)) → k ≤ 1 / 2 :=
sorry

end NUMINAMATH_GPT_tangent_line_equation_range_of_k_l734_73491


namespace NUMINAMATH_GPT_apples_problem_l734_73480

theorem apples_problem :
  ∃ (jackie rebecca : ℕ), (rebecca = 2 * jackie) ∧ (∃ (adam : ℕ), (adam = jackie + 3) ∧ (adam = 9) ∧ jackie = 6 ∧ rebecca = 12) :=
by
  sorry

end NUMINAMATH_GPT_apples_problem_l734_73480


namespace NUMINAMATH_GPT_bacteria_growth_rate_l734_73472

theorem bacteria_growth_rate
  (r : ℝ) 
  (h1 : ∃ B D : ℝ, B * r^30 = D) 
  (h2 : ∃ B D : ℝ, B * r^25 = D / 32) :
  r = 2 := 
by 
  sorry

end NUMINAMATH_GPT_bacteria_growth_rate_l734_73472


namespace NUMINAMATH_GPT_coprime_exist_m_n_l734_73467

theorem coprime_exist_m_n (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_a : a ≥ 1) (h_b : b ≥ 1) :
  ∃ (m n : ℕ), m ≥ 1 ∧ n ≥ 1 ∧ a^m + b^n ≡ 1 [MOD a * b] :=
by
  use Nat.totient b, Nat.totient a
  sorry

end NUMINAMATH_GPT_coprime_exist_m_n_l734_73467


namespace NUMINAMATH_GPT_boxes_left_l734_73499

theorem boxes_left (boxes_sat : ℕ) (boxes_sun : ℕ) (apples_per_box : ℕ) (apples_sold : ℕ)
  (h1 : boxes_sat = 50) (h2 : boxes_sun = 25) (h3 : apples_per_box = 10) (h4 : apples_sold = 720) :
  (boxes_sat * apples_per_box + boxes_sun * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end NUMINAMATH_GPT_boxes_left_l734_73499


namespace NUMINAMATH_GPT_solution_l734_73420

-- Define the conditions based on the given problem
variables {A B C D : Type}
variables {AB BC CD DA : ℝ} (h1 : AB = 65) (h2 : BC = 105) (h3 : CD = 125) (h4 : DA = 95)
variables (cy_in_circle : CyclicQuadrilateral A B C D)
variables (circ_inscribed : TangentialQuadrilateral A B C D)

-- Function that computes the absolute difference between segments x and y on side of length CD
noncomputable def find_absolute_difference (x y : ℝ) (h5 : x + y = 125) : ℝ := |x - y|

-- The proof statement
theorem solution :
  ∃ (x y : ℝ), x + y = 125 ∧
  (find_absolute_difference x y (by sorry) = 14) := sorry

end NUMINAMATH_GPT_solution_l734_73420


namespace NUMINAMATH_GPT_solve_system_l734_73484

theorem solve_system : 
  ∀ (a b c : ℝ), 
  (a * (b^2 + c) = c * (c + a * b) ∧ 
   b * (c^2 + a) = a * (a + b * c) ∧ 
   c * (a^2 + b) = b * (b + c * a)) 
   → (∃ t : ℝ, a = t ∧ b = t ∧ c = t) :=
by
  intros a b c h
  sorry

end NUMINAMATH_GPT_solve_system_l734_73484


namespace NUMINAMATH_GPT_domain_of_log2_function_l734_73432

theorem domain_of_log2_function :
  {x : ℝ | 2 * x - 1 > 0} = {x : ℝ | x > 1 / 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_log2_function_l734_73432


namespace NUMINAMATH_GPT_number_of_cars_l734_73476

theorem number_of_cars (n s t C : ℕ) (h1 : n = 9) (h2 : s = 4) (h3 : t = 3) (h4 : n * s = t * C) : C = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cars_l734_73476


namespace NUMINAMATH_GPT_y_relationship_l734_73494

theorem y_relationship (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1: y1 = -4 / x1) (h2: y2 = -4 / x2) (h3: y3 = -4 / x3)
  (h4: x1 < 0) (h5: 0 < x2) (h6: x2 < x3) :
  y1 > y3 ∧ y3 > y2 :=
by
  sorry

end NUMINAMATH_GPT_y_relationship_l734_73494


namespace NUMINAMATH_GPT_circle_radius_l734_73433

theorem circle_radius (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y + 1 = 0) : 
    ∃ r : ℝ, r = 2 ∧ (x - 2)^2 + (y - 1)^2 = r^2 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l734_73433


namespace NUMINAMATH_GPT_trade_in_value_of_old_phone_l734_73454

-- Define the given conditions
def cost_of_iphone : ℕ := 800
def earnings_per_week : ℕ := 80
def weeks_worked : ℕ := 7

-- Define the total earnings from babysitting
def total_earnings : ℕ := earnings_per_week * weeks_worked

-- Define the final proof statement
theorem trade_in_value_of_old_phone : cost_of_iphone - total_earnings = 240 :=
by
  unfold cost_of_iphone
  unfold total_earnings
  -- Substitute in the values
  have h1 : 800 - (80 * 7) = 240 := sorry
  exact h1

end NUMINAMATH_GPT_trade_in_value_of_old_phone_l734_73454


namespace NUMINAMATH_GPT_rectangle_width_eq_six_l734_73439

theorem rectangle_width_eq_six (w : ℝ) :
  ∃ w, (3 * w = 25 - 7) ↔ w = 6 :=
by
  -- Given the conditions as stated:
  -- Length of the rectangle: 3 inches
  -- Width of the square: 5 inches
  -- Difference in area between the square and the rectangle: 7 square inches
  -- We can show that the width of the rectangle is 6 inches.
  sorry

end NUMINAMATH_GPT_rectangle_width_eq_six_l734_73439


namespace NUMINAMATH_GPT_cupboard_cost_price_l734_73448

theorem cupboard_cost_price (C SP NSP : ℝ) (h1 : SP = 0.84 * C) (h2 : NSP = 1.16 * C) (h3 : NSP = SP + 1200) : C = 3750 :=
by
  sorry

end NUMINAMATH_GPT_cupboard_cost_price_l734_73448


namespace NUMINAMATH_GPT_lines_perpendicular_to_same_plane_are_parallel_l734_73438

variables {Point Line Plane : Type*}
variables [MetricSpace Point] [LinearOrder Line]

def line_parallel_to_plane (a : Line) (M : Plane) : Prop := sorry -- Define the formal condition
def line_perpendicular_to_plane (a : Line) (M : Plane) : Prop := sorry -- Define the formal condition
def lines_parallel (a b : Line) : Prop := sorry -- Define the formal condition

theorem lines_perpendicular_to_same_plane_are_parallel 
  (a b : Line) (M : Plane) 
  (h₁ : line_perpendicular_to_plane a M) 
  (h₂ : line_perpendicular_to_plane b M) : 
  lines_parallel a b :=
sorry

end NUMINAMATH_GPT_lines_perpendicular_to_same_plane_are_parallel_l734_73438


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l734_73442

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l734_73442


namespace NUMINAMATH_GPT_f_of_f_five_l734_73428

noncomputable def f : ℝ → ℝ := sorry

axiom f_periodicity (x : ℝ) : f (x + 2) = 1 / f x
axiom f_initial_value : f 1 = -5

theorem f_of_f_five : f (f 5) = -1 / 5 :=
by sorry

end NUMINAMATH_GPT_f_of_f_five_l734_73428


namespace NUMINAMATH_GPT_mul_exponents_l734_73427

theorem mul_exponents (m : ℝ) : 2 * m^3 * 3 * m^4 = 6 * m^7 :=
by sorry

end NUMINAMATH_GPT_mul_exponents_l734_73427


namespace NUMINAMATH_GPT_reciprocal_of_neg3_l734_73456

theorem reciprocal_of_neg3 : 1 / (-3 : ℝ) = - (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg3_l734_73456


namespace NUMINAMATH_GPT_mean_of_numbers_is_10_l734_73411

-- Define the list of numbers
def numbers : List ℕ := [6, 8, 9, 11, 16]

-- Define the length of the list
def n : ℕ := numbers.length

-- Define the sum of the list
def sum_numbers : ℕ := numbers.sum

-- Define the mean (average) calculation for the list
def average : ℕ := sum_numbers / n

-- Prove that the mean of the list is 10
theorem mean_of_numbers_is_10 : average = 10 := by
  sorry

end NUMINAMATH_GPT_mean_of_numbers_is_10_l734_73411


namespace NUMINAMATH_GPT_problem_l734_73409

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem problem (a b c : ℝ) (h0 : f a b c 0 = f a b c 4) (h1 : f a b c 0 > f a b c 1) :
  a > 0 ∧ 4 * a + b = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_l734_73409


namespace NUMINAMATH_GPT_compute_pqr_l734_73457

theorem compute_pqr
  (p q r : ℤ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (h_sum : p + q + r = 30)
  (h_eq : 1 / p + 1 / q + 1 / r + 240 / (p * q * r) = 1) :
  p * q * r = 1080 := by
  sorry

end NUMINAMATH_GPT_compute_pqr_l734_73457


namespace NUMINAMATH_GPT_trapezoid_area_l734_73462

noncomputable def area_trapezoid (B1 B2 h : ℝ) : ℝ := (1 / 2 * (B1 + B2) * h)

theorem trapezoid_area
    (h1 : ∀ x : ℝ, 3 * x = 10 → x = 10 / 3)
    (h2 : ∀ x : ℝ, 3 * x = 5 → x = 5 / 3)
    (h3 : B1 = 10 / 3)
    (h4 : B2 = 5 / 3)
    (h5 : h = 5)
    : area_trapezoid B1 B2 h = 12.5 := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l734_73462


namespace NUMINAMATH_GPT_rita_bought_4_jackets_l734_73490

/-
Given:
  - Rita bought 5 short dresses costing $20 each.
  - Rita bought 3 pairs of pants costing $12 each.
  - The jackets cost $30 each.
  - She spent an additional $5 on transportation.
  - Rita had $400 initially.
  - Rita now has $139.

Prove that the number of jackets Rita bought is 4.
-/

theorem rita_bought_4_jackets :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let transportation_cost := 5
  let initial_amount := 400
  let remaining_amount := 139
  let jackets_cost_per_unit := 30
  let total_spent := initial_amount - remaining_amount
  let total_clothes_transportation_cost := dresses_cost + pants_cost + transportation_cost
  let jackets_cost := total_spent - total_clothes_transportation_cost
  let number_of_jackets := jackets_cost / jackets_cost_per_unit
  number_of_jackets = 4 :=
by
  sorry

end NUMINAMATH_GPT_rita_bought_4_jackets_l734_73490


namespace NUMINAMATH_GPT_smallest_positive_period_of_h_l734_73421

-- Definitions of f and g with period 1
axiom f : ℝ → ℝ
axiom g : ℝ → ℝ
axiom T1 : ℝ
axiom T2 : ℝ

-- Given conditions
@[simp] axiom f_periodic : ∀ x, f (x + T1) = f x
@[simp] axiom g_periodic : ∀ x, g (x + T2) = g x
@[simp] axiom T1_eq_one : T1 = 1
@[simp] axiom T2_eq_one : T2 = 1

-- Statement to prove the smallest positive period of h(x) = f(x) + g(x) is 1/k
theorem smallest_positive_period_of_h (k : ℕ) (h : ℝ → ℝ) (hk: k > 0) :
  (∀ x, h (x + 1) = h x) →
  (∀ T > 0, (∀ x, h (x + T) = h x) → (∃ k : ℕ, T = 1 / k)) :=
by sorry

end NUMINAMATH_GPT_smallest_positive_period_of_h_l734_73421


namespace NUMINAMATH_GPT_part_i_part_ii_l734_73450

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - 2 * a) + abs (x - a)

-- Part I: Prove solution to the inequality.
theorem part_i (x : ℝ) : f x 1 > 3 ↔ x ∈ {x | x < 0} ∪ {x | x > 3} :=
sorry

-- Part II: Prove the inequality for general a and b with condition for equality.
theorem part_ii (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  f b a ≥ f a a ∧ ((2 * a - b = 0 ∨ b - a = 0) ∨ (2 * a - b > 0 ∧ b - a > 0) ∨ (2 * a - b < 0 ∧ b - a < 0)) ↔ f b a = f a a :=
sorry

end NUMINAMATH_GPT_part_i_part_ii_l734_73450


namespace NUMINAMATH_GPT_math_problem_l734_73414

theorem math_problem : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end NUMINAMATH_GPT_math_problem_l734_73414


namespace NUMINAMATH_GPT_football_team_practiced_hours_l734_73410

-- Define the daily practice hours and missed days as conditions
def daily_practice_hours : ℕ := 6
def missed_days : ℕ := 1

-- Define the total number of days in a week
def days_in_week : ℕ := 7

-- Define a function to calculate the total practiced hours in a week, 
-- given the daily practice hours, missed days, and total days in a week
def total_practiced_hours (daily_hours : ℕ) (missed : ℕ) (total_days : ℕ) : ℕ :=
  (total_days - missed) * daily_hours

-- Prove that the total practiced hours is 36
theorem football_team_practiced_hours :
  total_practiced_hours daily_practice_hours missed_days days_in_week = 36 := 
sorry

end NUMINAMATH_GPT_football_team_practiced_hours_l734_73410


namespace NUMINAMATH_GPT_sum_six_consecutive_integers_l734_73458

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_six_consecutive_integers_l734_73458


namespace NUMINAMATH_GPT_millet_exceeds_half_l734_73473

noncomputable def seeds_millet_day (n : ℕ) : ℝ :=
  0.2 * (1 - 0.7 ^ n) / (1 - 0.7) + 0.2 * 0.7 ^ n

noncomputable def seeds_other_day (n : ℕ) : ℝ :=
  0.3 * (1 - 0.1 ^ n) / (1 - 0.1) + 0.3 * 0.1 ^ n

noncomputable def prop_millet (n : ℕ) : ℝ :=
  seeds_millet_day n / (seeds_millet_day n + seeds_other_day n)

theorem millet_exceeds_half : ∃ n : ℕ, prop_millet n > 0.5 ∧ n = 3 :=
by sorry

end NUMINAMATH_GPT_millet_exceeds_half_l734_73473


namespace NUMINAMATH_GPT_mass_percentage_Ba_in_BaI2_l734_73496

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_BaI2 : ℝ := molar_mass_Ba + 2 * molar_mass_I

theorem mass_percentage_Ba_in_BaI2 : 
  (molar_mass_Ba / molar_mass_BaI2) * 100 = 35.11 := 
  by 
    -- implementing the proof here would demonstrate that (137.33 / 391.13) * 100 = 35.11
    sorry

end NUMINAMATH_GPT_mass_percentage_Ba_in_BaI2_l734_73496


namespace NUMINAMATH_GPT_relay_race_length_correct_l734_73429

def relay_race_length (num_members distance_per_member : ℕ) : ℕ := num_members * distance_per_member

theorem relay_race_length_correct :
  relay_race_length 5 30 = 150 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_relay_race_length_correct_l734_73429


namespace NUMINAMATH_GPT_solve_for_x_l734_73419

theorem solve_for_x (x : ℝ) :
  (x - 5)^4 = (1/16)⁻¹ → x = 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l734_73419
