import Mathlib

namespace solve_quadratic_inequality_l1098_109802

theorem solve_quadratic_inequality (a x : ℝ) (h : a < 1) : 
  x^2 - (a + 1) * x + a < 0 ↔ (a < x ∧ x < 1) :=
by
  sorry

end solve_quadratic_inequality_l1098_109802


namespace smallest_w_for_factors_l1098_109829

theorem smallest_w_for_factors (w : ℕ) (h_pos : 0 < w) :
  (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (13^2 ∣ 936 * w) ↔ w = 156 := 
sorry

end smallest_w_for_factors_l1098_109829


namespace greatest_integer_less_than_or_equal_to_frac_l1098_109853

theorem greatest_integer_less_than_or_equal_to_frac (a b c d : ℝ)
  (ha : a = 4^100) (hb : b = 3^100) (hc : c = 4^95) (hd : d = 3^95) :
  ⌊(a + b) / (c + d)⌋ = 1023 := 
by
  sorry

end greatest_integer_less_than_or_equal_to_frac_l1098_109853


namespace card_game_final_amounts_l1098_109875

theorem card_game_final_amounts
  (T : ℝ)
  (aldo_initial_ratio : ℝ := 7)
  (bernardo_initial_ratio : ℝ := 6)
  (carlos_initial_ratio : ℝ := 5)
  (aldo_final_ratio : ℝ := 6)
  (bernardo_final_ratio : ℝ := 5)
  (carlos_final_ratio : ℝ := 4)
  (aldo_won : ℝ := 1200) :
  aldo_won = (1 / 90) * T →
  T = 108000 →
  (36 / 90) * T = 43200 ∧ (30 / 90) * T = 36000 ∧ (24 / 90) * T = 28800 := sorry

end card_game_final_amounts_l1098_109875


namespace coeff_of_z_in_eq2_l1098_109800

-- Definitions of the conditions from part a)
def equation1 (x y z : ℤ) := 6 * x - 5 * y + 3 * z = 22
def equation2 (x y z : ℤ) := 4 * x + 8 * y - z = (7 : ℚ) / 11
def equation3 (x y z : ℤ) := 5 * x - 6 * y + 2 * z = 12
def sum_xyz (x y z : ℤ) := x + y + z = 10

-- Theorem stating that the coefficient of z in equation 2 is -1.
theorem coeff_of_z_in_eq2 {x y z : ℤ} (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) (h4 : sum_xyz x y z) :
    -1 = -1 :=
by
  -- This is a placeholder for the proof.
  sorry

end coeff_of_z_in_eq2_l1098_109800


namespace shopper_saved_percentage_l1098_109892

theorem shopper_saved_percentage (amount_paid : ℝ) (amount_saved : ℝ) (original_price : ℝ)
  (h1 : amount_paid = 45) (h2 : amount_saved = 5) (h3 : original_price = amount_paid + amount_saved) :
  (amount_saved / original_price) * 100 = 10 :=
by
  -- The proof is omitted
  sorry

end shopper_saved_percentage_l1098_109892


namespace hundredth_ring_square_count_l1098_109857

-- Conditions
def center_rectangle : ℤ × ℤ := (1, 2)
def first_ring_square_count : ℕ := 10
def square_count_nth_ring (n : ℕ) : ℕ := 8 * n + 2

-- Problem Statement
theorem hundredth_ring_square_count : square_count_nth_ring 100 = 802 := 
  sorry

end hundredth_ring_square_count_l1098_109857


namespace age_ratio_l1098_109840

theorem age_ratio (x : ℕ) (h : (5 * x - 4) = (3 * x + 4)) :
    (5 * x + 4) / (3 * x - 4) = 3 :=
by sorry

end age_ratio_l1098_109840


namespace general_term_formula_of_arithmetic_seq_l1098_109801

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem general_term_formula_of_arithmetic_seq 
  (a : ℕ → ℝ) (h_arith : arithmetic_seq a)
  (h1 : a 3 * a 7 = -16) 
  (h2 : a 4 + a 6 = 0) :
  (∀ n : ℕ, a n = 2 * n - 10) ∨ (∀ n : ℕ, a n = -2 * n + 10) :=
by
  sorry

end general_term_formula_of_arithmetic_seq_l1098_109801


namespace factory_A_higher_output_l1098_109880

theorem factory_A_higher_output (a x : ℝ) (a_pos : a > 0) (x_pos : x > 0) 
  (h_eq_march : 1 + 2 * a = (1 + x) ^ 2) : 
  1 + a > 1 + x :=
by
  sorry

end factory_A_higher_output_l1098_109880


namespace speed_of_car_B_is_correct_l1098_109861

def carB_speed : ℕ := 
  let speedA := 50 -- Car A's speed in km/hr
  let timeA := 6 -- Car A's travel time in hours
  let ratio := 3 -- The ratio of distances between Car A and Car B
  let distanceA := speedA * timeA -- Calculate Car A's distance
  let timeB := 1 -- Car B's travel time in hours
  let distanceB := distanceA / ratio -- Calculate Car B's distance
  distanceB / timeB -- Calculate Car B's speed

theorem speed_of_car_B_is_correct : carB_speed = 100 := by
  sorry

end speed_of_car_B_is_correct_l1098_109861


namespace intersection_of_M_and_N_l1098_109806

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_of_M_and_N_l1098_109806


namespace division_remainder_l1098_109899

theorem division_remainder :
  let p := fun x : ℝ => 5 * x^4 - 9 * x^3 + 3 * x^2 - 7 * x - 30
  let q := 3 * x - 9
  p 3 % q = 138 :=
by
  sorry

end division_remainder_l1098_109899


namespace sam_more_than_sarah_l1098_109838

-- Defining the conditions
def street_width : ℤ := 25
def block_length : ℤ := 450
def block_width : ℤ := 350
def alleyway : ℤ := 25

-- Defining the distances run by Sarah and Sam
def sarah_long_side : ℤ := block_length + alleyway
def sarah_short_side : ℤ := block_width
def sam_long_side : ℤ := block_length + 2 * street_width
def sam_short_side : ℤ := block_width + 2 * street_width

-- Defining the total distance run by Sarah and Sam in one lap
def sarah_total_distance : ℤ := 2 * sarah_long_side + 2 * sarah_short_side
def sam_total_distance : ℤ := 2 * sam_long_side + 2 * sam_short_side

-- Proving the difference between Sam's and Sarah's running distances
theorem sam_more_than_sarah : sam_total_distance - sarah_total_distance = 150 := by
  -- The proof is omitted
  sorry

end sam_more_than_sarah_l1098_109838


namespace inverse_sum_l1098_109887

def f (x : ℝ) : ℝ := x * |x|

theorem inverse_sum (h1 : ∃ x : ℝ, f x = 9) (h2 : ∃ x : ℝ, f x = -81) :
  ∃ a b: ℝ, f a = 9 ∧ f b = -81 ∧ a + b = -6 :=
by
  sorry

end inverse_sum_l1098_109887


namespace median_interval_60_64_l1098_109811

theorem median_interval_60_64 
  (students : ℕ) 
  (f_45_49 f_50_54 f_55_59 f_60_64 : ℕ) :
  students = 105 ∧ 
  f_45_49 = 8 ∧ 
  f_50_54 = 15 ∧ 
  f_55_59 = 20 ∧ 
  f_60_64 = 18 ∧ 
  (8 + 15 + 20 + 18) ≥ (105 + 1) / 2
  → 60 ≤ (105 + 1) / 2  ∧ (105 + 1) / 2 ≤ 64 :=
sorry

end median_interval_60_64_l1098_109811


namespace kim_saplings_left_l1098_109855

def sprouted_pits (total_pits num_sprouted_pits: ℕ) (percent_sprouted: ℝ) : Prop :=
  percent_sprouted * total_pits = num_sprouted_pits

def sold_saplings (total_saplings saplings_sold saplings_left: ℕ) : Prop :=
  total_saplings - saplings_sold = saplings_left

theorem kim_saplings_left
  (total_pits : ℕ) (num_sprouted_pits : ℕ) (percent_sprouted : ℝ)
  (saplings_sold : ℕ) (saplings_left : ℕ) :
  total_pits = 80 →
  percent_sprouted = 0.25 →
  saplings_sold = 6 →
  sprouted_pits total_pits num_sprouted_pits percent_sprouted →
  sold_saplings num_sprouted_pits saplings_sold saplings_left →
  saplings_left = 14 :=
by
  intros
  sorry

end kim_saplings_left_l1098_109855


namespace find_f_neg2_l1098_109822

theorem find_f_neg2 (a b : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = x^5 + a*x^3 + x^2 + b*x + 2) (h₂ : f 2 = 3) : f (-2) = 9 :=
by
  sorry

end find_f_neg2_l1098_109822


namespace crayons_total_l1098_109836

def crayons_per_child := 6
def number_of_children := 12
def total_crayons := 72

theorem crayons_total :
  crayons_per_child * number_of_children = total_crayons := by
  sorry

end crayons_total_l1098_109836


namespace possible_measures_A_l1098_109886

open Nat

theorem possible_measures_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k > 0 ∧ A = k * B) ∧ A + B = 180 ∧ (∃! n : ℕ, n = 17) :=
by
  sorry

end possible_measures_A_l1098_109886


namespace john_newspaper_percentage_less_l1098_109897

theorem john_newspaper_percentage_less
  (total_newspapers : ℕ)
  (selling_price : ℝ)
  (percentage_sold : ℝ)
  (profit : ℝ)
  (total_cost : ℝ)
  (cost_per_newspaper : ℝ)
  (percentage_less : ℝ)
  (h1 : total_newspapers = 500)
  (h2 : selling_price = 2)
  (h3 : percentage_sold = 0.80)
  (h4 : profit = 550)
  (h5 : total_cost = 800 - profit)
  (h6 : cost_per_newspaper = total_cost / total_newspapers)
  (h7 : percentage_less = ((selling_price - cost_per_newspaper) / selling_price) * 100) :
  percentage_less = 75 :=
by
  sorry

end john_newspaper_percentage_less_l1098_109897


namespace prop3_prop4_l1098_109820

-- Definitions to represent planes and lines
variable (Plane Line : Type)

-- Predicate representing parallel planes or lines
variable (parallel : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Predicate representing perpendicular planes or lines
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Distinct planes and a line
variables (α β γ : Plane) (l : Line)

-- Proposition 3: If l ⊥ α and l ∥ β, then α ⊥ β
theorem prop3 : perpendicular_line_plane l α ∧ parallel_line_plane l β → perpendicular α β :=
sorry

-- Proposition 4: If α ∥ β and α ⊥ γ, then β ⊥ γ
theorem prop4 : parallel α β ∧ perpendicular α γ → perpendicular β γ :=
sorry

end prop3_prop4_l1098_109820


namespace minimum_a_l1098_109863

def f (x a : ℝ) : ℝ := x^2 - 2*x - abs (x-1-a) - abs (x-2) + 4

theorem minimum_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a = -2 :=
sorry

end minimum_a_l1098_109863


namespace terez_farm_pregnant_cows_percentage_l1098_109810

theorem terez_farm_pregnant_cows_percentage (total_cows : ℕ) (female_percentage : ℕ) (pregnant_females : ℕ) 
  (ht : total_cows = 44) (hf : female_percentage = 50) (hp : pregnant_females = 11) :
  (pregnant_females * 100 / (female_percentage * total_cows / 100) = 50) :=
by 
  sorry

end terez_farm_pregnant_cows_percentage_l1098_109810


namespace excess_percentage_l1098_109814

theorem excess_percentage (A B : ℝ) (x : ℝ) 
  (hA' : A' = A * (1 + x / 100))
  (hB' : B' = B * (1 - 5 / 100))
  (h_area_err : A' * B' = 1.007 * (A * B)) : x = 6 :=
by
  sorry

end excess_percentage_l1098_109814


namespace tan_beta_eq_minus_one_seventh_l1098_109888

theorem tan_beta_eq_minus_one_seventh {α β : ℝ} 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := 
by
  sorry

end tan_beta_eq_minus_one_seventh_l1098_109888


namespace neg_abs_value_eq_neg_three_l1098_109858

theorem neg_abs_value_eq_neg_three : -|-3| = -3 := 
by sorry

end neg_abs_value_eq_neg_three_l1098_109858


namespace fraction_red_knights_magical_l1098_109878

theorem fraction_red_knights_magical (total_knights red_knights blue_knights magical_knights : ℕ)
  (fraction_red fraction_magical : ℚ)
  (frac_red_mag : ℚ) :
  (red_knights = total_knights * fraction_red) →
  (fraction_red = 3 / 8) →
  (magical_knights = total_knights * fraction_magical) →
  (fraction_magical = 1 / 4) →
  (frac_red_mag * red_knights + (frac_red_mag / 3) * blue_knights = magical_knights) →
  (frac_red_mag = 3 / 7) :=
by
  -- Skipping proof
  sorry

end fraction_red_knights_magical_l1098_109878


namespace subtract_eq_l1098_109862

theorem subtract_eq (x y : ℝ) (h1 : 4 * x - 3 * y = 2) (h2 : 4 * x + y = 10) : 4 * y = 8 :=
by
  sorry

end subtract_eq_l1098_109862


namespace evaluate_expression_l1098_109831

theorem evaluate_expression : 150 * (150 - 4) - (150 * 150 - 6 + 2) = -596 :=
by
  sorry

end evaluate_expression_l1098_109831


namespace trajectory_of_Q_is_parabola_l1098_109876

/--
Given a point P (x, y) moves on a unit circle centered at the origin,
prove that the trajectory of point Q (u, v) defined by u = x + y and v = xy 
satisfies u^2 - 2v = 1 and is thus a parabola.
-/
theorem trajectory_of_Q_is_parabola 
  (x y u v : ℝ) 
  (h1 : x^2 + y^2 = 1) 
  (h2 : u = x + y) 
  (h3 : v = x * y) :
  u^2 - 2 * v = 1 :=
sorry

end trajectory_of_Q_is_parabola_l1098_109876


namespace union_P_complement_Q_l1098_109884

-- Define sets P and Q
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

-- Define the complement of Q in ℝ
def C_RQ : Set ℝ := {x | -2 < x ∧ x < 2}

-- State the main theorem
theorem union_P_complement_Q : (P ∪ C_RQ) = {x | -2 < x ∧ x ≤ 3} := 
by
  sorry

end union_P_complement_Q_l1098_109884


namespace total_students_l1098_109841

theorem total_students (N : ℕ) (num_provincial : ℕ) (sample_provincial : ℕ) 
(sample_experimental : ℕ) (sample_regular : ℕ) (sample_sino_canadian : ℕ) 
(ratio : ℕ) 
(h1 : num_provincial = 96) 
(h2 : sample_provincial = 12) 
(h3 : sample_experimental = 21) 
(h4 : sample_regular = 25) 
(h5 : sample_sino_canadian = 43) 
(h6 : ratio = num_provincial / sample_provincial) 
(h7 : ratio = 8) 
: N = ratio * (sample_provincial + sample_experimental + sample_regular + sample_sino_canadian) := 
by 
  sorry

end total_students_l1098_109841


namespace solve_fraction_equation_l1098_109817

theorem solve_fraction_equation :
  ∀ x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) → x = -2 / 11 :=
by
  intro x
  intro h
  sorry

end solve_fraction_equation_l1098_109817


namespace compare_fractions_l1098_109837

theorem compare_fractions : (-1 / 3) > (-1 / 2) := sorry

end compare_fractions_l1098_109837


namespace largest_integer_solution_of_abs_eq_and_inequality_l1098_109865

theorem largest_integer_solution_of_abs_eq_and_inequality : 
  ∃ x : ℤ, |x - 3| = 15 ∧ x ≤ 20 ∧ (∀ y : ℤ, |y - 3| = 15 ∧ y ≤ 20 → y ≤ x) :=
sorry

end largest_integer_solution_of_abs_eq_and_inequality_l1098_109865


namespace tangent_line_at_point_l1098_109809

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 4 * x + 2

def point : ℝ × ℝ := (1, -3)

def tangent_line (x y : ℝ) : Prop := 5 * x + y - 2 = 0

theorem tangent_line_at_point : tangent_line 1 (-3) :=
  sorry

end tangent_line_at_point_l1098_109809


namespace sugar_snap_peas_l1098_109873

theorem sugar_snap_peas (P : ℕ) (h1 : P / 7 = 72 / 9) : P = 56 := 
sorry

end sugar_snap_peas_l1098_109873


namespace find_ab_bc_value_l1098_109896

theorem find_ab_bc_value
  (a b c : ℝ)
  (h : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  (a + b) / (b - c) = -7 := by
sorry

end find_ab_bc_value_l1098_109896


namespace vector_BC_correct_l1098_109860

-- Define the conditions
def vector_AB : ℝ × ℝ := (-3, 2)
def vector_AC : ℝ × ℝ := (1, -2)

-- Define the problem to be proved
theorem vector_BC_correct :
  let vector_BC := (vector_AC.1 - vector_AB.1, vector_AC.2 - vector_AB.2)
  vector_BC = (4, -4) :=
by
  sorry -- The proof is not required, but the structure indicates where it would go

end vector_BC_correct_l1098_109860


namespace system_of_equations_solution_l1098_109824

theorem system_of_equations_solution (a b x y : ℝ) 
  (h1 : x = 1) 
  (h2 : y = 2)
  (h3 : a * x + y = -1)
  (h4 : 2 * x - b * y = 0) : 
  a + b = -2 := 
sorry

end system_of_equations_solution_l1098_109824


namespace problem_1_problem_2_l1098_109813

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.log (x + 1) + Real.log (1 - x) + a * (x + 1)

def mono_intervals (a : ℝ) : Set ℝ × Set ℝ := 
  if a = 1 then ((Set.Ioo (-1) (Real.sqrt 2 - 1)), (Set.Ico (Real.sqrt 2 - 1) 1)) 
  else (∅, ∅)

theorem problem_1 (a : ℝ) (h_pos : a > 0) : 
  mono_intervals a = if a = 1 then ((Set.Ioo (-1) (Real.sqrt 2 - 1)), (Set.Ico (Real.sqrt 2 - 1) 1)) else (∅, ∅) :=
sorry

theorem problem_2 (h_max : f a 0 = 1) (h_pos : a > 0) : 
  a = 1 :=
sorry

end problem_1_problem_2_l1098_109813


namespace solution_set_inequality_l1098_109827

theorem solution_set_inequality (a m : ℝ) (h : ∀ x : ℝ, (x > m ∧ x < 1) ↔ 2 * x^2 - 3 * x + a < 0) : m = 1 / 2 :=
by
  -- Insert the proof here
  sorry

end solution_set_inequality_l1098_109827


namespace line_equation_sum_l1098_109864

theorem line_equation_sum (m b x y : ℝ) (hx : x = 4) (hy : y = 2) (hm : m = -5) (hline : y = m * x + b) : m + b = 17 := by
  sorry

end line_equation_sum_l1098_109864


namespace tan_alpha_plus_405_deg_l1098_109849

theorem tan_alpha_plus_405_deg (α : ℝ) (h : Real.tan (180 - α) = -4 / 3) : Real.tan (α + 405) = -7 := 
sorry

end tan_alpha_plus_405_deg_l1098_109849


namespace rides_first_day_l1098_109832

variable (total_rides : ℕ) (second_day_rides : ℕ)

theorem rides_first_day (h1 : total_rides = 7) (h2 : second_day_rides = 3) : total_rides - second_day_rides = 4 :=
by
  sorry

end rides_first_day_l1098_109832


namespace similar_inscribed_triangle_exists_l1098_109826

variable {α : Type*} [LinearOrderedField α]

-- Representing points and triangles
structure Point (α : Type*) := (x : α) (y : α)
structure Triangle (α : Type*) := (A B C : Point α)

-- Definitions for inscribed triangles and similarity conditions
def isInscribed (inner outer : Triangle α) : Prop :=
  -- Dummy definition, needs correct geometric interpretation
  sorry

def areSimilar (Δ1 Δ2 : Triangle α) : Prop :=
  -- Dummy definition, needs correct geometric interpretation
  sorry

-- Main theorem
theorem similar_inscribed_triangle_exists (Δ₁ Δ₂ : Triangle α) (h_ins : isInscribed Δ₂ Δ₁) :
  ∃ Δ₃ : Triangle α, isInscribed Δ₃ Δ₂ ∧ areSimilar Δ₁ Δ₃ :=
sorry

end similar_inscribed_triangle_exists_l1098_109826


namespace at_most_two_greater_than_one_l1098_109804

theorem at_most_two_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  ¬ (2 * a - 1 / b > 1 ∧ 2 * b - 1 / c > 1 ∧ 2 * c - 1 / a > 1) :=
by
  sorry

end at_most_two_greater_than_one_l1098_109804


namespace cos_double_angle_zero_l1098_109898

theorem cos_double_angle_zero (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = Real.cos (Real.pi / 6 + α)) : Real.cos (2 * α) = 0 := 
sorry

end cos_double_angle_zero_l1098_109898


namespace A_half_B_l1098_109868

-- Define the arithmetic series sum function
def series_sum (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define A and B according to the problem conditions
def A : ℕ := (Finset.range 2022).sum (λ m => series_sum (m + 1))

def B : ℕ := (Finset.range 2022).sum (λ m => (m + 1) * (m + 2))

-- The proof statement
theorem A_half_B : A = B / 2 :=
by
  sorry

end A_half_B_l1098_109868


namespace john_has_48_l1098_109869

variable (Ali Nada John : ℕ)

theorem john_has_48 
  (h1 : Ali + Nada + John = 67)
  (h2 : Ali = Nada - 5)
  (h3 : John = 4 * Nada) : 
  John = 48 := 
by 
  sorry

end john_has_48_l1098_109869


namespace problem_3_problem_4_l1098_109893

open Classical

section
  variable {x₁ x₂ : ℝ}
  theorem problem_3 (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) : (Real.log (x₁ * x₂) = Real.log x₁ + Real.log x₂) :=
  by
    sorry

  theorem problem_4 (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hlt : x₁ < x₂) : ((Real.log x₁ - Real.log x₂) / (x₁ - x₂) > 0) :=
  by
    sorry
end

end problem_3_problem_4_l1098_109893


namespace selling_price_ratio_l1098_109852

theorem selling_price_ratio (C : ℝ) (hC : C > 0) :
  let S₁ := 1.60 * C
  let S₂ := 4.20 * C
  S₂ / S₁ = 21 / 8 :=
by
  let S₁ := 1.60 * C
  let S₂ := 4.20 * C
  sorry

end selling_price_ratio_l1098_109852


namespace range_of_a_l1098_109805
noncomputable def exponential_quadratic (a : ℝ) : Prop :=
  ∃ x : ℝ, 0 < x ∧ (1/4)^x + (1/2)^(x-1) + a = 0

theorem range_of_a (a : ℝ) : exponential_quadratic a ↔ -3 < a ∧ a < 0 :=
sorry

end range_of_a_l1098_109805


namespace intersection_P_Q_l1098_109885

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}

theorem intersection_P_Q :
  P ∩ Q = {-1, 0, 1} :=
sorry

end intersection_P_Q_l1098_109885


namespace pipe_A_time_to_fill_l1098_109846

theorem pipe_A_time_to_fill (T_B : ℝ) (T_combined : ℝ) (T_A : ℝ): 
  T_B = 75 → T_combined = 30 → 
  (1 / T_B + 1 / T_A = 1 / T_combined) → T_A = 50 :=
by
  -- Placeholder proof
  intro h1 h2 h3
  have h4 : T_B = 75 := h1
  have h5 : T_combined = 30 := h2
  have h6 : 1 / T_B + 1 / T_A = 1 / T_combined := h3
  sorry

end pipe_A_time_to_fill_l1098_109846


namespace star_value_l1098_109856

-- Define the operation a star b
def star (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

-- We want to prove that 5 star 3 = 4
theorem star_value : star 5 3 = 4 := by
  sorry

end star_value_l1098_109856


namespace Charles_chocolate_milk_total_l1098_109812

theorem Charles_chocolate_milk_total (milk_per_glass syrup_per_glass total_milk total_syrup : ℝ) 
(h_milk_glass : milk_per_glass = 6.5) (h_syrup_glass : syrup_per_glass = 1.5) (h_total_milk : total_milk = 130) (h_total_syrup : total_syrup = 60) :
  (min (total_milk / milk_per_glass) (total_syrup / syrup_per_glass) * (milk_per_glass + syrup_per_glass) = 160) :=
by
  sorry

end Charles_chocolate_milk_total_l1098_109812


namespace fraction_arithmetic_l1098_109843

theorem fraction_arithmetic : ((3 / 5 : ℚ) + (4 / 15)) * (2 / 3) = 26 / 45 := 
by
  sorry

end fraction_arithmetic_l1098_109843


namespace z_real_iff_m_1_or_2_z_complex_iff_not_m_1_and_2_z_pure_imaginary_iff_m_neg_half_z_in_second_quadrant_l1098_109825

variables (m : ℝ)

def z_re (m : ℝ) : ℝ := 2 * m^2 - 3 * m - 2
def z_im (m : ℝ) : ℝ := m^2 - 3 * m + 2

-- Part (Ⅰ) Question 1
theorem z_real_iff_m_1_or_2 (m : ℝ) :
  z_im m = 0 ↔ (m = 1 ∨ m = 2) :=
sorry

-- Part (Ⅰ) Question 2
theorem z_complex_iff_not_m_1_and_2 (m : ℝ) :
  ¬ (m = 1 ∨ m = 2) ↔ (m ≠ 1 ∧ m ≠ 2) :=
sorry

-- Part (Ⅰ) Question 3
theorem z_pure_imaginary_iff_m_neg_half (m : ℝ) :
  z_re m = 0 ∧ z_im m ≠ 0 ↔ (m = -1/2) :=
sorry

-- Part (Ⅱ) Question
theorem z_in_second_quadrant (m : ℝ) :
  z_re m < 0 ∧ z_im m > 0 ↔ -1/2 < m ∧ m < 1 :=
sorry

end z_real_iff_m_1_or_2_z_complex_iff_not_m_1_and_2_z_pure_imaginary_iff_m_neg_half_z_in_second_quadrant_l1098_109825


namespace annulus_area_l1098_109859

theorem annulus_area (r R x : ℝ) (hR_gt_r : R > r) (h_tangent : r^2 + x^2 = R^2) : 
  π * x^2 = π * (R^2 - r^2) :=
by
  sorry

end annulus_area_l1098_109859


namespace cards_distribution_l1098_109890

theorem cards_distribution (total_cards people : ℕ) (h1 : total_cards = 48) (h2 : people = 7) :
  (people - (total_cards % people)) = 1 :=
by
  sorry

end cards_distribution_l1098_109890


namespace maximize_NPM_l1098_109848

theorem maximize_NPM :
  ∃ (M N P : ℕ), 
    (∀ M, M < 10 → (11 * M * M) = N * 100 + P * 10 + M) →
    N * 100 + P * 10 + M = 396 :=
by
  sorry

end maximize_NPM_l1098_109848


namespace jane_oldest_child_age_l1098_109839

-- Define the conditions
def jane_start_age : ℕ := 20
def jane_current_age : ℕ := 32
def stopped_babysitting_years_ago : ℕ := 10
def baby_sat_condition (jane_age child_age : ℕ) : Prop := child_age ≤ jane_age / 2

-- Define the proof problem
theorem jane_oldest_child_age :
  (∃ age_stopped child_age,
    stopped_babysitting_years_ago = jane_current_age - age_stopped ∧
    baby_sat_condition age_stopped child_age ∧
    (32 - stopped_babysitting_years_ago = 22) ∧ -- Jane's age when she stopped baby-sitting
    child_age = 22 / 2 ∧ -- Oldest child she could have baby-sat at age 22
    child_age + stopped_babysitting_years_ago = 21) --  current age of the oldest person for whom Jane could have baby-sat
:= sorry

end jane_oldest_child_age_l1098_109839


namespace speed_of_current_l1098_109815

-- Definitions
def speed_boat_still_water := 60
def speed_downstream := 77
def speed_upstream := 43

-- Theorem statement
theorem speed_of_current : ∃ x, speed_boat_still_water + x = speed_downstream ∧ speed_boat_still_water - x = speed_upstream ∧ x = 17 :=
by
  unfold speed_boat_still_water speed_downstream speed_upstream
  sorry

end speed_of_current_l1098_109815


namespace minimum_value_of_f_l1098_109889

def f (x : ℝ) : ℝ := |x - 4| + |x + 6| + |x - 5|

theorem minimum_value_of_f :
  ∃ x : ℝ, (x = -6 ∧ f (-6) = 1) ∧ ∀ y : ℝ, f y ≥ 1 :=
by
  sorry

end minimum_value_of_f_l1098_109889


namespace convex_quad_sum_greater_diff_l1098_109851

theorem convex_quad_sum_greater_diff (α β γ δ : ℝ) 
    (h_sum : α + β + γ + δ = 360) 
    (h_convex : α < 180 ∧ β < 180 ∧ γ < 180 ∧ δ < 180) :
    ∀ (x y z w : ℝ), (x = α ∨ x = β ∨ x = γ ∨ x = δ) → (y = α ∨ y = β ∨ y = γ ∨ y = δ) → 
                     (z = α ∨ z = β ∨ z = γ ∨ z = δ) → (w = α ∨ w = β ∨ w = γ ∨ w = δ) 
                     → x + y > |z - w| := 
by
  sorry

end convex_quad_sum_greater_diff_l1098_109851


namespace number_of_balls_l1098_109823

noncomputable def totalBalls (frequency : ℚ) (yellowBalls : ℕ) : ℚ :=
  yellowBalls / frequency

theorem number_of_balls (h : totalBalls 0.3 6 = 20) : true :=
by
  sorry

end number_of_balls_l1098_109823


namespace product_mnp_l1098_109818

theorem product_mnp (a x y b : ℝ) (m n p : ℕ):
  (a ^ 8 * x * y - 2 * a ^ 7 * y - 3 * a ^ 6 * x = 2 * a ^ 5 * (b ^ 5 - 2)) ∧
  (a ^ 8 * x * y - 2 * a ^ 7 * y - 3 * a ^ 6 * x + 6 * a ^ 5 = (a ^ m * x - 2 * a ^ n) * (a ^ p * y - 3 * a ^ 3)) →
  m = 5 ∧ n = 4 ∧ p = 3 ∧ m * n * p = 60 :=
by
  intros h
  sorry

end product_mnp_l1098_109818


namespace cube_edge_length_l1098_109877

theorem cube_edge_length (a : ℝ) (h : 6 * a^2 = 24) : a = 2 :=
by sorry

end cube_edge_length_l1098_109877


namespace proportion_solution_l1098_109833

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 6) : x = 0.9 := by
  sorry

end proportion_solution_l1098_109833


namespace quadratic_no_real_roots_min_k_l1098_109854

theorem quadratic_no_real_roots_min_k :
  ∀ (k : ℤ), 
    (∀ x : ℝ, 3*x*(k*x-5) - 2*x^2 + 8 ≠ 0) ↔ 
    (k ≥ 3) := 
by 
  sorry

end quadratic_no_real_roots_min_k_l1098_109854


namespace shpuntik_can_form_triangle_l1098_109883

-- Define lengths of the sticks before swap
variables {a b c d e f : ℝ}

-- Conditions before the swap
-- Both sets of sticks can form a triangle
-- The lengths of Vintik's sticks are a, b, c
-- The lengths of Shpuntik's sticks are d, e, f
axiom triangle_ineq_vintik : a + b > c ∧ b + c > a ∧ c + a > b
axiom triangle_ineq_shpuntik : d + e > f ∧ e + f > d ∧ f + d > e
axiom sum_lengths_vintik : a + b + c = 1
axiom sum_lengths_shpuntik : d + e + f = 1

-- Define lengths of the sticks after swap
-- x1, x2, x3 are Vintik's new sticks; y1, y2, y3 are Shpuntik's new sticks
variables {x1 x2 x3 y1 y2 y3 : ℝ}

-- Neznaika's swap
axiom swap_stick_vintik : x1 = a ∧ x2 = b ∧ x3 = f ∨ x1 = a ∧ x2 = d ∧ x3 = c ∨ x1 = e ∧ x2 = b ∧ x3 = c
axiom swap_stick_shpuntik : y1 = d ∧ y2 = e ∧ y3 = c ∨ y1 = e ∧ y2 = b ∧ y3 = f ∨ y1 = a ∧ y2 = b ∧ y3 = f 

-- Total length after the swap remains unchanged
axiom sum_lengths_after_swap : x1 + x2 + x3 + y1 + y2 + y3 = 2

-- Vintik cannot form a triangle with the current lengths
axiom no_triangle_vintik : x1 >= x2 + x3

-- Prove that Shpuntik can still form a triangle
theorem shpuntik_can_form_triangle : y1 + y2 > y3 ∧ y2 + y3 > y1 ∧ y3 + y1 > y2 := sorry

end shpuntik_can_form_triangle_l1098_109883


namespace david_older_than_scott_l1098_109821

-- Define the ages of Richard, David, and Scott
variables (R D S : ℕ)

-- Given conditions
def richard_age_eq : Prop := R = D + 6
def richard_twice_scott : Prop := R + 8 = 2 * (S + 8)
def david_current_age : Prop := D = 14

-- Prove the statement
theorem david_older_than_scott (h1 : richard_age_eq R D) (h2 : richard_twice_scott R S) (h3 : david_current_age D) :
  D - S = 8 :=
  sorry

end david_older_than_scott_l1098_109821


namespace sin_A_in_right_triangle_l1098_109835

theorem sin_A_in_right_triangle (B C A : Real) (hBC: B + C = π / 2) 
(h_sinB: Real.sin B = 3 / 5) (h_sinC: Real.sin C = 4 / 5) : 
Real.sin A = 1 := 
by 
  sorry

end sin_A_in_right_triangle_l1098_109835


namespace projectile_reaches_24m_at_12_7_seconds_l1098_109895

theorem projectile_reaches_24m_at_12_7_seconds :
  ∃ t : ℝ, (y = -4.9 * t^2 + 25 * t) ∧ y = 24 ∧ t = 12 / 7 :=
by
  use 12 / 7
  sorry

end projectile_reaches_24m_at_12_7_seconds_l1098_109895


namespace why_build_offices_l1098_109866

structure Company where
  name : String
  hasSkillfulEmployees : Prop
  uniqueComfortableWorkEnvironment : Prop
  integratedWorkLeisureSpaces : Prop
  reducedEmployeeStress : Prop
  flexibleWorkSchedules : Prop
  increasesProfit : Prop

theorem why_build_offices (goog_fb : Company)
  (h1 : goog_fb.hasSkillfulEmployees)
  (h2 : goog_fb.uniqueComfortableWorkEnvironment)
  (h3 : goog_fb.integratedWorkLeisureSpaces)
  (h4 : goog_fb.reducedEmployeeStress)
  (h5 : goog_fb.flexibleWorkSchedules) :
  goog_fb.increasesProfit := 
sorry

end why_build_offices_l1098_109866


namespace optometrist_sales_l1098_109850

noncomputable def total_pairs_optometrist_sold (H S : ℕ) (total_sales: ℝ) : Prop :=
  (S = H + 7) ∧ 
  (total_sales = 0.9 * (95 * ↑H + 175 * ↑S)) ∧ 
  (total_sales = 2469)

theorem optometrist_sales :
  ∃ H S : ℕ, total_pairs_optometrist_sold H S 2469 ∧ H + S = 17 :=
by 
  sorry

end optometrist_sales_l1098_109850


namespace minimize_notch_volume_l1098_109834

noncomputable def total_volume (theta phi : ℝ) : ℝ :=
  let part1 := (2 / 3) * Real.tan phi
  let part2 := (2 / 3) * Real.tan (theta - phi)
  part1 + part2

theorem minimize_notch_volume :
  ∀ (theta : ℝ), (0 < theta ∧ theta < π) →
  ∃ (phi : ℝ), (0 < phi ∧ phi < θ) ∧
  (∀ ψ : ℝ, (0 < ψ ∧ ψ < θ) → total_volume theta ψ ≥ total_volume theta (theta / 2)) :=
by
  -- Proof is omitted
  sorry

end minimize_notch_volume_l1098_109834


namespace parallel_vectors_l1098_109819

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (2, m)

theorem parallel_vectors (m : ℝ) (h : ∃ k : ℝ, vector_a = (k • vector_b m)) : m = -4 :=
by {
  sorry
}

end parallel_vectors_l1098_109819


namespace van_speed_maintain_l1098_109881

theorem van_speed_maintain 
  (D : ℕ) (T T_new : ℝ) 
  (initial_distance : D = 435) 
  (initial_time : T = 5) 
  (new_time : T_new = T / 2) : 
  D / T_new = 174 := 
by 
  sorry

end van_speed_maintain_l1098_109881


namespace team_combinations_l1098_109882

/-- 
The math club at Walnutridge High School has five girls and seven boys. 
How many different teams, comprising two girls and two boys, can be formed 
if one boy on each team must also be designated as the team leader?
-/
theorem team_combinations (girls boys : ℕ) (h_girls : girls = 5) (h_boys : boys = 7) :
  ∃ n, n = 420 :=
by
  sorry

end team_combinations_l1098_109882


namespace min_value_I_is_3_l1098_109872

noncomputable def min_value_I (a b c x y : ℝ) : ℝ :=
  1 / (2 * a^3 * x + b^3 * y^2) + 1 / (2 * b^3 * x + c^3 * y^2) + 1 / (2 * c^3 * x + a^3 * y^2)

theorem min_value_I_is_3 {a b c x y : ℝ} (h1 : a^6 + b^6 + c^6 = 3) (h2 : (x + 1)^2 + y^2 ≤ 2) :
  3 ≤ min_value_I a b c x y :=
sorry

end min_value_I_is_3_l1098_109872


namespace simplify_expression_l1098_109808

theorem simplify_expression :
  2 + 1 / (2 + 1 / (2 + 1 / 2)) = 29 / 12 :=
by
  sorry  -- Proof will be provided here

end simplify_expression_l1098_109808


namespace scott_earnings_l1098_109847

theorem scott_earnings
  (price_smoothie : ℝ)
  (price_cake : ℝ)
  (cups_sold : ℝ)
  (cakes_sold : ℝ)
  (earnings_smoothies : ℝ := cups_sold * price_smoothie)
  (earnings_cakes : ℝ := cakes_sold * price_cake) :
  price_smoothie = 3 → price_cake = 2 → cups_sold = 40 → cakes_sold = 18 → 
  (earnings_smoothies + earnings_cakes) = 156 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end scott_earnings_l1098_109847


namespace regular_2020_gon_isosceles_probability_l1098_109871

theorem regular_2020_gon_isosceles_probability :
  let n := 2020
  let totalTriangles := (n * (n - 1) * (n - 2)) / 6
  let isoscelesTriangles := n * ((n - 2) / 2)
  let probability := isoscelesTriangles * 6 / totalTriangles
  let (a, b) := (1, 673)
  100 * a + b = 773 := by
    sorry

end regular_2020_gon_isosceles_probability_l1098_109871


namespace smallest_odd_number_divisible_by_3_l1098_109891

theorem smallest_odd_number_divisible_by_3 : ∃ n : ℕ, n = 3 ∧ ∀ m : ℕ, (m % 2 = 1 ∧ m % 3 = 0) → m ≥ n := 
by
  sorry

end smallest_odd_number_divisible_by_3_l1098_109891


namespace ratio_w_to_y_l1098_109830

variable (w x y z : ℚ)
variable (h1 : w / x = 5 / 4)
variable (h2 : y / z = 5 / 3)
variable (h3 : z / x = 1 / 5)

theorem ratio_w_to_y : w / y = 15 / 4 := sorry

end ratio_w_to_y_l1098_109830


namespace attendees_proportion_l1098_109844

def attendees (t k : ℕ) := k / t

theorem attendees_proportion (n t new_t : ℕ) (h1 : n * t = 15000) (h2 : t = 50) (h3 : new_t = 75) : attendees new_t 15000 = 200 :=
by
  -- Proof omitted, main goal is to assert equivalency
  sorry

end attendees_proportion_l1098_109844


namespace shanghai_mock_exam_problem_l1098_109828

noncomputable def a_n : ℕ → ℝ := sorry -- Defines the arithmetic sequence 

theorem shanghai_mock_exam_problem 
  (a_is_arithmetic : ∃ d a₀, ∀ n, a_n n = a₀ + n * d)
  (h₁ : a_n 1 + a_n 3 + a_n 5 = 9)
  (h₂ : a_n 2 + a_n 4 + a_n 6 = 15) :
  a_n 3 + a_n 4 = 8 := 
  sorry

end shanghai_mock_exam_problem_l1098_109828


namespace jenna_eel_length_l1098_109874

theorem jenna_eel_length (j b : ℕ) (h1 : b = 3 * j) (h2 : b + j = 64) : j = 16 := by 
  sorry

end jenna_eel_length_l1098_109874


namespace student_calculation_no_error_l1098_109845

theorem student_calculation_no_error :
  let correct_result : ℚ := (7 * 4) / (5 / 3)
  let student_result : ℚ := (7 * 4) * (3 / 5)
  correct_result = student_result → 0 = 0 := 
by
  intros correct_result student_result h
  sorry

end student_calculation_no_error_l1098_109845


namespace sum_of_squares_of_coeffs_l1098_109807

theorem sum_of_squares_of_coeffs (a b c : ℕ) : (a = 6) → (b = 24) → (c = 12) → (a^2 + b^2 + c^2 = 756) :=
by
  sorry

end sum_of_squares_of_coeffs_l1098_109807


namespace sum_difference_arithmetic_sequences_l1098_109842

open Nat

def arithmetic_sequence_sum (a d n : Nat) : Nat :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_difference_arithmetic_sequences :
  arithmetic_sequence_sum 2101 1 123 - arithmetic_sequence_sum 401 1 123 = 209100 := by
  sorry

end sum_difference_arithmetic_sequences_l1098_109842


namespace rect_RS_over_HJ_zero_l1098_109867

theorem rect_RS_over_HJ_zero :
  ∃ (A B C D H I J R S: ℝ × ℝ),
    (A = (0, 6)) ∧
    (B = (8, 6)) ∧
    (C = (8, 0)) ∧
    (D = (0, 0)) ∧
    (H = (5, 6)) ∧
    (I = (8, 4)) ∧
    (J = (3, 0)) ∧
    (R = (15 / 13, -12 / 13)) ∧
    (S = (15 / 13, -12 / 13)) ∧
    (RS = dist R S) ∧
    (HJ = dist H J) ∧
    (HJ ≠ 0) ∧
    (RS / HJ = 0) :=
sorry

end rect_RS_over_HJ_zero_l1098_109867


namespace inequality_solution_set_l1098_109803

theorem inequality_solution_set (x : ℝ) :
  (x - 3)^2 - 2 * Real.sqrt ((x - 3)^2) - 3 < 0 ↔ 0 < x ∧ x < 6 :=
by
  sorry

end inequality_solution_set_l1098_109803


namespace tourists_walking_speed_l1098_109879

-- Define the conditions
def tourists_start_time := 3 + 10 / 60 -- 3:10 A.M.
def bus_pickup_time := 5 -- 5:00 A.M.
def bus_speed := 60 -- 60 km/h
def early_arrival := 20 / 60 -- 20 minutes earlier

-- This is the Lean 4 theorem statement
theorem tourists_walking_speed : 
  (bus_speed * (10 / 60) / (100 / 60)) = 6 := 
by
  sorry

end tourists_walking_speed_l1098_109879


namespace quadratic_has_real_root_l1098_109816

theorem quadratic_has_real_root (a : ℝ) : 
  ¬(∀ x : ℝ, x^2 + a * x + a - 1 ≠ 0) :=
sorry

end quadratic_has_real_root_l1098_109816


namespace scooter_price_l1098_109870

theorem scooter_price (total_cost: ℝ) (h: 0.20 * total_cost = 240): total_cost = 1200 :=
by
  sorry

end scooter_price_l1098_109870


namespace smallest_value_of_a_minus_b_l1098_109894

theorem smallest_value_of_a_minus_b (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) : a - b = 4 :=
by
  sorry

end smallest_value_of_a_minus_b_l1098_109894
