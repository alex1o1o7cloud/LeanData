import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.EuclideanDomain.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Geometry.Pyramid
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.LocalExtr
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Ineq.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Multiplicity
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Tactic

namespace total_skips_correct_l500_500741

def bob_skip_rate := 12
def jim_skip_rate := 15
def sally_skip_rate := 18

def bob_rocks := 10
def jim_rocks := 8
def sally_rocks := 12

theorem total_skips_correct : 
  (bob_skip_rate * bob_rocks) + (jim_skip_rate * jim_rocks) + (sally_skip_rate * sally_rocks) = 456 := by
  sorry

end total_skips_correct_l500_500741


namespace tan_alpha_value_l500_500880

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l500_500880


namespace p_necessary_but_not_sufficient_for_q_l500_500947

variables (x y : ℝ)

def p := x + y > 2 ∧ x * y > 1
def q := x > 1 ∧ y > 1

theorem p_necessary_but_not_sufficient_for_q :
  (∀ x y, q → p) ∧ (∃ x y, p ∧ ¬q) :=
by
  sorry

end p_necessary_but_not_sufficient_for_q_l500_500947


namespace pioneer_ages_l500_500333

def pioneer_data (Burov Gridnev Klimenko Kolya Petya Grisha : String) (Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age : ℕ) :=
  Burov ≠ Kolya ∧
  Petya_age = 12 ∧
  Gridnev_age = Petya_age + 1 ∧
  Grisha_age = Petya_age + 1 ∧
  Burov_age = Grisha_age ∧
-- defining the names corresponding to conditions given in problem
  Burov = Grisha ∧ Gridnev = Kolya ∧ Klimenko = Petya 

theorem pioneer_ages (Burov Gridnev Klimenko Kolya Petya Grisha : String) (Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age : ℕ)
  (h : pioneer_data Burov Gridnev Klimenko Kolya Petya Grisha Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age) :
  (Burov, Burov_age) = (Grisha, 13) ∧ 
  (Gridnev, Gridnev_age) = (Kolya, 13) ∧ 
  (Klimenko, Klimenko_age) = (Petya, 12) :=
by
  sorry

end pioneer_ages_l500_500333


namespace perpendicular_planes_implies_perpendicular_line_l500_500488

-- Definitions of lines and planes and their properties in space
variable {Space : Type}
variable (m n l : Line Space) -- Lines in space
variable (α β γ : Plane Space) -- Planes in space

-- Conditions: m, n, and l are non-intersecting lines, α, β, and γ are non-intersecting planes
axiom non_intersecting_lines : ¬ (m = n) ∧ ¬ (m = l) ∧ ¬ (n = l)
axiom non_intersecting_planes : ¬ (α = β) ∧ ¬ (α = γ) ∧ ¬ (β = γ)

-- To prove: if α ⊥ γ, β ⊥ γ, and α ∩ β = l, then l ⊥ γ
theorem perpendicular_planes_implies_perpendicular_line
  (h1 : α ⊥ γ) 
  (h2 : β ⊥ γ)
  (h3 : α ∩ β = l) : l ⊥ γ := 
  sorry

end perpendicular_planes_implies_perpendicular_line_l500_500488


namespace cone_volume_ratio_l500_500966

theorem cone_volume_ratio (a b : ℝ)
  (h₁ : a = 1) (h₂ : b = sqrt 3) :
  let V1 := (1/3) * π * b^2 * a,
      V2 := (1/3) * π * a^2 * b
  in V1 / V2 = sqrt 3 :=
by 
  let V1 := (1/3) * π * b^2 * a,
  let V2 := (1/3) * π * a^2 * b,
  sorry

end cone_volume_ratio_l500_500966


namespace tan_alpha_fraction_l500_500911

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l500_500911


namespace f_above_g_l500_500127

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.exp x) / (x - m)
def g (x : ℝ) : ℝ := x^2 + x

theorem f_above_g (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1/2) : 
  ∀ x, m ≤ x ∧ x ≤ m + 1 → f x m > g x := 
sorry

end f_above_g_l500_500127


namespace arithmetic_geometric_mean_inequality_l500_500469

theorem arithmetic_geometric_mean_inequality {n : ℕ} (h : 2 ≤ n) (a : Fin n → ℝ) (ha : ∀ i, 0 < a i) :
  (∑ i, a i) / n ≥ (∏ i, a i) ^ (1 / n) :=
by
  -- proof goes here
  sorry

end arithmetic_geometric_mean_inequality_l500_500469


namespace Tim_words_known_l500_500336

def Tim_original_words : ℕ := 14600

theorem Tim_words_known (days_in_two_years : ℕ) (daily_words_learned : ℕ) 
(increase_percentage : ℚ) (total_words_learned : ℕ) (original_words : ℕ) :
  days_in_two_years = 730 →
  daily_words_learned = 10 →
  increase_percentage = 0.5 →
  total_words_learned = days_in_two_years * daily_words_learned →
  original_words = total_words_learned * 2 →
  original_words = Tim_original_words :=
by
  intros h1 h2 h3 h4 h5
  rw [←h5, ←h4, h1, h2]
  unfold Tim_original_words
  sorry

end Tim_words_known_l500_500336


namespace range_of_function_l500_500297

def f (x : ℝ) : ℝ := 1 + log x / log 2

theorem range_of_function : (∀ x:ℝ, x ≥ 4 → ∃ y, y = f x ∧ y ≥ 3) ∧ (∀ y:ℝ, y ≥ 3 → ∃ x, x ≥ 4 ∧ y = f x) :=
by
  sorry

end range_of_function_l500_500297


namespace poem_mode_median_l500_500331

def poem_counts : List (ℕ × ℕ) := [(4, 3), (5, 4), (6, 4), (7, 5), (8, 7), (9, 5), (10, 1), (11, 1)]

def mode (counts : List (ℕ × ℕ)) : ℕ :=
  counts.foldr (λ p acc, if p.2 > acc.2 then p else acc) (0, 0) |>.fst

def median (counts : List (ℕ × ℕ)) : ℕ :=
  let sortedData := counts.flatMap (λ ⟨poem, num⟩, List.replicate num poem)
  let mid1 := sortedData.nth ((sortedData.length / 2) - 1)
  let mid2 := sortedData.nth (sortedData.length / 2)
  match mid1, mid2 with
  | some x, some y => (x + y) / 2
  | _, _ => 0

theorem poem_mode_median : mode poem_counts = 8 ∧ median poem_counts = 7 := by
  sorry

end poem_mode_median_l500_500331


namespace max_value_area_under_curve_volume_of_solid_l500_500086

-- Define the curve's function
def curve (x : ℝ) : ℝ := x * sqrt (9 - x^2)

-- Define the conditions
axiom nonneg_x (x : ℝ) : Prop := x ≥ 0

-- Define the maximum value problem
theorem max_value : ∀ x : ℝ, nonneg_x x → curve x ≤ 4.5 ∧ curve (3 * sqrt 2 / 2) = 4.5 :=
sorry

-- Define the area under the curve
theorem area_under_curve : ∫ x in 0..3, curve x = 9 :=
sorry

-- Define the volume of the solid by revolution around y-axis
theorem volume_of_solid : ∫ x in 0..3, π * (curve x)^2 = (162 * π) / 5 :=
sorry

end max_value_area_under_curve_volume_of_solid_l500_500086


namespace range_of_m_l500_500615

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, (x^2)/(9 - m) + (y^2)/(m - 5) = 1 → 
  (∃ m, (7 < m ∧ m < 9))) := 
sorry

end range_of_m_l500_500615


namespace binomial_sum_l500_500454

theorem binomial_sum (k : ℕ) (h1 : (Nat.choose 29 4 + Nat.choose 29 5 = Nat.choose 30 k)) :
  k = 5 ∨ k = 25 → k = 5 + k = 25 → k ∈ [5, 25] :=
by
  sorry

end binomial_sum_l500_500454


namespace find_a_9_coefficient_l500_500796

theorem find_a_9_coefficient :
  let x : ℝ := sorry
  let a : ℕ → ℝ := sorry
  let (b : ℝ → ℝ) : ℝ → ℝ := fun t => (1 + x)^10 - (1 - t)^10
  
  -- Given conditions (These are placeholders, adjust as per Lean syntax requirements)
  (h1 : (1 + x)^10 = a(0) + a(1) * (1 - x) + a(2) * (1 - x)^2 + ... + a(10) * (1 - x)^10) 
  (h2 : (1 + x)^10 = (-1 - x)^10) 
  (h3 : (1 + x)^10 = [(-2) + (1 - x)]^10) 
  -- Prove that 
  (h4 : a(9) = -20) : a(9) = -20 :=
sorry

end find_a_9_coefficient_l500_500796


namespace michelle_january_cost_l500_500690

noncomputable def cell_phone_cost (base_cost : ℕ) (text_rate : ℕ) (extra_minute_rate : ℕ)
  (included_hours : ℕ) (texts_sent : ℕ) (talked_hours : ℕ) : ℕ :=
  let cost_base := base_cost
  let cost_texts := texts_sent * text_rate / 100
  let extra_minutes := (talked_hours - included_hours) * 60
  let cost_extra_minutes := extra_minutes * extra_minute_rate / 100
  cost_base + cost_texts + cost_extra_minutes

theorem michelle_january_cost : cell_phone_cost 20 5 10 30 100 30.5 = 28 :=
by
  sorry

end michelle_january_cost_l500_500690


namespace tan_alpha_proof_l500_500898

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l500_500898


namespace infinite_friendly_squares_l500_500712

def is_friendly (N : ℕ) : Prop :=
  ∃ (pairs : list (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 + p.2 = (math.sqrt (p.1 + p.2))^2) ∧ 
    list.sum (list.map (λ p, p.1 + p.2) pairs) = N ∧
    (∀ x : ℕ, x ∈ list.map prod.fst pairs ∨ x ∈ list.map prod.snd pairs)

theorem infinite_friendly_squares :
  ∀ p ≥ 2, ∃ N, is_friendly (2^(2*p - 3)) :=
by
  sorry

end infinite_friendly_squares_l500_500712


namespace A_and_D_independent_l500_500309

-- Define the probabilities of elementary events
def prob_A : ℚ := 1 / 6
def prob_B : ℚ := 1 / 6
def prob_C : ℚ := 5 / 36
def prob_D : ℚ := 1 / 6

-- Define the joint probability of A and D
def prob_A_and_D : ℚ := 1 / 36

-- Define the independence condition
def independent (P_X P_Y P_XY : ℚ) : Prop := P_XY = P_X * P_Y

-- Prove that events A and D are independent
theorem A_and_D_independent : 
  independent prob_A prob_D prob_A_and_D := by
  -- The proof is skipped
  sorry

end A_and_D_independent_l500_500309


namespace checkerboard_has_55_squares_with_at_least_six_black_squares_l500_500024

def is_black_square (x y : ℕ) : Prop :=
  (x % 2 = 0 ∧ y % 2 = 0) ∨ (x % 2 = 1 ∧ y % 2 = 1)

def count_squares_with_at_least_six_black (n : ℕ) (m : ℕ) : ℕ :=
  let count_in_r (r : ℕ) :=
    if r < 3
    then 0
    else (r - 2) * (r - 2) * (if m % 2 = 0 then r * r / 2 else r * r / 2 + r)  -- Count based on inherent black squares
  finset.sum (finset.range (n - 2)) count_in_r

theorem checkerboard_has_55_squares_with_at_least_six_black_squares :
  count_squares_with_at_least_six_black 8 6 = 55 :=
by
  sorry

end checkerboard_has_55_squares_with_at_least_six_black_squares_l500_500024


namespace concurrency_of_perpendiculars_l500_500466

open EuclideanGeometry

variable {ABC : Triangle}
variable {D E F L M N P Q R : Point}
variable {I O : Circle}
variable {l_A l_B l_C : Line}

noncomputable def incircle := incircle ABC
noncomputable def circumcircle := circumcircle ABC

-- Conditions
axiom incircle_tangent (h1 : incircle TABC = I) (h2 : tangent I (side BC ABC) D)
                       (h3 : tangent I (side CA ABC) E) (h4 : tangent I (side AB ABC) F)
axiom circumcircle_intersections {P : Point} (h1 : circumcircle ABC = O) 
                                 (h2 : intersects (line AI ABC) O L) 
                                 (h3 : intersects (line BI ABC) O M) 
                                 (h4 : intersects (line CI ABC) O N)
axiom intersections_with_lines (h1 : intersects (line LD) O P) 
                               (h2 : intersects (line ME) O Q) 
                               (h3 : intersects (line NF) O R)

-- Construction of perpendiculars
noncomputable def perpendicular_to_PA_through_P : Line := ⊥ (segment PA P)
noncomputable def perpendicular_to_QB_through_Q : Line := ⊥ (segment QB Q)
noncomputable def perpendicular_to_RC_through_R : Line := ⊥ (segment RC R)

-- The theorem we want to prove
theorem concurrency_of_perpendiculars 
  (h1 : incircle_tangent I D E F)
  (h2 : circumcircle_intersections L M N)
  (h3 : intersections_with_lines P Q R) :
  ∃ (I : Point), concurrent (perpendicular_to_PA_through_P) (perpendicular_to_QB_through_Q) (perpendicular_to_RC_through_R) ∧
  is_incenter I ABC :=
sorry

end concurrency_of_perpendiculars_l500_500466


namespace commute_time_variance_l500_500006

theorem commute_time_variance
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : (x - 10)^2 + (y - 10)^2 = 8) :
  x^2 + y^2 = 208 :=
by
  sorry

end commute_time_variance_l500_500006


namespace min_int_solution_inequality_l500_500625

theorem min_int_solution_inequality : ∃ x : ℤ, 4 * (x + 1) + 2 > x - 1 ∧ ∀ y : ℤ, 4 * (y + 1) + 2 > y - 1 → y ≥ x := 
by 
  sorry

end min_int_solution_inequality_l500_500625


namespace tan_alpha_solution_l500_500863

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l500_500863


namespace integer_solutions_range_l500_500090

def operation (p q : ℝ) : ℝ := p + q - p * q

theorem integer_solutions_range (m : ℝ) :
  (∃ (x1 x2 : ℤ), (operation 2 x1 > 0) ∧ (operation x1 3 ≤ m) ∧ (operation 2 x2 > 0) ∧ (operation x2 3 ≤ m) ∧ (x1 ≠ x2)) ↔ (3 ≤ m ∧ m < 5) :=
by sorry

end integer_solutions_range_l500_500090


namespace circumscribed_sphere_surface_area_l500_500798

theorem circumscribed_sphere_surface_area
  (x y z : ℝ)
  (h1 : x * y = Real.sqrt 6)
  (h2 : y * z = Real.sqrt 2)
  (h3 : z * x = Real.sqrt 3) :
  let l := Real.sqrt (x^2 + y^2 + z^2)
  let R := l / 2
  4 * Real.pi * R^2 = 6 * Real.pi :=
by sorry

end circumscribed_sphere_surface_area_l500_500798


namespace count_integers_in_range_l500_500424

theorem count_integers_in_range :
  {n : ℕ | 15 ≤ n ∧ n ≤ 225}.card = 211 :=
by
  sorry

end count_integers_in_range_l500_500424


namespace find_f_neg_2_l500_500816

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x: ℝ, f (-x) = -f x

-- Problem statement
theorem find_f_neg_2 (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_fx_pos : ∀ x : ℝ, x > 0 → f x = 2 * x ^ 2 - 7) : 
  f (-2) = -1 :=
by
  sorry

end find_f_neg_2_l500_500816


namespace polynomial_degree_l500_500417

theorem polynomial_degree :
  ∀ (x : ℝ), degree ((x^4) * (x^2 - (1/x)) * (2 - (1/x) + (4/(x^3)))) = 6 :=
by
  sorry

end polynomial_degree_l500_500417


namespace least_value_l500_500045

-- Define the quadratic function and its conditions
def quadratic_function (p q r : ℝ) (x : ℝ) : ℝ :=
  p * x^2 + q * x + r

-- Define the conditions for p, q, and r
def conditions (p q r : ℝ) : Prop :=
  p > 0 ∧ (q^2 - 4 * p * r < 0)

-- State the theorem that given the conditions the least value is (4pr - q^2) / 4p
theorem least_value (p q r : ℝ) (h : conditions p q r) :
  ∃ x : ℝ, (∀ y : ℝ, quadratic_function p q r y ≥ quadratic_function p q r x) ∧
  quadratic_function p q r x = (4 * p * r - q^2) / (4 * p) :=
sorry

end least_value_l500_500045


namespace A_and_D_independent_l500_500318

-- Definitions of the events based on given conditions
def event_A (x₁ : ℕ) : Prop := x₁ = 1
def event_B (x₂ : ℕ) : Prop := x₂ = 2
def event_C (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 8
def event_D (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 7

-- Probabilities based on uniform distribution and replacement
def probability_event (event : ℕ → ℕ → Prop) : ℚ :=
  if h : ∃ x₁ : ℕ, ∃ x₂ : ℕ, x₁ ∈ finset.range 1 7 ∧ x₂ ∈ finset.range 1 7 ∧ event x₁ x₂
  then ((finset.card (finset.filter (λ x, event x.1 x.2)
                (finset.product (finset.range 1 7) (finset.range 1 7)))) : ℚ) / 36
  else 0

noncomputable def P_A : ℚ := 1 / 6
noncomputable def P_D : ℚ := 1 / 6
noncomputable def P_A_and_D : ℚ := 1 / 36

-- Independence condition (by definition): P(A ∩ D) = P(A) * P(D)
theorem A_and_D_independent :
  P_A_and_D = P_A * P_D := by
  sorry

end A_and_D_independent_l500_500318


namespace intersection_complement_l500_500226

set_option pp.unicode true

universe u

def U := set ℤ
def M := {1, 2}
def P : set ℤ := { x | abs x ≤ 2 }

theorem intersection_complement (U : set ℤ) :
  let C_U_M := U \ M in
  P ∩ C_U_M = {-2, -1, 0} :=
by 
  let C_U_M := U \ M;
  sorry

end intersection_complement_l500_500226


namespace clive_change_l500_500037

theorem clive_change (money : ℝ) (olives_needed : ℕ) (olives_per_jar : ℕ) (price_per_jar : ℝ) : 
  (money = 10) → 
  (olives_needed = 80) → 
  (olives_per_jar = 20) →
  (price_per_jar = 1.5) →
  money - (olives_needed / olives_per_jar) * price_per_jar = 4 := by
  sorry

end clive_change_l500_500037


namespace pages_wed_calculation_l500_500759

def pages_mon : ℕ := 23
def pages_tue : ℕ := 38
def pages_thu : ℕ := 12
def pages_fri : ℕ := 2 * pages_thu
def total_pages : ℕ := 158

theorem pages_wed_calculation (pages_wed : ℕ) : 
  pages_mon + pages_tue + pages_wed + pages_thu + pages_fri = total_pages → pages_wed = 61 :=
by
  intros h
  sorry

end pages_wed_calculation_l500_500759


namespace distance_center_to_line_l500_500280

-- Definitions of the circle and the line
def circle_eq (x y : ℝ) := x^2 + y^2 - 2 * x + 4 * y + 3 = 0
def line_eq (x y : ℝ) := x - y = 1

-- Definition of the center of the circle
def circle_center := (1 : ℝ, -2 : ℝ)

-- Distance formula between a point and a line
def point_to_line_distance (x0 y0 : ℝ) (a b c : ℝ) :=
  abs (a * x0 + b * y0 + c) / real.sqrt (a^2 + b^2)

-- Specifying the constants a, b, and c in the line equation
def line_coeff_a := 1
def line_coeff_b := -1
def line_coeff_c := -1

-- Providing the proof statement
theorem distance_center_to_line : 
  point_to_line_distance (circle_center.1) (circle_center.2) line_coeff_a line_coeff_b line_coeff_c = real.sqrt 2 := 
by {
  sorry
}

end distance_center_to_line_l500_500280


namespace sin_alpha_add_pi_div_three_eq_sqrt_three_div_two_l500_500822

-- Definition and conditions
variable (α: ℝ) (k: ℤ)
def P := (1, real.sqrt 3)
def terminal_side (α : ℝ) : Prop :=
  ∃ (k: ℤ), α = real.pi / 3 + 2 * k * real.pi

-- Lean statement for the proof problem
theorem sin_alpha_add_pi_div_three_eq_sqrt_three_div_two
  (h₁: terminal_side α) :
  real.sin (α + real.pi / 3) = real.sqrt 3 / 2 :=
sorry

end sin_alpha_add_pi_div_three_eq_sqrt_three_div_two_l500_500822


namespace ella_distance_from_start_l500_500062

noncomputable def compute_distance (m1 : ℝ) (f1 f2 m_to_f : ℝ) : ℝ :=
  let f1' := m1 * m_to_f
  let total_west := f1' + f2
  let distance_in_feet := Real.sqrt (f1^2 + total_west^2)
  distance_in_feet / m_to_f

theorem ella_distance_from_start :
  let starting_west := 10
  let first_north := 30
  let second_west := 40
  let meter_to_feet := 3.28084 
  compute_distance starting_west first_north second_west meter_to_feet = 24.01 := sorry

end ella_distance_from_start_l500_500062


namespace unique_necklace_arrangements_l500_500174

-- Definitions
def num_beads : Nat := 7

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- The number of unique ways to arrange the beads on a necklace
-- considering rotations and reflections
theorem unique_necklace_arrangements : (factorial num_beads) / (num_beads * 2) = 360 := 
by
  sorry

end unique_necklace_arrangements_l500_500174


namespace javier_speech_time_l500_500553

theorem javier_speech_time :
  ∃ (W : ℕ), W > 30 ∧ 30 + W + W/2 = 117 → W - 30 = 28 :=
begin
  sorry
end

end javier_speech_time_l500_500553


namespace unique_positive_solution_l500_500452

noncomputable def equation (x : ℝ) : Prop :=
  cos (arcsin (cot (arccos (real.sqrt x)))) = x

theorem unique_positive_solution :
  ∃! x : ℝ, 0 < x ∧ x ≤ 1 ∧ equation x :=
sorry

end unique_positive_solution_l500_500452


namespace max_profit_achieved_at_18_thousand_toys_l500_500373

noncomputable def fixed_cost := 100
noncomputable def sales_price_per_toy := 20

def variable_cost (x : ℝ) : ℝ :=
  if h: 0 < x ∧ x < 15 then 12*x - 12*Real.log(x + 1)
  else if h: x ≥ 15 then 21*x + 256/(x - 2) - 200
  else 0 -- default case, should ideally never happen if x is properly bounded

def profit (x : ℝ) : ℝ :=
  let revenue := sales_price_per_toy * x
  let cost := variable_cost x + fixed_cost
  revenue - cost

theorem max_profit_achieved_at_18_thousand_toys :
  profit 18 = 156 := sorry

end max_profit_achieved_at_18_thousand_toys_l500_500373


namespace age_difference_l500_500961

noncomputable def years_older (A B : ℕ) : ℕ :=
A - B

theorem age_difference (A B : ℕ) (h1 : B = 39) (h2 : A + 10 = 2 * (B - 10)) :
  years_older A B = 9 :=
by
  rw [years_older]
  rw [h1] at h2
  sorry

end age_difference_l500_500961


namespace pipe_A_alone_fills_tank_in_28_hours_l500_500390

-- Definitions according to the conditions
def rate_A (A: ℝ) := A
def rate_B (A: ℝ) := 2 * rate_A A
def rate_C (A: ℝ) := 2 * rate_B A
def combined_rate (A: ℝ) := rate_A A + rate_B A + rate_C A
def time_to_fill_tank (A: ℝ) := 1 / combined_rate A

theorem pipe_A_alone_fills_tank_in_28_hours (A : ℝ) (h : combined_rate A = 1 / 4) : 1 / rate_A A = 28 := by
  sorry

end pipe_A_alone_fills_tank_in_28_hours_l500_500390


namespace construct_using_five_twos_l500_500345

theorem construct_using_five_twos :
  (∃ (a b c d e f : ℕ), (22 * (a / b)) / c = 11 ∧
                        (22 / d) + (e / f) = 12 ∧
                        (22 + g + h) / i = 13 ∧
                        (2 * 2 * 2 * 2 - j) = 14 ∧
                        (22 / k) + (2 * 2) = 15) := by
  sorry

end construct_using_five_twos_l500_500345


namespace problem_l500_500092

-- Define the operation otimes as described
def otimes (a b c : ℝ) : ℝ := a / (b - c)

-- Declare the variables and conditions
variables (a b c d e f : ℝ) 
theorem problem (h₁ : b ≠ c) (h₂ : e ≠ f) (h₃ : d ≠ e) :
  otimes (otimes 1 3 4) (otimes 2 4 3) (otimes 4 3 2) = 1 / 2 :=
by sorry

end problem_l500_500092


namespace sum_of_integers_l500_500612

variable (x y : ℕ)

theorem sum_of_integers (h1 : x - y = 10) (h2 : x * y = 56) : x + y = 18 := 
by 
  sorry

end sum_of_integers_l500_500612


namespace triangle_height_l500_500270

theorem triangle_height (base height area : ℝ) 
(h_base : base = 3) (h_area : area = 9) 
(h_area_eq : area = (base * height) / 2) :
  height = 6 := 
by 
  sorry

end triangle_height_l500_500270


namespace intervals_of_monotonicity_and_extremum_range_of_a_for_distinct_roots_log_inequality_l500_500834

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.log x

theorem intervals_of_monotonicity_and_extremum (a : ℝ) : 
  let f' x := (2 * a * x^2 - 1) / x in 
  (a ≤ 0 → (∀ x > 0, f' x < 0)) ∧
  (a > 0 → (
    ∀ x > 0, 
      (x ≥ (Real.sqrt (2 * a) / (2 * a))) → f' x > 0 ∧ 
      (x < (Real.sqrt (2 * a) / (2 * a))) → f' x < 0 ∧ 
      (∃ x₀, x₀ = (Real.sqrt (2 * a) / (2 * a)) ∧ f a x₀ = (1 / 2) * (1 + Real.log (2 * a)))))

theorem range_of_a_for_distinct_roots {a k : ℝ} (a : ℝ) :
  (1 / (2 * Real.exp 2) < a) ∧ (a < Real.exp 2 / 2) → 
  ∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (f a x₁ - k = 0) ∧ (f a x₂ - k = 0) ∧ (1 / Real.exp 1 ≤ x₁ ∧ x₁ ≤ Real.exp 1) ∧ (1 / Real.exp 1 ≤ x₂ ∧ x₂ ≤ Real.exp 1))

noncomputable def g (x : ℝ) : ℝ := Real.log x / x^2

theorem log_inequality : 
  (∑ i in Finset.range (n+1-2) .map (λ x, x + 2), g i) < 1 / (2 * Real.exp 1) :=
begin
  sorry
end

end intervals_of_monotonicity_and_extremum_range_of_a_for_distinct_roots_log_inequality_l500_500834


namespace certain_number_correct_l500_500955

def k := 4
def coefficient := 0.0010101
def certainNumber := 10.101

theorem certain_number_correct (k_is_integer : k ∈ Int) (least_possible_value : 4.9956356288922485 > k) : 
  coefficient * 10 ^ k = certainNumber := by
  sorry

end certain_number_correct_l500_500955


namespace pyramid_volume_l500_500713

noncomputable def volume_of_pyramid (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) : ℝ :=
  let base_area := base_length * base_width
  let diagonal_length := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal_length / 2)^2)
  (1 / 3) * base_area * height

theorem pyramid_volume : volume_of_pyramid 7 9 15 = 21 * Real.sqrt 192.5 := by
  sorry

end pyramid_volume_l500_500713


namespace ratio_of_triangle_areas_l500_500334

theorem ratio_of_triangle_areas (a k : ℝ) (h_pos_a : 0 < a) (h_pos_k : 0 < k)
    (h_triangle_division : true) (h_square_area : ∃ s, s = a^2) (h_area_one_triangle : ∃ t, t = k * a^2) :
    ∃ r, r = (1 / (4 * k)) :=
by
  sorry

end ratio_of_triangle_areas_l500_500334


namespace train_length_proof_l500_500368

-- Conditions as definitions
def length_first_train : ℝ := 200
def speed_first_train_kph : ℝ := 120
def speed_second_train_kph : ℝ := 80
def crossing_time_seconds : ℝ := 9

-- Conversion from km/h to m/s
def speed_conversion (speed_kph : ℝ) : ℝ := speed_kph * 1000 / 3600
def speed_first_train := speed_conversion speed_first_train_kph
def speed_second_train := speed_conversion speed_second_train_kph

-- Relative speed and total distance covered during crossing
def relative_speed : ℝ := speed_first_train + speed_second_train
def total_distance : ℝ := relative_speed * crossing_time_seconds

-- Lean statement to prove the length of the second train
theorem train_length_proof : ∃ L : ℝ, 200 + L = 499.95 := 
begin
  use total_distance - length_first_train,
  sorry,
end

end train_length_proof_l500_500368


namespace inv_proportion_through_point_l500_500528

theorem inv_proportion_through_point (m : ℝ) (x y : ℝ) (h1 : y = m / x) (h2 : x = 2) (h3 : y = -3) : m = -6 := by
  sorry

end inv_proportion_through_point_l500_500528


namespace prime_count_60_to_70_l500_500148

theorem prime_count_60_to_70 : ∃ primes : Finset ℕ, primes.card = 2 ∧ ∀ p ∈ primes, 60 < p ∧ p < 70 ∧ Nat.Prime p :=
by
  sorry

end prime_count_60_to_70_l500_500148


namespace perimeter_ratio_l500_500560

theorem perimeter_ratio (s : ℝ) :
  let ABC := s in
  let AB := s in
  let BC := s in
  let CA := s in
  let BB' := 2 * AB in
  let AB' := AB + BB' in
  let CC' := 4 * BC in
  let BC' := BC + CC' in
  let AA' := 6 * CA in
  let CA' := CA + AA' in
  let perimeter_ABC := AB + BC + CA in
  let perimeter_A'B'C' := AB' + BC' + CA' in
  (perimeter_A'B'C' / perimeter_ABC) = 5 :=
by
  sorry

end perimeter_ratio_l500_500560


namespace quadratic_eq_of_sum_and_product_l500_500122

theorem quadratic_eq_of_sum_and_product (a b c : ℝ) (h_sum : -b / a = 4) (h_product : c / a = 3) :
    ∀ (x : ℝ), a * x^2 + b * x + c = a * x^2 - 4 * a * x + 3 * a :=
by
  sorry

end quadratic_eq_of_sum_and_product_l500_500122


namespace plane_equation_l500_500618

theorem plane_equation
  (x y z: ℝ)
  (h1: (x, y, z) = (10, -2, 5))
  (h2: ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1
    ∧ (A: ℝ) * x + (B: ℝ) * y + (C: ℝ) * z + D = 0):
  h2 → (∃ (A B C D: ℤ), A = 10 ∧ B = -2 ∧ C = 5 ∧ D = -129 ∧ (A: ℝ) * x + (B: ℝ) * y + (C: ℝ) * z + D = 0) :=
by
  sorry

end plane_equation_l500_500618


namespace eccentricity_proof_slope_proof_l500_500804

variable {a b x0 y0 k : ℝ}

axiom ellipse_eq : a > b → b > 0 → x0^2 / a^2 + y0^2 / b^2 = 1 → 
                   let kAP := y0 / (x0 + a) in
                   let kBP := y0 / (x0 - a) in 
                   kAP * kBP = -1 / 2 → 
                   (a^2 = 2 * b^2)

axiom slope_condition : a > b → b > 0 → x0 / a^2 + k^2 * x0^2 / b^2 = 1 → 
                         | x0 + a | = a → 
                         | k | > sqrt 3

theorem eccentricity_proof : 
  ∀ a b : ℝ, 
  ∀ x0 y0 : ℝ,
  a > b → 
  b > 0 → 
  x0^2 / a^2 + y0^2 / b^2 = 1 → 
  let kAP := y0 / (x0 + a) in
  let kBP := y0 / (x0 - a) in 
  kAP * kBP = -1 / 2 → 
  (a^2 = 2 * b^2) → 
  sqrt (a^2 - b^2) / a = sqrt(2) / 2 :=
by
sorry

theorem slope_proof :
  ∀ a b : ℝ,
  ∀ x0 k : ℝ,
  a > b → 
  b > 0 → 
  x0 / a^2 + k^2 * x0^2 / b^2 = 1 → 
  | x0 + a | = a → 
  | k | > sqrt 3 :=
by
sorry

end eccentricity_proof_slope_proof_l500_500804


namespace midline_problem_l500_500027

variables {A B C M N P D E : Type}
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables (MN : A) (ABC : B) (P : C) (BP : B) (CP : C) (AC : D) (AB : E) 

theorem midline_problem 
  (h1: IsMidline MN ABC) 
  (h2: OnLine P MN) 
  (h3: Intersects BP P AC D) 
  (h4: Intersects CP P AB E) : 
  (AD / DC + AE / EB = 1) := 
sorry

end midline_problem_l500_500027


namespace polygon_edge_sum_impossible_l500_500641

theorem polygon_edge_sum_impossible : 
  ∀ (polygons : fin 99 → fin 101 → nat) (flipped : fin 99 → bool),
    (∀ (i : fin 99) (j j' : fin 101), polygons i j = if flipped i then 101 - polygons i j' else polygons i j') → 
    ¬ (∃ s : nat, ∀ (i : fin 99) (j : fin 101), polygons i j + polygons i ((j + 1) % 101) = s) := 
by sorry

end polygon_edge_sum_impossible_l500_500641


namespace soft_drink_company_bottle_count_l500_500389

theorem soft_drink_company_bottle_count
  (B : ℕ)
  (initial_small_bottles : ℕ := 6000)
  (percent_sold_small : ℝ := 0.12)
  (percent_sold_big : ℝ := 0.14)
  (bottles_remaining_total : ℕ := 18180) :
  (initial_small_bottles * (1 - percent_sold_small) + B * (1 - percent_sold_big) = bottles_remaining_total) → B = 15000 :=
by
  sorry

end soft_drink_company_bottle_count_l500_500389


namespace new_percentage_of_water_l500_500436

noncomputable def initial_weight : ℝ := 100
noncomputable def initial_percentage_water : ℝ := 99 / 100
noncomputable def initial_weight_water : ℝ := initial_weight * initial_percentage_water
noncomputable def initial_weight_non_water : ℝ := initial_weight - initial_weight_water
noncomputable def new_weight : ℝ := 25

theorem new_percentage_of_water :
  ((new_weight - initial_weight_non_water) / new_weight) * 100 = 96 :=
by
  sorry

end new_percentage_of_water_l500_500436


namespace sum_of_first_nine_terms_l500_500495

variable {a : ℕ → ℝ} -- a_n is the geometric sequence

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

def S (n : ℕ) : ℝ := ∑ i in Finset.range(n+1), a i

theorem sum_of_first_nine_terms 
  (h : is_geometric_sequence a)
  (h1 : a 0 + a 1 + a 2 = 2)
  (h2 : a 3 + a 4 + a 5 = 6) :
  S 8 = 26 :=
by 
  sorry

end sum_of_first_nine_terms_l500_500495


namespace probability_even_product_l500_500601

theorem probability_even_product :
  let S := finset.range 15 in
  let total_pairs := S.choose 2 in
  let even_numbers := S.filter (λ x, x % 2 = 0) in
  let odd_numbers := S.filter (λ x, x % 2 = 1) in
  let odd_pairs := odd_numbers.choose 2 in
  let even_or_mixed_pairs := total_pairs.card - odd_pairs.card in
    even_or_mixed_pairs = 77 ∧ total_pairs.card = 105 →
    (even_or_mixed_pairs / total_pairs.card : ℚ) = 77 / 105 := 
by 
  sorry

end probability_even_product_l500_500601


namespace different_ways_to_paint_fence_l500_500347

def num_ways_to_paint_fence (n : ℕ) : ℕ :=
  if n = 1 then 3
  else if n = 2 then 6
  else 3 * 2^(n-1) - 6 * ((n % 2 = 0) + (n % 2 = 1))

theorem different_ways_to_paint_fence (n : ℕ) (h : n = 10) :
  num_ways_to_paint_fence n = 1530 :=
by
  have h1 : num_ways_to_paint_fence 10 = 1530 := sorry
  exact h1

end different_ways_to_paint_fence_l500_500347


namespace tan_alpha_solution_l500_500873

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l500_500873


namespace tan_alpha_value_l500_500886

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l500_500886


namespace pyramid_volume_84sqrt10_l500_500715

noncomputable def volume_of_pyramid (a b c : ℝ) : ℝ :=
  (1/3) * a * b * c

theorem pyramid_volume_84sqrt10 :
  let height := 4 * (Real.sqrt 10)
  let area_base := 7 * 9
  (volume_of_pyramid area_base height) = 84 * (Real.sqrt 10) :=
by
  intros
  simp [volume_of_pyramid]
  sorry

end pyramid_volume_84sqrt10_l500_500715


namespace james_eats_4_slices_l500_500198

def total_slices : ℕ := 20
def tom_slices : ℕ := 5
def alice_slices : ℕ := 3
def bob_slices : ℕ := 4
def total_friends_slices : ℕ := tom_slices + alice_slices + bob_slices
def remaining_slices : ℕ := total_slices - total_friends_slices
def james_slices : ℕ := remaining_slices / 2

theorem james_eats_4_slices : james_slices = 4 := by
  dsimp [james_slices, remaining_slices, total_friends_slices, bob_slices, alice_slices, tom_slices, total_slices]
  norm_num

end james_eats_4_slices_l500_500198


namespace original_words_l500_500338

def words_learned_per_day := 10
def days_in_a_year := 365
def total_years := 2
def words_learned_in_years := words_learned_per_day * days_in_a_year * total_years
def increased_by := 0.50
def total_vocabulary_increased := words_learned_in_years + words_learned_in_years * increased_by

theorem original_words : ∃ W_initial : ℕ, W_initial = total_vocabulary_increased / (1 + increased_by) := 
sorry

end original_words_l500_500338


namespace tan_alpha_value_l500_500935

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l500_500935


namespace trapezoid_area_calculation_l500_500396

def is_isosceles_trapezoid (a b c d : ℝ) (l d1 d2 b1 : ℝ) : Prop :=
  l = 20 ∧ 
  d1 = 25 ∧ d2 = 25 ∧
  b1 = 30 ∧ 
  a = l ∧ c = l ∧ (a + b + c + d) = (2 * l + b1 + d)

noncomputable def trapezoid_area (b1 b2 h : ℝ) : ℝ :=
  0.5 * (b1 + b2) * h

theorem trapezoid_area_calculation : 
  ∀ (a b c d l d1 d2 b1 b2 height : ℝ),
  is_isosceles_trapezoid a b c d l d1 d2 b1 →   
  b1 = 30 →
  l = 20 →
  d1 = 25 →
  d2 = 25 →
  height = (20 * 25 / 30) →
  b2 = 30 - 2 * (real.sqrt (l^2 - height^2)) →
  trapezoid_area b1 b2 height = 318.93 :=
by 
  intros a b c d l d1 d2 b1 b2 height Hiso Hb1 Hl Hd1 Hd2 Hheight Hb2
  sorry

end trapezoid_area_calculation_l500_500396


namespace congruence_solution_count_l500_500491

theorem congruence_solution_count :
  ∃! x : ℕ, x < 50 ∧ x + 20 ≡ 75 [MOD 43] := 
by
  sorry

end congruence_solution_count_l500_500491


namespace rhombus_problem_l500_500175

variables (A B C D M N : Type) [metric_space A]

variables [add_group A] [vector_space ℝ A] [inner_product_space ℝ A]

def rhombus (A B C D : A) := 
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A 

def midpoint (X Y M : A) := dist X M = dist Y M ∧ 2 • M = X + Y

theorem rhombus_problem (A B C D M N : A)
  (h_rhombus : rhombus A B C D)
  (h_midpoint_AB : midpoint A B M)
  (h_midpoint_AD : midpoint A D N)
  (h_CN : dist C N = 7)
  (h_DM : dist D M = 24) :
  (dist A B)^2 = 262 := sorry

end rhombus_problem_l500_500175


namespace greg_in_seat_2_l500_500769

variable (seating : Fin 4 → Option String)

-- Definitions based on conditions
def fiona_sitting_in_seat_4 (h: seating 3 = some "Fiona") : Prop := True

def fiona_next_to_hank_false (h: seating 3 ≠ some "Fiona" → 
  (∀ i : Fin 4, seating i = some "Hank" → (i ≠ 2) ∧ (i ≠ 3))) : Prop := True

def ella_between_fiona_and_hank_false (h: 
  ∀ i j k : Fin 4, seating i = some "Ella" → seating j = some "Fiona" → seating k = some "Hank" 
  → ¬ ((i < j ∧ j < k) ∨ (k < j ∧ j < i))) : Prop := True

-- Mathematical statement to prove
theorem greg_in_seat_2 (h1 : fiona_sitting_in_seat_4 seating) 
                       (h2 : fiona_next_to_hank_false seating) 
                       (h3 : ella_between_fiona_and_hank_false seating) :
                       seating 1 = some "Greg" := 
sorry -- Proof of the statement goes here

noncomputable def seating : Fin 4 → Option String
| 0 => some "Hank"
| 1 => some "Greg"
| 2 => some "Ella"
| 3 => some "Fiona"

end greg_in_seat_2_l500_500769


namespace total_time_correct_l500_500675

-- Defining the parameters given in the conditions
def speed_boat_standing_water : ℝ := 9 -- in kmph
def speed_stream : ℝ := 6 -- in kmph
def distance : ℝ := 210 -- in km

-- Defining the downstream and upstream speeds
def downstream_speed : ℝ := speed_boat_standing_water + speed_stream
def upstream_speed : ℝ := speed_boat_standing_water - speed_stream

-- Defining the time taken to row downstream and upstream
def time_downstream : ℝ := distance / downstream_speed
def time_upstream : ℝ := distance / upstream_speed

-- Defining the total time taken
def total_time : ℝ := time_downstream + time_upstream

-- The statement to prove
theorem total_time_correct : total_time = 84 := by
  sorry

end total_time_correct_l500_500675


namespace isabella_renovation_l500_500552

def isabella_walls_paint_area (length width height door_window_area : ℕ) (num_bedrooms : ℕ) : ℕ :=
  let wall_area := 2 * (length * height) + 2 (width * height) - door_window_area
  wall_area * num_bedrooms

def isabella_carpet_area (length width fixed_furniture_area : ℕ) (num_bedrooms : ℕ) : ℕ :=
  let floor_area := length * width - fixed_furniture_area
  floor_area * num_bedrooms

theorem isabella_renovation :
  isabella_walls_paint_area 14 11 9 70 4 = 1520 ∧ 
  isabella_carpet_area 14 11 24 4 = 520 :=
by
  sorry

end isabella_renovation_l500_500552


namespace A_and_D_independent_l500_500310

-- Define the probabilities of elementary events
def prob_A : ℚ := 1 / 6
def prob_B : ℚ := 1 / 6
def prob_C : ℚ := 5 / 36
def prob_D : ℚ := 1 / 6

-- Define the joint probability of A and D
def prob_A_and_D : ℚ := 1 / 36

-- Define the independence condition
def independent (P_X P_Y P_XY : ℚ) : Prop := P_XY = P_X * P_Y

-- Prove that events A and D are independent
theorem A_and_D_independent : 
  independent prob_A prob_D prob_A_and_D := by
  -- The proof is skipped
  sorry

end A_and_D_independent_l500_500310


namespace tan_alpha_value_l500_500889

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l500_500889


namespace unique_solution_theorem_l500_500788

noncomputable def unique_solution_values : Prop :=
  ∀ (a : ℝ), 
    (∀ (x y z : ℝ), x^2 + y^2 + z^2 + 2 * y = 0 → x + a * y + a * z - a = 0) ↔ 
    (a = sqrt 2 / 2 ∨ a = -sqrt 2 / 2)

theorem unique_solution_theorem : unique_solution_values := 
  sorry

end unique_solution_theorem_l500_500788


namespace original_words_l500_500337

def words_learned_per_day := 10
def days_in_a_year := 365
def total_years := 2
def words_learned_in_years := words_learned_per_day * days_in_a_year * total_years
def increased_by := 0.50
def total_vocabulary_increased := words_learned_in_years + words_learned_in_years * increased_by

theorem original_words : ∃ W_initial : ℕ, W_initial = total_vocabulary_increased / (1 + increased_by) := 
sorry

end original_words_l500_500337


namespace maximum_points_on_opposite_faces_l500_500581

def points_on_face (n : ℕ) : ℕ := 5 + n

def opposite_faces_max_points : ℕ :=
  points_on_face 5 + points_on_face 3

theorem maximum_points_on_opposite_faces 
  (points_on_face : ℕ → ℕ)
  (opposite_faces_max_points = 18)
  : 5 ≤ points_on_face 0 ∧ points_on_face 0 + 5 = points_on_face 5
  ∧ points_on_face 0 + 4 = points_on_face 4
  ∧ points_on_face 0 + 3 = points_on_face 3
  ∧ points_on_face 0 + 2 = points_on_face 2
  ∧ points_on_face 0 + 1 = points_on_face 1 := sorry

end maximum_points_on_opposite_faces_l500_500581


namespace AH_perpendicular_BP_l500_500170

open_locale classical
noncomputable theory

variables {A B C H M P : Type}

def is_midpoint (M : Type) (A C : Type) : Prop := 
  ∃ (midpoint_val : Type), midpoint_val = M

def is_perpendicular (L1 L2 : Type) : Prop := 
  ∃ (perpendicular_val : Type), perpendicular_val = L1 ∧ perpendicular_val = L2

def is_triangle_ABC (A B C : Type) : Prop :=
  ∃ (triangle_val : Type), triangle_val = (A, B, C)

/-- Given an isosceles triangle ABC with AB = BC, the midpoint M of AC, point H on BC such that MH
    is perpendicular to BC, and point P the midpoint of MH, we prove that AH is perpendicular to BP. -/
theorem AH_perpendicular_BP 
  (A B C H M P : Type)
  (h_isosceles : A = B ∧ B = C)
  (h_mid_AC : is_midpoint M A C)
  (h_H_on_BC : ∃ (intersection_val : Type), intersection_val = H ∧ H = B ∧ H = C)
  (h_MH_perpendicular_BC : is_perpendicular M H)
  (h_mid_MH : is_midpoint P M H) :
  is_perpendicular A H ∧ B P :=
sorry

end AH_perpendicular_BP_l500_500170


namespace circle_STX_passes_through_midpoint_BC_l500_500358

open Set Classical

variables {P A B C E F T S X : Type} [Point P]
variables {BP CP AC AB : Line}
variables {AP : Line}
variables (circle_ABC : Circle ABC)
variables (circle_AEF : Circle AEF)

-- Assume that P is a point inside triangle ABC
axiom P_inside_ABC :
  ∃ (P : Point), P ∉ [A, B, C] ∧ P ∉ Line [A, B] ∧ P ∉ Line [B, C] ∧ P ∉ Line [C, A]

-- Assume BP and CP intersect AC and AB at E and F, respectively
axiom BP_intersect_AC_at_E : Line_intersection BP AC = E
axiom CP_intersect_AB_at_F : Line_intersection CP AB = F

-- Assume AP intersects circle_ABC again at X
axiom AP_intersect_circle_ABC_at_X : Circle_intersection AP circle_ABC = X

-- Assume circle_ABC and circle_AEF intersect again at S
axiom circle_ABC_and_circle_AEF_intersect_at_S : Circle_intersection circle_ABC circle_AEF = S

-- Assume T is a point on BC such that PT || EF
axiom T_on_BC_and_PT_parallel_EF :
 ∃ (T : Point), T ∈ Line [B, C] ∧ Line_parallel PT Line [E, F]

-- Define the midpoint M of BC
def midpoint_BC : Point := midpoint B C

-- The theorem that needs to be proved
theorem circle_STX_passes_through_midpoint_BC :
  Midpoint_BC ∈ Circle [S, T, X] := 
sorry

end circle_STX_passes_through_midpoint_BC_l500_500358


namespace fraction_of_seats_sold_l500_500383

theorem fraction_of_seats_sold
  (ticket_price : ℕ) (number_of_rows : ℕ) (seats_per_row : ℕ) (total_earnings : ℕ)
  (h1 : ticket_price = 10)
  (h2 : number_of_rows = 20)
  (h3 : seats_per_row = 10)
  (h4 : total_earnings = 1500) :
  (total_earnings / ticket_price : ℕ) / (number_of_rows * seats_per_row : ℕ) = 3 / 4 := by
  sorry

end fraction_of_seats_sold_l500_500383


namespace line_equation_parallel_l500_500076

theorem line_equation_parallel (x₁ y₁ m : ℝ) (h₁ : (x₁, y₁) = (1, -2)) (h₂ : m = 2) :
  ∃ a b c : ℝ, a * x₁ + b * y₁ + c = 0 ∧ a * 2 + b * 1 + c = 4 := by
sorry

end line_equation_parallel_l500_500076


namespace tan_alpha_sqrt_15_over_15_l500_500914

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l500_500914


namespace most_suitable_sampling_method_l500_500681

/-- A unit has 28 elderly people, 54 middle-aged people, and 81 young people. 
    A sample of 36 people needs to be drawn in a way that accounts for age.
    The most suitable method for drawing a sample is to exclude one elderly person first,
    then use stratified sampling. -/
theorem most_suitable_sampling_method 
  (elderly : ℕ) (middle_aged : ℕ) (young : ℕ) (sample_size : ℕ) (suitable_method : String)
  (condition1 : elderly = 28) 
  (condition2 : middle_aged = 54) 
  (condition3 : young = 81) 
  (condition4 : sample_size = 36) 
  (condition5 : suitable_method = "Exclude one elderly person first, then stratify sampling") : 
  suitable_method = "Exclude one elderly person first, then stratify sampling" := 
by sorry

end most_suitable_sampling_method_l500_500681


namespace inequality_min_m_l500_500132

theorem inequality_min_m (m : ℝ) (x : ℝ) (hx : 1 < x) : 
  x + m * Real.log x + 1 / Real.exp x ≥ Real.exp (m * Real.log x) :=
sorry

end inequality_min_m_l500_500132


namespace probability_blue_before_green_l500_500687

open Finset Nat

noncomputable def num_arrangements : ℕ := choose 9 2  -- total number of ways to arrange the chips

noncomputable def num_favorable_arrangements : ℕ := 2 * (choose 7 4)  -- favorable arrangements where blue chips are among the first 8

noncomputable def probability_all_blue_before_green : ℚ := num_favorable_arrangements / num_arrangements

theorem probability_blue_before_green : probability_all_blue_before_green = 17 / 36 :=
sorry

end probability_blue_before_green_l500_500687


namespace mary_regular_hours_l500_500232

theorem mary_regular_hours (x y : ℕ) :
  8 * x + 10 * y = 760 ∧ x + y = 80 → x = 20 :=
by
  intro h
  sorry

end mary_regular_hours_l500_500232


namespace coloring_circle_sectors_l500_500058

theorem coloring_circle_sectors (n m : ℕ) (hn : 2 ≤ n) (hm : 2 ≤ m) :
  let a_n := (m - 1) ^ n + (-1) ^ n * (m - 1)
  a_n = (m - 1) ^ n + (-1) ^ n * (m - 1) :=
by
  sorry

end coloring_circle_sectors_l500_500058


namespace find_angle_A_max_area_l500_500548

variables (A B C a b c : ℝ) (triangle_ABC : Triangle a b c)

axiom condition1 : c = a * Real.cos B + b * Real.sin A

-- Prove that angle A is π/4
theorem find_angle_A (h : c = a * Real.cos B + b * Real.sin A) : A = Real.pi / 4 :=
sorry

-- Prove the maximum area of the triangle when a = 2
theorem max_area (h : c = 2 * Real.cos B + b * Real.sin A) (ha : a = 2) : 
  (1/2) * b * c * Real.sin A ≤ sqrt 2 + 1 :=
sorry

end find_angle_A_max_area_l500_500548


namespace distance_sum_l500_500842

-- Define the polar equation of C
def polar_curve (ρ θ : ℝ) : Prop := ρ - 4 * cos θ + 3 * ρ * sin θ^2 = 0

-- Cartesian equation from the polar definition
def cartesian_curve (x y : ℝ) : Prop := (x - 2)^2 + 4 * y^2 = 4

-- Define the parametric equation of line l
def parametric_line (t : ℝ) : ℝ × ℝ := (1 + (sqrt 3 / 2) * t, (1 / 2) * t)

-- Define the transformed curve C'
def transformed_curve (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Let |MA| + |MB| be the distance sum for intersection points A and B
theorem distance_sum : 
  let C_param (t : ℝ) := parametric_line t
  ∃ t1 t2 : ℝ, transformed_curve (C_param t1).1 (C_param t1).2 ∧ transformed_curve (C_param t2).1 (C_param t2).2 ∧ 
  abs t1 + abs t2 = sqrt 15 :=
begin
  sorry
end

end distance_sum_l500_500842


namespace A_and_D_independent_l500_500320

-- Definitions of the events based on given conditions
def event_A (x₁ : ℕ) : Prop := x₁ = 1
def event_B (x₂ : ℕ) : Prop := x₂ = 2
def event_C (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 8
def event_D (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 7

-- Probabilities based on uniform distribution and replacement
def probability_event (event : ℕ → ℕ → Prop) : ℚ :=
  if h : ∃ x₁ : ℕ, ∃ x₂ : ℕ, x₁ ∈ finset.range 1 7 ∧ x₂ ∈ finset.range 1 7 ∧ event x₁ x₂
  then ((finset.card (finset.filter (λ x, event x.1 x.2)
                (finset.product (finset.range 1 7) (finset.range 1 7)))) : ℚ) / 36
  else 0

noncomputable def P_A : ℚ := 1 / 6
noncomputable def P_D : ℚ := 1 / 6
noncomputable def P_A_and_D : ℚ := 1 / 36

-- Independence condition (by definition): P(A ∩ D) = P(A) * P(D)
theorem A_and_D_independent :
  P_A_and_D = P_A * P_D := by
  sorry

end A_and_D_independent_l500_500320


namespace find_angle_A_l500_500980

variable {A B C a b c : ℝ}
variable {triangle_ABC : Prop}

theorem find_angle_A
  (h1 : a^2 + c^2 = b^2 + 2 * a * c * Real.cos C)
  (h2 : a = 2 * b * Real.sin A)
  (h3 : Real.cos B = Real.cos C)
  (h_triangle_angles : triangle_ABC) : A = 2 * Real.pi / 3 := 
by
  sorry

end find_angle_A_l500_500980


namespace tan_alpha_fraction_l500_500905

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l500_500905


namespace distance_AD_proof_l500_500181

-- Defining the conditions of the problem
def equilateral_triangle (ABC : Triangle) : Prop :=
  ABC.AB = 6 ∧ ABC.AC = 6 ∧ ABC.BC = 6

def isosceles_triangle (BCD : Triangle) : Prop :=
  BCD.BD = 9 ∧ BCD.DC = 9

-- Assuming necessary geometric properties and distances
def distance_AD (A B C D : Point) (ABC : Triangle) (BCD : Triangle) :=
  has_distance A D (6 * √2 + 3 * √3)

-- Main statement to prove using the above conditions
theorem distance_AD_proof :
  ∀ (A B C D : Point) (ABC : Triangle) (BCD : Triangle),
    equilateral_triangle ABC →
    isosceles_triangle BCD →
    collinear B C D →  -- D is aligned with BC
    ∃ h : ℝ, h = (6 * √2 + 3 * √3) ∧
    distance_AD A B C D ABC BCD :=
by
  intros A B C D ABC BCD hABC hBCD hCollinear
  use 6 * √2 + 3 * √3
  split
  sorry

end distance_AD_proof_l500_500181


namespace tan_alpha_value_l500_500942

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l500_500942


namespace tan_alpha_value_l500_500888

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l500_500888


namespace area_of_triangle_ABC_l500_500534

open Real

theorem area_of_triangle_ABC (B : ℝ) (AC BC : ℝ) (hB : B = π / 3) (hAC : AC = 2 * sqrt 3) (hBC : BC = 4) : 
  let S : ℝ := 1 / 2 * AC * (BC / 2) in
  S = 2 * sqrt 3 :=
by
  sorry

end area_of_triangle_ABC_l500_500534


namespace find_a2_a3_arithmetic_seq_sum_first_n_terms_l500_500134

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 / 2 ∧ ∀ n > 1, (n + 1) * a n = 1 - 1 / (n * a (n - 1) + 1)

theorem find_a2_a3 (a : ℕ → ℝ) (h_seq : seq a) :
  a 2 = 1/6 ∧ a 3 = 1/12 :=
sorry

theorem arithmetic_seq (a : ℕ → ℝ) (h_seq : seq a) :
  ∃ (b : ℕ → ℝ), (∀ n > 0, b (n + 1) = b n + 1)
  ∧ (∀ n > 0, b n = 1 / ((n + 1) * a n)) :=
sorry

theorem sum_first_n_terms (a : ℕ → ℝ) (h_seq : seq a) (S : ℕ → ℝ) :
  (∀ n, S n = ∑ i in finset.range n, a (i+1)) → (∀ n, S n = n / (n + 1)) :=
sorry

end find_a2_a3_arithmetic_seq_sum_first_n_terms_l500_500134


namespace petya_guarantees_win_l500_500242

-- Definitions based on the conditions
def initial_sequence : list ℕ := [1,1,2,2,3,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,7]

-- Function to model a move
def erase_digits (seq : list ℕ) (digit : ℕ) : list ℕ :=
  seq.filter (≠ digit)

-- Predicate to check if a player wins
def wins (seq : list ℕ) : Prop :=
  seq.empty

-- Predicate to check if Petya can guarantee a win
def can_petya_win (init_seq : list ℕ) : Prop :=
  ∀ vasya_move : ℕ, (∃ petya_move : ℕ, wins (erase_digits (erase_digits init_seq vasya_move) petya_move))

theorem petya_guarantees_win : can_petya_win initial_sequence :=
sorry

end petya_guarantees_win_l500_500242


namespace piles_with_one_stone_invariant_l500_500719

theorem piles_with_one_stone_invariant :
  ∀ (piles : List ℕ),
  (piles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) →
  (∀ p q : List ℕ, ((∀ i j : ℕ, i ∈ p → i ≥ 2 ∧ j ∈ q → j ≥ 2 → p ++ [1] ++ q)) ∨
                    (∀ i : ℕ, i ∈ p → i ≥ 4 → (p.erase i) ++ [i-2, 2])) →
  (∃ k : ℕ, k = 23) :=
by
  sorry

end piles_with_one_stone_invariant_l500_500719


namespace range_of_g_l500_500781

noncomputable def g (x : ℝ) : ℝ :=
  (sin x ^ 3 + 8 * sin x ^ 2 + 2 * sin x + 2 * (cos x ^ 2) - 10) / (sin x - 1)

theorem range_of_g :
  ∀ x : ℝ, sin x ≠ 1 → 3 ≤ g x ∧ g x < 15 :=
by
  sorry

end range_of_g_l500_500781


namespace remaining_area_inside_large_square_l500_500003

theorem remaining_area_inside_large_square :
  let large_square_side := 3
      large_square_area := large_square_side * large_square_side
      small_square_side := 1
      small_square_area := small_square_side * small_square_side
      triangle_base := 1
      triangle_height := 3
      triangle_area := (1 / 2) * triangle_base * triangle_height
  in large_square_area - (small_square_area + triangle_area) = 6.5 :=
by
  let large_square_side := 3
  let large_square_area := large_square_side * large_square_side
  let small_square_side := 1
  let small_square_area := small_square_side * small_square_side
  let triangle_base := 1
  let triangle_height := 3
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  exact Eq.refl 6.5

end remaining_area_inside_large_square_l500_500003


namespace solution_set_inequality_l500_500422

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_inequality (f_cond : ∀ x : ℝ, f(x) + f'(x) > 1)
    (f_zero : f(0) = 4) : 
    ∀ x : ℝ, (e^x * f(x) > e^x + 3) ↔ x > 0 :=
sorry

end solution_set_inequality_l500_500422


namespace tan_alpha_value_l500_500937

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l500_500937


namespace double_series_evaluation_l500_500065

theorem double_series_evaluation :
    (∑' m : ℕ, ∑' n : ℕ, (1 : ℝ) / (m * n * (m + n + 2))) = (3 / 2 : ℝ) :=
sorry

end double_series_evaluation_l500_500065


namespace angle_between_vectors_l500_500213

open Real EuclideanGeometry

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Given conditions
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 2) (hapb_perp_a : inner (a + b) a = 0)

-- Statement to prove
theorem angle_between_vectors : angle a b = 2 * π / 3 :=
sorry

end angle_between_vectors_l500_500213


namespace tan_alpha_proof_l500_500902

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l500_500902


namespace determine_length_GH_l500_500547

namespace TrapezoidProof

def trapezoid_EFGH : Type :=
{EF GH : ℝ // EF = 7 ∧ FG = 4 * Real.sqrt 2 ∧ ∠FGH = 60 ∧ ∠GHE = 30 ∧ EF ∥ GH}

def length_GH (EF GH FG : ℝ) (∠FGH ∠GHE : ℝ) : ℝ :=
2 * Real.sqrt 2 + 7 + 2 * Real.sqrt 6

theorem determine_length_GH {EF GH FG : ℝ} {∠FGH ∠GHE : ℝ} (h : EF = 7) (k : FG = 4 * Real.sqrt 2) (l : ∠FGH = 60) (m : ∠GHE = 30):
  GH = length_GH EF GH FG ∠FGH ∠GHE :=
by 
  sorry

end TrapezoidProof

end determine_length_GH_l500_500547


namespace integral_log_eq_ln2_l500_500442

theorem integral_log_eq_ln2 :
  ∫ x in (0 : ℝ)..(1 : ℝ), (1 / (x + 1)) = Real.log 2 :=
by
  sorry

end integral_log_eq_ln2_l500_500442


namespace calc_result_l500_500412

-- Given expression
def expr : ℝ := (π - 3)^0 + 3^(-1) * (2 + 1/4)^(1/2)

-- Expected value
def expected : ℝ := 3 / 2

-- The theorem statement proving the expression equals the expected value
theorem calc_result : expr = expected := by
  sorry

end calc_result_l500_500412


namespace area_enclosed_by_abs_eq_l500_500779

theorem area_enclosed_by_abs_eq (f : ℝ × ℝ → ℝ) (g : ℝ × ℝ → ℝ) :
  (∀ x y, f (x,y) = |x-60| + |y| ∧ g (x,y) = |x/4|) →
  (region : set (ℝ × ℝ)) (h : region = {p : ℝ × ℝ | f p = g p}) →
  (area : ℝ) (h_area : area_of_region region area) →
  area = 480 :=
begin
  intros H region_def area_def,
  sorry
end

def area_of_region (region : set (ℝ × ℝ)) (area : ℝ) : Prop := sorry

end area_enclosed_by_abs_eq_l500_500779


namespace tan_alpha_sqrt_15_over_15_l500_500917

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l500_500917


namespace cosine_neg_alpha_l500_500098

theorem cosine_neg_alpha (alpha : ℝ) (h : Real.sin (π/2 + alpha) = -3/5) : Real.cos (-alpha) = -3/5 :=
sorry

end cosine_neg_alpha_l500_500098


namespace ratio_BC_MK_l500_500192

theorem ratio_BC_MK (A B C K M : Type*) 
  [Plane A] [Plane B] [Plane C] [Plane K] [Plane M]
  (h₁ : ∠ ACB = 60)
  (h₂ : ∠ ABC = 45)
  (h₃ : AC = CK)
  (h₄ : ∆ CMK ~ ∆ ABC)
  (h₅ : CM < MK) :
  BC / MK = (2 + sqrt 3) / sqrt 6 :=
sorry

end ratio_BC_MK_l500_500192


namespace shaded_region_area_l500_500185

noncomputable def regular_octagon_area := 1
constant A B C D E F G H : Type -- Vertices of the octagon
constant S : ℝ -- Area of the shaded region
constant A_C_E_G : Type -- Square formed by vertices A, C, E, and G
constant P_Q_R_S : Type -- Another square, nature deduced from symmetry and properties of regular polygons

axiom octagon_area_eq_one : regular_octagon_area = 1
axiom two_squares_in_octagon : True -- Hypothetical axiom representing the presence of two squares

theorem shaded_region_area : S = 1 / 2 :=
sorry -- Proof of the theorem based on the given conditions

end shaded_region_area_l500_500185


namespace dot_product_is_minus_2_l500_500967

-- Definitions
variables {A B C : Type}
variables (triangle_ABC : IsoscelesTriangle A B C)
variable (vertex_angle : ∠A = 2 * Real.pi / 3)
variable (base_length : BC = 2 * Real.sqrt 3)

-- The proof goal statement
theorem dot_product_is_minus_2 :
  ∃ (BA AC : Vector ℝ), (|BA| = 2) ∧ (|AC| = 2) ∧ (BA ∙ AC = -2) := sorry

end dot_product_is_minus_2_l500_500967


namespace tan_alpha_solution_l500_500856

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l500_500856


namespace eqdistant_points_perpendicular_bisector_l500_500849

open Complex

-- Define the conditions
variable (z z1 z2 : ℂ)
variable (h_distinct : z1 ≠ z2)
variable (h_condition : |z - z1| - |z - z2| = 0)

-- State the theorem
theorem eqdistant_points_perpendicular_bisector (h : |z - z1| = |z - z2|) :
    -- Define what it means for a point to be on the perpendicular bisector
    let mid_point := (z1 + z2) / 2
    let is_perpendicular_bisector := ∀ z, |z - z1| = |z - z2| → ↑(Re z) = ↑(Re mid_point)
    is_perpendicular_bisector :=
sorry

end eqdistant_points_perpendicular_bisector_l500_500849


namespace median_combined_list_l500_500753

theorem median_combined_list :
  let integers := (list.range 1000).map (+ 1) in
  let cubes := (list.range 1000).map (λ n, (n + 1) ^ 3) in
  let evens := (list.range 1000).map (λ n, 2 * (n + 1)) in
  let combined_list := integers ++ cubes ++ evens in
  let sorted_list := combined_list.qsort (≤) in
  (sorted_list.nth 1499 + sorted_list.nth 1500) / 2 = 500.5 :=
begin
  sorry
end

end median_combined_list_l500_500753


namespace value_of_y_l500_500526

noncomputable def angle : Real := -π / 3

noncomputable def P : Real × Real := (2, y)

noncomputable def tan_value := Real.tan angle

theorem value_of_y (y : Real) (h1 : tan_value = -√3) (h2 : P = (2, y)) : y = -2 * √3 := by
  sorry

end value_of_y_l500_500526


namespace compute_hemisphere_flux_l500_500752

noncomputable def vector_field_a (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (x^2, y^2, z^2)

def hemisphere_surface (x y z R : ℝ) : Prop :=
  x^2 + y^2 + z^2 = R^2 ∧ z ≥ 0

theorem compute_hemisphere_flux {R : ℝ} (hR : R ≥ 0) :
  ∀ x y z, hemisphere_surface x y z R →
  let div_a := 2 * x + 2 * y + 2 * z in
  let flux := ∫∫∫ (λ (x, y, z), div_a) in
  flux = (π * R^4) / 2 :=
sorry

end compute_hemisphere_flux_l500_500752


namespace f_odd_function_range_of_k_l500_500050

noncomputable def f : ℝ → ℝ := sorry

axiom h_monotonic : monotone f
axiom h_eq1 : f 3 = Real.log 3 / Real.log 2
axiom h_additivity : ∀ x y : ℝ, f (x + y) = f x + f y

theorem f_odd_function (x : ℝ) : f (-x) = -f x :=
by
  sorry

theorem range_of_k (k : ℝ) (h : ∀ x : ℝ, f (k * 3^x) + f (3^x - 9^x - 2) < 0) : k < 1 :=
by
  sorry

end f_odd_function_range_of_k_l500_500050


namespace coffee_y_ratio_is_1_to_5_l500_500364

-- Define the conditions
variables {p v x y : Type}
variables (p_x p_y v_x v_y : ℕ) -- Coffee amounts in lbs
variables (total_p total_v : ℕ) -- Total amounts of p and v

-- Definitions based on conditions
def coffee_amounts_initial (total_p total_v : ℕ) : Prop :=
  total_p = 24 ∧ total_v = 25

def coffee_x_conditions (p_x v_x : ℕ) : Prop :=
  p_x = 20 ∧ 4 * v_x = p_x

def coffee_y_conditions (p_y v_y total_p total_v : ℕ) : Prop :=
  p_y = total_p - 20 ∧ v_y = total_v - (20 / 4)

-- Statement to prove
theorem coffee_y_ratio_is_1_to_5 {total_p total_v : ℕ}
  (hc1 : coffee_amounts_initial total_p total_v)
  (hc2 : coffee_x_conditions 20 5)
  (hc3 : coffee_y_conditions 4 20 total_p total_v) : 
  (4 / 20 = 1 / 5) :=
sorry

end coffee_y_ratio_is_1_to_5_l500_500364


namespace equation_of_line_AC_l500_500549

-- Define the given points A and B
def A : (ℝ × ℝ) := (1, 1)
def B : (ℝ × ℝ) := (-3, -5)

-- Define the line m as a predicate
def line_m (p : ℝ × ℝ) : Prop := 2 * p.1 + p.2 + 6 = 0

-- Define the condition that line m is the angle bisector of ∠ACB
def is_angle_bisector (A B C : ℝ × ℝ) (m : (ℝ × ℝ) → Prop) : Prop := sorry

-- The symmetric point of B with respect to line m
def symmetric_point (B : ℝ × ℝ) (m : (ℝ × ℝ) → Prop) : (ℝ × ℝ) := sorry

-- Proof statement
theorem equation_of_line_AC :
  ∀ (A B : ℝ × ℝ) (m : (ℝ × ℝ) → Prop),
  A = (1, 1) →
  B = (-3, -5) →
  m = line_m →
  is_angle_bisector A B (symmetric_point B m) m →
  AC = {p : ℝ × ℝ | p.1 = 1} := sorry

end equation_of_line_AC_l500_500549


namespace prove_optimal_path_l500_500579

noncomputable def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def total_distance (path : list point) : ℝ :=
  (path.zip (path.tail)).sum (λ⟨a, b⟩, distance a b)

def A : point := (0, 428)
def B : point := (9, 85)
def C : point := (42, 865)
def D : point := (192, 875)
def E : point := (193, 219)
def F : point := (204, 108)
def G : point := (292, 219)
def H : point := (316, 378)
def I : point := (375, 688)
def J : point := (597, 498)
def K : point := (679, 766)
def L : point := (739, 641)
def M : point := (772, 307)
def N : point := (793, 0)

def optimal_path : list point :=
  [A, C, D, I, K, L, J, M, N, H, G, E, F, B, A]

def minimal_total_distance : ℝ :=
  total_distance optimal_path

theorem prove_optimal_path :
  ∃ path : list point,       -- There exists a path,
    path.head = A ∧           -- starting at point A,
    path.last = A ∧           -- ending at point A,
    ∀ pt ∈ path.tail,         -- visiting each point exactly once,
    pt ∈ [B, C, D, E, F, G, H, I, J, K, L, M, N] ∧
    total_distance path = minimal_total_distance -- with the minimal total distance.
:= 
begin
  use optimal_path,
  split, sorry, -- Proof of path starting at A
  split, sorry, -- Proof of path ending at A
  split, sorry, -- Proof that each point in the path is visited exactly once
  sorry,        -- Proof of the total minimal distance
end

end prove_optimal_path_l500_500579


namespace two_integer_solutions_iff_m_l500_500088

def op (p q : ℝ) : ℝ := p + q - p * q

theorem two_integer_solutions_iff_m (m : ℝ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ op 2 x1 > 0 ∧ op x1 3 ≤ m ∧ op 2 x2 > 0 ∧ op x2 3 ≤ m) ↔ 3 ≤ m ∧ m < 5 :=
by
  sorry

end two_integer_solutions_iff_m_l500_500088


namespace area_XPQ_l500_500193

open Real

variable {X Y Z P Q : Type}
variable [triangleXYZ : Triangle XYZ]
variable [pointP : Point P]
variable [pointQ : Point Q]

-- Conditions given in the problem
axiom condition_XY : Distance XY = 10
axiom condition_YZ : Distance YZ = 12
axiom condition_XZ : Distance XZ = 13
axiom condition_XP : Distance XP = 6
axiom condition_XQ : Distance XQ = 8
axiom P_on_XY : OnLine P XY
axiom Q_on_XZ : OnLine Q XZ

-- The statement of the proof problem
theorem area_XPQ : Area (Triangle XPQ) = 16.94 := by
  sorry

end area_XPQ_l500_500193


namespace num_wheels_in_parking_lot_l500_500537

theorem num_wheels_in_parking_lot:
  (num_cars num_bikes wheels_per_car wheels_per_bike : ℕ)
  (h1: num_cars = 14)
  (h2: num_bikes = 5)
  (h3: wheels_per_car = 4)
  (h4: wheels_per_bike = 2) :
  num_cars * wheels_per_car + num_bikes * wheels_per_bike = 66 :=
by
  sorry

end num_wheels_in_parking_lot_l500_500537


namespace equal_parts_l500_500048

-- Definitions based on problem conditions
structure Point :=
  (x : ℕ)
  (y : ℕ)

def is_square (grid : List (List ℕ)) : Prop :=
  List.all (fun row => List.length row = List.length grid) grid

def cut_lines (p1 p2 p3 p4 : Point) : Prop :=
  -- Defining cut lines from the specific coordinates conditions
  p1 = ⟨0, 4⟩ ∧ p2 = ⟨6, 2⟩ ∧ p3 = ⟨4, 6⟩ ∧ p4 = ⟨2, 0⟩

-- The theorem to prove the equal division
theorem equal_parts (grid : List (List ℕ)) 
  (h_square : is_square grid) 
  (p1 p2 p3 p4 : Point) 
  (h_cut : cut_lines p1 p2 p3 p4) : 
  ∃ (parts : List (List (List ℕ))), 
    parts.length = 3 ∧
    -- Each part should be congruent under rotation or translation
    sorry := 
begin
  -- Proof goes here
  sorry
end

end equal_parts_l500_500048


namespace final_selling_price_l500_500388

theorem final_selling_price 
    (A_cost_price : ℝ)
    (A_profit_percentage : ℝ)
    (B_profit_percentage : ℝ)
    (A_cost_price_eq : A_cost_price = 120)
    (A_profit_percentage_eq : A_profit_percentage = 50)
    (B_profit_percentage_eq : B_profit_percentage = 25) :
    let A_selling_price := A_cost_price + (A_profit_percentage / 100) * A_cost_price in
    let B_selling_price := A_selling_price + (B_profit_percentage / 100) * A_selling_price in
    B_selling_price = 225 := 
    by sorry

end final_selling_price_l500_500388


namespace parallel_lines_a_value_l500_500138

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * a * y - 5 = 0 ↔ (3 * a - 1) * x - a * y - 2 = 0) →
  (a = 0 ∨ a = -1 / 6) :=
by
  sorry

end parallel_lines_a_value_l500_500138


namespace rectangle_clear_area_l500_500250

theorem rectangle_clear_area (EF FG : ℝ)
  (radius_E radius_F radius_G radius_H : ℝ) : 
  EF = 4 → FG = 6 → 
  radius_E = 2 → radius_F = 3 → radius_G = 1.5 → radius_H = 2.5 → 
  abs ((EF * FG) - (π * radius_E^2 / 4 + π * radius_F^2 / 4 + π * radius_G^2 / 4 + π * radius_H^2 / 4)) - 7.14 < 0.5 :=
by sorry

end rectangle_clear_area_l500_500250


namespace total_hours_for_double_papers_l500_500683

/-- Definition for the total man-hours required to check the original set of exam papers by 4 men. -/
def original_man_hours (men days hours_per_day : ℕ) : ℕ := men * days * hours_per_day

/-- Given conditions -/
def total_man_hours_1 : ℕ := original_man_hours 4 8 5  -- 160 man-hours

def double_work_man_hours : ℕ := total_man_hours_1 * 2  -- 320 man-hours

/-- Lean 4 statement to prove -/
theorem total_hours_for_double_papers :
  let H := double_work_man_hours / (2 * 20) in
  2 * 20 * H = 320 :=
by
  let H := 320 / 40
  sorry

end total_hours_for_double_papers_l500_500683


namespace paint_pyramid_l500_500580

theorem paint_pyramid (colors : Finset ℕ) (n : ℕ) (h : colors.card = 5) :
  let ways_to_paint := 5 * 4 * 3 * 2 * 1
  n = ways_to_paint
:=
sorry

end paint_pyramid_l500_500580


namespace exists_acute_triangle_l500_500070

theorem exists_acute_triangle (n : ℕ) (h_n : n ≥ 13) 
  (a : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i)
  (h_cond : (Finset.finRange n).sup a ≤ n * (Finset.finRange n).inf a) :
  ∃ (i j k : Fin n), ∃ (h_ijk : i ≠ j ∧ j ≠ k ∧ i ≠ k), 
  a i + a j > a k ∧ a j + a k > a i ∧ a k + a i > a j ∧ 
  ((a i) ^ 2 + (a j) ^ 2 > (a k) ^ 2) :=
sorry

end exists_acute_triangle_l500_500070


namespace equal_sharing_l500_500201

def john_initial : ℝ := 54.5
def cole_initial : ℝ := 45.75
def aubrey_initial : ℝ := 37
def maria_initial : ℝ := 70.25
def liam_initial : ℝ := 28.5
def emma_initial : ℝ := 32.5

-- Defining doubled amounts
def john_doubled : ℝ := john_initial * 2
def cole_doubled : ℝ := cole_initial * 2
def aubrey_doubled : ℝ := aubrey_initial * 2
def maria_doubled : ℝ := maria_initial * 2
def liam_doubled : ℝ := liam_initial * 2
def emma_doubled : ℝ := emma_initial * 2

-- Total amount after doubling
def total_gum : ℝ := john_doubled + cole_doubled + aubrey_doubled + maria_doubled + liam_doubled + emma_doubled

-- Total number of people
def num_people : ℝ := 6

-- Amount of gum each person gets
def gum_per_person : ℝ := total_gum / num_people

theorem equal_sharing :
  gum_per_person = 89.5 :=
by
  -- calculation steps 
  sorry

end equal_sharing_l500_500201


namespace cube_expression_l500_500153

theorem cube_expression (a : ℝ) (h : (a + 1/a)^2 = 5) : a^3 + 1/a^3 = 2 * Real.sqrt 5 :=
by
  sorry

end cube_expression_l500_500153


namespace find_b_from_polynomial_l500_500458

theorem find_b_from_polynomial
  (a b c d : ℝ)
  (z w : ℂ)
  (hz : z * w = 7 + 4 * complex.I)
  (hw_bar : z.conj + w.conj = -2 + 3 * complex.I)
  (real_coeffs : ∀ (p : ℂ), (p ∈ roots (polynomial.map complex.of_real (polynomial.C a * polynomial.X ^ 3 + polynomial.C b * polynomial.X ^ 2 + polynomial.C c * polynomial.X + polynomial.C d))) → p.conj ∈ roots (polynomial.map complex.of_real (polynomial.C a * polynomial.X ^ 3 + polynomial.C b * polynomial.X ^ 2 + polynomial.C c * polynomial.X + polynomial.C d)))
  : b = 27 :=
sorry

end find_b_from_polynomial_l500_500458


namespace novelists_count_l500_500018

theorem novelists_count (n p : ℕ) (h1 : n / (n + p) = 5 / 8) (h2 : n + p = 24) : n = 15 :=
sorry

end novelists_count_l500_500018


namespace gcd_expression_l500_500115

theorem gcd_expression (a : ℤ) (k : ℤ) (h1 : a = k * 1171) (h2 : k % 2 = 1) (prime_1171 : Prime 1171) : 
  Int.gcd (3 * a^2 + 35 * a + 77) (a + 15) = 1 :=
by
  sorry

end gcd_expression_l500_500115


namespace radius_of_lattice_point_influence_l500_500008

theorem radius_of_lattice_point_influence : 
  ∃ d : ℝ, (∀ d, 0.2 ≤ d ∧ d ≤ 0.6) ∧ (abs (d - 0.3) < 0.1) ∧ (π * d^2 = 1/4) := 
begin
  sorry
end

end radius_of_lattice_point_influence_l500_500008


namespace age_of_15th_person_l500_500605

theorem age_of_15th_person :
  let T := 17 * 15
  let S₁ := 5 * 14
  let S₂ := 9 * 16
  T = S₁ + S₂ + 41 :=
by
  let T := 17 * 15
  let S₁ := 5 * 14
  let S₂ := 9 * 16
  have h₀ : 255 = T := by norm_num
  have h₁ : 70 = S₁ := by norm_num
  have h₂ : 144 = S₂ := by norm_num
  calc
    T = 255 := h₀
    ... = 70 + 144 + 41 := by rw [h₁, h₂]; norm_num

#check age_of_15th_person

end age_of_15th_person_l500_500605


namespace number_of_tiles_required_l500_500009

theorem number_of_tiles_required (room_length room_width : ℝ) (tile_length tile_width : ℝ) (room_area tile_area : ℝ) (n : ℝ) :
    room_length = 15 →
    room_width = 18 →
    tile_length = 3 / 12 →
    tile_width = 9 / 12 →
    room_area = room_length * room_width →
    tile_area = tile_length * tile_width →
    n = room_area / tile_area →
    n = 1440 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6, h7]
  sorry

end number_of_tiles_required_l500_500009


namespace tan_alpha_solution_l500_500858

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l500_500858


namespace tan_alpha_solution_l500_500862

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l500_500862


namespace determine_denominator_of_fraction_l500_500352

theorem determine_denominator_of_fraction (x : ℝ) (h : 57 / x = 0.0114) : x = 5000 :=
by
  sorry

end determine_denominator_of_fraction_l500_500352


namespace minimum_boxes_needed_l500_500108

def can_form_two_digit_number (box : ℕ) (sheet_number : ℕ) : Prop :=
  let box_str := box.toString in
  let sheet_str := sheet_number.toString in
  (sheet_str.length = 2) ∧ 
  (box_str.length = 3) ∧
  (sheet_str = box_str.eraseIdx 0 ∨ sheet_str = box_str.eraseIdx 1 ∨ sheet_str = box_str.eraseIdx 2)

def all_sheets_covered (boxes : list ℕ) : Prop :=
  ∀ sheet_number : ℕ, (sheet_number < 100) -> 
    ∃ box : ℕ, box ∈ boxes ∧ can_form_two_digit_number (box) (sheet_number)

theorem minimum_boxes_needed : ∃ (boxes : list ℕ), list.length boxes = 34 ∧ all_sheets_covered boxes :=
sorry

end minimum_boxes_needed_l500_500108


namespace min_large_buses_proof_l500_500702

def large_bus_capacity : ℕ := 45
def small_bus_capacity : ℕ := 30
def total_students : ℕ := 523
def min_small_buses : ℕ := 5

def min_large_buses_required (large_capacity small_capacity total small_buses : ℕ) : ℕ :=
  let remaining_students := total - (small_buses * small_capacity)
  let buses_needed := remaining_students / large_capacity
  if remaining_students % large_capacity = 0 then buses_needed else buses_needed + 1

theorem min_large_buses_proof :
  min_large_buses_required large_bus_capacity small_bus_capacity total_students min_small_buses = 9 :=
by
  sorry

end min_large_buses_proof_l500_500702


namespace part_I_part_II_l500_500471

noncomputable def S_n (a_n : ℕ → ℤ) (n : ℕ) : ℤ := 2 * a_n n - 2 * n
noncomputable def a_n_formula (n : ℕ) : ℤ := 2^(n + 1) - 2

theorem part_I (a_n: ℕ → ℤ) (h: ∀ n : ℕ, n > 0 → S_n a_n n = 2 * a_n n - 2 * n) :
  ∀ n : ℕ, n > 0 ∧ a_n (n - 1) = a_n_formula (n - 1) → a_n n + 2 = 2 * (a_n (n - 1) + 2):=
sorry  

noncomputable def b_n (a_n : ℕ → ℤ) (n : ℕ) : ℤ := int.log (a_n n + 2) / int.log 2
noncomputable def T_n (b_n : ℕ → ℤ) (n : ℕ) : ℤ :=
  finset.sum (finset.range n) (λ k, 1 / (b_n k * b_n (k + 1)))

theorem part_II (a_n: ℕ → ℤ) (b_n: ℕ → ℤ) (T_n: ℕ → ℤ)
  (h1: ∀ n : ℕ, n > 0 → b_n n = int.log (a_n n + 2) / int.log 2)
  (h2: ∀ n : ℕ, n > 0 → T_n n = (1 / (n + 1)) - (1 / (n + 2))) :
  ∀ a : ℤ , T_n < a → a ≥ 1 / 2 :=
sorry  

end part_I_part_II_l500_500471


namespace tan_alpha_proof_l500_500901

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l500_500901


namespace tan_alpha_fraction_l500_500912

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l500_500912


namespace number_of_columns_per_section_l500_500379

variables (S C : ℕ)

-- Define the first condition: S * C + (S - 1) / 2 = 1223
def condition1 := S * C + (S - 1) / 2 = 1223

-- Define the second condition: S = 2 * C + 5
def condition2 := S = 2 * C + 5

-- Formulate the theorem that C = 23 given the two conditions
theorem number_of_columns_per_section
  (h1 : condition1 S C)
  (h2 : condition2 S C) :
  C = 23 :=
sorry

end number_of_columns_per_section_l500_500379


namespace tan_alpha_value_l500_500930

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l500_500930


namespace reflect_parabola_y_axis_l500_500252

theorem reflect_parabola_y_axis (x y : ℝ) :
  (y = 2 * (x - 1)^2 - 4) → (y = 2 * (-x - 1)^2 - 4) :=
sorry

end reflect_parabola_y_axis_l500_500252


namespace tan_alpha_solution_l500_500872

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l500_500872


namespace MNPQ_is_pseudo_square_pseudo_square_to_square_condition_l500_500209

open Complex

-- Definitions for the points and directed quadrilateral
variables {α : Type*} [normedAddCommGroup α] [normedSpace ℂ α]

structure DirectedQuadrilateral (α : Type*) extends AffineSpace ℂ α :=
(A B C D : α)

structure IsoscelesRightTriangle (M A B : α) (oriented : bool) : Prop :=
(angle_MBA : ∠ M B A = π / 2)
(distance_ratio : dist M A = dist M B)

-- Definitions for the points M, N, P, Q
variables {A B C D M N P Q : α}

noncomputable def construct_M (A B M : α) : Prop :=
IsoscelesRightTriangle M B A tt

noncomputable def construct_N (B C N : α) : Prop :=
IsoscelesRightTriangle N C B tt

noncomputable def construct_P (C D P : α) : Prop :=
IsoscelesRightTriangle P D C tt

noncomputable def construct_Q (D A Q : α) : Prop :=
IsoscelesRightTriangle Q A D tt

def MPQ_is_pseudo_square (M P Q N : α) : Prop :=
dist M P = dist N Q ∧ (⟪P - M, Q - N⟫ = 0)

def condition_for_square (M P N Q : α) (ABCD : DirectedQuadrilateral α) : Prop :=
M + P = N + Q

theorem MNPQ_is_pseudo_square (ABCD : DirectedQuadrilateral α) 
  (hM : construct_M A B M) 
  (hN : construct_N B C N) 
  (hP : construct_P C D P) 
  (hQ : construct_Q D A Q)
  : MPQ_is_pseudo_square M N P Q :=
sorry

theorem pseudo_square_to_square_condition (ABCD : DirectedQuadrilateral α) 
  (hM : construct_M A B M) 
  (hN : construct_N B C N) 
  (hP : construct_P C D P) 
  (hQ : construct_Q D A Q)
  : condition_for_square M P N Q ABCD ↔ parallelogram ABCD :=
sorry

end MNPQ_is_pseudo_square_pseudo_square_to_square_condition_l500_500209


namespace balls_in_boxes_ways_l500_500245

theorem balls_in_boxes_ways : 
  let balls := {1, 2, 3, 4, 5} in
  let boxes := {box1, box2, box3} in
  (∑ (split : {subset : set (set ℕ) // ∀ s ∈ subset, s ≠ ∅ ∧ s ⊆ balls} // split.val.card = 3) 
   ∏ (s ∈ split.val), finite s ∧ disjoint s split.val - {s}) * fact 3 = 150 :=
sorry

end balls_in_boxes_ways_l500_500245


namespace minimum_numbers_to_ensure_pair_sum_multiple_of_ten_l500_500462

theorem minimum_numbers_to_ensure_pair_sum_multiple_of_ten :
  ∀ (s : Finset ℕ), (∀ a b ∈ s, a ≠ b → ((a + b) % 10 = 0 → false)) 
  → s.card > 10 
  → ∃ a b ∈ s, a ≠ b ∧ (a + b) % 10 = 0 :=
by
  intros s h hs
  sorry

end minimum_numbers_to_ensure_pair_sum_multiple_of_ten_l500_500462


namespace proofCalcFourthPower_l500_500036

noncomputable def calcFourthPower : ℝ :=
  √(1 + √(2 + √(3 + √4)))

theorem proofCalcFourthPower : (calcFourthPower ^ 4) = 3 + √5 + 2 * √(2 + √5) :=
by sorry

end proofCalcFourthPower_l500_500036


namespace hyperbola_eccentricity_proof_l500_500505

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (asymptote : a = 2 * b) : ℝ :=
  let e := Real.sqrt ((a^2 + b^2) / a^2) in
  e

theorem hyperbola_eccentricity_proof (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (asymptote : a = 2 * b) :
  hyperbola_eccentricity a b h_a h_b asymptote = (Real.sqrt 5) / 2 :=
by
  unfold hyperbola_eccentricity
  have h_asym : a = 2 * b := asymptote
  sorry

end hyperbola_eccentricity_proof_l500_500505


namespace a_ne_0_for_all_nat_l500_500517

def a : ℕ → ℤ
| 0       := 1
| 1       := 2
| (n + 2) := if (a n) * (a (n + 1)) % 2 = 0 then
               5 * (a (n + 1)) - 3 * (a n)
             else
               (a (n + 1)) - (a n)

theorem a_ne_0_for_all_nat : ∀ n : ℕ, a n ≠ 0 :=
by {
  sorry
}

end a_ne_0_for_all_nat_l500_500517


namespace correct_inequality_l500_500815

variable {R : Type*} [LinearOrder R] (f : R → R) (x : R)

def even_function (f : R → R) : Prop :=
  ∀ x : R, f x = f (-x)

def monotonic_nonpos (f : R → R) : Prop :=
  ∀ x y : R, x ≤ 0 → y ≤ 0 → x < y → f x < f y

variables (h1 : even_function f) (h2 : monotonic_nonpos f) (h3 : f (-2) < f 1)

theorem correct_inequality : f 5 < f (-3) ∧ f (-3) < f (-1) :=
  sorry

end correct_inequality_l500_500815


namespace problem_x_fraction_l500_500289

theorem problem_x_fraction : 
  ∃ m n : ℕ, let x := 61727, y := 24690 in nat.gcd x y = 1 ∧ 2.5081081081081 = (61727 : ℚ) / 24690 ∧ x + y = 86417 :=
begin
  sorry,
end

end problem_x_fraction_l500_500289


namespace probability_even_sum_of_two_cards_l500_500791

def num_ways_to_draw_two_cards (n : ℕ) : ℕ :=
  Nat.choose n 2

def valid_even_sum_pairs (cards : Finset ℕ) : Finset (ℕ × ℕ) :=
  (cards.product cards).filter (λ p, p.1 < p.2 ∧ (p.1 + p.2) % 2 = 0)

theorem probability_even_sum_of_two_cards :
  let cards := {1, 2, 3, 4, 5}
  (num_ways_to_draw_two_cards 5) = 10 →
  valid_even_sum_pairs cards = { (1, 3), (2, 4), (1, 5), (3, 5) } →
  (valid_even_sum_pairs cards).card = 4 →
  (4 / num_ways_to_draw_two_cards 5) = (2 / 5) :=
by
  sorry

end probability_even_sum_of_two_cards_l500_500791


namespace prob_X_eq_2_prob_X_le_4_l500_500985

/-- Probability of A winning a single game is 2/3 -/
def prob_A : ℝ := 2 / 3

/-- Probability of B winning a single game is 1/3 -/
def prob_B : ℝ := 1 / 3

/-- Number of games to determine the champion -/
def X : ℝ := sorry

/-- Prove that the probability of the championship being determined in exactly 2 games is 5/9 -/
theorem prob_X_eq_2 : P(X = 2) = 5 / 9 := by sorry

/-- Prove that the probability of the championship being determined in at most 4 games in the best of five format is 19/27 -/
theorem prob_X_le_4 : P(X ≤ 4) = 19 / 27 := by sorry

end prob_X_eq_2_prob_X_le_4_l500_500985


namespace kim_tv_daily_hours_l500_500991

-- Conditions given in problem
def tv_power_watts := 125
def electricity_cost_per_kwh := 0.14
def weekly_cost := 0.49
def days_per_week := 7

-- Convert given power to kilowatts
def tv_power_kw := tv_power_watts / 1000.0

-- Proving the daily running hours
theorem kim_tv_daily_hours : (weekly_cost / electricity_cost_per_kwh) / tv_power_kw / days_per_week = 4 := 
by
  sorry

end kim_tv_daily_hours_l500_500991


namespace find_m_l500_500131

-- Definition of the function as a direct proportion function with respect to x
def isDirectProportion (m : ℝ) : Prop :=
  m^2 - 8 = 1

-- Definition of the graph passing through the second and fourth quadrants
def passesThroughQuadrants (m : ℝ) : Prop :=
  m - 2 < 0

-- The theorem combining the conditions and proving the correct value of m
theorem find_m (m : ℝ) 
  (h1 : isDirectProportion m)
  (h2 : passesThroughQuadrants m) : 
  m = -3 :=
  sorry

end find_m_l500_500131


namespace main_l500_500809

noncomputable def x1 (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def x2 (a b : ℝ) : ℝ := Real.sqrt (a * b)
noncomputable def x3 (a b : ℝ) : ℝ := Real.sqrt ((a ^ 2 + b ^ 2) / 2)

theorem main (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : x3 a b ≥ x1 a b ∧ x1 a b ≥ x2 a b :=
by
  split
  sorry
  sorry

end main_l500_500809


namespace A_and_D_independent_l500_500312

-- Define the probabilities of elementary events
def prob_A : ℚ := 1 / 6
def prob_B : ℚ := 1 / 6
def prob_C : ℚ := 5 / 36
def prob_D : ℚ := 1 / 6

-- Define the joint probability of A and D
def prob_A_and_D : ℚ := 1 / 36

-- Define the independence condition
def independent (P_X P_Y P_XY : ℚ) : Prop := P_XY = P_X * P_Y

-- Prove that events A and D are independent
theorem A_and_D_independent : 
  independent prob_A prob_D prob_A_and_D := by
  -- The proof is skipped
  sorry

end A_and_D_independent_l500_500312


namespace unique_b_for_quadratic_l500_500081

theorem unique_b_for_quadratic (c : ℝ) (h_c : c ≠ 0) : (∃! b : ℝ, b > 0 ∧ (2*b + 2/b)^2 - 4*c = 0) → c = 4 :=
by
  sorry

end unique_b_for_quadratic_l500_500081


namespace seq_converges_to_one_l500_500223

-- Define the sequence u_n
def u (n : ℕ) := (n + (-1)^n) / (n + 2)

-- State that the sequence converges to 1
theorem seq_converges_to_one : ∃ l, (∀ ϵ > 0, ∃ N, ∀ n ≥ N, |u n - l| < ϵ) ∧ l = 1 := 
by
  sorry

end seq_converges_to_one_l500_500223


namespace arithmetic_sequence_a8_is_4_l500_500971

theorem arithmetic_sequence_a8_is_4 (a : ℕ → ℕ) (d : ℕ) (h : ∀ n : ℕ, a n = a 1 + (n - 1) * d) 
  (condition : a 2 + a 7 + a 15 = 12) : a 8 = 4 := 
begin
  sorry
end

end arithmetic_sequence_a8_is_4_l500_500971


namespace incentres_coincide_l500_500628

noncomputable def triangle (A B C : Type) := ∃ (α β γ : Type), α → β → γ → Prop

variables {ABC : Triangle} {A1 B1 C1 : Point} {O_A O_B O_C : Circle} {I_ABC I_OAOBOC : Point}

-- Conditions:
def point_on_sides (A1 : Point) (B1 : Point) (C1 : Point) (ABC : Triangle) : Prop :=
  lies_on A1 (side BC ∩ ABC) ∧
  lies_on B1 (side CA ∩ ABC) ∧
  lies_on C1 (side AB ∩ ABC)

def equal_differences (A1 C1 B1 : Point) (ABC : Triangle) : Prop :=
  distance(AB1, AC1) = distance(CA1, CB1) ∧
  distance(CA1, CB1) = distance(BC1, BA1)

def circumcenter (P Q R : Point) : Circle := sorry
def incenter (P Q R : Point) : Point := sorry

# The Problem Statement:
theorem incentres_coincide 
  (A1 B1 C1 : Point)
  (ABC : Triangle)
  (h1 : point_on_sides A1 B1 C1 ABC)
  (h2 : equal_differences A1 C1 B1 ABC) 
  (O_A := circumcenter A B1 C1)
  (O_B := circumcenter A1 B C1)
  (O_C := circumcenter A1 B1 C) :
  incenter O_A O_B O_C = incenter A B C :=
sorry

end incentres_coincide_l500_500628


namespace tetrahedron_ABCD_volume_l500_500189

noncomputable def tetrahedron_volume (AB CD dist angle : ℝ) : ℝ :=
  (AB * CD * dist * (Real.sin angle)) / 6

theorem tetrahedron_ABCD_volume
  (AB CD dist angle : ℝ)
  (h1 : AB = 1)
  (h2 : CD = sqrt 3)
  (h3 : dist = 2)
  (h4 : angle = Real.pi / 3) :
  tetrahedron_volume AB CD dist angle = 1 / 2 :=
by
  rw [h1, h2, h3, h4]
  sorry

end tetrahedron_ABCD_volume_l500_500189


namespace tan_alpha_value_l500_500881

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l500_500881


namespace tan_alpha_value_l500_500887

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l500_500887


namespace permutation_inequality_l500_500569

open Finset

theorem permutation_inequality (n : ℕ) (h : 2 ≤ n) (P : Fin n → ℕ)
  (perm : Perm (coe ∘ P) (range n)) : 
  (∑ i in range (n-1), 1 / (P i + P (i + 1))) > (n - 1) / (n + 2) :=
by
  sorry

end permutation_inequality_l500_500569


namespace angle_BDC_eq_105_l500_500678

open Locale.Real

-- Definitions and conditions
variables {A B C D : Type*}

-- Assume the angles and properties given in conditions
variables (angle_A angle_B angle_C : ℝ)
variables (is_right_angle_ATC : angle_C = 90)
variables (angle_A_value : angle_A = 30)
variables (BD_bisector : ∃ D, angle_B - angle_A / 2 = 15)
variables (C_on_ATC : ∃ C, true)

-- The theorem we want to prove
theorem angle_BDC_eq_105
  (h1 : is_right_angle_ATC)
  (h2 : angle_A_value)
  (h3 : BD_bisector)
  (h4 : C_on_ATC) : 
  angle_B = 60 → 
  (angle_BDC = 180 - angle_B - angle_BD / 2) := sorry

end angle_BDC_eq_105_l500_500678


namespace dice_probability_l500_500949

-- Definitions based on conditions
def six_sided_die : Finset ℕ := {1, 2, 3, 4, 5, 6}
def fair_probability (n : ℕ) (s : Finset ℕ) := if n ∈ s then (1 : ℚ) / s.card else 0

-- Proposition to be proved
theorem dice_probability :
  (∑ perm in Finset.perm (Finset.range 6),
      let counts := perm.val.map (λ n, if n = 0 then 1 else if n = 1 then 1 else 0) in
      if ((counts.count 0 = 4) ∧ (counts.count 1 = 1)) then
        (fair_probability 1 six_sided_die)^4 *
        (fair_probability 2 six_sided_die) *
        (fair_probability 3 six_sided_die)
      else 0) = 5 / 648 :=
by
  sorry

end dice_probability_l500_500949


namespace simplify_inverse_sum_l500_500852

variable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem simplify_inverse_sum :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (a * b + a * c + b * c) :=
by sorry

end simplify_inverse_sum_l500_500852


namespace typist_original_salary_l500_500731

theorem typist_original_salary (S : ℝ) (h : 0.97 * 1.12 * 0.93 * 1.15 * S = 7600.35) :
  S ≈ 7041.77 :=
sorry

end typist_original_salary_l500_500731


namespace tan_alpha_sqrt_15_over_15_l500_500923

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l500_500923


namespace find_angle_A_max_perimeter_incircle_l500_500117

-- Definition of the triangle and the conditions
variables {A B C : Real} {a b c : Real} 

-- The conditions given in the problem
def triangle_conditions (a b c A B C : Real) : Prop :=
  (b + c = a * (Real.cos C + Real.sqrt 3 * Real.sin C)) ∧
  A + B + C = Real.pi

-- Part 1: Prove the value of angle A
theorem find_angle_A (a b c A B C : Real) 
(h : triangle_conditions a b c A B C) : 
A = Real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter of the incircle when a=2
theorem max_perimeter_incircle (b c A B C : Real) 
(h : triangle_conditions 2 b c A B C) : 
2 * Real.pi * (Real.sqrt 3 / 6 * (b + c - 2)) ≤ (2 * Real.sqrt 3 / 3) * Real.pi := sorry

end find_angle_A_max_perimeter_incircle_l500_500117


namespace find_number_l500_500363

variable (x : ℝ)

theorem find_number (h : 0.20 * x = 0.40 * 140 + 80) : x = 680 :=
by
  sorry

end find_number_l500_500363


namespace population_after_ten_years_l500_500629

-- Define the initial population and constants
def initial_population : ℕ := 100000
def birth_increase_rate : ℝ := 0.6
def emigration_per_year : ℕ := 2000
def immigration_per_year : ℕ := 2500
def years : ℕ := 10

-- Proving the total population at the end of 10 years
theorem population_after_ten_years :
  initial_population + (initial_population * birth_increase_rate).to_nat +
  (immigration_per_year * years - emigration_per_year * years) = 165000 := by
sorry

end population_after_ten_years_l500_500629


namespace closest_value_l500_500664

def a : ℝ := 0.000287
def b : ℝ := 9502430
def c : ℝ := 120
def d : ℝ := 2900

theorem closest_value (a b c d : ℝ) (h₁ : a = 0.000287) (h₂ : b = 9502430) (h₃ : c = 120) (h₄ : d = 2900) :
  abs ((a * b + c) - d) ≤ abs ((a * b + c) - x) → ∀ (x ∈ {2500, 2600, 2700, 2800, 2900}) :=
by {
  sorry 
}

end closest_value_l500_500664


namespace length_AP_l500_500982

variables (A M P : ℝ × ℝ)
variables (r : ℝ)
variables (ω : ℝ × ℝ → Prop)
variables (ABCD : set (ℝ × ℝ))

def is_unit_square (ABCD : set (ℝ × ℝ)) : Prop :=
  ABCD = {p | (0 ≤ p.1 ∧ p.1 ≤ 1) ∧ (0 ≤ p.2 ∧ p.2 ≤ 1)}

def is_inscribed_circle (ω : ℝ × ℝ → Prop) (r : ℝ) : Prop :=
  ∀ (P : ℝ × ℝ), ω P ↔ P.1^2 + P.2^2 = r^2

theorem length_AP (ABCD_square : is_unit_square ABCD) 
  (circle_center : ω 0.5) 
  (circle_radius : r = 0.5) 
  (M_on_CD : M = (0, -0.5)) 
  (P_on_ω : ω P)
  (P_different_M : P ≠ M)
  : dist A P = real.sqrt(5) / 10 :=
sorry

end length_AP_l500_500982


namespace salmon_trip_l500_500766

theorem salmon_trip (male_female_sum : 712261 + 259378 = 971639) : 
  712261 + 259378 = 971639 := 
by 
  exact male_female_sum

end salmon_trip_l500_500766


namespace equations_have_one_contact_point_l500_500057

theorem equations_have_one_contact_point (c : ℝ):
  (∃ x : ℝ, x^2 + 1 = 4 * x + c) ∧ (∀ x1 x2 : ℝ, (x1 ≠ x2) → ¬(x1^2 + 1 = 4 * x1 + c ∧ x2^2 + 1 = 4 * x2 + c)) ↔ c = -3 :=
by
  sorry

end equations_have_one_contact_point_l500_500057


namespace no_solution_intervals_l500_500780

theorem no_solution_intervals :
    ¬ ∃ x : ℝ, (2 / 3 < x ∧ x < 4 / 3) ∧ (1 / 5 < x ∧ x < 3 / 5) :=
by
  sorry

end no_solution_intervals_l500_500780


namespace solution_set_inequality_l500_500810

variable (f : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = -f(x)

def is_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f(x) > f(y)

theorem solution_set_inequality (hf_odd : is_odd f) (hf_decr : is_decreasing f (Set.Ioo (-2) 2)) :
  {x | f(x / 3) + f(2 * x - 1) > 0} = Set.Ioo (-1 / 2) (3 / 7) :=
by
  sorry

end solution_set_inequality_l500_500810


namespace total_num_arrangements_l500_500365

theorem total_num_arrangements (students : Fin₆ → α) (A B C : Finset α) :
  A.card = 3 ∧ B.card = 1 ∧ C.card = 2 ∧ (A ∪ B ∪ C = Finset.univ : Finset α) ∧
  ∀ s ∈ (A ∩ B ∪ A ∩ C ∪ B ∩ C), False →
  (Finset.choose _ _).card * (Finset.choose _ _).card * (Finset.choose _ _).card = 60 := by
  sorry

end total_num_arrangements_l500_500365


namespace problem_1_problem_2_l500_500141

open Real

noncomputable def vec_a (x : ℝ) := (sin x, sqrt 3 * cos x)
noncomputable def vec_b := (-1 : ℝ, 1 : ℝ)
noncomputable def vec_c := (1 : ℝ, 1 : ℝ)

theorem problem_1 (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ π) 
  (h2 : ∃ k : ℝ, vec_a x.1 + vec_b = k • vec_c) : 
  x = 5 * π / 6 :=
sorry

theorem problem_2 (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ π) 
  (h2 : vec_a x.1.1 * vec_b = 1/2) : 
  sin (x + π/6) = sqrt 15 / 4 :=
sorry

end problem_1_problem_2_l500_500141


namespace integral_sup_condition_l500_500570

def f_in_C (f : ℝ → ℝ) : Prop :=
  ∃ (f'' : ℝ → ℝ) (x₁ x₂ : ℝ), x₁ ∈ (Set.Icc 0 1) ∧ x₂ ∈ (Set.Icc 0 1) ∧ x₁ ≠ x₂ ∧
  ((f x₁ = 0 ∧ f x₂ = 0) ∨ (f x₁ = 0 ∧ ∃ (f' : ℝ → ℝ), f' x₁ = 0)) ∧
  (∀ x ∈ (Set.Icc 0 1), f'' x < 1)

def sup_integral_cond (f : ℝ → ℝ) : Prop :=
  f ∈ {f | f_in_C f}

theorem integral_sup_condition :
  ∃ (f∗ : ℝ → ℝ), sup_integral_cond f∗ ∧
  ∫ x in 0..1, abs (f∗ x) = (1 : ℝ) / (12 : ℝ) :=
sorry

end integral_sup_condition_l500_500570


namespace maximizing_take_home_pay_l500_500538

noncomputable def tax_rate (x : ℝ) : ℝ := 2 * x / 100
noncomputable def income (x : ℝ) : ℝ := 1000 * x
noncomputable def tax_collected (x : ℝ) : ℝ := tax_rate(x) * income(x)
noncomputable def take_home_pay (x : ℝ) : ℝ := income(x) - tax_collected(x)

theorem maximizing_take_home_pay : ∀ (x : ℝ), take_home_pay x ≤ take_home_pay 25 :=
begin
  sorry
end

end maximizing_take_home_pay_l500_500538


namespace value_of_f_g_6_squared_l500_500840

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 11

theorem value_of_f_g_6_squared : (f (g 6))^2 = 26569 :=
by
  -- Place your proof here
  sorry

end value_of_f_g_6_squared_l500_500840


namespace Petya_can_guarantee_win_l500_500243

def initial_number := "11223334445555666677777"

def erases_last_digit_wins (current_number : String) (player : String) : Bool :=
  -- Define what it means to "erase the last digit and win" here.
  -- This is a placeholder function for the game's winning condition.
  sorry

theorem Petya_can_guarantee_win : erases_last_digit_wins initial_number "Petya" := by
  -- Petya's strategy guarantees a win
  sorry

end Petya_can_guarantee_win_l500_500243


namespace independence_of_A_and_D_l500_500326

noncomputable def balls : Finset ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def draw_one : ℕ := (1 : ℕ)
noncomputable def draw_two : ℕ := (2 : ℕ)

def event_A : ℕ → Prop := λ n, n = 1
def event_B : ℕ → Prop := λ n, n = 2
def event_C : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 8
def event_D : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 7

def prob (event : ℕ → Prop) : ℚ := 1 / 6
def joint_prob (event1 event2 : ℕ → Prop) : ℚ := (1 / 36)

theorem independence_of_A_and_D :
  joint_prob (λ n, event_A n) (λ n, event_D (draw_one, draw_two)) = prob event_A * prob (λ n, event_D (draw_one, draw_two)) :=
by
  sorry

end independence_of_A_and_D_l500_500326


namespace min_seated_to_sit_next_l500_500306

theorem min_seated_to_sit_next (n : ℕ) (h_n : n = 10) :
  ∃ m, m = 4 ∧ ∀ (occupied : set ℕ), (∀ x ∈ occupied, x ≤ n) →
  (∀ x, x ∉ occupied → ∃ y ∈ occupied, abs (x - y) = 1) :=
sorry

end min_seated_to_sit_next_l500_500306


namespace weighted_avg_M_B_eq_l500_500305

-- Define the weightages and the given weighted total marks equation
def weight_physics : ℝ := 1.5
def weight_chemistry : ℝ := 2
def weight_mathematics : ℝ := 1.25
def weight_biology : ℝ := 1.75
def weighted_total_M_B : ℝ := 250
def weighted_sum_M_B : ℝ := weight_mathematics + weight_biology

-- Theorem statement: Prove that the weighted average mark for mathematics and biology is 83.33
theorem weighted_avg_M_B_eq :
  (weighted_total_M_B / weighted_sum_M_B) = 83.33 :=
by
  sorry

end weighted_avg_M_B_eq_l500_500305


namespace percentage_of_water_in_mixture_is_17_14_l500_500674

def Liquid_A_water_percentage : ℝ := 0.10
def Liquid_B_water_percentage : ℝ := 0.15
def Liquid_C_water_percentage : ℝ := 0.25
def Liquid_D_water_percentage : ℝ := 0.35

def parts_A : ℝ := 3
def parts_B : ℝ := 2
def parts_C : ℝ := 1
def parts_D : ℝ := 1

def part_unit : ℝ := 100

noncomputable def total_units : ℝ := 
  parts_A * part_unit + parts_B * part_unit + parts_C * part_unit + parts_D * part_unit

noncomputable def total_water_units : ℝ :=
  parts_A * part_unit * Liquid_A_water_percentage +
  parts_B * part_unit * Liquid_B_water_percentage +
  parts_C * part_unit * Liquid_C_water_percentage +
  parts_D * part_unit * Liquid_D_water_percentage

noncomputable def percentage_water : ℝ := (total_water_units / total_units) * 100

theorem percentage_of_water_in_mixture_is_17_14 :
  percentage_water = 17.14 := sorry

end percentage_of_water_in_mixture_is_17_14_l500_500674


namespace miles_per_hour_l500_500734

theorem miles_per_hour (total_distance : ℕ) (total_hours : ℕ) (h1 : total_distance = 81) (h2 : total_hours = 3) :
  total_distance / total_hours = 27 :=
by
  sorry

end miles_per_hour_l500_500734


namespace Tim_words_known_l500_500335

def Tim_original_words : ℕ := 14600

theorem Tim_words_known (days_in_two_years : ℕ) (daily_words_learned : ℕ) 
(increase_percentage : ℚ) (total_words_learned : ℕ) (original_words : ℕ) :
  days_in_two_years = 730 →
  daily_words_learned = 10 →
  increase_percentage = 0.5 →
  total_words_learned = days_in_two_years * daily_words_learned →
  original_words = total_words_learned * 2 →
  original_words = Tim_original_words :=
by
  intros h1 h2 h3 h4 h5
  rw [←h5, ←h4, h1, h2]
  unfold Tim_original_words
  sorry

end Tim_words_known_l500_500335


namespace tan_alpha_proof_l500_500895

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l500_500895


namespace Q_eq_N_l500_500847

variables {α : Type*} (P R Q M N : set α)

-- Define the conditions
def P := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
def R := {x : ℝ | ∃ y : ℝ, y = x^2 + 1}
def Q := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
def M := {(x, y) : ℝ × ℝ | y = x^2 + 1}
def N := {x : ℝ | x ≥ 1}

-- Statement to prove that Q = N
theorem Q_eq_N : Q = N :=
sorry

end Q_eq_N_l500_500847


namespace tan_alpha_value_l500_500938

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l500_500938


namespace sum_n_k_of_binomial_coefficient_ratio_l500_500616

theorem sum_n_k_of_binomial_coefficient_ratio :
  ∃ (n k : ℕ), (n = (7 * k + 5) / 2) ∧ (2 * (n - k) = 5 * (k + 1)) ∧ 
    ((k % 2 = 1) ∧ (n + k = 7 ∨ n + k = 16)) ∧ (23 = 7 + 16) :=
by
  sorry

end sum_n_k_of_binomial_coefficient_ratio_l500_500616


namespace monthly_fee_l500_500987

theorem monthly_fee (x : ℝ) : 
  (∃ x, (12.02 = x + 0.25 * 28.08)) → x = 5.00 :=
by
  intro h
  cases h with fee h_fee
  have : fee = 12.02 - 0.25 * 28.08 := 
    by linarith
  rw this
  norm_num
  assumption

end monthly_fee_l500_500987


namespace ball_hits_ground_l500_500370

theorem ball_hits_ground : 
  ∃ t : ℝ, (0 < t) ∧ (t = (1 + Real.sqrt 31) / 2) :=
by
  have initial_velocity : ℝ := 16
  have initial_height : ℝ := 120
  have height_equation : ℝ → ℝ := fun t => -16 * t^2 + initial_velocity * t + initial_height
  have quadratic_formula_solution_1 : ℝ := (1 + Real.sqrt 31) / 2
  have quadratic_formula_solution_2 : ℝ := (1 - Real.sqrt 31) / 2
  have ball_hits_ground_time : ℝ := quadratic_formula_solution_1
  exact Exists.intro ball_hits_ground_time ⟨by simp [ball_hits_ground_time, height_equation, quadratic_formula_solution_1, quadratic_formula_solution_2], by rfl⟩

end ball_hits_ground_l500_500370


namespace quadratic_two_distinct_real_roots_l500_500093

theorem quadratic_two_distinct_real_roots (k : ℝ) : ∃ x : ℝ, x^2 + 2 * x - k = 0 ∧ 
  (∀ x1 x2: ℝ, x1 ≠ x2 → x1^2 + 2 * x1 - k = 0 ∧ x2^2 + 2 * x2 - k = 0) ↔ k > -1 :=
by
  sorry

end quadratic_two_distinct_real_roots_l500_500093


namespace rowing_problem_l500_500706

theorem rowing_problem (R S x y : ℝ) 
  (h1 : R = y + x) 
  (h2 : S = y - x) : 
  x = (R - S) / 2 ∧ y = (R + S) / 2 :=
by
  sorry

end rowing_problem_l500_500706


namespace monotonically_decreasing_interval_l500_500264

theorem monotonically_decreasing_interval (f : ℝ → ℝ) (a : ℝ) (h : a < 0)
  (hf' : ∀ x, deriv f x = -x * (x + 1)) :
  ∀ x, (x ∈ set.Ioo (1/a) 0 ↔ (deriv (λ x, f (a * x - 1)) x < 0)) :=
sorry

end monotonically_decreasing_interval_l500_500264


namespace distinct_exponentiation_values_l500_500043

theorem distinct_exponentiation_values : 
  ∃ (standard other1 other2 other3 : ℕ), 
    standard ≠ other1 ∧ 
    standard ≠ other2 ∧ 
    standard ≠ other3 ∧ 
    other1 ≠ other2 ∧ 
    other1 ≠ other3 ∧ 
    other2 ≠ other3 := 
sorry

end distinct_exponentiation_values_l500_500043


namespace remaining_grandchild_share_l500_500142

theorem remaining_grandchild_share 
  (total : ℕ) 
  (half_share : ℕ) 
  (remaining : ℕ) 
  (n : ℕ) 
  (total_eq : total = 124600)
  (half_share_eq : half_share = total / 2)
  (remaining_eq : remaining = total - half_share)
  (n_eq : n = 10) 
  : remaining / n = 6230 := 
by sorry

end remaining_grandchild_share_l500_500142


namespace bicycle_cost_price_l500_500673

variable (CP_A SP_B SP_C : ℝ)

theorem bicycle_cost_price 
  (h1 : SP_B = CP_A * 1.20) 
  (h2 : SP_C = SP_B * 1.25) 
  (h3 : SP_C = 225) :
  CP_A = 150 := 
by
  sorry

end bicycle_cost_price_l500_500673


namespace g_diff_l500_500619

noncomputable def g : ℝ → ℝ := sorry 

theorem g_diff :
  (∀ x : ℝ, g (x + 1) - g x = 6) →
  g(2) - g(7) = -30 :=
by
  intros h
  -- Given the linearity and the condition, we need to prove g(2) - g(7) = -30
  sorry

end g_diff_l500_500619


namespace find_fg3_l500_500154

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 + 1

theorem find_fg3 : f (g 3) = 15 :=
by
  sorry

end find_fg3_l500_500154


namespace tan_alpha_sqrt_15_over_15_l500_500919

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l500_500919


namespace tan_alpha_solution_l500_500868

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l500_500868


namespace parametric_to_standard_eq_l500_500077

theorem parametric_to_standard_eq (t : ℝ) (x y : ℝ) (ht : t ≥ 0) (hx : x = sqrt t) (hy : y = 1 - 2 * sqrt t) :
  2 * x + y = 1 ∧ x ≥ 0 :=
by
  sorry

end parametric_to_standard_eq_l500_500077


namespace exists_sum_at_least_five_l500_500222

variables (x : fin 9 → ℝ) 

theorem exists_sum_at_least_five (h1 : ∀ i, 0 ≤ x i)
                                (h2 : ∑ i, (x i)^2 ≥ 25) :
  ∃ (i j k : fin 9), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ x i + x j + x k ≥ 5 :=
sorry

end exists_sum_at_least_five_l500_500222


namespace distance_from_center_to_chord_l500_500696

-- Let s be the side length of the square
variable (s : ℝ)

-- Let r be the radius of the circle
variable (r : ℝ)

-- Conditions
def inscribed_in_right_angle_triangle : Prop := 
  s = 2 * r ∧ s = 2

-- Theorem to prove
theorem distance_from_center_to_chord (s r : ℝ) (h : inscribed_in_right_angle_triangle s r) : r = 1 :=
by
  unfold inscribed_in_right_angle_triangle at h
  cases h with h1 h2
  rw h2 at h1
  linarith

end distance_from_center_to_chord_l500_500696


namespace tan_alpha_value_l500_500879

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l500_500879


namespace mean_proportional_example_l500_500492

theorem mean_proportional_example (a b c : ℝ) (ha : a = 2) (hb : b = 4) (h : b^2 = a * c) : c = 8 :=
by
  subst ha
  subst hb
  rw mul_comm at h
  have : 4^2 = 16 := rfl
  rw this at h
  norm_num at h
  sorry

end mean_proportional_example_l500_500492


namespace problem_l500_500490

-- Given conditions
variables {OA OB AB : ℝ} -- OA, OB are real vectors

-- Assuming conditions
axiom OA_norm : |OA| = 1
axiom OB_norm : |OB| = 1
axiom AB_norm : |OA - OB| = sqrt 3

-- Goals
theorem problem :
  OA • OB = -1/2 ∧ |OA + OB| = 1 :=
by
  sorry

end problem_l500_500490


namespace hyperbola_eccentricity_l500_500096

variable {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)

def hyperbola (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

def asymptote1 (x y : ℝ) : Prop := y = (b / a) * x
def asymptote2 (x y : ℝ) : Prop := y = -(b / a) * x

def focal_distance_A (FA : ℝ) : Prop := FA = b
def focal_distance_B (FB : ℝ) : Prop := FB = (c / 2) * real.sqrt(1 + b^2 / a^2)

axiom distances_relation (FA FB : ℝ) (hFA : focal_distance_A FA) (hFB : focal_distance_B FB) : FA = 4/5 * FB

theorem hyperbola_eccentricity (hC : ∀ x y, hyperbola x y) (hl1 : ∀ x y, asymptote1 x y) (hl2 : ∀ x y, asymptote2 x y)
  (FA FB : ℝ) (hFA : focal_distance_A FA) (hFB : focal_distance_B FB) (hRel : distances_relation FA FB hFA hFB) :
  e = real.sqrt 5 ∨ e = real.sqrt 5 / 2 :=
sorry

end hyperbola_eccentricity_l500_500096


namespace min_value_expression_l500_500218

theorem min_value_expression (x y : ℝ) : ∃ m, (∀ x y : ℝ, 5 * x ^ 2 + 4 * y ^ 2 - 8 * x * y + 2 * x + 4 ≥ m) ∧
  m = 3 :=
by
  use 3
  split
  · intro x y
    sorry
  · refl

end min_value_expression_l500_500218


namespace remainder_count_l500_500520

theorem remainder_count (n : ℕ) (h : n > 5) : 
  ∃ l : List ℕ, l.length = 5 ∧ ∀ x ∈ l, x ∣ 42 ∧ x > 5 := 
sorry

end remainder_count_l500_500520


namespace find_speed_of_A_l500_500240

-- Definitions for the given problem
def time_a := 30                  -- Time after which 乙 and 丙 start (minutes)
def catch_up_b := 20              -- Time 乙 takes to catch up 甲 (minutes)
def fraction_c := 1 / 5           -- Fraction of journey when 丙 catches up with 甲
def distance_meet := 1530         -- Distance between 甲 and the meeting point of 乙 and 丙 (meters)
def speed_ratio_c := 2            -- Speed of 丙 is twice the speed of 甲
def speed_increase_b := 1.20      -- Speed of 乙 increases by 20%

-- Main variable
variable (V_a : ℝ)                -- Speed of 甲 (meters per minute)

-- Definitions derived from conditions
def V_b := (5 / 2) * V_a          -- Speed relation between 甲 and 乙 
def V_c := speed_ratio_c * V_a    -- Speed of 丙
def V_b_new := speed_increase_b * V_b  -- New speed of 乙 after acceleration

-- Main theorem to state the problem
theorem find_speed_of_A (V_a : ℝ) : ∃ (V_a : ℝ), -- There exists a speed V_a such that
  V_b * (time_a + catch_up_b) = V_a * (time_a + 2 * catch_up_b) ∧ -- Condition for 乙 catching up 甲
  distance_meet = V_a * (V_c * (catch_up_b / fraction_c)) := sorry   -- Given distances matching as per condition

end find_speed_of_A_l500_500240


namespace store_A_better_deal_for_300_notebooks_l500_500402

def cost_A (x : ℕ) : ℝ :=
if x ≤ 100 then 5 * x else 4 * x + 100

def cost_B (x : ℕ) : ℝ := 4.5 * x

theorem store_A_better_deal_for_300_notebooks :
  cost_A 300 < cost_B 300 :=
by {
  rw [cost_A, cost_B],
  split_ifs,
  norm_num,
  linarith,
  sorry
}

end store_A_better_deal_for_300_notebooks_l500_500402


namespace minimize_distance_l500_500812

noncomputable def f (x : ℝ) := x^2 - 2 * x
noncomputable def P (x : ℝ) : ℝ × ℝ := (x, f x)
def Q : ℝ × ℝ := (4, -1)

theorem minimize_distance : ∃ (x : ℝ), dist (P x) Q = Real.sqrt 5 := by
  sorry

end minimize_distance_l500_500812


namespace odd_terms_arithmetic_even_terms_arithmetic_S_2n_l500_500820

def S (n : ℕ) : ℕ := sorry
def a (n : ℕ) : ℕ := sorry

axiom sum_eq (n : ℕ) : S n + S (n + 1) = n^2 + n + 1
axiom a1_eq : a 1 = 1

theorem odd_terms_arithmetic : ∀ n : ℕ, a (2 * n + 1) - a (2 * n - 1) = a (1) - a 1 := sorry

theorem even_terms_arithmetic : ∀ n : ℕ, a (2 * n) - a (2 * n - 2) = a (2) - a(2) := sorry

theorem S_2n : ∀ n : ℕ, S (2 * n) = 2 * n^2 := sorry

end odd_terms_arithmetic_even_terms_arithmetic_S_2n_l500_500820


namespace tan_alpha_proof_l500_500894

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l500_500894


namespace problem_l500_500836

noncomputable def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 21*x - 26

theorem problem
  (h_tangent_line : ∃ (a b : ℝ), f(a) = 0 ∧ f'(a) = 9)
  (h_ineq : ∀ (x : ℝ), 1 < x ∧ x < 5 → 21*x + k - 80 < f(x) ∧ f(x) < 9*x + k)
  : 9 < k ∧ k < 22 := by
  -- Proof omitted here
  sorry

end problem_l500_500836


namespace emma_ate_more_than_liam_l500_500439

-- Definitions based on conditions
def emma_oranges : ℕ := 8
def liam_oranges : ℕ := 1

-- Lean statement to prove the question
theorem emma_ate_more_than_liam : emma_oranges - liam_oranges = 7 := by
  sorry

end emma_ate_more_than_liam_l500_500439


namespace independence_of_A_and_D_l500_500327

noncomputable def balls : Finset ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def draw_one : ℕ := (1 : ℕ)
noncomputable def draw_two : ℕ := (2 : ℕ)

def event_A : ℕ → Prop := λ n, n = 1
def event_B : ℕ → Prop := λ n, n = 2
def event_C : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 8
def event_D : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 7

def prob (event : ℕ → Prop) : ℚ := 1 / 6
def joint_prob (event1 event2 : ℕ → Prop) : ℚ := (1 / 36)

theorem independence_of_A_and_D :
  joint_prob (λ n, event_A n) (λ n, event_D (draw_one, draw_two)) = prob event_A * prob (λ n, event_D (draw_one, draw_two)) :=
by
  sorry

end independence_of_A_and_D_l500_500327


namespace flagpole_height_l500_500789

theorem flagpole_height (h c d : ℝ) 
  (h1 : h^2 + c^2 = 170^2) 
  (h2 : h^2 + d^2 = 100^2) 
  (h3 : c^2 + d^2 = 120^2) : 
  h = 50 * real.sqrt 7 := 
sorry

end flagpole_height_l500_500789


namespace max_value_ratio_l500_500997

variables (A B C D P Q R S K L M N : Point)
variables (PQ_mid QR_mid RS_mid SP_mid : Point)
variables [IsConvexQuadrilateral A B C D]

noncomputable def equilateral_triangle1 (A B : Point) : Triangle := sorry
noncomputable def equilateral_triangle2 (B C : Point) : Triangle := sorry
noncomputable def equilateral_triangle3 (C D : Point) : Triangle := sorry
noncomputable def equilateral_triangle4 (D A : Point) : Triangle := sorry

axiom midpoint_def : ∀ (P Q R S : Point), Midpoint P Q = PQ_mid ∧ Midpoint Q R = QR_mid ∧ Midpoint R S = RS_mid ∧ Midpoint S P = SP_mid

theorem max_value_ratio :
  ∀ (A B C D P Q R S K L M N : Point), IsConvexQuadrilateral A B C D →
  let P := equilateral_triangle4 D A in
  let Q := equilateral_triangle1 A B in
  let R := equilateral_triangle2 B C in
  let S := equilateral_triangle3 C D in
  Midpoint P Q = K →
  Midpoint Q R = L →
  Midpoint R S = M →
  Midpoint S P = N →
  ∃ KM LN AC BD : ℝ, 
  KM = dist K M →
  LN = dist L N →
  AC = dist A C →
  BD = dist B D →
  KM + LN = (sqrt 3 + 1) / 2 * (AC + BD) :=
sorry

end max_value_ratio_l500_500997


namespace tan_alpha_value_l500_500927

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l500_500927


namespace vectors_opposite_direction_l500_500139

noncomputable def a : ℝ × ℝ := (-2, 4)
noncomputable def b : ℝ × ℝ := (1, -2)

theorem vectors_opposite_direction : a = (-2 : ℝ) • b :=
by
  sorry

end vectors_opposite_direction_l500_500139


namespace general_term_theorem_sum_bn_theorem_l500_500484

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n+1) = a n + d

def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

def bn (Sn : ℕ → ℝ) (n : ℕ) : ℝ :=
  1 / Sn n

-- Define the general term formula property
def general_term (a : ℕ → ℝ) :=
  ∀ n, a n = 2 * n + 1

-- Define the sum of the first n terms of bn property
def sum_bn (T : ℕ → ℝ) :=
  ∀ n, T n = (3 / 4) - ((2 * n + 3) / (2 * (n + 1) * (n + 2)))

-- Positive arithmetic sequence condition
def positive_arithmetic_sequence (a : ℕ → ℝ) :=
  is_arithmetic_sequence a ∧ (∀ n, a n > 0)

-- Define the conditions in Lean notation
def conditions (a : ℕ → ℝ) (Sn : ℕ → ℝ) :=
  a 1 = 3 ∧ a 2 * a 3 = Sn 5 ∧ positive_arithmetic_sequence a

-- The main theorem for first proof
theorem general_term_theorem (a : ℕ → ℝ) (Sn : ℕ → ℝ) :
  conditions a Sn → general_term a :=
by
  sorry

-- The main theorem for second proof
theorem sum_bn_theorem (a : ℕ → ℝ) (Sn : ℕ → ℝ) (T : ℕ → ℝ) :
  conditions a Sn →
  (∀ n, Sn n = n * (a 1 + a n) / 2) →
  (∀ n, T n = ∑ i in finset.range n, bn Sn i) →
  sum_bn T :=
by
  sorry

end general_term_theorem_sum_bn_theorem_l500_500484


namespace range_of_a_l500_500839

theorem range_of_a (a b : ℝ) 
  (h : ∀ x : ℝ, 2^(x - 1) + a ≥ b * (2^(1 - x - 1) + a)) 
  (min_solution : ∃ x : ℝ, 2 * x - 1 ∧ (∀ y : ℝ, 2 * y - 1 → y ≥ x))
: a ≤ -2 ∨ a > -1 / 4 :=
sorry

end range_of_a_l500_500839


namespace johns_drawing_time_l500_500554

variables (D : ℝ)

def drawing_time (pictures : ℕ) (time_per_drawing : ℝ) := pictures * time_per_drawing
def coloring_time (pictures : ℕ) (time_per_coloring : ℝ) := pictures * time_per_coloring
def total_time (pictures : ℕ) (time_per_drawing : ℝ) (time_per_coloring : ℝ) :=
  drawing_time pictures time_per_drawing + coloring_time pictures time_per_coloring

theorem johns_drawing_time :
  (D : ℝ) = 2 :=
begin
  -- John's conditions
  let num_pictures : ℕ := 10,
  let draw_time : ℝ := D,
  let color_time : ℝ := 0.70 * D,
  let total_spent_time : ℝ := 34,

  -- Calculation
  have total_eq : total_time num_pictures draw_time color_time = total_spent_time,
  { unfold total_time drawing_time coloring_time,
    sorry },
  sorry
end

end johns_drawing_time_l500_500554


namespace area_of_triangle_l500_500053

-- Define the parabolic equation
def parabola (x : ℝ) : ℝ := x^2

-- Define the line equation
def line (x : ℝ) : ℝ := 2 - x

-- Define points of intersection of the parabola and the line
def intersection_points : set (ℝ × ℝ) :=
  {p | ∃ (x : ℝ), p = (x, parabola x) ∧ line x = parabola x}

-- Define the center of the circle
def circle_center : ℝ × ℝ := (0, 0)

-- Triangle vertices are the intersection points and the circle center
def triangle_vertices : set (ℝ × ℝ) :=
  intersection_points ∪ {circle_center}

-- Function to calculate the area of a triangle given three vertices
def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * real.abs (v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2))

-- Extract the specific intersection points
def v1 : ℝ × ℝ := (-2, 4)
def v2 : ℝ × ℝ := (1, 1)

-- State the equivalent proof problem
theorem area_of_triangle : triangle_area circle_center v1 v2 = 3 := by
  sorry

end area_of_triangle_l500_500053


namespace gcd_lcm_252_l500_500666

theorem gcd_lcm_252 {a b : ℕ} (h : Nat.gcd a b * Nat.lcm a b = 252) :
  ∃ S : Finset ℕ, S.card = 8 ∧ ∀ d ∈ S, d = Nat.gcd a b :=
by sorry

end gcd_lcm_252_l500_500666


namespace num_of_multiples_l500_500851

theorem num_of_multiples (n : ℕ) :
  (nat.count (λ x, x ∈ finset.Icc 1 1500 ∧ ((x % 2 = 0 ∨ x % 5 = 0) ∧ x % 10 ≠ 0)) (finset.Icc 1 1500).val) = 900 :=
by
  sorry

end num_of_multiples_l500_500851


namespace probability_red_then_green_l500_500541

-- Total number of balls and their representation
def total_balls : ℕ := 3
def red_balls : ℕ := 2
def green_balls : ℕ := 1

-- The total number of outcomes when drawing two balls with replacement
def total_outcomes : ℕ := total_balls * total_balls

-- The desired outcomes: drawing a red ball first and a green ball second
def desired_outcomes : ℕ := 2 -- (1,3) and (2,3)

-- Calculating the probability of drawing a red ball first and a green ball second
def probability_drawing_red_then_green : ℚ := desired_outcomes / total_outcomes

-- The theorem we need to prove
theorem probability_red_then_green :
  probability_drawing_red_then_green = 2 / 9 :=
by 
  sorry

end probability_red_then_green_l500_500541


namespace plane_equation_midpoint_perpendicular_l500_500749

theorem plane_equation_midpoint_perpendicular
  (A D C B : ℝ × ℝ × ℝ)
  (hA : A = (-1, 2, -3))
  (hD : D = (-5, 6, -1))
  (hC : C = (-3, 10, -5))
  (hB : B = (3, 4, 1)) :
  ∃ (α : ℝ → ℝ → ℝ → Prop),
    (∀ x y z, α x y z ↔ x - y + z + 9 = 0) ∧
    (∃ M : ℝ × ℝ × ℝ, M = ((fst A + fst D) / 2, (snd A + snd D) / 2, (trd A + trd D) / 2)) ∧
    (∃ n : ℝ × ℝ × ℝ, n = (fst B - fst C, snd B - snd C, trd B - trd C)) ∧
    α (fst ((fst A + fst D) / 2, (snd A + snd D) / 2, (trd A + trd D) / 2)) (snd ((fst A + fst D) / 2, (snd A + snd D) / 2, (trd A + trd D) / 2)) (trd ((fst A + fst D) / 2, (snd A + snd D) / 2, (trd A + trd D) / 2)) :=
by
  sorry

end plane_equation_midpoint_perpendicular_l500_500749


namespace tan_alpha_value_l500_500876

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l500_500876


namespace soccer_team_lineup_count_l500_500013

theorem soccer_team_lineup_count :
  ∃ (n : ℕ), n = 15 * 14 * 13 * 12 * 11 * 10 * 9 ∧ n = 2541600 :=
by {
  use 15 * 14 * 13 * 12 * 11 * 10 * 9,
  split,
  { refl, },
  { refl, },
}

end soccer_team_lineup_count_l500_500013


namespace length_A_l500_500559

noncomputable def A := (0 : ℝ, 7 : ℝ)
noncomputable def B := (0 : ℝ, 11 : ℝ)
noncomputable def C := (3 : ℝ, 7 : ℝ)

noncomputable def A' := (7 : ℝ, 7 : ℝ)
noncomputable def B' := (33 / 7 : ℝ, 33 / 7 : ℝ)

theorem length_A'B' :
  let dist := (p q : ℝ × ℝ) → Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)
  in dist A' B' = 32 / 7 :=
by
  intros
  let dist := fun (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)
  sorry

end length_A_l500_500559


namespace width_of_foil_covered_prism_l500_500330

theorem width_of_foil_covered_prism (L W H : ℝ) 
  (h1 : W = 2 * L)
  (h2 : W = 2 * H)
  (h3 : L * W * H = 128)
  (h4 : L = H) :
  W + 2 = 8 :=
sorry

end width_of_foil_covered_prism_l500_500330


namespace final_output_value_of_m_l500_500499

variables (a b m : ℕ)

theorem final_output_value_of_m (h₁ : a = 2) (h₂ : b = 3) (program_logic : (a > b → m = a) ∧ (a ≤ b → m = b)) :
  m = 3 :=
by
  have h₃ : a ≤ b := by
    rw [h₁, h₂]
    exact le_of_lt (by norm_num)
  exact (program_logic.right h₃).trans h₂

end final_output_value_of_m_l500_500499


namespace equation1_solver_equation2_solver_l500_500262

-- Equation (1) proof statement in Lean 4
theorem equation1_solver (x : ℝ) :
  x = 1 + (sqrt 15) / 3 ∨ x = 1 - (sqrt 15) / 3 ↔ 3 * x^2 - 6 * x - 2 = 0 :=
by sorry

-- Equation (2) proof statement in Lean 4
theorem equation2_solver (x : ℝ) :
  x = (3 + sqrt 17) / 2 ∨ x = (3 - sqrt 17) / 2 ↔ x^2 - 3 * x - 2 = 0 :=
by sorry

end equation1_solver_equation2_solver_l500_500262


namespace fifth_term_arithmetic_sequence_l500_500302

variable (a d : ℤ)

def arithmetic_sequence (n : ℤ) : ℤ :=
  a + (n - 1) * d

theorem fifth_term_arithmetic_sequence :
  arithmetic_sequence a d 20 = 12 →
  arithmetic_sequence a d 21 = 15 →
  arithmetic_sequence a d 5 = -33 :=
by
  intro h20 h21
  sorry

end fifth_term_arithmetic_sequence_l500_500302


namespace solution_set_inequality_l500_500496

/-- Given a function f from ℝ to ℝ, if the solution set of f(x) > 0 is (-∞, 1) ∪ (2, ∞),
then the solution set of f(3^x) ≤ 0 is [0, log_3{2}] -/
theorem solution_set_inequality (f : ℝ → ℝ) :
  {x : ℝ | 0 ≤ x ∧ x ≤ log 3 2} = {x : ℝ | f (3^x) ≤ 0} :=
begin
  sorry
end

end solution_set_inequality_l500_500496


namespace train_speed_l500_500729

theorem train_speed :
  ∀ (length : ℝ) (time : ℝ),
    length = 135 ∧ time = 3.4711508793582233 →
    (length / time) * 3.6 = 140.0004 :=
by
  sorry

end train_speed_l500_500729


namespace arithmetic_sequence_Sn_l500_500300

noncomputable def S (n : ℕ) : ℕ := sorry -- S is the sequence function

theorem arithmetic_sequence_Sn {n : ℕ} (h1 : S n = 2) (h2 : S (3 * n) = 18) : S (4 * n) = 26 :=
  sorry

end arithmetic_sequence_Sn_l500_500300


namespace sausage_shop_period_l500_500267

theorem sausage_shop_period
  (strips_per_sandwich : ℕ)
  (time_per_sandwich : ℕ)
  (total_strips : ℕ)
  (h_strips : strips_per_sandwich = 4)
  (h_time : time_per_sandwich = 5)
  (h_total : total_strips = 48) :
  (total_strips / strips_per_sandwich) * time_per_sandwich = 60 := by
  sorry

end sausage_shop_period_l500_500267


namespace tan_alpha_value_l500_500943

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l500_500943


namespace determinant_is_zero_l500_500744

theorem determinant_is_zero (a b c d : ℝ) :
  det ![
    [a^2 * (a+1)^2 * (a+2)^2 * (a+3)^2],
    [b^2 * (b+1)^2 * (b+2)^2 * (b+3)^2],
    [c^2 * (c+1)^2 * (c+2)^2 * (c+3)^2],
    [d^2 * (d+1)^2 * (d+2)^2 * (d+3)^2]
  ] = 0 := 
by 
  sorry

end determinant_is_zero_l500_500744


namespace proof_l500_500165

noncomputable def condition1 (A B C a b c : ℝ) : Prop := 2 * Real.sin B * Real.cos C - 2 * Real.sin A + Real.sin C = 0

noncomputable def question1 (A B C a b c : ℝ) (h : condition1 A B C a b c) : Prop := B = Real.pi / 3

noncomputable def condition2 (b : ℝ) : Prop := b = 2

noncomputable def R (b : ℝ) (B : ℝ) (h_b : condition2 b) : ℝ := b / (2 * Real.sin (B / 2))

noncomputable def question2 (R b : ℝ) (h_b : condition2 b) : ℝ := Real.sqrt (R^2 - (b / 2)^2)

theorem proof (A B C a b c : ℝ) (h1 : condition1 A B C a b c) (h2 : condition2 b) :
  question1 A B C a b c h1 ∧ question2 (R b B h2) b h2 = Real.sqrt(3) / 3 :=
by sorry

end proof_l500_500165


namespace evaluate_x_from_geometric_series_l500_500441

theorem evaluate_x_from_geometric_series (x : ℝ) (h : ∑' n : ℕ, x ^ n = 4) : x = 3 / 4 :=
sorry

end evaluate_x_from_geometric_series_l500_500441


namespace string_tie_length_l500_500643

theorem string_tie_length {original_length given_length : ℕ} (h1 : original_length = 90) (h2 : given_length = 30) :
  let remaining_length := original_length - given_length in
  let tie_fraction := 8 / 15 in
  remaining_length * tie_fraction = 32 :=
by
  unfold remaining_length
  unfold tie_fraction
  sorry

end string_tie_length_l500_500643


namespace range_of_a_l500_500126

noncomputable def f (x : ℝ) : ℝ := -2 * x^5 - x^3 - 7 * x + 2

theorem range_of_a (a : ℝ) (h : f(a^2) + f(a - 2) > 4) : -2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l500_500126


namespace find_a_and_M_l500_500483

-- Define the inequality
def inequality (a x : ℝ) : Prop := x^2 + (a-4)*x - (a+1)*(2a-3) < 0

-- Condition: 0 is an element of M
def condition_one (a : ℝ) : Prop := inequality a 0

-- Computing the range of a
def range_of_a (a : ℝ) : Prop := a < -1 ∨ a > (3/2)

-- Expressing M in terms of a
def M (a x : ℝ) : Prop :=
  (a < -1 ∧ x ∈ Ioo (a+1) (3-2*a)) ∨ (a > (3/2) ∧ x ∈ Ioo (3-2*a) (a+1))

-- The main statement
theorem find_a_and_M (a : ℝ) :
  condition_one a → range_of_a a ∧ ∀ x, inequality a x ↔ M a x :=
sorry

end find_a_and_M_l500_500483


namespace probability_of_reaching_corner_l500_500790

-- Definitions for the problem
def edge_squares_without_corners : Set (ℕ × ℕ) := {(1, 0), (2, 0), (0, 1), (3, 1), (0, 2), (3, 2), (1, 3), (2, 3)}
def corner_squares : Set (ℕ × ℕ) := {(0, 0), (0, 3), (3, 0), (3, 3)}
def inner_squares : Set (ℕ × ℕ) := {(1, 1), (1, 2), (2, 1), (2, 2)}

-- Probability function assuming transition probabilities and recursive calculations
def p_n : ℕ → (ℕ × ℕ) → ℚ -- probability of reaching corner in n hops from position (i, j)
| 0, pos => if pos ∈ corner_squares then 1 else 0
| n + 1, pos => 
  let move_pos (i j : ℕ × ℕ) :=
    [(i, (j + 1) % 4), ((i + 1) % 4, j), (i, (j + 3) % 4), ((i + 3) % 4, j)]
  let next_positions := move_pos pos.1 pos.2
  (p_n n (next_positions[0]) + p_n n (next_positions[1]) + p_n n (next_positions[2]) + p_n n (next_positions[3])) / 4

-- Theorem: Probability of reaching a corner square within 5 hops is 57/64
theorem probability_of_reaching_corner : p_n 5 (2, 0) = 57 / 64 :=
by {
  sorry
}

end probability_of_reaching_corner_l500_500790


namespace rooms_needed_l500_500010

/-
  We are given that there are 30 students and each hotel room accommodates 5 students.
  Prove that the number of rooms required to accommodate all students is 6.
-/
theorem rooms_needed (total_students : ℕ) (students_per_room : ℕ) (h1 : total_students = 30) (h2 : students_per_room = 5) : total_students / students_per_room = 6 := by
  -- proof
  sorry

end rooms_needed_l500_500010


namespace Elisa_painting_ratio_l500_500235

variable (T : ℕ)
variable (Monday : ℕ := 30)
variable (Wednesday : ℕ := 15)
variable (Total : ℕ := 105)

theorem Elisa_painting_ratio :
  (30 + T + 15 = 105) → (T / 30 = 2) :=
by
  assume h1 : 30 + T + 15 = 105
  sorry

end Elisa_painting_ratio_l500_500235


namespace claire_speed_l500_500748

def distance := 2067
def time := 39

def speed (d : ℕ) (t : ℕ) : ℕ := d / t

theorem claire_speed : speed distance time = 53 := by
  sorry

end claire_speed_l500_500748


namespace point_symmetric_yOz_l500_500975

theorem point_symmetric_yOz (x y z : ℝ) (A : ℝ × ℝ × ℝ) (hA : A = (2, -3, 4)) :
  (x, y, z) = (-2, -3, 4) :=
  by
    rw [hA]
    sorry

end point_symmetric_yOz_l500_500975


namespace distance_A1_to_plane_DBEF_is_1_l500_500797

noncomputable def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def is_plane (a b c d : ℝ × ℝ × ℝ) : Prop :=
  ∃ n : ℝ × ℝ × ℝ, ∀ x ∈ {a, b, c, d}, 
    let (x1, x2, x3) := x;
    let (n1, n2, n3) := n in
    n1 * x1 + n2 * x2 + n3 * x3 = n1 * a.1 + n2 * a.2 + n3 * a.3

def distance_point_plane (p : ℝ × ℝ × ℝ) (a b c d : ℝ × ℝ × ℝ) : ℝ :=
  let v := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let w := (c.1 - a.1, c.2 - a.2, c.3 - a.3)
  let n := (v.2 * w.3 - v.3 * w.2, v.3 * w.1 - v.1 * w.3, v.1 * w.2 - v.2 * w.1)
  let plane_d := -(n.1 * a.1 + n.2 * a.2 + n.3 * a.3)
  let numerator := abs (n.1 * p.1 + n.2 * p.2 + n.3 * p.3 + plane_d)
  let denominator := (n.1 ^ 2 + n.2 ^ 2 + n.3 ^ 2).sqrt
  numerator / denominator

theorem distance_A1_to_plane_DBEF_is_1 :
  let A := (0, 0, 0)
  let B := (1, 0, 0)
  let C := (1, 1, 0)
  let D := (0, 1, 0)
  let A1 := (0, 0, 1)
  let B1 := (1, 0, 1)
  let C1 := (1, 1, 1)
  let D1 := (0, 1, 1)
  let E := midpoint B1 C1
  let F := midpoint C1 D1
  is_plane D B E F → distance_point_plane A1 D B E F = 1 :=
by
  sorry

end distance_A1_to_plane_DBEF_is_1_l500_500797


namespace largest_three_digit_multiple_of_8_with_digit_sum_24_l500_500658

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (8 ∣ n) ∧ nat.digits 10 n.sum = 24 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ (8 ∣ m) ∧ nat.digits 10 m.sum = 24 → m ≤ n :=
begin
  sorry
end

end largest_three_digit_multiple_of_8_with_digit_sum_24_l500_500658


namespace length_of_boat_l500_500686

-- Definitions based on the conditions
def breadth : ℝ := 3
def sink_depth : ℝ := 0.01
def man_mass : ℝ := 120
def g : ℝ := 9.8 -- acceleration due to gravity

-- Derived from the conditions
def weight_man : ℝ := man_mass * g
def density_water : ℝ := 1000

-- Statement to be proved
theorem length_of_boat : ∃ L : ℝ, (breadth * sink_depth * L * density_water * g = weight_man) → L = 4 :=
by
  sorry

end length_of_boat_l500_500686


namespace tan_alpha_value_l500_500882

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l500_500882


namespace solve_trig_equation_l500_500596

noncomputable def validate_solution (x k : ℝ) : Prop :=
  cos (11 * x) - cos (3 * x) - sin (11 * x) + sin (3 * x) = sqrt 2 * cos (14 * x) ↔
  (x = -π / 28 + (π * k) / 7) ∨ (x = π / 12 + (2 * π * k) / 3) ∨ (x = 5 * π / 44 + (2 * π * k) / 11)

theorem solve_trig_equation :
  ∀ x k : ℝ, validate_solution x k :=
by
  intros x k
  sorry

end solve_trig_equation_l500_500596


namespace tan_alpha_solution_l500_500871

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l500_500871


namespace ratio_front_to_top_l500_500049

theorem ratio_front_to_top (l w h : ℝ) 
  (V : l * w * h = 192)
  (A_top : l * w = 1.5 * w * h) 
  (A_side : w * h ≈ 32) :
  (h / w) = 1 / 2 :=
by sorry

end ratio_front_to_top_l500_500049


namespace find_segment_CE_l500_500755

-- Definitions and conditions from the problem statement

noncomputable def is_isosceles {A B C : ℝ} (AB AC : ℝ) : Prop :=
AB = AC

noncomputable def altitude_length_to_base (AD : ℝ) : Prop :=
AD = 3

noncomputable def angle_values {A C E B : ℝ} (angle_ACE angle_ECB : ℝ) : Prop :=
angle_ACE = 18 ∧ angle_ECB = 18

-- Given that triangle ABC is isosceles, AD is the altitude from A, and angles are given,
-- prove the length of segment CE.
theorem find_segment_CE {A B C D E : ℝ} (AB AC AD : ℝ) (angle_ACE angle_ECB : ℝ) 
  (h1 : is_isosceles AB AC) 
  (h2 : altitude_length_to_base AD) 
  (h3 : angle_values angle_ACE angle_ECB) :
  ∃ CE : ℝ, -- The existence of CE
  CE = sorry := -- Placeholder for resulting length of CE
begin
  sorry -- The proof steps would be here
end

end find_segment_CE_l500_500755


namespace last_triangle_perimeter_l500_500212

noncomputable def T₁ : Triangle := { a := 1011, b := 1012, c := 1013 }

noncomputable def perimeter (T : Triangle) : ℚ :=
  (T.a + T.b + T.c : ℚ)

theorem last_triangle_perimeter :
  let Tn_recursive (n : ℕ) (T : Triangle) : Triangle :=
    { a := (T.b + T.c - T.a) / 2, b := (T.a + T.c - T.b) / 2, c := (T.a + T.b - T.c) / 2 }
  let sequences := list.iterate (Tn_recursive (n+1)) ⟨T₁⟩ in
  n = 10 →
  perimeter (sequences.get_last) = 759 / 128 :=
by
  sorry

end last_triangle_perimeter_l500_500212


namespace inequality_and_equality_condition_l500_500558

open Nat

-- Define the Euler's totient function
def euler_totient (n : ℕ) : ℕ := n.totient

-- Define the function f
def f (n : ℕ) : ℕ :=
  {x : ℕ | x ∈ Finset.range n.succ ∧ (gcd (x, n) = 1 ∨ is_prime (gcd (x, n)))}.card

-- Define the function s
def s (d : ℕ) : ℕ :=
  {x : ℕ | x ∈ Finset.range d.succ ∧ is_prime (gcd (x, d))}.card

theorem inequality_and_equality_condition (n : ℕ) :
  (∑ d in (Finset.divisors n), f d) +  euler_totient n ≥ 2 * n ∧ 
  (∀ n, (∑ d in (Finset.divisors n), f d) +  euler_totient n = 2 * n ↔ is_prime n) := 
by
  sorry

end inequality_and_equality_condition_l500_500558


namespace find_pool_depth_l500_500604

noncomputable def pool_depth (rate volume capacity_percent time length width : ℝ) :=
  volume / (length * width * rate * time / capacity_percent)

theorem find_pool_depth :
  pool_depth 60 75000 0.8 1000 150 50 = 10 := by
  simp [pool_depth] -- Simplifying the complex expression should lead to the solution.
  sorry

end find_pool_depth_l500_500604


namespace find_number_satisfies_equation_l500_500626

theorem find_number_satisfies_equation :
  let a := (1440 / 7 : ℚ) in
  (128⁻¹ * a + (9 / 7)) / (5 - (4 * (2 / 21) * 0.75)) / (((1 / 3) + (5 / 7) * 1.4) / ((4 - 2 * (2 / 3)) * 3)) = 4.5 :=
sorry

end find_number_satisfies_equation_l500_500626


namespace total_selling_price_correct_l500_500728

def meters_sold : ℕ := 85
def cost_price_per_meter : ℕ := 80
def profit_per_meter : ℕ := 25

def selling_price_per_meter : ℕ :=
  cost_price_per_meter + profit_per_meter

def total_selling_price : ℕ :=
  selling_price_per_meter * meters_sold

theorem total_selling_price_correct :
  total_selling_price = 8925 := by
  sorry

end total_selling_price_correct_l500_500728


namespace distance_A_to_plane_is_50_div_7_l500_500449

-- Define the point A
def A : ℝ × ℝ × ℝ := (2, 3, -4)

-- Define the plane equation
def plane (x y z : ℝ) : ℝ := 2 * x + 6 * y - 3 * z + 16

-- Define the distance formula from a point to a plane
noncomputable def distance_from_point_to_plane (p : ℝ × ℝ × ℝ) (a b c d : ℝ) : ℝ :=
  let (x1, y1, z1) := p in
  (abs (a * x1 + b * y1 + c * z1 + d)) / (Real.sqrt (a * a + b * b + c * c))

-- The actual statement of the proof problem
theorem distance_A_to_plane_is_50_div_7 :
  distance_from_point_to_plane A 2 6 (-3) 16 = 50 / 7 := 
by 
  sorry

end distance_A_to_plane_is_50_div_7_l500_500449


namespace correct_propositions_l500_500125

-- Define proposition 1
def prop1 (p : (∀ x : ℝ, x ≥ 0 → x^2 + x ≥ 0) → False) : Prop :=
  (∃ x0 : ℝ, x0 < 0 ∧ x0^2 + x0 < 0)

-- Define proposition 2
def prop2 (lin_rel : (∀ x y : ℝ, ∃ a b : ℝ, y = a * x + b) ∧ (∀ x y : ℝ, x + y = 2) → False) : Prop :=
  False

-- Define proposition 3
def prop3 (BC AC : ℝ) (angleB : ℝ) : Prop :=
  BC = 2 ∧ AC = 3 ∧ angleB = Real.pi / 3 → (BC^2 + AC^2 - 2 * BC * AC * Real.cos angleB) > 0

-- Define proposition 4
def prop4 (l : ℝ) : Prop :=
  l = 8 → (λ x : ℝ, 0 < x ∧ x < l/2) → (((x * (l/2 - x)) > 3) → 1/2)

-- Define proposition 5
def prop5 (a b c : ℝ) : Prop :=
  a > b ∧ b > c ∧ c > 0 ∧ 2 * b > a + c → (b / (a - b) > c / (b - c))

-- The main theorem to prove which propositions are correct
theorem correct_propositions : 
  prop1 ∧ prop2 ∧ prop3 2 3 (Real.pi / 3) ∧ prop4 8 ∧ prop5 :=
  by
    -- These are placeholders as we are not required to prove in this task
    trivial

end correct_propositions_l500_500125


namespace independence_of_A_and_D_l500_500325

noncomputable def balls : Finset ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def draw_one : ℕ := (1 : ℕ)
noncomputable def draw_two : ℕ := (2 : ℕ)

def event_A : ℕ → Prop := λ n, n = 1
def event_B : ℕ → Prop := λ n, n = 2
def event_C : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 8
def event_D : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 7

def prob (event : ℕ → Prop) : ℚ := 1 / 6
def joint_prob (event1 event2 : ℕ → Prop) : ℚ := (1 / 36)

theorem independence_of_A_and_D :
  joint_prob (λ n, event_A n) (λ n, event_D (draw_one, draw_two)) = prob event_A * prob (λ n, event_D (draw_one, draw_two)) :=
by
  sorry

end independence_of_A_and_D_l500_500325


namespace part1_part2_l500_500785

-- Definitions provided by the problem
def f (n : ℕ) : ℕ :=
  minimize (cardinality S)
  where
    S : set ℕ := {s | s ∈ S ∧ 1 ∈ S ∧ n ∈ S ∧ ∀ x ∈ S, x > 1 → ∃ a b ∈ S, x = a + b}

-- Part 1
theorem part1 (n : ℕ) (h : n ≥ 2) : f(n) ≥ (log2 n).toFloor + 1 := 
by sorry

-- Part 2
theorem part2 : ∃ᶠ n in (at_top : filter ℕ), f(n) = f(n + 1) := 
by sorry

end part1_part2_l500_500785


namespace simplify_expression_l500_500259

theorem simplify_expression :
  (sqrt (3 + 4 + 5 + 6) / 3) + ((3 * 4 + 10) / 4) = sqrt 2 + 5.5 :=
by
  sorry

end simplify_expression_l500_500259


namespace number_of_pickers_l500_500143

theorem number_of_pickers (drums_per_day : ℕ) (total_days : ℕ) (total_drums : ℕ) (h1 : drums_per_day = 108) (h2 : total_days = 58) (h3 : total_drums = 6264) : total_drums / drums_per_day = 58 :=
by
  rw [h1, h2, h3]
  exact rfl

end number_of_pickers_l500_500143


namespace find_a_and_b_solve_inequality_l500_500793

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

theorem find_a_and_b (a b : ℝ) (h : ∀ x : ℝ, f x a b > 0 ↔ x < 0 ∨ x > 2) : a = -2 ∧ b = 0 :=
by sorry

theorem solve_inequality (a b : ℝ) (m : ℝ) (h1 : a = -2) (h2 : b = 0) :
  (∀ x : ℝ, f x a b < m^2 - 1 ↔ 
    (m = 0 → ∀ x : ℝ, false) ∧
    (m > 0 → (1 - m < x ∧ x < 1 + m)) ∧
    (m < 0 → (1 + m < x ∧ x < 1 - m))) :=
by sorry

end find_a_and_b_solve_inequality_l500_500793


namespace meeting_point_distance_from_midpoint_l500_500342

theorem meeting_point_distance_from_midpoint
    (d_AB : ℝ) (d_AB = 240)
    (v1 : ℝ) (v1 = 50)
    (v2 : ℝ) (v2 = 90) :
    ∃ t d1 d2 : ℝ,
      (d1 = v1 * t) ∧
      (d2 = v2 * t) ∧
      (d1 + d2 = d_AB) ∧
      abs (d1 - 120) = 34.29 := 
sorry

end meeting_point_distance_from_midpoint_l500_500342


namespace function_monotonicity_l500_500129

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_monotonicity :
  ∀ x₁ x₂, -Real.pi / 6 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ Real.pi / 3 → f x₁ ≤ f x₂ :=
by
  sorry

end function_monotonicity_l500_500129


namespace find_q_l500_500069

theorem find_q (q : ℝ → ℝ) : 
  (∀ x : ℝ, q(x) = -(x-3)*(x+1)*x) → 
  (∀ a b c : ℝ, a = x → b = 3 → c = -1 → 
    ((∃ d : ℝ, q(4) = d ∧ d = -20) ∧ 
     polynomial.degree (polynomial.monomial 3 1) = polynomial.degree (polynomial.monomial 3 1))) :=
begin
  intro h,
  intros a b c ha hb hc,
  use -20,
  split,
  {
    simp only [ha, hb, hc, h] at *,
    norm_num,
  },
  {
    simp,
  }
end

end find_q_l500_500069


namespace angle_BAD_obtuse_l500_500303

theorem angle_BAD_obtuse (A B C D N M : Point)
    (midpoint_BC : ∀ (x : Point), midpoint x B C = N)
    (midpoint_CD : ∀ (x : Point), midpoint x C D = M)
    (parallelogram_ABCD : parallelogram A B C D)
    (segment_condition : segment A N = 2 * segment A M) :
    angle A B D > 90 :=
by
  sorry

end angle_BAD_obtuse_l500_500303


namespace buffon_deviation_random_l500_500742

noncomputable def buffon_coin_toss (n : ℕ) (X : ℕ) (mu : ℕ) : Prop :=
  ∃ ε : ℝ, n = 4040 ∧ X = 2048 ∧ mu = 2020 ∧ ε = 28 ∧ 
    P (|(X : ℝ) - (mu : ℝ)| ≥ ε) ≈ 0.3783

theorem buffon_deviation_random : buffon_coin_toss 4040 2048 2020 :=
by
  -- conditions
  use 28
  split
  { exact rfl } -- n = 4040
  split
  { exact rfl } -- X = 2048
  split
  { exact rfl } -- mu = 2020
  split
  { exact rfl } -- ε = 28
  sorry -- P(|X - mu| ≥ ε) ≈ 0.3783

end buffon_deviation_random_l500_500742


namespace rounding_range_3_58_l500_500391

noncomputable def rounded_to_two_decimals (x : ℝ) : ℝ :=
  Float.toRational ((Float.ofReal x).round 2)

theorem rounding_range_3_58 (x : ℝ) (h : rounded_to_two_decimals x = 3.58) :
  3.575 ≤ x ∧ x ≤ 3.584 :=
sorry

end rounding_range_3_58_l500_500391


namespace saturated_function_1_check_l500_500953

def saturated_function_1 (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

def function_1 : ℝ → ℝ := λ x, 1 / x
def function_2 : ℝ → ℝ := λ x, 2^x
def function_3 : ℝ → ℝ := λ x, Real.log (x^2 + 2)
def function_4 : ℝ → ℝ := λ x, Real.cos (Real.pi * x)

theorem saturated_function_1_check :
  saturated_function_1 function_2 ∧ saturated_function_1 function_4 ∧ 
  ¬ saturated_function_1 function_1 ∧ ¬ saturated_function_1 function_3 :=
by
  sorry

end saturated_function_1_check_l500_500953


namespace exists_disjoint_chords_same_sum_l500_500348

theorem exists_disjoint_chords_same_sum:
  ∃ (chords : Finset (ℕ × ℕ)), 
  chords.card = 100 ∧ 
  ∀ (c ∈ chords), 
  (c.1 + c.2) ∈ {s : ℕ | 3 ≤ s ∧ s ≤ 2 ^ 501 - 1} ∧ 
  (∀ (c1 c2 ∈ chords), c1 ≠ c2 → ¬ are_intersecting c1 c2) :=
sorry

def are_intersecting (c1 c2 : ℕ × ℕ) : Prop :=
  let a := min c1.1 c1.2 in
  let b := max c1.1 c1.2 in
  let x := min c2.1 c2.2 in
  let y := max c2.1 c2.2 in
  (a < x ∧ x < b ∧ b < y) ∨ (x < a ∧ a < y ∧ y < b)

end exists_disjoint_chords_same_sum_l500_500348


namespace tan_alpha_solution_l500_500870

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l500_500870


namespace maximum_value_of_2ab_2bc_sqrt2_l500_500215

noncomputable def max_value_ab_bc_sqrt2 (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : ℝ :=
  2 * a * b + 2 * b * c * real.sqrt 2

theorem maximum_value_of_2ab_2bc_sqrt2 (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  a ≥ 0 → b ≥ 0 → c ≥ 0 → max_value_ab_bc_sqrt2 a b c h ≤ real.sqrt (3 / 2) :=
  sorry

end maximum_value_of_2ab_2bc_sqrt2_l500_500215


namespace min_value_of_expression_l500_500112

theorem min_value_of_expression (x y : ℝ) (h1 : xy + 3 * x = 3) (h2 : 0 < x ∧ x < 1/2) : 
  ∃ x y, (xy + 3 * x = 3 ∧ 0 < x ∧ x < 1/2) ∧ ∀ x y, xy + 3 * x = 3 → 0 < x → x < 1/2 → (3 / x + 1 / (y - 3)) ≥ 8 :=
by 
  sorry

end min_value_of_expression_l500_500112


namespace value_of_g_neg3_l500_500423

def g (x : ℝ) : ℝ := x^3 - 2 * x

theorem value_of_g_neg3 : g (-3) = -21 := by
  sorry

end value_of_g_neg3_l500_500423


namespace two_integer_solutions_iff_m_l500_500089

def op (p q : ℝ) : ℝ := p + q - p * q

theorem two_integer_solutions_iff_m (m : ℝ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ op 2 x1 > 0 ∧ op x1 3 ≤ m ∧ op 2 x2 > 0 ∧ op x2 3 ≤ m) ↔ 3 ≤ m ∧ m < 5 :=
by
  sorry

end two_integer_solutions_iff_m_l500_500089


namespace eq_iff_solution_l500_500094

theorem eq_iff_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x^y + y^x = x^x + y^y ↔ x = y :=
by sorry

end eq_iff_solution_l500_500094


namespace mod_inv_sum_l500_500349

theorem mod_inv_sum :
  (let inv5 := Nat.gcdA 5 13 % 13 in
   let inv52 := (inv5 * inv5) % 13 in
   (inv5 + inv52) % 13 = 7) :=
by
  let inv5 := Nat.gcdA 5 13 % 13 -- This calculates the modular inverse of 5 modulo 13
  let inv52 := (inv5 * inv5) % 13 -- This calculates (inv5)^2 modulo 13
  have h1 : inv5 = 8 := by
    -- Proof omitted for brevity
    sorry
  have h2 : inv52 = 12 := by
    -- Proof omitted for brevity
    sorry
  show (inv5 + inv52) % 13 = 7 from
    calc
      (inv5 + inv52) % 13 = (8 + 12) % 13 := by rw [h1, h2]
                       ... = 20 % 13 := by simp
                       ... = 7 := rfl

end mod_inv_sum_l500_500349


namespace integral_x_cubed_from_neg3_to_3_eq_zero_l500_500771

theorem integral_x_cubed_from_neg3_to_3_eq_zero :
  ∫ x in -3..3, x^3 = 0 :=
sorry

end integral_x_cubed_from_neg3_to_3_eq_zero_l500_500771


namespace num_correct_propositions_l500_500735

theorem num_correct_propositions :
  let P1 := (∀ x y : ℝ, x = y → sin x = sin y) -- proposition 1
  let P2 := ∀ (α β : set ℝ³) (m n : ℝ³), α ≠ β ∧ m ∥ α ∧ n ∥ β ∧ α ⟂ β → ¬ (m ⟂ n) -- proposition 2
  let P3 := ∀ (a : ℝ), (∃ (a1 a2 : ℝ), a1 = 2*a ∧ a2 = 1) → (∀ (b1 b2 : ℝ), b1 = 1 ∧ b2 = 2*a) ∧ (a ≠ 1/2 ∧ a ≠ -1/2) -- proposition 3
  let P4 := ∫ x in -1..1, sin x = 0 -- proposition 4
  (ite P1 1 0) + (ite P2 1 0) + (ite P3 1 0) + (ite P4 1 0) = 2 :=
by sorry

end num_correct_propositions_l500_500735


namespace height_larger_cylinder_l500_500700

noncomputable def volume_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

/-- Let V_large be the volume of the larger cylinder and V_small be the
volume of the smaller cylinder. The diameter of the larger cylinder is 6 meters,
the radius of the smaller cylinders is 2 meters, and the height of the smaller cylinders
is 5 meters. Also, the larger cylinder can fill 3 smaller cylinders.
We prove that h, the height of the larger cylinder, is 20/3 or 6.67 meters.-/
theorem height_larger_cylinder : 
  ∀ (d_large : ℝ) (r_small h_small : ℝ) (n_large_fills : ℝ) (h_large : ℝ),
  d_large = 6 ∧ r_small = 2 ∧ h_small = 5 ∧ n_large_fills = 3 ∧ 
  (volume_cylinder (d_large / 2) h_large = n_large_fills * volume_cylinder r_small h_small) →
  h_large = 20 / 3 :=
begin
  intros d_large r_small h_small n_large_fills h_large conditions,
  rw [volume_cylinder, volume_cylinder] at conditions,
  sorry,  -- Proof goes here
end

end height_larger_cylinder_l500_500700


namespace total_profit_calculation_l500_500587

-- Define the parameters of the problem
def rajan_investment : ℕ := 20000
def rakesh_investment : ℕ := 25000
def mukesh_investment : ℕ := 15000
def rajan_investment_time : ℕ := 12 -- in months
def rakesh_investment_time : ℕ := 4 -- in months
def mukesh_investment_time : ℕ := 8 -- in months
def rajan_final_share : ℕ := 2400

-- Calculation for total profit
def total_profit (rajan_investment rakesh_investment mukesh_investment
                  rajan_investment_time rakesh_investment_time mukesh_investment_time
                  rajan_final_share : ℕ) : ℕ :=
  let rajan_share := rajan_investment * rajan_investment_time
  let rakesh_share := rakesh_investment * rakesh_investment_time
  let mukesh_share := mukesh_investment * mukesh_investment_time
  let total_investment := rajan_share + rakesh_share + mukesh_share
  (rajan_final_share * total_investment) / rajan_share

-- Proof problem statement
theorem total_profit_calculation :
  total_profit rajan_investment rakesh_investment mukesh_investment
               rajan_investment_time rakesh_investment_time mukesh_investment_time
               rajan_final_share = 4600 :=
by sorry

end total_profit_calculation_l500_500587


namespace integer_values_of_b_for_polynomial_root_l500_500775

theorem integer_values_of_b_for_polynomial_root
    (b : ℤ) :
    (∃ x : ℤ, x^3 + 6 * x^2 + b * x + 12 = 0) ↔
    b = -217 ∨ b = -74 ∨ b = -43 ∨ b = -31 ∨ b = -22 ∨ b = -19 ∨
    b = 19 ∨ b = 22 ∨ b = 31 ∨ b = 43 ∨ b = 74 ∨ b = 217 :=
    sorry

end integer_values_of_b_for_polynomial_root_l500_500775


namespace identify_logical_structure_is_conditional_l500_500400

def logicalStructure (entryPoint: Nat) (exitPoints: Nat): Type :=
  if entryPoint = 1 ∧ exitPoints = 2 then "Conditional Structure"
  else "Other Structure"

theorem identify_logical_structure_is_conditional (e: Nat) (ex: Nat) :
  e = 1 → ex = 2 → logicalStructure e ex = "Conditional Structure" :=
by
  intro h1 h2
  rw [logicalStructure]
  rw [h1,h2]
  sorry

end identify_logical_structure_is_conditional_l500_500400


namespace tan_alpha_fraction_l500_500913

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l500_500913


namespace total_volume_is_333_l500_500746

def cube_volume (side : ℕ) := side ^ 3

def Carl_volumes := [cube_volume 1, cube_volume 2, cube_volume 3, cube_volume 4, cube_volume 5]
def Kate_volumes := [cube_volume 3] * 4

def Carl_total_volume := Carl_volumes.sum
def Kate_total_volume := Kate_volumes.sum
def total_volume := Carl_total_volume + Kate_total_volume

theorem total_volume_is_333 :
  total_volume = 333 := by
  sorry

end total_volume_is_333_l500_500746


namespace largest_three_digit_multiple_of_8_with_sum_24_l500_500654

theorem largest_three_digit_multiple_of_8_with_sum_24 :
  ∃ n : ℕ, (n ≥ 100 ∧ n < 1000) ∧ (∃ k, n = 8 * k) ∧ (n.digits.sum = 24) ∧
           ∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (∃ k', m = 8 * k') ∧ (m.digits.sum = 24) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_8_with_sum_24_l500_500654


namespace sum_base6_to_base10_l500_500782

theorem sum_base6_to_base10 :
  let n := 36 in
  (∑ k in finset.range (n + 1), k) = 666 :=
  sorry

end sum_base6_to_base10_l500_500782


namespace tan_alpha_fraction_l500_500909

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l500_500909


namespace older_brother_older_than_twice_younger_age_l500_500234

theorem older_brother_older_than_twice_younger_age :
  ∃ (M O Y : ℕ), 
  Y = 5 ∧
  O = 3 * Y ∧
  M + O + Y = 28 ∧
  O - 2 * (M - 1) = 1 :=
by {
  use 8, 15, 5,
  split, exact rfl,
  split, exact rfl,
  split, norm_num,
  norm_num,
  sorry
}

end older_brother_older_than_twice_younger_age_l500_500234


namespace angle_BAE_is_135_degrees_l500_500533

variables (A B C D E : Type) [Euclidean_geometry]
variables {a b c d e : Point A} -- Point needs to be defined, placeholders used
variables (ABC : Triangle a b c) (BCDE : Square b c d e)

-- Equilateral triangle condition
axiom equilateral_triangle_ABC : triangle.equilateral ABC

-- Square construction on line
axiom square_construction : square BCDE -- essentially implies side lengths are equal, angles are 90°

-- Condition stating D and E lie on line AC with D closer to C
axiom points_on_AC : line a c ∋ d ∧ line a c ∋ e ∧ distance a d > distance a c

-- The aim is to conclude the value of angle BAE
theorem angle_BAE_is_135_degrees (ABC_equilateral_condition : triangle.equilateral ABC) 
                                 (BCD_square_condition : square BCDE) 
                                 (point_cond : line a c ∋ d ∧ line a c ∋ e ∧ distance a d > distance a e)
                                 : angle BAE = 135 :=
by
  sorry

end angle_BAE_is_135_degrees_l500_500533


namespace count_equilateral_triangles_in_lattice_l500_500765

def Point : Type := ℝ × ℝ
def hexagonal_lattice : set Point :=
  { (x, y) | ∃ i j : ℤ, ((i % 6 == 0 ∧ j % 6 == 0) ∨
                        (i % 6 == 1 ∧ j % 6 == 1) ∨
                        (i % 6 == 2 ∧ j % 6 == 2) ∨
                        (i % 6 == 3 ∧ j % 6 == 3) ∨
                        (i % 6 == 4 ∧ j % 6 == 4) ∨
                        (i % 6 == 5 ∧ j % 6 == 5) ∧
                        dist (x, y) (x + 1, y) = 1)}

def is_equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

theorem count_equilateral_triangles_in_lattice (lattice : set Point) (H : lattice = hexagonal_lattice) :
   ∃ n : ℕ, n = 8 :=
sorry

end count_equilateral_triangles_in_lattice_l500_500765


namespace solve_abs_inequality_l500_500085

/-- Given the inequality 2 ≤ |x - 3| ≤ 8, we want to prove that the solution is [-5 ≤ x ≤ 1] ∪ [5 ≤ x ≤ 11] --/
theorem solve_abs_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
sorry

end solve_abs_inequality_l500_500085


namespace equal_share_of_candles_l500_500026

-- Define conditions
def ambika_candles : ℕ := 4
def aniyah_candles : ℕ := 6 * ambika_candles
def bree_candles : ℕ := 2 * aniyah_candles
def caleb_candles : ℕ := bree_candles + (bree_candles / 2)

-- Define the total candles and the equal share
def total_candles : ℕ := ambika_candles + aniyah_candles + bree_candles + caleb_candles
def each_share : ℕ := total_candles / 4

-- State the problem
theorem equal_share_of_candles : each_share = 37 := by
  sorry

end equal_share_of_candles_l500_500026


namespace tan_alpha_sqrt_15_over_15_l500_500915

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l500_500915


namespace maximum_possible_scores_l500_500169

theorem maximum_possible_scores (correct wrong unanswered : ℕ) :
  (correct + wrong + unanswered = 10) → 
  (score = correct * 4 - wrong) → 
  set.univ.to_finset.card = 45 :=
sorry

end maximum_possible_scores_l500_500169


namespace find_possible_values_of_y_l500_500221

theorem find_possible_values_of_y (x : ℝ) 
  (h : x^2 + 9 * (x / (x - 3))^2 = 54) : 
  let y := (x - 3)^2 * (x + 4) / (2 * x - 4) in 
  y = 11.25 ∨ y = 10.125 :=
sorry

end find_possible_values_of_y_l500_500221


namespace find_certain_number_l500_500288

theorem find_certain_number (x : ℕ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 62) 
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) : 
  x = 58 := by
  sorry

end find_certain_number_l500_500288


namespace max_cos_alpha_l500_500676

open Real

-- Define the condition as a hypothesis
def cos_sum_eq (α β : ℝ) : Prop :=
  cos (α + β) = cos α + cos β

-- State the maximum value theorem
theorem max_cos_alpha (α β : ℝ) (h : cos_sum_eq α β) : ∃ α, cos α = sqrt 3 - 1 :=
by
  sorry   -- Proof is omitted

#check max_cos_alpha

end max_cos_alpha_l500_500676


namespace hexagon_area_q_equality_l500_500561

def is_distinct (l : List ℕ) : Prop :=
  l.nodup

variables {P Q R S T U V : ℝ × ℝ}
variables (q : ℝ) (x y : ℕ)

noncomputable def P := (0, 0)
noncomputable def Q := (q, 3)
noncomputable def vertices_y_coords := [0, 3, 6, 9, 12, 15]

def convex_eq_hexagon : Prop :=
  geom_euclidean.triangle_area P Q V ≠ 0 ∧
  geom_euclidean.triangle_area Q R S ≠ 0 ∧
  geom_euclidean.triangle_area R S T ≠ 0 ∧
  geom_euclidean.triangle_area S T U ≠ 0 ∧
  geom_euclidean.triangle_area T U V ≠ 0 ∧
  geom_euclidean.triangle_area U V P ≠ 0

def angle_VPQ := (angle P Q V) = 120

def parallel_PQ_ST := (P.x = Q.x) ≠ (S.x = T.x)
def parallel_QR_TU := (Q.x = R.x) ≠ (T.x = U.x)
def parallel_RS_UV := (R.x = S.x) ≠ (U.x = V.x)

def all_coords_distinct := is_distinct vertices_y_coords

theorem hexagon_area_q_equality :
  convex_eq_hexagon ∧
  angle_VPQ ∧
  parallel_PQ_ST ∧
  parallel_QR_TU ∧
  parallel_RS_UV ∧
  all_coords_distinct →
  (∃ x y : ℕ, x + y = 111) :=
by { sorry }

end hexagon_area_q_equality_l500_500561


namespace correct_propositions_l500_500826

-- Defining propositions as conditions
def proposition1 : Prop := ∀ (P Q : Plane), (∃ x : Point, x ∈ P ∧ x ∈ Q) → P = Q
def proposition2 : Prop := ∀ (P Q : Plane) (L : Line), (L ⊥ Q) → (P ∋ L) → P ⊥ Q
def proposition3 : Prop := ∀ (L M N : Line), (L ⊥ N) ∧ (M ⊥ N) → (L ∥ M)
def proposition4 : Prop := ∀ (P Q : Plane) (L : Line), P ⊥ Q → (L ∈ P) ∧ (L ⊥ (P ∩ Q)) → (L ⊥ Q)

-- Defining the proof problem
theorem correct_propositions :
  proposition2 ∧ proposition4 ∧
  ¬proposition1 ∧
  ¬proposition3 :=
by sorry

end correct_propositions_l500_500826


namespace middle_term_arithmetic_sequence_l500_500818

theorem middle_term_arithmetic_sequence (m : ℝ) (h : 2 * m = 1 + 5) : m = 3 :=
by
  sorry

end middle_term_arithmetic_sequence_l500_500818


namespace intersect_ellipse_l500_500799

-- Given conditions
def ellipse : Set (ℝ × ℝ) := {(x, y) | x^2 + 2*y^2 = 2}

def intersects (l : ℝ → ℝ) : Prop := 
  ∃ P1 P2 : ℝ × ℝ, P1 ∈ ellipse ∧ P2 ∈ ellipse ∧ 
  ∃ k1 : ℝ, k1 ≠ 0 ∧ ∀ x y, l x = y → (y - P1.2) / (x - P1.1) = k1

def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

def slope (O P : ℝ × ℝ) : ℝ := P.2 / P.1

-- Lean theorem
theorem intersect_ellipse (l : ℝ → ℝ) (h_l : intersects l) 
  (k1 k2 : ℝ) (P1 P2 : ℝ × ℝ) (M : ℝ × ℝ)
  (h_P1 : P1 ∈ ellipse) (h_P2 : P2 ∈ ellipse) 
  (h_midpoint : M = midpoint P1 P2) 
  (O : ℝ × ℝ) (h_O : O = (0, 0))  
  (h_k1 : ∀ x y, l x = y → (y - P1.2) / (x - P1.1) = k1) 
  (h_k2 : k2 = slope O M)
  : k1 * k2 = -1/2 := 
sorry

end intersect_ellipse_l500_500799


namespace total_students_l500_500329

-- Definition of the given conditions
def M : ℕ := 50
def E : ℕ := 4 * M - 3
def H : ℕ := 2 * E
def Total : ℕ := E + M + H

-- The statement to prove
theorem total_students : Total = 641 := by
  sorry

end total_students_l500_500329


namespace log_expression_simplification_fraction_value_l500_500404

-- Defining the standard logarithm properties (Part 1) and assumptions (Part 2)

-- Part 1: Logarithmic expression simplification
theorem log_expression_simplification :
  log 5 * (log 8 + log 1000) + (log (2 ^ sqrt 3)) ^ 2 = 3 := 
sorry

-- Part 2: Given a - a⁻¹ = 2, find the specified fraction value
variable (a : ℝ)

-- Condition for part 2
axiom inv_condition : a - a⁻¹ = 2

theorem fraction_value (h : a - a⁻¹ = 2) :
  (a^3 + a⁻³) * (a^2 + a⁻² - 3) / (a^4 - a⁻⁴) = 5 / 4 := 
sorry

end log_expression_simplification_fraction_value_l500_500404


namespace sqrt_three_plus_two_mul_sqrt_three_minus_two_eq_neg_one_l500_500745

theorem sqrt_three_plus_two_mul_sqrt_three_minus_two_eq_neg_one :
  (\sqrt 3 + 2) * (\sqrt 3 - 2) = -1 :=
by
  sorry

end sqrt_three_plus_two_mul_sqrt_three_minus_two_eq_neg_one_l500_500745


namespace equation1_solution_equation2_solution_l500_500261

theorem equation1_solution (x : ℝ) (h : 3 * x - 1 = x + 7) : x = 4 := by
  sorry

theorem equation2_solution (x : ℝ) (h : (x + 1) / 2 - 1 = (1 - 2 * x) / 3) : x = 5 / 7 := by
  sorry

end equation1_solution_equation2_solution_l500_500261


namespace cell_phone_bill_l500_500691

-- Definitions
def base_cost : ℝ := 20
def cost_per_text : ℝ := 0.05
def cost_per_extra_minute : ℝ := 0.10
def texts_sent : ℕ := 100
def hours_talked : ℝ := 30.5
def included_hours : ℝ := 30

-- Calculate extra minutes used
def extra_minutes : ℝ := (hours_talked - included_hours) * 60

-- Total cost calculation
def total_cost : ℝ := 
  base_cost + 
  (texts_sent * cost_per_text) + 
  (extra_minutes * cost_per_extra_minute)

-- Proof problem statement
theorem cell_phone_bill : total_cost = 28 := by
  sorry

end cell_phone_bill_l500_500691


namespace distance_from_center_to_line_is_one_l500_500614

/-- Define the circle equation and the line equation as given conditions -/
def circle : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 - 4*x + 2*y = 0

def line : ℝ → ℝ → Prop :=
  λ x y, 3*x + 4*y + 3 = 0

/-- Define the center of the circle -/
def center : ℝ × ℝ := (2, -1)

/-- Prove the distance from the center of the circle to the line is 1 -/
theorem distance_from_center_to_line_is_one :
  let d := λ (c : ℝ × ℝ) (a b c' : ℝ), (|a * c.1 + b * c.2 + c'| / (Real.sqrt (a^2 + b^2))) in
  d center 3 4 3 = 1 :=
by
  sorry

end distance_from_center_to_line_is_one_l500_500614


namespace original_gain_percentage_l500_500705

/-- Define the values according to the conditions given in part (a):
    1. Cost price (CP) of the article is ₹800.
    2. If bought at 5% less and sold ₹4 less, the profit is 10%.
    3. Prove that the original gain percentage \(G\) is 5%.
-/
variable (CP : ℝ) (CP' SP SP' : ℝ) (G : ℝ) (H1 : CP = 800)
          (H2 : CP' = 800 * 0.95)
          (H3 : SP' = 760 * 1.10)
          (H4 : SP' = SP - 4)

theorem original_gain_percentage :
  SP = 800 * (1 + G / 100) →
  G = 5 :=
by
  intro hSP
  sorry

end original_gain_percentage_l500_500705


namespace probability_of_multiples_l500_500044

theorem probability_of_multiples (n : ℕ) (h1 : n = 600) : 
  (∀ k, k ∣ n → ¬(30 ∣ k ∧ 42 ∣ k ∧ 56 ∣ k)) → 
  (prob : ℤ) = 0 := 
by
  intro h
  have h_lcm : lcm 30 (lcm 42 56) = 840 := by decide
  have h_factor : ¬(840 ∣ n) := by
    intro h2
    have h3 : 840 > 600 := by decide
    exact not_le_of_gt h3 h2
  intro k h_div
  exact h_factor (dvd_trans (dvd_lcm_right _ _) h_div)

#eval probability_of_multiples 600 (by decide) (by decide)

end probability_of_multiples_l500_500044


namespace largest_x_y_sum_l500_500621

variable (numbers : Finset ℕ) (x y : ℕ)
variable (placements : List (ℕ × ℕ)) -- pairs of square placements

-- The set of given integers
def given_numbers : Finset ℕ := {1, 2, 4, 5, 6, 9, 10, 11, 13}

-- Condition functions checking if sum of neighbors condition is met
def valid_circle (cir : ℕ, sq1 sq2 : ℕ) := cir = sq1 + sq2

-- Problem statement definition
def valid_arrangement (p : List (ℕ × ℕ)) : Prop :=
  -- Ensures every number is used exactly once
  (p.map Prod.fst).toFinset = given_numbers ∧ -- square numbers
  (p.map Prod.snd).toFinset = given_numbers ∧ -- circle numbers
  -- Ensure each circle is sum of pair squares
  ∀ (s1 s2 c: ℕ), (s1, c) ∈ p ∧ (s2, c) ∈ p → valid_circle c s1 s2

theorem largest_x_y_sum : ∃ x y, valid_arrangement placements → x ∈ given_numbers ∧ y ∈ given_numbers ∧ x + y = 20 :=
  sorry

end largest_x_y_sum_l500_500621


namespace tan_alpha_value_l500_500891

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l500_500891


namespace product_of_p_r_s_l500_500152

theorem product_of_p_r_s :
  ∃ (p r s : ℤ), 4^p + 4^3 = 272 ∧ 3^r + 27 = 54 ∧ 2^5 + 7^s = 1686 ∧ p * r * s = 36 :=
by {
  use [4, 3, 3],
  split; {
    exact ⟨by linarith, by linarith, by linarith, by linarith⟩,
  },
  sorry
}

end product_of_p_r_s_l500_500152


namespace quadratic_inequality_solution_non_empty_l500_500527

theorem quadratic_inequality_solution_non_empty
  (a b c : ℝ) (h : a < 0) :
  ∃ x : ℝ, ax^2 + bx + c < 0 :=
sorry

end quadratic_inequality_solution_non_empty_l500_500527


namespace range_a_l500_500825

theorem range_a (a : ℝ) : (∃ x, x < 0 ∧ abs x = a * x + 1) ∧ (∀ x, x > 0 → abs x ≠ a * x + 1) ↔ a ∈ set.Ioo (-1 : ℝ) 1 := sorry

end range_a_l500_500825


namespace eccentricity_value_l500_500623

/-
Given the lengths of the major axis (2a), minor axis (2b), and focal distance (2c) of an ellipse form an arithmetic sequence,
prove that the eccentricity of the ellipse is 3/5.
-/

noncomputable def ellipse_eccentricity (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : b^2 = a^2 - c^2) : ℝ :=
  let e := c / a in
  if h3 : 0 < e ∧ e < 1 then e else 0

theorem eccentricity_value (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : b^2 = a^2 - c^2) (h3 : 5 * (c / a)^2 + 2 * (c / a) - 3 = 0) (h4 : 0 < c / a ∧ c / a < 1) :
  ellipse_eccentricity a b c h1 h2 = 3 / 5 :=
by
  sorry

end eccentricity_value_l500_500623


namespace orange_area_percentage_l500_500721

theorem orange_area_percentage (flag_area : ℝ) (cross_area_percentage : ℝ) (orange_area_percentage : ℝ) : 
  (flag_area = 1) → 
  (cross_area_percentage = 49) → 
  (orange_area_percentage = 2.45) → 
  ((0.49 * flag_area) * (orange_area_percentage / 100) = 0.0245) := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  sorry

end orange_area_percentage_l500_500721


namespace kite_ratio_equality_l500_500186

-- Definitions for points, lines, and conditions in the geometric setup
variables {Point : Type*} [MetricSpace Point]

-- Assuming A, B, C, D, P, E, F, G, H, I, J are points
variable (A B C D P E F G H I J : Point)

-- Conditions based on the problem
variables (AB_eq_AD : dist A B = dist A D)
          (BC_eq_CD : dist B C = dist C D)
          (on_BD : P ∈ line B D)
          (line_PE_inter_AD : E ∈ line P E ∧ E ∈ line A D)
          (line_PF_inter_BC : F ∈ line P F ∧ F ∈ line B C)
          (line_PG_inter_AB : G ∈ line P G ∧ G ∈ line A B)
          (line_PH_inter_CD : H ∈ line P H ∧ H ∈ line C D)
          (GF_inter_BD_at_I : I ∈ line G F ∧ I ∈ line B D)
          (EH_inter_BD_at_J : J ∈ line E H ∧ J ∈ line B D)

-- The statement to prove
theorem kite_ratio_equality :
  dist P I / dist P B = dist P J / dist P D := sorry

end kite_ratio_equality_l500_500186


namespace tank_emptying_time_l500_500381

def tank_capacity : ℕ := 5760
def leak_time_hours : ℕ := 6
def inlet_rate_per_minute : ℕ := 4
def net_emptying_time_hours := 8

theorem tank_emptying_time :
  ∀ (tank_capacity leak_time_hours inlet_rate_per_minute net_emptying_time_hours : ℕ),
    let leak_rate := tank_capacity / leak_time_hours in
    let inlet_rate := inlet_rate_per_minute * 60 in
    let net_rate := leak_rate - inlet_rate in
    net_emptying_time_hours = tank_capacity / net_rate :=
sorry

end tank_emptying_time_l500_500381


namespace binom_20_4_l500_500415

theorem binom_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end binom_20_4_l500_500415


namespace mandy_more_cinnamon_l500_500230

def cinnamon : ℝ := 0.67
def nutmeg : ℝ := 0.5

theorem mandy_more_cinnamon : cinnamon - nutmeg = 0.17 :=
by
  sorry

end mandy_more_cinnamon_l500_500230


namespace parabola_hyperbola_tangent_l500_500430

theorem parabola_hyperbola_tangent : ∃ m : ℝ, 
  (∀ x y : ℝ, y = x^2 - 2 * x + 2 → y^2 - m * x^2 = 1) ↔ m = 1 :=
by
  sorry

end parabola_hyperbola_tangent_l500_500430


namespace inequality_solution_count_l500_500284

theorem inequality_solution_count :
  {x : ℕ | 3 * x - 5 < 3 + x}.finite.to_finset.card = 4 :=
by
  sorry

end inequality_solution_count_l500_500284


namespace quadratic_no_real_roots_l500_500481

-- Given conditions
variables {p q a b c : ℝ}
variables (hp_pos : 0 < p) (hq_pos : 0 < q) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
variables (hp_neq_q : p ≠ q)

-- p, a, q form a geometric sequence
variables (h_geo : a^2 = p * q)

-- p, b, c, q form an arithmetic sequence
variables (h_arith1 : 2 * b = p + c)
variables (h_arith2 : 2 * c = b + q)

-- Proof statement
theorem quadratic_no_real_roots (hp_pos hq_pos ha_pos hb_pos hc_pos hp_neq_q h_geo h_arith1 h_arith2 : ℝ) :
    (b * (x : ℝ)^2 - 2 * a * x + c = 0) → false :=
sorry

end quadratic_no_real_roots_l500_500481


namespace collinear_probability_l500_500190

theorem collinear_probability : 
  let total_ways := Nat.choose 25 4,
      collinear_ways := 64
  in 
  (collinear_ways: ℚ) / (total_ways: ℚ) = 64 / 12650 := 
by 
  let total_ways : ℕ := Nat.choose 25 4
  let collinear_ways : ℕ := 64
  have h_total: total_ways = 12650 := rfl
  have h_collinear: collinear_ways = 64 := rfl
  calc 
    (collinear_ways: ℚ) / (total_ways: ℚ)
      = (64: ℚ) / (12650: ℚ) : by rw [h_total, h_collinear]

-- Mark the skipped proof with sorry
sorry

end collinear_probability_l500_500190


namespace chord_length_of_ellipse_l500_500704

theorem chord_length_of_ellipse : (line_passes_focus : ∃ (line : ℝ → ℝ), 
    ∃ (F : ℝ × ℝ), F = (2, 0) ∧ line = λ x, sqrt 3 * (x - 2)) →
  (ellipse : set (ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ ellipse ↔ (p.1^2 / 6 + p.2^2 / 2 = 1)) →
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧ A ∈ ellipse ∧ B ∈ ellipse ∧ 
    ∃ (line_eq : ∀ (x : ℝ), sqrt 3 * (x - 2)), 
      A.2 = line_eq A.1 ∧ B.2 = line_eq B.1) →
  sqrt (4 * ((18 / 5)^2 - 4 * 3)) = 4 * sqrt 6 / 5 :=
by
  sorry

end chord_length_of_ellipse_l500_500704


namespace problem_l500_500051

variable {R : Type} [LinearOrder R] [NormedLinearOrder R] [Field R] [NormedSpace R R]

variable (f : R → R)

-- f is odd function
axiom h_odd : ∀ x : R, f (-x) = -f x

-- f(x+1) = -f(x-1)
axiom h_rec : ∀ x : R, f (x + 1) = -f (x - 1)

theorem problem (h_odd : ∀ x : R, f (-x) = -f x) (h_rec : ∀ x : R, f (x + 1) = -f (x - 1)) : 
  f 0 + f 1 + f 2 + f 3 + f 4 = 0 := 
sorry

end problem_l500_500051


namespace find_a_range_l500_500832

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2*x^2 - a*x + 1

theorem find_a_range :
  (∃ (a : ℝ), (f'(-1)*f'(1) < 0))
    → (a ∈ range_of_a) :=
sorry

end find_a_range_l500_500832


namespace pyramid_volume_l500_500714

noncomputable def volume_of_pyramid (base_length : ℝ) (base_width : ℝ) (edge_length : ℝ) : ℝ :=
  let base_area := base_length * base_width
  let diagonal_length := Real.sqrt (base_length^2 + base_width^2)
  let height := Real.sqrt (edge_length^2 - (diagonal_length / 2)^2)
  (1 / 3) * base_area * height

theorem pyramid_volume : volume_of_pyramid 7 9 15 = 21 * Real.sqrt 192.5 := by
  sorry

end pyramid_volume_l500_500714


namespace value_of_pi_quotient_l500_500525

noncomputable def seq : ℕ → ℚ
| 0 => 3 / 5
| 1 => 1 / 4
| n => if h : 1 < n then seq (n - 1) * seq (n - 2) else 1 -- placeholder for valid recursion

def pi (n : ℕ) : ℚ := (Finset.range n).prod seq

theorem value_of_pi_quotient :
  let a2022 := seq 2021
  let π2023 := pi 2023
  π2023 / a2022 = 1 / 4 :=
sorry

end value_of_pi_quotient_l500_500525


namespace factorial_mod_13_l500_500663

theorem factorial_mod_13: (10! % 13) = 6 := by
  sorry

end factorial_mod_13_l500_500663


namespace total_curved_surface_area_l500_500387

-- Conditions: the dimensions of the cylinders
def height1 : ℝ := 5
def radius1 : ℝ := 2
def height2 : ℝ := 3
def radius2 : ℝ := 4

-- Formula for the lateral surface area of a cylinder
def lateral_area (r h : ℝ) : ℝ := 2 * Real.pi * r * h

-- Areas of both cylinders
def area1 : ℝ := lateral_area radius1 height1
def area2 : ℝ := lateral_area radius2 height2

-- Total area of the curved surfaces
def total_area : ℝ := area1 + area2

-- Theorem: the total area is 44π
theorem total_curved_surface_area : total_area = 44 * Real.pi := by
  sorry

end total_curved_surface_area_l500_500387


namespace half_of_flour_l500_500384

theorem half_of_flour : 
  let original_flour := 4 + 1 / 2 in
  let half_flour := original_flour / 2 in
  half_flour = 2 + 1 / 4 :=
by
  sorry

end half_of_flour_l500_500384


namespace surface_area_of_sphere_l500_500479

theorem surface_area_of_sphere 
  {P A B C : Type}
  (h_sphere : ∃ R : ℝ, 
    (P.dist(A) = 3) ∧ 
    (P.dist(B) = 4) ∧ 
    (P.dist(C) = 5) ∧ 
    (pairwise_perpendicular P A B C): 
  4 * real.pi * (R ^ 2) = 50 * real.pi :=
by sorry

end surface_area_of_sphere_l500_500479


namespace find_f_prime_neg_one_third_l500_500828

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * (f' (-1 / 3)) * x

theorem find_f_prime_neg_one_third : (f' (-1 / 3)) = 2 / 3 :=
by
  sorry

end find_f_prime_neg_one_third_l500_500828


namespace find_a_l500_500283

def f (x : ℝ) : ℝ :=
if -1 < x ∧ x < 0 then
  Real.sin (π * x ^ 2)
else
  if x ≥ 0 then
    Real.exp (x - 1)
  else
    0 -- this should never be invoked based on the input domain but helps with Lean's totality checking

theorem find_a (a : ℝ) (h: f 1 + f a = 2) :
  a = 1 ∨ a = -Real.sqrt 2 / 2 :=
sorry

end find_a_l500_500283


namespace problem_l500_500464

theorem problem (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < 1) 
  (h3 : a^(2 * real.log a / real.log 3) = 81 * real.sqrt 3) : 
  1 / a^2 + real.log a / real.log 9 = 105 / 4 :=
by
  sorry

end problem_l500_500464


namespace number_of_functions_f_is_1_l500_500211

noncomputable def S : Set ℝ := {x : ℝ | x ≠ 0}

def f (x : ℝ) : ℝ := sorry

axiom f_at_2 : f 2 = 1

axiom property_ii : ∀ x y ∈ S, x + y ∈ S → f (1 / (x + y)) = f (2 / x) + f (2 / y)

axiom property_iii : ∀ x y ∈ S, x + y ∈ S → (x + y) * f (x + y) = 4 * x * y * f (x) * f (y)

theorem number_of_functions_f_is_1 : ∃! f : ℝ → ℝ, 
  (∀ x, x ∈ S → f x = (1 / x)) ∧ 
  (f 2 = 1) ∧ 
  (∀ x y ∈ S, x + y ∈ S → f (1 / (x + y)) = f (2 / x) + f (2 / y)) ∧ 
  (∀ x y ∈ S, x + y ∈ S → (x + y) * f (x + y) = 4 * x * y * f (x) * f (y)) :=
sorry

end number_of_functions_f_is_1_l500_500211


namespace probability_of_selecting_one_defective_l500_500461

/-- A statement about the probability of selecting exactly 1 defective item from a batch -/
theorem probability_of_selecting_one_defective (good_items : ℕ) (defective_items : ℕ) (selected_items : ℕ) :
  (good_items = 6) → (defective_items = 2) → (selected_items = 3) →
  let X := (C(good_items, 2) * C(defective_items, 1)) / C(good_items + defective_items, 3) in
  X = (15 / 28) := by
  intros h1 h2 h3
  sorry

noncomputable def C (n k : ℕ) : ℚ := (nat.choose n k : ℚ)

end probability_of_selecting_one_defective_l500_500461


namespace smallest_solution_of_equation_l500_500428

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (x^4 - 14*x^2 + 49 = 0) ∧ (∀ y : ℝ, (y^4 - 14*y^2 + 49 = 0) → x ≤ y) ∧ x = -real.sqrt 7 :=
by
  sorry

end smallest_solution_of_equation_l500_500428


namespace fraction_addition_l500_500407

variable (a : ℝ)

theorem fraction_addition (ha : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by
  sorry

end fraction_addition_l500_500407


namespace decreasing_on_zero_to_one_div_e_l500_500831

noncomputable def f (x : ℝ) := x * Real.log x

theorem decreasing_on_zero_to_one_div_e :
  ∀ x, 0 < x ∧ x < 1 / Real.exp 1 → (f' x < 0)
by
  intro x
  sorry

end decreasing_on_zero_to_one_div_e_l500_500831


namespace work_done_by_force_l500_500377

def force : ℝ × ℝ := (-1, -2)
def displacement : ℝ × ℝ := (3, 4)

def work_done (F S : ℝ × ℝ) : ℝ :=
  F.1 * S.1 + F.2 * S.2

theorem work_done_by_force :
  work_done force displacement = -11 := 
by
  sorry

end work_done_by_force_l500_500377


namespace relativelyPrimeDaysInFebruary_l500_500762

-- Define the condition that February in a leap year has 29 days.
def februaryLeapYearDays : Nat := 29

-- Define the condition that a day is relatively prime to 2 if it is not divisible by 2.
def isRelativelyPrimeTo2 (d : Nat) : Bool :=
  d % 2 ≠ 0

-- Define the statement to count the relatively prime days in February.
def countRelativelyPrimeDays (daysInMonth : Nat) : Nat :=
  (List.range daysInMonth).countp (λ d, isRelativelyPrimeTo2 (d+1))

-- The theorem statement asserting the count of relatively prime days.
theorem relativelyPrimeDaysInFebruary : countRelativelyPrimeDays februaryLeapYearDays = 15 := by
  -- proof goes here
  sorry

end relativelyPrimeDaysInFebruary_l500_500762


namespace minimal_polynomial_correct_l500_500083

noncomputable def minimal_polynomial : Polynomial ℚ :=
  (X^2 - C 4 * X - C 1) * (X^2 - C 6 * X + C 2)

theorem minimal_polynomial_correct :
  (minimal_polynomial = Polynomial.X^4 - Polynomial.C 10 * Polynomial.X^3 + Polynomial.C 28 * Polynomial.X^2 - Polynomial.C 26 * Polynomial.X - Polynomial.C 2) ∧
  (∀ (α : ℚ), is_root minimal_polynomial α ↔ α = 2 + real.sqrt 5 ∨ α = 2 - real.sqrt 5 ∨ α = 3 + real.sqrt 7 ∨ α = 3 - real.sqrt 7) :=
by
  sorry

end minimal_polynomial_correct_l500_500083


namespace inequality_solution_set_l500_500299

theorem inequality_solution_set (x : ℝ) :
  x^2 * (x^2 + 2*x + 1) > 2*x * (x^2 + 2*x + 1) ↔
  ((x < -1) ∨ (-1 < x ∧ x < 0) ∨ (2 < x)) :=
sorry

end inequality_solution_set_l500_500299


namespace isosceles_triangle_inradius_l500_500583

theorem isosceles_triangle_inradius (ABC : Triangle) (b : Real) (r : Real)
  (h_isosceles : ABC.IsIsosceles AC BC)
  (h_ac : AC = b)
  (h_bc : BC = b)
  (h_inradius : ABC.inradius = r) :
  b > Real.pi * r :=
sorry

end isosceles_triangle_inradius_l500_500583


namespace problem1_problem2_problem3_l500_500405

-- Problem 1
theorem problem1 : (π - 3)^0 + (1 / 2)^(-3) - 3^2 + (-1)^(2024) = 1 :=
by
  sorry

-- Problem 2
theorem problem2 (m n : ℝ) : 3 * m * 2 * n^2 - (2 * n)^2 * (1 / 2) * m = 4 * m * n^2 :=
by
  sorry

-- Problem 3
theorem problem3 (x y : ℝ) (hx : x = -3) (hy : y = 2) :
  ((x + 2 * y)^2 - y * (x + 3 * y) + (x - y) * (x + y)) / (2 * x) = 0 :=
by
  sorry

end problem1_problem2_problem3_l500_500405


namespace train_crossing_time_is_correct_l500_500144

noncomputable def train_cross_bridge_time : ℝ :=
  let length_of_train := 110 -- meters
  let length_of_bridge := 170 -- meters
  let speed_kmph := 60 -- kilometers per hour
  let speed_mps := speed_kmph * (1000 / 3600) -- Convert kmph to m/s
  let total_distance := length_of_train + length_of_bridge -- Total distance to be covered in meters
  total_distance / speed_mps -- Time = Distance / Speed

theorem train_crossing_time_is_correct : train_cross_bridge_time ≈ 16.79 :=
by
  -- Implicitly stating the proof to be completed
  sorry

end train_crossing_time_is_correct_l500_500144


namespace independence_of_A_and_D_l500_500323

noncomputable def balls : Finset ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def draw_one : ℕ := (1 : ℕ)
noncomputable def draw_two : ℕ := (2 : ℕ)

def event_A : ℕ → Prop := λ n, n = 1
def event_B : ℕ → Prop := λ n, n = 2
def event_C : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 8
def event_D : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 7

def prob (event : ℕ → Prop) : ℚ := 1 / 6
def joint_prob (event1 event2 : ℕ → Prop) : ℚ := (1 / 36)

theorem independence_of_A_and_D :
  joint_prob (λ n, event_A n) (λ n, event_D (draw_one, draw_two)) = prob event_A * prob (λ n, event_D (draw_one, draw_two)) :=
by
  sorry

end independence_of_A_and_D_l500_500323


namespace carlos_bus_route_distance_l500_500747

-- Conditions
variables {d : ℝ} (h1 : 120 * d / 14400 = d) -- Normal bus speed condition
variables (h2 : (1/4) * d / (d / 14400) + (1/2) * d / ((d / 14400) - 15 / 60) + (1/4) * d / ((d / 14400) + 10 / 60) = 150) -- Special day conditions

-- Theorem
theorem carlos_bus_route_distance : 
  ∃ d, 
  (120 * d / 14400 = d ∧ 
  (1/4) * d / (d / 14400) + (1/2) * d / ((d / 14400) - 15 / 60) + (1/4) * d / ((d / 14400) + 10 / 60) = 150 ∧ 
  d = 124) :=
begin
  sorry
end

end carlos_bus_route_distance_l500_500747


namespace unpainted_unit_cubes_in_4x4x4_cube_l500_500684

/-- A 4x4x4 cube is formed by assembling 64 unit cubes.
    Three large squares are painted on each of the six faces of the cube,
    where the squares are arranged to cover the four corners on each face.
    Prove that 56 unit cubes have no paint on them. -/
theorem unpainted_unit_cubes_in_4x4x4_cube : 
  let n := 4 in 
  let total_cubes := n * n * n in
  let corner_cubes := 8 in -- 8 corners in a 4x4x4 cube
  total_cubes - corner_cubes = 56 := 
by 
  let n := 4
  let total_cubes := n * n * n
  let corner_cubes := 8
  show total_cubes - corner_cubes = 56
  sorry

end unpainted_unit_cubes_in_4x4x4_cube_l500_500684


namespace probability_exactly_three_even_numbers_l500_500023

/-- The probability that exactly three of four 8-sided dice show an even number is 1/4. -/
theorem probability_exactly_three_even_numbers :
  let even_number_prob := 1 / 2
  let comb := 4
  let indiv_prob := (1 / 2) ^ 4
  ∃ prob : ℚ, prob = (comb * indiv_prob) ∧ prob = 1 / 4 :=
by
  -- Definitions from the conditions
  let even_number_prob := 1 / 2
  let comb := 4
  let indiv_prob := (1 / 2) ^ 4

  -- The mathematically equivalent proof problem:
  let prob := comb * indiv_prob
  
  use prob
  split
  -- Show that prob = comb * indiv_prob
  rw mul_comm,
  simp [comb, indiv_prob, prob],
  
  -- Show that prob = 1 / 4
  have h : comb * indiv_prob = (4:ℚ) * (1 / 2) ^ 4 := rfl
  rw h,
  norm_num,
  exact fact_num 1 4,
  sorry

end probability_exactly_three_even_numbers_l500_500023


namespace largest_three_digit_multiple_of_8_and_sum_24_is_888_l500_500660

noncomputable def largest_three_digit_multiple_of_8_with_digit_sum_24 : ℕ :=
  888

theorem largest_three_digit_multiple_of_8_and_sum_24_is_888 :
  ∃ n : ℕ, (300 ≤ n ∧ n ≤ 999) ∧ (n % 8 = 0) ∧ ((n.digits 10).sum = 24) ∧ n = largest_three_digit_multiple_of_8_with_digit_sum_24 :=
by
  existsi 888
  sorry

end largest_three_digit_multiple_of_8_and_sum_24_is_888_l500_500660


namespace term_largest_binomial_coeff_constant_term_in_expansion_l500_500116

theorem term_largest_binomial_coeff {n : ℕ} (h : n = 8) :
  ∃ (k : ℕ) (coeff : ℤ), coeff * x ^ k = 1120 * x^4 :=
by
  sorry

theorem constant_term_in_expansion :
  ∃ (const : ℤ), const = 1280 :=
by
  sorry

end term_largest_binomial_coeff_constant_term_in_expansion_l500_500116


namespace p_is_sufficient_but_not_necessary_for_q_l500_500103

def p (x : ℝ) : Prop := log10 (x - 3) < 0
def q (x : ℝ) : Prop := (x - 2) / (x - 4) < 0

theorem p_is_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) := 
sorry

end p_is_sufficient_but_not_necessary_for_q_l500_500103


namespace solve_puzzle_l500_500952

-- Definitions for digits.
def is_digit (n : ℕ) : Prop := n < 10

theorem solve_puzzle : ∃ (a b : ℕ), is_digit a ∧ is_digit b ∧ ((3 * 10 + a) * (b * 10 + 4) = 116) ∧ a + b = 7 :=
by {
  -- Variables a and b as digits
  use [4, 3],
  
  -- Proving the conditions
  split,
  { exact dec_trivial }, -- is_digit 4
  split,
  { exact dec_trivial }, -- is_digit 3
  split,
  { norm_num }, -- (3 * 10 + 4) * (3 * 10 + 4) = 116
  { norm_num } -- 4 + 3 = 7
}

end solve_puzzle_l500_500952


namespace manager_wage_l500_500401

variable (M D C : ℝ)

theorem manager_wage :
  (D = M / 2) →
  (C = 1.22 * D) →
  (C = M - 3.315) →
  M = 8.5 :=
by
  intros h1 h2 h3
  -- Substitute D from h1 into h2, and then solve for M using h3
  sorry

end manager_wage_l500_500401


namespace coefficient_x4_in_binomial_expansion_l500_500447

open Nat

theorem coefficient_x4_in_binomial_expansion :
  (∃ (k : ℕ), (k = choose 6 4 * 2^4) ∧ k = 240) :=
begin
  use (choose 6 4 * 2^4),
  split,
  { refl, },
  { norm_num, },
end

end coefficient_x4_in_binomial_expansion_l500_500447


namespace ColorInfiniteSeq_l500_500600

/-- Given that every positive integer is colored either red or blue arbitrarily,
  prove that there exists an infinite sequence of positive integers a_1, a_2, a_3, ... 
  such that the sequence a_1, (a_1 + a_2) / 2, a_2, (a_2 + a_3) / 2, a_3, (a_3 + a_4) / 2, ...
  all share the same color. -/
theorem ColorInfiniteSeq :
  (∀ n : ℕ, n > 0 → (is_red n ∨ is_blue n)) →
  ∃ (a : ℕ → ℕ), (strict_anti a) ∧
  ((∀ n, is_red (a n)) ∨ (∀ n, is_blue (a n))) ∧
  (∀ n, is_red ((a n + a (n + 1)) / 2) ∨ is_blue ((a n + a (n + 1)) / 2)) :=
by
  intro h
  sorry

end ColorInfiniteSeq_l500_500600


namespace megan_markers_l500_500233

def initial_markers : ℕ := 217
def roberts_gift : ℕ := 109
def sarah_took : ℕ := 35

def final_markers : ℕ := initial_markers + roberts_gift - sarah_took

theorem megan_markers : final_markers = 291 := by
  sorry

end megan_markers_l500_500233


namespace angle_DXY_is_45_l500_500367

variables 
  (A B C D X Y : Type)
  [in_square : InSquare A B C D]
  [on_edge_X : OnEdge X A B]
  [on_edge_Y : OnEdge Y B C]
  (angle_ADX angle_CDY : ℝ)
  (hx : angle_ADX = 15)
  (hy : angle_CDY = 30)

theorem angle_DXY_is_45
  (angle_DXY : ℝ) (h : angle_DXY = 45) : angle_ADX = 15 → angle_CDY = 30 → angle_DXY = 45 :=
by
  sorry

end angle_DXY_is_45_l500_500367


namespace power_six_tens_digit_l500_500339

def tens_digit (x : ℕ) : ℕ := (x / 10) % 10

theorem power_six_tens_digit (n : ℕ) (hn : tens_digit (6^n) = 1) : n = 3 :=
sorry

end power_six_tens_digit_l500_500339


namespace A_and_D_independent_l500_500321

-- Definitions of the events based on given conditions
def event_A (x₁ : ℕ) : Prop := x₁ = 1
def event_B (x₂ : ℕ) : Prop := x₂ = 2
def event_C (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 8
def event_D (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 7

-- Probabilities based on uniform distribution and replacement
def probability_event (event : ℕ → ℕ → Prop) : ℚ :=
  if h : ∃ x₁ : ℕ, ∃ x₂ : ℕ, x₁ ∈ finset.range 1 7 ∧ x₂ ∈ finset.range 1 7 ∧ event x₁ x₂
  then ((finset.card (finset.filter (λ x, event x.1 x.2)
                (finset.product (finset.range 1 7) (finset.range 1 7)))) : ℚ) / 36
  else 0

noncomputable def P_A : ℚ := 1 / 6
noncomputable def P_D : ℚ := 1 / 6
noncomputable def P_A_and_D : ℚ := 1 / 36

-- Independence condition (by definition): P(A ∩ D) = P(A) * P(D)
theorem A_and_D_independent :
  P_A_and_D = P_A * P_D := by
  sorry

end A_and_D_independent_l500_500321


namespace range_of_a_l500_500162

theorem range_of_a 
  (a : ℝ) 
  (h : ∀ x : ℝ, |x - 2 * a| + |2 * x - a| ≥ a ^ 2) :
  a ∈ set.Icc (-3 / 2) (3 / 2) := sorry

end range_of_a_l500_500162


namespace money_left_correct_l500_500228

-- Define the costs of each type of the app.
def productivity_app_cost := 4
def gaming_app_cost := 5
def lifestyle_app_cost := 3

-- Define the quantities of each type of app needed.
def productivity_apps := 5
def gaming_apps := 7
def lifestyle_apps := 3

-- Define the total budget Lidia has.
def total_budget := 66

-- Define the discount and tax rates.
def discount_rate := 0.15
def tax_rate := 0.10

-- Calculate the total cost before discounts and taxes
def total_cost_before_discount := 
  productivity_apps * productivity_app_cost + 
  gaming_apps * gaming_app_cost + 
  lifestyle_apps * lifestyle_app_cost

-- Calculate the cost after the discount
def discount_amount := total_cost_before_discount * discount_rate
def cost_after_discount := total_cost_before_discount - discount_amount

-- Calculate the total cost after applying the sales tax
def sales_tax := cost_after_discount * tax_rate
def total_cost_after_tax := cost_after_discount + sales_tax

-- Calculate the remaining money Lidia has after purchasing the apps.
def money_left := total_budget - total_cost_after_tax

theorem money_left_correct : 
  money_left = 6.16 := by 
  sorry

end money_left_correct_l500_500228


namespace part1_part2_l500_500838

-- Define the function y in Lean
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part (1)
theorem part1 (x : ℝ) : y (1/2) x < 0 ↔ -1 < x ∧ x < 2 :=
  sorry

-- Part (2)
theorem part2 (x m : ℝ) : y m x < (1 - m) * x - 1 ↔ 
  (m = 0 → x > 0) ∧ 
  (m > 0 → 0 < x ∧ x < 1 / m) ∧ 
  (m < 0 → x < 1 / m ∨ x > 0) :=
  sorry

end part1_part2_l500_500838


namespace tan_alpha_sqrt_15_over_15_l500_500916

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l500_500916


namespace least_n_for_phi_d_eq_64000_l500_500651

/-- Definition of two integers being relatively prime, i.e., gcd(a, b) = 1 --/
def relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- Euler's totient function definition --/
def φ (n : ℕ) : ℕ :=
  (Finset.range n).filter (λ m => relatively_prime m n) |>.card

/-- φ_d(n) definition --/
def φ_d (d n : ℕ) : ℕ :=
  d * φ n

/-- Main theorem statement --/
theorem least_n_for_phi_d_eq_64000 :
  ∃ (n : ℕ), φ n = 40 ∧ ∀ (m : ℕ), (φ m = 40 → n ≤ m) := by sorry

end least_n_for_phi_d_eq_64000_l500_500651


namespace emma_widgets_difference_l500_500236

theorem emma_widgets_difference (t : ℕ) : 
  let w := 3 * t in
  (w * t) - ((w + 5) * (t - 3)) = 4 * t + 15 :=
by
  sorry

end emma_widgets_difference_l500_500236


namespace tan_alpha_proof_l500_500903

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l500_500903


namespace find_B_l500_500298

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def seven_digit_number (B : ℕ) : ℕ :=
  9031510 + B

theorem find_B : ∃ B : ℕ, B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ is_prime (seven_digit_number B) ∧ B = 7 :=
by
  sorry

end find_B_l500_500298


namespace pascal_two_arithmetic_structure_l500_500584

-- Definition of 2-arithmetic Pascal's Triangle's row
def two_pascal_triangle (n : ℕ) : ℕ → ℕ
| 0     := 1
| k + 1 := (two_pascal_triangle k) + (two_pascal_triangle k - 1)

-- Predicate to check if a row in the 2-arithmetic Pascal's Triangle is all zeros except the extremes
def all_zeros_except_extremes (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 ∧ k < n → two_pascal_triangle n k = 0

-- The main theorem stating the equivalence
theorem pascal_two_arithmetic_structure (n : ℕ) :
  (∃ k : ℕ, n = 2^k) ↔ all_zeros_except_extremes n :=
sorry

end pascal_two_arithmetic_structure_l500_500584


namespace latest_ant_off_checkerboard_l500_500998

theorem latest_ant_off_checkerboard (m : ℕ) (h : 0 < m)
  (ants : Finset (ℤ × ℤ))  -- A set of initial positions of ants
  (initial_positions_valid : ∀ p ∈ ants, 1 ≤ p.1 ∧ p.1 ≤ m ∧ 1 ≤ p.2 ∧ p.2 ≤ m)
  (movements_valid : ∀ (t : ℕ) (p : ℤ × ℤ), 
    p ∈ (ants.map (λ (p : ℤ × ℤ), (p.1 + t, p.2 + t)) ∪ 
    ants.map (λ (p : ℤ × ℤ), (p.1 - t, p.2 + t)) ∪ 
    ants.map (λ (p : ℤ × ℤ), (p.1 + t, p.2 - t)) ∪ 
    ants.map (λ (p : ℤ × ℤ), (p.1 - t, p.2 - t)) ) →
      1 ≤ p.1 ∧ p.1 ≤ m ∧ 1 ≤ p.2 ∧ p.2 ≤ m) :
  ∃ t_max, ∀ t, t ≤ t_max :=
begin
  sorry
end

end latest_ant_off_checkerboard_l500_500998


namespace find_measure_A_and_b_c_sum_l500_500171

open Real

noncomputable def triangle_abc (a b c A B C : ℝ) : Prop :=
  ∀ (A B C : ℝ),
  A + B + C = π ∧
  a = sin A ∧
  b = sin B ∧
  c = sin C ∧
  cos (A - C) - cos (A + C) = sqrt 3 * sin C

theorem find_measure_A_and_b_c_sum (a b c A B C : ℝ)
  (h_triangle : triangle_abc a b c A B C) 
  (h_area : (1/2) * b * c * (sin A) = (3 * sqrt 3) / 16) 
  (h_b_def : b = sin B) :
  A = π / 3 ∧ b + c = sqrt 3 := by
  sorry

end find_measure_A_and_b_c_sum_l500_500171


namespace range_of_a_neg_p_true_l500_500105

theorem range_of_a_neg_p_true :
  (∀ x : ℝ, x ∈ Set.Ioo (-2:ℝ) 0 → x^2 + (2*a - 1)*x + a ≠ 0) →
  ∀ a : ℝ, a ∈ Set.Icc 0 ((2 + Real.sqrt 3) / 2) :=
sorry

end range_of_a_neg_p_true_l500_500105


namespace solution_set_of_inequality_l500_500504

def f : Int → Int
| -1 => -1
| 0 => -1
| 1 => 1
| _ => 0 -- Assuming this default as table provided values only for -1, 0, 1

def g : Int → Int
| -1 => 1
| 0 => 1
| 1 => -1
| _ => 0 -- Assuming this default as table provided values only for -1, 0, 1

theorem solution_set_of_inequality :
  {x | f (g x) > 0} = { -1, 0 } :=
by
  sorry

end solution_set_of_inequality_l500_500504


namespace tan_alpha_proof_l500_500897

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l500_500897


namespace number_of_friends_l500_500414

-- Definitions based on the given problem conditions
def total_candy := 420
def candy_per_friend := 12

-- Proof statement in Lean 4
theorem number_of_friends : total_candy / candy_per_friend = 35 := by
  sorry

end number_of_friends_l500_500414


namespace time_after_9876_seconds_l500_500196

-- Define the initial time in seconds
def initial_seconds : ℕ := 6 * 3600

-- Define the elapsed time in seconds
def elapsed_seconds : ℕ := 9876

-- Convert given time in seconds to hours, minutes, and seconds
def time_in_hms (total_seconds : ℕ) : (ℕ × ℕ × ℕ) :=
  let hours := total_seconds / 3600
  let minutes := (total_seconds % 3600) / 60
  let seconds := total_seconds % 60
  (hours, minutes, seconds)

-- Define the final time in 24-hour format (08:44:36)
def final_time : (ℕ × ℕ × ℕ) := (8, 44, 36)

-- The question's proof statement
theorem time_after_9876_seconds : 
  time_in_hms (initial_seconds + elapsed_seconds) = final_time :=
sorry

end time_after_9876_seconds_l500_500196


namespace find_p_l500_500770

theorem find_p (a : ℕ) (ha : a = 2030) : 
  let p := 2 * a + 1;
  let q := a * (a + 1);
  p = 4061 ∧ Nat.gcd p q = 1 := by
  sorry

end find_p_l500_500770


namespace find_k_l500_500287

theorem find_k 
  (x y: ℝ) 
  (h1: y = 5 * x + 3) 
  (h2: y = -2 * x - 25) 
  (h3: y = 3 * x + k) : 
  k = -5 :=
sorry

end find_k_l500_500287


namespace set_of_possible_values_l500_500563

-- Define the variables and the conditions as a Lean definition
noncomputable def problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) : Set ℝ :=
  {x : ℝ | x = (1 / a + 1 / b + 1 / c)}

-- Define the theorem to state that the set of all possible values is [9, ∞)
theorem set_of_possible_values (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  problem a b c ha hb hc sum_eq_one = {x : ℝ | 9 ≤ x} :=
sorry

end set_of_possible_values_l500_500563


namespace composite_number_property_l500_500774

theorem composite_number_property (n : ℕ) 
  (h1 : n > 1) 
  (h2 : ¬ Prime n) 
  (h3 : ∀ (d : ℕ), d ∣ n → 1 ≤ d → d < n → n - 20 ≤ d ∧ d ≤ n - 12) :
  n = 21 ∨ n = 25 :=
by
  sorry

end composite_number_property_l500_500774


namespace fangfang_travel_time_l500_500773

theorem fangfang_travel_time (time_1_to_5 : ℕ) (start_floor end_floor : ℕ) (floors_1_to_5 : ℕ) (floors_2_to_7 : ℕ) :
  time_1_to_5 = 40 →
  floors_1_to_5 = 5 - 1 →
  floors_2_to_7 = 7 - 2 →
  end_floor = 7 →
  start_floor = 2 →
  (end_floor - start_floor) * (time_1_to_5 / floors_1_to_5) = 50 :=
by 
  sorry

end fangfang_travel_time_l500_500773


namespace minimum_buses_needed_l500_500688

theorem minimum_buses_needed (bus_capacity : ℕ) (students : ℕ) (h : bus_capacity = 38 ∧ students = 411) :
  ∃ n : ℕ, 38 * n ≥ students ∧ ∀ m : ℕ, 38 * m ≥ students → n ≤ m :=
by sorry

end minimum_buses_needed_l500_500688


namespace triangle_ABC_properties_l500_500172

variable {α : Type*}

noncomputable theory

def triangle (a b c : ℝ) := a + b > c ∧ b + c > a ∧ c + a > b

def perimeter (a b c : ℝ) : ℝ := a + b + c

def circumradius (a b c : ℝ) : ℝ := 
if h : triangle a b c then sorry else 0 -- Circumradius definition placeholder

def inradius (a b c : ℝ) : ℝ := 
if h : triangle a b c then sorry else 0 -- Inradius definition placeholder

theorem triangle_ABC_properties (a b c : ℝ) (h : triangle a b c):
  ¬ (perimeter a b c > circumradius a b c + inradius a b c) ∧ 
  ¬ (perimeter a b c ≤ circumradius a b c + inradius a b c) ∧
  ¬ ((perimeter a b c) / 6 < circumradius a b c + inradius a b c ∧ circumradius a b c + inradius a b c < 6 * (perimeter a b c)) := 
sorry

end triangle_ABC_properties_l500_500172


namespace pentagon_area_eq_eleven_l500_500238

noncomputable def point (x y : ℝ) : (ℝ × ℝ) := (x, y)

def A : (ℝ × ℝ) := point 1 3
def B : (ℝ × ℝ) := point 1 9
def C : (ℝ × ℝ) := point 6 8
def E : (ℝ × ℝ) := point 5 1

def line (p1 p2 : ℝ × ℝ) : (ℝ → ℝ) := λ x, ((p2.2 - p1.2) / (p2.1 - p1.1)) * (x - p1.1) + p1.2

def AC := line A C
def BE := line B E

def D : (ℝ × ℝ) := (3, 5) -- Intersection point of AC and BE, calculated from the system of equations

def area_pentagon (p1 p2 p3 p4 p5 : (ℝ × ℝ)) :=
  let triangle_area (q1 q2 q3 : (ℝ × ℝ)) : ℝ :=
    0.5 * abs (q1.1 * (q2.2 - q3.2) + q2.1 * (q3.2 - q1.2) + q3.1 * (q1.2 - q2.2))
  in triangle_area p1 p2 p5 + triangle_area p2 p3 p5 + triangle_area p3 p4 p5 + triangle_area p4 p1 p5

theorem pentagon_area_eq_eleven :
  area_pentagon A B C E D = 11 := 
sorry

end pentagon_area_eq_eleven_l500_500238


namespace polygon_sides_sum_2340_l500_500163

theorem polygon_sides_sum_2340 (n : ℕ) (angles : Fin n → ℝ) (h_convex : ∀ i, 0 < angles i ∧ angles i < 180) :
  (∑ i in Finset.univ, angles i) = 180 * (n - 2) → (∑ i in Finset.eraseₓ Finset.univ 2, angles i = 2340) → 
  n = 16 :=
sorry

end polygon_sides_sum_2340_l500_500163


namespace tan_alpha_solution_l500_500867

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l500_500867


namespace largest_three_digit_multiple_of_8_and_sum_24_is_888_l500_500662

noncomputable def largest_three_digit_multiple_of_8_with_digit_sum_24 : ℕ :=
  888

theorem largest_three_digit_multiple_of_8_and_sum_24_is_888 :
  ∃ n : ℕ, (300 ≤ n ∧ n ≤ 999) ∧ (n % 8 = 0) ∧ ((n.digits 10).sum = 24) ∧ n = largest_three_digit_multiple_of_8_with_digit_sum_24 :=
by
  existsi 888
  sorry

end largest_three_digit_multiple_of_8_and_sum_24_is_888_l500_500662


namespace tan_alpha_value_l500_500936

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l500_500936


namespace main_theorem_l500_500110

def point (x y : ℝ) : Prop := true

def A : Prop := point (-3) 0
def B : Prop := point 3 0

def distance (p₁ p₂ : ℝ × ℝ) : ℝ := real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

def P (x y : ℝ) : Prop := distance (x, y) (-3, 0) = 2 * distance (x, y) (3, 0)

def curve_C (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

def l1 (x y : ℝ) : Prop := x + y + 3 = 0

def min_QM (Q M : ℝ × ℝ) : ℝ := distance Q M

-- Main theorem
theorem main_theorem (Q : ℝ × ℝ)
  (hQ : l1 Q.1 Q.2)
  (M : ℝ × ℝ)
  (hM : curve_C M.1 M.2)
  (h_intersect : ∀ l2 : ℝ × ℝ → Prop, l2 Q → curve_C M.1 M.2 → M ∈ l2 → (∀ B ≠ (M = B → ¬curve_C B.1 B.2))) :
  curve_C (5 : ℝ) 0 ∧ (∃ min_val, min_val = 4 ∧ min_val = min_QM Q M) :=
by sorry

end main_theorem_l500_500110


namespace common_area_of_squares_is_correct_l500_500294

-- Define the side lengths of the smaller and larger squares
def side_small_square : ℝ := 1
def side_large_square : ℝ := 4

-- Define the areas of the smaller and the larger squares
def area_small_square : ℝ := side_small_square ^ 2
def area_large_square : ℝ := side_large_square ^ 2

-- Define the areas of the non-overlapping triangles
def area_small_triangle : ℝ := (1 / 2) * side_small_square * side_small_square
def area_large_triangle : ℝ := (1 / 2) * (side_small_square * 2) * (side_small_square * 2)

-- Define the correct answer for the area of the common part (overlapping area)
def overlapping_area : ℝ := area_large_square - (area_small_triangle + area_large_triangle)

-- The proof statement
theorem common_area_of_squares_is_correct :
  overlapping_area = 13.5 :=
by
  -- Establish the expected relationship
  sorry

end common_area_of_squares_is_correct_l500_500294


namespace extra_postage_envelopes_count_l500_500269

structure Envelope where
  length : ℝ
  height : ℝ

def requiresExtraPostage (e : Envelope) : Bool :=
  (e.length / e.height < 1.2) ∨ (e.length / e.height > 3.0) ∨ (e.height < 3)

def envelopeA : Envelope := { length := 7, height := 5 }
def envelopeB : Envelope := { length := 10, height := 2 }
def envelopeC : Envelope := { length := 8, height := 8 }
def envelopeD : Envelope := { length := 12, height := 4 }

def envelopes : List Envelope := [envelopeA, envelopeB, envelopeC, envelopeD]

def countEnvelopesRequiringExtraPostage (envs : List Envelope) : ℕ :=
  envs.count requiresExtraPostage

theorem extra_postage_envelopes_count : 
  countEnvelopesRequiringExtraPostage envelopes = 3 := by
    sorry

end extra_postage_envelopes_count_l500_500269


namespace isosceles_triangle_leg_length_l500_500635

theorem isosceles_triangle_leg_length (A B C : Point) (AB AC : Real) (alpha l : Real)
  [isosceles : AB = AC]
  [vertex_angle : angle A B C = alpha]
  [altitudes_sum : AA₁ + BB₁ = l] :
  AC = l / (2 * sin((π + alpha) / 4) * cos((π - 3 * alpha) / 4)) := 
sorry

end isosceles_triangle_leg_length_l500_500635


namespace pigpen_area_is_correct_l500_500738

def trapezoid_area (b1 b2 h : ℝ) : ℝ := (b1 + b2) * h / 2

theorem pigpen_area_is_correct :
  let total_fence := 35
  let height := 7
  let short_base := total_fence - height
  let long_base := 0
  trapezoid_area short_base long_base height = 98 :=
by
  have total_fence := 35
  have height := 7
  have short_base := total_fence - height
  have long_base := 0
  calc
    trapezoid_area short_base long_base height
      = (short_base + long_base) * height / 2 : by rfl
    ... = (28 + 0) * 7 / 2                : by rfl
    ... = 196 / 2                        : by norm_num
    ... = 98                             : by norm_num

end pigpen_area_is_correct_l500_500738


namespace area_of_region_l500_500425

theorem area_of_region (x y : ℝ) (h : x^2 + y^2 + 6 * x - 8 * y - 5 = 0) : 
  ∃ (r : ℝ), (π * r^2 = 30 * π) :=
by -- Starting the proof, skipping the detailed steps
sorry -- Proof placeholder

end area_of_region_l500_500425


namespace quadratic_eq_two_real_roots_find_p_value_theorem_l500_500802

noncomputable def quadratic_eq_has_two_real_roots (p : ℝ) : Prop :=
  let Δ := ((2 * p + 1)^2) in Δ ≥ 0

noncomputable def find_p_value (p x1 x2 : ℝ) : Prop :=
  x1 + x2 = 5 ∧
  x1 * x2 = 6 - p^2 - p ∧
  x1^2 + x2^2 - x1 * x2 = 3 * p^2 + 1 → p = -2

theorem quadratic_eq_two_real_roots (p : ℝ) : quadratic_eq_has_two_real_roots p := by
  sorry  -- Skipping the proof

theorem find_p_value_theorem (p x1 x2 : ℝ) (h1 : x1 + x2 = 5) (h2 : x1 * x2 = 6 - p^2 - p) (h3 : x1^2 + x2^2 - x1 * x2 = 3 * p^2 + 1) : find_p_value p x1 x2 := by
  sorry  -- Skipping the proof

end quadratic_eq_two_real_roots_find_p_value_theorem_l500_500802


namespace average_skips_proof_l500_500591

-- Defining the skips per round for Sam
def sam_skips : ℕ := 16

-- Round-specific conditions for Jeff
def jeff_first_round_skips (sam_skips : ℕ) : ℕ := sam_skips - 1
def jeff_second_round_skips (sam_skips : ℕ) : ℕ := sam_skips - 3
def jeff_third_round_skips (sam_skips : ℕ) : ℕ := sam_skips + 4
def jeff_fourth_round_skips (sam_skips : ℕ) : ℕ := sam_skips / 2
def jeff_fifth_round_skips (sam_skips : ℕ) (jeff_fourth_round_skips : ℕ) : ℕ :=
  let diff := sam_skips - jeff_fourth_round_skips
  in jeff_fourth_round_skips + diff + 2

-- Total skips by Jeff
def total_jeff_skips (sam_skips : ℕ) : ℕ :=
  jeff_first_round_skips sam_skips +
  jeff_second_round_skips sam_skips +
  jeff_third_round_skips sam_skips +
  jeff_fourth_round_skips sam_skips +
  jeff_fifth_round_skips sam_skips (jeff_fourth_round_skips sam_skips)

-- Average skips per round completed by Jeff
def jeff_average_skips (sam_skips : ℕ) : ℚ :=
  total_jeff_skips sam_skips / 5

theorem average_skips_proof : jeff_average_skips sam_skips = 14.8 := by
  sorry

end average_skips_proof_l500_500591


namespace measure_of_angle_D_in_scalene_triangle_l500_500649

-- Define the conditions
def is_scalene (D E F : ℝ) : Prop :=
  D ≠ E ∧ E ≠ F ∧ D ≠ F

-- Define the measure of angles based on the given conditions
def measure_of_angle_D (D E F : ℝ) : Prop :=
  E = 2 * D ∧ F = 40

-- Define the sum of angles in a triangle
def triangle_angle_sum (D E F : ℝ) : Prop :=
  D + E + F = 180

theorem measure_of_angle_D_in_scalene_triangle (D E F : ℝ) (h_scalene : is_scalene D E F) 
  (h_measures : measure_of_angle_D D E F) (h_sum : triangle_angle_sum D E F) : D = 140 / 3 :=
by 
  sorry

end measure_of_angle_D_in_scalene_triangle_l500_500649


namespace sum_probability_symmetry_l500_500617

theorem sum_probability_symmetry (S : ℕ) (S = 15) : 
  let p := probability_of_sum 9 S in 
  probability_of_sum 9 (2 * (9 + 54) / 2 - S) = p :=
sorry

end sum_probability_symmetry_l500_500617


namespace farmer_plant_beds_l500_500701

theorem farmer_plant_beds :
  ∀ (bean_seedlings pumpkin_seeds radishes seedlings_per_row_pumpkin seedlings_per_row_radish radish_rows_per_bed : ℕ),
    bean_seedlings = 64 →
    seedlings_per_row_pumpkin = 7 →
    pumpkin_seeds = 84 →
    seedlings_per_row_radish = 6 →
    radish_rows_per_bed = 2 →
    (bean_seedlings / 8 + pumpkin_seeds / seedlings_per_row_pumpkin + radishes / seedlings_per_row_radish) / radish_rows_per_bed = 14 :=
by
  -- sorry to skip the proof
  sorry

end farmer_plant_beds_l500_500701


namespace max_min_sin_x_minus_cos_sq_y_l500_500104

theorem max_min_sin_x_minus_cos_sq_y :
  ∀ (x y : ℝ),
  sin x + sin y = 1 / 3 →
  (∃ (max min : ℝ), max = 4 / 3 ∧ min = -11 / 12 ∧
    ∀ (sin_x_minus_cos_sq_y : ℝ), sin_x_minus_cos_sq_y = sin x - (1 - (cos y)^2) →
    max ≥ sin_x_minus_cos_sq_y ∧ min ≤ sin_x_minus_cos_sq_y) :=
by sorry

end max_min_sin_x_minus_cos_sq_y_l500_500104


namespace mask_proof_problem_l500_500060

open Real

/- Definitions based on the conditions -/
def firstBatchCost : ℝ := 4000
def secondBatchCost : ℝ := 7500

-- Total costs of both batches
def totalCost : ℝ := firstBatchCost + secondBatchCost

-- Total profit condition
def maxProfit : ℝ := 3500

-- Establishing the number of masks in the first and second batch
def firstBatchPacks (x : ℝ) : ℝ := x
def secondBatchPacks (x : ℝ) : ℝ := 1.5 * x

-- Cost per pack equations
def firstBatchCostPerPack (x : ℝ) : ℝ := firstBatchCost / x
def secondBatchCostPerPack (x : ℝ) : ℝ := secondBatchCost / (1.5 * x)

-- Selling price per pack
def sellingPrice (x y : ℝ) : Prop := (firstBatchPacks x * y + secondBatchPacks x * y) - totalCost ≤ maxProfit

theorem mask_proof_problem :
  (∃ x : ℝ, 
    (secondBatchCostPerPack x = firstBatchCostPerPack x + 0.5) ∧
    ∃ y : ℝ, sellingPrice x y ∧ y ≤ 3) :=
by
  -- Placeholder for proof steps
  sorry

end mask_proof_problem_l500_500060


namespace range_of_magnitude_difference_l500_500140

theorem range_of_magnitude_difference :
  let a : ℝ × ℝ := (1, real.sqrt 3)
  let b : ℝ × ℝ := (0, 1)
  let range := {x : ℝ | ∃ t : ℝ, t ∈ set.Icc (-real.sqrt 3) 2 ∧ x = real.norm ⟨1 - t • 0, real.sqrt 3 - t • 1⟩}
  range = set.Icc 1 (real.sqrt 13) :=
by
  sorry

end range_of_magnitude_difference_l500_500140


namespace arithmetic_sequence_fibonacci_l500_500268

noncomputable def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem arithmetic_sequence_fibonacci (a b c : ℕ) 
  (h1 : fib 1 = 1) 
  (h2 : fib 2 = 1) 
  (h3 : ∀ n ≥ 3, fib n = fib (n - 1) + fib (n - 2))
  (h4 : fib b > fib (b-1) ∧ fib (b-2) < fib (b-1))
  (h5 : a + b + c = 3000)
  (h6 : a = b-2 ∧ c = b+2) :
  a = 998 :=
sorry

end arithmetic_sequence_fibonacci_l500_500268


namespace person_speed_l500_500708

theorem person_speed (distance time : ℕ) (h_distance : distance = 240) (h_time : time = 6) : 
  distance / time = 40 :=
by
  rw [h_distance, h_time]
  norm_num
  sorry

end person_speed_l500_500708


namespace olympiad_frequency_within_millennium_l500_500291

open Nat

def valid_olympiad_year (N Y : Nat) : Prop :=
  N = (Y % 100 / 10) + (Y % 10) * 10

theorem olympiad_frequency_within_millennium :
  ∃ (years : List Nat), ∀ y ∈ years, valid_olympiad_year 70 y ∧ (2000 ≤ y ∧ y < 3000) ∧ (length years = 2) :=
by
  sorry

end olympiad_frequency_within_millennium_l500_500291


namespace range_of_P_l500_500807

theorem range_of_P (x y : ℝ) (h : x^2 / 3 + y^2 = 1) :
    2 ≤ |2 * x + y - 4| + |4 - x - 2 * y| ∧ |2 * x + y - 4| + |4 - x - 2 * y| ≤ 14 :=
begin
  sorry
end

end range_of_P_l500_500807


namespace polar_to_cartesian_eq_minimum_perimeter_PQRS_l500_500974

noncomputable def polar_to_rectangular (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Real.cos theta, rho * Real.sin theta)

theorem polar_to_cartesian_eq :
  ∀ (rho : ℝ), ∀ (theta : ℝ),
  (rho^2 = 3 / (1 + 2 * (Real.sin theta)^2)) →
  (∃ (x y : ℝ), polar_to_rectangular rho theta = (x, y) ∧ (x^2 + 3 * y^2 = 3)) :=
by
  sorry

theorem minimum_perimeter_PQRS :
  ∀ (theta : ℝ),
  (∃ (P Q R S : ℝ × ℝ), 
    let xP := sqrt 3 / 2 in
    let yP := 1 / 2 in
    let perimeter := 4 - 2 * Real.sin (theta + π / 3) in
    P = (xP, yP) →
    (perimeter = 4) ∧ (P = (3/2, 1/2))) :=
by
  sorry

end polar_to_cartesian_eq_minimum_perimeter_PQRS_l500_500974


namespace Petya_can_guarantee_win_l500_500244

def initial_number := "11223334445555666677777"

def erases_last_digit_wins (current_number : String) (player : String) : Bool :=
  -- Define what it means to "erase the last digit and win" here.
  -- This is a placeholder function for the game's winning condition.
  sorry

theorem Petya_can_guarantee_win : erases_last_digit_wins initial_number "Petya" := by
  -- Petya's strategy guarantees a win
  sorry

end Petya_can_guarantee_win_l500_500244


namespace regular_pyramid_dihedral_angle_l500_500622

noncomputable def dihedral_angle (n : ℕ) : ℝ :=
  if n = 3 then 70 + 32 / 60
  else if n = 4 then 109 + 28 / 60
  else if n = 5 then 138 + 12 / 60
  else 0  -- undefined for other n, can be adjusted if needed

theorem regular_pyramid_dihedral_angle (n : ℕ) (h : n = 3 ∨ n = 4 ∨ n = 5)
  (H : ∀ k : ℕ, k = 3 ∨ k = 4 ∨ k = 5 → dihedral_angle k = if k = 3 then 70 + 32 / 60 else if k = 4 then 109 + 28 / 60 else 138 + 12 / 60) :
  dihedral_angle n = if n = 3 then 70 + 32 / 60 else if n = 4 then 109 + 28 / 60 else 138 + 12 / 60 :=
by
  cases h
  · rw [h]; exact H 3 (or.inl rfl)
  · cases h
    · rw [h]; exact H 4 (or.inr (or.inl rfl))
    · rw [h]; exact H 5 (or.inr (or.inr rfl))

end regular_pyramid_dihedral_angle_l500_500622


namespace measure_of_angle_in_regular_100_gon_l500_500157

theorem measure_of_angle_in_regular_100_gon :
  let d := 100 
  let vertex_angle := 34.2
  ∀ (P : Fin d → Point), RegularPolygon P d → ∠(P 20) (P 2) (P 1) = vertex_angle :=
by
  sorry

end measure_of_angle_in_regular_100_gon_l500_500157


namespace sum_of_powers_l500_500950

theorem sum_of_powers (n : ℕ) (i : ℂ) (h_mul6 : ∃ m : ℕ, n = 6 * m)
  (h_i : i^2 = -1) : 
  ∑ k in (Finset.range (n + 1)), (k + 1) * i^k = (4 * n) / 3 + 1 + (n / 2) * i := 
by
  sorry

end sum_of_powers_l500_500950


namespace isabella_most_efficient_jumper_l500_500589

noncomputable def weight_ricciana : ℝ := 120
noncomputable def jump_ricciana : ℝ := 4

noncomputable def weight_margarita : ℝ := 110
noncomputable def jump_margarita : ℝ := 2 * jump_ricciana - 1

noncomputable def weight_isabella : ℝ := 100
noncomputable def jump_isabella : ℝ := jump_ricciana + 3

noncomputable def ratio_ricciana : ℝ := weight_ricciana / jump_ricciana
noncomputable def ratio_margarita : ℝ := weight_margarita / jump_margarita
noncomputable def ratio_isabella : ℝ := weight_isabella / jump_isabella

theorem isabella_most_efficient_jumper :
  ratio_isabella < ratio_margarita ∧ ratio_isabella < ratio_ricciana :=
by
  sorry

end isabella_most_efficient_jumper_l500_500589


namespace A_and_D_independent_l500_500308

-- Define the probabilities of elementary events
def prob_A : ℚ := 1 / 6
def prob_B : ℚ := 1 / 6
def prob_C : ℚ := 5 / 36
def prob_D : ℚ := 1 / 6

-- Define the joint probability of A and D
def prob_A_and_D : ℚ := 1 / 36

-- Define the independence condition
def independent (P_X P_Y P_XY : ℚ) : Prop := P_XY = P_X * P_Y

-- Prove that events A and D are independent
theorem A_and_D_independent : 
  independent prob_A prob_D prob_A_and_D := by
  -- The proof is skipped
  sorry

end A_and_D_independent_l500_500308


namespace tan_alpha_sqrt_15_over_15_l500_500921

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l500_500921


namespace range_of_a_for_inequality_solutions_to_equation_l500_500501

noncomputable def f (x a : ℝ) := x^2 + 2 * a * x + 1
noncomputable def f_prime (x a : ℝ) := 2 * x + 2 * a

theorem range_of_a_for_inequality :
  (∀ x, -2 ≤ x ∧ x ≤ -1 → f x a ≤ f_prime x a) → a ≥ 3 / 2 :=
sorry

theorem solutions_to_equation (a : ℝ) (x : ℝ) :
  f x a = |f_prime x a| ↔ 
  (if a < -1 then x = -1 ∨ x = 1 - 2 * a 
  else if -1 ≤ a ∧ a ≤ 1 then x = 1 ∨ x = -1 ∨ x = 1 - 2 * a ∨ x = -(1 + 2 * a)
  else x = 1 ∨ x = -(1 + 2 * a)) :=
sorry

end range_of_a_for_inequality_solutions_to_equation_l500_500501


namespace octagon_area_in_square_l500_500397

/--
An octagon is inscribed in a square such that each vertex of the octagon cuts off a corner
triangle from the square. Each triangle has legs equal to one-fourth of the square's side.
If the perimeter of the square is 160 centimeters, what is the area of the octagon?
-/
theorem octagon_area_in_square
  (side_of_square : ℝ)
  (h1 : 4 * (side_of_square / 4) = side_of_square)
  (h2 : 8 * (side_of_square / 4) = side_of_square)
  (perimeter_of_square : ℝ)
  (h3 : perimeter_of_square = 160)
  (area_of_square : ℝ)
  (h4 : area_of_square = side_of_square^2)
  : ∃ (area_of_octagon : ℝ), area_of_octagon = 1400 := by
  sorry

end octagon_area_in_square_l500_500397


namespace digits_reversal_divisible_by_9_l500_500805

theorem digits_reversal_divisible_by_9 {N : ℕ} (N' : ℕ) (h_reverse : ∀ (d : ℕ), d ∈ digits N → d ∈ digits N') : (N - N').natAbs % 9 = 0 := by
  sorry

end digits_reversal_divisible_by_9_l500_500805


namespace exists_root_in_interval_l500_500078

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1) - Real.log (x - 1) / Real.log 2

theorem exists_root_in_interval :
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  -- Proof goes here
  sorry

end exists_root_in_interval_l500_500078


namespace discount_percentage_l500_500650

theorem discount_percentage 
  (evening_ticket_cost : ℝ) (food_combo_cost : ℝ) (savings : ℝ) (discounted_food_combo_cost : ℝ) (discounted_total_cost : ℝ) 
  (h1 : evening_ticket_cost = 10) 
  (h2 : food_combo_cost = 10)
  (h3 : discounted_food_combo_cost = 10 * 0.5)
  (h4 : discounted_total_cost = evening_ticket_cost + food_combo_cost - savings)
  (h5 : savings = 7)
: (1 - discounted_total_cost / (evening_ticket_cost + food_combo_cost)) * 100 = 20 :=
by
  sorry

end discount_percentage_l500_500650


namespace three_digit_numbers_no_6_or_8_l500_500149

theorem three_digit_numbers_no_6_or_8 : ∀ n, 100 ≤ n ∧ n ≤ 999 → 
  (n % 10 ≠ 6 ∧ n % 10 ≠ 8) ∧ (n / 10 % 10 ≠ 6 ∧ n / 10 % 10 ≠ 8) ∧ (n / 100 ≠ 6 ∧ n / 100 ≠ 8) → 
  {n | 100 ≤ n ∧ n ≤ 999 ∧ (n % 10 ≠ 6 ∧ n % 10 ≠ 8) ∧ (n / 10 % 10 ≠ 6 ∧ n / 10 % 10 ≠ 8) ∧ (n / 100 ≠ 6 ∧ n / 100 ≠ 8)}.card = 448 :=
by sorry

end three_digit_numbers_no_6_or_8_l500_500149


namespace board_coloring_l500_500756

open Finset

theorem board_coloring :
  ∃ (configs : Finset (Matrix (Fin 8) (Fin 8) Bool)), 
    (∀ m ∈ configs, (card (filter (λ c, c = tt) m.entries)) = 31 ∧ 
                   ∀ i j, (m i j = tt → ∀ (di dj : Fin 8), 
                           (abs (i - di) + abs (j - dj) = 1 → m di dj ≠ tt))) ∧ 
    card configs = 68 :=
sorry

end board_coloring_l500_500756


namespace scientific_notation_of_28_million_l500_500000

def scientific_notation_form (a : ℝ) (n : ℤ) : Prop := 1 ≤ |a| ∧ |a| < 10

theorem scientific_notation_of_28_million :
  ∃ a n, (scientific_notation_form a n) ∧ 28000000 = a * 10^n ∧ a = 2.8 ∧ n = 7 :=
by {
  use [2.8, 7],
  exact ⟨⟨by norm_num, by norm_num⟩, by norm_num, rfl, rfl⟩,
}

end scientific_notation_of_28_million_l500_500000


namespace abs_diff_x_y_l500_500205

noncomputable def floor (z : ℝ) : ℤ := int.floor z
noncomputable def frac (z : ℝ) : ℝ := z - floor z

theorem abs_diff_x_y (x y : ℝ)
  (h1 : floor x + frac y = 3.7)
  (h2 : frac x + floor y = 4.2) :
  abs (x - y) = 1.5 :=
by
  sorry

end abs_diff_x_y_l500_500205


namespace fraction_addition_l500_500406

variable (a : ℝ)

theorem fraction_addition (ha : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by
  sorry

end fraction_addition_l500_500406


namespace abs_diff_eq_1point5_l500_500207

theorem abs_diff_eq_1point5 (x y : ℝ)
    (hx : (⌊x⌋ : ℝ) + (y - ⌊y⌋) = 3.7)
    (hy : (x - ⌊x⌋) + (⌊y⌋ : ℝ) = 4.2) :
        |x - y| = 1.5 :=
by
  sorry

end abs_diff_eq_1point5_l500_500207


namespace area_PQRSTU_is_41_point_5_l500_500064

-- Define the lengths of the sides
def PQ : ℝ := 4
def QR : ℝ := 7
def RS : ℝ := 5
def ST : ℝ := 6
def TU : ℝ := 3

-- Define the condition that PT is parallel to QR
def PT_parallel_QR : (PT : ℝ) → Prop := λ PT, True  -- Parallelism input is generalized here

-- Define that PU divides the polygon into a triangle and a trapezoid
def PU_divides := True  -- This would require further geometric definitions, simplified here

-- Prove the total area of polygon PQRSTU given the conditions
theorem area_PQRSTU_is_41_point_5 :
  ∃ (PT : ℝ) (PU : ℝ), PT_parallel_QR PT ∧ PU_divides ∧
  (1/2 * PT * PU + 1/2 * (PQ + QR) * ST = 41.5) :=
begin
  sorry
end

end area_PQRSTU_is_41_point_5_l500_500064


namespace parabola_focus_find_p_and_min_value_l500_500841

theorem parabola_focus_find_p_and_min_value (p : ℝ) (M : ℝ × ℝ) :
  (∃ (F : ℝ × ℝ), F = (2, 0) ∧ ∃ (A : ℝ × ℝ), A = (6, 3) ∧ 
  ∀ (M : ℝ × ℝ), M.1 = (M.2 ^ 2) / (2 * p) ∧ 
  (F.1 = 2 ∧ F.2 = 0)) →
  (∃ (p_val : ℝ), p_val = 4 ∧ 
  ∃ (min_val : ℝ), min_val = 8) :=
begin
  sorry
end

end parabola_focus_find_p_and_min_value_l500_500841


namespace larger_number_is_17_l500_500652

noncomputable def x : ℤ := 17
noncomputable def y : ℤ := 12

def sum_condition : Prop := x + y = 29
def diff_condition : Prop := x - y = 5

theorem larger_number_is_17 (h_sum : sum_condition) (h_diff : diff_condition) : x = 17 :=
by {
  sorry
}

end larger_number_is_17_l500_500652


namespace total_unbroken_seashells_l500_500239

/-
Given:
On the first day, Tom found 7 seashells but 4 were broken.
On the second day, he found 12 seashells but 5 were broken.
On the third day, he found 15 seashells but 8 were broken.

We need to prove that Tom found 17 unbroken seashells in total over the three days.
-/

def first_day_total := 7
def first_day_broken := 4
def first_day_unbroken := first_day_total - first_day_broken

def second_day_total := 12
def second_day_broken := 5
def second_day_unbroken := second_day_total - second_day_broken

def third_day_total := 15
def third_day_broken := 8
def third_day_unbroken := third_day_total - third_day_broken

def total_unbroken := first_day_unbroken + second_day_unbroken + third_day_unbroken

theorem total_unbroken_seashells : total_unbroken = 17 := by
  sorry

end total_unbroken_seashells_l500_500239


namespace part1_part2_part3_l500_500837

def f (x : ℝ) (a : ℝ) : ℝ := 2^x / a + a / 2^x - 1

theorem part1 (h : ∀ x : ℝ, f x a = f (-x) a) (pos_a : a > 0) : a = 1 :=
by
  -- Proof omitted
  sorry

theorem part2 (h_a : a = 1) : { x : ℝ | f x 1 < 13 / 4 } = set.Ioo (-2 : ℝ) 2 :=
by 
  -- Proof omitted
  sorry

theorem part3 (h_ineq : ∀ x : ℝ, x > 0 → m * f x 1 ≥ 2^(-x) - m) : m ≥ 1/2 :=
by
  -- Proof omitted
  sorry

end part1_part2_part3_l500_500837


namespace tan_alpha_value_l500_500883

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l500_500883


namespace proposition_a_is_true_l500_500667

-- Define a quadrilateral
structure Quadrilateral (α : Type*) [Ring α] :=
(a b c d : α)

-- Define properties of a Quadrilateral
def parallel_and_equal_opposite_sides (Q : Quadrilateral ℝ) : Prop := sorry  -- Assumes parallel and equal opposite sides
def is_parallelogram (Q : Quadrilateral ℝ) : Prop := sorry  -- Defines a parallelogram

-- The theorem we need to prove
theorem proposition_a_is_true (Q : Quadrilateral ℝ) (h : parallel_and_equal_opposite_sides Q) : is_parallelogram Q :=
sorry

end proposition_a_is_true_l500_500667


namespace max_gold_coins_l500_500356

theorem max_gold_coins : ∃ n : ℕ, (∃ k : ℕ, n = 7 * k + 2) ∧ 50 < n ∧ n < 150 ∧ n = 149 :=
by
  sorry

end max_gold_coins_l500_500356


namespace line_intersects_y_axis_at_0_2_l500_500740

theorem line_intersects_y_axis_at_0_2 (P1 P2 : ℝ × ℝ) (h1 : P1 = (2, 8)) (h2 : P2 = (6, 20)) :
  ∃ y : ℝ, (0, y) = (0, 2) :=
by {
  sorry
}

end line_intersects_y_axis_at_0_2_l500_500740


namespace problem1_problem2_problem3_l500_500177

noncomputable def P₀ := (0 : ℤ, 1 : ℤ)
noncomputable def P₀' := (0 : ℤ, 0 : ℤ)

def delta_x (k : ℕ) (x : ℕ → ℤ) := x k - x (k - 1)
def delta_y (k : ℕ) (y : ℕ → ℤ) := y k - y (k - 1)
def delta_cond (k : ℕ) (x y : ℕ → ℤ) : Prop := abs (delta_x k x) * abs (delta_y k y) = 2

-- (1) Problem: Given point P₀ (0, 1) and point P₁ satisfies Δy₁ > Δx₁ > 0, find the coordinates of P₁
theorem problem1 {x y : ℕ → ℤ} (hP₀ : x 0 = 0 ∧ y 0 = 1) (h1 : 0 < delta_x 1 x ∧ delta_x 1 x < delta_y 1 y)
  (hc: delta_cond 1 x y) : x 1 = 1 ∧ y 1 = 3 := sorry

-- (2) Problem: Given point P₀ (0, 1), Δxₖ = 1, and the sequence {yₖ} is increasing. Pₙ is on the line y = 3x - 8, find n
theorem problem2 {x y : ℕ → ℤ} (hP₀ : x 0 = 0 ∧ y 0 = 1) (hx : ∀ k, delta_x k x = 1)
  (hy_incr : ∀ k n, k < n → y k < y n) (hc : ∀ k, delta_cond k x y)
  (hl : ∃ n, y n = 3 * x n - 8) : ∃ n, n = 9 := sorry

-- (3) Problem: If the coordinates of point P₀ are (0,0) and y_{2016}=100, find the maximum value of x⁰ + x₁ + x₂ + … + x_{2016}
theorem problem3 {x y : ℕ → ℤ} (hP₀ : x 0 = 0 ∧ y 0 = 0) (hy : y 2016 = 100)
  (hc: ∀ k, delta_cond k x y) : x 0 + x 1 + x 2 + ... + x 2016 ≤ 4066272 := sorry

end problem1_problem2_problem3_l500_500177


namespace Flora_initial_daily_milk_l500_500457

def total_gallons : ℕ := 105
def total_weeks : ℕ := 3
def days_per_week : ℕ := 7
def total_days : ℕ := total_weeks * days_per_week
def extra_gallons_daily : ℕ := 2

theorem Flora_initial_daily_milk : 
  (total_gallons / total_days) = 5 := by
  sorry

end Flora_initial_daily_milk_l500_500457


namespace max_area_of_rectangle_is_square_l500_500248

theorem max_area_of_rectangle_is_square (p a : ℝ) (h_perim : p = 2 * (a + (p / 2 - a))) : 
  ∃ A, A = (p / 4) ^ 2 ∧ (∀ a, a ≠ p / 4 → (p / 4) ^ 2 > a * (p / 2 - a)) :=
by
  have h_a := calc_area_eq p
  have h_sq := (p / 4)^2
  use h_sq
  split
  {
      -- Prove square area
      sorry
  }
  {
    -- Prove maximality condition 
    sorry
  }

-- Auxiliary lemma to derive area formula
lemma calc_area_eq (p a : ℝ) : ∃ A, A = a * (p / 2 - a) := 
  exists.intro (a * (p / 2 - a)) rfl

end max_area_of_rectangle_is_square_l500_500248


namespace circumcircle_sums_l500_500981

-- Definitions of geometric objects and conditions.
variables {A B C D F G : Point}
variables (TriangleABC : Triangle A B C)
variables (D_on_AB : OnLine D (Line_through A B))
variables (F_on_BC : OnLine F (Line_through B C))
variables (G_on_AC : OnLine G (Line_through A C))
variables (DF_parallel_AC : Parallel (Line_through D F) (Line_through A C))
variables (DG_parallel_BC : Parallel (Line_through D G) (Line_through B C))

-- Statement of the theorem.
theorem circumcircle_sums (R_ABC : Circle) (R_ADG : Circle) (R_BDF : Circle)
    (Circumcircle_ABC : Circumcircle TriangleABC = R_ABC)
    (Circumcircle_ADG : Circumcircle (Triangle A D G) = R_ADG)
    (Circumcircle_BDF : Circumcircle (Triangle B D F) = R_BDF) :
    R_ADG.radius + R_BDF.radius = R_ABC.radius := 
sorry

end circumcircle_sums_l500_500981


namespace monomial_count_correct_l500_500179

def is_monomial (e : String) : Bool :=
  -- Implement the logic to check if the given expression is a monomial
  sorry

def number_of_monomials (expressions : List String) : Nat :=
  expressions.count is_monomial

def e_1 := "-1"
def e_2 := "-(2/3)*a^2"
def e_3 := "(1/6)*x^2*y"
def e_4 := "3*a + b"
def e_5 := "0"
def e_6 := "(x-1)/2"

def expressions := [e_1, e_2, e_3, e_4, e_5, e_6]

theorem monomial_count_correct :
  number_of_monomials expressions = 4 :=
  sorry

end monomial_count_correct_l500_500179


namespace chord_length_l500_500695

theorem chord_length (r d : ℝ) (h_r : r = 5) (h_d : d = 3) : 
  ∃ PQ : ℝ, PQ = 8 :=
by
  -- Define the points and apply conditions 
  let FG := sqrt (r^2 - d^2)  -- Calculate the length FG
  have h_FG : FG = 4,
  {
    calc FG
        = sqrt (r^2 - d^2) : rfl
    ... = sqrt (25 - 9)     : by rw [h_r, h_d]
    ... = sqrt 16           : rfl
    ... = 4                 : by norm_num
  }

  -- Chord length PQ is twice FG
  use 2 * FG,
  calc 2 * FG
      = 2 * 4 : by rw [h_FG]
  ... = 8   : by norm_num

end chord_length_l500_500695


namespace lawrence_shares_marbles_l500_500556

theorem lawrence_shares_marbles :
  ∃ (n : ℕ), 5504 = n * 86 ∧ n = 64 :=
by
  use 64
  split
  sorry
  rfl

end lawrence_shares_marbles_l500_500556


namespace trip_distance_l500_500994

def gas_coverage : ℕ := 15 -- km per liter
def gas_cost_per_liter : ℝ := 0.90 -- dollars per liter
def option1_base_cost : ℝ := 50 -- cost in dollars
def option2_base_cost : ℝ := 90 -- cost in dollars
def cost_saving : ℝ := 22 -- savings in dollars

noncomputable def distance_each_way : ℝ :=
  let round_trip := 2 * D
  let gas_needed := round_trip / gas_coverage
  let gas_cost := gas_needed * gas_cost_per_liter
  let option1_total_cost := option1_base_cost + gas_cost
  let option2_total_cost := option2_base_cost
  option2_total_cost - option1_total_cost = cost_saving

theorem trip_distance
  (gas_coverage : ℕ)
  (gas_cost_per_liter : ℝ)
  (option1_base_cost : ℝ)
  (option2_base_cost : ℝ)
  (cost_saving : ℝ)
  (D : ℝ) :
  2 * D / (gas_coverage) * gas_cost_per_liter + option1_base_cost = option2_base_cost - cost_saving →
  D = 150 :=
by
  sorry

end trip_distance_l500_500994


namespace measure_of_angle_A_l500_500764

-- Definitions derived from the conditions
variables {A B C D : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

-- Type class for Cyclic Quadrilateral and conditions
class CyclicQuadrilateral (A B C D : Type) :=
  (is_bisector : ∀ (AC : A → B → Prop), (∃ x : ℕ, AC = (λ A B, angle A B = 2 * x) ∧ (angle BAC = x ∧ angle BDC = 3 * x) ∧ (angle ACD = x ∧ angle ABD = 3 * x)))
  (opp_angles_sum_180 : ∀ (A B : Prop), angle A + angle B = 180)

-- The problem statement derived from the lean equivalent question
theorem measure_of_angle_A (Q : CyclicQuadrilateral A B C D) : ∃ m : ℝ, m ∈ {108, 72, 60, 120} :=
sorry

end measure_of_angle_A_l500_500764


namespace chord_length_of_larger_circle_tangent_to_smaller_circle_l500_500272

theorem chord_length_of_larger_circle_tangent_to_smaller_circle :
  ∀ (A B C : ℝ), B = 5 → π * (A ^ 2 - B ^ 2) = 50 * π → (C / 2) ^ 2 + B ^ 2 = A ^ 2 → C = 10 * Real.sqrt 2 :=
by
  intros A B C hB hArea hChord
  sorry

end chord_length_of_larger_circle_tangent_to_smaller_circle_l500_500272


namespace probability_domain_of_f_is_two_thirds_l500_500487

noncomputable def probability_valid_domain : ℚ := 2/3

theorem probability_domain_of_f_is_two_thirds (a : ℝ) (h : 0 ≤ a ∧ a ≤ 6) :
  ∃ p : ℚ, p = probability_valid_domain ∧ 
  ∀ x : ℝ, a = 0 ∨ (a^2 - 4 * a < 0) → f(x) = log(ax^2 - ax + 1) :=
sorry

end probability_domain_of_f_is_two_thirds_l500_500487


namespace tangent_at_1_is_x_y_minus_1_eq_0_l500_500817

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + x else -(x^2 + x)

theorem tangent_at_1_is_x_y_minus_1_eq_0 :
  (∀ x : ℝ, f(-x) = -f(x)) ∧ (∀ x : ℝ, (x ≤ 0) → (f(x) = x^2 + x)) →
  (∃ m b : ℝ, (∀ x y : ℝ, y = f(x) → y = m * x + b) ∧ m = 1 ∧ b = -1) :=
by
  sorry

end tangent_at_1_is_x_y_minus_1_eq_0_l500_500817


namespace tan_alpha_sqrt_15_over_15_l500_500922

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l500_500922


namespace polygon_diagonals_l500_500711

theorem polygon_diagonals (n : ℕ) (h : ∀ i, i < n → (40 : ℝ) = 360 / n) :
  (n = 9) →
  (∃ (d : ℕ),
    1 / 2 * n * (n - 3) = 27 ∧
    d = 27) :=
by
  intro hn
  use 27
  split
  sorry
  exact hn

end polygon_diagonals_l500_500711


namespace tan_alpha_proof_l500_500900

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l500_500900


namespace function_properties_l500_500565

noncomputable def f : ℝ → ℝ := sorry

variables (x y : ℝ)

theorem function_properties 
  (h1 : ∀ x : ℝ, f x = f (-x))
  (h2 : ∀ x y : ℝ, x < y → x < 0 ∧ y < 0 → f x < f y) :
  f (2 ^ (1 / 2023)) < f (cos (2023 * π)) ∧
  f (cos (2023 * π)) < f (log 2022 / log (1 / 2023)) :=
by
  sorry

end function_properties_l500_500565


namespace angle_in_third_quadrant_l500_500951

theorem angle_in_third_quadrant
  (α : ℝ) (hα : 270 < α ∧ α < 360) : 90 < 180 - α ∧ 180 - α < 180 :=
by
  sorry

end angle_in_third_quadrant_l500_500951


namespace video_length_l500_500369

variable (x : ℝ)
variable (Lila_time_per_video Roger_time_per_video : ℝ)

-- Define the conditions
def Lila_condition := Lila_time_per_video = x / 2
def Roger_condition := Roger_time_per_video = x
def total_time_condition := 6 * Lila_time_per_video + 6 * Roger_time_per_video = 900

-- Prove that x = 100 given the conditions
theorem video_length (h1 : Lila_condition) (h2 : Roger_condition) (h3 : total_time_condition) : x = 100 := by
  sorry

end video_length_l500_500369


namespace contradiction_proof_l500_500164

theorem contradiction_proof (a : ℝ) (ha : a > 1) : a^2 > 1 := 
by
  assume h : a^2 ≤ 1
  -- Here would be the contradiction proof steps
  sorry

end contradiction_proof_l500_500164


namespace tan_alpha_value_l500_500878

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l500_500878


namespace simplify_expression_l500_500671

theorem simplify_expression :
  10 / (2 / 0.3) / (0.3 / 0.04) / (0.04 / 0.05) = 0.25 :=
by
  sorry

end simplify_expression_l500_500671


namespace candy_distribution_l500_500645

theorem candy_distribution :
  let total_candies := 25689
      red_candies := 1342
      blue_candies := 8965
      remaining_candies := total_candies - (red_candies + blue_candies)
      each_color_candies := remaining_candies / 3 in
  each_color_candies = 5127 :=
by
  have h1 : 25689 - (1342 + 8965) = 15382,
  { exact rfl, }
  have h2 : 15382 / 3 = 5127,
  { norm_num, sorry }
  exact floor_eq_iff.mpr ⟨le_refl _, by norm_num⟩,

end candy_distribution_l500_500645


namespace find_imaginary_part_l500_500824

noncomputable def z : ℂ := 3 - 4 * complex.I
noncomputable def z_conj : ℂ := complex.conj z
noncomputable def divisor : ℂ := 1 + complex.I
noncomputable def result : ℂ := z_conj / divisor

theorem find_imaginary_part : complex.im result = 1 / 2 :=
by
  sorry

end find_imaginary_part_l500_500824


namespace part1_part2_l500_500489

noncomputable def z (m : ℝ) : ℂ := (m - 1) + m * complex.I

def p (m : ℝ) : Prop := (m - 1 < 0) ∧ (m > 0)
def q (m : ℝ) : Prop := complex.abs (z m) ≤ real.sqrt 5

theorem part1 {m : ℝ} (hnp : ¬ p m) : m ≤ 0 ∨ m ≥ 1 :=
sorry

theorem part2 {m : ℝ} (hpq : p m ∨ q m) : -1 ≤ m ∧ m ≤ 2 :=
sorry

end part1_part2_l500_500489


namespace open_parking_spots_fourth_level_l500_500382

theorem open_parking_spots_fourth_level :
  ∀ (n_first n_total : ℕ)
    (n_second_diff n_third_diff : ℕ),
    n_first = 4 →
    n_second_diff = 7 →
    n_third_diff = 6 →
    n_total = 46 →
    ∃ (n_first n_second n_third n_fourth : ℕ),
      n_second = n_first + n_second_diff ∧
      n_third = n_second + n_third_diff ∧
      n_first + n_second + n_third + n_fourth = n_total ∧
      n_fourth = 14 := by
  sorry

end open_parking_spots_fourth_level_l500_500382


namespace new_people_last_year_l500_500993

theorem new_people_last_year (born : ℕ) (immigrated : ℕ) (h1 : born = 90171) (h2 : immigrated = 16320) :
    born + immigrated = 106491 :=
by
  rw [h1, h2]
  norm_num

end new_people_last_year_l500_500993


namespace num_tangent_lines_l500_500107

/-- 
Given a point (1, 1) and the curve y = x³, 
prove that the number of lines that pass through this point and are tangent to the curve is 2.
-/
theorem num_tangent_lines (P : ℝ × ℝ) (curve : ℝ → ℝ) 
  (hP : P = (1, 1)) (hcurve : curve = λ x, x^3) : 
  ∃ (n : ℕ), n = 2 ∧ ∀ L, tangent_line_through_point P L curve → L = 2 :=
by
  sorry

end num_tangent_lines_l500_500107


namespace circle_area_through_A_D_C_l500_500191

theorem circle_area_through_A_D_C (A B C D : Points) (a b c ad bd cd : Real)
  (h1 : isosceles_triangle A B C)
  (h2 : distance A B = a)
  (h3 : distance A C = a)
  (h4 : distance B C = c)
  (h5 : midpoint D B C)
  (h6 : c = 10)
  (h7 : a = 8)
  (h8 : distance A D ≠ 0) : 
  area_of_circle_passing_through A D C = (39936 / 1521 * π) :=
by 
  sorry

end circle_area_through_A_D_C_l500_500191


namespace example_number_exists_l500_500249

def is_multiple_of (n d : ℕ) : Prop := ∃ k, n = d * k
def sum_of_digits (n : ℕ) : ℕ := (n.toDigits 10).sum

theorem example_number_exists : 
  ∃ N, (is_multiple_of N 2020) ∧ (is_multiple_of (sum_of_digits N) 2020) :=
by
  let N := BigInt.of_nat (10 ^ (505 * 4) * (2 * 10^3 + 2 * 10^1))
  use N
  sorry

end example_number_exists_l500_500249


namespace volume_ratio_of_circumscribed_sphere_to_regular_triangular_prism_l500_500720

theorem volume_ratio_of_circumscribed_sphere_to_regular_triangular_prism
    (a : ℝ) -- side length of the base of the prism
    (π : ℝ) : -- The constant π (pi)

    -- Conditions
    let S : ℝ := a^2 * (Real.sqrt 3) / 4 in -- Area of the base of the prism
    let H : ℝ := 2 * a in -- Height of the prism
    let V_prism : ℝ := S * H in -- Volume of the prism
    let R : ℝ := 2 * a / (Real.sqrt 3) in -- Radius of the circumscribed sphere
    let V_sphere : ℝ := (4 / 3) * π * R^3 in -- Volume of the sphere
    
    -- Conclusion
    V_sphere / V_prism = (64 * π) / 27 :=
by
  sorry

end volume_ratio_of_circumscribed_sphere_to_regular_triangular_prism_l500_500720


namespace combined_distance_l500_500296

noncomputable def radius_wheel1 : ℝ := 22.4
noncomputable def revolutions_wheel1 : ℕ := 750

noncomputable def radius_wheel2 : ℝ := 15.8
noncomputable def revolutions_wheel2 : ℕ := 950

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def distance_covered (r : ℝ) (rev : ℕ) : ℝ := circumference r * rev

theorem combined_distance :
  distance_covered radius_wheel1 revolutions_wheel1 + distance_covered radius_wheel2 revolutions_wheel2 = 199896.96 := by
  sorry

end combined_distance_l500_500296


namespace sum_of_integers_l500_500611

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 10) (h3 : x * y = 56) : x + y = 18 := 
sorry

end sum_of_integers_l500_500611


namespace parametric_line_segment_l500_500133

theorem parametric_line_segment :
  ∃ (a b c d : ℝ),
    b = -3 ∧
    d = 6 ∧
    a + b = 4 ∧
    c + d = 14 ∧
    (a^2 + b^2 + c^2 + d^2 = 158) :=
by
  let b := -3
  let d := 6
  let a := 7
  let c := 8
  use [a, b, c, d]
  split; simp
  split; simp
  split; simp
  sorry

end parametric_line_segment_l500_500133


namespace polynomial_inequality_l500_500557

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

end polynomial_inequality_l500_500557


namespace spices_combination_l500_500592

theorem spices_combination (n k : ℕ) (h_n : n = 7) (h_k : k = 3) :
  nat.choose n k = 35 :=
by {
  rw [h_n, h_k],
  simp,
  exact nat.choose_spec 7 3,
  norm_num,
}

end spices_combination_l500_500592


namespace valid_x_count_l500_500787

noncomputable def count_valid_x : ℕ :=
  (finset.range 24).filter (λ x, x > 16 ∧ x < 24).card

theorem valid_x_count : count_valid_x = 7 := by
  sorry

end valid_x_count_l500_500787


namespace part1_f1_part1_f2_part1_f3_part2_f_gt_1_for_n_1_2_part2_f_lt_1_for_n_ge_3_l500_500102

noncomputable def a (n : ℕ) : ℚ := 1 / (n : ℚ)

noncomputable def S (n : ℕ) : ℚ := (Finset.range (n+1)).sum (λ k => a (k + 1))

noncomputable def f (n : ℕ) : ℚ :=
  if n = 1 then S 2
  else S (2 * n) - S (n - 1)

theorem part1_f1 : f 1 = 3 / 2 := by sorry

theorem part1_f2 : f 2 = 13 / 12 := by sorry

theorem part1_f3 : f 3 = 19 / 20 := by sorry

theorem part2_f_gt_1_for_n_1_2 (n : ℕ) (h₁ : n = 1 ∨ n = 2) : f n > 1 := by sorry

theorem part2_f_lt_1_for_n_ge_3 (n : ℕ) (h₁ : n ≥ 3) : f n < 1 := by sorry

end part1_f1_part1_f2_part1_f3_part2_f_gt_1_for_n_1_2_part2_f_lt_1_for_n_ge_3_l500_500102


namespace translation_coordinates_l500_500970

theorem translation_coordinates (A : ℝ × ℝ) (T : ℝ × ℝ) (A' : ℝ × ℝ) 
  (hA : A = (-4, 3)) (hT : T = (2, 0)) (hA' : A' = (A.1 + T.1, A.2 + T.2)) : 
  A' = (-2, 3) := sorry

end translation_coordinates_l500_500970


namespace train_passing_time_l500_500983

theorem train_passing_time :
  ∀ (length : ℕ) (speed_kmph : ℕ),
    length = 120 →
    speed_kmph = 72 →
    ∃ (time : ℕ), time = 6 :=
by
  intro length speed_kmph hlength hspeed_kmph
  sorry

end train_passing_time_l500_500983


namespace total_profit_calculation_l500_500019

-- Definitions
def investment_B := Rs 333.3333333333334
def investment_A := 3 * investment_B
def investment_C := (3 / 2) * investment_A
def C_share := Rs 3000.0000000000005
def total_profit := 17 * investment_B

-- Theorem statement
theorem total_profit_calculation :
  (C_share = (9 / 2) * investment_B) →
  total_profit = Rs 5666.666666666667 :=
by
  sorry

end total_profit_calculation_l500_500019


namespace min_value_product_l500_500480

theorem min_value_product (n : ℕ) (x : Fin n → ℝ) (h : 0 ≤ x ∧ ∑ i, x i ≤ 1/2) : 
  ∃ (m : ℝ), (∀ y : Fin n → ℝ, (0 ≤ y ∧ ∑ i, y i ≤ 1/2) → ∏ i, (1 - y i) ≥ m) ∧ m = 1/2 := 
begin
  sorry
end

end min_value_product_l500_500480


namespace tan_alpha_value_l500_500940

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l500_500940


namespace division_by_fraction_example_problem_l500_500032

theorem division_by_fraction (a b : ℝ) (hb : b ≠ 0) : 
  a / (1 / b) = a * b :=
by
  -- Proof goes here
  sorry

theorem example_problem : 12 / (1 / 6) = 72 :=
by
  have h : 6 ≠ 0 := by norm_num
  rw division_by_fraction 12 6 h
  norm_num

end division_by_fraction_example_problem_l500_500032


namespace sum_of_segments_AK_KB_eq_AB_l500_500029

-- Given conditions: length of segment AB is 9 cm
def length_AB : ℝ := 9

-- For any point K on segment AB, prove that AK + KB = AB
theorem sum_of_segments_AK_KB_eq_AB (K : ℝ) (h : 0 ≤ K ∧ K ≤ length_AB) : 
  K + (length_AB - K) = length_AB := by
  sorry

end sum_of_segments_AK_KB_eq_AB_l500_500029


namespace Ada_originally_in_5_l500_500595

def seat := Fin 6
def friends := {Ada Bea Ceci Dee Edie Fay : seat}

-- Conditions
variable (Bea : seat) (Ceci : seat) (Dee : seat) (Edie : seat) (Fay : seat)
variable h1 : Bea + 1 < 6 
variable h2 : Ceci < 6
variable h3 : Dee - 2 ≥ 0
variable h4 : Edie < 6
variable h5 : Fay < 6
variable h_end_empty : 0 < 6 ∨ 5 < 6

-- Goal: Prove that Ada was originally sitting in seat 5
theorem Ada_originally_in_5 : friends → Ada = 4 :=
by
  sorry

end Ada_originally_in_5_l500_500595


namespace A_and_D_mutual_independent_l500_500317

-- Probability theory definitions and assumptions.
noncomputable def prob_1_6 : ℚ := 1 / 6
noncomputable def prob_5_36 : ℚ := 5 / 36
noncomputable def prob_6_36 : ℚ := 6 / 36
noncomputable def prob_1_36 : ℚ := 1 / 36

-- Definitions of events with their corresponding probabilities.
def event_A (P : ℚ) : Prop := P = prob_1_6
def event_B (P : ℚ) : Prop := P = prob_1_6
def event_C (P : ℚ) : Prop := P = prob_5_36
def event_D (P : ℚ) : Prop := P = prob_6_36

-- Intersection probabilities:
def intersection_A_C (P : ℚ) : Prop := P = 0
def intersection_A_D (P : ℚ) : Prop := P = prob_1_36
def intersection_B_C (P : ℚ) : Prop := P = prob_1_36
def intersection_C_D (P : ℚ) : Prop := P = 0

-- Mutual independence definition.
def mutual_independent (P_X : ℚ) (P_Y : ℚ) (P_intersect : ℚ) : Prop :=
  P_X * P_Y = P_intersect

-- Theorem to prove:
theorem A_and_D_mutual_independent :
  event_A prob_1_6 →
  event_D prob_6_36 →
  intersection_A_D prob_1_36 →
  mutual_independent prob_1_6 prob_6_36 prob_1_36 := 
by 
  intros hA hD hAD
  rw [event_A, event_D, intersection_A_D] at hA hD hAD
  exact hA.symm ▸ hD.symm ▸ hAD.symm 

#check A_and_D_mutual_independent

end A_and_D_mutual_independent_l500_500317


namespace add_fractions_l500_500410

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by sorry

end add_fractions_l500_500410


namespace find_original_number_l500_500672

-- Define the given conditions
def increased_by_twenty_percent (x : ℝ) : ℝ := x * 1.20

-- State the theorem
theorem find_original_number (x : ℝ) (h : increased_by_twenty_percent x = 480) : x = 400 :=
by
  sorry

end find_original_number_l500_500672


namespace proof_QR_PS_l500_500210

def volume_rectangular_prism (a b c : ℝ) : ℝ :=
  a * b * c

def surface_area_rectangular_prism (a b c : ℝ) : ℝ :=
  2 * (a * b + b * c + a * c)

def Q := 12 * Real.pi
def R := 88
def S := volume_rectangular_prism 2 4 6
def P := (4 * Real.pi) / 3

theorem proof_QR_PS : (Q * R) / (P * S) = 16.5 :=
by
  let Q_val := 12 * Real.pi
  let R_val := 88
  let S_val := volume_rectangular_prism 2 4 6
  let P_val := (4 * Real.pi) / 3
  calc
    (Q_val * R_val) / (P_val * S_val) = (12 * Real.pi * 88) / ((4 * Real.pi / 3) * S_val) : sorry
    ... = 16.5 : sorry

end proof_QR_PS_l500_500210


namespace union_of_M_N_l500_500846

theorem union_of_M_N {a b : ℝ} 
  (M : set ℝ) (N : set ℝ)
  (hM : M = {2, real.log 3 a}) 
  (hN : N = {a, b}) 
  (h_inter : M ∩ N = {1}) : 
  M ∪ N = {1, 2, 3} := 
by 
  sorry

end union_of_M_N_l500_500846


namespace area_of_BEDC_l500_500273

theorem area_of_BEDC (BC BE ED height : ℝ) 
  (hBC : BC = 12) (hBE : BE = 7) (hED : ED = 5) (hHeight : height = 9) :
  let area_ABCD := BC * height,
      area_ABE := (1 / 2) * BE * height 
  in area_ABCD - area_ABE = 76.5 :=
by 
  intros BC BE ED height hBC hBE hED hHeight
  let area_ABCD := BC * height
  let area_ABE := (1 / 2) * BE * height
  have h1 : area_ABCD = 108 := by rw [hBC, hHeight]; norm_num
  have h2 : area_ABE = 31.5 := by rw [hBE, hHeight]; norm_num
  show area_ABCD - area_ABE = 76.5, by rw [h1, h2]; norm_num

end area_of_BEDC_l500_500273


namespace num_possible_values_a_l500_500582

theorem num_possible_values_a (a b c d : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h_order : a > b ∧ b > c ∧ c > d)
    (h_sum : a + b + c + d = 2020)
    (h_diff : a^2 - b^2 + c^2 - d^2 = 2024) : 
    ∃ (n : ℕ), n = 503 :=
begin
  sorry
end

end num_possible_values_a_l500_500582


namespace ellipse_area_l500_500778

theorem ellipse_area : 
  let E : Ellipse := { 
    equation := λ x y => x^2 + 4*x + 9*y^2 + 18*y + 20 = 0 
  }
  in area E = (7 * Real.pi) / 3 := 
by
  sorry

end ellipse_area_l500_500778


namespace unique_extremum_range_two_extremum_points_ineq_l500_500830

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - 1 + a * x^2

-- Part (1)
theorem unique_extremum_range {a : ℝ} :
  (∀ x : ℝ, f a x ≠ f' a x) ↔ a ∈ Set.Ioi 0 := sorry

-- Part (2)
theorem two_extremum_points_ineq {a x1 x2 : ℝ} (h : a < -Real.exp 1 / 2) :
  (f' a x1 = 0 ∧ f' a x2 = 0) → x1^2 + x2^2 > 2 * (a + 1) + Real.exp 1 := sorry

end unique_extremum_range_two_extremum_points_ineq_l500_500830


namespace maximize_profit_l500_500602

def fixed_cost : ℝ := 4

def p (x : ℝ) : ℝ :=
  if x < 70 then (1/2) * x^2 + 40 * x
  else 101 * x + (6400 / x) - 2060

def revenue (x : ℝ) : ℝ := 100 * x

def profit (x : ℝ) : ℝ := revenue x - (fixed_cost + p x)

theorem maximize_profit :
  profit 80 = 1500 :=
sorry

end maximize_profit_l500_500602


namespace arithmetic_seq_a10_l500_500540

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

theorem arithmetic_seq_a10 (h_arith : arithmetic_sequence a) (h2 : a 3 = 5) (h5 : a 6 = 11) : a 10 = 19 := by
  sorry

end arithmetic_seq_a10_l500_500540


namespace max_winning_strategy_l500_500638

/-- 
Theorem: Max has a winning strategy if and only if n ≡ 1 mod 5 or n ≡ 4 mod 5.
-/
theorem max_winning_strategy (n : ℕ) : (n % 5 = 1 ∨ n % 5 = 4) ↔ 
  ∃ strategy : (ℕ → ℕ), ∀ turns : list ℕ, (turns.length = n) → 
  ( (∀ k, k < n → ∃ t, t < turns.length ∧ turns.get t = k) 
    ↔ n % 5 = 1 ∨ n % 5 = 4 ) sorry

end max_winning_strategy_l500_500638


namespace average_in_options_l500_500290

-- Definitions of conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def in_range (n : ℕ) : Prop := 15 < n ∧ n < 23
def average (a b c : ℕ) : ℕ := (a + b + c) / 3

-- The theorem to be proven
theorem average_in_options (N : ℕ) (h_even : is_even N) (h_range : in_range N) :
  let avg := average 8 12 N in avg = 12 ∨ avg = 14 :=
sorry

end average_in_options_l500_500290


namespace sum_of_digits_9ab_l500_500976

noncomputable def a : Nat := 8 * ((10 ^ 2000 - 1) / 9)
noncomputable def b : Nat := 5 * ((10 ^ 2000 - 1) / 9)

theorem sum_of_digits_9ab : 
  (let s := String.toList (Nat.toDigits 10 (9 * a * b))
   in s.foldr (λ c acc, acc + (c.toNat - '0'.toNat)) 0) = 18005 := by
  sorry

end sum_of_digits_9ab_l500_500976


namespace AD_length_l500_500968

noncomputable def length_AD {A B C D : Type*} [metric_space B] 
  (AB BC CD : ℝ) 
  (angle_B : ∠ABC = 90)
  (angle_C : ∠BCD = 120) 
  (h_AB : AB = 7)
  (h_BC : BC = 10)
  (h_CD : CD = 24)
  : ℝ := 
    sqrt (10^2 + (24 - 10)^2 - 2 * 10 * (24 - 10) * cos (60))

theorem AD_length 
  {A B C D : Type*} [metric_space B] 
  (AB BC CD : ℝ) 
  (angle_B : ∠ABC = 90)
  (angle_C : ∠BCD = 120)
  (h_AB : AB = 7)
  (h_BC : BC = 10)
  (h_CD : CD = 24)
  : length_AD AB BC CD angle_B angle_C h_AB h_BC h_CD = 2* sqrt 39 := 
  sorry

end AD_length_l500_500968


namespace function_properties_l500_500120

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - π / 6)

theorem function_properties :
  (∀ x, f x = 2 * Real.sin (2 * x - π / 6)) ∧
  (∀ k : ℤ, (∀ x, (-π / 6 + k * π) ≤ x ∧ x ≤ π / 3 + k * π → Monotone f)) ∧
  (x ∈ [0, π/2] → f x ≤ 2) ∧ (x ∈ [0, π/2] → f x ≥ -1) ∧
  (∃ x ∈ [0, π/2], f x = 2) ∧
  (∃ x ∈ [0, π/2], f x = -1) :=
by
  -- Proof goes here
  sorry

end function_properties_l500_500120


namespace tan_alpha_proof_l500_500899

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l500_500899


namespace percentage_of_bottle_danny_drank_l500_500421

theorem percentage_of_bottle_danny_drank
    (x : ℝ)  -- percentage of the first bottle Danny drinks, represented as a real number
    (b1 b2 b3 : ℝ)  -- volumes of the three bottles, represented as real numbers
    (h_b1 : b1 = 1)  -- first bottle is full (1 bottle)
    (h_b2 : b2 = 1)  -- second bottle is full (1 bottle)
    (h_b3 : b3 = 1)  -- third bottle is full (1 bottle)
    (h_given_away1 : b2 * 0.7 = 0.7)  -- gave away 70% of the second bottle
    (h_given_away2 : b3 * 0.7 = 0.7)  -- gave away 70% of the third bottle
    (h_soda_left : b1 * (1 - x) + b2 * 0.3 + b3 * 0.3 = 0.7)  -- 70% of bottle left
    : x = 0.9 :=
by
  sorry

end percentage_of_bottle_danny_drank_l500_500421


namespace number_of_integer_solutions_is_zero_l500_500761

-- Define the problem conditions
def eq1 (x y z : ℤ) : Prop := x^2 - 3 * x * y + 2 * y^2 - z^2 = 27
def eq2 (x y z : ℤ) : Prop := -x^2 + 6 * y * z + 2 * z^2 = 52
def eq3 (x y z : ℤ) : Prop := x^2 + x * y + 8 * z^2 = 110

-- State the theorem to be proved
theorem number_of_integer_solutions_is_zero :
  ∀ (x y z : ℤ), eq1 x y z → eq2 x y z → eq3 x y z → false :=
by
  sorry

end number_of_integer_solutions_is_zero_l500_500761


namespace find_k_l500_500136

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

theorem find_k (k : ℝ) (h : dot_product (k * a.1, k * a.2 + b.2) (3 * a.1, 3 * a.2 - b.2) = 0) :
  k = 1 / 3 :=
by
  sorry

end find_k_l500_500136


namespace sum_of_terms_in_sequence_is_215_l500_500754

theorem sum_of_terms_in_sequence_is_215 (a d : ℕ) (h1: Nat.Prime a) (h2: Nat.Prime d)
  (hAP : a + 50 = a + 50)
  (hGP : (a + d) * (a + 50) = (a + 2 * d) ^ 2) :
  (a + (a + d) + (a + 2 * d) + (a + 50)) = 215 := sorry

end sum_of_terms_in_sequence_is_215_l500_500754


namespace monotonically_increasing_minimum_value_inequality_ln_sum_l500_500500

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x > 0 then log x + (1 - x) / (a * x) else 0

theorem monotonically_increasing (a : ℝ) (ha : 0 < a) : a ≥ 1 →
  ∀ x ∈ set.Ici (1 : ℝ), f a x ≤ f a (max 1 x) :=
begin
  -- sorry proof skipped.
  sorry
end

theorem minimum_value (a : ℝ) (ha : 0 < a) :
  (0 < a ∧ a ≤ 1/2 → ∃ m, ∀ x ∈ set.Icc (1 : ℝ) 2, f a x ≥ m ∧ m = log 2 - 1 / (2 * a)) ∧
  (1/2 < a ∧ a < 1 → ∃ m, ∀ x ∈ set.Icc (1 : ℝ) 2, f a (1/a) ≤ f a x ∧ f a x ≥ m ∧ m = log (1/a) + 1 - 1/a) ∧
  (a ≥ 1 → ∃ m, ∀ x ∈ set.Icc (1 : ℝ) 2, f a 1 ≤ f a x ∧ f a x ≥ m ∧ m = 0) :=
begin
  -- sorry proof skipped.
  sorry
end

theorem inequality_ln_sum (n : ℕ) (hn : 1 < n) :  
  log n > ∑ (k : ℕ) in finset.range (n - 1), (1 / (k + 2 : ℝ)) :=
begin
  -- sorry proof skipped.
  sorry
end

end monotonically_increasing_minimum_value_inequality_ln_sum_l500_500500


namespace proof_problem_1_proof_problem_2_l500_500510

def S (n : ℕ) : Finset (Fin n → bool) :=
  Finset.univ

def d {n : ℕ} (U V : Fin n → bool) : ℕ :=
  Finset.card ((Finset.filter (fun i => U i ≠ V i) Finset.univ) : Finset (Fin n))

def problem_1 : Prop :=
  let U := fun _ => true
  let S6 := S 6
  ∃! m, m = Finset.card (Finset.filter (fun V => d U V = 2) S6) ∧ m = 15

def problem_2 (n : ℕ) (hn : n ≥ 2) : Prop :=
  let S_n := S n
  ∀ U : Fin n → bool,
  Σ (V : Finset (Fin n → bool)), (V ∈ S_n) → (d U V) = n * 2^(n - 1)

-- Problem statements without proof
theorem proof_problem_1 : problem_1 :=
by sorry

theorem proof_problem_2 (n : ℕ) (hn : n ≥ 2) : problem_2 n hn :=
by sorry

end proof_problem_1_proof_problem_2_l500_500510


namespace xy_value_x2_y2_value_l500_500516

noncomputable def x : ℝ := Real.sqrt 7 + Real.sqrt 3
noncomputable def y : ℝ := Real.sqrt 7 - Real.sqrt 3

theorem xy_value : x * y = 4 := by
  -- proof goes here
  sorry

theorem x2_y2_value : x^2 + y^2 = 20 := by
  -- proof goes here
  sorry

end xy_value_x2_y2_value_l500_500516


namespace five_digit_number_count_with_product_2000_l500_500521

def digit_product (num : ℕ) : ℕ :=
  num.digits.to_list.product

def is_five_digit (num : ℕ) : Prop :=
  num >= 10000 ∧ num < 100000

def valid_digit_range (num : ℕ) : Prop :=
  ∀ d ∈ num.digits, 1 ≤ d ∧ d ≤ 9

theorem five_digit_number_count_with_product_2000 :
  ∃ n : ℕ, n = 30 ∧ ∀ num : ℕ, is_five_digit num ∧ digit_product num = 2000 ∧ valid_digit_range num → num = 30 := sorry

end five_digit_number_count_with_product_2000_l500_500521


namespace train_speed_l500_500730

theorem train_speed (x : ℕ) (h1 : 3 * x > 0) (h2 : 20 > 0) (h3 : 26 > 0) : 
  (∃ V : ℝ, 26 = 3 * x / (x / V + 2 * x / 20) ∧ V = 65) :=
by
  let V := 65
  have h4 : x / V + 2 * x / 20 = (20 * x + 2 * V * x) / (20 * V)
  have h5 : 26 * (20 * x + 2 * V * x) = 3 * x * 20 * V
  have h6 : 520 * x = 60 * V * x - 52 * V * x
  have h7 : 520 = 8 * V
  have h8 : V = 65
  existsi V
  split
  assumption β= sorry

end train_speed_l500_500730


namespace remainder_hx10_div_hx_l500_500566

noncomputable def h (x : ℂ) : ℂ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_hx10_div_hx :
  let hx := h,
      hx10 := h (λ x, x^10)
  in  ∃ r : ℂ, r = 7 ∧ 
       ∃ q : ℂ[X], hx10 = q * hx + polynomial.C r :=
by
  sorry

end remainder_hx10_div_hx_l500_500566


namespace maximize_ratio_coordinates_l500_500123

noncomputable def curve_equation (n : ℕ) (x : ℝ) : ℝ :=
  n * x^2

noncomputable def point_P (n : ℕ) : ℝ × ℝ :=
  (1 / (2 * n), 1 / (4 * n))

theorem maximize_ratio_coordinates (n : ℕ) (hn : n > 0) :
  curve_equation n (point_P n).1 = (point_P n).2 :=
by image sorry

#check maximize_ratio_coordinates

end maximize_ratio_coordinates_l500_500123


namespace customers_left_l500_500392

-- Definitions based on problem conditions
def initial_customers : ℕ := 14
def remaining_customers : ℕ := 3

-- Theorem statement based on the question and the correct answer
theorem customers_left : initial_customers - remaining_customers = 11 := by
  sorry

end customers_left_l500_500392


namespace tan_alpha_value_l500_500892

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l500_500892


namespace tan_alpha_value_l500_500941

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l500_500941


namespace circle_equation_tangent_lines_l500_500493

noncomputable def circle_center : (ℝ × ℝ) := (1, 1)

def chord_length : ℝ := real.sqrt 2

def line_equation (x y : ℝ) : Prop := x + y = 1

def point_outside : (ℝ × ℝ) := (2, 3)

theorem circle_equation (r : ℝ) :
  (chord_length = real.sqrt 2) →
  (line_equation 1 1) →
  ((1 - 1)^2 + (1 - 1)^2 = r^2) →
  ((1)^2 + (1)^2 = (r^2 + (chord_length / real.sqrt 2)^2)) →
  (r^2 = 1) →
  ∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 :=
sorry

theorem tangent_lines :
  (x = 2 ∨ 3 * x - 4 * y + 6 = 0) ∧
  (point_outside = (2, 3)) →
  ∃ x y : ℝ, (x = 2) ∨ (3 * x - 4 * y + 6 = 0) :=
sorry

end circle_equation_tangent_lines_l500_500493


namespace tan_alpha_solution_l500_500865

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l500_500865


namespace area_under_curve_eq_37_over_12_l500_500072

open Real

noncomputable def f (x : ℝ) : ℝ := -x^3 + x^2 + 2*x

theorem area_under_curve_eq_37_over_12 :
  (∫ x in -1..0, -f x) + (∫ x in 0..2, f x) = 37 / 12 := sorry

end area_under_curve_eq_37_over_12_l500_500072


namespace jesse_initial_gift_amount_l500_500200

-- defining the conditions
variables (novel_cost lunch_cost total_spent amount_left gift_amount : ℕ)

-- assume cost of the novel is $7
axiom novel_cost_val : novel_cost = 7 

-- assume cost of lunch is twice the novel
axiom lunch_cost_val : lunch_cost = 2 * novel_cost

-- assume Jesse had $29 left after all expenses
axiom amount_left_val : amount_left = 29

-- total amount spent at the mall
def total_spent_val : total_spent = novel_cost + lunch_cost := by
  rw [novel_cost_val, lunch_cost_val]
  exact rfl

-- defining the total initial gift amount
def initial_gift_amount : gift_amount = total_spent + amount_left := by
  rw total_spent_val
  exact rfl

-- proving the initial gift amount
theorem jesse_initial_gift_amount : gift_amount = 50 := by
  rw [initial_gift_amount, <- total_spent_val, novel_cost_val, lunch_cost_val, amount_left_val]
  simp
  rfl

end jesse_initial_gift_amount_l500_500200


namespace car_license_combinations_l500_500722

-- Define the conditions
def letters := 2
def digit_choices := 10
def digits := 6

-- Define the problem statement
theorem car_license_combinations (letters: ℕ) (digit_choices: ℕ) (digits: ℕ) :
  letters * digit_choices^digits = 2000000 :=
by
  unfold letters digit_choices digits
  sorry

end car_license_combinations_l500_500722


namespace max_height_AC_l500_500959

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {h : ℝ}

-- Given conditions
def b_value : Prop := b = 2 * Real.sqrt 3
def trig_condition : Prop := Real.sqrt 3 * Real.sin C = (Real.sin A + Real.sqrt 3 * Real.cos A) * Real.sin B

theorem max_height_AC (h : ℝ) :
  b_value ∧ trig_condition →
  h ≤ 3 :=
by
  intro h_bound
  intro h_condition
  -- proofs would go here
  sorry

end max_height_AC_l500_500959


namespace qz_length_l500_500183

theorem qz_length (AB YZ AQ BQ QY QZ : ℝ) (h1 : AB ∥ YZ) (h2 : AQ = 36) (h3 : BQ = 18) (h4 : QY = 27) :
  QZ = 54 :=
by
  sorry

end qz_length_l500_500183


namespace expression_value_l500_500494

theorem expression_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x^2 + y^2 + x * y = 529) 
  (h5 : x^2 + z^2 + real.sqrt 3 * x * z = 441) 
  (h6 : z^2 + y^2 = 144) : 
  real.sqrt 3 * x * y + 2 * y * z + x * z = 224 * real.sqrt 5 :=
sorry

end expression_value_l500_500494


namespace sugar_needed_l500_500385

theorem sugar_needed (sugar_needed_for_full_recipe : ℚ) (fraction_of_recipe : ℚ) :
  sugar_needed_for_full_recipe = 23 / 3 → fraction_of_recipe = 1 / 3 → 
  sugar_needed_for_full_recipe * fraction_of_recipe = 2 + (5 / 9) :=
by
  sorry

end sugar_needed_l500_500385


namespace find_simple_annual_rate_l500_500523

-- Conditions from part a).
-- 1. Principal initial amount (P) is $5,000.
-- 2. Annual interest rate for compounded interest (r) is 0.06.
-- 3. Number of times it compounds per year (n) is 2 (semi-annually).
-- 4. Time period (t) is 1 year.
-- 5. The interest earned after one year for simple interest is $6 less than compound interest.

noncomputable def principal : ℝ := 5000
noncomputable def annual_rate_compound : ℝ := 0.06
noncomputable def times_compounded : ℕ := 2
noncomputable def time_years : ℝ := 1
noncomputable def compound_interest : ℝ := principal * (1 + annual_rate_compound / times_compounded) ^ (times_compounded * time_years) - principal
noncomputable def simple_interest : ℝ := compound_interest - 6

-- Question from part a) translated to Lean statement using the condition that simple interest satisfaction
theorem find_simple_annual_rate : 
    ∃ r : ℝ, principal * r * time_years = simple_interest :=
by
  exists (0.0597)
  sorry

end find_simple_annual_rate_l500_500523


namespace relationship_must_hold_l500_500813

-- Definitions for the quadratic equation and the condition on the axis of symmetry
def quadratic_eqn (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def axis_of_symmetry (b a : ℝ) : Prop := 1 = -b / (2 * a)

-- Theorem statement proving the specific relationship given the conditions
theorem relationship_must_hold (a b c : ℝ) (h : axis_of_symmetry b a) : 2 * c < 3 * b :=
by {
  -- The condition for axis of symmetry implies b = -2a
  have h1 : b = -2 * a, from sorry,
  -- Substitute b = -2a in the quadratic inequalities
  rw h1 at *,
  -- Prove the required inequality
  sorry
}

end relationship_must_hold_l500_500813


namespace reflect_parabola_y_axis_l500_500251

theorem reflect_parabola_y_axis (x y : ℝ) :
  (y = 2 * (x - 1)^2 - 4) → (y = 2 * (-x - 1)^2 - 4) :=
sorry

end reflect_parabola_y_axis_l500_500251


namespace trig_expression_l500_500808

theorem trig_expression (θ : ℝ) (h : Real.tan (Real.pi / 4 + θ) = 3) : 
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = -4 / 5 :=
by sorry

end trig_expression_l500_500808


namespace part1_part2_l500_500535

variables {a b c : ℝ} 
variables {A B C : ℝ}
variables {S : ℝ}

-- Conditions
def condition1 := c = 2
def condition2 := C = π / 3
def condition3 := S = sqrt 3
def condition4 := sin B = 2 * sin A

-- Part (1)
theorem part1 (h1 : condition1) (h2 : condition2) (h3 : 1 / 2 * a * b * sin C = sqrt 3) : 
  a = 2 ∧ b = 2 := 
sorry

-- Part (2)
theorem part2 (h1 : condition1) (h2 : condition2) (h4 : condition4) : 
  S = 2 * sqrt 3 / 3 := 
sorry

end part1_part2_l500_500535


namespace digits_in_base_ten_representation_l500_500946

noncomputable def num_digits_of_x (x : ℝ) : ℕ :=
  let log10 := Real.logb 10
  in if log10 (log10 (log10 x)) = 1
  then (10^10).nat_cast + 1
  else 0

theorem digits_in_base_ten_representation (x : ℝ) 
  (h : Real.logb 10 (Real.logb 10 (Real.logb 10 x)) = 1) : 
  num_digits_of_x x = (10^10).nat_cast + 1 :=
by
  sorry

end digits_in_base_ten_representation_l500_500946


namespace lattice_points_5_dist_l500_500977

theorem lattice_points_5_dist : 
  {p : ℕ × ℕ × ℕ // p.1^2 + p.2.1^2 + p.2.2^2 = 25}.card = 9 := 
by
  sorry

end lattice_points_5_dist_l500_500977


namespace largest_three_digit_multiple_of_8_with_sum_24_l500_500656

theorem largest_three_digit_multiple_of_8_with_sum_24 :
  ∃ n : ℕ, (n ≥ 100 ∧ n < 1000) ∧ (∃ k, n = 8 * k) ∧ (n.digits.sum = 24) ∧
           ∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (∃ k', m = 8 * k') ∧ (m.digits.sum = 24) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_8_with_sum_24_l500_500656


namespace exterior_angle_parallel_lines_l500_500972

theorem exterior_angle_parallel_lines
  (k l : Prop) 
  (triangle_has_angles : ∃ (a b c : ℝ), a = 40 ∧ b = 40 ∧ c = 100 ∧ a + b + c = 180)
  (exterior_angle_eq : ∀ (y : ℝ), y = 180 - 100) :
  ∃ (x : ℝ), x = 80 :=
by
  sorry

end exterior_angle_parallel_lines_l500_500972


namespace speed_of_boat_in_still_water_l500_500634

variable (c d t v : ℝ) 

-- Given conditions
def rate_of_current : ℝ := 4
def distance : ℝ := 5.133333333333334
def time : ℝ := 14 / 60

-- The statement to prove
theorem speed_of_boat_in_still_water 
  (hc : c = rate_of_current)
  (hd : d = distance)
  (ht : t = time)
  (h_eq : (v + c) * t = d) : 
  v = 18 := 
  sorry

end speed_of_boat_in_still_water_l500_500634


namespace tan_alpha_fraction_l500_500907

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l500_500907


namespace smallest_possible_difference_l500_500346

def digits : List ℕ := [1, 3, 5, 7, 8]

def isThreeDigitNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def isTwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def usingAllDigitsExactlyOnce (a b : ℕ) : Prop :=
  let aDigits := a.digits
  let bDigits := b.digits
  aDigits.length + bDigits.length = digits.length ∧ (aDigits ++ bDigits).all (∈ digits) ∧ (aDigits ++ bDigits).nodup

noncomputable def smallestDifference := 48

theorem smallest_possible_difference :
  ∃ (a b : ℕ), isThreeDigitNumber a ∧ isTwoDigitNumber b ∧ usingAllDigitsExactlyOnce a b ∧ (a.digits.head = 1 ∨ a.digits.head = 8) ∧ (a - b = smallestDifference) := 
  sorry

end smallest_possible_difference_l500_500346


namespace zeros_of_f_l500_500637

noncomputable def f (x : ℝ) : ℝ := (x - 1) * (x ^ 2 - 2 * x - 3)

theorem zeros_of_f :
  { x : ℝ | f x = 0 } = {1, -1, 3} :=
sorry

end zeros_of_f_l500_500637


namespace gcd_of_g_y_l500_500114

def g (y : ℕ) : ℕ := (3 * y + 4) * (8 * y + 3) * (11 * y + 5) * (y + 11)

theorem gcd_of_g_y (y : ℕ) (hy : ∃ k, y = 30492 * k) : Nat.gcd (g y) y = 660 :=
by
  sorry

end gcd_of_g_y_l500_500114


namespace vasya_average_not_exceed_4_l500_500518

variable (a b c d e : ℕ) 

-- Total number of grades
def total_grades : ℕ := a + b + c + d + e

-- Initial average condition
def initial_condition : Prop := 
  (a + 2 * b + 3 * c + 4 * d + 5 * e) < 3 * (total_grades a b c d e)

-- New average condition after grade changes
def changed_average (a b c d e : ℕ) : ℚ := 
  ((2 * b + 3 * (a + c) + 4 * d + 5 * e) : ℚ) / (total_grades a b c d e)

-- Proof problem to show the new average grade does not exceed 4
theorem vasya_average_not_exceed_4 (h : initial_condition a b c d e) : 
  (changed_average 0 b (c + a) d e) ≤ 4 := 
sorry

end vasya_average_not_exceed_4_l500_500518


namespace ab_minus_c_eq_six_l500_500768

theorem ab_minus_c_eq_six (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a * b + b - 9) : 
  a * b - c = 6 := 
by
  sorry

end ab_minus_c_eq_six_l500_500768


namespace find_range_l500_500954

noncomputable def capricious_function_step_lower (k : ℝ) (x : ℝ) : Prop :=
  k ≥ x - 2

noncomputable def capricious_function_step_upper (k : ℝ) (x : ℝ) : Prop :=
  k ≤ (x + 1) * (Real.log x + 1) / x

noncomputable def capricious_function (k : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 Real.exp 1, capricious_function_step_lower k x ∧ capricious_function_step_upper k x

theorem find_range (k : ℝ) : capricious_function k ↔ k ∈ Set.Icc (Real.exp 1 - 2) 2 :=
by
  sorry

end find_range_l500_500954


namespace mean_value_point_range_m_l500_500160

noncomputable theory

def has_mean_value_point (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ ∈ set.Icc a b, f x₀ = (f b - f a) / (b - a)

theorem mean_value_point_range_m (m : ℝ) (a b : ℝ) 
  (h_ab : 0 ≤ a ∧ a ≤ b ∧ b ≤ 1) : m ∈ set.Icc (-17 / 4) (-2) → 
  has_mean_value_point (λ x, 3^(2 * x) - 3^(x + 1) - m) a b :=
sorry

end mean_value_point_range_m_l500_500160


namespace surface_area_correct_l500_500544

noncomputable def surfaceAreaCircumscribedSphere
  (AB BC AC PC : ℝ)
  (hAB : AB = Real.sqrt 15)
  (hBC : BC = Real.sqrt 15)
  (hAC : AC = 6)
  (hPC : PC = 2)
  (hPerpendicular : ∀ (A B C P : ℝ × ℝ × ℝ), P = (0, 0, 2) → P ⬝ (A, B, C) = 0)
  : ℝ :=
  let radiusSquared := (15/4) * 6 + 1 in
  let surfaceArea := 4 * Real.pi * radiusSquared in
  surfaceArea

theorem surface_area_correct
  (AB BC AC PC : ℝ)
  (hAB : AB = Real.sqrt 15)
  (hBC : BC = Real.sqrt 15)
  (hAC : AC = 6)
  (hPC : PC = 2)
  (hPerpendicular : ∀ (A B C P : ℝ × ℝ × ℝ), P = (0, 0, 2) → P ⬝ (A, B, C) = 0)
  : surfaceAreaCircumscribedSphere AB BC AC PC hAB hBC hAC hPC hPerpendicular = (83 / 2) * Real.pi :=
by
  sorry

end surface_area_correct_l500_500544


namespace tan_alpha_value_l500_500884

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l500_500884


namespace certain_number_is_30_l500_500366

-- Define the condition
def condition : Prop :=
  let p₁ := 0.6 * 50
  let p₂ := 0.42 * x
  p₁ = p₂ + 17.4

-- Prove the given problem
theorem certain_number_is_30 (x : ℝ) (h : condition x) : x = 30 :=
by
  sorry

end certain_number_is_30_l500_500366


namespace wheel_revolutions_l500_500295

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def revolutions (d : ℝ) (r : ℝ) : ℕ := Int.ceil (d / circumference r)

theorem wheel_revolutions
  (r : ℝ := 22.4)
  (d : ℝ := 4224)
  (h : circumference r = 2 * Real.pi * r) :
  revolutions d r = 30 :=
by
  -- Proof steps go here
  sorry

end wheel_revolutions_l500_500295


namespace arrange_scores_l500_500965

variable {K Q M S : ℝ}

theorem arrange_scores (h1 : Q > K) (h2 : M > S) (h3 : S < max Q (max M K)) : S < M ∧ M < Q := by
  sorry

end arrange_scores_l500_500965


namespace even_and_nonneg_range_l500_500354

theorem even_and_nonneg_range : 
  (∀ x : ℝ, abs x = abs (-x) ∧ (abs x ≥ 0)) ∧ (∀ x : ℝ, x^2 + abs x = ( (-x)^2) + abs (-x) ∧ (x^2 + abs x ≥ 0)) := sorry

end even_and_nonneg_range_l500_500354


namespace convex_quad_cos_floor_l500_500173

theorem convex_quad_cos_floor 
  (A B C D : ℝ) 
  (is_convex : convex_quadrilateral A B C D)
  (angle_A_lt_angle_C : A < C)
  (ab_cd_eq_200 : AB = 200 ∧ CD = 200)
  (ad_ne_bc : AD ≠ BC)
  (perimeter_720 : AB + CD + AD + BC = 720) : 
  ∀ (cos_A : ℝ), floor (1000 * cos_A) = 625 := 
sorry

end convex_quad_cos_floor_l500_500173


namespace centroid_calculation_correct_l500_500341

-- Define the vertices of the triangle
def P : ℝ × ℝ := (2, 3)
def Q : ℝ × ℝ := (-1, 4)
def R : ℝ × ℝ := (4, -2)

-- Define the coordinates of the centroid
noncomputable def S : ℝ × ℝ := ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

-- Prove that 7x + 2y = 15 for the centroid
theorem centroid_calculation_correct : 7 * S.1 + 2 * S.2 = 15 :=
by 
  -- Placeholder for the proof steps
  sorry

end centroid_calculation_correct_l500_500341


namespace polynomials_degrees_even_l500_500577

noncomputable def degree_even (P Q R : Real → Real) :=
  (degree P).land 1 = 0 ∧ (degree Q).land 1 = 0 ∧ (degree R).land 1 = 0

theorem polynomials_degrees_even 
  (P Q R : Polynomial ℝ) 
  (h₁ : P + Q + R = 0)
  (h₂ : P.eval ≫ Q + Q.eval ≫ R + R.eval ≫ P = 0) :
  degree_even P Q R :=
  sorry

end polynomials_degrees_even_l500_500577


namespace letters_with_both_l500_500963

/-
In a certain alphabet, some letters contain a dot and a straight line. 
36 letters contain a straight line but do not contain a dot. 
The alphabet has 60 letters, all of which contain either a dot or a straight line or both. 
There are 4 letters that contain a dot but do not contain a straight line. 
-/
def L_no_D : ℕ := 36
def D_no_L : ℕ := 4
def total_letters : ℕ := 60

theorem letters_with_both (DL : ℕ) : 
  total_letters = D_no_L + L_no_D + DL → 
  DL = 20 :=
by
  intros h
  sorry

end letters_with_both_l500_500963


namespace constant_term_expansion_l500_500182

theorem constant_term_expansion :
  let expr := (x - 4 + 4 / x)^3 in
  ∃ (c : ℤ), ∀ (x : ℝ), x ≠ 0 → (expr = c) ∧ (c = -160) := 
begin
  sorry
end

end constant_term_expansion_l500_500182


namespace inverse_30_deg_right_triangle_l500_500285

theorem inverse_30_deg_right_triangle (T : Type) [EuclideanTriangle T] (Δ : T) :
  (Δ.is_right ∧ ∃ hypotenuse leg : ℝ, hypotenuse > 0 ∧ Δ.hypotenuse = hypotenuse ∧ Δ.leg = leg ∧ leg = hypotenuse / 2) →
  (∃ θ : ℝ, Δ.angle_opposite_leg θ ∧ θ = 30) :=
by
  sorry

end inverse_30_deg_right_triangle_l500_500285


namespace tan_sum_example_l500_500455

theorem tan_sum_example :
  let t1 := Real.tan (17 * Real.pi / 180)
  let t2 := Real.tan (43 * Real.pi / 180)
  t1 + t2 + Real.sqrt 3 * t1 * t2 = Real.sqrt 3 := sorry

end tan_sum_example_l500_500455


namespace solution_set_for_a_l500_500829

theorem solution_set_for_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (a^2 - 1) ^ x > (a^2 - 1) ^ y) →
  (-sqrt 2 < a ∧ a < -1) ∨ (1 < a ∧ a < sqrt 2) :=
by
  sorry

end solution_set_for_a_l500_500829


namespace table_height_l500_500344

-- Define the variables
variables (h l w : ℝ) (r s : ℝ)

-- Define the conditions
def condition1 := l + h - w = r
def condition2 := w + h - l = s

-- Prove the height of the table
theorem table_height (h l w : ℝ) (r s : ℝ) (h1 : r = 40) (h2 : s = 34) : 
  l + h - w = r → w + h - l = s → h = 37 :=
by
  intro h1 h2 h3 h4
  have h5 : l + h - w + w + h - l = r + s,
  { linarith },
  rw [h1, h2] at h5,
  linarith

end table_height_l500_500344


namespace area_of_triangle_l500_500246

variables (r_c : ℝ) (α β γ : ℝ)
variables (S : ℝ) (r : ℝ) (p : ℝ)

def tan (x : ℝ) : ℝ := Math.tan x
def cot (x : ℝ) : ℝ := 1 / (Math.tan x)

axiom h_r : r = r_c * tan (α / 2) * tan (β / 2)
axiom h_p : p = r_c * cot (γ / 2)

theorem area_of_triangle :
  S = r_c^2 * tan (α / 2) * tan (β / 2) * cot (γ / 2) :=
by 
  have h_S : S = r * p := sorry,
  rw [h_r, h_p] at h_S,
  sorry

end area_of_triangle_l500_500246


namespace parabola_focus_l500_500118

theorem parabola_focus (p : ℝ) (h : 0 < p)
  (Hpoint : ∃ (directrix_pt : ℝ × ℝ), directrix_pt = (-1, 1)) :
  let focus := (p / 2, 0) in focus = (1, 0) :=
sorry

end parabola_focus_l500_500118


namespace triangle_AC_length_l500_500187

theorem triangle_AC_length (ABC : Triangle) (h_obtuse : ABC.is_obtuse)
  (h_area : ABC.area = 1/2) (h_AB : ABC.side_length AB = 1) 
  (h_BC : ABC.side_length BC = sqrt(2)) : ABC.side_length AC = sqrt(5) :=
sorry

end triangle_AC_length_l500_500187


namespace median_length_AM_cases_l500_500020

-- Given triangle ABC with altitude AD and angle bisector AE
structure Triangle :=
  (A B C D E : Point)
  (AD AE AM : Real)
  (_ : AD = 12)
  (_ : AE = 13)
  (M : Point)
  (D : D between A and B, C, AD perpendicular to BC)
  (M : M is the midpoint of B, C)
  (E : E on BC such that AE bisects ∠BAC)

-- Solutions to the length of the median AM where angle A can be acute, obtuse or a right angle
theorem median_length_AM_cases (T : Triangle) :
  ∀ (angle_A : ℝ),
  (acute ∠A → ∃ AM_possible : ℝ, AM = AM_possible) ∧
  (obtuse ∠A → ∃ AM_possible : ℝ, AM = AM_possible) ∧
  (right_angle ∠A → ∃ AM_possible : ℝ, AM = AM_possible) :=
sorry

end median_length_AM_cases_l500_500020


namespace side_length_of_square_problem_l500_500254

noncomputable def side_length_of_square_on_hypotenuse_of_right_triangle (PQ PR : ℝ) (h₁ : PQ = 5) (h₂ : PR = 12) (right_angle_at_P : ∀ (A B C: ℝ), A^2 + B^2 = C^2) : ℝ := 
  have hypotenuse : ℝ := (PQ^2 + PR^2).sqrt
  have square_side : ℝ := (480.525 / 101.925)
  square_side

theorem side_length_of_square_problem:
  ∀ (PQ PR : ℝ), PQ = 5 → PR = 12 → (∀ (A B C: ℝ), A^2 + B^2 = C^2) → 
  side_length_of_square_on_hypotenuse_of_right_triangle PQ PR 5 12 (λ A B C, A^2 + B^2 = C^2) = 480.525 / 101.925 := 
sorry

end side_length_of_square_problem_l500_500254


namespace arrange_abc_l500_500794

open Real

noncomputable def ln_x (x : ℝ) (h : x ∈ Ioo (exp (-1 : ℝ)) 1) : ℝ :=
  log x

theorem arrange_abc (x : ℝ) (hx : x ∈ Ioo (exp (-1 : ℝ)) 1) :
  let a := ln_x x hx
  let b := 2 * (ln_x x hx)
  let c := (ln_x x hx) ^ 3
  b < a ∧ a < c := 
sorry

end arrange_abc_l500_500794


namespace percent_decrease_is_20_l500_500989

-- Defining the original price and the sale price
def original_price : ℝ := 100
def sale_price : ℝ := 80

-- Defining the percent decrease according to the conditions
def percent_decrease : ℝ := ((original_price - sale_price) / original_price) * 100

-- Stating the theorem to prove that the percent decrease is 20%
theorem percent_decrease_is_20 : percent_decrease = 20 := by
  sorry

end percent_decrease_is_20_l500_500989


namespace max_min_f_in_rectangle_l500_500079

def f (x y : ℝ) : ℝ := x^3 + y^3 + 6 * x * y

def in_rectangle (x y : ℝ) : Prop := 
  (-3 ≤ x ∧ x ≤ 1) ∧ (-3 ≤ y ∧ y ≤ 2)

theorem max_min_f_in_rectangle :
  ∃ (x_max y_max x_min y_min : ℝ),
    in_rectangle x_max y_max ∧ in_rectangle x_min y_min ∧
    (∀ x y, in_rectangle x y → f x y ≤ f x_max y_max) ∧
    (∀ x y, in_rectangle x y → f x_min y_min ≤ f x y) ∧
    f x_max y_max = 21 ∧ f x_min y_min = -55 :=
by
  sorry

end max_min_f_in_rectangle_l500_500079


namespace calculate_square_difference_l500_500853

theorem calculate_square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 :=
by
  sorry

end calculate_square_difference_l500_500853


namespace eccentricity_of_hyperbola_l500_500724

variables {a b c : ℝ} (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_pos : c > 0)
definition hyperbola : set (ℝ × ℝ) := { p | ∃ (x y : ℝ), (x, y) = p ∧ x^2 / a^2 - y^2 / b^2 = 1 }
definition circle : set (ℝ × ℝ) := { p | ∃ (x y : ℝ), (x, y) = p ∧ x^2 + y^2 = a^2 / 4 }
def F := (-c, 0)
def tangent_point : ℝ × ℝ -- Assume point E is defined.
def E := tangent_point
def P : ℝ × ℝ -- Assume point P is defined.
def OP : ℝ × ℝ -- vector from origin to P
def OE : ℝ × ℝ -- vector from origin to E
def OF : ℝ × ℝ := F -- vector from origin to F

def relation (OP OE OF : ℝ × ℝ) : Prop := OP = 2 * OE - OF

noncomputable def eccentricity (a b c : ℝ) := c / a

theorem eccentricity_of_hyperbola
  (h1 : E ∈ circle)
  (h2 : relation OP OE OF)
  (h3 : P ∈ hyperbola) : 
  eccentricity a b c = sqrt 10 / 2 := sorry

end eccentricity_of_hyperbola_l500_500724


namespace avg_height_first_30_girls_l500_500276

theorem avg_height_first_30_girls (H : ℝ)
  (h1 : ∀ x : ℝ, 30 * x + 10 * 156 = 40 * 159) :
  H = 160 :=
by sorry

end avg_height_first_30_girls_l500_500276


namespace division_by_fraction_example_problem_l500_500033

theorem division_by_fraction (a b : ℝ) (hb : b ≠ 0) : 
  a / (1 / b) = a * b :=
by
  -- Proof goes here
  sorry

theorem example_problem : 12 / (1 / 6) = 72 :=
by
  have h : 6 ≠ 0 := by norm_num
  rw division_by_fraction 12 6 h
  norm_num

end division_by_fraction_example_problem_l500_500033


namespace find_S_when_R_18_T_2_l500_500670

-- Define the conditions
variable (R S T k : ℝ)

-- State the main relationship
axiom R_eq_k_S_sq_over_T (h : R = k * (S^2) / T)

-- Given initial conditions to find k
axiom initial_condition (R : ℝ) (S : ℝ) (T : ℝ) : R = 2 ∧ S = 1 ∧ T = 8

-- The final theorem we want to prove: finding S when R = 18 and T = 2
theorem find_S_when_R_18_T_2 (h1 : R = k * (S^2) / T)
                            (h2 : ∃ k, R = 2 ∧ (S = 1 ∧ T = 8))
                            (h3 : R = 18 ∧ T = 2) :
  S = 1.5 := 
by
sorry

end find_S_when_R_18_T_2_l500_500670


namespace clive_change_l500_500040

theorem clive_change (total_money : ℝ) (num_olives_needed : ℕ) (olives_per_jar : ℕ) (cost_per_jar : ℝ)
  (h1 : total_money = 10)
  (h2 : num_olives_needed = 80)
  (h3 : olives_per_jar = 20)
  (h4 : cost_per_jar = 1.5) : total_money - (num_olives_needed / olives_per_jar) * cost_per_jar = 4 := by
  sorry

end clive_change_l500_500040


namespace simplify_complex_fraction_l500_500257

theorem simplify_complex_fraction : 
  (3 + 3 * Complex.i) / (-4 + 5 * Complex.i) = (3 / 41: ℂ) - (27 / 41) * Complex.i :=
  sorry

end simplify_complex_fraction_l500_500257


namespace find_m_plus_b_l500_500456

theorem find_m_plus_b : 
  let m := (4 - (-1)) / (-1 - 2), 
      b := (-1 - m * 2) 
  in m + b = 2 / 3 := by
  sorry

end find_m_plus_b_l500_500456


namespace patty_weighs_more_l500_500590

variable (R : ℝ) (P_0 : ℝ) (L : ℝ) (P : ℝ) (D : ℝ)

theorem patty_weighs_more :
  (R = 100) →
  (P_0 = 4.5 * R) →
  (L = 235) →
  (P = P_0 - L) →
  (D = P - R) →
  D = 115 := by
  sorry

end patty_weighs_more_l500_500590


namespace max_neg_integers_l500_500357

theorem max_neg_integers (a b c d e f : ℤ) (h : a * b + c * d * e * f < 0) : 
  ∃ A B, list.map int.sign [a, b, c, d, e, f] = A ++ B ∧ length A = 4 ∧ 
  (∀ x ∈ A, x = -1) ∧ (∀ y ∈ B, y = 1) :=
sorry

end max_neg_integers_l500_500357


namespace A_and_D_independent_l500_500322

-- Definitions of the events based on given conditions
def event_A (x₁ : ℕ) : Prop := x₁ = 1
def event_B (x₂ : ℕ) : Prop := x₂ = 2
def event_C (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 8
def event_D (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 7

-- Probabilities based on uniform distribution and replacement
def probability_event (event : ℕ → ℕ → Prop) : ℚ :=
  if h : ∃ x₁ : ℕ, ∃ x₂ : ℕ, x₁ ∈ finset.range 1 7 ∧ x₂ ∈ finset.range 1 7 ∧ event x₁ x₂
  then ((finset.card (finset.filter (λ x, event x.1 x.2)
                (finset.product (finset.range 1 7) (finset.range 1 7)))) : ℚ) / 36
  else 0

noncomputable def P_A : ℚ := 1 / 6
noncomputable def P_D : ℚ := 1 / 6
noncomputable def P_A_and_D : ℚ := 1 / 36

-- Independence condition (by definition): P(A ∩ D) = P(A) * P(D)
theorem A_and_D_independent :
  P_A_and_D = P_A * P_D := by
  sorry

end A_and_D_independent_l500_500322


namespace validValuesForN_l500_500603

noncomputable def possibleValuesOfN (K : Cone) (G : Sphere) (n : ℕ) (r : ℝ) : Prop :=
  let S := fin n → Sphere
  ∀ (g : S),
    -- The centers of the spheres form a regular n-gon with side length 2r.
    (∀ i j, i ≠ j → dist (g i).center (g j).center = 2 * r) ∧
    -- Each sphere touches the lateral surface of the cone, the base of the cone, and the inscribed sphere.
    (∀ i, touches (g i) K.lateralSurface ∧ touches (g i) K.base ∧ touches (g i) G)

theorem validValuesForN (K : Cone) (G : Sphere) (r : ℝ) :
  ∃ n, possibleValuesOfN K G n r ∧ n ∈ {7, 8, 9, 10, 11, 12, 13, 14, 15} := 
sorry

end validValuesForN_l500_500603


namespace part1_part2_l500_500474

variable (a S : ℕ → ℝ)
variable (n : ℕ)
variable (h1 : a 1 = 1)
variable (h_parallel : ∀ n, (S (n + 1) - 2 * S n, S n) = (2, n) → n * (S (n + 1) - 2 * S n) = 2 * S n)

theorem part1 : (∃ r, ∀ n, S n / n = r^n) :=
  sorry

theorem part2 : (∀ n, S n = n * 2^(n-1)) → (∃ T, T = (n - 1) * 2^n + 1) :=
  sorry

end part1_part2_l500_500474


namespace larks_combinations_l500_500992

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0
def in_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 30

theorem larks_combinations :
  (∑ x in (Finset.range 31).filter (λ n, is_odd n), 1) *
  (∑ y in (Finset.range 31).filter (λ n, is_even n), 1) *
  (∑ z in (Finset.range 31).filter (λ n, is_multiple_of_3 n), 1) = 2250 := by
  sorry

end larks_combinations_l500_500992


namespace sum_of_squares_increase_l500_500795

theorem sum_of_squares_increase (a : Fin 100 → ℝ)
  (h : ∑ i, (a i + 1)^2 = ∑ i, (a i)^2) :
  ∑ i, (a i + 2)^2 = ∑ i, (a i)^2 + 200 := 
sorry

end sum_of_squares_increase_l500_500795


namespace sum_of_prime_f_values_l500_500459

def f (n : ℕ) : ℕ := n ^ 4 - 400 * n ^ 2 + 441

theorem sum_of_prime_f_values : ∑ p in {f n | n ∈ ℕ \ {0} ∧ prime (f n)}, p = 397 := by
  sorry

end sum_of_prime_f_values_l500_500459


namespace estimate_birds_in_forest_on_april_10_l500_500386

theorem estimate_birds_in_forest_on_april_10 (
  tagged_april_birds : ℕ := 120,
  august_sample : ℕ := 150,
  tagged_august_sample : ℕ := 4,
  percent_not_present_in_august : ℕ := 30,
  percent_new_arrivals_in_august : ℕ := 50
) : 
  ∃ (total_april_birds : ℕ),
  let tagged_still_present := (tagged_april_birds * (100 - percent_not_present_in_august)) / 100,
      birds_present_in_april := august_sample * (100 - percent_new_arrivals_in_august) / 100 in
  (tagged_august_sample : ℚ) / birds_present_in_april = (tagged_still_present : ℚ) / total_april_birds ∧
  total_april_birds = 1575 :=
by
  let tagged_still_present := (tagged_april_birds * (100 - percent_not_present_in_august)) / 100;
  let birds_present_in_april := august_sample * (100 - percent_new_arrivals_in_august) / 100;
  existsi 1575;
  sorry

end estimate_birds_in_forest_on_april_10_l500_500386


namespace tan_alpha_solution_l500_500869

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l500_500869


namespace eqdistant_points_perpendicular_bisector_l500_500848

open Complex

-- Define the conditions
variable (z z1 z2 : ℂ)
variable (h_distinct : z1 ≠ z2)
variable (h_condition : |z - z1| - |z - z2| = 0)

-- State the theorem
theorem eqdistant_points_perpendicular_bisector (h : |z - z1| = |z - z2|) :
    -- Define what it means for a point to be on the perpendicular bisector
    let mid_point := (z1 + z2) / 2
    let is_perpendicular_bisector := ∀ z, |z - z1| = |z - z2| → ↑(Re z) = ↑(Re mid_point)
    is_perpendicular_bisector :=
sorry

end eqdistant_points_perpendicular_bisector_l500_500848


namespace Intersection_Distance_Sum_l500_500176

open Real

-- Define the parametric equations for curve C1
def curve_C1_x (t : ℝ) : ℝ := 2 * sqrt 2 - (sqrt 2 / 2) * t
def curve_C1_y (t : ℝ) : ℝ := sqrt 2 + (sqrt 2 / 2) * t

-- Define the Cartesian equation for curve C2 obtained from polar coordinates
def curve_C2 (x y : ℝ) : Prop := x^2 + (y - 2 * sqrt 2)^2 = 8

-- Define the coordinates of point P
def point_P := (sqrt 2, 2 * sqrt 2)

-- The key theorem to be proven: |PA| + |PB| = 2 * sqrt 7
theorem Intersection_Distance_Sum : 
  ∃ (A B : ℝ × ℝ), A ∈ curve_C2 ∧ B ∈ curve_C2 ∧ ∃ t1 t2 : ℝ, 
  A = (curve_C1_x t1, curve_C1_y t1) ∧ B = (curve_C1_x t2, curve_C1_y t2) ∧
  @Euclidean_dist _ _ ⟨sqrt 2, 2 * sqrt 2⟩ A + @Euclidean_dist _ _ ⟨sqrt 2, 2 * sqrt 2⟩ B = 2 * sqrt 7 := 
sorry

end Intersection_Distance_Sum_l500_500176


namespace residue_of_927_mod_37_l500_500056

-- Define the condition of the problem, which is the modulus and the number
def modulus : ℤ := 37
def number : ℤ := -927

-- Define the statement we need to prove: that the residue of -927 mod 37 is 35
theorem residue_of_927_mod_37 : (number % modulus + modulus) % modulus = 35 := by
  sorry

end residue_of_927_mod_37_l500_500056


namespace good_numbers_l500_500161

theorem good_numbers (n : ℕ) (hn : n ≥ 33) : 
  (∃ k (a : ℕ → ℕ), (0 < k) ∧ (n = ∑ i in finset.range k, a i) ∧ (∑ i in finset.range k, (1 : ℚ) / a i = 1)) :=
  sorry

-- Supplemental definition for the good numbers from 33 to 73:
def known_good_numbers : set ℕ := {n | 33 ≤ n ∧ n ≤ 73}

axiom good_numbers_33_to_73 : ∀ n, n ∈ known_good_numbers →
  (∃ k (a : ℕ → ℕ), (0 < k) ∧ (n = ∑ i in finset.range k, a i) ∧ (∑ i in finset.range k, (1 : ℚ) / a i = 1))

end good_numbers_l500_500161


namespace product_of_squares_l500_500084

theorem product_of_squares (x : ℝ) (h : |5 * x| + 4 = 49) : x^2 * (if x = 9 then 9 else -9)^2 = 6561 :=
by
  sorry

end product_of_squares_l500_500084


namespace tan_alpha_value_l500_500939

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l500_500939


namespace sum_of_all_lucky_numbers_divisible_by_13_l500_500227

def is_lucky (n : ℕ) : Prop :=
  let d₁ := n / 100000 % 10
  let d₂ := n / 10000 % 10
  let d₃ := n / 1000 % 10
  let d₄ := n / 100 % 10
  let d₅ := n / 10 % 10
  let d₆ := n % 10
  d₁ + d₂ + d₃ = d₄ + d₅ + d₆

theorem sum_of_all_lucky_numbers_divisible_by_13 :
  let lucky_numbers := {n : ℕ | 100000 ≤ n ∧ n < 1000000 ∧ is_lucky n}
  (∑ n in lucky_numbers.to_finset, n) % 13 = 0 :=
sorry

end sum_of_all_lucky_numbers_divisible_by_13_l500_500227


namespace max_unique_planes_from_15_points_l500_500978

-- Defining the problem conditions
def fifteen_points : Set (Array ℝ) := sorry -- A set representing the 15 points in 3D space

-- Assumption: No four points lie on the same plane
axiom no_four_points_on_same_plane (points : Set (Array ℝ)) : 
  ∀ (p1 p2 p3 p4 : Array ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points → p4 ∈ points → 
  ¬ (affine_span ℝ {p1, p2, p3, p4} = affine_span ℝ {p1, p2, p3})

-- Assumption: Not all points are collinear
axiom not_all_collinear (points : Set (Array ℝ)) : 
  ∃ (p1 p2 p3 : Array ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
  ¬ (affine_span ℝ {p1, p2} = affine_span ℝ {p1, p2, p3})

-- The theorem to prove
theorem max_unique_planes_from_15_points : 
  ∀ (points : Set (Array ℝ)), 
  no_four_points_on_same_plane points → 
  not_all_collinear points → 
  points.card = 15 → 
  card {plane | ∃ (p1 p2 p3 : Array ℝ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ plane = affine_span ℝ {p1, p2, p3}} = 455 :=
by
  sorry

end max_unique_planes_from_15_points_l500_500978


namespace tan_alpha_fraction_l500_500908

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l500_500908


namespace lambda_range_l500_500508

theorem lambda_range (λ : ℝ) (a b : ℕ → ℝ) (hA1 : a 1 = 1)
  (hAn : ∀ n : ℕ, a (n + 1) = a n / (a n + 2))
  (hB1 : b 1 = -λ)
  (hBn : ∀ n : ℕ, b (n + 1) = (n - λ) * (1 / a n + 1))
  (monotonic_b : ∀ n : ℕ, b n ≤ b (n + 1)) :
  λ < 2 :=
  sorry

end lambda_range_l500_500508


namespace fraction_addition_l500_500408

variable (a : ℝ)

theorem fraction_addition (ha : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by
  sorry

end fraction_addition_l500_500408


namespace stabilization_of_O_l500_500265

noncomputable def centroid (ps: List (ℝ × ℝ)) : ℝ × ℝ :=
  let (sum_x, sum_y) := ps.foldl (λ (acc: ℝ × ℝ) (p: ℝ × ℝ), (acc.1 + p.1, acc.2 + p.2)) (0, 0)
  (sum_x / ps.length, sum_y / ps.length)

theorem stabilization_of_O 
  (n : ℕ) 
  (points : Fin n → ℝ × ℝ)
  (r : ℝ) 
  (O : ℝ × ℝ)
  (h_at_least_one_in_circle : ∃ i, (points i).fst^2 + (points i).snd^2 ≤ r^2) 
  : ∃ O_fixed : ℝ × ℝ, ∃ t : ℕ, ∀ t' : ℕ, t' ≥ t → O = O_fixed :=
by
  have O_next : ℝ × ℝ := centroid (List.filter (λ p, (p.fst)^2 + (p.snd)^2 ≤ r^2) (List.ofFin points))
  sorry

end stabilization_of_O_l500_500265


namespace scientific_notation_l500_500260

theorem scientific_notation : 899000 = 8.99 * 10^5 := 
by {
  -- We start by recognizing that we need to express 899,000 in scientific notation.
  -- Placing the decimal point after the first non-zero digit yields 8.99.
  -- Count the number of places moved (5 places to the left).
  -- Thus, 899,000 in scientific notation is 8.99 * 10^5.
  sorry
}

end scientific_notation_l500_500260


namespace range_of_a_l500_500572

-- Definitions of given conditions
def f (x : ℝ) : ℝ := sorry -- Placeholder for the function f
def D := set.Ici (0 : ℝ) -- Domain D = [0, +∞)
def A : set ℝ := sorry -- Range A of f

-- Condition 1: Function f satisfies f(x) = f(1 / (x + 1))
axiom h_f_eq : ∀ x : ℝ, f (x) = f (1 / (x + 1))

-- Condition 4: The set {y | y = f(x), x ∈ [0, a]}
def f_set (a : ℝ) := {y | ∃ x ∈ set.Icc (0 : ℝ) a, y = f x}

-- The correct answer to prove:
theorem range_of_a : ∀ a : ℝ, (∀ y ∈ A, ∃ x ∈ set.Icc (0 : ℝ) a, y = f x)
    ↔ a ∈ set.Ici ((√5 - 1) / 2) := sorry

end range_of_a_l500_500572


namespace limit_solution_l500_500743

noncomputable def limit_problem : Prop :=
  (∀ f : ℝ → ℝ, (∀ x, f x = (ln (4 * x - 1)) / (sqrt (1 - cos (Real.pi * x)) - 1)) → 
  filter.tendsto f (nhds (1 / 2)) (nhds (8 / Real.pi)))

theorem limit_solution : limit_problem :=
  begin
    sorry
  end

end limit_solution_l500_500743


namespace seohee_routes_l500_500532

variable (num_bus_routes : ℕ) (num_subway_routes : ℕ)

theorem seohee_routes (h1 : num_bus_routes = 3) (h2 : num_subway_routes = 2) : num_bus_routes + num_subway_routes = 5 :=
by
  rw [h1, h2]
  rfl

end seohee_routes_l500_500532


namespace average_age_of_coaches_l500_500275

theorem average_age_of_coaches 
  (total_members : ℕ) (average_age_members : ℕ)
  (num_girls : ℕ) (average_age_girls : ℕ)
  (num_boys : ℕ) (average_age_boys : ℕ)
  (num_coaches : ℕ) :
  total_members = 30 →
  average_age_members = 20 →
  num_girls = 10 →
  average_age_girls = 18 →
  num_boys = 15 →
  average_age_boys = 19 →
  num_coaches = 5 →
  (600 - (num_girls * average_age_girls) - (num_boys * average_age_boys)) / num_coaches = 27 :=
by
  intros
  sorry

end average_age_of_coaches_l500_500275


namespace candy_bars_eaten_l500_500307

theorem candy_bars_eaten (calories_per_candy : ℕ) (total_calories : ℕ) (h1 : calories_per_candy = 31) (h2 : total_calories = 341) :
  total_calories / calories_per_candy = 11 :=
by
  sorry

end candy_bars_eaten_l500_500307


namespace range_of_a_l500_500128

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  real.log x + (1 / 2) * x^2 + a * x

def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  1 / x + x + a

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f_prime x1 a = 0 ∧ f_prime x2 a = 0 ∧ x1 > 0 ∧ x2 > 0) →
  a < -2 :=
by
  sorry

end range_of_a_l500_500128


namespace fraction_exponent_simplification_l500_500281

theorem fraction_exponent_simplification :
  (7^((1 : ℝ) / 4)) / (7^((1 : ℝ) / 6)) = 7^((1 : ℝ) / 12) :=
sorry

end fraction_exponent_simplification_l500_500281


namespace intersection_eq_l500_500845

-- Define the sets M and N using the given conditions
def M : Set ℝ := { x | x < 1 / 2 }
def N : Set ℝ := { x | x ≥ -4 }

-- The goal is to prove that the intersection of M and N is { x | -4 ≤ x < 1 / 2 }
theorem intersection_eq : M ∩ N = { x | -4 ≤ x ∧ x < (1 / 2) } :=
by
  sorry

end intersection_eq_l500_500845


namespace proof_problem_l500_500418

noncomputable def is_rational (r : ℝ) : Prop := ∃ q : ℚ, r = q

theorem proof_problem (x : ℝ) :
  is_rational (x + sqrt (x^2 + 4*x) - 1 / (x + sqrt (x^2 + 4*x))) ↔ is_rational (x + sqrt (x^2 + 4*x)) := by
  sorry

end proof_problem_l500_500418


namespace train_people_count_l500_500403

theorem train_people_count :
  let initial := 332
  let first_station_on := 119
  let first_station_off := 113
  let second_station_off := 95
  let second_station_on := 86
  initial + first_station_on - first_station_off - second_station_off + second_station_on = 329 := 
by
  sorry

end train_people_count_l500_500403


namespace pencil_price_units_l500_500158

noncomputable def price_pencil (base_price: ℕ) (extra_cost: ℕ): ℝ :=
  (base_price + extra_cost) / 10000.0

theorem pencil_price_units (base_price: ℕ) (extra_cost: ℕ) (h_base: base_price = 5000) (h_extra: extra_cost = 20) : 
  price_pencil base_price extra_cost = 0.5 := by
  sorry

end pencil_price_units_l500_500158


namespace total_distance_l500_500709

variable (D : ℝ) -- Define the total distance
variable (speed_walk : ℝ := 4) -- speed of walking
variable (speed_run : ℝ := 8) -- speed of running
variable (total_time : ℝ := 1.5) -- total time for covering the distance

theorem total_distance : (D / 2) / speed_walk + (D / 2) / speed_run = total_time → D = 4 := by
  intros h
  simp at h
  sorry

end total_distance_l500_500709


namespace A_and_D_mutual_independent_l500_500314

-- Probability theory definitions and assumptions.
noncomputable def prob_1_6 : ℚ := 1 / 6
noncomputable def prob_5_36 : ℚ := 5 / 36
noncomputable def prob_6_36 : ℚ := 6 / 36
noncomputable def prob_1_36 : ℚ := 1 / 36

-- Definitions of events with their corresponding probabilities.
def event_A (P : ℚ) : Prop := P = prob_1_6
def event_B (P : ℚ) : Prop := P = prob_1_6
def event_C (P : ℚ) : Prop := P = prob_5_36
def event_D (P : ℚ) : Prop := P = prob_6_36

-- Intersection probabilities:
def intersection_A_C (P : ℚ) : Prop := P = 0
def intersection_A_D (P : ℚ) : Prop := P = prob_1_36
def intersection_B_C (P : ℚ) : Prop := P = prob_1_36
def intersection_C_D (P : ℚ) : Prop := P = 0

-- Mutual independence definition.
def mutual_independent (P_X : ℚ) (P_Y : ℚ) (P_intersect : ℚ) : Prop :=
  P_X * P_Y = P_intersect

-- Theorem to prove:
theorem A_and_D_mutual_independent :
  event_A prob_1_6 →
  event_D prob_6_36 →
  intersection_A_D prob_1_36 →
  mutual_independent prob_1_6 prob_6_36 prob_1_36 := 
by 
  intros hA hD hAD
  rw [event_A, event_D, intersection_A_D] at hA hD hAD
  exact hA.symm ▸ hD.symm ▸ hAD.symm 

#check A_and_D_mutual_independent

end A_and_D_mutual_independent_l500_500314


namespace abs_diff_eq_1point5_l500_500206

theorem abs_diff_eq_1point5 (x y : ℝ)
    (hx : (⌊x⌋ : ℝ) + (y - ⌊y⌋) = 3.7)
    (hy : (x - ⌊x⌋) + (⌊y⌋ : ℝ) = 4.2) :
        |x - y| = 1.5 :=
by
  sorry

end abs_diff_eq_1point5_l500_500206


namespace intersect_values_parallel_values_perpendicular_values_l500_500512

def l1 (x y m : ℝ) : Prop := x + (1 + m) * y = 2 - m

def l2 (x y m : ℝ) : Prop := 2 * m * x + 4 * y = -16

theorem intersect_values (m : ℝ) : (∃ x y : ℝ, l1 x y m ∧ l2 x y m) ↔ (m ≠ -2 ∧ m ≠ 1) :=
sorry

theorem parallel_values (m : ℝ) : (∀ x1 x2 y1 y2 : ℝ, l1 x1 y1 m ∧ l2 x2 y2 m → 2 * m * (1 + m) - 4 = 0) ↔ (m = 1) :=
sorry

theorem perpendicular_values (m : ℝ) : (∀ x y : ℝ, l1 x y m ∧ (-1/(1 + m)) = (-2 * m / 4) → -1) ↔ (m = -2/3) :=
sorry

end intersect_values_parallel_values_perpendicular_values_l500_500512


namespace find_moles_CH3Cl_l500_500451

def moles_CH3Cl_formed (n_CH4 n_Cl2 n_CH3Cl : ℕ) : Prop :=
  n_CH4 = 1 ∧ n_Cl2 = 1 ∧ n_CH3Cl = 1

theorem find_moles_CH3Cl :
  ∀ (n_CH4 n_Cl2 n_CH3Cl : ℕ), n_CH4 = 1 → n_Cl2 ≥ 1 → n_CH3Cl = 1 → moles_CH3Cl_formed n_CH4 n_Cl2 1 :=
by
  intros n_CH4 n_Cl2 n_CH3Cl h_CH4 h_Cl2 h_CH3Cl
  unfold moles_CH3Cl_formed
  exact ⟨h_CH4, h_Cl2, h_CH3Cl⟩

end find_moles_CH3Cl_l500_500451


namespace find_m_direction_vector_parallel_plane_l500_500814

-- Define the Lean 4 statement
theorem find_m_direction_vector_parallel_plane :
  ∀ (m : ℝ),
    let dir_vector : ℝ × ℝ × ℝ := (2, m, 1)
    let norm_vector : ℝ × ℝ × ℝ := (1, 1/2, 2)
    (2 * 1 + m * (1/2) + 1 * 2 = 0) → m = -8 := 
by
  intros m dir_vector norm_vector
  let dir_vector := (2, m, 1)
  let norm_vector := (1, 1/2, 2)
  simp only [mul_eq_zero]
  sorry

end find_m_direction_vector_parallel_plane_l500_500814


namespace quadratic_equation_reciprocal_integer_roots_l500_500080

noncomputable def quadratic_equation_conditions (a b c : ℝ) : Prop :=
  (∃ r : ℝ, (r * (1/r) = 1) ∧ (r + (1/r) = 4)) ∧ 
  (c = a) ∧ 
  (b = -4 * a)

theorem quadratic_equation_reciprocal_integer_roots (a b c : ℝ) (h1 : quadratic_equation_conditions a b c) : 
  c = a ∧ b = -4 * a :=
by
  obtain ⟨r, hr₁, hr₂⟩ := h1.1
  sorry

end quadratic_equation_reciprocal_integer_roots_l500_500080


namespace seven_circles_radius_l500_500697

theorem seven_circles_radius (r: ℝ): 
  (∀ (C: ℝ), C = 1 → 
    (∃ (CoverCircles: Fin 7 → ℝ), 
      (∀ i, CoverCircles i = r) → 
      (C ≤ ∑ i, CoverCircles i) 
    )
  ) → 
  (r ≥ 1 / 2) :=
by
  intro h
  sorry

end seven_circles_radius_l500_500697


namespace points_cyclic_l500_500178

theorem points_cyclic {A B C B' C' P Q : Type} [Inhabited A] [Inhabited B] [Inhabited C]
  (h_acute : triangle_acute A B C)
  (h_alt_B : is_altitude B B' C)
  (h_alt_C : is_altitude C C' B)
  (h_tangent_P : circle_through_tangent A C' P B C)
  (h_tangent_Q : circle_through_tangent A C' Q B C)
  : cyclic_points A B' P Q :=
by {
  -- proof goes here
  sorry
}

end points_cyclic_l500_500178


namespace binary_to_decimal_101101_l500_500420

theorem binary_to_decimal_101101 : 
  (1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5) = 45 := 
by 
  sorry

end binary_to_decimal_101101_l500_500420


namespace sum_of_exponents_of_powers_of_two_l500_500444

theorem sum_of_exponents_of_powers_of_two (n : ℕ) (h : n = 2345) : 
  ∃ (exps : list ℕ), list.pairwise (≠) exps ∧ (∑ k in exps, 2^k) = n ∧ (∑ k in exps, k) = 27 := 
by {
  sorry
}

end sum_of_exponents_of_powers_of_two_l500_500444


namespace rhombus_locus_l500_500803

-- Define the coordinates of the vertices of the rhombus
structure Point :=
(x : ℝ)
(y : ℝ)

def A (e : ℝ) : Point := ⟨e, 0⟩
def B (f : ℝ) : Point := ⟨0, f⟩
def C (e : ℝ) : Point := ⟨-e, 0⟩
def D (f : ℝ) : Point := ⟨0, -f⟩

-- Define the distance squared from a point P to a point Q
def dist_sq (P Q : Point) : ℝ := (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the geometric locus problem
theorem rhombus_locus (P : Point) (e f : ℝ) :
  dist_sq P (A e) = dist_sq P (B f) + dist_sq P (C e) + dist_sq P (D f) ↔
  (if e > f then
    (dist_sq P (A e) = (e^2 - f^2) ∨ dist_sq P (C e) = (e^2 - f^2))
   else if e = f then
    (P = A e ∨ P = B f ∨ P = C e ∨ P = D f)
   else
    false) :=
sorry

end rhombus_locus_l500_500803


namespace sum_of_abs_coeffs_polynomial_l500_500800

theorem sum_of_abs_coeffs_polynomial (n : ℕ) :
  let P : ℚ[X] := ∑ h in Finset.range (n + 1), (Nat.choose n h) * X^(n - h) * (X - 1)^h
  in (∑ i in Finset.range (n + 1), |(P.coeff i)|) = 3^n :=
by
  sorry

end sum_of_abs_coeffs_polynomial_l500_500800


namespace find_f_pi_div_4_l500_500835

noncomputable def f (x : ℝ) : ℝ :=
  (f' (π / 4) : ℝ) * Real.cos x + Real.sin x

noncomputable def f' (x : ℝ) : ℝ :=
  -(f' (π / 4) : ℝ) * Real.sin x + Real.cos x

theorem find_f_pi_div_4 : f (π / 4) = 1 :=
sorry

end find_f_pi_div_4_l500_500835


namespace trajectory_G_l500_500823

-- Point and Circle definitions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def Circle (c : Point) (r : ℝ) := 
  { p : Point // (p.x - c.x)^2 + (p.y - c.y)^2 = r^2 }

-- Given Definitions
def M : Circle := sorry -- a circle with center (-√7, 0) and radius 8
def N : Point := ⟨√7, 0⟩
def P : Point := sorry -- a moving point on the circle M
def Q : Point := sorry -- a point on the line segment NP such that NP = 2NQ
def G : Point := sorry -- a point on the line segment MP such that GQ ⊥ NP

-- The trajectory equation to prove
theorem trajectory_G :
  let M_center : Point := ⟨-√7, 0⟩
  let M_radius : ℝ := 8
  let M_equation : ∀ p : Point, (p.x + √7)^2 + p.y^2 = 64 := sorry
  let Q_condition : Q.x = N.x + 1/2 * (P.x - N.x) ∧ Q.y = N.y + 1/2 * (P.y - N.y) := sorry
  let G_condition : (G.x * (P.x - N.x) + G.y * (P.y - N.y)) = 0 := sorry
  ∃ G_traj : set Point, 
  (∀ pt : Point, pt ∈ G_traj ↔ (pt.x^2 / 16 + pt.y^2 / 9 = 1)) := 
sorry

end trajectory_G_l500_500823


namespace fixed_point_of_line_l500_500576

theorem fixed_point_of_line (m : ℝ) : 
  (m - 2) * (-3) - 8 + 3 * m + 2 = 0 :=
by
  sorry

end fixed_point_of_line_l500_500576


namespace proof_statements_l500_500668

namespace ProofProblem

-- Definitions for each condition
def is_factor (x y : ℕ) : Prop := ∃ n : ℕ, y = n * x
def is_divisor (x y : ℕ) : Prop := is_factor x y

-- Lean 4 statement for the problem
theorem proof_statements :
  is_factor 4 20 ∧
  (is_divisor 19 209 ∧ ¬ is_divisor 19 63) ∧
  (¬ is_divisor 12 75 ∧ ¬ is_divisor 12 29) ∧
  (is_divisor 11 33 ∧ ¬ is_divisor 11 64) ∧
  is_factor 9 180 :=
by
  sorry

end ProofProblem

end proof_statements_l500_500668


namespace new_arithmetic_sequence_l500_500011

variable {α : Type*} [CommRing α]

def is_arithmetic_seq (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) - a n = d

theorem new_arithmetic_sequence (a : ℕ → α) (d : α)
  (h_arith : is_arithmetic_seq a d) (h_d_ne_zero : d ≠ 0) :
  is_arithmetic_seq (λ n, a n + a (n + 2)) (2 * d) :=
  sorry

end new_arithmetic_sequence_l500_500011


namespace find_a_for_parallel_lines_l500_500529

theorem find_a_for_parallel_lines (a : ℝ) (h : (x - y = 0) ∥ (2 * x + a * y - 1 = 0)) : a = -2 := by
  -- Proof sketch: Since parallel lines have equal slopes, and recognizing the slope 
  -- form from the given line equations, we can isolate a and solve.
  sorry

end find_a_for_parallel_lines_l500_500529


namespace cube_root_neg_frac_l500_500608

theorem cube_root_neg_frac : (-(1/3 : ℝ))^3 = - 1 / 27 := by
  sorry

end cube_root_neg_frac_l500_500608


namespace mandy_cinnamon_amount_correct_l500_500231

def mandy_cinnamon_amount (nutmeg : ℝ) (cinnamon : ℝ) : Prop :=
  cinnamon = nutmeg + 0.17

theorem mandy_cinnamon_amount_correct :
  mandy_cinnamon_amount 0.5 0.67 :=
by
  sorry

end mandy_cinnamon_amount_correct_l500_500231


namespace A_and_D_mutual_independent_l500_500313

-- Probability theory definitions and assumptions.
noncomputable def prob_1_6 : ℚ := 1 / 6
noncomputable def prob_5_36 : ℚ := 5 / 36
noncomputable def prob_6_36 : ℚ := 6 / 36
noncomputable def prob_1_36 : ℚ := 1 / 36

-- Definitions of events with their corresponding probabilities.
def event_A (P : ℚ) : Prop := P = prob_1_6
def event_B (P : ℚ) : Prop := P = prob_1_6
def event_C (P : ℚ) : Prop := P = prob_5_36
def event_D (P : ℚ) : Prop := P = prob_6_36

-- Intersection probabilities:
def intersection_A_C (P : ℚ) : Prop := P = 0
def intersection_A_D (P : ℚ) : Prop := P = prob_1_36
def intersection_B_C (P : ℚ) : Prop := P = prob_1_36
def intersection_C_D (P : ℚ) : Prop := P = 0

-- Mutual independence definition.
def mutual_independent (P_X : ℚ) (P_Y : ℚ) (P_intersect : ℚ) : Prop :=
  P_X * P_Y = P_intersect

-- Theorem to prove:
theorem A_and_D_mutual_independent :
  event_A prob_1_6 →
  event_D prob_6_36 →
  intersection_A_D prob_1_36 →
  mutual_independent prob_1_6 prob_6_36 prob_1_36 := 
by 
  intros hA hD hAD
  rw [event_A, event_D, intersection_A_D] at hA hD hAD
  exact hA.symm ▸ hD.symm ▸ hAD.symm 

#check A_and_D_mutual_independent

end A_and_D_mutual_independent_l500_500313


namespace jason_total_games_l500_500199

theorem jason_total_games :
  let jan_games := 11
  let feb_games := 17
  let mar_games := 16
  let apr_games := 20
  let may_games := 14
  let jun_games := 14
  let jul_games := 14
  jan_games + feb_games + mar_games + apr_games + may_games + jun_games + jul_games = 106 :=
by
  sorry

end jason_total_games_l500_500199


namespace max_min_of_f_in_M_l500_500680

noncomputable def domain (x : ℝ) : Prop := 3 - 4*x + x^2 > 0

def M : Set ℝ := { x | domain x }

noncomputable def f (x : ℝ) : ℝ := 2^(x+2) - 3 * 4^x

theorem max_min_of_f_in_M :
  ∃ (xₘ xₘₐₓ : ℝ), xₘ ∈ M ∧ xₘₐₓ ∈ M ∧ 
  (∀ x ∈ M, f xₘₐₓ ≥ f x) ∧ 
  (∀ x ∈ M, f xₘ ≠ f xₓₐₓ) :=
by
  sorry

end max_min_of_f_in_M_l500_500680


namespace tan_alpha_value_l500_500926

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l500_500926


namespace tan_alpha_solution_l500_500864

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l500_500864


namespace matthew_more_strawberries_than_betty_l500_500030

noncomputable def B : ℕ := 16

theorem matthew_more_strawberries_than_betty (M N : ℕ) 
  (h1 : M > B)
  (h2 : M = 2 * N) 
  (h3 : B + M + N = 70) : M - B = 20 :=
by
  sorry

end matthew_more_strawberries_than_betty_l500_500030


namespace new_shoes_last_for_two_years_l500_500693

def cost_per_year_repair := 13.50
def cost_new_shoes := 32.00
def increase_percentage := 0.1852
def cost_per_year_new_shoes (x : ℝ) := cost_new_shoes / x
def effective_cost := cost_per_year_repair + (increase_percentage * cost_per_year_repair)

theorem new_shoes_last_for_two_years :
  ∀ (x : ℝ), cost_per_year_new_shoes x = effective_cost → x = 2 := 
by {
  intro x,
  intro h,
  sorry
}

end new_shoes_last_for_two_years_l500_500693


namespace awards_distribution_l500_500594

variable (awards students : ℕ)

theorem awards_distribution :
  awards = 6 ∧ students = 4 ∧ 
  ∀ s : (students → ℕ), (∀ i, s i > 0) ∧ (∑ i, s i = awards) 
  → ∃ n : ℕ, n = 1560 :=
by
  intros h
  sorry

end awards_distribution_l500_500594


namespace find_Z_l500_500703

open Classical

variable (n : ℕ)
variable (people : Fin n → Type)
variable [∀ i, Fintype (people i)]
variable knows : (Π i j, Prop) → People n
variable Z : Type → People n

def knows_all (Z : ∀ i, People n) :=
  ∀ i, i ≠ Z → knows Z i

def unknown_by_all (Z : ∀ i, People n) :=
  ∀ i, i ≠ Z → ¬ knows i Z

theorem find_Z (n : ℕ) (knows : Π i j, Prop) (Z : Π i, Z): 
  (∃ Z, knows_all Z ∧ unknown_by_all Z) → 
  (∀ strategy : ℕ → ℕ → bool, 
    (∃ i, i < n - 1,
    ∀ j < i, strategy j n ∧ 
    (∃ k, knows j k → Z = k)) ∧
    ¬ (∃ i, i < n - 2, 
    ∀ j < i, strategy j (n - 2) ∧ 
    (∃ k, knows j k → Z = k)) :=
begin
  sorry
end

end find_Z_l500_500703


namespace intersection_of_lines_l500_500426

-- Define the conditions of the problem
def first_line (x y : ℝ) : Prop := y = -3 * x + 1
def second_line (x y : ℝ) : Prop := y + 1 = 15 * x

-- Prove the intersection point of the two lines
theorem intersection_of_lines : 
  ∃ (x y : ℝ), first_line x y ∧ second_line x y ∧ x = 1 / 9 ∧ y = 2 / 3 :=
by
  sorry

end intersection_of_lines_l500_500426


namespace tan_alpha_value_l500_500933

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l500_500933


namespace find_constants_l500_500446

theorem find_constants (A B C : ℤ) (h1 : 1 = A + B) (h2 : -2 = C) (h3 : 5 = -A) :
  A = -5 ∧ B = 6 ∧ C = -2 :=
by {
  sorry
}

end find_constants_l500_500446


namespace mirror_area_l500_500413

-- Definitions based on the given conditions
def frame_perimeter : ℕ := 160
def frame_width : ℕ := 8

-- Proof problem statement
theorem mirror_area : (let frame_side_length := frame_perimeter / 4 in
                       let mirror_side_length := frame_side_length - 2 * frame_width in
                       mirror_side_length * mirror_side_length = 576) :=
sorry

end mirror_area_l500_500413


namespace percentage_increase_volume_l500_500375

noncomputable def volume_of_torus (R t : ℝ) : ℝ :=
  2 * real.pi^2 * R * t^2

theorem percentage_increase_volume (R t : ℝ) (hR : 0 < R) (ht : 0 < t) :
  let V1 := volume_of_torus R t in
  let V2 := volume_of_torus (1.01 * R) t in
  ((V2 - V1) / V1) * 100 = 1 :=
by
  sorry

end percentage_increase_volume_l500_500375


namespace x_coordinate_of_x_intercept_of_new_line_n_l500_500229

noncomputable def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

noncomputable def tangent (θ : ℝ) : ℝ := Real.tan θ

def line_m (x y : ℝ) : Prop := 4 * x - 3 * y + 24 = 0

def rotation_point : (ℝ × ℝ) := (10, -10)

def rotate_slope (m θ : ℝ) : ℝ :=
  (m + tangent(θ)) / (1 - m * tangent(θ))

def line_n (x y : ℝ) (m' : ℝ) : Prop :=
  y + 10 = m' * (x - 10)

def x_intercept (f : ℝ → ℝ → Prop) (m' : ℝ) : ℝ :=
  let n := f in
  ((10 : ℝ) / m') + 10

theorem x_coordinate_of_x_intercept_of_new_line_n :
  ∃ x : ℝ, x_intercept (line_n x 0) (rotate_slope (4 / 3) (π / 6)) = x :=
sorry

end x_coordinate_of_x_intercept_of_new_line_n_l500_500229


namespace quad_equal_angles_square_l500_500359

theorem quad_equal_angles_square:
  ∀ (Q : Type) [h : quadratic Q], (Q.has_equal_interior_angles) → ¬Q.is_square ∨ Q.is_square :=
sorry

end quad_equal_angles_square_l500_500359


namespace pipe_individual_empty_time_l500_500723

variable (a b c : ℝ)

noncomputable def timeToEmptyFirstPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * c + b * c - a * b)

noncomputable def timeToEmptySecondPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * b + b * c - a * c)

noncomputable def timeToEmptyThirdPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * b + a * c - b * c)

theorem pipe_individual_empty_time
  (x y z : ℝ)
  (h1 : 1 / x + 1 / y = 1 / a)
  (h2 : 1 / x + 1 / z = 1 / b)
  (h3 : 1 / y + 1 / z = 1 / c) :
  x = timeToEmptyFirstPipe a b c ∧ y = timeToEmptySecondPipe a b c ∧ z = timeToEmptyThirdPipe a b c :=
sorry

end pipe_individual_empty_time_l500_500723


namespace tan_alpha_sqrt_15_over_15_l500_500918

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l500_500918


namespace comb_eq_comb_imp_n_eq_18_l500_500095

theorem comb_eq_comb_imp_n_eq_18 {n : ℕ} (h : Nat.choose n 14 = Nat.choose n 4) : n = 18 :=
sorry

end comb_eq_comb_imp_n_eq_18_l500_500095


namespace sum_of_triangle_angles_sin_halves_leq_one_l500_500567

theorem sum_of_triangle_angles_sin_halves_leq_one (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hABC : A + B + C = Real.pi) : 
  8 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 := 
sorry 

end sum_of_triangle_angles_sin_halves_leq_one_l500_500567


namespace minimum_attacking_pairs_l500_500969

noncomputable theory

open_locale big_operators

-- Define the chessboard as an 8x8 grid
def Chessboard := Fin 8 × Fin 8

-- Define rooks' placement on the chessboard
def rooks_placement (placements : Finset Chessboard) := 
  placements.card = 16

-- Define the attacking pairs of rooks
def attacking_pairs (placements : Finset Chessboard) : ℕ :=
  let row_contrib := λ (r : Fin 8), (placements.filter (λ p, p.1 = r)).card - 1
  let col_contrib := λ (c : Fin 8), (placements.filter (λ p, p.2 = c)).card - 1
  ∑ r, row_contrib r + ∑ c, col_contrib c

-- The theorem states the minimum number of attacking pairs of rooks in a valid placement
theorem minimum_attacking_pairs (placements : Finset Chessboard)
  (h : rooks_placement placements) :
  attacking_pairs placements = 16 :=
sorry

end minimum_attacking_pairs_l500_500969


namespace positive_whole_numbers_with_fourth_root_less_than_six_l500_500147

theorem positive_whole_numbers_with_fourth_root_less_than_six :
  {n : ℕ | n > 0 ∧ (n : ℝ)^(1/4) < 6}.to_finset.card = 1295 :=
sorry

end positive_whole_numbers_with_fourth_root_less_than_six_l500_500147


namespace solve_f_x_eq_1_l500_500121

-- Functions definitions as per the conditions in the problem
noncomputable def f_inv (theta : ℝ) (x : ℝ) : ℝ :=
  log (sin^2 theta) (1 / x - cos^2 theta)

theorem solve_f_x_eq_1 (theta : ℝ) (h_theta : 0 < theta ∧ theta < pi / 2) :
  f_inv theta 1 = 1 := 
sorry

end solve_f_x_eq_1_l500_500121


namespace tan_alpha_value_l500_500893

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l500_500893


namespace sum_of_squares_of_real_solutions_l500_500783

theorem sum_of_squares_of_real_solutions :
  (∑ x in { x : ℝ | x^512 = 16^128, x * x }) = 8 :=
by sorry

end sum_of_squares_of_real_solutions_l500_500783


namespace largest_lambda_l500_500801

variable (n : ℕ) (a : ℕ → ℕ)
variable (h_n : 2 ≤ n)
variable (h_a : ∀ m : ℕ, m < n → 0 < a m ∧ a m < a (m + 1))

theorem largest_lambda (n : ℕ) (a : ℕ → ℕ) (h1 : 2 ≤ n) (h2 : ∀ m < n, 0 < a m ∧ a m < a (m + 1)) :
  ∃ λ, λ = (2 * n - 4) / (n - 1) ∧ ∀ n, a (n-1) ^ 2 ≥ λ * ∑ i in finset.range (n-1), a i + 2 * a (n-1) := 
sorry

end largest_lambda_l500_500801


namespace simplified_expression_evaluation_l500_500258

theorem simplified_expression_evaluation (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ -2) (h3 : x ≠ 2) :
  (let e := (1 / (x + 3) - 1) / ((x^2 - 4) / (x^2 + 6 * x + 9))
  in e = (x + 3) / (2 - x)) ∧
  ((x = 0) → ((x + 3) / (2 - x) = 3 / 2)) ∧
  ((x = 3) → ((x + 3) / (2 - x) = -6)) :=
begin
  sorry
end

end simplified_expression_evaluation_l500_500258


namespace sum_of_squares_of_rates_l500_500061

theorem sum_of_squares_of_rates (b j s : ℕ) 
  (h1 : 3 * b + 2 * j + 4 * s = 66) 
  (h2 : 3 * j + 2 * s + 4 * b = 96) : 
  b^2 + j^2 + s^2 = 612 := 
by 
  sorry

end sum_of_squares_of_rates_l500_500061


namespace a_2n_perfect_square_l500_500216

def fib : ℕ → ℕ
| 0        => 1
| 1        => 1
| (n + 2)  => fib n + fib (n + 1)

def a : ℕ → ℕ
| 0        => 0
| 1        => 1
| 2        => 1
| 3        => 2
| n        => a (n - 4) + a (n - 3) + a (n - 1)

theorem a_2n_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k * k :=
by
  sorry

end a_2n_perfect_square_l500_500216


namespace polynomial_descending_order_l500_500737

theorem polynomial_descending_order (x y : ℝ) :
  let p := -2 * x^3 * y + 4 * x * y^3 + 1 - 3 * x^2 * y^2 in
  p = 4 * x * y^3 - 3 * x^2 * y^2 - 2 * x^3 * y + 1 :=
by sorry

end polynomial_descending_order_l500_500737


namespace tan_alpha_value_l500_500875

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l500_500875


namespace probability_log_floor_eq_1_l500_500588

noncomputable def probability_interval (x y : ℝ) : ℝ :=
  if 1/9 ≤ x ∧ x < 1/3 ∧ 1/9 ≤ y ∧ y < 1/3 then (2/9)^2 else 0

theorem probability_log_floor_eq_1 :
  let x := ℝ
  let y := ℝ
  let x_uniform : x → ℝ := sorry    -- x is chosen from (0,1) independently and uniformly
  let y_uniform : y → ℝ := sorry    -- y is chosen from (0,1) independently and uniformly
  let event := probability_interval x y
  event = 4/81 :=
sorry

end probability_log_floor_eq_1_l500_500588


namespace find_x_y_l500_500025

theorem find_x_y (x y : ℝ) (h1 : x - (3/8) * x = 25) (h2 : 40^y = 125) : x = 40 ∧ y ≠ ∅ := 
by
  sorry

end find_x_y_l500_500025


namespace fraction_least_l500_500301

noncomputable def solve_fraction_least : Prop :=
  ∃ (x y : ℚ), x + y = 5/6 ∧ x * y = 1/8 ∧ (min x y = 1/6)
  
theorem fraction_least : solve_fraction_least :=
sorry

end fraction_least_l500_500301


namespace plane_equation_l500_500710

-- Define the point and the normal vector
def point : ℝ × ℝ × ℝ := (8, -2, 2)
def normal_vector : ℝ × ℝ × ℝ := (8, -2, 2)

-- Define integers A, B, C, D such that the plane equation satisfies the conditions
def A : ℤ := 4
def B : ℤ := -1
def C : ℤ := 1
def D : ℤ := -18

-- Prove the equation of the plane
theorem plane_equation (x y z : ℝ) :
  A * x + B * y + C * z + D = 0 ↔ 4 * x - y + z - 18 = 0 :=
by
  sorry

end plane_equation_l500_500710


namespace tan_alpha_solution_l500_500854

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l500_500854


namespace tan_alpha_value_l500_500877

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l500_500877


namespace relationship_between_a_b_c_l500_500099

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

def a := log_base 1.5 3
def b := Real.exp 0.4
def c := log_base 0.8 9

theorem relationship_between_a_b_c : c < b ∧ b < a := by
  sorry

end relationship_between_a_b_c_l500_500099


namespace problem_l500_500151

theorem problem (x y : ℝ) (h1 : 2^x = 3) (h2 : log 4 (8 / 3) = y) : x + 2 * y = 3 := by
  sorry

end problem_l500_500151


namespace units_digit_N_l500_500562

def P (n : ℕ) : ℕ := (n / 10) * (n % 10)
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem units_digit_N (N : ℕ) (h1 : 10 ≤ N ∧ N ≤ 99) (h2 : N = P N + S N) : N % 10 = 9 :=
by
  sorry

end units_digit_N_l500_500562


namespace trapezoid_area_sum_ineq_l500_500475

-- Definitions for the rational numbers and integers not divisible by the square of any prime
variables (r1 r2 r3 n1 n2 : ℚ)

-- Conditions
def trapezoid_area_sum (r1 r2 r3 n1 n2 : ℕ) : ℝ :=
  (r1 * real.sqrt n1 + r2 * real.sqrt n2 + r3)

-- Define that we are given this specific trapezoid problem
axiom trapezoid_given :
  r1 = 4 ∧ r2 = 3 ∧ r3 = 3 ∧ n1 = 3 ∧ n2 = 5

-- Main statement to prove
theorem trapezoid_area_sum_ineq : 
  ⌊r1 + r2 + r3 + n1 + n2⌋ = 18 := 
by {
  sorry
}

end trapezoid_area_sum_ineq_l500_500475


namespace average_for_remainder_l500_500156

variable {N : ℕ} -- N is the total number of students
variable {R : ℝ} -- R is the average test score for the remainder of the class

-- Conditions
def condition1 := 0.45 * N * 95
def condition2 := 0.5 * N * 78
def condition3 := 0.05 * N * R
def overall_average := (condition1 + condition2 + condition3) / N = 84.75

-- Statement to prove
theorem average_for_remainder : overall_average → R = 60 := by
  sorry

end average_for_remainder_l500_500156


namespace fraction_exponent_simplification_l500_500282

theorem fraction_exponent_simplification :
  (7^((1 : ℝ) / 4)) / (7^((1 : ℝ) / 6)) = 7^((1 : ℝ) / 12) :=
sorry

end fraction_exponent_simplification_l500_500282


namespace length_BF_proof_l500_500184

noncomputable def find_BF {A B C D E F : ℝ} 
  (angle_A_right : ∠A = 90)
  (angle_C_right : ∠C = 90)
  (points_on_AC : E ∈ AC ∧ F ∈ AC)
  (perpendicular_DE : DE ⊥ AC)
  (perpendicular_BF : BF ⊥ AC)
  (AE : ℝ := 3)
  (DE : ℝ := 5)
  (CE : ℝ := 7) : ℝ :=
  BF

theorem length_BF_proof {A B C D E F : ℝ} 
  (angle_A_right : ∠ A = 90)
  (angle_C_right : ∠ C = 90)
  (points_on_AC : E ∈ AC ∧ F ∈ AC)
  (perpendicular_DE : DE ⊥ AC)
  (perpendicular_BF : BF ⊥ AC)
  (AE : ℝ := 3)
  (DE : ℝ := 5)
  (CE : ℝ := 7) : find_BF angle_A_right angle_C_right points_on_AC perpendicular_DE perpendicular_BF AE DE CE = 4.2 :=
sorry

end length_BF_proof_l500_500184


namespace prime_pair_perfect_square_l500_500071

theorem prime_pair_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : ∃ a : ℕ, p^2 + p * q + q^2 = a^2) : (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) := 
sorry

end prime_pair_perfect_square_l500_500071


namespace base8_subtraction_and_conversion_l500_500068

-- Define the base 8 numbers
def num1 : ℕ := 7463 -- 7463 in base 8
def num2 : ℕ := 3254 -- 3254 in base 8

-- Define the subtraction in base 8 and conversion to base 10
def result_base8 : ℕ := 4207 -- Expected result in base 8
def result_base10 : ℕ := 2183 -- Expected result in base 10

-- Helper function to convert from base 8 to base 10
def convert_base8_to_base10 (n : ℕ) : ℕ := 
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8 + (n % 10)
 
-- Main theorem statement
theorem base8_subtraction_and_conversion :
  (num1 - num2 = result_base8) ∧ (convert_base8_to_base10 result_base8 = result_base10) :=
by
  sorry

end base8_subtraction_and_conversion_l500_500068


namespace cos_beta_value_l500_500792

theorem cos_beta_value (α β : ℝ) (h1 : sin α = (sqrt 5) / 5) (h2 : sin (α - β) = -(sqrt 10) / 10)
  (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) :
  cos β = sqrt 2 / 2 :=
sorry

end cos_beta_value_l500_500792


namespace quad_area_correct_l500_500717

noncomputable def area_quad (a b c k : ℝ) : ℝ :=
  (a + b) * (c + max (a - k) (-b - k)) / 2

theorem quad_area_correct (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  let area := area_quad a b c k in
  area = (a + b) * (c + max (a - k) (-b - k)) / 2 := by
  funext; sorry

end quad_area_correct_l500_500717


namespace trigonometric_identity_l500_500431

theorem trigonometric_identity :
  sin (real.pi / 10) * sin (13 * real.pi / 30) - cos (9 * real.pi / 10) * cos (13 * real.pi / 30) = 1 / 2 := 
by
  sorry

end trigonometric_identity_l500_500431


namespace answer1_answer2_answer3_l500_500361

open Finset

def problem1 : ℕ :=
  nat.choose 5 2 * nat.choose 4 2 * nat.factorial 4

theorem answer1 : problem1 = 1440 := by
  sorry

def problem2 : ℕ :=
  nat.choose 7 2 * nat.factorial 4

theorem answer2 : problem2 = 504 := by
  sorry

def problem3 : ℕ :=
  nat.choose 5 3 * nat.choose 4 1 * nat.factorial 4 + nat.factorial 5

theorem answer3 : problem3 = 1080 := by
  sorry

end answer1_answer2_answer3_l500_500361


namespace tan_alpha_value_l500_500925

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l500_500925


namespace sum_of_integers_l500_500613

variable (x y : ℕ)

theorem sum_of_integers (h1 : x - y = 10) (h2 : x * y = 56) : x + y = 18 := 
by 
  sorry

end sum_of_integers_l500_500613


namespace mike_owes_john_l500_500990

theorem mike_owes_john : 
  let dollars_per_car := (9 : ℚ) / 4
  let cars_washed := (10 : ℚ) / 3
  dollars_per_car * cars_washed = 15 / 2 :=
by {
  let g := (dollars_per_car * cars_washed : ℚ)
  calc g = (9 / 4) * (10 / 3) : rfl
      ... = (9 * 10) / (4 * 3) : by rw [div_mul_div]
      ... = 90 / 12 : rfl
      ... = (90 / 6) / (12 / 6) : by rw [div_div_eq_div_mul]
      ... = 15 / 2 : by norm_num
} 

end mike_owes_john_l500_500990


namespace profit_per_metre_is_10_l500_500726

-- Define the given conditions as constants
def cost_price_per_metre : ℝ := 140
def selling_price_for_30_metres : ℝ := 4500
def number_of_metres : ℝ := 30

-- Define the total cost price
def total_cost_price : ℝ := cost_price_per_metre * number_of_metres

-- Define the total profit
def total_profit : ℝ := selling_price_for_30_metres - total_cost_price

-- Define the profit per metre
def profit_per_metre : ℝ := total_profit / number_of_metres

-- The statement we want to prove
theorem profit_per_metre_is_10 : profit_per_metre = 10 := 
by
  sorry

end profit_per_metre_is_10_l500_500726


namespace pyramid_volume_84sqrt10_l500_500716

noncomputable def volume_of_pyramid (a b c : ℝ) : ℝ :=
  (1/3) * a * b * c

theorem pyramid_volume_84sqrt10 :
  let height := 4 * (Real.sqrt 10)
  let area_base := 7 * 9
  (volume_of_pyramid area_base height) = 84 * (Real.sqrt 10) :=
by
  intros
  simp [volume_of_pyramid]
  sorry

end pyramid_volume_84sqrt10_l500_500716


namespace sin_alpha_eq_sqrt6_plus_3_div_6_l500_500944

theorem sin_alpha_eq_sqrt6_plus_3_div_6 (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos (α + π / 3) = -sqrt 3 / 3) : sin α = (sqrt 6 + 3) / 6 := 
by sorry

end sin_alpha_eq_sqrt6_plus_3_div_6_l500_500944


namespace polynomial_has_composite_value_l500_500247

noncomputable def polynomial_with_integer_coeff (n : ℕ) :=
  {P : ℕ → ℤ // ∃ a_n a_{n-1} ... a_0 : ℤ, P(x) = a_n * x^n + a_{n-1} * x^{n-1} + ... + a_1 * x + a_0 ∧ n ≥ 1 }

theorem polynomial_has_composite_value (P : polynomial_with_integer_coeff n x) (hP: ∀ i, P.coeffs i ∈ ℤ) (hDegree : P.degree ≥ 1) :
  ∃ x : ℕ, ∃ y : ℕ, ∃ z : ℕ, y > 1 ∧ z > 1 ∧ y * z = P x := 
begin
  sorry
end

end polynomial_has_composite_value_l500_500247


namespace reciprocals_of_product_one_l500_500524

theorem reciprocals_of_product_one (x y : ℝ) (h : x * y = 1) : x = 1 / y ∧ y = 1 / x :=
by 
  sorry

end reciprocals_of_product_one_l500_500524


namespace count_more_3s_than_7s_in_pages_l500_500437

-- Define the predicate for counting occurrences of a digit in a range of numbers
def count_digit (d : ℕ) (n : ℕ) : ℕ :=
  (list.range' 1 n).map (λ x, (x.digits 10).count d).sum

-- The theorem statement
theorem count_more_3s_than_7s_in_pages : count_digit 3 351 - count_digit 7 351 = 56 :=
sorry

end count_more_3s_than_7s_in_pages_l500_500437


namespace rational_inequality_solution_l500_500597

open Set

theorem rational_inequality_solution (x : ℝ) :
  (x < -1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 5)) ↔ (x - 5) / ((x - 2) * (x^2 - 1)) < 0 := 
sorry

end rational_inequality_solution_l500_500597


namespace sample_processing_l500_500371

-- Define sample data
def standard: ℕ := 220
def samples: List ℕ := [230, 226, 218, 223, 214, 225, 205, 212]

-- Calculate deviations
def deviations (samples: List ℕ) (standard: ℕ) : List ℤ :=
  samples.map (λ x => x - standard)

-- Total dosage of samples
def total_dosage (samples: List ℕ): ℕ :=
  samples.sum

-- Total cost to process to standard dosage
def total_cost (deviations: List ℤ) (cost_per_ml_adjustment: ℤ) : ℤ :=
  cost_per_ml_adjustment * (deviations.map Int.natAbs).sum

-- Theorem statement
theorem sample_processing :
  let deviation_vals := deviations samples standard;
  let total_dosage_val := total_dosage samples;
  let total_cost_val := total_cost deviation_vals 10;
  deviation_vals = [10, 6, -2, 3, -6, 5, -15, -8] ∧
  total_dosage_val = 1753 ∧
  total_cost_val = 550 :=
by
  sorry

end sample_processing_l500_500371


namespace unique_point_on_circle_conditions_l500_500109

noncomputable def point : Type := ℝ × ℝ

-- Define points A and B
def A : point := (-1, 4)
def B : point := (2, 1)

def PA_squared (P : point) : ℝ :=
  let (x, y) := P
  (x + 1) ^ 2 + (y - 4) ^ 2

def PB_squared (P : point) : ℝ :=
  let (x, y) := P
  (x - 2) ^ 2 + (y - 1) ^ 2

-- Define circle C
def on_circle (a : ℝ) (P : point) : Prop :=
  let (x, y) := P
  (x - a) ^ 2 + (y - 2) ^ 2 = 16

-- Define the condition PA² + 2PB² = 24
def condition (P : point) : Prop :=
  PA_squared P + 2 * PB_squared P = 24

-- The main theorem stating the possible values of a
theorem unique_point_on_circle_conditions :
  ∃ (a : ℝ), ∀ (P : point), on_circle a P → condition P → (a = -1 ∨ a = 3) :=
sorry

end unique_point_on_circle_conditions_l500_500109


namespace overall_support_percentage_correct_l500_500964

-- Define the number of male participants
def male_participants : ℕ := 150

-- Define the percentage of male participants who supported the policy
def male_support_percentage : ℝ := 0.55

-- Define the number of female participants
def female_participants : ℕ := 850

-- Define the percentage of female participants who supported the policy
def female_support_percentage : ℝ := 0.70

-- Define the total number of participants
def total_participants : ℕ := male_participants + female_participants

-- Define the number of male supporters
def male_supporters : ℕ := (male_support_percentage * male_participants).to_nat

-- Define the number of female supporters
def female_supporters : ℕ := (female_support_percentage * female_participants).to_nat

-- Define the overall number of supporters
def total_supporters : ℕ := male_supporters + female_supporters

-- Define the overall percentage of supporters
def overall_support_percentage : ℝ := (total_supporters.to_nat : ℝ) / (total_participants : ℝ) * 100

theorem overall_support_percentage_correct :
  overall_support_percentage = 67.8 := by
  sorry

end overall_support_percentage_correct_l500_500964


namespace Q_is_7_continuous_Q_is_not_8_continuous_min_k_for_8_continuous_l500_500465

def is_continuous_representable_sequence (Q : List ℤ) (m : ℕ) : Prop :=
  ∀ n ∈ List.range (m + 1).tail, ∃ (i j : ℕ), i + j < Q.length ∧ (finset.Ico i (i + j + 1)).val.sum (λ k, Q.nthLe k sorry) = n

def Q_7_continuous := [2, 1, 4, 2]
def Q_8_continuous := [2, 1, 4, 2]

theorem Q_is_7_continuous : is_continuous_representable_sequence Q_7_continuous 7 := by sorry

theorem Q_is_not_8_continuous : ¬ is_continuous_representable_sequence Q_8_continuous 8 := by sorry

theorem min_k_for_8_continuous (Q : List ℤ) : is_continuous_representable_sequence Q 8 → Q.length ≥ 4 := by sorry

end Q_is_7_continuous_Q_is_not_8_continuous_min_k_for_8_continuous_l500_500465


namespace chicken_fried_steak_cost_l500_500197

-- Define the conditions
def cost_meal : ℝ := 16
def friend_share (C: ℝ) : ℝ := (cost_meal + C) / 2
def tip (C: ℝ) : ℝ := 0.2 * (cost_meal + C)
def james_payment (C: ℝ) : ℝ := friend_share(C) + tip(C)

-- Prove that the cost of the chicken fried steak is $14 given that James paid $21 in total.
theorem chicken_fried_steak_cost : james_payment 14 = 21 :=
  by
    -- Skip the proof
    sorry

end chicken_fried_steak_cost_l500_500197


namespace bench_section_least_M_l500_500012

theorem bench_section_least_M (M : ℕ) (hM_pos : 0 < M)
  (hM_senior : ∃ s, 5 * M = s)
  (hM_kid : ∃ k, 13 * M = k)
  (h_equal : ∀ s k, s = k) : 
  M = 13 :=
by
  -- Since s = k, we have:
  sorry

end bench_section_least_M_l500_500012


namespace polar_to_cartesian_intersection_value_l500_500506

-- Define the conditions
def polar_eq (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 = 4 * (Real.cos θ)

def parametric_eq (t : ℝ) : ℝ × ℝ := ((1 : ℝ) + (2 * Real.sqrt 5 / 5) * t, (1 : ℝ) + (Real.sqrt 5 / 5) * t)

-- Define the first proof statement
theorem polar_to_cartesian (x y : ℝ) :
  (∃ ρ θ, polar_eq ρ θ ∧ ρ^2 = x^2 + y^2 ∧ ρ * (Real.cos θ) = x ∧ ρ * (Real.sin θ) = y) ↔ y^2 = 4 * x :=
sorry

-- Define the second proof statement
theorem intersection_value (x y : ℝ) (t1 t2 : ℝ) 
  (P : (ℝ × ℝ)) (P_def : P = (1,1))
  (l_intersect_C : ∀ t, parametric_eq t = (x, y) → y^2 = 4 * x) :
  |(1 + 2 * Real.sqrt 5 / 5 * t1) - (1 + 2 * Real.sqrt 5 / 5 * t2)| = 4 * Real.sqrt 15 :=
sorry

end polar_to_cartesian_intersection_value_l500_500506


namespace angle_A_sum_b_c_l500_500979

-- First part: Proving A = π/3
theorem angle_A (a b : ℝ) (A B : ℝ) (h_m_dot_n : a * (Real.sin B) - (sqrt 3) * b * (Real.cos A) = 0)
  (h_sine_rule : ∀ f, Real.sin f ≠ 0) :
  A = π / 3 :=
sorry

-- Second part: Proving b + c = 11 / 2 given a = 7 / 2 and S = 3 / 2 * sqrt 3
theorem sum_b_c (a b c A : ℝ) (S : ℝ)
  (h_a : a = 7 / 2) (h_S : S = 3 / 2 * sqrt 3)
  (h_area : S = 1 / 2 * b * c * (Real.sin A)) :
  b + c = 11 / 2 :=
sorry

end angle_A_sum_b_c_l500_500979


namespace tan_alpha_value_l500_500929

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l500_500929


namespace prob_first_red_light_third_intersection_l500_500015

noncomputable def red_light_at_third_intersection (p : ℝ) (h : p = 2/3) : ℝ :=
(1 - p) * (1 - (1/2)) * (1/2)

theorem prob_first_red_light_third_intersection (h : 2/3 = (2/3 : ℝ)) :
  red_light_at_third_intersection (2/3) h = 1/12 := sorry

end prob_first_red_light_third_intersection_l500_500015


namespace all_terms_are_positive_integers_terms_product_square_l500_500046

def seq (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧
  x 2 = 4 ∧
  ∀ n > 1, x n = Nat.sqrt (x (n - 1) * x (n + 1) + 1)

theorem all_terms_are_positive_integers (x : ℕ → ℕ) (h : seq x) : ∀ n, x n > 0 :=
sorry

theorem terms_product_square (x : ℕ → ℕ) (h : seq x) : ∀ n ≥ 1, ∃ k, 2 * x n * x (n + 1) + 1 = k ^ 2 :=
sorry

end all_terms_are_positive_integers_terms_product_square_l500_500046


namespace rook_tour_count_l500_500203

def rook_tour (n : ℕ) : list (ℕ × ℕ) → Prop :=
  λ p, (∀ i, 1 ≤ i ∧ i < 3 * n → (p[i] ∈ { (a, b) | a = 1 ∨ a = 2 ∨ a = n ∧ b = 1 ∨ b = 2 ∨ b = 3})) ∧
      (∀ i, 1 ≤ i ∧ i < 3 * n - 1 → dist (p[i]) (p[i+1]) = 1) ∧
      (∀ q ∈ S, ∃! i, 1 ≤ i ∧ i < 3 * n ∧ p[i] = q)

def S (n : ℕ) : set (ℕ × ℕ) := { (a, b) | 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ 3 }

theorem rook_tour_count (n : ℕ) : 
  (∑ (p : list (ℕ × ℕ)), rook_tour n p ∧ list.head p = (1, 1) ∧ list.last p = (n, 1)) = 2^(n-2) :=
sorry

end rook_tour_count_l500_500203


namespace euler_theorem_convex_polyhedron_l500_500786

theorem euler_theorem_convex_polyhedron
  (f p a : ℕ) -- faces, vertices, edges
  (h1 : (∑ (S : Point), dihedral_angles_at S) = 4 * π)
  (h2 : (∑ (v : Vertex), dihedral_angles_at v) = 2 * π * p)
  (h3 : (∑ (α : Angle), α) = 2 * π * (a - f)) :
  f + p - a = 2 := 
sorry

end euler_theorem_convex_polyhedron_l500_500786


namespace max_mn_value_l500_500159

-- Definition of the function and conditions
def f (x m n : ℝ) : ℝ := (1 / 2) * (m - 2) * x^2 + (n - 8) * x + 1

-- The proof statement
theorem max_mn_value {m n : ℝ} (h_m_nonneg : m ≥ 0) (h_n_nonneg : n ≥ 0)
    (h_monotonic : ∀ x1 x2, (1 / 2) ≤ x1 → x1 ≤ 2 → (1 / 2) ≤ x2 → x2 ≤ 2 → x1 ≤ x2 → f x1 m n ≥ f x2 m n) :
  mn = 18 :=
sorry

end max_mn_value_l500_500159


namespace coach_mike_change_in_usd_l500_500041

theorem coach_mike_change_in_usd :
  ∀ (cost_per_cup amount_paid conversion_rate : ℝ),
    cost_per_cup = 0.58 →
    amount_paid = 1 →
    conversion_rate = 1.18 →
    (amount_paid - cost_per_cup) * conversion_rate ≈ 0.50 :=
by
  intros cost_per_cup amount_paid conversion_rate h1 h2 h3
  sorry

end coach_mike_change_in_usd_l500_500041


namespace fixed_point_on_circle_l500_500124

noncomputable theory

def circle (x y : ℝ) := x^2 + y^2 - 4 * x + 3 = 0
def line (x y : ℝ) := 2 * x - y = 0

theorem fixed_point_on_circle (t : ℝ) (x y : ℝ) 
  (hP : line t (2 * t)) 
  (hA hB : ∃ A B : ℝ × ℝ, circle A.1 A.2 ∧ circle B.1 B.2 ∧ (t, 2 * t) ≠ A ∧ (t, 2 * t) ≠ B ∧ circle A.1 A.2 ∧ circle B.1 B.2) :
  ∀ C : ℝ × ℝ, C = (2, 0) →
    let D := ( (t + 2)/2, t ) in
    let radius := 1/2 * real.sqrt((t - 2)^2 + (2 * t)^2) in
    ∃ k : ℚ,
      (k = 1/2 * real.sqrt(5 * t^2 - 4 * t + 4)) ∧
      (x - (t + 2)/2)^2 + (y - t)^2 = (5 * t^2 - 4 * t + 4) / 4 → 
      ( (2/5)^2 + (4/5)^2 = radius^2 ) :=
sorry

end fixed_point_on_circle_l500_500124


namespace largest_three_digit_multiple_of_8_with_digit_sum_24_l500_500657

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (8 ∣ n) ∧ nat.digits 10 n.sum = 24 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ (8 ∣ m) ∧ nat.digits 10 m.sum = 24 → m ≤ n :=
begin
  sorry
end

end largest_three_digit_multiple_of_8_with_digit_sum_24_l500_500657


namespace convex_function_derivative_inequality_l500_500217

variables {a b x1 x2 x3 x4 : ℝ}
variable {f : ℝ → ℝ}

def convex_on_open_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ ⦃x y z : ℝ⦄, a < x → x < y → y < z → z < b → 
  (f y - f x) / (y - x) ≤ (f z - f y) / (z - y)

def left_derivative_le_right_derivative (f : ℝ → ℝ) (x : ℝ) := 
  f'_-(x) ≤ f'_+(x)

def non_decreasing_derivative (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ ⦃x y⦄, a < x → x < y → y < b → f'_+(x) ≤ f'_-(y)

theorem convex_function_derivative_inequality 
  (convex_f : convex_on_open_interval f a b)
  (h1 : a < x1) (h2 : x1 < x2) (h3 : x2 < x3) (h4 : x3 < x4) (h5 : x4 < b)
  : 
  (f x2 - f x1) / (x2 - x1) ≤ f'_-(x2) ∧ f'_-(x2) ≤ f'_+(x2) ∧ f'_+(x2) ≤ f'_-(x3) ∧ f'_-(x3) ≤ f'_+(x3) ∧ f'_+(x3) ≤ (f x4 - f x3) / (x4 - x3) :=
begin
  sorry
end

end convex_function_derivative_inequality_l500_500217


namespace lamps_on_with_limited_switching_l500_500639

theorem lamps_on_with_limited_switching 
  (n k : ℕ)                                    -- n lamps, k switches
  (initial_states : Fin n → Bool)              -- initial state of each lamp (on or off)
  (connections : Fin n → Fin 2020 → Fin k)     -- each lamp connected to 2020 switches 
  (can_all_be_turned_on : ∃ S : Finset (Fin k), (∀ lamp, (initial_states lamp = false) 
    → (∃ flip_count : ℕ, ∀ switch ∈ S, flip_count = (count_flip (connections lamp switch)))) 
    → (∀ lamp, (initial_states lamp = true) &&
    (∃ C : Finset (Fin k), ∀ lamp, (∀ c ∈ C, switch lamp (connections lamp c)))
-- ∃ S1, S2 such that S1 ⊎ S2 = {1, ..., k}, and toggling any two
  (h : ∃ S1 S2 : Finset (Fin k), S1 ∪ S2 = Finset.univ ∧
    (∀ lamp, (∃ flip_count_1 : ℕ, ∀ switch ∈ S1, flip_count_1 = (count_flip (connections lamp switch))) ∧
    (∃ flip_count_2 : ℕ, ∀ switch ∈ S2, flip_count_2 = (count_flip (connections lamp switch))) )) : 
  (∃ S : Finset (Fin k), S.card ≤ k / 2 ∧
    (∀ lamp, (initial_states lamp = false) 
    → (∃ flip_count : ℕ, ∀ switch ∈ S, flip_count = (count_flip (connections lamp switch)))) ∧ 
    (∀ lamp, initial_states lamp = true ∧ 
    (∃ C : Finset (Fin k), ∀ lamp, (∀ c ∈ C, switch lamp (connections lamp c))))) :=
sorry

end lamps_on_with_limited_switching_l500_500639


namespace inv_mod_sum_l500_500350

theorem inv_mod_sum : 
  let a := nat.gcd_a 5 25 in
  a * 5 ≡ 1 [MOD 25] →
  let b := nat.gcd_a 25 25 in
  b * 25 ≡ 1 [MOD 25] →
  (a + b) % 25 = 5 :=
by
  sorry

end inv_mod_sum_l500_500350


namespace eq_line_BC_slope_line_BC_constant_l500_500470

def circle : set (ℝ × ℝ) := { p | p.1 ^ 2 + p.2 ^ 2 = 25 }

def point_A : ℝ × ℝ := (3, 4)

def is_inscribed_in_circle (points : set (ℝ × ℝ)) : Prop :=
  ∀ p ∈ points, (p.1 ^ 2 + p.2 ^ 2 = 25)

def triangle_ABC (B C : ℝ × ℝ) : set (ℝ × ℝ) :=
  {point_A, B, C}

def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

def inclination_complementary (A B C : ℝ × ℝ) : Prop :=
  let θ_AB := real.arctan (B.2 - A.2) / (B.1 - A.1)
  let θ_AC := real.arctan (C.2 - A.2) / (C.1 - A.1)
  in θ_AB + θ_AC = π / 2

noncomputable def line_eq (B C : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ p, p.2 - (B.2 - C.2) / (B.1 - C.1) * (p.1 - B.1) = B.2 - (B.2 - C.2) / (B.1 - C.1) * B.1

theorem eq_line_BC (B C : ℝ × ℝ) 
  (h₁ : is_inscribed_in_circle (triangle_ABC B C))
  (h₂ : centroid point_A B C = (5 / 3, 2)) :
  line_eq B C = λ p, p.1 + p.2 - 2 = 0
:= sorry

theorem slope_line_BC_constant (B C : ℝ × ℝ) 
  (h₁ : is_inscribed_in_circle (triangle_ABC B C))
  (h₂ : centroid point_A B C = (5 / 3, 2))
  (h₃ : inclination_complementary point_A B C) :
  (B.2 - C.2) / (B.1 - C.1) = 3 / 4
:= sorry

end eq_line_BC_slope_line_BC_constant_l500_500470


namespace integer_solutions_range_l500_500091

def operation (p q : ℝ) : ℝ := p + q - p * q

theorem integer_solutions_range (m : ℝ) :
  (∃ (x1 x2 : ℤ), (operation 2 x1 > 0) ∧ (operation x1 3 ≤ m) ∧ (operation 2 x2 > 0) ∧ (operation x2 3 ≤ m) ∧ (x1 ≠ x2)) ↔ (3 ≤ m ∧ m < 5) :=
by sorry

end integer_solutions_range_l500_500091


namespace min_expression_l500_500467

theorem min_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 1/b = 2) : 
  ∃ c, ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 1/y = 2 → (4/x + y ≥ c) := 
begin
  use 9 / 2,
  intros x y hx hy hxy,
  -- Placeholder for the actual proof
  sorry
end

end min_expression_l500_500467


namespace minimum_value_of_k_l500_500999

theorem minimum_value_of_k (m n a k : ℕ) (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) (hk : 1 < k) (h : 5^m + 63 * n + 49 = a^k) : k = 5 :=
sorry

end minimum_value_of_k_l500_500999


namespace find_original_mean_l500_500624

noncomputable def original_mean (M : ℝ) : Prop :=
  let num_observations := 50
  let decrement := 47
  let updated_mean := 153
  M * num_observations - (num_observations * decrement) = updated_mean * num_observations

theorem find_original_mean : original_mean 200 :=
by
  unfold original_mean
  simp [*, mul_sub_left_distrib] at *
  sorry

end find_original_mean_l500_500624


namespace projections_intersect_diagonal_l500_500399

theorem projections_intersect_diagonal (A B C D M P Q R S T : Type)
  [is_rectangle A B C D]
  [is_point_on_arc M A B]
  [projections M A D P]
  [projections M A B Q]
  [projections M B C R]
  [projections M C D S]
  (hPQ: is_line PQ : Type)
  (hRS: is_line RS : Type)
  (hPQ_RS: intersect_at PQ RS T)
  (hdiagonal: is_diagonal T) : T ∈ diagonal A C ∨ T ∈ diagonal B D :=
begin
  sorry
end

end projections_intersect_diagonal_l500_500399


namespace triangle_AO_length_l500_500293

theorem triangle_AO_length (a b c x y : ℕ) (h_perimeter : b + c + a = 190)
  (h_right_angle : b^2 + c^2 = a^2) (h_radius : 23 > 0) (h_tangent : 23 = (a - c).natAbs)
  (hx_rel_prime : x.gcd y = 1) (h_bigger_a : a > 0) :
  x = 67 → x + y = 90 := 
by
  sorry

end triangle_AO_length_l500_500293


namespace cos_range_in_triangle_proof_l500_500958

noncomputable def cos_range_in_triangle (a b c : ℝ) (C : ℝ) 
  (AB AC : ℝ) (C_in : 0 < C ∧ C < Real.pi) 
  (cosC_def : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) : Prop :=
  ∀ c = 2, ∀ b = 3, (Real.cos C) ∈ Set.Ico (Real.sqrt 5 / 3) 1

theorem cos_range_in_triangle_proof : 
  cos_range_in_triangle (AB := 2) (AC := 3) := by
  sorry

end cos_range_in_triangle_proof_l500_500958


namespace intersection_A_B_l500_500135

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | x < 2 }

theorem intersection_A_B : A ∩ B = { x | -1 ≤ x ∧ x < 2 } := 
by 
  sorry

end intersection_A_B_l500_500135


namespace area_gray_black_parts_eq_l500_500546

open EuclideanGeometry

variables {A B C D M K : Point}
variables (h : IsTrapezoid A B C D) (h1 : IsParallelogram M B D K)
          (h2 : Parallel (Line.mk M B) (Line.mk A C)) (h3 : Parallel (Line.mk B D) (Line.mk A C))

theorem area_gray_black_parts_eq :
  let S := fun Δ => area Δ in
  S (Triangle.mk K M D) = S (Triangle.mk A K C) :=
by
  sorry

end area_gray_black_parts_eq_l500_500546


namespace range_of_m_double_mean_value_on_interval_l500_500760

def is_double_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x1 x2, a < x1 ∧ x1 < x2 ∧ x2 < b ∧ f'' x1 = (f b - f a) / (b - a) ∧ f'' x2 = (f b - f a) / (b - a)

noncomputable def f : ℝ → ℝ := λ x, x^3 - (6/5) * x^2

theorem range_of_m_double_mean_value_on_interval :
  (∀ m : ℝ, is_double_mean_value_function f 0 m → (3/5 < m ∧ m ≤ 6/5)) :=
by
  sorry

end range_of_m_double_mean_value_on_interval_l500_500760


namespace sin_sum_eq_prod_90_sin_sum_eq_prod_2017_l500_500498

theorem sin_sum_eq_prod_90:
  {n : ℕ} (h1 : 1 ≤ n ∧ n ≤ 90) :
  (\sum i in finset.range n, real.sin (i+1) * real.pi / 180) 
  = (\prod i in finset.range n, real.sin (i+1) * real.pi / 180) ↔ n = 1 :=
sorry

theorem sin_sum_eq_prod_2017:
  {n : ℕ} (h2 : 1 ≤ n ∧ n ≤ 2017) :
  (finset.card (n ∈ finset.range 2017 
  where (\sum i in finset.range n, real.sin (i+1) * real.pi / 180) 
  = (\prod i in finset.range n, real.sin (i+1) * real.pi / 180))) = 11 :=
sorry

end sin_sum_eq_prod_90_sin_sum_eq_prod_2017_l500_500498


namespace eight_digit_integers_count_l500_500519

-- Define 8-digit positive integers with specific conditions on the first digit
def is_valid_eight_digit_integer (n : ℕ) : Prop :=
  10000000 ≤ n ∧ n < 100000000 ∧ (2 ≤ n / 10000000) ∧ (n / 10000000 ≤ 9)

theorem eight_digit_integers_count : 
  {n : ℕ | is_valid_eight_digit_integer n }.card = 80_000_000 :=
by
  sorry

end eight_digit_integers_count_l500_500519


namespace simplify_and_evaluate_l500_500593

theorem simplify_and_evaluate (x : ℝ) (hx : x = Real.sqrt 2) :
  ( ( (2 * x - 1) / (x + 1) - x + 1 ) / (x - 2) / (x^2 + 2 * x + 1) ) = -2 - Real.sqrt 2 :=
by sorry

end simplify_and_evaluate_l500_500593


namespace no_distributive_laws_l500_500052

def star (a b : ℝ) : ℝ := 3 * (a + b) / 2

theorem no_distributive_laws (x y z : ℝ) :
  ¬ ((x * star (y + z) = (x * star y) + (x * star z)) ∨
    (x + star y * z = star (x + y) * star (x + z)) ∨
    (x * star (y * star z) = star (x *  y) * star (x * star z))) :=
by sorry

end no_distributive_laws_l500_500052


namespace volunteering_plans_count_l500_500188

theorem volunteering_plans_count :
  let People := {A, B, C, D}
  let Projects := {ProjectA, ProjectB, ProjectC}
  ∃ plans : ℕ, plans = 36 ∧
    (∀ project ∈ Projects, ∃ volunteer ∈ People, volunteer ∈ projects.volunteers) :=
begin
  sorry
end

end volunteering_plans_count_l500_500188


namespace gabby_mom_gave_20_l500_500463

theorem gabby_mom_gave_20 (makeup_set_cost saved_money more_needed total_needed mom_money : ℕ)
  (h1 : makeup_set_cost = 65)
  (h2 : saved_money = 35)
  (h3 : more_needed = 10)
  (h4 : total_needed = makeup_set_cost - saved_money)
  (h5 : total_needed - mom_money = more_needed) :
  mom_money = 20 :=
by
  sorry

end gabby_mom_gave_20_l500_500463


namespace arithmetic_sequence_a6_value_l500_500180

theorem arithmetic_sequence_a6_value
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 14) :
  a 6 = 2 :=
by
  sorry

end arithmetic_sequence_a6_value_l500_500180


namespace no_two_var_poly_satisfies_conditions_l500_500434

theorem no_two_var_poly_satisfies_conditions :
  ¬ ∃ (P : ℝ[X][Y]), (∀ x y : ℝ, 0 < P.eval (x, y) ↔ 0 < x ∧ 0 < y) :=
by
  sorry

end no_two_var_poly_satisfies_conditions_l500_500434


namespace tan_alpha_proof_l500_500896

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l500_500896


namespace tan_alpha_fraction_l500_500904

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l500_500904


namespace sample_size_second_grade_l500_500636

theorem sample_size_second_grade
    (total_students : ℕ)
    (ratio_first : ℕ)
    (ratio_second : ℕ)
    (ratio_third : ℕ)
    (sample_size : ℕ) :
    total_students = 2000 →
    ratio_first = 5 → ratio_second = 3 → ratio_third = 2 →
    sample_size = 20 →
    (20 * (3 / (5 + 3 + 2)) = 6) :=
by
  intros ht hr1 hr2 hr3 hs
  -- The proof would continue from here, but we're finished as the task only requires the statement.
  sorry

end sample_size_second_grade_l500_500636


namespace a_in_M_l500_500225

def M : Set ℝ := { x | x ≤ 5 }
def a : ℝ := 2

theorem a_in_M : a ∈ M :=
by
  -- Proof omitted
  sorry

end a_in_M_l500_500225


namespace wire_length_before_cutting_l500_500017

-- Definitions and problem statement
def S := 17.14285714285714

axiom relation_S_L : ∃ L, S = (2 / 5) * L

axiom total_length_of_wire (L : ℝ) : S = 17.14285714285714 → S = (2 / 5) * L → S + L = 60

-- Proof placeholder
theorem wire_length_before_cutting : S + (classical.some relation_S_L) = 60 :=
by
  sorry

end wire_length_before_cutting_l500_500017


namespace evaluate_expression_l500_500066

theorem evaluate_expression :
  let x := 16
  in (2 + x * (2 + Real.sqrt x) - 4^2) / (Real.sqrt x - 4 + x^2) = 41 / 128 := by
    let x := 16
    have h_sqrt : Real.sqrt x = 4 := by sorry
    have h_eq : (2 + x * (2 + Real.sqrt x) - 4^2) / (Real.sqrt x - 4 + x^2) = (2 + 16 * (2 + 4) - 16) / (4 - 4 + 256) := by sorry
    have h_simp : (2 + 16 * (2 + 4) - 16) / (4 - 4 + 256) = (2 + 96 - 16) / 256 := by sorry
    have h_final : (2 + 96 - 16) / 256 = 41 / 128 := by sorry
    exact eq.trans (eq.trans h_eq h_simp) h_final

end evaluate_expression_l500_500066


namespace michelle_january_cost_l500_500689

noncomputable def cell_phone_cost (base_cost : ℕ) (text_rate : ℕ) (extra_minute_rate : ℕ)
  (included_hours : ℕ) (texts_sent : ℕ) (talked_hours : ℕ) : ℕ :=
  let cost_base := base_cost
  let cost_texts := texts_sent * text_rate / 100
  let extra_minutes := (talked_hours - included_hours) * 60
  let cost_extra_minutes := extra_minutes * extra_minute_rate / 100
  cost_base + cost_texts + cost_extra_minutes

theorem michelle_january_cost : cell_phone_cost 20 5 10 30 100 30.5 = 28 :=
by
  sorry

end michelle_january_cost_l500_500689


namespace A_and_D_independent_l500_500311

-- Define the probabilities of elementary events
def prob_A : ℚ := 1 / 6
def prob_B : ℚ := 1 / 6
def prob_C : ℚ := 5 / 36
def prob_D : ℚ := 1 / 6

-- Define the joint probability of A and D
def prob_A_and_D : ℚ := 1 / 36

-- Define the independence condition
def independent (P_X P_Y P_XY : ℚ) : Prop := P_XY = P_X * P_Y

-- Prove that events A and D are independent
theorem A_and_D_independent : 
  independent prob_A prob_D prob_A_and_D := by
  -- The proof is skipped
  sorry

end A_and_D_independent_l500_500311


namespace runner_catch_up_time_l500_500653

theorem runner_catch_up_time (t : ℕ) : 
  (∀. t ≥ 56, ⟹ ((t - 56 * 7 / 8) ≤ (t)) :=
begin
  sorry -- Proof detail to be filled in later
end

end runner_catch_up_time_l500_500653


namespace a_divisible_by_six_l500_500777

theorem a_divisible_by_six (a : ℚ) : (∀ n : ℕ, a * n * (n + 2) * (n + 3) * (n + 4) ∈ ℤ) → (∃ k : ℤ, a = k / 6) :=
sorry

end a_divisible_by_six_l500_500777


namespace cash_realized_correct_l500_500279

-- Definitions for the conditions
def total_amount_before_brokerage : ℝ := 106
def brokerage_rate : ℝ := 1 / 4 / 100

-- Definition for the brokerage amount
def brokerage_amount : ℝ := brokerage_rate * total_amount_before_brokerage

-- Definition for the cash realized on selling the stock after deducting the brokerage
def cash_realized : ℝ := total_amount_before_brokerage - brokerage_amount

-- The theorem stating the correct answer
theorem cash_realized_correct : cash_realized = 105.735 :=
by
  -- skip the proof
  sorry

end cash_realized_correct_l500_500279


namespace find_a1_l500_500542

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) :=
  ∀ n, a (n + 1) = a n + d

theorem find_a1 (h_arith : is_arithmetic_sequence a 3) (ha2 : a 2 = -5) : a 1 = -8 :=
sorry

end find_a1_l500_500542


namespace angle_B_measure_l500_500340

theorem angle_B_measure:
  ∀ (A B C : ℝ),
  B = 3 * A + 10 →
  C = B →
  A + B + C = 180 →
  B = 550 / 7 :=
by
  -- Given conditions
  intros A B C h1 h2 h3,
  -- Proof would go here
  sorry

end angle_B_measure_l500_500340


namespace arrangement_count_l500_500460

theorem arrangement_count :
  let boys := 4
  let girls := 3
  let total_individuals := boys + girls
  ∃ arrangements : ℕ,
    arrangements = 3600 ∧
    arrangements = 
      (factorial total_individuals - (factorial (total_individuals - 2) * 2 * 3)) - 
      (factorial (total_individuals - 3 + 1) * factorial 3) :=
by sorry

end arrangement_count_l500_500460


namespace find_salary_of_Thomas_l500_500277

-- Declare the variables representing the salaries of Raj, Roshan, and Thomas
variables (R S T : ℝ)

-- Given conditions as definitions
def avg_salary_Raj_Roshan : Prop := (R + S) / 2 = 4000
def avg_salary_Raj_Roshan_Thomas : Prop := (R + S + T) / 3 = 5000

-- Stating the theorem
theorem find_salary_of_Thomas
  (h1 : avg_salary_Raj_Roshan R S)
  (h2 : avg_salary_Raj_Roshan_Thomas R S T) : T = 7000 :=
by
  sorry

end find_salary_of_Thomas_l500_500277


namespace sum_of_integers_l500_500610

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 10) (h3 : x * y = 56) : x + y = 18 := 
sorry

end sum_of_integers_l500_500610


namespace min_value_of_reciprocals_l500_500111

theorem min_value_of_reciprocals (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) : 
  ∃ x, x = (1 / a) + (1 / b) ∧ x ≥ 4 := 
sorry

end min_value_of_reciprocals_l500_500111


namespace Alice_favorite_number_exists_l500_500394

def is_sum_of_digits_multiple_of_four (n : ℕ) : Prop :=
  (n.digits.sum % 4 = 0)

theorem Alice_favorite_number_exists :
  ∃ x : ℤ, (100 < x ∧ x < 150) ∧ (13 ∣ x) ∧ (¬ 3 ∣ x) ∧ is_sum_of_digits_multiple_of_four x.natAbs := 
by
  sorry

end Alice_favorite_number_exists_l500_500394


namespace probability_at_least_four_stayed_l500_500767

theorem probability_at_least_four_stayed (h1 : 4 = 4) (h2 : 1 / 3 = 1 / 3) :
  let P := 
    (1 : ℝ) * ((2 : ℝ) / 3) ^ 4 + 
    (4 : ℝ) * ((1 : ℝ) / 3) * ((2 : ℝ) / 3) ^ 3 + 
    (6 : ℝ) * ((1 : ℝ) / 9) * ((4 : ℝ) / 9) + 
    (4 : ℝ) * ((1 : ℝ) / 27) * ((2 : ℝ) / 3) + 
    ((1 : ℝ) / 81) 
  in P = 1 :=
by sorry

end probability_at_least_four_stayed_l500_500767


namespace nat_arrangement_property_l500_500776

theorem nat_arrangement_property (n : ℕ) :
  (∃ (a : Fin n → ℕ), 
    ∀ k : ℕ, 1 ≤ k ∧ k < n → 
      let Ak := (Finset.range k).sum (λ i, a ⟨i, sorry⟩),
          Bnk := (Finset.range (n - k)).sum (λ i, a ⟨k + i, sorry⟩)
      in Ak ∣ Bnk ∨ Bnk ∣ Ak) ↔ n = 3 ∨ n = 4 ∨ n = 5 :=
sorry

end nat_arrangement_property_l500_500776


namespace max_value_of_function_l500_500450

noncomputable def max_value_function : ℝ :=
  real.cosh

theorem max_value_of_function :
  ∃ x, ∀ y, y = ( √3 / 2 * sin (x + π / 2) + cos (π / 6 - x) ) → y ≤ √13 / 2 :=
sorry

end max_value_of_function_l500_500450


namespace ratio_of_radii_l500_500568

-- Define the geometrical and tangency conditions for the problem
def rectangle (A B C D : Type) : Prop := sorry
def point_on_segment (E D A : Type) (on_AD : E ∈ segment A D) : Prop := sorry
def inscribed_circle (ω : Type) (quad : Type) (tangent_to_BE_at_T : ω.tangentor quad BE at T) : Prop := sorry
def incircle (ω : Type) (triangle : Type) (tangent_to_BE_at_T : ω.tangent triangle BE at T) : Prop := sorry

-- Setting up the main statement of the proof
theorem ratio_of_radii 
{A B C D E : Type} 
(h_rect : rectangle A B C D) 
(h_point : point_on_segment E D A) 
{ω1 ω2 : Type} 
(h_inscribed : inscribed_circle ω1 (quad B C D E) (tangent_to_T : tangent ω1 BE at T))
(h_incircle : incircle ω2 (triangle A B E) (tangent_to_T : tangent ω2 BE at T)) : 
  real := 
(3 + real.sqrt 5) / 2 := 
sorry

end ratio_of_radii_l500_500568


namespace length_of_one_side_l500_500627

-- Definitions according to the conditions
def perimeter (nonagon : Type) : ℝ := 171
def sides (nonagon : Type) : ℕ := 9

-- Math proof problem to prove
theorem length_of_one_side (nonagon : Type) : perimeter nonagon / sides nonagon = 19 :=
by
  sorry

end length_of_one_side_l500_500627


namespace inequality_cubic_inequality_ln_l500_500155

theorem inequality_cubic (x y : ℝ) (h : x > y) : x^3 > y^3 :=
sorry

theorem inequality_ln (x y : ℝ) (h : x > y) : ln x > ln y :=
sorry

end inequality_cubic_inequality_ln_l500_500155


namespace distance_from_point_to_line_l500_500074

open Real

def point : ℝ^3 := ⟨2, 4, 6⟩
def line (t : ℝ) : ℝ^3 := ⟨5 * t + 8, 2 * t + 9, -3 * t + 9⟩
def direction_vector : ℝ^3 := ⟨5, 2, -3⟩

noncomputable def closest_point_distance : ℝ :=
  let t := -31 / 38
  let closest_point := line t
  dist point closest_point

theorem distance_from_point_to_line :
  closest_point_distance = 7 :=
sorry

end distance_from_point_to_line_l500_500074


namespace infinite_sets_existence_l500_500059

theorem infinite_sets_existence (n : ℕ) (h : n = 1983) :
  ∃ (C : ℕ → set ℕ), (∀ k : ℕ, ∃ a : ℕ, a ≠ 1 ∧ (C k).card = 1983 ∧ 
      ∀ x ∈ C k, ∃ b : ℕ, b ∣ x ∧ b = a^1983) :=
by sorry

end infinite_sets_existence_l500_500059


namespace min_area_and_tangent_line_l500_500507

noncomputable def P (t : ℝ) : ℝ × ℝ :=
  (t, t^2 / 4)

def parabola_eq (P : ℝ × ℝ) : Prop :=
  4 * P.2 = P.1^2

def focus : ℝ × ℝ := (0, 1)

def vector_condition (P F Q : ℝ × ℝ) (μ : ℝ) : Prop :=
  (P.1 - F.1, P.2 - F.2) = (μ * (F.1 - Q.1), μ * (F.2 - Q.2))

def min_area (P Q R : ℝ × ℝ) : ℝ :=
  abs ((Q.1 - P.1) * (R.2 - P.2) - (Q.2 - P.2) * (R.1 - P.1)) / 2

def tangent_line_eq (P : ℝ × ℝ) (line_eq : ℝ × ℝ → Prop) : Prop :=
  ∀ x y : ℝ, line_eq (x, y) ↔ y = (√3/3) * x - 1/3 ∨ y = -(√3/3) * x - 1/3

theorem min_area_and_tangent_line :
  ∃ P Q R : ℝ × ℝ,
    parabola_eq P ∧
    (∃ μ : ℝ, vector_condition P focus Q μ) ∧
    min_area P Q R = 16 * √3 / 9 ∧
    tangent_line_eq P (λ x : ℝ × ℝ, parabola_eq (x.1, x.2)) :=
  sorry

end min_area_and_tangent_line_l500_500507


namespace find_circumference_of_tangent_circle_l500_500539

-- Definitions of given conditions
variables (A B : Point)
variables (r1 r2 : ℝ)  -- radii of the initial circles
variables (arcBCLength arcACLength : ℝ)
variables (radiusTangentCircle : ℝ)

-- Define lengths from the problem
def length_of_arcBC : ℝ := 15
def length_of_arcAC : ℝ := 10

-- Tangent circle radius calculation; assume function that computes based on multiple tangents
def radius_of_tangent_circle (r1 r2 : ℝ) : ℝ := (r1 + r2) / 4 -- Simplified assumption

-- Circumference of the tangent circle
def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius

-- Main theorem to prove
theorem find_circumference_of_tangent_circle
  (r1 r2 : ℝ)
  (h1 : r1 = 15 * 360 / (2 * Real.pi * 60))
  (h2 : r2 = 10 * 360 / (2 * Real.pi * 60))
  (h3 : radiusTangentCircle = (r1 + r2) / 4)
  : circumference radiusTangentCircle = 25 * Real.pi / 2 := 
  by sorry

end find_circumference_of_tangent_circle_l500_500539


namespace ratio_of_green_to_yellow_l500_500355

-- Given conditions
def total_rows : ℕ := 6
def flowers_per_row : ℕ := 13
def yellow_flowers : ℕ := 12
def red_flowers : ℕ := 42
def total_flowers : ℕ := total_rows * flowers_per_row

-- Number of green flowers calculated from given condition
def green_flowers : ℕ := total_flowers - (yellow_flowers + red_flowers)

-- Our goal statement
theorem ratio_of_green_to_yellow :
  (green_flowers:ℚ) / yellow_flowers = 2 := 
by
  sorry

end ratio_of_green_to_yellow_l500_500355


namespace BKING_2023_reappears_at_20_l500_500286

-- Defining the basic conditions of the problem
def cycle_length_BKING : ℕ := 5
def cycle_length_2023 : ℕ := 4

-- Formulating the proof problem statement
theorem BKING_2023_reappears_at_20 :
  Nat.lcm cycle_length_BKING cycle_length_2023 = 20 :=
by
  sorry

end BKING_2023_reappears_at_20_l500_500286


namespace cost_of_jeans_l500_500362

-- Define the initial prices
def socks_price := 5
def tshirt_price := socks_price + 10
def jeans_price := 2 * tshirt_price

-- Define the discounts
def tshirt_discount := 0.10
def jeans_discount := 0.15

-- Apply the discounts
def discounted_tshirt_price := tshirt_price * (1 - tshirt_discount)
def discounted_jeans_price := jeans_price * (1 - jeans_discount)

-- Define the tax rate
def tax_rate := 0.08

-- Apply the tax to the jeans
def taxed_jeans_price := discounted_jeans_price * (1 + tax_rate)

-- Expected total cost for the jeans after discount and including sales tax
def expected_cost := 27.54

-- The theorem to prove
theorem cost_of_jeans : taxed_jeans_price = expected_cost :=
  by
    -- Insert the proof here
    sorry

end cost_of_jeans_l500_500362


namespace problem1_problem2_l500_500478

variables (a b c d e f : ℝ)

-- Define the probabilities and the sum condition
def total_probability (a b c d e f : ℝ) : Prop := a + b + c + d + e + f = 1

-- Define P and Q
def P (a b c d e f : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2 + e^2 + f^2
def Q (a b c d e f : ℝ) : ℝ := (a + c + e) * (b + d + f)

-- Problem 1
theorem problem1 (h : total_probability a b c d e f) : P a b c d e f ≥ 1/6 := sorry

-- Problem 2
theorem problem2 (h : total_probability a b c d e f) : 
  1/4 ≥ Q a b c d e f ∧ Q a b c d e f ≥ 1/2 - 3/2 * P a b c d e f := sorry

end problem1_problem2_l500_500478


namespace find_complex_number_l500_500811

theorem find_complex_number (z : ℂ) (hz1 : ∀ x, z = x → ∃ b : ℝ, x = 4 - b*complex.I) 
  (hz2 : (z / (2 - complex.I)).im = 0) : z = 4 - 2 * complex.I :=
by sorry

end find_complex_number_l500_500811


namespace num_solutions_20_l500_500578

-- Define the number of integer solutions function
def num_solutions (n : ℕ) : ℕ := 4 * n

-- Given conditions
axiom h1 : num_solutions 1 = 4
axiom h2 : num_solutions 2 = 8

-- Theorem to prove the number of solutions for |x| + |y| = 20 is 80
theorem num_solutions_20 : num_solutions 20 = 80 :=
by sorry

end num_solutions_20_l500_500578


namespace mathcity_buses_l500_500962

theorem mathcity_buses (r : ℕ) (r_odd : ¬ even r) (m : ℕ) (stations : Finset ℕ) 
  (h_station_index : ∀ s ∈ stations, ∃ k : ℕ, s = 2^k) (favorite_number : ℕ) : 
  ∃ bus_number : ℕ, ∃ a : ℕ, ∃ k : ℕ, (bus_number = a + r * k) ∧ (∀ s ∈ stations, s ∈ { n | n ≤ bus_number ∧ ∃ k, n = 2^k }) :=
sorry

end mathcity_buses_l500_500962


namespace find_k_value_l500_500647

theorem find_k_value :
  let x := [1.0, 1.2, 1.4, 1.6, 1.8]
  let y := [5.0, 5.8, (k : ℝ), 8.1, 8.8]
  let y_hat := λ x, 5 * x - 0.04
  ∃ (k : ℝ), (27.7 + k) / 5 = 6.96 ∧ k = 7.1 :=
by
  sorry

end find_k_value_l500_500647


namespace ROI_diff_after_2_years_is_10_l500_500063

variables (investment_Emma : ℝ) (investment_Briana : ℝ)
variables (yield_Emma : ℝ) (yield_Briana : ℝ)
variables (years : ℝ)

def annual_ROI_Emma (investment_Emma yield_Emma : ℝ) : ℝ :=
  yield_Emma * investment_Emma

def annual_ROI_Briana (investment_Briana yield_Briana : ℝ) : ℝ :=
  yield_Briana * investment_Briana

def total_ROI_Emma (investment_Emma yield_Emma years : ℝ) : ℝ :=
  annual_ROI_Emma investment_Emma yield_Emma * years

def total_ROI_Briana (investment_Briana yield_Briana years : ℝ) : ℝ :=
  annual_ROI_Briana investment_Briana yield_Briana * years

def ROI_difference (investment_Emma investment_Briana yield_Emma yield_Briana years : ℝ) : ℝ :=
  total_ROI_Briana investment_Briana yield_Briana years - total_ROI_Emma investment_Emma yield_Emma years

theorem ROI_diff_after_2_years_is_10 :
  ROI_difference 300 500 0.15 0.10 2 = 10 :=
by
  sorry

end ROI_diff_after_2_years_is_10_l500_500063


namespace train_car_speed_ratio_l500_500644

def bus_speed (d t : ℕ) : ℕ := d / t

def train_speed (bus_speed : ℕ) : ℕ := bus_speed * 5 / 4

def car_speed (d t : ℕ) : ℕ := d / t

def speed_ratio (train_speed car_speed : ℕ) : ℚ := train_speed / car_speed

theorem train_car_speed_ratio
  (bus_distance bus_time car_distance car_time : ℕ)
  (h_bus : bus_distance = 320) (h_bus_time : bus_time = 5)
  (h_car : car_distance = 525) (h_car_time : 7)
  (h1 : train_speed (bus_speed bus_distance bus_time) = 80)
  (h2 : car_speed car_distance car_time = 75) :
  speed_ratio (train_speed (bus_speed bus_distance bus_time)) (car_speed car_distance car_time) = 16 / 15 :=
by sorry

end train_car_speed_ratio_l500_500644


namespace range_of_a_l500_500509

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 + a * x + 1 = 0}
def B : Set ℝ := {1, 2}
def condition (a : ℝ) : Prop := A a ⊆ B

theorem range_of_a : {a : ℝ | condition a} = Icc (-2 : ℝ) 2 :=
by
  sorry

end range_of_a_l500_500509


namespace const_if_ineq_forall_l500_500984

theorem const_if_ineq_forall {f : ℝ → ℝ}
    (H : ∀ x y : ℝ, (f x - f y)^2 ≤ |x - y|^3) :
    ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end const_if_ineq_forall_l500_500984


namespace remaining_bins_capacity_l500_500380

theorem remaining_bins_capacity : ∀ (total_bins : ℕ) (capacity : ℕ) (num_20ton_bins : ℕ) (capacity_20ton_bin : ℕ) (remaining_bins : ℕ),
  total_bins = 30 →
  capacity = 510 →
  num_20ton_bins = 12 →
  capacity_20ton_bin = 20 →
  remaining_bins = total_bins - num_20ton_bins →
  (capacity - (num_20ton_bins * capacity_20ton_bin)) / remaining_bins = 15 :=
begin
  intros total_bins capacity num_20ton_bins capacity_20ton_bin remaining_bins,
  assume h1 h2 h3 h4 h5,
  sorry
end

end remaining_bins_capacity_l500_500380


namespace find_a_b_and_phi_l500_500515

noncomputable def f (a b x : ℝ) : ℝ :=
  ((2 * a * Real.cos x, Real.sin x) • (Real.cos x, b * Real.cos x)) - Real.sqrt 3 / 2

theorem find_a_b_and_phi
  (a b : ℝ)
  (h_y_intercept : f a b 0 = Real.sqrt 3 / 2)
  (h_highest_pt : f a b (Real.pi / 12) = 1)
  (h_phi_pos : ∀ φ : ℕ, φ > 0 → (λ x, f a b (x - φ)) = (λ x, Real.sin x)) :
  a = Real.sqrt 3 / 2 ∧ b = 1 ∧ (∃ φ : ℝ, φ = 5 * Real.pi / 6 ∧ φ > 0) :=
sorry

end find_a_b_and_phi_l500_500515


namespace real_part_of_complex_division_l500_500633

-- Definitions based on the conditions provided
def imaginary_unit : ℂ := Complex.i

-- The statement to prove: The real part of (i / (1 + i)) is 1/2
theorem real_part_of_complex_division : 
  realPart (imaginary_unit / (1 + imaginary_unit)) = 1 / 2 :=
by
  sorry

end real_part_of_complex_division_l500_500633


namespace pencils_in_drawer_l500_500640

theorem pencils_in_drawer (x : ℕ) : 
  let initialPencils := 41
  let mikesPencils := 30
  let totalAfterMike := initialPencils + mikesPencils
  let totalPencils := totalAfterMike + x
  totalPencils = 71 + x :=
by 
  let initialPencils := 41
  let mikesPencils := 30
  let totalAfterMike := initialPencils + mikesPencils
  let totalPencils := totalAfterMike + x
  have h₁ : totalAfterMike = 71 := by
    unfold initialPencils mikesPencils
    simp
  rw [h₁]
  exact rfl

end pencils_in_drawer_l500_500640


namespace population_after_10_years_l500_500631

def initial_population : ℕ := 100000
def birth_increase_percent : ℝ := 0.6
def emigration_per_year : ℕ := 2000
def immigration_per_year : ℕ := 2500
def years : ℕ := 10

theorem population_after_10_years :
  let birth_increase := initial_population * birth_increase_percent
  let total_emigration := emigration_per_year * years
  let total_immigration := immigration_per_year * years
  let net_movement := total_immigration - total_emigration
  let final_population := initial_population + birth_increase + net_movement
  final_population = 165000 :=
by
  sorry

end population_after_10_years_l500_500631


namespace abs_diff_x_y_l500_500204

noncomputable def floor (z : ℝ) : ℤ := int.floor z
noncomputable def frac (z : ℝ) : ℝ := z - floor z

theorem abs_diff_x_y (x y : ℝ)
  (h1 : floor x + frac y = 3.7)
  (h2 : frac x + floor y = 4.2) :
  abs (x - y) = 1.5 :=
by
  sorry

end abs_diff_x_y_l500_500204


namespace sequence_formula_l500_500472

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = 2 * a (n - 1) + 1

theorem sequence_formula (a : ℕ → ℕ) (h : sequence a) : ∀ n, a n = 2^n - 1 :=
by
  sorry

end sequence_formula_l500_500472


namespace tan_alpha_solution_l500_500860

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l500_500860


namespace inequality_proof_l500_500482

theorem inequality_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
  (1 - 2 * x) / Real.sqrt (x * (1 - x)) + 
  (1 - 2 * y) / Real.sqrt (y * (1 - y)) + 
  (1 - 2 * z) / Real.sqrt (z * (1 - z)) ≥ 
  Real.sqrt (x / (1 - x)) + 
  Real.sqrt (y / (1 - y)) + 
  Real.sqrt (z / (1 - z)) :=
by
  sorry

end inequality_proof_l500_500482


namespace hike_distance_l500_500551

theorem hike_distance :
  let total_distance := 0.7
  let car_to_stream := 0.2
  let meadow_to_campsite := 0.1
  let stream_to_meadow := total_distance - (car_to_stream + meadow_to_campsite)
  stream_to_meadow = 0.4 :=
by
  let total_distance := 0.7
  let car_to_stream := 0.2
  let meadow_to_campsite := 0.1
  let stream_to_meadow := total_distance - (car_to_stream + meadow_to_campsite)
  show stream_to_meadow = 0.4
  sorry

end hike_distance_l500_500551


namespace population_after_10_years_l500_500632

def initial_population : ℕ := 100000
def birth_increase_percent : ℝ := 0.6
def emigration_per_year : ℕ := 2000
def immigration_per_year : ℕ := 2500
def years : ℕ := 10

theorem population_after_10_years :
  let birth_increase := initial_population * birth_increase_percent
  let total_emigration := emigration_per_year * years
  let total_immigration := immigration_per_year * years
  let net_movement := total_immigration - total_emigration
  let final_population := initial_population + birth_increase + net_movement
  final_population = 165000 :=
by
  sorry

end population_after_10_years_l500_500632


namespace count_inverosimils_up_to_2022_l500_500007

def isInverosimil (n : ℕ) : Prop :=
  ∃ a : Fin n → ℤ, (∑ i : Fin n, a i) = n ∧ (∏ i : Fin n, a i) = n

theorem count_inverosimils_up_to_2022 : 
  (Finset.range 2023).filter isInverosimil).card = 1010 := by
  sorry

end count_inverosimils_up_to_2022_l500_500007


namespace probability_prime_three_of_six_is_correct_l500_500733

noncomputable def probability_prime_three_of_six : ℚ :=
  (nat.choose 6 3) * ((5/12) ^ 3) * ((7/12) ^ 3)

theorem probability_prime_three_of_six_is_correct :
  probability_prime_three_of_six = 312500 / 248832 :=
by 
  -- sorry to skip the proof
  sorry

end probability_prime_three_of_six_is_correct_l500_500733


namespace total_handshakes_l500_500167

def total_people := 40
def group_x_people := 25
def group_x_known_others := 5
def group_y_people := 15
def handshakes_between_x_y := group_x_people * group_y_people
def handshakes_within_x := 25 * (25 - 1 - 5) / 2
def handshakes_within_y := (15 * (15 - 1)) / 2

theorem total_handshakes 
    (h1 : total_people = 40)
    (h2 : group_x_people = 25)
    (h3 : group_x_known_others = 5)
    (h4 : group_y_people = 15) :
    handshakes_between_x_y + handshakes_within_x + handshakes_within_y = 717 := 
by
  sorry

end total_handshakes_l500_500167


namespace gravitational_force_on_space_station_l500_500620

-- Define the problem conditions and gravitational relationship
def gravitational_force_proportionality (f d : ℝ) : Prop :=
  ∃ k : ℝ, f * d^2 = k

-- Given conditions
def earth_surface_distance : ℝ := 6371
def space_station_distance : ℝ := 100000
def surface_gravitational_force : ℝ := 980
def proportionality_constant : ℝ := surface_gravitational_force * earth_surface_distance^2

-- Statement of the proof problem
theorem gravitational_force_on_space_station :
  gravitational_force_proportionality surface_gravitational_force earth_surface_distance →
  ∃ f2 : ℝ, f2 = 3.977 ∧ gravitational_force_proportionality f2 space_station_distance :=
sorry

end gravitational_force_on_space_station_l500_500620


namespace adam_has_10_apples_l500_500022

theorem adam_has_10_apples
  (Jackie_has_2_apples : ∀ Jackie_apples, Jackie_apples = 2)
  (Adam_has_8_more_apples : ∀ Adam_apples Jackie_apples, Adam_apples = Jackie_apples + 8)
  : ∀ Adam_apples, Adam_apples = 10 :=
by {
  sorry
}

end adam_has_10_apples_l500_500022


namespace problem_part_I_problem_part_II_l500_500960

-- Problem (I)
theorem problem_part_I (a b c : ℝ) (A B C : ℝ) (h1 : b * (1 + Real.cos C) = c * (2 - Real.cos B)) : 
  a + b = 2 * c -> (a + b) = 2 * c :=
by
  intros h
  sorry

-- Problem (II)
theorem problem_part_II (a b c : ℝ) (A B C : ℝ) 
  (h1 : C = Real.pi / 3) 
  (h2 : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3) 
  (h3 : c^2 = a^2 + b^2 - 2 * a * b * Real.cos C)
  (h4 : a + b = 2 * c) : c = 4 :=
by
  intros
  sorry

end problem_part_I_problem_part_II_l500_500960


namespace karen_kept_cookies_l500_500202

def total_cookies : ℕ := 50
def cookies_to_grandparents : ℕ := 8
def number_of_classmates : ℕ := 16
def cookies_per_classmate : ℕ := 2

theorem karen_kept_cookies (x : ℕ) 
  (H1 : x = total_cookies - (cookies_to_grandparents + number_of_classmates * cookies_per_classmate)) :
  x = 10 :=
by
  -- proof omitted
  sorry

end karen_kept_cookies_l500_500202


namespace fewer_scarves_l500_500256

theorem fewer_scarves (h s : ℕ) (hs : s = 3 * h) : 
  let normal_day := s * h
  let tiring_day := (s - 2) * (h - 3)
  normal_day - tiring_day = 11 * h - 6 := 
by 
  have h1 : normal_day = 3 * h * h := by rw [hs]
  have h2 : tiring_day = (3 * h - 2) * (h - 3) := by rw [hs]
  have h3 : normal_day - tiring_day = (3 * h * h) - ((3 * h - 2) * (h - 3)) := by rw [h1, h2]
  have h4 : (3 * h - 2) * (h - 3) = 3 * h * h - 11 * h + 6 := sorry
  rw [h3, h4]
  ring
  sorry

end fewer_scarves_l500_500256


namespace fit_pieces_correctly_result_l500_500665

theorem fit_pieces_correctly_result (p1 p2 p3 p4 p5 : ℤ) (h1 : p1 = 2) (h2 : p2 = 2) (h3 : p3 = 0) (h4 : p4 = -) (h5 : p5 = 100) :
  (2 * 10 + 0) - 102 = -82 :=
by
  sorry

end fit_pieces_correctly_result_l500_500665


namespace everton_college_calculators_l500_500443

theorem everton_college_calculators (total_cost : ℤ) (num_scientific_calculators : ℤ) 
  (cost_per_scientific : ℤ) (cost_per_graphing : ℤ) (total_scientific_cost : ℤ) 
  (num_graphing_calculators : ℤ) (total_graphing_cost : ℤ) (total_calculators : ℤ) :
  total_cost = 1625 ∧
  num_scientific_calculators = 20 ∧
  cost_per_scientific = 10 ∧
  cost_per_graphing = 57 ∧
  total_scientific_cost = num_scientific_calculators * cost_per_scientific ∧
  total_graphing_cost = num_graphing_calculators * cost_per_graphing ∧
  total_cost = total_scientific_cost + total_graphing_cost ∧
  total_calculators = num_scientific_calculators + num_graphing_calculators → 
  total_calculators = 45 :=
by
  intros
  sorry

end everton_college_calculators_l500_500443


namespace distinct_remainders_3n_l500_500220

theorem distinct_remainders_3n (n : ℕ) (hn : Odd n) :
  ∃ (a b : Fin n → ℤ), (∀ i : Fin n, a (i + 1) = 3 * (i + 1) - 2 ∧ b (i + 1) = 3 * (i + 1) - 3) ∧ 
  (∀ k : ℕ, 0 < k ∧ k < n → 
    (∀ i : Fin n, 
      let ai_1 := a i + a ((i + 1) % n)
      let ai_bi := a i + b i
      let bi_bik := b i + b ((i + k) % n)
      (ai_1 % (3 * n) ≠ ai_bi % (3 * n)) ∧ 
      (ai_1 % (3 * n) ≠ bi_bik % (3 * n)) ∧ 
      (ai_bi % (3 * n) ≠ bi_bik % (3 * n))))
  :=
sorry

end distinct_remainders_3n_l500_500220


namespace binom_20_4_l500_500750

theorem binom_20_4 : nat.choose 20 4 = 4845 :=
by
  sorry

end binom_20_4_l500_500750


namespace sum_of_factorials_last_two_digits_l500_500054

theorem sum_of_factorials_last_two_digits :
  let s := (1! + 2! + 3! + 4! + 5! + 6! + 7! + 8! + 9!) % 100 in
  s = 13 :=
  by
  sorry

end sum_of_factorials_last_two_digits_l500_500054


namespace base8_base6_eq_l500_500606

-- Defining the base representations
def base8 (A C : ℕ) := 8 * A + C
def base6 (C A : ℕ) := 6 * C + A

-- The main theorem stating that the integer is 47 in base 10 given the conditions
theorem base8_base6_eq (A C : ℕ) (hAC: base8 A C = base6 C A) (hA: A = 5) (hC: C = 7) : 
  8 * A + C = 47 :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end base8_base6_eq_l500_500606


namespace no_real_roots_of_equation_l500_500757

theorem no_real_roots_of_equation :
  ¬ ∃ x : ℝ, sqrt (x + 7) - sqrt (x - 5) + 2 = 0 :=
by sorry

end no_real_roots_of_equation_l500_500757


namespace log_equation_solution_l500_500055

theorem log_equation_solution (x : ℝ) (hx : 0 < x) :
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 ↔ (x = 4 ∨ x = 8) :=
by
  sorry

end log_equation_solution_l500_500055


namespace problem_1_problem_2_l500_500850

noncomputable def m (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x - cos x, 1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (cos x, 1 / 2)
noncomputable def f (x : ℝ) : ℝ := (m x).fst * (n x).fst + (m x).snd * (n x).snd

theorem problem_1 (k : ℤ) : ∀ x, f x = (1 / 2) * sin (2 * x - (π / 6)) ∧ 
    (∀ x, x ∈ Set.Icc (k * π - π / 6) (k * π + π / 3) → cos (2 * x - π / 6) > 0) :=
sorry

noncomputable def a : ℝ := 2 * sqrt 3
noncomputable def c : ℝ := 4
noncomputable def A : ℝ := π / 3

theorem problem_2 : f A = 1 →
  let b := 2 in
  let S := (1/2) * b * c * sin A in
  S = 2 * sqrt 3 :=
sorry

end problem_1_problem_2_l500_500850


namespace number_of_possible_values_of_t_l500_500224

-- Define the sequence according to the given problem
def sequence (n : ℕ) (t : ℝ) : ℝ :=
  if n = 0 then t
  else 4 * (sequence (n - 1) t) * (1 - (sequence (n - 1) t))

theorem number_of_possible_values_of_t :
  ∃ (t : ℝ) (k : ℕ), (sequence 1998 t = 0) ∧ (0 ≤ k ∧ k ≤ 2^1996) :=
sorry

end number_of_possible_values_of_t_l500_500224


namespace digits_divisibility_property_l500_500219

-- Definition: Example function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

-- Theorem: Prove the correctness of the given mathematical problem
theorem digits_divisibility_property:
  ∀ n : ℕ, (n = 18 ∨ n = 27 ∨ n = 45 ∨ n = 63) →
  (sum_of_digits n % 9 = 0) → (n % 9 = 0) := by
  sorry

end digits_divisibility_property_l500_500219


namespace findNAndConstantTerm_l500_500119

def expansionHasFiveTerms (n : ℕ) : Prop :=
  (x + 2)^n has exactly 5 terms

def constantTerm (n : ℕ) : ℕ :=
  if 4 - r = 0 then (Nat.choose 4 4) * 2 ^ 4 else 0

theorem findNAndConstantTerm (n : ℕ) (h : expansionHasFiveTerms n) : n = 4 ∧ constantTerm n = 16 :=
by
  sorry

end findNAndConstantTerm_l500_500119


namespace C_moves_along_segment_l500_500304

theorem C_moves_along_segment :
  ∀ (A B C P : Point) (hABC : Triangle ABC) (right_angle_C : ∠ACB = 90°)
    (slide_A_B_from_P : slides_along_sides A B P) (fixed_C : C = hABC.vertex) (right_triangle : RightTriangle ABC),
    moves_along_segment C :=
by
  sorry

end C_moves_along_segment_l500_500304


namespace locus_of_intersection_is_parallel_l500_500374

/-
  Question: What is the locus of the intersection points of the second tangents drawn to the circle k from M and M′ as M travels along t?
  Conditions:
  1. Circle k has a tangent line t.
  2. Tangent line t touches the circle at points A and M.
  3. M′ is the reflection of M over A.
  4. Second tangents are drawn to the circle k from M and M′.
  Answer: The line passing through C and parallel to OA.
-/

variable {k : Type} [metric_space k] [normed_group k] [normed_space ℝ k]
variable (O A M C : k)

-- Define point reflection property
def reflection (A M : k) : k := 2 • A - M

-- Define tangent condition
variable (tangent : Set k)
variable [is_tangent : is_tangent_line tangent k A M]

-- Define points intersection condition
variable (P : k) [is_intersection_point P tangent (reflection A M)]

-- The conjecture that the locus of intersection points P is the line through C parallel to OA
theorem locus_of_intersection_is_parallel (P : k) [condition1 : is_point_on_line P k]
  [condition2 : is_second_tangent_point P tangent k]
  (HC : is_point_on_line C k)
  (HOA : is_point_on_line O A) :
  is_line_parallel (line_through C P) (line_through O A) :=
sorry

end locus_of_intersection_is_parallel_l500_500374


namespace tan_alpha_solution_l500_500861

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l500_500861


namespace A_and_D_mutual_independent_l500_500315

-- Probability theory definitions and assumptions.
noncomputable def prob_1_6 : ℚ := 1 / 6
noncomputable def prob_5_36 : ℚ := 5 / 36
noncomputable def prob_6_36 : ℚ := 6 / 36
noncomputable def prob_1_36 : ℚ := 1 / 36

-- Definitions of events with their corresponding probabilities.
def event_A (P : ℚ) : Prop := P = prob_1_6
def event_B (P : ℚ) : Prop := P = prob_1_6
def event_C (P : ℚ) : Prop := P = prob_5_36
def event_D (P : ℚ) : Prop := P = prob_6_36

-- Intersection probabilities:
def intersection_A_C (P : ℚ) : Prop := P = 0
def intersection_A_D (P : ℚ) : Prop := P = prob_1_36
def intersection_B_C (P : ℚ) : Prop := P = prob_1_36
def intersection_C_D (P : ℚ) : Prop := P = 0

-- Mutual independence definition.
def mutual_independent (P_X : ℚ) (P_Y : ℚ) (P_intersect : ℚ) : Prop :=
  P_X * P_Y = P_intersect

-- Theorem to prove:
theorem A_and_D_mutual_independent :
  event_A prob_1_6 →
  event_D prob_6_36 →
  intersection_A_D prob_1_36 →
  mutual_independent prob_1_6 prob_6_36 prob_1_36 := 
by 
  intros hA hD hAD
  rw [event_A, event_D, intersection_A_D] at hA hD hAD
  exact hA.symm ▸ hD.symm ▸ hAD.symm 

#check A_and_D_mutual_independent

end A_and_D_mutual_independent_l500_500315


namespace c1_minus_c3_l500_500502

noncomputable def f (x c1 c2 c3 : ℝ) : ℝ :=
  (x^2 - 6*x + c1) * (x^2 - 6*x + c2) * (x^2 - 6*x + c3)

theorem c1_minus_c3 {c1 c2 c3 : ℝ} 
  (M : set ℝ)
  (hM : M = {x | f x c1 c2 c3 = 0})
  (hx : ∀ x ∈ M, x > 0)
  (hc : c1 ≥ c2 ∧ c2 ≥ c3)
  (hM_finite : M.finite)
  (hM_card : M.to_finset.card = 5)
  (hpos : ∀ x ∈ M, x ∈ set.univ) :
  c1 - c3 = 4 :=
sorry

end c1_minus_c3_l500_500502


namespace fraction_of_grid_is_5_over_42_l500_500016

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (6, 2)
def C : ℝ × ℝ := (5, 5)

-- Define the area of the grid
def area_of_grid : ℝ := 7 * 6

-- Calculate the coordinates differences
def x1 : ℝ := A.1
def y1 : ℝ := A.2
def x2 : ℝ := B.1
def y2 : ℝ := B.2
def x3 : ℝ := C.1
def y3 : ℝ := C.2

-- Calculate the area of the triangle using Shoelace Theorem
def area_of_triangle : ℝ := 0.5 * |(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))|

-- Calculate the fraction of the grid covered by the triangle
def fraction_covered : ℝ := area_of_triangle / area_of_grid

-- Prove that the fraction of the grid covered by the triangle is 5/42
theorem fraction_of_grid_is_5_over_42 : fraction_covered = 5 / 42 :=
by
  sorry

end fraction_of_grid_is_5_over_42_l500_500016


namespace jessica_money_left_l500_500986

theorem jessica_money_left : 
  let initial_amount := 11.73
  let amount_spent := 10.22
  initial_amount - amount_spent = 1.51 :=
by
  sorry

end jessica_money_left_l500_500986


namespace incorrect_rep_decimal_l500_500772

variables {t u : ℕ} {R K : ℚ}

def M : ℚ := R * 10^(-t) + (K : ℚ) * 10^(-t-u) / (1 - 10^-u)

theorem incorrect_rep_decimal (M = R * 10^(-t) + (K : ℚ) * 10^(-t-u) / (1 - 10^-u)) :
  10^t * (10^u - 1) * M ≠ K * (R - 1) :=
sorry

end incorrect_rep_decimal_l500_500772


namespace calculate_principal_sum_l500_500609

noncomputable def principal_sum (r t difference : ℝ) : ℝ :=
  let P := difference / ((1 + r / 100)^t - 1 - r * t / 100)
  P

theorem calculate_principal_sum :
  principal_sum 12 3 824 ≈ 18334.63 :=
by 
  -- Definitions
  let r: ℝ := 12
  let t: ℝ := 3
  let difference: ℝ := 824
    
  -- Application of the principal_sum calculation
  let approxP := difference / ((1 + r / 100)^t - 1 - r * t / 100) 
  -- Approximate equality check
  show abs (approxP - 18334.63) < 0.01, sorry

end calculate_principal_sum_l500_500609


namespace quadratic_increasing_implies_m_gt_1_l500_500844

theorem quadratic_increasing_implies_m_gt_1 (m : ℝ) (x : ℝ) 
(h1 : x > 1) 
(h2 : ∀ x, (y = x^2 + (m-3) * x + m + 1) → (∀ z > x, y < z^2 + (m-3) * z + m + 1)) 
: m > 1 := 
sorry

end quadratic_increasing_implies_m_gt_1_l500_500844


namespace third_circle_radius_l500_500694

-- Define the initial conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def Circle1 : Circle := ⟨(0, 0), 2⟩
def Circle2 : Circle := ⟨(0, 6), 6⟩

-- External tangency condition between two circles
def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.fst - c2.center.fst) ^ 2 + (c1.center.snd - c2.center.snd) ^ 2 = (c1.radius + c2.radius) ^ 2

-- Tangency condition for the third circle with both circles
def tangent_to_both (c3 : Circle) (c1 c2 : Circle) : Prop :=
  (c3.center.fst - c1.center.fst) ^ 2 + (c3.center.snd - c1.center.snd) ^ 2 = (c3.radius + c1.radius) ^ 2 ∧
  (c3.center.fst - c2.center.fst) ^ 2 + (c3.center.snd - c2.center.snd) ^ 2 = (c3.radius + c2.radius) ^ 2

-- The third circle should be tangent to one of their common external tangents
-- For simplicity, assume the center of the third circle lies on the y-axis similar to the explanation in the problem
def Circle3 : Circle := ⟨(0, r), r⟩

-- Proof statement
theorem third_circle_radius :
  externally_tangent Circle1 Circle2 →
  (∃ r : ℝ, r = 3 ∧ tangent_to_both (Circle ⟨(0, r), r⟩) Circle1 Circle2) :=
by
  sorry

end third_circle_radius_l500_500694


namespace positive_whole_numbers_with_fourth_root_less_than_six_l500_500146

theorem positive_whole_numbers_with_fourth_root_less_than_six :
  {n : ℕ | n > 0 ∧ (n : ℝ)^(1/4) < 6}.to_finset.card = 1295 :=
sorry

end positive_whole_numbers_with_fourth_root_less_than_six_l500_500146


namespace binom_20_4_l500_500751

theorem binom_20_4 : nat.choose 20 4 = 4845 :=
by
  sorry

end binom_20_4_l500_500751


namespace line_intersects_hyperbola_length_l500_500004

theorem line_intersects_hyperbola_length :
    ∀ (A B : ℝ × ℝ), let F2 := (3, 0)
    let line := λ x : ℝ, (x - 3) * (Real.sqrt 3 / 3)
    let hyperbola := λ x y : ℝ, (x^2 / 3) - (y^2 / 6) = 1
    hyperbola A.1 A.2 →
    hyperbola B.1 B.2 →
    A.2 = line A.1 →
    B.2 = line B.1 →
    A ≠ B →
    dist A B = 16 / 5 * Real.sqrt 3 := 
by 
  sorry

end line_intersects_hyperbola_length_l500_500004


namespace chord_length_tangent_line_l500_500806

theorem chord_length_tangent_line
  (x y : ℝ)
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : x + 2*y = 3)
  (h4 : x * y ≤ 9 / 8) :
  let P := (x, y)
  let center := (1/2 : ℝ, -1/4 : ℝ)
  let circle := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 1/2}
  length_of_chord := sqrt 6 / 2 := by
    sorry

end chord_length_tangent_line_l500_500806


namespace termination_condition_correct_l500_500827

theorem termination_condition_correct :
  ∃ i S : ℕ, i = 12 ∧ S = 1 ∧ 
  (∀ S i S' i', (S' = S * i ∧ i' = i - 1) → 
    (i' = 8 → (S' = 11880))) :=
by
  existsi (12 : ℕ)
  existsi (1 : ℕ)
  split
  { refl }
  split
  { refl }
  intros S i S' i' h h_end
  cases h with h1 h2
  rw [h1, h2]
  intro h_eq
  rw [← h_eq]
  sorry

end termination_condition_correct_l500_500827


namespace license_plate_count_l500_500150

theorem license_plate_count :
  let letters := 26
  let digits := 10
  let total_count := letters * (letters - 1) + letters
  total_count * digits = 6760 :=
by sorry

end license_plate_count_l500_500150


namespace tangent_line_eq_range_f_l500_500130

-- Given the function f(x) = 2x^3 - 9x^2 + 12x
def f(x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x

-- (1) Prove that the equation of the tangent line to y = f(x) at (0, f(0)) is y = 12x
theorem tangent_line_eq : ∀ x, x = 0 → f x = 0 → (∃ m, m = 12 ∧ (∀ y, y = 12 * x)) :=
by
  sorry

-- (2) Prove that the range of f(x) on the interval [0, 3] is [0, 9]
theorem range_f : Set.Icc 0 9 = Set.image f (Set.Icc (0 : ℝ) 3) :=
by
  sorry

end tangent_line_eq_range_f_l500_500130


namespace probability_of_prime_sum_l500_500646

def is_prime_sum (a b c : ℕ) : Prop :=
  Nat.Prime (a + b + c)

def valid_die_roll (n : ℕ) : Prop :=
  n >= 1 ∧ n <= 6

noncomputable def prime_probability : ℚ :=
  let outcomes := [(a, b, c) | a in Fin 6, b in Fin 6, c in Fin 6, 1 ≤ a + 1, 1 ≤ b + 1, 1 ≤ c + 1]
  let prime_outcomes := outcomes.filter (λ (a, b, c), is_prime_sum (a + 1) (b + 1) (c + 1))
  prime_outcomes.length / outcomes.length

theorem probability_of_prime_sum :
  prime_probability = 37 / 216 := sorry

end probability_of_prime_sum_l500_500646


namespace converse_l500_500607

variables {x : ℝ}

def P (x : ℝ) : Prop := x < 0
def Q (x : ℝ) : Prop := x^2 > 0

theorem converse (h : Q x) : P x :=
sorry

end converse_l500_500607


namespace younger_brother_silver_fraction_l500_500278

def frac_silver (x y : ℕ) : ℚ := (100 - x / 7 ) / y

theorem younger_brother_silver_fraction {x y : ℕ} 
    (cond1 : x / 5 + y / 7 = 100) 
    (cond2 : x / 7 + (100 - x / 7) = 100) : 
    frac_silver x y = 5 / 14 := 
sorry

end younger_brother_silver_fraction_l500_500278


namespace function_satisfies_condition_l500_500445

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 - x + 4)

theorem function_satisfies_condition :
  ∀ (x : ℝ), 2 * f (1 - x) + 1 = x * f x :=
by
  intro x
  unfold f
  sorry

end function_satisfies_condition_l500_500445


namespace range_of_x_l500_500973

noncomputable def y (x : ℝ) : ℝ := (x + 2) / (x - 1)

theorem range_of_x : ∀ x : ℝ, (y x ≠ 0) → x ≠ 1 := by
  intro x h
  sorry

end range_of_x_l500_500973


namespace consistent_scale_l500_500237

-- Conditions definitions

def dist_gardensquare_newtonsville : ℕ := 3  -- in inches
def dist_newtonsville_madison : ℕ := 4  -- in inches
def speed_gardensquare_newtonsville : ℕ := 50  -- mph
def time_gardensquare_newtonsville : ℕ := 2  -- hours
def speed_newtonsville_madison : ℕ := 60  -- mph
def time_newtonsville_madison : ℕ := 3  -- hours

-- Actual distances calculated
def actual_distance_gardensquare_newtonsville : ℕ := speed_gardensquare_newtonsville * time_gardensquare_newtonsville
def actual_distance_newtonsville_madison : ℕ := speed_newtonsville_madison * time_newtonsville_madison

-- Prove the scale is consistent across the map
theorem consistent_scale :
  actual_distance_gardensquare_newtonsville / dist_gardensquare_newtonsville =
  actual_distance_newtonsville_madison / dist_newtonsville_madison :=
by
  sorry

end consistent_scale_l500_500237


namespace walter_age_end_of_2000_l500_500028

theorem walter_age_end_of_2000 :
  ∃ (x : ℝ), (3988 - 4 * x = 3858) ∧ (x + 6 = 38.5) :=
begin
  -- Let Walter's age in 1994 be x
  let x := 32.5,
  use x,
  split,
  { 
    -- At the end of 1994, Walter was a third as old as his grandmother.
    -- The sum of the years in which they were born was 3858.
    -- Solve 3988 - 4x = 3858
    calc 3988 - 4 * x = 3988 - 4 * 32.5 : by rw [x]
               ... = 3988 - 130 : by simp
               ... = 3858 : by norm_num,
  },
  {
    -- Age at the end of 2000 is Walter's age in 1994 plus 6
    calc x + 6 = 32.5 + 6 : by rw [x]
           ... = 38.5 : by norm_num,
  }
end

end walter_age_end_of_2000_l500_500028


namespace solution_set_l500_500503

def f (x : ℝ) : ℝ :=
if x ≥ 0 then 1 else -1

theorem solution_set : 
  { x : ℝ | x + (x + 2) * f (x + 2) ≤ 5 } = 
  set.Iic 3/2 :=
sorry

end solution_set_l500_500503


namespace percentage_caterpillars_failed_l500_500988

-- Definitions of the conditions
def jars : ℕ := 4
def caterpillars_per_jar : ℕ := 10
def total_caterpillars : ℕ := jars * caterpillars_per_jar
def price_per_butterfly : ℝ := 3
def total_money_made : ℝ := 72
def butterflies_sold : ℝ := total_money_made / price_per_butterfly
def caterpillars_failed : ℝ := total_caterpillars - butterflies_sold

-- Proof problem as a Lean theorem
theorem percentage_caterpillars_failed : 
  (caterpillars_failed / total_caterpillars) * 100 = 40 :=
by
  sorry

end percentage_caterpillars_failed_l500_500988


namespace minimum_value_f_minimum_value_f_achieved_l500_500545

noncomputable def star (a b : ℝ) : ℝ := a * b + a + b

def f (x : ℝ) : ℝ := star (Real.exp x) (1 / Real.exp x)

theorem minimum_value_f : ∀ x : ℝ, f x ≥ 3 :=
by
  intros x
  have h : f x = 1 + Real.exp x + 1 / Real.exp x := by
    unfold f star
    have exp_pos : Real.exp x ≠ 0 := Real.exp_ne_zero x
    calc
      star (Real.exp x) (1 / Real.exp x) = Real.exp x * (1 / Real.exp x) + Real.exp x + 1 / Real.exp x := by rw star
      ... = 1 + Real.exp x + 1 / Real.exp x := by rw [mul_div_cancel' _ exp_pos]
  calc
    1 + Real.exp x + 1 / Real.exp x ≥ 3 := by sorry

theorem minimum_value_f_achieved : f 0 = 3 :=
by
  unfold f star
  have exp_zero : Real.exp 0 = 1 := Real.exp_zero
  calc
    star (Real.exp 0) (1 / Real.exp 0) = 1 * 1 + 1 + 1 := by rw [exp_zero, div_one]; rfl
    ... = 3 := by norm_num

end minimum_value_f_minimum_value_f_achieved_l500_500545


namespace tan_alpha_value_l500_500890

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l500_500890


namespace john_has_more_black_pens_than_red_l500_500555

/-- Definition of the given conditions -/
variables 
  (total_pens : ℕ)
  (blue_pens : ℕ)
  (black_pens : ℕ)
  (red_pens : ℕ)

def has_exactly_31_pens := total_pens = 31
def blue_black_red_sum := blue_pens + black_pens + red_pens = total_pens
def twice_as_many_blue_as_black := blue_pens = 2 * black_pens
def blue_pens_18 := blue_pens = 18

/-- Theorem stating the proof problem -/
theorem john_has_more_black_pens_than_red 
  (h1 : has_exactly_31_pens total_pens)
  (h2 : blue_black_red_sum blue_pens black_pens red_pens total_pens)
  (h3 : twice_as_many_blue_as_black blue_pens black_pens)
  (h4 : blue_pens_18 blue_pens) :
  (black_pens - red_pens) = 5 :=
sorry

end john_has_more_black_pens_than_red_l500_500555


namespace A_and_D_mutual_independent_l500_500316

-- Probability theory definitions and assumptions.
noncomputable def prob_1_6 : ℚ := 1 / 6
noncomputable def prob_5_36 : ℚ := 5 / 36
noncomputable def prob_6_36 : ℚ := 6 / 36
noncomputable def prob_1_36 : ℚ := 1 / 36

-- Definitions of events with their corresponding probabilities.
def event_A (P : ℚ) : Prop := P = prob_1_6
def event_B (P : ℚ) : Prop := P = prob_1_6
def event_C (P : ℚ) : Prop := P = prob_5_36
def event_D (P : ℚ) : Prop := P = prob_6_36

-- Intersection probabilities:
def intersection_A_C (P : ℚ) : Prop := P = 0
def intersection_A_D (P : ℚ) : Prop := P = prob_1_36
def intersection_B_C (P : ℚ) : Prop := P = prob_1_36
def intersection_C_D (P : ℚ) : Prop := P = 0

-- Mutual independence definition.
def mutual_independent (P_X : ℚ) (P_Y : ℚ) (P_intersect : ℚ) : Prop :=
  P_X * P_Y = P_intersect

-- Theorem to prove:
theorem A_and_D_mutual_independent :
  event_A prob_1_6 →
  event_D prob_6_36 →
  intersection_A_D prob_1_36 →
  mutual_independent prob_1_6 prob_6_36 prob_1_36 := 
by 
  intros hA hD hAD
  rw [event_A, event_D, intersection_A_D] at hA hD hAD
  exact hA.symm ▸ hD.symm ▸ hAD.symm 

#check A_and_D_mutual_independent

end A_and_D_mutual_independent_l500_500316


namespace no_real_x_solution_l500_500427

open Real

-- Define the conditions.
def log_defined (x : ℝ) : Prop :=
  0 < x + 5 ∧ 0 < x - 3 ∧ 0 < x^2 - 7*x - 18

-- Define the equation to prove.
def log_eqn (x : ℝ) : Prop :=
  log (x + 5) + log (x - 3) = log (x^2 - 7*x - 18)

-- The mathematicall equivalent proof problem.
theorem no_real_x_solution : ¬∃ x : ℝ, log_defined x ∧ log_eqn x :=
by
  sorry

end no_real_x_solution_l500_500427


namespace rectangular_prism_unit_cubes_l500_500393

theorem rectangular_prism_unit_cubes (a b c : ℕ) (h : a + b + c = 12)
  (hab : 2 * (a * b + b * c + a * c) = 6 * (a * b * c) / 3) :
  a = 3 ∧ b = 4 ∧ c = 5 :=
by {
  split,
  all_goals { try repeat { assumption } },
  sorry
}

end rectangular_prism_unit_cubes_l500_500393


namespace value_of_fraction_l500_500168

variables {a_1 q : ℝ}

-- Define the conditions and the mathematical equivalent of the problem.
def geometric_sequence (a_1 q : ℝ) (h_pos : a_1 > 0 ∧ q > 0) :=
  2 * a_1 + a_1 * q = a_1 * q^2

theorem value_of_fraction (h_pos : a_1 > 0 ∧ q > 0) (h_geom : geometric_sequence a_1 q h_pos) :
  (a_1 * q^3 + a_1 * q^4) / (a_1 * q^2 + a_1 * q^3) = 2 :=
sorry

end value_of_fraction_l500_500168


namespace minimum_killed_gangsters_l500_500266

theorem minimum_killed_gangsters (points : Finset (EuclideanSpace ℝ (Fin 2))) (h_distinct : ∀ (p1 p2 : EuclideanSpace ℝ (Fin 2)), p1 ≠ p2 → ∃ d, d = dist p1 p2) (h_card : points.card = 10) :
  ∃ (S : Finset (EuclideanSpace ℝ (Fin 2))), S.card ≥ 3 ∧ ∀ p ∈ points, ∃ q ∈ points, q ≠ p ∧ dist p q = (finset.min' (points.erase p) (dist_mem_finset_of_distinct h_distinct p)) :=
sorry

end minimum_killed_gangsters_l500_500266


namespace short_bingo_first_column_possibilities_l500_500536

theorem short_bingo_first_column_possibilities : 
  ∃ (choices : Finset (Finset ℕ)), choices.card = 2520 ∧ 
  (∀ c ∈ choices, c.card = 5 ∧ (∀ x ∈ c, 1 ≤ x ∧ x ≤ 7) ∧ ∀ x y ∈ c, x ≠ y) :=
sorry

end short_bingo_first_column_possibilities_l500_500536


namespace range_of_m_l500_500957

noncomputable def has_two_distinct_zeros (m : ℝ) : Prop :=
  let Δ := m^2 - 4 * 1 * 1 in
  Δ > 0

theorem range_of_m (m : ℝ) : has_two_distinct_zeros m ↔ m ∈ (Set.Ioo (2 : ℝ) ⊤ ∪ Set.Ioo ⊥ (-2)) :=
  sorry

end range_of_m_l500_500957


namespace length_of_fourth_side_l500_500698

-- Declare the problem setting and conditions in Lean 4
variable {a b c d x y u v : ℝ}

-- Given conditions
def is_convex_quadrilateral (a b c d : ℝ) : Prop := true  -- Convexity condition (implicit)
def side_lengths (a b c : ℝ) : Prop := (a > 0 ∧ b > 0 ∧ c > 0)
def diagonals_perpendicular (x y u v : ℝ) : Prop :=
  (x^2 + v^2 = a^2) ∧ (y^2 + u^2 = c^2) ∧
  (x^2 + u^2 = b^2) ∧ (y^2 + v^2 = d^2)

-- Given side lengths
axiom side_lengths_given : side_lengths 8 1 4

-- Correct answer to prove
theorem length_of_fourth_side :
  diagonals_perpendicular x y u v → d = 7 := 
sorry

end length_of_fourth_side_l500_500698


namespace part1_a_n_part1_b_n_part2_T_n_l500_500485

theorem part1_a_n (n : ℕ) (hn : 1 ≤ n) :
    ∃ q, (3 ^ 3 = 27) ∧ (a_n  = 3 ^ (n - 1)) :=
sorry

theorem part1_b_n (n : ℕ) (hn : 1 ≤ n) :
    ∃ d, (5 * 3 + (5 * 4 / 2) * 2 = 35) ∧ (b_n = 2 * n + 1) :=
sorry

theorem part2_T_n (n : ℕ) (hn : 1 ≤ n) :
    ∑ (k : ℕ) in finset.range n, (a_k.succ * b_k.succ) = n * 3 ^ n :=
sorry

end part1_a_n_part1_b_n_part2_T_n_l500_500485


namespace largest_three_digit_multiple_of_8_with_digit_sum_24_l500_500659

theorem largest_three_digit_multiple_of_8_with_digit_sum_24 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (8 ∣ n) ∧ nat.digits 10 n.sum = 24 ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ (8 ∣ m) ∧ nat.digits 10 m.sum = 24 → m ≤ n :=
begin
  sorry
end

end largest_three_digit_multiple_of_8_with_digit_sum_24_l500_500659


namespace keith_picked_zero_pears_l500_500573

def apples_picked_mike : ℝ := 7.0
def apples_picked_keith : ℝ := 6.0
def apples_eaten_nancy : ℝ := 3.0
def apples_left : ℝ := 10.0

theorem keith_picked_zero_pears :
  let total_apples_picked := apples_picked_mike + apples_picked_keith in
  let apples_after_eaten := total_apples_picked - apples_eaten_nancy in
  apples_after_eaten = apples_left → 
  0 = 0 :=
by
  intros h
  sorry

end keith_picked_zero_pears_l500_500573


namespace tan_alpha_solution_l500_500857

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l500_500857


namespace center_of_mass_distance_correct_l500_500332

noncomputable def center_of_mass_distance : Real :=
let m1 : Real := 100 / 1000 -- converting g to kg
let m2 : Real := 200 / 1000 -- converting g to kg
let m3 : Real := 400 / 1000 -- converting g to kg
let x1 : Real := 0 -- position of m1
let x2 : Real := 0.5 -- position of m2
let x3 : Real := x2 + 2 -- position of m3
let total_mass : Real := m1 + m2 + m3
let xcm : Real := (m1 * x1 + m2 * x2 + m3 * x3) / total_mass
xcm = 1.57

-- The theorem statement
theorem center_of_mass_distance_correct : center_of_mass_distance = 1.57 := sorry

end center_of_mass_distance_correct_l500_500332


namespace minimum_sum_of_fractions_l500_500208

-- Definitions of the conditions
def is_digit (n : ℕ) : Prop := n ≤ 9
def is_prime_digit (n : ℕ) : Prop := is_digit n ∧ n ∈ {2, 3, 5, 7}

-- Problem statement
theorem minimum_sum_of_fractions : 
  ∃ (A B C D : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    is_digit A ∧ is_prime_digit B ∧ is_digit C ∧ is_prime_digit D ∧
    A / B + C / D = 1 / 5 :=
by sorry

end minimum_sum_of_fractions_l500_500208


namespace polar_to_rectangular_l500_500758

theorem polar_to_rectangular (r θ : ℝ) (h1 : r = 3 * Real.sqrt 2) (h2 : θ = Real.pi / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (3, 3) :=
by
  -- Proof goes here
  sorry

end polar_to_rectangular_l500_500758


namespace valid_pairs_count_l500_500082
open Nat

-- Define the function to check if a number has the digit 0 or 5
def hasZeroOrFive (n : ℕ) : Prop := 
  ∃ d, d = 0 ∨ d = 5 ∧ toDigits n ≈ list (fin 10) ∧ d ∈ toDigits n

-- Define the function to compute the number of valid ordered pairs
def countValidPairs : ℕ := 
  ∑ i in range(1, 2000), if !(hasZeroOrFive i) && !(hasZeroOrFive (2000 - i)) then 1 else 0

theorem valid_pairs_count :
  countValidPairs = 820 :=
sorry

end valid_pairs_count_l500_500082


namespace odd_multiple_of_3_l500_500677

theorem odd_multiple_of_3 (M P S : ℕ)
  (h1 : ∀ n, n ∈ M -> n ∈ P) 
  (h2 : S ∈ M) 
  (h3 : M = {x | x % 9 = 0}) 
  (h4 : P = {y | y % 3 = 0}) 
  (h5 : ∃ k, S = 2 * k + 1): S ∈ P := 
by
  sorry

end odd_multiple_of_3_l500_500677


namespace congruent_triangles_value_of_x_l500_500821

-- Definition of the side lengths of the triangles
def side_lengths_ABC : List ℕ := [3, 4, 5]
def side_lengths_DEF (x : ℕ) : List ℕ := [3, 3 * x - 2, 2 * x + 1]

-- The congruence condition
def are_congruent (sides1 sides2 : List ℕ) : Prop :=
  sides1 = sides2

-- The theorem to be proven
theorem congruent_triangles_value_of_x {x : ℕ} (h : are_congruent side_lengths_ABC (side_lengths_DEF x)) : x = 2 :=
sory

end congruent_triangles_value_of_x_l500_500821


namespace positive_whole_numbers_with_fourth_root_less_than_six_l500_500145

theorem positive_whole_numbers_with_fourth_root_less_than_six :
  {n : ℕ | n > 0 ∧ (n : ℝ)^(1/4) < 6}.to_finset.card = 1295 :=
sorry

end positive_whole_numbers_with_fourth_root_less_than_six_l500_500145


namespace toms_dog_is_12_l500_500648

def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := toms_rabbit_age * 3

theorem toms_dog_is_12 : toms_dog_age = 12 :=
by
  sorry

end toms_dog_is_12_l500_500648


namespace number_of_tickets_bought_l500_500707

noncomputable def ticketCost : ℕ := 5
noncomputable def popcornCost : ℕ := (80 * ticketCost) / 100
noncomputable def sodaCost : ℕ := (50 * popcornCost) / 100
noncomputable def totalSpent : ℕ := 36
noncomputable def numberOfPopcorns : ℕ := 2 
noncomputable def numberOfSodas : ℕ := 4

theorem number_of_tickets_bought : 
  (totalSpent - (numberOfPopcorns * popcornCost + numberOfSodas * sodaCost)) = 4 * ticketCost :=
by
  sorry

end number_of_tickets_bought_l500_500707


namespace different_prime_factors_mn_is_five_l500_500550

theorem different_prime_factors_mn_is_five {m n : ℕ} 
  (m_prime_factors : ∃ (p_1 p_2 p_3 p_4 : ℕ), True)  -- m has 4 different prime factors
  (n_prime_factors : ∃ (q_1 q_2 q_3 : ℕ), True)  -- n has 3 different prime factors
  (gcd_m_n : Nat.gcd m n = 15) : 
  (∃ k : ℕ, k = 5 ∧ (∃ (x_1 x_2 x_3 x_4 x_5 : ℕ), True)) := sorry

end different_prime_factors_mn_is_five_l500_500550


namespace population_after_ten_years_l500_500630

-- Define the initial population and constants
def initial_population : ℕ := 100000
def birth_increase_rate : ℝ := 0.6
def emigration_per_year : ℕ := 2000
def immigration_per_year : ℕ := 2500
def years : ℕ := 10

-- Proving the total population at the end of 10 years
theorem population_after_ten_years :
  initial_population + (initial_population * birth_increase_rate).to_nat +
  (immigration_per_year * years - emigration_per_year * years) = 165000 := by
sorry

end population_after_ten_years_l500_500630


namespace octahedron_angles_sum_l500_500106

/-- Given a centrally symmetric octahedron \(ABC'A'B'C'\) 
    such that the sums of the planar angles at each vertex 
    of the octahedron are equal to \(240^\circ\). -/
theorem octahedron_angles_sum (A B C A' B' C' : Type)
  (h : centrally_symmetric_octahedron ABC'A'B'C') :
  (∀ v ∈ {A, B, C, A', B', C'}, sum_of_plane_angles v = 240) :=
sorry

end octahedron_angles_sum_l500_500106


namespace algebraic_expression_correct_l500_500468

theorem algebraic_expression_correct (a b : ℝ) (h : a = 7 - 3 * b) : a^2 + 6 * a * b + 9 * b^2 = 49 := 
by sorry

end algebraic_expression_correct_l500_500468


namespace tan_alpha_solution_l500_500859

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l500_500859


namespace elsa_lost_3_marbles_at_breakfast_l500_500440

variable {x : ℕ} -- x is the number of marbles Elsa lost at breakfast

def initial_marbles : ℕ := 40
def after_breakfast (x : ℕ) : ℕ := initial_marbles - x
def after_giving_susie (x : ℕ) : ℕ := after_breakfast x - 5
def after_receiving_from_mom (x : ℕ) : ℕ := after_giving_susie x + 12
def after_receiving_back_from_susie (x : ℕ) : ℕ := after_receiving_from_mom x + (2 * 5)
def final_marbles (x : ℕ) : ℕ := after_receiving_back_from_susie x

theorem elsa_lost_3_marbles_at_breakfast : final_marbles x = 54 → x = 3 :=
by
  intro h
  have h_eq : initial_marbles - x + 17 = 54 := by
    calc
    initial_marbles - x + 17 = 40 - x + 17 : rfl
                         ... = 57 - x     : by simp only [add_comm, add_left_comm, nat.add_sub_assoc]
    
  rw h_eq at h
  linarith

end elsa_lost_3_marbles_at_breakfast_l500_500440


namespace independence_of_A_and_D_l500_500324

noncomputable def balls : Finset ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def draw_one : ℕ := (1 : ℕ)
noncomputable def draw_two : ℕ := (2 : ℕ)

def event_A : ℕ → Prop := λ n, n = 1
def event_B : ℕ → Prop := λ n, n = 2
def event_C : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 8
def event_D : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 7

def prob (event : ℕ → Prop) : ℚ := 1 / 6
def joint_prob (event1 event2 : ℕ → Prop) : ℚ := (1 / 36)

theorem independence_of_A_and_D :
  joint_prob (λ n, event_A n) (λ n, event_D (draw_one, draw_two)) = prob event_A * prob (λ n, event_D (draw_one, draw_two)) :=
by
  sorry

end independence_of_A_and_D_l500_500324


namespace cell_phone_bill_l500_500692

-- Definitions
def base_cost : ℝ := 20
def cost_per_text : ℝ := 0.05
def cost_per_extra_minute : ℝ := 0.10
def texts_sent : ℕ := 100
def hours_talked : ℝ := 30.5
def included_hours : ℝ := 30

-- Calculate extra minutes used
def extra_minutes : ℝ := (hours_talked - included_hours) * 60

-- Total cost calculation
def total_cost : ℝ := 
  base_cost + 
  (texts_sent * cost_per_text) + 
  (extra_minutes * cost_per_extra_minute)

-- Proof problem statement
theorem cell_phone_bill : total_cost = 28 := by
  sorry

end cell_phone_bill_l500_500692


namespace beyonce_total_songs_l500_500031

theorem beyonce_total_songs :
  let singles := 20
  let albums := 6
  let songs_per_cd_1 := 22
  let songs_per_cd_2 := 18
  let songs_per_cd_3 := 15
  let total_songs_albums := albums * (songs_per_cd_1 + songs_per_cd_2 + songs_per_cd_3)
  let total_songs := singles + total_songs_albums
  in total_songs = 350 :=
by 
  sorry

end beyonce_total_songs_l500_500031


namespace polynomial_not_separable_l500_500585

theorem polynomial_not_separable (f g : Polynomial ℂ) :
  (∀ x y : ℂ, f.eval x * g.eval y = x^200 * y^200 + 1) → False :=
sorry

end polynomial_not_separable_l500_500585


namespace original_number_l500_500005

-- Define the original statement and conditions
theorem original_number (x : ℝ) (h : 3 * (2 * x + 9) = 81) : x = 9 := by
  -- Sorry placeholder stands for the proof steps
  sorry

end original_number_l500_500005


namespace remainder_when_divided_by_30_l500_500263

theorem remainder_when_divided_by_30 (y : ℤ)
  (h1 : 4 + y ≡ 9 [ZMOD 8])
  (h2 : 6 + y ≡ 8 [ZMOD 27])
  (h3 : 8 + y ≡ 27 [ZMOD 125]) :
  y ≡ 4 [ZMOD 30] :=
sorry

end remainder_when_divided_by_30_l500_500263


namespace triangle_construction_l500_500419

noncomputable
def construct_triangle (m_a m_b m_c : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    let d_A := dist A B, d_B := dist B C, d_C := dist C A in
    (m_a = (d_A / 2)) ∧
    (m_b = (d_B / 2)) ∧
    (m_c = (d_C / 2))

theorem triangle_construction (m_a m_b m_c : ℝ) : construct_triangle m_a m_b m_c :=
sorry

end triangle_construction_l500_500419


namespace original_ratio_l500_500343

theorem original_ratio (x y : ℕ) (h1 : y = 15) (h2 : x + 10 = y) : x / y = 1 / 3 :=
by
  sorry

end original_ratio_l500_500343


namespace hexagon_area_l500_500002

theorem hexagon_area (a b : ℝ) (angle : ℝ) (h1 : a = 1) (h2 : b = sqrt 3) (h3 : angle = 120) : 
  ∃ A : ℝ, A = 3 + sqrt 3 ∧ 
    let sides := [a, b, a, b, a, b] in
    let angles := [angle, angle, angle, angle, angle, angle] in
    hexagon_area sides angles = A :=
begin
  -- some work
  sorry
end

end hexagon_area_l500_500002


namespace complement_subset_l500_500097

universe u

variables {U : Type u} (M N : Set U)

theorem complement_subset (U : Set U) (M N : Set U) (h₁ : M ∩ N = N) (h₂ : M ⊆ U) (h₃ : N ⊆ U) :
  Mᶜ ⊆ Nᶜ :=
  sorry

end complement_subset_l500_500097


namespace translation_correct_l500_500292

def parabola1 (x : ℝ) : ℝ := -2 * (x + 2)^2 + 3
def parabola2 (x : ℝ) : ℝ := -2 * (x - 1)^2 - 1

theorem translation_correct :
  ∀ x : ℝ, parabola2 (x - 3) = parabola1 x - 4 :=
by
  sorry

end translation_correct_l500_500292


namespace homogeneous_eq_determines_ratio_l500_500255

theorem homogeneous_eq_determines_ratio (n : ℕ) (x y : ℝ) 
  (a : Fin (n + 1) → ℝ) (h : a 0 * x^n + a 1 * x^(n-1) * y + a 2 * x^(n-2) * y^2 + 
                               ∑ i in Finset.range (n-1), a (i+3) * x^(n-3-i) * y^(i+3) + a n * y^n = 0) :
  ∃ t : ℝ, t = x / y ∧ ∑ i in Finset.range (n+1), a i * t^(n-i) = 0 :=
sorry

end homogeneous_eq_determines_ratio_l500_500255


namespace count_5_more_than_9_l500_500438

def count_digit (d : ℕ) (n : ℕ) : ℕ :=
(n.digits 10).count d

theorem count_5_more_than_9 (p : ℕ) (h : 1 ≤ p ∧ p ≤ 600) : 
  (∑ i in finset.range 601, count_digit 5 i) - 
  (∑ i in finset.range 601, count_digit 9 i) = 101 :=
by sorry

end count_5_more_than_9_l500_500438


namespace find_b1_b7_b10_value_l500_500679

open Classical

theorem find_b1_b7_b10_value
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_arith_seq : ∀ n m : ℕ, a n + a m = 2 * a ((n + m) / 2))
  (h_geom_seq : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r)
  (a3_condition : a 3 - 2 * (a 6)^2 + 3 * a 7 = 0)
  (b6_a6_eq : b 6 = a 6)
  (non_zero_seq : ∀ n : ℕ, a n ≠ 0) :
  b 1 * b 7 * b 10 = 8 := 
by 
  sorry

end find_b1_b7_b10_value_l500_500679


namespace triangle_properties_l500_500194

/--
In a triangle ABC with sides opposite to angles A, B, and C being a, b, and c respectively,
prove the required properties given the conditions:
1. (sqrt(3) * cos 10 - sin 10) * cos (B + 35) = sin 80 proves B = 15
2. Given 2 * b * cos A = c - b, and AD = 2 where D is the intersection of the angle bisector of A with BC, prove c = sqrt(6) + sqrt(2)
--/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (D : point)
  (h1 : (sqrt 3 * cos 10 - sin 10) * cos (B + 35) = sin 80)
  (h2 : 2 * b * cos A = c - b)
  (h3 : angle_bisector_intersects_AD A B C D = 2) : 
  B = 15 ∧ c = (sqrt 6 + sqrt 2) :=
  sorry

end triangle_properties_l500_500194


namespace stratified_sampling_l500_500376

theorem stratified_sampling
  (total_products : ℕ)
  (sample_size : ℕ)
  (workshop_products : ℕ)
  (h1 : total_products = 2048)
  (h2 : sample_size = 128)
  (h3 : workshop_products = 256) :
  (workshop_products / total_products) * sample_size = 16 := 
by
  rw [h1, h2, h3]
  norm_num
  
  sorry

end stratified_sampling_l500_500376


namespace find_angle_A_find_max_S_plus_term_l500_500166

-- Define the problem conditions
def triangle_problem (a b c: ℝ) := a * a = b * b + c * c + sqrt 3 * a * b

-- Part I: Prove A = 5π/6
theorem find_angle_A (a b c A : ℝ) (h : triangle_problem a b c) : 
  A = 5 * Real.pi / 6 :=
sorry

-- Part II: Given a = sqrt 3, find the maximum value of S + 3cosBcosC and B when that happens
def S (a b c : ℝ) (A B C: ℝ) := 1 / 2 * b * c * sin A

theorem find_max_S_plus_term (S B : ℝ) : 
  ∃ Smax : ℝ, ∃ Bval : ℝ, (S + 3 * cos B * cos (π/12)) ≤ Smax ∧ Smax = 3  := 
sorry

end find_angle_A_find_max_S_plus_term_l500_500166


namespace area_of_triangle_with_medians_l500_500274

variables (S : ℝ)

theorem area_of_triangle_with_medians (h : S > 0) : 
  let T := S in
  area (triangle_with_sides_equal_to_medians (triangle.mk A B C)) = (3 / 4) * S :=
sorry

end area_of_triangle_with_medians_l500_500274


namespace tan_alpha_value_l500_500932

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l500_500932


namespace cubical_angle_range_l500_500543

variables {a b h x y : ℝ}
variables (A B B1 A1 C1 M N : ℝ × ℝ × ℝ)

def is_cube (A B B1 A1 C1 M N : ℝ × ℝ × ℝ) : Prop :=
(A = (0,0,0)) ∧
(B = (a,0,0)) ∧
(B1 = (a,0,h)) ∧
(A1 = (0,0,h)) ∧
(M = (x,0,0)) ∧
(0 < x) ∧ (x < a) ∧
(N = (a,0,y)) ∧
(0 < y) ∧ (y < h) ∧
(x = y - h)

noncomputable def angle_between_vectors (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
let dot_prod := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 in
let magnitude_v1 := Real.sqrt (v1.1^2 + v1.2^2 + v1.3^2) in
let magnitude_v2 := Real.sqrt (v2.1^2 + v2.2^2 + v2.3^2) in
Real.arccos (dot_prod / (magnitude_v1 * magnitude_v2))

theorem cubical_angle_range
  (h_cube : is_cube A B B1 A1 C1 M N) :
  0 < angle_between_vectors (x, 0, -h) (0, b, x) ∧ 
  angle_between_vectors (x, 0, -h) (0, b, x) < Real.pi / 3 :=
begin
  sorry
end

end cubical_angle_range_l500_500543


namespace find_coordinates_of_M_l500_500137

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := {x := 0, y := 0, z := 1}
def B : Point3D := {x := -1, y := 1, z := 1}
def C : Point3D := {x := 1, y := 2, z := -3}

def line (p1 p2 : Point3D) : ℝ → Point3D :=
  λ λ', { x := p1.x + λ' * (p2.x - p1.x),
          y := p1.y + λ' * (p2.y - p1.y),
          z := p1.z + λ' * (p2.z - p1.z) }

def M (λ : ℝ) : Point3D := line A B λ

noncomputable def dotProduct (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def vec (p1 p2 : Point3D) : Point3D :=
  { x := p2.x - p1.x, y := p2.y - p1.y, z := p2.z - p1.z }

def isPerpendicular (v1 v2 : Point3D) : Prop :=
  dotProduct v1 v2 = 0

theorem find_coordinates_of_M :
  ∃ (λ : ℝ), M λ = {x := -1/2, y := 1/2, z := 1} ∧ isPerpendicular (vec C (M λ)) (vec A B) :=
by {
  use 1/2,
  sorry
}

end find_coordinates_of_M_l500_500137


namespace garden_width_l500_500642

theorem garden_width (side_length_playground : ℕ) (length_garden : ℕ) (total_fencing : ℕ) 
  (h1 : side_length_playground = 27)
  (h2 : length_garden = 12)
  (h3 : total_fencing = 150) : 
  ∃ w : ℕ, 4 * side_length_playground + 2 * length_garden + 2 * w = total_fencing ∧ w = 9 := 
begin
  use 9,
  split,
  {
    rw [h1, h2, h3],
    norm_num,
  },
  {
    -- The width of the garden is 9 yards.
    refl,
  }
end

end garden_width_l500_500642


namespace tan_alpha_value_l500_500931

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l500_500931


namespace find_m_l500_500513

def vector (α : Type*) := α × α

def a : vector ℤ := (1, -2)
def b : vector ℤ := (3, 0)

def two_a_plus_b (a b : vector ℤ) : vector ℤ := (2 * a.1 + b.1, 2 * a.2 + b.2)
def m_a_minus_b (m : ℤ) (a b : vector ℤ) : vector ℤ := (m * a.1 - b.1, m * a.2 - b.2)

def parallel (v w : vector ℤ) : Prop := v.1 * w.2 = v.2 * w.1

theorem find_m : parallel (two_a_plus_b a b) (m_a_minus_b (-2) a b) :=
by
  sorry -- proof placeholder

end find_m_l500_500513


namespace model_quadratic_l500_500021

-- Definition of data points
def data_points : List (ℕ × ℝ) := [(0, 8.6), (5, 10.4), (10, 12.9)]

-- Definition of prediction point
def pred_point : ℕ × ℝ := (15, 16.1)

-- Statement that the data points and prediction can be modeled by a quadratic function
theorem model_quadratic (a b c : ℝ) :
  (∀ (x y : ℕ) (h : (x, y) ∈ data_points), (y:ℝ) = a * (x:ℝ)^2 + b * (x:ℝ) + c)
  ∧ ((pred_point.snd:ℝ) = a * (pred_point.fst:ℝ)^2 + b * (pred_point.fst:ℝ) + c) :=
sorry

end model_quadratic_l500_500021


namespace x_value_eq_max_f_value_l500_500514

open Real

-- Definitions of the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, sin x)
def b (x : ℝ) : ℝ × ℝ := (cos x, sin x)

-- Conditions on x
def x_domain := {x : ℝ | 0 ≤ x ∧ x ≤ π / 2}

-- Length of vectors
def norm_sq (v : ℝ × ℝ) := v.1 ^ 2 + v.2 ^ 2

-- Problem part (1)
theorem x_value_eq : ∀ x ∈ x_domain, norm_sq (a x) = norm_sq (b x) → x = π / 6 := by
  sorry

-- Dot product f(x)
def f (x : ℝ) := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Problem part (2)
theorem max_f_value : ∀ x ∈ x_domain, f(x) ≤ 3 / 2 ∧ (∃ x ∈ x_domain, f(x) = 3 / 2) := by
  sorry

end x_value_eq_max_f_value_l500_500514


namespace all_positive_integers_appear_l500_500087

def smallest_prime_not_dividing (k : ℕ) : ℕ := Nat.find (λ p, Nat.Prime p ∧ ¬ p ∣ k)

def sequence (a : ℕ) : ℕ → ℕ
| 0 => a
| (n+1) => Nat.find (λ m, m > 0 ∧ m ≠ sequence a n ∧ sequence a n^m % smallest_prime_not_dividing (sequence a n) = 1)

theorem all_positive_integers_appear (a : ℕ) (m : ℕ) (h_pos : m > 0) : 
  ∃ n : ℕ, sequence a n = m :=
sorry

end all_positive_integers_appear_l500_500087


namespace tan_alpha_solution_l500_500866

theorem tan_alpha_solution (α : ℝ) (h1 : 0 < α) (h2 : α < (Real.pi / 2))
  (h3 : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α)) :
  Real.tan α = (Real.sqrt 15) / 15 := 
sorry

end tan_alpha_solution_l500_500866


namespace min_distance_tangent_line_eq_l500_500435

theorem min_distance_tangent_line_eq (x y : ℝ) :
  let circle_eq := (x-2)^2 + (y-2)^2 = 1 in
  let line_eq := 4*x + 4*y - 7 = 0 in
  (circle_eq ∧ line_eq) → 
  min (λ MN, MN) = 7 * real.sqrt 2 / 8 :=
sorry

end min_distance_tangent_line_eq_l500_500435


namespace white_balls_count_l500_500372

theorem white_balls_count (w : ℕ) (h : (w / 15) * ((w - 1) / 14) = (1 : ℚ) / 21) : w = 5 := by
  sorry

end white_balls_count_l500_500372


namespace parity_of_f_max_of_f_min_of_f_l500_500100
open Real

def f (x : ℝ) := 2 + cos x

theorem parity_of_f : ∀ x : ℝ, f (-x) = f x :=
by sorry

theorem max_of_f : ∀ k : ℤ, f (2 * k * π) = 3 :=
by sorry

theorem min_of_f : ∀ k : ℤ, f (π + 2 * k * π) = 1 :=
by sorry

end parity_of_f_max_of_f_min_of_f_l500_500100


namespace tan_alpha_fraction_l500_500910

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l500_500910


namespace tan_alpha_fraction_l500_500906

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l500_500906


namespace farmer_plow_l500_500067

theorem farmer_plow (P : ℕ) (M : ℕ) (H1 : M = 12) (H2 : 8 * P + M * (8 - (55 / P)) = 30) (H3 : 55 % P = 0) : P = 10 :=
by
  sorry

end farmer_plow_l500_500067


namespace cost_price_of_cloth_l500_500727

theorem cost_price_of_cloth:
  ∀ (meters_sold profit_per_meter : ℕ) (selling_price : ℕ),
  meters_sold = 45 →
  profit_per_meter = 12 →
  selling_price = 4500 →
  (selling_price - (profit_per_meter * meters_sold)) / meters_sold = 88 :=
by
  intros meters_sold profit_per_meter selling_price h1 h2 h3
  sorry

end cost_price_of_cloth_l500_500727


namespace standard_eq_of_ellipse_max_dot_product_l500_500477

-- Definitions
variable {x y c a b : ℝ}

-- Conditions for the ellipse and line
def eccentricity_eq : Prop := c^2 / a^2 = 3 / 4
def a_eq_4b : Prop := a^2 = 4 * b^2
def ellipse_eq : Prop := x^2 / 4 + y^2 = 1

-- Slope of line l and vector condition
def slope_eq_one : Prop := true  -- Given slope is 1, no need to prove.
def vector_cond (PM QM : ℝ) : Prop := PM = -3 / 5 * QM

-- Definitions and Theorem for standard equation of the ellipse
theorem standard_eq_of_ellipse (PM QM : ℝ) (h1 : eccentricity_eq) (h2 : a_eq_4b) (h3 : vector_cond PM QM) : ellipse_eq := 
  sorry

-- Right vertex and inclination angle conditions
variable {α k : ℝ}

-- Inclination angle and calculation for maximum value of dot product
def inclination_angle_eq (α : ℝ) : Prop := α = 90

theorem max_dot_product (AP AQ : ℝ) (h1 : inclination_angle_eq α) : (AP * AQ).max = 33 / 4 :=
  sorry

end standard_eq_of_ellipse_max_dot_product_l500_500477


namespace max_value_a_l500_500571

def event_A (a : ℝ) (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  x^2 + y^2 ≤ a

def event_B (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  x - y + 1 ≥ 0 ∧ 5*x - 2*y - 4 ≤ 0 ∧ 2*x + y + 2 ≥ 0

theorem max_value_a (a : ℝ) (P : ℝ × ℝ → Prop) :
  (∀ M : ℝ × ℝ, event_A a M → event_B M) → a ≤ 1/2 :=
begin
  -- The proof is omitted
  sorry
end

end max_value_a_l500_500571


namespace domain_of_f_l500_500763

def log_base_2 (x : ℝ) := log x / log 2

theorem domain_of_f :
  (x : ℝ) (h : x ≥ 0) → -2 + log_base_2 x ≥ 0 → x ≥ 4 := 
sorry

end domain_of_f_l500_500763


namespace tan_alpha_value_l500_500924

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l500_500924


namespace finland_forest_percentage_l500_500784

noncomputable def finlandForestedArea : ℝ := 53.42  -- in million hectares
noncomputable def worldForestedArea : ℝ := 8076    -- 8.076 billion hectares converted to million hectares

theorem finland_forest_percentage :
  (finlandForestedArea / worldForestedArea) * 100 ≈ 0.6615 := 
sorry

end finland_forest_percentage_l500_500784


namespace advertising_cost_for_target_probability_three_selected_l500_500353

/-- Problem conditions -/
def advertising_costs : List ℕ := [2, 3, 5, 6, 8, 12]
def sales_volume : List ℕ := [30, 34, 40, 45, 50, 60]
def sum_xiy_i : ℕ := 1752

/-- Helper definitions -/
def mean (lst : List ℕ) : Rat := (lst.sum : Rat) / (lst.length : Rat)
def sum_of_squares (lst : List ℕ) : ℕ := lst.sumBy (λ x => x * x)
def n : ℕ := advertising_costs.length -- 6

noncomputable def regression_slope : Rat := (sum_xiy_i - n * mean advertising_costs * mean sales_volume) / 
                                            (sum_of_squares advertising_costs - n * (mean advertising_costs)^2)

noncomputable def regression_intercept : Rat := mean sales_volume - regression_slope * mean advertising_costs

/-- Proof to predict advertising cost for sales target --/
theorem advertising_cost_for_target : ∀ x : Rat, (3 * x + 151 / 6 ≥ 100) → x ≥ 25 := by
  intros x h
  sorry

def is_efficient (store_id : ℕ) : Prop := (sales_volume.get? store_id).getD 0 / (advertising_costs.get? store_id).getD 1 >= 9

def three_ok_efficient (ids : List ℕ) : Prop := ids.size = 3 ∧ ∃ i, i ∈ ids ∧ is_efficient i

theorem probability_three_selected : (choose 6 3).filter (λ ids, three_ok_efficient ids).length / (choose 6 3).length = 4 / 5 := by
  sorry

end advertising_cost_for_target_probability_three_selected_l500_500353


namespace width_of_cookie_sheet_l500_500685

-- Define the necessary parameters and conditions
def length := 2
def perimeter := 24

-- Define the width as a variable
variable (W : ℕ)

-- State the theorem we want to prove
theorem width_of_cookie_sheet
  (h1 : length = 2)
  (h2 : perimeter = 2 * (W + length)) :
  W = 10 :=
by
  simp [h1, h2]
  sorry

end width_of_cookie_sheet_l500_500685


namespace distance_between_points_l500_500073

-- Define the points and the distance formula
def point1 := (2, 5)
def point2 := (5, -1)
def distance_formula (x1 y1 x2 y2 : ℤ) : ℝ := 
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Statement of the problem
theorem distance_between_points : 
  distance_formula 2 5 5 (-1) = 3 * real.sqrt 5 :=
by {
  -- Proof is omitted.
  sorry
}

end distance_between_points_l500_500073


namespace balloons_difference_l500_500669

theorem balloons_difference (yours friends : ℝ) (hyours : yours = -7) (hfriends : friends = 4.5) :
  friends - yours = 11.5 :=
by
  rw [hyours, hfriends]
  sorry

end balloons_difference_l500_500669


namespace ellipse_properties_l500_500476

-- Definition of the ellipse and its properties
def ellipse (a b : ℝ) := λ x y : ℝ, x^2 / (a^2) + y^2 / (b^2) = 1

-- Given conditions
def a (c : ℝ) : ℝ := 2*c
def b (c : ℝ) : ℝ := Real.sqrt (4*c^2 - c^2)

-- Theorem Statement
theorem ellipse_properties :
  ∃ (a b c : ℝ), a = 2*c ∧ b = Real.sqrt (4*c^2 - c^2) ∧ c = 1 ∧ a = 2 ∧ b = Real.sqrt 3 ∧ (∀ x y : ℝ, ellipse a b x y) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, y₁ = 2 + Real.sqrt 3 * x₁ ∧ y₂ = 2 + Real.sqrt 3 * x₂ ∧ 
     ellipse a b x₁ y₁ ∧ ellipse a b x₂ y₂ ∧ 
     let AB_length := Real.sqrt ((x₁ + x₂)^2 - 4 * x₁ * x₂) in 
     let d := 1 in -- Given: The distance from origin to the line AB
     abs ((1 / 2) * AB_length * d) = 4 * Real.sqrt 177 / 15) :=
sorry

end ellipse_properties_l500_500476


namespace range_of_m_l500_500511

noncomputable def A : set ℝ := {x | (x + 2) * (x - 5) > 0}
noncomputable def neg_R_A : set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
noncomputable def B (m : ℝ) : set ℝ := {x | m ≤ x ∧ x < m + 1}

theorem range_of_m (m : ℝ) : B m ⊆ neg_R_A → -2 ≤ m ∧ m ≤ 4 :=
by {
  sorry
}

end range_of_m_l500_500511


namespace petya_guarantees_win_l500_500241

-- Definitions based on the conditions
def initial_sequence : list ℕ := [1,1,2,2,3,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,7]

-- Function to model a move
def erase_digits (seq : list ℕ) (digit : ℕ) : list ℕ :=
  seq.filter (≠ digit)

-- Predicate to check if a player wins
def wins (seq : list ℕ) : Prop :=
  seq.empty

-- Predicate to check if Petya can guarantee a win
def can_petya_win (init_seq : list ℕ) : Prop :=
  ∀ vasya_move : ℕ, (∃ petya_move : ℕ, wins (erase_digits (erase_digits init_seq vasya_move) petya_move))

theorem petya_guarantees_win : can_petya_win initial_sequence :=
sorry

end petya_guarantees_win_l500_500241


namespace Irja_wins_probability_l500_500195

noncomputable def probability_irja_wins : ℚ :=
  let X0 : ℚ := 4 / 7
  X0

theorem Irja_wins_probability :
  probability_irja_wins = 4 / 7 :=
sorry

end Irja_wins_probability_l500_500195


namespace mean_squares_first_n_integers_l500_500034

theorem mean_squares_first_n_integers (n : ℕ) :
  (∑ k in Finset.range n, (k + 1)^2) / n = (n + 1) * (2 * n + 1) / 6 :=
by
  sorry

end mean_squares_first_n_integers_l500_500034


namespace fraction_of_area_above_line_l500_500416

noncomputable def square_vertices : set (ℝ × ℝ) :=
  {(2, 1), (5, 1), (5, 4), (2, 4)}

noncomputable def line_segment : ℝ × ℝ → ℝ × ℝ → set (ℝ × ℝ)
| (x1, y1), (x2, y2) :=
  { p | ∃ t : ℝ, t ∈ Icc (0 : ℝ) 1 ∧ p = (t * x1 + (1 - t) * x2, t * y1 + (1 - t) * y2) }

noncomputable def area_of_square : ℝ :=
  9

noncomputable def area_of_triangle_below_line : ℝ :=
  1.5

theorem fraction_of_area_above_line :
  (1 - area_of_triangle_below_line / area_of_square) = 5 / 6 :=
by
  sorry

end fraction_of_area_above_line_l500_500416


namespace migrating_geese_percentage_l500_500725

-- Define conditions as parameters
variables (G : ℕ) -- Total number of geese
variables (G_m G_f : ℕ) -- Number of migrating male and female geese
variables (P_m : ℚ) -- Percentage of migrating male geese

-- Assume 50% of the geese in the study were male
def fifty_percent_male (G_male G_female : ℕ) : Prop :=
  G_male + G_female = G ∧ G_male = 0.5 * G ∧ G_female = 0.5 * G

-- Ratio of migration rate for male geese to female geese is 0.25
def migration_rate_ratio (M_m M_f : ℚ) : Prop :=
  M_m = 0.25 * M_f

-- Prove that P_m equals 20%
theorem migrating_geese_percentage :
  (fifty_percent_male G_m G_f)
  ∧ (migration_rate_ratio (G_m / (0.5 * G)) (G_f / (0.5 * G))) →
  P_m = 0.20 :=
sorry

end migrating_geese_percentage_l500_500725


namespace trig_identity_l500_500945

theorem trig_identity (θ : ℝ) (h : cos (π/6 - θ) = 1/3) : 
  cos (5*π/6 + θ) - sin^2 (θ - π/6) = -11/9 :=
by
  sorry

end trig_identity_l500_500945


namespace exists_three_distinct_digits_l500_500432

theorem exists_three_distinct_digits :
  ∃ (A B C : ℕ), 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
    (A < 10) ∧ (B < 10) ∧ (C < 10) ∧ 
    (∃ (n1 n2 n3 : ℕ), 
      (A * 100 + B * 10 + C = n1 * n1) ∧ 
      (C * 100 + B * 10 + A = n2 * n2) ∧
      (C * 100 + A * 10 + B = n3 * n3) ∧
      A * 100 + B * 10 + C = 961 ∧ 
      C * 100 + B * 10 + A = 169 ∧ 
      C * 100 + A * 10 + B = 196) :=
begin
  use [9, 6, 1],
  split, 
  { exact ne_of_gt (by norm_num) },
  split,
  { exact ne_of_gt (by norm_num) },
  split,
  { exact ne_of_gt (by norm_num) },
  split,
  { exact by norm_num },
  split,
  { exact by norm_num },
  split,
  { exact by norm_num },
  use [31, 13, 14],
  split,
  { exact by norm_num },
  split,
  { exact by norm_num },
  split,
  { exact by norm_num },
  split,
  { exact by norm_num },
  split,
  { exact by norm_num },
  exact by norm_num
end

end exists_three_distinct_digits_l500_500432


namespace a6_value_l500_500473

noncomputable def fib (n : ℕ) : ℕ :=
  if n = 1 then 1 else
  if n = 2 then 1 else
  fib (n - 1) + fib (n - 2)

theorem a6_value :
  (fib 6) = 8 :=
by
  sorry

end a6_value_l500_500473


namespace det_B_squared_sub_3B_eq_10_l500_500522

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2, 3], ![2, 2]]

theorem det_B_squared_sub_3B_eq_10 : 
  Matrix.det (B * B - 3 • B) = 10 := by
  sorry

end det_B_squared_sub_3B_eq_10_l500_500522


namespace smallest_M_conditions_met_l500_500453

def isDivisible (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = k * b

theorem smallest_M_conditions_met :
  ∃ M : ℕ, 
    (M > 0) ∧
    (isDivisible 484 8) ∧
    (isDivisible 485 121) ∧
    (isDivisible 486 9) ∧
    (!isDivisible 484 49 ∧ !isDivisible 485 49 ∧ !isDivisible 486 49) ∧
    (M = 484) :=
begin
  use 484,
  split,
  { norm_num },
  split,
  { use 60, norm_num },
  split,
  { use 4, norm_num },
  split,
  { use 54, norm_num },
  split,
  { split,
    { intro h, cases h with k hk, norm_num at hk },
    split,
    { intro h, cases h with k hk, norm_num at hk },
    { intro h, cases h with k hk, norm_num at hk },
  },
  { norm_num }
end

end smallest_M_conditions_met_l500_500453


namespace exists_distance_one_set_l500_500586

def distance (p q : ℝ × ℝ) : ℝ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2) ^ (1/2)

theorem exists_distance_one_set (n : ℕ) (hn : n ≥ 1) :
  ∃ (A : finset (ℝ × ℝ)), ∀ p ∈ A, (A.filter (λ q, distance p q = 1)).card = n :=
sorry

end exists_distance_one_set_l500_500586


namespace ratio_near_integer_l500_500956

theorem ratio_near_integer (a b : ℝ) (h1 : (a + b) / 2 = 3 * (Real.sqrt (a * b))) (h2 : a > b) (h3 : b > 0) : Int.nearestInteger (a / b) = 28 :=
sorry

end ratio_near_integer_l500_500956


namespace area_of_other_triangle_l500_500253

theorem area_of_other_triangle (P R S T : Type) [geometry P R S T] :
  (¬ right_angled_trapezoid P R S T) →
  (equilateral_triangle T R P) →
  (right_angled_triangle T R S) →
  ((area T R S = 10) ∨ (area T R P = 10)) →
  ((area T R S = 5) ∨ (area T R S = 20) ∨ (area T R P = 5) ∨ (area T R P = 20)) :=
by
  sorry

end area_of_other_triangle_l500_500253


namespace add_fractions_l500_500411

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by sorry

end add_fractions_l500_500411


namespace tan_alpha_value_l500_500885

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l500_500885


namespace bricks_required_l500_500699

noncomputable def area_rectangle (length : ℝ) (width : ℝ) : ℝ :=
length * width

noncomputable def area_semicircle (diameter : ℝ) : ℝ :=
(let r := diameter / 2
in (real.pi * r ^ 2) / 2)

noncomputable def area_brick (length_cm : ℝ) (width_cm : ℝ) : ℝ :=
(length_cm * width_cm) / 10000

noncomputable def total_bricks (area_courtyard : ℝ) (area_brick : ℝ) : ℤ :=
int.ceil (area_courtyard / area_brick)

theorem bricks_required
    (length_rect : ℝ) (width_rect : ℝ) 
    (diam_semicircle : ℝ)
    (length_brick_cm : ℝ) (width_brick_cm : ℝ) :
    total_bricks (area_rectangle length_rect width_rect + area_semicircle diam_semicircle) 
                 (area_brick length_brick_cm width_brick_cm) = 117257 :=
by
    have area_courtyard := area_rectangle length_rect width_rect + area_semicircle diam_semicircle
    have area_one_brick := area_brick length_brick_cm width_brick_cm
    exact int.eq_of_sub_eq_zero (int.ceil_eq_iff.mpr (eq.symm sorry : area_courtyard / area_one_brick = 117257))

end bricks_required_l500_500699


namespace percent_students_in_range_l500_500378

theorem percent_students_in_range (students_90_100 : ℕ) (students_80_89 : ℕ) 
(students_70_79 : ℕ) (students_60_69 : ℕ) (students_below_60 : ℕ) 
(h_students_90_100 : students_90_100 = 5) 
(h_students_80_89 : students_80_89 = 8) 
(h_students_70_79 : students_70_79 = 10) 
(h_students_60_69 : students_60_69 = 4) 
(h_students_below_60 : students_below_60 = 6) : 
  (students_70_79 : ℝ) / (students_90_100 + students_80_89 + students_70_79 + students_60_69 + students_below_60 : ℝ) * 100 ≈ 30.3 :=
by
  sorry

end percent_students_in_range_l500_500378


namespace sequence_100th_term_is_981_l500_500395

/-- The sequence of positive integers which are either multiples of 3 or sums of different multiples of 3. -/
def valid_sequence : ℕ → Prop :=
λ n, ∃ (k : ℕ) (c : ℕ → ℕ), (∀ i, c i = 0 ∨ c i = 3^i) ∧ n = ∑ i in finset.range k, c i

/-- The 100th term in the specific sequence is 981. -/
theorem sequence_100th_term_is_981 : ∃ n, valid_sequence n ∧ n = 981 :=
sorry

end sequence_100th_term_is_981_l500_500395


namespace group_B_basis_l500_500736

-- Definitions of vectors for the groups
def vecA1 := (0, 0 : ℝ × ℝ)
def vecA2 := (1, 2 : ℝ × ℝ)

def vecB1 := (0, -1 : ℝ × ℝ)
def vecB2 := (-1, 0 : ℝ × ℝ)

def vecC1 := (-2, 3 : ℝ × ℝ)
def vecC2 := (4, -6 : ℝ × ℝ)

def vecD1 := (1, 3 : ℝ × ℝ)
def vecD2 := (4, 12 : ℝ × ℝ)

-- Condition to check if two vectors are collinear
def collinear (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Theorem stating Group B vectors are not collinear and serve as a basis
theorem group_B_basis : ¬ collinear vecB1 vecB2 :=
by sorry

end group_B_basis_l500_500736


namespace tan_alpha_value_l500_500928

open Real

theorem tan_alpha_value (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 := 
sorry

end tan_alpha_value_l500_500928


namespace convex_polygon_exterior_angles_sum_constant_l500_500530

theorem convex_polygon_exterior_angles_sum_constant (n : ℕ) (h : n ≥ 3) : 
  ∑ (i : fin n), exterior_angle i = 360 :=
by
  sorry

end convex_polygon_exterior_angles_sum_constant_l500_500530


namespace tan_alpha_value_l500_500934

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2)
  (hα2 : tan (2 * α) = cos α / (2 - sin α)) : tan α = sqrt 15 / 15 := 
by
  sorry

end tan_alpha_value_l500_500934


namespace conditional_probability_l500_500035

variables {Ω : Type} {event_A event_B : Ω → Prop}

def P(event : Ω → Prop) : ℝ := sorry

-- Given conditions encoded as Lean variables/constants
axiom total_number_of_outcomes : ℝ := 7
axiom number_of_favorable_outcomes_A_and_B : ℝ := 3
axiom P_joint_A_and_B : ℝ := number_of_favorable_outcomes_A_and_B / total_number_of_outcomes

-- Define the probability of event A
noncomputable def P_A : ℝ := 1 -- Since the problem states P(A) = 1 based on assumptions.

-- Define a theorem to prove the conditional probability
theorem conditional_probability (P_A : ℝ) : P(event_B | event_A) = number_of_favorable_outcomes_A_and_B / total_number_of_outcomes := 
by sorry

end conditional_probability_l500_500035


namespace correct_propositions_l500_500101

-- Definitions of lines and planes
variables (m n : Line) (α β : Plane)

-- Propositions given as conditions
def proposition1 := (m ⊆ α ∧ n ∥ α) → m ∥ n
def proposition2 := (m ⟂ α ∧ n ∥ α) → m ⟂ n
def proposition3 := (m ⟂ α ∧ m ⟂ β) → α ∥ β
def proposition4 := (m ∥ α ∧ n ∥ α) → m ∥ n

-- Statement asserting which propositions are true
theorem correct_propositions :
  (¬proposition1) ∧ proposition2 ∧ proposition3 ∧ (¬proposition4) :=
by sorry

end correct_propositions_l500_500101


namespace product_of_largest_primes_l500_500351

theorem product_of_largest_primes :
  let p1 := 5 in
  let p2 := 7 in
  let p3 := 97 in
  let p4 := 997 in
  p1 * p2 * p3 * p4 = 3383815 :=
by
  let p1 := 5
  let p2 := 7
  let p3 := 97
  let p4 := 997
  show p1 * p2 * p3 * p4 = 3383815
  sorry

end product_of_largest_primes_l500_500351


namespace area_equality_of_triangles_l500_500995

theorem area_equality_of_triangles 
  (A B C M N P R S Q : Type) 
  [triangle : is_triangle A B C]
  [on_AB : is_point_on_segment M A B] 
  [on_BC : is_point_on_segment N B C] 
  [on_CA : is_point_on_segment P C A]
  [parallelogram : is_parallelogram C P M N]
  (R_def : is_intersection R (line_through A N) (line_through M P))
  (S_def : is_intersection S (line_through B P) (line_through M N))
  (Q_def : is_intersection Q (line_through A N) (line_through B P)) : 
  area_of_quadrilateral M R Q S = area_of_quadrilateral N Q P :=
sorry

end area_equality_of_triangles_l500_500995


namespace sqrt_of_a_possible_values_l500_500113

theorem sqrt_of_a_possible_values (a : ℝ) (h : a = 3 ∨ a = -1 ∨ a = -6 ∨ a = -7): (0 ≤ a) → (a = 3) :=
begin
  intros ha,
  cases h,
  { exact h },
  { cases h,
    { exfalso, linarith },
    { cases h,
      { exfalso, linarith },
      { exfalso, linarith } } }
end

end sqrt_of_a_possible_values_l500_500113


namespace kishore_savings_l500_500732

noncomputable def rent := 5000
noncomputable def milk := 1500
noncomputable def groceries := 4500
noncomputable def education := 2500
noncomputable def petrol := 2000
noncomputable def miscellaneous := 700
noncomputable def total_expenses := rent + milk + groceries + education + petrol + miscellaneous
noncomputable def salary : ℝ := total_expenses / 0.9 -- given that savings is 10% of salary

theorem kishore_savings : (salary * 0.1) = 1800 :=
by
  sorry

end kishore_savings_l500_500732


namespace closest_fraction_to_medals_won_l500_500739

theorem closest_fraction_to_medals_won (won_medals total_medals : ℕ)
  (won_medals = 20) (total_medals = 120)
  (fractions : List ℚ)
  (fractions = [1/5, 1/6, 1/7, 1/8, 1/9]) :
  ∃ f ∈ fractions, abs ((won_medals / total_medals : ℚ) - f) = abs ((20 / 120 : ℚ) - 1/6) := by
sor

end closest_fraction_to_medals_won_l500_500739


namespace Corey_found_golf_balls_on_Saturday_l500_500047

def goal : ℕ := 48
def golf_balls_found_on_sunday : ℕ := 18
def golf_balls_needed : ℕ := 14
def golf_balls_found_on_saturday : ℕ := 16

theorem Corey_found_golf_balls_on_Saturday :
  (goal - golf_balls_found_on_sunday - golf_balls_needed) = golf_balls_found_on_saturday := 
by
  sorry

end Corey_found_golf_balls_on_Saturday_l500_500047


namespace tan_alpha_sqrt_15_over_15_l500_500920

theorem tan_alpha_sqrt_15_over_15 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
sorry

end tan_alpha_sqrt_15_over_15_l500_500920


namespace students_did_not_take_test_l500_500574

theorem students_did_not_take_test (total_students : ℕ) (q1_correct : ℕ) (q2_correct : ℕ) :
  total_students = 25 ∧ q1_correct = 22 ∧ q2_correct = 20 →
  (total_students - 22) = 3 :=
begin
  sorry
end

end students_did_not_take_test_l500_500574


namespace trout_split_equally_l500_500575

-- Conditions: Nancy and Joan caught 18 trout and split them equally
def total_trout : ℕ := 18
def equal_split (n : ℕ) : ℕ := n / 2

-- Theorem: Prove that if they equally split the trout, each person will get 9 trout.
theorem trout_split_equally : equal_split total_trout = 9 :=
by 
  -- Placeholder for the actual proof
  sorry

end trout_split_equally_l500_500575


namespace tan_alpha_solution_l500_500855

theorem tan_alpha_solution (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : tan (2 * α) = cos α / (2 - sin α)) : 
  tan α = (Real.sqrt 15) / 15 :=
  sorry

end tan_alpha_solution_l500_500855


namespace largest_three_digit_multiple_of_8_with_sum_24_l500_500655

theorem largest_three_digit_multiple_of_8_with_sum_24 :
  ∃ n : ℕ, (n ≥ 100 ∧ n < 1000) ∧ (∃ k, n = 8 * k) ∧ (n.digits.sum = 24) ∧
           ∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (∃ k', m = 8 * k') ∧ (m.digits.sum = 24) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_8_with_sum_24_l500_500655


namespace height_of_right_triangle_l500_500531

-- Using noncomputable to define constants and main proof
noncomputable def height_on_longest_side (a b c : ℝ) (h_triangle: a^2 + b^2 = c^2) : ℝ :=
  let area := (a * b) / 2
  in (2 * area) / c

theorem height_of_right_triangle : 
  height_on_longest_side 6 8 10 (by norm_num) = 4.8 :=
by
  sorry -- Proof is not provided

end height_of_right_triangle_l500_500531


namespace exists_n_sum_digits_eq_125_l500_500433

-- Define the sum of the digits of a natural number n
def S (n : ℕ) : ℕ :=
  n.digits.sum

-- The main statement to prove
theorem exists_n_sum_digits_eq_125 : ∃ n : ℕ, n + S n = 125 ∧ n = 121 :=
by
  sorry

end exists_n_sum_digits_eq_125_l500_500433


namespace common_chord_of_circles_l500_500075

theorem common_chord_of_circles
  (x y : ℝ)
  (h1 : x^2 + y^2 + 2 * x = 0)
  (h2 : x^2 + y^2 - 4 * y = 0)
  : x + 2 * y = 0 := 
by
  -- Lean will check the logical consistency of the statement.
  sorry

end common_chord_of_circles_l500_500075


namespace orthogonal_projection_length_l500_500271

theorem orthogonal_projection_length 
  (r : ℝ) (h_r : r = 1)
  (S : ℝ) (h_S : S = π * r^2)
  (S' : ℝ) (h_S' : S' = 1)
  (a : ℝ) :
  a = 2 * r * (sqrt (π^2 - 1) / π) :=
  by
  sorry

end orthogonal_projection_length_l500_500271


namespace add_fractions_l500_500409

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by sorry

end add_fractions_l500_500409


namespace curve_is_hyperbola_l500_500448

theorem curve_is_hyperbola (r : ℝ) (θ : ℝ) :
  r = 5 * tan θ * sec θ + 2 →
  ∃ (x y : ℝ), (r = real.sqrt (x^2 + y^2)) → 
  (x^4 + 2*x^2*y^2 + y^4 = 25*y^2) → 
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 := by
sorry

end curve_is_hyperbola_l500_500448


namespace problem_statement_l500_500819

theorem problem_statement (a b c m : ℝ) (h_nonzero_a : a ≠ 0) (h_nonzero_b : b ≠ 0)
  (h_nonzero_c : c ≠ 0) (h1 : a + b + c = m) (h2 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2 * a)^2 + b * (m - 2 * b)^2 + c * (m - 2 * c)^2) / (a * b * c) = 12 :=
sorry

end problem_statement_l500_500819


namespace clive_change_l500_500038

theorem clive_change (money : ℝ) (olives_needed : ℕ) (olives_per_jar : ℕ) (price_per_jar : ℝ) : 
  (money = 10) → 
  (olives_needed = 80) → 
  (olives_per_jar = 20) →
  (price_per_jar = 1.5) →
  money - (olives_needed / olives_per_jar) * price_per_jar = 4 := by
  sorry

end clive_change_l500_500038


namespace A_and_D_independent_l500_500319

-- Definitions of the events based on given conditions
def event_A (x₁ : ℕ) : Prop := x₁ = 1
def event_B (x₂ : ℕ) : Prop := x₂ = 2
def event_C (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 8
def event_D (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 7

-- Probabilities based on uniform distribution and replacement
def probability_event (event : ℕ → ℕ → Prop) : ℚ :=
  if h : ∃ x₁ : ℕ, ∃ x₂ : ℕ, x₁ ∈ finset.range 1 7 ∧ x₂ ∈ finset.range 1 7 ∧ event x₁ x₂
  then ((finset.card (finset.filter (λ x, event x.1 x.2)
                (finset.product (finset.range 1 7) (finset.range 1 7)))) : ℚ) / 36
  else 0

noncomputable def P_A : ℚ := 1 / 6
noncomputable def P_D : ℚ := 1 / 6
noncomputable def P_A_and_D : ℚ := 1 / 36

-- Independence condition (by definition): P(A ∩ D) = P(A) * P(D)
theorem A_and_D_independent :
  P_A_and_D = P_A * P_D := by
  sorry

end A_and_D_independent_l500_500319


namespace socks_shirts_different_probability_l500_500014

-- Define the sets of colors for socks and shirts
def socks_colors := {blue, red, green}
def shirts_colors := {blue, red, green, yellow}

-- Define the probability of different combinations
def probability_socks_shirts_different : ℚ := 
  let total_configurations := (3 : ℚ) * 4
  let mismatching_configurations := (3 : ℚ) * 3
  mismatching_configurations / total_configurations

-- State the theorem
theorem socks_shirts_different_probability : probability_socks_shirts_different = 3 / 4 := by
  sorry

end socks_shirts_different_probability_l500_500014


namespace largest_three_digit_multiple_of_8_and_sum_24_is_888_l500_500661

noncomputable def largest_three_digit_multiple_of_8_with_digit_sum_24 : ℕ :=
  888

theorem largest_three_digit_multiple_of_8_and_sum_24_is_888 :
  ∃ n : ℕ, (300 ≤ n ∧ n ≤ 999) ∧ (n % 8 = 0) ∧ ((n.digits 10).sum = 24) ∧ n = largest_three_digit_multiple_of_8_with_digit_sum_24 :=
by
  existsi 888
  sorry

end largest_three_digit_multiple_of_8_and_sum_24_is_888_l500_500661


namespace pasta_ratio_l500_500360

theorem pasta_ratio (total_students : ℕ) (spaghetti : ℕ) (manicotti : ℕ) 
  (h1 : total_students = 650) 
  (h2 : spaghetti = 250) 
  (h3 : manicotti = 100) : 
  (spaghetti : ℤ) / (manicotti : ℤ) = 5 / 2 :=
by
  sorry

end pasta_ratio_l500_500360


namespace sin_double_angle_l500_500497

-- Given that the terminal side of angle α passes through the point (1, -2),
-- find the value of sin 2α.
theorem sin_double_angle (α : ℝ) (h₁ : ∃ α, ∀ α, sin α = -2 / (real.sqrt 5) ∧ cos α = 1 / (real.sqrt 5)) :
  real.sin (2 * α) = -4 / 5 :=
by
  sorry

end sin_double_angle_l500_500497


namespace tan_alpha_value_l500_500874

theorem tan_alpha_value (α : ℝ) (hα1 : α ∈ Ioo 0 (π / 2)) (hα2 : tan (2 * α) = cos α / (2 - sin α)) :
  tan α = sqrt 15 / 15 :=
by
  sorry

end tan_alpha_value_l500_500874


namespace sales_fifth_month_l500_500001

theorem sales_fifth_month
  (a1 a2 a3 a4 a6 : ℕ)
  (h1 : a1 = 2435)
  (h2 : a2 = 2920)
  (h3 : a3 = 2855)
  (h4 : a4 = 3230)
  (h6 : a6 = 1000)
  (avg : ℕ)
  (h_avg : avg = 2500) :
  a1 + a2 + a3 + a4 + (15000 - 1000 - (a1 + a2 + a3 + a4)) + a6 = avg * 6 :=
by
  sorry

end sales_fifth_month_l500_500001


namespace C_is_knight_l500_500599

-- Definitions
def A_statement : Prop := ∀ x, x = A → liar x
def B_statement : Prop := ∃ x, (x = A ∨ x = B ∨ x = C) ∧ liar x ∧ ∀ y, liar y → y = x

-- Proving C is a knight
theorem C_is_knight (A_statement : A_statement) (B_statement : B_statement) : C = knight := 
by
  -- proof goes here
  sorry

end C_is_knight_l500_500599


namespace isosceles_triangle_BC_squared_l500_500996

namespace TriangleProof

variables {A B C I : Type}
variables (AB AC AI BC : ℝ)

def IsIsoscelesTriangle (AB AC : ℝ) : Prop := AB = AC

def IncentreDistanceFromBC (distance : ℝ) := distance = 2

def IncentreDistanceFromA (AI : ℝ) := AI = 3

theorem isosceles_triangle_BC_squared
  (h1 : IsIsoscelesTriangle AB AC)
  (h2 : IncentreDistanceFromBC 2)
  (h3 : IncentreDistanceFromA 3)
  : BC^2 = 80 :=
begin
  sorry
end

end TriangleProof

end isosceles_triangle_BC_squared_l500_500996


namespace arrange_magnitudes_l500_500486

theorem arrange_magnitudes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  (2 * a * b / (a + b)) < sqrt (a * b) ∧ sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < sqrt ((a^2 + b^2) / 2) :=
by
  sorry

end arrange_magnitudes_l500_500486


namespace count_complex_numbers_satisfying_conditions_l500_500564

def is_valid_z (z : ℂ) : Prop :=
  let f := z^2 + complex.I * z + 1
  ∃ (a b : ℤ), abs a ≤ 15 ∧ abs b ≤ 15 ∧ f.re = a ∧ f.im = b

noncomputable def count_valid_z : ℕ := sorry

theorem count_complex_numbers_satisfying_conditions :
  ∃ n : ℕ, n = count_valid_z ∧ ∀ z : ℂ, (is_valid_z z ∧ z.im > 0) ↔ count_valid_z = n :=
sorry

end count_complex_numbers_satisfying_conditions_l500_500564


namespace exists_F_xi_gt_l500_500598

noncomputable def F (x : ℝ) (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  x^n + ∑ i in finset.range n, a i * x^i

theorem exists_F_xi_gt (n : ℕ) (a : ℕ → ℝ) (x : ℕ → ℝ)
  (h1 : ∀ i j : ℕ, i < n → j < n → i < j → x j < x i)
  (h2 : ∀ i, i ≤ n → x i ∈ ℤ) :
  ∃ i, i ≤ n ∧ |F (x i) n a| > n.factorial / 2^n :=
by
  sorry

end exists_F_xi_gt_l500_500598


namespace wash_cycle_time_l500_500328

-- Definitions for the conditions
def num_loads : Nat := 8
def dry_cycle_time_minutes : Nat := 60
def total_time_hours : Nat := 14
def total_time_minutes : Nat := total_time_hours * 60

-- The actual statement we need to prove
theorem wash_cycle_time (x : Nat) (h : num_loads * x + num_loads * dry_cycle_time_minutes = total_time_minutes) : x = 45 :=
by
  sorry

end wash_cycle_time_l500_500328


namespace sqrt_5sq_4six_eq_320_l500_500042

theorem sqrt_5sq_4six_eq_320 : Real.sqrt (5^2 * 4^6) = 320 :=
by sorry

end sqrt_5sq_4six_eq_320_l500_500042


namespace power_equality_l500_500948

theorem power_equality (x : ℝ) (n : ℕ) (h : x^(2 * n) = 3) : x^(4 * n) = 9 := 
by 
  sorry

end power_equality_l500_500948


namespace min_days_to_plant_trees_l500_500718

def trees_planted (n : ℕ) : ℕ := (List.range (n+1)).map (λ i, 2^i) |>.sum

theorem min_days_to_plant_trees (n : ℕ) (h₀ : 100 ≤ trees_planted n) : n ≥ 6 :=
by
  sorry

end min_days_to_plant_trees_l500_500718


namespace irrational_b_eq_neg_one_l500_500214

theorem irrational_b_eq_neg_one
  (a : ℝ) (b : ℝ)
  (h_irrational : ¬ ∃ q : ℚ, a = (q : ℝ))
  (h_eq : ab + a - b = 1) :
  b = -1 :=
sorry

end irrational_b_eq_neg_one_l500_500214


namespace extrema_range_of_m_l500_500833

def has_extrema (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, (∀ z : ℝ, z ≤ x → f z ≤ f x) ∧ (∀ z : ℝ, z ≥ y → f z ≤ f y)

noncomputable def f (m x : ℝ) : ℝ :=
  x^3 + m * x^2 + (m + 6) * x + 1

theorem extrema_range_of_m (m : ℝ) :
  has_extrema (f m) ↔ (m ∈ Set.Iic (-3) ∪ Set.Ici 6) :=
by
  sorry

end extrema_range_of_m_l500_500833


namespace clive_change_l500_500039

theorem clive_change (total_money : ℝ) (num_olives_needed : ℕ) (olives_per_jar : ℕ) (cost_per_jar : ℝ)
  (h1 : total_money = 10)
  (h2 : num_olives_needed = 80)
  (h3 : olives_per_jar = 20)
  (h4 : cost_per_jar = 1.5) : total_money - (num_olives_needed / olives_per_jar) * cost_per_jar = 4 := by
  sorry

end clive_change_l500_500039


namespace salt_percentage_l500_500682

theorem salt_percentage :
  ∀ (salt water : ℝ), salt = 10 → water = 90 → 
  100 * (salt / (salt + water)) = 10 :=
by
  intros salt water h_salt h_water
  sorry

end salt_percentage_l500_500682


namespace determine_m_l500_500429

theorem determine_m (m : ℕ) (h : m * (m + 1)! + (m + 1)! = 5040) : m = 5 :=
sorry

end determine_m_l500_500429


namespace connected_distinct_points_with_slope_change_l500_500398

-- Defining the cost function based on the given conditions
def cost_function (n : ℕ) : ℕ := 
  if n <= 10 then 20 * n else 18 * n

-- The main theorem to prove the nature of the graph as described in the problem
theorem connected_distinct_points_with_slope_change : 
  (∀ n, (1 ≤ n ∧ n ≤ 20) → 
    (∃ k, cost_function n = k ∧ 
    (n <= 10 → cost_function n = 20 * n) ∧ 
    (n > 10 → cost_function n = 18 * n))) ∧
  (∃ n, n = 10 ∧ cost_function n = 200 ∧ cost_function (n + 1) = 198) :=
sorry

end connected_distinct_points_with_slope_change_l500_500398


namespace power_function_property_l500_500843

theorem power_function_property : 
  (∃ a : ℝ, ∀ x : ℝ, f x = x ^ a ∧ f 2 = real.sqrt 2) →
  f 16 = 4 :=
by
  intro h
  obtain ⟨a, ha, h2⟩ := h
  have ha2 : 2 ^ a = real.sqrt 2 := h2
  have ha_eq_1_2 : a = 1 / 2 := sorry -- Solve 2^a = sqrt(2) for a
  rw ha_eq_1_2 at ha
  have h16 : f 16 = 16 ^ (1 / 2) := ha 16
  calc f 16 = 16 ^ (1 / 2) : h16
        ... = 4 : sorry -- Compute 16^(1/2)

end power_function_property_l500_500843
