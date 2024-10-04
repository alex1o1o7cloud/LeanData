import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Combinatorial.Combinatorics
import Mathlib.Algebra.Divisors
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Lcm
import Mathlib.Algebra.Main
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Trig
import Mathlib.Analysis.Calculus.Monotonicity
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialLaws
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Parity
import Mathlib.Data.List
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Nat.Factors
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Div
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Vector
import Mathlib.Geometry.Euclidean
import Mathlib.LinearAlgebra.QuadraticForms
import Mathlib.Probability.Basic
import Mathlib.RingTheory.Polynomial.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic

namespace loss_percentage_is_5_l399_399639

-- define the constants for purchase price and selling price
def purchasePrice : ℝ := 490
def sellingPrice : ℝ := 465.50

-- loss amount calculated from purchase price and selling price
def lossAmount : ℝ := purchasePrice - sellingPrice

-- loss percentage calculated as (loss amount / purchase price) * 100
def lossPercentage : ℝ := (lossAmount / purchasePrice) * 100

-- state the proof problem: prove the loss percentage is 5%
theorem loss_percentage_is_5 :
  lossPercentage = 5 := 
by 
  -- You can compute it directly or assume so
  sorry -- proof is skipped as per instruction

end loss_percentage_is_5_l399_399639


namespace units_digit_of_square_l399_399680

theorem units_digit_of_square (a b : ℕ) (h₁ : (10 * a + b) ^ 2 % 100 / 10 = 7) : b = 6 :=
sorry

end units_digit_of_square_l399_399680


namespace tabitha_final_amount_is_six_l399_399165

def initial_amount : ℕ := 25
def amount_given_to_mom : ℕ := 8
def num_items_bought : ℕ := 5
def cost_per_item : ℝ := 0.5

def amount_after_giving_mom : ℝ := initial_amount - amount_given_to_mom
def amount_invested : ℝ := amount_after_giving_mom / 2
def amount_after_investment : ℝ := amount_after_giving_mom - amount_invested
def total_cost_of_items : ℝ := num_items_bought * cost_per_item
def final_amount : ℝ := amount_after_investment - total_cost_of_items

theorem tabitha_final_amount_is_six :
  final_amount = 6 := 
by 
  -- sorry to skip the proof
  sorry

end tabitha_final_amount_is_six_l399_399165


namespace scale_balanced_after_placing_all_weights_l399_399792

theorem scale_balanced_after_placing_all_weights
  (n : ℕ)
  (weights : Fin (n+1) → ℕ)
  (h_total_weight : (∑ i, weights i) = 2 * n)
  (h_sorted : ∀ i j : Fin (n+1), i < j → weights i ≥ weights j)
  (h_placement : ∀ i : Fin (n+1), 
                   let left_weight := (∑ j in Finset.filter (λ k, k < i) Finset.univ, weights j),
                       right_weight := (∑ j in Finset.filter (λ k, k ≥ i) Finset.univ, weights j)
                   in left_weight ≤ right_weight → left_weight = right_weight ∨ right_weight = left_weight):
  let final_left_weight := (∑ j in Finset.filter (λ k, true) Finset.univ, if h_placement i then weights j else 0),
      final_right_weight := (∑ j in Finset.filter (λ k, true) Finset.univ, if ¬ h_placement i then weights j else 0)
  in final_left_weight = final_right_weight :=
by
  sorry

end scale_balanced_after_placing_all_weights_l399_399792


namespace no_integer_solution_for_equation_l399_399555

theorem no_integer_solution_for_equation :
  ¬ ∃ (x y : ℤ), x^2 + 3 * x * y - 2 * y^2 = 122 :=
sorry

end no_integer_solution_for_equation_l399_399555


namespace centroid_property_l399_399237

-- Definitions of vertices of triangle PQR
def P : (ℝ × ℝ) := (-3, 3)
def Q : (ℝ × ℝ) := (4, -2)
def R : (ℝ × ℝ) := (0, 7)

-- Definition of centroid of triangle PQR
def centroid (P Q R : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

-- Computing the value of 8m + 5n where (m, n) is the centroid of triangle PQR
def point_S (P Q R : (ℝ × ℝ)) : ℝ :=
  let (m, n) := centroid P Q R in 8 * m + 5 * n

theorem centroid_property : point_S P Q R = 16 := 
  by
    sorry

end centroid_property_l399_399237


namespace value_of_a2_b2_l399_399000

theorem value_of_a2_b2 (a b : ℝ) (i : ℂ) (hi : i^2 = -1) (h : (a - i) * i = b - i) : a^2 + b^2 = 2 :=
by sorry

end value_of_a2_b2_l399_399000


namespace possible_starting_lineups_l399_399950

theorem possible_starting_lineups (n : ℕ) (bar : n = 15) (bob_yogi_moe : Finset ℕ)
    (h1 : bob_yogi_moe.card = 3) : 
    ∑ i in (Finset.range (3 + 1)), if i = 3 then (Nat.choose 12 (i + 1)) else (Nat.choose 12 4) = 2277 := by
  sorry

end possible_starting_lineups_l399_399950


namespace monotonicity_of_f_range_of_a_if_no_zeros_l399_399071

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end monotonicity_of_f_range_of_a_if_no_zeros_l399_399071


namespace equal_segments_l399_399487

variables {A B C D E F G M N : Type} 

-- assuming these are points on the plane
variables [incidence_geometry A B C D E F G M N ]
 
-- Given conditions
def conditions (h_triangle : triangle A B C) 
  (h_angle : angle A B C ≠ 60)
  (h_bd_ce_tangent : tangent A B D ∧ tangent A C E)
  (h_bd_ce_eq_bc : distance B D = distance C E ∧ distance B D = distance B C)
  (h_DE_intersects : intersects D E (extended_line A B) F ∧ intersects D E (extended_line A C) G)
  (h_CF_intersection : intersection_point C F B D = M)
  (h_CE_intersection : intersection_point C E B G = N) : Prop := 
  true

-- Goal to prove
theorem equal_segments
  (h_triangle : triangle A B C) 
  (h_angle : angle A B C ≠ 60)
  (h_bd_ce_tangent : tangent A B D ∧ tangent A C E)
  (h_bd_ce_eq_bc : distance B D = distance C E ∧ distance B D = distance B C)
  (h_DE_intersects : intersects D E (extended_line A B) F ∧ intersects D E (extended_line A C) G)
  (h_CF_intersection : intersection_point C F B D = M)
  (h_CE_intersection : intersection_point C E B G = N) :
  distance A M = distance A N := 
sorry

end equal_segments_l399_399487


namespace distance_between_hyperbola_vertices_l399_399744

theorem distance_between_hyperbola_vertices :
  let a : ℝ := 12
  let b : ℝ := 7
  ∀ (x y : ℝ), (x^2 / 144 - y^2 / 49 = 1) →
  (2 * a = 24) := 
by
  intros
  unfold a
  unfold b
  sorry

end distance_between_hyperbola_vertices_l399_399744


namespace area_relation_l399_399579

-- Defining the conditions
def line_equation (x : ℝ) : ℝ := -2/5 * x + 10

def point_P : ℝ × ℝ := (25, 0)
def point_Q : ℝ × ℝ := (0, 10)

noncomputable def area_POQ : ℝ := 1/2 * 25 * 10

-- Defining point T
def point_T (r s : ℝ) : Prop := s = line_equation r

-- Defining the area relationship
def area_POT (r s : ℝ) : ℝ := 1/2 * 25 * s

-- The proof statement
theorem area_relation (r s : ℝ) (hT : point_T r s) (hQ : area_POQ = 4 * area_POT r s) : r + s = 21.25 :=
by
  sorry

end area_relation_l399_399579


namespace A_share_in_profit_l399_399309

-- Define the investments and profits
def A_investment : ℕ := 6300
def B_investment : ℕ := 4200
def C_investment : ℕ := 10500
def total_profit : ℕ := 12200

-- Define the total investment
def total_investment : ℕ := A_investment + B_investment + C_investment

-- Define A's ratio in the investment
def A_ratio : ℚ := A_investment / total_investment

-- Define A's share in the profit
def A_share : ℚ := total_profit * A_ratio

-- The theorem to prove
theorem A_share_in_profit : A_share = 3660 := by
  sorry

end A_share_in_profit_l399_399309


namespace modulus_of_z_l399_399783

theorem modulus_of_z (z : ℂ) (h : (2 - I) * z = -3 + 4 * I) : complex.abs z = real.sqrt 5 :=
by
  sorry

end modulus_of_z_l399_399783


namespace fuel_at_40_min_fuel_l399_399686

section FuelConsumption

noncomputable def fuel_consumption (x : ℝ) : ℝ := (1 / 128000) * x^3 - (3 / 80) * x + 8

noncomputable def total_fuel (x : ℝ) : ℝ := (fuel_consumption x) * (100 / x)

theorem fuel_at_40 : total_fuel 40 = 17.5 :=
by sorry

theorem min_fuel : total_fuel 80 = 11.25 ∧ ∀ x, (0 < x ∧ x ≤ 120) → total_fuel x ≥ total_fuel 80 :=
by sorry

end FuelConsumption

end fuel_at_40_min_fuel_l399_399686


namespace intersection_locus_l399_399514

theorem intersection_locus
  (a b : ℝ) (a_gt_b : a > b) (b_gt_zero : b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1) :
  ∃ (x y : ℝ), (x^2)/(a^2) - (y^2)/(b^2) = 1 :=
sorry

end intersection_locus_l399_399514


namespace proof_problem_l399_399421

theorem proof_problem (a t : ℝ) (ha : a = 8) (ht : t = 63) : a + t = 71 := by {
    rw [ha, ht],
    norm_num,
}

end proof_problem_l399_399421


namespace extreme_values_at_a_eq_2_monotonicity_of_f_l399_399113

def f (a x : ℝ) := a * log x - x - (1 / 2) * x^2

theorem extreme_values_at_a_eq_2 :
  f 2 1 = -3 / 2 ∧ (∀ x > 0, (f 2 x ≥ -3/ 2)) := 
sorry

theorem monotonicity_of_f (a : ℝ) :
  (a ≤ -1 / 4 → ∀ x > 0, ∃ y > x, f a x ≥ f a y) ∧
  (-1 / 4 < a ∧ a ≤ 0 → ∀ x > 0, ∃ y > x, f a x ≥ f a y) ∧
  (a > 0 → 
    (∀ x ∈ (0, -1 + sqrt (1 + 4 * a) / 2 : Set ℝ), ∃ y > x, f a y ≥ f a x) ∧ 
    (∀ x ∈ (-1 + sqrt (1 + 4 * a) / 2, + ∞ : Set ℝ), ∃ y > x, f a x ≥ f a y)) :=
sorry

end extreme_values_at_a_eq_2_monotonicity_of_f_l399_399113


namespace complementary_angles_decrease_percentage_l399_399199

theorem complementary_angles_decrease_percentage :
  ∀ (x : ℝ), (3 * x + 7 * x = 90) →
  (3 * x * 1.2 + 7 * x = 90) →
  (3 * x > 0) →
  (7 * x > 0) →
  let original_larger_angle : ℝ := 7 * x in
  let new_smaller_angle : ℝ := 3 * x * 1.2 in
  let new_larger_angle : ℝ := 90 - new_smaller_angle in
  let decrease : ℝ := original_larger_angle - new_larger_angle in
  let percentage_decrease : ℝ := (decrease / original_larger_angle) * 100 in
  percentage_decrease = 8.57 := 
sorry

end complementary_angles_decrease_percentage_l399_399199


namespace even_function_is_a_4_l399_399004

def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_is_a_4 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 4 := by
  sorry

end even_function_is_a_4_l399_399004


namespace points_in_set_condition_l399_399527

-- Define the essential conditions and theorems based on the problem statement
theorem points_in_set_condition (n : ℕ) (S : set (ℝ × ℝ × ℝ)) 
  (h1 : ∀ (planes : fin n → set (ℝ × ℝ × ℝ)), ¬∀ p ∈ S, ∃ i, p ∈ planes i)
  (h2 : ∀ X ∈ S, ∃ planes : fin n → set (ℝ × ℝ × ℝ), ∀ p ∈ (S \ {X}), ∃ i, p ∈ planes i) :
  3 * n + 1 ≤ S.card ∧ S.card ≤ (n + 3).choose 3 :=
sorry

end points_in_set_condition_l399_399527


namespace triangle_BCD_is_isosceles_l399_399500

theorem triangle_BCD_is_isosceles 
  (A B C D O₁ O₂ : Type)
  [triangle ABC]
  (O₁_center_inscribed_circle : is_center_of_inscribed_circle O₁ ABC)
  (D_on_extension_AB_beyond_B : is_on_extension D AB B)
  (O₂_center_circle : is_center_of_circle_tangent_to_segments O₂ CD AB AC)
  (equal_distances : dist O₁ C = dist O₂ C) :
  is_isosceles_triangle B C D :=
sorry

end triangle_BCD_is_isosceles_l399_399500


namespace gerbil_sales_l399_399299

theorem gerbil_sales (x : ℕ): x = 14 :=
  (68 - x = 54) → x = 14 :=
  sorry

end gerbil_sales_l399_399299


namespace scale_model_height_is_correct_l399_399312

noncomputable def height_of_scale_model (h_real : ℝ) (V_real : ℝ) (V_scale : ℝ) : ℝ :=
  h_real / (V_real / V_scale)^(1/3:ℝ)

theorem scale_model_height_is_correct :
  height_of_scale_model 90 500000 0.2 = 0.66 :=
by
  sorry

end scale_model_height_is_correct_l399_399312


namespace find_cos_theta_l399_399428

theorem find_cos_theta {θ : ℝ} :
  let coeff_x2 := (5.choose 2) * (cos θ)^2,
      coeff_x3 := (4.choose 3) * (5 / 4)
  in coeff_x2 = coeff_x3 -> cos θ = sqrt(2)/2 ∨ cos θ = -sqrt(2)/2 :=
by
  sorry

end find_cos_theta_l399_399428


namespace incorrect_increase_rate_l399_399501

-- Conditions
def initial_temp (t : ℕ) (y : ℕ) : Prop := 
  ((t, y) = (0, 10)) ∨ ((t, y) = (10, 30)) ∨ ((t, y) = (20, 50)) ∨ ((t, y) = (30, 70)) ∨ ((t, y) = (40, 90))

-- Prove that statement "every 10 seconds of heating, the oil temperature increases by 30°C" is incorrect.
theorem incorrect_increase_rate : ¬ (∀ t, (t % 10 = 0 ∧ t < 40) → ∃ Δy, Δy = 30) :=
begin
  -- Assume every 10 seconds of heating, the increase of temperature is 30°C.
  assume h : ∀ t, (t % 10 = 0 ∧ t < 40) → ∃ Δy, Δy = 30,
  -- Contradiction with the given data
  have h0 : initial_temp 0 10 := by simp [initial_temp],
  have h10 : initial_temp 10 30 := by simp [initial_temp],
  have h20 : initial_temp 20 50 := by simp [initial_temp],
  have h30 : initial_temp 30 70 := by simp [initial_temp],
  have h40 : initial_temp 40 90 := by simp [initial_temp],
  -- The temperature increase is not 30°C, but 20°C.
  have h_incorrect : ¬ (∀ t, (t % 10 = 0  ∧ t < 40) → ∃ Δy, Δy = 30) := 
    begin
      intro t,
      intro ht,
      cases ht.2,
      -- Possible contradiction with provided table values
      case 0   {exact exists.intro 20 rfl},
      case 10  {exact exists.intro 20 rfl},
      case 20  {exact exists.intro 20 rfl},
      case 30  {exact exists.intro 20 rfl},
    end,
  -- Contradiction achieved, thus proving the statement incorrect
  exact h_incorrect,
end

end incorrect_increase_rate_l399_399501


namespace monotonicity_no_zeros_range_of_a_l399_399063

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * (Real.log x) + 1

theorem monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1/a → ∀ t : ℝ, t ∈ Ioo x y → f' a t < 0 ) ∧ 
  (∀ x y : ℝ, 1/a < x ∧ x < y → ∀ t : ℝ, t ∈ Ioo x y → f' a t > 0 ) :=
sorry

theorem no_zeros_range_of_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → f a x > 0) : 
  a > 1/Real.exp 1 :=
sorry

end monotonicity_no_zeros_range_of_a_l399_399063


namespace range_of_a_l399_399130

theorem range_of_a (a : ℝ) : (¬(∀ x : ℝ, x ∈ set.Icc 1 2 → x^2 - a ≥ 0)) → (1 < a) :=
  by
    sorry

end range_of_a_l399_399130


namespace sum_of_projections_of_centroid_to_sides_l399_399496

theorem sum_of_projections_of_centroid_to_sides 
  (X Y Z : Type) [MetricSpace X]
  (triangle_XYZ : X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X)
  (XY : dist X Y = 4)
  (XZ : dist X Z = 6)
  (YZ : dist Y Z = 5)
  (T : X) (is_centroid : Centroid T X Y Z)
  (U : X) (V : X) (W : X)
  (proj_UYZ : Projection U YZ T)
  (proj_VXZ : Projection V XZ T)
  (proj_WXY : Projection W XY T) :
  dist T U + dist T V + dist T W = 4.082 :=
sorry

end sum_of_projections_of_centroid_to_sides_l399_399496


namespace max_PA_squared_minus_PB_squared_l399_399181

noncomputable def C₁_parametric (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

noncomputable def C₂_parametric (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

def C₂_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

def point_A := (-2 : ℝ, 0 : ℝ)
def point_B := (1 : ℝ, 1 : ℝ)

noncomputable def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

noncomputable def PA_squared (θ : ℝ) : ℝ :=
  dist_squared (C₂_parametric θ) point_A

noncomputable def PB_squared (θ : ℝ) : ℝ :=
  dist_squared (C₂_parametric θ) point_B

theorem max_PA_squared_minus_PB_squared : 
  (∃ θ : ℝ, PA_squared θ - PB_squared θ = 2 + 2 * Real.sqrt 39) := 
begin
  sorry
end

end max_PA_squared_minus_PB_squared_l399_399181


namespace cost_per_dvd_l399_399277

theorem cost_per_dvd (total_cost : ℝ) (num_dvds : ℕ) (cost_per_dvd : ℝ) :
  total_cost = 4.80 ∧ num_dvds = 4 → cost_per_dvd = 1.20 :=
by
  intro h
  sorry

end cost_per_dvd_l399_399277


namespace right_handed_players_total_l399_399594

theorem right_handed_players_total
    (total_players : ℕ)
    (throwers : ℕ)
    (left_handed : ℕ)
    (right_handed : ℕ) :
    total_players = 150 →
    throwers = 60 →
    left_handed = (total_players - throwers) / 2 →
    right_handed = (total_players - throwers) / 2 →
    total_players - throwers = 2 * left_handed →
    left_handed + right_handed + throwers = total_players →
    ∀ throwers : ℕ, throwers = 60 →
    right_handed + throwers = 105 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end right_handed_players_total_l399_399594


namespace percentage_decrease_of_larger_angle_l399_399190

noncomputable def complementary_angles_decrease_percentage : Real :=
let total_degrees := 90
let ratio_sum := 3 + 7
let part := total_degrees / ratio_sum
let smaller_angle := 3 * part
let larger_angle := 7 * part
let increased_smaller_angle := smaller_angle * 1.2
let new_larger_angle := total_degrees - increased_smaller_angle
let decrease_amount := larger_angle - new_larger_angle
(decrease_amount / larger_angle) * 100

theorem percentage_decrease_of_larger_angle
  (smaller_increased_percentage : Real := 20)
  (ratio_three : Real := 3)
  (ratio_seven : Real := 7)
  (total_degrees : Real := 90)
  (expected_decrease : Real := 8.57):
  complementary_angles_decrease_percentage = expected_decrease := 
sorry

end percentage_decrease_of_larger_angle_l399_399190


namespace distance_between_hyperbola_vertices_l399_399747

theorem distance_between_hyperbola_vertices :
  let a : ℝ := 12
  let b : ℝ := 7
  ∀ (x y : ℝ), (x^2 / 144 - y^2 / 49 = 1) →
  (2 * a = 24) := 
by
  intros
  unfold a
  unfold b
  sorry

end distance_between_hyperbola_vertices_l399_399747


namespace jon_percentage_increase_l399_399509

def initial_speed : ℝ := 80
def trainings : ℕ := 4
def weeks_per_training : ℕ := 4
def speed_increase_per_week : ℝ := 1

theorem jon_percentage_increase :
  let total_weeks := trainings * weeks_per_training
  let total_increase := total_weeks * speed_increase_per_week
  let final_speed := initial_speed + total_increase
  let percentage_increase := (total_increase / initial_speed) * 100
  percentage_increase = 20 :=
by
  sorry

end jon_percentage_increase_l399_399509


namespace arithmetic_mean_reciprocal_primes_l399_399341

theorem arithmetic_mean_reciprocal_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := (1 : ℚ) / p1
  let r2 := (1 : ℚ) / p2
  let r3 := (1 : ℚ) / p3
  let r4 := (1 : ℚ) / p4
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 := by
sorry

end arithmetic_mean_reciprocal_primes_l399_399341


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l399_399346

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  let primes := [2, 3, 5, 7] in
  let reciprocals := primes.map (λ p => (1 : ℚ) / p) in
  let sum_reciprocals := reciprocals.sum in
  let mean := sum_reciprocals / (primes.length : ℚ) in
  mean = 247 / 840 :=
by
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p => (1 : ℚ) / p)
  let sum_reciprocals := reciprocals.sum
  let mean := sum_reciprocals / (primes.length : ℚ)
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l399_399346


namespace ralph_socks_problem_l399_399557

theorem ralph_socks_problem :
  ∃ x y z : ℕ, x + y + z = 10 ∧ x + 2 * y + 4 * z = 30 ∧ 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ x = 2 :=
by
  sorry

end ralph_socks_problem_l399_399557


namespace decimal_place_150_of_8_over_9_l399_399251

theorem decimal_place_150_of_8_over_9 :
  (decimal_place 150 (8 / 9)) = 8 :=
begin
  sorry
end

end decimal_place_150_of_8_over_9_l399_399251


namespace hyperbola_vertex_distance_l399_399771

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  (x^2 / 144 - y^2 / 49 = 1) →
  ∃ (a : ℝ), a = 12 ∧ 2 * a = 24 :=
by
  intro x y h
  have h1 : 12^2 = 144 := by norm_num
  use 12
  split
  case left =>
    exact rfl
  case right =>
    calc
      2 * 12 = 24 : by norm_num

end hyperbola_vertex_distance_l399_399771


namespace boat_downstream_distance_l399_399661

-- Given conditions
def speed_boat_still_water : ℕ := 25
def speed_stream : ℕ := 5
def travel_time_downstream : ℕ := 3

-- Proof statement: The distance travelled downstream is 90 km
theorem boat_downstream_distance :
  speed_boat_still_water + speed_stream * travel_time_downstream = 90 :=
by
  -- omitting the actual proof steps
  sorry

end boat_downstream_distance_l399_399661


namespace base_n_representation_of_b_l399_399800

theorem base_n_representation_of_b (n : ℤ) (h : n > 8) (b : ℤ)
  (H : ∃ x : ℤ, x^2 - (2*n + 1)*x + b = 0) :
  nat.to_digits n b = [1, 0, 1] :=
sorry

end base_n_representation_of_b_l399_399800


namespace time_approx_equal_l399_399241

-- Definitions of the conditions as given in part a
def length_first_train : ℕ := 360
def speed_first_train_km_hr : ℝ := 45
def length_second_train : ℕ := 480
def speed_second_train_km_hr : ℝ := 60
def length_platform : ℕ := 240

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_hr : ℝ) : ℝ :=
  speed_km_hr * 1000 / 3600

-- Converted speeds from km/hr to m/s
def speed_first_train_m_s : ℝ := km_per_hr_to_m_per_s speed_first_train_km_hr
def speed_second_train_m_s : ℝ := km_per_hr_to_m_per_s speed_second_train_km_hr

-- Relative speed of the two trains in m/s
def relative_speed_m_s : ℝ := speed_first_train_m_s + speed_second_train_m_s

-- Total distance to be covered by the first train
def total_distance_m : ℕ := length_first_train + length_platform

-- Time it takes for the first train to pass the platform while the second train moves in the opposite direction
def time_to_pass_platform : ℝ := total_distance_m / relative_speed_m_s

-- Desired answer
def expected_time : ℝ := 20.57

-- Lean 4 statement to prove that the calculated time is approximately equal to the expected time
theorem time_approx_equal : abs (time_to_pass_platform - expected_time) < 0.01 := by
  sorry

end time_approx_equal_l399_399241


namespace biology_marks_l399_399361

theorem biology_marks (english : ℕ) (math : ℕ) (physics : ℕ) (chemistry : ℕ) (average : ℕ) (biology : ℕ) 
  (h1 : english = 36) 
  (h2 : math = 35) 
  (h3 : physics = 42) 
  (h4 : chemistry = 57) 
  (h5 : average = 45) 
  (h6 : (english + math + physics + chemistry + biology) / 5 = average) : 
  biology = 55 := 
by
  sorry

end biology_marks_l399_399361


namespace distance_to_lightning_is_3_25_miles_l399_399927

def speed_of_sound : ℕ := 1100
def time_delay : ℕ := 15
def feet_per_mile : ℕ := 5280

-- Convert feet to miles
def feet_to_miles (feet : ℕ) : ℝ := feet / feet_per_mile

-- Round to the nearest quarter mile
def round_to_quarter_mile (miles : ℝ) : ℝ := 
  ((miles * 4).round : ℝ) / 4

theorem distance_to_lightning_is_3_25_miles :
  round_to_quarter_mile (feet_to_miles (speed_of_sound * time_delay)) = 3.25 :=
by {
  sorry -- Proof to be provided
}

end distance_to_lightning_is_3_25_miles_l399_399927


namespace a_minus_b_l399_399457

theorem a_minus_b (a b : ℕ) (h1 : 2^4 + 2^4 = 2^a) (h2 : 3^5 + 3^5 + 3^5 = 3^b) : a - b = -1 :=
by
  sorry

end a_minus_b_l399_399457


namespace parabola_standard_equation_l399_399672

theorem parabola_standard_equation :
  ∃ p : ℝ, (vertex : ℝ × ℝ) (focus : ℝ × ℝ), vertex = (0, 0) ∧ focus = (2, 0) ∧ (y : ℝ) (x : ℝ), y^2 = 4 * p * x ∧ p = 2 :=
by
  sorry

end parabola_standard_equation_l399_399672


namespace laptop_repair_cost_l399_399699

theorem laptop_repair_cost
  (price_phone_repair : ℝ)
  (price_computer_repair : ℝ)
  (price_laptop_repair : ℝ)
  (condition1 : price_phone_repair = 11)
  (condition2 : price_computer_repair = 18)
  (condition3 : 5 * price_phone_repair + 2 * price_laptop_repair + 2 * price_computer_repair = 121) :
  price_laptop_repair = 15 :=
by
  sorry

end laptop_repair_cost_l399_399699


namespace log_expression_value_l399_399623

noncomputable def logb (b a : ℝ) := Real.log a / Real.log b

theorem log_expression_value : 
  (logb 2 80 / logb 40 2) - (logb 2 160 / logb 20 2) = 2 :=
by
  sorry

end log_expression_value_l399_399623


namespace trick_deck_cost_l399_399725

theorem trick_deck_cost :
  (∃ x : ℝ, 4 * x + 4 * x = 72) → ∃ x : ℝ, x = 9 := sorry

end trick_deck_cost_l399_399725


namespace quadrilateral_not_necessarily_planar_l399_399258

theorem quadrilateral_not_necessarily_planar :
  ¬ ∀ (q : Type) [quadrilateral q] (equal_sides : Bool), planar q :=
begin
  -- Using conditions:
  -- 1. Triangle is a planar shape
  -- 2. Trapezoid is a planar shape
  -- 3. Parallelogram is a planar shape
  
  -- Show:
  -- Quadrilateral with equal sides is not necessarily a planar shape
  sorry
end

end quadrilateral_not_necessarily_planar_l399_399258


namespace next_birthday_monday_l399_399139
open Nat

-- Define the basic structure and parameters of our problem
def is_leap_year (year : ℕ) : Prop := 
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def day_of_week (start_day : ℕ) (year_diff : ℕ) (is_leap : ℕ → Prop) : ℕ :=
  (start_day + year_diff + (year_diff / 4) - (year_diff / 100) + (year_diff / 400)) % 7

-- Specify problem conditions
def initial_year := 2009
def initial_day := 5 -- 2009-06-18 is Friday, which is 5 if we start counting from Sunday as 0
def end_day := 1 -- target day is Monday, which is 1

-- Main theorem
theorem next_birthday_monday : ∃ year, year > initial_year ∧
  day_of_week initial_day (year - initial_year) is_leap_year = end_day := by
  use 2017
  -- The proof would go here, skipping with sorry
  sorry

end next_birthday_monday_l399_399139


namespace monotonically_increasing_l399_399552

variable {R : Type} [LinearOrderedField R]

def f (x : R) : R := 3 * x + 1

theorem monotonically_increasing : ∀ x₁ x₂ : R, x₁ < x₂ → f x₁ < f x₂ :=
by
  intro x₁ x₂ h
 -- this is where the proof would go
  sorry

end monotonically_increasing_l399_399552


namespace x_in_C_is_necessary_but_not_sufficient_for_x_in_A_l399_399011

noncomputable theory

variable {α : Type*}
variable (A B C : set α)
variable (x : α)

-- Conditions
axiom h_non_empty_A : A.nonempty
axiom h_non_empty_B : B.nonempty
axiom h_non_empty_C : C.nonempty
axiom h_union : A ∪ B = C
axiom h_not_subset : ¬ (B ⊆ A)

-- Prove
theorem x_in_C_is_necessary_but_not_sufficient_for_x_in_A :
  (x ∈ C) → (x ∈ A) ↔ (x ∈ C) := sorry

end x_in_C_is_necessary_but_not_sufficient_for_x_in_A_l399_399011


namespace last_four_digits_of_5_pow_2013_l399_399542

theorem last_four_digits_of_5_pow_2013 : (5 ^ 2013) % 10000 = 3125 :=
by
  sorry

end last_four_digits_of_5_pow_2013_l399_399542


namespace joe_lift_ratio_l399_399876

theorem joe_lift_ratio (F S : ℕ) 
  (h1 : F + S = 1800) 
  (h2 : F = 700) 
  (h3 : 2 * F = S + 300) : F / S = 7 / 11 :=
by
  sorry

end joe_lift_ratio_l399_399876


namespace number_of_factors_l399_399395

theorem number_of_factors (n : ℕ) (h : n = 2^2 * 3^2 * 7^2) : (finset.univ.filter (λ d, d ∣ n)).card = 27 :=
by sorry

end number_of_factors_l399_399395


namespace honest_people_count_l399_399229

-- Definitions and conditions
def person (n : ℕ) : Prop :=
  n = 1 -> ¬(exists y, y = (0 : ℕ)) ∧
  n = 2 -> (forall y : ℕ, y <= 1) ∧
  n = 3 -> (forall y : ℕ, y <= 2) ∧
  n = 4 -> (forall y : ℕ, y <= 3) ∧
  n = 5 -> (forall y : ℕ, y <= 4) ∧
  n = 6 -> (forall y : ℕ, y <= 5) ∧
  n = 7 -> (forall y : ℕ, y <= 6) ∧
  n = 8 -> (forall y : ℕ, y <= 7) ∧
  n = 9 -> (forall y : ℕ, y <= 8) ∧
  n = 10 -> (forall y : ℕ, y <= 9) ∧
  n = 11 -> (forall y : ℕ, y <= 10) and
  n = 12 -> (forall y : ℕ, y <= 11)

-- Proof statement
theorem honest_people_count : (exists y : ℕ, y = 6) ∧
  (∀ n, ¬(person n) -> y < n - 1) ∧
  (∀ n, person n -> y <= n - 1) :=
sorry

end honest_people_count_l399_399229


namespace range_of_a_squared_plus_b_l399_399410

variable (a b : ℝ)

theorem range_of_a_squared_plus_b (h1 : a < -2) (h2 : b > 4) : ∃ y, y = a^2 + b ∧ 8 < y :=
by
  sorry

end range_of_a_squared_plus_b_l399_399410


namespace complementary_angles_decrease_86_percent_l399_399191

theorem complementary_angles_decrease_86_percent (x : ℝ) (h : 10 * x = 90) :
  let small_angle := 3 * x
  let increased_small_angle := small_angle * 1.2
  let large_angle := 7 * x
  let new_large_angle := 90 - increased_small_angle
  (new_large_angle / large_angle) * 100 = 91.4 :=
by
  sorry

end complementary_angles_decrease_86_percent_l399_399191


namespace exists_bounding_constant_l399_399300

-- Define the initial plane with its chessboard pattern and recoloring
structure ChessboardPattern where
  is_red : ℝ × ℝ → Prop
  is_blue : ℝ × ℝ → Prop
  condition : ∀ (a b : ℝ × ℝ), 
    (a ≠ b ∧ (abs (a.1 - b.1) = 1 ∧ abs (a.2 - b.2) = 1 )) → 
    (is_red a → is_blue b) ∧ (is_blue a → is_red b)

-- Define the line and line segment
structure LineSegment (ℓ : ℝ → ℝ) where
  start : ℝ × ℝ
  end : ℝ × ℝ
  parallel_condition : (end.2 - start.2) / (end.1 - start.1) = ℓ (start.1)

-- Define the main theorem
theorem exists_bounding_constant :
  ∃ C : ℝ, ∀ (ℓ : ℝ → ℝ), 
  (∀ (p1 p2 : ℝ × ℝ), (p1 ≠ p2) → (ℓ p1.1 ≠ p1.2) ∧ (ℓ p2.1 ≠ p2.2) → 
  (∀ I : LineSegment ℓ, 
      let B := ∫ x in I.start.1..I.end.1, is_blue (x, ℓ x)
      let R := ∫ x in I.start.1..I.end.1, is_red (x, ℓ x)
      abs (B - R) ≤ C)) :=
sorry

end exists_bounding_constant_l399_399300


namespace odd_last_member_teams_l399_399998

def number_of_teams_with_odd_last_numbers (n : ℕ) :=
  let team_sizes := List.range' 10 18
  let last_numbers := List.scanl (· + ·) 0 team_sizes
  List.countp (odd : ℕ → Bool) last_numbers

theorem odd_last_member_teams : number_of_teams_with_odd_last_numbers 18 = 10 :=
  sorry

end odd_last_member_teams_l399_399998


namespace monotonicity_and_no_real_roots_l399_399093

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

theorem monotonicity_and_no_real_roots 
  (a : ℝ) (ha : 0 < a) : 
  (∀ x : ℝ, (0 < x ∧ x < (1 / a) → deriv (f a) x < 0) ∧ (x > (1 / a) → deriv (f a) x > 0)) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≠ 0 → a > (1 / real.exp 1))) :=
begin
  sorry
end

end monotonicity_and_no_real_roots_l399_399093


namespace prob1_monotonic_intervals_prob2_range_of_a_l399_399442

noncomputable def prob1_f (x : ℝ) : ℝ := x ^ 2 - 4 * x - 2 * log x

theorem prob1_monotonic_intervals : 
  (∀ x, x ∈ Ioo (0 : ℝ) (1 + Real.sqrt 2) -> deriv (prob1_f x) < 0) ∧
  (∀ x, x ∈ Ioi (1 + Real.sqrt 2) -> deriv (prob1_f x) > 0) :=
sorry

noncomputable def prob2_f (x : ℝ) (a : ℝ) : ℝ := x ^ 2 + 2 * a * x + a * log x

theorem prob2_range_of_a :
  (∀ (a : ℝ), a ≤ 0 → ∀ (x : ℝ), 0 < x → prob2_f x a > (1 / 2) * (2 * Real.exp 1 + 1) * a → 
   a ∈ Icc (-2 * (Real.exp 1) ^ 2 / (2 * Real.exp 1 + 1)) 0) :=
sorry

end prob1_monotonic_intervals_prob2_range_of_a_l399_399442


namespace initial_amount_value_l399_399378

def increased_amount (P : ℝ) (n : ℕ) : ℝ := P * (10/9)^n

-- Stating the conditions
def amount_after_two_years (P : ℝ) : Prop :=
  increased_amount P 2 = 79012.34567901235

-- Stating the proof problem 
theorem initial_amount_value (P : ℝ) (h : amount_after_two_years P) : P = 64000 :=
sorry

end initial_amount_value_l399_399378


namespace complementary_angle_decrease_l399_399210

theorem complementary_angle_decrease (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3 / 7) : 
  (∃ new_α : ℝ, new_α = α * 1.2) →
  ∃ new_β : ℝ, new_β = (1 - 0.0857) * β :=
by
  intro h3
  cases h3 with new_α h_newα
  use 90 - new_α
  sorry

end complementary_angle_decrease_l399_399210


namespace difference_in_legs_and_heads_l399_399871

-- We define the constants and variables involved
variable (D : ℕ) -- number of ducks
def C : ℕ := 14 -- number of cows, given constant
def L : ℕ := 2 * D + 4 * C -- total number of legs
def H : ℕ := D + C -- total number of heads

-- Define the difference X
def X : ℕ := L - 2 * H

-- Theorem to prove the difference X
theorem difference_in_legs_and_heads : X = 28 := by
  sorry

end difference_in_legs_and_heads_l399_399871


namespace complex_root_product_value_l399_399908

noncomputable def complex_root_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : ℂ :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1)

theorem complex_root_product_value (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : complex_root_product r h1 h2 = 14 := 
  sorry

end complex_root_product_value_l399_399908


namespace value_of_B_l399_399460

theorem value_of_B (x y : ℕ) (h1 : x > y) (h2 : y > 1) (h3 : x * y = x + y + 22) :
  (x / y) = 12 :=
sorry

end value_of_B_l399_399460


namespace polynomial_value_at_zero_l399_399522

theorem polynomial_value_at_zero :
  ∃ p : ℝ[X], 
  (∀ n : ℕ, n ≤ 6 → p.eval (3 ^ n) = (1 / 3 ^ n)) ∧ (p.degree = 6) ∧ p.eval 0 = 2186 / 2187 :=
by {
  have h_exists_p: ∃ p: ℝ[X], (∀ n : ℕ, n ≤ 6 → p.eval (3 ^ n) = (1 / 3 ^ n)) ∧ (p.degree = 6),
  { sorry },
  cases h_exists_p with p hp,
  use p,
  split,
  { exact hp.1 },
  split,
  { exact hp.2 },
  { sorry }
}

end polynomial_value_at_zero_l399_399522


namespace height_differences_l399_399483

theorem height_differences
  (height_CN_Tower : ℕ)
  (height_diff_CN_SpaceNeedle : ℕ)
  (height_Eiffel_Tower : ℕ)
  (predicted_height_Jeddah_Tower : ℕ)
  (height_CN_Tower_eq : height_CN_Tower = 553)
  (height_diff_CN_SpaceNeedle_eq : height_diff_CN_SpaceNeedle = 369)
  (height_Eiffel_Tower_eq : height_Eiffel_Tower = 330)
  (predicted_height_Jeddah_Tower_eq : predicted_height_Jeddah_Tower = 1000) :
  let height_SpaceNeedle := height_CN_Tower - height_diff_CN_SpaceNeedle,
      height_diff_Eiffel_SpaceNeedle := height_Eiffel_Tower - height_SpaceNeedle,
      height_diff_Eiffel_Jeddah := predicted_height_Jeddah_Tower - height_Eiffel_Tower in
  height_diff_Eiffel_SpaceNeedle = 146 ∧ height_diff_Eiffel_Jeddah = 670 :=
by
  sorry

end height_differences_l399_399483


namespace tan_A_is_5_over_12_l399_399030

-- Defining the context of triangle ABC.
variables {A B C : Type} [AffineGeometry A] [AffineGeometry B] [AffineGeometry C]

-- Given conditions
def angle_C_is_90 (ABC : Triangle A B C) : Prop := ABC.angle C = 90
def AB_equals_13 (ABC : Triangle A B C) : Prop := ABC.side_length AB = 13
def BC_equals_5 (ABC : Triangle A B C) : Prop := ABC.side_length BC = 5

-- Concluding the result on tan A
theorem tan_A_is_5_over_12 (ABC : Triangle A B C)
  (h1 : angle_C_is_90 ABC)
  (h2 : AB_equals_13 ABC)
  (h3 : BC_equals_5 ABC) : 
  ABC.tan A = 5 / 12 := sorry

end tan_A_is_5_over_12_l399_399030


namespace distance_between_vertices_l399_399738

theorem distance_between_vertices (a b : ℝ) (a_pos : a = real.sqrt 144) (h : ∀ x y, x^2 / 144 - y^2 / 49 = 1): (2 * a) = 24 := by
  have ha : a = 12 := by sorry
  have h2a : 2 * a = 24 := by linarith
  exact h2a

end distance_between_vertices_l399_399738


namespace audrey_not_dreaming_fraction_l399_399321

theorem audrey_not_dreaming_fraction :
  let cycle1_not_dreaming := 3 / 4
  let cycle2_not_dreaming := 5 / 7
  let cycle3_not_dreaming := 2 / 3
  let cycle4_not_dreaming := 4 / 7
  cycle1_not_dreaming + cycle2_not_dreaming + cycle3_not_dreaming + cycle4_not_dreaming = 227 / 84 :=
by
  let cycle1_not_dreaming := 3 / 4
  let cycle2_not_dreaming := 5 / 7
  let cycle3_not_dreaming := 2 / 3
  let cycle4_not_dreaming := 4 / 7
  sorry

end audrey_not_dreaming_fraction_l399_399321


namespace distance_between_signs_l399_399922

def total_distance : ℕ := 1000
def first_sign_distance : ℕ := 350
def distance_after_second_sign : ℕ := 275

theorem distance_between_signs : 
  let second_sign_distance := total_distance - distance_after_second_sign in
  second_sign_distance - first_sign_distance = 375 :=
by
  let second_sign_distance := total_distance - distance_after_second_sign
  have h1 : second_sign_distance = 725 := by sorry
  have h2 : second_sign_distance - first_sign_distance = 375 := by sorry
  exact h2

end distance_between_signs_l399_399922


namespace monotonicity_and_range_of_a_l399_399080

noncomputable def f (a x : ℝ) : ℝ :=
  a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, x ∈ (set.Ioo 0 (1 / a)) → deriv (λ x, f a x) x < 0) ∧
  (∀ x, x ∈ (set.Ioi (1 / a)) → deriv (λ x, f a x) x > 0) ∧
  (∀ a, a > Real.exp (-1) → ∃ x : ℝ, ∀ x, f a x > 0) :=
by
  sorry

end monotonicity_and_range_of_a_l399_399080


namespace max_value_of_A_l399_399643

theorem max_value_of_A (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) / (a^3 + b^3 + c^3 - 2 * a * b * c) ≤ 6 :=
sorry

end max_value_of_A_l399_399643


namespace initial_cakes_l399_399322

variable (friend_bought : Nat) (baker_has : Nat)

theorem initial_cakes (h1 : friend_bought = 140) (h2 : baker_has = 15) : 
  (friend_bought + baker_has = 155) := 
by
  sorry

end initial_cakes_l399_399322


namespace slope_of_line_l399_399829

theorem slope_of_line 
  (p l t : ℝ) (p_pos : p > 0)
  (h_parabola : (2:ℝ)*p = 4) -- Since the parabola passes through M(l,2)
  (h_incircle_center : ∃ (k m : ℝ), (k + 1 = 0) ∧ (k^2 - k - 2 = 0)) :
  ∃ (k : ℝ), k = -1 :=
by {
  sorry
}

end slope_of_line_l399_399829


namespace single_digit_solution_l399_399669

theorem single_digit_solution :
  ∃ A : ℕ, A < 10 ∧ A^3 = 210 + A ∧ A = 6 :=
by
  existsi 6
  sorry

end single_digit_solution_l399_399669


namespace tabitha_final_amount_l399_399159

def initial_amount : ℝ := 25
def amount_given_to_mom : ℝ := 8
def items_bought : ℕ := 5
def cost_per_item : ℝ := 0.5

theorem tabitha_final_amount :
  let remaining_after_mom := initial_amount - amount_given_to_mom in
  let remaining_after_investment := remaining_after_mom / 2 in
  let spent_on_items := items_bought * cost_per_item in
  let final_amount := remaining_after_investment - spent_on_items in
  final_amount = 6 := by
  sorry

end tabitha_final_amount_l399_399159


namespace Julie_simple_interest_l399_399051

variable (S : ℝ) (r : ℝ) (A : ℝ) (C : ℝ)

def initially_savings (S : ℝ) := S = 784
def half_savings_in_each_account (S A : ℝ) := A = S / 2
def compound_interest_after_two_years (A r : ℝ) := A * (1 + r)^2 - A = 120

theorem Julie_simple_interest
  (S : ℝ) (r : ℝ) (A : ℝ)
  (h1 : initially_savings S)
  (h2 : half_savings_in_each_account S A)
  (h3 : compound_interest_after_two_years A r) :
  A * r * 2 = 112 :=
by 
  sorry

end Julie_simple_interest_l399_399051


namespace cookie_radius_l399_399167

theorem cookie_radius (x y : ℝ) (h : x^2 + y^2 + 2 * x - 4 * y = 4) : 
  ∃ r : ℝ, (x + 1)^2 + (y - 2)^2 = r^2 ∧ r = 3 := by
  sorry

end cookie_radius_l399_399167


namespace eccentricity_of_hyperbola_l399_399794

variables (a b c e : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c = Real.sqrt (a^2 + b^2))
variable (h4 : 3 * -(a^2 / c) + c = a^2 * c / (b^2 - a^2) + c)
variable (h5 : e = c / a)

theorem eccentricity_of_hyperbola : e = Real.sqrt 3 :=
by {
  sorry
}

end eccentricity_of_hyperbola_l399_399794


namespace z_cubed_eq_one_l399_399459
open Complex

theorem z_cubed_eq_one :
  let z := cos (2 * π / 3) - sin (π / 3) * I in
  z ^ 3 = 1 :=
by
  sorry

end z_cubed_eq_one_l399_399459


namespace duke_scored_more_three_pointers_l399_399450

theorem duke_scored_more_three_pointers :
  let old_record := 257
  let points_to_tie := 17
  let points_above_record := 5
  let free_throws := 5
  let regular_baskets := 4
  let normal_three_pointers := 2
  let points_per_free_throw := 1
  let points_per_regular_basket := 2
  let points_per_three_pointer := 3
  let points_in_final_game := points_to_tie + points_above_record
  let points_from_free_throws := free_throws * points_per_free_throw
  let points_from_regular_baskets := regular_baskets * points_per_regular_basket
  let total_points_from_free_throws_and_regular_baskets := points_from_free_throws + points_from_regular_baskets
  let points_from_three_pointers := points_in_final_game - total_points_from_free_throws_and_regular_baskets
  let three_pointers_in_final_game := points_from_three_pointers / points_per_three_pointer
  in three_pointers_in_final_game - normal_three_pointers = 1 :=
by
  sorry

end duke_scored_more_three_pointers_l399_399450


namespace exists_solution_in_interval_l399_399578

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem exists_solution_in_interval : ∃ x ∈ Ioo 0 1, f x = 0 := by
  let f := λ x : ℝ, 2^x + x - 2
  have h0 : f 0 < 0 := by norm_num
  have h1 : f 1 > 0 := by norm_num
  exact intermediate_value_Ioo 0 1 h0 h1

end exists_solution_in_interval_l399_399578


namespace distance_from_edge_l399_399674

theorem distance_from_edge (wall_width picture_width x : ℕ) (h_wall : wall_width = 24) (h_picture : picture_width = 4) (h_centered : x + picture_width + x = wall_width) : x = 10 := by
  -- Proof is omitted
  sorry

end distance_from_edge_l399_399674


namespace relationship_not_universal_l399_399877

-- Definitions based on the conditions
variable (ABC : Type)
variables (l R r : ℝ)

-- Conditions: There exists a triangle ΔABC with perimeter l, circumradius R, and inradius r
axioms (is_triangle : triangle ABC)
  (perimeter : perimeter ABC = l)
  (circumradius : circumradius ABC = R)
  (inradius : inradius ABC = r)

-- The main theorem based on the question and answer
theorem relationship_not_universal :
  ¬ ((∀ {ABC : Type} (l R r : ℝ), triangle ABC → perimeter ABC = l → circumradius ABC = R → inradius ABC = r → l > R + r)
     ∨ (∀ {ABC : Type} (l R r : ℝ), triangle ABC → perimeter ABC = l → circumradius ABC = R → inradius ABC = r → l ≤ R + r)
     ∨ (∀ {ABC : Type} (l R r : ℝ), triangle ABC → perimeter ABC = l → circumradius ABC = R → inradius ABC = r → (1 / 6) < R + r ∧ R + r < 6 * l)) :=
sorry

end relationship_not_universal_l399_399877


namespace graph_symmetric_intersect_one_point_l399_399968

theorem graph_symmetric_intersect_one_point (a b c d : ℝ) :
  (∀ x : ℝ, 2a + (1 / (x - b)) = 2c + (1 / (x - d)) = a + c) →
  (x = (b + d) / 2) →
  ((a - c) * (b - d) = 2) := 
sorry

end graph_symmetric_intersect_one_point_l399_399968


namespace monotonicity_and_range_of_a_l399_399079

noncomputable def f (a x : ℝ) : ℝ :=
  a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, x ∈ (set.Ioo 0 (1 / a)) → deriv (λ x, f a x) x < 0) ∧
  (∀ x, x ∈ (set.Ioi (1 / a)) → deriv (λ x, f a x) x > 0) ∧
  (∀ a, a > Real.exp (-1) → ∃ x : ℝ, ∀ x, f a x > 0) :=
by
  sorry

end monotonicity_and_range_of_a_l399_399079


namespace quadratic_distinct_real_roots_l399_399780

-- Defining the main hypothesis
theorem quadratic_distinct_real_roots (k : ℝ) :
  (k < 4 / 3) ∧ (k ≠ 1) ↔ (∀ x : ℂ, ((k-1) * x^2 - 2 * x + 3 = 0) → ∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ ((k-1) * x₁ ^ 2 - 2 * x₁ + 3 = 0) ∧ ((k-1) * x₂ ^ 2 - 2 * x₂ + 3 = 0)) := by
sorry

end quadratic_distinct_real_roots_l399_399780


namespace no_integer_solution_l399_399136

theorem no_integer_solution (x y z : ℤ) (h : x ≠ 0) : ¬(2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) :=
sorry

end no_integer_solution_l399_399136


namespace find_beta_and_mn_l399_399236

-- Definitions for side lengths of the triangle
def AB : ℝ := 15
def BC : ℝ := 26
def CA : ℝ := 25

-- Semiperimeter of the triangle
def s : ℝ := (AB + BC + CA) / 2

-- Area of the triangle ABC using Heron's formula
def Area_ABC : ℝ := Real.sqrt (s * (s - AB) * (s - BC) * (s - CA))

-- Area of the rectangle PQRS
def Area_PQRS (ω : ℝ) (α β : ℝ) : ℝ := α * ω - β * ω^2

-- Main statement to prove
theorem find_beta_and_mn :
    let β := 33 / 28
    ∃ (m n : ℕ), Nat.coprime m n ∧ β = m / n ∧ m + n = 61 :=
by
  -- Proof is omitted (sorry is used here to skip the proof)
  sorry

end find_beta_and_mn_l399_399236


namespace intersecting_lines_area_l399_399227

theorem intersecting_lines_area : 
  let l1 := λ x : ℝ, x
  let l2 := λ x : ℝ, -7
  let intersection := (-7, -7)
  let base := Real.sqrt ((0 - (-7))^2 + 0^2)
  let height := Real.sqrt ((-7 - 0)^2 + (-7 - 0)^2)
  let area := (1/2 : ℝ) * base * height
  area = 24.5 := 
by 
  let l1 := λ x : ℝ, x
  let l2 := λ x : ℝ, -7
  let intersection := (-7, -7)
  let base := Real.sqrt ((0 - (-7))^2 + 0^2)
  let height := Real.sqrt ((-7 - (-7))^2 + (0 - 0)^2)
  let area := (1 / 2 : ℝ) * base * height
  have h_base: base = 7 := by sorry
  have h_height: height = 7 := by sorry
  have h_area: area = (1/2 : ℝ) * 7 * 7 := by sorry
  have h_final: (1 / 2 : ℝ) * 7 * 7 = 24.5 := by norm_num
  exact h_final

end intersecting_lines_area_l399_399227


namespace properties_of_lines_and_planes_l399_399613

-- Define the geometric entities: Points, Lines, and Planes
universe u
variable {Point : Type u}
variable {Line : Type u}
variable {Plane : Type u}

-- Define the conditions of the problem
variables {p1 p2 : Point} (ℓ : Line) (π : Plane)

-- Define the properties being referred to in the problem
def line_in_plane (ℓ : Line) (π : Plane) : Prop :=
  ∀ (p1 p2 : Point), p1 ∈ ℓ ∧ p2 ∈ ℓ ∧ p1 ∈ π ∧ p2 ∈ π → ∀ p : Point, p ∈ ℓ → p ∈ π

def line_divides_plane (ℓ : Line) (π : Plane) : Prop :=
  ∀ (R1 R2 : set Point), convex R1 ∧ convex R2 ∧ ∀ p : Point, p ∈ (R1 ∪ R2) → p ∉ (R1 ∩ R2) → (p ∉ ℓ ∨ p ∈ π) 

-- Theorem statement
theorem properties_of_lines_and_planes (ℓ : Line) (π : Plane) :
  (line_in_plane ℓ π ∧ line_divides_plane ℓ π) → 
  (line_in_plane ℓ π ∧ line_divides_plane ℓ π) :=
  by sorry

end properties_of_lines_and_planes_l399_399613


namespace true_proposition_l399_399013

-- Define propositions p and q
variable (p q : Prop)

-- Assume p is true and q is false
axiom h1 : p
axiom h2 : ¬q

-- Prove that p ∧ ¬q is true
theorem true_proposition (p q : Prop) (h1 : p) (h2 : ¬q) : p ∧ ¬q :=
by
  sorry

end true_proposition_l399_399013


namespace arithmetic_mean_reciprocal_primes_l399_399343

theorem arithmetic_mean_reciprocal_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := (1 : ℚ) / p1
  let r2 := (1 : ℚ) / p2
  let r3 := (1 : ℚ) / p3
  let r4 := (1 : ℚ) / p4
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 := by
sorry

end arithmetic_mean_reciprocal_primes_l399_399343


namespace tabitha_final_amount_l399_399162

theorem tabitha_final_amount 
  (initial_amount : ℕ := 25)
  (given_amount : ℕ := 8)
  (item_count : ℕ := 5)
  (item_cost : ℚ := 0.5) :
  let after_giving := initial_amount - given_amount,
      after_investing := after_giving / 2,
      total_spending := item_count * item_cost,
      final_amount := after_investing - total_spending
  in final_amount = 6 := 
by
  -- use actual values
  let after_giving := (25 : ℚ) - 8,
  let after_investing := after_giving / 2,
  let total_spending := (5 : ℚ) * 0.5,
  have h_after_giving : after_giving = 17 := by norm_num,
  have h_after_investing : after_investing = 8.5 := by norm_num,
  have h_total_spending : total_spending = 2.5 := by norm_num,
  let final_amount := after_investing - total_spending,
  have h_final_amount : final_amount = 6 := by norm_num,
  exact h_final_amount

end tabitha_final_amount_l399_399162


namespace duke_scored_more_three_pointers_l399_399451

theorem duke_scored_more_three_pointers :
  let old_record := 257
  let points_to_tie := 17
  let points_above_record := 5
  let free_throws := 5
  let regular_baskets := 4
  let normal_three_pointers := 2
  let points_per_free_throw := 1
  let points_per_regular_basket := 2
  let points_per_three_pointer := 3
  let points_in_final_game := points_to_tie + points_above_record
  let points_from_free_throws := free_throws * points_per_free_throw
  let points_from_regular_baskets := regular_baskets * points_per_regular_basket
  let total_points_from_free_throws_and_regular_baskets := points_from_free_throws + points_from_regular_baskets
  let points_from_three_pointers := points_in_final_game - total_points_from_free_throws_and_regular_baskets
  let three_pointers_in_final_game := points_from_three_pointers / points_per_three_pointer
  in three_pointers_in_final_game - normal_three_pointers = 1 :=
by
  sorry

end duke_scored_more_three_pointers_l399_399451


namespace quarter_circle_perimeter_l399_399302

theorem quarter_circle_perimeter (side_length : ℝ) (h : side_length = 4 / real.pi ) :
  let diameter := side_length in
  let circumference := π * diameter in
  let quarter_circle_circumference := circumference / 4 in
  let total_perimeter := 4 * quarter_circle_circumference in
  total_perimeter = 4 :=
by
  sorry

end quarter_circle_perimeter_l399_399302


namespace find_positive_real_unique_solution_l399_399392

theorem find_positive_real_unique_solution (x : ℝ) (h : 0 < x ∧ (x - 6) / 16 = 6 / (x - 16)) : x = 22 :=
sorry

end find_positive_real_unique_solution_l399_399392


namespace car_speed_l399_399611

-- Definitions based on the conditions
def distance_marker (a b : ℕ) : ℕ := 10 * a + b
def distance_marker_reversed (a b : ℕ) : ℕ := 10 * b + a

-- Speed calculation function based on given conditions
noncomputable def speed (a b A : ℕ) : ℕ :=
  if (A = 10 * a + b + 96) && (A = 10 * b + a + 96)
  then 96
  else 0

-- Lean 4 statement (question, conditions, and correct answer)
theorem car_speed (a b A : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : A = 19 * b - 8 * a): 
  (b = 6 ∧ a = 1 ∧ A = 106) → speed a b A = 96 := 
begin
  intros,
  unfold speed,
  split_ifs,
  -- Skipping proof
  sorry
end

end car_speed_l399_399611


namespace max_tangent_BAD_l399_399029

-- Define the conditions
variables (A B C D : Point) (BC : ℝ) (angleC : ℝ)
variable [midpoint D B C]

-- Given conditions
def conditions : Prop := 
  ∠C = 45 ∧ BC = 6

-- Define the target to prove
def largest_tangent_BAD : ℝ := 3 * sqrt 2

-- The theorem to prove
theorem max_tangent_BAD 
  (h : conditions A B C D BC angleC) : tangent_angle_BAD A B C D ≤ largest_tangent_BAD :=
sorry

end max_tangent_BAD_l399_399029


namespace sqrt_log_expression_eq_one_l399_399356

theorem sqrt_log_expression_eq_one :
  sqrt((log 2 5)^2 - 6 * log 2 5 + 9) + log 2 3 - log 2 (12 / 5) = 1 := by
  sorry

end sqrt_log_expression_eq_one_l399_399356


namespace triangle_CME_and_ABD_are_similar_EM_perpendicular_AB_l399_399027

theorem triangle_CME_and_ABD_are_similar
  (A B C M D E : Type)
  [triangle A B C]
  (h1 : angle A C B = 90)
  (h2 : is_midpoint M A B)
  (h3 : parallel (line_through M) (line_through B C) (intersection D (line_through A C M)))
  (h4 : is_midpoint E C D)
  (h5 : perpendicular (line_through B D) (line_through C M)) :
  similar (triangle C M E) (triangle A B D) := sorry

theorem EM_perpendicular_AB
  (A B C M D E : Type)
  [triangle A B C]
  (h1 : angle A C B = 90)
  (h2 : is_midpoint M A B)
  (h3 : parallel (line_through M) (line_through B C) (intersection D (line_through A C M)))
  (h4 : is_midpoint E C D)
  (h5 : perpendicular (line_through B D) (line_through C M))
  (h6 : similar (triangle C M E) (triangle A B D)) :
  perpendicular (line_through E M) (line_through A B) := sorry

end triangle_CME_and_ABD_are_similar_EM_perpendicular_AB_l399_399027


namespace edge_lengths_correct_l399_399953

noncomputable def edge_lengths (a b c : ℝ) : Prop :=
  (a * b : b * c) = (16 : 21) ∧ 
  (a * b : a * c) = (16 : 28) ∧ 
  √(a^2 + b^2 + c^2) = 29

theorem edge_lengths_correct : ∃ a b c : ℝ, edge_lengths a b c :=
begin
  use [16, 12, 21],
  simp [edge_lengths],
  split,
  {
    -- Prove ab : bc = 16 : 21
    exact sorry,
  },
  split,
  {
    -- Prove ab : ac = 16 : 28
    exact sorry,
  },
  {
    -- Prove √(a² + b² + c²) = 29
    exact sorry,
  },
end

end edge_lengths_correct_l399_399953


namespace one_fifth_of_five_times_nine_l399_399385

theorem one_fifth_of_five_times_nine (a b : ℕ) (h1 : a = 5) (h2 : b = 9) : (1 / 5 : ℚ) * (a * b) = 9 := by
  sorry

end one_fifth_of_five_times_nine_l399_399385


namespace compare_numbers_l399_399785

noncomputable def a : ℝ := 2^(-1/3)
noncomputable def b : ℝ := Real.log 1/3 / Real.log 2
noncomputable def c : ℝ := Real.log 1/3 / Real.log (1/2)

theorem compare_numbers : b < a ∧ a < c := by
  sorry

end compare_numbers_l399_399785


namespace smallest_value_of_n_l399_399901

def g (n : ℕ) : ℕ :=
  Inf { m : ℕ | factorial m ∣ n }

theorem smallest_value_of_n (n : ℕ) (h : 35 ∣ n) : n = 595 ↔ g(n) > 15 :=
begin
  sorry
end

end smallest_value_of_n_l399_399901


namespace calculate_total_surface_area_l399_399655

noncomputable def surfaceArea (total_cube_side : ℕ) (small_cube_side : ℕ) 
  (num_small_cubes : ℕ) (removed_cubes : ℕ) 
  (additional_surface_exposed : ℕ) : ℕ :=
  let remaining_cubes := num_small_cubes - removed_cubes
  remaining_cubes * (6 * small_cube_side * small_cube_side - additional_surface_exposed)

theorem calculate_total_surface_area 
  (total_cube_side : ℕ = 12) 
  (small_cube_side : ℕ = 3) 
  (num_small_cubes : ℕ = 64) 
  (removed_cubes : ℕ = 32) 
  (additional_surface_exposed : ℕ = 24) : 
  surfaceArea total_cube_side small_cube_side num_small_cubes removed_cubes additional_surface_exposed = 2496 := 
by
  sorry

end calculate_total_surface_area_l399_399655


namespace housewife_more_kgs_l399_399722

theorem housewife_more_kgs (P R money more_kgs : ℝ)
  (hR: R = 40)
  (hReduction: R = P - 0.25 * P)
  (hMoney: money = 800)
  (hMoreKgs: more_kgs = (money / R) - (money / P)) :
  more_kgs = 5 :=
  by
    sorry

end housewife_more_kgs_l399_399722


namespace black_triangles_count_l399_399231

theorem black_triangles_count (n : ℕ) (h : n = 2008) :
  let black_per_pattern := 4
  let pattern_length := 6

  -- Number of complete patterns
  let complete_patterns := n / pattern_length
  -- Remaining triangles after accounting for complete patterns
  let remainder := n % pattern_length

  -- Total black triangles in complete patterns
  let total_black_complete_patterns := complete_patterns * black_per_pattern

  -- Number of black triangles in the remaining part
  let additional_black := match remainder with
    | 1 => 1
    | 2 => 2
    | 3 => 2
    | 4 => 3
    | 5 => 3
    | 0 => 0

  -- Total black triangles
  in (total_black_complete_patterns + additional_black) = 1004 :=
begin
  sorry
end

end black_triangles_count_l399_399231


namespace sum_of_squares_exceeds_sum_condition_sum_of_squares_is_194_l399_399592

structure ConsecutiveNatNumbers (x : ℕ) :=
  (x_1 x :: ℕ)
  (nat_condition : x_1 = x - 1 ∧ x_3 = x + 1)

theorem sum_of_squares_exceeds_sum_condition (x : ℕ) (h : ConsecutiveNatNumbers x):
  (x - 1) ^ 2 + x ^ 2 + (x + 1) ^ 2 = 8 * (x - 1 + x + x + 1) + 2 :=
by
  sorry

theorem sum_of_squares_is_194 : ∃ x : ℕ, 
  (∃ h : ConsecutiveNatNumbers x, 
  sum_of_squares_exceeds_sum_condition x h → 
  (x - 1) ^ 2 + x ^ 2 + (x + 1) ^ 2 = 194) :=
by
  use 8
  sorry

end sum_of_squares_exceeds_sum_condition_sum_of_squares_is_194_l399_399592


namespace josh_total_payment_with_tax_and_discount_l399_399050

-- Definitions
def total_string_cheeses (pack1 : ℕ) (pack2 : ℕ) (pack3 : ℕ) : ℕ :=
  pack1 + pack2 + pack3

def total_cost_before_tax_and_discount (n : ℕ) (cost_per_cheese : ℚ) : ℚ :=
  n * cost_per_cheese

def discount_amount (cost : ℚ) (discount_rate : ℚ) : ℚ :=
  cost * discount_rate

def discounted_cost (cost : ℚ) (discount : ℚ) : ℚ :=
  cost - discount

def sales_tax_amount (cost : ℚ) (tax_rate : ℚ) : ℚ :=
  cost * tax_rate

def total_cost (cost : ℚ) (tax : ℚ) : ℚ :=
  cost + tax

-- The statement
theorem josh_total_payment_with_tax_and_discount :
  let cost_per_cheese := 0.10
  let discount_rate := 0.05
  let tax_rate := 0.12
  total_cost (discounted_cost (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese)
                              (discount_amount (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese) discount_rate))
             (sales_tax_amount (discounted_cost (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese)
                                               (discount_amount (total_cost_before_tax_and_discount (total_string_cheeses 18 22 24) cost_per_cheese) discount_rate)) tax_rate) = 6.81 := 
  sorry

end josh_total_payment_with_tax_and_discount_l399_399050


namespace eden_bears_count_l399_399710

-- Define the main hypothesis
def initial_bears : Nat := 20
def favorite_bears : Nat := 8
def remaining_bears := initial_bears - favorite_bears

def number_of_sisters : Nat := 3
def bears_per_sister := remaining_bears / number_of_sisters

def eden_initial_bears : Nat := 10
def eden_final_bears := eden_initial_bears + bears_per_sister

theorem eden_bears_count : eden_final_bears = 14 :=
by
  unfold eden_final_bears eden_initial_bears bears_per_sister remaining_bears initial_bears favorite_bears
  norm_num
  sorry

end eden_bears_count_l399_399710


namespace symmetry_one_common_point_l399_399970

theorem symmetry_one_common_point
  (a b c d : ℝ)
  (f g : ℝ → ℝ)
  (hx : ∀ x, f x = 2 * a + 1 / (x - b))
  (hy : ∀ x, g x = 2 * c + 1 / (x - d))
  (sym_point : ℝ × ℝ := (b + d) / 2, a + c)
  (symmetric : ∀ x, f (2 * sym_point.1 - x) = 2 * sym_point.2 - f x ∧ g (2 * sym_point.1 - x) = 2 * sym_point.2 - g x)
  (common_point : ∃ x, f x = g x)
  : (a - c) * (b - d) = 2 :=
begin
  sorry
end

end symmetry_one_common_point_l399_399970


namespace median_of_set_l399_399518

noncomputable def median_set (c : ℤ) (d : ℝ) : ℝ :=
  if (c ≠ 0 ∧ d > 0 ∧ c * d^3 = Real.log10 d) then
    d
  else
    0  -- Include an arbitrary value for invalid inputs

theorem median_of_set {c : ℤ} {d : ℝ} (h1 : c ≠ 0) (h2 : d > 0) (h3 : c * d^3 = Real.log10 d) :
  median_set c d = d :=
by
  sorry

end median_of_set_l399_399518


namespace tabitha_final_amount_l399_399161

theorem tabitha_final_amount 
  (initial_amount : ℕ := 25)
  (given_amount : ℕ := 8)
  (item_count : ℕ := 5)
  (item_cost : ℚ := 0.5) :
  let after_giving := initial_amount - given_amount,
      after_investing := after_giving / 2,
      total_spending := item_count * item_cost,
      final_amount := after_investing - total_spending
  in final_amount = 6 := 
by
  -- use actual values
  let after_giving := (25 : ℚ) - 8,
  let after_investing := after_giving / 2,
  let total_spending := (5 : ℚ) * 0.5,
  have h_after_giving : after_giving = 17 := by norm_num,
  have h_after_investing : after_investing = 8.5 := by norm_num,
  have h_total_spending : total_spending = 2.5 := by norm_num,
  let final_amount := after_investing - total_spending,
  have h_final_amount : final_amount = 6 := by norm_num,
  exact h_final_amount

end tabitha_final_amount_l399_399161


namespace sum_of_powers_eq_zero_l399_399526

theorem sum_of_powers_eq_zero
  (a b c : ℝ)
  (n : ℝ)
  (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = 0) :
  a^(2* ⌊n⌋ + 1) + b^(2* ⌊n⌋ + 1) + c^(2* ⌊n⌋ + 1) = 0 := by
  sorry

end sum_of_powers_eq_zero_l399_399526


namespace complementary_angles_decrease_percentage_l399_399197

theorem complementary_angles_decrease_percentage :
  ∀ (x : ℝ), (3 * x + 7 * x = 90) →
  (3 * x * 1.2 + 7 * x = 90) →
  (3 * x > 0) →
  (7 * x > 0) →
  let original_larger_angle : ℝ := 7 * x in
  let new_smaller_angle : ℝ := 3 * x * 1.2 in
  let new_larger_angle : ℝ := 90 - new_smaller_angle in
  let decrease : ℝ := original_larger_angle - new_larger_angle in
  let percentage_decrease : ℝ := (decrease / original_larger_angle) * 100 in
  percentage_decrease = 8.57 := 
sorry

end complementary_angles_decrease_percentage_l399_399197


namespace problem_statement_l399_399515

variables {A B C G P Q : Type} [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ G] 
          [AffineSpace ℝ P] [AffineSpace ℝ Q]
variables {m n : ℝ}
variables [affine_is_perfect _ _] [affine_is_perfect _ _] [affine_is_perfect _ _] 
          [affine_is_perfect _ _] [affine_is_perfect _ _] [affine_is_perfect _ _]
variables {C A B centroid : A}
variables {CP CA CQ CB : AffineMap A B}
variables (triangle_centroid : A → B → C → G)
variables {m n : ℝ}

axiom centroid_def (CA CB : ℝ → B) (AB : Type) : 
  triangle_centroid A B C = G → centroid A B C 

axiom line_through_centroid (P Q : Type) :
  line_through P Q G = m-→ CA ∧ n-→ CB


theorem problem_statement (h1 : triangle_centroid A B C = G)
    (h2 : line_through_centroid P Q G) 
    (h3 : CA = m ⋅ CA)
    (h4 : CB = n ⋅ CB) :
  1/m + 1/n = 3 :=
  sorry

end problem_statement_l399_399515


namespace ratio_of_adults_to_children_l399_399571

-- Defining conditions as functions
def admission_fees_condition (a c : ℕ) : ℕ := 30 * a + 15 * c

-- Stating the problem
theorem ratio_of_adults_to_children (a c : ℕ) 
  (h1 : admission_fees_condition a c = 2250)
  (h2 : a ≥ 1) 
  (h3 : c ≥ 1) 
  : a / c = 2 := 
sorry

end ratio_of_adults_to_children_l399_399571


namespace rectangle_width_length_ratio_l399_399475

theorem rectangle_width_length_ratio (w : ℕ) (h : ℕ) (P : ℕ) (H1 : h = 10) (H2 : P = 30) (H3 : 2 * w + 2 * h = P) :
  w / h = 1 / 2 :=
by
  sorry

end rectangle_width_length_ratio_l399_399475


namespace triangle_BH_perp_QH_l399_399028

theorem triangle_BH_perp_QH
  (A B C I M P H Q : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited I]
  [Inhabited M] [Inhabited P] [Inhabited H] [Inhabited Q]
  (triangle_ABC : Triangle ABC)
  (is_incenter : Incenter I ABC)
  (is_midpoint_M : Midpoint M B I)
  (on_segment_P : OnSegment P A C)
  (AP_eq_3PC : AP = 3 * PC)
  (on_line_H_PI : OnLine H P I)
  (MH_perp_PH: Perpendicular M H P H)
  (Q_arc_midpoint: MidpointArc Q A B (Circumcircle ABC)) :
  Perpendicular B H Q H := sorry

end triangle_BH_perp_QH_l399_399028


namespace a_3_value_l399_399878

variable {a : ℕ → ℝ} -- Define the sequence a: ℕ → ℝ

-- Condition: Given the property a_2 + a_4 = 5
axiom a2_a4_eq_five : a 2 + a 4 = 5

-- Definition: Arithmetic sequence which implies that a_3 = (a_2 + a_4) / 2
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n

-- Main theorem: Prove that a_3 = 5 / 2
theorem a_3_value (a : ℕ → ℝ) [arithmetic_sequence a] : a 3 = 5 / 2 :=
by
  -- We use the given condition and properties to prove this
  sorry

end a_3_value_l399_399878


namespace hyperbola_vertex_distance_l399_399763

theorem hyperbola_vertex_distance : 
  ∀ x y: ℝ, (x^2 / 144 - y^2 / 49 = 1) → (∃ a: ℝ, a = 12 ∧ 2 * a = 24) :=
by 
  sorry

end hyperbola_vertex_distance_l399_399763


namespace understanding_related_to_gender_linear_regression_equation_l399_399609

section Part1

variable (a b c d n : ℕ) (critical_value : ℕ)
variable (conf_cond : ℕ -> Prop)

-- Define the K2 formula
def K2 (a b c d n : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / (a + b) / (c + d) / (a + c) / (b + d)

-- Calculate K2 for given values
theorem understanding_related_to_gender (a b c d n : ℕ) (critical_value : ℚ)
  (ha : a = 20) (hb : b = 30) (hc : c = 10) (hd : d = 40) (hn : n = 100) (hcv : critical_value = 3.841):
  K2 a b c d n > critical_value :=
by
  rw [ha, hb, hc, hd, hn]
  simp [K2]
  norm_num
  sorry

end Part1

section Part2

variables {α : Type*}

def mean (xs : list ℚ) : ℚ := (xs.sum) / xs.length

def lin_reg (ts ys : list ℚ) (n : ℕ) : ℚ × ℚ :=
let t_mean := mean ts;
    y_mean := mean ys;
    b := (list.sum (list.map2 (λ t y, t * y) ts ys) - n * t_mean * y_mean) /
         (list.sum (list.map (λ t, t ^ 2) ts) - n * t_mean ^ 2);
    a := y_mean - b * t_mean
in (b, a)

theorem linear_regression_equation (ts ys : list ℚ) (ts_list ys_list : list ℚ) (n : ℕ)
  (h_ts : ts = [2, 4, 6, 8, 10]) (h_ys : ys = [0.3, 0.3, 0.5, 0.7, 0.8]) (h_n : n = 5) :
  lin_reg ts ys n = (0.07, 0.1) :=
by
  rw [h_ts, h_ys, h_n]
  simp [lin_reg, mean]
  norm_num
  sorry

end Part2

end understanding_related_to_gender_linear_regression_equation_l399_399609


namespace value_range_distinct_set_l399_399994

theorem value_range_distinct_set {x : ℝ} :
  x ≠ 0 ∧ x ≠ 1 ↔ ∃ y z : ℝ, y ≠ z ∧ {y, z} = {x^2 + x, 2x} :=
by
  sorry

end value_range_distinct_set_l399_399994


namespace painted_cubes_count_l399_399286

-- Definitions and conditions
def edge_length : ℕ := 6
def num_smaller_cubes : ℕ := edge_length ^ 3
def smaller_cube_edge_length : ℕ := 1

-- Properties of the stripes
def blue_stripe (x y : ℕ) : Prop := (1 ≤ x ∧ x ≤ edge_length) ∧ (y = (edge_length + 1) / 2)
def orange_stripe (x y : ℕ) : Prop := (1 ≤ x ∧ x <= edge_length) ∧ (y == (edge_length + 1) / 2)
def green_stripe (x y : ℕ) : Prop := (1 ≤ x ∧ x ≤ edge_length) ∧ (y == x) 

-- The problem
theorem painted_cubes_count :
∀ (f : ℕ → ℕ → Prop)
    (B O G : Prop),
    (B = blue_stripe) -> (O = orange_stripe) -> (G = green_stripe) ->
    preimages_set : ∀ f x y, if (f x y )= true then (x,y) ∈ painted_set ∀ edges of cube →(B = O) ∨ (O = G)∨(G = B) → 12 :=
by
  sorry -/

end painted_cubes_count_l399_399286


namespace solve_for_x_l399_399854

theorem solve_for_x (x : ℝ) (h : log 2 (x + 1) = 3) : x = 7 :=
sorry

end solve_for_x_l399_399854


namespace jeff_cat_shelter_l399_399506

theorem jeff_cat_shelter :
  let initial_cats := 20
  let monday_cats := 2
  let tuesday_cats := 1
  let people_adopted := 3
  let cats_per_person := 2
  let total_cats := initial_cats + monday_cats + tuesday_cats
  let adopted_cats := people_adopted * cats_per_person
  total_cats - adopted_cats = 17 := 
by
  sorry

end jeff_cat_shelter_l399_399506


namespace average_and_difference_l399_399955

theorem average_and_difference
  (x y : ℚ) 
  (h1 : (15 + 24 + x + y) / 4 = 20)
  (h2 : x - y = 6) :
  x = 23.5 ∧ y = 17.5 := by
  sorry

end average_and_difference_l399_399955


namespace roots_expression_value_l399_399517

noncomputable def polynomial_roots (a b c d : ℝ) : Prop :=
  a + b + c + d = 34 ∧ abc + abd + acd + bcd = -42 ∧ abcd = -8

noncomputable def root_expression (a b c d : ℝ) : ℝ :=
  (a^2 + b^2 + c^2 + d^2) / -7

theorem roots_expression_value (a b c d : ℝ) (h : polynomial_roots a b c d) : 
  root_expression a b c d = -161 := 
sorry

end roots_expression_value_l399_399517


namespace max_value_expression_l399_399645

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∀ a b c > 0, a = b ∧ b = c → (a^2*(b+c) + b^2*(c+a) + c^2*(a+b)) / (a^3 + b^3 + c^3 - 2*a*b*c) ≤ 6) :=
begin
  sorry
end

end max_value_expression_l399_399645


namespace solve_cosine_equation_l399_399148

noncomputable def solution_set : set ℝ :=
  { x | (x = π/2 ∨ x = π/6 ∨ x = 7*π/6 ∨ x = 5*π/6 ∨ x = 11*π/6 ∨
          x = π/4 ∨ x = 5*π/4 ∨ x = 3*π/4 ∨ x = 7*π/4) }

theorem solve_cosine_equation :
  { x : ℝ | 0 ≤ x ∧ x < 2 * π ∧ cos x ^ 2 + cos (2 * x) ^ 2 + cos (3 * x) ^ 2 = 1 } = solution_set :=
by
  -- proof goes here
  sorry

end solve_cosine_equation_l399_399148


namespace monotonicity_no_zeros_range_of_a_l399_399065

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * (Real.log x) + 1

theorem monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1/a → ∀ t : ℝ, t ∈ Ioo x y → f' a t < 0 ) ∧ 
  (∀ x y : ℝ, 1/a < x ∧ x < y → ∀ t : ℝ, t ∈ Ioo x y → f' a t > 0 ) :=
sorry

theorem no_zeros_range_of_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → f a x > 0) : 
  a > 1/Real.exp 1 :=
sorry

end monotonicity_no_zeros_range_of_a_l399_399065


namespace problem_statement_l399_399809

theorem problem_statement {x₁ x₂ : ℝ} (h1 : 3 * x₁^2 - 9 * x₁ - 21 = 0) (h2 : 3 * x₂^2 - 9 * x₂ - 21 = 0) :
  (3 * x₁ - 4) * (6 * x₂ - 8) = -202 := sorry

end problem_statement_l399_399809


namespace frank_worked_days_l399_399404

def total_hours : ℝ := 8.0
def hours_per_day : ℝ := 2.0

theorem frank_worked_days :
  (total_hours / hours_per_day = 4.0) :=
by sorry

end frank_worked_days_l399_399404


namespace number_of_events_second_in_high_jump_l399_399598

variable (a b c : ℕ)
variable (n n_a n_b n_c n_a' n_b' n_c' n_a'' n_b'' n_c'' : ℕ)

-- Conditions
axiom scores_a : n_a * a + n_b * b + n_c * c = 22
axiom scores_b : n_a' * a + n_b' * b + n_c' * c = 9
axiom scores_c : n_a'' * a + n_b'' * b + n_c'' * c = 9
axiom total_events_a : n_a + n_b + n_c = n
axiom total_events_b : n_a' + n_b' + n_c' = n
axiom total_events_c : n_a'' + n_b'' + n_c'' = n
axiom b_won_100_meters : n_a' >= 1

theorem number_of_events (n : ℕ) : n = 4 := by
  sorry

theorem second_in_high_jump : "A" := by
  sorry

end number_of_events_second_in_high_jump_l399_399598


namespace red_envelope_distribution_l399_399469

theorem red_envelope_distribution :
  ∃ (total_ways : ℕ), total_ways = 36 ∧
  ∃ (C_4_2 : ℕ) (A_3_3 : ℕ),
  C_4_2 = 6 ∧ A_3_3 = 6 ∧ total_ways = C_4_2 * A_3_3 ∧
  -- Conditions:
  (∀ envelopes : fin 4 → fin 3, 
    (∀ (i j : fin 3), i ≠ j → ∃ k, envelopes k = i ∧ envelopes k = j → False) ∧ -- Each envelope to one person
    (∀ p : fin 3, ∃ e : fin 4, envelopes e = p)) -- Each person gets at least one envelope

end red_envelope_distribution_l399_399469


namespace height_increase_percentage_l399_399269

variable (B_height : ℕ) (A_height : ℕ)

-- Given condition
axiom height_relation : A_height = B_height * 6 / 10

theorem height_increase_percentage : 
  (B_height - A_height) / A_height * 100 ≈ 66.67 := by
  sorry

end height_increase_percentage_l399_399269


namespace ConfuciusBirthYear_l399_399178

-- Definitions based on the conditions provided
def birthYearAD (year : Int) : Int := year

def birthYearBC (year : Int) : Int := -year

theorem ConfuciusBirthYear :
  birthYearBC 551 = -551 :=
by
  sorry

end ConfuciusBirthYear_l399_399178


namespace proof1_proof2_l399_399351

def expr1 := 0.064 ^ (-1 / 3) - (-1 / 8) ^ 0 + 16 ^ (3 / 4) + 0.25 ^ (1 / 2)
def expr2 := (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9)

theorem proof1 : expr1 = 10 := by
  -- proof omitted
  sorry

theorem proof2 : expr2 = 5 / 4 := by
  -- proof omitted
  sorry

end proof1_proof2_l399_399351


namespace chords_concur_l399_399700

variable {Ω : Type} [MetricSpace Ω] [NormedSpace ℂ Ω]

-- Definitions based on the given conditions
variable (A1 A2 A3 A4 A5 A6 O B1 B2 B3 B4 B5 B6 : Ω)
variable (on_circle_Omega : ∀ x : Ω, dist x O = 1)
variable (concurr_chords : dist A1 O + dist A2 O = dist A3 O + dist A4 O ∧ dist A3 O + dist A4 O = dist A5 O + dist A6 O)
variable (common_point_Bi : ∀ i : Fin 6, dist Bi O = dist (complex.conj Bi) A1)

-- Proof that the chords concur at a point T
theorem chords_concur :
  ∃ T : Ω, ∃ B1 B2 B3 B4 B5 B6, 
    (dist B1 T + dist B2 T = dist B3 T + dist B4 T ∧ dist B5 T + dist B6 T = dist B3 T + dist B4 T) :=
sorry

end chords_concur_l399_399700


namespace num_triangles_with_area_2_l399_399962

-- Define the grid and points
def is_grid_point (x y : ℕ) : Prop := x ≤ 3 ∧ y ≤ 3

-- Function to calculate the area of a triangle using vertices (x1, y1), (x2, y2), and (x3, y3)
def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℕ) : ℤ := 
  (x1 * y2 + x2 * y3 + x3 * y1) 
  - (y1 * x2 + y2 * x3 + y3 * x1)

-- Check if the area is 2 (since we are dealing with a lattice grid, 
-- we can consider non-fractional form by multiplying by 2 to avoid half-area)
def has_area_2 (x1 y1 x2 y2 x3 y3 : ℕ) : Prop :=
  abs (area_of_triangle x1 y1 x2 y2 x3 y3) = 4

-- Define the main theorem that needs to be proved
theorem num_triangles_with_area_2 : 
  ∃ (n : ℕ), n = 64 ∧
  ∀ (x1 y1 x2 y2 x3 y3 : ℕ), 
  is_grid_point x1 y1 ∧ is_grid_point x2 y2 ∧ is_grid_point x3 y3 ∧ 
  has_area_2 x1 y1 x2 y2 x3 y3 → n = 64 :=
sorry

end num_triangles_with_area_2_l399_399962


namespace desk_chair_production_l399_399291

theorem desk_chair_production (x : ℝ) (h₁ : x > 0) (h₂ : 540 / x - 540 / (x + 2) = 3) : 
  ∃ x, 540 / x - 540 / (x + 2) = 3 := 
by
  sorry

end desk_chair_production_l399_399291


namespace option_C_is_neither_even_nor_odd_l399_399313

noncomputable def f_A (x : ℝ) : ℝ := x^2 + |x|
noncomputable def f_B (x : ℝ) : ℝ := 2^x - 2^(-x)
noncomputable def f_C (x : ℝ) : ℝ := x^2 - 3^x
noncomputable def f_D (x : ℝ) : ℝ := 1/(x+1) + 1/(x-1)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - (f x)

theorem option_C_is_neither_even_nor_odd : ¬ is_even f_C ∧ ¬ is_odd f_C :=
by
  sorry

end option_C_is_neither_even_nor_odd_l399_399313


namespace locus_of_centers_l399_399975

-- Given two fixed points A and B in the plane and a fixed radius a
variables {A B : ℝ × ℝ} {a : ℝ}

-- Definition of distance between two points
def dist (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Prove that if dist A B = 2 * a, then the locus of the centers of circles passing through A and B with radius a
-- is the midpoint of A and B. Otherwise, no such locus exists.
theorem locus_of_centers (A B : ℝ × ℝ) (a : ℝ) : 
  dist A B = 2 * a → 
  (∃ O : ℝ × ℝ, dist O A = a ∧ dist O B = a ∧ O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧ 
  (dist A B ≠ 2 * a → ¬∃ O : ℝ × ℝ, dist O A = a ∧ dist O B = a) :=
sorry

end locus_of_centers_l399_399975


namespace class_total_students_l399_399484

theorem class_total_students (x y : ℕ)
  (initial_absent : y = (1/6) * x)
  (after_sending_chalk : y = (1/5) * (x - 1)) :
  x + y = 7 :=
by
  sorry

end class_total_students_l399_399484


namespace value_of_expression_l399_399001

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 5 = 23 :=
by
  -- proof goes here
  sorry

end value_of_expression_l399_399001


namespace num_circles_tangent_l399_399239

noncomputable def num_tangent_circles (r1 r2 : ℝ) (d : ℝ) (r3 : ℝ) : ℕ :=
  if r1 = 2 ∧ r2 = 2 ∧ d = 4 ∧ r3 = 5 then 4 else sorry

theorem num_circles_tangent (C1 C2 : ℝ → ℝ) (P : ℝ × ℝ) :
  let r1 := 2
  let r2 := 2
  let r3 := 5
  let d := dist P (C2 0) in
  (r1 = 2 ∧ r2 = 2 ∧ C1 P = 0 ∧ C2 P = 0 ∧ d = 4) →
  num_tangent_circles r1 r2 d r3 = 4 :=
by
  intros
  sorry

end num_circles_tangent_l399_399239


namespace system_solution_l399_399151

theorem system_solution (a b : ℝ) 
  (h1 : a * real.sqrt a + b * real.sqrt b = 183) 
  (h2 : a * real.sqrt b + b * real.sqrt a = 182) : 
  (a = 196 / 9 ∧ b = 169 / 9) ∨ (a = 169 / 9 ∧ b = 196 / 9) := 
  sorry

end system_solution_l399_399151


namespace ellipse_standard_equation_l399_399813

-- Define the conditions
def equation1 (x y : ℝ) : Prop := x^2 + (y^2 / 2) = 1
def equation2 (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def equation3 (x y : ℝ) : Prop := x^2 + (y^2 / 4) = 1
def equation4 (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Define the points
def point1 (x y : ℝ) : Prop := (x = 1 ∧ y = 0)
def point2 (x y : ℝ) : Prop := (x = 0 ∧ y = 2)

-- Define the main theorem
theorem ellipse_standard_equation :
  (equation4 1 0 ∧ equation4 0 2) ↔
  ((equation1 1 0 ∧ equation1 0 2) ∨
   (equation2 1 0 ∧ equation2 0 2) ∨
   (equation3 1 0 ∧ equation3 0 2) ∨
   (equation4 1 0 ∧ equation4 0 2)) :=
by
  sorry

end ellipse_standard_equation_l399_399813


namespace tangent_from_A_equal_AC_l399_399473

open Real EuclideanGeometry

-- Define the main circle
def main_circle (O : Point) (A B : Point): Circle :=
  Circle.mk O (dist O A)

-- Define the smaller circle
def smaller_circle (O1 : Point) (r : Real): Circle :=
  Circle.mk O1 r

-- Given conditions: Diameter AB, chord CD perpendicular to AB and a smaller circle tangent to CD and arc CBD
variables {O A B C D O₁: Point} {CD AB: Line} {K: Point} 
  (h_diameter_AB : diameter AB O)
  (h_perpendicular_CD_AB : is_perpendicular CD AB)
  (tangent_smaller_circle_to_CD: is_tangent smaller_circle CD)
  (tangent_smaller_circle_to_arc_CBD: is_tangent smaller_circle (arc C B D))

-- Question: Prove tangent from A to smaller circle is equal to AC
theorem tangent_from_A_equal_AC (t : Real) :
  tangent_length A smaller_circle = dist A C :=
sorry

end tangent_from_A_equal_AC_l399_399473


namespace find_digit_l399_399012

theorem find_digit (p q r : ℕ) (hq : p ≠ q) (hr : p ≠ r) (hq' : q ≠ r) 
    (hp_pos : 0 < p ∧ p < 10)
    (hq_pos : 0 < q ∧ q < 10)
    (hr_pos : 0 < r ∧ r < 10)
    (h1 : 10 * p + q = 17)
    (h2 : 10 * p + r = 13)
    (h3 : p + q + r = 11) : 
    q = 7 :=
sorry

end find_digit_l399_399012


namespace trig_identity_proof_l399_399131

theorem trig_identity_proof :
  ∑ i in [1, 3, 5, 7].map (λ n, (Real.sin (n * Real.pi / 8)) ^ 4) = 3 / 2 :=
by sorry

end trig_identity_proof_l399_399131


namespace determine_n_for_11111_base_n_is_perfect_square_l399_399715

theorem determine_n_for_11111_base_n_is_perfect_square:
  ∃ m : ℤ, m^2 = 3^4 + 3^3 + 3^2 + 3 + 1 :=
by
  sorry

end determine_n_for_11111_base_n_is_perfect_square_l399_399715


namespace range_of_a_l399_399904

variables (a x : ℝ)

def p := |4*x - 3| ≤ 1
def q := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

def negation_p := ¬ p
def negation_q := ¬ q

theorem range_of_a :
  (∃ a : ℝ, (negation_p → negation_q) ∧ (negation_p ∧ ¬ negation_q) ↔ a ∈ Iio 0 ∪ Ioi 0) :=
sorry

end range_of_a_l399_399904


namespace complementary_angles_decrease_86_percent_l399_399194

theorem complementary_angles_decrease_86_percent (x : ℝ) (h : 10 * x = 90) :
  let small_angle := 3 * x
  let increased_small_angle := small_angle * 1.2
  let large_angle := 7 * x
  let new_large_angle := 90 - increased_small_angle
  (new_large_angle / large_angle) * 100 = 91.4 :=
by
  sorry

end complementary_angles_decrease_86_percent_l399_399194


namespace difference_of_numbers_l399_399604

theorem difference_of_numbers (a b : ℕ) (h1 : a = 2 * b) (h2 : (a + 4) / (b + 4) = 5 / 7) : a - b = 8 := 
by
  sorry

end difference_of_numbers_l399_399604


namespace bill_and_ted_white_spotted_mushrooms_l399_399695

def total_white_spotted_mushrooms
  (red_mushrooms_bill : ℕ)
  (brown_mushrooms_bill : ℕ)
  (green_mushrooms_ted : ℕ)
  (blue_mushrooms_ted : ℕ)
  (white_spotted_from_blue : ℕ → ℕ)
  (white_spotted_from_red : ℕ → ℕ)
  (white_spotted_from_brown : ℕ → ℕ) : ℕ :=
  white_spotted_from_blue blue_mushrooms_ted +
  white_spotted_from_red red_mushrooms_bill +
  white_spotted_from_brown brown_mushrooms_bill

theorem bill_and_ted_white_spotted_mushrooms :
  total_white_spotted_mushrooms 12 6 14 6 (λ b, b / 2) (λ r, (2 * r) / 3) (λ br, br) = 17 :=
by sorry

end bill_and_ted_white_spotted_mushrooms_l399_399695


namespace express_vector_c_as_linear_combination_l399_399782

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b : ℝ × ℝ := (1, -1)
noncomputable def c : ℝ × ℝ := (2, 3)

theorem express_vector_c_as_linear_combination :
  ∃ x y : ℝ, c = (x * (1, 1).1 + y * (1, -1).1, x * (1, 1).2 + y * (1, -1).2) ∧
             x = 5 / 2 ∧ y = -1 / 2 :=
by
  sorry

end express_vector_c_as_linear_combination_l399_399782


namespace total_surface_area_is_1426_l399_399925

-- Defining the volumes of the cubes
def volumes : List ℕ := [1, 8, 27, 64, 125, 216, 343, 512, 729]

-- Calculate the side length of a cube given its volume
def side_length (v : ℕ) : ℕ := Nat.root v 3

-- Calculate the surface area of a cube given its volume
def surface_area (v : ℕ) : ℕ := 6 * (side_length v) ^ 2

-- Calculate the total exposed surface area of the tower
def total_exposed_surface_area (volumes : List ℕ) : ℕ := 
  let surface_areas := volumes.map surface_area
  let exposed_surface_areas := List.zipWith (+) (surface_areas.tail.map (λ sa, sa - (side_length (surface_areas.head!)) ^ 2)) [surface_area (volumes.head!)]
  exposed_surface_areas.foldr (· + ·) 0

-- The theorem that proves the total exposed surface area is 1426
theorem total_surface_area_is_1426 : total_exposed_surface_area volumes = 1426 := 
  by
  -- omitted detailed proofs
  sorry

end total_surface_area_is_1426_l399_399925


namespace quadratic_intersects_x_axis_length_AB_for_positive_integer_sum_of_lengths_l399_399795

noncomputable def func (a : ℝ) (x : ℝ) : ℝ :=
  a * (a + 1) * x^2 - (2 * a + 1) * x + 1

theorem quadratic_intersects_x_axis {a : ℝ} :
  (∀ x : ℝ, func a x = 0 → 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ func a x1 = 0 ∧ func a x2 = 0)
  ↔ (a ≠ 0 ∧ a ≠ -1) :=
sorry

theorem length_AB_for_positive_integer {a : ℕ} (h : a > 0) :
  (∀ x1 x2 : ℝ, func a x1 = 0 ∧ func a x2 = 0 → 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |x1 - x2| = 1 / (a * (a + 1))) :=
sorry

theorem sum_of_lengths (n : ℕ) (h : n = 2010) :
  ∑ i in finset.range n, (1 : ℝ) / (i + 1) / (i + 2) = 2010 / 2011 :=
sorry

end quadratic_intersects_x_axis_length_AB_for_positive_integer_sum_of_lengths_l399_399795


namespace confidence_level_l399_399860

-- Define the conditions
def chi_squared_value : ℝ := 4.073
def prob_GTE_3_841 : ℝ := 0.05
def prob_GTE_5_024 : ℝ := 0.025

theorem confidence_level :
  (chi_squared_value = 4.073) →
  (prob_GTE_3_841 ≈ 0.05) →
  (prob_GTE_5_024 ≈ 0.025) →
  ∃ confidence : ℝ, confidence = 0.95 :=
by
  intros h1 h2 h3
  use 0.95
  sorry

end confidence_level_l399_399860


namespace ratios_of_square_areas_l399_399155

variable (x : ℝ)

def square_area (side_length : ℝ) : ℝ := side_length^2

theorem ratios_of_square_areas (hA : square_area x = x^2)
                               (hB : square_area (5 * x) = 25 * x^2)
                               (hC : square_area (2 * x) = 4 * x^2) :
  (square_area x / square_area (5 * x) = 1 / 25 ∧
   square_area (2 * x) / square_area (5 * x) = 4 / 25) := 
by {
  sorry
}

end ratios_of_square_areas_l399_399155


namespace time_for_double_ladies_to_complete_half_work_l399_399154

variable {n : ℕ} {S : ℝ}

-- Condition 1: Some ladies can do a piece of work in 12 days.
-- Condition 2: Each lady has a different work speed, with the fastest being 20% faster than the slowest.
def slowest_speed := S
def fastest_speed := 1.20 * S
def average_speed := (S + 1.20 * S) / 2

-- Defining the work done in 12 days by one lady
def work := 12 * average_speed

-- Condition 3: Twice the number of ladies work on half of the work together.
def half_work := work / 2
def total_speed_of_2n_ladies := 2 * n * average_speed

-- Question: How long will it take twice the number of ladies to complete half the work?
theorem time_for_double_ladies_to_complete_half_work (h : 0 < n) : 
  (half_work / total_speed_of_2n_ladies) = 3 := by
sorry

end time_for_double_ladies_to_complete_half_work_l399_399154


namespace church_full_capacity_l399_399596

theorem church_full_capacity :
  let first_section := 15 * 8 in
  let second_section := 20 * 6 in
  let third_section := 25 * 10 in
  first_section + second_section + third_section = 490 := 
by
  -- Definitions from the conditions are used here
  let first_section := 15 * 8
  let second_section := 20 * 6
  let third_section := 25 * 10
  have : first_section = 120 := rfl
  have : second_section = 120 := rfl
  have : third_section = 250 := rfl
  have sum := first_section + second_section + third_section
  show sum = 490
  calc
    sum = first_section + second_section + third_section : by rfl
    ... = 120 + 120 + 250 : by congr;
    ... = 490 : rfl

end church_full_capacity_l399_399596


namespace solution_set_f_x_leq_x_range_of_a_l399_399532

-- Definition of the function f
def f (x : ℝ) : ℝ := |2 * x - 7| + 1

-- Proof Problem for Question (1):
-- Given: f(x) = |2x - 7| + 1
-- Prove: The solution set of the inequality f(x) <= x is {x | 8/3 <= x <= 6}
theorem solution_set_f_x_leq_x :
  { x : ℝ | f x ≤ x } = { x : ℝ | 8 / 3 ≤ x ∧ x ≤ 6 } :=
sorry

-- Definition of the function g
def g (x : ℝ) : ℝ := f x - 2 * |x - 1|

-- Proof Problem for Question (2):
-- Given: f(x) = |2x - 7| + 1 and g(x) = f(x) - 2 * |x - 1|
-- Prove: If ∃ x, g(x) <= a, then a >= -4
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, g x ≤ a) → a ≥ -4 :=
sorry

end solution_set_f_x_leq_x_range_of_a_l399_399532


namespace tabitha_final_amount_l399_399158

def initial_amount : ℝ := 25
def amount_given_to_mom : ℝ := 8
def items_bought : ℕ := 5
def cost_per_item : ℝ := 0.5

theorem tabitha_final_amount :
  let remaining_after_mom := initial_amount - amount_given_to_mom in
  let remaining_after_investment := remaining_after_mom / 2 in
  let spent_on_items := items_bought * cost_per_item in
  let final_amount := remaining_after_investment - spent_on_items in
  final_amount = 6 := by
  sorry

end tabitha_final_amount_l399_399158


namespace emily_final_lives_l399_399373

/-- Initial number of lives Emily had. --/
def initialLives : ℕ := 42

/-- Number of lives Emily lost in the hard part of the game. --/
def livesLost : ℕ := 25

/-- Number of lives Emily gained in the next level. --/
def livesGained : ℕ := 24

/-- Final number of lives Emily should have after the changes. --/
def finalLives : ℕ := (initialLives - livesLost) + livesGained

theorem emily_final_lives : finalLives = 41 := by
  /-
  Proof is omitted as per instructions.
  Prove that the final number of lives Emily has is 41.
  -/
  sorry

end emily_final_lives_l399_399373


namespace find_k_l399_399493

variables (m n k : ℤ)  -- Declaring m, n, k as integer variables.

theorem find_k (h1 : m = 2 * n + 5) (h2 : m + 2 = 2 * (n + k) + 5) : k = 1 :=
by
  sorry

end find_k_l399_399493


namespace circle_through_ABC_l399_399447

-- Define points A, B, and C
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (1, 4)

-- Define the circle equation components to be proved
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 3*y - 3 = 0

-- The theorem statement that we need to prove
theorem circle_through_ABC : 
  ∃ (D E F : ℝ), (∀ x y, (x, y) = A ∨ (x, y) = B ∨ (x, y) = C → x^2 + y^2 + D*x + E*y + F = 0) 
  → circle_eqn x y :=
sorry

end circle_through_ABC_l399_399447


namespace monotonicity_and_range_of_a_l399_399076

noncomputable def f (a x : ℝ) : ℝ :=
  a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, x ∈ (set.Ioo 0 (1 / a)) → deriv (λ x, f a x) x < 0) ∧
  (∀ x, x ∈ (set.Ioi (1 / a)) → deriv (λ x, f a x) x > 0) ∧
  (∀ a, a > Real.exp (-1) → ∃ x : ℝ, ∀ x, f a x > 0) :=
by
  sorry

end monotonicity_and_range_of_a_l399_399076


namespace monotonicity_and_range_l399_399083

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end monotonicity_and_range_l399_399083


namespace housewife_more_oil_l399_399679

-- Definitions for the conditions
def original_price (P : ℝ) : Prop := P - 0.30 * P = 48
def reduced_price : ℝ := 48
def spending : ℝ := 800

-- Function to calculate the oil quantity for a given price
def quantity (price : ℝ) (amount : ℝ) : ℝ := amount / price

-- Theorem statement
theorem housewife_more_oil (P : ℝ) (hP : original_price P) :
  let X := quantity P spending in
  let Y := quantity reduced_price spending in
  Y - X = 5 := by
sorry

end housewife_more_oil_l399_399679


namespace subset_count_l399_399142

theorem subset_count (S : Finset ℕ) (hS : S = {x | 1 ≤ x ∧ x ≤ 11}.toFinset) :
  {T | T ⊆ S ∧ T.card = 5 ∧ ∀ (x ∈ T) (y ∈ T), x + y ≠ 12}.toFinset.card = 112 :=
sorry

end subset_count_l399_399142


namespace vector_magnitude_proof_l399_399405

noncomputable def vector_a : ℝ × ℝ × ℝ := (1, 1, 0)
noncomputable def vector_b : ℝ × ℝ × ℝ := (-1, 0, 2)

-- Function to calculate the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- The theorem statement
theorem vector_magnitude_proof : magnitude (2 • vector_a - vector_b) = real.sqrt 17 := by
  sorry

end vector_magnitude_proof_l399_399405


namespace distance_between_vertices_l399_399739

theorem distance_between_vertices (a b : ℝ) (a_pos : a = real.sqrt 144) (h : ∀ x y, x^2 / 144 - y^2 / 49 = 1): (2 * a) = 24 := by
  have ha : a = 12 := by sorry
  have h2a : 2 * a = 24 := by linarith
  exact h2a

end distance_between_vertices_l399_399739


namespace cot6_add_cot2_eq_3_l399_399458

theorem cot6_add_cot2_eq_3
  (x : ℝ)
  (h_geom_seq : ∃ r : ℝ, cos x = r * sin x ∧ sin x = r * tan x) :
  cot^6 x + cot^2 x = 3 :=
by
  -- sorry to skip the proof
  sorry

end cot6_add_cot2_eq_3_l399_399458


namespace p_necessary_not_sufficient_for_q_l399_399808

-- Lean statement
theorem p_necessary_not_sufficient_for_q (a b : ℝ) :
  (a + b = 0) ↔ (a^2 + b^2 = 0) → (a + b = 0) ∧ ¬((a + b = 0) → (a^2 + b^2 = 0)) :=
by
  intro h
  have h1 : a^2 + b^2 = 0 ↔ a = 0 ∧ b = 0,
  {
    -- since squares of real numbers are non-negative
    sorry
  }
  split
  { intro h2,
    rw h1 at h2,
    cases h2,
    exact add_self_eq_zero.2 h2_left,
    -- necessary condition
    exact ⟨h2_left, h2_right⟩
  }
  {
    intro h3,
    -- case where p is not sufficient
    intros h4,
    use (1, -1),
    simp,
    exact h3,
    sorry
  }
  exact ⟨sorry, sorry⟩

end p_necessary_not_sufficient_for_q_l399_399808


namespace functional_eq_solution_l399_399895

def R_star := { x : ℝ // x ≠ 1 }

noncomputable def f : R_star → ℝ := sorry

theorem functional_eq_solution (x : R_star) :
  let y := (x + 2009) / (x - 1)
  in x + f x + 2 * f y = 2010 → f x = (1/3 : ℝ) * (x + 2010 - 2 * y) := sorry

end functional_eq_solution_l399_399895


namespace mothers_day_discount_l399_399043

noncomputable def cost_after_discounts (initial_price : ℝ) (mother_discount : ℝ) (children_discount : ℝ) (vip_discount : ℝ) : ℝ :=
let price_after_mother = initial_price * (1 - mother_discount)
let price_after_children = price_after_mother * (1 - children_discount)
price_after_children * (1 - vip_discount)

theorem mothers_day_discount :
  ∀ (shoes : ℝ) (handbag : ℝ) (scarf : ℝ) (mother_discount : ℝ) (children_discount : ℝ) (vip_discount : ℝ),
    let total_price := shoes + handbag + scarf in
    shoes = 125 ∧ handbag = 75 ∧ scarf = 45 ∧ mother_discount = 0.10 ∧ children_discount = 0.04 ∧ vip_discount = 0.05 →
    total_price * (1 - mother_discount) * (1 - children_discount) * (1 - vip_discount) = 201.10 :=
by
  intros shoes handbag scarf mother_discount children_discount vip_discount
  intro h
  cases h
  unfold cost_after_discounts
  sorry

end mothers_day_discount_l399_399043


namespace solve_cosine_equation_l399_399147

noncomputable def solution_set : set ℝ :=
  { x | (x = π/2 ∨ x = π/6 ∨ x = 7*π/6 ∨ x = 5*π/6 ∨ x = 11*π/6 ∨
          x = π/4 ∨ x = 5*π/4 ∨ x = 3*π/4 ∨ x = 7*π/4) }

theorem solve_cosine_equation :
  { x : ℝ | 0 ≤ x ∧ x < 2 * π ∧ cos x ^ 2 + cos (2 * x) ^ 2 + cos (3 * x) ^ 2 = 1 } = solution_set :=
by
  -- proof goes here
  sorry

end solve_cosine_equation_l399_399147


namespace ball_total_distance_l399_399659

noncomputable def height (t : ℝ) : ℝ := 10 * t - 5 * t^2

theorem ball_total_distance :
  (2 * (10)) = 10 :=
by
  /* Proof goes here */
  sorry

end ball_total_distance_l399_399659


namespace monotonicity_and_range_l399_399087

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end monotonicity_and_range_l399_399087


namespace project_completed_in_20_days_l399_399265

noncomputable def project_completion_days (a b : ℝ) (t : ℝ) : Prop :=
(a ≠ 0) ∧ (b ≠ 0) ∧ ((t - 10) * (1/a) + t * (1/b) = 1)

theorem project_completed_in_20_days : 
  ∀ (a b t : ℝ), project_completion_days 20 40 t → t = 20 :=
begin
  intros a b t h,
  obtain ⟨ha, hb, h_eqn⟩ := h,
  sorry -- proof omitted
end

end project_completed_in_20_days_l399_399265


namespace degree_P3_sub_Q3_ge_2n_l399_399910

theorem degree_P3_sub_Q3_ge_2n (n : ℕ) (P Q : polynomial ℝ) (hPn : P.degree = n) (hQn : Q.degree = n) (hPQ : P ≠ Q) :
  (P^3 - Q^3).degree ≥ 2 * n := 
begin
  sorry
end

end degree_P3_sub_Q3_ge_2n_l399_399910


namespace nonexist_a_b_l399_399058

noncomputable def set_A (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ n : ℤ, p = (n : ℝ, a * n + b)}

noncomputable def set_B : Set (ℝ × ℝ) :=
  {p | ∃ m : ℤ, p = (m : ℝ, 3 * m^2 + 15)}

noncomputable def set_C : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 ≤ 144}

theorem nonexist_a_b (a b : ℝ) :
  (∃ p, p ∈ set_A a b ∧ p ∈ set_B) ∧ (a, b) ∈ set_C → False :=
sorry

end nonexist_a_b_l399_399058


namespace tabitha_final_amount_is_six_l399_399166

def initial_amount : ℕ := 25
def amount_given_to_mom : ℕ := 8
def num_items_bought : ℕ := 5
def cost_per_item : ℝ := 0.5

def amount_after_giving_mom : ℝ := initial_amount - amount_given_to_mom
def amount_invested : ℝ := amount_after_giving_mom / 2
def amount_after_investment : ℝ := amount_after_giving_mom - amount_invested
def total_cost_of_items : ℝ := num_items_bought * cost_per_item
def final_amount : ℝ := amount_after_investment - total_cost_of_items

theorem tabitha_final_amount_is_six :
  final_amount = 6 := 
by 
  -- sorry to skip the proof
  sorry

end tabitha_final_amount_is_six_l399_399166


namespace females_on_police_force_l399_399640

theorem females_on_police_force (H : ∀ (total_female_officers total_officers_on_duty female_officers_on_duty : ℕ), 
  total_officers_on_duty = 500 ∧ female_officers_on_duty = total_officers_on_duty / 2 ∧ female_officers_on_duty = total_female_officers / 4) :
  ∃ total_female_officers : ℕ, total_female_officers = 1000 := 
by {
  sorry
}

end females_on_police_force_l399_399640


namespace monotonic_intervals_minimum_value_l399_399281

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Math.exp(-x) * (a * x^2 + a + 1)

theorem monotonic_intervals (a : ℝ) :
  if a = 0 then ∀ x, f a x < f a (x + 1)
  else if a > 0 then ∀ x, f a x < f a (x + 1)
  else ∃ r1 r2, r1 < r2 ∧
        (∀ x, x < r1 → f a x < f a (x + 1)) ∧
        (∀ x, r1 < x ∧ x < r2 → f a x > f a (x + 1)) ∧
        (∀ x, x > r2 → f a x < f a (x + 1)) := sorry

theorem minimum_value (a : ℝ) (h : -1 < a ∧ a < 0) :
  ∀ x ∈ Icc 1 2, f a x ≥ f a 2 := sorry

end monotonic_intervals_minimum_value_l399_399281


namespace sum_of_coefficients_l399_399897

/-- Given a polynomial Q(z) = x^3 + px^2 + qx + r with real coefficients,
and complex roots v + 2i, v + 6i, and 3v - 5, prove that p + q + r = -115/16. -/
theorem sum_of_coefficients (p q r : ℝ) (v : ℂ) (h : ∃ (v : ℂ), 
  Q(z) = x^3 + px^2 + qx + r ∧
  (v + 2 * I) * (v + 6 * I) * (3 * v - 5) = 0 ∧ is_real p ∧ is_real q ∧ is_real r) : 
  p + q + r = - 115 / 16 :=
by
  sorry

end sum_of_coefficients_l399_399897


namespace julian_pennies_l399_399891

theorem julian_pennies (p : ℕ) : 
  4 * 10 + 10 * 0.05 + p * 0.01 = 45.50 → p = 500 := 
by
  sorry

end julian_pennies_l399_399891


namespace distance_from_origin_to_point_l399_399873

def point : ℝ × ℝ := (12, -16)
def origin : ℝ × ℝ := (0, 0)

theorem distance_from_origin_to_point : 
  let (x1, y1) := origin
  let (x2, y2) := point 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 20 :=
by
  sorry

end distance_from_origin_to_point_l399_399873


namespace distance_between_hyperbola_vertices_l399_399757

theorem distance_between_hyperbola_vertices :
  (∃ a : ℝ, a = real.sqrt 144 ∧ ∀ d : ℝ, d = 2 * a → d = 24) :=
begin
  use real.sqrt 144,
  split,
  { refl },
  { intros d hd,
    rw hd,
    refl }
end

end distance_between_hyperbola_vertices_l399_399757


namespace integer_solution_is_three_l399_399380

theorem integer_solution_is_three (n : ℤ) : (∃ k : ℤ, (n^3 - 3*n^2 + n + 2) = 5^k) ↔ n = 3 := 
by
  sorry

end integer_solution_is_three_l399_399380


namespace find_omega_and_phi_l399_399823

open Real

/-- Definition of the function f -/
def f (ω x φ : ℝ) : ℝ := 2 * sin (ω * x + φ)

/-- Proof problem statement given the conditions -/
theorem find_omega_and_phi :
  ∃ (ω φ : ℝ), 0 < ω ∧ 0 < φ ∧ φ < π ∧ (∀ x : ℝ, f ω x φ = f ω (-x) φ) ∧
  (∃ x1 x2 : ℝ, 
    2 = 2 * sin (ω * x1 + φ) ∧ 
    2 = 2 * sin (ω * x2 + φ) ∧ 
    abs (x1 - x2) = π) ∧ 
    ω = 2 ∧ φ = π / 2 :=
begin
  -- Proof steps will be added here
  sorry
end

end find_omega_and_phi_l399_399823


namespace tan_of_A_l399_399037

theorem tan_of_A (A B C : Type) [IsTriangle A B C] (hC : angle C = 90) (hAB : distance A B = 13) (hBC : distance BC = 5) : 
  tangent A = 5 / 12 :=
by
  sorry

end tan_of_A_l399_399037


namespace quadrilateral_perimeter_div_a_l399_399224

-- Given conditions
def square_vertices (a : ℝ) : set (ℝ × ℝ) :=
  {(-a, -a), (a, -a), (-a, a), (a, a)}

def cutting_line (x : ℝ) : ℝ := (2/3) * x

-- The transformed proof problem
theorem quadrilateral_perimeter_div_a (a : ℝ) (h : a ≠ 0) :
  let p1 := (a, (2/3)*a),
      p2 := (-a, -(2/3)*a),
      vertical_len := (a + (2/3)*a),
      horizontal_len := 2*a,
      diagonal_len := Real.sqrt (4*a^2 + (16/9)*a^2)
  in (2 * vertical_len + horizontal_len + diagonal_len) / a = (16 + 2*Real.sqrt 13) / 3 :=
by 
  sorry

end quadrilateral_perimeter_div_a_l399_399224


namespace enclosed_area_correct_l399_399951

noncomputable def enclosed_area : ℝ :=
  ∫ x in (1/2)..2, (-x + 5/2 - 1/x)

theorem enclosed_area_correct :
  enclosed_area = (15/8) - 2 * Real.log 2 :=
by
  sorry

end enclosed_area_correct_l399_399951


namespace quadratic_factors_l399_399228

-- Define the quadratic polynomial
def quadratic (b c x : ℝ) : ℝ := x^2 + b * x + c

-- Define the roots
def root1 : ℝ := -2
def root2 : ℝ := 3

-- Theorem: If the quadratic equation has roots -2 and 3, then it factors as (x + 2)(x - 3)
theorem quadratic_factors (b c : ℝ) (h1 : quadratic b c root1 = 0) (h2 : quadratic b c root2 = 0) :
  ∀ x : ℝ, quadratic b c x = (x + 2) * (x - 3) :=
by
  sorry

end quadratic_factors_l399_399228


namespace frank_sales_quota_l399_399403

theorem frank_sales_quota (x : ℕ) :
  (3 * x + 12 + 23 = 50) → x = 5 :=
by sorry

end frank_sales_quota_l399_399403


namespace part_a_part_b_l399_399516

variable (p : ℕ → ℕ)
axiom primes_sequence : ∀ n, (∀ m < p n, m ∣ p n → m = 1 ∨ m = p n) ∧ p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 5 ∧ p 4 = 7 ∧ p 5 = 11

theorem part_a (n : ℕ) (h : n ≥ 5) : p n > 2 * n := 
  by sorry

theorem part_b (n : ℕ) : p n > 3 * n ↔ n ≥ 12 := 
  by sorry

end part_a_part_b_l399_399516


namespace diamond_expression_l399_399360

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Declare the main theorem
theorem diamond_expression :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29 / 132 := 
by
  sorry

end diamond_expression_l399_399360


namespace gcd_eight_digit_repeating_four_digit_l399_399690

theorem gcd_eight_digit_repeating_four_digit :
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) →
  Nat.gcd (10001 * n) (10001 * m) = 10001) :=
by
  intros n hn m hm
  sorry

end gcd_eight_digit_repeating_four_digit_l399_399690


namespace solve_system_of_equations_l399_399152

theorem solve_system_of_equations (a b c x y z : ℝ)
  (h1 : a^3 + a^2 * x + a * y + z = 0)
  (h2 : b^3 + b^2 * x + b * y + z = 0)
  (h3 : c^3 + c^2 * x + c * y + z = 0) :
  x = -(a + b + c) ∧ y = ab + ac + bc ∧ z = -abc :=
by {
  sorry
}

end solve_system_of_equations_l399_399152


namespace distance_between_adjacent_parallel_lines_l399_399599

theorem distance_between_adjacent_parallel_lines (r d : ℝ) 
  (h1 : 16000 + 10 * d^2 = 40 * r^2)
  (h2 : 11664 + 81 * d^2 = 36 * r^2) :
  d = sqrt(4336 / 71) := 
sorry

end distance_between_adjacent_parallel_lines_l399_399599


namespace area_of_union_of_scaled_triangle_l399_399495

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_union_of_scaled_triangle :
  ∀ (A B C : ℝ), A = 25 → B = 39 → C = 42 →
  let ABC_area := heron_area A B C
  let A'B'C'_area := (1 / 4) * ABC_area
  ABC_area = Real.sqrt 293192 :=
by
  intros A B C hA hB hC
  have h_ABC_area : heron_area A B C = Real.sqrt 293192 := sorry
  have h_scaled := (1 / 4) * heron_area A B C = (1 / 4) * Real.sqrt 293192 := sorry
  exact h_ABC_area

end area_of_union_of_scaled_triangle_l399_399495


namespace negation_of_universal_proposition_l399_399581

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - 3 * x + 2 > 0)) ↔ (∃ x : ℝ, x^2 - 3 * x + 2 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l399_399581


namespace f_neg1_f_3_l399_399916

def f (x : ℝ) : ℝ :=
  if x < 2 then 2 * x - 4 else 10 - 3 * x

theorem f_neg1 : f (-1) = -6 := by
  sorry

theorem f_3 : f 3 = 1 := by
  sorry

end f_neg1_f_3_l399_399916


namespace exists_same_color_points_l399_399182

-- Define the color type
inductive Color
| red
| blue

-- Define the problem statement
theorem exists_same_color_points (x : ℝ) (hx : x > 0) :
  ∃ (c : Color) (p₁ p₂ : ℝ × ℝ), (p₁ ≠ p₂) ∧ (p₁.dist p₂ = x) ∧ (p₁.color = c) ∧ (p₂.color = c) := by
sorry

end exists_same_color_points_l399_399182


namespace monotonicity_no_zeros_range_of_a_l399_399060

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * (Real.log x) + 1

theorem monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1/a → ∀ t : ℝ, t ∈ Ioo x y → f' a t < 0 ) ∧ 
  (∀ x y : ℝ, 1/a < x ∧ x < y → ∀ t : ℝ, t ∈ Ioo x y → f' a t > 0 ) :=
sorry

theorem no_zeros_range_of_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → f a x > 0) : 
  a > 1/Real.exp 1 :=
sorry

end monotonicity_no_zeros_range_of_a_l399_399060


namespace selling_price_l399_399315

theorem selling_price (cost_price profit_percentage : ℝ) (h_cost : cost_price = 250) (h_profit : profit_percentage = 0.60) :
  cost_price + profit_percentage * cost_price = 400 := sorry

end selling_price_l399_399315


namespace fiftieth_term_arithmetic_seq_l399_399015

theorem fiftieth_term_arithmetic_seq : 
  (∀ (n : ℕ), (2 + (n - 1) * 5) = 247) := by
  sorry

end fiftieth_term_arithmetic_seq_l399_399015


namespace discount_problem_l399_399215

theorem discount_problem
  (orig_price : ℝ) 
  (second_discount : ℝ) 
  (final_price : ℝ) 
  (first_discount : ℝ) :
  orig_price = 400 ∧
  second_discount = 0.20 ∧ 
  final_price = 224 →
  first_discount = 30 := 
begin
  intros h,
  cases h with h_orig h_rest,
  cases h_rest with h_second h_final,
  have h_eq : (orig_price * (1 - first_discount / 100)) * (1 - second_discount) = final_price,
  { rw [h_orig, h_final, h_second],
    exact sorry },
  sorry
end

end discount_problem_l399_399215


namespace plane_divides_segment_AM_l399_399796

variables {M A B C D E F K X : Point}
variables [RegularHexagonalPyramid M A B C D E F]
variables (bisects : K ∈ Segment B M)
variables (divides : X ∈ Segment A M)
variables (onPlane : X ∈ Plane F E K)

theorem plane_divides_segment_AM (M A B C D E F K X : Point)
  [RegularHexagonalPyramid M A B C D E F]
  (bisects : K ∈ Segment B M)
  (divides : X ∈ Segment A M)
  (onPlane : X ∈ Plane F E K) :
  ratio (Segment M X) (Segment X A) = 2 := 
sorry

end plane_divides_segment_AM_l399_399796


namespace monotonicity_of_f_range_of_a_if_no_zeros_l399_399067

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end monotonicity_of_f_range_of_a_if_no_zeros_l399_399067


namespace find_t_and_m_l399_399828

theorem find_t_and_m 
  (t m : ℝ) 
  (ineq : ∀ x : ℝ, x^2 - 3 * x + t < 0 ↔ 1 < x ∧ x < m) : 
  t = 2 ∧ m = 2 :=
sorry

end find_t_and_m_l399_399828


namespace p_true_of_and_not_p_false_l399_399467

variable {p q : Prop}

theorem p_true_of_and_not_p_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : p :=
sorry

end p_true_of_and_not_p_false_l399_399467


namespace complementary_angle_decrease_l399_399213

theorem complementary_angle_decrease (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3 / 7) : 
  (∃ new_α : ℝ, new_α = α * 1.2) →
  ∃ new_β : ℝ, new_β = (1 - 0.0857) * β :=
by
  intro h3
  cases h3 with new_α h_newα
  use 90 - new_α
  sorry

end complementary_angle_decrease_l399_399213


namespace shaded_area_correct_l399_399303

noncomputable def hexagon_side_length : ℝ := 8
noncomputable def sector_radius : ℝ := 4
noncomputable def sector_angle_deg : ℝ := 60
noncomputable def hexagon_area : ℝ := 6 * (sqrt 3 / 4) * hexagon_side_length ^ 2
noncomputable def sector_area : ℝ := 6 * (sector_angle_deg / 360) * π * sector_radius ^ 2
noncomputable def shaded_area : ℝ := hexagon_area - sector_area

theorem shaded_area_correct :
  shaded_area = 96 * sqrt 3 - 16 * π :=
by
  sorry

end shaded_area_correct_l399_399303


namespace non_neg_int_solutions_eq_2014_number_of_solutions_eq_339024_l399_399583

theorem non_neg_int_solutions_eq_2014 : 
  ∃ (f : ℕ × ℕ × ℕ → ℕ), ∀ z y x, (x + 2 * y + 3 * z = 2014 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) ↔ (f (x, y, z) = 1) :=
begin
  -- Define f to be 1 if the equation is satisfied and the variables are non-negative
  let f := λ (p : ℕ × ℕ × ℕ), if (p.1 + 2 * p.2 + 3 * p.3 = 2014) then 1 else 0,
  -- State the main condition and equivalency
  existsi f,
  -- Introduce variables
  intros z y x,
  -- Express the condition and equivalency formally
  split,
  { intro h,
    cases h,
    rw [h_left],
    rw [if_pos h_left],
    exact rfl, },
  { intro h,
    rw [← h],
    split,
    { apply if_pos h, },
    exact ⟨‹x ≥ 0›, ⟨‹y ≥ 0›, ‹z ≥ 0›⟩⟩, },
  sorry,
end

theorem number_of_solutions_eq_339024 : count (λ (x y z : ℕ), x + 2 * y + 3 * z = 2014) (λ (x y z : ℕ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) = 339024 :=
by sorry

end non_neg_int_solutions_eq_2014_number_of_solutions_eq_339024_l399_399583


namespace eccentricity_of_hyperbola_l399_399827

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a_gt_0 : a > 0) (h_b_gt_0 : b > 0)
  (h_asymptotes : b / a = 3 / 4) : 
  let e := Real.sqrt (1 + (b / a) ^ 2)
  in e = 5 / 4 :=
by
  sorry

end eccentricity_of_hyperbola_l399_399827


namespace find_m_l399_399368

theorem find_m (m : ℝ) :
  (100 : ℝ) ^ m = 100 ^ (-3) * sqrt (100 ^ 55 / 0.0001) ↔ m = 25.5 :=
by
  sorry

end find_m_l399_399368


namespace right_triangle_hypotenuse_l399_399615

theorem right_triangle_hypotenuse (a b : ℕ) (h₁ : a = 75) (h₂ : b = 100) : ∃ c, c = 125 ∧ c^2 = a^2 + b^2 :=
by
  sorry

end right_triangle_hypotenuse_l399_399615


namespace compare_abs_k_abs_b_l399_399025

-- Given conditions translated into Lean definitions
variables (k b a : ℝ)
variable (linear_function : ∀ x : ℝ, x = k * x + b)
variable (y_intercept : b > 0)
variable (x_intercept : a > 1 ∧ a = -b / k)

-- Statement of the problem
theorem compare_abs_k_abs_b (h1 : linear_function) (h2 : y_intercept) (h3 : x_intercept) : |k| < |b| :=
sorry

end compare_abs_k_abs_b_l399_399025


namespace div_127m_by_2_pow_m1_sub_1_l399_399511

theorem div_127m_by_2_pow_m1_sub_1
  (p : ℕ) (hp_prime : Prime p) (hp_gt_7 : p > 7) (hp_mod_6 : p % 6 = 1) :
  let m := 2 ^ p - 1 in 
  127 * m ∣ 2 ^ (m - 1) - 1 :=
by 
  let m := 2 ^ p - 1
  exact sorry

end div_127m_by_2_pow_m1_sub_1_l399_399511


namespace sum_digits_least_time_l399_399997

/-- Given 12 horses each running a lap in the k-th prime minute,
  return the sum of the digits of the least time (in minutes) at which at least 5 horses meet 
  simultaneously at the starting point -/

theorem sum_digits_least_time (T : ℕ) (first_12_primes : ∀ k : ℕ, k < 12 → Nat.Prime (nth_prime k)) :
  (∀ i ∈ [2, 3, 5, 7, 11], i ∣ T) ∧ ∀ d, d < T → ((∀ i ∈ [2, 3, 5, 7, 11], ¬ (i ∣ d)))
  → (T.digits 10).sum = 6 := sorry

end sum_digits_least_time_l399_399997


namespace integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l399_399382

theorem integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5 :
  ∀ n : ℤ, ∃ k : ℕ, (n^3 - 3 * n^2 + n + 2 = 5^k) ↔ n = 3 :=
by
  intro n
  exists sorry
  sorry

end integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l399_399382


namespace quadratic_vertex_point_properties_l399_399577

theorem quadratic_vertex_point_properties
  (a b c : ℤ)
  (h_vertex : ∀ x, (y = a * x^2 + b * x + c) → (2, -3) ∈ vertex x)
  (h_pass : (3, 0) ∈ curve x)
  (y : ℤ)
  (vertex : ℤ × ℤ)
  (curve : ℤ → ℤ)
  : a - b + c = 24 :=
  sorry

end quadratic_vertex_point_properties_l399_399577


namespace arithmetic_mean_reciprocals_primes_l399_399333

theorem arithmetic_mean_reciprocals_primes : 
  let p := [2, 3, 5, 7] in 
  let reciprocals := p.map (λ n => 1 / (n : ℚ)) in
  (reciprocals.sum / reciprocals.length) = (247 / 840 : ℚ) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l399_399333


namespace red_paint_cans_l399_399127

/--
If the ratio of red paint to white paint is 4 to 3, and there are 35 cans of the mixture in total, then the number of cans of red paint needed is 20.
-/
theorem red_paint_cans (total_cans : ℕ) (red_to_white_ratio : ℕ) (red_ratio : ℕ) : 
  total_cans = 35 ∧ red_to_white_ratio = 3 ∧ red_ratio = 4 → 
  4 * total_cans / (4 + 3) = 20 :=
begin
  sorry
end

end red_paint_cans_l399_399127


namespace monotonicity_and_no_x_intercept_l399_399097

noncomputable theory

def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

def is_monotonic (a : ℝ) (x : ℝ) : Prop := 
  if x < 1 / a then 
    f a x > f a (x + ε) -- ε is a small positive value
  else if x > 1 / a then 
    f a x < f a (x - ε)
  else
    true -- At x = 1/a, the function transits from decreasing to increasing

theorem monotonicity_and_no_x_intercept 
  (a : ℝ) (h1 : 0 < a) : 
  (∀ x : ℝ, 0 < x → is_monotonic a x) ∧ 
  (∀ x : ℝ, f a x ≠ 0) ↔ 
  (a ∈ Ioi (1 / real.exp 1)) := 
sorry

end monotonicity_and_no_x_intercept_l399_399097


namespace distance_between_vertices_l399_399742

theorem distance_between_vertices (a b : ℝ) (a_pos : a = real.sqrt 144) (h : ∀ x y, x^2 / 144 - y^2 / 49 = 1): (2 * a) = 24 := by
  have ha : a = 12 := by sorry
  have h2a : 2 * a = 24 := by linarith
  exact h2a

end distance_between_vertices_l399_399742


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l399_399347

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  let primes := [2, 3, 5, 7] in
  let reciprocals := primes.map (λ p => (1 : ℚ) / p) in
  let sum_reciprocals := reciprocals.sum in
  let mean := sum_reciprocals / (primes.length : ℚ) in
  mean = 247 / 840 :=
by
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p => (1 : ℚ) / p)
  let sum_reciprocals := reciprocals.sum
  let mean := sum_reciprocals / (primes.length : ℚ)
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l399_399347


namespace integer_values_of_n_l399_399399

theorem integer_values_of_n : 
  {n : ℤ | ∃ (n : ℤ), 16200 * (3/4)^n ∈ ℤ}.finite ∧ {n : ℤ | ∃ (n : ℤ), 16200 * (3/4)^n ∈ ℤ}.card = 3 :=
by
  sorry

end integer_values_of_n_l399_399399


namespace tom_already_has_4_pounds_of_noodles_l399_399601

-- Define the conditions
def beef : ℕ := 10
def noodle_multiplier : ℕ := 2
def packages : ℕ := 8
def weight_per_package : ℕ := 2

-- Define the total noodles needed
def total_noodles_needed : ℕ := noodle_multiplier * beef

-- Define the total noodles bought
def total_noodles_bought : ℕ := packages * weight_per_package

-- Define the already owned noodles
def already_owned_noodles : ℕ := total_noodles_needed - total_noodles_bought

-- State the theorem to prove
theorem tom_already_has_4_pounds_of_noodles :
  already_owned_noodles = 4 :=
  sorry

end tom_already_has_4_pounds_of_noodles_l399_399601


namespace seventy_fifth_number_in_S_l399_399273

-- Definition of the set S
def S : Set ℕ := { n | ∃ k : ℕ, k ≥ 0 ∧ n = 8 * k + 5 }

-- Statement of the proof problem
theorem seventy_fifth_number_in_S : (finset.last (finset.range 75).attach + 1) * 8 + 5 = 597 := by
  sorry

end seventy_fifth_number_in_S_l399_399273


namespace lowry_earnings_l399_399537

-- Definitions of the prices of bonsai
def price_small := 30
def price_medium := 45
def price_large := 60
def price_xlarge := 85

-- Daily sales quantities
def sales_small := 8
def sales_medium := 11
def sales_large := 4
def sales_xlarge := 3

-- Function to calculate discount
def discount (original : ℕ) (percentage : ℕ) : ℕ :=
  original * percentage / 100

-- Conditions for discounts
def discount_small (amount : ℕ) : ℕ :=
  if amount >= 4 then discount (amount * price_small) 10 else 0

def discount_medium (amount : ℕ) : ℕ :=
  if amount >= 3 then discount (amount * price_medium) 15 else 0

def discount_large (amount : ℕ) : ℕ :=
  if amount > 5 then discount (amount * price_large) 5 else 0

def discount_xlarge (amount : ℕ) : ℕ :=
  if amount >= 2 then discount (amount * price_xlarge) 8 else 0

def additional_discount (total : ℕ) (small_count : ℕ) (medium_count : ℕ) : ℕ :=
  if small_count + medium_count >= 10 then discount total 3 else 0

-- Lean 4 theorem
theorem lowry_earnings :
  let earnings := (sales_small * price_small - discount_small sales_small) +
                  (sales_medium * price_medium - discount_medium sales_medium) +
                  (sales_large * price_large - discount_large sales_large) +
                  (sales_xlarge * price_xlarge - discount_xlarge sales_xlarge)
  in earnings - additional_discount earnings sales_small sales_medium = 1078.01 :=
  sorry

end lowry_earnings_l399_399537


namespace distance_between_vertices_l399_399750

def hyperbola_eq (x y : ℝ) : Prop := 
  x^2 / 144 - y^2 / 49 = 1

theorem distance_between_vertices : 2 * real.sqrt 144 = 24 :=
by {
    -- use sqrt calculation for clarity
    have h : real.sqrt 144 = 12, by {
        exact real.sqrt_eq_iff_sq_eq.mpr (or.inl (by norm_num)),
    },
    rw [h],
    norm_num
}

end distance_between_vertices_l399_399750


namespace monotonicity_no_zeros_range_of_a_l399_399064

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * (Real.log x) + 1

theorem monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1/a → ∀ t : ℝ, t ∈ Ioo x y → f' a t < 0 ) ∧ 
  (∀ x y : ℝ, 1/a < x ∧ x < y → ∀ t : ℝ, t ∈ Ioo x y → f' a t > 0 ) :=
sorry

theorem no_zeros_range_of_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → f a x > 0) : 
  a > 1/Real.exp 1 :=
sorry

end monotonicity_no_zeros_range_of_a_l399_399064


namespace projectile_height_30_in_2_seconds_l399_399961

theorem projectile_height_30_in_2_seconds (t y : ℝ) : 
  (y = -5 * t^2 + 25 * t ∧ y = 30) → t = 2 :=
by
  sorry

end projectile_height_30_in_2_seconds_l399_399961


namespace complementary_angle_decrease_l399_399212

theorem complementary_angle_decrease (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3 / 7) : 
  (∃ new_α : ℝ, new_α = α * 1.2) →
  ∃ new_β : ℝ, new_β = (1 - 0.0857) * β :=
by
  intro h3
  cases h3 with new_α h_newα
  use 90 - new_α
  sorry

end complementary_angle_decrease_l399_399212


namespace wilson_theorem_non_prime_divisibility_l399_399132

theorem wilson_theorem (p : ℕ) (h : Nat.Prime p) : p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

theorem non_prime_divisibility (p : ℕ) (h : ¬ Nat.Prime p) : ¬ p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

end wilson_theorem_non_prime_divisibility_l399_399132


namespace complementary_angles_decrease_86_percent_l399_399192

theorem complementary_angles_decrease_86_percent (x : ℝ) (h : 10 * x = 90) :
  let small_angle := 3 * x
  let increased_small_angle := small_angle * 1.2
  let large_angle := 7 * x
  let new_large_angle := 90 - increased_small_angle
  (new_large_angle / large_angle) * 100 = 91.4 :=
by
  sorry

end complementary_angles_decrease_86_percent_l399_399192


namespace min_value_f_sum_l399_399418

variables {x1 x2 : ℝ} (f : ℝ → ℝ)

-- Definitions based on given conditions
def f_def : Prop := ∀ x, f(x) + 1 = 4^x * (1 - f(x))

def valid_inputs : Prop := x1 > 0 ∧ x2 > 0

def sum_condition : Prop := f(x1) + f(x2) = 1

-- Theorem: Proving the minimum value of f(x1 + x2) is 4/5
theorem min_value_f_sum : valid_inputs → f_def f → sum_condition f → f(x1 + x2) = 4 / 5 :=
sorry

end min_value_f_sum_l399_399418


namespace incorrect_statement_A_l399_399630

theorem incorrect_statement_A (a b : ℝ) : a^2 > b^2 → a * b > 0 → ¬ (1 / a < 1 / b) :=
by
  intro h1 h2
  -- specific counterexample where statement A fails
  let a := -4
  let b := -2
  have h1 : a^2 > b^2 := by sorry -- show that (-4)^2 > (-2)^2
  have h2 : a * b > 0 := by sorry -- show that (-4) * (-2) > 0
  show ¬ (1 / a < 1 / b) by sorry -- show that 1 / (-4) < 1 / (-2) is false

end incorrect_statement_A_l399_399630


namespace ratio_of_magnets_given_away_l399_399311

-- Define the conditions of the problem
def initial_magnets_adam : ℕ := 18
def magnets_peter : ℕ := 24
def final_magnets_adam (x : ℕ) : ℕ := initial_magnets_adam - x
def half_magnets_peter : ℕ := magnets_peter / 2

-- The main statement to prove
theorem ratio_of_magnets_given_away (x : ℕ) (h : final_magnets_adam x = half_magnets_peter) :
    (x : ℚ) / initial_magnets_adam = 1 / 3 :=
by
  sorry -- This is where the proof would go

end ratio_of_magnets_given_away_l399_399311


namespace total_students_left_l399_399987

def initial_boys : Nat := 14
def initial_girls : Nat := 10
def boys_dropout : Nat := 4
def girls_dropout : Nat := 3

def boys_left : Nat := initial_boys - boys_dropout
def girls_left : Nat := initial_girls - girls_dropout

theorem total_students_left : boys_left + girls_left = 17 :=
by 
  sorry

end total_students_left_l399_399987


namespace find_x_l399_399386

theorem find_x (x : ℝ) : log 32 (x + 32) = 7 / 5 → x = 96 := by
  intro h
  -- proof steps go here
  sorry

end find_x_l399_399386


namespace monotonicity_and_no_real_roots_l399_399090

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

theorem monotonicity_and_no_real_roots 
  (a : ℝ) (ha : 0 < a) : 
  (∀ x : ℝ, (0 < x ∧ x < (1 / a) → deriv (f a) x < 0) ∧ (x > (1 / a) → deriv (f a) x > 0)) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≠ 0 → a > (1 / real.exp 1))) :=
begin
  sorry
end

end monotonicity_and_no_real_roots_l399_399090


namespace percentage_decrease_of_larger_angle_l399_399186

noncomputable def complementary_angles_decrease_percentage : Real :=
let total_degrees := 90
let ratio_sum := 3 + 7
let part := total_degrees / ratio_sum
let smaller_angle := 3 * part
let larger_angle := 7 * part
let increased_smaller_angle := smaller_angle * 1.2
let new_larger_angle := total_degrees - increased_smaller_angle
let decrease_amount := larger_angle - new_larger_angle
(decrease_amount / larger_angle) * 100

theorem percentage_decrease_of_larger_angle
  (smaller_increased_percentage : Real := 20)
  (ratio_three : Real := 3)
  (ratio_seven : Real := 7)
  (total_degrees : Real := 90)
  (expected_decrease : Real := 8.57):
  complementary_angles_decrease_percentage = expected_decrease := 
sorry

end percentage_decrease_of_larger_angle_l399_399186


namespace constant_term_in_expansion_l399_399784

noncomputable def integral_a : ℝ :=
  ∫ x in (-1 : ℝ)..(1 : ℝ), Real.sqrt (1 - x^2)

theorem constant_term_in_expansion :
  integral_a = (π / 2) →
  let expr := (integral_a + 2 - (π / 2)) * (x : ℝ) - (1 / x) in
  let expansion := expr^6 in
  (∃ c, c = -160 ∧ expansion.eval x = c) :=
by
  sorry

end constant_term_in_expansion_l399_399784


namespace Mandy_gym_workout_time_0_l399_399921

variable (exercise yoga bike gym : ℕ)
variable (ratio_yoga_exercise : yoga = 2 * (exercise / 3))
variable (exercise_total : exercise = bike + gym)
variable (bike_time : bike = 12)

theorem Mandy_gym_workout_time_0 (h1 : ratio_yoga_exercise) (h2 : exercise_total) (h3 : bike_time) :
  gym = 0 :=
sorry

end Mandy_gym_workout_time_0_l399_399921


namespace f_odd_and_increasing_l399_399900

def f (x : ℝ) : ℝ := x^3 - (1 / x^3)

theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = - f x) ∧ 
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y) :=
by 
  sorry

end f_odd_and_increasing_l399_399900


namespace distance_between_hyperbola_vertices_l399_399758

theorem distance_between_hyperbola_vertices :
  (∃ a : ℝ, a = real.sqrt 144 ∧ ∀ d : ℝ, d = 2 * a → d = 24) :=
begin
  use real.sqrt 144,
  split,
  { refl },
  { intros d hd,
    rw hd,
    refl }
end

end distance_between_hyperbola_vertices_l399_399758


namespace problem1_solution_set_problem2_range_a_l399_399822

theorem problem1_solution_set (x : ℝ) : 
  let f := λ x => |x^2 - 2 * x + 2| - 15
  in f x ≥ -10 ↔ x ≥ 3 ∨ x ≤ -1 :=
sorry

theorem problem2_range_a (a : ℝ) :
  (∀ x, |x^2 - 2 * x + a - 1| - a^2 - 2 * a ≥ 0) ↔ -2 ≤ a ∧ a ≤ 0 :=
sorry

end problem1_solution_set_problem2_range_a_l399_399822


namespace distance_on_third_day_is_36_difference_between_longest_and_shortest_is_57_average_daily_distance_is_50_l399_399260

-- Definitions for each day's recorded distance deviation
def day_1_distance := -8
def day_2_distance := -11
def day_3_distance := -14
def day_4_distance := 0
def day_5_distance := 8
def day_6_distance := 41
def day_7_distance := -16

-- Parameters and conditions
def actual_distance (recorded: Int) : Int := 50 + recorded

noncomputable def distance_3rd_day : Int := actual_distance day_3_distance
noncomputable def longest_distance : Int :=
    max (max (max (day_1_distance) (day_2_distance)) (max (day_3_distance) (day_4_distance)))
        (max (max (day_5_distance) (day_6_distance)) (day_7_distance))
noncomputable def shortest_distance : Int :=
    min (min (min (day_1_distance) (day_2_distance)) (min (day_3_distance) (day_4_distance)))
        (min (min (day_5_distance) (day_6_distance)) (day_7_distance))
noncomputable def average_distance : Int :=
    50 + (day_1_distance + day_2_distance + day_3_distance + day_4_distance +
          day_5_distance + day_6_distance + day_7_distance) / 7

-- Theorems to prove each part of the problem
theorem distance_on_third_day_is_36 : distance_3rd_day = 36 := by
  sorry

theorem difference_between_longest_and_shortest_is_57 : 
  (actual_distance longest_distance - actual_distance shortest_distance) = 57 := by
  sorry

theorem average_daily_distance_is_50 : average_distance = 50 := by
  sorry

end distance_on_third_day_is_36_difference_between_longest_and_shortest_is_57_average_daily_distance_is_50_l399_399260


namespace rounding_to_hundredth_l399_399605

theorem rounding_to_hundredth {x : ℝ} (h : x = 2.0359) : (Real.toHundredth x) = 2.04 :=
by
  rw h
  sorry

end rounding_to_hundredth_l399_399605


namespace sum_of_integer_solutions_l399_399249

theorem sum_of_integer_solutions
  (h1 : ∀ n : ℤ, |n-5| < 15)
  (h2 : ∀ n : ℤ, |n| < |n-5|) :
  ∑ n in (-10 : ℤ) .. 2, n = -44 :=
by
  sorry

end sum_of_integer_solutions_l399_399249


namespace molecular_weight_correct_l399_399617

-- Atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 15.999
def atomic_weight_H : ℝ := 1.008

-- Number of each type of atom in the compound
def num_Al : ℕ := 1
def num_O : ℕ := 3
def num_H : ℕ := 3

-- Molecular weight calculation
def molecular_weight : ℝ :=
  (num_Al * atomic_weight_Al) + (num_O * atomic_weight_O) + (num_H * atomic_weight_H)

theorem molecular_weight_correct : molecular_weight = 78.001 := by
  sorry

end molecular_weight_correct_l399_399617


namespace subset_of_possible_values_l399_399834

theorem subset_of_possible_values (a : ℝ) :
  (∀ x, x ∈ {x | ax = 1} → x ∈ {x | x^2 = 1}) →
  (a = 0 ∨ a = 1 ∨ a = -1) :=
begin
  sorry
end

end subset_of_possible_values_l399_399834


namespace jack_started_diet_approx_26_months_ago_l399_399885

noncomputable def weight_loss_per_month : ℝ := (222 - 180) / (45 + (198 - 180) / ((222 - 198) / 45))

theorem jack_started_diet_approx_26_months_ago :
  (222 - 198) / weight_loss_per_month ≈ 26 :=
by
  have h1 : 222 - 198 = 24 := by norm_num
  have h2 : 198 - 180 = 18 := by norm_num
  have h3 : 24 + 18 = 42 := by norm_num
  have h4 : weight_loss_per_month = 42 / 45 := by
    simp [weight_loss_per_month, h3]
  have h5 : (24 : ℝ) / (42 / 45) ≈ 26 := by
    rw [h4]
    norm_num
  exact h5

end jack_started_diet_approx_26_months_ago_l399_399885


namespace min_distance_AB_l399_399882

-- Definitions of the polar coordinates equations for the curves
def C1_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin (θ + Real.pi / 3)
def C2_polar (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) = 4

-- Definitions of the rectangular coordinates equations for the curves
def C1_rect (x y : ℝ) : Prop := x^2 + y^2 - Real.sqrt 3 * x - y = 0
def C2_rect (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 8 = 0

-- Conversion of polar to rectangular coordinates
axiom polar_to_rect_C1 (ρ θ x y : ℝ) (h: C1_polar ρ θ) : 
  C1_rect (ρ * Real.cos θ) (ρ * Real.sin θ) 

axiom polar_to_rect_C1 (ρ θ x y : ℝ) (h: C1_polar ρ θ) : 
  (ρ * Real.sin θ = y) -> (ρ * Real.cos θ = x) -> C1_rect x y

axiom polar_to_rect_C2 (ρ θ x y : ℝ) (h: C2_polar ρ θ) : 
  2 * (ρ * Real.sin θ/2 ) + Real.sqrt 3 * (ρ * Real.cos θ/2) = 4

axiom polar_to_rect_C2 (ρ θ x y : ℝ) (h: C2_polar ρ θ) : 
  (ρ * Real.sin θ = y) -> (ρ * Real.cos θ = x) -> C2_rect x y

-- Minimum value of |AB|
theorem min_distance_AB (x1 y1 x2 y2 : ℝ) (hx1y1 : C1_rect x1 y1) (hx2y2 : C2_rect x2 y2):
  |((x1 - x2)^2 + (y1 - y2)^2 - 4)| ^(1/2) = 2 :=
sorry

end min_distance_AB_l399_399882


namespace monotonicity_and_range_of_a_l399_399075

noncomputable def f (a x : ℝ) : ℝ :=
  a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, x ∈ (set.Ioo 0 (1 / a)) → deriv (λ x, f a x) x < 0) ∧
  (∀ x, x ∈ (set.Ioi (1 / a)) → deriv (λ x, f a x) x > 0) ∧
  (∀ a, a > Real.exp (-1) → ∃ x : ℝ, ∀ x, f a x > 0) :=
by
  sorry

end monotonicity_and_range_of_a_l399_399075


namespace correct_triangle_l399_399628

-- Define the conditions for the sides of each option
def sides_A := (1, 2, 3)
def sides_B := (3, 4, 5)
def sides_C := (3, 1, 1)
def sides_D := (3, 4, 7)

-- Conditions for forming a triangle
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Prove the problem statement
theorem correct_triangle : is_triangle 3 4 5 :=
by
  sorry

end correct_triangle_l399_399628


namespace arithmetic_mean_reciprocals_first_four_primes_l399_399338

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l399_399338


namespace find_a_l399_399824

-- Define the function f and assume the conditions
def f (x : ℝ) (a : ℝ) : ℝ := 2^x - a * 2^(-x)

theorem find_a (a : ℝ) : (∀ x : ℝ, f (x+a) = - f (-x + a)) → a = 1 :=
by 
  sorry

end find_a_l399_399824


namespace quadrilateral_area_l399_399023

noncomputable def sin_120_eq_sqrt3_div_2 : Real.sin (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
Real.sin_eq_sqrt3_div_2 120 sorry

theorem quadrilateral_area (AB BC CD BD : ℝ)
  (angle_B angle_C : ℝ)
  (h1 : AB = 5)
  (h2 : BC = 6)
  (h3 : CD = 7)
  (h4 : BD = 8)
  (h5 : angle_B = 120 * Real.pi / 180)
  (h6 : angle_C = 120 * Real.pi / 180) :
  let Area_ABC := 1/2 * AB * BD * Real.sin angle_B
  let Area_BCD := 1/2 * BC * CD * Real.sin angle_C
  let total_area := Area_ABC + Area_BCD
in total_area = 20.5 * Real.sqrt 3 :=
by
  sorry

end quadrilateral_area_l399_399023


namespace abc_is_square_l399_399417

variables (a b c : ℕ)

def odd (n : ℕ) : Prop := n % 2 = 1
def coprime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem abc_is_square (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
                      (h_a_odd : odd a) (h_b_gt_c : b > c) (h_coprime_abc : coprime a b ∧ coprime b c ∧ coprime a c)
                      (h_eq : a * (b - c) = 2 * b * c) : ∃ n : ℕ, n^2 = a * b * c :=
begin
  sorry
end

end abc_is_square_l399_399417


namespace social_survey_arrangements_l399_399988

theorem social_survey_arrangements :
  let total_ways := Nat.choose 9 3
  let all_male_ways := Nat.choose 5 3
  let all_female_ways := Nat.choose 4 3
  total_ways - all_male_ways - all_female_ways = 70 :=
by
  sorry

end social_survey_arrangements_l399_399988


namespace inequality_proof_l399_399790

variable (a b c d e p q : ℝ)

theorem inequality_proof
  (h₀ : 0 < p)
  (h₁ : p ≤ a) (h₂ : a ≤ q)
  (h₃ : p ≤ b) (h₄ : b ≤ q)
  (h₅ : p ≤ c) (h₆ : c ≤ q)
  (h₇ : p ≤ d) (h₈ : d ≤ q)
  (h₉ : p ≤ e) (h₁₀ : e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 := 
by
  sorry -- The actual proof will be filled here

end inequality_proof_l399_399790


namespace smallest_n_satisfying_inequality_l399_399775

theorem smallest_n_satisfying_inequality :
  ∃ n : ℕ, 0 < n ∧ (n = 8001) ∧ √(5 * n) - √(5 * n - 4) < 0.01 := by
  sorry

end smallest_n_satisfying_inequality_l399_399775


namespace radius_of_second_smallest_circle_l399_399488

def geometric_sequence := ∃ (a b c d : ℝ), 
    a = 5 ∧ d = 20 ∧ 
    (∃ r, r = (d / a)^(1/3) ∧ b = a * r)

theorem radius_of_second_smallest_circle : ∃ b, b = 10 := 
by 
    have h : geometric_sequence := sorry
    rcases h with ⟨a, b, c, d, ha, hd, hr⟩
    use b
    rw hr.2
    sorry

end radius_of_second_smallest_circle_l399_399488


namespace find_k_l399_399660

def green_balls : ℕ := 7

noncomputable def probability_green (k : ℕ) : ℚ := green_balls / (green_balls + k)
noncomputable def probability_purple (k : ℕ) : ℚ := k / (green_balls + k)

noncomputable def winning_for_green : ℤ := 3
noncomputable def losing_for_purple : ℤ := -1

noncomputable def expected_value (k : ℕ) : ℚ :=
  (probability_green k) * (winning_for_green : ℚ) + (probability_purple k) * (losing_for_purple : ℚ)

theorem find_k (k : ℕ) (h : expected_value k = 1) : k = 7 :=
  sorry

end find_k_l399_399660


namespace cos_eq_solution_is_valid_l399_399149

noncomputable def solve_cos_eq (x : ℝ) : Prop :=
  ∃ (n : ℤ), x = 90 + n * 180 ∨
             x = 30 + n * 180 ∨
             x = 210 + n * 180 ∨
             x = 150 + n * 180 ∨
             x = 330 + n * 180 ∨
             x = 45 + n * 180 ∨
             x = 225 + n * 180 ∨
             x = 135 + n * 180 ∨
             x = 315 + n * 180

theorem cos_eq_solution_is_valid (x : ℝ) (h : 0 ≤ x ∧ x < 360) :
  (cos x) ^ 2 + (cos (2 * x)) ^ 2 + (cos (3 * x)) ^ 2 = 1 ↔ solve_cos_eq x :=
sorry

end cos_eq_solution_is_valid_l399_399149


namespace monotonicity_of_f_range_of_a_if_no_zeros_l399_399068

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end monotonicity_of_f_range_of_a_if_no_zeros_l399_399068


namespace A_share_correct_l399_399636

-- Define the contributions of A and B
def A_investment : ℕ := 400
def B_investment : ℕ := 200
def months_A : ℕ := 12
def months_B : ℕ := 6
def total_profit : ℕ := 100

-- Define the calculation of investment-months
def A_investment_months := A_investment * months_A
def B_investment_months := B_investment * months_B
def total_investment_months := A_investment_months + B_investment_months
def A_proportion := A_investment_months.to_rat / total_investment_months.to_rat
def A_share := (total_profit.to_rat * A_proportion)

-- State the theorem
theorem A_share_correct : A_share = 80 :=
by
  sorry

end A_share_correct_l399_399636


namespace third_dimension_of_box_l399_399678

-- Define the given conditions
def volume_of_cube (V: ℕ) : Prop := V = 27
def number_of_cubes (N : ℕ) : Prop := N = 24
def box_dimensions (l w: ℕ) : Prop := l = 8 ∧ w = 9
def total_box_volume (V : ℕ) [hcube: ∀ V, volume_of_cube V] [hn : ∀ N, number_of_cubes N] : ℕ := hn 24 * hcube 27

-- Define the proof problem
theorem third_dimension_of_box
 (l w h V : ℕ)
 [cube : ∀ V, volume_of_cube V]
 [numcubes : ∀ N, number_of_cubes N]
 [dims : box_dimensions l w] :
  l * w * h = total_box_volume (numcubes 24 * cube 27) → h = 9 :=
by
  sorry

end third_dimension_of_box_l399_399678


namespace tan_cot_sum_l399_399850

theorem tan_cot_sum (α β : ℝ) 
  (h : (sin α)^4 / (sin β)^2 + (cos α)^4 / (cos β)^2 = 1) : 
  (tan β)^4 / (tan α)^2 + (cot β)^4 / (cot α)^2 = 2 := 
by 
  sorry

end tan_cot_sum_l399_399850


namespace find_solutions_l399_399111

noncomputable def smallestPrimeFactor (n : ℕ) : ℕ :=
  if h : 0 < n then classical.some (Nat.exists_prime_and_dvd (classical.some_spec (Nat.exists_infinite_primes h)))
  else 0

noncomputable def largestPrimeFactor (n : ℕ) : ℕ :=
  if h : 0 < n then classical.some (Nat.exists_prime_and_dvd (classical.some_spec (Nat.exists_greatest_prime h)))
  else 0

theorem find_solutions (n p q: ℕ) (hp: Nat.Prime p) (hq: Nat.Prime q) (hsp: smallestPrimeFactor n = p)
  (hlp: largestPrimeFactor n = q) (h : p^2 + q^2 = n + 9):
  n = 9 ∨ n = 20 :=
by
  sorry

end find_solutions_l399_399111


namespace P_X_Y_collinear_l399_399041

open Real

noncomputable def collinear (A B C : Point) : Prop :=
∃ (l : Line), A ∈ l ∧ B ∈ l ∧ C ∈ l

variables (A B C D P Q X Y : Point)

axiom quadrilateral : poly Quadrilateral A B C D
axiom P_inside_ABCD : inside poly Quadrilateral P A B C D
axiom AP_intersect_CP : ∃ Q, line_through A P ∩ line_through C P = {Q}
axiom PQ_ext_ang_bisectors : external_angle_bisector (angle B P A) (line_through P Q) ∧ external_angle_bisector (angle D P C) (line_through P Q)
axiom X_angle_bisector_BPQ : angle_bisector (angle B P Q) X
axiom Y_angle_bisector_DPQ : angle_bisector (angle D P Q) Y

theorem P_X_Y_collinear : collinear P X Y := by
  sorry

end P_X_Y_collinear_l399_399041


namespace triangle_isosceles_parallel_height_l399_399874

theorem triangle_isosceles_parallel_height (A B C D E : Point) (h : ℝ) (AB AC BC CD CE : ℝ)
  (hA : is_equilateral_triangle A B C AB AC)
  (hD : is_on_extrapolation_segment D B A beyond_A)
  (hE : is_on_segment E B C)
  (h_parallel : parallel CD AE)
  (h_height : height_from_A A B C h) :
  CD = (2 * h * CE) / BC := 
sorry

end triangle_isosceles_parallel_height_l399_399874


namespace magnitude_of_vector_l399_399408

def a : ℝ × ℝ × ℝ := (1, 1, 0)
def b : ℝ × ℝ × ℝ := (-1, 0, 2)

theorem magnitude_of_vector :
  ‖(2 • a.1 - b.1, 2 • a.2 - b.2, 2 • a.3 - b.3)‖ = Real.sqrt 17 := by
  sorry

end magnitude_of_vector_l399_399408


namespace unique_intersection_condition_l399_399973

theorem unique_intersection_condition
  (a c b d : ℝ)
  (h_sym_central : ∀ x y : ℝ, (y = 2a + 1/(x-b) ∧ y = 2c + 1/(x-d) → (x, y) = (1/2*(b+d), a+c)))
  (h_common_point : 2a + 1/(1/2*(b+d) - b) = 2c + 1/(1/2*(b+d) - d) ∧ 2a + 1/(1/2*(b+d) - b) = a + c) :
  (a - c) * (b - d) = 2 :=
by
  -- Proof goes here
  sorry

end unique_intersection_condition_l399_399973


namespace sum_le_product_l399_399053

variable {n : ℕ} (a b : Fin n → ℝ)
variable (h1 : ∀ i, 0 < a i) (h2 : ∀ i, 0 < b i)
variable (h3 : (∑ i, a i) = (∏ i, a i))
variable (h4 : ∀ i, a i ≤ b i)

theorem sum_le_product (h1 : ∀ i, 0 < a i) 
  (h2 : ∀ i, 0 < b i) (h3 : (∑ i, a i) = (∏ i, a i)) 
  (h4 : ∀ i, a i ≤ b i) : 
  (∑ i, b i) ≤ (∏ i, b i) :=
by
  sorry

end sum_le_product_l399_399053


namespace remainder_of_130_div_k_l399_399400

theorem remainder_of_130_div_k (k a : ℕ) (hk : 90 = a * k^2 + 18) : 130 % k = 4 :=
sorry

end remainder_of_130_div_k_l399_399400


namespace quadrilateral_midpoints_intersect_at_one_point_l399_399133

open EuclideanGeometry -- Open the Euclidean geometry section of Mathlib

noncomputable def quadrilateral (A B C D : Point) : Prop :=
  true -- Define a quadrilateral with four points

noncomputable def midpoint (A B : Point) : Point :=
  Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2) -- Define the midpoint of two points

noncomputable def segment (A B : Point) : Line := 
  Line.through A B -- Define a segment as a line through two points

theorem quadrilateral_midpoints_intersect_at_one_point
  (A B C D : Point)
  (H_quad : quadrilateral A B C D)
  (P : Point := midpoint A B)
  (H : Point := midpoint C D)
  (T : Point := midpoint B C)
  (E : Point := midpoint A D)
  (K : Point := midpoint B D)
  (F : Point := midpoint A C) :
  ∃ M : Point, (Line.through P H).contains M ∧ (Line.through T E).contains M ∧ (Line.through K F).contains M ∧
               M = midpoint P H ∧ M = midpoint T E ∧ M = midpoint K F :=
begin
  sorry
end

end quadrilateral_midpoints_intersect_at_one_point_l399_399133


namespace necklace_profit_l399_399233

theorem necklace_profit :
  let cost_per_charm_A := 10
  let cost_per_charm_B := 18
  let charms_A := 8
  let charms_B := 12
  let selling_price_A := 125
  let selling_price_B := 280
  let num_A := 45
  let num_B := 35 in
  let profit_A := selling_price_A - (charms_A * cost_per_charm_A)
  let profit_B := selling_price_B - (charms_B * cost_per_charm_B) in
  profit_A = 45 ∧
  profit_B = 64 ∧
  (num_A * profit_A + num_B * profit_B) = 4265 :=
by
  let cost_per_charm_A := 10
  let cost_per_charm_B := 18
  let charms_A := 8
  let charms_B := 12
  let selling_price_A := 125
  let selling_price_B := 280
  let num_A := 45
  let num_B := 35
  let profit_A := selling_price_A - (charms_A * cost_per_charm_A)
  let profit_B := selling_price_B - (charms_B * cost_per_charm_B)
  have h1: profit_A = 45, from sorry
  have h2: profit_B = 64, from sorry
  have h3: (num_A * profit_A + num_B * profit_B) = 4265, from sorry
  exact ⟨h1, h2, h3⟩

end necklace_profit_l399_399233


namespace complementary_angles_decrease_86_percent_l399_399193

theorem complementary_angles_decrease_86_percent (x : ℝ) (h : 10 * x = 90) :
  let small_angle := 3 * x
  let increased_small_angle := small_angle * 1.2
  let large_angle := 7 * x
  let new_large_angle := 90 - increased_small_angle
  (new_large_angle / large_angle) * 100 = 91.4 :=
by
  sorry

end complementary_angles_decrease_86_percent_l399_399193


namespace calculate_p2_plus_s_neg2_l399_399902

noncomputable def g : ℚ[X] := 3 * X ^ 4 + 9 * X ^ 3 - 7 * X ^ 2 + 2 * X + 4
noncomputable def h : ℚ[X] := X ^ 2 + 2 * X - 3
noncomputable def p := 3 * X ^ 2 + 3
noncomputable def s := 9 * X - 20

theorem calculate_p2_plus_s_neg2 :
  g = p * h + s ∧ degree s < degree h → eval 2 p + eval (-2) s = -23 :=
by
  -- Definitions and assumptions here
  sorry

end calculate_p2_plus_s_neg2_l399_399902


namespace arithmetic_mean_reciprocal_primes_l399_399342

theorem arithmetic_mean_reciprocal_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := (1 : ℚ) / p1
  let r2 := (1 : ℚ) / p2
  let r3 := (1 : ℚ) / p3
  let r4 := (1 : ℚ) / p4
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 := by
sorry

end arithmetic_mean_reciprocal_primes_l399_399342


namespace mass_of_fourth_metal_l399_399293

theorem mass_of_fourth_metal 
  (m1 m2 m3 m4 : ℝ)
  (total_mass : m1 + m2 + m3 + m4 = 20)
  (h1 : m1 = 1.5 * m2)
  (h2 : m2 = 3/4 * m3)
  (h3 : m3 = 5/6 * m4) :
  m4 = 20 * (48 / 163) :=
sorry

end mass_of_fourth_metal_l399_399293


namespace complex_conjugate_point_is_correct_l399_399434

def complex_conjugate_point (z : ℂ) : ℂ := complex.conj z

noncomputable def z : ℂ := (1 + 2 * complex.I) / ((1 - complex.I) ^ 2)

theorem complex_conjugate_point_is_correct :
  complex_conjugate_point z = -1 - (1/2) * complex.I :=
by
  unfold complex_conjugate_point
  sorry

end complex_conjugate_point_is_correct_l399_399434


namespace fraction_eval_l399_399353

theorem fraction_eval : 
  ((12^4 + 324) * (24^4 + 324) * (36^4 + 324)) / 
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324)) = (84 / 35) :=
by
  sorry

end fraction_eval_l399_399353


namespace solve_system_of_equations_l399_399153

theorem solve_system_of_equations (x y z : ℝ) (φ : ℝ) (k : ℤ) 
    (h1 : 2 * x + x^2 * y = y)
    (h2 : 2 * y + y^2 * z = z)
    (h3 : 2 * z + z^2 * x = x) 
    (hφ : φ = k * real.pi / 7) :
    (x = real.tan φ ∧ 
     y = real.tan (2 * φ) ∧ 
     z = real.tan (4 * φ)) ∨
    (x = real.tan (k * real.pi / 7) ∧ 
     y = real.tan (2 * k * real.pi / 7) ∧ 
     z = real.tan (4 * k * real.pi / 7)) :=
sorry

end solve_system_of_equations_l399_399153


namespace find_center_of_circle_l399_399736

noncomputable def center_of_circle (x y : ℝ) : Prop :=
  x^2 - 8 * x + y^2 + 4 * y = 16

theorem find_center_of_circle (x y : ℝ) (h : center_of_circle x y) : (x, y) = (4, -2) :=
by 
  sorry

end find_center_of_circle_l399_399736


namespace range_of_m_l399_399419

-- Definition of propositions p and q
def p (m : ℝ) : Prop := (2 * m - 3)^2 - 4 > 0
def q (m : ℝ) : Prop := m > 2

-- The main theorem stating the range of values for m
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ (m < 1/2 ∨ (2 < m ∧ m ≤ 5/2)) :=
by
  sorry

end range_of_m_l399_399419


namespace symmetry_one_common_point_l399_399969

theorem symmetry_one_common_point
  (a b c d : ℝ)
  (f g : ℝ → ℝ)
  (hx : ∀ x, f x = 2 * a + 1 / (x - b))
  (hy : ∀ x, g x = 2 * c + 1 / (x - d))
  (sym_point : ℝ × ℝ := (b + d) / 2, a + c)
  (symmetric : ∀ x, f (2 * sym_point.1 - x) = 2 * sym_point.2 - f x ∧ g (2 * sym_point.1 - x) = 2 * sym_point.2 - g x)
  (common_point : ∃ x, f x = g x)
  : (a - c) * (b - d) = 2 :=
begin
  sorry
end

end symmetry_one_common_point_l399_399969


namespace game_returns_to_A_after_three_rolls_l399_399944

theorem game_returns_to_A_after_three_rolls :
  (∃ i j k : ℕ, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ 1 ≤ k ∧ k ≤ 6 ∧ (i + j + k) % 12 = 0) → 
  true :=
by
  sorry

end game_returns_to_A_after_three_rolls_l399_399944


namespace meditation_hours_per_week_l399_399886

-- Define the conditions
def meditation_per_day : Nat := 30 * 2 / 60  -- 30 minutes twice a day, converted to hours
def days_per_week : Nat := 7

-- Define the main statement to be proven
theorem meditation_hours_per_week (h : meditation_per_day = 1) : days_per_week * meditation_per_day = 7 :=
by
  have h1 : days_per_week = 7 := rfl
  rw [h1, h]
  exact rfl

end meditation_hours_per_week_l399_399886


namespace line_through_circumcenter_l399_399482

variables {A B C M N O P : Type} [triangle : is_acute_triangle ABC] 
variables [height : is_height A BC] [midM : is_midpoint M A B] [midN : is_midpoint N A C]
variables [symmetric : is_symmetric P A MN] [circumcenter : is_circumcenter O ABC]

theorem line_through_circumcenter : contains_circumcenter MN O :=
sorry

end line_through_circumcenter_l399_399482


namespace inequality_of_sums_l399_399919

variable {a b c : ℝ}

theorem inequality_of_sums
  (positives : a > 0 ∧ b > 0 ∧ c > 0)
  (sum_eq_one : a + b + c = 1) :
  (sqrt (ab / (c + ab)) + sqrt (bc / (a + bc)) + sqrt (ac / (b + ac)) ≤ 3 / 2) := by
  sorry

end inequality_of_sums_l399_399919


namespace figures_and_largest_pieces_l399_399548

noncomputable def figures_assembled : ℕ := 8
noncomputable def largest_figure_pieces : ℕ := 16

theorem figures_and_largest_pieces (total_pieces : ℕ) 
  (pieces_smallest_figures : ℕ)
  (pieces_largest_figures : ℕ)
  (pieces_used_each_figure_different : ∀ i j, i ≠ j → a_i ≠ a_j) :
  total_pieces = 80 → pieces_smallest_figures = 14 → pieces_largest_figures = 43 →
  sum_of_middle_figures = 23 →
  figures_assembled = 8 ∧ largest_figure_pieces = 16 :=
begin
  sorry
end

end figures_and_largest_pieces_l399_399548


namespace sixty_term_sequence_l399_399218

def contains_two (n : ℕ) : Prop :=
  n.digits 10.some (λ d, d = 2)

def filtered_sequence : List ℕ :=
  List.filter (λ x, contains_two x) 
    (List.range (4 * (4 * 60)))  -- Large enough range to include enough terms

def sequence_60th_term : ℕ :=
List.get 59 $ List.filter (λ x, contains_two x ∧ x % 4 = 0) 
         (List.range (4 * 61))

theorem sixty_term_sequence : sequence_60th_term = 204 :=
sorry

end sixty_term_sequence_l399_399218


namespace sum_first_100_terms_l399_399217

noncomputable def sequence (a : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then a 
  else if (sequence a (n - 1)).even then (sequence a (n - 1)) / 2 
  else 3 * (sequence a (n - 1)) + 1

theorem sum_first_100_terms : 
  let a1 := 34 in
  let sum := (Finset.range 100).sum (λ n => sequence a1 (n + 1)) in
  sum = 450 :=
begin
  sorry
end

end sum_first_100_terms_l399_399217


namespace stella_profit_l399_399565

-- Definitions based on the conditions
def number_of_dolls := 6
def price_per_doll := 8
def number_of_clocks := 4
def price_per_clock := 25
def number_of_glasses := 8
def price_per_glass := 6
def number_of_vases := 3
def price_per_vase := 12
def number_of_postcards := 10
def price_per_postcard := 3
def cost_of_merchandise := 250

-- Calculations based on given problem and solution
def revenue_from_dolls := number_of_dolls * price_per_doll
def revenue_from_clocks := number_of_clocks * price_per_clock
def revenue_from_glasses := number_of_glasses * price_per_glass
def revenue_from_vases := number_of_vases * price_per_vase
def revenue_from_postcards := number_of_postcards * price_per_postcard
def total_revenue := revenue_from_dolls + revenue_from_clocks + revenue_from_glasses + revenue_from_vases + revenue_from_postcards
def profit := total_revenue - cost_of_merchandise

-- Main theorem statement
theorem stella_profit : profit = 12 := by
  sorry

end stella_profit_l399_399565


namespace original_apples_l399_399635

theorem original_apples (s : Nat) (h1 : s = 500) (h2 : 0.80 * s = 0.20 * (x : Nat)) : x = 2500 := by
  sorry

end original_apples_l399_399635


namespace sqrt_equation_y_value_l399_399566

theorem sqrt_equation_y_value (y : ℝ) : 
  (sqrt (2 + sqrt (3 * y - 4)) = sqrt 9) → y = 53 / 3 :=
by
  sorry

end sqrt_equation_y_value_l399_399566


namespace monotonicity_no_zeros_range_of_a_l399_399062

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * (Real.log x) + 1

theorem monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1/a → ∀ t : ℝ, t ∈ Ioo x y → f' a t < 0 ) ∧ 
  (∀ x y : ℝ, 1/a < x ∧ x < y → ∀ t : ℝ, t ∈ Ioo x y → f' a t > 0 ) :=
sorry

theorem no_zeros_range_of_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → f a x > 0) : 
  a > 1/Real.exp 1 :=
sorry

end monotonicity_no_zeros_range_of_a_l399_399062


namespace tank_capacity_l399_399685

theorem tank_capacity (C : ℕ) 
  (h : 0.9 * (C : ℝ) - 0.4 * (C : ℝ) = 63) : C = 126 := 
by
  sorry

end tank_capacity_l399_399685


namespace jessica_repay_l399_399507

theorem jessica_repay (P : ℝ) (r : ℝ) (n : ℝ) (x : ℕ)
  (hx : P = 20)
  (hr : r = 0.12)
  (hn : n = 3 * P) :
  x = 17 :=
sorry

end jessica_repay_l399_399507


namespace nine_tuples_satisfy_condition_l399_399896

noncomputable def num_satisfying_tuples : ℕ :=
  1 + (24 * 9 * 8) + (24 * Nat.choose 9 2 * Nat.choose 8 2) + 1

theorem nine_tuples_satisfy_condition :
  ∀ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ),
  (∀ (i j k : Fin 9),
    ∃ (l : Fin 9),
    l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ i < j ∧ j < k ∧ a_1 + a_2 + a_3 + a_4 = 100) →
  num_satisfying_tuples = 1 + (24 * 9 * 8) + (24 * Nat.choose 9 2 * Nat.choose 8 2) + 1 :=
sorry

end nine_tuples_satisfy_condition_l399_399896


namespace max_sum_of_squares_condition_l399_399917

noncomputable def maximum_sum_of_squares (a : ℕ → ℝ) :=
  a 1^2 + a 2^2 + ∑ i in finset.range (101).filter (λ i, 3 ≤ i), a i^2

theorem max_sum_of_squares_condition {a : ℕ → ℝ} (h1 : ∀ i, 1 ≤ i → i ≤ 100 → 0 ≤ a i)
  (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ 100 → a i ≥ a j)
  (h3 : a 1 + a 2 ≤ 100)
  (h4 : ∑ i in finset.range (101).filter (λ i, 3 ≤ i), a i ≤ 100) :
  maximum_sum_of_squares a ≤ 10000 :=
begin
  sorry
end

end max_sum_of_squares_condition_l399_399917


namespace fraction_of_friends_l399_399480

variable (x y : ℕ) -- number of first-grade students and sixth-grade students

-- Conditions from the problem
def condition1 : Prop := ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a * x = b * y ∧ 1 / 3 = a / (a + b)
def condition2 : Prop := ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c * y = d * x ∧ 2 / 5 = c / (c + d)

-- Theorem statement to prove that the fraction of students who are friends is 4/11
theorem fraction_of_friends (h1 : condition1 x y) (h2 : condition2 x y) :
  (1 / 3 : ℚ) * y + (2 / 5 : ℚ) * x / (x + y) = 4 / 11 :=
sorry

end fraction_of_friends_l399_399480


namespace total_cost_of_books_l399_399847

-- Conditions from the problem
def C1 : ℝ := 350
def loss_percent : ℝ := 0.15
def gain_percent : ℝ := 0.19
def SP1 : ℝ := C1 - (loss_percent * C1) -- Selling price of the book sold at a loss
def SP2 : ℝ := SP1 -- Selling price of the book sold at a gain

-- Statement to prove the total cost
theorem total_cost_of_books : C1 + (SP2 / (1 + gain_percent)) = 600 := by
  sorry

end total_cost_of_books_l399_399847


namespace divide_milk_in_half_l399_399266

theorem divide_milk_in_half (bucket : ℕ) (a : ℕ) (b : ℕ) (a_liters : a = 5) (b_liters : b = 7) (bucket_liters : bucket = 12) :
  ∃ x y : ℕ, x = 6 ∧ y = 6 ∧ x + y = bucket := by
  sorry

end divide_milk_in_half_l399_399266


namespace sum_of_permutations_of_1234567_l399_399394

theorem sum_of_permutations_of_1234567 : 
  let factorial_7 := 5040
  let sum_of_digits := 1 + 2 + 3 + 4 + 5 + 6 + 7
  let geometric_series_sum := (10 ^ 7 - 1) / (10 - 1)
  sum_of_digits * factorial_7 * geometric_series_sum = 22399997760 :=
by
  let factorial_7 := 5040
  let sum_of_digits := 1 + 2 + 3 + 4 + 5 + 6 + 7
  let geometric_series_sum := (10^7 - 1) / (10 - 1)
  sorry

end sum_of_permutations_of_1234567_l399_399394


namespace pages_allocation_correct_l399_399471

-- Define times per page for Alice, Bob, and Chandra
def t_A := 40
def t_B := 60
def t_C := 48

-- Define pages read by Alice, Bob, and Chandra
def pages_A := 295
def pages_B := 197
def pages_C := 420

-- Total pages in the novel
def total_pages := 912

-- Calculate the total time each one spends reading
def total_time_A := t_A * pages_A
def total_time_B := t_B * pages_B
def total_time_C := t_C * pages_C

-- Theorem: Prove the correct allocation of pages
theorem pages_allocation_correct : 
  total_pages = pages_A + pages_B + pages_C ∧
  total_time_A = total_time_B ∧
  total_time_B = total_time_C :=
by 
  -- Place end of proof here 
  sorry

end pages_allocation_correct_l399_399471


namespace henri_movie_length_l399_399844

theorem henri_movie_length (x : ℝ) (h1 : 8 - (x + 1.5) = 3) : x = 3.5 :=
by
  have reading_time_in_minutes := 1800 / 10
  have reading_time_in_hours := reading_time_in_minutes / 60
  have h2 : reading_time_in_hours = 3 := by sorry
  have total_hours_spent := x + 1.5 + reading_time_in_hours
  have h3 : total_hours_spent = 8 := by sorry
  show x = 3.5 from calc
    x + 4.5 = 8 : by sorry
    x = 3.5 : by sorry

end henri_movie_length_l399_399844


namespace range_of_m_l399_399468

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) ↔ -5 ≤ m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l399_399468


namespace john_spends_on_paintballs_l399_399045

theorem john_spends_on_paintballs:
  (plays_per_month : ℕ) 
  (boxes_per_play : ℕ) 
  (cost_per_box : ℕ)
  (plays_per_month = 3) 
  (boxes_per_play = 3) 
  (cost_per_box = 25) :
  (plays_per_month * boxes_per_play * cost_per_box = 225) :=
by
  sorry

end john_spends_on_paintballs_l399_399045


namespace part1_part2_l399_399441

-- Define the function f
def f (x : ℝ) : ℝ := x - Real.log x

-- Define the function g
def g (x : ℝ) : ℝ := (Real.exp 1 - 1) * x

-- Define the piecewise function F
def F (x a : ℝ) : ℝ :=
if x >= a then f x else g x

-- Prove the first part: x₀ = e if the tangent line at x₀ passes through the origin
theorem part1 (h : ∀ (x₀ : ℝ), f'(x₀) = 1 - 1 / x₀ := sorry) :
  ∃ (x₀ : ℝ), (derivative (fun x => f x) x₀ = 0) ↔ x₀ = Real.exp 1 :=
sorry

-- Prove the second part: The range of a
theorem part2 {a : ℝ} :
  (∀ x >= a, f x) ∪ (∀ x < a, g x) = set.univ → a >= 1 / (Real.exp 1 - 1) :=
sorry

end part1_part2_l399_399441


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l399_399349

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  let primes := [2, 3, 5, 7] in
  let reciprocals := primes.map (λ p => (1 : ℚ) / p) in
  let sum_reciprocals := reciprocals.sum in
  let mean := sum_reciprocals / (primes.length : ℚ) in
  mean = 247 / 840 :=
by
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p => (1 : ℚ) / p)
  let sum_reciprocals := reciprocals.sum
  let mean := sum_reciprocals / (primes.length : ℚ)
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l399_399349


namespace initial_machines_l399_399664

theorem initial_machines (n x : ℕ) (hx : x > 0) (h : x / (4 * n) = x / 20) : n = 5 :=
by sorry

end initial_machines_l399_399664


namespace area_of_polygon_ABCDEF_l399_399387

theorem area_of_polygon_ABCDEF : 
  ∀ (AB BC DC FA GF : ℝ)
  (G : ℝ)
  (cond1 : AB = 10) 
  (cond2 : BC = 15)
  (cond3 : DC = 7)
  (cond4 : FA = 12)
  (cond5 : GF = 6),
  let ABCG := AB * BC in
  let ED := FA - DC in
  let GFED := (1 / 2) * GF * ED in
  let ABCDEF := ABCG  - GFED in
  ABCDEF = 135 :=
begin
  intros,
  rw [cond1, cond2, cond3, cond4, cond5],
  let ABCG := 10 * 15,
  let ED := 12 - 7,
  let GFED := (1 / 2) * 6 * 5,
  let ABCDEF := ABCG - GFED,
  have h1 : ABCG = 150, by norm_num,
  have h2 : ED = 5, by norm_num,
  have h3 : GFED = 15, by norm_num,
  have h4 : ABCDEF = 150 - 15, by rw [h1, h3],
  have h5 : 150 - 15 = 135, by norm_num,
  rw h5,
  refl
end

end area_of_polygon_ABCDEF_l399_399387


namespace find_number_of_cows_l399_399017

-- Define the conditions
def some_cows_eat (total_bags : Nat) (total_days : Nat) : Prop :=
  total_bags = 30 ∧ total_days = 30

def one_cow_eats (single_bag : Nat) (single_days : Nat) : Prop :=
  single_bag = 1 ∧ single_days = 30

-- Define the number of cows based on the conditions
def number_of_cows (total_bags : Nat) (single_bag : Nat) : Nat :=
  total_bags / single_bag

-- Prove the number of cows in the farm
theorem find_number_of_cows (tb sb : Nat) (H1 : some_cows_eat tb 30) (H2 : one_cow_eats sb 30) : number_of_cows tb sb = 30 :=
by
  unfold some_cows_eat at H1
  unfold one_cow_eats at H2
  unfold number_of_cows
  cases H1 with Htotal_bags Htotal_days
  cases H2 with Hsingle_bag Hsingle_days
  rw Htotal_bags
  rw Hsingle_bag
  exact Nat.div_self (by decide) -- Using decidable instance for division

end find_number_of_cows_l399_399017


namespace projection_equality_intersection_at_single_point_l399_399529

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Line3D :=
(p1 p2 : Point3D)

def projection (P : Point3D) (L : Line3D) : Point3D := sorry

axiom A B C D : Point3D
axiom A_not_B : A ≠ B
axiom B_not_C : B ≠ C
axiom C_not_D : C ≠ D
axiom D_not_A : D ≠ A
axiom not_coplanar : ¬ collinear A B C ∧ ¬ coplanar A B C D 

def P : Point3D := sorry
def E := projection P (Line3D.mk A B)
def F := projection P (Line3D.mk B C)
def G := projection P (Line3D.mk C D)
def H := projection P (Line3D.mk D A)

noncomputable def distance_squared (P Q : Point3D) : ℝ :=
(P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2

def AE := distance_squared A E
def BF := distance_squared B F
def CG := distance_squared C G
def DH := distance_squared D H
def EB := distance_squared E B
def FC := distance_squared F C
def GD := distance_squared G D
def HA := distance_squared H A

theorem projection_equality :
  AE + BF + CG + DH = EB + FC + GD + HA := sorry

theorem intersection_at_single_point (h : AE + BF + CG + DH = EB + FC + GD + HA) :
  ∃ Q : Point3D, 
    ∃ plane_E plane_F plane_G plane_H : Point3D → ℝ,
    (∀ pt : Point3D, plane_E pt = (pt.x - E.x)) ∧
    (∀ pt : Point3D, plane_F pt = (pt.x - F.x)) ∧
    (∀ pt : Point3D, plane_G pt = (pt.x - G.x)) ∧
    (∀ pt : Point3D, plane_H pt = (pt.x - H.x)) ∧
    (∀ pt : Point3D, 
      (plane_E pt = 0 ∧ plane_F pt = 0 ∧ plane_G pt = 0) ↔ pt = Q) := sorry

end projection_equality_intersection_at_single_point_l399_399529


namespace sarah_score_is_122_l399_399558

-- Define the problem parameters and state the theorem
theorem sarah_score_is_122 (s g : ℝ)
  (h1 : s = g + 40)
  (h2 : (s + g) / 2 = 102) :
  s = 122 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end sarah_score_is_122_l399_399558


namespace coeff_x14_in_quotient_l399_399253

open Polynomial

noncomputable def P : Polynomial ℤ := X ^ 1051 - 1
noncomputable def D : Polynomial ℤ := X ^ 4 + X ^ 3 + 2 * X ^ 2 + X + 1

-- Define the quotient of P by D
noncomputable def Q : Polynomial ℤ := P / D

-- The statement we need to prove
theorem coeff_x14_in_quotient : coeff Q 14 = -1 := 
sorry

end coeff_x14_in_quotient_l399_399253


namespace power_rule_for_fractions_calculate_fraction_l399_399324

theorem power_rule_for_fractions (a b : ℚ) (n : ℕ) : (a / b)^n = (a^n) / (b^n) := 
by sorry

theorem calculate_fraction (a b n : ℕ) (h : a = 3 ∧ b = 5 ∧ n = 3) : (a / b)^n = 27 / 125 :=
by
  obtain ⟨ha, hb, hn⟩ := h
  simp [ha, hb, hn, power_rule_for_fractions (3 : ℚ) (5 : ℚ) 3]

end power_rule_for_fractions_calculate_fraction_l399_399324


namespace distance_between_hyperbola_vertices_l399_399748

theorem distance_between_hyperbola_vertices :
  let a : ℝ := 12
  let b : ℝ := 7
  ∀ (x y : ℝ), (x^2 / 144 - y^2 / 49 = 1) →
  (2 * a = 24) := 
by
  intros
  unfold a
  unfold b
  sorry

end distance_between_hyperbola_vertices_l399_399748


namespace tan_A_in_triangle_l399_399034

theorem tan_A_in_triangle (A B C : Point) (hC : Angle C = 90) (h1 : distance A B = 13) (h2 : distance B C = 5) :
  tan (angle A) = 5 / 12 :=
sorry

end tan_A_in_triangle_l399_399034


namespace maximum_possible_sum_lengths_l399_399268

def length_of_integer (k : ℕ) : ℕ := ∑ p in (multiset.filter prime (multiset.factor_mult_set (k : ℕ))), p.count p

theorem maximum_possible_sum_lengths (x y z : ℕ) 
  (hx : 1 < x) 
  (hy : 1 < y) 
  (hz : 1 < z) 
  (hx_prime : ∃ p q : ℕ, p ≠ q ∧ prime p ∧ prime q ∧ x = p * q)
  (hy_prime : ∃ p q r : ℕ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ prime p ∧ prime q ∧ prime r ∧ y = p * q * r)
  (h_constraint : x + 3 * y + 5 * z < 5000) :
  length_of_integer x + length_of_integer y + length_of_integer z ≤ 14 :=
by
  sorry

end maximum_possible_sum_lengths_l399_399268


namespace triangle_poly_roots_l399_399568

theorem triangle_poly_roots (A B C : ℝ) (p q : ℝ) (h1 : A + B + C = -p)
  (h2 : A * B + B * C + C * A = q) (h3 : A * B * C = -p) (h4 : A > 0) (h5 : B > 0) (h6 : C > 0)
  (h7 : q ≠ 1) : p ≤ -3*sqrt 3 ∧ q > 1 := by
  sorry

end triangle_poly_roots_l399_399568


namespace length_of_square_side_l399_399591

-- Definitions based on conditions
def perimeter_of_triangle : ℝ := 46
def total_perimeter : ℝ := 78
def perimeter_of_square : ℝ := total_perimeter - perimeter_of_triangle

-- Lean statement for the problem
theorem length_of_square_side : perimeter_of_square / 4 = 8 := by
  sorry

end length_of_square_side_l399_399591


namespace delay_time_l399_399298

theorem delay_time (v : ℝ) (d : ℝ) :
  let usual_time := 18;
      increased_speed_factor := 1.2;
      late_arrival_time := 2;
      speed := d / usual_time;    -- v, distance/18 minutes
      increased_speed := speed * increased_speed_factor;
      travel_time := d / increased_speed;
      actual_travel_time := travel_time + late_arrival_time
  in (usual_time - actual_travel_time) = 5 :=
by
  sorry

end delay_time_l399_399298


namespace motorcyclist_average_speed_l399_399297

theorem motorcyclist_average_speed :
  let distance_ab := 120
  let speed_ab := 45
  let distance_bc := 130
  let speed_bc := 60
  let distance_cd := 150
  let speed_cd := 50
  let time_ab := distance_ab / speed_ab
  let time_bc := distance_bc / speed_bc
  let time_cd := distance_cd / speed_cd
  (time_ab = time_bc + 2)
  → (time_cd = time_ab / 2)
  → avg_speed = (distance_ab + distance_bc + distance_cd) / (time_ab + time_bc + time_cd)
  → avg_speed = 2400 / 47 := sorry

end motorcyclist_average_speed_l399_399297


namespace medicine_duration_l399_399044

theorem medicine_duration (one_third_pill_days : ℕ) (total_pills : ℕ) (average_days_per_month : ℕ) :
  (one_third_pill_days = 3) →
  (total_pills = 60) →
  (average_days_per_month = 30) →
  ((total_pills * one_third_pill_days * 3) / average_days_per_month = 18) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end medicine_duration_l399_399044


namespace tan_theta_eq_neg_3_over_4_implies_expressions_l399_399422

theorem tan_theta_eq_neg_3_over_4_implies_expressions (θ : ℝ) (h : Real.tan θ = -3/4) :
  ( (sin θ - cos θ) / (2 * sin (π + θ) - cos (π - θ)) = -7/10 ) ∧
  ( 2 + sin θ * cos θ - cos θ ^ 2 = 22 / 25 ) :=
by
  sorry

end tan_theta_eq_neg_3_over_4_implies_expressions_l399_399422


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l399_399348

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  let primes := [2, 3, 5, 7] in
  let reciprocals := primes.map (λ p => (1 : ℚ) / p) in
  let sum_reciprocals := reciprocals.sum in
  let mean := sum_reciprocals / (primes.length : ℚ) in
  mean = 247 / 840 :=
by
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p => (1 : ℚ) / p)
  let sum_reciprocals := reciprocals.sum
  let mean := sum_reciprocals / (primes.length : ℚ)
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l399_399348


namespace complementary_angles_decrease_86_percent_l399_399195

theorem complementary_angles_decrease_86_percent (x : ℝ) (h : 10 * x = 90) :
  let small_angle := 3 * x
  let increased_small_angle := small_angle * 1.2
  let large_angle := 7 * x
  let new_large_angle := 90 - increased_small_angle
  (new_large_angle / large_angle) * 100 = 91.4 :=
by
  sorry

end complementary_angles_decrease_86_percent_l399_399195


namespace sum_of_seventh_powers_l399_399651

noncomputable theory

open Complex

-- Given conditions
variables (ζ1 ζ2 ζ3 : ℂ)
variables (h1 : ζ1 + ζ2 + ζ3 = 2) 
variables (h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5) 
variables (h3 : ζ1^3 + ζ2^3 + ζ3^3 = 14) 

-- Prove the required statement
theorem sum_of_seventh_powers : ζ1^7 + ζ2^7 + ζ3^7 = 233 :=
sorry

end sum_of_seventh_powers_l399_399651


namespace find_f_neg3_l399_399825

noncomputable def f : ℝ → ℝ :=
fun x => if x > -1 then 2^x - 1 else f (x + 2)

theorem find_f_neg3 : f (-3) = 1 :=
by
  sorry

end find_f_neg3_l399_399825


namespace min_value_frac_square_sum_l399_399014

theorem min_value_frac_square_sum (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : log 2 x + log 2 y = 1) : 
  ∃ k : ℝ, k = 4 ∧ ∀ t : ℝ, t = (x^2 + y^2) / (x - y) → t ≥ k :=
by
  sorry

end min_value_frac_square_sum_l399_399014


namespace six_smallest_distinct_integers_l399_399393

theorem six_smallest_distinct_integers:
  ∃ (a b c d e f : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a * b * c * d * e = 999999 ∧ a = 3 ∧
    f = 37 ∨
    a * b * c * d * f = 999999 ∧ a = 3 ∧ e = 13 ∨ 
    a * b * d * f * e = 999999 ∧ c = 9 ∧ 
    a * c * d * e * f = 999999 ∧ b = 7 ∧ 
    b * c * d * e * f = 999999 ∧ a = 3 := 
sorry

end six_smallest_distinct_integers_l399_399393


namespace distance_between_vertices_l399_399751

def hyperbola_eq (x y : ℝ) : Prop := 
  x^2 / 144 - y^2 / 49 = 1

theorem distance_between_vertices : 2 * real.sqrt 144 = 24 :=
by {
    -- use sqrt calculation for clarity
    have h : real.sqrt 144 = 12, by {
        exact real.sqrt_eq_iff_sq_eq.mpr (or.inl (by norm_num)),
    },
    rw [h],
    norm_num
}

end distance_between_vertices_l399_399751


namespace tan_A_is_5_over_12_l399_399032

-- Defining the context of triangle ABC.
variables {A B C : Type} [AffineGeometry A] [AffineGeometry B] [AffineGeometry C]

-- Given conditions
def angle_C_is_90 (ABC : Triangle A B C) : Prop := ABC.angle C = 90
def AB_equals_13 (ABC : Triangle A B C) : Prop := ABC.side_length AB = 13
def BC_equals_5 (ABC : Triangle A B C) : Prop := ABC.side_length BC = 5

-- Concluding the result on tan A
theorem tan_A_is_5_over_12 (ABC : Triangle A B C)
  (h1 : angle_C_is_90 ABC)
  (h2 : AB_equals_13 ABC)
  (h3 : BC_equals_5 ABC) : 
  ABC.tan A = 5 / 12 := sorry

end tan_A_is_5_over_12_l399_399032


namespace fib_equiv_formula_l399_399705

def fib (n : ℕ) : ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define the roots of the characteristic equation
noncomputable def φ : Real := (1 + Real.sqrt 5) / 2
noncomputable def φ' : Real := (1 - Real.sqrt 5) / 2

-- Define the Fibonacci formula in terms of the roots
noncomputable def fib_formula (n : ℕ) : Real :=
  (φ^n - φ'^n) / Real.sqrt 5

theorem fib_equiv_formula (n : ℕ) : fib n = Int.ofNat (Int.floor (fib_formula n)) :=
  sorry

end fib_equiv_formula_l399_399705


namespace find_x_range_l399_399804

noncomputable def is_obtuse (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 < 0

theorem find_x_range (x : ℝ) :
  is_obtuse (x, 2 * x) (-3 * x, 2) ↔ x ∈ Set.Ioo (-∞ : ℝ) (-1/3) ∪ Set.Ioo (-1/3 : ℝ) 0 ∪ Set.Ioo (4 / 3 : ℝ) ∞ :=
by 
  sorry

end find_x_range_l399_399804


namespace largest_x_undefined_largest_solution_l399_399774

theorem largest_x_undefined (x : ℝ) :
  (10 * x ^ 2 - 85 * x + 10 = 0) → x = 10 ∨ x = 1 / 10 :=
by
  sorry

theorem largest_solution (x : ℝ) :
  (10 * x ^ 2 - 85 * x + 10 = 0 → x ≤ 10) :=
by
  sorry

end largest_x_undefined_largest_solution_l399_399774


namespace train_cross_platform_time_l399_399283

-- Define the conditions: length of the train, time to cross a signal pole, length of the platform
def train_length : ℝ := 300
def time_pole : ℝ := 18
def platform_length : ℝ := 600.0000000000001

-- Define the expected answer: time to cross the platform
def expected_time_cross_platform : ℝ := 54.00000000000001

-- The proof statement, asserting that the calculated time to cross the platform equals the expected time
theorem train_cross_platform_time :
  let speed := train_length / time_pole in
  let total_distance := train_length + platform_length in
  let time_cross_platform := total_distance / speed in
  time_cross_platform = expected_time_cross_platform :=
by
  sorry

end train_cross_platform_time_l399_399283


namespace ratio_perimeters_of_squares_l399_399954

theorem ratio_perimeters_of_squares 
  (s₁ s₂ : ℝ)
  (h : (s₁ ^ 2) / (s₂ ^ 2) = 25 / 36) :
  (4 * s₁) / (4 * s₂) = 5 / 6 :=
by
  sorry

end ratio_perimeters_of_squares_l399_399954


namespace semicircle_arc_sum_approaches_l399_399958

-- Definitions based on the conditions in the problem.
def diameter (D : ℝ) : ℝ := D
def n_parts (D : ℝ) (n : ℕ) : (ℕ → ℝ) := λ k, D / n
def semicircle_length (D : ℝ) (n : ℕ) : ℝ := (π * D) / (2 * n)
def sum_lengths (D : ℝ) (n : ℕ) : ℝ := n * (π * D) / (2 * n)

-- The statement of the problem to be proven in Lean 4.
theorem semicircle_arc_sum_approaches (D : ℝ) (n : ℕ) (h : 0 < D) (h' : 0 < n) :
  filter.tendsto (sum_lengths D) filter.at_top (𝓝 ((π * D) / 2)) :=
sorry

end semicircle_arc_sum_approaches_l399_399958


namespace distance_between_hyperbola_vertices_l399_399760

theorem distance_between_hyperbola_vertices :
  (∃ a : ℝ, a = real.sqrt 144 ∧ ∀ d : ℝ, d = 2 * a → d = 24) :=
begin
  use real.sqrt 144,
  split,
  { refl },
  { intros d hd,
    rw hd,
    refl }
end

end distance_between_hyperbola_vertices_l399_399760


namespace distance_between_hyperbola_vertices_l399_399745

theorem distance_between_hyperbola_vertices :
  let a : ℝ := 12
  let b : ℝ := 7
  ∀ (x y : ℝ), (x^2 / 144 - y^2 / 49 = 1) →
  (2 * a = 24) := 
by
  intros
  unfold a
  unfold b
  sorry

end distance_between_hyperbola_vertices_l399_399745


namespace clerical_staff_percentage_l399_399543

theorem clerical_staff_percentage (total_employees clerical_fraction reduction_fraction : ℕ) 
  (h_total_emp : total_employees = 3600)
  (h_clerical_frac : clerical_fraction = 1 / 3)
  (h_reduction_frac : reduction_fraction = 1 / 3) :
  (let initial_clerical := clerical_fraction * total_employees in
   let remaining_clerical := (1 - reduction_fraction) * initial_clerical in
   let total_remaining := total_employees - reduction_fraction * initial_clerical in
   (remaining_clerical / total_remaining) * 100 = 25) :=
by
  sorry

end clerical_staff_percentage_l399_399543


namespace total_reptiles_l399_399216

constant numSwamps : ℕ := 4
constant reptilesPerSwamp : ℕ := 356

theorem total_reptiles : numSwamps * reptilesPerSwamp = 1424 := by
  sorry

end total_reptiles_l399_399216


namespace number_of_triangles_from_12_points_l399_399603

theorem number_of_triangles_from_12_points 
  (points : Finset (ℝ × ℝ))
  (h_distinct : points.card = 12)
  (h_on_circle : ∃ (C : ℝ × ℝ) (r : ℝ), ∀ p ∈ points, dist C p = r)
  (h_no_collinear : ∀ p1 p2 p3 ∈ points, ¬Collinear ℝ ({p1, p2, p3} : set (ℝ × ℝ))) : 
  points.card.choose 3 = 220 :=
by
  sorry

end number_of_triangles_from_12_points_l399_399603


namespace simplification_evaluation_l399_399561

noncomputable def simplify_and_evaluate (x : ℤ) : ℚ :=
  (1 - 1 / (x - 1)) * ((x - 1) / ((x - 2) * (x - 2)))

theorem simplification_evaluation (x : ℤ) (h1 : x > 0) (h2 : 3 - x ≥ 0) : 
  simplify_and_evaluate x = 1 :=
by
  have h3 : x = 3 := sorry
  rw [simplify_and_evaluate, h3]
  simp [h3]
  sorry

end simplification_evaluation_l399_399561


namespace tabitha_final_amount_is_six_l399_399164

def initial_amount : ℕ := 25
def amount_given_to_mom : ℕ := 8
def num_items_bought : ℕ := 5
def cost_per_item : ℝ := 0.5

def amount_after_giving_mom : ℝ := initial_amount - amount_given_to_mom
def amount_invested : ℝ := amount_after_giving_mom / 2
def amount_after_investment : ℝ := amount_after_giving_mom - amount_invested
def total_cost_of_items : ℝ := num_items_bought * cost_per_item
def final_amount : ℝ := amount_after_investment - total_cost_of_items

theorem tabitha_final_amount_is_six :
  final_amount = 6 := 
by 
  -- sorry to skip the proof
  sorry

end tabitha_final_amount_is_six_l399_399164


namespace dice_composite_probability_l399_399009

open Nat

noncomputable def probability_composite_number (total_outcomes : ℕ) (non_composite_outcomes : ℕ) : ℚ :=
  1 - (non_composite_outcomes / total_outcomes : ℚ)

theorem dice_composite_probability :
  let total_outcomes := 6^5 * 3 in
  let non_composite_outcomes := 4 in
  probability_composite_number total_outcomes non_composite_outcomes = 5831 / 5832 :=
by 
  sorry

end dice_composite_probability_l399_399009


namespace distance_between_vertices_l399_399753

def hyperbola_eq (x y : ℝ) : Prop := 
  x^2 / 144 - y^2 / 49 = 1

theorem distance_between_vertices : 2 * real.sqrt 144 = 24 :=
by {
    -- use sqrt calculation for clarity
    have h : real.sqrt 144 = 12, by {
        exact real.sqrt_eq_iff_sq_eq.mpr (or.inl (by norm_num)),
    },
    rw [h],
    norm_num
}

end distance_between_vertices_l399_399753


namespace arithmetic_mean_reciprocals_primes_l399_399332

theorem arithmetic_mean_reciprocals_primes : 
  let p := [2, 3, 5, 7] in 
  let reciprocals := p.map (λ n => 1 / (n : ℚ)) in
  (reciprocals.sum / reciprocals.length) = (247 / 840 : ℚ) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l399_399332


namespace train_speed_initial_l399_399307

variable (x : ℝ)
variable (v : ℝ)
variable (average_speed : ℝ := 40 / 3)
variable (initial_distance : ℝ := x)
variable (initial_speed : ℝ := v)
variable (next_distance : ℝ := 4 * x)
variable (next_speed : ℝ := 20)

theorem train_speed_initial : 
  (5 * x) / ((x / v) + (x / 5)) = 40 / 3 → v = 40 / 7 :=
by
  -- Definition of average speed in the context of the problem
  let t1 := x / v
  let t2 := (4 * x) / 20
  let total_distance := 5 * x
  let total_time := t1 + t2
  have avg_speed_eq : total_distance / total_time = 40 / 3 := by sorry
  sorry

end train_speed_initial_l399_399307


namespace percentage_decrease_of_larger_angle_l399_399187

noncomputable def complementary_angles_decrease_percentage : Real :=
let total_degrees := 90
let ratio_sum := 3 + 7
let part := total_degrees / ratio_sum
let smaller_angle := 3 * part
let larger_angle := 7 * part
let increased_smaller_angle := smaller_angle * 1.2
let new_larger_angle := total_degrees - increased_smaller_angle
let decrease_amount := larger_angle - new_larger_angle
(decrease_amount / larger_angle) * 100

theorem percentage_decrease_of_larger_angle
  (smaller_increased_percentage : Real := 20)
  (ratio_three : Real := 3)
  (ratio_seven : Real := 7)
  (total_degrees : Real := 90)
  (expected_decrease : Real := 8.57):
  complementary_angles_decrease_percentage = expected_decrease := 
sorry

end percentage_decrease_of_larger_angle_l399_399187


namespace distance_1_minute_before_catch_up_is_250_l399_399546

noncomputable def distance_between_trucks := 
  let initial_distance := 4 -- initial distance between Truck A and Truck B in kilometers
  let speed_A := 45        -- speed of Truck A in kilometers per hour
  let speed_B := 60        -- speed of Truck B in kilometers per hour
  let time_to_catch_up := (initial_distance / (speed_B - speed_A)) -- time for Truck B to catch up to Truck A
  let one_minute_in_hours := 1 / 60   -- one minute in hours
  let time_before_catch_up := time_to_catch_up - one_minute_in_hours  -- time 1 minute before catching up
  (speed_A * time_before_catch_up + initial_distance - speed_B * time_before_catch_up) * 1000 -- in meters

theorem distance_1_minute_before_catch_up_is_250 :
  distance_between_trucks = 250 :=
by
  unfold distance_between_trucks
  rw [←rat.cast_coe_nat (4 : ℚ), div_sub_div_same, div_eq_mul_inv, sub_mul, div_div_eq_div_mul, sub_eq_add_neg,
      add_assoc, add_comm (↑4 * (((↑15 : ℚ)⁻¹) : ℚ)) ((↑15 : ℚ)⁻¹),
      ←sub_neg_eq_add, ←sub_add, mul_iff_smul_left (Lt.le (Mul.nonneg_of_lqual 0 : ℚ))]

  sorry

end distance_1_minute_before_catch_up_is_250_l399_399546


namespace minimum_area_of_folded_triangle_l399_399684

theorem minimum_area_of_folded_triangle (A B C : Point) (h : Triangle ABC.area = 1) : 
  ∃ p q r s : Point, IsTriangle p q r ∧ (Area (FoldedTriangle ABC p q r s) = 2 / 3) :=
sorry

end minimum_area_of_folded_triangle_l399_399684


namespace monotonicity_and_range_l399_399081

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end monotonicity_and_range_l399_399081


namespace Nadia_notes_per_minute_l399_399119

theorem Nadia_notes_per_minute (H1 : ∀ n : ℕ, 3 * n = 40 * 0.075 * n)
                               (H2 : ∀ t : ℕ, 0.075 * 480 * t = 36 * t / 8)
                               (H3 : ∀ x : ℕ, x = 480 ∧ t = 8) :       
  (480 / 8) = 60 := 
  sorry

end Nadia_notes_per_minute_l399_399119


namespace min_tangent_length_l399_399812

open Real

theorem min_tangent_length (C : set (ℝ×ℝ)) (l : set (ℝ×ℝ))
  (hC : ∀ x y : ℝ, (x, y) ∈ C ↔ (x - 1)^2 + (y - 2)^2 = 4)
  (h_max_dist : ∀ P : ℝ×ℝ, P ∈ C → distance P l ≤ 6)
  (h_dist_center_line : ∃ A : ℝ×ℝ, A ∈ l ∧ distance (1, 2) l = 4) :
  ∃ A B : ℝ×ℝ, A ∈ l ∧ B ∈ C ∧ is_tangent A B C ∧ distance A B = 2 * sqrt 3 :=
sorry

end min_tangent_length_l399_399812


namespace gecko_eats_crickets_on_fourth_day_l399_399292

theorem gecko_eats_crickets_on_fourth_day :
  let total_crickets := 92.5
  let first_day_crickets := 0.35 * total_crickets
  let second_day_crickets := first_day_crickets + 0.10 * first_day_crickets
  let total_first_two_days := first_day_crickets + second_day_crickets
  let fourth_day_crickets := total_crickets - total_first_two_days
  round fourth_day_crickets = 25 :=
by
  sorry

end gecko_eats_crickets_on_fourth_day_l399_399292


namespace graph_symmetric_intersect_one_point_l399_399967

theorem graph_symmetric_intersect_one_point (a b c d : ℝ) :
  (∀ x : ℝ, 2a + (1 / (x - b)) = 2c + (1 / (x - d)) = a + c) →
  (x = (b + d) / 2) →
  ((a - c) * (b - d) = 2) := 
sorry

end graph_symmetric_intersect_one_point_l399_399967


namespace distance_between_hyperbola_vertices_l399_399746

theorem distance_between_hyperbola_vertices :
  let a : ℝ := 12
  let b : ℝ := 7
  ∀ (x y : ℝ), (x^2 / 144 - y^2 / 49 = 1) →
  (2 * a = 24) := 
by
  intros
  unfold a
  unfold b
  sorry

end distance_between_hyperbola_vertices_l399_399746


namespace maximum_value_of_BF2_AF2_l399_399435

theorem maximum_value_of_BF2_AF2 
  (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0) 
  (F1 F2 M N A B : ℝ × ℝ)
  (ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1) 
  (minor_axis : M = (0, -b) ∧ N = (0, b))
  (quadrilateral_perimeter : 2 * (a - b) = 2)
  (line_l : A = F1 ∨ B = F1)
  (AB_dist : dist A B = 4 / 3) :
  max (λ BF2 AF2 : ℝ, BF2 * AF2) (\{(x, y) | x ≤ a + 2/3 ∧ y ≤ a - 2/3 }) = 16 / 9 :=
begin
  sorry
end

end maximum_value_of_BF2_AF2_l399_399435


namespace hyperbola_vertex_distance_l399_399768

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  (x^2 / 144 - y^2 / 49 = 1) →
  ∃ (a : ℝ), a = 12 ∧ 2 * a = 24 :=
by
  intro x y h
  have h1 : 12^2 = 144 := by norm_num
  use 12
  split
  case left =>
    exact rfl
  case right =>
    calc
      2 * 12 = 24 : by norm_num

end hyperbola_vertex_distance_l399_399768


namespace sequence_is_arithmetic_and_general_term_l399_399798

variable (a S : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

def sum_terms (a S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (finset.range (n + 1)).sum a

noncomputable def sequence_condition (a S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 2 * S n = a n ^ 2 + n - 4

theorem sequence_is_arithmetic_and_general_term (a S : ℕ → ℝ) 
  (positive_terms : ∀ n : ℕ, 0 < a n)
  (sum_def : sum_terms a S)
  (cond_seq : sequence_condition a S) :
  is_arithmetic_sequence a ∧ (∀ n : ℕ, a n = n + 2) := 
sorry

end sequence_is_arithmetic_and_general_term_l399_399798


namespace john_monthly_paintball_expense_l399_399047

theorem john_monthly_paintball_expense
  (times_per_month : ℕ)
  (boxes_per_time : ℕ)
  (cost_per_box : ℕ)
  (h1 : times_per_month = 3)
  (h2 : boxes_per_time = 3)
  (h3 : cost_per_box = 25) :
  times_per_month * boxes_per_time * cost_per_box = 225 :=
by
  rw [h1, h2, h3]
  norm_num

end john_monthly_paintball_expense_l399_399047


namespace function_decomposition_l399_399934

theorem function_decomposition (f : ℝ → ℝ) :
  ∃ (a : ℝ) (f₁ f₂ : ℝ → ℝ), a > 0 ∧ (∀ x, f₁ x = f₁ (-x)) ∧ (∀ x, f₂ x = f₂ (2 * a - x)) ∧ (∀ x, f x = f₁ x + f₂ x) :=
sorry

end function_decomposition_l399_399934


namespace angle_sum_equals_l399_399802

theorem angle_sum_equals (
  (EL : Point) (M : Point) (I : Point) (L : Point)
  (h1 : ∠ LME + ∠ MEI = 180)
  (h2 : EL = EI + LM)
  ) : ∠ LEM + ∠ EMI = ∠ MIE :=
sorry

end angle_sum_equals_l399_399802


namespace perp_line_plane_parallel_line_plane_to_perp_lines_l399_399838

variables {Point : Type} [metric_space Point]
variables (m n : set Point) (α : set Point)

def is_line (l : set Point) : Prop := sorry -- Define a line
def is_plane (p : set Point) : Prop := sorry -- Define a plane

def perp (l : set Point) (p : set Point) : Prop := sorry -- Define perpendicularity of line and plane
def parallel (l : set Point) (p : set Point) : Prop := sorry -- Define parallelism of line and plane
def perp_lines (l₁ l₂ : set Point) : Prop := sorry -- Define perpendicularity of two lines

theorem perp_line_plane_parallel_line_plane_to_perp_lines
  (hlm : is_line m) (hln : is_line n) (hp : is_plane α)
  (hm_perp_α : perp m α) (hn_parallel_α : parallel n α) :
  perp_lines m n :=
sorry

end perp_line_plane_parallel_line_plane_to_perp_lines_l399_399838


namespace coeff_x_squared_in_binomial_expansion_is_neg192_l399_399057

-- Define the function and conditions
def f (x : ℝ) : ℝ := cos x + (sqrt 3) * sin x

-- Define the maximum value condition
def a : ℝ := 2

-- Define the binomial expression
noncomputable def binomial_expansion (x : ℝ) :=
  (a * sqrt x - 1 / sqrt x)^6

-- State the proof problem
theorem coeff_x_squared_in_binomial_expansion_is_neg192 : 
  coeff (binomial_expansion x) 2 = -192 :=
by
  sorry

end coeff_x_squared_in_binomial_expansion_is_neg192_l399_399057


namespace monotonicity_of_f_range_of_a_if_no_zeros_l399_399072

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end monotonicity_of_f_range_of_a_if_no_zeros_l399_399072


namespace max_arithmetic_mean_l399_399735

theorem max_arithmetic_mean (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100)
  (h : (a + b) / 2 = (25 / 24) * real.sqrt (a * b)) : 
  (a + b) / 2 ≤ 75 := 
sorry

end max_arithmetic_mean_l399_399735


namespace no_monotone_function_has_uncountably_many_preimages_no_C1_function_has_uncountably_many_preimages_l399_399933

-- Problem (a)
theorem no_monotone_function_has_uncountably_many_preimages (f : ℝ → ℝ) (h₁ : ∀ (x ∈ Icc (0 : ℝ) 1), f x ∈ Icc (0 : ℝ) 1) (h₂ : monotone f) :
  ¬ ∀ y ∈ Icc (0 : ℝ) 1, ∃ S, S ⊆ Icc (0 : ℝ) 1 ∧ uncountable S ∧ ∀ x ∈ S, f x = y :=
sorry

-- Problem (b)
theorem no_C1_function_has_uncountably_many_preimages (f : ℝ → ℝ) (h₁ : ∀ (x ∈ Icc (0 : ℝ) 1), f x ∈ Icc (0 : ℝ) 1) (h₂ : differentiable ℝ f) :
  ¬ ∀ y ∈ Icc (0 : ℝ) 1, ∃ S, S ⊆ Icc (0 : ℝ) 1 ∧ uncountable S ∧ ∀ x ∈ S, f x = y :=
sorry

end no_monotone_function_has_uncountably_many_preimages_no_C1_function_has_uncountably_many_preimages_l399_399933


namespace tabitha_final_amount_l399_399163

theorem tabitha_final_amount 
  (initial_amount : ℕ := 25)
  (given_amount : ℕ := 8)
  (item_count : ℕ := 5)
  (item_cost : ℚ := 0.5) :
  let after_giving := initial_amount - given_amount,
      after_investing := after_giving / 2,
      total_spending := item_count * item_cost,
      final_amount := after_investing - total_spending
  in final_amount = 6 := 
by
  -- use actual values
  let after_giving := (25 : ℚ) - 8,
  let after_investing := after_giving / 2,
  let total_spending := (5 : ℚ) * 0.5,
  have h_after_giving : after_giving = 17 := by norm_num,
  have h_after_investing : after_investing = 8.5 := by norm_num,
  have h_total_spending : total_spending = 2.5 := by norm_num,
  let final_amount := after_investing - total_spending,
  have h_final_amount : final_amount = 6 := by norm_num,
  exact h_final_amount

end tabitha_final_amount_l399_399163


namespace number_of_solutions_l399_399366

theorem number_of_solutions :
  ∃ (sols : Finset ℝ), 
    (∀ x, x ∈ sols → 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 2 * (Real.sin x)^3 - 5 * (Real.sin x)^2 + 2 * Real.sin x = 0) 
    ∧ Finset.card sols = 5 := 
by
  sorry

end number_of_solutions_l399_399366


namespace math_problem_l399_399924

variable {a b c d e f : ℕ}
variable (h1 : f < a)
variable (h2 : (a * b * d + 1) % c = 0)
variable (h3 : (a * c * e + 1) % b = 0)
variable (h4 : (b * c * f + 1) % a = 0)

theorem math_problem
  (h5 : (d : ℚ) / c < 1 - (e : ℚ) / b) :
  (d : ℚ) / c < 1 - (f : ℚ) / a :=
by {
  skip -- Adding "by" ... "sorry" to make the statement complete since no proof is required.
  sorry
}

end math_problem_l399_399924


namespace red_paint_needed_l399_399125

theorem red_paint_needed (ratio_red_white : ℚ) (total_cans : ℕ) 
  (h_ratio : ratio_red_white = 4/3) (h_total : total_cans = 35) : 
  ⌊(4 / (4 + 3)) * 35⌋ = 20 :=
by 
sorry

end red_paint_needed_l399_399125


namespace circumcircle_radius_proof_l399_399864

noncomputable def circumcircle_radius (AB A S : ℝ) : ℝ :=
  if AB = 3 ∧ A = 120 ∧ S = 9 * Real.sqrt 3 / 4 then 3 else 0

theorem circumcircle_radius_proof :
  circumcircle_radius 3 120 (9 * Real.sqrt 3 / 4) = 3 := by
  sorry

end circumcircle_radius_proof_l399_399864


namespace correct_factorization_l399_399255

theorem correct_factorization : 
  (¬ (6 * x^2 * y^3 = 2 * x^2 * 3 * y^3)) ∧ 
  (¬ (x^2 + 2 * x + 1 = x * (x^2 + 2) + 1)) ∧ 
  (¬ ((x + 2) * (x - 3) = x^2 - x - 6)) ∧ 
  (x^2 - 9 = (x - 3) * (x + 3)) :=
by 
  sorry

end correct_factorization_l399_399255


namespace find_l_in_triangle_l399_399040

/-- In triangle XYZ, if XY = 5, YZ = 12, XZ = 13, and YM is the angle bisector from vertex Y with YM = l * sqrt 2, then l equals 60/17. -/
theorem find_l_in_triangle (XY YZ XZ : ℝ) (YM l : ℝ) (hXY : XY = 5) (hYZ : YZ = 12) (hXZ : XZ = 13) (hYM : YM = l * Real.sqrt 2) : 
    l = 60 / 17 :=
sorry

end find_l_in_triangle_l399_399040


namespace arithmetic_mean_reciprocals_first_four_primes_l399_399335

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l399_399335


namespace arithmetic_mean_reciprocals_primes_l399_399329

theorem arithmetic_mean_reciprocals_primes
  (p : Finset ℕ)
  (h_p : p = {2, 3, 5, 7})
  : (p.sum (λ x, 1 / ↑x) / 4) = (247 / 840) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l399_399329


namespace military_exercise_arrangements_l399_399872

theorem military_exercise_arrangements :
  let n := 20 - 2 in -- excluding first and last post
  let k := 5 in
  let total_gaps := n - k in
  let valid_configurations := 580 in -- computed by combinatorial methods
  (total_gaps.choose (k-1) - 6 * total_gaps.choose (k-1 - 4) + 15 * total_gaps.choose (k-1 - 9)) * k.factorial = 69600 :=
by
  sorry

end military_exercise_arrangements_l399_399872


namespace count_integers_in_interval_excluding_zero_l399_399455

theorem count_integers_in_interval_excluding_zero : 
  (finset.filter (λ x, x ≠ 0) (finset.Icc (-3:ℤ) 2)).card = 4 :=
by sorry

end count_integers_in_interval_excluding_zero_l399_399455


namespace even_function_f_l399_399789

noncomputable def f (x : ℝ) : ℝ := if x < 0 then 1 + 2 * x else 1 - 2 * x

theorem even_function_f (x : ℝ) (h_even : ∀ x, f x = f (-x)) (h_f_def : ∀ x < 0, f x = 1 + 2 * x) : x > 0 → f x = 1 - 2 * x :=
by
  intro hx_pos
  have h_neg : -x < 0,
  from neg_pos hx_pos
  
  rw [h_even x, h_f_def (-x) h_neg]
  sorry  

end even_function_f_l399_399789


namespace mary_books_count_l399_399116

noncomputable def initial_books : ℕ := 15

noncomputable def after_first_return : ℕ := initial_books - (40 * initial_books / 100) + 8

noncomputable def after_second_return : ℕ := after_first_return - (25 * after_first_return / 100) + 6

noncomputable def after_third_return : ℕ := after_second_return - (30 * after_second_return / 100) + 12

noncomputable def after_fourth_return : ℕ := after_third_return - (50 * after_third_return / 100) + 10

theorem mary_books_count :
  after_fourth_return = 23 :=
by
  have h1: ℕ := (initial_books - (40 * 15 / 100) + 8),
  have h2: ℕ := (h1 - (25 * h1 / 100) + 6),
  have h3: ℕ := (h2 - (30 * h2 / 100) + 12),
  have h4: ℕ := (h3 - (50 * h3 / 100) + 10),
  exact calculation,

end mary_books_count_l399_399116


namespace arithmetic_geometric_sum_relation_l399_399818

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b_n (n : ℕ) (q : ℕ) : ℕ := q^n

noncomputable def S_n (a : ℕ → ℕ) (n : ℕ) : ℕ := 
  ∑ i in finset.range n, a i

noncomputable def T_n (b : ℕ → ℕ) (n : ℕ): ℕ := 
  ∑ i in finset.range n, b i

theorem arithmetic_geometric_sum_relation (q : ℕ) (hq : q > 0) :
  ∀ (n : ℕ), T_n (b_n q) (2 * n) + 1 = S_n a_n (q^n) → a_n n = 2 * n - 1 :=
begin
  intro n,
  intro h,
  sorry
end

end arithmetic_geometric_sum_relation_l399_399818


namespace problem_l399_399913

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry
noncomputable def c : ℂ := sorry

def condition1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 := sorry
def condition2 : a + b + c = 30 := sorry
def condition3 : (a - b) ^ 2 + (a - c) ^ 2 + (b - c) ^ 2 = 2 * a * b * c := sorry

theorem problem : condition1 ∧ condition2 ∧ condition3 → (a^3 + b^3 + c^3) / (a * b * c) = 33 := 
by 
  sorry

end problem_l399_399913


namespace real_part_of_z_l399_399463

open Complex

theorem real_part_of_z (z : ℂ) (h : I * z = 1 + 2 * I) : z.re = 2 :=
sorry

end real_part_of_z_l399_399463


namespace digit_of_fraction_find_215th_digit_fraction_decimal_215_digit_l399_399245

theorem digit_of_fraction (n : ℕ) (k : ℕ) (hk : k > 0) (r : ℤ) (h : r = (n % k).nat_abs) : 
  (0 : ℤ) ≤ r ∧ r < k :=
  by sorry

theorem find_215th_digit : (215 % 28).nat_abs = 19 := 
  by sorry

theorem fraction_decimal_215_digit :
  ∃ d : ℕ, (d = 3) ∧ (digit_of_fraction 215 28 28 (215 % 28).nat_abs) :=
  by sorry

end digit_of_fraction_find_215th_digit_fraction_decimal_215_digit_l399_399245


namespace test_total_points_l399_399264

theorem test_total_points:
  (total_questions = 40) →
  (two_point_questions = 30) →
  (four_point_questions = 10) →
  (points_for_two_point_questions = 2) →
  (points_for_four_point_questions = 4) →
  (total_points = 100) :=
by
  intro h_total_questions
  intro h_two_point_questions
  intro h_four_point_questions
  intro h_points_for_two_point_questions
  intro h_points_for_four_point_questions
  let total_points_from_two = two_point_questions * points_for_two_point_questions
  let total_points_from_four = four_point_questions * points_for_four_point_questions
  let total_points_calculated = total_points_from_two + total_points_from_four
  have h_points_correct : total_points_calculated = 100 :=
    by sorry
  exact h_points_correct

end test_total_points_l399_399264


namespace problem_l399_399588

def seq_a (a : ℕ → ℚ) := a 1 = 1 / 2 ∧ ∀ n : ℕ, n > 0 → a n - a (n + 1) - 2 * a n * a (n + 1) = 0
def seq_b (b : ℕ → ℚ) (S : ℕ → ℚ) := ∀ n : ℕ, n > 0 → b n - 1 = 2 / 3 * S n

theorem problem :
  (∀ a : ℕ → ℚ, seq_a a → ¬ (∃ n : ℕ, a n = 1 / 2023))
  ∧ (∀ a b S : ℕ → ℚ, seq_a a ∧ seq_b b S → ∀ n : ℕ, n > 0 → (finset.range n).sum (λ k, 1 / a k - b k) = n^2 + n - (3^(n+1))/2 + 3/2)
  ∧ (∀ a : ℕ → ℚ, seq_a a → ∀ n : ℕ, n > 0 → (finset.range n).sum (λ k, a k * a (k + 1)) < 1 / 4)
  ∧ (∀ a b S : ℕ → ℚ, seq_a a ∧ seq_b b S → (finset.range 10).sum (λ k, b (k + 1) / a (k + 1)) = (19 * 3^11) / 2 + 3 / 2) :=
sorry

end problem_l399_399588


namespace area_of_triangle_BCM_l399_399547

theorem area_of_triangle_BCM (AD : ℝ) (h0 : AD = 10 * x)
  (M_on_AD : ℝ) (h1 : M_on_AD = 0.3 * AD)
  (BM_MC_eq_11 : ℝ) (h2 : BM_MC_eq_11 = 11) :
  ∃ (S : ℝ), S = 20 * real.sqrt 6 ∧ 
               S = 0.5 * (10 * x) * (4 * real.sqrt 6) :=
by
  sorry

end area_of_triangle_BCM_l399_399547


namespace number_of_incorrect_statements_l399_399980

-- Definitions for statements in Euclidean geometry:

def statement1 : Prop :=
  ∀ (P : Point) (l : Line), ∃! (m : Line), m ≠ l ∧ parallel m l ∧ passes_through m P

def statement2 : Prop :=
  ∀ (P : Point) (l : Line), ∃! (m : Line), perpendicular m l ∧ passes_through m P

def statement3 : Prop :=
  ∀ (l m : Line), l ≠ m ∧ (parallel l m ∨ intersects l m)

def statement4 : Prop :=
  ∀ (l m : Line), ¬(intersects l m) ∧ (parallel l m)

-- Proof problem:
theorem number_of_incorrect_statements : 
  (¬ statement1) + (¬ statement2) + (¬ statement3) + (¬ statement4) = 0 :=
by sorry

end number_of_incorrect_statements_l399_399980


namespace johnson_family_children_count_l399_399570

theorem johnson_family_children_count (m : ℕ) (x : ℕ) (xy : ℕ) :
  (m + 50 + xy = (2 + x) * 21) ∧ (2 * m + xy = 60) → x = 1 :=
by
  sorry

end johnson_family_children_count_l399_399570


namespace romeo_profit_l399_399937

theorem romeo_profit {chocolates_bought : ℕ} (bars : ℕ)
  (cost_per_chocolate : ℕ) (total_sales : ℕ)
  (packaging_cost_per_chocolate : ℕ)
  (h_bars : bars = 5)
  (h_cost_per_chocolate : cost_per_chocolate = 5)
  (h_total_sales : total_sales = 90)
  (h_packaging_cost_per_chocolate : packaging_cost_per_chocolate = 2) :
  let total_cost := bars * cost_per_chocolate + bars * packaging_cost_per_chocolate in
  let profit := total_sales - total_cost in
  profit = 55 := by 
  intros
  sorry

end romeo_profit_l399_399937


namespace other_root_l399_399856

theorem other_root (m : ℝ) (h : 1^2 + m*1 + 3 = 0) : 
  ∃ α : ℝ, (1 + α = -m ∧ 1 * α = 3) ∧ α = 3 := 
by 
  sorry

end other_root_l399_399856


namespace find_a2005_l399_399990

theorem find_a2005 (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 4)
  (h2 : ∀ n, a (n + 1) = S n + 2 * (n + 1) + 1) 
  (h3 : ∀ n, S n = ∑ i in Finset.range (n + 1), a i) :
  a 2005 = 11 * 2 ^ 2003 - 2 :=
by 
  sorry

end find_a2005_l399_399990


namespace john_monthly_paintball_expense_l399_399048

theorem john_monthly_paintball_expense
  (times_per_month : ℕ)
  (boxes_per_time : ℕ)
  (cost_per_box : ℕ)
  (h1 : times_per_month = 3)
  (h2 : boxes_per_time = 3)
  (h3 : cost_per_box = 25) :
  times_per_month * boxes_per_time * cost_per_box = 225 :=
by
  rw [h1, h2, h3]
  norm_num

end john_monthly_paintball_expense_l399_399048


namespace jeff_cat_shelter_l399_399505

theorem jeff_cat_shelter :
  let initial_cats := 20
  let monday_cats := 2
  let tuesday_cats := 1
  let people_adopted := 3
  let cats_per_person := 2
  let total_cats := initial_cats + monday_cats + tuesday_cats
  let adopted_cats := people_adopted * cats_per_person
  total_cats - adopted_cats = 17 := 
by
  sorry

end jeff_cat_shelter_l399_399505


namespace quadratic_no_real_roots_l399_399960

theorem quadratic_no_real_roots (k : ℝ) : (∀ x : ℝ, x^2 + 2*x + k ≠ 0) ↔ k > 1 :=
by
  sorry

end quadratic_no_real_roots_l399_399960


namespace range_of_x_l399_399357

-- Define the function properties
variable {f : ℝ → ℝ}
variable h_even : ∀ x, f x = f (-x)
variable h_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x

-- Prove the range of x given the condition
theorem range_of_x (x : ℝ) (h_condition : f (Real.log x) > f 1) : 1 / 10 < x ∧ x < 10 :=
by
  sorry

end range_of_x_l399_399357


namespace problem_solution_l399_399456

theorem problem_solution (a b c d e : ℤ) (h : (x - 3)^4 = ax^4 + bx^3 + cx^2 + dx + e) :
  b + c + d + e = 15 :=
by
  sorry

end problem_solution_l399_399456


namespace acme_vowel_soup_word_count_l399_399310

theorem acme_vowel_soup_word_count :
  ∃ (words : ℕ), words = 1920 ∧
    (∀ (word : String), word.length = 5 → 
      (∀ (v : Char), v ∈ "AEIOU" → word.count v ≤ 2)) :=
by
  -- Define the number of words
  let words := 1920
  
  -- Assume a five-letter word satisfying conditions
  assume word : String,
  assume hl : word.length = 5,
  assume (v : Char),
  assume hv : v ∈ "AEIOU"
  
  -- Conditions for vowel counts
  have count_cond : word.count v ≤ 2 from sorry
  
  -- Combine conditions to assert the total word count
  exact ⟨ words, ⟨ rfl, λ word hl hv, count_cond ⟩ ⟩

end acme_vowel_soup_word_count_l399_399310


namespace area_of_triangle_ABC_l399_399388
  
noncomputable def det2x2 (a b c d : ℝ) : ℝ :=
a * d - b * c

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
(u.1 - v.1, u.2 - v.2)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
let v := vector_sub A C
let w := vector_sub B C
in (1 / 2) * abs (det2x2 v.1 v.2 w.1 w.2)

theorem area_of_triangle_ABC :
  let A := (-3, 2) : ℝ × ℝ;
  let B := (5, -1) : ℝ × ℝ;
  let C := (9, 6) : ℝ × ℝ
  in area_of_triangle A B C = 34 :=
by
  -- Definitions of vertices
  let A := (-3, 2)
  let B := (5, -1)
  let C := (9, 6)
  
  -- prove the area
  sorry

end area_of_triangle_ABC_l399_399388


namespace find_d_l399_399778

theorem find_d : ∃ d : ℝ, (∀ x : ℝ, 2 * x^2 + 9 * x + d = 0 ↔ x = (-9 + Real.sqrt 17) / 4 ∨ x = (-9 - Real.sqrt 17) / 4) ∧ d = 8 :=
by
  sorry

end find_d_l399_399778


namespace f_odd_and_define_f_neg_l399_399425

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 - 2 * Real.sin x else -((-x)^2 + 2 * Real.sin x)

theorem f_odd_and_define_f_neg (x : ℝ) (h1 : ∀ x, f(-x) = -f(x))
(h2 : ∀ x, x ≥ 0 → f(x) = x^2 - 2 * Real.sin x) : 
∀ x, x < 0 → f(x) = -x^2 - 2 * Real.sin x := by
sorry

end f_odd_and_define_f_neg_l399_399425


namespace lisa_interest_after_10_years_l399_399948

noncomputable def compounded_amount (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r) ^ n

theorem lisa_interest_after_10_years :
  let P := 2000
  let r := (2 : ℚ) / 100
  let n := 10
  let A := compounded_amount P r n
  A - P = 438 := by
    let P := 2000
    let r := (2 : ℚ) / 100
    let n := 10
    let A := compounded_amount P r n
    have : A - P = 438 := sorry
    exact this

end lisa_interest_after_10_years_l399_399948


namespace staff_battle_station_l399_399698

/-- Given that Captain Zarnin has six job openings and received 36 resumes, 
where one-third of the resumes are unsuitable,
there are 255,024,240 ways to staff his battle station. -/
theorem staff_battle_station :
  let total_resumes := 36
  let unsuitable_resumes := total_resumes / 3
  let suitable_resumes := total_resumes - unsuitable_resumes
  let num_positions := 6
  suitable_resumes = 24 →
  Nat.fact 24 / Nat.fact (24 - num_positions) = 255024240 :=
by
  intros
  sorry

end staff_battle_station_l399_399698


namespace inequality_constant_l399_399719

noncomputable def smallest_possible_real_constant : ℝ :=
  1.0625

theorem inequality_constant (C : ℝ) : 
  (∀ x y z : ℝ, (x + y + z = -1) → 
    |x^3 + y^3 + z^3 + 1| ≤ C * |x^5 + y^5 + z^5 + 1| ) ↔ C ≥ smallest_possible_real_constant :=
sorry

end inequality_constant_l399_399719


namespace geometric_prog_105_l399_399593

theorem geometric_prog_105 {a q : ℝ} 
  (h_sum : a + a * q + a * q^2 = 105) 
  (h_arith : a * q - a = (a * q^2 - 15) - a * q) :
  (a = 15 ∧ q = 2) ∨ (a = 60 ∧ q = 0.5) :=
by
  sorry

end geometric_prog_105_l399_399593


namespace find_reflection_coefficient_l399_399683

-- Define variables and assumptions
def reflectionCoefficient (k : ℝ) (I0 : ℝ) : Prop :=
  let t := 1 - k in
  I0 = 16 * I0 * (1 - k)^4

-- Mathematically equivalent proof problem statement in Lean 4
theorem find_reflection_coefficient (k : ℝ) (I0 : ℝ) : 
  reflectionCoefficient k I0 → k = 0.5 :=
by 
  sorry

end find_reflection_coefficient_l399_399683


namespace percent_boys_in_classroom_l399_399474

def students_ratio_boys_girls (boys girls : ℕ) : Prop :=
  3 * girls = 4 * boys

def total_students (boys girls total : ℕ) : Prop :=
  boys + girls = total

def percentage (part whole : ℕ) : ℝ :=
  (part.cast / whole.cast) * 100

theorem percent_boys_in_classroom : ∃ boys girls, students_ratio_boys_girls boys girls ∧ total_students boys girls 49 ∧ percentage boys 49 = 42.857 := by
  sorry

end percent_boys_in_classroom_l399_399474


namespace stickers_initial_count_l399_399120

theorem stickers_initial_count (S : ℕ) 
  (h1 : (3 / 5 : ℝ) * (2 / 3 : ℝ) * S = 54) : S = 135 := 
by
  sorry

end stickers_initial_count_l399_399120


namespace number_of_revolutions_l399_399584

def diameter := 8 -- feet
def one_mile := 5280 -- feet

def wheel_circumference (d : ℝ) : ℝ :=
  let r := d / 2
  2 * Real.pi * r

def revolutions_to_cover_distance (circumference distance : ℝ) : ℝ :=
  distance / circumference

theorem number_of_revolutions :
  revolutions_to_cover_distance (wheel_circumference diameter) one_mile = 660 / Real.pi :=
by
  let d := diameter
  let C := wheel_circumference d
  let dist := one_mile
  let N := revolutions_to_cover_distance C dist
  sorry

end number_of_revolutions_l399_399584


namespace trajectory_equation_line_equation_l399_399414

-- Given points A and B
def A : ℝ × ℝ := (-Real.sqrt 2, 0)
def B : ℝ × ℝ := (Real.sqrt 2, 0)

-- Moving point P with coordinates (x, y)
variable (x y : ℝ)

-- Condition: The product of the slopes of the lines PA and PB is -1/2
def condition : Prop := (y / (x + Real.sqrt 2)) * (y / (x - Real.sqrt 2)) = -1/2

-- (I) The trajectory equation of P
theorem trajectory_equation (h : condition x y) : (x^2 / 2) + y^2 = 1 := sorry

-- (II) The equation of the line l
variable (k : ℝ)
variable (x1 y1 x2 y2 : ℝ)

-- Condition: Line l intersects curve C at M and N and |MN| = 4*sqrt(2)/3
def line_intersects_curve : Prop :=
  (k = 1 ∨ k = -1) ∧ (x_y_relation x1 y1 k x2 y2)

-- Helper predicate to define the intersection points
def x_y_relation (x1 y1 k x2 y2 : ℝ) : Prop :=
  y1 = k * x1 + 1 ∧ y2 = k * x2 + 1 ∧ 
  ((x1 + x2)^2 - 4 * x1 * x2 = (4/3)^2 * (2 / (1 + 2*k^2)))

-- Prove the specific line equation given condition
theorem line_equation (h : condition x y) (h_inter : line_intersects_curve k x1 y1 x2 y2) :
  (k = 1 ∨ k = -1) :=
begin
  sorry, -- Proof to be implemented
end

end trajectory_equation_line_equation_l399_399414


namespace distance_between_hyperbola_vertices_l399_399749

theorem distance_between_hyperbola_vertices :
  let a : ℝ := 12
  let b : ℝ := 7
  ∀ (x y : ℝ), (x^2 / 144 - y^2 / 49 = 1) →
  (2 * a = 24) := 
by
  intros
  unfold a
  unfold b
  sorry

end distance_between_hyperbola_vertices_l399_399749


namespace find_standard_equation_of_line_find_rectangular_equation_of_curve_find_area_of_triangle_l399_399491

-- Definitions of the parametric and polar equations from problem conditions
def parametric_eq (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 - t, 1 + sqrt 3 * t)

def polar_eq (θ : ℝ) : ℝ :=
  4 * real.sin (θ + real.pi / 3)

-- Statements to prove
theorem find_standard_equation_of_line :
  ∃ a b c, a = sqrt 3 ∧ b = 1 ∧ c = -4 ∧ ∀ t x y,
  (x, y) = parametric_eq t → a * x + b * y + c = 0 :=
sorry

theorem find_rectangular_equation_of_curve :
  ∀ θ ρ x y,
  (ρ = polar_eq θ) ∧
  (x = ρ * real.cos θ) ∧
  (y = ρ * real.sin θ) →
  (x - sqrt 3)^2 + (y - 1)^2 = 4 :=
sorry

theorem find_area_of_triangle (M N : ℝ × ℝ) :
  ∀ x y,
  ((x = sqrt 3 - t) ∧ (y = 1 + sqrt 3 * t) → x^2 + y^2 = 4) ∧
  (distance (0, 0) (sqrt 3, 1) = 2) →
  let M := (sqrt 3 - @param ∂ 0, 0);
      N := (1 + cos (θ + π / 3), 1 + sin (θ+π / 3) ):
  ∃ area, area = 4 * distance (M.1,M.2) (N.1,N.2) :=
sorry

end find_standard_equation_of_line_find_rectangular_equation_of_curve_find_area_of_triangle_l399_399491


namespace john_spends_on_paintballs_l399_399046

theorem john_spends_on_paintballs:
  (plays_per_month : ℕ) 
  (boxes_per_play : ℕ) 
  (cost_per_box : ℕ)
  (plays_per_month = 3) 
  (boxes_per_play = 3) 
  (cost_per_box = 25) :
  (plays_per_month * boxes_per_play * cost_per_box = 225) :=
by
  sorry

end john_spends_on_paintballs_l399_399046


namespace find_cos_Q_l399_399479

-- Given definitions
variables {P Q R : Type} -- Points
variables (PQ PR QR : ℝ) -- Distances

-- Conditions
def right_triangle_PQR : Prop := PQ^2 + QR^2 = PR^2
def length_PQ : PQ = 40
def length_PR : PR = 41

-- Problem Statement
theorem find_cos_Q (h : right_triangle_PQR ∧ length_PQ ∧ length_PR) : 
  let cos_Q := PQ / PR in cos_Q = 40 / 41 := sorry

end find_cos_Q_l399_399479


namespace exists_subset_2000_no_double_l399_399369

theorem exists_subset_2000_no_double : 
  ∃ S : Finset ℕ, S ⊆ Finset.range 3001 ∧ S.card = 2000 ∧ (∀ x y ∈ S, x ≠ 2 * y ∧ y ≠ 2 * x) :=
sorry

end exists_subset_2000_no_double_l399_399369


namespace total_points_correct_l399_399138

-- Define player statistics
structure PlayerStats where
  twoPointers : Nat
  threePointers : Nat
  freeThrows : Nat
  steals : Nat
  rebounds : Nat
  fouls : Nat

-- Define the points calculation function
def total_points (stats : PlayerStats) : Int :=
  let pointsFromTwoPointers := stats.twoPointers * 2
  let pointsFromThreePointers := stats.threePointers * 3
  let pointsFromFreeThrows := stats.freeThrows * 1
  let pointsFromSteals := stats.steals * 1
  let pointsFromRebounds := stats.rebounds * 2
  let pointsFromFouls := stats.fouls * -5
  pointsFromTwoPointers + pointsFromThreePointers + pointsFromFreeThrows + pointsFromSteals + pointsFromRebounds + pointsFromFouls

-- Player statistics
def SamStats : PlayerStats :=
  { twoPointers := 20, threePointers := 10, freeThrows := 5, steals := 4, rebounds := 6, fouls := 2 }

def AlexStats : PlayerStats :=
  { twoPointers := 15, threePointers := 8, freeThrows := 5, steals := 6, rebounds := 3, fouls := 3 }

def JakeStats : PlayerStats :=
  { twoPointers := 10, threePointers := 6, freeThrows := 3, steals := 7, rebounds := 5, fouls := 4 }

def LilyStats : PlayerStats :=
  { twoPointers := 16, threePointers := 4, freeThrows := 7, steals := 3, rebounds := 7, fouls := 1 }

-- Total points calculation
def total_points_all : Int :=
  total_points SamStats + total_points AlexStats + total_points JakeStats + total_points LilyStats

theorem total_points_correct : total_points_all = 238 :=
  by sorry

end total_points_correct_l399_399138


namespace max_value_parabola_l399_399010

noncomputable def parabola (x : ℝ) : ℝ := -3 * x^2 + 7

theorem max_value_parabola : ∃ x : ℝ, parabola x = 7 :=
begin
  use 0,
  simp [parabola],
end

end max_value_parabola_l399_399010


namespace expected_balls_in_original_position_l399_399143

/-- 
  The expected number of balls that are in their original positions after three successive transpositions,
  given seven balls arranged in a circle and three people (Chris, Silva, and Alex) each randomly interchanging two adjacent balls, is 3.2.
-/
theorem expected_balls_in_original_position : 
  let n := 7 in
  let transpositions := 3 in
  let expected_position (n : ℕ) (transpositions : ℕ) : ℚ := sorry in
  expected_position n transpositions = 3.2 := 
sorry

end expected_balls_in_original_position_l399_399143


namespace sum_abs_binom_coeff_l399_399420

theorem sum_abs_binom_coeff (a a1 a2 a3 a4 a5 a6 a7 : ℤ)
    (h : (1 - 2 * x) ^ 7 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) :
    |a1| + |a2| + |a3| + |a4| + |a5| + |a6| + |a7| = 3 ^ 7 - 1 := sorry

end sum_abs_binom_coeff_l399_399420


namespace grid_partition_equal_cells_l399_399708

variable (G : Grid) -- assuming Grid is a defined type representing the grid with dark and light cells.
variable (D L : Nat) -- number of dark cells and light cells
variable (n : Nat) -- total number of cells

-- Hypotheses
-- Total number of dark and light cells are equal
hypothesis (h1 : D = L)
-- Total number of cells is even and divisible by 12.
hypothesis (h2 : n % 12 = 0)
-- Total number of cells is equal to the sum of dark and light cells
hypothesis (h3 : D + L = n)

theorem grid_partition_equal_cells (G : Grid) (D L n : Nat) 
  (h1 : D = L) (h2 : n % 12 = 0) (h3 : D + L = n) :
  ∃ parts : List (List Cell), 
    parts.length = 12 ∧ 
    ∀ part ∈ parts, 
      (dark_cells part).length = (light_cells part).length :=
sorry

end grid_partition_equal_cells_l399_399708


namespace domain_f_log2_theorem_l399_399429

noncomputable def log_base2 (x : ℝ) := log x / log 2

noncomputable def domain_f_log2 (f : ℝ → ℝ) (domain_f_2x : set ℝ) : set ℝ :=
  {x : ℝ | log_base2 x ∈ domain_f_2x}

theorem domain_f_log2_theorem : 
  (domain_f_log2 f (-1, 1) = set.Ioo (real.sqrt 2) 4) :=
begin
  sorry
end

end domain_f_log2_theorem_l399_399429


namespace round_to_hundredth_l399_399607

theorem round_to_hundredth (x : ℝ) (h : x = 2.0359) : Real.approximate_to_hundredth x = 2.04 :=
by
  sorry

end round_to_hundredth_l399_399607


namespace inclination_angle_l399_399222

theorem inclination_angle (x y: ℝ) (h : 2 * x + y + 1 = 0) : 
  let m := -2 in
  let θ := real.arctan m in
  θ = real.pi - real.arctan 2 :=
  sorry

end inclination_angle_l399_399222


namespace area_of_enclosed_figure_l399_399952

noncomputable def area_enclosed_by_curves : ℝ :=
  ∫ (x : ℝ) in (0 : ℝ)..(1 : ℝ), ((x)^(1/2) - x^2)

theorem area_of_enclosed_figure :
  area_enclosed_by_curves = (1 / 3) :=
by
  sorry

end area_of_enclosed_figure_l399_399952


namespace union_A_B_intersection_complement_R_B_A_subset_C_A_implies_a_leq_3_l399_399836

variable {R : Set ℝ} (A B C : Set ℝ)
variable (a : ℝ)

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | 2 < x}
def C := {x : ℝ | 1 < x ∧ x < a}
def complement_R_B := {x : ℝ | x ≤ 2}

theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x ≤ 3} :=
  sorry

theorem intersection_complement_R_B_A : (complement_R_B) ∩ A = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
  sorry

theorem subset_C_A_implies_a_leq_3 (h : C ⊆ A) : a ≤ 3 :=
  sorry

end union_A_B_intersection_complement_R_B_A_subset_C_A_implies_a_leq_3_l399_399836


namespace sin_angle_AOB_l399_399485

-- Define the rectangle ABCD with the given properties.
variables (A B C D O : Type) [DecidableEq A] [AddGroup B]

-- Define the lengths of the sides of the rectangle.
variable (AB BC : ℚ)
noncomputable def AB_value := (15 : ℚ)
noncomputable def BC_value := (8 : ℚ)

-- Define the diagonals intersecting at point O.
def diagonals_intersect_at_O (A B C D O : Type) :=
  let AC := (⟨(AB_value + BC_value)^(1/2), sorry⟩ : ℝ) in
  ∃ (O : ℝ), (AC / 2 = O)

-- Prove sin(angle), knowing the above conditions.
theorem sin_angle_AOB :
  ∀ (A B C D O : Type) (AB BC : ℚ), 
  AB = AB_value ∧ BC = BC_value ∧ diagonals_intersect_at_O A B C D O → 
  sin ∠AOB = (sqrt 389) / 17 := sorry

end sin_angle_AOB_l399_399485


namespace average_comparison_l399_399274

theorem average_comparison (x : ℝ) : 
    (14 + 32 + 53) / 3 = 3 + (21 + 47 + x) / 3 → 
    x = 22 :=
by 
  sorry

end average_comparison_l399_399274


namespace monotonicity_and_no_x_intercept_l399_399096

noncomputable theory

def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

def is_monotonic (a : ℝ) (x : ℝ) : Prop := 
  if x < 1 / a then 
    f a x > f a (x + ε) -- ε is a small positive value
  else if x > 1 / a then 
    f a x < f a (x - ε)
  else
    true -- At x = 1/a, the function transits from decreasing to increasing

theorem monotonicity_and_no_x_intercept 
  (a : ℝ) (h1 : 0 < a) : 
  (∀ x : ℝ, 0 < x → is_monotonic a x) ∧ 
  (∀ x : ℝ, f a x ≠ 0) ↔ 
  (a ∈ Ioi (1 / real.exp 1)) := 
sorry

end monotonicity_and_no_x_intercept_l399_399096


namespace decode_CLUE_is_8671_l399_399226

def BEST_OF_LUCK_code : List (Char × Nat) :=
  [('B', 0), ('E', 1), ('S', 2), ('T', 3), ('O', 4), ('F', 5),
   ('L', 6), ('U', 7), ('C', 8), ('K', 9)]

def decode (code : List (Char × Nat)) (word : String) : Option Nat :=
  word.toList.mapM (λ c => List.lookup c code) >>= (λ digits => 
  Option.some (Nat.ofDigits 10 digits))

theorem decode_CLUE_is_8671 :
  decode BEST_OF_LUCK_code "CLUE" = some 8671 :=
by
  -- Proof omitted
  sorry

end decode_CLUE_is_8671_l399_399226


namespace complex_seventh_root_of_unity_l399_399905

theorem complex_seventh_root_of_unity (r : ℂ) (h1 : r^7 = 1) (h2: r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by
  sorry

end complex_seventh_root_of_unity_l399_399905


namespace paths_from_A_to_B_l399_399663

-- Definitions based on the conditions given in part a)
def paths_from_red_to_blue : ℕ := 2
def paths_from_blue_to_green : ℕ := 3
def paths_from_green_to_purple : ℕ := 2
def paths_from_purple_to_orange : ℕ := 1
def num_red_arrows : ℕ := 2
def num_blue_arrows : ℕ := 2
def num_green_arrows : ℕ := 4
def num_purple_arrows : ℕ := 4
def num_orange_arrows : ℕ := 4

-- Prove the total number of distinct paths from A to B
theorem paths_from_A_to_B : 
  (paths_from_red_to_blue * num_red_arrows) * 
  (paths_from_blue_to_green * num_blue_arrows) * 
  (paths_from_green_to_purple * num_green_arrows) * 
  (paths_from_purple_to_orange * num_purple_arrows) = 16 := 
by sorry

end paths_from_A_to_B_l399_399663


namespace rachel_total_reading_homework_l399_399556

theorem rachel_total_reading_homework : 
  ∀ (literature_pages reading_pages : ℕ), 
  literature_pages = 10 → 
  reading_pages = 6 → 
  literature_pages + reading_pages = 16 := 
by 
  intros literature_pages reading_pages h1 h2
  rw [h1, h2]
  rfl

end rachel_total_reading_homework_l399_399556


namespace functions_linearly_dependent_l399_399134

-- Define the functions sin_x, sin_plus_pi_over_8, and sin_minus_pi_over_8
def sin_x (x : ℝ) := Real.sin x
def sin_plus_pi_over_8 (x : ℝ) := Real.sin (x + Real.pi / 8)
def sin_minus_pi_over_8 (x : ℝ) := Real.sin (x - Real.pi / 8)

-- Define the linear dependence theorem
theorem functions_linearly_dependent : ∃ (α₁ α₂ α₃ : ℝ), (α₁ ≠ 0 ∨ α₂ ≠ 0 ∨ α₃ ≠ 0) ∧ ∀ x : ℝ, α₁ * sin_x x + α₂ * sin_plus_pi_over_8 x + α₃ * sin_minus_pi_over_8 x = 0 :=
by
  sorry

end functions_linearly_dependent_l399_399134


namespace paintings_in_four_weeks_l399_399316

theorem paintings_in_four_weeks (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) :
  hours_per_week = 30 → hours_per_painting = 3 → num_weeks = 4 → 
  (hours_per_week / hours_per_painting) * num_weeks = 40 :=
by
  -- Sorry is used since we are not providing the proof
  sorry

end paintings_in_four_weeks_l399_399316


namespace sum_infinite_series_l399_399702

noncomputable def series_term (n : ℕ) : ℚ := 
  (2 * n + 3) / (n * (n + 1) * (n + 2))

noncomputable def partial_fractions (n : ℕ) : ℚ := 
  (3 / 2) / n - 1 / (n + 1) - (1 / 2) / (n + 2)

theorem sum_infinite_series : 
  (∑' n : ℕ, series_term (n + 1)) = 5 / 4 := 
by
  sorry

end sum_infinite_series_l399_399702


namespace find_q_l399_399497

variables {X Y Z M N Q : Type}
variables [inner_product_space ℝ X] [inner_product_space ℝ Y]
variables [inner_product_space ℝ Z] [inner_product_space ℝ M]
variables [inner_product_space ℝ N]

noncomputable def vect_m : X → Y → Z → Y × Z := sorry -- Add correct definitions
noncomputable def vect_n : X → Z → X × Z := sorry -- Add correct definitions
noncomputable def vect_z : X → Y → Z → Y × Z × X := sorry -- Add correct definitions
noncomputable def vect_q : Y → Z → X := sorry -- Add correct definitions

theorem find_q (overvect_m : M) (overvect_n : N) :
  overvect_m = (4/5 : ℝ) * overvect_z + (1/5 : ℝ) * overvect_y →
  overvect_n = (1/4 : ℝ) * overvect_z + (3/4 : ℝ) * overvect_x →
  ∃ a b c : ℝ, ∃ (overvect_q : Q),
  overvect_q = a • overvect_x + b • overvect_y + c • overvect_z ∧ a + b + c = 1 ∧
  (a = 0) ∧ (b = 9/13) ∧ (c = 4/13) := 
sorry

end find_q_l399_399497


namespace median_of_weights_is_40_l399_399996

def weights : List ℕ := [36, 42, 38, 42, 35, 45, 40]

theorem median_of_weights_is_40 : List.median weights = 40 := 
by 
-- Swap beginning of list sorting
let sorted_weights := List.qsort (≤) weights in
have middle_val := sorted_weights.nth (sorted_weights.length / 2) sorry
exact middle_val = 40
sorry

end median_of_weights_is_40_l399_399996


namespace round_to_hundredth_l399_399608

theorem round_to_hundredth (x : ℝ) (h : x = 2.0359) : Real.approximate_to_hundredth x = 2.04 :=
by
  sorry

end round_to_hundredth_l399_399608


namespace range_of_a_for_monotonicity_l399_399184

open Real

def quadratic_function (a x : ℝ) : ℝ :=
  x^2 - 2*a*x + 1

-- Statement of the problem
theorem range_of_a_for_monotonicity :
  ∀ (a : ℝ), monotone_on (quadratic_function a) (Icc (-2) 2) ↔ a ≤ -2 :=
sorry

end range_of_a_for_monotonicity_l399_399184


namespace tan_alpha_val_expr_value_first_quadrant_expr_value_third_quadrant_l399_399114

variable (α β : ℝ)

-- Definitions derived from the conditions
def vec_a := (Real.cos (α + β), Real.sin (α + β))
def vec_b := (Real.cos (α - β), Real.sin (α - β))
def vec_sum := (4 / 5, 3 / 5)

-- The leaned statement corresponding to the mathematical proof problem
theorem tan_alpha_val (h : vec_a + vec_b = vec_sum) : Real.tan α = 3 / 4 := sorry

theorem expr_value_first_quadrant (h : vec_a + vec_b = vec_sum) (h_tan : Real.tan α = 3 / 4) (h_first_quadrant : α ∈ Set.Ioo 0 (Real.pi / 2)) :
  (2 * Real.cos α ^ 2 - 4 * Real.sin α - 1) / (Real.sqrt 2 * Real.sin (α - Real.pi / 4)) = 53 / 5 := sorry

theorem expr_value_third_quadrant (h : vec_a + vec_b = vec_sum) (h_tan : Real.tan α = 3 / 4) (h_third_quadrant : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  (2 * Real.cos α ^ 2 - 4 * Real.sin α - 1) / (Real.sqrt 2 * Real.sin (α - Real.pi / 4)) = 67 / 5 := sorry

end tan_alpha_val_expr_value_first_quadrant_expr_value_third_quadrant_l399_399114


namespace sin_cos_sum_le_a_l399_399718

theorem sin_cos_sum_le_a (a : ℝ) :
  (∀ x ∈ set.Icc (0 : ℝ) (Real.pi / 4), Real.sin (2 * x) + Real.cos (2 * x) ≤ a) ↔ a ≥ Real.sqrt 2 := 
sorry

end sin_cos_sum_le_a_l399_399718


namespace general_solution_l399_399931

theorem general_solution:
  ∀ (C₁ C₂ : ℝ), (t : ℝ), 
  (∀ x : ℝ,  d^2 x / d t^2 + x = 0) → 
  x(t) = C₁ * cos(t) + C₂ * sin(t) :=
by
  intros C₁ C₂ t d2x_eq
  sorry

end general_solution_l399_399931


namespace integral_problem_a_integral_problem_b_integral_problem_c_integral_problem_d_l399_399276

-- Definition for the first integral problem statement
theorem integral_problem_a : 
  ∀ (C : ℝ), ∫ x in Set.Ioi 0, x^3 * real.log x = (1/4) * (x^4 * real.log x) - (1/16) * x^4 + C := 
by sorry

-- Definition for the second integral problem statement:
theorem integral_problem_b : 
  ∀ (C : ℝ), ∫ x in Set.univ, real.arctan x = x * real.arctan x - (1/2) * real.log (1 + x^2) + C := 
by sorry

-- Definition for the third integral problem statement:
theorem integral_problem_c : 
  ∀ (C : ℝ), ∫ x in Set.univ, x * real.cos x = x * real.sin x + real.cos x + C := 
by sorry

-- Definition for the fourth integral problem statement:
theorem integral_problem_d : 
  ∀ (C : ℝ), ∫ x in Set.univ, x * real.exp x = x * real.exp x - real.exp x + C := 
by sorry

end integral_problem_a_integral_problem_b_integral_problem_c_integral_problem_d_l399_399276


namespace monotonicity_and_range_l399_399082

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end monotonicity_and_range_l399_399082


namespace part1_part2_l399_399787

variable (m : ℝ)

def p (m : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * x - 2 ≥ m^2 - 3 * m
def q (m : ℝ) : Prop := ∃ x0 : ℝ, -1 ≤ x0 ∧ x0 ≤ 1 ∧ m ≤ x0

theorem part1 (h : p m) : 1 ≤ m ∧ m ≤ 2 := sorry

theorem part2 (h : ¬(p m ∧ q m) ∧ (p m ∨ q m)) : (m < 1) ∨ (1 < m ∧ m ≤ 2) := sorry

end part1_part2_l399_399787


namespace number_exceeds_part_by_40_l399_399637

theorem number_exceeds_part_by_40 : ∃ x : ℝ, x = 64 ∧ x - (3 / 8) * x = 40 :=
by
  use 64
  split
  { rfl }
  { sorry }

end number_exceeds_part_by_40_l399_399637


namespace complementary_angles_decrease_percentage_l399_399200

theorem complementary_angles_decrease_percentage :
  ∀ (x : ℝ), (3 * x + 7 * x = 90) →
  (3 * x * 1.2 + 7 * x = 90) →
  (3 * x > 0) →
  (7 * x > 0) →
  let original_larger_angle : ℝ := 7 * x in
  let new_smaller_angle : ℝ := 3 * x * 1.2 in
  let new_larger_angle : ℝ := 90 - new_smaller_angle in
  let decrease : ℝ := original_larger_angle - new_larger_angle in
  let percentage_decrease : ℝ := (decrease / original_larger_angle) * 100 in
  percentage_decrease = 8.57 := 
sorry

end complementary_angles_decrease_percentage_l399_399200


namespace mass_of_fourth_metal_l399_399294

theorem mass_of_fourth_metal 
  (m1 m2 m3 m4 : ℝ)
  (total_mass : m1 + m2 + m3 + m4 = 20)
  (h1 : m1 = 1.5 * m2)
  (h2 : m2 = 3/4 * m3)
  (h3 : m3 = 5/6 * m4) :
  m4 = 20 * (48 / 163) :=
sorry

end mass_of_fourth_metal_l399_399294


namespace arithmetic_mean_reciprocals_primes_l399_399328

theorem arithmetic_mean_reciprocals_primes
  (p : Finset ℕ)
  (h_p : p = {2, 3, 5, 7})
  : (p.sum (λ x, 1 / ↑x) / 4) = (247 / 840) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l399_399328


namespace task_assignment_ways_l399_399232

theorem task_assignment_ways :
  let A := 1; B := 2; C := 3; total_products := 6
  in combinatorics.choose total_products A * combinatorics.choose (total_products - A) B * combinatorics.choose (total_products - A - B) C = combinatorics.choose 6 1 * combinatorics.choose 5 2 * combinatorics.choose 3 3 :=
by
  sorry

end task_assignment_ways_l399_399232


namespace hyperbola_vertex_distance_l399_399762

theorem hyperbola_vertex_distance : 
  ∀ x y: ℝ, (x^2 / 144 - y^2 / 49 = 1) → (∃ a: ℝ, a = 12 ∧ 2 * a = 24) :=
by 
  sorry

end hyperbola_vertex_distance_l399_399762


namespace find_n_value_l399_399817

theorem find_n_value : ∃ n : ℕ, 30 * 25 + 15 * 5 = 15 * 25 + n * 5 ∧ n = 90 :=
by
  use 90
  sorry

end find_n_value_l399_399817


namespace alien_trees_base10_l399_399314

def base7_to_base10 (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.reverse.enum.map (λ ⟨i, d⟩ => d * base ^ i).sum

theorem alien_trees_base10 :
  base7_to_base10 [2, 5, 3] 7 = 136 := by
  sorry

end alien_trees_base10_l399_399314


namespace imaginary_part_of_z2_div_z1_l399_399489

noncomputable def z1 : ℂ := 2 - complex.i
noncomputable def z2 : ℂ := 1 - 3 * complex.i

theorem imaginary_part_of_z2_div_z1 : 
  (complex.im (z2 / z1)) = -1 :=
sorry

end imaginary_part_of_z2_div_z1_l399_399489


namespace sum_100th_row_l399_399358

noncomputable def f : ℕ → ℕ
| 1     := 0
| n + 2 := 2 * f (n + 1) + 4

theorem sum_100th_row : f 100 = 6 * 2^99 - 4 :=
by
  sorry

end sum_100th_row_l399_399358


namespace necessary_and_sufficient_condition_for_parallel_lines_l399_399278

theorem necessary_and_sufficient_condition_for_parallel_lines (a l : ℝ) :
  (a = -1) ↔ (∀ x y : ℝ, ax + 3 * y + 3 = 0 → x + (a - 2) * y + l = 0) := 
sorry

end necessary_and_sufficient_condition_for_parallel_lines_l399_399278


namespace sum_b_n_l399_399920

noncomputable def S (n : ℕ) : ℚ := (3^n + 3) / 2

noncomputable def a : ℕ → ℚ
| 1       := 3
| (n + 1) := 3^n

noncomputable def b : ℕ → ℚ
| 1       := 1 / 3
| (n + 1) := (n : ℚ) * 3^(-n)

noncomputable def T (n : ℕ) : ℚ :=
  if n = 1 then b 1 else (1 / 3) + ∑ i in finset.range n, (i : ℚ) * 3^(-(i + 1))

theorem sum_b_n (n : ℕ) : T n = (13 / 12) - ((6 * n + 3) / (4 * 3^n)) := sorry

end sum_b_n_l399_399920


namespace length_OP_l399_399535

def triangle := {A B C : Type} [metric_space A] [metric_space B] [metric_space C]

variables {A B C O P Q : Type}

-- Assumptions and Conditions
variable (h : ∀ a b c : triangle, dist a b = dist a c)    -- AB = AC
variable (g : ∀ o : triangle, o = A ∧☰ g centroid)        -- O is the centroid
variable (m : ∀ q : triangle, midpoint B C q)             -- Q is the midpoint of BC
variable (h_oq : dist O Q = 5)                            -- OQ = 5 cm

-- Conclusion to prove
theorem length_OP : dist O P = 10 := by
  sorry

end length_OP_l399_399535


namespace sum_of_arithmetic_progression_l399_399059

theorem sum_of_arithmetic_progression 
  (a_1 : ℕ) (b_1 : ℕ)
  (d_a d_b : ℚ)
  (h1 : a_1 = 30)
  (h2 : b_1 = 70)
  (h3 : ∃ d_a d_b, 49 * (d_a + d_b) = 100 ∧ 
      (∀ n, a_1 + (n-1) * d_a + b_1 + (n-1) * d_b = 200)) :
  (∑ n in finset.range 50, a_1 + b_1 + (n-1) * (d_a + d_b)) = 7500 := 
sorry

end sum_of_arithmetic_progression_l399_399059


namespace trailing_zeros_10_factorial_base_15_l399_399717

theorem trailing_zeros_10_factorial_base_15 : 
  let factorial (n : ℕ) : ℕ := n.factorial in
  ∀ (n b : ℕ), n = 10 → b = 15 → 
  ∃ k : ℕ, ∀ m : ℕ, n.divisor_count b = m → trailing_zeros (factorial n) b = k → k = 2 := 
by
    intros _ _ hn hb
    existsi 2
    intros m h1 h2
    sorry

end trailing_zeros_10_factorial_base_15_l399_399717


namespace proof_avg_failures_l399_399171

noncomputable def avg_failures (T : ℝ) (p : ℝ) (N : ℕ) : ℝ :=
  let λ := N * p in
  if 1 - exp(-λ) = 0.98 then λ else 0

noncomputable def problem_statement : Prop :=
  avg_failures T p N = 3.9

theorem proof_avg_failures (T : ℝ) (p : ℝ) (N : ℕ) : problem_statement :=
by
  unfold problem_statement
  rw [avg_failures]
  sorry

end proof_avg_failures_l399_399171


namespace monotonicity_of_f_range_of_a_if_no_zeros_l399_399073

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end monotonicity_of_f_range_of_a_if_no_zeros_l399_399073


namespace hyperbola_vertex_distance_l399_399765

theorem hyperbola_vertex_distance : 
  ∀ x y: ℝ, (x^2 / 144 - y^2 / 49 = 1) → (∃ a: ℝ, a = 12 ∧ 2 * a = 24) :=
by 
  sorry

end hyperbola_vertex_distance_l399_399765


namespace angle_between_vectors_l399_399840

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Define the conditions 
def magnitude_a : ℝ := 2
def magnitude_b : ℝ := 1

def perpendicular_condition : Prop := 
  let dot_product := (a + 3 • b) ⬝ (2 • a - b)
  dot_product = 0

-- Define the angle θ 
def θ : ℝ := 2 * Real.pi / 3

-- Prove the angle between vectors a and b is θ given the conditions
theorem angle_between_vectors (a b : EuclideanSpace ℝ (Fin 2)) 
  (h1 : ‖a‖ = magnitude_a) (h2 : ‖b‖ = magnitude_b) (h3 : perpendicular_condition): 
  Real.angle a b = θ := 
sorry

end angle_between_vectors_l399_399840


namespace remainder_4x_div_9_l399_399641

theorem remainder_4x_div_9 (x : ℕ) (k : ℤ) (h : x = 9 * k + 5) : (4 * x) % 9 = 2 := 
by sorry

end remainder_4x_div_9_l399_399641


namespace cubic_sum_l399_399567

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = 1) (h3 : a * b * c = -2) : 
  a^3 + b^3 + c^3 = -6 :=
sorry

end cubic_sum_l399_399567


namespace janet_total_lives_l399_399650

/-
  Janet's initial lives: 38
  Lives lost: 16
  Lives gained: 32
  Prove that total lives == 54 after the changes
-/

theorem janet_total_lives (initial_lives lost_lives gained_lives : ℕ) 
(h1 : initial_lives = 38)
(h2 : lost_lives = 16)
(h3 : gained_lives = 32):
  initial_lives - lost_lives + gained_lives = 54 := by
  sorry

end janet_total_lives_l399_399650


namespace anne_speed_ratio_l399_399323

theorem anne_speed_ratio (B A A' : ℝ) (h_A : A = 1/12) (h_together_current : (B + A) * 4 = 1) (h_together_new : (B + A') * 3 = 1) :
  A' / A = 2 := 
by
  sorry

end anne_speed_ratio_l399_399323


namespace correct_propositions_l399_399627

-- Proposition A: For all x in ℝ, 2^(x-1) > 0
def proposition_A : Prop := ∀ x : ℝ, 2^(x - 1) > 0

-- Proposition B: For all x in ℕ*, (x-1)^2 > 0
def proposition_B : Prop := ∀ x : ℕ, x > 0 → (x - 1)^2 > 0

-- Proposition C: There exists x₀ in ℝ such that lg x₀ < 1
def proposition_C : Prop := ∃ x₀ : ℝ, real.log x₀ < 1

-- Proposition D: There exists x₀ in ℝ such that tan x₀ = 2
def proposition_D : Prop := ∃ x₀ : ℝ, real.tan x₀ = 2

-- Theorems to be proved
theorem correct_propositions : proposition_A ∧ ¬ proposition_B ∧ proposition_C ∧ proposition_D :=
by 
  split,
  sorry,
  split,
  sorry,
  split,
  sorry,
  sorry

end correct_propositions_l399_399627


namespace length_of_RU_l399_399039

-- Definitions of lengths and intersections
variable (P Q R S T U : Type) [MetricSpace P]
variable (PQ QR PR PT PU : ℝ)
variable (anglePQR : ℝ)
variable (PR_intersect_S : Intersect PR S)
variable (angle_bisector_intersect_T : Intersect (AngleBisector anglePQR) T) 
variable (circumPS_circle_intersect_U : Intersect (Circumcircle PTS) U)

-- Given conditions
axiom pq_eq : PQ = 13
axiom qr_eq : QR = 30
axiom pr_eq : PR = 34
axiom ps_intersects_pr_at_S : PR_intersect_S
axiom bisector_intersects_circumcircle_at_T : angle_bisector_intersect_T
axiom circumPS_intersects_pq_at_U_not_P : circumPS_circle_intersect_U

-- Goal: Prove the length of RU is 29
theorem length_of_RU : RU = 29 :=
sorry

end length_of_RU_l399_399039


namespace rectangle_width_length_ratio_l399_399490

theorem rectangle_width_length_ratio (w : ℕ) (h : w + 10 = 15) : w / 10 = 1 / 2 :=
by sorry

end rectangle_width_length_ratio_l399_399490


namespace farm_corn_cobs_total_l399_399289

theorem farm_corn_cobs_total 
  (field1_rows : ℕ) (field1_cobs_per_row : ℕ) 
  (field2_rows : ℕ) (field2_cobs_per_row : ℕ)
  (field3_rows : ℕ) (field3_cobs_per_row : ℕ)
  (field4_rows : ℕ) (field4_cobs_per_row : ℕ)
  (h1 : field1_rows = 13) (h2 : field1_cobs_per_row = 8)
  (h3 : field2_rows = 16) (h4 : field2_cobs_per_row = 12)
  (h5 : field3_rows = 9) (h6 : field3_cobs_per_row = 10)
  (h7 : field4_rows = 20) (h8 : field4_cobs_per_row = 6) :
  field1_rows * field1_cobs_per_row + 
  field2_rows * field2_cobs_per_row + 
  field3_rows * field3_cobs_per_row + 
  field4_rows * field4_cobs_per_row = 506 :=
by
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  norm_num
  sorry

end farm_corn_cobs_total_l399_399289


namespace min_value_of_f_l399_399965

theorem min_value_of_f (ϕ : ℝ) (hϕ : |ϕ| < π) :
  let g := (λ x : ℝ, sin (2 * x + ϕ))
  in (g (0)) - (g (π / 2)) / 2 =
  - (sqrt 3 / 2) := 
sorry

end min_value_of_f_l399_399965


namespace Jane_total_profit_is_87_6_l399_399888

def profit_per_week (eggs_per_bird_per_week : ℕ → ℕ) (price_per_dozen : ℕ → ℕ) (feeding_cost_per_bird : ℕ → ℕ) 
  (num_birds : ℕ → ℕ) (eggs_sold_fraction : ℕ → ℚ) : ℚ :=
  let total_eggs (bird_type : ℕ) := num_birds bird_type * eggs_per_bird_per_week bird_type
  let dozens (bird_type : ℕ) := (total_eggs bird_type * eggs_sold_fraction bird_type) / 12 
  let earnings (bird_type : ℕ) := dozens bird_type * price_per_dozen bird_type
  let total_earnings := earnings 0 + earnings 1 + earnings 2
  let feeding_cost := num_birds 0 * feeding_cost_per_bird 0 + num_birds 1 * feeding_cost_per_bird 1 + num_birds 2 * feeding_cost_per_bird 2
  total_earnings - feeding_cost

def total_profit : ℚ :=
  let weekdays := [1, 2, 3]
  let eggs_per_bird_per_week := λ bird_type, if bird_type = 0 then 6 else if bird_type = 1 then 4 else 10
  let price_per_dozen := λ bird_type, if bird_type = 0 then 2 else if bird_type = 1 then 3 else 4
  let feeding_cost_per_bird := λ bird_type, if bird_type = 0 then 0.5 else if bird_type = 1 then 0.75 else 0.6
  let num_birds := λ bird_type, if bird_type = 0 then 10 else if bird_type = 1 then 8 else 12
  let eggs_sold_fractions := [0.5, 0.75, 1.0]
  weekdays.map (λ week, profit_per_week eggs_per_bird_per_week price_per_dozen feeding_cost_per_bird num_birds 
                (λ bird_type, if week = 1 then (if bird_type = 2 then 0.5 else 1.0) else if week = 2 then (if bird_type = 1 then 0.75 else 1.0) else (if bird_type = 0 then 0 else 1.0)))
    |>.sum

theorem Jane_total_profit_is_87_6 : total_profit = 87.6 := by
  sorry

end Jane_total_profit_is_87_6_l399_399888


namespace truck_mileage_l399_399993

theorem truck_mileage (front_tire_lifetime rear_tire_lifetime : ℕ) 
  (h_front : front_tire_lifetime = 25000) 
  (h_rear : rear_tire_lifetime = 15000) : 
  ∃ x : ℕ, x = 18750 :=
by {
  have h_wear_rate_front : ℚ := 1 / 25000,
  have h_wear_rate_rear : ℚ := 1 / 15000,
  sorry -- The actual proof would follow 
}

end truck_mileage_l399_399993


namespace maximize_huabei_l399_399280

noncomputable def digit_assignment : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
λ shao jun jingtan lun shu, (shao ≠ jun ∧ shao ≠ jingtan ∧ shao ≠ lun ∧ shao ≠ shu ∧ 
                            jun ≠ jingtan ∧ jun ≠ lun ∧ jun ≠ shu ∧ 
                            jingtan ≠ lun ∧ jingtan ≠ shu ∧ lun ≠ shu ∧ 
                            shao ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                            jun ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                            jingtan ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                            lun ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                            shu ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})

theorem maximize_huabei (shao jun jingtan lun shu : ℕ)
  (h : digit_assignment shao jun jingtan lun shu):
  (shao * 100 + jun * 10 + jingtan = 975) ∧  
  (15 = (shao * jun + jingtan + lun * shu) / (\text{华杯赛})) :=
  sorry

end maximize_huabei_l399_399280


namespace range_of_a_l399_399223

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 5 * a) ↔ (4 ≤ a ∨ a ≤ 1) :=
by
  sorry

end range_of_a_l399_399223


namespace three_dice_product_probability_divisible_by_10_l399_399397

open ProbabilityTheory

def die_faces := {1, 2, 3, 4, 5, 6}

def is_divisible_by_10 (n : ℕ) : Prop :=
  (10 ∣ n)

theorem three_dice_product_probability_divisible_by_10 :
  (∑ outcome in
    (Finset.product₃ die_faces die_faces die_faces),
    if is_divisible_by_10 (outcome.1 * outcome.2 * outcome.3) then (1 : ℝ) else 0) / (6 * 6 * 6) = 2 / 3 := 
by sorry

end three_dice_product_probability_divisible_by_10_l399_399397


namespace ratio_of_perpendiculars_l399_399478

-- Definitions from the conditions
variable (A B C D M E F : Type)
variable [parallelogram A B C D]
variable [on_diag M A C]
variable [perp M E A B]
variable [perp M F A D]

-- The theorem to prove
theorem ratio_of_perpendiculars
  (h_parallelogram : parallelogram A B C D)
  (h_on_diag : on_diag M A C)
  (h_perp1 : perp M E A B)
  (h_perp2 : perp M F A D) :
  (ME / MF) = (AD / AB) :=
sorry

end ratio_of_perpendiculars_l399_399478


namespace vacation_cost_in_usd_l399_399362

theorem vacation_cost_in_usd :
  let n := 7
  let rent_per_person_eur := 65
  let transport_per_person_usd := 25
  let food_per_person_gbp := 50
  let activities_per_person_jpy := 2750
  let eur_to_usd := 1.20
  let gbp_to_usd := 1.40
  let jpy_to_usd := 0.009
  let total_rent_usd := n * rent_per_person_eur * eur_to_usd
  let total_transport_usd := n * transport_per_person_usd
  let total_food_usd := n * food_per_person_gbp * gbp_to_usd
  let total_activities_usd := n * activities_per_person_jpy * jpy_to_usd
  let total_cost_usd := total_rent_usd + total_transport_usd + total_food_usd + total_activities_usd
  total_cost_usd = 1384.25 := by
    sorry

end vacation_cost_in_usd_l399_399362


namespace major_axis_length_l399_399676

theorem major_axis_length :
  ∀ (r : ℝ), r = 2 → ∀ (major_factor : ℝ), major_factor = 0.75 →
  ∀ (minor_axis major_axis : ℝ), minor_axis = 2 * r → major_axis = minor_axis + major_factor * minor_axis →
  major_axis = 7 :=
by
  intros r hr major_factor hf minor_axis h_minor major_axis h_major
  rw [hr, hf] at *
  rw [h_minor] at h_major
  sorry

end major_axis_length_l399_399676


namespace arithmetic_sequence_term_difference_l399_399619

theorem arithmetic_sequence_term_difference :
  let a : ℕ := 3
  let d : ℕ := 6
  let t1 := a + 1499 * d
  let t2 := a + 1503 * d
  t2 - t1 = 24 :=
    by
    sorry

end arithmetic_sequence_term_difference_l399_399619


namespace no_solution_inequality_l399_399466

theorem no_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 2 * a ≥ 0) → a ≥ (Real.sqrt 3 + 1) / 4 :=
by
  intro h
  sorry

end no_solution_inequality_l399_399466


namespace total_students_left_l399_399984

-- Definitions for given conditions
def initialBoys := 14
def initialGirls := 10
def boysDropOut := 4
def girlsDropOut := 3

-- The proof problem statement
theorem total_students_left : 
  initialBoys - boysDropOut + (initialGirls - girlsDropOut) = 17 := 
by
  sorry

end total_students_left_l399_399984


namespace Morley_l399_399691

-- define the vertices and trisectors
variable (A B C : Point)
variable (α1 α1' α2 α2' α3 α3' : Ray)
variable (β1 β1' β2 β2' β3 β3' : Ray)
variable (γ1 γ1' γ2 γ2' γ3 γ3' : Ray)

-- define the angles between the trisectors
axiom trisector_condition1 :
  (angle AB α1 = π / 3) ∧ (angle α1 α1' = π / 3) ∧ (angle α1' AC = π / 3)
axiom trisector_condition2 :
  (angle AB α2 = π / 3) ∧ (angle α2 α2' = π / 3) ∧ (angle α2' AC = π / 3)
axiom trisector_condition3 :
  (angle AB α3 = π / 3) ∧ (angle α3 α3' = π / 3) ∧ (angle α3' AC = π / 3)
-- similarly for β and γ conditions

-- define the triangles
noncomputable def triangle_formed (i j k : ℕ) : Triangle := {
  p1 := intersection_line (α_i i) (β_j' j)
  p2 := intersection_line (β_j j) (γ_k' k)
  p3 := intersection_line (γ_k k) (α_i' i)
}

-- main theorem
theorem Morley's_theorem_equilateral (i j k : ℕ) (h : (i + j + k - 1) % 3 ≠ 0) :
  is_equilateral (triangle_formed i j k) ∧
  sides_parallel (triangle_formed i j k) ∧
  vertices_on_lines (triangle_formed i j k) :=
sorry

end Morley_l399_399691


namespace min_period_pi_and_decreasing_l399_399689

def period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x > f y

theorem min_period_pi_and_decreasing (f : ℝ → ℝ) :
  (f = λ x, 2 * |sin x|) →
  period f π ∧ decreasing_on f (set.Ioo (π / 2) π) :=
by
  sorry

end min_period_pi_and_decreasing_l399_399689


namespace sum_valid_n_l399_399713

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def sum_of_n (n_max : ℕ) : ℕ :=
  ∑ n in finset.range (n_max + 1), 
    if is_perfect_square ((n + 1) / 2) then n else 0

theorem sum_valid_n : sum_of_n 100 = 273 := by
  sorry

end sum_valid_n_l399_399713


namespace exists_three_partition_solutions_l399_399128

theorem exists_three_partition_solutions :
  ∃ (a b c d e f g : ℕ),
    ∃ (p : list ℕ) (q : list ℕ) (r : list ℕ),
      p.perm [a, b, c, d, e, f, g] ∧
        q.perm [a, b, c, d, e, f, g] ∧
          r.perm [a, b, c, d, e, f, g] ∧
            -- Conditions ensuring the arrangements satisfy the equal sum conditions
            ((p.take 3).sum = (p.drop 3).sum) ∧
            ((q.take 3).sum = (q.drop 3).sum) ∧
            ((r.take 3).sum = (r.drop 3).sum) :=
sorry

end exists_three_partition_solutions_l399_399128


namespace odd_function_def_l399_399815

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x * (x - 1)
else -x * (x + 1)

theorem odd_function_def {x : ℝ} (h : x > 0) :
  f x = -x * (x + 1) :=
by
  sorry

end odd_function_def_l399_399815


namespace decreasing_order_l399_399779

def part_x (x : ℝ) (y : ℝ) : ℝ := x / y

noncomputable def part_Alex := part_x 1 6
noncomputable def part_Beth := part_x 1 4
noncomputable def part_Cyril := part_x 1 3
noncomputable def part_Elsa := part_x 1 7
noncomputable def part_Fred := 1 - (part_Alex + part_Beth + part_Cyril + part_Elsa)

theorem decreasing_order : 
    (part_Cyril > part_Beth) ∧ 
    (part_Beth > part_Alex) ∧ 
    (part_Alex > part_Elsa) ∧ 
    (part_Elsa > part_Fred) :=
by sorry

end decreasing_order_l399_399779


namespace raghu_investment_l399_399243

theorem raghu_investment
  (R trishul vishal : ℝ)
  (h1 : trishul = 0.90 * R)
  (h2 : vishal = 0.99 * R)
  (h3 : R + trishul + vishal = 6647) :
  R = 2299.65 :=
by
  sorry

end raghu_investment_l399_399243


namespace lambda_range_l399_399841

theorem lambda_range (λ θ : ℝ) (a b : ℝ × ℝ)
  (h_a : a = (λ, 1))
  (h_b : b = (1, -1)) 
  (h_acute : ∃ θ : ℝ, 0 < θ ∧ θ < π/2 ∧ 
             ((1 * λ + 1 * (-1) = λ - 1 > 0))) : λ > 1 :=
sorry

end lambda_range_l399_399841


namespace rectangle_coloring_l399_399837

theorem rectangle_coloring (m n : ℕ) (h1 : m > 0) (h2 : n > 0) :
  ∃ (num_colorings : ℕ), num_colorings = 18 * 2^(m*n - 1) * 3^(m+n-2) :=
by
  use 18 * 2^(m*n - 1) * 3^(m+n-2)
  sorry

end rectangle_coloring_l399_399837


namespace smallest_rectangles_to_cover_square_l399_399620

theorem smallest_rectangles_to_cover_square :
  ∃ n : ℕ, 
    (∃ a : ℕ, a = 3 * 4) ∧
    (∃ k : ℕ, k = lcm 3 4) ∧
    (∃ s : ℕ, s = k * k) ∧
    (s / a = n) ∧
    n = 12 :=
by
  sorry

end smallest_rectangles_to_cover_square_l399_399620


namespace cannot_finish_third_l399_399940

-- Define the racers
inductive Racer
| P | Q | R | S | T | U
open Racer

-- Define the conditions
def beats (a b : Racer) : Prop := sorry  -- placeholder for strict order
def ties (a b : Racer) : Prop := sorry   -- placeholder for tie condition
def position (r : Racer) (p : Fin (6)) : Prop := sorry  -- placeholder for position in the race

theorem cannot_finish_third :
  (beats P Q) ∧
  (ties P R) ∧
  (beats Q S) ∧
  ∃ p₁ p₂ p₃, position P p₁ ∧ position T p₂ ∧ position Q p₃ ∧ p₁ < p₂ ∧ p₂ < p₃ ∧
  ∃ p₄ p₅, position U p₄ ∧ position S p₅ ∧ p₄ < p₅ →
  ¬ position P (3 : Fin (6)) ∧ ¬ position U (3 : Fin (6)) ∧ ¬ position S (3 : Fin (6)) :=
by sorry   -- Proof is omitted

end cannot_finish_third_l399_399940


namespace distance_between_hyperbola_vertices_l399_399761

theorem distance_between_hyperbola_vertices :
  (∃ a : ℝ, a = real.sqrt 144 ∧ ∀ d : ℝ, d = 2 * a → d = 24) :=
begin
  use real.sqrt 144,
  split,
  { refl },
  { intros d hd,
    rw hd,
    refl }
end

end distance_between_hyperbola_vertices_l399_399761


namespace solve_quadratic_1_solve_quadratic_2_l399_399146

theorem solve_quadratic_1 : ∀ x : ℝ, x^2 - 5 * x + 4 = 0 ↔ x = 4 ∨ x = 1 :=
by sorry

theorem solve_quadratic_2 : ∀ x : ℝ, x^2 = 4 - 2 * x ↔ x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5 :=
by sorry

end solve_quadratic_1_solve_quadratic_2_l399_399146


namespace duke_extra_three_pointers_l399_399452

theorem duke_extra_three_pointers
    (old_record : ℕ)
    (points_needed_to_tie : ℕ)
    (exceeded_points : ℕ)
    (free_throws : ℕ)
    (free_throw_value : ℕ)
    (regular_baskets : ℕ)
    (regular_basket_value : ℕ)
    (normal_three_pointers : ℕ)
    (three_pointer_value : ℕ)
    (final_game_points : ℕ) :
    old_record = 257 →
    points_needed_to_tie = 17 →
    exceeded_points = 5 →
    free_throws = 5 →
    free_throw_value = 1 →
    regular_baskets = 4 →
    regular_basket_value = 2 →
    normal_three_pointers = 2 →
    three_pointer_value = 3 →
    final_game_points = points_needed_to_tie + exceeded_points →
    (final_game_points - (free_throws * free_throw_value + regular_baskets * regular_basket_value)) / three_pointer_value - normal_three_pointers = 1 :=
begin
    intros h_old_record h_points_needed_to_tie h_exceeded_points h_free_throws h_free_throw_value h_regular_baskets h_regular_basket_value h_normal_three_pointers h_three_pointer_value h_final_game_points,
    rw [h_old_record, h_points_needed_to_tie, h_exceeded_points, h_free_throws, h_free_throw_value, h_regular_baskets, h_regular_basket_value, h_normal_three_pointers, h_three_pointer_value, h_final_game_points],
    sorry
end

end duke_extra_three_pointers_l399_399452


namespace people_with_uncool_parents_l399_399230

theorem people_with_uncool_parents :
  ∀ (total cool_dads cool_moms cool_both : ℕ),
    total = 50 →
    cool_dads = 25 →
    cool_moms = 30 →
    cool_both = 15 →
    (total - (cool_dads + cool_moms - cool_both)) = 10 := 
by
  intros total cool_dads cool_moms cool_both h1 h2 h3 h4
  sorry

end people_with_uncool_parents_l399_399230


namespace sum_powers_of_i_l399_399354

noncomputable def i : ℂ := complex.I

theorem sum_powers_of_i : 
  (i ^ 4 = 1) → 2 * (∑ k in (finset.range 101).map (int.to_nat ∘ (+ (-50))), i^k) = 2 :=
by
  sorry

end sum_powers_of_i_l399_399354


namespace original_denominator_l399_399682

theorem original_denominator (d : ℕ) (h : 3 * (d : ℚ) = 2) : d = 3 := 
by
  sorry

end original_denominator_l399_399682


namespace rook_path_exists_l399_399926

theorem rook_path_exists (chessboard : Matrix (Fin 8) (Fin 8) ℕ) (sq1 sq2 : Fin 8 × Fin 8) (c : ℕ) :
  (∀ i j, (chessboard i j = c ↔ ((i + j) % 2 = 0))) → 
  chessboard sq1.1 sq1.2 = c → chessboard sq2.1 sq2.2 = c → 
  ∃ f : ℕ → Fin 8 × Fin 8, 
    (f 0 = sq1) ∧ 
    (f (64 * 2 - 1) = sq2) ∧ 
    (∀ t, t < 64 → f (t + 1) ≠ f t ∧ rook_move (f t) (f (t + 1))) ∧ 
    (∀ t₁ t₂, t₁ < 64 → t₂ < 64 → t₁ ≠ t₂ → f t₁ ≠ f t₂) ∧
    (∃ t₃, t₃ < 64 * 2 - 1 ∧ f t₃ = sq2)
:= sorry

end rook_path_exists_l399_399926


namespace discount_each_book_l399_399472

-- Definition of conditions
def original_price : ℝ := 5
def num_books : ℕ := 10
def total_paid : ℝ := 45

-- Theorem statement to prove the discount
theorem discount_each_book (d : ℝ) 
  (h1 : original_price * (num_books : ℝ) - d * (num_books : ℝ) = total_paid) : 
  d = 0.5 := 
sorry

end discount_each_book_l399_399472


namespace original_group_size_l399_399668

theorem original_group_size (x : ℕ) (h1 : ∀ x, (∃ y, y = x - 5 ∧ 12 * (y * 10 / x) = 10 * ((y * 12 + 5) / x))) : x = 25 :=
by
  use 25
  sorry

end original_group_size_l399_399668


namespace symmetric_graph_inverse_l399_399861

theorem symmetric_graph_inverse :
  (∀ x : ℝ, f (e^(x + 1)) = x) →
  f = λ x, ln x - 1 :=
by
  sorry

end symmetric_graph_inverse_l399_399861


namespace kiara_needs_7_containers_l399_399892

theorem kiara_needs_7_containers (h₁ : 50 > 0) (h₂ : 75 > 0) (h₃ : 36 > 0) : 
  let GCD := Nat.gcd (Nat.gcd 50 75) 36 in
  GCD = 5 ∧ 
  ((50 / GCD = 10) ∧ 
   (75 / GCD = 15) ∧ 
   (36 / GCD = 7)) →
  7 = min (50 / GCD) (min (75 / GCD) (36 / GCD)) :=
by
  sorry

end kiara_needs_7_containers_l399_399892


namespace domain_intersection_l399_399531

open Set Real

def domain_sqrt (f : ℝ → ℝ) : Set ℝ := {x | 4 - x^2 ≥ 0}
def domain_ln (g : ℝ → ℝ) : Set ℝ := {x | 1 - x > 0}

theorem domain_intersection : (domain_sqrt (λ x, sqrt (4 - x^2)) ∩ domain_ln (λ x, log (1 - x))) = Icc (-2 : ℝ) 2 ∩ Iio 1 :=
by
  sorry

end domain_intersection_l399_399531


namespace cyclic_points_of_altitudes_in_circle_l399_399513

theorem cyclic_points_of_altitudes_in_circle
  (ABC : Triangle)
  (h_acute : is_acute ABC)
  (K L : Point)
  (h_alt_B : altitude_from B ABC ∩ circle_with_diameter AC = {K, L})
  (M N : Point)
  (h_alt_C : altitude_from C ABC ∩ circle_with_diameter AB = {M, N})
  : are_cyclic K L M N := 
sorry

end cyclic_points_of_altitudes_in_circle_l399_399513


namespace xiao_hong_home_due_north_of_xiao_liang_home_l399_399262

-- Definitions for the conditions
structure Location where
  x : ℝ
  y : ℝ

def tv_tower : Location := { x := 0, y := 0 }

def xiao_hong_home : Location := { x := -200 / (2 ^ (1 / 2)), y := 200 / (2 ^ (1 / 2)) }
def xiao_liang_home : Location := { x := -200 / (2 ^ (1 / 2)), y := -200 / (2 ^ (1 / 2)) }

-- The main theorem which we need to prove
theorem xiao_hong_home_due_north_of_xiao_liang_home :
  xiao_hong_home.x = xiao_liang_home.x ∧ xiao_hong_home.y > xiao_liang_home.y :=
by
  -- Proof to be filled in later
  sorry

end xiao_hong_home_due_north_of_xiao_liang_home_l399_399262


namespace g_neg2_undefined_l399_399706

def g (x : ℝ) : ℝ := (x - 3) / (x + 2)

theorem g_neg2_undefined : ¬∃ y : ℝ, y = g (-2) := by
  intro h
  cases h with y hy
  unfold g at hy
  rw [←hy] at hy
  sorry

end g_neg2_undefined_l399_399706


namespace find_other_number_l399_399052

theorem find_other_number (x y : ℤ) (h1 : 3 * x + 2 * y = 145) (h2 : x = 35 ∨ y = 35) : y = 20 :=
sorry

end find_other_number_l399_399052


namespace sum_of_digits_largest_product_of_primes_digit_sum_19_l399_399102

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit_prime_less_than_30 (p : ℕ) : Prop := 
  is_prime p ∧ (10 ≤ p) ∧ (p < 30)

def is_valid_primes (d e : ℕ) : Prop := 
  is_two_digit_prime_less_than_30 d ∧
  is_two_digit_prime_less_than_30 e ∧ 
  d ≠ e ∧
  is_prime (100 * d + e)

theorem sum_of_digits_largest_product_of_primes_digit_sum_19 :
  ∃ n : ℕ, 
    (∃ d e : ℕ, is_valid_primes d e ∧ n = d * e * (100 * d + e)) ∧
    (∀ n' : ℕ, 
      (∃ d' e' : ℕ, is_valid_primes d' e' ∧ n' = d' * e' * (100 * d' + e')) → n' ≤ n) ∧
    (n.digits.sum = 19) := 
sorry

end sum_of_digits_largest_product_of_primes_digit_sum_19_l399_399102


namespace find_AX_l399_399932

-- Defining points A, B, C, D, and X
variable (A B C D X : ℝ)

-- Defining the circle with diameter 1
axiom circle : metric.sphere 0 0.5 (A, B, C, D)

-- X lying on diameter AD
axiom on_diameter : X ∈ segment A D

-- BX = CX
axiom BX_eq_CX : dist B X = dist C X

-- Angle conditions
axiom angle_cond_1 : 3 * ∠ B A C = 36
axiom angle_cond_2 : ∠ B X C = 36

-- Prove AX = cos 6° * sin 12° * csc 18°
theorem find_AX : dist A X = Real.cos (6 * π / 180) * Real.sin (12 * π / 180) * Real.csc (18 * π / 180) := sorry

end find_AX_l399_399932


namespace numberOfBooks_correct_l399_399662

variable (totalWeight : ℕ) (weightPerBook : ℕ)

def numberOfBooks (totalWeight weightPerBook : ℕ) : ℕ :=
  totalWeight / weightPerBook

theorem numberOfBooks_correct (h1 : totalWeight = 42) (h2 : weightPerBook = 3) :
  numberOfBooks totalWeight weightPerBook = 14 := by
  sorry

end numberOfBooks_correct_l399_399662


namespace truck_travel_distance_l399_399308

theorem truck_travel_distance (b t : ℝ) (ht : t > 0) (ht30 : t + 30 > 0) : 
  let converted_feet := 4 * 60
  let time_half := converted_feet / 2
  let speed_first_half := b / 4
  let speed_second_half := b / 4
  let distance_first_half := speed_first_half * time_half / t
  let distance_second_half := speed_second_half * time_half / (t + 30)
  let total_distance_feet := distance_first_half + distance_second_half
  let result_yards := total_distance_feet / 3
  result_yards = (10 * b / t) + (10 * b / (t + 30))
:= by
  -- proof skipped
  sorry

end truck_travel_distance_l399_399308


namespace pencils_brought_l399_399693

-- Given conditions
variables (A B : ℕ)

-- There are 7 people in total
def total_people : Prop := A + B = 7

-- 11 charts in total
def total_charts : Prop := A + 2 * B = 11

-- Question: Total pencils
def total_pencils : ℕ := 2 * A + B

-- Statement to be proved
theorem pencils_brought
  (h1 : total_people A B)
  (h2 : total_charts A B) :
  total_pencils A B = 10 := by
  sorry

end pencils_brought_l399_399693


namespace complementary_angles_ratio_decrease_l399_399203

theorem complementary_angles_ratio_decrease 
  (a b : ℝ) (h_ratio : a / b = 3 / 7) (h_comp : a + b = 90) : 
  let a' := a * 1.20 in
  let b' := 90 - a' in
  ((b' / b) * 100 = 91.43) ∧ (100 - (b' / b) * 100 = 8.57) :=
by
  have : a / b = 3 / 7 := h_ratio
  have : a + b = 90 := h_comp
  let a' := a * 1.20
  let b' := 90 - a'
  have : b' / b = 57.6 / 63 := sorry -- Calculations omitted for brevity
  have : (b' / b) * 100 = 91.43 := by sorry
  have : 100 - (b' / b) * 100 = 8.57 := by sorry
  split;
  assumption

end complementary_angles_ratio_decrease_l399_399203


namespace alpha_beta_store_ways_l399_399317

open Finset

theorem alpha_beta_store_ways:
  let oreo_flavors := 5
  let milk_flavors := 3
  let total_products := oreo_flavors + milk_flavors
  let alpha_choices (alpha_items: finset (fin total_products)) := alpha_items.card ≤ 3
  let beta_choices (beta_items: multiset (fin oreo_flavors)) := beta_items.card ≤ 3
  let total_ways := 
    (card (powerset_len 3 (range total_products))) + 
    (card (powerset_len 2 (range total_products)) * card (multiset.range oreo_flavors)) + 
    (card (powerset_len 1 (range total_products)) * (card (powerset_len 2 (range oreo_flavors)) + oreo_flavors)) + 
    (card (multiset.powerset_len 3 (range oreo_flavors)))
  in total_ways = 351 :=
by
  let oreo_flavors := 5
  let milk_flavors := 3
  let total_products := oreo_flavors + milk_flavors
  
  -- Define choices for Alpha and Beta, and their respective constraints
  let alpha_choices (alpha_items : finset (fin total_products)) := alpha_items.card ≤ 3
  let beta_choices (beta_items : multiset (fin oreo_flavors)) := beta_items.card ≤ 3
  
  -- Calculate total ways
  let total_ways :=
    (card (powerset_len 3 (range total_products))) + 
    (card (powerset_len 2 (range total_products)) * card (multiset.range oreo_flavors)) + 
    (card (powerset_len 1 (range total_products)) * (card (powerset_len 2 (range oreo_flavors)) + oreo_flavors)) + 
    (card (multiset.powerset_len 3 (range oreo_flavors)))
  
  -- Assert the final number of ways
  show total_ways = 351, from sorry

end alpha_beta_store_ways_l399_399317


namespace monotonicity_and_no_real_roots_l399_399089

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

theorem monotonicity_and_no_real_roots 
  (a : ℝ) (ha : 0 < a) : 
  (∀ x : ℝ, (0 < x ∧ x < (1 / a) → deriv (f a) x < 0) ∧ (x > (1 / a) → deriv (f a) x > 0)) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≠ 0 → a > (1 / real.exp 1))) :=
begin
  sorry
end

end monotonicity_and_no_real_roots_l399_399089


namespace part1_k_range_part2_sum_reciprocal_of_roots_l399_399863

theorem part1_k_range (k : ℝ) 
  (has_distinct_real_roots : ∀ (k : ℝ), ∃ (a b c : ℝ), a = k ∧ b = k - 2 ∧ c = k / 4 ∧ (b^2 - 4 * a * c > 0)) : 
  k ∈ Set.Iio 1 ∧ k ≠ 0 := sorry

theorem part2_sum_reciprocal_of_roots (k : ℝ)
  (sum_reciprocal_of_roots : ∀ (k : ℝ), ∃ (a b c : ℝ), a = k ∧ b = k - 2 ∧ c = k / 4 ∧ (b^2 - 4 * a * c > 0) → (∃ x1 x2 : ℝ, x1 + x2 = -(b / a) ∧ x1 * x2 = (c / a) ∧ (1 / x1 + 1 / x2 = 0))) : 
  ¬ (∃ k : ℝ, k ∈ Set.Iio 1 ∧ k ≠ 0 ∧ (1 / (let x1, x2 in x1 + x2 = -(b / a) ∧ x1 * x2 = (c / a)) = 0)) :=
sorry

end part1_k_range_part2_sum_reciprocal_of_roots_l399_399863


namespace math_problem_l399_399899

theorem math_problem 
  (a b c : ℝ)
  (h1 : a < b)
  (h2 : ∀ x, (x < -2 ∨ |x - 30| ≤ 2) ↔ ( (x - a) * (x - b) / (x - c) ≤ 0 )) :
  a + 2 * b + 3 * c = 86 :=
sorry

end math_problem_l399_399899


namespace monotonicity_and_range_l399_399085

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end monotonicity_and_range_l399_399085


namespace root_interval_l399_399572

-- Define the function f
def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

-- Conditions
lemma f_two_neg : f 2 < 0 := 
  by
    sorry

lemma f_three_pos : f 3 > 0 := 
  by
    sorry

-- Theorem statement
theorem root_interval : ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 :=
  by
    sorry

end root_interval_l399_399572


namespace total_students_left_l399_399986

def initial_boys : Nat := 14
def initial_girls : Nat := 10
def boys_dropout : Nat := 4
def girls_dropout : Nat := 3

def boys_left : Nat := initial_boys - boys_dropout
def girls_left : Nat := initial_girls - girls_dropout

theorem total_students_left : boys_left + girls_left = 17 :=
by 
  sorry

end total_students_left_l399_399986


namespace average_sales_is_5750_l399_399667

-- Define the monthly sales figures
def month1_sales : ℤ := 5266
def month2_sales : ℤ := 5744
def month3_sales : ℤ := 5864
def month4_sales : ℤ := 6122
def month5_sales : ℤ := 6588
def month6_sales : ℤ := 4916

-- Define the total sales over the six months
def total_sales : ℤ := month1_sales + month2_sales + month3_sales + month4_sales + month5_sales + month6_sales

-- Define the average sales over the six months
def average_sales : ℤ := total_sales / 6

-- Prove that the average sale is Rs. 5750
theorem average_sales_is_5750 : average_sales = 5750 :=
by
  -- Compute the total sales
  have h1 : total_sales = 34500 := by sorry
  rw h1
  
  -- Compute the average sale
  have h2 : average_sales = total_sales / 6 := by sorry
  rw h2
  
  -- Simplify final result
  exact sorry

end average_sales_is_5750_l399_399667


namespace problem_statement_l399_399799

noncomputable def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  a 2 = 4 ∧ 2 * a 4 - a 5 = 7

noncomputable def geometric_seq (b : ℕ → ℤ) : Prop :=
  b 3 = 4 ∧ b 4 + b 5 = 8 * (b 1 + b 2) ∧ ∀ n, b n ≠ -1

noncomputable def gen_formula_a (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 3 * n - 2

noncomputable def gen_formula_b (b : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, b n = 2 ^ (n - 1)

noncomputable def sequence_c (a : ℕ → ℤ) (c : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, c n = 3 / ((a n) * (a (n + 1)))

noncomputable def sum_c_n (c : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = (∑ k in Finset.range n, c k) = 3 * n / (3 * n + 1)

theorem problem_statement (a b : ℕ → ℤ) (c : ℕ → ℚ) (S : ℕ → ℚ) :
  arithmetic_seq a →
  geometric_seq b →
  gen_formula_a a →
  gen_formula_b b →
  sequence_c a c →
  sum_c_n c S :=
by
  sorry

end problem_statement_l399_399799


namespace knights_on_red_chairs_l399_399999

variable (K L Kr Lb : ℕ)
variable (h1 : K + L = 20)
variable (h2 : Kr + Lb = 10)
variable (h3 : Kr = L - Lb)

/-- Given the conditions:
1. There are 20 seats with knights and liars such that K + L = 20.
2. Half of the individuals claim to be sitting on blue chairs, and half on red chairs such that Kr + Lb = 10.
3. Knights on red chairs (Kr) must be equal to liars minus liars on blue chairs (Lb).
Prove that the number of knights now sitting on red chairs is 5. -/
theorem knights_on_red_chairs : Kr = 5 :=
by
  sorry

end knights_on_red_chairs_l399_399999


namespace original_price_l399_399374

theorem original_price (spent : ℝ) (less_percentage : ℝ) (original_price : ℝ) : 
  spent = 7500 → less_percentage = 0.25 → original_price = 10000 := 
begin
  intros h1 h2,
  have h3 : spent = (1 - less_percentage) * original_price,
  { rw [h1, h2], ring },
  have h4 : 7500 = 0.75 * original_price,
  { rw h3 },
  have h5 : original_price = 7500 / 0.75,
  { linarith },
  rw div_eq_mul_inv at h5,
  simp at h5,
  exact h5,
  sorry
end

end original_price_l399_399374


namespace slope_of_vertical_line_l399_399936

theorem slope_of_vertical_line (l : affine_plane.line) (h : l.equation = (λ p : affine_plane.point, p.x = -1)) :
  ¬∃ m : ℝ, is_slope l m :=
by
  sorry

end slope_of_vertical_line_l399_399936


namespace distance_between_vertices_l399_399743

theorem distance_between_vertices (a b : ℝ) (a_pos : a = real.sqrt 144) (h : ∀ x y, x^2 / 144 - y^2 / 49 = 1): (2 * a) = 24 := by
  have ha : a = 12 := by sorry
  have h2a : 2 * a = 24 := by linarith
  exact h2a

end distance_between_vertices_l399_399743


namespace max_vec_diff_magnitude_l399_399449

open Real

noncomputable def vec_a (θ : ℝ) : ℝ × ℝ := (1, sin θ)
noncomputable def vec_b (θ : ℝ) : ℝ × ℝ := (1, cos θ)

noncomputable def vec_diff_magnitude (θ : ℝ) : ℝ :=
  let a := vec_a θ
  let b := vec_b θ
  abs ((a.1 - b.1)^2 + (a.2 - b.2)^2)^(1/2)

theorem max_vec_diff_magnitude : ∀ θ : ℝ, vec_diff_magnitude θ ≤ sqrt 2 :=
by
  intro θ
  sorry

end max_vec_diff_magnitude_l399_399449


namespace point_A_moves_to_vertex_3_l399_399170

-- Definition of the problem conditions
inductive Face
| green
| far_white
| right_bottom_white
| new_green
| new_far_white
| new_left_upper_white

def initial_vertex_A : Face → bool
| Face.green := true
| Face.far_white := true
| Face.right_bottom_white := true
| _ := false

def rotated_vertex_A : Face → bool
| Face.new_green := true
| Face.new_far_white := true
| Face.new_left_upper_white := true
| _ := false

-- Theorem statement to prove that point A moves to vertex number 3
theorem point_A_moves_to_vertex_3 : rotated_vertex_A = (λ f, f = Face.new_green ∨ f = Face.new_far_white ∨ f = Face.new_left_upper_white) :=
by
  sorry

end point_A_moves_to_vertex_3_l399_399170


namespace probability_multiple_of_3_or_4_l399_399179

theorem probability_multiple_of_3_or_4 :
  let n := 30 in
  let multiples_of_3 := { k | k ∈ finset.range (n + 1) ∧ k % 3 = 0 } in
  let multiples_of_4 := { k | k ∈ finset.range (n + 1) ∧ k % 4 = 0 } in
  let favorable := multiples_of_3 ∪ multiples_of_4 in
  let total := finset.range (n + 1) in
  (favorable.card : ℚ) / total.card = 1 / 2 :=
by
  sorry

end probability_multiple_of_3_or_4_l399_399179


namespace cos_2x_value_find_f_B_range_l399_399842

/-
Define vectors, function and conditions
-/

def vector_m (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, -1)
def vector_n (x : ℝ) : ℝ × ℝ := (sin x, cos x ^ 2)
def f (x : ℝ) : ℝ := (vector_m x).1 * (vector_n x).1 + (vector_m x).2 * (vector_n x).2 + 1/2

/-
Proof of part 1
-/
theorem cos_2x_value (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ π / 4) (h₃ : f x = sqrt 3 / 3) 
  : cos (2 * x) = sqrt 2 / 2 - sqrt 3 / 6 := by 
sorry

/-
Triangle condition definition
-/
structure triangle :=
  (a b c : ℝ)
  (angle_A angle_B angle_C : ℝ)
  (triangle_ineq : 2 * b * cos angle_A ≤ 2 * c - sqrt 3 * a)

/-
Proof of part 2
-/
noncomputable def f_ang (B : ℝ) : ℝ := sin (2 * B - π / 6)

theorem find_f_B_range (T : triangle) (h : T.triangle_ineq) 
  (B_pos : 0 < T.angle_B) (B_lt_pi_3 : T.angle_B ≤ π / 6)
  : f_ang T.angle_B ∈ set.Ioc (-1/2 : ℝ) (1/2 : ℝ) := by 
sorry

end cos_2x_value_find_f_B_range_l399_399842


namespace complex_projective_form_and_fixed_points_l399_399938

noncomputable def complex_projective_transformation (a b c d : ℂ) (z : ℂ) : ℂ :=
  (a * z + b) / (c * z + d)

theorem complex_projective_form_and_fixed_points (a b c d : ℂ) (h : d ≠ 0) :
  (∃ (f : ℂ → ℂ), ∀ z, f z = complex_projective_transformation a b c d z)
  ∧ ∃ (z₁ z₂ : ℂ), complex_projective_transformation a b c d z₁ = z₁ ∧ complex_projective_transformation a b c d z₂ = z₂ :=
by
  -- omitted proof, this is just the statement
  sorry

end complex_projective_form_and_fixed_points_l399_399938


namespace odd_function_b_value_f_monotonically_increasing_l399_399820

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.log (Real.sqrt (4 * x^2 + b) + 2 * x)

-- part (1): Prove that if y = f(x) is an odd function, then b = 1
theorem odd_function_b_value :
  (∀ x : ℝ, f x b + f (-x) b = 0) → b = 1 := sorry

-- part (2): Prove that y = f(x) is monotonically increasing for all x in ℝ given b = 1
theorem f_monotonically_increasing (b : ℝ) :
  b = 1 → ∀ x1 x2 : ℝ, x1 < x2 → f x1 b < f x2 b := sorry

end odd_function_b_value_f_monotonically_increasing_l399_399820


namespace birds_not_hp_pk_35_percent_l399_399470

-- Assuming there are 100 birds in the nature reserve
def total_birds := 100

-- 30 percent of the birds are hawks
def hawk_percent := 0.30
def hawks := total_birds * hawk_percent

-- Remaining birds are non-hawks
def non_hawks := total_birds - hawks

-- 40 percent of non-hawks are paddyfield-warblers
def paddyfield_warbler_percent := 0.40
def paddyfield_warblers := non_hawks * paddyfield_warbler_percent

-- 25 percent as many kingfishers as paddyfield-warblers
def kingfisher_percent := 0.25
def kingfishers := paddyfield_warblers * kingfisher_percent

-- Total number of hawks, paddyfield-warblers, and kingfishers
def total_hp_pk := hawks + paddyfield_warblers + kingfishers

-- The number of birds not in the categories of hawks, paddyfield-warblers, or kingfishers
def non_hp_pk := total_birds - total_hp_pk

-- Calculate the percentage of these birds
def non_hp_pk_percent := non_hp_pk / total_birds * 100

theorem birds_not_hp_pk_35_percent : non_hp_pk_percent = 35 := by
  sorry

end birds_not_hp_pk_35_percent_l399_399470


namespace unique_intersection_condition_l399_399972

theorem unique_intersection_condition
  (a c b d : ℝ)
  (h_sym_central : ∀ x y : ℝ, (y = 2a + 1/(x-b) ∧ y = 2c + 1/(x-d) → (x, y) = (1/2*(b+d), a+c)))
  (h_common_point : 2a + 1/(1/2*(b+d) - b) = 2c + 1/(1/2*(b+d) - d) ∧ 2a + 1/(1/2*(b+d) - b) = a + c) :
  (a - c) * (b - d) = 2 :=
by
  -- Proof goes here
  sorry

end unique_intersection_condition_l399_399972


namespace probability_of_first_four_cards_each_suit_l399_399008

noncomputable def probability_first_four_different_suits : ℚ := 3 / 32

theorem probability_of_first_four_cards_each_suit :
  let n := 52
  let k := 5
  let suits := 4
  (probability_first_four_different_suits = (3 / 32)) :=
by
  sorry

end probability_of_first_four_cards_each_suit_l399_399008


namespace cats_in_shelter_l399_399504

-- Define the initial conditions
def initial_cats := 20
def monday_addition := 2
def tuesday_addition := 1
def wednesday_subtraction := 3 * 2

-- Problem statement: Prove that the total number of cats after all events is 17
theorem cats_in_shelter : initial_cats + monday_addition + tuesday_addition - wednesday_subtraction = 17 :=
by
  sorry

end cats_in_shelter_l399_399504


namespace tan_A_is_5_over_12_l399_399031

-- Defining the context of triangle ABC.
variables {A B C : Type} [AffineGeometry A] [AffineGeometry B] [AffineGeometry C]

-- Given conditions
def angle_C_is_90 (ABC : Triangle A B C) : Prop := ABC.angle C = 90
def AB_equals_13 (ABC : Triangle A B C) : Prop := ABC.side_length AB = 13
def BC_equals_5 (ABC : Triangle A B C) : Prop := ABC.side_length BC = 5

-- Concluding the result on tan A
theorem tan_A_is_5_over_12 (ABC : Triangle A B C)
  (h1 : angle_C_is_90 ABC)
  (h2 : AB_equals_13 ABC)
  (h3 : BC_equals_5 ABC) : 
  ABC.tan A = 5 / 12 := sorry

end tan_A_is_5_over_12_l399_399031


namespace largest_expr_l399_399512

noncomputable def A : ℝ := 2 * 1005 ^ 1006
noncomputable def B : ℝ := 1005 ^ 1006
noncomputable def C : ℝ := 1004 * 1005 ^ 1005
noncomputable def D : ℝ := 2 * 1005 ^ 1005
noncomputable def E : ℝ := 1005 ^ 1005
noncomputable def F : ℝ := 1005 ^ 1004

theorem largest_expr : A - B > B - C ∧ A - B > C - D ∧ A - B > D - E ∧ A - B > E - F :=
by
  sorry

end largest_expr_l399_399512


namespace total_distance_walked_l399_399843

noncomputable def hazel_total_distance : ℕ := 3

def distance_first_hour := 2  -- The distance traveled in the first hour (in kilometers)
def distance_second_hour := distance_first_hour * 2  -- The distance traveled in the second hour
def distance_third_hour := distance_second_hour / 2  -- The distance traveled in the third hour, with a 50% speed decrease

theorem total_distance_walked :
  distance_first_hour + distance_second_hour + distance_third_hour = 8 :=
  by
    sorry

end total_distance_walked_l399_399843


namespace find_n_l399_399007

theorem find_n (x n : ℝ) (h_x : x = 0.5) : (9 / (1 + n / x) = 1) → n = 4 := 
by
  intro h
  have h_x_eq : x = 0.5 := h_x
  -- Proof content here covering the intermediary steps
  sorry

end find_n_l399_399007


namespace rectangle_sides_l399_399220

theorem rectangle_sides (x y : ℝ) (h1 : 4 * x = 3 * y) (h2 : x * y = 2 * (x + y)) :
  (x = 7 / 2 ∧ y = 14 / 3) ∨ (x = 14 / 3 ∧ y = 7 / 2) :=
by {
  sorry
}

end rectangle_sides_l399_399220


namespace number_of_positive_integer_values_l399_399716

theorem number_of_positive_integer_values (N : ℕ) :
  ∃ (card_positive_N : ℕ), 
  (∀ (N : ℕ), (N > 0) → ∃ (d : ℕ), d ∣ 49 ∧ d > 3 ∧ N = d - 3)
  ∧ card_positive_N = 2 := 
sorry

end number_of_positive_integer_values_l399_399716


namespace true_statement_l399_399632

def complementary_angles_adjacent (α β : ℝ) : Prop := (α + β = 90) ∧ (adjacent α β)
def equal_vertical_angles (α β : ℝ) : Prop := (α = β) ∧ (vertical α β)
def square_root_of_4 : Prop := sqrt 4 = 2 ∧ sqrt 4 = -2
def exterior_angle_greater (α β γ δ : ℝ) : Prop := (δ = α + β) ∧ (γ > α) ∧ (γ > β)

theorem true_statement :
  ∃ (α β γ δ : ℝ), ¬(complementary_angles_adjacent α β) ∧
                   ¬(equal_vertical_angles α β) ∧
                   ¬(square_root_of_4) ∧
                   (exterior_angle_greater α β γ δ) :=
by
  sorry

end true_statement_l399_399632


namespace probability_three_digit_multiple_of_3_l399_399658

/-- A bag contains four pieces of paper, each labeled with one of four consecutive integers
randomly chosen from the numbers 1 through 6. Three of these pieces are drawn, one at a time
without replacement, to construct a three-digit number. What is the probability that the
three-digit number is a multiple of 3? -/
theorem probability_three_digit_multiple_of_3 :
  ∃ (bag : set (finset ℕ)) (n : ℕ) (drawn : finset ℕ),
    (bag = {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}}) →
    (drawn.card = 3) →
    ((∃ p ∈ bag, ∃ c ∈ p.powerset.filter (λ (s : finset ℕ), s.card = 3),
      c.sum % 3 = 0) → ↑(5 / 12) :=
sorry

end probability_three_digit_multiple_of_3_l399_399658


namespace probability_ending_at_multiple_of_3_l399_399889

-- Definitions based directly on the given conditions.

def first_spin_options : Finset ℤ := {−1, 1, 1, 2}
def multiple_of_3 (n: ℤ) : Prop := n % 3 = 0
def probability(p: ℚ): ℚ := p

-- Theorem statement
theorem probability_ending_at_multiple_of_3 :
  probability (
    (3 / 10) * (5 / 16) +  -- Starting point is a multiple of 3
    (4 / 10) * (1 / 4) +  -- Starting point is one more than a multiple of 3
    (3 / 10) * (1 / 4)    -- Starting point is one less than a multiple of 3
  ) = 43 / 160 :=
sorry

end probability_ending_at_multiple_of_3_l399_399889


namespace subsets_list_A_l399_399833

-- Define set A
def A := {1, 2}

-- Define the set of all subsets of A
def subsets_of_A := {∅, {1}, {2}, {1, 2}}

-- Theorem: subsets_of_A is the set of all subsets of A
theorem subsets_list_A : (subsets A) = subsets_of_A :=
by
  sorry -- proof goes here

end subsets_list_A_l399_399833


namespace max_value_expression_l399_399646

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∀ a b c > 0, a = b ∧ b = c → (a^2*(b+c) + b^2*(c+a) + c^2*(a+b)) / (a^3 + b^3 + c^3 - 2*a*b*c) ≤ 6) :=
begin
  sorry
end

end max_value_expression_l399_399646


namespace fewest_handshakes_is_zero_l399_399019

noncomputable def fewest_handshakes (n k : ℕ) : ℕ :=
  if h : (n * (n - 1)) / 2 + k = 325 then k else 325

theorem fewest_handshakes_is_zero :
  ∃ n k : ℕ, (n * (n - 1)) / 2 + k = 325 ∧ 0 = fewest_handshakes n k :=
by
  sorry

end fewest_handshakes_is_zero_l399_399019


namespace minimum_value_f_l399_399963

noncomputable def f (x : ℝ) : ℝ := (x^2 / 8) + x * (Real.cos x) + (Real.cos (2 * x))

theorem minimum_value_f : ∃ x : ℝ, f x = -1 :=
by {
  sorry
}

end minimum_value_f_l399_399963


namespace two_common_tangents_l399_399525

noncomputable theory

-- Definitions for circle, point, projection, and tangent relationships
variables {A B C D : Point}
variable {gamma gamma1 gamma2 gamma3 : Circle}
variables (O O1 O2 O3 : Point)
variables (r1 r2 r3 : ℝ)

-- Conditions
def is_diameter (A B : Point) (gamma : Circle) := ∃ O, O = midpoint A B ∧ center gamma = O
def on_circle (C : Point) (gamma : Circle) := C ∈ gamma
def is_projection (D : Point) (C : Point) (AB : Line) := orthogonal_projection AB C = D
def common_tangent (AB : Line) (gamma : Circle) := ∃ T, is_tangent AB T γ
def inscribed (gamma1 : Circle) (ABC : Triangle) := ∃ (O1: Point), incenter ABC = O1 ∧ gamma1 = inscribed_circle ABC
def tangent_to_segment_and_circle (gamma : Circle) (CD : Segment) (gamma1 : Circle) := tangent_to_segment gamma1 CD ∧ tangent_to_circle gamma1 gamma

-- Theorem Statement
theorem two_common_tangents
  (diam : is_diameter A B gamma)
  (C_on_gamma : C ≠ A ∧ C ≠ B ∧ on_circle C gamma)
  (D_proj : is_projection D C (line_through A B))
  (com_tangent1 : common_tangent (line_through A B) gamma1)
  (com_tangent2 : common_tangent (line_through A B) gamma2)
  (com_tangent3 : common_tangent (line_through A B) gamma3)
  (insc_gamma1 : inscribed gamma1 (triangle A B C))
  (tan_seg_cir_gamma2 : tangent_to_segment_and_circle gamma gamma2 (segment C D))
  (tan_seg_cir_gamma3 : tangent_to_segment_and_circle gamma gamma3 (segment C D)) :
  ∃ T1 T2 : Line, T1.is_tangent gamma1 ∧ T1.is_tangent gamma2 ∧ T1.is_tangent gamma3 ∧ 
               T2.is_tangent gamma1 ∧ T2.is_tangent gamma2 ∧ T2.is_tangent gamma3 :=
sorry

end two_common_tangents_l399_399525


namespace ratio_transformation_l399_399983

theorem ratio_transformation (x1 y1 x2 y2 : ℚ) (h₁ : x1 / y1 = 7 / 5) (h₂ : x2 = x1 * y1) (h₃ : y2 = y1 * x1) : x2 / y2 = 1 := by
  sorry

end ratio_transformation_l399_399983


namespace length_AE_equals_20_l399_399638

variable (A B C D E : Type)
variable [IsoscelesTrapezoid ABCE]
variable [Rectangle ACDE]
variable (AB EC AE : ℝ)
variable (AB10 : AB = 10)
variable (EC20 : EC = 20)

theorem length_AE_equals_20 :
  AE = 20 :=
by
  sorry

end length_AE_equals_20_l399_399638


namespace number_of_points_l399_399144

theorem number_of_points (initial_sum new_sum : ℝ) (decrease : ℝ) (n : ℕ) :
  initial_sum = -1.5 →
  new_sum = -15.5 →
  decrease = 2 * n →
  new_sum = initial_sum - decrease →
  n = 7 :=
by
  assume h_initial_sum h_new_sum h_decrease h_new_eq
  sorry

end number_of_points_l399_399144


namespace log_a_plus_b_eq_zero_l399_399903

open Complex

noncomputable def a_b_expression : ℂ := (⟨2, 1⟩ / ⟨1, 1⟩ : ℂ)

noncomputable def a : ℝ := a_b_expression.re

noncomputable def b : ℝ := a_b_expression.im

theorem log_a_plus_b_eq_zero : log (a + b) = 0 := by
  sorry

end log_a_plus_b_eq_zero_l399_399903


namespace numbers_equal_l399_399401

theorem numbers_equal (a b c d : ℕ)
  (h1 : (a + b)^2 % (c * d) = 0)
  (h2 : (a + c)^2 % (b * d) = 0)
  (h3 : (a + d)^2 % (b * c) = 0)
  (h4 : (b + c)^2 % (a * d) = 0)
  (h5 : (b + d)^2 % (a * c) = 0)
  (h6 : (c + d)^2 % (a * b) = 0) :
  a = b ∨ b = c ∨ c = d ∨ a = c ∨ a = d ∨ b = d ∨ (a = b ∧ b = c) ∨ (b = c ∧ c = d) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) :=
sorry

end numbers_equal_l399_399401


namespace range_of_a_l399_399443

noncomputable def function_with_extreme_at_zero_only (a b : ℝ) : Prop :=
∀ x : ℝ, x ≠ 0 → 4 * x^2 + 3 * a * x + 4 > 0

theorem range_of_a (a b : ℝ) (h : function_with_extreme_at_zero_only a b) : 
  -8 / 3 ≤ a ∧ a ≤ 8 / 3 :=
sorry

end range_of_a_l399_399443


namespace percentage_decrease_of_larger_angle_l399_399185

noncomputable def complementary_angles_decrease_percentage : Real :=
let total_degrees := 90
let ratio_sum := 3 + 7
let part := total_degrees / ratio_sum
let smaller_angle := 3 * part
let larger_angle := 7 * part
let increased_smaller_angle := smaller_angle * 1.2
let new_larger_angle := total_degrees - increased_smaller_angle
let decrease_amount := larger_angle - new_larger_angle
(decrease_amount / larger_angle) * 100

theorem percentage_decrease_of_larger_angle
  (smaller_increased_percentage : Real := 20)
  (ratio_three : Real := 3)
  (ratio_seven : Real := 7)
  (total_degrees : Real := 90)
  (expected_decrease : Real := 8.57):
  complementary_angles_decrease_percentage = expected_decrease := 
sorry

end percentage_decrease_of_larger_angle_l399_399185


namespace find_quadruples_l399_399379

-- Define the natural number tuples we are interested in
structure Quadruple where
  a b c d : Nat 

-- The main theorem statement
theorem find_quadruples (q : Quadruple) (h1 : q.a ≤ q.b) (h2 : q.b ≤ q.c) 
  (h3 : q.a.factorial + q.b.factorial + q.c.factorial = 3 ^ q.d) :
  q = ⟨1, 1, 1, 1⟩ ∨ q = ⟨1, 2, 3, 2⟩ ∨ q = ⟨1, 2, 4, 3⟩ :=
by
  sorry

end find_quadruples_l399_399379


namespace pandas_increase_l399_399270

theorem pandas_increase 
  (C P : ℕ) -- C: Number of cheetahs 5 years ago, P: Number of pandas 5 years ago
  (h_ratio_5_years_ago : C / P = 1 / 3)
  (h_cheetahs_increase : ∃ z : ℕ, z = 2)
  (h_ratio_now : ∃ k : ℕ, (C + k) / (P + x) = 1 / 3) :
  x = 6 :=
by
  sorry

end pandas_increase_l399_399270


namespace determine_m_l399_399436

noncomputable theory

open Classical

theorem determine_m (m : ℝ) :
  (∀ x: ℝ, x > 0 → (x^2 + (5 - 2 * m) * x + m - 3) / (x - 1) ≠ 2 * x + m) ↔ m = 3 :=
by sorry

end determine_m_l399_399436


namespace vector_magnitude_proof_l399_399406

noncomputable def vector_a : ℝ × ℝ × ℝ := (1, 1, 0)
noncomputable def vector_b : ℝ × ℝ × ℝ := (-1, 0, 2)

-- Function to calculate the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- The theorem statement
theorem vector_magnitude_proof : magnitude (2 • vector_a - vector_b) = real.sqrt 17 := by
  sorry

end vector_magnitude_proof_l399_399406


namespace a_eq_b_if_fraction_is_integer_l399_399384

theorem a_eq_b_if_fraction_is_integer (a b : ℕ) (h_pos_a : 1 ≤ a) (h_pos_b : 1 ≤ b) :
  ∃ k : ℕ, (a^4 + a^3 + 1) = k * (a^2 * b^2 + a * b^2 + 1) -> a = b :=
by
  sorry

end a_eq_b_if_fraction_is_integer_l399_399384


namespace sandy_marks_l399_399140

def correct_marks (correct sums : ℕ) : ℕ := 3 * correct sums
def incorrect_marks (incorrect sums : ℕ) : ℕ := 2 * incorrect sums

def total_marks (total sums correct sums : ℕ) : ℕ :=
  correct_marks correct sums - incorrect_marks (total sums - correct sums)

theorem sandy_marks (total_sums correct_sums : ℕ) (h_total : total_sums = 30) (h_correct : correct_sums = 21) :
  total_marks total_sums correct_sums = 45 :=
by
  rw [total_marks, h_total, h_correct]
  norm_num
  sorry

end sandy_marks_l399_399140


namespace problem_statement_l399_399002

-- Define the problem as a theorem in Lean
theorem problem_statement
  (x y : ℝ)
  (h : x / (1 + complex.i) = 2 - y * complex.i) :
  x - y = 2 := 
sorry

end problem_statement_l399_399002


namespace factorial_expression_l399_399621

theorem factorial_expression : (12! - 11!) / 10! = 121 := sorry

end factorial_expression_l399_399621


namespace hyperbola_vertex_distance_l399_399769

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  (x^2 / 144 - y^2 / 49 = 1) →
  ∃ (a : ℝ), a = 12 ∧ 2 * a = 24 :=
by
  intro x y h
  have h1 : 12^2 = 144 := by norm_num
  use 12
  split
  case left =>
    exact rfl
  case right =>
    calc
      2 * 12 = 24 : by norm_num

end hyperbola_vertex_distance_l399_399769


namespace farmer_farm_size_l399_399290

theorem farmer_farm_size 
  (sunflowers flax : ℕ)
  (h1 : flax = 80)
  (h2 : sunflowers = flax + 80) :
  (sunflowers + flax = 240) :=
by
  sorry

end farmer_farm_size_l399_399290


namespace find_a_l399_399801

theorem find_a (a : ℝ) : 
  let A := (a, 6)
  let line := 3 * a - 4 * 6 - 4
  let d := 4
  d = abs(line) / real.sqrt(3^2 + (-4)^2) → 
  a = 16 ∨ a = 8 / 3 :=
sorry

end find_a_l399_399801


namespace repeating_8_9_eq_9_l399_399733

-- Define 8.\overline{9} as an infinite sum.
def repeating_8_9 : ℝ := 8 + ∑' n : ℕ, 9 / 10^(n+1)

theorem repeating_8_9_eq_9 : repeating_8_9 = 9 := by
  sorry

end repeating_8_9_eq_9_l399_399733


namespace complementary_angles_ratio_decrease_l399_399205

theorem complementary_angles_ratio_decrease 
  (a b : ℝ) (h_ratio : a / b = 3 / 7) (h_comp : a + b = 90) : 
  let a' := a * 1.20 in
  let b' := 90 - a' in
  ((b' / b) * 100 = 91.43) ∧ (100 - (b' / b) * 100 = 8.57) :=
by
  have : a / b = 3 / 7 := h_ratio
  have : a + b = 90 := h_comp
  let a' := a * 1.20
  let b' := 90 - a'
  have : b' / b = 57.6 / 63 := sorry -- Calculations omitted for brevity
  have : (b' / b) * 100 = 91.43 := by sorry
  have : 100 - (b' / b) * 100 = 8.57 := by sorry
  split;
  assumption

end complementary_angles_ratio_decrease_l399_399205


namespace divisibility_equivalence_distinct_positive_l399_399108

variable (a b c : ℕ)

theorem divisibility_equivalence_distinct_positive (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c) ∣ (a^3 * b + b^3 * c + c^3 * a)) ↔ ((a + b + c) ∣ (a * b^3 + b * c^3 + c * a^3)) :=
by sorry

end divisibility_equivalence_distinct_positive_l399_399108


namespace line_parallel_or_within_plane_l399_399462

variables {l : Line} {π : Plane}

-- Let's assume that the angle between the line l and the plane π is given as 0° 
-- The goal is to prove that l is either parallel to π or l is within π.
theorem line_parallel_or_within_plane (h : angle l π = 0) : 
  (parallel l π) ∨ (within l π) :=
sorry

end line_parallel_or_within_plane_l399_399462


namespace max_value_inverse_powers_of_roots_l399_399371

noncomputable def polynomial_with_given_roots (t q : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 2 - Polynomial.C t * Polynomial.X + Polynomial.C q

theorem max_value_inverse_powers_of_roots (α β q : ℝ) (h_root_α : polynomial_with_given_roots 2 q.eval₂ α = 0)
    (h_root_β : polynomial_with_given_roots 2 q.eval₂ β = 0) :
    (∀ n : ℕ, n ≥ 1 → α ^ n + β ^ n = 2) →
    (1 / α ^ 2011 + 1 / β ^ 2011 = 2) :=
by
  sorry

end max_value_inverse_powers_of_roots_l399_399371


namespace distance_between_vertices_l399_399741

theorem distance_between_vertices (a b : ℝ) (a_pos : a = real.sqrt 144) (h : ∀ x y, x^2 / 144 - y^2 / 49 = 1): (2 * a) = 24 := by
  have ha : a = 12 := by sorry
  have h2a : 2 * a = 24 := by linarith
  exact h2a

end distance_between_vertices_l399_399741


namespace inequality_proof_l399_399811

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i) (h_prod : (∏ i, a i) = 1) :
  (∏ i, (2 + a i)) ≥ 3^n :=
sorry

end inequality_proof_l399_399811


namespace number_of_members_l399_399564

-- Define the conditions
def members : ℕ
def necklaces_per_member := 2
def beads_per_necklace := 50
def total_beads_needed := 900

-- Calculate the number of beads needed per member
def beads_per_member := necklaces_per_member * beads_per_necklace

-- Statement to prove
theorem number_of_members (h : total_beads_needed = members * beads_per_member) : members = 9 :=
by
  sorry

end number_of_members_l399_399564


namespace arithmetic_mean_reciprocals_primes_l399_399326

theorem arithmetic_mean_reciprocals_primes
  (p : Finset ℕ)
  (h_p : p = {2, 3, 5, 7})
  : (p.sum (λ x, 1 / ↑x) / 4) = (247 / 840) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l399_399326


namespace meet_at_starting_point_second_time_in_minutes_l399_399982

theorem meet_at_starting_point_second_time_in_minutes :
  let racing_magic_time := 60 -- in seconds
  let charging_bull_time := 3600 / 40 -- in seconds
  let lcm_time := Nat.lcm racing_magic_time charging_bull_time -- LCM of the round times in seconds
  let answer := lcm_time / 60 -- convert seconds to minutes
  answer = 3 :=
by
  sorry

end meet_at_starting_point_second_time_in_minutes_l399_399982


namespace greatest_integer_value_l399_399614

theorem greatest_integer_value (x : ℤ) : ∃ x, (∀ y, (x^2 + 2 * x + 10) % (x - 3) = 0 → x ≥ y) → x = 28 :=
by
  sorry

end greatest_integer_value_l399_399614


namespace count_valid_x_satisfying_heartsuit_condition_l399_399898

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem count_valid_x_satisfying_heartsuit_condition :
  (∃ n, ∀ x, 1 ≤ x ∧ x < 1000 → digit_sum (digit_sum x) = 4 → n = 36) :=
by
  sorry

end count_valid_x_satisfying_heartsuit_condition_l399_399898


namespace find_a3_l399_399415

-- Define the arithmetic sequences and conditions
variables {a : ℕ → ℝ} -- Sequence {a_n}
variables {S : ℕ → ℝ} -- Sequence {S_n} sum of first n terms of a_n
variables (d : ℝ) -- Common difference

-- Sum of first n terms of a_n
def S_n (n : ℕ) := ∑ i in finset.range n.succ, a i 

-- Define the arithmetic sequence properties
axiom h1 : ∀ n, a (n + 1) = a n + d   -- Sequence a_n is arithmetic with common difference d
axiom h2 : ∀ n, sqrt (S_n n) = sqrt (S_n 0) + n * d  -- Sequence sqrt(S_n) is arithmetic with the same common difference d

-- Separate conditions extracted from the given problem
axiom h3 : a 0 > 0  -- The sequence {a_n} is positive

-- Target proof
theorem find_a3 : 
  a 3 = 5 / 4 :=
sorry

end find_a3_l399_399415


namespace odd_function_f_value_l399_399431

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^3 + x + 1 else x^3 + x - 1

theorem odd_function_f_value : 
  f 2 = 9 := by
  sorry

end odd_function_f_value_l399_399431


namespace sector_area_120_deg_radius_3_l399_399427

theorem sector_area_120_deg_radius_3 (r : ℝ) (theta_deg : ℝ) (theta_rad : ℝ) (A : ℝ)
  (h1 : r = 3)
  (h2 : theta_deg = 120)
  (h3 : theta_rad = (2 * Real.pi / 3))
  (h4 : A = (1 / 2) * theta_rad * r^2) :
  A = 3 * Real.pi :=
  sorry

end sector_area_120_deg_radius_3_l399_399427


namespace probability_multiple_of_3_or_4_l399_399180

theorem probability_multiple_of_3_or_4 :
  let n := 30 in
  let multiples_of_3 := { k | k ∈ finset.range (n + 1) ∧ k % 3 = 0 } in
  let multiples_of_4 := { k | k ∈ finset.range (n + 1) ∧ k % 4 = 0 } in
  let favorable := multiples_of_3 ∪ multiples_of_4 in
  let total := finset.range (n + 1) in
  (favorable.card : ℚ) / total.card = 1 / 2 :=
by
  sorry

end probability_multiple_of_3_or_4_l399_399180


namespace red_paint_needed_l399_399124

theorem red_paint_needed (ratio_red_white : ℚ) (total_cans : ℕ) 
  (h_ratio : ratio_red_white = 4/3) (h_total : total_cans = 35) : 
  ⌊(4 / (4 + 3)) * 35⌋ = 20 :=
by 
sorry

end red_paint_needed_l399_399124


namespace distance_between_vertices_l399_399755

def hyperbola_eq (x y : ℝ) : Prop := 
  x^2 / 144 - y^2 / 49 = 1

theorem distance_between_vertices : 2 * real.sqrt 144 = 24 :=
by {
    -- use sqrt calculation for clarity
    have h : real.sqrt 144 = 12, by {
        exact real.sqrt_eq_iff_sq_eq.mpr (or.inl (by norm_num)),
    },
    rw [h],
    norm_num
}

end distance_between_vertices_l399_399755


namespace arithmetic_to_geometric_l399_399652

variable {a1 a2 a3 a4 d : ℤ}

def arithmetic_sequence (a1 a2 a3 a4 d : ℤ) : Prop :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d

def is_geometric_sequence (x y z: ℤ) : Prop :=
  y * y = x * z

theorem arithmetic_to_geometric (h1 : d ≠ 0)
  (h2 : arithmetic_sequence a1 a2 a3 a4 d)
  (h3 : is_geometric_sequence a1 a2 a3 ∨ is_geometric_sequence a1 a2 a4 ∨ is_geometric_sequence a1 a3 a4 ∨ is_geometric_sequence a2 a3 a4) :
  (\(\frac {a1} {d}\)) = 1 ∨ (\(\frac {a1} {d}\)) = -4 :=
sorry

end arithmetic_to_geometric_l399_399652


namespace Dima_unusual_dice_result_l399_399642

-- Conditions
def even_prob := 2 * odd_prob
def total_prob := odd_prob * 3 + even_prob * 3 = 1

-- Required Calculation
def prob (x : ℕ) : ℚ := if x % 2 = 1 then odd_prob else even_prob

def sum_prob_le_3 : ℚ := prob 1 + prob 2 + prob 3

-- Theorem Statement
theorem Dima_unusual_dice_result : sum_prob_le_3 = 13 := sorry

end Dima_unusual_dice_result_l399_399642


namespace least_integer_j_l399_399991

theorem least_integer_j
  (b : ℕ → ℝ)
  (h₁ : b 1 = 1)
  (h₂ : ∀ n ≥ 1, 2^(b (n + 1) - b n) - 1 = 1 / (n + 3 / 4)) : 
  ∃ j, j > 1 ∧ b j ∈ ℤ ∧ (∀ k > 1, b k ∈ ℤ → k ≥ j) :=
begin
  sorry
end

end least_integer_j_l399_399991


namespace magnitude_of_vector_l399_399407

def a : ℝ × ℝ × ℝ := (1, 1, 0)
def b : ℝ × ℝ × ℝ := (-1, 0, 2)

theorem magnitude_of_vector :
  ‖(2 • a.1 - b.1, 2 • a.2 - b.2, 2 • a.3 - b.3)‖ = Real.sqrt 17 := by
  sorry

end magnitude_of_vector_l399_399407


namespace jane_wins_probability_l399_399887

theorem jane_wins_probability : 
  let spins := {1, 2, 3, 4, 5, 6}
  in ∃ (jane_spin brother_spin : ℕ), jane_spin ∈ spins ∧ brother_spin ∈ spins ∧ 
  (abs (jane_spin - brother_spin) < 4) / 36 = 5 / 6 := 
sorry

end jane_wins_probability_l399_399887


namespace bouquet_cost_l399_399320

theorem bouquet_cost (c₁ : ℕ) (r₁ r₂ : ℕ) (c_discount : ℕ) (discount_percentage: ℕ) :
  (c₁ = 30) → (r₁ = 15) → (r₂ = 45) → (c_discount = 81) → (discount_percentage = 10) → 
  ((c₂ : ℕ) → (c₂ = (c₁ * r₂) / r₁) → (r₂ > 30) → 
  (c_discount = c₂ - (c₂ * discount_percentage / 100))) → 
  c_discount = 81 :=
by
  intros h1 h2 h3 h4 h5
  subst_vars
  sorry

end bouquet_cost_l399_399320


namespace find_AD_l399_399319

noncomputable def AD (A B C D : Point) (AB BC CA : ℝ) (h1 : dist A B = 9) (h2 : dist B C = 8) (h3 : dist C A = 7) : ℝ :=
  let O1 := circle A B
  let O2 := circle A C
  let E := midpoint B C
  (dist A E) - (dist E D) = 33 / 7

theorem find_AD
  (A B C D : Point) 
  (AB BC CA : ℝ) 
  (h1 : dist A B = 9) 
  (h2 : dist B C = 8) 
  (h3 : dist C A = 7)
  (O1 : Circle A B)
  (O2 : Circle A C)
  (h4 : tangent O1 B C)
  (h5 : tangent O2 C A)
  (h6 : distinct_point (intersect O1 O2) ≠ A)
  : AD A B C D AB BC CA h1 h2 h3 = 33 / 7 :=
    by
    sorry

end find_AD_l399_399319


namespace roses_and_orchids_difference_l399_399597

theorem roses_and_orchids_difference :
  let roses_now := 11
  let orchids_now := 20
  orchids_now - roses_now = 9 := 
by
  sorry

end roses_and_orchids_difference_l399_399597


namespace pass_through_triangle_hole_l399_399553

theorem pass_through_triangle_hole
  (T Q : Triangle) 
  (a b c : ℝ) 
  (h_areaQ : Q.area = 3)
  (h_sidesQ : Q.side_lengths = [a, b, c] ∧ a ≤ b ∧ b ≤ c)
  (h_angle_small : Q.smallest_angle ≤ 60)
  (h_areaT : T.area < 4)
  (h_rigidT : T.rigid_planar) :
  can_pass_through T Q := 
sorry

end pass_through_triangle_hole_l399_399553


namespace circle_intersection_l399_399845

theorem circle_intersection :
  let x1 := 3 / 2
  let y1 := 0
  let r1 := 3 / 2
  let x2 := 0
  let y2 := 5 / 2
  let r2 := 5 / 2
  (x1, y1, r1) ≠ (x2, y2, r2) → 
  (∃ P1 P2 P3 P4, (P1, P2, P3, P4) ∈ intersection_circles x1 y1 r1 x2 y2 r2) :=
sorry

end circle_intersection_l399_399845


namespace fourth_metal_mass_l399_399296

noncomputable def alloy_mass_problem (m1 m2 m3 m4 : ℝ) : Prop :=
  m1 + m2 + m3 + m4 = 20 ∧
  m1 = 1.5 * m2 ∧
  m2 = (3 / 4) * m3 ∧
  m3 = (5 / 6) * m4

theorem fourth_metal_mass :
  ∃ (m4 : ℝ), (∃ (m1 m2 m3 : ℝ), alloy_mass_problem m1 m2 m3 m4) ∧ abs (m4 - 5.89) < 0.01 :=
begin
  sorry  -- Proof is skipped
end

end fourth_metal_mass_l399_399296


namespace washing_time_is_45_l399_399352

-- Definitions based on conditions
variables (x : ℕ) -- time to wash one load
axiom h1 : 2 * x + 75 = 165 -- total laundry time equation

-- The statement to prove: washing one load takes 45 minutes
theorem washing_time_is_45 : x = 45 :=
by
  sorry

end washing_time_is_45_l399_399352


namespace monotonicity_and_no_x_intercept_l399_399099

noncomputable theory

def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

def is_monotonic (a : ℝ) (x : ℝ) : Prop := 
  if x < 1 / a then 
    f a x > f a (x + ε) -- ε is a small positive value
  else if x > 1 / a then 
    f a x < f a (x - ε)
  else
    true -- At x = 1/a, the function transits from decreasing to increasing

theorem monotonicity_and_no_x_intercept 
  (a : ℝ) (h1 : 0 < a) : 
  (∀ x : ℝ, 0 < x → is_monotonic a x) ∧ 
  (∀ x : ℝ, f a x ≠ 0) ↔ 
  (a ∈ Ioi (1 / real.exp 1)) := 
sorry

end monotonicity_and_no_x_intercept_l399_399099


namespace solution_l399_399363

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)

axiom periodic_f : ∀ x : ℝ, f (x - 3) = - f x

axiom increasing_f_on_interval : ∀ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ 3 ∧ 0 ≤ x2 ∧ x2 ≤ 3 ∧ x1 ≠ x2) → (f x1 - f x2) / (x1 - x2) > 0

theorem solution : f 49 < f 64 ∧ f 64 < f 81 :=
by
  sorry

end solution_l399_399363


namespace jane_journey_duration_l399_399502

noncomputable def hours_to_seconds (h : ℕ) : ℕ := h * 3600 + 30

theorem jane_journey_duration :
  ∃ (start_time end_time : ℕ), 
    (start_time > 10 * 3600) ∧ (start_time < 11 * 3600) ∧
    (end_time > 17 * 3600) ∧ (end_time < 18 * 3600) ∧
    end_time - start_time = hours_to_seconds 7 :=
by sorry

end jane_journey_duration_l399_399502


namespace determine_functions_l399_399714

noncomputable def functional_eq_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = y + (f x) ^ 2

theorem determine_functions (f : ℝ → ℝ) (h : functional_eq_condition f) : 
  (∀ x, f x = x) ∨ (∀ x, f x = -x) :=
sorry

end determine_functions_l399_399714


namespace digit_at_206788_is_7_l399_399688

-- Definitions based on conditions in the problem
def single_digit_numbers := {n : ℕ | 1 ≤ n ∧ n ≤ 9}
def double_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
def triple_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
def four_digit_numbers := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999}
def five_digit_numbers := {n : ℕ | 10000 ≤ n}

-- Theorem statement to prove the digit at position 206788 is 7
theorem digit_at_206788_is_7 : digit_at_position 206788 = 7 := 
sorry

end digit_at_206788_is_7_l399_399688


namespace maximal_regions_convex_quadrilaterals_l399_399244

theorem maximal_regions_convex_quadrilaterals (n : ℕ) (hn : n ≥ 1) : 
  ∃ a_n : ℕ, a_n = 4*n^2 - 4*n + 2 :=
by
  sorry

end maximal_regions_convex_quadrilaterals_l399_399244


namespace kiwis_to_apples_l399_399156

theorem kiwis_to_apples :
  (1 / 4) * 20 = 10 → (3 / 4) * 12 * (2 / 5) = 18 :=
by
  sorry

end kiwis_to_apples_l399_399156


namespace evaluate_expr_l399_399377

-- Definitions from conditions
def expr := 3 * Real.sqrt 32 + 2 * Real.sqrt 50
def result := 22 * Real.sqrt 2

-- Theorem statement
theorem evaluate_expr : expr = result := by
  sorry

end evaluate_expr_l399_399377


namespace real_part_condition_l399_399851

theorem real_part_condition (m : ℝ) : 
  (∀ (z : ℂ), z = m + complex.i → (z^2).im = 0) ↔ (m = 1 ∨ m = -1) := sorry

end real_part_condition_l399_399851


namespace arithmetic_mean_reciprocals_primes_l399_399331

theorem arithmetic_mean_reciprocals_primes : 
  let p := [2, 3, 5, 7] in 
  let reciprocals := p.map (λ n => 1 / (n : ℚ)) in
  (reciprocals.sum / reciprocals.length) = (247 / 840 : ℚ) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l399_399331


namespace polynomial_non_negative_l399_399135

theorem polynomial_non_negative (x : ℝ) : x^8 + x^6 - 4*x^4 + x^2 + 1 ≥ 0 := 
sorry

end polynomial_non_negative_l399_399135


namespace expression_value_l399_399622

theorem expression_value (a b : ℕ) (h₁ : a = 37) (h₂ : b = 12) : 
  (a + b)^2 - (a^2 + b^2) = 888 := by
  sorry

end expression_value_l399_399622


namespace sin_alpha_l399_399426

-- Define a hypothesis for the point P(3a, 4a) lying on the terminal side of angle α
-- and the restriction that a ≠ 0.
variable {a : ℝ} (ha : a ≠ 0)

-- Define the coordinates (3a, 4a)
def P := (3 * a, 4 * a)

-- The theorem statement
theorem sin_alpha (ha : a ≠ 0) : (let α := ε in P_sin α P = 4 / 5 ∨ P_sin α P = -4 / 5) :=
sorry

end sin_alpha_l399_399426


namespace cube_surface_area_unchanged_l399_399656

def cubeSurfaceAreaAfterCornersRemoved
  (original_side : ℕ)
  (corner_side : ℕ)
  (original_surface_area : ℕ)
  (number_of_corners : ℕ)
  (surface_reduction_per_corner : ℕ)
  (new_surface_addition_per_corner : ℕ) : Prop :=
  (original_side * original_side * 6 = original_surface_area) →
  (corner_side * corner_side * 3 = surface_reduction_per_corner) →
  (corner_side * corner_side * 3 = new_surface_addition_per_corner) →
  original_surface_area - (number_of_corners * surface_reduction_per_corner) + (number_of_corners * new_surface_addition_per_corner) = original_surface_area
  
theorem cube_surface_area_unchanged :
  cubeSurfaceAreaAfterCornersRemoved 4 1 96 8 3 3 :=
by
  intro h1 h2 h3
  sorry

end cube_surface_area_unchanged_l399_399656


namespace total_students_left_l399_399985

-- Definitions for given conditions
def initialBoys := 14
def initialGirls := 10
def boysDropOut := 4
def girlsDropOut := 3

-- The proof problem statement
theorem total_students_left : 
  initialBoys - boysDropOut + (initialGirls - girlsDropOut) = 17 := 
by
  sorry

end total_students_left_l399_399985


namespace acute_angles_in_triangle_l399_399875

theorem acute_angles_in_triangle (α β γ : ℝ) (A_ext B_ext C_ext : ℝ) 
  (h_sum : α + β + γ = 180) 
  (h_ext1 : A_ext = 180 - β) 
  (h_ext2 : B_ext = 180 - γ) 
  (h_ext3 : C_ext = 180 - α) 
  (h_ext_acute1 : A_ext < 90 → β > 90) 
  (h_ext_acute2 : B_ext < 90 → γ > 90) 
  (h_ext_acute3 : C_ext < 90 → α > 90) : 
  ((α < 90 ∧ β < 90) ∨ (α < 90 ∧ γ < 90) ∨ (β < 90 ∧ γ < 90)) ∧ 
  ((A_ext < 90 → ¬ (B_ext < 90 ∨ C_ext < 90)) ∧ 
   (B_ext < 90 → ¬ (A_ext < 90 ∨ C_ext < 90)) ∧ 
   (C_ext < 90 → ¬ (A_ext < 90 ∨ B_ext < 90))) :=
sorry

end acute_angles_in_triangle_l399_399875


namespace min_value_m_l399_399992

noncomputable def a : ℕ → ℝ
| 0       => 1
| (n + 1) => 1 / (a n) / real.sqrt (1 / (a n) ^ 2 + 4)

noncomputable def S (n : ℕ) : ℝ :=
(finset.range n).sum (λ i, (a i) ^ 2)

theorem min_value_m : ∀ n : ℕ, (S (2 * n + 1) - S n) ≤ 1 / 30 * 10 :=
by
  sorry

end min_value_m_l399_399992


namespace sum_of_solutions_l399_399720

theorem sum_of_solutions :
  let equation := (λ x : ℝ, (x^2 - 5*x + 3)^(x^2 - 6*x + 5) = 1)
  ∑ x in {x : ℝ | equation x}, x = 11 := sorry

end sum_of_solutions_l399_399720


namespace find_product_abcd_l399_399819

def prod_abcd (a b c d : ℚ) :=
  4 * a - 2 * b + 3 * c + 5 * d = 22 ∧
  2 * (d + c) = b - 2 ∧
  4 * b - c = a + 1 ∧
  c + 1 = 2 * d

theorem find_product_abcd (a b c d : ℚ) (h : prod_abcd a b c d) :
  a * b * c * d = -30751860 / 11338912 :=
sorry

end find_product_abcd_l399_399819


namespace find_divisor_and_dividend_l399_399587

theorem find_divisor_and_dividend:
  ∃ (x: ℕ) (d: ℕ), 
    let q := 3 in
    let r := 20 in
    let dividend := q * x + r in
    (dividend + x + q + r = 303) ∧ (x = 65) ∧ (dividend = 215) :=
by
  sorry

end find_divisor_and_dividend_l399_399587


namespace speed_of_Mr_A_is_30_l399_399923

variables (distance_between_A : ℝ) (speed_Mrs_A : ℝ) (speed_bee : ℝ) (distance_traveled_bee : ℝ)
variable (speed_Mr_A : ℝ)

-- Constants based on the problem's conditions
def distance_between_A_and_Mrs_A := 120 -- km
def speed_Mrs_A := 10 -- kmph
def speed_bee := 60 -- kmph
def distance_traveled_bee := 180 -- km

-- The final theorem to be proven
theorem speed_of_Mr_A_is_30
  (h_distance_between_A : distance_between_A = distance_between_A_and_Mrs_A)
  (h_speed_Mrs_A : speed_Mrs_A = speed_Mrs_A)
  (h_speed_bee : speed_bee = speed_bee)
  (h_distance_traveled_bee : distance_traveled_bee = distance_traveled_bee) :
  speed_Mr_A = 30 :=
sorry

end speed_of_Mr_A_is_30_l399_399923


namespace least_positive_x_l399_399915

variable (a b : ℝ)

noncomputable def tan_inv (x : ℝ) : ℝ := Real.arctan x

theorem least_positive_x (x k : ℝ) 
  (h1 : Real.tan x = a / b)
  (h2 : Real.tan (2 * x) = b / (a + b))
  (h3 : Real.tan (3 * x) = (a - b) / (a + b))
  (h4 : x = tan_inv k)
  : k = 13 / 9 := sorry

end least_positive_x_l399_399915


namespace coffee_ratio_l399_399049

/-- Define the conditions -/
def initial_coffees_per_day := 4
def initial_price_per_coffee := 2
def price_increase_percentage := 50 / 100
def savings_per_day := 2

/-- Define the price calculations -/
def new_price_per_coffee := initial_price_per_coffee + (initial_price_per_coffee * price_increase_percentage)
def initial_daily_cost := initial_coffees_per_day * initial_price_per_coffee
def new_daily_cost := initial_daily_cost - savings_per_day
def new_coffees_per_day := new_daily_cost / new_price_per_coffee

/-- Prove the ratio -/
theorem coffee_ratio : (new_coffees_per_day / initial_coffees_per_day) = (1 : ℝ) / (2 : ℝ) :=
  by sorry

end coffee_ratio_l399_399049


namespace expression_not_defined_at_x_eq_5_l399_399781

theorem expression_not_defined_at_x_eq_5 :
  ∃ x : ℝ, x^3 - 15 * x^2 + 75 * x - 125 = 0 ↔ x = 5 :=
by
  sorry

end expression_not_defined_at_x_eq_5_l399_399781


namespace magnitude_of_b_l399_399839

open Real

-- Given vectors a, b, and c
variables (a b c : ℝ → ℝ)
-- Conditions
variables (h₁ : a - b + 2 * c = 0)
variables (h₂ : a ∙ c = 0)
variables (h₃ : ∥a∥ = 2)
variables (h₄ : ∥c∥ = 1)

-- Required to prove
theorem magnitude_of_b (a b c : ℝ → ℝ) (h₁ : a - b + 2 * c = 0)
(h₂ : a ∙ c = 0) (h₢ : ∥a∥ = 2) (h₣ : ∥c∥ = 1) : ∥b∥ = 2 * √2 :=
  sorry

end magnitude_of_b_l399_399839


namespace extremal_points_f_l399_399911

noncomputable def f (α β γ : ℝ) (K : Set (ℝ × ℝ × ℝ)) : (ℝ × ℝ × ℝ) → ℝ :=
  λ ⟨x, y, z⟩ => x^α + y^β + z^γ

theorem extremal_points_f (α β γ : ℝ) (hαβγ: 0 < α ∧ α < β ∧ β < γ) :
  ∃ x y z, (x, y, z) ∈ { (x, y, z) | x >= 0 ∧ y >= 0 ∧ z >= 0 ∧ x^β + y^β + z^β = 1 } ∧ 
      (f α β γ { (x, y, z) | x >= 0 ∧ y >= 0 ∧ z >= 0 ∧ x^β + y^β + z^β = 1 } (x, y, z) = ( 
      (α / β)^(1 / (β - α)) + (1 - (α / β)^(β / (β - α)))^(1 / β) + 0 ∨ 
      f α β γ { (x, y, z) | x >= 0 ∧ y >= 0 ∧ z >= 0 ∧ x^β + y^β + z^β = 1 } (x, y, z) = 
      0 + (1 - (β / γ)^(β / (γ - β)))^(1 / β) + (β / γ)^(1 / (γ - β))) :=
sorry

end extremal_points_f_l399_399911


namespace original_price_l399_399375

theorem original_price (spent : ℝ) (less_percentage : ℝ) (original_price : ℝ) : 
  spent = 7500 → less_percentage = 0.25 → original_price = 10000 := 
begin
  intros h1 h2,
  have h3 : spent = (1 - less_percentage) * original_price,
  { rw [h1, h2], ring },
  have h4 : 7500 = 0.75 * original_price,
  { rw h3 },
  have h5 : original_price = 7500 / 0.75,
  { linarith },
  rw div_eq_mul_inv at h5,
  simp at h5,
  exact h5,
  sorry
end

end original_price_l399_399375


namespace monotonicity_and_no_real_roots_l399_399091

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

theorem monotonicity_and_no_real_roots 
  (a : ℝ) (ha : 0 < a) : 
  (∀ x : ℝ, (0 < x ∧ x < (1 / a) → deriv (f a) x < 0) ∧ (x > (1 / a) → deriv (f a) x > 0)) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≠ 0 → a > (1 / real.exp 1))) :=
begin
  sorry
end

end monotonicity_and_no_real_roots_l399_399091


namespace subset_r_elements_max_m_l399_399909

open Finset Nat

theorem subset_r_elements_max_m (n r m : ℕ) (S : Finset ℕ) 
    (A : Finset (Finset ℕ)) (hS : S = (range n).filter (λ x, 1 ≤ x)) 
    (hm : m ≥ 2) 
    (hAi : ∀ Ai ∈ A, card Ai = r) 
    (hAj : ∀ Ai ∈ A, ∃ Aj ∈ A, Aj ≠ Ai ∧ ((Finset.max' Ai (nonempty_of_card_eq_succ (hAi Ai (by_ho_ad_filter Ai))).elim_left =  
                                                Finset.min' Aj (nonempty_of_card_eq_succ (hAi Aj (by_ho_ad_filter Aj))).elim_left) ∨ 
                                               (Finset.min' Ai (nonempty_of_card_eq_succ (hAi Ai (by_ho_ad_filter Ai))).elim_left = 
                                                Finset.max' Aj (nonempty_of_card_eq_succ (hAi Aj (by_ho_ad_filter Aj))).elim_left))) :
  2 ≤ r ∧ r ≤ (n + 1) / 2 ∧ m ≤ 2 * choose (n + 1) r - choose (n + 2 - 2 * r) r := 
sorry

end subset_r_elements_max_m_l399_399909


namespace solve_z6_eq_neg4_l399_399777

open Complex

theorem solve_z6_eq_neg4 :
  let z1 := (root (6 : ℝ) (4/13)) + (root (6 : ℝ) (4/13)) * Complex.I
  let z2 := (root (6 : ℝ) (4/13)) - (root (6 : ℝ) (4/13)) * Complex.I
  let z3 := -(root (6 : ℝ) (4/13)) + (root (6 : ℝ) (4/13)) * Complex.I
  let z4 := -(root (6 : ℝ) (4/13)) - (root (6 : ℝ) (4/13)) * Complex.I
  (z1^6 = -4) ∧ (z2^6 = -4) ∧ (z3^6 = -4) ∧ (z4^6 = -4) :=
by
  let z1 := (root (6 : ℝ) (4/13)) + (root (6 : ℝ) (4/13)) * Complex.I
  let z2 := (root (6 : ℝ) (4/13)) - (root (6 : ℝ) (4/13)) * Complex.I
  let z3 := -(root (6 : ℝ) (4/13)) + (root (6 : ℝ) (4/13)) * Complex.I
  let z4 := -(root (6 : ℝ) (4/13)) - (root (6 : ℝ) (4/13)) * Complex.I
  have h1 : z1^6 = -4 := sorry
  have h2 : z2^6 = -4 := sorry
  have h3 : z3^6 = -4 := sorry
  have h4 : z4^6 = -4 := sorry
  exact ⟨h1, h2, h3, h4⟩

end solve_z6_eq_neg4_l399_399777


namespace quadratic_solution_downward_solution_minimum_solution_l399_399444

def is_quadratic (m : ℝ) : Prop :=
  m^2 + 3 * m - 2 = 2

def opens_downwards (m : ℝ) : Prop :=
  m + 3 < 0

def has_minimum (m : ℝ) : Prop :=
  m + 3 > 0

theorem quadratic_solution (m : ℝ) :
  is_quadratic m → (m = -4 ∨ m = 1) :=
sorry

theorem downward_solution (m : ℝ) :
  is_quadratic m → opens_downwards m → m = -4 :=
sorry

theorem minimum_solution (m : ℝ) :
  is_quadratic m → has_minimum m → m = 1 :=
sorry

end quadratic_solution_downward_solution_minimum_solution_l399_399444


namespace john_clean_portion_in_one_third_nick_time_l399_399261

theorem john_clean_portion_in_one_third_nick_time (john_time_nick_time_ratio : ℕ) (john_time : ℕ):
    (john_cleaning_time := 6) (combined_time := 36 / 10) : 
    (nick_time := 9) :
    (portion := (1 / (3 * nick_time)) * john_cleaning_time) :
    portion = 1 / 2 :=
by
    have h₁ : john_cleaning_time = 6 := rfl
    have h₂ : combined_time = 3.6 := by norm_num
    have h₃: nick_time = 9 := by norm_num
    have h₄ : portion = 1 / (3 * 9) * 6 := rfl
    have h₅ : 1 / (3 * 9) * 6 = 1 / 2 := by norm_num
    exact h₅

end john_clean_portion_in_one_third_nick_time_l399_399261


namespace parents_to_students_ratio_l399_399020

-- Define the total money donated
variables (total_donation parent_donation teacher_donation student_donation : ℝ)

-- Conditions
-- 1. 25% of the money donated came from parents.
def condition1 : Prop := parent_donation = 0.25 * total_donation

-- 2. Define the amounts donated by teachers and students
def condition2 : Prop := teacher_donation + student_donation = 0.75 * total_donation

-- 3. The ratio of the amount of money donated by teachers to the amount donated by students is 2:3
def condition3 : Prop := teacher_donation / student_donation = 2 / 3

-- Theorem
theorem parents_to_students_ratio :
    condition1 → condition2 → condition3 → (parent_donation / student_donation = 5 / 9) :=
by
  intros h1 h2 h3
  sorry

end parents_to_students_ratio_l399_399020


namespace smallest_multiple_greater_than_30_l399_399776

-- Define the problem conditions in Lean 4
def is_multiple (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

-- Define the least common multiple function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Define the statement to prove
theorem smallest_multiple_greater_than_30 : 
  ∃ n > 30, (∀ m ∈ {1, 2, 3, 4, 5}, is_multiple n m) ∧ 
            ∀ k, ((∀ m ∈ {1, 2, 3, 4, 5}, is_multiple k m) ∧ k > 30) → n ≤ k :=
begin
  -- Proof goes here
  sorry
end

end smallest_multiple_greater_than_30_l399_399776


namespace cube_root_of_sum_of_powers_l399_399250

theorem cube_root_of_sum_of_powers :
  (∛ (3^5 + 3^5 + 3^5) = 9) := by
  sorry

end cube_root_of_sum_of_powers_l399_399250


namespace complementary_angles_ratio_decrease_l399_399207

theorem complementary_angles_ratio_decrease 
  (a b : ℝ) (h_ratio : a / b = 3 / 7) (h_comp : a + b = 90) : 
  let a' := a * 1.20 in
  let b' := 90 - a' in
  ((b' / b) * 100 = 91.43) ∧ (100 - (b' / b) * 100 = 8.57) :=
by
  have : a / b = 3 / 7 := h_ratio
  have : a + b = 90 := h_comp
  let a' := a * 1.20
  let b' := 90 - a'
  have : b' / b = 57.6 / 63 := sorry -- Calculations omitted for brevity
  have : (b' / b) * 100 = 91.43 := by sorry
  have : 100 - (b' / b) * 100 = 8.57 := by sorry
  split;
  assumption

end complementary_angles_ratio_decrease_l399_399207


namespace limit_fraction_l399_399703

open Filter

theorem limit_fraction (a b : ℕ → ℝ) (ha : ∀ n, a n = n^2 + n + 1) (hb : ∀ n, b n = 2 * n^2 + 3 * n + 2) :
  tendsto (λ n, (a n) / (b n)) at_top (𝓝 (1/2)) :=
sorry

end limit_fraction_l399_399703


namespace tabitha_final_amount_l399_399160

def initial_amount : ℝ := 25
def amount_given_to_mom : ℝ := 8
def items_bought : ℕ := 5
def cost_per_item : ℝ := 0.5

theorem tabitha_final_amount :
  let remaining_after_mom := initial_amount - amount_given_to_mom in
  let remaining_after_investment := remaining_after_mom / 2 in
  let spent_on_items := items_bought * cost_per_item in
  let final_amount := remaining_after_investment - spent_on_items in
  final_amount = 6 := by
  sorry

end tabitha_final_amount_l399_399160


namespace real_part_of_conjugate_of_z_l399_399433

noncomputable def z : ℂ := (1 + 3*complex.i) / (1 - complex.i)
def z_conj := complex.conj z
def z_conj_real_part := z_conj.re

theorem real_part_of_conjugate_of_z : z_conj_real_part = -1 := 
by {
  sorry
}

end real_part_of_conjugate_of_z_l399_399433


namespace product_of_def_l399_399157

open Complex

noncomputable def Q (x : ℂ) (d e f : ℂ) : ℂ := x^3 + d*x^2 + e*x + f

theorem product_of_def {d e f : ℂ}
  (hroots : ∀ {r : ℂ}, r ∈ ({cos (π / 9), cos (5 * π / 9), cos (7 * π / 9)}) ↔ Q r d e f = 0) :
  d * e * f = -1/256 := by
  sorry

end product_of_def_l399_399157


namespace triangle_reflection_collinearity_iff_circumradius_l399_399409

variables {A B C D E F H O : Point}
variable {R : ℝ}
noncomputable def is_reflected_over (p q r : Point) : Point :=
  sorry -- Definition of reflection of a point over a line.

def is_collinear (p q r : Point) : Prop :=
  sorry -- Definition of collinearity of three points.

def orthocenter (A B C : Point) : Point :=
  sorry -- Definition of the orthocenter of a triangle.

def circumcenter (A B C : Point) : Point :=
  sorry -- Definition of the circumcenter of a triangle.

def circumradius (A B C : Point) : ℝ :=
  sorry -- Definition of the circumradius of a triangle.

def distance (p q : Point) : ℝ :=
  sorry -- Definition of the distance between two points.

theorem triangle_reflection_collinearity_iff_circumradius (A B C : Point) :
  let H := orthocenter A B C
  let O := circumcenter A B C
  let R := circumradius A B C
  D = is_reflected_over A B C BC →
  E = is_reflected_over B C A CA →
  F = is_reflected_over C A B AB →
  (is_collinear D E F ↔ distance O H = 2 * R) :=
by
  intros H O R hD hE hF
  exact sorry

end triangle_reflection_collinearity_iff_circumradius_l399_399409


namespace area_under_curve_l399_399054

noncomputable def piecewise_function (x : ℝ) : ℝ :=
if x ≥ 0 ∧ x ≤ 6 then 2 * x
else if x > 6 ∧ x ≤ 8 then 3 * x - 6
else if x > 8 ∧ x ≤ 10 then x + 6
else 0

theorem area_under_curve : 
  ∫ x in 0..10, piecewise_function x = 132 :=
sorry

end area_under_curve_l399_399054


namespace cut_third_link_allows_payment_l399_399123

def chain := list nat

-- Conditions
def has_seven_links (c: chain) : Prop := c.length = 7
def pays_one_link_per_day (days: ℕ) (links_cut: ℕ) : Prop := days = 7 ∧ links_cut = 1

-- The theorem to be proven
theorem cut_third_link_allows_payment 
  (c: chain) 
  (h1: has_seven_links c)
  (h2: pays_one_link_per_day 7 1) : 
  ∃ links_after_cut: chain, subset links_after_cut [3, 1, 1, 1, 1, 1, 1] :=
sorry

end cut_third_link_allows_payment_l399_399123


namespace victor_earnings_l399_399610

def hourly_wage := 6 -- dollars per hour
def hours_monday := 5 -- hours
def hours_tuesday := 5 -- hours

theorem victor_earnings : (hourly_wage * (hours_monday + hours_tuesday)) = 60 :=
by
  sorry

end victor_earnings_l399_399610


namespace jane_not_finish_probability_l399_399585

theorem jane_not_finish_probability :
  (1 : ℚ) - (5 / 8) = (3 / 8) := by
  sorry

end jane_not_finish_probability_l399_399585


namespace find_a_and_b_max_min_g_l399_399440

-- Definitions for the conditions and the function
def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b * real.log x

-- The proof problem: Proving the values of a and b
theorem find_a_and_b :
  ∃ a b : ℝ, (∀ x : ℝ, ∃ y : ℝ, 
    (y = a * x + b * real.log x ∧ 
     (x = 1 → y = a ∧ (a + b = -1) ∧ (y = 3 - x)))) ∧ a = 2 ∧ b = -3 :=
by
  sorry

-- Defining the function g(x)
def g (x : ℝ) (a b : ℝ) : ℝ := f x a b - 1 / x

-- Maximum and minimum values proof problem for g(x)
theorem max_min_g :
  ∃ c d : ℝ, 
    (∀ x ∈ set.Icc (1 / 2) 2, g x 2 (-3) ≥ c ∧ g x 2 (-3) ≤ d) ∧ 
    c = 1 ∧ d = (7 / 2 - 3 * real.log 2) :=
by
  sorry

end find_a_and_b_max_min_g_l399_399440


namespace factorization_of_x_squared_minus_4_l399_399626

theorem factorization_of_x_squared_minus_4 (x : ℝ) : x^2 - 4 = (x - 2) * (x + 2) :=
by
  sorry

end factorization_of_x_squared_minus_4_l399_399626


namespace distance_between_vertices_l399_399752

def hyperbola_eq (x y : ℝ) : Prop := 
  x^2 / 144 - y^2 / 49 = 1

theorem distance_between_vertices : 2 * real.sqrt 144 = 24 :=
by {
    -- use sqrt calculation for clarity
    have h : real.sqrt 144 = 12, by {
        exact real.sqrt_eq_iff_sq_eq.mpr (or.inl (by norm_num)),
    },
    rw [h],
    norm_num
}

end distance_between_vertices_l399_399752


namespace exists_substring_divisible_by_2011_l399_399939

theorem exists_substring_divisible_by_2011 :
  ∃ N : ℕ, ∀ a : ℤ, (0 < a - N) → ∃ s : ℕ, (s > 0 ∧ s ≤ a) ∧ substring_is_contiguous a s ∧ s % 2011 = 0 :=
by
  sorry

-- Helper definition for checking if a substring is contiguous
def substring_is_contiguous (a : ℤ) (s : ℕ) : Prop :=
  ∃ i j : ℕ, 0 ≤ i ∧ i ≤ j ∧ j < a.to_string.length ∧ 
             (s = (a.to_string.slice i j).to_nat)

attribute [simp] substring_is_contiguous

end exists_substring_divisible_by_2011_l399_399939


namespace number_of_Z_tetromino_placements_l399_399022

-- Define the size of the chessboard and the shape of the tetromino
def board_size : ℕ := 8
def tetromino_shape : Type := -- (Define the Z-shape tetromino specifically)

-- Statement to prove
theorem number_of_Z_tetromino_placements :
  (count_placements board_size tetromino_shape) = 168 :=
begin
  -- Proof goes here (but is omitted according to the instructions)
  sorry
end

end number_of_Z_tetromino_placements_l399_399022


namespace john_books_per_day_l399_399508

theorem john_books_per_day (books_per_week := 2) (weeks := 6) (total_books := 48) :
  (total_books / (books_per_week * weeks) = 4) :=
by
  sorry

end john_books_per_day_l399_399508


namespace complementary_angles_ratio_decrease_l399_399206

theorem complementary_angles_ratio_decrease 
  (a b : ℝ) (h_ratio : a / b = 3 / 7) (h_comp : a + b = 90) : 
  let a' := a * 1.20 in
  let b' := 90 - a' in
  ((b' / b) * 100 = 91.43) ∧ (100 - (b' / b) * 100 = 8.57) :=
by
  have : a / b = 3 / 7 := h_ratio
  have : a + b = 90 := h_comp
  let a' := a * 1.20
  let b' := 90 - a'
  have : b' / b = 57.6 / 63 := sorry -- Calculations omitted for brevity
  have : (b' / b) * 100 = 91.43 := by sorry
  have : 100 - (b' / b) * 100 = 8.57 := by sorry
  split;
  assumption

end complementary_angles_ratio_decrease_l399_399206


namespace area_sum_constant_for_all_points_l399_399550

noncomputable def area_of_triangle (A B P : Point) : Real := sorry

theorem area_sum_constant_for_all_points 
  {n : ℕ}
  {A : Fin 2n → Point} 
  {P1 P2 P3 : Point}
  (h_convex : convex (A 0) (A 1) ⋯ (A (2 * n - 1)))
  (h_P1_interior : is_in_interior (A 0) (A 1) ⋯ (A (2 * n - 1)) P1)
  (h_P2_interior : is_in_interior (A 0) (A 1) ⋯ (A (2 * n - 1)) P2)
  (h_P3_interior : is_in_interior (A 0) (A 1) ⋯ (A (2 * n - 1)) P3)
  (h_sum_areas_P1 : ∑ k in finRange n, area_of_triangle (A (2 * k)) (A (2 * k + 1)) P1 = c)
  (h_sum_areas_P2 : ∑ k in finRange n, area_of_triangle (A (2 * k)) (A (2 * k + 1)) P2 = c)
  (h_sum_areas_P3 : ∑ k in finRange n, area_of_triangle (A (2 * k)) (A (2 * k + 1)) P3 = c) 
  (P : Point)
  (h_P_interior : is_in_interior (A 0) (A 1) ⋯ (A (2 * n - 1)) P) : 
  ∑ k in finRange n, area_of_triangle (A (2 * k)) (A (2 * k + 1)) P = c :=
sorry

end area_sum_constant_for_all_points_l399_399550


namespace rounding_to_hundredth_l399_399606

theorem rounding_to_hundredth {x : ℝ} (h : x = 2.0359) : (Real.toHundredth x) = 2.04 :=
by
  rw h
  sorry

end rounding_to_hundredth_l399_399606


namespace percent_paddyfield_warblers_kingfishers_l399_399866

variable (birds: ℝ)  -- total number of birds
variable (hawks: ℝ)  -- number of hawks
variable (paddyfield_warblers: ℝ)  -- number of paddyfield-warblers
variable (kingfishers: ℝ)  -- number of kingfishers

-- Given conditions as Lean statements
def condition1 : Prop := hawks / birds = 0.3
def condition2 : Prop := paddyfield_warblers = 0.4 * (birds - hawks)
def condition3 : Prop := kingfishers / birds = x
def condition4 : Prop := (birds - hawks - paddyfield_warblers - kingfishers) / birds = 0.35

-- Question to be proved
theorem percent_paddyfield_warblers_kingfishers (h1: condition1)
  (h2: condition2) (h3: condition3) (h4: condition4) :
  (kingfishers / paddyfield_warblers) * 100 = 25 := by
  sorry

end percent_paddyfield_warblers_kingfishers_l399_399866


namespace ghee_mixture_weight_l399_399275

theorem ghee_mixture_weight 
  (wa wb : ℕ) (ra_a ra_b total_vol : ℕ)
  (w_a_l w_b_l : ℕ) 
  (wa_eq : wa = 900)
  (wb_eq : wb = 750)
  (ra_eq : ra_a = 3 ∧ ra_b = 2)
  (vol_eq : total_vol = 4) 
  (res_eq : (wa * (ra_a * total_vol) / (ra_a + ra_b) + wb * (ra_b * total_vol) / (ra_a + ra_b)) / 1000 = 3.36) : 
  True :=
by
  sorry

end ghee_mixture_weight_l399_399275


namespace odd_function_value_l399_399424

variable {R : Type} [LinearOrderedField R]

def is_odd (f : R → R) :=
  ∀ x, f (-x) = -f x

theorem odd_function_value (f : R → R) 
  (hodd : is_odd f)
  (hpos : ∀ x, 0 < x → f x = - x / (2 * x + 1))
  (x : R) (hx : x < 0) : f x = x / (2 * x - 1) :=
by
  have hnx : 0 < -x := by linarith
  rw [← hodd, hpos (-x) hnx]
  ring

end odd_function_value_l399_399424


namespace triangle_to_square_impossible_l399_399042

theorem triangle_to_square_impossible :
  ∀ (A B C : ℝ) (h1 : A = 20000) (h2 : B = 1 / 10000), ¬(∃ parts : set (set (ℝ × ℝ)), 
    parts.card = 1000 ∧ (∀ p ∈ parts, is_triangle_subpart p A B C) ∧ 
    (∃ square : set (ℝ × ℝ), is_square square ∧ (⋃ p ∈ parts, p) = square)) :=
by
  intros A B C h1 h2
  sorry

end triangle_to_square_impossible_l399_399042


namespace hyperbola_vertex_distance_l399_399767

theorem hyperbola_vertex_distance : 
  ∀ x y: ℝ, (x^2 / 144 - y^2 / 49 = 1) → (∃ a: ℝ, a = 12 ∧ 2 * a = 24) :=
by 
  sorry

end hyperbola_vertex_distance_l399_399767


namespace rank_from_left_l399_399681

theorem rank_from_left (total_students rank_from_right : ℕ) (h1 : total_students = 21) (h2 : rank_from_right = 17) : 
  ∃ rank_from_left : ℕ, rank_from_left = 5 :=
by 
  have h : total_students = rank_from_right + rank_from_left - 1,
  sorry

end rank_from_left_l399_399681


namespace no_real_roots_of_quadratic_l399_399221

theorem no_real_roots_of_quadratic :
  ∀ (x : ℝ), ¬ (x^2 + 2 * x + 3 = 0) :=
begin
  intro x,
  have h : 2^2 - 4 * 1 * 3 = -8, by norm_num,
  have discriminant_neg : 2^2 - 4 * 1 * 3 < 0, from by { norm_num },
  rw [←add_zero (x^2 + 2 * x), show 3 = 1 + 2, from by norm_num],
  intro h_eq,
  apply discriminant_neg,
end

end no_real_roots_of_quadratic_l399_399221


namespace function_increasing_on_neg_inf_0_l399_399438

def f (x : ℝ) : ℝ := 8 + 2*x - x^2

theorem function_increasing_on_neg_inf_0 : ∀ x y : ℝ, x < y → y ≤ 0 → f(x) < f(y) := 
by
  sorry

end function_increasing_on_neg_inf_0_l399_399438


namespace desired_ratio_milk_to_water_l399_399018

noncomputable def initial_volume := 60 -- in litres
def initial_ratio_milk := 2
def initial_ratio_water := 1
noncomputable def added_water := 60 -- in litres

theorem desired_ratio_milk_to_water :
  let total_parts := initial_ratio_milk + initial_ratio_water,
      each_part_volume := initial_volume / total_parts,
      initial_milk := initial_ratio_milk * each_part_volume,
      initial_water := initial_ratio_water * each_part_volume,
      new_water := initial_water + added_water
  in  initial_milk / new_water = 1 / 2 :=
by
  -- proof omitted
  sorry

end desired_ratio_milk_to_water_l399_399018


namespace tan_A_in_triangle_l399_399033

theorem tan_A_in_triangle (A B C : Point) (hC : Angle C = 90) (h1 : distance A B = 13) (h2 : distance B C = 5) :
  tan (angle A) = 5 / 12 :=
sorry

end tan_A_in_triangle_l399_399033


namespace solution_set_inequality_l399_399432

noncomputable def f : ℝ → ℝ := sorry -- assume differentiable function f
noncomputable def g (x : ℝ) : ℝ := f(x) / Real.exp(x)

theorem solution_set_inequality :
  (∀ x : ℝ, (f(x) - f(x - 1)) / (x - 1) > 0) →
  (∀ (x : ℝ), g(x) = f(x) / Real.exp(x)) →
  (∀ (y : ℝ), (f(y) / Real.exp(y)) < f(0) ↔ ((-1 < x ∧ x < 0) ∨ (1 < x ∧ x < 2))) := by
  sorry

end solution_set_inequality_l399_399432


namespace monotonicity_and_no_x_intercept_l399_399098

noncomputable theory

def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

def is_monotonic (a : ℝ) (x : ℝ) : Prop := 
  if x < 1 / a then 
    f a x > f a (x + ε) -- ε is a small positive value
  else if x > 1 / a then 
    f a x < f a (x - ε)
  else
    true -- At x = 1/a, the function transits from decreasing to increasing

theorem monotonicity_and_no_x_intercept 
  (a : ℝ) (h1 : 0 < a) : 
  (∀ x : ℝ, 0 < x → is_monotonic a x) ∧ 
  (∀ x : ℝ, f a x ≠ 0) ↔ 
  (a ∈ Ioi (1 / real.exp 1)) := 
sorry

end monotonicity_and_no_x_intercept_l399_399098


namespace arithmetic_mean_reciprocal_primes_l399_399344

theorem arithmetic_mean_reciprocal_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := (1 : ℚ) / p1
  let r2 := (1 : ℚ) / p2
  let r3 := (1 : ℚ) / p3
  let r4 := (1 : ℚ) / p4
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 := by
sorry

end arithmetic_mean_reciprocal_primes_l399_399344


namespace hyperbola_vertex_distance_l399_399764

theorem hyperbola_vertex_distance : 
  ∀ x y: ℝ, (x^2 / 144 - y^2 / 49 = 1) → (∃ a: ℝ, a = 12 ∧ 2 * a = 24) :=
by 
  sorry

end hyperbola_vertex_distance_l399_399764


namespace base_12_addition_example_l399_399687

theorem base_12_addition_example :
  let a := 857
  let b := 296
  let sum := "B31"
  (a₁ := 8) (a₂ := 5) (a₃ := 7)
  (b₁ := 2) (b₂ := 9) (b₃ := 6)
  (sum₁ := "B") (sum₂ := 3) (sum₃ := 1)
  (base := 12)
  show Nat.fromDigits base [a₁,a₂,a₃] + Nat.fromDigits base [b₁,b₂,b₃] == Nat.fromDigits base [sum₁,"3","1"] by
  sorry

end base_12_addition_example_l399_399687


namespace general_formula_a_sum_of_bn_l399_399797

-- Definitions given in the conditions
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - 2

-- Questions translated into proof statements involving the given conditions
theorem general_formula_a (a : ℕ → ℕ) :
  (∀ n, S a n = S a (n - 1) → a n = 2 * a (n - 1)) →
  a 1 = 2 →
  ∃ f : ℕ → ℕ, ∀ n, a n = f n ∧ f n = 2 ^ n :=
by
  intros h1 ha1
  use (λ n, 2 ^ n)
  intros n
  sorry

theorem sum_of_bn (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) :
  (∀ n, a n = 2 ^ n) →
  (∀ n, b n = a n * int.log_base 2 (a (n + 1))) →
  (∀ n, T n = ∑ i in finset.range n, b i) →
  ∃ f : ℕ → ℕ, ∀ n, T n = f n ∧ f n = (n + 1) * 2 ^ (n + 1) - 2 :=
by
  intros ha hb hT
  use (λ n, (n + 1) * 2 ^ (n + 1) - 2)
  intros n
  sorry

end general_formula_a_sum_of_bn_l399_399797


namespace arithmetic_mean_reciprocals_primes_l399_399327

theorem arithmetic_mean_reciprocals_primes
  (p : Finset ℕ)
  (h_p : p = {2, 3, 5, 7})
  : (p.sum (λ x, 1 / ↑x) / 4) = (247 / 840) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l399_399327


namespace puzzle_solution_exists_l399_399370

def position := ℕ
def board := finset position

def is_jumpable (b : board) (src dst over : position) : Prop :=
  b ∈ (insert src (insert over ∅)) ∧ dst ∉ b ∧ 
  (src = 9 ∨ src ≠ 9) -- first jump by piece in position 9, other jumps can be any position

def perform_jump (b : board) (src dst over : position) : board :=
  if is_jumpable b src dst over
  then (b.erase src).erase over ∪ {dst}
  else b

def move_sequence (initial_board : board) (seq : list (position × position × position)) : board :=
  seq.foldl (λ b (src, dst, over), perform_jump b src dst over) initial_board

def puzzle_solved (initial_board : board) (final_board : board) : Prop :=
  ∃ seq : list (position × position × position), (move_sequence initial_board seq) = final_board

theorem puzzle_solution_exists :
  ∃ seq : list (position × position × position), 
    ∃ initial_board final_board : board,
      (initial_board.card = 17) ∧ 
      (final_board.card = 1) ∧
      initial_board 9 ∧ 
      final_board 9 ∧
      (perform_jump initial_board seq = final_board) := 
sorry

end puzzle_solution_exists_l399_399370


namespace find_cost_price_l399_399673

variable (CP : ℝ)

def selling_price (CP : ℝ) := CP * 1.40

theorem find_cost_price (h : selling_price CP = 1680) : CP = 1200 :=
by
  sorry

end find_cost_price_l399_399673


namespace total_shaded_area_l399_399305

theorem total_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) : 
  S^2 + 12 * T^2 = 15.75 :=
by 
  sorry

end total_shaded_area_l399_399305


namespace min_quality_inspection_machines_l399_399586

theorem min_quality_inspection_machines (z x : ℕ) :
  (z + 30 * x) / 30 = 1 →
  (z + 10 * x) / 10 = 2 →
  (z + 5 * x) / 5 ≥ 4 :=
by
  intros h1 h2
  sorry

end min_quality_inspection_machines_l399_399586


namespace power_function_properties_l399_399862

theorem power_function_properties (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x ^ a) (h2 : f 2 = Real.sqrt 2) : 
  a = 1 / 2 ∧ ∀ x, 0 ≤ x → f x ≤ f (x + 1) :=
by
  sorry

end power_function_properties_l399_399862


namespace reasonable_conclusion_l399_399977

/-- Variables representing the production and sales functions, 
    along with their respective growth rates. -/
variables (l1 l2 : ℝ → ℝ) (g1 g2 : ℝ)
/-- Conditions stating that production and sales are increasing linearly 
    and there exists a situation of supply exceeding demand. -/
axiom cond1 : ∀ t, l1(t) = g1 * t 
axiom cond2 : ∀ t, l2(t) = g2 * t 
axiom cond3 : ∃ t, l1(t) > l2(t)
axiom cond4 : g1 > g2

/-- Theorem proving that due to supply exceeding demand and increasing 
    inventory, production should be reduced or sales increased. -/
theorem reasonable_conclusion : cond2 ∧ cond3 :=
by sorry

end reasonable_conclusion_l399_399977


namespace complementary_angle_decrease_l399_399209

theorem complementary_angle_decrease (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3 / 7) : 
  (∃ new_α : ℝ, new_α = α * 1.2) →
  ∃ new_β : ℝ, new_β = (1 - 0.0857) * β :=
by
  intro h3
  cases h3 with new_α h_newα
  use 90 - new_α
  sorry

end complementary_angle_decrease_l399_399209


namespace monotonicity_and_no_real_roots_l399_399088

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

theorem monotonicity_and_no_real_roots 
  (a : ℝ) (ha : 0 < a) : 
  (∀ x : ℝ, (0 < x ∧ x < (1 / a) → deriv (f a) x < 0) ∧ (x > (1 / a) → deriv (f a) x > 0)) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≠ 0 → a > (1 / real.exp 1))) :=
begin
  sorry
end

end monotonicity_and_no_real_roots_l399_399088


namespace smallest_number_is_1111_in_binary_l399_399256

theorem smallest_number_is_1111_in_binary :
  let a := 15   -- Decimal equivalent of 1111 in binary
  let b := 78   -- Decimal equivalent of 210 in base 6
  let c := 64   -- Decimal equivalent of 1000 in base 4
  let d := 65   -- Decimal equivalent of 101 in base 8
  a < b ∧ a < c ∧ a < d := 
by
  let a := 15
  let b := 78
  let c := 64
  let d := 65
  show a < b ∧ a < c ∧ a < d
  sorry

end smallest_number_is_1111_in_binary_l399_399256


namespace converse_proposition_true_l399_399257

theorem converse_proposition_true (x y : ℝ) (h : x > abs y) : x > y := 
by
sorry

end converse_proposition_true_l399_399257


namespace tan_A_in_triangle_l399_399035

theorem tan_A_in_triangle (A B C : Point) (hC : Angle C = 90) (h1 : distance A B = 13) (h2 : distance B C = 5) :
  tan (angle A) = 5 / 12 :=
sorry

end tan_A_in_triangle_l399_399035


namespace number_of_pairs_l399_399989

theorem number_of_pairs :
  ∃ (n : ℕ), ∀ (a r : ℕ), 
    (a > 0 ∧ r > 0) →
    (λ (seq : ℕ → ℕ), seq 1 = a ∧ (∀ (k : ℕ), 1 ≤ k ∧ k < 13 → seq (k + 1) = seq k * r)) ∧
    (list.sum (list.map (λ i, real.logb 2015 (a * r^i)) (list.range 13)) = 2015) →
    n = 26^3 :=
sorry

end number_of_pairs_l399_399989


namespace remainder_base14_2641_div_10_l399_399625

theorem remainder_base14_2641_div_10 :
  let n := 2 * 14^3 + 6 * 14^2 + 4 * 14^1 + 1 * 14^0 in
  n % 10 = 1 :=
by
  sorry

end remainder_base14_2641_div_10_l399_399625


namespace binomial_expansion_calculation_l399_399697

theorem binomial_expansion_calculation :
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 101^5 :=
by
  sorry

end binomial_expansion_calculation_l399_399697


namespace binom_expansion_properties_l399_399881

-- Definitions and conditions definition as per part a):
def binom_expansion (a b : ℤ) (n : ℕ) : ℤ := (a - b) ^ n

-- Translating the proof problem into Lean 4:
theorem binom_expansion_properties (x : ℤ) (h : x ≠ 0) :
  let expansion := binom_expansion (1 / x) x 6 in
  ( ∃ k: ℕ, k = 4 ∧ nat.choose 6 k = list.max (list.map (λ i, nat.choose 6 i) (finset.range 7).toList) ) ∧
  ( expansion.eval (1) = 0 ) :=
by
  sorry

end binom_expansion_properties_l399_399881


namespace conic_section_classification_l399_399259

-- Define the given equation as a condition.
def equation (x y : ℝ) := abs (y - 3) = real.sqrt ((x + 4)^2 + (y + 2)^2)

-- State the theorem that represents the problem.
theorem conic_section_classification : 
  ∀ x y : ℝ,
  equation x y →
  Type ≠ "C" ∧ Type ≠ "P" ∧ Type ≠ "E" ∧ Type ≠ "H" ∧ Type = "N" :=
  sorry

end conic_section_classification_l399_399259


namespace original_deck_card_count_l399_399288

variable (r b : ℕ)

theorem original_deck_card_count (h1 : r / (r + b) = 1 / 4) (h2 : r / (r + b + 6) = 1 / 6) : r + b = 12 :=
by
  -- The proof goes here
  sorry

end original_deck_card_count_l399_399288


namespace smallest_n_l399_399137

theorem smallest_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) : n > 15 → n = 52 :=
by
  intro h3
  have h4 : n = 52, from sorry
  exact h4

end smallest_n_l399_399137


namespace hundredth_term_is_981_l399_399536

noncomputable def base3_representation (n : ℕ) : list ℕ :=
  if h : n = 0 then [] else
    let rec base3_rep n r :=
      if n = 0 then r
      else base3_rep (n / 3) ((n % 3) :: r)
    in base3_rep n []

def is_valid_sequence_number (n : ℕ) : Prop :=
  ∀ d ∈ base3_representation n, d = 0 ∨ d = 1

noncomputable def nth_valid_sequence_number (k : ℕ) : ℕ :=
  let rec find_nth n c :=
      if c = k then n
      else if is_valid_sequence_number n then find_nth (n + 1) (c + 1)
      else find_nth (n + 1) c
    in find_nth 1 1

theorem hundredth_term_is_981 :
  nth_valid_sequence_number 100 = 981 :=
sorry

end hundredth_term_is_981_l399_399536


namespace monotonicity_and_range_of_a_l399_399078

noncomputable def f (a x : ℝ) : ℝ :=
  a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, x ∈ (set.Ioo 0 (1 / a)) → deriv (λ x, f a x) x < 0) ∧
  (∀ x, x ∈ (set.Ioi (1 / a)) → deriv (λ x, f a x) x > 0) ∧
  (∀ a, a > Real.exp (-1) → ∃ x : ℝ, ∀ x, f a x > 0) :=
by
  sorry

end monotonicity_and_range_of_a_l399_399078


namespace revenue_fall_percentage_l399_399957

theorem revenue_fall_percentage
  (old_revenue : ℝ)
  (new_revenue : ℝ)
  (h_old : old_revenue = 69.0)
  (h_new : new_revenue = 48.0) :
  ((old_revenue - new_revenue) / old_revenue) * 100 ≈ 30.43 :=
by
  sorry

end revenue_fall_percentage_l399_399957


namespace distance_between_vertices_l399_399754

def hyperbola_eq (x y : ℝ) : Prop := 
  x^2 / 144 - y^2 / 49 = 1

theorem distance_between_vertices : 2 * real.sqrt 144 = 24 :=
by {
    -- use sqrt calculation for clarity
    have h : real.sqrt 144 = 12, by {
        exact real.sqrt_eq_iff_sq_eq.mpr (or.inl (by norm_num)),
    },
    rw [h],
    norm_num
}

end distance_between_vertices_l399_399754


namespace distance_between_vertices_l399_399740

theorem distance_between_vertices (a b : ℝ) (a_pos : a = real.sqrt 144) (h : ∀ x y, x^2 / 144 - y^2 / 49 = 1): (2 * a) = 24 := by
  have ha : a = 12 := by sorry
  have h2a : 2 * a = 24 := by linarith
  exact h2a

end distance_between_vertices_l399_399740


namespace space_station_arrangement_l399_399976

theorem space_station_arrangement :
  ∃ (n : ℕ), (n = 6) ∧ ∃ (modules : ℕ), (modules = 3) ∧
  (∀ (a b c : ℕ), a + b + c = n → (1 ≤ a ∧ a ≤ 3) ∧ (1 ≤ b ∧ b ≤ 3) ∧ (1 ≤ c ∧ c ≤ 3) →
  (module_arrangements (a, b, c).fst (a, b, c).snd (a, b, c).trd = 450)) :=
begin
  sorry
end

end space_station_arrangement_l399_399976


namespace foci_ellipsoid_hyperboloid_l399_399964

theorem foci_ellipsoid_hyperboloid (a b : ℝ) 
(h1 : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → dist (0,y) (0, 5) = 5)
(h2 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → dist (x,0) (7, 0) = 7) :
  |a * b| = Real.sqrt 444 := sorry

end foci_ellipsoid_hyperboloid_l399_399964


namespace midpoint_trajectory_l399_399814

theorem midpoint_trajectory (x y : ℝ) (A_x A_y : ℝ) (hA : (A_x + 1)^2 + A_y^2 = 4) :
    let B_x := 8
    let B_y := 6
    let P_x := x
    let P_y := y
    let A_x := 2 * P_x - B_x
    let A_y := 2 * P_y - B_y
    (x - 7 / 2)^2 + (y - 3)^2 = 1 := 
begin
  -- The proof goes here
  sorry
end

end midpoint_trajectory_l399_399814


namespace transport_cost_correct_l399_399949

-- Defining the weights of the sensor unit and communication module in grams
def weight_sensor_grams : ℕ := 500
def weight_comm_module_grams : ℕ := 1500

-- Defining the transport cost per kilogram
def cost_per_kg_sensor : ℕ := 25000
def cost_per_kg_comm_module : ℕ := 20000

-- Converting weights to kilograms
def weight_sensor_kg : ℚ := weight_sensor_grams / 1000
def weight_comm_module_kg : ℚ := weight_comm_module_grams / 1000

-- Calculating the transport costs
def cost_sensor : ℚ := weight_sensor_kg * cost_per_kg_sensor
def cost_comm_module : ℚ := weight_comm_module_kg * cost_per_kg_comm_module

-- Total cost of transporting both units
def total_cost : ℚ := cost_sensor + cost_comm_module

-- Proving that the total cost is $42500
theorem transport_cost_correct : total_cost = 42500 := by
  sorry

end transport_cost_correct_l399_399949


namespace infinite_right_triangles_with_consecutive_hypotenuse_and_side_l399_399359

theorem infinite_right_triangles_with_consecutive_hypotenuse_and_side :
  ∃ᶠ (a b c : ℤ), (c = a + 1) ∧ (a * a + b * b = c * c) :=
sorry

end infinite_right_triangles_with_consecutive_hypotenuse_and_side_l399_399359


namespace negative_represents_opposite_l399_399879

variable {m : ℝ}

def represents_eastward (distance : ℝ) : Prop := distance > 0

theorem negative_represents_opposite (h : represents_eastward 60) : represents_eastward (-80) = false :=
by
  have pos_to_east := h
  have neg_direction : ∀ d, represents_eastward (-d) = false := 
    λ d, by sorry
  exact neg_direction 80

end negative_represents_opposite_l399_399879


namespace claire_balloons_l399_399701

theorem claire_balloons :
  let initial := 50
  let given_away_to_girl := 1
  let floated_away := 12
  let given_away_over_time := 9
  let received_from_coworker := 11
  initial - given_away_to_girl - floated_away - given_away_over_time + received_from_coworker = 39 :=
by
  let initial := 50
  let given_away_to_girl := 1
  let floated_away := 12
  let given_away_over_time := 9
  let received_from_coworker := 11
  show initial - given_away_to_girl - floated_away - given_away_over_time + received_from_coworker = 39 from sorry

end claire_balloons_l399_399701


namespace election_percentage_l399_399021

theorem election_percentage (total_votes : ℕ) (invalid_percentage : ℝ) (votes_for_candidate : ℕ) (valid_vote_percentage : ℝ) : 
  total_votes = 560000 → 
  invalid_percentage = 15 → 
  votes_for_candidate = 357000 → 
  valid_vote_percentage = 85 → 
  (votes_for_candidate.to_real / ((valid_vote_percentage / 100) * total_votes)) * 100 = 75 :=
by 
  intros h1 h2 h3 h4
  sorry

end election_percentage_l399_399021


namespace car_license_plate_count_l399_399868

theorem car_license_plate_count :
  let letters := 26
  let digits := 10
  let choose_digits := Nat.permutations 4 digits
  letters * letters * choose_digits = 26 * 26 * Nat.permutations 4 10 :=
by
  trivial -- to ensure the Lean code can be built successfully

end car_license_plate_count_l399_399868


namespace monotonicity_no_zeros_range_of_a_l399_399066

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * (Real.log x) + 1

theorem monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1/a → ∀ t : ℝ, t ∈ Ioo x y → f' a t < 0 ) ∧ 
  (∀ x y : ℝ, 1/a < x ∧ x < y → ∀ t : ℝ, t ∈ Ioo x y → f' a t > 0 ) :=
sorry

theorem no_zeros_range_of_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → f a x > 0) : 
  a > 1/Real.exp 1 :=
sorry

end monotonicity_no_zeros_range_of_a_l399_399066


namespace tan_of_A_l399_399038

theorem tan_of_A (A B C : Type) [IsTriangle A B C] (hC : angle C = 90) (hAB : distance A B = 13) (hBC : distance BC = 5) : 
  tangent A = 5 / 12 :=
by
  sorry

end tan_of_A_l399_399038


namespace dima_walking_speed_l399_399721

def Dima_station_time := 18 * 60 -- in minutes
def Dima_actual_arrival := 17 * 60 + 5 -- in minutes
def car_speed := 60 -- in km/h
def early_arrival := 10 -- in minutes

def walking_speed (arrival_time actual_arrival car_speed early_arrival : ℕ) : ℕ :=
(car_speed * early_arrival / 60) * (60 / (arrival_time - actual_arrival - early_arrival))

theorem dima_walking_speed :
  walking_speed Dima_station_time Dima_actual_arrival car_speed early_arrival = 6 :=
sorry

end dima_walking_speed_l399_399721


namespace square_roots_N_l399_399852

theorem square_roots_N (m N : ℤ) (h1 : (3 * m - 4) ^ 2 = N) (h2 : (7 - 4 * m) ^ 2 = N) : N = 25 := 
by
  sorry

end square_roots_N_l399_399852


namespace problem1_problem2_l399_399806

section
variable {α : Real}
variable (tan_α : Real)
variable (sin_α cos_α : Real)

def trigonometric_identities (tan_α sin_α cos_α : Real) : Prop :=
  tan_α = 2 ∧ sin_α = tan_α * cos_α

theorem problem1 (h : trigonometric_identities tan_α sin_α cos_α) :
  (4 * sin_α - 2 * cos_α) / (5 * cos_α + 3 * sin_α) = 6 / 11 := by
  sorry

theorem problem2 (h : trigonometric_identities tan_α sin_α cos_α) :
  (1 / 4 * sin_α^2 + 1 / 3 * sin_α * cos_α + 1 / 2 * cos_α^2) = 13 / 30 := by
  sorry
end

end problem1_problem2_l399_399806


namespace determine_length_DB_l399_399499

theorem determine_length_DB (AB CD AE : ℝ) (h_nonzero: AB ≠ 0 ∧ CD ≠ 0 ∧ AE ≠ 0)
  (angle_A_acute: ∃ A B C, ∠A < π/2 ∧ CD = height(AB, C) ∧ AE = height(BC, A)) :
  (∃ BD : ℝ, BD = length(AB, CD, AE) ∧ (BD ≠ 0)) ∨
  (∃ BD : ℝ, BD = length(AB, CD, AE) ∧ (BD = 0)) ∨
  (∀ BD : ℝ, BD ≠ length(AB, CD, AE)) ↔ E :=
sorry

end determine_length_DB_l399_399499


namespace total_charge_for_trip_l399_399271

-- Define the initial fee
def initial_fee : ℝ := 2.25

-- Define the charge per 2/5 mile increment
def charge_per_increment : ℝ := 0.15

-- Define the number of miles traveled
def miles_traveled : ℝ := 3.6

-- Define the number of increments in 3.6 miles
def increments_in_miles (miles : ℝ) : ℝ := miles * (5 / 2)

-- Define the total charge calculation
def total_charge (init_fee : ℝ) (charge_per_inc : ℝ) (miles : ℝ) : ℝ :=
  init_fee + charge_per_inc * increments_in_miles miles

-- State the theorem that the total charge is $4.68
theorem total_charge_for_trip :
  total_charge initial_fee charge_per_increment miles_traveled = 4.68 :=
by
  sorry

end total_charge_for_trip_l399_399271


namespace largest_number_with_two_moves_l399_399540

theorem largest_number_with_two_moves (n : Nat) (matches_limit : Nat) (initial_number : Nat)
  (h_n : initial_number = 1405) (h_limit: matches_limit = 2) : n = 7705 :=
by
  sorry

end largest_number_with_two_moves_l399_399540


namespace hyperbola_vertex_distance_l399_399773

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  (x^2 / 144 - y^2 / 49 = 1) →
  ∃ (a : ℝ), a = 12 ∧ 2 * a = 24 :=
by
  intro x y h
  have h1 : 12^2 = 144 := by norm_num
  use 12
  split
  case left =>
    exact rfl
  case right =>
    calc
      2 * 12 = 24 : by norm_num

end hyperbola_vertex_distance_l399_399773


namespace integer_solution_is_three_l399_399381

theorem integer_solution_is_three (n : ℤ) : (∃ k : ℤ, (n^3 - 3*n^2 + n + 2) = 5^k) ↔ n = 3 := 
by
  sorry

end integer_solution_is_three_l399_399381


namespace monotonicity_and_no_x_intercept_l399_399101

noncomputable theory

def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

def is_monotonic (a : ℝ) (x : ℝ) : Prop := 
  if x < 1 / a then 
    f a x > f a (x + ε) -- ε is a small positive value
  else if x > 1 / a then 
    f a x < f a (x - ε)
  else
    true -- At x = 1/a, the function transits from decreasing to increasing

theorem monotonicity_and_no_x_intercept 
  (a : ℝ) (h1 : 0 < a) : 
  (∀ x : ℝ, 0 < x → is_monotonic a x) ∧ 
  (∀ x : ℝ, f a x ≠ 0) ↔ 
  (a ∈ Ioi (1 / real.exp 1)) := 
sorry

end monotonicity_and_no_x_intercept_l399_399101


namespace taylor_ln_expansion_l399_399732

def f (x : ℝ) : ℝ := Real.log (1 - 2 * x)

def taylor_series_ln (x : ℝ) : ℝ :=
  ∑' (n : ℕ), if n > 0 then (-1 : ℝ)^n * (2 * x)^n / n else 0

theorem taylor_ln_expansion (x : ℝ) (h : x ∈ Icc (-1/2:ℝ) (1/2:ℝ)) :
  f x = taylor_series_ln x := by
  sorry

end taylor_ln_expansion_l399_399732


namespace polar_equation_correct_chord_length_correct_l399_399831

noncomputable def parametric_curve (alpha : ℝ) : ℝ × ℝ :=
  (3 + real.sqrt 10 * real.cos alpha, 1 + real.sqrt 10 * real.sin alpha)

noncomputable def polar_curve (theta : ℝ) : ℝ :=
  6 * real.cos theta + 2 * real.sin theta

theorem polar_equation_correct :
  ∀ (α : ℝ), ∃ θ ρ, parametric_curve α = (ρ * real.cos θ, ρ * real.sin θ) ∧ ρ = polar_curve θ := 
sorry

noncomputable def line_polar_eq (theta ρ : ℝ) : Prop :=
  sin theta - cos theta = 1 / ρ

def distance_from_center_to_line : ℝ := 3 * real.sqrt 2 / 2

def chord_length (r : ℝ) (d : ℝ) : ℝ :=
  real.sqrt (2 * r^2 - d^2)

theorem chord_length_correct :
  chord_length (real.sqrt 10) distance_from_center_to_line = real.sqrt 22 := 
sorry

end polar_equation_correct_chord_length_correct_l399_399831


namespace evaluate_expression_l399_399376

theorem evaluate_expression :
  (let x := (1 / 4 : ℚ), y := (3 / 4 : ℚ), z := (3 : ℚ) in x^2 * y^3 * z) = (81 / 1024 : ℚ) :=
by
  sorry

end evaluate_expression_l399_399376


namespace length_of_first_video_l399_399510

theorem length_of_first_video
  (total_time : ℕ)
  (second_video_time : ℕ)
  (last_two_videos_time : ℕ)
  (first_video_time : ℕ)
  (total_seconds : total_time = 510)
  (second_seconds : second_video_time = 4 * 60 + 30)
  (last_videos_seconds : last_two_videos_time = 60 + 60)
  (total_watch_time : total_time = second_video_time + last_two_videos_time + first_video_time) :
  first_video_time = 120 :=
by
  sorry

end length_of_first_video_l399_399510


namespace circle_condition_l399_399169

-- Define the given equation
def equation (m x y : ℝ) : Prop := x^2 + y^2 + 4 * m * x - 2 * y + 5 * m = 0

-- Define the condition for the equation to represent a circle
def represents_circle (m x y : ℝ) : Prop :=
  (x + 2 * m)^2 + (y - 1)^2 = 4 * m^2 - 5 * m + 1 ∧ 4 * m^2 - 5 * m + 1 > 0

-- The main theorem to be proven
theorem circle_condition (m : ℝ) : represents_circle m x y → (m < 1/4 ∨ m > 1) := 
sorry

end circle_condition_l399_399169


namespace hyperbola_foci_l399_399176

theorem hyperbola_foci :
  (∀ x y : ℝ, x^2 - 2 * y^2 = 1) →
  (∃ c : ℝ, c = (Real.sqrt 6) / 2 ∧ (x = c ∨ x = -c) ∧ y = 0) :=
by
  sorry

end hyperbola_foci_l399_399176


namespace prime_divides_expression_l399_399521

theorem prime_divides_expression (p : ℕ) (hp : Nat.Prime p) : ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) := 
by
  sorry

end prime_divides_expression_l399_399521


namespace max_and_min_f_when_a_one_g_correctness_l399_399464

noncomputable def f (x a : Real) : Real := x^2 - 2 * a * x + 3

theorem max_and_min_f_when_a_one :
  let a : Real := 1
  let f := f _ a
  (0 ≤ x ∧ x ≤ 2 → f x ≤ 11) ∧ (0 ≤ x ∧ x ≤ 2 → f x ≥ 2) :=
sorry

noncomputable def g (a : Real) : Real :=
  let M := if a ≤ -2 then 7 + 4 * a else 
           if a > -2 ∧ a <= 0 then 7 - 4 * a else 
           if a > 0 ∧ a < 2 then 7 + 4 * a else 
           7 + 4 * a -- Note: This should be revised correctly based on function analysis
  let m := if a ≤ -2 then 7 - 4 * a else 
           if a > -2 ∧ a <= 0 then -a^2 + 3 else 
           if a > 0 ∧ a < 2 then -a^2 + 3 else 
           7 - 4 * a -- Note: This should be revised correctly based on function analysis
  M - m

theorem g_correctness :
  ∀ a : Real, 
  g(a) = if a ≤ -2 then -8 * a else
         if -2 <a ∧ a <= 0 then a^2 - 4 * a + 4 else
         if 0 <a ∧ a < 2 then a^2 + 4 * a + 4 else 
         8 * a ∧
  (∀ a : Real, g(a) = 4) :=
sorry

end max_and_min_f_when_a_one_g_correctness_l399_399464


namespace count_numbers_with_digit_3_correct_l399_399846

def contains_digit_3 (n : ℕ) : Prop :=
  '3' ∈ n.digits 10.map Char.mk

def count_numbers_with_digit_3 (low high : ℕ) : ℕ :=
  (List.range' low (high - low + 1)).count contains_digit_3

theorem count_numbers_with_digit_3_correct :
  count_numbers_with_digit_3 200 499 = 138 := by
  sorry

end count_numbers_with_digit_3_correct_l399_399846


namespace percentage_decrease_of_larger_angle_l399_399189

noncomputable def complementary_angles_decrease_percentage : Real :=
let total_degrees := 90
let ratio_sum := 3 + 7
let part := total_degrees / ratio_sum
let smaller_angle := 3 * part
let larger_angle := 7 * part
let increased_smaller_angle := smaller_angle * 1.2
let new_larger_angle := total_degrees - increased_smaller_angle
let decrease_amount := larger_angle - new_larger_angle
(decrease_amount / larger_angle) * 100

theorem percentage_decrease_of_larger_angle
  (smaller_increased_percentage : Real := 20)
  (ratio_three : Real := 3)
  (ratio_seven : Real := 7)
  (total_degrees : Real := 90)
  (expected_decrease : Real := 8.57):
  complementary_angles_decrease_percentage = expected_decrease := 
sorry

end percentage_decrease_of_larger_angle_l399_399189


namespace circle_area_circumference_l399_399549

open Real

def point (x y : ℝ) : ℝ × ℝ := (x, y)
def distance (p1 p2 : ℝ × ℝ) : ℝ := sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def radius (A B : ℝ × ℝ) : ℝ := distance A B / 2

theorem circle_area_circumference
  (A B : ℝ × ℝ)
  (h_A : A = point 4 7)
  (h_B : B = point 8 12) :
  let r := radius A B in
  (π * r^2 = 41 * π / 4) ∧ (2 * π * r = sqrt 41 * π) :=
by
  sorry

end circle_area_circumference_l399_399549


namespace symmetry_one_common_point_l399_399971

theorem symmetry_one_common_point
  (a b c d : ℝ)
  (f g : ℝ → ℝ)
  (hx : ∀ x, f x = 2 * a + 1 / (x - b))
  (hy : ∀ x, g x = 2 * c + 1 / (x - d))
  (sym_point : ℝ × ℝ := (b + d) / 2, a + c)
  (symmetric : ∀ x, f (2 * sym_point.1 - x) = 2 * sym_point.2 - f x ∧ g (2 * sym_point.1 - x) = 2 * sym_point.2 - g x)
  (common_point : ∃ x, f x = g x)
  : (a - c) * (b - d) = 2 :=
begin
  sorry
end

end symmetry_one_common_point_l399_399971


namespace modified_sequence_sum_third_fifth_l399_399476

theorem modified_sequence_sum_third_fifth :
  let a := λ (n : ℕ), if n = 1 then 1 else ((n + 1) / n : ℚ) ^ 2
  in a 3 + a 5 = 724 / 225 := by
  sorry

end modified_sequence_sum_third_fifth_l399_399476


namespace molecular_weight_n2o_l399_399616

theorem molecular_weight_n2o (w : ℕ) (n : ℕ) (h : w = 352 ∧ n = 8) : (w / n = 44) :=
sorry

end molecular_weight_n2o_l399_399616


namespace cosine_of_angle_ACB_evaluation_l399_399884

variable {A B C D O : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited O]
variable (ABC_triangle : Triangle A B C)
variable (D_on_BC : PointOnLine D B C)
variable (AD_line : Line A D)
variable (C_bisector : AngleBisector C)
variable (circle_center_on_AC : CircleCenterOnSide A C)
variable (AC_AB_ratio : Ratio AC AB 3 2)
variable (DAC_DAB_angle_ratio : AngleRatio DAC DAB 3)

theorem cosine_of_angle_ACB_evaluation :
  ∃ (cos_val : ℝ), cos_val = (2 / sqrt 7) :=
sorry

end cosine_of_angle_ACB_evaluation_l399_399884


namespace math_equiv_proof_l399_399734

theorem math_equiv_proof :
  (Real.ceil ((12:ℝ) / 7 * (-29) / 3) - Real.floor ((12:ℝ) / 7 * Real.floor ((-29:ℝ) / 3))) = 2 :=
by
  sorry

end math_equiv_proof_l399_399734


namespace length_of_rod_l399_399006

theorem length_of_rod (w1 w2 l1 l2 : ℝ) (h_uniform : ∀ m n, m * w1 = n * w2) (h1 : w1 = 42.75) (h2 : l1 = 11.25) : 
  l2 = 6 := 
  by
  have wpm := w1 / l1
  have h3 : 22.8 / wpm = l2 := by sorry
  rw [h1, h2] at *
  simp at *
  sorry

end length_of_rod_l399_399006


namespace real_animal_count_l399_399539

variable (Mary_counted_animals : ℕ)
variable (double_counted_sheep : ℕ)
variable (forgotten_pigs : ℕ)
variable (corrected_animals : ℕ)

theorem real_animal_count (Mary_counted_animals = 60) (double_counted_sheep = 7) (forgotten_pigs = 3) :
  corrected_animals = Mary_counted_animals - double_counted_sheep + forgotten_pigs := by
  sorry

end real_animal_count_l399_399539


namespace complex_seventh_root_of_unity_l399_399906

theorem complex_seventh_root_of_unity (r : ℂ) (h1 : r^7 = 1) (h2: r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by
  sorry

end complex_seventh_root_of_unity_l399_399906


namespace loss_percentage_is_21_08_l399_399287

def original_cost := 1500
def sales_tax_rate := 0.10
def shipping_fee := 300

def selling_price := 1620
def seller_commission_rate := 0.05

-- total cost price with sales tax and shipping fee
def total_cost_price := original_cost * (1 + sales_tax_rate) + shipping_fee

-- selling price after seller's commission
def net_selling_price := selling_price * (1 - seller_commission_rate)

-- loss calculation
def loss := total_cost_price - net_selling_price

-- loss percentage
def loss_percentage := ((loss / total_cost_price) * 100 : ℚ)

theorem loss_percentage_is_21_08 :
  loss_percentage = 21.08 := by
  sorry

end loss_percentage_is_21_08_l399_399287


namespace system1_solution_system2_solution_l399_399563

-- Statement for the Part 1 Equivalent Problem.
theorem system1_solution :
  ∀ (x y : ℤ),
    (x - 3 * y = -10) ∧ (x + y = 6) → (x = 2 ∧ y = 4) :=
by
  intros x y h
  rcases h with ⟨h1, h2⟩
  sorry

-- Statement for the Part 2 Equivalent Problem.
theorem system2_solution :
  ∀ (x y : ℚ),
    (x / 2 - (y - 1) / 3 = 1) ∧ (4 * x - y = 8) → (x = 12 / 5 ∧ y = 8 / 5) :=
by
  intros x y h
  rcases h with ⟨h1, h2⟩
  sorry

end system1_solution_system2_solution_l399_399563


namespace constant_term_expansion_l399_399575

theorem constant_term_expansion :
  let f := (x^2 + 2) * (1/x - 1)^5
  is_constant_term (f) = -12 :=
by
  -- Definitions and calculations can be done here
  sorry

end constant_term_expansion_l399_399575


namespace probability_winning_ticket_l399_399129

noncomputable def log_base (b x : ℕ) : ℚ := Real.log x / Real.log b

theorem probability_winning_ticket (S : Finset ℕ) (h₁ : ∀ n ∈ S, 1 ≤ n ∧ n ≤ 60) 
  (h₂ : S.card = 5) (h₃ : ∑ n in S, log_base 12 n ∈ ℤ)
  (W : Finset ℕ) (h₄ : ∀ m ∈ W, 1 ≤ m ∧ m ≤ 60) (h₅ : W.card = 5) 
  (h₆ : ∑ m in W, log_base 12 m ∈ ℤ) : 
  W = S :=
sorry

end probability_winning_ticket_l399_399129


namespace monotonicity_and_no_real_roots_l399_399094

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

theorem monotonicity_and_no_real_roots 
  (a : ℝ) (ha : 0 < a) : 
  (∀ x : ℝ, (0 < x ∧ x < (1 / a) → deriv (f a) x < 0) ∧ (x > (1 / a) → deriv (f a) x > 0)) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≠ 0 → a > (1 / real.exp 1))) :=
begin
  sorry
end

end monotonicity_and_no_real_roots_l399_399094


namespace sum_pq_l399_399110

theorem sum_pq (p q : ℝ) (h1 : p ≠ q) (h2 : p > q)
  (h3 : |2 -5 8, 1 p q, 1 q p| = 0) : p + q = -13 / 2 := sorry

end sum_pq_l399_399110


namespace percentage_answered_first_correctly_l399_399285

variable (A B C D : ℝ)

-- Conditions translated to Lean
variable (hB : B = 0.65)
variable (hC : C = 0.20)
variable (hD : D = 0.60)

-- Statement to prove
theorem percentage_answered_first_correctly (hI : A + B - D = 1 - C) : A = 0.75 := by
  -- import conditions
  rw [hB, hC, hD] at hI
  -- solve the equation
  sorry

end percentage_answered_first_correctly_l399_399285


namespace duke_extra_three_pointers_l399_399453

theorem duke_extra_three_pointers
    (old_record : ℕ)
    (points_needed_to_tie : ℕ)
    (exceeded_points : ℕ)
    (free_throws : ℕ)
    (free_throw_value : ℕ)
    (regular_baskets : ℕ)
    (regular_basket_value : ℕ)
    (normal_three_pointers : ℕ)
    (three_pointer_value : ℕ)
    (final_game_points : ℕ) :
    old_record = 257 →
    points_needed_to_tie = 17 →
    exceeded_points = 5 →
    free_throws = 5 →
    free_throw_value = 1 →
    regular_baskets = 4 →
    regular_basket_value = 2 →
    normal_three_pointers = 2 →
    three_pointer_value = 3 →
    final_game_points = points_needed_to_tie + exceeded_points →
    (final_game_points - (free_throws * free_throw_value + regular_baskets * regular_basket_value)) / three_pointer_value - normal_three_pointers = 1 :=
begin
    intros h_old_record h_points_needed_to_tie h_exceeded_points h_free_throws h_free_throw_value h_regular_baskets h_regular_basket_value h_normal_three_pointers h_three_pointer_value h_final_game_points,
    rw [h_old_record, h_points_needed_to_tie, h_exceeded_points, h_free_throws, h_free_throw_value, h_regular_baskets, h_regular_basket_value, h_normal_three_pointers, h_three_pointer_value, h_final_game_points],
    sorry
end

end duke_extra_three_pointers_l399_399453


namespace count_distinct_lines_l399_399122

-- Definitions of the conditions
def is_from_set (x : ℤ) : Prop :=
  x ∈ {-3, -2, -1, 0, 1, 2, 3}

def distinct_three (a b c : ℤ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ b ≠ c

def acute_angle_line (a b : ℤ) : Prop :=
  (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0)

-- The formal statement of the proof problem
theorem count_distinct_lines : 
  ∃ n, n = 88 ∧ 
       ∀ (a b c : ℤ), 
       is_from_set a → is_from_set b → is_from_set c →
       distinct_three a b c →
       acute_angle_line a b → n = 88 :=
sorry

end count_distinct_lines_l399_399122


namespace find_angle_B_correct_max_value_expression_correct_l399_399534

variable {A B C a b c S : ℝ}

noncomputable def find_angle_B (S a b c : ℝ) : ℝ :=
  if S = (sqrt 3 / 4) * (a^2 + c^2 - b^2) then π/3 else 0

theorem find_angle_B_correct (h : S = (sqrt 3 / 4) * (a^2 + c^2 - b^2)) :
  find_angle_B S a b c = π / 3 := 
by sorry

noncomputable def max_value_expression (b : ℝ) (f : ℝ → ℝ) : ℝ :=
  if b = sqrt 3 then sup (set.range f) else 0

theorem max_value_expression_correct (S : ℝ) (h : S = (sqrt 3 / 4) * (a^2 + c^2 - b^2)) (h_b : b = sqrt 3)
  (A : ℝ → ℝ) (f : ℝ → ℝ) :
  max_value_expression b f = 2 * sqrt 6 := 
by sorry

end find_angle_B_correct_max_value_expression_correct_l399_399534


namespace solution_part1_solution_part2_l399_399448

noncomputable def pointM := (-2 : ℝ, 0 : ℝ)
noncomputable def pointN := (2 : ℝ, 0 : ℝ)

def is_on_curve_C (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in y^2 = -8 * x

def is_moving_point_P (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  (real.sqrt ((x + 2)^2 + y^2) + real.sqrt ((x - 2)^2 + y^2)) = 0

def curve_C_equation : Prop :=
  ∀ P : ℝ × ℝ, is_moving_point_P P ↔ is_on_curve_C P

def line_l (k : ℝ) : ℝ × ℝ → Prop :=
  λ P, let (x, y) := P in y = k * (x - 2)

def is_in_second_quadrant (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in x < 0 ∧ y > 0

def perpendicular_bisector (S T : ℝ × ℝ) : ℝ → ℝ :=
  λ x, -x + (fst S + snd S + fst T + snd T) / 2

def x_coordinate_Q (S T : ℝ × ℝ) : ℝ :=
  let B := midpoint S T in
  -b - 2

def second_quadrant_intersection (S T : ℝ × ℝ) (_ : is_in_second_quadrant S) (_ : is_in_second_quadrant T) :
  x_coordinate_Q S T < -6 :=
sorry

theorem solution_part1 : curve_C_equation :=
sorry

theorem solution_part2 (k : ℝ) (S T : ℝ × ℝ) (h1 : line_l k S) (h2 : line_l k T) (hs : is_in_second_quadrant S) (ht : is_in_second_quadrant T) :
  -1 < k ∧ k < 0 → second_quadrant_intersection S T hs ht :=
sorry

end solution_part1_solution_part2_l399_399448


namespace arithmetic_mean_reciprocals_first_four_primes_l399_399336

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l399_399336


namespace compute_expression_l399_399355

theorem compute_expression :
  let a := 21
  let b := 23
  (∏ i in Finset.range (a + 1) \set \{0}, (1 + a / i)) / (∏ i in Finset.range (b), (1 + b / i)) = 1 :=
by
  sorry

end compute_expression_l399_399355


namespace harmonic_equivalent_l399_399524

variables {A B C D X Y : Type}
variables [cyclic_quadrilateral A B C D]

-- Define the cyclic quadrilateral property
class cyclic_quadrilateral (A B C D : Type) :=
  (opposite_angles_sum_180 : ∀ (α β : Prop), α ∧ β → ∡α + ∡β = 180)

-- Define the existence of points X and Y satisfying the required angle conditions
def X_exists (X : Type) 
  (HX : X ∈ segment BD) : Prop :=
  angle BAC = angle XAD ∧ angle BCA = angle XCD

def Y_exists (Y : Type) 
  (HY : Y ∈ segment AC) : Prop :=
  angle CBD = angle YBA ∧ angle CDB = angle YDA

theorem harmonic_equivalent
  (h1 : ∃ (X : Type), X_exists X)
  (h2 : ∃ (Y : Type), Y_exists Y) :
  (∃ (X : Type), X_exists X) ↔ (∃ (Y : Type), Y_exists Y) :=
begin
  sorry
end

end harmonic_equivalent_l399_399524


namespace yellow_scores_l399_399600

theorem yellow_scores (W B : ℕ) 
  (h₁ : W / B = 7 / 6)
  (h₂ : (2 / 3 : ℚ) * (W - B) = 4) : 
  W + B = 78 :=
sorry

end yellow_scores_l399_399600


namespace probability_all_white_balls_drawn_l399_399284

noncomputable def binomial_coefficient (n k : ℕ) : ℚ :=
  (nat.factorial n).to_rat / ((nat.factorial k).to_rat * (nat.factorial (n - k)).to_rat)

theorem probability_all_white_balls_drawn :
  let whiteBalls := 8
  let blackBalls := 10
  let totalBalls := whiteBalls + blackBalls
  let drawnBalls := 7
  let prob := (binomial_coefficient whiteBalls drawnBalls / binomial_coefficient totalBalls drawnBalls)
  prob = 1 / 3980 := by
  sorry

end probability_all_white_balls_drawn_l399_399284


namespace monotonicity_no_zeros_range_of_a_l399_399061

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * (Real.log x) + 1

theorem monotonicity (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 1/a → ∀ t : ℝ, t ∈ Ioo x y → f' a t < 0 ) ∧ 
  (∀ x y : ℝ, 1/a < x ∧ x < y → ∀ t : ℝ, t ∈ Ioo x y → f' a t > 0 ) :=
sorry

theorem no_zeros_range_of_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → f a x > 0) : 
  a > 1/Real.exp 1 :=
sorry

end monotonicity_no_zeros_range_of_a_l399_399061


namespace parabola_zeros_difference_l399_399981

theorem parabola_zeros_difference :
  ∃ (a b c p q : ℝ),
    (∀ x, a * x ^ 2 + b * x + c = 4.5 * (x - 3) ^ 2 - 3) ∧
      ((3 + sqrt 6 / 3, 3 - sqrt 6 / 3)) ∈ {r : ℝ × ℝ | p > q ∧ p = r.1 ∧ q = r.2} ∧
      (p - q = 2 * sqrt 6 / 3) := sorry

end parabola_zeros_difference_l399_399981


namespace carB_speed_l399_399959

variable (distance : ℝ) (time : ℝ) (ratio : ℝ) (speedB : ℝ)

theorem carB_speed (h1 : distance = 240) (h2 : time = 1.5) (h3 : ratio = 3 / 5) 
(h4 : (speedB + ratio * speedB) * time = distance) : speedB = 100 := 
by 
  sorry

end carB_speed_l399_399959


namespace hyperbola_vertex_distance_l399_399772

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  (x^2 / 144 - y^2 / 49 = 1) →
  ∃ (a : ℝ), a = 12 ∧ 2 * a = 24 :=
by
  intro x y h
  have h1 : 12^2 = 144 := by norm_num
  use 12
  split
  case left =>
    exact rfl
  case right =>
    calc
      2 * 12 = 24 : by norm_num

end hyperbola_vertex_distance_l399_399772


namespace complementary_angles_ratio_decrease_l399_399204

theorem complementary_angles_ratio_decrease 
  (a b : ℝ) (h_ratio : a / b = 3 / 7) (h_comp : a + b = 90) : 
  let a' := a * 1.20 in
  let b' := 90 - a' in
  ((b' / b) * 100 = 91.43) ∧ (100 - (b' / b) * 100 = 8.57) :=
by
  have : a / b = 3 / 7 := h_ratio
  have : a + b = 90 := h_comp
  let a' := a * 1.20
  let b' := 90 - a'
  have : b' / b = 57.6 / 63 := sorry -- Calculations omitted for brevity
  have : (b' / b) * 100 = 91.43 := by sorry
  have : 100 - (b' / b) * 100 = 8.57 := by sorry
  split;
  assumption

end complementary_angles_ratio_decrease_l399_399204


namespace michelle_needs_more_racks_l399_399252

def cups_of_flour_per_bag : ℕ := 12
def number_of_bags : ℕ := 6
def flour_type1_per_pound : ℕ := 3
def flour_type2_per_pound : ℕ := 4
def initial_racks : ℕ := 2
def pounds_per_rack : ℕ := 5

theorem michelle_needs_more_racks : 
  let total_flour := number_of_bags * cups_of_flour_per_bag,
      flour_per_type := total_flour / 2,
      total_pounds_of_pasta := (flour_per_type / flour_type1_per_pound) + (flour_per_type / flour_type2_per_pound),
      total_racks_needed := (total_pounds_of_pasta + pounds_per_rack - 1) / pounds_per_rack
  in total_racks_needed - initial_racks = 3 := by
  sorry

end michelle_needs_more_racks_l399_399252


namespace y_coordinate_of_M_PQ_passes_through_fixed_point_minimum_area_l399_399810

-- Define the conditions for points P and Q on the parabola
def on_parabola (P Q : ℝ × ℝ) : Prop :=
  P.2 = P.1^2 ∧ Q.2 = Q.1^2

-- Define the tangents at P and Q and the perpendicularity condition
def tangents_perpendicular (P Q : ℝ × ℝ) : Prop :=
  let l1_slope := 2 * P.1
  let l2_slope := 2 * Q.1
  l1_slope * l2_slope = -1

-- Define the intersection point M of the tangents
def intersection_point (P Q M : ℝ × ℝ) : Prop :=
  M.1 = (P.1 + Q.1) / 2 ∧ M.2 = P.1 * Q.1

-- Proof problem 1: y-coordinate of M is -1/4
theorem y_coordinate_of_M (P Q M : ℝ × ℝ):
  on_parabola P Q → tangents_perpendicular P Q → intersection_point P Q M → M.2 = -1/4 :=
  sorry

-- Proof problem 2: PQ passes through a fixed point
theorem PQ_passes_through_fixed_point (P Q : ℝ × ℝ):
  on_parabola P Q → tangents_perpendicular P Q → ∃ (C : ℝ × ℝ), ∀ (x : ℝ), (x, x^2) = P ∨ (x, x^2) = Q → ∃a b : ℝ, (x = a * x + b) = (x = C.1 * x + C.2) :=
  sorry

-- Proof problem 3: Minimum area of triangle PQM
theorem minimum_area (P Q M : ℝ × ℝ):
  on_parabola P Q → tangents_perpendicular P Q → intersection_point P Q M → ∃ (A : ℝ), area_triangle PQ M = 1/4 :=
  sorry

end y_coordinate_of_M_PQ_passes_through_fixed_point_minimum_area_l399_399810


namespace monotonicity_and_range_of_a_l399_399077

noncomputable def f (a x : ℝ) : ℝ :=
  a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, x ∈ (set.Ioo 0 (1 / a)) → deriv (λ x, f a x) x < 0) ∧
  (∀ x, x ∈ (set.Ioi (1 / a)) → deriv (λ x, f a x) x > 0) ∧
  (∀ a, a > Real.exp (-1) → ∃ x : ℝ, ∀ x, f a x > 0) :=
by
  sorry

end monotonicity_and_range_of_a_l399_399077


namespace Z_divisible_by_1001_l399_399056

/-- 
  Let Z be a 7-digit positive integer in the form abcabca, where a, b, and c are digits
  and a ≠ 0. Prove that Z is divisible by 1001.
-/
theorem Z_divisible_by_1001 (a b c : ℕ) (h : 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9) :
  let Z := 1000000 * a + 100000 * b + 10000 * c + 1000 * a + 100 * b + 10 * c + a 
  in 1001 ∣ Z :=
sorry

end Z_divisible_by_1001_l399_399056


namespace monotonicity_of_f_range_of_a_if_no_zeros_l399_399069

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end monotonicity_of_f_range_of_a_if_no_zeros_l399_399069


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l399_399345

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  let primes := [2, 3, 5, 7] in
  let reciprocals := primes.map (λ p => (1 : ℚ) / p) in
  let sum_reciprocals := reciprocals.sum in
  let mean := sum_reciprocals / (primes.length : ℚ) in
  mean = 247 / 840 :=
by
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p => (1 : ℚ) / p)
  let sum_reciprocals := reciprocals.sum
  let mean := sum_reciprocals / (primes.length : ℚ)
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l399_399345


namespace complex_root_product_value_l399_399907

noncomputable def complex_root_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : ℂ :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1)

theorem complex_root_product_value (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) : complex_root_product r h1 h2 = 14 := 
  sorry

end complex_root_product_value_l399_399907


namespace product_sequence_l399_399729

theorem product_sequence : ( ∏ n in Finset.range 98, (1 - 1 / (n + 2 : ℝ))) = 1 / 99 :=
by
  sorry

end product_sequence_l399_399729


namespace arithmetic_mean_reciprocals_primes_l399_399325

theorem arithmetic_mean_reciprocals_primes
  (p : Finset ℕ)
  (h_p : p = {2, 3, 5, 7})
  : (p.sum (λ x, 1 / ↑x) / 4) = (247 / 840) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l399_399325


namespace part1_part2_l399_399793

section
variables {m : ℝ} (z : ℂ)
-- Define the complex number z
def z_def := (6 - 4 * m * Complex.I) / (1 + Complex.I)

-- Statement Part (1): If z is real, then m = -3/2
theorem part1 (hm : Im (z_def z) = 0) : m = -3 / 2 :=
sorry

-- Conjugate of z
def z_conj := conj (z_def z)

-- Statement Part (2): If Conjugate of z - 4z is in the first quadrant, then m > 3/2
theorem part2 (hq : Re (z_conj z - 4 * z_def z) > 0 ∧ Im (z_conj z - 4 * z_def z) > 0) : m > 3 / 2 :=
sorry
end

end part1_part2_l399_399793


namespace find_Y_value_l399_399853

-- Define the conditions
def P : ℕ := 4020 / 4
def Q : ℕ := P * 2
def Y : ℤ := P - Q

-- State the theorem
theorem find_Y_value : Y = -1005 := by
  -- Proof goes here
  sorry

end find_Y_value_l399_399853


namespace max_fat_triangles_l399_399675

-- Define the number of points in set P
def num_points : ℕ := 2021

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the conditions for fat triangles
def is_fat_triangle_condition (O : Point) (P : set Point) (triangle : Triangle) : Prop :=
  O ∈ triangle.interior

-- Define the main theorem: the maximum number of fat triangles
theorem max_fat_triangles (O : Point) (P : finset Point)
  (h1 : P.card = num_points)
  (h2 : ∀ (A B C : Point), A ∈ P → B ∈ P → C ∈ P → A ≠ B → B ≠ C → A ≠ C → ¬ collinear {A, B, C})
  (h3 : ∀ (A B : Point), A ∈ P → B ∈ P → A ≠ B → ¬ collinear {O, A, B}) :
  (∑ t in P.triangle_set, if is_fat_triangle_condition O P t then 1 else 0) =
  binom num_points 3 - num_points * binom 1010 2 :=
sorry

end max_fat_triangles_l399_399675


namespace arithmetic_sequence_sum_of_first_10_terms_l399_399573

noncomputable def arithmetic_sequence_sum : ℕ → ℕ → ℝ := 
λ n a1, (n * (2 * a1 + (n - 1))) / 2

theorem arithmetic_sequence_sum_of_first_10_terms 
  (a_n : ℕ → ℝ) 
  (h1 : ∀ n, a_n n > 0) 
  (h2 : a_n 3 ^ 2 + a_n 6 ^ 2 + 2 * a_n 3 * a_n 6 = 9) : 
  ∑ i in finset.range 10, a_n i = 15 :=
by
  sorry

end arithmetic_sequence_sum_of_first_10_terms_l399_399573


namespace complementary_angle_decrease_l399_399214

theorem complementary_angle_decrease (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3 / 7) : 
  (∃ new_α : ℝ, new_α = α * 1.2) →
  ∃ new_β : ℝ, new_β = (1 - 0.0857) * β :=
by
  intro h3
  cases h3 with new_α h_newα
  use 90 - new_α
  sorry

end complementary_angle_decrease_l399_399214


namespace sqrt_sum_eq_nine_point_six_l399_399947

variable (y : ℝ)

/- Given conditions -/
def condition (y : ℝ) : Prop :=
  sqrt (64 - y^2) - sqrt (16 - y^2) = 5

/- Statement to prove -/
theorem sqrt_sum_eq_nine_point_six (hy : condition y) : 
  sqrt (64 - y^2) + sqrt (16 - y^2) = 9.6 :=
sorry

end sqrt_sum_eq_nine_point_six_l399_399947


namespace olivia_money_left_l399_399544

-- Defining hourly wages
def wage_monday : ℕ := 10
def wage_wednesday : ℕ := 12
def wage_friday : ℕ := 14
def wage_saturday : ℕ := 20

-- Defining hours worked each day
def hours_monday : ℕ := 5
def hours_wednesday : ℕ := 4
def hours_friday : ℕ := 3
def hours_saturday : ℕ := 2

-- Defining business-related expenses and tax rate
def expenses : ℕ := 50
def tax_rate : ℝ := 0.15

-- Calculate total earnings
def total_earnings : ℕ :=
  (hours_monday * wage_monday) +
  (hours_wednesday * wage_wednesday) +
  (hours_friday * wage_friday) +
  (hours_saturday * wage_saturday)

-- Earnings after expenses
def earnings_after_expenses : ℕ :=
  total_earnings - expenses

-- Calculate tax amount
def tax_amount : ℝ :=
  tax_rate * (total_earnings : ℝ)

-- Final amount Olivia has left
def remaining_amount : ℝ :=
  (earnings_after_expenses : ℝ) - tax_amount

theorem olivia_money_left : remaining_amount = 103 := by
  sorry

end olivia_money_left_l399_399544


namespace range_of_a_l399_399112

noncomputable def proposition_p (a : ℝ) : Prop :=
  16 * (a - 1) * (a - 3) < 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  a^2 - 4 < 0

theorem range_of_a (a : ℝ) : proposition_p a ∨ proposition_q a → -2 < a ∧ a < 3 :=
begin
  sorry
end

end range_of_a_l399_399112


namespace arithmetic_mean_reciprocals_first_four_primes_l399_399339

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l399_399339


namespace cos_eq_solution_is_valid_l399_399150

noncomputable def solve_cos_eq (x : ℝ) : Prop :=
  ∃ (n : ℤ), x = 90 + n * 180 ∨
             x = 30 + n * 180 ∨
             x = 210 + n * 180 ∨
             x = 150 + n * 180 ∨
             x = 330 + n * 180 ∨
             x = 45 + n * 180 ∨
             x = 225 + n * 180 ∨
             x = 135 + n * 180 ∨
             x = 315 + n * 180

theorem cos_eq_solution_is_valid (x : ℝ) (h : 0 ≤ x ∧ x < 360) :
  (cos x) ^ 2 + (cos (2 * x)) ^ 2 + (cos (3 * x)) ^ 2 = 1 ↔ solve_cos_eq x :=
sorry

end cos_eq_solution_is_valid_l399_399150


namespace a_minus_b_eq_one_l399_399807

variable (a b : ℕ)

theorem a_minus_b_eq_one
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : Real.sqrt 18 = a * Real.sqrt 2) 
  (h4 : Real.sqrt 8 = 2 * Real.sqrt b) : 
  a - b = 1 := 
sorry

end a_minus_b_eq_one_l399_399807


namespace molly_age_is_63_l399_399141

variable (Sandy_age Molly_age : ℕ)

theorem molly_age_is_63 (h1 : Sandy_age = 49) (h2 : Sandy_age / Molly_age = 7 / 9) : Molly_age = 63 :=
by
  sorry

end molly_age_is_63_l399_399141


namespace planes_distance_correct_l399_399737

def distance_between_planes (n : ℝ × ℝ × ℝ) (d1 d2 : ℝ) : ℝ :=
  let (a, b, c) := n in
  (abs (d2 - d1)) / (real.sqrt (a*a + b*b + c*c))

def planes_distance_proof : Prop := 
  let n := (2 : ℝ, -3 : ℝ, 1 : ℝ) in
  let d1 := -1 in
  let d2 := 4 in
  distance_between_planes n d1 d2 = 5 / real.sqrt 14

theorem planes_distance_correct : planes_distance_proof :=
by sorry

end planes_distance_correct_l399_399737


namespace washer_total_cost_l399_399115

variable (C : ℝ)
variable (h : 0.25 * C = 200)

theorem washer_total_cost : C = 800 :=
by
  sorry

end washer_total_cost_l399_399115


namespace hyperbola_vertex_distance_l399_399766

theorem hyperbola_vertex_distance : 
  ∀ x y: ℝ, (x^2 / 144 - y^2 / 49 = 1) → (∃ a: ℝ, a = 12 ∧ 2 * a = 24) :=
by 
  sorry

end hyperbola_vertex_distance_l399_399766


namespace union_of_sets_l399_399446

open Set

-- Define the sets A and B
def A : Set ℤ := {-2, 0}
def B : Set ℤ := {-2, 3}

-- Prove that the union of A and B equals {–2, 0, 3}
theorem union_of_sets : A ∪ B = {-2, 0, 3} := by
  sorry

end union_of_sets_l399_399446


namespace digits_same_in_prime_power_l399_399528

theorem digits_same_in_prime_power
  (p : ℕ) (n : ℕ)
  (hp_prime : Nat.Prime p)
  (hp_gt_3 : p > 3)
  (hp_pow_20_digit : Nat.digits 10 (p ^ n) = 20) :
  ∃ d : ℕ, ∃ k ≥ 3, List.count d (Nat.digits 10 (p ^ n)) = k :=
sorry

end digits_same_in_prime_power_l399_399528


namespace number_of_exercise_books_l399_399930

variable (E P : ℕ)
variable (ratio_exercise_books : ℕ) := 3
variable (ratio_pencils : ℕ) := 14
variable (number_of_pencils : ℕ) := 140

theorem number_of_exercise_books : E = 30 :=
by 
  -- Define the given conditions
  have h1 : P = number_of_pencils := rfl
  have h2 : ratio_pencils * E = ratio_exercise_books * P := by
    calc
      ratio_pencils * E = ratio_pencils * 30 := by -- substituting E = 30
        sorry
      ... = ratio_exercise_books * number_of_pencils := by
        sorry
  -- Assert and prove E = 30 using the conditions
  sorry

end number_of_exercise_books_l399_399930


namespace sum_of_roots_is_three_l399_399858

theorem sum_of_roots_is_three :
  ∀ (x1 x2 : ℝ), (x1^2 - 3 * x1 - 4 = 0) ∧ (x2^2 - 3 * x2 - 4 = 0) → x1 + x2 = 3 :=
by sorry

end sum_of_roots_is_three_l399_399858


namespace monotonicity_and_range_of_a_l399_399074

noncomputable def f (a x : ℝ) : ℝ :=
  a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range_of_a (a : ℝ) (h : a > 0) :
  (∀ x, x ∈ (set.Ioo 0 (1 / a)) → deriv (λ x, f a x) x < 0) ∧
  (∀ x, x ∈ (set.Ioi (1 / a)) → deriv (λ x, f a x) x > 0) ∧
  (∀ a, a > Real.exp (-1) → ∃ x : ℝ, ∀ x, f a x > 0) :=
by
  sorry

end monotonicity_and_range_of_a_l399_399074


namespace probability_all_evens_before_first_odd_l399_399666

theorem probability_all_evens_before_first_odd : 
  let prob_even := 1 / 2
  let prob_odd := 1 / 2
  let required_probability := 1 / 20
  ∃ n : ℕ, 
    (∀ k < n, (k ≠ 2 ∧ k ≠ 4 ∧ k ≠ 6)) ∧ (nth_roll n = 1 ∨ nth_roll n = 3 ∨ nth_roll n = 5) → 
      (probability_event all_evens_before_first_odd = required_probability) := sorry

end probability_all_evens_before_first_odd_l399_399666


namespace arithmetic_mean_reciprocals_primes_l399_399330

theorem arithmetic_mean_reciprocals_primes : 
  let p := [2, 3, 5, 7] in 
  let reciprocals := p.map (λ n => 1 / (n : ℚ)) in
  (reciprocals.sum / reciprocals.length) = (247 / 840 : ℚ) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l399_399330


namespace monotonicity_and_range_l399_399084

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end monotonicity_and_range_l399_399084


namespace monotonicity_and_no_x_intercept_l399_399100

noncomputable theory

def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

def is_monotonic (a : ℝ) (x : ℝ) : Prop := 
  if x < 1 / a then 
    f a x > f a (x + ε) -- ε is a small positive value
  else if x > 1 / a then 
    f a x < f a (x - ε)
  else
    true -- At x = 1/a, the function transits from decreasing to increasing

theorem monotonicity_and_no_x_intercept 
  (a : ℝ) (h1 : 0 < a) : 
  (∀ x : ℝ, 0 < x → is_monotonic a x) ∧ 
  (∀ x : ℝ, f a x ≠ 0) ↔ 
  (a ∈ Ioi (1 / real.exp 1)) := 
sorry

end monotonicity_and_no_x_intercept_l399_399100


namespace prove_angles_equal_dot_product_BC_D_l399_399498

variable {α β γ a b c : ℝ}
variable {A B C AD BD CD : ℝ}
variable {BC : ℝ → ℝ}

-- Conditions of the problem
def triangle_conditions (α β γ a b c A B C : ℝ) : Prop :=
  α = a / sin A ∧ β = b / sin B ∧ γ = c / sin C

def given_equation (b A C a : ℝ) : Prop :=
  b * sin A * sin C = a * cos( A / 2) ^ 2

-- Proofs to demonstrate
theorem prove_angles_equal (A B C a b c : ℝ) (h₁ : b * sin A * sin C = a * cos (A / 2) ^ 2)
  (h₂ : triangle_conditions α β γ a b cA B C) : A = B := 
sorry

theorem dot_product_BC_D (A B C :ℝ)(hB_eq_C : B = C)
  (AD : ℝ) (BD CD : ℝ)(b : ℝ)(h_AD : AD = sqrt 3) (h_b : b = 2) : 
  \overrightarrow{BD} \cdot \overrightarrow{CD} = -1 :=
sorry


end prove_angles_equal_dot_product_BC_D_l399_399498


namespace beads_allocation_difference_l399_399595

theorem beads_allocation_difference :
  ∃ (n y z : ℕ) (x : ℕ), 
    n = 80 ∧ 
    y = x + 5 ∧ 
    z = x + 7 ∧ 
    z = y + 2 ∧ 
    let b1 := (n * (x + 2)) / (2 * x + 2),
        b2 := (n * (x + 7)) / (2 * x + 12) in
        b1 - b2 = 4 := 
begin
  sorry
end

end beads_allocation_difference_l399_399595


namespace percentage_decrease_of_larger_angle_l399_399188

noncomputable def complementary_angles_decrease_percentage : Real :=
let total_degrees := 90
let ratio_sum := 3 + 7
let part := total_degrees / ratio_sum
let smaller_angle := 3 * part
let larger_angle := 7 * part
let increased_smaller_angle := smaller_angle * 1.2
let new_larger_angle := total_degrees - increased_smaller_angle
let decrease_amount := larger_angle - new_larger_angle
(decrease_amount / larger_angle) * 100

theorem percentage_decrease_of_larger_angle
  (smaller_increased_percentage : Real := 20)
  (ratio_three : Real := 3)
  (ratio_seven : Real := 7)
  (total_degrees : Real := 90)
  (expected_decrease : Real := 8.57):
  complementary_angles_decrease_percentage = expected_decrease := 
sorry

end percentage_decrease_of_larger_angle_l399_399188


namespace monotonicity_and_no_x_intercept_l399_399095

noncomputable theory

def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

def is_monotonic (a : ℝ) (x : ℝ) : Prop := 
  if x < 1 / a then 
    f a x > f a (x + ε) -- ε is a small positive value
  else if x > 1 / a then 
    f a x < f a (x - ε)
  else
    true -- At x = 1/a, the function transits from decreasing to increasing

theorem monotonicity_and_no_x_intercept 
  (a : ℝ) (h1 : 0 < a) : 
  (∀ x : ℝ, 0 < x → is_monotonic a x) ∧ 
  (∀ x : ℝ, f a x ≠ 0) ↔ 
  (a ∈ Ioi (1 / real.exp 1)) := 
sorry

end monotonicity_and_no_x_intercept_l399_399095


namespace complement_union_M_N_l399_399055

noncomputable theory

-- Define the set M
def M : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }

-- Define the set N
def N : Set ℝ := { y : ℝ | y ≥ 1 }

-- Main theorem statement
theorem complement_union_M_N :
  { x : ℝ | x ∉ (M ∪ N) } = { x : ℝ | 1 ≤ x } :=
sorry

end complement_union_M_N_l399_399055


namespace sum_inequality_l399_399107

variables {S : ℕ → ℝ}
variables {a : ℕ → ℝ}
variables {m n p k : ℕ}

-- Definitions for the conditions given in the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i j, a (i + 1) - a i = a (j + 1) - a j

def sum_of_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a 1 + a (n - 1)) / 2

def non_negative_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≥ 0

-- The theorem to prove
theorem sum_inequality (arith_seq : is_arithmetic_sequence a)
  (S_eq : sum_of_arithmetic_sequence S a)
  (nn_seq : non_negative_sequence a)
  (h1 : m + n = 2 * p) (m_pos : 0 < m) (n_pos : 0 < n) (p_pos : 0 < p) :
  1 / (S m) ^ k + 1 / (S n) ^ k ≥ 2 / (S p) ^ k :=
by sorry

end sum_inequality_l399_399107


namespace distance_between_points_l399_399246

theorem distance_between_points :
  let p1 := (1 : ℝ, 2 : ℝ)
  let p2 := (-2 : ℝ, -3 : ℝ)
  ∀ d : ℝ, d = Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) → d = Real.sqrt 34 :=
by
  intros p1 p2 d h
  rw [←h]
  sorry

end distance_between_points_l399_399246


namespace find_x_l399_399654

theorem find_x (x : ℝ) (h : 0.95 * x - 12 = 178) : x = 200 := 
by 
  sorry

end find_x_l399_399654


namespace first_batch_price_is_50_max_number_of_type_a_tools_l399_399234

-- Define the conditions
def first_batch_cost : Nat := 2000
def second_batch_cost : Nat := 2200
def price_increase : Nat := 5
def max_total_cost : Nat := 2500
def type_b_cost : Nat := 40
def total_third_batch : Nat := 50

-- First batch price per tool
theorem first_batch_price_is_50 (x : Nat) (h1 : first_batch_cost * (x + price_increase) = second_batch_cost * x) :
  x = 50 :=
sorry

-- Second batch price per tool & maximum type A tools in third batch
theorem max_number_of_type_a_tools (y : Nat)
  (h2 : 55 * y + type_b_cost * (total_third_batch - y) ≤ max_total_cost) :
  y ≤ 33 :=
sorry

end first_batch_price_is_50_max_number_of_type_a_tools_l399_399234


namespace sequence_inequality_l399_399832

noncomputable def sequence (a : ℕ → ℝ) : ℕ → ℝ
| 1 := 1
| (n+1) := a n / (a n)^3 + 1

theorem sequence_inequality (a : ℕ → ℝ) (h : ∀ n, a (n+1) = a n / ((a n)^3 + 1) ∧ a 1 = 1) :
  ∀ n, a n > 1 / ( (3 * n + log n + (14 / 9))^(1/3) ) :=
by
  sorry

end sequence_inequality_l399_399832


namespace Andy_2020th_turn_l399_399318

noncomputable def Andy_position (n : ℕ) : ℤ × ℤ :=
  if n = 0 then (-20, 20)
  else
    let (x, y) := Andy_position (n - 1) in
    match ((n - 1) % 4) with
      | 0 => (x + n, y) -- east
      | 1 => (x, y + n) -- north
      | 2 => (x - n, y) -- west
      | 3 => (x, y - n) -- south
      | _ => (x, y) -- impossible case, but helps Lean recognize totality of pattern matching

theorem Andy_2020th_turn : Andy_position 2020 = (-1030, -990) :=
  sorry

end Andy_2020th_turn_l399_399318


namespace interval_of_increase_l399_399175

-- Definition of the power function passing through the point (2, 4)
def power_function_passing_through (α : ℝ) : Prop :=
  (2:ℝ) ^ α = 4

-- Definition for the interval of monotonic increase
def interval_monotonic_increase (α : ℝ) : Set ℝ :=
  if α > 0 then Ioi (0:ℝ) else Iio (0:ℝ)

-- Statement asserting the interval of monotonic increase for power function passing through (2, 4)
theorem interval_of_increase :
  power_function_passing_through 2 →
  interval_monotonic_increase 2 = Ioi (0:ℝ) :=
by
  intros h
  sorry

end interval_of_increase_l399_399175


namespace distance_between_hyperbola_vertices_l399_399759

theorem distance_between_hyperbola_vertices :
  (∃ a : ℝ, a = real.sqrt 144 ∧ ∀ d : ℝ, d = 2 * a → d = 24) :=
begin
  use real.sqrt 144,
  split,
  { refl },
  { intros d hd,
    rw hd,
    refl }
end

end distance_between_hyperbola_vertices_l399_399759


namespace find_vol_sa_of_new_body_l399_399995

-- Definitions of given conditions
variables (V S : ℝ)
variables (n : ℕ) -- number of edges
variables (l : Fin n → ℝ) -- function that assigns length to each edge
variables (φ : Fin n → ℝ) -- function that assigns dihedral angle to each edge
variable (d : ℝ) -- distance from the polyhedron

-- Statement of the theorem
theorem find_vol_sa_of_new_body {V S : ℝ} {n : ℕ} {l : Fin n → ℝ} {φ : Fin n → ℝ} {d : ℝ} :
  (Volume : ℝ × SurfaceArea : ℝ) = 
  let volume := V + S * d + (1/2) * d^2 * (∑ i, (Real.pi - φ i) * l i) + (4 / 3) * Real.pi * d^3 in
  let surface_area := S + d * (∑ i, (Real.pi - φ i) * l i) + 4 * Real.pi * d^2 in
  (volume, surface_area) := sorry

end find_vol_sa_of_new_body_l399_399995


namespace x_tangent_coordinate_is_root_l399_399445

noncomputable def tangency_x_coordinate (t : ℝ) (h : t ≠ 0) : ℝ :=
  let f := λ x => (Real.exp x) / x
  let f' := λ x => (Real.exp x * (x - 1)) / (x * x)
  let eqns (m : ℝ) (n : ℝ) (h_m : 0 < m) : Prop :=
    (m - t * n - 2 = 0) ∧ (f m = n) ∧ (f' m = 1 / t)
  if ∃ m n, eqns m n (by apply_instance) then 2 + Real.sqrt 2 else 0
  -- The above line should technically check for the m satisfying all conditions resulting in 2 ± sqrt 2
-- The next line is just to declare the theorem without proof
theorem x_tangent_coordinate_is_root (t : ℝ) (h : t ≠ 0) :
  ∃ m : ℝ, m = 2 + Real.sqrt 2 ∨ m = 2 - Real.sqrt 2 := sorry

end x_tangent_coordinate_is_root_l399_399445


namespace sin_cos_identity_l399_399367

theorem sin_cos_identity : 
  sin (80 * real.pi / 180) * cos (40 * real.pi / 180) + 
  cos (80 * real.pi / 180) * sin (40 * real.pi / 180) = 
  sqrt 3 / 2 := 
sorry

end sin_cos_identity_l399_399367


namespace cats_in_shelter_l399_399503

-- Define the initial conditions
def initial_cats := 20
def monday_addition := 2
def tuesday_addition := 1
def wednesday_subtraction := 3 * 2

-- Problem statement: Prove that the total number of cats after all events is 17
theorem cats_in_shelter : initial_cats + monday_addition + tuesday_addition - wednesday_subtraction = 17 :=
by
  sorry

end cats_in_shelter_l399_399503


namespace angle_A_is_pi_over_three_l399_399883

-- Definitions from condition
def triangle := Type
variable (a b c A B C : ℝ)

-- Conditions
def condition (A B C : ℝ) := 
  (A + B + C = π) ∧ 
  (A > 0) ∧ (B > 0) ∧ (C > 0) ∧
  (tan(A + B) * (1 - tan A * tan B) = (sqrt 3 * sin(C)) / (sin A * cos B))

-- Theorem statement
theorem angle_A_is_pi_over_three
  (A B C : ℝ) 
  (A_is_triangle_angle : 0 < A ∧ A < π)
  (h : condition A B C) :
  A = π / 3 :=
sorry

end angle_A_is_pi_over_three_l399_399883


namespace eden_stuffed_bears_l399_399711

theorem eden_stuffed_bears
  (initial_bears : ℕ)
  (favorite_bears : ℕ)
  (sisters : ℕ)
  (eden_initial_bears : ℕ)
  (remaining_bears := initial_bears - favorite_bears)
  (bears_per_sister := remaining_bears / sisters)
  (eden_bears_now := eden_initial_bears + bears_per_sister)
  (h1 : initial_bears = 20)
  (h2 : favorite_bears = 8)
  (h3 : sisters = 3)
  (h4 : eden_initial_bears = 10) :
  eden_bears_now = 14 := by
{
  sorry
}

end eden_stuffed_bears_l399_399711


namespace exists_nat_add_sum_of_digits_eq_1980_consecutive_nat_in_sum_form_l399_399648

-- Part (a)
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem exists_nat_add_sum_of_digits_eq_1980 :
  ∃ n : ℕ, n + sum_of_digits n = 1980 := sorry

-- Part (b)
theorem consecutive_nat_in_sum_form :
  ∀ n : ℕ, (∃ m : ℕ, m = n ∨ m = n + 1) := sorry

end exists_nat_add_sum_of_digits_eq_1980_consecutive_nat_in_sum_form_l399_399648


namespace light_and_line_l399_399177

-- Define the concepts based on the number of endpoints
def is_ray (x : Type) : Prop := ∃ (a : Type) (b : Type), x = a ∨ x = a ∧ b
def is_line_segment (x : Type) : Prop := ∃ (a : Type) (b : Type), x = a ∧ b
def is_straight_line (x : Type) : Prop := (∀ (a : Type), x = a) ∧ ∀ (b : Type), x = b

-- Define types for the sun's light and the line between telephone poles
constant light_emitted_by_sun : Type
constant line_between_telephone_poles : Type

-- The mathematical equivalent proof problem in Lean 4
theorem light_and_line (h_ray : is_ray light_emitted_by_sun)
    (h_line_segment : is_line_segment line_between_telephone_poles) :
  (∃ (x : Type), is_ray x ∧ x = light_emitted_by_sun) ∧
  (∃ (y : Type), is_line_segment y ∧ y = line_between_telephone_poles) :=
by
  sorry

end light_and_line_l399_399177


namespace sum_of_three_digit_numbers_divisible_by_8_and_9_remainder_1_l399_399589

theorem sum_of_three_digit_numbers_divisible_by_8_and_9_remainder_1 :
  (∑ n in {145, 217, 289, 361, 433, 505, 577, 649, 721, 793, 865, 937}, n) = 6492 :=
by
  sorry

end sum_of_three_digit_numbers_divisible_by_8_and_9_remainder_1_l399_399589


namespace square_free_polynomial_count_l399_399612

theorem square_free_polynomial_count (p m : ℕ) (hp : Nat.Prime p) (hm : m ≥ 2) :
  let S := {P : Polynomial ℤ | Polynomial.monic P ∧ P.degree = m ∧ 
                               (∀ Q R : Polynomial ℤ, Q.natDegree ≠ 0 → 
                                 ¬ (P ≡ Q^2 * R [MOD p]))}
  S.card = p^m - p^{m - 1} :=
by
  sorry

end square_free_polynomial_count_l399_399612


namespace complementary_angles_decrease_86_percent_l399_399196

theorem complementary_angles_decrease_86_percent (x : ℝ) (h : 10 * x = 90) :
  let small_angle := 3 * x
  let increased_small_angle := small_angle * 1.2
  let large_angle := 7 * x
  let new_large_angle := 90 - increased_small_angle
  (new_large_angle / large_angle) * 100 = 91.4 :=
by
  sorry

end complementary_angles_decrease_86_percent_l399_399196


namespace sum_of_squares_correct_l399_399254

def expr := 3 * (λ x : ℝ, x^2 - 3*x + 3) - 8 * (λ x : ℝ, x^3 - 2*x^2 + 4*x - 1)

def sum_of_squares_of_coefficients (p : ℝ → ℝ) : ℝ :=
  let cs := [ -8, 19, -41, 17 ]  -- coefficients of the simplified expression
  cs.map (λ c, c * c) |>.sum

theorem sum_of_squares_correct :
  sum_of_squares_of_coefficients expr = 2395 := by
  sorry

end sum_of_squares_correct_l399_399254


namespace find_k_l399_399267

theorem find_k (k : ℕ) (h : (64 : ℕ) / k = 4) : k = 16 := by
  sorry

end find_k_l399_399267


namespace arithmetic_mean_reciprocal_primes_l399_399340

theorem arithmetic_mean_reciprocal_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := (1 : ℚ) / p1
  let r2 := (1 : ℚ) / p2
  let r3 := (1 : ℚ) / p3
  let r4 := (1 : ℚ) / p4
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 := by
sorry

end arithmetic_mean_reciprocal_primes_l399_399340


namespace incenter_distance_identity_l399_399894

variables {A B C P: Point} -- Define the points A, B, C, and P
variables {a b c: ℝ} -- Define the side lengths a, b, c
variables [PA PB PC QA QB QC QP: ℝ] -- Define the distances from the points

-- Define the center of the inscribed circle Q
noncomputable def is_incircle_center (Q: Point) : Prop :=
∀ P: Point, 
  a * (PA)^2 + b * (PB)^2 + c * (PC)^2 = (a + b + c) * (QP)^2 + a * (QA)^2 + b * (QB)^2 + c * (QC)^2

-- The theorem we want to prove
theorem incenter_distance_identity :
  is_incircle_center Q → 
  (∀ P: Point, a * (PA)^2 + b * (PB)^2 + c * (PC)^2 = a * (QA)^2 + b * (QB)^2 + c * (QC)^2 + (a + b + c) * (QP)^2) :=
sorry

end incenter_distance_identity_l399_399894


namespace dry_person_exists_when_odd_all_drenched_when_even_l399_399928

-- Define the structure of the field with distinct pairwise distances.
structure Field where
  positions : ℕ → ℝ × ℝ
  distinct_distances : ∀ i j : ℕ, i ≠ j → (dist (positions i) (positions j)) ≠ (dist (positions i') (positions j'))

-- Define the function to model the scenario where each person fires at the closest person
def closest_person (f : Field) : ℕ → ℕ := 
  λ i, argmin (λ j, dist (f.positions i) (f.positions j)) (finset.filter (λ j, j ≠ i) (finset.range n))

-- Prove that if n is odd, there is at least one person left dry
theorem dry_person_exists_when_odd (f : Field) (n : ℕ) (h_odd : n % 2 = 1): 
  ∃ i : ℕ, ¬ ∃ j : ℕ, closest_person f j = i := by
  sorry

-- Prove if n is even, there might not be any person left dry (constructive existence)
theorem all_drenched_when_even (n : ℕ) (f : Field) (h_even : n % 2 = 0):
  ¬ ∃ i : ℕ, ¬ ∃ j : ℕ, closest_person f j = i -> False :=
  sorry

end dry_person_exists_when_odd_all_drenched_when_even_l399_399928


namespace minValue_is_9_minValue_achieves_9_l399_399523

noncomputable def minValue (x y : ℝ) : ℝ :=
  (x^2 + 1/(y^2)) * (1/(x^2) + 4 * y^2)

theorem minValue_is_9 (x y : ℝ) (hx : x > 0) (hy : y > 0) : minValue x y ≥ 9 :=
  sorry

theorem minValue_achieves_9 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 1/2) : minValue x y = 9 :=
  sorry

end minValue_is_9_minValue_achieves_9_l399_399523


namespace tan_domain_l399_399172

theorem tan_domain :
  (∀ x, tan x ≠ x (k : ℤ) (h : k π + π / 2)) →
    (∀ x, x = kπ + π / 2 → x ∉ dom tan) :=
  sorry -- Proof to be filled in

end tan_domain_l399_399172


namespace unique_intersection_condition_l399_399974

theorem unique_intersection_condition
  (a c b d : ℝ)
  (h_sym_central : ∀ x y : ℝ, (y = 2a + 1/(x-b) ∧ y = 2c + 1/(x-d) → (x, y) = (1/2*(b+d), a+c)))
  (h_common_point : 2a + 1/(1/2*(b+d) - b) = 2c + 1/(1/2*(b+d) - d) ∧ 2a + 1/(1/2*(b+d) - b) = a + c) :
  (a - c) * (b - d) = 2 :=
by
  -- Proof goes here
  sorry

end unique_intersection_condition_l399_399974


namespace rhombus_area_correct_l399_399238

-- Define the rhombus with given conditions
structure Rhombus where
  side_length : ℝ
  angle_deg : ℝ
  (angle_cond : angle_deg = 45)
  (side_cond : side_length = 4)

-- Define the area function for a rhombus
def rhombus_area : Rhombus → ℝ
  | ⟨s, θ, _, _⟩ => s * s * Real.sin (θ * Real.pi / 180) / 2

-- Theorem stating the area of the rhombus is 8√2 square centimeters given the conditions
theorem rhombus_area_correct (r : Rhombus) : rhombus_area r = 8 * Real.sqrt 2 :=
by
  cases r
  simp [Rhombus.angle_cond, Rhombus.side_cond, rhombus_area, Real.sin, Real.sqrt]
  sorry

end rhombus_area_correct_l399_399238


namespace find_extrema_l399_399391

def f (x : ℝ) : ℝ := x^3 - (3/2)*x^2 + 5

theorem find_extrema : 
  ∃ (a b : ℝ), a = -9 ∧ b = 7 ∧ ∀ x ∈ Set.Icc (-(2:ℝ)) (2:ℝ), 
    f x ≥ a ∧ f x ≤ b := 
  sorry

end find_extrema_l399_399391


namespace function_inequality_on_interval_l399_399786

def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 3

theorem function_inequality_on_interval :
  ∀ x, x ∈ Icc (-2 : ℝ) 2 → f x ≤ 3 :=
begin
  sorry
end

end function_inequality_on_interval_l399_399786


namespace scrap_cookie_radius_l399_399559

theorem scrap_cookie_radius 
  (r_large : ℝ) (r_small : ℝ) (num_small : ℕ) (areas : Real) 
  (h1 : r_large = 3)
  (h2 : r_small = 1)
  (h3 : num_small = 7)
  (area_large_cookie : areas = Real.pi * r_large^2)
  (area_one_small_cookie : areas = Real.pi * r_small^2)
  (total_area_small_cookies : areas = num_small * area_one_small_cookie)
  (leftover_area : areas = area_large_cookie - total_area_small_cookies) :
  ∃ r_scrap : ℝ, r_scrap = Real.sqrt 2 :=
by
  sorry

end scrap_cookie_radius_l399_399559


namespace sum_sequence_equals_m_m1_div_2_l399_399634

theorem sum_sequence_equals_m_m1_div_2 (x : ℕ → ℝ) :
  (∀ n : ℕ, (∑ i in Finset.range (n+1), x i)^2 = ∑ i in Finset.range (n+1), (x i)^3) →
  ∀ n : ℕ, ∃ m : ℕ, (∑ i in Finset.range (n+1), x i) = m * (m + 1) / 2 :=
by
  sorry

end sum_sequence_equals_m_m1_div_2_l399_399634


namespace tangent_line_eqn_at_half_f_max_min_val_l399_399439

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x - Real.log x

theorem tangent_line_eqn_at_half :
  let x := 1 / 2
  let y := f x in
  y = 2 * x - 2 + Real.log 2 :=
sorry

theorem f_max_min_val :
  let interval := Set.Icc (1 / 4) Real.e
  let max_val := (0 : ℝ)
  let min_val := Real.log 4 - 3 in
  (Set.image f interval).Sup = max_val ∧
  (Set.image f interval).Inf = min_val :=
sorry

end tangent_line_eqn_at_half_f_max_min_val_l399_399439


namespace alternating_numbers_divisible_by_15_l399_399461

def is_alternating_number (n : ℕ) : Prop :=
  ∀ (d1 d2 : ℕ) (p : list nat), 
    n.to_digits = d1 :: d2 :: p →
    (∀ i : ℕ, i < p.length → if i % 2 = 0 then p.nth i = some d1 else p.nth i = some d2) ∨ 
    (∀ i : ℕ, i < p.length → if i % 2 = 0 then p.nth i = some d2 else p.nth i = some d1)

def is_divisible_by_15 (n : ℕ) : Prop := 
  (n % 5 = 0) ∧ ((n.to_digits.sum) % 3 = 0)

theorem alternating_numbers_divisible_by_15 :
  (∃ (n : ℕ), n.digits.length = 5 ∧ is_alternating_number n ∧ is_divisible_by_15 n) ↔
  4 := sorry

end alternating_numbers_divisible_by_15_l399_399461


namespace eden_bears_count_l399_399709

-- Define the main hypothesis
def initial_bears : Nat := 20
def favorite_bears : Nat := 8
def remaining_bears := initial_bears - favorite_bears

def number_of_sisters : Nat := 3
def bears_per_sister := remaining_bears / number_of_sisters

def eden_initial_bears : Nat := 10
def eden_final_bears := eden_initial_bears + bears_per_sister

theorem eden_bears_count : eden_final_bears = 14 :=
by
  unfold eden_final_bears eden_initial_bears bears_per_sister remaining_bears initial_bears favorite_bears
  norm_num
  sorry

end eden_bears_count_l399_399709


namespace cosine_cosine_greater_sine_sine_l399_399412

theorem cosine_cosine_greater_sine_sine (x : ℝ) : cos (cos x) > sin (sin x) :=
sorry

end cosine_cosine_greater_sine_sine_l399_399412


namespace degrees_to_radians_l399_399707

theorem degrees_to_radians : (800 : ℝ) * (Real.pi / 180) = (40 / 9) * Real.pi :=
by
  sorry

end degrees_to_radians_l399_399707


namespace correlation_coefficient_is_0_92_regression_equation_is_0_16x_plus_0_84_l399_399941

noncomputable def market_data : List (ℕ × ℝ) :=
  [(1, 0.9), (2, 1.2), (3, 1.5), (4, 1.4), (5, 1.6)]

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

def sum_of_products (data : List (ℕ × ℝ)) : ℝ :=
  data.foldr (λ (xy : ℕ × ℝ) acc, acc + xy.1 * xy.2) 0

def variance (l : List ℝ) (mean : ℝ) : ℝ :=
  l.foldr (λ y acc, acc + (y - mean) ^ 2) 0

noncomputable def correlation_coefficient : ℝ :=
  let x_vals := market_data.map Prod.fst
  let y_vals := market_data.map Prod.snd
  let x_mean := mean (x_vals.map (λ x, x : ℝ))
  let y_mean := mean y_vals
  let sum_products := sum_of_products market_data
  let sum_xi_yi := sum_products - 5 * x_mean * y_mean
  let sqrt_xi := sqrt (10 : ℝ)
  let sqrt_yi := 0.55
  sum_xi_yi / (sqrt_xi * sqrt_yi)

noncomputable def regression_equation : ℝ × ℝ :=
  let x_vals := market_data.map Prod.fst
  let y_vals := market_data.map Prod.snd
  let x_mean := mean (x_vals.map (λ x, x : ℝ))
  let y_mean := mean y_vals
  let numerator := 1.6
  let denominator := 55 - 5 * x_mean ^ 2
  let b := numerator / denominator
  let a := y_mean - b * x_mean
  (b, a)

-- Proof statements
theorem correlation_coefficient_is_0_92 : correlation_coefficient ≈ 0.92 := sorry

theorem regression_equation_is_0_16x_plus_0_84 : regression_equation = (0.16, 0.84) := sorry

end correlation_coefficient_is_0_92_regression_equation_is_0_16x_plus_0_84_l399_399941


namespace friend_reading_time_l399_399118

-- Define the conditions
def my_reading_time : ℝ := 1.5 * 60 -- 1.5 hours converted to minutes
def friend_speed_multiplier : ℝ := 5 -- Friend reads 5 times faster than I do
def distraction_time : ℝ := 15 -- Friend is distracted for 15 minutes

-- Define the time taken for my friend to read the book accounting for distraction
theorem friend_reading_time :
  (my_reading_time / friend_speed_multiplier) + distraction_time = 33 := by
  sorry

end friend_reading_time_l399_399118


namespace sum_coordinates_D_l399_399569

structure Point where
  x : Int
  y : Int

def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

def A : Point := { x := -1, y := 2 }
def C : Point := { x := 7, y := -4 }
def B : Point := { x := 3, y := -6 }

def D : Point := { x := 3, y := 4 }

theorem sum_coordinates_D :
  let D := { x := 3, y := 4 }
  D.x + D.y = 7 :=
sorry

end sum_coordinates_D_l399_399569


namespace polynomial_expansion_correct_l399_399731

-- Define the polynomials P(z) and Q(z)
def P (z : ℂ) : ℂ := 3 * z^2 + 4 * z - 5
def Q (z : ℂ) : ℂ := 4 * z^3 - 3 * z^2 + 2 * z - 1

-- State the theorem
theorem polynomial_expansion_correct (z : ℂ) :
  P(z) * Q(z) = 12 * z^5 + 25 * z^4 - 41 * z^3 - 14 * z^2 + 28 * z - 5 :=
by
  -- Placeholder for the proof
  sorry

end polynomial_expansion_correct_l399_399731


namespace find_S2011_l399_399416

-- Conditions
def a (n : ℕ) : ℕ := if n = 1 then 1 else n - 2 * S (n - 1)
def S : ℕ → ℕ
| 0 => 0
| 1 => a 1
| n + 1 => a n + S n

-- Thereom stating the goal to prove
theorem find_S2011 : S 2011 = 1006 :=
sorry

end find_S2011_l399_399416


namespace seven_nat_sum_divisible_by_5_l399_399560

theorem seven_nat_sum_divisible_by_5 
  (a b c d e f g : ℕ)
  (h1 : (b + c + d + e + f + g) % 5 = 0)
  (h2 : (a + c + d + e + f + g) % 5 = 0)
  (h3 : (a + b + d + e + f + g) % 5 = 0)
  (h4 : (a + b + c + e + f + g) % 5 = 0)
  (h5 : (a + b + c + d + f + g) % 5 = 0)
  (h6 : (a + b + c + d + e + g) % 5 = 0)
  (h7 : (a + b + c + d + e + f) % 5 = 0)
  : 
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0 ∧ f % 5 = 0 ∧ g % 5 = 0 :=
sorry

end seven_nat_sum_divisible_by_5_l399_399560


namespace triangle_ABC_isosceles_or_right_l399_399865

variable {A B C : Type} [Triangle A B C]
variable (a b : ℝ)
variables (angle_A angle_B : ℝ)
variable (A B C : TriangleVertices)

axiom a_cos_A_eq_b_cos_B : a * Real.cos angle_A = b * Real.cos angle_B

theorem triangle_ABC_isosceles_or_right 
  (h : a * Real.cos angle_A = b * Real.cos angle_B) : 
  IsIscoscelesOrRight A B C := 
  sorry

end triangle_ABC_isosceles_or_right_l399_399865


namespace plumber_fix_cost_toilet_l399_399301

noncomputable def fixCost_Sink : ℕ := 30
noncomputable def fixCost_Shower : ℕ := 40

theorem plumber_fix_cost_toilet
  (T : ℕ)
  (Earnings1 : ℕ := 3 * T + 3 * fixCost_Sink)
  (Earnings2 : ℕ := 2 * T + 5 * fixCost_Sink)
  (Earnings3 : ℕ := T + 2 * fixCost_Shower + 3 * fixCost_Sink)
  (MaxEarnings : ℕ := 250) :
  Earnings2 = MaxEarnings → T = 50 :=
by
  sorry

end plumber_fix_cost_toilet_l399_399301


namespace inverse_h_l399_399519

noncomputable def f (x : ℝ) := 3 * x + 4
noncomputable def g (x : ℝ) := 2 * x - 3
noncomputable def h (x : ℝ) := f (g x)

-- Prove that the inverse of h is (x + 5) / 6
theorem inverse_h : ∀ x : ℝ, h ((x + 5) / 6) = x :=
by
  intros x
  -- calculate h((x + 5) / 6)
  show f (g ((x + 5) / 6)) = x
  rw [g, apply_instance, f] -- use the definitions of g and f
  sorry

end inverse_h_l399_399519


namespace factorization_of_x4_plus_81_l399_399173

theorem factorization_of_x4_plus_81 :
  ∀ x : ℝ, x^4 + 81 = (x^2 - 3 * x + 4.5) * (x^2 + 3 * x + 4.5) :=
by
  intros x
  sorry

end factorization_of_x4_plus_81_l399_399173


namespace tablet_and_smartphone_battery_lifetime_l399_399538

noncomputable def tablet_standby_hours : ℝ := 14 - 2
noncomputable def tablet_usage_hours : ℝ := 2
noncomputable def smartphone_standby_hours : ℝ := 20 - 3
noncomputable def smartphone_usage_hours : ℝ := 3

noncomputable def tablet_battery_life : ℝ :=
tablet_standby_hours / 18 + tablet_usage_hours / 6

noncomputable def smartphone_battery_life : ℝ :=
smartphone_standby_hours / 30 + smartphone_usage_hours / 4

noncomputable def remaining_smartphone_battery : ℝ :=
1 - smartphone_battery_life

noncomputable def remaining_smartphone_hours : ℝ :=
remaining_smartphone_battery / (1 / 30)

theorem tablet_and_smartphone_battery_lifetime :
  tablet_battery_life = 1 → remaining_smartphone_hours = 9 :=
by {
  intro h,
  rw ←h,
  simp*}

end tablet_and_smartphone_battery_lifetime_l399_399538


namespace max_g_l399_399398

def g (x : ℝ) : ℝ := min (min (3 * x + 3) (x + 2)) (-1/2 * x + 8)

theorem max_g : ∃ x : ℝ, g x = 6 := by
sorry

end max_g_l399_399398


namespace john_bought_soap_l399_399890

theorem john_bought_soap (weight_per_bar : ℝ) (cost_per_pound : ℝ) (total_spent : ℝ) (h1 : weight_per_bar = 1.5) (h2 : cost_per_pound = 0.5) (h3 : total_spent = 15) : 
  total_spent / (weight_per_bar * cost_per_pound) = 20 :=
by
  -- The proof would go here
  sorry

end john_bought_soap_l399_399890


namespace range_of_a_l399_399465

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 2| + |x - 1| ≥ a) ↔ a ≤ 3 := by
  sorry

end range_of_a_l399_399465


namespace solve_equation_l399_399943
-- Import the necessary library

-- Define a noncomputable function that can handle real numbers
noncomputable theory
open_locale classical

-- Statement of the proof
theorem solve_equation :
  (∃ x : ℝ, x / 3 + (30 - x) / 2 = 5) → (∃ x : ℝ, x = 60) :=
begin
  sorry
end

end solve_equation_l399_399943


namespace sum_pth_powers_of_roots_l399_399935

open Complex

-- Definitions for the roots of unity and their properties
def nthRootsOfUnity (n : ℕ) : Finset ℂ := (Finset.range n).map ⟨λ k, exp(2 * π * I * k / n), sorry⟩

theorem sum_pth_powers_of_roots (n p : ℕ) (hn : 0 < n) :
  (∑ z in nthRootsOfUnity n, z ^ p) = if p % n = 0 then n else 0 :=
begin
  sorry
end

end sum_pth_powers_of_roots_l399_399935


namespace math_problem_l399_399671

theorem math_problem : 
  ∀ n : ℕ, 
  n = 5 * 96 → 
  ((n + 17) * 69) = 34293 := 
by
  intros n h
  sorry

end math_problem_l399_399671


namespace inclination_angle_relation_l399_399533

theorem inclination_angle_relation (a b c : ℝ) (α : ℝ) 
  (h_line_eq : ∀ (x y : ℝ), a * x + b * y + c = 0) 
  (h_sin_cos : sin α + cos α = 0) : a - b = 0 :=
sorry

end inclination_angle_relation_l399_399533


namespace true_statements_l399_399629

theorem true_statements :
  (5 ∣ 25) ∧ (19 ∣ 209 ∧ ¬ (19 ∣ 63)) ∧ (30 ∣ 90) ∧ (14 ∣ 28 ∧ 14 ∣ 56) ∧ (9 ∣ 180) :=
by
  have A : 5 ∣ 25 := sorry
  have B1 : 19 ∣ 209 := sorry
  have B2 : ¬ (19 ∣ 63) := sorry
  have C : 30 ∣ 90 := sorry
  have D1 : 14 ∣ 28 := sorry
  have D2 : 14 ∣ 56 := sorry
  have E : 9 ∣ 180 := sorry
  exact ⟨A, ⟨B1, B2⟩, C, ⟨D1, D2⟩, E⟩

end true_statements_l399_399629


namespace ratio_male_to_female_l399_399486

theorem ratio_male_to_female (total_members female_members : ℕ) (h_total : total_members = 18) (h_female : female_members = 6) :
  (total_members - female_members) / Nat.gcd (total_members - female_members) female_members = 2 ∧
  female_members / Nat.gcd (total_members - female_members) female_members = 1 :=
by
  sorry

end ratio_male_to_female_l399_399486


namespace red_triangle_intersections_is_not_2023_l399_399024

theorem red_triangle_intersections_is_not_2023
  (yellow_points : Fin 3 → Point)
  (red_points : Fin 40 → Point)
  (no_four_coplanar : ∀ (p1 p2 p3 p4 : Point), ¬coplanar {p1, p2, p3, p4}) :
  ¬∃ (r1 r2 : ℕ), r1 + r2 = 40 ∧ r1 * r2 = 2023 :=
by
  sorry

end red_triangle_intersections_is_not_2023_l399_399024


namespace find_x_l399_399979

theorem find_x :
  ∃ x : ℕ, 
    let a1 := 11 in
    let a2 := 49 in
    let b1 := a1 in
    let b2 := 6 + x in
    let b3 := x + 7 in
    let c1 := b1 + b2 in
    let c2 := b2 + b3 in
    let d := c1 + c2 in
    d = 60 ∧ x = 10 :=
begin
  sorry
end

end find_x_l399_399979


namespace seating_arrangement_of_athletes_l399_399477

theorem seating_arrangement_of_athletes:
  (teamA teamB teamC : Finset ℕ) (ha : teamA.card = 4) (hb : teamB.card = 3) (hc : teamC.card = 3)
  (disjointAB : Disjoint teamA teamB) (disjointAC : Disjoint teamA teamC) (disjointBC : Disjoint teamB teamC) :
  ∑ x in teamA ∪ teamB ∪ teamC, 1 = 10 → 
  3! * 4! * 3! * 3! = 5184 := 
by
  sorry

end seating_arrangement_of_athletes_l399_399477


namespace sum_of_roots_eq_neg_seven_fourths_l399_399174

theorem sum_of_roots_eq_neg_seven_fourths :
  let f : ℚ → ℚ := λ x, 4 * x^2 + 7 * x + 3
  ∃ p q : ℚ, f p = 0 ∧ f q = 0 ∧ p + q = -7 / 4 :=
by
  sorry

end sum_of_roots_eq_neg_seven_fourths_l399_399174


namespace number_of_pairs_l399_399848

theorem number_of_pairs (books_by_author1 : ℕ) (books_by_author2 : ℕ) (total_books : ℕ) (h1 : books_by_author1 = 6) (h2 : books_by_author2 = 9) (h3 : total_books = 15) : books_by_author1 * books_by_author2 = 54 :=
by
  rw [h1, h2]
  exact rfl

end number_of_pairs_l399_399848


namespace complementary_angles_decrease_percentage_l399_399201

theorem complementary_angles_decrease_percentage :
  ∀ (x : ℝ), (3 * x + 7 * x = 90) →
  (3 * x * 1.2 + 7 * x = 90) →
  (3 * x > 0) →
  (7 * x > 0) →
  let original_larger_angle : ℝ := 7 * x in
  let new_smaller_angle : ℝ := 3 * x * 1.2 in
  let new_larger_angle : ℝ := 90 - new_smaller_angle in
  let decrease : ℝ := original_larger_angle - new_larger_angle in
  let percentage_decrease : ℝ := (decrease / original_larger_angle) * 100 in
  percentage_decrease = 8.57 := 
sorry

end complementary_angles_decrease_percentage_l399_399201


namespace total_cost_l399_399692

theorem total_cost (cost_sandwich : ℕ) (cost_soda : ℕ) (num_sandwiches : ℕ) (num_sodas : ℕ) (discount_threshold : ℕ) (discount_rate : ℝ) :
  cost_sandwich = 4 → 
  cost_soda = 3 → 
  num_sandwiches = 7 → 
  num_sodas = 4 → 
  discount_threshold = 30 → 
  discount_rate = 0.10 → 
  let initial_cost := num_sandwiches * cost_sandwich + num_sodas * cost_soda in
  let discount := if initial_cost > discount_threshold then initial_cost * discount_rate else 0 in
  let final_cost := initial_cost - discount in
  final_cost = 36 :=
by
  sorry

end total_cost_l399_399692


namespace nonagon_perimeter_l399_399618

theorem nonagon_perimeter :
  (2 + 2 + 3 + 3 + 1 + 3 + 2 + 2 + 2 = 20) := by
  sorry

end nonagon_perimeter_l399_399618


namespace max_cube_volume_in_pyramid_l399_399677

-- Define the variables for the problem
variables {a : ℝ} (h_pyramid_height : ℝ) (cube_volume : ℝ)

-- Define constants for the pyramid base side length and the height
def side_length : ℝ := 2 -- side length of the equilateral triangle base
def height : ℝ := 3 -- height from base to apex of the pyramid

-- Assumption that the cube's orientation follows the given conditions
def cube_side_length : ℝ := (3 * real.sqrt 6) / 4

-- Calculate the volume of the cube
def volume_of_cube : ℝ := cube_side_length ^ 3

-- Main theorem stating the volume of the cube given the conditions
theorem max_cube_volume_in_pyramid 
  (h_base : side_length = 2) 
  (h_height : height = 3)
  : volume_of_cube = 81 * real.sqrt 6 / 32 :=
sorry

end max_cube_volume_in_pyramid_l399_399677


namespace evaluate_using_horners_method_l399_399242

def f (x : ℝ) : ℝ := 3 * x^6 + 12 * x^5 + 8 * x^4 - 3.5 * x^3 + 7.2 * x^2 + 5 * x - 13

theorem evaluate_using_horners_method :
  f 6 = 243168.2 :=
by
  sorry

end evaluate_using_horners_method_l399_399242


namespace triangle_proof_l399_399545

theorem triangle_proof 
  (A B C K S : Type)
  [InnerProductSpace ℝ (A × ℝ)]
  [InnerProductSpace ℝ (B × ℝ)]
  [InnerProductSpace ℝ (C × ℝ)]
  [InnerProductSpace ℝ (K × ℝ)]
  [InnerProductSpace ℝ (S × ℝ)]
  (h1 : ∠ A C B = 45)
  (h2 : K ∈ Segment ℝ A C)
  (h3 : dist A K = 2 * dist K C)
  (h4 : S ∈ Segment ℝ B K)
  (h5 : ∠ A S K = 60)
  (h6 : ∥(A − S) × (B − K)∥ = 0):
  dist A S = dist B S :=
sorry

end triangle_proof_l399_399545


namespace no_b_satisfies_quadratic_inequality_l399_399364

def quadratic_inequality_has_three_integer_solutions (b : ℤ) : Prop :=
  ∃ (s : Set ℤ), (∀ x ∈ s, x^2 + b*x + 5 ≤ 0) ∧ s.card = 3

theorem no_b_satisfies_quadratic_inequality :
  ∀ b : ℤ, ¬ quadratic_inequality_has_three_integer_solutions b :=
by
  sorry

end no_b_satisfies_quadratic_inequality_l399_399364


namespace ratio_of_male_to_female_students_l399_399870

-- Given conditions:
def num_female_students : ℕ := 13
def total_students : ℕ := 52

-- The statement to proof:
theorem ratio_of_male_to_female_students :
  let num_male_students := total_students - num_female_students in
  let ratio := num_male_students / num_female_students in
  ratio = 3 := by
  sorry

end ratio_of_male_to_female_students_l399_399870


namespace solve_for_x_l399_399145

theorem solve_for_x (x : ℝ) (h₁ : (7 * x) / (x + 4) - 4 / (x + 4) = 2 / (x + 4)) (h₂ : x ≠ -4) : x = 6 / 7 :=
by
  sorry

end solve_for_x_l399_399145


namespace distinct_convex_quadrilaterals_l399_399411

open Nat

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem distinct_convex_quadrilaterals (n : ℕ) (h : n > 4) 
  (no_three_collinear : ℕ → Prop) :
  ∃ k, k ≥ combinations n 5 / (n - 4) :=
by
  sorry

end distinct_convex_quadrilaterals_l399_399411


namespace prove_fraction_identity_l399_399104

theorem prove_fraction_identity (x y : ℂ) (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) : 
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 41 / 20 := 
by 
  sorry

end prove_fraction_identity_l399_399104


namespace factorial_divisibility_l399_399365

theorem factorial_divisibility (n : ℕ) : (∃ k : ℕ, (n! % (n + 2)) = 0 → (k! % (k + 2)) ≠ 0) ∧ 
  (∃ p q : ℕ, (n + 2 = p * q → n! % (p * q) = 0) ∨ 
  (∃ k : ℕ, prime k ∧ n = k ^ 2 - 2 ∧ (n! % (k ^ 2) = 0))) :=
sorry

end factorial_divisibility_l399_399365


namespace sum_arithmetic_sequence_l399_399225

theorem sum_arithmetic_sequence 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : (a 2 - 1)^3 + 2014 * (a 2 - 1) = real.sin (2011 * real.pi / 3))
  (h2 : (a 2013 - 1)^3 + 2014 * (a 2013 - 1) = real.cos (2011 * real.pi / 6))
  (Sn_eq : ∀ n, S n = n / 2 * (a 1 + a n)) :
  S 2014 = 2014 :=
by
  sorry

end sum_arithmetic_sequence_l399_399225


namespace monotonic_intervals_logarithmic_inequality_l399_399821

noncomputable def f (x : ℝ) : ℝ := x^2 - x - Real.log x

theorem monotonic_intervals :
  (∀ x ∈ Set.Ioo 0 1, f x > f (x + 1E-9) ∧ f x < f (x - 1E-9)) ∧ 
  (∀ y ∈ Set.Ioi 1, f y < f (y + 1E-9) ∧ f y > f (y - 1E-9)) := sorry

theorem logarithmic_inequality (a : ℝ) (ha : a > 0) (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (hneq : x1 ≠ x2)
  (h_eq1 : a * x1 + f x1 = x1^2 - x1) (h_eq2 : a * x2 + f x2 = x2^2 - x2) :
  Real.log x1 + Real.log x2 + 2 * Real.log a < 0 := sorry

end monotonic_intervals_logarithmic_inequality_l399_399821


namespace complementary_angles_decrease_percentage_l399_399198

theorem complementary_angles_decrease_percentage :
  ∀ (x : ℝ), (3 * x + 7 * x = 90) →
  (3 * x * 1.2 + 7 * x = 90) →
  (3 * x > 0) →
  (7 * x > 0) →
  let original_larger_angle : ℝ := 7 * x in
  let new_smaller_angle : ℝ := 3 * x * 1.2 in
  let new_larger_angle : ℝ := 90 - new_smaller_angle in
  let decrease : ℝ := original_larger_angle - new_larger_angle in
  let percentage_decrease : ℝ := (decrease / original_larger_angle) * 100 in
  percentage_decrease = 8.57 := 
sorry

end complementary_angles_decrease_percentage_l399_399198


namespace x_plus_y_value_l399_399855

theorem x_plus_y_value (x y : ℝ) (h : sqrt (1 - x) + abs (2 - y) = 0) : x + y = 3 :=
sorry

end x_plus_y_value_l399_399855


namespace remainder_when_two_pow_thirty_three_div_nine_l399_399248

-- Define the base and the exponent
def base : ℕ := 2
def exp : ℕ := 33
def modulus : ℕ := 9

-- The main statement to prove
theorem remainder_when_two_pow_thirty_three_div_nine :
  (base ^ exp) % modulus = 8 :=
by
  sorry

end remainder_when_two_pow_thirty_three_div_nine_l399_399248


namespace ellipse_eccentricity_l399_399726

def Ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def PointOutsideEllipse (a b x0 y0 : ℝ) : Prop :=
  x0^2 / a^2 + y0^2 / b^2 > 1

def TangentsFromPointToEllipse (a b x0 y0 k1 k2 : ℝ) : Prop :=
  let t (k : ℝ) : ℝ := y0 - k * x0
  let equation : (ℝ → ℝ) := λ k, (a^2 - x0^2) * k^2 + 2 * x0 * y0 * k + b^2 - y0^2
  equation k1 + equation k2 = 0 ∧ equation k1 * equation k2 = (y0^2 - b^2) * (x0^2 - a^2)

def TangentAngleCondition (k1 k2 : ℝ) : Prop :=
  abs ((k1 - k2) / (1 + k1 * k2)) = 2

def LocusCondition (a b k m : ℝ) : Prop :=
  (a^2 - b^2) * m^2 = -k^2 + (3 * a^2 + 2 * b^2) * k - (a^2 + b^2)^2 - a^2 * b^2

def CircleIntersectsLocus (k m : ℝ) : Prop :=
  ∃ x1 x2 xn : ℝ, x1^2 + x2^2 + xn^2 = k ∧ abs (xn) = m

def GivenCondition (m k : ℝ) : Prop :=
  m^2 / k = 9 / 10

def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity (a b : ℝ) :
  (∀ (x y : ℝ), Ellipse a b x y) →
  (∃ (x0 y0 : ℝ), PointOutsideEllipse a b x0 y0 ∧
    ∃ (k1 k2 : ℝ), TangentsFromPointToEllipse a b x0 y0 k1 k2 ∧
      TangentAngleCondition k1 k2 ∧
      ∃ (k m : ℝ), LocusCondition a b k m ∧
        CircleIntersectsLocus k m ∧
        GivenCondition m k) →
  Eccentricity a b = Real.sqrt (30) / 6 :=
by
  intros h1 h2
  sorry

end ellipse_eccentricity_l399_399726


namespace octadecagon_identity_l399_399918

theorem octadecagon_identity (a r : ℝ) (h : a = 2 * r * Real.sin (π / 18)) :
  a^3 + r^3 = 3 * r^2 * a :=
sorry

end octadecagon_identity_l399_399918


namespace world_forest_area_l399_399396

theorem world_forest_area
  (F : ℝ) (P : ℝ) (W : ℝ)
  (hF : F = 53.42)
  (hP : P = 0.66)
  (conversion_factor : F / 1000 = 0.05342)
  (proportion : 0.05342 / W = P / 100) : W ≈ 8.09 :=
by sorry

end world_forest_area_l399_399396


namespace cone_volume_l399_399665

-- Define the radius of the circle
def R : ℝ := 6

-- Define the proportion of the remaining sector
def proportion_remaining : ℝ := 5 / 6

-- Define the formula for the base radius r of the cone
def base_radius : ℝ := (proportion_remaining * 2 * Real.pi * R) / (2 * Real.pi)

-- Define the formula for the height h of the cone
def height : ℝ := Real.sqrt (R^2 - base_radius^2)

-- Define the volume V of the cone
def volume : ℝ := (1 / 3) * Real.pi * (base_radius^2) * height

theorem cone_volume : volume = (25 * Real.sqrt 11 / 3) * Real.pi := 
by {
  -- This is where you would provide the proof.
  sorry
}

end cone_volume_l399_399665


namespace find_y_l399_399168

theorem find_y (y : ℤ) (h : (15 + 26 + y) / 3 = 23) : y = 28 :=
by sorry

end find_y_l399_399168


namespace sequence_never_contains_010101_l399_399492

noncomputable def L (n : Nat) : Nat := n % 10

def a : Nat → Nat
| 1  => 1
| 2  => 0
| 3  => 1
| 4  => 0
| 5  => 1
| 6  => 0
| (n+7) => L (a (n+1) + a (n+2) + a (n+3) + a (n+4) + a (n+5) + a (n+6))

theorem sequence_never_contains_010101 :
  ¬ ∃ n : Nat, a n = 0 ∧ a (n+1) = 1 ∧ a (n+2) = 0 ∧ a (n+3) = 1 ∧ a (n+4) = 0 ∧ a (n+5) = 1 := sorry

end sequence_never_contains_010101_l399_399492


namespace solve_for_x_l399_399942

variable (x : ℝ)

theorem solve_for_x (h : (4 * x + 2) / (5 * x - 5) = 3 / 4) : x = -23 := 
by
  sorry

end solve_for_x_l399_399942


namespace expression_for_f_l399_399437

theorem expression_for_f {f : ℤ → ℤ} (h : ∀ x, f (x + 1) = 3 * x + 4) : ∀ x, f x = 3 * x + 1 :=
by
  sorry

end expression_for_f_l399_399437


namespace recurrence_solution_l399_399109

variables {k : ℕ} {u : ℕ → ℝ} {a : Fin k → ℝ} {x : Fin k → ℝ} {c : Fin k → ℝ}

-- Conditions
def recurrence_relation (u : ℕ → ℝ) (a : Fin k → ℝ) (k : ℕ) : Prop :=
  ∀ n, u (n + k) = ∑ i : Fin k, a i * u (n + k - (i + 1))

def distinct_roots (x : Fin k → ℝ) : Prop :=
  ∀ i j, i ≠ j → x i ≠ x j

def characteristic_polynomial (a : Fin k → ℝ) (x : ℝ) : Prop :=
  x ^ k = ∑ i : Fin k, a i * x ^ (k - (i + 1))

-- Proof statement
theorem recurrence_solution (a : Fin k → ℝ) (x : Fin k → ℝ) (c : Fin k → ℝ) (u : ℕ → ℝ) :
  recurrence_relation u a k →
  distinct_roots x →
  (∀ x_j : Fin k, characteristic_polynomial a (x x_j)) →
  ∃ c : Fin k → ℝ, u = λ n, ∑ i : Fin k, c i * (x i) ^ (n - 1) :=
sorry

end recurrence_solution_l399_399109


namespace sugar_price_l399_399183

theorem sugar_price (P : ℝ) (initial_price : ℝ) (consumption_reduction : ℝ) : initial_price = 3 → consumption_reduction = 0.4 → P = initial_price / (1 - consumption_reduction) → P = 5 :=
by
  assume h₁ : initial_price = 3
  assume h₂ : consumption_reduction = 0.4
  assume h₃ : P = initial_price / (1 - consumption_reduction)
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end sugar_price_l399_399183


namespace fraction_evaluation_l399_399728

theorem fraction_evaluation :
  ( (1 / 2 * 1 / 3 * 1 / 4 * 1 / 5 + 3 / 2 * 3 / 4 * 3 / 5) / 
    (1 / 2 * 2 / 3 * 2 / 5) ) = 41 / 8 :=
by
  sorry

end fraction_evaluation_l399_399728


namespace cash_realized_correct_l399_399956

-- Define the given conditions
def total_before_brokerage : ℝ := 101
def brokerage_rate : ℝ := 0.25 / 100

-- Define the brokerage fee calculation
def brokerage_fee : ℝ := brokerage_rate * total_before_brokerage

-- Rounded brokerage fee
def rounded_brokerage_fee : ℝ := 0.25

-- Define the cash realized calculation
def cash_realized : ℝ := total_before_brokerage - rounded_brokerage_fee

-- Prove that the cash realized is Rs. 100.75
theorem cash_realized_correct : cash_realized = 100.75 :=
by
  unfold cash_realized
  unfold total_before_brokerage
  unfold rounded_brokerage_fee
  norm_num
  norm_cast
  sorry

end cash_realized_correct_l399_399956


namespace distance_between_first_and_last_tree_l399_399372

theorem distance_between_first_and_last_tree (n : ℕ) (d : ℕ) (h₁ : n = 8) (h₂ : d = 100) : 
  let distance_between_trees := d / (5 - 1) in
  let total_distance := distance_between_trees * (n - 1) in
  total_distance = 175 := 
by
  have h₃ : distance_between_trees = 25 := by
    rw [h₂]
    norm_num
  have h₄ : total_distance = 25 * 7 := by
    rw [h₃]
    norm_num
  rw [h₄]
  norm_num

end distance_between_first_and_last_tree_l399_399372


namespace min_value_geom_seq_l399_399816

theorem min_value_geom_seq (a : ℕ → ℝ) (r m n : ℕ) (h_geom : ∃ r, ∀ i, a (i + 1) = a i * r)
  (h_ratio : r = 2) (h_a_m : 4 * a 1 = a m) :
  ∃ (m n : ℕ), (m + n = 6) → (1 / m + 4 / n) = 3 / 2 :=
by 
  sorry

end min_value_geom_seq_l399_399816


namespace possible_r_squared_values_l399_399803

theorem possible_r_squared_values
  (P Q : Set ℕ)
  (hP : P = {x | x = 2 ∨ x = 1})
  (hQ : Q = {y | y = 1 ∨ y = 2 ∨ y = 3 ∨ y = 4 ∨ y = 5 ∨ y = 6 ∨ y = 7 ∨ y = 8 ∨ y = 9})
  (hPQ : P ⊆ Q)
  (condition: ∀ (x y : ℕ), (x = 2 ∧ y ∈ {3, 4, 5, 6, 7, 8, 9}) ∨ (x = y ∧ x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) →
    ((∃ (r^2 : ℕ), (4 / 14) = (2 / 7) ↔ 4 < r^2 < 32) ∧
    ∀ (p : ℕ × ℕ), p.fst = 2 ∧ p.snd ∈ {3, 4, 5, 6, 7, 8, 9} ∧ p.fst^2 + p.snd^2 < r^2)) :
  (r^2 = 30 ∨ r^2 = 31) :=
sorry

end possible_r_squared_values_l399_399803


namespace number_of_ordered_quadruples_div_100_l399_399520

-- Definition of the problem statement
theorem number_of_ordered_quadruples_div_100 : 
  let m := {quadruples | ∃ (x1 x2 x3 x4 : ℕ), 2 ∣ x1 ∧ x1 ≠ 0 ∧ 2 ∣ x2 ∧ x2 ≠ 0 ∧ 
                              2 ∣ x3 ∧ x3 ≠ 0 ∧ 2 ∣ x4 ∧ x4 ≠ 0 ∧ 
                              x1 % 2 = 1 ∧ x2 % 2 = 1 ∧ x3 % 2 = 1 ∧ x4 % 2 = 1 ∧ 
                              x1 + x2 + x3 + x4 = 66} in
  (m.card : ℝ) / 100 = 59.84 :=
begin
  sorry
end

end number_of_ordered_quadruples_div_100_l399_399520


namespace correct_log_values_l399_399240

variables {a b c : ℝ}

namespace LogProofs

-- Conditions
def log_two : ℝ := 1 - a - c
def log_five : ℝ := a + c
def log_three : ℝ := 2 * a - b
def log_six : ℝ := 1 + a - b - c

-- Theorem to prove
theorem correct_log_values :
  (log 1.5 = 3 * a - b + c - 1) ∧ (log 7 = 2 * (b + c)) :=
sorry

end LogProofs

end correct_log_values_l399_399240


namespace last_integer_in_sequence_l399_399219

def seq_step (n : ℤ) : ℤ := n / 3

def seq : ℕ → ℤ
| 0     := 1234567
| (n+1) := seq_step (seq n)

theorem last_integer_in_sequence : ∃ N : ℕ, seq N = 2 ∧ ∀ n m : ℕ, n > m → seq n < 2 :=
by
  sorry

end last_integer_in_sequence_l399_399219


namespace general_formula_a_n_sum_first_n_terms_l399_399413

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}

-- Condition 1: The product of the first 5 terms of the geometric sequence is 243.
axiom product_condition {a_1 a_2 a_3 a_4 a_5 : ℝ} : a_1 * a_2 * a_3 * a_4 * a_5 = 243

-- Condition 2: The common ratio of the geometric sequence is not 1.
axiom common_ratio_condition {q : ℝ} : q ≠ 1

-- Condition 3: 2a_3 is the arithmetic mean of 3a_2 and a_4.
axiom arithmetic_mean_condition {a_2 a_3 a_4 : ℝ} : 2 * a_3 = (3 * a_2 + a_4) / 2

-- Condition 4: For the sequence b_n, it satisfies b_n = b_{n-1} * log_3(a_{n+2}), and b_1 = 1.
axiom b_sequence_condition {n : ℕ} (h : n ≥ 2) : b n = b (n - 1) * log 3 (a (n + 2))
axiom initial_b1_condition : b 1 = 1

-- Define a_n
def a (n : ℕ) : ℝ := 3^(n - 2)

-- The general formula for the term a_n is 3^(n-2).
theorem general_formula_a_n (n : ℕ) : a n = 3^(n - 2) :=
sorry

-- Define the sequence that uses b_n
def transformed_sequence (n : ℕ) : ℝ := (n - 1)! / b (n + 1)

-- Define the sum of the first n terms of the transformed sequence
def sum_transformed_sequence (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, transformed_sequence (k + 1))

-- The sum of the first n terms S_n of the sequence { (n-1)! / b_{n+1} } is n / (n+1).
theorem sum_first_n_terms (n : ℕ) : S n = n / (n + 1) :=
sorry

end general_formula_a_n_sum_first_n_terms_l399_399413


namespace solve_for_y_l399_399945

theorem solve_for_y (y : ℚ) : 
  3 + 1 / (1 + 1 / (3 + 3 / (4 + y))) = 169 / 53 → y = -605 / 119 :=
by
  intro h
  sorry

end solve_for_y_l399_399945


namespace yura_coins_l399_399633

theorem yura_coins (n : ℕ) (coins : Fin n → ℕ) (h_n : n = 2001)
  (h1 : ∀ i, coins i = 1 → i < n - 1 → coins (i + 1) ≠ 1)
  (h2 : ∀ i, coins i = 2 → i < n - 2 → coins (i + 1) ≠ 2 ∧ coins (i + 2) ≠ 2)
  (h3 : ∀ i, coins i = 3 → i < n - 3 → coins (i + 1) ≠ 3 ∧ coins (i + 2) ≠ 3 ∧ coins (i + 3) ≠ 3) :
  ∃ k, k = (500 : ℕ) ∨ k = (501 : ℕ) ∧ (∑ i in Finset.range n, if coins i = 3 then 1 else 0) = k := by
  sorry

end yura_coins_l399_399633


namespace volume_ratio_is_one_third_l399_399551

variables {A B C D A1 B1 C1 D1 : Type*} [normed_group A] [normed_group B] [normed_group C] [normed_group D] [normed_group A1] [normed_group B1] [normed_group C1] [normed_group D1]
variables (planes : list (normed_group)) (parallel_lines : list (normed_group))

-- Assuming the conditions in the problem
-- Points A1, B1, C1, D1 are on the planes of the faces of tetrahedron ABCD
-- Lines AA1, BB1, CC1, DD1 are parallel

def tetrahedron_volume_ratio (t1 t2 : Type*) [normed_group t1] [normed_group t2] (pts1 pts2 : list (normed_group)) (parallel : list (normed_group)) : ℚ :=
  if pts1 = [A, B, C, D] ∧ pts2 = [A1, B1, C1, D1] ∧ parallel = [A, A1, B, B1, C, C1, D, D1] then 1/3 else 0

theorem volume_ratio_is_one_third :
  tetrahedron_volume_ratio ABCD A1B1C1D1 [A, B, C, D] [A1, B1, C1, D1] [A, A1, B, B1, C, C1, D, D1] = 1/3 :=
sorry

end volume_ratio_is_one_third_l399_399551


namespace tickets_used_correct_l399_399694

def ferris_wheel_rides : ℕ := 7
def bumper_car_rides : ℕ := 3
def cost_per_ride : ℕ := 5

def total_rides : ℕ := ferris_wheel_rides + bumper_car_rides
def total_tickets_used : ℕ := total_rides * cost_per_ride

theorem tickets_used_correct : total_tickets_used = 50 := by
  sorry

end tickets_used_correct_l399_399694


namespace range_g_l399_399105

noncomputable def g (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

theorem range_g (T : set ℝ) (hT : ∀ y, y ∈ T ↔ ∃ x : ℝ, x > 0 ∧ g x = y) :
  (¬ ∃ m, ∀ y ∈ T, m ≤ y) ∧ (¬ ∃ M, ∀ y ∈ T, y ≤ M) :=
by
  sorry

end range_g_l399_399105


namespace probability_of_sum_20_is_correct_l399_399576

noncomputable def probability_sum_20 : ℚ :=
  let total_outcomes := 12 * 12
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_of_sum_20_is_correct :
  probability_sum_20 = 5 / 144 :=
by
  sorry

end probability_of_sum_20_is_correct_l399_399576


namespace sum_of_cubics_l399_399914

noncomputable def root_polynomial (x : ℝ) := 5 * x^3 + 2003 * x + 3005

theorem sum_of_cubics (a b c : ℝ)
  (h1 : root_polynomial a = 0)
  (h2 : root_polynomial b = 0)
  (h3 : root_polynomial c = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 :=
sorry

end sum_of_cubics_l399_399914


namespace combined_votes_l399_399723

theorem combined_votes {A B : ℕ} (h1 : A = 14) (h2 : 2 * B = A) : A + B = 21 := 
by 
sorry

end combined_votes_l399_399723


namespace max_ad_revenue_l399_399657

-- Definitions
def num_seconds := 120
def fee_15_sec_ad := 0.6
def fee_30_sec_ad := 1

-- Conditions
variables (x y : ℕ) 
-- Total time constraint
def total_time := 15 * x + 30 * y = num_seconds
-- Minimum airing constraints
def min_air_x := 2 ≤ x
def min_air_y := 2 ≤ y

-- Define the revenue calculation
def revenue := (fee_15_sec_ad * x + fee_30_sec_ad * y)

-- Main theorem statement
theorem max_ad_revenue 
  (h1 : total_time x y) 
  (h2 : min_air_x x) 
  (h3 : min_air_y y) : 
  revenue x y = 4.4 :=
sorry

end max_ad_revenue_l399_399657


namespace percentage_decrease_to_gain_30_percent_profit_l399_399670

theorem percentage_decrease_to_gain_30_percent_profit
  (C : ℝ) (P : ℝ) (S : ℝ) (S_new : ℝ) 
  (C_eq : C = 60)
  (S_eq : S = 1.25 * C)
  (S_new_eq1 : S_new = S - 12.60)
  (S_new_eq2 : S_new = 1.30 * (C - P * C)) : 
  P = 0.20 := by
  sorry

end percentage_decrease_to_gain_30_percent_profit_l399_399670


namespace integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l399_399383

theorem integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5 :
  ∀ n : ℤ, ∃ k : ℕ, (n^3 - 3 * n^2 + n + 2 = 5^k) ↔ n = 3 :=
by
  intro n
  exists sorry
  sorry

end integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l399_399383


namespace alex_working_day_ends_at_530_l399_399724

noncomputable def working_day_end_time (start_time lunch_start total_hours work_before_lunch lunch_duration : ℕ) : ℕ :=
  let resumed_work_time := lunch_start + lunch_duration
  let remaining_hours := total_hours - work_before_lunch
  resumed_work_time + remaining_hours

theorem alex_working_day_ends_at_530 :
  working_day_end_time 8 13 9 5 0.5 = 17.5 := 
sorry

end alex_working_day_ends_at_530_l399_399724


namespace incorrectA_l399_399631

-- Definitions based on the conditions in the problem
def conditionA : Prop := ∀ (atom: Type) (protons neutrons : ℕ), relativeAtomicMass(atom) = protons + neutrons
def conditionB : Prop := ∀ (crystals: Type), deliquescence(crystals) = absorbingWaterToFormSolution(crystals)
def conditionC : Prop := ∀ (metals: Type) (activitySeries : Type), inferSmeltingMethods(metals, activitySeries) = true
def conditionD : Prop := ∀ (solubilityTable: Type) (solutions: Type), canDoubleDisplacementOccur(solubilityTable, solutions) = true

-- The proof problem
theorem incorrectA (A B C D : Prop) : A → ¬A :=
begin
  sorry
end

-- Assumptions based on the propositions used in the theorem
axiom protons_neutrons_iso_mass : ∃ (atom: Type) (protons neutrons : ℕ), relativeAtomicMass(atom) ≠ protons + neutrons
axiom deliquescence_correct : conditionB
axiom smelting_correct : conditionC
axiom double_displacement_correct : conditionD

end incorrectA_l399_399631


namespace buffalo_theft_l399_399730

theorem buffalo_theft (initial_apples falling_apples remaining_apples stolen_apples : ℕ)
  (h1 : initial_apples = 79)
  (h2 : falling_apples = 26)
  (h3 : remaining_apples = 8) :
  initial_apples - falling_apples - stolen_apples = remaining_apples ↔ stolen_apples = 45 :=
by sorry

end buffalo_theft_l399_399730


namespace incorrect_value_initial_calculation_l399_399978

theorem incorrect_value_initial_calculation (initial_mean : ℝ) (correct_value : ℝ) (correct_mean : ℝ) : 
  initial_mean = 190 ∧ correct_value = 165 ∧ correct_mean = 191.4 → 
  ∃ (incorrect_value : ℝ), incorrect_value = 200 :=
by
  intro h
  cases h with h_initial h_correct
  cases h_correct with h_correct_value h_correct_mean
  use 200
  sorry

end incorrect_value_initial_calculation_l399_399978


namespace arithmetic_mean_reciprocals_first_four_primes_l399_399337

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l399_399337


namespace equidistant_point_A_l399_399389

def points : Type := (ℝ × ℝ × ℝ)

def A (x : ℝ) : points := (x, 0, 0)
def B : points := (-2, -4, -6)
def C : points := (-1, -2, -3)

def distance (p1 p2 : points) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem equidistant_point_A :
  ∃ x : ℝ, distance (A x) B = distance (A x) C ∧ A x = (-21, 0, 0) :=
by
  sorry

end equidistant_point_A_l399_399389


namespace monotonicity_and_no_real_roots_l399_399092

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * real.log x + 1

theorem monotonicity_and_no_real_roots 
  (a : ℝ) (ha : 0 < a) : 
  (∀ x : ℝ, (0 < x ∧ x < (1 / a) → deriv (f a) x < 0) ∧ (x > (1 / a) → deriv (f a) x > 0)) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≠ 0 → a > (1 / real.exp 1))) :=
begin
  sorry
end

end monotonicity_and_no_real_roots_l399_399092


namespace problem_intersection_complement_l399_399835

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 + x - 2 > 0}

def N : Set ℝ := {x | (1 / 2)^(x - 1) ≥ 2}

theorem problem_intersection_complement :
  ((U \ M) ∩ N) = {x | -2 ≤ x ∧ x ≤ 0} :=
by
  sorry

end problem_intersection_complement_l399_399835


namespace tan_alpha_value_l399_399003

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin (Real.pi + α) = 3 / 5) 
  (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.tan α = 3 / 4 := 
sorry

end tan_alpha_value_l399_399003


namespace complementary_angle_decrease_l399_399211

theorem complementary_angle_decrease (α β : ℝ) (h1 : α + β = 90) (h2 : α / β = 3 / 7) : 
  (∃ new_α : ℝ, new_α = α * 1.2) →
  ∃ new_β : ℝ, new_β = (1 - 0.0857) * β :=
by
  intro h3
  cases h3 with new_α h_newα
  use 90 - new_α
  sorry

end complementary_angle_decrease_l399_399211


namespace tan_of_A_l399_399036

theorem tan_of_A (A B C : Type) [IsTriangle A B C] (hC : angle C = 90) (hAB : distance A B = 13) (hBC : distance BC = 5) : 
  tangent A = 5 / 12 :=
by
  sorry

end tan_of_A_l399_399036


namespace O_bisects_OK_l399_399106

noncomputable theory
open_locale classical

variables {A B C O C1 D K : Type} [euclidean_geometry A B C O C1 D K]

-- Given:
-- O is the center of the circumscribed circle of the acute-angled, scalene triangle ABC
axiom circumscribed_circle_center (ABC : triangle) : point.is_center O ABC

-- C1 is the point symmetric to C with respect to O
axiom symmetric_point (C O C1 : point) : symmetric C O = C1

-- D is the midpoint of side AB
axiom midpoint (A B D : point) : midpoint D A B

-- K is the center of the circumscribed circle of triangle ODC1
axiom circumscribed_circle_center_ODC1 (ODC1 : triangle) : point.is_center K ODC1

-- Prove:
theorem O_bisects_OK :
  let OK_line := line_segment O K in
  let midpoint_OK := midpoint_ct OK_line O in
  midpoint_OK = O :=
sorry

end O_bisects_OK_l399_399106


namespace dot_product_constant_l399_399805

variables {P A B C : Type} [inner_product_space ℝ P]
variables (a b : P)

def on_side_bc (P : P) (B C : P) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • B + t • C

variables (h_bc : on_side_bc P B C)
variables (h_a_squared : ⟪a, a⟫ = 16) (h_b_squared : ⟪b, b⟫ = 16)
variables (h_a_dot_b : ⟪a, b⟫ = 8)

theorem dot_product_constant :
  ⟪P • (a + b), a + b⟫ = 24 := 
sorry

end dot_product_constant_l399_399805


namespace algae_population_exceeds_2000_l399_399481

/-
In a specific culture of algae, the population triples every day. 
The culture starts with 5 algae. Determine the number of the first day 
which ends with the colony having more than 2000 algae.
-/
noncomputable def population (n : ℕ) : ℕ := 5 * 3^n

theorem algae_population_exceeds_2000 :
  ∃ n : ℕ, 5 * 3 ^ n > 2000 ∧ ∀ m < n, 5 * 3 ^ m ≤ 2000 :=
by
  exists 7
  split
  simp [population]
  norm_num
  intros m hm
  simp [population]
  norm_num
  sorry

end algae_population_exceeds_2000_l399_399481


namespace surfaceAreaOfCircumscribedSphere_l399_399304

-- Define given conditions
def isRightTriangularPyramidBase (a : ℝ) (baseSideLength : ℝ) : Prop :=
  baseSideLength = real.sqrt 2

-- The theorem to be proved
theorem surfaceAreaOfCircumscribedSphere (a : ℝ) (baseSideLength : ℝ) (h : isRightTriangularPyramidBase a baseSideLength) : 
  (baseSideLength = real.sqrt 2) → 
  ∃ (r : ℝ), 4 * real.pi * r^2 = 3 * real.pi :=
by {
  intros,
  sorry
}

end surfaceAreaOfCircumscribedSphere_l399_399304


namespace problem_l399_399912

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry
noncomputable def c : ℂ := sorry

def condition1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 := sorry
def condition2 : a + b + c = 30 := sorry
def condition3 : (a - b) ^ 2 + (a - c) ^ 2 + (b - c) ^ 2 = 2 * a * b * c := sorry

theorem problem : condition1 ∧ condition2 ∧ condition3 → (a^3 + b^3 + c^3) / (a * b * c) = 33 := 
by 
  sorry

end problem_l399_399912


namespace weight_loss_percentage_l399_399263

theorem weight_loss_percentage (W : ℝ) :
  let final_weight := W * 0.85 * 1.03 * 0.99 * 1.02 in
  (1 - final_weight / W) * 100 ≈ 8.08 :=
by
  sorry

end weight_loss_percentage_l399_399263


namespace no_extreme_points_f_l399_399582

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x

theorem no_extreme_points_f : ∀ f : ℝ → ℝ, f = (λ x, x^3 - 3 * x^2 + 3 * x) → (∀ x, ∂ ∂x f(x) = 3 * (x - 1)^2) → ∃! x, ∂ ∂x f(x) = 0 := 
by
  intros f hf hderiv
  sorry

end no_extreme_points_f_l399_399582


namespace degree_of_g_is_five_l399_399005

-- Define f as given in the problem
noncomputable def f : polynomial ℝ := -9 * X^5 + 2 * X^3 - 3 * X^2 + 7

-- Define the condition that the degree of f(x) + g(x) is 3
def condition (g : polynomial ℝ) : Prop :=
  (f + g).degree = 3

-- The theorem to prove: there exists a polynomial g such that its degree is 5 when the sum f(x) + g(x) has degree 3
theorem degree_of_g_is_five (g : polynomial ℝ) (h : condition g) : g.degree = 5 := sorry

end degree_of_g_is_five_l399_399005


namespace arithmetic_sequence_S_12_l399_399859

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Assume an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Assume all terms are positive
def all_terms_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

-- Given condition
def given_condition (a : ℕ → ℝ) : Prop :=
  a 3 * a 5 + a 3 * a 8 + a 5 * a 10 + a 8 * a 10 = 64

-- Sum of the first 12 terms of the arithmetic sequence
def S_12 (a : ℕ → ℝ) : ℝ :=
  ∑ i in finset.range 12, a i

-- Main theorem to prove:
theorem arithmetic_sequence_S_12 (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : all_terms_positive a) (h3 : given_condition a) :
  S_12 a = 48 :=
by
  sorry

end arithmetic_sequence_S_12_l399_399859


namespace red_paint_cans_l399_399126

/--
If the ratio of red paint to white paint is 4 to 3, and there are 35 cans of the mixture in total, then the number of cans of red paint needed is 20.
-/
theorem red_paint_cans (total_cans : ℕ) (red_to_white_ratio : ℕ) (red_ratio : ℕ) : 
  total_cans = 35 ∧ red_to_white_ratio = 3 ∧ red_ratio = 4 → 
  4 * total_cans / (4 + 3) = 20 :=
begin
  sorry
end

end red_paint_cans_l399_399126


namespace no_solution_in_positive_integers_l399_399554

theorem no_solution_in_positive_integers
    (x y : ℕ)
    (h : x > 0 ∧ y > 0) :
    x^2006 - 4 * y^2006 - 2006 ≠ 4 * y^2007 + 2007 * y :=
by
  sorry

end no_solution_in_positive_integers_l399_399554


namespace min_lcm_of_triplet_l399_399791

open Nat

def gcd (a b : ℕ) : ℕ := nat.gcd a b

def lcm (a b : ℕ) : ℕ := (a * b) / gcd a b

theorem min_lcm_of_triplet (a b c : ℕ) (h₁ : lcm a b = 12) (h₂ : lcm b c = 15) : 
  ∃ (min_value : ℕ), min_value = 20 ∧ min_value = lcm a c :=
by
  sorry -- The proof goes here; omitted as per instructions

end min_lcm_of_triplet_l399_399791


namespace plot_length_57_l399_399272

def length_of_plot (breadth : ℕ) (cost_per_meter total_cost : ℕ) : ℕ :=
  2 * (breadth + breadth + 14)

theorem plot_length_57 :
  ∀ (b : ℕ) (cost_per_meter total_cost : ℕ),
  26.50 * (length_of_plot b) = 5300 →
  b = 43 →
  length_of_plot b = 57 := 
by
  intros b cost_per_meter total_cost h_cost h_breadth
  rw h_breadth
  sorry

end plot_length_57_l399_399272


namespace ones_digit_of_sum_is_0_l399_399247

-- Define the integer n
def n : ℕ := 2012

-- Define the ones digit function
def ones_digit (x : ℕ) : ℕ := x % 10

-- Define the power function mod 10
def power_mod_10 (d a : ℕ) : ℕ := (d^a) % 10

-- Define the sequence sum for ones digits
def seq_sum_mod_10 (m : ℕ) : ℕ :=
  Finset.sum (Finset.range m) (λ k => power_mod_10 (k+1) n)

-- Define the final sum mod 10 considering the repeating cycle and sum
def total_ones_digit_sum (a b : ℕ) : ℕ :=
  let cycle_sum := Finset.sum (Finset.range 10) (λ k => power_mod_10 (k+1) n)
  let s := cycle_sum * (a / 10) + Finset.sum (Finset.range b) (λ k => power_mod_10 (k+1) n)
  s % 10

-- Prove that the ones digit of the sum is 0
theorem ones_digit_of_sum_is_0 : total_ones_digit_sum n (n % 10) = 0 :=
sorry

end ones_digit_of_sum_is_0_l399_399247


namespace Namjoon_walk_extra_l399_399541

-- Define the usual distance Namjoon walks to school
def usual_distance := 1.2

-- Define the distance Namjoon walked to the intermediate point
def intermediate_distance := 0.3

-- Define the total distance Namjoon walked today
def total_distance_today := (intermediate_distance * 2) + usual_distance

-- Define the extra distance walked today compared to usual
def extra_distance := total_distance_today - usual_distance

-- State the theorem to prove that the extra distance walked today is 0.6 km
theorem Namjoon_walk_extra : extra_distance = 0.6 := 
by
  sorry

end Namjoon_walk_extra_l399_399541


namespace apple_pie_cost_per_serving_l399_399454

noncomputable def granny_smith_price : ℝ := 1.80
noncomputable def gala_price : ℝ := 2.20
noncomputable def honeycrisp_price : ℝ := 2.50
noncomputable def pie_crust_cost : ℝ := 2.50
noncomputable def lemon_cost : ℝ := 0.60
noncomputable def butter_cost : ℝ := 1.80

noncomputable def granny_smith_weight : ℝ := 0.5
noncomputable def gala_weight : ℝ := 0.8
noncomputable def honeycrisp_weight : ℝ := 0.7

noncomputable def discount_rate : ℝ := 0.20

noncomputable def num_servings : ℕ := 8

theorem apple_pie_cost_per_serving :
  let granny_smith_discount_price := granny_smith_price * (1 - discount_rate)
  let granny_smith_total := granny_smith_weight * granny_smith_discount_price
  let gala_total := gala_weight * gala_price
  let honeycrisp_total := honeycrisp_weight * honeycrisp_price
  let total_apple_cost := granny_smith_total + gala_total + honeycrisp_total
  let other_ingredients_cost := pie_crust_cost + lemon_cost + butter_cost
  let total_cost := total_apple_cost + other_ingredients_cost
  let cost_per_serving := total_cost / num_servings
  (Real.round (cost_per_serving * 100) / 100 = 1.14)
:= by
  sorry

end apple_pie_cost_per_serving_l399_399454


namespace tangent_line_equation_correct_l399_399390

-- Define the circle
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

-- Define the tangent condition
def is_tangent_line (line : ℝ → ℝ → Prop) (x₀ y₀ : ℝ) : Prop :=
  ∀ (x y : ℝ), circle x y → (x, y) ≠ (x₀, y₀) → 
    (line x y = 0) ∨ 
    (((y - 2) / (x - 1) = 2) → ((2 * y - y₀) / (x₀ - x) = -2))

-- The point P
def point_P : ℝ × ℝ := (2, 4)

-- The tangent line function
def tangent_line (x y : ℝ) : Prop := x + 2 * y - 10 = 0

-- The theorem to be proved
theorem tangent_line_equation_correct :
  is_tangent_line tangent_line point_P.1 point_P.2 :=
sorry

end tangent_line_equation_correct_l399_399390


namespace radius_of_ade_eq_distance_between_centers_l399_399494

-- Define triangle ABC and its properties
variables (A B C D E : Type) 
           [MetricSpace A] [MetricSpace B] [MetricSpace C] 
           [MetricSpace D] [MetricSpace E]
           (triangle_ABC : Triangle A B C)
           (BC : ℝ)
           (smallest_side : BC ≤ side_length triangle_ABC BC)
           (BD_CE_on_rays : ∀ D E, D ∈ ray B A ∧ E ∈ ray C A ∧ BD = BC ∧ CE = BC)

-- Centers and distances definitions
variables (O O1 : Type)
           [Center O triangle_ABC.Incircle]
           [Center O1 triangle_ABC.Circumcircle]

-- Main theorem statement
theorem radius_of_ade_eq_distance_between_centers :
    (radius (circumscribed_circle (triangle_abc D E A)) = distance O O1) := by
  sorry

end radius_of_ade_eq_distance_between_centers_l399_399494


namespace evaluate_expression_l399_399727

theorem evaluate_expression (a b c : ℤ) 
  (h1 : c = b - 12) 
  (h2 : b = a + 4) 
  (h3 : a = 5) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  ((a + 3) / (a + 2) * (b + 1) / (b - 3) * (c + 10) / (c + 7) = 10 / 3) :=
by
  sorry

end evaluate_expression_l399_399727


namespace matrix_property_l399_399530

theorem matrix_property 
  {a b c d : ℝ} 
  (h1 : (matrix.transpose ![[a, b], [c, d]]) = (matrix.inverse ![[a, b], [c, d]]))
  (h2 : matrix.det ![[a, b], [c, d]] = 1) :
  a^2 + b^2 + c^2 + d^2 = 2 := 
by 
  sorry

end matrix_property_l399_399530


namespace stripes_distance_l399_399306

theorem stripes_distance (d : ℝ) (L : ℝ) (c : ℝ) (y : ℝ) 
  (hd : d = 40) (hL : L = 50) (hc : c = 15)
  (h_ratio : y / d = c / L) : y = 12 :=
by
  rw [hd, hL, hc] at h_ratio
  sorry

end stripes_distance_l399_399306


namespace ice_cream_combination_l399_399849

/-- Ice-cream-o-rama problem: ice cream scoop combinations --/
theorem ice_cream_combination :
  let flavors : ℕ := 4
  let scoops : ℕ := 5
  combinatorics.choose (scoops + flavors - 1) (flavors - 1) = 56 := by
  sorry

end ice_cream_combination_l399_399849


namespace cubes_divisible_by_squares_l399_399647

theorem cubes_divisible_by_squares (m n : ℕ) 
  (h1 : m ≥ 16) 
  (h2 : n ≥ 24) :
  ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 9 ∧ 1 ≤ j ∧ j ≤ 9 ∧ i ≠ j →
  (∃ k l : ℕ,  a_i = 2^k * 3^l ∧ a_j = 2^(m+(i-1)) * 3^(n-(i-1)) ∧ 
  a_i^3 mod a_j^2 = 0) := 
sorry

end cubes_divisible_by_squares_l399_399647


namespace fourth_metal_mass_l399_399295

noncomputable def alloy_mass_problem (m1 m2 m3 m4 : ℝ) : Prop :=
  m1 + m2 + m3 + m4 = 20 ∧
  m1 = 1.5 * m2 ∧
  m2 = (3 / 4) * m3 ∧
  m3 = (5 / 6) * m4

theorem fourth_metal_mass :
  ∃ (m4 : ℝ), (∃ (m1 m2 m3 : ℝ), alloy_mass_problem m1 m2 m3 m4) ∧ abs (m4 - 5.89) < 0.01 :=
begin
  sorry  -- Proof is skipped
end

end fourth_metal_mass_l399_399295


namespace monotonicity_and_range_l399_399086

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end monotonicity_and_range_l399_399086


namespace shaded_area_is_six_pi_l399_399880

-- Define the geometric conditions
def center_point := Point

structure Circle (O : center_point) (r : ℝ) :=
  (center : O)
  (radius : r)

-- Given circles C1 and C2 with specific radius conditions
def C1 := Circle center_point (Real.sqrt 2)
def C2 := Circle center_point (2 * Real.sqrt 2)

-- Define the area of a circle
def circle_area (c : Circle center_point) : ℝ := π * (c.radius ^ 2)

-- Define the area of the shaded region as the difference between the areas of C2 and C1
def shaded_area : ℝ := circle_area C2 - circle_area C1

-- The theorem to prove
theorem shaded_area_is_six_pi : shaded_area = 6 * π :=
  sorry

end shaded_area_is_six_pi_l399_399880


namespace hyperbola_vertex_distance_l399_399770

theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ),
  (x^2 / 144 - y^2 / 49 = 1) →
  ∃ (a : ℝ), a = 12 ∧ 2 * a = 24 :=
by
  intro x y h
  have h1 : 12^2 = 144 := by norm_num
  use 12
  split
  case left =>
    exact rfl
  case right =>
    calc
      2 * 12 = 24 : by norm_num

end hyperbola_vertex_distance_l399_399770


namespace percentage_of_girls_in_school_l399_399590

noncomputable def percentage_of_girls (B G : ℕ) : ℝ :=
  (G.to_real / (B + G).to_real) * 100

theorem percentage_of_girls_in_school :
  ∀ (B G : ℕ), B = 92 → B + G = 1150 → percentage_of_girls B G ≈ 91.913043478 :=
by sorry

end percentage_of_girls_in_school_l399_399590


namespace problem_statement_l399_399103

noncomputable def alpha : ℝ := 3 + Real.sqrt 8
noncomputable def x : ℝ := alpha ^ 1000
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := by
  sorry

end problem_statement_l399_399103


namespace exponent_division_l399_399788

theorem exponent_division (m n : ℕ) (h : m - n = 1) : 5 ^ m / 5 ^ n = 5 :=
by {
  sorry
}

end exponent_division_l399_399788


namespace word_count_50A_50B_l399_399704

theorem word_count_50A_50B :
  (∃ (w : Fin 100 → Fin 2), 
  (∀ (i : Fin 100), w i = 0 ∨ w i = 1) ∧ 
  (Finset.card (Finset.filter (λ x, w x = 0) (Finset.range 100)) = 50) ∧ 
  (Finset.card (Finset.filter (λ x, w x = 1) (Finset.range 100)) = 50) ∧ 
  (∀ (i : Fin 100), 
    Finset.card (Finset.filter (λ x, w x = 0) (Finset.range (i+1))) ≤ 
    Finset.card (Finset.filter (λ x, w x = 1) (Finset.range (i+1)))) ∧ 
  (∀ (i : Fin 14), (44 ≤ i ∧ i < 57) → 
    (w (⟨i, nat.lt_succ_iff.mpr (nat.le_trans (nat.le_of_succ_le_pred (nat.succ_le_succ (nat.le_of_lt (nat.lt_of_succ_le nat.le_refl))))) 
                                          (by linarith [-1<58])) = 1) → 
    (w (⟨i+1, by linarith⟩) = 1))) = (Nat.choose 85 35 - Nat.choose 85 34) := 
sorry

end word_count_50A_50B_l399_399704


namespace number_of_lineups_l399_399867

-- Define the conditions
def TeamSize : ℕ := 5
def TotalPlayers : ℕ := 8
def Centers : ℕ := 2
def PointGuards : ℕ := 2
def OtherPlayers : ℕ := 4
def OneCenter : ℕ := 1
def MinPointGuards : ℕ := 1

-- Define combinations function
noncomputable def C (n k : ℕ) : ℕ := nat.choose n k

-- Define Case 1 and Case 2
noncomputable def case1 : ℕ := C PointGuards 1 * C Centers 1 * C OtherPlayers 3
noncomputable def case2 : ℕ := C PointGuards 2 * C Centers 1 * C OtherPlayers 2

-- Define total lineups
noncomputable def totalLineups : ℕ := case1 + case2

-- The proof problem
theorem number_of_lineups : totalLineups = 28 := by
  sorry

end number_of_lineups_l399_399867


namespace median_of_set_l399_399580

noncomputable def y := 92
def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

theorem median_of_set :
  mean [92, 90, 86, 88, 89, y] = 89.5 →
  let set := (List.cons y [92, 90, 86, 88, 89]).sort (<=)
  set.nthLe 2 (by decide) + set.nthLe 3 (by decide) / 2 = 89.5 :=
by
  sorry

end median_of_set_l399_399580


namespace large_beads_per_bracelet_l399_399696

-- Conditions
variable (total_beads : ℕ) (equal_large_small : total_beads % 2 = 0) 
variable (beads_per_bracelet : ℕ → ℕ)

-- 528 beads in total
def beads_in_total : total_beads = 528 := rfl

-- Equal amounts of large and small beads
def equal_beads_amount : total_beads / 2 = 264 := by 
  rw [←eq.symm equal_large_small, nat.add_sub_cancel, nat.add_comm, nat.div_add_mod]

-- Each bracelet takes twice as many small beads as large beads
def beads_relationship (L : ℕ) : beads_per_bracelet L = 2 * L := by 
  sorry

-- Caitlin can make 11 bracelets 
def bracelets_capacity : 11 * beads_per_bracelet 24 = 264 := by 
  sorry

-- Prove the number of large beads each bracelet uses is 24
theorem large_beads_per_bracelet : (total_beads / 2) / 11 = 24 := by 
  calc
    (total_beads / 2) / 11 = 264 / 11 : by rw equal_beads_amount
                       ... = 24       : by norm_num

end large_beads_per_bracelet_l399_399696


namespace tom_has_9_balloons_l399_399602

-- Define Tom's and Sara's yellow balloon counts
variables (total_balloons saras_balloons toms_balloons : ℕ)

-- Given conditions
axiom total_balloons_def : total_balloons = 17
axiom saras_balloons_def : saras_balloons = 8
axiom toms_balloons_total : toms_balloons + saras_balloons = total_balloons

-- Theorem stating that Tom has 9 yellow balloons
theorem tom_has_9_balloons : toms_balloons = 9 := by
  sorry

end tom_has_9_balloons_l399_399602


namespace geometric_series_sum_l399_399350

theorem geometric_series_sum :
  let a := 1
  let r := (1 : ℝ) / 5
  ∑' n : ℕ, a * r ^ n = 5 / 4 :=
by
  sorry

end geometric_series_sum_l399_399350


namespace tom_ate_total_calories_l399_399235

variable (carrotCalories : ℝ) (broccoliCalories : ℝ) (spinachCalories : ℝ) (cauliflowerCalories : ℝ)
variable (totalCalories : ℝ)

-- Given conditions
def conditions := 
  carrotCalories = 51 ∧ 
  broccoliCalories = 1/4 * carrotCalories ∧
  spinachCalories = 0.6 * carrotCalories ∧
  cauliflowerCalories = 0.8 * broccoliCalories

-- Calculate total calories based on the weights of the vegetables Tom ate
def calculateTotalCalories (carrotCalories : ℝ) (broccoliCalories : ℝ) (spinachCalories : ℝ) (cauliflowerCalories : ℝ) : ℝ := 
  3 * carrotCalories + 2 * broccoliCalories + 1 * spinachCalories + 4 * cauliflowerCalories

-- Main statement to prove
theorem tom_ate_total_calories (h : conditions carrotCalories broccoliCalories spinachCalories cauliflowerCalories) : 
  totalCalories = 249.9 :=
by 
  unfold conditions at h
  unfold calculateTotalCalories
  sorry

end tom_ate_total_calories_l399_399235


namespace max_value_function_l399_399857

theorem max_value_function (x : ℝ) (h : x > 4) : -x + (1 / (4 - x)) ≤ -6 :=
sorry

end max_value_function_l399_399857


namespace evaluate_f_at_3_l399_399423

noncomputable def f : ℝ → ℝ := sorry

axiom h_odd : ∀ x : ℝ, f (-x) = -f x 
axiom h_periodic : ∀ x : ℝ, f x = f (x + 4)
axiom h_def : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem evaluate_f_at_3 : f 3 = -2 := by
  sorry

end evaluate_f_at_3_l399_399423


namespace graph_symmetric_intersect_one_point_l399_399966

theorem graph_symmetric_intersect_one_point (a b c d : ℝ) :
  (∀ x : ℝ, 2a + (1 / (x - b)) = 2c + (1 / (x - d)) = a + c) →
  (x = (b + d) / 2) →
  ((a - c) * (b - d) = 2) := 
sorry

end graph_symmetric_intersect_one_point_l399_399966


namespace monotonicity_of_f_range_of_a_if_no_zeros_l399_399070

noncomputable def f (a x : ℝ) := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_of_f (a : ℝ) (h : a > 0) :
  (∀ x, x > 0 → x < 1/a → deriv (f a) x < 0) ∧
  (∀ x, x > 1/a → deriv (f a) x > 0) := sorry

theorem range_of_a_if_no_zeros 
  (h1 : ∀ x > 0, f a x ≠ 0) : a > 1 / Real.exp 1 := sorry

end monotonicity_of_f_range_of_a_if_no_zeros_l399_399070


namespace parabola_focus_distance_l399_399430

open Real (sqrt)

theorem parabola_focus_distance {x y : ℝ} (h : y^2 = 8 * x) (hx : x + 2 = 4 ∨ x + 2 = 4) : 
  (sqrt (x^2 + y^2) = 2 * sqrt 5) :=
begin
  sorry
end

end parabola_focus_distance_l399_399430


namespace seventh_grade_problem_l399_399869

theorem seventh_grade_problem (x y : ℕ) (h1 : x + y = 12) (h2 : 6 * x = 3 * 4 * y) :
  (x + y = 12 ∧ 6 * x = 3 * 4 * y) :=
by
  apply And.intro
  . exact h1
  . exact h2

end seventh_grade_problem_l399_399869


namespace largest_trig_function_l399_399624

theorem largest_trig_function (x : ℝ) (hx: 0 < x ∧ x < π / 4) :
  max (max (sin (cos x)) (sin (sin x))) (max (cos (sin x)) (cos (cos x))) = cos (sin x) :=
by
  sorry

end largest_trig_function_l399_399624


namespace equal_shared_expenses_l399_399893

theorem equal_shared_expenses (A B C : ℝ) : 
  let shared_excluding_C := A + B - C in
  let equal_share := shared_excluding_C / 2 in
  let leroy_additional_payment := equal_share - A in
  leroy_additional_payment = (B - A - C) / 2 :=
by
  sorry

end equal_shared_expenses_l399_399893


namespace eden_stuffed_bears_l399_399712

theorem eden_stuffed_bears
  (initial_bears : ℕ)
  (favorite_bears : ℕ)
  (sisters : ℕ)
  (eden_initial_bears : ℕ)
  (remaining_bears := initial_bears - favorite_bears)
  (bears_per_sister := remaining_bears / sisters)
  (eden_bears_now := eden_initial_bears + bears_per_sister)
  (h1 : initial_bears = 20)
  (h2 : favorite_bears = 8)
  (h3 : sisters = 3)
  (h4 : eden_initial_bears = 10) :
  eden_bears_now = 14 := by
{
  sorry
}

end eden_stuffed_bears_l399_399712


namespace wire_cutting_segments_l399_399402

theorem wire_cutting_segments :
  ∀ (lengths : List ℕ),
  lengths = [1008, 1260, 882, 1134] →
  let gcd := Nat.gcd 1008 (Nat.gcd 1260 (Nat.gcd 882 1134))
  gcd = 126 ∧ 
  List.sum (List.map (λ x, x / gcd) lengths) = 34 :=
by {
  intros lengths h,
  let gcd := Nat.gcd 1008 (Nat.gcd 1260 (Nat.gcd 882 1134)),
  have h1 : gcd = 126 := sorry,
  have h2 : List.sum (List.map (λ x, x / gcd) lengths) = 34 := sorry,
  exact ⟨h1, h2⟩
}

end wire_cutting_segments_l399_399402


namespace exists_sum_of_distinct_divisors_harmonic_series_lower_bound_l399_399653

-- Part (a)
theorem exists_sum_of_distinct_divisors (k : ℕ) (h : k ≥ 3) : 
  ∃ (n : ℕ), ∃ (d : finset ℕ), d.card = k ∧ (∀ x ∈ d, x ∣ n) ∧ d.sum = n :=
sorry

-- Part (b)
theorem harmonic_series_lower_bound (n k p : ℕ) (h : k ≥ 3)
    (h_n : ∃ (d : finset ℕ), d.card = k ∧ (∀ x ∈ d, x ∣ n) ∧ d.sum = n)
    (h_p : p.prime ∧ ∀ q : ℕ, q.prime ∧ q ∣ n → p ≤ q) :
  (∑ i in finset.range k, 1 / (p + i : ℝ)) ≥ 1 :=
sorry

end exists_sum_of_distinct_divisors_harmonic_series_lower_bound_l399_399653


namespace complementary_angles_decrease_percentage_l399_399202

theorem complementary_angles_decrease_percentage :
  ∀ (x : ℝ), (3 * x + 7 * x = 90) →
  (3 * x * 1.2 + 7 * x = 90) →
  (3 * x > 0) →
  (7 * x > 0) →
  let original_larger_angle : ℝ := 7 * x in
  let new_smaller_angle : ℝ := 3 * x * 1.2 in
  let new_larger_angle : ℝ := 90 - new_smaller_angle in
  let decrease : ℝ := original_larger_angle - new_larger_angle in
  let percentage_decrease : ℝ := (decrease / original_larger_angle) * 100 in
  percentage_decrease = 8.57 := 
sorry

end complementary_angles_decrease_percentage_l399_399202


namespace orange_balls_count_l399_399016

theorem orange_balls_count (P_black : ℚ) (O : ℕ) (total_balls : ℕ) 
  (condition1 : total_balls = O + 7 + 6) 
  (condition2 : P_black = 7 / total_balls) 
  (condition3 : P_black = 0.38095238095238093) :
  O = 5 := 
by
  sorry

end orange_balls_count_l399_399016


namespace fa_fb_value_l399_399830

-- Definitions and conditions
def parametric_eq_C1 := ∃ t : ℝ, (1 + (1/2) * t, (sqrt 3 / 2) * t)
def polar_eq_C2 (ρ θ : ℝ) := ρ^2 = 12 / (3 + (sin θ)^2)
def ordinary_eq_C1 (x y : ℝ) := y = sqrt 3 * (x - 1)
def rect_eq_C2 (x y : ℝ) := x^2 / 4 + y^2 / 3 = 1
def intersection_points (A B F : ℝ × ℝ) := ∃ t1 t2 : ℝ, 
  (A = (1 + (1/2) * t1, (sqrt 3 / 2) * t1)) ∧
  (B = (1 + (1/2) * t2, (sqrt 3 / 2) * t2)) ∧
  (rect_eq_C2 (1 + (1/2) * t1) ((sqrt 3 / 2) * t1)) ∧ 
  (rect_eq_C2 (1 + (1/2) * t2) ((sqrt 3 / 2) * t2))

-- Main proof statement
theorem fa_fb_value (A B F : ℝ × ℝ) (hF : F = (1, 0)) : 
  intersection_points A B F → 
  ∃ FA FB : ℝ, 1/|FA| + 1/|FB| = 4/3 := 
by 
  sorry

end fa_fb_value_l399_399830


namespace correct_calculation_of_mistake_l399_399282

theorem correct_calculation_of_mistake (x : ℝ) (h : x - 48 = 52) : x + 48 = 148 :=
by
  sorry

end correct_calculation_of_mistake_l399_399282


namespace shirt_wallet_ratio_l399_399117

theorem shirt_wallet_ratio
  (F W S : ℕ)
  (hF : F = 30)
  (hW : W = F + 60)
  (h_total : S + W + F = 150) :
  S / W = 1 / 3 := by
  sorry

end shirt_wallet_ratio_l399_399117


namespace simplify_expression_l399_399562

theorem simplify_expression :
  (1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3) = 1 / 39 :=
by
  sorry

end simplify_expression_l399_399562


namespace inequality_l399_399826

variables (a m x x1 x2 : ℝ)

def f (x : ℝ) : ℝ := 2 * (real.log x)

def g (x : ℝ) : ℝ := x + (1/x) + (2/x)

def h (x : ℝ) : ℝ := g x - f x

axiom extremum_point_h : h 3 = 0

axiom roots (f_eq_zero : ∀ x, f x + m * x = 0) (x1 x2 : ℝ) : x2 / x1 ≥ real.exp (f' x2 / f' x1)

theorem inequality (f' : ℝ → ℝ) :
  roots (λ x => 2 * (real.log x) + m * x) x1 x2 → (f' (x1 + x2) + m) / f' (x1 - x2) > 6 / 5 := sorry

end inequality_l399_399826


namespace solve_equation_l399_399649

theorem solve_equation (x : ℝ) (floor_x : ℝ) (h : floor_x = ⌊x⌋) :
  x^2 - floor_x = 2019 → (x = -sqrt 1974) ∨ (x = sqrt 2064) :=
by
  sorry

end solve_equation_l399_399649


namespace max_value_of_A_l399_399644

theorem max_value_of_A (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) / (a^3 + b^3 + c^3 - 2 * a * b * c) ≤ 6 :=
sorry

end max_value_of_A_l399_399644


namespace rectangle_area_with_complex_value_l399_399279

theorem rectangle_area_with_complex_value (ABCD : Type) [is_rectangle ABCD]
  (A B C D E F B' C' : ABCD) (BE CF : ℝ) (h_BE_CF : BE > CF)
  (AB'_eq_7 : dist A B' = 7) (BE_eq_31 : dist B E = 31)
  (folding_condition : ∠ A B' C' = ∠ B' E A):
  ∃ (a b c : ℤ), (c ≥ 0 ∧ ¬(∃ p : ℤ, prime p ∧ p^2 ∣ c)) ∧
                let area := (a : ℝ) + b * real.sqrt (c : ℝ) in a + b + c = 400 :=
by
  sorry  

end rectangle_area_with_complex_value_l399_399279


namespace total_cost_of_items_l399_399574

theorem total_cost_of_items (m n : ℕ) : (8 * m + 5 * n) = 8 * m + 5 * n := 
by sorry

end total_cost_of_items_l399_399574


namespace points_subtracted_each_incorrect_answer_l399_399929

theorem points_subtracted_each_incorrect_answer :
  ∃ x : ℚ, (∀ total_questions correct_answers final_score incorrect_answers : ℕ,
    total_questions = 120 ∧ 
    correct_answers = 104 ∧ 
    final_score = 100 ∧ 
    incorrect_answers = total_questions - correct_answers →
    final_score = correct_answers - incorrect_answers * (x : ℚ)) ∧ x = 1 / 4 :=
begin
  sorry
end

end points_subtracted_each_incorrect_answer_l399_399929


namespace crabapple_sequences_l399_399946

theorem crabapple_sequences (students : ℕ) (days : ℕ) (h_students : students = 11) (h_days : days = 5) :
  students ^ days = 161051 :=
by
  rw [h_students, h_days]
  norm_num
  sorry

end crabapple_sequences_l399_399946


namespace grandmother_mistaken_l399_399121

-- Definitions of the given conditions:
variables (N : ℕ) (x n : ℕ)
variable (initial_split : N % 4 = 0)

-- Conditions
axiom cows_survived : 4 * (N / 4) / 5 = N / 5
axiom horses_pigs : x = N / 4 - N / 5
axiom rabbit_ratio : (N / 4 - n) = 5 / 14 * (N / 5 + N / 4 + N / 4 - n)

-- Goal: Prove the grandmother is mistaken, i.e., some species avoided casualties
theorem grandmother_mistaken : n = 0 :=
sorry

end grandmother_mistaken_l399_399121


namespace arithmetic_mean_reciprocals_primes_l399_399334

theorem arithmetic_mean_reciprocals_primes : 
  let p := [2, 3, 5, 7] in 
  let reciprocals := p.map (λ n => 1 / (n : ℚ)) in
  (reciprocals.sum / reciprocals.length) = (247 / 840 : ℚ) :=
by
  sorry

end arithmetic_mean_reciprocals_primes_l399_399334


namespace distance_between_hyperbola_vertices_l399_399756

theorem distance_between_hyperbola_vertices :
  (∃ a : ℝ, a = real.sqrt 144 ∧ ∀ d : ℝ, d = 2 * a → d = 24) :=
begin
  use real.sqrt 144,
  split,
  { refl },
  { intros d hd,
    rw hd,
    refl }
end

end distance_between_hyperbola_vertices_l399_399756


namespace trapezoid_PQRS_PQ_squared_l399_399026

theorem trapezoid_PQRS_PQ_squared
  (PR PS PQ : ℝ)
  (cond1 : PR = 13)
  (cond2 : PS = 17)
  (h : PQ^2 + PR^2 = PS^2) :
  PQ^2 = 120 :=
by
  rw [cond1, cond2] at h
  sorry

end trapezoid_PQRS_PQ_squared_l399_399026


namespace complementary_angles_ratio_decrease_l399_399208

theorem complementary_angles_ratio_decrease 
  (a b : ℝ) (h_ratio : a / b = 3 / 7) (h_comp : a + b = 90) : 
  let a' := a * 1.20 in
  let b' := 90 - a' in
  ((b' / b) * 100 = 91.43) ∧ (100 - (b' / b) * 100 = 8.57) :=
by
  have : a / b = 3 / 7 := h_ratio
  have : a + b = 90 := h_comp
  let a' := a * 1.20
  let b' := 90 - a'
  have : b' / b = 57.6 / 63 := sorry -- Calculations omitted for brevity
  have : (b' / b) * 100 = 91.43 := by sorry
  have : 100 - (b' / b) * 100 = 8.57 := by sorry
  split;
  assumption

end complementary_angles_ratio_decrease_l399_399208
