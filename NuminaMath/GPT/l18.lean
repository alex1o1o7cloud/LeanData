import Mathlib
import Mathlib.Algebra.BigOperators.Factorials
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.BigOperators.Ring
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Polynomial.Division
import Mathlib.Analysis.Asymptotics
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Normed.Field.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.GCD
import Mathlib.Data.Nat.LCM
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Vector
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Topology.Algebra.Order
import Mathlib.Topology.SubsetProperties

namespace find_price_of_fourth_variety_theorem_l18_18447

-- Define the variables and conditions
variables (P1 P2 P3 P4 : ℝ) (Q1 Q2 Q3 Q4 : ℝ) (P_avg : ℝ)

-- Given conditions
def price_of_fourth_variety : Prop :=
  P1 = 126 ∧
  P2 = 135 ∧
  P3 = 156 ∧
  P_avg = 165 ∧
  Q1 / Q2 = 2 / 3 ∧
  Q1 / Q3 = 2 / 4 ∧
  Q1 / Q4 = 2 / 5 ∧
  (P1 * Q1 + P2 * Q2 + P3 * Q3 + P4 * Q4) / (Q1 + Q2 + Q3 + Q4) = P_avg

-- Prove that the price of the fourth variety of tea is Rs. 205.8 per kg
theorem find_price_of_fourth_variety_theorem : price_of_fourth_variety P1 P2 P3 P4 Q1 Q2 Q3 Q4 P_avg → P4 = 205.8 :=
by {
  sorry
}

end find_price_of_fourth_variety_theorem_l18_18447


namespace sum_of_solutions_eq_zero_l18_18078

noncomputable def f (x : ℝ) := 2^(|x|) + 3*|x|

theorem sum_of_solutions_eq_zero :
  (∑ x in {x : ℝ | f x = 32}.toFinset, x) = 0 :=
sorry

end sum_of_solutions_eq_zero_l18_18078


namespace tangent_circle_of_weighted_sum_constant_l18_18325

open Real
open EuclideanSpace

theorem tangent_circle_of_weighted_sum_constant 
  (A B : EuclideanSpace 3) (m n : ℝ)
  (hm : 0 < m) (hn : 0 < n) 
  (L : Set (EuclideanSpace 3)) 
  (H : ∀ (L : Set (EuclideanSpace 3)) (d1 d2 : ℝ), 
    distance_point_to_line A L = d1 ∧ 
    distance_point_to_line B L = d2 → 
    m * d1 + n * d2 = k) : 
  ∃ (C : EuclideanSpace 3) (r : ℝ), 
  r = k / (m + n) ∧ 
  ∀ (L : Set (EuclideanSpace 3)) (d1 d2 : ℝ), 
    distance_point_to_line A L = d1 ∧ 
    distance_point_to_line B L = d2 -> 
      L = tangentToCircle C r :=
 sorry

end tangent_circle_of_weighted_sum_constant_l18_18325


namespace find_inequality_solution_l18_18426

theorem find_inequality_solution :
  {x : ℝ | (x + 1) / (x - 2) + (x + 3) / (2 * x + 1) ≤ 2}
  = {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 1 ∨ 2 ≤ x ∧ x ≤ 9} :=
by
  -- The proof steps are omitted.
  sorry

end find_inequality_solution_l18_18426


namespace tan_sum_identity_l18_18469

theorem tan_sum_identity (theta : Real) (h : Real.tan theta = 1 / 3) :
  Real.tan (theta + Real.pi / 4) = 2 :=
by
  sorry

end tan_sum_identity_l18_18469


namespace sum_first_2010_terms_l18_18929

def sequence (n : ℕ) : ℤ :=
  if n % 4 = 0 then -2
  else if n % 4 = 1 then -1
  else if n % 4 = 2 then 1
  else -2

theorem sum_first_2010_terms :
  let S2010 := (Finset.range 2010).sum (λ n, sequence n)
  S2010 = -2011 :=
by
  sorry

end sum_first_2010_terms_l18_18929


namespace fraction_of_jam_eaten_for_dinner_l18_18932

-- Define the problem
theorem fraction_of_jam_eaten_for_dinner :
  ∃ (J : ℝ) (x : ℝ), 
  J > 0 ∧
  (1 / 3) * J + (x * (2 / 3) * J) + (4 / 7) * J = J ∧
  x = 1 / 7 :=
by
  sorry

end fraction_of_jam_eaten_for_dinner_l18_18932


namespace solve_for_A_l18_18206

def hash (A B : ℝ) : ℝ := A^2 + B^2

theorem solve_for_A (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 :=
by
  sorry

end solve_for_A_l18_18206


namespace ellipse_major_axis_l18_18999

-- Problem Conditions
def ellipse_equation (m : ℝ) : Prop := ∀ x y : ℝ, m * x^2 + y^2 = 1
def eccentricity : ℝ := (real.sqrt 3) / 2

-- To Prove
theorem ellipse_major_axis (m : ℝ) (h : ellipse_equation m) (e : eccentricity = (real.sqrt 3) / 2) :
  major_axis_length = 2 ∨ major_axis_length = 4 := 
sorry

end ellipse_major_axis_l18_18999


namespace simplify_fraction_l18_18980

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) : 
  (a^2 / (a - 2) - (4 * a - 4) / (a - 2)) = a - 2 :=
  sorry

end simplify_fraction_l18_18980


namespace max_value_quadratic_l18_18146

theorem max_value_quadratic :
  ∀ (x : ℝ), x ∈ Icc (-1 : ℝ) 4 → (x^2 - 4 * x + 3) ≤ 8 :=
by
  intros x hx
  sorry

end max_value_quadratic_l18_18146


namespace find_a_set_l18_18150

-- Given the set A and the condition
def setA (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3a + 3}

-- The main statement
theorem find_a_set (a : ℝ) : (1 ∈ setA a) → a = 0 :=
sorry

end find_a_set_l18_18150


namespace one_plus_i_pow_four_eq_neg_four_l18_18757

theorem one_plus_i_pow_four_eq_neg_four : (1 + complex.I)^4 = -4 :=
by
  sorry

end one_plus_i_pow_four_eq_neg_four_l18_18757


namespace total_number_of_vehicles_l18_18710

theorem total_number_of_vehicles 
  (lanes : ℕ) 
  (trucks_per_lane : ℕ) 
  (buses_per_lane : ℕ) 
  (cars_per_lane : ℕ := 2 * lanes * trucks_per_lane) 
  (motorcycles_per_lane : ℕ := 3 * buses_per_lane)
  (total_trucks : ℕ := lanes * trucks_per_lane)
  (total_cars : ℕ := lanes * cars_per_lane)
  (total_buses : ℕ := lanes * buses_per_lane)
  (total_motorcycles : ℕ := lanes * motorcycles_per_lane)
  (total_vehicles : ℕ := total_trucks + total_cars + total_buses + total_motorcycles)
  (hlanes : lanes = 4) 
  (htrucks : trucks_per_lane = 60) 
  (hbuses : buses_per_lane = 40) :
  total_vehicles = 2800 := sorry

end total_number_of_vehicles_l18_18710


namespace number_of_valid_N_l18_18108

theorem number_of_valid_N (N : ℕ) (h₁ : N ≤ 50) (h₂ : ∃ d ∈ ({42, 43, 46, 49, N} : Finset ℕ), 
(mean : (42 + 43 + 46 + 49 + N) / 5 = d ∧ d = median ({one, two, three, four, five}))) :
  ∃ (N_values : Finset ℕ), N_values.card = 3 ∧ ∀ x ∈ N_values, x = 35 ∨ x = 45 ∨ x = 50 :=
begin
  sorry,
end

end number_of_valid_N_l18_18108


namespace part1_part2_part3_l18_18843

-- Part 1
theorem part1 (f : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x, f x = log a (x^2 + 2)) (h₂ : f 5 = 3) : a = 3 :=
sorry

-- Part 2
theorem part2 (f : ℝ → ℝ) (h : ∀ x, f x = log 3 (x^2 + 2)) : f (sqrt 7) = 2 :=
sorry

-- Part 3
theorem part3 (f : ℝ → ℝ) (h : ∀ x, f x = log 3 (x^2 + 2)) : ∀ x, x > -1 → f x < f (x + 2) :=
sorry

end part1_part2_part3_l18_18843


namespace gcd_of_4557_1953_5115_l18_18288

def GCD (a b : ℕ) : ℕ :=
  if b = 0 then a else GCD b (a % b)

theorem gcd_of_4557_1953_5115 : GCD 4557 (GCD 1953 5115) = 93 :=
by
  sorry

end gcd_of_4557_1953_5115_l18_18288


namespace inequality_has_solutions_iff_a_ge_4_l18_18900

theorem inequality_has_solutions_iff_a_ge_4 (a x : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) ↔ a ≥ 4 :=
sorry

end inequality_has_solutions_iff_a_ge_4_l18_18900


namespace scalene_triangle_proof_l18_18942

variable {A B C D E M N : Point}
variable {AB AC BC DE MN A_angle_B_angle_C_angle : ℝ}

-- Scalars representing the lengths of sides AB, AC, and BC, DE, MN
variable [is_scalene A B C]: AB < BC ∧ BC < CA
variable (D_projection : is_projection A B C D)
variable (E_projection : is_projection A C B E)
variable (Intersect_DE_AB: intersects A M B DE)
variable (Intersect_DE_AC: intersects A N C DE)

theorem scalene_triangle_proof :
  AB + AC / BC = DE / MN + 1 :=
sorry

end scalene_triangle_proof_l18_18942


namespace smallest_n_l18_18385

theorem smallest_n (n : ℕ) (m : ℕ) (h1 : 1.04 * m = 100 * n) (h2 : ∃ n_dollars : ℕ, 1.04 * m = 100 * n_dollars) : n = 13 :=
sorry

end smallest_n_l18_18385


namespace service_cleaning_fee_percentage_is_correct_l18_18939

noncomputable def daily_rate : ℝ := 125
noncomputable def pet_fee : ℝ := 100
noncomputable def duration : ℕ := 14
noncomputable def security_deposit_percentage : ℝ := 0.5
noncomputable def security_deposit : ℝ := 1110

noncomputable def total_expected_cost : ℝ := (daily_rate * duration) + pet_fee
noncomputable def entire_bill : ℝ := security_deposit / security_deposit_percentage
noncomputable def service_cleaning_fee : ℝ := entire_bill - total_expected_cost

theorem service_cleaning_fee_percentage_is_correct : 
  (service_cleaning_fee / entire_bill) * 100 = 16.67 :=
by 
  sorry

end service_cleaning_fee_percentage_is_correct_l18_18939


namespace general_term_formula_sum_of_bn_l18_18411

noncomputable def sequence_an (n : ℕ) : ℕ :=
if n = 0 then 0 else 3 * n - 2

noncomputable def sum_Sn (n : ℕ) : ℕ :=
(n * (3 * n - 1)) / 2

noncomputable def sequence_bn (n : ℕ) :=
if n = 0 then 0 else (2 * sum_Sn n) * (2 ^ n) / (3 * n - 1)

noncomputable def sum_Tn (n : ℕ) : ℕ :=
if n = 0 then 0 else nat.rec_on n 0 
  (λ k hk, hk + sequence_bn (k + 1))

theorem general_term_formula (n : ℕ) (h : n > 0) : 
  sequence_an n = 3 * n - 2 := 
sorry

theorem sum_of_bn (n : ℕ) (h : n > 0) :
  sum_Tn n = (n - 1) * 2 ^ (n + 1) + 2 := 
sorry

end general_term_formula_sum_of_bn_l18_18411


namespace length_of_median_AM_l18_18659

-- Define the isosceles triangle and the conditions
variable (A B C M : Type) [metric_space A]
variable (dist : A → A → ℝ)
variable (isosceles_triangle : (dist A B = 10) ∧ (dist A C = 10) ∧ (dist B C = 12))
variable (M : A) (BC_midpoint : (dist B M = 6) ∧ (dist C M = 6))

-- Theorem statement for the proof problem
theorem length_of_median_AM : dist A M = 8 := by
  sorry

end length_of_median_AM_l18_18659


namespace number_of_matching_pages_l18_18690

theorem number_of_matching_pages : 
  ∃ (n : Nat), n = 13 ∧ ∀ x, 1 ≤ x ∧ x ≤ 63 → (x % 10 = (64 - x) % 10) ↔ x % 10 = 2 ∨ x % 10 = 7 :=
by
  sorry

end number_of_matching_pages_l18_18690


namespace mirror_area_correct_l18_18583

noncomputable def width_of_mirror (frame_width : ℕ) (side_width : ℕ) : ℕ :=
  frame_width - 2 * side_width

noncomputable def height_of_mirror (frame_height : ℕ) (side_width : ℕ) : ℕ :=
  frame_height - 2 * side_width

noncomputable def area_of_mirror (frame_width : ℕ) (frame_height : ℕ) (side_width : ℕ) : ℕ :=
  width_of_mirror frame_width side_width * height_of_mirror frame_height side_width

theorem mirror_area_correct :
  area_of_mirror 50 70 7 = 2016 :=
by
  sorry

end mirror_area_correct_l18_18583


namespace product_gcd_lcm_8_12_l18_18334

theorem product_gcd_lcm_8_12 : Nat.gcd 8 12 * Nat.lcm 8 12 = 96 := by
  sorry

end product_gcd_lcm_8_12_l18_18334


namespace shortest_distance_zero_l18_18098

def parabola_point (x : ℝ) : ℝ :=
  x^2 / 4

def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x2 - x1)^2 + (y2 - y1)^2

theorem shortest_distance_zero :
  ∃ P : ℝ × ℝ,
  P = (8, 16) ∧
  ∀ x : ℝ, distance_squared 8 16 x (parabola_point x) = 0  :=
sorry

end shortest_distance_zero_l18_18098


namespace minimize_perimeter_for_right_angle_triangle_with_area_1_l18_18068

-- Define all the necessary components and functions
def right_angle_triangle_area (a b : ℝ) : ℝ := (a * b) / 2

def hypotenuse_length (a b : ℝ) : ℝ := real.sqrt (a^2 + b^2)

def perimeter (a b c : ℝ) : ℝ := a + b + c

theorem minimize_perimeter_for_right_angle_triangle_with_area_1 
  (x : ℝ)
  (hx : x > 0)
  (area_constraint : right_angle_triangle_area x (2 / x) = 1) :
  let a := x
  let b := 2 / x
  let c := hypotenuse_length x (2 / x)
  in 
  (perimeter a b c) ≈ 4.82 ∧ 
  (perimeter a b c) = 5 := 
sorry

end minimize_perimeter_for_right_angle_triangle_with_area_1_l18_18068


namespace regression_equation_correct_l18_18244

noncomputable def exponential_regression_model (x : ℕ) : ℝ := 1.1 * (1.5 ^ x)

def exceeds_threshold_2023 : Prop := exponential_regression_model 6 > 10

def binomial_distribution_X (n : ℕ) (p : ℚ) : ℕ → ℚ
| k := (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

def expected_value_X (n : ℕ) (p : ℚ) : ℚ := n * p

theorem regression_equation_correct :
  (∀ x : ℕ, exponential_regression_model x = 1.1 * (1.5 ^ x)) ∧
  exceeds_threshold_2023 ∧
  (∀ k : ℕ, binomial_distribution_X 3 (2/5) k = (nat.choose 3 k) * ((2/5:ℚ) ^ k) * ((3/5:ℚ) ^ (3 - k))) ∧
  (expected_value_X 3 (2/5) = (6/5:ℚ)) :=
by 
  sorry

end regression_equation_correct_l18_18244


namespace tower_unique_heights_l18_18745

/-- Azmi has four blocks, each in the shape of a rectangular prism and each can have 
 dimensions 2, 3, or 6 as their possible heights. Stacking these four blocks can create
 towers of various heights. Prove that there are 14 unique heights for these towers. -/
theorem tower_unique_heights : 
  let heights := {2, 3, 6}
  in (∃ (block1 block2 block3 block4 : ℕ), block1 ∈ heights ∧ block2 ∈ heights ∧ block3 ∈ heights ∧ block4 ∈ heights ∧ 
       finset.card (finset.image (λ (comb : (ℕ × ℕ × ℕ × ℕ)), comb.1 + comb.2 + comb.3 + comb.4) 
          (finset.product (finset.product (finset.product heights heights) heights) heights)) = 14) :=
sorry

end tower_unique_heights_l18_18745


namespace partnership_profit_l18_18673

noncomputable def totalProfit (P Q R : ℕ) (unit_value_per_share : ℕ) : ℕ :=
  let profit_p := 36 * 2 + 18 * 10
  let profit_q := 24 * 12
  let profit_r := 36 * 12
  (profit_p + profit_q + profit_r) * unit_value_per_share

theorem partnership_profit (P Q R : ℕ) (unit_value_per_share : ℕ) :
  (P / Q = 3 / 2) → (Q / R = 4 / 3) → 
  (unit_value_per_share = 144 / 288) → 
  totalProfit P Q R (unit_value_per_share * 1) = 486 := 
by
  intros h1 h2 h3
  sorry

end partnership_profit_l18_18673


namespace sum_of_adjacent_to_8_l18_18788

noncomputable def grid : Type := Array (Array Nat)

-- Assuming the numbers 1, 2, 3, and 4 are placed in specific positions in the grid
def initialGrid : grid := #[#[1, 2, 3], #[_, _, 4], #[_, _, _]]

-- Define the adjacency relationship
def adjacent (i j : Nat) (r c : Nat) : Prop :=
  (i = r ∧ (j = c + 1 ∨ j + 1 = c)) ∨ (j = c ∧ (i = r + 1 ∨ i + 1 = r))

def sum_adjacent_cells (g : grid) (a b : Nat) : Nat :=
  (Finset.univ.filter (λ ⟨i,j⟩, adjacent i j a b)).sum (λ ⟨i,j⟩, g[i][j])

-- The theorem we need to prove
theorem sum_of_adjacent_to_8 (g : grid) (h_fill : (∀ i j, ((g[i][j] : Option Nat).isSome )) -- assuming all cell are filled correctly
(h_distinct: Function.Injective g.entries) -- assuming uniqueness of all numbers from 1 to 9. 
(h1 : g[0][0] = 1) (h2 : g[0][1] = 2) (h3 : g[0][2] = 3) (h4 : g[1][2] = 4)
(h_sum_9 : ∃ i j, g[i][j] = 9 ∧ sum_adjacent_cells g i j = 15)
(h_loc_8 : ∃ i j, g[i][j] = 8) :     
  sum_adjacent_cells g (classical.some h_loc_8) (classical.some_spec h_loc_8) = 27 :=
sorry

end sum_of_adjacent_to_8_l18_18788


namespace handrail_length_approximation_l18_18725

noncomputable def handrail_length_of_spiral_staircase 
  (radius : ℝ) (height : ℝ) (angle : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let arc_length := (angle / 360) * circumference
  let diagonal := Real.sqrt (height^2 + arc_length^2)
  diagonal

theorem handrail_length_approximation :
  handrail_length_of_spiral_staircase 3 12 180 ≈ 15.3 :=
sorry

end handrail_length_approximation_l18_18725


namespace binary_string_sum_power_of_two_l18_18066

theorem binary_string_sum_power_of_two (b : list ℕ) (h : b.count 1 ≥ 2017) :
  ∃ (s : list ℕ), sum s = 2^n :=
sorry

end binary_string_sum_power_of_two_l18_18066


namespace compute_AO_l18_18192

open Classical

variable (A B C O B' C' T : Type → Prop)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O] [MetricSpace B'] [MetricSpace C'] [MetricSpace T]
variable (AB AC AT : ℝ)
variable (Triangle : ∀ (A B C : Type → Prop), Prop)
variable (CircleThrough : ∀ (O B C : Type → Prop), Type → Prop)
variable (Intersects : ∀ (ω AB AC B' C' T : Type → Prop), Prop)

axiom triangle_ABC : Triangle A B C
axiom circle_with_center_O : CircleThrough O B C ω
axiom intersects_AB_AC : Intersects ω AB AC B' C' T
axiom tangent_externally : ∀ diam_BB' diam_CC', ExternallyTangent diams_BB'_diam_CC' T
axiom lengths : AB = 18 ∧ AC = 36 ∧ AT = 12

theorem compute_AO :
  ∀ (AO : ℝ), AO = 65 / 3 := sorry

end compute_AO_l18_18192


namespace anna_age_at_marriage_l18_18552

noncomputable def age_when_married (josh_age_when_married anna_age_when_married josh_age_today combined_age_today : ℕ) : Prop :=
  josh_age_when_married = 22 ∧
  josh_age_today = josh_age_when_married + 30 ∧
  combined_age_today = 5 * josh_age_when_married ∧
  combined_age_today = josh_age_today + (anna_age_when_married + 30)

theorem anna_age_at_marriage :
  ∃ (anna_age_when_married : ℕ),
    age_when_married 22 anna_age_when_married 52 110 ∧ anna_age_when_married = 28 :=
begin
  use 28,
  dsimp [age_when_married],
  refine ⟨rfl, _, _, _⟩,
  { norm_num },
  { norm_num },
  { norm_num }
end

end anna_age_at_marriage_l18_18552


namespace number_of_students_selected_l18_18646

theorem number_of_students_selected 
  (students_per_school : ℕ)
  (students_not_picked : ℕ)
  (number_of_schools : ℕ)
  (students_per_school = 120)
  (students_not_picked = 10)
  (number_of_schools = 15)
  : (students_per_school - students_not_picked) * number_of_schools = 1650 :=
begin
  sorry
end

end number_of_students_selected_l18_18646


namespace postage_cost_correct_l18_18703

-- Define the base rate and the additional rate
def base_rate := 35   -- in cents
def additional_rate := 20  -- per additional ounce in cents

-- Define the letter's weight
def letter_weight := 5.25  -- in ounces

-- Function to calculate the postage cost
noncomputable def postage_cost (base_rate : ℕ) (additional_rate : ℕ) (weight : ℕ) :=
  let additional_weight := weight - 1
  let additional_cost := additional_rate * Nat.ceil additional_weight
  let total_cost := base_rate + additional_cost
  total_cost

-- Converting from cents to dollars
noncomputable def postage_cost_dollars := postage_cost base_rate additional_rate 525 / 100

-- Prove that the total postage cost is $1.35
theorem postage_cost_correct : postage_cost_dollars = 1.35 := by
  sorry

end postage_cost_correct_l18_18703


namespace sheela_deposit_l18_18257

variable (income deposit : ℝ)
variable (percentage : ℝ := 15 / 100)

theorem sheela_deposit :
  income = 22666.67 →
  deposit = percentage * income →
  deposit = 3400 :=
by
  intros h_income h_deposit
  rw [h_income] at h_deposit
  -- sorry to skip the proof
  exact h_deposit

end sheela_deposit_l18_18257


namespace distance_between_intersections_l18_18287

theorem distance_between_intersections :
  (let intersections := { p : ℝ × ℝ | p.1 ^ 2 + p.2 = 12 ∧ p.1 + p.2 = 12 } in
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersections ∧ p2 ∈ intersections ∧ p1 ≠ p2 ∧
  (dist p1 p2 = Real.sqrt 2)) :=
by { sorry }

end distance_between_intersections_l18_18287


namespace minimize_sum_squares_l18_18227

theorem minimize_sum_squares (x y s : ℝ) (h : x + y + s = 0) : ∃ z : ℝ, z = x^2 + y^2 + s^2 ∧ ∀ a b c : ℝ, a + b + c = 0 → x^2 + y^2 + s^2 ≥ z := by
  use 0
  split
  { sorry }
  { intro a b c habc
    sorry }

end minimize_sum_squares_l18_18227


namespace variance_relationship_l18_18501

theorem variance_relationship 
  (n : ℕ) 
  (x y : fin n → ℝ) 
  (a b : ℝ) 
  (h_relation : ∀ i, y i = a * x i + b)
  (variance_x variance_y : ℝ)
  (h_variance_x : variance_x = ∑ i, ((x i) - (∑ i, x i / n))^2 / n) 
  (h_variance_y : variance_y = ∑ i, ((y i) - (∑ i, y i / n))^2 / n)
  : variance_y = a^2 * variance_x := 
by
  sorry

end variance_relationship_l18_18501


namespace minimal_number_sum_of_special_numbers_l18_18076

def has_only_digits_0_and_9 (x : ℝ) : Prop :=
  ∀ n, ¬(x = (∑ k in finset.range n, if (decidable.to_bool((x * 10^k)%10 = 9)) then 9 / 10^k else 0))

def unique_non_repeating_pattern (x : ℝ) : Prop :=
  ∃ n > 0, ∀ m, x ≠ (∑ k in finset.range m, 9 / 10^(k + m * n))

theorem minimal_number_sum_of_special_numbers :
  ∃ (n : ℕ), (∀ {xs : fin n → ℝ}, (∀ i, has_only_digits_0_and_9 (xs i)) ∧ (∀ i ≠ j, unique_non_repeating_pattern (xs i)) → ∑ i, xs i = 1) ∧ (∀ (m : ℕ), m < 9 → ¬(∀ {xs : fin m → ℝ}, (∀ i, has_only_digits_0_and_9 (xs i)) ∧ (∀ i ≠ j, unique_non_repeating_pattern (xs i)) → ∑ i, xs i = 1)) :=
  sorry

end minimal_number_sum_of_special_numbers_l18_18076


namespace smallest_value_l18_18516

theorem smallest_value (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : 
  ∃ (v : ℝ), v = (x^2 ∨ 2*x ∨ real.sqrt(x^2) ∨ x ∨ 1/x) ∧
  v = 1/x :=
by
  sorry

end smallest_value_l18_18516


namespace circumscribed_locus_l18_18116

theorem circumscribed_locus
  (a b c r : ℝ) (A B O : ℝ × ℝ)
  (hA : A = (a, 0)) (hB : B = (b, c)) (hO : O = (0, 0)) :
  let γ := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2},
      secant := {p : ℝ × ℝ | ∃ λ : ℝ, p.2 = λ * (p.1 - b) + c ∧ p ∈ γ} in
  ∀ M N : ℝ × ℝ,
    M ∈ secant → N ∈ secant → M ≠ N →
    let L := (λ (M N : ℝ × ℝ),
      let P := ((M.1 + N.1) / 2, (M.2 + N.2) / 2) in
      (P.1, P.2)) M N in
    (2 * (a - b) * (L M N).1 - 2 * c * (L M N).2 + r^2 - a^2 = 0) :=
sorry

end circumscribed_locus_l18_18116


namespace remainder_of_product_mod_7_l18_18651

theorem remainder_of_product_mod_7
  (a b c : ℕ)
  (ha : a ≡ 2 [MOD 7])
  (hb : b ≡ 3 [MOD 7])
  (hc : c ≡ 4 [MOD 7]) :
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_mod_7_l18_18651


namespace inequality_solution_l18_18608

theorem inequality_solution (x : ℝ) (h : x ≠ -7) : 
  (x^2 - 49) / (x + 7) < 0 ↔ x ∈ Set.Ioo (-∞) (-7) ∪ Set.Ioo (-7) 7 := by 
  sorry

end inequality_solution_l18_18608


namespace proof_uninsured_employees_l18_18531

-- Define the problem's variables and constraints
variables (T P U : ℕ)
variable (probability_neither : ℚ)
variable (percentage_uninsured_parttime : ℚ)

-- Set the given conditions
def conditions : Prop :=
  T = 350 ∧
  P = 54 ∧
  percentage_uninsured_parttime = 0.125 ∧
  probability_neither = 0.5857142857142857

-- The statement we want to prove
theorem proof_uninsured_employees :
  conditions T P U probability_neither percentage_uninsured_parttime → 
  U = 104 :=
begin
  assume h,
  sorry
end

end proof_uninsured_employees_l18_18531


namespace reflection_matrix_squared_is_identity_l18_18562

noncomputable def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let ⟨a, b⟩ := v
  let norm_sq := a^2 + b^2
  Matrix.of_vec [
    (a^2 - b^2) / norm_sq, 2 * a * b / norm_sq, 
    2 * a * b / norm_sq, (b^2 - a^2) / norm_sq
  ]

theorem reflection_matrix_squared_is_identity (v : ℝ × ℝ) : 
  reflection_matrix v * reflection_matrix v = 1 :=
by 
  let R := reflection_matrix (2, -1)
  have H : R * R = 1 -- This H represents our goal which will be proved
  sorry -- Proof goes here but is omitted

end reflection_matrix_squared_is_identity_l18_18562


namespace correct_statements_l18_18341

theorem correct_statements :
  (∀ α : ℝ, 0 ≤ α ∧ α < real.pi → true) ∧
  (¬ (∀ α : ℝ, (α ∈ set.Icc 0 real.pi ∧ tan α = slope α) → true)) ∧
  (∀ α : ℝ, true → (α = real.pi_div_two → false) → true) ∧
  (¬ (∀ α : ℝ, α ∈ set.Icc 0 real.pi → tan α increases_with α)) → 
  (A ∧ C) :=
begin
  sorry
end

end correct_statements_l18_18341


namespace squareB_perimeter_l18_18264

theorem squareB_perimeter (A_perimeter : ℝ) (B_area : ℝ) (hA : A_perimeter = 32) 
(hB : B_area = (A_perimeter / 4) ^ 2 / 3) : 
4 * real.sqrt (B_area) = 32 * real.sqrt 3 / 3 :=
by
  rw [hA, A_perimeter] at hB
  sorry

end squareB_perimeter_l18_18264


namespace cannot_tile_63_squares_l18_18347

theorem cannot_tile_63_squares (board : fin 8 × fin 8) (a1_removed : board = ⟨0, 0⟩) : 
  ¬ (∃ tiling : set (set (fin 8 × fin 8)), (∀ d ∈ tiling, size d = 2 ∧ ∀ p ∈ d, p ≠ ⟨0, 0⟩) ∧ 
  (⋃ d ∈ tiling, d) = (board \ {⟨0, 0⟩}) ) := 
sorry 

end cannot_tile_63_squares_l18_18347


namespace extinction_prob_one_l18_18722

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the extinction probability function
def extinction_prob (v : ℕ → ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else p * v (k + 1) + q * v (k - 1)

-- State the theorem
theorem extinction_prob_one (v : ℕ → ℝ) :
  extinction_prob v 1 = 2 / 3 :=
sorry

end extinction_prob_one_l18_18722


namespace find_a_l18_18821

theorem find_a (a b c d e : ℝ) 
  (h1 : a * b = 2)
  (h2 : b * c = 3)
  (h3 : c * d = 4)
  (h4 : d * e = 15)
  (h5 : e * a = 10)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (he : 0 < e) : 
  a = 4 / 3 :=
begin
  sorry,
end

end find_a_l18_18821


namespace jim_mpg_l18_18548

theorem jim_mpg (tank_size : ℕ) (fraction_used : ℚ) (total_miles : ℕ) :
  tank_size = 12 → fraction_used = 1 / 3 → total_miles = 20 → total_miles / (tank_size * fraction_used) = 5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have h4 : tank_size * fraction_used = 4 := by rw [h1, h2]; norm_num
  rw h4
  norm_num
  sorry

end jim_mpg_l18_18548


namespace constant_term_expansion_eq_neg4_l18_18621

/-- The constant term in the expansion of (x^3 - 1/x)^4 is -4. -/
theorem constant_term_expansion_eq_neg4 : 
  ∀ (x : ℝ), (∃ r : ℕ, (binom 4 r) * (x ^ (3 * (4 - r))) * ((-1 / x) ^ r) = -4) :=
by
  intro x
  use 3
  simp
  sorry

end constant_term_expansion_eq_neg4_l18_18621


namespace correct_slope_angle_statements_l18_18342

theorem correct_slope_angle_statements :
  (0 ≤ α ∧ α < Real.pi) ∧ (∀ α, 0 ≤ α ∧ α < Real.pi → ∃ unique_slope_angle α) :=
sorry

end correct_slope_angle_statements_l18_18342


namespace toby_total_sales_at_garage_sale_l18_18319

noncomputable def treadmill_price : ℕ := 100
noncomputable def chest_of_drawers_price : ℕ := treadmill_price / 2
noncomputable def television_price : ℕ := treadmill_price * 3
noncomputable def three_items_total : ℕ := treadmill_price + chest_of_drawers_price + television_price
noncomputable def total_sales : ℕ := three_items_total / (3 / 4) -- 75% is 0.75 or 3/4

theorem toby_total_sales_at_garage_sale : total_sales = 600 :=
by
  unfold treadmill_price chest_of_drawers_price television_price three_items_total total_sales
  simp
  exact sorry

end toby_total_sales_at_garage_sale_l18_18319


namespace question_1_question_2_l18_18806

variables {a b : ℝ} (k : ℝ)
variables [NormedField ℝ]

-- Condition: |a| = 1
axiom norm_a_eq_1 : ‖a‖ = 1
-- Condition: |b| = 2
axiom norm_b_eq_2 : ‖b‖ = 2
-- Condition: a parallel to b
axiom a_parallel_b : a ∥ b
-- Condition: a - b is perpendicular to a
axiom a_minus_b_perp_a : ⟪a - b, a⟫ = 0

-- Question 1: Prove a ⋅ b = ±2
theorem question_1 : a ⋅ b = 2 ∨ a ⋅ b = -2 := by
  sorry

-- Question 2: Prove k = 3 such that (ka - b) ⟂ (a + 2b)
theorem question_2 : (k * a - b) ⊥ (a + 2 * b) → k = 3 := by
  sorry

end question_1_question_2_l18_18806


namespace a_4_val_a_general_formula_l18_18816

noncomputable def a : ℕ → ℕ
| 0     := 2
| (n+1) := 2 * a n + 1

theorem a_4_val : a 3 = 23 :=
by sorry

theorem a_general_formula (n : ℕ) : a n = 3 * 2 ^ (n - 1) - 1 :=
by sorry

end a_4_val_a_general_formula_l18_18816


namespace find_x_eq_l18_18091

-- Given conditions
variables (c b θ : ℝ)

-- The proof problem
theorem find_x_eq :
  ∃ x : ℝ, x^2 + c^2 * (Real.sin θ)^2 = (b - x)^2 ∧
          x = (b^2 - c^2 * (Real.sin θ)^2) / (2 * b) :=
by
    sorry

end find_x_eq_l18_18091


namespace judy_pencil_cost_l18_18937

theorem judy_pencil_cost :
  (∀ (pencils_per_week days_per_week pencils_per_pack cost_per_pack total_days), 
    (pencils_per_week = 10 → days_per_week = 5 → pencils_per_pack = 30 → cost_per_pack = 4 → total_days = 45 → 
    let pencils_per_day := pencils_per_week / days_per_week in
    let total_pencils_needed := pencils_per_day * total_days in
    let packs_needed := total_pencils_needed / pencils_per_pack in
    let total_cost := packs_needed * cost_per_pack in
    total_cost = 12)) :=
by 
  intros pencils_per_week days_per_week pencils_per_pack cost_per_pack total_days hw hd hp hc ht
  calc 
    let pencils_per_day := pencils_per_week / days_per_week in
    let total_pencils_needed := pencils_per_day * total_days in
    let packs_needed := total_pencils_needed / pencils_per_pack in
    let total_cost := packs_needed * cost_per_pack in
    total_cost = 12 : sorry

end judy_pencil_cost_l18_18937


namespace tan_alpha_plus_pi_over_4_l18_18126

noncomputable def tan_sum_formula (α : ℝ) : ℝ :=
  (Real.tan α + Real.tan (Real.pi / 4)) / (1 - Real.tan α * Real.tan (Real.pi / 4))

theorem tan_alpha_plus_pi_over_4 
  (α : ℝ) 
  (h1 : Real.cos (2 * α) + Real.sin α * (2 * Real.sin α - 1) = 2 / 5) 
  (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : 
  tan_sum_formula α = 1 / 7 := 
sorry

end tan_alpha_plus_pi_over_4_l18_18126


namespace train_length_l18_18730

theorem train_length (speed_kmh : ℕ) (time_seconds : ℕ) (bridge_length_m : ℕ) : 
  speed_kmh = 45 → 
  time_seconds = 30 → 
  bridge_length_m = 220 → 
  ∃ train_length_m : ℕ, train_length_m = 155 :=
begin
  intros,
  sorry
end

end train_length_l18_18730


namespace simplify_complex_division_l18_18259

noncomputable def complex_division_simplification : ℂ :=
  (7 + 8 * complex.I) / (3 - 4 * complex.I)

theorem simplify_complex_division :
  complex_division_simplification = -11/25 + 52/25 * complex.I := 
  by sorry

end simplify_complex_division_l18_18259


namespace inequality_solution_l18_18609

variable {x : ℝ}

theorem inequality_solution :
  x ∈ Set.Ioo (-∞ : ℝ) 7 ∪ Set.Ioo (-∞ : ℝ) (-7) ∪ Set.Ioo (-7) 7 ↔ (x^2 - 49) / (x + 7) < 0 :=
by
  sorry

end inequality_solution_l18_18609


namespace probability_sum_less_than_product_l18_18324

theorem probability_sum_less_than_product :
  (∑ (a b : ℕ) in {x | x ∈ {1, 2, 3, 4, 5, 6} ∧ (x % 2 = 0)}, 
  {y | y ∈ {1, 2, 3, 4, 5, 6} ∧ (y % 2 ≠ 0)},
  (a + b < a * b).indicator 1).to_nat / 18 = 2 / 3 :=
by
  sorry

end probability_sum_less_than_product_l18_18324


namespace number_of_purple_balls_l18_18689

theorem number_of_purple_balls (k : ℕ) (h : k > 0) (E : (24 - k) / (8 + k) = 1) : k = 8 :=
by {
  sorry
}

end number_of_purple_balls_l18_18689


namespace prime_gt_5_divides_fn_l18_18570

def g : ℕ → ℕ 
| 1 := 0
| 2 := 1
| (n+2) := g(n) + g(n+1)

theorem prime_gt_5_divides_fn (n : ℕ) (h1 : Nat.prime n) (h2 : 5 < n) : n ∣ (g n * (g n + 1)) := 
by
  sorry

end prime_gt_5_divides_fn_l18_18570


namespace system_of_inequalities_solution_l18_18154

theorem system_of_inequalities_solution (a b : ℝ) :
  (∀ x : ℝ, 2x - a < 1 ∧ x - 2b > 3 → -1 < x ∧ x < 1) →
  (a + 1) * (b - 1) = -6 := by
  sorry

end system_of_inequalities_solution_l18_18154


namespace new_number_properties_l18_18771

def new_number_gen (a b : ℕ) : ℕ := a * b + a + b

def is_new_number (c : ℕ) : Prop :=
  ∃ (a b : ℕ), (a = 1 ∨ a = 4 ∨ a = 9 ∨ a = 49) ∧
                  (b = 1 ∨ b = 4 ∨ b = 9 ∨ b = 49) ∧
                  c = new_number_gen a b

theorem new_number_properties :
  ¬ is_new_number 2008 ∧
  (∀ c, is_new_number c → (c + 1) % 2 = 0) ∧
  (∀ c, is_new_number c → (c + 1) % 10 = 0) :=
begin
  sorry
end

end new_number_properties_l18_18771


namespace part_a_win_part_b_no_win_l18_18225

-- Define the necessary parameters and conditions
variables (k n : ℕ) (N : ℕ) (x : ℕ)
variables (S : set ℕ)

-- State the first part of the theorem
theorem part_a_win (hk : k > 0) (hn : n ≥ 2^k) (hN : N > 0) (hx : 1 ≤ x ∧ x ≤ N) :
  ∃ (T : set ℕ), (T.card ≤ n) ∧ (x ∈ T) :=
sorry

-- State the second part of the theorem
theorem part_b_no_win (hk_large : k > 0) (n' : ℕ) (hn_large : n' ≥ 1.99^k) :
  ∃ N' x', (1 ≤ x' ∧ x' ≤ N') ∧ (¬ ∃ (T' : set ℕ), (T'.card ≤ n') ∧ (x' ∈ T')) :=
sorry

end part_a_win_part_b_no_win_l18_18225


namespace frustum_volume_ratio_l18_18299

noncomputable def ratio_of_volumes (area_upper : ℝ) (area_lower : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * height * (area_upper + area_lower + real.sqrt (area_upper * area_lower))

theorem frustum_volume_ratio :
  ∀ (area_upper area_lower : ℝ) (height : ℝ),
    (area_upper / area_lower = 1 / 9) →
    (ratio_of_volumes area_upper area_lower height / ratio_of_volumes area_lower (4 * area_upper) (height / 2) = 7 / 19) :=
by
  intros area_upper area_lower height h
  sorry

end frustum_volume_ratio_l18_18299


namespace unit_digit_of_25_17_18_factorials_l18_18407

theorem unit_digit_of_25_17_18_factorials : (25! + 17! - 18!) % 10 = 0 := by
  sorry

end unit_digit_of_25_17_18_factorials_l18_18407


namespace trigonometric_identity_l18_18450

theorem trigonometric_identity (α : Real) (h : Real.sin (Real.pi + α) = -1/3) : 
  (Real.sin (2 * α) / Real.cos α) = 2/3 :=
by
  sorry

end trigonometric_identity_l18_18450


namespace alex_loan_comparison_l18_18392

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r * t)

theorem alex_loan_comparison :
  let P : ℝ := 15000
  let r1 : ℝ := 0.08
  let r2 : ℝ := 0.10
  let n : ℕ := 12
  let t1_10 : ℝ := 10
  let t1_5 : ℝ := 5
  let t2 : ℝ := 15
  let owed_after_10 := compound_interest P r1 n t1_10
  let payment_after_10 := owed_after_10 / 2
  let remaining_after_10 := owed_after_10 / 2
  let owed_after_15 := compound_interest remaining_after_10 r1 n t1_5
  let total_payment_option1 := payment_after_10 + owed_after_15
  let total_payment_option2 := simple_interest P r2 t2
  total_payment_option1 - total_payment_option2 = 4163 :=
by
  sorry

end alex_loan_comparison_l18_18392


namespace D_n_form_of_power_of_3_D_n_exact_power_of_3_l18_18109

theorem D_n_form_of_power_of_3 :
  ∀ n : ℕ, ∃ k : ℕ, ∀ (a : ℕ), gcd (a^n + (a + 1)^n + (a + 2)^n) (a^n + (a + 1)^n + (a + 2)^n + 3^n) = 3^k := sorry

theorem D_n_exact_power_of_3 :
  ∀ k : ℕ, ∃ n : ℕ, gcd (λ a : ℕ, a^n + (a + 1)^n + (a + 2)^n) (λ a : ℕ, a^n + (a + 1)^n + (a + 2)^n + 3^n) = 3^k := sorry

end D_n_form_of_power_of_3_D_n_exact_power_of_3_l18_18109


namespace molecular_weight_of_acetone_l18_18333

def molecular_weight_of_n_moles (n : ℕ) (weight : ℕ) : ℕ :=
  weight / n

variable (n : ℕ) (weight : ℕ)
variable (hmoles : n = 9) (htotal_weight : weight = 522)

theorem molecular_weight_of_acetone : molecular_weight_of_n_moles n weight = 58 :=
by
  have h1 : n = 9 := hmoles
  have h2 : weight = 522 := htotal_weight
  rw [h1, h2]
  simp [molecular_weight_of_n_moles]
  norm_num
  exact Nat.div_self dec_trivial

end molecular_weight_of_acetone_l18_18333


namespace ascetic_height_l18_18321

theorem ascetic_height (h m : ℝ) (x : ℝ) (hx : h * (m + 1) = (x + h)^2 + (m * h)^2) : x = h * m / (m + 2) :=
sorry

end ascetic_height_l18_18321


namespace meaningful_expressions_l18_18736

theorem meaningful_expressions (n : ℕ) (x : ℝ) :
  (Real.sqrt 4 ((-4)^(2*n))).Real.nonneg ∧
  (Real.sqrt 4 ((-4)^(2*n + 1)) = ∅) ∧
  (Real.sqrt 5 (x^2)).Real.nonneg ∧
  (Real.sqrt 5 (-x^2)).Real.nonneg := by
  sorry

end meaningful_expressions_l18_18736


namespace modular_inverse_of_17_mod_800_l18_18330

    theorem modular_inverse_of_17_mod_800 :
      ∃ x : ℤ, 0 ≤ x ∧ x < 800 ∧ (17 * x) % 800 = 1 :=
    by
      use 47
      sorry
    
end modular_inverse_of_17_mod_800_l18_18330


namespace distance_PQ_l18_18495

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem distance_PQ : 
  let P := (-1 : ℝ, 2 : ℝ)
  let Q := (3 : ℝ, 0 : ℝ)
  distance P Q = 2 * Real.sqrt 5 := 
by
  sorry

end distance_PQ_l18_18495


namespace largest_n_satisfying_conditions_l18_18430

def U := Finset.range 2009 -- {1, 2, ..., 2008}
def S (i : Nat) := U \ A i

def condition1 (S : Nat -> Finset Nat) (i j : Nat) :=
  (1 ≤ i) → (1 ≤ j) → (i ≤ n) → (j ≤ n) → (i ≠ j) → (S i ∪ S j).card ≤ 2004

def condition2 (S : Nat -> Finset Nat) (i j k : Nat) :=
  (1 ≤ i) → (1 ≤ j) → (1 ≤ k) → (i < j) → (j < k) → (S i ∪ S j ∪ S k) = U

theorem largest_n_satisfying_conditions : 
  ∃ (n : Nat) (S : Nat -> Finset Nat), (∀ i j, condition1 S i j) ∧ (∀ i j k, condition2 S i j k) ∧ n = 32 := 
sorry

end largest_n_satisfying_conditions_l18_18430


namespace choose_books_different_genres_l18_18883

/-- 
Given 4 distinct mystery novels, 3 distinct fantasy novels, and 2 distinct biographies, 
the number of ways to choose 2 books of different genres is 26.
-/
theorem choose_books_different_genres :
  let num_mystery := 4
  let num_fantasy := 3
  let num_biography := 2
  (num_mystery * num_fantasy) + (num_mystery * num_biography) + (num_fantasy * num_biography) = 26 := 
by
  let num_mystery := 4
  let num_fantasy := 3
  let num_biography := 2
  have h_mf := num_mystery * num_fantasy   -- number of pairs (mystery, fantasy)
  have h_mb := num_mystery * num_biography -- number of pairs (mystery, biography)
  have h_fb := num_fantasy * num_biography -- number of pairs (fantasy, biography)
  calc
    h_mf + h_mb + h_fb = (4 * 3) + (4 * 2) + (3 * 2) := by sorry
    ... = 12 + 8 + 6 := by sorry
    ... = 26 := by sorry

end choose_books_different_genres_l18_18883


namespace nine_appears_25_times_l18_18723

-- Define the range of house numbers and the function to count digit '9'
def count9_in_range (start : ℕ) (end : ℕ) : ℕ :=
  ((list.range' start (end - start + 1)).map (λ n, (n.digits 10).count (λ d, d = 9))).sum

-- Proof statement
theorem nine_appears_25_times : count9_in_range 1 150 = 25 := 
by 
    sorry

end nine_appears_25_times_l18_18723


namespace find_a_set_l18_18151

-- Given the set A and the condition
def setA (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3a + 3}

-- The main statement
theorem find_a_set (a : ℝ) : (1 ∈ setA a) → a = 0 :=
sorry

end find_a_set_l18_18151


namespace min_value_of_f_on_pos_reals_range_of_a_for_inequality_l18_18452

def f (x : ℝ) : ℝ := x * Real.log x
def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 3

noncomputable def f_min_value := -1 / Real.exp 1

theorem min_value_of_f_on_pos_reals : 
  ∀ x ∈ Set.Ioi 0, f x ≥ f_min_value := 
begin
  intros x hx,
  sorry
end

noncomputable def h (x : ℝ) : ℝ := x + 2 * Real.log x + 3 / x
noncomputable def h_min_value := 4

theorem range_of_a_for_inequality :
  ∀ x ∈ Set.Ioi 0, ∀ a, 2 * f x ≥ g x a ↔ a ≤ h_min_value := 
begin
  intros x hx a,
  sorry
end

end min_value_of_f_on_pos_reals_range_of_a_for_inequality_l18_18452


namespace ratio_EG_ES_l18_18535

theorem ratio_EG_ES (EF EH EQ ER x : ℝ) (Q R S E G F H : Type) [parallelogram E F G H]
  (hQ : EQ = 19/1000 * EF)
  (hR : ER = 19/2051 * EH)
  (hS : intersects EG QR = S) :
  EG/ES = 3051/19 :=
sorry

end ratio_EG_ES_l18_18535


namespace simplify_expression_l18_18578

variable {a b c : ℝ}
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

def x := (b / c) - (c / b)
def y := (a / c) - (c / a)
def z := (a / b) - (b / a)

theorem simplify_expression : x^2 + y^2 + z^2 + x * y * z = -4 := by
  sorry

end simplify_expression_l18_18578


namespace total_letters_sent_l18_18256

def letters_January : ℕ := 6
def letters_February : ℕ := 9
def letters_March : ℕ := 3 * letters_January

theorem total_letters_sent : letters_January + letters_February + letters_March = 33 :=
by
  -- This is where the proof would go
  sorry

end total_letters_sent_l18_18256


namespace ratio_tuesday_monday_l18_18587

-- Define the conditions
variables (M T W : ℕ) (hM : M = 450) (hW : W = 300) (h_rel : W = T + 75)

-- Define the theorem
theorem ratio_tuesday_monday : (T : ℚ) / M = 1 / 2 :=
by
  -- Sorry means the proof has been omitted in Lean.
  sorry

end ratio_tuesday_monday_l18_18587


namespace bills_equal_at_80_minutes_l18_18656

variable (m : ℝ)

def C_U : ℝ := 8 + 0.25 * m
def C_A : ℝ := 12 + 0.20 * m

theorem bills_equal_at_80_minutes (h : C_U m = C_A m) : m = 80 :=
by {
  sorry
}

end bills_equal_at_80_minutes_l18_18656


namespace trigonometric_identity_l18_18837

-- Define the problem conditions
variable (α : ℝ)
hypothesis (h1 : π / 2 < α ∧ α < π)  -- α is in the second quadrant
hypothesis (h2 : Real.sin α = (Real.sqrt 15) / 4)

-- The statement to prove
theorem trigonometric_identity :
  (Real.sin (α + π / 4)) / (Real.sin (2 * α) + Real.cos (2 * α) + 1) = - Real.sqrt 2 :=
by
  -- The proof goes here
  sorry

end trigonometric_identity_l18_18837


namespace shortest_paths_l18_18641

theorem shortest_paths (h k : ℕ) : 
  (finset.card (finset.filter (λ path : list (ℕ × ℕ),
    ∃ (n_move e_move : ℕ), n_move = h ∧ e_move = k ∧
    path.count (0, 1) = n_move ∧ path.count (1, 0) = e_move) 
    (list.permutations (list.replicate h (0, 1) ++ list.replicate k (1, 0))))) = 
  nat.choose (h + k) h := sorry

end shortest_paths_l18_18641


namespace sum_of_real_values_l18_18643

theorem sum_of_real_values (x : ℝ) (h : |3 * x + 1| = 3 * |x - 3|) : x = 4 / 3 := sorry

end sum_of_real_values_l18_18643


namespace transform_equation_l18_18316

theorem transform_equation (x : ℝ) : x^2 - 2 * x - 2 = 0 ↔ (x - 1)^2 = 3 :=
sorry

end transform_equation_l18_18316


namespace total_pay_l18_18025

/-
  Given:
  (1) Regular pay is $3 per hour up to 40 hours.
  (2) Overtime is twice the payment for regular time.
  (3) He worked 12 hours overtime.
  Prove:
  The total pay is $192.
-/
theorem total_pay (regular_rate : ℕ) (regular_hours : ℕ) (overtime_hours : ℕ) (overtime_rate : ℕ) (total_pay : ℕ)  :
  regular_rate = 3 →
  regular_hours = 40 →
  overtime_hours = 12 →
  overtime_rate = 2 * regular_rate →
  total_pay = regular_rate * regular_hours + overtime_rate * overtime_hours →
  total_pay = 192 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  sorry

end total_pay_l18_18025


namespace sum_of_coefficients_expansion_l18_18786

theorem sum_of_coefficients_expansion (d : ℝ) :
  let expanded_form := -(5 - d) * (d + 2 * (5 - d))
  in (expanded_form.coeff 2 + expanded_form.coeff 1 + expanded_form.coeff 0) = -36 :=
by
  let expanded_form := -(5 - d) * (d + 2 * (5 - d))
  have h : expanded_form = -d^2 + 15*d - 50 := sorry
  rw h
  exact (-1) + 15 + (-50) = -36

end sum_of_coefficients_expansion_l18_18786


namespace sum_eq_l18_18353

noncomputable def non_neg_random_variables (n : ℕ) := ℕ → nnreal

variables {ξ : ℕ → ℕ → nnreal}
variables (Hp : ∀ n, ∃ p > 1, E[ξ n ^ p] < ∞)
variables (Cp : ∃ Cm, ∀ n, ∃ p > 1, Cm^p = sup_n (E[ξ(n)^p]/(E ξ(n))^p))

theorem sum_eq (ξ : ℕ → ℕ → nnreal)
  (Hp : ∃ p > 1, ∀ n, E[ξ n ^ p] < ∞)
  (Cp : ∃ Cm, Cm^p = ∀ n, sup (E[ξ n ^ p]/(E ξ n) ^ p) < ∞) :
  tsum (λ n, ξ n) < ∞ ↔ tsum (λ n, E ξ n) < ∞ ↔ tsum (λ n, ∥ξ n∥ p) < ∞ :=
sorry

end sum_eq_l18_18353


namespace tangent_chord_length_l18_18270

noncomputable def length_of_tangent_chord (r1 r2 : ℝ) (ar : ℝ) (h : 2 * r2 = r1) (ha : π * (r1^2 - r2^2) = ar) : ℝ :=
  if H : 2 * r2 = r1 then 
    if HA : π * (r1^2 - r2^2) = ar then 
      let c := 2 * Real.sqrt (r1^2 - r2^2)
      in c
    else sorry
  else sorry

theorem tangent_chord_length
  (r1 r2 : ℝ)
  (ar : ℕ) 
  (h : 2 * r2 = r1)
  (ha : π * (r1^2 - r2^2) = ar) :
  length_of_tangent_chord r1 r2 ar h ha = Real.sqrt 75 :=
sorry

end tangent_chord_length_l18_18270


namespace Fangfang_time_l18_18787

-- Define the time taken for specific floors
def time_between_floors (start : ℕ) (end : ℕ) : ℕ := 30

-- Define the floors we are concerned with
def start_floor_a := 1
def end_floor_a := 3

def start_floor_b := 2
def end_floor_b := 6

-- Calculate the number of floors between given start and end
def number_of_floors (start : ℕ) (end : ℕ) : ℕ := end - start

-- Calculate the time per floor
def time_per_floor : ℕ := time_between_floors start_floor_a end_floor_a / number_of_floors start_floor_a end_floor_a

-- Calculate the total time for the second segment
def time_second_segment : ℕ := number_of_floors start_floor_b end_floor_b * time_per_floor

-- Theorem stating the time to go from the 2nd floor to the 6th floor is 60 seconds.
theorem Fangfang_time :
  time_second_segment = 60 := by
  sorry

end Fangfang_time_l18_18787


namespace log_a_b_equality_l18_18559

theorem log_a_b_equality (a b : ℝ) (M N : Set ℝ) (hM : M = {2, 4}) (hN : N = {a, b}) (hMN : M = N) :
  Real.log a b = 2 ∨ Real.log a b = 1 / 2 :=
by
  sorry

end log_a_b_equality_l18_18559


namespace arithmetic_series_sum_l18_18443

theorem arithmetic_series_sum :
  let a := -45
  let l := 1
  let d := 2
  let n := (l - a) / d + 1
  (n = 24) →
  (∀ i, a + (i - 1) * d = a + (i - 1) * d) →
  S := n / 2 * (a + l)
  S = -528
:=
by
  intro a l d n
  dsimp [a, l, d, n]
  sorry

end arithmetic_series_sum_l18_18443


namespace find_a_l18_18480

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then -x else x^2

theorem find_a (a : ℝ) (h : f a = 9) : a = -9 ∨ a = 3 :=
by
  sorry

end find_a_l18_18480


namespace min_value_of_x4_y3_z2_l18_18221

noncomputable def min_value_x4_y3_z2 (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem min_value_of_x4_y3_z2 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : 1/x + 1/y + 1/z = 9) : 
  min_value_x4_y3_z2 x y z = 1 / 3456 :=
by
  sorry

end min_value_of_x4_y3_z2_l18_18221


namespace area_of_triangle_ABC_l18_18969

open Real

-- Defining the conditions as per the problem
def triangle_side_equality (AB AC : ℝ) : Prop := AB = AC
def angle_relation (angleBAC angleBTC : ℝ) : Prop := angleBAC = 2 * angleBTC
def side_length_BT (BT : ℝ) : Prop := BT = 70
def side_length_AT (AT : ℝ) : Prop := AT = 37

-- Proving the area of triangle ABC given the conditions
theorem area_of_triangle_ABC
  (AB AC : ℝ)
  (angleBAC angleBTC : ℝ)
  (BT AT : ℝ)
  (h1 : triangle_side_equality AB AC)
  (h2 : angle_relation angleBAC angleBTC)
  (h3 : side_length_BT BT)
  (h4 : side_length_AT AT) 
  : ∃ area : ℝ, area = 420 :=
sorry

end area_of_triangle_ABC_l18_18969


namespace part1_part2_part3_l18_18114

variable {x y : ℚ}

def star (x y : ℚ) : ℚ := x * y + 1

theorem part1 : star 2 4 = 9 := by
  sorry

theorem part2 : star (star 1 4) (-2) = -9 := by
  sorry

theorem part3 (a b c : ℚ) : star a (b + c) + 1 = star a b + star a c := by
  sorry

end part1_part2_part3_l18_18114


namespace largest_number_among_given_l18_18777

theorem largest_number_among_given (
  A B C D E : ℝ
) (hA : A = 0.936)
  (hB : B = 0.9358)
  (hC : C = 0.9361)
  (hD : D = 0.935)
  (hE : E = 0.921):
  C = max A (max B (max C (max D E))) :=
by
  sorry

end largest_number_among_given_l18_18777


namespace unit_digit_14_pow_100_eq_6_l18_18337

-- Define the problem with conditions and the answer.
theorem unit_digit_14_pow_100_eq_6 :
    let unit_digit_of (n : ℕ) : ℕ := n % 10
    unit_digit_of (14 ^ 100) = 6 :=
by
  -- Unit digit of 14 is same as unit digit of 4
  let unit_digit_of := λ n : ℕ, n % 10
  have h1 : unit_digit_of 14 = 4 := by rfl
  -- Unit digit of 4^n cycles between 4 and 6 for odd and even n respectively
  have h2 : unit_digit_of 4 = 4 := by rfl
  have h3 : unit_digit_of (4 ^ 2) = 6 := by rfl
  have h4 : ∀ k : ℕ, k % 2 = 0 → unit_digit_of (4 ^ k) = 6 := by intros k hk; sorry
  -- Since 100 is even, unit digit of 4^100 is 6
  show unit_digit_of (14 ^ 100) = 6 from h4 100 (by rfl)
  sorry

end unit_digit_14_pow_100_eq_6_l18_18337


namespace convex_quadrilateral_angles_l18_18910

variable (A B C D : ℝ)
variable (a b c d : ℝ)

theorem convex_quadrilateral_angles
  (h1 : ∃ (A B C D : ℝ), A + B + C + D = π)
  (h2 : tan_angle_convex_quadrilateral A B C D)
  : (sin A)/(b * c) + (sin C)/(d * a) = (sin B)/(c * d) + (sin D)/(a * b) := 
by
  sorry

end convex_quadrilateral_angles_l18_18910


namespace f_2_plus_f_5_eq_2_l18_18112

noncomputable def f : ℝ → ℝ := sorry

open Real

-- Conditions: f(3^x) = x * log 9
axiom f_cond (x : ℝ) : f (3^x) = x * log 9

-- Question: f(2) + f(5) = 2
theorem f_2_plus_f_5_eq_2 : f 2 + f 5 = 2 := sorry

end f_2_plus_f_5_eq_2_l18_18112


namespace handrail_length_approximation_l18_18724

noncomputable def handrail_length_of_spiral_staircase 
  (radius : ℝ) (height : ℝ) (angle : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let arc_length := (angle / 360) * circumference
  let diagonal := Real.sqrt (height^2 + arc_length^2)
  diagonal

theorem handrail_length_approximation :
  handrail_length_of_spiral_staircase 3 12 180 ≈ 15.3 :=
sorry

end handrail_length_approximation_l18_18724


namespace sum_b_infinite_l18_18032

noncomputable def b : ℕ → ℝ
| 1 => 2
| 2 => 2
| (n + 3) => (1 / 2) * b (n + 2) + (1 / 3) * b (n + 1)

def sum_b (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem sum_b_infinite : (∑' n, b (n + 1)) = 18 := 
sorry

end sum_b_infinite_l18_18032


namespace maximal_area_equilateral_triangle_l18_18819

/-- Given an acute-angled triangle ABC, prove that there exists an equilateral triangle XYZ
with one side passing through each of the points A, B, and C such that the area of XYZ is maximal. -/
theorem maximal_area_equilateral_triangle
  (A B C : Point)
  (hABC : acute_triangle A B C)
  : ∃ (X Y Z : Point), 
    equilateral_triangle X Y Z ∧ 
    (side_passes_through A B C X Y Z) ∧
    (∀ (X' Y' Z' : Point), equilateral_triangle X' Y' Z' ∧ side_passes_through A B C X' Y' Z' → 
      area X Y Z ≥ area X' Y' Z') := sorry

end maximal_area_equilateral_triangle_l18_18819


namespace sum_of_zeros_of_g_l18_18770

noncomputable def f : ℝ → ℝ
| x := if 1 ≤ x ∧ x ≤ 2 then 4 - 8 * |x - 3 / 2| else (1 / 2) * f (x / 2)

def g (x : ℝ) : ℝ := x * f x - 6

theorem sum_of_zeros_of_g (n : ℕ) :
  ∑ x in (set_of (λ x, g x = 0)).filter (λ x, 1 ≤ x ∧ x ≤ 2^n), x = (3 / 2) * (2^n - 1) :=
sorry

end sum_of_zeros_of_g_l18_18770


namespace num_ways_to_select_2_from_9_l18_18920

theorem num_ways_to_select_2_from_9 : (nat.choose 9 2) = 36 := by
  sorry

end num_ways_to_select_2_from_9_l18_18920


namespace solve_inequality_l18_18613

noncomputable def polynomial := Polynomial.C 1 * (Polynomial.X - 3) * (Polynomial.X - (2 + Real.sqrt 3)) * (Polynomial.X - (2 - Real.sqrt 3))

theorem solve_inequality : ∃ x : ℝ, (x = 2 + Real.sqrt 3) ∧
  (Real.sqrt (x^3 + 2*x - 58) + 5) * (abs (x^3 - 7*x^2 + 13*x - 3)) ≤ 0 ∧
  (∀ y : ℝ, (Real.sqrt (y^3 + 2*y - 58) + 5) * (abs (y^3 - 7*y^2 + 13*y - 3)) ≤ 0 → y = 2 + Real.sqrt 3) :=
begin
  sorry
end

end solve_inequality_l18_18613


namespace ellipse_x_intercept_l18_18396

theorem ellipse_x_intercept
    (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) (F1_x : F1 = (1, 3)) (F2_x : F2 = (4, 1))
    (intersect1 : ℝ × ℝ) (intersect1_x : intersect1 = (0, 0))
    : ∃ (x : ℝ), x = 1 + sqrt 40 ∧ (x, 0) ≠ intersect1 :=
begin
    sorry
end

end ellipse_x_intercept_l18_18396


namespace find_m_parity_f_monotone_f_l18_18847

def f (x : ℝ) (m : ℝ) : ℝ := x + m / x

-- Part 1: Prove m = 4
theorem find_m (m : ℝ) (h : f 1 m = 5) : m = 4 := 
sorry

-- Part 2: Prove f(x) is odd
theorem parity_f (m : ℝ) : (∀ x : ℝ, f (-x) m = -f x m) :=
sorry

-- Conditions for Part 3
variables (m : ℝ)
variables (h_m : m = 4)

def f' (x : ℝ) : ℝ := 1 - m / x^2

-- Part 3: Prove monotonicity on [2, +∞)
theorem monotone_f : ∀ x ≥ 2, f' x ≥ 0 :=
sorry

end find_m_parity_f_monotone_f_l18_18847


namespace shopkeeper_profit_percent_l18_18382

noncomputable def profit_percent : ℚ := 
let cp_each := 1       -- Cost price of each article
let sp_each := 1.2     -- Selling price of each article without discount
let discount := 0.05   -- 5% discount
let tax := 0.10        -- 10% sales tax
let articles := 30     -- Number of articles
let cp_total := articles * cp_each      -- Total cost price
let sp_after_discount := sp_each * (1 - discount)    -- Selling price after discount
let revenue_before_tax := articles * sp_after_discount   -- Total revenue before tax
let tax_amount := revenue_before_tax * tax   -- Sales tax amount
let revenue_after_tax := revenue_before_tax + tax_amount -- Total revenue after tax
let profit := revenue_after_tax - cp_total -- Profit
(profit / cp_total) * 100 -- Profit percent

theorem shopkeeper_profit_percent : profit_percent = 25.4 :=
by
  -- Here follows the proof based on the conditions and steps above
  sorry

end shopkeeper_profit_percent_l18_18382


namespace cost_price_decrease_proof_l18_18898

theorem cost_price_decrease_proof (x y : ℝ) (a : ℝ) (h1 : y - x = x * a / 100)
    (h2 : y = (1 + a / 100) * x)
    (h3 : y - 0.9 * x = (0.9 * x * a / 100) + 0.9 * x * 20 / 100) : a = 80 :=
  sorry

end cost_price_decrease_proof_l18_18898


namespace smallest_difference_between_reversed_ages_l18_18269

theorem smallest_difference_between_reversed_ages :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a ≠ b →
  ∃ (d : ℕ), d = 9 ∧ abs ((10 * a + b) - (10 * b + a)) = d :=
by {
  sorry
}

end smallest_difference_between_reversed_ages_l18_18269


namespace adjacent_pairs_odd_l18_18354

theorem adjacent_pairs_odd (p q r : ℕ) (h_p : 1 < p) (h_q : 1 < q) (h_r : 1 < r) (h_even : p % 2 = 0) :
  ∃ (balls : Fin (p*q) → Fin (p*r)), 
  (∀ i, |balls i - balls (i + 1) % (p*q)| = 1 ∨ 
       |balls i - balls (i + 1) % (p*q)| = p*r - 1) ∧ 
  (∀ i, balls (i + q) % (p*q) = balls i + r ∨ 
       balls (i + q) % (p*q) = balls i - (p-1)*r) ∧ 
  (count (λ i, (balls i = 1 ∧ balls (i + 1) % (p*q) = 2) ∨ 
              (balls i = 2 ∧ balls (i + 1) % (p*q) = 1)) <| Fin (p*q)) % 2 = 1 :=
  sorry

end adjacent_pairs_odd_l18_18354


namespace cannot_be_inverse_l18_18648

noncomputable def f {R : Type} [linear_ordered_field R] (g : R → R) := ∀ x: R, f (g x) = x

def g1 (x : ℝ) : ℝ := real.cos x
def g2 (x : ℝ) : ℝ := if x >= 0 then -x^2 else x^2
def g3 (x : ℝ) : ℝ := x^3 - x
def g4 (x : ℝ) : ℝ := real.exp x - real.exp (-x)

theorem cannot_be_inverse (hg1 : ¬∃ f : ℝ → ℝ, ∀ x, f (g1 x) = x)
                          (hg2 : ∃ f : ℝ → ℝ, ∀ x, f (g2 x) = x)
                          (hg3 : ¬∃ f : ℝ → ℝ, ∀ x, f (g3 x) = x)
                          (hg4 : ∃ f : ℝ → ℝ, ∀ x, f (g4 x) = x) :
  true := by sorry

end cannot_be_inverse_l18_18648


namespace derek_dogs_l18_18774

theorem derek_dogs (d c : ℕ) (h1 : d = 90) 
  (h2 : c = d / 3) 
  (h3 : c + 210 = 2 * (d + 120 - d)) : 
  d + 120 - d = 120 :=
by
  sorry

end derek_dogs_l18_18774


namespace find_alpha_find_function_l18_18460

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

theorem find_alpha (α : ℝ) (h : f 2 α = Real.sqrt 2) : α = 1 / 2 := 
  sorry

theorem find_function (h : f 2 (1/2) = Real.sqrt 2) : f = fun x => x ^ (1 / 2) :=
  funext (λ x, by simp [f, h]; sorry)

end find_alpha_find_function_l18_18460


namespace tangent_line_eqn_range_of_t_l18_18484

-- Part (1): Prove the equation of the tangent line
theorem tangent_line_eqn (x : ℝ) (h : x = Real.exp 1) :
  let F := λ x : ℝ, x * Real.log x in
  (∀ y : ℝ, (2 * x - y - Real.exp 1 = 0) ↔ 
    (∃ x y : ℝ, y - Real.exp 1 = (Real.log (Real.exp 1) + 1) * (x - Real.exp 1))) :=
  sorry

-- Part (2): Prove the range of values for t
theorem range_of_t (t : ℝ) :
  let F := λ x : ℝ, x * Real.log x in
  (∀ x : ℝ, x ∈ Icc (Real.exp (-2)) 1 →
    (F x - t = 0) ∨ (F x - t ≠ 0)) →
  (t ∈ Icc (-Real.exp (-1)) (-2 * Real.exp (-2))) :=
  sorry

end tangent_line_eqn_range_of_t_l18_18484


namespace max_min_values_f_monotonic_range_f_l18_18491

open Real

noncomputable def f (x θ : ℝ) : ℝ :=
  x^2 + 2 * x * tan θ - 1

theorem max_min_values_f (θ : ℝ) (x : ℝ) (h_θ : θ = -π / 6) (hx : x ∈ Icc (-1) (sqrt 3)) :
  let f_x := (x - sqrt 3 / 3)^2 - 4 / 3 in
  (∀ x', x' = sqrt 3 / 3 → f x' θ = -4 / 3) ∧
  (∀ x', x' = -1 → f x' θ = 2 * sqrt 3 / 3) :=
sorry

theorem monotonic_range_f (θ : ℝ) :
  (∀ x ∈ Icc (-1) (sqrt 3), f x θ = f x θ) →
  θ ∈ Icc (-π / 2) (π / 2) →
  θ ∈ Icc (-π / 2) (-π / 3) ∪ Icc (π / 4) (π / 2) :=
sorry

end max_min_values_f_monotonic_range_f_l18_18491


namespace percent_increase_end_of_second_quarter_l18_18742

-- Define the initial share price as a non-negative real number.
def share_price_at_beginning_of_year (P : ℝ) := P ≥ 0

-- Define the share price at the end of the first quarter,
-- which is 30% higher than at the beginning of the year.
def share_price_after_first_quarter (P : ℝ) := 1.30 * P

-- Define the share price at the end of the second quarter,
-- which is 15.384615384615374% higher than at the end of the first quarter.
def percent_increase_second_quarter := 0.15384615384615374

noncomputable def share_price_after_second_quarter (P : ℝ) :=
  share_price_after_first_quarter P * (1 + percent_increase_second_quarter)

-- The proof statement:
theorem percent_increase_end_of_second_quarter (P : ℝ) (hP : share_price_at_beginning_of_year P) :
  share_price_after_second_quarter P = 1.50 * P := 
sorry

end percent_increase_end_of_second_quarter_l18_18742


namespace find_t_l18_18839

variable (g V V0 c S t : ℝ)
variable (h1 : V = g * t + V0 + c)
variable (h2 : S = (1/2) * g * t^2 + V0 * t + c * t^2)

theorem find_t
  (h1 : V = g * t + V0 + c)
  (h2 : S = (1/2) * g * t^2 + V0 * t + c * t^2) :
  t = 2 * S / (V + V0 - c) :=
sorry

end find_t_l18_18839


namespace smallest_possible_difference_l18_18913

theorem smallest_possible_difference
  (persons : Finset Person) (h_card : persons.card = 2017)
  (h_friends : ∀ (p1 p2 : Person), p1 ≠ p2 → ∃! (common_friend : Person), 
    common_friend ≠ p1 ∧ common_friend ≠ p2 ∧ friends p1 common_friend ∧ friends p2 common_friend) :
  ∃ p_max p_min : Person, 
    (∀ p, num_friends p_max ≥ num_friends p ∧ num_friends p_min ≤ num_friends p) ∧ 
    num_friends p_max - num_friends p_min = 2014 := 
sorry

end smallest_possible_difference_l18_18913


namespace length_XP_l18_18947

-- Definitions representing the given points and distances
variables (X Y Z P Q R : Type*)
variables [metric_space X] [metric_space Y] [metric_space Z]
variables (d_XP : ℝ) (d_YR : ℝ) (d_PQ : ℝ) (d_XZ : ℝ)
variables (P_not_on_XZ : P ≠ XZ)
variables (Q_on_XZ : Q = XZ)
variables (PQ_perpendicular_XZ : is_perpendicular P Q XZ)
variables (YR_perpendicular_XP : is_perpendicular Y R X P)
variables (YR_eq_6 : d_YR = 6)
variables (PQ_eq_10 : d_PQ = 10)
variables (XZ_eq_7 : d_XZ = 7)

-- To prove
theorem length_XP : d_XP = 35/3 :=
by { sorry }

end length_XP_l18_18947


namespace salt_concentration_third_solution_l18_18971

theorem salt_concentration_third_solution (X : ℝ) :
  (let initial_solution_salt_percent := 15 / 100 in
   let second_solution_salt_percent := 16 / 100 in
   let final_solution_salt_percent := X / 100 in
   let first_salt := 15 in
   let removed_first_solution := 1/4 * first_salt in
   let first_remaining_salt := first_salt - removed_first_solution in
   let second_solution_total_salt := 16 in
   let added_second_solution_salt := second_solution_total_salt - first_remaining_salt in
   let removed_second_solution := 1/3 * second_solution_total_salt in
   let second_remaining_salt := second_solution_total_salt - removed_second_solution in
   let final_solution_salt := second_remaining_salt in
   let added_third_solution_salt := (X - final_solution_salt) in
   let added_third_solution_amount := 1/3 * 100 in
   100 * (added_third_solution_salt / added_third_solution_amount)) = 3 * X - 32 :=
by
  sorry

end salt_concentration_third_solution_l18_18971


namespace one_plus_i_pow_four_eq_neg_four_l18_18758

theorem one_plus_i_pow_four_eq_neg_four : (1 + complex.I)^4 = -4 :=
by
  sorry

end one_plus_i_pow_four_eq_neg_four_l18_18758


namespace steps_per_floor_l18_18744

def elevator_time_to_ground := 60 -- in seconds
def jake_extra_time := 30 -- in seconds
def jake_steps_per_second := 3 -- steps per second
def total_floors := 9

theorem steps_per_floor : 
  ∀ (elevator_time_to_ground jake_extra_time jake_steps_per_second total_floors : ℕ),
  elevator_time_to_ground = 60 →
  jake_extra_time = 30 →
  jake_steps_per_second = 3 →
  total_floors = 9 →
  let jake_total_time := elevator_time_to_ground + jake_extra_time in
  let jake_total_steps := jake_steps_per_second * jake_total_time in
  jake_total_steps / total_floors = 30 :=
by
  intros
  rw [←nat.mul_div_cancel' (by norm_num : 9 ∣ 270)]
  simp[sorry]

end steps_per_floor_l18_18744


namespace length_PN_eq_l18_18368

-- Define the parameters and theorems
theorem length_PN_eq :
  let O := (0 : ℝ, 0 : ℝ),
      A := (-4 : ℝ, 0 : ℝ),
      B := (4 : ℝ, 0 : ℝ),
      C := (0 : ℝ, 4√2 : ℝ),
      N := ((A.1 + C.1) / 2, (A.2 + C.2) / 2),
      P := (0 : ℝ, 4 : ℝ),
      PN := real.sqrt ((P.1 - N.1)^2 + (P.2 - N.2)^2)
  in PN = 4 * real.sqrt (2 - 2 * real.sqrt 2) :=
begin
  sorry
end

end length_PN_eq_l18_18368


namespace minimum_value_of_reciprocal_sum_l18_18901

theorem minimum_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 * a * (-1) - b * 2 + 2 = 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a * (-1) - b * 2 + 2 = 0 ∧ (a + b = 1) ∧ (a = 1/2 ∧ b = 1/2) ∧ (1/a + 1/b = 4) :=
by
  sorry

end minimum_value_of_reciprocal_sum_l18_18901


namespace product_of_roots_l18_18165

theorem product_of_roots (x1 x2 : ℝ) (h : x1 * x2 = -2) :
  (∀ x : ℝ, x^2 + x - 2 = 0 → (x = x1 ∨ x = x2)) → x1 * x2 = -2 :=
by
  intros h_root
  exact h

end product_of_roots_l18_18165


namespace part_a_part_b_l18_18348

-- Part A: Proving the specific values of p and q
theorem part_a (p q : ℝ) : 
  (∀ x : ℝ, (x + 3) ^ 2 + (7 * x + p) ^ 2 = (kx + m) ^ 2) ∧
  (∀ x : ℝ, (3 * x + 5) ^ 2 + (p * x + q) ^ 2 = (cx + d) ^ 2) → 
  p = 21 ∧ q = 35 :=
sorry

-- Part B: Proving the new polynomial is a square of a linear polynomial
theorem part_b (a b c A B C : ℝ) (hab : a ≠ 0) (hA : A ≠ 0) (hb : b ≠ 0) (hB : B ≠ 0)
  (habc : (∀ x : ℝ, (a * x + b) ^ 2 + (A * x + B) ^ 2 = (kx + m) ^ 2) ∧
         (∀ x : ℝ, (b * x + c) ^ 2 + (B * x + C) ^ 2 = (cx + d) ^ 2)) :
  ∀ x : ℝ, (c * x + a) ^ 2 + (C * x + A) ^ 2 = (lx + n) ^ 2 :=
sorry

end part_a_part_b_l18_18348


namespace consecutive_cubes_sum_to_perfect_square_l18_18000

theorem consecutive_cubes_sum_to_perfect_square :
  (∃ n m : ℕ, n ≠ 1 ∧ m > 3 ∧ ∑ i in finset.range ((m - n) + 1), (n + i) ^ 3 = (∑ j in finset.range ((m - n) + 1), (n + j) ^ 3)^2) ∧
  (∀ s t : ℕ, (s ≠ 1 ∧ t > 3 ∧ ∑ i in finset.range ((t - s) + 1), (s + i) ^ 3 = (∑ j in finset.range ((t - s) + 1), (s + j) ^ 3)^2) →
  (s, t) = (14, 25)) :=
by
  sorry

end consecutive_cubes_sum_to_perfect_square_l18_18000


namespace acute_angled_triangles_l18_18967

theorem acute_angled_triangles (n : ℕ) (h : 3 < n) :
  (∃ (pts : finset ℝ), pts.card = n ∧ 
    (∀ (t : finset ℝ), t.card = 3 → 
      (t ⊆ pts) → (acute_angle_triangle t) ↔ 
      (t.center_in_circle))) → (n = 4 ∨ n = 5) :=
sorry

end acute_angled_triangles_l18_18967


namespace profit_margin_in_terms_of_retail_price_l18_18833

theorem profit_margin_in_terms_of_retail_price
  (k c P_R : ℝ) (h1 : ∀ C, P = k * C) (h2 : ∀ C, P_R = c * (P + C)) :
  P = (k / (c * (k + 1))) * P_R :=
by sorry

end profit_margin_in_terms_of_retail_price_l18_18833


namespace tan_half_angle_l18_18834

noncomputable def point := (-1 : ℝ, 2 : ℝ)
noncomputable def alpha : ℝ := sorry

theorem tan_half_angle (h : ∃ θ, (cos θ) = -1 / real.sqrt 5 ∧ (sin θ) = 2 / real.sqrt 5) : 
  real.tan (alpha / 2) = (1 + real.sqrt 5) / 2 :=
sorry

end tan_half_angle_l18_18834


namespace ratio_of_segments_l18_18799

variable {a b : ℝ}

theorem ratio_of_segments (h_distinct : distinct_lines_on_plane 5) 
  (h_lengths : segment_lengths [a, a, a, a, a, a, sqrt 2 * a, b]) : 
  b / a = sqrt 2 :=
sorry

end ratio_of_segments_l18_18799


namespace log_inequality_l18_18887

theorem log_inequality (a : ℝ) (h : real.log_base a (2/3) > 1) : a ∈ set.Ioo (2/3 : ℝ) 1 :=
by sorry

end log_inequality_l18_18887


namespace solve_equation_1_solve_equation_2_l18_18599

theorem solve_equation_1 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by sorry

theorem solve_equation_2 (x : ℝ) : 2 * x^2 + 5 * x + 3 = 0 ↔ (x = -1 ∨ x = -3/2) :=
by sorry

end solve_equation_1_solve_equation_2_l18_18599


namespace max_sum_nonneg_l18_18005

theorem max_sum_nonneg (a b c d : ℝ) (h : a + b + c + d = 0) : 
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0 := 
sorry

end max_sum_nonneg_l18_18005


namespace stacy_history_paper_pages_l18_18265

def stacy_paper := 1 -- Number of pages Stacy writes per day
def days_to_finish := 12 -- Number of days Stacy has to finish the paper

theorem stacy_history_paper_pages : stacy_paper * days_to_finish = 12 := by
  sorry

end stacy_history_paper_pages_l18_18265


namespace bisection_method_next_interval_l18_18327

theorem bisection_method_next_interval (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - x - 5) :
  f 1 = -5 ∧ f 2 = 1 ∧ f 1.5 = -25/8 → ∃ c : ℝ, 1.5 < c ∧ c < 2 ∧ f c = 0 :=
by {
  intro h1,
  sorry
}

end bisection_method_next_interval_l18_18327


namespace root_of_unity_product_l18_18510

theorem root_of_unity_product (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  (1 - ω + ω^2) * (1 + ω - ω^2) = 1 :=
  sorry

end root_of_unity_product_l18_18510


namespace elena_butter_l18_18087

theorem elena_butter (cups_flour butter : ℕ) (h1 : cups_flour * 4 = 28) (h2 : butter * 4 = 12) : butter = 3 := 
by
  sorry

end elena_butter_l18_18087


namespace folded_triangle_sqrt_equals_l18_18706

noncomputable def folded_triangle_length_squared (s : ℕ) (d : ℕ) : ℚ :=
  let x := (2 * s * s - 2 * d * s)/(2 * d)
  let y := (2 * s * s - 2 * (s - d) * s)/(2 * (s - d))
  x * x - x * y + y * y

theorem folded_triangle_sqrt_equals :
  folded_triangle_length_squared 15 11 = (60118.9025 / 1681 : ℚ) := sorry

end folded_triangle_sqrt_equals_l18_18706


namespace cost_of_2000_pieces_of_gum_l18_18996

theorem cost_of_2000_pieces_of_gum
  (cost_per_piece_in_cents : Nat)
  (pieces_of_gum : Nat)
  (conversion_rate_cents_to_dollars : Nat)
  (h1 : cost_per_piece_in_cents = 5)
  (h2 : pieces_of_gum = 2000)
  (h3 : conversion_rate_cents_to_dollars = 100) :
  (cost_per_piece_in_cents * pieces_of_gum) / conversion_rate_cents_to_dollars = 100 := 
by
  sorry

end cost_of_2000_pieces_of_gum_l18_18996


namespace board_partition_possible_l18_18011

variable (m n : ℕ)

theorem board_partition_possible (hm : m > 15) (hn : n > 15) :
  ((∃ k1, m = 5 * k1 ∧ ∃ k2, n = 4 * k2) ∨ (∃ k3, m = 4 * k3 ∧ ∃ k4, n = 5 * k4)) :=
sorry

end board_partition_possible_l18_18011


namespace projection_correct_l18_18133

noncomputable def projection_vector (e1 e2 : ℝ^2) : ℝ :=
  let angle := 60 -- degrees
  let cos60 := (1:ℝ) / 2
  let e1_dot_e2 := e1 • e2
  let unit_vector := (norm e1) = 1 ∧ (norm e2) = 1
  if angle.to_real = 60 and e1_dot_e2 = cos60 and unit_vector then
    (e2 - (2:ℝ) • e1) • (e1 + e2) / norm (e1 + e2)
  else
    0

theorem projection_correct (e1 e2 : ℝ^2) (h_angle : angle = 60)
  (h_unit : norm e1 = 1 ∧ norm e2 = 1) (h_dot : e1 • e2 = (1:ℝ) / 2) :
  projection_vector e1 e2 = -((sqrt 3) / 6) :=
  sorry

end projection_correct_l18_18133


namespace positive_even_diffs_count_l18_18505

-- Define the set from 1 to 20
def my_set : Set ℕ := {x | 1 ≤ x ∧ x ≤ 20}

-- Define the predicate for a positive even integer that can be represented as a difference of two distinct elements of the set
def is_even_diff (x : ℕ) : Prop :=
  ∃ a b ∈ my_set, a ≠ b ∧ x = abs (a - b) ∧ x % 2 = 0

-- Define the set of positive even integers within the desired range
def even_diffs : Set ℕ := {x | is_even_diff x ∧ x > 0 ∧ x % 2 = 0}

-- The number of different positive even integers
def num_even_diffs : ℕ := (even_diffs).to_finset.card

-- The theorem statement
theorem positive_even_diffs_count : num_even_diffs = 9 := by
  sorry

end positive_even_diffs_count_l18_18505


namespace N_lies_on_AC_l18_18589

variables {A B C D K H M N : Type}

-- Definitions
def is_square (A B C D : Type) : Prop := sorry
def point_on_side (P : Type) (A B : Type) : Prop := sorry
def point_on_segment (P : Type) (K H : Type) : Prop := sorry
def circumcircle (Δ : Type) : Type := sorry
def intersect (γ1 γ2 : Type) : set Type := sorry

-- Conditions
axiom square_ABCD : is_square A B C D
axiom K_on_AB : point_on_side K A B
axiom H_on_CD : point_on_side H C D
axiom M_on_KH : point_on_segment M K H

-- Circumcircles and intersection
noncomputable def Ω₁ : Type := circumcircle (A, K, M)
noncomputable def Ω₂ : Type := circumcircle (M, H, C)
noncomputable def N_intersect_Ω₁_Ω₂ : ∃ N ≠ M, N ∈ intersect Ω₁ Ω₂ := sorry

-- Conclusion
theorem N_lies_on_AC :
  ∃ N ≠ M, N ∈ intersect Ω₁ Ω₂ →
  (N ∈ AC) :=
by
  assume hN : ∃ N ≠ M, N ∈ intersect Ω₁ Ω₂
  show N ∈ AC
  from sorry

end N_lies_on_AC_l18_18589


namespace trapezoid_tangent_lengths_equal_l18_18677

-- Define the geometry of the trapezoid and intersecting angle bisectors
structure Trapezoid (A B C D E K M N : Type*) where
  AB_parallel_CD : Parallel (Line A B) (Line C D)
  angle_bisectors_intersect_E : Bisection (Angle A) (Line A D) E ∧ Bisection (Angle D) (Line D A) E
  E_on_BC : OnLine (Line B C) E
  inscribed_circles : InscribedCircle (Triangle A D E) ∧ InscribedCircle (Triangle D C E) ∧ InscribedCircle (Triangle A B E)
  circle_touches_AB_at_K : Touches (InscribedCircle (Triangle A B E)) (Line A B) K
  circles_touches_DE_at_M_and_N : Touches (InscribedCircle (Triangle A D E)) (Line D E) M ∧ Touches (InscribedCircle (Triangle D C E)) (Line D E) N

-- Prove BK = MN
theorem trapezoid_tangent_lengths_equal
  {A B C D E K M N : Type*}
  (trap : Trapezoid A B C D E K M N) :
  SegmentLength (Line B K) = SegmentLength (Line M N) :=
sorry

end trapezoid_tangent_lengths_equal_l18_18677


namespace inequality_solution_set_l18_18982

theorem inequality_solution_set :
  { x : ℝ | (10 * x^2 + 20 * x - 68) / ((2 * x - 3) * (x + 4) * (x - 2)) < 3 } =
  { x : ℝ | (-4 < x ∧ x < -2) ∨ (-1 / 3 < x ∧ x < 3 / 2) } :=
by
  sorry

end inequality_solution_set_l18_18982


namespace abs_nested_expression_l18_18572

theorem abs_nested_expression (x : ℝ) (h : x = 2023) : 
  abs (abs (abs x - x) - abs x) - x = 0 :=
by
  subst h
  sorry

end abs_nested_expression_l18_18572


namespace geometric_sequence_term_302_l18_18542

def geometric_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ (n - 1)

theorem geometric_sequence_term_302 :
  let a := 8
  let r := -2
  geometric_sequence a r 302 = -2^304 := by
  sorry

end geometric_sequence_term_302_l18_18542


namespace correct_linear_regression_statement_l18_18927

-- Definitions based on the conditions:
def linear_regression (b a e : ℝ) (x : ℝ) : ℝ := b * x + a + e

def statement_A (b a e : ℝ) (x : ℝ) : Prop := linear_regression b a e x = b * x + a + e

def statement_B (b a e : ℝ) (x : ℝ) : Prop := ∀ x1 x2, (linear_regression b a e x1 ≠ linear_regression b a e x2) → (x1 ≠ x2)

def statement_C (b a e : ℝ) (x : ℝ) : Prop := ∃ (other_factors : ℝ), linear_regression b a e x = b * x + a + other_factors + e

def statement_D (b a e : ℝ) (x : ℝ) : Prop := (e ≠ 0) → false

-- The proof statement
theorem correct_linear_regression_statement (b a e : ℝ) (x : ℝ) :
  (statement_C b a e x) :=
sorry

end correct_linear_regression_statement_l18_18927


namespace division_reciprocal_multiplication_l18_18789

theorem division_reciprocal_multiplication : (4 / (8 / 13 : ℚ)) = (13 / 2 : ℚ) := 
by
  sorry

end division_reciprocal_multiplication_l18_18789


namespace min_colors_required_l18_18455

theorem min_colors_required (m : ℕ) (n : ℕ) (grid : matrix (fin m) (fin m) (fin n)) :
  (∀ (c : fin n), ∀ (i : fin m), ∀ (j : fin m), 
    grid i j = c →
    ∀ (a b c d : fin m), 
    a < b ∧ b < c ∧ c < d ∧ 
    grid a j = c ∧ grid b j = c ∧ grid c j = c ∧ grid d j = c →
    ∀ (k : fin m) (r : fin m),
    k < a → grid k r ≠ c → 
    r > d → grid k r ≠ c 
  ) →
  n ≥ 506 :=
by
  sorry

end min_colors_required_l18_18455


namespace question_1_smallest_positive_period_question_1_interval_monotonic_decrease_question_2_solution_inequality_l18_18582

-- Define the given function f
def f (x : ℝ) (a : ℝ) : ℝ := sqrt 3 * sin x * cos x + cos x ^ 2 + a

-- Given conditions
def cond_1 := (∃ a : ℝ, min (f (x := -π/6) a) + max (f (x := π/3) a) = 3 / 2)
def domain := x ∈ Icc (-π/6) (π/3)

-- The proofs to be stated (without steps)
theorem question_1_smallest_positive_period (a : ℝ) : 
  ∀ x : ℝ, f x a = f (x + π) a := sorry

theorem question_1_interval_monotonic_decrease (a : ℝ) :
  ∀ k : ℤ, x ∈ Icc (k * π + π / 6) (k * π + 2 * π / 3) → 
  (f (x + δ) a < f x a → δ > 0) := sorry

theorem question_2_solution_inequality (a : ℝ) (H : a = 0) : 
  ∀ x : ℝ, x ∈ Icc (-π/6) (π/3) → (f x a > 1 ↔ x ∈ Ioc 0 (π/3)) := sorry

end question_1_smallest_positive_period_question_1_interval_monotonic_decrease_question_2_solution_inequality_l18_18582


namespace system_of_inequalities_solution_l18_18155

theorem system_of_inequalities_solution (a b : ℝ) :
  (∀ x : ℝ, 2x - a < 1 ∧ x - 2b > 3 → -1 < x ∧ x < 1) →
  (a + 1) * (b - 1) = -6 := by
  sorry

end system_of_inequalities_solution_l18_18155


namespace max_mn_condition_l18_18201

theorem max_mn_condition (m n : ℤ) (p q : ℕ) (hmn_int : m + n = p) (hmn_prime1 : nat.prime p) 
(hpq_int : m - n = q) (hmn_prime2 : nat.prime q) 
(hp_lt_100 : p < 100) (hq_lt_100 : q < 100) : 
  ∃ (m n : ℤ), m + n = p ∧ m - n = q ∧ m * n = 2350 :=
  sorry

end max_mn_condition_l18_18201


namespace train_speed_is_36_kmph_l18_18387

def train_speed (train_length platform_length crossing_time: ℝ) : ℝ :=
  let distance := train_length + platform_length
  let speed_mps := distance / crossing_time
  let speed_kmph := speed_mps * 3.6
  speed_kmph

theorem train_speed_is_36_kmph
  (train_length : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (h_train_length : train_length = 300)
  (h_platform_length : platform_length = 300)
  (h_crossing_time : crossing_time = 60) :
  train_speed train_length platform_length crossing_time = 36 :=
by
  simp [train_speed, h_train_length, h_platform_length, h_crossing_time]
  sorry

end train_speed_is_36_kmph_l18_18387


namespace problem_statement_l18_18541

noncomputable def area_of_quad_given_conditions : Prop :=
  let AE := 30
  let BE := AE / 2
  let AB := BE * Real.sqrt 3
  let CE := BE / 2
  let BC := CE * Real.sqrt 3
  let ED := CE / 2
  let CD := ED * Real.sqrt 3
  let AD := ED / 2
  let DA := AD * Real.sqrt 3
  let area_△ABE := (AB * BE) / 2
  let area_△BCE := (BC * CE) / 2
  let area_△CDE := (CD * ED) / 2
  let area_△DEA := (DA * AD) / 2
  let total_area := area_△ABE + area_△BCE + area_△CDE + area_△DEA
  total_area = 149.4140625 * Real.sqrt 3

theorem problem_statement : area_of_quad_given_conditions :=
sorry

end problem_statement_l18_18541


namespace judy_pencil_cost_l18_18934

theorem judy_pencil_cost 
  (pencils_per_week : ℕ)
  (days_per_week : ℕ)
  (pack_cost : ℕ)
  (pack_size : ℕ)
  (total_days : ℕ)
  (pencil_usage : pencils_per_week = 10)
  (school_days : days_per_week = 5)
  (cost_per_pack : pack_cost = 4)
  (pencils_per_pack : pack_size = 30)
  (duration : total_days = 45) : 
  ∃ (total_cost : ℕ), total_cost = 12 :=
sorry

end judy_pencil_cost_l18_18934


namespace total_letters_correct_l18_18253

-- Define the conditions
def letters_January := 6
def letters_February := 9
def letters_March := 3 * letters_January

-- Definition of the total number of letters sent
def total_letters := letters_January + letters_February + letters_March

-- The statement we need to prove in Lean
theorem total_letters_correct : total_letters = 33 := 
by
  sorry

end total_letters_correct_l18_18253


namespace arithmetic_sequence_k_l18_18540

theorem arithmetic_sequence_k (d : ℝ) (h : d ≠ 0) (k : ℕ) :
  let a : ℕ → ℝ := λ n, (n - 1) * d in
  (∑ i in finset.range 7, a (i + 1)) = a k → (k = 22) :=
by
  intro a_sum_eq_ak
  sorry

end arithmetic_sequence_k_l18_18540


namespace decreasing_exponential_iff_l18_18628

theorem decreasing_exponential_iff {a : ℝ} :
  (∀ x y : ℝ, x < y → (a - 1)^y < (a - 1)^x) ↔ (1 < a ∧ a < 2) :=
by 
  sorry

end decreasing_exponential_iff_l18_18628


namespace complement_A_in_U_l18_18231

variable U : Set ℝ := Set.univ
variable A : Set ℝ := {x | |x - 1| > 1}

theorem complement_A_in_U :
  U \ A = Set.Icc 0 2 :=
by
  sorry

end complement_A_in_U_l18_18231


namespace sum_binomial_coeffs_is_128_coefficient_of_x_squared_is_minus_7_l18_18186

noncomputable def expansion_sum_binomial_coeffs (x : ℂ) : ℂ :=
  (∑ k in Finset.range 8, Complex.binom 7 k * Complex.sqrt x^(7 - k) * (-1)^k * x^(-k))

noncomputable def coefficient_of_x_squared (x : ℂ) : ℂ :=
  let term := Complex.mul (Complex.binom 7 1) (Complex.mul (-1) (x^((7 - 3 * 1) / 2)))
  term

theorem sum_binomial_coeffs_is_128 (x : ℂ) : 
  expansion_sum_binomial_coeffs x = 128 :=
by 
  sorry

theorem coefficient_of_x_squared_is_minus_7 (x : ℂ) :
  coefficient_of_x_squared x = -7 :=
by
  sorry

end sum_binomial_coeffs_is_128_coefficient_of_x_squared_is_minus_7_l18_18186


namespace curve_is_circle_l18_18626

-- Definitions in Cartesian and Polar Coordinates
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

-- The given polar equation
def polar_equation (θ : ℝ) : ℝ :=
  5 * sin θ

-- The corresponding cartesian equation derived from polar
def cartesian_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 5 * y

theorem curve_is_circle : ∀ θ : ℝ, ∃ x y : ℝ, (polar_to_cartesian (polar_equation θ) θ = (x, y)) ∧ cartesian_equation x y :=
by
  sorry

end curve_is_circle_l18_18626


namespace probability_of_appearing_at_least_once_l18_18983

-- Definition of the conditions
def fair_die : Type := Fin 6

def dice_rolls (n : Nat) : Type := Vector fair_die n

def appears_at_least_once (rolls : dice_rolls 10) : Prop := 
  ∀ n : fair_die, ∃ i : Fin 10, rolls[i] = n

-- The theorem statement
theorem probability_of_appearing_at_least_once :
  (P_{\pi[dice_rolls 10]} (appears_at_least_once) = 0.729) :=
sorry

end probability_of_appearing_at_least_once_l18_18983


namespace cost_of_one_box_l18_18691

def boxContains (b : ℕ) : Prop := b = 3
def friends (f : ℕ) : Prop := f = 6
def barsPerFriend (bp : ℕ) : Prop := bp = 2
def costPerFriend (cf : ℕ) : Prop := cf = 5
def totalCost (tc : ℝ) : Prop := tc = 30
def boxesNeeded (bn : ℕ) : Prop (b f bp : ℕ) := (f * bp) / b
def costPerBox (cb : ℝ) (tc bn : ℕ) : Prop := cb = tc / bn

theorem cost_of_one_box (b f bp cf : ℕ) (tc cb : ℝ) :
  boxContains b → friends f → barsPerFriend bp → costPerFriend cf → totalCost tc → 
  costPerBox cb tc (boxesNeeded b f bp) → cb = 7.50 :=
by
  intros
  sorry

end cost_of_one_box_l18_18691


namespace circle_tangent_l18_18761

def is_right_triangle (X Y Z : Type) (right_angle_at : X -> Y -> Z -> Prop) : Prop := sorry

def distance (X Y : Type) : Type := sorry

theorem circle_tangent (X Y Z : Type)
  (triangle_right : is_right_triangle X Y Z)
  (XZ_sq : distance X Z = real.sqrt 85)
  (XY_len : distance X Y = 7)
  (circle_center : (center : X -> distance center Y)) :
  ∃ ZQ, distance Z Q = 6 := sorry

end circle_tangent_l18_18761


namespace lowest_fraction_of_job_in_one_hour_l18_18350

-- Define the rates at which each person can work
def rate_A : ℚ := 1/3
def rate_B : ℚ := 1/4
def rate_C : ℚ := 1/6

-- Define the combined rates for each pair of people
def combined_rate_AB : ℚ := rate_A + rate_B
def combined_rate_AC : ℚ := rate_A + rate_C
def combined_rate_BC : ℚ := rate_B + rate_C

-- The Lean 4 statement to prove
theorem lowest_fraction_of_job_in_one_hour : min combined_rate_AB (min combined_rate_AC combined_rate_BC) = 5/12 :=
by 
  -- Here we state that the minimum combined rate is 5/12
  sorry

end lowest_fraction_of_job_in_one_hour_l18_18350


namespace owen_sleep_hours_l18_18972

variable (work_hours : ℕ) (chores_hours : ℕ) (total_hours : ℕ := 24)

def sleep_hours (work_hours chores_hours : ℕ) : ℕ :=
  total_hours - (work_hours + chores_hours)

theorem owen_sleep_hours :
  sleep_hours 6 7 = 11 :=
  by
    unfold sleep_hours
    simp
    sorry

end owen_sleep_hours_l18_18972


namespace scaled_height_of_model_l18_18044

/-- 
Alice is creating a scaled model of a local landmark, a cylindrical tower that is 50 meters tall with a conical roof of 10 meters. 
The actual total volume of the landmark is 150,000 cubic meters. 
Alice's model will contain a total volume of 1.5 cubic meters. 
What should be the height of Alice's scaled model?
-/
theorem scaled_height_of_model :
  let actual_volume := 150000 -- The volume of the actual landmark in cubic meters
  let model_volume := 1.5 -- The volume of the scaled model in cubic meters
  let actual_height := 60 -- The total height of the actual landmark in meters (50 + 10)
  let volume_ratio := actual_volume / model_volume -- The volume ratio
  let scale_factor := real.cbrt volume_ratio -- The scale factor for the height
  let model_height := actual_height / scale_factor -- The height of the scaled model
  (abs (model_height - 1.29) < 0.01) -- approximately 1.29 meters
  :=
by
  sorry

end scaled_height_of_model_l18_18044


namespace sum_rational_roots_h_l18_18082

def h : ℚ[X] := X^3 - 6*X^2 + 11*X - 6

theorem sum_rational_roots_h : ∑ r in (roots h).filter (λ x, x.is_rational), r = 6 := 
sorry

end sum_rational_roots_h_l18_18082


namespace tire_circumference_is_one_meter_l18_18895

-- Definitions for the given conditions
def car_speed : ℕ := 24 -- in km/h
def tire_rotations_per_minute : ℕ := 400

-- Conversion factors
def km_to_m : ℕ := 1000
def hour_to_min : ℕ := 60

-- The equivalent proof problem
theorem tire_circumference_is_one_meter 
  (hs : car_speed * km_to_m / hour_to_min = 400 * tire_rotations_per_minute)
  : 400 = 400 * 1 := 
by
  sorry

end tire_circumference_is_one_meter_l18_18895


namespace prob_two_red_two_blue_is_3_over_14_l18_18363

def red_marbles : ℕ := 15
def blue_marbles : ℕ := 10
def total_marbles : ℕ := red_marbles + blue_marbles
def chosen_marbles : ℕ := 4

noncomputable def prob_two_red_two_blue : ℚ :=
  let total_ways := (Nat.choose total_marbles chosen_marbles : ℚ)
  let ways_two_red := (Nat.choose red_marbles 2)
  let ways_two_blue := (Nat.choose blue_marbles 2)
  let favorable_outcomes := 6 * ways_two_red * ways_two_blue
  favorable_outcomes / total_ways

theorem prob_two_red_two_blue_is_3_over_14 : prob_two_red_two_blue = 3 / 14 :=
  sorry

end prob_two_red_two_blue_is_3_over_14_l18_18363


namespace additional_rocks_needed_l18_18966

-- Define the dimensions of the garden
def length (garden : Type) : ℕ := 15
def width (garden : Type) : ℕ := 10
def rock_cover (rock : Type) : ℕ := 1

-- Define the number of rocks Mrs. Hilt has
def rocks_possessed (mrs_hilt : Type) : ℕ := 64

-- Define the perimeter of the garden
def perimeter (garden : Type) : ℕ :=
  2 * (length garden + width garden)

-- Define the number of rocks required for the first layer
def rocks_first_layer (garden : Type) : ℕ :=
  perimeter garden

-- Define the number of rocks required for the second layer (only longer sides)
def rocks_second_layer (garden : Type) : ℕ :=
  2 * length garden

-- Define the total number of rocks needed
def total_rocks_needed (garden : Type) : ℕ :=
  rocks_first_layer garden + rocks_second_layer garden

-- Prove the number of additional rocks Mrs. Hilt needs
theorem additional_rocks_needed (garden : Type) (mrs_hilt : Type):
  total_rocks_needed garden - rocks_possessed mrs_hilt = 16 := by
  sorry

end additional_rocks_needed_l18_18966


namespace proof_b_minus_d_l18_18129

theorem proof_b_minus_d (a b c d : ℕ) 
    (h₁ : log a b = log c d)
    (h₂ : a - c = 9)
    (h₃ : a^3 = b^2)
    (h₄ : c^5 = d^4) : b - d = 93 :=
sorry

end proof_b_minus_d_l18_18129


namespace total_present_ages_l18_18685

variables (P Q : ℕ)

theorem total_present_ages :
  (P - 8 = (Q - 8) / 2) ∧ (P * 4 = Q * 3) → (P + Q = 28) :=
by
  sorry

end total_present_ages_l18_18685


namespace num_pos_integers_congruent_to_4_mod_7_l18_18880

theorem num_pos_integers_congruent_to_4_mod_7 (n : ℕ) (h1 : n < 500) (h2 : ∃ k : ℕ, n = 7 * k + 4) : 
  ∃ total : ℕ, total = 71 :=
sorry

end num_pos_integers_congruent_to_4_mod_7_l18_18880


namespace n_squared_plus_2n_plus_3_mod_50_l18_18892

theorem n_squared_plus_2n_plus_3_mod_50 (n : ℤ) (hn : n % 50 = 49) : (n^2 + 2 * n + 3) % 50 = 2 := 
sorry

end n_squared_plus_2n_plus_3_mod_50_l18_18892


namespace sum_of_perfect_square_divisors_of_544_l18_18102

def divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n + 1))

def is_perfect_square (n : ℕ) : Bool :=
  ∃ k : ℕ, k * k = n

def perfect_square_divisors (n : ℕ) : List ℕ :=
  (divisors n).filter is_perfect_square

theorem sum_of_perfect_square_divisors_of_544 : 
  ∑ d in perfect_square_divisors 544, d = 21 :=
sorry

end sum_of_perfect_square_divisors_of_544_l18_18102


namespace wife_weekly_savings_correct_l18_18377

-- Define constants
def monthly_savings_husband := 225
def num_months := 4
def weeks_per_month := 4
def num_weeks := num_months * weeks_per_month
def stocks_per_share := 50
def num_shares := 25
def invested_amount := num_shares * stocks_per_share
def total_savings := 2 * invested_amount

-- Weekly savings amount to prove
def weekly_savings_wife := 100

-- Total savings calculation condition
theorem wife_weekly_savings_correct :
  (monthly_savings_husband * num_months + weekly_savings_wife * num_weeks) = total_savings :=
by
  sorry

end wife_weekly_savings_correct_l18_18377


namespace limit_of_sin_x_plus_x_power_sin_x_plus_x_l18_18405

open Real

theorem limit_of_sin_x_plus_x_power_sin_x_plus_x : 
  (tendsto (λ x, (x + sin x) ^ (sin x + x)) (𝓝 ↑π) (𝓝 (π^π))) := 
by
  sorry

end limit_of_sin_x_plus_x_power_sin_x_plus_x_l18_18405


namespace negation_of_exists_l18_18290

theorem negation_of_exists (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
sorry

example : (¬ ∃ x : ℝ, 2^x < 1) ↔ (∀ x : ℝ, 2^x ≥ 1) :=
by
  exact negation_of_exists (λ x, 2^x < 1)

end negation_of_exists_l18_18290


namespace rectangle_area_l18_18352

theorem rectangle_area (L B : ℕ) 
  (h1 : L - B = 23)
  (h2 : 2 * L + 2 * B = 186) :
  L * B = 2030 := by
  sorry

end rectangle_area_l18_18352


namespace converse_proposition_l18_18622

theorem converse_proposition (x : ℝ) (h : x = 1 → x^2 = 1) : x^2 = 1 → x = 1 :=
by
  sorry

end converse_proposition_l18_18622


namespace distance_between_foci_xy_eq_4_l18_18067

open Real

theorem distance_between_foci_xy_eq_4 : ∀ (x y : ℝ), (x * y = 4) →
  dist (mk 2 2) (mk (-2) (-2)) = 4 * sqrt 2 :=
by
  intros x y h
  have : dist (2, 2) ((-2), (-2)) = sqrt ((2 - (-2))^2 + (2 - (-2))^2) := by sorry
  have : sqrt ((2 - (-2))^2 + (2 - (-2))^2) = sqrt (16 + 16) := by sorry
  have : sqrt (16 + 16) = sqrt 32 := by sorry
  have : sqrt 32 = 4 * sqrt 2 := by sorry
  exact this

end distance_between_foci_xy_eq_4_l18_18067


namespace hardey_fitness_center_ratio_l18_18988

theorem hardey_fitness_center_ratio
  (f m : ℕ)
  (avg_female_weight : ℕ := 140)
  (avg_male_weight : ℕ := 180)
  (avg_overall_weight : ℕ := 160)
  (h1 : avg_female_weight * f + avg_male_weight * m = avg_overall_weight * (f + m)) :
  f = m :=
by
  sorry

end hardey_fitness_center_ratio_l18_18988


namespace t_sum_max_min_l18_18855

noncomputable def t_max (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) : ℝ := sorry
noncomputable def t_min (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) : ℝ := sorry

theorem t_sum_max_min (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) :
  t_max a b h + t_min a b h = 16 / 7 := sorry

end t_sum_max_min_l18_18855


namespace simplify_expression_l18_18981

theorem simplify_expression (x y : ℝ) (h : x ≠ y) : (x^2 - x * y) / (x - y)^2 = x / (x - y) :=
by sorry

end simplify_expression_l18_18981


namespace is_parallel_O_l18_18194

variables {A B C A' B' C' O: Point} 
variables {A1 B1 C1 A1' B1' C1' : Point}

-- Assume necessary conditions
axiom acute_angle_abc : acute_angle_triangle A B C
axiom acute_angle_ab'c' : acute_angle_triangle A' B' C'
axiom point_O_inside_ABC : is_point_inside_triangle O A B C
axiom point_O'_inside_A'B'C' : is_point_inside_triangle O' A' B' C'
axiom perp_OA1_BC : is_perpendicular O A1 B C
axiom perp_OB1_CA : is_perpendicular O B1 C A
axiom perp_OC1_AB : is_perpendicular O C1 A B
axiom perp_O'A1'_B'C' : is_perpendicular O' A1' B' C'
axiom perp_O'B1'_CA' : is_perpendicular O' B1' C' A'
axiom perp_O'C1'_A'B' : is_perpendicular O' C1' A' B'
axiom parallel_OA1_O'A1' : is_parallel A1 O (A1' O')
axiom parallel_OB1_O'B1' : is_parallel B1 O (B1' O')
axiom parallel_OC1_O'C1' : is_parallel C1 O (C1' O')
axiom equal_products: 
  (O.distance A1 * O'.distance A1' =  O.distance B1 * O'.distance B1')
  ∧ (O.distance A1 * O'.distance A1' = O.distance C1 * O'.distance C1')

-- Theorem we want to prove
theorem is_parallel_O'A1'_OA (cond1 cond2 cond3 cond4 cond5 cond6: Prop):
  cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ cond5 ∧ cond6 →
  (is_parallel A1' O' A O) ∧ 
  (is_parallel B1' O' B O) ∧
  (is_parallel C1' O' C O) ∧
  (O.distance A * O'.distance A1' = O.distance B * O'.distance B1') ∧
  (O.distance A * O'.distance A1' = O.distance C * O'.distance C1') :=
sorry

end is_parallel_O_l18_18194


namespace total_bouncy_balls_l18_18234

def red_packs := 4
def yellow_packs := 8
def green_packs := 4
def blue_packs := 6

def red_balls_per_pack := 12
def yellow_balls_per_pack := 10
def green_balls_per_pack := 14
def blue_balls_per_pack := 8

def total_red_balls := red_packs * red_balls_per_pack
def total_yellow_balls := yellow_packs * yellow_balls_per_pack
def total_green_balls := green_packs * green_balls_per_pack
def total_blue_balls := blue_packs * blue_balls_per_pack

def total_balls := total_red_balls + total_yellow_balls + total_green_balls + total_blue_balls

theorem total_bouncy_balls : total_balls = 232 :=
by
  -- calculation proof goes here
  sorry

end total_bouncy_balls_l18_18234


namespace sum_of_digits_squared_22222_l18_18336

theorem sum_of_digits_squared_22222 : 
  let num := 22222 
  in (num ^ 2).digits.sum = 46 := 
by 
  let num := 22222 
  let square_num := num ^ 2 
  let sum_digits := square_num.digits.sum
  have h : square_num = 493817284 := sorry
  have h_sum : sum_digits = 46 := sorry
  exact h_sum

end sum_of_digits_squared_22222_l18_18336


namespace perfect_squares_between_30_and_200_l18_18873

theorem perfect_squares_between_30_and_200 : 
  {n : ℕ | 30 < n ∧ n < 200 ∧ ∃ k : ℕ, n = k^2}.card = 9 :=
by
  -- The proof is omitted.
  sorry

end perfect_squares_between_30_and_200_l18_18873


namespace maximum_k_value_l18_18804

theorem maximum_k_value :
  ∃ (k : ℕ), k ≤ 808 ∧ ∀ (pairs : finset (ℕ × ℕ)), pairs.card = k →
    (∀ (p ∈ pairs) (q ∈ pairs), p ≠ q → p.1 ≠ q.1 ∧ p.2 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.2) →
    (∀ (p ∈ pairs), p.1 < p.2) →
    (∀ (p ∈ pairs), p.1 + p.2 ≤ 2021) →
    (∀ (p q ∈ pairs), p ≠ q → p.1 + p.2 ≠ q.1 + q.2) :=
begin
  sorry,
end

end maximum_k_value_l18_18804


namespace best_purchase_option_l18_18277

-- Define the prices and discount conditions for each store
def technik_city_price_before_discount : ℝ := 2000 + 4000
def technomarket_price_before_discount : ℝ := 1500 + 4800

def technik_city_discount : ℝ := technik_city_price_before_discount * 0.10
def technomarket_bonus : ℝ := technomarket_price_before_discount * 0.20

def technik_city_final_price : ℝ := technik_city_price_before_discount - technik_city_discount
def technomarket_final_price : ℝ := technomarket_price_before_discount

-- The theorem stating the ultimate proof problem
theorem best_purchase_option : technik_city_final_price < technomarket_final_price :=
by
  -- Replace 'sorry' with the actual proof if required
  sorry

end best_purchase_option_l18_18277


namespace problem_p_problem_q_l18_18976

noncomputable def log (a x : ℝ) := Real.log x / Real.log a

theorem problem_p (a : ℝ) (h : a ∈ (set.Ioo 0 1 ∪ set.Ioi 1)) : log a 1 = 0 :=
by
  simp only [log, Real.log_one, zero_div]

theorem problem_q : ¬ ∃ x : ℕ, (x : ℝ)^3 < (x : ℝ)^2 :=
by
  intro ⟨x, hx⟩
  cases x
  · norm_num at hx
  cases x
  · norm_num at hx
  · refine' lt_irrefl _ _
    exact hx

end problem_p_problem_q_l18_18976


namespace find_side_b_of_triangle_l18_18172

theorem find_side_b_of_triangle
  (A B : Real) (a b : Real)
  (hA : A = Real.pi / 6)
  (hB : B = Real.pi / 4)
  (ha : a = 2) :
  b = 2 * Real.sqrt 2 :=
sorry

end find_side_b_of_triangle_l18_18172


namespace partnership_total_annual_gain_l18_18731

theorem partnership_total_annual_gain 
  (x : ℝ) 
  (G : ℝ)
  (hA_investment : x * 12 = A_investment)
  (hB_investment : 2 * x * 6 = B_investment)
  (hC_investment : 3 * x * 4 = C_investment)
  (A_share : (A_investment / (A_investment + B_investment + C_investment)) * G = 6000) :
  G = 18000 := 
sorry

end partnership_total_annual_gain_l18_18731


namespace find_x_satisfying_condition_l18_18964

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

theorem find_x_satisfying_condition : ∀ x : ℝ, (A x ∪ B x = A x) ↔ (x = 2 ∨ x = -2 ∨ x = 0) := by
  sorry

end find_x_satisfying_condition_l18_18964


namespace collinear_O1_O2_A_l18_18922

open EuclideanGeometry

variables {A B C M N O1 O2 : Point}

def triangle_acute (Δ : Triangle) : Prop :=
  Δ.angleA < π / 2 ∧ Δ.angleB < π / 2 ∧ Δ.angleC < π / 2

noncomputable def is_circumcenter (Δ : Triangle) (O : Point) : Prop :=
  ∀ X, X ∈ Δ.vertices ↔ dist O X = circumradius Δ

variables (ΔABC : Triangle) (ΔAMN : Triangle)
  [hABC : triangle_acute ΔABC]
  (hABgtAC : ΔABC.sideB.length > ΔABC.sideC.length)
  (hM_on_BC : M ∈ Line.through B C)
  (hN_on_BC : N ∈ Line.through B C)
  (h_angles : ∠A B M = ∠A C N)
  (hO1_circumcenter : is_circumcenter ΔABC O1)
  (hO2_circumcenter : is_circumcenter ΔAMN O2)

theorem collinear_O1_O2_A : Collinear O1 O2 A :=
sorry

end collinear_O1_O2_A_l18_18922


namespace cannot_tile_regular_pentagon_l18_18733

def internal_angle (n : ℕ) : ℝ :=
  180 - 360 / n

def can_tile (n : ℕ) : Prop :=
  360 % internal_angle n = 0

theorem cannot_tile_regular_pentagon : ¬ can_tile 5 :=
by
  let θ_5 := internal_angle 5
  have h : θ_5 = 108 := by sorry
  have h1 : 360 % 108 ≠ 0 := by sorry
  show ¬ can_tile 5 from by sorry

end cannot_tile_regular_pentagon_l18_18733


namespace students_present_l18_18676

theorem students_present (total_students : ℕ) (absent_percentage : ℝ) (h : total_students = 100) (hp : absent_percentage = 14) : 
  let present_percentage := 100 - absent_percentage in 
  let present_students := total_students * present_percentage / 100 in 
  present_students = 86 := 
by 
  sorry

end students_present_l18_18676


namespace angle_measure_l18_18620

theorem angle_measure (x : ℝ) (h1 : x + 3 * x^2 + 10 = 90) : x = 5 :=
by
  sorry

end angle_measure_l18_18620


namespace ball_bounce_below_5_l18_18009

theorem ball_bounce_below_5 (a : ℝ) (r : ℝ) (h_k : ℕ → ℝ) 
  (ha : a = 2000) (hr : r = 0.4) (h_hk : ∀ k, h_k k = a * r^k) : 
  ∃ k : ℕ, h_k k < 5 ∧ ∀ j : ℕ, j < k → h_k j ≥ 5 :=
by
  -- Define the initial height and bounce ratio
  let a := 2000
  let r := 0.4

  -- The height after the k-th bounce in terms of initial height and ratio
  def h_k (k : ℕ) : ℝ := a * r^k

  -- Find k such that h_k(k) < 5 for the first time and for all j < k, h_k(j) ≥ 5
  existsi 7
  split
  { -- Proof that h_k(7) < 5
    sorry },
  { -- Proof that for all j < 7, h_k(j) ≥ 5
    sorry }

end ball_bounce_below_5_l18_18009


namespace number_of_words_with_at_least_one_consonant_l18_18870

def letters := ['A', 'B', 'C', 'D', 'E', 'F']

def is_consonant (c : Char) : Prop := c ∈ ['B', 'C', 'D', 'F']

def is_vowel (c : Char) : Prop := c ∈ ['A', 'E']

def all_3_letter_words := letters.product letters.product letters

def words_with_no_consonants : List (Char × Char × Char) :=
  all_3_letter_words.filter (λ (w : Char × Char × Char), is_vowel w.1 ∧ is_vowel w.2 ∧ is_vowel w.3)

def words_with_at_least_one_consonant : List (Char × Char × Char) :=
  all_3_letter_words.filter (λ (w : Char × Char × Char), ¬ (is_vowel w.1 ∧ is_vowel w.2 ∧ is_vowel w.3))

theorem number_of_words_with_at_least_one_consonant :
  words_with_at_least_one_consonant.length = 208 :=
by
  have total_words := 6 * 6 * 6
  have words_with_no_consonants := 2 * 2 * 2
  have desired_result := total_words - words_with_no_consonants
  exact Nat.eq_of_succ_eq_succ rfl sorry

end number_of_words_with_at_least_one_consonant_l18_18870


namespace arithmetic_sequence_problem_l18_18462

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_problem 
  (a : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_diff_nonzero : ∃ d, d ≠ 0 ∧ (∀ n, a (n + 1) = a n + d))
  (h_cond : a 2 + a 3 = a 6) :
  (a 1 + a 2) / (a 3 + a 4 + a 5) = 1 / 3 :=
begin
  sorry
end

end arithmetic_sequence_problem_l18_18462


namespace area_AGE_is_correct_l18_18590

/-- Geometry Problem -/
noncomputable def area_of_triangle_AGE (A B C D E G : Point) : ℝ :=
  if h1: (A = (0,0)) ∧ (B = (5,0)) ∧ (C = (5,5)) ∧ (D = (0,5))
  if h2: (E = (5,2)) ∧ (∃ G, G = (x,y) ∧ on_circumcircle (A,B,E) G ∧ on_line (B,D) G)
  then area_of (triangle A G E)
  else 0

theorem area_AGE_is_correct : ∀ (A B C D E G : Point), 
  (A = (0, 0)) ∧ (B = (5, 0)) ∧ (C = (5, 5)) ∧ (D = (0, 5)) →
  (E = (5, 2)) →
  (∃ G, (G.on_circle_circumcircle (A, B, E) ∧ G.on_line (B, D)) ∧ calculated_area = 43.25) :=
by
  sorry

end area_AGE_is_correct_l18_18590


namespace sum_six_seven_l18_18127

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom arithmetic_sequence : ∀ (n : ℕ), a (n + 1) = a n + d
axiom sum_condition : a 2 + a 5 + a 8 + a 11 = 48

theorem sum_six_seven : a 6 + a 7 = 24 :=
by
  -- Using given axioms and properties of arithmetic sequence
  sorry

end sum_six_seven_l18_18127


namespace find_positive_value_of_A_l18_18209

variable (A : ℝ)

-- Given conditions
def relation (A B : ℝ) : ℝ := A^2 + B^2

-- The proof statement
theorem find_positive_value_of_A (h : relation A 7 = 200) : A = Real.sqrt 151 := sorry

end find_positive_value_of_A_l18_18209


namespace smallest_integer_solution_of_inequality_l18_18303

theorem smallest_integer_solution_of_inequality : ∃ x : ℤ, (3 * x ≥ x - 5) ∧ (∀ y : ℤ, 3 * y ≥ y - 5 → y ≥ -2) := 
sorry

end smallest_integer_solution_of_inequality_l18_18303


namespace min_value_sin_cos_l18_18435

open Real

theorem min_value_sin_cos (x : ℝ) : 
  ∃ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 = 2 / 3 ∧ ∀ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 ≥ 2 / 3 :=
sorry

end min_value_sin_cos_l18_18435


namespace lines_are_coplanar_when_k_is_minus_8_9_l18_18763

open Real

def line1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (1 + 2 * s, 4 - 3 * k * s, 5 + 2 * k * s)
def line2 (t : ℝ) : ℝ × ℝ × ℝ := (-1 + 3 * t / 2, 3 + 2 * t, 8 - 3 * t)

theorem lines_are_coplanar_when_k_is_minus_8_9 :
  (∃ (s t : ℝ), line1 s (-8 / 9) = line2 t) ↔
  (-8 / 9 ∈ {k : ℝ | ∃ (s t : ℝ), line1 s k = line2 t}) := sorry

end lines_are_coplanar_when_k_is_minus_8_9_l18_18763


namespace correct_slope_angle_statements_l18_18343

theorem correct_slope_angle_statements :
  (0 ≤ α ∧ α < Real.pi) ∧ (∀ α, 0 ≤ α ∧ α < Real.pi → ∃ unique_slope_angle α) :=
sorry

end correct_slope_angle_statements_l18_18343


namespace part1_part2_part3_l18_18144

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x + 2 / x

-- Part (1): Equation of the tangent line
theorem part1 :
  ∀ (a : ℝ), (a = 1) → ((∃ m b, ∀ x, f 1 x = m * x + b) → (∀ y, 2 * 1 + y - 3 = 0)) :=
by 
  sorry

-- Part (2): Value of a
theorem part2 :
  ∀ (a : ℝ), (∀ x, f a x ≤ 2 / x - 1) → (a = 1) :=
by
  sorry

-- Part (3): Product inequality
theorem part3 : 
  ∀ (n : ℕ), (2 ≤ n) → ((∏ k in finset.range n, (1 + 1 / (↑k ^ 2))) < Real.exp 1) :=
by
  sorry

end part1_part2_part3_l18_18144


namespace hannah_strawberries_l18_18865

-- Definitions for the conditions
def daily_harvest : ℕ := 5
def days_in_april : ℕ := 30
def strawberries_given_away : ℕ := 20
def strawberries_stolen : ℕ := 30

-- The statement we need to prove
theorem hannah_strawberries (harvested_strawberries : ℕ)
  (total_harvest := daily_harvest * days_in_april)
  (total_lost := strawberries_given_away + strawberries_stolen)
  (final_count := total_harvest - total_lost) :
  harvested_strawberries = final_count :=
sorry

end hannah_strawberries_l18_18865


namespace combination_98_96_eq_4753_l18_18351

theorem combination_98_96_eq_4753 : nat.choose 98 96 = 4753 := by
  sorry

end combination_98_96_eq_4753_l18_18351


namespace total_surface_area_exposed_l18_18086

-- Define the volumes of the cubes
def volumes : List ℕ := [1, 8, 27, 64, 125, 216, 343, 512]

-- Define the function to calculate side length from volume
noncomputable def side_length (v : ℕ) : ℕ := 
  (Real.toNat (Real.cbrt (Real.ofNat v)))

-- Problem conditions and required proof
theorem total_surface_area_exposed : 
  let cubes := List.map side_length volumes;
  let stacked_cubes := List.dropLast cubes;       -- The first seven
  let offset_cube := cubes.head!;
  let total_area := (5 * (offset_cube * offset_cube)) +
                    (4 * (List.sum (List.map (λ x => x * x) (List.drop 1 stacked_cubes)))) +
                    (6 * 4) + -- surface area of offset cube
                    6; -- surface area of the smallest cube (1 unit^3)
  total_area = 890 := sorry

end total_surface_area_exposed_l18_18086


namespace bottles_from_shop_c_correct_l18_18083

-- Definitions for the given conditions
def total_bottles := 550
def bottles_from_shop_a := 150
def bottles_from_shop_b := 180

-- Definition for the bottles from Shop C
def bottles_from_shop_c := total_bottles - (bottles_from_shop_a + bottles_from_shop_b)

-- The statement to prove
theorem bottles_from_shop_c_correct : bottles_from_shop_c = 220 :=
by
  -- proof will be filled later
  sorry

end bottles_from_shop_c_correct_l18_18083


namespace minimum_value_x_plus_y_l18_18827

theorem minimum_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y * (x - y)^2 = 1) : x + y ≥ 2 :=
sorry

end minimum_value_x_plus_y_l18_18827


namespace milan_total_minutes_l18_18237

-- Conditions
variables (x : ℝ) -- minutes on the second phone line
variables (minutes_first : ℝ := x + 20) -- minutes on the first phone line
def total_cost (x : ℝ) := 3 + 0.15 * (x + 20) + 4 + 0.10 * x

-- Statement to prove
theorem milan_total_minutes (x : ℝ) (h : total_cost x = 56) :
  x + (x + 20) = 252 :=
sorry

end milan_total_minutes_l18_18237


namespace reflection_squared_identity_l18_18565

noncomputable def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let (a, b) := v
  let normSq := a * a + b * b
  ! let scale := (2 / normSq)
  ! Matrix.of !₂ !₂ [
  [1 - scale * b * b, scale * b * a],
  [scale * a * b, 1 - scale * a * a]
] 

theorem reflection_squared_identity :
  let v := (2, -1)
  let R := reflection_matrix v
  R * R = 1 :=
  by
  let v := (2, -1)
  let R := reflection_matrix v
  sorry

end reflection_squared_identity_l18_18565


namespace theta_in_fourth_quadrant_l18_18514

/-- Given that cos θ > 0 and sin 2θ < 0, prove that the terminal side of θ lies in the fourth quadrant. -/
theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  (θ / π ∈ Icc 0 2 ∧ θ % (π/2) < π/2 ∧ θ % (π/2) < π) ∧
  (θ < (3/4) * π * 2 ∧ θ > (1/2) * π * 2) :=
sorry

end theta_in_fourth_quadrant_l18_18514


namespace extinction_prob_l18_18711

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the base case condition
def v_0 : ℝ := 1

-- Define the extinction probability function
def v : ℕ → ℝ
| 0 => v_0
| k => p * v (k + 1) + q * v (k - 1)

-- Define the target value for v_1
def v_1 : ℝ := 2 / 3

-- The main theorem: Prove that v 1 = 2 / 3 given the conditions
theorem extinction_prob : v 1 = v_1 := by
  -- Proof will be provided here
  sorry

end extinction_prob_l18_18711


namespace number_of_ways_to_choose_numbers_l18_18978

theorem number_of_ways_to_choose_numbers : 
    let S := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    (∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (a + b + c) % 2 = 0 ∧ a + b + c ≥ 10 ∧
      (∃ K:finset ℕ, K \ 3)) := 51 := 
sorry

end number_of_ways_to_choose_numbers_l18_18978


namespace garage_sale_total_l18_18317

theorem garage_sale_total (treadmill chest_of_drawers television total_sales : ℝ)
  (h1 : treadmill = 100) 
  (h2 : chest_of_drawers = treadmill / 2) 
  (h3 : television = treadmill * 3) 
  (partial_sales : ℝ) 
  (h4 : partial_sales = treadmill + chest_of_drawers + television) 
  (h5 : partial_sales = total_sales * 0.75) : 
  total_sales = 600 := 
by
  sorry

end garage_sale_total_l18_18317


namespace salary_increase_percentage_l18_18592

-- Definitions based on conditions
def total_employees : ℕ := 480
def travel_allowance_percentage : ℝ := 20 / 100
def employees_no_increase : ℕ := 336

-- Prove that the percentage of employees who got a salary increase is 10%
theorem salary_increase_percentage :
  let travel_allowance_increase := travel_allowance_percentage * total_employees
  let employees_with_increase := total_employees - employees_no_increase
  let salary_increase_only := employees_with_increase - travel_allowance_increase
  let S_percentage := (salary_increase_only / total_employees) * 100
  S_percentage = 10 := 
by 
  sorry

end salary_increase_percentage_l18_18592


namespace send_messages_ways_l18_18588

theorem send_messages_ways : (3^4 = 81) :=
by
  sorry

end send_messages_ways_l18_18588


namespace quadratic_inequality_solution_set_l18_18802

variable (a b c : ℝ) (α β : ℝ)

theorem quadratic_inequality_solution_set
  (hαβ : α < β)
  (hα_lt_0 : α < 0) 
  (hβ_lt_0 : β < 0)
  (h_sol_set : ∀ x : ℝ, a * x^2 + b * x + c < 0 ↔ (x < α ∨ x > β)) :
  (∀ x : ℝ, c * x^2 - b * x + a > 0 ↔ (-(1 / α) < x ∧ x < -(1 / β))) :=
  sorry

end quadratic_inequality_solution_set_l18_18802


namespace max_real_roots_among_polynomials_l18_18199

noncomputable def largest_total_real_roots (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : ℕ :=
  4  -- representing the largest total number of real roots

theorem max_real_roots_among_polynomials
  (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  largest_total_real_roots a b c h_a h_b h_c = 4 :=
sorry

end max_real_roots_among_polynomials_l18_18199


namespace sum_rational_roots_h_l18_18081

def h : ℚ[X] := X^3 - 6*X^2 + 11*X - 6

theorem sum_rational_roots_h : ∑ r in (roots h).filter (λ x, x.is_rational), r = 6 := 
sorry

end sum_rational_roots_h_l18_18081


namespace num_four_digit_numbers_l18_18159

theorem num_four_digit_numbers : 
  ∃ n : ℕ, 
    n = 8 ∧ 
    ∀ (d1 d2 d3 d4 : ℕ), 
    (d1, d2, d3, d4) ∈ {♯[(2, 0, 2, 5), (2, 2, 0, 5), (2, 5, 2, 0), (2, 5, 0, 2),
                         (5, 2, 0, 2), (0, 2, 5, 2), (0, 2, 2, 5), (5, 2, 2, 0)]} ∧ 
    (d2 = 2 ∨ d4 = 2) :=
sorry

end num_four_digit_numbers_l18_18159


namespace sum_of_2001_terms_l18_18121

noncomputable def sum_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(finset.range n).sum a

theorem sum_of_2001_terms:
  (∀ n ≥ 3, a n = a (n-1) - a (n-2)) ∧
  sum_seq a 1492 = 1985 ∧
  sum_seq a 1985 = 1492 →
  sum_seq a 2001 = 986 :=
by
  sorry

end sum_of_2001_terms_l18_18121


namespace no_integer_a_for_integer_roots_l18_18422

theorem no_integer_a_for_integer_roots :
  ∀ a : ℤ, ¬ (∃ x : ℤ, x^2 - 2023 * x + 2022 * a + 1 = 0) := 
by
  intro a
  rintro ⟨x, hx⟩
  sorry

end no_integer_a_for_integer_roots_l18_18422


namespace determine_a7_l18_18560

noncomputable def arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0     => a1
| (n+1) => a1 + n * d

noncomputable def sum_arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => a1 * (n + 1) + (n * (n + 1) * d) / 2

theorem determine_a7 (a1 d : ℤ) (a2 : a1 + d = 7) (S7 : sum_arithmetic_seq a1 d 7 = -7) : arithmetic_seq a1 d 7 = -13 :=
by
  sorry

end determine_a7_l18_18560


namespace gcd_of_repeated_three_digit_integers_is_1001001_l18_18033

theorem gcd_of_repeated_three_digit_integers_is_1001001 :
  ∀ (n : ℕ), (100 ≤ n ∧ n <= 999) →
  ∃ d : ℕ, d = 1001001 ∧
    (∀ m : ℕ, m = n * 1001001 →
      ∃ k : ℕ, m = k * d) :=
by
  sorry

end gcd_of_repeated_three_digit_integers_is_1001001_l18_18033


namespace range_of_f_l18_18135

theorem range_of_f (f : ℝ → ℝ) (h : ∀ y : ℝ, y ∈ set.range (λ x, f(x + 2011)) ↔ y ∈ set.Ioo (-1:ℝ) 1) : 
  ∀ y : ℝ, y ∈ set.range f ↔ y ∈ set.Ioo (-1:ℝ) 1 :=
by
  sorry

end range_of_f_l18_18135


namespace best_purchase_option_l18_18276

-- Define the prices and discount conditions for each store
def technik_city_price_before_discount : ℝ := 2000 + 4000
def technomarket_price_before_discount : ℝ := 1500 + 4800

def technik_city_discount : ℝ := technik_city_price_before_discount * 0.10
def technomarket_bonus : ℝ := technomarket_price_before_discount * 0.20

def technik_city_final_price : ℝ := technik_city_price_before_discount - technik_city_discount
def technomarket_final_price : ℝ := technomarket_price_before_discount

-- The theorem stating the ultimate proof problem
theorem best_purchase_option : technik_city_final_price < technomarket_final_price :=
by
  -- Replace 'sorry' with the actual proof if required
  sorry

end best_purchase_option_l18_18276


namespace route_inequality_l18_18029

noncomputable def f (m n : ℕ) : ℕ :=
-- Definition of f based on the problem context which could be obtained
-- Possibly using binomial coefficients or simple combinatorial properties

theorem route_inequality (m n : ℕ) : f(m, n) ≤ 2^(m * n) := sorry

end route_inequality_l18_18029


namespace percentage_of_knives_is_40_l18_18062

theorem percentage_of_knives_is_40 
  (initial_knives : ℕ) (initial_forks : ℕ) (initial_spoons : ℕ) 
  (traded_knives : ℕ) (traded_spoons : ℕ) : 
  initial_knives = 6 → 
  initial_forks = 12 → 
  initial_spoons = 3 * initial_knives → 
  traded_knives = 10 → 
  traded_spoons = 6 → 
  let final_knives := initial_knives + traded_knives in
  let final_spoons := initial_spoons - traded_spoons in
  let total_silverware := final_knives + final_spoons + initial_forks in
  (final_knives : ℝ) / total_silverware * 100 = 40 :=
by sorry

end percentage_of_knives_is_40_l18_18062


namespace necessary_but_not_sufficient_l18_18682

theorem necessary_but_not_sufficient (x : ℝ) : (x > 3) → (x > 1) ∧ ¬((x > 1) → (x > 3)) := by
  intro h
  split
  -- x > 3 implies x > 1
  exact lt_of_lt_of_le h (le_refl x)
  -- x > 1 does not imply x > 3
  intro h1
  exact h1 not_h

end necessary_but_not_sufficient_l18_18682


namespace sum_of_perfect_square_divisors_of_544_l18_18100

-- Conditions and Definitions
def is_divisor (a b : ℕ) : Prop := b % a = 0
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def sum_perfect_square_divisors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ d, is_divisor d n ∧ is_perfect_square d) (Finset.range (n + 1))), d

-- Proof to be conducted
theorem sum_of_perfect_square_divisors_of_544 : sum_perfect_square_divisors 544 = 21 :=
by sorry

end sum_of_perfect_square_divisors_of_544_l18_18100


namespace instantaneous_speed_at_t1_is_4_l18_18623

-- Define the distance function
def distance_function (t : ℝ) : ℝ := 2 * t^3 - t^2 + 2

-- Define the speed function as the derivative of the distance function
def speed_function (t : ℝ) : ℝ := deriv distance_function t

-- Problem statement: Prove that the instantaneous speed of the car at t = 1 is 4
theorem instantaneous_speed_at_t1_is_4 : speed_function 1 = 4 :=
by {
  sorry
}

end instantaneous_speed_at_t1_is_4_l18_18623


namespace length_of_JK_l18_18653

theorem length_of_JK
  (GH HI KL JK : ℝ)
  (similar : ∃ a b c d e f : ℝ, a / b = c / d = e / f)
  (GH_eq : GH = 8)
  (HI_eq : HI = 16)
  (KL_eq : KL = 32)
  (HI_twice_JK : HI = 2 * JK) :
  JK = 16 :=
by {
  -- Proof omitted
  sorry
}

end length_of_JK_l18_18653


namespace constant_term_position_l18_18273

-- Define the necessary constants and the general term formula
variable (a : ℝ) -- Assume 'a' is a real number
noncomputable def general_term (r : ℕ) : ℝ :=
  (-2)^r * (Nat.choose 30 r : ℝ) *  a^((90 - 5 * r) / 6)

-- The theorem we need to prove: The constant term is at position 19
theorem constant_term_position :
  ∃ r : ℕ, (90 - 5 * r) / 6 = 0 ∧ r + 1 = 19 :=
by
  use 18
  simp    -- Use simplification to show (90 - 5 * 18) / 6 = 0 and 18 + 1 = 19
  sorry   -- We put sorry to skip the proof, since it is not required

end constant_term_position_l18_18273


namespace david_more_pushups_than_zachary_l18_18070

-- Definitions based on conditions
def david_pushups : ℕ := 37
def zachary_pushups : ℕ := 7

-- Theorem statement proving the answer
theorem david_more_pushups_than_zachary : david_pushups - zachary_pushups = 30 := by
  sorry

end david_more_pushups_than_zachary_l18_18070


namespace find_tan_beta_l18_18470

theorem find_tan_beta
  (α β : ℝ) 
  (hα_quadrant : π/2 < α ∧ α < π) 
  (hα_sin : sin α = 3 / sqrt 10) 
  (h_tan_sum : tan (α + β) = -2) : 
  tan β = 1 / 7 :=
sorry

end find_tan_beta_l18_18470


namespace domain_of_sqrt_function_l18_18624

noncomputable def domain_of_function : Set ℝ :=
  {x : ℝ | 3 - 2 * x - x^2 ≥ 0}

theorem domain_of_sqrt_function : domain_of_function = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end domain_of_sqrt_function_l18_18624


namespace ratio_time_B_to_A_l18_18015

-- Definitions for the given conditions
def T_A : ℕ := 10
def work_rate_A : ℚ := 1 / T_A
def combined_work_rate : ℚ := 0.3

-- Lean 4 statement for the problem
theorem ratio_time_B_to_A (T_B : ℚ) (h : (work_rate_A + 1 / T_B) = combined_work_rate) :
  (T_B / T_A) = (1 / 2) := by
  sorry

end ratio_time_B_to_A_l18_18015


namespace sqrt_ineq_l18_18708

theorem sqrt_ineq (x : ℝ) (hx : 0 < x) : (sqrt x > 3 * x) ↔ (0 < x ∧ x < 1 / 9) :=
by sorry -- skipping the proof

end sqrt_ineq_l18_18708


namespace min_colors_required_l18_18456

theorem min_colors_required (m : ℕ) (n : ℕ) (grid : matrix (fin m) (fin m) (fin n)) :
  (∀ (c : fin n), ∀ (i : fin m), ∀ (j : fin m), 
    grid i j = c →
    ∀ (a b c d : fin m), 
    a < b ∧ b < c ∧ c < d ∧ 
    grid a j = c ∧ grid b j = c ∧ grid c j = c ∧ grid d j = c →
    ∀ (k : fin m) (r : fin m),
    k < a → grid k r ≠ c → 
    r > d → grid k r ≠ c 
  ) →
  n ≥ 506 :=
by
  sorry

end min_colors_required_l18_18456


namespace part1_part2_l18_18498

variables {R : Type} [LinearOrderedField R]

def setA := {x : R | -1 < x ∧ x ≤ 5}
def setB (m : R) := {x : R | x^2 - 2*x - m < 0}
def complementB (m : R) := {x : R | x ≤ -1 ∨ x ≥ 3}

theorem part1 : 
  {x : R | 6 / (x + 1) ≥ 1} = setA := 
by 
  sorry

theorem part2 (m : R) (hm : m = 3) : 
  setA ∩ complementB m = {x : R | 3 ≤ x ∧ x ≤ 5} := 
by 
  sorry

end part1_part2_l18_18498


namespace incorrect_median_description_l18_18386

/-- Define the data set and corresponding variance calculation -/
def data_set := [8, 6, 5, 6, 10]

noncomputable def mean (xs : List ℕ) : ℝ :=
  (List.sum xs) / (List.length xs)

noncomputable def variance (xs : List ℕ) : ℝ :=
  let μ := mean xs
  (List.sum $ List.map (λ x => (x - μ) ^ 2) xs) / (List.length xs)

theorem incorrect_median_description : ∀ xs, xs = data_set → ¬(median xs = 5) := by
  intros
  sorry

end incorrect_median_description_l18_18386


namespace line_passes_through_fixed_point_range_of_k_no_second_quadrant_min_area_triangle_l18_18852

-- Problem 1: The line passes through a fixed point
theorem line_passes_through_fixed_point (k : ℝ) : ∃ P : ℝ × ℝ, P = (1, -2) ∧ (∀ x y, k * x - y - 2 - k = 0 → P = (x, y)) :=
by
  sorry

-- Problem 2: Range of values for k if the line does not pass through the second quadrant
theorem range_of_k_no_second_quadrant (k : ℝ) : ¬ (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ k * x - y - 2 - k = 0) → k ∈ Set.Ici (0) :=
by
  sorry

-- Problem 3: Minimum area of triangle AOB
theorem min_area_triangle (k : ℝ) :
  let A := (2 + k) / k
  let B := -2 - k
  (∀ x y, k * x - y - 2 - k = 0 ↔ (x = A ∧ y = 0) ∨ (x = 0 ∧ y = B)) →
  ∃ S : ℝ, S = 4 ∧ (∀ x y : ℝ, (k = 2 ∧ k * x - y - 4 = 0) → S = 4) :=
by
  sorry

end line_passes_through_fixed_point_range_of_k_no_second_quadrant_min_area_triangle_l18_18852


namespace extinction_probability_l18_18715

-- Definitions from conditions
def prob_divide : ℝ := 0.6
def prob_die : ℝ := 0.4

-- Statement of the theorem
theorem extinction_probability :
  ∃ (v : ℝ), v = 2 / 3 :=
by
  sorry

end extinction_probability_l18_18715


namespace find_arithmetic_sequence_n_l18_18923

open Real

noncomputable def arithmetic_sequence_n_value (a_n S : ℕ → ℕ) (n : ℕ) :=
  (S 9 = 18) ∧ (S n = 240) ∧ (a_n (n - 4) = 30) → n = 15

theorem find_arithmetic_sequence_n (a_n S : ℕ → ℕ) (n : ℕ) :
  arithmetic_sequence_n_value a_n S n :=
by
  intros h
  cases h with h1 h2
  cases h2 with h2 h3
  sorry

end find_arithmetic_sequence_n_l18_18923


namespace triangle_XYZ_is_isosceles_l18_18373

-- Definitions and conditions
variables (A B C D E F X Y Z : Type) [Circle A B C D E F]

variable h1 : A = B
variable h2 : C = E
variables (HX : (AC : Line) ∩ (BE : Diagonal) = X)
variables (HY : (BE : Diagonal) ∩ (DF : Diagonal) = Y)
variables (HZ : (BF : Diagonal) ∩ (AE : Diagonal) = Z)

-- Theorem statement
theorem triangle_XYZ_is_isosceles
    (h1 : AB = BD)
    (h2 : CE = EF)
    (HX : af X)
    (HY : bg Y)
    (HZ : ch Z) :
  is_isosceles_triangle X Y Z :=
sorry

end triangle_XYZ_is_isosceles_l18_18373


namespace complex_number_real_l18_18477

theorem complex_number_real (m : ℝ) :
  let z := m^2 * (1 + complex.I) - m * (m + complex.I)
  (z.im = 0) → (m = 0 ∨ m = 1) :=
by
  intro h
  sorry

end complex_number_real_l18_18477


namespace people_on_bus_l18_18743

theorem people_on_bus (initial_people : ℕ) (additional_people : ℕ) (total_people : ℕ) 
  (h1 : initial_people = 4) (h2 : additional_people = 13) : total_people = 17 :=
by
  rw [h1, h2]
  sorry -- Proof goes here

end people_on_bus_l18_18743


namespace inequality_solution_l18_18604

theorem inequality_solution (x : ℝ) (hx : x ≠ -7) :
  (x^2 - 49) / (x + 7) < 0 ↔ x ∈ set.Ioo (-∞) (-7) ∪ set.Ioo (-7) 7 := by
  sorry

end inequality_solution_l18_18604


namespace correct_distance_AB_correct_product_AF_BF_l18_18828

def trajectory_condition (x y : ℝ) : Prop :=
  sqrt((x - 1)^2 + y^2) = abs x + 1

def line_l1 (x y : ℝ) : Prop :=
  y = x + 1

def line_l2 (x y : ℝ) : Prop :=
  ∃ t : ℝ, x = 1 + 3 * t ∧ y = sqrt(3) * t

def intersection_point_l1_trajectory : ℝ × ℝ :=
  (1, 2) -- intersection point

def intersection_points_l2_trajectory (A B : ℝ × ℝ) : Prop :=
  A.1 = 12 + 2 * sqrt(3) ∧ A.2 = 4 ∧ B.1 = 2 - 2 * sqrt(3) ∧ B.2 = -4

def distance_AB (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem correct_distance_AB (A B : ℝ × ℝ) (hA : trajectory_condition A.1 A.2) (hB : trajectory_condition B.1 B.2) :
  distance_AB A B = 16 := sorry

def distance_AF (A : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - 1)^2 + A.2^2)

def distance_BF (B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - 1)^2 + B.2^2)

theorem correct_product_AF_BF (A B : ℝ × ℝ) (hA : trajectory_condition A.1 A.2) (hB : trajectory_condition B.1 B.2) :
  distance_AF A * distance_BF B = 16 := sorry

end correct_distance_AB_correct_product_AF_BF_l18_18828


namespace nathan_total_earnings_l18_18084

variable (x : ℝ) -- Nathan's hourly wage
variable (hours_week2 hours_week3 : ℝ) -- Hours worked in the second and third weeks
variable (extra_earnings_week3 : ℝ) -- Extra earnings in the third week compared to the second week

-- Given conditions
def conditions :=
  hours_week2 = 12 ∧
  hours_week3 = 18 ∧
  extra_earnings_week3 = 36 ∧
  (hours_week3 - hours_week2) * x = extra_earnings_week3

-- Prove the total earnings for the two weeks
noncomputable def total_earnings :=
  hours_week2 * x + hours_week3 * x

-- The statement to be proved
theorem nathan_total_earnings : conditions →
  total_earnings = 180 :=
by
  sorry

end nathan_total_earnings_l18_18084


namespace tessellation_coloring_l18_18660

-- Definitions matching the problem conditions
structure Tessellation where
  rectangles : ℕ -- Number of rectangles
  circles : ℕ    -- Number of circles
  overlaps : ∀ c : ℕ, c < circles → Fin 4 → Fin rectangles -- Each circle overlaps 4 unique rectangles

-- Proposition that states the requirement and expectation
def least_colors_needed (t : Tessellation) : ℕ :=
  3  -- As we know from the solution steps, exactly 3 colors are essential

-- The proof statement
theorem tessellation_coloring (t : Tessellation)
  (H1 : ∀ r1 r2 : Fin t.rectangles, (r1 ≠ r2 → (∃ c : Fin t.circles, overlaps t c <| 0 = r1) → (overlaps t c <| 2 ≠ r2)) )
  (H2 : ∀ c : Fin t.circles, ∃ rects : Vector (Fin t.rectangles) 4, ∀ i j : Fin 4, i ≠ j → overlaps t c i ≠ overlaps t c j) :
  least_colors_needed t = 3 :=
by
  sorry

end tessellation_coloring_l18_18660


namespace nth_term_of_sequence_l18_18815

-- Definitions for the sequence
def sequence (a b u0 : ℝ) : ℕ → ℝ
| 0       := u0
| (n + 1) := a * (sequence n) + b

theorem nth_term_of_sequence (a b u0 : ℝ) (n : ℕ) :
  sequence a b u0 n =
    if a = 1 then u0 + n * b
    else a^n * u0 + b * (1 - a^(n + 1)) / (1 - a) :=
by
  sorry

end nth_term_of_sequence_l18_18815


namespace range_of_a_l18_18857

theorem range_of_a (a : ℝ) :
  (∃ (x : ℝ), 
    ((2 * x + 5) / 3 > x - 5) ∧ 
    ((x + 3) / 2 < x + a) ∧ 
    x ∈ {15, 16, 17, 18, 19}) 
  → (-6 < a ∧ a ≤ (-11 / 2)) :=
by
  sorry

end range_of_a_l18_18857


namespace triangle_side_QR_eq_6_l18_18191

theorem triangle_side_QR_eq_6 
  (P Q R : Type) 
  (cos : ℝ → ℝ) 
  (sin : ℝ → ℝ) 
  (sqrt : ℝ → ℝ)
  (PQ QR PR : ℝ) 
  (P_deg : ℝ) 
  (Q_deg : ℝ)
  (R_deg : ℝ)
  (cos2P_minus_Q_plus_sinP_plus_Q_eq_sqrt2 : cos(2 * P_deg - Q_deg) + sin(P_deg + Q_deg) = sqrt 2)
  (PQ_eq_6 : PQ = 6)
  (P_is_45 : P_deg = 45) :
  QR = 6 :=
by sorry

end triangle_side_QR_eq_6_l18_18191


namespace gretchen_total_money_l18_18861

def charge_per_drawing : ℕ := 20
def sold_saturday : ℕ := 24
def sold_sunday : ℕ := 16
def total_caricatures := sold_saturday + sold_sunday

theorem gretchen_total_money : charge_per_drawing * total_caricatures = 800 := by
  have total_caricatures_eq : total_caricatures = 40 := by
    unfold total_caricatures
    simp
  rw [total_caricatures_eq]
  calc
    20 * 40 = 800 := by norm_num

end gretchen_total_money_l18_18861


namespace exists_distinct_positive_integers_l18_18004

theorem exists_distinct_positive_integers (n : ℕ) (h : 0 < n) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

end exists_distinct_positive_integers_l18_18004


namespace triangle_inequality_l18_18594

theorem triangle_inequality (a b c p S r : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b)
  (hp : p = (a + b + c) / 2)
  (hS : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (hr : r = S / p):
  1 / (p - a) ^ 2 + 1 / (p - b) ^ 2 + 1 / (p - c) ^ 2 ≥ 1 / r ^ 2 :=
sorry

end triangle_inequality_l18_18594


namespace actual_cost_of_article_l18_18670

theorem actual_cost_of_article (x : ℝ) (hx : 0.76 * x = 988) : x = 1300 :=
sorry

end actual_cost_of_article_l18_18670


namespace residue_neg_1237_mod_29_l18_18077

theorem residue_neg_1237_mod_29 : ∃ (r : ℤ), 0 ≤ r ∧ r < 29 ∧ (-1237 ≡ r [MOD 29]) ∧ r = 10 :=
by
  sorry

end residue_neg_1237_mod_29_l18_18077


namespace extinction_probability_l18_18716

-- Definitions from conditions
def prob_divide : ℝ := 0.6
def prob_die : ℝ := 0.4

-- Statement of the theorem
theorem extinction_probability :
  ∃ (v : ℝ), v = 2 / 3 :=
by
  sorry

end extinction_probability_l18_18716


namespace modular_inverse_of_17_mod_800_l18_18329

    theorem modular_inverse_of_17_mod_800 :
      ∃ x : ℤ, 0 ≤ x ∧ x < 800 ∧ (17 * x) % 800 = 1 :=
    by
      use 47
      sorry
    
end modular_inverse_of_17_mod_800_l18_18329


namespace least_possible_value_l18_18230

-- Define a set containing 7 distinct prime numbers
def is_set_of_7_distinct_primes (q : Set ℕ) : Prop :=
  q.card = 7 ∧ ∀ n ∈ q, Nat.Prime n

-- Define a function that checks if the sum of elements in the set is even
def sum_is_even (q : Set ℕ) : Prop :=
  (q.toFinset.sum id) % 2 = 0

-- Define a function that checks if a given number is the least member
def is_least_member (x : ℕ) (q : Set ℕ) : Prop :=
  x ∈ q ∧ ∀ y ∈ q, x ≤ y

-- The formal statement of our problem
theorem least_possible_value (q : Set ℕ) (x : ℕ) :
  is_set_of_7_distinct_primes q → sum_is_even q → x ∈ q → is_least_member 2 q :=
by
  sorry

end least_possible_value_l18_18230


namespace friendly_function_f_0_g_is_friendly_function_f_fixpoint_l18_18458

def friendly_function (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ f x) ∧
  (f 1 = 1) ∧
  (∀ x1 x2, 0 ≤ x1 → 0 ≤ x2 → x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2)

theorem friendly_function_f_0  (f : ℝ → ℝ) (h : friendly_function f) : f 0 = 0 :=
  sorry

theorem g_is_friendly_function : friendly_function (λ x, 2^x - 1) :=
  sorry

theorem f_fixpoint (f : ℝ → ℝ) (h : friendly_function f) (x0 : ℝ) (hx0_1 : 0 ≤ x0) (hx0_2 : x0 ≤ 1) (hx0_3 : 0 ≤ f x0) (hx0_4 : f x0 ≤ 1) (hx0_5 : f (f x0) = x0) : f x0 = x0 :=
  sorry

end friendly_function_f_0_g_is_friendly_function_f_fixpoint_l18_18458


namespace probability_circle_contains_O_l18_18226

theorem probability_circle_contains_O
  (A B C O : Point) (height_ABC : ℝ) 
  (is_equilateral : equilateral_triangle A B C) 
  (is_center : centroid A B C O) 
  (height_eq : height_ABC = 13) 
  (X : Point)
  (inside_triangle : is_inside_triangle X A B C)
  (circle_center_X : ∀ (X : Point), circle X 1 inside_triangle):
  probability (circle_contains_point X 1 O) = (real.sqrt(3) * real.pi) / 121 := 
sorry

end probability_circle_contains_O_l18_18226


namespace coordinates_2023rd_point_l18_18539

def point_sequence : ℕ → ℤ × ℤ
| 0 => (0, 0)
| n + 1 => let (x, y) := point_sequence n
           (x + 1, [0, 1, 0, -1][(x+1) % 4])

theorem coordinates_2023rd_point : point_sequence 2022 = (2022, 0) :=
sorry

end coordinates_2023rd_point_l18_18539


namespace coordinates_of_C_l18_18031

theorem coordinates_of_C (A B : ℝ × ℝ) (C : ℝ × ℝ) 
    (hA : A = (1, 3)) (hB : B = (9, -3)) (hBC_AB : dist B C = 1/2 * dist A B) : 
    C = (13, -6) :=
sorry

end coordinates_of_C_l18_18031


namespace only_prime_solution_l18_18093

def is_prime (n : ℕ) : Prop := nat.prime n

theorem only_prime_solution :
  ∃ (p : ℕ), is_prime p ∧ is_prime (p + 4) ∧ is_prime (p + 8) ∧ p = 3 :=
by
  sorry

end only_prime_solution_l18_18093


namespace ouav_ouav_base10_correct_l18_18383

def sound_to_digit (s : Char) : Int :=
  match s with
  | 'о' => 0
  | 'у' => 1
  | 'в' => 2
  | 'а' => 3
  | _ => -1 -- An invalid character

def base4_to_base10 (s : String) : Int :=
  s.toList.reverse.enum.foldl (λ acc (c, i) => acc + sound_to_digit c * Int.pow 4 i) 0

theorem ouav_ouav_base10_correct : base4_to_base10 "оуавоуав" = 7710 := by
  sorry

end ouav_ouav_base10_correct_l18_18383


namespace find_b_l18_18524

theorem find_b (a b c : ℝ) (h1 : a = 2) (h2 : b + c = 7) (h3 : real.cos B = -1/4) : b = 4 :=
sorry

end find_b_l18_18524


namespace K_positive_for_any_x_K_negative_for_any_x_l18_18941

-- Definitions based on the problem
def K (m x : ℝ) : ℝ :=
  ((m^2 + 2 * m - 24) * x^2 - 6 * (m + 6) * x + (m^2 + 10 * m + 24)) / 
  ((m^2 - 9 * m + 18) * x^2 - 8 * (m - 6) * x + (m^2 - 3 * m - 18))

-- Theorem statement
theorem K_positive_for_any_x (m : ℝ) (x : ℝ) : (0 < |m| ∧ |m| > 6) → (K m x > 0) :=
by
  sorry

theorem K_negative_for_any_x (m : ℝ) (x : ℝ) : (0 < |m| ∧ |m| < 6) → (K m x < 0) :=
by
  sorry

end K_positive_for_any_x_K_negative_for_any_x_l18_18941


namespace ratio_of_fake_purses_to_total_purses_l18_18315

theorem ratio_of_fake_purses_to_total_purses :
  ∀ (total_purses total_handbags : ℕ) (fake_handbags_fraction authentic_items : ℚ),
  total_purses = 26 ∧ total_handbags = 24 ∧ 
  fake_handbags_fraction = 1 / 4 ∧ 
  authentic_items = 31 →
  (let fake_handbags := fake_handbags_fraction * total_handbags in
   let total_items := total_purses + total_handbags in
   let fake_items := total_items - authentic_items in
   let fake_purses := fake_items - fake_handbags in
   fake_purses / total_purses = 1 / 2) :=
by
  intros _ _ _ _
  rintros ⟨hpurses, hhandbags, hfake_frac, hauthentic⟩
  let fake_handbags := hfake_frac * hhandbags
  let total_items := hpurses + hhandbags
  let fake_items := total_items - hauthentic
  let fake_purses := fake_items - fake_handbags
  have : fake_purses = 13 := sorry  -- provided by the solution step
  have : total_purses = 26 := hpurses
  have ratio := (fake_purses : ℚ) / total_purses
  show ratio = 1 / 2, by
    sorry

end ratio_of_fake_purses_to_total_purses_l18_18315


namespace exists_pretty_hexagon_max_area_pretty_hexagon_l18_18020

-- Define the condition of a "pretty" hexagon
structure PrettyHexagon (L ℓ h : ℝ) : Prop :=
  (diag1 : (L + ℓ)^2 + h^2 = 1)
  (diag2 : (L + ℓ)^2 + h^2 = 1)
  (diag3 : (L + ℓ)^2 + h^2 = 1)
  (diag4 : (L + ℓ)^2 + h^2 = 1)
  (L_pos : L > 0) (L_lt_1 : L < 1)
  (ℓ_pos : ℓ > 0) (ℓ_lt_1 : ℓ < 1)
  (h_pos : h > 0) (h_lt_1 : h < 1)

-- Area of the hexagon given L, ℓ, and h
def hexagon_area (L ℓ h : ℝ) := 2 * (L + ℓ) * h

-- Question (a): Existence of a pretty hexagon with a given area
theorem exists_pretty_hexagon (k : ℝ) (hk : 0 < k ∧ k < 1) : 
  ∃ L ℓ h : ℝ, PrettyHexagon L ℓ h ∧ hexagon_area L ℓ h = k :=
sorry

-- Question (b): Maximum area of any pretty hexagon is at most 1
theorem max_area_pretty_hexagon : 
  ∀ L ℓ h : ℝ, PrettyHexagon L ℓ h → hexagon_area L ℓ h ≤ 1 :=
sorry

end exists_pretty_hexagon_max_area_pretty_hexagon_l18_18020


namespace point_E_on_line_BD_l18_18117

-- Lean 4 statement based on the provided problem and conditions
theorem point_E_on_line_BD 
    (A B C D E F : Point)
    (hConvex: ConvexQuadrilateral A B C D)
    (hTangentsE: AreCommonExternalTangents [Circle A B C, Circle B C D] E)
    (hTangentsF: AreCommonExternalTangents [Circle A B D, Circle B C D] F)
    (hCollinear: Collinear [A, C, F]) :
  LiesOnLine E [B, D] :=
  sorry

end point_E_on_line_BD_l18_18117


namespace valid_knight_arrangements_l18_18268

-- Define a chessboard, represented as an 8x8 grid
def chessboard := fin 8 × fin 8

-- Assume a checkerboard coloring
def is_red_square (pos : fin 8 × fin 8) : Prop :=
  (pos.fst.val + pos.snd.val) % 2 = 0

def is_blue_square (pos : fin 8 × fin 8) : Prop :=
  ¬ is_red_square pos

-- Define knight placement condition
def valid_knight_placement (k1 k2 : fin 8 × fin 8) : Prop :=
  is_red_square k1 ∧ is_blue_square k2 ∨ is_blue_square k1 ∧ is_red_square k2

-- Define the problem statement
theorem valid_knight_arrangements : 
  ∃ (k1 k2 : chessboard), valid_knight_placement k1 k2 ∧ (32 * 32 = 1024) :=
by sorry

end valid_knight_arrangements_l18_18268


namespace p_and_q_necessary_but_not_sufficient_l18_18002

theorem p_and_q_necessary_but_not_sufficient (p q : Prop) :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := 
by 
  sorry

end p_and_q_necessary_but_not_sufficient_l18_18002


namespace local_minimum_at_2_l18_18486

noncomputable def f (x c : ℝ) := x * (x - c) ^ 2

theorem local_minimum_at_2 (c : ℝ) : 
  (∃ x, f x c = x * (x - c) ^ 2 ∧ f'(2) = 0 ∧ ∀ y, f y c ≥ f 2 c) ↔ c = 2 :=
by
  sorry

end local_minimum_at_2_l18_18486


namespace no_zero_in_interval_3_7_l18_18286

def my_function (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 10 then 3 * 2^x - 24
  else if 10 < x ∧ x ≤ 20 then 126 - 2^(x - 5)
  else 0 -- undefined outside the interval [0, 20]

theorem no_zero_in_interval_3_7 : ∀ x, 3 < x ∧ x < 7 → my_function x ≠ 0 := by
  -- proof placeholder
  sorry

end no_zero_in_interval_3_7_l18_18286


namespace volume_ratio_of_spheres_l18_18637

theorem volume_ratio_of_spheres (r1 r2 r3 : ℝ) 
  (h : r1 / r2 = 1 / 2 ∧ r2 / r3 = 2 / 3) : 
  (4/3 * π * r3^3) = 3 * (4/3 * π * r1^3 + 4/3 * π * r2^3) :=
by
  sorry

end volume_ratio_of_spheres_l18_18637


namespace sequence_sum_formula_l18_18148

def sequence_term (n : ℕ) : ℕ := 2^(n-1) + 1

theorem sequence_sum_formula (n : ℕ) :
  (∑ k in Finset.range(n+1), (sequence_term (k+1)) * Nat.choose n k) = 3^n + 2^n :=
by
  sorry

end sequence_sum_formula_l18_18148


namespace num_pos_integers_congruent_to_4_mod_7_l18_18881

theorem num_pos_integers_congruent_to_4_mod_7 (n : ℕ) (h1 : n < 500) (h2 : ∃ k : ℕ, n = 7 * k + 4) : 
  ∃ total : ℕ, total = 71 :=
sorry

end num_pos_integers_congruent_to_4_mod_7_l18_18881


namespace increasing_or_decreasing_subseq_l18_18561

theorem increasing_or_decreasing_subseq (a : Fin (m * n + 1) → ℝ) :
  ∃ (s : Fin (m + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (s i) ≤ a (s j)) ∨
  ∃ (t : Fin (n + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (t i) ≥ a (t j)) :=
sorry

end increasing_or_decreasing_subseq_l18_18561


namespace find_z_l18_18797

noncomputable def vector_u (z : ℚ) : ℚ × ℚ × ℚ :=
(2, 4, z)

def vector_v : ℚ × ℚ × ℚ :=
(1, -2, 3)

def dot_product (a b : ℚ × ℚ × ℚ) : ℚ :=
a.1 * b.1 + a.2 * b.2 + a.3 * b.3

theorem find_z (z : ℚ) (h : vector_u z = (2, 4, z)) (proj_eq : ∃ w : ℚ, w * vector_v = (5 / 14) * vector_v) :
  z = 11 / 3 :=
by
  sorry

end find_z_l18_18797


namespace proof_problem_l18_18184

-- Define the polar equation of C1
def C1_polar_eq (ρ θ a : ℝ) : Prop :=
  ρ * Math.sin (θ + Real.pi / 4) = Real.sqrt 2 / 2 * a

-- Define the Cartesian equation of C1
def C1_cartesian_eq (x y a : ℝ) : Prop :=
  x + y - a = 0

-- Define the parametric equations of C2
def C2_parametric_eqs (x y θ : ℝ) : Prop :=
  x = -1 + Real.cos θ ∧ y = -1 + Real.sin θ

-- Define the Cartesian equation of C2 
def C2_cartesian_eq (x y : ℝ) : Prop :=
  (x + 1) ^ 2 + (y + 1) ^ 2 = 1

-- Define the distance d from the circle center to the line
def distance_from_center_to_line (a : ℝ) : ℝ :=
  abs (-1 - 1 - a) / Real.sqrt 2

-- Prove the main statement
theorem proof_problem (a : ℝ) :
  (∀ ρ θ, C1_polar_eq ρ θ a → ∃ x y, C1_cartesian_eq x y a) ∧ 
  (∀ x y, C2_cartesian_eq x y → ∀ θ, C2_parametric_eqs x y θ) →
  (2 * Real.sqrt 2 > abs (a + 2)) :=
by
  sorry -- Proof omitted

end proof_problem_l18_18184


namespace construction_mayhem_l18_18765

theorem construction_mayhem (n k : ℕ) (h1 : ∀ n, start n = 1 + 2 * (n - 1)) (h2 : ∀ n, duration n = 2 + n) (h3 : k = 26) : 
  ∃ d, (d ≥ 51) ∧ (∑ i in range (k-9, k+1), 2 ≤ 3 * i - 1 ≤ d - 1 + i) :=
begin
  sorry
end

end construction_mayhem_l18_18765


namespace volume_of_intersection_l18_18445

def cube_volume (a : ℝ) : ℝ := a^3

def common_volume (a : ℝ) : ℝ := cube_volume a / 4

theorem volume_of_intersection (a : ℝ) :
  let cube1_volume := cube_volume a in
  let cube2_volume := cube_volume a in
  let intersection_volume := common_volume a in
  intersection_volume = a^3 / 4 :=
by
  -- Assuming the given conditions and definitions, then we have
  -- the required conclusion.
  sorry

end volume_of_intersection_l18_18445


namespace count_congruent_to_4_mod_7_l18_18879

theorem count_congruent_to_4_mod_7 : 
  ∃ (n : ℕ), 
  n = 71 ∧ 
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 70 → ∃ m : ℕ, m = 4 + 7 * k ∧ m < 500 := 
by
  sorry

end count_congruent_to_4_mod_7_l18_18879


namespace expression_value_l18_18338

theorem expression_value :
  let x := (3 + 1 : ℚ)⁻¹ * 2
  let y := x⁻¹ * 2
  let z := y⁻¹ * 2
  z = (1 / 2 : ℚ) :=
by
  sorry

end expression_value_l18_18338


namespace locus_of_point_P_is_circle_AD_l18_18240

open EuclideanGeometry

noncomputable def midpoint (p1 p2 : Point) : Point := sorry
noncomputable def perpendicular (line1 line2 : Line) : Prop := sorry
noncomputable def foot_of_perpendicular (line : Line) (p : Point) : Point := sorry

theorem locus_of_point_P_is_circle_AD
  (circle : Circle)
  (A B : Point)
  (hA : A ∈ circle)
  (hB : B ∈ circle)
  (M : Point)
  (hM : M ∈ circle)
  (K : Point := midpoint M B)
  (P : Point := foot_of_perpendicular (line A M) K)
  (C : Point := diametrically_opposite A circle)
  (D : Point := midpoint B C) :
  is_locus_of_points { P | P = foot_of_perpendicular (line A M) (midpoint M B) } (circle_with_diameter A D) :=
sorry

end locus_of_point_P_is_circle_AD_l18_18240


namespace magnitude_sum_l18_18475

open real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

-- Conditions from the problem
def condition1 := a + b + c = 0
def condition2 := ⟪a - b, c⟫ = 0
def condition3 := ⟪a, b⟫ = 0
def condition4 := ‖a‖ = 1

-- Proof statement
theorem magnitude_sum (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) (h4 : condition4 a) :
  ‖a‖^2 + ‖b‖^2 + ‖c‖^2 = 4 :=
by sorry

end magnitude_sum_l18_18475


namespace judy_pencil_cost_l18_18936

theorem judy_pencil_cost :
  (∀ (pencils_per_week days_per_week pencils_per_pack cost_per_pack total_days), 
    (pencils_per_week = 10 → days_per_week = 5 → pencils_per_pack = 30 → cost_per_pack = 4 → total_days = 45 → 
    let pencils_per_day := pencils_per_week / days_per_week in
    let total_pencils_needed := pencils_per_day * total_days in
    let packs_needed := total_pencils_needed / pencils_per_pack in
    let total_cost := packs_needed * cost_per_pack in
    total_cost = 12)) :=
by 
  intros pencils_per_week days_per_week pencils_per_pack cost_per_pack total_days hw hd hp hc ht
  calc 
    let pencils_per_day := pencils_per_week / days_per_week in
    let total_pencils_needed := pencils_per_day * total_days in
    let packs_needed := total_pencils_needed / pencils_per_pack in
    let total_cost := packs_needed * cost_per_pack in
    total_cost = 12 : sorry

end judy_pencil_cost_l18_18936


namespace log_eq_exp_solution_l18_18304

theorem log_eq_exp_solution :
  ∀ x : ℝ, log 5 (3^x + 4^x) = log 4 (5^x - 3^x) → x = 2 :=
by
  sorry

end log_eq_exp_solution_l18_18304


namespace prove_area_S_l18_18949

noncomputable def floor_real (t : ℝ) : ℝ := Real.floor t

noncomputable def S' (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - (t - floor_real t))^2 + (p.2 - 1)^2 ≤ (t - floor_real t + 0.5)^2}

def area_S' (t : ℝ) : ℝ :=
  π * (t - floor_real t + 0.5)^2

theorem prove_area_S' (t : ℝ) (ht : t ≥ 0) : 0.25 * π ≤ area_S' t ∧ area_S' t ≤ 2.25 * π :=
by
  sorry

end prove_area_S_l18_18949


namespace binomial_expansion_third_term_l18_18666

--- Define the parameters for the binomial expression
variables (a x : ℝ)

--- Define the binomial expression
def binomial_term (k: ℕ) (n: ℕ) (x: ℝ) (y: ℝ) : ℝ := 
  (nat.choose n k) * (x ^ (n - k)) * (y ^ k)

--- The specific parameters for our problem
def x_param := 2 * a / (real.sqrt x)
def y_param := -(real.sqrt x) / (3 * a ^ 2)
def n_val := 5

--- Statement of the theorem we want to prove:
theorem binomial_expansion_third_term :
  binomial_term 2 n_val x_param y_param = -80 / (9 * a * real.sqrt x) :=
by sorry

end binomial_expansion_third_term_l18_18666


namespace sum_1_over_2008_array_l18_18416

def sum_of_fractional_series (p : ℕ) : ℚ :=
  let series1 := ∑' r, (1 / (3 * p)^r : ℚ)
  let series2 := ∑' c, (1 / p^c : ℚ)
  series1 * series2

theorem sum_1_over_2008_array :
  let p := 2008
  let sum := sum_of_fractional_series p
  let frac := (3 * p^2) / ((3 * p - 1) * (p - 1))
  let m := 3 * p^2
  let n := (3 * p - 1) * (p - 1)
  sum = frac ∧ (m + n) % 2009 = 1 :=
by
  let p := 2008
  let sum := sum_of_fractional_series p
  let frac := (3 * p^2) / ((3 * p - 1) * (p - 1))
  have h_sum_eq : sum = frac := sorry
  let m := 3 * p^2
  let n := (3 * p - 1) * (p - 1)
  have h_mod_eq : (m + n) % 2009 = 1 := sorry
  exact ⟨h_sum_eq, h_mod_eq⟩

end sum_1_over_2008_array_l18_18416


namespace T_number_square_l18_18223

theorem T_number_square (a b : ℤ) : ∃ c d : ℤ, (a^2 + a * b + b^2)^2 = c^2 + c * d + d^2 := by
  sorry

end T_number_square_l18_18223


namespace total_watermelons_l18_18038

theorem total_watermelons 
  (A B C : ℕ) 
  (h1 : A + B = C - 6) 
  (h2 : B + C = A + 16) 
  (h3 : C + A = B + 8) :
  A + B + C = 18 :=
by
  sorry

end total_watermelons_l18_18038


namespace eccentricity_of_hyperbola_l18_18989

theorem eccentricity_of_hyperbola (a b : ℝ) (A B F1 F2 : ℝ × ℝ)
  (h_hyperbola : a > 0 ∧ b > 0)
  (h_symmetric : A = (-fst B, -snd B))
  (h_ptolemy : dist A B * dist F1 F2 = dist A F1 * dist B F2 + dist A F2 * dist B F1)
  (h_angle : angle A F1 F2 = π / 6) :
  (fun c => c / a = sqrt 3 + 1) :=
sorry

end eccentricity_of_hyperbola_l18_18989


namespace derivative_extreme_value_condition_l18_18640

noncomputable def f (x : ℝ) : ℝ := x^3

theorem derivative_extreme_value_condition :
  (∀ x : ℝ, deriv f x = 0 → 
    (∀ x : ℝ, has_deriv_at f x 0 → (∃ c, ∀ y, f y ≥ f c ∨ f y ≤ f c)) :=
    sorry

end derivative_extreme_value_condition_l18_18640


namespace number_of_zeros_l18_18635

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 2 else Real.log x - 1

theorem number_of_zeros : ∃! (a : ℝ) (b : ℝ), f a = 0 ∧ f b = 0 ∧ a ≠ b :=
by
  sorry

end number_of_zeros_l18_18635


namespace compute_one_plus_i_power_four_l18_18759

theorem compute_one_plus_i_power_four (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 :=
by
  sorry

end compute_one_plus_i_power_four_l18_18759


namespace find_positive_value_of_A_l18_18211

variable (A : ℝ)

-- Given conditions
def relation (A B : ℝ) : ℝ := A^2 + B^2

-- The proof statement
theorem find_positive_value_of_A (h : relation A 7 = 200) : A = Real.sqrt 151 := sorry

end find_positive_value_of_A_l18_18211


namespace derek_dogs_count_l18_18775

theorem derek_dogs_count
  (initial_dogs : ℕ)
  (initial_cars : ℕ)
  (cars_after_10_years : ℕ)
  (dogs_after_10_years : ℕ)
  (h1 : initial_dogs = 90)
  (h2 : initial_dogs = 3 * initial_cars)
  (h3 : cars_after_10_years = initial_cars + 210)
  (h4 : cars_after_10_years = 2 * dogs_after_10_years) :
  dogs_after_10_years = 120 :=
by
  sorry

end derek_dogs_count_l18_18775


namespace solve_for_A_l18_18208

def hash (A B : ℝ) : ℝ := A^2 + B^2

theorem solve_for_A (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 :=
by
  sorry

end solve_for_A_l18_18208


namespace total_letters_sent_l18_18255

def letters_January : ℕ := 6
def letters_February : ℕ := 9
def letters_March : ℕ := 3 * letters_January

theorem total_letters_sent : letters_January + letters_February + letters_March = 33 :=
by
  -- This is where the proof would go
  sorry

end total_letters_sent_l18_18255


namespace radius_of_inner_circle_l18_18907

theorem radius_of_inner_circle (R a x : ℝ) (hR : 0 < R) (ha : 0 ≤ a) (haR : a < R) :
  (a ≠ R ∧ a ≠ 0) → x = (R^2 - a^2) / (2 * R) :=
by
  sorry

end radius_of_inner_circle_l18_18907


namespace fraction_subtraction_l18_18798

theorem fraction_subtraction :
  (12 / 30) - (1 / 7) = 9 / 35 :=
by sorry

end fraction_subtraction_l18_18798


namespace trigonometric_expression_value_l18_18511

noncomputable def trigonometric_identity (x : ℝ) : Prop :=
  sin x - 2 * cos x = 0

theorem trigonometric_expression_value (x : ℝ) (h : trigonometric_identity x) :
  (cos (π / 2 + x) * sin (-π - x)) / (cos (11 * π / 2 - x) * sin (9 * π / 2 + x)) = 2 :=
by
  sorry

end trigonometric_expression_value_l18_18511


namespace prob_correct_l18_18644

def distances : List (Nat × Nat × Nat) := [
  (0, 1, 6500), (0, 2, 6700), (0, 3, 6100), (0, 4, 8600),
  (1, 2, 11800), (1, 3, 6200), (1, 4, 7800),
  (2, 3, 7300), (2, 4, 4900),
  (3, 4, 3500)
]

def city_pairs := Finset.univ.sublistsLen 2

noncomputable def distance_less_than_8000_count : ℕ :=
  distances.countp (λ ⟨_, _, d⟩ => d < 8000)

def total_pairs_count : ℕ :=
  city_pairs.card

def probability : ℚ :=
  distance_less_than_8000_count / total_pairs_count

theorem prob_correct : probability = 3 / 5 := by
  sorry

end prob_correct_l18_18644


namespace area_of_quadrilateral_PQRS_l18_18415

theorem area_of_quadrilateral_PQRS :
  ∃ (P Q R S : Type) [metric_space P] [metric_space Q] [metric_space R] [metric_space S],
  (dist P Q = 9) ∧ (dist Q R = 6) ∧ (dist R S = 5) ∧ (dist S P = 17) ∧
  (angle P Q S = 90) ∧ (angle Q R S = 90) ∧
  (area_of_convex_quadrilateral P Q R S = 64.5) :=
sorry

end area_of_quadrilateral_PQRS_l18_18415


namespace total_letters_sent_l18_18254

def letters_January : ℕ := 6
def letters_February : ℕ := 9
def letters_March : ℕ := 3 * letters_January

theorem total_letters_sent : letters_January + letters_February + letters_March = 33 :=
by
  -- This is where the proof would go
  sorry

end total_letters_sent_l18_18254


namespace product_of_sequence_positive_l18_18530

theorem product_of_sequence_positive
  (n : ℕ) (a : Fin n → ℤ)
  (h : n = 39)
  (h_nonzero : ∀ i, a i ≠ 0)
  (h_adj_sum_pos : ∀ i (h_in : i < n - 1), a i + a ⟨i + 1, Nat.succ_lt_succ_iff.mpr h_in⟩ > 0)
  (h_total_sum_neg : ∑ i, a i < 0) :
  0 < ∏ i, a i := 
begin
  sorry
end

end product_of_sequence_positive_l18_18530


namespace subtracted_value_from_numbers_l18_18992

theorem subtracted_value_from_numbers (A B C D E X : ℝ) 
  (h1 : (A + B + C + D + E) / 5 = 5)
  (h2 : ((A - X) + (B - X) + (C - X) + (D - X) + E) / 5 = 3.4) :
  X = 2 :=
by
  sorry

end subtracted_value_from_numbers_l18_18992


namespace sum_of_powers_is_perfect_square_l18_18258

theorem sum_of_powers_is_perfect_square
  (k : ℕ)
  (n : ℕ)
  (roots_of_eq : ∀ (x₁ x₂ x₃ : ℝ), x₁^3 + n * x₁ + k = 0 ∧ 
                                    x₂^3 + n * x₂ + k = 0 ∧ 
                                    x₃^3 + n * x₃ + k = 0) :
  ∃ (a : ℕ), (a^2 = 49 * n^2 ∧ |x₁^7 + x₂^7 + x₃^7| = a^2) :=
by
  sorry


end sum_of_powers_is_perfect_square_l18_18258


namespace find_positive_integer_M_l18_18662

theorem find_positive_integer_M : 
  ∃ M : ℕ, 12^2 * 30^2 = 15^2 * M^2 ∧ M = 24 :=
begin
  use 24,
  split,
  { -- Prove that 12^2 * 30^2 = 15^2 * 24^2
    sorry },
  { -- Prove that M = 24 is the positive integer
    refl }
end

end find_positive_integer_M_l18_18662


namespace find_g_values_l18_18021

-- Definitions of the conditions
def condition1 (g : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, g(x + 3) - g(x) = 6 * x + 9

def condition2 (g : ℤ → ℤ) : Prop :=
  ∀ x : ℤ, g(x^2 - 4) = (g(x) - x)^2 + x^2 - 8

-- Statement of the proof problem
theorem find_g_values (g : ℤ → ℤ) (h1 : condition1 g) (h2 : condition2 g) : g 0 = 0 ∧ g 1 = 1 :=
  sorry

end find_g_values_l18_18021


namespace businesses_can_apply_l18_18746

-- Define conditions
def total_businesses : ℕ := 72
def businesses_fired : ℕ := 36 -- Half of total businesses (72 / 2)
def businesses_quit : ℕ := 24 -- One third of total businesses (72 / 3)

-- Theorem: Number of businesses Brandon can still apply to
theorem businesses_can_apply : (total_businesses - (businesses_fired + businesses_quit)) = 12 := 
by
  sorry

end businesses_can_apply_l18_18746


namespace f_even_l18_18893

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_identically_zero : ∃ x : ℝ, f x ≠ 0

axiom f_functional_eqn : ∀ a b : ℝ, 
  f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_even (x : ℝ) : f (-x) = f x :=
  sorry

end f_even_l18_18893


namespace problem_statement_l18_18003

variables {P A B C D : Type} -- Declare types for the points
variables (inside_square : P → Prop) -- Declare that P is a point inside the square
variables (angle : P → P → P → ℝ) -- Declare an angle function

-- Define the condition that P is inside the square ABCD
axiom P_is_inside_square : inside_square P

-- Define the given condition that angles are equal
axiom angle_condition : angle P A B = angle P C B

-- The main proposition we need to prove
theorem problem_statement : ∀ (P A B C D : Type) 
  (inside_square : P → Prop) 
  (angle : P → P → P → ℝ),
  P_is_inside_square → angle_condition → angle P B A = angle P D A :=
by sorry  -- Skip the proof

end problem_statement_l18_18003


namespace calculate_order_cost_l18_18238

-- Defining the variables and given conditions
variables (C E S D W : ℝ)

-- Given conditions as assumptions
axiom h1 : (2 / 5) * C = E * S
axiom h2 : (1 / 4) * (3 / 5) * C = D * W

-- Theorem statement for the amount paid for the orders
theorem calculate_order_cost (C E S D W : ℝ) (h1 : (2 / 5) * C = E * S) (h2 : (1 / 4) * (3 / 5) * C = D * W) : 
  (9 / 20) * C = C - ((2 / 5) * C + (3 / 20) * C) :=
sorry

end calculate_order_cost_l18_18238


namespace number_of_incorrect_statements_l18_18395

theorem number_of_incorrect_statements :
  let statement1 := ∀ (P Q : Prop), (P → Q) → (¬Q → ¬P)
  let statement2 := ¬(∀ x : ℝ, x^2 - x ≤ 0) ↔ ∃ x : ℝ, x^2 - x > 0
  let statement3 := ¬(∀ (Q : Type) [field Q], 
    ∀ (a b c d : Q), (a^2 + b^2 = c^2 + d^2 ↔ a = c ∧ b = d))
  let statement4 := ∀ x : ℝ, x ≠ 3 → |x| ≠ 3 
  let incorrect_statements :=
    ¬statement2 ∧ ¬statement3 ∧ ¬statement4
  @true →
  3 = (if ¬statement1 then 1 else 0) + 
      (if statement2 then 1 else 0) + 
      (if statement3 then 1 else 0) + 
      (if statement4 then 1 else 0)
:= sorry

end number_of_incorrect_statements_l18_18395


namespace inequality_A_only_inequality_B_not_always_l18_18975

theorem inequality_A_only (a b c : ℝ) (h1 : 2 * b > c) (h2 : c > a) (h3 : c > b) :
  a < c / 3 := 
sorry

theorem inequality_B_not_always (a b c : ℝ) (h1 : 2 * b > c) (h2 : c > a) (h3 : c > b) :
  ¬ (b < c / 3) := 
sorry

end inequality_A_only_inequality_B_not_always_l18_18975


namespace domain_of_log_function_l18_18829

theorem domain_of_log_function {f : ℝ → ℝ} (h : ∀ x, -1 ≤ f (2^x) ∧ f (2^x) ≤ 1) :
  ∀ x : ℝ, (1/2 : ℝ) ≤ f (log x) ∧ f (log x) ≤ 2 := 
sorry

end domain_of_log_function_l18_18829


namespace S_10_eq_210_l18_18950

-- Define the function S
def S (n : ℕ) : ℕ :=
  ∑ i in Finset.range (2 * n + 1), (nat.floor ∘ nat.sqrt) (n^2 + i)

-- Define n for the specific problem
def n : ℕ := 10

-- State the theorem
theorem S_10_eq_210 : S 10 = 210 :=
by
  -- Skipping the proof
  sorry

end S_10_eq_210_l18_18950


namespace f_neg2_eq_neg4_l18_18956

noncomputable def f (x : ℝ) : ℝ :=
  if hx : x >= 0 then 3^x - 2*x - 1
  else - (3^(-x) - 2*(-x) - 1)

theorem f_neg2_eq_neg4
: f (-2) = -4 :=
by
  sorry

end f_neg2_eq_neg4_l18_18956


namespace exists_indices_l18_18817

theorem exists_indices (a : ℕ → ℕ) 
  (h_seq_perm : ∀ n, ∃ m, a m = n) : 
  ∃ ℓ m, 1 < ℓ ∧ ℓ < m ∧ a 1 + a m = 2 * a ℓ :=
by
  sorry

end exists_indices_l18_18817


namespace probability_divisible_by_256_l18_18517

theorem probability_divisible_by_256 (n : ℕ) (h : 1 ≤ n ∧ n ≤ 1000) :
  ((n * (n + 1) * (n + 2)) % 256 = 0) → (∃ p : ℚ, p = 0.006 ∧ (∃ k : ℕ, k ≤ 1000 ∧ (n = k))) :=
sorry

end probability_divisible_by_256_l18_18517


namespace proportional_segments_l18_18735

def proportional (a b c d : ℕ) : Prop :=
  a * d = b * c

theorem proportional_segments : 
  ((proportional 2 3 4 5 = false) ∧
  (proportional 1 3 5 10 = false) ∧
  (proportional 2 3 4 6 = true) ∧
  (proportional 3 4 5 6 = false)) :=
by {
  sorry,
}

end proportional_segments_l18_18735


namespace sum_of_digits_of_N_eq_14_l18_18035

theorem sum_of_digits_of_N_eq_14 :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ (N % 10 + N / 10 % 10 = 14) :=
by
  sorry

end sum_of_digits_of_N_eq_14_l18_18035


namespace probability_AB_in_A_l18_18915

-- Definition of combination (C(n, k))
noncomputable def combination (n k : ℕ) : ℕ :=
nat.choose n k

-- The main statement
theorem probability_AB_in_A :
  let n := combination 4 2 * combination 2 2,
      m := combination 2 2 * combination 2 2 in
  m / n = 1 / 6 :=
by
  sorry

end probability_AB_in_A_l18_18915


namespace factorizable_using_complete_square_formula_l18_18138

def polynomial1 (x y : ℝ) := x^2 + y + y^2
def polynomial2 (x y : ℝ) := -x^2 + 2xy - y^2
def polynomial3 (x y : ℝ) := x^2 + 6xy - 9y^2
def polynomial4 (x : ℝ) := x^2 - x + (1/4)

theorem factorizable_using_complete_square_formula :
  (∃ a b : ℝ, polynomial2 a b = -(a - b)^2) ∧ (∃ a : ℝ, polynomial4 a = (a - 1/2)^2) :=
by
  sorry

end factorizable_using_complete_square_formula_l18_18138


namespace equivalent_form_l18_18894

theorem equivalent_form (p q : ℝ) (hp₁ : p ≠ 0) (hp₂ : p ≠ 5) (hq₁ : q ≠ 0) (hq₂ : q ≠ 7) :
  (3/p + 4/q = 1/3) ↔ (p = 9*q/(q - 12)) :=
by
  sorry

end equivalent_form_l18_18894


namespace necessary_not_sufficient_condition_l18_18822

-- Definitions of the vectors and conditions
noncomputable def a1 : ℝ × ℝ := (√3 / 2, 1 / 2)
def is_unit_vector (v : ℝ × ℝ) : Prop := v.1 ^ 2 + v.2 ^ 2 = 1
def a2 : ℝ × ℝ := (x, y)

-- Main theorem statement
theorem necessary_not_sufficient_condition (a1 : ℝ × ℝ) (a2 : ℝ × ℝ) 
  (h1 : is_unit_vector a1) (h2 : is_unit_vector a2) : 
  a1 = (√3 / 2, 1 / 2) → (a1 + a2 = (√3, 1) → a1 ≠ a2 ∧ ∀ (a1 : ℝ × ℝ), (is_unit_vector a1 → ¬ (requirement a1 a2))) := 
sorry


end necessary_not_sufficient_condition_l18_18822


namespace positive_value_of_A_l18_18212

theorem positive_value_of_A (A : ℝ) :
  (A ^ 2 + 7 ^ 2 = 200) → A = Real.sqrt 151 :=
by
  intros h
  sorry

end positive_value_of_A_l18_18212


namespace sequence_b_l18_18072

theorem sequence_b (b : ℕ → ℝ) (h₁ : b 1 = 1)
  (h₂ : ∀ n : ℕ, n ≥ 1 → (b (n + 1)) ^ 4 = 64 * (b n) ^ 4) :
  b 50 = 2 ^ 49 := by
  sorry

end sequence_b_l18_18072


namespace total_amount_paid_l18_18312

-- Define the conditions
def each_company_ad_spaces : ℕ := 10
def ad_space_length : ℝ := 12
def ad_space_width : ℝ := 5
def cost_per_square_foot : ℝ := 60

-- Area of one ad space
def area_of_one_ad_space : ℝ := ad_space_length * ad_space_width

-- Total area purchased by one company
def total_area_one_company : ℝ := area_of_one_ad_space * each_company_ad_spaces

-- Cost for one company
def cost_for_one_company : ℝ := total_area_one_company * cost_per_square_foot

-- Total cost for all three companies
def total_cost_three_companies : ℝ := cost_for_one_company * 3

-- Proof statement
theorem total_amount_paid (each_company_ad_spaces = 10) (ad_space_length = 12) (ad_space_width = 5) (cost_per_square_foot = 60):
  total_cost_three_companies = 108000 := sorry

end total_amount_paid_l18_18312


namespace initial_investment_B_l18_18008
-- Import necessary Lean library

-- Define the necessary conditions and theorems
theorem initial_investment_B (x : ℝ) (profit_A : ℝ) (profit_total : ℝ)
  (initial_A : ℝ) (initial_A_after_8_months : ℝ) (profit_B : ℝ) 
  (initial_A_months : ℕ) (initial_A_after_8_months_months : ℕ) 
  (initial_B_months : ℕ) (initial_B_after_8_months_months : ℕ) : 
  initial_A = 3000 ∧ initial_A_after_8_months = 2000 ∧
  profit_A = 240 ∧ profit_total = 630 ∧ 
  profit_B = profit_total - profit_A ∧
  (initial_A * initial_A_months + initial_A_after_8_months * initial_A_after_8_months_months) /
  ((initial_B_months * x + initial_B_after_8_months_months * (x + 1000))) = 
  profit_A / profit_B →
  x = 4000 :=
by
  sorry

end initial_investment_B_l18_18008


namespace gretchen_total_earnings_l18_18864

-- Define the conditions
def price_per_drawing : ℝ := 20.0
def caricatures_sold_saturday : ℕ := 24
def caricatures_sold_sunday : ℕ := 16

-- The total caricatures sold
def total_caricatures_sold : ℕ := caricatures_sold_saturday + caricatures_sold_sunday

-- The total amount of money made
def total_money_made : ℝ := total_caricatures_sold * price_per_drawing

-- The theorem to be proven
theorem gretchen_total_earnings : total_money_made = 800.0 := by
  sorry

end gretchen_total_earnings_l18_18864


namespace min_value_sin_cos_l18_18434

open Real

theorem min_value_sin_cos (x : ℝ) : 
  ∃ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 = 2 / 3 ∧ ∀ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 ≥ 2 / 3 :=
sorry

end min_value_sin_cos_l18_18434


namespace sum_of_possible_values_CDF_l18_18705

theorem sum_of_possible_values_CDF 
  (C D F : ℕ) 
  (hC: 0 ≤ C ∧ C ≤ 9)
  (hD: 0 ≤ D ∧ D ≤ 9)
  (hF: 0 ≤ F ∧ F ≤ 9)
  (hdiv: (C + 4 + 9 + 8 + D + F + 4) % 9 = 0) :
  C + D + F = 2 ∨ C + D + F = 11 → (2 + 11 = 13) :=
by sorry

end sum_of_possible_values_CDF_l18_18705


namespace propositions_correct_l18_18482

-- Given an acute-angled triangle and a geometric sequence
variable {A B C : ℝ} -- Angles in the triangle
variable {q : ℝ} -- Common ratio of the geometric sequence
variable {a_n : ℕ → ℝ} -- Geometric sequence

-- Condition 1: All angles are acute
abbreviation is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

-- Condition 2: Sine and cosine properties in acute-angled triangles
def acute_angles (A B C : ℝ) : Prop := is_acute_angle A ∧ is_acute_angle B ∧ is_acute_angle C

-- Condition 3: Sum of sines is greater than the sum of cosines in acute-angled triangles
def sum_sines_greater_cosines (A B C : ℝ) : Prop :=
  sin A + sin B + sin C > cos A + cos B + cos C

-- Condition 4: Increasing geometric sequence
def is_geom_seq (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n * q

def is_increasing_seq (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a_n n < a_n (n + 1)

-- The actual theorem statement combining the conditions and the proof problem
theorem propositions_correct (h_acute : acute_angles A B C) 
                            (h_geom : is_geom_seq a_n q)
                            (h_incr : is_increasing_seq a_n) :
  sum_sines_greater_cosines A B C ∧ (q > 1 → is_increasing_seq a_n) :=
by
  sorry

end propositions_correct_l18_18482


namespace percentage_of_knives_is_40_l18_18064

theorem percentage_of_knives_is_40 
  (initial_knives : ℕ) (initial_forks : ℕ) (initial_spoons : ℕ) 
  (traded_knives : ℕ) (traded_spoons : ℕ) : 
  initial_knives = 6 → 
  initial_forks = 12 → 
  initial_spoons = 3 * initial_knives → 
  traded_knives = 10 → 
  traded_spoons = 6 → 
  let final_knives := initial_knives + traded_knives in
  let final_spoons := initial_spoons - traded_spoons in
  let total_silverware := final_knives + final_spoons + initial_forks in
  (final_knives : ℝ) / total_silverware * 100 = 40 :=
by sorry

end percentage_of_knives_is_40_l18_18064


namespace savings_percentage_correct_l18_18391

-- Define the individual expenses and savings
def rent := 5000
def milk := 1500
def groceries := 4500
def education := 2500
def petrol := 2000
def miscellaneous := 5650
def savings := 2350

-- Define the total expenses, total salary, and percentage saved
def total_expenses := rent + milk + groceries + education + petrol + miscellaneous
def total_salary := total_expenses + savings
def percentage_saved := (savings / total_salary.toFloat) * 100

-- Statement to prove
theorem savings_percentage_correct:
  percentage_saved ≈ 8.87 := by
  -- Placeholder for the proof
  sorry

end savings_percentage_correct_l18_18391


namespace imaginary_part_of_complex_l18_18097

open Complex

theorem imaginary_part_of_complex : 
  (∃ (i : ℂ), i = Complex.I ∧ (im (i^2 / (2 * i - 1)) = 2 / 5)) :=
by
  use Complex.I,
  split,
  rfl,
  sorry

end imaginary_part_of_complex_l18_18097


namespace median_incenter_division_eq_l18_18952

theorem median_incenter_division_eq (a b c : ℝ)  
  (h₁ : a ≤ c) (h₂ : ¬(a = 0 ∨ b = 0 ∨ c = 0)) :
  (∃ B C A : ℝ, AB = c ∧ BC = a ∧ CA = b ∧ 
    (∃ I : ℝ, 
    let BM := (1 / 2) * BC ^ 2 + (1 / 2) * AB ^ 2 - (1 / 4) * CA ^ 2,
    let BS := (a + c - b) / 2,
    let MT := (c - a) / 2 in 
    (BM ^ 2 = 9 * x ^ 2 ∧ BS ^ 2 = 2 * x ^ 2 ∧ MT = x))) ↔
  (a / 5 = b / 10 ∧ b / 10 = c / 13) := sorry

end median_incenter_division_eq_l18_18952


namespace average_movers_per_hour_l18_18543

-- Define the main problem parameters
def total_people : ℕ := 3200
def days : ℕ := 4
def hours_per_day : ℕ := 24
def total_hours : ℕ := hours_per_day * days
def average_people_per_hour := total_people / total_hours

-- State the theorem to prove
theorem average_movers_per_hour :
  average_people_per_hour = 33 :=
by
  -- Proof is omitted
  sorry

end average_movers_per_hour_l18_18543


namespace best_scrap_year_limit_l18_18997

theorem best_scrap_year_limit
    (purchase_cost : ℝ)
    (annual_expenses : ℝ)
    (base_maintenance_cost : ℝ)
    (annual_maintenance_increase : ℝ)
    (n : ℕ)
    (n_min_avg : ℝ) :
    purchase_cost = 150000 ∧
    annual_expenses = 15000 ∧
    base_maintenance_cost = 3000 ∧
    annual_maintenance_increase = 3000 ∧
    n = 10 →
    n_min_avg = 10 := by
  sorry

end best_scrap_year_limit_l18_18997


namespace least_difference_l18_18596

noncomputable def geometric_seq (a r n : ℕ) := a * r^n

noncomputable def arithmetic_seq (a d n : ℕ) := a + d * n

theorem least_difference :
  let C := {a ∈ finset.range 5 | let term := geometric_seq 3 3 a; term ≤ 200 → term}
  let D := {b ∈ finset.range 10 | let term := arithmetic_seq 10 20 b; term ≤ 200 → term}
  ∃ (x ∈ C) (y ∈ D), |x - y| = 9 :=
by
  let C := {term : ℕ | ∃ n, term = geometric_seq 3 3 n ∧ term ≤ 200 ∧ n < 5}
  let D := {term : ℕ | ∃ n, term = arithmetic_seq 10 20 n ∧ term ≤ 200 ∧ n < 10}
  have hC : C = {3, 9, 27, 81} := by sorry
  have hD : D = {10, 30, 50, 70, 90, 110, 130, 150, 170, 190} := by sorry
  use 81
  use 90
  split
  all_goals {sorry}

end least_difference_l18_18596


namespace probability_drawing_balls_l18_18688

theorem probability_drawing_balls :
  let total_balls := 15
  let red_balls := 10
  let blue_balls := 5
  let drawn_balls := 4
  let num_ways_to_draw_4_balls := Nat.choose total_balls drawn_balls
  let num_ways_to_draw_3_red_1_blue := (Nat.choose red_balls 3) * (Nat.choose blue_balls 1)
  let num_ways_to_draw_1_red_3_blue := (Nat.choose red_balls 1) * (Nat.choose blue_balls 3)
  let total_favorable_outcomes := num_ways_to_draw_3_red_1_blue + num_ways_to_draw_1_red_3_blue
  let probability := total_favorable_outcomes / num_ways_to_draw_4_balls
  probability = (140 : ℚ) / 273 :=
sorry

end probability_drawing_balls_l18_18688


namespace square_area_l18_18384

theorem square_area (side length width x y : ℝ) (h1 : side length = 5 + 2 * x)
    (h2 : x = 7.5) (h3 : y ≠ 0) (h4 : ∀ (i : ℕ), i < 5 -> (2 * x * y) = (3 * y * width)) :
    (side length)^2 = 400 :=
by
  sorry

end square_area_l18_18384


namespace intervals_of_increase_and_count_zeros_l18_18489

noncomputable def f (x ω : ℝ) : ℝ := 2 * sin (ω * x) * cos (ω * x) - sqrt 3 + 2 * sqrt 3 * sin (ω * x)^2

theorem intervals_of_increase_and_count_zeros (omega : ℝ) (hω : omega > 0):
  (intervals_of_increase f omega = set.interval_Union (λ k : ℤ, set.Icc (k * π - π / 12) (k * π + 5 * π / 12))) ∧
  (count_zeros g (λ x, 2 * sin (1 * x) + 1) (0, 20) = 40) :=
sorry

end intervals_of_increase_and_count_zeros_l18_18489


namespace zero_x_intersections_l18_18890

theorem zero_x_intersections 
  (a b c : ℝ) 
  (h_geom_seq : b^2 = a * c) 
  (h_ac_pos : a * c > 0) : 
  ∀ x : ℝ, ¬(ax^2 + bx + c = 0) := 
by 
  sorry

end zero_x_intersections_l18_18890


namespace find_a_l18_18683

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * log x

theorem find_a (a : ℝ) (h : (a * (1 + log 1) = 3)) : 
  a = 3 :=
by 
  have h1: log 1 = 0 := by sorry
  simpa [h1] using h
  sorry

end find_a_l18_18683


namespace num_terms_arithmetic_seq_l18_18161

theorem num_terms_arithmetic_seq :
  ∀ (a₁ aₙ d : ℤ), a₁ = 165 ∧ d = -6 ∧ aₙ = 21 → (∃ n : ℕ, aₙ = a₁ + (n - 1) * d ∧ n = 24) :=
by
  intros a₁ aₙ d h
  cases h with h₁ h_rest
  cases h_rest with h₂ h₃
  use 24
  split
  { 
    -- simplified equation: aₙ = a₁ + (24 - 1) * d
    calc
      aₙ = 21          : by rw [h₃]
      ... = 165 + (24 - 1) * (-6) : by sorry
  }
  { 
    rfl
  }

end num_terms_arithmetic_seq_l18_18161


namespace extinction_prob_one_l18_18721

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the extinction probability function
def extinction_prob (v : ℕ → ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else p * v (k + 1) + q * v (k - 1)

-- State the theorem
theorem extinction_prob_one (v : ℕ → ℝ) :
  extinction_prob v 1 = 2 / 3 :=
sorry

end extinction_prob_one_l18_18721


namespace sum_of_n_plus_k_l18_18283

theorem sum_of_n_plus_k (n k : ℕ) (h1 : 2 * (n - k) = 3 * (k + 1)) (h2 : 3 * (n - k - 1) = 4 * (k + 2)) : n + k = 47 := by
  sorry

end sum_of_n_plus_k_l18_18283


namespace part1_part2_part3_l18_18219

noncomputable def f (a x : ℝ) : ℝ := Real.log (x + 1) + a * (x^2 - x)

theorem part1 (x : ℝ) (hx : 0 < x) : f 0 x < x := by sorry

theorem part2 (a x : ℝ) :
  (0 ≤ a ∧ a ≤ 8/9 → 0 = 0) ∧
  (a > 8/9 → 2 = 2) ∧
  (a < 0 → 1 = 1) := by sorry

theorem part3 (a : ℝ) (h : ∀ x > 0, f a x ≥ 0) : 0 ≤ a ∧ a ≤ 1 := by sorry

end part1_part2_part3_l18_18219


namespace elephant_walk_distance_l18_18764

theorem elephant_walk_distance (r_small r_large : ℝ) (h1 : r_small = 15) (h2 : r_large = 30):
  let quarter_arc_smaller := (1/4) * 2 * Real.pi * r_small,
      radial_path := r_large - r_small,
      third_arc_larger := (1/3) * 2 * Real.pi * r_large,
      total_distance := 2 * quarter_arc_smaller + 2 * radial_path + third_arc_larger
  in total_distance = 35 * Real.pi + 30 :=
by
  sorry

end elephant_walk_distance_l18_18764


namespace bob_first_six_probability_l18_18394

theorem bob_first_six_probability :
  let P_six := 1/6,
      P_not_six := 5/6,
      P_bob_first_six := (P_not_six * P_not_six * P_six) / (1 - (P_not_six * P_not_six * P_not_six)) 
  in
  P_bob_first_six = 25 / 91 :=
by
  sorry

end bob_first_six_probability_l18_18394


namespace extinction_probability_l18_18718

-- Definitions from conditions
def prob_divide : ℝ := 0.6
def prob_die : ℝ := 0.4

-- Statement of the theorem
theorem extinction_probability :
  ∃ (v : ℝ), v = 2 / 3 :=
by
  sorry

end extinction_probability_l18_18718


namespace find_point_P_l18_18931

noncomputable def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

noncomputable def pointP : ℝ × ℝ × ℝ :=
let L := (2, 3, 1) in
let M := (1, 8, -2) in
let N := (3, 1, 5) in
let MN_midpoint := midpoint N M in
(2 * MN_midpoint.1 - L.1, 2 * MN_midpoint.2 - L.2, 2 * MN_midpoint.3 - L.3)

theorem find_point_P (L M N : ℝ × ℝ × ℝ) (hL : L = (2, 3, 1)) (hM : M = (1, 8, -2)) (hN : N = (3, 1, 5)) :
  pointP = (2, 6, 2) :=
by
  rw [pointP, hL, hM, hN]
  -- Calculate the midpoint of MN
  let MN_midpoint := midpoint N M
  -- Calculate point P based on MN_midpoint and L
  have hP : pointP = (2 * MN_midpoint.1 - L.1, 2 * MN_midpoint.2 - L.2, 2 * MN_midpoint.3 - L.3), by rfl
  rw [midpoint] at MN_midpoint
  -- Substitute the midpoint calculation results
  simp [MN_midpoint]
  -- Simplify to show the coordinates of P
  sorry

end find_point_P_l18_18931


namespace sum_of_smallest_angles_l18_18441

noncomputable def Q (x : ℂ) : ℂ :=
  (∑ i in Finset.range 20, x ^ i) ^ 2 - x ^ 19

theorem sum_of_smallest_angles :
  let α := (Finset.range 21).erase 0 ∪ (Finset.range 19).erase 0
  let smallest_angles := (α.map (λ k => (k : ℂ) / 21) ∪ α.map (λ k => (k : ℂ) / 19)).eraseDuplicates.sort
  let first_five_smallest_angles := smallest_angles.take 5
  (first_five_smallest_angles.sum = 183 / 399) :=
sorry

end sum_of_smallest_angles_l18_18441


namespace find_constants_and_extrema_l18_18113

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 - 2 * x + c

theorem find_constants_and_extrema :
  (∀ x : ℝ, ∃ a b c : ℝ, (f a b c x = a * x^3 + b * x^2 - 2 * x + c) ∧
  f_derivative : ∀ x, derivative (f a b c) x = 3 * a * x^2 + 2 * b * x - 2) →
  (f_derivative (-2) = 12 * a - 4 * b - 2 = 0 ∧ f_derivative 1 = 3 * a + 2 * b - 2 = 0 ∧ f (-2) = -8 * a + 4 * b + 4 + c = 6) →
  a = 1/3 ∧ b = 1/2 ∧ c = 8/3 ∧
  (∀ x : ℝ, x ∈ Icc (-3 : ℝ) (2 : ℝ) → 
    (derivative (f 1/3 1/2 8/3) x = x^2 + x - 2 = 0 ∧ 
    f (-3) = 4/6 ∧ f (-2) = 6 ∧ f (1) = 3/2 ∧ f (2) = 19/3) →
  (∀ x : ℝ, x ∈ Icc (-3 : ℝ) (2 : ℝ) → f x ≤ (19/3)) ∧
  (∀ x : ℝ, x ∈ Icc (-3 : ℝ) (2 : ℝ) → f x ≥ (3/2))) :=
  sorry

end find_constants_and_extrema_l18_18113


namespace find_X_sum_coordinates_l18_18205

/- Define points and their coordinates -/
variables (X Y Z : ℝ × ℝ)
variable  (XY XZ ZY : ℝ)
variable  (k : ℝ)
variable  (hxz : XZ = (3/4) * XY)
variable  (hzy : ZY = (1/4) * XY)
variable  (hy : Y = (2, 9))
variable  (hz : Z = (1, 5))

/-- Lean 4 statement for the proof problem -/
theorem find_X_sum_coordinates :
  (Y.1 = 2) ∧ (Y.2 = 9) ∧ (Z.1 = 1) ∧ (Z.2 = 5) ∧
  XZ = (3/4) * XY ∧ ZY = (1/4) * XY →
  (X.1 + X.2) = -9 := 
by
  sorry

end find_X_sum_coordinates_l18_18205


namespace largest_and_smallest_values_quartic_real_roots_l18_18157

noncomputable def function_y (a b x : ℝ) : ℝ :=
  (4 * a^2 * x^2 + b^2 * (x^2 - 1)^2) / (x^2 + 1)^2

theorem largest_and_smallest_values (a b : ℝ) (h : a > b) :
  ∃ x y, function_y a b x = y^2 ∧ y = a ∧ y = b :=
by
  sorry

theorem quartic_real_roots (a b y : ℝ) (h₁ : a > b) (h₂ : y > b) (h₃ : y < a) :
  ∃ x₀ x₁ x₂ x₃, function_y a b x₀ = y^2 ∧ function_y a b x₁ = y^2 ∧ function_y a b x₂ = y^2 ∧ function_y a b x₃ = y^2 :=
by
  sorry

end largest_and_smallest_values_quartic_real_roots_l18_18157


namespace solve_exponential_eq_l18_18094

theorem solve_exponential_eq (x : ℝ) : 
  (∃ x : ℝ, (8^x + 27^x) / (12^x + 18^x) = 7 / 6) ↔ x = -1 ∨ x = 1 :=
begin
  sorry
end

end solve_exponential_eq_l18_18094


namespace inequality_solution_l18_18610

variable {x : ℝ}

theorem inequality_solution :
  x ∈ Set.Ioo (-∞ : ℝ) 7 ∪ Set.Ioo (-∞ : ℝ) (-7) ∪ Set.Ioo (-7) 7 ↔ (x^2 - 49) / (x + 7) < 0 :=
by
  sorry

end inequality_solution_l18_18610


namespace hannah_strawberries_l18_18866

-- Definitions for the conditions
def daily_harvest : ℕ := 5
def days_in_april : ℕ := 30
def strawberries_given_away : ℕ := 20
def strawberries_stolen : ℕ := 30

-- The statement we need to prove
theorem hannah_strawberries (harvested_strawberries : ℕ)
  (total_harvest := daily_harvest * days_in_april)
  (total_lost := strawberries_given_away + strawberries_stolen)
  (final_count := total_harvest - total_lost) :
  harvested_strawberries = final_count :=
sorry

end hannah_strawberries_l18_18866


namespace extinction_prob_one_l18_18720

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the extinction probability function
def extinction_prob (v : ℕ → ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else p * v (k + 1) + q * v (k - 1)

-- State the theorem
theorem extinction_prob_one (v : ℕ → ℝ) :
  extinction_prob v 1 = 2 / 3 :=
sorry

end extinction_prob_one_l18_18720


namespace line_passes_through_fixed_point_l18_18631

theorem line_passes_through_fixed_point (k : ℝ) : ∀ x y : ℝ, (y - 1 = k * (x + 2)) → (x = -2 ∧ y = 1) :=
by
  intro x y h
  sorry

end line_passes_through_fixed_point_l18_18631


namespace problem_bound_l18_18454

theorem problem_bound (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by 
  sorry

end problem_bound_l18_18454


namespace distance_closest_points_l18_18754

-- Define the coordinates of the centers and the tangencies
def center1 : ℝ × ℝ := (3, 3)
def center2 : ℝ × ℝ := (20, 12)
def tangency_line1 : ℝ := 1
def tangency_axis2 := "x-axis"

-- Distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Radii calculation
def radius1 (center : ℝ × ℝ) (line : ℝ) : ℝ := 
  center.2 - line

def radius2 (center : ℝ × ℝ) : ℝ := 
  center.2

-- Distance between closest points of the circles
theorem distance_closest_points : 
  distance center1 center2 - (radius1 center1 tangency_line1 + radius2 center2) = Real.sqrt 370 - 14 :=
  by
  -- Proof is skipped
  sorry

end distance_closest_points_l18_18754


namespace curve_is_semicircle_l18_18096

theorem curve_is_semicircle (r θ : ℝ) (h₁ : r = 3) (h₂ : cos θ > 0) : 
  (r = 3 ∧ (∀ θ, -real.pi / 2 < θ ∧ θ < real.pi / 2 → cos θ > 0)) :=
sorry

end curve_is_semicircle_l18_18096


namespace find_f_of_5_l18_18453

def f (x : ℕ) : ℕ :=
if x >= 10 then x - 2 else f (f (x + 6))

theorem find_f_of_5 : f 5 = 11 := by
sorry

end find_f_of_5_l18_18453


namespace max_candies_l18_18306

/-- There are 28 ones written on the board. Every minute, Karlsson erases two arbitrary numbers
and writes their sum on the board, and then eats an amount of candy equal to the product of 
the two erased numbers. Prove that the maximum number of candies he could eat in 28 minutes is 378. -/
theorem max_candies (karlsson_eats_max_candies : ℕ → ℕ → ℕ) (n : ℕ) (initial_count : n = 28) :
  (∀ a b, karlsson_eats_max_candies a b = a * b) →
  (∃ max_candies, max_candies = 378) :=
sorry

end max_candies_l18_18306


namespace quadratic_completion_l18_18503

noncomputable def find_b (n : ℝ) : ℝ := 2 * n

theorem quadratic_completion (n b : ℝ)
  (h1 : (x : ℝ) : (x + n)^2 + 16 = x^2 + b * x + 24)
  (h2 : b > 0) : 
  b = find_b (real.sqrt 2) :=
sorry

end quadratic_completion_l18_18503


namespace inequality_solution_l18_18606

theorem inequality_solution (x : ℝ) (h : x ≠ -7) : 
  (x^2 - 49) / (x + 7) < 0 ↔ x ∈ Set.Ioo (-∞) (-7) ∪ Set.Ioo (-7) 7 := by 
  sorry

end inequality_solution_l18_18606


namespace range_of_values_for_a_l18_18845

theorem range_of_values_for_a {f : ℝ → ℝ}
  (h : ∀ x ∈ Iic 1, f x = sqrt (a * x + 1)) : -1 ≤ a ∧ a < 0 :=
by
  sorry

end range_of_values_for_a_l18_18845


namespace area_of_triangle_ABC_l18_18467

def i : ℝ^2 := ⟨1, 0⟩
def j : ℝ^2 := ⟨0, 1⟩

def AB : ℝ^2 := 4 • i + 2 • j
def AC : ℝ^2 := 3 • i + 4 • j

def BC : ℝ^2 := AC - AB

noncomputable def length_squared (v : ℝ^2) : ℝ := v.1^2 + v.2^2

def area_of_triangle (a b : ℝ^2) : ℝ := 1/2 * real.sqrt (length_squared a) * real.sqrt (length_squared b)

theorem area_of_triangle_ABC : area_of_triangle AB BC = 5 := 
by sorry

end area_of_triangle_ABC_l18_18467


namespace cube_odd_dot_probability_l18_18369

def faces := [2, 3, 4, 5, 6, 7]

def total_dots : ℕ := faces.sum

def probability_face_odd : ℚ :=
  let even_prob := (2 + 4 + 6) / total_dots * 2 / 3 in
  let odd_prob := (3 + 5 + 7) / total_dots * 1 / 3 in
  even_prob + odd_prob

theorem cube_odd_dot_probability : probability_face_odd = 13 / 27 := by
  sorry

end cube_odd_dot_probability_l18_18369


namespace pyramid_certain_height_l18_18616

noncomputable def certain_height (h : ℝ) : Prop :=
  let height := h + 20
  let width := height + 234
  (height + width = 1274) → h = 1000 / 3

theorem pyramid_certain_height (h : ℝ) : certain_height h :=
by
  let height := h + 20
  let width := height + 234
  have h_eq : (height + width = 1274) → h = 1000 / 3 := sorry
  exact h_eq

end pyramid_certain_height_l18_18616


namespace avg_of_last_11_eq_41_l18_18271

def sum_of_first_11 : ℕ := 11 * 48
def sum_of_all_21 : ℕ := 21 * 44
def eleventh_number : ℕ := 55

theorem avg_of_last_11_eq_41 (S1 S : ℕ) :
  S1 = sum_of_first_11 →
  S = sum_of_all_21 →
  (S - S1 + eleventh_number) / 11 = 41 :=
by
  sorry

end avg_of_last_11_eq_41_l18_18271


namespace find_f_of_f_inv_e_l18_18841

def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

theorem find_f_of_f_inv_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 :=
by
  sorry

end find_f_of_f_inv_e_l18_18841


namespace prob_not_has_bio_test_combined_prob_neither_bio_nor_chem_l18_18294

def prob_has_bio_test : ℚ := 5 / 8
def prob_not_has_chem_test : ℚ := 1 / 2

theorem prob_not_has_bio_test : 1 - 5 / 8 = 3 / 8 := by
  sorry

theorem combined_prob_neither_bio_nor_chem :
  (1 - 5 / 8) * (1 / 2) = 3 / 16 := by
  sorry

end prob_not_has_bio_test_combined_prob_neither_bio_nor_chem_l18_18294


namespace find_f_1_div_2007_l18_18769

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_1_div_2007 :
  f 0 = 0 ∧
  (∀ x, f x + f (1 - x) = 1) ∧
  (∀ x, f (x / 5) = f x / 2) ∧
  (∀ x1 x2, 0 ≤ x1 → x1 < x2 → x2 ≤ 1 → f x1 ≤ f x2) →
  f (1 / 2007) = 1 / 32 :=
sorry

end find_f_1_div_2007_l18_18769


namespace intercept_sum_l18_18413

theorem intercept_sum (x y : ℤ) (hx : 0 ≤ x) (hx1 : x < 40) (hy : 0 ≤ y) (hy1 : y < 40) 
  (h : 5 * x ≡ 3 * y - 2 [MOD 40]) : x + y = 38 :=
by
  sorry

end intercept_sum_l18_18413


namespace find_f_2015_l18_18142

def f (a b x : ℝ) : ℝ := a * Real.log x / Real.log 2 + b * Real.log x / Real.log 3 + 2

theorem find_f_2015
  (a b : ℝ)
  (h₁ : f a b (1 / 2015) = 4) :
  f a b 2015 = 0 :=
sorry -- Proof is omitted

end find_f_2015_l18_18142


namespace min_value_h12_l18_18397

variable (h : ℕ → ℕ)
variable (is_quibbling : ∀ x y : ℕ, x > 0 → y > 0 → h(x) + h(y) ≥ x^2 + 10 * y)
variable (minimum_sum : (List.range 16).tail.map h).sum = 1395

theorem min_value_h12 : h(12) ≥ 144 := by
  sorry

end min_value_h12_l18_18397


namespace total_amount_gathered_l18_18906

theorem total_amount_gathered (total_students : ℕ) (full_amount : ℕ) (half_paid_students : ℕ) (each_full_paid_amount : ℕ) (each_half_paid_amount : ℕ) :
  total_students = 25 →
  full_amount = 50 →
  half_paid_students = 4 →
  each_full_paid_amount = full_amount →
  each_half_paid_amount = full_amount / 2 →
  (let full_paid_students := total_students - half_paid_students in
   let total_full_amount := full_paid_students * each_full_paid_amount in
   let total_half_amount := half_paid_students * each_half_paid_amount in
   total_full_amount + total_half_amount = 1150) :=
begin
  sorry
end

end total_amount_gathered_l18_18906


namespace complex_addition_l18_18808

variables {a b : ℝ}
constant i : ℝ
axiom i_squared : i^2 = -1

theorem complex_addition :
  (2 + i) * (1 - b * i) = a + i → a + b = 2 :=
begin
  sorry,
end

end complex_addition_l18_18808


namespace simplify_and_evaluate_l18_18597

-- Define the condition as a predicate
def condition (a b : ℝ) : Prop := (a + 1/2)^2 + |b - 2| = 0

-- The simplified expression
def simplified_expression (a b : ℝ) : ℝ := 12 * a^2 * b - 6 * a * b^2

-- Statement: Given the condition, prove that the simplified expression evaluates to 18
theorem simplify_and_evaluate : ∀ (a b : ℝ), condition a b → simplified_expression a b = 18 :=
by
  intros a b hc
  sorry  -- Proof omitted

end simplify_and_evaluate_l18_18597


namespace ellipse_properties_trajectory_midpoint_slope_range_l18_18479

noncomputable def ellipse_equation (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def point_on_ellipse (a b : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (x y : ℝ) (hx : x = 1) (hy : y = sqrt 3 / 2) : Prop :=
  ellipse_equation a b h₀ h₁ x y

theorem ellipse_properties :
  ∀ (a b : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (hx : 2 * a = 4) (hy : point_on_ellipse a b h₀ h₁ 1 (sqrt 3 / 2))
  (P : ℝ × ℝ) (hP : P = (1, sqrt 3 / 2))
  (F₁ F₂ : ℝ × ℝ) (hF₁ : F₁ = (-sqrt 3, 0)) (hF₂ : F₂ = (sqrt 3, 0))
  (sum_dist_P : (dist P F₁ + dist P F₂ = 4)),
  ellipse_equation a b h₀ h₁ 1 (sqrt 3 / 2) ∧ 
  ∃ Q : (ℝ × ℝ), (x y : ℝ) →
    ((x = (-sqrt 3 + 2 * x) / 2) → (y = 2 * y) → ellipse_equation a b h₀ h₁ (2 * x + sqrt 3) (2 * y)) :=
sorry

theorem trajectory_midpoint :
  ∀ (a b x y : ℝ) (h₀ : a = 2) (h₁ : b = sqrt 1) (x y : ℝ) 
  (F₁ F₂ : ℝ × ℝ) (T : ℝ × ℝ) (hT : T = (x + sqrt 3 / 2, 2 * y))
  (h_ellipse : ellipse_equation a b (by sorry) (by sorry)),
  (x + sqrt 3 / 2)^2 + 4 * y^2 = 1 :=
sorry

theorem slope_range (M : ℝ × ℝ) (k : ℝ) :
  M = (0, 2) →
  ∀ (A B : ℝ × ℝ) (origin : ℝ × ℝ) (hO : origin = (0, 0))
  (hAB : ∀ (A B : ℝ × ℝ), ∠ A O B < π / 2)
  (l : ℝ) (h_l : l = (λ x : ℝ, k * x + 2))
  (h_range : ∃ k : ℝ, (3/4 < k^2 ∧ k^2 < 4)),
  k ∈ ((-2, -sqrt 3 / 2) ∪ (sqrt 3 / 2, 2)) :=
sorry

end ellipse_properties_trajectory_midpoint_slope_range_l18_18479


namespace area_CEH_is_34_l18_18048

-- Variables for geometrical entities
variables {A B C D E H : Type}
variables [trapezoid A B C D]
variables [height BH]
variables [on_diagonal E AC]

-- Given areas of specific triangles
variables (area_DEH : ℝ := 56)
variables (area_BEH : ℝ := 50)
variables (area_BCH : ℝ := 40)

-- Theorem statement to prove that the area of triangle CEH is 34
theorem area_CEH_is_34 : area (triangle C E H) = 34 := by
  sorry

end area_CEH_is_34_l18_18048


namespace cheese_balls_in_35oz_barrel_l18_18158

theorem cheese_balls_in_35oz_barrel
  (servings_in_24oz_barrel : ℕ)
  (cheese_balls_per_serving : ℕ)
  (barrel_size_24oz : ℕ)
  (barrel_size_35oz : ℕ)
  (h1 : servings_in_24oz_barrel = 60)
  (h2 : cheese_balls_per_serving = 12)
  (h3 : barrel_size_24oz = 24)
  (h4 : barrel_size_35oz = 35) :
  let total_cheese_balls_24oz := servings_in_24oz_barrel * cheese_balls_per_serving,
      cheese_balls_per_oz := (total_cheese_balls_24oz : ℚ) / barrel_size_24oz
  in barrel_size_35oz * cheese_balls_per_oz = 1050 := 
sorry

end cheese_balls_in_35oz_barrel_l18_18158


namespace ana_strip_palindrome_l18_18398

theorem ana_strip_palindrome :
  ∀ time : ℕ,
  ∀ ana_word bora_word : list char,
  (ana_word = ['A'] ∧ bora_word = ['B']) ∧ 
  (
    ∀ t ≤ time, 
    (ana_word.length ≤ t → 
      ∃ prefix suffix : list char, ana_word = prefix ++ suffix ∧ suffix ++ prefix = list.reverse (suffix ++ prefix))
  ) → 
  (
    ∃ prefix suffix : list char, ana_word = prefix ++ suffix ∧ suffix ++ prefix = list.reverse (suffix ++ prefix)
  ) :=
by
  intros,
  sorry

end ana_strip_palindrome_l18_18398


namespace plane_parallel_conditions_l18_18481

/-- 
Given conditions:
1. Planes α and β are both perpendicular to plane γ.
2. Within plane α, there exist three non-collinear points equidistant from plane β.
3. l and m are two lines within plane α, and both are parallel to plane β.
4. l and m are skew lines, and both are parallel to planes α and β.
Prove that none of these conditions alone is sufficient to conclude that plane α is parallel to plane β.
-/
theorem plane_parallel_conditions
  (α β γ : Plane)
  (h1 : IsPerpendicular α γ)
  (h2 : IsPerpendicular β γ)
  (nc_pts : ∃ (p1 p2 p3 : Point), NonCollinear p1 p2 p3 ∧ EquidistantFromPlane p1 β ∧ EquidistantFromPlane p2 β ∧ EquidistantFromPlane p3 β)
  (l m : Line)
  (h3 : InPlane l α ∧ InPlane m α ∧ IsParallel l β ∧ IsParallel m β)
  (h4 : IsSkew l m ∧ IsParallel l α ∧ IsParallel m α ∧ IsParallel l β ∧ IsParallel m β) :
  ¬ (Parallel α β) :=
sorry

end plane_parallel_conditions_l18_18481


namespace cos_angle_half_l18_18859

noncomputable theory -- Declares that we use non-computable definitions (e.g., angles, cosines)

open Real -- To work with real numbers easily

variables (a b : ℝ^3) -- Define vectors a and b in 3D real space
variables (θ : ℝ) -- Define θ as a real number (the angle between vectors a and b)

-- The conditions given in the problem
axiom nonzero_a : a ≠ 0
axiom nonzero_b : b ≠ 0
axiom condition : ‖a‖ = ‖b‖ ∧ ‖a‖ = ‖a - b‖

-- The theorem we need to prove
theorem cos_angle_half : cos θ = 1 / 2 :=
sorry -- Proof omitted

end cos_angle_half_l18_18859


namespace conic_is_hyperbola_l18_18275

noncomputable def polar_equation (theta : Real) : Real :=
  1 / (1 - Real.cos theta + Real.sin theta)

theorem conic_is_hyperbola (theta : Real) :
  let rho := polar_equation theta in
  let e := Real.sqrt 2 in
  e > 1 ∧ e.is_hyperbola :=
sorry

end conic_is_hyperbola_l18_18275


namespace triangle_area_l18_18037

def line1 (x : ℚ) : ℚ := x + 2
def line2 (x : ℚ) : ℚ := -3 * x + 9
def line3 (y : ℚ) : Prop := y = 2

theorem triangle_area :
  let A := (0, 2)
  let B := ((7 : ℚ) / 3, 2)
  let C := ((7 : ℚ) / 4, (15 : ℚ) / 4)
  let base := (7 : ℚ) / 3
  let height := (7 : ℚ) / 4
  area (triangle A B C) = (49 : ℚ) / 24 :=
by sorry

end triangle_area_l18_18037


namespace base_conversion_l18_18766

theorem base_conversion (b2_to_b10_step : 101101 = 1 * 2 ^ 5 + 0 * 2 ^ 4 + 1 * 2 ^ 3 + 1 * 2 ^ 2 + 0 * 2 + 1)
  (b10_to_b7_step1 : 45 / 7 = 6) (b10_to_b7_step2 : 45 % 7 = 3) (b10_to_b7_step3 : 6 / 7 = 0) (b10_to_b7_step4 : 6 % 7 = 6) :
  101101 = 45 ∧ 45 = 63 :=
by {
  -- Conversion steps from the proof will be filled in here
  sorry
}

end base_conversion_l18_18766


namespace range_of_x_min_y_eq_1_l18_18439

theorem range_of_x_min_y_eq_1 :
  (∃ x : ℝ, y = |x^2 - 1| + |2x^2 - 1| + |3x^2 - 1| ∧ y = 1) ↔ 
  (-sqrt(1 / 2) ≤ x ∧ x ≤ -sqrt(1 / 3)) ∨ (sqrt(1 / 3) ≤ x ∧ x ≤ sqrt(1 / 2)) := 
sorry

end range_of_x_min_y_eq_1_l18_18439


namespace order_b_gt_c_gt_a_l18_18809

noncomputable def a : ℝ := Real.log 2.6
def b : ℝ := 0.5 * 1.8^2
noncomputable def c : ℝ := 1.1^5

theorem order_b_gt_c_gt_a : b > c ∧ c > a := by
  sorry

end order_b_gt_c_gt_a_l18_18809


namespace expected_difference_tea_coffee_l18_18393

theorem expected_difference_tea_coffee :
  let prime_numbers := {2, 3, 5, 7}
  let composite_numbers := {4, 6, 8}
  let total_days := 365
  let probability_tea := 4 / 7
  let probability_coffee := 3 / 7
  let expected_tea_days := probability_tea * total_days
  let expected_coffee_days := probability_coffee * total_days
  expected_tea_days - expected_coffee_days = 53 :=
by
  sorry

end expected_difference_tea_coffee_l18_18393


namespace num_valid_pairs_l18_18437

theorem num_valid_pairs : ∃ N : ℕ, 
  N = 16 ∧ 
  (∀ (p a : ℕ), prime p ∧ p > 2 ∧ 1 ≤ a ∧ a ≤ 2024 ∧ a < p^4 ∧ 
    (∃ x : ℕ, x^2 = ap^4 + 2p^3 + 2p^2 + 1) ↔ 
    (((p ∈ {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}) ∧ a = (p + 1)^2) ∨ 
    ((p ∈ {3, 5, 7}) ∧ a = p^4 - 2p^3 - p^2 + 2p - 1))) :=
sorry

end num_valid_pairs_l18_18437


namespace find_BI_l18_18049

-- Define the conditions as variables
variables (ABCD EFGH : Type)
variables (sides_parallel : ∀ (a b c d : ABCD) (e f g h : EFGH), a ≠ b ∧ a ≠ c ∧ e ≠ f ∧ e ≠ h → ∃ p : ABCD → EFGH, a = p b ∧ c = p d ∧ e = p f ∧ h = p g)
variables (BD_length : ℝ)
variables (area_BFC : ℝ)
variables (area_CHD : ℝ)

-- Given conditions
axiom sides_parallel_ABCD_EFGH : sides_parallel ABCD EFGH
axiom length_BD : BD_length = 10
axiom area_BFC_value : area_BFC = 3
axiom area_CHD_value : area_CHD = 5

-- The proof goal
theorem find_BI : ∃ (BI : ℝ), BI = 15 / 4 :=
by
  -- We acknowledge the proof steps and the final result as given in the problem
  sorry

end find_BI_l18_18049


namespace election_votes_l18_18528

theorem election_votes (V : ℝ) (h1 : ∃ geoff_votes : ℝ, geoff_votes = 0.01 * V)
                       (h2 : ∀ candidate_votes : ℝ, (candidate_votes > 0.51 * V) → candidate_votes > 0.51 * V)
                       (h3 : ∃ needed_votes : ℝ, needed_votes = 3000 ∧ 0.01 * V + needed_votes = 0.51 * V) :
                       V = 6000 :=
by sorry

end election_votes_l18_18528


namespace toothpicks_15th_stage_l18_18527
-- Import the required library

-- Define the arithmetic sequence based on the provided conditions.
def toothpicks (n : ℕ) : ℕ :=
  if n = 1 then 5 else 3 * (n - 1) + 5

-- State the theorem
theorem toothpicks_15th_stage : toothpicks 15 = 47 :=
by {
  -- Provide the proof here, but currently using sorry as instructed
  sorry
}

end toothpicks_15th_stage_l18_18527


namespace ratio_first_part_l18_18694

theorem ratio_first_part (x : ℕ) (h1 : x / 3 = 2) : x = 6 :=
by
  sorry

end ratio_first_part_l18_18694


namespace max_wooden_pencils_l18_18909

theorem max_wooden_pencils (m w : ℕ) (p : ℕ) (h1 : m + w = 72) (h2 : m = w + p) (hp : Nat.Prime p) : w = 35 :=
by
  sorry

end max_wooden_pencils_l18_18909


namespace even_function_a_min_value_l18_18899

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2^x - a * (1/2)^x

-- The main theorem stating the required conditions and conclusions
theorem even_function_a_min_value (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → (a = -1 ∧ ∀ x : ℝ, f (-1) x ≥ 2) :=
begin
  intro h,
  split,
  {
    -- Proof that a = -1
    sorry,
  },
  {
    -- Proof that the minimum value of f(x) when a = -1 is 2
    intro x,
    sorry,
  }
end

end even_function_a_min_value_l18_18899


namespace fifth_odd_number_in_21st_row_is_809_l18_18739

-- Conditions
def is_pattern (n k : ℕ) : Prop :=
  k = 2 * n - 1

def total_numbers_in_first_n_rows (n : ℕ) : ℕ :=
  (n * (2 * n + 1)) / 2

-- Question rephrased as a Lean statement
theorem fifth_odd_number_in_21st_row_is_809 :
  ∀ (row pos : ℕ), row = 21 → pos = 5 → 
  let total_in_20_rows := total_numbers_in_first_n_rows 20 in
  let kth_odd_number := 2 * (total_in_20_rows + pos) - 1 in
  kth_odd_number = 809 :=
by
  intros row pos h_row h_pos total_in_20_rows kth_odd_number
  sorry

end fifth_odd_number_in_21st_row_is_809_l18_18739


namespace min_distance_from_origin_to_line_l18_18537

open Real

theorem min_distance_from_origin_to_line :
    ∀ x y : ℝ, (3 * x + 4 * y - 4 = 0) -> dist (0, 0) (x, y) = 4 / 5 :=
by
  sorry

end min_distance_from_origin_to_line_l18_18537


namespace reflection_matrix_squared_is_identity_l18_18563

noncomputable def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let ⟨a, b⟩ := v
  let norm_sq := a^2 + b^2
  Matrix.of_vec [
    (a^2 - b^2) / norm_sq, 2 * a * b / norm_sq, 
    2 * a * b / norm_sq, (b^2 - a^2) / norm_sq
  ]

theorem reflection_matrix_squared_is_identity (v : ℝ × ℝ) : 
  reflection_matrix v * reflection_matrix v = 1 :=
by 
  let R := reflection_matrix (2, -1)
  have H : R * R = 1 -- This H represents our goal which will be proved
  sorry -- Proof goes here but is omitted

end reflection_matrix_squared_is_identity_l18_18563


namespace integer_values_n_l18_18872

theorem integer_values_n (n : ℤ) : 
    (∃ n1 n2 n3 : ℤ, 
        n = n1 ∧ (n1 + Complex.I) ^ 6 ∈ ℤ ∧ 
        n = n2 ∧ (n2 + Complex.I) ^ 6 ∈ ℤ ∧ 
        n = n3 ∧ (n3 + Complex.I) ^ 6 ∈ ℤ) :=
begin
  sorry
end

end integer_values_n_l18_18872


namespace elimination_game_basic_events_l18_18323

/-- Two individuals independently participate in a TV station's elimination game, 
    where passing the round earns them 1 point and failing to pass earns them 0 points.
    Determine the set of basic events. -/
theorem elimination_game_basic_events :
  (∃ A : set (ℕ × ℕ), A = {(1, 1), (1, 0), (0, 1), (0, 0)}) :=
by
  use {(1, 1), (1, 0), (0, 1), (0, 0)}
  sorry

end elimination_game_basic_events_l18_18323


namespace minimize_PQRS_area_l18_18986

theorem minimize_PQRS_area (k : ℝ) (h₀ : 0 < k) (h₁ : k < 4)
  (A B C D P Q R S : ℝ × ℝ)
  (h_square : (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 
               ∧ (D.1 - C.1)^2 + (D.2 - C.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2 
               ∧ (A.1 - D.1) * (C.1 - B.1) + (A.2 - D.2) * (C.2 - B.2) = 0) 
  (h_side_length : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 16)
  (h_ratio : ∀ {X Y Z : ℝ × ℝ}, X ≠ Y → Y ≠ Z → Z ≠ X → ∃ k ℝ, 
             (vector_length (Y - X) / vector_length (Z - Y) = k / (4 - k))) 
  : k = 8 / 3 :=
sorry

end minimize_PQRS_area_l18_18986


namespace intersection_points_polar_coords_l18_18854

theorem intersection_points_polar_coords :
  (∀ (x y : ℝ), ((x - 4)^2 + (y - 5)^2 = 25 ∧ (x^2 + y^2 - 2*y = 0)) →
  (∃ (ρ θ : ℝ), ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
    ((x, y) = (ρ * Real.cos θ, ρ * Real.sin θ)) ∧
    ((ρ = 2 ∧ θ = Real.pi / 2) ∨ (ρ = Real.sqrt 2 ∧ θ = Real.pi / 4)))) :=
sorry

end intersection_points_polar_coords_l18_18854


namespace find_first_number_l18_18300

theorem find_first_number 
  (second_number : ℕ)
  (increment : ℕ)
  (final_number : ℕ)
  (h1 : second_number = 45)
  (h2 : increment = 11)
  (h3 : final_number = 89)
  : ∃ first_number : ℕ, first_number + increment = second_number := 
by
  sorry

end find_first_number_l18_18300


namespace minimal_minutes_to_fill_barrels_l18_18233

theorem minimal_minutes_to_fill_barrels :
  let fill_condition := λ (n k : ℕ), n * (k + 1)
  ∃ (n : ℕ), n * (n + 1) ≥ 2012 ∧ n + 1 = 46 := sorry

end minimal_minutes_to_fill_barrels_l18_18233


namespace f_monotonically_increasing_on_e_to_infinity_f_no_minimum_value_on_0_to_1_f_local_minimum_at_e_l18_18801

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem f_monotonically_increasing_on_e_to_infinity :
  ∀ x y : ℝ, e < x → e < y → x < y → f x < f y :=
sorry

theorem f_no_minimum_value_on_0_to_1 :
  ¬ ∃ m : ℝ, ∀ x : ℝ, 0 < x → x < 1 → f(x) ≥ m :=
sorry

theorem f_local_minimum_at_e : ∃ δ > 0, ∀ x : ℝ, |x - e| < δ → f e ≤ f x :=
sorry

end f_monotonically_increasing_on_e_to_infinity_f_no_minimum_value_on_0_to_1_f_local_minimum_at_e_l18_18801


namespace original_ratio_l18_18298

theorem original_ratio (B D : ℕ) (hB : B = 150) (h_new_B : B + 30 = 180) (h_ratio : (B + 30):D = 1/2) :
  B:D = 5:12 := 
by
  sorry

end original_ratio_l18_18298


namespace student_marks_l18_18918

theorem student_marks (x : ℕ) :
  let total_questions := 60
  let correct_answers := 38
  let wrong_answers := total_questions - correct_answers
  let total_marks := 130
  let marks_from_correct := correct_answers * x
  let marks_lost := wrong_answers * 1
  let net_marks := marks_from_correct - marks_lost
  net_marks = total_marks → x = 4 :=
by
  intros
  sorry

end student_marks_l18_18918


namespace parabola_coefficients_sum_l18_18630

theorem parabola_coefficients_sum (a b c : ℝ)
  (h_eqn : ∀ y, (-1) = a * y^2 + b * y + c)
  (h_vertex : (-1, -10) = (-a/(2*a), (4*a*c - b^2)/(4*a)))
  (h_pass_point : 0 = a * (-9)^2 + b * (-9) + c) 
  : a + b + c = 120 := 
sorry

end parabola_coefficients_sum_l18_18630


namespace constant_term_expansion_l18_18903

theorem constant_term_expansion (n : ℕ)
  (h : (binomial_coeff n 2 : ℚ) * (1/14) = binomial_coeff n 4 * (3/14) ) :
  let c := (x^2 - (1 / sqrt(x))) in constant_term(expand_binomial c n) = 45 := 
sorry

def binomial_coeff (n k : ℕ) : ℚ := (nat.choose n k : ℚ)

def expand_binomial (pol : polynomial ℚ) (n : ℕ) : polynomial ℚ :=
  polynomial.expand _ n pol

def constant_term (p : polynomial ℚ) : ℚ :=
  p.coeff 0

noncomputable def x : ℚ := sorry -- placeholder for x

end constant_term_expansion_l18_18903


namespace no_points_P_l18_18220

noncomputable def exists_point_P (P A B : Point) : Prop :=
  APB = 90 ∧ dist(P, A)^2 + dist(P, B)^2 = 10

theorem no_points_P (P : Point) (A B : Point) : 
  let circle_radius := 2
  let AB := diameter(circle_radius)
  let dist_AP := dist(P, A)
  let dist_BP := dist(P, B)
  exists_point_P (P A B) → False :=
by
  sorry

end no_points_P_l18_18220


namespace arithmetic_sequence_difference_l18_18406

theorem arithmetic_sequence_difference (a d : ℕ) (n m : ℕ) (hnm : m > n) (h_a : a = 3) (h_d : d = 7) (h_n : n = 1001) (h_m : m = 1004) :
  (a + (m - 1) * d) - (a + (n - 1) * d) = 21 :=
by
  sorry

end arithmetic_sequence_difference_l18_18406


namespace sqrt_arithmetic_identity_l18_18753

theorem sqrt_arithmetic_identity : 4 * (Real.sqrt 2) * (Real.sqrt 3) - (Real.sqrt 12) / (Real.sqrt 2) + (Real.sqrt 24) = 5 * (Real.sqrt 6) := by
  sorry

end sqrt_arithmetic_identity_l18_18753


namespace handshakes_max_number_of_men_l18_18307

theorem handshakes_max_number_of_men (n : ℕ) (h: n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end handshakes_max_number_of_men_l18_18307


namespace trajectory_of_M_is_hyperbola_l18_18297

noncomputable def circle (O : Point) (r : ℝ) : set Point :=
  { P : Point | dist O P = r }

noncomputable def is_hyperbola (O A : Point) (M : set Point) : Prop :=
  ∀ Q : Point, Q ∈ M → abs (dist Q O - dist Q A) = dist O A

theorem trajectory_of_M_is_hyperbola 
  (O A : Point) (r : ℝ) 
  (hO : O ≠ A) 
  (P : Point) (M : Point) 
  (P_on_circle : P ∈ circle O r)
  (M_bisector : dist M A = dist M P) :
  is_hyperbola O A (trajectory M) :=
sorry

end trajectory_of_M_is_hyperbola_l18_18297


namespace set_intersection_complement_l18_18500
open Set

variable (U A B : Set ℕ)

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {1, 2}

theorem set_intersection_complement :
  A ∩ (U \ B) = {3, 4} :=
by
  sorry

end set_intersection_complement_l18_18500


namespace product_of_solutions_of_quadratic_l18_18335

theorem product_of_solutions_of_quadratic :
  ∀ (x p q : ℝ), 36 - 9 * x - x^2 = 0 ∧ (x = p ∨ x = q) → p * q = -36 :=
by sorry

end product_of_solutions_of_quadratic_l18_18335


namespace parabola_equation_l18_18836

open Real

theorem parabola_equation (vertex focus : ℝ × ℝ) (h_vertex : vertex = (0, 0)) (h_focus : focus = (0, 3)) :
  ∃ a : ℝ, x^2 = 12 * y := by
  sorry

end parabola_equation_l18_18836


namespace intersection_complement_eq_l18_18156

open Set

def U : Set Int := univ
def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 3}

theorem intersection_complement_eq :
  (U \ M) ∩ N = {3} :=
  by sorry

end intersection_complement_eq_l18_18156


namespace minimum_area_of_rectangle_l18_18051

theorem minimum_area_of_rectangle (x y : ℝ) (h1 : x = 3) (h2 : y = 4) : 
  (min_area : ℝ) = (2.3 * 3.3) :=
by
  have length_min := x - 0.7
  have width_min := y - 0.7
  have min_area := length_min * width_min
  sorry

end minimum_area_of_rectangle_l18_18051


namespace compute_one_plus_i_power_four_l18_18760

theorem compute_one_plus_i_power_four (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 :=
by
  sorry

end compute_one_plus_i_power_four_l18_18760


namespace no_prime_of_form_3811_l18_18779

theorem no_prime_of_form_3811 (n : ℕ) : 
  ¬ (prime (3 * 10^(n + 1) + 8 * 10^n + (10^n - 1) / 9)) :=
by
  sorry

end no_prime_of_form_3811_l18_18779


namespace no_adjacent_alder_trees_probability_l18_18371

noncomputable def trees_probability : ℚ := 2 / 4290

theorem no_adjacent_alder_trees_probability :
  let cedar_trees := 4
  let pine_trees := 3
  let alder_trees := 6
  let total_trees := cedar_trees + pine_trees + alder_trees
  ∀ arrangements : Finset (Finset (Fin total_trees)),
  (∀ a1 a2 ∈ arrangements, a1 ≠ a2 → (a1 ∩ a2 = ∅)) →
  (∀ t ∈ arrangements, t.card = alder_trees) →
  ( ∃ valid_arrangement : Finset (Fin total_trees), 
    valid_arrangement.card = combinations total_trees alder_trees.val ∧ -- Total valid arrangements
    set.card {arr ∈ arrangements | no two alder trees are adjacent (in arr)} = 28) →
  probability = trees_probability :=
sorry

end no_adjacent_alder_trees_probability_l18_18371


namespace part1_part2_l18_18810

-- Part (1)
theorem part1 (m n : ℝ) (hm : |m| = 1) (hn : |n| = 4) (opposite : m * n < 0) :
  m + n = -3 ∨ m + n = 3 :=
sorry

-- Part (2)
theorem part2 (m n : ℝ) (hm : |m| = 1) (hn : |n| = 4) :
  (m - n) ≤ 5 :=
sorry

end part1_part2_l18_18810


namespace probability_at_least_one_of_A_B_hired_l18_18448

theorem probability_at_least_one_of_A_B_hired :
  let applicants := {A, B, C, D}
  let hired_combinations := applicants.subsets 2
  let total_prob := (hired_combinations.card : ℚ) / (applicants.powerset.card : ℚ)
  let neither_A_nor_B := (hired_combinations.erase {C, D}).card
  have prob_neither_A_nor_B := (neither_A_nor_B : ℚ) / (hired_combinations.card : ℚ)
  prob := 1 - prob_neither_A_nor_B
  prob = 5 / 6 := 
begin
  sorry
end

end probability_at_least_one_of_A_B_hired_l18_18448


namespace board_partition_possible_l18_18010

variable (m n : ℕ)

theorem board_partition_possible (hm : m > 15) (hn : n > 15) :
  ((∃ k1, m = 5 * k1 ∧ ∃ k2, n = 4 * k2) ∨ (∃ k3, m = 4 * k3 ∧ ∃ k4, n = 5 * k4)) :=
sorry

end board_partition_possible_l18_18010


namespace stock_price_is_500_l18_18182

-- Conditions
def income : ℝ := 1000
def dividend_rate : ℝ := 0.50
def investment : ℝ := 10000
def face_value : ℝ := 100

-- Theorem Statement
theorem stock_price_is_500 : 
  (dividend_rate * face_value / (investment / 1000)) = 500 := by
  sorry

end stock_price_is_500_l18_18182


namespace minimize_surface_area_ratio_l18_18457

theorem minimize_surface_area_ratio (V : ℝ) (r h : ℝ) (π : ℝ := real.pi) 
  (volume_eq : V = π * r^2 * h) 
  (surface_area_eq : Π = π * r^2 + 4 * π * r * h) :
  (h / r = 4) := by
    sorry

end minimize_surface_area_ratio_l18_18457


namespace coin_toss_sequences_count_l18_18748

-- Definitions of the conditions in the problem.
def is_valid_sequence (s : List Bool) : Prop :=
  s.length = 12 ∧
  (s.filter (λ (p : Bool × Bool), p = (true, true)).length = 1) ∧
  (s.filter (λ (p : Bool × Bool), p = (true, false)).length = 2) ∧
  (s.filter (λ (p : Bool × Bool), p = (false, true)).length = 2) ∧
  (s.filter (λ (p : Bool × Bool), p = (false, false)).length = 3)

-- The main theorem statement with conditions.
theorem coin_toss_sequences_count : 
  ∃ (s : List Bool), is_valid_sequence s ∧ 
  ((s : List Bool).length = 12 ∧
  (s.filter (λ (p : Bool × Bool), p = (true, true)).length = 1) ∧
  (s.filter (λ (p : Bool × Bool), p = (true, false)).length = 2) ∧
  (s.filter (λ (p : Bool × Bool), p = (false, true)).length = 2) ∧
  (s.filter (λ (p : Bool × Bool), p = (false, false)).length = 3))
    -> (∃ n, n = 20) := sorry

end coin_toss_sequences_count_l18_18748


namespace garrison_initial_men_l18_18372

theorem garrison_initial_men (x : ℕ) (h₁ : 54 * x - 15 * x = 39 * x)
  (h₂ : 20 * (x + 1900) = 39 * x) : x = 2000 :=
by {
  have h := congr_fun (congr_arg has_sub.sub h₁) x,
  rw [h₂, ←has_add.add_assoc] at h,
  have h₃ : 39 * x - 20 * x = 19 * x := by { ring },
  rw h₃ at h,
  have h₄ : 19 * x = 38000 := by { linarith },
  linarith,
}

end garrison_initial_men_l18_18372


namespace problem_part1_problem_part2_l18_18459

def circle_eq : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 6*x + 5 = 0

def line_passing_through_origin (K : ℝ) : ℝ → ℝ → Prop := 
  λ x y, y = K * x

noncomputable def midpoint_trajectory (x y : ℝ) : Prop := 
  (x - 3/2)^2 + y^2 = 9/4

noncomputable def K_range : ℝ → Prop := 
  λ K, -2 * real.sqrt 5 / 5 ≤ K ∧ K ≤ 2 * real.sqrt 5 / 5

theorem problem_part1 (K : ℝ) : 
  (∃ x y, circle_eq x y ∧ line_passing_through_origin K x y) → K_range K := 
sorry

theorem problem_part2 (K : ℝ) (x1 y1 x2 y2 : ℝ) (x y : ℝ) : 
  (circle_eq x1 y1 ∧ circle_eq x2 y2 ∧ line_passing_through_origin K x1 y1 ∧ 
    line_passing_through_origin K x2 y2 ∧ x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2 ∧ 
    K^2 < 4/5 ∧ K_range K) → 
  (midpoint_trajectory x y ∧ 5/3 < x ∧ x ≤ 3 := 
sorry

end problem_part1_problem_part2_l18_18459


namespace reciprocal_hcf_eq_one_l18_18617

theorem reciprocal_hcf_eq_one (a b : ℕ) (h₁ : a = 24) (h₂ : b = 169) : 1 / (Nat.gcd a b) = 1 :=
by
  rw [h₁, h₂]
  have h_gcd : Nat.gcd 24 169 = 1 := by sorry
  rw [h_gcd]
  norm_num

end reciprocal_hcf_eq_one_l18_18617


namespace problem_l18_18107

noncomputable def f : ℕ → ℕ
| x => if x = 0 then 1 else if ∃ k, x = 3 * k then f (x - 3) + 2 * (x - 3) + 3 else 0

theorem problem : f 1500 = 750001 := by
  sorry

end problem_l18_18107


namespace find_f_of_3_l18_18513

theorem find_f_of_3 (a b c : ℝ) (f : ℝ → ℝ)
  (h₁ : f(1) = 3) (h₂ : f(2) = 5) (h₃ : ∀ x, f(x) = a*x^2 + b*x + c) :
  f(3) = 7 :=
by
  sorry

end find_f_of_3_l18_18513


namespace fit_small_boxes_l18_18529

def larger_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def small_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

theorem fit_small_boxes (L W H l w h : ℕ)
  (larger_box_dim : L = 12 ∧ W = 14 ∧ H = 16)
  (small_box_dim : l = 3 ∧ w = 7 ∧ h = 2)
  (min_boxes : larger_box_volume L W H / small_box_volume l w h = 64) :
  ∃ n, n ≥ 64 :=
by
  sorry

end fit_small_boxes_l18_18529


namespace complementary_events_l18_18803

def bag : Type := { red := 2, white := 2 }

def draw_two_balls (bag : Type) : list (string × string) :=
  [("white", "white"), ("white", "red"), ("red", "red")]

/- Event Definitions -/
def at_least_one_white (draw : (string × string)) : Prop :=
  draw.fst = "white" ∨ draw.snd = "white"

def both_white (draw : (string × string)) : Prop :=
  draw.fst = "white" ∧ draw.snd = "white"

def at_least_one_red (draw : (string × string)) : Prop :=
  draw.fst = "red" ∨ draw.snd = "red"

def exactly_one_white (draw : (string × string)) : Prop :=
  (draw.fst = "white" ∧ draw.snd = "red") ∨ (draw.fst = "red" ∧ draw.snd = "white")

def both_red (draw : (string × string)) : Prop :=
  draw.fst = "red" ∧ draw.snd = "red"

/- Proof Statement -/
theorem complementary_events : ∀ (draw : string × string), 
  (at_least_one_white draw ∨ both_red draw) ∧ (¬ (at_least_one_white draw ∧ both_red draw)) := sorry

end complementary_events_l18_18803


namespace max_value_func1_l18_18356

theorem max_value_func1 (x : ℝ) (h : 0 < x ∧ x < 2) : 
  ∃ y, y = x * (4 - 2 * x) ∧ (∀ z, z = x * (4 - 2 * x) → z ≤ 2) :=
sorry

end max_value_func1_l18_18356


namespace school_profit_calc_l18_18740

-- Definitions based on the conditions provided
def pizza_slices : Nat := 8
def slices_per_pizza : ℕ := 8
def slice_price : ℝ := 1.0 -- Defining price per slice
def pizzas_bought : ℕ := 55
def cost_per_pizza : ℝ := 6.85
def total_revenue : ℝ := pizzas_bought * slices_per_pizza * slice_price
def total_cost : ℝ := pizzas_bought * cost_per_pizza

-- The lean mathematical statement we need to prove
theorem school_profit_calc :
  total_revenue - total_cost = 63.25 := by
  sorry

end school_profit_calc_l18_18740


namespace reflection_across_y_axis_coordinates_l18_18995

def coordinates_after_reflection (x y : ℤ) : ℤ × ℤ :=
  (-x, y)

theorem reflection_across_y_axis_coordinates :
  coordinates_after_reflection (-3) 4 = (3, 4) :=
by
  sorry

end reflection_across_y_axis_coordinates_l18_18995


namespace find_abc_l18_18515

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom condition1 : a * b = 45 * (3 : ℝ)^(1/3)
axiom condition2 : a * c = 75 * (3 : ℝ)^(1/3)
axiom condition3 : b * c = 30 * (3 : ℝ)^(1/3)

theorem find_abc : a * b * c = 75 * (2 : ℝ)^(1/2) := sorry

end find_abc_l18_18515


namespace seven_stars_needed_l18_18115

-- Define the grid size and properties
def grid_size : ℕ := 4

def suff_stars : ℕ := 7

def in_grid (i j : ℕ) : Prop := i < grid_size ∧ j < grid_size

def star_placed (f : ℕ → ℕ → Prop) (i j : ℕ) : Prop := in_grid i j ∧ f i j

-- The main theorem
theorem seven_stars_needed (f : ℕ → ℕ → Prop) : 
  (∃ (f' : ℕ → ℕ → Prop), (∀ i j, in_grid i j → f' i j → f i j) ∧ 
  (∀ r1 r2 c1 c2, r1 < grid_size → r2 < grid_size → c1 < grid_size → c2 < grid_size → 
  ∃ i j, in_grid i j ∧ ¬(i = r1 ∨ i = r2 ∨ j = c1 ∨ j = c2) ∧ f' i j) ∧
  (∀ r1 r2 c1 c2, r1 < grid_size → r2 < grid_size → c1 < grid_size → c2 < grid_size → 
   cardinality (λ i j, in_grid i j ∧ ¬(i = r1 ∨ i = r2 ∨ j = c1 ∨ j = c2) ∧ f' i j) ≥ 1)) → 
   (cardinality (λ i j, f i j) ≥ suff_stars) :=
sorry

end seven_stars_needed_l18_18115


namespace sum_of_g_47_l18_18958

def f (x : ℝ) : ℝ := 5 * x^2 - 3
def g (y : ℝ) : ℝ := y^2 - y + 2

theorem sum_of_g_47 :
  let vals := (λ (x : ℝ), g 47) '' ({x : ℝ | f x = 47}) in
  ∀ sums : vals, sums = {24} :=
by
  sorry

end sum_of_g_47_l18_18958


namespace tan_alpha_is_2_l18_18466

noncomputable def α : ℝ := sorry -- α is an angle
def cosα := - (Real.sqrt 5) / 5
def quadrant_3 (x : ℝ) := sorry -- some condition to enforce that α is in the third quadrant

theorem tan_alpha_is_2 (h_cos : cos α = cosα) (h_quad : quadrant_3 α) : 
  Real.tan α = 2 :=
  sorry

end tan_alpha_is_2_l18_18466


namespace combined_cost_of_ads_l18_18309

theorem combined_cost_of_ads
  (num_companies : ℕ)
  (ads_per_company : ℕ)
  (length : ℕ)
  (width : ℕ)
  (cost_per_sqft : ℕ)
  : num_companies = 3 →
    ads_per_company = 10 →
    length = 12 →
    width = 5 →
    cost_per_sqft = 60 →
    num_companies * ads_per_company * (length * width) * cost_per_sqft = 108000 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end combined_cost_of_ads_l18_18309


namespace ordered_pairs_count_l18_18506

theorem ordered_pairs_count :
  let N := 499 in
  ∃ f : ℝ → ℝ → Prop,
    (∀ a b, (a > 0 ∧ (b : ℤ) ∈ set.Icc 2 500 ∧ f a b) ↔ (log b a ^ 1001 = log b (a ^ 1001))) ∧
    ∃ p : {n : ℕ // n = 3 * N},
      p.1 = 1497 :=
by
  sorry

end ordered_pairs_count_l18_18506


namespace sum_of_sequence_l18_18305

variable (S a b : ℝ)

theorem sum_of_sequence :
  (S - a) / 100 = 2022 →
  (S - b) / 100 = 2023 →
  (a + b) / 2 = 51 →
  S = 202301 :=
by
  intros h1 h2 h3
  sorry

end sum_of_sequence_l18_18305


namespace length_of_cistern_l18_18696

-- Definitions for the problem
def width := 8 -- width of the cistern in meters
def depth := 1.25 -- depth of the water in meters
def total_wet_surface_area := 62 -- total wet surface area in square meters

-- Theorem statement
theorem length_of_cistern : 
  ∃ (L : ℝ), (L * width + 2 * (L * depth) + 2 * (width * depth) = total_wet_surface_area) ∧ L = 4 :=
by
  sorry

end length_of_cistern_l18_18696


namespace determine_polynomial_l18_18074

theorem determine_polynomial (p : ℝ → ℝ) (h0 : p 0 = 0)
  (hf : ∀ n : ℕ, (λ x, Real.floor (p (Real.floor (p n)))) + n = 4 * Real.floor (p n)):
  p = (λ x, (2 + Real.sqrt 3) * x) := 
sorry

end determine_polynomial_l18_18074


namespace regular_polygon_angle_properties_l18_18472

theorem regular_polygon_angle_properties :
  (∀ (n : ℕ), ∃ (x : ℝ), n > 2 ∧ x = 180 - (x + 90)) → 
  (∃ (sum_of_interior_angles : ℝ) (num_sides : ℕ), sum_of_interior_angles = 1080 ∧ num_sides = 8) :=
by {
  intro h,
  sorry
}

end regular_polygon_angle_properties_l18_18472


namespace board_partition_possible_l18_18012

-- Definition of natural numbers m and n greater than 15
variables (m n : ℕ)
-- m > 15
def m_greater_than_15 := m > 15
-- n > 15
def n_greater_than_15 := n > 15

-- Definition of m and n divisibility conditions
def divisible_by_4_or_5 (x : ℕ) : Prop :=
  x % 4 = 0 ∨ x % 5 = 0

def partition_possible (m n : ℕ) : Prop :=
  (m % 4 = 0 ∧ n % 5 = 0) ∨ (m % 5 = 0 ∧ n % 4 = 0)

-- The final statement of Lean
theorem board_partition_possible :
  m_greater_than_15 m → n_greater_than_15 n → partition_possible m n :=
by
  intro h_m h_n
  sorry

end board_partition_possible_l18_18012


namespace value_OP_squared_l18_18695

variables (O A B C D P E F : Type)
noncomputable def circle (O : Type) := {p : O // dist O p = 20}
noncomputable def EqMid (A B : Type) (m : Type) := dist A m = dist B m
noncomputable def DistMidChords (E F : Type) := dist E F = 16

axiom dist : O → O → ℝ
axiom radius_O : dist O A = 20
axiom length_AB : dist A B = 24
axiom length_CD : dist C D = 16
axiom midpoints_EF_dist : DistMidChords E F
axiom chord_AB_AP_PB : ∃(P : O), dist A P = 2 * dist P B

theorem value_OP_squared : exists OP_squared : ℝ, OP_squared = 290 := by
  sorry

end value_OP_squared_l18_18695


namespace symmetrical_point_of_P_is_correct_l18_18538

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the function to get the symmetric point with respect to the origin
def symmetrical_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Prove that the symmetrical point of P with respect to the origin is (1, -2)
theorem symmetrical_point_of_P_is_correct : symmetrical_point P = (1, -2) :=
  sorry

end symmetrical_point_of_P_is_correct_l18_18538


namespace parabola_slope_problem_l18_18471

theorem parabola_slope_problem
  (B C : ℝ × ℝ)
  (p : ℝ)
  (hP : p > 0)
  (hB : B.2^2 = 2 * p * B.1)
  (hC : C.2^2 = 2 * p * C.1)
  (hB_quadrant : B.1 > 0 ∧ B.2 < 0)
  (hC_quadrant : C.1 > 0 ∧ C.2 > 0)
  (h_angle_OBC : B.1 * C.2 - B.2 * C.1 = |B| * |C| * (1/2))
  (h_angle_BOC : B.1 * C.1 + B.2 * C.2 = |B| * |C| * (1/2 + sqrt(3)/2)) :
  let k := C.2 / C.1 in
  k^3 + 2 * k = sqrt 3 :=
sorry

end parabola_slope_problem_l18_18471


namespace segment_length_l18_18970

theorem segment_length (AB BC AC : ℝ) (hAB : AB = 4) (hBC : BC = 3) :
  AC = 7 ∨ AC = 1 :=
sorry

end segment_length_l18_18970


namespace time_to_cross_pole_l18_18672

theorem time_to_cross_pole (train_length : ℝ) (train_speed_kmh : ℝ) :
  train_length = 50 → train_speed_kmh = 144 → 
  let train_speed_ms := train_speed_kmh * (1000 / 3600) in
  let time := train_length / train_speed_ms in
  time = 1.25 := by
    intros h1 h2
    simp [h1, h2]
    sorry

end time_to_cross_pole_l18_18672


namespace positive_value_of_A_l18_18214

theorem positive_value_of_A (A : ℝ) :
  (A ^ 2 + 7 ^ 2 = 200) → A = Real.sqrt 151 :=
by
  intros h
  sorry

end positive_value_of_A_l18_18214


namespace length_of_BD_l18_18930

theorem length_of_BD
  (A B C D E : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  [Triangle ABC]
  (angle_C : angle A B C = 90)
  (AC : A.dist B C = 9)
  (BC : A.dist C A = 12)
  (D_on_AB : D ∈ (A.segment B))
  (E_on_BC : E ∈ (B.segment C))
  (angle_BED : angle B E D = 90)
  (DE : E.dist D = 5)
  : B.dist D = 25 / 3 := sorry

end length_of_BD_l18_18930


namespace book_original_price_l18_18293

noncomputable def original_price : ℝ := 420 / 1.40

theorem book_original_price (new_price : ℝ) (percentage_increase : ℝ) : 
  new_price = 420 → percentage_increase = 0.40 → original_price = 300 :=
by
  intros h1 h2
  exact sorry

end book_original_price_l18_18293


namespace arrange_f_values_l18_18073

noncomputable def f : ℝ → ℝ := sorry -- Assuming the actual definition is not necessary

-- The function f is even
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- The function f is strictly decreasing on (-∞, 0)
def strictly_decreasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 → (x1 < x2 ↔ f x1 > f x2)

theorem arrange_f_values (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_decreasing : strictly_decreasing_on_negative f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  -- The actual proof would go here.
  sorry

end arrange_f_values_l18_18073


namespace rectangle_cos_angle_BOD_l18_18245

theorem rectangle_cos_angle_BOD 
  (A B C D O : Point)
  (h_rect : Rectangle A B C D)
  (h_intersect : line_intersection (segment A C) (segment B D) = O) 
  (h_AC : distance A C = 26)
  (h_BD : distance B D = 14) : 
  cos (angle B O D) = 1 / 2 :=
sorry

end rectangle_cos_angle_BOD_l18_18245


namespace intersection_eq_l18_18856

def A : Set ℤ := {m | m ≤ -3 ∨ m ≥ 2}
def B : Set ℕ := {n | -1 ≤ n ∧ n < 3}
def negA : Set ℤ := {m | ¬ (m ≤ -3 ∨ m ≥ 2)}

theorem intersection_eq : B ∩ (negA : Set ℕ) = {0, 1} := by
  sorry

end intersection_eq_l18_18856


namespace hyperbola_eccentricity_l18_18492

variables {a b c : ℝ} (h : b > a ∧ a > 0)
variables (h1 : 0 < b) (h2 : a > 0) (h3 : c^2 = a^2 + b^2)

def hyperbola : Prop := ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1

def line (l : ℝ → ℝ → Prop) : Prop := l a 0 ∧ l 0 b

def dist_to_origin (d : ℝ) : Prop := d = sqrt 3 / 4 * c

theorem hyperbola_eccentricity (h : hyperbola) (l : ∀ x y : ℝ, line (λ x y, b * x + a * y - a * b = 0))
                             (d : dist_to_origin (abs (a * b) / sqrt (a^2 + b^2)))
                             (hf : c^2 = a^2 + b^2) : 
                             ∃ e : ℝ, e = c / a ∧ e = 2 :=
sorry

end hyperbola_eccentricity_l18_18492


namespace johnny_age_multiple_l18_18339

theorem johnny_age_multiple
  (current_age : ℕ)
  (age_in_2_years : ℕ)
  (age_3_years_ago : ℕ)
  (k : ℕ)
  (h1 : current_age = 8)
  (h2 : age_in_2_years = current_age + 2)
  (h3 : age_3_years_ago = current_age - 3)
  (h4 : age_in_2_years = k * age_3_years_ago) :
  k = 2 :=
by
  sorry

end johnny_age_multiple_l18_18339


namespace student_good_probability_l18_18534

-- Defining the conditions as given in the problem
def P_A1 := 0.25          -- Probability of selecting a student from School A
def P_A2 := 0.4           -- Probability of selecting a student from School B
def P_A3 := 0.35          -- Probability of selecting a student from School C

def P_B_given_A1 := 0.3   -- Probability that a student's level is good given they are from School A
def P_B_given_A2 := 0.6   -- Probability that a student's level is good given they are from School B
def P_B_given_A3 := 0.5   -- Probability that a student's level is good given they are from School C

-- Main theorem statement
theorem student_good_probability : 
  P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3 = 0.49 := 
by sorry

end student_good_probability_l18_18534


namespace radius_of_circle_l18_18619
open Real

theorem radius_of_circle (a b r : ℝ) (h1 : a + b = 6 * r) (h2 : 1/2 * a * b = 27) : r = 3 := by
  have area_eq : 1/2 * a * b = 3 * r^2 := by
    sorry
  have eq_r_squared : 3 * r^2 = 27 := by
    sorry
  show r = 3
  sorry

end radius_of_circle_l18_18619


namespace max_l_a_l18_18844

theorem max_l_a (a : ℝ) (l : ℝ) (f : ℝ → ℝ) 
  (h_f_def : ∀ x, f(x) = a * x^2 + 8 * x + 3)
  (h_a_neg : a < 0)
  (h_ineq : ∀ x ∈ Set.Icc (0 : ℝ) l, |f(x)| ≤ 5)
  (h_l_maximized : ∀ b : ℝ, b ≠ a → (∀ x ∈ Set.Icc (0 : ℝ) l, |f(x)| ≤ 5) → l ≤ l)
  : a = -8 :=
  sorry

end max_l_a_l18_18844


namespace coordinate_equations_and_product_l18_18474

-- Definitions of given conditions
def pole_at_origin := (0, 0)
def polar_axis := ∀ θ, θ = 0 → (1, 0)

def polar_equation_circle_C (θ : ℝ) : ℝ := 4 * (Float.cos θ)
def parametric_equation_line_l (t : ℝ) : (ℝ × ℝ) :=
  (- (3 / 5) * t + 2, (4 / 5) * t + 1)

def point_P : (ℝ × ℝ) := (2, 1)

-- Definition to prove Cartesian coordinate equation of line l
def cartesian_equation_line_l (x y : ℝ) : Prop :=
  4 * x + 3 * y - 11 = 0

-- Definition to prove Cartesian coordinate equation of circle C
def cartesian_equation_circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x = 0

-- Main theorem
theorem coordinate_equations_and_product :
  (∀ x y : ℝ, (∃ t : ℝ, parametric_equation_line_l t = (x, y)) → cartesian_equation_line_l x y)
  ∧ (∀ x y : ℝ, (polar_equation_circle_C x = y) → cartesian_equation_circle_C x y)
  ∧ (∃ A B : ℝ × ℝ, parametric_equation_line_l A = (2, 1) ∧
    parametric_equation_line_l B = (2, 1) → 
    (abs (dist point_P A) * abs (dist point_P B) = 3)) :=
by
  sorry

end coordinate_equations_and_product_l18_18474


namespace jim_mpg_is_5_l18_18547

variable (total_gas_capacity : ℕ)
variable (gas_fraction_left : ℚ)
variable (distance_to_work_one_way : ℕ)

def total_distance_travelled (d : ℕ) : ℕ := 2 * d

def gas_used (capacity : ℕ) (fraction_left : ℚ) : ℚ :=
  capacity * (1 - fraction_left)

def miles_per_gallon (total_distance : ℕ) (gallons_used : ℚ) : ℚ :=
  total_distance / gallons_used

theorem jim_mpg_is_5 
  (hg_cap : total_gas_capacity = 12)
  (hf_left : gas_fraction_left = 2 / 3)
  (d_work : distance_to_work_one_way = 10) :
  miles_per_gallon (total_distance_travelled d_work) (gas_used total_gas_capacity gas_fraction_left) = 5 := by
  -- The proof goes here
  sorry

end jim_mpg_is_5_l18_18547


namespace judy_pencil_cost_l18_18935

theorem judy_pencil_cost 
  (pencils_per_week : ℕ)
  (days_per_week : ℕ)
  (pack_cost : ℕ)
  (pack_size : ℕ)
  (total_days : ℕ)
  (pencil_usage : pencils_per_week = 10)
  (school_days : days_per_week = 5)
  (cost_per_pack : pack_cost = 4)
  (pencils_per_pack : pack_size = 30)
  (duration : total_days = 45) : 
  ∃ (total_cost : ℕ), total_cost = 12 :=
sorry

end judy_pencil_cost_l18_18935


namespace cubic_function_increasing_l18_18977

theorem cubic_function_increasing (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = x^3) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ :=
by 
  intros x₁ x₂ h₀
  have h₁ : f x₁ = x₁ ^ 3 := h x₁
  have h₂ : f x₂ = x₂ ^ 3 := h x₂
  rw [h₁, h₂]
  sorry only at the end to indicate the partial proof as the steps are not required.

end cubic_function_increasing_l18_18977


namespace find_t_l18_18858

-- Definitions of the vectors and parallel condition
def a : ℝ × ℝ := (-1, 1)
def b (t : ℝ) : ℝ × ℝ := (3, t)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v ∨ v = k • u

-- The theorem statement
theorem find_t (t : ℝ) (h : is_parallel (b t) (a + b t)) : t = -3 := by
  sorry

end find_t_l18_18858


namespace correct_calculated_value_l18_18508

theorem correct_calculated_value (x : ℤ) (h : x - 749 = 280) : x + 479 = 1508 :=
by 
  sorry

end correct_calculated_value_l18_18508


namespace derek_dogs_l18_18773

theorem derek_dogs (d c : ℕ) (h1 : d = 90) 
  (h2 : c = d / 3) 
  (h3 : c + 210 = 2 * (d + 120 - d)) : 
  d + 120 - d = 120 :=
by
  sorry

end derek_dogs_l18_18773


namespace continuous_linear_function_l18_18131

theorem continuous_linear_function {f : ℝ → ℝ} (h_cont : Continuous f) 
  (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_a_half : a < 1/2) (h_b_half : b < 1/2) 
  (h_eq : ∀ x : ℝ, f (f x) = a * f x + b * x) : 
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ (k * k - a * k - b = 0) := 
sorry

end continuous_linear_function_l18_18131


namespace max_gcd_of_linear_combinations_l18_18001

theorem max_gcd_of_linear_combinations (a b c : ℕ) (h1 : a + b + c ≤ 3000000) (h2 : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  gcd (a * b + 1) (gcd (a * c + 1) (b * c + 1)) ≤ 998285 :=
sorry

end max_gcd_of_linear_combinations_l18_18001


namespace bisectors_intersect_in_single_point_l18_18185

-- Definition of a convex pentagon with all interior angles equal to 108 degrees.
structure ConvexPentagon (A B C D E : Type) :=
(interior_angles_eq : ∀ θ ∈ { angle A B C, angle B C D, angle C D E, angle D E A, angle E A B }, θ = 108)

-- The definition of perpendicular bisectors and angle bisectors.
def perpendicular_bisector (AB : Type) : Type := sorry
def angle_bisector (CDE : Type) : Type := sorry

-- Main theorem to be proved.
theorem bisectors_intersect_in_single_point
  (A B C D E P : Type)
  (pentagon : ConvexPentagon A B C D E)
  (P_on_perpendicular_bisector_EA : P ∈ perpendicular_bisector (segment A E))
  (P_on_perpendicular_bisector_BC : P ∈ perpendicular_bisector (segment B C))
  (P_on_angle_bisector_CDE : P ∈ angle_bisector (angle C D E)) :
  P = P :=
sorry

end bisectors_intersect_in_single_point_l18_18185


namespace f_of_f_of_fourth_l18_18490

def f (x : ℝ) : ℝ := 
  if x > 0 then real.log x / real.log 2 
  else 3^x

theorem f_of_f_of_fourth : f (f (1 / 4)) = 1 / 9 := by
  sorry

end f_of_f_of_fourth_l18_18490


namespace train_speed_calculation_l18_18026

variable (p : ℝ) (h_p : p > 0)

/-- The speed calculation of a train that covers 200 meters in p seconds is correctly given by 720 / p km/hr. -/
theorem train_speed_calculation (h_p : p > 0) : (200 / p * 3.6 = 720 / p) :=
by
  sorry

end train_speed_calculation_l18_18026


namespace problem_l18_18140

noncomputable theory

def f (x : ℝ) : ℝ := cos x ^ 2 - sqrt 3 * sin x * cos x - 1 / 2

theorem problem
  (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π)
  (h_transform : ∀ x, f x = A * sin (ω * x + φ))
  (a : ℝ) (A_angle : A = φ) (a_length : a = 2) :
  A = 1 ∧ ω = 2 ∧ φ = 5 * π / 6 ∧ 
  (∀ k : ℤ, ∀ x, (k * π - 2 * π / 3 ≤ x ∧ x ≤ k * π - π / 6) → f' x ≥ 0) ∧ 
  (angle_B = π / 12 ∧ side_c = sqrt 6 - sqrt 2) :=
begin
  sorry
end

end problem_l18_18140


namespace polynomial_divisibility_l18_18292

noncomputable def A := sorry
noncomputable def B := sorry
noncomputable def C := sorry

theorem polynomial_divisibility (A B C : ℝ) 
    (h : ∀ x : ℂ, x^2 + x + 1 = 0 → x^103 + C * x^2 + A * x + B = 0) : 
    A + B + C = 3 * C - 1 :=
by
  sorry

end polynomial_divisibility_l18_18292


namespace sum_first_6_is_correct_l18_18496

namespace ProofProblem

def sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 1) ∧ ∀ n : ℕ, n ≥ 2 → a (n - 1) = 2 * a n

def sum_first_6 (a : ℕ → ℚ) : ℚ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_first_6_is_correct (a : ℕ → ℚ) (h : sequence a) :
  sum_first_6 a = 63 / 32 :=
sorry

end ProofProblem

end sum_first_6_is_correct_l18_18496


namespace plane_speed_still_air_l18_18707

variable (p : ℝ)

def speed_with_wind := p + 20
def speed_against_wind := p - 20
def time_with_wind := 400 / speed_with_wind
def time_against_wind := 320 / speed_against_wind

theorem plane_speed_still_air (h : time_with_wind p = time_against_wind p) : p = 180 :=
by sorry

end plane_speed_still_air_l18_18707


namespace average_of_shots_l18_18017

theorem average_of_shots :
  ∃ (s : Finset ℕ), s.card = 5 ∧ 
                    (median (s : Multiset ℕ)) = 8 ∧ 
                    (∃ x, mode (s : Multiset ℕ) = x ∧ x = 9) ∧
                    (finset_max s - finset_min s = 3) ∧ 
                    (s.sum / s.card : ℚ) = 7.8 := 
sorry

end average_of_shots_l18_18017


namespace find_m_solution_l18_18902

def lines_are_parallel (m : ℝ) : Prop :=
  x + (1 + m) * y - 2 = 0 → mx + 2 * y + 4 = 0 → (1 ≠ (m + 2)) ∧ 2 = m * (1 + m)

theorem find_m_solution (m : ℝ) (h1 : x + (1 + m) * y - 2 = 0) (h2 : mx + 2 * y + 4 = 0) :
  lines_are_parallel m → m = 1 :=
by {
  sorry
}

end find_m_solution_l18_18902


namespace discriminant_of_quadratic_eq_l18_18749

-- Define the quadratic equation's coefficients
def a : ℝ := 2
def b : ℝ := (3 - 1/2)
def c : ℝ := 1/2

-- Define the discriminant formula
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The theorem to prove the discriminant of the given quadratic equation
theorem discriminant_of_quadratic_eq : discriminant a b c = 9/4 := by
  sorry

end discriminant_of_quadratic_eq_l18_18749


namespace root_relation_l18_18261

theorem root_relation (a b x y : ℝ)
  (h1 : x + y = a)
  (h2 : (1 / x) + (1 / y) = 1 / b)
  (h3 : x = 3 * y)
  (h4 : y = a / 4) :
  b = 3 * a / 16 :=
by
  sorry

end root_relation_l18_18261


namespace domain_log_function_l18_18998

theorem domain_log_function :
  { x : ℝ | 12 + x - x^2 > 0 } = { x : ℝ | -3 < x ∧ x < 4 } :=
sorry

end domain_log_function_l18_18998


namespace circles_through_two_points_in_4x4_grid_l18_18284

noncomputable def number_of_circles (n : ℕ) : ℕ :=
  if n = 4 then
    52
  else
    sorry

theorem circles_through_two_points_in_4x4_grid :
  number_of_circles 4 = 52 :=
by
  exact rfl  -- Reflexivity of equality shows the predefined value of 52

end circles_through_two_points_in_4x4_grid_l18_18284


namespace sum_of_integers_from_100_to_1999_l18_18665

-- Define the lower and upper bounds of the range
def lower_bound : ℕ := 100
def upper_bound : ℕ := 1999

-- Define the number of terms in the range
def num_terms : ℕ := upper_bound - lower_bound + 1

-- Define the average of the first and last term of the sequence
def average : ℝ := (lower_bound + upper_bound) / 2.0

-- Define the sum of the sequence as the product of the average and the number of terms
def sum_of_sequence : ℝ := average * num_terms

-- Theorem statement: Prove that the sum of all integers from 100 to 1999 is 1994050
theorem sum_of_integers_from_100_to_1999 : sum_of_sequence = 1994050 := by
  -- Proof is omitted
  sorry

end sum_of_integers_from_100_to_1999_l18_18665


namespace count_congruent_to_4_mod_7_l18_18878

theorem count_congruent_to_4_mod_7 : 
  ∃ (n : ℕ), 
  n = 71 ∧ 
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 70 → ∃ m : ℕ, m = 4 + 7 * k ∧ m < 500 := 
by
  sorry

end count_congruent_to_4_mod_7_l18_18878


namespace feed_campers_proof_l18_18973

noncomputable def total_fish_weight : ℝ := 8 + 6 * 2 + 2 * 12
def wastage_percentage : ℝ := 0.20
noncomputable def wastage_weight : ℝ := wastage_percentage * total_fish_weight
noncomputable def edible_fish_weight : ℝ := total_fish_weight - wastage_weight
def adult_consumption : ℝ := 3
def child_consumption : ℝ := 1
def camper_ratio_adult : ℚ := 2
def camper_ratio_child : ℚ := 5
def max_campers : ℕ := 12
def campers_total (x : ℕ) : ℕ := (2 * x + 5 * x)
def fish_required (adults children : ℕ) : ℝ := adults * adult_consumption + children * child_consumption

theorem feed_campers_proof :
  ∃ (x : ℕ), campers_total x ≤ max_campers ∧ 
             fish_required (2 * x) (5 * x) ≤ edible_fish_weight ∧ 
             2 * x = 2 ∧ 5 * x = 5 := 
by {
  use 1,
  split; norm_num,
  sorry
}

end feed_campers_proof_l18_18973


namespace square_ABCD_BF_l18_18614

theorem square_ABCD_BF (AB : ℝ) (O : ℝ × ℝ) (E F : ℝ × ℝ) (AE BF EF : ℝ) (p q r : ℤ) (h_AB: AB = 900) 
(h_O: O = (450, 450)) 
(h_EF: EF = 350) 
(h_ae_bf: AE < BF) 
(h_angle_EOF: ∠(O, E, F) = 60)
(h_condition: ∃ p q r : ℤ, BF = p + q * real.sqrt r ∧ r > 0 ∧ ∀ k : ℤ, k^2 ∣ r → k = 1):
  p + q + r = 300 := 
sorry

end square_ABCD_BF_l18_18614


namespace example_theorem_l18_18039

noncomputable def Angle45 := π / 4
noncomputable def Angle90 := π / 2

structure Tetrahedron :=
(A B C D : ℝ × ℝ × ℝ)
(AD : ℝ)
(ABC_perpendicular_BD : ℝ)
(angle_ABD : ℝ)
(angle_CBD : ℝ)
(angle_ABC : ℝ)

def example : Tetrahedron := {
  A := (1 / real.sqrt 2, 0, 0),
  B := (0, 0, 0),
  C := (0, 1 / real.sqrt 2, 0),
  D := (0, 0, 1 / real.sqrt 2),
  AD := 1,
  ABC_perpendicular_BD := 0,
  angle_ABD := Angle45,
  angle_CBD := Angle45,
  angle_ABC := Angle90
}

theorem example_theorem (T : Tetrahedron) : 
  let A := T.A 
  let B := T.B 
  let C := T.C 
  let D := T.D 
  let Q := (0, 0, -1 / real.sqrt 2) 
  let R := (0, real.sqrt 2, -1 / real.sqrt 2) in
  dist A Q = 1 ∧ -- AQ = 1
  dist A R = real.sqrt 3 ∧ -- AR = √3
  dist Q R = real.sqrt 2 ∧ -- QR = √2
  ∠ADC = Angle90 -- ∠ADC = 90°
:=
by sorry

end example_theorem_l18_18039


namespace positive_value_of_A_l18_18213

theorem positive_value_of_A (A : ℝ) :
  (A ^ 2 + 7 ^ 2 = 200) → A = Real.sqrt 151 :=
by
  intros h
  sorry

end positive_value_of_A_l18_18213


namespace abs_sqrt_expression_eq_neg_one_l18_18667

theorem abs_sqrt_expression_eq_neg_one (a : ℝ) (h : a < -1) : |1 + a| - real.sqrt (a ^ 2) = -1 :=
by
  sorry

end abs_sqrt_expression_eq_neg_one_l18_18667


namespace total_amount_after_refunds_and_discounts_l18_18755

-- Definitions
def individual_bookings : ℤ := 12000
def group_bookings_before_discount : ℤ := 16000
def discount_rate : ℕ := 10
def refund_individual_1 : ℤ := 500
def count_refund_individual_1 : ℕ := 3
def refund_individual_2 : ℤ := 300
def count_refund_individual_2 : ℕ := 2
def total_refund_group : ℤ := 800

-- Calculation proofs
theorem total_amount_after_refunds_and_discounts : 
(individual_bookings + (group_bookings_before_discount - (discount_rate * group_bookings_before_discount / 100))) - 
((count_refund_individual_1 * refund_individual_1) + (count_refund_individual_2 * refund_individual_2) + total_refund_group) = 23500 := by
    sorry

end total_amount_after_refunds_and_discounts_l18_18755


namespace drums_per_day_l18_18869

theorem drums_per_day (total_drums : Nat) (days : Nat) (total_drums_eq : total_drums = 6264) (days_eq : days = 58) :
  total_drums / days = 108 :=
by
  sorry

end drums_per_day_l18_18869


namespace find_original_number_l18_18642

variable (r : ℝ)

def val1 := 1.20 * r
def val2 := 1.35 * r
def val3 := 0.50 * r

theorem find_original_number (h : (val1 - r) + (val2 - r) - (r - val3) = 110) : 
  r = 2200 :=
by 
  sorry

end find_original_number_l18_18642


namespace sum_of_digits_of_N_eq_14_l18_18036

theorem sum_of_digits_of_N_eq_14 :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ (N % 10 + N / 10 % 10 = 14) :=
by
  sorry

end sum_of_digits_of_N_eq_14_l18_18036


namespace work_completion_days_l18_18365

theorem work_completion_days (A B C : ℕ) 
  (hA : A = 4) (hB : B = 8) (hC : C = 8) : 
  2 = 1 / (1 / A + 1 / B + 1 / C) :=
by
  -- skip the proof for now
  sorry

end work_completion_days_l18_18365


namespace number_of_subsets_with_odd_sum_l18_18414

def odd (n : ℕ) : Prop := n % 2 = 1
def even (n : ℕ) : Prop := n % 2 = 0

def my_set : finset ℕ := {87, 92, 98, 101, 135, 142, 168}

def odd_from_three (a b c : ℕ) : Prop := odd (a + b + c)

noncomputable def count_subsets_odd_sum : ℕ :=
  (finset.powerset_len 3 my_set).filter (λ s, ∃ a b c, s = {a, b, c} ∧ odd_from_three a b c).card

theorem number_of_subsets_with_odd_sum : count_subsets_odd_sum = 19 := sorry

end number_of_subsets_with_odd_sum_l18_18414


namespace sum_segments_eq_three_l18_18647

theorem sum_segments_eq_three {n : ℕ} (h : n ≥ 2) (a : Fin n → ℝ) (ha : ∑ (i : Fin n), a i = 3) : True :=
by sorry

end sum_segments_eq_three_l18_18647


namespace sum_of_other_endpoint_coordinates_l18_18591

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem sum_of_other_endpoint_coordinates :
  ∀ (x y : ℝ), midpoint (-2, 5) (x, y) = (1, 0) → x + y = -1 :=
by
  rintros x y h
  have h1 : (x + -2) / 2 = 1 := congr_arg Prod.fst h
  have h2 : (y + 5) / 2 = 0 := congr_arg Prod.snd h
  sorry

end sum_of_other_endpoint_coordinates_l18_18591


namespace handshake_count_l18_18359

-- Define the number of participants
def n : ℕ := 12

-- Define the combination formula for choosing 2 people out of n
def combination (n k : ℕ) : ℕ := nat.choose n k

-- The theorem stating the number of handshakes
theorem handshake_count (n : ℕ) : combination n 2 = 66 := by
  -- Given n = 12
  have h : n = 12 := rfl
  -- Compute the combination
  rw h
  exact nat.choose_succ_succ 11 1 sorry -- Substitute and simplify for binomial coefficient

end handshake_count_l18_18359


namespace planes_are_perpendicular_l18_18571

-- Definitions only include conditions identified in part a)

variable (Line Plane : Type)

variable (l : Line) (alpha beta : Plane)

-- Conditions:
axiom parallel_l_alpha : l ∥ alpha
axiom perpendicular_l_beta : l ⟂ beta

-- The following Lean statement encapsulates the problem we have:
theorem planes_are_perpendicular 
  (h₁ : l ∥ alpha) 
  (h₂ : l ⟂ beta) 
  : alpha ⟂ beta :=
sorry

end planes_are_perpendicular_l18_18571


namespace problem_solution_l18_18569

noncomputable def f (x : ℕ) : ℝ := sorry

axiom condition1 : ∀ m n : ℕ, m > 0 → n > 0 → f(m + n) = f(m) * f(n)
axiom f_one : f(1) = 2

theorem problem_solution : 
  (∑ i in finset.range 1008, (f((2 * (i + 1))) / f((2 * i) + 1))) = 2016 :=
sorry

end problem_solution_l18_18569


namespace functionA_neither_odd_nor_even_l18_18734

def functionA (x : ℝ) : ℝ := 1 / x - 3 ^ x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem functionA_neither_odd_nor_even :
  ¬ is_even functionA ∧ ¬ is_odd functionA := by
  sorry

end functionA_neither_odd_nor_even_l18_18734


namespace range_of_b_not_strictly_decreasing_l18_18850

def f (b x : ℝ) : ℝ := -x^3 + b*x^2 - (2*b + 3)*x + 2 - b

theorem range_of_b_not_strictly_decreasing :
  {b : ℝ | ¬(∀ (x1 x2 : ℝ), x1 < x2 → f b x1 > f b x2)} = {b | b < -1 ∨ b > 3} :=
by
  sorry

end range_of_b_not_strictly_decreasing_l18_18850


namespace smallest_integer_in_consecutive_set_l18_18914

theorem smallest_integer_in_consecutive_set (n : ℤ) (h : n + 6 < 2 * (n + 3)) : n > 0 := by
  sorry

end smallest_integer_in_consecutive_set_l18_18914


namespace majority_of_votes_l18_18917

theorem majority_of_votes (V : ℝ) (hV : V = 455) (p : ℝ) (hp : p = 0.70) :
  let winning_votes := p * V,
      losing_votes := (1 - p) * V,
      majority := winning_votes - losing_votes
  in majority = 182 :=
by
  sorry

end majority_of_votes_l18_18917


namespace complex_solutions_eq_two_l18_18436

theorem complex_solutions_eq_two : 
  ∀ (z : ℂ), (z^3 - 1) = 0 → (z^2 - z - 2) ≠ 0 → 
    (∃ z1 z2 : ℂ, z1 ≠ z2 ∧ z^3 - 1 = 0 ∧ z1 ∈ ℂ ∧ z2 ∈ ℂ) :=
by
  sorry

end complex_solutions_eq_two_l18_18436


namespace num_pos_integers_congruent_to_4_mod_7_l18_18882

theorem num_pos_integers_congruent_to_4_mod_7 (n : ℕ) (h1 : n < 500) (h2 : ∃ k : ℕ, n = 7 * k + 4) : 
  ∃ total : ℕ, total = 71 :=
sorry

end num_pos_integers_congruent_to_4_mod_7_l18_18882


namespace total_letters_sent_l18_18250

-- Define the number of letters sent in each month
def letters_in_January : ℕ := 6
def letters_in_February : ℕ := 9
def letters_in_March : ℕ := 3 * letters_in_January

-- Theorem statement: the total number of letters sent across the three months
theorem total_letters_sent :
  letters_in_January + letters_in_February + letters_in_March = 33 := by
  sorry

end total_letters_sent_l18_18250


namespace triangles_exceed_million_l18_18282

/-- 
Given an equilateral triangle ABC divided into N convex polygons, 
and each line intersects no more than 40 of these polygons, 
prove that N can exceed one million.
-/
theorem triangles_exceed_million (N : ℕ) (ABC : Triangle) 
  (divided : ∀ t : Triangle, divides_into_convex_polygons t N)
  (intersects : ∀ l : Line, N_intersecting_polygons l ≤ 40) : 
  N > 1000000 :=
sorry

end triangles_exceed_million_l18_18282


namespace number_of_selections_l18_18449

theorem number_of_selections (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 3) : 
  ∃ (total_selections : ℕ), 
    (total_selections = (Nat.choose boys 3) * (Nat.choose girls 1) + 
                        (Nat.choose boys 2) * (Nat.choose girls 2) + 
                        (Nat.choose boys 1) * (Nat.choose girls 3)) ∧ 
    total_selections = 34 := 
by
  replace h_boys := h_boys.symm
  replace h_girls := h_girls.symm
  have comb1 := Nat.choose 4 3 * Nat.choose 3 1
  have comb2 := Nat.choose 4 2 * Nat.choose 3 2
  have comb3 := Nat.choose 4 1 * Nat.choose 3 3
  let total_selections := comb1 + comb2 + comb3
  use total_selections
  split
  . simp [total_selections, comb1, comb2, comb3]
  . norm_num
    exact h_boys
    exact h_girls
    sorry -- skipping the proof

end number_of_selections_l18_18449


namespace smallest_reducible_fraction_l18_18440

open Nat

-- Define gcd using Euclidean algorithm to confirm it's done right
def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

-- Define the specific reducibility condition
def is_reducible (n : ℕ) : Prop :=
  gcd (n - 17) (7 * n + 8) > 1

-- State the theorem
theorem smallest_reducible_fraction : ∃ n : ℕ, n > 0 ∧ is_reducible n ∧ ∀ m : ℕ, is_reducible m → m > 0 → n ≤ m ∧ n = 144 :=
by
  sorry

end smallest_reducible_fraction_l18_18440


namespace Wayne_blocks_l18_18658

theorem Wayne_blocks (initial_blocks : ℕ) (additional_blocks : ℕ) (total_blocks : ℕ) 
  (h1 : initial_blocks = 9) (h2 : additional_blocks = 6) 
  (h3 : total_blocks = initial_blocks + additional_blocks) : 
  total_blocks = 15 :=
by {
  -- h1: initial_blocks = 9
  -- h2: additional_blocks = 6
  -- h3: total_blocks = initial_blocks + additional_blocks
  sorry
}

end Wayne_blocks_l18_18658


namespace cos_B_when_a_eq_b_eq_2_area_when_B_eq_90_a_eq_sqrt2_l18_18905

-- Defining the elements in the triangle and the condition
variables {A B C : ℝ} -- Angles in the triangle, but they are not directly used
variables {a b c : ℝ} -- Sides opposite to angles A, B, and C

-- Condition given in the problem: sin^2 B = 2 * sin A * sin C
axiom sin_squared_B_eq : sin B * sin B = 2 * sin A * sin C

-- Part I: Prove that if a = b = 2, then cos B = 1/4
theorem cos_B_when_a_eq_b_eq_2 (h1 : a = 2) (h2 : b = 2) : 
  cos B = 1 / 4 := 
sorry

-- Part II: Prove that if B = 90 degrees and a = sqrt(2), then the area of triangle ABC is 1
theorem area_when_B_eq_90_a_eq_sqrt2 (h1 : B = π / 2) (h2 : a = real.sqrt 2) : 
  (1 / 2) * a * c = 1 := 
sorry

end cos_B_when_a_eq_b_eq_2_area_when_B_eq_90_a_eq_sqrt2_l18_18905


namespace lengths_of_segments_l18_18963

theorem lengths_of_segments
  (a b c : ℝ)
  (E F D : ℝ → ℝ → ℝ)
  (AE BF CD : ℝ)
  (h1 : ∀ x : ℝ, E (x / 2 + x / 2) = x)
  (h2 : ∀ x : ℝ, F (x / 2 + x / 2) = x)
  (h3 : ∀ x : ℝ, D (x / 2 + x / 2) = x) :
  AE = (b + c - a) / 2 ∧
  BF = (a + c - b) / 2 ∧
  CD = (a + b - c) / 2 :=
sorry

end lengths_of_segments_l18_18963


namespace correct_statements_l18_18340

theorem correct_statements :
  (∀ α : ℝ, 0 ≤ α ∧ α < real.pi → true) ∧
  (¬ (∀ α : ℝ, (α ∈ set.Icc 0 real.pi ∧ tan α = slope α) → true)) ∧
  (∀ α : ℝ, true → (α = real.pi_div_two → false) → true) ∧
  (¬ (∀ α : ℝ, α ∈ set.Icc 0 real.pi → tan α increases_with α)) → 
  (A ∧ C) :=
begin
  sorry
end

end correct_statements_l18_18340


namespace find_real_solution_to_given_equation_l18_18095

noncomputable def sqrt_96_minus_sqrt_84 : ℝ := Real.sqrt 96 - Real.sqrt 84

theorem find_real_solution_to_given_equation (x : ℝ) (hx : x + 4 ≥ 0) :
  x^2 + 6 * x + 6 * x * Real.sqrt (x + 4) = 60 ↔ x = sqrt_96_minus_sqrt_84 := 
by
  sorry

end find_real_solution_to_given_equation_l18_18095


namespace select_event_organization_committee_l18_18697

noncomputable def number_of_ways_to_select_committee (n : ℕ) : ℕ :=
  (nat.choose n 4)

-- We need to prove that if there are 20 ways to select the three-person cleanup committee,
-- then the number of ways to select the four-person event organization committee is 15.
theorem select_event_organization_committee :
  ∃ n : ℕ, (nat.choose n 3 = 20) → (number_of_ways_to_select_committee n = 15) :=
begin
  sorry
end

end select_event_organization_committee_l18_18697


namespace fibonacci_x_value_l18_18928

theorem fibonacci_x_value :
  ∃ x : ℕ, 
    (∀ n ≥ 2, ∀ a1 a2 : ℕ, (fib_seq n = fib_seq (n-1) + fib_seq (n-2))) → 
    (fib_seq 6 = 13) → 
    (fib_seq 7 = x) →
    x = 21 :=
by
  sorry

-- Definitions for Fibonacci sequence
def fib_seq : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => fib_seq (n + 1) + fib_seq n

end fibonacci_x_value_l18_18928


namespace total_rebate_correct_l18_18381

def prices : List ℝ := [28, 35, 40, 45, 50]
def rebates : List ℝ := [0.10, 0.12, 0.15, 0.18, 0.20]

def calculate_rebate (price rebate: ℝ) : ℝ :=
  price * rebate

def total_rebate : ℝ :=
  List.sum (List.map (λ (pr_re : ℝ × ℝ), calculate_rebate pr_re.fst pr_re.snd) (List.zip prices rebates))

theorem total_rebate_correct : total_rebate = 31.1 :=
by 
  sorry

end total_rebate_correct_l18_18381


namespace train_speed_l18_18388

theorem train_speed (length : ℕ) (time : ℕ) (v : ℕ)
  (h1 : length = 750)
  (h2 : time = 1)
  (h3 : v = (length + length) / time)
  (h4 : v = 1500) :
  (v * 60 / 1000 = 90) :=
by
  sorry

end train_speed_l18_18388


namespace triangle_angle_measurements_l18_18818

theorem triangle_angle_measurements
  (A B C P : Type)
  [is_triangle A B C]
  (angle_ACB : ℝ)
  (is_50_degrees : angle_ACB = 50)
  (altitude_from_A : altitude_segment A P)
  (bisector_angle_ABC : angle_bisector B P)
  (angle_APB : ℝ)
  (is_105_degrees : angle_APB = 105)
  : angle_measure A B C = 100 ∧ angle_measure B A C = 30 := by
  sorry

end triangle_angle_measurements_l18_18818


namespace find_a5_l18_18924

noncomputable def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
a₁ + (n - 1) * d

theorem find_a5 (a₁ d : ℚ) (h₁ : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 5 - arithmetic_sequence a₁ d 8 = 1)
(h₂ : arithmetic_sequence a₁ d 9 - arithmetic_sequence a₁ d 2 = 5) :
arithmetic_sequence a₁ d 5 = 6 :=
sorry

end find_a5_l18_18924


namespace division_reciprocal_multiplication_l18_18790

theorem division_reciprocal_multiplication : (4 / (8 / 13 : ℚ)) = (13 / 2 : ℚ) := 
by
  sorry

end division_reciprocal_multiplication_l18_18790


namespace garage_sale_total_l18_18318

theorem garage_sale_total (treadmill chest_of_drawers television total_sales : ℝ)
  (h1 : treadmill = 100) 
  (h2 : chest_of_drawers = treadmill / 2) 
  (h3 : television = treadmill * 3) 
  (partial_sales : ℝ) 
  (h4 : partial_sales = treadmill + chest_of_drawers + television) 
  (h5 : partial_sales = total_sales * 0.75) : 
  total_sales = 600 := 
by
  sorry

end garage_sale_total_l18_18318


namespace calculate_expression_l18_18053

theorem calculate_expression : -1^2021 + 1^2022 = 0 := by
  sorry

end calculate_expression_l18_18053


namespace problem_1_problem_2_l18_18846

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 3) 

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

theorem problem_1 (k : ℤ) : 
  is_monotonically_decreasing 
    f 
    (2 * k * π + π / 6) 
    (2 * k * π + 7 * π / 6) := 
begin
  sorry
end

variables (α : ℝ)
axiom α_cond : α ∈ (π / 6, 2 * π / 3)
axiom f_alpha : f α = 6 / 5

theorem problem_2 :
  f (α - π / 6) = (4 + 3 * sqrt 3) / 5 :=
begin
  sorry
end

end problem_1_problem_2_l18_18846


namespace result_of_operation_l18_18663

variable (x : ℝ)

def operation (x : ℝ) : ℝ := 40 + x * 12 / (180 / 3)

theorem result_of_operation (h : operation x = 41) : x = 5 :=
sorry

end result_of_operation_l18_18663


namespace problem_l18_18835

def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {x | 3^x > 1}

theorem problem : A ∩ (Set.compl B) = {x : ℝ | -2 < x ∧ x ≤ 0} :=
by sorry

end problem_l18_18835


namespace subsequence_converges_to_limit_l18_18840

-- Define the sequence u_n and its convergence to l
variables {u : ℕ → ℝ} {l : ℝ}
hypothesis h1 : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n - l| < ε

-- Define the strictly increasing function φ and subsequence v_n
variables {φ : ℕ → ℕ}
hypothesis h2 : ∀ n m : ℕ, (n < m) → (φ n < φ m)
definition v (n : ℕ) : ℝ := u (φ n)

-- The theorem stating that the subsequence v_n also converges to l
theorem subsequence_converges_to_limit : ∀ ε > 0, ∃ N' : ℕ, ∀ n ≥ N', |v n - l| < ε := 
by {
  -- The proof would be written here, but we use sorry to indicate that the proof is omitted.
  sorry
}

end subsequence_converges_to_limit_l18_18840


namespace customers_not_tipping_l18_18401

theorem customers_not_tipping (number_of_customers tip_per_customer total_earned_in_tips : ℕ)
  (h_number : number_of_customers = 7)
  (h_tip : tip_per_customer = 3)
  (h_earned : total_earned_in_tips = 6) :
  number_of_customers - (total_earned_in_tips / tip_per_customer) = 5 :=
by
  sorry

end customers_not_tipping_l18_18401


namespace solve_for_A_l18_18207

def hash (A B : ℝ) : ℝ := A^2 + B^2

theorem solve_for_A (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 :=
by
  sorry

end solve_for_A_l18_18207


namespace balance_after_transactions_l18_18235

variable (x : ℝ)

def monday_spent : ℝ := 0.525 * x
def tuesday_spent (remaining : ℝ) : ℝ := 0.106875 * remaining
def wednesday_spent (remaining : ℝ) : ℝ := 0.131297917 * remaining
def thursday_spent (remaining : ℝ) : ℝ := 0.040260605 * remaining

def final_balance (x : ℝ) : ℝ :=
  let after_monday := x - monday_spent x
  let after_tuesday := after_monday - tuesday_spent after_monday
  let after_wednesday := after_tuesday - wednesday_spent after_tuesday
  after_wednesday - thursday_spent after_wednesday

theorem balance_after_transactions (x : ℝ) :
  final_balance x = 0.196566478 * x :=
by
  sorry

end balance_after_transactions_l18_18235


namespace train_passes_tree_in_20_seconds_l18_18346

def train_passing_time 
  (length_of_train : ℕ)
  (speed_kmh : ℕ)
  (conversion_factor : ℚ) : ℚ :=
  length_of_train / (speed_kmh * conversion_factor)

theorem train_passes_tree_in_20_seconds 
  (length_of_train : ℕ := 350)
  (speed_kmh : ℕ := 63)
  (conversion_factor : ℚ := 1000 / 3600) : 
  train_passing_time length_of_train speed_kmh conversion_factor = 20 :=
  sorry

end train_passes_tree_in_20_seconds_l18_18346


namespace total_letters_correct_l18_18251

-- Define the conditions
def letters_January := 6
def letters_February := 9
def letters_March := 3 * letters_January

-- Definition of the total number of letters sent
def total_letters := letters_January + letters_February + letters_March

-- The statement we need to prove in Lean
theorem total_letters_correct : total_letters = 33 := 
by
  sorry

end total_letters_correct_l18_18251


namespace total_letters_sent_l18_18249

-- Define the number of letters sent in each month
def letters_in_January : ℕ := 6
def letters_in_February : ℕ := 9
def letters_in_March : ℕ := 3 * letters_in_January

-- Theorem statement: the total number of letters sent across the three months
theorem total_letters_sent :
  letters_in_January + letters_in_February + letters_in_March = 33 := by
  sorry

end total_letters_sent_l18_18249


namespace tan_quadratic_roots_sum_eq_pi_l18_18105

noncomputable def sum_of_roots_between_zero_and_pi : ℝ :=
  let r1 := (10 + Real.sqrt 88) / 2
  let r2 := (10 - Real.sqrt 88) / 2
  let α := Real.atan r1
  let β := Real.atan r2
  α + β

theorem tan_quadratic_roots_sum_eq_pi :
  let f : ℝ → ℝ := λ x, Real.tan x
  let r1 := (10 + Real.sqrt 88) / 2
  let r2 := (10 - Real.sqrt 88) / 2
  let α := Real.atan r1
  let β := Real.atan r2
  (0 ≤ α ∧ α ≤ Real.pi) ∧ (0 ≤ β ∧ β ≤ Real.pi) →
  α + β = Real.pi :=
by
  sorry

end tan_quadratic_roots_sum_eq_pi_l18_18105


namespace product_factors_l18_18751

theorem product_factors :
  (finset.range 11).prod (λ n, 1 - (1 / (n + 2 : ℝ))) = 1 / 12 :=
by sorry

end product_factors_l18_18751


namespace relationship_l18_18075

-- Definitions for the given points and their relationships with x and y.
def bool_eq {α : Type*} [DecidableEq α] (a b : α) : bool := a = b

section
variables (x y : ℕ → ℤ) -- x and y as sequences

-- Definitions for the points given in the table:
def points : Prop :=
  (x 0 = 0 ∧ y 0 = 0) ∧ 
  (x 1 = 1 ∧ y 1 = -15) ∧ 
  (x 2 = 2 ∧ y 2 = -40) ∧ 
  (x 3 = 3 ∧ y 3 = -75) ∧ 
  (x 4 = 4 ∧ y 4 = -120)

-- Equation definition derived from solution results:
def equation : Prop :=
  ∀ n : ℕ, y n = -5 * (x n)^2 - 10 * (x n)

-- The proof statement:
theorem relationship : points x y → equation x y :=
by
  intros,
  sorry

end relationship_l18_18075


namespace max_visible_sum_cubes_l18_18649

theorem max_visible_sum_cubes :
  ∃ (c1 c2 c3 : list ℕ), 
  (c1 ~ [1, 3, 9, 27, 81, 243]) ∧ (c2 ~ [1, 3, 9, 27, 81, 243]) ∧ (c3 ~ [1, 3, 9, 27, 81, 243]) ∧
  (∀ l, l ⊆ c1 ∨ l ⊆ c2 ∨ l ⊆ c3 → length l = 6) ∧
  (maximize_visible_sum c1 c2 c3) = 1087 := 
sorry

end max_visible_sum_cubes_l18_18649


namespace approx_val_l18_18990

variable (x : ℝ) (y : ℝ)

-- Definitions based on rounding condition
def approx_0_000315 : ℝ := 0.0003
def approx_7928564 : ℝ := 8000000

-- Main theorem statement
theorem approx_val (h1: x = approx_0_000315) (h2: y = approx_7928564) :
  x * y = 2400 := by
  sorry

end approx_val_l18_18990


namespace coaches_meet_together_l18_18784

theorem coaches_meet_together (e s n a : ℕ)
  (h₁ : e = 5) (h₂ : s = 3) (h₃ : n = 9) (h₄ : a = 8) :
  Nat.lcm (Nat.lcm e s) (Nat.lcm n a) = 360 :=
by
  sorry

end coaches_meet_together_l18_18784


namespace convert_decimal_to_vulgar_fraction_l18_18349

theorem convert_decimal_to_vulgar_fraction : (32 : ℝ) / 100 = (8 : ℝ) / 25 :=
by
  sorry

end convert_decimal_to_vulgar_fraction_l18_18349


namespace reflection_squared_identity_l18_18564

noncomputable def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let (a, b) := v
  let normSq := a * a + b * b
  ! let scale := (2 / normSq)
  ! Matrix.of !₂ !₂ [
  [1 - scale * b * b, scale * b * a],
  [scale * a * b, 1 - scale * a * a]
] 

theorem reflection_squared_identity :
  let v := (2, -1)
  let R := reflection_matrix v
  R * R = 1 :=
  by
  let v := (2, -1)
  let R := reflection_matrix v
  sorry

end reflection_squared_identity_l18_18564


namespace nested_logarithm_solution_l18_18598

theorem nested_logarithm_solution (x : ℝ) :
  log (3/4) (log (2/3) (log (1/3) (log 16 x))) = 0 ↔ x = 2 / 3 :=
by sorry

end nested_logarithm_solution_l18_18598


namespace illiterate_employee_count_l18_18179

variable (I : ℕ) -- Number of illiterate employees
variable (literate_count : ℕ) -- Number of literate employees
variable (initial_wage_illiterate : ℕ) -- Initial average wage of illiterate employees
variable (new_wage_illiterate : ℕ) -- New average wage of illiterate employees
variable (average_salary_decrease : ℕ) -- Decrease in the average salary of all employees

-- Given conditions:
def condition1 : initial_wage_illiterate = 25 := by sorry
def condition2 : new_wage_illiterate = 10 := by sorry
def condition3 : average_salary_decrease = 10 := by sorry
def condition4 : literate_count = 10 := by sorry

-- Main proof statement:
theorem illiterate_employee_count :
  initial_wage_illiterate - new_wage_illiterate = 15 →
  average_salary_decrease * (literate_count + I) = (initial_wage_illiterate - new_wage_illiterate) * I →
  I = 20 := by
  intros h1 h2
  -- provided conditions
  exact sorry

end illiterate_employee_count_l18_18179


namespace angle_sum_l18_18014

variable (A B C D E F : Type)
variable [IsoscelesTriangle A B C where h : ∠BAC = 15°]
variable [IsoscelesTriangle D E F where h : ∠EDF = 25°]

theorem angle_sum (h1 : ∠BAC = 15°) (h2 : ∠EDF = 25°)
  (h3 : AB = AC) (h4 : DE = DF) :
  ∠DAC + ∠ADE = 160°
:= sorry

end angle_sum_l18_18014


namespace flea_reaches_B_in_1554_jumps_l18_18965

variable (A B : Type) [MetricSpace A] [MetricSpace B]

theorem flea_reaches_B_in_1554_jumps (L d : ℕ) (hL : L = 20190) (hd : d = 13)
  (hn : ℕ) (hn_def : hn = L / d) (hn_floor : hn = 1553):
  ∃ n, n = hn + 1 :=
by { use (hn + 1), rw [hn_floor], norm_num }

end flea_reaches_B_in_1554_jumps_l18_18965


namespace even_and_increasing_function_l18_18629

theorem even_and_increasing_function :
  ∃! f : ℝ → ℝ,
    (∀ x : ℝ, f (-x) = f x) ∧ -- even function
    (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂) ∧ -- monotonically increasing on (0, +∞)
    (f = λ x, -x^2 ∨ f = λ x, x⁻¹ ∨ f = λ x, log 2 |x| ∨ f = λ x, -2^x) :=
by 
  sorry

end even_and_increasing_function_l18_18629


namespace proof_sum_of_solutions_l18_18104

noncomputable def solution_sum : ℝ :=
  let f := λ x : ℝ, |x^2 - 8 * x + 12|
  let g := λ x : ℝ, (20 : ℝ) / 3 - x
  ∑ x in { x : ℝ | f x = g x }, x

theorem proof_sum_of_solutions :
  solution_sum = 16 :=
  sorry

end proof_sum_of_solutions_l18_18104


namespace inequality_solution_l18_18600

theorem inequality_solution (x : ℝ) (hx : x > 0) : 
  (5 ^ (1 / 4 * (Real.log x / Real.log 5) ^ 2) ≥ 5 * x ^ (1 / 5 * Real.log x / Real.log 5)) ↔ 
  (x ∈ Set.Icc 0 (5 ^ (-2 * Real.sqrt 5)) ∪ Set.Icc (5 ^ (2 * Real.sqrt 5)) ∞) :=
by 
  sorry

end inequality_solution_l18_18600


namespace cos_double_angle_l18_18509

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 2/3) : Real.cos (2 * θ) = -1/9 := 
  sorry

end cos_double_angle_l18_18509


namespace modular_inverse_17_mod_800_l18_18331

theorem modular_inverse_17_mod_800 :
  ∃ x : ℤ, 17 * x ≡ 1 [MOD 800] ∧ 0 ≤ x ∧ x < 800 ∧ x = 753 := by
  sorry

end modular_inverse_17_mod_800_l18_18331


namespace problem_statement_l18_18897

theorem problem_statement (a b c : ℝ) (h1 : a + 2 * b + 3 * c = 12) (h2 : a^2 + b^2 + c^2 = a * b + a * c + b * c) :
  a + b^2 + c^3 = 14 :=
sorry

end problem_statement_l18_18897


namespace cubes_product_fraction_l18_18239

theorem cubes_product_fraction :
  (4^3 * 6^3 * 8^3 * 9^3 : ℚ) / (10^3 * 12^3 * 14^3 * 15^3) = 576 / 546875 := 
sorry

end cubes_product_fraction_l18_18239


namespace star_p_is_24_l18_18217

def digits_sum (x : ℕ) : ℕ :=
    x.toString.toList.map (λ c : Char, c.toNat - '0'.toNat).sum

def T : Finset ℕ := 
    Finset.filter (λ n, digits_sum n = 15) (Finset.range (10^6))

def p : ℕ := T.card

theorem star_p_is_24 : digits_sum p = 24 :=
by sorry

end star_p_is_24_l18_18217


namespace quadrilateral_PQRS_is_parallelogram_l18_18944

-- Define the rectangle and properties
variables (A B C D E P Q R S : Type) [geometry A B C D] 

-- Define the intersection point of diagonals
variables [is_intersection_of_diagonals E A B C D]

-- Define the circumcenters of the triangles
variables [is_circumcenter P A B E]
variables [is_circumcenter Q B C E]
variables [is_circumcenter R C D E]
variables [is_circumcenter S A D E]

theorem quadrilateral_PQRS_is_parallelogram 
  (h_rect: is_rectangle A B C D)
  (h_diag: diagonals_intersect_at A B C D E)
  (h_centerP: circumcenter_of P A B E)
  (h_centerQ: circumcenter_of Q B C E)
  (h_centerR: circumcenter_of R C D E)
  (h_centerS: circumcenter_of S A D E) :
  is_parallelogram P Q R S :=
sorry

end quadrilateral_PQRS_is_parallelogram_l18_18944


namespace right_triangle_hypotenuse_squared_l18_18756

noncomputable def polynomial_roots (P : Polynomial ℂ) : Set ℂ := {z : ℂ | Polynomial.eval z P = 0}

theorem right_triangle_hypotenuse_squared :
  let a, b, c : ℂ := classical.some (polynomial_roots (Polynomial.Coe.z3Sub2Z2Add2Add4))
    in |a|^2 + |b|^2 + |c|^2 = 300 
    ∧ ∃ (h : ℝ), √(|b - a|^2 + |c - a|^2) = h ∧ (b - a) * (c - a) = 0 →
    h^2 = 450 :=
by sorry

end right_triangle_hypotenuse_squared_l18_18756


namespace inequality_solution_l18_18603

theorem inequality_solution (x : ℝ) (hx : x ≠ -7) :
  (x^2 - 49) / (x + 7) < 0 ↔ x ∈ set.Ioo (-∞) (-7) ∪ set.Ioo (-7) 7 := by
  sorry

end inequality_solution_l18_18603


namespace actual_distance_between_towns_l18_18968

-- Definitions based on conditions
def scale_inch_to_miles : ℚ := 8
def map_distance_inches : ℚ := 27 / 8

-- Proof statement
theorem actual_distance_between_towns : scale_inch_to_miles * map_distance_inches / (1 / 4) = 108 := by
  sorry

end actual_distance_between_towns_l18_18968


namespace find_f_ignoring_parentheses_correct_l18_18232

theorem find_f_ignoring_parentheses_correct (a b c d f : ℤ) (h₁ : a = 2) (h₂ : b = 3)
(h₃ : c = 4) (h₄ : d = 5)
(h_student : a - b + c - d - f = -2 - f)
(h_correct: a - (b + (c - (d + f))) = f) :
    f = -1 := 
by
  rw [h₁, h₂, h₃, h₄] at h_student h_correct
  rw [← h_correct] at h_student
  exact h_student.symm

end find_f_ignoring_parentheses_correct_l18_18232


namespace total_letters_correct_l18_18252

-- Define the conditions
def letters_January := 6
def letters_February := 9
def letters_March := 3 * letters_January

-- Definition of the total number of letters sent
def total_letters := letters_January + letters_February + letters_March

-- The statement we need to prove in Lean
theorem total_letters_correct : total_letters = 33 := 
by
  sorry

end total_letters_correct_l18_18252


namespace find_all_pairs_l18_18378

def is_valid_pair (w l : ℕ) : Prop :=
  w * l = 18

def valid_pairs : set (ℕ × ℕ) :=
  {p | is_valid_pair p.1 p.2 ∧ p.1 > 0 ∧ p.2 > 0}

theorem find_all_pairs :
  valid_pairs = {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} :=
by sorry

end find_all_pairs_l18_18378


namespace integer_power_sums_l18_18959

theorem integer_power_sums (x : ℝ) (h : x + (1 / x) ∈ ℤ) (n : ℕ) : 
  x^n + (1 / x^n) ∈ ℤ := 
sorry

end integer_power_sums_l18_18959


namespace bricks_in_row_l18_18654

theorem bricks_in_row 
  (total_bricks : ℕ) 
  (rows_per_wall : ℕ) 
  (num_walls : ℕ)
  (total_rows : ℕ)
  (h1 : total_bricks = 3000)
  (h2 : rows_per_wall = 50)
  (h3 : num_walls = 2) 
  (h4 : total_rows = rows_per_wall * num_walls) :
  total_bricks / total_rows = 30 :=
by
  sorry

end bricks_in_row_l18_18654


namespace planes_intersecting_l18_18132

variables (a b : ℝ^3 → Prop) (α β : ℝ^3 → Prop)

def is_skew (a b : ℝ^3 → Prop) : Prop :=
  ¬ ∃ p : ℝ^3, a p ∧ b p

def perp_to_plane (l : ℝ^3 → Prop) (π : ℝ^3 → Prop) : Prop :=
  ∃ v : ℝ^3, ∃ n : ℝ^3, (∀ p : ℝ^3, l p → v = p) ∧ (∀ p : ℝ^3, π p → n ≠ 0 ∧ n ⬝ v = 0)

theorem planes_intersecting
  (skew_ab : is_skew a b)
  (perp_a_α : perp_to_plane a α)
  (perp_b_β : perp_to_plane b β) :
  ∃ p : ℝ^3, α p ∧ β p :=
sorry

end planes_intersecting_l18_18132


namespace derek_dogs_count_l18_18776

theorem derek_dogs_count
  (initial_dogs : ℕ)
  (initial_cars : ℕ)
  (cars_after_10_years : ℕ)
  (dogs_after_10_years : ℕ)
  (h1 : initial_dogs = 90)
  (h2 : initial_dogs = 3 * initial_cars)
  (h3 : cars_after_10_years = initial_cars + 210)
  (h4 : cars_after_10_years = 2 * dogs_after_10_years) :
  dogs_after_10_years = 120 :=
by
  sorry

end derek_dogs_count_l18_18776


namespace stock_percentage_increase_l18_18403

theorem stock_percentage_increase (x : ℝ) :
  let first_day_value := 0.75 * x,
      second_day_value := first_day_value + 0.4 * first_day_value in
  second_day_value = 1.05 * x := by
  sorry

end stock_percentage_increase_l18_18403


namespace strawberries_count_l18_18867

theorem strawberries_count (harvest_per_day : ℕ) (days_in_april : ℕ) (given_away : ℕ) (stolen : ℕ) :
  (harvest_per_day = 5) →
  (days_in_april = 30) →
  (given_away = 20) →
  (stolen = 30) →
  (harvest_per_day * days_in_april - given_away - stolen = 100) :=
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
  sorry,
}

end strawberries_count_l18_18867


namespace describe_set_T_l18_18948

theorem describe_set_T:
  ( ∀ (x y : ℝ), ((x + 2 = 4 ∧ y - 5 ≤ 4) ∨ (y - 5 = 4 ∧ x + 2 ≤ 4) ∨ (x + 2 = y - 5 ∧ 4 ≤ x + 2)) →
    ( ∃ (x y : ℝ), x = 2 ∧ y ≤ 9 ∨ y = 9 ∧ x ≤ 2 ∨ y = x + 7 ∧ x ≥ 2 ∧ y ≥ 9) ) :=
sorry

end describe_set_T_l18_18948


namespace minimum_acquainted_pairs_l18_18526

theorem minimum_acquainted_pairs (n : ℕ) (h_n_eq : n = 225)
    (h_condition : ∀ s : Finset ℕ, s.card = 6 → ∃ t : Finset (Finset ℕ), t.card = 3 ∧ (∀ p ∈ t, ∃ a b : ℕ, a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ (a, b) ∈ pairs)) :
    ∃ pairs : ℕ, pairs = 24750 :=
by
  sorry

end minimum_acquainted_pairs_l18_18526


namespace PK_eq_PH_l18_18916

namespace Geometry

variables {A B C H K P : Point}
variables {E F : Line}
variables [acute_triangle : acute_angle_triangle A B C]
variables [altitudes_BE_CF : ∀ (B E F : Point), is_altitude B E ∧ is_altitude C F]
variables [H_orthocenter : is_orthocenter A B C H]
variables [perpendicular_to_EF : is_perpendicular H E F]
variables [intersection_with_arc : Arc_Circumcircle := λ _, intersects_arc_circumcircle A B C H E F K]
variables [AK_meets_BC_at_P : meets_on_line AK BC P]

theorem PK_eq_PH : distance P K = distance P H := 
by 
  sorry

end Geometry

end PK_eq_PH_l18_18916


namespace strawberry_rows_l18_18585

theorem strawberry_rows (yield_per_row total_harvest : ℕ) (h1 : yield_per_row = 268) (h2 : total_harvest = 1876) :
  total_harvest / yield_per_row = 7 := 
by 
  sorry

end strawberry_rows_l18_18585


namespace number_of_correct_statements_l18_18830

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic (f : ℝ → ℝ) : Prop := ∀ x, f (x + 6) = f (x) + f 3

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x1 x2 ∈ set.Icc a b, x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0

theorem number_of_correct_statements (f : ℝ → ℝ) 
  (h1: even_function f) 
  (h2: periodic f) 
  (h3: increasing_on_interval f 0 3) : 1 :=
begin
  sorry
end

end number_of_correct_statements_l18_18830


namespace exists_unit_vector_l18_18791

noncomputable def v1 : ℝ × ℝ × ℝ := (2, 2, 1)
noncomputable def v2 : ℝ × ℝ × ℝ := (2, 1, 4)

theorem exists_unit_vector (u : ℝ × ℝ × ℝ) :
  (u.1 * v1.1 + u.2 * v1.2 + u.3 * v1.3 = 0) ∧
  (u.1 * v2.1 + u.2 * v2.2 + u.3 * v2.3 = 0) ∧
  (u.1^2 + u.2^2 + u.3^2 = 1) :=
sorry

end exists_unit_vector_l18_18791


namespace steve_halfway_longer_than_danny_l18_18767

theorem steve_halfway_longer_than_danny :
  let T_d : Float := 31
  let T_s : Float := 2 * T_d
  (T_s / 2) - (T_d / 2) = 15.5 :=
by
  let T_d : Float := 31
  let T_s : Float := 2 * T_d
  show (T_s / 2) - (T_d / 2) = 15.5
  sorry

end steve_halfway_longer_than_danny_l18_18767


namespace difference_divisible_by_base_minus_one_l18_18409

-- Definitions for the base of the numeral system and the number
variables (b n : ℕ) (a i : ℕ → ℕ) (ip : ℕ → ℕ) -- assuming ip as permutation function

noncomputable def N : ℕ := ∑ i in Finset.range (n + 1), a i * b ^ (n - i)
noncomputable def N' : ℕ := ∑ i in Finset.range (n + 1), a (ip i) * b ^ (n - i)

theorem difference_divisible_by_base_minus_one (b n : ℕ) (a i : ℕ → ℕ) (ip : ℕ → ℕ) :
  (N b n a - N' b n a ip) % (b - 1) = 0 :=
by
  sorry

end difference_divisible_by_base_minus_one_l18_18409


namespace equation_of_line_through_point_area_l18_18376

theorem equation_of_line_through_point_area (a T' : ℝ) (h : T' > 0) :
  ∃ (x y : ℝ), 2 * T' * x - a^2 * y + 2 * a * T' = 0 :=
by {
  use (a, 0),
  sorry
}

end equation_of_line_through_point_area_l18_18376


namespace triangle_ABC_A_eq_B_l18_18478

theorem triangle_ABC_A_eq_B 
  (A B C : ℝ) 
  (h : 1^2 - 1 * cos A * cos B - cos^2 (C / 2) = 0) 
  (hABC : A + B + C = π) : 
  A = B :=
sorry

end triangle_ABC_A_eq_B_l18_18478


namespace arctan_equation_l18_18679

theorem arctan_equation :
  4 * arctan (1 / 5) - arctan (1 / 239) = π / 4 :=
sorry

end arctan_equation_l18_18679


namespace length_of_other_train_l18_18655

def speed1 := 90 -- speed in km/hr
def speed2 := 90 -- speed in km/hr
def length_train1 := 1.10 -- length in km
def crossing_time := 40 -- time in seconds

theorem length_of_other_train : 
  ∀ s1 s2 l1 t l2 : ℝ,
  s1 = 90 → s2 = 90 → l1 = 1.10 → t = 40 → 
  ((s1 + s2) / 3600 * t - l1 = l2) → 
  l2 = 0.90 :=
by
  intros s1 s2 l1 t l2 hs1 hs2 hl1 ht hdist
  sorry

end length_of_other_train_l18_18655


namespace knives_percentage_l18_18057

-- Definitions based on conditions
def initial_knives : ℕ := 6
def initial_forks : ℕ := 12
def initial_spoons : ℕ := 3 * initial_knives
def traded_knives : ℕ := 10
def traded_spoons : ℕ := 6

-- Definitions for calculations
def final_knives : ℕ := initial_knives + traded_knives
def final_spoons : ℕ := initial_spoons - traded_spoons
def total_silverware : ℕ := final_knives + final_spoons + initial_forks

-- Theorem to prove the percentage of knives
theorem knives_percentage : (final_knives * 100) / total_silverware = 40 := by
  sorry

end knives_percentage_l18_18057


namespace harmonic_sum_inequality_l18_18483

theorem harmonic_sum_inequality (n : ℕ) : 
  1 + ∑ i in Finset.range (2^(n+1)-1), (1/(i + 2)) < n + 1 := 
by
  sorry

end harmonic_sum_inequality_l18_18483


namespace tan_alpha_beta_cot_beta_eq_two_l18_18468

theorem tan_alpha_beta_cot_beta_eq_two
  (α β : ℝ)
  (h : Real.sin (α + 2 * β) = 3 * Real.sin α) :
  Real.tan (α + β) * Real.cot β = 2 := 
sorry

end tan_alpha_beta_cot_beta_eq_two_l18_18468


namespace sum_of_perfect_square_divisors_of_544_l18_18101

-- Conditions and Definitions
def is_divisor (a b : ℕ) : Prop := b % a = 0
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def sum_perfect_square_divisors (n : ℕ) : ℕ :=
  ∑ d in (Finset.filter (λ d, is_divisor d n ∧ is_perfect_square d) (Finset.range (n + 1))), d

-- Proof to be conducted
theorem sum_of_perfect_square_divisors_of_544 : sum_perfect_square_divisors 544 = 21 :=
by sorry

end sum_of_perfect_square_divisors_of_544_l18_18101


namespace water_level_representation_l18_18522

theorem water_level_representation :
  (∀ x : ℤ, x > 0 → "+" ++ toString x ++ " meters") →
  ∃ y : ℤ, y < 0 → "-" ++ toString (abs y) ++ " meters" :=
by
  intro h
  use -2
  intro hy
  sorry

end water_level_representation_l18_18522


namespace find_t_l18_18842

noncomputable def f (x t k : ℝ): ℝ := (1/3) * x^3 - (t/2) * x^2 + k * x

theorem find_t (a b t k : ℝ) (h1 : t > 0) (h2 : k > 0) 
  (h3 : a + b = t) (h4 : a * b = k)
  (h5 : 2 * a = b - 2)
  (h6 : (-2) ^ 2 = a * b) : 
  t = 5 :=
by 
  sorry

end find_t_l18_18842


namespace range_of_a_l18_18487

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 3 - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Icc (-2 : ℝ) 2 → f x a ≥ 0) ↔ a ∈ Icc (-7) 2 :=
sorry

end range_of_a_l18_18487


namespace find_x_sums_to_neg7_l18_18664

def mean (x : ℝ) : ℝ := (5 + 7 + 10 + x + 20) / 5

def is_median_equal_to_mean (x : ℝ) : Prop := 
  mean x = list.median [5, 7, 10, x, 20]

def sum_of_smallest_two (x : ℝ) : Prop := 
  let sorted_list := list.sort (≤) [5, 7, 10, x, 20]
  (sorted_list.nth 0).get_or_else 0 + (sorted_list.nth 1).get_or_else 0 = 12

theorem find_x_sums_to_neg7 :
  ∃ x : ℝ, is_median_equal_to_mean x ∧ sum_of_smallest_two x ∧ x = -7 :=
by
  sorry

end find_x_sums_to_neg7_l18_18664


namespace total_distance_is_75_l18_18502

def distance1 : ℕ := 30
def distance2 : ℕ := 20
def distance3 : ℕ := 25

def total_distance : ℕ := distance1 + distance2 + distance3

theorem total_distance_is_75 : total_distance = 75 := by
  sorry

end total_distance_is_75_l18_18502


namespace ratio_of_ages_in_two_years_l18_18024

-- Define the constants
def son_age : ℕ := 24
def age_difference : ℕ := 26

-- Define the equations based on conditions
def man_age := son_age + age_difference
def son_future_age := son_age + 2
def man_future_age := man_age + 2

-- State the theorem for the required ratio
theorem ratio_of_ages_in_two_years : man_future_age / son_future_age = 2 := by
  sorry

end ratio_of_ages_in_two_years_l18_18024


namespace find_f_of_f_l18_18518

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else (4 * x + 1 - 2 / x) / 3

theorem find_f_of_f (h : ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 2 * x + 1) : 
  f 2 = -1/3 :=
sorry

end find_f_of_f_l18_18518


namespace linear_or_large_derivative_l18_18943

noncomputable def continuous_differentiable_function (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Icc (0 : ℝ) 1, ContinuousOn f (Icc (0 : ℝ) 1) ∧ DifferentiableOn ℝ f (Ioo (0 : ℝ) 1)

theorem linear_or_large_derivative (f : ℝ → ℝ) (hf : continuous_differentiable_function f) :
  (∃ a b : ℝ, ∀ x ∈ Icc (0 : ℝ) 1, f x = a * x + b) ∨ (∃ t ∈ Ioo (0 : ℝ) 1, |f 1 - f 0| < |f' t|) :=
sorry

end linear_or_large_derivative_l18_18943


namespace sum_of_perfect_square_divisors_of_544_l18_18103

def divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, n % d = 0) (List.range (n + 1))

def is_perfect_square (n : ℕ) : Bool :=
  ∃ k : ℕ, k * k = n

def perfect_square_divisors (n : ℕ) : List ℕ :=
  (divisors n).filter is_perfect_square

theorem sum_of_perfect_square_divisors_of_544 : 
  ∑ d in perfect_square_divisors 544, d = 21 :=
sorry

end sum_of_perfect_square_divisors_of_544_l18_18103


namespace eighteen_factorial_digit_sum_l18_18993

theorem eighteen_factorial_digit_sum :
  ∀ T M : ℕ,
    (18.factorial.to_digits = [6, 4, 0, 2, 3, T, 7, 5, 2, M, 0, 8, 8, 0, 0]) →
    ((47 + T + M) % 3 = 0) →
    ((6 + T - M + 1) % 11 = 0) →
    (T + M = 11) :=
by
  sorry

end eighteen_factorial_digit_sum_l18_18993


namespace best_purchase_option_l18_18278

theorem best_purchase_option 
  (blend_techn_city : ℕ := 2000)
  (meat_techn_city : ℕ := 4000)
  (discount_techn_city : ℤ := 10)
  (blend_techn_market : ℕ := 1500)
  (meat_techn_market : ℕ := 4800)
  (bonus_techn_market : ℤ := 20) : 
  0.9 * (blend_techn_city + meat_techn_city) < (blend_techn_market + meat_techn_market) :=
by
  sorry

end best_purchase_option_l18_18278


namespace four_digit_number_count_l18_18871

theorem four_digit_number_count :
  let first_digit_choices := {1, 3}, -- The leftmost digit is odd and less than 5
      second_digit_choices := {0, 2, 4}, -- The second digit is an even number less than 6
      all_digits := (finset.range 10).filter (finset.filter (≠ first_digit ∧ ≠ second_digit)),
      valid_choices (first second third fourth : ℕ) :=
        first ∈ first_digit_choices ∧
        second ∈ second_digit_choices ∧
        third ∈ all_digits ∧
        fourth ∈ {0, 5} ∧
        first ≠ second ∧
        first ≠ third ∧
        first ≠ fourth ∧
        second ≠ third ∧
        second ≠ fourth ∧
        third ≠ fourth ∧
        (10^3 * first + 10^2 * second + 10 * third + fourth) % 5 = 0 in
  (finset.range 10).filter(valid_choices).card = 48 := by
sorry

end four_digit_number_count_l18_18871


namespace water_difference_after_transfer_l18_18701

def initial_large_bucket : ℕ := 7
def initial_small_bucket : ℕ := 5
def transferred_water : ℕ := 2

theorem water_difference_after_transfer :
  let final_large_bucket := initial_large_bucket + transferred_water in
  let final_small_bucket := initial_small_bucket - transferred_water in
  final_large_bucket - final_small_bucket = 6 :=
by
  let final_large_bucket := initial_large_bucket + transferred_water
  let final_small_bucket := initial_small_bucket - transferred_water
  show final_large_bucket - final_small_bucket = 6
  sorry

end water_difference_after_transfer_l18_18701


namespace subtract_correct_calculation_l18_18884

theorem subtract_correct_calculation:
  (∃ n : ℕ, 40 + n = 52 ∧ 20 - n = 8) :=
begin
  -- Proof omitted
  sorry
end

end subtract_correct_calculation_l18_18884


namespace total_amount_paid_l18_18311

-- Define the conditions
def each_company_ad_spaces : ℕ := 10
def ad_space_length : ℝ := 12
def ad_space_width : ℝ := 5
def cost_per_square_foot : ℝ := 60

-- Area of one ad space
def area_of_one_ad_space : ℝ := ad_space_length * ad_space_width

-- Total area purchased by one company
def total_area_one_company : ℝ := area_of_one_ad_space * each_company_ad_spaces

-- Cost for one company
def cost_for_one_company : ℝ := total_area_one_company * cost_per_square_foot

-- Total cost for all three companies
def total_cost_three_companies : ℝ := cost_for_one_company * 3

-- Proof statement
theorem total_amount_paid (each_company_ad_spaces = 10) (ad_space_length = 12) (ad_space_width = 5) (cost_per_square_foot = 60):
  total_cost_three_companies = 108000 := sorry

end total_amount_paid_l18_18311


namespace isosceles_triangle_perimeter_l18_18464

theorem isosceles_triangle_perimeter
    (a b : ℝ)
    (h1 : |a - 2| + b^2 - 12 * b + 36 = 0)
    (h2 : a = 6 ∨ b = 6 ∧ (a = 2 ∨ b = 2)) :
  a + a + b = 14 ∨ b + b + a = 14 :=
begin
  sorry
end

end isosceles_triangle_perimeter_l18_18464


namespace AugustHasFiveFridays_l18_18985

theorem AugustHasFiveFridays (N : ℕ) (JulyHasFiveTuesdays : ∃ (startDay : ℕ), 0 ≤ startDay < 7 ∧ (startDay = 2 ∨ (startDay + 7 ≤ 31 ∧ startDay + 14 ≤ 31 ∧ startDay + 21 ≤ 31 ∧ startDay + 28 ≤ 31 ∧ startDay + 7 = 1))) (JulyAndAugust31Days : (true)) : ∃ (friday : ℕ), 0 ≤ friday < 7 ∧ ∀ (August : list ℕ), August = list.range 31 ∧ ∀ d ∈ August, (d + startDay) % 7 = friday → (list.count friday August = 5) :=
by
  sorry

end AugustHasFiveFridays_l18_18985


namespace data_set_range_l18_18911

theorem data_set_range (max_val min_val : ℕ) (h_max : max_val = 78) (h_min : min_val = 21) : max_val - min_val = 57 := by
  rw [h_max, h_min]
  norm_num
  sorry

end data_set_range_l18_18911


namespace no_rational_x_y_m_n_with_conditions_l18_18200

noncomputable def f (t : ℚ) : ℚ := t^3 + t

theorem no_rational_x_y_m_n_with_conditions :
  ¬ ∃ (x y : ℚ) (m n : ℕ), xy = 3 ∧ m > 0 ∧ n > 0 ∧
    (f^[m] x = f^[n] y) := 
sorry

end no_rational_x_y_m_n_with_conditions_l18_18200


namespace sum_of_solutions_x_l18_18183

theorem sum_of_solutions_x (x : ℝ) (y : ℝ) (h1 : y = 8) (h2 : x^3 + y^2 = 169) : 
  x = real.cbrt 105 :=
by
  have h3 : 8^2 = 64 := by norm_num
  rw h1 at h2
  rw h3 at h2
  rw add_comm at h2
  have h4 : x^3 = 105 := by linarith
  exact real.cbrt_eq_iff.mpr ⟨h4, rfl⟩

end sum_of_solutions_x_l18_18183


namespace math_proof_problem_l18_18825

noncomputable def problemStatement : Prop :=
  ∃ (α : ℝ), 5 * Real.sin (2 * α) = Real.sin (2 * Real.pi / 180) ∧ 
  (Real.tan (α + Real.pi / 180) / Real.tan (α - Real.pi / 180) = -3/2)

theorem math_proof_problem : problemStatement := 
by 
  sorry

end math_proof_problem_l18_18825


namespace count_congruent_numbers_l18_18418

theorem count_congruent_numbers :
  {x : ℕ // x < 150 ∧ x % 45 = 5}.card = 4 :=
by
    sorry

end count_congruent_numbers_l18_18418


namespace sector_radius_l18_18991

theorem sector_radius (A : ℝ) (θ : ℝ) (r : ℝ) (h1 : θ = 42) (h2 : A = 36.67) (h3 : A = (θ / 360) * real.pi * r^2) : r = 10 := 
by
  sorry

end sector_radius_l18_18991


namespace Patricia_center_seconds_l18_18974

theorem Patricia_center_seconds:
  let point_guard_seconds := 130
  let shooting_guard_seconds := 145
  let small_forward_seconds := 85
  let power_forward_seconds := 60
  let total_players := 5
  let average_time_per_player := 2 * 60 -- 2 minutes converted to seconds
  let total_time := total_players * average_time_per_player
  let recorded_time_four_players := point_guard_seconds + shooting_guard_seconds + small_forward_seconds + power_forward_seconds
  let center_seconds := total_time - recorded_time_four_players
in center_seconds = 180 := by
  sorry

end Patricia_center_seconds_l18_18974


namespace lcm_prime_factors_l18_18431

-- Conditions
def n1 := 48
def n2 := 180
def n3 := 250

-- The equivalent proof problem
theorem lcm_prime_factors (l : ℕ) (h1: l = Nat.lcm n1 (Nat.lcm n2 n3)) :
  l = 18000 ∧ (∀ a : ℕ, a ∣ l ↔ a ∣ 2^4 * 3^2 * 5^3) :=
by
  sorry

end lcm_prime_factors_l18_18431


namespace expressionEquals243_l18_18747

noncomputable def calculateExpression : ℕ :=
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 *
  (1 / 19683) * 59049

theorem expressionEquals243 : calculateExpression = 243 := by
  sorry

end expressionEquals243_l18_18747


namespace common_ratio_of_geom_seq_l18_18812

-- Define the conditions: geometric sequence and the given equation
def is_geom_seq (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem common_ratio_of_geom_seq
  (a : ℕ → ℝ)
  (h_geom : is_geom_seq a)
  (h_eq : a 1 * a 7 = 3 * a 3 * a 4) :
  ∃ q : ℝ, is_geom_seq a ∧ q = 3 := 
sorry

end common_ratio_of_geom_seq_l18_18812


namespace length_AC_correct_l18_18176

noncomputable def length_AC (A B C D : Type) : ℝ := 105 / 17

variable {A B C D : Type}
variables (angle_BAC angle_ADB length_AD length_BC : ℝ)

theorem length_AC_correct
  (h1 : angle_BAC = 60)
  (h2 : angle_ADB = 30)
  (h3 : length_AD = 3)
  (h4 : length_BC = 9) :
  length_AC A B C D = 105 / 17 :=
sorry

end length_AC_correct_l18_18176


namespace extinction_prob_l18_18712

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the base case condition
def v_0 : ℝ := 1

-- Define the extinction probability function
def v : ℕ → ℝ
| 0 => v_0
| k => p * v (k + 1) + q * v (k - 1)

-- Define the target value for v_1
def v_1 : ℝ := 2 / 3

-- The main theorem: Prove that v 1 = 2 / 3 given the conditions
theorem extinction_prob : v 1 = v_1 := by
  -- Proof will be provided here
  sorry

end extinction_prob_l18_18712


namespace contest_path_count_l18_18357

/-- Define the 8x8 grid with the letters. -/
def grid : list (list char) := [
  [' ', ' ', ' ', ' ', ' ', ' ', ' ', 'C', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', ' ', 'C', 'O', 'C', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', 'C', 'O', 'N', 'O', 'C', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', 'C', 'O', 'N', 'T', 'N', 'O', 'C', ' ', ' ', ' '],
  [' ', ' ', ' ', 'C', 'O', 'N', 'T', 'E', 'T', 'N', 'O', 'C', ' ', ' '],
  [' ', ' ', 'C', 'O', 'N', 'T', 'E', 'S', 'T', 'N', 'O', 'C', ' ', ' '],
  [' ', 'C', 'O', 'N', 'T', 'E', 'S', 'T', 'S', 'T', 'N', 'O', 'C', ' '],
  ['C', 'O', 'N', 'T', 'E', 'S', 'T', 'S', 'E', 'S', 'T', 'N', 'O', 'C']
]
  
/-- A function that counts the number of valid paths spelling "CONTESTS" -/
def countPaths : list (list char) → nat :=
  sorry -- Placeholder for the implementation

/-- Theorem stating the count of valid paths spelling "CONTESTS" is 255 -/
theorem contest_path_count :
  countPaths grid = 255 :=
sorry

end contest_path_count_l18_18357


namespace min_value_sin4_plus_2_cos4_l18_18433

theorem min_value_sin4_plus_2_cos4 : ∀ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 ≥ 2 / 3 :=
by
  intro x
  sorry

end min_value_sin4_plus_2_cos4_l18_18433


namespace train_speed_l18_18389

theorem train_speed (length : ℕ) (time : ℕ) (v : ℕ)
  (h1 : length = 750)
  (h2 : time = 1)
  (h3 : v = (length + length) / time)
  (h4 : v = 1500) :
  (v * 60 / 1000 = 90) :=
by
  sorry

end train_speed_l18_18389


namespace interest_rate_6_l18_18246

theorem interest_rate_6 (P SI : ℝ) (hP : P = 1800) (hSI : SI = 632) : 
  ∃ (R : ℝ), (T : ℝ) (hT : T = R), SI = P * R * T / 100 ∧ R = 6 :=
by
  sorry

end interest_rate_6_l18_18246


namespace jim_mpg_l18_18549

theorem jim_mpg (tank_size : ℕ) (fraction_used : ℚ) (total_miles : ℕ) :
  tank_size = 12 → fraction_used = 1 / 3 → total_miles = 20 → total_miles / (tank_size * fraction_used) = 5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have h4 : tank_size * fraction_used = 4 := by rw [h1, h2]; norm_num
  rw h4
  norm_num
  sorry

end jim_mpg_l18_18549


namespace go_state_space_complexity_l18_18040

theorem go_state_space_complexity :
  let M := 3^361
  let N := 10^80
  M / N ≈ 10^93 :=
by
  sorry

end go_state_space_complexity_l18_18040


namespace problem_statement_l18_18280

theorem problem_statement (f : ℝ → ℝ) 
  (D : set ℝ := {x | x ≠ 0})
  (H1 : ∀ x y ∈ D, f (x * y) = f x + f y)
  (H2 : f 4 = 1)
  (H3 : ∀ x, f (3 * x + 1) + f (2 * x - 6) ≤ 3)
  (H4 : ∀ x > 0, ∀ y > 0, x < y → f x < f y) :
  (f 1 = 0) ∧ 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x : ℝ, x ∈ Icc (-∞ : ℝ) (-1 / 3 : ℝ) ∨ x ∈ Ioo (3 / 2 : ℝ) 3 ∨ x ∈ Icc 3 5) ↔ (D) :=
sorry

end problem_statement_l18_18280


namespace smallest_in_set_l18_18638

-- Definition of the set of numbers
def numbers_set : set ℝ := {3.2, 2.3, 3, 2.23, 3.22}

-- The smallest number in the set
def smallest_number : ℝ := 2.23

-- The theorem states that 2.23 is the smallest number in the set
theorem smallest_in_set : ∀ x ∈ numbers_set, smallest_number ≤ x :=
by sorry

end smallest_in_set_l18_18638


namespace g_neg_five_l18_18229

def g (x : ℝ) : ℝ :=
if x < -3 then 3 * x + 4 else 7 - 3 * x

theorem g_neg_five : g (-5) = -11 := by
  sorry

end g_neg_five_l18_18229


namespace leap_day_2040_is_tuesday_l18_18198

-- Define the given condition that 29th February 2012 is Wednesday
def feb_29_2012_is_wednesday : Prop := sorry

-- Define the calculation of the day of the week for February 29, 2040
def day_of_feb_29_2040 (initial_day : Nat) : Nat := (10228 % 7 + initial_day) % 7

-- Define the proof statement
theorem leap_day_2040_is_tuesday : feb_29_2012_is_wednesday →
  (day_of_feb_29_2040 3 = 2) := -- Here, 3 represents Wednesday and 2 represents Tuesday
sorry

end leap_day_2040_is_tuesday_l18_18198


namespace unique_books_l18_18174

variables (books : Type)
variables (Tony_books Dean_books Breanna_books Piper_books Asher_books : Finset books)
variables (Tony_Dean_shared Breanna_Piper_Asher_shared Dean_Piper_shared Tony_Shared_three : Finset books)
variables (Asher_Breanna_Shared_Tony Tony_Shared_all_five Breanna_Piper_Shared Dean_Shared_with_Breanna_Piper Asher_Shared_with_Breanna_Piper : Finset books)

-- Condition definitions
variables (hTony : Tony_books.card = 23)
variables (hDean : Dean_books.card = 20)
variables (hBreanna : Breanna_books.card = 30)
variables (hPiper : Piper_books.card = 26)
variables (hAsher : Asher_books.card = 25)
variables (hTony_Dean_shared : Tony_Dean_shared.card = 5)
variables (hBreanna_Piper_Asher_shared : Breanna_Piper_Asher_shared.card = 7)
variables (hDean_Piper_shared : Dean_Piper_shared.card = 6)
variables (hTony_Shared_three : Tony_Shared_three.card = 3)
variables (hAsher_Breanna_Shared_Tony : Asher_Breanna_Shared_Tony.card = 8)
variables (hTony_Shared_all_five : Tony_Shared_all_five.card = 2)
variables (hBreanna_Piper_Shared : Breanna_Piper_Shared.card = 9)
variables (hDean_Shared_with_Breanna_Piper : 4 ∈ Breanna_Piper_Shared.card)
variables (hAsher_Shared_with_Breanna_Piper : 2 ∈ Breanna_Piper_Shared.card)

-- Theorem statement
theorem unique_books :
  Tony_books.card +
  Dean_books.card +
  Breanna_books.card +
  Piper_books.card +
  Asher_books.card -
  Tony_Dean_shared.card -
  2 * Breanna_Piper_Asher_shared.card -
  2 * (Dean_Piper_shared.card - Tony_Shared_three.card) -
  3 -
  2 * (Asher_Breanna_Shared_Tony.card - 2 * Tony_Shared_all_five.card) -
  4 * Tony_Shared_all_five.card -
  (Breanna_Piper_Shared.card  - Dean_Shared_with_Breanna_Piper - Asher_Shared_with_Breanna_Piper) = 61 :=
by
  rw [hTony, hDean, hBreanna, hPiper, hAsher, hTony_Dean_shared, hBreanna_Piper_Asher_shared, hDean_Piper_shared, hTony_Shared_three, hAsher_Breanna_Shared_Tony, hTony_Shared_all_five, hBreanna_Piper_Shared, hDean_Shared_with_Breanna_Piper, hAsher_Shared_with_Breanna_Piper]
  sorry

end unique_books_l18_18174


namespace part1_part2_l18_18055

-- Part 1: Simplified form of 28x^4y^2 / 7x^3y is 4xy.
theorem part1 (x y : ℝ) : (28 * x^4 * y^2) / (7 * x^3 * y) = 4 * x * y :=
by sorry

-- Part 2: Value of (2x + 3y)^2 - (2x + y)(2x - y) when x = 1/3 and y = 1/2 is 4.5.
theorem part2 : (let x := 1 / 3; let y := 1 / 2 in (2 * x + 3 * y) ^ 2 - (2 * x + y) * (2 * x - y)) = 4.5 :=
by sorry

end part1_part2_l18_18055


namespace max_a4a8_geometric_sequence_l18_18912

noncomputable def max_value_of_a4a8 (a : ℕ → ℝ) (r : ℝ) : ℝ :=
let a2 := a 1 * r in
let a6 := a 1 * r ^ 5 in
let a5 := a 1 * r ^ 4 in
let a11 := a 1 * r ^ 10 in
let a4 := a 1 * r ^ 3 in
let a8 := a 1 * r ^ 7 in
if a2 * a6 + a5 * a11 = 16 then
  max (a4 * a8) 8
else
  0

theorem max_a4a8_geometric_sequence (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a (n + 1) = a n * r) (pos : ∀ n, 0 < a n) (condition : a 2 * a 6 + a 5 * a 11 = 16) :
  max_value_of_a4a8 a r = 8 :=
by
  sorry

end max_a4a8_geometric_sequence_l18_18912


namespace find_a7_of_arithmetic_sequence_l18_18181

variable (a : ℕ → ℝ)
variable (n : ℕ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2 : ℝ) * (a 1 + a n)

theorem find_a7_of_arithmetic_sequence (h_arith : is_arithmetic_sequence a) 
    (h_sum : sum_first_n_terms a 13 = 52) : 
    a 7 = 4 :=
sorry

end find_a7_of_arithmetic_sequence_l18_18181


namespace num_real_elements_in_union_l18_18222

inductive Hn : ℕ → Type
| H1 : Hn 1
| Hn_Succ : Π (n : ℕ) (x : Hn n), Hn (n+1)

noncomputable def element : Π (n : ℕ), (Hn n) → ℝ
| 1 Hn.H1 := real.sqrt 2
| (n+1) (Hn.Hn_Succ _ x) := real.sqrt (2 + element n x) -- Real.sqrt has both positive and negative roots, picking positive root

def unique_elements : Π (n : ℕ), set ℝ
| 1 := {real.sqrt 2}
| (n+1) := { real.sqrt (2 + x) | x ∈ unique_elements n } ∪ { real.sqrt (2 - x) | x ∈ unique_elements n }

noncomputable def union_Hn : set ℝ := ⋃ n = 1 to 2000, unique_elements n

theorem num_real_elements_in_union :
  finset.card (∪ (finset.range 2000).map (λ n, unique_elements (n+1))) = 2 ^ 2000 - 1 :=
sorry

end num_real_elements_in_union_l18_18222


namespace unique_id_tag_sequences_l18_18692

theorem unique_id_tag_sequences : 
  let chars := ['T', 'I', 'M', 'E', '2', '0', '1', '1', '6'] in
  let all_sequences := (chars.toList.permutations.filter (λ l, l.length = 6)) in
  all_sequences.length = 6120 := 
sorry

end unique_id_tag_sequences_l18_18692


namespace intersection_M_N_l18_18497

def M : Set ℝ := { x | -2 ≤ x ∧ x < 2 }
def N : Set ℝ := { x | x ≥ -2 }

theorem intersection_M_N : M ∩ N = { x | -2 ≤ x ∧ x < 2 } := by
  sorry

end intersection_M_N_l18_18497


namespace extinction_prob_l18_18714

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the base case condition
def v_0 : ℝ := 1

-- Define the extinction probability function
def v : ℕ → ℝ
| 0 => v_0
| k => p * v (k + 1) + q * v (k - 1)

-- Define the target value for v_1
def v_1 : ℝ := 2 / 3

-- The main theorem: Prove that v 1 = 2 / 3 given the conditions
theorem extinction_prob : v 1 = v_1 := by
  -- Proof will be provided here
  sorry

end extinction_prob_l18_18714


namespace carolyn_silverware_knives_percentage_l18_18061

theorem carolyn_silverware_knives_percentage :
  (let knives_initial := 6 in
   let forks_initial := 12 in
   let spoons_initial := 3 * knives_initial in
   let total_silverware_initial := knives_initial + forks_initial + spoons_initial in
   let knives_after_trade := 0 in
   let spoons_after_trade := spoons_initial + 6 in
   let total_silverware_after_trade := knives_after_trade + forks_initial + spoons_after_trade in
   percentage_knives := (knives_after_trade * 100) / total_silverware_after_trade in
   percentage_knives = 0) :=
by
  sorry

end carolyn_silverware_knives_percentage_l18_18061


namespace dot_product_is_negative_seventy_two_l18_18065

-- Define the first vector
def vec1 : Vector ℕ 3 := ⟨[8, -2, 4], by simp⟩

-- Define the second vector
def vec2 : Vector ℕ 3 := ⟨[-3, 12, -6], by simp⟩

-- Prove the dot product of vec1 and vec2 is -72
theorem dot_product_is_negative_seventy_two : dotProduct vec1 vec2 = -72 := by
  sorry

end dot_product_is_negative_seventy_two_l18_18065


namespace gretchen_total_earnings_l18_18863

-- Define the conditions
def price_per_drawing : ℝ := 20.0
def caricatures_sold_saturday : ℕ := 24
def caricatures_sold_sunday : ℕ := 16

-- The total caricatures sold
def total_caricatures_sold : ℕ := caricatures_sold_saturday + caricatures_sold_sunday

-- The total amount of money made
def total_money_made : ℝ := total_caricatures_sold * price_per_drawing

-- The theorem to be proven
theorem gretchen_total_earnings : total_money_made = 800.0 := by
  sorry

end gretchen_total_earnings_l18_18863


namespace base9_mult_218x5_l18_18795

/-- Define the base 9 numeral system -/
def base9 : Type := ℕ

/-- Define the conversion from base 9 to base 10 -/
def base9_to_base10 (n : ℕ) (digits : List ℕ) : ℕ :=
  digits.reverse.enum_from(0).foldl (λ acc ⟨i, d⟩, acc + d * (n ^ i)) 0

/-- Define the multiplication in base 9, and verify the correctness in base 9 -/
theorem base9_mult_218x5 :
  ∀ a b : base9,
  a = 218 ∧ b = 5 →
  base9_to_base10 9 ([4, 0, 2, 1] : List ℕ) = 1204 :=
begin
  sorry
end

end base9_mult_218x5_l18_18795


namespace dive_point_value_l18_18525

noncomputable def dive_scores : list ℝ := [7.5, 8.1, 9.0, 6.0, 8.5]
noncomputable def degree_of_difficulty : ℝ := 3.2

theorem dive_point_value :
  let remaining_scores := dive_scores.erase (dive_scores.maximum).get_or_else 0 
                          |>.erase (dive_scores.minimum).get_or_else 0
  in (remaining_scores.sum) * degree_of_difficulty = 77.12 :=
by
  -- sorry will be replaced with the formal proof here
  sorry

end dive_point_value_l18_18525


namespace longest_side_of_triangle_l18_18289

-- Definitions of the conditions in a)
def side1 : ℝ := 9
def side2 (x : ℝ) : ℝ := x + 5
def side3 (x : ℝ) : ℝ := 2 * x + 3
def perimeter : ℝ := 40

-- Statement of the mathematically equivalent proof problem.
theorem longest_side_of_triangle (x : ℝ) (h : side1 + side2 x + side3 x = perimeter) : 
  max side1 (max (side2 x) (side3 x)) = side3 x := 
sorry

end longest_side_of_triangle_l18_18289


namespace circle_center_radius_sum_l18_18946

noncomputable theory

open Real

theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), (∀ x y : ℝ, x^2 - 6 * x + y^2 - 4 * y = 20 ↔ (x - c)^2 + (y - d)^2 = s^2) ∧ 
  c = 3 ∧ d = 2 ∧ s = sqrt 33 ∧ c + d + s = 5 + sqrt 33 :=
by {
  sorry
}

end circle_center_radius_sum_l18_18946


namespace total_corn_cobs_l18_18699

-- Definitions for the conditions
def rows_first_field : ℕ := 13
def rows_second_field : ℕ := 16
def cobs_per_row : ℕ := 4

-- Statement to prove
theorem total_corn_cobs : (rows_first_field * cobs_per_row + rows_second_field * cobs_per_row) = 116 :=
by sorry

end total_corn_cobs_l18_18699


namespace triangle_side_BC_eqn_l18_18190

/-- In triangle ABC with A(3, -1), 
the equation of the median from vertex A to side AB is 6x + 10y - 59 = 0,
and the equation of the angle bisector of ∠B is x - 4y + 10 = 0.
Prove that the equation of the line containing side BC is 2x + 9y - 65 = 0. -/
theorem triangle_side_BC_eqn 
  (A : ℝ × ℝ) 
  (median_eq : ℝ → ℝ → ℝ) 
  (angle_bisector_eq : ℝ → ℝ → ℝ) :
  A = (3, -1) ∧ median_eq = (λ x y, 6 * x + 10 * y - 59) ∧ angle_bisector_eq = (λ x y, x - 4 * y + 10) →
  ∃ (line_BC_eq : ℝ → ℝ → ℝ), line_BC_eq = (λ x y, 2 * x + 9 * y - 65) :=
sorry

end triangle_side_BC_eqn_l18_18190


namespace f_even_f_period_l18_18519

-- Define the function as given
def f (x : ℝ) : ℝ := (sin x) ^ 2 - (1 / 2)

-- State the property that f is even
theorem f_even : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

-- State the property that the smallest positive period of f is π
theorem f_period : ∀ x : ℝ, f (x + π) = f x :=
by
  sorry

end f_even_f_period_l18_18519


namespace clusters_per_spoonful_l18_18236

theorem clusters_per_spoonful (spoonfuls_per_bowl : ℕ) (clusters_per_box : ℕ) (bowls_per_box : ℕ) 
  (h_spoonfuls : spoonfuls_per_bowl = 25) 
  (h_clusters : clusters_per_box = 500)
  (h_bowls : bowls_per_box = 5) : 
  clusters_per_box / bowls_per_box / spoonfuls_per_bowl = 4 := 
by 
  have clusters_per_bowl := clusters_per_box / bowls_per_box
  have clusters_per_spoonful := clusters_per_bowl / spoonfuls_per_bowl
  sorry

end clusters_per_spoonful_l18_18236


namespace color_swap_rectangle_l18_18778

theorem color_swap_rectangle 
  (n : ℕ) 
  (square_size : ℕ := 2*n - 1) 
  (colors : Finset ℕ := Finset.range n) 
  (vertex_colors : Fin (square_size + 1) × Fin (square_size + 1) → ℕ) 
  (h_vertex_colors : ∀ v, vertex_colors v ∈ colors) :
  ∃ row, ∃ (v₁ v₂ : Fin (square_size + 1) × Fin (square_size + 1)),
    (v₁.1 = row ∧ v₂.1 = row ∧ v₁ ≠ v₂ ∧
    (∃ r₀ r₁ r₂, r₀ ≠ r₁ ∧ r₁ ≠ r₂ ∧ r₂ ≠ r₀ ∧
    vertex_colors v₁ = vertex_colors (r₀, v₁.2) ∧
    vertex_colors v₂ = vertex_colors (r₀, v₂.2) ∧
    vertex_colors (r₁, v₁.2) = vertex_colors (r₂, v₂.2))) := 
sorry

end color_swap_rectangle_l18_18778


namespace shifted_parabola_l18_18170

theorem shifted_parabola (x : ℝ) : 
  let G := λ x, (x^2 - 1) + 3 in 
  G x = x^2 + 2 :=
by
  sorry

end shifted_parabola_l18_18170


namespace parabola_kite_area_l18_18291

theorem parabola_kite_area (a b : ℝ)
  (h_intersections : ∀ x y, ((y = a * x^2 - 4 ∧ y = 0) ∨ (y = 8 - b * x^2 ∧ y = 0)) ∧
    (x ≠ 0 ∧ y ≠ 0))
  (h_kite_area : 1 / 2 * 12 * 4 * real.sqrt (1 / a) = 24) :
  a + b = 3 :=
sorry -- Proof is omitted

end parabola_kite_area_l18_18291


namespace problem_1_problem_2_problem_3_l18_18860

-- Definitions
def a : ℝ × ℝ := (1, Real.sqrt 3)
def b : ℝ × ℝ := (-2, 0)
def t : set ℝ := Icc (-1 : ℝ) 1

-- Problem 1: |a - b| = 2√3
theorem problem_1 : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 2 * Real.sqrt 3 := sorry

-- Problem 2: The angle between (a - b) and b is 5π/6
theorem problem_2 :
  let ab := (a.1 - b.1, a.2 - b.2) in
  Real.arccos ((ab.1 * b.1 + ab.2 * b.2) / (Real.sqrt (ab.1^2 + ab.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 5 * Real.pi / 6 := sorry

-- Problem 3: Range of |a - t * b| for t ∈ [-1, 1] is [√3, 2√3]
theorem problem_3 :
  let ab_t (t : ℝ) := (a.1 - t * b.1, a.2 - t * b.2) in
  set.range (λ t : ℝ, Real.sqrt ((ab_t t).1^2 + (ab_t t).2^2)) = set.interval (Real.sqrt 3) (2 * Real.sqrt 3) := sorry

end problem_1_problem_2_problem_3_l18_18860


namespace tangent_line_through_origin_to_circle_in_third_quadrant_l18_18022

theorem tangent_line_through_origin_to_circle_in_third_quadrant :
  ∃ m : ℝ, (∀ x y : ℝ, y = m * x) ∧ (∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0) ∧ (x < 0 ∧ y < 0) ∧ y = -3 * x :=
sorry

end tangent_line_through_origin_to_circle_in_third_quadrant_l18_18022


namespace value_of_product_l18_18153

theorem value_of_product (a b : ℝ) (h1 : ∀ x : ℝ, 2 * x - a < 1 → x > 2 * b + 3) :
  (a+1) * (b-1) = -6 :=
by
  have h11 : 0 < 1, from zero_lt_one
  have h2 := h1 (2 * (b + 3) - (a + 1) / 2 - 1)
  have h3 := λ x : ℝ, x < (a+1) / 2 ∧ x > 2 * b + 3  
  have h4 := h3 (a / 2 - 3 * b - 1)
  sorry

end value_of_product_l18_18153


namespace power_function_at_one_fourth_l18_18831

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_at_one_fourth (α : ℝ) (h : power_function α 4 = 2) :
  power_function α (1/4) = 1/2 :=
begin
  have hα : α = 1 / 2,
  { sorry },
  rw hα,
  unfold power_function,
  simp,
end

end power_function_at_one_fourth_l18_18831


namespace vents_per_zone_l18_18551

theorem vents_per_zone (total_cost : ℝ) (number_of_zones : ℝ) (cost_per_vent : ℝ) (h_total_cost : total_cost = 20000) (h_zones : number_of_zones = 2) (h_cost_per_vent : cost_per_vent = 2000) : 
  (total_cost / cost_per_vent) / number_of_zones = 5 :=
by 
  sorry

end vents_per_zone_l18_18551


namespace is_isosceles_triangle_l18_18173

theorem is_isosceles_triangle 
  (A B C O A1 B1 C1 : Type) 
  (h_side_ab : ∀ ABC : Triangle, ABC.AB = c) 
  (h_side_bc : ∀ ABC : Triangle, ABC.BC = a)
  (h_side_ca : ∀ ABC : Triangle, ABC.CA = b) 
  (h_angle_bisectors : ∀ ABC : Triangle, ABC.angle_bisectors AA1 BB1 CC1 = O) 
  (h_area_equal : area (triangle O C1 B) = area (triangle O B1 C)) : 
  is_isosceles ABC :=
sorry

end is_isosceles_triangle_l18_18173


namespace sum_of_inverses_of_squares_lt_fraction_l18_18586

theorem sum_of_inverses_of_squares_lt_fraction (n : ℕ) (hn : n ≥ 2) :
  (finset.range n).sum (λ i, 1 / (i + 1)^2) < (2 * n - 1) / n := by
  sorry

end sum_of_inverses_of_squares_lt_fraction_l18_18586


namespace Anna_age_at_marriage_l18_18554

def Josh_and_Anna (J A : ℕ) : Prop :=
  ∃ (marriage_years : ℕ),
  ∃ (current_age_Josh : ℕ),
  ∃ (current_age_combined : ℕ),
  marriage_years = 30 ∧
  J = 22 + marriage_years ∧
  current_age_Josh = 22 + marriage_years ∧
  current_age_combined = 5 * 22 ∧
  current_age_combined = current_age_Josh + A ∧
  (A = current_age_combined - current_age_Josh - marriage_years)

theorem Anna_age_at_marriage : ∃ J A : ℕ, Josh_and_Anna J A :=
by {
  use 52,
  use 28,
  sorry
}

end Anna_age_at_marriage_l18_18554


namespace problem_statement_l18_18128

noncomputable def measureAngleC (a b c : ℝ) (A B C : ℝ) 
  (h1 : ∃ (A B C : ℝ), a = c * (Real.sin B) / (sqrt 3 * b * (Real.cos C)))
  : ℝ := π / 3

noncomputable def areaTriangle (a b c : ℝ) (A B C : ℝ)
  (h1 : ∃ (A B C : ℝ), a = c * (Real.sin B) / (sqrt 3 * b * (Real.cos C)))
  (h2 : c = 3) (h3 : Real.sin A = 2 * Real.sin B)
  : ℝ := 3 / 2

theorem problem_statement
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : ∃ (A B C : ℝ), a = c * (Real.sin B) / (sqrt 3 * b * (Real.cos C))) 
  (h2 : c = 3) (h3 : Real.sin A = 2 * Real.sin B)
  : measureAngleC a b c A B C h1 = π / 3 ∧ areaTriangle a b c A B C h1 h2 h3 = 3 / 2  := by
  sorry

end problem_statement_l18_18128


namespace head_start_l18_18030

theorem head_start (V_b : ℝ) (S : ℝ) : 
  ((7 / 4) * V_b) = V_b → 
  196 = (196 - S) → 
  S = 84 := 
sorry

end head_start_l18_18030


namespace find_a_l18_18888

variable (a b : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_eq_pow : a^b = b^a)
variable (h_eq_b : b = 4 * a)

theorem find_a : a = real.cbrt 4 := by
  sorry

end find_a_l18_18888


namespace find_f_of_2_l18_18228

def f (x : ℝ) : ℝ :=
if x < 1 then 3 * x - 4 else 7 - 3 * x

theorem find_f_of_2 : f 2 = 1 := by
  sorry

end find_f_of_2_l18_18228


namespace not_right_triangle_of_proportion_l18_18045

theorem not_right_triangle_of_proportion (A B C : ℕ)
  (h1 : A + B + C = 180)
  (h2 : A = 3 * k)
  (h3 : B = 4 * k)
  (h4 : C = 5 * k)
  (hk : 3 * k + 4 * k + 5 * k = 180)
  (neqk : k ≠ 0)
  : A < 90 ∧ B < 90 ∧ C < 90 := by
begin
  sorry
end

end not_right_triangle_of_proportion_l18_18045


namespace ordered_pairs_divide_condition_l18_18424

theorem ordered_pairs_divide_condition :
  { (m, n) : ℕ × ℕ // m > 0 ∧ n > 0 ∧ (mn - 1) ∣ (n^3 + 1) } = 
  {(1, 2), (1, 3), (2, 1), (2, 2), (2, 5), (3, 1), (3, 5), (5, 2), (5, 3)} :=
sorry

end ordered_pairs_divide_condition_l18_18424


namespace axis_of_symmetry_of_sin_cos_function_l18_18625

theorem axis_of_symmetry_of_sin_cos_function :
  ∃ k ∈ ℤ, (k:ℝ) * (π / 2) + (5 * π / 12) = - (π / 12) ∧
  (∀ x : ℝ, y = sin (2 * x) - sqrt 3 * cos (2 * x) → y = 2 * sin ((2 * x) - (π / 3))) :=
sorry

end axis_of_symmetry_of_sin_cos_function_l18_18625


namespace cos_A_value_l18_18904

theorem cos_A_value (a b c : ℝ) (A B C : ℝ) (h_triangle : a = b * cos C + c * sin A)
  (m : ℝ × ℝ := (sqrt(3) * b - c, cos C))
  (n : ℝ × ℝ := (a, cos A))
  (h_parallel : m.1 * n.2 = m.2 * n.1) :
  cos A = sqrt(3) / 3 :=
sorry

end cos_A_value_l18_18904


namespace extinction_probability_l18_18717

-- Definitions from conditions
def prob_divide : ℝ := 0.6
def prob_die : ℝ := 0.4

-- Statement of the theorem
theorem extinction_probability :
  ∃ (v : ℝ), v = 2 / 3 :=
by
  sorry

end extinction_probability_l18_18717


namespace common_region_area_l18_18737

noncomputable def rectangle_width := 8
noncomputable def rectangle_height := 2 * sqrt 2
noncomputable def circle_radius := 2
noncomputable def shared_center := true

theorem common_region_area : 
  let rectangle_area := rectangle_width * rectangle_height
  let circle_area := π * circle_radius ^ 2
  let common_area := 2 * π + 4
  rectangle_has_center (rectangle_width, rectangle_height) shared_center → circle_has_center circle_radius shared_center →
  commonRegionArea (rectangle_width, rectangle_height) circle_radius = 2 * π + 4 :=
by
  sorry

end common_region_area_l18_18737


namespace students_not_enrolled_l18_18908

theorem students_not_enrolled (total_students French German both : ℕ) :
  total_students = 78 →
  French = 41 →
  German = 22 →
  both = 9 →
  total_students - (French + German - both) = 24 :=
by
  intros h_total h_french h_german h_both
  rw [h_total, h_french, h_german, h_both]
  sorry

end students_not_enrolled_l18_18908


namespace relationship_among_a_b_c_l18_18134

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f(x)

def g (f : ℝ → ℝ) (x : ℝ) : ℝ := x * f(x)

theorem relationship_among_a_b_c 
  (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_ineq : ∀ x : ℝ, x < 0 → f(x) + x * (deriv f x) < 0)
  (a := 3^(0.3) * f(3^(0.3)))
  (b := (Real.log π 3) * f(Real.log π 3))
  (c := (Real.log 3 (1 / 9)) * f(Real.log 3 (1 / 9))):
  c > a ∧ a > b := 
sorry

end relationship_among_a_b_c_l18_18134


namespace solve_the_problem_l18_18143
-- Import required Lean libraries

-- Define the given functions and prove the properties
open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ x - a + 1
noncomputable def g (x : ℝ) : ℝ := f (1 / 2) (x + 1 / 2) - 1
noncomputable def F (m : ℝ) (x : ℝ) : ℝ := g (2 * x) - m * g (x - 1)
noncomputable def h (m : ℝ) : ℝ :=
if h: m ≤ 1 then
  1 - 2 * m
else if h: 1 < m ∧ m < 2 then
  -m ^ 2
else
  4 - 4 * m

-- Define the theorem with equivalent conditions and answer
theorem solve_the_problem :
  (∃ a : ℝ, 0 < a ∧ a ≠ 1 ∧ f a (1/2) = 2 ∧ a = 1/2) 
  ∧ (∀ x : ℝ, g x = (1/2) ^ x) 
  ∧ (∀ m : ℝ, ∀ x ∈ Icc (-1 : ℝ) (0 : ℝ), h m = (if h: m ≤ 1 then 1 - 2 * m else if h: 1 < m ∧ m < 2 then -m ^ 2 else 4 - 4 * m))
  := by
  sorry

end solve_the_problem_l18_18143


namespace find_solutions_l18_18427

theorem find_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + 4^y = 5^z ↔ (x = 3 ∧ y = 2 ∧ z = 2) ∨ (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 11 ∧ y = 1 ∧ z = 3) :=
by sorry

end find_solutions_l18_18427


namespace smallest_z_value_l18_18926

-- Definitions: w, x, y, and z as consecutive even positive integers
def consecutive_even_cubes (w x y z : ℤ) : Prop :=
  w % 2 = 0 ∧ x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧
  w < x ∧ x < y ∧ y < z ∧
  x = w + 2 ∧ y = x + 2 ∧ z = y + 2

-- Problem statement: Smallest possible value of z
theorem smallest_z_value :
  ∃ w x y z : ℤ, consecutive_even_cubes w x y z ∧ w^3 + x^3 + y^3 = z^3 ∧ z = 12 :=
by
  sorry

end smallest_z_value_l18_18926


namespace associate_professors_bring_2_pencils_l18_18050

theorem associate_professors_bring_2_pencils (A B P : ℕ) 
  (h1 : A + B = 5)
  (h2 : P * A + B = 10)
  (h3 : A + 2 * B = 5)
  : P = 2 :=
by {
  -- Proof goes here
  sorry
}

end associate_professors_bring_2_pencils_l18_18050


namespace repeating_pattern_sum_23_l18_18023

def repeating_pattern_sum (n : ℕ) : ℤ :=
  let pattern := [4, -3, 2, -1, 0]
  let block_sum := List.sum pattern
  let complete_blocks := n / pattern.length
  let remainder := n % pattern.length
  complete_blocks * block_sum + List.sum (pattern.take remainder)

theorem repeating_pattern_sum_23 : repeating_pattern_sum 23 = 11 := 
  sorry

end repeating_pattern_sum_23_l18_18023


namespace quadratic_positive_range_l18_18800

theorem quadratic_positive_range (k : ℝ) (h : k ∈ set.Icc (-1 : ℝ) 1) (x : ℝ) :
  let f := λ x : ℝ, x^2 + (k - 4) * x - 2 * k + 4 in
  (∀ x, f x > 0) ↔ (x < 1 ∨ x > 3) :=
sorry

end quadratic_positive_range_l18_18800


namespace unique_concatenations_and_multiplications_l18_18160

-- Define the Lean theorem statement corresponding to the problem
theorem unique_concatenations_and_multiplications : ∃ n : ℕ, n = 7 :=
by
  -- Given conditions
  have h1 : ℕ := 5, -- We have exactly five '5's

  -- Define the set of partitions corresponding to unique ways of concatenation and multiplication
  let partitions := {
    {55555},
    {5555, 5},
    {555, 55},
    {555, 5, 5},
    {55, 55, 5},
    {55, 5, 5, 5},
    {5, 5, 5, 5, 5}
  },

  -- Since these partitions give us unique ways 
  -- Prove that the size of partitions is 7.
  let possible_combinations := partitions.to_finset.card,
  have : possible_combinations = 7 := sorry,

  -- Conclude the proof
  exact ⟨possible_combinations, this⟩

end unique_concatenations_and_multiplications_l18_18160


namespace log_exp_identity_l18_18164

noncomputable def a : ℝ := Real.log 3 / Real.log 4

theorem log_exp_identity : 2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 := 
by
  sorry

end log_exp_identity_l18_18164


namespace determine_n_l18_18204

def L : ℚ := ∑ i in finset.range 16 \ {0}, 1 / (i:ℚ)

def T_n (n : ℕ) : ℚ := (n * 16^(n-1)) * L + 1

theorem determine_n : ∃ n : ℕ, T_n n ∈ ℤ := 
begin
  use 15015,
  sorry
end

end determine_n_l18_18204


namespace max_pogs_l18_18404

theorem max_pogs (x y z : ℕ) (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : z ≥ 1) (h4 : 3 * x + 4 * y + 9 * z = 75) :
  z ≤ 7 :=
begin
  sorry
end

end max_pogs_l18_18404


namespace count_integers_modulo_l18_18875

theorem count_integers_modulo (n : ℕ) (h₁ : n < 500) (h₂ : n % 7 = 4) : (setOf (λ n, n > 0 ∧ n < 500 ∧ n % 7 = 4)).card = 71 :=
sorry

end count_integers_modulo_l18_18875


namespace annie_accident_chance_l18_18738

def temperature_effect (temp: ℤ) : ℚ := ((32 - temp) / 3 * 5)

def road_condition_effect (condition: ℚ) : ℚ := condition

def wind_speed_effect (speed: ℤ) : ℚ := if (speed > 20) then ((speed - 20) / 10 * 3) else 0

def skid_chance (temp: ℤ) (condition: ℚ) (speed: ℤ) : ℚ :=
  temperature_effect temp + road_condition_effect condition + wind_speed_effect speed

def accident_chance (skid_chance: ℚ) (tire_effect: ℚ) : ℚ :=
  skid_chance * tire_effect

theorem annie_accident_chance :
  (temperature_effect 8 + road_condition_effect 15 + wind_speed_effect 35) * 0.75 = 43.5 :=
by sorry

end annie_accident_chance_l18_18738


namespace total_letters_sent_l18_18248

-- Define the number of letters sent in each month
def letters_in_January : ℕ := 6
def letters_in_February : ℕ := 9
def letters_in_March : ℕ := 3 * letters_in_January

-- Theorem statement: the total number of letters sent across the three months
theorem total_letters_sent :
  letters_in_January + letters_in_February + letters_in_March = 33 := by
  sorry

end total_letters_sent_l18_18248


namespace smallest_nat_divisible_by_48_squared_l18_18796

theorem smallest_nat_divisible_by_48_squared :
  ∃ n : ℕ, (n % (48^2) = 0) ∧ 
           (∀ (d : ℕ), d ∈ (Nat.digits n 10) → d = 0 ∨ d = 1) ∧ 
           (n = 11111111100000000) := sorry

end smallest_nat_divisible_by_48_squared_l18_18796


namespace solution_largest_n_l18_18728

noncomputable def largest_n (n : ℕ) : Prop :=
  ∃ mark_cells : fin n → fin n, 
    ∀ (r1 r2 c1 c2 : fin n), 
      (r1 ≤ r2 ∧ c1 ≤ c2 ∧ (r2 + 1 - r1) * (c2 + 1 - c1) ≥ n) → 
      (∃ i, r1 ≤ mark_cells i ∧ mark_cells i ≤ r2 ∧ c1 ≤ i ∧ i ≤ c2)

theorem solution_largest_n : ∃ n : ℕ, largest_n n ∧ n = 7 := 
by
  sorry

end solution_largest_n_l18_18728


namespace point_on_parabola_distance_to_directrix_is_4_l18_18896

noncomputable def distance_from_point_to_directrix (x y : ℝ) (directrix : ℝ) : ℝ :=
  abs (x - directrix)

def parabola (t : ℝ) : ℝ × ℝ :=
  (4 * t^2, 4 * t)

theorem point_on_parabola_distance_to_directrix_is_4 (m : ℝ) (t : ℝ) :
  parabola t = (3, m) → distance_from_point_to_directrix 3 m (-1) = 4 :=
by
  sorry

end point_on_parabola_distance_to_directrix_is_4_l18_18896


namespace initial_winner_margin_l18_18532

theorem initial_winner_margin 
  (total_votes : ℕ)
  (lost_to_win_margin : ℤ)
  (delta_votes : ℕ)
  (percentage_margin_post_change : ℚ) 
  (h1 : total_votes = 20000)
  (h2 : delta_votes = 2000)
  (h3 : percentage_margin_post_change = 0.10) :
  let W : ℤ := (total_votes : ℤ) / 2 + (lost_to_win_margin / 2) in
  let L : ℤ := (total_votes : ℤ) / 2 - (lost_to_win_margin / 2) in
  W - L = 2000 / (total_votes : ℤ) * 100 :=
sorry

end initial_winner_margin_l18_18532


namespace max_consecutive_sum_l18_18328

theorem max_consecutive_sum (N a : ℕ) (h : N * (2 * a + N - 1) = 240) : N ≤ 15 :=
by
  -- proof goes here
  sorry

end max_consecutive_sum_l18_18328


namespace anna_age_at_marriage_l18_18553

noncomputable def age_when_married (josh_age_when_married anna_age_when_married josh_age_today combined_age_today : ℕ) : Prop :=
  josh_age_when_married = 22 ∧
  josh_age_today = josh_age_when_married + 30 ∧
  combined_age_today = 5 * josh_age_when_married ∧
  combined_age_today = josh_age_today + (anna_age_when_married + 30)

theorem anna_age_at_marriage :
  ∃ (anna_age_when_married : ℕ),
    age_when_married 22 anna_age_when_married 52 110 ∧ anna_age_when_married = 28 :=
begin
  use 28,
  dsimp [age_when_married],
  refine ⟨rfl, _, _, _⟩,
  { norm_num },
  { norm_num },
  { norm_num }
end

end anna_age_at_marriage_l18_18553


namespace percent_women_fair_hair_l18_18684

theorem percent_women_fair_hair (E : ℕ) :
  (0.3 * E) / (0.75 * E) * 100 = 40 :=
by
  -- Sorry is used to skip the proof step in the theorem statement.
  sorry

end percent_women_fair_hair_l18_18684


namespace exists_distinct_abc_l18_18412

theorem exists_distinct_abc
  (n : ℕ) (p : ℕ) (k : ℕ)
  (hn_even : Even n)
  (hn_square_free : square_free n)
  (hp_prime : Nat.Prime p)
  (hp_coprime : Nat.gcd n p = 1)
  (hp_bound : p ≤ 2 * Nat.sqrt n)
  (hk_condition : p ∣ (n + k^2)) :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = a * b + b * c + c * a := 
  sorry

end exists_distinct_abc_l18_18412


namespace polar_coordinate_conversion_l18_18189

theorem polar_coordinate_conversion (x y : ℝ) (ρ θ : ℝ) 
  (h_xy : (x, y) = (1, -√3)) 
  (h_nonneg : ρ ≥ 0) 
  (h_theta : -π ≤ θ ∧ θ < π) 
  (h_rho : ρ = Real.sqrt (x ^ 2 + y ^ 2)) 
  (h_theta_calc : θ = Real.arctan y x) 
  (h_quadrant : (x > 0 ∧ y < 0)) : 
  (ρ, θ) = (2, -π / 3) :=
sorry

end polar_coordinate_conversion_l18_18189


namespace trace_bag_weight_is_two_l18_18652

-- Define the weights of Gordon's shopping bags
def weight_gordon1 : ℕ := 3
def weight_gordon2 : ℕ := 7

-- Summarize Gordon's total weight
def total_weight_gordon : ℕ := weight_gordon1 + weight_gordon2

-- Provide necessary conditions from problem statement
def trace_bags_count : ℕ := 5
def trace_total_weight : ℕ := total_weight_gordon
def trace_one_bag_weight : ℕ := trace_total_weight / trace_bags_count

theorem trace_bag_weight_is_two : trace_one_bag_weight = 2 :=
by 
  -- Placeholder for proof
  sorry

end trace_bag_weight_is_two_l18_18652


namespace final_water_fraction_is_correct_l18_18686

-- Define initial conditions
def initial_water_quarts : ℝ := 18
def initial_antifreeze_quarts : ℝ := 2
def total_quarts : ℝ := 20
def quarts_removed_each_step : ℝ := 6
def steps : ℕ := 4 -- total number of steps including initial state

-- Define the final water quantity using the provided iterative process
def final_water_quantity (initial_water : ℝ) (initial_antifreeze : ℝ) 
(total : ℝ) (removed : ℝ) (steps : ℕ) : ℝ :=
@Nat.recOn (fun _ => ℝ) steps initial_water (fun _ acc => 
acc * (total - removed) / total)

-- State the main theorem
theorem final_water_fraction_is_correct :
  final_water_quantity initial_water_quarts initial_antifreeze_quarts 
  total_quarts quarts_removed_each_step steps / total_quarts = 4.322 / 20 :=
by
  sorry

end final_water_fraction_is_correct_l18_18686


namespace naomi_mean_score_l18_18046

theorem naomi_mean_score (
  s1 s2 s3 s4 s5 s6 : ℝ
  (h_scores : s1 = 82 ∧ s2 = 88 ∧ s3 = 90 ∧ s4 = 91 ∧ s5 = 95 ∧ s6 = 96)
  (h_liam_avg : (s1 + s4 + s5) / 3 = 89)
  (h_naomi : s6 > s2 ∧ s6 - s2 = 10)
) : ((s2 + s3 + s6) / 3 = 91 + 2 / 3) :=
by
  sorry

end naomi_mean_score_l18_18046


namespace necessary_but_not_sufficient_l18_18681

theorem necessary_but_not_sufficient (a b : ℝ) : 
 (a > b) ↔ (a-1 > b+1) :=
by {
  sorry
}

end necessary_but_not_sufficient_l18_18681


namespace cube_volume_expansion_l18_18780

-- State the conditions and the final proof goal.
theorem cube_volume_expansion (a : ℝ) (h : a > 0) :
  ∃ b : ℝ, b = a * real.cbrt 3 ∧ b^3 = 3 * a^3 :=
by 
  use a * real.cbrt 3
  split
  { 
    refl
  }
  { 
    have cq := real.cbrt_pos.mpr (show 3 > 0, by linarith)
    field_simp [real.cbrt_mul]
    ring
  }

end cube_volume_expansion_l18_18780


namespace power_sum_int_l18_18961

theorem power_sum_int {x : ℝ} (hx : ∃ k : ℤ, x + 1/x = k) : ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by
  sorry

end power_sum_int_l18_18961


namespace sum_of_distances_const_l18_18593

/-- Define the structure of an equilateral triangle with given height m -/
structure EquilateralTriangle (m : ℝ) :=
  (A B C : ℝ × ℝ)
  (side_length_eq : dist A B = dist B C ∧ dist B C = dist C A)
  (height_eq : ∃ h, h = m ∧ ∀ P : ℝ × ℝ, (h = dist A P + dist B P + dist C P))

/-- Prove the sum of distances from a point inside an equilateral triangle to its three sides is constant -/
theorem sum_of_distances_const (m : ℝ) (T : EquilateralTriangle m) (P : ℝ × ℝ) :
  let D := dist P T.A
      E := dist P T.B
      F := dist P T.C in
  D + E + F = m :=
sorry

end sum_of_distances_const_l18_18593


namespace geometric_series_formula_l18_18326

theorem geometric_series_formula (a : ℝ) (n : ℕ) (h1 : a ≠ 1) (h2 : n ≠ 0) : 
  (1 + a + a^2 + ... + a^(n+1) = (1 - a^(n+2)) / (1 - a)) :=
by
  sorry

end geometric_series_formula_l18_18326


namespace fraction_of_oranges_picked_l18_18781

-- Definitions based on the problem conditions
def total_trees : ℕ := 8
def fruits_per_tree : ℕ := 200
def remaining_fruits : ℕ := 960

-- Lean statement that proves the fraction of the oranges picked from each tree is 2/5
theorem fraction_of_oranges_picked : 
  ∀ (total_trees fruits_per_tree remaining_fruits : ℕ),
    total_trees = 8 →
    fruits_per_tree = 200 →
    remaining_fruits = 960 →
    let total_fruits := total_trees * fruits_per_tree in
    let picked_fruits := total_fruits - remaining_fruits in
    let picked_fruits_per_tree := picked_fruits / total_trees in
    (picked_fruits_per_tree : ℚ) / fruits_per_tree = 2 / 5 :=
by
  intros total_trees fruits_per_tree remaining_fruits ht hf hr
  rw [ht, hf, hr]
  let total_fruits := total_trees * fruits_per_tree
  have h1 : total_fruits = 1600 := by sorry
  let picked_fruits := total_fruits - remaining_fruits
  have h2 : picked_fruits = 640 := by sorry
  let picked_fruits_per_tree := picked_fruits / total_trees
  have h3 : picked_fruits_per_tree = 80 := by sorry
  have h_fraction : (picked_fruits_per_tree : ℚ) / fruits_per_tree = 80 / 200 := by sorry
  have h_fraction_simplified : 80 / 200 = 2 / 5 := by sorry
  exact h_fraction_simplified
  sorry

end fraction_of_oranges_picked_l18_18781


namespace textbook_weight_ratio_l18_18197

def jon_textbooks_weights : List ℕ := [2, 8, 5, 9]
def brandon_textbooks_weight : ℕ := 8

theorem textbook_weight_ratio : 
  (jon_textbooks_weights.sum : ℚ) / (brandon_textbooks_weight : ℚ) = 3 :=
by 
  sorry

end textbook_weight_ratio_l18_18197


namespace logarithmic_conditions_l18_18162

-- Define the conditions used in the problem
variable (m n : ℝ) 

-- The core statement we need to prove, wrapped in a theorem
theorem logarithmic_conditions (h : log m 3 < log n 3 ∧ log n 3 < 0) : 1 > m ∧ m > n ∧ n > 0 := sorry

end logarithmic_conditions_l18_18162


namespace determine_factorial_l18_18446

-- Define factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

/-- Prove that if 5! * 9! = 12 * N!, then N = 10 -/
theorem determine_factorial (N : ℕ) (h : factorial 5 * factorial 9 = 12 * factorial N) : N = 10 := 
sorry

end determine_factorial_l18_18446


namespace inequality_solution_l18_18607

theorem inequality_solution (x : ℝ) (h : x ≠ -7) : 
  (x^2 - 49) / (x + 7) < 0 ↔ x ∈ Set.Ioo (-∞) (-7) ∪ Set.Ioo (-7) 7 := by 
  sorry

end inequality_solution_l18_18607


namespace Anna_age_at_marriage_l18_18555

def Josh_and_Anna (J A : ℕ) : Prop :=
  ∃ (marriage_years : ℕ),
  ∃ (current_age_Josh : ℕ),
  ∃ (current_age_combined : ℕ),
  marriage_years = 30 ∧
  J = 22 + marriage_years ∧
  current_age_Josh = 22 + marriage_years ∧
  current_age_combined = 5 * 22 ∧
  current_age_combined = current_age_Josh + A ∧
  (A = current_age_combined - current_age_Josh - marriage_years)

theorem Anna_age_at_marriage : ∃ J A : ℕ, Josh_and_Anna J A :=
by {
  use 52,
  use 28,
  sorry
}

end Anna_age_at_marriage_l18_18555


namespace sum_of_all_real_solutions_eq_seven_l18_18099

noncomputable def find_sum_of_solutions : ℝ :=
  let solutions := {x : ℝ | (3^x - 27)^2 + (5^x - 625)^2 = (3^x + 5^x - 652)^2}
  in solutions.sum

theorem sum_of_all_real_solutions_eq_seven :
  find_sum_of_solutions = 7 :=
by
  sorry

end sum_of_all_real_solutions_eq_seven_l18_18099


namespace areaMaximizationDistance_l18_18149

def intersectEllipseLine (m : ℝ) : Finset (ℝ × ℝ) :=
  let ys := {y | y = 2 * m + m}
  let xs := {x | x^2 / 5 + (2 * x + m)^2 = 1}
  (ys ∩ xs).image (λ y x, (x, y))

noncomputable def distanceAB (A B : ℝ × ℝ) : ℝ :=
  let dx := A.1 - B.1
  let dy := A.2 - B.2
  (dx^2 + dy^2)^0.5

noncomputable def maximumAreaDistance (m : ℝ) (A B : ℝ × ℝ) (O : ℝ × ℝ) : ℝ :=
  let d := abs(m) / (5)^0.5
  let ab := distanceAB A B
  0.5 * d * ab

theorem areaMaximizationDistance
  (m : ℝ)
  (A B : ℝ × ℝ)
  (O : ℝ × ℝ := (0, 0))
  (h₁ : A ∈ intersectEllipseLine m)
  (h₂ : B ∈ intersectEllipseLine m)
  (h₃ : maximumAreaDistance m A B O = (5/2) * (5)^0.5) :
  distanceAB A B = (5 * (5)^0.5 * (21 - m^2)^0.5) / (3 * 7) := by
  sorry

end areaMaximizationDistance_l18_18149


namespace curve_C2_eqn_ab_distance_l18_18921

open Real

noncomputable def param_curve_C1 : (ℝ → ℝ × ℝ) := 
  λ α, (sqrt 2 * cos α, sqrt 2 + sqrt 2 * sin α)

noncomputable def param_curve_C2 : (ℝ → ℝ × ℝ) := 
  λ α, (2 * sqrt 2 * cos α, 2 * sqrt 2 + 2 * sqrt 2 * sin α)

theorem curve_C2_eqn : 
  ∀ α : ℝ, 
  let (x₁, y₁) := param_curve_C1 α,
      (x₂, y₂) := param_curve_C2 α
  in x₂ = 2 * x₁ ∧ y₂ = 2 * (y₁ - sqrt 2) + sqrt 2 :=
by { intros, simp [param_curve_C1, param_curve_C2], sorry }

noncomputable def polar_curve_C1 : ℝ → ℝ :=
  λ θ, 2 * sqrt 2 * sin θ

noncomputable def polar_curve_C2 : ℝ → ℝ :=
  λ θ, 4 * sqrt 2 * sin θ

theorem ab_distance : 
  let θ := (π/4 : ℝ),
      ρ₁ := polar_curve_C1 θ,
      ρ₂ := polar_curve_C2 θ
  in |ρ₁ - ρ₂| = 2 :=
by { simp [polar_curve_C1, polar_curve_C2], sorry }

end curve_C2_eqn_ab_distance_l18_18921


namespace distance_between_C_and_D_l18_18832

theorem distance_between_C_and_D 
  (A B : ℝ) 
  (hA : A = 1)
  (hB : B = 3)
  (hC : C = A - 2)
  (hD : D = B + 2) :
  |D - C| = 6 :=
by 
  -- Using the conditions
  rw [hA, hB] at hC hD,
  -- Calculate C, D and the distance between them
  have hC := hC; simp at hC,
  have hD := hD; simp at hD,
  have hDist := hD - hC,
  -- Simplify to get the distance
  norm_num at hDist,
  exact hDist,

end distance_between_C_and_D_l18_18832


namespace line_intersects_curve_C_l18_18544

noncomputable def polar_to_rect_coor (ρ θ : ℝ) : ℝ × ℝ :=
(ρ * Real.cos θ, ρ * Real.sin θ)

theorem line_intersects_curve_C (α : ℝ) :
  let P := polar_to_rect_coor 2 π,
      rect_eq : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x^2 + 3 * y^2 = 12),
      line_eq : ℝ → ℝ × ℝ := λ t, ⟨-2 + t * Real.cos α, t * Real.sin α⟩ in
  ∃ A B : ℝ × ℝ, rect_eq A ∧ rect_eq B ∧ (∃ t₁ t₂ : ℝ, line_eq t₁ = A ∧ line_eq t₂ = B ∧
  (1 / dist P A) + (1 / dist P B) ∈ Set.Icc (Real.sqrt 3 / 2) (Real.sqrt 6 / 2)) :=
by
  sorry

end line_intersects_curve_C_l18_18544


namespace limit_proof_l18_18668

-- Define the function to be evaluated
def function_to_evaluate (h : ℝ) : ℝ := ((3 + h)^2 - 3^2) / h

-- State the limit theorem to be proven
theorem limit_proof : tendsto function_to_evaluate (nhds 0) (nhds 6) :=
sorry

end limit_proof_l18_18668


namespace range_of_x_l18_18267

theorem range_of_x (a : ℝ) (x : ℝ) (h₁ : a = 1) (h₂ : (x - a) * (x - 3 * a) < 0) (h₃ : 2 < x ∧ x ≤ 3) : 2 < x ∧ x < 3 :=
by sorry

end range_of_x_l18_18267


namespace problem1_problem2_problem3_l18_18119

-- 1. Expression for f(x)
theorem problem1 (m n : ℝ) (h_even : ∀ x, f x = f (-x)) (h_min : ∃ x, f x = 1) :
  ∃ (f : ℝ → ℝ), f = λ x, x^2 + 1 :=
begin
  -- Define the quadratic function
  let f := λ x, x^2 + mx + n,
  -- Prove the statement based on the conditions
  sorry
end

-- 2. Inequality for g(x)
theorem problem2 (x : ℝ) :
  let g := λ x, 6 * x / (x^2 + 1) in
  (g (2^x) > 2^x) ↔ (x < (1 / 2) * (Real.log 5 / Real.log 2)) :=
begin
  -- Simplify the inequality and prove
  sorry
end

-- 3. Maximum value for k
theorem problem3 (m n : ℝ) (h : ∀ x ∈ Icc (-1:ℝ) 1, abs (f x) ≤ M) : 
  M ≥ 1/2 :=
begin
  -- Define the absolute value function
  let f := λ x, x^2 + mx + n,
  -- Prove the statement based on the conditions
  sorry
end

end problem1_problem2_problem3_l18_18119


namespace most_calories_per_dollar_l18_18196

structure FoodOption :=
  (price : ℝ)
  (quantity : ℕ)
  (calories_per_unit : ℝ)
  (protein_per_unit : ℝ)
  (tax_rate : ℝ := 0)
  (discount_rate : ℝ := 0)
  (buy_one_get_one_free_on : ℕ := 0)

def final_cost (option : FoodOption) : ℝ :=
  let base_cost := option.price
  let cost_with_tax := base_cost * (1 + option.tax_rate)
  let cost_with_discount := base_cost * (1 - option.discount_rate)
  if option.tax_rate > 0 then cost_with_tax else cost_with_discount

def total_calories (option : FoodOption) : ℝ :=
  let bonus_quantity := if option.buy_one_get_one_free_on > 0 
    then (option.quantity / option.buy_one_get_one_free_on) 
    else 0
  (option.quantity + bonus_quantity) * option.calories_per_unit

def calories_per_dollar (option : FoodOption) : ℝ :=
  total_calories(option) / final_cost(option)

-- Define the food options based on the given problem
def burritos := FoodOption.mk 6 10 120 6 0.1 0 0
def burgers := FoodOption.mk 8 5 400 20 0 0.05 0
def pizza := FoodOption.mk 10 8 300 12 0.15 0 0
def donuts := FoodOption.mk 12 15 250 3 0 0 3

theorem most_calories_per_dollar : calories_per_dollar(donuts) = max (calories_per_dollar(burritos))
              (max (calories_per_dollar(burgers))(calories_per_dollar(pizza))) ∧ 
              calories_per_dollar(donuts) - calories_per_dollar(burgers) = 153.51 :=
by
  sorry

end most_calories_per_dollar_l18_18196


namespace inequality_solution_l18_18605

theorem inequality_solution (x : ℝ) (h : x ≠ -7) : 
  (x^2 - 49) / (x + 7) < 0 ↔ x ∈ Set.Ioo (-∞) (-7) ∪ Set.Ioo (-7) 7 := by 
  sorry

end inequality_solution_l18_18605


namespace sum_of_leading_digits_l18_18360

-- Define the problem conditions
def M : ℕ := 888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888 (a 210-digit number where each digit is 8)
def g (r : ℕ) : ℕ := -- A function that gives the leading digit of the r-th root of M

-- Define individual leading digits
def g2 : ℕ := g 2
def g3 : ℕ := g 3
def g4 : ℕ := g 4
def g5 : ℕ := g 5
def g6 : ℕ := g 6

-- Define the mathematical proof problem
theorem sum_of_leading_digits : g2 + g3 + g4 + g5 + g6 = 8 :=
sorry

end sum_of_leading_digits_l18_18360


namespace circumcenter_of_BL_circumcircle_of_AB_l18_18122

variables (A B C D L E : Type) [EuclideanGeometry]

-- Definitions and conditions
def is_isosceles_trapezoid (AB CD AD : Type) : Prop :=
parallel AB CD ∧ non_parallel AD

def angle_bisector_at_B (A B D L : Type) : Prop :=
  is_angle_bisector_of (angle A B D) L

def circumcenter_BL (E : Type) : Prop :=
  is_circumcenter_of E (triangle B L D)

-- Theorem statement
theorem circumcenter_of_BL_circumcircle_of_AB:
  ∀ (AB CD AD : Type) (B L D : Type) (E : Type),
  is_isosceles_trapezoid AB CD AD →
  angle_bisector_at_B A B D L →
  circumcenter_BL E →
  lies_on_circumcircle E (circumcircle AB)
:= sorry

end circumcenter_of_BL_circumcircle_of_AB_l18_18122


namespace steve_first_stack_plastic_cups_l18_18266

theorem steve_first_stack_plastic_cups (cups_n : ℕ -> ℕ)
  (h_prop : ∀ n, cups_n (n + 1) = cups_n n + 4)
  (h_second : cups_n 2 = 21)
  (h_third : cups_n 3 = 25)
  (h_fourth : cups_n 4 = 29) :
  cups_n 1 = 17 :=
sorry

end steve_first_stack_plastic_cups_l18_18266


namespace dot_product_ndim_l18_18494

variables {n : ℕ} 
variables {a b : Fin n → ℝ}

theorem dot_product_ndim (a b : Fin n → ℝ) : 
  ∑ i, (a i) * (b i) = (List.range n).sum (λ i, a (Fin.mk i sorry) * b (Fin.mk i sorry)) := 
by 
  sorry

end dot_product_ndim_l18_18494


namespace valid_calendars_are_factorial_l18_18007
open Classical

noncomputable def valid_calendars_counter (M N : ℕ) : ℕ :=
  -- define the conditions for a valid calendar
  let calendar := (fin M × fin N) → bool in
  let red_squares_condition := λ (c : calendar), (card (set_of (λ (p : fin M × fin N), c p = tt)) = 10) in
  let no_N_consecutive_whites_in_rows := λ (c : calendar),
    ∀ (i : fin M) (j : fin (N - 1)), 
      ¬ (∀ k : fin N, k < j + N → c ⟨i, k⟩ = ff) in
  let no_M_consecutive_whites_in_columns := λ (c : calendar),
    ∀ (i : fin M) (j : fin (M - 1)), 
      ¬ (∀ k : fin M, k < j + M → c ⟨k, i⟩ = ff) in

  -- count the number of valid configurations
  finset.card {c : calendar | red_squares_condition c ∧ no_N_consecutive_whites_in_rows c ∧ no_M_consecutive_whites_in_columns c}

theorem valid_calendars_are_factorial : ∀ M N : ℕ, valid_calendars_counter M N = 10! :=
by
  -- proof goes here
  sorry

end valid_calendars_are_factorial_l18_18007


namespace min_value_f_exist_distinct_a_b_l18_18485

noncomputable def f (x t : ℝ) := Real.log (x^2 + t*x + 1)

theorem min_value_f {t : ℝ} (h : t > -2) :
  (t ≥ 0 → ∀ x ∈ Icc (0:ℝ) 2, f x t ≥ 0 ∧ f 0 t = 0) ∧
  (-2 < t ∧ t < 0 → ∀ x ∈ Icc (0:ℝ) 2, f x t ≥ Real.log (1 - t^2 / 4)) ∧
  (t ≤ -2 → ¬(∃ x ∈ Icc (0:ℝ) 2, true)) :=
begin
  sorry
end

theorem exist_distinct_a_b {t : ℝ} (h : t > -2) :
  (∃ a b ∈ Ioo 0 2, a ≠ b ∧ f a t = Real.log a ∧ f b t = Real.log b) ↔ t ∈ Ioo (-3/2) (-1) :=
begin
  sorry
end

end min_value_f_exist_distinct_a_b_l18_18485


namespace second_number_in_sequence_l18_18089

theorem second_number_in_sequence :
  let seq := (List.range' 1 100).map (λ n => if n % 2 = 0 then (-1)^(n // 2) * (101 - n)^2 else 0) in
  seq.sum = 5050 →
  (seq.nth 1).getD 0 = 199 :=
by
  let seq := (List.range' 1 100).map (λ n => if n % 2 = 0 then (-1)^(n // 2) * (101 - n)^2 else 0)
  assume h : seq.sum = 5050
  sorry

end second_number_in_sequence_l18_18089


namespace minimal_degree_condition_l18_18987

-- Definition of the Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n + 1) + fib n

-- Define minimal degree condition for a polynomial satisfying certain Fibonacci values
theorem minimal_degree_condition 
  (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) :
  ∃ f : polynomial ℕ, 
    (∀ k : ℕ, k ≤ n → f.eval k = fib (m + k)) →
    (minimal_degree f) = if m = n then n - 1 else n :=
sorry

end minimal_degree_condition_l18_18987


namespace largest_median_of_eleven_numbers_l18_18088

-- Definitions for given conditions
def given_numbers : List ℕ := [2, 3, 4, 5, 6, 7, 8]

def remaining_numbers_condition (l : List ℕ) : Prop :=
  ∃ x ∈ l, x ≥ 10

def combined_list (additional_numbers : List ℕ) : List ℕ :=
  given_numbers ++ additional_numbers

def is_sorted (l : List ℕ) : Prop :=
  l = l.sort (fun x y => x ≤ y)

def median (l : List ℕ) : ℕ :=
  (l.sort (fun x y => x ≤ y)).nthLe (5) (by linarith [List.length_sort])

-- Main theorem statement
theorem largest_median_of_eleven_numbers
  (additional_numbers : List ℕ)
  (h1 : List.length additional_numbers = 4)
  (h2 : remaining_numbers_condition additional_numbers) :
  ∃ l : List ℕ, combined_list additional_numbers = l ∧ is_sorted l ∧ median l = 7 := sorry

end largest_median_of_eleven_numbers_l18_18088


namespace solution_1_solution_2_l18_18750

noncomputable def problem_1 : ℚ :=
  (25 / 9) ^ (1 / 2) + (27 / 64) ^ (-2 / 3) + (0.1) ^ (-2) - 3 * (π ^ 0)

theorem solution_1 : problem_1 = (904 / 9) :=
by
  sorry

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

noncomputable def problem_2 : ℝ :=
  Real.log (1 / 2) / Real.log 10 - Real.log (5 / 8) / Real.log 10 + Real.log 12.5 / Real.log 10 - log_base 8 9 * log_base 27 8

theorem solution_2 : problem_2 = (1 / 3) :=
by
  sorry

end solution_1_solution_2_l18_18750


namespace conic_is_hyperbola_l18_18274

noncomputable def polar_equation (theta : Real) : Real :=
  1 / (1 - Real.cos theta + Real.sin theta)

theorem conic_is_hyperbola (theta : Real) :
  let rho := polar_equation theta in
  let e := Real.sqrt 2 in
  e > 1 ∧ e.is_hyperbola :=
sorry

end conic_is_hyperbola_l18_18274


namespace dogs_with_pointy_ears_l18_18584

theorem dogs_with_pointy_ears (total_dogs with_spots with_pointy_ears: ℕ) 
  (h1: with_spots = total_dogs / 2)
  (h2: total_dogs = 30) :
  with_pointy_ears = total_dogs / 5 :=
by
  sorry

end dogs_with_pointy_ears_l18_18584


namespace approximate_root_l18_18168

def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x - 2

theorem approximate_root (h1 : f 1 = -2) 
                        (h2 : f 1.5 = 0.625)
                        (h3 : f 1.25 = -0.984)
                        (h4 : f 1.375 = -0.26)
                        (h5 : f 1.4375 = 0.162)
                        (h6 : f 1.40625 = -0.054) : 
  |1.4 - x| < 0.1 :=
sorry

end approximate_root_l18_18168


namespace number_of_full_rows_in_first_field_l18_18370

-- Define the conditions
def total_corn_cobs : ℕ := 116
def rows_in_second_field : ℕ := 16
def cobs_per_row : ℕ := 4
def cobs_in_second_field : ℕ := rows_in_second_field * cobs_per_row
def cobs_in_first_field : ℕ := total_corn_cobs - cobs_in_second_field

-- Define the theorem to be proven
theorem number_of_full_rows_in_first_field : 
  cobs_in_first_field / cobs_per_row = 13 :=
by
  sorry

end number_of_full_rows_in_first_field_l18_18370


namespace solution_set_l18_18421

-- Definitions based on given conditions
def inequality (x : ℝ) : Prop := x^2 - 2 * x ≤ 0

-- The theorem to prove
theorem solution_set : {x : ℝ | inequality x} = set.Icc 0 2 := 
by
  sorry

end solution_set_l18_18421


namespace sum_of_squares_of_consecutive_even_integers_l18_18636

theorem sum_of_squares_of_consecutive_even_integers : 
  ∀ (a : ℝ), (2a-2) * 2a * (2a+2) = 72a → (2a-2)^2 + (2a)^2 + (2a+2)^2 = 468 := 
by
  intros a h
  sorry

end sum_of_squares_of_consecutive_even_integers_l18_18636


namespace knives_percentage_l18_18056

-- Definitions based on conditions
def initial_knives : ℕ := 6
def initial_forks : ℕ := 12
def initial_spoons : ℕ := 3 * initial_knives
def traded_knives : ℕ := 10
def traded_spoons : ℕ := 6

-- Definitions for calculations
def final_knives : ℕ := initial_knives + traded_knives
def final_spoons : ℕ := initial_spoons - traded_spoons
def total_silverware : ℕ := final_knives + final_spoons + initial_forks

-- Theorem to prove the percentage of knives
theorem knives_percentage : (final_knives * 100) / total_silverware = 40 := by
  sorry

end knives_percentage_l18_18056


namespace axis_of_symmetry_sin_l18_18281

def is_axis_of_symmetry (x : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, f (2*x - t) = f t

theorem axis_of_symmetry_sin :
  is_axis_of_symmetry (π / 12) (λ x : ℝ, Real.sin (2 * x + π / 3)) :=
sorry

end axis_of_symmetry_sin_l18_18281


namespace vector_magnitude_sum_l18_18807

noncomputable section

open Real

/-- Given two vectors a = (1, x) and b = (x + 2, -2) in ℝ^2 that are perpendicular, 
prove that the magnitude of their sum is equal to 5. -/
theorem vector_magnitude_sum (x : ℝ)
  (a := (1, x) : ℝ × ℝ)
  (b := (x + 2, -2) : ℝ × ℝ)
  (h : a.1 * b.1 + a.2 * b.2 = 0) :
  sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) = 5 :=
by
  sorry

end vector_magnitude_sum_l18_18807


namespace solve_for_n_l18_18364

def number_of_balls : ℕ := sorry

axiom A : number_of_balls = 2

theorem solve_for_n (n : ℕ) (h : (1 + 1 + n = number_of_balls) ∧ ((n : ℝ) / (1 + 1 + n) = 1 / 2)) : n = 2 :=
sorry

end solve_for_n_l18_18364


namespace find_x_plus_3y_l18_18536

variables {α : Type*} {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (x y : ℝ)
variables (OA OB OC OD OE : V)

-- Defining the conditions
def condition1 := OA = (1/2) • OB + x • OC + y • OD
def condition2 := OB = 2 • x • OC + (1/3) • OD + y • OE

-- Writing the theorem statement
theorem find_x_plus_3y (h1 : condition1 x y OA OB OC OD) (h2 : condition2 x y OB OC OD OE) : 
  x + 3 * y = 7 / 6 := 
sorry

end find_x_plus_3y_l18_18536


namespace number_of_incorrect_statements_l18_18634

-- Definitions for the given problem conditions
def statement_1 := ∀ (data : List ℝ) (c : ℝ), (data.map (λ x => x + c)).variance = data.variance
def statement_2 := ∀ (x y : ℝ), y = 3 - 5 * x → ∀ (Δx : ℝ), (Δx = 1) → (y : ℝ) = y - 5 * Δx
def statement_3 := ∀ (x y : ℝ) (r : ℝ), (0 ≤ |r| ∧ |r| ≤ 1) → |r| = 0 → x.correlation y r <--> stronger linearity
def statement_4 := ∀ (k_squared : ℝ), k_squared > 0 → k_squared ^ 2 > k_squared

-- The proof problem to verify the number of incorrect statements.
theorem number_of_incorrect_statements : 
  (¬ statement_2) ∧ (¬ statement_3) ∧ statement_1 ∧ statement_4 → (number_of_incorrect_statements = 2) :=
by
  sorry

end number_of_incorrect_statements_l18_18634


namespace imag_part_of_z_l18_18476

noncomputable def sqrt2 : ℂ := complex.sqrt 2

theorem imag_part_of_z (z : ℂ)
  (h : z * (1 - complex.I) = sqrt2 + complex.I) :
  complex.im z = (sqrt2 + 1) / 2 :=
sorry

end imag_part_of_z_l18_18476


namespace john_speed_with_dog_l18_18933

-- Define the conditions as constants
def john_speed_alone : ℝ := 4
def time_with_dog : ℝ := 0.5
def time_alone : ℝ := 0.5
def total_distance : ℝ := 5

-- Define the statement to prove John's speed with his dog
theorem john_speed_with_dog : 
  let distance_alone := john_speed_alone * time_alone in
  let distance_with_dog := total_distance - distance_alone in
  let speed_with_dog := distance_with_dog / time_with_dog in
  speed_with_dog = 6 := 
by 
  sorry

end john_speed_with_dog_l18_18933


namespace inequality_solution_l18_18611

variable {x : ℝ}

theorem inequality_solution :
  x ∈ Set.Ioo (-∞ : ℝ) 7 ∪ Set.Ioo (-∞ : ℝ) (-7) ∪ Set.Ioo (-7) 7 ↔ (x^2 - 49) / (x + 7) < 0 :=
by
  sorry

end inequality_solution_l18_18611


namespace probability_at_most_one_art_between_cultural_is_0_l18_18693

/-- Define the total number of classes and their types --/
def total_classes := ["Chinese", "Mathematics", "Foreign Language", "Art", "Art", "Art"]

/-- Define a function to calculate the probability of having at most one art class 
between any two adjacent cultural classes --/
def probability_at_most_one_art_between_cultural (classes : List String) : ℝ :=
  -- (Note: The detailed calculation would go here, but the proof is not required)
  sorry

theorem probability_at_most_one_art_between_cultural_is_0.6 :
  probability_at_most_one_art_between_cultural total_classes = 0.6 :=
by
  -- (The detailed proof would go here, but it's not required as per the instructions)
  sorry

end probability_at_most_one_art_between_cultural_is_0_l18_18693


namespace second_horse_revolutions_l18_18374

noncomputable def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius
noncomputable def distance_traveled (circumference : ℝ) (revolutions : ℕ) : ℝ := circumference * (revolutions : ℝ)
noncomputable def revolutions_needed (distance : ℝ) (circumference : ℝ) : ℕ := ⌊distance / circumference⌋₊

theorem second_horse_revolutions :
  let r1 := 30
  let r2 := 10
  let revolutions1 := 40
  let c1 := circumference r1
  let c2 := circumference r2
  let d1 := distance_traveled c1 revolutions1
  (revolutions_needed d1 c2) = 120 :=
by
  sorry

end second_horse_revolutions_l18_18374


namespace carolyn_silverware_knives_percentage_l18_18060

theorem carolyn_silverware_knives_percentage :
  (let knives_initial := 6 in
   let forks_initial := 12 in
   let spoons_initial := 3 * knives_initial in
   let total_silverware_initial := knives_initial + forks_initial + spoons_initial in
   let knives_after_trade := 0 in
   let spoons_after_trade := spoons_initial + 6 in
   let total_silverware_after_trade := knives_after_trade + forks_initial + spoons_after_trade in
   percentage_knives := (knives_after_trade * 100) / total_silverware_after_trade in
   percentage_knives = 0) :=
by
  sorry

end carolyn_silverware_knives_percentage_l18_18060


namespace probability_P_plus_S_mod_7_eq_3_l18_18322

theorem probability_P_plus_S_mod_7_eq_3 :
  let count_all_pairs := (100.choose 2)
  let valid_pairs := count_all_pairs - 225 + 15 in
  (valid_pairs : ℚ) / count_all_pairs = 237 / 247 := by
  let a b : ℕ := sorry
  let S := a + b
  let P := a * b
  have valid_pairs_condition : ¬(a = b) ∧ (P + S) % 7 = 3 := sorry
  sorry

end probability_P_plus_S_mod_7_eq_3_l18_18322


namespace terminating_decimal_expansion_l18_18106

theorem terminating_decimal_expansion : (11 / 125 : ℝ) = 0.088 := 
by
  sorry

end terminating_decimal_expansion_l18_18106


namespace find_possible_first_term_l18_18645

noncomputable def geometric_sequence_first_term (a r : ℝ) : Prop :=
  (a * r^2 = 3) ∧ (a * r^4 = 27)

theorem find_possible_first_term (a r : ℝ) (h : geometric_sequence_first_term a r) :
    a = 1 / 3 :=
by
  sorry

end find_possible_first_term_l18_18645


namespace problem_statement_l18_18110

def P (a b x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + b*x^2 + a*x + 1

noncomputable def count_pairs_with_one_real_root : ℕ :=
  let pairs := [(1, 2), (3, 5), (5, 7), (7, 11)] in
  pairs.count (λ (ab : ℝ × ℝ), let a := ab.1 in let b := ab.2 in
    let Q := λ x : ℝ, x^4 + (a-1)*x^3 + (b-a+1)*x^2 + (a-1)*x + 1 in
    let Q_no_real_roots := ∀ x : ℝ, Q x ≠ 0 in
    ∃ x : ℝ, x = -1 ∧ Q_no_real_roots)
    
theorem problem_statement : count_pairs_with_one_real_root = 2 := sorry

end problem_statement_l18_18110


namespace negation_of_proposition_l18_18633

theorem negation_of_proposition
  (h : ∀ x : ℝ, x^2 - 2 * x + 2 > 0) :
  ∃ x : ℝ, x^2 - 2 * x + 2 ≤ 0 :=
sorry

end negation_of_proposition_l18_18633


namespace self_intersections_l18_18241

noncomputable def numberOfSelfIntersections (α : ℝ) (h : 0 < α ∧ α < 180) : ℝ :=
  Real.floor (180 / α)

theorem self_intersections (α : ℝ) (h : 0 < α ∧ α < 180) :
  ∃ n : ℕ, n = numberOfSelfIntersections α h := 
  sorry

end self_intersections_l18_18241


namespace point_b_after_move_l18_18243

theorem point_b_after_move (A : ℤ) (h : A = -2) : A + 3 = 1 :=
by
  rw h
  norm_num

end point_b_after_move_l18_18243


namespace problem1_problem2_l18_18355

-- Problem 1: Calculation Proof
theorem problem1 : (3 - Real.pi)^0 - Real.sqrt 4 + 4 * Real.sin (Real.pi * 60 / 180) + |Real.sqrt 3 - 3| = 2 + Real.sqrt 3 :=
by
  sorry

-- Problem 2: Inequality Systems Proof
theorem problem2 (x : ℝ) :
  (5 * (x + 3) > 4 * x + 8) ∧ (x / 6 - 1 < (x - 2) / 3) → x > -2 :=
by
  sorry

end problem1_problem2_l18_18355


namespace eccentricity_of_ellipse_l18_18579

theorem eccentricity_of_ellipse (a b : ℝ) (P F1 F2 : ℝ × ℝ) 
  (ha : a > b) (hb : b > 0) 
  (hP : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1)
  (h_angle_PF1F2 : ∀ (F1 F2 P : ℝ × ℝ), angle F1 P F2 = 75) 
  (h_angle_PF2F1 : ∀ (F1 F2 P : ℝ × ℝ), angle F2 P F1 = 15) : 
  eccentricity a b = (sqrt 6) / 3 :=
sorry

end eccentricity_of_ellipse_l18_18579


namespace symmetric_graph_l18_18577

theorem symmetric_graph {f : ℝ → ℝ} (a : ℝ) (h : ∀ x : ℝ, f(a - x) = -f(a + x)) :
  ∀ x : ℝ, f(2 * a - x) = -f(x) :=
by
  intros x
  have H := h (a - x)
  rw [sub_sub_cancel] at H
  rw [← neg_eq_iff_right_eq] at H
  exact H
  -- sorry (Proof here)

end symmetric_graph_l18_18577


namespace trajectory_equation_of_moving_circle_center_constant_sum_lambda_mu_l18_18118

section

variables {S : ℝ × ℝ} (hS : S = (2, 0))
variables {C : ℝ × ℝ → Prop} (hC : ∀ x y, C (x, y) ↔ (y^2 = 4 * x))
variables {l : ℝ → ℝ × ℝ} (hl : ∀ t, l t = (t * y + 2, y) ∧ C (t * y + 2, y))
variables {A B T : ℝ × ℝ}

def find_trajectory_equation : Prop :=
  ∀ x y, C (x, y) → y^2 = 4 * x

def prove_lambda_mu_sum : Prop :=
  ∀ (A B T : ℝ × ℝ) (λ μ : ℝ), 
    (λ = 1 + 2 / (y * t1) ∧ μ = 1 + 2 / (y * t2)) → 
    (λ + μ = 1)

end

theorem trajectory_equation_of_moving_circle_center (hS : S = (2, 0))
    (hC : ∀ x y, C (x, y) ↔ y^2 = 4 * x) :
  find_trajectory_equation := 
begin
  sorry
end

theorem constant_sum_lambda_mu (hS : S = (2, 0))
    (hC : ∀ x y, C (x, y) ↔ y^2 = 4 * x) 
    (hl : ∀ t, l t = (t * y + 2, y) ∧ C (t * y + 2, y)) :
  prove_lambda_mu_sum := 
begin
  sorry
end

end trajectory_equation_of_moving_circle_center_constant_sum_lambda_mu_l18_18118


namespace binomial_coefficient_200_200_l18_18408

theorem binomial_coefficient_200_200 : (Nat.binomial 200 200) = 1 :=
by sorry

end binomial_coefficient_200_200_l18_18408


namespace find_n_from_equation_l18_18137

theorem find_n_from_equation (n m : ℕ) (h1 : (1^m / 5^m) * (1^n / 4^n) = 1 / (2 * 10^31)) (h2 : m = 31) : n = 16 := 
by
  sorry

end find_n_from_equation_l18_18137


namespace find_b_given_conditions_l18_18639

theorem find_b_given_conditions (a b k : ℝ) (h1 : a^2 * real.sqrt b = k)
  (h2 : k = 36) (h3 : a * b = 48) : b = 16 :=
sorry

end find_b_given_conditions_l18_18639


namespace map_strip_to_UHP_l18_18979

noncomputable def maps_strip_to_upper_half_plane (h : ℝ) (z : ℂ) : Prop :=
0 < z.im ∧ z.im < h → (complex.exp (real.pi * z / h)).im > 0

set_option pp.all true

-- Here's the proof statement without the proof itself
theorem map_strip_to_UHP (h : ℝ) (z : ℂ) (hz : 0 < z.im ∧ z.im < h) :
  (complex.exp (real.pi * z / h)).im > 0 :=
sorry

end map_strip_to_UHP_l18_18979


namespace coefficient_of_x_l18_18429

theorem coefficient_of_x : 
  let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2)
  ∃ (a b c : ℝ), expr = a * x^2 + b * x + c ∧ b = 5 := by
    let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2)
    exact sorry

end coefficient_of_x_l18_18429


namespace board_partition_possible_l18_18013

-- Definition of natural numbers m and n greater than 15
variables (m n : ℕ)
-- m > 15
def m_greater_than_15 := m > 15
-- n > 15
def n_greater_than_15 := n > 15

-- Definition of m and n divisibility conditions
def divisible_by_4_or_5 (x : ℕ) : Prop :=
  x % 4 = 0 ∨ x % 5 = 0

def partition_possible (m n : ℕ) : Prop :=
  (m % 4 = 0 ∧ n % 5 = 0) ∨ (m % 5 = 0 ∧ n % 4 = 0)

-- The final statement of Lean
theorem board_partition_possible :
  m_greater_than_15 m → n_greater_than_15 n → partition_possible m n :=
by
  intro h_m h_n
  sorry

end board_partition_possible_l18_18013


namespace peach_pies_l18_18545

theorem peach_pies (total_pies : ℕ) (apple_ratio blueberry_ratio peach_ratio : ℕ)
  (h_ratio : apple_ratio + blueberry_ratio + peach_ratio = 10)
  (h_total : total_pies = 30)
  (h_ratios : apple_ratio = 3 ∧ blueberry_ratio = 2 ∧ peach_ratio = 5) :
  total_pies / (apple_ratio + blueberry_ratio + peach_ratio) * peach_ratio = 15 :=
by
  sorry

end peach_pies_l18_18545


namespace additional_holes_needed_l18_18805

theorem additional_holes_needed
  (circumference : ℕ)
  (initial_interval : ℕ)
  (final_interval : ℕ)
  (initial_holes : ℕ) :
  circumference = 300 →
  initial_interval = 3 →
  final_interval = 5 →
  initial_holes = 30 →
  (let total_final_holes := circumference / final_interval in
   let common_interval := Nat.lcm initial_interval final_interval in
   let common_positions := circumference / common_interval in
   let initial_common_positions := initial_holes / (common_interval / initial_interval) in
   let new_holes_needed := total_final_holes - initial_common_positions in
   new_holes_needed - initial_holes = 20) := 
by 
  intros h1 h2 h3 h4
  let total_final_holes := 300 / 5
  let common_interval := (Nat.lcm 3 5)
  let common_positions := 300 / common_interval
  let initial_common_positions := 30 / (common_interval / 3)
  let new_holes_needed := total_final_holes - initial_common_positions
  show new_holes_needed - 30 = 20
  sorry

end additional_holes_needed_l18_18805


namespace find_good_integer_l18_18028

def is_digit (a : ℕ) : Prop := 0 ≤ a ∧ a ≤ 9

def digits_reorder (n m : ℕ) : Prop :=
  (n.digits).sorted = (m.digits).sorted

def good_integer (n : ℕ) : Prop :=
  digits_reorder (3 * n) n

def four_digit_integer (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def divisible_by (n divisor : ℕ) : Prop :=
  n % divisor = 0

theorem find_good_integer :
  ∃ n : ℕ, four_digit_integer n ∧ divisible_by n 11 ∧ good_integer n ∧ n = 2475 :=
by
  use 2475
  sorry

end find_good_integer_l18_18028


namespace exists_perfect_matching_l18_18216

variables {A B : Type} [Inhabited A] [Inhabited B]
variables {E : set (A × B)}
variables {G : Type} [A_infinite : Infinite A] [B_infinite : Infinite B]

-- Define a bipartite graph structure
noncomputable def bipartite_graph : Prop :=
∃ (A B : set G) E, (A ∩ B = ∅) ∧ (A ∪ B = G) ∧ (∀ e ∈ E, ∃ a ∈ A, ∃ b ∈ B, e = (a, b))

-- Define the degree function as the number of edges incident to a vertex
def degree (v : G) := if h : finite (E.filter (λ e, e.1 = v ∨ e.2 = v)) then h.to_finset.card else 0

-- Matching condition
def perfect_matching (f : A → B) : Prop :=
  (∀ a ∈ A, ∃ b ∈ B, (a, b) ∈ E) ∧
  (∀ x y ∈ A, x ≠ y → f x ≠ f y) ∧ -- injective
  (∀ b ∈ B, ∃ a ∈ A, f a = b)       -- surjective

theorem exists_perfect_matching (h : bipartite_graph ∧
  (∀ v : A ∪ B, 0 < degree v)) :
  ∃ f : A → B, perfect_matching f :=
sorry

end exists_perfect_matching_l18_18216


namespace perp_sufficient_and_necessary_plane_l18_18512

-- Definitions:
-- a and b are lines in 3D space (represented as sets of points),
-- skew (non-intersecting and non-parallel),
-- perpendicularity for lines and planes defined appropriately.

section
variables {Point : Type} [Nonempty Point]

-- Let's assume we have a type for Line consisting of sets of points
structure Line := (pts : set Point)

-- Definitions for skew lines and perpendicularity
def skew (a b : Line) : Prop := 
  ¬(∃ p, p ∈ a.pts ∧ p ∈ b.pts) ∧ 
  ¬ (∃ (v : Line), v ≠ a ∧ v ≠ b ∧ v ⊆ a.pts ∪ b.pts)

def perp (a b : Line) : Prop := 
  ∃ (n : Point → Point → ℝ), 
  (∀ p q ∈ a.pts, n p q = 0) ∧ (∀ p q ∈ b.pts, n p q = 0) ∧ 
  (∀ p ∈ a.pts, ∀ q ∈ b.pts, n p q ≠ 0)

-- Definition for a plane containing a line and being perpendicular to another line
structure Plane := (pts : set Point)

def exists_plane_perp_to_line (a b : Line) : Prop :=
  ∃ α : Plane, (a.pts ⊆ α.pts) ∧ perp (α, b)

-- Problem statement
theorem perp_sufficient_and_necessary_plane {a b : Line} (h_skew : skew a b) : 
  perp a b ↔ exists_plane_perp_to_line a b :=
sorry
end

end perp_sufficient_and_necessary_plane_l18_18512


namespace probability_inside_circle_is_2_div_9_l18_18669

noncomputable def probability_point_in_circle : ℚ := 
  let total_points := 36
  let points_inside := 8
  points_inside / total_points

theorem probability_inside_circle_is_2_div_9 :
  probability_point_in_circle = 2 / 9 :=
by
  -- we acknowledge the mathematical computation here
  sorry

end probability_inside_circle_is_2_div_9_l18_18669


namespace jim_mpg_is_5_l18_18546

variable (total_gas_capacity : ℕ)
variable (gas_fraction_left : ℚ)
variable (distance_to_work_one_way : ℕ)

def total_distance_travelled (d : ℕ) : ℕ := 2 * d

def gas_used (capacity : ℕ) (fraction_left : ℚ) : ℚ :=
  capacity * (1 - fraction_left)

def miles_per_gallon (total_distance : ℕ) (gallons_used : ℚ) : ℚ :=
  total_distance / gallons_used

theorem jim_mpg_is_5 
  (hg_cap : total_gas_capacity = 12)
  (hf_left : gas_fraction_left = 2 / 3)
  (d_work : distance_to_work_one_way = 10) :
  miles_per_gallon (total_distance_travelled d_work) (gas_used total_gas_capacity gas_fraction_left) = 5 := by
  -- The proof goes here
  sorry

end jim_mpg_is_5_l18_18546


namespace james_score_on_fourth_test_l18_18556

theorem james_score_on_fourth_test
  (scores : Fin 5 → ℕ) -- Scores of the 5 tests
  (h_range : ∀ i, 71 ≤ scores i ∧ scores i ≤ 80) -- Scores are integers between 71 and 80, inclusive
  (h_distinct : ∀ i j, i ≠ j → scores i ≠ scores j) -- Scores are different
  (h_avg_integer : ∀ n, n < 5 → ((∑ k in Finset.range (n + 1), scores ⟨k, by linarith⟩ : ℤ) % (n + 1) = 0)) -- Average after each test is an integer
  (h_fifth_score : scores ⟨4, by linarith⟩ = 78) -- Score on the fifth test is 78
  : scores ⟨3, by linarith⟩ = 80 := -- Prove that the score on the fourth test is 80
sorry

end james_score_on_fourth_test_l18_18556


namespace find_w_l18_18824

noncomputable def proof_problem (z : ℂ) (w : ℂ) : Prop :=
  (∃ x y : ℝ, z = x + y * I ∧ (1 + 3 * I) * z.im = 0) ∧
  w = z / (2 + I) ∧
  |w| = 5 * real.sqrt 2

theorem find_w (z : ℂ) (w : ℂ) (h : proof_problem z w) :
  w = 1 + 7 * I ∨ w = -1 - 7 * I :=
sorry

end find_w_l18_18824


namespace cosine_sum_l18_18444

noncomputable def z : ℂ := complex.exp ((π / 11) * complex.I)

theorem cosine_sum :
  z ^ 11 = -1 →
  z ^ 22 = 1 →
  (cos (π / 11) + cos (3 * π / 11) + cos (5 * π / 11) + cos (7 * π / 11) + cos (9 * π / 11)) = 1 / 2 :=
by
  intro h1 h2
  sorry

end cosine_sum_l18_18444


namespace polynomial_division_l18_18438

-- Define the polynomials P and D
noncomputable def P : Polynomial ℤ := 5 * Polynomial.X ^ 4 - 3 * Polynomial.X ^ 3 + 7 * Polynomial.X ^ 2 - 9 * Polynomial.X + 12
noncomputable def D : Polynomial ℤ := Polynomial.X - 3
noncomputable def Q : Polynomial ℤ := 5 * Polynomial.X ^ 3 + 12 * Polynomial.X ^ 2 + 43 * Polynomial.X + 120
def R : ℤ := 372

-- State the theorem
theorem polynomial_division :
  P = D * Q + Polynomial.C R := 
sorry

end polynomial_division_l18_18438


namespace number_of_C_values_of_C_and_m_l18_18465

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 6}

-- Number of sets C that meet the conditions
theorem number_of_C : 
  (∃ C : Set ℕ, C ⊆ A ∧ C ∩ B ≠ ∅ → 
  {C : Set ℕ // C ⊆ A ∧ C ∩ B ≠ ∅}.to_finset.card = 12
) :=
sorry

-- Possible values of set C and m in the quadratic equation
theorem values_of_C_and_m :
  (∃ (C : Set ℕ) (m : ℕ), (C = { x ∈ ℕ | (x^2 - m*x + 4 = 0)} ∧ C ⊆ A) →
    (C = {1, 4} ∧ m = 5) 
    ∨ (C = {2} ∧ m = 4)
  ) :=
sorry

end number_of_C_values_of_C_and_m_l18_18465


namespace pyramid_volume_l18_18272

noncomputable def volume_of_pyramid (AB AD BD AE : ℝ) (p : AB = 9 ∧ AD = 10 ∧ BD = 11 ∧ AE = 10.5) : ℝ :=
  1 / 3 * (60 * (2 ^ (1 / 2))) * (5 * (2 ^ (1 / 2)))

theorem pyramid_volume (AB AD BD AE : ℝ) (h1 : AB = 9) (h2 : AD = 10) (h3 : BD = 11) (h4 : AE = 10.5)
  (V : ℝ) (hV : V = 200) : 
  volume_of_pyramid AB AD BD AE (⟨h1, ⟨h2, ⟨h3, h4⟩⟩⟩) = V :=
sorry

end pyramid_volume_l18_18272


namespace smallest_n_multiple_of_9_l18_18615

theorem smallest_n_multiple_of_9 (x y n : ℤ) (hx : x + 2 ≡ 0 [ZMOD 9]) (hy : y - 2 ≡ 0 [ZMOD 9]) : n = 6 ≡ 0 [ZMOD 9] → x^2 - x * y + y^2 + n ≡ 0 [ZMOD 9] :=
sorry

end smallest_n_multiple_of_9_l18_18615


namespace tangent_line_at_point_l18_18627

theorem tangent_line_at_point (x y : ℝ) (h : y = x^2 - 2 * x + 1) (hx : x = 1) (hy : y = 0) :
  ∃ c : ℝ, c = 1 ∧ ∀ x, ∂ (λ x, x^2 - 2*x + 1)/ ∂x at 1 = 0 → y = c :=
by {
  sorry
}

end tangent_line_at_point_l18_18627


namespace find_percentage_l18_18167

theorem find_percentage (P N : ℕ) (h1 : N = 100) (h2 : (P : ℝ) / 100 * N = 50 / 100 * 40 + 10) :
  P = 30 :=
by
  sorry

end find_percentage_l18_18167


namespace school_students_count_l18_18380

theorem school_students_count (S : ℕ) : 
  (S / 2 * 2 = S) ∧ 
  (0.20 * (S / 2) + 0.10 * (S / 2) = 15) → 
  S = 100 :=
begin
  sorry
end

end school_students_count_l18_18380


namespace line_perpendicular_exists_k_line_intersects_circle_l18_18853

theorem line_perpendicular_exists_k (k : ℝ) :
  ∃ k, (k * (1 / 2)) = -1 :=
sorry

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (k * x - y + 2 * k = 0) ∧ (x^2 + y^2 = 8) :=
sorry

end line_perpendicular_exists_k_line_intersects_circle_l18_18853


namespace max_students_can_participate_l18_18782

theorem max_students_can_participate (max_funds rent cost_per_student : ℕ) (h_max_funds : max_funds = 800) (h_rent : rent = 300) (h_cost_per_student : cost_per_student = 15) :
  ∃ x : ℕ, x ≤ (max_funds - rent) / cost_per_student ∧ x = 33 :=
by
  sorry

end max_students_can_participate_l18_18782


namespace parametric_curve_expression_l18_18698

theorem parametric_curve_expression :
  ∀ (t : ℝ), ∃ (a b c : ℝ), 
    (x y : ℝ) (h : x = 3 * Real.cos t - 2 * Real.sin t ∧ y = 5 * Real.sin t), 
    a * x ^ 2 + b * x * y + c * y ^ 2 = 1 ∧ 
    a = (1 / 9) ∧ b = (4 / 45) ∧ c = (13 / 225) :=
by
  sorry

end parametric_curve_expression_l18_18698


namespace find_f_f_1_l18_18139

def f (x : ℝ) : ℝ :=
  if x > 2 then x + 1 / (x - 2)
  else x^2 + 2

theorem find_f_f_1 : f (f 1) = 4 := by
  sorry

end find_f_f_1_l18_18139


namespace cos_A_of_triangle_l18_18523

theorem cos_A_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : b = Real.sqrt 2 * c)
  (h2 : Real.sin A + Real.sqrt 2 * Real.sin C = 2 * Real.sin B)
  (h3 : a = Real.sin A / Real.sin A * b) -- Sine rule used implicitly

: Real.cos A = Real.sqrt 2 / 4 := by
  -- proof will be skipped, hence 'sorry' included
  sorry

end cos_A_of_triangle_l18_18523


namespace triangle_geometry_RS_l18_18567

theorem triangle_geometry_RS
    (CH AH BH : ℝ)
    (AB AC BC : ℕ)
    (h1 : AB = 2000)
    (h2 : AC = 1997)
    (h3 : BC = 1998)
    (H_CH : ∃ H: ℝ, CH = √(H^2 + 1997^2) ∧ sqrt((2000 - H)^2 + H^2) = 1998)
    (H_AH_BH :
        AH = 4000001 / 4000 ∧
        BH = 4000 - AH)
    (RS := abs ((AH - BH) / H)):
    ∃ m n : ℕ, m / n = RS ∧ nat.gcd m n = 1 :=
by
  sorry

end triangle_geometry_RS_l18_18567


namespace best_purchase_option_l18_18279

theorem best_purchase_option 
  (blend_techn_city : ℕ := 2000)
  (meat_techn_city : ℕ := 4000)
  (discount_techn_city : ℤ := 10)
  (blend_techn_market : ℕ := 1500)
  (meat_techn_market : ℕ := 4800)
  (bonus_techn_market : ℤ := 20) : 
  0.9 * (blend_techn_city + meat_techn_city) < (blend_techn_market + meat_techn_market) :=
by
  sorry

end best_purchase_option_l18_18279


namespace trajectory_eq_min_area_eq_and_line_eq_l18_18813

-- Conditions
def P : Type := (ℝ × ℝ)
def A : P := (-3, 0)
def B : P := (3, 0)
def dist (P1 P2 : P) : ℝ := Real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

-- Theorem to prove the equation of trajectory Γ
theorem trajectory_eq (P : P) (h1 : dist P A / dist P B = 2) : (P.1 - 5)^2 + P.2^2 = 16 :=
    sorry

-- Additional definition for point Q
def Q : P := (2, 3)

-- Theorem to prove the minimum area of quadrilateral OMCN and the equation of line MN
theorem min_area_eq_and_line_eq (P : P) (Q : P) (h1: dist P A / dist P B = 2) (h2: Q.2 - Q.1 = 1) : 
  (4 * Real.sqrt 2) ∧ (3 * P.2 - 2 * P.1 - 1 = 0) :=
    sorry

end trajectory_eq_min_area_eq_and_line_eq_l18_18813


namespace ap_number_of_terms_l18_18180

theorem ap_number_of_terms (a d : ℕ) (n : ℕ) (ha1 : (n - 1) * d = 12) (ha2 : a + 2 * d = 6)
  (h_odd_sum : (n / 2) * (2 * a + (n - 2) * d) = 36) (h_even_sum : (n / 2) * (2 * a + n * d) = 42) :
    n = 12 :=
by
  sorry

end ap_number_of_terms_l18_18180


namespace pentagon_percentage_l18_18177

theorem pentagon_percentage (l w : ℕ) (total_area pentagon_area : ℕ) (h1 : l = 3) (h2 : w = 3) (h3 : total_area = l * w) (h4 : pentagon_area = 2) :
  pentagon_area * 100 / total_area = 22.2 :=
by
  -- Sorry, as we are skipping the proof
  sorry

end pentagon_percentage_l18_18177


namespace total_turnips_l18_18938

theorem total_turnips (k_tpd : ℕ) (k_days : ℕ) (a_tptd : ℕ) (a_days : ℕ) 
  (h₀ : k_tpd = 6) (h₁ : k_days = 7) (h₂ : a_tptd = 9) (h₃ : a_days = 14) : 
  k_tpd * k_days + a_tptd * (a_days / 2) = 105 :=
by
  rw [h₀, h₁, h₂, h₃]
  norm_num
  sorry

end total_turnips_l18_18938


namespace least_non_lucky_multiple_of_24_l18_18702

def sum_of_digits (n : Nat) : Nat :=
  n.toString.foldl (λ (acc : Nat) (c : Char) => acc + (c.toNat - '0'.toNat)) 0

def is_lucky (n : Nat) : Prop :=
  n % sum_of_digits n = 0

def is_multiple_of_24 (n : Nat) : Prop :=
  n % 24 = 0

theorem least_non_lucky_multiple_of_24 : ∃ (n : Nat), is_multiple_of_24 n ∧ ¬ is_lucky n ∧ ∀ (m : Nat), is_multiple_of_24 m → ¬ is_lucky m → n ≤ m :=
by
  use 120
  sorry

end least_non_lucky_multiple_of_24_l18_18702


namespace find_positive_value_of_A_l18_18210

variable (A : ℝ)

-- Given conditions
def relation (A B : ℝ) : ℝ := A^2 + B^2

-- The proof statement
theorem find_positive_value_of_A (h : relation A 7 = 200) : A = Real.sqrt 151 := sorry

end find_positive_value_of_A_l18_18210


namespace angle_AHB_l18_18732

-- Define the problem statement
theorem angle_AHB (A B C D E H : Point) (h1 : altitude A D B C)
                               (h2 : altitude B E A C)
                               (h3 : intersection A D B E H)
                               (h4 : ∠ BAC = 58)
                               (h5 : ∠ ABC = 67)
                               : ∠ AHB = 125 :=
begin
  sorry
end

end angle_AHB_l18_18732


namespace symmetric_point_and_line_l18_18820

theorem symmetric_point_and_line (
  A : Point, l1 l2 : Line)
  (hA : A.x = 0 ∧ A.y = 1)
  (hl1 : l1.equation = λ (x y : ℝ), x - y - 1)
  (hl2 : l2.equation = λ (x y : ℝ), x - 2y + 2)
  (hB : symmetric_point A l1 = (2, -1))
  (hl_symmetric : symmetric_line l2 l1.equation = λ (x y : ℝ), 2 * x - y - 5)
  : 
      (∃ B : Point, B.x = 2 ∧ B.y = -1)
      ∧ 
      (∃ l_symmetric : Line, l_symmetric.equation = λ (x y : ℝ), 2 * x - y - 5) :=
by 
  sorry

end symmetric_point_and_line_l18_18820


namespace locus_of_centroids_is_a_line_segment_l18_18314

variables {A B C P : Type} [AffineSpace ℝ A]
variables (M1 M2 N1 N2 G : A → A) (AB BC CA : AffineSubspace ℝ A)

-- Conditions based on the problem statements:
def is_median (A B C M1 M2 : A) : Prop :=
  midpoint_virtual ℕ (B -ᵥ C) (C -ᵥ B) = (M1 -ᵥ A) / (M2 -ᵥ B) ∧
  midpoint_virtual ℕ (A -ᵥ C) (C -ᵥ A) = (M2 -ᵥ B) / (M1 -ᵥ A)

axiom parallel_to_median_AM1_BM2 (P M1 M2 N1 N2 : A) :
  parallel (line_through P M1) (line_through A M1) ∧
  parallel (line_through P M2) (line_through B M2)

-- Main statement of the theorem in Lean
theorem locus_of_centroids_is_a_line_segment
  (ABC_P : P ∈ segment ℝ (A, B))
  (N1_on_BC : N1 ∈ BC)
  (N2_on_CA : N2 ∈ CA)
  (parallel_AM1_BM2 : ∀ (P : A), parallel_to_median_AM1_BM2 P M1 M2 N1 N2)
    : ∃ line_through_AC : AffineSubspace ℝ A,
      ∀ (P : A), P ∈ (segment ℝ (line_through A B)) →
                 centroid {(P, N1 P, N2 P)} ∈ line_through_AC :=
sorry

end locus_of_centroids_is_a_line_segment_l18_18314


namespace max_vector_value_l18_18826

-- Given definitions and conditions

def ellipse (a : ℝ) : set (ℝ × ℝ) :=
  {p | let (x, y) := p in (x^2 / a^2) + (y^2 / (a^2 - 1)) = 1}

def is_focus (a : ℝ) (p : ℝ × ℝ) : Prop :=
  p = (-√(a^2 - (a^2 - 1)), 0) ∨ p = (√(a^2 - (a^2 - 1)), 0)

def is_on_ellipse (a : ℝ) (p : ℝ × ℝ) : Prop :=
  p ∈ ellipse a

def vector_length (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

def origin : ℝ × ℝ := (0, 0)

def max_value (a : ℝ) := 2 * a

-- Main theorem
theorem max_vector_value (a : ℝ) (h_a : 1 < a) (P Q : ℝ × ℝ)
  (hP : is_on_ellipse a P) (hQ : is_on_ellipse a Q)
  (F1 F2 : ℝ × ℝ) (hF1 : is_focus a F1) (hF2 : is_focus a F2) :
  ∃ M : ℝ, (∀ (P Q : ℝ × ℝ), 
    is_on_ellipse a P → is_on_ellipse a Q →
    let vec_sum := (vector_sub (vector_sub (P.1, P.2) F1) 
                               (vector_sub (P.1, P.2) F2))
        vec_2PQ := vector_sub P Q in
    vector_length (vector_sub vec_sum (vec_2PQ)) ≤ M) ∧ M = max_value a :=
begin
  sorry
end

end max_vector_value_l18_18826


namespace angle_QTR_cyclic_quadrilateral_l18_18175

-- Definitions for cyclic quadrilateral and given angles
variable (P Q R S T : Type) [CyclicQuadrilateral P Q R S]
variable (A B : Angle Type)
variable (QTS : Angle Type)

-- Given conditions
theorem angle_QTR_cyclic_quadrilateral 
  (h1 : ∠ PRS = 74) 
  (h2 : ∠ PSQ = 106) : 
  ∠ QTR = 74 :=
by
  sorry

end angle_QTR_cyclic_quadrilateral_l18_18175


namespace v3_value_at_2_l18_18052

def f (x : ℝ) : ℝ :=
  x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

def v3 (x : ℝ) : ℝ :=
  ((x - 12) * x + 60) * x - 160

theorem v3_value_at_2 :
  v3 2 = -80 :=
by
  sorry

end v3_value_at_2_l18_18052


namespace volume_pyramid_eq_750_l18_18957

-- Given conditions of the problem
variable {V H P A B C D : Type}
variable [HasDist V H P 3 5]
variables {VH : ℝ}
variables {VP : ℝ} {H_mid : V -> H}
variables (is_midpoint_VH: SorryProp $ H_mid P = midpoint V H)
variables (is_distance_to_face: SorryProp $ dist P (face_point V B C) = 3)
variables (is_distance_to_base: SorryProp $ dist P (base_point A B C D) = 5)
variables (VH_val: VH = 10) (VP_val: VP = 5)
variables (face_point V B C : point_on_face V B C) (base_point A B C D: point_on_base A B C D)

-- To prove: The volume of the pyramid is 750
theorem volume_pyramid_eq_750 : volume (pyramid V A B C D) = 750 :=
by
  sorry

end volume_pyramid_eq_750_l18_18957


namespace solve_diff_eq_system_l18_18680

noncomputable def solve_system (C₁ C₂ C₃ : ℝ) : ℝ → ℝ × ℝ × ℝ := 
  λ t => 
    let x1 := Real.log (abs (C₁ * t + C₂)) + C₁ + C₃;
    let x2 := Real.log (abs (C₁ * t + C₂)) + C₃;
    let x3 := (C₁ + 1) * t + C₂;
    (x1, x2, x3)

theorem solve_diff_eq_system :
  ∀ (C₁ C₂ C₃ : ℝ) (t : ℝ), 
    let (x1, x2, x3) := solve_system C₁ C₂ C₃ t;
    (∃ x1 x2 x3 t, 
      (differentiable ℝ (λ t => x1 t) ∧ differentiable ℝ (λ t => x2 t) ∧ differentiable ℝ (λ t => x3 t)) ∧ 
      (deriv (λ t => x1) t = (x1 t - x2 t) / (x3 t - t) ∧ 
      deriv (λ t => x2) t = (x1 t - x2 t) / (x3 t - t) ∧ 
      deriv (λ t => x3) t = x1 t - x2 t + 1)) :=
by sorry

end solve_diff_eq_system_l18_18680


namespace second_shift_production_l18_18504

-- Question: Prove that the number of cars produced by the second shift is 1,100 given the conditions
-- Conditions:
-- 1. P_day = 4 * P_second
-- 2. P_day + P_second = 5,500

theorem second_shift_production (P_day P_second : ℕ) (h1 : P_day = 4 * P_second) (h2 : P_day + P_second = 5500) :
  P_second = 1100 := by
  sorry

end second_shift_production_l18_18504


namespace percentage_of_adult_men_l18_18994

theorem percentage_of_adult_men (total_members : ℕ) (children : ℕ) (p : ℕ) :
  total_members = 2000 → children = 200 → 
  (∀ adult_men_percentage : ℕ, adult_women_percentage = 2 * adult_men_percentage) → 
  (100 - p) = 3 * (p - 10) →  p = 30 :=
by sorry

end percentage_of_adult_men_l18_18994


namespace sum_of_rational_roots_of_h_l18_18079

noncomputable def h (x : ℚ) : ℚ := x^3 - 6 * x^2 + 11 * x - 6

theorem sum_of_rational_roots_of_h :
  let roots := {r : ℚ | h r = 0} in
  ∑ r in roots.to_finset, r = 6 :=
sorry

end sum_of_rational_roots_of_h_l18_18079


namespace handrail_length_nearest_tenth_l18_18726

noncomputable def handrail_length (rise : ℝ) (turn_degree : ℝ) (radius : ℝ) : ℝ :=
  let arc_length := (turn_degree / 360) * (2 * Real.pi * radius)
  Real.sqrt (rise^2 + arc_length^2)

theorem handrail_length_nearest_tenth
  (h_rise : rise = 12)
  (h_turn_degree : turn_degree = 180)
  (h_radius : radius = 3) : handrail_length rise turn_degree radius = 13.1 :=
  by
  sorry

end handrail_length_nearest_tenth_l18_18726


namespace max_sqrt_sum_leq_six_l18_18451

-- Definitions from conditions
variables (a b : ℝ)
def is_nonzero (x : ℝ) := x ≠ 0
def collinear (m n : ℝ × ℝ) := ∃ k : ℝ, m = k • n

-- Given conditions
axiom nonzero_a : is_nonzero a
axiom nonzero_b : is_nonzero b
axiom collinear_mn : collinear (2, 12 - 2 * a) (1, 2 * b)

-- To prove the maximum value statement
theorem max_sqrt_sum_leq_six : 
  sqrt (2 * a + b) + sqrt (a + 5 * b) ≤ 6 := sorry

end max_sqrt_sum_leq_six_l18_18451


namespace largest_side_of_triangle_l18_18302

theorem largest_side_of_triangle (x y Δ c : ℕ)
  (h1 : (x + 2 * Δ / x = y + 2 * Δ / y))
  (h2 : x = 60)
  (h3 : y = 63) :
  c = 87 :=
sorry

end largest_side_of_triangle_l18_18302


namespace relationship_between_a_and_b_l18_18218

noncomputable section
open Classical

theorem relationship_between_a_and_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b :=
sorry

end relationship_between_a_and_b_l18_18218


namespace f_eq_zero_of_le_zero_l18_18574

variable {R : Type*} [LinearOrderedField R]
variable {f : R → R}
variable (cond : ∀ x y : R, f (x + y) ≤ y * f x + f (f x))

theorem f_eq_zero_of_le_zero (x : R) (h : x ≤ 0) : f x = 0 :=
sorry

end f_eq_zero_of_le_zero_l18_18574


namespace total_salmon_trip_l18_18085

theorem total_salmon_trip (male_salmon : ℕ) (female_salmon : ℕ) (h_male : male_salmon = 712261) (h_female : female_salmon = 259378) : male_salmon + female_salmon = 971639 :=
by 
  rw [h_male, h_female]
  exact rfl

end total_salmon_trip_l18_18085


namespace a7_not_prime_l18_18071

def reverse_digits (n : ℕ) : ℕ :=
-- Implementation of digit reversal is needed, skipped here
sorry

def sequence_a : ℕ → ℕ
| 1 := a₁
| (n + 1) := sequence_a n + reverse_digits (sequence_a n)

theorem a7_not_prime (a₁ : ℕ) (h_pos : a₁ > 0) : ¬ prime (sequence_a 7) :=
sorry

end a7_not_prime_l18_18071


namespace max_expression_value_l18_18576

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem max_expression_value (p : Fin 97 → ℕ) (h : ∀ i, is_prime (p i)) : 
  (∑ i, f (p i)) ≤ 38 :=
sorry

end max_expression_value_l18_18576


namespace derek_initial_lunch_cost_l18_18772

-- Definitions based on conditions
def derek_initial_money : ℕ := 40
def derek_dad_lunch_cost : ℕ := 11
def derek_more_lunch_cost : ℕ := 5
def dave_initial_money : ℕ := 50
def dave_mom_lunch_cost : ℕ := 7
def dave_difference : ℕ := 33

-- Variable X to represent Derek's initial lunch cost
variable (X : ℕ)

-- Definitions based on conditions
def derek_total_spending (X : ℕ) := X + derek_dad_lunch_cost + derek_more_lunch_cost
def derek_remaining_money (X : ℕ) := derek_initial_money - derek_total_spending X
def dave_remaining_money := dave_initial_money - dave_mom_lunch_cost

-- The main theorem to prove Derek spent $14 initially
theorem derek_initial_lunch_cost (h : dave_remaining_money = derek_remaining_money X + dave_difference) : X = 14 := by
  sorry

end derek_initial_lunch_cost_l18_18772


namespace fresh_permutation_inequality_l18_18027

-- Definitions used in the problem
def is_fresh_permutation (σ : List ℕ) (m : ℕ) : Prop :=
  ∀ (k : ℕ), (0 < k) → (k < m) → (σ.take k ≠ (List.range k).map (λ i, i + 1))

def f (m : ℕ) : ℕ :=
  Nat.card {σ : List ℕ // σ.Permutation (List.range m).map (λ i, i + 1) ∧ is_fresh_permutation σ m}

-- Problem statement as a theorem in Lean 4
theorem fresh_permutation_inequality (n : ℕ) (hn : n ≥ 3) : f n ≥ n * f (n - 1) := sorry

end fresh_permutation_inequality_l18_18027


namespace common_difference_is_3_l18_18136

variable {a : ℕ → ℤ} {d : ℤ}

-- Definitions of conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition_1 (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 3 + a 11 = 24

def condition_2 (a : ℕ → ℤ) : Prop :=
  a 4 = 3

-- Theorem statement to prove
theorem common_difference_is_3 (a : ℕ → ℤ) (d : ℤ)
  (ha : is_arithmetic_sequence a d)
  (hc1 : condition_1 a d)
  (hc2 : condition_2 a) :
  d = 3 := by
  sorry

end common_difference_is_3_l18_18136


namespace prove_circle_and_m_values_l18_18123

noncomputable def find_circle_parameters (a r : ℝ) (h_r : 0 < r) :=
  (∀ x y : ℝ, (x - a)^2 + (y - 4)^2 = r^2 → ((x = 0 ∧ y = 0) ∨ (x = 6 ∧ y = 0)))

noncomputable def find_m_values (m : ℝ) :=
  ∃ a r : ℝ, (a = 3 ∧ r = 5 ∧ (∀ m : ℝ, 4 * x + 3 * y + m = 0 ∧ 6 ∧ 
  abs (24 + m) / 5 = sqrt (5^2 - 3^2)))

theorem prove_circle_and_m_values :
  ∃ a r : ℝ, a = 3 ∧ r = 5 ∧
  (4 * x + 3 * y + (-4) = 0 ∨ 4 * x + 3 * y + (-44) = 0) :=
begin
  sorry,
end

end prove_circle_and_m_values_l18_18123


namespace percentage_of_knives_is_40_l18_18063

theorem percentage_of_knives_is_40 
  (initial_knives : ℕ) (initial_forks : ℕ) (initial_spoons : ℕ) 
  (traded_knives : ℕ) (traded_spoons : ℕ) : 
  initial_knives = 6 → 
  initial_forks = 12 → 
  initial_spoons = 3 * initial_knives → 
  traded_knives = 10 → 
  traded_spoons = 6 → 
  let final_knives := initial_knives + traded_knives in
  let final_spoons := initial_spoons - traded_spoons in
  let total_silverware := final_knives + final_spoons + initial_forks in
  (final_knives : ℝ) / total_silverware * 100 = 40 :=
by sorry

end percentage_of_knives_is_40_l18_18063


namespace sphere_volume_increase_by_eight_l18_18520

theorem sphere_volume_increase_by_eight (r : ℝ) :
  let original_volume := (4 / 3) * Real.pi * r^3,
      new_volume := (4 / 3) * Real.pi * (2 * r)^3
  in new_volume = 8 * original_volume :=
by
  let original_volume := (4 / 3) * Real.pi * r^3
  let new_volume := (4 / 3) * Real.pi * (2 * r)^3
  sorry

end sphere_volume_increase_by_eight_l18_18520


namespace isosceles_triangle_base_division_l18_18533

theorem isosceles_triangle_base_division :
  ∀ (A B C : Type) [metric_space A] [metric_space B] [metric_space C], 
  ∀ (midpoint K : Type) [metric_space K],
  ∀ (perpendicular M : Type) [metric_space M],
  isosceles_triangle A B C →
  ∃ (AC BC AB : ℝ), AC = 32 ∧ BC = 20 ∧ AB = 20 →
  ∀ (K : AC/2),
  ∀ (BK : 12),
  ∀ (MK : 12),
  divides_base(AC, 7, 25).

end isosceles_triangle_base_division_l18_18533


namespace integer_power_sums_l18_18960

theorem integer_power_sums (x : ℝ) (h : x + (1 / x) ∈ ℤ) (n : ℕ) : 
  x^n + (1 / x^n) ∈ ℤ := 
sorry

end integer_power_sums_l18_18960


namespace max_value_Sn_over_an_l18_18762

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α) (S : ℕ → α)

-- Definitions derived from conditions
def Sn (n : ℕ) : α := (n : α) / 2 * (a 1 + a n)
axiom Sn_17_pos : Sn a 17 > 0
axiom Sn_18_neg : Sn a 18 < 0

-- The target theorem
theorem max_value_Sn_over_an :
  max (list.map (λ n, Sn a n / a n) (list.range' 1 15)) = Sn a 9 / a 9 :=
sorry

end max_value_Sn_over_an_l18_18762


namespace words_per_hour_l18_18390

theorem words_per_hour (total_words : ℕ) (total_hours : ℕ) (h_words : total_words = 75000) (h_hours : total_hours = 150) :
  total_words / total_hours = 500 := by
  rw [h_words, h_hours]
  norm_num
  sorry

end words_per_hour_l18_18390


namespace power_sum_int_l18_18962

theorem power_sum_int {x : ℝ} (hx : ∃ k : ℤ, x + 1/x = k) : ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by
  sorry

end power_sum_int_l18_18962


namespace f_of_3_is_log2_3_l18_18488

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_condition : ∀ x : ℝ, f (2 ^ x) = x

theorem f_of_3_is_log2_3 : f 3 = Real.log 3 / Real.log 2 := sorry

end f_of_3_is_log2_3_l18_18488


namespace handrail_length_nearest_tenth_l18_18727

noncomputable def handrail_length (rise : ℝ) (turn_degree : ℝ) (radius : ℝ) : ℝ :=
  let arc_length := (turn_degree / 360) * (2 * Real.pi * radius)
  Real.sqrt (rise^2 + arc_length^2)

theorem handrail_length_nearest_tenth
  (h_rise : rise = 12)
  (h_turn_degree : turn_degree = 180)
  (h_radius : radius = 3) : handrail_length rise turn_degree radius = 13.1 :=
  by
  sorry

end handrail_length_nearest_tenth_l18_18727


namespace part_I_solution_part_II_solution_l18_18145

noncomputable def f (x m : ℝ) : ℝ := |x - m| + |x + 6|

-- Part (I)
theorem part_I_solution (x : ℝ) : 
  {x | -13/2 ≤ x ∧ x ≤ 11/2} = {x | f x 5 ≤ 12} :=
by
  sorry

-- Part (II)
theorem part_II_solution (m : ℝ) : 
  (∀ x, f x m ≥ 7) ↔ m ∈ (-∞, -13] ∪ [1, ∞) :=
by
  sorry

end part_I_solution_part_II_solution_l18_18145


namespace candidate_vote_percentage_l18_18367

noncomputable def candidate_lost_votes : ℕ := 833
noncomputable def total_votes : ℕ := 2450

theorem candidate_vote_percentage :
  ∃ P : ℝ, (P / 100) * total_votes + candidate_lost_votes = ((100 - P) / 100) * total_votes ∧ P ≈ 33 :=
by
  sorry

end candidate_vote_percentage_l18_18367


namespace extinction_prob_l18_18713

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the base case condition
def v_0 : ℝ := 1

-- Define the extinction probability function
def v : ℕ → ℝ
| 0 => v_0
| k => p * v (k + 1) + q * v (k - 1)

-- Define the target value for v_1
def v_1 : ℝ := 2 / 3

-- The main theorem: Prove that v 1 = 2 / 3 given the conditions
theorem extinction_prob : v 1 = v_1 := by
  -- Proof will be provided here
  sorry

end extinction_prob_l18_18713


namespace part1_part2_l18_18811

-- Definition of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 2 * y + 1 = 0

-- Definition of the line
def line_eq (k x y : ℝ) : Prop := y = k * x

-- Definition of points A and B on the circle and line
def points_A_B (k x1 y1 x2 y2 : ℝ) : Prop := 
  circle_eq x1 y1 ∧ line_eq k x1 y1 ∧
  circle_eq x2 y2 ∧ line_eq k x2 y2

-- Condition of perpendicular vectors
def perp_vectors_MA_MB (x1 y1 x2 y2 b : ℝ) : Prop := 
  (x1 * x2) + ((y1 - b) * (y2 - b)) = 0

-- Problem 1: Prove k = 1 when b = 1
theorem part1 (k : ℝ) (b : ℝ) (h : b = 1) : 
  ∀ x1 y1 x2 y2, 
    points_A_B k x1 y1 x2 y2 → 
    perp_vectors_MA_MB x1 y1 x2 y2 b → 
    k = 1 :=
by
  sorry

-- Problem 2: Prove the range of k when b ∈ (1, 3/2)
theorem part2 (k b : ℝ) (hb : 1 < b ∧ b < 3 / 2) : 
  ∀ x1 y1 x2 y2, 
    points_A_B k x1 y1 x2 y2 → 
    perp_vectors_MA_MB x1 y1 x2 y2 b → 
    1 < k ∧ k < 6 - sqrt 23 ∨ 
    6 + sqrt 23 < k :=
by
  sorry

end part1_part2_l18_18811


namespace find_sin_theta_l18_18215

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (theta : ℝ)

noncomputable def sin_theta_condition := ∥a∥ = 2 ∧ ∥b∥ = 7 ∧ ∥c∥ = 6 ∧ a × (a × b) = c

theorem find_sin_theta (h : sin_theta_condition a b c) : real.sin theta = 3 / 14 :=
begin
  sorry
end

end find_sin_theta_l18_18215


namespace average_writing_speed_l18_18704

theorem average_writing_speed (total_words : ℕ) (total_hours : ℕ) (h_words: total_words = 60000) (h_hours: total_hours = 120) : 
  total_words / total_hours = 500 := 
by 
  rw [h_words, h_hours]
  norm_num
  sorry

end average_writing_speed_l18_18704


namespace power_difference_l18_18891

noncomputable theory

variables {a m n : ℝ}

theorem power_difference (h1 : a^m = 12) (h2 : a^n = 3) : a^(m - n) = 4 :=
by sorry

end power_difference_l18_18891


namespace x_power6_y_power6_l18_18202

theorem x_power6_y_power6 (x y a b : ℝ) (h1 : x + y = a) (h2 : x * y = b) :
  x^6 + y^6 = a^6 - 6 * a^4 * b + 9 * a^2 * b^2 - 2 * b^3 :=
sorry

end x_power6_y_power6_l18_18202


namespace find_n_l18_18092

def has_sixteen_divisors (n : ℕ) : Prop :=
  ∃ (d : fin 16 → ℕ), d 0 = 1 ∧ d 5 = 18 ∧ (d 8 - d 7 = 17) ∧ d 15 = n ∧ strict_mono d.val ∧ ∀ i, i < 15 → n % d i = 0

theorem find_n (n : ℕ) (h : has_sixteen_divisors n) : n = 1998 ∨ n = 3834 :=
sorry

end find_n_l18_18092


namespace smallest_N_exists_l18_18178

def find_smallest_N (N : ℕ) : Prop :=
  ∃ (c1 c2 c3 c4 c5 c6 : ℕ),
  (N ≠ 0) ∧ 
  (c1 = 6 * c2 - 1) ∧ 
  (N + c2 = 6 * c3 - 2) ∧ 
  (2 * N + c3 = 6 * c4 - 3) ∧ 
  (3 * N + c4 = 6 * c5 - 4) ∧ 
  (4 * N + c5 = 6 * c6 - 5) ∧ 
  (5 * N + c6 = 6 * c1)

theorem smallest_N_exists : ∃ (N : ℕ), find_smallest_N N :=
sorry

end smallest_N_exists_l18_18178


namespace simplify_and_evaluate_l18_18260

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sin (Real.pi / 6)) :
  (1 - 2 / (x - 1)) / ((x - 3) / (x^2 - 1)) = 3 / 2 :=
by
  -- simplify and evaluate the expression given the condition on x
  sorry

end simplify_and_evaluate_l18_18260


namespace concyclic_points_l18_18124

variables {A B C D D' E E' F F' : Type}
variables [Point : Type] [triangle : ABC → Type] 
variables [circle₁: D → D' → E → E' → Type] [circle₂: E → E' → F → F' → Type] [circle₃: F → F' → D → D' → Type]
variables [triangle_point {Point} {D} {D'} {E} {E'} {F} {F'}]

theorem concyclic_points
  (condition_1 : D ∈ triangle)
  (condition_2 : D' ∈ triangle)
  (condition_3 : E ∈ triangle)
  (condition_4 : E' ∈ triangle)
  (condition_5 : F ∈ triangle)
  (condition_6 : F' ∈ triangle)
  (condition_7 : circle₁ D D' E E')
  (condition_8 : circle₂ E E' F F')
  (condition_9 : circle₃ F F' D D') :
  ∃ (circle₄: D → D' → E → E' → F → F' → Type), circle₄ D D' E E' F F' := 
by sorry

end concyclic_points_l18_18124


namespace parallel_transitivity_l18_18657

theorem parallel_transitivity (a b c : Type) [HasParallel a b] [HasParallel b c] [HasParallel a c]
  (h1 : is_parallel a c) (h2 : is_parallel b c) : is_parallel a b :=
begin
  sorry,
end

end parallel_transitivity_l18_18657


namespace true_propositions_count_l18_18285

def quadrilateral_diagonal_equal_and_bisect (Q : Type) : Prop := -- Proposition 1
  ∃ (d1 d2 : Q), (equal_diagonals d1 d2) ∧ (bisect_each_other d1 d2)

def quadrilateral_perpendicular_diagonals (Q : Type) : Prop := -- Proposition 2
  ∃ (d1 d2 : Q), (perpendicular_diagonals d1 d2)

def quadrilateral_perpendicular_and_equal_diagonals (Q : Type) : Prop := -- Proposition 3
  ∃ (d1 d2 : Q), (perpendicular_diagonals d1 d2) ∧ (equal_diagonals d1 d2)

def quadrilateral_equal_sides (Q : Type) : Prop := -- Proposition 4
  ∃ (s1 s2 s3 s4 : Q), (equal_sides s1 s2 s3 s4)

theorem true_propositions_count : 
  (num_true_propositions = 3) ↔
  (quadrilateral_diagonal_equal_and_bisect Q ∧
   ¬quadrilateral_perpendicular_diagonals Q ∧
   quadrilateral_perpendicular_and_equal_diagonals Q ∧
   quadrilateral_equal_sides Q) :=
sorry

end true_propositions_count_l18_18285


namespace sum_of_exponents_l18_18090

theorem sum_of_exponents (n : ℕ) (h : n = 2023) : 
  ∃ (exponents : List ℕ), 
    (∀ (e : ℕ), e ∈ exponents →  2^e ∈ (List.map (λ e, 2^e) exponents) ∧ n = exponents.foldr (+) 0) 
    ∧ n = 2023
    ∧ exponents.sum = 48 := by
  sorry

end sum_of_exponents_l18_18090


namespace slope_line_MN_eq_1_equation_line_MN_l18_18130

-- Given Conditions
variables (M N : ℝ × ℝ) (B : ℝ × ℝ := (-2, 0)) (O : ℝ × ℝ := (0, 0))
def parabola (P : ℝ × ℝ) := P.snd^2 = 4 * P.fst
def sum_of_y_coords (P Q : ℝ × ℝ) := P.snd + Q.snd = 4
def angle_OBM_eq_OBN (M N B O : ℝ × ℝ) := ∠ (B, O, M) = ∠ (B, O, N)

-- Prove the slope of line MN is 1
theorem slope_line_MN_eq_1
  (hM : parabola M)
  (hN : parabola N)
  (h_sum_y : sum_of_y_coords M N)
  (h_angles : angle_OBM_eq_OBN M N B O) :
  slope(M N) = 1 :=
sorry

-- Prove the equation of the line MN is y = x - 2
theorem equation_line_MN
  (hM : parabola M)
  (hN : parabola N)
  (h_sum_y : sum_of_y_coords M N)
  (h_angles : angle_OBM_eq_OBN M N B O) :
  equation(M N) = fun y x => y = x - 2 :=
sorry

end slope_line_MN_eq_1_equation_line_MN_l18_18130


namespace log_sum_geometric_sequence_l18_18187

noncomputable theory

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, n ≠ m → a n = r * a (m + 1 - n)

def roots_of_quadratic_eq (a : ℕ → ℝ) : Prop :=
(a 3) * (a 10) = 1 / 2

theorem log_sum_geometric_sequence (a : ℕ → ℝ) (ha1 : is_geometric_sequence a)
  (ha2 : roots_of_quadratic_eq a) :
  (∑ n in finset.range 12, real.log 2 (a (n + 1))) = -6 :=
sorry

end log_sum_geometric_sequence_l18_18187


namespace simplify_expression_l18_18953

def real_numbers (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a^3 + b^3 = a^2 + b^2

theorem simplify_expression (a b : ℝ) (h : real_numbers a b) :
  (a^2 / b + b^2 / a - 1 / (a * a * b * b)) = (a^4 + 2 * a * b + b^4 - 1) / (a * b) :=
by
  sorry

end simplify_expression_l18_18953


namespace clothing_probability_l18_18345

theorem clothing_probability :
  let shirts := 3
  let pants := 6
  let socks := 9
  let total_clothing := shirts + pants + socks
  let total_ways := choose total_clothing 4
  let shirt_ways := choose shirts 1
  let pants_ways := choose pants 2
  let socks_ways := choose socks 1
  let favorable_ways := shirt_ways * pants_ways * socks_ways
  let probability := favorable_ways / total_ways
in
  probability = 15 / 114 :=
by
  sorry

end clothing_probability_l18_18345


namespace ellipse_equation_l18_18463

open Real

noncomputable def ellipse_C (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def circle_radius (b : ℝ) (x y : ℝ) := (x^2 + y^2 = b^2)

def line_eq (m k : ℝ) (x y : ℝ) : Prop := y = k*x + m

theorem ellipse_equation :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ ∃ (c : ℝ), c^2 = a^2 - b^2 ∧
  c = a * sqrt (3 / 4) ∧
  ellipse_C a b x y ∧
  circle_radius b x y ∧
  ∀ (m k : ℝ) (l : line_eq m k x y) (G H : ℝ × ℝ),
  area_triangle_origin G H = 1 ∧
  symmetric_about_xaxis G H x ∧
  (∃ (s : ℝ), symmetric_about_origin M N x s ∧
  ( product_slopes (P G H) M N = -1/4) :=
sorry

end ellipse_equation_l18_18463


namespace divisible_iff_l18_18851

-- Definitions from the conditions
def a : ℕ → ℕ
  | 0     => 0
  | 1     => 1
  | (n+2) => 2 * a (n + 1) + a n

-- Main theorem statement.
theorem divisible_iff (n k : ℕ) : 2^k ∣ a n ↔ 2^k ∣ n := by
  sorry

end divisible_iff_l18_18851


namespace lambda_equals_half_l18_18400

variables (a b : ℝ) (l : ℝ)

-- Assumptions
axiom not_parallel (a b : ℝ) : (a ≠ b)
axiom parallel_eq (a b : ℝ) (l : ℝ) : ∃ m : ℝ, l * a + b = m * (a + 2 * b)

theorem lambda_equals_half (a b λ : ℝ) (h₁ : not_parallel a b) (h₂ : parallel_eq a b λ) : λ = 1 / 2 :=
sorry

end lambda_equals_half_l18_18400


namespace trigonometric_inequality_solution_l18_18263

theorem trigonometric_inequality_solution (k : ℤ) :
  ∃ x : ℝ, x = - (3 * Real.pi) / 2 + 4 * Real.pi * k ∧
           (Real.cos (x / 2) + Real.sin (x / 2) ≤ (Real.sin x - 3) / Real.sqrt 2) :=
by
  sorry

end trigonometric_inequality_solution_l18_18263


namespace smallest_region_area_l18_18793

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

def abs_line_eq (x y : ℝ) : Prop := y = abs x

theorem smallest_region_area :
  ∃ r θ (Hc : ∀ x y, circle_eq x y) (Hl : ∀ x y, abs_line_eq x y),
  r = 2 ∧ θ = π / 2 ∧ (1/2) * r^2 * θ = π :=
by
  exists 2, (π / 2)
  intros x y
  intros x y
  split
  sorry

end smallest_region_area_l18_18793


namespace total_v_l18_18362

variables (P V Vx Vy Px Py : ℝ)

-- Given conditions
def coffee_p_total : ℝ := 24   -- Total lbs of coffee p
def coffee_p_in_x : ℝ := 20    -- Coffee x contains 20 lbs of p
def ratio_p_v_in_x : ℝ := 4    -- Ratio of p to v in coffee x is 4:1
def ratio_p_v_in_y : ℝ := 1/5  -- Ratio of p to v in coffee y is 1:5

-- Definitions derived from conditions
def Vx := (1 / ratio_p_v_in_x) * coffee_p_in_x  -- Amount of v in coffee x
def Py := coffee_p_total - coffee_p_in_x         -- Amount of p in coffee y
def Vy := (1 / ratio_p_v_in_y) * Py              -- Amount of v in coffee y

-- Statement to prove
theorem total_v : Vx + Vy = 25 := by
  sorry

end total_v_l18_18362


namespace discount_proof_l18_18729

-- Define the original price and the discounts
variable (original_price : ℝ)

-- Define the conditions:
def first_discounted_price : ℝ := 0.60 * original_price
def second_discounted_price : ℝ := 0.90 * first_discounted_price

-- Claimed discount
def claimed_discount : ℝ := 0.55 * original_price

-- Actual discount computed after both discounts are applied
def actual_discount : ℝ := original_price - second_discounted_price

-- Prove the actual discount and the difference between claimed and actual discount
theorem discount_proof :
  actual_discount = 0.46 * original_price ∧ 
  claimed_discount - actual_discount = 0.09 * original_price :=
by
  sorry

end discount_proof_l18_18729


namespace molecular_weight_of_7_moles_boric_acid_l18_18661

-- Define the given constants.
def atomic_weight_H : ℝ := 1.008
def atomic_weight_B : ℝ := 10.81
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula for boric acid.
def molecular_weight_H3BO3 : ℝ :=
  3 * atomic_weight_H + 1 * atomic_weight_B + 3 * atomic_weight_O

-- Define the number of moles.
def moles_boric_acid : ℝ := 7

-- Calculate the total weight for 7 moles of boric acid.
def total_weight_boric_acid : ℝ :=
  moles_boric_acid * molecular_weight_H3BO3

-- The target statement to prove.
theorem molecular_weight_of_7_moles_boric_acid :
  total_weight_boric_acid = 432.838 := by
  sorry

end molecular_weight_of_7_moles_boric_acid_l18_18661


namespace compound_interest_years_l18_18794

-- Definitions for the given conditions
def principal : ℝ := 1200
def rate : ℝ := 0.20
def compound_interest : ℝ := 873.60
def compounded_yearly : ℝ := 1

-- Calculate the future value from principal and compound interest
def future_value : ℝ := principal + compound_interest

-- Statement of the problem: Prove that the number of years t was 3 given the conditions
theorem compound_interest_years :
  ∃ (t : ℝ), future_value = principal * (1 + rate / compounded_yearly)^(compounded_yearly * t) := sorry

end compound_interest_years_l18_18794


namespace hyperbola_problem_l18_18700

theorem hyperbola_problem (s : ℝ) :
    (∃ b > 0, ∀ (x y : ℝ), (x, y) = (-4, 5) → (x^2 / 9) - (y^2 / b^2) = 1)
    ∧ (∀ (x y : ℝ), (x, y) = (-3, 0) → (x^2 / 9) - (y^2 / b^2) = 1)
    ∧ (∀ b > 0, (x, y) = (s, 3) → (x^2 / 9) - (7 * y^2 / 225) = 1)
    → s^2 = (288 / 25) :=
by
  sorry

end hyperbola_problem_l18_18700


namespace min_a_ln_ineq_l18_18886

theorem min_a_ln_ineq (a : ℝ) : (∀ x : ℝ, (0 < x) ∧ (ln (2 * x) - (a * exp x) / 2 ≤ ln a)) → (a ≥ 2 / exp 1) :=
sorry

end min_a_ln_ineq_l18_18886


namespace find_general_formula_Tn_lt_three_over_four_l18_18120

noncomputable def sequence (n : ℕ) : ℕ → ℤ := 
  sorry

theorem find_general_formula :
  ∀ n, sequence (n + 1) = -3^(n + 1) + 1 := 
  sorry

def Sn (n : ℕ) : ℤ :=
  ∑ i in finset.range n, sequence i

def Tn (n : ℕ) : ℝ :=
  ∑ i in finset.range n, 1 / (real.log 3 (-sequence i + 1) * real.log 3 (-sequence (i + 2) + 1))

theorem Tn_lt_three_over_four (n : ℕ) : Tn n < 3 / 4 := 
  sorry

end find_general_formula_Tn_lt_three_over_four_l18_18120


namespace cotangent_identity_l18_18954

theorem cotangent_identity (a b c : ℝ) (α β γ : ℝ) (h_triangle : a^2 + b^2 = 2 * c^2) (h1 : α ≠ 0) (h2 : β ≠ 0) (h3 : γ ≠ 0):
  (Real.cot γ / (Real.cot α + Real.cot β)) = 1 / 2 :=
by
  sorry

end cotangent_identity_l18_18954


namespace ratio_areas_tends_to_zero_l18_18925

-- Definitions
def center_of_circle (O : Point) : Prop :=
  True

def radius_of_circle (r : ℝ) : Prop :=
  r > 0

def parallel (AB CD : Segment) : Prop :=
  True

def collinear (O P Q R : Point) : Prop :=
  True

def midpoint (P : Point) (A B : Point) : Prop :=
  True

def condition_RQ_QP (RQ QP : ℝ) : Prop :=
  RQ = 2 * QP

def area_of_square (AP : ℝ) : ℝ :=
  AP^2

def area_of_trapezoid (AP PC d : ℝ) : ℝ :=
  (AP + PC) / 2 * d

-- Theorem
theorem ratio_areas_tends_to_zero 
(O P Q R A B C D : Point) 
(r x : ℝ) 
(AP PC d T S : ℝ)
(hc : center_of_circle O) 
(hr : radius_of_circle r) 
(hpa : parallel (segment A B) (segment C D)) 
(hcol : collinear O P Q R) 
(hmp : midpoint P A B) 
(hrq_qp : condition_RQ_QP (2 * x) x)
(hAP : AP = sqrt (r^2 - (r - 3 * x)^2))
(hPC : PC = sqrt (r^2 - (r - 2 * x)^2))
(hS : S = area_of_square AP)
(hT : T = area_of_trapezoid AP PC (2 * x)):
tendsto (λ x, (T / S)) (𝓝 0) :=
by sorry

end ratio_areas_tends_to_zero_l18_18925


namespace toby_total_sales_at_garage_sale_l18_18320

noncomputable def treadmill_price : ℕ := 100
noncomputable def chest_of_drawers_price : ℕ := treadmill_price / 2
noncomputable def television_price : ℕ := treadmill_price * 3
noncomputable def three_items_total : ℕ := treadmill_price + chest_of_drawers_price + television_price
noncomputable def total_sales : ℕ := three_items_total / (3 / 4) -- 75% is 0.75 or 3/4

theorem toby_total_sales_at_garage_sale : total_sales = 600 :=
by
  unfold treadmill_price chest_of_drawers_price television_price three_items_total total_sales
  simp
  exact sorry

end toby_total_sales_at_garage_sale_l18_18320


namespace paintable_wall_area_l18_18785

theorem paintable_wall_area :
  let bedroom1_length := 14
      bedroom1_width := 11
      bedroom1_height := 9
      bedroom1_non_paintable := 50
      
      bedroom2_length := 13
      bedroom2_width := 12
      bedroom2_height := 9
      bedroom2_non_paintable := 55
      
      bedroom3_length := 15
      bedroom3_width := 10
      bedroom3_height := 9
      bedroom3_non_paintable := 45
      
      area1 := 2 * (bedroom1_length * bedroom1_height) + 2 * (bedroom1_width * bedroom1_height) - bedroom1_non_paintable
      area2 := 2 * (bedroom2_length * bedroom2_height) + 2 * (bedroom2_width * bedroom2_height) - bedroom2_non_paintable
      area3 := 2 * (bedroom3_length * bedroom3_height) + 2 * (bedroom3_width * bedroom3_height) - bedroom3_non_paintable
  in area1 + area2 + area3 = 1200 :=
begin
  sorry
end

end paintable_wall_area_l18_18785


namespace find_average_of_25_results_l18_18618

def avg_of_list (l : List ℝ) : ℝ := (l.sum) / (l.length)

theorem find_average_of_25_results : 
    ∀ (results : List ℝ), 
    (results.length = 25) → 
    (avg_of_list (results.take 12) = 14) → 
    (avg_of_list (results.drop 13) = 17) → 
    (results.nth 12 = some 878) → 
    avg_of_list results = 50 := 
by 
    intros results len_25 avg_first_12 avg_last_12 middle_val 
    sorry

end find_average_of_25_results_l18_18618


namespace count_integers_modulo_l18_18874

theorem count_integers_modulo (n : ℕ) (h₁ : n < 500) (h₂ : n % 7 = 4) : (setOf (λ n, n > 0 ∧ n < 500 ∧ n % 7 = 4)).card = 71 :=
sorry

end count_integers_modulo_l18_18874


namespace find_x_l18_18955

variable (a b x : ℝ)

def star (a b : ℝ) : ℝ := (Real.sqrt (a + b)) / (Real.sqrt (a - b))

theorem find_x (h : star x 10 = 6) : x = 74 / 7 := by
  sorry

end find_x_l18_18955


namespace modular_inverse_17_mod_800_l18_18332

theorem modular_inverse_17_mod_800 :
  ∃ x : ℤ, 17 * x ≡ 1 [MOD 800] ∧ 0 ≤ x ∧ x < 800 ∧ x = 753 := by
  sorry

end modular_inverse_17_mod_800_l18_18332


namespace count_non_one_five_digits_l18_18507

theorem count_non_one_five_digits (n : ℕ) : 
  (n >= 1 ∧ n <= 500) →
  (∀ (d : ℕ), d ∈ digits 10 n → d ≠ 1 ∧ d ≠ 5) →
  (count (λ x : ℕ, (x >= 1 ∧ x <= 500) ∧ (∀ d : ℕ, d ∈ digits 10 x → d ≠ 1 ∧ d ≠ 5)) (list.range 501)) = 511 :=
by sorry

end count_non_one_five_digits_l18_18507


namespace powers_of_two_div7_l18_18358

theorem powers_of_two_div7 (n : ℕ) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, n = 3 * k := sorry

end powers_of_two_div7_l18_18358


namespace extinction_prob_one_l18_18719

-- Define the probabilities
def p : ℝ := 0.6
def q : ℝ := 0.4

-- Define the extinction probability function
def extinction_prob (v : ℕ → ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1
  else p * v (k + 1) + q * v (k - 1)

-- State the theorem
theorem extinction_prob_one (v : ℕ → ℝ) :
  extinction_prob v 1 = 2 / 3 :=
sorry

end extinction_prob_one_l18_18719


namespace max_value_expression_l18_18111

noncomputable def max_expression (a b c : ℝ) : ℝ :=
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^3)

theorem max_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  max_expression a b c ≤ 1 / 12 := 
sorry

end max_value_expression_l18_18111


namespace volume_difference_l18_18379

def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h

def sphere_volume (r : ℝ) : ℝ := (4/3) * π * r^3

theorem volume_difference (r_c : ℝ) (r_s : ℝ) (h : ℝ) (W : ℝ) :
  r_c = 4 → r_s = 6 → h = 4 * real.sqrt 5 → W = 288 - 64 * real.sqrt 5 →
  sphere_volume r_s - cylinder_volume r_c h = W * π :=
by
  intros _
        _
        _
        _,
  sorry

end volume_difference_l18_18379


namespace handshake_problem_l18_18006

theorem handshake_problem (n : ℕ) (h : n = 60) : (n * (n - 1)) / 2 = 1770 :=
by
  rw h
  sorry

end handshake_problem_l18_18006


namespace compare_f_g_l18_18919

def R (m n : ℕ) : ℕ := sorry
def L (m n : ℕ) : ℕ := sorry

def f (m n : ℕ) : ℕ := R m n + L m n - sorry
def g (m n : ℕ) : ℕ := R m n + L m n - sorry

theorem compare_f_g (m n : ℕ) : f m n ≤ g m n := sorry

end compare_f_g_l18_18919


namespace find_a_l18_18838

noncomputable def curve (x : ℝ) (a : ℝ) : ℝ := x^4 + a * x^2 + 1
noncomputable def derivative (x : ℝ) (a : ℝ) : ℝ := 4 * x^3 + 2 * a * x

theorem find_a (a : ℝ) : (derivative (-1) a = 8) → (a = -6) :=
by
  assume h : (derivative (-1) a = 8)
  sorry

end find_a_l18_18838


namespace ice_cream_flavors_l18_18885

theorem ice_cream_flavors :
  let flavors := 3 in
  let scoops := 5 in
  combinatorics.binomial (scoops + flavors - 1) (flavors - 1) = 21 :=
by
  let flavors := 3
  let scoops := 5
  have h : combinatorics.binomial (scoops + flavors - 1) (flavors - 1) = combinatorics.binomial 7 2 := rfl
  rw [h]
  exact combinatorics.binomial_eq_21

end ice_cream_flavors_l18_18885


namespace remainder_of_exponentiated_sum_modulo_seven_l18_18420

theorem remainder_of_exponentiated_sum_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end remainder_of_exponentiated_sum_modulo_seven_l18_18420


namespace value_of_product_l18_18152

theorem value_of_product (a b : ℝ) (h1 : ∀ x : ℝ, 2 * x - a < 1 → x > 2 * b + 3) :
  (a+1) * (b-1) = -6 :=
by
  have h11 : 0 < 1, from zero_lt_one
  have h2 := h1 (2 * (b + 3) - (a + 1) / 2 - 1)
  have h3 := λ x : ℝ, x < (a+1) / 2 ∧ x > 2 * b + 3  
  have h4 := h3 (a / 2 - 3 * b - 1)
  sorry

end value_of_product_l18_18152


namespace abs_a_k_le_fractional_l18_18224

variable (a : ℕ → ℝ) (n : ℕ)

-- Condition 1: a_0 = a_(n+1) = 0
axiom a_0 : a 0 = 0
axiom a_n1 : a (n + 1) = 0

-- Condition 2: |a_{k-1} - 2a_k + a_{k+1}| ≤ 1 for k = 1, 2, ..., n
axiom abs_diff_ineq (k : ℕ) (h : 1 ≤ k ∧ k ≤ n) : 
  |a (k - 1) - 2 * a k + a (k + 1)| ≤ 1

-- Theorem statement
theorem abs_a_k_le_fractional (k : ℕ) (h : 0 ≤ k ∧ k ≤ n + 1) : 
  |a k| ≤ k * (n + 1 - k) / 2 := sorry

end abs_a_k_le_fractional_l18_18224


namespace sum_cis_theta_is_104_l18_18752

-- Define cis and use function definitions to encapsulate conditions.
def cis (θ : ℝ) : ℂ := complex.exp (complex.I * θ)

-- Define the series sum under the given conditions.
def series_sum : ℂ :=
  (cis (60 * π / 180) +
   cis (68 * π / 180) +
   cis (76 * π / 180) +
   cis (84 * π / 180) +
   cis (92 * π / 180) +
   cis (100 * π / 180) +
   cis (108 * π / 180) +
   cis (116 * π / 180) +
   cis (124 * π / 180) +
   cis (132 * π / 180) +
   cis (140 * π / 180) +
   cis (148 * π / 180))

-- Express the sum as r * cis θ and prove θ = 104 degrees.
theorem sum_cis_theta_is_104 : ∃ r : ℝ, series_sum = r * cis (104 * π / 180) :=
sorry

end sum_cis_theta_is_104_l18_18752


namespace min_value_sin4_plus_2_cos4_l18_18432

theorem min_value_sin4_plus_2_cos4 : ∀ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 ≥ 2 / 3 :=
by
  intro x
  sorry

end min_value_sin4_plus_2_cos4_l18_18432


namespace range_of_a_sqrt10_e_bounds_l18_18147

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f a x ≤ g x) ↔ a ≤ 1 :=
by
  sorry

theorem sqrt10_e_bounds : 
  (1095 / 1000 : ℝ) < Real.exp (1/10 : ℝ) ∧ Real.exp (1/10 : ℝ) < (2000 / 1791 : ℝ) :=
by
  sorry

end range_of_a_sqrt10_e_bounds_l18_18147


namespace sum_of_real_roots_l18_18442

theorem sum_of_real_roots (P : Polynomial ℝ) (hP : P = Polynomial.C 1 * X^4 - Polynomial.C 8 * X - Polynomial.C 2) :
  P.roots.sum = 2 :=
by {
  sorry
}

end sum_of_real_roots_l18_18442


namespace children_less_than_adults_l18_18018

theorem children_less_than_adults (total_members : ℕ)
  (percent_adults : ℝ) (percent_teenagers : ℝ) (percent_children : ℝ) :
  total_members = 500 →
  percent_adults = 0.45 →
  percent_teenagers = 0.25 →
  percent_children = 1 - percent_adults - percent_teenagers →
  (percent_children * total_members) - (percent_adults * total_members) = -75 := 
by
  intros h_total h_adults h_teenagers h_children
  sorry

end children_less_than_adults_l18_18018


namespace bert_fraction_spent_l18_18402

theorem bert_fraction_spent (f : ℝ) :
  let initial := 52
  let after_hardware := initial - initial * f
  let after_cleaners := after_hardware - 9
  let after_grocery := after_cleaners / 2
  let final := 15
  after_grocery = final → f = 1/4 :=
by
  intros h
  sorry

end bert_fraction_spent_l18_18402


namespace no_transformation_l18_18242

-- Define conditions as functions or predicates
def seated_people (n : Nat) : Prop := (n = 100)
def consecutive_four (l : List (Bool × Nat)) : Prop :=
  ∀ i: Nat, 0 ≤ i → i < l.length → 
    ((l.get! i = (true, 1) ∧ l.get! (i + 1 % l.length) = (false, 1) ∧ l.get! (i + 2 % l.length) = (true, 1) ∧ l.get! (i + 3 % l.length) = (false, 1)) ∨ 
     (l.get! i = (false, 1) ∧ l.get! (i + 1 % l.length) = (true, 1) ∧ l.get! (i + 2 % l.length) = (false, 1) ∧ l.get! (i + 3 % l.length) = (true, 1)))

def enough_wizards (l : List (Bool × Nat)) : Prop :=
∀ i: Nat, l.get! i = (true, 1) → 
          (l.get! ((i - 2 + l.length) % l.length) = (true, 1) ∨ 
           l.get! ((i - 1 + l.length) % l.length) = (true, 1) ∨ 
           l.get! ((i + 1) % l.length) = (true, 1) ∨ 
           l.get! ((i + 2) % l.length) = (true, 1))

-- Main Theorem
theorem no_transformation : 
  seated_people 100 → 
  consecutive_four [(true, 1), (false, 1), ... , (true, 1), (false, 1)] →
  ¬(enough_wizards [(true, 1), (false, 1), ... , (true, 1), (false, 1)]) :=
sorry

end no_transformation_l18_18242


namespace net_change_in_price_l18_18171

theorem net_change_in_price (P : ℝ) : 
  ((P * 0.75) * 1.2 = P * 0.9) → 
  ((P * 0.9 - P) / P = -0.1) :=
by
  intro h
  sorry

end net_change_in_price_l18_18171


namespace fraction_of_area_outside_triangle_isosceles_right_l18_18568

noncomputable def fraction_outside_circle (r : ℝ) : ℝ :=
  let area_triangle := (1 / 2) * r^2
  let area_sector := (pi * r^2) / 4
  (area_triangle - area_sector) / area_triangle

theorem fraction_of_area_outside_triangle_isosceles_right (r : ℝ) :
  fraction_outside_circle r = 1 - (pi / 2) := 
sorry

end fraction_of_area_outside_triangle_isosceles_right_l18_18568


namespace containers_needed_l18_18984

-- Definitions for conditions
def suki_bags : ℝ := 6.75
def suki_weight_per_bag : ℝ := 27.0
def jimmy_bags : ℝ := 4.25
def jimmy_weight_per_bag : ℝ := 23.0
def container_weight : ℝ := 11.0

-- Total weights for Suki and Jimmy
def suki_total_weight : ℝ := suki_bags * suki_weight_per_bag
def jimmy_total_weight : ℝ := jimmy_bags * jimmy_weight_per_bag
def total_weight : ℝ := suki_total_weight + jimmy_total_weight

-- Number of containers needed
def num_containers : ℝ := total_weight / container_weight

theorem containers_needed : ⌈num_containers⌉ = 26 := by
  sorry

end containers_needed_l18_18984


namespace Emma_clock_problem_l18_18423

noncomputable def actual_time_when_600PM (synchronize_time correct_time lag_time : ℕ) (consistent_lag_rate : Prop) : ℕ :=
  sorry  -- The actual implementation of the function is omitted.

theorem Emma_clock_problem:
  actual_time_when_600PM 8 9 8.8 consistent_lag_rate = 19.367 ~= 7.22 :=
by
  sorry  -- Proof is omitted.

end Emma_clock_problem_l18_18423


namespace part1_part2_l18_18580

section EllipseProblem

variable {a b xM yM : ℝ}
variable {P Q : ℝ × ℝ}

/-- The conditions of the problem --/
def conditions := 
  (a > 0 ∧ b > 0 ∧ a > b) ∧ 
  (a - xM = xM / 3 ∧ -yM = (yM - b) / 3 ∧ xM = 3 * a / 4 ∧ yM = b / 4 ∧ b / (3 * a) = 1 / 6)

/-- First part to prove: Prove that a = 2b --/
theorem part1 (h : conditions) : a = 2 * b :=
sorry

/-- The second part to prove: Finding the equation of the ellipse --/
def circle (C : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) : Prop := 
  (p.1 - C.1)^2 + (p.2 - C.2)^2 = r

theorem part2 (h : conditions) (h_p : circle (2, 1) (sqrt 5) P) (h_q : circle (2, 1) (sqrt 5) Q) :
  (P.1 * Q.1 = 8 - 2 * b^2) →
  (|P.1 - Q.1| = 2 * sqrt 5) →
  ellipse E : ∃ b : ℝ, ∃ a : ℝ, (a = 2 * b ∧ ∀ x y : ℝ, (x^2) / (4 * b^2) + (y^2) / (b^2) = 1) :=
sorry

end EllipseProblem

end part1_part2_l18_18580


namespace value_of_a_l18_18499

noncomputable def sqrt (x : ℝ) : ℝ :=
  if x < 0 then 0 else Real.sqrt x

theorem value_of_a (a : ℝ) (hA : A = {-1, 0, a}) (hB : B = {0, sqrt a}) (h_subset : B ⊆ A) : a = 1 :=
by
  have h_sqrt_a_mem_A : sqrt a ∈ A :=
    sorry -- Reasoning that if B ⊆ A, then sqrt a ∈ A
  have h_sqrt_a_eq_a : sqrt a = a :=
    sorry -- Newton leveraging B ⊆ A and definitions of A and B
  have h_a_nonzero : a ≠ 0 :=
    sorry -- sqroot value constraints
  exact sorry -- Conclusion that combination of conditions leads to a = 1

end value_of_a_l18_18499


namespace frequency_of_largest_rectangle_area_l18_18188

theorem frequency_of_largest_rectangle_area (a : ℕ → ℝ) (sample_size : ℕ)
    (h_geom : ∀ n, a (n + 1) = 2 * a n) (h_sum : a 0 + a 1 + a 2 + a 3 = 1)
    (h_sample : sample_size = 300) : 
    sample_size * a 3 = 160 := by
  sorry

end frequency_of_largest_rectangle_area_l18_18188


namespace inequality_solution_l18_18612

variable {x : ℝ}

theorem inequality_solution :
  x ∈ Set.Ioo (-∞ : ℝ) 7 ∪ Set.Ioo (-∞ : ℝ) (-7) ∪ Set.Ioo (-7) 7 ↔ (x^2 - 49) / (x + 7) < 0 :=
by
  sorry

end inequality_solution_l18_18612


namespace strawberries_count_l18_18868

theorem strawberries_count (harvest_per_day : ℕ) (days_in_april : ℕ) (given_away : ℕ) (stolen : ℕ) :
  (harvest_per_day = 5) →
  (days_in_april = 30) →
  (given_away = 20) →
  (stolen = 30) →
  (harvest_per_day * days_in_april - given_away - stolen = 100) :=
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
  sorry,
}

end strawberries_count_l18_18868


namespace complex_square_real_iff_ab_zero_l18_18019

theorem complex_square_real_iff_ab_zero (a b : ℝ) : ((a + b * complex.I) ^ 2).im = 0 ↔ a * b = 0 :=
by
  sorry

end complex_square_real_iff_ab_zero_l18_18019


namespace contrapositive_equivalence_l18_18296

variable (p q : Prop)

theorem contrapositive_equivalence : (p → ¬q) ↔ (q → ¬p) := by
  sorry

end contrapositive_equivalence_l18_18296


namespace calculate_bc_over_ad_l18_18945

noncomputable def volume_S_r (r : ℝ) (a b c d : ℝ) : ℝ :=
  a * r^3 + b * r^2 + c * r + d

theorem calculate_bc_over_ad :
  let B := (2, 4, 6) in
  let a := 4 * Real.pi / 3 in
  let b := 144 * Real.pi in
  let c := 88 in
  let d := 48 in
  0 <= r → (bc / ad = 66) :=
by
  sorry

end calculate_bc_over_ad_l18_18945


namespace find_original_polynomial_calculate_correct_result_l18_18344

variable {P : Polynomial ℝ}
variable (Q : Polynomial ℝ := 2 * X ^ 2 + X - 5)
variable (R : Polynomial ℝ := X ^ 2 + 3 * X - 1)

theorem find_original_polynomial (h : P - Q = R) : P = 3 * X ^ 2 + 4 * X - 6 :=
by
  sorry

theorem calculate_correct_result (h : P = 3 * X ^ 2 + 4 * X - 6) : P - Q = X ^ 2 + X + 9 :=
by
  sorry

end find_original_polynomial_calculate_correct_result_l18_18344


namespace value_of_expression_l18_18473

-- Definitions of the conditions
def α := π / 4
def β := sorry

-- Main theorem statement
theorem value_of_expression : 
    (tan (α + β) = 1 / 3) → 
    (1 - cos (2 * β)) / (sin (2 * β)) = -1 / 2 :=
by sorry

end value_of_expression_l18_18473


namespace sachin_age_l18_18674

variable {S R : ℕ}

theorem sachin_age
  (h1 : R = S + 7)
  (h2 : S * 3 = 2 * R) :
  S = 14 :=
sorry

end sachin_age_l18_18674


namespace land_for_cattle_l18_18550

-- Define the conditions as Lean definitions
def total_land : ℕ := 150
def house_and_machinery : ℕ := 25
def future_expansion : ℕ := 15
def crop_production : ℕ := 70

-- Statement to prove
theorem land_for_cattle : total_land - (house_and_machinery + future_expansion + crop_production) = 40 :=
by
  sorry

end land_for_cattle_l18_18550


namespace diagonal_segments_in_rectangle_l18_18461

/-- The number of segments into which the grid lines divide the diagonal of a rectangle
    of dimensions 100 × 101 is 200. -/
theorem diagonal_segments_in_rectangle (a b : ℕ) (h₁ : a = 100) (h₂ : b = 101) :
  let gcd_ab := Nat.gcd a b in
  let num_segments := a + b - gcd_ab in
  num_segments = 200 :=
by {
  rw [h₁, h₂],
  have h_gcd : Nat.gcd 100 101 = 1 := Nat.gcd_eq_one_iff_coprime.mpr $ by norm_num,
  rw [h_gcd],
  norm_num,
}

example : diagonal_segments_in_rectangle 100 101 100.rfl 101.rfl := by sorry

end diagonal_segments_in_rectangle_l18_18461


namespace combined_average_speed_l18_18313

theorem combined_average_speed 
    (dA tA dB tB dC tC : ℝ)
    (mile_feet : ℝ)
    (hA : dA = 300) (hTA : tA = 6)
    (hB : dB = 400) (hTB : tB = 8)
    (hC : dC = 500) (hTC : tC = 10)
    (hMileFeet : mile_feet = 5280) :
    (1200 / 5280) / (24 / 3600) = 34.09 := 
by
  sorry

end combined_average_speed_l18_18313


namespace not_sufficient_info_to_find_time_l18_18687

-- Defining the given conditions
constant length_of_train : ℝ
constant time_to_cross_signal : ℝ
constant length_of_bridge : ℝ

axioms
  (length_of_train_eq_600 : length_of_train = 600)
  (time_to_cross_signal_eq_40 : time_to_cross_signal = 40)
  (length_of_bridge_eq_18000 : length_of_bridge = 18000) -- 18 kilometers converted to meters

-- The statement to be proved
theorem not_sufficient_info_to_find_time
  (u a : ℝ) : ¬(∃ t : ℝ, t = (length_of_bridge + u * time_to_cross_signal + (1/2) * a * time_to_cross_signal^2) / (u + a * time_to_cross_signal)) :=
by sorry

end not_sufficient_info_to_find_time_l18_18687


namespace expression_defined_iff_l18_18419

theorem expression_defined_iff (a : ℝ) :
  (∃ y, y = (sqrt(4 * a - 2) / log 4 (3 - a))) ↔ (1 / 2 ≤ a ∧ a < 2) ∨ (2 < a ∧ a < 3) :=
sorry

end expression_defined_iff_l18_18419


namespace curve_transformation_l18_18041

-- Define the scaling transformation
def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (5 * x, 3 * y)

-- Define the transformed curve
def transformed_curve (x' y' : ℝ) : Prop :=
  2 * x' ^ 2 + 8 * y' ^ 2 = 1

-- Define the curve C's equation after scaling
def curve_C (x y : ℝ) : Prop :=
  50 * x ^ 2 + 72 * y ^ 2 = 1

-- Statement of the proof problem
theorem curve_transformation (x y : ℝ) (h : transformed_curve (5 * x) (3 * y)) : curve_C x y :=
by {
  -- The actual proof would be filled in here
  sorry
}

end curve_transformation_l18_18041


namespace rational_sum_zero_cube_nonzero_fifth_power_zero_l18_18889

theorem rational_sum_zero_cube_nonzero_fifth_power_zero
  (a b c : ℚ) 
  (h_sum : a + b + c = 0)
  (h_cube_nonzero : a^3 + b^3 + c^3 ≠ 0) 
  : a^5 + b^5 + c^5 = 0 :=
sorry

end rational_sum_zero_cube_nonzero_fifth_power_zero_l18_18889


namespace combine_syllables_to_computer_l18_18308

/-- Conditions provided in the problem -/
def first_syllable : String := "ком" -- A big piece of a snowman
def second_syllable : String := "пьют" -- Something done by elephants at a watering hole
def third_syllable : String := "ер" -- The old name of the hard sign

/-- The result obtained by combining the three syllables should be "компьютер" -/
theorem combine_syllables_to_computer :
  (first_syllable ++ second_syllable ++ third_syllable) = "компьютер" :=
by
  -- Proof to be provided
  sorry

end combine_syllables_to_computer_l18_18308


namespace MEMOrable_rectangle_count_l18_18557

section MEMOrable_rectangles

variables (K L : ℕ) (hK : K > 0) (hL : L > 0) 

/-- In a 2K x 2L board, if the ant starts at (1,1) and ends at (2K, 2L),
    and some squares may remain unvisited forming a MEMOrable rectangle,
    then the number of such MEMOrable rectangles is (K(K+1)L(L+1))/2. -/
theorem MEMOrable_rectangle_count :
  ∃ (n : ℕ), n = K * (K + 1) * L * (L + 1) / 2 :=
by
  sorry

end MEMOrable_rectangles

end MEMOrable_rectangle_count_l18_18557


namespace carolyn_silverware_knives_percentage_l18_18059

theorem carolyn_silverware_knives_percentage :
  (let knives_initial := 6 in
   let forks_initial := 12 in
   let spoons_initial := 3 * knives_initial in
   let total_silverware_initial := knives_initial + forks_initial + spoons_initial in
   let knives_after_trade := 0 in
   let spoons_after_trade := spoons_initial + 6 in
   let total_silverware_after_trade := knives_after_trade + forks_initial + spoons_after_trade in
   percentage_knives := (knives_after_trade * 100) / total_silverware_after_trade in
   percentage_knives = 0) :=
by
  sorry

end carolyn_silverware_knives_percentage_l18_18059


namespace chain_of_concepts_l18_18193

-- Definitions based on the conditions (steps in the chain of concepts)
inductive GeometricFigure : Type
| general_figure : GeometricFigure
deriving DecidableEq

inductive PlaneGeometricFigure : Type
| plane_figure : PlaneGeometricFigure
deriving DecidableEq

inductive ClosedPlaneGeometricFigure : Type
| closed_plane_figure : ClosedPlaneGeometricFigure
deriving DecidableEq

inductive Polygon : Type
| polygon : Polygon
deriving DecidableEq

inductive Quadrilateral : Type
| quadrilateral : Quadrilateral
deriving DecidableEq

inductive Parallelogram : Type
| parallelogram : Parallelogram
deriving DecidableEq

inductive Rhombus : Type
| rhombus : Rhombus
deriving DecidableEq

inductive Rectangle : Type
| rectangle : Rectangle
deriving DecidableEq

inductive Square : Type
| square : Square
deriving DecidableEq

-- The specific square A B C D
structure SquareABCD : Type :=
  (A B C D : Square)
deriving DecidableEq

-- Statement to prove the conceptual chain leading to a specific square AB CD
theorem chain_of_concepts:
  ∃ (gf: GeometricFigure) (pgf: PlaneGeometricFigure) (cpgf: ClosedPlaneGeometricFigure)
    (p: Polygon) (q: Quadrilateral) (par: Parallelogram) (r: Rhombus) (rect: Rectangle) (s: Square),
    (gf = GeometricFigure.general_figure) ∧
    (pgf = PlaneGeometricFigure.plane_figure) ∧
    (cpgf = ClosedPlaneGeometricFigure.closed_plane_figure) ∧
    (p = Polygon.polygon) ∧
    (q = Quadrilateral.quadrilateral) ∧
    (par = Parallelogram.parallelogram) ∧
    (r = Rhombus.rhombus) ∧
    (rect = Rectangle.rectangle) ∧
    (s = Square.square) ∧
    (specific_square: SquareABCD := {A := s, B := s, C := s, D := s}):
  True :=
by
  sorry

end chain_of_concepts_l18_18193


namespace resulting_total_mass_l18_18709

-- Define initial conditions
def initial_total_mass : ℝ := 12
def initial_white_paint_mass : ℝ := 0.8 * initial_total_mass
def initial_black_paint_mass : ℝ := initial_total_mass - initial_white_paint_mass

-- Required condition for the new mixture
def final_white_paint_percentage : ℝ := 0.9

-- Prove that the resulting total mass of paint is 24 kg
theorem resulting_total_mass (x : ℝ) (h1 : initial_total_mass = 12) 
                            (h2 : initial_white_paint_mass = 0.8 * initial_total_mass)
                            (h3 : initial_black_paint_mass = initial_total_mass - initial_white_paint_mass)
                            (h4 : final_white_paint_percentage = 0.9) 
                            (h5 : (initial_white_paint_mass + x) / (initial_total_mass + x) = final_white_paint_percentage) : 
                            initial_total_mass + x = 24 :=
by 
  -- Temporarily assume the proof without detailing the solution steps
  sorry

end resulting_total_mass_l18_18709


namespace pow_div_l18_18054

theorem pow_div (a : ℝ) : (-a) ^ 6 / a ^ 3 = a ^ 3 := by
  sorry

end pow_div_l18_18054


namespace area_BEIH_l18_18741

open Real

def midpoint (p1 p2 : Point) : Point := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def intersect (l1 l2 : Line) : Point := 
  sorry -- given the equations of the lines, find and return the intersection point

def quadrilateral_area (p1 p2 p3 p4 : Point) : Real := 
  let triangle_area (A B C : Point) := (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))
  triangle_area p1 p2 p3 + triangle_area p1 p3 p4

theorem area_BEIH : 
  (ABCD : Square) (E := midpoint ABCD.A ABCD.B) (F := midpoint ABCD.B ABCD.C)
  (I := intersect (line_through ABCD.A F) (line_through ABCD.D E))
  (H := intersect (line_through ABCD.B ABCD.D) (line_through ABCD.A F)) :
  quadrilateral_area ABCD.B E I H = 7 / 15 := by
  sorry

end area_BEIH_l18_18741


namespace minimum_roots_in_interval_l18_18768

noncomputable def g : ℝ → ℝ := sorry

lemma symmetry_condition_1 (x : ℝ) : g (3 + x) = g (3 - x) := sorry
lemma symmetry_condition_2 (x : ℝ) : g (8 + x) = g (8 - x) := sorry
lemma initial_condition : g 1 = 0 := sorry

theorem minimum_roots_in_interval : 
  ∃ k, ∀ x, -1000 ≤ x ∧ x ≤ 1000 → g x = 0 ∧ 
  (2 * k) = 286 := sorry

end minimum_roots_in_interval_l18_18768


namespace make_all_stones_black_l18_18361

theorem make_all_stones_black (n : ℕ) (hn : n = 2009) : 
  ∃ k : ℕ, k = 1005 ∧ 
  (∀ stones : Fin n → Prop, 
     (stones 1005 = true) ∧ 
     (∀ i : Fin n, i ≠ 1005 → stones i = false) → 
     (∃ m : ℕ, 
        (∀ j : Fin n, stones j = true))) :=
by
  sorry

end make_all_stones_black_l18_18361


namespace degree_measure_angle_ACB_l18_18042

-- Define the spherical coordinates of points A and B
structure Point where
  rho : ℝ
  theta : ℝ
  phi : ℝ

-- Define the conversion to Cartesian coordinates
def toCartesian (p : Point) : ℝ × ℝ × ℝ :=
  let x := p.rho * Math.sin p.theta * Math.cos p.phi
  let y := p.rho * Math.sin p.theta * Math.sin p.phi
  let z := p.rho * Math.cos p.theta
  (x, y, z)

-- Define the dot product of two vectors
def dotProduct (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Math.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Define the angle between two vectors
def angleBetween (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  Math.acos (dotProduct v1 v2 / (magnitude v1 * magnitude v2))

-- Problem statement in Lean 4
theorem degree_measure_angle_ACB :
  let A := Point.mk 1 (Math.pi / 2) 0
  let B := Point.mk 1 (Math.pi / 4) (3 * Math.pi / 4)
  angleBetween (toCartesian A) (toCartesian B) = 2 * Math.pi / 3 :=
by
  sorry

end degree_measure_angle_ACB_l18_18042


namespace transformation_preserves_point_l18_18203

variables {P : Type} [affine_space P ℝ] [linear_map ℝ P]
variables {l_A l_B l_C l_gamma_A : set P} {gamma_B gamma_C : set P}
variable (P : P)

-- Assume P is a point on line l_A
variable (P_on_lA : P ∈ l_A)

-- Definitions of transformations f_A, f_B, and f_C
def f_A (P : P) : P := sorry -- Define transformation corresponding to line l_A
def f_B (P : P) : P := sorry -- Define transformation corresponding to line l_B
def f_C (P : P) : P := sorry -- Define transformation corresponding to line l_C

-- Condition: f_C(f_B(f_A(P))) should be equal to P
theorem transformation_preserves_point : f_C (f_B (f_A P)) = P :=
sorry

end transformation_preserves_point_l18_18203


namespace cube_angle_face_diagonals_l18_18632

def cube_face_diagonal_angle (c : ℝ) : ℝ :=
  let a := c * (Real.sqrt 2)
  let b := c * (Real.sqrt 2)
  let cos_γ := (a^2 + b^2 - c^2) / (2 * a * b)
  Real.arccos cos_γ

theorem cube_angle_face_diagonals : 
  ∀ (c : ℝ), cube_face_diagonal_angle c = 60 :=
by 
  intros c 
  have h_face_diag : c * (Real.sqrt 2)^2 = 2 * c^2 := sorry
  have h_cos_γ : ((c * Real.sqrt 2)^2 + (c * Real.sqrt 2)^2 - c^2) / (2 * (c * Real.sqrt 2) * (c * Real.sqrt 2)) = 1 / 2 := sorry
  have h_arccos : Real.arccos (1 / 2) = 60 := sorry
  exact h_arccos

end cube_angle_face_diagonals_l18_18632


namespace identity_equality_l18_18195

theorem identity_equality (a b m n x y : ℝ) :
  ((a^2 + b^2) * (m^2 + n^2) * (x^2 + y^2)) =
  ((a * n * y - a * m * x - b * m * y + b * n * x)^2 + (a * m * y + a * n * x + b * m * x - b * n * y)^2) :=
by
  sorry

end identity_equality_l18_18195


namespace cost_is_695_l18_18166

noncomputable def cost_of_article (C : ℝ) : Prop :=
    let gain₇₈₅ := 785 - C in
    let gain₈₉₅ := 895 - C in
    (1.075 * gain₇₈₅) = gain₈₉₅

theorem cost_is_695 : cost_of_article 695 :=
by
    let C := 695
    let gain₇₈₅ := 785 - C
    let gain₈₉₅ := 895 - C
    have h₁ : (1.075 * gain₇₈₅) = gain₈₉₅ := sorry
    exact h₁

end cost_is_695_l18_18166


namespace positive_int_conditions_l18_18425

open Nat

/-- 
Proof that the only positive integers n satisfying:
1. the number of positive divisors of n is not a multiple of 8
2. for all integers x, x^n ≡ x mod n
are n = 1 and n is a prime number.
-/
theorem positive_int_conditions (n : ℕ) (h1 : ¬ ∃ m, numberDivisors n = 8 * m) (h2 : ∀ x : ℤ, x^n % n = x % n) : 
  n = 1 ∨ is_prime n := 
sorry

end positive_int_conditions_l18_18425


namespace find_coordinates_of_P_l18_18125

noncomputable def find_point (P : ℝ × ℝ) :=
  ∃ x y : ℝ, 
    P = (x, y) ∧ 
    y = x^2 ∧
    abs y = abs x + 3 ∧
    (P = (1, 4) ∨ P = (-1, 4))

theorem find_coordinates_of_P :
  ∃ P : ℝ × ℝ, find_point P :=
begin
  sorry
end

end find_coordinates_of_P_l18_18125


namespace evaluate_g_at_2_l18_18849

def g (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem evaluate_g_at_2 : g 2 = 5 :=
by
  sorry

end evaluate_g_at_2_l18_18849


namespace seating_arrangement_l18_18783

theorem seating_arrangement (x y : ℕ) (h : 9 * x + 7 * y = 112) : x = 7 :=
by
  sorry

end seating_arrangement_l18_18783


namespace combined_cost_of_ads_l18_18310

theorem combined_cost_of_ads
  (num_companies : ℕ)
  (ads_per_company : ℕ)
  (length : ℕ)
  (width : ℕ)
  (cost_per_sqft : ℕ)
  : num_companies = 3 →
    ads_per_company = 10 →
    length = 12 →
    width = 5 →
    cost_per_sqft = 60 →
    num_companies * ads_per_company * (length * width) * cost_per_sqft = 108000 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end combined_cost_of_ads_l18_18310


namespace find_b_l18_18295

noncomputable def projection_vector := λ (a : ℝ) (b : ℝ), (a / 14) * (15 + b) = 1 / 7

theorem find_b (b : ℝ) (h : projection_vector 3 1 2): b = -13 :=
by
  unfold projection_vector at h
  sorry

end find_b_l18_18295


namespace part_one_part_two_l18_18493

-- Define the set M implicitly by the condition
def M := {x : ℝ | -1/2 < x ∧ x < 1/2}

-- Assume a, b are elements of the set M
variables (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M)

-- Prove the first statement
theorem part_one (ha : a ∈ M) (hb : b ∈ M) : 
  abs(1/3 * a + 1/6 * b) < 1/4 :=
sorry

-- Prove the second statement
theorem part_two (ha : a ∈ M) (hb : b ∈ M) : 
  abs(1 - 4 * a * b) > 2 * abs(a - b) :=
sorry

end part_one_part_two_l18_18493


namespace percentage_good_oranges_tree_A_l18_18595

theorem percentage_good_oranges_tree_A
  (total_trees : ℕ)
  (trees_A : ℕ)
  (trees_B : ℕ)
  (total_good_oranges : ℕ)
  (oranges_A_per_month : ℕ) 
  (oranges_B_per_month : ℕ)
  (good_oranges_B_ratio : ℚ)
  (good_oranges_total_B : ℕ) 
  (good_oranges_total_A : ℕ)
  (good_oranges_total : ℕ)
  (x : ℚ) 
  (total_trees_eq : total_trees = 10)
  (tree_percentage_eq : trees_A = total_trees / 2 ∧ trees_B = total_trees / 2)
  (oranges_A_per_month_eq : oranges_A_per_month = 10)
  (oranges_B_per_month_eq : oranges_B_per_month = 15)
  (good_oranges_B_ratio_eq : good_oranges_B_ratio = 1/3)
  (good_oranges_total_eq : total_good_oranges = 55)
  (good_oranges_total_B_eq : good_oranges_total_B = trees_B * oranges_B_per_month * good_oranges_B_ratio)
  (good_oranges_total_A_eq : good_oranges_total_A = total_good_oranges - good_oranges_total_B):
  trees_A * oranges_A_per_month * x = good_oranges_total_A → 
  x = 0.6 := by
  sorry

end percentage_good_oranges_tree_A_l18_18595


namespace seats_to_remove_l18_18034

/-- Definition: number of seats per row -/
def seats_per_row : ℕ := 15

/-- Definition: total number of seats -/
def total_seats : ℕ := 225

/-- Definition: number of anticipated attendees -/
def anticipated_attendees : ℕ := 160

/-- Theorem: number of seats to be removed to minimize empty seats while maintaining complete rows -/
theorem seats_to_remove :
  let closest_multiple := seats_per_row * ((anticipated_attendees + seats_per_row - 1) / seats_per_row) in
  total_seats - closest_multiple = 60 :=
by
  sorry

end seats_to_remove_l18_18034


namespace convert_base8_to_base7_l18_18069

def base8_to_base10 (n : ℕ) : ℕ :=
  5 * 8^2 + 3 * 8^1 + 1 * 8^0

def base10_to_base7 (n : ℕ) : ℕ :=
  1002  -- Directly providing the result from conditions given.

theorem convert_base8_to_base7 :
  base10_to_base7 (base8_to_base10 531) = 1002 := by
  sorry

end convert_base8_to_base7_l18_18069


namespace translation_rotation_change_l18_18301

/-- The shape, size, and direction of a graph do not change when it is rotated and translated -/
def correct_statement := False

/-- The shape and size of a graph do not change when it is rotated -/
axiom shape_size_invariant_rotation : ∀ (G : Graph), ∃ shape size, shape_size_of_graph_after_rotation(G) = shape_size

/-- The direction changes when a graph is rotated -/
axiom direction_changes_rotation : ∀ (G : Graph), ∃ direction, direction_of_graph_after_rotation(G) ≠ direction_of_graph(G)

/-- The shape and size of a graph do not change when it is translated -/
axiom shape_size_invariant_translation : ∀ (G : Graph), ∃ shape size, shape_size_of_graph_after_translation(G) = shape_size

/-- The direction of a graph remains unchanged when it is translated -/
axiom direction_invariant_translation : ∀ (G : Graph), ∃ direction, direction_of_graph_after_translation(G) = direction_of_graph(G)

theorem translation_rotation_change (G : Graph) : correct_statement :=
by
  -- Given the conditions, we need to show this theorem
  sorry

end translation_rotation_change_l18_18301


namespace knives_percentage_l18_18058

-- Definitions based on conditions
def initial_knives : ℕ := 6
def initial_forks : ℕ := 12
def initial_spoons : ℕ := 3 * initial_knives
def traded_knives : ℕ := 10
def traded_spoons : ℕ := 6

-- Definitions for calculations
def final_knives : ℕ := initial_knives + traded_knives
def final_spoons : ℕ := initial_spoons - traded_spoons
def total_silverware : ℕ := final_knives + final_spoons + initial_forks

-- Theorem to prove the percentage of knives
theorem knives_percentage : (final_knives * 100) / total_silverware = 40 := by
  sorry

end knives_percentage_l18_18058


namespace average_speed_round_trip_l18_18016

/--
Let \( d = 150 \) miles be the distance from City \( X \) to City \( Y \).
Let \( v1 = 50 \) mph be the speed from \( X \) to \( Y \).
Let \( v2 = 30 \) mph be the speed from \( Y \) to \( X \).
Then the average speed for the round trip is 37.5 mph.
-/
theorem average_speed_round_trip :
  let d := 150
  let v1 := 50
  let v2 := 30
  (2 * d) / ((d / v1) + (d / v2)) = 37.5 :=
by
  sorry

end average_speed_round_trip_l18_18016


namespace sum_of_squares_of_coefficients_l18_18163

theorem sum_of_squares_of_coefficients :
  ∃ (a b c d e f : ℤ), (∀ x : ℤ, 8 * x ^ 3 + 64 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) ∧ 
  (a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 + e ^ 2 + f ^ 2 = 356) := 
by
  sorry

end sum_of_squares_of_coefficients_l18_18163


namespace total_bending_angle_l18_18410

theorem total_bending_angle (n : ℕ) (h : n > 4) (θ : ℝ) (hθ : θ = 360 / (2 * n)) : 
  ∃ α : ℝ, α = 180 :=
by
  sorry

end total_bending_angle_l18_18410


namespace line_segment_length_l18_18375

theorem line_segment_length (x : ℝ) (h : x > 0) :
  (Real.sqrt ((x - 2)^2 + (6 - 2)^2) = 5) → (x = 5) :=
by
  intro h1
  sorry

end line_segment_length_l18_18375


namespace total_votes_l18_18366

theorem total_votes (P R : ℝ) (hP : P = 0.35) (diff : ℝ) (h_diff : diff = 1650) : 
  ∃ V : ℝ, P * V + (P * V + diff) = V ∧ V = 5500 :=
by
  use 5500
  sorry

end total_votes_l18_18366


namespace unique_prime_p_l18_18417

open Nat

theorem unique_prime_p (p : ℕ) : Prime p → Prime (p^2 - 6) → Prime (p^2 + 6) → p = 5 := 
by
  -- This is where the proof would go, but as requested, we provide only the statement.
  sorry

end unique_prime_p_l18_18417


namespace standing_arrangements_l18_18650

def people := ["甲", "乙", "丙"]

def steps := fin 7

axiom max_two_people_per_step (s : steps) : bool

axiom indistinguishable_positions (p1 p2 : people) (s : steps) : p1 = p2 → max_two_people_per_step s = tt

theorem standing_arrangements : 
  ∃ (arrangements : ℕ), arrangements = 336 ∧
  let combinations := 
    (for all (p1 p2 p3 : people), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → 
    (∃ s1 s2 s3 : steps, s1 ≠ s2 → s2 ≠ s3 → s1 ≠ s3 →
    max_two_people_per_step s1 = ff ∧ max_two_people_per_step s2 = ff ∧ max_two_people_per_step s3 = ff)) +
   (for all (p1 p2 : people) (sing : people), p1 ≠ p2 → sing ≠ p1 → sing ≠ p2 → 
    (∃ s1 s2 : steps, s1 ≠ s2 → 
    max_two_people_per_step s1 = tt ∧ indistinguishable_positions p1 p2 s1 ∧ max_two_people_per_step s2 = ff)) in 
  arrangements = combinations.succ := 
sorry

end standing_arrangements_l18_18650


namespace count_integers_modulo_l18_18876

theorem count_integers_modulo (n : ℕ) (h₁ : n < 500) (h₂ : n % 7 = 4) : (setOf (λ n, n > 0 ∧ n < 500 ∧ n % 7 = 4)).card = 71 :=
sorry

end count_integers_modulo_l18_18876


namespace min_value_of_f_range_of_a_l18_18141

-- Definitions for the problem conditions
def f (x a : ℝ) : ℝ := log (x + a / x - 2)

-- Assertions to be proven in Lean
theorem min_value_of_f (a : ℝ) (h1 : 1 < a) (h2 : a < 4) : ∃ x : ℝ, 2 ≤ x ∧ (∀ x' : ℝ, 2 ≤ x' → f x' a ≥ f x a) ∧ f x a = log (a / 2) :=
sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 2 ≤ x → f x a > 0) : a > 2 :=
sorry

end min_value_of_f_range_of_a_l18_18141


namespace f_g_3_eq_l18_18575

def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 15 / Real.sqrt x

def g (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 4

theorem f_g_3_eq : f (g 3) = (66 / 17) * Real.sqrt 17 := 
by
  sorry

end f_g_3_eq_l18_18575


namespace volume_of_cuboid_l18_18399

def height : ℕ := 3
def width : ℕ := 3
def depth : ℕ := 2

theorem volume_of_cuboid :
  height * width * depth = 18 := by
  sorry

end volume_of_cuboid_l18_18399


namespace count_congruent_to_4_mod_7_l18_18877

theorem count_congruent_to_4_mod_7 : 
  ∃ (n : ℕ), 
  n = 71 ∧ 
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 70 → ∃ m : ℕ, m = 4 + 7 * k ∧ m < 500 := 
by
  sorry

end count_congruent_to_4_mod_7_l18_18877


namespace min_norm_vector_condition_l18_18951

open Real

variable (u : ℝ × ℝ)

def vector_condition (u : ℝ × ℝ) : Prop :=
  ∥(u.1 + 4, u.2 + 2)∥ = 10

theorem min_norm_vector_condition (u : ℝ × ℝ) (h : vector_condition u) : 
  ∥u∥ = 10 - 2 * sqrt 5 :=
sorry

end min_norm_vector_condition_l18_18951


namespace inequality_solution_l18_18602

theorem inequality_solution (x : ℝ) (hx : x ≠ -7) :
  (x^2 - 49) / (x + 7) < 0 ↔ x ∈ set.Ioo (-∞) (-7) ∪ set.Ioo (-7) 7 := by
  sorry

end inequality_solution_l18_18602


namespace gretchen_total_money_l18_18862

def charge_per_drawing : ℕ := 20
def sold_saturday : ℕ := 24
def sold_sunday : ℕ := 16
def total_caricatures := sold_saturday + sold_sunday

theorem gretchen_total_money : charge_per_drawing * total_caricatures = 800 := by
  have total_caricatures_eq : total_caricatures = 40 := by
    unfold total_caricatures
    simp
  rw [total_caricatures_eq]
  calc
    20 * 40 = 800 := by norm_num

end gretchen_total_money_l18_18862


namespace triangle_cosine_theorem_l18_18169

def triangle_sums (a b c : ℝ) : ℝ := 
  b^2 + c^2 - a^2 + a^2 + c^2 - b^2 + a^2 + b^2 - c^2

theorem triangle_cosine_theorem (a b c : ℝ) (cos_A cos_B cos_C : ℝ) :
  a = 2 → b = 3 → c = 4 → 2 * b * c * cos_A + 2 * c * a * cos_B + 2 * a * b * cos_C = 29 :=
by
  intros h₁ h₂ h₃
  sorry

end triangle_cosine_theorem_l18_18169


namespace angle_between_u_v_l18_18566

open Real

variables (u v w : ℝ^3)
variables (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (hw : ‖w‖ = 1)
variables (h_cross : u × (v × w) = (2 * v - w) / 3)
variables (h_independent : linear_independent ℝ ![u, v, w])

theorem angle_between_u_v : ∃ θ : ℝ, cos θ = -1/3 := by
  sorry

end angle_between_u_v_l18_18566


namespace sum_of_rational_roots_of_h_l18_18080

noncomputable def h (x : ℚ) : ℚ := x^3 - 6 * x^2 + 11 * x - 6

theorem sum_of_rational_roots_of_h :
  let roots := {r : ℚ | h r = 0} in
  ∑ r in roots.to_finset, r = 6 :=
sorry

end sum_of_rational_roots_of_h_l18_18080


namespace inequality_solution_l18_18601

theorem inequality_solution (x : ℝ) (hx : x ≠ -7) :
  (x^2 - 49) / (x + 7) < 0 ↔ x ∈ set.Ioo (-∞) (-7) ∪ set.Ioo (-7) 7 := by
  sorry

end inequality_solution_l18_18601


namespace min_A_b_l18_18558

section
  variable (a b : ℝ)

  def f_ab (x y : ℝ) : ℝ × ℝ :=
    (a - b*y - x^2, x)

  def iter_f_ab (n : ℕ) (p : ℝ × ℝ) : ℝ × ℝ :=
    Nat.recOn n p (λ k pk => f_ab a b pk.1 pk.2)

  def per (a b : ℝ) (p : ℝ × ℝ) : Prop :=
    ∃ n : ℕ, n > 0 ∧ iter_f_ab a b n p = p

  def A_b (b : ℝ) : Set ℝ :=
    {a | ∃ p : ℝ × ℝ, per a b p}

  theorem min_A_b (b : ℝ) : ∃ m : ℝ, is_least (A_b b) m ∧ m = -(b+1)^2 / 4 :=
  by
    sorry
end

end min_A_b_l18_18558


namespace range_k_not_monotonic_l18_18848

def f (x : ℝ) : ℝ := x^2 + x - Real.log x + 1

theorem range_k_not_monotonic :
  ∀ (k : ℝ), (∃ (x : ℝ), 2 * k - 1 < x ∧ x < k + 2 ∧ f' x = 0) ↔ (1 / 2 ≤ k ∧ k < 3 / 4) :=
by
  -- Say that f' is the derivative of f
  let f' := fun x : ℝ => 2 * x + 1 - (1 / x)
  apply sorry

end range_k_not_monotonic_l18_18848


namespace set_infinite_l18_18573

variable {S : Set ℕ}

theorem set_infinite 
  (h1 : ∀ a ∈ S, ∃ b c ∈ S, a = (b * (3 * c - 5)) / 15) : 
  S.Infinite :=
by
  sorry

end set_infinite_l18_18573


namespace inequality_solution_l18_18262

theorem inequality_solution (x : ℝ) : 
  (1 - (1 / (3 * x + 4)) < 3) ↔ 
    (x ∈ set.Ioo (-∞) (-5/3 : ℝ) ∪ set.Ioo (-4/3 : ℝ) ∞) :=
by
  sorry

end inequality_solution_l18_18262


namespace angle_of_inclination_of_line_l18_18428

theorem angle_of_inclination_of_line :
  ∀ (θ : ℝ), 3 * x + sqrt 3 * y + 1 = 0 → θ = 120 :=
sorry

end angle_of_inclination_of_line_l18_18428


namespace common_tangent_intersection_l18_18678

noncomputable def triangle (A B C K1 K2 : Type) :=
∀ (a b c k1 k2 : A),
  k1 ∈ (insert b (insert c ∅)) ∧
  k2 ∈ (insert b (insert c ∅))

theorem common_tangent_intersection
  (ABC : Type) [triangle ABC ABC ABC ABC ABC]
  (K1 K2 : ABC) :
  ∃ (O : ABC), 
  (common_external_tangents (incircle (triangle ABC ABC ABC K1)) (incircle (triangle ABC ABC K2 ABC))) =
  (common_external_tangents (incircle (triangle ABC ABC ABC K2)) (incircle (triangle ABC ABC K1 ABC))) := 
sorry

end common_tangent_intersection_l18_18678


namespace actual_average_height_correct_l18_18675

-- Definitions based on conditions
def initial_average_height : ℝ := 183
def incorrect_height : ℝ := 166
def actual_height : ℝ := 106
def number_of_boys : ℝ := 35

-- Calculate the error and corrected total height
def height_error : ℝ := incorrect_height - actual_height
def incorrect_total_height : ℝ := initial_average_height * number_of_boys
def corrected_total_height : ℝ := incorrect_total_height - height_error

-- Calculate the actual average height
def actual_average_height : ℝ := corrected_total_height / number_of_boys

-- Statement to prove: The actual average height is 181.29 cm (rounded to two decimal places)
theorem actual_average_height_correct : actual_average_height = 181.29 := by
  sorry

end actual_average_height_correct_l18_18675


namespace part1_part2_l18_18581

def f (x : ℝ) : ℝ := x - Real.sin x
def seq (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a n.succ = f (a n)

theorem part1 (a : ℕ → ℝ) (h : seq a) (h1 : a 1 = 2) : a 2 > a 3 :=
  by sorry

theorem part2 (a : ℕ → ℝ) (h : seq a) (h1 : 0 < a 1) (h2 : a 1 < 1) :
  ∀ n : ℕ, 0 < a n.succ ∧ a n.succ < 1 :=
  by sorry

end part1_part2_l18_18581


namespace find_functional_form_l18_18792

theorem find_functional_form (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by
  sorry

end find_functional_form_l18_18792


namespace circumcenter_of_ATI_on_BC_l18_18814

-- Definitions corresponding to conditions in a)
variables {A B C I T B' C' B_1 C_1 : Point}
variable [NonIsoscelesTriangle A B C]
variable [Incenter I]

-- Perpendiculars from I to AI intersecting AC and AB at B' and C' respectively
axiom perp_from_I_to_AI_intersection_AC_B' : IsPerpendicular I AI AC B'
axiom perp_from_I_to_AI_intersection_AB_C' : IsPerpendicular I AI AB C'

-- Definitions of points B1 and C1 on rays BC and CB respectively
axiom B1_on_ray_BC : OnRay B C B_1
axiom C1_on_ray_CB : OnRay C B C_1

-- Constraints on distances
axiom AB_eq_BB1 : dist A B = dist B B_1
axiom AC_eq_CC1 : dist A C = dist C C_1

-- Second intersection point T of two circumcircles
axiom circumcircle_intersection_T : SecondIntersection (circumcircle A B_1 C') (circumcircle A C_1 B') T

-- Proof goal
theorem circumcenter_of_ATI_on_BC :
  IsOnLine (circumcenter A T I) (line B C) :=
sorry

end circumcenter_of_ATI_on_BC_l18_18814


namespace number_of_lamps_bought_l18_18940

-- Define the given conditions
def price_of_lamp : ℕ := 7
def price_of_bulb : ℕ := price_of_lamp - 4
def bulbs_bought : ℕ := 6
def total_spent : ℕ := 32

-- Define the statement to prove
theorem number_of_lamps_bought : 
  ∃ (L : ℕ), (price_of_lamp * L + price_of_bulb * bulbs_bought = total_spent) ∧ (L = 2) :=
sorry

end number_of_lamps_bought_l18_18940


namespace number_of_RNA_molecules_l18_18247

-- Define the type RNA_Base which represents the four bases: A, C, G, U.
inductive RNA_Base
| A
| C
| G
| U

-- Define the number of positions in the RNA sequence.
def num_positions : ℕ := 100

-- Define the proof statement: the number of distinct RNA molecules with num_positions bases is 4^num_positions.
theorem number_of_RNA_molecules : (fintype.card (fin num_positions → RNA_Base) = 4 ^ num_positions) :=
by sorry

end number_of_RNA_molecules_l18_18247


namespace total_selection_methods_l18_18047

-- Define the students and days
inductive Student
| S1 | S2 | S3 | S4 | S5

inductive Day
| Wednesday | Thursday | Friday | Saturday | Sunday

-- The condition where S1 cannot be on Saturday and S2 cannot be on Sunday
def valid_arrangement (arrangement : Day → Student) : Prop :=
  arrangement Day.Saturday ≠ Student.S1 ∧
  arrangement Day.Sunday ≠ Student.S2

-- The main statement
theorem total_selection_methods : ∃ (arrangement_count : ℕ), 
  arrangement_count = 78 ∧
  ∀ (arrangement : Day → Student), valid_arrangement arrangement → 
  arrangement_count = 78 :=
sorry

end total_selection_methods_l18_18047


namespace binomial_distribution_probability_l18_18823

open ProbabilityTheory

variables {n : ℕ} {p : ℝ} {ξ : ℕ → ℝ}

theorem binomial_distribution_probability :
  (∀ ξ, Binomial(12, 0.5) ξ) →
  (ξ = 6) →
  (Var ξ = 3) →
  (BinomialPDF n p ξ 1) = 3 * 2^(-10) :=
by sorry

end binomial_distribution_probability_l18_18823


namespace complex_number_additive_inverses_l18_18521

theorem complex_number_additive_inverses (m : ℂ) :
  let z := (1 - m * complex.I) / (1 - 2 * complex.I) in
  (z.re + z.im = 0) → m = -3 :=
by
  let z := (1 - m * complex.I) / (1 - 2 * complex.I)
  assume h : z.re + z.im = 0
  sorry

end complex_number_additive_inverses_l18_18521


namespace factor_multiplication_l18_18671

theorem factor_multiplication (w m z : ℝ) :
  let q := 5 * w / (4 * m * (z ^ 2)) in
  let q_new := (5 * (4 * w)) / (4 * (2 * m) * ((3 * z) ^ 2)) in
  (q_new / q) = (2 / 9) :=
by
  sorry

end factor_multiplication_l18_18671


namespace cookies_unclaimed_fraction_l18_18043

theorem cookies_unclaimed_fraction (x : ℝ) (hx : x > 0) :
  let al_share := (4 / 9) * x
  let remaining_after_al := x - al_share
  let bert_share := (3 / 9) * remaining_after_al
  let remaining_after_bert := remaining_after_al - bert_share
  let carl_share := (2 / 9) * remaining_after_bert
  let remaining_after_carl := remaining_after_bert - carl_share
  remaining_after_carl / x = 230 / 243 :=
by {
  have h1 : al_share = (4 / 9) * x := rfl,
  have h2 : remaining_after_al = x - al_share := rfl,
  have h3 : bert_share = (3 / 9) * remaining_after_al := rfl,
  have h4 : remaining_after_bert = remaining_after_al - bert_share := rfl,
  have h5 : carl_share = (2 / 9) * remaining_after_bert := rfl,
  have h6 : remaining_after_carl = remaining_after_bert - carl_share := rfl,
  have remaining_eq : remaining_after_carl = (230 / 243) * x, {
    sorry
  },
  calc
    remaining_after_carl / x = (230 / 243) * x / x : by rw remaining_eq
    ... = 230 / 243 : by field_simp [hx],
}

end cookies_unclaimed_fraction_l18_18043
