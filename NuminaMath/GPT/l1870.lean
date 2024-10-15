import Mathlib

namespace NUMINAMATH_GPT_add_decimals_l1870_187060

theorem add_decimals : 4.3 + 3.88 = 8.18 := 
sorry

end NUMINAMATH_GPT_add_decimals_l1870_187060


namespace NUMINAMATH_GPT_only_one_student_remains_l1870_187065

theorem only_one_student_remains (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2002) :
  (∃! k, k = n ∧ n % 1331 = 0) ↔ n = 1331 :=
by
  sorry

end NUMINAMATH_GPT_only_one_student_remains_l1870_187065


namespace NUMINAMATH_GPT_sqrt_12_lt_4_l1870_187054

theorem sqrt_12_lt_4 : Real.sqrt 12 < 4 := sorry

end NUMINAMATH_GPT_sqrt_12_lt_4_l1870_187054


namespace NUMINAMATH_GPT_boys_belong_to_other_communities_l1870_187010

/-- In a school of 300 boys, if 44% are Muslims, 28% are Hindus, and 10% are Sikhs,
then the number of boys belonging to other communities is 54. -/
theorem boys_belong_to_other_communities
  (total_boys : ℕ)
  (percentage_muslims percentage_hindus percentage_sikhs : ℕ)
  (b : total_boys = 300)
  (m : percentage_muslims = 44)
  (h : percentage_hindus = 28)
  (s : percentage_sikhs = 10) :
  total_boys * ((100 - (percentage_muslims + percentage_hindus + percentage_sikhs)) / 100) = 54 := 
sorry

end NUMINAMATH_GPT_boys_belong_to_other_communities_l1870_187010


namespace NUMINAMATH_GPT_problem_statement_l1870_187006

noncomputable def m (α : ℝ) : ℝ := - (Real.sqrt 2) / 4

noncomputable def tan_alpha (α : ℝ) : ℝ := 2 * Real.sqrt 2

theorem problem_statement (α : ℝ) (P : (ℝ × ℝ)) (h1 : P = (m α, 1)) (h2 : Real.cos α = - 1 / 3) :
  (P.1 = - (Real.sqrt 2) / 4) ∧ (Real.tan α = 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1870_187006


namespace NUMINAMATH_GPT_inequality_non_empty_solution_l1870_187087

theorem inequality_non_empty_solution (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x + 1 < 0) → a ≤ 1 := sorry

end NUMINAMATH_GPT_inequality_non_empty_solution_l1870_187087


namespace NUMINAMATH_GPT_range_of_a_l1870_187061
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 1 + a
noncomputable def g (x : ℝ) : ℝ := 3 * Real.log x

theorem range_of_a (h : ∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f x a = -g x) : 
  0 ≤ a ∧ a ≤ Real.exp 3 - 4 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1870_187061


namespace NUMINAMATH_GPT_parabola_tangent_circle_radius_l1870_187004

noncomputable def radius_of_tangent_circle : ℝ :=
  let r := 1 / 4
  r

theorem parabola_tangent_circle_radius :
  ∃ (r : ℝ), (∀ (x : ℝ), x^2 - x + r = 0 → (-1)^2 - 4 * 1 * r = 1 - 4 * r) ∧ r = 1 / 4 :=
by
  use 1 / 4
  sorry

end NUMINAMATH_GPT_parabola_tangent_circle_radius_l1870_187004


namespace NUMINAMATH_GPT_product_of_coordinates_of_intersection_l1870_187093

-- Conditions: Defining the equations of the two circles
def circle1_eq (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 25 = 0
def circle2_eq (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 37 = 0

-- Translated problem to prove the question equals the correct answer
theorem product_of_coordinates_of_intersection :
  ∃ (x y : ℝ), circle1_eq x y ∧ circle2_eq x y ∧ x * y = 10 :=
sorry

end NUMINAMATH_GPT_product_of_coordinates_of_intersection_l1870_187093


namespace NUMINAMATH_GPT_rebecca_tent_stakes_l1870_187078

-- Given conditions
variable (x : ℕ) -- number of tent stakes

axiom h1 : x + 3 * x + (x + 2) = 22 -- Total number of items equals 22

-- Proof objective
theorem rebecca_tent_stakes : x = 4 :=
by 
  -- Place for the proof. Using sorry to indicate it.
  sorry

end NUMINAMATH_GPT_rebecca_tent_stakes_l1870_187078


namespace NUMINAMATH_GPT_find_x_eq_eight_l1870_187018

theorem find_x_eq_eight (x : ℕ) : 3^(x-2) = 9^3 → x = 8 := 
by
  sorry

end NUMINAMATH_GPT_find_x_eq_eight_l1870_187018


namespace NUMINAMATH_GPT_train_speed_l1870_187088

theorem train_speed (length : ℕ) (time : ℝ)
  (h_length : length = 160)
  (h_time : time = 18) :
  (length / time * 3.6 : ℝ) = 32 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1870_187088


namespace NUMINAMATH_GPT_train_pass_bridge_time_l1870_187027

noncomputable def totalDistance (trainLength bridgeLength : ℕ) : ℕ :=
  trainLength + bridgeLength

noncomputable def speedInMPerSecond (speedInKmPerHour : ℕ) : ℝ :=
  (speedInKmPerHour * 1000) / 3600

noncomputable def timeToPass (totalDistance : ℕ) (speedInMPerSecond : ℝ) : ℝ :=
  totalDistance / speedInMPerSecond

theorem train_pass_bridge_time
  (trainLength : ℕ) (bridgeLength : ℕ) (speedInKmPerHour : ℕ)
  (h_train : trainLength = 300)
  (h_bridge : bridgeLength = 115)
  (h_speed : speedInKmPerHour = 35) :
  timeToPass (totalDistance trainLength bridgeLength) (speedInMPerSecond speedInKmPerHour) = 42.7 :=
by
  sorry

end NUMINAMATH_GPT_train_pass_bridge_time_l1870_187027


namespace NUMINAMATH_GPT_cindy_marbles_l1870_187083

-- Define the initial constants and their values
def initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def number_of_friends : ℕ := 4

-- Define the problem statement in Lean 4
theorem cindy_marbles :
  4 * (initial_marbles - (marbles_per_friend * number_of_friends)) = 720 := by
  sorry

end NUMINAMATH_GPT_cindy_marbles_l1870_187083


namespace NUMINAMATH_GPT_inverse_of_73_mod_74_l1870_187056

theorem inverse_of_73_mod_74 :
  73 * 73 ≡ 1 [MOD 74] :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_73_mod_74_l1870_187056


namespace NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l1870_187048

theorem equation_one_solution (x : ℝ) : (6 * x - 7 = 4 * x - 5) ↔ (x = 1) := by
  sorry

theorem equation_two_solution (x : ℝ) : ((x + 1) / 2 - 1 = 2 + (2 - x) / 4) ↔ (x = 4) := by
  sorry

end NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l1870_187048


namespace NUMINAMATH_GPT_complement_union_result_l1870_187012

open Set

variable (U A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3})
variable (hA : A = {1, 2})
variable (hB : B = {2, 3})

theorem complement_union_result : compl A ∪ B = {0, 2, 3} :=
by
  -- Our proof steps would go here
  sorry

end NUMINAMATH_GPT_complement_union_result_l1870_187012


namespace NUMINAMATH_GPT_smaller_number_l1870_187091

theorem smaller_number (L S : ℕ) (h₁ : L - S = 2395) (h₂ : L = 6 * S + 15) : S = 476 :=
by
sorry

end NUMINAMATH_GPT_smaller_number_l1870_187091


namespace NUMINAMATH_GPT_product_xyz_eq_one_l1870_187081

theorem product_xyz_eq_one (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) : x * y * z = 1 := 
sorry

end NUMINAMATH_GPT_product_xyz_eq_one_l1870_187081


namespace NUMINAMATH_GPT_squared_sum_l1870_187015

theorem squared_sum (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 :=
by
  sorry

end NUMINAMATH_GPT_squared_sum_l1870_187015


namespace NUMINAMATH_GPT_simplify_expression_l1870_187066

variable (t : ℝ)

theorem simplify_expression (ht : t > 0) (ht_ne : t ≠ 1 / 2) :
  (1 - Real.sqrt (2 * t)) / ( (1 - Real.sqrt (4 * t ^ (3 / 4))) / (1 - Real.sqrt (2 * t ^ (1 / 4))) - Real.sqrt (2 * t)) *
  (Real.sqrt (1 / (1 / 2) + Real.sqrt (4 * t ^ 2)) / (1 + Real.sqrt (1 / (2 * t))) - Real.sqrt (2 * t))⁻¹ = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1870_187066


namespace NUMINAMATH_GPT_problem_statement_l1870_187040

noncomputable def f : ℝ → ℝ := sorry

axiom func_condition : ∀ a b : ℝ, b^2 * f a = a^2 * f b
axiom f2_nonzero : f 2 ≠ 0

theorem problem_statement : (f 6 - f 3) / f 2 = 27 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1870_187040


namespace NUMINAMATH_GPT_almonds_walnuts_ratio_l1870_187017

-- Define the given weights and parts
def w_a : ℝ := 107.14285714285714
def w_m : ℝ := 150
def p_a : ℝ := 5

-- Now we will formulate the statement to prove the ratio of almonds to walnuts
theorem almonds_walnuts_ratio : 
  ∃ (p_w : ℝ), p_a / p_w = 5 / 2 :=
by
  -- It is given that p_a / p_w = 5 / 2, we need to find p_w
  sorry

end NUMINAMATH_GPT_almonds_walnuts_ratio_l1870_187017


namespace NUMINAMATH_GPT_jim_total_payment_l1870_187033

def lamp_cost : ℕ := 7
def bulb_cost : ℕ := lamp_cost - 4
def num_lamps : ℕ := 2
def num_bulbs : ℕ := 6

def total_cost : ℕ := (num_lamps * lamp_cost) + (num_bulbs * bulb_cost)

theorem jim_total_payment : total_cost = 32 := by
  sorry

end NUMINAMATH_GPT_jim_total_payment_l1870_187033


namespace NUMINAMATH_GPT_marcus_percentage_of_team_points_l1870_187074

theorem marcus_percentage_of_team_points
  (three_point_goals : ℕ)
  (two_point_goals : ℕ)
  (team_points : ℕ)
  (h1 : three_point_goals = 5)
  (h2 : two_point_goals = 10)
  (h3 : team_points = 70) :
  ((three_point_goals * 3 + two_point_goals * 2) / team_points : ℚ) * 100 = 50 := by
sorry

end NUMINAMATH_GPT_marcus_percentage_of_team_points_l1870_187074


namespace NUMINAMATH_GPT_pipe_fill_time_without_leak_l1870_187096

theorem pipe_fill_time_without_leak (T : ℝ) (h1 : T > 0) 
  (h2 : 1/T - 1/8 = 1/8) :
  T = 4 := 
sorry

end NUMINAMATH_GPT_pipe_fill_time_without_leak_l1870_187096


namespace NUMINAMATH_GPT_part1_l1870_187058

def U : Set ℝ := Set.univ
def P (a : ℝ) : Set ℝ := {x | 4 ≤ x ∧ x ≤ 7}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem part1 (a : ℝ) (P_def : P 3 = {x | 4 ≤ x ∧ x ≤ 7}) :
  ((U \ P a) ∩ Q = {x | -2 ≤ x ∧ x < 4}) := by
  sorry

end NUMINAMATH_GPT_part1_l1870_187058


namespace NUMINAMATH_GPT_cost_of_two_burritos_and_five_quesadillas_l1870_187041

theorem cost_of_two_burritos_and_five_quesadillas
  (b q : ℝ)
  (h1 : b + 4 * q = 3.50)
  (h2 : 4 * b + q = 4.10) :
  2 * b + 5 * q = 5.02 := 
sorry

end NUMINAMATH_GPT_cost_of_two_burritos_and_five_quesadillas_l1870_187041


namespace NUMINAMATH_GPT_birds_meeting_distance_l1870_187024

theorem birds_meeting_distance :
  ∀ (d distance speed1 speed2: ℕ),
  distance = 20 →
  speed1 = 4 →
  speed2 = 1 →
  (d / speed1) = ((distance - d) / speed2) →
  d = 16 :=
by
  intros d distance speed1 speed2 hdist hspeed1 hspeed2 htime
  sorry

end NUMINAMATH_GPT_birds_meeting_distance_l1870_187024


namespace NUMINAMATH_GPT_logical_equivalence_l1870_187034

variables {α : Type} (A B : α → Prop)

theorem logical_equivalence :
  (∀ x, A x → B x) ↔
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, ¬ B x → ¬ A x) :=
by sorry

end NUMINAMATH_GPT_logical_equivalence_l1870_187034


namespace NUMINAMATH_GPT_intersection_coords_perpendicular_line_l1870_187001

def line1 (x y : ℝ) := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) := x + y - 2 = 0

theorem intersection_coords : ∃ P : ℝ × ℝ, line1 P.1 P.2 ∧ line2 P.1 P.2 ∧ P = (1, 1) := by
  sorry

theorem perpendicular_line (x y : ℝ) (P : ℝ × ℝ) (hP: P = (1, 1)) : 
  (line2 P.1 P.2) → x - y = 0 := by
  sorry

end NUMINAMATH_GPT_intersection_coords_perpendicular_line_l1870_187001


namespace NUMINAMATH_GPT_initial_roses_in_vase_l1870_187003

theorem initial_roses_in_vase (added_roses current_roses : ℕ) (h1 : added_roses = 8) (h2 : current_roses = 18) : 
  current_roses - added_roses = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_roses_in_vase_l1870_187003


namespace NUMINAMATH_GPT_intersection_A_B_l1870_187031

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | abs (x - 2) ≥ 1}
def answer : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_A_B :
  A ∩ B = answer :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1870_187031


namespace NUMINAMATH_GPT_delaney_missed_bus_time_l1870_187086

def busDepartureTime : Nat := 480 -- 8:00 a.m. = 8 * 60 minutes
def travelTime : Nat := 30 -- 30 minutes
def departureFromHomeTime : Nat := 470 -- 7:50 a.m. = 7 * 60 + 50 minutes

theorem delaney_missed_bus_time :
  (departureFromHomeTime + travelTime - busDepartureTime) = 20 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_delaney_missed_bus_time_l1870_187086


namespace NUMINAMATH_GPT_augustus_makes_3_milkshakes_l1870_187098

def augMilkshakePerHour (A : ℕ) (Luna : ℕ) (hours : ℕ) (totalMilkshakes : ℕ) : Prop :=
  (A + Luna) * hours = totalMilkshakes

theorem augustus_makes_3_milkshakes :
  augMilkshakePerHour 3 7 8 80 :=
by
  -- We assume the proof here
  sorry

end NUMINAMATH_GPT_augustus_makes_3_milkshakes_l1870_187098


namespace NUMINAMATH_GPT_smallest_n_for_simplest_form_l1870_187069

-- Definitions and conditions
def simplest_form_fractions (n : ℕ) :=
  ∀ k : ℕ, 7 ≤ k ∧ k ≤ 31 → Nat.gcd k (n + 2) = 1

-- Problem statement
theorem smallest_n_for_simplest_form :
  ∃ n : ℕ, simplest_form_fractions (n) ∧ ∀ m : ℕ, m < n → ¬ simplest_form_fractions (m) := 
by 
  sorry

end NUMINAMATH_GPT_smallest_n_for_simplest_form_l1870_187069


namespace NUMINAMATH_GPT_michael_lost_at_least_800_l1870_187021

theorem michael_lost_at_least_800 
  (T F : ℕ) 
  (h1 : T + F = 15) 
  (h2 : T = F + 1 ∨ T = F - 1) 
  (h3 : 10 * T + 50 * F = 1270) : 
  1270 - (10 * T + 50 * F) = 800 :=
by
  sorry

end NUMINAMATH_GPT_michael_lost_at_least_800_l1870_187021


namespace NUMINAMATH_GPT_num_div_divided_by_10_l1870_187051

-- Given condition: the number divided by 10 equals 12
def number_divided_by_10_gives_12 (x : ℝ) : Prop :=
  x / 10 = 12

-- Lean statement for the mathematical problem
theorem num_div_divided_by_10 (x : ℝ) (h : number_divided_by_10_gives_12 x) : x = 120 :=
by
  sorry

end NUMINAMATH_GPT_num_div_divided_by_10_l1870_187051


namespace NUMINAMATH_GPT_stratified_sampling_elderly_l1870_187080

theorem stratified_sampling_elderly (total_elderly middle_aged young total_sample total_population elderly_to_sample : ℕ) 
  (h1: total_elderly = 30) 
  (h2: middle_aged = 90) 
  (h3: young = 60) 
  (h4: total_sample = 36) 
  (h5: total_population = total_elderly + middle_aged + young) 
  (h6: 1 / 5 * total_elderly = elderly_to_sample)
  : elderly_to_sample = 6 := 
  by 
    sorry

end NUMINAMATH_GPT_stratified_sampling_elderly_l1870_187080


namespace NUMINAMATH_GPT_not_right_triangle_D_l1870_187076

theorem not_right_triangle_D : 
  ¬ (Real.sqrt 3)^2 + 4^2 = 5^2 ∧
  (1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2) ∧
  (7^2 + 24^2 = 25^2) ∧
  (5^2 + 12^2 = 13^2) := 
by 
  have hA : 1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2 := by norm_num
  have hB : 7^2 + 24^2 = 25^2 := by norm_num
  have hC : 5^2 + 12^2 = 13^2 := by norm_num
  have hD : ¬ (Real.sqrt 3)^2 + 4^2 = 5^2 := by norm_num
  exact ⟨hD, hA, hB, hC⟩

#print axioms not_right_triangle_D

end NUMINAMATH_GPT_not_right_triangle_D_l1870_187076


namespace NUMINAMATH_GPT_points_on_ellipse_l1870_187038

theorem points_on_ellipse (u : ℝ) :
  let x := (Real.cos u + Real.sin u)
  let y := (4 * (Real.cos u - Real.sin u))
  (x^2 / 2 + y^2 / 32 = 1) :=
by
  let x := (Real.cos u + Real.sin u)
  let y := (4 * (Real.cos u - Real.sin u))
  sorry

end NUMINAMATH_GPT_points_on_ellipse_l1870_187038


namespace NUMINAMATH_GPT_min_e1_plus_2e2_l1870_187011

noncomputable def e₁ (r : ℝ) : ℝ := 2 / (4 - r)
noncomputable def e₂ (r : ℝ) : ℝ := 2 / (4 + r)

theorem min_e1_plus_2e2 (r : ℝ) (h₀ : 0 < r) (h₂ : r < 2) :
  e₁ r + 2 * e₂ r = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end NUMINAMATH_GPT_min_e1_plus_2e2_l1870_187011


namespace NUMINAMATH_GPT_min_value_fraction_l1870_187092

theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : (1 / m + 2 / n) ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l1870_187092


namespace NUMINAMATH_GPT_part1_correct_part2_correct_part3_correct_l1870_187002

-- Example survival rates data (provided conditions)
def survivalRatesA : List (Option Float) := [some 95.5, some 92, some 96.5, some 91.6, some 96.3, some 94.6, none, none, none, none]
def survivalRatesB : List (Option Float) := [some 95.1, some 91.6, some 93.2, some 97.8, some 95.6, some 92.3, some 96.6, none, none, none]
def survivalRatesC : List (Option Float) := [some 97, some 95.4, some 98.2, some 93.5, some 94.8, some 95.5, some 94.5, some 93.5, some 98, some 92.5]

-- Define high-quality project condition
def isHighQuality (rate : Float) : Bool := rate > 95.0

-- Problem 1: Probability of two high-quality years from farm B
noncomputable def probabilityTwoHighQualityB : Float := (4.0 * 3.0) / (7.0 * 6.0)

-- Problem 2: Distribution of high-quality projects from farms A, B, and C
structure DistributionX := 
(P0 : Float) -- probability of 0 high-quality years
(P1 : Float) -- probability of 1 high-quality year
(P2 : Float) -- probability of 2 high-quality years
(P3 : Float) -- probability of 3 high-quality years

noncomputable def distributionX : DistributionX := 
{ P0 := 3.0 / 28.0,
  P1 := 5.0 / 14.0,
  P2 := 11.0 / 28.0,
  P3 := 1.0 / 7.0 
}

-- Problem 3: Inference of average survival rate from high-quality project probabilities
structure AverageSurvivalRates := 
(avgB : Float) 
(avgC : Float)
(probHighQualityB : Float)
(probHighQualityC : Float)
(canInfer : Bool)

noncomputable def avgSurvivalRates : AverageSurvivalRates := 
{ avgB := (95.1 + 91.6 + 93.2 + 97.8 + 95.6 + 92.3 + 96.6) / 7.0,
  avgC := (97 + 95.4 + 98.2 + 93.5 + 94.8 + 95.5 + 94.5 + 93.5 + 98 + 92.5) / 10.0,
  probHighQualityB := 4.0 / 7.0,
  probHighQualityC := 5.0 / 10.0,
  canInfer := false
}

-- Definitions for proof statements indicating correctness
theorem part1_correct : probabilityTwoHighQualityB = (2.0 / 7.0) := sorry

theorem part2_correct : distributionX = 
{ P0 := 3.0 / 28.0,
  P1 := 5.0 / 14.0,
  P2 := 11.0 / 28.0,
  P3 := 1.0 / 7.0 
} := sorry

theorem part3_correct : avgSurvivalRates.canInfer = false := sorry

end NUMINAMATH_GPT_part1_correct_part2_correct_part3_correct_l1870_187002


namespace NUMINAMATH_GPT_find_f_2023_l1870_187026

noncomputable def f : ℤ → ℤ := sorry

theorem find_f_2023 (h1 : ∀ x : ℤ, f (x+2) + f x = 3) (h2 : f 1 = 0) : f 2023 = 3 := sorry

end NUMINAMATH_GPT_find_f_2023_l1870_187026


namespace NUMINAMATH_GPT_triple_layer_area_l1870_187050

theorem triple_layer_area (A B C X Y : ℕ) 
  (h1 : A + B + C = 204) 
  (h2 : 140 = (A + B + C) - X - 2 * Y + X + Y)
  (h3 : X = 24) : 
  Y = 64 := by
  sorry

end NUMINAMATH_GPT_triple_layer_area_l1870_187050


namespace NUMINAMATH_GPT_remainder_27_pow_482_div_13_l1870_187062

theorem remainder_27_pow_482_div_13 :
  27^482 % 13 = 1 :=
sorry

end NUMINAMATH_GPT_remainder_27_pow_482_div_13_l1870_187062


namespace NUMINAMATH_GPT_cadence_old_company_salary_l1870_187025

variable (S : ℝ)

def oldCompanyMonths : ℝ := 36
def newCompanyMonths : ℝ := 41
def newSalaryMultiplier : ℝ := 1.20
def totalEarnings : ℝ := 426000

theorem cadence_old_company_salary :
  (oldCompanyMonths * S) + (newCompanyMonths * newSalaryMultiplier * S) = totalEarnings → 
  S = 5000 :=
by
  sorry

end NUMINAMATH_GPT_cadence_old_company_salary_l1870_187025


namespace NUMINAMATH_GPT_endpoint_of_vector_a_l1870_187082

theorem endpoint_of_vector_a (x y : ℝ) (h : (x - 3) / -3 = (y + 1) / 4) : 
    x = 13 / 5 ∧ y = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_endpoint_of_vector_a_l1870_187082


namespace NUMINAMATH_GPT_calc_101_cubed_expression_l1870_187030

theorem calc_101_cubed_expression : 101^3 + 3 * (101^2) - 3 * 101 + 9 = 1060610 := 
by
  sorry

end NUMINAMATH_GPT_calc_101_cubed_expression_l1870_187030


namespace NUMINAMATH_GPT_total_fencing_cost_is_correct_l1870_187019

-- Defining the lengths of each side
def length1 : ℝ := 50
def length2 : ℝ := 75
def length3 : ℝ := 60
def length4 : ℝ := 80
def length5 : ℝ := 65

-- Defining the cost per unit length for each side
def cost_per_meter1 : ℝ := 2
def cost_per_meter2 : ℝ := 3
def cost_per_meter3 : ℝ := 4
def cost_per_meter4 : ℝ := 3.5
def cost_per_meter5 : ℝ := 5

-- Calculating the total cost for each side
def cost1 : ℝ := length1 * cost_per_meter1
def cost2 : ℝ := length2 * cost_per_meter2
def cost3 : ℝ := length3 * cost_per_meter3
def cost4 : ℝ := length4 * cost_per_meter4
def cost5 : ℝ := length5 * cost_per_meter5

-- Summing up the total cost for all sides
def total_cost : ℝ := cost1 + cost2 + cost3 + cost4 + cost5

-- The theorem to be proven
theorem total_fencing_cost_is_correct : total_cost = 1170 := by
  sorry

end NUMINAMATH_GPT_total_fencing_cost_is_correct_l1870_187019


namespace NUMINAMATH_GPT_emily_small_gardens_l1870_187059

theorem emily_small_gardens 
  (total_seeds : Nat)
  (big_garden_seeds : Nat)
  (small_garden_seeds : Nat)
  (remaining_seeds : total_seeds = big_garden_seeds + (small_garden_seeds * 3)) :
  3 = (total_seeds - big_garden_seeds) / small_garden_seeds :=
by
  have h1 : total_seeds = 42 := by sorry
  have h2 : big_garden_seeds = 36 := by sorry
  have h3 : small_garden_seeds = 2 := by sorry
  have h4 : 6 = total_seeds - big_garden_seeds := by sorry
  have h5 : 3 = 6 / small_garden_seeds := by sorry
  sorry

end NUMINAMATH_GPT_emily_small_gardens_l1870_187059


namespace NUMINAMATH_GPT_convex_n_hedral_angle_l1870_187020

theorem convex_n_hedral_angle (n : ℕ) 
  (sum_plane_angles : ℝ) (sum_dihedral_angles : ℝ) 
  (h1 : sum_plane_angles = sum_dihedral_angles)
  (h2 : sum_plane_angles < 2 * Real.pi)
  (h3 : sum_dihedral_angles > (n - 2) * Real.pi) :
  n = 3 := 
by 
  sorry

end NUMINAMATH_GPT_convex_n_hedral_angle_l1870_187020


namespace NUMINAMATH_GPT_blocks_per_tree_l1870_187063

def trees_per_day : ℕ := 2
def blocks_after_5_days : ℕ := 30
def days : ℕ := 5

theorem blocks_per_tree : (blocks_after_5_days / (trees_per_day * days)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_blocks_per_tree_l1870_187063


namespace NUMINAMATH_GPT_find_positive_integer_n_l1870_187049

noncomputable def is_largest_prime_divisor (p n : ℕ) : Prop :=
  (∃ k, n = p * k) ∧ ∀ q, Prime q ∧ q ∣ n → q ≤ p

noncomputable def is_least_prime_divisor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n ∧ ∀ q, Prime q ∧ q ∣ n → p ≤ q

theorem find_positive_integer_n :
  ∃ n : ℕ, n > 0 ∧ 
    (∃ p, is_largest_prime_divisor p (n^2 + 3) ∧ is_least_prime_divisor p (n^4 + 6)) ∧
    ∀ m : ℕ, m > 0 ∧ 
      (∃ q, is_largest_prime_divisor q (m^2 + 3) ∧ is_least_prime_divisor q (m^4 + 6)) → m = 3 :=
by sorry

end NUMINAMATH_GPT_find_positive_integer_n_l1870_187049


namespace NUMINAMATH_GPT_num_different_configurations_of_lights_l1870_187079

-- Definition of initial conditions
def num_rows : Nat := 6
def num_columns : Nat := 6
def possible_switch_states (n : Nat) : Nat := 2^n

-- Problem statement to be verified
theorem num_different_configurations_of_lights :
  let num_configurations := (possible_switch_states num_rows - 1) * (possible_switch_states num_columns - 1) + 1
  num_configurations = 3970 :=
by
  sorry

end NUMINAMATH_GPT_num_different_configurations_of_lights_l1870_187079


namespace NUMINAMATH_GPT_field_ratio_l1870_187077

theorem field_ratio
  (l w : ℕ)
  (pond_length : ℕ)
  (pond_area_ratio : ℚ)
  (field_length : ℕ)
  (field_area : ℕ)
  (hl : l = 24)
  (hp : pond_length = 6)
  (hr : pond_area_ratio = 1 / 8)
  (hm : l % w = 0)
  (ha : field_area = 36 * 8)
  (hf : l * w = field_area) :
  l / w = 2 :=
by
  sorry

end NUMINAMATH_GPT_field_ratio_l1870_187077


namespace NUMINAMATH_GPT_hall_length_l1870_187000

theorem hall_length (L : ℝ) (H : ℝ) 
  (h1 : 2 * (L * 15) = 2 * (L * H) + 2 * (15 * H)) 
  (h2 : L * 15 * H = 1687.5) : 
  L = 15 :=
by 
  sorry

end NUMINAMATH_GPT_hall_length_l1870_187000


namespace NUMINAMATH_GPT_sign_of_x_and_y_l1870_187073

theorem sign_of_x_and_y (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ 0) : x > 0 ∧ y > 0 :=
sorry

end NUMINAMATH_GPT_sign_of_x_and_y_l1870_187073


namespace NUMINAMATH_GPT_factorization_identity_l1870_187075

theorem factorization_identity (x : ℝ) : 
  3 * x^2 + 12 * x + 12 = 3 * (x + 2)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_identity_l1870_187075


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1870_187028

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_start : a 1 < 0)
  (h_increasing : ∀ n, a n < a (n + 1)) : 0 < q ∧ q < 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1870_187028


namespace NUMINAMATH_GPT_bus_speed_express_mode_l1870_187071

theorem bus_speed_express_mode (L : ℝ) (t_red : ℝ) (speed_increase : ℝ) (x : ℝ) (normal_speed : ℝ) :
  L = 16 ∧ t_red = 1 / 15 ∧ speed_increase = 8 ∧ normal_speed = x - 8 ∧ 
  (16 / normal_speed - 16 / x = 1 / 15) → x = 48 :=
by
  sorry

end NUMINAMATH_GPT_bus_speed_express_mode_l1870_187071


namespace NUMINAMATH_GPT_percentage_hate_german_l1870_187035

def percentage_hate_math : ℝ := 0.01
def percentage_hate_english : ℝ := 0.02
def percentage_hate_french : ℝ := 0.01
def percentage_hate_all_four : ℝ := 0.08

theorem percentage_hate_german : (0.08 - (0.01 + 0.02 + 0.01)) = 0.04 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_hate_german_l1870_187035


namespace NUMINAMATH_GPT_games_given_away_correct_l1870_187055

-- Define initial and remaining games
def initial_games : ℕ := 50
def remaining_games : ℕ := 35

-- Define the number of games given away
def games_given_away : ℕ := initial_games - remaining_games

-- Prove that the number of games given away is 15
theorem games_given_away_correct : games_given_away = 15 := by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_games_given_away_correct_l1870_187055


namespace NUMINAMATH_GPT_Carla_pays_more_than_Bob_l1870_187053

theorem Carla_pays_more_than_Bob
  (slices : ℕ := 12)
  (veg_slices : ℕ := slices / 2)
  (non_veg_slices : ℕ := slices / 2)
  (base_cost : ℝ := 10)
  (extra_cost : ℝ := 3)
  (total_cost : ℝ := base_cost + extra_cost)
  (per_slice_cost : ℝ := total_cost / slices)
  (carla_slices : ℕ := veg_slices + 2)
  (bob_slices : ℕ := 3)
  (carla_payment : ℝ := carla_slices * per_slice_cost)
  (bob_payment : ℝ := bob_slices * per_slice_cost) :
  (carla_payment - bob_payment) = 5.41665 :=
sorry

end NUMINAMATH_GPT_Carla_pays_more_than_Bob_l1870_187053


namespace NUMINAMATH_GPT_total_blocks_fell_l1870_187042

-- Definitions based on the conditions
def first_stack_height := 7
def second_stack_height := first_stack_height + 5
def third_stack_height := second_stack_height + 7

def first_stack_fallen_blocks := first_stack_height  -- All blocks fell down
def second_stack_fallen_blocks := second_stack_height - 2  -- 2 blocks left standing
def third_stack_fallen_blocks := third_stack_height - 3  -- 3 blocks left standing

-- Total fallen blocks
def total_fallen_blocks := first_stack_fallen_blocks + second_stack_fallen_blocks + third_stack_fallen_blocks

-- Theorem to prove the total number of fallen blocks
theorem total_blocks_fell : total_fallen_blocks = 33 :=
by
  -- Proof omitted, statement given as required
  sorry

end NUMINAMATH_GPT_total_blocks_fell_l1870_187042


namespace NUMINAMATH_GPT_prime_cubic_solution_l1870_187094

theorem prime_cubic_solution :
  ∃ p1 p2 : ℕ, (Nat.Prime p1 ∧ Nat.Prime p2) ∧ p1 ≠ p2 ∧
  (p1^3 + p1^2 - 18*p1 + 26 = 0) ∧ (p2^3 + p2^2 - 18*p2 + 26 = 0) :=
by
  sorry

end NUMINAMATH_GPT_prime_cubic_solution_l1870_187094


namespace NUMINAMATH_GPT_cos_double_angle_sum_l1870_187095

theorem cos_double_angle_sum (α : ℝ) (hα : 0 < α ∧ α < π / 2) 
  (h : Real.sin (α + π/6) = 3/5) : 
  Real.cos (2*α + π/12) = 31 / 50 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_cos_double_angle_sum_l1870_187095


namespace NUMINAMATH_GPT_polynomial_j_value_l1870_187043

noncomputable def polynomial_roots_in_ap (a d : ℝ) : Prop :=
  let r1 := a
  let r2 := a + d
  let r3 := a + 2 * d
  let r4 := a + 3 * d
  ∀ (r : ℝ), r = r1 ∨ r = r2 ∨ r = r3 ∨ r = r4

theorem polynomial_j_value (a d : ℝ) (h_ap : polynomial_roots_in_ap a d)
  (h_poly : ∀ (x : ℝ), (x - (a)) * (x - (a + d)) * (x - (a + 2 * d)) * (x - (a + 3 * d)) = x^4 + j * x^2 + k * x + 256) :
  j = -80 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_j_value_l1870_187043


namespace NUMINAMATH_GPT_avg_decreased_by_one_l1870_187013

noncomputable def avg_decrease (n : ℕ) (average_initial : ℝ) (obs_new : ℝ) : ℝ :=
  (n * average_initial + obs_new) / (n + 1)

theorem avg_decreased_by_one (init_avg : ℝ) (obs_new : ℝ) (num_obs : ℕ)
  (h₁ : num_obs = 6)
  (h₂ : init_avg = 12)
  (h₃ : obs_new = 5) :
  init_avg - avg_decrease num_obs init_avg obs_new = 1 :=
by
  sorry

end NUMINAMATH_GPT_avg_decreased_by_one_l1870_187013


namespace NUMINAMATH_GPT_equation_represents_3x_minus_7_equals_2x_plus_5_l1870_187070

theorem equation_represents_3x_minus_7_equals_2x_plus_5 (x : ℝ) :
  (3 * x - 7 = 2 * x + 5) :=
sorry

end NUMINAMATH_GPT_equation_represents_3x_minus_7_equals_2x_plus_5_l1870_187070


namespace NUMINAMATH_GPT_social_event_handshakes_l1870_187084

def handshake_count (total_people : ℕ) (group_a : ℕ) (group_b : ℕ) : ℕ :=
  let introductions_handshakes := group_b * (group_b - 1) / 2
  let direct_handshakes := group_b * (group_a - 1)
  introductions_handshakes + direct_handshakes

theorem social_event_handshakes :
  handshake_count 40 25 15 = 465 := by
  sorry

end NUMINAMATH_GPT_social_event_handshakes_l1870_187084


namespace NUMINAMATH_GPT_correct_average_mark_l1870_187097

theorem correct_average_mark (
  num_students : ℕ := 50)
  (incorrect_avg : ℚ := 85.4)
  (wrong_mark_A : ℚ := 73.6) (correct_mark_A : ℚ := 63.5)
  (wrong_mark_B : ℚ := 92.4) (correct_mark_B : ℚ := 96.7)
  (wrong_mark_C : ℚ := 55.3) (correct_mark_C : ℚ := 51.8) :
  (incorrect_avg*num_students + 
   (correct_mark_A - wrong_mark_A) + 
   (correct_mark_B - wrong_mark_B) + 
   (correct_mark_C - wrong_mark_C)) / 
   num_students = 85.214 :=
sorry

end NUMINAMATH_GPT_correct_average_mark_l1870_187097


namespace NUMINAMATH_GPT_evaporation_period_days_l1870_187057

theorem evaporation_period_days
    (initial_water : ℝ)
    (daily_evaporation : ℝ)
    (evaporation_percentage : ℝ)
    (total_evaporated_water : ℝ)
    (number_of_days : ℝ) :
    initial_water = 10 ∧
    daily_evaporation = 0.06 ∧
    evaporation_percentage = 0.12 ∧
    total_evaporated_water = initial_water * evaporation_percentage ∧
    number_of_days = total_evaporated_water / daily_evaporation →
    number_of_days = 20 :=
by
  sorry

end NUMINAMATH_GPT_evaporation_period_days_l1870_187057


namespace NUMINAMATH_GPT_range_g_l1870_187044

def f (x: ℝ) : ℝ := 4 * x - 3
def g (x: ℝ) : ℝ := f (f (f (f (f x))))

theorem range_g (x: ℝ) (h: 0 ≤ x ∧ x ≤ 3) : -1023 ≤ g x ∧ g x ≤ 2049 :=
by
  sorry

end NUMINAMATH_GPT_range_g_l1870_187044


namespace NUMINAMATH_GPT_cartesian_to_polar_coords_l1870_187089

theorem cartesian_to_polar_coords :
  ∃ ρ θ : ℝ, 
  (ρ = 2) ∧ (θ = 2 * Real.pi / 3) ∧ 
  (-1, Real.sqrt 3) = (ρ * Real.cos θ, ρ * Real.sin θ) :=
sorry

end NUMINAMATH_GPT_cartesian_to_polar_coords_l1870_187089


namespace NUMINAMATH_GPT_intersection_eq_l1870_187064

open Set

-- Define the sets M and N
def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {-3, -1, 1, 3, 5}

-- The goal is to prove that M ∩ N = {-1, 1, 3}
theorem intersection_eq : M ∩ N = {-1, 1, 3} :=
  sorry

end NUMINAMATH_GPT_intersection_eq_l1870_187064


namespace NUMINAMATH_GPT_ratio_of_ages_l1870_187052

variable (x : Nat) -- The multiple of Marie's age
variable (marco_age marie_age : Nat) -- Marco's and Marie's ages

-- Conditions from (a)
axiom h1 : marie_age = 12
axiom h2 : marco_age = (12 * x) + 1
axiom h3 : marco_age + marie_age = 37

-- Statement to be proved
theorem ratio_of_ages : (marco_age : Nat) / (marie_age : Nat) = (25 / 12) :=
by
  -- Proof steps here
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l1870_187052


namespace NUMINAMATH_GPT_calc_expression_eq_3_solve_quadratic_eq_l1870_187005

-- Problem 1
theorem calc_expression_eq_3 :
  (-1 : ℝ) ^ 2020 + (- (1 / 2)⁻¹) - (3.14 - Real.pi) ^ 0 + abs (-3) = 3 :=
by
  sorry

-- Problem 2
theorem solve_quadratic_eq {x : ℝ} :
  (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_eq_3_solve_quadratic_eq_l1870_187005


namespace NUMINAMATH_GPT_sequence_eventually_congruent_mod_l1870_187047

theorem sequence_eventually_congruent_mod (n : ℕ) (hn : n ≥ 1) : 
  ∃ N, ∀ m ≥ N, ∃ k, m = k * n + N ∧ (2^N.succ - 2^k) % n = 0 :=
by
  sorry

end NUMINAMATH_GPT_sequence_eventually_congruent_mod_l1870_187047


namespace NUMINAMATH_GPT_intersection_points_on_ellipse_l1870_187023

theorem intersection_points_on_ellipse (s x y : ℝ)
  (h_line1 : s * x - 3 * y - 4 * s = 0)
  (h_line2 : x - 3 * s * y + 4 = 0) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_on_ellipse_l1870_187023


namespace NUMINAMATH_GPT_original_cylinder_weight_is_24_l1870_187045

noncomputable def weight_of_original_cylinder (cylinder_weight cone_weight : ℝ) : Prop :=
  cylinder_weight = 3 * cone_weight

-- Given conditions in Lean 4
variables (cone_weight : ℝ) (h_cone_weight : cone_weight = 8)

-- Proof problem statement
theorem original_cylinder_weight_is_24 :
  weight_of_original_cylinder 24 cone_weight :=
by
  sorry

end NUMINAMATH_GPT_original_cylinder_weight_is_24_l1870_187045


namespace NUMINAMATH_GPT_cost_of_pears_l1870_187014

theorem cost_of_pears 
  (initial_amount : ℕ := 55) 
  (left_amount : ℕ := 28) 
  (banana_count : ℕ := 2) 
  (banana_price : ℕ := 4) 
  (asparagus_price : ℕ := 6) 
  (chicken_price : ℕ := 11) 
  (total_spent : ℕ := 27) :
  initial_amount - left_amount - (banana_count * banana_price + asparagus_price + chicken_price) = 2 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_pears_l1870_187014


namespace NUMINAMATH_GPT_intersection_of_sets_l1870_187008

variable {x : ℝ}

def SetA : Set ℝ := {x | x + 1 > 0}
def SetB : Set ℝ := {x | x - 3 < 0}

theorem intersection_of_sets : SetA ∩ SetB = {x | -1 < x ∧ x < 3} :=
by sorry

end NUMINAMATH_GPT_intersection_of_sets_l1870_187008


namespace NUMINAMATH_GPT_solve_for_x_l1870_187072

variable (a b c x y z : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem solve_for_x (h1 : (x * y) / (x + y) = a)
                   (h2 : (x * z) / (x + z) = b)
                   (h3 : (y * z) / (y + z) = c) :
                   x = (2 * a * b * c) / (a * c + b * c - a * b) :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1870_187072


namespace NUMINAMATH_GPT_extremum_point_iff_nonnegative_condition_l1870_187068

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 + x) - (a * x) / (x + 1)

theorem extremum_point_iff (a : ℝ) (h : 0 < a) :
  (∃ (x : ℝ), x = 1 ∧ ∀ (f' : ℝ), f' = (1 + x - a) / (x + 1)^2 ∧ f' = 0) ↔ a = 2 :=
by
  sorry

theorem nonnegative_condition (a : ℝ) (h0 : 0 < a) :
  (∀ (x : ℝ), x ∈ Set.Ici 0 → f a x ≥ 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_extremum_point_iff_nonnegative_condition_l1870_187068


namespace NUMINAMATH_GPT_orthocentric_tetrahedron_equivalence_l1870_187067

def isOrthocentricTetrahedron 
  (sums_of_squares_of_opposite_edges_equal : Prop) 
  (products_of_cosines_of_opposite_dihedral_angles_equal : Prop)
  (angles_between_opposite_edges_equal : Prop) : Prop :=
  sums_of_squares_of_opposite_edges_equal ∨
  products_of_cosines_of_opposite_dihedral_angles_equal ∨
  angles_between_opposite_edges_equal

theorem orthocentric_tetrahedron_equivalence
  (sums_of_squares_of_opposite_edges_equal 
    products_of_cosines_of_opposite_dihedral_angles_equal
    angles_between_opposite_edges_equal : Prop) :
  isOrthocentricTetrahedron
    sums_of_squares_of_opposite_edges_equal
    products_of_cosines_of_opposite_dihedral_angles_equal
    angles_between_opposite_edges_equal :=
sorry

end NUMINAMATH_GPT_orthocentric_tetrahedron_equivalence_l1870_187067


namespace NUMINAMATH_GPT_place_b_left_of_a_forms_correct_number_l1870_187029

noncomputable def form_three_digit_number (a b : ℕ) : ℕ :=
  100 * b + a

theorem place_b_left_of_a_forms_correct_number (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 1 ≤ b ∧ b < 10) :
  form_three_digit_number a b = 100 * b + a :=
by sorry

end NUMINAMATH_GPT_place_b_left_of_a_forms_correct_number_l1870_187029


namespace NUMINAMATH_GPT_ab_value_l1870_187007

theorem ab_value (a b : ℝ) (h₁ : a - b = 3) (h₂ : a^2 + b^2 = 33) : a * b = 18 := 
by
  sorry

end NUMINAMATH_GPT_ab_value_l1870_187007


namespace NUMINAMATH_GPT_find_a1_and_d_l1870_187037

variable (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) (a5 : ℤ := -1) (a8 : ℤ := 2)

def arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

theorem find_a1_and_d
  (h : arithmetic_sequence a d)
  (h_a5 : a 5 = -1)
  (h_a8 : a 8 = 2) :
  a 1 = -5 ∧ d = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_and_d_l1870_187037


namespace NUMINAMATH_GPT_minimum_value_of_4a_plus_b_l1870_187009

noncomputable def minimum_value (a b : ℝ) :=
  if a > 0 ∧ b > 0 ∧ a^2 + a*b - 3 = 0 then 4*a + b else 0

theorem minimum_value_of_4a_plus_b :
  ∀ (a b : ℝ), a > 0 → b > 0 → a^2 + a*b - 3 = 0 → 4*a + b ≥ 6 :=
by
  intros a b ha hb hab
  sorry

end NUMINAMATH_GPT_minimum_value_of_4a_plus_b_l1870_187009


namespace NUMINAMATH_GPT_find_smaller_number_l1870_187085

theorem find_smaller_number (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) : y = 18.5 :=
by
  -- Proof steps will be filled in here
  sorry

end NUMINAMATH_GPT_find_smaller_number_l1870_187085


namespace NUMINAMATH_GPT_cos_triangle_inequality_l1870_187016

theorem cos_triangle_inequality (α β γ : ℝ) (h_sum : α + β + γ = Real.pi) 
    (h_α : 0 < α) (h_β : 0 < β) (h_γ : 0 < γ) (h_α_lt : α < Real.pi) (h_β_lt : β < Real.pi) (h_γ_lt : γ < Real.pi) : 
    (Real.cos α * Real.cos β + Real.cos β * Real.cos γ + Real.cos γ * Real.cos α) ≤ 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cos_triangle_inequality_l1870_187016


namespace NUMINAMATH_GPT_certain_number_is_120_l1870_187036

theorem certain_number_is_120 : ∃ certain_number : ℤ, 346 * certain_number = 173 * 240 ∧ certain_number = 120 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_120_l1870_187036


namespace NUMINAMATH_GPT_vector_decomposition_unique_l1870_187099

variable {m : ℝ}
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (m - 1, m + 3)

theorem vector_decomposition_unique (m : ℝ) : (m + 3 ≠ 2 * (m - 1)) ↔ (m ≠ 5) := 
sorry

end NUMINAMATH_GPT_vector_decomposition_unique_l1870_187099


namespace NUMINAMATH_GPT_yellow_balls_in_bag_l1870_187090

theorem yellow_balls_in_bag (r y : ℕ) (P : ℚ) 
  (h1 : r = 10) 
  (h2 : P = 2 / 7) 
  (h3 : P = r / (r + y)) : 
  y = 25 := 
sorry

end NUMINAMATH_GPT_yellow_balls_in_bag_l1870_187090


namespace NUMINAMATH_GPT_smallest_positive_integer_l1870_187022

theorem smallest_positive_integer (
    b : ℤ 
) : 
    (b % 4 = 1) → (b % 5 = 2) → (b % 6 = 3) → b = 21 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1870_187022


namespace NUMINAMATH_GPT_possible_degrees_of_remainder_l1870_187039

theorem possible_degrees_of_remainder (p : Polynomial ℝ) :
  ∃ r q : Polynomial ℝ, p = q * (3 * X^3 - 4 * X^2 + 5 * X - 6) + r ∧ r.degree < 3 :=
sorry

end NUMINAMATH_GPT_possible_degrees_of_remainder_l1870_187039


namespace NUMINAMATH_GPT_sandy_total_money_l1870_187032

def half_dollar_value := 0.5
def quarter_value := 0.25
def dime_value := 0.1
def nickel_value := 0.05
def dollar_value := 1.0

def monday_total := 12 * half_dollar_value + 5 * quarter_value + 10 * dime_value
def tuesday_total := 8 * half_dollar_value + 15 * quarter_value + 5 * dime_value
def wednesday_total := 3 * dollar_value + 4 * half_dollar_value + 10 * quarter_value + 7 * nickel_value
def thursday_total := 5 * dollar_value + 6 * half_dollar_value + 8 * quarter_value + 5 * dime_value + 12 * nickel_value
def friday_total := 2 * dollar_value + 7 * half_dollar_value + 20 * nickel_value + 25 * dime_value

def total_amount := monday_total + tuesday_total + wednesday_total + thursday_total + friday_total

theorem sandy_total_money : total_amount = 44.45 := by
  sorry

end NUMINAMATH_GPT_sandy_total_money_l1870_187032


namespace NUMINAMATH_GPT_problem_solution_l1870_187046

theorem problem_solution
  (a b c d : ℕ)
  (h1 : a^6 = b^5)
  (h2 : c^4 = d^3)
  (h3 : c - a = 25) :
  d - b = 561 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1870_187046
