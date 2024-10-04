import Complex
import Mathlib
import Mathlib.Algebra.Field
import Mathlib.Algebra.LinearEquations
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Algebra.Polynomial.Rewrite
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.FDeriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.SpecialFunctions.Ceil
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Planar
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Gcd.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Triangle.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.Probability
import Mathlib.Probability.Distribution.Poisson
import Mathlib.Probability.Independence
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Limits
import Real

namespace math_scores_between_70_and_80_l510_510841

-- Declare the conditions of the problem
def num_students : ℕ := 3000
def normal_distribution := true  -- This is a placeholder for normal distribution properties.

-- Problem statement
theorem math_scores_between_70_and_80 
  (student_count : ℕ = num_students)
  (distribution : normal_distribution)
  : ∃ num_students_between_70_80 : ℕ, num_students_between_70_80 = 408
  :=
  sorry

end math_scores_between_70_and_80_l510_510841


namespace convex_quadrilateral_parallelogram_l510_510971

theorem convex_quadrilateral_parallelogram
  (A B C D : Point)
  (acute_angle : Angle)
  (h_convex : ConvexQuadrilateralInscribed A B C D acute_angle)
  (h_dist_sum_eq_line1 : ∀ (m : Line), 
    sum_of_distances_to_line A C m = sum_of_distances_to_line B D m)
  (h_dist_sum_eq_line2 : ∀ (n : Line), 
    sum_of_distances_to_line A C n = sum_of_distances_to_line B D n) :
  Parallelogram A B C D := 
sorry -- proof will need to be provided

end convex_quadrilateral_parallelogram_l510_510971


namespace sum_of_complex_powers_l510_510680

theorem sum_of_complex_powers :
  (Complex.i + 2 * Complex.i ^ 2 + 3 * Complex.i ^ 3 + 4 * Complex.i ^ 4 + 5 * Complex.i ^ 5 +
  6 * Complex.i ^ 6 + 7 * Complex.i ^ 7 + 8 * Complex.i ^ 8 + 9 * Complex.i ^ 9) =
  (4 : ℂ) + 5 * Complex.i := 
by
  sorry

end sum_of_complex_powers_l510_510680


namespace compute_u2_plus_v2_l510_510101

theorem compute_u2_plus_v2 (u v : ℝ) (hu : 1 < u) (hv : 1 < v)
  (h : (Real.log u / Real.log 3)^4 + (Real.log v / Real.log 7)^4 = 10 * (Real.log u / Real.log 3) * (Real.log v / Real.log 7)) :
  u^2 + v^2 = 3^(Real.sqrt 5) + 7^(Real.sqrt 5) :=
by
  sorry

end compute_u2_plus_v2_l510_510101


namespace train_and_car_combined_time_l510_510285

noncomputable def combined_time (car_time : ℝ) (extra_time : ℝ) : ℝ :=
  car_time + (car_time + extra_time)

theorem train_and_car_combined_time : 
  ∀ (car_time : ℝ) (extra_time : ℝ), car_time = 4.5 → extra_time = 2.0 → combined_time car_time extra_time = 11 :=
by
  intros car_time extra_time hcar hextra
  sorry

end train_and_car_combined_time_l510_510285


namespace hendrix_class_students_l510_510216

theorem hendrix_class_students (initial_students new_students : ℕ) (transferred_fraction : ℚ) 
  (h1 : initial_students = 160) (h2 : new_students = 20) (h3 : transferred_fraction = 1/3) :
  let total_students := initial_students + new_students
  let transferred_students := total_students * transferred_fraction
  let remaining_students := total_students - transferred_students
  remaining_students = 120 :=
by
  let total_students := initial_students + new_students
  let transferred_students := total_students * transferred_fraction
  let remaining_students := total_students - transferred_students
  have h4 : total_students = 180, by sorry
  have h5 : transferred_students = 60, by sorry
  have h6 : remaining_students = 120, by sorry
  exact h6

end hendrix_class_students_l510_510216


namespace A_n_on_circle_C_l510_510742

noncomputable def C_center : ℝ × ℝ := (1/2, 0)
noncomputable def C_radius : ℝ := (sqrt 5) / 2

def sequence (n : ℕ) : (ℝ × ℝ) 
| 1 => (0, 1)
| (nat.succ n) => let (a_n, b_n) := sequence n in (1 + a_n / (a_n^2 + b_n^2), -b_n / (a_n^2 + b_n^2))

theorem A_n_on_circle_C (n : ℕ) (h : n ≥ 4) : 
  let A_n := sequence n
  let (a_n, b_n) := A_n in
  (a_n - C_center.1)^2 + b_n^2 = C_radius^2 :=
sorry

end A_n_on_circle_C_l510_510742


namespace triangle_side_length_l510_510047

theorem triangle_side_length {A B C : Type} [EuclideanGeometry A B C] :
  ∀ (AB BC CA : ℝ),
  (angle_ratio : angle A B C : angle B C A : angle C A B = 1 : 2 : 3) →
  (AB = 6) →
  BC = 3 :=
by
  sorry

end triangle_side_length_l510_510047


namespace Hendrix_class_end_of_year_students_l510_510211

def initial_students := 160
def new_students := 20
def fraction_transferred := 1 / 3

theorem Hendrix_class_end_of_year_students : 
  let total_students := initial_students + new_students in
  let transferred_students := fraction_transferred * total_students in
  let final_students := total_students - transferred_students in
  final_students = 120 :=
by
  sorry

end Hendrix_class_end_of_year_students_l510_510211


namespace train_speed_l510_510653

theorem train_speed (train_length bridge_length : ℕ) (time_seconds : ℕ) : 
  train_length = 240 →
  bridge_length = 150 →
  time_seconds = 20 →
  (train_length + bridge_length) / time_seconds * 3.6 = 70.2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- Providing proof steps would resolve the theorem
  sorry

end train_speed_l510_510653


namespace total_conversion_cost_correct_l510_510638

-- Definition of the side lengths in meters
def a : ℝ := 32
def b : ℝ := 68 
-- Convert the angle from degrees to radians for the sin function
def C : ℝ := Real.pi / 6  -- 30 degrees in radians

-- Cost per square meter in yuan
def cost_per_square_meter : ℝ := 50

-- Function to compute the area of the triangle using two sides and the included angle
def triangle_area (a b C : ℝ) : ℝ := (1 / 2) * a * b * Real.sin(C)

-- The total cost for converting the triangular plot of land into a green space
def total_cost (a b C cost_per_square_meter : ℝ) : ℝ := cost_per_square_meter * triangle_area(a, b, C)

theorem total_conversion_cost_correct :
  total_cost(a, b, C, cost_per_square_meter) = 54400 :=
by
  sorry

end total_conversion_cost_correct_l510_510638


namespace greatest_prime_factor_of_5_pow_3_plus_10_pow_4_l510_510579

theorem greatest_prime_factor_of_5_pow_3_plus_10_pow_4 :
  let n := 5^3 + 10^4 in
  let prime_factors := {p : ℕ | p.prime ∧ p ∣ n} in
  ∃ p, p ∈ prime_factors ∧ p = 5 ∧ ∀ q, q ∈ prime_factors → q ≤ 5 :=
by
  sorry

end greatest_prime_factor_of_5_pow_3_plus_10_pow_4_l510_510579


namespace general_term_sequence_l510_510116

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 1
  else if n = 1 then 1
  else Real.sqrt (sequence (n-2) * sequence (n)) - Real.sqrt (sequence (n-2) * sequence (n-1)) = 2 * sequence (n-1)

theorem general_term_sequence (n : ℕ) : 
  sequence n = ∏ k in (range n).map (λ x, (2 ^ (x + 1)) - 1)^2 :=
sorry

end general_term_sequence_l510_510116


namespace place_mat_length_l510_510643

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) (inner_touch : Bool)
  (h1 : r = 4)
  (h2 : n = 6)
  (h3 : w = 1)
  (h4 : inner_touch = true)
  : x = (3 * Real.sqrt 7 - Real.sqrt 3) / 2 :=
sorry

end place_mat_length_l510_510643


namespace punch_added_after_cousin_drinks_l510_510868

variable (x : ℝ)

def amount_of_punch_initial := 16
def cousin_drink_half (x : ℝ) := (amount_of_punch_initial - x) / 2
def mark_add_punch (x : ℝ) := cousin_drink_half x + x
def sally_drink (x : ℝ) := mark_add_punch x - 2
def final_addition := 12
def bowl_capacity := 16

theorem punch_added_after_cousin_drinks : x = 12 :=
by
  have h : sally_drink x + final_addition = bowl_capacity
  sorry

end punch_added_after_cousin_drinks_l510_510868


namespace isosceles_triangle_area_l510_510949

theorem isosceles_triangle_area 
  (P Q R : Type) 
  (hPQ : segment_length P Q 20) 
  (hQR : segment_length Q R 20) 
  (hPR : segment_length P R 36) : 
  area P Q R = 72 * Real.sqrt 19 := 
sorry

end isosceles_triangle_area_l510_510949


namespace train_and_car_combined_time_l510_510288

theorem train_and_car_combined_time (car_time : ℝ) (train_time : ℝ) 
  (h1 : train_time = car_time + 2) (h2 : car_time = 4.5) : 
  car_time + train_time = 11 := 
by 
  -- Proof goes here
  sorry

end train_and_car_combined_time_l510_510288


namespace remaining_money_l510_510324

-- Define the conditions
def num_pies : ℕ := 200
def price_per_pie : ℕ := 20
def fraction_for_ingredients : ℚ := 3 / 5

-- Define the total sales
def total_sales : ℕ := num_pies * price_per_pie

-- Define the cost for ingredients
def cost_for_ingredients : ℚ := fraction_for_ingredients * total_sales 

-- Prove the remaining money
theorem remaining_money : (total_sales : ℚ) - cost_for_ingredients = 1600 := 
by {
  -- This is where the proof would go
  sorry
}

end remaining_money_l510_510324


namespace monotonic_decreasing_interval_l510_510546

noncomputable def f (x : ℝ) : ℝ := √3 * x + 2 * Real.cos x

theorem monotonic_decreasing_interval : 
  ∃ a b, a = π / 3 ∧ b = 2 * π / 3 ∧ ∀ x, a < x ∧ x < b → f'(x) < 0 := 
by
  sorry

end monotonic_decreasing_interval_l510_510546


namespace total_penalty_kicks_l510_510528

theorem total_penalty_kicks (num_players : ℕ) (num_goalies : ℕ) (h1 : num_players = 26) (h2 : num_goalies = 4) : 
  (num_goalies * (num_players - 1)) = 100 :=
by
  -- given conditions
  have h3 : (num_players - 1) = 25, from sorry
  show (num_goalies * (num_players - 1)) = 100, from sorry

end total_penalty_kicks_l510_510528


namespace Hendrix_class_end_of_year_students_l510_510213

def initial_students := 160
def new_students := 20
def fraction_transferred := 1 / 3

theorem Hendrix_class_end_of_year_students : 
  let total_students := initial_students + new_students in
  let transferred_students := fraction_transferred * total_students in
  let final_students := total_students - transferred_students in
  final_students = 120 :=
by
  sorry

end Hendrix_class_end_of_year_students_l510_510213


namespace maximum_n_for_positive_S_l510_510897

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
n * (a 1 + a n) / 2

theorem maximum_n_for_positive_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (S : ℕ → ℝ)
  (a1_pos : a 1 > 0)
  (d_neg : d < 0)
  (S4_eq_S8 : S 4 = S 8)
  (h1 : is_arithmetic_sequence a d)
  (h2 : ∀ n, S n = sum_of_first_n_terms a n) :
  ∃ n, ∀ m, m ≤ n → S m > 0 ∧ ∀ k, k > n → S k ≤ 0 ∧ n = 11 :=
sorry

end maximum_n_for_positive_S_l510_510897


namespace joan_missed_games_l510_510082

def total_games : ℕ := 864
def night_games_fraction : ℝ := 0.30
def daytime_games_fraction := 1 - night_games_fraction
def percentage_daytime_attended : ℝ := 0.40
def percentage_night_attended : ℝ := 0.20

theorem joan_missed_games : 
  let night_games := total_games * night_games_fraction in
  let daytime_games := total_games - night_games in
  let daytime_games_attended := daygames * percentage_daytime_attended in
  let night_games_attended := night_games * percentage_night_attended in
  let total_games_attended := daytime_games_attended + night_games_attended in
  let games_missed := total_games - total_games_attended in 
  games_missed = 571 :=
by {
  sorry
}

end joan_missed_games_l510_510082


namespace angle_measure_l510_510956

variable (x : ℝ)

def complement (x : ℝ) : ℝ := 90 - x

def supplement (x : ℝ) : ℝ := 180 - x

theorem angle_measure (h : supplement x = 8 * complement x) : x = 540 / 7 := by
  sorry

end angle_measure_l510_510956


namespace smallest_number_satisfying_conditions_l510_510819

theorem smallest_number_satisfying_conditions :
  ∃ (n : ℕ), n % 6 = 2 ∧ n % 7 = 3 ∧ n % 8 = 4 ∧ ∀ m, (m % 6 = 2 → m % 7 = 3 → m % 8 = 4 → n ≤ m) :=
  sorry

end smallest_number_satisfying_conditions_l510_510819


namespace Jan_more_miles_than_Ian_l510_510595

-- Defining the values and conditions
variables (d t s : ℕ)      -- distance, time, speed of Ian
variables (dH tH sH : ℕ)    -- distance, time, speed of Han
variables (dJ tJ sJ : ℕ)    -- distance, time, speed of Jan

-- Conditions
def IanTravel := d = s * t
def HanTravel := dH = (s + 10) * (t + 2)
def JanTravel := dJ = (s + 15) * (t + 3)
def HanDistance := dH = d + 100

-- Theorem to prove
theorem Jan_more_miles_than_Ian (h1 : IanTravel) (h2 : HanTravel) (h3 : JanTravel) (h4 : HanDistance) : dJ = d + 165 :=
by
  -- Definitions and conditions are in place
  sorry

end Jan_more_miles_than_Ian_l510_510595


namespace set_intersection_and_subsets_l510_510032

open Set

variable (U S T : Set ℕ)
variable hU : U = {1, 2, 3, 4, 5, 6}
variable hS : S = {1, 2, 5}
variable hT : T = {2, 3, 6}
noncomputable def complement_U_T := U \ T
noncomputable def S_inter_complement_U_T := S ∩ complement_U_T

theorem set_intersection_and_subsets :
  (S ∩ (U \ T) = {1, 5}) ∧ (card S.to_finset.powerset = 8) := by
  sorry

end set_intersection_and_subsets_l510_510032


namespace find_polynomials_l510_510333

noncomputable def polynomial_satisfies (P : ℝ → ℝ) :=
  ∀ a b c : ℝ, P (√3 * (a - b)) + P (√3 * (b - c)) + P (√3 * (c - a)) =
               P (2 * a - b - c) + P (-a + 2 * b - c) + P (-a - b + 2 * c)

theorem find_polynomials (P : ℝ → ℝ) [is_polynomial : polynomial_satisfies P] :
  ∃ (C : ℝ) (a b : ℝ), (P = λ x, C) ∨ (P = λ x, a * x^2 + b * x + C) :=
sorry

end find_polynomials_l510_510333


namespace arithmetic_seq_max_sum_l510_510439

noncomputable def binom := nat.choose
noncomputable def arithSeq (a1 d : ℤ) (n : ℕ) := a1 + d * n
noncomputable def sumArithSeq (a1 d : ℤ) (k : ℕ) : ℤ := k * (a1 + (a1 + d * (k - 1))) / 2

theorem arithmetic_seq_max_sum :
  ∀ n : ℕ, 
    (C(5 * n, 11 - 2 * n) - A(11 - 3 * n, 2 * n - 2) = 100) →
    (77^77 - 15) % 19 = 5 →
    let a1 := 100 
    let d := -4 
    sumArithSeq a1 d 25 = 1300 ∧ 
    sumArithSeq a1 d 26 = 1300 :=
  sorry

end arithmetic_seq_max_sum_l510_510439


namespace compute_k_l510_510687

theorem compute_k (k : ℤ) (h : k > 2) :
  (\log 10 ((k - 2)!) + \log 10 ((k - 1)!) + 1.7 = 2 * \log 10 (k!)) ↔ k = 5 := 
sorry

end compute_k_l510_510687


namespace air_conditioner_sale_price_l510_510666

theorem air_conditioner_sale_price (P : ℝ) (d1 d2 : ℝ) (hP : P = 500) (hd1 : d1 = 0.10) (hd2 : d2 = 0.20) :
  ((P * (1 - d1)) * (1 - d2)) / P * 100 = 72 :=
by
  sorry

end air_conditioner_sale_price_l510_510666


namespace lindsey_integer_l510_510867

theorem lindsey_integer (n : ℕ) (a b c : ℤ) (h1 : n < 50)
                        (h2 : n = 6 * a - 1)
                        (h3 : n = 8 * b - 5)
                        (h4 : n = 3 * c + 2) :
  n = 41 := 
  by sorry

end lindsey_integer_l510_510867


namespace minimum_value_expression_l510_510346

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ( (3*a*b - 6*b + a*(1-a))^2 + (9*b^2 + 2*a + 3*b*(1-a))^2 ) / (a^2 + 9*b^2) ≥ 4 :=
sorry

end minimum_value_expression_l510_510346


namespace cars_sold_on_second_day_l510_510620

theorem cars_sold_on_second_day (x : ℕ) 
  (h1 : 14 + x + 27 = 57) : x = 16 :=
by 
  sorry

end cars_sold_on_second_day_l510_510620


namespace smallest_x_value_satisfies_equation_l510_510962

theorem smallest_x_value_satisfies_equation :
  ∃ x : ℝ, 0 < x ∧ sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
by
  sorry

end smallest_x_value_satisfies_equation_l510_510962


namespace error_percentage_correct_l510_510823

variable (L W : ℝ)

def L_measured := 1.12 * L
def W_measured := 0.95 * W

def Area_actual := L * W
def Area_calculated := L_measured * W_measured

def Error := Area_calculated - Area_actual

def Error_percentage := (Error / Area_actual) * 100

theorem error_percentage_correct : Error_percentage L W = 6.4 := by
  unfold Error_percentage Error Area_calculated L_measured W_measured Area_actual
  -- continue the proof steps as needed
  sorry

end error_percentage_correct_l510_510823


namespace maximize_take_home_pay_l510_510818

def tax_collected (x : ℝ) : ℝ :=
  10 * x^2

def take_home_pay (x : ℝ) : ℝ :=
  1000 * x - tax_collected x

theorem maximize_take_home_pay : ∃ x : ℝ, (x * 1000 = 50000) ∧ (∀ y : ℝ, take_home_pay x ≥ take_home_pay y) := 
sorry

end maximize_take_home_pay_l510_510818


namespace center_in_first_quadrant_of_circle_l510_510804

theorem center_in_first_quadrant_of_circle (m : ℝ) :
  let eq := (λ x y : ℝ, x^2 + y^2 - 2*m*x + (2*m - 2)*y + 2*m^2) in
  (∃ x y : ℝ, eq x y = 0) → -- represents a circle if it has at least one solution
  (0 < m) ∧ (m < 1) :=
by
  sorry

end center_in_first_quadrant_of_circle_l510_510804


namespace domain_of_f_l510_510071

noncomputable def f (x : ℝ) : ℝ := (sqrt (x + 2)) + (1 / (x - 1))

theorem domain_of_f :
  ∀ x : ℝ, (x ≥ -2 ∧ x ≠ 1) ↔ ∃ y : ℝ, y = f x :=
begin
  sorry
end

end domain_of_f_l510_510071


namespace cos_add_phi_zeros_l510_510806

theorem cos_add_phi_zeros (φ : ℝ) (h : ∀ (x : ℝ), cos (x + φ) = - cos (-x + φ)) :
  ∃ k : ℤ, ∀ x : ℝ, (cos (x + φ) = 0) → (∃ k : ℤ, x = k * π) := 
sorry

end cos_add_phi_zeros_l510_510806


namespace common_ratio_of_geom_seq_l510_510001

noncomputable theory

def is_arith_seq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def is_geom_seq (a1 a2 a4 : ℝ) :=
  a2 * a2 = a1 * a4

theorem common_ratio_of_geom_seq (a : ℕ → ℝ) (d : ℝ) (h_arith : is_arith_seq a d) (hd_ne_zero : d ≠ 0) (h_geom : is_geom_seq (a 1) (a 2) (a 4)) : 
  (a 2) / (a 1) = 2 :=
by
  sorry

end common_ratio_of_geom_seq_l510_510001


namespace length_first_train_is_121_l510_510186

noncomputable def relative_speed (v1 v2 : ℝ) : ℝ := 
  (v1 + v2) * (1000 / 3600)

noncomputable def distance_covered (relative_speed time : ℝ) : ℝ := 
  relative_speed * time

noncomputable def length_of_first_train (total_distance length_second_train : ℝ) : ℝ := 
  total_distance - length_second_train

theorem length_first_train_is_121.197 :
  let v1 := 80
  let v2 := 65
  let L2 := 165
  let t := 7.100121645440779
  let relative_speed := relative_speed v1 v2
  let total_distance := distance_covered relative_speed t
  length_of_first_train total_distance L2 ≈ 121.197 := 
by 
  sorry

end length_first_train_is_121_l510_510186


namespace x_cubed_inverse_varies_with_y_squared_l510_510899

theorem x_cubed_inverse_varies_with_y_squared (x y : ℝ) :
  (x^3 * y^2 = 2000) → (y = 8 → x^3 = 31.25) :=
by
  assume h : x^3 * y^2 = 2000
  assume h1 : y = 8
  sorry

end x_cubed_inverse_varies_with_y_squared_l510_510899


namespace min_f_a_eq_1_min_f_a_le_neg1_min_f_neg1_lt_a_lt_0_l510_510415

-- Define the quadratic function
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 5

-- Prove the minimum value for a = 1 and x in [-1, 0]
theorem min_f_a_eq_1 : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f 1 x ≥ 5 :=
by
  sorry

-- Prove the minimum value for a < 0 and x in [-1, 0], when a ≤ -1
theorem min_f_a_le_neg1 (h : ∀ a : ℝ, a ≤ -1) : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f a (-1) ≤ f a x :=
by
  sorry

-- Prove the minimum value for a < 0 and x in [-1, 0], when -1 < a < 0
theorem min_f_neg1_lt_a_lt_0 (h : ∀ a : ℝ, -1 < a ∧ a < 0) : ∀ x : ℝ, x ∈ Set.Icc (-1) 0 → f a a ≤ f a x :=
by
  sorry

end min_f_a_eq_1_min_f_a_le_neg1_min_f_neg1_lt_a_lt_0_l510_510415


namespace non_pizza_eaters_count_l510_510672

theorem non_pizza_eaters_count (T S : ℕ) (ate_pizza_T ate_pizza_S : ℕ) :
  T = 75 ∧ S = 120 ∧ ate_pizza_T = (7 * T) / 15 ∧ ate_pizza_S = (3 * S) / 8 →
  (T - ate_pizza_T) + (S - ate_pizza_S) = 115 :=
by
  intros h
  have hT : T = 75 := h.1
  have hS : S = 120 := h.2.1
  have hate_pizza_T : ate_pizza_T = (7 * 75) / 15 := congr_arg (λ x, x) h.2.2.1
  have hate_pizza_S : ate_pizza_S = (3 * 120) / 8 := congr_arg (λ x, x) h.2.2.2
  sorry

end non_pizza_eaters_count_l510_510672


namespace range_of_x_l510_510069

theorem range_of_x (x : ℝ) : (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end range_of_x_l510_510069


namespace find_q_l510_510185

theorem find_q (p q : ℕ) (hp_prime : Nat.Prime p) (hq_prime : Nat.Prime q) (hp_congr : 5 * p ≡ 3 [MOD 4]) (hq_def : q = 13 * p + 2) : q = 41 := 
sorry

end find_q_l510_510185


namespace probability_sum_four_l510_510530

/-- The problem statement:
    What is the probability that the sum of two integers is 4 after splitting 3.7 into 
    two nonnegative real numbers uniformly at random and rounding each to its nearest integer?
-/
theorem probability_sum_four :
  let x : ℝ := (0 : ℝ) in
  ∃ (f : ℝ → ℝ), f = λ x, (if x < 0.5 then 4
                         else if x < 1.5 then 3
                         else if x < 2.5 then 2
                         else if x < 3.5 then 1
                         else 0) ..
  by sorry

end probability_sum_four_l510_510530


namespace length_of_chord_AB_l510_510827

noncomputable def lengthOfChord {
  t : Type*
} (line : t → ℝ × ℝ) (circle : ℝ × ℝ) : ℝ :=
  let l := (λ t, (3 / 5 * t, 4 / 5 * t))
  let C := (1, 0)
  let d := 4 / Real.sqrt (4^2 + (-3)^2)
  2 * Real.sqrt (1 - d^2)

theorem length_of_chord_AB :
  lengthOfChord (λ t, (3 / 5 * t, 4 / 5 * t)) (1, 0) = 6 / 5 :=
by sorry

end length_of_chord_AB_l510_510827


namespace garden_width_l510_510981

theorem garden_width (w : ℝ) (h : w * (3 * w) = 675) : w = 15 :=
by
sory

end garden_width_l510_510981


namespace routes_from_M_to_N_is_4_l510_510187

-- Definitions of the nodes and their routes
inductive Node
| M | B | X | A | C | D | N

open Node

-- Definition of routes between nodes
def routes : Node → List Node
| M => [B, X]
| B => [A, N]
| X => [A, D]
| A => [C, D]
| C => [N]
| D => [N]
| N => []

-- Function to count routes from a given node to node N
def count_routes_to_N : Node → ℕ
| N => 1
| node => (routes node).map count_routes_to_N |>.sum

-- Theorem stating the number of routes from M to N is 4
theorem routes_from_M_to_N_is_4 : count_routes_to_N M = 4 :=
sorry

end routes_from_M_to_N_is_4_l510_510187


namespace find_f4_l510_510762

theorem find_f4 (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = -f (-x + 1)) 
  (h2 : ∀ x, f (x - 1) = f (-x - 1)) 
  (h3 : f 0 = 2) : 
  f 4 = -2 :=
sorry

end find_f4_l510_510762


namespace general_solution_diff_eq_l510_510147

/-- Given the differential equation (x + y^2) dx - 2xy dy = 0, 
    the general solution is x = C * exp(y^2 / x). -/
theorem general_solution_diff_eq 
  (C : ℝ) (x y : ℝ) :
  (x + y^2) * (deriv (λ x, x)) - 2 * x * y * (deriv (λ y, y)) = 0 → 
  x = C * exp(y^2 / x) :=
by
  sorry

end general_solution_diff_eq_l510_510147


namespace smallest_x_value_satisfies_equation_l510_510961

theorem smallest_x_value_satisfies_equation :
  ∃ x : ℝ, 0 < x ∧ sqrt (3 * x) = 5 * x ∧ x = 3 / 25 :=
by
  sorry

end smallest_x_value_satisfies_equation_l510_510961


namespace yellow_ball_count_l510_510051

def total_balls : ℕ := 500
def red_balls : ℕ := total_balls / 3
def remaining_after_red : ℕ := total_balls - red_balls
def blue_balls : ℕ := remaining_after_red / 5
def remaining_after_blue : ℕ := remaining_after_red - blue_balls
def green_balls : ℕ := remaining_after_blue / 4
def yellow_balls : ℕ := total_balls - (red_balls + blue_balls + green_balls)

theorem yellow_ball_count : yellow_balls = 201 := by
  sorry

end yellow_ball_count_l510_510051


namespace no_possible_arrangement_l510_510837

theorem no_possible_arrangement :
  ¬ ∃ (f : ℕ × ℕ → ℤ), (∀ i j, f (i, j) = 1 ∨ f (i, j) = -1) ∧
  (abs (∑ i in fin_range 600, ∑ j in fin_range 600, f (i, j)) < 90000) ∧
  (∀ (x y : ℕ) (h1 : x ≤ 596) (h2 : y ≤ 594),
    abs (∑ i in fin_range 4, ∑ j in fin_range 6, f (x + i, y + j)) > 4) ∧
  (∀ (x y : ℕ) (h1 : x ≤ 594) (h2 : y ≤ 596),
    abs (∑ i in fin_range 6, ∑ j in fin_range 4, f (x + i, y + j)) > 4) :=
by sorry

end no_possible_arrangement_l510_510837


namespace color_grids_l510_510057

-- Defining the grid and color conditions
def is_red (color : (ℕ × ℕ) → bool) (m n : ℕ) : bool := color (m, n)
def is_blue (color : (ℕ × ℕ) → bool) (m n : ℕ) : bool := !is_red color m n

def valid_coloring (color : (ℕ × ℕ) → bool) : Prop :=
  ∀ (m n : ℕ), 
  (1 ≤ m ∧ m < 6 ∧ 1 ≤ n ∧ n < 6) → 
  ((is_red color m n ∧ is_red color (m + 1) n ∧ is_blue color m (n + 1) ∧ is_blue color (m + 1) (n + 1)) ∨
   (is_red color m n ∧ is_blue color (m + 1) n ∧ is_red color m (n + 1) ∧ is_blue color (m + 1) (n + 1)) ∨
   (is_blue color m n ∧ is_red color (m + 1) n ∧ is_blue color m (n + 1) ∧ is_red color (m + 1) (n + 1)) ∨
   (is_blue color m n ∧ is_blue color (m + 1) n ∧ is_red color m (n + 1) ∧ is_red color (m + 1) (n + 1)))

-- The number of valid colorings
def number_of_colorings : ℕ := 126

-- The theorem stating the required proof
theorem color_grids :
  ∃ color : (ℕ × ℕ) → bool, valid_coloring color ∧ number_of_colorings = 126 :=
by
  sorry

end color_grids_l510_510057


namespace correlation_approx_neg_one_regression_equation_l510_510229

variables (x y : Fin 5 → ℝ) (x_vals : Vector ℝ 5) (y_vals : Vector ℝ 5)
          (x_bar : ℝ := (x_vals.getAvg))
          (y_bar : ℝ := (y_vals.getAvg))
          (n : ℕ := 5) -- Since we have 5 data points

-- Given x and y values
constant x_data : Fin 5 → ℝ := fun i => match i with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5

constant y_data : Fin 5 → ℝ := fun i => match i with
  | 0 => 9.5
  | 1 => 8.6
  | 2 => 7.8
  | 3 => 7.0
  | 4 => 6.1

-- Condition Definitions
def sum_x_xbar_squared : ℝ := ∑ i, (x_data i - x_bar) ^ 2
def sum_y_ybar_squared : ℝ := ∑ i, (y_data i - y_bar) ^ 2
def sum_x_ybar_prod : ℝ := ∑ i, (x_data i - x_bar) * (y_data i - y_bar)

-- Question 1: Prove the correlation coefficient approximately equals -1
theorem correlation_approx_neg_one :
  let r := sum_x_ybar_prod / (Real.sqrt (sum_x_xbar_squared * sum_y_ybar_squared))
  r ≈ -1 := sorry

-- Question 2: Prove the regression equation
theorem regression_equation :
  let hat_b := sum_x_ybar_prod / sum_x_xbar_squared
  let hat_a := y_bar - hat_b * x_bar
  ∀ (x: ℝ), hat_a + hat_b * x = -0.84 * x + 10.32 := sorry

end correlation_approx_neg_one_regression_equation_l510_510229


namespace units_digit_of_power_17_l510_510701

theorem units_digit_of_power_17 (n : ℕ) (k : ℕ) (h_n4 : n % 4 = 3) : (17^n) % 10 = 3 :=
  by
  -- Since units digits of powers repeat every 4
  sorry

-- Specific problem instance
example : (17^1995) % 10 = 3 := units_digit_of_power_17 1995 17 (by norm_num)

end units_digit_of_power_17_l510_510701


namespace proof_problem_l510_510103

variables {O1 O2 O3 : Type*} [circle O1] [circle O2] [circle O3]
variables {A D B C E P Q M N : point}
variables (h1 : common_chord AD O1 O2)
variables (h2 : line_through D)
variables (h3 : intersects D B O1)
variables (h4 : intersects D C O2)
variables (h5 : on_segment AD E)
variables (h6 : distinct A D E)
variables (h7 : intersects_line_segment CE O1 P Q)
variables (h8 : intersects_line_segment BE O2 M N)

theorem proof_problem :
  concyclic P M Q N ∧ center_of_circle P M Q N = O3 ∧ perpendicular (line_through DO3) (line_segment BC) :=
begin
  sorry
end

end proof_problem_l510_510103


namespace maria_candy_remaining_l510_510119

theorem maria_candy_remaining :
  let c := 520.75
  let e := c / 2
  let g := 234.56
  let r := e - g
  r = 25.815 := by
  sorry

end maria_candy_remaining_l510_510119


namespace village_distance_l510_510555

theorem village_distance
  (d : ℝ)
  (uphill_speed : ℝ) (downhill_speed : ℝ)
  (total_time : ℝ)
  (h1 : uphill_speed = 15)
  (h2 : downhill_speed = 30)
  (h3 : total_time = 4) :
  d = 40 :=
by
  sorry

end village_distance_l510_510555


namespace find_interest_rate_l510_510713

variable (P t n A r : ℝ)
variable (hP : P = 1000)
variable (ht : t = 2)
variable (hn : n = 2)
variable (hA : A = 1082.43216000000007)
variable (CI : A = P * (1 + r / n)^(n * t))

theorem find_interest_rate :
  r = 0.039892 :=
by
  have h : P * (1 + r / n)^(n * t) = A := CI
  simp only [Real.pow_nat, mul_eq_mul_left_iff, one_mul, add_mul, eq_self_iff_true, mul_one, add_neg_eq_sub, div_eq_inv_mul, one_div, pow_bit1, eq_iff_true_of_sub_eq_zero, hP, ht, hn, hA] at h
  sorry

end find_interest_rate_l510_510713


namespace smallest_students_proof_l510_510260

noncomputable def smallest_number_of_students (arrangements : ℕ → ℕ) (d : ℕ) :=
  -- Returns the smallest number of students given arrangements per row for the first d days.
  let n := arrangements 1 in
  let possible_arrangements := {s | ∃ (f: ℕ → ℕ), (∀ i : ℕ, 1 ≤ i ∧ i ≤ d → (s = f i)) ∧
    (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ d → f i ≠ f j) ∧ (∀ i : ℕ, 1 ≤ i ∧ i ≤ d → (s mod (f i) = 0)) } in
  Classical.choose (Finset.min'_mem possible_arrangements ⟨72, sorry⟩)

theorem smallest_students_proof : smallest_number_of_students (λ n, 18) 12 = 72 := sorry

end smallest_students_proof_l510_510260


namespace find_t_l510_510396

variable (a t : ℝ)

def f (x : ℝ) : ℝ := a * x + 19

theorem find_t (h1 : f a 3 = 7) (h2 : f a t = 15) : t = 1 :=
by
  sorry

end find_t_l510_510396


namespace volume_tetrahedron_KLMN_is_approximately_2_09_l510_510465

-- Given side lengths
constant EF GH : ℝ := 7
constant EG FH : ℝ := 10
constant EH FG : ℝ := 11

-- Tetrahderon with given side lengths and K, L, M, N as the centers of the inscribed circles
noncomputable def volume_tetrahedron_KLMN (EF GH EG FH EH FG : ℝ) : ℝ :=
  (some_calculation_function EF GH EG FH EH FG)  -- Placeholder for the actual volume calculation function

-- Assert that the volume is approximately 2.09
theorem volume_tetrahedron_KLMN_is_approximately_2_09 :
  abs (volume_tetrahedron_KLMN EF GH EG FH EH FG - 2.09) < 0.01 :=
by
  sorry

end volume_tetrahedron_KLMN_is_approximately_2_09_l510_510465


namespace group_d_forms_triangle_l510_510917

-- Definitions for the stick lengths in each group
def group_a := (1, 2, 6)
def group_b := (2, 2, 4)
def group_c := (1, 2, 3)
def group_d := (2, 3, 4)

-- Statement to prove that Group D can form a triangle
theorem group_d_forms_triangle (a b c : ℕ) : a = 2 → b = 3 → c = 4 → a + b > c ∧ a + c > b ∧ b + c > a := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  apply And.intro
  sorry
  apply And.intro
  sorry
  sorry

end group_d_forms_triangle_l510_510917


namespace scaled_circle_area_l510_510137

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem scaled_circle_area :
  ∃ (A B : ℝ × ℝ), 
  A = (-1, 1) ∧ B = (3, 6) ∧
  (∃ (scaled_diameter : ℝ), scaled_diameter = 3 * dist A B ∧
  (∃ (radius : ℝ), radius = scaled_diameter / 2 ∧
  (π * radius^2 = 369 * π / 4))) :=
by 
  sorry

end scaled_circle_area_l510_510137


namespace simplify_and_evaluate_l510_510892

noncomputable def expression (x : ℤ) : ℤ :=
  ( (-2 * x^3 - 6 * x) / (-2 * x) - 2 * (3 * x + 1) * (3 * x - 1) + 7 * x * (x - 1) )

theorem simplify_and_evaluate : 
  (expression (-3) = -64) := by
  sorry

end simplify_and_evaluate_l510_510892


namespace determine_k_and_p_l510_510042

theorem determine_k_and_p (x y z : ℕ) (k p : ℤ)
  (h1 : x = 9)
  (h2 : y = 343)
  (h3 : z = 2)
  (h4 : (x / 3)^32 * (y / 125)^k * ((z^3) / (7^3))^p = 1 / (27^32 * z^15)) :
  k = 5 ∧ p = -5 := 
by
  sorry

end determine_k_and_p_l510_510042


namespace suff_not_necessary_condition_for_parallel_l510_510768

section
variable {α : Type*} [AddCommGroup α] [Module ℝ α]

def are_parallel (a b : α) : Prop := ∃ (λ : ℝ), a = λ • b

theorem suff_not_necessary_condition_for_parallel
  (a b : α) (h_not_both_zero : ¬(a = 0 ∧ b = 0)) :
  (∃ (λ : ℝ), a = λ • b) ↔ (a ≠ 0 ∧ are_parallel a b) := 
sorry
end

end suff_not_necessary_condition_for_parallel_l510_510768


namespace binary_digit_difference_l510_510967

theorem binary_digit_difference :
  let digits := λ n : ℕ, (Nat.log n 2).succ in
  digits 1600 - digits 400 = 2 :=
by 
  sorry

end binary_digit_difference_l510_510967


namespace pi_is_the_only_irrational_number_among_0_1_2_pi_sqrt4_l510_510223

theorem pi_is_the_only_irrational_number_among_0_1_2_pi_sqrt4 : 
  (irrational π) ∧ (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1/2 ∧ x ≠ √4) :=
by
  have pi_irrational : irrational π := sorry
  have zero_rational : ∃ p q : ℤ, 0 = p / q := sorry
  have half_rational : ∃ p q : ℤ, 1/2 = p / q := sorry
  have sqrt4_rational : ∃ p q : ℤ, √4 = p / q := sorry
  exact ⟨pi_irrational, by { rintros ⟨p, q, h⟩, dsimp only,
  { simp [h, zero_rational, half_rational, sqrt4_rational] } } ⟩

end pi_is_the_only_irrational_number_among_0_1_2_pi_sqrt4_l510_510223


namespace sum_S_values_l510_510741

def S (n : ℕ) : ℤ :=
  (List.sum (List.map (λ k, if k % 2 = 0 then (4 * k - 3) else -(4 * k - 3)) (List.range n).map (λ x, x + 1)))

theorem sum_S_values : S 15 + S 22 - S 31 = -76 := by
  sorry

end sum_S_values_l510_510741


namespace one_cow_one_bag_l510_510601

-- Definitions based on the conditions provided.
def cows : ℕ := 45
def bags : ℕ := 45
def days : ℕ := 45

-- Problem statement: Prove that one cow will eat one bag of husk in 45 days.
theorem one_cow_one_bag (h : cows * bags = bags * days) : days = 45 :=
by
  sorry

end one_cow_one_bag_l510_510601


namespace total_packs_l510_510873

theorem total_packs (cards_per_person cards_per_pack : ℕ) (num_people : ℕ) 
  (h1 : cards_per_person = 540) 
  (h2 : cards_per_pack = 20) 
  (h3 : num_people = 4) : 
  (cards_per_person / cards_per_pack) * num_people = 108 := 
by
  sorry

end total_packs_l510_510873


namespace turtle_speed_l510_510120

theorem turtle_speed
  (hare_speed : ℝ)
  (race_distance : ℝ)
  (head_start : ℝ) :
  hare_speed = 10 → race_distance = 20 → head_start = 18 → 
  (race_distance / (head_start + race_distance / hare_speed) = 1) :=
by
  intros
  sorry

end turtle_speed_l510_510120


namespace count_total_squares_with_odd_factors_l510_510422

-- Define the predicates for two-digit and three-digit perfect squares
def is_two_digit_square (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ ∃ k, k^2 = n

def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k, k^2 = n

-- Auxiliary lemma to count all two-digit squares
lemma count_two_digit_squares : {n : ℕ | is_two_digit_square n}.to_finset.card = 6 := sorry

-- Auxiliary lemma to count all three-digit squares
lemma count_three_digit_squares : {n : ℕ | is_three_digit_square n}.to_finset.card = 22 := sorry

-- Main theorem
theorem count_total_squares_with_odd_factors :
  {n : ℕ | is_two_digit_square n ∨ is_three_digit_square n}.to_finset.card = 28 :=
begin
  rw [← finset.card_union_eq],
  { exact add_eq_of_eq count_two_digit_squares count_three_digit_squares },
  -- Proven that the sets are disjoint
  { sorry }
end

end count_total_squares_with_odd_factors_l510_510422


namespace radioactive_half_life_l510_510163

theorem radioactive_half_life (x : ℝ) (lg2 lg3 : ℝ) (hlf : 0.9^x = 0.5) (h_lg2 : Real.log 2 = lg2) (h_lg3 : Real.log 3 = lg3) :
  x ≈ 6.6 :=
by
  rw [←h_lg2, ←h_lg3]
  sorry

end radioactive_half_life_l510_510163


namespace final_salary_adjustment_l510_510659

theorem final_salary_adjustment (S : ℝ) :
  let S1 := S * 1.30,
      S2 := S1 * 0.80,
      S3 := S2 * 1.10,
      S4 := S3 * 0.75
  in S4 = S * 1.04 :=
by
  sorry

end final_salary_adjustment_l510_510659


namespace isosceles_triangle_angle_Q_l510_510566

theorem isosceles_triangle_angle_Q (x : ℝ) (PQR : Triangle)
  (h1 : PQR.angles Q = PQR.angles R)
  (h2 : PQR.angles R = 5 * PQR.angles P)
  (sum_angles : PQR.angles P + PQR.angles Q + PQR.angles R = 180) :
  PQR.angles Q = 900 / 11 :=
by
  sorry

end isosceles_triangle_angle_Q_l510_510566


namespace intersection_A_B_l510_510428

open Set

-- Define the sets A and B based on the conditions provided
def A : Set ℤ := { x | x^2 - 2 * x - 8 ≤ 0 }
def B : Set ℤ := { x | log 2 (x : ℝ) > 1 }

-- State the theorem that proves the intersection of A and B equals {3, 4}
theorem intersection_A_B : A ∩ B = { 3, 4 } := by
  sorry

end intersection_A_B_l510_510428


namespace range_of_x_l510_510068

theorem range_of_x (x : ℝ) : (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end range_of_x_l510_510068


namespace like_terms_proof_l510_510750

variable (a b : ℤ)
variable (x y : ℤ)

theorem like_terms_proof (hx : x = 2) (hy : 3 = 1 - y) : x * y = -4 := by
  rw [hx, hy]
  sorry

end like_terms_proof_l510_510750


namespace shaded_region_area_l510_510497

structure Point where
  x : ℝ
  y : ℝ

def W : Point := ⟨0, 0⟩
def X : Point := ⟨5, 0⟩
def Y : Point := ⟨5, 2⟩
def Z : Point := ⟨0, 2⟩
def Q : Point := ⟨1, 0⟩
def S : Point := ⟨5, 0.5⟩
def R : Point := ⟨0, 1⟩
def D : Point := ⟨1, 2⟩

def triangle_area (A B C : Point) : ℝ :=
  0.5 * |(A.x * B.y + B.x * C.y + C.x * A.y) - (B.x * A.y + C.x * B.y + A.x * C.y)|

theorem shaded_region_area : triangle_area R D Y = 1 := by
  sorry

end shaded_region_area_l510_510497


namespace 数字花园_proof_l510_510829

-- Definitions reflecting the problem conditions
def digit_repr (n : ℕ) : Prop := n < 10

def unique_digits (a b c : ℕ) : Prop :=
  ∀ x y, (x = a ∨ x = b ∨ x = c) → (y = a ∨ y = b ∨ y = c) → x ≠ y → x ≠ y

def sum_1_to_10 : ℕ := list.sum (list.range 11)  -- This calculates the sum 1 + 2 + ... + 10

variables {x y z : ℕ}

-- The proof problem statement with conditions
theorem 数字花园_proof :
  ∀ (x y z : ℕ),
    digit_repr x ∧ 100 ≤ y ∧ y < 1000 ∧ 100 ≤ z ∧ z < 1000 ∧
    100 * x + y + z = 2015 ∧
    z + sum_1_to_10 = y ∧
    unique_digits x y z →
    100 * x + y = 1985 :=
by
  sorry

end 数字花园_proof_l510_510829


namespace probability_exactly_one_pass_l510_510650

theorem probability_exactly_one_pass (p : ℝ) (h : p = 1/2) : 
  let p_pass := p in
  let p_fail := 1 - p in
  let pf := p_pass * p_fail in
  let fp := p_fail * p_pass in
  pf + fp = 1/2 :=
by
  sorry

end probability_exactly_one_pass_l510_510650


namespace inverse_proportion_range_l510_510028

theorem inverse_proportion_range (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (y = (m + 5) / x) → ((x > 0 → y < 0) ∧ (x < 0 → y > 0))) →
  m < -5 :=
by
  intros h
  -- Skipping proof with sorry as specified
  sorry

end inverse_proportion_range_l510_510028


namespace range_of_m_l510_510424

theorem range_of_m (m : ℝ) (h : (2 - m) * ( |m| - 3) > 0) :
  m ∈ set.Ioo (-∞:ℝ) (-3) ∪ set.Ioo 2 3 :=
sorry

end range_of_m_l510_510424


namespace probability_of_playing_exactly_one_instrument_l510_510455

noncomputable def total_people : ℕ := 800
noncomputable def fraction_playing_at_least_one : ℚ := 2/5
noncomputable def number_playing_two_or_more : ℕ := 96

noncomputable def number_playing_at_least_one : ℕ := (fraction_playing_at_least_one * total_people).to_nat
noncomputable def number_playing_exactly_one : ℕ := number_playing_at_least_one - number_playing_two_or_more
noncomputable def probability_playing_exactly_one : ℚ := number_playing_exactly_one / total_people

theorem probability_of_playing_exactly_one_instrument : 
  probability_playing_exactly_one = 28 / 100 := 
by
  calc
    probability_playing_exactly_one
      = number_playing_exactly_one / total_people : by rfl
    ... = (number_playing_at_least_one - number_playing_two_or_more : ℕ) / total_people : by sorry
    ... = 224 / 800 : by sorry
    ... = 28 / 100 : by sorry

#eval probability_of_playing_exactly_one_instrument

end probability_of_playing_exactly_one_instrument_l510_510455


namespace percentage_decrease_l510_510089

variables {S : ℝ} (X : ℝ)

-- Conditions
def original_salary (S : ℝ) : ℝ := S
def decreased_salary (S : ℝ) (X : ℝ) : ℝ := S * (100 - X) / 100
def increased_salary (S : ℝ) (X : ℝ) : ℝ := decreased_salary S X * 130 / 100

-- Given final condition
def final_salary_condition (S : ℝ) : ℝ := S * 65 / 100

-- The main statement to prove
theorem percentage_decrease : increased_salary S X = final_salary_condition S → X = 50 :=
by
  sorry

end percentage_decrease_l510_510089


namespace avg_weight_ab_l510_510532

theorem avg_weight_ab (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 30) 
  (h2 : (B + C) / 2 = 28) 
  (h3 : B = 16) : 
  (A + B) / 2 = 25 := 
by 
  sorry

end avg_weight_ab_l510_510532


namespace max_tan_angle_MPN_l510_510913

open Real

theorem max_tan_angle_MPN 
  (θ : ℝ) 
  (hC1 : ∀ (x y : ℝ), (x-3)^2 + y^2 = 4/25) 
  (hC2 : ∀ (x y : ℝ), (x-3-cos θ)^2 + (y-sin θ)^2 = 1/25) 
  : ∃ P M N : ℝ × ℝ, 
    (P ∈ C2 ∧ tangent_point PM C1 M ∧ tangent_point PN C1 N) 
    → tan (angle MPN) ≤ 4 * sqrt 2 / 7
    where
      P := (3 + cos θ, sin θ)
      M and N are the points of tangency of the line through P on C2 to C1
    sorry

end max_tan_angle_MPN_l510_510913


namespace sum_of_coefficients_l510_510446

theorem sum_of_coefficients (a b c : ℕ) (h1 : c > 0) (h2 : c = 21) (h3 : a = 28) (h4 : b = 24) :
  (sqrt 3 + 1 / sqrt 3 + sqrt 7 + 1 / sqrt 7) = (a * sqrt 3 + b * sqrt 7) / c ∧ a + b + c = 73 :=
by
  sorry

end sum_of_coefficients_l510_510446


namespace projection_onto_w_l510_510554

def vector_proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let denom := v.1 * v.1 + v.2 * v.2
  ( (u.1 * v.1 + u.2 * v.2) / denom * v.1, (u.1 * v.1 + u.2 * v.2) / denom * v.2 )

-- Define the given conditions
def given_conditions : Prop :=
  vector_proj (2, 5) (4 / 13, 10 / 13) = (4 / 13, 10 / 13)

-- Prove the projection of (3, 4) onto the vector satisfies the expected projection
theorem projection_onto_w :
  given_conditions →
  vector_proj (3, 4) (4 / 13, 10 / 13) = (52 / 29, 130 / 29) :=
by
  intros h
  sorry

end projection_onto_w_l510_510554


namespace perpendicular_vectors_l510_510789

variables (m : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-4, m)

theorem perpendicular_vectors : a.1 * b.1 + a.2 * b.2 = 0 → m = 2 := by
  sorry

end perpendicular_vectors_l510_510789


namespace nominal_rate_of_interest_l510_510538

noncomputable def EAR := 0.1664
noncomputable def n := 2

theorem nominal_rate_of_interest (i : ℝ) :
  (1 + i / n) ^ n - 1 = EAR → i = 0.16 :=
by
  sorry

end nominal_rate_of_interest_l510_510538


namespace midpoint_after_translation_l510_510551

structure Point2D :=
(x : ℝ)
(y : ℝ)

def midpoint (p1 p2 : Point2D) : Point2D :=
{ x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2 }

def translate (p : Point2D) (dx dy : ℝ) : Point2D :=
{ x := p.x + dx,
  y := p.y + dy }

theorem midpoint_after_translation :
  let A := Point2D.mk 3 3,
      H := Point2D.mk 7 3,
      A' := translate A (-6) 3,
      H' := translate H (-6) 3,
      M := midpoint A H,
      M' := translate M (-6) 3
  in M' = Point2D.mk (-1) 6 :=
by {
  sorry
}

end midpoint_after_translation_l510_510551


namespace max_a_value_l510_510856

theorem max_a_value (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : b + d = 200) : a ≤ 449 :=
by sorry

end max_a_value_l510_510856


namespace keith_score_l510_510847

theorem keith_score (K : ℕ) (h : K + 3 * K + (3 * K + 5) = 26) : K = 3 :=
by
  sorry

end keith_score_l510_510847


namespace sum_of_series_l510_510685

theorem sum_of_series :
  (∑ a in Finset.range 100, ∑ b in Finset.range 100, ∑ c in Finset.range 100, ∑ d in Finset.range 100, if 1 ≤ a ∧ a < b ∧ b < c ∧ c < d then 1 / (2^a * 3^b * 5^c * 7^d : ℝ) else 0) = 1 / 4518816 := 
by
  sorry

end sum_of_series_l510_510685


namespace least_not_wonderful_multiple_of_12_l510_510291

def sum_of_digits (n : ℕ) : ℕ :=
  n.toString.toList.map (λ c => c.toString.toNat!).sum

def is_wonderful (n : ℕ) : Prop :=
  n > 0 ∧ n % sum_of_digits n = 0

theorem least_not_wonderful_multiple_of_12 : ∀ (n : ℕ), n = 96 → n % 12 = 0 →  ¬ is_wonderful n :=
by
  intro n hn hm
  simp [is_wonderful, sum_of_digits]
  sorry

end least_not_wonderful_multiple_of_12_l510_510291


namespace typing_cost_per_page_l510_510927

-- Define the cost per page for the first time a page is typed
variable (x : ℝ)

-- Conditions: Typing details and costs
variable (pages : ℝ := 100)
variable (revised_once : ℝ := 30)
variable (revised_twice : ℝ := 20)
variable (revision_cost : ℝ := 5)
variable (total_cost : ℝ := 1350)

-- Total pages typed
def typing_cost := pages * x

-- Total revision costs
def revision_cost_once := revised_once * revision_cost
def revision_cost_twice := revised_twice * revision_cost * 2
def total_revision_cost := revision_cost_once + revision_cost_twice

-- Total cost
def total_typing_cost := typing_cost + total_revision_cost

-- The main theorem to prove
theorem typing_cost_per_page :
  total_typing_cost = total_cost → x = 10 := by
-- placeholder to skip the proof
  sorry

end typing_cost_per_page_l510_510927


namespace train_speed_correct_l510_510656

def speed_of_train (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (18 / 5)

theorem train_speed_correct :
  speed_of_train 140 9 = 56 := by
    sorry

end train_speed_correct_l510_510656


namespace find_m_value_l510_510442

theorem find_m_value 
  (h : ∀ x y m : ℝ, 2*x + y + m = 0 → (1 : ℝ)*x + (-2 : ℝ)*y + 0 = 0)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y = 0) :
  ∃ m : ℝ, m = 0 :=
sorry

end find_m_value_l510_510442


namespace pythagorean_triples_count_l510_510254

def is_pythagorean_triple (x y z : ℕ) : Prop :=
  x^2 + y^2 = z^2 ∧ x < y

def appears_in_distinct_triples (k : ℕ) (n : ℕ) : Prop :=
  ∃ (triples : finset (ℕ × ℕ × ℕ)), 
    (∀ t ∈ triples, is_pythagorean_triple t.1 t.2 t.2) ∧
    (triples.filter (λ t, t.1 = k ∨ t.2 = k ∨ t.2 = k)).card = n

theorem pythagorean_triples_count (n : ℕ) :
  appears_in_distinct_triples (2 ^ (n + 1)) n :=
sorry

end pythagorean_triples_count_l510_510254


namespace racecar_fixing_cost_l510_510473

theorem racecar_fixing_cost (C : ℝ) 
  (discounted_cost : C * 0.8)
  (prize : ℝ := 70000)
  (keeps : prize * 0.9)
  (earnings : keeps - discounted_cost) :=
  earnings = 47000 → C = 20000 :=
by
  sorry

end racecar_fixing_cost_l510_510473


namespace angle_quadrant_l510_510994

def angle := Real

-- Define the function to calculate the equivalent angle in the standard [0, 360) range.
def angle_mod360 (θ : angle) : angle :=
  θ % 360

-- Define a function to check and determine the quadrant
def quadrant (θ : angle) : String :=
  let θ_mod := angle_mod360 θ in
  if θ_mod ≥ 0 ∧ θ_mod < 90 then "First"
  else if θ_mod ≥ 90 ∧ θ_mod < 180 then "Second"
  else if θ_mod ≥ 180 ∧ θ_mod < 270 then "Third"
  else if θ_mod ≥ 270 ∧ θ_mod < 360 then "Fourth"
  else "Unknown"

-- The theorem statement
theorem angle_quadrant (θ : angle) (h : θ = 3600.5) : quadrant θ = "First" :=
by
  rw [h]
  sorry

end angle_quadrant_l510_510994


namespace find_length_EF_l510_510160

-- Define the input parameters
variables {a : ℝ} (ABCD : regular_tetrahedron a)
variables (E F : point)
variables (hE : E ∈ edge AB)
variables (hF : F ∈ edge CD)
variables (M N : point)
variables (circumsphere_intersects : intersects_circumsphere ABCD E F M N)
variables (ME_EF_FN_ratio : ME_EF_FN_Ratio M E EF F N 3 12 4)

noncomputable def length_EF : ℝ :=
  (2 * a) / real.sqrt 7

-- Statement to be proved
theorem find_length_EF {a : ℝ} (ABCD : regular_tetrahedron a) 
                        (E F : point)
                        (hE : E ∈ edge AB) 
                        (hF : F ∈ edge CD) 
                        (M N : point)
                        (circumsphere_intersects : intersects_circumsphere ABCD E F M N) 
                        (ME_EF_FN_ratio : ME_EF_FN_Ratio M E EF F N 3 12 4) : 
      length_EF = (2 * a) / real.sqrt 7 :=
sorry

end find_length_EF_l510_510160


namespace max_disjoint_regions_l510_510093

theorem max_disjoint_regions {p : ℕ} (hp : Nat.Prime p) (hp_ge3 : 3 ≤ p) : ∃ R, R = 3 * p^2 - 3 * p + 1 :=
by
  sorry

end max_disjoint_regions_l510_510093


namespace deck_of_1000_transformable_l510_510648

def shuffle (n : ℕ) (deck : List ℕ) : List ℕ :=
  -- Definition of the shuffle operation as described in the problem
  sorry

noncomputable def transformable_in_56_shuffles (n : ℕ) : Prop :=
  ∀ (initial final : List ℕ) (h₁ : initial.length = n) (h₂ : final.length = n),
  -- Prove that any initial arrangement can be transformed to any final arrangement in at most 56 shuffles
  sorry

theorem deck_of_1000_transformable : transformable_in_56_shuffles 1000 :=
  -- Implement the proof here
  sorry

end deck_of_1000_transformable_l510_510648


namespace interest_rate_per_annum_l510_510155

-- Given conditions
variables (BG TD t : ℝ) (FV r : ℝ)
axiom bg_eq : BG = 6
axiom td_eq : TD = 50
axiom t_eq : t = 1
axiom bankers_gain_eq : BG = FV * r * t - (FV - TD) * r * t

-- Proof problem
theorem interest_rate_per_annum : r = 0.12 :=
by sorry

end interest_rate_per_annum_l510_510155


namespace correlation_and_regression_l510_510228

noncomputable def mean (xs : List ℝ) : ℝ :=
  (List.sum xs) / (xs.length)

def squared_diff (xs : List ℝ) (mean : ℝ) : ℝ :=
  List.sum (xs.map (λ x => (x - mean)^2))

def covariance (xs ys : List ℝ) (x_mean y_mean : ℝ) : ℝ :=
  List.sum ((List.zipWith (λ x y => (x - x_mean) * (y - y_mean)) xs ys))

def correlation_coefficient (xs ys : List ℝ) : ℝ :=
  let x_mean := mean xs
  let y_mean := mean ys
  covariance xs ys x_mean y_mean / 
    (Real.sqrt (squared_diff xs x_mean * squared_diff ys y_mean))

noncomputable def regression_coefficient_b (xs ys : List ℝ) : ℝ :=
  let x_mean := mean xs
  let y_mean := mean ys
  covariance xs ys x_mean y_mean / squared_diff xs x_mean

noncomputable def regression_coefficient_a (xs ys : List ℝ) : ℝ :=
  let x_mean := mean xs
  let y_mean := mean ys
  y_mean - (regression_coefficient_b xs ys * x_mean)

theorem correlation_and_regression (xs ys : List ℝ) :
  let x_data := [1.0, 2.0, 3.0, 4.0, 5.0]
  let y_data := [9.5, 8.6, 7.8, 7.0, 6.1]
  mean x_data = 3.0 ∧ mean y_data = 7.8 ∧
  squared_diff x_data (mean x_data) = 10.0 ∧
  squared_diff y_data (mean y_data) = 7.06 ∧
  covariance x_data y_data (mean x_data) (mean y_data) = -8.4 ∧
  correlation_coefficient x_data y_data ≈ -1.0 ∧
  regression_coefficient_b x_data y_data ≈ -0.84 ∧
  regression_coefficient_a x_data y_data ≈ 10.32 :=
  sorry

end correlation_and_regression_l510_510228


namespace proof_contains_rectangle_and_triangle_l510_510854

noncomputable def contains_rectangle_and_triangle (S : Finset (ℝ × ℝ)) : Prop :=
  (∃ (x_min x_max y_min y_max : ℝ),
    (∀ p ∈ S, x_min ≤ p.1 ∧ p.1 ≤ x_max ∧ y_min ≤ p.2 ∧ p.2 ≤ y_max) ∧
    (x_max - x_min) * (y_max - y_min) ≤ 4) ∧
  (∃ (A B C : ℝ × ℝ),
    (A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧
    (∀ D ∈ S, ∃ α β γ : ℝ,
      0 ≤ α ∧ 0 ≤ β ∧ 0 ≤ γ ∧ α + β + γ = 1 ∧
      (α • A.1 + β • B.1 + γ • C.1 = D.1) ∧
      (α • A.2 + β • B.2 + γ • C.2 = D.2)) ∧
    (1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) ≤ 1) ∧
    (4 * 1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))) ≤ 4)

theorem proof_contains_rectangle_and_triangle (S : Finset (ℝ × ℝ))
  (h_condition : ∀ A B C ∈ S, 1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) ≤ 1) :
  contains_rectangle_and_triangle S := sorry

end proof_contains_rectangle_and_triangle_l510_510854


namespace greatest_positive_integer_x_l510_510202

theorem greatest_positive_integer_x (x : ℕ) (h₁ : x^2 < 12) (h₂ : ∀ y: ℕ, y^2 < 12 → y ≤ x) : 
  x = 3 := 
by
  sorry

end greatest_positive_integer_x_l510_510202


namespace find_income_l510_510978

theorem find_income :
  ∃ (I : ℝ), 
    (0.5527125 * I = 5548) ∧ 
    (I = 10038.46) := 
begin
  sorry
end

end find_income_l510_510978


namespace cone_surface_area_l510_510157

theorem cone_surface_area (l : ℝ) (α : ℝ) :
  S = \frac{2 * real.pi * l^2 * real.cot (α / 2)}{9 * real.sin (2 * α)} :=
sorry

end cone_surface_area_l510_510157


namespace ernesto_distance_more_prove_l510_510888

noncomputable def renaldo_distance := 15
noncomputable def ernesto_distance_more_than_third_renaldo := x : ℕ
noncomputable def total_distance := 27
noncomputable def one_third_renaldo_distance := renaldo_distance / 3

theorem ernesto_distance_more_prove  :
  ∃ (x : ℕ), ernesto_distance_more_than_third_renaldo + one_third_renaldo_distance = 7 :=
by
  sorry

end ernesto_distance_more_prove_l510_510888


namespace nonneg_diff_roots_l510_510583

theorem nonneg_diff_roots : 
  ∀ (a b c : ℤ), a = 1 → b = 42 → c = 384 → 
  let Δ = b * b - 4 * a * c in
  Δ ≥ 0 → 
  let r1 := (-b + Δ.sqrt) / (2 * a) in
  let r2 := (-b - Δ.sqrt) / (2 * a) in
  abs (r1 - r2) = 8 :=
by
  sorry

end nonneg_diff_roots_l510_510583


namespace ellipse_eccentricity_l510_510019

theorem ellipse_eccentricity :
  (∃ (e : ℝ), (∀ (x y : ℝ), ((x^2 / 9) + y^2 = 1) → (e = 2 * Real.sqrt 2 / 3))) :=
by
  sorry

end ellipse_eccentricity_l510_510019


namespace value_of_y_l510_510043

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 2) (h2 : x = -6) : y = 38 :=
by
  sorry

end value_of_y_l510_510043


namespace max_red_beads_l510_510272

theorem max_red_beads (n : ℕ)
  (total_beads : n = 150)
  (at_least_one_green_in_every_six : ∀ (s : ℕ), s < n - 5 → (∃ i, s ≤ i ∧ i < s + 6 ∧ green i))
  (at_least_one_blue_in_every_eleven : ∀ (s : ℕ), s < n - 10 → (∃ i, s ≤ i ∧ i < s + 11 ∧ blue i)) :
  ∃ red_count, red_count ≤ 112 :=
by sorry

end max_red_beads_l510_510272


namespace solve_equation_l510_510930

theorem solve_equation (x : ℝ) (h : x ≠ 1) :
  (3 * x) / (x - 1) = 2 + 1 / (x - 1) → x = -1 :=
by
  sorry

end solve_equation_l510_510930


namespace number_of_eggs_l510_510895

def price_of_oranges (O : ℝ) : ℝ := 5 * O
def new_price_of_eggs (E : ℝ) : ℝ := E * 1.03
def new_price_of_oranges (O : ℝ) : ℝ := O * 1.06
def cost_difference (initial_E : ℝ) (initial_O : ℝ): ℝ := ((new_price_of_eggs initial_E) - initial_E) + ((new_price_of_oranges initial_O * 10) - (initial_O * 5))

theorem number_of_eggs : 
  ∀ (E O : ℝ), 
  E = 5 * O → 
  cost_difference E O = 8.999999999999986 → 
  E / (price_of_oranges O) = 5 := 
by
  intros E O hEO hCostDiff
  rw [price_of_oranges, hEO] at hCostDiff
  have := congr_arg (λ x, 8.999999999999986) hCostDiff
  sorry

end number_of_eggs_l510_510895


namespace product_of_third_and_fourth_numbers_l510_510154

theorem product_of_third_and_fourth_numbers 
  (mean : ℕ → ℕ → ℕ → ℕ → ℕ := λ a b c d, (a + b + c + d) / 4)
  (h_mean : mean 14 25 c d = 20)
  (h_third_less : c = d - 3) : c * d = 418 :=
sorry

end product_of_third_and_fourth_numbers_l510_510154


namespace total_bags_l510_510946

-- Definitions based on the conditions
def bags_on_monday : ℕ := 4
def bags_next_day : ℕ := 8

-- Theorem statement
theorem total_bags : bags_on_monday + bags_next_day = 12 :=
by
  -- Proof will be added here
  sorry

end total_bags_l510_510946


namespace rod_length_l510_510641

theorem rod_length (num_pieces : ℝ) (length_per_piece : ℝ) (h1 : num_pieces = 118.75) (h2 : length_per_piece = 0.40) : 
  num_pieces * length_per_piece = 47.5 := by
  sorry

end rod_length_l510_510641


namespace last_two_digits_l510_510542

theorem last_two_digits (a b : ℕ) (n : ℕ) (h : b ≡ 25 [MOD 100]) (h_pow : (25 : ℕ) ^ n ≡ 25 [MOD 100]) :
  (33 * b ^ n) % 100 = 25 :=
by
  sorry

end last_two_digits_l510_510542


namespace find_age_l510_510636

theorem find_age (A : ℤ) (h : 4 * (A + 4) - 4 * (A - 4) = A) : A = 32 :=
by sorry

end find_age_l510_510636


namespace subsets_with_mean_of_five_l510_510421

def numbers : set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem subsets_with_mean_of_five :
  ∃ (s : set (set ℕ)), 
    s.card = 4 ∧
    (∀ p ∈ s, p.card = 2 ∧ (∑ x in p, x) = 10) ∧
    ∀ (p ∈ s), ∑ x in numbers \ p, x / (numbers.card - p.card) = 5 :=
by
  sorry

end subsets_with_mean_of_five_l510_510421


namespace estimated_probability_l510_510006

noncomputable def wind_levels : Set ℕ := {1, 2, 3, 4, 5, 6}

noncomputable def wind_statistics : List ℕ := [342, 136, 556, 461, 336, 516, 225, 213, 112, 341]

noncomputable def total_groups : ℕ := 10

noncomputable def qualifying_groups : ℕ := 3

-- Question: Prove that the probability of having at least two days with wind levels of 4 or above in every three days is equal to 0.37.
theorem estimated_probability : qualifying_groups / total_groups = 0.37 := sorry

end estimated_probability_l510_510006


namespace ratio_owners_riding_to_total_l510_510621

theorem ratio_owners_riding_to_total (h_num_legs : 70 = 4 * (14 - W) + 6 * W) (h_total : 14 = W + (14 - W)) :
  (14 - W) / 14 = 1 / 2 :=
by
  sorry

end ratio_owners_riding_to_total_l510_510621


namespace number_of_regular_polyhedra_l510_510941

-- Definition: Regular polyhedra are polyhedra with congruent faces of regular polygons and the same number of faces meeting at each vertex.
def regular_polyhedron (P : Type) : Prop :=
  ∃ (faces : list (set P)), (∀ (f : set P), f ∈ faces → is_regular_polygon f) ∧ (∀ (v : P), ∃ (n : ℕ), number_of_faces_meeting_at_vertex v = n)

-- Theorem: There are exactly 5 kinds of regular polyhedra.
theorem number_of_regular_polyhedra : ∃ (n : ℕ), n = 5 :=
by
  sorry

end number_of_regular_polyhedra_l510_510941


namespace jade_transactions_l510_510240

theorem jade_transactions :
  ∀ (transactions_mabel transactions_anthony transactions_cal transactions_jade : ℕ),
    transactions_mabel = 90 →
    transactions_anthony = transactions_mabel + transactions_mabel / 10 →
    transactions_cal = (transactions_anthony * 2) / 3 →
    transactions_jade = transactions_cal + 19 →
    transactions_jade = 85 :=
by
  intros transactions_mabel transactions_anthony transactions_cal transactions_jade
  intros h_mabel h_anthony h_cal h_jade
  sorry

end jade_transactions_l510_510240


namespace solve_for_y_l510_510522

theorem solve_for_y (y : ℤ) (h : (y ≠ 2) → ((y^2 - 10*y + 24)/(y-2) + (4*y^2 + 8*y - 48)/(4*y - 8) = 0)) : y = 0 :=
by
  sorry

end solve_for_y_l510_510522


namespace part1_part2_l510_510021

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sin x + Real.cos x)

theorem part1 : f (Real.pi / 4) = 2 := sorry

theorem part2 : ∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8 → 
  (2 * Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4) > 0) := sorry

end part1_part2_l510_510021


namespace basketball_team_total_games_l510_510997

theorem basketball_team_total_games
  (games_won_first_40 : ℕ)
  (total_games : ℕ)
  (remaining_games : ℕ)
  (H1 : games_won_first_40 = 14)
  (H2 : 0.55 * total_games = 14 + 0.70 * remaining_games)
  (H3 : total_games = 40 + remaining_games)
  : total_games = 94 := 
by
  sorry

end basketball_team_total_games_l510_510997


namespace negative_number_among_options_l510_510665

theorem negative_number_among_options :
  let A := abs (-1)
  let B := -(2^2)
  let C := (-(Real.sqrt 3))^2
  let D := (-3)^0
  B < 0 ∧ A > 0 ∧ C > 0 ∧ D > 0 :=
by
  sorry

end negative_number_among_options_l510_510665


namespace smaller_angle_at_5_15_l510_510578

def degrees_per_hour : ℝ := 30
def degrees_per_minute : ℝ := 6
def hour_position_at_5 : ℝ := 5 * degrees_per_hour
def hour_hand_movement_5_15 : ℝ := (15 / 60) * degrees_per_hour
def final_hour_position : ℝ := hour_position_at_5 + hour_hand_movement_5_15
def minute_position_at_15 : ℝ := 15 * degrees_per_minute

theorem smaller_angle_at_5_15 :
  abs (final_hour_position - minute_position_at_15) = 67.5 :=
by
  sorry

end smaller_angle_at_5_15_l510_510578


namespace area_of_quadrilateral_is_195_l510_510712

-- Definitions and conditions
def diagonal_length : ℝ := 26
def offset1 : ℝ := 9
def offset2 : ℝ := 6

-- Prove the area of the quadrilateral is 195 cm²
theorem area_of_quadrilateral_is_195 :
  1 / 2 * diagonal_length * offset1 + 1 / 2 * diagonal_length * offset2 = 195 := 
by
  -- The proof steps would go here
  sorry

end area_of_quadrilateral_is_195_l510_510712


namespace roots_of_quadratic_polynomial_neg_l510_510926

variable {R : Type*} [LinearOrderedField R]

theorem roots_of_quadratic_polynomial_neg (f : R → R)
  (h_distinct_roots : ∃ x y : R, x ≠ y ∧ f(x) = 0 ∧ f(y) = 0)
  (h_inequality : ∀ a b : R, f(a^2 + b^2) ≥ f(2 * a * b)) :
  ∃ x : R, f(x) = 0 ∧ x < 0 := 
sorry

end roots_of_quadratic_polynomial_neg_l510_510926


namespace candy_per_smaller_bag_l510_510730

-- Define the variables and parameters
def george_candy : ℕ := 648
def friends : ℕ := 3
def total_people : ℕ := friends + 1
def smaller_bags : ℕ := 8

-- Define the theorem
theorem candy_per_smaller_bag : (george_candy / total_people) / smaller_bags = 20 :=
by
  -- Assume the proof steps, not required to actually complete
  sorry

end candy_per_smaller_bag_l510_510730


namespace finite_operations_result_in_zero_l510_510247

theorem finite_operations_result_in_zero (r_1 r_2 r_3 : ℝ) (a_1 a_2 a_3 : ℤ) (h_nonneg : r_1 ≥ 0 ∧ r_2 ≥ 0 ∧ r_3 ≥ 0) (h_combination : a_1 * r_1 + a_2 * r_2 + a_3 * r_3 = 0) (h_nonzero: a_1 ≠ 0 ∨ a_2 ≠ 0 ∨ a_3 ≠ 0):
  ∃ n : ℕ, ∃ r_1' r_2' r_3' : ℝ, (∃ i : {i // i ≤ n}, r_1' = 0 ∨ r_2' = 0 ∨ r_3' = 0) ∧
  ∀ (x y : ℝ), x ∈ {r_1', r_2', r_3'} → y ∈ {r_1', r_2', r_3'} → x ≤ y → ∃ r_1'' r_2'' r_3'' : ℝ, {r_1'', r_2'', r_3''} = ({r_1', r_2', r_3'} \ {y}) ∪ {y - x} := 
  sorry

end finite_operations_result_in_zero_l510_510247


namespace evaluate_expression_l510_510704

theorem evaluate_expression (i : ℂ) (h : i^4 = 1) : i^8 + i^{20} + i^{-32} = 3 := by
  sorry

end evaluate_expression_l510_510704


namespace average_episodes_per_year_l510_510610

def num_years : Nat := 14
def seasons_15_episodes : Nat := 8
def episodes_per_season_15 : Nat := 15
def seasons_20_episodes : Nat := 4
def episodes_per_season_20 : Nat := 20
def seasons_12_episodes : Nat := 2
def episodes_per_season_12 : Nat := 12

theorem average_episodes_per_year :
  (seasons_15_episodes * episodes_per_season_15 +
   seasons_20_episodes * episodes_per_season_20 +
   seasons_12_episodes * episodes_per_season_12) / num_years = 16 := by
  sorry

end average_episodes_per_year_l510_510610


namespace avg_last_three_numbers_l510_510902

-- Definitions of conditions
def avg_seven_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.sum / 7 = 60)

def avg_first_four_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.take 4).sum / 4 = 55

-- Proof statement
theorem avg_last_three_numbers (numbers : List ℝ) (h_len : numbers.length = 7)
  (h1 : avg_seven_numbers numbers h_len)
  (h2 : avg_first_four_numbers numbers h_len) :
  (numbers.drop 4).sum / 3 = 200 / 3 :=
sorry

end avg_last_three_numbers_l510_510902


namespace circulation_A_circle_zero_l510_510309

-- Define the vector field A under the given conditions
def A (x y z : ℝ) : ℝ × ℝ × ℝ := (x^2 * y^2, 1, z)

-- Circle and plane conditions
def circle (x y : ℝ) (a : ℝ) : Prop := x^2 + y^2 = a^2
def plane (z : ℝ) : Prop := z = 0

theorem circulation_A_circle_zero (a : ℝ) :
  ∫∫ x y, (circle x y a ∧ plane 0) →
    -2 * x^2 * y * dx * dy = 0 := sorry

end circulation_A_circle_zero_l510_510309


namespace vertex_on_ho_circle_l510_510191

variables {A B C H O I : Type} 
variables [IsAcuteAngledTriangle A B C] [IsNotIsosceles A B C]
variables [IsOrthocenter H A B C] [IsCircumcenter O A B C] [IsIncenter I A B C]

theorem vertex_on_ho_circle (hA : LiesOnCircle A H O I) : ∃ B', LiesOnCircle B' H O I :=
sorry

end vertex_on_ho_circle_l510_510191


namespace exp_form_theta_l510_510592

noncomputable def modulus (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

noncomputable def argument (x y : ℝ) : ℝ := 
  let θ1 := Real.arccos (x / (modulus x y))
  let θ :=
    if y < 0 then
      if x < 0 then θ1 + π else -θ1
    else θ1
  θ

theorem exp_form_theta {x y : ℝ} (h : x = 2) (h_ : y = -2 * Real.sqrt 2) (h_range : (-π < argument x y) ∧ (argument x y ≤ π)) :
  argument x y = -5/4 * π :=
by
  rw [h, h_]
  sorry

end exp_form_theta_l510_510592


namespace total_selling_price_of_cloth_l510_510651

theorem total_selling_price_of_cloth
  (profit_per_meter : ℕ)
  (cost_price_per_meter : ℕ)
  (total_meters : ℕ)
  (total_selling_price : ℕ) :
  profit_per_meter = 7 →
  cost_price_per_meter = 118 →
  total_meters = 80 →
  total_selling_price = (cost_price_per_meter + profit_per_meter) * total_meters →
  total_selling_price = 10000 :=
by
  intros h_profit h_cost h_total h_selling_price
  rw [h_profit, h_cost, h_total] at h_selling_price
  exact h_selling_price

end total_selling_price_of_cloth_l510_510651


namespace increasing_on_interval_0_infty_l510_510664

noncomputable def f (x : ℝ) : ℝ := -Real.log x / Real.log 2
noncomputable def g (x : ℝ) : ℝ := -1 / Real.sqrt (x + 1)
noncomputable def h (x : ℝ) : ℝ := (1 / 2) ^ x
noncomputable def k (x : ℝ) : ℝ := 2 * x + 1 / x

theorem increasing_on_interval_0_infty : 
  (∀ x > 0, g x > g (x - ε) ∧ 
  (∀ x > 0, f x < f (x - ε)) ∧ 
  (∀ x > 0, h x < h (x - ε)) ∧ 
  (∃ x1 x2 > 0, x1 ≠ x2 ∧ (k x1 > k (x1 - ε) ∧ k x2 < k (x2 - ε)))) := 
  sorry

end increasing_on_interval_0_infty_l510_510664


namespace white_balls_count_l510_510629

theorem white_balls_count (W B R : ℕ) (h1 : B = W + 14) (h2 : R = 3 * (B - W)) (h3 : W + B + R = 1000) : W = 472 :=
sorry

end white_balls_count_l510_510629


namespace max_red_beads_l510_510274

-- Define the structure of the beads string
structure Beads (n : Nat) :=
  (total_beads : Nat)
  (green_condition : ∀ (i : Nat), i < total_beads - 5 → ∃ (j : Nat), j ∈ (i..(i+6)) ∧ isGreen j)
  (blue_condition : ∀ (i : Nat), i < total_beads - 10 → ∃ (j : Nat), j ∈ (i..(i+11)) ∧ isBlue j)
  (count_beads : Nat)

-- Define the necessary conditions
def beads_string : Beads 150 :=
  { total_beads := 150,
    green_condition := sorry,
    blue_condition := sorry,
    count_beads := sorry }

-- Define the maximum number of red beads proof
theorem max_red_beads : ∃ red_beads : Nat, red_beads = 112 :=
  sorry

end max_red_beads_l510_510274


namespace average_episodes_per_year_l510_510611

def num_years : Nat := 14
def seasons_15_episodes : Nat := 8
def episodes_per_season_15 : Nat := 15
def seasons_20_episodes : Nat := 4
def episodes_per_season_20 : Nat := 20
def seasons_12_episodes : Nat := 2
def episodes_per_season_12 : Nat := 12

theorem average_episodes_per_year :
  (seasons_15_episodes * episodes_per_season_15 +
   seasons_20_episodes * episodes_per_season_20 +
   seasons_12_episodes * episodes_per_season_12) / num_years = 16 := by
  sorry

end average_episodes_per_year_l510_510611


namespace positive_difference_perimeters_l510_510609

theorem positive_difference_perimeters :
  ∀ (r : ℝ) (s : ℝ) (t : ℝ) (u : ℝ), 
  ((r = 2 <-> s = 12 <-> t = 2 <-> u = 12) ∨ 
   (r = 6 <-> s = 2 <-> t = 6 <-> u = 2) ∨ 
   (r = 3 <-> s = 6 <-> t = 3 <-> u = 6) ∨ 
   (r = 4 <-> s = 3 <-> t = 4 <-> u = 3)) → 
  6 * r * s = 6 * 12 ∧ 6 * (2 * (r + s)) = 14 := sorry

end positive_difference_perimeters_l510_510609


namespace loan_repayment_l510_510623

open Real

theorem loan_repayment
  (a r : ℝ) (h_r : 0 ≤ r) :
  ∃ x : ℝ, 
    x = (a * r * (1 + r)^5) / ((1 + r)^5 - 1) :=
sorry

end loan_repayment_l510_510623


namespace negation_of_exists_l510_510919

theorem negation_of_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 2 > 0) = ∀ x : ℝ, x^2 - x + 2 ≤ 0 := by
  sorry

end negation_of_exists_l510_510919


namespace minimum_races_to_find_top3_l510_510129

-- Define a constant to represent the number of horses and maximum horses per race
def total_horses : ℕ := 25
def max_horses_per_race : ℕ := 5

-- Define the problem statement as a theorem
theorem minimum_races_to_find_top3 (total_horses : ℕ) (max_horses_per_race : ℕ) : ℕ :=
  if total_horses = 25 ∧ max_horses_per_race = 5 then 7 else sorry

end minimum_races_to_find_top3_l510_510129


namespace Hendrix_class_end_of_year_students_l510_510212

def initial_students := 160
def new_students := 20
def fraction_transferred := 1 / 3

theorem Hendrix_class_end_of_year_students : 
  let total_students := initial_students + new_students in
  let transferred_students := fraction_transferred * total_students in
  let final_students := total_students - transferred_students in
  final_students = 120 :=
by
  sorry

end Hendrix_class_end_of_year_students_l510_510212


namespace wendy_score_l510_510192

def score_per_treasure : ℕ := 5
def treasures_first_level : ℕ := 4
def treasures_second_level : ℕ := 3

theorem wendy_score :
  score_per_treasure * treasures_first_level + score_per_treasure * treasures_second_level = 35 :=
by
  sorry

end wendy_score_l510_510192


namespace parabola_x_intercepts_l510_510791

theorem parabola_x_intercepts (a b c : ℝ) (h : a = 3 ∧ b = -4 ∧ c = 1) : 
  let discriminant := b^2 - 4*a*c in 
  discriminant > 0 → 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (3 * x1^2 - 4 * x1 + 1 = 0) ∧ (3 * x2^2 - 4 * x2 + 1 = 0) :=
by
  intros
  sorry

end parabola_x_intercepts_l510_510791


namespace range_of_a_l510_510799

open Real

noncomputable def increasing_log (a : ℝ) (f : ℝ → ℝ) : Prop := ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

theorem range_of_a (a : ℝ) :
  (1 < a ∧ a < 2) ↔ increasing_log a (λ x, log a (a * x + 2)) ∧ ∀ x, x ≥ -1 → a * x + 2 > 0 := 
by
  sorry

end range_of_a_l510_510799


namespace part_one_part_two_l510_510249

variable {a : ℕ → ℝ} {S : ℕ → ℝ} {λ : ℝ}

theorem part_one (h₁ : ∀ n, a n * a (n + 1) = λ * S n - 1) (h₂ : a 1 = 1) (h₃ : ∀ n, a n ≠ 0) :
  ∀ n, a (n + 2) - a n = λ :=
sorry

theorem part_two (h₁ : ∀ n, a n * a (n + 1) = λ * S n - 1) (h₂ : a 1 = 1) (h₃ : ∀ n, a n ≠ 0) :
  (∀ n, a (n + 2) - a n = 4) → ∀ λ, λ = 4 :=
sorry

end part_one_part_two_l510_510249


namespace product_geq_n_minus_1_power_n_l510_510108

theorem product_geq_n_minus_1_power_n {n : ℕ} (x : Fin n → ℝ) (h : ∀ i, 0 < x i) (h_sum : ∑ i, x i = 1) :
  (∏ i, (1 / x i - 1)) ≥ (n - 1) ^ n :=
sorry

end product_geq_n_minus_1_power_n_l510_510108


namespace pascal_triangle_eighth_row_l510_510450

def sum_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose (n-1) (k-1) 

theorem pascal_triangle_eighth_row:
  sum_interior_numbers 8 = 126 ∧ binomial_coefficient 8 3 = 21 :=
by
  sorry

end pascal_triangle_eighth_row_l510_510450


namespace largest_value_of_c_l510_510342

theorem largest_value_of_c : 
  ∀ x : ℝ, (c = 2 - x + 2 * real.sqrt (x - 1)) ∧ (x > 1) → (c ≤ 2) :=
sorry

end largest_value_of_c_l510_510342


namespace points_for_a_eq_3_points_for_a_eq_2_l510_510225

variable {a x y : ℝ}

def fractional_part_mod_3 (z : ℝ) : ℝ := z - 3 * floor (z / 3)

theorem points_for_a_eq_3:
  ∀ x y : ℝ, 
    fractional_part_mod_3 x + fractional_part_mod_3 y = 3 ↔ 
    (1 ≤ fractional_part_mod_3 x ∧ fractional_part_mod_3 x < 2 ∧ 2 ≤ fractional_part_mod_3 y ∧ fractional_part_mod_3 y < 3) ∨
    (2 ≤ fractional_part_mod_3 x ∧ fractional_part_mod_3 x < 3 ∧ 1 ≤ fractional_part_mod_3 y ∧ fractional_part_mod_3 y < 2) :=
by sorry

theorem points_for_a_eq_2:
  ∀ x y : ℝ, 
    fractional_part_mod_3 x + fractional_part_mod_3 y = 2 ↔ 
    (0 ≤ fractional_part_mod_3 x ∧ fractional_part_mod_3 x < 1 ∧ 2 ≤ fractional_part_mod_3 y ∧ fractional_part_mod_3 y < 3) ∨
    (1 ≤ fractional_part_mod_3 x ∧ fractional_part_mod_3 x < 2 ∧ 1 ≤ fractional_part_mod_3 y ∧ fractional_part_mod_3 y < 2) ∨
    (2 ≤ fractional_part_mod_3 x ∧ fractional_part_mod_3 x < 3 ∧ 0 ≤ fractional_part_mod_3 y ∧ fractional_part_mod_3 y < 1) :=
by sorry

end points_for_a_eq_3_points_for_a_eq_2_l510_510225


namespace tangent_line_range_of_a_positive_range_of_a_negative_positive_l510_510777

-- Definitions and Conditions
def f (x: ℝ) := (1/2) * x^2
def g (x: ℝ) (a: ℝ) := a * Real.log x
def h (x: ℝ) (a: ℝ) := f x + g x a

-- Proof Problems

-- 1. The tangent line of the curve y = f(x) - g(x, a) at x = 1 is 6x - 2y - 5 = 0, prove a = -2
theorem tangent_line (a: ℝ) : (∃ t: ℝ, t = a ∧ (∀ x: ℝ, (6:ℝ)*x = (2:ℝ)*(f x - g x a) + 5)) -> a = -2 :=
sorry

-- 2. For any two distinct positive numbers x₁ and x₂, ∀ x₁ x₂ ∈ ℝ, x₁ ≠ x₂, (h(x₁, a) - h(x₂, a)) / (x₁ - x₂) > 2, prove a ≥ 1
theorem range_of_a_positive (a: ℝ) : (∀ x₁ x₂: ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 -> (h x₁ a - h x₂ a) / (x₁ - x₂) > 2) -> a ≥ 1 :=
sorry

-- 3. There exists an x₀ ∈ [1,e] such that f'(x₀) + 1/f'(x₀) < g(x₀, a) - g'(x₀, a), prove range of a is (-∞, -2) ∪ ( (e^2 + 1) / (e - 1), ∞)
theorem range_of_a_negative_positive (a: ℝ) : (
  ∃ x₀: ℝ, (1 ≤ x₀ ∧ x₀ ≤ Real.exp 1) ∧ (f' x₀ + 1 / f' x₀ < g x₀ a - (g' x₀ a)) 
) -> a < -2 ∨ a > (Real.exp 2 + 1) / (Real.exp 1 - 1) :=
sorry

end tangent_line_range_of_a_positive_range_of_a_negative_positive_l510_510777


namespace different_languages_comb_same_language_comb_total_comb_l510_510175

namespace BookSelection

def numberOfJapaneseBooks : ℕ := 5
def numberOfEnglishBooks : ℕ := 7
def numberOfChineseBooks : ℕ := 10

def differentLanguagesCombinations : ℕ :=
  (numberOfJapaneseBooks * numberOfEnglishBooks) +
  (numberOfChineseBooks * numberOfJapaneseBooks) +
  (numberOfChineseBooks * numberOfEnglishBooks)

def sameLanguageCombinations : ℕ :=
  (numberOfChineseBooks * (numberOfChineseBooks - 1)) / 2 +
  (numberOfEnglishBooks * (numberOfEnglishBooks - 1)) / 2 +
  (numberOfJapaneseBooks * (numberOfJapaneseBooks - 1)) / 2

def totalCombinations : ℕ := (numberOfChineseBooks + numberOfJapaneseBooks + numberOfEnglishBooks) * ((numberOfChineseBooks + numberOfJapaneseBooks + numberOfEnglishBooks) - 1) / 2

theorem different_languages_comb : differentLanguagesCombinations = 155 := by
  sorry

theorem same_language_comb : sameLanguageCombinations = 76 := by
  sorry

theorem total_comb : totalCombinations = 231 := by
  sorry

end BookSelection

end different_languages_comb_same_language_comb_total_comb_l510_510175


namespace real_condition_proof_l510_510864

noncomputable def real_condition_sufficient_but_not_necessary : Prop := 
∀ x : ℝ, (|x - 2| < 1) → ((x^2 + x - 2) > 0) ∧ (¬ ( ∀ y : ℝ, (y^2 + y - 2) > 0 → |y - 2| < 1))

theorem real_condition_proof : real_condition_sufficient_but_not_necessary :=
by
  sorry

end real_condition_proof_l510_510864


namespace probability_of_at_most_one_rainy_day_l510_510923

noncomputable def rain_probability : ℝ := 1 / 20

def rainy_days_in_July (days : ℕ) (prob_rain : ℝ) : ℕ → ℝ 
| 0 => (1 - prob_rain) ^ days
| 1 => days * prob_rain * (1 - prob_rain) ^ (days - 1)
| _ => 0

theorem probability_of_at_most_one_rainy_day : 
  abs (rainy_days_in_July 31 rain_probability 0 + rainy_days_in_July 31 rain_probability 1 - 0.271) < 0.001 :=
by
  -- Proof would go here; using sorry for now
  sorry

end probability_of_at_most_one_rainy_day_l510_510923


namespace read_decimal_11246_23_l510_510449

theorem read_decimal_11246_23 :
  ∀ (d : ℝ), d = 11246.23 → decimal_reading d = "eleven thousand two hundred forty-six point two three" :=
by
  -- Given that the decimal d is indeed 11246.23, we need to show its reading.
  intro d h
  sorry

end read_decimal_11246_23_l510_510449


namespace parabola_focus_coordinates_l510_510317

theorem parabola_focus_coordinates : 
  ∀ (x y : ℝ), y = 4 * x^2 → (0, y / 16) = (0, 1 / 16) :=
by
  intros x y h
  sorry

end parabola_focus_coordinates_l510_510317


namespace sum_even_difference_l510_510236

def sum_even (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  in n * (a + b) / 2

-- Given i is the sum of even integers from 2 to 224
def i : ℕ := sum_even 2 224

-- Given k is the sum of even integers from 8 to 80
def k : ℕ := sum_even 8 80

-- Prove that i - k equals to 11028
theorem sum_even_difference : i - k = 11028 := by
  sorry

end sum_even_difference_l510_510236


namespace value_of_x_squared_plus_reciprocal_squared_l510_510392

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (h : x^4 + (1 / x^4) = 23) :
  x^2 + (1 / x^2) = 5 := by
  sorry

end value_of_x_squared_plus_reciprocal_squared_l510_510392


namespace angle_between_vectors_l510_510413

variables {ℝ : Type*} [real_vector_space ℝ] (a b : vector ℝ 2) -- Assuming a and b are 2D vectors

theorem angle_between_vectors (h1 : (a + b) ⬝ b = 7)
                                 (h2 : ∥a∥ = real.sqrt 3) 
                                 (h3 : ∥b∥ = 2) : 
    ∃ θ : ℝ, θ = real.pi / 6 :=
by {
    sorry
}

end angle_between_vectors_l510_510413


namespace joe_reaches_170_pounds_l510_510472

def joe_initial_weight : ℕ := 222
def months_until_now : ℕ := 5

def weight_changes : List ℤ := [-12, -6, 2, -4, -4]
def current_weight : ℕ := joe_initial_weight - 12 - 6 + 2 - 8

def monthly_pattern : List ℤ := [-12, -6, 2, -4, -4]
def weight_to_lose : ℕ := current_weight - 170

def cycle_loss (pattern : List ℤ) : ℤ :=
  List.sum pattern

def months_needed (loss_per_cycle additional_loss target_loss : ℕ) : ℕ :=
  (target_loss / loss_per_cycle) * 5 + additional_loss

theorem joe_reaches_170_pounds :
  months_needed (cycle_loss monthly_pattern) 1 weight_to_lose = 6 := 
  by
    -- Proof goes here
    sorry

end joe_reaches_170_pounds_l510_510472


namespace miller_rabin_probability_at_least_half_l510_510517

theorem miller_rabin_probability_at_least_half
  {n : ℕ} (hcomp : ¬Nat.Prime n) (s d : ℕ) (hd_odd : d % 2 = 1) (h_decomp : n - 1 = 2^s * d)
  (a : ℤ) (ha_range : 2 ≤ a ∧ a ≤ n - 2) :
  ∃ P : ℝ, P ≥ 1 / 2 ∧ ∀ a, (2 ≤ a ∧ a ≤ n - 2) → ¬(a^(d * 2^s) % n = 1)
  :=
sorry

end miller_rabin_probability_at_least_half_l510_510517


namespace like_terms_proof_l510_510751

variable (a b : ℤ)
variable (x y : ℤ)

theorem like_terms_proof (hx : x = 2) (hy : 3 = 1 - y) : x * y = -4 := by
  rw [hx, hy]
  sorry

end like_terms_proof_l510_510751


namespace smallest_t_l510_510478

variable {S : set ℝ} [convex ℝ S] [nonempty S] 

/-- B_S(K, t) is the band of points whose distance from the line parallel to K and midway
between the support lines L1 and L2 is at most (t/2) * w, where w is the distance
between L1 and L2. -/
def B_S (K : line ℝ) (t : ℝ) : set ℝ := 
  {x | ∃ p ∈ S, dist x p ≤ (t / 2) * (distance_between_support_lines S K)}

/-- The smallest t such that the intersection of S with B_S(K, t) is nonempty for all lines K. -/
theorem smallest_t (S : set ℝ) [convex ℝ S] [nonempty S] : ∃ t > 0, (∀ K : line ℝ, (S ∩ B_S K t).nonempty) :=
begin
  use [1 / 3, by norm_num],
  intros K,
  sorry
end

end smallest_t_l510_510478


namespace range_y2_l510_510393

theorem range_y2 (x y : ℝ) (x1 y1 x2 y2 : ℝ) (AB BC : ℝ × ℝ):
  ellipse_C1 (x y : ℝ) (h : x ^ 2 / 3 + y ^ 2 / 2 = 1) ->
  foci_F1_and_F2 : F1 = (-1, 0) ∧ F2 = (1, 0) ->
  line_l1 : x = -1 ->
  moving_line_l2 (t : ℝ) : y = t (P : ℝ × ℝ) : P = (-1, t) ->
  curve_C2 (y : ℝ) : C2 = y ^ 2 = 4 * x ->
  points_on_C2 (A B C : ℝ × ℝ) : A = (1, 2) ∧ B = ((1 / 4) * y1 ^ 2, y1) ∧ C = ((1 / 4) * y2 ^ 2, y2) ->
  perpendicular_vectors (AB BC : ℝ × ℝ) : AB = (x1 - 1, y1 - 2) ∧ BC = (x2 - x1, y2 - y1) ∧ (AB.1 * BC.1 + AB.2 * BC.2) = 0 -> 
  y2_range : y2 < -6 ∨ y2 ≥ 10 := 
sorry

end range_y2_l510_510393


namespace minimum_sum_of_distances_l510_510373

variable {V : Type*} [normed_group V] [normed_space ℝ V]

-- Definition of a regular pentagon in the plane
structure RegularPentagon (V : Type*) [normed_group V] [normed_space ℝ V] :=
(center : V)
(vertices : Fin 5 → V)
(side_length : ℝ)
(property : ∀ i, dist (vertices i) (center) = dist (vertices 0) (center) ∧ 
          ∀ i j, dist (vertices i) (vertices j) = side_length → i ≠ j)

-- Function to compute the sum of the distances from P to the vertices
def sum_of_distances_to_vertices (pentagon : RegularPentagon V) (P : V) : ℝ :=
  Finset.univ.sum (λ i, dist P (pentagon.vertices i))

-- Theorem statement
theorem minimum_sum_of_distances (pentagon : RegularPentagon V) (P : V) :
  sum_of_distances_to_vertices pentagon P ≥ sum_of_distances_to_vertices pentagon pentagon.center :=
sorry

end minimum_sum_of_distances_l510_510373


namespace cos_difference_inequality_l510_510708

theorem cos_difference_inequality (x y : ℝ) (hx : 0 ≤ x) (h'x : x ≤ π) (hy : 0 ≤ y) (h'y : y ≤ π) : 
  cos (x - y) ≥ cos x - cos y :=
sorry

end cos_difference_inequality_l510_510708


namespace parabola_equation_l510_510634

theorem parabola_equation (x y : ℝ)
    (focus : x = 1 ∧ y = -2)
    (directrix : 5 * x + 2 * y = 10) :
    4 * x^2 - 20 * x * y + 25 * y^2 + 158 * x + 156 * y + 16 = 0 := 
by
  -- use the given conditions and intermediate steps to derive the final equation
  sorry

end parabola_equation_l510_510634


namespace soccer_tournament_probability_l510_510891

-- Define basic assumptions and conditions
constant teams : Type
constant plays : teams → teams → Prop
constant wins : teams → teams → Prop
constant no_ties : ∀ (A B : teams), plays A B → wins A B ∨ wins B A
constant each_team_has_50_percent : ∀ (A B : teams), plays A B → 1/2
constant outcomes_independent : ∀ (A B : teams), independent (λ X Y : teams, plays X Y) 
constant points_awarded : ∀ (A B : teams), wins A B → 1

-- Define the specific setting of the problem
constant A B : teams
constant first_game_outcome : wins A B

-- Define the statement of the problem
theorem soccer_tournament_probability (m n : ℕ) (h_rel_prime : nat.coprime m n) :
  (m : ℝ) / (n : ℝ) = 319 / 512 → m + n = 831 :=
by sorry

end soccer_tournament_probability_l510_510891


namespace cos_positive_in_first_or_fourth_sector_properties_l510_510771

-- Define the conditions for quadrants and cosine values
def inFirstOrFourthQuadrant (θ : ℝ) : Prop :=
  (0 ≤ θ ∧ θ < π / 2) ∨ (3 * π / 2 < θ ∧ θ < 2 * π)

theorem cos_positive_in_first_or_fourth (θ : ℝ) (h : inFirstOrFourthQuadrant θ) : 0 < cos θ :=
by
  sorry
  
-- Define the conditions for the sector area and circumference  
def sector_area (r θ : ℝ) : ℝ := (1/2)*θ*r^2
def sector_circumference (r θ : ℝ) : ℝ := r*θ + 2*r

theorem sector_properties (r θ : ℝ) (area_eq_one : sector_area r θ = 1) (circ_eq_four : sector_circumference r θ = 4): θ = 2 :=
by
  sorry

end cos_positive_in_first_or_fourth_sector_properties_l510_510771


namespace Petya_strategy_l510_510049

-- Given definitions from the conditions
def chests := 1011
def coins := 2022

-- Vasya picks the first coin, which implies that the next turns alternate between Vasya and Petya
def first_turn_Vasya := true

theorem Petya_strategy : 
  ∃ strategy : (Nat → Nat) → Bool,
  (first_turn_Vasya →
  ∀ (n m : Nat) (A : (fin n → Nat)),
    (has_sum (λ i, A i) coins) ∧
    (∀ i, A i ≤ 2) →
    ∀ t : Nat, 
      t < coins →
      ∃ final_chest : Nat,
        (final_chest < chests) ∧
        (A final_chest = 2)) :=
sorry

end Petya_strategy_l510_510049


namespace hours_difference_is_approximately_l510_510866

theorem hours_difference_is_approximately : 
  ∀ (Kate Pat Mark Alex : ℝ), 
    Alex = (5/4) * Kate → 
    Pat + Kate + Mark + Alex = 212 → 
    Pat = 2 * Kate → 
    Pat = (1/3) * Mark → 
    (Mark - Kate) ≈ 103.4 :=
by 
  intros Kate Pat Mark Alex h1 h2 h3 h4;
  sorry

end hours_difference_is_approximately_l510_510866


namespace dana_total_earnings_l510_510315

theorem dana_total_earnings :
  let wage := 13
  let hours_friday := 9
  let hours_saturday := 10
  let hours_sunday := 3
  let earnings_friday := hours_friday * wage
  let earnings_saturday := hours_saturday * wage
  let earnings_sunday := hours_sunday * wage
  let total_earnings := earnings_friday + earnings_saturday + earnings_sunday
  in total_earnings = 286 := 
sorry

end dana_total_earnings_l510_510315


namespace real_solution_inequality_l510_510710

theorem real_solution_inequality (x : ℝ) :
  (1 / (x^2 + 4) < 2 / x + 27 / 10) ↔ (x ∈ set.union (set.union set.Ioo (-2 : ℝ) (-1)) (set.union set.Ioo (-1) 0) (set.Ioo 0 (-10 / 27))) :=
by
  sorry

end real_solution_inequality_l510_510710


namespace edge_length_of_cube_l510_510159

theorem edge_length_of_cube (SA : ℝ) (edge_length : ℝ) (h : SA = 6 * edge_length^2) (h_SA : SA = 150) : edge_length = 5 := 
by 
  rw [h_SA, h]
  have h1 : 6 * edge_length^2 = 150 := by assumption
  rw mul_comm at h1
  have h2 : edge_length^2 = 150 / 6 := by linarith
  rw [div_eq_mul_inv, mul_comm] at h2
  have h3 : edge_length^2 = 25 := by norm_num1 at h2 -- alternative way of simplification
  have h4 : edge_length = real.sqrt 25 := by exact (pow_eq_pow' zero_le_zero).elim_right (eq.symm h3)
  exact h4


end edge_length_of_cube_l510_510159


namespace countFibSequences_l510_510575

-- Define what it means for a sequence to be Fibonacci-type
def isFibType (a : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, a n = a (n - 1) + a (n - 2)

-- Define a Fibonacci-type sequence condition with given constraints
def fibSeqCondition (a : ℤ → ℤ) (N : ℤ) : Prop :=
  isFibType a ∧ ∃ n : ℤ, 0 < a n ∧ a n ≤ N ∧ 0 < a (n + 1) ∧ a (n + 1) ≤ N

-- Main theorem
theorem countFibSequences (N : ℤ) :
  ∃ count : ℤ,
    (N % 2 = 0 → count = (N / 2) * (N / 2 + 1)) ∧
    (N % 2 = 1 → count = ((N + 1) / 2) ^ 2) ∧
    (∀ a : ℤ → ℤ, fibSeqCondition a N → (∃ n : ℤ, a n = count)) :=
by
  sorry

end countFibSequences_l510_510575


namespace cylinder_volume_l510_510627

noncomputable def volume_of_cylinder (h : ℝ) (C : ℝ) : ℝ :=
  let r := C / (2 * real.pi) in
  real.pi * r^2 * h

theorem cylinder_volume : volume_of_cylinder 4 (10 * real.pi) = 100 * real.pi :=
by
  sorry

end cylinder_volume_l510_510627


namespace max_subsets_nonempty_intersection_l510_510375

theorem max_subsets_nonempty_intersection (n : ℕ) (S : finset (fin n)) :
  ∃ (k : ℕ), k = 2^(n-1) ∧
    ∀ (T : finset (finset (fin n))),
      T.card = k →
      (∀ (A B : finset (fin n)), A ∈ T → B ∈ T → A ≠ B → (A ∩ B).nonempty) →
      k ≤ 2^(n-1) :=
begin
  sorry
end

end max_subsets_nonempty_intersection_l510_510375


namespace irrational_pi_l510_510221

theorem irrational_pi :
  irrational real.pi := by
  sorry

end irrational_pi_l510_510221


namespace solution_set_of_inequality_l510_510700

theorem solution_set_of_inequality :
  ∀ x : ℝ, 3 * x^2 - 2 * x + 1 > 7 ↔ (x < -2/3 ∨ x > 3) :=
by
  sorry

end solution_set_of_inequality_l510_510700


namespace gcd_459_357_l510_510573

theorem gcd_459_357 :
  Nat.gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l510_510573


namespace grid_count_l510_510690

theorem grid_count :
  let vals := [0, 1, 2] in
  let valid_grids := {g : (ℕ × ℕ × ℕ) × (ℕ × ℕ × ℕ) |
                        g.1.1 ∈ vals ∧ g.1.2 ∈ vals ∧ g.1.3 ∈ vals ∧
                        g.2.1 ∈ vals ∧ g.2.2 ∈ vals ∧ g.2.3 ∈ vals ∧
                        (g.1.1 + g.1.2 + g.1.3) % 3 = 0 ∧
                        (g.2.1 + g.2.2 + g.2.3) % 3 = 0 ∧
                        (g.1.1 + g.2.1) % 3 = 0 ∧
                        (g.1.2 + g.2.2) % 3 = 0 ∧
                        (g.1.3 + g.2.3) % 3 = 0 } in
  finset.card valid_grids = 10 :=
sorry

end grid_count_l510_510690


namespace volume_of_pyramid_l510_510204

noncomputable def greatest_pyramid_volume (AB AC sin_α : ℝ) (max_angle : ℝ) : ℝ :=
  if AB = 3 ∧ AC = 5 ∧ sin_α = 4 / 5 ∧ max_angle ≤ 60 then
    5 * Real.sqrt 39 / 2
  else
    0

theorem volume_of_pyramid :
  greatest_pyramid_volume 3 5 (4 / 5) 60 = 5 * Real.sqrt 39 / 2 := by
  sorry -- Proof omitted as per instruction

end volume_of_pyramid_l510_510204


namespace number_of_correct_propositions_l510_510758

variables (m n : Line) (α β : Plane)

-- Definitions based on conditions
def prop1 (m_perp_α : m ⊥ α) (m_perp_β : m ⊥ β) : Prop := α ∥ β
def prop2 (m_subset_α : m ⊆ α) (n_subset_β : n ⊆ β) (m_parallel_n : m ∥ n) : Prop := α ∥ β
def prop3 (m_parallel_n : m ∥ n) (m_perp_α : m ⊥ α) : Prop := n ⊥ α
def prop4 (m_perp_α : m ⊥ α) (m_subset_β : m ⊆ β) : Prop := α ⊥ β

-- The theorem to be proved
theorem number_of_correct_propositions (h1 : prop1) (h2 : ¬prop2) (h3 : prop3) (h4 : prop4) :
  3 := sorry

end number_of_correct_propositions_l510_510758


namespace determine_y_minus_x_l510_510169

theorem determine_y_minus_x (x y : ℝ) (h1 : x + y = 360) (h2 : x / y = 3 / 5) : y - x = 90 := sorry

end determine_y_minus_x_l510_510169


namespace find_m_n_sum_l510_510479

theorem find_m_n_sum (a b c : ℝ) (hab : a ≠ 0) (hbc : b ≠ 0) (hca : c ≠ 0)
  (h1 : ab / (a + b) = 3) (h2 : bc / (b + c) = 4) (h3 : ca / (c + a) = 5) :
  let m := 120 in let n := 47 in m + n = 167 :=
by sorry

end find_m_n_sum_l510_510479


namespace find_general_term_find_sum_of_bn_l510_510765

section
variable (n : ℕ)

def S (n : ℕ) : ℤ := n^2 - 4*n - 5
def a : ℕ → ℤ
| 1 := -8
| n := if n ≥ 2 then 2*n - 5 else 0 -- default case for other n to define total function

def b (n : ℕ) : ℤ := Int.abs (a n)

-- T(n)
def T : ℕ → ℤ
| 1 := 8
| 2 := 9
| n := if n ≥ 3 then n^2 - 4*n + 13 else 0 -- use default case

theorem find_general_term :
  ∀ n ≥ 2, a n = S n - S (n - 1) := sorry

theorem find_sum_of_bn :
  ∀ n, T n = 
  match n with
  | 1 := 8
  | 2 := 9
  | _ := if n ≥ 3 then n^2 - 4*n + 13 else 0 := sorry

end

end find_general_term_find_sum_of_bn_l510_510765


namespace angle_measure_l510_510957

variable (x : ℝ)

def complement (x : ℝ) : ℝ := 90 - x

def supplement (x : ℝ) : ℝ := 180 - x

theorem angle_measure (h : supplement x = 8 * complement x) : x = 540 / 7 := by
  sorry

end angle_measure_l510_510957


namespace problem_1_solution_set_problem_2_minimum_value_a_l510_510405

-- Define the function f with given a value
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Problem 1: Prove the solution set for f(x) > 5 when a = -2 is {x | x < -4/3 ∨ x > 2}
theorem problem_1_solution_set (x : ℝ) : f x (-2) > 5 ↔ x < -4 / 3 ∨ x > 2 :=
by
  sorry

-- Problem 2: Prove the minimum value of a ensures f(x) ≤ a * |x + 3| is 1/2
theorem problem_2_minimum_value_a : (∀ x : ℝ, f x a ≤ a * |x + 3| ∨ a ≥ 1/2) :=
by
  sorry

end problem_1_solution_set_problem_2_minimum_value_a_l510_510405


namespace annie_crayons_l510_510298

def initial_crayons : ℕ := 4
def additional_crayons : ℕ := 36
def total_crayons : ℕ := initial_crayons + additional_crayons

theorem annie_crayons : total_crayons = 40 :=
by
  sorry

end annie_crayons_l510_510298


namespace problem1_problem2_problem3_l510_510764

variable {a b : ℕ → ℝ}
variable (λ : ℝ)

-- 1. Prove general formula for {a_n} and the sum of the first n terms
theorem problem1 
  (h_arith : ∀ n, a (n + 1) - a n = 2)
  (h_a3 : a 3 = 5)
  (h_geo : a 2 ^ 2 = a 1 * a 5)
  : ∃ a, ∀ n, a n = 2 * n - 1 ∧ (S n = n ^ 2) := sorry

-- 2. Prove the sum of the first n terms of {b_n}
theorem problem2 
  (h_bn : ∀ n, b n = 1 / (a n * a (n + 1)))
  : ∀ n, T n = n / (2 * n + 1) := sorry

-- 3. Prove the range of λ
theorem problem3 
  (h_Tn : ∀ n, T n = n / (2 * n + 1))
  : ∀ n, (λ T n < n + 8 * (-1) ^ n) → λ < -21 := sorry

end problem1_problem2_problem3_l510_510764


namespace statement_A_l510_510294

variable {A B : Type}

-- Define what it means for a mapping from A to B.
def mapping (f : A → B) := True

-- Statement A: There exists a function where an element in B may have more than one pre-image.
theorem statement_A (f : A → B) : ∃ b ∈ set.range f, ∃ a1 a2, a1 ≠ a2 ∧ f a1 = b ∧ f a2 = b :=
begin
  sorry
end

end statement_A_l510_510294


namespace minimum_value_of_function_l510_510365

theorem minimum_value_of_function (x : ℝ) (h : x > -1) : 
  2x + 1 / (x + 1) ≥ 2 * real.sqrt 2 - 2 :=
sorry

end minimum_value_of_function_l510_510365


namespace sum_is_10_l510_510226

theorem sum_is_10 (a b c : ℕ) (h1 : a * b * c = 24) (h2 : a + b + c < 25) (h3 : (a + b + c) % 2 = 0) (h4 : 2 ≤ a ∧ 2 ≤ b ∧ 2 ≤ c ∧ a + b + c < 25) :
  (a + b + c = 10) ∨ (a + b + c = 12) ∨ sorry :=
by
  cases h4 with a_pos b_pos c_pos
  sorry

end sum_is_10_l510_510226


namespace nonneg_diff_between_roots_l510_510585

theorem nonneg_diff_between_roots : 
  ∀ x : ℝ, 
  (x^2 + 42 * x + 384 = 0) → 
  (∃ r1 r2 : ℝ, x = r1 ∨ x = r2) → 
  (abs (r1 - r2) = 8) := 
sorry

end nonneg_diff_between_roots_l510_510585


namespace frank_change_l510_510356

theorem frank_change (n_c n_b money_given c_c c_b : ℕ) 
  (h1 : n_c = 5) 
  (h2 : n_b = 2) 
  (h3 : money_given = 20) 
  (h4 : c_c = 2) 
  (h5 : c_b = 3) : 
  money_given - (n_c * c_c + n_b * c_b) = 4 := 
by
  sorry

end frank_change_l510_510356


namespace common_ratio_values_l510_510831

open Real

noncomputable theory

-- Define the geometric sequence properties
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a 1 * q^n

-- Define the integral calculation
def S_3_integral_value : ℝ := 3 * ∫ x in 0..3, x^3

-- The proof problem in Lean 4 statement
theorem common_ratio_values 
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 3 = 9)
  (h2 : ∑ i in finset.range 3, a (i + 1) = S_3_integral_value)
  (h_geom : geometric_sequence a q) :
  q = 1 ∨ q = -1/2 :=
sorry

end common_ratio_values_l510_510831


namespace tangent_line_at_one_l510_510340

noncomputable def equation_of_tangent_line (x : ℝ) : ℝ :=
  3 * (x^(1/3) - 2 * x^(1/2))

theorem tangent_line_at_one :
  let x := 1 in
  let y0 := equation_of_tangent_line x in
  let y' := deriv equation_of_tangent_line x in
  y' = -2 ∧ y0 = -3 ∧ (∀ x, equation_of_tangent_line x - y0 = y' * (x - 1) → equation_of_tangent_line x - y0 = -2 * (x - 1)) :=
by
  sorry

end tangent_line_at_one_l510_510340


namespace rectangle_XYZW_l510_510371

open Point Set

noncomputable def parallelogram : Type := sorry

variables
  (A B C D X Y Z T : Point)
  (ABCD : parallelogram)
  (h₁ : (A, B, C, D) ∈ ABCD)
  (h₂ : inscribed_circle ABC touches AC at X)
  (h₃ : inscribed_circle ADC touches AC at Y)
  (h₄ : inscribed_circle BCD touches BD at Z)
  (h₅ : inscribed_circle BAD touches BD at T)
  (h₆ : X ≠ Y ∧ Y ≠ Z ∧ Z ≠ T ∧ T ≠ X)

theorem rectangle_XYZW : is_rectangle {X, Y, Z, T} :=
sorry

end rectangle_XYZW_l510_510371


namespace product_of_means_eq_pm20_l510_510924

theorem product_of_means_eq_pm20 :
  let a := (2 + 8) / 2
  let b := Real.sqrt (2 * 8)
  a * b = 20 ∨ a * b = -20 :=
by
  -- Placeholders for the actual proof
  let a := (2 + 8) / 2
  let b := Real.sqrt (2 * 8)
  sorry

end product_of_means_eq_pm20_l510_510924


namespace numDistPrimePairs_l510_510060

-- Define the function to count the number of pairs of distinct primes summing to n
def distinctPrimeSumWays (n : ℕ) : ℕ :=
  let isPrime (p : ℕ) : Bool := p > 1 ∧ (∀ d, d > 1 ∧ d < p → p % d ≠ 0)
  let validPairs := { (p1, p2) | isPrime p1 ∧ isPrime p2 ∧ p1 ≠ p2 ∧ p1 + p2 = n }
  validPairs.card

-- Proposition: The number of pairs of distinct primes summing to 9998 is 2
theorem numDistPrimePairs : distinctPrimeSumWays 9998 = 2 := by
  sorry

end numDistPrimePairs_l510_510060


namespace exists_constant_sum_and_value_l510_510151

theorem exists_constant_sum_and_value :
  ∃ N : ℕ, ∃ S : ℕ, (∀ n ≥ N, ∑ k in finset.range(2022).map (finset.Ico n (n+1)), a (n+k)) = S ∧ S = 4043^2 :=
begin
  -- Assume sequence a is defined and satisfies given conditions
  assume (a : ℕ → ℕ),
  assume h1: ∀ i ≥ 2022, ∃ x : ℕ, x = a i ∧ x + (∑ k in range (i-2021, i), a k) = (some z : ℕ , z * z),
  assume h2: ∃ G : set ℕ, infinite G ∧ ∀ n ∈ G, a n = 4 * 2022 - 3,
  sorry,
end

end exists_constant_sum_and_value_l510_510151


namespace evaluate_sum_T_l510_510044

variable {t : ℕ} (ht : t = 50)

noncomputable def x (i : ℕ) : ℝ :=
  (i : ℝ) / 101

noncomputable def T : ℝ :=
  ∑ i in Finset.range 102, (x i) ^ 3 / (3 * (x t) ^ 2 - 3 * (x t) + 1)

theorem evaluate_sum_T : T ht = 51 := 
  sorry

end evaluate_sum_T_l510_510044


namespace chessboard_coloring_no_rectangle_same_color_l510_510975
-- Import the entirety of Mathlib for necessary library support

-- Define the statements for the problem
axiom chessboard_coloring_exists_rectangle_same_color :
  ∀ (color : Fin 4 → Fin 7 → Bool), ∃ (r1 r2 : Fin 4) (c1 c2 : Fin 7), r1 ≠ r2 ∧ c1 ≠ c2 ∧
    color r1 c1 = color r1 c2 ∧
    color r1 c1 = color r2 c1 ∧
    color r1 c1 = color r2 c2

noncomputable def example_coloring_4x6 : Fin 4 → Fin 6 → Bool :=
  λ r c, match c with
    | 0 => r < 2
    | 1 => r % 2 == 0
    | 2 => r = 0 ∨ r = 1 ∨ r = 3
    | 3 => r = 0 ∨ r = 2 ∨ r = 3
    | 4 => r = 1 ∨ r = 2 ∨ r = 3
    | 5 => r > 1
    | _ => false

theorem chessboard_coloring_no_rectangle_same_color :
  ∀ (color : Fin 4 → Fin 6 → Bool), ¬∃ (r1 r2 : Fin 4) (c1 c2 : Fin 6), r1 ≠ r2 ∧ c1 ≠ c2 ∧
    color r1 c1 = color r1 c2 ∧
    color r1 c1 = color r2 c1 ∧
    color r1 c1 = color r2 c2 :=
by
  intro color
  exists example_coloring_4x6
  sorry

end chessboard_coloring_no_rectangle_same_color_l510_510975


namespace find_HK_correct_l510_510508

noncomputable def triangle_lengths (a b : ℝ) : ℝ :=
  real.sqrt (a ^ 2 + b ^ 2)

def find_HK : ℝ :=
  let AC := 3
  let BC := 2
  let AB := triangle_lengths AC BC
  let CH := (AC * BC) / AB
  let DF := AB
  let CK := DF / 2
  (CH + CK) 

theorem find_HK_correct :
  find_HK = 25 / (2 * real.sqrt 13) :=
  sorry

end find_HK_correct_l510_510508


namespace parallel_line_plane_l510_510436

-- Definitions
variable {Point : Type} [Field Point]
structure Line (P : Type) [Field P] :=
(loc : P → Prop)
(para : P → P → Prop)

structure Plane (P : Type) [Field P] :=
(loc : P → Prop)

variables {P : Type} [Field P] (a b : Line P) (π : Plane P)

-- Conditions (Hypothesis)
axiom parallel_lines : a.para = b.para

-- The Problem Statement
theorem parallel_line_plane (h : ∀ π, (b.loc : P → Prop) π → (a.para P π)) : false :=
sorry

end parallel_line_plane_l510_510436


namespace magnitude_difference_l510_510030

variables {ℝ : Type*} [normed_field ℝ] [normed_space ℝ ℝ]

-- Define the planar vectors
variables (a b : ℝ)

-- Given conditions
axiom condition1 : ∥a + b∥ = 1
axiom condition2 : ∥a - b∥ = √2
axiom condition3 : real_inner (a + b) (a - b) = π / 4

-- The theorem to prove
theorem magnitude_difference : ∥a - 5 * b∥ = √5 := 
by {
  sorry,
}

end magnitude_difference_l510_510030


namespace meals_without_restrictions_l510_510122

theorem meals_without_restrictions (total_clients vegan kosher gluten_free halal dairy_free nut_free vegan_kosher vegan_gluten_free kosher_gluten_free halal_dairy_free gluten_free_nut_free vegan_halal_gluten_free kosher_dairy_free_nut_free : ℕ) 
  (h_tc : total_clients = 80)
  (h_vegan : vegan = 15)
  (h_kosher : kosher = 18)
  (h_gluten_free : gluten_free = 12)
  (h_halal : halal = 10)
  (h_dairy_free : dairy_free = 8)
  (h_nut_free : nut_free = 4)
  (h_vegan_kosher : vegan_kosher = 5)
  (h_vegan_gluten_free : vegan_gluten_free = 6)
  (h_kosher_gluten_free : kosher_gluten_free = 3)
  (h_halal_dairy_free : halal_dairy_free = 4)
  (h_gluten_free_nut_free : gluten_free_nut_free = 2)
  (h_vegan_halal_gluten_free : vegan_halal_gluten_free = 2)
  (h_kosher_dairy_free_nut_free : kosher_dairy_free_nut_free = 1) : 
  (total_clients - (vegan + kosher + gluten_free + halal + dairy_free + nut_free 
  - vegan_kosher - vegan_gluten_free - kosher_gluten_free - halal_dairy_free - gluten_free_nut_free 
  + vegan_halal_gluten_free + kosher_dairy_free_nut_free) = 30) :=
by {
  -- solution steps here
  sorry
}

end meals_without_restrictions_l510_510122


namespace greatest_prime_factor_of_k_l510_510599

-- Definition of k as the sum of all even multiples of 25 between 295 and 615
def k : Nat :=
  (List.range (600 - 300) / 50 + 1).map (λ i => 300 + 50 * i) |>.sum

-- Prove that the greatest prime factor of k is 53
theorem greatest_prime_factor_of_k : ∀ (k = 2650), 53 ∈ primeFactors k ∧ ∀ p ∈ primeFactors k, p ≤ 53 := by
  sorry

end greatest_prime_factor_of_k_l510_510599


namespace an_eq_n_l510_510761

theorem an_eq_n (a : ℕ → ℝ) (h_positive : ∀ n : ℕ, a n > 0)
  (h_eq : ∀ n : ℕ, (∑ j in Finset.range (n+1), (a j)^3) = (∑ j in Finset.range (n+1), a j)^2) :
  ∀ n : ℕ, a n = n := 
by
  sorry

end an_eq_n_l510_510761


namespace min_coach_handshakes_l510_510055

theorem min_coach_handshakes (n k1 k2 : ℕ) 
  (h1 : k1 + k2 = n) 
  (total_handshakes : ℕ := ∑ i in finset.range(n), i) 
  (h2 :  total_handshakes + k1 + k2 = 456) : 
  ∃ (m : ℕ), m = 1 := 
sorry

end min_coach_handshakes_l510_510055


namespace exists_b_c_l510_510596

theorem exists_b_c (a : ℕ) (h1 : a > 17) (h2 : ∃ m : ℕ, 3 * a - 2 = m^2) : 
  ∃ (b c : ℕ), b ≠ c ∧ is_square (a + b) ∧ is_square (a + c) ∧ is_square (b + c) ∧ is_square (a + b + c) :=
sorry

end exists_b_c_l510_510596


namespace socks_pair_count_l510_510824

theorem socks_pair_count :
  let white := 5 in let brown := 5 in let blue := 3 in let red := 2 in
  (white * (white - 1) / 2) + (brown * (brown - 1) / 2) + (blue * (blue - 1) / 2) + (red * (red - 1) / 2) = 24 :=
by
  let white := 5
  let brown := 5
  let blue := 3
  let red := 2 
  calc
    (white * (white - 1) / 2) + (brown * (brown - 1) / 2) + (blue * (blue - 1) / 2) + (red * (red - 1) / 2) 
      = (5 * 4 / 2) + (5 * 4 / 2) + (3 * 2 / 2) + (2 * 1 / 2) : by
        simp [white, brown, blue, red]
      ... = 10 + 10 + 3 + 1 : by
        norm_num
      ... = 24 : by
        norm_num

end socks_pair_count_l510_510824


namespace angle_MHB_30_l510_510447

-- Define the problem's constants and conditions
variables (A B C H M : Type)
variables [Angle A] [Angle B] [Angle C] [Triangle ABC]
variables [right_angle : ∠ A = 90] [angle_B : ∠ B = 60] [angle_C : ∠ C = 30] 
variables [altitude_AH : Altitude(∥AH)] [bisector_BM : Bisector(∠ ABC, ∥BM)]

-- Define the theorem statement to prove
theorem angle_MHB_30 (A B C H M : Angle) (ABC : Triangle ABC) 
(altitude_AH : Altitude A H) (bisector_BM : Bisector B M) 
(right_angle : ∠ A = 90) (angle_B : ∠ B = 60) (angle_C : ∠ C = 30) 
: (∠ M H B = 30) :=
by {
  sorry
}

end angle_MHB_30_l510_510447


namespace num_outliers_correct_l510_510694

noncomputable section

def data_set : List ℝ := [10, 24, 35, 35, 35, 42, 42, 45, 58, 62]
def Q1 : ℝ := 35
def Q3 : ℝ := 45
def Q2 : ℝ := 38.5

def IQR : ℝ := Q3 - Q1
def lower_threshold : ℝ := Q1 - 1.5 * IQR
def upper_threshold : ℝ := Q3 + 1.5 * IQR
def mean_data_set : ℝ := (10 + 24 + 35 + 35 + 35 + 42 + 42 + 45 + 58 + 62) / 10 -- Mean calculation
def stddev_data_set : ℝ := Real.sqrt ((List.sum (data_set.map (λ x, (x - mean_data_set)^2))) / 10) -- Standard deviation calculation

def is_outlier (x : ℝ) : Prop :=
  x < lower_threshold ∨ x > upper_threshold ∨ x < mean_data_set - 2 * stddev_data_set ∨ x > mean_data_set + 2 * stddev_data_set

def num_outliers : ℕ :=
  List.length (List.filter is_outlier data_set)

theorem num_outliers_correct : num_outliers = 1 := by
  sorry

end num_outliers_correct_l510_510694


namespace exchange_rate_decrease_l510_510667

theorem exchange_rate_decrease
  (x y z : ℝ)
  (hx : 0 < |x| ∧ |x| < 1)
  (hy : 0 < |y| ∧ |y| < 1)
  (hz : 0 < |z| ∧ |z| < 1)
  (h_eq : (1 + x) * (1 + y) * (1 + z) = (1 - x) * (1 - y) * (1 - z)) :
  (1 - x^2) * (1 - y^2) * (1 - z^2) < 1 :=
by
  sorry

end exchange_rate_decrease_l510_510667


namespace total_packs_l510_510874

theorem total_packs (cards_bought : ℕ) (cards_per_pack : ℕ) (num_people : ℕ)
  (h1 : cards_bought = 540) (h2 : cards_per_pack = 20) (h3 : num_people = 4) :
  (cards_bought / cards_per_pack) * num_people = 108 :=
by
  sorry

end total_packs_l510_510874


namespace math_equivalent_proof_problem_l510_510370

-- Definitions for the problem conditions

-- Given points
def A := (-4 : ℝ, 0 : ℝ)
def B := (-1 : ℝ, 0 : ℝ)

-- Distance formula
def dist (P Q : ℝ × ℝ) := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^(1/2)

-- Condition on distances
def cond (M : ℝ × ℝ) := dist M A = 2 * dist M B

-- Object to prove
def trajectory (M : ℝ × ℝ) := M.1^2 + M.2^2 = 4

-- Perimeter of triangle
def perimeter (M : ℝ × ℝ) := dist M A + dist M B + dist A B

-- Range of perimeter
def range_perimeter := {P : ℝ | 6 < P ∧ P < 12}

-- Definition for areas ratio range
def ratio_range := {r : ℝ | 1/3 < r ∧ r < 3}

-- Lean 4 statement for the proof problem
theorem math_equivalent_proof_problem (M M' : ℝ × ℝ) :
  cond M →
  cond M' → 
  trajectory M ∧
  (perimeter M ∈ range_perimeter) ∧
  ((dist M B ≠ 0) ∧ (dist M' B ≠ 0) →
    dist M.2 0 ∧ dist M'.2 0 →
    abs (M.2 / M'.2) ∈ ratio_range) :=
by
  sorry

end math_equivalent_proof_problem_l510_510370


namespace prove_a_eq_b_l510_510840

theorem prove_a_eq_b (a b : ℝ) (h : 1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b)) : a = b :=
sorry

end prove_a_eq_b_l510_510840


namespace number_of_adult_tickets_l510_510279

-- Definitions from conditions
variables (A S : Nat)

-- Hypotheses representing the conditions
def condition1 : Prop := S = 2 * A
def condition2 : Prop := A + S = 366

-- The theorem we want to prove
theorem number_of_adult_tickets (h1 : condition1) (h2 : condition2) : A = 122 := by
  sorry

end number_of_adult_tickets_l510_510279


namespace angle_measure_proof_l510_510955

noncomputable def angle_measure (x : ℝ) : Prop :=
  let supplement := 180 - x
  let complement := 90 - x
  supplement = 8 * complement

theorem angle_measure_proof : ∃ x : ℝ, angle_measure x ∧ x = 540 / 7 :=
by
  have angle_eq : ∀ x, angle_measure x ↔ (180 - x = 8 * (90 - x)) := by
    intro x
    dsimp [angle_measure]
    rfl
  use 540 / 7
  rw angle_eq
  split
  · dsimp
    linarith
  · rfl

end angle_measure_proof_l510_510955


namespace scout_hours_worked_l510_510513

variable (h : ℕ) -- number of hours worked on Saturday
variable (base_pay : ℕ) -- base pay per hour
variable (tip_per_customer : ℕ) -- tip per customer
variable (saturday_customers : ℕ) -- customers served on Saturday
variable (sunday_hours : ℕ) -- hours worked on Sunday
variable (sunday_customers : ℕ) -- customers served on Sunday
variable (total_earnings : ℕ) -- total earnings over the weekend

theorem scout_hours_worked {h : ℕ} (base_pay : ℕ) (tip_per_customer : ℕ) (saturday_customers : ℕ) (sunday_hours : ℕ) (sunday_customers : ℕ) (total_earnings : ℕ) :
  base_pay = 10 → 
  tip_per_customer = 5 → 
  saturday_customers = 5 → 
  sunday_hours = 5 → 
  sunday_customers = 8 → 
  total_earnings = 155 → 
  10 * h + 5 * 5 + 10 * 5 + 5 * 8 = 155 → 
  h = 4 :=
by
  intros
  sorry

end scout_hours_worked_l510_510513


namespace sampled_items_l510_510258

theorem sampled_items (total_products : ℕ) (sampled_products : ℕ) (workshop_products : ℕ) (total_products_eq : total_products = 2048) (sampled_products_eq : sampled_products = 128) (workshop_products_eq : workshop_products = 256) :
  ∃ x : ℕ, x = 16 ∧ (x * workshop_products = sampled_products * total_products) :=
by
  use 16
  split
  exact rfl
  rw [total_products_eq, sampled_products_eq, workshop_products_eq]
  norm_num
  sorry

end sampled_items_l510_510258


namespace domain_of_f_l510_510070

noncomputable def f (x : ℝ) : ℝ := (sqrt (x + 2)) + (1 / (x - 1))

theorem domain_of_f :
  ∀ x : ℝ, (x ≥ -2 ∧ x ≠ 1) ↔ ∃ y : ℝ, y = f x :=
begin
  sorry
end

end domain_of_f_l510_510070


namespace find_m_l510_510608

theorem find_m (m : ℝ) : (∀ x : ℝ, x^2 - m*x - 15 = (x + 3) * (some_term x)) → m = 2 := by
  assume h
  -- the actual proof is omitted, represented by sorry
  sorry

end find_m_l510_510608


namespace interesting_moments_from_0001_to_1201_l510_510985

noncomputable def interesting_moments : ℕ := 143

theorem interesting_moments_from_0001_to_1201 :
  ∃ (count : ℕ), count = 143 ∧ 
    (∀ (X Y : ℝ), 
      (0 < X ∧ X < 12 + 1/60) ∧ 
      (∃ (Y : ℝ), (30 * Y = 360 * (X - ⌊X⌋) ∧ 360 * (Y - ⌊Y⌋) = 30 * X)) → count = 143) :=
begin
  use 143,
  split,
  { refl }, -- This confirms the count is 143
  { intros X Y hX hY,
    sorry -- Proof part here to show the calculations add up to 143
  }

end interesting_moments_from_0001_to_1201_l510_510985


namespace parametric_graph_right_half_circle_l510_510915

theorem parametric_graph_right_half_circle (θ : ℝ) (x y : ℝ) (hx : x = 3 * Real.cos θ) (hy : y = 3 * Real.sin θ) (hθ : -Real.pi / 2 ≤ θ ∧ θ ≤ Real.pi / 2) :
  x^2 + y^2 = 9 ∧ x ≥ 0 :=
by
  sorry

end parametric_graph_right_half_circle_l510_510915


namespace tickets_to_be_sold_l510_510940

theorem tickets_to_be_sold (tickets_total : ℕ) (tickets_jude : ℕ) (tickets_andrea : ℕ) (tickets_sandra : ℕ) :
  tickets_total = 100 →
  tickets_jude = 16 →
  tickets_andrea = 2 * tickets_jude →
  tickets_sandra = (tickets_jude / 2) + 4 →
  tickets_total - (tickets_jude + tickets_andrea + tickets_sandra) = 40 :=
by {
  intros h_total h_jude h_andrea h_sandra,
  simp [h_total, h_jude, h_andrea, h_sandra],
  sorry
}

end tickets_to_be_sold_l510_510940


namespace distance_between_planes_l510_510339

theorem distance_between_planes :
  let plane1 (x y z : ℝ) := 3 * x - y + 2 * z - 3 = 0
  let plane2 (x y z : ℝ) := 6 * x - 2 * y + 4 * z + 7 = 0
  ∃ d : ℝ, d = 13 * real.sqrt 14 / 28 := by
{
  sorry
}

end distance_between_planes_l510_510339


namespace isosceles_triangle_angles_l510_510568

theorem isosceles_triangle_angles (y : ℝ) (h : y > 0) :
  let P := y
  let R := 5 * y
  let Q := R
  P + Q + R = 180 → Q = 81.82 :=
by
  sorry

end isosceles_triangle_angles_l510_510568


namespace commute_time_absolute_difference_l510_510637

theorem commute_time_absolute_difference 
  (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) :
  |x - y| = 4 :=
by sorry

end commute_time_absolute_difference_l510_510637


namespace zero_in_interval_l510_510173

noncomputable def f (x : ℝ) := log x / log 3 - 8 + 2 * x

theorem zero_in_interval : ∃ c : ℝ, 3 < c ∧ c < 4 ∧ f c = 0 :=
by
  sorry

end zero_in_interval_l510_510173


namespace _l510_510545

-- Definition of the problem conditions and the theorem statement
structure RightTriangle (A B C M : Type) :=
(orthocenter : ∀ a b c : A, ∠B = 90⁰)
(median_AD : divides AC into two equal segments at D)
(angle_bisector_CE : bisects ∠C)
(intersect_at_M : ∃ M, AD ∩ CE = M)
(length_CM : d M C = 8)
(length_ME : d M E = 5)

noncomputable def area_of_triangle_ABC (A B C M : Type)
[RightTriangle A B C M] :
area A B C = 1352 / 15 := sorry

end _l510_510545


namespace Peter_finishes_all_tasks_at_5_30_PM_l510_510135

-- Definitions representing the initial conditions
def start_time : ℕ := 9 * 60 -- 9:00 AM in minutes
def third_task_completion_time : ℕ := 11 * 60 + 30 -- 11:30 AM in minutes
def task_durations : List ℕ :=
  [30, 30, 60, 120, 240] -- Durations of the 5 tasks in minutes
  
-- Statement for the proof problem
theorem Peter_finishes_all_tasks_at_5_30_PM :
  let total_duration := task_durations.sum 
  let finish_time := start_time + total_duration
  finish_time = 17 * 60 + 30 := -- 5:30 PM in minutes
  sorry

end Peter_finishes_all_tasks_at_5_30_PM_l510_510135


namespace choose_logarithmic_function_l510_510992

-- Define the conditions
variables {x y : ℝ}
def initial_rapid_growth : Prop := ∃ a, ∀ b > a, y > x
def growth_slows_down_over_time : Prop := ∀ x, ∃ t > x, (differential t x < differential x x)
def increasing_function : Prop := ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

-- The main theorem to be proved
theorem choose_logarithmic_function (h1 : initial_rapid_growth) 
                                    (h2 : growth_slows_down_over_time) 
                                    (h3 : increasing_function) : 
                                    function_model y = log x :=
begin
  sorry
end

end choose_logarithmic_function_l510_510992


namespace sum_coordinates_32_l510_510176

def point := (ℝ × ℝ)

noncomputable def satisfies_conditions (p : point) : Prop :=
  (|p.snd - 10| = 4) ∧ (Real.sqrt ((p.fst - 3)^2 + (p.snd - 10)^2) = 15)

noncomputable def four_points : List point :=
  [(3 + Real.sqrt 209, 14), (3 - Real.sqrt 209, 14), (3 + Real.sqrt 209, 6), (3 - Real.sqrt 209, 6)]

noncomputable def sum_of_coordinates (points : List point) : ℝ :=
  List.sum (points.map (λ p => p.fst + p.snd))

theorem sum_coordinates_32 : sum_of_coordinates four_points = 32 :=
  by
    have satisfies_conditions_all : ∀ p ∈ four_points, satisfies_conditions p := sorry
    show sum_of_coordinates four_points = 32 from sorry

end sum_coordinates_32_l510_510176


namespace find_fx_values_l510_510096

noncomputable def S : Set ℝ := { x | x ≠ 0 }

variable (f : S → S)
variable h : ∀ x y : S, x.val + y.val ≠ 0 → 
  f x + f y = f (⟨(x.val * y.val) / f (subtype.mk (x.val + y.val) sorry) + 1 / (x.val + y.val), sorry⟩)

theorem find_fx_values :
  let n : ℕ := 1 
  let s : ℝ := 1 / 9
  n * s = 1 / 9 :=
by sorry

end find_fx_values_l510_510096


namespace arithmetic_expression_evaluation_l510_510197

theorem arithmetic_expression_evaluation :
  12 / 4 - 3 - 6 + 3 * 5 = 9 :=
by
  sorry

end arithmetic_expression_evaluation_l510_510197


namespace math_problem_proof_l510_510887

noncomputable def proof_problem : Prop :=
  ∀ (x y : ℝ), 
    (x + y) / 2 = 20 ∧ (xy)^(1/2) = √100 → x^2 + y^2 = 1400

theorem math_problem_proof : proof_problem :=
begin
  intros x y h,
  cases h with h1 h2,
  sorry
end

end math_problem_proof_l510_510887


namespace simplify_expression_l510_510519

theorem simplify_expression :
  (real.sqrt 768 / real.sqrt 192 - real.sqrt 98 / real.sqrt 49) = (2 - real.sqrt 2) :=
sorry

end simplify_expression_l510_510519


namespace greatest_volume_of_pyramid_l510_510205

noncomputable def max_pyramid_volume (AB AC : ℝ) (sin_BAC : ℝ) (angle_limit : ℝ) : ℝ :=
  if AB = 3 ∧ AC = 5 ∧ sin_BAC = 4/5 ∧ angle_limit = π / 3 then 5 * Real.sqrt 39 / 2 else 0

theorem greatest_volume_of_pyramid :
  let AB := 3
  let AC := 5
  let sin_BAC := 4/5
  let angle_limit := π / 3
  max_pyramid_volume AB AC sin_BAC angle_limit = 5 * Real.sqrt 39 / 2 := by 
  sorry

end greatest_volume_of_pyramid_l510_510205


namespace vector_dot_product_and_magnitude_difference_l510_510013

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2)
variables (θ : ℝ) (hθ : θ = Real.pi / 3)

theorem vector_dot_product_and_magnitude_difference :
  (a • b = 1) ∧ (‖a - b‖ = Real.sqrt 3) :=
by
  sorry

end vector_dot_product_and_magnitude_difference_l510_510013


namespace min_value_x2_y2_l510_510803

theorem min_value_x2_y2 (x y : ℝ) (h : x + y = 2) : ∃ m, m = x^2 + y^2 ∧ (∀ (x y : ℝ), x + y = 2 → x^2 + y^2 ≥ m) ∧ m = 2 := 
sorry

end min_value_x2_y2_l510_510803


namespace total_number_of_balls_in_pyramid_display_l510_510058

theorem total_number_of_balls_in_pyramid_display (a d l : ℕ) (n : ℕ) (Sn : ℕ)
  (h1 : a = 35)
  (h2 : d = -4)
  (h3 : l = 1)
  (h4 : l = a + (n - 1) * d)
  (h5 : Sn = n * (a + l) / 2) 
  : Sn = 180 :=
sorry

end total_number_of_balls_in_pyramid_display_l510_510058


namespace find_n_l510_510361

theorem find_n (n : ℕ) (h : (nat.choose (n + 1) 7) - (nat.choose n 7) = nat.choose n 8) : n = 14 :=
sorry

end find_n_l510_510361


namespace ellipse_equation_proof_slope_AB_proof_max_area_AOB_proof_l510_510062

-- Defining the basic constants and conditions
variables (a b : ℝ) (P : ℝ × ℝ)
def ellipse_equation (x y : ℝ) := (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1
def eccentricity := a > b > 0 ∧ √(a ^ 2 - b ^ 2) / a = √2 / 2
def point_on_ellipse := P = (2, 1) ∧ ellipse_equation P.1 P.2

-- Main math proof problems
theorem ellipse_equation_proof (a b : ℝ) (P : ℝ × ℝ)
  (h1 : eccentricity a b)
  (h2 : point_on_ellipse a b P) : 
  (a = √6) ∧ (b =√3) ∧ ((∀ (x y : ℝ), ellipse_equation a b x y ↔ (x^2 / 6) + (y^2 / 3) = 1)) :=
sorry

theorem slope_AB_proof (a b : ℝ) (P : ℝ × ℝ)
  (h1 : eccentricity a b)
  (h2 : point_on_ellipse a b P)
  (A B : ℝ × ℝ) (hA_B_ellipse : ellipse_equation a b A.1 A.2 ∧ ellipse_equation a b B.1 B.2) 
  (M_on_OP : (A.1 + B.1) / 2 = (P.1 / 2) ∧ (A.2 + B.2) / 2 = (P.2 / 2)) :
  (A ≠ B ∧ ((A.2 - B.2) / (A.1 - B.1) = -1)) :=
sorry

theorem max_area_AOB_proof (a b : ℝ) (P : ℝ × ℝ)
  (h1 : eccentricity a b)
  (h2 : point_on_ellipse a b P)
  (A B : ℝ × ℝ) (hA_B_ellipse : ellipse_equation a b A.1 A.2 ∧ ellipse_equation a b B.1 B.2)
  (M_on_OP : (A.1 + B.1) / 2 = (P.1 / 2) ∧ (A.2 + B.2) / 2 = (P.2 / 2)) :
  (A ≠ B ∧ (∀ (t : ℝ), -3 < t ∧ t < 3 → (1 / 2) * (sqrt 2 / 3 * sqrt (t^2 * (9 - t^2))) ≤ 3 * sqrt 2 / 2)) :=
sorry

end ellipse_equation_proof_slope_AB_proof_max_area_AOB_proof_l510_510062


namespace max_red_beads_l510_510275

-- Define the structure of the beads string
structure Beads (n : Nat) :=
  (total_beads : Nat)
  (green_condition : ∀ (i : Nat), i < total_beads - 5 → ∃ (j : Nat), j ∈ (i..(i+6)) ∧ isGreen j)
  (blue_condition : ∀ (i : Nat), i < total_beads - 10 → ∃ (j : Nat), j ∈ (i..(i+11)) ∧ isBlue j)
  (count_beads : Nat)

-- Define the necessary conditions
def beads_string : Beads 150 :=
  { total_beads := 150,
    green_condition := sorry,
    blue_condition := sorry,
    count_beads := sorry }

-- Define the maximum number of red beads proof
theorem max_red_beads : ∃ red_beads : Nat, red_beads = 112 :=
  sorry

end max_red_beads_l510_510275


namespace symmetric_point_coords_symmetric_line_eq_l510_510410

-- Definitions and Problem Specifications
def line (a b c : ℝ) := ∀ x y : ℝ, a * x + b * y + c = 0

-- Given conditions for the first part
def l1 := line 2 (-3) 1
def A : ℝ × ℝ := (-1, -2)

-- Conditions for the second part
def m := line 3 (-2) (-6)

-- Prove the desired results
theorem symmetric_point_coords :
  let A1 := (-(33 / 13), 4 / 13) in
  ∃ (A1 : ℝ × ℝ), is_symmetric_point l1 A A1 :=
begin
  sorry
end

theorem symmetric_line_eq :
  ∃ (l2 : ℝ → ℝ → Prop), 
  (∀ x y : ℝ, l2 x y ↔ 3 * x - 11 * y + 34 = 0) :=
begin
  sorry
end

end symmetric_point_coords_symmetric_line_eq_l510_510410


namespace eccentricity_ellipse_l510_510394

noncomputable def ellipse (a b : ℝ) := 
  a > b ∧ b > 0 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

noncomputable def foci (a b : ℝ) : set (ℝ × ℝ) := 
  {(c, 0), (-c, 0) | c = real.sqrt (a^2 - b^2)}

noncomputable def projection_condition {x y a b c : ℝ} (P : ℝ × ℝ) : Prop :=
  let F1 := (-c, 0) in 
  let F2 := (c, 0) in 
  let PF1 := (fst P - fst F1, snd P - snd F1) in 
  let PF2 := (fst F2 - fst P, snd F2 - snd P) in 
  let F1F2 := (fst F2 - fst F1, snd F2 - snd F1) in
  ∥PF1∥ = real.sqrt (3) * c ∧ ∥PF2∥ = c

noncomputable def angle_condition {x y a b : ℝ} (P : ℝ × ℝ) : Prop :=
  acos ((P.1 * (-c)) + (P.2 * 0) / (abs (P.1) * abs (-c))) = π / 6

theorem eccentricity_ellipse {a b e : ℝ} (P : ℝ × ℝ) 
  (h_ellipse : ellipse a b)
  (h_proj : projection_condition P)
  (h_angle : angle_condition P)
  (h_focus : foci a b) : e = real.sqrt(3) - 1 :=
sorry

end eccentricity_ellipse_l510_510394


namespace compute_seventy_five_squared_minus_thirty_five_squared_l510_510686

theorem compute_seventy_five_squared_minus_thirty_five_squared :
  75^2 - 35^2 = 4400 := by
  sorry

end compute_seventy_five_squared_minus_thirty_five_squared_l510_510686


namespace Malcolm_facebook_followers_l510_510118

theorem Malcolm_facebook_followers :
  ∃ F : ℕ, 
    (let instagram := 240 in
    let twitter := (240 + F) / 2 in
    let tiktok := 3 * twitter in
    let youtube := tiktok + 510 in
    instagram + F + twitter + tiktok + youtube = 3840) → 
    F = 375 :=
by
  sorry

end Malcolm_facebook_followers_l510_510118


namespace mean_temperature_for_week_l510_510548

theorem mean_temperature_for_week :
  let temps : List ℝ := [80, 79, 81, 85, 87, 89, 87] in
  (temps.sum / temps.length) = 84 :=
by
  sorry

end mean_temperature_for_week_l510_510548


namespace least_integer_square_eq_12_more_than_three_times_l510_510580

theorem least_integer_square_eq_12_more_than_three_times (x : ℤ) (h : x^2 = 3 * x + 12) : x = -3 :=
sorry

end least_integer_square_eq_12_more_than_three_times_l510_510580


namespace dave_total_rides_l510_510303

theorem dave_total_rides (rides_first_day rides_second_day : ℕ) (h1 : rides_first_day = 4) (h2 : rides_second_day = 3) :
  rides_first_day + rides_second_day = 7 :=
by
  sorry

end dave_total_rides_l510_510303


namespace solve_equation_l510_510146

-- Define log functions and the given equation
def equation (x : ℝ) : Prop := log10 (4^x + 3) = log10 (2^x) + log10 4

-- State the theorem to be proved
theorem solve_equation : ∀ x : ℝ, equation x -> (x = 0 ∨ x = log 2 3) :=
by
  intro x
  intro h
  sorry

end solve_equation_l510_510146


namespace vanessa_points_l510_510188

theorem vanessa_points (total_points : ℕ) (average_other_players : ℝ) (num_other_players : ℕ) (points_scored_by_team : ℝ) (V : ℝ) :
  total_points = 48 ∧ average_other_players = 3.5 ∧ num_other_players = 6 ∧ 
  points_scored_by_team = (average_other_players * num_other_players) ∧ 
  points_scored_by_team + V = real.of_nat total_points → V = 27 :=
by
  sorry

end vanessa_points_l510_510188


namespace dislike_both_tv_and_video_games_l510_510134

theorem dislike_both_tv_and_video_games (total_people : ℕ) (percent_dislike_tv : ℝ) (percent_dislike_tv_and_games : ℝ) :
  let people_dislike_tv := percent_dislike_tv * total_people
  let people_dislike_both := percent_dislike_tv_and_games * people_dislike_tv
  total_people = 1800 ∧ percent_dislike_tv = 0.4 ∧ percent_dislike_tv_and_games = 0.25 →
  people_dislike_both = 180 :=
by {
  sorry
}

end dislike_both_tv_and_video_games_l510_510134


namespace find_p_minus_q_l510_510387

theorem find_p_minus_q (p q : ℝ) (h1 : is_root (λ x : ℂ => 2*x^2 + ↑p*x + ↑q) (2*complex.i - 3))
                        (h2 : is_root (λ x : ℂ => 2*x^2 + ↑p*x + ↑q) (-2*complex.i - 3)) :
    p - q = -14 :=
by
  sorry

end find_p_minus_q_l510_510387


namespace find_resistance_x_l510_510059

theorem find_resistance_x (y r x : ℝ) (h₁ : y = 5) (h₂ : r = 1.875) (h₃ : 1/r = 1/x + 1/y) : x = 3 :=
by
  sorry

end find_resistance_x_l510_510059


namespace pies_not_eaten_with_forks_l510_510996

def percentage_pies_eaten_with_forks: ℝ := 0.68
def total_pies: ℕ := 2000

theorem pies_not_eaten_with_forks : 
  let pies_not_with_forks := total_pies * (1 - percentage_pies_eaten_with_forks) in
  pies_not_with_forks = 640 :=
by
  sorry

end pies_not_eaten_with_forks_l510_510996


namespace find_omega_l510_510807

-- Define the two functions and the condition
def f (ω x : ℝ) : ℝ := Real.sin (ω * x - π / 6)
def g (ω x : ℝ) : ℝ := Real.cos (ω * x)

theorem find_omega :
  (∃ ω : ℝ, ∀ x : ℝ,  Real.sin (ω * (x + π / 3) - π / 6) = Real.cos (ω * x) → ω = 2) := 
sorry

end find_omega_l510_510807


namespace average_infections_l510_510266

theorem average_infections (x : ℝ) (h : 1 + x + x^2 = 121) : x = 10 :=
sorry

end average_infections_l510_510266


namespace tan_mul_tan_l510_510794

variables {α β : ℝ}

theorem tan_mul_tan (h : 3 * Real.cos (2 * α + β) + 5 * Real.cos β = 0) : 
  Real.tan (α + β) * Real.tan α = -4 :=
sorry

end tan_mul_tan_l510_510794


namespace limit_omega_l510_510038

open Real

noncomputable def harmonic_number (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), (1 : ℝ) / (k + 1)

noncomputable def omega_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, ∫ (x : ℝ) in (-1 / (k + 1))..(1 / (k + 1)),
    (2 * x ^ 10 + 3 * x ^ 8 + 1) * real.acos ((k + 1) * x)

noncomputable def zeta (s : ℝ) : ℝ :=
  ∑' (k : ℕ) in filter ((· ≠ 0) : ℕ → Prop) Finset.univ, 1 / (k ^ s)

theorem limit_omega : 
  tendsto (λ n, omega_n n - π * harmonic_number n) at_top (nhds (π * ((2 / 11) * zeta 11 + (3 / 9) * zeta 9))) :=
by
  sorry

end limit_omega_l510_510038


namespace floor_sum_inequality_l510_510606

theorem floor_sum_inequality (n : ℕ) (a : Fin n → ℕ) :
  (⌊(Finset.univ.sum (λ i, (a i)^2 / a (i + 1 % n)) : ℝ)⌋ : ℕ) ≥ n := 
sorry

end floor_sum_inequality_l510_510606


namespace permutations_remainder_l510_510853

theorem permutations_remainder :
  let s := "AAAAABBBBBCCCCCDD".toList
  ∃N : ℕ, N = (number_of_permutations s (λ x, ¬(A ∈ x.take 5) ∧ ¬(B ∈ x.drop 5.take 6) ∧ ¬(C ∈ x.drop 11)) % 1000) ∧ N = 340 :=
sorry

end permutations_remainder_l510_510853


namespace parallel_vectors_x_value_l510_510731

theorem parallel_vectors_x_value :
  ∀ x : ℝ, (2, 3) ∥ (x, -6) → x = -4 :=
by
  sorry

end parallel_vectors_x_value_l510_510731


namespace duck_weight_l510_510844

-- Definitions based on conditions
def num_ducks : ℕ := 30
def cost_per_duck : ℝ := 10
def selling_price_per_pound : ℝ := 5
def profit : ℝ := 300
def total_cost := num_ducks * cost_per_duck
def total_revenue := total_cost + profit
def total_weight := total_revenue / selling_price_per_pound
def weight_per_duck := total_weight / num_ducks

-- Theorem to be proved
theorem duck_weight : weight_per_duck = 4 := 
by sorry

end duck_weight_l510_510844


namespace triangle_area_l510_510952

theorem triangle_area (P : ℝ × ℝ)
  (Q : ℝ × ℝ) (R : ℝ × ℝ)
  (P_eq : P = (3, 2))
  (Q_eq : ∃ b, Q = (7/3, 0) ∧ 2 = 3 * 3 + b ∧ 0 = 3 * (7/3) + b)
  (R_eq : ∃ b, R = (4, 0) ∧ 2 = -2 * 3 + b ∧ 0 = -2 * 4 + b) :
  (1/2) * abs (Q.1 - R.1) * abs (P.2) = 5/3 :=
by
  sorry

end triangle_area_l510_510952


namespace log4_80_cannot_be_found_without_additional_values_l510_510813

-- Conditions provided in the problem
def log4_16 : Real := 2
def log4_32 : Real := 2.5

-- Lean statement of the proof problem
theorem log4_80_cannot_be_found_without_additional_values :
  ¬(∃ (log4_80 : Real), log4_80 = log4_16 + log4_5) :=
sorry

end log4_80_cannot_be_found_without_additional_values_l510_510813


namespace five_letter_words_start_end_same_l510_510084

def num_five_letter_words_start_end_same : ℕ :=
  26 ^ 4

theorem five_letter_words_start_end_same :
  num_five_letter_words_start_end_same = 456976 :=
by
  -- Sorry is used as a placeholder for the proof.
  sorry

end five_letter_words_start_end_same_l510_510084


namespace inequality_proof_l510_510106

theorem inequality_proof (n : ℕ) 
  (x : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < x i) 
  (h_sum : (∑ i, x i) = 1) :
  (∏ i, (1 / x i - 1)) ≥ (n - 1) ^ n :=
sorry

end inequality_proof_l510_510106


namespace sasha_claim_l510_510737

noncomputable def largest_k (n : ℕ) : ℕ :=
  2 * Nat.ceil (n / 2 : ℝ)

theorem sasha_claim (n : ℕ) (rays : Fin n → Ray) (h : ∀ i j, rays i ≠ rays j):
  ∃ k, (∀ (points : Fin k → Point), ∃ (s : Sphere), ∀ i, points i ∈ s) ∧ 
       k = largest_k n :=
by
  sorry

end sasha_claim_l510_510737


namespace problem_conditions_find_period_of_f_l510_510774

def f (x : ℝ) : ℝ := sin ((7 * Real.pi / 6) - 2 * x) - 2 * (sin x)^2 + 1

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem problem_conditions (A b c : ℝ) (h1 : f A = 1/2) (h2 : 2 * A = b + c) (h3 : b*c = 18)
  (h4 : b + c = 2 * A) (h5 : cos A = 1/2) : a = 3 * sqrt 2 :=
sorry

theorem find_period_of_f : is_periodic f Real.pi :=
sorry 

end problem_conditions_find_period_of_f_l510_510774


namespace average_episodes_per_year_is_16_l510_510614

-- Define the number of years the TV show has been running
def years : Nat := 14

-- Define the number of seasons and episodes for each category
def seasons_8_15 : Nat := 8
def episodes_per_season_8_15 : Nat := 15
def seasons_4_20 : Nat := 4
def episodes_per_season_4_20 : Nat := 20
def seasons_2_12 : Nat := 2
def episodes_per_season_2_12 : Nat := 12

-- Define the total number of episodes
def total_episodes : Nat :=
  (seasons_8_15 * episodes_per_season_8_15) + 
  (seasons_4_20 * episodes_per_season_4_20) + 
  (seasons_2_12 * episodes_per_season_2_12)

-- Define the average number of episodes per year
def average_episodes_per_year : Nat :=
  total_episodes / years

-- State the theorem to prove the average number of episodes per year is 16
theorem average_episodes_per_year_is_16 : average_episodes_per_year = 16 :=
by
  sorry

end average_episodes_per_year_is_16_l510_510614


namespace bob_coloring_l510_510676

/-
  Problem:
  Find the number of ways to color five points in {(x, y) | 1 ≤ x, y ≤ 5} blue 
  such that the distance between any two blue points is not an integer.
-/

def is_integer_distance (p1 p2 : ℤ × ℤ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let d := Int.gcd ((x2 - x1)^2 + (y2 - y1)^2)
  d ≠ 1

def valid_coloring (points : List (ℤ × ℤ)) : Prop :=
  points.length = 5 ∧ 
  (∀ (p1 p2 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 → ¬ is_integer_distance p1 p2)

theorem bob_coloring : ∃ (points : List (ℤ × ℤ)), valid_coloring points ∧ points.length = 80 :=
sorry

end bob_coloring_l510_510676


namespace range_of_m_l510_510026

noncomputable def f (x m : ℝ) : ℝ :=
  x^2 + m

noncomputable def g (x m : ℝ) : ℝ :=
  (1/2)^x - m

theorem range_of_m (m : ℝ)
  (h : ∀ x1 ∈ set.Icc (-1 : ℝ) 3, ∃ x2 ∈ set.Icc (0 : ℝ) 2, f x1 m ≥ g x2 m) :
  m ≥ 1/8 :=
sorry

end range_of_m_l510_510026


namespace geometric_seq_problem_l510_510830

noncomputable def geometric_sequence (aₙ : ℕ → ℝ) : Prop := 
∃ q : ℝ, ∀ n : ℕ, aₙ (n + 1) = q * aₙ n

theorem geometric_seq_problem (aₙ : ℕ → ℝ) 
  (h₁ : aₙ 7 * aₙ 11 = 6)
  (h₂ : aₙ 4 + aₙ 14 = 5)
  (h₃ : geometric_sequence aₙ) :
  ∃ k : ℝ, k ∈ ({3 / 2, 2 / 3} : set ℚ) ∧ k = aₙ 20 / aₙ 10 :=
sorry

end geometric_seq_problem_l510_510830


namespace weekly_rental_fee_percentage_l510_510088

theorem weekly_rental_fee_percentage
  (camera_value : ℕ)
  (rental_period_weeks : ℕ)
  (friend_percentage : ℚ)
  (john_paid : ℕ)
  (percentage : ℚ)
  (total_rental_fee : ℚ)
  (weekly_rental_fee : ℚ)
  (P : ℚ)
  (camera_value_pos : camera_value = 5000)
  (rental_period_weeks_pos : rental_period_weeks = 4)
  (friend_percentage_pos : friend_percentage = 0.40)
  (john_paid_pos : john_paid = 1200)
  (percentage_pos : percentage = 1 - friend_percentage)
  (total_rental_fee_calc : total_rental_fee = john_paid / percentage)
  (weekly_rental_fee_calc : weekly_rental_fee = total_rental_fee / rental_period_weeks)
  (weekly_rental_fee_equation : weekly_rental_fee = P * camera_value)
  (P_calc : P = weekly_rental_fee / camera_value) :
  P * 100 = 10 := 
by 
  sorry

end weekly_rental_fee_percentage_l510_510088


namespace wicket_keeper_age_l510_510156

/-- The cricket team consists of 11 members with an average age of 22 years.
    One member is 25 years old, and the wicket keeper is W years old.
    Excluding the 25-year-old and the wicket keeper, the average age of the remaining players is 21 years.
    Prove that the wicket keeper is 6 years older than the average age of the team. -/
theorem wicket_keeper_age (W : ℕ) (team_avg_age : ℕ := 22) (total_team_members : ℕ := 11) 
                          (other_member_age : ℕ := 25) (remaining_avg_age : ℕ := 21) :
    W = 28 → W - team_avg_age = 6 :=
by
  intros
  sorry

end wicket_keeper_age_l510_510156


namespace sum_log_difference_l510_510312

theorem sum_log_difference :
  (∑ k in Finset.range 501, k * (⌈Real.log k / Real.log (Real.sqrt 3)⌉ - ⌊Real.log k / Real.log (Real.sqrt 3)⌋)) = 124886 :=
by sorry

end sum_log_difference_l510_510312


namespace child_gender_events_l510_510628

-- Declare the types for Gender and the pairs of children's genders.
inductive Gender
| Male : Gender
| Female : Gender

def childPairs : List (Gender × Gender) :=
  [(Gender.Male, Gender.Male), (Gender.Male, Gender.Female),
   (Gender.Female, Gender.Male), (Gender.Female, Gender.Female)]

theorem child_gender_events : 
  (∀ (a b : Gender), (a, b) ∈ List.head childPairs) ↔ 
  ((Gender.Male, Gender.Female) ∈ childPairs ∧ 
   (Gender.Male, Gender.Male) ∈ childPairs ∧ 
   (Gender.Female, Female.Male) ∈ childPairs ∧ 
   (Gender.Female, Gender.Female) ∈ childPairs) :=
by
  sorry

end child_gender_events_l510_510628


namespace nonneg_diff_roots_l510_510581

theorem nonneg_diff_roots : 
  ∀ (a b c : ℤ), a = 1 → b = 42 → c = 384 → 
  let Δ = b * b - 4 * a * c in
  Δ ≥ 0 → 
  let r1 := (-b + Δ.sqrt) / (2 * a) in
  let r2 := (-b - Δ.sqrt) / (2 * a) in
  abs (r1 - r2) = 8 :=
by
  sorry

end nonneg_diff_roots_l510_510581


namespace number_of_ways_to_balance_l510_510105

theorem number_of_ways_to_balance (n : ℕ) (h : n > 0) :
  ∑ (i : ℕ) in range (2*n-1), ite (even i) (factorial i) 0 = factorial (n - 1) :=
sorry

end number_of_ways_to_balance_l510_510105


namespace number_of_paths_to_spell_MATH_l510_510693

theorem number_of_paths_to_spell_MATH : 
  let staggered_grid := [ ["", "", "", "", "M", "", "", ""], 
                          ["", "", "", "M", "A", "", "", ""], 
                          ["", "", "M", "A", "T", "", "", ""], 
                          ["", "M", "A", "T", "H", "", "", ""] ]
  ∀ (grid : List (List String)) (path_segments : List (List (Nat × Nat))),
  grid = staggered_grid → -- This condition sets the initial grid formation
  (∀ (segment : (Nat × Nat)), segment ∈ path_segments → 
      -- Ensuring all path segments are valid with respect to adjacency and movement.
      ∃ (x y : Nat), 
      ((x, y) = (segment.1, segment.2 + 1)) ∨ -- Valid rightward movement
      ((x, y) = (segment.1 + 1, segment.2)) → -- Valid downward movement
      (segment.1 < grid.length) → (segment.2 < grid.head.length))
  → -- Ensuring that segments are within the bounds of the grid.
  (∃ (paths : List (List (Nat × Nat))), -- Paths that spell "MATH"
  (∀ (path : List (Nat × Nat)), 
      ((path.1.2 = "M") → (path.last.2 = "H")) ∧ 
      (path.length = 4) ∧ -- Path length must be 4 to spell "MATH"
      ∀ i, (i < path.length - 1 → path_segments ∋ (path.i, path.i + 1)) -- Path segments follow adjacency rules
  )
  → paths.length = 104 -- Correct answer for solution

end number_of_paths_to_spell_MATH_l510_510693


namespace price_per_gallon_l510_510080

-- constant definitions for the problem
constant rain_gallons_per_inch : ℕ
constant rain_monday_inches : ℕ
constant rain_tuesday_inches : ℕ
constant total_sales_dollars : ℚ

-- conditions from the problem
axiom h1 : rain_gallons_per_inch = 15
axiom h2 : rain_monday_inches = 4
axiom h3 : rain_tuesday_inches = 3
axiom h4 : total_sales_dollars = 126

-- the goal to prove
theorem price_per_gallon : total_sales_dollars / (rain_gallons_per_inch * (rain_monday_inches + rain_tuesday_inches)) = 1.20 :=
by sorry

end price_per_gallon_l510_510080


namespace parallelogram_area_is_correct_l510_510635

-- Define the base and height of the parallelogram
def base_length : ℝ := 20
def height : ℝ := 4

-- Define the area function for a parallelogram
def area_parallelogram (b h : ℝ) : ℝ :=
  b * h

-- The main theorem statement
theorem parallelogram_area_is_correct :
  area_parallelogram base_length height = 80 :=
by
  sorry

end parallelogram_area_is_correct_l510_510635


namespace train_pass_time_l510_510977

def train_length : ℝ := 110  -- length of the train in meters
def train_speed_kmh : ℝ := 30  -- speed of the train in km/h
def man_speed_kmh : ℝ := 3  -- speed of the man in km/h
def conversion_factor : ℝ := 5 / 18  -- conversion factor from km/h to m/s
def expected_time : ℝ := 12  -- expected time to pass the man in seconds

theorem train_pass_time :
  let relative_speed_kmh := train_speed_kmh + man_speed_kmh in
  let relative_speed_ms := relative_speed_kmh * conversion_factor in
  let time_to_pass := train_length / relative_speed_ms in
  time_to_pass ≈ expected_time := 
by
  sorry

end train_pass_time_l510_510977


namespace locus_of_points_equidistant_from_axes_l510_510914

-- Define the notion of being equidistant from the x-axis and the y-axis
def is_equidistant_from_axes (P : (ℝ × ℝ)) : Prop :=
  abs P.1 = abs P.2

-- The proof problem: given a moving point, the locus equation when P is equidistant from both axes
theorem locus_of_points_equidistant_from_axes (x y : ℝ) :
  is_equidistant_from_axes (x, y) → abs x - abs y = 0 :=
by
  intros h
  exact sorry

end locus_of_points_equidistant_from_axes_l510_510914


namespace fraction_spent_on_furniture_l510_510502

theorem fraction_spent_on_furniture (original_savings : ℝ) (cost_of_tv : ℝ) (f : ℝ)
  (h1 : original_savings = 1800) 
  (h2 : cost_of_tv = 450) 
  (h3 : f * original_savings + cost_of_tv = original_savings) :
  f = 3 / 4 := 
by 
  sorry

end fraction_spent_on_furniture_l510_510502


namespace perpendicular_implies_norm_dot_product_neg_implies_range_l510_510790

noncomputable def a (x : ℝ) : ℝ × ℝ := (x, -1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x - 2, 3)
noncomputable def c (x : ℝ) : ℝ × ℝ := (1 - 2 * x, 6)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def norm (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 ^ 2 + u.2 ^ 2)
noncomputable def d (x : ℝ) : ℝ × ℝ :=
  let u := b x
  let v := c x
  (2 * u.1 + v.1, 2 * u.2 + v.2)

theorem perpendicular_implies_norm (x : ℝ) (h : dot_product (a x) (d x) = 0) :
  norm (b x) = 3 * Real.sqrt 5 := sorry

theorem dot_product_neg_implies_range (x : ℝ) (h : dot_product (a x) (b x) < 0) :
  -1 < x ∧ x < 3 := sorry

end perpendicular_implies_norm_dot_product_neg_implies_range_l510_510790


namespace min_marked_cells_l510_510960

theorem min_marked_cells (m n : ℕ) (board : Fin m × Fin n)
  (h_m : m = 8) (h_n : n = 9) :
  ∃ k : ℕ, (∀ (shape : Fin 2 × Fin 2) (pos : Fin m × Fin n), 
             ∃ (marked_pos : Fin m × Fin n), 
             marked_pos ∈ set_of_marked_cells k) 
  ∧ k = 16 := 
by
  sorry

noncomputable def set_of_marked_cells (k : ℕ) : set (Fin 8 × Fin 9) := sorry

end min_marked_cells_l510_510960


namespace parabola_decreasing_l510_510412

noncomputable def parabola (b c : ℝ) : ℝ → ℝ := λ x, -x^2 + b * x + c

theorem parabola_decreasing (b c : ℝ) :
  ∀ x, x > 3 → ∀ x', x' > x → parabola b c x' < parabola b c x :=
by
  sorry

end parabola_decreasing_l510_510412


namespace rectangles_vertex_sum_lt_l510_510238

theorem rectangles_vertex_sum_lt (n : ℕ) (R : Type) (v : R → ℕ) : 
    -- Conditions: 
    (∀ (rects : fin n → set ℝ × ℝ), 
    (∀ i j : fin n, rects i ≠ rects j → 
    sides_parallel (rects i) (rects j) ∧ sides_distinct_lines (rects i) (rects j)) →
    -- Question: show the sum of vertices over the regions
    (∑ (r in {r | ∃ i, r ∈ rects i}), v r < 40 * n))
    sorry

end rectangles_vertex_sum_lt_l510_510238


namespace complementary_events_C_and_D_independent_events_B_and_C_l510_510933

def event_A (balls : List ℕ) (draw : List ℕ) : Prop := (draw.head % 2 = 0)

def event_B (balls : List ℕ) (draw : List ℕ) : Prop := (draw.tail.head % 2 = 1)

def event_C (draw : List ℕ) : Prop := ((draw.head + draw.tail.head) % 2 = 0)

def event_D (draw : List ℕ) : Prop := ((draw.head + draw.tail.head) % 2 = 1)

def all_balls : List ℕ := [1, 2, 3, 4, 5, 6]

def draw_without_replacement : List (List ℕ) :=
  (all_balls.bind (λ x, all_balls.filter (λ y, y ≠ x).map (λ y, [x, y])))

theorem complementary_events_C_and_D :
  ∀ draw ∈ draw_without_replacement, event_C draw ∨ event_D draw :=
by
  intros
  sorry

theorem independent_events_B_and_C :
  ∀ draw ∈ draw_without_replacement, (event_B all_balls draw ∧ event_C draw) = (event_B all_balls draw ∧ (event_C draw)) :=
by
  intros
  sorry

end complementary_events_C_and_D_independent_events_B_and_C_l510_510933


namespace graph_shift_l510_510179

theorem graph_shift (x : ℝ) :
  ∀ x, sin (2 * x - π / 3) = sin(2 * (x - π / 6)) := 
  sorry

end graph_shift_l510_510179


namespace marie_needs_8_days_to_pay_for_cash_register_l510_510503

-- Definitions of the conditions
def cost_of_cash_register : ℕ := 1040
def price_per_loaf : ℕ := 2
def loaves_per_day : ℕ := 40
def price_per_cake : ℕ := 12
def cakes_per_day : ℕ := 6
def daily_rent : ℕ := 20
def daily_electricity : ℕ := 2

-- Derive daily income and expenses
def daily_income : ℕ := (price_per_loaf * loaves_per_day) + (price_per_cake * cakes_per_day)
def daily_expenses : ℕ := daily_rent + daily_electricity
def daily_profit : ℕ := daily_income - daily_expenses

-- Define days needed to pay for the cash register
def days_needed : ℕ := cost_of_cash_register / daily_profit

-- Proof goal
theorem marie_needs_8_days_to_pay_for_cash_register : days_needed = 8 := by
  sorry

end marie_needs_8_days_to_pay_for_cash_register_l510_510503


namespace obtuse_angle_sufficient_not_necessary_l510_510988

-- Definitions for the conditions
def is_obtuse_angle (α : ℝ) : Prop :=
  π / 2 < α ∧ α < π

-- Statement to be proven
theorem obtuse_angle_sufficient_not_necessary (α : ℝ) :
  (is_obtuse_angle α → (sin α > 0 ∧ cos α < 0)) ∧
  (¬((sin α > 0 ∧ cos α < 0) → is_obtuse_angle α)) :=
sorry

end obtuse_angle_sufficient_not_necessary_l510_510988


namespace triangle_is_right_at_X_l510_510484

-- Definitions corresponding to the conditions of the problem
variable (Γ1 Γ2 : Type)
variable [circle Γ1] [circle Γ2]
variable (X : point)
variable (tangent : line)
variable [externallyTangent Γ1 Γ2 X]
variable [tangentToLine Γ1 tangent Y]
variable [tangentToLine Γ2 tangent Z]
variable [doesNotPassThrough tangent X]

-- Theorem statement corresponding to the question
theorem triangle_is_right_at_X :
  is_right_triangle XYZ X :=
sorry

end triangle_is_right_at_X_l510_510484


namespace sum_of_digits_range_l510_510703

-- Definitions based on the given conditions
def vertex_sums : List ℕ := [6, 7, 9, 10, 11, 12, 14, 15]

-- Main theorem stating the problem in Lean 4
theorem sum_of_digits_range (n : ℕ) (h : 48 ≤ n ∧ n ≤ 120) : 
  ∃ (cube : List ℕ), (∀ v ∈ cube, v ∈ vertex_sums) ∧ list.sum cube = n ∧ cube.length = 8 :=
sorry

end sum_of_digits_range_l510_510703


namespace cyclic_sum_inequality_l510_510005

theorem cyclic_sum_inequality
  (a b c d e : ℝ)
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e)
  (h6 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1 / 5 :=
by
  sorry

end cyclic_sum_inequality_l510_510005


namespace white_checker_capture_impossible_l510_510246

-- Definitions for checkerboard and checkers placement
def checkerboard := ℕ × ℕ  -- an infinite checkerboard
def black_checker_positions : set checkerboard := {⟨0, 0⟩, ⟨1, 1⟩}  -- two diagonally adjacent squares with black checkers

-- Checker's capturing rule
def can_capture (start : checkerboard) (jump : checkerboard) (end : checkerboard) : Prop :=
  (start, jump, end) = ((x, y), (x+1, y+1), (x+2, y+2)) ∨ 
  (start, jump, end) = ((x, y), (x-1, y-1), (x-2, y-2))

-- Theorem statement
theorem white_checker_capture_impossible : ¬ ∃ (white_checker : checkerboard) 
  (extra_black : set checkerboard), 
  ∀ black ∈ black_checker_positions ∪ extra_black,
    ∃ (path : list (checkerboard × checkerboard × checkerboard)), 
      ∀ t ∈ path, 
      (can_capture (white_checker :: (path.map prod.snd)).head.1 t.2.1 t.2.2) :=
sorry

end white_checker_capture_impossible_l510_510246


namespace total_shaded_area_l510_510067

theorem total_shaded_area (r R : ℝ) (h1 : π * R^2 = 100 * π) (h2 : r = R / 2) : 
    (1/4) * π * R^2 + (1/4) * π * r^2 = 31.25 * π :=
by
  sorry

end total_shaded_area_l510_510067


namespace nat_diff_same_prime_divisors_l510_510142

theorem nat_diff_same_prime_divisors (n : ℕ) : 
  ∃ a b : ℤ, a.number_of_prime_divisors = b.number_of_prime_divisors ∧ n = a - b := 
sorry

end nat_diff_same_prime_divisors_l510_510142


namespace total_packs_l510_510872

theorem total_packs (cards_per_person cards_per_pack : ℕ) (num_people : ℕ) 
  (h1 : cards_per_person = 540) 
  (h2 : cards_per_pack = 20) 
  (h3 : num_people = 4) : 
  (cards_per_person / cards_per_pack) * num_people = 108 := 
by
  sorry

end total_packs_l510_510872


namespace remainder_mod_l510_510589

theorem remainder_mod (a b : ℕ) (ha : a ≡ 7 [MOD 8]) (hb : b ≡ 1 [MOD 8]) : ((a - b) * (b + a)) % 8 = 0 := by
  sorry

example : ((71 ^ 7 - 73 ^ 10) * (73 ^ 5 + 71 ^ 3)) % 8 = 0 := 
begin
  have h1 : 71 % 8 = 7, by norm_num,
  have h2 : 73 % 8 = 1, by norm_num,
  have h3 : (71 ^ 7) % 8 = 7 % 8, -- Computing 71^7 mod 8
  { rw ← mod_pow (71 : ℕ) 7 8,
    norm_num },
  have h4 : (73 ^ 10) % 8 = 1 % 8, -- Computing 73^10 mod 8
  { rw ← mod_pow (73 : ℕ) 10 8,
    norm_num },
  have h5 : (73 ^ 5) % 8 = 1 % 8, -- Computing 73^5 mod 8
  { rw ← mod_pow (73 : ℕ) 5 8,
    norm_num },
  have h6 : (71 ^ 3) % 8 = 7 % 8, -- Computing 71^3 mod 8
  { rw ← mod_pow (71 : ℕ) 3 8,
    norm_num },
  exact remainder_mod (71 ^ 7) (73 ^ 10) h3 h4,
end

end remainder_mod_l510_510589


namespace length_of_intercepted_segment_l510_510460

-- definitions for conditions
def line_in_polar (p θ : ℝ) : Prop := p * cos θ = 1
def circle_in_polar (p θ : ℝ) : Prop := p = 4 * cos θ

-- statement of the theorem
theorem length_of_intercepted_segment : 
  ∀ (p θ : ℝ), circle_in_polar p θ → line_in_polar p θ → 2 * sqrt 3 = 2 * sqrt 3 :=
by
  intros p θ h_circle h_line
  sorry

end length_of_intercepted_segment_l510_510460


namespace rectangle_symmetry_l510_510969

-- Definitions of symmetry properties
def isAxisymmetric (shape : Type) : Prop := sorry
def isCentrallySymmetric (shape : Type) : Prop := sorry

-- Specific shapes
def EquilateralTriangle : Type := sorry
def Parallelogram : Type := sorry
def Rectangle : Type := sorry
def RegularPentagon : Type := sorry

-- The theorem we want to prove
theorem rectangle_symmetry : 
  isAxisymmetric Rectangle ∧ isCentrallySymmetric Rectangle := sorry

end rectangle_symmetry_l510_510969


namespace square_area_l510_510150

theorem square_area (XY ZQ : ℕ) (inscribed_square : Prop) : (XY = 35) → (ZQ = 65) → inscribed_square → ∃ (a : ℕ), a^2 = 2275 :=
by
  intros hXY hZQ hinscribed
  use 2275
  sorry

end square_area_l510_510150


namespace simplify_expr_one_compute_expr_two_l510_510893

-- Problem 1
theorem simplify_expr_one: 
  ( (27/8 : ℝ) ^ (-2/3) - (49/9 : ℝ) ^ (1/2) + (0.008 : ℝ) ^ (-2/3) * (2/25) ) = (1/9 : ℝ) :=
sorry

-- Problem 2
theorem compute_expr_two: 
  ( (Real.log 5 * Real.log 8000 + (Real.log (2 ^ Real.sqrt 3)) ^ 2) / (Real.log 600 - 1/2 * Real.log 36 - 1/2 * Real.log 0.01) ) = 1 :=
sorry

end simplify_expr_one_compute_expr_two_l510_510893


namespace megan_final_balance_percentage_l510_510870

noncomputable def initial_balance_usd := 125.0
noncomputable def increase_percentage_babysitting := 0.25
noncomputable def exchange_rate_usd_to_eur_1 := 0.85
noncomputable def decrease_percentage_shoes := 0.20
noncomputable def exchange_rate_eur_to_usd := 1.15
noncomputable def increase_percentage_stocks := 0.15
noncomputable def decrease_percentage_medical := 0.10
noncomputable def exchange_rate_usd_to_eur_2 := 0.88

theorem megan_final_balance_percentage :
  let new_balance_after_babysitting := initial_balance_usd * (1 + increase_percentage_babysitting)
  let balance_in_eur := new_balance_after_babysitting * exchange_rate_usd_to_eur_1
  let balance_after_shoes := balance_in_eur * (1 - decrease_percentage_shoes)
  let balance_back_to_usd := balance_after_shoes * exchange_rate_eur_to_usd
  let balance_after_stocks := balance_back_to_usd * (1 + increase_percentage_stocks)
  let balance_after_medical := balance_after_stocks * (1 - decrease_percentage_medical)
  let final_balance_in_eur := balance_after_medical * exchange_rate_usd_to_eur_2
  let initial_balance_in_eur := initial_balance_usd * exchange_rate_usd_to_eur_1
  (final_balance_in_eur / initial_balance_in_eur) * 100 = 104.75 := by
  sorry

end megan_final_balance_percentage_l510_510870


namespace cube_diagonal_angle_min_l510_510718

-- Define a structure for a cube
structure Cube :=
  (vertices : set ℝ)
  (A : ℝ)
  (C1 : ℝ)

-- Define the condition of the cube
def is_diagonal (A C1 : ℝ) : Prop := true -- Placeholder for the actual diagonal condition

-- Define the points on the cube surface excluding A and C1
def cube_surface_points (cube : Cube) : set ℝ :=
  cube.vertices \ {cube.A, cube.C1}

-- Define the main theorem
theorem cube_diagonal_angle_min (cube : Cube) (h : is_diagonal cube.A cube.C1) : 
  ∃ (points : set ℝ), points = cube_surface_points cube ∧
  ∀ P ∈ points, ∠ (cube.A) (P) (cube.C1) = π / 2 :=
sorry

end cube_diagonal_angle_min_l510_510718


namespace shape_is_plane_l510_510354

def spherical_plane (c : ℝ) : Type :=
  { point : ℝ × ℝ × ℝ // let ⟨ρ, θ, φ⟩ := point in θ = c }

theorem shape_is_plane (c : ℝ) :
  ∀ (p : spherical_plane c), ∃ x y z, let ⟨ρ, θ, φ⟩ := (x, y, z) in θ = c ∧ (p.val = (ρ, θ, φ)) → x^2 + y^2 + z^2 = ρ^2 ∧ θ = c :=
by
  sorry

end shape_is_plane_l510_510354


namespace interval_length_implies_difference_l510_510320

theorem interval_length_implies_difference (a b : ℝ) (h : (b - 5) / 3 - (a - 5) / 3 = 15) : b - a = 45 := by
  sorry

end interval_length_implies_difference_l510_510320


namespace simplify_fraction_l510_510518

theorem simplify_fraction (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (15 * x^2 * y^3) / (9 * x * y^2) = 20 := by
  sorry

end simplify_fraction_l510_510518


namespace inequality_proof_l510_510110

theorem inequality_proof (n : ℕ) (x : Fin n → ℝ) (h : ∀ i : Fin n, 0 < x i ∧ x i < 2 * Real.pi) (h_ordered : StrictMono x) :
  ∑ i j in (Finset.univ : Finset (Fin n)).filter (λ p, p.1 ≠ p.2), 
  (1 / |x i - x j| + 1 / (2 * Real.pi - |x i - x j|)) ≥ (n^2 / Real.pi) * (∑ k in Finset.range (n - 1), 1 / (k + 1)) :=
by
  sorry

end inequality_proof_l510_510110


namespace asparagus_spears_needed_is_1200_l510_510306

noncomputable def total_asparagus_spears_needed 
    (bridgettes_guests : ℕ) 
    (alexs_fraction : ℚ) 
    (extra_plates : ℕ) 
    (asparagus_per_plate : ℕ) : ℕ :=
  let alexs_guests := (bridgettes_guests * alexs_fraction).toNat
  let total_guests := bridgettes_guests + alexs_guests
  let total_plates := total_guests + extra_plates
  total_plates * asparagus_per_plate

theorem asparagus_spears_needed_is_1200 :
  total_asparagus_spears_needed 84 (2 / 3) 10 8 = 1200 :=
by
  sorry

end asparagus_spears_needed_is_1200_l510_510306


namespace triangle_areas_l510_510918

theorem triangle_areas (r s : ℝ) (h1 : s = (1/2) * r + 6)
                       (h2 : (12 + r) * ((1/2) * r + 6) = 18) :
  r + s = -3 :=
by
  sorry

end triangle_areas_l510_510918


namespace roots_cosine_triangle_condition_l510_510302

theorem roots_cosine_triangle_condition
  (p q r : ℝ)
  (h : ∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = -p ∧
    ab + bc + ca = q ∧
    abc = -r ∧
    a = b ∧ b = c ∧
    a^2 + b^2 + c^2 + 2*abc = 1 ∧
    a = cos α ∧ b = cos β ∧ c = cos γ ∧ 
    α + β + γ = π) :
  p^2 = 2*q + 2*r + 1 :=
by
  sorry

end roots_cosine_triangle_condition_l510_510302


namespace lines_parallel_if_perpendicular_to_same_plane_l510_510184

-- Define a plane as a placeholder for other properties
axiom Plane : Type
-- Define Line as a placeholder for other properties
axiom Line : Type

-- Definition of what it means for a line to be perpendicular to a plane
axiom perpendicular_to_plane (l : Line) (π : Plane) : Prop

-- Definition of parallel lines
axiom parallel_lines (l1 l2 : Line) : Prop

-- Define the proof problem in Lean 4
theorem lines_parallel_if_perpendicular_to_same_plane
    (π : Plane) (l1 l2 : Line)
    (h1 : perpendicular_to_plane l1 π)
    (h2 : perpendicular_to_plane l2 π) :
    parallel_lines l1 l2 :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l510_510184


namespace students_count_l510_510607

theorem students_count (initial: ℕ) (left: ℕ) (new: ℕ) (result: ℕ) 
  (h1: initial = 31)
  (h2: left = 5)
  (h3: new = 11)
  (h4: result = initial - left + new) : result = 37 := by
  sorry

end students_count_l510_510607


namespace river_width_proof_l510_510639
noncomputable def river_width (V FR D : ℝ) : ℝ := V / (FR * D)

theorem river_width_proof :
  river_width 2933.3333333333335 33.33333333333333 4 = 22 :=
by
  simp [river_width]
  norm_num
  sorry

end river_width_proof_l510_510639


namespace hyperbola_right_focus_l510_510909

theorem hyperbola_right_focus (x y : ℝ) (h : (x^2 / 3) - y^2 = 1) : (2, 0) = (sqrt 4, 0) := 
by 
  sorry

end hyperbola_right_focus_l510_510909


namespace no_possible_arrangement_l510_510836

theorem no_possible_arrangement :
  ¬ ∃ (f : ℕ × ℕ → ℤ), (∀ i j, f (i, j) = 1 ∨ f (i, j) = -1) ∧
  (abs (∑ i in fin_range 600, ∑ j in fin_range 600, f (i, j)) < 90000) ∧
  (∀ (x y : ℕ) (h1 : x ≤ 596) (h2 : y ≤ 594),
    abs (∑ i in fin_range 4, ∑ j in fin_range 6, f (x + i, y + j)) > 4) ∧
  (∀ (x y : ℕ) (h1 : x ≤ 594) (h2 : y ≤ 596),
    abs (∑ i in fin_range 6, ∑ j in fin_range 4, f (x + i, y + j)) > 4) :=
by sorry

end no_possible_arrangement_l510_510836


namespace transformed_point_is_correct_l510_510550

def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

def reflect_xz_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2, p.3)

def rotate_180_about_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2, -p.3)

def reflect_yz_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2, p.3)

def translate (p v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2, p.3 + v.3)

noncomputable def final_point : ℝ × ℝ × ℝ :=
  let p1 := reflect_xz_plane initial_point
  let p2 := rotate_180_about_x p1
  let p3 := reflect_yz_plane p2
  translate p3 (1, -1, 0)

theorem transformed_point_is_correct : final_point = (-1, 1, -2) :=
by
  sorry

end transformed_point_is_correct_l510_510550


namespace find_root_in_interval_l510_510719

theorem find_root_in_interval :
  ∃ x ∈ set.Ioo (5 : ℝ) 6, real.log x = x - 5 :=
by
  sorry

end find_root_in_interval_l510_510719


namespace runners_meet_again_l510_510945

theorem runners_meet_again (v1 v2 v3 : ℝ) (C : ℝ) (t : ℤ) 
  (h1 : v1 = 3.6) (h2 : v2 = 3.8) (h3 : v3 = 4.2) (h4 : C = 400)
  (h5 : 0.2 * (t:ℝ) ≡ 0 [MOD C]) (h6 : 0.4 * (t:ℝ) ≡ 0 [MOD C]) (h7 : 0.6 * (t:ℝ) ≡ 0 [MOD C]) :
  t = 2000 :=
sorry

end runners_meet_again_l510_510945


namespace bob_equals_alice_l510_510457

-- Define conditions as constants
def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.08
def discount_rate : ℝ := 0.25

-- Bob's total calculation
def bob_total : ℝ := (original_price * (1 + tax_rate)) * (1 - discount_rate)

-- Alice's total calculation
def alice_total : ℝ := (original_price * (1 - discount_rate)) * (1 + tax_rate)

-- Theorem statement to be proved
theorem bob_equals_alice : bob_total = alice_total := by sorry

end bob_equals_alice_l510_510457


namespace asparagus_spears_needed_is_1200_l510_510307

noncomputable def total_asparagus_spears_needed 
    (bridgettes_guests : ℕ) 
    (alexs_fraction : ℚ) 
    (extra_plates : ℕ) 
    (asparagus_per_plate : ℕ) : ℕ :=
  let alexs_guests := (bridgettes_guests * alexs_fraction).toNat
  let total_guests := bridgettes_guests + alexs_guests
  let total_plates := total_guests + extra_plates
  total_plates * asparagus_per_plate

theorem asparagus_spears_needed_is_1200 :
  total_asparagus_spears_needed 84 (2 / 3) 10 8 = 1200 :=
by
  sorry

end asparagus_spears_needed_is_1200_l510_510307


namespace integral_evaluation_l510_510327

noncomputable def integral_value : ℝ :=
  ∫ x in -1..1, (x + sin x)

theorem integral_evaluation : integral_value = 0 := by
  sorry

end integral_evaluation_l510_510327


namespace _l510_510787

variables {a_n b_n : ℕ → ℕ} {S T : ℕ → ℕ}

lemma arithmetic_sequences_theorem (h1 : ∀ n, S n = n * (2 * a_n 0 + (n - 1) * a_n 1) / 2)
                                   (h2 : ∀ n, T n = n * (2 * b_n 0 + (n - 1) * b_n 1) / 2)
                                   (h3 : ∀ n, S n / T n = (7 * n + 45) / (n + 3))
                                   (h4 : ∀ n, (a_n n / b_n (2 * n)) ∈ ℤ) :
                                   ∃ n, n = 15 := 
by
  sorry

end _l510_510787


namespace find_rate_percent_l510_510959

theorem find_rate_percent (SI P T : ℝ) (h1 : SI = 160) (h2 : P = 800) (h3 : T = 5) : P * (4:ℝ) * T / 100 = SI :=
by
  sorry

end find_rate_percent_l510_510959


namespace find_a_l510_510114

open ProbabilityTheory

noncomputable def normalDist (mean variance : ℝ) : Measure ℝ := {
  toMeasure := sorry, -- Definition of the normal distribution measure
}

theorem find_a (ξ : ℝ → probability_theory.Pmf ℝ)
  (h1 : ξ = normalDist 3 4)
  (h2 : ∀ a : ℝ, P (ξ < 2 * a - 2) = P (ξ > a + 2)) :
  a = 2 :=
begin
  sorry
end

end find_a_l510_510114


namespace john_total_expenses_l510_510474

theorem john_total_expenses :
  (let epiPenCost := 500
   let yearlyMedicalExpenses := 2000
   let firstEpiPenInsuranceCoverage := 0.75
   let secondEpiPenInsuranceCoverage := 0.60
   let medicalExpensesCoverage := 0.80
   let firstEpiPenCost := epiPenCost * (1 - firstEpiPenInsuranceCoverage)
   let secondEpiPenCost := epiPenCost * (1 - secondEpiPenInsuranceCoverage)
   let totalEpiPenCost := firstEpiPenCost + secondEpiPenCost
   let yearlyMedicalExpensesCost := yearlyMedicalExpenses * (1 - medicalExpensesCoverage)
   let totalCost := totalEpiPenCost + yearlyMedicalExpensesCost
   totalCost) = 725 := sorry

end john_total_expenses_l510_510474


namespace tan_a3a5_equals_sqrt3_l510_510389

noncomputable def geometric_seq_property (a: ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r -- for some common ratio r

theorem tan_a3a5_equals_sqrt3 (a : ℕ → ℝ) 
  (h_geom : geometric_seq_property a)
  (h_cond : a 2 * a 6 + 2 * (a 4)^2 = real.pi) :
  real.tan (a 3 * a 5) = real.sqrt 3 :=
sorry

end tan_a3a5_equals_sqrt3_l510_510389


namespace expressions_for_c_and_d_l510_510418

variables {a b c d r s : ℝ}

-- Conditions of the problem
def first_quadratic (x : ℝ) := x^2 + a * x + b = 0
def second_quadratic (x : ℝ) := x^2 + c * x + d = 0
def roots_r_s : Prop := first_quadratic r ∧ first_quadratic s
def roots_r2_s2 : Prop := second_quadratic (r^2) ∧ second_quadratic (s^2)
def rs_eq_2b : Prop := r * s = 2 * b

-- Target to prove
theorem expressions_for_c_and_d (h_a_b_c_d : rs_eq_2b ∧ roots_r_s ∧ roots_r2_s2) : 
  c = -a^2 + 2 * b ∧ d = b^2 :=
sorry

end expressions_for_c_and_d_l510_510418


namespace valid_number_of_m_values_l510_510090

theorem valid_number_of_m_values : 
  (∃ m : ℕ, 2 ≤ m ∧ m ∣ 420 ∧ 2 ≤ (420 / m)) ∧ ∀ m, 2 ≤ m ∧ m ∣ 420 ∧ 2 ≤ (420 / m) → m > 1  → 
  ∃ n : ℕ, n = 22 :=
by
  sorry

end valid_number_of_m_values_l510_510090


namespace johns_deductions_l510_510087

theorem johns_deductions 
  (earnings : ℝ) (D : ℝ) (tax_paid : ℝ)
  (h_earning : earnings = 100000)
  (h_tax_paid : tax_paid = 12000)
  (h_first_bracket_tax : 20000 * 0.10 = 2000)
  (h_remaining_bracket_tax : (earnings - D - 20000) * 0.20 + 2000 = tax_paid) :
  D = 30000 :=
by
  rw [h_earning, h_tax_paid] at h_remaining_bracket_tax
  have h1 : 2000 + (100000 - D - 20000) * 0.20 = 12000 := h_remaining_bracket_tax
  have h2 : 2000 + 16000 - 0.20 * D = 12000 := by rw [sub_right_comm, sub_self 20000, add_right_comm]
  have h3 : 18000 - 0.20 * D = 12000 := h2
  have h4 : 6000 = 0.20 * D := by linarith
  have h5 : D = 30000 := by linarith
  exact h5

end johns_deductions_l510_510087


namespace train_and_car_combined_time_l510_510284

noncomputable def combined_time (car_time : ℝ) (extra_time : ℝ) : ℝ :=
  car_time + (car_time + extra_time)

theorem train_and_car_combined_time : 
  ∀ (car_time : ℝ) (extra_time : ℝ), car_time = 4.5 → extra_time = 2.0 → combined_time car_time extra_time = 11 :=
by
  intros car_time extra_time hcar hextra
  sorry

end train_and_car_combined_time_l510_510284


namespace rectangle_width_l510_510167

-- Define the conditions
def length := 6
def area_triangle := 60
def area_ratio := 2/5

-- The theorem: proving that the width of the rectangle is 4 cm
theorem rectangle_width (w : ℝ) (A_triangle : ℝ) (len : ℝ) 
  (ratio : ℝ) (h1 : A_triangle = 60) (h2 : len = 6) (h3 : ratio = 2 / 5) 
  (h4 : (len * w) / A_triangle = ratio) : 
  w = 4 := 
by 
  sorry

end rectangle_width_l510_510167


namespace max_distance_is_sqrt_193_l510_510180

noncomputable def max_distance_between_circle_centers : ℝ :=
  let w := 20
  let h := 15
  let d := 8
  in Real.sqrt ((w - d) ^ 2 + (h - d) ^ 2)

theorem max_distance_is_sqrt_193 :
  max_distance_between_circle_centers = Real.sqrt 193 :=
by {
  let w := 20
  let h := 15
  let d := 8
  let x := w - d
  let y := h - d
  have hx : x = 12, by sorry
  have hy : y = 7, by sorry
  rw [hx, hy],
  simp,
  sorry
}

end max_distance_is_sqrt_193_l510_510180


namespace circle_diameter_l510_510200

theorem circle_diameter (A : ℝ) (h : A = 64 * Real.pi) : (2 * Real.sqrt(64)) = 16 :=
by
  sorry

end circle_diameter_l510_510200


namespace sample_size_is_correct_l510_510646

-- Define the conditions
def total_students : ℕ := 40 * 50
def students_selected : ℕ := 150

-- Theorem: The sample size is 150 given that 150 students are selected
theorem sample_size_is_correct : students_selected = 150 := by
  sorry  -- Proof to be completed

end sample_size_is_correct_l510_510646


namespace average_salary_decrease_l510_510822

theorem average_salary_decrease 
    (avg_wage_illiterate_initial : ℝ)
    (avg_wage_illiterate_new : ℝ)
    (num_illiterate : ℕ)
    (num_literate : ℕ)
    (num_total : ℕ)
    (total_decrease : ℝ) :
    avg_wage_illiterate_initial = 25 →
    avg_wage_illiterate_new = 10 →
    num_illiterate = 20 →
    num_literate = 10 →
    num_total = num_illiterate + num_literate →
    total_decrease = (avg_wage_illiterate_initial - avg_wage_illiterate_new) * num_illiterate →
    total_decrease / num_total = 10 :=
by
  intros avg_wage_illiterate_initial_eq avg_wage_illiterate_new_eq num_illiterate_eq num_literate_eq num_total_eq total_decrease_eq
  sorry

end average_salary_decrease_l510_510822


namespace triangle_ABC_is_right_angled_l510_510832

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2)

noncomputable def isRightAngledTriangle (A B C : Point3D) : Prop :=
  let AB := distance A B
  let BC := distance B C
  let CA := distance C A
  (AB^2 + BC^2 = CA^2) ∨ (AB^2 + CA^2 = BC^2) ∨ (CA^2 + BC^2 = AB^2)

theorem triangle_ABC_is_right_angled (A B C : Point3D)
  (hA : A = ⟨-1, 2, 2⟩)
  (hB : B = ⟨2, -2, 3⟩)
  (hC : C = ⟨4, -1, 1⟩) : isRightAngledTriangle A B C := 
by 
  rw [hA, hB, hC]
  sorry

end triangle_ABC_is_right_angled_l510_510832


namespace sum_of_interior_angles_at_vertex_A_l510_510076

-- Definitions of the interior angles for a square and a regular octagon.
def square_interior_angle : ℝ := 90
def octagon_interior_angle : ℝ := 135

-- Theorem that states the sum of the interior angles at vertex A formed by the square and octagon.
theorem sum_of_interior_angles_at_vertex_A : square_interior_angle + octagon_interior_angle = 225 := by
  sorry

end sum_of_interior_angles_at_vertex_A_l510_510076


namespace nonneg_diff_between_roots_l510_510588

theorem nonneg_diff_between_roots : 
  ∀ x : ℝ, 
  (x^2 + 42 * x + 384 = 0) → 
  (∃ r1 r2 : ℝ, x = r1 ∨ x = r2) → 
  (abs (r1 - r2) = 8) := 
sorry

end nonneg_diff_between_roots_l510_510588


namespace elem_in_set_l510_510031

def A : set ℝ := { x | x ≤ Real.sqrt 13 } -- Define the set A
def a : ℝ := 3 -- Define the element a

theorem elem_in_set : a ∈ A := by
  -- Placeholder for the actual proof, which is not needed according to instructions
  sorry

end elem_in_set_l510_510031


namespace product_xyz_l510_510431

theorem product_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * y = 30 * (4:ℝ)^(1/3)) (h5 : x * z = 45 * (4:ℝ)^(1/3)) (h6 : y * z = 18 * (4:ℝ)^(1/3)) :
  x * y * z = 540 * Real.sqrt 3 :=
sorry

end product_xyz_l510_510431


namespace simplify_expression_l510_510521

theorem simplify_expression (α : ℝ) : 
  (cos (2 * real.pi + α) * tan (real.pi + α)) / cos (real.pi / 2 - α) = 1 :=
by
  sorry

end simplify_expression_l510_510521


namespace coin_flip_probability_find_p_plus_q_l510_510966

theorem coin_flip_probability (h : ℚ) (H : 0 < h ∧ h < 1) :
  (choose 4 1) * h * (1 - h)^3 = (choose 4 2) * h^2 * (1 - h)^2 →
  (choose 4 2) * (2/5)^2 * (3/5)^2 = 216 / 625 :=
by
  sorry

lemma p_plus_q :
  216 + 625 = 841 :=
by
  exact rfl

theorem find_p_plus_q (h : ℚ) (H : 0 < h ∧ h < 1) :
  (choose 4 1) * h * (1 - h)^3 = (choose 4 2) * h^2 * (1 - h)^2 →
  (216 + 625 = 841) :=
by
  intro H1
  have H2 : (choose 4 2) * (2 / 5) ^ 2 * (3 / 5) ^ 2 = 216 / 625 := coin_flip_probability h H H1
  exact p_plus_q

end coin_flip_probability_find_p_plus_q_l510_510966


namespace combined_time_l510_510281

theorem combined_time {t_car t_train t_combined : ℝ} 
  (h1: t_car = 4.5) 
  (h2: t_train = t_car + 2) 
  (h3: t_combined = t_car + t_train) : 
  t_combined = 11 := by
  sorry

end combined_time_l510_510281


namespace equal_radii_l510_510681

-- Definitions based on the given conditions
def radiusA : ℝ := 5
def diameterB : ℝ := 10
def circumferenceC : ℝ := 10 * Real.pi

-- Mathematical equivalence to prove: All circles have equal radii
theorem equal_radii (r_A r_B r_C : ℝ) 
  (hA : r_A = radiusA) 
  (hB : r_B = diameterB / 2) 
  (hC : r_C = circumferenceC / (2 * Real.pi)) : 
  r_A = r_B ∧ r_B = r_C :=
by
  sorry

end equal_radii_l510_510681


namespace perpendicular_TS_T_l510_510986

-- Define the geometric setup
variables {A B C R Q P E F F' E' S T S' T' : Type}
variables [knaching[R : IsMidpoint A B]] [IsMidpoint Q A C] [IsMidpoint P B C]
variables [pt_intersection_AP_RQ : Intersects AP RQ E]
variables [circumcircle_ABC : IsOnCircumcircle B C]
variables [intersection_AF'_circumcircle : Intersects AP (CircumcircleTriangleIntersect ABC) F]

-- Define perpendicular conditions
variables (ES_perpendicular_PQ : Perpendicular ES PQ)
variables (ET_perpendicular_RP : Perpendicular ET RP)

-Define other points and intersections
variables [diameter_FF' : IsDiameter F F']
variables [intersection_AF'_BC : Intersects (LineSegment AF') BC E']
variables [perpendicular_E'S'_AB : Perpendicular (LineSegment E' S') (LineSegment AB)]
variables [perpendicular_E'T'_AC : Perpendicular (LineSegment E' T') (LineSegment AC)]

-- Main proof statement
theorem perpendicular_TS_T'S' :
  Perpendicular (LineSegment T S) (LineSegment T' S') :=
sorry

end perpendicular_TS_T_l510_510986


namespace problem_part1_problem_part2_problem_part3_l510_510400

variables {a n m : ℝ}

def f (x : ℝ) := log a ((1 - m * x) / (x - 1))

theorem problem_part1 (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∀ x, f x + f (-x) = 0) : 
  m = -1 := sorry

theorem problem_part2 (h1 : a > 0) (h2 : a ≠ 1) (h4 : m = -1) : 
  (∀ x1 x2, 1 < x1 ∧ x1 > x2 ∧ x2 > 1 → 
  (if a > 1 then f x1 < f x2 else if 0 < a ∧ a < 1 then f x1 > f x2 else false)) := sorry

theorem problem_part3 (h1 : a > 0) (h2 : a ≠ 1) (h4 : m = -1) (h5 : (1 < n ∧ n < a - 2) ∨ (n < a - 2 ∧ a - 2 < -1)) : 
  (n = 1 ∧ a = 2 + sqrt 3) := sorry

end problem_part1_problem_part2_problem_part3_l510_510400


namespace sin_cos_identity_l510_510000

variables {α β : ℝ}

-- Definitions based on given conditions
def tan_root_1_3_neg3 (x : ℝ) : Prop := x^2 - 3 * x - 3 = 0
def tan_alpha := (∃ α β : ℝ, tan_root_1_3_neg3 (tan α) ∧ tan_root_1_3_neg3 (tan β))

-- Theorem to prove: final identity based on sin and cos of sum of angles
theorem sin_cos_identity (α β : ℝ) (hαβ: tan_root_1_3_neg3 (tan α) ∧ tan_root_1_3_neg3 (tan β)) :
  (sin (α + β))^2 - 3 * (sin (α + β)) * (cos (α + β)) - 3 * (cos (α + β))^2 = -3 :=
sorry

end sin_cos_identity_l510_510000


namespace integral_evaluation_l510_510678

theorem integral_evaluation : ∫ x in (0 : ℝ)..(π / 2), (1 - 5 * x^2) * sin x = 11 - 5 * π := 
by
  sorry

end integral_evaluation_l510_510678


namespace combined_salary_l510_510908

theorem combined_salary (S_B : ℝ) (S_A : ℝ) (h1 : S_B = 8000) (h2 : 0.20 * S_A = 0.15 * S_B) : 
S_A + S_B = 14000 :=
by {
  sorry
}

end combined_salary_l510_510908


namespace find_x4_plus_y4_l510_510432

theorem find_x4_plus_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^4 + y^4 = 135.5 :=
by
  sorry

end find_x4_plus_y4_l510_510432


namespace greatest_gcd_of_ten_numbers_l510_510174

theorem greatest_gcd_of_ten_numbers (a : ℕ → ℕ) (h₀ : (∑ i in finset.range 10, a i) = 1001) :
∃ d, d = 91 ∧ (∀ n, n ∈ finset.range 10 → d ∣ a n) ∧ (∀ d', (∀ n, n ∈ finset.range 10 → d' ∣ a n) → d' ≤ d) :=
sorry

end greatest_gcd_of_ten_numbers_l510_510174


namespace tyler_meal_choices_l510_510572

theorem tyler_meal_choices : 
  let meat_choices := 3
  let vegetable_choices := 5
  let dessert_choices := 4
  let vegetable_combinations := Nat.choose vegetable_choices 2
  in meat_choices * vegetable_combinations * dessert_choices = 120 := by
  sorry

end tyler_meal_choices_l510_510572


namespace problem_statement_l510_510423

noncomputable def poly := (x^2 + 1) * (x - 3)^11

theorem problem_statement (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ : ℚ) (h : (x^2 + 1) * (x - 3)^11 = a + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3 + a₄ * (x - 2)^4 + a₅ * (x - 2)^5 + a₆ * (x - 2)^6 + a₇ * (x - 2)^7 + a₈ * (x - 2)^8 + a₉ * (x - 2)^9 + a₁₀ * (x - 2)¹⁰ + a₁₁ * (x - 2)¹¹ + a₁₂ * (x - 2)¹² + a₁₃ * (x - 2)¹³) : 
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ + a₁₂ = 5 :=
sorry

end problem_statement_l510_510423


namespace cube_edge_traversal_probabilities_l510_510267

-- Defining the vertices and transition probabilities
def vertices : Type := {A A' B B' C C' D D'}

-- Transition probability for moving along one of the three edges from any vertex
def transition_prob (start : vertices) (end : vertices) : ℚ :=
  if end = B' ∨ end = C' then 1/3
  else if start = A ∨ start = A' ∨ start = B ∨ start = D then 1/3
  else 0

-- Initial conditions
axiom prob_start_A_Stop_B' : ℚ
axiom prob_start_A_Stop_C' : ℚ
axiom prob_start_A_Never_Stop : ℚ

-- Define the probabilities of stopping at B', stopping at C', and never stopping
def prob_stop_at_B' := prob_start_A_Stop_B'
def prob_stop_at_C' := prob_start_A_Stop_C'
def prob_never_stop := prob_start_A_Never_Stop

-- The probabilities must sum to 1
axiom prob_sum_to_1 : prob_stop_at_B' + prob_stop_at_C' + prob_never_stop = 1

-- Stating the Lean 4 assertions to prove the solution
theorem cube_edge_traversal_probabilities : 
  prob_stop_at_B' = 4/7 ∧ prob_stop_at_C' = 3/7 ∧ prob_never_stop = 0 :=
by 
  sorry

end cube_edge_traversal_probabilities_l510_510267


namespace wall_height_l510_510948

/--
John has 10 hours to paint and has 5 hours to spare, so he spends 5 hours painting.
He can paint 1 square meter every 10 minutes.
Each wall is 3 meters wide and there are 5 walls in total.
We need to determine the height of each wall.
-/
theorem wall_height :
  ∀ (H : ℝ), 
  (10 - 5) * 60 * (1 / 10) = 30 →
  5 * H * 3 = 30 →
  H = 2 :=
by
  intros H time_paintable total_area.
  have total_time_painting := (10 - 5) * 60
  rw [time_paintable] at total_time_painting
  simp at total_time_painting
  rw [total_area] at total_time_painting
  linarith
-- sorry

end wall_height_l510_510948


namespace arithmetic_sequence_sum_divisible_by_8_l510_510590

theorem arithmetic_sequence_sum_divisible_by_8 :
  ∃ (S : ℕ), S = ∑ i in finset.range (49), 6 * i ∧ S % 8 = 0 :=
by
  sorry

end arithmetic_sequence_sum_divisible_by_8_l510_510590


namespace ellipse_equation_line_intersection_condition_l510_510906

theorem ellipse_equation (a b c : ℝ) (h_c : c = 2) (h_b : b = 2) (h_a : a = 2 * Real.sqrt 2) : 
  ∃ C : set (ℝ × ℝ), ∀ p : ℝ × ℝ, p ∈ C ↔ (p.1^2) / 8 + (p.2^2) / 4 = 1 := 
by 
  sorry

theorem line_intersection_condition (k m : ℝ) : 
  (m > Real.sqrt 2 ∨ m < -Real.sqrt 2) ↔ 
  ∃ (A B : ℝ × ℝ), (A.2 = k * A.1 + m) ∧ (B.2 = k * B.1 + m) ∧
  ((A.1^2 + 2*(k*A.1 + m)^2 = 8) ∧ (B.1^2 + 2*(k*B.1 + m)^2 = 8)) ∧
  ((A.1^2 + A.2^2) + (B.1^2 + B.2^2) = 0) := 
by 
  sorry

end ellipse_equation_line_intersection_condition_l510_510906


namespace compute_expression_l510_510095

def P (x : ℝ) : ℝ := 3 * real.sqrt x
def Q (x : ℝ) : ℝ := x^3

theorem compute_expression : P (Q (P (Q (P (Q 2))))) = 1512 * real.sqrt 3 :=
by 
  sorry

end compute_expression_l510_510095


namespace solve_abs_inequality_l510_510720

theorem solve_abs_inequality :
  { x : ℝ | 3 ≤ |x - 2| ∧ |x - 2| ≤ 6 } = { x : ℝ | -4 ≤ x ∧ x ≤ -1 } ∪ { x : ℝ | 5 ≤ x ∧ x ≤ 8 } :=
sorry

end solve_abs_inequality_l510_510720


namespace find_a5_l510_510374

def sequence_sum (n : ℕ) : ℕ := n^2 + 1

theorem find_a5 : sequence_sum 5 - sequence_sum 4 = 9 := by
  calc
    sequence_sum 5 - sequence_sum 4 = (5^2 + 1) - (4^2 + 1) : by rfl
    ... = 26 - 17 : by rfl
    ... = 9 : by rfl

end find_a5_l510_510374


namespace inverse_function_correct_l510_510399

noncomputable def f (x : ℝ) : ℝ := log (3 * x + 1)

noncomputable def f_inv (y : ℝ) : ℝ := (exp y - 1)^3

theorem inverse_function_correct :
  ∀ y : ℝ, f_inv (f y) = y :=
by
  sorry

end inverse_function_correct_l510_510399


namespace inscribed_circle_radius_l510_510890

-- Define the given sector and its conditions
structure Sector where
  radius : ℝ
  angle : ℝ

-- The sector OAB is a quarter of a circle with a radius of 5 cm
def sector_OAB : Sector := { radius := 5, angle := π / 2 }

-- Define the circle inscribed within the sector
structure Circles where
  inscribed_radius : ℝ

-- The radius of the inscribed circle should be 5 * (sqrt 2 - 1)
theorem inscribed_circle_radius (s : Sector) (c : Circles) (h : s = sector_OAB) : 
  c.inscribed_radius = 5 * (Real.sqrt 2 - 1) :=
sorry

end inscribed_circle_radius_l510_510890


namespace average_episodes_per_year_is_16_l510_510615

-- Define the number of years the TV show has been running
def years : Nat := 14

-- Define the number of seasons and episodes for each category
def seasons_8_15 : Nat := 8
def episodes_per_season_8_15 : Nat := 15
def seasons_4_20 : Nat := 4
def episodes_per_season_4_20 : Nat := 20
def seasons_2_12 : Nat := 2
def episodes_per_season_2_12 : Nat := 12

-- Define the total number of episodes
def total_episodes : Nat :=
  (seasons_8_15 * episodes_per_season_8_15) + 
  (seasons_4_20 * episodes_per_season_4_20) + 
  (seasons_2_12 * episodes_per_season_2_12)

-- Define the average number of episodes per year
def average_episodes_per_year : Nat :=
  total_episodes / years

-- State the theorem to prove the average number of episodes per year is 16
theorem average_episodes_per_year_is_16 : average_episodes_per_year = 16 :=
by
  sorry

end average_episodes_per_year_is_16_l510_510615


namespace parabola_intersections_circle_radius_squared_l510_510921

theorem parabola_intersections_circle_radius_squared :
  (∀ x y : ℝ, ((y = (x + 2)^2) ∧ (x + 8 = (y - 2)^2)) → ((x + 2)^2 + (y - 2)^2 = 4)) → 
  (∃ r : ℝ, r^2 = 4) :=
by
  intro h
  use 2
  simp
  sorry

end parabola_intersections_circle_radius_squared_l510_510921


namespace number_of_factorizable_polynomials_l510_510947

theorem number_of_factorizable_polynomials : 
  (card {n : ℕ | n ≤ 100 ∧ ∃ a b : ℤ, a + b = 2 ∧ a * b = -n}) = 9 :=
by
  sorry

end number_of_factorizable_polynomials_l510_510947


namespace fixed_point_coordinates_l510_510535

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  2 * a^(x + 1) - 3

theorem fixed_point_coordinates (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = -1 :=
by
  sorry

end fixed_point_coordinates_l510_510535


namespace net_income_after_tax_l510_510194

theorem net_income_after_tax (gross_income : ℝ) (tax_rate : ℝ) : 
  (gross_income = 45000) → (tax_rate = 0.13) → 
  (gross_income - gross_income * tax_rate = 39150) :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end net_income_after_tax_l510_510194


namespace midpoint_PQ_circle_l510_510181

-- Assume we have points, circles, and other necessary geometric constructs
variables (A B P Q : Point) (C1 C2 : Circle)

-- Condition: Two circles intersect at points A and B
def circles_intersect_at_A_B (C1 C2 : Circle) (A B : Point) : Prop :=
  C1.contains A ∧ C1.contains B ∧ C2.contains A ∧ C2.contains B

-- Condition: A secant passing through point A intersects the circles again at points P and Q
def secant_through_A (C1 C2 : Circle) (A P Q : Point) : Prop :=
  Line(A, P).intersects_circle C1 = P ∧ Line(A, Q).intersects_circle C2 = Q

-- The goal to prove
theorem midpoint_PQ_circle (C1 C2 : Circle) (A B P Q : Point)
  (hIntersect : circles_intersect_at_A_B C1 C2 A B)
  (hSecant: secant_through_A C1 C2 A P Q) :
  ∃ (C : Circle), ∀ (P Q : Point), midpoint P Q ∈ C :=
sorry

end midpoint_PQ_circle_l510_510181


namespace new_supervisor_salary_l510_510820

-- Define all the given conditions as Lean definitions
def average_salary_before : ℕ := 5300
def num_workers : ℕ := 15
def num_supervisors : ℕ := 3
def supervisor_A_salary : ℕ := 6200
def supervisor_B_salary : ℕ := 7200
def supervisor_C_salary : ℕ := 8200
def average_salary_after : ℕ := 5100
def total_people : ℕ := num_workers + num_supervisors

-- Define the theorem that proves the salary of the new supervisor
theorem new_supervisor_salary :
  let total_salary_before := average_salary_before * total_people,
      total_salary_sup_AB := supervisor_A_salary + supervisor_B_salary,
      total_salary_workers := total_salary_before - total_salary_sup_AB,
      total_salary_after := average_salary_after * total_people,
      salary_difference := total_salary_before - total_salary_after,
      new_supervisor_salary := supervisor_C_salary - salary_difference
  in new_supervisor_salary = 4600 := by
  sorry  

end new_supervisor_salary_l510_510820


namespace willie_gave_emily_7_stickers_l510_510594

theorem willie_gave_emily_7_stickers (initial_stickers : ℕ) (final_stickers : ℕ) (given_stickers : ℕ) 
  (h1 : initial_stickers = 36) (h2 : final_stickers = 29) (h3 : given_stickers = initial_stickers - final_stickers) : 
  given_stickers = 7 :=
by
  rw [h1, h2] at h3 -- Replace initial_stickers with 36 and final_stickers with 29 in h3
  exact h3  -- given_stickers = 36 - 29 which is equal to 7.


end willie_gave_emily_7_stickers_l510_510594


namespace softball_players_l510_510054

theorem softball_players (cricket hockey football total : ℕ) (h1 : cricket = 12) (h2 : hockey = 17) (h3 : football = 11) (h4 : total = 50) : 
  total - (cricket + hockey + football) = 10 :=
by
  sorry

end softball_players_l510_510054


namespace marie_days_to_pay_cash_register_l510_510506

def daily_revenue_bread (loaves: Nat) (price_per_loaf: Nat) : Nat := loaves * price_per_loaf
def daily_revenue_cakes (cakes: Nat) (price_per_cake: Nat) : Nat := cakes * price_per_cake
def total_daily_revenue (loaves: Nat) (price_per_loaf: Nat) (cakes: Nat) (price_per_cake: Nat) : Nat :=
  daily_revenue_bread loaves price_per_loaf + daily_revenue_cakes cakes price_per_cake

def daily_expenses (rent: Nat) (electricity: Nat) : Nat := rent + electricity

def daily_profit (loaves: Nat) (price_per_loaf: Nat) (cakes: Nat) (price_per_cake: Nat) (rent: Nat) (electricity: Nat) : Nat :=
  total_daily_revenue loaves price_per_loaf cakes price_per_cake - daily_expenses rent electricity

def days_to_pay_cash_register (register_cost: Nat) (profit: Nat) : Nat :=
  register_cost / profit

theorem marie_days_to_pay_cash_register :
  days_to_pay_cash_register 1040 (daily_profit 40 2 6 12 20 2) = 8 :=
by
  calc
    days_to_pay_cash_register 1040 (daily_profit 40 2 6 12 20 2)
        = 1040 / daily_profit 40 2 6 12 20 2 : by rfl
    ... = 1040 / 130 : by rfl
    ... = 8 : by rfl

end marie_days_to_pay_cash_register_l510_510506


namespace externally_tangent_chord_length_2sqrt7_l510_510769

-- Definition of circle C
def circle_C (x y : ℝ) (m : ℝ) : Prop :=
  x^2 + y^2 - 6 * x - 8 * y + m = 0

-- Definition of second circle
def circle_1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Definition of line
def line (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Problem statement (I)
theorem externally_tangent (m : ℝ) :
  (∃ x y : ℝ, circle_C x y m) ∧  (∃ x y : ℝ, circle_1 x y) ∧ (∃ x : ℝ, ∃ y : ℝ, (circle_C x y m) = (circle_1 x y)) → m = 9 :=
by
  sorry

-- Problem statement (II)
theorem chord_length_2sqrt7 (m : ℝ) :
  (∃ x y : ℝ, circle_C x y m) ∧ (∃ x y : ℝ, line x y) ∧ (∃ l : ℝ, l = (2*sqrt(7))) → m = 10 :=
by
  sorry

end externally_tangent_chord_length_2sqrt7_l510_510769


namespace smallest_repeating_block_of_4_div_7_has_6_digits_l510_510037

theorem smallest_repeating_block_of_4_div_7_has_6_digits : 
  ∃ n : ℕ, n = 6 ∧ (∃ m : ℕ, (m ≥ 0 ∧ (4 / 7 : ℚ).decimalExpansion m = "571428")) := 
sorry

end smallest_repeating_block_of_4_div_7_has_6_digits_l510_510037


namespace sum_of_root_and_square_of_other_root_eq_2007_l510_510795

/-- If α and β are the two real roots of the equation x^2 - x - 2006 = 0,
    then the value of α + β^2 is 2007. --/
theorem sum_of_root_and_square_of_other_root_eq_2007
  (α β : ℝ)
  (hα : α^2 - α - 2006 = 0)
  (hβ : β^2 - β - 2006 = 0) :
  α + β^2 = 2007 := sorry

end sum_of_root_and_square_of_other_root_eq_2007_l510_510795


namespace contrary_implies_mutually_exclusive_contrary_sufficient_but_not_necessary_l510_510571

variable {A B : Prop}

def contrary (A : Prop) : Prop := A ∧ ¬A
def mutually_exclusive (A B : Prop) : Prop := ¬(A ∧ B)

theorem contrary_implies_mutually_exclusive (A : Prop) : contrary A → mutually_exclusive A (¬A) :=
by sorry

theorem contrary_sufficient_but_not_necessary (A B : Prop) :
  (∃ (A : Prop), contrary A) → mutually_exclusive A B →
  (∃ (A : Prop), contrary A ∧ mutually_exclusive A B) :=
by sorry

end contrary_implies_mutually_exclusive_contrary_sufficient_but_not_necessary_l510_510571


namespace triangle_sides_l510_510456

theorem triangle_sides (m_c : ℝ) (γ : ℝ) (f_c : ℝ) (a b c : ℝ) :
  m_c = 4 ∧ γ = 100 ∧ f_c = 5 →
  (a ≈ 4.107) ∧ (b ≈ 73.26) ∧ (c ≈ 74.08) :=
by sorry

end triangle_sides_l510_510456


namespace dealer_incurred_loss_l510_510257

-- Define the conditions
def cost_price_A := 15 * 25
def cost_price_B := 20 * 40
def cost_price_C := 30 * 55
def cost_price_D := 10 * 80

def selling_price_A := 12 * 38
def selling_price_B := 18 * 50
def selling_price_C := 22 * 65
def selling_price_D := 8 * 100

def total_cost_price := cost_price_A + cost_price_B + cost_price_C + cost_price_D
def total_selling_price := selling_price_A + selling_price_B + selling_price_C + selling_price_D

def loss := total_cost_price - total_selling_price

-- Calculate the loss percentage
def loss_percentage := (loss.toFloat / total_cost_price.toFloat) * 100

-- Lean statement for proof
theorem dealer_incurred_loss :
  |loss_percentage - 1.075| < 0.001 :=
by
  sorry

end dealer_incurred_loss_l510_510257


namespace root_multiplicity_two_l510_510728

noncomputable def poly (A B n : ℕ) (x : ℝ) : ℝ := A * x^(n+1) + B * x^n + 1
noncomputable def deriv_poly (A B n : ℕ) (x : ℝ) : ℝ := (n+1) * A * x^n + n * B * x^(n-1)

theorem root_multiplicity_two 
  {A B n : ℕ}
  (h₁ : poly A B n 1 = 0)
  (h₂ : deriv_poly A B n 1 = 0) : 
  A = n ∧ B = -(n+1) :=
  sorry

end root_multiplicity_two_l510_510728


namespace rhombus_in_convex_symmetric_polygon_l510_510509

theorem rhombus_in_convex_symmetric_polygon (P : Set Point) (O : Point) (h₁ : Convex P) (h₂ : CentrallySymmetric P O) :
  ∃(R : Set Point), Rhombus R ∧ (Area R = (1/2) * Area P) ∧ (R ⊆ P) := 
  sorry

end rhombus_in_convex_symmetric_polygon_l510_510509


namespace degree_of_polynomial_l510_510199

noncomputable def polynomial_1 (a b c : ℝ) : polynomial ℝ :=
  polynomial.C a * polynomial.X ^ 7 + polynomial.X ^ 4 + polynomial.C b * polynomial.X + polynomial.C c

noncomputable def polynomial_2 (d e : ℝ) : polynomial ℝ :=
  polynomial.X ^ 3 + polynomial.C d * polynomial.X ^ 2 + polynomial.C e

noncomputable def polynomial_3 (f : ℝ) : polynomial ℝ :=
  polynomial.X + polynomial.C f

theorem degree_of_polynomial (a b c d e f : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hd : d ≠ 0) (he : e ≠ 0) (hf : f ≠ 0) :
  polynomial.degree (polynomial_1 a b c * polynomial_2 d e * polynomial_3 f) = polynomial.degree (polynomial.C a * polynomial.X ^ 11) := by
  sorry

end degree_of_polynomial_l510_510199


namespace sum_of_integer_solutions_l510_510523

theorem sum_of_integer_solutions :
  (∀ x : ℝ, (2 * (x + 3) > 4) → ((x - 1) / 3 ≥ x / 2 - 1)) →
  (Σ (x : ℤ) in finset.filter (λ x, (-1 : ℝ) < x ∧ (x : ℝ) ≤ 4) (finset.Icc (-1 : ℤ) 4), x) = 10 :=
by
  intro h
  sorry

end sum_of_integer_solutions_l510_510523


namespace k_value_l510_510367

open Real EuclideanGeometry

variable (a b : EuclideanSpace ℝ (Fin 3))
variable (k : ℝ)
variable (c d : EuclideanSpace ℝ (Fin 3))

noncomputable def magnitude (v : EuclideanSpace ℝ (Fin 3)) := sqrt (v ⬝ v)

axiom a_norm : magnitude a = 1
axiom b_norm : magnitude b = 2
axiom angle_ab : ∀ (u v : EuclideanSpace ℝ (Fin 3)), u ≠ 0 → v ≠ 0 → cos_angle u v = (1/2)
axiom c_def : c = 2 • a + 3 • b
axiom d_def : d = k • a - b
axiom c_perp_d : c ⬝ d = 0

theorem k_value : k = 14 / 5 :=
by
sorry

end k_value_l510_510367


namespace maxwell_distance_l510_510121

-- Define the given conditions
def distance_between_homes : ℝ := 65
def maxwell_speed : ℝ := 2
def brad_speed : ℝ := 3

-- The statement we need to prove
theorem maxwell_distance :
  ∃ (x t : ℝ), 
    x = maxwell_speed * t ∧
    distance_between_homes - x = brad_speed * t ∧
    x = 26 := by sorry

end maxwell_distance_l510_510121


namespace area_CFDE_eq_l510_510358

noncomputable def right_triangle_area (a b : ℝ) : ℝ :=
  a * b / 2

theorem area_CFDE_eq :
  ∀ (a b : ℝ), ∃ (CD CE CF : ℝ),
  let AB := Real.sqrt (a^2 + b^2) in
  let CD := a * b / AB in
  let CF := CD^2 / a in
  let CE := CD^2 / b in
  (CF * CE = a^3 * b^3 / (a^2 + b^2)^2) :=
by
  sorry

end area_CFDE_eq_l510_510358


namespace limit_central_difference_l510_510002

variable {α : Type*} [RealSpace α]

-- Definitions and assumptions derived from problem conditions
def differentiable_at (f : α → α) (x : α) : Prop := ∃ f' : α, limit (λ h, (f (x+h) - f x) / h) 0 f'

variable (f : α → α) (x : α)
variable (h : α)

-- The main statement to prove
theorem limit_central_difference (hx : differentiable_at f x) :
  limit (λ h, (f (x + h) - f (x - h)) / (2 * h)) 0 (f' x) := 
sorry

end limit_central_difference_l510_510002


namespace g_neither_even_nor_odd_l510_510079

def g (x : ℝ) : ℝ := 3 / (2 * x^3 - 5)

theorem g_neither_even_nor_odd : ¬(∀ x : ℝ, g (-x) = g x) ∧ ¬(∀ x : ℝ, g (-x) = -g x) := by
  sorry

end g_neither_even_nor_odd_l510_510079


namespace determine_a_if_slope_angle_is_45_degrees_l510_510445

-- Define the condition that the slope angle of the given line is 45°
def is_slope_angle_45_degrees (a : ℝ) : Prop :=
  let m := -a / (2 * a - 3)
  m = 1

-- State the theorem we need to prove
theorem determine_a_if_slope_angle_is_45_degrees (a : ℝ) :
  is_slope_angle_45_degrees a → a = 1 :=
by
  intro h
  sorry

end determine_a_if_slope_angle_is_45_degrees_l510_510445


namespace unique_arithmetic_sequence_l510_510894

def is_arithmetic_sequence (seq : List ℕ) : Prop :=
  ∀ i, i < seq.length - 1 → seq[i+1] - seq[i] = seq[1] - seq[0]

theorem unique_arithmetic_sequence :
  ∃ (A B C D E F : ℕ),
    is_arithmetic_sequence [A, B, C, D, E, F] ∧
    A = 2 ∧
    A + B + C + D + E + F = 42 ∧
    A + C + E = 2 * (B + D + F) ∧
    B + F = 2 * A ∧
    D = 2 :=
begin
  sorry
end

end unique_arithmetic_sequence_l510_510894


namespace value_range_of_sum_difference_l510_510480

theorem value_range_of_sum_difference (a b c : ℝ) (h₁ : a < b)
  (h₂ : a + b = b / a) (h₃ : a * b = c / a) (h₄ : a + b > c)
  (h₅ : a + c > b) (h₆ : b + c > a) : 
  ∃ x y, x = 7 / 8 ∧ y = Real.sqrt 5 - 1 ∧ x < a + b - c ∧ a + b - c < y := sorry

end value_range_of_sum_difference_l510_510480


namespace combine_expr_l510_510661

variable (a b : ℝ)

theorem combine_expr : 3 * (2 * a - 3 * b) - 6 * (a - b) = -3 * b := by
  sorry

end combine_expr_l510_510661


namespace nonneg_diff_roots_l510_510582

theorem nonneg_diff_roots : 
  ∀ (a b c : ℤ), a = 1 → b = 42 → c = 384 → 
  let Δ = b * b - 4 * a * c in
  Δ ≥ 0 → 
  let r1 := (-b + Δ.sqrt) / (2 * a) in
  let r2 := (-b - Δ.sqrt) / (2 * a) in
  abs (r1 - r2) = 8 :=
by
  sorry

end nonneg_diff_roots_l510_510582


namespace coffee_left_l510_510877

theorem coffee_left (coffee_start : ℚ) 
   (initial_drink_ratio : ℚ) (office_drink_ratio : ℚ) 
   (espresso_addition : ℚ) (lunch_drink_ratio : ℚ) 
   (cold_drink_ratio : ℚ) :
   (coffee_start = 12) →
   (initial_drink_ratio = 1/4) →
   (office_drink_ratio = 1/3) →
   (espresso_addition = 2.5) →
   (lunch_drink_ratio = 0.75) →
   (cold_drink_ratio = 0.6) →
   let after_initial : ℚ := coffee_start * (1 - initial_drink_ratio) in
   let after_office : ℚ := after_initial * (1 - office_drink_ratio) in
   let after_espresso : ℚ := after_office + espresso_addition in
   let after_lunch : ℚ := after_espresso * (1 - lunch_drink_ratio) in
   let after_cold : ℚ := after_lunch * (1 - cold_drink_ratio) in
   after_cold = 0.85 :=
by sorry

end coffee_left_l510_510877


namespace car_time_interval_l510_510560

-- Define the conditions
def road_length := 3 -- in miles
def total_time := 10 -- in hours
def number_of_cars := 30

-- Define the conversion factor and the problem to prove
def hours_to_minutes (hours: ℕ) : ℕ := hours * 60
def time_interval_per_car (total_time_minutes: ℕ) (number_of_cars: ℕ) : ℕ := total_time_minutes / number_of_cars

-- The Lean 4 statement for the proof problem
theorem car_time_interval :
  time_interval_per_car (hours_to_minutes total_time) number_of_cars = 20 :=
by
  sorry

end car_time_interval_l510_510560


namespace min_sum_at_11_or_12_l510_510778

def seq_term (n : ℕ) : ℤ := n^2 - 11*n - 12

def partial_sum (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), seq_term i

theorem min_sum_at_11_or_12 : min (partial_sum 11) (partial_sum 12) ≤ partial_sum n := sorry

end min_sum_at_11_or_12_l510_510778


namespace trader_sold_90_pens_l510_510652

theorem trader_sold_90_pens (C N : ℝ) (gain_percent : ℝ) (H1 : gain_percent = 33.33333333333333) (H2 : 30 * C = (gain_percent / 100) * N * C) :
  N = 90 :=
by
  sorry

end trader_sold_90_pens_l510_510652


namespace two_categorical_variables_l510_510462

-- Definitions based on the conditions
def smoking (x : String) : Prop := x = "Smoking" ∨ x = "Not smoking"
def sick (y : String) : Prop := y = "Sick" ∨ y = "Not sick"

def category1 (z : String) : Prop := z = "Whether smoking"
def category2 (w : String) : Prop := w = "Whether sick"

-- The main proof statement
theorem two_categorical_variables : 
  (category1 "Whether smoking" ∧ smoking "Smoking" ∧ smoking "Not smoking") ∧
  (category2 "Whether sick" ∧ sick "Sick" ∧ sick "Not sick") →
  "Whether smoking, Whether sick" = "Whether smoking, Whether sick" :=
by
  sorry

end two_categorical_variables_l510_510462


namespace y_coordinate_of_A_l510_510008

theorem y_coordinate_of_A (a : ℝ) (y : ℝ) (h1 : y = a * 1) (h2 : y = (4 - a) / 1) : y = 2 :=
by
  sorry

end y_coordinate_of_A_l510_510008


namespace domain_of_log_l510_510537

noncomputable def domain_is_correct (x : ℝ) : Prop :=
  ∀ x : ℝ, (∃ y : ℝ, y = log 10 (x - 2)) ↔ x > 2

theorem domain_of_log (x : ℝ) : domain_is_correct x :=
by
  sorry

end domain_of_log_l510_510537


namespace arithmetic_mean_of_first_n_odd_positive_integers_l510_510198

theorem arithmetic_mean_of_first_n_odd_positive_integers (n : ℕ) :
  let Sn := (Finset.range n).sum (λ i, 2 * i + 1)
  let A := Sn / n
  n ≠ 0 → A = n :=
by
  -- define the sum
  let Sn := (Finset.range n).sum (λ i, 2 * i + 1)
  -- calculate the arithmetic mean
  let A := Sn / n
  -- prove the final statement
  intro hnz
  sorry

end arithmetic_mean_of_first_n_odd_positive_integers_l510_510198


namespace profit_june_correct_l510_510534

-- Define conditions
def profit_in_May : ℝ := 20000
def profit_in_July : ℝ := 28800

-- Define the monthly growth rate variable
variable (x : ℝ)

-- The growth factor per month
def growth_factor : ℝ := 1 + x

-- Given condition translated to an equation
def profit_relation (x : ℝ) : Prop :=
  profit_in_May * (growth_factor x) * (growth_factor x) = profit_in_July

-- The profit in June should be computed
def profit_in_June (x : ℝ) : ℝ :=
  profit_in_May * (growth_factor x)

-- The target profit in June we want to prove
def target_profit_in_June := 24000

-- Statement to prove
theorem profit_june_correct (h : profit_relation x) : profit_in_June x = target_profit_in_June :=
  sorry  -- proof to be completed

end profit_june_correct_l510_510534


namespace smallest_x_satisfies_sqrt3x_eq_5x_l510_510963

theorem smallest_x_satisfies_sqrt3x_eq_5x :
  ∃ x : ℝ, (sqrt (3 * x) = 5 * x) ∧ (∀ y : ℝ, sqrt (3 * y) = 5 * y → x ≤ y) ∧ (x = 0) :=
sorry

end smallest_x_satisfies_sqrt3x_eq_5x_l510_510963


namespace total_height_difference_eq_638_l510_510297

variables (Anne Bella Cathy Daisy Ellie : ℝ)

-- Conditions
def Anne_height : Prop := Anne = 80
def Cathy_height : Prop := Cathy = Anne / 2
def Bella_height : Prop := Bella = 3 * Anne
def Daisy_height : Prop := Daisy = (Cathy + Anne) / 2
def Ellie_height : Prop := Ellie = Real.sqrt (Bella * Cathy)
def total_height_difference : ℝ := abs (Bella - Cathy) +
                                    abs (Bella - Daisy) +
                                    abs (Bella - Ellie) +
                                    abs (Daisy - Cathy) +
                                    abs (Ellie - Cathy) +
                                    abs (Ellie - Daisy)

-- Goal
theorem total_height_difference_eq_638
  (hA : Anne_height Anne)
  (hC : Cathy_height Anne Cathy)
  (hB : Bella_height Anne Bella)
  (hD : Daisy_height Anne Cathy Daisy)
  (hE : Ellie_height Bella Cathy Ellie) :
  total_height_difference Anne Bella Cathy Daisy Ellie = 638 := by
  sorry

end total_height_difference_eq_638_l510_510297


namespace initial_distance_between_trucks_is_940_km_l510_510182

variable (t : ℕ) -- time in hours that Driver B has driven when they meet
variable (D_A D_B : ℕ) -- distances driven by Driver A and Driver B when they meet

axiom (h1 : D_A = D_B + 140)
axiom (h2 : D_A = 90 * (t + 1))
axiom (h3 : D_B = 80 * t)

theorem initial_distance_between_trucks_is_940_km :
  (D_A + D_B) = 940 := sorry

end initial_distance_between_trucks_is_940_km_l510_510182


namespace max_subset_count_l510_510855

-- Define the problem conditions in Lean 4
def is_valid_subset (T : Finset ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬ (a + b) % 5 = 0

theorem max_subset_count :
  ∃ (T : Finset ℕ), (is_valid_subset T) ∧ T.card = 18 := by
  sorry

end max_subset_count_l510_510855


namespace x_1998_l510_510270

noncomputable theory
open_locale classical

def sequence (a b : ℝ) (n : ℕ) : ℝ :=
  nat.rec_on n a (λ n xₙ, nat.cases_on n b (λ n xₙ₊₁, (1 + xₙ₊₁) / xₙ))

lemma sequence_period_5 (a b : ℝ) : 
  ∀ n, sequence a b (n + 5) = sequence a b n :=
sorry

theorem x_1998 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  sequence a b 1998 = (a + b + 1) / (a * b) :=
sorry

end x_1998_l510_510270


namespace prob_one_exceeds_90_l510_510061

open Probability

-- Definitions of problem conditions
def normal_distribution (mean variance : ℝ) : ℝ → ℝ :=
λ x, 1 / (sqrt (2 * π * variance)) * exp (-((x - mean)^2) / (2 * variance))

def P_between (X : ℝ → ℝ) (a b : ℝ) : ℝ :=
stintegral ℝ (indicator (set.Icc a b) X)

def P_exceeds (X : ℝ → ℝ) (threshold : ℝ) : ℝ :=
1 - P_between X (threshold - real.pi) threshold

-- Problem conditions
noncomputable def X := normal_distribution 80 (sigma^2)
axiom P_X_between : P_between X 70 90 = 1 / 3
axiom number_of_students : ℕ := 3

-- Theorem statement
theorem prob_one_exceeds_90 :
  let P_X_exceeds_90 := P_exceeds X 90 in
  P_X_exceeds_90 = 1 / 3 →
  let P_event_A := (number_of_students.choose 1 * (2 / 3)^2 * (1 / 3)) in
  P_event_A = 4 / 9 :=
sorry

end prob_one_exceeds_90_l510_510061


namespace digit_1983_is_7_l510_510493

noncomputable def x : ℝ :=
Real.zero + ∑ i in (Finset.range 999 + 1), i / (10 ^ (i.to_string.length * (i)))

theorem digit_1983_is_7 :
  ∃ n : ℕ, (x - Real.floor x) * 10^1983 ≈ 7 := 
sorry

end digit_1983_is_7_l510_510493


namespace sin_inequality_solution_l510_510149

theorem sin_inequality_solution (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (sin x * sin (2 * x) < sin (3 * x) * sin (4 * x)) ↔ (0 < x ∧ x < π / 5 ∨ 2 * π / 5 < x ∧ x < π / 2) := 
sorry

end sin_inequality_solution_l510_510149


namespace S_4k_value_l510_510500

variable {S : ℕ → ℕ}
variable {k : ℕ}

-- Given conditions
def S_k_condition : Prop := S k = 2
def S_3k_condition : Prop := S (3 * k) = 18

-- Math proof problem translated to Lean 4
theorem S_4k_value (S_k : S_k_condition) (S_3k : S_3k_condition) : S (4 * k) = 32 := by
  sorry

end S_4k_value_l510_510500


namespace units_digit_of_power_ends_in_nine_l510_510209

theorem units_digit_of_power_ends_in_nine (n : ℕ) (h : (3^n) % 10 = 9) : n % 4 = 2 :=
sorry

end units_digit_of_power_ends_in_nine_l510_510209


namespace range_of_a_l510_510383

def p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a < 0 ∨ (1/4 < a ∧ a < 4) := 
sorry

end range_of_a_l510_510383


namespace asparagus_spears_needed_is_1200_l510_510308

noncomputable def total_asparagus_spears_needed 
    (bridgettes_guests : ℕ) 
    (alexs_fraction : ℚ) 
    (extra_plates : ℕ) 
    (asparagus_per_plate : ℕ) : ℕ :=
  let alexs_guests := (bridgettes_guests * alexs_fraction).toNat
  let total_guests := bridgettes_guests + alexs_guests
  let total_plates := total_guests + extra_plates
  total_plates * asparagus_per_plate

theorem asparagus_spears_needed_is_1200 :
  total_asparagus_spears_needed 84 (2 / 3) 10 8 = 1200 :=
by
  sorry

end asparagus_spears_needed_is_1200_l510_510308


namespace leopards_to_rabbits_ratio_l510_510845

theorem leopards_to_rabbits_ratio :
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let total_animals := 605
  let leopards := total_animals - antelopes - rabbits - hyenas - wild_dogs
  leopards / rabbits = 1 / 2 :=
by
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let total_animals := 605
  let leopards := total_animals - antelopes - rabbits - hyenas - wild_dogs
  sorry

end leopards_to_rabbits_ratio_l510_510845


namespace ben_fraction_of_taxes_l510_510675

theorem ben_fraction_of_taxes 
  (gross_income : ℝ) (car_payment : ℝ) (fraction_spend_on_car : ℝ) (after_tax_income_fraction : ℝ) 
  (h1 : gross_income = 3000) (h2 : car_payment = 400) (h3 : fraction_spend_on_car = 0.2) :
  after_tax_income_fraction = (1 / 3) :=
by
  sorry

end ben_fraction_of_taxes_l510_510675


namespace length_rest_of_body_l510_510123

theorem length_rest_of_body (height legs head arms rest_of_body : ℝ) 
  (hlegs : legs = (1/3) * height)
  (hhead : head = (1/4) * height)
  (harms : arms = (1/5) * height)
  (htotal : height = 180)
  (hr: rest_of_body = height - (legs + head + arms)) : 
  rest_of_body = 39 :=
by
  -- proof is not required
  sorry

end length_rest_of_body_l510_510123


namespace Shelby_drive_time_in_rain_l510_510516

open Real  -- to use real number operations

-- Define the speed of Shelby's scooter in sun and rain in miles per minute
def speed_sun := (40 : ℝ) / 60
def speed_rain := (25 : ℝ) / 60

-- The total distance and time
def total_distance := 20
def total_time := 40

-- Define the problem statement
theorem Shelby_drive_time_in_rain :
  ∃ x : ℝ, speed_sun * (total_time - x) + speed_rain * x = total_distance ∧ x = 27 :=
by
  sorry

end Shelby_drive_time_in_rain_l510_510516


namespace pi_is_the_only_irrational_number_among_0_1_2_pi_sqrt4_l510_510222

theorem pi_is_the_only_irrational_number_among_0_1_2_pi_sqrt4 : 
  (irrational π) ∧ (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1/2 ∧ x ≠ √4) :=
by
  have pi_irrational : irrational π := sorry
  have zero_rational : ∃ p q : ℤ, 0 = p / q := sorry
  have half_rational : ∃ p q : ℤ, 1/2 = p / q := sorry
  have sqrt4_rational : ∃ p q : ℤ, √4 = p / q := sorry
  exact ⟨pi_irrational, by { rintros ⟨p, q, h⟩, dsimp only,
  { simp [h, zero_rational, half_rational, sqrt4_rational] } } ⟩

end pi_is_the_only_irrational_number_among_0_1_2_pi_sqrt4_l510_510222


namespace vanessa_points_l510_510189

theorem vanessa_points (total_points : ℕ) (average_other_players : ℝ) (num_other_players : ℕ) (points_scored_by_team : ℝ) (V : ℝ) :
  total_points = 48 ∧ average_other_players = 3.5 ∧ num_other_players = 6 ∧ 
  points_scored_by_team = (average_other_players * num_other_players) ∧ 
  points_scored_by_team + V = real.of_nat total_points → V = 27 :=
by
  sorry

end vanessa_points_l510_510189


namespace price_decrease_is_50_percent_l510_510549

-- Original price is 50 yuan
def original_price : ℝ := 50

-- Price after 100% increase
def increased_price : ℝ := original_price * (1 + 1)

-- Required percentage decrease to return to original price
def required_percentage_decrease (x : ℝ) : ℝ := increased_price * (1 - x)

theorem price_decrease_is_50_percent : required_percentage_decrease 0.5 = 50 :=
  by 
    sorry

end price_decrease_is_50_percent_l510_510549


namespace intersection_M_N_l510_510852

def set_M : Set ℝ := { x | x * (x - 1) ≤ 0 }
def set_N : Set ℝ := { x | x < 1 }

theorem intersection_M_N : set_M ∩ set_N = { x | 0 ≤ x ∧ x < 1 } := sorry

end intersection_M_N_l510_510852


namespace polynomial_has_factor_of_form_xsq_plus_qx_plus_2_l510_510922

noncomputable def find_c (c q k : ℚ) : Prop :=
  (3*q + k = 0) ∧ (2*k + 3*q = c) ∧ (2*k = 8)

theorem polynomial_has_factor_of_form_xsq_plus_qx_plus_2 : ∃ (c : ℚ), find_c c (-4/3) 4 :=
by {
  use 4,
  unfold find_c,
  split,
  { ring },
  split,
  { ring },
  { norm_num }
}

end polynomial_has_factor_of_form_xsq_plus_qx_plus_2_l510_510922


namespace pure_imaginary_m_eq_one_second_quadrant_range_and_min_value_l510_510735

variables {m : ℝ}
def z (m : ℝ) : ℂ := complex.mk (m-1) (2*m+1)

-- Statement (1)
theorem pure_imaginary_m_eq_one (h : ∃ m, z m = complex.I * (2*m+1)) : m = 1 :=
sorry

-- Statement (2)
theorem second_quadrant_range_and_min_value (h₁ : m-1 < 0) (h₂ : 2*m+1 > 0) :
  (-1/2 < m ∧ m < 1) ∧ (∃ m, m = -1/5 ∧ ∀ m, -1/2 < m ∧ m < 1 → abs (z m) ≥ abs (z (-1/5))) :=
sorry

end pure_imaginary_m_eq_one_second_quadrant_range_and_min_value_l510_510735


namespace angle_difference_is_90_l510_510454

variables (PQ QR RS SP : ℕ) (angle_S angle_P angle_Q angle_R : ℕ) 

axiom PQ_eq_RS : PQ = RS
axiom sqrt3_plus_1_QR_eq_SP : (√3 + 1) * QR = SP
axiom angle_difference : angle_S - angle_P = 30

theorem angle_difference_is_90 :
  angle_Q - angle_R = 90 :=
sorry

end angle_difference_is_90_l510_510454


namespace grace_pumpkin_pies_l510_510034

def pumpkin_pies_pieces_left (baked sold given [Inhabited {total_pies_sliced : Nat}] : Nat) : Nat :=
  let remaining_pies := baked - sold - given in
  let total_pieces :=
    (2 * 6) + 8 in -- Total slices from 2 pies each sliced into 6 pieces plus 1 pie sliced into 8 pieces.
  let pieces_eaten :=
    ((5 * 2 * 6) / 6) + ((3 * 8) / 8) in -- Family ate 5/6 of first kind and 3/8 of second kind
  total_pies_sliced - pieces_eaten

theorem grace_pumpkin_pies :
  pumpkin_pies_pieces_left 8 (3 + 1/2) (1 + 3/4) 
  = 7 := by
  sorry

end grace_pumpkin_pies_l510_510034


namespace exists_inscribed_quadrilateral_l510_510970

variables (A B C D M : Type) [ConvexQuadrilateral A B C D] [PointInQuadrilateral M A B C D]

noncomputable def InscribedQuadrilateral : Prop :=
  ∃ (P : Type), 
    (isInQuadrilateral P A B C D) ∧ 
    (ParallelSides P A B C D M) ∧ 
    (Intersection P = Intersection (P₁ P₂) (CD))

theorem exists_inscribed_quadrilateral : 
  InscribedQuadrilateral A B C D M :=
sorry

end exists_inscribed_quadrilateral_l510_510970


namespace triangle_area_ratio_l510_510301

-- Define the problem conditions
variables (B C E A D : Type) [CompleteLattice B] [CompleteLattice C] [CompleteLattice E] [CompleteLattice A] [CompleteLattice D]
variables (BC BE AC CD : ℝ)

-- Hypotheses based on given conditions
def bc_be := BC = 3 * BE
def ac_cd := AC = 4 * CD

-- The Lean statement to show the area ratio
theorem triangle_area_ratio (BC BE AC CD : ℝ) (bc_be_hyp : bc_be BC BE) (ac_cd_hyp : ac_cd AC CD) :
  (area_of_triangle ABC) = 2 * (area_of_triangle ADE) :=
sorry

end triangle_area_ratio_l510_510301


namespace permutation_nineteenth_position_l510_510943

theorem permutation_nineteenth_position :
  let digits := [1, 3, 6, 9] in
  let permutations := List.permutations digits in
  let ordered_permutations := perms.sort (fun a b => a < b) in
  ordered_permutations.nth 18 = [9, 1, 3, 6] :=
sorry

end permutation_nineteenth_position_l510_510943


namespace find_x_for_g_inv_eq_3_l510_510025

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 5

-- State the theorem
theorem find_x_for_g_inv_eq_3 : ∃ x : ℝ, g x = 113 :=
by
  exists 3
  unfold g
  norm_num

end find_x_for_g_inv_eq_3_l510_510025


namespace combined_time_l510_510282

theorem combined_time {t_car t_train t_combined : ℝ} 
  (h1: t_car = 4.5) 
  (h2: t_train = t_car + 2) 
  (h3: t_combined = t_car + t_train) : 
  t_combined = 11 := by
  sorry

end combined_time_l510_510282


namespace problem_projection_eq_l510_510033

variable (m n : ℝ × ℝ)
variable (m_val : m = (1, 2))
variable (n_val : n = (2, 3))

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def projection (u v : ℝ × ℝ) : ℝ :=
  (dot_product u v) / (magnitude v)

theorem problem_projection_eq : projection m n = (8 * Real.sqrt 13) / 13 :=
by
  rw [m_val, n_val]
  sorry

end problem_projection_eq_l510_510033


namespace petya_wins_if_and_only_if_odd_l510_510937

theorem petya_wins_if_and_only_if_odd (n : ℕ) : (odd n ↔ ∃ win_strategy : (ℕ → Prop), win_strategy n) :=
by sorry

end petya_wins_if_and_only_if_odd_l510_510937


namespace geo_series_sum_l510_510351

theorem geo_series_sum (a r : ℚ) (n: ℕ) (ha : a = 1/3) (hr : r = 1/2) (hn : n = 8) : 
    (a * (1 - r^n) / (1 - r)) = 85 / 128 := 
by
  sorry

end geo_series_sum_l510_510351


namespace solution_set_of_inequality_l510_510722

theorem solution_set_of_inequality :
  {x : ℝ | -6 * x ^ 2 - x + 2 < 0} = {x : ℝ | x < -(2 / 3)} ∪ {x | x > 1 / 2} := 
sorry

end solution_set_of_inequality_l510_510722


namespace point_P_locus_is_circumcircle_l510_510984

-- Define the points and triangles involved in the statement
variables {A B C K M N P : Type} [euclidean_geometry]

-- Define the conditions for the problem
axiom on_line_AB (K : Point) (A B : Point) : K ∈ line A B ∨ K ∈ extension (line A B)
axiom secant_through_K (K : Point) (A B C : Point) : ∃ (M N : Point), (line K M).secant (line A C) ∧ (line K N).secant (line B C)
axiom circumcircles_intersect (A K M B K N P : Point) : P ∈ circle A K M ∧ P ∈ circle B K N

-- Define the proof problem statement
theorem point_P_locus_is_circumcircle (A B C K M N P : Point)
  (h1 : K ∈ line A B ∨ K ∈ extension (line A B))
  (h2 : ∃ (M N : Point), (line K M).secant (line A C) ∧ (line K N).secant (line B C))
  (h3 : P ∈ circle A K M ∧ P ∈ circle B K N):
  P ∈ circumcircle (triangle A B C) :=
sorry

end point_P_locus_is_circumcircle_l510_510984


namespace unique_two_digit_u_l510_510561

theorem unique_two_digit_u:
  ∃! u : ℤ, 10 ≤ u ∧ u < 100 ∧ 
            (15 * u) % 100 = 45 ∧ 
            u % 17 = 7 :=
by
  -- To be completed in proof
  sorry

end unique_two_digit_u_l510_510561


namespace find_eccentricity_l510_510386

noncomputable def ellipse {a b : ℝ} (ha : a > 0) (hb : b > 0) (h_ab : a > b) :=
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def focus (a b : ℝ) :=
  sqrt (a^2 - b^2)

theorem find_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_ab : a > b) :
  let c := focus a b in
  let e := c / a in 
  (∃ P : ℝ × ℝ, (P.1 > 0) ∧ (P.2 > 0) ∧ ellipse ha hb h_ab P.1 P.2 ∧ right_triangle (0, 0) P (c, 0)) →
  e = sqrt 3 - 1 :=
by
  sorry

end find_eccentricity_l510_510386


namespace books_per_week_l510_510683

theorem books_per_week 
  (total_books : ℕ)
  (books_first_week : ℕ)
  (books_second_week : ℕ)
  (total_weeks : ℕ)
  (remaining_books : ℕ)
  (remaining_weeks : ℕ)
  (x : ℕ) :
  total_books = 54 →
  books_first_week = 6 →
  books_second_week = 3 →
  total_weeks = 7 →
  remaining_books = total_books - (books_first_week + books_second_week) →
  remaining_weeks = total_weeks - 2 →
  x = remaining_books / remaining_weeks →
  x = 9 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  have h_rem_books := calc
    remaining_books = total_books - (books_first_week + books_second_week) : by rw [h5]
                    ... = 54 - (6 + 3)                     : by rw [h1, h2, h3]
                    ... = 45                               : by norm_num
  have h_rem_weeks := calc
    remaining_weeks = total_weeks - 2 : by rw [h6]
                    ... = 7 - 2       : by rw [h4]
                    ... = 5           : by norm_num
  have h_div := calc
    x = remaining_books / remaining_weeks : by rw [h7]
      ... = 45 / 5                         : by rw [h_rem_books, h_rem_weeks]
      ... = 9                              : by norm_num
  exact h_div

end books_per_week_l510_510683


namespace marie_days_to_pay_cash_register_l510_510505

def daily_revenue_bread (loaves: Nat) (price_per_loaf: Nat) : Nat := loaves * price_per_loaf
def daily_revenue_cakes (cakes: Nat) (price_per_cake: Nat) : Nat := cakes * price_per_cake
def total_daily_revenue (loaves: Nat) (price_per_loaf: Nat) (cakes: Nat) (price_per_cake: Nat) : Nat :=
  daily_revenue_bread loaves price_per_loaf + daily_revenue_cakes cakes price_per_cake

def daily_expenses (rent: Nat) (electricity: Nat) : Nat := rent + electricity

def daily_profit (loaves: Nat) (price_per_loaf: Nat) (cakes: Nat) (price_per_cake: Nat) (rent: Nat) (electricity: Nat) : Nat :=
  total_daily_revenue loaves price_per_loaf cakes price_per_cake - daily_expenses rent electricity

def days_to_pay_cash_register (register_cost: Nat) (profit: Nat) : Nat :=
  register_cost / profit

theorem marie_days_to_pay_cash_register :
  days_to_pay_cash_register 1040 (daily_profit 40 2 6 12 20 2) = 8 :=
by
  calc
    days_to_pay_cash_register 1040 (daily_profit 40 2 6 12 20 2)
        = 1040 / daily_profit 40 2 6 12 20 2 : by rfl
    ... = 1040 / 130 : by rfl
    ... = 8 : by rfl

end marie_days_to_pay_cash_register_l510_510505


namespace common_diff_is_2_l510_510744

-- Define the arithmetic sequence and the sum of the first n terms S_n
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∃ a1 : ℝ, ∀ n : ℕ, a n = a1 + n * d

def sum_first_n_terms {a : ℕ → ℝ} (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = n / 2 * (2 * a 0 + (n - 1) * (a 1 - a 0))

-- Given conditions in the problem
def condition (S : ℕ → ℝ) :=
  (S 4 / 4) - (S 2 / 2) = 2 

-- Main theorem to prove
theorem common_diff_is_2 (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : is_arithmetic_sequence a)
  (h2 : sum_first_n_terms S) (h3 : condition S) : 
  ∃ d : ℝ, d = 2 :=
by
  sorry

end common_diff_is_2_l510_510744


namespace peter_wins_if_and_only_if_n_is_odd_l510_510935

-- Define the problem parameters and assumptions
variable (n : ℕ) (k : ℕ) 
variable (empty_cups : Fin 2n → Bool)
variable (peter_wins : Bool)

-- Define the conditions
def symmetric (x y : Fin 2n) : Prop := x.val + y.val = 2n
def can_pour_tea (x : Fin 2n) : Prop := empty_cups x = true
def can_pour_symmetric_tea (x y : Fin 2n) : Prop := symmetric x y ∧ empty_cups x = true ∧ empty_cups y = true

-- Define a property stating Peter wins if and only if n is odd
def peter_wins_iff (n : ℕ) : Prop := peter_wins = (Odd n)

theorem peter_wins_if_and_only_if_n_is_odd :
  (∀ n, (∀ (x : Fin 2n), can_pour_tea empty_cups x ∨ (∃ y, can_pour_symmetric_tea empty_cups x y) ∨ ¬ can_pour_tea empty_cups x) →
  peter_wins_iff n) :=
by
  sorry

end peter_wins_if_and_only_if_n_is_odd_l510_510935


namespace smallest_n_sequence_property_l510_510115

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  (a 1 = 1) ∧
  (a 2 = 3) ∧
  (∀ n : ℕ, n ≥ 3 → a n = 3 * a (n-1) - a (n-2))

theorem smallest_n_sequence_property (a : ℕ → ℤ) (h : sequence a) :
  ∃ n : ℕ, (2^2016 ∣ a n) ∧ (¬ (2^2017 ∣ a n)) ∧ n = 3 * 2^2013 :=
  sorry

end smallest_n_sequence_property_l510_510115


namespace hyperbola_asymptotes_identical_l510_510314

theorem hyperbola_asymptotes_identical (M : ℚ) :
  (∀ x y : ℚ, (x^2 / 9 - y^2 / 16 = 1) → (y = 4/3 * x ∨ y = -4/3 * x)) →
  (∀ x y : ℚ, (y^2 / 25 - x^2 / M = 1) → (y = 5 / (sqrt M) * x ∨ y = -5 / (sqrt M) * x)) →
  M = 225 / 16 :=
by
  sorry

end hyperbola_asymptotes_identical_l510_510314


namespace prob_sin_2x_l510_510263

open Real

/-- Proof problem: 
Given x randomly selected from the interval [-π/4, π/4],
prove that the probability of sin(2x) falling between 0 and √3/2 is 1/3. 
-/
theorem prob_sin_2x (h : x ∈ Icc (-π/4) (π/4)) : 
  let P := measure (Icc 0 (π/6)) / measure (Icc (-π/4) (π/4)) 
  in P = 1/3 := by sorry

variables 
(x : ℝ)
(sqrt_three_div_two := Real.sqrt (3/2))
(h : x ∈ Icc (-π/4) (π/4))

noncomputable def prob_sin_2x_statement : ℚ :=
  let total_interval_length := measure (Icc (-π/4) (π/4))
  let interval_sin := measure (Icc 0 (π/6))
  interval_sin / total_interval_length

end prob_sin_2x_l510_510263


namespace coefficient_of_x_is_nine_l510_510723

theorem coefficient_of_x_is_nine (x : ℝ) (c : ℝ) (h : x = 0.5) (eq : 2 * x^2 + c * x - 5 = 0) : c = 9 :=
by
  sorry

end coefficient_of_x_is_nine_l510_510723


namespace bruce_purchased_mangoes_l510_510677

noncomputable def calculate_mango_quantity (grapes_quantity : ℕ) (grapes_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : ℕ :=
  let cost_of_grapes := grapes_quantity * grapes_rate
  let cost_of_mangoes := total_paid - cost_of_grapes
  cost_of_mangoes / mango_rate

theorem bruce_purchased_mangoes :
  calculate_mango_quantity 8 70 55 1055 = 9 :=
by
  sorry

end bruce_purchased_mangoes_l510_510677


namespace slope_of_line_through_directrix_intersecting_parabola_l510_510780

theorem slope_of_line_through_directrix_intersecting_parabola
  (p : ℝ) (hp : 0 < p)
  (A : ℝ × ℝ)
  (hA : A.2 ^ 2 = 2 * p * A.1)
  (M : ℝ × ℝ)
  (hM : M = (-p / 2, 0))
  (AF : ℝ)
  (hAF : AF = A.1 + p / 2)
  (hAM_AF : real.sqrt ((A.1 + p / 2) ^ 2 + A.2^2) = (5 / 4) * (A.1 + p / 2))
  : ∃ k : ℝ, k = 3 / 4 ∨ k = -3 / 4 :=
by
  sorry

end slope_of_line_through_directrix_intersecting_parabola_l510_510780


namespace product_less_than_ignored_l510_510695

def sequence_a (n : ℕ) : ℕ := 2^(2^n) + 1

theorem product_less_than_ignored (k : ℕ) : 
  (∏ i in Finset.range k, sequence_a i) = sequence_a k - 2 := 
sorry

end product_less_than_ignored_l510_510695


namespace find_y_l510_510524

def star (a b : ℕ) : ℕ := 4 * a - b

theorem find_y :
  ∃ y : ℕ, star 3 (star 6 y) = 4 ∧ y = 16 :=
by
  use 16
  unfold star
  simp
  sorry

end find_y_l510_510524


namespace train_and_car_combined_time_l510_510286

theorem train_and_car_combined_time (car_time : ℝ) (train_time : ℝ) 
  (h1 : train_time = car_time + 2) (h2 : car_time = 4.5) : 
  car_time + train_time = 11 := 
by 
  -- Proof goes here
  sorry

end train_and_car_combined_time_l510_510286


namespace P_lies_on_BD_l510_510826

-- Define the points, the polygon, and the required conditions.
variables {A B C D E F P : Type} [metric_space P]

-- Assume points A, B, C, D form a rhombus.
axiom rhombus (A B C D : P) : geometric. rhombus A B C D

-- E is the midpoint of segment AB
axiom midpoint_E {A B E : P} : geometric. midpoint A B E

-- F is the midpoint of segment BC
axiom midpoint_F {B C F : P} : geometric. midpoint B C F

-- P satisfies PA = PF
axiom pa_eq_pf {A F P : P} : pa = pf

-- P satisfies PE = PC
axiom pe_eq_pc {E C P : P} : pe = pc

-- Prove that P lies on the line BD
theorem P_lies_on_BD (h : geometric. rhombus A B C D) 
  (h1 : geometric.midpoint A B E) 
  (h2 : geometric.midpoint B C F) 
  (h3: pa_eq_pf) 
  (h4: pe_eq_pc) : 
  geometric.lies_on_line P (geometric.line_through B D) :=
sorry

end P_lies_on_BD_l510_510826


namespace verify_a_eq_x0_verify_p_squared_ge_4x0q_l510_510243

theorem verify_a_eq_x0 (p q x0 a b : ℝ) (hx0_root : x0^3 + p * x0 + q = 0) 
  (h_eq : ∀ x : ℝ, x^3 + p * x + q = (x - x0) * (x^2 + a * x + b)) : 
  a = x0 :=
by
  sorry

theorem verify_p_squared_ge_4x0q (p q x0 b : ℝ) (hx0_root : x0^3 + p * x0 + q = 0) 
  (h_eq : ∀ x : ℝ, x^3 + p * x + q = (x - x0) * (x^2 + x0 * x + b)) : 
  p^2 ≥ 4 * x0 * q :=
by
  sorry

end verify_a_eq_x0_verify_p_squared_ge_4x0q_l510_510243


namespace length_of_hall_l510_510056

theorem length_of_hall 
  (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
  (width_eq : width = 15) 
  (cost_per_sqm_eq : cost_per_sqm = 50) 
  (total_cost_eq : total_cost = 47500) :
  let total_area := total_cost / cost_per_sqm in 
  let length := total_area / width in
  length ≈ 63.33 :=
by
  sorry

end length_of_hall_l510_510056


namespace triangle_area_ratio_l510_510817

open Set 

variables {X Y Z W : Type} 
variable [LinearOrder X]

noncomputable def ratio_areas (XW WZ : ℕ) (h : ℕ) : ℚ :=
  (8 * h : ℚ) / (12 * h)

theorem triangle_area_ratio (XW WZ : ℕ) (h : ℕ)
  (hXW : XW = 8)
  (hWZ : WZ = 12) :
  ratio_areas XW WZ h = 2 / 3 :=
by
  rw [hXW, hWZ]
  unfold ratio_areas
  norm_num
  sorry

end triangle_area_ratio_l510_510817


namespace one_add_i_cubed_eq_one_sub_i_l510_510429

theorem one_add_i_cubed_eq_one_sub_i (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end one_add_i_cubed_eq_one_sub_i_l510_510429


namespace probability_second_question_correct_l510_510800

variables {Ω : Type} [ProbabilitySpace Ω]
variables (A B : Event Ω)

theorem probability_second_question_correct
  (hA : P(A) = 0.75)
  (hA_and_B : P(A ∩ B) = 0.30)
  (hA_not_B_not : P(Aᶜ ∩ Bᶜ) = 0.20) :
  P(B) = 0.35 :=
by
  sorry

end probability_second_question_correct_l510_510800


namespace circles_tangent_iff_l510_510788

noncomputable def C1 := { p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1 }
noncomputable def C2 (m: ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 - 8 * p.1 + 8 * p.2 + m = 0 }

theorem circles_tangent_iff (m: ℝ) : (∀ p ∈ C1, p ∈ C2 m → False) ↔ (m = -4 ∨ m = 16) := 
sorry

end circles_tangent_iff_l510_510788


namespace min_value_a_plus_b_l510_510271

noncomputable def sequence (a b : ℕ) : ℕ → ℕ
| 0     := a
| 1     := b
| (n+2) := sequence n + sequence (n+1)

theorem min_value_a_plus_b (a b : ℕ) (n : ℕ)
  (h : ∃ k, sequence a b k = 1000) :
  a + b = 10 :=
sorry

end min_value_a_plus_b_l510_510271


namespace sum_of_digits_n_minus_one_l510_510632

noncomputable def digitSum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

def digitsAreDistinct (n : ℕ) : Prop :=
let ds := n.digits 10 in
ds.nodup

def validNumber (n : ℕ) : Prop :=
digitsAreDistinct n ∧ digitSum n = 44

theorem sum_of_digits_n_minus_one (n : ℕ) (h : validNumber n) :
  digitSum (n - 1) = 43 ∨ digitSum (n - 1) = 52 := 
sorry

end sum_of_digits_n_minus_one_l510_510632


namespace num_divisors_of_81n4_l510_510726

theorem num_divisors_of_81n4 (n : ℕ) (hn_pos : 0 < n) (h_div : 110.divisor_count = 110) :
  (81 * n ^ 4).divisor_count = 325 :=
sorry

end num_divisors_of_81n4_l510_510726


namespace total_profit_l510_510660

-- Definitions based on the conditions
variables (A B C : ℝ) (P : ℝ)
variables (hA : A = 3 * B) (hB : B = (2 / 3) * C) (hB_share : ((2 / 11) * P) = 1400)

-- The theorem we are going to prove
theorem total_profit (A B C P : ℝ) (hA : A = 3 * B) (hB : B = (2 / 3) * C) (hB_share : ((2 / 11) * P) = 1400) : 
  P = 7700 :=
by
  sorry

end total_profit_l510_510660


namespace permutation_sum_eq_factorial_l510_510697

noncomputable def permutation_sum (n : ℕ) : ℚ :=
  ∑ σ in equiv.perm (fin n), 1 / (finset.range n).sum (λ i, (σ i).val.succ)

theorem permutation_sum_eq_factorial (n : ℕ) : permutation_sum n = 1 / n.factorial := by
  sorry

end permutation_sum_eq_factorial_l510_510697


namespace painted_cube_faces_l510_510626

theorem painted_cube_faces (a : ℕ) (h : 2 < a) :
  ∃ (one_face two_faces three_faces : ℕ),
  (one_face = 6 * (a - 2) ^ 2) ∧
  (two_faces = 12 * (a - 2)) ∧
  (three_faces = 8) := by
  sorry

end painted_cube_faces_l510_510626


namespace compound_interest_semiannual_l510_510602

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_semiannual :
  compound_interest 150 0.20 2 1 = 181.50 :=
by
  sorry

end compound_interest_semiannual_l510_510602


namespace roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells_l510_510861

theorem roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells 
  (k n : ℕ) (h_k : k = 4) (h_n : n = 100)
  (shift_rule : ∀ (m : ℕ), m ≤ n → 
    ∃ (chips_moved : ℕ), chips_moved = 1 ∧ chips_moved ≤ m) 
  : ∃ m, m ≤ n ∧ m = 50 := 
by
  sorry

end roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells_l510_510861


namespace time_to_cover_escalator_l510_510296

-- Define the given conditions
def escalator_speed : ℝ := 20 -- feet per second
def escalator_length : ℝ := 360 -- feet
def delay_time : ℝ := 5 -- seconds
def person_speed : ℝ := 4 -- feet per second

-- Define the statement to be proven
theorem time_to_cover_escalator : (delay_time + (escalator_length - (escalator_speed * delay_time)) / (person_speed + escalator_speed)) = 15.83 := 
by {
  sorry
}

end time_to_cover_escalator_l510_510296


namespace mean_increased_by_30_l510_510692

theorem mean_increased_by_30 (a : ℕ → ℝ) (h_len : ∀ n, n < 15 → a n ≠ 0) :
  let original_mean := (∑ i in finset.range 15, a i) / 15
  let new_set := λ i, a i + 30
  let new_mean := (∑ i in finset.range 15, new_set i) / 15
  in new_mean = original_mean + 30 :=
by
  sorry

end mean_increased_by_30_l510_510692


namespace fraction_of_left_handed_non_throwers_l510_510878

theorem fraction_of_left_handed_non_throwers 
  (total_players : ℕ) (throwers : ℕ) (right_handed_players : ℕ) (all_throwers_right_handed : throwers ≤ right_handed_players) 
  (total_players_eq : total_players = 70) 
  (throwers_eq : throwers = 46) 
  (right_handed_players_eq : right_handed_players = 62) 
  : (total_players - throwers) = 24 → ((right_handed_players - throwers) = 16 → (24 - 16) = 8 → ((8 : ℚ) / 24 = 1/3)) := 
by 
  intros;
  sorry

end fraction_of_left_handed_non_throwers_l510_510878


namespace crossing_time_l510_510237

-- Define the given conditions
def train_length : Real := 120
def train_speed_kmh : Real := 72
def kmh_to_ms_factor : Real := 5 / 18

-- Convert the speed to meters per second
def train_speed_ms : Real := train_speed_kmh * kmh_to_ms_factor

-- Prove the time taken for the train to cross an electric pole
theorem crossing_time (length : Real) (speed_kmh : Real) (factor : Real) : (length / (speed_kmh * factor)) = 6 :=
by
  have speed_in_ms := speed_kmh * factor
  have time := length / speed_in_ms
  sorry

end crossing_time_l510_510237


namespace option_B_is_incorrect_l510_510968

theorem option_B_is_incorrect (a : ℝ) : (|a^2| ≠ a) := by
  have h_cond : ∃ a, |a^2| ≠ a, from
    exists.intro (-1) (by 
      dsimp only [abs];
      norm_num
    )
  exact h_cond

end option_B_is_incorrect_l510_510968


namespace annual_growth_rate_l510_510622

theorem annual_growth_rate (P₁ P₂ : ℝ) (y : ℕ) (r : ℝ)
  (h₁ : P₁ = 1) 
  (h₂ : P₂ = 1.21)
  (h₃ : y = 2)
  (h_growth : P₂ = P₁ * (1 + r) ^ y) :
  r = 0.1 :=
by {
  sorry
}

end annual_growth_rate_l510_510622


namespace average_last_three_l510_510903

theorem average_last_three {a b c d e f g : ℝ} 
  (h_avg_all : (a + b + c + d + e + f + g) / 7 = 60)
  (h_avg_first_four : (a + b + c + d) / 4 = 55) : 
  (e + f + g) / 3 = 200 / 3 :=
by
  sorry

end average_last_three_l510_510903


namespace prove_B_eq_C_from_condition1_prove_B_eq_C_from_condition2_find_minimum_value_l510_510468

-- Definitions for angles and sides of the triangle
variables {A B C : ℝ} -- Angles in the triangle
variables {a b c : ℝ} -- Sides opposite to angles A, B, C respectively

-- Conditions as hypotheses
def condition1 (A B : ℝ) : Prop := (sin A) / (1 - cos A) = (sin (2 * B)) / (1 + cos (2 * B))
def condition2 (A B C : ℝ) : Prop := sin C * sin (B - A) = sin B * sin (C - A)

-- Proof problems:
theorem prove_B_eq_C_from_condition1 (h1 : condition1 A B) : B = C := sorry

theorem prove_B_eq_C_from_condition2 (h2 : condition2 A B C) : B = C := sorry

-- Given the equality B = C, find the minimum value
theorem find_minimum_value (h : B = C) : ∀ (a b c : ℝ), (frac (2 * a + b) c + (1 / cos B)) ≥ 5 :=
sorry

end prove_B_eq_C_from_condition1_prove_B_eq_C_from_condition2_find_minimum_value_l510_510468


namespace find_eccentricity_find_slope_l510_510746

open Real

variables (a b c x y k : ℝ) 

def ellipse_eq := x^2 / a^2 + y^2 / b^2 = 1
def foci (a b c : ℝ) := a > b ∧ b > 0 ∧ c > 0 ∧ a^2 = 3 * c^2 
def e := c / a
def line_eq := y = k * (x - a^2 / c)
def line_intersects_ellipse := (∃ (x₁ x₂ : ℝ), ellipse_eq x₁ (k * (x₁ - a^2 / c)) ∧ ellipse_eq x₂ (k * (x₂ - a^2 / c))) 

theorem find_eccentricity (h₁ : foci a b c) : e = sqrt 3 / 3 :=
by sorry

theorem find_slope (h₁ : foci a b c) (h₂ : b^2 = a^2 - c^2) (h₃ : line_intersects_ellipse a b c) : k = sqrt 2 / 3 ∨ k = -sqrt 2 / 3 := 
by sorry

end find_eccentricity_find_slope_l510_510746


namespace ratio_of_boys_to_girls_l510_510277

variable {α β γ : ℝ}
variable (x y : ℕ)

theorem ratio_of_boys_to_girls (hα : α ≠ 1/2) (hprob : (x * β + y * γ) / (x + y) = 1/2) :
  (x : ℝ) / (y : ℝ) = (1/2 - γ) / (β - 1/2) :=
by
  sorry

end ratio_of_boys_to_girls_l510_510277


namespace hyperbola_condition_l510_510430

theorem hyperbola_condition (k : ℝ) : (k > 1) -> ( ∀ x y : ℝ, (k - 1) * (k + 1) > 0 ↔ ( ∃ x y : ℝ, (k > 1) ∧ ((x * x) / (k - 1) - (y * y) / (k + 1)) = 1)) :=
sorry

end hyperbola_condition_l510_510430


namespace part_a_part_b_l510_510598

theorem part_a (x y : ℕ) (h : x^3 + 5 * y = y^3 + 5 * x) : x = y :=
sorry

theorem part_b : ∃ (x y : ℝ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (x^3 + 5 * y = y^3 + 5 * x) :=
sorry

end part_a_part_b_l510_510598


namespace smallest_square_perimeter_l510_510232

theorem smallest_square_perimeter 
  (perimeter_4_coins : ℝ) 
  (side_4_coins : ℝ) 
  (diameter_coin : ℝ) 
  (perimeter_441_coins : ℝ) : Prop :=
  (perimeter_4_coins = 56) →
  (side_4_coins = perimeter_4_coins / 4) →
  (diameter_coin = side_4_coins / 2) →
  let side_441_coins := 21 * diameter_coin in
  perimeter_441_coins = 4 * side_441_coins →
  perimeter_441_coins = 588


end smallest_square_perimeter_l510_510232


namespace is_isosceles_triangle_with_base_BC_l510_510094

variables {A B C O : Point}
variable {V : Type*} [InnerProductSpace ℝ V]
variable p : AffineSpace V ℝ

def is_noncollinear (A B C : Point) : Prop :=
  ¬ collinear p {A, B, C}

def vector_eq (P Q R S : Point) : Prop :=
  (P -ᵥ Q) ⋅ (R -ᵥ S) = 0

noncomputable def midpoint (P Q : Point) : Point :=
  (P +ᵥ Q) / 2

def is_isosceles_with_base (A B C : Point) : Prop :=
  dist p A B = dist p A C

theorem is_isosceles_triangle_with_base_BC 
  (h_noncollinear : is_noncollinear A B C)
  (h_condition : vector_eq (B -ᵥ O) (C -ᵥ O) (B +ᵥ C - 2 • A) 0) :
  is_isosceles_with_base A B C := 
sorry

end is_isosceles_triangle_with_base_BC_l510_510094


namespace net_income_correct_l510_510195

-- Definition of income before tax
def total_income_before_tax : ℝ := 45000

-- Definition of tax rate
def tax_rate : ℝ := 0.13

-- Definition of tax amount
def tax_amount : ℝ := tax_rate * total_income_before_tax

-- Definition of net income after tax
def net_income_after_tax : ℝ := total_income_before_tax - tax_amount

-- Theorem statement
theorem net_income_correct : net_income_after_tax = 39150 := by
  sorry

end net_income_correct_l510_510195


namespace carousel_seats_count_l510_510938

theorem carousel_seats_count :
  ∃ (yellow blue red : ℕ), 
  (yellow + blue + red = 100) ∧ 
  (yellow = 34) ∧ 
  (blue = 20) ∧ 
  (red = 46) ∧ 
  (∀ i : ℕ, i < yellow → ∃ j : ℕ, j = yellow.succ * j ∧ (j < 100 ∧ j ≠ yellow.succ * j)) ∧ 
  (∀ k : ℕ, k < blue → ∃ m : ℕ, m = blue.succ * m ∧ (m < 100 ∧ m ≠ blue.succ * m)) ∧ 
  (∀ n : ℕ, n < red → ∃ p : ℕ, p = red.succ * p ∧ (p < 100 ∧ p ≠ red.succ * p)) :=
sorry

end carousel_seats_count_l510_510938


namespace length_AB_trajectory_C_l510_510007

-- Definition of vertices A and B as foci of the ellipse
def ellipse : Type := set (ℝ × ℝ)
def fociA : ℝ × ℝ := (-2, 0)
def fociB : ℝ × ℝ := (2, 0)

-- Internal angles relation between A, B, C in triangle ABC
constant A B C : ℝ
axiom angles_relation : sin B - sin A = 1/2 * sin C

-- Given the properties of the ellipse and angles, prove the following:
theorem length_AB : dist fociA fociB = 4 := sorry

theorem trajectory_C : ∀ (x y : ℝ), ((x^2 - (y^2 / 3) = 1) → x > 1) := sorry

end length_AB_trajectory_C_l510_510007


namespace fraction_product_eq_l510_510684

theorem fraction_product_eq :
  (∏ n in finset.range 5, (n + 2)^3 - 1) / (∏ n in finset.range 5, (n + 2)^3 + 1) = 43 / 63 :=
by
  sorry

end fraction_product_eq_l510_510684


namespace minimum_value_expression_l510_510347

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ( (3*a*b - 6*b + a*(1-a))^2 + (9*b^2 + 2*a + 3*b*(1-a))^2 ) / (a^2 + 9*b^2) ≥ 4 :=
sorry

end minimum_value_expression_l510_510347


namespace Tarun_worked_days_l510_510299

-- Including all necessary imports
open Classical

variables (W : ℝ) -- Represents the total amount of work

-- Let's define the conditions
variables (combined_days : ℝ) (arun_days : ℝ) (arun_remaining_days : ℝ)
hypothesis h_combined : combined_days = 10
hypothesis h_arun_alone : arun_days = 60
hypothesis h_arun_remaining : arun_remaining_days = 36

-- The combined work rate of Arun and Tarun
def combined_rate := W / combined_days

-- The work rate of Arun alone
def arun_rate := W / arun_days

-- Number of days Tarun worked before going to his village
noncomputable def tarun_days_worked (T : ℝ) : Prop :=
  T * combined_rate + arun_remaining_days * arun_rate = W

-- The proof goal is to show T = 4 given the conditions
theorem Tarun_worked_days (T : ℝ) (h : tarun_days_worked W combined_days arun_days arun_remaining_days T) : T = 4 :=
sorry

end Tarun_worked_days_l510_510299


namespace xy_value_l510_510752

variable (a b x y : ℝ)
variable (h1 : 2 * a^x * b^3 = - a^2 * b^(1 - y))
variable (hx : x = 2)
variable (hy : y = -2)

theorem xy_value : x * y = -4 := 
by
  sorry

end xy_value_l510_510752


namespace impossible_to_end_with_zero_l510_510132

theorem impossible_to_end_with_zero :
  let initial_set := {n | 1 ≤ n ∧ n ≤ 1997} in
  ¬ (∃ seq : (Set ℕ) → (Set ℕ), 
    (seq initial_set = {0}) ∧
    (∀ (s: Set ℕ), ∀ (a b ∈ s), 
      a ≠ b → seq s = (s \ {a, b}) ∪ {|a - b|})) :=
by
  sorry

end impossible_to_end_with_zero_l510_510132


namespace center_of_gravity_hemisphere_correct_l510_510337

noncomputable def center_of_gravity_hemisphere (R k : ℝ) : ℝ :=
  let ρ := λ x y : ℝ, k * (x^2 + y^2)
  let z := λ x y : ℝ, Real.sqrt (R^2 - x^2 - y^2)
  let dS := λ x y : ℝ, R / Real.sqrt (R^2 - x^2 - y^2)
  let numerator := ∫∫ (λ x y, (z x y) * (ρ x y) * dS x y) {x y | x^2 + y^2 <= R^2}
  let denominator := ∫∫ (λ x y, (ρ x y) * dS x y) {x y | x^2 + y^2 <= R^2}
  numerator / denominator

theorem center_of_gravity_hemisphere_correct (R k : ℝ) : 
  center_of_gravity_hemisphere R k = (3 / 8) * R :=
sorry

end center_of_gravity_hemisphere_correct_l510_510337


namespace kendall_total_distance_l510_510849

def distance_with_mother : ℝ := 0.17
def distance_with_father : ℝ := 0.5
def total_distance : ℝ := 0.67

theorem kendall_total_distance :
  (distance_with_mother + distance_with_father = total_distance) :=
sorry

end kendall_total_distance_l510_510849


namespace average_episodes_per_year_l510_510612

def num_years : Nat := 14
def seasons_15_episodes : Nat := 8
def episodes_per_season_15 : Nat := 15
def seasons_20_episodes : Nat := 4
def episodes_per_season_20 : Nat := 20
def seasons_12_episodes : Nat := 2
def episodes_per_season_12 : Nat := 12

theorem average_episodes_per_year :
  (seasons_15_episodes * episodes_per_season_15 +
   seasons_20_episodes * episodes_per_season_20 +
   seasons_12_episodes * episodes_per_season_12) / num_years = 16 := by
  sorry

end average_episodes_per_year_l510_510612


namespace bake_sale_donation_l510_510669

theorem bake_sale_donation :
  let total_earning := 400
  let cost_of_ingredients := 100
  let donation_homeless_piggy := 10
  let total_donation_homeless := 160
  let donation_homeless := total_donation_homeless - donation_homeless_piggy
  let available_for_donation := total_earning - cost_of_ingredients
  let donation_food_bank := available_for_donation - donation_homeless
  (donation_homeless / donation_food_bank) = 1 := 
by
  sorry

end bake_sale_donation_l510_510669


namespace traffic_jam_speed_correct_l510_510323

noncomputable def find_traffic_jam_speed 
  (speed_car_traffic_jam : ℝ) 
  (speed_car_outside_jam : ℝ) 
  (v : ℝ) 
  (identical_time_readings : Prop) : Prop :=
  let t := (60 + v) * t / 60 = v * t / 10 in
  v = 12

theorem traffic_jam_speed_correct :
  find_traffic_jam_speed 10 60 v true :=
by
  sorry

end traffic_jam_speed_correct_l510_510323


namespace quadratic_complete_square_sum_l510_510925

theorem quadratic_complete_square_sum : ∃ b c : ℝ, (∀ x : ℝ, 2 * x^2 - 28 * x + 50 = 2 * (x + b)^2 + c) ∧ (b + c = -55) :=
by {
  use [-7, -48],
  split,
  { intro x, 
    calc 2 * x^2 - 28 * x + 50
        = 2 * (x^2 - 14 * x) + 50 : by ring
    ... = 2 * ((x - 7)^2 - 49) + 50 : by { congr, ring }
    ... = 2 * (x - 7)^2 - 98 + 50 : by ring
    ... = 2 * (x - 7)^2 - 48 : by ring },
  { norm_num }
}

end quadratic_complete_square_sum_l510_510925


namespace triangle_angle_contradiction_l510_510217

theorem triangle_angle_contradiction (α β γ : ℝ) (h : α + β + γ = 180) :
  (α > 60 ∧ β > 60 ∧ γ > 60) -> false :=
by
  sorry

end triangle_angle_contradiction_l510_510217


namespace verify_formula_n1_l510_510993

theorem verify_formula_n1 (a : ℝ) (ha : a ≠ 1) : 1 + a = (a^3 - 1) / (a - 1) :=
by 
  sorry

end verify_formula_n1_l510_510993


namespace village_population_l510_510995

/-- Define the initial population P for the given conditions. -/
def initial_population (P : ℝ) : Prop :=
  let after_bombardment := 0.95 * P;
  let after_fear := 0.85 * after_bombardment;
  after_fear = 2553

/-- Given the conditions in the problem, prove the initial population P. -/
theorem village_population : ∃ (P : ℝ), initial_population P ∧ P = 3162 :=
by
  -- We declare the initial population P as 3162.
  use 3162
  unfold initial_population
  -- Now, we need to check if the given value satisfies the conditions.
  have : 0.85 * (0.95 * 3162) = 2553 := by
    norm_num
  exact ⟨this, rfl⟩

end village_population_l510_510995


namespace area_of_triangle_COB_l510_510111

def point (ℝ : Type) := ℝ × ℝ

variable {x p : ℝ}

axiom B_is_on_x_axis (h : x > 0) : point ℝ := (x, 0)
axiom Q_is_on_y_axis : point ℝ := (0, 15)
axiom C_is_on_y_axis_0_lt_p_lt_15 (hp: 0 < p ∧ p < 15) : point ℝ := (0, p)

theorem area_of_triangle_COB {x p : ℝ} (hx : x > 0) (hp : 0 < p ∧ p < 15) :
  let B := B_is_on_x_axis hx,
      C := C_is_on_y_axis_0_lt_p_lt_15 hp,
      O : point ℝ := (0, 0)
  in (1 / 2) * x * p = (1 / 2) * x * p :=
by sorry

end area_of_triangle_COB_l510_510111


namespace find_m_l510_510727

theorem find_m (m x : ℝ) (h : (m - 2) * x^2 + 3 * x + m^2 - 4 = 0) (hx : x = 0) : m = -2 :=
by sorry

end find_m_l510_510727


namespace domain_of_ln_function_l510_510158

theorem domain_of_ln_function :
  {x : ℝ | 2 + x - x^2 > 0} = set.Ioo (-1 : ℝ) 2 :=
by
  sorry

end domain_of_ln_function_l510_510158


namespace smallest_repeating_block_of_4_div_7_has_6_digits_l510_510036

theorem smallest_repeating_block_of_4_div_7_has_6_digits : 
  ∃ n : ℕ, n = 6 ∧ (∃ m : ℕ, (m ≥ 0 ∧ (4 / 7 : ℚ).decimalExpansion m = "571428")) := 
sorry

end smallest_repeating_block_of_4_div_7_has_6_digits_l510_510036


namespace decreasing_quadratic_function_l510_510440

theorem decreasing_quadratic_function (a : ℝ) :
  (∀ x : ℝ, x ≤ (3/2) → deriv (λ x, x^2 + 2 * a * x - 1) x ≤ 0) →
  a ≤ - (3 / 2) :=
by
  intro h
  sorry

end decreasing_quadratic_function_l510_510440


namespace distances_B_C_possible_l510_510130

variable (A B C D : Type)
variable [Islander A] [Islander B] [Islander C] [Islander D]

variable (truth_teller liar : IslanderType)
variable (distance : Islander → Islander → ℕ)

-- Given conditions: 
axiom cond1 : ∃ k₁ k₂ ∈ {A, B, C, D}, k₁ ≠ k₂ ∧ ∀ x ∈ {A, B, C, D}, x ∈ {k₁, k₂} ↔ (x speaks_truth)
axiom cond2 : statement A = "My fellow tribesman in this row stands 3 meters away from me."
axiom cond3 : statement D = "My fellow tribesman in this row stands 2 meters away from me."

-- Define speaks_truth and statement functions
def speaks_truth (x : Islander) : Prop := 
  (x = A ∧ distance A x = 3 ∨ (x = D ∧ distance D x = 2))

-- Theorem to prove the possible distances mentioned by B and C
theorem distances_B_C_possible : 
  (distance B D = 2) ∧ (distance C D ≠ 2 ∧ distance C ≠ distance B) :=
by
  sorry

end distances_B_C_possible_l510_510130


namespace sum_of_oldest_three_ages_l510_510514

theorem sum_of_oldest_three_ages (a : ℕ) :
  let ages := [a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6] in
  (a + (a + 1) + (a + 2) = 42) → ((a + 4) + (a + 5) + (a + 6) = 54) :=
by {
  intro h,
  sorry
}

end sum_of_oldest_three_ages_l510_510514


namespace correlation_approx_neg_one_regression_equation_l510_510230

variables (x y : Fin 5 → ℝ) (x_vals : Vector ℝ 5) (y_vals : Vector ℝ 5)
          (x_bar : ℝ := (x_vals.getAvg))
          (y_bar : ℝ := (y_vals.getAvg))
          (n : ℕ := 5) -- Since we have 5 data points

-- Given x and y values
constant x_data : Fin 5 → ℝ := fun i => match i with
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5

constant y_data : Fin 5 → ℝ := fun i => match i with
  | 0 => 9.5
  | 1 => 8.6
  | 2 => 7.8
  | 3 => 7.0
  | 4 => 6.1

-- Condition Definitions
def sum_x_xbar_squared : ℝ := ∑ i, (x_data i - x_bar) ^ 2
def sum_y_ybar_squared : ℝ := ∑ i, (y_data i - y_bar) ^ 2
def sum_x_ybar_prod : ℝ := ∑ i, (x_data i - x_bar) * (y_data i - y_bar)

-- Question 1: Prove the correlation coefficient approximately equals -1
theorem correlation_approx_neg_one :
  let r := sum_x_ybar_prod / (Real.sqrt (sum_x_xbar_squared * sum_y_ybar_squared))
  r ≈ -1 := sorry

-- Question 2: Prove the regression equation
theorem regression_equation :
  let hat_b := sum_x_ybar_prod / sum_x_xbar_squared
  let hat_a := y_bar - hat_b * x_bar
  ∀ (x: ℝ), hat_a + hat_b * x = -0.84 * x + 10.32 := sorry

end correlation_approx_neg_one_regression_equation_l510_510230


namespace number_of_correct_propositions_l510_510766

noncomputable def perp_planes (plane1 plane2 : Plane) : Prop :=
  Plane.perpendicular plane1 plane2

def prop1 (line : Line) (plane1 plane2 : Plane) [L₁ : Line ℝ plane1] [L₂ : Line ℝ plane2] : Prop :=
  ∀ (l₁ : Line), ∃ (l₂ : Line), Line.in_plane l₁ plane1 → Line.in_plane l₂ plane2 → Line.perpendicular l₁ l₂

def prop2 (line : Line) (plane1 plane2 : Plane) [L₁ : Line ℝ plane1] [L₂ : Line ℝ plane2] : Prop :=
  ∃ (l₁ : Line), Line.in_plane l₁ plane1 → ∀ (l₂ : Line), Line.in_plane l₂ plane2 → Line.perpendicular l₁ l₂

def prop3 (plane1 plane2 : Plane) : Prop :=
  ∀ (l₁ : Line), Line.in_plane l₁ plane1 → ∀ (l₂ : Line), Line.in_plane l₂ plane2 → Line.perpendicular l₁ l₂

def prop4 (point : Point) (plane1 plane2 : Plane) [P : Point ℝ plane1] : Prop :=
  ∃ (l₁ : Line), P ∈ Line.of_point l₁ → Line.in_plane l₁ plane1 → ∃ (l₂ : Line), P ∈ Line.of_point l₂ → Line.in_plane l₂ plane2 → Line.perpendicular l₁ l₂

theorem number_of_correct_propositions (plane1 plane2 : Plane) (l : Line) (p : Point) :
  perp_planes plane1 plane2 → 
  ∃! (i : ℕ), i = (if prop1 l plane1 plane2 then 1 else 0) +
                   (if prop2 l plane1 plane2 then 1 else 0) +
                   (if prop3 plane1 plane2 then 1 else 0) +
                   (if prop4 p plane1 plane2 then 1 else 0) :=
by sorry

end number_of_correct_propositions_l510_510766


namespace part1_part2_l510_510403

noncomputable def f (m x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem part1 (t : ℝ) :
  (1 / 2 < t ∧ t < 1) →
  (∃! t : ℝ, f 1 t = 0) := sorry

theorem part2 :
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ ∀ x : ℝ, x > 0 → f m x > 0) := sorry

end part1_part2_l510_510403


namespace projection_a_onto_c_l510_510420

-- Define the vectors a, b, and c
def vector_a (m : ℝ) : ℝ × ℝ := (2, m)
def vector_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)
def vector_c : ℝ × ℝ := (2, 1)

-- Define the dot product of two vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude squared of a vector
def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

-- Define the projection of a vector a onto a vector c
def projection (a c : ℝ × ℝ) : ℝ × ℝ := 
  let factor := dot_product a c / magnitude_squared c in
  (factor * c.1, factor * c.2)

-- The main theorem to be proven
theorem projection_a_onto_c (m : ℝ)
  (h1 : vector_a m = (2, -2)) -- From solving the condition that a and b are opposite
  : projection (vector_a m) vector_c = (4/5, 2/5) :=
sorry

end projection_a_onto_c_l510_510420


namespace range_of_lambda_l510_510797

variable (λ : ℝ)

def vector_a : ℝ × ℝ := (λ, 2)
def vector_b : ℝ × ℝ := (3, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem range_of_lambda
  (h_ac : dot_product (vector_a λ) (vector_b) > 0)
  (h_collinear : 4 * λ ≠ 6) :
  λ > -8/3 ∧ λ ≠ -3/2 :=
sorry

end range_of_lambda_l510_510797


namespace matrix_problem_l510_510313

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -3], ![6, 1]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![-7, 8], ![3, -5]]
def RHS : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 2], ![15, -3]]

theorem matrix_problem : 
  2 • A + B = RHS :=
by
  sorry

end matrix_problem_l510_510313


namespace tangent_line_at_1_a_eq_4_range_a_for_f_positive_l510_510775

def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_at_1_a_eq_4 : 
  let f := f (·) 4 in 
  let f_prime := (λ x : ℝ, 1 + Real.log x - 4) in
  let tangent_line (x : ℝ) := -2 * (x - 1) 
  (f 1 = 0) ∧ (f_prime 1 = -2) ∧ 
  ∀ x : ℝ, tangent_line x = f x → 2 * x + tangent_line x - 2 = 0 := 
by
  sorry

theorem range_a_for_f_positive : 
  (∀ x : ℝ, x > 1 → f x a > 0) ↔ a ≤ 2 :=
by
  sorry

end tangent_line_at_1_a_eq_4_range_a_for_f_positive_l510_510775


namespace rational_number_div_eq_l510_510040

theorem rational_number_div_eq :
  ∃ x : ℚ, (-2 : ℚ) / x = 8 ∧ x = -1 / 4 :=
by
  existsi (-1 / 4 : ℚ)
  sorry

end rational_number_div_eq_l510_510040


namespace sine_product_identity_l510_510511

noncomputable def sine_product (n : ℕ) : ℝ := ∏ k in finset.range (n - 1), Real.sin ((k + 1) * Real.pi / n)

theorem sine_product_identity (n : ℕ) (h : n ≥ 2) : 
  sine_product n = n / 2^(n-1) := 
sorry

end sine_product_identity_l510_510511


namespace find_number_l510_510802

theorem find_number (n p q : ℝ) (h1 : n / p = 6) (h2 : n / q = 15) (h3 : p - q = 0.3) : n = 3 :=
by
  sorry

end find_number_l510_510802


namespace ellipse_equation_l510_510379

theorem ellipse_equation (a b c r : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : r > 0) (h4 : r < b)
  (heccentricity : (sqrt 3) / 2 = c / a) (hminor_axis : 2 * b = 2)  :
  (a = 2) ∧ (b = 1) ∧ ((∃ c, c = sqrt 3) ∧ (a^2 - c^2 = b^2)) ∧ (∀ (l : ℝ) (A B : ℝ × ℝ), 
  (A ≠ B) ∧ (l = r) → (A = (r, sqrt(1 - r^2 / 4)) ∨ A = (r, -sqrt(1 - r^2 / 4)))
  → (B = (r, sqrt(1 - r^2 / 4)) ∨ B = (r, -sqrt(1 - r^2 / 4))) → 
  (0 < r) ∧ (r < b) → (r = 2 * sqrt(5) / 5)) :=
begin
  sorry
end

end ellipse_equation_l510_510379


namespace tan_ratio_l510_510098

theorem tan_ratio (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5 / 8)
  (h2 : Real.sin (a - b) = 1 / 4) : 
  Real.tan a / Real.tan b = 7 / 3 := 
sorry

end tan_ratio_l510_510098


namespace collinearity_of_tangency_and_projection_l510_510851

/-- Let ABC be a triangle, with F and E being the points of tangency of the incircle with AB and AC respectively. 
Let P be the orthogonal projection of C onto the angle bisector from B. Show that E, F, and P are collinear. -/
theorem collinearity_of_tangency_and_projection
  (ABC : Triangle)
  (F E : Point)
  (hF : TangentToIncircle F (side AB))
  (hE : TangentToIncircle E (side AC))
  (P : Point)
  (hP : OrthogonalProjection P C (AngleBisector B))
  : Collinear E F P :=
by
  sorry

end collinearity_of_tangency_and_projection_l510_510851


namespace probability_factor_of_84_is_less_than_8_l510_510958

open Nat

noncomputable def number_of_factors (n : ℕ) : ℕ :=
  (1 to n).filter (λ d, n % d = 0).length

noncomputable def factors_less_than (n k : ℕ) : ℕ :=
  (1 to n).filter (λ d, n % d = 0 ∧ d < k).length

theorem probability_factor_of_84_is_less_than_8 : 
  (factors_less_than 84 8 : ℚ) / number_of_factors 84 = 1/2 := 
by {
  sorry -- Proof is skipped.
}

end probability_factor_of_84_is_less_than_8_l510_510958


namespace volume_of_pyramid_l510_510203

noncomputable def greatest_pyramid_volume (AB AC sin_α : ℝ) (max_angle : ℝ) : ℝ :=
  if AB = 3 ∧ AC = 5 ∧ sin_α = 4 / 5 ∧ max_angle ≤ 60 then
    5 * Real.sqrt 39 / 2
  else
    0

theorem volume_of_pyramid :
  greatest_pyramid_volume 3 5 (4 / 5) 60 = 5 * Real.sqrt 39 / 2 := by
  sorry -- Proof omitted as per instruction

end volume_of_pyramid_l510_510203


namespace angle_y_axis_and_plane_alpha_l510_510152

noncomputable def angle_between_vector_and_plane (
  j : ℝ × ℝ × ℝ,
  n : ℝ × ℝ × ℝ
) : ℝ :=
let cos_theta := (j.1 * n.1 + j.2 * n.2 + j.3 * n.3) / (real.sqrt (j.1^2 + j.2^2 + j.3^2) * real.sqrt (n.1^2 + n.2^2 + n.3^2)) in
real.arccos (abs cos_theta)

theorem angle_y_axis_and_plane_alpha :
  angle_between_vector_and_plane (0, 1, 0) (1, -1, 0) = π / 4 :=
by { sorry }

end angle_y_axis_and_plane_alpha_l510_510152


namespace richard_twice_scott_in_years_l510_510999

theorem richard_twice_scott_in_years :
  ∀ (Richard David Scott : ℕ), 
    (Richard = David + 6) → 
    (David = Scott + 8) → 
    (David = 7 + 7) →
    ∃ (x : ℕ), Richard + x = 2 * (Scott + x) ∧ x = 8 :=
by
  intros Richard David Scott h1 h2 h3
  use 8
  split
  . sorry
  . sorry

end richard_twice_scott_in_years_l510_510999


namespace fraction_decimal_equality_l510_510670

theorem fraction_decimal_equality : (1 / 9 : ℝ) + (1 / 11 : ℝ) = 0.2[0] := by
  sorry

end fraction_decimal_equality_l510_510670


namespace horse_speed_l510_510153

theorem horse_speed (A T : ℝ) (ha : A = 400) (ht : T = 4) : 
  let side := real.sqrt A in
  let perimeter := 4 * side in
  let speed := perimeter / T in
  speed = 20 :=
by
  sorry

end horse_speed_l510_510153


namespace two_numbers_are_opposites_l510_510244

theorem two_numbers_are_opposites (x y z : ℝ) (h : (1 / x) + (1 / y) + (1 / z) = 1 / (x + y + z)) :
  (x + y = 0) ∨ (x + z = 0) ∨ (y + z = 0) :=
by
  sorry

end two_numbers_are_opposites_l510_510244


namespace percent_formula_l510_510385

theorem percent_formula (x y p : ℝ) (h : x = (p / 100) * y) : p = 100 * x / y :=
by
    sorry

end percent_formula_l510_510385


namespace max_norm_of_z_l510_510433

open Complex

theorem max_norm_of_z (z : ℂ) (h : |z - (3 + 4i)| ≤ 2) : |z| ≤ 7 :=
sorry

end max_norm_of_z_l510_510433


namespace tape_thickness_correct_l510_510125

-- Define the given values
def side_duration : ℝ := 30 * 60  -- in seconds
def smallest_diameter : ℝ := 2    -- in cm
def largest_diameter : ℝ := 4.5   -- in cm
def playback_speed : ℝ := 4.75    -- in cm/s
def correct_thickness_mm : ℝ := 0.00373  -- in mm

-- Define the necessary constants and parameters converted to the appropriate units
def tape_duration : ℝ := 2 * side_duration
def smallest_radius : ℝ := smallest_diameter / 2  -- in cm
def largest_radius : ℝ := largest_diameter / 2  -- in cm

-- Calculate Total Length of the Tape
def tape_length : ℝ := playback_speed * tape_duration  -- in cm

-- Calculate the Thickness of the Tape
def tape_thickness_cm := (largest_radius^2 - smallest_radius^2) * Real.pi / tape_length  -- in cm

-- Convert Thickness to mm
def tape_thickness_mm : ℝ := tape_thickness_cm * 10  -- in mm

-- Theorem to be proven
theorem tape_thickness_correct : tape_thickness_mm = correct_thickness_mm := by
  -- skipping proof
  sorry

end tape_thickness_correct_l510_510125


namespace largest_possible_median_is_7_5_l510_510207

noncomputable def largest_possible_median (x y : ℤ) : ℚ :=
let s := {x, 2 * x, y, 3, 4, 7}.toFinset in
let sorted_s := s.sort in
(real.to_rat((sorted_s.nth_le 2) + (sorted_s.nth_le 3)) / 2 : ℚ)

theorem largest_possible_median_is_7_5 : ∃ x y : ℤ, largest_possible_median x y = 7.5 := 
sorry

end largest_possible_median_is_7_5_l510_510207


namespace new_ratio_of_partners_to_associates_l510_510259

theorem new_ratio_of_partners_to_associates
  (partners associates : ℕ)
  (rat_partners_associates : 2 * associates = 63 * partners)
  (partners_count : partners = 18)
  (add_assoc : associates + 45 = 612) :
  (partners:ℚ) / (associates + 45) = 1 / 34 :=
by
  -- Actual proof goes here
  sorry

end new_ratio_of_partners_to_associates_l510_510259


namespace combined_time_l510_510280

theorem combined_time {t_car t_train t_combined : ℝ} 
  (h1: t_car = 4.5) 
  (h2: t_train = t_car + 2) 
  (h3: t_combined = t_car + t_train) : 
  t_combined = 11 := by
  sorry

end combined_time_l510_510280


namespace distinct_values_of_g_l510_510496

def floor_fun {R : Type*} [LinearOrderedField R] (x : R) := ⌊x⌋

noncomputable def g (x : ℝ) :=
  ∑ k in Finset.range (15 - 5 + 1) \ + 5, floor_fun (k * x) - k * floor_fun x

theorem distinct_values_of_g :
  ∀ x, x ≥ 0 → ∃ n, n = 67 :=
by
  sorry

end distinct_values_of_g_l510_510496


namespace symmetric_trapezoid_property_quadrilateral_symmetric_trapezoid_l510_510235

-- Part (a)
theorem symmetric_trapezoid_property (a b c e : ℝ) (h: a > b) :
  (a + b < 2*c) ∧ (c < e) ∧ (√(c^2 - (a - b)^2 / 4) = e / 2) → (a * b = e^2 - c^2) :=
by sorry

-- Part (b)
theorem quadrilateral_symmetric_trapezoid (p q r : ℝ) (h: p < q) :
  (RS = (q^2 - p^2) / r) 
  ∧ (∃ PQ, quad PQ p q r = PQ) 
  → symmetric_trapezoid PQ SR :=
by sorry

end symmetric_trapezoid_property_quadrilateral_symmetric_trapezoid_l510_510235


namespace price_of_baseball_bat_l510_510889

theorem price_of_baseball_bat 
  (price_A : ℕ) (price_B : ℕ) (price_bat : ℕ) 
  (hA : price_A = 10 * 29)
  (hB : price_B = 14 * (25 / 10))
  (h0 : price_A = price_B + price_bat + 237) :
  price_bat = 18 :=
by
  sorry

end price_of_baseball_bat_l510_510889


namespace angle_BCA_45_degrees_l510_510075

variables {A B C D E : Type} {angle : A → B → C → Real} 
variables (α : A) (β : A → (C → D → Real)) 
variables (γ : E → Real) (tangent : CIRCumcircle → A → D → Prop)
variables (projection : E → B)
variables (CA CD AE CE : Real) {D BC AB}

theorem angle_BCA_45_degrees
  (h1 : angle A > 90)
  (h2 : D ∈ BC)
  (h3 : tangent A (circumcircle (triangle A C D)) A)
  (h4 : projection B A = E)
  (h5 : CA = CD)
  (h6 : AE = CE) :
  angle B C A = 45 :=
sorry

end angle_BCA_45_degrees_l510_510075


namespace part_a_part_b_l510_510879

-- Definitions and conditions
variable (A B C A1 B1 C1 T : Type)
variable (x : ℝ)  -- let x be the scale

-- BT : B1T = 2 : 5 implies x is the common factor for the ratio
def BT : ℝ := 2 * x
def B1T : ℝ := 5 * x
def BB1 : ℝ := 7 * x

-- Using the Pythagorean theorem on the triangle ABT 
def AB : ℝ := x * Real.sqrt 21

-- Conditions equations derived
def height_ratio (h : ℝ) : ℝ := h / AB

-- Given CC1 = 7
def height_prism : ℝ := 7

-- Calculating the volume of the cone
noncomputable def circumradius : ℝ := 10 * x * Real.sqrt 7 / Real.sqrt 37
noncomputable def cone_height : ℝ := 15 * x / Real.sqrt 37
noncomputable def cone_volume : ℝ := (3500 * Real.pi * x^3) / (37 * Real.sqrt 37)

-- Lean statements (No proofs included)
theorem part_a (x : ℝ) (h : ℝ) : height_ratio h = Real.sqrt (7 / 3) := sorry

theorem part_b (x : ℝ) : cone_volume = (3500 * Real.pi) / (37 * Real.sqrt 37) := sorry

end part_a_part_b_l510_510879


namespace total_tax_percent_is_correct_l510_510239

variable (amount_spent : ℝ) (clothing_percent food_percent other_percent : ℝ)
variable (clothing_tax_rate food_tax_rate other_tax_rate : ℝ)

-- Conditions provided in the problem
def proportion_of_clothing := clothing_percent * amount_spent / 100
def proportion_of_food := food_percent * amount_spent / 100
def proportion_of_other := other_percent * amount_spent / 100

def clothing_tax := clothing_tax_rate * proportion_of_clothing / 100
def food_tax := food_tax_rate * proportion_of_food / 100
def other_tax := other_tax_rate * proportion_of_other / 100

def total_tax_paid := clothing_tax + food_tax + other_tax

def total_tax_percent := total_tax_paid / amount_spent * 100

-- Given conditions
axiom h_clothing_percent : clothing_percent = 50
axiom h_food_percent : food_percent = 20
axiom h_other_percent : other_percent = 30
axiom h_clothing_tax_rate : clothing_tax_rate = 4
axiom h_food_tax_rate : food_tax_rate = 0
axiom h_other_tax_rate : other_tax_rate = 8

-- Final goal to prove
theorem total_tax_percent_is_correct :
  total_tax_percent = 4.40 := by
  sorry

end total_tax_percent_is_correct_l510_510239


namespace set_of_a_where_A_subset_B_l510_510809

variable {a x : ℝ}

theorem set_of_a_where_A_subset_B (h : ∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) :
  6 ≤ a ∧ a ≤ 9 :=
by
  sorry

end set_of_a_where_A_subset_B_l510_510809


namespace min_triangle_area_l510_510565

def point_on_ellipse (x y : ℝ) : Prop :=
  y^2 + (x^2 / 4) = 1

def tangent_to_circle (x y : ℝ) (θ : ℝ) : Prop :=
  2 * x * cos θ + y * sin θ = 1

def min_area_triangle (E F : ℝ × ℝ) : Prop :=
  let area := 1 / 2 * (E.1 * F.2 - E.2 * F.1) in
  area = 1 / 2

theorem min_triangle_area 
  (P : ℝ × ℝ) 
  (h1 : point_on_ellipse P.1 P.2)
  (h2 : ∃ θ : ℝ, tangent_to_circle P.1 P.2 θ)
  (E : ℝ × ℝ) 
  (F : ℝ × ℝ) 
  (h3 : E = (1 / (2 * cos (π / 4)), 0) ∧ F = (0, 1 / sin (π / 4))) : min_area_triangle E F :=
by
  sorry

end min_triangle_area_l510_510565


namespace minimum_races_to_find_top3_l510_510128

-- Define a constant to represent the number of horses and maximum horses per race
def total_horses : ℕ := 25
def max_horses_per_race : ℕ := 5

-- Define the problem statement as a theorem
theorem minimum_races_to_find_top3 (total_horses : ℕ) (max_horses_per_race : ℕ) : ℕ :=
  if total_horses = 25 ∧ max_horses_per_race = 5 then 7 else sorry

end minimum_races_to_find_top3_l510_510128


namespace range_of_x0_l510_510481

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | -x^2 + 4*x - 3 ≥ 0}
def f (x : ℝ) : ℝ := x + 1

theorem range_of_x0 (x0 : ℝ) (h1 : x0 ∈ A) (h2 : f (f x0) ∈ A) : x0 ∈ Iio (-3) := by
  sorry

end range_of_x0_l510_510481


namespace remainder_7623_div_11_l510_510208

theorem remainder_7623_div_11 : 7623 % 11 = 0 := 
by sorry

end remainder_7623_div_11_l510_510208


namespace parabola_equation_l510_510172

/-- 
Given a parabola with vertex at the origin and symmetric about the y-axis,
and a chord passing through the focus and perpendicular to the y-axis having length 16,
the equation of the parabola is either x^2 = 32y or x^2 = -32y.
-/

theorem parabola_equation (h_vertex : ∀ x y : ℝ, (0, 0) = (x, y) → vertex (x, y))
    (h_symmetry : ∀ x y : ℝ, symmetric_about_y_axis (parabola (x, y)))
    (h_chord : chord_length = 16) :
    (exists p : ℝ, x^2 = 4 * 8 * y ∨ x^2 = -4 * 8 * y) :=
by
  sorry

end parabola_equation_l510_510172


namespace conjugate_in_third_quadrant_l510_510368

-- Define the complex number z 
def z : ℂ := complex.I * (2 + complex.I)

-- Define the conjugate of the complex number z
def z_conj : ℂ := complex.conj z

-- Prove that the conjugate of z is in the third quadrant
theorem conjugate_in_third_quadrant : 
  (z_conj.re < 0) ∧ (z_conj.im < 0) :=
by
  -- Using sorry to skip the proof details
  sorry

end conjugate_in_third_quadrant_l510_510368


namespace correct_conclusions_l510_510965

noncomputable def quadratic_function (x b : ℝ) : ℝ :=
  x^2 - 2 * b * x + 3

-- 1. For any real number m, if m(m-2b) ≥ 1-2b always holds, then b=1
def conclusion_1 (b : ℝ) : Prop :=
  (∀ m : ℝ, m * (m - 2 * b) ≥ 1 - 2 * b) → b = 1

-- 2. The vertex of the quadratic function always lies on the parabola y = -x² + 3
def vertex_on_parabola (b : ℝ) : Prop :=
  let vertex_x := b in
  let vertex_y := -b^2 + 3 in
  vertex_y = quadratic_function b b

-- 3. Within the range -1 ≤ x ≤ 5, when the value of y is maximum at x = -1, we have m1 + m2 > 4 for m1 ≠ m2
def conclusion_3 (b x m1 m2 : ℝ) : Prop :=
  -1 ≤ x ∧ x ≤ 5 ∧ m1 ≠ m2 ∧ (8 = 2 * b) → m1 + m2 > 4

-- 4. For points (b-2n, y1) and (b+n, y2) lying on the graph, we have y1 < y2
def conclusion_4 (b n : ℝ) : Prop :=
  n ≠ 0 →
  let y1 := quadratic_function (b - 2 * n) b in
  let y2 := quadratic_function (b + n) b in
  y1 < y2

theorem correct_conclusions (b x m1 m2 n : ℝ) :
  conclusion_1 b ∧ vertex_on_parabola b ∧ conclusion_3 b x m1 m2 ∧ ¬ conclusion_4 b n :=
  sorry

end correct_conclusions_l510_510965


namespace area_of_trapezoid_in_acres_l510_510536

/-
Conditions:
1. The plot of land is represented such that 1 cm = 1 mile.
2. Every square mile equals 640 acres.
3. The plot is a trapezoid.
4. The bottom base of the trapezoid is 20 cm.
5. The top base of the trapezoid is 25 cm.
6. The height of the trapezoid is 15 cm.
-/

theorem area_of_trapezoid_in_acres
(top_base : ℝ) (bottom_base : ℝ) (height : ℝ)
(h_top_base : top_base = 25)
(h_bottom_base : bottom_base = 20)
(h_height : height = 15)
(one_cm_equals_one_mile : ∀ (x : ℝ), x * 1 = x)
(every_square_mile_640_acres : ∀ (x : ℝ), x * 640 = 640 * x) :
  let trapezoid_area_cm2 := ((bottom_base + top_base) / 2) * height in
  let trapezoid_area_mile2 := trapezoid_area_cm2 in
  trapezoid_area_mile2 * 640 = 216000 :=
by
  intros
  sorry

end area_of_trapezoid_in_acres_l510_510536


namespace find_common_ratio_l510_510015

variable {a : ℕ → ℝ}
variable (q : ℝ)

-- Define the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given conditions
def condition_1 := a 1 = 32
def condition_2 := a 6 = -1

-- Proof statement
theorem find_common_ratio (h1 : is_geometric_sequence a q)
                          (h2 : condition_1)
                          (h3 : condition_2) :
  q = -1/2 :=
by
  -- Proof goes here
  sorry

end find_common_ratio_l510_510015


namespace no_perfect_square_integers_l510_510330

open Nat

def Q (x : ℤ) : ℤ := x^4 + 4 * x^3 + 10 * x^2 + 4 * x + 29

theorem no_perfect_square_integers : ∀ x : ℤ, ¬∃ a : ℤ, Q x = a^2 :=
by
  sorry

end no_perfect_square_integers_l510_510330


namespace sin_and_tan_of_angle_l510_510743

variables {θ : ℝ} {x : ℝ}

theorem sin_and_tan_of_angle (h1 : x ≠ 0) (h2 : cos θ = x / 3) (P : x^2 + 4 = 9) :
  (sin θ = -2 / 3) ∧ (tan θ = 2 * sqrt 5 / 5 ∨ tan θ = -2 * sqrt 5 / 5) :=
by
  sorry

end sin_and_tan_of_angle_l510_510743


namespace hendrix_class_students_l510_510215

theorem hendrix_class_students (initial_students new_students : ℕ) (transferred_fraction : ℚ) 
  (h1 : initial_students = 160) (h2 : new_students = 20) (h3 : transferred_fraction = 1/3) :
  let total_students := initial_students + new_students
  let transferred_students := total_students * transferred_fraction
  let remaining_students := total_students - transferred_students
  remaining_students = 120 :=
by
  let total_students := initial_students + new_students
  let transferred_students := total_students * transferred_fraction
  let remaining_students := total_students - transferred_students
  have h4 : total_students = 180, by sorry
  have h5 : transferred_students = 60, by sorry
  have h6 : remaining_students = 120, by sorry
  exact h6

end hendrix_class_students_l510_510215


namespace slope_ge_one_sum_pq_eq_17_l510_510482

noncomputable def Q_prob_satisfaction : ℚ := 1/16

theorem slope_ge_one_sum_pq_eq_17 :
  let p := 1
  let q := 16
  p + q = 17 := by
  sorry

end slope_ge_one_sum_pq_eq_17_l510_510482


namespace part_I_part_II_l510_510734

-- Part (I): If a = 1, prove that q implies p
theorem part_I (x : ℝ) (h : 3 < x ∧ x < 4) : (1 < x) ∧ (x < 4) :=
by sorry

-- Part (II): Prove the range of a for which p is necessary but not sufficient for q
theorem part_II (a : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, (a < x ∧ x < 4 * a) → (3 < x ∧ x < 4)) : 1 < a ∧ a ≤ 3 :=
by sorry

end part_I_part_II_l510_510734


namespace final_cost_cooking_gear_sets_l510_510329

-- Definitions based on conditions
def hand_mitts_cost : ℕ := 14
def apron_cost : ℕ := 16
def utensils_cost : ℕ := 10
def knife_cost : ℕ := 2 * utensils_cost
def discount_rate : ℚ := 0.25
def sales_tax_rate : ℚ := 0.08
def number_of_recipients : ℕ := 3 + 5

-- Proof statement: calculate the final cost
theorem final_cost_cooking_gear_sets :
  let total_cost_before_discount := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
  let discounted_cost_per_set := (total_cost_before_discount : ℚ) * (1 - discount_rate)
  let total_cost_for_recipients := (discounted_cost_per_set * number_of_recipients : ℚ)
  let final_cost := total_cost_for_recipients * (1 + sales_tax_rate)
  final_cost = 388.80 :=
by
  sorry

end final_cost_cooking_gear_sets_l510_510329


namespace find_common_difference_l510_510931

-- Definitions of the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k : ℕ, a_n (k + 1) = a_n k + d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : Prop :=
  S_n = (n : ℝ) / 2 * (a_n 1 + a_n n)

variables {a_1 d : ℝ}
variables (a_n : ℕ → ℝ)
variables (S_3 S_9 : ℝ)

-- Conditions from the problem statement
axiom a2_eq_3 : a_n 2 = 3
axiom S9_eq_6S3 : S_9 = 6 * S_3

-- The proof we need to write
theorem find_common_difference 
  (h1 : arithmetic_sequence a_n d)
  (h2 : sum_of_first_n_terms a_n 3 S_3)
  (h3 : sum_of_first_n_terms a_n 9 S_9) :
  d = 1 :=
by
  sorry

end find_common_difference_l510_510931


namespace time_to_pass_telegraph_post_time_to_pass_cyclist_l510_510469

def car_length : ℝ := 10
def car_speed_kmph : ℝ := 36
def cyclist_length : ℝ := 2
def cyclist_speed_kmph : ℝ := 18
def kmph_to_mps : ℝ := 5 / 18

def car_speed_mps : ℝ := car_speed_kmph * kmph_to_mps
def cyclist_speed_mps : ℝ := cyclist_speed_kmph * kmph_to_mps

theorem time_to_pass_telegraph_post : 
  (car_length / car_speed_mps) = 1 := sorry

theorem time_to_pass_cyclist : 
  ((car_length + cyclist_length) / (car_speed_mps - cyclist_speed_mps)) = 2.4 := sorry

end time_to_pass_telegraph_post_time_to_pass_cyclist_l510_510469


namespace product_multiple_of_5_probability_l510_510562

theorem product_multiple_of_5_probability : 
  let total_numbers := 2020
  let multiples_of_5 := 404
  let p := 1 - ((1616 / total_numbers) * (1615 / (total_numbers - 1)) * (1614 / (total_numbers - 2)))
  p ≈ 0.485 := sorry

end product_multiple_of_5_probability_l510_510562


namespace domain_of_expression_l510_510714

theorem domain_of_expression (x : ℝ) : 
  (∃ f : ℝ → ℝ, f = (λ x, (sqrt (x - 3) + sqrt (x - 1)) / sqrt (6 - x)) ∧ ∀ x, 3 ≤ x ∧ x < 6 → f x = (sqrt (x - 3) + sqrt (x - 1)) / sqrt (6 - x) ) ↔ (3 ≤ x ∧ x < 6) :=
sorry

end domain_of_expression_l510_510714


namespace sequence_equality_l510_510245

theorem sequence_equality 
  (n : ℕ) (h_n : 0 < n) 
  (a b : Fin n → ℝ)
  (h_a : ∀ i j, i ≤ j → a i ≤ a j)
  (h_b : ∀ i j, i ≤ j → b i ≤ b j)
  (sum_cond1 : ∀ i, i < n → ∑ k in Finset.range (i + 1), a (Fin.mk k sorry) ≤ ∑ k in Finset.range (i + 1), b (Fin.mk k sorry))
  (sum_cond2 : ∑ i, a i = ∑ i, b i)
  (pair_cond : ∀ m : ℝ, (Finset.univ.filter (λ p, a p.1 - a p.2 = m)).card = (Finset.univ.filter (λ p, b p.1 - b p.2 = m)).card) :
  ∀ i, a i = b i := 
by
  sorry

end sequence_equality_l510_510245


namespace circumradius_of_triangle_l510_510290

theorem circumradius_of_triangle (a b c : ℝ)
  (ha : a = 8) (hb : b = 15) (hc : c = 17)
  (h_triangle : a^2 + b^2 = c^2) : 
  let R := c / 2 in 
  R = 17 / 2 := 
by 
  sorry

end circumradius_of_triangle_l510_510290


namespace average_episodes_per_year_l510_510618

theorem average_episodes_per_year (total_years : ℕ) (n1 n2 n3 e1 e2 e3 : ℕ) 
  (h1 : total_years = 14)
  (h2 : n1 = 8) (h3 : e1 = 15)
  (h4 : n2 = 4) (h5 : e2 = 20)
  (h6 : n3 = 2) (h7 : e3 = 12) :
  (n1 * e1 + n2 * e2 + n3 * e3) / total_years = 16 := by
  -- Skip the proof steps
  sorry

end average_episodes_per_year_l510_510618


namespace tetrahedron_volume_l510_510821

noncomputable def calculate_tetrahedron_volume
  (angle_BCD_ABC : ℝ)
  (area_triangle_ABC : ℝ)
  (area_triangle_BCD : ℝ)
  (BC_length : ℝ) : ℝ :=
  (1/3) * (16 * (sin (30 * (π / 180)))) * area_triangle_ABC

theorem tetrahedron_volume
  (h_angle : ∠BCD_ABC = 30)
  (h_area_ABC : area_triangle_ABC = 120)
  (h_area_BCD : area_triangle_BCD = 80)
  (h_BC : BC_length = 10) :
  calculate_tetrahedron_volume 30 120 80 10 = 320 :=
begin
  sorry
end

end tetrahedron_volume_l510_510821


namespace minimum_value_expression_l510_510348

theorem minimum_value_expression : 
  ∀ (a b : ℝ), (a > 0) → (b > 0) → 
  ( ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ 
    (a_0 b_0 : ℝ) = 4
  (3 * a * b - 6 * b + a * (1 - a))^2 + (9 * b^2 + 2 * a + 3 * b * (1 - a))^2 / (a^2 + 9 * b^2) = 4 
sory

end minimum_value_expression_l510_510348


namespace maximize_altitude_in_triangle_l510_510077

open Real

theorem maximize_altitude_in_triangle (A B C : Type) [RealPoint A] [RealPoint B] [RealPoint C]
    (BAC CBA BCA : Real)
    (AB AC BC : Real)
    (h1 : BAC = 50)
    (h2 : CBA ≤ 100)
    (h3 : AB = 1)
    (h4 : AC ≥ BC)
    (D : Type) [Segment D] (BD DC : Real)
    (h5 : BD = DC)
    (h6 : angle_of_triangles_eq A D B D = angle_of_triangles_eq A D C D) :
    CBA = 80 := 
by
  sorry

end maximize_altitude_in_triangle_l510_510077


namespace find_S16_l510_510097

variables (S : ℕ → ℤ) (a : ℕ → ℤ)

-- Given conditions
def condition_1 := a 12 = -8
def condition_2 := S 9 = -9

-- Definition of arithmetic sequence sum
def arithmetic_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (n : ℕ) :=
  S n = (n * (a 1 + a n) / 2)

-- Statement to prove
theorem find_S16 
  (h1 : condition_1 a)
  (h2 : condition_2 S) 
  (h3 : ∀ n, S n = arithmetic_sum S a n):
  S 16 = -72 :=
sorry

end find_S16_l510_510097


namespace problem_statement_l510_510801

-- Define the functions
def f (x : ℤ) : ℤ := x^2
def g (x : ℤ) : ℤ := 2 * x - 5

-- Define the main theorem statement
theorem problem_statement : f (g (-2)) = 81 := by
  sorry

end problem_statement_l510_510801


namespace m_value_if_linear_l510_510039

theorem m_value_if_linear (m : ℝ) (x : ℝ) (h : (m + 2) * x^(|m| - 1) + 8 = 0) (linear : |m| - 1 = 1) : m = 2 :=
sorry

end m_value_if_linear_l510_510039


namespace combinations_difference_is_n_14_l510_510359

theorem combinations_difference_is_n_14 (n : ℕ) :
  (nat.choose (n + 1) 7 - nat.choose n 7 = nat.choose n 8) -> n = 14 :=
by
  sorry

end combinations_difference_is_n_14_l510_510359


namespace fox_can_equalize_cheese_pieces_l510_510563

theorem fox_can_equalize_cheese_pieces :
  ∃ (n : ℕ), 
    ∃ (a : ℕ → (ℕ × ℕ)), 
    ∃ (cheese_weights : ℕ → ℕ), 
      cheese_weights 0 = 5 ∧ cheese_weights 1 = 8 ∧ cheese_weights 2 = 11 ∧
      (∀ i < n, 
        ∃ j k, 
          j ≠ k ∧ 
          a i = (j, k) ∧ 
          cheese_weights (i + 1) = λ w, 
            w - ite (w = j ∨ w = k) 1 0 ∨ w = cheese_weights i (j - 1) + cheese_weights i (k - 1)) ∧
      cheese_weights n = λ _, 2 := 
begin
  -- proof goes here
  sorry
end

end fox_can_equalize_cheese_pieces_l510_510563


namespace number_exceeds_fraction_l510_510976

theorem number_exceeds_fraction (x : ℝ) (h : x = (3/8) * x + 15) : x = 24 :=
sorry

end number_exceeds_fraction_l510_510976


namespace max_negative_sum_l510_510325

theorem max_negative_sum (a : Fin 50 → ℤ) (h : ∀ i, a i = 1 ∨ a i = -1) :
  ∑ i in Finset.filter (λ i, i.1 < i.2) (Finset.product Finset.univ Finset.univ), a i.1 * a i.2 = -23 :=
by
  sorry

end max_negative_sum_l510_510325


namespace exists_isosceles_right_triangle_cover_l510_510747

def obtuse_triangle (A B C : Type) : Prop :=
∃ (angle_A angle_B angle_C : ℝ),
  angle_A + angle_B + angle_C = 180 ∧ (angle_A > 90 ∨ angle_B > 90 ∨ angle_C > 90)

def circumcircle_radius_one (A B C O : Type) : Prop :=
  (∀ M : Type, obtuse_triangle A B C →
  ∃ (r : ℝ), r = 1)

noncomputable def isosceles_right_triangle_cover 
(A B C : Type) (h : obtuse_triangle A B C) 
(H : ∃ O, circumcircle_radius_one A B C O) : Prop :=
∃ D E F : Type, 
isosceles_right_triangle_cover = (∃ (hypotenuse_length : ℝ),
  hypotenuse_length = 1 + real.sqrt 2 ∧
  ∃ (S T : Type), obtuse_triangle S T F)

theorem exists_isosceles_right_triangle_cover 
(A B C : Type) 
(h : obtuse_triangle A B C) 
(H : ∃ O, circumcircle_radius_one A B C O) :
  isosceles_right_triangle_cover A B C h H := by
  sorry

end exists_isosceles_right_triangle_cover_l510_510747


namespace num_solutions_l510_510148

variable (n : ℕ)
variable (x : Fin n → ℝ)

noncomputable def system_solutions := ∀ (i : Fin n), 
  if i = 0 then (x (Fin.last n) ≠ 0) ∧ (x 0 = 1 / (x (Fin.last n))) else
    let t := List.prod (List.ofFn (λ j : Fin (i.val+1).succ => x j)) in
    (t - (if i = Fin.last n then 1 else (x (i+1)))) = 1

theorem num_solutions : system_solutions n x → (∃ t : List (Fin (n-1) → ℝ),
  (∀ i, t i = (1 + Real.sqrt5) / 2 ∨ t i = (1 - Real.sqrt5) / 2) ∧
  List.length t = n - 1 ∧
  2^(n-1) = List.length t)
:= 
sorry

end num_solutions_l510_510148


namespace max_numbers_called_l510_510881

/--
Given ten fifth-degree polynomials,
and given that Vasya calls consecutive natural numbers,
and Petya substitutes each called number into one of the ten polynomials,
if the recorded values form an arithmetic progression,
then the maximum number of numbers Vasya could call is 50.
-/
theorem max_numbers_called (P : Fin 10 → ℕ → ℕ)
  (h_poly : ∀ i, degree (P i) = 5)
  (h_prog : ∃ a b : ℤ, ∀ n, ∃ i, P i (Vasya_call n) = a * n + b) :
  Vasya_max_call P = 50 :=
sorry

end max_numbers_called_l510_510881


namespace isosceles_triangle_angle_Q_l510_510567

theorem isosceles_triangle_angle_Q (x : ℝ) (PQR : Triangle)
  (h1 : PQR.angles Q = PQR.angles R)
  (h2 : PQR.angles R = 5 * PQR.angles P)
  (sum_angles : PQR.angles P + PQR.angles Q + PQR.angles R = 180) :
  PQR.angles Q = 900 / 11 :=
by
  sorry

end isosceles_triangle_angle_Q_l510_510567


namespace avg_last_three_numbers_l510_510901

-- Definitions of conditions
def avg_seven_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.sum / 7 = 60)

def avg_first_four_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.take 4).sum / 4 = 55

-- Proof statement
theorem avg_last_three_numbers (numbers : List ℝ) (h_len : numbers.length = 7)
  (h1 : avg_seven_numbers numbers h_len)
  (h2 : avg_first_four_numbers numbers h_len) :
  (numbers.drop 4).sum / 3 = 200 / 3 :=
sorry

end avg_last_three_numbers_l510_510901


namespace shoe_length_at_size_15_l510_510262

noncomputable def shoe_length_size_15 : ℝ := 10.4

def size_range : ℕ × ℕ := (8, 17)

def unit_increase_in_length : ℝ := 1/5

def length_increase (initial_length : ℝ) (initial_size : ℕ) (target_size : ℕ) : ℝ :=
  initial_length + (target_size - initial_size) * unit_increase_in_length

def large_is_20_percent_longer (small_length : ℝ) (large_length : ℝ) : Prop :=
  large_length = 1.20 * small_length

-- Lean statement to prove the shoe length in size 15 is 10.4 inches given the conditions
theorem shoe_length_at_size_15 (small_length : ℝ) :
  (∃ small_length,
    (length_increase small_length 8 17 = 1.20 * small_length) ∧
    (shoe_length_size_15 = length_increase small_length 8 15)) :=
sorry

end shoe_length_at_size_15_l510_510262


namespace eccentricity_of_arithmetic_ellipse_is_three_fifths_slope_of_line_through_D_circle_through_fixed_points_l510_510668

noncomputable def eccentricity_of_arithmetic_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 2 * b = a + c) (h4 : a^2 - b^2 = c^2) : ℝ :=
  c / a

theorem eccentricity_of_arithmetic_ellipse_is_three_fifths (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : 2 * b = a + c) 
  (h4 : a^2 - b^2 = c^2) :
  eccentricity_of_arithmetic_ellipse a b c h1 h2 h3 h4 = 3 / 5 := 
  sorry

theorem slope_of_line_through_D (a b c k : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : 2 * b = a + c) 
  (h4 : a^2 - b^2 = c^2)
  (h5 : ∃ D : ℝ × ℝ, D = (0, -a)) :
  k = 3 / 5 ∨ k = -3 / 5 := 
  sorry

theorem circle_through_fixed_points (a b c P Q : ℝ × ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : 2 * b = a + c) 
  (h4 : a^2 - b^2 = c^2)
  (h5 : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (h6 : Q = (-P.1, -P.2))
  (h7 : P ≠ (a, 0)) :
  (0, -a * P.2 / (P.1 - a)) = (a, 0) ∨ (0, -a * P.2 / (P.1 + a)) = (-a, 0) := 
  sorry

end eccentricity_of_arithmetic_ellipse_is_three_fifths_slope_of_line_through_D_circle_through_fixed_points_l510_510668


namespace area_of_triangle_right_triangle_l510_510289

def point := ℝ × ℝ

def A : point := (3, -3)
def B : point := (8, 4)
def C : point := (3, 4)

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs ((fst B - fst A) * (snd C - snd A) - (fst C - fst A) * (snd B - snd A))

theorem area_of_triangle_right_triangle
  (A B C : point) (hA : A = (3, -3)) (hB : B = (8, 4)) (hC : C = (3, 4)) :
  area_of_triangle A B C = 17.5 :=
sorry

end area_of_triangle_right_triangle_l510_510289


namespace find_b2_b8_b11_product_l510_510738

noncomputable theory

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, b (n + 1) = b n * q

theorem find_b2_b8_b11_product 
  (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_geom : geometric_sequence b)
  (h_nonzero : ∀ n, a n ≠ 0) 
  (h_arith_cond : a 6 - (a 7)^2 + a 8 = 0) 
  (h_geom_eq : b 7 = a 7) : 
  b 2 * b 8 * b 11 = 8 := 
sorry

end find_b2_b8_b11_product_l510_510738


namespace greatest_b_solution_l510_510699

def f (b : ℝ) : ℝ := b^2 - 10 * b + 24

theorem greatest_b_solution : ∃ (b : ℝ), (f b ≤ 0) ∧ (∀ (b' : ℝ), (f b' ≤ 0) → b' ≤ b) ∧ b = 6 :=
by
  sorry

end greatest_b_solution_l510_510699


namespace probability_X_ge_zero_l510_510453
open Classical

noncomputable def normal_distribution (μ σ : ℝ) : MeasureTheory.ProbabilityMeasure ℝ := 
 by sorry -- Placeholder for the actual normal distribution definition

-- X follows a normal distribution N(1, σ^2)
def X : MeasureTheory.Measure ℝ := normal_distribution 1 σ

-- Given condition: The probability P(0 < X < 2) = 0.8
axiom cond1 : MeasureTheory.MeasureOf (normal_distribution 1 σ).toOuterMeasure (Set.Ioo 0 2) = 0.8

-- Proof problem statement: Given the normal distribution and condition, prove P(X ≥ 0) = 0.9
theorem probability_X_ge_zero (σ : ℝ) (hσ : 0 < σ) 
  (cond1 : MeasureTheory.MeasureOf (normal_distribution 1 σ).toOuterMeasure (Set.Ioo 0 2) = 0.8) : 
  MeasureTheory.MeasureOf (normal_distribution 1 σ).toOuterMeasure (Set.Ici 0) = 0.9 := 
 by sorry

end probability_X_ge_zero_l510_510453


namespace sufficient_but_not_necessary_condition_l510_510219

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x^2 + y^2 ≤ 1) → ((x - 1)^2 + y^2 ≤ 4) ∧ ¬ ((x - 1)^2 + y^2 ≤ 4 → x^2 + y^2 ≤ 1) :=
by sorry

end sufficient_but_not_necessary_condition_l510_510219


namespace find_a_l510_510912

-- Definitions for the problem
def quadratic_distinct_roots (a : ℝ) : Prop :=
  let Δ := a^2 - 16
  Δ > 0

def satisfies_root_equation (x1 x2 : ℝ) : Prop :=
  (x1^2 - (20 / (3 * x2^3)) = x2^2 - (20 / (3 * x1^3)))

-- Main statement of the proof problem
theorem find_a (a x1 x2 : ℝ) (h_quadratic_roots : quadratic_distinct_roots a)
               (h_root_equation : satisfies_root_equation x1 x2)
               (h_vieta_sum : x1 + x2 = -a) (h_vieta_product : x1 * x2 = 4) :
  a = -10 :=
by
  sorry

end find_a_l510_510912


namespace even_suff_not_nec_l510_510990

theorem even_suff_not_nec (f g : ℝ → ℝ) 
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hg_even : ∀ x : ℝ, g (-x) = g x) :
  ∀ x : ℝ, (f x + g x) = ((f + g) x) ∧ (∀ h : ℝ → ℝ, ∃ f g : ℝ → ℝ, h = f + g ∧ ∀ x : ℝ, (h (-x) = h x) ↔ (f (-x) = f x ∧ g (-x) = g x)) :=
by 
  sorry

end even_suff_not_nec_l510_510990


namespace sin_tan_ineq_l510_510814

noncomputable def triangle_geom_sequence (A B C : ℝ) (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b →
∃ q : ℝ, 
  b = a * q ∧ c = a * q^2 ∧
  (sin A * ((1 / tan A) + (1 / tan B)) = q ∧ 
  (real.sqrt 5 - 1) / 2 < q ∧ q < (real.sqrt 5 + 1) / 2)

theorem sin_tan_ineq (A B C : ℝ) (a b c : ℝ) (h : triangle_geom_sequence A B C a b c) :
  ∃ q : ℝ, 
    sin A * ((1 / tan A) + (1 / tan B)) = q ∧ 
    (real.sqrt 5 - 1) / 2 < q ∧ q < (real.sqrt 5 + 1) / 2 := by
  sorry

end sin_tan_ineq_l510_510814


namespace minimum_value_sum_function_l510_510716

noncomputable def sum_function (x y : ℝ) : ℝ :=
  ∑ i in (Finset.range 10).map (λ n, n + 1), 
  ∑ j in (Finset.range 10).map (λ n, n + 1), 
  ∑ k in (Finset.range 10).map (λ n, n + 1), 
    |(x + y - 10 * i) * (3 * x - 6 * y - 36 * j) * (19 * x + 95 * y - 95 * k)| * k

theorem minimum_value_sum_function : ∃ x y : ℝ, sum_function x y = 2394 * 10^6 :=
by
  use 55, -4
  sorry

end minimum_value_sum_function_l510_510716


namespace range_of_alpha_l510_510724

-- Define the function f
def f (x : ℝ) : ℝ := Real.cos x

-- Define the derivative of the function f
-- Prove that the derivative is -sin(x)
lemma derivative_of_f : ∀ x, deriv f x = -Real.sin x := by
  intro x
  apply deriv_cos

-- Define the condition that k is the tangent of the angle of inclination alpha
def k (x : ℝ) : ℝ := -Real.sin x

-- The main statement proving the range of alpha given the conditions
theorem range_of_alpha : 
  {α | 0 ≤ α ∧ α < π ∧ α ≠ π / 2 ∧ ∃ x, k (x) = Real.tan α} = 
  {α | (0 ≤ α ∧ α ≤ π / 4) ∨ (3 * π / 4 ≤ α ∧ α < π)} := 
by
  intro α
  sorry

end range_of_alpha_l510_510724


namespace correlation_and_regression_l510_510227

noncomputable def mean (xs : List ℝ) : ℝ :=
  (List.sum xs) / (xs.length)

def squared_diff (xs : List ℝ) (mean : ℝ) : ℝ :=
  List.sum (xs.map (λ x => (x - mean)^2))

def covariance (xs ys : List ℝ) (x_mean y_mean : ℝ) : ℝ :=
  List.sum ((List.zipWith (λ x y => (x - x_mean) * (y - y_mean)) xs ys))

def correlation_coefficient (xs ys : List ℝ) : ℝ :=
  let x_mean := mean xs
  let y_mean := mean ys
  covariance xs ys x_mean y_mean / 
    (Real.sqrt (squared_diff xs x_mean * squared_diff ys y_mean))

noncomputable def regression_coefficient_b (xs ys : List ℝ) : ℝ :=
  let x_mean := mean xs
  let y_mean := mean ys
  covariance xs ys x_mean y_mean / squared_diff xs x_mean

noncomputable def regression_coefficient_a (xs ys : List ℝ) : ℝ :=
  let x_mean := mean xs
  let y_mean := mean ys
  y_mean - (regression_coefficient_b xs ys * x_mean)

theorem correlation_and_regression (xs ys : List ℝ) :
  let x_data := [1.0, 2.0, 3.0, 4.0, 5.0]
  let y_data := [9.5, 8.6, 7.8, 7.0, 6.1]
  mean x_data = 3.0 ∧ mean y_data = 7.8 ∧
  squared_diff x_data (mean x_data) = 10.0 ∧
  squared_diff y_data (mean y_data) = 7.06 ∧
  covariance x_data y_data (mean x_data) (mean y_data) = -8.4 ∧
  correlation_coefficient x_data y_data ≈ -1.0 ∧
  regression_coefficient_b x_data y_data ≈ -0.84 ∧
  regression_coefficient_a x_data y_data ≈ 10.32 :=
  sorry

end correlation_and_regression_l510_510227


namespace product_geq_n_minus_1_power_n_l510_510109

theorem product_geq_n_minus_1_power_n {n : ℕ} (x : Fin n → ℝ) (h : ∀ i, 0 < x i) (h_sum : ∑ i, x i = 1) :
  (∏ i, (1 / x i - 1)) ≥ (n - 1) ^ n :=
sorry

end product_geq_n_minus_1_power_n_l510_510109


namespace unique_solution_for_m_l510_510798

theorem unique_solution_for_m (m : ℤ) (h : {3, 4, m^2 - 3 * m - 1} ∩ {2 * m, -3} = {-3}) : m = 1 :=
by
  sorry

end unique_solution_for_m_l510_510798


namespace number_of_distinct_pairs_is_four_l510_510717

noncomputable def num_distinct_pairs : Nat :=
  let pairs := {p : ℝ × ℝ // p.1 = p.1^2 + p.2^2 ∧ p.2 = 2 * p.1 * p.2 + p.2^3}
  pairs.to_finset.card

theorem number_of_distinct_pairs_is_four : num_distinct_pairs = 4 := 
by 
  sorry

end number_of_distinct_pairs_is_four_l510_510717


namespace P_eq_CU_M_union_CU_N_l510_510501

open Set

-- Definitions of U, M, N
def U : Set (ℝ × ℝ) := { p | True }
def M : Set (ℝ × ℝ) := { p | p.2 ≠ p.1 }
def N : Set (ℝ × ℝ) := { p | p.2 ≠ -p.1 }
def CU_M : Set (ℝ × ℝ) := { p | p.2 = p.1 }
def CU_N : Set (ℝ × ℝ) := { p | p.2 = -p.1 }

-- Theorem statement
theorem P_eq_CU_M_union_CU_N :
  { p : ℝ × ℝ | p.2^2 ≠ p.1^2 } = CU_M ∪ CU_N :=
sorry

end P_eq_CU_M_union_CU_N_l510_510501


namespace median_and_variance_l510_510165

-- Define the data set
def data_set : List ℝ := [2, 3, 3, 3, 6, 6, 4, 5]

-- Helper function to calculate median
def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (≤)
  if sorted.length % 2 = 0 then
    (sorted.nth (sorted.length / 2 - 1) + sorted.nth (sorted.length / 2)) / 2
  else
    sorted.nth (sorted.length / 2)

-- Helper function to calculate mean
def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

-- Helper function to calculate variance
def variance (l : List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x => (x - μ)^2)).sum / l.length

-- The theorem stating the median and variance of the given data set
theorem median_and_variance :
  median data_set = 3.5 ∧ variance data_set = 2 :=
by
  -- We state the properties but do not prove them here
  sorry

end median_and_variance_l510_510165


namespace system_of_equations_15_vars_system_of_equations_n_vars_l510_510979

theorem system_of_equations_15_vars :
  (∃ x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 : ℝ,
    1 - x1 * x2 = 0 ∧
    1 - x2 * x3 = 0 ∧
    1 - x3 * x4 = 0 ∧
    1 - x4 * x5 = 0 ∧
    1 - x5 * x6 = 0 ∧
    1 - x6 * x7 = 0 ∧
    1 - x7 * x8 = 0 ∧
    1 - x8 * x9 = 0 ∧
    1 - x9 * x10 = 0 ∧
    1 - x10 * x11 = 0 ∧
    1 - x11 * x12 = 0 ∧
    1 - x12 * x13 = 0 ∧
    1 - x13 * x14 = 0 ∧
    1 - x14 * x15 = 0 ∧
    1 - x15 * x1 = 0) ↔
  (∀ i ∈ {1, 2, 3, ..., 15}, xi = 1 ∨ xi = -1) :=
by sorry

theorem system_of_equations_n_vars (n : ℕ) :
  (∃ x : fin n → ℝ,
    ∀ i : fin n, 1 - x i * x (i + 1 % n.succ) = 0) ↔
  (if n % 2 = 0
  then ∃ a : ℝ, ∀ i: fin n, x i = if i % 2 = 0 then a else a⁻¹
  else ∀ i : fin n, x i = 1 ∨ x i = -1) :=
by sorry

end system_of_equations_15_vars_system_of_equations_n_vars_l510_510979


namespace intersection_point_exists_l510_510261

theorem intersection_point_exists :
  ∃ t u x y : ℚ,
    (x = 2 + 3 * t) ∧ (y = 3 - 4 * t) ∧
    (x = 4 + 5 * u) ∧ (y = -6 + u) ∧
    (x = 175 / 23) ∧ (y = 19 / 23) :=
by
  sorry

end intersection_point_exists_l510_510261


namespace train_time_saved_l510_510654

theorem train_time_saved (s : ℝ) (h : s > 0) : 
    let d := 4 * s in
    let time_half_speed := 8 in
    let time_full_speed := d / s in
    time_half_speed - time_full_speed = 4 :=
by
    have d_def : d = 4 * s := rfl
    have time_half_speed_def : time_half_speed = 8 := rfl
    have time_full_speed_def : time_full_speed = d / s := rfl
    rw [d_def, time_half_speed_def, time_full_speed_def]
    have time_calc : d / s = 4 := by
        simp [d_def]
        field_simp [h]
        linarith
    rw [time_calc]
    norm_num

# Note: The proof is not required. 'sorry' can be used to skip the proof part.

end train_time_saved_l510_510654


namespace new_ratio_first_term_l510_510264

theorem new_ratio_first_term (x : ℕ) (r1 r2 : ℕ) (new_r1 : ℕ) :
  r1 = 4 → r2 = 15 → x = 29 → new_r1 = r1 + x → new_r1 = 33 :=
by
  intros h_r1 h_r2 h_x h_new_r1
  rw [h_r1, h_x] at h_new_r1
  exact h_new_r1

end new_ratio_first_term_l510_510264


namespace triangle_inequality_positive_difference_l510_510833

theorem triangle_inequality_positive_difference (x : ℤ) (h1 : 1 < x) (h2 : x < 11) : 
  let lower_bound := 2 in
  let upper_bound := 10 in
  let greatest_least_difference := upper_bound - lower_bound in
  greatest_least_difference = 8 :=
by
  -- Definitions for the bounds
  let lower_bound := 2
  let upper_bound := 10
  let greatest_least_difference := upper_bound - lower_bound
  
  -- Now we state the goal
  have goal : greatest_least_difference = 8
  
  -- Show the goal is proven
  exact goal

end triangle_inequality_positive_difference_l510_510833


namespace slope_of_line_through_origin_and_center_l510_510603

def Point := (ℝ × ℝ)

def is_center (p : Point) : Prop :=
  p = (3, 1)

def is_dividing_line (l : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, l x = y → y / x = 1 / 3

theorem slope_of_line_through_origin_and_center :
  ∃ l : ℝ → ℝ, (∀ p1 p2 : Point,
  p1 = (0, 0) →
  p2 = (3, 1) →
  is_center p2 →
  is_dividing_line l) :=
sorry

end slope_of_line_through_origin_and_center_l510_510603


namespace odd_function_value_l510_510733

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then -x^2 + x + 1 else if x < 0 then x^2 + x - 1 else 0

theorem odd_function_value (x : ℝ) (h : f x = -f (-x)) :
  (f 0 = 0) ∧ (∀ (x : ℝ), x < 0 → f x = x^2 + x - 1) :=
by
  split
  { 
    -- proof for f(0) = 0
    sorry 
  }
  {
    -- proof for f(x) when x < 0
    intros x hx
    have h1 : f x = x^2 + x - 1,
    sorry
    exact h1,
  }

end odd_function_value_l510_510733


namespace correct_number_of_propositions_l510_510920

def proposition1 : Prop :=
  ∀ x : ℝ, x^2 - 3 * x + 2 = 0 → x = 1

def proposition2 : Prop :=
  ∀ x : ℝ, let y := 1 + 2 * x in x = x + 1 → y = y - 2

def proposition3 : Prop :=
  ∀ p q : Prop, (p = false ∧ q = false) → (p = false ∨ q = false)

def proposition4 : Prop :=
  ∀ x : ℝ, x^2 + x + 1 ≥ 0

def proposition5 : Prop :=
  ∀ ξ : ℝ, P (ξ > 1) = P0 → P (-1 < ξ ∧ ξ < 0) = 1/2 - P0

def num_correct_propositions : ℕ :=
  (if proposition1 then 1 else 0) +
  (if proposition4 then 1 else 0) +
  (if proposition5 then 1 else 0)

theorem correct_number_of_propositions :
  num_correct_propositions = 3 := sorry

end correct_number_of_propositions_l510_510920


namespace part_one_l510_510406

def f (x k : ℝ) : ℝ := abs (2 * x - 1) + abs (2 * x - k)

theorem part_one (k : ℝ) : (∀ x : ℝ, f x k ≥ 1) → (k ≤ 0 ∨ k ≥ 2) :=
begin
  sorry
end

end part_one_l510_510406


namespace largest_median_is_eleven_l510_510900

-- Definitions based on conditions from problem
def known_numbers : List ℕ := [3, 5, 8, 9, 11]
def additional_numbers : List ℕ := [11, 11, 11, 11, 11]

-- Complete list from conditions
def complete_list : List ℕ := known_numbers ++ additional_numbers

-- Condition for median calculation
def median (l : List ℕ) : ℚ :=
  let sorted_list := l.qsort (· ≤ ·)
  (sorted_list.get! 4 + sorted_list.get! 5) / 2

-- Theorem statement based on the question and given conditions
theorem largest_median_is_eleven : median complete_list = 11 := 
  sorry

end largest_median_is_eleven_l510_510900


namespace problem1_problem2_l510_510024

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 + x^2)

theorem problem1 (a b : ℝ) (h : a ≠ b) : abs (f a - f b) < abs (a - b) :=
sorry

theorem problem2 (a b c : ℝ) (h : a + b + c = f (2 * real.sqrt 2)) : a + b + c ≥ a * b + b * c + c * a :=
sorry

end problem1_problem2_l510_510024


namespace vector_parallel_find_x_l510_510767

theorem vector_parallel_find_x (x : ℝ) (λ : ℝ) (a b : ℝ × ℝ)
  (ha : a = (-1, 2))
  (hb : b = (x, 4))
  (h_parallel : a = λ • b) :
  x = -2 :=
by sorry

end vector_parallel_find_x_l510_510767


namespace angles_of_triangle_l510_510091

theorem angles_of_triangle (ABC : Triangle) (A B C : Point) (AD : Line)
  (hAD : isAltitude AD A B C)
  (AE : Line) (hAE : isMedian AE A B C)
  (B E D : Point) (hBEDC : liesOnLine [B, E, D, C] BC)
  (I1 I2 : Point)
  (hI1 : isIncenter I1 (Triangle.mk A B E))
  (hI2 : isIncenter I2 (Triangle.mk A D C))
  (hI1_on_AD : liesOnLine I1 AD)
  (hI2_on_AE : liesOnLine I2 AE) :
  angle A B C = 60 ∧ angle B C A = 30 ∧ angle C A B = 90 :=
by {
  sorry
}

end angles_of_triangle_l510_510091


namespace part1_part2_l510_510366

variables {V : Type*} [inner_product_space ℝ V]

-- Definitions from the conditions
variables (a b : V)
axiom a_norm : ∥a∥ = 4
axiom b_norm : ∥b∥ = 3
axiom a_b_angle : ⟪a, b⟫ = ∥a∥ * ∥b∥ * real.cos (2 * real.pi / 3)

-- Proof problem:
theorem part1 : (2 • a - 3 • b) ⬝ (2 • a + b) = 61 :=
by sorry

theorem part2 : ∥a + b∥ = real.sqrt 13 :=
by sorry

end part1_part2_l510_510366


namespace sheet_length_proof_l510_510631

noncomputable def length_of_sheet (L : ℝ) : ℝ := 48

theorem sheet_length_proof (L : ℝ) (w : ℝ) (s : ℝ) (V : ℝ) (h : ℝ) (new_w : ℝ) :
  w = 36 →
  s = 8 →
  V = 5120 →
  h = s →
  new_w = w - 2 * s →
  V = (L - 2 * s) * new_w * h →
  L = 48 :=
by
  intros hw hs hV hh h_new_w h_volume
  -- conversion of the mathematical equivalent proof problem to Lean's theorem
  sorry

end sheet_length_proof_l510_510631


namespace set_inclusion_interval_l510_510810

theorem set_inclusion_interval (a : ℝ) :
    (A : Set ℝ) = {x : ℝ | (2 * a + 1) ≤ x ∧ x ≤ (3 * a - 5)} →
    (B : Set ℝ) = {x : ℝ | 3 ≤ x ∧ x ≤ 22} →
    (2 * a + 1 ≤ 3 * a - 5) →
    (A ⊆ B ↔ 6 ≤ a ∧ a ≤ 9) :=
by sorry

end set_inclusion_interval_l510_510810


namespace number_of_zeros_800_factorial_l510_510698

theorem number_of_zeros_800_factorial : 
  (⟦800 / 5⟧ + ⟦800 / 25⟧ + ⟦800 / 125⟧ + ⟦800 / 625⟧) = 199 := 
sorry

end number_of_zeros_800_factorial_l510_510698


namespace gcd_1755_1242_l510_510201

theorem gcd_1755_1242 : Nat.gcd 1755 1242 = 27 := 
by
  sorry

end gcd_1755_1242_l510_510201


namespace increasing_interval_l510_510401

noncomputable theory
open Real

-- Defining the conditions
def f (x : ℝ) (ϕ : ℝ) : ℝ := sin (2 * x + ϕ)
def cond1 (ϕ : ℝ) : Prop := abs ϕ < π
def cond2 (ϕ : ℝ) : Prop := ∀ (x : ℝ), f x ϕ ≤ abs (f (π / 6) ϕ)
def cond3 (ϕ : ℝ) : Prop := f (π / 2) ϕ > f π ϕ

-- Proving the increasing interval
theorem increasing_interval (ϕ : ℝ) (x : ℝ) (k : ℤ)
  (h1 : cond1 ϕ)
  (h2 : cond2 ϕ)
  (h3 : cond3 ϕ) :
  x ∈ Icc (k * π + π / 6) (k * π + 2 * π / 3) :=
sorry

end increasing_interval_l510_510401


namespace gcd_459_357_l510_510162

/-- Prove that the greatest common divisor of 459 and 357 is 51. -/
theorem gcd_459_357 : gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l510_510162


namespace sequence_divisible_by_4_l510_510540

theorem sequence_divisible_by_4 {a : ℕ → ℕ} (h₁ : ∀ n, a 1 > 0)
  (h₂ : ∀ n, (a (n + 1) = if a n % 2 = 0 then a n / 2 else 3 * a n + 1)) :
  ∃ n, 4 ∣ a n :=
by
  sorry

end sequence_divisible_by_4_l510_510540


namespace nonneg_diff_between_roots_l510_510586

theorem nonneg_diff_between_roots : 
  ∀ x : ℝ, 
  (x^2 + 42 * x + 384 = 0) → 
  (∃ r1 r2 : ℝ, x = r1 ∨ x = r2) → 
  (abs (r1 - r2) = 8) := 
sorry

end nonneg_diff_between_roots_l510_510586


namespace darrel_receives_27_dollars_after_fee_l510_510696

-- Definitions and conditions
def quarters := 76
def dimes := 85
def nickels := 20
def pennies := 150

def quarter_value := 0.25
def dime_value := 0.10
def nickel_value := 0.05
def penny_value := 0.01

def total_value := (quarters * quarter_value) 
                 + (dimes * dime_value) 
                 + (nickels * nickel_value) 
                 + (pennies * penny_value)

def fee_rate := 0.10
def fee := total_value * fee_rate

def amount_after_fee := total_value - fee

-- Mathematical proof statement
theorem darrel_receives_27_dollars_after_fee : amount_after_fee = 27 :=
by sorry

end darrel_receives_27_dollars_after_fee_l510_510696


namespace complex_quadrant_l510_510458

theorem complex_quadrant (z : ℂ) (h : z = i * (2 - i)) : 1 :=
begin
  sorry
end

end complex_quadrant_l510_510458


namespace simplify_and_evaluate_l510_510520

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  (1 / (x - 1) - 2 / (x ^ 2 - 1)) = -1 := by
  sorry

end simplify_and_evaluate_l510_510520


namespace squares_have_consecutive_digits_generalized_squares_have_many_consecutive_digits_l510_510597

theorem squares_have_consecutive_digits (n : ℕ) (h : ∃ j : ℕ, n = 33330 + j ∧ j < 10) :
    ∃ (a b : ℕ), n ^ 2 / 10 ^ a % 10 = n ^ 2 / 10 ^ (a + 1) % 10 :=
by
  sorry

theorem generalized_squares_have_many_consecutive_digits (k : ℕ) (n : ℕ)
  (h1 : k ≥ 4)
  (h2 : ∃ j : ℕ, n = 33333 * 10 ^ (k - 4) + j ∧ j < 10 ^ (k - 4)) :
    ∃ m, ∃ l : ℕ, ∀ i < m, n^2 / 10 ^ (l + i) % 10 = n^2 / 10 ^ l % 10 :=
by
  sorry

end squares_have_consecutive_digits_generalized_squares_have_many_consecutive_digits_l510_510597


namespace intersection_A_B_l510_510427

open Set

-- Define the sets A and B based on the conditions provided
def A : Set ℤ := { x | x^2 - 2 * x - 8 ≤ 0 }
def B : Set ℤ := { x | log 2 (x : ℝ) > 1 }

-- State the theorem that proves the intersection of A and B equals {3, 4}
theorem intersection_A_B : A ∩ B = { 3, 4 } := by
  sorry

end intersection_A_B_l510_510427


namespace union_of_A_and_B_l510_510784

-- Definitions for sets A and B
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

-- Theorem statement to prove the union of A and B
theorem union_of_A_and_B : A ∪ B = {2, 3, 5, 6} := by
  sorry

end union_of_A_and_B_l510_510784


namespace curve_equation_and_slope_constant_l510_510865

theorem curve_equation_and_slope_constant :
  (∀ (x y : ℝ), x ≥ 0 ∧ ((x - 1)^2 + y^2) = (x^2 + 1) + 1 → y^2 = 4x + 4)
  ∧ 
  (∀ D : ℝ × ℝ, D = (0, 2) → 
    ∀ k : ℝ, k ≠ 0 → 
      let l1 := λ x, k * x + 2,
          l2 := λ x, -k * x + 2,
          M := ( (k - 2)^2 / k^2, l1 ( (k - 2)^2 / k^2 ) ),
          N := ( (k + 2)^2 / k^2, l2 ( (k + 2)^2 / k^2 ) ) in
      ((snd M - snd N) / (fst M - fst N)) = -1) :=
begin
  -- Placeholder for proof
  sorry
end

end curve_equation_and_slope_constant_l510_510865


namespace root_intervals_necessity_l510_510558

theorem root_intervals_necessity (k : ℝ) : 
  (∃ x ∈ set.Icc (−1 : ℝ) 1, k * x - 3 = 0) → (k ≥ 3) ∨ (k ≤ -3) :=
begin
  sorry
end

end root_intervals_necessity_l510_510558


namespace original_fraction_2_7_l510_510443

theorem original_fraction_2_7 (N D : ℚ) : 
  (1.40 * N) / (0.50 * D) = 4 / 5 → N / D = 2 / 7 :=
by
  intro h
  sorry

end original_fraction_2_7_l510_510443


namespace volume_tetrahedron_inscribed_l510_510464

noncomputable def volume_of_KLMN (EF GH EG FH EH FG : ℝ) : ℝ :=
  let V := (1 / 288) * real.sqrt(
    -EF^4 * GH^4 - GH^4 * EG^4 - EG^4 * EF^4 - EF^4 * EG^4 
    + 4 * EF^2 * GH^2 * EG^2) in
  V / 3 -- considering the symmetry factor

theorem volume_tetrahedron_inscribed (EF GH EG FH EH FG : ℝ) 
  (h1 : EF = 7) (h2 : GH = 7) (h3 : EG = 10) 
  (h4 : FH = 10) (h5 : EH = 11) (h6 : FG = 11) :
  abs (volume_of_KLMN EF GH EG FH EH FG - 2.09) < 0.01 :=
by
  sorry

end volume_tetrahedron_inscribed_l510_510464


namespace hyperbola_quadrilateral_area_l510_510027

theorem hyperbola_quadrilateral_area (k : ℝ) (h1 : -16 < k) (h2 : k < 8)
  (h3 : ∀ x y : ℝ, (x ^ 2) / (16 + k) - (y ^ 2) / (8 - k) = 1) 
  (h4 : ∃ x : ℝ, ∃ y : ℝ, y = -√3 * x) :
  3 * 2 * 2 * √6 = 12 * √6 :=
by
  sorry

end hyperbola_quadrilateral_area_l510_510027


namespace fault_line_total_movement_l510_510300

theorem fault_line_total_movement (a b : ℝ) (h1 : a = 1.25) (h2 : b = 5.25) : a + b = 6.50 := by
  -- Definitions:
  rw [h1, h2]
  -- Proof:
  sorry

end fault_line_total_movement_l510_510300


namespace speed_of_second_train_l510_510953

theorem speed_of_second_train(
  length_first_train length_second_train : ℝ,
  speed_first_train time_to_cross : ℝ,
  (h1 : length_first_train = 140),
  (h2 : length_second_train = 160),
  (h3 : speed_first_train = 60),
  (h4 : time_to_cross = 9.99920006399488)
) : ∃ speed_second_train : ℝ, speed_second_train = 48.00287997120036 :=
by 
  have total_length : ℝ := length_first_train + length_second_train
  have relative_speed : ℝ := total_length / time_to_cross * 3.6
  have speed_second_train : ℝ := relative_speed - speed_first_train
  exact ⟨speed_second_train, sorry⟩

end speed_of_second_train_l510_510953


namespace solution_exists_l510_510331

noncomputable def find_A_and_B : Prop :=
  ∃ A B : ℚ, 
    (A, B) = (75 / 16, 21 / 16) ∧ 
    ∀ x : ℚ, x ≠ 12 ∧ x ≠ -4 → 
    (6 * x + 3) / ((x - 12) * (x + 4)) = A / (x - 12) + B / (x + 4)

theorem solution_exists : find_A_and_B :=
sorry

end solution_exists_l510_510331


namespace max_value_of_f_l510_510345

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + 4 * (Real.cos x)

theorem max_value_of_f : ∃ x : ℝ, f x ≤ 4 :=
sorry

end max_value_of_f_l510_510345


namespace angle_of_inclination_l510_510009

noncomputable def angle := (Real.pi / 6) -- we choose an arbitrary value such that it doesn't assume solution steps

theorem angle_of_inclination:
  (θ : ℝ) 
  (tan_θ : ℝ) 
  (H : tan_θ = -2) 
  : (sin θ + cos θ) / (sin θ - cos θ) = (1 : ℝ) / 3 :=
begin
  sorry
end

end angle_of_inclination_l510_510009


namespace wyatt_undamaged_envelopes_l510_510231

theorem wyatt_undamaged_envelopes : 
  let blue_envelopes := 25 in
  let yellow_envelopes := blue_envelopes - 8 in
  let red_envelopes := yellow_envelopes + 5 in
  let damaged_blue := (0.05 * blue_envelopes).to_nat in
  let damaged_red := ((2 / 3) * red_envelopes).nat_floor in
  let found_green := 3 in
  let undamaged_blue := blue_envelopes - damaged_blue in
  let undamaged_yellow := yellow_envelopes in
  let undamaged_red := red_envelopes - damaged_red in
  let undamaged_green := found_green in
  undamaged_blue + undamaged_yellow + undamaged_red + undamaged_green = 52 :=
by
  let blue_envelopes := 25
  let yellow_envelopes := blue_envelopes - 8
  let red_envelopes := yellow_envelopes + 5
  let damaged_blue := (0.05 * blue_envelopes).to_nat
  let damaged_red := ((2 / 3) * red_envelopes).nat_floor
  let found_green := 3
  let undamaged_blue := blue_envelopes - damaged_blue
  let undamaged_yellow := yellow_envelopes
  let undamaged_red := red_envelopes - damaged_red
  let undamaged_green := found_green
  linarith

end wyatt_undamaged_envelopes_l510_510231


namespace externally_tangent_circles_solution_l510_510907

theorem externally_tangent_circles_solution (R1 R2 d : Real)
  (h1 : R1 > 0) (h2 : R2 > 0) (h3 : R1 + R2 > d) :
  (1/R1) + (1/R2) = 2/d :=
sorry

end externally_tangent_circles_solution_l510_510907


namespace set_inclusion_interval_l510_510811

theorem set_inclusion_interval (a : ℝ) :
    (A : Set ℝ) = {x : ℝ | (2 * a + 1) ≤ x ∧ x ≤ (3 * a - 5)} →
    (B : Set ℝ) = {x : ℝ | 3 ≤ x ∧ x ≤ 22} →
    (2 * a + 1 ≤ 3 * a - 5) →
    (A ⊆ B ↔ 6 ≤ a ∧ a ≤ 9) :=
by sorry

end set_inclusion_interval_l510_510811


namespace smaller_number_of_product_l510_510553

theorem smaller_number_of_product :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 5610 ∧ a = 34 :=
by
  -- Proof would go here
  sorry

end smaller_number_of_product_l510_510553


namespace characterize_function_l510_510987

namespace Polynomial

open Polynomial

noncomputable def valid_function (f : Polynomial ℤ → Polynomial ℤ) : Prop :=
∀ P Q : Polynomial ℤ, ∀ r : ℤ, (P.eval r ∣ Q.eval r ↔ (f P).eval r ∣ (f Q).eval r)

theorem characterize_function (f : Polynomial ℤ → Polynomial ℤ) :
  valid_function f ↔ ∃ (A : Polynomial ℤ) (d : ℕ), d > 0 ∧ A ≠ 0 ∧
  (∀ x : ℤ, A.eval x ≠ 0) ∧ 
  ∀ P : Polynomial ℤ, f P = C (int.sign (P.leading_coeff)) * A * P^d :=
sorry

end Polynomial

end characterize_function_l510_510987


namespace polar_coordinate_equations_and_area_l510_510029

noncomputable def line_l_parametric : ℝ → (ℝ × ℝ) :=
λ t, (0.5 * t, (sqrt 3 / 2) * t)

noncomputable def curve_C_parametric : ℝ → (ℝ × ℝ) :=
λ θ, (1 + 2 * cos θ, 2 * sqrt 3 + 2 * sin θ)

def point_P_polar := (2 * sqrt 3, 2 * π / 3)

theorem polar_coordinate_equations_and_area :
  (∀ θ : ℝ, let ρ := 2 in (θ = π / 3 ∧ 
  ρ^2 - 2*ρ*cos θ - 4*sqrt 3*ρ*sin θ + 9 = 0))
  ∧
  (∀ (A B : ℝ × ℝ), 
   let ρ1 := 3 + sqrt 13 in 
   let ρ2 := 3 - sqrt 13 in 
   let |AB| := sqrt ((ρ1 + ρ2) ^ 2 - 4 * ρ1 * ρ2) in
   let d := 2 * sqrt 3 * sin (π / 3) in
   let S := (1 / 2) * |AB| * d in
   S = 3 * sqrt 13 / 2) :=
sorry

end polar_coordinate_equations_and_area_l510_510029


namespace sequence_unique_remainders_l510_510435

theorem sequence_unique_remainders (n : ℕ) (h_pos : n > 0)
  (a : Fin n → ℤ)
  (h_distinct_mod_n : Function.Injective (λ i, a i % n)) :
  ∀ m : ℤ, m ∈ Finset.image (λ i, a i % n) (Finset.univ : Finset (Fin n)) ↔ ∃ i, m = a i % n :=
by {
  sorry
}

end sequence_unique_remainders_l510_510435


namespace max_constant_ineq_l510_510341

theorem max_constant_ineq (a b c d : ℝ) (h : a ∈ set.Icc 0 1) (h1 : b ∈ set.Icc 0 1) (h2 : c ∈ set.Icc 0 1) (h3 : d ∈ set.Icc 0 1) : 
  a^2 * b + b^2 * c + c^2 * d + d^2 * a + 4 ≥ 2 * (a^3 + b^3 + c^3 + d^3) :=
sorry

end max_constant_ineq_l510_510341


namespace unique_zero_iff_a_eq_half_l510_510404

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + a * (Real.exp (x - 1) + Real.exp (1 - x))

theorem unique_zero_iff_a_eq_half :
  (∃! x : ℝ, f x a = 0) ↔ a = 1 / 2 :=
by
  sorry

end unique_zero_iff_a_eq_half_l510_510404


namespace find_n_l510_510362

theorem find_n (n : ℕ) (h : (nat.choose (n + 1) 7) - (nat.choose n 7) = nat.choose n 8) : n = 14 :=
sorry

end find_n_l510_510362


namespace proof_problem_l510_510756

noncomputable def a_sequence (d : ℤ) (a1 : ℤ) (n : ℕ) : ℤ := a1 + d * (n - 1)
def s_sequence (a1 d : ℤ) : ℕ → ℤ
| 0        := 0
| (n + 1) := s_sequence a1 d n + a_sequence d a1 (n + 1)

def b_sequence (n : ℕ) : ℤ := 3^n

def c_sequence (a : ℕ → ℤ) (b : ℕ → ℤ) (n : ℕ) : ℚ :=
(a n * a n + 8 * Real.log (b n) (3) : ℚ) / ((a (n + 1) * b n) : ℚ)

def m_sequence (c : ℕ → ℚ) : ℕ → ℚ
| 0        := 0
| (n + 1) := m_sequence c n + c (n + 1)

theorem proof_problem : 
  (∀ a1 d, s_sequence a1 d 3 = 9 → 
           (a1 + d) * (a1 + 13*d) = (a1 + 4*d)^2 → 
           a_sequence d a1 = λ n, 2 * n - 1) ∧
  (T : ℤ → ℤ := λ n, (3^(n + 1) - 3)/2, 
           b_sequence = λ n, 3^n) ∧
  (c_sequence (λ n, 2 * n - 1) b_sequence = λ n, (ℚ.ofNat (2 * n + 1)) / (3^n)) ∧
  (∀ n, m_sequence (λ n, (ℚ.ofNat (2 * n + 1)) / (3^n)) n = (5/2 : ℚ) - ((2 * n + 7) : ℚ) / (2 * 3^n)) :=
  sorry

end proof_problem_l510_510756


namespace jackson_pbj_sandwiches_l510_510842

-- The number of Wednesdays and Fridays in the 36-week school year
def total_weeks : ℕ := 36
def total_wednesdays : ℕ := total_weeks
def total_fridays : ℕ := total_weeks

-- Public holidays on Wednesdays and Fridays
def holidays_wednesdays : ℕ := 2
def holidays_fridays : ℕ := 3

-- Days Jackson missed
def missed_wednesdays : ℕ := 1
def missed_fridays : ℕ := 2

-- Number of times Jackson asks for a ham and cheese sandwich every 4 weeks
def weeks_for_ham_and_cheese : ℕ := total_weeks / 4

-- Number of ham and cheese sandwich days
def ham_and_cheese_wednesdays : ℕ := weeks_for_ham_and_cheese
def ham_and_cheese_fridays : ℕ := weeks_for_ham_and_cheese * 2

-- Remaining days for peanut butter and jelly sandwiches
def remaining_wednesdays : ℕ := total_wednesdays - holidays_wednesdays - missed_wednesdays
def remaining_fridays : ℕ := total_fridays - holidays_fridays - missed_fridays

def pbj_wednesdays : ℕ := remaining_wednesdays - ham_and_cheese_wednesdays
def pbj_fridays : ℕ := remaining_fridays - ham_and_cheese_fridays

-- Total peanut butter and jelly sandwiches
def total_pbj : ℕ := pbj_wednesdays + pbj_fridays

theorem jackson_pbj_sandwiches : total_pbj = 37 := by
  -- We don't require the proof steps, just the statement
  sorry

end jackson_pbj_sandwiches_l510_510842


namespace zero_point_of_g_l510_510773

-- Define the function f.
def f (x : ℝ) : ℝ := (x - 1) / x

-- Define the function g using f.
def g (x : ℝ) : ℝ := f (4 * x) - x

-- State the theorem to prove g(x) = 0 at x = 1 / 2
theorem zero_point_of_g : g (1 / 2) = 0 :=
by
  -- Proof would go here.
  sorry

end zero_point_of_g_l510_510773


namespace net_income_correct_l510_510196

-- Definition of income before tax
def total_income_before_tax : ℝ := 45000

-- Definition of tax rate
def tax_rate : ℝ := 0.13

-- Definition of tax amount
def tax_amount : ℝ := tax_rate * total_income_before_tax

-- Definition of net income after tax
def net_income_after_tax : ℝ := total_income_before_tax - tax_amount

-- Theorem statement
theorem net_income_correct : net_income_after_tax = 39150 := by
  sorry

end net_income_correct_l510_510196


namespace integral_value_l510_510689

theorem integral_value : ∫ x in 0..(Real.pi / 2), (3 * x + Real.sin x) = (3 * Real.pi^2 / 8) + 1 :=
by
  sorry

end integral_value_l510_510689


namespace triangle_side_length_AB_l510_510834

theorem triangle_side_length_AB (A B C : ℝ) (AC AB : ℝ) (hA : A = 105) (hB : B = 45) (hAC : AC = 2)
  (hSumAngles : A + B + C = 180) (hSinRule : AC / real.sin (B * real.pi / 180) = AB / real.sin (C * real.pi / 180)) :
  AB = real.sqrt 2 :=
by sorry

end triangle_side_length_AB_l510_510834


namespace pipe_probability_l510_510633

theorem pipe_probability :
  let total_length : ℝ := 100
  let area_total := (total_length^2) / 2
  let sub_region_area := (75^2) / 2
  (sub_region_area / area_total) = 9 / 16 :=
by
  let total_length : ℝ := 100
  let area_total := (total_length^2) / 2
  let sub_region_area := (75^2) / 2
  have h₁ : sub_region_area / area_total = (75^2) / 10000 := by { sorry }
  have h₂ : (75^2) / 10000 = 9 / 16 := by { sorry }
  rw [h₁, h₂]
  exact rfl

end pipe_probability_l510_510633


namespace midpoint_G_IF_l510_510755

open EuclideanGeometry

variables {A B C D I E F G : Point}
variables {ω : Circle}

-- Circumcircle of triangle ABC
axiom h1 : ω.center ∈ circle (A, B, C)

-- I and E are the incenter and an excenter of triangle ABD
axiom h2 : is_incenter I (triangle A B D)
axiom h3 : is_excenter E (triangle A B D)

-- External angle bisector of ∠BAC intersects extended line BC at D
axiom h4 : ∃ D, is_external_bisector (line A B) (line A C) (line B C) D

-- Line IF is perpendicular to DE at F
axiom h5 : perpendicular (line I F) (line D E) ∧ incident F (line I F) ∧ incident F (line D E)

-- IF intersects the circumcircle O at G
axiom h6 : ∃ G, incidents G (line I F) ∧ incidents G ω

theorem midpoint_G_IF : is_midpoint G I F :=
sorry

end midpoint_G_IF_l510_510755


namespace hamburgers_purchased_by_second_group_l510_510729

theorem hamburgers_purchased_by_second_group
  (goal_amount : ℕ)
  (hamburger_price : ℕ)
  (first_group_purchased : ℕ)
  (additional_needed : ℕ)
  (total_goal : ℕ) :
  goal_amount = 50 →
  hamburger_price = 5 →
  first_group_purchased = 4 →
  additional_needed = 4 →
  total_goal = 10 →
  ∃ second_group_purchased : ℕ, second_group_purchased = 2 :=
by
  intros
  use 2
  sorry

end hamburgers_purchased_by_second_group_l510_510729


namespace complement_union_A_B_complement_A_intersection_B_l510_510786

open Set

-- Definitions of A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Proving the complement of A ∪ B
theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | x ≤ 2 ∨ 10 ≤ x} :=
by sorry

-- Proving the intersection of the complement of A with B
theorem complement_A_intersection_B : (Aᶜ ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
by sorry

end complement_union_A_B_complement_A_intersection_B_l510_510786


namespace find_f3_l510_510763

-- Define g(x) function
def g (x : ℝ) : ℝ := Real.log (x^2 + x + 2) / Real.log 2

-- Given symmetry condition as stated in the problem
def symmetric (f g : ℝ → ℝ) (x : ℝ) (k : ℝ) : Prop :=
  ∀ a : ℝ, f (k - a) = g (k + a)

-- Prove that f(3) = 2 given the symmetry condition
theorem find_f3 (f : ℝ → ℝ) :
  symmetric f g 2 → f 3 = 2 :=
by
  intros h
  specialize h (-1)
  simp [g, Real.log, -Real.log] at h
  sorry

end find_f3_l510_510763


namespace find_FH_l510_510825

theorem find_FH (x : ℤ) : 
  (13 < x ∧ x < 20) → 
  (EF = 7 ∧ FG = 13 ∧ GH = 7 ∧ HE = 20 ∧ EH = ∃ n : ℤ, n) → 
  (x = 14 ∨ x = 15 ∨ x = 16 ∨ x = 17 ∨ x = 18 ∨ x = 19) := 
by
  intro h1 h2
  cases h1 with h1a h1b
  cases h2 with h2a h2b
  sorry -- proof to be provided

end find_FH_l510_510825


namespace right_triangle_cosine_l510_510448

theorem right_triangle_cosine (XY XZ YZ : ℝ) (hXY_pos : XY > 0) (hXZ_pos : XZ > 0) (hYZ_pos : YZ > 0)
  (angle_XYZ : angle_1 = 90) (tan_Z : XY / XZ = 5 / 12) : (XZ / YZ = 12 / 13) :=
by
  sorry

end right_triangle_cosine_l510_510448


namespace length_of_common_chord_l510_510391

noncomputable def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
def commonChordLength : ℝ := 2 * Real.sqrt 5

theorem length_of_common_chord (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : 
  ∃ (l : ℝ), l = commonChordLength := 
sorry

end length_of_common_chord_l510_510391


namespace hendrix_class_students_l510_510214

theorem hendrix_class_students (initial_students new_students : ℕ) (transferred_fraction : ℚ) 
  (h1 : initial_students = 160) (h2 : new_students = 20) (h3 : transferred_fraction = 1/3) :
  let total_students := initial_students + new_students
  let transferred_students := total_students * transferred_fraction
  let remaining_students := total_students - transferred_students
  remaining_students = 120 :=
by
  let total_students := initial_students + new_students
  let transferred_students := total_students * transferred_fraction
  let remaining_students := total_students - transferred_students
  have h4 : total_students = 180, by sorry
  have h5 : transferred_students = 60, by sorry
  have h6 : remaining_students = 120, by sorry
  exact h6

end hendrix_class_students_l510_510214


namespace problem_given_conditions_l510_510662

noncomputable def tan_deg (deg: ℝ) : ℝ := Real.tan (deg * Real.pi / 180)

theorem problem_given_conditions:
  (sin 15 * cos 15 ≠ 1/2) ∧ 
  ((cos (π / 12))^2 - (sin (π / 12))^2 ≠ 1/2) ∧ 
  (sqrt ((1 + cos (π / 6)) / 2) ≠ 1/2) →
  tan_deg 22.5 / (1 - (tan_deg 22.5)^2) = 1/2 := by
  sorry

end problem_given_conditions_l510_510662


namespace perpendicular_line_to_plane_l510_510858

variables {m n : Line} {α : Plane}

theorem perpendicular_line_to_plane (h1 : m ⊥ α) (h2 : m ∥ n) : n ⊥ α :=
sorry

end perpendicular_line_to_plane_l510_510858


namespace integral_f_equals_l510_510772

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 else Real.sqrt (2 - x^2)

-- Define the integral from -1 to sqrt(2) of the function f
noncomputable def integral_f : ℝ :=
  ∫ x in -1..Real.sqrt 2, f x

-- Theorem statement asserting the value of the integral
theorem integral_f_equals : integral_f = (Real.pi / 2) + (1 / 3) :=
  sorry

end integral_f_equals_l510_510772


namespace rent_spent_amount_l510_510292

variable (S R : ℝ) -- Mr. Kishore's monthly salary and rent spent
variable (E : ℝ) -- Total expenses excluding rent

-- Conditions provided in the problem
def saved_10_percent_of_salary (S : ℝ) : Prop := 0.10 * S = 2350
def total_excluding_rent : Prop := E = 1500 + 4500 + 2500 + 2000 + 5650
def net_salary_minus_expenses (S E R: ℝ) : Prop := S - 2350 - E = R

-- Theorem that we need to prove based on the given conditions
theorem rent_spent_amount (S : ℝ) (h1 : saved_10_percent_of_salary S) (h2 : total_excluding_rent) : R = 4850 :=
by
  have h3 : net_salary_minus_expenses S E R := sorry
  -- Proof logic goes here (to be filled)

  -- Final assertion to match the expected rent
  sorry

end rent_spent_amount_l510_510292


namespace max_red_beads_l510_510273

theorem max_red_beads (n : ℕ)
  (total_beads : n = 150)
  (at_least_one_green_in_every_six : ∀ (s : ℕ), s < n - 5 → (∃ i, s ≤ i ∧ i < s + 6 ∧ green i))
  (at_least_one_blue_in_every_eleven : ∀ (s : ℕ), s < n - 10 → (∃ i, s ≤ i ∧ i < s + 11 ∧ blue i)) :
  ∃ red_count, red_count ≤ 112 :=
by sorry

end max_red_beads_l510_510273


namespace find_a_l510_510046

-- Definition of a point
structure Point :=
  (x : ℝ) (y : ℝ)

-- Equation of the circle
def circle_eq (p : Point) : Prop :=
  p.x^2 + p.y^2 + 2 * p.x - 4 * p.y = 0

-- Equation of the line
def line_eq (p : Point) (a : ℝ) : Prop :=
  3 * p.x + p.y + a = 0

-- The center of the circle, calculated by completing the square
def center_of_circle : Point := ⟨ -1, 2 ⟩

-- The proof task
theorem find_a : ∃ (a : ℝ), line_eq center_of_circle a :=
begin
  use 1,
  unfold line_eq center_of_circle,
  sorry,
end

end find_a_l510_510046


namespace bx_squared_l510_510487

open Isosceles
open Triangle

-- Definitions based on the problem conditions
structure Triangle :=
(A B C : Point)
(M : Point) (is_midpoint_M : midpoint A C M)
(N : Point) (is_bisector_CN : bisector C A B N)
(X : Point) (intersection_BM_CN : intersectsLineSegment B M X ∧ intersectsLineSegment C N X)

-- Defining the triangle with given conditions
def triangle_data : Triangle := {
  A := (0, 0),
  B := (0, sqrt 3 / 2),
  C := (4, 0),
  M := (2, 0),
  N := (-1 / 2, 0),
  X := (1 / 2, 0),
  is_midpoint_M := sorry,
  is_bisector_CN := sorry,
  intersection_BM_CN := sorry,
}

-- The main theorem to prove
theorem bx_squared (T : Triangle) (h_iso : isosceles T.B T.X T.N 1 1) : 
  length_squared T.B T.X = 1 := 
sorry

end bx_squared_l510_510487


namespace num_students_excelling_in_both_l510_510682

namespace ProofProblem

variables {α : Type} (students : Finset α)
variables (excelsInChinese excelsInMath : Finset α)

def total_students : ℕ := 45
def students_excelling_in_chinese : ℕ := 34
def students_excelling_in_math : ℕ := 39
def condition (s : Finset α) : Prop := s.card = total_students ∧
                                       excelsInChinese.card = students_excelling_in_chinese ∧
                                       excelsInMath.card = students_excelling_in_math ∧
                                       ∀ x, x ∈ s → x ∈ excelsInChinese ∨ x ∈ excelsInMath

theorem num_students_excelling_in_both
    (h : condition students) : 
    (excelsInChinese ∩ excelsInMath).card = 28 :=
sorry

end ProofProblem

end num_students_excelling_in_both_l510_510682


namespace trapezoid_concurrent_l510_510252

def is_trapezoid (A B C D : Point) : Prop :=
  ∃ AB_parallel_CD : line_parallel (line_through A B) (line_through C D),
  trapezoid A B C D

def equilateral_triangle (A B C : Point) : Prop :=
  triangle A B C ∧ ∀ a b c, dist a b = dist b c ∧ dist b c = dist c a

noncomputable def lines_concurrent (A B C D E F : Point) : Prop :=
  ∃ I : Point, line_contains (line_through A C) I ∧ line_contains (line_through B D) I ∧ line_contains (line_through E F) I

theorem trapezoid_concurrent (A B C D E F : Point)
  (h1 : is_trapezoid A B C D)
  (h2 : equilateral_triangle A B E)
  (h3 : equilateral_triangle C D F) :
  lines_concurrent A C B D E F :=
by
  sorry

end trapezoid_concurrent_l510_510252


namespace find_a_n_l510_510556

-- Definitions of the sequence of sums and relations between them
def S : ℕ → ℕ
| 0       := 0
| 1       := 1
| 2       := 2
| (n + 1) := 3 * (S n) - 2 * (S (n - 1))

axiom S1 : S 1 = 1
axiom S2 : S 2 = 2
axiom recurrence_relation (n : ℕ) : n ≥ 2 → S (n + 1) - 3 * S n + 2 * S (n - 1) = 0

-- Definition of the sequence a_n
def a : ℕ → ℕ
| 1       := 1
| (n + 2) := 2 * (a (n + 1))
| _       := 0  -- For cases where n = 0, which is not typical for this problem

-- Define the required theorem
theorem find_a_n (n : ℕ) : 
  a n = 
  match n with
  | 1     := 1
  | (n + 2) := 2^(n)  -- based on a_{n+1} = 2 * a_n, so recursive times leads to a geometric progression
  end :=
sorry  -- Proof omitted

end find_a_n_l510_510556


namespace total_packs_l510_510875

theorem total_packs (cards_bought : ℕ) (cards_per_pack : ℕ) (num_people : ℕ)
  (h1 : cards_bought = 540) (h2 : cards_per_pack = 20) (h3 : num_people = 4) :
  (cards_bought / cards_per_pack) * num_people = 108 :=
by
  sorry

end total_packs_l510_510875


namespace investment_years_l510_510338

noncomputable def P : ℝ := 10000
noncomputable def r : ℝ := 0.04
noncomputable def n : ℝ := 2
noncomputable def CI : ℝ := 824.32

def compound_interest_years (A P r n CI : ℝ) (t : ℝ) : Prop :=
  A = P * (1 + r / n)^(n * t)

theorem investment_years : 
  ∃ (t : ℝ), 
  let A := P + CI in compound_interest_years A P r n CI t ∧ abs (t - 2) < 0.0001 :=
by
  sorry

end investment_years_l510_510338


namespace average_test_score_before_dropping_l510_510471

theorem average_test_score_before_dropping (A B C : ℝ) :
  (A + B + C) / 3 = 40 → (A + B + C + 20) / 4 = 35 :=
by
  intros h
  sorry

end average_test_score_before_dropping_l510_510471


namespace Hillary_reading_time_on_sunday_l510_510035

-- Define the assigned reading times for both books
def assigned_time_book_a : ℕ := 60 -- minutes
def assigned_time_book_b : ℕ := 45 -- minutes

-- Define the reading times already spent on each book
def time_spent_friday_book_a : ℕ := 16 -- minutes
def time_spent_saturday_book_a : ℕ := 28 -- minutes
def time_spent_saturday_book_b : ℕ := 15 -- minutes

-- Calculate the total time already read for each book
def total_time_read_book_a : ℕ := time_spent_friday_book_a + time_spent_saturday_book_a
def total_time_read_book_b : ℕ := time_spent_saturday_book_b

-- Calculate the remaining time needed for each book
def remaining_time_book_a : ℕ := assigned_time_book_a - total_time_read_book_a
def remaining_time_book_b : ℕ := assigned_time_book_b - total_time_read_book_b

-- Calculate the total remaining time and the equal time division
def total_remaining_time : ℕ := remaining_time_book_a + remaining_time_book_b
def equal_time_division : ℕ := total_remaining_time / 2

-- Theorem statement to prove Hillary's reading time for each book on Sunday
theorem Hillary_reading_time_on_sunday : equal_time_division = 23 := by
  sorry

end Hillary_reading_time_on_sunday_l510_510035


namespace correct_statement_proof_l510_510224

noncomputable def probability_correct_statement (trial_count : ℕ) (frequency : ℝ) (probability : ℝ) : Prop :=
  ∀ (n : ℕ), n > trial_count →
  abs (frequency - probability) < ε

theorem correct_statement_proof :
  let trial_count := arbitrary ℕ,
      frequency := arbitrary ℝ,
      probability := arbitrary ℝ in
  ∀ (n : ℕ), n > trial_count →
  abs (frequency - probability) < ε :=
sorry

end correct_statement_proof_l510_510224


namespace circle_chord_angle_relationship_l510_510053

theorem circle_chord_angle_relationship (O A B C D : Point) (r : ℝ)
  (circle : Circle O r)
  (hAB : LineSegment A B)
  (hBC : LineSegment B C) (hBC_eq_r : hBC.length = r)
  (hCO : LineSegment C O)
  (hCO_extends_to_D : Line C O = Line C D)
  (hAO : Line A O)
  (x y : ℝ)
  (h_x_y_relation : ∃ α, α = 2 * y ∧ x = α + y)
  : x = 3 * y :=
sorry

end circle_chord_angle_relationship_l510_510053


namespace g_of_neg3_l510_510541

def g (x : ℝ) : ℝ := x^2 + 2 * x

theorem g_of_neg3 : g (-3) = 3 :=
by
  sorry

end g_of_neg3_l510_510541


namespace jacoby_lottery_expense_l510_510843

-- Definitions based on the conditions:
def jacoby_trip_fund_needed : ℕ := 5000
def jacoby_hourly_wage : ℕ := 20
def jacoby_work_hours : ℕ := 10
def cookies_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_winnings : ℕ := 500
def sister_gift : ℕ := 500
def num_sisters : ℕ := 2
def money_still_needed : ℕ := 3214

-- The statement to prove:
theorem jacoby_lottery_expense : 
  (jacoby_hourly_wage * jacoby_work_hours) + (cookies_price * cookies_sold) +
  lottery_winnings + (sister_gift * num_sisters) 
  - (jacoby_trip_fund_needed - money_still_needed) = 10 :=
by {
  sorry
}

end jacoby_lottery_expense_l510_510843


namespace height_of_C_l510_510531

noncomputable def height_A_B_C (h_A h_B h_C : ℝ) : Prop := 
  (h_A + h_B + h_C) / 3 = 143 ∧ 
  h_A + 4.5 = (h_B + h_C) / 2 ∧ 
  h_B = h_C + 3

theorem height_of_C (h_A h_B h_C : ℝ) (h : height_A_B_C h_A h_B h_C) : h_C = 143 :=
  sorry

end height_of_C_l510_510531


namespace max_value_of_3a_plus_4b_plus_5c_l510_510104

theorem max_value_of_3a_plus_4b_plus_5c (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) : (3 * a + 4 * b + 5 * c) ≤ sqrt 6 :=
sorry

end max_value_of_3a_plus_4b_plus_5c_l510_510104


namespace problem1_problem2_l510_510310

theorem problem1 (h1 : real.sqrt 12 = 2 * real.sqrt 3)
                 (h2 : abs (1 - real.sqrt 3) = 0)
                 (h3 : (real.pi - 3)^0 = 1) :
  real.sqrt 12 + abs (1 - real.sqrt 3) - (real.pi - 3)^0 = 2 * real.sqrt 3 - 1 :=
by 
  sorry

theorem problem2 (h1 : real.sqrt 18 = 3 * real.sqrt 2)
                 (h2 : (real.sqrt 2 - 1)^2 = 2 - 2 * real.sqrt 2 + 1) :
  (real.sqrt 18 + 2) / real.sqrt 2 - (real.sqrt 2 - 1)^2 = 3 * real.sqrt 2 :=
by 
  sorry

end problem1_problem2_l510_510310


namespace fiber_length_related_to_soil_environment_l510_510064

theorem fiber_length_related_to_soil_environment (
  (fieldA_counts : List ℕ) (fieldB_counts : List ℕ) 
  (A_total: fieldA_counts.length = 5) (B_total: fieldB_counts.length = 5)
  (total_fibers : 40 + 40 = 80)
  (X2_formula : ∀ (a b c d n : ℕ), (n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d)))) :
  let a := fieldA_counts[3] + fieldA_counts[4] in
  let b := fieldA_counts[0] + fieldA_counts[1] + fieldA_counts[2] in
  let c := fieldB_counts[3] + fieldB_counts[4] in
  let d := fieldB_counts[0] + fieldB_counts[1] + fieldB_counts[2] in
  let n := 80 in
  let X2 := X2_formula a b c d n in
  X2 > 6.635 → 
  "The fiber length is related to the soil environment." :=
by
  intros
  have a : ℕ := fieldA_counts[3] + fieldA_counts[4]
  have b : ℕ := fieldA_counts[0] + fieldA_counts[1] + fieldA_counts[2]
  have c : ℕ := fieldB_counts[3] + fieldB_counts[4]
  have d : ℕ := fieldB_counts[0] + fieldB_counts[1] + fieldB_counts[2]
  have n : ℕ := 80
  let X2 := n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))
  linarith
  sorry

end fiber_length_related_to_soil_environment_l510_510064


namespace nonneg_diff_roots_l510_510584

theorem nonneg_diff_roots : 
  ∀ (a b c : ℤ), a = 1 → b = 42 → c = 384 → 
  let Δ = b * b - 4 * a * c in
  Δ ≥ 0 → 
  let r1 := (-b + Δ.sqrt) / (2 * a) in
  let r2 := (-b - Δ.sqrt) / (2 * a) in
  abs (r1 - r2) = 8 :=
by
  sorry

end nonneg_diff_roots_l510_510584


namespace range_of_a_l510_510805

noncomputable def f (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 2, (a - 1 / x) ≥ 0) ↔ (a ≥ 1 / 2) :=
by
  sorry

end range_of_a_l510_510805


namespace impossible_arrangement_l510_510838

theorem impossible_arrangement :
  ¬ ∃ (f : Fin 600 → Fin 600 → Int),
    (∀ i j, abs (f i j) = 1) ∧
    abs (Finset.sum (Finset.univ.image (λ ⟨i, j⟩, f i j))) < 90000 ∧
    (∀ (i j : Fin 597),
      abs (Finset.sum (Finset.range 4).bind (λ i', Finset.range 6).image (λ j', f (i + i') (j + j'))) > 4 ∧
      abs (Finset.sum (Finset.range 6).bind (λ i', Finset.range 4).image (λ j', f (i + i') (j + j'))) > 4) :=
sorry

end impossible_arrangement_l510_510838


namespace cube_partition_l510_510625

theorem cube_partition (N : ℕ) :
  (∃ (cubes : list ℕ), 
    (∀ x ∈ cubes, x ∈ [1, 2, 3] ∧ ∃ (y : ℕ), x = y^3) ∧
    cubes.sum = 64 ∧
    list.card cubes = N ∧
    (∃ x₁ x₂ ∈ cubes, x₁ ≠ x₂)
  ) → N = 10 :=
by
  sorry

end cube_partition_l510_510625


namespace sum_of_possible_values_of_a_6_l510_510732

theorem sum_of_possible_values_of_a_6 {a : ℤ} (h_prime : nat.prime (abs (4 * a ^ 2 - 12 * a - 27))) :
  (a = -1 ∨ a = -2 ∨ a = 4 ∨ a = 5) → ( -1 + -2 + 4 + (5 : ℤ) = 6) :=
sorry

end sum_of_possible_values_of_a_6_l510_510732


namespace max_lamps_on_road_l510_510640

theorem max_lamps_on_road (k: ℕ) (lk: ℕ): 
  lk = 1000 → (∀ n: ℕ, n < k → n≥ 1 ∧ ∀ m: ℕ, if m > n then m > 1 else true) → (lk ≤ k) ∧ 
  (∀ i:ℕ,∃ j, (i ≠ j) → (lk < 1000)) → k = 1998 :=
by sorry

end max_lamps_on_road_l510_510640


namespace min_dominoes_8x8_grid_l510_510533

theorem min_dominoes_8x8_grid :
  ∀ (grid : matrix (fin 8) (fin 8) ℕ),
    (∀ i j, grid i j = 0 ∨ grid i j = 1) ∧
    (∀ i j, grid i j = 1 → (
      (i > 0 → grid (i - 1) j = 0) ∧
      (i < 7 → grid (i + 1) j = 0) ∧
      (j > 0 → grid i (j - 1) = 0) ∧
      (j < 7 → grid i (j + 1) = 0)
    )) →
    (∃ n tiles : matrix (fin 8) (fin 8) bool, 
      n = 28 ∧
      (∀ i j, 
          ((tiles i j = tt ∧ tiles i (j + 1) = tt) ∨ 
           (tiles i j = tt ∧ tiles (i + 1) j = tt)) ∧ 
          grid i j = 1))
    sorry

end min_dominoes_8x8_grid_l510_510533


namespace comb_identity_a_comb_identity_b_comb_identity_c_comb_identity_d_comb_identity_e_l510_510140

-- Statement for problem (a)
theorem comb_identity_a (r m k : ℕ) (h1 : 0 ≤ k) (h2 : k ≤ m) (h3 : m ≤ r) :
    C(r, m) * C(m, k) = C(r, k) * C(r - k, m - k) := 
sorry

-- Statement for problem (b)
theorem comb_identity_b (n m : ℕ) :
    C(n + 1, m + 1) = C(n, m) + C(n, m + 1) := 
sorry

-- Statement for problem (c)
theorem comb_identity_c (n : ℕ) :
    C(2 * n, n) = ∑ i in (range (n + 1)), (C(n, i))^2 := 
sorry

-- Statement for problem (d)
theorem comb_identity_d (m n k : ℕ) (h1 : 0 ≤ k) (h2 : k ≤ n) :
    C(m + n, k) = ∑ p in (range (k + 1)), C(n, p) * C(m, k - p) :=
sorry

-- Statement for problem (e)
theorem comb_identity_e (n k : ℕ) :
    C(n, k) = ∑ i in (range (n - k + 1)), C(n - i , k - 1) :=
sorry

end comb_identity_a_comb_identity_b_comb_identity_c_comb_identity_d_comb_identity_e_l510_510140


namespace speed_of_train_l510_510657

-- Define the conditions as constants
constant distance : ℕ := 140   -- distance in meters
constant time : ℕ := 9        -- time in seconds

-- Define the conversion factor from m/s to km/hr
constant conversion_factor : ℚ := 18 / 5

-- Calculate speed in m/s
def speed_m_per_s : ℚ := distance / time

-- Calculate speed in km/hr
def speed_km_per_hr : ℚ := speed_m_per_s * conversion_factor

-- State the theorem
theorem speed_of_train : speed_km_per_hr = 280 := by
  -- placeholder for the proof
  sorry

end speed_of_train_l510_510657


namespace planar_graph_edge_orientation_l510_510139

open SimpleGraph

theorem planar_graph_edge_orientation {G : SimpleGraph V} [DecidableRel G.adj] 
  (h1 : G.IsSimple) (h2 : G.IsPlanar) (h3 : Finite V) :
  ∃ (orient : ∀ (u v : V), G.adj u v → Prop), 
    (∀ (c : G.cycle), (∃ k, length c = k ∧ count (λ e, orient e.fst e.snd e.right) c.edges ≤ (3 * k) / 4)) ∧
  (∀ (H : SimpleGraph V') [DecidableRel H.adj] (p1 : H.IsSimple) (p2 : H.IsPlanar) (p3 : Finite V'), 
   ∃ (c : H.cycle), (∃ m, length c = m ∧ count (λ e, orient e.fst e.snd e.right) c.edges = (3 * m) / 4)). 

end planar_graph_edge_orientation_l510_510139


namespace perfect_square_sequence_exists_l510_510759

theorem perfect_square_sequence_exists (A : ℕ) (hA : 10^15 ≤ A ∧ A < 10^16) (digits : Fin 16 → ℕ)
  (h_nonzero : digits 0 ≠ 0 ∧ ∀ i, digits i ≠ 0) :
  ∃ (i j : Fin 16), i < j ∧ (∏ k in Finset.Icc i j, digits k) ∣ ∏ m in Finset.Icc i j, digits m :=
by sorry

end perfect_square_sequence_exists_l510_510759


namespace num_parents_correct_l510_510896

/- Define the given conditions -/
def num_fifth_graders := 109
def num_sixth_graders := 115
def num_seventh_graders := 118
def num_teachers := 4
def num_buses := 5
def seats_per_bus := 72

/- Define the total number of people (students and teachers) -/
def total_students_and_teachers := num_fifth_graders + num_sixth_graders + num_seventh_graders + num_teachers

/- Define the total seating capacity of the buses -/
def total_seats := num_buses * seats_per_bus

/- Define the number of parents going on the trip -/
def num_parents := total_seats - total_students_and_teachers

/- Define the number of parents from each grade such that their sum equals num_parents -/
def parents_fifth_grade := 4
def parents_sixth_grade := 5
def parents_seventh_grade := 5

theorem num_parents_correct :
  num_parents = parents_fifth_grade + parents_sixth_grade + parents_seventh_grade :=
by
  /- Calculate the number of people going on the trip -/
  have h1: total_students_and_teachers = 109 + 115 + 118 + 4 := rfl
  have h2 : total_students_and_teachers = 346 := by norm_num
   
  /- Calculate the total seating capacity -/
  have h3: total_seats = 5 * 72 := rfl
  have h4 : total_seats = 360 := by norm_num
  
  /- Calculate the number of parents -/
  have h5: num_parents = 360 - 346 := rfl
  have h6 : num_parents = 14 := by norm_num 
   
  /- Distribute parents correctly across the grades -/
  have h7: parents_fifth_grade + parents_sixth_grade + parents_seventh_grade = 4 + 5 + 5 := rfl
  have h8 : parents_fifth_grade + parents_sixth_grade + parents_seventh_grade = 14 := by norm_num

  /- Conclude the proof -/
  exact h6.symm ▸ h8 

end num_parents_correct_l510_510896


namespace yellow_fraction_after_tripling_l510_510451

theorem yellow_fraction_after_tripling
  (n : ℕ)
  (h_non_zero : n ≠ 0)
  (h_green : 2 / 3 * n = ⌊2 / 3 * n⌋) -- ensures number of marbles is whole
  (h_green_marbles : ∀ n, 2 / 3 * n)
  : (∀ n, 3 / 5 = (n / 3 + 3 * (n / 3)) / (n + n / 3)) :=
by sorry

end yellow_fraction_after_tripling_l510_510451


namespace speed_of_train_l510_510658

-- Define the conditions as constants
constant distance : ℕ := 140   -- distance in meters
constant time : ℕ := 9        -- time in seconds

-- Define the conversion factor from m/s to km/hr
constant conversion_factor : ℚ := 18 / 5

-- Calculate speed in m/s
def speed_m_per_s : ℚ := distance / time

-- Calculate speed in km/hr
def speed_km_per_hr : ℚ := speed_m_per_s * conversion_factor

-- State the theorem
theorem speed_of_train : speed_km_per_hr = 280 := by
  -- placeholder for the proof
  sorry

end speed_of_train_l510_510658


namespace pentagon_area_l510_510576

def pentagon_vertices : list (ℕ × ℕ) := [(1,1), (4,1), (5,3), (3,5), (1,4)]

noncomputable def area_of_polygon (vertices : list (ℕ × ℕ)) : ℝ := 
  0.5 * abs (
    (vertices.zip (vertices.tail.concat vertices.head))
    .foldl (λ acc ⟨(x1, y1), (x2, y2)⟩ => acc + x1 * y2 - x2 * y1) 0)

theorem pentagon_area : area_of_polygon pentagon_vertices = 12 :=
by
  sorry

end pentagon_area_l510_510576


namespace sum_interior_angles_l510_510483

-- Define the condition that each interior angle is 6 times its corresponding exterior angle
def interior_angle_eq_six_times_exterior (W : Polygon) :=
  ∀ (n : ℕ) (h : n < W.num_sides), W.interior_angle n = 6 * W.exterior_angle n

-- Prove the sum of the interior angles
theorem sum_interior_angles (W : Polygon) 
  (h_cond : interior_angle_eq_six_times_exterior W) : 
  W.sum_interior_angles = 2160 ∧ (W.is_regular ∨ ¬W.is_regular) :=
by
  sorry

end sum_interior_angles_l510_510483


namespace min_omega_condition_l510_510441

theorem min_omega_condition :
  ∃ (ω: ℝ) (k: ℤ), (ω > 0) ∧ (ω = 6 * k + 1 / 2) ∧ (∀ (ω' : ℝ), (ω' > 0) ∧ (∃ (k': ℤ), ω' = 6 * k' + 1 / 2) → ω ≤ ω') := 
sorry

end min_omega_condition_l510_510441


namespace exactly_one_wins_at_most_two_win_l510_510564

def prob_A : ℚ := 4 / 5 
def prob_B : ℚ := 3 / 5 
def prob_C : ℚ := 7 / 10

theorem exactly_one_wins :
  (prob_A * (1 - prob_B) * (1 - prob_C) + 
   (1 - prob_A) * prob_B * (1 - prob_C) + 
   (1 - prob_A) * (1 - prob_B) * prob_C) = 47 / 250 := 
by sorry

theorem at_most_two_win :
  (1 - (prob_A * prob_B * prob_C)) = 83 / 125 :=
by sorry

end exactly_one_wins_at_most_two_win_l510_510564


namespace curve_representation_l510_510910

   theorem curve_representation :
     ∀ (x y : ℝ), x^4 - y^4 - 4*x^2 + 4*y^2 = 0 ↔ (x + y = 0 ∨ x - y = 0 ∨ x^2 + y^2 = 4) :=
   by
     sorry
   
end curve_representation_l510_510910


namespace xy_value_l510_510753

variable (a b x y : ℝ)
variable (h1 : 2 * a^x * b^3 = - a^2 * b^(1 - y))
variable (hx : x = 2)
variable (hy : y = -2)

theorem xy_value : x * y = -4 := 
by
  sorry

end xy_value_l510_510753


namespace solve_for_y_l510_510357

theorem solve_for_y (x y : ℝ) (h : 4 * x + y = 9) : y = 9 - 4 * x :=
by sorry

end solve_for_y_l510_510357


namespace pyramid_volume_l510_510905

theorem pyramid_volume (S : ℝ) :
  ∃ (V : ℝ),
  (∀ (a b h : ℝ), S = a * b ∧
  h = a * (Real.tan (60 * (Real.pi / 180))) ∧
  h = b * (Real.tan (30 * (Real.pi / 180))) ∧
  V = (1/3) * S * h) →
  V = (S * Real.sqrt S) / 3 :=
by
  sorry

end pyramid_volume_l510_510905


namespace g_at_6_l510_510490

variable {ℝ : Type*}

noncomputable def g (v : ℝ) : ℝ := ( ((v + 2) / 4)^2 + 2 * ((v + 2) / 4) + 3 )

theorem g_at_6 : g 6 = 11 :=
sorry

end g_at_6_l510_510490


namespace number_of_true_propositions_l510_510491

noncomputable def proposition1 (l : Line) (α β : Plane) : Prop :=
  (l ⊥ α ∧ l ⊥ β) → (α ∥ β)

noncomputable def proposition2 (l : Line) (α β : Plane) : Prop :=
  (l ∥ α ∧ l ∥ β) → (α ⊥ β)

noncomputable def proposition3 (l : Line) (α β : Plane) : Prop :=
  (α ⊥ β ∧ l ∥ α) → (l ∥ β)

noncomputable def proposition4 (m n : Line) (α : Plane) : Prop :=
  (m ∥ n ∧ m ⊥ α) → (n ⊥ α)

theorem number_of_true_propositions (l m n : Line) (α β : Plane) :
  let p1_correct := proposition1 l α β = true,
      p2_correct := proposition2 l α β = false,
      p3_correct := proposition3 l α β = false,
      p4_correct := proposition4 m n α = true in
  (p1_correct ∨ ¬p1_correct) ∧
  (p2_correct ∨ ¬p2_correct) ∧
  (p3_correct ∨ ¬p3_correct) ∧
  (p4_correct ∨ ¬p4_correct) ∧
  (p1_correct + p2_correct + p3_correct + p4_correct = 2) := 
by
    sorry

end number_of_true_propositions_l510_510491


namespace net_income_after_tax_l510_510193

theorem net_income_after_tax (gross_income : ℝ) (tax_rate : ℝ) : 
  (gross_income = 45000) → (tax_rate = 0.13) → 
  (gross_income - gross_income * tax_rate = 39150) :=
by
  intro h1 h2
  rw [h1, h2]
  sorry

end net_income_after_tax_l510_510193


namespace eighth_vertex_on_sphere_l510_510369

theorem eighth_vertex_on_sphere
  (hexagon : Type)
  (vertices : hexagon → Point)
  (quads : set (hexagon → set Point))
  (sphere : set Point)
  (h_convex : convex hexagon)
  (h_quads : ∀ face ∈ quads, ∃ (a b c d : hexagon), face = {vertices a, vertices b, vertices c, vertices d})
  (h_seven_vertices : ∃ (pts : list Point), pts.length = 7 ∧ ∀ pt ∈ pts, pt ∈ sphere)
  (h_quad_faces_on_sphere : ∀ pt ∈ seven_verts, pt ∈ sphere):
  ∀ (eighth_vertex : hexagon), vertices eighth_vertex ∈ sphere := 
sorry

end eighth_vertex_on_sphere_l510_510369


namespace field_length_l510_510164

theorem field_length (w l : ℕ) (Pond_Area : ℕ) (Pond_Field_Ratio : ℚ) (Field_Length_Ratio : ℕ) 
  (h1 : Length = 2 * Width)
  (h2 : Pond_Area = 8 * 8)
  (h3 : Pond_Field_Ratio = 1 / 50)
  (h4 : Pond_Area = Pond_Field_Ratio * Field_Area)
  : l = 80 := 
by
  -- begin solution
  sorry

end field_length_l510_510164


namespace perfect_cube_three_distinct_integers_l510_510092

theorem perfect_cube_three_distinct_integers 
  (S : Finset ℕ) 
  (h_distinct : S.card = 9) 
  (h_prime_factors : ∀ n ∈ S, ∀ p, nat.prime p → p ∣ n → p = 2 ∨ p = 3) 
  : ∃ a b c ∈ S, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ∃ k, a * b * c = k^3 :=
begin
  sorry
end

end perfect_cube_three_distinct_integers_l510_510092


namespace eigenvalues_of_M_l510_510411

variable (a b : ℝ)
def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, a], ![-1, b]]
def v : Fin 2 → ℝ := ![1, -1]
def w : Fin 2 → ℝ := ![-1, 5]

theorem eigenvalues_of_M :
  (M a b).mulVec v = w → (∀ λ, λ ∈ (M a b).eigenvalues ↔ λ = 2 ∨ λ = 3) :=
by
  sorry

end eigenvalues_of_M_l510_510411


namespace sum_two_numbers_in_AP_and_GP_equals_20_l510_510944

theorem sum_two_numbers_in_AP_and_GP_equals_20 :
  ∃ a b : ℝ, 
    (a > 0) ∧ (b > 0) ∧ 
    (4 < a) ∧ (a < b) ∧ 
    (4 + (a - 4) = a) ∧ (4 + 2 * (a - 4) = b) ∧
    (a * (b / a) = b) ∧ (b * (b / a) = 16) ∧ 
    a + b = 20 :=
by
  sorry

end sum_two_numbers_in_AP_and_GP_equals_20_l510_510944


namespace problem_statement_l510_510857

def a : ℝ := 0.98 + Real.sin 0.01
def b : ℝ := Real.exp (-0.01)
def c : ℝ := (Real.log 2022 / Real.log 2021) / (Real.log 2023 / Real.log 2022)

theorem problem_statement :
  c > b ∧ b > a := by
  sorry

end problem_statement_l510_510857


namespace odd_function_a_plus_b_l510_510398

def f (x : ℝ) (a b : ℝ) : ℝ :=
if x > 0 then x - 1
else if x = 0 then a
else x + b

theorem odd_function_a_plus_b (a b : ℝ) (f_odd : ∀ x : ℝ, f x a b = - f (-x) a b) :
  a + b = 1 := by
sorry

end odd_function_a_plus_b_l510_510398


namespace exists_root_in_interval_l510_510335

noncomputable def f : ℝ → ℝ := λ x, Real.log x + (1/2) * x - 2

theorem exists_root_in_interval : ∃ x ∈ Ioo 2 3, f x = 0 :=
by
  have f2 : f 2 < 0 := by simp [f, Real.log_two_lt, add_sub_lt_iff_right, half_lt_self]
  have f3 : f 3 > 0 := by simp [f, Real.log_three_gt_one]
  exact exists_Ioo f2 f3
  sorry

end exists_root_in_interval_l510_510335


namespace three_digit_numbers_l510_510711

def is_valid_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧
  let (a, rest) := n / 100, n % 100 in
  let (c, b) := rest / 10, rest % 10 in
  n = 100 * a + 10 * c + b ∧
  100 * a + 10 * c + b = 11 * (a * a + c * c + b * b)

theorem three_digit_numbers (n : ℕ) : is_valid_number n → (n = 550 ∨ n = 803) :=
begin
  sorry
end

end three_digit_numbers_l510_510711


namespace probability_sum_greater_than_three_l510_510951

-- Define the sample space and the events
def sample_space : set (ℕ × ℕ) := {(i, j) | i ∈ finset.range 1 7, j ∈ finset.range 1 7}

def event_sum_greater_than_three : set (ℕ × ℕ) :=
  {(i, j) | i + j > 3 ∧ i ∈ finset.range 1 7 ∧ j ∈ finset.range 1 7}

-- Probability calculation
noncomputable def probability (event : set (ℕ × ℕ)) : ℚ :=
  (finset.card (event ∩ sample_space).to_finset) / (finset.card sample_space.to_finset)

-- Define the conditions and statement
theorem probability_sum_greater_than_three : probability event_sum_greater_than_three = 11 / 12 :=
by sorry

end probability_sum_greater_than_three_l510_510951


namespace largest_angle_is_120_l510_510078

variable (d e f : ℝ)
variable (h1 : d + 3 * e + 3 * f = d^2)
variable (h2 : d + 3 * e - 3 * f = -4)

theorem largest_angle_is_120 (h1 : d + 3 * e + 3 * f = d^2) (h2 : d + 3 * e - 3 * f = -4) : 
  ∃ (F : ℝ), F = 120 :=
by
  sorry

end largest_angle_is_120_l510_510078


namespace mass_percentage_B_in_boric_acid_l510_510343

def atomic_mass_H : ℝ := 1.01
def atomic_mass_B : ℝ := 10.81
def atomic_mass_O : ℝ := 16.00

def molar_mass_H₃BO₃ : ℝ :=
  3 * atomic_mass_H + atomic_mass_B + 3 * atomic_mass_O

def mass_percentage_B_in_H₃BO₃ : ℝ :=
  (atomic_mass_B / molar_mass_H₃BO₃) * 100

theorem mass_percentage_B_in_boric_acid :
  mass_percentage_B_in_H₃BO₃ = 17.48 := by
  sorry

end mass_percentage_B_in_boric_acid_l510_510343


namespace number_of_planes_l510_510190

theorem number_of_planes
  (total_wings : ℕ) (wings_per_plane : ℕ) (h_total_wings : total_wings = 50) (h_wings_per_plane : wings_per_plane = 2) :
  total_wings / wings_per_plane = 25 :=
by
  rw [h_total_wings, h_wings_per_plane]
  norm_num

end number_of_planes_l510_510190


namespace minimum_period_l510_510166

noncomputable def f (x : ℝ) : ℝ := Real.sin x * (4 * (Real.cos x)^2 - 1)

theorem minimum_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', (∀ x, f (x + T') = f x) → T' ≥ T) :=
begin
  use (2 * Real.pi / 3),
  sorry,
end

end minimum_period_l510_510166


namespace water_delivery_rate_l510_510183

variables (x V t : ℝ)
-- Conditions as Lean definitions
def identical_pools_started_at_same_time : Prop := True

def first_pool_rate : ℝ := x + 30
def second_pool_rate : ℝ := x

def total_water_equal_to_volume_at_certain_moment : Prop := (first_pool_rate t + second_pool_rate t) = V

def first_pool_filled_in_time : Prop := (t + 8/3) * first_pool_rate = V
def second_pool_filled_in_time : Prop := (t + 10/3) * second_pool_rate = V

-- Mathematical proof problem statement
theorem water_delivery_rate :
  identical_pools_started_at_same_time →
  total_water_equal_to_volume_at_certain_moment →
  first_pool_filled_in_time →
  second_pool_filled_in_time →
  first_pool_rate = 90 ∧ second_pool_rate = 60 :=
by
  intros
  -- The remainder of the proof can be filled in here.
  sorry

end water_delivery_rate_l510_510183


namespace sum_of_areas_reflections_equal_original_l510_510884

-- Definitions reflecting the conditions
def is_reflection_about_circumcenter (O A A' : Point) : Prop :=
  reflection_about O A A'

def is_acute_triangle (A B C : Point) : Prop :=
  acute_triangle A B C

-- Problem statement as a theorem in Lean 4
theorem sum_of_areas_reflections_equal_original
  (A B C O A' B' C' : Point)
  (hacute : is_acute_triangle A B C)
  (hreflectA : is_reflection_about_circumcenter O A A')
  (hreflectB : is_reflection_about_circumcenter O B B')
  (hreflectC : is_reflection_about_circumcenter O C C') :
  area (triangle A' B C) + area (triangle A B' C) + area (triangle A B C') = area (triangle A B C) :=
sorry

end sum_of_areas_reflections_equal_original_l510_510884


namespace f_odd_f_shift_f_in_range_find_f_7_5_l510_510363

def f : ℝ → ℝ := sorry  -- We define the function f (implementation is not needed here)

theorem f_odd (x : ℝ) : f (-x) = -f x := sorry

theorem f_shift (x : ℝ) : f (x + 2) = -f x := sorry

theorem f_in_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x = x := sorry

theorem find_f_7_5 : f 7.5 = 0.5 :=
by
  sorry

end f_odd_f_shift_f_in_range_find_f_7_5_l510_510363


namespace square_area_from_perimeter_l510_510336

theorem square_area_from_perimeter (P : ℝ) (hP : P = 40) : ∃ A : ℝ, A = 100 :=
by
  let s := P / 4
  have hs : s = 10 := by
    rw [hP]
    norm_num
  let A := s^2
  have hA : A = 100 := by
    rw [hs]
    norm_num
  use A
  exact hA

end square_area_from_perimeter_l510_510336


namespace find_f_8_l510_510364

/-- Define the function f as a piecewise function --/
def f : ℝ → ℝ :=
  λ x, if x ≤ 10 then f (f (x + 2)) else x ^ 2 - 131

/-- Prove that f (8) = 1313 --/
theorem find_f_8 : f 8 = 1313 := by
  sorry

end find_f_8_l510_510364


namespace angle_C_greater_half_angle_A_l510_510883

theorem angle_C_greater_half_angle_A (A B C : Type)
  [real_inner_product_space ℝ A] [real_inner_product_space ℝ B] [real_inner_product_space ℝ C]
  (angle_B : ∠B > 90°)
  (h : |AB| = |AC| / 2) :
  ∠C > (∠A / 2) :=
by
  sorry

end angle_C_greater_half_angle_A_l510_510883


namespace perpendicular_vectors_implies_value_of_m_l510_510419

variable (m : ℝ)

-- Definitions based on conditions
def vector_a : (ℝ × ℝ) := (1, 2)
def vector_b (m : ℝ) : (ℝ × ℝ) := (-1, m)

-- Main theorem
theorem perpendicular_vectors_implies_value_of_m : vector_a ⬝ vector_b m = 0 → m = 1 / 2 :=
by
  sorry

end perpendicular_vectors_implies_value_of_m_l510_510419


namespace isosceles_triangle_angles_l510_510569

theorem isosceles_triangle_angles (y : ℝ) (h : y > 0) :
  let P := y
  let R := 5 * y
  let Q := R
  P + Q + R = 180 → Q = 81.82 :=
by
  sorry

end isosceles_triangle_angles_l510_510569


namespace trains_meet_at_midpoint_bridge_l510_510253

theorem trains_meet_at_midpoint_bridge :
  ∀ (length_train_A length_train_B : ℕ) (time_to_cross_signal_post : ℕ) (length_bridge : ℕ),
    length_train_A = 600 →
    time_to_cross_signal_post = 40 →
    length_train_B = 800 →
    length_bridge = 9000 →
    let speed_train_A := length_train_A / time_to_cross_signal_post in
    let speed_train_B := speed_train_A in
    let relative_speed := speed_train_A + speed_train_B in
    let time_to_meet := length_bridge / relative_speed in
    let distance_A_travels := speed_train_A * time_to_meet in
    time_to_meet = 300 ∧ distance_A_travels = 4500 := 
by
  intros length_train_A length_train_B time_to_cross_signal_post length_bridge 
         hlength_train_A htime_to_cross_signal_post hlength_train_B hlength_bridge
         speed_train_A speed_train_B relative_speed time_to_meet distance_A_travels,
  sorry

end trains_meet_at_midpoint_bridge_l510_510253


namespace line_circle_intersect_l510_510409

theorem line_circle_intersect (a : ℝ) (h : a > 1) :
  ∃ x y : ℝ, (x - a)^2 + (y - 1)^2 = 2 ∧ x - a * y - 2 = 0 :=
sorry

end line_circle_intersect_l510_510409


namespace hexagon_valid_assignments_l510_510512

noncomputable theory

def is_valid_assignment (digits : Finset ℕ) (assignment : Fin 7 → ℕ) : Prop :=
  (assignment 0 ≠ assignment 1) ∧ (assignment 0 ≠ assignment 2) ∧ (assignment 0 ≠ assignment 3) ∧
  (assignment 0 ≠ assignment 4) ∧ (assignment 0 ≠ assignment 5) ∧ (assignment 0 ≠ assignment 6) ∧
  (assignment 1 ≠ assignment 2) ∧ (assignment 1 ≠ assignment 3) ∧ (assignment 1 ≠ assignment 4) ∧
  (assignment 1 ≠ assignment 5) ∧ (assignment 1 ≠ assignment 6) ∧ (assignment 2 ≠ assignment 3) ∧
  (assignment 2 ≠ assignment 4) ∧ (assignment 2 ≠ assignment 5) ∧ (assignment 2 ≠ assignment 6) ∧
  (assignment 3 ≠ assignment 4) ∧ (assignment 3 ≠ assignment 5) ∧ (assignment 3 ≠ assignment 6) ∧
  (assignment 4 ≠ assignment 5) ∧ (assignment 4 ≠ assignment 6) ∧ (assignment 5 ≠ assignment 6) ∧
  ((assignment 0) + (assignment 4) + (assignment 6) = 
  (assignment 1) + (assignment 3) + (assignment 5)) ∧
  ((assignment 0) + (assignment 4) + (assignment 6) = 
  (assignment 2) + (assignment 3) + (assignment 5)) ∧
  ((assignment 0) + (assignment 4) + (assignment 6) = 
  (assignment 3) + (assignment 5) + (assignment 6))

theorem hexagon_valid_assignments : ∃! (assignment : Fin 7 → ℕ), 
  is_valid_assignment (Finset.range 8).erase 0 assignment = 144 := 
begin
  sorry
end

end hexagon_valid_assignments_l510_510512


namespace place_mat_length_l510_510642

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) (inner_touch : Bool)
  (h1 : r = 4)
  (h2 : n = 6)
  (h3 : w = 1)
  (h4 : inner_touch = true)
  : x = (3 * Real.sqrt 7 - Real.sqrt 3) / 2 :=
sorry

end place_mat_length_l510_510642


namespace system_of_equations_solutions_l510_510170

theorem system_of_equations_solutions :
  ∃ (sol : Finset (ℝ × ℝ)), sol.card = 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ sol ↔ (x + 3 * y = 3 ∧ abs (abs x - abs y) = 1)) :=
by
  sorry

end system_of_equations_solutions_l510_510170


namespace no_solution_k_eq_7_l510_510321

-- Define the condition that x should not be equal to 4 and 8
def condition (x : ℝ) : Prop := x ≠ 4 ∧ x ≠ 8

-- Define the equation
def equation (x k : ℝ) : Prop := (x - 3) / (x - 4) = (x - k) / (x - 8)

-- Prove that for the equation to have no solution, k must be 7
theorem no_solution_k_eq_7 : (∀ x, condition x → ¬ equation x 7) ↔ (∃ k, k = 7) :=
by
  sorry

end no_solution_k_eq_7_l510_510321


namespace is_quadratic_function_l510_510663

theorem is_quadratic_function (x : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = 2 * x + 3) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = 2 / x) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = (x - 1)^2 - x^2) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = 3 * x^2 - 1) ∧ (∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c))) :=
by
  sorry

end is_quadratic_function_l510_510663


namespace percentage_republicans_vote_for_X_l510_510052

-- Defining the constants and variables
variables (R : ℝ) (p : ℝ)

-- Defining the conditions
def ratio_republicans_democrats : Prop :=
  ∀ R : ℝ, 3 * R + 2 * R = 5 * R

def votes_republicans_for_X : ℝ := (p / 100) * 3 * R
def votes_democrats_for_X : ℝ := 0.25 * 2 * R
def total_votes_X : ℝ := votes_republicans_for_X + votes_democrats_for_X

def votes_republicans_for_Y : ℝ := (1 - p / 100) * 3 * R
def votes_democrats_for_Y : ℝ := (1 - 0.25) * 2 * R
def total_votes_Y : ℝ := votes_republicans_for_Y + votes_democrats_for_Y

def win_margin : ℝ := 0.16000000000000014 * 5 * R

-- Statement to prove
theorem percentage_republicans_vote_for_X :
  ∀ R : ℝ, total_votes_X R p = total_votes_Y R p + win_margin R → p = 80 :=
begin
  sorry,
end

end percentage_republicans_vote_for_X_l510_510052


namespace find_y_divisor_l510_510268

noncomputable def y_divisor (x : ℝ) : ℝ := (x * 8) / (x^2)

theorem find_y_divisor :
  let x := 2.6666666666666665 in
  y_divisor x = 3 :=
by
  -- We leave the proof as an exercise to work out manually.
  sorry

end find_y_divisor_l510_510268


namespace arithmetic_mean_1001_l510_510117

def M : Finset ℕ := (Finset.range 1000).image (λ n, n + 1)

def a_X (X : Finset ℕ) : ℕ := X.max' (Finset.nonempty_of_ne_empty (by assumption)) + X.min' (Finset.nonempty_of_ne_empty (by assumption))

theorem arithmetic_mean_1001 :
  let S : Finset (Finset ℕ) := Finset.powerset M \ { ∅ }
  ∑ x in S, a_X x / S.card = 1001 := sorry

end arithmetic_mean_1001_l510_510117


namespace school_B_saving_l510_510880

def cost_A (kg_price : ℚ) (kg_amount : ℚ) : ℚ :=
  kg_price * kg_amount

def effective_kg_B (total_kg : ℚ) (extra_percentage : ℚ) : ℚ :=
  total_kg / (1 + extra_percentage)

def cost_B (kg_price : ℚ) (effective_kg : ℚ) : ℚ :=
  kg_price * effective_kg

theorem school_B_saving
  (kg_amount : ℚ) (price_A: ℚ) (discount: ℚ) (extra_percentage : ℚ) 
  (expected_saving : ℚ)
  (h1 : kg_amount = 56)
  (h2 : price_A = 8.06)
  (h3 : discount = 0.56)
  (h4 : extra_percentage = 0.05)
  (h5 : expected_saving = 51.36) :
  cost_A price_A kg_amount - cost_B (price_A - discount) (effective_kg_B kg_amount extra_percentage) = expected_saving := 
by 
  sorry

end school_B_saving_l510_510880


namespace percentage_loss_is_five_l510_510630

-- Definitions of Cost Price and Selling Price
def CP : ℝ := 1400
def SP : ℝ := 1330

-- Definition of Loss
def Loss : ℝ := CP - SP

-- Definition of Percentage Loss
def PercentageLoss : ℝ := (Loss / CP) * 100

-- Theorem: The percentage of loss is 5%
theorem percentage_loss_is_five : PercentageLoss = 5 := by
  sorry

end percentage_loss_is_five_l510_510630


namespace earnings_ratio_l510_510470

-- Definitions for conditions
def jerusha_earnings : ℕ := 68
def total_earnings : ℕ := 85
def lottie_earnings : ℕ := total_earnings - jerusha_earnings

-- Prove that the ratio of Jerusha's earnings to Lottie's earnings is 4:1
theorem earnings_ratio : 
  ∃ (k : ℕ), jerusha_earnings = k * lottie_earnings ∧ (jerusha_earnings + lottie_earnings = total_earnings) ∧ (jerusha_earnings = 68) ∧ (total_earnings = 85) →
  68 / (total_earnings - 68) = 4 := 
by
  sorry

end earnings_ratio_l510_510470


namespace rope_cut_probability_l510_510434

noncomputable theory

def probability_pieces_at_least_1_meter (total_length : ℝ) (min_length : ℝ) : ℝ :=
  let allowable_range := total_length - 2 * min_length
  allowable_range / total_length

theorem rope_cut_probability : probability_pieces_at_least_1_meter 5 1 = 3 / 5 :=
by
  unfold probability_pieces_at_least_1_meter
  sorry

end rope_cut_probability_l510_510434


namespace striker_path_length_le_sum_sides_l510_510702

theorem striker_path_length_le_sum_sides 
  (x y : ℝ)
  (n : ℕ)
  (s : Fin n → ℝ)
  (x_i y_i : Fin n → ℝ)
  (h1 : ∑ i, x_i i ≤ x)
  (h2 : ∑ i, y_i i ≤ y)
  (h3 : ∀ i, s i ≥ x_i i + y_i i) :
  ∑ i, s i ≤ x + y := 
sorry

end striker_path_length_le_sum_sides_l510_510702


namespace nonneg_diff_between_roots_l510_510587

theorem nonneg_diff_between_roots : 
  ∀ x : ℝ, 
  (x^2 + 42 * x + 384 = 0) → 
  (∃ r1 r2 : ℝ, x = r1 ∨ x = r2) → 
  (abs (r1 - r2) = 8) := 
sorry

end nonneg_diff_between_roots_l510_510587


namespace find_ellipse_equation_find_slope_l510_510745

-- Define the variables and conditions
variables (a b : ℝ)
variables (point_on_ellipse : ℝ × ℝ)
variables (eccentricity : ℝ)
variables (A : ℝ × ℝ)
variables (right_focus : ℝ × ℝ)
variables (M : ℝ × ℝ)
variables (circle_center : ℝ × ℝ)
variables (P : ℝ × ℝ)
variables (k : ℝ)

-- Define the conditions mentioned in the problem
def ellipse_equation (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def passing_through := point_on_ellipse = (sqrt 2, sqrt 3)
def eccentricity_condition := eccentricity = sqrt 2 / 2
def A_coord := A = (0, -2)
def right_focus_condition := right_focus = (2, 0)
def M_condition := M = (2/3, 0)
def tangent_condition := P = (0.5 * A.1 + 0.5 * (8 * k / (2 * k^2 + 1)), 0.5 * A.2 + 0.5 * ((4 * k^2 - 2) / (2 * k^2 + 1)))

-- Define the target properties
def target_equation := (a = 2 * sqrt 2) ∧ (b = 2)

-- Prove the properties
theorem find_ellipse_equation :
  (ellipse_equation point_on_ellipse.fst point_on_ellipse.snd) ∧
  (passing_through) ∧
  (eccentricity_condition) → target_equation :=
by
  -- To be proved
  sorry

-- Define the target slope
def target_slope := (k = 1/2) ∨ (k = 1)

-- Prove the slope of line AB
theorem find_slope :
  (A_coord) ∧ (right_focus_condition) ∧ (M_condition) ∧ (tangent_condition) → target_slope :=
by
  -- To be proved
  sorry

end find_ellipse_equation_find_slope_l510_510745


namespace modulus_of_z_l510_510017

def z : ℂ := (1 + 2 * Complex.i) * (2 - Complex.i)

theorem modulus_of_z : Complex.abs z = 5 :=
sorry

end modulus_of_z_l510_510017


namespace sum_of_three_lowest_scores_l510_510145

-- Definitions for the conditions
def scores : Type := list ℕ

def mean (l : scores) : ℕ := (list.sum l) / (list.length l)

def is_median (l : scores) (m : ℕ) : Prop :=
(sorted l).nth (list.length l / 2) = some m ∨ (sorted l).nth (list.length l / 2 - 1) = some m

def is_mode (l : scores) (mo : ℕ) : Prop :=
list.foldr (λ n (acc : ℕ × ℕ), if n = mo then (acc.1 + 1, acc.1.max acc.2) else (0, acc.2.max acc.1)) (0, 0) l |>.2 > 1

-- The statement of the theorem to be proved
theorem sum_of_three_lowest_scores (l : scores) 
    (h_len : list.length l = 6) 
    (h_mean : mean l = 92) 
    (h_median : is_median l 95) 
    (h_mode : is_mode l 97) : 
    list.sum (take 3 (sorted l)) = 263 := 
sorry

end sum_of_three_lowest_scores_l510_510145


namespace circle_equation_l510_510014

theorem circle_equation (r : ℝ) (h k : ℝ) (eq1 : r = 2) (center_on_xaxis : k = 0) (tangent_to_yaxis : h - r = 0) : 
  (x y : ℝ) : x^2 + y^2 - 4 * x = 0 :=
by sorry

end circle_equation_l510_510014


namespace sequence_periodicity_find_b_2006_l510_510495

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 1 then 3 else if n = 2 then 4 else sequence (n - 1)^2 / sequence (n - 2)

theorem sequence_periodicity :
  ∀ n ≥ 3, sequence n = sequence 4 :=
by
  intros n hn
  induction n using Nat.case_strong_induction_on with n ih
  case 0 => contradiction
  case 1 => contradiction
  case 2 => contradiction
  case succ n ih_n =>
    sorry

theorem find_b_2006 : sequence 2006 = 64 / 9 :=
by
  have h_period : ∀ n ≥ 3, sequence n = sequence 4 := sequence_periodicity
  exact h_period 2006 (le_of_lt (by norm_num))

end sequence_periodicity_find_b_2006_l510_510495


namespace place_mats_length_l510_510644

theorem place_mats_length (r : ℝ) (w : ℝ) (n : ℕ) (x : ℝ) :
  r = 4 ∧ w = 1 ∧ n = 6 ∧
  (∀ i, i < n → 
    let inner_corners_touch := true in
    place_mat_placement_correct r w x i inner_corners_touch) →
  x = (3 * real.sqrt 7 - real.sqrt 3) / 2 :=
by sorry

end place_mats_length_l510_510644


namespace arc_length_of_curve_l510_510242

open Real

noncomputable def curve_function (x : ℝ) : ℝ := (x^2 / 4) - (log x / 2)

theorem arc_length_of_curve :
  ∫ x in 1..2, sqrt (1 + ((deriv curve_function x)^2)) = 3 / 4 + 1 / 2 * log 2 :=
by
  sorry

end arc_length_of_curve_l510_510242


namespace principal_argument_conjugate_l510_510018

open Complex Real

noncomputable def principal_arg_of_conjugate (z : ℂ) (θ : ℝ) (h1 : z = 1 - sin θ + I * cos θ) (h2 : π / 2 < θ) (h3 : θ < π) : ℝ :=
  arg (conj z)

theorem principal_argument_conjugate (θ : ℝ) (hθ1 : π / 2 < θ) (hθ2 : θ < π) :
  let z := 1 - sin θ + I * cos θ in
  principal_arg_of_conjugate z θ (by sorry) hθ1 hθ2 = (3 * π / 4) - (θ / 2) :=
sorry

end principal_argument_conjugate_l510_510018


namespace five_letter_words_start_end_same_l510_510083

def num_five_letter_words_start_end_same : ℕ :=
  26 ^ 4

theorem five_letter_words_start_end_same :
  num_five_letter_words_start_end_same = 456976 :=
by
  -- Sorry is used as a placeholder for the proof.
  sorry

end five_letter_words_start_end_same_l510_510083


namespace probability_of_exactly_5_calls_probability_of_no_more_than_4_calls_probability_of_at_least_3_calls_l510_510278

noncomputable def number_of_subscribers : ℕ := 400
noncomputable def prob_of_call : ℝ := 0.01
noncomputable def mean_calls : ℝ := number_of_subscribers * prob_of_call

theorem probability_of_exactly_5_calls :
  Probability.Poisson.mean mean_calls (5) = 0.1563 :=
sorry

theorem probability_of_no_more_than_4_calls :
  (Probability.Poisson.mean mean_calls (0) +
   Probability.Poisson.mean mean_calls (1) +
   Probability.Poisson.mean mean_calls (2) +
   Probability.Poisson.mean mean_calls (3) +
   Probability.Poisson.mean mean_calls (4)) = 0.6289 :=
sorry

theorem probability_of_at_least_3_calls :
  (1 - (Probability.Poisson.mean mean_calls (0) +
       Probability.Poisson.mean mean_calls (1) +
       Probability.Poisson.mean mean_calls (2))) = 0.7619 :=
sorry

end probability_of_exactly_5_calls_probability_of_no_more_than_4_calls_probability_of_at_least_3_calls_l510_510278


namespace concyclic_points_l510_510815

open EuclideanGeometry

variable (A B C E F P D M: Point)
variable {Γ : Circle}
variable [incircle Γ B C]

-- Given conditions based on problem setup
variable (h1 : Γ ∈ Circle A B C)
variable (h2 : E ∈ AC)
variable (h3 : F ∈ AB)
variable (h4 : BE ∩ CF = P)
variable (h5 : AP ∩ BC = D)
variable (h6 : midpoint M B C)

theorem concyclic_points (h : on_circle Γ B ∧ on_circle Γ C ∧ on_circle Γ E ∧ on_circle Γ F) : concyclic D M E F :=
sorry

end concyclic_points_l510_510815


namespace bob_tilling_problem_l510_510304

noncomputable def bob_tilling_time (base1 base2 height tiller_width tiller_length tiller_time obstacle_areas extra_time num_obstacles : ℝ) : ℝ :=
  let trapezoid_area := (base1 + base2) / 2 * height
  let total_obstacle_area := obstacle_areas.sum
  let effective_area := trapezoid_area - total_obstacle_area
  let tiller_area_per_time := tiller_width * tiller_length
  let intervals := effective_area / tiller_area_per_time
  let total_time := intervals * tiller_time
  let total_time_with_obstacles := total_time + num_obstacles * extra_time
  total_time_with_obstacles / 60

theorem bob_tilling_problem : 
  bob_tilling_time 135 170 90 2.5 1.5 3 [200, 450, 150] 15 3 ≈ 173.08 := 
by
  sorry

end bob_tilling_problem_l510_510304


namespace range_of_x_l510_510770

-- Define the ellipse equation
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  (x^2 / 9) + (y^2 / 4) = 1

-- Define the foci of the ellipse
def F1 := (-Real.sqrt 5, 0)
def F2 := (Real.sqrt 5, 0)

-- Check if the angle F1 P F2 is obtuse
def obtuse_angle_condition (P : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  ((x + Real.sqrt 5)^2 + y^2) + ((x - Real.sqrt 5)^2 + y^2) < 20

-- Prove the range of x when the angle F1 P F2 is obtuse and P is on the ellipse
theorem range_of_x (P : ℝ × ℝ) (hx : is_on_ellipse P) (h_obtuse : obtuse_angle_condition P) :
  - (3 * Real.sqrt 5) / 5 < P.1 ∧ P.1 < (3 * Real.sqrt 5) / 5 :=
by 
  sorry

end range_of_x_l510_510770


namespace medians_concurrent_at_centroid_l510_510143

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C G M : V)

-- assumption of G being the centroid implies medians intersect at G
def is_centroid (A B C G : V) : Prop :=
  (G +ᵥ (G +ᵥ (G +ᵥ -A))) = (0 : V)

-- Given definition of medians are lines from vertex to midpoint of opposite side
def is_median (A B C M : V) : Prop :=
  M = (B +ᵥ C) / 2 ∨ M = (A +ᵥ C) / 2 ∨ M = (A +ᵥ B) / 2

-- Proof Sketch: Need to show that if G is centroid, then condition fulfills
theorem medians_concurrent_at_centroid
  (hG : is_centroid A B C G) (hM : is_median A B C M) :
  \0 = (G +ᵥ (G +ᵥ (G +ᵥ -A))) :=
sorry

end medians_concurrent_at_centroid_l510_510143


namespace divisor_count_condition_num_of_divisors_l510_510898

theorem divisor_count_condition (a b: ℤ) (h1: 5 * b = 12 - 3 * a) :
  ∀ n ∈ [1, 2, 3, 4, 5], n ∣ (3 * b + 15) ↔ n = 1 ∨ n = 3 ∨ n = 5 :=
by sorry

theorem num_of_divisors (a b: ℤ) (h1: 5 * b = 12 - 3 * a) :
  2 = (∑ n in {1, 2, 3, 4}.to_finset, if n ∣ (3 * b + 15) then 1 else 0) :=
by sorry

end divisor_count_condition_num_of_divisors_l510_510898


namespace ship_sails_straight_line_l510_510649

-- Define third-degree polynomial
def third_degree_poly (f : ℝ → ℝ) : Prop :=
∃ (a₃ a₂ a₁ a₀ : ℝ), ∀ t : ℝ, f t = a₃ * t^3 + a₂ * t^2 + a₁ * t + a₀

theorem ship_sails_straight_line (f g : ℝ → ℝ) 
  (hf : third_degree_poly f) (hg : third_degree_poly g)
  (h1 : f 14 = f 13) (h2 : g 14 = g 13)
  (h3 : f 20 = f 19) (h4 : g 20 = g 19) : 
  ∃ λ : ℝ, ∃ C : ℝ, ∀ t : ℝ, f t = λ * g t + C := 
by
  sorry

end ship_sails_straight_line_l510_510649


namespace chi_squared_significance_l510_510050

theorem chi_squared_significance :
  ∀ (chi_squared : ℝ), chi_squared ≈ 13.097 → chi_squared > 10.828 → mistake_probability ≤ 0.001 :=
by
  assume (chi_squared : ℝ)
  assume h_chi_squared : chi_squared ≈ 13.097
  assume h_threshold : chi_squared > 10.828
  sorry

end chi_squared_significance_l510_510050


namespace average_episodes_per_year_is_16_l510_510613

-- Define the number of years the TV show has been running
def years : Nat := 14

-- Define the number of seasons and episodes for each category
def seasons_8_15 : Nat := 8
def episodes_per_season_8_15 : Nat := 15
def seasons_4_20 : Nat := 4
def episodes_per_season_4_20 : Nat := 20
def seasons_2_12 : Nat := 2
def episodes_per_season_2_12 : Nat := 12

-- Define the total number of episodes
def total_episodes : Nat :=
  (seasons_8_15 * episodes_per_season_8_15) + 
  (seasons_4_20 * episodes_per_season_4_20) + 
  (seasons_2_12 * episodes_per_season_2_12)

-- Define the average number of episodes per year
def average_episodes_per_year : Nat :=
  total_episodes / years

-- State the theorem to prove the average number of episodes per year is 16
theorem average_episodes_per_year_is_16 : average_episodes_per_year = 16 :=
by
  sorry

end average_episodes_per_year_is_16_l510_510613


namespace sum_of_roots_tan_quadratic_l510_510352

open Real

theorem sum_of_roots_tan_quadratic :
  (∑ x in {x : ℝ | 0 ≤ x ∧ x < 2 * π ∧ (tan x) * (tan x) - 13 * (tan x) + 4 = 0}, x) = 3 * π :=
by 
  sorry

end sum_of_roots_tan_quadratic_l510_510352


namespace geometric_sequence_common_ratio_l510_510459

theorem geometric_sequence_common_ratio (q : ℝ) (a : ℕ → ℝ) 
  (h1 : a 2 = 1/2)
  (h2 : a 5 = 4)
  (h3 : ∀ n, a n = a 1 * q^(n - 1)) : 
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l510_510459


namespace find_b_l510_510485

namespace VectorProof

def a : ℝ × ℝ × ℝ := (3, 2, 4)

def satisfies_conditions (b : ℝ × ℝ × ℝ) : Prop :=
  let (bx, by, bz) := b in
  (3 * bx + 2 * by + 4 * bz = 20) ∧ 
  ((2 * bz - 4 * by) = -8) ∧ 
  ((4 * bx - 3 * bz) = 5) ∧ 
  ((3 * by - 2 * bx) = 1)

theorem find_b :
  (∃ b : ℝ × ℝ × ℝ, satisfies_conditions b) :=
begin
  use (-17/4, -5/3, -22/3),
  simp [satisfies_conditions],
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { norm_num, }
end

end VectorProof

end find_b_l510_510485


namespace circumcircles_intersect_at_point_on_BC_l510_510378

variables {A B C M N R O : Type*}
variables [Point A] [Point B] [Point C] [Point M] [Point N] [Point R] [Point O]

-- Assume the definitions of triangle, midpoint, altitude, bisectors, and circumcircle are predefined.
-- You would need to customize these to your suited definitions.

-- Definitions assumed here
variable [Triangle ABC : Triangle A B C]
variable [Altitude BN : Altitude B N]
variable [Altitude CM : Altitude C M]
variable [Midpoint OBC : Midpoint O B C]
variable [AngleBisector AR : AngleBisector A R]
variable [AngleBisector MR : AngleBisector M R]
variable [Circumcircle BMR : CircumcircleTriangle B M R]
variable [Circumcircle CNR : CircumcircleTriangle C N R]

theorem circumcircles_intersect_at_point_on_BC :
  ∃ P : Point, P ∈ BC ∧ P ∈ CircumcircleTriangle B M R ∧ P ∈ CircumcircleTriangle C N R := sorry

end circumcircles_intersect_at_point_on_BC_l510_510378


namespace sum_of_reciprocals_roots_transformed_eq_neg11_div_4_l510_510099

theorem sum_of_reciprocals_roots_transformed_eq_neg11_div_4 :
  (∃ a b c : ℝ, (a^3 - a - 2 = 0) ∧ (b^3 - b - 2 = 0) ∧ (c^3 - c - 2 = 0)) → 
  ( ∃ a b c : ℝ, a^3 - a - 2 = 0 ∧ b^3 - b - 2 = 0 ∧ c^3 - c - 2 = 0 ∧ 
  (1 / (a - 2) + 1 / (b - 2) + 1 / (c - 2) = - 11 / 4)) :=
by
  sorry

end sum_of_reciprocals_roots_transformed_eq_neg11_div_4_l510_510099


namespace find_scaling_matrix_l510_510344

open Matrix

theorem find_scaling_matrix (a b c d : ℕ) :
  ∃ (M : Matrix (Fin 2) (Fin 2) ℕ), M ⬝ !![![a, b], ![c, d]] = !![![2 * a, 2 * b], ![3 * c, 3 * d]] :=
  let M : Matrix (Fin 2) (Fin 2) ℕ := !![![2, 0], ![0, 3]]
  in ⟨M, by simp [Matrix.mul]⟩

end find_scaling_matrix_l510_510344


namespace largest_prime_factor_with_more_than_two_distinct_prime_factors_l510_510706

def hasMoreThanTwoDistinctPrimeFactors (n : ℕ) : Prop :=
  (nat.factors n).toFinset.card > 2

theorem largest_prime_factor_with_more_than_two_distinct_prime_factors :
  (∀ n ∈ [ 105, 142, 165, 187, 221],
    hasMoreThanTwoDistinctPrimeFactors n → ∀ m, m ∣ n → nat.prime m → m ≤ 11) ∧
  hasMoreThanTwoDistinctPrimeFactors 165 ∧ (∀ m, m ∣ 165 → nat.prime m → m ≤ 11) :=
sorry

end largest_prime_factor_with_more_than_two_distinct_prime_factors_l510_510706


namespace equilateral_triangle_concyclic_midpoints_l510_510004

variables (A B C O D E : Point)

def is_equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

noncomputable def midpoint (P Q : Point) : Point :=
  sorry  -- Definition of midpoint, based on specific geometric construction

noncomputable def is_circumcircle (X Y Z P : Point) : Prop :=
  sorry  -- Definition of circumcircle, based on specific geometric properties

noncomputable def is_concyclic (X Y Z W : Point) : Prop :=
  sorry  -- Definition of concyclic points, based on specific geometric properties

theorem equilateral_triangle_concyclic_midpoints
  (h_eq : is_equilateral_triangle A B C)
  (h_center : O = centroid A B C)
  (h_inter : ∃ D E, line_through C ∧ circumcircle A O B D E) :
  let D' := midpoint B D in
  let E' := midpoint B E in
  is_concyclic A O D' E' :=
by
  sorry  -- Proof to be filled in later

end equilateral_triangle_concyclic_midpoints_l510_510004


namespace onlyD_is_PythagoreanTriple_l510_510218

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def validTripleA := ¬ isPythagoreanTriple 12 15 18
def validTripleB := isPythagoreanTriple 3 4 5 ∧ (¬ (3 = 3 ∧ 4 = 4 ∧ 5 = 5)) -- Since 0.3, 0.4, 0.5 not integers
def validTripleC := ¬ isPythagoreanTriple 15 25 30 -- Conversion of 1.5, 2.5, 3 to integers
def validTripleD := isPythagoreanTriple 12 16 20

theorem onlyD_is_PythagoreanTriple : validTripleA ∧ validTripleB ∧ validTripleC ∧ validTripleD :=
by {
  sorry
}

end onlyD_is_PythagoreanTriple_l510_510218


namespace interesting_sets_l510_510862

theorem interesting_sets (p : ℕ) (hp : Nat.Prime p) (S : Finset ℕ) (hS : S.card = p + 2) 
    (hdiv : ∀ (x y : ℕ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x ≠ y), 
      let T := S.erase x in
      let U := T.erase y in 
      T.sum % y = 0 ∧ T.sum % x = 0) : 
    ∃ (m : ℕ), S = Finset.replicate (p+2) m ∨ S = Finset.insert (p * m) (Finset.replicate p m) :=
sorry

end interesting_sets_l510_510862


namespace sum_of_cubes_eq_square_of_sum_l510_510124

theorem sum_of_cubes_eq_square_of_sum (n : ℕ) : 
  (∑ k in Finset.range (n + 1), k^3) = ((∑ k in Finset.range (n + 1), k) ^ 2) := 
sorry

end sum_of_cubes_eq_square_of_sum_l510_510124


namespace find_a_l510_510010

def complexNumberPurelyImaginary (a : ℝ) : Prop := 
  ∃ (b : ℝ), (a^2 - 3*a +2) + b * Complex.i = b * Complex.i

theorem find_a (a : ℝ) : complexNumberPurelyImaginary a → a = 2 := 
sorry

end find_a_l510_510010


namespace petya_wins_if_and_only_if_odd_l510_510936

theorem petya_wins_if_and_only_if_odd (n : ℕ) : (odd n ↔ ∃ win_strategy : (ℕ → Prop), win_strategy n) :=
by sorry

end petya_wins_if_and_only_if_odd_l510_510936


namespace rose_joined_after_six_months_l510_510475

noncomputable def profit_shares (m : ℕ) : ℕ :=
  12000 * (12 - m) - 9000 * 8

theorem rose_joined_after_six_months :
  ∃ (m : ℕ), profit_shares m = 370 :=
by
  use 6
  unfold profit_shares
  norm_num
  sorry

end rose_joined_after_six_months_l510_510475


namespace percent_profit_l510_510045

theorem percent_profit (C S : ℝ) (h : 58 * C = 50 * S) : 
  (S - C) / C * 100 = 16 :=
by
  sorry

end percent_profit_l510_510045


namespace fraction_males_on_time_l510_510673

variable (attendees: ℕ) (fraction_males: ℚ) (fraction_females_on_time: ℚ) (fraction_not_on_time: ℚ) 

def attendees_who_arrived_on_time (attendees: ℕ) (fraction_not_on_time: ℚ) : ℕ := 
  attendees - fraction_not_on_time * attendees

def male_attendees (attendees: ℕ) (fraction_males: ℚ) : ℕ := 
  fraction_males * attendees

def female_attendees (attendees: ℕ) (fraction_females: ℚ) : ℕ := 
  fraction_females * attendees

def female_arrived_on_time (female_attendees: ℕ) (fraction_females_on_time: ℚ) : ℕ := 
  fraction_females_on_time * female_attendees

def male_arrived_on_time (total_arrived_on_time: ℕ) (female_arrived_on_time: ℕ) : ℕ := 
  total_arrived_on_time - female_arrived_on_time

theorem fraction_males_on_time : 
  (fraction_males = 3/5) → 
  (fraction_females_on_time = 5/6) → 
  (fraction_not_on_time = 0.18666666666666662) → 
  (attendees = 30) → 
  (fraction (male_arrived_on_time (attendees_who_arrived_on_time attendees fraction_not_on_time) (female_arrived_on_time (female_attendees attendees (2/5)) fraction_females_on_time)) (male_attendees attendees fraction_males) = 7/9) :=
by 
  sorry

end fraction_males_on_time_l510_510673


namespace sum_div_prod_eq_nat_div_nat_add_one_l510_510882

noncomputable def sumDivProd (n : ℕ) : ℚ :=
  ∑ k in Finset.range n + 1, 1 / ((k + 1 : ℕ) * (k + 2 : ℕ))

theorem sum_div_prod_eq_nat_div_nat_add_one (n : ℕ) : sumDivProd n = (n : ℚ) / (n + 1 : ℚ) :=
  sorry

end sum_div_prod_eq_nat_div_nat_add_one_l510_510882


namespace intersection_of_A_and_B_l510_510783

-- Definitions of sets A and B
def set_A : Set ℝ := { x | x^2 - x - 6 < 0 }
def set_B : Set ℝ := { x | (x + 4) * (x - 2) > 0 }

-- Theorem statement for the intersection of A and B
theorem intersection_of_A_and_B : set_A ∩ set_B = { x | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_of_A_and_B_l510_510783


namespace prime_divisor_of_form_l510_510250

theorem prime_divisor_of_form (a p : ℕ) (hp1 : a > 0) (hp2 : Prime p) (hp3 : p ∣ (a^3 - 3 * a + 1)) (hp4 : p ≠ 3) :
  ∃ k : ℤ, p = 9 * k + 1 ∨ p = 9 * k - 1 :=
by
  sorry

end prime_divisor_of_form_l510_510250


namespace min_positive_period_f_l510_510402

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 3)

theorem min_positive_period_f : (∃ p > 0, ∀ x, f (x + p) = f x) ∧
  ((∀ x, ℝ) ((pi/12) < x ∧ x < (7 * pi/12) → Real.tan (2 * x + Real.pi / 3) < Real.tan (2 * (x + 1) + Real.pi / 3))) :=
by 
  sorry

end min_positive_period_f_l510_510402


namespace find_y_l510_510600

theorem find_y : ∃ y : ℤ, (55 + 48 + 507 + 2 + 684 + y) / 6 = 223 ∧ y = 42 := by 
sorzyć
end find_y_l510_510600


namespace max_xy_value_l510_510760

theorem max_xy_value (x y : ℕ) (h : 27 * x + 35 * y ≤ 1000) : x * y ≤ 252 :=
sorry

end max_xy_value_l510_510760


namespace min_value_9x_plus_3y_l510_510494

theorem min_value_9x_plus_3y :
  ∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 2 * x + y = 6 → 9^x + 3^y = 54 :=
by
  intro x y
  intro h
  cases h with hx hxy
  cases hxy with hy hxy_eq
  sorry

end min_value_9x_plus_3y_l510_510494


namespace intersection_A_and_B_l510_510426

-- Define the sets based on the conditions
def setA : Set ℤ := {x : ℤ | x^2 - 2 * x - 8 ≤ 0}
def setB : Set ℤ := {x : ℤ | 1 < Real.log x / Real.log 2}

-- State the theorem (Note: The logarithmic condition should translate the values to integers)
theorem intersection_A_and_B : setA ∩ setB = {3, 4} :=
sorry

end intersection_A_and_B_l510_510426


namespace karen_age_is_10_l510_510477

-- Definitions for the given conditions
def ages : List ℕ := [2, 4, 6, 8, 10, 12, 14]

def to_park (a b : ℕ) : Prop := a + b = 20
def to_pool (a b : ℕ) : Prop := 3 < a ∧ a < 9 ∧ 3 < b ∧ b < 9
def stayed_home (karen_age : ℕ) : Prop := karen_age = 10

-- Theorem stating Karen's age is 10 given the conditions
theorem karen_age_is_10 :
  ∃ (a b c d e f g : ℕ),
  ages = [a, b, c, d, e, f, g] ∧
  ((to_park a b ∨ to_park a c ∨ to_park a d ∨ to_park a e ∨ to_park a f ∨ to_park a g ∨
  to_park b c ∨ to_park b d ∨ to_park b e ∨ to_park b f ∨ to_park b g ∨
  to_park c d ∨ to_park c e ∨ to_park c f ∨ to_park c g ∨
  to_park d e ∨ to_park d f ∨ to_park d g ∨
  to_park e f ∨ to_park e g ∨
  to_park f g)) ∧
  ((to_pool a b ∨ to_pool a c ∨ to_pool a d ∨ to_pool a e ∨ to_pool a f ∨ to_pool a g ∨
  to_pool b c ∨ to_pool b d ∨ to_pool b e ∨ to_pool b f ∨ to_pool b g ∨
  to_pool c d ∨ to_pool c e ∨ to_pool c f ∨
  to_pool d e ∨ to_pool d f ∨
  to_pool e f ∨
  to_pool f g)) ∧
  stayed_home 4 :=
sorry

end karen_age_is_10_l510_510477


namespace intersection_points_are_integers_l510_510859

theorem intersection_points_are_integers :
  ∀ (a b : Fin 2021 → ℕ), Function.Injective a → Function.Injective b →
  ∀ i j, i ≠ j → 
  ∃ x : ℤ, (∃ y : ℚ, y = (a i : ℚ) / (x + (b i : ℚ))) ∧ 
           (∃ y : ℚ, y = (a j : ℚ) / (x + (b j : ℚ))) := 
sorry

end intersection_points_are_integers_l510_510859


namespace part_I_part_II_l510_510816

variables {A B C a b c : ℝ}

-- Conditions
axiom triangle_abc (a b c : ℝ) (h : a^2 + c^2 - b^2 = (1/2) * a * c) : Prop

-- (I) Prove \(\sin^2 \dfrac{A+C}{2} + \cos 2B = -\dfrac{1}{4}\).
theorem part_I (h : triangle_abc a b c) :
  sin^2 ((A + C) / 2) + cos (2 * B) = -1/4 :=
begin
  sorry,
end

-- (II) Prove the maximum area \(S_{\triangle ABC} = \dfrac{\sqrt{15}}{3}\) when \(b = 2\) and \(a^2 + c^2 - b^2 = \dfrac{1}{2}ac\).
theorem part_II (h : triangle_abc a 2 c) :
  let S := (1/2) * a * c * sin B in S ≤ sqrt 15 / 3 :=
begin
  sorry,
end

end part_I_part_II_l510_510816


namespace find_number_l510_510265

theorem find_number
  (x a b c : ℕ)
  (h1 : x * a = 494)
  (h2 : x * b = 988)
  (h3 : x * c = 1729) :
  x = 247 :=
sorry

end find_number_l510_510265


namespace seating_arrangements_l510_510871

theorem seating_arrangements :
  let num_children := 3 in
  let num_family_members := 2 + num_children in
  let driver_choices := 2 in
  let front_row_choices := Nat.choose (num_family_members - 1) 2 in
  let front_row_arrangements := front_row_choices * Nat.factorial 2 in
  let back_row_arrangements := Nat.factorial (num_family_members - 2) in
  driver_choices * front_row_arrangements * back_row_arrangements = 240 :=
by
  sorry

end seating_arrangements_l510_510871


namespace bee_returns_starting_position_l510_510998

theorem bee_returns_starting_position (N : ℕ) (hN : N ≥ 3) :
  ∃ (flights : list (ℤ × ℤ)), (flight_sums_to_zero N flights) :=
sorry

-- Additional definitions and auxiliary functions that might be necessary
def hex_neighbor_offsets : list (ℤ × ℤ) :=
  [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

def flight_sums_to_zero (N : ℕ) (flights : list (ℤ × ℤ)) : Prop :=
  flights.length = N ∧ 
  list.sum (flights.take N) = (0, 0) ∧
  ∀ (i : ℕ), (i < N) → 
    (flights.nth i).is_some ∧ 
    ((flights.nth i).get ∈ hex_neighbor_offsets)

end bee_returns_starting_position_l510_510998


namespace tangent_line_eq_l510_510715

def curve (x : ℝ) : ℝ := x + Real.sin x

theorem tangent_line_eq (x y : ℝ) (h : y = 2 * x) :
  ∃ t : ℝ, curve t = 0 ∧ t = 0 ∧ (∃ k : ℝ, k = 2 ∧ ∀ x, y = 2 * x) :=
sorry

end tangent_line_eq_l510_510715


namespace smallest_x_satisfies_sqrt3x_eq_5x_l510_510964

theorem smallest_x_satisfies_sqrt3x_eq_5x :
  ∃ x : ℝ, (sqrt (3 * x) = 5 * x) ∧ (∀ y : ℝ, sqrt (3 * y) = 5 * y → x ≤ y) ∧ (x = 0) :=
sorry

end smallest_x_satisfies_sqrt3x_eq_5x_l510_510964


namespace exists_periodic_k_l510_510860

def func (A : Type) := A → A

def a_sequence {A : Type} (f : func (Fin 2007)) : ℕ → Fin 2007
| 1     := f 1
| (n+1) := f (a_sequence n)

theorem exists_periodic_k (f : func (Fin 2007)) :
  ∃ k : ℕ+, a_sequence f (2 * k) = a_sequence f k :=
sorry

end exists_periodic_k_l510_510860


namespace sequence_formula_l510_510416

noncomputable def a : ℕ+ → ℚ
| ⟨1, _⟩ => 1 / 2
| ⟨n + 1, _⟩ => a ⟨n, Nat.succ_pos _⟩ / (1 + 2 * a ⟨n, Nat.succ_pos _⟩)

theorem sequence_formula (n : ℕ+) : a n = 1 / (2 * n) :=
by
  sorry

end sequence_formula_l510_510416


namespace min_value_expression_l510_510112

open Real

theorem min_value_expression (x y z: ℝ) (h1: 0 < x) (h2: 0 < y) (h3: 0 < z)
    (h4: (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10):
    (x / y + y / z + z / x) * (y / x + z / y + x / z) = 25 :=
by
  sorry

end min_value_expression_l510_510112


namespace radius_of_fourth_circle_l510_510950

theorem radius_of_fourth_circle (r R : ℝ) (h1 : R = 13) (h2 : r = 23) :
  (π * (23 ^ 2 - 13 ^ 2) = π * (6 * sqrt 10) ^ 2) :=
by
  sorry

end radius_of_fourth_circle_l510_510950


namespace regression_decrease_l510_510781

theorem regression_decrease (x : ℝ) : (let y = 7 - 3 * x in let y_new = 7 - 3 * (x + 2) in y_new = y - 6) :=
by
  let y := 7 - 3 * x
  let y_new := 7 - 3 * (x + 2)
  sorry

end regression_decrease_l510_510781


namespace elizabeth_needs_to_borrow_more_money_l510_510326

-- Define the costs of the items
def pencil_cost : ℝ := 6.00 
def notebook_cost : ℝ := 3.50 
def pen_cost : ℝ := 2.25 

-- Define the amount of money Elizabeth initially has and what she borrowed
def elizabeth_money : ℝ := 5.00 
def borrowed_money : ℝ := 0.53 

-- Define the total cost of the items
def total_cost : ℝ := pencil_cost + notebook_cost + pen_cost

-- Define the total amount of money Elizabeth has
def total_money : ℝ := elizabeth_money + borrowed_money

-- Define the additional amount Elizabeth needs to borrow
def amount_needed_to_borrow : ℝ := total_cost - total_money

-- The theorem to prove that Elizabeth needs to borrow an additional $6.22
theorem elizabeth_needs_to_borrow_more_money : 
  amount_needed_to_borrow = 6.22 := by 
    -- Proof goes here
    sorry

end elizabeth_needs_to_borrow_more_money_l510_510326


namespace intersection_of_diagonals_on_XO_l510_510886

theorem intersection_of_diagonals_on_XO
  (A B C D O X : Point)
  (hABCD_circle : InscribedQuadrilateral A B C D O)
  (hBAX : angle A B X = 90)
  (hCDX : angle C D X = 90) :
  Collinear [O, X, intersection_of_diagonals A B C D] :=
sorry

end intersection_of_diagonals_on_XO_l510_510886


namespace negation_of_triangle_angle_sum_negation_of_absolute_value_l510_510972

theorem negation_of_triangle_angle_sum (h : ∃ (T : Type) [triangle T], sum_angle T ≠ 180) :
  ∀ (T : Type) [triangle T], sum_angle T = 180 :=
sorry

theorem negation_of_absolute_value (h : ∀ (x : ℝ), |x| + x^2 ≥ 0) :
  ∃ (x : ℝ), |x| + x^2 < 0 :=
sorry

end negation_of_triangle_angle_sum_negation_of_absolute_value_l510_510972


namespace tom_average_speed_l510_510982

theorem tom_average_speed 
  (d1 d2 : ℝ) (s1 s2 t1 t2 : ℝ)
  (h_d1 : d1 = 30) 
  (h_d2 : d2 = 50) 
  (h_s1 : s1 = 30) 
  (h_s2 : s2 = 50) 
  (h_t1 : t1 = d1 / s1) 
  (h_t2 : t2 = d2 / s2)
  (h_total_distance : d1 + d2 = 80) 
  (h_total_time : t1 + t2 = 2) :
  (d1 + d2) / (t1 + t2) = 40 := 
by {
  sorry
}

end tom_average_speed_l510_510982


namespace greatest_volume_of_pyramid_l510_510206

noncomputable def max_pyramid_volume (AB AC : ℝ) (sin_BAC : ℝ) (angle_limit : ℝ) : ℝ :=
  if AB = 3 ∧ AC = 5 ∧ sin_BAC = 4/5 ∧ angle_limit = π / 3 then 5 * Real.sqrt 39 / 2 else 0

theorem greatest_volume_of_pyramid :
  let AB := 3
  let AC := 5
  let sin_BAC := 4/5
  let angle_limit := π / 3
  max_pyramid_volume AB AC sin_BAC angle_limit = 5 * Real.sqrt 39 / 2 := by 
  sorry

end greatest_volume_of_pyramid_l510_510206


namespace first_course_cost_l510_510293

theorem first_course_cost (x : ℝ) (h1 : 60 - (x + (x + 5) + 0.25 * (x + 5)) = 20) : x = 15 :=
by sorry

end first_course_cost_l510_510293


namespace area_of_remaining_rectangle_l510_510269

axiom area_rect_HILK : ℕ := 40
axiom area_rect_DEIH : ℕ := 20
axiom area_rect_ABHG : ℕ := 126
axiom area_rect_GHKJ : ℕ := 63
axiom area_large_rect_DFMK : ℕ := 161

noncomputable def area_rect_EFML : ℕ :=
area_large_rect_DFMK - area_rect_DEIH - area_rect_HILK

theorem area_of_remaining_rectangle : area_rect_EFML = 101 :=
by
  rw [area_rect_EFML]
  sorry

end area_of_remaining_rectangle_l510_510269


namespace polynomial_root_l510_510350

noncomputable def positive_root : Real :=
  3 - Real.sqrt 3

theorem polynomial_root (x : Real) : 
  (x^3 - 4 * x^2 - 2 * x - Real.sqrt 3 = 0) ∧ (x > 0) :=
begin
  use positive_root,
  split,
  { sorry },  -- Proof that 3 - sqrt(3) is a root of the polynomial
  { sorry }   -- Proof that 3 - sqrt(3) is positive
end

end polynomial_root_l510_510350


namespace junior_prom_total_kids_l510_510177

-- Definition of conditions
def total_kids_in_junior_prom := Nat
def one_fourth (P : Nat) := P / 4
def total_dancers (P : Nat) := 35

-- Theorem statement
theorem junior_prom_total_kids (P : total_kids_in_junior_prom) :
  one_fourth P = total_dancers P → P = 140 := 
by
  intros h
  sorry

end junior_prom_total_kids_l510_510177


namespace houses_order_count_l510_510131

theorem houses_order_count :
  ∃ n : ℕ, n = 4 ∧ 
  (∀ (order : list string), order.perm ["G", "P", "R", "B"] →
    ((list.index_of "G" order < list.index_of "P" order) ∧
     (list.index_of "R" order < list.index_of "B" order) ∧
     (abs (list.index_of "R" order - list.index_of "G" order) = 1))
     → order ∈ permutations_with_conditions) :=
sorry

end houses_order_count_l510_510131


namespace best_fault_locating_method_l510_510559

-- Define the conditions as variables.
variable (A B : Prop)

-- Define the condition stating that there is electricity at point A.
axiom A_has_electricity : A

-- Define the condition stating that there is no electricity at point B.
axiom B_has_no_electricity : ¬ B

-- Define the problem statement.
theorem best_fault_locating_method : A ∧ (¬ B) → "Bisection method" = "best method to locate the fault" := 
by
  intro h,
  sorry -- Proof goes here.

end best_fault_locating_method_l510_510559


namespace tangent_line_equation_and_max_area_parallelogram_PQA_min_value_of_rotated_line_l510_510779

noncomputable def parabola (x : ℝ) : ℝ := - (3 / 8) * x^2 + (3 / 4) * x + 3

def line_bc (x : ℝ) : ℝ := - (3 / 4) * x + 3

def point_C : (ℝ × ℝ) := (4, 0)
def point_B : (ℝ × ℝ) := (0, 3)
def point_A : (ℝ × ℝ) := (-2, 0)

theorem tangent_line_equation_and_max_area :
  ∃ E : ℝ × ℝ, E = (2, parabola 2) ∧ 
  (∀ x, line_bc x = -(3 / 4) * x + 3 → parabola x = parabola 2 ∧
  ∃ l : ℝ → ℝ, l x = -(3 / 4) * x + 9 / 2 ∧ ∃ S, S = 3) :=
by
  exists (2, 3)
  simp [parabola]
  sorry

theorem parallelogram_PQA :
  ∃ P : ℝ × ℝ, (P = (5, -(21 / 8)) ∨ P = (-3, -(21 / 8)) ∨ P = (-1, 15 / 8)) :=
by 
  sorry

theorem min_value_of_rotated_line:
  ∃ α : ℝ, 0 < α ∧ α < π / 2 → 
  ∃ N' : ℝ × ℝ, N' = (cos α, sin α) → 
  ∃ min_val, min_val = sqrt (145) / 3 :=
by 
  sorry

end tangent_line_equation_and_max_area_parallelogram_PQA_min_value_of_rotated_line_l510_510779


namespace system_of_equations_l510_510828

-- Given conditions: Total number of fruits and total cost of the fruits purchased
def total_fruits := 1000
def total_cost := 999
def cost_of_sweet_fruit := (11 : ℚ) / 9
def cost_of_bitter_fruit := (4 : ℚ) / 7

-- Variables representing the number of sweet and bitter fruits
variables (x y : ℚ)

-- Problem statement in Lean 4
theorem system_of_equations :
  (x + y = total_fruits) ∧ (cost_of_sweet_fruit * x + cost_of_bitter_fruit * y = total_cost) ↔
  ((x + y = 1000) ∧ (11 / 9 * x + 4 / 7 * y = 999)) :=
by
  sorry

end system_of_equations_l510_510828


namespace find_m_l510_510390

theorem find_m (m : ℝ) (h1 : ∃ (α : ℝ), (∃ (x y : ℝ), x = -8 * m ∧ y = -3 ∧ (x^2 + y^2 > 0 ∧ cos α = x / sqrt (x^2 + y^2))) ∧ cos α = -4 / 5) : m = 1 / 2 :=
sorry

end find_m_l510_510390


namespace average_episodes_per_year_l510_510616

theorem average_episodes_per_year (total_years : ℕ) (n1 n2 n3 e1 e2 e3 : ℕ) 
  (h1 : total_years = 14)
  (h2 : n1 = 8) (h3 : e1 = 15)
  (h4 : n2 = 4) (h5 : e2 = 20)
  (h6 : n3 = 2) (h7 : e3 = 12) :
  (n1 * e1 + n2 * e2 + n3 * e3) / total_years = 16 := by
  -- Skip the proof steps
  sorry

end average_episodes_per_year_l510_510616


namespace percentage_relationship_l510_510041

theorem percentage_relationship (a b : ℝ) (h : a = 1.2 * b) : ¬ (b = 0.8 * a) :=
by
  -- assumption: a = 1.2 * b
  -- goal: ¬ (b = 0.8 * a)
  sorry

end percentage_relationship_l510_510041


namespace symmetric_polynomial_representation_l510_510100

noncomputable def f : ℝ × ℝ × ℝ → ℝ
noncomputable def g1 (x y z: ℝ) : ℝ := x * (x - y) * (x - z) + y * (y - z) * (y - x) + z * (z - x) * (z - y)
noncomputable def g2 (x y z: ℝ) : ℝ := (y + z) * (x - y) * (x - z) + (z + x) * (y - z) * (y - x) + (x + y) * (z - x) * (z - y)
noncomputable def g3 (x y z: ℝ) : ℝ := x * y * z

theorem symmetric_polynomial_representation 
  (f : ℝ × ℝ × ℝ → ℝ) 
  (h_sym : ∀ x y z : ℝ, f (x, y, z) = f (y, z, x)) 
  : ∃ (a b c : ℝ), (∀ x y z : ℝ, f (x, y, z) = a * g1 x y z + b * g2 x y z + c * g3 x y z) → 
    (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → f (x, y, z) ≥ 0 ↔ a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :=
begin
  sorry
end

end symmetric_polynomial_representation_l510_510100


namespace range_of_a_l510_510407

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x + 2) / (x - a) ≤ 2 ↔ x ∈ P) → (1 ∉ P) → a ∈ Icc (-1 / 2) 1 :=
sorry

end range_of_a_l510_510407


namespace smallest_positive_period_f_min_value_of_f_f_inverse_one_l510_510020

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos x * sin (x + π / 3) - sqrt 3 * sin x ^ 2 + sin x * cos x

-- 1. Prove the smallest positive period of f(x) is π.
theorem smallest_positive_period_f : ∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ ε > 0, ε < T → ∃ δ > 0, δ < ε → f (x + δ) != f x := 
  sorry

-- 2. Find the minimum value of f(x) and the corresponding values of x
theorem min_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f x = -2 ∧ ∃ k : ℤ, x = k * π - 5 * π / 12 :=
  sorry

-- 3. If the inverse function of \( f(x) \) exists in the specified interval, find \( f^{-1}(1) = \frac{\pi}{4} \).
theorem f_inverse_one : ∃ I : Set ℝ, I = Icc (π / 12) (7 * π / 12) ∧ (∀ x ∈ I, Function.Injective (fun x => f x)) ∧ Function.LeftInverse (fun y => f (y)) (fun x => x) 1 = π / 4 :=
  sorry

end smallest_positive_period_f_min_value_of_f_f_inverse_one_l510_510020


namespace water_cost_10_tons_water_cost_27_tons_water_cost_between_20_30_water_cost_above_30_l510_510178

-- Define the tiered water pricing function
def tiered_water_cost (m : ℕ) : ℝ :=
  if m ≤ 20 then
    1.6 * m
  else if m ≤ 30 then
    1.6 * 20 + 2.4 * (m - 20)
  else
    1.6 * 20 + 2.4 * 10 + 4.8 * (m - 30)

-- Problem 1
theorem water_cost_10_tons : tiered_water_cost 10 = 16 := 
sorry

-- Problem 2
theorem water_cost_27_tons : tiered_water_cost 27 = 48.8 := 
sorry

-- Problem 3
theorem water_cost_between_20_30 (m : ℕ) (h : 20 < m ∧ m < 30) : tiered_water_cost m = 2.4 * m - 16 := 
sorry

-- Problem 4
theorem water_cost_above_30 (m : ℕ) (h : m > 30) : tiered_water_cost m = 4.8 * m - 88 := 
sorry

end water_cost_10_tons_water_cost_27_tons_water_cost_between_20_30_water_cost_above_30_l510_510178


namespace log_value_eq_l510_510749

theorem log_value_eq (a b : ℝ) (h1 : 2^a = 3) (h2 : 3^b = 7) : log 7 56 = 1 + 3 / (a * b) :=
by
  sorry

end log_value_eq_l510_510749


namespace min_races_top_3_l510_510127

theorem min_races_top_3 (max_horses_per_race : ℕ) (total_horses : ℕ) (no_timer : Prop) 
    (max_horses_per_race_condition : max_horses_per_race = 5) (total_horses_condition : total_horses = 25) 
    (no_timing_condition : no_timer) : 
    (∃ min_races : ℕ, min_races = 7) :=
by
  use 7
  sorry

end min_races_top_3_l510_510127


namespace odd_function_f2_f3_max_min_l510_510736

variable (f : ℝ → ℝ)

-- Given conditions
axiom def_domain : ∀ x, ∃ k : ℤ, x ≠ k * Real.pi
axiom def_equation : ∀ x y, x ≠ k * Real.pi ∧ y ≠ k * Real.pi → f(x - y) = (f(x) * f(y) + 1) / (f(y) - f(x))
axiom def_f1 : f(1) = 1
axiom def_pos : ∀ x, 0 < x ∧ x < 2 → 0 < f(x)

-- Questions
theorem odd_function : ∀ x, f (-x) = -f x := by sorry
theorem f2_f3_max_min :
  f(2) = 0 ∧ f(3) = -1 ∧ (∀ x, 2 ≤ x ∧ x ≤ 3 → f(2) ≥ f(x) ∧ f(x) ≥ f(3)) := by sorry

end odd_function_f2_f3_max_min_l510_510736


namespace simplify_fraction_l510_510144

theorem simplify_fraction (x : ℝ) (h : x ≠ 0) : 
  (x ^ (3 / 4) - 25 * x ^ (1 / 4)) / (x ^ (1 / 2) + 5 * x ^ (1 / 4)) = x ^ (1 / 4) - 5 :=
by
  sorry

end simplify_fraction_l510_510144


namespace fraction_of_dutch_americans_with_window_seats_l510_510507

theorem fraction_of_dutch_americans_with_window_seats 
    (total_people : ℕ)
    (fraction_dutch : ℚ)
    (fraction_dutch_american : ℚ)
    (dutch_americans_with_window_seats : ℕ) : 
    total_people = 90 →
    fraction_dutch = 3/5 →
    fraction_dutch_american = 1/2 →
    dutch_americans_with_window_seats = 9 →
    (dutch_americans_with_window_seats : ℚ) / (fraction_dutch_american * fraction_dutch * total_people : ℚ) = 1/3 := 
by
  intros total_people_eq ninety_eq half fractional_eq nine_eq;
  sorry

end fraction_of_dutch_americans_with_window_seats_l510_510507


namespace range_of_m_l510_510438
open Real

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x ∈ Icc (-1) 1 ∧ x^2 - x - (m + 1) = 0) ↔ m ∈ Icc (-(5/4 : ℝ)) 1 :=
by
  sorry

end range_of_m_l510_510438


namespace positive_integers_in_inequality_l510_510552

theorem positive_integers_in_inequality (x : Int) : 1 < x ∧ x ≤ 4 → (x = 2 ∨ x = 3 ∨ x = 4) :=
by
  intro h
  cases h.1
  cases h.2
  sorry

end positive_integers_in_inequality_l510_510552


namespace increase_factor_is_46_8_l510_510574

-- Definitions for the conditions
def old_plates : ℕ := 26^3 * 10^3
def new_plates_type_A : ℕ := 26^2 * 10^4
def new_plates_type_B : ℕ := 26^4 * 10^2
def average_new_plates := (new_plates_type_A + new_plates_type_B) / 2

-- The Lean 4 statement to prove that the increase factor is 46.8
theorem increase_factor_is_46_8 :
  (average_new_plates : ℚ) / (old_plates : ℚ) = 46.8 := by
  sorry

end increase_factor_is_46_8_l510_510574


namespace inv_trig_values_l510_510171

theorem inv_trig_values :
  (Real.arctan (Real.sqrt 3) - Real.arcsin (-1 / 2) + Real.arccos 0 = Real.pi) := 
by 
  have h1 : Real.arctan (Real.sqrt 3) = Real.pi / 3 := sorry
  have h2 : Real.arcsin (-1 / 2) = -Real.pi / 6 := sorry
  have h3 : Real.arccos 0 = Real.pi / 2 := sorry
  calc
    Real.arctan (Real.sqrt 3) - Real.arcsin (-1 / 2) + Real.arccos 0
      = Real.pi / 3 - (-Real.pi / 6) + Real.pi / 2 : by rw [h1, h2, h3]
  ... = Real.pi : sorry

end inv_trig_values_l510_510171


namespace velocity_of_point_M_l510_510604

variable (L : ℝ) (ω : ℝ) (t : ℝ) (AB : ℝ)

def theta : ℝ := ω * t
def MB : ℝ := AB / 3
def velocity_m (t : ℝ) : ℝ := 300 * Real.sqrt (8 * Real.sin(θ t)^2 + 1)

theorem velocity_of_point_M 
  (hL : L = 90)
  (hω : ω = 10)
  (hAB : AB = L)
  (hMB : MB = AB / 3)
  (htheta : theta = ω * t) :
  velocity_m t = 300 * Real.sqrt (8 * Real.sin(10 * t)^2 + 1) := 
sorry

end velocity_of_point_M_l510_510604


namespace largest_m_such_that_six_divides_product_l510_510725

-- Definitions and conditions
def largest_power_of_largest_prime (n : ℕ) : ℕ :=
  if n ≥ 2 then 
    let p := (n.factorization.to_finset : _) in
    let q := p.max' sorry in -- p is nonempty since n >= 2
    q ^ n.factorization q
  else 1 -- we only care about n >= 2

-- The theorem statement
theorem largest_m_such_that_six_divides_product :
  ∃ m : ℕ, ∀ n : ℕ, 2 ≤ n ∧ n ≤ 100 →
  (6 ^ m ∣ ∏ k in (finset.range (100 - 1)).map (λ i, i + 2), largest_power_of_largest_prime k) ∧
  (∀ m' : ℕ, m' > m → ¬(6 ^ m' ∣ ∏ k in (finset.range (100 - 1)).map (λ i, i + 2), largest_power_of_largest_prime k)) ∧
  (m = 4) :=
begin
  sorry
end

end largest_m_such_that_six_divides_product_l510_510725


namespace evaluate_expression_l510_510705

theorem evaluate_expression : (25 + 15)^2 - (25^2 + 15^2 + 150) = 600 := by
  sorry

end evaluate_expression_l510_510705


namespace third_competitor_jump_difference_l510_510073

theorem third_competitor_jump_difference :
  ∀ (first second third fourth : ℕ),
    first = 22 →
    second = first + 1 →
    fourth = 24 →
    fourth = third + 3 →
    (second - third = 2) :=
by
  intros first second third fourth h_first h_second h_fourth1 h_fourth2
  subst h_first
  subst h_second
  subst h_fourth1
  subst h_fourth2
  exact eq.refl _ ⟩

end third_competitor_jump_difference_l510_510073


namespace tickets_to_be_sold_l510_510939

theorem tickets_to_be_sold (tickets_total : ℕ) (tickets_jude : ℕ) (tickets_andrea : ℕ) (tickets_sandra : ℕ) :
  tickets_total = 100 →
  tickets_jude = 16 →
  tickets_andrea = 2 * tickets_jude →
  tickets_sandra = (tickets_jude / 2) + 4 →
  tickets_total - (tickets_jude + tickets_andrea + tickets_sandra) = 40 :=
by {
  intros h_total h_jude h_andrea h_sandra,
  simp [h_total, h_jude, h_andrea, h_sandra],
  sorry
}

end tickets_to_be_sold_l510_510939


namespace julie_savings_multiple_l510_510846

theorem julie_savings_multiple (S : ℝ) (hS : 0 < S) :
  (12 * 0.25 * S) / (0.75 * S) = 4 :=
by
  sorry

end julie_savings_multiple_l510_510846


namespace pond_algae_coverage_at_24_l510_510529

noncomputable def algal_coverage (day : ℕ) : ℚ :=
  if day >= 28 then 1
  else let factor := (day % 2 = 0) in
    if factor then (1 / 3 : ℚ) ^ ((28 - day) / 2)
    else (1 / 3 : ℚ) ^ (((28 - day - 1) / 2) + 1)

theorem pond_algae_coverage_at_24 :
  algal_coverage 24 = 11.11 / 100 :=
sorry

end pond_algae_coverage_at_24_l510_510529


namespace hyperbola_eccentricity_l510_510437

theorem hyperbola_eccentricity
    (a b e : ℝ)
    (ha : a > 0)
    (hb : b > 0)
    (h_hyperbola : ∀ x y, x ^ 2 / a^2 - y^2 / b^2 = 1)
    (h_circle : ∀ x y, (x - 2) ^ 2 + y ^ 2 = 4)
    (h_chord_length : ∀ x y, (x ^ 2 + y ^ 2)^(1/2) = 2) :
    e = 2 := 
sorry

end hyperbola_eccentricity_l510_510437


namespace collective_apples_l510_510136

theorem collective_apples :
  let Pinky_apples := 36.5
  let Danny_apples := 73.2
  let Benny_apples := 48.8
  let Lucy_sales := 15.7
  (Pinky_apples + Danny_apples + Benny_apples - Lucy_sales) = 142.8 := by
  let Pinky_apples := 36.5
  let Danny_apples := 73.2
  let Benny_apples := 48.8
  let Lucy_sales := 15.7
  show (Pinky_apples + Danny_apples + Benny_apples - Lucy_sales) = 142.8
  sorry

end collective_apples_l510_510136


namespace general_formula_an_l510_510754

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

axiom h₁ : ∀ n : ℕ, 0 < n → 2 * S n = (a n) ^ 2 + a n
axiom h₂ : ∀ n : ℕ, 0 < n → S n = (finset.range n).sum a

theorem general_formula_an (n : ℕ) (hn : 0 < n) : a n = n := 
begin
  sorry
end

end general_formula_an_l510_510754


namespace nuts_in_mason_car_l510_510869

-- Define the constants for the rates of stockpiling
def busy_squirrel_rate := 30 -- nuts per day
def sleepy_squirrel_rate := 20 -- nuts per day
def days := 40 -- number of days
def num_busy_squirrels := 2 -- number of busy squirrels
def num_sleepy_squirrels := 1 -- number of sleepy squirrels

-- Define the total number of nuts
def total_nuts_in_mason_car : ℕ :=
  (num_busy_squirrels * busy_squirrel_rate * days) +
  (num_sleepy_squirrels * sleepy_squirrel_rate * days)

theorem nuts_in_mason_car :
  total_nuts_in_mason_car = 3200 :=
sorry

end nuts_in_mason_car_l510_510869


namespace smallest_n_exists_l510_510072

theorem smallest_n_exists (G : Type) [Fintype G] [DecidableEq G] (connected : G → G → Prop)
  (distinct_naturals : G → ℕ) :
  (∀ a b : G, ¬ connected a b → gcd (distinct_naturals a + distinct_naturals b) 15 = 1) ∧
  (∀ a b : G, connected a b → gcd (distinct_naturals a + distinct_naturals b) 15 > 1) →
  (∀ n : ℕ, 
    (∀ a b : G, ¬ connected a b → gcd (distinct_naturals a + distinct_naturals b) n = 1) ∧
    (∀ a b : G, connected a b → gcd (distinct_naturals a + distinct_naturals b) n > 1) →
    15 ≤ n) :=
sorry

end smallest_n_exists_l510_510072


namespace log_expression_l510_510539

theorem log_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  (log (x^2) / log (y^4)) * (log (y^3) / log (x^3)) * 
  (log (x^4) / log (y^5)) * (log (y^4) / log (x^2)) * 
  (log (x^3) / log (y^3)) = (1 / 5) * log x / log y :=
sorry

end log_expression_l510_510539


namespace circles_intersect_l510_510319

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Definition of two circles C1 and C2
def C1 : Circle := ⟨(1, 0), 1⟩
def C2 : Circle := ⟨(0, 2), 2⟩

-- Distance between the centers of two circles
def distance (C1 C2 : Circle) : ℝ :=
  Real.sqrt ((C1.center.1 - C2.center.1)^2 + (C1.center.2 - C2.center.2)^2)

-- Prove that circles C1 and C2 intersect
theorem circles_intersect (C1 C2 : Circle) (h1 : C1.center = (1, 0)) (h2 : C2.center = (0, 2)) (r1 : C1.radius = 1) (r2 : C2.radius = 2) : (1 + 2 > distance C1 C2) ∧ (distance C1 C2 > 2 - 1) :=
by
  -- Using given conditions to find the distance between circles' centers
  have h_distance : distance C1 C2 = Real.sqrt ((1 - 0)^2 + (0 - 2)^2) := by sorry
  -- Using calculated distance to prove the circles intersect
  rw [h_distance]
  rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow, Real.sqrt_eq_rpow]
  have : Real.sqrt 5 = (5:ℝ)^ (1 / 2) := by sorry
  rw [this]
  split
  - linarith
  - linarith

end circles_intersect_l510_510319


namespace suko_puzzle_unique_solution_count_l510_510074

def is_distinct (lst : List ℕ) : Prop := lst.nodup

def sum_four_numbers (a b c d : ℕ) (target : ℕ) : Prop := a + b + c + d = target

theorem suko_puzzle_unique_solution_count :
  ∃ (a b c d e f g h i : ℕ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
    g ≠ h ∧ g ≠ i ∧
    h ≠ i) ∧
    (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧ 
    (1 ≤ d ∧ d ≤ 9) ∧ (1 ≤ e ∧ e ≤ 9) ∧ (1 ≤ f ∧ f ≤ 9) ∧ 
    (1 ≤ g ∧ g ≤ 9) ∧ (1 ≤ h ∧ h ≤ 9) ∧ (1 ≤ i ∧ i ≤ 9) ∧
    sum_four_numbers d e g h 11 ∧
    sum_four_numbers b c e f 28 ∧
    sum_four_numbers e f h i 18 ∧
    is_distinct [a, b, c, d, e, f, g, h, i] ∧
    set.to_finset {a, b, c, d, e, f, g, h, i} = set.to_finset {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    fintype.card { x : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ // 
                   (sum_four_numbers d e g h 11 ∧ 
                   sum_four_numbers b c e f 28 ∧ 
                   sum_four_numbers e f h i 18 ∧ 
                   is_distinct [a, b, c, d, e, f, g, h, i]) } = 2 :=
sorry

end suko_puzzle_unique_solution_count_l510_510074


namespace polynomial_inequality_holds_l510_510355

def polynomial (x : ℝ) : ℝ := x^6 + 4 * x^5 + 2 * x^4 - 6 * x^3 - 2 * x^2 + 4 * x - 1

theorem polynomial_inequality_holds (x : ℝ) :
  (x ≤ -1 - Real.sqrt 2 ∨ x = (-1 - Real.sqrt 5) / 2 ∨ x ≥ -1 + Real.sqrt 2) →
  polynomial x ≥ 0 :=
by
  sorry

end polynomial_inequality_holds_l510_510355


namespace find_valid_pairs_l510_510334

def is_valid_pair (m n : ℕ) : Prop :=
  (∃! (m n : ℕ), (m = 1 ∧ n = 995) ∨ (m = 10 ∧ n = 176) ∨ (m = 21 ∧ n = 80))

theorem find_valid_pairs :
  ∀ (m n : ℕ),
  let regions := (n * (n + 1)) / 2 + 1 + m * (n + 1) in
  regions = 1992 → is_valid_pair m n :=
by
  intros m n regions h
  sorry

end find_valid_pairs_l510_510334


namespace min_races_top_3_l510_510126

theorem min_races_top_3 (max_horses_per_race : ℕ) (total_horses : ℕ) (no_timer : Prop) 
    (max_horses_per_race_condition : max_horses_per_race = 5) (total_horses_condition : total_horses = 25) 
    (no_timing_condition : no_timer) : 
    (∃ min_races : ℕ, min_races = 7) :=
by
  use 7
  sorry

end min_races_top_3_l510_510126


namespace max_p2x2y2_proof_l510_510251

noncomputable def max_p2x2y2 (p x y : ℕ) : ℕ := p^2 + x^2 + y^2

theorem max_p2x2y2_proof : ∃ p x y : ℕ, prime p ∧ 9 * x * y = p * (p + 3 * x + 6 * y) ∧ max_p2x2y2 p x y = 29 :=
by {
  -- Note: Proof is omitted as per the instructions
  sorry
}

end max_p2x2y2_proof_l510_510251


namespace find_point_symmetric_about_y_axis_l510_510063

def point := ℤ × ℤ

def symmetric_about_y_axis (A B : point) : Prop :=
  B.1 = -A.1 ∧ B.2 = A.2

theorem find_point_symmetric_about_y_axis (A B : point) 
  (hA : A = (-5, 2)) 
  (hSym : symmetric_about_y_axis A B) : 
  B = (5, 2) := 
by
  -- We declare the proof but omit the steps for this exercise.
  sorry

end find_point_symmetric_about_y_axis_l510_510063


namespace d_divisibility_l510_510525

theorem d_divisibility (p d : ℕ) (h_p : 0 < p) (h_d : 0 < d)
  (h1 : Prime p) 
  (h2 : Prime (p + d)) 
  (h3 : Prime (p + 2 * d)) 
  (h4 : Prime (p + 3 * d)) 
  (h5 : Prime (p + 4 * d)) 
  (h6 : Prime (p + 5 * d)) : 
  (2 ∣ d) ∧ (3 ∣ d) ∧ (5 ∣ d) :=
by
  sorry

end d_divisibility_l510_510525


namespace max_x_value_l510_510102

theorem max_x_value (x y z : ℝ) (h1 : x + y + z = 7) (h2 : x * y + x * z + y * z = 12) : x ≤ 1 :=
by sorry

end max_x_value_l510_510102


namespace sequence_ratio_l510_510647

theorem sequence_ratio :
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
    (a 1 = 2) →
    (∀ n, a (n + 1) = (2 * (n + 2) * a n) / (n + 1)) →
    (S 0 = 0) →
    (∀ n, S (n + 1) = S n + a (n + 1)) →
    (∑ k in range(2016+1), a k = S 2016) →
  (a 2017 : ℝ) / ∑ k in range(2017), a k = 1009 / 1008 :=
by
  sorry

end sequence_ratio_l510_510647


namespace p_is_necessary_but_not_sufficient_for_q_l510_510382

def condition_p (a : ℝ) : Prop := abs a = 2

def tangent_line (a : ℝ) : Prop := 
  let line := λ x, a * x + 1 - a
  let parabola := λ x, x^2
  ∃ t : ℝ, line t = parabola t ∧ ∀ ξ : ℝ, ξ ≠ t → line ξ ≠ parabola ξ

theorem p_is_necessary_but_not_sufficient_for_q :
  (∀ a : ℝ, tangent_line a → condition_p a) ∧ ¬(∀ a : ℝ, condition_p a → tangent_line a) :=
by
  sorry

end p_is_necessary_but_not_sufficient_for_q_l510_510382


namespace speed_of_boat_in_still_water_l510_510557

theorem speed_of_boat_in_still_water :
  ∀ (v : ℚ), (33 = (v + 3) * (44 / 60)) → v = 42 := 
by
  sorry

end speed_of_boat_in_still_water_l510_510557


namespace sum_of_series_is_401_288_l510_510488

theorem sum_of_series_is_401_288 :
  ∃ a b : ℕ, Nat.gcd a b = 1 ∧
            (a : ℚ) / b = 
            ∑ n in (Finset.range 1000), if n % 2 = 0 then (n + 1 : ℚ) / (2 : ℚ)^(n + 1) else (n + 1 : ℚ) / (3 : ℚ)^(n + 1) ∧
            a + b = 689 :=
by
  sorry

end sum_of_series_is_401_288_l510_510488


namespace a_formula_T_formula_l510_510739

variable {a : Nat → ℤ} {b : Nat → ℤ} {T : Nat → ℤ}

-- Condition for the sequence a_n
axiom a_sequence : ∀ n : Nat, a (n + 1) - a n = 2^n

-- Initial condition for a_1
axiom a_init : a 1 = 1

-- General formula for a_n
theorem a_formula (n : Nat) : a n = 2^n - 1 :=
by
  induction n with
  | zero => 
    -- case n = 0, not applicable since a_1 is given, continue to n = 1.
    sorry
  | succ n ih => 
    -- using the induction hypothesis and the sequence relationship
    sorry

-- Definition for b_n
def b (n : Nat) : ℤ := (a n + 1) / (a n * a (n + 1))

-- Sum of the first n terms T_n
def T (n : Nat) : ℤ := ∑ i in Finset.range n, b (i + 1)

-- General formula for T_n
theorem T_formula (n : Nat) : T n = 1 - 1 / (2^(n+1) - 1) :=
by 
  induction n with
  | zero => 
    -- case n = 0
    sorry
  | succ n ih => 
    -- using the induction hypothesis and previously proven results
    sorry

end a_formula_T_formula_l510_510739


namespace a_n_formula_b_sum_formula_lambda_range_l510_510740

-- Definitions
def S (n : ℕ) : ℝ := 2^(n + 1) - 2
def a (n : ℕ) : ℝ := 2^n
def b (n : ℕ) : ℝ := 
if n % 2 = 0 then (2^n / 3) + (2 / 3) else (2^n / 3) - (2 / 3)

-- Theorem Statements
theorem a_n_formula (n : ℕ) (n_pos : n > 0) : a n = 2^n := sorry

theorem b_sum_formula (n : ℕ) (n_pos : n > 0) : 
S n = if n % 2 = 0 then ((2^(n+1)) / 3) - (2 / 3) else ((2^(n+1)) / 3) - (4 / 3) := sorry

theorem lambda_range (λ : ℝ) : (∀ n > 0, b n < λ * b (n+1)) → 1 < λ := sorry

end a_n_formula_b_sum_formula_lambda_range_l510_510740


namespace factorization_l510_510707

theorem factorization (a : ℝ) : 4 * a^2 - 1 = (2 * a + 1) * (2 * a - 1) :=
by
  sorry

end factorization_l510_510707


namespace hypotenuse_length_l510_510467

variables {P Q R S T : Type} [MetricSpace P]

-- Given triangle PQR is a right triangle at P, with PQ and PR as legs
def is_right_triangle (P Q R S T : Type) [MetricSpace P] : Prop :=
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ a = dist P Q ∧ b = dist P R ∧ c = dist Q R

-- Given the points S and T such that PS:SQ = PT:TR = 1:3
def points_ratio (P Q R S T : P) : Prop :=
  ∃ (k : ℝ), k = 1/4 ∧
             dist P S = k * dist P Q ∧ dist S Q = (1 - k) * dist P Q ∧
             dist P T = k * dist P R ∧ dist T R = (1 - k) * dist P R

-- Given distances, QT = 20 units and SR = 35 units
def distances (P Q R S T : P) : Prop :=
  dist Q T = 20 ∧ dist S R = 35

-- Prove the hypotenuse QR's length
theorem hypotenuse_length (P Q R S T : Type) [MetricSpace P]
  (h1 : is_right_triangle P Q R S T)
  (h2 : points_ratio P Q R S T)
  (h3 : distances P Q R S T) :
  dist Q R = 39 :=
sorry

end hypotenuse_length_l510_510467


namespace angle_A_solution_l510_510048

theorem angle_A_solution (a b : ℝ) (B A : ℝ) :
  a = Real.sqrt 3 ∧ b = 1 ∧ B = Real.pi / 6 → (A = Real.pi / 3 ∨ A = 2 * Real.pi / 3) :=
by
  assume h : a = Real.sqrt 3 ∧ b = 1 ∧ B = Real.pi / 6,
  sorry

end angle_A_solution_l510_510048


namespace aras_current_height_l510_510515

-- Define the variables and conditions
variables (x : ℝ) (sheas_original_height : ℝ := x) (ars_original_height : ℝ := x)
variables (sheas_growth_factor : ℝ := 0.30) (sheas_current_height : ℝ := 65)
variables (sheas_growth : ℝ := sheas_current_height - sheas_original_height)
variables (aras_growth : ℝ := sheas_growth / 3)

-- Define a theorem for Ara's current height
theorem aras_current_height (h1 : sheas_current_height = (1 + sheas_growth_factor) * sheas_original_height)
                           (h2 : sheas_original_height = ars_original_height) :
                           aras_growth + ars_original_height = 55 :=
by
  sorry

end aras_current_height_l510_510515


namespace probability_odd_sum_of_two_positive_integers_l510_510241

theorem probability_odd_sum_of_two_positive_integers (x y: ℕ) (hx: 0 < x) (hy: 0 < y) :
  ∃ p: ℝ, p = 1 / 2 ∧ (Prob (x + y is odd)) = p :=
by
  sorry

end probability_odd_sum_of_two_positive_integers_l510_510241


namespace area_relation_l510_510295

-- Define the setup for the triangles and their respective areas
variables {ABC : Type*} [acute_triangle ABC]
variables {A0 B0 C0 A1 B1 C1 A2 B2 C2 A3 B3 C3 : Type*}
variables (T0 T1 T2 T3 : ℝ) 

-- Define the areas of the triangles
noncomputable def area (triangle : Type*) : ℝ := sorry

-- Conditions on touching points and incircle/excircle areas
axiom incircle_touch (A0 B0 C0 : Type*) : ∀ (side : Type*), is_tangent (incircle triangle) side
axiom excircle_touch (A1 A2 A3 B1 B2 B3 C1 C2 C3 : Type*) : ∀ (side : Type*), 
  is_tangent (excircle triangle) side
axiom area_A0B0C0 : T0 = area A0B0C0
axiom area_A1B1C1 : T1 = area A1B1C1
axiom area_A2B2C2 : T2 = area A2B2C2
axiom area_A3B3C3 : T3 = area A3B3C3

-- The theorem to be proven
theorem area_relation : 
  1 / T0 = 1 / T1 + 1 / T2 + 1 / T3 := sorry

end area_relation_l510_510295


namespace _l510_510605

-- Part (a)
lemma unique_point_exists (a b : ℕ) (h_coprime : Nat.coprime a b) :
  ∀ (c : ℕ), ∃! (x y : ℕ), 0 ≤ x ∧ x ≤ b - 1 ∧ ax + by = c := 
sorry

-- Part (b)
lemma sylvesters_theorem (a b : ℕ) (h_coprime : Nat.coprime a b) :
  ∃ (c : ℕ), (∀ (x y : ℕ), ax + by = c → x < 0 ∨ y < 0) ∧ 
  (∀ (d : ℕ), d > c → ∃ (x y : ℕ), ax + by = d) :=
begin
  let c := a * b - a - b,
  use c,
  split,
  { sorry },
  { sorry }
end

end _l510_510605


namespace number_of_divisors_64m4_l510_510372

theorem number_of_divisors_64m4
  (m : ℕ)
  (hm : (150 * m^3).divisorCount = 150) :
  (64 * m^4).divisorCount = 675 := sorry

end number_of_divisors_64m4_l510_510372


namespace four_digit_cubes_divisible_by_16_count_l510_510792

theorem four_digit_cubes_divisible_by_16_count :
  ∃ (count : ℕ), count = 3 ∧
    ∀ (m : ℕ), 1000 ≤ 64 * m^3 ∧ 64 * m^3 ≤ 9999 → (m = 3 ∨ m = 4 ∨ m = 5) :=
by {
  -- our proof would go here
  sorry
}

end four_digit_cubes_divisible_by_16_count_l510_510792


namespace BN_eq_CM_l510_510671

theorem BN_eq_CM (ABC : Triangle) (Γ : Circle)
  (hAcute : IsAcuteABC ABC)
  (hCircumcircle : CircumcircleABC ABC Γ)
  (B C P : Point)
  (hTangentB : IsTangent Γ B P)
  (hTangentC : IsTangent Γ C P)
  (D E F N M : Point)
  (hProjDP : IsProjection P D BC)
  (hProjEP : IsProjection P E AC)
  (hProjFP : IsProjection P F AB)
  (hCircDEF : CircumcircleDEF DEF N)
  (hDiffD : N ≠ D)
  (hProjMA : IsProjection A M BC) :
  BN = CM := by
  sorry

end BN_eq_CM_l510_510671


namespace coefficient_x6_expansion_l510_510577

noncomputable def binomial_expansion_coefficient (n : ℕ) (a b : ℤ) (k : ℕ) : ℤ :=
  nat.choose n k * a^k * b^(n-k)

theorem coefficient_x6_expansion : 
  let f := (λ x : ℤ, (3 * x + 2)^8 * (4 * x - 1)^2) in
  polynomial.coeff (polynomial.expand ℤ f) 6 = -792108 := 
by
  sorry

end coefficient_x6_expansion_l510_510577


namespace circumscribed_sphere_surface_area_correct_l510_510376

noncomputable def circumscribed_sphere_surface_area {a b c : ℝ} (h_c : c = 1)
  (h1 : (sqrt 3 * a * c) / sqrt (a^2 + c^2) = b)
  (h2 : (b * c) / sqrt (b^2 + c^2) = a) : ℝ :=
  if (a = sqrt 2 / 2 ∧ b = 1) then 4 * Real.pi * ((sqrt (a^2 + b^2 + c^2)) / 2)^2 else 0

theorem circumscribed_sphere_surface_area_correct :
  circumscribed_sphere_surface_area (c := 1) _ _ = 5 * Real.pi / 2 := sorry

end circumscribed_sphere_surface_area_correct_l510_510376


namespace cupcake_cookie_price_ratio_l510_510476

theorem cupcake_cookie_price_ratio
  (c k : ℚ)
  (h1 : 5 * c + 3 * k = 23)
  (h2 : 4 * c + 4 * k = 21) :
  k / c = 13 / 29 :=
  sorry

end cupcake_cookie_price_ratio_l510_510476


namespace fixed_points_of_function_l510_510113

theorem fixed_points_of_function :
  let f := fun x => x^2 - 2*x - 10 in
  (∃ x₀, f x₀ = x₀) → x₀ = -2 ∨ x₀ = 5 :=
by
  sorry

end fixed_points_of_function_l510_510113


namespace find_n_l510_510492

-- Definition of the problem conditions
def sequence (n : ℕ) (b : ℕ → ℝ) : Prop :=
  (b 0 = 25) ∧ (b 1 = 56) ∧ (b n = 0) ∧ ∀ k, (1 ≤ k ∧ k < n) → b (k+1) = b (k-1) - 7 / b k

theorem find_n (n : ℕ) (b : ℕ → ℝ) (h : sequence n b) : n = 201 :=
sorry

end find_n_l510_510492


namespace intersect_A_B_l510_510498

def A : Set ℕ := {x | x ≤ 6}
def B : Set ℝ := {x | x < 0 ∨ x > 3}
def AB_intersection : Set ℝ := {4, 5, 6}

theorem intersect_A_B : (A : Set ℝ) ∩ B = AB_intersection := by
  sorry

end intersect_A_B_l510_510498


namespace total_outfits_l510_510973

def numRedShirts : ℕ := 7
def numGreenShirts : ℕ := 5
def numPants : ℕ := 6
def numRedHats : ℕ := 7
def numGreenHats : ℕ := 9

theorem total_outfits : 
  ((numRedShirts * numPants * numGreenHats) + 
   (numGreenShirts * numPants * numRedHats) + 
   ((numRedShirts * numRedHats + numGreenShirts * numGreenHats) * numPants)
  ) = 1152 := 
by
  sorry

end total_outfits_l510_510973


namespace find_integer_n_l510_510796

theorem find_integer_n (n : ℤ) (h : (⌊(n^2 : ℤ)/4⌋ - (⌊n/2⌋)^2 = 2)) : n = 5 :=
sorry

end find_integer_n_l510_510796


namespace root_of_function_is_four_l510_510384

noncomputable def f (x : ℝ) : ℝ := 2 - Real.log x / Real.log 2

theorem root_of_function_is_four (a : ℝ) (h : f a = 0) : a = 4 :=
by
  sorry

end root_of_function_is_four_l510_510384


namespace F_range_l510_510812

-- Define the function f and its range condition
def f (x : ℝ) : ℝ := sorry -- Placeholder for the function definition

axiom f_range : ∀ x, 1 ≤ f(x) ∧ f(x) ≤ 3

-- Define the function F based on f
def F (x : ℝ) : ℝ := 1 - 2 * f(x + 3)

-- Prove that the range of F is [-5, -1]
theorem F_range : ∀ y, (∃ x, F(x) = y) ↔ -5 ≤ y ∧ y ≤ -1 :=
by
  sorry

end F_range_l510_510812


namespace unique_functional_equation_l510_510709

noncomputable def nu2 (q : ℚ) : ℤ := sorry
noncomputable def f_defined (r : ℚ) : ℝ :=
  if r = 0 then 0 else 2 ^ (-nu2 r)

theorem unique_functional_equation (f : ℚ → ℝ) :
  (∀ x y : ℚ, f(x + y) ≤ f x + f y) →
  (∀ x y : ℚ, f (x * y) = f x * f y) →
  f 2 = 1 / 2 →
  (∀ r : ℚ, r ≠ 0 → f r = 2 ^ (-nu2 r)) ∧ f 0 = 0 :=
by
  -- Proof goes here
  sorry

end unique_functional_equation_l510_510709


namespace chords_do_not_bisect_l510_510885

noncomputable def point := ℝ × ℝ
noncomputable def circle := { center : point // ∃ r : ℝ, r > 0 }

variables (O : circle) (A B C D P : point)
variables (h1 : ∃ r, r > 0 ∧ ∀ (x : point), (x.1 - O.center.1)^2 + (x.2 - O.center.2)^2 = r^2)
(h2 : line_through P A ≠ line_through P B ∧ line_through P C ≠ line_through P D)
(h3 : ¬ collinear O.center A B ∧ ¬ collinear O.center C D)

theorem chords_do_not_bisect : 
  (¬ (∃ k : ℝ, k > 0 ∧ dist A P = k ∧ dist B P = k
  ∧ dist C P = k ∧ dist D P = k)) := sorry

end chords_do_not_bisect_l510_510885


namespace count_five_letter_words_l510_510085

theorem count_five_letter_words : (26 ^ 4 = 456976) :=
by {
    sorry
}

end count_five_letter_words_l510_510085


namespace sufficient_but_not_necessary_condition_l510_510757

theorem sufficient_but_not_necessary_condition
  (a b : ℝ) (h : a > b + 1) : (a > b) ∧ ¬ (∀ (a b : ℝ), a > b → a > b + 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l510_510757


namespace expression_value_l510_510591

theorem expression_value (a b : ℤ) (h₁ : a = -5) (h₂ : b = 3) :
  -a - b^4 + a * b = -91 := by
  sorry

end expression_value_l510_510591


namespace parallel_lines_in_intersecting_planes_l510_510003

variables (m n : Line) (α β : Plane)

-- Conditions
variables (h1 : m ∥ α) (h2 : m ⊆ β) (h3 : α ∩ β = n)

theorem parallel_lines_in_intersecting_planes : m ∥ n :=
sorry

end parallel_lines_in_intersecting_planes_l510_510003


namespace find_a_range_find_value_x1_x2_l510_510414

noncomputable def quadratic_equation_roots_and_discriminant (a : ℝ) :=
  ∃ x1 x2 : ℝ, 
      (x1^2 - 3 * x1 + 2 * a + 1 = 0) ∧ 
      (x2^2 - 3 * x2 + 2 * a + 1 = 0) ∧
      (x1 ≠ x2) ∧ 
      (∀ Δ > 0, Δ = 9 - 8 * a - 4)

theorem find_a_range (a : ℝ) : 
  (quadratic_equation_roots_and_discriminant a) → a < 5 / 8 :=
sorry

theorem find_value_x1_x2 (a : ℤ) (h : a = 0) (x1 x2 : ℝ) :
  (x1^2 - 3 * x1 + 2 * a + 1 = 0) ∧ 
  (x2^2 - 3 * x2 + 2 * a + 1 = 0) ∧ 
  (x1 + x2 = 3) ∧ 
  (x1 * x2 = 1) → 
  (x1^2 * x2 + x1 * x2^2 = 3) :=
sorry

end find_a_range_find_value_x1_x2_l510_510414


namespace train_and_car_combined_time_l510_510287

theorem train_and_car_combined_time (car_time : ℝ) (train_time : ℝ) 
  (h1 : train_time = car_time + 2) (h2 : car_time = 4.5) : 
  car_time + train_time = 11 := 
by 
  -- Proof goes here
  sorry

end train_and_car_combined_time_l510_510287


namespace oil_ratio_l510_510256

theorem oil_ratio (x : ℝ) (initial_small_tank : ℝ) (initial_large_tank : ℝ) (total_capacity_large : ℝ)
  (half_capacity_large : ℝ) (additional_needed : ℝ) :
  initial_small_tank = 4000 ∧ initial_large_tank = 3000 ∧ total_capacity_large = 20000 ∧
  half_capacity_large = total_capacity_large / 2 ∧ additional_needed = 4000 ∧
  (initial_large_tank + x + additional_needed = half_capacity_large) →
  x / initial_small_tank = 3 / 4 :=
by
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩
  sorry

end oil_ratio_l510_510256


namespace smallest_a1_l510_510489

theorem smallest_a1 (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_rec : ∀ n > 1, a n = 7 * a (n - 1) - n) :
  a 1 ≥ 13 / 36 :=
by
  sorry

end smallest_a1_l510_510489


namespace sample_40th_number_drawn_l510_510527

theorem sample_40th_number_drawn 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (students_in_part : ℕ) 
  (first_random_number : ℕ) 
  (step : ℕ) 
  (desired_number : ℕ) 
  (hs1 : total_students = 1000) 
  (hs2 : sample_size = 50) 
  (hs3 : students_in_part = total_students / sample_size) 
  (hs4 : first_random_number = 15) 
  (hs5 : step = 20) 
  (hs6 : desired_number = first_random_number + (39 * step)) : 
  desired_number = 795 := 
by
  rw [hs4, hs5]
  rw [(39 * 20)]
  sorry

end sample_40th_number_drawn_l510_510527


namespace avg_weight_section_A_l510_510932

/-- Given the conditions about student distributions and average weights,
prove the average weight of section A. -/
theorem avg_weight_section_A 
  (num_students_A num_students_B : ℕ)
  (avg_weight_B avg_weight_class : ℝ) 
  (total_students : ℕ := num_students_A + num_students_B)
  (total_weight_class : ℝ := total_students * avg_weight_class)
  (total_weight_B : ℝ := num_students_B * avg_weight_B)
  (avg_weight_A : ℝ) :
  num_students_A = 30 ->
  num_students_B = 20 ->
  avg_weight_B = 35 ->
  avg_weight_class = 38 ->
  total_weight_class = total_weight_B + num_students_A * avg_weight_A ->
  avg_weight_A = 40 :=
by
  intros hA hB hB_weight hClass_weight hTotal_weight
  rw [hA, hB, hB_weight, hClass_weight, hTotal_weight]
  sorry
 
end avg_weight_section_A_l510_510932


namespace integer_roots_of_quadratic_eq_are_neg3_and_neg7_l510_510332

theorem integer_roots_of_quadratic_eq_are_neg3_and_neg7 :
  {k : ℤ | ∃ x : ℤ, k * x^2 - 2 * (3 * k - 1) * x + 9 * k - 1 = 0} = {-3, -7} :=
by
  sorry

end integer_roots_of_quadratic_eq_are_neg3_and_neg7_l510_510332


namespace train_speed_correct_l510_510655

def speed_of_train (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * (18 / 5)

theorem train_speed_correct :
  speed_of_train 140 9 = 56 := by
    sorry

end train_speed_correct_l510_510655


namespace scheduling_arrangements_l510_510452

-- Definitions from conditions
def Subject := {math, physics, history, chinese, physical_education}

def valid_schedule (schedule : List Subject) : Prop :=
  schedule.length = 5 ∧
  schedule.head ≠ physical_education ∧
  ∀ i, (i < schedule.length - 1) →
    (schedule.nth i ≠ some chinese ∨ schedule.nth (i + 1) ≠ some physics) ∧
    (schedule.nth i ≠ some physics ∨ schedule.nth (i + 1) ≠ some chinese)

-- Statement of the problem
def total_valid_schedules : Nat :=
  List.permutations [math, physics, history, chinese, physical_education].count valid_schedule

theorem scheduling_arrangements : total_valid_schedules = 48 := by
  sorry

end scheduling_arrangements_l510_510452


namespace minimum_value_expression_l510_510547

variable {a c m n x₁ x₂ : ℝ}

def line_passes_through_point (a c m n : ℝ) : Prop :=
  2 * a * m + (a + c) * n + 2 * c = 0

def sum_constraint (x₁ x₂ m n : ℝ) : Prop :=
  x₁ + x₂ + m + n = 15

theorem minimum_value_expression
  (ha : a ∈ ℝ)
  (hc : c ∈ ℝ)
  (hx₁_gt_hx₂ : x₁ > x₂)
  (h_line : line_passes_through_point a c m n)
  (h_sum : sum_constraint x₁ x₂ m n) :
  ∃ (min_val : ℝ), min_val = 16 :=
begin
  -- Proof goes here
  sorry
end

end minimum_value_expression_l510_510547


namespace find_x_l510_510016

-- We define the given condition in Lean
theorem find_x (x : ℝ) (h : 6 * x - 12 = -(4 + 2 * x)) : x = 1 :=
sorry

end find_x_l510_510016


namespace maximum_value_inequality_l510_510863

noncomputable def max_value (a b c d : ℝ) (h : a + b + c + d = 1) := 
  (ab / (a + b)) + (ac / (a + c)) + (ad / (a + d)) + (bc / (b + c)) + (bd / (b + d)) + (cd / (c + d))

theorem maximum_value_inequality 
  (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (h : a + b + c + d = 1) :
  max_value a b c d h ≤ 1 / 2 :=
sorry

end maximum_value_inequality_l510_510863


namespace entrance_exam_proof_l510_510980

-- Define the conditions
variables (x y : ℕ)
variables (h1 : x + y = 70)
variables (h2 : 3 * x - y = 38)

-- The proof goal
theorem entrance_exam_proof : x = 27 :=
by
  -- The actual proof steps are omitted here
  sorry

end entrance_exam_proof_l510_510980


namespace necessarily_positive_l510_510141

theorem necessarily_positive (x y w : ℝ) (h1 : 0 < x ∧ x < 0.5) (h2 : -0.5 < y ∧ y < 0) (h3 : 0.5 < w ∧ w < 1) : 
  0 < w - y :=
sorry

end necessarily_positive_l510_510141


namespace JaneTotalEarningsIs138_l510_510081

structure FarmData where
  chickens : ℕ
  ducks : ℕ
  quails : ℕ
  chickenEggsPerWeek : ℕ
  duckEggsPerWeek : ℕ
  quailEggsPerWeek : ℕ
  chickenPricePerDozen : ℕ
  duckPricePerDozen : ℕ
  quailPricePerDozen : ℕ

def JaneFarmData : FarmData := {
  chickens := 10,
  ducks := 8,
  quails := 12,
  chickenEggsPerWeek := 6,
  duckEggsPerWeek := 4,
  quailEggsPerWeek := 10,
  chickenPricePerDozen := 2,
  duckPricePerDozen := 3,
  quailPricePerDozen := 4
}

def eggsLaid (f : FarmData) : ℕ × ℕ × ℕ :=
((f.chickens * f.chickenEggsPerWeek), 
 (f.ducks * f.duckEggsPerWeek), 
 (f.quails * f.quailEggsPerWeek))

def earningsForWeek1 (f : FarmData) : ℕ :=
let (chickenEggs, duckEggs, quailEggs) := eggsLaid f
let chickenDozens := chickenEggs / 12
let duckDozens := duckEggs / 12
let quailDozens := (quailEggs / 12) / 2
(chickenDozens * f.chickenPricePerDozen) + (duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def earningsForWeek2 (f : FarmData) : ℕ :=
let (chickenEggs, duckEggs, quailEggs) := eggsLaid f
let chickenDozens := chickenEggs / 12
let duckDozens := (3 * duckEggs / 4) / 12
let quailDozens := quailEggs / 12
(chickenDozens * f.chickenPricePerDozen) + (duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def earningsForWeek3 (f : FarmData) : ℕ :=
let (_, duckEggs, quailEggs) := eggsLaid f
let duckDozens := duckEggs / 12
let quailDozens := quailEggs / 12
(duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def totalEarnings (f : FarmData) : ℕ :=
earningsForWeek1 f + earningsForWeek2 f + earningsForWeek3 f

theorem JaneTotalEarningsIs138 : totalEarnings JaneFarmData = 138 := by
  sorry

end JaneTotalEarningsIs138_l510_510081


namespace find_a_plus_b_l510_510397

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + x + b

theorem find_a_plus_b (a b : ℝ)
  (h1 : ∀ x : ℝ, f a b (-x) = - f a b x)
  (h2 : ∃ t : ℝ, ∀ m : ℝ, y : ℝ, m = 3 * a * 1^2 + 1 → y = a * 1^3 + 1 + b → y - (a + 1) = (3 * a + 1) * (t - 1)
    → 6 - a - 1 = 3 * a + 1) : a + b = 1 :=
sorry

end find_a_plus_b_l510_510397


namespace volume_tetrahedron_inscribed_l510_510463

noncomputable def volume_of_KLMN (EF GH EG FH EH FG : ℝ) : ℝ :=
  let V := (1 / 288) * real.sqrt(
    -EF^4 * GH^4 - GH^4 * EG^4 - EG^4 * EF^4 - EF^4 * EG^4 
    + 4 * EF^2 * GH^2 * EG^2) in
  V / 3 -- considering the symmetry factor

theorem volume_tetrahedron_inscribed (EF GH EG FH EH FG : ℝ) 
  (h1 : EF = 7) (h2 : GH = 7) (h3 : EG = 10) 
  (h4 : FH = 10) (h5 : EH = 11) (h6 : FG = 11) :
  abs (volume_of_KLMN EF GH EG FH EH FG - 2.09) < 0.01 :=
by
  sorry

end volume_tetrahedron_inscribed_l510_510463


namespace find_m_l510_510782

-- Definitions based on the conditions

def is_solution (a b c x : ℝ) : Prop :=
  x^2 - a * x + b * x + c = 0

def A (m : ℝ) : set ℝ :=
  {x | is_solution 1 (-4*m) (2*m + 6) x}

def B : set ℝ := {x | x < 0}

-- Proving that "A ∩ B = ∅" is false implies m ≤ -1
theorem find_m (m : ℝ) (h: ¬(A m ∩ B = ∅)) : m ≤ -1 :=
sorry

end find_m_l510_510782


namespace problem1_problem2_l510_510835

theorem problem1 (a b c : ℝ) (A B C : ℝ) (h : 2 * a * sin A = (2 * b - c) * sin B + (2 * c - b) * sin C) :
  A = 60 := sorry

theorem problem2 (a b c : ℝ) (A B C : ℝ) (hA : A = 60) (hSC : sin B + sin C = sqrt 3) :
  A = B ∧ B = C ∧ C = A := sorry

end problem1_problem2_l510_510835


namespace shaded_region_area_correct_l510_510928

def area_of_shaded_region (hypotenuse : ℝ) (num_squares : ℕ) (side_length_eq_legs : Bool) : ℝ :=
  if hypotenuse = 10 ∧ num_squares = 12 ∧ side_length_eq_legs then
    275 / 3
  else
    0

theorem shaded_region_area_correct :
  area_of_shaded_region 10 12 true = 275 / 3 :=
sorry

end shaded_region_area_correct_l510_510928


namespace number_of_correct_propositions_l510_510381

open Function

variables (m n : Type) (α β : Type)
variables [HasParallel m α] [HasParallel n α]
variables [HasPerpendicular n α] [HasParallel m β]
variables [HasPerpendicular m α] [HasPerpendicular α β]

theorem number_of_correct_propositions : 
  (if (m ∥ α ∧ n ∥ α) then m ∥ n else false) ∧ 
  (if (m ∥ α ∧ n ⟂ α) then n ⟂ m else false) ∧ 
  (if (m ⟂ α ∧ m ∥ β) then α ⟂ β else false) →
  (count (λ p : bool, p = true) [if (m ∥ α ∧ n ∥ α) then m ∥ n else false,
                                 if (m ∥ α ∧ n ⟂ α) then n ⟂ m else false,
                                 if (m ⟂ α ∧ m ∥ β) then α ⟂ β else false] = 2) :=
begin
  sorry
end

end number_of_correct_propositions_l510_510381


namespace AreaTheorem_l510_510989

variables (A B C D E F P : Point)
          (ABD ADC : Triangle)
          (f : Altitude ADC)
          (g : Altitude ABD)
          (h : Altitude BEC)
          (i : Point) -- placeholder for intersections.

structure EquilateralTriangle :=
(is_equilateral : ABD ≡ ABDADC ∧ ABD ≡ BADC ∧ ABD ≡ ADC)

structure Altitudes :=
(AD BE CF : Altitude)

axiom any_point (P : Point) -- axiom to introduce arbitrary point

-- We state the theorem/formula to be proven.
theorem AreaTheorem 
  (hA : EquilateralTriangle ABD ADC)
  (hB : Altitudes AD BE CF)
  (hP : any_point P) :
  (Area (Triangle P A D) = Area (Triangle P B E) + Area (Triangle P C F)) ∨
  (Area (Triangle P B E) = Area (Triangle P A D) + Area (Triangle P C F)) ∨
  (Area (Triangle P C F) = Area (Triangle P A D) + Area (Triangle P B E)) := 
sorry

end AreaTheorem_l510_510989


namespace probability_blue_ball_l510_510942

-- Define the probabilities of drawing a red and yellow ball
def P_red : ℝ := 0.48
def P_yellow : ℝ := 0.35

-- Define the total probability formula in this sample space
def total_probability (P_red P_yellow P_blue : ℝ) : Prop :=
  P_red + P_yellow + P_blue = 1

-- The theorem we need to prove
theorem probability_blue_ball :
  ∃ P_blue : ℝ, total_probability P_red P_yellow P_blue ∧ P_blue = 0.17 :=
sorry

end probability_blue_ball_l510_510942


namespace simple_interest_amount_is_58_l510_510444

noncomputable def principal (CI : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  CI / ((1 + r / 100)^t - 1)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t / 100

theorem simple_interest_amount_is_58 (CI : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) :
  CI = 59.45 -> r = 5 -> t = 2 -> P = principal CI r t ->
  simple_interest P r t = 58 :=
by
  sorry

end simple_interest_amount_is_58_l510_510444


namespace limit_problem_l510_510679

noncomputable def limit_expression (x : ℝ) : ℝ := 
  (2*x - 1)^2 / (Real.exp (Real.sin (Real.pi * x)) - Real.exp (- Real.sin (3 * Real.pi * x)))

theorem limit_problem : filter.tendsto (limit_expression) (nhds_within (1/2) (set.univ)) (nhds (1 / (Real.exp 1 * Real.pi^2))) :=
sorry

end limit_problem_l510_510679


namespace intersection_A_and_B_l510_510425

-- Define the sets based on the conditions
def setA : Set ℤ := {x : ℤ | x^2 - 2 * x - 8 ≤ 0}
def setB : Set ℤ := {x : ℤ | 1 < Real.log x / Real.log 2}

-- State the theorem (Note: The logarithmic condition should translate the values to integers)
theorem intersection_A_and_B : setA ∩ setB = {3, 4} :=
sorry

end intersection_A_and_B_l510_510425


namespace find_a_l510_510776

def f (x a : ℝ) : ℝ := Real.logb 2 (x + a)

theorem find_a (a : ℝ) (h : f 2 a = 2) : a = 2 := by
  unfold f at h
  sorry

end find_a_l510_510776


namespace sin2x_solution_l510_510793

theorem sin2x_solution
  (x : ℝ)
  (h : sin x - cos x + tan x - cot x + sec x - csc x = 1) :
  sin (2 * x) = (5 - sqrt 57) / 2 :=
sorry

end sin2x_solution_l510_510793


namespace evaluate_ratio_l510_510328

theorem evaluate_ratio : (2^3002 * 3^3005 / 6^3003 : ℚ) = 9 / 2 := 
sorry

end evaluate_ratio_l510_510328


namespace brad_trips_to_fill_barrel_l510_510305

noncomputable def bucket_volume (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r^3

noncomputable def barrel_volume (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem brad_trips_to_fill_barrel :
  let r_bucket := 8  -- radius of the hemisphere bucket in inches
  let r_barrel := 8  -- radius of the cylindrical barrel in inches
  let h_barrel := 20 -- height of the cylindrical barrel in inches
  let V_bucket := bucket_volume r_bucket
  let V_barrel := barrel_volume r_barrel h_barrel
  (Nat.ceil (V_barrel / V_bucket) = 4) :=
by
  sorry

end brad_trips_to_fill_barrel_l510_510305


namespace average_last_three_l510_510904

theorem average_last_three {a b c d e f g : ℝ} 
  (h_avg_all : (a + b + c + d + e + f + g) / 7 = 60)
  (h_avg_first_four : (a + b + c + d) / 4 = 55) : 
  (e + f + g) / 3 = 200 / 3 :=
by
  sorry

end average_last_three_l510_510904


namespace inverse_prop_function_decreasing_l510_510408

theorem inverse_prop_function_decreasing (x y k : ℝ) (h1 : y = k / x) (h2 : (1, 4) ∈ set_of (λ p : ℝ × ℝ, p.snd = k / p.fst)) :
  ∀ x, x > 0 → (y = k / x → ∀ x₁ x₂, 0 < x₁ < x₂ → k / x₁ > k / x₂) :=
by
  sorry

end inverse_prop_function_decreasing_l510_510408


namespace find_ffx_equals_neg6_l510_510022

def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 4 else 2 * x

theorem find_ffx_equals_neg6 (x : ℝ) (h : f x = -3) : f (f x) = -6 :=
  by
    sorry

end find_ffx_equals_neg6_l510_510022


namespace planes_perpendicular_l510_510377

noncomputable section

open_locale real

structure Point3D := (x y z : ℝ)

def D : Point3D := ⟨0, 0, 0⟩ 
def A₁ : Point3D := ⟨1, 0, 1⟩ 
def B : Point3D := ⟨1, 1, 0⟩ 
def E : Point3D := ⟨0, 1, 0.5⟩ 

def vector (p1 p2 : Point3D) : Point3D := 
  ⟨p2.x - p1.x, p2.y - p1.y, p2.z - p1.z⟩

def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def normal_vector (v1 v2 : Point3D) : Point3D := 
  ⟨v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x⟩

def plane_perpendicular (p1 p2 p3 p4 p5 p6 : Point3D) : Prop :=
  let n1 := normal_vector (vector p1 p2) (vector p1 p3)
  let n2 := normal_vector (vector p4 p5) (vector p4 p6)
  in dot_product n1 n2 = 0

theorem planes_perpendicular : 
  plane_perpendicular D B A₁ E B D :=
by 
  sorry

end planes_perpendicular_l510_510377


namespace determine_x_l510_510322

noncomputable def x_candidates := { x : ℝ | x = (3 + Real.sqrt 105) / 24 ∨ x = (3 - Real.sqrt 105) / 24 }

theorem determine_x (x y : ℝ) (h_y : y = 3 * x) 
  (h_eq : 4 * y ^ 2 + 2 * y + 7 = 3 * (8 * x ^ 2 + y + 3)) :
  x ∈ x_candidates :=
by
  sorry

end determine_x_l510_510322


namespace exp_fn_max_min_diff_l510_510388

theorem exp_fn_max_min_diff (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (max (a^1) (a^0) - min (a^1) (a^0)) = 1 / 2 → (a = 1 / 2 ∨ a = 3 / 2) :=
by
  sorry

end exp_fn_max_min_diff_l510_510388


namespace min_value_of_expression_l510_510748

theorem min_value_of_expression (a b : ℝ) (h1 : a > 0) (h2 : b < 0) (h3 : a - b = 5) :
  ∃ (c : ℝ), is_least { y | ∃ (a b : ℝ), a > 0 ∧ b < 0 ∧ a - b = 5 ∧ y = (1 / (a+1) + 1 / (2-b)) } c ∧ c = 1/2 :=
begin
  sorry
end

end min_value_of_expression_l510_510748


namespace volume_tetrahedron_KLMN_is_approximately_2_09_l510_510466

-- Given side lengths
constant EF GH : ℝ := 7
constant EG FH : ℝ := 10
constant EH FG : ℝ := 11

-- Tetrahderon with given side lengths and K, L, M, N as the centers of the inscribed circles
noncomputable def volume_tetrahedron_KLMN (EF GH EG FH EH FG : ℝ) : ℝ :=
  (some_calculation_function EF GH EG FH EH FG)  -- Placeholder for the actual volume calculation function

-- Assert that the volume is approximately 2.09
theorem volume_tetrahedron_KLMN_is_approximately_2_09 :
  abs (volume_tetrahedron_KLMN EF GH EG FH EH FG - 2.09) < 0.01 :=
by
  sorry

end volume_tetrahedron_KLMN_is_approximately_2_09_l510_510466


namespace set_of_a_where_A_subset_B_l510_510808

variable {a x : ℝ}

theorem set_of_a_where_A_subset_B (h : ∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) :
  6 ≤ a ∧ a ≤ 9 :=
by
  sorry

end set_of_a_where_A_subset_B_l510_510808


namespace complement_domain_of_f_l510_510417

def universal_set := set.univ : set ℝ

def function_f (x : ℝ) : ℝ := real.log (x - 1)

def domain_M : set ℝ := { x | x > 1 }

def complement_U_M : set ℝ := { x | x ≤ 1 }

theorem complement_domain_of_f :
  { x ∈ universal_set | ¬ (x ∈ domain_M) } = complement_U_M :=
by
  sorry

end complement_domain_of_f_l510_510417


namespace combinations_difference_is_n_14_l510_510360

theorem combinations_difference_is_n_14 (n : ℕ) :
  (nat.choose (n + 1) 7 - nat.choose n 7 = nat.choose n 8) -> n = 14 :=
by
  sorry

end combinations_difference_is_n_14_l510_510360


namespace m_plus_n_eq_five_l510_510011

theorem m_plus_n_eq_five (m n : ℝ) (h1 : m - 2 = 0) (h2 : 1 + n - 2 * m = 0) : m + n = 5 := 
  by 
  sorry

end m_plus_n_eq_five_l510_510011


namespace remaining_speed_l510_510255

theorem remaining_speed (D : ℝ) (V : ℝ) 
  (h1 : 0.35 * D / 35 + 0.65 * D / V = D / 50) : V = 32.5 :=
by sorry

end remaining_speed_l510_510255


namespace math_proof_problem_l510_510168

noncomputable def sequence_a : ℕ → ℚ
| 0       := 1
| 1       := 2
| (n + 2) := (sequence_a n + sequence_a (n + 1)) / 2

def sequence_b (n : ℕ) : ℚ :=
  sequence_a (n + 1) - sequence_a n

def exists_general_formula_b : Prop :=
  ∃ r : ℕ → ℚ, ∀ n, r n = (-1/2 : ℚ)^(n - 1) ∧ sequence_b n = r n

def sequence_a_formula : ℕ → ℚ
| 1       := 1
| n       := 5/3 - (2/3) * (-1/2)^(n - 1)

def find_smallest_N (N : ℕ) : Prop :=
  ∃ N, ∀ n, n > N → abs (sequence_a n - 5/3) < 2/(9 * n)

def sequence_c (n : ℕ) : ℚ :=
  (3/2) * abs (sequence_a n - 5/3)

def T_n (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, sequence_c i

def c_m2 (m : ℕ) : ℚ :=
  (1/2)^(m + 1)

def inequality_condition : Prop :=
  ∃ (m n : ℕ), (T_n (n + 1) - m) / (T_n n - m) > c_m2 m

-- The main theorem that wraps everything
theorem math_proof_problem :
  (exists_general_formula_b) ∧
  (∃ N, find_smallest_N N) ∧
  (inequality_condition) :=
sorry

end math_proof_problem_l510_510168


namespace stickers_on_fifth_page_l510_510674

theorem stickers_on_fifth_page :
  ∀ (stickers : ℕ → ℕ),
    stickers 1 = 8 →
    stickers 2 = 16 →
    stickers 3 = 24 →
    stickers 4 = 32 →
    (∀ n, stickers (n + 1) = stickers n + 8) →
    stickers 5 = 40 :=
by
  intros stickers h1 h2 h3 h4 pattern
  apply sorry

end stickers_on_fifth_page_l510_510674


namespace determine_x_l510_510983

theorem determine_x (x y : ℤ) (h1 : x + 2 * y = 20) (h2 : y = 5) : x = 10 := 
by 
  sorry

end determine_x_l510_510983


namespace tangent_lines_range_of_a_l510_510380

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * (x^2 - x)

theorem tangent_lines (f g : ℝ → ℝ) (a : ℝ) (x : ℝ) :
  (f x = Real.exp x) →
  (g x = a * (x^2 - x)) →
  (a = -1) →
  (∃ m1 m2 : ℝ, ∀ y : ℝ, ∀ m : ℝ, 
    (y ≠ 0 → (f m - y) / (m - 0) = y / m ∧ y = Real.exp(m) * m)
    ∧ (y ≠ 0 → (g m - y) / (m - 0) = y / m ∧ y = (m - m^2) * m)) :=
sorry

theorem range_of_a (a : ℝ) (x : ℝ) :
  (f x = Real.exp x) →
  (g x = a * (x^2 - x)) →
  (∀ y : ℝ, f y + g y - Real.cos y ≥ 0) →
  (a = 1) :=
sorry

end tangent_lines_range_of_a_l510_510380


namespace female_officers_count_l510_510133

theorem female_officers_count (F : ℕ) (h1 : 0.285 * F = (3/5) * 265)
    (h2 : 325 = 325) (h3 : 265 = 265) (h4 : (3/5) * 265 = (3/5) * 265) :
    F = 558 := sorry

end female_officers_count_l510_510133


namespace no_periodic_sequence_first_non_zero_digit_from_right_l510_510066

theorem no_periodic_sequence_first_non_zero_digit_from_right:
  ¬ ∃ N : ℕ, ∃ T : ℕ, ∀ n, n ≥ N → (a_(n + T) = a_n) := sorry

end no_periodic_sequence_first_non_zero_digit_from_right_l510_510066


namespace binary_mod_4_remainder_l510_510210

theorem binary_mod_4_remainder :
  let n : ℕ := 0b111001011110 in  -- Lean allows binary literals with the 0b prefix
  n % 4 = 2 :=
by
  sorry

end binary_mod_4_remainder_l510_510210


namespace no_odd_stars_in_5_by_200_rect_l510_510311

theorem no_odd_stars_in_5_by_200_rect :
  ∀ (num_small_rects : ℕ) (small_rect_area : ℕ) (total_area : ℕ) (num_rows : ℕ) (num_cols : ℕ),
    num_small_rects = 500 →
    small_rect_area = 2 →
    num_rows = 5 →
    num_cols = 200 →
    total_area = num_small_rects * small_rect_area →
    ¬ (∀ i, i < num_rows → (total_area / num_rows) % 2 = 1) ∧
    ¬ (∀ j, j < num_cols → (total_area / num_cols) % 2 = 1) :=
  by
    intros num_small_rects small_rect_area total_area num_rows num_cols
    assume h1 h2 h3 h4 h5
    sorry

end no_odd_stars_in_5_by_200_rect_l510_510311


namespace find_y_l510_510526

theorem find_y (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 10) (hx : x = -4) : y = 41 / 2 :=
by
  sorry

end find_y_l510_510526


namespace vanessa_ribbon_length_l510_510929

theorem vanessa_ribbon_length :
  let s := 12 * real.sqrt 5
  let perimeter := 3 * s
  let estimated_ribbon_length := perimeter + 3
  estimated_ribbon_length ≈ 63 :=
by {
  sorry
}

end vanessa_ribbon_length_l510_510929


namespace abs_w_eq_one_l510_510850

noncomputable
def complex_z : ℂ := ((-7 + 15*Complex.i)^2 * (18 - 4*Complex.i)^3) / (5 + 2*Complex.i)

noncomputable
def w : ℂ := Complex.conj complex_z / complex_z

theorem abs_w_eq_one : Complex.abs w = 1 := by
  sorry

end abs_w_eq_one_l510_510850


namespace dice_probability_l510_510848

theorem dice_probability (D1 D2 D3 : ℕ) (hD1 : 0 ≤ D1) (hD1' : D1 < 10) (hD2 : 0 ≤ D2) (hD2' : D2 < 10) (hD3 : 0 ≤ D3) (hD3' : D3 < 10) :
  ∃ p : ℚ, p = 1 / 10 :=
by
  let outcomes := 10 * 10 * 10
  let favorable := 100
  let expected_probability : ℚ := favorable / outcomes
  use expected_probability
  sorry

end dice_probability_l510_510848


namespace maximum_k_for_coloring_l510_510353

theorem maximum_k_for_coloring :
  ∃ k : ℕ,
    (∀ R1 R2 R3 C1 C2 C3 : fin 6,
      R1 ≠ R2 → R1 ≠ R3 → R2 ≠ R3 →
      C1 ≠ C2 → C1 ≠ C3 → C2 ≠ C3 →
      ∃ r c : fin 6, r ∈ {R1, R2, R3} ∧ c ∈ {C1, C2, C3} ∧ ¬colored r c) ∧
    (∀ n > k,
      ¬ (∀ R1 R2 R3 C1 C2 C3 : fin 6,
        R1 ≠ R2 → R1 ≠ R3 → R2 ≠ R3 →
        C1 ≠ C2 → C1 ≠ C3 → C2 ≠ C3 →
        ∃ r c : fin 6, r ∈ {R1, R2, R3} ∧ c ∈ {C1, C2, C3} ∧ ¬colored r c)) :=
begin
  sorry  -- proof should be constructed here
end

end maximum_k_for_coloring_l510_510353


namespace line_eq_slope_form_l510_510510

theorem line_eq_slope_form (a b c : ℝ) (h : b ≠ 0) :
    ∃ k l : ℝ, ∀ x y : ℝ, (a * x + b * y + c = 0) ↔ (y = k * x + l) := 
sorry

end line_eq_slope_form_l510_510510


namespace zoe_recycled_correctly_l510_510233

-- Let Z be the number of pounds recycled by Zoe
def pounds_by_zoe (total_points : ℕ) (friends_pounds : ℕ) (pounds_per_point : ℕ) : ℕ :=
  total_points * pounds_per_point - friends_pounds

-- Given conditions
def total_points : ℕ := 6
def friends_pounds : ℕ := 23
def pounds_per_point : ℕ := 8

-- Lean statement for the proof problem
theorem zoe_recycled_correctly : pounds_by_zoe total_points friends_pounds pounds_per_point = 25 :=
by
  -- proof to be provided here
  sorry

end zoe_recycled_correctly_l510_510233


namespace negate_universal_statement_l510_510991

theorem negate_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negate_universal_statement_l510_510991


namespace solution_set_of_inequality_l510_510012

theorem solution_set_of_inequality 
  {f : ℝ → ℝ}
  (hf : ∀ x y : ℝ, x < y → f x > f y)
  (hA : f 0 = -2)
  (hB : f (-3) = 2) :
  {x : ℝ | |f (x - 2)| > 2 } = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end solution_set_of_inequality_l510_510012


namespace total_bathing_suits_l510_510619

theorem total_bathing_suits (men_women_bathing_suits : Nat)
                            (men_bathing_suits : Nat := 14797)
                            (women_bathing_suits : Nat := 4969) :
    men_bathing_suits + women_bathing_suits = 19766 := by
  sorry

end total_bathing_suits_l510_510619


namespace min_value_geometric_seq_l510_510461

theorem min_value_geometric_seq (a : ℕ → ℝ) (r : ℝ) (n : ℕ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = a n * r)
  (h3 : a 5 * a 4 * a 2 * a 1 = 16) :
  a 1 + a 5 = 4 :=
sorry

end min_value_geometric_seq_l510_510461


namespace nursing_home_medicine_boxes_l510_510876

theorem nursing_home_medicine_boxes (vitamins supplements total : ℕ) (hvitamins : vitamins = 472) (hsupplements : supplements = 288) : vitamins + supplements = total → total = 760 :=
by
  intros
  rw [hvitamins, hsupplements]
  exact h
  sorry

end nursing_home_medicine_boxes_l510_510876


namespace animal_group_divisor_l510_510974

theorem animal_group_divisor (cows sheep goats total groups : ℕ)
    (hc : cows = 24) 
    (hs : sheep = 7) 
    (hg : goats = 113) 
    (ht : total = cows + sheep + goats) 
    (htotal : total = 144) 
    (hdiv : groups ∣ total) 
    (hexclude1 : groups ≠ 1) 
    (hexclude144 : groups ≠ 144) : 
    ∃ g, g = groups ∧ g ∈ [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72] :=
  by 
  sorry

end animal_group_divisor_l510_510974


namespace train_and_car_combined_time_l510_510283

noncomputable def combined_time (car_time : ℝ) (extra_time : ℝ) : ℝ :=
  car_time + (car_time + extra_time)

theorem train_and_car_combined_time : 
  ∀ (car_time : ℝ) (extra_time : ℝ), car_time = 4.5 → extra_time = 2.0 → combined_time car_time extra_time = 11 :=
by
  intros car_time extra_time hcar hextra
  sorry

end train_and_car_combined_time_l510_510283


namespace find_f2016_l510_510023

def f : ℤ → ℤ
| x => if x < 0 then x - 1 else f(x - 1) + 1

theorem find_f2016 : f 2016 = 2015 := by
  sorry

end find_f2016_l510_510023


namespace difference_in_radii_l510_510911

noncomputable def spheres (r₁ r₂ : ℝ) : Prop :=
  (4 * π * r₁ ^ 2 - 4 * π * r₂ ^ 2 = 48 * π) ∧
  (2 * π * r₁ + 2 * π * r₂ = 12 * π)

theorem difference_in_radii (r₁ r₂ : ℝ) (h : spheres r₁ r₂) : |r₁ - r₂| = 2 :=
by
  sorry

end difference_in_radii_l510_510911


namespace smallest_digit_for_divisibility_by_11_l510_510721

-- Definitions based on conditions
def digit_sum_odd_positions (d : ℕ) : ℕ := 8 + 2 + d + 7
def digit_sum_even_positions : ℕ := 5 + 1 + 8 + 4

-- Main statement for the proof
theorem smallest_digit_for_divisibility_by_11 : 
  ∃ d : ℕ, digit_sum_odd_positions d - digit_sum_even_positions ≡ 0 [MOD 11] ∧ 
           d ≥ 0 ∧ d ≤ 9 ∧ 
           ∀ d' : ℕ, (d' ≥ 0 ∧ d' ≤ 9 ∧ digit_sum_odd_positions d' - digit_sum_even_positions ≡ 0 [MOD 11]) → d ≤ d' := 
begin
  sorry
end

end smallest_digit_for_divisibility_by_11_l510_510721


namespace marie_needs_8_days_to_pay_for_cash_register_l510_510504

-- Definitions of the conditions
def cost_of_cash_register : ℕ := 1040
def price_per_loaf : ℕ := 2
def loaves_per_day : ℕ := 40
def price_per_cake : ℕ := 12
def cakes_per_day : ℕ := 6
def daily_rent : ℕ := 20
def daily_electricity : ℕ := 2

-- Derive daily income and expenses
def daily_income : ℕ := (price_per_loaf * loaves_per_day) + (price_per_cake * cakes_per_day)
def daily_expenses : ℕ := daily_rent + daily_electricity
def daily_profit : ℕ := daily_income - daily_expenses

-- Define days needed to pay for the cash register
def days_needed : ℕ := cost_of_cash_register / daily_profit

-- Proof goal
theorem marie_needs_8_days_to_pay_for_cash_register : days_needed = 8 := by
  sorry

end marie_needs_8_days_to_pay_for_cash_register_l510_510504


namespace average_episodes_per_year_l510_510617

theorem average_episodes_per_year (total_years : ℕ) (n1 n2 n3 e1 e2 e3 : ℕ) 
  (h1 : total_years = 14)
  (h2 : n1 = 8) (h3 : e1 = 15)
  (h4 : n2 = 4) (h5 : e2 = 20)
  (h6 : n3 = 2) (h7 : e3 = 12) :
  (n1 * e1 + n2 * e2 + n3 * e3) / total_years = 16 := by
  -- Skip the proof steps
  sorry

end average_episodes_per_year_l510_510617


namespace place_mats_length_l510_510645

theorem place_mats_length (r : ℝ) (w : ℝ) (n : ℕ) (x : ℝ) :
  r = 4 ∧ w = 1 ∧ n = 6 ∧
  (∀ i, i < n → 
    let inner_corners_touch := true in
    place_mat_placement_correct r w x i inner_corners_touch) →
  x = (3 * real.sqrt 7 - real.sqrt 3) / 2 :=
by sorry

end place_mats_length_l510_510645


namespace chromium_percentage_bounds_l510_510499

noncomputable def new_alloy_chromium_bounds : Prop :=
  ∃ (x y z k : ℝ), 
    (x + y + z = 1) ∧ 
    (0.9 * x + 0.3 * z = 0.45) ∧ 
    (0.4 * x + 0.1 * y + 0.5 * z = k) ∧ 
    (0.25 ≤ k ∧ k ≤ 0.4)

theorem chromium_percentage_bounds : new_alloy_chromium_bounds :=
by
  sorry

end chromium_percentage_bounds_l510_510499


namespace rectangle_area_correct_l510_510543

def side_of_square : ℕ := 69
def area_of_square : ℕ := side_of_square * side_of_square
def radius_of_circle : ℕ := side_of_square
def length_of_rectangle : ℕ := (2 * radius_of_circle) / 3
def breadth_of_rectangle : ℕ := 13
def area_of_rectangle : ℕ := length_of_rectangle * breadth_of_rectangle

theorem rectangle_area_correct : 
  side_of_square * side_of_square = 4761 → 
  length_of_rectangle = (2 * radius_of_circle) / 3 →
  radius_of_circle = side_of_square →
  breadth_of_rectangle = 13 →
  area_of_rectangle = 598 :=
by
  intros 
  unfold side_of_square area_of_square radius_of_circle length_of_rectangle breadth_of_rectangle area_of_rectangle
  sorry

end rectangle_area_correct_l510_510543


namespace infinitely_many_good_pairs_l510_510316

def is_triangular (t : ℕ) : Prop :=
  ∃ n : ℕ, t = n * (n + 1) / 2

theorem infinitely_many_good_pairs :
  ∃ (a b : ℕ), (0 < a) ∧ (0 < b) ∧ 
  ∀ t : ℕ, is_triangular t ↔ is_triangular (a * t + b) :=
sorry

end infinitely_many_good_pairs_l510_510316


namespace peter_wins_if_and_only_if_n_is_odd_l510_510934

-- Define the problem parameters and assumptions
variable (n : ℕ) (k : ℕ) 
variable (empty_cups : Fin 2n → Bool)
variable (peter_wins : Bool)

-- Define the conditions
def symmetric (x y : Fin 2n) : Prop := x.val + y.val = 2n
def can_pour_tea (x : Fin 2n) : Prop := empty_cups x = true
def can_pour_symmetric_tea (x y : Fin 2n) : Prop := symmetric x y ∧ empty_cups x = true ∧ empty_cups y = true

-- Define a property stating Peter wins if and only if n is odd
def peter_wins_iff (n : ℕ) : Prop := peter_wins = (Odd n)

theorem peter_wins_if_and_only_if_n_is_odd :
  (∀ n, (∀ (x : Fin 2n), can_pour_tea empty_cups x ∨ (∃ y, can_pour_symmetric_tea empty_cups x y) ∨ ¬ can_pour_tea empty_cups x) →
  peter_wins_iff n) :=
by
  sorry

end peter_wins_if_and_only_if_n_is_odd_l510_510934


namespace area_of_triangle_ABC_l510_510570

noncomputable def area_triangle_ABC 
  (A B C : Point) 
  (O O' : Point) 
  (r1 r2 : ℝ) 
  (tangent : Bool) 
  (radius_smaller : r1 = 3) 
  (radius_larger : r2 = 4) 
  (dist_7 : dist O O' = 7)
  (AB_BC_ratio : dist A B = 2 * dist B C) 
  (non_congruent : dist A B ≠ dist B C) 
  (ABC_tangent : tangent = True) 
  (top_vertex : A.y > B.y ∧ A.y > C.y ∧ B.y < C.y ∧ C.y < B.y): ℝ := by
  sorry

theorem area_of_triangle_ABC
  (A B C : Point) 
  (O O' : Point) 
  (r1 r2 : ℝ) 
  (tangent : Bool) 
  (radius_smaller : r1 = 3) 
  (radius_larger : r2 = 4) 
  (dist_7 : dist O O' = 7)
  (AB_BC_ratio : dist A B = 2 * dist B C) 
  (non_congruent : dist A B ≠ dist B C) 
  (ABC_tangent : tangent = True) 
  (top_vertex : A.y > B.y ∧ A.y > C.y ∧ B.y < C.y ∧ C.y < B.y)
  : area_triangle_ABC A B C O O' r1 r2 tangent radius_smaller radius_larger dist_7 AB_BC_ratio non_congruent ABC_tangent top_vertex = 124.2 := by
  sorry

end area_of_triangle_ABC_l510_510570


namespace impossible_arrangement_l510_510839

theorem impossible_arrangement :
  ¬ ∃ (f : Fin 600 → Fin 600 → Int),
    (∀ i j, abs (f i j) = 1) ∧
    abs (Finset.sum (Finset.univ.image (λ ⟨i, j⟩, f i j))) < 90000 ∧
    (∀ (i j : Fin 597),
      abs (Finset.sum (Finset.range 4).bind (λ i', Finset.range 6).image (λ j', f (i + i') (j + j'))) > 4 ∧
      abs (Finset.sum (Finset.range 6).bind (λ i', Finset.range 4).image (λ j', f (i + i') (j + j'))) > 4) :=
sorry

end impossible_arrangement_l510_510839


namespace willie_gave_emily_7_stickers_l510_510593

theorem willie_gave_emily_7_stickers (initial_stickers : ℕ) (final_stickers : ℕ) (given_stickers : ℕ) 
  (h1 : initial_stickers = 36) (h2 : final_stickers = 29) (h3 : given_stickers = initial_stickers - final_stickers) : 
  given_stickers = 7 :=
by
  rw [h1, h2] at h3 -- Replace initial_stickers with 36 and final_stickers with 29 in h3
  exact h3  -- given_stickers = 36 - 29 which is equal to 7.


end willie_gave_emily_7_stickers_l510_510593


namespace fifth_term_of_geometric_sequence_l510_510318

noncomputable def sequence (x : ℝ) (n : ℕ) : ℝ :=
  n * x + (n + 1)

theorem fifth_term_of_geometric_sequence {x : ℝ} 
  (h : ∀ n, (sequence x (n + 1)) / (sequence x n) = (sequence x (n + 2)) / (sequence x (n + 1))) :
  sequence x 5 = -1 :=
sorry

end fifth_term_of_geometric_sequence_l510_510318


namespace angle_measure_proof_l510_510954

noncomputable def angle_measure (x : ℝ) : Prop :=
  let supplement := 180 - x
  let complement := 90 - x
  supplement = 8 * complement

theorem angle_measure_proof : ∃ x : ℝ, angle_measure x ∧ x = 540 / 7 :=
by
  have angle_eq : ∀ x, angle_measure x ↔ (180 - x = 8 * (90 - x)) := by
    intro x
    dsimp [angle_measure]
    rfl
  use 540 / 7
  rw angle_eq
  split
  · dsimp
    linarith
  · rfl

end angle_measure_proof_l510_510954


namespace problem_statement_l510_510161

noncomputable def c := 3 + Real.sqrt 21
noncomputable def d := 3 - Real.sqrt 21

theorem problem_statement : 
  (c + 2 * d) = 9 - Real.sqrt 21 :=
by
  sorry

end problem_statement_l510_510161


namespace minimum_value_expression_l510_510349

theorem minimum_value_expression : 
  ∀ (a b : ℝ), (a > 0) → (b > 0) → 
  ( ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ 
    (a_0 b_0 : ℝ) = 4
  (3 * a * b - 6 * b + a * (1 - a))^2 + (9 * b^2 + 2 * a + 3 * b * (1 - a))^2 / (a^2 + 9 * b^2) = 4 
sory

end minimum_value_expression_l510_510349


namespace matrix_mul_correct_l510_510688

def matrix_mult (A B : Matrix (Fin 2) (Fin 2) Int) : Matrix (Fin 2) (Fin 2) Int :=
  λ i j, ∑ k, A i k * B k j

def A : Matrix (Fin 2) (Fin 2) Int := ![
  ![3, -1],
  ![4, 2]
]

def B : Matrix (Fin 2) (Fin 2) Int := ![
  ![0, 6],
  ![-2, 3]
]

def C : Matrix (Fin 2) (Fin 2) Int := ![
  ![2, 15],
  ![-4, 30]
]

theorem matrix_mul_correct : matrix_mult A B = C := 
by sorry

end matrix_mul_correct_l510_510688


namespace solve_system_l510_510785

theorem solve_system (x y : ℝ) (hx1 : 0 < x) (hx2 : x ≠ 1) (hy1 : 0 < y) (hy2 : y ≠ 1) (hxy_sum : x + y = 12)
  (hlog_eq : 2 * (2 * log y^2 x - log (1 / x) y) = 5) : (x = 9 ∧ y = 3) ∨ (x = 3 ∧ y = 9) :=
by
  sorry

end solve_system_l510_510785


namespace planes_parallel_l510_510276

def Plane (α : Type*) := α → α → Prop

variables {α β : Type*} [Plane α] [Plane β]

-- Conditions: defining that planes α and β are parallel based on skew lines cond.
def skew_lines_parallel_planes (a b : α) (h₁ : a ∈ α) (h₂ : b ∈ β) (h₃ : ¬a ∥ b) (h₄ : a ∥ β) (h₅ : b ∥ α) : Prop :=
  (∀ p q : α, (α p q) → (β p q) → (p = q))

theorem planes_parallel (a b : α) (h₁ : a ∈ α) (h₂ : b ∈ β) (h₃ : ¬a ∥ b) (h₄ : a ∥ β) (h₅ : b ∥ α) : (−α = β (skew_lines_parallel_planes a b h₁ h₂ h₃ h₄ h₅)) RetProp :=
  sorry

end planes_parallel_l510_510276


namespace find_a_and_b_monotonicity_l510_510248

-- Given conditions:
variables (f : ℝ → ℝ)
variables (a b : ℝ)

-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)

-- Condition 2: f(2) = a + b^2
def value_at_2 (f : ℝ → ℝ) (a b : ℝ) : Prop := f(2) = a + b^2

-- Proof goals:
theorem find_a_and_b (h_odd : odd_function f) (h_val : value_at_2 f a b) :
  a = 2 ∧ b = 0 :=
sorry

theorem monotonicity (h_odd : odd_function f) (h_val : value_at_2 f a b) (ha : a = 2) (hb : b = 0) :
  ∀ x1 x2 : ℝ, x1 < x2 ∧ x2 ≤ -1 → f(x1) < f(x2) :=
sorry

end find_a_and_b_monotonicity_l510_510248


namespace smallest_period_of_f_range_of_transformed_f_l510_510395

noncomputable def f (x : ℝ) : ℝ := cos x - 8 * (cos (x / 4))^4

theorem smallest_period_of_f : smallest_period f = 4 * Real.pi :=
sorry

theorem range_of_transformed_f : 
  Set.range (λ x, f (2 * x - (Real.pi / 6))) =
    Set.Icc (-5 : ℝ) (-4 : ℝ) :=
sorry

end smallest_period_of_f_range_of_transformed_f_l510_510395


namespace count_five_letter_words_l510_510086

theorem count_five_letter_words : (26 ^ 4 = 456976) :=
by {
    sorry
}

end count_five_letter_words_l510_510086


namespace arithmetic_seq_formula_sum_first_n_terms_l510_510065

/-- Define the given arithmetic sequence an -/
def arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0       => a1
| (n + 1) => arithmetic_seq a1 d n + d

variable {a3 a7 : ℤ}
variable (a3_eq : arithmetic_seq 1 2 2 = 5)
variable (a7_eq : arithmetic_seq 1 2 6 = 13)

/-- Define the sequence bn -/
def b_seq (n : ℕ) : ℚ :=
  1 / ((2 * n + 1) * (arithmetic_seq 1 2 n))

/-- Define the sum of the first n terms of the sequence bn -/
def sum_b_seq : ℕ → ℚ
| 0       => 0
| (n + 1) => sum_b_seq n + b_seq (n + 1)
          
theorem arithmetic_seq_formula:
  ∀ (n : ℕ), arithmetic_seq 1 2 n = 2 * n - 1 :=
by
  intros
  sorry

theorem sum_first_n_terms:
  ∀ (n : ℕ), sum_b_seq n = n / (2 * n + 1) :=
by
  intros
  sorry

end arithmetic_seq_formula_sum_first_n_terms_l510_510065


namespace container_capacity_l510_510234

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 36 = 0.75 * C) : 
  C = 80 :=
sorry

end container_capacity_l510_510234


namespace inequality_proof_l510_510107

theorem inequality_proof (n : ℕ) 
  (x : Fin n → ℝ) 
  (h_pos : ∀ i, 0 < x i) 
  (h_sum : (∑ i, x i) = 1) :
  (∏ i, (1 / x i - 1)) ≥ (n - 1) ^ n :=
sorry

end inequality_proof_l510_510107


namespace lateral_area_of_cylinder_l510_510916

-- Define the conditions
def r : ℝ := 2 -- radius in cm
def h : ℝ := 5 -- height (generatrix) in cm

-- The theorem to prove the lateral area
theorem lateral_area_of_cylinder : 2 * real.pi * r * h = 20 * real.pi := by
  sorry

end lateral_area_of_cylinder_l510_510916


namespace irrational_pi_l510_510220

theorem irrational_pi :
  irrational real.pi := by
  sorry

end irrational_pi_l510_510220


namespace honey_servings_l510_510624

theorem honey_servings (total_honey : ℚ) (serving_size : ℚ) : 
  total_honey = 142/3 → serving_size = 10/3 → total_honey / serving_size = 71/5 := 
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end honey_servings_l510_510624


namespace brianchons_theorem_l510_510138

-- Given a circumscribed hexagon ABCDEF (i.e., there exists a circumscribed circle)
variables (A B C D E F : Type) [circumscribed_hexagon A B C D E F]

theorem brianchons_theorem (hex : circumscribed_hexagon A B C D E F) :
  intersects_at_single_point (diagonal A D) (diagonal B E) (diagonal C F) :=
sorry

end brianchons_theorem_l510_510138


namespace max_distance_l510_510544

-- Definitions of the functions.
def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 12)
def g (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x - Real.pi / 12)

-- The theorem stating the maximum value of |PQ|
theorem max_distance : ∀ t : ℝ, |f(t) - g(t)| ≤ 2 :=
sorry

end max_distance_l510_510544


namespace area_of_triangle_l510_510486

def vector_a : ℝ × ℝ := (4, 3)
def vector_b : ℝ × ℝ := (-6, 5)

theorem area_of_triangle :
  let parallelogram_area := (vector_a.1 * vector_b.2 - vector_a.2 * vector_b.1).abs
  let triangle_area := parallelogram_area / 2
  triangle_area = 19 := 
by 
  sorry

end area_of_triangle_l510_510486


namespace checkerboard_corner_sum_is_164_l510_510691

def checkerboard_sum_corners : ℕ :=
  let top_left := 1
  let top_right := 9
  let bottom_left := 73
  let bottom_right := 81
  top_left + top_right + bottom_left + bottom_right

theorem checkerboard_corner_sum_is_164 :
  checkerboard_sum_corners = 164 :=
by
  sorry

end checkerboard_corner_sum_is_164_l510_510691
