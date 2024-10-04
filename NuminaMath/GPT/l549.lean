import Mathlib
import Mathlib.Algebra.BigOperators.FinProd
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Field.Defs
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order.Nonneg
import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Algebra.Ring.Defs
import Mathlib.Analysis.Calculus.Continuity
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Sequences.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.NumberTheory.LcmGcd.Basic
import Mathlib.Probability
import Mathlib.Probability.Martingales.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Mathlib.Topology.Geometry
import algebra.pi_instances
import data.real.basic
import tactic

namespace modulus_of_complex_l549_549706

theorem modulus_of_complex (z : ℂ) (h : z = (4 - 2 * complex.I) / (1 - complex.I)) : complex.abs z = real.sqrt 10 := 
by 
  sorry

end modulus_of_complex_l549_549706


namespace shaded_percentage_is_50_l549_549551

noncomputable def shaded_percentage : real :=
  let total_squares := 6 * 6
  let shaded_squares_per_row := 3
  let total_rows := 6
  let shaded_squares := shaded_squares_per_row * total_rows
  (shaded_squares : real) / total_squares * 100

theorem shaded_percentage_is_50 :
  shaded_percentage = 50 :=
by
  sorry

end shaded_percentage_is_50_l549_549551


namespace cone_central_angle_l549_549361

theorem cone_central_angle (A : ℝ) (h : 3 * A = 3 * A):
  let base_area := A in
  let total_surface_area := 3 * base_area in
  let lateral_surface_area := total_surface_area - base_area in
  let angle := 240 in
  angle = 240 :=
by sorry

end cone_central_angle_l549_549361


namespace flight_duration_NY_CapeTown_l549_549645

theorem flight_duration_NY_CapeTown :
  let london_departure_time := 6 -- 6:00 a.m. Monday ET
  let flight_to_NY_duration := 18 -- 18 hours flight to New York
  let NY_arrival_time := (london_departure_time + flight_to_NY_duration) % 24  -- 12:00 a.m. Tuesday ET
  let capetown_arrival_time := 10 -- 10:00 a.m. Tuesday ET
  capetown_arrival_time - NY_arrival_time = 10 := 
by
  have london_departure_time : ℕ := 6,
  have flight_to_NY_duration : ℕ := 18,
  have NY_arrival_time : ℕ := (london_departure_time + flight_to_NY_duration) % 24,
  have capetown_arrival_time : ℕ := 10,
  have flight_duration := capetown_arrival_time - NY_arrival_time,
  show flight_duration = 10, from sorry

end flight_duration_NY_CapeTown_l549_549645


namespace probability_all_girls_is_correct_l549_549222

noncomputable def probability_all_girls : ℚ :=
  let total_members := 15
  let boys := 7
  let girls := 8
  let choose_3_from_15 := Nat.choose total_members 3
  let choose_3_from_8 := Nat.choose girls 3
  choose_3_from_8 / choose_3_from_15

theorem probability_all_girls_is_correct : 
  probability_all_girls = 8 / 65 := by
sorry

end probability_all_girls_is_correct_l549_549222


namespace domain_k_function_l549_549939

def k (x : ℝ) : ℝ := 1 / (x + 9) + 1 / (x^2 + 9) + 1 / (x^3 + 9)

theorem domain_k_function :
  ∀ x : ℝ, x ≠ -9 → x ≠ -(9)^(1/3) → 
  ∀ y : ℝ, y ∈ k x → y ∈ ((set.Iio (-9)) ∪ (set.Ioo (-9) (-((9)^(1/3)))) ∪ (set.Ioi (-((9)^(1/3))))) :=
sorry

end domain_k_function_l549_549939


namespace find_number_l549_549991

theorem find_number (x : ℝ) (h : x / 0.025 = 40) : x = 1 := 
by sorry

end find_number_l549_549991


namespace trapezoid_area_l549_549639

def line_eq (p1 p2 : LEAN.weights Float) : String :=
  let m := (p2.2 - p1.2) / (p2.1 - p1.1)
  let c := p1.2 - m * p1.1 in
  s!"y = {m}x + {c}"

def intersection_point (m1 m2 c1 c2 : Float) : LEAN.weights Float :=
  let x := (c2 - c1) / (m1 - m2)
  let y := m1 * x + c1 in
  (x, y)

def area_trapezoid (b1 b2 h : Float) : Float :=
  0.5 * (b1 + b2) * h

theorem trapezoid_area :
  let p1 := (0, 3)
  let p2 := (5, 0)
  let p3 := (2, 6)
  let l1 := line_eq p1 p2
  let l2 := line_eq p3 p2
  let (m1, b1) := (-3/5 : Float, 3 : Float)
  let (m2, b2) := (-2 : Float, 12 : Float)
  let (x_int, y_int) := intersection_point m1 m2 b1 b2
  let b1_area := 3
  let b2_area := 12
  let h := 5
  area_trapezoid b1_area b2_area h = 37.5 := 
by 
  sorry

end trapezoid_area_l549_549639


namespace time_to_fill_tank_l549_549203

variable (A B : ℝ) (rate_A : A = 1 / 12) (rate_B : B = 3 * (1 / 12))

theorem time_to_fill_tank (open_pipes_rate : A + B): open_pipes_rate = 1 / 3 → (1 / open_pipes_rate) = 3 :=
begin
  intro h,
  rw h,
  norm_num,
end

end time_to_fill_tank_l549_549203


namespace find_number_l549_549118

theorem find_number : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 :=
by {
  sorry
}

end find_number_l549_549118


namespace smallest_four_digit_palindrome_divisible_by_3_l549_549945

theorem smallest_four_digit_palindrome_divisible_by_3 :
  ∃ (n : ℕ), (n = 2112) ∧ (1000 ≤ n) ∧ (n < 10000) ∧
    (∀ m : ℕ, (1000 ≤ m) → (m < 10000) → 
        (m / 1000 = m % 10) ∧ ((m / 100) % 10 = (m / 10) % 10) →
        (m % 3 ≠ 0 ∨ m ≥ n)) := 
begin
  sorry
end

end smallest_four_digit_palindrome_divisible_by_3_l549_549945


namespace perpendicular_tangents_intersection_points_l549_549684

-- Define the angle between the asymptotes
variable (ϕ : ℝ) -- represents the angle in degrees

-- Define the proof statement
theorem perpendicular_tangents_intersection_points
  (h : ϕ < 90 ∨ ϕ ≥ 90) :
  (ϕ < 90 → ∃ c : Set (ℝ × ℝ), is_circle c ∧ c.excludes_four_points_on_asymptotes ∧ center_of_circle c = center_of_hyperbola) ∧
  (ϕ ≥ 90 → intersection_points = ∅) :=
by
  sorry

end perpendicular_tangents_intersection_points_l549_549684


namespace largest_four_digit_number_divisible_by_33_l549_549940

theorem largest_four_digit_number_divisible_by_33 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (33 ∣ n) ∧ ∀ m : ℕ, (1000 ≤ m ∧ m < 10000 ∧ 33 ∣ m → m ≤ 9999) :=
by
  sorry

end largest_four_digit_number_divisible_by_33_l549_549940


namespace number_of_flower_sets_l549_549808

theorem number_of_flower_sets (total_flowers : ℕ) (flowers_per_set : ℕ) (sets : ℕ) 
  (h1 : total_flowers = 270) 
  (h2 : flowers_per_set = 90) 
  (h3 : sets = total_flowers / flowers_per_set) : 
  sets = 3 := 
by 
  sorry

end number_of_flower_sets_l549_549808


namespace age_sum_problem_l549_549523

noncomputable def problem := sorry

theorem age_sum_problem (A B C D : ℕ) (h_sum : A + B + C + D = 100)
  (h_A : A = 32) (h_ratios : A + B = 3 * (C + D)) (h_diff : C = D + 3) :
  B + D = 54 := sorry

end age_sum_problem_l549_549523


namespace sum_of_5_vertical_squares_is_20_l549_549660

structure SquareArrangement :=
  (A1 A2 A3 B1 B2 B3 C1 C2 C3 : ℕ)
  (unique_numbers : {A1, A2, A3, B1, B2, B3, C1, C2, C3}.card = 9)
  (valid_numbers : ∀ x ∈ {A1, A2, A3, B1, B2, B3, C1, C2, C3}, x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (C3_value : C3 = 7)
  (bottom_sum : B1 + B2 + C1 + C2 + C3 = 20)

theorem sum_of_5_vertical_squares_is_20 (s : SquareArrangement) : 
  s.A1 + s.B1 + s.C1 + s.B2 + s.C3 = 20 :=
begin
  -- Proof omitted, as per instruction
  sorry
end

end sum_of_5_vertical_squares_is_20_l549_549660


namespace tens_digits_of_squares_l549_549334

theorem tens_digits_of_squares (a b : ℕ) (x y : ℕ) (ha : nat.sqrt a ^ 2 = a) (hb : nat.sqrt b ^ 2 = b) 
    (a_units_digit : a % 10 = 1) (a_tens_digit : a / 10 % 10 = x) 
    (b_units_digit : b % 10 = 6) (b_tens_digit : b / 10 % 10 = y) : 
  (x % 2 = 0) ∧ (y % 2 = 1) := 
sorry

end tens_digits_of_squares_l549_549334


namespace num_solutions_3x_plus_2y_eq_806_l549_549280

theorem num_solutions_3x_plus_2y_eq_806 :
  (∃ y : ℕ, ∃ x : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 806) ∧
  ((∃ t : ℤ, x = 268 - 2 * t ∧ y = 1 + 3 * t) ∧ (∃ t : ℤ, 0 ≤ t ∧ t ≤ 133)) :=
sorry

end num_solutions_3x_plus_2y_eq_806_l549_549280


namespace ball_distribution_l549_549924

theorem ball_distribution : 
  let red_balls := 10
  let black_balls := 15
  let white_balls := 20
  let min_balls_per_boy := 2
  let min_balls_per_girl := 3
  let total_students := 3 
  let remaining_red := red_balls - total_students * min_balls_per_boy - min_balls_per_girl
  let remaining_black := black_balls - total_students * min_balls_per_boy - min_balls_per_girl
  let remaining_white := white_balls - total_students * min_balls_per_boy - min_balls_per_girl
  in combinatorial_count remaining_red total_students * 
     combinatorial_count remaining_black total_students * 
     combinatorial_count remaining_white total_students = 47250 :=
by
  sorry

end ball_distribution_l549_549924


namespace mean_of_five_numbers_l549_549170

theorem mean_of_five_numbers (S : ℚ) (n : ℕ) (h1 : S = 3/4) (h2 : n = 5) :
  (S / n) = 3/20 :=
by
  rw [h1, h2]
  sorry

end mean_of_five_numbers_l549_549170


namespace jenna_bill_eel_ratio_l549_549427

theorem jenna_bill_eel_ratio:
  ∀ (B : ℕ), (B + 16 = 64) → (16 / B = 1 / 3) :=
by
  intros B h
  sorry

end jenna_bill_eel_ratio_l549_549427


namespace sum_of_n_and_k_l549_549894

theorem sum_of_n_and_k (n k : ℕ) 
  (h1 : (n.choose k) * 3 = (n.choose (k + 1)))
  (h2 : (n.choose (k + 1)) * 2 = (n.choose (k + 2))) :
  n + k = 13 :=
by
  sorry

end sum_of_n_and_k_l549_549894


namespace other_function_value_at_20_l549_549540

def linear_function (k b : ℝ) (x : ℝ) : ℝ :=
  k * x + b

theorem other_function_value_at_20
    (k1 k2 b1 b2 : ℝ)
    (h_intersect : linear_function k1 b1 2 = linear_function k2 b2 2)
    (h_diff_at_8 : abs (linear_function k1 b1 8 - linear_function k2 b2 8) = 8)
    (h_y1_at_20 : linear_function k1 b1 20 = 100) :
  linear_function k2 b2 20 = 76 ∨ linear_function k2 b2 20 = 124 :=
sorry

end other_function_value_at_20_l549_549540


namespace card_cost_l549_549503

theorem card_cost (
  x y : ℕ
) (h1 : 1.25 * x + 1.75 * y = 18) 
: (1.25 * x + 1.75 * y = 18) :=
by
  exact h1

end card_cost_l549_549503


namespace antifreeze_concentration_pure_l549_549982

theorem antifreeze_concentration_pure (
    (final_mixture_volume : ℝ),
    (final_concentration : ℝ),
    (certain_antifreeze_volume : ℝ),
    (ten_percent_mixture_concentration : ℝ)
    :
    final_mixture_volume = 55 ∧ final_concentration = 0.20 ∧ certain_antifreeze_volume = 6.11 ∧ ten_percent_mixture_concentration = 0.10)
    :
    C = 100 :=
begin
    sorry
end

end antifreeze_concentration_pure_l549_549982


namespace integral_2x_minus_x_sq_integral_3_minus_2x_integral_one_third_x_sq_integral_cosine_l549_549263

open Real

theorem integral_2x_minus_x_sq : ∫ x in 0..1, (2 * x - x^2) = 2/3 := 
    sorry

theorem integral_3_minus_2x : ∫ x in 2..4, (3 - 2 * x) = -6 := 
    sorry

theorem integral_one_third_x_sq : ∫ x in 0..1, (1/3) * x^2 = 1/9 := 
    sorry

theorem integral_cosine : ∫ x in 0..(2 * π), cos x = 0 := 
    sorry

end integral_2x_minus_x_sq_integral_3_minus_2x_integral_one_third_x_sq_integral_cosine_l549_549263


namespace ratio_of_side_lengths_l549_549908

theorem ratio_of_side_lengths (a b c : ℕ) (ha : a = 5) (hb : b = 6) (hc : c = 16) (h_ratio : (75 : ℝ) / 128 = (5 : ℝ) * (real.sqrt 6) / 16) : a + b + c = 27 := by
  sorry

end ratio_of_side_lengths_l549_549908


namespace car_travel_first_hour_l549_549915

-- Define the conditions as variables and the ultimate equality to be proved
theorem car_travel_first_hour (x : ℕ) (h : 12 * x + 132 = 612) : x = 40 :=
by
  -- Proof will be completed here
  sorry

end car_travel_first_hour_l549_549915


namespace round_3_46_to_nearest_tenth_l549_549866

theorem round_3_46_to_nearest_tenth :
  (nearest_tenth 3.46) = 3.5 := 
by
  sorry

end round_3_46_to_nearest_tenth_l549_549866


namespace true_propositions_l549_549350

variable (x y : ℝ)

def p : Prop := x > y → -x < -y
def q : Prop := x > y → x^2 > y^2

theorem true_propositions (hp : p) (hq : ¬q) : (p ∨ q) ∧ (p ∧ ¬q) :=
by {
  sorry
}

end true_propositions_l549_549350


namespace hexagon_area_l549_549609

theorem hexagon_area (r : ℝ) (h : r = 2) : 
  let hex_area := 6 * (r^2 * (Real.sqrt 3) / 4)
  in hex_area = 6 * Real.sqrt 3 :=
by
  let s := 2 * r / 2, -- side length of the hexagon
  have triangle_area := (s^2 * (Real.sqrt 3) / 4),
  have total_area := 6 * triangle_area,
  calc total_area = 6 * (s^2 * (Real.sqrt 3) / 4) : by sorry
             ... = 6 * (2^2 * (Real.sqrt 3) / 4)  : by sorry
             ... = 6 * Real.sqrt 3                : by sorry

end hexagon_area_l549_549609


namespace row_coverage_l549_549995

/-- Define the number of rows and columns in the rectangular board -/
def rows := 10
def cols := 5

/-- Define the sequence of shaded squares -/
def shadedSquares : List Nat :=
  let rec aux (n skip acc) :=
    if n >= rows * cols then acc
    else aux (n + skip + 1) (skip + 1) (acc ++ [n])
  aux 1 2 []

/-- Helper function to determine row of a given square number -/
def squareRow (n : Nat) : Nat :=
  (n / cols) + 1

/-- Main theorem to determine the least number to have at least 1 shaded square in each row -/
theorem row_coverage : 
  ∃ n, ∀ row, row ≤ rows → ∃ k, shadedSquares.take n !k ≤ row * cols ∧ shadedSquares.take n !k > (row - 1) * cols :=
sorry

end row_coverage_l549_549995


namespace misha_total_students_l549_549468

-- Definitions based on the conditions
def misha_best_rank : ℕ := 75
def misha_worst_rank : ℕ := 75

-- Statement of the theorem to be proved
theorem misha_total_students (misha_is_best : misha_best_rank = 75) (misha_is_worst : misha_worst_rank = 75) : 
  (misha_best_rank - 1) + (misha_worst_rank - 1) + 1 = 149 :=
by
  sorry

end misha_total_students_l549_549468


namespace number_of_ordered_arrays_eq_6_l549_549628

theorem number_of_ordered_arrays_eq_6 (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|) →
  (∃ p : finset (ℝ × ℝ × ℝ), p.card = 6 ∧ ∀ q ∈ p, q = (0, 0, 1) ∨ q = (0, 1, 0) ∨ q = (1, 0, 0) ∨ q = (0, 0, -1) ∨ q = (0, -1, 0) ∨ q = (-1, 0, 0)) :=
sorry

end number_of_ordered_arrays_eq_6_l549_549628


namespace apples_difference_l549_549070

def jimin_apples : ℕ := 7
def grandpa_apples : ℕ := 13
def younger_brother_apples : ℕ := 8
def younger_sister_apples : ℕ := 5

theorem apples_difference :
  grandpa_apples - younger_sister_apples = 8 :=
by
  sorry

end apples_difference_l549_549070


namespace vector_basis_l549_549749

def vector_eq (a b : ℝ) (e₁ e₂ : ℝ × ℝ) : Prop :=
  ∃ λ μ : ℝ, (a, b) = λ • e₁ + μ • e₂

theorem vector_basis (u v : ℝ × ℝ)
  (h : ∃ λ μ : ℝ, (-3, 4) = λ • (-1, 2 : ℝ × ℝ) + μ • (3, -1 : ℝ × ℝ)) :
  ¬collinear (-1, 2 : ℝ × ℝ) (3, -1 : ℝ × ℝ) :=
  sorry

end vector_basis_l549_549749


namespace discount_rate_is_correct_l549_549960

def marked_price : ℝ := 200
def selling_price : ℝ := 120
def discount (M S : ℝ) : ℝ := M - S
def rate_of_discount (D M : ℝ) : ℝ := (D / M) * 100

theorem discount_rate_is_correct :
  rate_of_discount (discount marked_price selling_price) marked_price = 40 :=
by
  -- Proof goes here
  sorry

end discount_rate_is_correct_l549_549960


namespace soccer_ball_problem_l549_549016

-- Variables for soccer ball prices
variables (price_A price_B : ℕ)

-- Condition: Brand A is 10 yuan more than Brand B
def condition1 : Prop := price_A = price_B + 10

-- Condition: Prices adding up to total cost for specific quantities
def condition2 : Prop := 20 * price_A + 15 * price_B = 3350

-- Variables for numbers of soccer balls
variables (m : ℕ)

-- Condition: Total number of soccer balls
def condition3 : Prop := m + (50 - m) = 50

-- Condition: Total cost constraint
def condition4 : Prop := 100 * m + 90 * (50 - m) ≤ 4650

-- Proof objectives
def conclusion1 : Prop := price_A = 100 ∧ price_B = 90
def conclusion2 : Prop := m ≤ 15

-- Theorem statement
theorem soccer_ball_problem :
  condition1 → condition2 → condition3 → condition4 → conclusion1 ∧ conclusion2 :=
by
  intro h1 h2 h3 h4
  sorry

end soccer_ball_problem_l549_549016


namespace misha_students_l549_549470

theorem misha_students : 
  ∀ (n : ℕ),
  (n = 74 + 1 + 74) ↔ (n = 149) :=
by
  intro n
  split
  · intro h
    rw [← h, nat.add_assoc]
    apply nat.add_right_cancel
    rw [nat.add_comm 1 74, nat.add_assoc]
    apply nat.add_right_cancel
    rw nat.add_comm
  · intro h
    exact h
  sorry

end misha_students_l549_549470


namespace ratio_of_weight_l549_549430

theorem ratio_of_weight (B : ℝ) : 
    (2 * (4 + B) = 16) → ((B = 4) ∧ (4 + B) / 2 = 4) := by
  intro h
  have h₁ : B = 4 := by
    linarith
  have h₂ : (4 + B) / 2 = 4 := by
    rw [h₁]
    norm_num
  exact ⟨h₁, h₂⟩

end ratio_of_weight_l549_549430


namespace problem_statement_l549_549697

theorem problem_statement (a : ℝ) 
  (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 :=
by
  sorry

end problem_statement_l549_549697


namespace chord_length_l549_549680

noncomputable def cos_pi_over_3 := real.cos (real.pi / 3)
noncomputable def sin_pi_over_3 := real.sin (real.pi / 3)

def line_parametric_eq (t : ℝ) : ℝ × ℝ :=
  (t * cos_pi_over_3, t * sin_pi_over_3)

def circle_eq (theta : ℝ) : ℝ :=
  4 * real.cos theta

theorem chord_length :
  ∃ (t : ℝ), (∃ (theta : ℝ), circle_eq theta = 4 * real.cos theta)
  → (∃ (l : ℝ), 2 * l = 2) :=
by
  -- Proof is omitted as per the instructions
  sorry

end chord_length_l549_549680


namespace proof_problem_l549_549773

noncomputable def problem (a b c d : ℝ) : Prop :=
(a + b + c = 3) ∧ 
(a + b + d = -1) ∧ 
(a + c + d = 8) ∧ 
(b + c + d = 0) ∧ 
(a * b + c * d = -127 / 9)

theorem proof_problem (a b c d : ℝ) : 
  (a + b + c = 3) → 
  (a + b + d = -1) →
  (a + c + d = 8) → 
  (b + c + d = 0) → 
  (a * b + c * d = -127 / 9) :=
by 
  intro h1 h2 h3 h4
  -- Proof is omitted, "sorry" indicates it is to be filled in
  admit

end proof_problem_l549_549773


namespace number_of_ordered_arrays_eq_6_l549_549627

theorem number_of_ordered_arrays_eq_6 (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|) →
  (∃ p : finset (ℝ × ℝ × ℝ), p.card = 6 ∧ ∀ q ∈ p, q = (0, 0, 1) ∨ q = (0, 1, 0) ∨ q = (1, 0, 0) ∨ q = (0, 0, -1) ∨ q = (0, -1, 0) ∨ q = (-1, 0, 0)) :=
sorry

end number_of_ordered_arrays_eq_6_l549_549627


namespace a_is_2_years_older_than_b_l549_549964

def a_b_age_difference : Prop :=
  ∀ (A B C : ℕ), B = 10 → B = 2 * C → A + B + C = 27 → A - B = 2

theorem a_is_2_years_older_than_b : a_b_age_difference :=
begin
  sorry
end

end a_is_2_years_older_than_b_l549_549964


namespace triangle_min_area_l549_549806

theorem triangle_min_area (A B C D : Type) [EuclideanGeometry] 
    (h1 : ∠B A C = 60) (h2 : B D = a)
    (h3 : ∠A D B = 120) (h4 : ∠A D C = 120) (h5 : ∠B D C = 120) :
    area_of_triangle A B C = (3 * a^2 * sqrt 3) / 4 := 
sorry

end triangle_min_area_l549_549806


namespace complex_quadratic_power_l549_549573

theorem complex_quadratic_power :
  (1 - Complex.i) ^ 4 = -4 :=
by
  sorry

end complex_quadratic_power_l549_549573


namespace GrandmaOlga_grandchildren_l549_549754

theorem GrandmaOlga_grandchildren :
  (∃ d : ℕ, d = 3 ∧ ∀ i : Fin d, 6 ∈ ℕ) ∧
  (∃ s : ℕ, s = 3 ∧ ∀ j : Fin s, 5 ∈ ℕ) →
  18 + 15 = 33 :=
by
  intros h
  cases' h with h_d h_s
  cases' h_d with d_vals num_d
  cases' d_vals with d_eq d_cond
  cases' h_s with s_vals num_s
  cases' s_vals with s_eq s_cond
  sorry

end GrandmaOlga_grandchildren_l549_549754


namespace quadrilateral_dot_product_l549_549050

theorem quadrilateral_dot_product (AB BC CD DA : ℝ)
  (h1 : AB = 1) (h2 : BC = 4) (h3 : CD = 2) (h4 : DA = 3) :
  let AC := 2 * (5/2) in
  let e := 2 * (5/2) / 5 in
  let BD := (2 / e) in
  (AC * BD) = 10 :=
by
  sorry

end quadrilateral_dot_product_l549_549050


namespace scarves_count_l549_549667

theorem scarves_count 
  (S : ℕ) 
  (scarf_wool : S * 3)
  (aaron_sweaters_wool : 5 * 4 = 20)
  (enid_sweaters_wool : 8 * 4 = 32)
  (total_wool : scarf_wool + 20 + 32 = 82) : 
  S = 10 :=
by 
  have total_scarf_wool := S * 3,
  have total_sweaters_wool := 20 + 32,
  have eq := total_scarf_wool + total_sweaters_wool = 82,
  sorry

end scarves_count_l549_549667


namespace general_term_is_correct_l549_549047

variable (a : ℕ → ℤ)
variable (n : ℕ)

def is_arithmetic_sequence := ∃ d a₁, ∀ n, a n = a₁ + d * (n - 1)

axiom a_10_eq_30 : a 10 = 30
axiom a_20_eq_50 : a 20 = 50

noncomputable def general_term (n : ℕ) : ℤ := 2 * n + 10

theorem general_term_is_correct (a: ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 10 = 30)
  (h3 : a 20 = 50)
  : ∀ n, a n = general_term n :=
sorry

end general_term_is_correct_l549_549047


namespace max_distance_from_origin_l549_549257

theorem max_distance_from_origin (θ : ℝ) : 
  let x := sqrt(3) + cos θ,
      y := 1 + sin θ,
      distance := sqrt((x)^2 + (y)^2)
  in distance ≤ 3 :=
by
  let x := sqrt(3) + cos θ
  let y := 1 + sin θ
  let distance := sqrt(x^2 + y^2)
  have h : distance = sqrt(5 + 4 * sin(θ + π / 3)) := sorry
  have h_max : 4 * sin(θ + π / 3) ≤ 4 := sorry
  have h_sqrt : sqrt(9) = 3 := sorry
  show distance ≤ 3, from
    calc distance
        ≤ sqrt(9) : by { rw [h], apply sqrt_le_sqrt, linarith }
        = 3 : h_sqrt

end max_distance_from_origin_l549_549257


namespace semicircle_rotated_volume_l549_549616

-- Problem Definition
def volume_of_solid_semicircle_rotation (R : ℝ) : ℝ :=
  (π / 3) * (3 * π - 4) * R^3

-- Theorem Statement
theorem semicircle_rotated_volume (R : ℝ) (h : R > 0) : 
  volume_of_solid_semicircle_rotation R = (π / 3) * (3 * π - 4) * R^3 :=
by
  sorry

end semicircle_rotated_volume_l549_549616


namespace second_train_speed_l549_549934

variable (t v : ℝ)

-- Defining the first condition: 20t = vt + 55
def condition1 : Prop := 20 * t = v * t + 55

-- Defining the second condition: 20t + vt = 495
def condition2 : Prop := 20 * t + v * t = 495

-- Prove that the speed of the second train is 16 km/hr under given conditions
theorem second_train_speed : ∃ t : ℝ, condition1 t 16 ∧ condition2 t 16 := sorry

end second_train_speed_l549_549934


namespace smallest_N_for_2002_terms_l549_549657

theorem smallest_N_for_2002_terms : ∃ N : ℕ, (binomial N 5 = 2002) ∧ (∀ M < N, binomial M 5 ≠ 2002) :=
by
  sorry

end smallest_N_for_2002_terms_l549_549657


namespace computer_room_arrangements_l549_549233

theorem computer_room_arrangements (n k : ℕ) (h1 : n = 6) (h2 : ∀ k, 2 ≤ k → k ≤ n) :
  (∑ k in finset.range (n+1), nat.choose n k) - 7 = 
  (nat.choose n 2 + nat.choose n 3 + nat.choose n 4 + nat.choose n 5 + nat.choose n 6) :=
by sorry

end computer_room_arrangements_l549_549233


namespace num_valid_m_divisors_of_1750_l549_549330

theorem num_valid_m_divisors_of_1750 : 
  ∃! (m : ℕ) (h1 : m > 0), ∃ (k : ℕ), k > 0 ∧ 1750 = k * (m^2 - 4) :=
sorry

end num_valid_m_divisors_of_1750_l549_549330


namespace tunnel_length_l549_549242

theorem tunnel_length :
  ∀ (l t1 t2 x y : ℕ),
    l = 300 →
    t1 = 60 →
    t2 = 30 →
    300 + x = 60 * y →
    x - 300 = 30 * y →
    x = 900 :=
by
  intros l t1 t2 x y hl ht1 ht2 h1 h2
  rw [hl, ht1, ht2] at h1 h2
  sorry

end tunnel_length_l549_549242


namespace area_circular_ring_eq_l549_549360

noncomputable def area_of_circular_ring (r : ℝ) : ℝ :=
  π * r ^ 2 - π * (r / 2) ^ 2

theorem area_circular_ring_eq (r : ℝ) :
  area_of_circular_ring r = (3 / 4) * π * r^2 :=
by sorry

end area_circular_ring_eq_l549_549360


namespace hyperbola_eccentricity_l549_549728

theorem hyperbola_eccentricity (a c e : ℝ) 
  (h₁ : a = 2) 
  (h₂ : c - a = 6) : 
  e = c / a :=
by 
  have h₃ : a = 2 := h₁,
  have h₄ : c - a = 6 := h₂,
  sorry

end hyperbola_eccentricity_l549_549728


namespace min_pq_sum_l549_549400

theorem min_pq_sum (p q : ℕ) (h1 : p > 0) (h2 : q > 0) 
  (h3 : (p : ℚ) / q = 0.198) : p + q = 121 :=
sorry

end min_pq_sum_l549_549400


namespace school_seat_payment_l549_549611

def seat_cost (num_rows : ℕ) (seats_per_row : ℕ) (cost_per_seat : ℕ) (discount : ℕ → ℕ → ℕ) : ℕ :=
  let total_seats := num_rows * seats_per_row
  let total_cost := total_seats * cost_per_seat
  let groups_of_ten := total_seats / 10
  let total_discount := groups_of_ten * discount 10 cost_per_seat
  total_cost - total_discount

-- Define the discount function as 10% of the cost of a group of 10 seats
def discount (group_size : ℕ) (cost_per_seat : ℕ) : ℕ := (group_size * cost_per_seat) / 10

theorem school_seat_payment :
  seat_cost 5 8 30 discount = 1080 :=
sorry

end school_seat_payment_l549_549611


namespace most_reasonable_sampling_method_l549_549268

-- Conditions: There are significant differences in the academic burden among students across the three grades of junior high and three grades of high school.
-- No significant differences between male and female students.
def differences_among_grades : Prop := 
  ∃ (G1 G2 G3 : Type), (G1 ≠ G2) ∧ (G2 ≠ G3) ∧ (G1 ≠ G3)

def no_differences_between_genders : Prop :=
  ∀ (M : Type) (F : Type), M = F

-- Conclusion: The most reasonable sampling method is stratified sampling by grade.
def most_reasonable_sampling_method_given_conditions : Prop :=
  differences_among_grades ∧ no_differences_between_genders → 
  (∀ method : Type, method = "Stratified sampling by grade")

theorem most_reasonable_sampling_method : 
  differences_among_grades ∧ no_differences_between_genders → 
  (∀ method : Type, method = "Stratified sampling by grade"):=
  by 
  sorry

end most_reasonable_sampling_method_l549_549268


namespace solve_inequality_part1_solve_inequality_part2_l549_549372

-- Define the first part of the problem
theorem solve_inequality_part1 (a : ℝ) (x : ℝ) :
  (x^2 - a * x - 2 * a^2 < 0) ↔ 
    (a = 0 ∧ false) ∨ 
    (a > 0 ∧ -a < x ∧ x < 2 * a) ∨ 
    (a < 0 ∧ 2 * a < x ∧ x < -a) := 
sorry

-- Define the second part of the problem
theorem solve_inequality_part2 (a b : ℝ) (x : ℝ) 
  (h : { x | x^2 - a * x - b < 0 } = { x | -1 < x ∧ x < 2 }) :
  { x | a * x^2 + x - b > 0 } = { x | x < -2 } ∪ { x | 1 < x } :=
sorry

end solve_inequality_part1_solve_inequality_part2_l549_549372


namespace circumcenter_distance_two_l549_549057

noncomputable def distance_between_circumcenter (A B C M : ℝ × ℝ)
  (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25)
  (hBC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 17)
  (hAC : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 16)
  (hM_on_AC : M.1 = C.1 - 1 ∧ M.2 = C.2)
  (hCM : (M.1 - C.1)^2 + (M.2 - C.2)^2 = 1)
  : ℝ :=
dist ( ( (A.1 + B.1) / 2, (A.2 + B.2) / 2 ) ) ( ( (B.1 + C.1) / 2, (B.2 + C.2) / 2 )) 

theorem circumcenter_distance_two (A B C M : ℝ × ℝ)
  (hAB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25)
  (hBC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 17)
  (hAC : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 16)
  (hM_on_AC : M.1 = C.1 - 1 ∧ M.2 = C.2)
  (hCM : (M.1 - C.1)^2 + (M.2 - C.2)^2 = 1) 
  : distance_between_circumcenter A B C M hAB hBC hAC hM_on_AC hCM = 2 :=
sorry

end circumcenter_distance_two_l549_549057


namespace cylinder_volume_expansion_l549_549900

noncomputable def initial_volume (r h : ℝ) : ℝ :=
  π * r^2 * h

noncomputable def expanded_volume (r h : ℝ) : ℝ :=
  π * (2 * r)^2 * (2 * h)

theorem cylinder_volume_expansion (r h : ℝ) : 
  expanded_volume r h = 8 * initial_volume r h :=
by
  sorry

end cylinder_volume_expansion_l549_549900


namespace regular_polygon_sides_l549_549610

theorem regular_polygon_sides (exterior_angle : ℝ) (h : exterior_angle = 15) : 
  ∃ n : ℕ, 360 / n = exterior_angle ∧ n = 24 :=
by
  use 24
  have hn : 24 > 0 := by norm_num
  field_simp [hn]
  rw [← h]
  norm_num
  exact ⟨rfl, rfl⟩

end regular_polygon_sides_l549_549610


namespace sum_after_transformation_l549_549921

variable (S : ℝ) (x y : ℝ)
-- Suppose x and y are the original numbers and their sum is S.
hypothesis h_sum : x + y = S

-- Define the final sum of the two numbers after the transformations.
def final_sum : ℝ := 3 * (x + 5) + 3 * (y + 5)

theorem sum_after_transformation : final_sum S x y = 3 * S + 30 :=
by
  rw [final_sum, h_sum]
  sorry

end sum_after_transformation_l549_549921


namespace complex_power_proof_l549_549585

noncomputable def complex_power : Prop :=
  (1 - complex.i)^4 = -4

theorem complex_power_proof : complex_power :=
by
  sorry

end complex_power_proof_l549_549585


namespace apples_total_l549_549693

-- Definitions as per conditions
def apples_on_tree : Nat := 5
def initial_apples_on_ground : Nat := 8
def apples_eaten_by_dog : Nat := 3

-- Calculate apples left on the ground
def apples_left_on_ground : Nat := initial_apples_on_ground - apples_eaten_by_dog

-- Calculate total apples left
def total_apples_left : Nat := apples_on_tree + apples_left_on_ground

theorem apples_total : total_apples_left = 10 := by
  -- the proof will go here
  sorry

end apples_total_l549_549693


namespace quadrilateral_inscribed_area_ratio_l549_549859

theorem quadrilateral_inscribed_area_ratio (r : ℝ) (h_pos : 0 < r) :
  let A := r * (Real.sqrt 3 / 2),
      B := r,
      C := r * (2 - (Real.sqrt 3 / 2)),
      D := r * (2 * (Real.sin 15)),
      area_quadrilateral := (1 / 2 : ℝ) * A * B + (1 / 2 : ℝ) * C * D,
      area_circle := Real.pi * r^2,
      ratio := area_quadrilateral / area_circle
  in ratio = (3 + Real.sqrt 3) / (4 * Real.pi) →
     3 + 3 + 4 = 10 :=
by
  intros
  sorry -- Proof to be filled in

end quadrilateral_inscribed_area_ratio_l549_549859


namespace range_of_a_l549_549704

theorem range_of_a (a : ℝ) : (∀ x : ℝ, abs (2 * x + 2) - abs (2 * x - 2) ≤ a) ↔ 4 ≤ a :=
sorry

end range_of_a_l549_549704


namespace tendsto_zero_l549_549518

open Filter Real

variable {α : Type*} (a : ℕ → α) [LinearOrder α] (f : α → ℕ)

def strictly_monotonic (a : ℕ → α) : Prop :=
  ∀ n m, n < m → a n < a m

def sum_inverses_converges (a : ℕ → ℝ) : Prop :=
  Summable (λ n, (a n)⁻¹)

def largest_below (a : ℕ → α) (x : α) : ℕ :=
  Sup { i : ℕ | a i < x }

noncomputable def f (a : ℕ → ℝ) (x : ℝ) : ℕ :=
  largest_below a x

theorem tendsto_zero (a : ℕ → ℝ) (h1 : strictly_monotonic a) (h2 : sum_inverses_converges a) : 
  Tendsto (λ x : ℝ, (f a x) / x) atTop (nhds 0) :=
begin
  sorry
end

end tendsto_zero_l549_549518


namespace bookstore_price_change_l549_549602

theorem bookstore_price_change (P : ℝ) (x : ℝ) (h : P > 0) : 
  (P * (1 + x / 100) * (1 - x / 100)) = 0.75 * P → x = 50 :=
by
  sorry

end bookstore_price_change_l549_549602


namespace travis_revenue_l549_549183

-- Declare nonnegative integers for apples, apples per box, and price per box
variables (apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ)

-- Specify the conditions
def conditions := apples = 10000 ∧ apples_per_box = 50 ∧ price_per_box = 35

-- State the theorem to be proved
theorem travis_revenue (h : conditions) : (apples / apples_per_box) * price_per_box = 7000 :=
by
  cases h with 
  | intro h1 h2 h3 =>
  rw [h1, h2, h3]
  sorry -- Proof is not required as per the instructions

end travis_revenue_l549_549183


namespace number_of_divisors_of_N_l549_549018

def N : ℕ := 2^5 * 3^1 * 5^3 * 7^2

theorem number_of_divisors_of_N : (finset.range (5 + 1)).card * (finset.range (1 + 1)).card * (finset.range (3 + 1)).card * (finset.range (2 + 1)).card = 144 :=
by
  sorry

end number_of_divisors_of_N_l549_549018


namespace perfect_squares_count_200_600_l549_549761

-- Define the range and the set of perfect squares
def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k * k

def perfect_squares_between (a b : ℕ) : Finset ℕ :=
  Finset.filter (λ n, is_perfect_square n) (Finset.Icc a b)

-- The main theorem
theorem perfect_squares_count_200_600 : 
  perfect_squares_between 200 600).card = 10 :=
by
  sorry

end perfect_squares_count_200_600_l549_549761


namespace cos_angle_BAD_l549_549052

theorem cos_angle_BAD (AB AC BC : ℝ) (hAB : AB = 4) (hAC : AC = 7) (hBC : BC = 9) (D : ℝ)
  (hD : D ∈ set.Icc 0 BC) : ∃ A : ℝ, ∃ BAD : ℝ, is_cos_div2_ABAC (AB, AC, BC, A, BAD) :=
begin
  sorry
end

end cos_angle_BAD_l549_549052


namespace find_number_l549_549120

theorem find_number : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 :=
by {
  sorry
}

end find_number_l549_549120


namespace angle_equality_l549_549822

-- Define the conditions
variables (ABC : Triangle) (O : Point) (H : Point)
variable (circumcenter_O : is_circumcenter O ABC)
variable (altitude_H : is_foot_of_altitude H A ABC)

-- Define the angles involved
variables (BAO CAH : Angle)

-- State the problem
theorem angle_equality (h₁ : has_angle ABC BAO)
                       (h₂ : has_angle ABC CAH)
                       (hO : is_circumcenter O ABC)
                       (hH : is_foot_of_altitude H A ABC) :
  BAO = CAH := sorry

end angle_equality_l549_549822


namespace parabola_intersection_points_l549_549933

theorem parabola_intersection_points :
  let parabola1 := λ x : ℝ => 4*x^2 + 3*x - 1
  let parabola2 := λ x : ℝ => x^2 + 8*x + 7
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ = -4/3 ∧ y₁ = -17/9 ∧
                        x₂ = 2 ∧ y₂ = 27 ∧
                        parabola1 x₁ = y₁ ∧ 
                        parabola2 x₁ = y₁ ∧
                        parabola1 x₂ = y₂ ∧
                        parabola2 x₂ = y₂ :=
by {
  sorry
}

end parabola_intersection_points_l549_549933


namespace number_is_seven_point_five_l549_549121

theorem number_is_seven_point_five (x : ℝ) (h : x^2 + 100 = (x - 20)^2) : x = 7.5 :=
by
  sorry

end number_is_seven_point_five_l549_549121


namespace parabola_tangent_y_intercept_correct_l549_549437

noncomputable def parabola_tangent_y_intercept (a : ℝ) : Prop :=
  let C := fun x : ℝ => x^2
  let slope := 2 * a
  let tangent_line := fun x : ℝ => slope * (x - a) + C a
  let Q := (0, tangent_line 0)
  Q = (0, -a^2)

-- Statement of the problem as a Lean theorem
theorem parabola_tangent_y_intercept_correct (a : ℝ) (h : a > 0) :
  parabola_tangent_y_intercept a := 
by 
  sorry

end parabola_tangent_y_intercept_correct_l549_549437


namespace complex_power_proof_l549_549586

noncomputable def complex_power : Prop :=
  (1 - complex.i)^4 = -4

theorem complex_power_proof : complex_power :=
by
  sorry

end complex_power_proof_l549_549586


namespace max_digits_in_product_l549_549544

theorem max_digits_in_product :
  let n := (99999 : Nat)
  let m := (999 : Nat)
  let product := n * m
  ∃ d : Nat, product < 10^d ∧ 10^(d-1) ≤ product :=
by
  sorry

end max_digits_in_product_l549_549544


namespace collinear_points_l549_549659

theorem collinear_points (k : ℝ) :
  let p1 := (3, 1)
  let p2 := (6, 4)
  let p3 := (10, k + 9)
  let slope (a b : ℝ × ℝ) : ℝ := (b.snd - a.snd) / (b.fst - a.fst)
  slope p1 p2 = slope p1 p3 → k = -1 :=
by 
  let p1 := (3, 1)
  let p2 := (6, 4)
  let p3 := (10, k + 9)
  let slope (a b : ℝ × ℝ) : ℝ := (b.snd - a.snd) / (b.fst - a.fst)
  sorry

end collinear_points_l549_549659


namespace power_of_one_minus_i_eq_neg_4_l549_549575

noncomputable def i : ℂ := complex.I

theorem power_of_one_minus_i_eq_neg_4 : (1 - i)^4 = -4 :=
by
  sorry

end power_of_one_minus_i_eq_neg_4_l549_549575


namespace grandma_Olga_grandchildren_l549_549759

def daughters : Nat := 3
def sons : Nat := 3
def sons_per_daughter : Nat := 6
def daughters_per_son : Nat := 5

theorem grandma_Olga_grandchildren : 
  (daughters * sons_per_daughter) + (sons * daughters_per_son) = 33 := by
  sorry

end grandma_Olga_grandchildren_l549_549759


namespace count_perfect_squares_between_200_and_600_l549_549764

-- Definition to express the condition of a perfect square within a specific range
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

-- Definition to express the count of perfect squares between 200 and 600
def perfect_squares_between (a b : ℕ) : ℕ :=
  (nat.floor (real.sqrt b) - nat.ceil (real.sqrt a)) + 1

theorem count_perfect_squares_between_200_and_600 :
  perfect_squares_between 200 600 = 10 :=
by
  sorry

end count_perfect_squares_between_200_and_600_l549_549764


namespace shift_parabola_left_l549_549502

theorem shift_parabola_left (x : ℝ) : (x + 1)^2 = y ↔ x^2 = y :=
sorry

end shift_parabola_left_l549_549502


namespace find_boundary_intersections_l549_549141

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Path :=
(start : Point3D)
(middle : Point3D)
(end : Point3D)

variables (A B C D E A1 B1 C1 D1 E1 L K M N : Point3D)

-- The prism's base vertices
variables (base_vertices : ℕ → Point3D) -- where base_vertices 0 = A, base_vertices 1 = B, base_vertices 2 = C, base_vertices 3 = D, base_vertices 4 = E
-- The opposite face vertices
variables (opposite_vertices : ℕ → Point3D) -- where opposite_vertices 0 = A1, opposite_vertices 1 = B1, opposite_vertices 2 = C1, opposite_vertices 3 = D1, opposite_vertices 4 = E1

-- Defined paths
def path1 := Path.mk C1 A D1
def path2 := Path.mk D B1 E

-- Define the common points
def common_point1 : Point3D := L
def common_point2 : Point3D := K

-- Prove the intersection with boundary surface
theorem find_boundary_intersections :
    ∃ (M N : Point3D), intersects_boundary(common_point1, common_point2, M, N) :=
sorry


end find_boundary_intersections_l549_549141


namespace smallest_palindrome_divisible_by_3_l549_549947

def is_palindrome (n : ℕ) : Prop :=
  let str := n.toDigits 10
  str = str.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  n.toDigits 10 |>.sum

theorem smallest_palindrome_divisible_by_3 : ∃ n, is_palindrome n ∧ n ≥ 1000 ∧ n < 10000 ∧ n % 3 = 0 ∧ (∀ m, is_palindrome m ∧ m ≥ 1000 ∧ m < 10000 ∧ m % 3 = 0 → n ≤ m) ∧ n = 1221 :=
  sorry

end smallest_palindrome_divisible_by_3_l549_549947


namespace percentage_of_supporters_is_71_percent_l549_549790

theorem percentage_of_supporters_is_71_percent :
  (let men := 150 in let women := 650 in
  let men_support := 0.75 * men in let women_support := 0.70 * women in
  let total_supporters := men_support + women_support in
  let total_surveyed := men + women in
  (total_supporters / total_surveyed) * 100 = 71) :=
by
  let men := 150 
  let women := 650 
  have men_support : ℝ := 0.75 * men
  have women_support : ℝ := 0.70 * women
  have total_supporters : ℝ := men_support + women_support
  have total_surveyed : ℝ := men + women
  have result : (total_supporters / total_surveyed) * 100 = 71 := sorry
  exact result

end percentage_of_supporters_is_71_percent_l549_549790


namespace sum_quadratic_polynomials_has_root_l549_549093

-- Definitions of quadratic polynomials with positive leading coefficients
variable {R : Type*} [LinearOrderedField R]

structure QuadraticPols (R : Type*) [LinearOrderedField R] :=
(dx squared x c : R)
(dx_pos : dx > 0)

def is_root (p : QuadraticPols R) (r : R) : Prop :=
  p.dx * r^2 + p.squared * r + p.c = 0

def has_common_root (p q : QuadraticPols R) : Prop :=
  ∃ x : R, is_root p x ∧ is_root q x

-- The statement of the theorem
theorem sum_quadratic_polynomials_has_root 
  (p1 p2 p3 : QuadraticPols R)
  (h1 : has_common_root p1 p2)
  (h2 : has_common_root p2 p3)
  (h3 : has_common_root p3 p1) :
  ∃ x : R, is_root (QuadraticPols.mk
    (p1.dx + p2.dx + p3.dx)
    (p1.squared + p2.squared + p3.squared)
    (p1.c + p2.c + p3.c) 
    (by linarith [p1.dx_pos, p2.dx_pos, p3.dx_pos])) x :=
sorry

end sum_quadratic_polynomials_has_root_l549_549093


namespace area_of_triangle_l549_549805

theorem area_of_triangle (AB AC : ℝ) (angle_A : ℝ) (hAB : AB = 3) (hAC : AC = 2) (hangle_A : angle_A = 60) :
  let S := 1/2 * AB * AC * (Real.sin (Real.pi / 3)) in
  S = 3 / 2 * Real.sqrt 3 :=
by sorry

end area_of_triangle_l549_549805


namespace percentage_decrease_l549_549152

variable (current_price original_price : ℝ)

theorem percentage_decrease (h1 : current_price = 760) (h2 : original_price = 1000) :
  (original_price - current_price) / original_price * 100 = 24 :=
by
  sorry

end percentage_decrease_l549_549152


namespace squares_in_H_l549_549519

def H : set (ℤ × ℤ) := { p | -8 ≤ p.1 ∧ p.1 ≤ 8 ∧ -8 ≤ p.2 ∧ p.2 ≤ 8 }

def side_lengths := { s | 8 ≤ s ∧ s ≤ 16 }

noncomputable def count_squares (s : ℕ) : ℕ :=
  let range := 17 - s in
  range * range

def total_squares (sides : set ℕ) : ℕ :=
  sides.sum count_squares

theorem squares_in_H : total_squares side_lengths = 285 :=
by {
  sorry -- the proof will go here
}

end squares_in_H_l549_549519


namespace volume_of_inscribed_cube_l549_549237

theorem volume_of_inscribed_cube (a : ℝ) (ha : a = 12) : ∃ V : ℝ, V = 192 * Real.sqrt 3 :=
by
  let s := 4 * Real.sqrt 3
  have h1 : s = 4 * Real.sqrt 3 := rfl
  have h2 : (a / Real.sqrt 3) = 4 * Real.sqrt 3 := by 
    rw [ha]
    norm_num
  use s ^ 3
  calc s ^ 3 = (4 * Real.sqrt 3) ^ 3 : by rw [h1]
             ... = 192 * Real.sqrt 3 : by norm_num
  sorry

end volume_of_inscribed_cube_l549_549237


namespace irrational_count_correct_l549_549863

def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

def numbers : list ℝ := [4 * Real.pi, 0, Real.sqrt 7, Real.sqrt 16 / 2, 0.1, 
                         (λ n, if n = 0 then 0.2 else 0.212212221 * 10^(-n)) (Nat.succ 0)]

def irrational_count := list.countp is_irrational numbers

theorem irrational_count_correct : irrational_count = 3 := 
sorry

end irrational_count_correct_l549_549863


namespace sin_ratio_triangle_area_l549_549404

namespace TriangleProblems

-- Define the problem conditions
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Sides

-- Assuming the given conditions
axiom cos_condition : (cos A - 2 * cos C) / cos B = (2 * c - a) / b
axiom angle_sum : A + B + C = π

-- Problem 1: Prove the ratio of sin C to sin A
theorem sin_ratio (h : sin C = 2 * sin A) : (sin C) / (sin A) = 2 :=
by {
  rw h,
  exact div_self (ne_of_gt (show sin A > 0, from sorry)) -- We assume sin(A) is nonzero
}

-- Problem 2: Compute the area of the triangle given additional conditions
noncomputable def area (cos_B : cos B = 1 / 4) (side_b : b = 2) : ℝ :=
1 / 2 * a * c * sqrt (1 - (cos B) ^ 2)

-- Theorem to state the area given the previous conditions
theorem triangle_area (h : sin C = 2 * sin A) (cos_B : cos B = 1 / 4) (side_b : b = 2) (a_value : a = 1) (c_value : c = 2) : area cos_B side_b = sqrt 15 / 4 :=
sorry -- Proof can be completed later

end TriangleProblems

end sin_ratio_triangle_area_l549_549404


namespace all_blue_figures_are_small_l549_549527

variables (Shape : Type) (Large Blue Small Square Triangle : Shape → Prop)

-- Given conditions
axiom h1 : ∀ (x : Shape), Large x → Square x
axiom h2 : ∀ (x : Shape), Blue x → Triangle x

-- The goal to prove
theorem all_blue_figures_are_small : ∀ (x : Shape), Blue x → Small x :=
by
  sorry

end all_blue_figures_are_small_l549_549527


namespace eq_expr_l549_549737

theorem eq_expr (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a⁻² * b⁻²) / (a⁻² + b⁻²) = (a² * b²) / (a² + b²) :=
by
  sorry

end eq_expr_l549_549737


namespace sqrt_of_sixteen_l549_549916

theorem sqrt_of_sixteen (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := 
sorry

end sqrt_of_sixteen_l549_549916


namespace geometric_sequence_terms_l549_549343

theorem geometric_sequence_terms 
    (r : ℝ) -- Common ratio of the sequence
    (n : ℕ) -- Half of the total number of terms
    (first_term : r ^ 0 = 1) -- First term is 1
    (sum_even_eq_twice_sum_odd : (sum (i in range n, r^(2*i+1))) = 2 * (sum (i in range n, r^(2*i))))
    (sum_middle_eq_24 : r^(n-1) + r^n = 24) : 
    2 * n = 8 := 
by 
  sorry

end geometric_sequence_terms_l549_549343


namespace value_of_expression_l549_549566

-- Definition of the constants involved in the problem
def a := 4.7
def b := 13.26
def c := 9.43
def d := 77.31
def result := 470

-- Statement of the problem
theorem value_of_expression : a * b + a * c + a * d = result :=
by sorry

end value_of_expression_l549_549566


namespace power_of_one_minus_i_eq_neg_4_l549_549576

noncomputable def i : ℂ := complex.I

theorem power_of_one_minus_i_eq_neg_4 : (1 - i)^4 = -4 :=
by
  sorry

end power_of_one_minus_i_eq_neg_4_l549_549576


namespace sum_of_digits_base2_365_l549_549547

theorem sum_of_digits_base2_365 :
  let binary_365 := 256 + 64 + 32 + 8 + 4 + 1 in
  let sum_of_digits := (1 + 0 + 1 + 1 + 0 + 1 + 1 + 0 + 1) in
  sum_of_digits = 6 := by
  sorry

end sum_of_digits_base2_365_l549_549547


namespace problem_expected_students_scoring_above_130_l549_549028

noncomputable def expected_students_scoring_above_130 (μ : ℝ) (σ : ℝ) (num_students : ℕ) : ℝ :=
  let z := (130 - μ) / σ in
  let prob_above_130 := 1 - (real.norm_cdf z) in
  prob_above_130 * num_students

theorem problem_expected_students_scoring_above_130 :
  expected_students_scoring_above_130 120 10 40 ≈ 6 := sorry

end problem_expected_students_scoring_above_130_l549_549028


namespace gcd_lcm_product_24_60_l549_549301

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l549_549301


namespace product_gcd_lcm_24_60_l549_549323

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l549_549323


namespace hyperbola_focal_length_l549_549895

def hyperbola_equation (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

def a_squared : ℝ := 3
def b_squared : ℝ := 1
def c : ℝ := real.sqrt (a_squared + b_squared)

theorem hyperbola_focal_length : ∀ x y : ℝ, hyperbola_equation x y → 2 * c = 4 :=
by
  intros x y h
  sorry

end hyperbola_focal_length_l549_549895


namespace marge_final_plants_l549_549463

def initial_seeds : ℕ := 23
def marigold_seeds : ℕ := 10
def sunflower_seeds : ℕ := 13
def seeds_never_grew : ℕ := 5
def marigold_growth_rate : ℕ := 40
def sunflower_growth_rate : ℕ := 60
def wilt_percentage : ℕ := 25
def animal_eat_percentage : ℕ := 50
def pest_control_success_rate : ℕ := 75
def weed_strangle_percentage : ℕ := 33
def final_weed_kept : ℕ := 1

theorem marge_final_plants : ℕ :=
  let seeds_grew := initial_seeds - seeds_never_grew
  let marigold_grew := (marigold_seeds * marigold_growth_rate) / 100
  let sunflower_grew := (sunflower_seeds * sunflower_growth_rate) / 100 - 1
  let remaining_sunflower := sunflower_grew - sunflower_grew * wilt_percentage / 100
  let remaining_marigold := marigold_grew - marigold_grew / 2
  let total_plants := remaining_marigold + remaining_sunflower
  let remaining_plants := total_plants - total_plants * weed_strangle_percentage / 100 + final_weed_kept
  let protected_plants := remaining_plants * pest_control_success_rate / 100 :=
  protected_plants := 6

#print marge_final_plants

end marge_final_plants_l549_549463


namespace incorrect_statement_about_zero_l549_549199

-- Definitions based on conditions in a)
def is_natural_number (n : ℕ) : Prop := n = 0 -- Definition interprets 0 as a natural number
def is_opposite_itself (a : ℤ) : Prop := a = 0 ∨ (-a = a) -- a number whose opposite is itself
def square_root_zero (x : ℝ) : Prop := x = 0 → sqrt x = 0 -- square root of 0 is 0
def is_rational_number (x : ℚ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ x = p / q -- definition of a rational number

theorem incorrect_statement_about_zero :
  ¬(is_rational_number (0 : ℚ) ∨ ¬is_rational_number (0 : ℚ)) := by
  sorry -- the actual proof goes here

end incorrect_statement_about_zero_l549_549199


namespace sequence_value_at_one_l549_549154

-- Define the sequence
def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 1 / (1 - a n)

-- Define the initial condition
def initial_condition (a : ℕ → ℝ) : Prop :=
  a 8 = 2

-- Define the objective: the value of a_1
theorem sequence_value_at_one (a : ℕ → ℝ) 
  (h_seq : sequence a) 
  (h_init : initial_condition a) : 
  a 1 = 1 / 2 := 
sorry

end sequence_value_at_one_l549_549154


namespace problem_proof_l549_549235

-- Definitions
def a_seq : ℕ → ℕ 
| 1 := 3
| (n + 1) := 2 * a_seq n + 2

def b_seq (n : ℕ) : ℝ := n / (a_seq n + 2)

def S_seq (n : ℕ) : ℝ := (Finset.range n).sum (λ i, b_seq (i + 1))

-- Theorem
theorem problem_proof :
  a_seq 2 = 8 ∧
  a_seq 3 = 18 ∧
  (∃ r : ℕ, ∀ n : ℕ, a_seq n + 2 = 5 * 2^(n - 1)) ∧
  (∀ n : ℕ, n > 0 → 1/5 ≤ S_seq n ∧ S_seq n < 4/5) :=
by
  sorry

end problem_proof_l549_549235


namespace find_least_N_and_digit_sum_l549_549994

def probability_condition (N : ℕ) : ℚ := 
  2 * (N / 3).floor + 1) / (N + 2)

def satisfies_condition (N : ℕ) : Prop := 
  probability_condition N < 7 / 10

def digit_sum (x : ℕ) : ℕ :=
  x.digits.sum

theorem find_least_N_and_digit_sum:
  (∃ N : ℕ, N % 7 = 0 ∧ satisfies_condition N ∧ digit_sum N = 7) := 
sorry

end find_least_N_and_digit_sum_l549_549994


namespace integral_of_neg_f_eq_l549_549399

noncomputable def f (x : ℝ) : ℝ := x^2 + x
noncomputable def f_neg (x : ℝ) : ℝ := f (-x)

theorem integral_of_neg_f_eq :
  (∫ x in -1..3, f_neg x) = 14/3 :=
by
  sorry

end integral_of_neg_f_eq_l549_549399


namespace collinear_points_l549_549380

theorem collinear_points (a b : ℝ) :
  let A := (1, -1, b)
      B := (2, a, 1)
      O := (0, 0, 0) in
  collinear O A B → a = -2 ∧ b = 1 / 2 :=
begin
  intros h,
  sorry
end

end collinear_points_l549_549380


namespace find_line_equation_l549_549143
open Real

def line_through_point_with_equal_intercepts (m c : ℝ) : Prop :=
  let abs_intercept_condition := (c ≠ 0) ∧ c * c = (c * c) / (m * m) in
  (3 = 2 * m + c) ∧ abs_intercept_condition

theorem find_line_equation :
  ∃ (m c : ℝ), m = 1 ∧ c = 1 ∧ line_through_point_with_equal_intercepts m c := 
by 
  use (1 : ℝ)
  use (1 : ℝ)
  sorry

end find_line_equation_l549_549143


namespace equal_and_perpendicular_segments_l549_549040

-- Definitions of the points and basic assumptions in the scenario
structure Triangle :=
  (A B C : Point)
  (midpoint : Point → Point → Point)
  (square_center : Point → Point → Point)

variables {T : Triangle}

-- Proving that the segments B1D and C1D are equal in length and perpendicular
theorem equal_and_perpendicular_segments 
  (B1 C1 D : Point)
  (hB1 : B1 = T.square_center T.A T.B)
  (hC1 : C1 = T.square_center T.A T.C)
  (hD : D = T.midpoint T.B T.C) :
  (dist B1 D = dist C1 D) ∧ (angle B1 D C1 = 90) :=
sorry

end equal_and_perpendicular_segments_l549_549040


namespace complex_quadratic_power_l549_549572

theorem complex_quadratic_power :
  (1 - Complex.i) ^ 4 = -4 :=
by
  sorry

end complex_quadratic_power_l549_549572


namespace relationship_among_a_b_c_l549_549830

noncomputable def a : ℝ := 3^0.2
noncomputable def b : ℝ := (1/3)^(-1.1)
noncomputable def c : ℝ := Real.logBase 3 2

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l549_549830


namespace arithmetic_sequence_sum_l549_549375

theorem arithmetic_sequence_sum (n : ℕ) (h : n > 0)
  (y : ℕ → ℚ)
  (h1 : y 1 = 2)
  (h2 : ∀ k, 1 ≤ k ∧ k < n → y (k + 1) = y k + 3/4) :
  ∑ i in finset.range n, y (i + 1) = n * (3 * n + 13) / 8 := sorry

end arithmetic_sequence_sum_l549_549375


namespace sum_of_squares_of_digits_of_N_l549_549912

theorem sum_of_squares_of_digits_of_N :
  let colorings_count (N : ℕ) : Prop :=
    -- Number of valid colorings (this assumes combinatorial logic of counting).
    N = 25088
  in
  ∑ d in Nat.digits 10 25088, d^2 = 157 :=
by
  sorry

end sum_of_squares_of_digits_of_N_l549_549912


namespace geometric_sequence_common_ratio_l549_549794

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q) 
  (h_inc : ∀ n, a (n + 1) > a n) (h2 : a 2 = 2) (h3 : a 4 - a 3 = 4) : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l549_549794


namespace closest_point_is_correct_l549_549298

def line_eq (x : ℝ) : ℝ := -3 * x + 5

def closest_point_on_line_to_given_point : Prop :=
  ∃ (x y : ℝ), y = line_eq x ∧ (x, y) = (17 / 10, -1 / 10) ∧
  (∀ (x' y' : ℝ), y' = line_eq x' → (x' - -4)^2 + (y' - -2)^2 ≥ (x - -4)^2 + (y - -2)^2)
  
theorem closest_point_is_correct : closest_point_on_line_to_given_point :=
sorry

end closest_point_is_correct_l549_549298


namespace gcd_lcm_product_24_60_l549_549312

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l549_549312


namespace max_min_f_l549_549903

noncomputable def f (x : ℝ) : ℝ :=
  if 6 ≤ x ∧ x ≤ 8 then
    (Real.sqrt (8 * x - x^2) - Real.sqrt (114 * x - x^2 - 48))
  else
    0

theorem max_min_f :
  ∀ x, 6 ≤ x ∧ x ≤ 8 → f x ≤ 2 * Real.sqrt 3 ∧ 0 ≤ f x :=
by
  intros
  sorry

end max_min_f_l549_549903


namespace fixed_point_and_circle_l549_549902

theorem fixed_point_and_circle {a b : ℝ} (h : a ∥ b) (A : ℝ × ℝ) (hA : A.2 = 0):
  ∃ P : ℝ × ℝ, 
    (P = (0, 2 * b)) ∧
    (∀ (p : ℝ) (γ : set (ℝ × ℝ)), 
      (γ = {x | (x.1)^2 + (x.2)^2 + p*x.1 + (p^2 - 4*b^2)/(4*b)*x.2 = 0}) →
      ∀ T : ℝ × ℝ, T ∈ γ → T.2 = 0 → 
      (∃ t, is_tangent_line_to_circle t γ T ∧ t.slope * (T.1 - (0: ℝ)) = (T.2 - 2 * b))) :=
by
  sorry

end fixed_point_and_circle_l549_549902


namespace infinite_consecutive_pairs_l549_549935

-- Define the relation
def related (x y : ℕ) : Prop :=
  ∀ d ∈ (Nat.digits 10 (x + y)), d = 0 ∨ d = 1

-- Define sets A and B
variable (A B : Set ℕ)

-- Define the conditions
axiom cond1 : ∀ a ∈ A, ∀ b ∈ B, related a b
axiom cond2 : ∀ c, (∀ a ∈ A, related c a) → c ∈ B
axiom cond3 : ∀ c, (∀ b ∈ B, related c b) → c ∈ A

-- Prove that one of the sets contains infinitely many pairs of consecutive numbers
theorem infinite_consecutive_pairs :
  (∃ a ∈ A, ∀ n : ℕ, a + n ∈ A ∧ a + n + 1 ∈ A) ∨ (∃ b ∈ B, ∀ n : ℕ, b + n ∈ B ∧ b + n + 1 ∈ B) :=
sorry

end infinite_consecutive_pairs_l549_549935


namespace number_of_teams_in_conference_l549_549796

theorem number_of_teams_in_conference (n : ℕ) :
  (∃ n, 2 * (n * (n - 1) / 2) = 90) → n = 10 :=
by
  intro hn
  have h := exists.elim hn (λ n hn_eq, hn_eq)
  have h_eq : n * (n - 1) = 90 := by calc
    2 * (n * (n - 1) / 2) = 90         : h  -- Given condition
                   ... = n * (n - 1)   : sorry  -- Simplification
  -- Solving n * (n - 1) = 90
  have h_quad : n^2 - n - 90 = 0 := by
    rw [h_eq, sub_eq_zero]
  sorry

end number_of_teams_in_conference_l549_549796


namespace joan_gave_27_apples_l549_549071

theorem joan_gave_27_apples (total_apples : ℕ) (current_apples : ℕ)
  (h1 : total_apples = 43) 
  (h2 : current_apples = 16) : 
  total_apples - current_apples = 27 := 
by
  sorry

end joan_gave_27_apples_l549_549071


namespace altitude_bisectors_l549_549858

variables {A B C A1 B1 C1 : Type*}
          [nonempty A] [nonempty B] [nonempty C]
          [nonempty A1] [nonempty B1] [nonempty C1]

-- Defining a type for points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Assuming the points form an acute triangle
axiom acute_triangle (A B C : Point) : Prop :=
  ∀ (a b c ∈ {A, B, C}), a ≠ b → ∠ a b c < 90

-- Altitudes of the triangle
def altitude (p1 p2 p3 foot : Point) : Prop :=
  ∠ p1 p2 foot = 90

-- Proving the feet of the altitudes form a triangle with specific properties
theorem altitude_bisectors (A B C A1 B1 C1 : Point)
  (h1 : acute_triangle A B C)
  (h2 : altitude A B C A1) 
  (h3 : altitude B C A B1)
  (h4 : altitude C A B C1) :
  ∃ (ΔA1B1C1 : {A1 B1 C1 : Point}),
    ∀ (p1 p2 p3 ∈ {A1, B1, C1}),
      (altitude A1 B C = bisector p1 p2 p3) :=
sorry

end altitude_bisectors_l549_549858


namespace total_computers_and_televisions_l549_549926

theorem total_computers_and_televisions (c t : ℕ) (hc : c = 32) (ht : t = 66) : c + t = 98 := by
  rw [hc, ht]
  norm_num

end total_computers_and_televisions_l549_549926


namespace gcd_lcm_product_24_60_l549_549300

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l549_549300


namespace cost_per_ream_is_27_l549_549213

-- Let ream_sheets be the number of sheets in one ream.
def ream_sheets : ℕ := 500

-- Let total_sheets be the total number of sheets needed.
def total_sheets : ℕ := 5000

-- Let total_cost be the total cost to buy the total number of sheets.
def total_cost : ℕ := 270

-- We need to prove that the cost per ream (in dollars) is 27.
theorem cost_per_ream_is_27 : (total_cost / (total_sheets / ream_sheets)) = 27 := 
by
  sorry

end cost_per_ream_is_27_l549_549213


namespace greatest_integer_radius_l549_549193

theorem greatest_integer_radius (r : ℝ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
sorry

end greatest_integer_radius_l549_549193


namespace cylinder_volume_increase_l549_549401

theorem cylinder_volume_increase 
  (r h : ℝ) 
  (V : ℝ := π * r^2 * h) 
  (new_h : ℝ := 3 * h) 
  (new_r : ℝ := 2 * r) : 
  (π * new_r^2 * new_h) = 12 * V := 
by
  sorry

end cylinder_volume_increase_l549_549401


namespace range_of_f_l549_549656

def f (x : ℝ) : ℝ := (x + 1) / (x^2 - 2 * x + 2)

theorem range_of_f :
  set.range f = {y : ℝ | (1 - real.sqrt 2) / 2 ≤ y ∧ y ≤ (1 + real.sqrt 2) / 2} :=
sorry

end range_of_f_l549_549656


namespace sum_of_zeros_fg_l549_549009

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^(x - 2) - 1 else x + 2

noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2 * x else 1 / x

noncomputable def fg (x : ℝ) : ℝ := f (g x)

theorem sum_of_zeros_fg : (1 + sqrt 3) + (-1 / 2) = (1 / 2 + sqrt 3) :=
  by
    simp
    sorry  -- proof to be completed

end sum_of_zeros_fg_l549_549009


namespace october_profit_condition_l549_549015

noncomputable def calculate_profit (price_reduction : ℝ) : ℝ :=
  (50 - price_reduction) * (500 + 20 * price_reduction)

theorem october_profit_condition (x : ℝ) (h : calculate_profit x = 28000) : x = 10 ∨ x = 15 := 
by
  sorry

end october_profit_condition_l549_549015


namespace div_by_27_l549_549856

theorem div_by_27 (n : ℕ) : 27 ∣ (10^n + 18 * n - 1) :=
sorry

end div_by_27_l549_549856


namespace tangerines_in_one_box_l549_549209

theorem tangerines_in_one_box (total_tangerines boxes remaining_tangerines tangerines_per_box : ℕ) 
  (h1 : total_tangerines = 29)
  (h2 : boxes = 8)
  (h3 : remaining_tangerines = 5)
  (h4 : total_tangerines - remaining_tangerines = boxes * tangerines_per_box) :
  tangerines_per_box = 3 :=
by 
  sorry

end tangerines_in_one_box_l549_549209


namespace n_plus_d_is_155_l549_549445

noncomputable def n_and_d_sum : Nat :=
sorry

theorem n_plus_d_is_155 (n d : Nat) (hn : 0 < n) (hd : d < 10) 
  (h1 : 4 * n^2 + 2 * n + d = 305) 
  (h2 : 4 * n^3 + 2 * n^2 + d * n + 1 = 577 + 8 * d) : n + d = 155 := 
sorry

end n_plus_d_is_155_l549_549445


namespace find_m_find_m_range_l549_549700

def imaginary_unit (i : ℂ) : Prop :=
  i = complex.I 

def z1 (z1 : ℂ) : Prop :=
  z1 = 1 + complex.I

def z2 (z2 : ℂ) (m : ℝ) : Prop :=
  z2 = ↑m - 2 * complex.I

def sum_z1_z2_eq (z1 z2 : ℂ) : Prop :=
  z1 + z2 = 2 - complex.I

theorem find_m (m : ℝ) (z1 z2 : ℂ) (hi : imaginary_unit complex.I)
  (hz1 : z1 = 1 + complex.I) (hz2 : z2 = ↑m - 2 * complex.I)
  (hsum : sum_z1_z2_eq z1 z2) : m = 1 := 
by sorry

def right_half_plane (z : ℂ) : Prop :=
  z.re > 0

theorem find_m_range (m : ℝ) (z1 z2 : ℂ) (hi : imaginary_unit complex.I)
  (hz1 : z1 = 1 + complex.I) (hz2 : z2 = ↑m - 2 * complex.I)
  (product : z1 * z2)
  (hproduct : product = m + 2 + (m - 2) * complex.I)
  (h_right : right_half_plane product) : m > -2 := 
by sorry

end find_m_find_m_range_l549_549700


namespace magnitude_of_sum_l549_549386

-- Define the vectors and conditions
def vec_a (m : ℝ) : ℝ × ℝ := (4, m)
def vec_b : ℝ × ℝ := (1, -2)

-- Condition: vectors are perpendicular
def vectors_perpendicular (m : ℝ) : Prop := (vec_a m).1 * vec_b.1 + (vec_a m).2 * vec_b.2 = 0

-- Prove the magnitude of the vector sum
theorem magnitude_of_sum (m : ℝ) (h : vectors_perpendicular m) :
  real.sqrt ((vec_a m).1 + 2 * vec_b.1)^2 + ((vec_a m).2 + 2 * vec_b.2)^2 = 2 * real.sqrt 10 :=
sorry

end magnitude_of_sum_l549_549386


namespace limit_hours_overtime_l549_549976

theorem limit_hours_overtime (R O : ℝ) (earnings total_hours : ℕ) (L : ℕ) 
    (hR : R = 16)
    (hO : O = R + 0.75 * R)
    (h_earnings : earnings = 864)
    (h_total_hours : total_hours = 48)
    (calc_earnings : earnings = L * R + (total_hours - L) * O) :
    L = 40 := by
  sorry

end limit_hours_overtime_l549_549976


namespace f_n_one_l549_549099

noncomputable def f : ℝ → ℝ := λ x, x / (x + 2)

def fn : ℕ+ → ℝ → ℝ 
| ⟨1, _⟩ x := f x
| ⟨n+1, h⟩ x := f (fn ⟨n + 1, nat.succ_pos n⟩ x)

theorem f_n_one (n : ℕ+) : fn n 1 = 1 / (2 ^ (n.val + 1) - 1) :=
sorry

end f_n_one_l549_549099


namespace slices_served_during_lunch_l549_549232

def total_slices_today : ℕ := 12
def dinner_slices : ℕ := 5
def lunch_slices : ℕ := total_slices_today - dinner_slices

theorem slices_served_during_lunch :
  lunch_slices = 7 :=
by
  unfold lunch_slices
  unfold total_slices_today
  unfold dinner_slices
  simp
  sorry

end slices_served_during_lunch_l549_549232


namespace dress_cost_l549_549767

theorem dress_cost (x : ℝ) 
  (h1 : 30 * x = 10 + x) 
  (h2 : 3 * ((10 + x) / 30) = x) : 
  x = 10 / 9 :=
by
  sorry

end dress_cost_l549_549767


namespace club_officer_selection_l549_549854

theorem club_officer_selection:
  let total_members := 24 in
  let boys := 12 in
  let girls := 12 in
  let ways_to_choose := girls * boys * (girls - 1) in
  ways_to_choose = 1584 :=
by
  sorry

end club_officer_selection_l549_549854


namespace smallest_b_inverse_undefined_l549_549194

def is_inverse_undefined_mod (b n : ℕ) : Prop :=
  Nat.gcd b n > 1

theorem smallest_b_inverse_undefined (b : ℕ) :
  is_inverse_undefined_mod b 36 ∧ is_inverse_undefined_mod b 55 ↔ b = 330 :=
begin
  sorry
end

end smallest_b_inverse_undefined_l549_549194


namespace compute_a_pow3_b_pow_neg2_l549_549096

-- Define the problem conditions
def a := (4 : ℚ) / 7
def b := (5 : ℚ) / 6

-- State the goal to prove the problem
theorem compute_a_pow3_b_pow_neg2 : a^3 * b^(-2) = 2304 / 8575 := by
  sorry

end compute_a_pow3_b_pow_neg2_l549_549096


namespace magician_trick_l549_549985

theorem magician_trick (T : ℕ) (cards : fin 52) (edge_choice : fin 52 → bool) :
  ∀ init_position, (init_position = 0 ∨ init_position = 51) →
  (∃ remaining_position, remaining_position = init_position ∧ (∀ k, cards k = true → k ≠ remaining_position)) :=
sorry

end magician_trick_l549_549985


namespace misha_total_students_l549_549467

-- Definitions based on the conditions
def misha_best_rank : ℕ := 75
def misha_worst_rank : ℕ := 75

-- Statement of the theorem to be proved
theorem misha_total_students (misha_is_best : misha_best_rank = 75) (misha_is_worst : misha_worst_rank = 75) : 
  (misha_best_rank - 1) + (misha_worst_rank - 1) + 1 = 149 :=
by
  sorry

end misha_total_students_l549_549467


namespace smallest_t_proof_l549_549911

noncomputable def smallest_t (a b : ℝ) (t : ℕ) : ℕ :=
  if h1 : a + t > b ∧ b + t > a ∧ a + b > t then t else 0

theorem smallest_t_proof :
  smallest_t 7.5 12 5 = 5 := by 
  have h1 : 7.5 + 5 > 12 := by norm_num
  have h2 : 12 + 5 > 7.5 := by norm_num
  have h3 : 7.5 + 12 > 5 := by norm_num
  simp [smallest_t, h1, h2, h3]
  sorry

end smallest_t_proof_l549_549911


namespace train_cross_post_time_proof_l549_549623

noncomputable def train_cross_post_time (speed_kmh : ℝ) (length_m : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  length_m / speed_ms

theorem train_cross_post_time_proof : train_cross_post_time 40 190.0152 = 17.1 := by
  sorry

end train_cross_post_time_proof_l549_549623


namespace percentage_of_Muscovy_ducks_l549_549526

theorem percentage_of_Muscovy_ducks
  (N : ℕ) (M : ℝ) (female_percentage : ℝ) (female_Muscovy : ℕ)
  (hN : N = 40)
  (hfemale_percentage : female_percentage = 0.30)
  (hfemale_Muscovy : female_Muscovy = 6)
  (hcondition : female_percentage * M * N = female_Muscovy) 
  : M = 0.5 := 
sorry

end percentage_of_Muscovy_ducks_l549_549526


namespace discounted_mysteries_l549_549215

variable (m : ℕ)

theorem discounted_mysteries (h1 : 20 = 20) (h2 : 12 = 12) 
                            (h3 : 0.375 * 12 * m + 0.055 * 20 * 5 = 19)
                            (h4 : 0.375 + 0.055 = 0.43)
                            (h5 : 0.375 = 0.375)
                            : m = 3 := 
by
  sorry

end discounted_mysteries_l549_549215


namespace largest_sum_ABC_l549_549051

theorem largest_sum_ABC (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : A * B * C = 3003) : 
  A + B + C ≤ 105 :=
sorry

end largest_sum_ABC_l549_549051


namespace f_2016_plus_f_2015_l549_549359

theorem f_2016_plus_f_2015 (f : ℝ → ℝ) 
  (H1 : ∀ x, f (-x) = -f x) -- Odd function property
  (H2 : ∀ x, f (x + 1) = f (-x + 1)) -- Even function property for f(x+1)
  (H3 : f 1 = 1) : 
  f 2016 + f 2015 = -1 :=
sorry

end f_2016_plus_f_2015_l549_549359


namespace shirts_per_minute_l549_549253

theorem shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) (h1 : total_shirts = 196) (h2 : total_minutes = 28) :
  total_shirts / total_minutes = 7 :=
by
  -- beginning of proof would go here
  sorry

end shirts_per_minute_l549_549253


namespace largest_non_factor_product_of_factors_of_100_l549_549536

theorem largest_non_factor_product_of_factors_of_100 :
  ∃ x y : ℕ, 
  (x ≠ y) ∧ 
  (0 < x ∧ 0 < y) ∧ 
  (x ∣ 100 ∧ y ∣ 100) ∧ 
  ¬(x * y ∣ 100) ∧ 
  (∀ a b : ℕ, 
    (a ≠ b) ∧ 
    (0 < a ∧ 0 < b) ∧ 
    (a ∣ 100 ∧ b ∣ 100) ∧ 
    ¬(a * b ∣ 100) → 
    (x * y) ≥ (a * b)) ∧ 
  (x * y) = 40 :=
by
  sorry

end largest_non_factor_product_of_factors_of_100_l549_549536


namespace train_alex_probability_l549_549239

/-- Definitions of randomized arrivals and waiting period -/
def train_arrival_range : set ℝ := {t : ℝ | 0 ≤ t ∧ t ≤ 120}
def alex_arrival_range : set ℝ := {t : ℝ | 60 ≤ t ∧ t ≤ 180}
def waiting_time : ℝ := 15

/-- The probability that the train will still be there when Alex arrives is 7/64 -/
theorem train_alex_probability : 
  (∑ x in alex_arrival_range, ∑ y in train_arrival_range, (if y ≤ x + waiting_time ∧ y ≥ x - waiting_time then 1 else 0)) 
  / ((120 - 0) * (180 - 60)) = 7 / 64 :=
sorry

end train_alex_probability_l549_549239


namespace problem_statement_l549_549085

noncomputable def g : ℝ → ℝ := sorry

axiom g_one : g 1 = 2
axiom functional_eq : ∀ (x y : ℝ), g (x^2 - y^2) = (x - y) * (g x + g y)

theorem problem_statement : let g3_values := {z : ℝ | ∃ (x : ℝ → ℝ) (H : ∀ x y, x = y), x = g(3)} in
  ∃ (n : ℕ) (s : ℝ), n = 1 ∧ s = 6 ∧ n * s = 6 :=
by
  sorry

end problem_statement_l549_549085


namespace extreme_point_inequality_l549_549370

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 - 2 * x + a * real.log x

theorem extreme_point_inequality (a x₁ x₂ : ℝ) 
  (h₀ : 0 < a) (h₁ : a < 1) (h₂ : x₁ < x₂) 
  (h₃ : f x₁ a = 0) (h₄ : f x₂ a = 0) 
  (h₅ : x₁ + x₂ = 2) (h₆ : x₁ * x₂ = a) : 
  f x₂ a > -2 := 
sorry

end extreme_point_inequality_l549_549370


namespace blue_pill_cost_l549_549064

theorem blue_pill_cost :
  ∀ (cost_yellow cost_blue : ℝ) (days : ℕ) (total_cost : ℝ),
    (days = 21) →
    (total_cost = 882) →
    (cost_blue = cost_yellow + 3) →
    (total_cost = days * (cost_blue + cost_yellow)) →
    cost_blue = 22.50 :=
by sorry

end blue_pill_cost_l549_549064


namespace octavio_can_reach_3_pow_2023_l549_549113

theorem octavio_can_reach_3_pow_2023 (n : ℤ) (hn : n ≥ 1) :
  ∃ (steps : ℕ → ℤ), steps 0 = n ∧ (∀ k, steps (k + 1) = 3 * (steps k)) ∧
  steps 2023 = 3 ^ 2023 :=
by
  sorry

end octavio_can_reach_3_pow_2023_l549_549113


namespace wage_difference_seven_l549_549969

-- Define the parameters and conditions
variables (P Q h : ℝ)

-- Given conditions
def condition1 : Prop := P = 1.5 * Q
def condition2 : Prop := P * h = 420
def condition3 : Prop := Q * (h + 10) = 420

-- Theorem to be proved
theorem wage_difference_seven (h : ℝ) (P Q : ℝ) 
  (h_condition1 : condition1 P Q)
  (h_condition2 : condition2 P h)
  (h_condition3 : condition3 Q h) :
  (P - Q) = 7 :=
  sorry

end wage_difference_seven_l549_549969


namespace sidney_initial_cans_l549_549490

theorem sidney_initial_cans :
  ∀ (kittens adult_cats : ℕ) (adult_cat_food kitten_food_per_day additional_food days : ℕ)
  (h1 : kittens = 4)
  (h2 : adult_cats = 3)
  (h3 : adult_cat_food = 1)
  (h4 : kitten_food_per_day = 3 / 4)
  (h5 : additional_food = 35)
  (h6 : days = 7),
  let total_adult_cat_food := adult_cats * adult_cat_food * days,
      total_kitten_food := kittens * kitten_food_per_day * days,
      total_food_needed := total_adult_cat_food + total_kitten_food,
      initial_food := total_food_needed - additional_food
  in initial_food = 7 := by
  sorry

end sidney_initial_cans_l549_549490


namespace product_gcd_lcm_l549_549306

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l549_549306


namespace vanessa_albums_l549_549542

theorem vanessa_albums (phone_pics camera_pics pics_per_album : ℕ) 
  (h_phone : phone_pics = 23)
  (h_camera : camera_pics = 7)
  (h_pics_per_album : pics_per_album = 6) : 
  (phone_pics + camera_pics) / pics_per_album = 5 := 
by 
  rw [h_phone, h_camera, h_pics_per_album]
  sorry

end vanessa_albums_l549_549542


namespace maria_compensation_l549_549109

def deposit_insurance (deposit : ℕ) : ℕ :=
  if deposit <= 1400000 then deposit else 1400000

def maria_deposit : ℕ := 1600000

theorem maria_compensation :
  deposit_insurance maria_deposit = 1400000 :=
by sorry

end maria_compensation_l549_549109


namespace product_gcd_lcm_24_60_l549_549322

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l549_549322


namespace gcd_lcm_product_24_60_l549_549311

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l549_549311


namespace number_of_triangles_l549_549691

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def num_valid_triangles : ℕ :=
  [ (9, 7, 5), (9, 7, 3), (9, 5, 3), (7, 5, 3) ].count (λ (t : ℕ × ℕ × ℕ), valid_triangle t.1 t.2 t.3)

theorem number_of_triangles : num_valid_triangles = 3 := by
  sorry

end number_of_triangles_l549_549691


namespace ball_of_yarn_costs_6_l549_549266

-- Define the conditions as variables and hypotheses
variable (num_sweaters : ℕ := 28)
variable (balls_per_sweater : ℕ := 4)
variable (price_per_sweater : ℕ := 35)
variable (gain_from_sales : ℕ := 308)

-- Define derived values
def total_revenue : ℕ := num_sweaters * price_per_sweater
def total_cost_of_yarn : ℕ := total_revenue - gain_from_sales
def total_balls_of_yarn : ℕ := num_sweaters * balls_per_sweater
def cost_per_ball_of_yarn : ℕ := total_cost_of_yarn / total_balls_of_yarn

-- The theorem to be proven
theorem ball_of_yarn_costs_6 :
  cost_per_ball_of_yarn = 6 :=
by sorry

end ball_of_yarn_costs_6_l549_549266


namespace trip_time_difference_l549_549219

theorem trip_time_difference 
  (speed : ℕ) (dist1 dist2 : ℕ) (time_per_hour : ℕ) 
  (h_speed : speed = 60) 
  (h_dist1 : dist1 = 360) 
  (h_dist2 : dist2 = 420) 
  (h_time_per_hour : time_per_hour = 60) : 
  ((dist2 / speed - dist1 / speed) * time_per_hour) = 60 := 
by
  sorry

end trip_time_difference_l549_549219


namespace gcd_lcm_product_24_60_l549_549299

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l549_549299


namespace intersection_eq_inter_l549_549748

noncomputable def M : Set ℝ := { x | x^2 < 4 }
noncomputable def N : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
noncomputable def inter : Set ℝ := { x | -1 < x ∧ x < 2 }

theorem intersection_eq_inter : M ∩ N = inter :=
by sorry

end intersection_eq_inter_l549_549748


namespace ice_cream_flavors_l549_549768

theorem ice_cream_flavors (n k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  (n + k - 1).choose (k - 1) = 84 :=
by
  have h3 : n = 6 := h1
  have h4 : k = 4 := h2
  rw [h3, h4]
  sorry

end ice_cream_flavors_l549_549768


namespace lee_cookies_with_efficiency_improvement_l549_549432

theorem lee_cookies_with_efficiency_improvement :
  let initial_cookies_per_cup := 24 / 3 in
  let improved_efficiency := initial_cookies_per_cup * 1.1 in
  let total_cookies := improved_efficiency * 4 in
  ∃ (cookies : ℕ), cookies = int.floor total_cookies ∧ cookies = 35 :=
sorry

end lee_cookies_with_efficiency_improvement_l549_549432


namespace concert_ratio_l549_549892

theorem concert_ratio (a c : ℕ) (h1 : 30 * a + 15 * c = 2250) (h2 : a ≥ 1) (h3 : c ≥ 1) :
  a = 50 ∧ c = 50 ∧ a = c := 
sorry

end concert_ratio_l549_549892


namespace cos_identity_l549_549771

theorem cos_identity
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_identity_l549_549771


namespace cubic_coefficient_relationship_l549_549709

theorem cubic_coefficient_relationship (a b c p q r : ℝ)
    (h1 : ∀ s1 s2 s3: ℝ, s1 + s2 + s3 = -a ∧ s1 * s2 + s2 * s3 + s3 * s1 = b ∧ s1 * s2 * s3 = -c)
    (h2 : ∀ s1 s2 s3: ℝ, s1^2 + s2^2 + s3^2 = -p ∧ s1^2 * s2^2 + s2^2 * s3^2 + s3^2 * s1^2 = q ∧ s1^2 * s2^2 * s3^2 = r) :
    p = a^2 - 2 * b ∧ q = b^2 + 2 * a * c ∧ r = c^2 :=
by
  sorry

end cubic_coefficient_relationship_l549_549709


namespace age_difference_l549_549961

-- Defining the conditions
variables (a b c : Nat)

-- Given conditions
def condition1 := b = 2 * c
def condition2 := a + b + c = 27
def condition3 := b = 10

-- Goal: Prove that a - b = 2
theorem age_difference (h1 : condition1) (h2 : condition2) (h3 : condition3) : a - b = 2 := by
  sorry

end age_difference_l549_549961


namespace total_cars_l549_549931

theorem total_cars (Tommy_cars Jessie_cars : ℕ) (older_brother_cars : ℕ) 
  (h1 : Tommy_cars = 3) 
  (h2 : Jessie_cars = 3)
  (h3 : older_brother_cars = Tommy_cars + Jessie_cars + 5) : 
  Tommy_cars + Jessie_cars + older_brother_cars = 17 := by
  sorry

end total_cars_l549_549931


namespace wholesom_bakery_sunday_loaves_l549_549500

theorem wholesom_bakery_sunday_loaves :
  ∃ a_Su : ℕ,
    ∃ a_W a_Th a_Fr a_Sa a_M : ℕ,
      a_W = 5 ∧
      a_Th = 7 ∧
      a_Fr = 10 ∧
      a_Sa = 14 ∧
      a_M = 25 ∧
      a_Th - a_W = 2 ∧
      a_Fr - a_Th = 3 ∧
      a_Sa - a_Fr = 4 ∧
      a_M - a_Su = 6 ∧
      a_Su = 19 :=
begin
  sorry
end

end wholesom_bakery_sunday_loaves_l549_549500


namespace area_sum_bound_l549_549818

theorem area_sum_bound (n : ℕ) (hn : 1 ≤ n) :
  let a : ℕ → ℝ := λ k, (1 / 2) ^ (k + 1) / (k + 1)
  in 
    0 ≤ Real.log 2 - 1/2 - ∑ i in Finset.range n, a (i + 1) ∧ 
    Real.log 2 - 1/2 - ∑ i in Finset.range n, a (i + 1) ≤ 1 / 2^(n + 1) :=
sorry

end area_sum_bound_l549_549818


namespace complex_square_sum_eq_zero_l549_549260

theorem complex_square_sum_eq_zero (i : ℂ) (h : i^2 = -1) : (1 + i)^2 + (1 - i)^2 = 0 :=
sorry

end complex_square_sum_eq_zero_l549_549260


namespace product_of_sequence_mod_5_l549_549324

theorem product_of_sequence_mod_5 :
  let seq := [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]
  (prod seq) % 5 = 4 :=
by
  sorry

end product_of_sequence_mod_5_l549_549324


namespace parallel_lines_a_value_perpendicular_lines_min_abs_ab_l549_549751

theorem parallel_lines_a_value (a : ℝ) (b : ℝ) 
  (h1 : b = -12) 
  (h2 : ∀ x y : ℝ, x + a^2 * y + 1 = 0 → (a^2 + 1) * x - b * y + 3 = 0 → ∀ (y2 : ℝ), (y2 = y) → x / y = -(a^2 + 1) / b):

  a = sqrt 3 ∨ a = -sqrt 3 := 
by
  sorry

theorem perpendicular_lines_min_abs_ab (a : ℝ) (b : ℝ) 
  (h1 : ∀ x y : ℝ, x + a^2 * y + 1 = 0 → (a^2 + 1) * x - b * y + 3 = 0 → ∀ (slope1 slope2 : ℝ),
    slope1 = -(1 / (a^2)) ∧ slope2 = b / -(a^2 + 1) → slope1 * slope2 = -1):

  ∃ min_value, min_value = 2 := 
by
  sorry

end parallel_lines_a_value_perpendicular_lines_min_abs_ab_l549_549751


namespace min_a_value_l549_549780

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x ≤ 1/2) → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_a_value_l549_549780


namespace mean_of_five_numbers_l549_549161

theorem mean_of_five_numbers (sum : ℚ) (h : sum = 3 / 4) : (sum / 5 = 3 / 20) :=
by
  -- Proof omitted
  sorry

end mean_of_five_numbers_l549_549161


namespace max_area_of_garden_l549_549997

theorem max_area_of_garden (x y : ℝ) (h : x + y = 18) : 
  (∀ x y, (x + y = 18) → (x * y ≤ 81) :=
begin
  sorry,
end.


end max_area_of_garden_l549_549997


namespace mass_percentages_C3H5ClO_l549_549279

def atomic_mass_C : ℚ := 12.01
def atomic_mass_H : ℚ := 1.01
def atomic_mass_Cl : ℚ := 35.45
def atomic_mass_O : ℚ := 16.00

def molar_mass_C3H5ClO : ℚ := 
  (3 * atomic_mass_C) + (5 * atomic_mass_H) + atomic_mass_Cl + atomic_mass_O

theorem mass_percentages_C3H5ClO :
  molar_mass_C3H5ClO = 92.53 ∧
  (5 * atomic_mass_H / molar_mass_C3H5ClO * 100) ≈ 5.46 ∧
  (3 * atomic_mass_C / molar_mass_C3H5ClO * 100) ≈ 38.94 ∧
  (atomic_mass_Cl / molar_mass_C3H5ClO * 100) ≈ 38.32 ∧
  (atomic_mass_O / molar_mass_C3H5ClO * 100) ≈ 17.29 := 
by
  -- The detailed steps would be added here to complete the proof
  -- Currently we skip the proof with sorry
  sorry

end mass_percentages_C3H5ClO_l549_549279


namespace inequality_holds_l549_549454

theorem inequality_holds (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1 / (x + y) + 4 / (y + z) + 9 / (x + z) ≥ 18 / (x + y + z) :=
begin
  sorry
end

end inequality_holds_l549_549454


namespace product_gcd_lcm_24_60_l549_549319

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l549_549319


namespace smallest_x_l549_549651

noncomputable def f (x : ℝ) : ℝ :=
if 1 ≤ x ∧ x ≤ 3 then 1 - |x - 2| else 3 * f (x / 3)

theorem smallest_x (x : ℝ) :
  (∀ x : ℝ, f(3 * x) = 3 * f(x)) →
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f(x) = 1 - |x - 2|) →
  f(2017) = f(x) →
  x = 413 := sorry

end smallest_x_l549_549651


namespace number_of_integer_values_not_satisfying_inequality_l549_549689

theorem number_of_integer_values_not_satisfying_inequality :
  finset.card {x : ℤ | 3 * x^2 + 11 * x + 14 ≤ 17} = 3 :=
by
  -- The detailed proof steps will go here.
  sorry

end number_of_integer_values_not_satisfying_inequality_l549_549689


namespace ending_number_of_X_is_12_l549_549486

open Finset

-- Define the sets X and Y based on the given conditions
variable (n : ℕ)
def X := Icc 1 n
def Y := Icc 0 20

-- Hypothesis: 12 distinct integers belong to both sets at the same time
variable (h : (X n ∩ Y).card = 12)

-- Prove the end number of set X
theorem ending_number_of_X_is_12 : n = 12 :=
by
  sorry

end ending_number_of_X_is_12_l549_549486


namespace perpendicular_planes_l549_549382

variables {Point : Type*} [affine_space Point]
variables {Line : Type*} [linear_space Line Point]
variables {Plane : Type*} [linear_space Plane Point]

def non_intersecting (l m : Line) : Prop := ∀ p : Point, ¬ (p ∈ l ∧ p ∈ m)
def perpendicular (l : Line) (p : Plane) : Prop := ∀ q : Line, q ∈ p → l ⊥ q
def line_perpendicular (l m : Line) : Prop := l ⊥ m
def plane_perpendicular (α β : Plane) : Prop := α ⊥ β

variables (l m : Line) (α β : Plane)

theorem perpendicular_planes
  (hlm : non_intersecting l m)
  (halphabeta : non_intersecting α β)
  (hlperpm : line_perpendicular l m)
  (hmperpbeta : perpendicular m β)
  (hlperpalpha : perpendicular l α) :
  plane_perpendicular α β :=
sorry

end perpendicular_planes_l549_549382


namespace estimate_total_fish_in_pond_l549_549179

theorem estimate_total_fish_in_pond :
  ∀ (total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample : ℕ),
  initial_sample_size = 100 →
  second_sample_size = 200 →
  tagged_in_second_sample = 10 →
  total_tagged_fish = 100 →
  (total_tagged_fish : ℚ) / (total_fish : ℚ) = tagged_in_second_sample / second_sample_size →
  total_fish = 2000 := by
  intros total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample
  intro h1 h2 h3 h4 h5
  sorry

end estimate_total_fish_in_pond_l549_549179


namespace log_46382_bounds_l549_549288

theorem log_46382_bounds : ∃ c d : ℤ, 4 < real.log10 46382 ∧ real.log10 46382 < 5 ∧ c + d = 9 := 
by {
  use [4, 5],
  constructor,
  { exact sorry }, -- proof that 4 < real.log10 46382
  { constructor,
    { exact sorry }, -- proof that real.log10 46382 < 5
    { norm_num } -- proof that 4 + 5 = 9
  }
}

end log_46382_bounds_l549_549288


namespace probability_even_palindromic_number_l549_549110

theorem probability_even_palindromic_number : 
  (let total_palindromic := 9 * 10 in
   let even_palindromic := 4 * 10 in
   even_palindromic / total_palindromic = 4 / 9) :=
by
  sorry

end probability_even_palindromic_number_l549_549110


namespace volume_of_rectangular_parallelepiped_l549_549890

theorem volume_of_rectangular_parallelepiped:
  ∀ (a b c : ℝ), 
    (a^2 + b^2 + c^2 = 169) →
    (a^2 + b^2 = (3 * (√17))^2) →
    (b^2 + c^2 = (4 * (√10))^2) →
    a * b * c = 144 := by
  sorry

end volume_of_rectangular_parallelepiped_l549_549890


namespace heloise_gives_dogs_to_janet_l549_549760

theorem heloise_gives_dogs_to_janet :
  ∃ d c : ℕ, d * 17 = c * 10 ∧ d + c = 189 ∧ d - 60 = 10 :=
by
  sorry

end heloise_gives_dogs_to_janet_l549_549760


namespace complex_power_l549_549581

theorem complex_power : (1 - Complex.i)^4 = -4 :=
by
  sorry

end complex_power_l549_549581


namespace stability_measure_of_performance_l549_549533

-- Define the conditions and the relevant statistical concepts
def variance (data : List ℝ) : ℝ :=
  let mean := (data.sum / data.length)
  (data.map (λ x => (x - mean) ^ 2)).sum / data.length

-- The theorem statement, which we need to prove
theorem stability_measure_of_performance (data : List ℝ) (h : data.length = 10) :
  (variance data = variance data) :=
by
  sorry

end stability_measure_of_performance_l549_549533


namespace arithmetic_seq_diff_1500_1520_l549_549545

theorem arithmetic_seq_diff_1500_1520 :
  let a_1 := -12
  let d := -7 - (-12)
  let a_1500 := a_1 + 1499 * d
  let a_1520 := a_1 + 1519 * d
  0 < a_1520 - a_1500 :=
  a_1520 - a_1500 = 100 := by
  sorry

end arithmetic_seq_diff_1500_1520_l549_549545


namespace sets_given_l549_549250

variable (initial_sets : Nat) (sets_left : Nat)

theorem sets_given (h1 : initial_sets = 90) (h2 : sets_left = 39) : initial_sets - sets_left = 51 :=
by
  rw [h1, h2]
  exact rfl

end sets_given_l549_549250


namespace series_divergence_counterexample_l549_549423

noncomputable def counterexample_sequence_a (n : ℕ) : ℝ :=
if n % 2 = 0 then (n^2 : ℝ) else (1 / (n^2 : ℝ))

noncomputable def counterexample_sequence_b (n : ℕ) : ℝ :=
if n % 2 = 0 then (1 / (n^2 : ℝ)) else (n^2 : ℝ)

theorem series_divergence_counterexample :
  (∀ n, 0 < counterexample_sequence_a n) ∧ (⊥ : ℝ∞) = ∑' n, counterexample_sequence_a n ∧
  (∀ n, 0 < counterexample_sequence_b n) ∧ (⊥ : ℝ∞) = ∑' n, counterexample_sequence_b n →
  (∞ : ℝ∞) ≠ ∑' n, 2 * counterexample_sequence_a n * counterexample_sequence_b n / (counterexample_sequence_a n + counterexample_sequence_b n) :=
sorry

end series_divergence_counterexample_l549_549423


namespace number_of_valid_n_l549_549402

theorem number_of_valid_n (a : ℝ) (f : ℝ → ℝ) :
  (∃ n : ℕ, ∀ x ∈ (0, n * Real.pi), f(x) = Real.cos (2 * x) - a * Real.sin x ∧ (set_of (λ x, f x = 0)).card = 2022) →
  {n : ℕ | ∃ x ∈ (0, n * Real.pi), f x = 0}.finite.card = 5 :=
sorry

end number_of_valid_n_l549_549402


namespace part_a_part_b_l549_549965

-- Define the condition for having no odd prime divisor less than a given number
def has_no_odd_prime_divisor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p ∧ p % 2 = 1 ∧ p < k → ¬ (p ∣ n)

-- Assume a noncomputable instance because we are dealing with possibly large numbers
noncomputable def pair_existance_condition (lst : List ℕ): Prop :=
  ∃ a b ∈ lst, has_no_odd_prime_divisor_less_than (a + b) 37

-- The main theorem for part (a)
theorem part_a (lst : List ℕ) (h : lst.length = 2019)
  (h_prime_divisor : ∀ n ∈ lst, has_no_odd_prime_divisor_less_than n 37) :
  pair_existance_condition lst :=
sorry

-- Define the condition for pair existence changing 37 to 38
noncomputable def pair_existance_condition_38 (lst : List ℕ): Prop :=
  ∃ a b ∈ lst, has_no_odd_prime_divisor_less_than (a + b) 38

-- The main theorem for part (b)
theorem part_b (lst : List ℕ) (h : lst.length = 2019)
  (h_prime_divisor : ∀ n ∈ lst, has_no_odd_prime_divisor_less_than n 38) :
  ¬ pair_existance_condition_38 lst :=
sorry

end part_a_part_b_l549_549965


namespace initial_condition_proof_move_to_1_proof_move_to_2_proof_recurrence_relation_proof_p_99_proof_p_100_proof_l549_549496

variable (p : ℕ → ℚ)

-- Given conditions
axiom initial_condition : p 0 = 1
axiom move_to_1 : p 1 = 1 / 2
axiom move_to_2 : p 2 = 3 / 4
axiom recurrence_relation : ∀ n : ℕ, 2 ≤ n → n ≤ 99 → p n - p (n - 1) = - 1 / 2 * (p (n - 1) - p (n - 2))
axiom p_99_cond : p 99 = 2 / 3 - 1 / (3 * 2^99)
axiom p_100_cond : p 100 = 1 / 3 + 1 / (3 * 2^99)

-- Proof that initial conditions are met
theorem initial_condition_proof : p 0 = 1 :=
sorry

theorem move_to_1_proof : p 1 = 1 / 2 :=
sorry

theorem move_to_2_proof : p 2 = 3 / 4 :=
sorry

-- Proof of the recurrence relation
theorem recurrence_relation_proof : ∀ n : ℕ, 2 ≤ n → n ≤ 99 → p n - p (n - 1) = - 1 / 2 * (p (n - 1) - p (n - 2)) :=
sorry

-- Proof of p_99
theorem p_99_proof : p 99 = 2 / 3 - 1 / (3 * 2^99) :=
sorry

-- Proof of p_100
theorem p_100_proof : p 100 = 1 / 3 + 1 / (3 * 2^99) :=
sorry

end initial_condition_proof_move_to_1_proof_move_to_2_proof_recurrence_relation_proof_p_99_proof_p_100_proof_l549_549496


namespace rectangle_fitting_condition_l549_549853

variables {a b c d : ℝ}

theorem rectangle_fitting_condition
  (h1: a < c ∧ c ≤ d ∧ d < b)
  (h2: a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b*c - a*d)^2 + (b*d - a*c)^2 :=
sorry

end rectangle_fitting_condition_l549_549853


namespace other_function_value_at_20_l549_549539

def linear_function (k b : ℝ) (x : ℝ) : ℝ :=
  k * x + b

theorem other_function_value_at_20
    (k1 k2 b1 b2 : ℝ)
    (h_intersect : linear_function k1 b1 2 = linear_function k2 b2 2)
    (h_diff_at_8 : abs (linear_function k1 b1 8 - linear_function k2 b2 8) = 8)
    (h_y1_at_20 : linear_function k1 b1 20 = 100) :
  linear_function k2 b2 20 = 76 ∨ linear_function k2 b2 20 = 124 :=
sorry

end other_function_value_at_20_l549_549539


namespace irrational_numbers_count_is_3_l549_549860

def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), (q : ℝ) = x

def irrational_count : ℕ :=
  [4 * Real.pi, 0, Real.sqrt 7, Real.sqrt 16 / 2, 0.1, 0.212212221 -- defined pattern needs to be handled
  ].filter (λ x, is_irrational x).length

theorem irrational_numbers_count_is_3 : irrational_count = 3 := by
  sorry

end irrational_numbers_count_is_3_l549_549860


namespace no_increasing_g_h_l549_549433

-- Assume the function f is given with the stated property
def f (x : ℝ) : ℝ := if h : ∃ (n : ℕ), x = 1 / (n + 1) then (-1)^(classical.some h) else sorry

-- Claim that there are no such functions g and h that are both increasing and f = g - h
theorem no_increasing_g_h (g h : ℝ → ℝ)
  (hg : ∀ x y, x < y → g x ≤ g y)
  (hh : ∀ x y, x < y → h x ≤ h y)
  (H : ∀ x, 0 < x ∧ x < 1 → f x = g x - h x) : 
  false := sorry

end no_increasing_g_h_l549_549433


namespace valid_license_plates_count_l549_549244

theorem valid_license_plates_count :
  let letters := 26 * 26 * 26
  let digits := 9 * 10 * 10
  letters * digits = 15818400 :=
by
  sorry

end valid_license_plates_count_l549_549244


namespace color_white_squares_l549_549848

/-- 
Given an infinite grid of white squares with a finite number of black squares such that 
each black square has an even number of adjacent white squares (0, 2, or 4), 
prove that it is possible to color the remaining white squares red or green 
such that each black square has an equal number of red and green adjacent squares.
-/
theorem color_white_squares (infinite_grid : ℕ → ℕ → ℝ) 
    (finite_black_squares : set (ℕ × ℕ)) 
    (condition : ∀ (b : ℕ × ℕ), b ∈ finite_black_squares → 
                 ∃ n, n ∈ {0, 2, 4} ∧ adjacent_white_squares b = n):
    ∃ (red_green_coloring : ℕ → ℕ → ℕ), 
    (∀ (b : ℕ × ℕ), b ∈ finite_black_squares → 
     count_adjacent_colors b red_green_coloring = 2) :=
sorry

end color_white_squares_l549_549848


namespace mean_of_five_numbers_l549_549162

theorem mean_of_five_numbers (sum : ℚ) (h : sum = 3 / 4) : (sum / 5 = 3 / 20) :=
by
  -- Proof omitted
  sorry

end mean_of_five_numbers_l549_549162


namespace taobao_villages_2023_l549_549788

variable (a1 : ℕ) (d : ℕ)

theorem taobao_villages_2023 : 
  a1 = 1311 → 
  d = 1000 → 
  ∃ n, n = 7 → a1 + n * d = 8311 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3] 
  sorry

end taobao_villages_2023_l549_549788


namespace odd_cube_difference_divisible_by_power_of_two_l549_549090

theorem odd_cube_difference_divisible_by_power_of_two {a b n : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 1) :
  (2^n ∣ (a^3 - b^3)) ↔ (2^n ∣ (a - b)) :=
by
  sorry

end odd_cube_difference_divisible_by_power_of_two_l549_549090


namespace range_of_f_l549_549655

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin (π / 4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x)

theorem range_of_f : Set.Icc (Real.min (f (π / 4)) (f (π / 2))) (Real.max (f (π / 4)) (f (π / 2))) = Set.Icc 2 3 :=
by
  sorry

end range_of_f_l549_549655


namespace problem_lean_l549_549144

theorem problem_lean :
  ∃ (D E F : ℤ), 
    (∀ x, x > 5 → (let g := λ x, x^2 / (D * x^2 + E * x + F) in g x > 0.3)) ∧
    (let g := λ x, x^2 / (D * x^2 + E * x + F) in g) → 
    (∃ h_asymptote : ℝ, h_asymptote = (1 : ℝ) / (D : ℝ) ∧ h_asymptote < 1 ∧ h_asymptote > 0.3) ∧
    (∀ x, (D * x^2 + E * x + F) = 0 → x = -3) ∧ 
    D + E + F = 48 := 
begin
  sorry
end

end problem_lean_l549_549144


namespace find_number_l549_549119

theorem find_number : ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = 7.5 :=
by {
  sorry
}

end find_number_l549_549119


namespace braiding_time_l549_549067

variables (n_dancers : ℕ) (b_braids_per_dancer : ℕ) (t_seconds_per_braid : ℕ)

theorem braiding_time : n_dancers = 8 → b_braids_per_dancer = 5 → t_seconds_per_braid = 30 → 
  (n_dancers * b_braids_per_dancer * t_seconds_per_braid) / 60 = 20 :=
by
  intros
  sorry

end braiding_time_l549_549067


namespace number_of_action_figures_removed_l549_549810

-- Definitions for conditions
def initial : ℕ := 15
def added : ℕ := 2
def current : ℕ := 10

-- The proof statement
theorem number_of_action_figures_removed (initial added current : ℕ) : 
  (initial + added - current) = 7 := by
  sorry

end number_of_action_figures_removed_l549_549810


namespace Petya_can_afford_l549_549553

open Real

def price_meat := 400
def price_onions := 50
def discount := 0.25
def initial_money := 1200
def qualify_discount_purchase := 900

def total_cost_no_discount (kg_meat kg_onions : ℝ) : ℝ := (kg_meat * price_meat) + (kg_onions * price_onions)

def first_purchase_cost (kg_meat kg_onions : ℝ) : ℝ := (kg_meat * price_meat) + (kg_onions * price_onions)

def discounted_price (price : ℝ) : ℝ := price * (1 - discount)

theorem Petya_can_afford : ∃ (kg_meat1 kg_onions1 kg_meat2 kg_onions2: ℝ), 
  kg_meat1 = 2 ∧ kg_onions1 = 2 ∧ kg_meat2 = 1 ∧ kg_onions2 = 0 ∧
  first_purchase_cost kg_meat1 kg_onions1 = qualify_discount_purchase ∧
  initial_money - qualify_discount_purchase = discounted_price (kg_meat2 * price_meat) ∧
  initial_money = qualify_discount_purchase + discounted_price (kg_meat2 * price_meat) :=
sorry

end Petya_can_afford_l549_549553


namespace sin_phi_l549_549828

variables (u v w : ℝ^3)
variable (φ : ℝ)

-- Conditions
axiom h1 : u ≠ 0 ∧ v ≠ 0 ∧ w ≠ 0
axiom h2 : ¬(∀ k : ℝ, u = k • v) ∧ ¬(∀ k : ℝ, v = k • w) ∧ ¬(∀ k : ℝ, w = k • u)
axiom h3 : (u × v) × w = (1 / 4) * ∥v∥ * ∥w∥ • u

-- Statement to prove
theorem sin_phi : sin φ = sqrt 15 / 4 :=
sorry

end sin_phi_l549_549828


namespace larger_number_l549_549146

theorem larger_number (A B : ℕ) (h : nat.gcd A B = 23) (lcm_factors : nat.lcm A B = 23 * 15 * 16) : max A B = 23 * 16 :=
begin
  sorry
end

end larger_number_l549_549146


namespace find_angle_l549_549355

-- Define vectors and conditions
variables {α : Type*} [inner_product_space ℝ α]
variables (a b : α)

-- Given conditions
def condition1 : ∥a∥ = 2 := sorry
def condition2 : ∥b∥ = 4 := sorry
def condition3 : inner (a + b) a = 0 := sorry

-- The angle between a and b
noncomputable def angle : ℝ := real.arccos ((inner a b) / (∥a∥ * ∥b∥))

-- The statement we want to prove
theorem find_angle : (condition1 ∧ condition2 ∧ condition3) → angle a b = 2 * real.pi / 3 := sorry

end find_angle_l549_549355


namespace solution_set_of_inequality_l549_549914

theorem solution_set_of_inequality (x : ℝ) : 
  \frac{x-2}{x+1} < 0 ↔ -1 < x ∧ x < 2 :=
sorry

end solution_set_of_inequality_l549_549914


namespace least_number_subtraction_l549_549196

theorem least_number_subtraction (n : ℤ) (d : ℤ) (r : ℤ) (k : ℤ) :
  n = 218791 ∧ d = 953 ∧ r = n % d ∧ k = n - r → k % d = 0 :=
by
  intros h
  rcases h with ⟨hn, hd, hr, hk⟩
  rw [hn, hd, hr, hk]
  sorry

end least_number_subtraction_l549_549196


namespace sum_of_digits_base2_365_l549_549548

theorem sum_of_digits_base2_365 :
  let binary_365 := 256 + 64 + 32 + 8 + 4 + 1 in
  let sum_of_digits := (1 + 0 + 1 + 1 + 0 + 1 + 1 + 0 + 1) in
  sum_of_digits = 6 := by
  sorry

end sum_of_digits_base2_365_l549_549548


namespace charity_race_finished_racers_l549_549930

theorem charity_race_finished_racers :
  let initial_racers := 50
  let joined_after_20_minutes := 30
  let doubled_after_30_minutes := 2
  let dropped_racers := 30
  let total_racers_after_20_minutes := initial_racers + joined_after_20_minutes
  let total_racers_after_50_minutes := total_racers_after_20_minutes * doubled_after_30_minutes
  let finished_racers := total_racers_after_50_minutes - dropped_racers
  finished_racers = 130 := by
    sorry

end charity_race_finished_racers_l549_549930


namespace solve_inequality_l549_549369

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2
  else x^2 + 2*x

theorem solve_inequality : { x : ℝ | f (f x) ≤ 3 } = { x : ℝ | x ≤ Real.sqrt 3 } :=
by
  sorry

end solve_inequality_l549_549369


namespace T_b_sum_l549_549327

noncomputable def T (r : ℝ) : ℝ :=
  if -1 < r ∧ r < 1 then 24 / (1 - r) else 0

theorem T_b_sum (b : ℝ) (h : -1 < b ∧ b < 1) (hT : T b * T (-b) = 4032) :
  T b + T (-b) = 336 :=
by
  have hb1 : -1 < b := h.1
  have hb2 : b < 1 := h.2
  have hT_pos : T b > 0 := sorry -- Acknowledge gaps in geometric series, provided as a known.
  have h_neg_b_pos : T (-b) > 0 := sorry -- Similarly acknowledge that -b is in -1 < r < 1.
  -- Assuming T evaluated as per defined in the interval.
  have T_b_eq : T b = 24 / (1 - b) := sorry
  have T_neg_b_eq : T (-b) = 24 / (1 + b) := sorry
  
  have hEq_mult : (24 / (1 - b)) * (24 / (1 + b)) = 4032 := sorry
  -- Reduce this multiplication and substituting assuming both positive derivation from hT to show next.
  have T_eq_4032 : (24 * 24) / ((1 - b) * (1 + b)) = 4032 := sorry
  -- This should lead showing 1-b^2 derivation ensuring through correct steps equivalency at (1 - b^2)
  have denominator_eq : 1 - b * b = 1 / 7 := sorry
  -- Using (1 - b^2) inversely leads to computing next equivalently.
  have numerator_sum : (24 / (1 - b)) + (24 / (1 + b)) = (48 /( 1 - b^2)) := sorry
  -- Further calculations combining factoring lead final results with equivalent valid derivations avoiding any assumed solution dependency specifics.
  show T b + T (-b) = 48 * 7 by sorry -- Here 336 as final equivalent evaluated steps forward simplifying operationally achieving 48 * 7 = 336. 

end T_b_sum_l549_549327


namespace smallest_four_digit_palindrome_divisible_by_3_l549_549946

theorem smallest_four_digit_palindrome_divisible_by_3 :
  ∃ (n : ℕ), (n = 2112) ∧ (1000 ≤ n) ∧ (n < 10000) ∧
    (∀ m : ℕ, (1000 ≤ m) → (m < 10000) → 
        (m / 1000 = m % 10) ∧ ((m / 100) % 10 = (m / 10) % 10) →
        (m % 3 ≠ 0 ∨ m ≥ n)) := 
begin
  sorry
end

end smallest_four_digit_palindrome_divisible_by_3_l549_549946


namespace fraction_increase_by_five_l549_549034

variable (x y : ℝ)

theorem fraction_increase_by_five :
  let f := fun x y => (x * y) / (2 * x - 3 * y)
  f (5 * x) (5 * y) = 5 * (f x y) :=
by
  sorry

end fraction_increase_by_five_l549_549034


namespace min_transportation_cost_l549_549114

theorem min_transportation_cost :
  let distance (w1 w2 : ℕ) : ℕ := (w2 - w1) * 100
  let cost_per_ton_km : ℝ := 0.5
  let goods_in_warehouse : ℕ → ℕ
  | 1 => 10
  | 2 => 20
  | 3 => 0
  | 4 => 0
  | 5 => 40
  ∀ (target_warehouse : ℕ), target_warehouse = 5 →
  let total_cost : ℝ :=
    ((distance 1 2) * (goods_in_warehouse 1) * cost_per_ton_km) + 
    ((distance 2 5) * ((goods_in_warehouse 1) + (goods_in_warehouse 2)) * cost_per_ton_km)
  total_cost = 5000 := 
by
  sorry

end min_transportation_cost_l549_549114


namespace cos_B_equiv_l549_549718

variable {A B C : ℝ}
variable {a b c : ℝ}
variable {GA GB GC : ℝ → ℝ}
variable {G : ℝ}

theorem cos_B_equiv :
  (∀ GA GB GC : ℝ → ℝ, 
  ∀ (A B C : ℝ) (a b c : ℝ),
    2 * sin A * GA = sqrt 3 * sin B * GB + 3 * sin C * GC → 
    2 * a * GA + sqrt 3 * b * GB + 3 * c * GC = 0 → 
    (GA + GB + GC = 0) →
    a = sqrt 3 / 2 * b → 
    c = sqrt 3 / 3 * b → 
    cos B = a^2 + c^2 - b^2 / 
    (2 * a * c) -> 
    cos B = 1 / 12) 
:= 
sorry

end cos_B_equiv_l549_549718


namespace max_value_and_period_l549_549101

noncomputable def vector_op (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 * b.1, a.2 * b.2)

def m : ℝ × ℝ := (2, 1/2)
def n : ℝ × ℝ := (Real.pi / 3, 0)
def f (x : ℝ) : ℝ := (1/2) * Real.sin ((1/2) * x - (Real.pi / 6))

theorem max_value_and_period :
  (∀ x, f x ≤ 1/2) ∧ (∀ t, t > 0 → (∀ x, f (x + t) = f x) → t = 4 * Real.pi) :=
by
  sorry

end max_value_and_period_l549_549101


namespace roots_of_unity_of_quadratic_eq_l549_549654

theorem roots_of_unity_of_quadratic_eq (a : ℤ) (z : ℂ) (hz : z^4 = 1) (h_eq : 2*z^2 + a*z + 1 = 0)
  (ha : -3 ≤ a ∧ a ≤ 3) : z = complex.I ∨ z = -complex.I :=
by
  sorry

end roots_of_unity_of_quadratic_eq_l549_549654


namespace T4_champion_probability_l549_549923

theorem T4_champion_probability : 
  let p := 256, q := 525 in 
  Nat.gcd p q = 1 ∧ p + q = 781 :=
by
  sorry

end T4_champion_probability_l549_549923


namespace probability_A8_l549_549125

/-- Define the probability of event A_n where the sum of die rolls equals n -/
def P (n : ℕ) : ℚ :=
  1/7 * (if n = 8 then 5/36 + 21/216 + 35/1296 + 35/7776 + 21/46656 +
    7/279936 + 1/1679616 else 0)

theorem probability_A8 : P 8 = (1/7) * (5/36 + 21/216 + 35/1296 + 35/7776 + 
  21/46656 + 7/279936 + 1/1679616) :=
by
  sorry

end probability_A8_l549_549125


namespace commute_time_difference_l549_549228

theorem commute_time_difference (x y : ℝ) 
  (h1 : x + y = 39)
  (h2 : (x - 10)^2 + (y - 10)^2 = 10) :
  |x - y| = 4 :=
by
  sorry

end commute_time_difference_l549_549228


namespace rational_segments_of_height_l549_549966

noncomputable theory
open Real

variables {A B C B₁ : ℝ}

/-- In triangle ABC with rational side lengths, if a height BB₁ is drawn from B, then the segments AB₁ and CB₁ are rational. -/
theorem rational_segments_of_height (AB BC AC : ℝ) (hAB : rat AB) (hBC : rat BC) (hAC : rat AC) (BB₁ : ℝ) (hBB₁ : BB₁^2 = AB^2 - ((AB*AC - BC^2)/(2*AC))^2) : 
  (AB₁ CB₁ : ℝ) (hAB₁ : rat AB₁) (hCB₁ : rat CB₁) :=
  sorry

end rational_segments_of_height_l549_549966


namespace proof_equation_solutions_l549_549958

noncomputable def solve_log_equation (x : ℝ) : Prop :=
  20 * Real.log (x.sqrt) / Real.log (4*x) +
  7 * Real.log (x^3) / Real.log (16*x) -
  3 * Real.log (x^2) / Real.log (x/2) = 0

def domain_conditions (x : ℝ) : Prop :=
  x > 0 ∧ x ≠ 1/4 ∧ x ≠ 1/16 ∧ x ≠ 2

theorem proof_equation_solutions :
  ∀ x : ℝ, domain_conditions x → solve_log_equation x → 
  x = 1 ∨ x = 1 / (4 * Real.root 8 5) ∨ x = 4 :=
begin
  sorry
end

end proof_equation_solutions_l549_549958


namespace friends_count_is_four_l549_549111

def number_of_friends (Melanie Benny Sally Jessica : ℕ) (total_cards : ℕ) : ℕ :=
  4

theorem friends_count_is_four (Melanie Benny Sally Jessica : ℕ) (total_cards : ℕ) (h1 : total_cards = 12) :
  number_of_friends Melanie Benny Sally Jessica total_cards = 4 :=
by
  sorry

end friends_count_is_four_l549_549111


namespace sqrt_expression_value_l549_549640

theorem sqrt_expression_value (A : ℝ) :
  (∀ θ ∈ {real.pi / 9, 2 * real.pi / 9, 4 * real.pi / 9}, real.cos (9 * θ) = 1) →
  (3 - real.cos (real.pi / 9)^2) * (3 - real.cos (2 * real.pi / 9)^2) * (3 - real.cos (4 * real.pi / 9)^2) = 196 →
  real.sqrt((3 - real.cos (real.pi / 9)^2) * (3 - real.cos (2 * real.pi / 9)^2) * (3 - real.cos (4 * real.pi / 9)^2)) = 14 / real.sqrt(A) :=
by sorry

end sqrt_expression_value_l549_549640


namespace midpoints_form_dodecagon_l549_549477

-- Definitions to set up the problem
variables (A B C D K L M N P : Point)
variables (AB: Line A B) (BC: Line B C) (CD: Line C D) (DA: Line D A)
variables (ABK: Triangle A B K) (BCL: Triangle B C L) (CDM: Triangle C D M) (DAN: Triangle D A N)

-- Conditions of the problem
axiom square_ABCD : square ABCD
axiom eq_triangle_ABK : equilateral_triangle ABK ∧ K ∈ interior (square ABCD)
axiom eq_triangle_BCL : equilateral_triangle BCL ∧ L ∈ interior (square ABCD)
axiom eq_triangle_CDM : equilateral_triangle CDM ∧ M ∈ interior (square ABCD)
axiom eq_triangle_DAN : equilateral_triangle DAN ∧ N ∈ interior (square ABCD)
axiom midpoint_P : midpoint P = midpoint_of (side (BCL, L)) -- Similar for sides in the other triangles
axiom midpoint_Q : midpoint Q = midpoint_of (side (CDM, M))
axiom midpoint_R : midpoint R = midpoint_of (side (DAN, N))
axiom midpoint_S : midpoint S = midpoint_of (side (ABK, K))

-- Prove that these points form a regular dodecagon
theorem midpoints_form_dodecagon : 
    regular_dodecagon { midpoint P, midpoint Q, midpoint R, midpoint S, 
                        midpoint (segment KL), midpoint (segment LM), 
                        midpoint (segment MN), midpoint (segment NK) } :=
sorry

end midpoints_form_dodecagon_l549_549477


namespace minimum_floor_radius_l549_549448

noncomputable def min_floor_radius (O A : ℝ × ℝ) (r : ℝ) : ℝ :=
  if OA = 30 then
    if ∃ B C : ℝ × ℝ, (on_circle O r B C) ∧ (angle_ABC_90 B C) ∧ (AB_eq_BC B C) then
      ⌊r⌋
    else
      sorry
  else
    sorry

theorem minimum_floor_radius : 
  ∀ (O A : ℝ × ℝ),
    ∀ (r : ℝ),
      min_floor_radius O A r = 12 :=
by
  sorry

end minimum_floor_radius_l549_549448


namespace satisfies_conditions_l549_549505

def f (x : ℝ) : ℝ := (1 / 2) ^ |x|

theorem satisfies_conditions :
  (f 0 = 1) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → f x > f y) ∧
  (∀ x : ℝ, f x = f (-x)) :=
by
  sorry

end satisfies_conditions_l549_549505


namespace sum_of_fourth_powers_correct_l549_549513

noncomputable def sum_of_fourth_powers (x : ℤ) : ℤ :=
  x^4 + (x+1)^4 + (x+2)^4

theorem sum_of_fourth_powers_correct (x : ℤ) (h : x * (x+1) * (x+2) = 36 * x + 12) : 
  sum_of_fourth_powers x = 98 :=
sorry

end sum_of_fourth_powers_correct_l549_549513


namespace compute_value_l549_549457

variable (p q : ℚ)
variable (h : ∀ x, 3 * x^2 - 7 * x - 6 = 0 → x = p ∨ x = q)

theorem compute_value (h_pq : p ≠ q) : (5 * p^3 - 5 * q^3) * (p - q)⁻¹ = 335 / 9 := by
  -- We assume p and q are the roots of the polynomial and p ≠ q.
  have sum_roots : p + q = 7 / 3 := sorry
  have prod_roots : p * q = -2 := sorry
  -- Additional steps to derive the required result (proof) are ignored here.
  sorry

end compute_value_l549_549457


namespace h_at_neg_one_l549_549097

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := 3 * x + 6
def g (x : ℝ) : ℝ := x ^ 3
def h (x : ℝ) : ℝ := f (g x)

-- The main statement to prove
theorem h_at_neg_one : h (-1) = 3 := by
  sorry

end h_at_neg_one_l549_549097


namespace max_elements_A_l549_549447

def M : Set ℕ := {n | n ≥ 1 ∧ n ≤ 1995}

def A (A_set : Set ℕ) : Prop :=
  A_set ⊆ M ∧ ∀ x ∈ A_set, 15 * x ∉ A_set

theorem max_elements_A : ∃ A_set : Set ℕ, A A_set ∧ (A_set.to_finset.card = 1870) :=
sorry

end max_elements_A_l549_549447


namespace product_of_gcd_and_lcm_1440_l549_549316

def product_of_gcd_and_lcm_of_24_and_60 : ℕ :=
  Nat.gcd 24 60 * Nat.lcm 24 60 

/-- The product of the gcd and lcm of 24 and 60 is 1440. -/
theorem product_of_gcd_and_lcm_1440 : product_of_gcd_and_lcm_of_24_and_60 = 1440 := 
by
  sorry

end product_of_gcd_and_lcm_1440_l549_549316


namespace questionnaires_to_send_l549_549881

theorem questionnaires_to_send 
  (rate_A rate_B rate_C : ℝ)
  (req_A req_B req_C : ℕ)
  (rate_A_pos : 0 < rate_A) (rate_A_le1 : rate_A ≤ 1)
  (rate_B_pos : 0 < rate_B) (rate_B_le1 : rate_B ≤ 1)
  (rate_C_pos : 0 < rate_C) (rate_C_le1 : rate_C ≤ 1) :
  (ceil (req_A / rate_A) = 231) ∧ 
  (ceil (req_B / rate_B) = 143) ∧ 
  (ceil (req_C / rate_C) = 59) ∧ 
  (ceil (req_A / rate_A) + ceil (req_B / rate_B) + ceil (req_C / rate_C) = 433) :=
by
  sorry

end questionnaires_to_send_l549_549881


namespace marble_counts_l549_549284

variables (Ed_origin Doug Charlie: ℕ)

-- Conditions
def Ed_initial_more : Prop := Ed_origin = Doug + 12
def Ed_current : Prop := Ed_origin - 20 = 17
def Charlie_marble_count : Prop := Charlie = 4 * Doug

-- Correct Answers
def Ed_correct_initial : Prop := Ed_origin = 37
def Doug_correct_count : Prop := Doug = 25
def Charlie_correct_count : Prop := Charlie = 100

theorem marble_counts :
  Ed_initial_more →
  Ed_current →
  Charlie_marble_count →
  Ed_correct_initial ∧ Doug_correct_count ∧ Charlie_correct_count := by
  intros
  sorry

end marble_counts_l549_549284


namespace power_function_expression_l549_549731

theorem power_function_expression (a : ℝ) :
  (∀ x : ℝ, f (x : ℝ) = x^a) → (f 2 = 8) → a = 3 :=
by 
  intros hyp1 hyp2 
  sorry

end power_function_expression_l549_549731


namespace perfect_squares_count_200_600_l549_549762

-- Define the range and the set of perfect squares
def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k * k

def perfect_squares_between (a b : ℕ) : Finset ℕ :=
  Finset.filter (λ n, is_perfect_square n) (Finset.Icc a b)

-- The main theorem
theorem perfect_squares_count_200_600 : 
  perfect_squares_between 200 600).card = 10 :=
by
  sorry

end perfect_squares_count_200_600_l549_549762


namespace sum_remainders_mod_1000_l549_549438

-- Define the set R as all possible remainders of 2^n mod 1000
def R : Set ℕ := {r | ∃ n : ℕ, r = (2^n % 1000)}

-- Define S as the sum of the elements in set R
def S : ℕ := Finset.sum (Finset.filter (λ x => x ∈ R) (Finset.range 1000)) id

-- Define the theorem stating that the remainder of S when divided by 1000 is 7
theorem sum_remainders_mod_1000 : S % 1000 = 7 := 
by {
  sorry
}

end sum_remainders_mod_1000_l549_549438


namespace weight_in_each_hand_l549_549245

variable (total_weight : ℕ)
variable (equal_weight_per_hand : total_weight % 2 = 0)

theorem weight_in_each_hand (h : total_weight = 14) : total_weight / 2 = 7 :=
by
  rw [h]
  norm_num
  sorry

end weight_in_each_hand_l549_549245


namespace imaginary_part_is_neg_3_l549_549901

noncomputable def complex_number := (3 + 4 * Complex.I) / Complex.I
def imaginary_part := Complex.im complex_number

theorem imaginary_part_is_neg_3 : imaginary_part = -3 :=
by
  sorry

end imaginary_part_is_neg_3_l549_549901


namespace number_is_seven_point_five_l549_549123

theorem number_is_seven_point_five (x : ℝ) (h : x^2 + 100 = (x - 20)^2) : x = 7.5 :=
by
  sorry

end number_is_seven_point_five_l549_549123


namespace probability_two_females_one_male_l549_549846

theorem probability_two_females_one_male
  (total_contestants : ℕ)
  (female_contestants : ℕ)
  (male_contestants : ℕ)
  (choose_count : ℕ)
  (total_combinations : ℕ)
  (female_combinations : ℕ)
  (male_combinations : ℕ)
  (favorable_outcomes : ℕ)
  (probability : ℚ)
  (h1 : total_contestants = 8)
  (h2 : female_contestants = 5)
  (h3 : male_contestants = 3)
  (h4 : choose_count = 3)
  (h5 : total_combinations = Nat.choose total_contestants choose_count)
  (h6 : female_combinations = Nat.choose female_contestants 2)
  (h7 : male_combinations = Nat.choose male_contestants 1)
  (h8 : favorable_outcomes = female_combinations * male_combinations)
  (h9 : probability = favorable_outcomes / total_combinations) :
  probability = 15 / 28 :=
by
  sorry

end probability_two_females_one_male_l549_549846


namespace missing_water_calculation_l549_549599

def max_capacity : ℝ := 350000
def loss_rate1 : ℝ := 32000
def time1 : ℝ := 5
def loss_rate2 : ℝ := 10000
def time2 : ℝ := 10
def fill_rate : ℝ := 40000
def fill_time : ℝ := 3

theorem missing_water_calculation :
  350000 - ((350000 - (32000 * 5 + 10000 * 10)) + 40000 * 3) = 140000 :=
by
  sorry

end missing_water_calculation_l549_549599


namespace find_income_l549_549508

-- Define the condition for savings
def savings_formula (income expenditure savings : ℝ) : Prop :=
  income - expenditure = savings

-- Define the ratio between income and expenditure
def ratio_condition (income expenditure : ℝ) : Prop :=
  income = 5 / 4 * expenditure

-- Given:
-- savings: Rs. 3400
-- We need to prove the income is Rs. 17000
theorem find_income (savings : ℝ) (income expenditure : ℝ) :
  savings_formula income expenditure savings →
  ratio_condition income expenditure →
  savings = 3400 →
  income = 17000 :=
sorry

end find_income_l549_549508


namespace permutations_of_three_l549_549388

-- No ties and 3 racers: Harry, Ron, and Neville
variable (racers : Finset String) (_ : racers = {"Harry", "Ron", "Neville"})

theorem permutations_of_three : racers.card = 3 → racers.toList.permutations.length = 6 :=
by
  intro h_card
  rw Finset.card_eq_iff at h_card
  subst h_card
  sorry

end permutations_of_three_l549_549388


namespace not_divisible_by_n_plus_4_l549_549479

theorem not_divisible_by_n_plus_4 (n : ℕ) (h : 0 < n) : ¬ (n + 4 ∣ n^2 + 8 * n + 15) := 
sorry

end not_divisible_by_n_plus_4_l549_549479


namespace fixed_point_of_line_l549_549373

theorem fixed_point_of_line (a : ℝ) : ∃ x y : ℝ, (x = -1 ∧ y = -1) ∧ (a * x + y + a + 1 = 0) :=
by
  use [-1, -1]
  split
  all_goals { sorry }

end fixed_point_of_line_l549_549373


namespace solve_for_x_l549_549874

theorem solve_for_x (x : ℝ) (h : 4^(2 * x + 3) = 1 / 16) : x = -5 / 2 :=
sorry

end solve_for_x_l549_549874


namespace area_quadrilateral_range_l549_549053

-- Define given conditions
variables (A B C D : Type) [normed_group A] [normed_group B]
variables [normed_group C] [normed_group D]

theorem area_quadrilateral_range 
    (AB BC AC : ℝ)
    (h1 : AB = 1)
    (h2 : BC = real.sqrt 3)
    (is_isosceles_right_ACD : ∀ (d : D), ∃ (α : ℝ) (ϕ : ℝ), tan ϕ = 2 ∧ 0 < α ∧ α < π)
  : ∃ S : real, S = (sqrt 15 / 2) * sin (α + ϕ) + 2 ∧ (2 - sqrt 15 / 2 < S ∧ S ≤ 2 + sqrt 15 / 2) := 
sorry

end area_quadrilateral_range_l549_549053


namespace house_ordering_count_l549_549864

def house_color : Type := Fin 5 -- Represents 5 unique house colors

variable (P T G B W : house_color) -- Assume P, T, G, B, W represent different house colors

-- Define the conditions as hypotheses
def is_valid_ordering (ordering : List house_color) : Prop :=
  (ordering.indexOf P < ordering.indexOf T) ∧ -- Purple before Teal
  (ordering.indexOf G < ordering.indexOf B) ∧ -- Green before Black
  (abs (ordering.indexOf T - ordering.indexOf B) ≠ 1) ∧ -- Teal not adjacent to Black
  (abs (ordering.indexOf G - ordering.indexOf B) ≠ 1) -- Green not adjacent to Black

-- Each house must occupy a unique position and thus exactly 5 permutations.
def perm_count : Nat := 5.fact

-- The total valid orderings giving the conditions specified
def count_valid_orderings : Nat :=
  List.filter is_valid_ordering (List.permutations [P, T, G, B, W]).length

theorem house_ordering_count : count_valid_orderings P T G B W = 3 := by
  sorry

end house_ordering_count_l549_549864


namespace smallest_palindrome_divisible_by_3_l549_549948

def is_palindrome (n : ℕ) : Prop :=
  let str := n.toDigits 10
  str = str.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  n.toDigits 10 |>.sum

theorem smallest_palindrome_divisible_by_3 : ∃ n, is_palindrome n ∧ n ≥ 1000 ∧ n < 10000 ∧ n % 3 = 0 ∧ (∀ m, is_palindrome m ∧ m ≥ 1000 ∧ m < 10000 ∧ m % 3 = 0 → n ≤ m) ∧ n = 1221 :=
  sorry

end smallest_palindrome_divisible_by_3_l549_549948


namespace probability_two_tails_three_coins_probability_two_tails_l549_549970

open ProbabilityTheory

theorem probability_two_tails_three_coins :
  probability (λ ω, count_tails(ω) = 2) = 3 / 8 :=
sorry

variables {Ω : Type*} [Fintype Ω]
          {p : Π (ω : Ω), ℝ} [π_fintype : Fintype (Π (ω : Ω), ℝ)]
          (coin_prob : probability_space (Ω))
          (h_fair : ∀ ω, coin_prob.event (ω = heads) = 0.5)

def toss_three_coins : Event Ω :=  
{finsupp: 3, values: [heads, tails] booleans}

def count_tails : Π (ω : Ω), ℕ 
| (ω.head) := if ω = tails then 1 else 0
| (ω.tail) := if ω = tails then 1 else count_tails ω

theorem probability_two_tails (h : 3 fair coins are tossed) :
  P {ω | count_tails ω = 2} = 3/8 :=
begin
  sorry
end

end probability_two_tails_three_coins_probability_two_tails_l549_549970


namespace distinct_values_of_fx_l549_549439

theorem distinct_values_of_fx :
  let f (x : ℝ) := ⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋ + ⌊4 * x⌋
  ∃ (s : Finset ℤ), (∀ x, 0 ≤ x ∧ x ≤ 10 → f x ∈ s) ∧ s.card = 61 :=
by
  sorry

end distinct_values_of_fx_l549_549439


namespace exponential_function_value_at_3_l549_549507

-- Define the exponential function and the condition it passes through a specific point
def f (x : ℝ) : ℝ := 2^x

-- State the main theorem to be proved, which follows from the given conditions
theorem exponential_function_value_at_3 : f(3) = 8 := by
  sorry

end exponential_function_value_at_3_l549_549507


namespace area_of_right_triangle_l549_549998

-- Define the conditions
def leg1 : ℝ := 6
def leg2 : ℝ := 8
def hypotenuse : ℝ := Real.sqrt (leg1^2 + leg2^2)
def median_to_hypotenuse : ℝ := hypotenuse / 2

-- The main statement to prove the area of the triangle
theorem area_of_right_triangle : median_to_hypotenuse = 5 → (1 / 2) * leg1 * leg2 = 24 :=
by
  -- Skip the proof part
  intros
  sorry

end area_of_right_triangle_l549_549998


namespace extreme_value_of_ratio_l549_549568

noncomputable def extremeRatio (x : Fin 5 → ℝ) := 
  Real.toReal (max x) / Real.toReal (min x)

theorem extreme_value_of_ratio (a : ℝ) (x : Fin 5 → ℝ) 
  (ha : a > 25)
  (hx : ∑ i, x i * ∑ i, 1 / (x i) = a)
  (hx_pos : ∀ i, 0 < x i) :
  extremeRatio x ∈
  set.Icc
    (Real.toReal ((Real.toReal ((Real.sqrt (a - 1) + Real.sqrt (a - 25)) / (2 * Real.sqrt 6)) ^ 2)))
    (Real.toReal ((Real.toReal ((Real.sqrt a - 3 + Real.sqrt (a + 5 - 6 * Real.sqrt a)) / 2)) ^ 2)) :=
sorry

end extreme_value_of_ratio_l549_549568


namespace separation_possible_l549_549272

-- Define the set S of 4031 points
constant S : Finset (ℝ × ℝ)
-- S has 4031 points
axiom S_card : S.card = 4031
-- No three points in S are collinear
axiom S_no_three_collinear : ∀ (p1 p2 p3 : (ℝ × ℝ)), p1 ∈ S → p2 ∈ S → p3 ∈ S → ¬ collinear {p1, p2, p3}
-- 2015 points in S are colored blue
constant is_blue : (ℝ × ℝ) → Prop
axiom blue_card : (S.filter is_blue).card = 2015
-- 2016 points in S are colored red
constant is_red : (ℝ × ℝ) → Prop
axiom red_card : (S.filter is_red).card = 2016
-- No point is both blue and red
axiom no_blue_red_overlap : ∀ p, is_blue p → ¬ is_red p

-- Statement to prove
theorem separation_possible : 
  ∃ L : Finset (ℝ × ℝ × ℝ), -- Set of lines
  L.card = 2015 ∧
  (∀ l ∈ L, ∀ p ∈ S, ¬ (passes_through l p)) ∧ -- No line passes through any point of S
  (∀ region, -- For each region formed
    (∀ p1 p2, p1 ∈ S → p2 ∈ S → p1 ≠ p2 → in_same_region region p1 p2 → 
    ((is_blue p1 ∧ is_blue p2) ∨ (is_red p1 ∧ is_red p2)))) -- No region contains points of different colors
  :=
sorry

end separation_possible_l549_549272


namespace polynomial_transformation_b_d_sum_l549_549648

theorem polynomial_transformation_b_d_sum :
  let P : Polynomial ℂ := Polynomial.Coeff (Polynomial.X ^ 4 + 6 * Polynomial.X ^ 3 + 8 * Polynomial.X ^ 2 + 12 * Polynomial.X + 9)
  let z := complex.roots P
  let f := λ z : ℂ, -2 * complex.I * complex.conj z
  let transformed_roots : List ℂ := z.map f
  let Q : Polynomial ℂ := Polynomial.CListRoot transformed_roots
  let B : ℂ := Q.coeff 2
  let D : ℂ := Q.coeff 0
  B + D = 176 := sorry

end polynomial_transformation_b_d_sum_l549_549648


namespace circle_inscribed_in_ellipse_condition_coordinates_of_tangency_area_enclosed_by_c1_c2_l549_549817

-- The condition for which the circle \( C_1 \) is inscribed in the ellipse \( C_2 \)
theorem circle_inscribed_in_ellipse_condition (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^2 = b^2 * (1 + b^2) :=
sorry

-- Given \( b = \frac{1}{\sqrt{3}}, a = \frac{2}{3} \), coordinates of point of tangency in the first quadrant
theorem coordinates_of_tangency (b : ℝ) (hb : b = 1/real.sqrt 3) :
  ∃ p q : ℝ, (a = 2/3) → (p = 1/2) ∧ (q = 1/2) :=
sorry

-- Under the condition in (1), find the area of the part enclosed by \( C_1, C_2 \) for \( x \geq p \)
theorem area_enclosed_by_c1_c2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 = b^2 * (1 + b^2)) :
  ∫ x in (a / (1 + b^2)) to (2 * a), (b * real.sqrt(1 - (x^2 / a^2)) - real.sqrt(a^2 - ((x - a)^2))) dx :=
sorry

end circle_inscribed_in_ellipse_condition_coordinates_of_tangency_area_enclosed_by_c1_c2_l549_549817


namespace area_5f_shifted_l549_549732

variable {f : ℝ → ℝ}

-- Given condition: The area under y = f(x) and above the x-axis is 15 square units.
axiom area_f : ∫ x in -∞..∞, f x = 15

-- We need to prove that the area under y = 5 * f(x - 4) and above the x-axis is 75 square units.
theorem area_5f_shifted :
  ∫ x in -∞..∞, 5 * f (x - 4) = 75 :=
by
  -- proof required here
  sorry

end area_5f_shifted_l549_549732


namespace apples_total_l549_549692

-- Definitions as per conditions
def apples_on_tree : Nat := 5
def initial_apples_on_ground : Nat := 8
def apples_eaten_by_dog : Nat := 3

-- Calculate apples left on the ground
def apples_left_on_ground : Nat := initial_apples_on_ground - apples_eaten_by_dog

-- Calculate total apples left
def total_apples_left : Nat := apples_on_tree + apples_left_on_ground

theorem apples_total : total_apples_left = 10 := by
  -- the proof will go here
  sorry

end apples_total_l549_549692


namespace find_k_l549_549026

theorem find_k (k x y : ℝ) (h1 : x = 2) (h2 : y = -3)
    (h3 : 2 * x^2 + k * x * y = 4) : k = 2 / 3 :=
by
  sorry

end find_k_l549_549026


namespace complex_power_proof_l549_549583

noncomputable def complex_power : Prop :=
  (1 - complex.i)^4 = -4

theorem complex_power_proof : complex_power :=
by
  sorry

end complex_power_proof_l549_549583


namespace blue_higher_than_yellow_l549_549535

theorem blue_higher_than_yellow :
  let Event := {k : ℕ // k > 0}
  let P : Event → ℝ := λ k, real.exp(-k * log 3)
  (∑' (k : ℕ), P ⟨k, Nat.succ_pos k⟩) = 1 :=
  -- probability distribution should sum to 1 to be valid
  by sorry

  let same_bin_prob := ∑' (k : ℕ), P ⟨k, Nat.succ_pos k⟩^2
  -- The probability of both balls landing in the same bin
  same_bin_prob = (1 / 8) :=
  -- Corresponding to calculation from geometric series sum
  by sorry

  let diff_bin_prob := 1 - same_bin_prob
  -- The probability of balls landing in different bins
  diff_bin_prob = (7 / 8) :=
  -- Calculating difference
  by sorry

  let final_prob := diff_bin_prob / 2
  -- Using symmetry to find the final probability for the blue ball to land in a higher bin
  final_prob = (7 / 16) :=
  (by sorry)

end blue_higher_than_yellow_l549_549535


namespace problem_one_problem_two_l549_549493

-- Problem 1
theorem problem_one : 
  3 * (27 / 8)^(-1 / 3) - Real.sqrt((π - 1)^2) = 3 - π := by
  sorry

-- Problem 2
theorem problem_two {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  ((a^(1/2) * Real.sqrt(a / b))^(1/3)) = a^(1/3) * b^(-1/6) := by
  sorry

end problem_one_problem_two_l549_549493


namespace solution_set_xf_x_neg_l549_549393

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

noncomputable def is_increasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

def cond1 (f : ℝ → ℝ) : Prop := is_odd f
def cond2 (f : ℝ → ℝ) : Prop := is_increasing_on f (Iio 0)
def cond3 (f : ℝ → ℝ) : Prop := f (-2) = 0

theorem solution_set_xf_x_neg (f : ℝ → ℝ) (h1 : cond1 f) (h2 : cond2 f) (h3 : cond3 f) :
  {x : ℝ | x * f x < 0} = set.Ioo (-2) 0 ∪ set.Ioo 0 2 :=
begin
  sorry -- proof goes here
end

end solution_set_xf_x_neg_l549_549393


namespace ratio_adults_children_l549_549259

-- Definitions based on conditions
def children := 45
def total_adults (A : ℕ) : Prop := (2 / 3 : ℚ) * A = 10

-- The theorem stating the problem
theorem ratio_adults_children :
  ∃ A, total_adults A ∧ (A : ℚ) / children = (1 / 3 : ℚ) :=
by {
  sorry
}

end ratio_adults_children_l549_549259


namespace log_incorrect_statement_l549_549337

theorem log_incorrect_statement (b x : ℝ) (h : b > 1) (hx1 hx2 : ℝ) 
  (hx1 : (sqrt b) < x) (hx2 : x < b) :
  ¬(0.5 < Real.log b x ∧ Real.log b x < 1.5) :=
by sorry

end log_incorrect_statement_l549_549337


namespace pentagon_triangle_area_percentage_l549_549992

def is_equilateral_triangle (s : ℝ) (area : ℝ) : Prop :=
  area = (s^2 * Real.sqrt 3) / 4

def is_square (s : ℝ) (area : ℝ) : Prop :=
  area = s^2

def pentagon_area (square_area triangle_area : ℝ) : ℝ :=
  square_area + triangle_area

noncomputable def percentage (triangle_area pentagon_area : ℝ) : ℝ :=
  (triangle_area / pentagon_area) * 100

theorem pentagon_triangle_area_percentage (s : ℝ) (h₁ : s > 0) :
  let square_area := s^2
  let triangle_area := (s^2 * Real.sqrt 3) / 4
  let pentagon_total_area := pentagon_area square_area triangle_area
  let triangle_percentage := percentage triangle_area pentagon_total_area
  triangle_percentage = (100 * (4 * Real.sqrt 3 - 3) / 13) :=
by
  sorry

end pentagon_triangle_area_percentage_l549_549992


namespace compute_expression_l549_549270

def sum_of_squares := 7^2 + 5^2
def square_of_sum := (7 + 5)^2
def sum_of_both := sum_of_squares + square_of_sum
def final_result := 2 * sum_of_both

theorem compute_expression : final_result = 436 := by
  sorry

end compute_expression_l549_549270


namespace volume_of_sand_pile_l549_549600

noncomputable def diameter : ℝ := 10
noncomputable def height : ℝ := 0.6 * diameter
noncomputable def radius : ℝ := diameter / 2
noncomputable def volume : ℝ := (π * radius^2 * height) / 3

theorem volume_of_sand_pile : volume = 50 * π := by
  sorry

end volume_of_sand_pile_l549_549600


namespace point_P_coordinates_l549_549020

noncomputable def point_coordinates (x : ℝ) : ℝ × ℝ × ℝ :=
  (x, 0, 0)

theorem point_P_coordinates :
  ∃ (x : ℝ), point_coordinates x = (3, 0, 0) ∧
  let A : ℝ × ℝ × ℝ := (1, 2, 1),
      B : ℝ × ℝ × ℝ := (2, 2, 2),
      PA_squared := (x - 1)^2 + (0 - 2)^2 + (0 - 1)^2,
      PB_squared := (x - 2)^2 + (0 - 2)^2 + (0 - 2)^2 in
  PA_squared = PB_squared :=
begin
  -- Proof is not required as per the instruction
  sorry
end

end point_P_coordinates_l549_549020


namespace couple_continent_gender_difference_l549_549525

theorem couple_continent_gender_difference :
  let Gaga_males   := 204
      Gaga_females := 468
      Nana_males   := 334
      Nana_females := 516
      Dada_males   := 427
      Dada_females := 458
      Lala_males   := 549
      Lala_females := 239
      total_males  := Gaga_males + Nana_males + Dada_males + Lala_males
      total_females := Gaga_females + Nana_females + Dada_females + Lala_females
  in total_females - total_males = 167 :=
by
  let Gaga_males := 204
  let Gaga_females := 468
  let Nana_males := 334
  let Nana_females := 516
  let Dada_males := 427
  let Dada_females := 458
  let Lala_males := 549
  let Lala_females := 239
  let total_males := Gaga_males + Nana_males + Dada_males + Lala_males
  let total_females := Gaga_females + Nana_females + Dada_females + Lala_females
  have h1 : total_males = 204 + 334 + 427 + 549 := rfl
  have h2 : total_females = 468 + 516 + 458 + 239 := rfl
  sorry

end couple_continent_gender_difference_l549_549525


namespace exists_infinitely_many_l549_549719

def seq (F : ℕ → ℤ) (a b : ℤ) : Prop :=
  (F 1 = a) ∧ 
  (F 2 = b) ∧
  ∀ n, F (n + 2) = F (n + 1) + F n

def property_P (a b : ℤ) (m : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 →
  ¬ isPerfectSquare (1 + m * F k * F (k + 2))

theorem exists_infinitely_many (a b : ℤ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a < b) 
  (F : ℕ → ℤ) (hF : seq F a b) : 
  ∃∞ m : ℕ, property_P a b m :=
sorry

end exists_infinitely_many_l549_549719


namespace tangerine_count_l549_549524

def initial_tangerines : ℕ := 10
def added_tangerines : ℕ := 6

theorem tangerine_count : initial_tangerines + added_tangerines = 16 :=
by
  sorry

end tangerine_count_l549_549524


namespace fliers_left_l549_549205

theorem fliers_left (initial_fliers : ℕ) (fraction_morning : ℕ) (fraction_afternoon : ℕ) :
  initial_fliers = 2000 → 
  fraction_morning = 1 / 10 → 
  fraction_afternoon = 1 / 4 → 
  (initial_fliers - initial_fliers * fraction_morning - 
  (initial_fliers - initial_fliers * fraction_morning) * fraction_afternoon) = 1350 := by
  intros initial_fliers_eq fraction_morning_eq fraction_afternoon_eq
  sorry

end fliers_left_l549_549205


namespace find_k_unique_solution_l549_549658

theorem find_k_unique_solution (k : ℝ) (h: k ≠ 0) : (∀ x : ℝ, (x + 3) / (k * x - 2) = x → k = -3/4) :=
sorry

end find_k_unique_solution_l549_549658


namespace emily_subtracts_99_l549_549532

theorem emily_subtracts_99 (a b : ℕ) : (a = 50) → (b = 1) → (49^2 = 50^2 - 99) :=
by
  sorry

end emily_subtracts_99_l549_549532


namespace problem_statement_l549_549344

noncomputable def f : ℝ → ℝ := sorry

variable (α : ℝ)

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 3) = -f x
axiom tan_alpha : Real.tan α = 2

theorem problem_statement : f (15 * Real.sin α * Real.cos α) = 0 := 
by {
  sorry
}

end problem_statement_l549_549344


namespace proof_problem_l549_549376

open Real

noncomputable def set_A : Set ℝ :=
  {x | x = tan (-19 * π / 6) ∨ x = sin (-19 * π / 6)}

noncomputable def set_B : Set ℝ :=
  {m | 0 <= m ∧ m <= 4}

noncomputable def set_C (a : ℝ) : Set ℝ :=
  {x | a + 1 < x ∧ x < 2 * a}

theorem proof_problem (a : ℝ) :
  set_A = {-sqrt 3 / 3, -1 / 2} ∧
  set_B = {m | 0 <= m ∧ m <= 4} ∧
  (set_A ∪ set_B) = {-sqrt 3 / 3, -1 / 2, 0, 4} →
  (∀ a, set_C a ⊆ (set_A ∪ set_B) → 1 < a ∧ a < 2) :=
sorry

end proof_problem_l549_549376


namespace magician_trick_success_l549_549986

theorem magician_trick_success {n : ℕ} (T_pos : ℕ) (deck_size : ℕ := 52) (discard_count : ℕ := 51):
  (T_pos = 1 ∨ T_pos = deck_size) → ∃ strategy : Type, ∀ spectator_choice : ℕ, (spectator_choice ≤ deck_size) → 
                          ((T_pos = 1 → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)
                          ∧ (T_pos = deck_size → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)) :=
sorry

end magician_trick_success_l549_549986


namespace misha_grade_students_l549_549466

theorem misha_grade_students (n : ℕ) (h1 : n = 75) (h2 : n = 75) : 2 * n - 1 = 149 := 
by
  sorry

end misha_grade_students_l549_549466


namespace least_number_added_1789_l549_549941

def least_number_added_to_divisible (n d : ℕ) : ℕ := d - (n % d)

theorem least_number_added_1789 :
  least_number_added_to_divisible 1789 (Nat.lcm (Nat.lcm 5 6) (Nat.lcm 4 3)) = 11 :=
by
  -- Step definitions
  have lcm_5_6 := Nat.lcm 5 6
  have lcm_4_3 := Nat.lcm 4 3
  have lcm_total := Nat.lcm lcm_5_6 lcm_4_3
  -- Computation of the final result
  have remainder := 1789 % lcm_total
  have required_add := lcm_total - remainder
  -- Conclusion based on the computed values
  sorry

end least_number_added_1789_l549_549941


namespace mean_of_five_numbers_l549_549163

theorem mean_of_five_numbers (sum : ℚ) (h : sum = 3 / 4) : (sum / 5 = 3 / 20) :=
by
  -- Proof omitted
  sorry

end mean_of_five_numbers_l549_549163


namespace street_trees_one_side_number_of_street_trees_l549_549957

-- Conditions
def road_length : ℕ := 2575
def interval : ℕ := 25
def trees_at_endpoints : ℕ := 2

-- Question: number of street trees on one side of the road
theorem street_trees_one_side (road_length interval : ℕ) (trees_at_endpoints : ℕ) : ℕ :=
  (road_length / interval) + 1

-- Proof of the provided problem
theorem number_of_street_trees : street_trees_one_side road_length interval trees_at_endpoints = 104 :=
by
  sorry

end street_trees_one_side_number_of_street_trees_l549_549957


namespace payment_is_variable_l549_549151

variable (x y : ℕ)

def price_of_pen : ℕ := 3

theorem payment_is_variable (x y : ℕ) (h : y = price_of_pen * x) : 
  (price_of_pen = 3) ∧ (∃ n : ℕ, y = 3 * n) :=
by 
  sorry

end payment_is_variable_l549_549151


namespace even_composition_of_even_and_odd_l549_549098

variables {ℝ : Type*} [linear_order ℝ]

-- Definitions for even and odd functions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

-- Main statement: if f is even and g is odd, then f ∘ g is even
theorem even_composition_of_even_and_odd (f g : ℝ → ℝ) 
  (hf : even_function f) (hg : odd_function g) : 
  even_function (f ∘ g) :=
by
  sorry

end even_composition_of_even_and_odd_l549_549098


namespace area_of_shaded_region_l549_549608

section
  variable (s : ℝ := 8) (r : ℝ := 4) (θ : ℝ := π / 3) -- π / 3 radians is 60 degrees

  def area_hexagon (s : ℝ) : ℝ := 6 * (sqrt 3 / 4 * s^2)

  def area_sector (r : ℝ) (θ : ℝ) : ℝ := 6 * (θ / (2 * π) * π * r^2)

  theorem area_of_shaded_region : area_hexagon s - area_sector r θ = 96 * sqrt 3 - 16 * π := by
    sorry
end

end area_of_shaded_region_l549_549608


namespace solve_length_of_faster_train_l549_549541

noncomputable def kmph_to_mps (speed_kmph : Float) : Float :=
  speed_kmph * (1000 / 3600)

theorem solve_length_of_faster_train
  (V1_kmph : Float) (V2_kmph : Float)
  (L1 : Float) (t : Float)
  (V1_kmph = 100) (V2_kmph = 120)
  (L1 = 500) (t = 19.6347928529354)
  : ∃ L2 : Float, L2 = 700 :=
by
  let V_rel := kmph_to_mps (V1_kmph + V2_kmph)
  have h_rel_speed : V_rel = 61.1111111111111 := by sorry
  let total_length := (L1 + 700) 
  have h_total_length : total_length = t * V_rel := by sorry
  use 700
  sorry

end solve_length_of_faster_train_l549_549541


namespace no_solution_system_iff_n_eq_neg_cbrt_four_l549_549023

variable (n : ℝ)

theorem no_solution_system_iff_n_eq_neg_cbrt_four :
    (∀ x y z : ℝ, ¬ (2 * n * x + 3 * y = 2 ∧ 3 * n * y + 4 * z = 3 ∧ 4 * x + 2 * n * z = 4)) ↔
    n = - (4 : ℝ)^(1/3) := 
by
  sorry

end no_solution_system_iff_n_eq_neg_cbrt_four_l549_549023


namespace sqrt_sum_expression_l549_549281

theorem sqrt_sum_expression : (sqrt (16 - 8 * sqrt 3) + sqrt (16 + 8 * sqrt 3)) = 8 :=
by sorry

end sqrt_sum_expression_l549_549281


namespace intersection_locus_is_vertical_line_l549_549338

/-- 
Given \( 0 < a < b \), lines \( l \) and \( m \) are drawn through the points \( A(a, 0) \) and \( B(b, 0) \), 
respectively, such that these lines intersect the parabola \( y^2 = x \) at four distinct points 
and these four points are concyclic. 

We want to prove that the locus of the intersection point \( P \) of lines \( l \) and \( m \) 
is the vertical line \( x = \frac{a + b}{2} \).
-/
theorem intersection_locus_is_vertical_line (a b : ℝ) (h : 0 < a ∧ a < b) :
  (∃ P : ℝ × ℝ, P.fst = (a + b) / 2) := 
sorry

end intersection_locus_is_vertical_line_l549_549338


namespace tangent_circle_line_pairs_l549_549292

theorem tangent_circle_line_pairs (a b : ℤ) : 
  (a = 4 + 8 / (b - 4)) ∧ ((a * (b - 4) - b^2)^2 - (a^2 + b^2) * (b - 4)^2 = 0) 
  → b - 4 ∈ {1, -1, 2, -2, 4, -4, 8, -8} 
  →
  (a, b) ∈ {(12, 5), (-4, 3), (8, 6), (0, 2), (6, 8), (2, 0), (5, 12), (3, -4)} :=
begin
  sorry
end

end tangent_circle_line_pairs_l549_549292


namespace angle_between_a_b_is_135_deg_l549_549035

open Real

noncomputable def vector := (ℝ × ℝ)

def a : vector := (1, 2)
def b : vector := (2, -1) - a -- Calculate b from given condition

def dot_prod (u v : vector) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : vector) : ℝ := sqrt (u.1 * u.1 + u.2 * u.2)

def angle_between (u v : vector) : ℝ := 
  Real.arccos (dot_prod u v / (magnitude u * magnitude v))

theorem angle_between_a_b_is_135_deg (a b : vector) 
  (h1 : a = (1,2)) (h2 : a + b = (2, -1)) : 
  angle_between a b = π * 3 / 4 :=
by {
  sorry,
}

end angle_between_a_b_is_135_deg_l549_549035


namespace probability_escher_consecutive_l549_549475

def total_pieces : Nat := 12
def escher_pieces : Nat := 4

theorem probability_escher_consecutive :
  (Nat.factorial 9 * Nat.factorial 4 : ℚ) / Nat.factorial 12 = 1 / 55 := 
sorry

end probability_escher_consecutive_l549_549475


namespace units_digit_sum_base8_l549_549264

theorem units_digit_sum_base8 : 
  ∀ (x y : ℕ), (x = 64 ∧ y = 34 ∧ (x % 8 = 4) ∧ (y % 8 = 4) → (x + y) % 8 = 0) :=
by
  sorry

end units_digit_sum_base8_l549_549264


namespace sqrt_15_minus_1_range_l549_549286

theorem sqrt_15_minus_1_range (h : 9 < 15 ∧ 15 < 16) : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := 
  sorry

end sqrt_15_minus_1_range_l549_549286


namespace total_coins_l549_549634

theorem total_coins (x y : ℕ) (h : x ≠ y) (h1 : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := by
  sorry

end total_coins_l549_549634


namespace log_inequality_express_log_l549_549588

theorem log_inequality (a : ℝ) (h : a > 1) : log a (3 / 4) < 1 := sorry

theorem express_log (a : ℝ) (h : a = log 3 2) : log 3 8 - 2 * log 3 6 = a - 2 := sorry

end log_inequality_express_log_l549_549588


namespace hyperbola_eccentricity_l549_549726

theorem hyperbola_eccentricity (b c : ℝ) (hb : b = 2) (hc : c = 6) : 
  let a := real.sqrt (c^2 - b^2) in
  a ≠ 0 → (c / a = 3 * real.sqrt 2 / 4) := 
by
  intros; sorry

end hyperbola_eccentricity_l549_549726


namespace find_a_l549_549392

theorem find_a (a : ℝ) (h : ∃ b : ℝ, (4:ℝ)*x^2 - (12:ℝ)*x + a = (2*x + b)^2) : a = 9 :=
sorry

end find_a_l549_549392


namespace equivalent_operation_l549_549198

theorem equivalent_operation (x : ℚ) : (x * (2 / 5)) / (4 / 7) = x * (7 / 10) :=
by
  sorry

end equivalent_operation_l549_549198


namespace smallest_even_n_sum_eq_l549_549446
  
theorem smallest_even_n_sum_eq (n : ℕ) (h_pos : n > 0) (h_even : n % 2 = 0) :
  n = 12 ↔ 
  let s₁ := n / 2 * (2 * 5 + (n - 1) * 6)
  let s₂ := n / 2 * (2 * 13 + (n - 1) * 3)
  s₁ = s₂ :=
by
  sorry

end smallest_even_n_sum_eq_l549_549446


namespace S_100_eq_1300_l549_549418

noncomputable def sequence (n : ℕ) : ℕ := 
  if h : n = 0 then 1
  else if h : n = 1 then 1
  else let a1 := sequence (n-1);
       let a2 := sequence (n-2);
       2 - (-1)^(n-2) * a2

def S (n : ℕ) : ℕ := 
  ∑ i in finset.range (n+1), sequence i

theorem S_100_eq_1300 : S 100 = 1300 :=
by
  -- Proof goes here
  sorry

end S_100_eq_1300_l549_549418


namespace solve_pancakes_l549_549258

theorem solve_pancakes :
  ∃ (x : ℕ), (∀ (num_customers_big_stack num_pancakes_big_stack num_pancakes_each_big_stack num_pancakes_each_short_stack: ℕ), 
                num_customers_big_stack = 6 → 
                num_pancakes_big_stack = 5 →
                num_pancakes_each_short_stack = 3 →
                num_pancakes_each_big_stack = num_customers_big_stack * num_pancakes_big_stack →
                3 * x + num_pancakes_each_big_stack = 57) → x = 9 :=
begin
  use 9,
  intros _ num_customers_big_stack num_pancakes_big_stack num_pancakes_each_big_stack num_pancakes_each_short_stack _ h_num_pancakes_big_stack h_num_pancakes_each_short_stack,
  rw [h_num_pancakes_big_stack, h_num_pancakes_each_short_stack] at *,
  dsimp at *,
  linarith,
end

end solve_pancakes_l549_549258


namespace gold_bars_left_after_events_l549_549425

def initial_gold : ℕ := 60
def tax_rate : ℝ := 0.10

def gold_after_tax (initial : ℕ) (rate : ℝ) : ℕ :=
  initial - (rate * initial).to_nat

def gold_after_divorce (remaining : ℕ) : ℕ :=
  remaining / 2

theorem gold_bars_left_after_events :
  gold_after_divorce (gold_after_tax initial_gold tax_rate) = 27 := by
  sorry

end gold_bars_left_after_events_l549_549425


namespace correct_statement_is_B_l549_549954

-- Define the propositions
def statement_A : Prop :=
  ∀ polyhedron, (polyhedron.is_enclosed_by 5 planes) → 
  (polyhedron.is_pyramid_with_quadrilateral_base)

def statement_B : Prop :=
  ∃ pyramid, pyramid.has_altitude_outside_solid

def statement_C : Prop :=
  ∀ hexahedron, (hexahedron.has_only_one_pair_of_parallel_faces) → 
  (hexahedron.is_frustum)

def statement_D : Prop :=
  ∀ solid, (solid.has_one_face_polygon_and_rest_triangles) → 
  (solid.is_pyramid)

-- The actual proof problem
theorem correct_statement_is_B : statement_B :=
  by sorry

end correct_statement_is_B_l549_549954


namespace bob_smallest_number_l549_549247

theorem bob_smallest_number :
  let alice_number := 30
  let bob_prime_factors := {2, 3, 5}
  (∀ p ∈ bob_prime_factors, p ∣ alice_number) →
  ∃ q, q ∉ bob_prime_factors →
  by ∃ bob_number,
    (∀ p ∈ bob_prime_factors, p ∣ bob_number) ∧
    q ∣ bob_number ∧
    bob_number = 210
  sorry

end bob_smallest_number_l549_549247


namespace magnitude_z_l549_549776

noncomputable def z : ℂ := (2 - complex.i) / (1 + 2 * complex.i)

theorem magnitude_z : complex.abs z = 1 :=
by
  sorry

end magnitude_z_l549_549776


namespace bacterium_radius_scientific_l549_549212

theorem bacterium_radius_scientific (r : ℝ) (h : r = 0.0000108) : r = 1.08 * 10^(-5) :=
by
  sorry

end bacterium_radius_scientific_l549_549212


namespace equal_angles_EF_AB_CD_l549_549225

variables {A B C D E F : Type*}
variables [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C] [vector_space ℝ D] [vector_space ℝ E] [vector_space ℝ F]

def is_midpoint (M X Y : Type*) [vector_space ℝ X] [vector_space ℝ Y] :=
  ∃ (s : ℝ), s = 1/2 ∧ M = s • X + (1-s) • Y

def is_convex_quadrilateral (A B C D : Type*) [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C] [vector_space ℝ D] :=
  true -- Placeholder for actual convex quadrilateral definition

def equal_side_lengths (A B C D: Type*) [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C] [vector_space ℝ D] :=
  ∃ (k : ℝ), A - B = C - D

theorem equal_angles_EF_AB_CD 
  (A B C D E F : Type*) [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C] [vector_space ℝ D] [vector_space ℝ E] [vector_space ℝ F]
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : equal_side_lengths A B C D)
  (h3 : is_midpoint E A C)
  (h4 : is_midpoint F B D)
  (h5 : E ≠ F) :
  ∃ (θ : ℝ), angle A E F = θ ∧ angle D F E = θ :=
sorry

end equal_angles_EF_AB_CD_l549_549225


namespace correct_statement_l549_549953

theorem correct_statement :
  (¬ (3 * (-3) = 1)) ∧
  (¬ (3 + (1 / 3) = 0)) ∧
  (0 + 0 = 0) ∧
  (¬ (|5| = -5)) ∧
  (∀ A B D, (¬ (3 * (-3) = 1)) ∧
            (¬ (3 + (1 / 3) = 0)) ∧
            (0 + 0 = 0) ∧
            (¬ (|5| = -5)) → false → (0 + 0 = 0) = true) :=
by
  sorry

end correct_statement_l549_549953


namespace max_value_x_plus_y_l549_549087

theorem max_value_x_plus_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 48) (hx_mult_4 : x % 4 = 0) : x + y ≤ 49 :=
sorry

end max_value_x_plus_y_l549_549087


namespace gcd_lcm_product_24_60_l549_549302

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l549_549302


namespace math_physics_books_selection_l549_549175

theorem math_physics_books_selection :
  ∀ (m p : ℕ), m = 3 → p = 2 → m * p = 6 :=
by
  intros m p hm hp
  rw [hm, hp]
  exact Nat.mul_comm 3 2
  exact Nat.mul_comm 2 3
  sorry

end math_physics_books_selection_l549_549175


namespace infinite_perfect_squares_l549_549452

theorem infinite_perfect_squares (n : ℕ) (h_n : 0 < n) :
  ∃∞ k : ℕ, ∃ m : ℕ, m * m = n * 2^k - 7 :=
by
  sorry

end infinite_perfect_squares_l549_549452


namespace simplify_fraction_l549_549870

theorem simplify_fraction (a b m n : ℕ) (h : a ≠ 0 ∧ b ≠ 0 ∧ m ≠ 0 ∧ n ≠ 0) : 
  (a^2 * b) / (m * n^2) / ((a * b) / (3 * m * n)) = 3 * a / n :=
by
  sorry

end simplify_fraction_l549_549870


namespace volume_of_box_inconsistent_l549_549217

def length_cm : ℝ := 4
def cm_to_inch (cm : ℝ) : ℝ := cm / 2.54
def base_length_inch : ℝ := cm_to_inch length_cm
def base_area_inch : ℝ := base_length_inch * base_length_inch
def height_condition (h : ℝ) : Prop := base_area_inch = h + 8

theorem volume_of_box_inconsistent :
  ∃ h : ℝ, height_condition h → false :=
by
  let base_length_inch := cm_to_inch 4
  let base_area_inch := base_length_inch * base_length_inch
  have h := base_area_inch - 8
  have h_neg : h < 0,
  { calc
      h = base_area_inch - 8 : rfl
       ... = ((4 / 2.54) ^ 2) - 8 : by norm_num
       ... < 0 : by norm_num },
  use h,
  intro hc,
  exact h_neg hc

end volume_of_box_inconsistent_l549_549217


namespace lines_intersect_at_a_single_point_l549_549332

-- Definitions based on given conditions
variables {α : Type*} [EuclideanGeometry α] 
variables (A B C P A1 B1 C1 : α) 
variables (PA PB PC : Line α)
variables (la lb lc : Line α)
variables [Nonempty α]

-- Given conditions
def perpendiculars_from_P_to_sides (P A1 B1 C1 : α) (PA PB PC : Line α) : Prop :=
  PA.perpendicular_on P A1 ∧ PB.perpendicular_on P B1 ∧ PC.perpendicular_on P C1

def line_la (A B1 C1 : α) (PA B1C1 : Segment α): Line α := 
  let mid_PA := PA.midpoint in
  let mid_B1C1 := B1C1.midpoint in
  Line.mk mid_PA mid_B1C1

def line_lb (B A1 C1 : α) (PB AC1 : Segment α): Line α := 
  let mid_PB := PB.midpoint in
  let mid_AC1 := AC1.midpoint in
  Line.mk mid_PB mid_AC1

def line_lc (C A1 B1 : α) (PC AB1 : Segment α): Line α := 
  let mid_PC := PC.midpoint in
  let mid_AB1 := AB1.midpoint in
  Line.mk mid_PC mid_AB1

-- The theorem to be proven
theorem lines_intersect_at_a_single_point 
  (h1 : perpendiculars_from_P_to_sides P A1 B1 C1 PA PB PC)
  (h2 : la = line_la A B1 C1 PA (Segment.mk B1 C1))
  (h3 : lb = line_lb B A1 C1 PB (Segment.mk A1 C1))
  (h4 : lc = line_lc C A1 B1 PC (Segment.mk A1 B1)) 
  : ∃ O : α, la.contains O ∧ lb.contains O ∧ lc.contains O 
:= sorry

end lines_intersect_at_a_single_point_l549_549332


namespace find_ellipse_eq_find_line_eq_l549_549724

-- Define the conditions of the problem
def ellipse_eq (a b x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def parabola_eq (x y : ℝ) := y^2 = 4 * x
def focus_of_parabola := (1, 0)
def directrix_of_parabola (x : ℝ) := x = -1
def chord_length := 3
def point_M := (1, 1 / 2 : ℝ)
def midpoint (x1 x2 y1 y2 : ℝ) := (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 1/2
def line_through_M := ∀ x y, 3 * x + 2 * y - 4 = 0

-- Statements to prove
theorem find_ellipse_eq (a b : ℝ) (h1 : parabola_eq 1 0) 
                        (h2 : a^2 - b^2 = 1) 
                        (h3 : (1 : ℝ) / a^2 + (9 / 4 * 1 / b^2) = 1) : 
                        ellipse_eq 2 √3 x y :=
sorry

theorem find_line_eq (x1 x2 y1 y2 : ℝ) (hx1 : 3 * x1^2 + 4 * y1^2 = 12) 
                          (hx2 : 3 * x2^2 + 4 * y2^2 = 12)
                          (hm : midpoint x1 x2 y1 y2) :
                          line_through_M :=
sorry

end find_ellipse_eq_find_line_eq_l549_549724


namespace concyclic_points_of_triangle_conditions_l549_549570

noncomputable def triangle (A B C : Type*) := true

variables (A B C K L N M : Type*)

-- Conditions
def is_triangle (ABC : triangle A B C) : Prop := true

def is_angle_bisector (P Q R S T U : Type*) (bisector : P) : Prop := true

def is_midpoint (A B C midpoint : Type*) : Prop := true

def is_perpendicular (A B line : Type*) : Prop := true

def is_concyclic (A B C D : Type*) : Prop := true

-- Problem statement
theorem concyclic_points_of_triangle_conditions 
  (ABC : triangle A B C)
  (P : is_triangle ABC)
  (K_foot : is_angle_bisector B C A K)
  (L_foot : is_angle_bisector C B A L)
  (N_mid : is_midpoint B C N)
  (M_alt : is_perpendicular A B M)
  :
  is_concyclic K L N M :=
sorry

end concyclic_points_of_triangle_conditions_l549_549570


namespace travis_revenue_l549_549182

-- Declare nonnegative integers for apples, apples per box, and price per box
variables (apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ)

-- Specify the conditions
def conditions := apples = 10000 ∧ apples_per_box = 50 ∧ price_per_box = 35

-- State the theorem to be proved
theorem travis_revenue (h : conditions) : (apples / apples_per_box) * price_per_box = 7000 :=
by
  cases h with 
  | intro h1 h2 h3 =>
  rw [h1, h2, h3]
  sorry -- Proof is not required as per the instructions

end travis_revenue_l549_549182


namespace range_of_m_l549_549522

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ (m > 2 ∨ m < -4) :=
by
  sorry

end range_of_m_l549_549522


namespace find_f2_f5_sum_l549_549878

theorem find_f2_f5_sum
  (f : ℤ → ℤ)
  (a b : ℤ)
  (h1 : f 1 = 4)
  (h2 : ∀ z : ℤ, f z = 3 * z + 6)
  (h3 : ∀ x y : ℤ, f (x + y) = f x + f y + a * x * y + b) :
  f 2 + f 5 = 33 :=
sorry

end find_f2_f5_sum_l549_549878


namespace angle_F1_P_F2_l549_549381

-- Definitions for the conditions given
def ellipse (x y m : ℝ) := (x^2 / m) + y^2 = 1
def hyperbola (x y n : ℝ) := (x^2 / n) - 3 * y^2 = 1

-- Main theorem to be proved
theorem angle_F1_P_F2 {x y m n : ℝ} 
  (h1 : m > 1) 
  (h2 : n > 0) 
  (h3 : ellipse x y m) 
  (h4 : hyperbola x y n) 
  (h5 : m = n + 4/3) : 
  ∠(F1 P F2) = 60 :=
sorry

end angle_F1_P_F2_l549_549381


namespace find_b_value_l549_549274

-- Definitions of the conditions
def quadratic_eq_has_one_solution (a b c : ℝ) : Prop :=
  discriminant a b c = 0

-- Main statement
theorem find_b_value :
  (∃ b : ℝ, quadratic_eq_has_one_solution 3 15 b ∧ b + 3 = 36 ∧ b > 3) →
  (3, (by cases ‹∃ b, _› with b hb; exact b)) = (3, 33) :=
by
  sorry

end find_b_value_l549_549274


namespace apples_left_l549_549695

theorem apples_left (apples_on_tree apples_on_ground apples_eaten : ℕ)
    (h1 : apples_on_tree = 5)
    (h2 : apples_on_ground = 8)
    (h3 : apples_eaten = 3) :
    apples_on_tree + apples_on_ground - apples_eaten = 10 :=
by
    rw [h1, h2, h3] -- rewrite using the conditions
    sorry -- proof goes here

end apples_left_l549_549695


namespace prove_ordered_pair_l549_549079

noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry

theorem prove_ordered_pair (h1 : p 0 = -24) (h2 : q 0 = 30) (h3 : ∀ x : ℝ, p (q x) = q (p x)) : (p 3, q 6) = (3, -24) := 
sorry

end prove_ordered_pair_l549_549079


namespace calculate_flight_duration_l549_549643

noncomputable def duration_of_flight (departure_NY_ET arrival_CT_ET : ℕ) (departure_LON_ET : ℕ) (flight_LON_NY : ℕ) :=
  let arrival_NY_ET := (departure_LON_ET + flight_LON_NY) % 24
  in (arrival_CT_ET + 24 - arrival_NY_ET) % 24

theorem calculate_flight_duration :
  let departure_LON_ET := 6  -- 6:00 a.m. ET on Monday
  let flight_LON_NY := 18   -- 18 hours of flight time
  let arrival_CT_ET := 10  -- 10:00 a.m. ET on Tuesday
  duration_of_flight 0 arrival_CT_ET departure_LON_ET flight_LON_NY = 10 := by
  sorry

end calculate_flight_duration_l549_549643


namespace max_n_for_expected_samples_l549_549974

-- Conditions in Lean
def blueberry_distribution : ProbabilityMassFunction ℝ :=
  ProbabilityMassFunction.normal 15 3

def premium_fruit (Z : ℝ) : Prop :=
  Z > 18

def premium_prob : ℝ :=
  ProbabilityMassFunction.prob_of blueberry_distribution premium_fruit

-- Maximum n such that the expected number of samples is at most 3
theorem max_n_for_expected_samples :
  let E (n : ℕ) := 5 * (1 - (0.8 : ℝ)^n)
  ∃ n : ℕ, E n ≤ 3 ∧ ∀ m : ℕ, m > n → E m > 3 :=
sorry

end max_n_for_expected_samples_l549_549974


namespace percentage_increase_calculation_l549_549603

-- Definition of lemon production and grove size
def normal_lemon_production_per_year : ℕ := 60
def grove_rows : ℕ := 50
def grove_columns : ℕ := 30
def total_lemons_produced_in_5_years : ℕ := 675000

-- Define the expected percentage increase
def expected_percentage_increase : ℕ := 50

-- Goal statement: We need to prove the percentage increase is 50%
theorem percentage_increase_calculation :
  let normal_tree_5_years := normal_lemon_production_per_year * 5
  let total_trees := grove_rows * grove_columns
  let per_tree_5_years := total_lemons_produced_in_5_years / total_trees
  let increase_per_tree := per_tree_5_years - normal_tree_5_years
  let percentage_increase := (increase_per_tree * 100) / normal_tree_5_years
  in percentage_increase = expected_percentage_increase :=
by sorry

end percentage_increase_calculation_l549_549603


namespace rope_length_before_folding_l549_549135

theorem rope_length_before_folding (L : ℝ) (h : L / 4 = 10) : L = 40 :=
by
  sorry

end rope_length_before_folding_l549_549135


namespace four_points_equally_dividing_unit_circle_l549_549952

open Real

theorem four_points_equally_dividing_unit_circle {α : ℝ} :
  (sin α + sin (α + π / 2) + sin (α + π) + sin (α + 3 * π / 2) = 0) :=
  have h1 : sin α + sin (α + π) = 0 := sorry,
  have h2 : sin α + sin (α + 2 * π / 3) + sin (α + 4 * π / 3) = 0 := sorry,
  sorry

end four_points_equally_dividing_unit_circle_l549_549952


namespace slope_of_reflected_ray_l549_549230

def point (x y : ℝ) := (x, y)

def pointA := point (-2) 3
def pointA' := point (-2) (-3)
def circle_center := point 3 2
def circle_radius := 1

def is_line_passing_through_point (line : ℝ → ℝ → Prop) (p : ℝ × ℝ) := line p.1 p.2

noncomputable def slope_candidates (k : ℝ) : Prop :=
  let line_eq := (λ x y, (k * x - y + 2 * k - 3 = 0)) in
  let dist_to_circle_center :=
    |(3 * k - 2 + 2 * k - 3)| / Math.sqrt (k^2 + 1) in
  dist_to_circle_center = circle_radius

theorem slope_of_reflected_ray :
  slope_candidates (4 / 3) ∨ slope_candidates (3 / 4) := sorry

end slope_of_reflected_ray_l549_549230


namespace mean_of_five_numbers_is_correct_l549_549164

-- Define the given sum of five numbers as three-quarters
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers, which is 5
def number_of_numbers : ℕ := 5

-- Define the mean calculation from the given sum and number of numbers
def mean_five_numbers (sum : ℚ) (count : ℕ) : ℚ := sum / count

-- Statement to prove: the mean of five numbers given their sum is 3/4 equals 3/20
theorem mean_of_five_numbers_is_correct :
  mean_five_numbers sum_of_five_numbers number_of_numbers = 3 / 20 :=
by
  -- Skipping the proof
  sorry

end mean_of_five_numbers_is_correct_l549_549164


namespace minimum_value_a_l549_549730

noncomputable def f (a b x : ℝ) := a * Real.log x - (1 / 2) * x^2 + b * x

theorem minimum_value_a (h : ∀ b x : ℝ, x > 0 → f a b x > 0) : a ≥ -Real.exp 3 := 
sorry

end minimum_value_a_l549_549730


namespace find_length_d_of_folded_rectangle_l549_549617

variable (d : ℝ) -- Define the variable d as a real number

-- Given conditions:
def is_rectangle_folded_correctly (a b : ℝ) : Prop :=
  a = 1 ∧ b = real.sqrt 2 ∧ (d = b - 1)

theorem find_length_d_of_folded_rectangle :
  is_rectangle_folded_correctly 1 (real.sqrt 2) →
  d = real.sqrt 2 - 1 :=
by
  sorry

end find_length_d_of_folded_rectangle_l549_549617


namespace sam_catches_alice_in_40_minutes_l549_549249

def sam_speed := 7 -- mph
def alice_speed := 4 -- mph
def initial_distance := 2 -- miles

theorem sam_catches_alice_in_40_minutes : 
  (initial_distance / (sam_speed - alice_speed)) * 60 = 40 :=
by sorry

end sam_catches_alice_in_40_minutes_l549_549249


namespace trisha_cookie_count_equals_twelve_l549_549255

-- Define the given amounts of dough and geometry of the cookies
def dough_amount : ℝ := 120
def side_length_art_cookie : ℝ := 4
def base_length_trisha_cookie : ℝ := 4
def height_trisha_cookie : ℝ := 5

-- Define the areas of the cookies
def area_art_cookie : ℝ := (sqrt 3 / 4) * (side_length_art_cookie ^ 2)
def area_trisha_cookie : ℝ := (1 / 2) * base_length_trisha_cookie * height_trisha_cookie

-- Define the number of cookies
def num_trisha_cookies (dough : ℝ) (cookie_area : ℝ) : ℝ := dough / cookie_area

theorem trisha_cookie_count_equals_twelve : num_trisha_cookies dough_amount area_trisha_cookie = 12 := by
  -- Placeholder proof
  sorry

end trisha_cookie_count_equals_twelve_l549_549255


namespace frac_simplification_l549_549669

theorem frac_simplification : 4.9 * 106 = 519.4 → 18 / (4.9 * 106) = 18 / 519.4 :=
by
  intro h
  rw h
  rfl

end frac_simplification_l549_549669


namespace train_probability_correct_l549_549621

noncomputable def train_prob (a_train b_train a_john b_john wait : ℝ) : ℝ :=
  let total_time_frame := (b_train - a_train) * (b_john - a_john)
  let triangle_area := (1 / 2) * wait * wait
  let rectangle_area := wait * wait
  let total_overlap_area := triangle_area + rectangle_area
  total_overlap_area / total_time_frame

theorem train_probability_correct :
  train_prob 120 240 150 210 30 = 3 / 16 :=
by
  sorry

end train_probability_correct_l549_549621


namespace expected_pairs_of_adjacent_face_cards_is_44_over_17_l549_549889
noncomputable def expected_adjacent_face_card_pairs : ℚ :=
  12 * (11 / 51)

theorem expected_pairs_of_adjacent_face_cards_is_44_over_17 :
  expected_adjacent_face_card_pairs = 44 / 17 :=
by
  sorry

end expected_pairs_of_adjacent_face_cards_is_44_over_17_l549_549889


namespace proof_unique_black_vertex_l549_549661

noncomputable def exists_unique_black_vertex : Prop :=
  ∃ (grid : ℕ × ℕ → bool), 
  (∃ i j, i < 100 ∧ j < 100 ∧ grid (i, j) = tt) ∧  -- there exists at least one black square
  (∃ i j, i < 100 ∧ j < 100 ∧ grid (i, j) = ff) ∧  -- there exists at least one white square
  ∃ x y, x <= 100 ∧ y <= 100 ∧ 
  (grid (x, y) = tt) ∧  -- The specified vertex corresponds to a black square
  (∀ i j, i < 100 ∧ j < 100 ∧ 
  (i < x ∨ j < y) → grid (i, j) = ff) -- All vertices left to x and above y must be white.
  
theorem proof_unique_black_vertex : exists_unique_black_vertex := 
  sorry

end proof_unique_black_vertex_l549_549661


namespace infinite_sum_equals_l549_549509

theorem infinite_sum_equals :
  10 * (79 * (1 / 7)) + (∑' n : ℕ, if n % 2 = 0 then (if n = 0 then 0 else 2 / 7 ^ n) else (1 / 7 ^ n)) = 3 / 16 :=
by
  sorry

end infinite_sum_equals_l549_549509


namespace fixed_point_l549_549591

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  ∃ x y : ℝ, x = -2 ∧ y = 2 ∧ (f : ℝ → ℝ) := (λ x, a * x + 3) :=
by
  sorry

end fixed_point_l549_549591


namespace right_triangle_thirty_degree_leg_l549_549791

theorem right_triangle_thirty_degree_leg (A B C : Type) [MetricSpace (A)] :
  ∃ (a b c : ℝ), 90° = \(\angle C\) ∧ 30° = \(\angle A\) ∧
  (\(isRightTriangle\). (A B C)) → (\(\legOppositeThirty\). a c = c / 2) :=
sorry

end right_triangle_thirty_degree_leg_l549_549791


namespace minimize_slope_min_a_value_l549_549352

theorem minimize_slope (a : ℝ) (h : a > 0) : 
  let f := λ x : ℝ, 2 * a * x^2 - (1 / (a * x))
  let k := (deriv f 1)
  k = 4 * a + (1 / a) := by
    sorry

theorem min_a_value : ∃ (a : ℝ), a > 0 ∧ 
  let f := λ x : ℝ, 2 * a * x^2 - (1 / (a * x))
  let k := (4 * a + (1 / a))
  ∀ (a' : ℝ), a' > 0 → (4 * a' + (1 / a')) ≥ k ∧ 
  k = 4 := by
    use (1 / 2)
    sorry

end minimize_slope_min_a_value_l549_549352


namespace arcsin_arccos_solution_l549_549136

theorem arcsin_arccos_solution (x : ℝ) (hx1 : -1 ≤ x) (hx2 : x ≤ 1) (hx3 : -1 ≤ x - 1) (hx4 : x - 1 ≤ 1) :
  arccos (1 - x) = arcsin x + arcsin (x - 1) → x = 1 :=
by
  sorry

end arcsin_arccos_solution_l549_549136


namespace angle_D_in_convex_quadrilateral_l549_549708

theorem angle_D_in_convex_quadrilateral 
  (A B C D : ℝ) 
  (hABCD_convex : (A + B + C + D = 360))
  (h_angle_C : C = 57)
  (h_sin_sum : sin A + sin B = sqrt 2)
  (h_cos_sum : cos A + cos B = 2 - sqrt 2) : 
  D = 168 :=
by 
  sorry

end angle_D_in_convex_quadrilateral_l549_549708


namespace complex_division_l549_549142

-- Definitions for the complex numbers involved
def z1 : ℂ := 2 - I
def z2 : ℂ := 2 + I
def result : ℂ := (3 / 5) - (4 / 5) * I

-- Problem Statement: Proving the division of complex numbers
theorem complex_division :
  z1 / z2 = result :=
sorry

end complex_division_l549_549142


namespace time_for_all_balls_l549_549630

noncomputable def alexia_rate := 18
noncomputable def ermias_rate := 25
noncomputable def leila_rate := 30
noncomputable def alexia_balls := 50
noncomputable def ermias_balls := alexia_balls + 12
noncomputable def leila_balls := ermias_balls - 5

def alexia_time : Nat := alexia_balls * alexia_rate
def ermias_time : Nat := ermias_balls * ermias_rate
def leila_time : Nat := leila_balls * leila_rate

def total_time := alexia_time + ermias_time + leila_time

theorem time_for_all_balls :
  total_time = 4160 := by
  sorry

end time_for_all_balls_l549_549630


namespace berry_ratio_l549_549498

-- Define the conditions
variables (S V R : ℕ) -- Number of berries Stacy, Steve, and Sylar have
axiom h1 : S + V + R = 1100
axiom h2 : S = 800
axiom h3 : V = 2 * R

-- Define the theorem to be proved
theorem berry_ratio (h1 : S + V + R = 1100) (h2 : S = 800) (h3 : V = 2 * R) : S / V = 4 :=
by
  sorry

end berry_ratio_l549_549498


namespace max_n_for_factoring_l549_549678

theorem max_n_for_factoring (n : ℤ) :
  (∃ A B : ℤ, (5 * B + A = n) ∧ (A * B = 90)) → n = 451 :=
by
  sorry

end max_n_for_factoring_l549_549678


namespace minimum_red_pieces_l549_549528

theorem minimum_red_pieces (w b r : ℕ) 
  (h1 : b ≤ w / 2) 
  (h2 : r ≥ 3 * b) 
  (h3 : w + b ≥ 55) : r = 57 := 
sorry

end minimum_red_pieces_l549_549528


namespace sum_last_two_digits_fib_factorial_series_l549_549549

def last_two_digits (n : ℕ) : ℕ := n % 100

def fib_factorial_series : List ℕ := [1!, 1!, 2!, 3!, 5!, 8!, 13!, 21!]

def sum_last_two_digits (l : List ℕ) : ℕ :=
  l.map last_two_digits |>.sum

theorem sum_last_two_digits_fib_factorial_series :
  sum_last_two_digits fib_factorial_series = 5 := by
  sorry

end sum_last_two_digits_fib_factorial_series_l549_549549


namespace total_time_to_braid_hair_l549_549066

constant dancers : ℕ := 8
constant braidsPerDancer : ℕ := 5
constant secondsPerBraid : ℕ := 30
constant secondsPerMinute : ℕ := 60

theorem total_time_to_braid_hair : 
  (dancers * braidsPerDancer * secondsPerBraid) / secondsPerMinute = 20 := 
by
  sorry

end total_time_to_braid_hair_l549_549066


namespace triangle_sides_ratio_l549_549406

theorem triangle_sides_ratio (A B C : ℝ) (a b c : ℝ)
  (h1 : a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = Real.sqrt 2 * a)
  (ha_pos : a > 0) : b / a = Real.sqrt 2 :=
sorry

end triangle_sides_ratio_l549_549406


namespace number_of_proper_subsets_of_M_l549_549377

def M : Set ℕ := {0, 2}

theorem number_of_proper_subsets_of_M : (Set.powerset M).card - 1 = 3 := 
by
  sorry

end number_of_proper_subsets_of_M_l549_549377


namespace tenth_term_of_arithmetic_sequence_l549_549010

theorem tenth_term_of_arithmetic_sequence :
  ∃ a : ℕ → ℤ, (∀ n : ℕ, a n + 1 - a n = 2) ∧ a 1 = 1 ∧ a 10 = 19 :=
sorry

end tenth_term_of_arithmetic_sequence_l549_549010


namespace power_of_one_minus_i_eq_neg_4_l549_549577

noncomputable def i : ℂ := complex.I

theorem power_of_one_minus_i_eq_neg_4 : (1 - i)^4 = -4 :=
by
  sorry

end power_of_one_minus_i_eq_neg_4_l549_549577


namespace mean_of_five_numbers_l549_549169

theorem mean_of_five_numbers (S : ℚ) (n : ℕ) (h1 : S = 3/4) (h2 : n = 5) :
  (S / n) = 3/20 :=
by
  rw [h1, h2]
  sorry

end mean_of_five_numbers_l549_549169


namespace product_of_slope_and_intercept_l549_549713

theorem product_of_slope_and_intercept :
  let m := (3 : ℚ) / 5
  let b := - (3 : ℚ) / 2
  -1 < m * b ∧ m * b < 0 :=
by
  let m := (3 / 5 : ℚ)
  let b := (-3 / 2 : ℚ)
  have h1 : m * b = -9 / 10 := by norm_num
  split
  {
    have : -1 < -9 / 10 := by norm_num
    exact this
  }
  {
    have : -9 / 10 < 0 := by norm_num
    exact this
  }

end product_of_slope_and_intercept_l549_549713


namespace expansion_correct_l549_549670

-- Define the polynomials
def poly1 (z : ℤ) : ℤ := 3 * z^2 + 4 * z - 5
def poly2 (z : ℤ) : ℤ := 4 * z^4 - 3 * z^2 + 2

-- Define the expected expanded polynomial
def expanded_poly (z : ℤ) : ℤ := 12 * z^6 + 16 * z^5 - 29 * z^4 - 12 * z^3 + 21 * z^2 + 8 * z - 10

-- The theorem that proves the equivalence of the expanded form
theorem expansion_correct (z : ℤ) : (poly1 z) * (poly2 z) = expanded_poly z := by
  sorry

end expansion_correct_l549_549670


namespace max_min_difference_l549_549006

noncomputable def f (x : ℝ) : ℝ := Real.log x - (x - 1) / x

theorem max_min_difference (M m : ℝ) :
  let f := λ x : ℝ, Real.log x - (x - 1) / x in
  (∀ x ∈ set.Icc 1 Real.exp 1, f(x) ≤ M) ∧ 
  (∀ x ∈ set.Icc 1 Real.exp 1, f(x) ≥ m) ∧ 
  (∃ x ∈ set.Icc 1 Real.exp 1, f x = m) ∧
  (∃ x ∈ set.Icc 1 Real.exp 1, f x = M) →
  M - m = 1 / Real.exp 1 :=
by
  sorry

end max_min_difference_l549_549006


namespace misha_students_l549_549472

theorem misha_students : 
  ∀ (n : ℕ),
  (n = 74 + 1 + 74) ↔ (n = 149) :=
by
  intro n
  split
  · intro h
    rw [← h, nat.add_assoc]
    apply nat.add_right_cancel
    rw [nat.add_comm 1 74, nat.add_assoc]
    apply nat.add_right_cancel
    rw nat.add_comm
  · intro h
    exact h
  sorry

end misha_students_l549_549472


namespace complex_quadratic_power_l549_549571

theorem complex_quadratic_power :
  (1 - Complex.i) ^ 4 = -4 :=
by
  sorry

end complex_quadratic_power_l549_549571


namespace mean_of_five_numbers_l549_549160

theorem mean_of_five_numbers (sum : ℚ) (h : sum = 3 / 4) : (sum / 5 = 3 / 20) :=
by
  -- Proof omitted
  sorry

end mean_of_five_numbers_l549_549160


namespace find_q_l549_549383

noncomputable def p (q : ℝ) : ℝ := 16 / (3 * q)

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 3/2) (h4 : p * q = 16/3) : q = 24 / 6 + 19.6 / 6 :=
by
  sorry

end find_q_l549_549383


namespace sqrt_of_16_l549_549919

theorem sqrt_of_16 (x : ℝ) (hx : x^2 = 16) : x = 4 ∨ x = -4 := 
by
  sorry

end sqrt_of_16_l549_549919


namespace find_integer_solutions_l549_549673

theorem find_integer_solutions :
  {p : ℤ × ℤ | 2 * p.1^3 + p.1 * p.2 = 7} = {(-7, -99), (-1, -9), (1, 5), (7, -97)} :=
by
  -- Proof not required
  sorry

end find_integer_solutions_l549_549673


namespace linear_functions_value_at_20_l549_549538

-- Definitions of linear functions intersection and properties
def linear_functions_intersect_at (k1 k2 b1 b2 : ℝ) (x : ℝ) : Prop :=
  k1 * x + b1 = k2 * x + b2

def value_difference_at (k1 k2 b1 b2 : ℝ) (x diff : ℝ) : Prop :=
  (k1 * x + b1 - (k2 * x + b2)).abs = diff

def function_value_at (k b x val : ℝ) : Prop :=
  k * x + b = val

-- Main proof statement
theorem linear_functions_value_at_20
  (k1 k2 b1 b2 : ℝ) :
  (linear_functions_intersect_at k1 k2 b1 b2 2) →
  (value_difference_at k1 k2 b1 b2 8 8) →
  (function_value_at k1 b1 20 100) →
  (k2 * 20 + b2 = 76 ∨ k2 * 20 + b2 = 124) :=
by
  sorry

end linear_functions_value_at_20_l549_549538


namespace third_side_length_of_triangle_l549_549362

theorem third_side_length_of_triangle {a b c : ℝ} (h1 : a^2 - 7 * a + 12 = 0) (h2 : b^2 - 7 * b + 12 = 0) 
  (h3 : a ≠ b) (h4 : a = 3 ∨ a = 4) (h5 : b = 3 ∨ b = 4) : 
  (c = 5 ∨ c = Real.sqrt 7) := by
  sorry

end third_side_length_of_triangle_l549_549362


namespace area_bisecting_line_slope_y_intercept_sum_l549_549275

-- Define points as pairs of coordinates
structure Point where
  x : ℝ
  y : ℝ

def D : Point := ⟨1, 6⟩
def E : Point := ⟨3, -2⟩
def F : Point := ⟨7, -2⟩

-- Midpoint function
def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2⟩

-- Calculate the midpoint of D and F
def M : Point := midpoint D F

-- Define a line through two given points and its slope
def slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

def y_intercept (m : ℝ) (P : Point) : ℝ :=
  P.y - m * P.x

-- Define the slope and y-intercept of the line passing through E and M and then sum them
noncomputable def slope_E_M : ℝ := slope E M
noncomputable def y_intercept_E_M : ℝ := y_intercept slope_E_M E

noncomputable def sum_slope_y_intercept : ℝ := slope_E_M + y_intercept_E_M

-- The theorem to be proven
theorem area_bisecting_line_slope_y_intercept_sum :
  sum_slope_y_intercept = -10 :=
by
  sorry

end area_bisecting_line_slope_y_intercept_sum_l549_549275


namespace impossible_grid_filling_l549_549807

/-- 
  Prove that it is impossible to fill a 100 × 100 grid with the numbers 0, 1, or 2 
  such that in any 3 × 4 subgrid, there are exactly 3 zeros, 4 ones, and 5 twos.
-/
theorem impossible_grid_filling :
  ∀ (grid : Fin 100 → Fin 100 → Fin 3),
  (∀ (i j : Fin 98) (k l : Fin 97), 
    let subgrid := {x | i ≤ x.1 < i + 3 ∧ j ≤ x.2 < j + 4},
    (subgrid.card (λ x => grid x.1 x.2 = 0) = 3) ∧
    (subgrid.card (λ x => grid x.1 x.2 = 1) = 4) ∧
    (subgrid.card (λ x => grid x.1 x.2 = 2) = 5)) → False :=
begin
  sorry
end

end impossible_grid_filling_l549_549807


namespace cost_of_4500_pencils_l549_549216

theorem cost_of_4500_pencils (price_per_box : ℚ) (pencils_per_box : ℕ) (bulk_qty : ℕ) (discount : ℚ) (desired_qty : ℕ) (h_price : price_per_box = 40) (h_pencils : pencils_per_box = 150) (h_bulk_qty : bulk_qty = 3000) (h_discount : discount = 0.10) (h_qty : desired_qty = 4500) : 
  let cost_per_pencil := price_per_box / pencils_per_box
      initial_cost := cost_per_pencil * desired_qty
      discounted_cost := initial_cost * (1 - discount)
  in discounted_cost = 1080 :=
by {
  have h_per_pencil : cost_per_pencil = 40 / 150, from by rw [h_price, h_pencils],
  have h_initial_cost : initial_cost = (40 / 150) * 4500, from by rw [h_per_pencil, h_qty],
  have h_discounted_cost : discounted_cost = ((40 / 150) * 4500) * (1 - 0.10), from by rw [h_initial_cost, h_discount],
  rw [show (40 / 150) * 4500 = 1200, by norm_num] at h_discounted_cost,
  rw [show 1200 * (1 - 0.10) = 1080, by norm_num] at h_discounted_cost,
  exact h_discounted_cost
}


end cost_of_4500_pencils_l549_549216


namespace min_n_binomial_ratio_l549_549416

theorem min_n_binomial_ratio : 
  ∃ (n : ℕ), (1 < n) ∧ (∀ r : ℕ, n.choose r (n - r).succ = 5 / 7 * (n.choose r * (n - r + 1).pred)) -> n = 11 :=
by
  sorry

end min_n_binomial_ratio_l549_549416


namespace find_f_2017_l549_549834

noncomputable def f (x p q : ℝ) : ℝ := x^2 + p*x + q

theorem find_f_2017 (p q : ℝ) (h : ∀ x, 3 ≤ x ∧ x ≤ 5 → abs (f x p q) ≤ 1 / 2) :
  f^[2017] (f (7 + sqrt 15) (7 - sqrt 15) / 2) p q ≈ 1.56 :=
sorry

end find_f_2017_l549_549834


namespace programmer_debugging_hours_l549_549282

theorem programmer_debugging_hours
    (total_hours : ℕ)
    (flow_chart_fraction : ℚ)
    (coding_fraction : ℚ)
    (meeting_fraction : ℚ)
    (flow_chart_hours : ℚ)
    (coding_hours : ℚ)
    (meeting_hours : ℚ)
    (debugging_hours : ℚ)
    (H1 : total_hours = 192)
    (H2 : flow_chart_fraction = 3 / 10)
    (H3 : coding_fraction = 3 / 8)
    (H4 : meeting_fraction = 1 / 5)
    (H5 : flow_chart_hours = flow_chart_fraction * total_hours)
    (H6 : coding_hours = coding_fraction * total_hours)
    (H7 : meeting_hours = meeting_fraction * total_hours)
    (H8 : debugging_hours = total_hours - (flow_chart_hours + coding_hours + meeting_hours))
    :
    debugging_hours = 24 :=
by 
  sorry

end programmer_debugging_hours_l549_549282


namespace arithmetic_progression_terms_l549_549905

theorem arithmetic_progression_terms
  (n : ℕ) (a d : ℝ)
  (hn_odd : n % 2 = 1)
  (sum_odd_terms : n / 2 * (2 * a + (n / 2 - 1) * d) = 30)
  (sum_even_terms : (n / 2 - 1) * (2 * (a + d) + (n / 2 - 2) * d) = 36)
  (sum_all_terms : n / 2 * (2 * a + (n - 1) * d) = 66)
  (last_first_diff : (n - 1) * d = 12) :
  n = 9 := sorry

end arithmetic_progression_terms_l549_549905


namespace prime_sum_divisible_l549_549857

theorem prime_sum_divisible (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q = p + 2) :
  (p ^ q + q ^ p) % (p + q) = 0 :=
by
  sorry

end prime_sum_divisible_l549_549857


namespace transformation_of_cosine_function_l549_549368

theorem transformation_of_cosine_function (f g : ℝ → ℝ) (ϕ : ℝ) (h1 : 0 < ϕ) (h2 : ϕ < π / 2) 
  (h_even : ∀ x, f x = f (-x)) 
  (h_f_def : ∀ x, f x = 1 + 2 * cos x * cos (x + 3 * ϕ)) 
  (h_g_def : ∀ x, g x = cos (2 * x - ϕ))
  : ∀ x, g x = f (x + ϕ / 3) := 
sorry

end transformation_of_cosine_function_l549_549368


namespace R_depends_on_a_d_n_l549_549086

-- Definition of sum of an arithmetic progression
def sum_arithmetic_progression (n : ℕ) (a d : ℤ) : ℤ := 
  n * (2 * a + (n - 1) * d) / 2

-- Definitions for s1, s2, and s4
def s1 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression n a d
def s2 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression (2 * n) a d
def s4 (n : ℕ) (a d : ℤ) : ℤ := sum_arithmetic_progression (4 * n) a d

-- Definition of R
def R (n : ℕ) (a d : ℤ) : ℤ := s4 n a d - s2 n a d - s1 n a d

-- Theorem stating R depends on a, d, and n
theorem R_depends_on_a_d_n : 
  ∀ (n : ℕ) (a d : ℤ), ∃ (p q r : ℤ), R n a d = p * a + q * d + r := 
by
  sorry

end R_depends_on_a_d_n_l549_549086


namespace fencing_cost_105_rupees_l549_549565

structure Rectangle :=
  (length : ℕ)
  (width : ℕ)
  (area : ℕ)
  (ratio_length_width : ℕ → ℕ → Prop := λ l w, l / w = 4 / 3)

def perimeter (rect : Rectangle) : ℕ :=
  2 * (rect.length + rect.width)

def cost_of_fencing (perimeter : ℕ) (cost_per_meter_paise : ℕ) : ℕ :=
  (perimeter * cost_per_meter_paise) / 100

theorem fencing_cost_105_rupees :
  ∃ (x : ℕ), 12 * x^2 = 10800 ∧
              let length := 4 * x in
              let width := 3 * x in
              let rect := Rectangle.mk length width (length * width) (λ _ _, true) in
              cost_of_fencing (perimeter rect) 25 = 105 :=
sorry

end fencing_cost_105_rupees_l549_549565


namespace gcd_polynomials_l549_549647

def P (n : ℤ) : ℤ := n^3 - 6 * n^2 + 11 * n - 6
def Q (n : ℤ) : ℤ := n^2 - 4 * n + 4

theorem gcd_polynomials (n : ℤ) (h : n ≥ 3) : Int.gcd (P n) (Q n) = n - 2 :=
by
  sorry

end gcd_polynomials_l549_549647


namespace push_pin_distribution_l549_549129

theorem push_pin_distribution :
  (∑ n₀ n₁ n₂ n₃ in Finset.range(7).filter(λ n, n₀ + n₁ + n₂ + n₃ = 6), 1 : ℕ) = 84 :=
sorry

end push_pin_distribution_l549_549129


namespace product_of_gcd_and_lcm_1440_l549_549315

def product_of_gcd_and_lcm_of_24_and_60 : ℕ :=
  Nat.gcd 24 60 * Nat.lcm 24 60 

/-- The product of the gcd and lcm of 24 and 60 is 1440. -/
theorem product_of_gcd_and_lcm_1440 : product_of_gcd_and_lcm_of_24_and_60 = 1440 := 
by
  sorry

end product_of_gcd_and_lcm_1440_l549_549315


namespace find_angle_B_l549_549055

theorem find_angle_B (a b c : ℝ) (A B C : ℝ) (h_triangle : a = b * cos B + c * cos A) : B = π / 3 :=
sorry

end find_angle_B_l549_549055


namespace valid_colorings_count_l549_549127

-- Definition of points and grid
def point (x: ℕ) (y: ℕ) := (x, y)

def square (n: ℕ) : set (ℕ × ℕ) :=
  { p | p.1 ≤ n ∧ p.2 ≤ n }

-- Coloring condition
def is_valid_coloring (color: ℕ × ℕ → bool) (n: ℕ) : Prop :=
  ∀ i j, i < n ∧ j < n → (literal) % 2 = 0

-- Theorem statement
theorem valid_colorings_count (n : ℕ) : ∃ (k : ℕ), k = 2^(n+2) - 2 :=
sorry

end valid_colorings_count_l549_549127


namespace linear_function_quadrants_l549_549031

theorem linear_function_quadrants (m : ℝ) :
  (∀ (x : ℝ), y = -3 * x + m →
  (x < 0 ∧ y > 0 ∨ x > 0 ∧ y < 0 ∨ x < 0 ∧ y < 0)) → m < 0 :=
sorry

end linear_function_quadrants_l549_549031


namespace find_a_plus_b_l549_549024

theorem find_a_plus_b (x a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hx : x = a + Real.sqrt b)
  (hxeq : x^2 + 5*x + 5/x + 1/(x^2) = 42) : a + b = 5 :=
sorry

end find_a_plus_b_l549_549024


namespace solve_first_train_length_l549_549188

noncomputable def first_train_length (time: ℝ) (speed1_kmh: ℝ) (speed2_kmh: ℝ) (length2: ℝ) : ℝ :=
  let speed1_ms := speed1_kmh * 1000 / 3600
  let speed2_ms := speed2_kmh * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  let total_distance := relative_speed * time
  total_distance - length2

theorem solve_first_train_length :
  first_train_length 7.0752960452818945 80 65 165 = 120.28 :=
by
  simp [first_train_length]
  norm_num
  sorry

end solve_first_train_length_l549_549188


namespace school_seat_payment_l549_549612

def seat_cost (num_rows : ℕ) (seats_per_row : ℕ) (cost_per_seat : ℕ) (discount : ℕ → ℕ → ℕ) : ℕ :=
  let total_seats := num_rows * seats_per_row
  let total_cost := total_seats * cost_per_seat
  let groups_of_ten := total_seats / 10
  let total_discount := groups_of_ten * discount 10 cost_per_seat
  total_cost - total_discount

-- Define the discount function as 10% of the cost of a group of 10 seats
def discount (group_size : ℕ) (cost_per_seat : ℕ) : ℕ := (group_size * cost_per_seat) / 10

theorem school_seat_payment :
  seat_cost 5 8 30 discount = 1080 :=
sorry

end school_seat_payment_l549_549612


namespace parallelogram_area_l549_549827

variables {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]
variables (p q : V)
variables (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (angle_45 : inner_product_geometry.angle p q = real.pi / 4)

theorem parallelogram_area : 
  ‖(p + 3 • q) ∧ (3 • p + q)‖ = 2 * real.sqrt 2 := by
  sorry

end parallelogram_area_l549_549827


namespace no_values_of_x_satisfy_f_f_x_eq_zero_l549_549743

noncomputable def f (x : ℝ) : ℝ :=
  if x <= -1 then -0.5 * x^2 + x + 3
  else if x <= 1 then -0.5 * x^2 + x + 3
  else 0.5 * x^2 - x + 1.5

theorem no_values_of_x_satisfy_f_f_x_eq_zero : 
  ∀ x ∈ set.Icc (-5 : ℝ) (5 : ℝ), f (f x) ≠ 0 := 
by
  sorry

end no_values_of_x_satisfy_f_f_x_eq_zero_l549_549743


namespace mixed_number_division_l549_549638

theorem mixed_number_division :
  (4 + 2 / 3 + 5 + 1 / 4) / (3 + 1 / 2 - 2 + 3 / 5) = 11 + 1 / 54 :=
by
  sorry

end mixed_number_division_l549_549638


namespace max_final_result_l549_549150

theorem max_final_result :
  ∃ (result : ℕ), result = 2030 ∧ 
  (∀ (nums : List ℕ) (h : nums = List.range 2031), 
    (∃ f : (List ℕ → (ℕ → ℕ)) → List ℕ → ℕ, 
      (∀ (op : List ℕ → (ℕ → ℕ)) (nums : List ℕ),
        nums.length = 1 → f op nums = nums.head!) ∧ 
      (∀ (op : List ℕ → (ℕ → ℕ)) (nums : List ℕ),
        nums.length > 1 → 
        ∃ next_nums : List ℕ,
          next_nums = nums.filter (λ x : ℕ, x ≠ op nums 0) ++ [op nums 0] ∧ 
          f op nums = f op next_nums)) → 
    result = f (λ (nums : List ℕ) (i : ℕ), |nums.nth_le 0 sorry - nums.nth_le 1 sorry|) nums)) :=
sorry

end max_final_result_l549_549150


namespace ellipse_a_range_l549_549366

theorem ellipse_a_range (a b : ℝ) (e : ℝ) :
  (0 < b) → (b < a) →
  (e = Real.sqrt (1 - b^2 / a^2)) →
  (e ∈ Set.Icc (1 / Real.sqrt 3) (1 / Real.sqrt 2)) →
  (∀ (M N : ℝ × ℝ), (M.2 = -M.1 + 1) → (N.2 = -N.1 + 1) →
    (M.1, M.2) ∈ {p | p.1^2 / a^2 + p.2^2 / b^2 = 1} →
    (N.1, N.2) ∈ {p | p.1^2 / a^2 + p.2^2 / b^2 = 1} →
    (M.1 * N.1 + M.2 * N.2 = 0)) →
  (Set.Icc (Real.sqrt 5 / 2) (Real.sqrt 6 / 2)).indicator a = 1 :=
sorry

end ellipse_a_range_l549_549366


namespace count_two_digit_numbers_gt_29_lt_36_l549_549766

theorem count_two_digit_numbers_gt_29_lt_36 : 
  {n : ℕ | 29 < n ∧ n < 36 ∧ 10 ≤ n ∧ n ≤ 99}.toFinset.card = 6 :=
by
  sorry

end count_two_digit_numbers_gt_29_lt_36_l549_549766


namespace estimate_total_fish_in_pond_l549_549178

theorem estimate_total_fish_in_pond :
  ∀ (total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample : ℕ),
  initial_sample_size = 100 →
  second_sample_size = 200 →
  tagged_in_second_sample = 10 →
  total_tagged_fish = 100 →
  (total_tagged_fish : ℚ) / (total_fish : ℚ) = tagged_in_second_sample / second_sample_size →
  total_fish = 2000 := by
  intros total_tagged_fish initial_sample_size second_sample_size tagged_in_second_sample
  intro h1 h2 h3 h4 h5
  sorry

end estimate_total_fish_in_pond_l549_549178


namespace average_of_numbers_between_6_and_34_divisible_by_5_l549_549561

theorem average_of_numbers_between_6_and_34_divisible_by_5 : 
  let nums := [10, 15, 20, 25, 30]
  in (nums.sum / nums.length = 20) := 
by 
  sorry

end average_of_numbers_between_6_and_34_divisible_by_5_l549_549561


namespace birthday_problem_l549_549407

def probability_at_least_two_students_same_birthday (n : ℕ) (d : ℕ) : ℝ :=
  1 - (∏ k in Finset.range n, (1 - k / d.to_real))

theorem birthday_problem (n : ℕ) (d : ℕ) (h_n : n = 30) (h_d : d = 365) :
  probability_at_least_two_students_same_birthday n d > 0.5 :=
by
  -- This is where the proof would be, but we provide a placeholder for now.
  sorry

end birthday_problem_l549_549407


namespace max_truth_tellers_l549_549635

theorem max_truth_tellers (f s : ℕ → Prop) (T : ℕ → Prop)
  (h1 : T 0) (h2 : T 1) (h3 : T 2)
  (hs : ∀ n, 3 ≤ n → s n → (f (n-1) ∧ f (n-2) ∧ ¬f (n-3)) ∨ (f (n-1) ∧ ¬f (n-2) ∧ f (n-3)) ∨ (¬f (n-1) ∧ f (n-2) ∧ f (n-3))) :
  {n : ℕ | f n}.card ≤ 45 :=
by sorry

end max_truth_tellers_l549_549635


namespace solve_for_a_l549_549004

noncomputable def f (x : ℝ) : ℝ := 3 * (x + 1) / 2 + a

theorem solve_for_a : ∃ (a : ℝ), (∀ (f : ℝ → ℝ), (∀ (x : ℝ), f (2 * x - 1) = 3 * x + a) ∧ f 3 = 2) → a = -4 :=
by
  sorry

end solve_for_a_l549_549004


namespace intersection_of_M_and_N_is_0_and_2_l549_549746

open Set

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N_is_0_and_2 : M ∩ N = {0, 2} :=
by
  sorry

end intersection_of_M_and_N_is_0_and_2_l549_549746


namespace find_x_l549_549604

def price_of_potted_plant (x : ℝ) : Prop :=
  let total_earnings_orchids := 20 * 50 in
  let total_earnings_plants := 15 * x in
  let total_earnings := total_earnings_orchids + total_earnings_plants in
  let payment_to_workers := 2 * 40 in
  let expenses_pots := 150 in
  let total_expenses := payment_to_workers + expenses_pots in
  let earnings_before_expenses := 1145 + total_expenses in
  total_earnings = earnings_before_expenses

theorem find_x : price_of_potted_plant 25 :=
by {
  let total_earnings_orchids := 20 * 50,
  let total_earnings_plants := 15 * 25,
  let total_earnings := total_earnings_orchids + total_earnings_plants,
  let payment_to_workers := 2 * 40,
  let expenses_pots := 150,
  let total_expenses := payment_to_workers + expenses_pots,
  let earnings_before_expenses := 1145 + total_expenses,
  show total_earnings = earnings_before_expenses,
  calc
    total_earnings_orchids = 1000 : by norm_num
    ...
    earnings_before_expenses = 1375 : by norm_num,
  sorry
}

end find_x_l549_549604


namespace part1_part2_l549_549220

-- Define the conditions
def cost_price := 30
def initial_selling_price := 40
def initial_sales_volume := 600
def sales_decrease_per_yuan := 10

-- Define the profit calculation function
def profit (selling_price : ℕ) : ℕ :=
  let profit_per_unit := selling_price - cost_price
  let new_sales_volume := initial_sales_volume - sales_decrease_per_yuan * (selling_price - initial_selling_price)
  profit_per_unit * new_sales_volume

-- Statements to prove
theorem part1 :
  profit 50 = 10000 :=
by
  sorry

theorem part2 :
  let max_profit_price := 60
  let max_profit := 12000
  max_profit = (fun price => max (profit price) 0) 60 :=
by
  sorry

end part1_part2_l549_549220


namespace sum_prime_factors_10_to_40_is_77_l549_549550

def prime_factors (n : ℕ) : List ℕ := sorry -- Function to find the prime factors.

noncomputable def sum_prime_factors_even_10_40 : ℕ := 
  let even_numbers := List.range' 10 31 |>.filter (λ x => x % 2 = 0)
  let all_prime_factors := even_numbers.bind prime_factors
  let unique_prime_factors := all_prime_factors.erase_dup
  unique_prime_factors.sum

theorem sum_prime_factors_10_to_40_is_77 : 
  sum_prime_factors_even_10_40 = 77 := sorry

end sum_prime_factors_10_to_40_is_77_l549_549550


namespace area_relation_of_TRIANGLE_ABC_l549_549095

noncomputable def triangle_area (A B C : Point) : Real := 
  sorry

noncomputable def circumcenter (A B C : Point) : Point := 
  sorry

noncomputable def orthocenter (A B C : Point) : Point := 
  sorry

theorem area_relation_of_TRIANGLE_ABC (A B C : Point)
  (h : acute_triangle A B C) 
  (O := circumcenter A B C) 
  (H := orthocenter A B C) :
  let area_AOH := triangle_area A O H
  let area_BOH := triangle_area B O H
  let area_COH := triangle_area C O H
  area_AOH = area_BOH + area_COH ∨
  area_BOH = area_AOH + area_COH ∨
  area_COH = area_AOH + area_BOH :=
sorry

end area_relation_of_TRIANGLE_ABC_l549_549095


namespace sum_of_three_numbers_l549_549172

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 20 :=
sorry

end sum_of_three_numbers_l549_549172


namespace dolls_in_dollhouses_l549_549841

theorem dolls_in_dollhouses :
  let total_ways := Nat.choose 7 2 * 6 * Nat.factorial 5 in
  total_ways = 15120 := by
  sorry

end dolls_in_dollhouses_l549_549841


namespace correct_choice_l549_549252

theorem correct_choice : 2 ∈ ({0, 1, 2} : Set ℕ) :=
sorry

end correct_choice_l549_549252


namespace find_sine_plus_cosine_l549_549391

theorem find_sine_plus_cosine (θ b : ℝ) (hθ: 0 < θ ∧ θ < π / 2) (h: cos (2 * θ) = b) : 
  sin θ + cos θ = sqrt (2 - b) :=
by
  sorry

end find_sine_plus_cosine_l549_549391


namespace area_triangle_AOB_l549_549342

theorem area_triangle_AOB (a b k m x1 y1 x2 y2 : ℝ) (h_ellipse_eq : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (e : ℝ) (h_eccentricity : e = 1/2) (hp1 : (1:ℝ), (3:ℝ)/2)
  (h_AB : ∀ A B, ∃ l : Line, l = Line.mk k m ∧ l.intersects (A, B)) 
  (h_PQ : ∀ A B, ∃ P Q, P = [A.x / a, A.y / b] ∧ Q = [B.x / a, B.y / b])
  (h_circle_origin : ∀ P Q, (P.1 * Q.1 / 4 + P.2 * Q.2 / 3 = 0)) :
  ∃ S : ℝ, S = √3 := 
begin
  sorry
end

end area_triangle_AOB_l549_549342


namespace find_c_l549_549273

noncomputable def value_of_c (a b c : ℝ) : Prop :=
  (∀ y, (-3) = a * (y + 1) ^ 2 + b * (y + 1) ∧ -3 ∈ (λ c : ℝ, ∃ a b : ℝ, -3 = a * (y + 1) ^ 2 + b * (y + 1) -3 )) ∧ 
  (a * (1 + 1)^2 - 3 = -1) ∧ 
  (c = a * (0 + 1)^2 - 3)

theorem find_c :
  ∃ (a b c : ℝ), value_of_c a b c :=
by
  simp [value_of_c]
  use [0.5, _, -2.5] -- values for a and c
  split
  · -- prove vertex condition
    intros y
    sorry
  · -- prove passing through (-1, 1)
    sorry
  · -- prove c calculation
    sorry

end find_c_l549_549273


namespace pieces_on_third_day_impossibility_of_2014_pieces_l549_549105

-- Define the process of dividing and eating chocolate pieces.
def chocolate_pieces (n : ℕ) : ℕ :=
  9 + 8 * n

-- The number of pieces after the third day.
theorem pieces_on_third_day : chocolate_pieces 3 = 25 :=
sorry

-- It's impossible for Maria to have exactly 2014 pieces on any given day.
theorem impossibility_of_2014_pieces : ∀ n : ℕ, chocolate_pieces n ≠ 2014 :=
sorry

end pieces_on_third_day_impossibility_of_2014_pieces_l549_549105


namespace probability_exact_four_out_of_twelve_dice_is_approx_0_089_l549_549932

noncomputable def dice_probability_exact_four_six : ℝ :=
  let p := (1/6 : ℝ)
  let q := (5/6 : ℝ)
  (Nat.choose 12 4) * (p ^ 4) * (q ^ 8)

theorem probability_exact_four_out_of_twelve_dice_is_approx_0_089 :
  abs (dice_probability_exact_four_six - 0.089) < 0.001 :=
sorry

end probability_exact_four_out_of_twelve_dice_is_approx_0_089_l549_549932


namespace intervals_monotonic_increasing_perimeter_triangle_l549_549722

open real

-- Define the vectors
def a (x : ℝ) : ℝ × ℝ :=
  (2 * sin x, cos (2 * x))

def b (x : ℝ) : ℝ × ℝ :=
  (cos x, sqrt 3)

-- Define the function f
def f (x : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Conditions for the proof
def A := π/6
def C := 2 * π / 3

noncomputable def f_half_A := f (A / 2)

-- Theorem for monotonic increasing intervals
theorem intervals_monotonic_increasing :
  ∀ (k : ℤ), ∀ (x : ℝ), (k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12) →
  (d f x / dx ≥ 0) :=
sorry

-- Theorem for perimeter of the triangle
theorem perimeter_triangle :
  f_half_A = 2 → C = 2 * π / 3 →
  (circle_area = 4 * π) → ∃ a b c : ℝ, (a + b + c = 4 + 2 * sqrt 3) :=
sorry

end intervals_monotonic_increasing_perimeter_triangle_l549_549722


namespace singers_and_dancers_assignment_l549_549927

theorem singers_and_dancers_assignment :
  let total_people := 9
  let can_sing := 7
  let can_dance := 5
  let both := 3
  let only_sing := can_sing - both
  let only_dance := can_dance - both
  let ways_if_neither_both := only_sing * only_dance
  let ways_if_one_both := (Mathlib.Combinatorics.choose both 1) * (only_sing + only_dance)
  let ways_if_both := (Mathlib.Combinatorics.choose both 2) * 2
  let total_ways := ways_if_neither_both + ways_if_one_both + ways_if_both
  total_people = 9 ∧ can_sing = 7 ∧ can_dance = 5 → total_ways = 32 :=
by {
  sorry
}

end singers_and_dancers_assignment_l549_549927


namespace binomial_p_value_l549_549499

noncomputable def binomial_expected_value (n : ℕ) (p : ℝ) : ℝ := n * p

theorem binomial_p_value (p : ℝ) : (binomial_expected_value 18 p = 9) → p = 1/2 :=
by
  intro h
  sorry

end binomial_p_value_l549_549499


namespace g_sum_l549_549814

def g (a b : ℚ) : ℚ :=
  if a^b + b ≤ 5 then (a^2 * b - a + 1) / (3 * a)
  else (a * b^2 - b + 1) / ((-3) * b)

theorem g_sum :
  g 1 2 + g 1 3 = 5 / 3 :=
by
  sorry

end g_sum_l549_549814


namespace complex_power_l549_549582

theorem complex_power : (1 - Complex.i)^4 = -4 :=
by
  sorry

end complex_power_l549_549582


namespace distance_from_circle_center_to_line_l549_549798

noncomputable def circle_center_distance_to_line : ℝ :=
  let center_x := -1
  let center_y := 0
  let line_a := 1
  let line_b := 1
  let line_c := -7 in
  (abs (line_a * center_x + line_b * center_y + line_c)) / (sqrt (line_a^2 + line_b^2))

theorem distance_from_circle_center_to_line :
  circle_center_distance_to_line = 4 * sqrt 2 :=
by
  sorry

end distance_from_circle_center_to_line_l549_549798


namespace ratio_of_sum_of_terms_l549_549710

variable {α : Type*}
variable [Field α]

def geometric_sequence (a : ℕ → α) := ∃ r, ∀ n, a (n + 1) = r * a n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) := S 0 = a 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem ratio_of_sum_of_terms (a : ℕ → α) (S : ℕ → α)
  (h_geom : geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h : S 8 / S 4 = 4) :
  S 12 / S 4 = 13 :=
by
  sorry

end ratio_of_sum_of_terms_l549_549710


namespace work_boots_cost_correct_l549_549072

def cost_of_nikes : ℝ := 150
def tax_rate : ℝ := 0.10
def total_paid : ℝ := 297

def cost_of_work_boots_before_tax (cost_of_nikes : ℝ) (tax_rate : ℝ) (total_paid : ℝ) : ℝ :=
  let total_cost_of_nikes := cost_of_nikes * (1 + tax_rate)
  let total_cost_of_work_boots := total_paid - total_cost_of_nikes
  total_cost_of_work_boots / (1 + tax_rate)

theorem work_boots_cost_correct :
  cost_of_work_boots_before_tax cost_of_nikes tax_rate total_paid = 120 := by
  sorry

end work_boots_cost_correct_l549_549072


namespace major_axis_length_of_C1_asymptotic_lines_of_C2_eccentricities_reciprocal_foci_positions_l549_549735

noncomputable def major_axis_length (c1 : ℝ × ℝ → Prop) : ℝ :=
  2 * real.sqrt 5

noncomputable def asymptotic_lines (c2 : ℝ × ℝ → Prop) : set (ℝ × ℝ) :=
  {p | p.2 = (1/2) * p.1 ∨ p.2 = -(1/2) * p.1}

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ :=
  real.sqrt (1 - (a^2 / b^2))

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  real.sqrt (1 + (b^2 / a^2))

noncomputable def foci_ellipse (c1 : ℝ × ℝ → Prop) : set (ℝ × ℝ) :=
  {(0, 2), (0, -2)}

noncomputable def foci_hyperbola (c2 : ℝ × ℝ → Prop) : set (ℝ × ℝ) :=
  {(real.sqrt 5, 0), (-real.sqrt 5, 0)}

-- Ellipse and Hyperbola Definitions
def C1 (p : ℝ × ℝ) : Prop := p.1^2 + (p.2^2 / 5) = 1
def C2 (p : ℝ × ℝ) : Prop := (p.1^2 / 4) - p.2^2 = 1

-- Theorem Statements
theorem major_axis_length_of_C1 : major_axis_length C1 = 2 * real.sqrt 5 := sorry

theorem asymptotic_lines_of_C2 : ∀ p, asymptotic_lines C2 p ↔ p.2 = (1 / 2) * p.1 ∨ p.2 = -(1 / 2) * p.1 := sorry

theorem eccentricities_reciprocal : ellipse_eccentricity 1 (real.sqrt 5) * hyperbola_eccentricity 2 1 = 1 := sorry

theorem foci_positions : foci_ellipse C1 = {(0, 2), (0, -2)} ∧ foci_hyperbola C2 = {(real.sqrt 5, 0), (-real.sqrt 5, 0)} := sorry

end major_axis_length_of_C1_asymptotic_lines_of_C2_eccentricities_reciprocal_foci_positions_l549_549735


namespace sufficient_but_not_necessary_x_eq_0_l549_549774

theorem sufficient_but_not_necessary_x_eq_0 (x : ℝ) :
  (x = 0 → x^2 - 2 * x = 0) ∧ ¬(∀ (x : ℝ), (x^2 - 2 * x = 0) → (x = 0)) :=
by
  split
  {
    intro h,
    rw h,
    norm_num,
  }
  {
    intro h,
    specialize h 2,
    norm_num at h,
    apply h,
  }

end sufficient_but_not_necessary_x_eq_0_l549_549774


namespace existence_of_solutions_l549_549207

noncomputable def check_solution_set_size (n : ℕ) (k : ℕ) (a : ℤ) (n_i : Fin k → ℤ) :=
  ∀ i j, i ≠ j → Int.gcd (n_i i) (n_i j) = 1 →
  (∀ i, a^n_i i ≡ 1 [MOD n_i i]) →
  (∀ i, X^(a-1) ≡ 0 [MOD n_i i]) →
  ∃ x, x > 1 ∧ a^x ≡ 1 [MOD x] ∧ (∃ num_sol, num_sol ≥ (2^(k+1) - 2))

theorem existence_of_solutions (k : ℕ) (a : ℤ) (n_i : Fin k → ℤ) :
  check_solution_set_size k (Fin k → ℤ) a n_i := by
  sorry

end existence_of_solutions_l549_549207


namespace savings_relationship_l549_549211

def combined_salary : ℝ := 3000
def salary_A : ℝ := 2250
def salary_B : ℝ := combined_salary - salary_A
def savings_A : ℝ := 0.05 * salary_A
def savings_B : ℝ := 0.15 * salary_B

theorem savings_relationship : savings_A = 112.5 ∧ savings_B = 112.5 := by
  have h1 : salary_B = 750 := by sorry
  have h2 : savings_A = 0.05 * 2250 := by sorry
  have h3 : savings_B = 0.15 * 750 := by sorry
  have h4 : savings_A = 112.5 := by sorry
  have h5 : savings_B = 112.5 := by sorry
  exact And.intro h4 h5

end savings_relationship_l549_549211


namespace bisect_square_area_l549_549649

theorem bisect_square_area
  (A B C C1 C2 : Point)
  (square_AC : is_square A C C1)
  (square_BC : is_square B C C2)
  (square_AB : is_square A B X) :
  bisects_area C1 C2 square_AB :=
by sorry

end bisect_square_area_l549_549649


namespace angle_equality_l549_549821

-- Define a structure for the geometric setup
structure Triangle (α : Type) [PlaneGeometry α] :=
  (A B C : α)

def is_circumcenter {α : Type} [PlaneGeometry α] (O : α) (T : Triangle α) : Prop :=
  ∀ P ∈ {T.A, T.B, T.C}, dist O P = dist O T.A

def is_foot_of_altitude {α : Type} [PlaneGeometry α] (H : α) (A : α) (BC_line : Line α) : Prop :=
  ∃ P Q : α, is_collinear P Q H ∧ P ∈ BC_line ∧ Q ∈ BC_line ∧ H ∈ line_through A

variables {α : Type} [PlaneGeometry α]

theorem angle_equality (T : Triangle α) (O H : α)
  (h₁ : is_circumcenter O T)
  (h₂ : is_foot_of_altitude H T.A (line_through T.B T.C)) :
  ∠(T.B, O, T.A) = ∠(T.C, A, H) :=
sorry

end angle_equality_l549_549821


namespace problem_number_eq_7_5_l549_549117

noncomputable def number : ℝ := 7.5

theorem problem_number_eq_7_5 :
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = number :=
by
  sorry

end problem_number_eq_7_5_l549_549117


namespace magnitude_of_vector_l549_549385

-- Define vectors a and b
def a : ℝ × ℝ := (Real.cos 5, Real.sin 5)
def b : ℝ × ℝ := (Real.cos 65, Real.sin 65)

-- Define the vector addition and scalar multiplication
def c := (a.1 + 2 * b.1, a.2 + 2 * b.2)

-- Placeholder for the proof
theorem magnitude_of_vector :
  Real.sqrt ((c.1)^2 + (c.2)^2) = Real.sqrt 7 :=
by
  sorry

end magnitude_of_vector_l549_549385


namespace find_value_of_a_l549_549772

theorem find_value_of_a (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h1 : a^b = b^a) (h2 : b = 4 * a) : 
  a = real.cbrt 4 :=
by
  sorry

end find_value_of_a_l549_549772


namespace min_colors_required_l549_549042

-- Define predicate for the conditions
def conditions (n : ℕ) (m : ℕ) (k : ℕ)(Paint : ℕ → Set ℕ) : Prop := 
  (∀ S : Finset ℕ, S.card = n → (∃ c ∈ ⋃ p ∈ S, Paint p, c ∈ S)) ∧ 
  (∀ c, ¬ (∀ i ∈ (Finset.range m).1, c ∈ Paint i))

-- The main theorem statement
theorem min_colors_required :
  ∀ (Paint : ℕ → Set ℕ), conditions 20 100 21 Paint → 
  ∃ k, conditions 20 100 k Paint ∧ k = 21 :=
sorry

end min_colors_required_l549_549042


namespace solve_inequality_system_l549_549989

theorem solve_inequality_system (y : ℝ) :
  (2 * (y + 1) < 5 * y - 7) ∧ ((y + 2) / 2 < 5) ↔ (3 < y) ∧ (y < 8) := 
by
  sorry

end solve_inequality_system_l549_549989


namespace net_profit_calculation_l549_549473

def original_purchase_price : ℝ := 80000
def annual_property_tax_rate : ℝ := 0.012
def annual_maintenance_cost : ℝ := 1500
def annual_mortgage_interest_rate : ℝ := 0.04
def selling_profit_rate : ℝ := 0.20
def broker_commission_rate : ℝ := 0.05
def years_of_ownership : ℕ := 5

noncomputable def net_profit : ℝ :=
  let selling_price := original_purchase_price * (1 + selling_profit_rate)
  let brokers_commission := original_purchase_price * broker_commission_rate
  let total_property_tax := original_purchase_price * annual_property_tax_rate * years_of_ownership
  let total_maintenance_cost := annual_maintenance_cost * years_of_ownership
  let total_mortgage_interest := original_purchase_price * annual_mortgage_interest_rate * years_of_ownership
  let total_costs := brokers_commission + total_property_tax + total_maintenance_cost + total_mortgage_interest
  (selling_price - original_purchase_price) - total_costs

theorem net_profit_calculation : net_profit = -16300 := by
  sorry

end net_profit_calculation_l549_549473


namespace most_likely_outcomes_l549_549685

noncomputable def probability_boy_or_girl : ℚ := 1 / 2

noncomputable def probability_all_boys (n : ℕ) : ℚ := probability_boy_or_girl^n

noncomputable def probability_all_girls (n : ℕ) : ℚ := probability_boy_or_girl^n

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_3_girls_2_boys : ℚ := binom 5 3 * probability_boy_or_girl^5

noncomputable def probability_3_boys_2_girls : ℚ := binom 5 2 * probability_boy_or_girl^5

theorem most_likely_outcomes :
  probability_3_girls_2_boys = 5/16 ∧
  probability_3_boys_2_girls = 5/16 ∧
  probability_all_boys 5 = 1/32 ∧
  probability_all_girls 5 = 1/32 ∧
  (5/16 > 1/32) :=
by
  sorry

end most_likely_outcomes_l549_549685


namespace number_of_students_l549_549851

noncomputable def num_students_total (jelly_beans_brought : ℕ) (jelly_beans_left : ℕ) (boys_more_than_girls : ℕ) (total_jelly_beans_given : ℕ) (distributed_jelly_beans : (ℕ → ℕ → ℕ) → (ℕ → ℕ → ℕ) → ℕ) := 
  distributed_jelly_beans (λ girls boys, 3 * girls ^ 2) (λ girls boys, 2 * (boys ^ 2)) = total_jelly_beans_given

theorem number_of_students (x : ℕ) :
  let girls := x,
      boys := x + 4,
      jelly_beans_brought := 420,
      jelly_beans_left := 6,
      boys_more_than_girls := 4,
      total_jelly_beans_given := 414 in

  num_students_total jelly_beans_brought jelly_beans_left boys_more_than_girls total_jelly_beans_given
    (λ girls boys, (3 * girls ^ 2 + 2 * (boys ^ 2)) = total_jelly_beans_given) →
  (girls + boys = 18) :=
by
  sorry

end number_of_students_l549_549851


namespace product_of_roots_l549_549646

theorem product_of_roots (a b c d x : ℝ) 
  (h_eq : 4 * x^3 - 3 * x^2 - 40 * x + 36 = 0) :
  let pqr := -(36 / 4) in pqr = -9 := by
  sorry

end product_of_roots_l549_549646


namespace travis_takes_home_money_l549_549184

-- Define the conditions
def total_apples : ℕ := 10000
def apples_per_box : ℕ := 50
def price_per_box : ℕ := 35

-- Define the main theorem to be proved
theorem travis_takes_home_money : (total_apples / apples_per_box) * price_per_box = 7000 := by
  sorry

end travis_takes_home_money_l549_549184


namespace T_n_formula_l549_549734

-- Define the given sequence sum S_n
def S (n : ℕ) : ℚ := (n^2 : ℚ) / 2 + (3 * n : ℚ) / 2

-- Define the general term a_n for the sequence {a_n}
def a (n : ℕ) : ℚ := if n = 1 then 2 else n + 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := a (n + 2) - a n + 1 / (a (n + 2) * a n)

-- Define the sum of the first n terms of the sequence {b_n}
def T (n : ℕ) : ℚ := 2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3))

-- Prove the equality of T_n with the given expression
theorem T_n_formula (n : ℕ) : T n = 2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3)) := sorry

end T_n_formula_l549_549734


namespace mean_of_five_numbers_l549_549168

theorem mean_of_five_numbers (S : ℚ) (n : ℕ) (h1 : S = 3/4) (h2 : n = 5) :
  (S / n) = 3/20 :=
by
  rw [h1, h2]
  sorry

end mean_of_five_numbers_l549_549168


namespace max_rational_products_50x50_table_l549_549476

/-- Oleg drew an empty 50 × 50 table and wrote a non-zero number at the top of each column
    and to the left of each row. All 100 written numbers are distinct, with 50 of them being 
    rational and the remaining 50 irrational. Then, in each cell of the table, he wrote the 
    product of the numbers written next to its row and column. -/
noncomputable def maxRationalProducts : ℕ :=
  let n := 50 in
  let rational_numbers := 25 in
  rational_numbers * rational_numbers

theorem max_rational_products_50x50_table :
  maxRationalProducts = 625 :=
by
  unfold maxRationalProducts
  norm_num
  sorry

end max_rational_products_50x50_table_l549_549476


namespace mean_of_solutions_correct_l549_549681

def mean_of_solutions : ℚ :=
  let a := 1
  let b := 4
  let c := -10
  let sol1 := 0
  let sol2 := (-b + real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let sol3 := (-b - real.sqrt (b^2 - 4 * a * c)) / (2 * a)
in (sol1 + sol2 + sol3) / 3

theorem mean_of_solutions_correct :
  mean_of_solutions = -4/3 :=
by
  sorry

end mean_of_solutions_correct_l549_549681


namespace orthocenter_coincides_with_circumcenter_l549_549153

noncomputable def triangle (α β γ : Type) [inner_product_space ℝ α] [inner_product_space ℝ β] [inner_product_space ℝ γ] : Type :=
{ a : α, b : β, c : γ }

variables {α β γ : Type} [inner_product_space ℝ α] [inner_product_space ℝ β] [inner_product_space ℝ γ]

structure circumcircle (α : Type) [inner_product_space ℝ α] :=
(center : α)
(radius : ℝ)

structure excircle (α : Type) [inner_product_space ℝ α] :=
(center : α)
(radius : ℝ)
(tangent_point_ab : α)
(tangent_point_ac_extension : α)
(tangent_point_bc_extension : α)

theorem orthocenter_coincides_with_circumcenter
  (A B C : α) (C' A' B' : α)
  (O : α)
  (circ : circumcircle α)
  (exc : excircle α)
  (circ_center : circ.center = O)
  (exc_center : exc.center = O)
  (circ_radius : circ.radius = exc.radius)
  (tangent_AB : exc.tangent_point_ab = C')
  (tangent_AC_ext : exc.tangent_point_ac_extension = A')
  (tangent_BC_ext : exc.tangent_point_bc_extension = B') :
  ∃ H : α, is_orthocenter (triangle.mk A' B' C') H ∧ H = O :=
sorry

end orthocenter_coincides_with_circumcenter_l549_549153


namespace possible_values_of_b2_l549_549999

theorem possible_values_of_b2 (b : ℕ → ℕ) (h1 : b 1 = 1001) (h2 : b 2 < 1001) (h3 : b 2007 = 0)
  (h_seq : ∀ n, n ≥ 1 → b (n + 2) = (b (n + 1) - b n).natAbs) :
  { b2 | b2 < 1001 ∧ ∃ b, b 1 = 1001 ∧ b 2 = b2 ∧ b 2007 = 0 ∧ ∀ n ≥ 1, b (n + 2) = (b (n + 1) - b n).natAbs}.card = 360 := sorry

end possible_values_of_b2_l549_549999


namespace nail_polishes_total_l549_549431

theorem nail_polishes_total :
  let k := 25
  let h := k + 8
  let r := k - 6
  h + r = 52 :=
by
  sorry

end nail_polishes_total_l549_549431


namespace square_distance_l549_549876

theorem square_distance (a b c d e f: ℝ) 
  (side_length : ℝ)
  (AB : a = 0 ∧ b = side_length)
  (BC : c = side_length ∧ d = 0)
  (BE_dist : (a - b)^2 + (b - b)^2 = 25)
  (AE_dist : a^2 + (c - b)^2 = 144)
  (DF_dist : (d)^2 + (d)^2 = 25)
  (CF_dist : (d - c)^2 + e^2 = 144) :
  (f - d)^2 + (e - a)^2 = 578 :=
by
  -- Required to bypass the proof steps
  sorry

end square_distance_l549_549876


namespace solution1_solution2_solution3_l549_549871

noncomputable def problem1 : Real :=
3.5 * 101

noncomputable def problem2 : Real :=
11 * 5.9 - 5.9

noncomputable def problem3 : Real :=
88 - 17.5 - 12.5

theorem solution1 : problem1 = 353.5 :=
by
  sorry

theorem solution2 : problem2 = 59 :=
by
  sorry

theorem solution3 : problem3 = 58 :=
by
  sorry

end solution1_solution2_solution3_l549_549871


namespace number_of_teachers_in_school_l549_549977

-- Define the conditions and the proof statement in Lean 4

variables (T_total S_sample S_students : ℕ) (S_teachers T_teachers : ℕ)

-- Define the conditions 
def conditions : Prop := 
  T_total = 2400 ∧ 
  S_sample = 120 ∧ 
  S_students = 110 ∧ 
  S_teachers = S_sample - S_students ∧ 
  S_teachers = 10

-- The assertion to be proved using the conditions
theorem number_of_teachers_in_school (h : conditions) : T_teachers = 200 :=
sorry -- Proof is omitted

end number_of_teachers_in_school_l549_549977


namespace determine_m_l549_549783

theorem determine_m (m : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 ↔ m * (x - 1) > x^2 - x) → m = 2 :=
sorry

end determine_m_l549_549783


namespace monitor_length_34_inch_l549_549990

noncomputable def monitor_horizontal_length (diag : ℝ) : ℝ :=
  let aspect_ratio := (16 : ℝ) / 9
  let diag_squared := 16^2 + 9^2
  let scale_factor := diag / Real.sqrt diag_squared
  aspect_ratio * scale_factor

theorem monitor_length_34_inch : monitor_horizontal_length 34 ≈ 29.63 :=
by
  sorry

end monitor_length_34_inch_l549_549990


namespace relationship_among_numbers_l549_549801

-- Define the smallest natural number
def a : ℕ := 0

-- Define the smallest odd number
def b : ℕ := 1

-- Define the smallest even number
def c : ℕ := 2

-- Define the smallest prime number
def d : ℕ := 2

-- Define the smallest composite number
def e : ℕ := 4

-- Prove the relationship among these numbers
theorem relationship_among_numbers : a < b ∧ b < c ∧ c = d ∧ d < e := by
  unfold a b c d e
  simp
  exact sorry

end relationship_among_numbers_l549_549801


namespace total_tourists_is_1008_l549_549979

-- Definitions based on conditions

def hourlyService (start time : Nat) (end time : Nat) : List Nat :=
  List.range (end time - start time + 1)

def touristsAtStart (t : Nat) : Nat :=
  if t = 9 then 120 else 0

def decrementTourists : Nat := 2

def numberOfTrips (start time end time : Nat) : Nat :=
  end time - start time + 1

def tourists (startT : Nat) (startTourists : Nat) (decrement : Nat) : List Nat :=
  List.map (λ n => startTourists - n * decrement) (hourlyService startT (startT + numberOfTrips 9 17 - 1))

def totalTourists (touristList : List Nat) : Nat :=
  List.sum touristList

-- Problem statement to prove

theorem total_tourists_is_1008 :
  totalTourists (tourists 9 120 decrementTourists) = 1008 :=
  sorry

end total_tourists_is_1008_l549_549979


namespace solve_for_x_l549_549592

theorem solve_for_x : ∃ x : ℝ, (0.65 * x = 0.20 * 747.50) ∧ x = 230 :=
by
  use 230
  split
  {
    -- ensuring the Lean statement compiles, providing the necessary condition
    calc
    0.65 * 230 = 0.65 * 230 : by rfl
    ... = 149.50 : by norm_num [0.65 * 230]
    ... = 0.20 * 747.50 : by norm_num [0.20 * 747.50]
  }
  sorry

end solve_for_x_l549_549592


namespace subset_div_chain_l549_549819

theorem subset_div_chain (m n : ℕ) (h_m : m > 0) (h_n : n > 0) (S : Finset ℕ) (hS : S.card = (2^m - 1) * n + 1) (hS_subset : S ⊆ Finset.range (2^(m) * n + 1)) :
  ∃ (a : Fin (m+1) → ℕ), (∀ i, a i ∈ S) ∧ (∀ k : ℕ, k < m → a k ∣ a (k + 1)) :=
sorry

end subset_div_chain_l549_549819


namespace Buratino_overpay_l549_549063

-- Definitions based on the given conditions
def loan_amount : ℝ := 100
def duration : ℝ := 128
def daily_rate_option1 : ℝ := 0.01
def daily_rate_option2 : ℝ := 0.02

-- Defining the total debt for Option 1 using the compound interest formula:
def total_debt_option1 : ℝ := loan_amount * (1 + daily_rate_option1)^duration

-- Defining the total debt for Option 2 using the simple interest formula:
def total_debt_option2 : ℝ := loan_amount * (1 + daily_rate_option2 * duration)

-- The difference between the total debts for both options
def debt_difference : ℝ := total_debt_option1 - total_debt_option2

-- Prove the debt difference equals 5 coins.
theorem Buratino_overpay : debt_difference ≈ 5 := by
  sorry

end Buratino_overpay_l549_549063


namespace m_leq_neg_one_l549_549030

theorem m_leq_neg_one (m : ℝ) :
    (∀ x : ℝ, 2^(-x) + m > 0 → x ≤ 0) → m ≤ -1 :=
by
  sorry

end m_leq_neg_one_l549_549030


namespace combined_area_l549_549482

noncomputable def combined_area_of_triangles
  (AB CD : ℝ) (M : ℝ × ℝ) (E F : ℝ × ℝ)
  (area_tr1 area_tr2 : ℝ) : ℝ :=
  area_tr1 + area_tr2

theorem combined_area
  (rectangle : Type*)
  (AB BC : ℝ) (h_ab : AB = 10) (h_bc : BC = 8)
  (M : ℝ × ℝ) (h_Mm : M = (AC / 2, 0))
  (E : ℝ × ℝ) (F : ℝ × ℝ)
  (h_EMperp : is_perpendicular M E AC)
  (h_FMperp : is_perpendicular M F AC)
  (area_AME area_BMF : ℝ)
  (h_area_AME : area_AME = 10)
  (h_area_BMF : area_BMF = 10) :
  combined_area_of_triangles AB BC M E F area_AME area_BMF = 41 / 2 := sorry

end combined_area_l549_549482


namespace sequence_value_l549_549664

theorem sequence_value : 
  ∃ (x y r : ℝ), 
    (4096 * r = 1024) ∧ 
    (1024 * r = 256) ∧ 
    (256 * r = x) ∧ 
    (x * r = y) ∧ 
    (y * r = 4) ∧  
    (4 * r = 1) ∧ 
    (x + y = 80) :=
by
  sorry

end sequence_value_l549_549664


namespace find_hypotenuse_of_right_angle_triangle_l549_549677

theorem find_hypotenuse_of_right_angle_triangle
  (PR : ℝ) (angle_QPR : ℝ)
  (h1 : PR = 16)
  (h2 : angle_QPR = Real.pi / 4) :
  ∃ PQ : ℝ, PQ = 16 * Real.sqrt 2 :=
by
  sorry

end find_hypotenuse_of_right_angle_triangle_l549_549677


namespace trapezoidal_ratio_l549_549397

noncomputable def ratio_of_bases (a : ℝ) : ℝ := a / (a^2)

theorem trapezoidal_ratio :
  (∃ (a : ℝ), (a > 0) ∧ (ratio_of_bases a = 3 / 16) ∧ (∀ AC, AC = 2 → AC^2 = (a - (a^2 - a)/2)^2 + a^2)) :=
begin
  use [4 / 3],
  split,
  linarith,
  split,
  { calc
    ratio_of_bases (4 / 3)
        = (4 / 3) / ((4 / 3)^2) : rfl
    ... = (4 / 3) / (16 / 9) : by norm_num
    ... = (4 * 3) / 16 : by field_simp
    ... = 12 / 16 : by norm_num
    ... = 3 / 4 : by norm_num, },
  {
    intros AC h,
    rw h,
    calc (4 : ℝ) 
        = ((4 / 3) - ((4 / 3)^2 - 4 / 3)/2)^2 + (4 / 3)^2 
        : by sorry
  }
end

end trapezoidal_ratio_l549_549397


namespace circle_radius_l549_549683

theorem circle_radius : ∀ (x y : ℝ), x^2 + 10*x + y^2 - 8*y + 25 = 0 → False := sorry

end circle_radius_l549_549683


namespace sqrt_of_16_l549_549918

theorem sqrt_of_16 (x : ℝ) (hx : x^2 = 16) : x = 4 ∨ x = -4 := 
by
  sorry

end sqrt_of_16_l549_549918


namespace waiter_tips_l549_549636

theorem waiter_tips (total_customers tipping_customers : ℕ) (total_earned_per_tip : ℕ) :
  total_customers = 7 → tipping_customers = total_customers - 5 → total_earned_per_tip = 6 → 
  ∃ (earn_per_customer : ℕ), earn_per_customer = total_earned_per_tip / tipping_customers ∧ earn_per_customer = 3 :=
begin
  intros h1 h2 h3,
  use 3,
  split,
  { sorry }, -- We would actually perform division here in proof
  { sorry }  -- We would actually show the result is 3
end

end waiter_tips_l549_549636


namespace smallest_number_in_set_l549_549802

theorem smallest_number_in_set (s : Set ℤ) (h : s = {0, -1, 1, 2}) : ∃ x, x ∈ s ∧ ∀ y ∈ s, x ≤ y :=
by
  rw h
  use -1
  split
  · simp
  · intros y hy
    simp at hy
    linarith

end smallest_number_in_set_l549_549802


namespace jessica_adjusted_mean_score_l549_549811

def drop_lowest_and_mean (scores : List ℝ) : ℝ :=
  let filtered_scores := List.erase scores (scores.min' (by decide))
  filtered_scores.sum / filtered_scores.length

theorem jessica_adjusted_mean_score :
  drop_lowest_and_mean [85, 90, 93, 87, 92, 95] = 91.4 := by
  sorry

end jessica_adjusted_mean_score_l549_549811


namespace triangle_area_formula_l549_549543

noncomputable def area_of_triangle (R a β : ℝ) : ℝ :=
  let sin_2β := 2 * Real.sin β * Real.cos β
  let term1 := (a^2 * sin_2β) / 4
  let term2 := (a * (Real.sin β)^2) / 2 * Real.sqrt(4 * R^2 - a^2)
  term1 + term2

theorem triangle_area_formula (R a β : ℝ) :
  triangle_area R a β = (a^2 * (2 * Real.sin β * Real.cos β)) / 4 + 
                         (a * (Real.sin β)^2) / 2 * Real.sqrt(4 * R^2 - a^2) :=
by
  sorry

end triangle_area_formula_l549_549543


namespace fourth_derivative_l549_549676

noncomputable def y (x : ℝ) : ℝ := Real.exp (x / 2) * Real.sin (2 * x)

theorem fourth_derivative :
  deriv (deriv (deriv (deriv y))) = λ x, 
    (161 / 16) * Real.exp (x / 2) * Real.sin (2 * x) - 
    15 * Real.exp (x / 2) * Real.cos (2 * x) :=
by sorry

end fourth_derivative_l549_549676


namespace odd_function_piecewise_l549_549345

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then x * (x + 1) else x * (1 - x)

theorem odd_function_piecewise (f : ℝ → ℝ) (h₁ : ∀ x, f (-x) = - f x) (h₂ : ∀ x ≤ 0, f x = x * (x + 1)) :
  ∀ x, f x = if x ≤ 0 then x * (x + 1) else x * (1 - x) :=
sorry

end odd_function_piecewise_l549_549345


namespace frog_stops_horizontal_side_l549_549981

def Q (x y : ℕ) : ℝ := sorry

axiom Q_boundary_y0 (x : ℕ) : Q x 0 = 1
axiom Q_boundary_y5 (x : ℕ) : Q x 5 = 1
axiom Q_boundary_x0 (y : ℕ) : Q 0 y = 0
axiom Q_boundary_x5 (y : ℕ) : Q 5 y = 0

axiom Q_recur (x y : ℕ) :
  Q x y = 
    (3 / 5 : ℝ) * (1 / 2 * Q x (y - 1) + 1 / 2 * Q x (y + 1)) +
    (2 / 5 : ℝ) * (1 / 2 * Q (x - 1) + 1 / 2 * Q (x + 1))

theorem frog_stops_horizontal_side : Q 2 1 = 1 / 5 :=
by sorry

end frog_stops_horizontal_side_l549_549981


namespace tangency_points_form_cyclic_quadrilateral_l549_549347

theorem tangency_points_form_cyclic_quadrilateral
  (S : ℕ → set (ℝ × ℝ))
  (H1 : ∀ i : ℕ, S i ⊆ metric.ball (0, 0) 1) -- Circles touching externally.
  (H2 : ∀ i : ℕ, S i ∩ S (i + 1) ≠ ∅) -- Prove the points of tangency.
  (H3 : S 5 = S 1) -- Cyclic condition.
  : ∃ A1 A2 A3 A4 : ℝ × ℝ,
    A1 ∈ S 1 ∧ A2 ∈ S 2 ∧ A3 ∈ S 3 ∧ A4 ∈ S 4 ∧
    (metric.angle A1 A2 A3 + metric.angle A3 A4 A1 = π ∨ metric.angle A2 A3 A4 + metric.angle A4 A1 A2 = π) :=
sorry

end tangency_points_form_cyclic_quadrilateral_l549_549347


namespace congruent_triangles_ACD_BDC_l549_549187

open EuclideanGeometry

-- Define points A, B, C, D, and O in a Euclidean space
variables {A B C D O : Point}

-- Conditions from the problem: O being the midpoint and intersection of AB and CD
def midpoint_O_AB_CD (A B C D O : Point) [EuclideanGeometry] : Prop :=
  midpoint O A B ∧ midpoint O C D ∧ (intersect_line_segment O (line_segment A B) ∧ intersect_line_segment O (line_segment C D))

theorem congruent_triangles_ACD_BDC 
  {A B C D O : Point} [EuclideanGeometry] 
  (h_mid : midpoint_O_AB_CD A B C D O) : 
  triangle_congruent (triangle A C D) (triangle B D C) :=
sorry

end congruent_triangles_ACD_BDC_l549_549187


namespace mean_of_five_numbers_is_correct_l549_549166

-- Define the given sum of five numbers as three-quarters
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers, which is 5
def number_of_numbers : ℕ := 5

-- Define the mean calculation from the given sum and number of numbers
def mean_five_numbers (sum : ℚ) (count : ℕ) : ℚ := sum / count

-- Statement to prove: the mean of five numbers given their sum is 3/4 equals 3/20
theorem mean_of_five_numbers_is_correct :
  mean_five_numbers sum_of_five_numbers number_of_numbers = 3 / 20 :=
by
  -- Skipping the proof
  sorry

end mean_of_five_numbers_is_correct_l549_549166


namespace max_gold_pieces_l549_549285

theorem max_gold_pieces (a b c d : ℕ):
  a = 8088 → b = 6066 → c = 4044 → d = 2022 →
  (∀ a b c d, (a - 1 >= 0) → (c - 1 >= 0) → (b + 1 >= 0) → (d + 1 >= 0) → True) →
  (∀ a b c d, (a - 1 >= 0) → (b - 1 >= 0) → (c + 1 >= 0) → (d + 1 >= 0) → True) →
  (∀ a b c d, (a + 1 >= 0) → (b - 1 >= 0) → (c - 1 >= 0) → (d + 1 >= 0) → True) →
  ∃ max_d, max_d = 20218 :=
begin
  sorry
end

end max_gold_pieces_l549_549285


namespace absolute_value_of_fraction_l549_549021

variable {a b : ℝ}

theorem absolute_value_of_fraction :
  a ≠ 0 → b ≠ 0 → a^2 + b^2 = 8 * a * b → abs ((a + b) / (a - b)) = sqrt 15 / 3 :=
by
  intros h1 h2 h3
  sorry

end absolute_value_of_fraction_l549_549021


namespace max_expression_value_l549_549770

theorem max_expression_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 900 ≤ b ∧ b ≤ 1500) :
  (sup {y | ∃ (a : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (b : ℝ) (hb : 900 ≤ b ∧ b ≤ 1500), y = (b - 100) / a}) = 14 / 3 :=
sorry

end max_expression_value_l549_549770


namespace usual_time_is_7_l549_549240

noncomputable def usual_time (S : ℝ) (T' : ℝ) : ℝ :=
  let T := (6 / 7) * (T' - (10 / 60))
  in T

theorem usual_time_is_7 (S : ℝ) (T : ℝ)
  (h1 : T' = T + 10 / 60)
  (h2 : (1 / T) = (6 / 7) / T') : T = 7 :=
by
  sorry

end usual_time_is_7_l549_549240


namespace production_profit_range_l549_549226

theorem production_profit_range (x : ℝ) (t : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 10) (h3 : 0 ≤ t) :
  (200 * (5 * x + 1 - 3 / x) ≥ 3000) → (3 ≤ x ∧ x ≤ 10) :=
sorry

end production_profit_range_l549_549226


namespace divisible_by_8_l549_549456

theorem divisible_by_8 (k : ℤ) : 
  let m := 2 * k + 1 
  let n := 2 * k + 3 
  8 ∣ (7 * m^2 - 5 * n^2 - 2) :=
by 
  let m := 2 * k + 1 
  let n := 2 * k + 3 
  sorry

end divisible_by_8_l549_549456


namespace regular_quadrilateral_pyramid_lateral_faces_equilateral_l549_549978

theorem regular_quadrilateral_pyramid_lateral_faces_equilateral :
  ∀ (P : Pyramid) (pentagon : Polygon),
    isRegularQuadrilateralPyramid P →
    isRegularPentagon (crossSection P) →
    (∀ face ∈ (lateralFaces P), isEquilateral face) :=
by
  sorry

end regular_quadrilateral_pyramid_lateral_faces_equilateral_l549_549978


namespace least_number_divisible_by_38_and_3_remainder_1_exists_l549_549562

theorem least_number_divisible_by_38_and_3_remainder_1_exists :
  ∃ n, n % 38 = 1 ∧ n % 3 = 1 ∧ ∀ m, m % 38 = 1 ∧ m % 3 = 1 → n ≤ m :=
sorry

end least_number_divisible_by_38_and_3_remainder_1_exists_l549_549562


namespace find_a_if_odd_function_l549_549777

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + Real.sqrt (a + x^2))

theorem find_a_if_odd_function (a : ℝ) :
  (∀ x : ℝ, f (-x) a = - f x a) → a = 1 :=
by
  sorry

end find_a_if_odd_function_l549_549777


namespace problem_number_eq_7_5_l549_549115

noncomputable def number : ℝ := 7.5

theorem problem_number_eq_7_5 :
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = number :=
by
  sorry

end problem_number_eq_7_5_l549_549115


namespace limit_derivative_l549_549458

variable {f : ℝ → ℝ}
variable (hf : Differentiable ℝ f)

theorem limit_derivative (hf : Differentiable ℝ f) :
  (lim (λ (Δx : ℝ) (hΔx : Δx ≠ 0), (f (1 + Δx) - f 1) / (3 * Δx)) (0 : ℝ)) =
  (1 / 3) * (deriv f 1) :=
by
  sorry

end limit_derivative_l549_549458


namespace range_of_m_l549_549029

noncomputable def quadratic_fn (x : ℝ) : ℝ := x^2 - 4 * x - 2

theorem range_of_m (m : ℝ) : 
  (∀ x ∈ set.Icc 0 m, quadratic_fn x ∈ set.Icc (-6) (-2)) → 
  m ∈ set.Icc 2 4 :=
by
  sorry

end range_of_m_l549_549029


namespace approx_value_log_sq_expr_l549_549950

noncomputable def log_sq_expr : ℝ := (Real.log10 (8 * Real.log10 1000))^2

theorem approx_value_log_sq_expr : log_sq_expr ≈ 1.9 := by
  -- Proof omitted as specified in the instruction.
  sorry

end approx_value_log_sq_expr_l549_549950


namespace power_of_one_minus_i_eq_neg_4_l549_549578

noncomputable def i : ℂ := complex.I

theorem power_of_one_minus_i_eq_neg_4 : (1 - i)^4 = -4 :=
by
  sorry

end power_of_one_minus_i_eq_neg_4_l549_549578


namespace intersection_point_sums_l549_549145

theorem intersection_point_sums :
  let f := λ x : ℝ, x^3 - 3 * x^2 + 2 * x + 1
  let g := λ y : ℝ, (3 - y) / 3
  (x1 x2 x3 y1 y2 y3 : ℝ),
  (f x1 = (3 - y1) / 3) →
  (f x2 = (3 - y2) / 3) →
  (f x3 = (3 - y3) / 3) →
  x1 + x2 + x3 = 3 ∧ y1 + y2 + y3 = 2 :=
by
  sorry

end intersection_point_sums_l549_549145


namespace volume_of_solid_l549_549786

theorem volume_of_solid :
  let region := {p : ℝ × ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 ∧ 0 ≤ p.3 ∧ p.3 ≤ 1 ∧ (p.1 - p.2)^2 + p.3^2 ≥ 1}
  ∫ p in region, (1 : ℝ) = (Real.pi / 3) - (1 + Real.sqrt 3 / 4) := 
sorry

end volume_of_solid_l549_549786


namespace certain_event_among_options_l549_549554

-- Definition of the proof problem
theorem certain_event_among_options (is_random_A : Prop) (is_random_C : Prop) (is_random_D : Prop) (is_certain_B : Prop) :
  (is_random_A → (¬is_certain_B)) ∧
  (is_random_C → (¬is_certain_B)) ∧
  (is_random_D → (¬is_certain_B)) ∧
  (is_certain_B ∧ ((¬is_random_A) ∧ (¬is_random_C) ∧ (¬is_random_D))) :=
by
  sorry

end certain_event_among_options_l549_549554


namespace product_of_gcd_and_lcm_1440_l549_549314

def product_of_gcd_and_lcm_of_24_and_60 : ℕ :=
  Nat.gcd 24 60 * Nat.lcm 24 60 

/-- The product of the gcd and lcm of 24 and 60 is 1440. -/
theorem product_of_gcd_and_lcm_1440 : product_of_gcd_and_lcm_of_24_and_60 = 1440 := 
by
  sorry

end product_of_gcd_and_lcm_1440_l549_549314


namespace masha_doll_arrangements_l549_549843

theorem masha_doll_arrangements : 
  let dolls := Fin 7
  let dollhouses := Fin 6 in
  (∃ a b : dolls, a ≠ b) ∧ 
  (∃ h : dollhouses, ∀ i ≠ h, ∃! d : dolls, d ≠ a ∧ d ≠ b) ∧ 
  ∃ h : dollhouses, ∃ a b : dolls, a ≠ b ∧ 
  (∀ (d1 d2 : dolls), d1 ≠ a ∧ d1 ≠ b ∧ d2 ≠ a ∧ d2 ≠ b → d1 ≠ d2) →
  21 * 6 * 120 = 15120 :=
by
  sorry

end masha_doll_arrangements_l549_549843


namespace continuous_fn_bounded_on_interval_l549_549480

variable {α : Type} [TopologicalSpace α] {a b : α} (f : α → ℝ) 

theorem continuous_fn_bounded_on_interval {α : Type} [TopologicalSpace α] {a b : α} (h_cont : ContinuousOn f (Set.Icc a b)) (hleq : a ≤ b) :
    ∃ M > 0, ∀ x ∈ Set.Icc a b, |f x| ≤ M :=
by
  sorry

end continuous_fn_bounded_on_interval_l549_549480


namespace solve_problem_l549_549328

noncomputable def t (n : ℕ) : ℕ := 
  nat.gcd n (2^nat.log2 n - 1)

noncomputable def r (n : ℕ) : ℕ := 
  (nat.find (λ k, k > 1 ∧ k ∣ n ∧ k % 2 = 1))

theorem solve_problem :
  ∀ (n : ℕ), n > 0 ∧ ¬(∃ k : ℕ, n = 2^k) →
  n = 3 * t(n) + 5 * r(n) →
  ∃ p : ℕ, nat.prime p ∧ n = 8 * p ∨ n = 60 ∨ n = 100 :=
by
  sorry

end solve_problem_l549_549328


namespace expansion_coefficients_arithmetic_sequence_sqrt_inequality_l549_549589

-- Proof Problem 1
theorem expansion_coefficients_arithmetic_sequence (n : ℕ) : 
  (∀ k : ℕ, k < 3 → let c : ℕ → ℕ := λ n, binomial n k * (1 / (2 * sqrt(k))) in 
    (c 0 + c 2 * (1 / 4)) = 2 * (1 / 2) * c 1) → n = 8 :=
sorry

-- Proof Problem 2
theorem sqrt_inequality (a : ℝ) (h : a > 1) : 
  sqrt(a + 1) - sqrt(a) < sqrt(a) - sqrt(a - 1) :=
sorry

end expansion_coefficients_arithmetic_sequence_sqrt_inequality_l549_549589


namespace quadratic_matches_sin_values_l549_549607

noncomputable def quadratic_function (x : ℝ) : ℝ := - (4 / (Real.pi ^ 2)) * (x ^ 2) + (4 / Real.pi) * x

theorem quadratic_matches_sin_values :
  (quadratic_function 0 = Real.sin 0) ∧
  (quadratic_function (Real.pi / 2) = Real.sin (Real.pi / 2)) ∧
  (quadratic_function Real.pi = Real.sin Real.pi) :=
by
  sorry

end quadratic_matches_sin_values_l549_549607


namespace max_min_values_on_interval_decreasing_interval_l549_549008

namespace Problem

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - (1/2) * x^2

-- Condition for a > 0
variable {a : ℝ} (h_a : 0 < a)

-- Problem 1: Prove maximum and minimum values for a = 1 on [0, 1]
theorem max_min_values_on_interval (x : ℝ) (h_x : 0 ≤ x ∧ x ≤ 1) : 
  f 1 x ≤ 1/2 ∧ f 1 x ≥ -1/54 := by
  sorry

-- Problem 2: Prove that f(x) is decreasing on the interval (0, 1/(6a))
theorem decreasing_interval (x : ℝ) (h_x : 0 < x ∧ x < 1/(6 * a)) : 
  ∀ x, f' a x < 0 := by
  sorry

end Problem

end max_min_values_on_interval_decreasing_interval_l549_549008


namespace marc_spent_total_correct_l549_549104

noncomputable def marcTotalSpent (model_cars_count : ℕ) (model_cars_cost : ℕ) 
  (paint_bottles_count : ℕ) (paint_bottles_cost : ℕ) 
  (paintbrushes_count : ℕ) (paintbrushes_cost : ℕ)
  (display_cases_count : ℕ) (display_cases_cost : ℕ)
  (model_cars_discount : ℝ) (paint_discount : ℕ)
  (gift_card : ℕ) (first_tax_rate : ℝ) (second_tax_rate : ℝ) : ℝ :=
let model_cars_total := model_cars_count * model_cars_cost
let discounted_model_cars_total := model_cars_total - model_cars_discount * model_cars_total
let paint_total := paint_bottles_count * paint_bottles_cost
let discounted_paint_total := paint_total - paint_discount
let paintbrushes_total := paintbrushes_count * paintbrushes_cost
let first_transaction_subtotal := discounted_model_cars_total + discounted_paint_total + paintbrushes_total
let first_transaction_subtotal_after_gift := first_transaction_subtotal - gift_card
let first_transaction_total := first_transaction_subtotal_after_gift * (1 + first_tax_rate)
let display_cases_total := display_cases_count * display_cases_cost
let display_cases_total_with_tax := display_cases_total * (1 + second_tax_rate)
let total_spent := first_transaction_total + display_cases_total_with_tax
total_spent

theorem marc_spent_total_correct : 
  marcTotalSpent 5 20 5 10 7 2 3 15 0.10 5 20 0.08 0.06 = 187.02 := 
by 
  unfold marcTotalSpent
  norm_num
  sorry

end marc_spent_total_correct_l549_549104


namespace money_spent_on_jacket_l549_549484

-- Define the initial amounts
def initial_money_sandy : ℝ := 13.99
def amount_spent_shirt : ℝ := 12.14
def additional_money_found : ℝ := 7.43

-- Amount of money left after buying the shirt
def remaining_after_shirt := initial_money_sandy - amount_spent_shirt

-- Total money after finding additional money
def total_after_additional := remaining_after_shirt + additional_money_found

-- Theorem statement: The amount Sandy spent on the jacket
theorem money_spent_on_jacket : total_after_additional = 9.28 :=
by
  sorry

end money_spent_on_jacket_l549_549484


namespace lanterns_top_level_l549_549403

theorem lanterns_top_level (a : ℕ) (h : ∑ i in finset.range 7, a * 2^i = 381) : a = 3 :=
by sorry

end lanterns_top_level_l549_549403


namespace no_arithmetic_sqrt_of_neg_real_l549_549139

theorem no_arithmetic_sqrt_of_neg_real (x : ℝ) (h : x < 0) : ¬ ∃ y : ℝ, y * y = x :=
by
  sorry

end no_arithmetic_sqrt_of_neg_real_l549_549139


namespace certain_event_birthday_example_l549_549557
-- Import the necessary library

-- Define the problem with conditions
def certain_event_people_share_birthday (num_days : ℕ) (num_people : ℕ) : Prop :=
  num_people > num_days

-- Define a specific instance based on the given problem
theorem certain_event_birthday_example : certain_event_people_share_birthday 365 400 :=
by
  sorry

end certain_event_birthday_example_l549_549557


namespace find_r_l549_549449

/-- Let f(x) be a polynomial defined as 3x^4 + x^3 + 2x^2 - 4x + r, for some real number r.
We need to find the value of r such that f(-1) = 0. -/
variable (r : ℝ)

def f (x : ℝ) : ℝ := 3 * x^4 + x^3 + 2 * x^2 - 4 * x + r

theorem find_r : f r (-1) = 0 → r = -8 := by
  intro h
  simp [f] at h
  linarith

end find_r_l549_549449


namespace difference_before_transfer_l549_549487

variable (S H : ℝ)
variable (h_S: S > H)
variable (h_diff_after: S - 0.43 - (H + 0.43) = 0.88)

theorem difference_before_transfer : S - H = 1.74 := 
by
  have h : S - H - 0.86 = 0.88 := by linarith [h_diff_after]
  linarith [h]

end difference_before_transfer_l549_549487


namespace smallest_positive_period_and_monotonic_intervals_max_min_values_on_interval_l549_549742

noncomputable def f (x : ℝ) : ℝ := (√3) * (Real.cos x)^2 - (Real.sin x) * (Real.cos (π - x))

theorem smallest_positive_period_and_monotonic_intervals :
  (∀ x : ℝ, f(x + π) = f(x)) ∧
  (∀ k : ℤ,
    (∀ x : ℝ, x ∈ Set.Icc (-π / 3 + k * π) (π / 6 + k * π) → f x ≤ f (x + π / 6)) ∧
    (∀ x : ℝ, x ∈ Set.Icc (π / 6 + k * π) (2 * π / 3 + k * π) → f (x + π / 6) ≥ f x)) :=
by
  sorry

theorem max_min_values_on_interval : 
  let a := - π / 4
  let b := π / 4
  (∀ x : ℝ, x ∈ Set.Icc a b → f x ≤ 1 + √3 / 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc a b → 0 ≤ f x) :=
by
  sorry

end smallest_positive_period_and_monotonic_intervals_max_min_values_on_interval_l549_549742


namespace tom_distance_proof_l549_549061

noncomputable def tom_jog_distance (time_jogging: ℝ) : ℝ := 
  let rate := 1 / 15 
  rate * time_jogging

theorem tom_distance_proof :
  tom_jog_distance 45 = 3 :=
by
  rw [tom_jog_distance, mul_div_cancel_left 45 (@of_nonzero ℝ _ 15 (ne_of_gt (by norm_num))),
      mul_one]
  norm_num
  sorry

end tom_distance_proof_l549_549061


namespace grandma_Olga_grandchildren_l549_549758

def daughters : Nat := 3
def sons : Nat := 3
def sons_per_daughter : Nat := 6
def daughters_per_son : Nat := 5

theorem grandma_Olga_grandchildren : 
  (daughters * sons_per_daughter) + (sons * daughters_per_son) = 33 := by
  sorry

end grandma_Olga_grandchildren_l549_549758


namespace sms_verification_l549_549481

def is_scam (sms_number : ℕ) (official_number : ℕ) (contacts_reset : ℕ → Prop) (verify_num : ℕ → ℕ → Prop) : Prop :=
  sms_number ≠ official_number ∧ verify_num sms_number official_number = false

def verify_sms_is_genuine (sms_number : ℕ) (official_number : ℕ) (contacts_reset : ℕ → Prop) (verify_num : ℕ → ℕ → Prop) : Prop :=
  sms_number = official_number ∧ verify_num sms_number official_number = true

theorem sms_verification (sms_number : ℕ) (official_number : ℕ) (contacts_reset : ℕ → Prop) (verify_num : ℕ → ℕ → Prop) :
  (verify_sms_is_genuine sms_number official_number contacts_reset verify_num ∨
   is_scam sms_number official_number contacts_reset verify_num) :=
begin
  sorry
end

end sms_verification_l549_549481


namespace max_halls_visited_l549_549511

-- Define the hall structure and conditions
structure Hall :=
(name : String)
(has_paintings : Bool)

def adjacency_relation (h1 h2 : Hall) : Prop :=
  -- Dummy implementation representing adjacency relation
  sorry

-- Given conditions in the form of premises in Lean
def halls : List Hall :=
  -- Dummy implementation representing the list of halls
  sorry

axiom hall_A : Hall
axiom hall_B : Hall

axiom hall_A_has_paintings : hall_A.has_paintings = true
axiom hall_B_has_paintings : hall_B.has_paintings = true
axiom tour_starts_at_A : ∃ h, h = hall_A
axiom tour_ends_at_B : ∃ h, h = hall_B
axiom total_halls : list.length halls = 16
axiom half_hall_paintings : list.count halls (λ h => h.has_paintings = true) = 8
axiom half_hall_sculptures : list.count halls (λ h => h.has_paintings = false) = 8
axiom alternate_arrangement : ∀ h1 h2, adjacency_relation h1 h2 → h1.has_paintings ≠ h2.has_paintings

-- Proof statement: The maximum number of unique halls the tourist can visit is 15
theorem max_halls_visited : ∃ route : List Hall,
  list.nodup route ∧ -- No hall is visited more than once
  leadsto route hall_A hall_B ∧ -- Route starts at A and ends at B
  (list.length route) = 15 :=
sorry

end max_halls_visited_l549_549511


namespace irrational_count_correct_l549_549862

def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

def numbers : list ℝ := [4 * Real.pi, 0, Real.sqrt 7, Real.sqrt 16 / 2, 0.1, 
                         (λ n, if n = 0 then 0.2 else 0.212212221 * 10^(-n)) (Nat.succ 0)]

def irrational_count := list.countp is_irrational numbers

theorem irrational_count_correct : irrational_count = 3 := 
sorry

end irrational_count_correct_l549_549862


namespace find_p_plus_q_l549_549254

-- Define the probabilities for the various combinations of health risk factors
def P_only_A := 0.1
def P_only_B := 0.1
def P_only_C := 0.1

def P_A_and_B_not_C := 0.14
def P_A_and_C_not_B := 0.14
def P_B_and_C_not_A := 0.14

def P_all_given_A_and_B := 1 / 3

-- The goal is to find p + q where the probability of no risk factors given no A is p/q
-- and p and q are relatively prime
theorem find_p_plus_q :
  let p := 21
  let q := 55
  Nat.gcd p q = 1 →
  p + q = 76 := 
by {
  simp,
  intro,
  exact dec_trivial,
}

#eval find_p_plus_q

end find_p_plus_q_l549_549254


namespace arithmetic_progression_primes_l549_549880

open Nat

theorem arithmetic_progression_primes {a : Nat → Nat} {d : Nat} 
  (h1 : ∀ n, Prime (a n))
  (h2 : d > 0)
  (h3 : a 0 > 15)
  (h4 : ∀ i j, i < j → a j = a i + (j - i) * d)
  (h5 : ∀ i j, a i < a j ↔ i < j) : d > 30000 :=
by 
  sorry

end arithmetic_progression_primes_l549_549880


namespace computer_price_in_2016_l549_549955

def price (p₀ : ℕ) (r : ℚ) (n : ℕ) : ℚ := p₀ * (r ^ (n / 4))

theorem computer_price_in_2016 :
  price 8100 (2/3 : ℚ) 16 = 1600 :=
by
  sorry

end computer_price_in_2016_l549_549955


namespace product_of_gcd_and_lcm_1440_l549_549317

def product_of_gcd_and_lcm_of_24_and_60 : ℕ :=
  Nat.gcd 24 60 * Nat.lcm 24 60 

/-- The product of the gcd and lcm of 24 and 60 is 1440. -/
theorem product_of_gcd_and_lcm_1440 : product_of_gcd_and_lcm_of_24_and_60 = 1440 := 
by
  sorry

end product_of_gcd_and_lcm_1440_l549_549317


namespace gcd_lcm_product_24_60_l549_549313

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l549_549313


namespace isosceles_right_triangle_area_l549_549937

variables (s : Real)
def hypotenuse (a : Real) : Real := a * Real.sqrt 2
def perimeter (a : Real) : Real := a * (2 + Real.sqrt 2)
def area (a : Real) : Real := (1 / 2) * a * a

theorem isosceles_right_triangle_area (a : Real) (h : perimeter a = 2 * s) :
  area a = (3 - 2 * Real.sqrt 2) * s^2 :=
sorry

end isosceles_right_triangle_area_l549_549937


namespace cubic_sum_l549_549027

theorem cubic_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 14) : x ^ 3 + y ^ 3 = 580 :=
by 
  sorry

end cubic_sum_l549_549027


namespace range_of_m_l549_549349

def quadratic_function (m : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + m * x + 1 = 0 → false)

def ellipse_condition (m : ℝ) : Prop :=
  0 < m

theorem range_of_m (m : ℝ) :
  (quadratic_function m ∨ ellipse_condition m) ∧ ¬ (quadratic_function m ∧ ellipse_condition m) →
  m ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi 2 :=
by
  sorry

end range_of_m_l549_549349


namespace rodent_control_total_rabbits_l549_549516

theorem rodent_control_total_rabbits :
  let R : ℕ → ℕ := λ n, 203 - 3 * n,
      S : ℕ → ℕ := λ n, 16 + 2 * n,
      n := Nat.ceil (187 / 5) in
  n = 38 →
  let total_rabbits := Finset.range (n + 1).sum (λ i, R (i + 1)) in
  total_rabbits = 5491 := 
by 
  sorry

end rodent_control_total_rabbits_l549_549516


namespace sin_double_angle_cos_sum_angle_l549_549701

-- Definition of the conditions
variables {α : ℝ}
axiom sin_alpha_eq : ∀ α : ℝ, sin α = 4 / 5
axiom alpha_second_quadrant : ∀ α : ℝ, π / 2 < α ∧ α < π

-- Proof goals
theorem sin_double_angle : ∀ α : ℝ, π / 2 < α ∧ α < π → sin α = 4 / 5 → sin (2 * α) = -24 / 25 :=
by
  intro α h1 h2
  sorry

theorem cos_sum_angle : ∀ α : ℝ, π / 2 < α ∧ α < π → sin α = 4 / 5 → cos (α + π / 4) = -7 * sqrt 2 / 10 :=
by
  intro α h1 h2
  sorry

end sin_double_angle_cos_sum_angle_l549_549701


namespace range_of_a_l549_549354

def p (a m : ℝ) : Prop := m^2 + 12 * a^2 < 7 * a * m ∧ a > 0
def q (m : ℝ) : Prop := 1 < m ∧ m < 3 / 2

theorem range_of_a (a : ℝ) :
  (∀ m : ℝ, p a m → q m) → 
  (∃ (a_lower a_upper : ℝ), a_lower ≤ a ∧ a ≤ a_upper ∧ a_lower = 1 / 3 ∧ a_upper = 3 / 8) :=
sorry

end range_of_a_l549_549354


namespace magic_square_sum_l549_549882

-- Define the range of integers
def int_range : Set ℤ := {-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}

-- Define the square matrix condition
def is_magic_square (matrix : Fin 6 → Fin 6 → ℤ) : Prop :=
  (∀ i: Fin 6, matrix i = matrix 0) ∧
  (∀ j: Fin 6, matrix 0 = matrix j) ∧
  (sum (λ k: Fin 6, matrix k k) = sum (λ k: Fin 6, matrix k (5 - k)) )

-- Define the problem as a theorem statement
theorem magic_square_sum :
  ∀ (arr : Fin 6 → Fin 6 → ℤ),
  (∀ i j, arr i j ∈ int_range) →
  is_magic_square arr →
  (∀ i: Fin 6, sum (arr i) = 12.5) ∧ (∀ j: Fin 6, sum (λ i, arr i j) = 12.5) ∧
  (sum (λ k: Fin 6, arr k k) = 12.5) ∧ (sum (λ k: Fin 6, arr k (5 - k)) = 12.5) :=
sorry

end magic_square_sum_l549_549882


namespace no_equal_2008_distances_l549_549690

def frog_positions : set ℕ := {0, 1, 2, 3}

theorem no_equal_2008_distances (A B C D : ℕ) (d1 d2 d3 : ℕ) :
  A ∈ frog_positions ∧ B ∈ frog_positions ∧ C ∈ frog_positions ∧ D ∈ frog_positions →
  d1 = 2008 ∧ d2 = 2008 ∧ d3 = 2008 →
  (B - A) + (C - B) + (D - C) = 3 →
  false :=
by
  sorry

end no_equal_2008_distances_l549_549690


namespace ratio_of_cards_lost_l549_549126

-- Definitions based on the conditions
def purchases_per_week : ℕ := 20
def weeks_per_year : ℕ := 52
def cards_left : ℕ := 520

-- Main statement to be proved
theorem ratio_of_cards_lost (total_cards : ℕ := purchases_per_week * weeks_per_year)
                            (cards_lost : ℕ := total_cards - cards_left) :
                            (cards_lost : ℚ) / total_cards = 1 / 2 :=
by
  sorry

end ratio_of_cards_lost_l549_549126


namespace find_triangle_angles_l549_549421

-- Definitions and conditions
variables {A B C H K : ℝ}
variables (triangle_ABC : ∀ (A B C : ℝ), triangle (A B C))
variables (CH AK AB BC : ℝ)
variable (cond1 : CH ≥ AB)
variable (cond2 : AK ≥ BC)

-- The statement to be proved
theorem find_triangle_angles (cond1 : CH ≥ AB) (cond2 : AK ≥ BC) : 
  ∃ (angle_A angle_B angle_C : ℝ), is_90_deg angle_B ∧
  is_45_deg angle_A ∧ is_45_deg angle_C :=
sorry

end find_triangle_angles_l549_549421


namespace linear_functions_value_at_20_l549_549537

-- Definitions of linear functions intersection and properties
def linear_functions_intersect_at (k1 k2 b1 b2 : ℝ) (x : ℝ) : Prop :=
  k1 * x + b1 = k2 * x + b2

def value_difference_at (k1 k2 b1 b2 : ℝ) (x diff : ℝ) : Prop :=
  (k1 * x + b1 - (k2 * x + b2)).abs = diff

def function_value_at (k b x val : ℝ) : Prop :=
  k * x + b = val

-- Main proof statement
theorem linear_functions_value_at_20
  (k1 k2 b1 b2 : ℝ) :
  (linear_functions_intersect_at k1 k2 b1 b2 2) →
  (value_difference_at k1 k2 b1 b2 8 8) →
  (function_value_at k1 b1 20 100) →
  (k2 * 20 + b2 = 76 ∨ k2 * 20 + b2 = 124) :=
by
  sorry

end linear_functions_value_at_20_l549_549537


namespace quadratic_b_value_l549_549812

theorem quadratic_b_value (b m : ℝ) (h_b_pos : 0 < b) (h_quad_form : ∀ x, x^2 + b * x + 108 = (x + m)^2 - 4)
  (h_m_pos_sqrt : m = 4 * Real.sqrt 7 ∨ m = -4 * Real.sqrt 7) : b = 8 * Real.sqrt 7 :=
by
  sorry

end quadratic_b_value_l549_549812


namespace find_c_l549_549049

-- Define the function and its properties
def quadratic_function (x b : ℝ) : ℝ := x^2 + b * x + 3

-- Define the range condition for the quadratic function
def range_condition (b : ℝ) : Prop :=
  ∀ y : ℝ, y = quadratic_function x b → y ∈ set.Ici (0 : ℝ)

-- Define the inequality and its solution set
def inequality_solution_set (b c m : ℝ) : Prop :=
  ∀ x : ℝ, (x < m ∧ x > m - 8) ↔ (quadratic_function x b < c)

-- Define the proof statement
theorem find_c (b m : ℝ) (h_b : b^2 = 12) (h_range : range_condition b) :
  inequality_solution_set b 16 m :=
sorry

end find_c_l549_549049


namespace complex_power_l549_549580

theorem complex_power : (1 - Complex.i)^4 = -4 :=
by
  sorry

end complex_power_l549_549580


namespace correct_statements_l549_549829

theorem correct_statements (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  ( (a^2 - b^2 = 1 → a - b < 1) ∧
    (∀ (a = 2) (b = 2 / 3), ¬ (a - b = ab) ∧ ∀ (hb: b ≠ 0) (ha : a ≠ 0),¬ (1/a - 1/b = 1 → a - b < 1)) ∧
    (|sqrt a - sqrt b| = 1 → |a - b| < 1) ∧
    (|a^3 - b^3| = 1 → |a - b| < 1) ) :=
by
  sorry

end correct_statements_l549_549829


namespace Emily_current_average_l549_549629

noncomputable def Emily_average_grade (E : ℕ) : ℕ :=
  if (9 * E + 90) / 10 = 91 then E else 0

theorem Emily_current_average :
  ∃ E : ℕ, Emily_average_grade E = 91 :=
by
  -- Conditions
  let assignments_done := 9
  let Ahmed_grade := 91
  let final_assignment_grade := 90
  let minimum_Ahmed_final := 100
  let total_assignments := 10
  let Ahmed_final_grade_needed := Ahmed_grade + (minimum_Ahmed_final - 91)
  
  -- Asserting Emily's average
  existsi 91
  unfold Emily_average_grade
  simp
  exact rfl

end Emily_current_average_l549_549629


namespace parabola_directrix_l549_549893

theorem parabola_directrix (x y : ℝ) : y = - (1 / 4) * x ^ 2 → directrix y = 1 :=
by sorry

end parabola_directrix_l549_549893


namespace bacteria_dish_filling_l549_549062

theorem bacteria_dish_filling :
  ∀ (days : ℕ), (days = 30) → (∀ t : ℕ, (bacteria_size t) = 2 ^ t) →
  (∃ t : ℕ, bacteria_size t = dish_size / 16 ∧ t = 26) :=
by
  intros days hdays hbacteria
  sorry

-- Definitions for bacteria size and dish size
def bacteria_size : ℕ → ℕ
| t => 2 ^ t

def dish_size : ℕ := bacteria_size 30

end bacteria_dish_filling_l549_549062


namespace value_of_fraction_l549_549387

theorem value_of_fraction (m n : ℝ) (h1 : m^2 - 2 * m - 1 = 0) (h2 : n^2 + 2 * n - 1 = 0) (h3 : m * n ≠ 1) : 
  (mn + n + 1) / n = 3 :=
by
  sorry

end value_of_fraction_l549_549387


namespace impossible_sequence_l549_549517

theorem impossible_sequence (a : ℕ → ℝ) (c : ℝ) (a1 : ℝ)
  (h_periodic : ∀ n, a (n + 3) = a n)
  (h_det : ∀ n, a n * a (n + 3) - a (n + 1) * a (n + 2) = c)
  (ha1 : a 1 = 2) (hc : c = 2) : false :=
by
  sorry

end impossible_sequence_l549_549517


namespace incorrect_median_l549_549687

def data : list ℕ := [2, 3, 6, 9, 3, 7]

/-- Prove that the incorrect statement is: The median is 6 -/
theorem incorrect_median : ¬(median data = 6) :=
by
  sorry

end incorrect_median_l549_549687


namespace factorization_l549_549293

variable {R : Type*} [CommRing R]
variables (x y z : R)

theorem factorization :
  2 * x^3 - x^2 * z - 4 * x^2 * y + 2 * x * y * z + 2 * x * y^2 - y^2 * z =
  (2 * x - z) * (x - y)^2 := 
sorry

end factorization_l549_549293


namespace sequence_sum_l549_549662

theorem sequence_sum (r : ℝ) (x y : ℝ)
  (a : ℕ → ℝ)
  (h1 : a 1 = 4096)
  (h2 : a 2 = 1024)
  (h3 : a 3 = 256)
  (h4 : a 6 = 4)
  (h5 : a 7 = 1)
  (h6 : a 8 = 0.25)
  (h_sequence : ∀ n, a (n + 1) = r * a n)
  (h_r : r = 1 / 4) :
  x + y = 80 :=
sorry

end sequence_sum_l549_549662


namespace irrational_numbers_count_is_3_l549_549861

def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), (q : ℝ) = x

def irrational_count : ℕ :=
  [4 * Real.pi, 0, Real.sqrt 7, Real.sqrt 16 / 2, 0.1, 0.212212221 -- defined pattern needs to be handled
  ].filter (λ x, is_irrational x).length

theorem irrational_numbers_count_is_3 : irrational_count = 3 := by
  sorry

end irrational_numbers_count_is_3_l549_549861


namespace exam_date_correct_l549_549201

def start_date := (4, 1)  -- (Month, Day) representing April 1st
def study_days := 65
def total_april_days := 30
def total_may_days := 31

def exam_date : Nat × Nat := do
  let days_in_april := total_april_days - 1  -- Days left in April
  let remaining_days := study_days - days_in_april
  if remaining_days <= total_may_days then
    (5, remaining_days)  -- Exam date in May
  else
    let remaining_days_in_may := remaining_days - total_may_days
    (6, remaining_days_in_may)  -- Exam date in June

theorem exam_date_correct :
  exam_date = (6, 5) := by
  sorry

end exam_date_correct_l549_549201


namespace neg_of_exists_l549_549148

theorem neg_of_exists (P : ℝ → Prop) : 
  (¬ ∃ x: ℝ, x ≥ 3 ∧ x^2 - 2 * x + 3 < 0) ↔ (∀ x: ℝ, x ≥ 3 → x^2 - 2 * x + 3 ≥ 0) :=
by
  sorry

end neg_of_exists_l549_549148


namespace sum_of_possible_values_of_x_l549_549515

-- Conditions
def radius (x : ℝ) : ℝ := x - 2
def semiMajor (x : ℝ) : ℝ := x - 3
def semiMinor (x : ℝ) : ℝ := x + 4

-- Theorem to be proved
theorem sum_of_possible_values_of_x (x : ℝ) :
  (π * semiMajor x * semiMinor x = 2 * π * (radius x) ^ 2) →
  (x = 5 ∨ x = 4) →
  5 + 4 = 9 :=
by
  intros
  rfl

end sum_of_possible_values_of_x_l549_549515


namespace find_y_intercept_l549_549913

theorem find_y_intercept (m : ℝ) (x_intercept: ℝ × ℝ) : (x_intercept.snd = 0) → (x_intercept = (-4, 0)) → m = 3 → (0, m * 4 - m * (-4)) = (0, 12) :=
by
  sorry

end find_y_intercept_l549_549913


namespace find_a_solve_inequality_l549_549374

-- Definition and conditions for the logarithmic function
def logarithmic_function (a x : ℝ) : ℝ := log a x

-- Problem 1: Given f(8) = 3, find the value of a
theorem find_a (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (h : logarithmic_function a 8 = 3) :
  a = 2 :=
sorry

-- Problem 2: Solve the inequality f(x) ≤ log_a (2 - 3x)
theorem solve_inequality (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (x : ℝ) :
  (a > 1 → (0 < x ∧ x ≤ 1/2)) ∧ (0 < a ∧ a < 1 → (1/2 ≤ x ∧ x < 2/3)) :=
sorry

end find_a_solve_inequality_l549_549374


namespace grandma_Olga_grandchildren_l549_549757

def daughters : Nat := 3
def sons : Nat := 3
def sons_per_daughter : Nat := 6
def daughters_per_son : Nat := 5

theorem grandma_Olga_grandchildren : 
  (daughters * sons_per_daughter) + (sons * daughters_per_son) = 33 := by
  sorry

end grandma_Olga_grandchildren_l549_549757


namespace percentage_discount_l549_549227

theorem percentage_discount (C S NewC NewS : ℝ) 
  (hC : C = 40)
  (hS : S = 1.25 * C)
  (hNewS : NewS = S - 8.4)
  (hProfit1 : S = C + 0.25 * C)
  (hProfit2 : NewS = NewC + 0.3 * NewC)
  (hNewC : NewC = C * (1 - 0.2)) :
  0.2 = (C - NewC) / C := 
by
  have h1 : 1.25 * C = 40 + 0.25 * 40, from hS ▸ hC ▸ rfl,
  have h2 : 1.30 * NewC = 41.60, from hProfit2 ▸ hNewS ▸ rfl,
  have h3 : NewC = 40 * 0.8, from hNewC ▸ rfl,
  sorry

end percentage_discount_l549_549227


namespace length_of_train_l549_549241

theorem length_of_train (speed : ℕ) (time : ℝ) (bridge_length : ℝ) : bridge_length = 135 → 
speed = 75 → time = 11.279097672186225 → 
let speed_m_s := (speed * 1000) / 3600 in
let total_distance := speed_m_s * time in 
let train_length := total_distance - bridge_length in
train_length = 100 :=
by
  intro h_bridge_length h_speed h_time
  rw h_bridge_length
  rw h_speed
  rw h_time
  let speed_m_s := (75 * 1000) / 3600
  let total_distance := speed_m_s * 11.279097672186225
  let train_length := total_distance - 135
  have : train_length = 100 := rfl -- this step is actually handled by Lean engine’s arithmetic simplifications.
  assumption

end length_of_train_l549_549241


namespace GrandmaOlga_grandchildren_l549_549755

theorem GrandmaOlga_grandchildren :
  (∃ d : ℕ, d = 3 ∧ ∀ i : Fin d, 6 ∈ ℕ) ∧
  (∃ s : ℕ, s = 3 ∧ ∀ j : Fin s, 5 ∈ ℕ) →
  18 + 15 = 33 :=
by
  intros h
  cases' h with h_d h_s
  cases' h_d with d_vals num_d
  cases' d_vals with d_eq d_cond
  cases' h_s with s_vals num_s
  cases' s_vals with s_eq s_cond
  sorry

end GrandmaOlga_grandchildren_l549_549755


namespace total_miles_driven_l549_549409

-- Define the required variables and their types
variables (avg1 avg2 : ℝ) (gallons1 gallons2 : ℝ) (miles1 miles2 : ℝ)

-- State the conditions
axiom sum_avg_mpg : avg1 + avg2 = 75
axiom first_car_gallons : gallons1 = 25
axiom second_car_gallons : gallons2 = 35
axiom first_car_avg_mpg : avg1 = 40

-- Declare the function to calculate miles driven
def miles_driven (avg_mpg gallons : ℝ) : ℝ := avg_mpg * gallons

-- Declare the theorem for proof
theorem total_miles_driven : miles_driven avg1 gallons1 + miles_driven avg2 gallons2 = 2225 := by
  sorry

end total_miles_driven_l549_549409


namespace unique_rectangles_perimeter_sum_correct_l549_549231

def unique_rectangle_sum_of_perimeters : ℕ :=
  let possible_pairs := [(4, 12), (6, 6)]
  let perimeters := possible_pairs.map (λ (p : ℕ × ℕ) => 2 * (p.1 + p.2))
  perimeters.sum

theorem unique_rectangles_perimeter_sum_correct : unique_rectangle_sum_of_perimeters = 56 :=
  by 
  -- skipping actual proof
  sorry

end unique_rectangles_perimeter_sum_correct_l549_549231


namespace gcd_exponentiation_l549_549451

theorem gcd_exponentiation (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) : 
  let a := 2^m - 2^n
  let b := 2^(m^2 + m * n + n^2) - 1
  let d := Nat.gcd a b
  d = 1 ∨ d = 7 :=
by
  sorry

end gcd_exponentiation_l549_549451


namespace fencing_rate_l549_549886

theorem fencing_rate (area_hectares : ℝ) (cost_fencing : ℝ) (rate_per_meter : ℝ) :
  area_hectares = 13.86 → cost_fencing = 5806.831494371739 → rate_per_meter = 4.40 :=
by
  -- Define constants
  let pi := Real.pi
  let area_square_meters := 13.86 * 10000  -- Convert hectares to square meters
  let radius := Real.sqrt (area_square_meters / pi)  -- Calculate radius
  let circumference := 2 * pi * radius  -- Calculate circumference
  let rate := cost_fencing / circumference  -- Calculate rate per meter
  
  -- Use the provided conditions
  intro h_area
  intro h_cost

  -- Perform the final assertions to match the expected rate
  have h_area_def : area_square_meters = 138600 := by sorry
  have h_radius_def : radius ≈ 210.162 := by sorry
  have h_circumference_def : circumference ≈ 1320.013 := by sorry
  have h_rate_def : rate ≈ 4.40 := by sorry

  -- Conclude the proof with the expected rate
  sorry

end fencing_rate_l549_549886


namespace equidistant_sections_equal_area_l549_549624

noncomputable def triangle_division (ABC : Triangle) (P : Fin 11 → Triangle.Vertex) (I : Triangle.Vertex) : Prop :=
  (∀ (i : Fin 11), 
    let area_i := Triangle.area (Triangle.mk I (P i) (P (i + 1 % 11))) in
    let area_j := Triangle.area (Triangle.mk I (P j) (P (j + 1 % 11))) in
    area_i = area_j )

theorem equidistant_sections_equal_area (ABC : Triangle) (P : Fin 11 → Triangle.Vertex) (I : Triangle.Vertex) :
  is_incenter ABC I →
  ∀ (i : Fin 11), 
    (P i) and (P ((i + 1) % 11)) lie on the perimeter of ABC →
    (∀ (i : Fin 11), distance ((P i) (P ((i + 1) % 11)) = perimeter ABC / 11) →
    triangle_division ABC P I :=
begin
  sorry
end

end equidistant_sections_equal_area_l549_549624


namespace solve_xy_eq_yx_l549_549875

theorem solve_xy_eq_yx (x y : ℕ) (hxy : x ≠ y) : x^y = y^x ↔ ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by
  sorry

end solve_xy_eq_yx_l549_549875


namespace Jurassic_Zoo_Total_l549_549884

theorem Jurassic_Zoo_Total
  (C : ℕ) (A : ℕ)
  (h1 : C = 161)
  (h2 : 8 * A + 4 * C = 964) :
  A + C = 201 := by
  sorry

end Jurassic_Zoo_Total_l549_549884


namespace travis_takes_home_money_l549_549185

-- Define the conditions
def total_apples : ℕ := 10000
def apples_per_box : ℕ := 50
def price_per_box : ℕ := 35

-- Define the main theorem to be proved
theorem travis_takes_home_money : (total_apples / apples_per_box) * price_per_box = 7000 := by
  sorry

end travis_takes_home_money_l549_549185


namespace infinite_series_value_l549_549972

theorem infinite_series_value :
  ∑' n in (Set.Ici 2), (n ^ 4 + 3 * n ^ 2 + 10 * n + 10) / (2 ^ n * (n ^ 4 + 4)) = 11 / 10 :=
sorry

end infinite_series_value_l549_549972


namespace charlyn_visible_area_l549_549267

noncomputable def visible_area (side_length vision_distance : ℝ) : ℝ :=
  let outer_rectangles_area := 4 * (side_length * vision_distance)
  let outer_squares_area := 4 * (vision_distance * vision_distance)
  let inner_square_area := 
    let inner_side_length := side_length - 2 * vision_distance
    inner_side_length * inner_side_length
  let total_walk_area := side_length * side_length
  total_walk_area - inner_square_area + outer_rectangles_area + outer_squares_area

theorem charlyn_visible_area :
  visible_area 10 2 = 160 := by
  sorry

end charlyn_visible_area_l549_549267


namespace min_abs_phi_l549_549898

-- Definitions
def function_translated_is_odd (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x, f (x + a) = -f (-x - a)

-- Conditions
def translated_cos_function_is_odd (φ : ℝ) :=
  function_translated_is_odd (λ x, Real.cos (2 * x + φ)) (π / 3)

-- Theorem statement
theorem min_abs_phi (φ : ℝ) (h : translated_cos_function_is_odd φ) :
  |φ| = π / 6 :=
sorry

end min_abs_phi_l549_549898


namespace domain_of_f_ax_plus_f_x_div_a_l549_549357

theorem domain_of_f_ax_plus_f_x_div_a {f : ℝ → ℝ} (h : ∀ x, -1 ≤ x ∧ x ≤ 1 → f x is_defined) {a : ℝ} (h_pos : a > 0) :
  (∀ x, (f (a * x)).is_defined ∧ (f (x / a)).is_defined → 
   ((a ≥ 1 → -1/a ≤ x ∧ x ≤ 1/a) ∧ (0 < a ∧ a < 1 → -a ≤ x ∧ x ≤ a))) :=
by
  sorry

end domain_of_f_ax_plus_f_x_div_a_l549_549357


namespace evaluate_correct_l549_549289

-- Defining constants based on the problem
def base1 := 27
def exp1 := (2 : ℝ) / 3
def base2 := 2
def log_arg := (1 : ℝ) / 8
def log_base := 2
def inner_exp := Real.log base2 3

-- Translating the conditions and equivalence to a Lean theorem
noncomputable def evaluate_expression : ℝ :=
  base1 ^ exp1 - base2 ^ inner_exp * Real.log log_base log_arg

theorem evaluate_correct : evaluate_expression = 18 :=
  by
  sorry

end evaluate_correct_l549_549289


namespace sale_price_tea_correct_l549_549620

noncomputable def sale_price_of_mixed_tea (weight1 weight2 price1 price2 profit_percentage : ℝ) : ℝ :=
let total_cost := weight1 * price1 + weight2 * price2
let total_weight := weight1 + weight2
let cost_price_per_kg := total_cost / total_weight
let profit_per_kg := profit_percentage * cost_price_per_kg
let sale_price_per_kg := cost_price_per_kg + profit_per_kg
sale_price_per_kg

theorem sale_price_tea_correct :
  sale_price_of_mixed_tea 80 20 15 20 0.20 = 19.2 :=
  by
  sorry

end sale_price_tea_correct_l549_549620


namespace modulus_of_z_l549_549365

-- defining the complex number z
def z : ℂ := 1 + real.sqrt 3 * complex.I

-- stating the proof problem
theorem modulus_of_z : 
  complex.abs z = 2 := 
sorry

end modulus_of_z_l549_549365


namespace sequence_value_l549_549665

theorem sequence_value : 
  ∃ (x y r : ℝ), 
    (4096 * r = 1024) ∧ 
    (1024 * r = 256) ∧ 
    (256 * r = x) ∧ 
    (x * r = y) ∧ 
    (y * r = 4) ∧  
    (4 * r = 1) ∧ 
    (x + y = 80) :=
by
  sorry

end sequence_value_l549_549665


namespace count_perfect_squares_between_200_and_600_l549_549763

-- Definition to express the condition of a perfect square within a specific range
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

-- Definition to express the count of perfect squares between 200 and 600
def perfect_squares_between (a b : ℕ) : ℕ :=
  (nat.floor (real.sqrt b) - nat.ceil (real.sqrt a)) + 1

theorem count_perfect_squares_between_200_and_600 :
  perfect_squares_between 200 600 = 10 :=
by
  sorry

end count_perfect_squares_between_200_and_600_l549_549763


namespace find_k_range_find_mn_distance_l549_549712

noncomputable def intersects_circle (k : ℝ) : Prop :=
  let A := (0 : ℝ, 1 : ℝ)
  let C := (2 : ℝ, 3 : ℝ)
  let R := 1
  (4 - Real.sqrt 7) / 3 < k ∧ k < (4 + Real.sqrt 7) / 3

theorem find_k_range : intersects_circle k :=
  sorry

noncomputable def distance_mn (k : ℝ) : ℝ :=
  let A := (0 : ℝ, 1 : ℝ)
  let C := (2 : ℝ, 3 : ℝ)
  let O := (0 : ℝ, 0 : ℝ)
  (1 + k^2) / (k + 1)^2 = 12 ∧ k = 1 → 2 -- Based on given condition \( \overrightarrow{OM} \cdot \overrightarrow{ON} \) and solving leads to \( k = 1 \), proving distance is 2

theorem find_mn_distance (k : ℝ) (h : intersects_circle k) : distance_mn k = 2 :=
  sorry

end find_k_range_find_mn_distance_l549_549712


namespace intersection_complement_eq_l549_549100

open Set

variable (U M N : Set ℕ)

theorem intersection_complement_eq :
  U = {1, 2, 3, 4, 5} →
  M = {1, 4} →
  N = {1, 3, 5} →
  N ∩ (U \ M) = {3, 5} := by 
sorry

end intersection_complement_eq_l549_549100


namespace a_is_2_years_older_than_b_l549_549963

def a_b_age_difference : Prop :=
  ∀ (A B C : ℕ), B = 10 → B = 2 * C → A + B + C = 27 → A - B = 2

theorem a_is_2_years_older_than_b : a_b_age_difference :=
begin
  sorry
end

end a_is_2_years_older_than_b_l549_549963


namespace common_divisors_of_40_and_72_l549_549019

theorem common_divisors_of_40_and_72 : 
  ∃ (n : ℕ), n = {x | x ∣ 40 ∧ x ∣ 72}.to_finset.card ∧ n = 4 :=
by 
  sorry

end common_divisors_of_40_and_72_l549_549019


namespace ab_sum_l549_549012

open Set

noncomputable def U : Set ℝ := univ
def A : Set ℝ := { x | -1 < x ∧ x < 5 }
def B : Set ℝ := { x | 2 < x ∧ x < 8 }
def C (a : ℝ) : Set ℝ := { x | a + 1 ≤ x ∧ x ≤ 2 * a - 2 }
def complement_A : Set ℝ := { x | x ≤ -1 ∨ x ≥ 5 }
def complement_B : Set ℝ := { x | x ≤ 2 ∨ x ≥ 8 }
def complement_A_and_C (a b : ℝ) : Set ℝ := { x | 6 ≤ x ∧ x ≤ b }

theorem ab_sum (a b: ℝ) (h: (complement_A ∩ C a) = complement_A_and_C a b) : a + b = 13 :=
by
  sorry

end ab_sum_l549_549012


namespace infinitely_many_multiples_of_7_l549_549133

def sequence (n : ℕ) : ℕ
| 0       := 0 -- Note: a₀ is typically undefined, but we need an initial value.
| 1       := 1
| (n + 2) := sequence (n + 1) + sequence (n / 2)

theorem infinitely_many_multiples_of_7 : ∀ k : ℕ, ∃ n > k, 7 ∣ sequence n :=
by
  sorry

end infinitely_many_multiples_of_7_l549_549133


namespace base_k_addition_is_ten_l549_549326

theorem base_k_addition_is_ten :
  ∃ k : ℕ, (k > 4) ∧ (5 * k^3 + 3 * k^2 + 4 * k + 2 + 6 * k^3 + 4 * k^2 + 2 * k + 1 = 1 * k^4 + 4 * k^3 + 1 * k^2 + 6 * k + 3) ∧ k = 10 :=
by
  sorry

end base_k_addition_is_ten_l549_549326


namespace equilateral_triangle_of_angle_and_side_sequences_l549_549405

variable {A B C a b c : ℝ}

theorem equilateral_triangle_of_angle_and_side_sequences
  (H_angles_arithmetic : 2 * B = A + C)
  (H_sum_angles : A + B + C = Real.pi)
  (H_sides_geometric : b^2 = a * c) :
  A = Real.pi / 3 ∧ B = Real.pi / 3 ∧ C = Real.pi / 3 ∧ a = b ∧ b = c :=
by
  sorry

end equilateral_triangle_of_angle_and_side_sequences_l549_549405


namespace parallelogram_area_l549_549826

variables {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]
variables (p q : V)
variables (hp : ‖p‖ = 1) (hq : ‖q‖ = 1) (angle_45 : inner_product_geometry.angle p q = real.pi / 4)

theorem parallelogram_area : 
  ‖(p + 3 • q) ∧ (3 • p + q)‖ = 2 * real.sqrt 2 := by
  sorry

end parallelogram_area_l549_549826


namespace angle_equality_l549_549823

-- Define the conditions
variables (ABC : Triangle) (O : Point) (H : Point)
variable (circumcenter_O : is_circumcenter O ABC)
variable (altitude_H : is_foot_of_altitude H A ABC)

-- Define the angles involved
variables (BAO CAH : Angle)

-- State the problem
theorem angle_equality (h₁ : has_angle ABC BAO)
                       (h₂ : has_angle ABC CAH)
                       (hO : is_circumcenter O ABC)
                       (hH : is_foot_of_altitude H A ABC) :
  BAO = CAH := sorry

end angle_equality_l549_549823


namespace range_of_angle_between_vectors_l549_549014

variable (a b : ℝ) (angle : ℝ)

noncomputable def magnitude_a : ℝ := 2 * Real.sqrt 2 * b
noncomputable def magnitude_b : ℝ := b

noncomputable def f (x : ℝ) : ℝ := 
  2 * x^3 + 3 * (magnitude_a .* a) * x^2 + 6 * (a * b) * x + 7

theorem range_of_angle_between_vectors 
  (h1 : magnitude_a = 2 * Real.sqrt 2 * magnitude_b) 
  (h2 : ∀ x : ℝ, (f x).deriv >= 0) : 
  0 ≤ angle ∧ angle ≤ π / 4 :=
sorry

end range_of_angle_between_vectors_l549_549014


namespace ammonium_bromide_total_weight_l549_549943

noncomputable def nitrogen_weight : ℝ := 14.01
noncomputable def hydrogen_weight : ℝ := 1.01
noncomputable def bromine_weight : ℝ := 79.90
noncomputable def ammonium_bromide_weight : ℝ := nitrogen_weight + 4 * hydrogen_weight + bromine_weight
noncomputable def moles : ℝ := 5
noncomputable def total_weight : ℝ := moles * ammonium_bromide_weight

theorem ammonium_bromide_total_weight :
  total_weight = 489.75 :=
by
  -- The proof is omitted.
  sorry

end ammonium_bromide_total_weight_l549_549943


namespace range_of_a_l549_549951

namespace ProofProblem

theorem range_of_a (a : ℝ) (H1 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → ∃ y : ℝ, y = a * x + 2 * a + 1 ∧ y > 0 ∧ y < 0) : 
  -1 < a ∧ a < -1/3 := 
sorry

end ProofProblem

end range_of_a_l549_549951


namespace probability_at_least_one_female_l549_549131

open Nat

theorem probability_at_least_one_female :
  let males := 2
  let females := 3
  let total_students := males + females
  let select := 2
  let total_ways := choose total_students select
  let ways_at_least_one_female : ℕ := (choose females 1) * (choose males 1) + choose females 2
  (ways_at_least_one_female / total_ways : ℚ) = 9 / 10 := by
  sorry

end probability_at_least_one_female_l549_549131


namespace james_total_toys_78_l549_549809

noncomputable def total_toys (initial_toy_cars : ℕ) (additional_toy_cars : ℕ) : ℕ :=
  let toy_cars := initial_toy_cars + additional_toy_cars
  let toy_soldiers := 2 * toy_cars
  toy_cars + toy_soldiers

theorem james_total_toys_78 :
  initial_toy_cars = 20 → additional_toy_cars = 6 → total_toys 20 6 = 78 :=
by {
  intros h_initial h_additional,
  have h_toy_cars : 20 + 6 = 26 := by norm_num,
  have h_toy_soldiers : 2 * 26 = 52 := by norm_num,
  show 26 + 52 = 78,
  exact add_comm _ _ ▸ (congr_arg (· + 52) h_toy_cars) ▸ (congr_arg (26 + ·) h_toy_soldiers) ▸ rfl,
}

end james_total_toys_78_l549_549809


namespace smallest_h_divisible_by_8_11_24_l549_549949

noncomputable def smallest_number_h : ℕ :=
  let lcm_val := Nat.lcm 8 11 
  let lcm_val := Nat.lcm lcm_val 24 
  lcm_val - 5

theorem smallest_h_divisible_by_8_11_24 :
  ∃ h : ℕ, (h + 5) % 8 = 0 ∧ (h + 5) % 11 = 0 ∧ (h + 5) % 24 = 0 ∧ h = 259 :=
begin
  use 259,
  have h_plus_5 : 259 + 5 = 264 := by norm_num,
  split,
  { rw [h_plus_5],
    exact Nat.mod_eq_zero_of_dvd (by norm_num : 8 ∣ 264), },
  split,
  { rw [h_plus_5],
    exact Nat.mod_eq_zero_of_dvd (by norm_num : 11 ∣ 264), },
  split,
  { rw [h_plus_5],
    exact Nat.mod_eq_zero_of_dvd (by norm_num : 24 ∣ 264), },
  { refl, }
end

end smallest_h_divisible_by_8_11_24_l549_549949


namespace torchbearer_probability_l549_549036

noncomputable def arithmetic_sequence_probability : ℚ :=
let torchbearers := finset.range 18,
    combinations := finset.powersetLen 3 torchbearers in
let is_valid_arith_seq (s : finset ℕ) : Prop :=
  ∃ a b c, s = {a, b, c} ∧ (b - a = 3 ∧ c - b = 3) in
let valid_combinations := combinations.filter is_valid_arith_seq in
valid_combinations.card / combinations.card

theorem torchbearer_probability :
  arithmetic_sequence_probability = 1 / 68 :=
sorry

end torchbearer_probability_l549_549036


namespace simple_interest_years_l549_549521

noncomputable def simple_interest (P r t : ℕ) : ℕ :=
  P * r * t / 100

noncomputable def compound_interest (P r n : ℕ) : ℕ :=
  P * (1 + r / 100)^n - P

theorem simple_interest_years
  (P_si r_si P_ci r_ci n_ci si_half_ci si_si : ℕ)
  (h_si : simple_interest P_si r_si si_si = si_half_ci)
  (h_ci : compound_interest P_ci r_ci n_ci = si_half_ci * 2) :
  si_si = 2 :=
by
  sorry

end simple_interest_years_l549_549521


namespace system1_solution_system2_solution_l549_549495

theorem system1_solution (x y : ℝ) (h1 : x - 2 * y = 0) (h2 : 3 * x + 2 * y = 8) : 
  x = 2 ∧ y = 1 := sorry

theorem system2_solution (x y : ℝ) (h1 : 3 * x - 5 * y = 9) (h2 : 2 * x + 3 * y = -6) : 
  x = -3 / 19 ∧ y = -36 / 19 := sorry

end system1_solution_system2_solution_l549_549495


namespace tank_missing_water_l549_549596

def max_capacity := 350000
def loss1_rate := 32000
def loss1_duration := 5
def loss2_rate := 10000
def loss2_duration := 10
def fill_rate := 40000
def fill_duration := 3

theorem tank_missing_water 
  (max_capacity = 350000)
  (loss1_rate = 32000)
  (loss1_duration = 5)
  (loss2_rate = 10000)
  (loss2_duration = 10)
  (fill_rate = 40000)
  (fill_duration = 3) : 
  (max_capacity - 
   ((loss1_rate * loss1_duration) + (loss2_rate * loss2_duration) - (fill_rate * fill_duration))) = 140000 :=
  by 
  unfold max_capacity loss1_rate loss1_duration loss2_rate loss2_duration fill_rate fill_duration
  sorry

end tank_missing_water_l549_549596


namespace arithmetic_sequence_term_count_l549_549653

theorem arithmetic_sequence_term_count (a d l n : ℕ) (h1 : a = 11) (h2 : d = 4) (h3 : l = 107) :
  l = a + (n - 1) * d → n = 25 := by
  sorry

end arithmetic_sequence_term_count_l549_549653


namespace length_of_ad_l549_549804

theorem length_of_ad (AB CD AD BC : ℝ) 
  (h1 : AB = 10) 
  (h2 : CD = 2 * AB) 
  (h3 : AD = BC) 
  (h4 : AB + BC + CD + AD = 42) : AD = 6 :=
by
  -- proof omitted
  sorry

end length_of_ad_l549_549804


namespace path_from_1_1_to_vertex_l549_549461

def vertex_set : Set (ℤ × ℤ) := { (a, b) | Int.gcd a b = 1 }

noncomputable def edge_set (a b : ℤ) : Set (ℤ × ℤ) := 
  { (a, b + k * a * b) | k : ℤ } ∪ { (a + k * a * b, b) | k : ℤ }

def path_exists_from (a b : ℤ) : Prop := ∃ p : List (ℤ × ℤ), List.head! p = (1, 1) ∧ List.last! p = (a, b) ∧ ∀ x y ∈ p, ∃ (z ∈ edge_set x y), z = y

theorem path_from_1_1_to_vertex (a b : ℤ) (h : (a, b) ∈ vertex_set) : path_exists_from a b :=
sorry

end path_from_1_1_to_vertex_l549_549461


namespace product_gcd_lcm_l549_549305

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l549_549305


namespace semicircle_area_ratio_l549_549132

open Real

theorem semicircle_area_ratio (r : ℝ) (hr : 0 < r) :
  let r_POQ := r / 2,
      r_ROS := (sqrt 2 / 2) * r,
      area_POQ := (pi * r_POQ^2) / 2,
      area_ROS := (pi * r_ROS^2) / 2,
      combined_area := area_POQ + area_ROS,
      area_circle := pi * r^2
  in combined_area / area_circle = 3 / 8 :=
by
  sorry

end semicircle_area_ratio_l549_549132


namespace misha_total_students_l549_549469

-- Definitions based on the conditions
def misha_best_rank : ℕ := 75
def misha_worst_rank : ℕ := 75

-- Statement of the theorem to be proved
theorem misha_total_students (misha_is_best : misha_best_rank = 75) (misha_is_worst : misha_worst_rank = 75) : 
  (misha_best_rank - 1) + (misha_worst_rank - 1) + 1 = 149 :=
by
  sorry

end misha_total_students_l549_549469


namespace dawn_hours_l549_549186

-- Define the conditions
def pedestrian_walked_from_A_to_B (x : ℕ) : Prop :=
  x > 0

def pedestrian_walked_from_B_to_A (x : ℕ) : Prop :=
  x > 0

def met_at_noon (x : ℕ) : Prop :=
  x > 0

def arrived_at_B_at_4pm (x : ℕ) : Prop :=
  x > 0

def arrived_at_A_at_9pm (x : ℕ) : Prop :=
  x > 0

-- Define the theorem to prove
theorem dawn_hours (x : ℕ) :
  pedestrian_walked_from_A_to_B x ∧ 
  pedestrian_walked_from_B_to_A x ∧
  met_at_noon x ∧ 
  arrived_at_B_at_4pm x ∧ 
  arrived_at_A_at_9pm x → 
  x = 6 := 
sorry

end dawn_hours_l549_549186


namespace total_cost_for_new_seats_l549_549613

-- Define the conditions
def seat_cost : ℕ := 30
def rows : ℕ := 5
def seats_per_row : ℕ := 8
def discount_percentage : ℕ := 10

-- Define the proof statement based on the conditions
theorem total_cost_for_new_seats:
  let total_seats := rows * seats_per_row in
  let cost_per_ten_seats := 10 * seat_cost in
  let discount_per_ten_seats := cost_per_ten_seats * discount_percentage / 100 in
  let discounted_cost_per_ten_seats := cost_per_ten_seats - discount_per_ten_seats in
  let num_sets_of_ten := total_seats / 10 in
  let total_cost := num_sets_of_ten * discounted_cost_per_ten_seats in
  total_cost = 1080 :=
by
  sorry

end total_cost_for_new_seats_l549_549613


namespace necessary_but_not_sufficient_l549_549988

-- Define the propositions
def Q : Prop := "I am a senior high school student majoring in liberal arts at Yueyang No.1 High School"
def D : Prop := "Some class"

-- The statement that D is a necessary but not sufficient condition for Q
theorem necessary_but_not_sufficient : (Q → D) ∧ ¬(D → Q) := 
by
  sorry

end necessary_but_not_sufficient_l549_549988


namespace perp_bisect_cd_l549_549891

theorem perp_bisect_cd (A B C D M P Q : Point) (h_cyclic : cyclic A B C D)
  (h_diagonals : ∠AMB = 90 ∧ ∠CMD = 90) 
  (h_perp_AB : ∠MP = 90) 
  (h_perp_to_CD_at_Q : M = intersection_of_diagonals A B C D ∧ Q = foot_of_perpendicular M C D ∧ line_through M perpendicular D Q) :
  divides_evenly Q C D :=
sorry

end perp_bisect_cd_l549_549891


namespace extreme_values_enclosed_area_l549_549739

noncomputable def f (x a b : ℝ) : ℝ := x^2 - 2*a*x + b

-- Condition: Local extremum of 2 at x = 1
def local_extremum_cond (a b : ℝ) : Prop :=
  (f 1 a b = 2) ∧ (2*1 - 2*a = 0)

-- Question 1: Maximum and minimum values on [0, 3]
theorem extreme_values (a b : ℝ) (h1 : local_extremum_cond a b) :
  max (f 0 a b) (f 3 a b) = 6 ∧ min (f 0 a b) (f 1 a b) = 2 :=
sorry

-- Question 2: Area enclosed by the curves
theorem enclosed_area (a b : ℝ) (h1 : local_extremum_cond a b) :
  ∫ x in (0 : ℝ)..(3 : ℝ), ((λ x, x + 3) x - f x a b) = 9 / 2 :=
sorry

end extreme_values_enclosed_area_l549_549739


namespace conjugate_of_z_l549_549001

theorem conjugate_of_z (z : ℂ) (h : z * (3 - 4 * complex.i) = 1 + 2 * complex.i) :
  complex.conj z = -1 / 5 - 2 / 5 * complex.i :=
sorry

end conjugate_of_z_l549_549001


namespace least_number_l549_549679

theorem least_number (n p q r s : ℕ) : 
  (n + p) % 24 = 0 ∧ 
  (n + q) % 32 = 0 ∧ 
  (n + r) % 36 = 0 ∧
  (n + s) % 54 = 0 →
  n = 863 :=
sorry

end least_number_l549_549679


namespace express_b_c_range_a_not_monotonic_l549_549738

noncomputable def f (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) * Real.exp (-x)
noncomputable def f' (a b c x : ℝ) : ℝ := 
    (a * x^2 + b * x + c) * (-Real.exp (-x)) + (2 * a * x + b) * Real.exp (-x)

theorem express_b_c (a : ℝ) : 
    (∃ b c : ℝ, f a b c 0 = 2 * a ∧ f' a b c 0 = Real.pi / 4) → 
    (∃ b c : ℝ, b = 1 + 2 * a ∧ c = 2 * a) := 
sorry

noncomputable def g (a x : ℝ) : ℝ := -a * x^2 - x + 1

theorem range_a_not_monotonic (a : ℝ) : 
    (¬ (∀ x y : ℝ, x ∈ Set.Ici (1 / 2) → y ∈ Set.Ici (1 / 2) → x < y → g a x ≤ g a y)) → 
    (-1 / 4 < a ∧ a < 2) := 
sorry

end express_b_c_range_a_not_monotonic_l549_549738


namespace count_multiples_5_or_7_not_35_l549_549017

def count_multiples_5 (n : ℕ) : ℕ := n / 5
def count_multiples_7 (n : ℕ) : ℕ := n / 7
def count_multiples_35 (n : ℕ) : ℕ := n / 35
def inclusion_exclusion (a b c : ℕ) : ℕ := a + b - c

theorem count_multiples_5_or_7_not_35 : 
  inclusion_exclusion (count_multiples_5 3000) (count_multiples_7 3000) (count_multiples_35 3000) = 943 :=
by
  sorry

end count_multiples_5_or_7_not_35_l549_549017


namespace translate_parabola_l549_549181

theorem translate_parabola (x : ℝ) :
  (x^2 + 3) = (x - 5)^2 + 3 :=
sorry

end translate_parabola_l549_549181


namespace heartsuit_zero_heartsuit_self_heartsuit_pos_l549_549278

def heartsuit (x y : Real) : Real := x^2 - y^2

theorem heartsuit_zero (x : Real) : heartsuit x 0 = x^2 :=
by
  sorry

theorem heartsuit_self (x : Real) : heartsuit x x = 0 :=
by
  sorry

theorem heartsuit_pos (x y : Real) (h : x > y) : heartsuit x y > 0 :=
by
  sorry

end heartsuit_zero_heartsuit_self_heartsuit_pos_l549_549278


namespace complement_of_A_l549_549011

open Set

theorem complement_of_A (U : Set ℝ) (A : Set ℝ) (hU : U = univ) (hA : A = {x : ℝ | x / (x - 1) > 0}) :
  U \ A = Ici 0 \ Ioi 1 :=
by
  rw [hU, univ_diff]
  exact sorry

end complement_of_A_l549_549011


namespace moon_speed_kmps_l549_549510

theorem moon_speed_kmps (speed_kmph : ℕ) (h : speed_kmph = 720) : speed_kmph / 3600 = 0.2 := 
by {
  -- skipping the proof
  sorry
}

end moon_speed_kmps_l549_549510


namespace remaining_rectangle_perimeter_l549_549594

theorem remaining_rectangle_perimeter :
  let original_side := 2018
  let larger_side := 2000
  let remaining_area := original_side^2 - larger_side^2
  ∃ (a b : ℤ), a * b = remaining_area ∧ abs (a - b) < 40 ∧ 2 * (a + b) = 1076 := 
  let original_side := 2018
  let larger_side := 2000
  let remaining_area := original_side^2 - larger_side^2
  ⟨270, 268, 
  by {
    simp [original_side, larger_side, remaining_area],
    norm_num,
  }, by norm_num,
  by norm_num⟩

end remaining_rectangle_perimeter_l549_549594


namespace max_abs_cubic_at_least_one_fourth_l549_549091

def cubic_polynomial (p q r x : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

theorem max_abs_cubic_at_least_one_fourth (p q r : ℝ) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, |cubic_polynomial p q r x| ≥ 1 / 4 :=
by
  sorry

end max_abs_cubic_at_least_one_fourth_l549_549091


namespace total_widgets_sold_after_20_days_l549_549849

-- Definition of the arithmetic sequence
def widgets_sold_on_day (n : ℕ) : ℕ :=
  2 * n - 1

-- Sum of the first n terms of the sequence
def sum_of_widgets_sold (n : ℕ) : ℕ :=
  n * (widgets_sold_on_day 1 + widgets_sold_on_day n) / 2

-- Prove that the total widgets sold after 20 days is 400
theorem total_widgets_sold_after_20_days : sum_of_widgets_sold 20 = 400 :=
by
  sorry

end total_widgets_sold_after_20_days_l549_549849


namespace class_size_l549_549606

theorem class_size (n : ℕ) (h1 : 85 - 33 + 90 - 40 = 102) (h2 : (102 : ℚ) / n = 1.5): n = 68 :=
by
  sorry

end class_size_l549_549606


namespace complex_quadratic_power_l549_549574

theorem complex_quadratic_power :
  (1 - Complex.i) ^ 4 = -4 :=
by
  sorry

end complex_quadratic_power_l549_549574


namespace num_remaining_integers_l549_549520

namespace ProofProblem

def T : Finset ℕ := (Finset.range 101).filter (λ x => x > 0)

def is_multiple (n : ℕ) (d : ℕ) : Prop := d ∣ n

def remove_multiples (S : Finset ℕ) (d : ℕ) : Finset ℕ := S.filter (λ x => ¬ is_multiple x d)

def T_removed_multiples_2_and_5 : Finset ℕ :=
  remove_multiples (remove_multiples T 2) 5

theorem num_remaining_integers : T_removed_multiples_2_and_5.card = 40 := 
  sorry

end ProofProblem

end num_remaining_integers_l549_549520


namespace crocodile_coloring_l549_549102

theorem crocodile_coloring (m n : ℤ) : 
  ∃ (coloring : ℤ × ℤ → ℕ), 
  (∀ x y : ℤ, coloring (x, y) = 0 ∨ coloring (x, y) = 1) ∧
  (∀ x y : ℤ, let move_1 := (x + m, y + n),
               let move_2 := (x + n, y + m) in
               coloring (x, y) ≠ coloring move_1 ∧ coloring (x, y) ≠ coloring move_2) :=
sorry

end crocodile_coloring_l549_549102


namespace geometric_to_arithmetic_common_ratio_greater_than_1_9_l549_549560

theorem geometric_to_arithmetic (q : ℝ) (h : q = (1 + Real.sqrt 5) / 2) :
  ∃ (a b c : ℝ), b - a = c - b ∧ a / b = b / c := 
sorry

theorem common_ratio_greater_than_1_9 (q : ℝ) (h_pos : q > 1.9 ∧ q < 2) :
  ∃ (n : ℕ), q^(n+1) - 2 * q^n + 1 = 0 :=
sorry

end geometric_to_arithmetic_common_ratio_greater_than_1_9_l549_549560


namespace perimeter_inequality_l549_549605

-- Define the problem parameters
variables {R S : ℝ}  -- radius and area of the inscribed polygon
variables (P : ℝ)    -- perimeter of the convex polygon formed by chosen points

-- Define the various conditions
def circle_with_polygon (r : ℝ) := r > 0 -- Circle with positive radius
def polygon_with_area (s : ℝ) := s > 0 -- Polygon with positive area

-- Main theorem to be proven
theorem perimeter_inequality (hR : circle_with_polygon R) (hS : polygon_with_area S) :
  P ≥ (2 * S / R) :=
sorry

end perimeter_inequality_l549_549605


namespace total_time_to_braid_hair_l549_549065

constant dancers : ℕ := 8
constant braidsPerDancer : ℕ := 5
constant secondsPerBraid : ℕ := 30
constant secondsPerMinute : ℕ := 60

theorem total_time_to_braid_hair : 
  (dancers * braidsPerDancer * secondsPerBraid) / secondsPerMinute = 20 := 
by
  sorry

end total_time_to_braid_hair_l549_549065


namespace digit_not_5_l549_549234

variable (digit : ℕ)
variable (three_true_one_false : (digit = 4 ∨ ¬ (digit = 5) ∨ digit = 6 ∨ ¬ (digit = 7)) ∧ 
                                 (digit = 4 → (¬ (digit = 5) ∧ ¬ (digit = 6) ∧ ¬ (digit = 7))) ∧ 
                                 (¬ (digit = 5) → (¬ (digit = 7) ∧ (digit = 4 ∨ digit = 6))) ∧ 
                                 (digit = 6 → (¬ (digit = 5) ∧ (digit = 4 → false) ∧ ¬ (digit = 7))) ∧
                                 (¬ (digit = 7) → (¬ (digit = 5) ∧ (digit = 4 ∨ digit = 6))))

theorem digit_not_5 (h : three_true_one_false) : ¬ (digit = 5) :=
sorry

end digit_not_5_l549_549234


namespace Katie_homework_problems_l549_549813

theorem Katie_homework_problems :
  let finished_problems := 5
  let remaining_problems := 4
  let total_problems := finished_problems + remaining_problems
  total_problems = 9 :=
by
  sorry

end Katie_homework_problems_l549_549813


namespace final_number_after_99_operations_l549_549149

theorem final_number_after_99_operations :
  let S : Finset ℕ := Finset.range 101 \ {0} in
  let op := λ (a b : ℕ), a + b - 1 in
  ∃ x : ℕ, 
    (∀ n ∈ S, ∃ op_seq : Fin (n - 1) → (ℕ × ℕ),
      op_seq ∀ (k : Fin (n - 1)), 
        (op_seq k).1 ∈ S ∧ (op_seq k).2 ∈ S ∧
        x = Finset.sum ((S : Set ℕ) \ {op_seq k | k ∈ Fin (n - 1)} ∪ {op (op_seq k).1 (op_seq k).2 | k ∈ Fin (n - 1)})) ∧
    x = 4951 :=
sorry

end final_number_after_99_operations_l549_549149


namespace dolls_in_dollhouses_l549_549842

theorem dolls_in_dollhouses :
  let total_ways := Nat.choose 7 2 * 6 * Nat.factorial 5 in
  total_ways = 15120 := by
  sorry

end dolls_in_dollhouses_l549_549842


namespace shift_parabola_left_l549_549501

theorem shift_parabola_left (x : ℝ) : (x + 1)^2 = y ↔ x^2 = y :=
sorry

end shift_parabola_left_l549_549501


namespace monotonicity_and_a_range_l549_549007

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + x^2 - x * log a

theorem monotonicity_and_a_range 
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x ∈ Ioi (0 : ℝ), derivate (f a) x > 0) ∧ 
  (∀ x ∈ Iio (0 : ℝ), derivate (f a) x < 0) ∧
  (∃ x1 x2 : ℝ, x1 ∈ Icc (-1) 1 ∧ x2 ∈ Icc (-1) 1 ∧ abs (f a x1 - f a x2) ≥ real.exp 1 - 1 → 
  (a ∈ Set.Icc (0 : ℝ) (1 / real.exp 1) ∪ Set.Ici (real.exp 1))) := 
sorry

end monotonicity_and_a_range_l549_549007


namespace misha_grade_students_l549_549465

theorem misha_grade_students (n : ℕ) (h1 : n = 75) (h2 : n = 75) : 2 * n - 1 = 149 := 
by
  sorry

end misha_grade_students_l549_549465


namespace cost_per_person_trip_trips_rental_cost_l549_549223

-- Define the initial conditions
def ticket_price_per_person := 60
def total_employees := 70
def small_car_seats := 4
def large_car_seats := 11
def extra_cost_small_car_per_person := 5
def extra_revenue_large_car := 50
def max_total_cost := 5000

-- Define the costs per person per trip for small and large cars
def large_car_cost_per_person := 10
def small_car_cost_per_person := large_car_cost_per_person + extra_cost_small_car_per_person

-- Define the number of trips for four-seater and eleven-seater cars
def four_seater_trips := 1
def eleven_seater_trips := 6

-- Prove the lean statements
theorem cost_per_person_trip : 
  (11 * large_car_cost_per_person) - (small_car_seats * small_car_cost_per_person) = extra_revenue_large_car := 
sorry

theorem trips_rental_cost (x y : ℕ) : 
  (small_car_seats * x + large_car_seats * y = total_employees) ∧
  ((total_employees * ticket_price_per_person) + (small_car_cost_per_person * small_car_seats * x) + (large_car_cost_per_person * large_car_seats * y) ≤ max_total_cost) :=
sorry

end cost_per_person_trip_trips_rental_cost_l549_549223


namespace incorrect_plane_PDF_perp_ABC_l549_549417

section
variables (P A B C D E F : Type)

-- Assumptions based on the conditions
variables [regular_tetrahedron P A B C]
variables [midpoint D A B] [midpoint E B C] [midpoint F C A]

-- The Lean statement representing the incorrectness of the given statement
theorem incorrect_plane_PDF_perp_ABC : ¬ (plane PDF ⊥ plane ABC) :=
sorry
end

end incorrect_plane_PDF_perp_ABC_l549_549417


namespace therapy_charge_l549_549221

variable (F A : ℝ)

-- Given conditions
def condition1 : Prop := F = A + 35
def condition2 : Prop := F + 4 * A = 350

-- Goal to prove
def goal : Prop := F + A = 161

theorem therapy_charge (h1 : condition1) (h2 : condition2) : goal :=
by
  -- Omitted the actual proof steps as per the instructions
  sorry

end therapy_charge_l549_549221


namespace solve_for_x_l549_549333

theorem solve_for_x (x : ℝ) (h : Real.log x 8 = 3) : x = 2 := 
sorry

end solve_for_x_l549_549333


namespace min_value_a8_lambda_a9_l549_549396

theorem min_value_a8_lambda_a9 (a : ℕ → ℝ) (λ q : ℝ) (h1 : ∀ n, a n = a 1 * q ^ (n - 1))
  (h2 : 1 + (a 2 - a 4) + λ * (a 3 - a 5) = 0) (h3 : q > 1) : (a 8 + λ * a 9) = 27 / 4 := 
sorry

end min_value_a8_lambda_a9_l549_549396


namespace perpendicular_transfer_l549_549353

variables {Line Plane : Type} 
variables (a b : Line) (α β : Plane)

def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry

theorem perpendicular_transfer
  (h1 : perpendicular a α)
  (h2 : parallel_planes α β) :
  perpendicular a β := 
sorry

end perpendicular_transfer_l549_549353


namespace shaded_triangle_area_l549_549256
-- Import the entire Mathlib library

-- Define areas of four parallelograms A, B, C, D
variables (area_A area_B area_C area_D : ℝ)

-- Given conditions
def conditions : Prop :=
  area_A = 30 ∧
  area_B = 15 ∧
  area_C = 20

-- Prove that the area of the shaded triangle is 30
theorem shaded_triangle_area (h : conditions) : ∃ I : ℝ, I = 30 :=
by
  exists 30
  sorry

end shaded_triangle_area_l549_549256


namespace bicentric_quad_lemma_l549_549705

-- Define the properties and radii of the bicentric quadrilateral
variables (KLMN : Type) (r ρ h : ℝ)

-- Assuming quadrilateral KLMN is bicentric with given radii
def is_bicentric (KLMN : Type) := true

-- State the theorem we wish to prove
theorem bicentric_quad_lemma (br : is_bicentric KLMN) : 
  (1 / (ρ + h) ^ 2) + (1 / (ρ - h) ^ 2) = (1 / r ^ 2) :=
sorry

end bicentric_quad_lemma_l549_549705


namespace probability_diff_color_ball_l549_549043

variable (boxA : List String) (boxB : List String)
def P_A (boxA := ["white", "white", "red", "red", "black"]) (boxB := ["white", "white", "white", "white", "red", "red", "red", "black", "black"]) : ℚ := sorry

theorem probability_diff_color_ball :
  P_A boxA boxB = 29 / 50 :=
sorry

end probability_diff_color_ball_l549_549043


namespace graph_shift_l549_549899

theorem graph_shift :
  ∀ x : ℝ, 
  (∀ x, cos x + sqrt 3 * sin x = 2 * sin (π / 6 + x)) →
  (∀ x, sin x - sqrt 3 * cos x = 2 * sin (x + 5 * π / 3)) →
  (∀ x, sin x - sqrt 3 * cos x = cos (x - 3 * π / 2) + sqrt 3 * sin (x - 3 * π / 2)) :=
by
  intros x hx1 hx2
  sorry

end graph_shift_l549_549899


namespace problem1_problem2_l549_549002

-- Declare the inputs
variables {x y m : ℝ}
variables {l_lhs l_rhs : ℝ} -- new line equation ax + by + c = 0

-- Define the given problem's conditions
def circle_eq (m : ℝ) : ℝ :=
  x^2 + y^2 - 2*x - 4*y + m

-- Statement 1: Finding the correct value for m
theorem problem1 (h : m < 5) (h_distance : |(λ x y : ℝ, x + 2*y -4 = 0)(x, y)| = (4*sqrt(5))/5) : circle_eq(m) = 0 → m = 4 :=
sorry

-- Statement 2: Finding the existence of line l using radius and center conditions
theorem problem2 (h : m = 4) : 
  ∃ c : ℝ, (∀ (x y : ℝ), circle_eq(4) = 0 → |(λ x y : ℝ, x - 2*y + c = 0)(x, y)| = sqrt(5)/5) ∧ (4 - sqrt(5) < c ∧ c < 2 + sqrt(5)) :=
sorry


end problem1_problem2_l549_549002


namespace mean_of_five_numbers_is_correct_l549_549167

-- Define the given sum of five numbers as three-quarters
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers, which is 5
def number_of_numbers : ℕ := 5

-- Define the mean calculation from the given sum and number of numbers
def mean_five_numbers (sum : ℚ) (count : ℕ) : ℚ := sum / count

-- Statement to prove: the mean of five numbers given their sum is 3/4 equals 3/20
theorem mean_of_five_numbers_is_correct :
  mean_five_numbers sum_of_five_numbers number_of_numbers = 3 / 20 :=
by
  -- Skipping the proof
  sorry

end mean_of_five_numbers_is_correct_l549_549167


namespace angle_trisector_triangle_l549_549420

theorem angle_trisector_triangle :
  ∀ (A B C A1 B1 C1 : Type) [IsTriangle A B C] (angleB120 : ∠ABC = 120°)
  (AA1_bisector : IsAngleBisector A A1)
  (BB1_bisector : IsAngleBisector B B1)
  (CC1_bisector : IsAngleBisector C C1),
  ∠A1B1C1 = 90° :=
by
  intros,
  sorry

end angle_trisector_triangle_l549_549420


namespace count_valid_subsets_l549_549838

theorem count_valid_subsets : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ A ∈ S, A ⊆ {1, 2, 3, 4, 5} ∧ 
    (∀ a ∈ A, 6 - a ∈ A)) ∧ 
    S.card = 7 := 
sorry

end count_valid_subsets_l549_549838


namespace problem_solution_l549_549721

-- Define propositions p and q
def p : Prop := ∃ x : ℝ, 2^x ≤ 3^x
def q : Prop := ¬ (¬ ∃ x : ℝ, ¬ (e^x > 0))

-- Theorem to prove that p ∨ q is true given the conditions
theorem problem_solution : p ∨ q := by
  sorry

end problem_solution_l549_549721


namespace find_extrema_of_A_l549_549702

theorem find_extrema_of_A (x y : ℝ) (h : x^2 + y^2 = 4) : 2 ≤ x^2 + x * y + y^2 ∧ x^2 + x * y + y^2 ≤ 6 :=
by 
  sorry

end find_extrema_of_A_l549_549702


namespace number_notebooks_in_smaller_package_l549_549112

theorem number_notebooks_in_smaller_package 
  (total_notebooks : ℕ)
  (large_packs : ℕ)
  (notebooks_per_large_pack : ℕ)
  (condition_1 : total_notebooks = 69)
  (condition_2 : large_packs = 7)
  (condition_3 : notebooks_per_large_pack = 7)
  (condition_4 : ∃ x : ℕ, x < 7 ∧ (total_notebooks - (large_packs * notebooks_per_large_pack)) % x = 0) :
  ∃ x : ℕ, x < 7 ∧ x = 5 := 
by 
  sorry

end number_notebooks_in_smaller_package_l549_549112


namespace proof_passes_through_fourth_quadrant_range_of_m_if_not_third_quadrant_intercepts_equal_line_equation_l549_549744

-- Definitions
def line_l (m : ℝ) : ℝ × ℝ → Prop := λ (x y : ℝ), (m-1) * x + 2 * m * y + 2 = 0

-- Conditions
def passes_through_fourth_quadrant (m : ℝ) : Prop :=
  (∃ (x y : ℝ), line_l m (x, y) ∧ x > 0 ∧ y < 0)

def does_not_pass_through_third_quadrant (m : ℝ) : Prop :=
  ¬ (∃ (x y : ℝ), line_l m (x, y) ∧ x < 0 ∧ y < 0)

def intercepts_equal (x y : ℝ) : Prop :=
  x = -1 / y

-- Statements
theorem proof_passes_through_fourth_quadrant (m : ℝ) : 
  passes_through_fourth_quadrant m :=
sorry

theorem range_of_m_if_not_third_quadrant (m : ℝ) :
  does_not_pass_through_third_quadrant m → (m ∈ set.Iic 0) :=
sorry

theorem intercepts_equal_line_equation (m : ℝ) :
  intercepts_equal (1 - m) 2 → m = -1 ∧ line_l m (1, 1) :=
sorry

end proof_passes_through_fourth_quadrant_range_of_m_if_not_third_quadrant_intercepts_equal_line_equation_l549_549744


namespace roots_always_real_triangle_perimeter_l549_549367

theorem roots_always_real (k : ℝ) : ∃ x1 x2 : ℝ, x^2 + (3 * k - 2) * x - 6 * k = 0 :=
by
  -- Translating the steps from the solution to Lean's existence theorem checking 
  sorry

theorem triangle_perimeter (a b c : ℝ) (h_a : a = 6) (h_root1: b^2 + (3 * k - 2) * b - 6 * k = 0) 
(h_root2: c^2 + (3 * k - 2) * c - 6 * k = 0) (h_iso : a = b ∨ a = c) : 
Perimeter a b c = 14 :=
by
  -- Translating the steps from the solution to Lean's perimeter calculation
  sorry

end roots_always_real_triangle_perimeter_l549_549367


namespace max_n_is_4024_l549_549351

noncomputable def max_n_for_positive_sum (a : ℕ → ℝ) (d : ℝ) (h1 : d < 0) (h2 : a 1 > 0) (h3 : a 2013 * (a 2012 + a 2013) < 0) : ℕ :=
  4024

theorem max_n_is_4024 (a : ℕ → ℝ) (d : ℝ) (h1 : d < 0) (h2 : a 1 > 0) (h3 : a 2013 * (a 2012 + a 2013) < 0) :
  max_n_for_positive_sum a d h1 h2 h3 = 4024 :=
by
  sorry

end max_n_is_4024_l549_549351


namespace hyperbola_eccentricity_l549_549729

theorem hyperbola_eccentricity (a c e : ℝ) 
  (h₁ : a = 2) 
  (h₂ : c - a = 6) : 
  e = c / a :=
by 
  have h₃ : a = 2 := h₁,
  have h₄ : c - a = 6 := h₂,
  sorry

end hyperbola_eccentricity_l549_549729


namespace find_side_c_l549_549793

theorem find_side_c
  (a b : ℝ) (area : ℝ) (ha : a = 4) (hb : b = 3) (harea : area = 3 * real.sqrt 3) :
  ∃ c : ℝ, c = real.sqrt 13 :=
by
  sorry

end find_side_c_l549_549793


namespace abc_two_bc_l549_549130

open Real Set

variable (A B C D O : Point ℝ)
variable (r : ℝ)
variable (h_inscribed : is_inscribed A B C D O r)
variable (h_AB_diameter : distance A B = 2 * r)
variable (h_AD_parallel_OC : parallel A D O C)
variable (S_AOCD : ℝ)
variable (S_DBC : ℝ)
variable (h_area_relation : S_AOCD = 2 * S_DBC)

theorem abc_two_bc :
  distance A B = 2 * distance B C :=
sorry

end abc_two_bc_l549_549130


namespace spinner_probability_divisible_by_8_l549_549619

-- Define the spinner sections
def sections : Set ℕ := {1, 2, 3, 4}

-- Check if a number is divisible by 8
def divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

-- Define the event of interest where the number is divisible by 8
def event (hundreds tens units : ℕ) : Prop :=
  divisible_by_8 (100 * hundreds + 10 * tens + units)

-- The main theorem stating the probability is 1/8
theorem spinner_probability_divisible_by_8 :
  (∑ h in sections, ∑ t in sections, ∑ u in sections, if event h t u then 1 else 0) / (sections.card ^ 3) = 1 / 8 := by
  sorry

end spinner_probability_divisible_by_8_l549_549619


namespace gcd_lcm_product_24_60_l549_549303

theorem gcd_lcm_product_24_60 : 
  (Nat.gcd 24 60) * (Nat.lcm 24 60) = 1440 := by
  sorry

end gcd_lcm_product_24_60_l549_549303


namespace find_modulus_of_alpha_l549_549081

theorem find_modulus_of_alpha 
  (α β : ℂ) 
  (h_conjugate : β = conj(α))
  (h_real_ratio : (α / (β^2)).im = 0)
  (h_diff : |α - β| = 4 * real.sqrt 3) : 
  |α| = 4 :=
by sorry

end find_modulus_of_alpha_l549_549081


namespace always_possible_to_pair_l549_549174

noncomputable def wrestlers : Finset ℕ := {1, 2, ..., 36}

def strength (w : ℕ) : ℕ := -- Define a function representing the strength of each wrestler
  sorry

axiom stronger_wins (w1 w2 : ℕ) : w1 < w2 → strength w1 < strength w2

axiom equal_draw (w1 w2 : ℕ) : w1 = w2 → strength w1 = strength w2

/-- Prove that it is always possible to pair wrestlers such that
  1. Winners are not weaker than any wrestlers who drew or lost. 
  2. Wrestlers who drew are not weaker than those who lost. -/
theorem always_possible_to_pair :
  ∃ (pairs : list (ℕ × ℕ)),
  (∀ (p : ℕ × ℕ) ∈ pairs, (p.fst ∈ wrestlers) ∧ (p.snd ∈ wrestlers)) ∧ 
  (∀ (p : ℕ × ℕ) ∈ pairs, (p.fst < p.snd) ∨ (p.fst = p.snd)) ∧
  (∀ (p : ℕ × ℕ), (p ∈ pairs) → (strength p.fst ≥ strength p.snd)) :=
begin
  sorry
end

end always_possible_to_pair_l549_549174


namespace GrandmaOlga_grandchildren_l549_549756

theorem GrandmaOlga_grandchildren :
  (∃ d : ℕ, d = 3 ∧ ∀ i : Fin d, 6 ∈ ℕ) ∧
  (∃ s : ℕ, s = 3 ∧ ∀ j : Fin s, 5 ∈ ℕ) →
  18 + 15 = 33 :=
by
  intros h
  cases' h with h_d h_s
  cases' h_d with d_vals num_d
  cases' d_vals with d_eq d_cond
  cases' h_s with s_vals num_s
  cases' s_vals with s_eq s_cond
  sorry

end GrandmaOlga_grandchildren_l549_549756


namespace expression_of_f_range_of_m_l549_549514

-- Given conditions
def f (x : ℝ) : ℝ := x^2 - x + 1

-- Proof problem 1: The expression for f(x)
theorem expression_of_f (h1 : ∀ x, f (x + 1) - f x = 2 * x) (h2 : f 0 = 1) :
  f = (λ x, x^2 - x + 1) := sorry

-- Proof problem 2: Range of the real number m
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Icc (-1 : ℝ) 1, f x > 2 * x + m) → m < -1 := sorry

end expression_of_f_range_of_m_l549_549514


namespace H_is_orthocenter_of_triangle_ABC_l549_549831

open EuclideanGeometry

variables {A B C O H : Point}
variables [hCircumcenter : Circumcenter O A B C]
variables [hOrthoCenterCond : OH = OA + OB + OC]

theorem H_is_orthocenter_of_triangle_ABC :
  OrthoCenter H A B C :=
sorry

end H_is_orthocenter_of_triangle_ABC_l549_549831


namespace find_digits_l549_549795

-- Define the conditions as part of our Lean problem statement
variable (A B : ℕ)

-- Given conditions
axiom avg_score : ∀ n : ℕ, n > 0 → (A * 1000 + 860 + B) / n = 90
axiom four_digit : 1000 ≤ A * 1000 + 860 + B ∧ A * 1000 + 860 + B < 10000

-- The actual statement to be proven
theorem find_digits (h_avg : avg_score A B) (h_four_digit : four_digit A B) : A = 4 ∧ B = 0 :=
  sorry

end find_digits_l549_549795


namespace relationship_among_abc_l549_549379

-- Definitions of the three numbers
def a : ℝ := 0.6 ^ 0.3
def b : ℝ := Real.log 3 / Real.log 0.6
def c : ℝ := Real.log Real.pi

-- Theorem stating the relationship among a, b, and c
theorem relationship_among_abc : b < a ∧ a < c :=
by
  sorry

end relationship_among_abc_l549_549379


namespace factorize_1_factorize_2_l549_549294

variable {a x y : ℝ}

theorem factorize_1 : 2 * a * x^2 - 8 * a * x * y + 8 * a * y^2 = 2 * a * (x - 2 * y)^2 := 
by
  sorry

theorem factorize_2 : 6 * x * y^2 - 9 * x^2 * y - y^3 = -y * (3 * x - y)^2 := 
by
  sorry

end factorize_1_factorize_2_l549_549294


namespace thirtieth_percentile_data_set_l549_549883

open_locale big_operators

theorem thirtieth_percentile_data_set : 
  let data_set := [68, 70, 80, 88, 89, 90, 96, 98] in
  ∃ p, p = data_set.nth_le (nat.ceil (8 * 0.3) - 1) sorry ∧ p = 80 := 
sorry

end thirtieth_percentile_data_set_l549_549883


namespace math_problem_l549_549033

theorem math_problem
  (a x : ℤ)
  (h1 : ∀ a, (0 ≤ a ∧ a < 7) → ∃ x, (x = 3 / (a - 3)) ∧ x ∈ ℤ)
  (h2 : ∀ a, (0 ≤ a ∧ a < 7) → y = (7-a)*x + a → (y ≥ 0 ∨ y ≤ 0)) :
  ∃ l : List ℤ, (∀ a ∈ l, 0 ≤ a ∧ a < 7 ∧ (a - 3) ∣ 3) ∧ (l.length = 3) :=
by
  sorry

end math_problem_l549_549033


namespace relationship_among_abc_l549_549443

noncomputable def a : ℝ := (1 / 2) ^ 10
noncomputable def b : ℝ := (1 / 5) ^ (-1 / 2)
noncomputable def c : ℝ := log (1 / 3) 10

theorem relationship_among_abc : b > a ∧ a > c := by
  sorry

end relationship_among_abc_l549_549443


namespace gcd_lcm_identity_l549_549569

open Nat

theorem gcd_lcm_identity 
  (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let gcd3 := gcd (gcd a b) c
  let lcm3 := lcm (lcm a b) c
  (lcm3 ^ 2) / ((lcm a b) * (lcm b c) * (lcm c a)) = (gcd3 ^ 2) / ((gcd a b) * (gcd b c) * (gcd c a)) :=
sorry

end gcd_lcm_identity_l549_549569


namespace sum_first_five_terms_eq_15_l549_549000

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d 

variable (a : ℕ → ℝ) (h_arith_seq : is_arithmetic_sequence a) (h_a3 : a 3 = 3)

theorem sum_first_five_terms_eq_15 : (a 1 + a 2 + a 3 + a 4 + a 5 = 15) :=
sorry

end sum_first_five_terms_eq_15_l549_549000


namespace same_heads_sum_l549_549248

-- Definition of the bias probability for the third coin
def biased_coin_heads_prob : ℚ := 5 / 8

-- Function to calculate the combined generating function coefficients
def combined_generating_function : list ℚ := [5, 13, 11, 3]

-- Calculation of the required probability
def same_heads_probability : ℚ := 81 / 256

-- Sum of the numerator and denominator of the reduced fraction
def sum_numerator_denominator : ℕ := 81 + 256

-- Theorem stating the equivalence of our condition to the problem given
theorem same_heads_sum :
  let prob := same_heads_probability in
  prob = 81 / 256 ∧ (prob.num + prob.denom = 337) :=
by
  sorry

end same_heads_sum_l549_549248


namespace circle_center_lies_on_another_circle_l549_549888

-- Definitions of the points
variables (A B C D K L P: Point)
-- Assumption of the conditions
variables (h_parallelogram: Parallelogram A B C D)
(h_bisector: IsBisector A B D K L)
(h_center_circle_CKL: Circumcenter P C K L)

theorem circle_center_lies_on_another_circle: OnCircle P B C D :=
sorry

end circle_center_lies_on_another_circle_l549_549888


namespace square_area_ratio_l549_549877

theorem square_area_ratio (s : ℝ) :
  ∃ (JKLM ABCD : ℝ) (AJ : ℝ), 
  ABCD = (4 * s) ^ 2 ∧
  AJ = 3 * (s) ∧
  JKLM = 2 * s^2 ∧
  (JKLM / ABCD) = 1/8 :=
begin
  let AJ := 3 * s,
  let JB := s,
  have h1 : (4 * s) = AJ + JB, from sorry,
  have h2 : (AJ = 3 * JB), from sorry,
  let JKLM := 2 * s^2,
  let ABCD := (4 * s) ^ 2,
  have ratio : (JKLM / ABCD) = 1/8, from sorry,
  exact ⟨JKLM, ABCD, AJ, h1, h2, JKLM, ABCD, ratio⟩,
end

end square_area_ratio_l549_549877


namespace length_BK_l549_549060

theorem length_BK (A B C K : Point)
    (hAK : dist A K = 1)
    (hKC : dist K C = Real.sqrt 3)
    (hAngleAKC : ∠ A K C = 120)
    (hAngleABK : ∠ A B K = 15)
    (hAngleKBC : ∠ K B C = 15) :
    dist B K = (3 * Real.sqrt 2 + Real.sqrt 6) / 2 := 
by
  sorry

end length_BK_l549_549060


namespace maria_compensation_l549_549108

def deposit_insurance (deposit : ℕ) : ℕ :=
  if deposit <= 1400000 then deposit else 1400000

def maria_deposit : ℕ := 1600000

theorem maria_compensation :
  deposit_insurance maria_deposit = 1400000 :=
by sorry

end maria_compensation_l549_549108


namespace cube_surface_area_to_volume_ratio_l549_549910

theorem cube_surface_area_to_volume_ratio (edge_length : ℕ) (h : edge_length = 1) :
  (6 * edge_length^2) / (edge_length^3) = 6 := by
  -- Edge length is specified to be 1
  have h1 : edge_length = 1 := h
  -- Calculate surface area
  let surface_area := 6 * edge_length^2
  -- Calculate volume
  let volume := edge_length^3
  -- Compute the ratio
  calc
    (6 * edge_length^2) / (edge_length^3) = 6 * 1^2 / 1^3 : by rw [h1]
                                  ... = 6 : by rfl
  sorry

end cube_surface_area_to_volume_ratio_l549_549910


namespace max_price_of_most_expensive_product_l549_549202

noncomputable def greatest_possible_price
  (num_products : ℕ)
  (avg_price : ℕ)
  (min_price : ℕ)
  (mid_price : ℕ)
  (higher_price_count : ℕ)
  (total_retail_price : ℕ)
  (least_expensive_total_price : ℕ)
  (remaining_price : ℕ)
  (less_expensive_total_price : ℕ) : ℕ :=
  total_retail_price - least_expensive_total_price - less_expensive_total_price

theorem max_price_of_most_expensive_product :
  greatest_possible_price 20 1200 400 1000 10 (20 * 1200) (10 * 400) (20 * 1200 - 10 * 400) (9 * 1000) = 11000 :=
by
  sorry

end max_price_of_most_expensive_product_l549_549202


namespace Linda_cookie_batches_l549_549462

theorem Linda_cookie_batches (classmates : ℕ) (cookies_per_classmate : ℕ) (cookies_per_batch : ℕ) 
  (chocolate_chip_batches : ℕ) (oatmeal_raisin_batches : ℕ) : 
  classmates = 24 → cookies_per_classmate = 10 → 
  cookies_per_batch = 48 → chocolate_chip_batches = 2 → 
  oatmeal_raisin_batches = 1 → 
  (classmates * cookies_per_classmate - (chocolate_chip_batches + oatmeal_raisin_batches) * cookies_per_batch) / cookies_per_batch = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end Linda_cookie_batches_l549_549462


namespace find_extrema_l549_549296

noncomputable def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 5*x^3 + 3

theorem find_extrema :
  let interval := Set.Icc (-1 : ℝ) 2,
      critical_points := (Set.Icc (-1 : ℝ) 2).image f ∪ {f (-1), f 2} in
  (∀ x ∈ interval, f x ≥ -8) ∧ (∃ x ∈ interval, f x = -8) ∧
  (∀ x ∈ interval, f x ≤ 4) ∧ (∃ x ∈ interval, f x = 4) := by
  sorry

end find_extrema_l549_549296


namespace find_m_l549_549032

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  2 * Real.log x - m / x

def f_deriv (m : ℝ) (x : ℝ) : ℝ :=
  (2 / x) - (m / x^2)

theorem find_m :
  ∃ m : ℝ, f_deriv m 1 = 3 := 
begin
  use 1,
  norm_num,
end

end find_m_l549_549032


namespace magician_trick_l549_549984

theorem magician_trick (T : ℕ) (cards : fin 52) (edge_choice : fin 52 → bool) :
  ∀ init_position, (init_position = 0 ∨ init_position = 51) →
  (∃ remaining_position, remaining_position = init_position ∧ (∀ k, cards k = true → k ≠ remaining_position)) :=
sorry

end magician_trick_l549_549984


namespace at_most_three_integers_l549_549077

def polynomial := λ (a b c d x : ℝ), (a * x^3 + b * x^2 + c * x + d)

theorem at_most_three_integers (a b c d : ℝ) (h : a > 4 / 3) :
  (∃ n : ℕ, ∀ u v w r : ℤ, (u < v) → (v < w) → (w < r) →
    ((polynomial a b c d u).abs ≤ 1) → 
    ((polynomial a b c d v).abs ≤ 1) → 
    ((polynomial a b c d w).abs ≤ 1) → 
    ((polynomial a b c d r).abs ≤ 1) → False) :=
sorry

end at_most_three_integers_l549_549077


namespace sum_of_consecutive_integers_l549_549782

theorem sum_of_consecutive_integers (a b : ℕ) 
  (h1 : a > b) 
  (h2 : b = a - 1)
  (h3 : ((5 / 3) * (6 / 5) * (7 / 6) * ... * (a / b)) = 15) : 
  a + b = 89 := 
sorry

end sum_of_consecutive_integers_l549_549782


namespace problem_number_eq_7_5_l549_549116

noncomputable def number : ℝ := 7.5

theorem problem_number_eq_7_5 :
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = number :=
by
  sorry

end problem_number_eq_7_5_l549_549116


namespace find_b_fixed_point_extremum_l549_549775

theorem find_b_fixed_point_extremum (f : ℝ → ℝ) (b : ℝ) :
  (∀ x : ℝ, f x = x ^ 3 + b * x + 3) →
  (∃ x₀ : ℝ, f x₀ = x₀ ∧ (∀ x : ℝ, deriv f x₀ = 3 * x₀ ^ 2 + b) ∧ deriv f x₀ = 0) →
  b = -3 :=
by
  sorry

end find_b_fixed_point_extremum_l549_549775


namespace tank_missing_water_l549_549597

def max_capacity := 350000
def loss1_rate := 32000
def loss1_duration := 5
def loss2_rate := 10000
def loss2_duration := 10
def fill_rate := 40000
def fill_duration := 3

theorem tank_missing_water 
  (max_capacity = 350000)
  (loss1_rate = 32000)
  (loss1_duration = 5)
  (loss2_rate = 10000)
  (loss2_duration = 10)
  (fill_rate = 40000)
  (fill_duration = 3) : 
  (max_capacity - 
   ((loss1_rate * loss1_duration) + (loss2_rate * loss2_duration) - (fill_rate * fill_duration))) = 140000 :=
  by 
  unfold max_capacity loss1_rate loss1_duration loss2_rate loss2_duration fill_rate fill_duration
  sorry

end tank_missing_water_l549_549597


namespace mean_of_five_numbers_l549_549159

theorem mean_of_five_numbers (sum_of_numbers : ℚ) (number_of_elements : ℕ)
  (h_sum : sum_of_numbers = 3 / 4) (h_elements : number_of_elements = 5) :
  (sum_of_numbers / number_of_elements : ℚ) = 3 / 20 :=
by
  sorry

end mean_of_five_numbers_l549_549159


namespace sum_of_squares_of_perfect_cubes_lt_50_l549_549195

theorem sum_of_squares_of_perfect_cubes_lt_50 :
  ∑ (n : ℕ) in finset.filter (λ k, k < 50) (finset.image (λ n, n ^ 6) (finset.range 50)) = 1 :=
by
  sorry

end sum_of_squares_of_perfect_cubes_lt_50_l549_549195


namespace cyclic_quadrilateral_inequality_l549_549094

-- Define the distinct points on a circle and the cyclic quadrilateral property
variables (A B C D : Type) [dist A] [dist B] [dist C] [dist D] (circle : List (Type) := [A, B, C, D])

-- Define the sides of the quadrilateral and the longest side condition
noncomputable def side_lengths (AB BD CB CD : ℝ) (h_ab_longest : AB > CB) : Prop :=
  AB + BD > CB + CD

theorem cyclic_quadrilateral_inequality (A B C D : Type) [dist A] [dist B] [dist C] [dist D] 
  (circle : List (Type) := [A, B, C, D]) 
  (AB BD CB CD : ℝ) (h_ab_longest : AB > CB) :
  side_lengths AB BD CB CD h_ab_longest := by sorry

end cyclic_quadrilateral_inequality_l549_549094


namespace u_dot_v_cross_w_zero_l549_549837

variables (u v w : ℝ^3)

-- Conditions
def is_unit_vector (u : ℝ^3) : Prop := ∥u∥ = 1
def norm_is_three (v : ℝ^3) : Prop := ∥v∥ = 3
def w_definition1 (u v w : ℝ^3) : Prop := u × v = w + 2 • u
def w_definition2 (u v w : ℝ^3) : Prop := w × u = -v

-- Main statement
theorem u_dot_v_cross_w_zero 
  (h1 : is_unit_vector u) 
  (h2 : norm_is_three v) 
  (h3 : w_definition1 u v w) 
  (h4 : w_definition2 u v w) : 
  u • (v × w) = 0 :=
sorry

end u_dot_v_cross_w_zero_l549_549837


namespace sum_of_local_minimum_values_l549_549460

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - Real.cos x)

theorem sum_of_local_minimum_values : (∑ k in Finset.range (1008 + 1), -Real.exp (2 * (k : ℝ) * Real.pi + 2 * Real.pi)) = 
  - (Real.exp (2 * Real.pi) * (1 - Real.exp (2016 * Real.pi)) / (1 - Real.exp (2 * Real.pi))) :=
by
  sorry

end sum_of_local_minimum_values_l549_549460


namespace no_possible_sequence_of_moves_l549_549155

noncomputable def sequence (n : ℕ) : ℕ :=
  if n % 99 = 0 then 3 else if n % 2 = 0 then 2 else 1

def allowed_move (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c

theorem no_possible_sequence_of_moves :
  ¬ ∃ (moves : ℕ → ℕ), 
    (∀ n, allowed_move (sequence (n-1)) (sequence n) (sequence (n+1))) ∧
    (∀ m, 
      (moves n) = 
        if (m % 99) = 0 then 3 
        else if (m % 2) = 0 then 
          if m != 98 then 2 else 1 
        else 
          if (m + 1) % 2 = 1 then 1 else 2)
  :=
sorry

end no_possible_sequence_of_moves_l549_549155


namespace cost_of_running_tv_for_week_l549_549075

def powerUsage : ℕ := 125
def hoursPerDay : ℕ := 4
def costPerkWh : ℕ := 14

theorem cost_of_running_tv_for_week :
  let dailyConsumption := powerUsage * hoursPerDay
  let dailyConsumptionkWh := dailyConsumption / 1000
  let weeklyConsumption := dailyConsumptionkWh * 7
  let weeklyCost := weeklyConsumption * costPerkWh
  weeklyCost = 49 := by
  let dailyConsumption := powerUsage * hoursPerDay
  let dailyConsumptionkWh := dailyConsumption / 1000
  let weeklyConsumption := dailyConsumptionkWh * 7
  let weeklyCost := weeklyConsumption * costPerkWh
  sorry

end cost_of_running_tv_for_week_l549_549075


namespace shorter_side_length_l549_549983

variables (x y : ℝ)
variables (h1 : 2 * x + 2 * y = 60)
variables (h2 : x * y = 200)

theorem shorter_side_length :
  min x y = 10 :=
by
  sorry

end shorter_side_length_l549_549983


namespace solve_fraction_l549_549563

variables (w x y : ℝ)

-- Conditions
def condition1 := w / x = 2 / 3
def condition2 := w / y = 6 / 15

-- Statement
theorem solve_fraction (h1 : condition1 w x) (h2 : condition2 w y) : (x + y) / y = 8 / 5 :=
sorry

end solve_fraction_l549_549563


namespace cost_of_fencing_correct_l549_549967

noncomputable def cost_of_fencing (d : ℝ) (r : ℝ) : ℝ :=
  Real.pi * d * r

theorem cost_of_fencing_correct : cost_of_fencing 30 5 = 471 :=
by
  sorry

end cost_of_fencing_correct_l549_549967


namespace order_of_a_b_c_l549_549836

theorem order_of_a_b_c (x : ℝ) (h : x ∈ Ioo (-1/2 : ℝ) 0) : 
  let a := cos (sin (x * π)),
      b := sin (cos (x * π)),
      c := cos ((x + 1) * π)
  in c < b ∧ b < a :=
by
  let a := cos (sin (x * π))
  let b := sin (cos (x * π))
  let c := cos ((x + 1) * π)
  sorry

end order_of_a_b_c_l549_549836


namespace hexagon_distances_l549_549422

theorem hexagon_distances (P : ℝ × ℝ) (hex : fin 6 → ℝ × ℝ)
  (h_reg : ∀ i, ∥hex (i + 1) - hex i∥ = 1) -- hexagon with side length 1
  (h_inside : ∃ i j k, P ∈ triangle hex i j k) -- P inside the hexagon
  :
  ∃ i j k, (∥P - hex i∥ ≥ 1 ∧ ∥P - hex j∥ ≥ 1 ∧ ∥P - hex k∥ ≥ 1 ) :=
sorry

end hexagon_distances_l549_549422


namespace exists_similarity_mapping_midpoint_l549_549559

variable (V : Type) [normed_group V] [normed_space ℝ V] [finite_dimensional ℝ V]

structure Similarity (V : Type) [normed_group V] [normed_space ℝ V] :=
(center : V)
(map : V → V)
(is_similarity : ∀ x y, ∥map x - map y∥ = ∥x - y∥)

noncomputable def midpoint (x y : V) : V := (x + y) / 2

theorem exists_similarity_mapping_midpoint (T : Similarity V)
  (O : V) (hO: T.center = O):
  ∃ S : Similarity V, ∀ M : V, S.map M = midpoint M (T.map M) :=
by
  sorry

end exists_similarity_mapping_midpoint_l549_549559


namespace Tom_green_marbles_l549_549867

-- Define the given variables
def Sara_green_marbles : Nat := 3
def Total_green_marbles : Nat := 7

-- The statement to be proven
theorem Tom_green_marbles : (Total_green_marbles - Sara_green_marbles) = 4 := by
  sorry

end Tom_green_marbles_l549_549867


namespace triangle_A_equidistant_O_H_l549_549688

-- Required for the proof
variables {A B C O H : Type*} [metric_space A] [metric_space B] [metric_space C] 
variables (α β : ℝ) (R : ℝ)

-- Definitions based on conditions
def is_acute_triangle (A B C : Type*) : Prop := sorry
def is_circumcenter (O : Type*) (A B C : Type*) : Prop := sorry
def is_orthocenter (H : Type*) (A B C : Type*) : Prop := sorry
def equidistant (x y z : Type*) (d : ℝ) : Prop := dist x y = d ∧ dist y z = d

-- Main theorem statement
theorem triangle_A_equidistant_O_H (h_acute : is_acute_triangle A B C) 
  (hcircum : is_circumcenter O A B C) (horth : is_orthocenter H A B C)
  (heqdist : equidistant A O H R) : 
  α = 60 := sorry

end triangle_A_equidistant_O_H_l549_549688


namespace total_cost_for_new_seats_l549_549614

-- Define the conditions
def seat_cost : ℕ := 30
def rows : ℕ := 5
def seats_per_row : ℕ := 8
def discount_percentage : ℕ := 10

-- Define the proof statement based on the conditions
theorem total_cost_for_new_seats:
  let total_seats := rows * seats_per_row in
  let cost_per_ten_seats := 10 * seat_cost in
  let discount_per_ten_seats := cost_per_ten_seats * discount_percentage / 100 in
  let discounted_cost_per_ten_seats := cost_per_ten_seats - discount_per_ten_seats in
  let num_sets_of_ten := total_seats / 10 in
  let total_cost := num_sets_of_ten * discounted_cost_per_ten_seats in
  total_cost = 1080 :=
by
  sorry

end total_cost_for_new_seats_l549_549614


namespace probability_prime_and_multiple_of_11_l549_549873

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def multiples_of_11 (n : ℕ) : Prop :=
  11 ∣ n 

def number_of_primes_and_multiples_of_11 : ℕ :=
  (Finset.filter (λ n, is_prime n ∧ multiples_of_11 n) (Finset.range 61)).card

def total_number_of_cards := 60

-- Probability that the number on the card is prime and is a multiple of 11
theorem probability_prime_and_multiple_of_11 : (number_of_primes_and_multiples_of_11 : ℚ) / total_number_of_cards = 1 / 60 :=
by
  sorry

end probability_prime_and_multiple_of_11_l549_549873


namespace unique_transformed_digits_l549_549189

-- Definitions based on conditions in a)
def matchsticks_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 3 => 5
  | 4 => 4
  | 5 => 5
  | 6 => 6
  | 7 => 3
  | 8 => 7
  | 9 => 6
  | _ => 0

-- Target theorem to prove the problem's question is equivalent to 6, given conditions
theorem unique_transformed_digits : ∃ n : ℕ, n = 6 :=
by
  use 6
  sorry

end unique_transformed_digits_l549_549189


namespace product_gcd_lcm_l549_549307

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l549_549307


namespace volume_ratio_of_octahedron_to_tetrahedron_l549_549039

theorem volume_ratio_of_octahedron_to_tetrahedron : 
  ∀ (tetra : ℝ → ℝ → ℝ → ℝ) 
  (V₁ V₂ V₃ V₄ : ℝ × ℝ × ℝ), 
  (tetra V₁.1 V₁.2 V₁.3 + tetra V₂.1 V₂.2 V₂.3 + 
   tetra V₃.1 V₃.2 V₃.3 + tetra V₄.1 V₄.2 V₄.3) = 1 → 
  (volume ((V₁.1 + V₂.1) / 2, (V₁.2 + V₂.2) / 2, (V₁.3 + V₂.3) / 2) + 
   volume ((V₁.1 + V₃.1) / 2, (V₁.2 + V₃.2) / 2, (V₁.3 + V₃.3) / 2) + 
   volume ((V₁.1 + V₄.1) / 2, (V₁.2 + V₄.2) / 2, (V₁.3 + V₄.3) / 2)) = 1 / 2 :=
by
  sorry

-- Additional definitions may be required for the volume function and tetra function if necessary.

end volume_ratio_of_octahedron_to_tetrahedron_l549_549039


namespace election_votes_l549_549041

theorem election_votes (total_votes : ℕ) (invalid_percentage : ℝ) (valid_vote_threshold : ℝ)
  (h_total_votes : total_votes = 10000)
  (h_invalid_percentage : invalid_percentage = 0.30)
  (h_valid_vote_threshold : valid_vote_threshold = 0.50)
  : ∃ (valid_votes needed_to_win : ℕ), valid_votes = 7000 ∧ needed_to_win = 3501 :=
by
  let invalid_votes := (invalid_percentage * total_votes).to_nat
  have h_invalid_votes : invalid_votes = 3000 :=
    by
      rw [h_invalid_percentage, h_total_votes]
      norm_num
  let valid_votes := total_votes - invalid_votes
  have h_valid_votes : valid_votes = 7000 :=
    by
      rw [h_invalid_votes, h_total_votes]
      norm_num
  let needed_to_win := (valid_vote_threshold * valid_votes).to_nat + 1
  have h_needed_to_win : needed_to_win = 3501 :=
    by
      rw [h_valid_vote_threshold, h_valid_votes]
      norm_num
  exact ⟨valid_votes, needed_to_win, h_valid_votes, h_needed_to_win⟩

end election_votes_l549_549041


namespace numbers_neither_5_nor_6_nice_below_500_l549_549682

def k_nice (k : ℕ) (N : ℕ) : Prop :=
  N % k = 1

theorem numbers_neither_5_nor_6_nice_below_500 :
  { n : ℕ | n < 500 ∧ ¬k_nice 5 n ∧ ¬k_nice 6 n }.finite.count = 333 :=
by
  sorry

end numbers_neither_5_nor_6_nice_below_500_l549_549682


namespace mean_of_five_numbers_is_correct_l549_549165

-- Define the given sum of five numbers as three-quarters
def sum_of_five_numbers : ℚ := 3 / 4

-- Define the number of numbers, which is 5
def number_of_numbers : ℕ := 5

-- Define the mean calculation from the given sum and number of numbers
def mean_five_numbers (sum : ℚ) (count : ℕ) : ℚ := sum / count

-- Statement to prove: the mean of five numbers given their sum is 3/4 equals 3/20
theorem mean_of_five_numbers_is_correct :
  mean_five_numbers sum_of_five_numbers number_of_numbers = 3 / 20 :=
by
  -- Skipping the proof
  sorry

end mean_of_five_numbers_is_correct_l549_549165


namespace tax_diminished_by_40_l549_549922

-- Define the conditions
variables {T C : ℝ} -- Representing original Tax and Consumption
variable {X : ℝ} -- Representing diminished percentage tax

-- Hypotheses from the conditions
hypothesis consumption_increase : ∀ C, new_consumption = C * 1.25
hypothesis revenue_decrease : ∀ T C X, 
  (T * (1 - X / 100) * C * 1.25) = 0.75 * T * C

-- Prove that the tax was diminished by 40%
theorem tax_diminished_by_40 (h1: consumption_increase C)
(h2 : revenue_decrease T C X) :
  X = 40 :=
sorry

end tax_diminished_by_40_l549_549922


namespace find_m2n_plus_mn2_minus_mn_l549_549394

def quadratic_roots (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0

theorem find_m2n_plus_mn2_minus_mn :
  ∃ m n : ℝ, quadratic_roots 1 2015 (-1) m n ∧ m^2 * n + m * n^2 - m * n = 2016 :=
by
  sorry

end find_m2n_plus_mn2_minus_mn_l549_549394


namespace length_of_EF_l549_549971

variables {R : Type*} [real_linear_ordered_ring R]

-- Given conditions
def AB : R := 4
def BC : R := 5
def DE : R := 3

-- Define length of EF
def EF := 2 * ((AB + BC / 2) - DE)

-- Prove EF length
theorem length_of_EF : EF = 7 :=
by
  unfold EF
  norm_num
  sorry

end length_of_EF_l549_549971


namespace sequence_sum_l549_549663

theorem sequence_sum (r : ℝ) (x y : ℝ)
  (a : ℕ → ℝ)
  (h1 : a 1 = 4096)
  (h2 : a 2 = 1024)
  (h3 : a 3 = 256)
  (h4 : a 6 = 4)
  (h5 : a 7 = 1)
  (h6 : a 8 = 0.25)
  (h_sequence : ∀ n, a (n + 1) = r * a n)
  (h_r : r = 1 / 4) :
  x + y = 80 :=
sorry

end sequence_sum_l549_549663


namespace jose_bottle_caps_l549_549074

def jose_start : ℕ := 7
def rebecca_gives : ℕ := 2
def final_bottle_caps : ℕ := 9

theorem jose_bottle_caps :
  jose_start + rebecca_gives = final_bottle_caps :=
by
  sorry

end jose_bottle_caps_l549_549074


namespace expression_evaluation_l549_549587

def eval_expression : Int := 
  let a := -2 ^ 3
  let b := abs (2 - 3)
  let c := -2 * (-1) ^ 2023
  a + b + c

theorem expression_evaluation :
  eval_expression = -5 :=
by
  sorry

end expression_evaluation_l549_549587


namespace second_discount_percentage_l549_549428

-- Definitions for the given conditions
def original_price : ℝ := 33.78
def first_discount_rate : ℝ := 0.25
def final_price : ℝ := 19.0

-- Intermediate calculations based on the conditions
def first_discount : ℝ := first_discount_rate * original_price
def price_after_first_discount : ℝ := original_price - first_discount
def second_discount_amount : ℝ := price_after_first_discount - final_price

-- Lean theorem statement
theorem second_discount_percentage : (second_discount_amount / price_after_first_discount) * 100 = 25 := by
  sorry

end second_discount_percentage_l549_549428


namespace no_arith_progression_with_r_reverse_l549_549567

-- Define the function r(n) which reverses the binary representation of n
def r (n : ℕ) : option ℕ :=
  let s := n.toDigits 2
  let rs := s.reverse
  let r_n := Nat.ofDigits 2 rs
  if r_n % 2 = 1 then some r_n else none

-- Definition of strictly increasing arithmetic progression
def is_strictly_increasing_arith_progression (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → a i < a j ∧ (a j = a i + d * (j - i))

-- Definition for odd positive integers
def all_odd_positive (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → (a i > 0 ∧ a i % 2 = 1)

-- The proof problem statement
theorem no_arith_progression_with_r_reverse :
  ¬ ∃ (a : ℕ → ℕ) (d : ℕ), (is_strictly_increasing_arith_progression a d) ∧ 
  (all_odd_positive a 8) ∧ 
  (is_strictly_increasing_arith_progression (λ i, Option.get (r (a i))) d) :=
by
  sorry

end no_arith_progression_with_r_reverse_l549_549567


namespace cos_of_angle_through_point_l549_549785

theorem cos_of_angle_through_point (α : ℝ) 
  (h1 : ∃ P : ℝ × ℝ, P = (-1, 3) ∧ ∃ θ, tan θ = P.snd / P.fst ∧ θ = α) :
  cos α = - (Real.sqrt 10) / 10 := by
  sorry

end cos_of_angle_through_point_l549_549785


namespace ordered_arrays_count_l549_549625

theorem ordered_arrays_count (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|) →
  (|a + b + c| = 1 ∧ |a| + |b| + |c| = 1 ∧ |a - b| + |b - c| + |c - a| = 2) →
  ∃ as : finset (ℝ × ℝ × ℝ), as.card = 6 ∧ ∀ p ∈ as, p = (0, 0, 1) ∨ p = (0, 0, -1) ∨ p = (0, 1, 0) ∨ p = (0, -1, 0) ∨ p = (1, 0, 0) ∨ p = (-1, 0, 0) :=
by
  sorry

end ordered_arrays_count_l549_549625


namespace parallel_vectors_l549_549753

open Real

theorem parallel_vectors (k : ℝ) 
  (a : ℝ × ℝ := (k-1, 1)) 
  (b : ℝ × ℝ := (k+3, k)) 
  (h : a.1 * b.2 = a.2 * b.1) : 
  k = 3 ∨ k = -1 :=
by
  sorry

end parallel_vectors_l549_549753


namespace trig_sin_cos_l549_549054

variable (c a : ℝ) (angle_A angle_C: ℝ)

-- Given conditions
axiom h1 : c = 2
axiom h2 : a = √3
axiom h3 : angle_A = π / 6

-- To prove
theorem trig_sin_cos (h1 : c = 2) (h2 : a = √3) (h3 : angle_A = π / 6) : 
  ∃ (angle_C : ℝ), sin angle_C = √3 / 3 ∧ cos (2 * angle_C) = 1 / 3 :=
by
  use angle_C
  sorry

end trig_sin_cos_l549_549054


namespace sum_free_subgroup_exists_l549_549686

theorem sum_free_subgroup_exists (N : ℕ) : 
  ∃ (p : ℕ) (α : ℕ) (S : set ℕ), 
    nat.prime p ∧ 
    α ≠ 0 ∧
    (∀ x ∈ S, x < p) ∧ 
    (∀ x ∈ S, ∃ k, x = α^k % p) ∧ 
    (∀ a b c ∈ S, a + b ≠ c % p) ∧ 
    (|S| ≥ N) := 
sorry

end sum_free_subgroup_exists_l549_549686


namespace circumscribed_quad_l549_549815

open EuclideanGeometry -- Assuming Euclidean Geometry is needed for the concepts like midpoints and excenters

theorem circumscribed_quad (ABC : Triangle)
  (M : Point) (hM : is_midpoint M ABC.BC)
  (X Y : Point) (hX : is_excenter X ABC.B)
  (hY : is_excenter Y ABC.C)
  (R P Q S : Point)
  (hR : on_line_segment M X R)
  (hP : on_line_segment M X P)
  (hQ : on_line_segment M Y Q)
  (hS : on_line_segment M Y S)
  (hRX : between R ABC.AB X)
  (hPX : between P ABC.AC X)
  (hQY : between Q ABC.AB Y)
  (hSY : between S ABC.AC Y) :
  is_circumscribed_quadrilateral R P Q S :=
sorry -- Proof is omitted

end circumscribed_quad_l549_549815


namespace A_necessary_for_B_not_sufficient_l549_549720
-- Define Proposition A
def A : Set ℝ := {x | (x^2 + x) / (x - 1) ≥ 0}

-- Define Proposition B
def B : Set ℝ := {x | log 3 (2 * x + 1) ≤ 0}

-- Prove that A is a necessary condition for B but not a sufficient condition
theorem A_necessary_for_B_not_sufficient : (∀ x, x ∈ B → x ∈ A) ∧ ∃ x, x ∈ A ∧ x ∉ B :=
sorry

end A_necessary_for_B_not_sufficient_l549_549720


namespace original_price_of_coat_l549_549907

theorem original_price_of_coat (P : ℝ) (h : 0.70 * P = 350) : P = 500 :=
sorry

end original_price_of_coat_l549_549907


namespace gcd_lcm_product_24_60_l549_549309

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l549_549309


namespace sum_of_solutions_l549_549444

def g (x : ℝ) : ℝ := 3 * x - 2

def g_inv (y : ℝ) : ℝ := (y + 2) / 3

theorem sum_of_solutions : 
  ∑ x in {x : ℝ | g_inv x = g (x ^ 2)}.toFinset = (2 / 3 : ℝ) := 
sorry

end sum_of_solutions_l549_549444


namespace sum_of_digits_final_row_l549_549243

theorem sum_of_digits_final_row (N : ℕ) :
  (∃ N : ℕ, N * (N + 1) = 10100 ∧ N = 100) → (digits_sum N = 1) :=
by
  sorry

def digits_sum (n : ℕ) : ℕ :=
  (digits 10 n).sum

end sum_of_digits_final_row_l549_549243


namespace math_problem_l549_549134

theorem math_problem (a : ℤ) (hx1 : 7 - 2 * a > -1) (hx2 : a - 1 ≥ 0) (h1 : a - 3 ≠ 0) (h2 : a - 1 ≠ 0) : 
  (\n : ℝ) = (\frac{a ^ 2 - 3}{a - 3} - a) / \frac{a - 1}{a ^ 2 - 6 * a + 9} = 3 * a - 9 :=
by
  sorry

end math_problem_l549_549134


namespace sets_intersection_l549_549336

theorem sets_intersection (x y : ℝ) (h : ({2, real.log x / real.log 3} ∩ {x, y}) = {0}) : x + y = 1 := sorry

end sets_intersection_l549_549336


namespace product_gcd_lcm_24_60_l549_549321

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l549_549321


namespace find_f_2017_l549_549835

noncomputable def f (x p q : ℝ) : ℝ := x^2 + p*x + q

theorem find_f_2017 (p q : ℝ) (h : ∀ x, 3 ≤ x ∧ x ≤ 5 → abs (f x p q) ≤ 1 / 2) :
  f^[2017] (f (7 + sqrt 15) (7 - sqrt 15) / 2) p q ≈ 1.56 :=
sorry

end find_f_2017_l549_549835


namespace number_of_linear_functions_is_3_l549_549896

def y1 (x : ℝ) : ℝ := x / 6
def y2 (x : ℝ) : ℝ := -4 / x
def y3 (x : ℝ) : ℝ := 3 - (1 / 2) * x
def y4 (x : ℝ) : ℝ := 3 * x ^ 2 - 2
def y5 (x : ℝ) : ℝ := x ^ 2 - (x - 3) * (x + 2)
def y6 (x : ℝ) : ℝ := 6 ^ x

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

theorem number_of_linear_functions_is_3 :
  (is_linear y1 ∧ is_linear y3 ∧ is_linear y5) ∧
  ¬ is_linear y2 ∧ ¬ is_linear y4 ∧ ¬ is_linear y6 :=
by
  split
  { repeat { split }
    · -- Prove y1 is linear
      -- We need to find m and b such that y1 x = m * x + b for all x
      use [1 / 6, 0]
      intro x
      simp [y1]

    · -- Prove y3 is linear
      -- We need to find m and b such that y3 x = m * x + b for all x
      use [-1 / 2, 3]
      intro x
      simp [y3]

    · -- Prove y5 is linear
      -- Simplifying y5: y = x^2 - (x^2 - x - 6) = x + 6
      use [1, 6]
      intro x
      simp [y5]
      ring
  }
  { repeat { split }
    · -- Prove y2 is not linear
      intro h
      rcases h with ⟨m, b, h⟩
      have hx := h 1
      simp at hx
      contradiction

    · -- Prove y4 is not linear
      intro h
      rcases h with ⟨m, b, h⟩
      have hx := h 1
      rw [pow_two] at hx
      contradiction

    · -- Prove y6 is not linear
      intro h
      rcases h with ⟨m, b, h⟩
      have hx := h 1
      have hy := h 0
      simp at hx hy
      contradiction
  }

end number_of_linear_functions_is_3_l549_549896


namespace certain_event_among_options_l549_549555

-- Definition of the proof problem
theorem certain_event_among_options (is_random_A : Prop) (is_random_C : Prop) (is_random_D : Prop) (is_certain_B : Prop) :
  (is_random_A → (¬is_certain_B)) ∧
  (is_random_C → (¬is_certain_B)) ∧
  (is_random_D → (¬is_certain_B)) ∧
  (is_certain_B ∧ ((¬is_random_A) ∧ (¬is_random_C) ∧ (¬is_random_D))) :=
by
  sorry

end certain_event_among_options_l549_549555


namespace scientific_notation_l549_549787

theorem scientific_notation {n : ℝ} (h₁ : 1 ≤ n) (h₁ : n < 10) : 
  51_800_000_000 = 5.18 * 10 ^ 10 := 
sorry

end scientific_notation_l549_549787


namespace probability_of_odd_sum_given_product_even_l549_549492

noncomputable def probability_odd_sum_given_even_product : ℚ :=
  let num_ways_all_odd := 4^6
  let total_ways := 8^6
  let num_ways_at_least_one_even := total_ways - num_ways_all_odd
  let num_ways_odd_sum :=
    6 * 4^6 + 
    (Nat.choose 6 3) * 4^6 + 
    6 * 4^6
  let favorable_ways := num_ways_odd_sum
  let probability := (favorable_ways : ℚ) / num_ways_at_least_one_even
  probability

theorem probability_of_odd_sum_given_product_even :
  probability_odd_sum_given_even_product = 32 / 63 := sorry

end probability_of_odd_sum_given_product_even_l549_549492


namespace blooming_days_from_nov_to_dec_2013_l549_549529

def is_odd_month (m : ℕ) : Prop := 
  m % 2 = 1
  
def is_even_month (m : ℕ) : Prop := 
  m % 2 = 0

def blooms_on (d m : ℕ) : bool :=
  if is_odd_month m then 
    let weekday := (Date.new! 2013 m d).dayOfWeek
    weekday = 3 || weekday = 5 -- Wednesdays and Fridays
  else if is_even_month m then 
    let weekday := (Date.new! 2013 m d).dayOfWeek
    weekday = 2 || weekday = 4 -- Tuesdays and Thursdays
  else 
    false

def blooming_days_in_month (m : ℕ) : ℕ :=
  (List.range (Date.daysInMonth 2013 m)).count (λ d, blooms_on (d+1) m)

def total_blooming_days_in_period : ℕ :=
  blooming_days_in_month 11 + blooming_days_in_month 12

theorem blooming_days_from_nov_to_dec_2013 : total_blooming_days_in_period = 18 := 
  by 
    -- Proof will show that total_blooming_days_in_period = 9 (Nov) + 9 (Dec)
    sorry

end blooming_days_from_nov_to_dec_2013_l549_549529


namespace find_distance_between_intersection_points_l549_549717

-- Definitions of ellipse and parabola
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 36 = 1
def parabola (x y : ℝ) : Prop := x = (y^2 / (8 * real.sqrt 5)) - 2 * real.sqrt 5

-- Statement to prove the distance between the intersection points of ellipse and parabola
theorem find_distance_between_intersection_points : 
  ∃ (points : list (ℝ × ℝ)), 
    (∀ p, p ∈ points → ellipse p.1 p.2 ∧ parabola p.1 p.2) ∧ 
    ( ∃ y1 y2, (ellipse 0 y1 ∧ ellipse 0 y2 ∧ parabola 0 y1 ∧ parabola 0 y2 ∧ 
     points = [(0, y1), (0, y2)] ) → 
    distance (0, y1) (0, y2) = some_distance) := 
sorry

end find_distance_between_intersection_points_l549_549717


namespace ricky_time_difference_l549_549865

noncomputable def old_man_time_per_mile : ℚ := 300 / 8
noncomputable def young_man_time_per_mile : ℚ := 160 / 12
noncomputable def time_difference : ℚ := old_man_time_per_mile - young_man_time_per_mile

theorem ricky_time_difference :
  time_difference = 24 := by
sorry

end ricky_time_difference_l549_549865


namespace composite_fraction_l549_549869

theorem composite_fraction (x : ℤ) (hx : x = 5^25) : 
  ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ a * b = x^4 + x^3 + x^2 + x + 1 :=
by sorry

end composite_fraction_l549_549869


namespace points_on_ellipse_satisfying_dot_product_l549_549080

theorem points_on_ellipse_satisfying_dot_product :
  ∃ P1 P2 : ℝ × ℝ,
    P1 = (0, 3) ∧ P2 = (0, -3) ∧
    ∀ P : ℝ × ℝ, 
    (P ∈ ({p : ℝ × ℝ | (p.1 / 5)^2 + (p.2 / 3)^2 = 1}) → 
     ((P.1 - (-4)) * (P.1 - 4) + P.2^2 = -7) →
     (P = P1 ∨ P = P2))
:=
sorry

end points_on_ellipse_satisfying_dot_product_l549_549080


namespace even_factors_count_l549_549395

noncomputable def n : ℕ := 2^3 * 3^2 * 5^2 * 7

def is_even_factor (m : ℕ) : Prop :=
  m ∣ n ∧ m % 2 = 0

def count_even_factors (n : ℕ) : ℕ :=
  (∑ m in (Finset.range (n + 1)), if is_even_factor m then 1 else 0)

theorem even_factors_count : count_even_factors n = 54 := by
  sorry

end even_factors_count_l549_549395


namespace angles_and_ratio_of_isosceles_triangle_l549_549909

theorem angles_and_ratio_of_isosceles_triangle
  (α : ℝ) (β : ℝ) (m : ℝ) :
  (0 < m ∧ m ≤ 1/2) →
  α = arccos ((1 + real.sqrt (1 - 2*m)) / 2) ∨ α = arccos ((1 - real.sqrt (1 - 2*m)) / 2) →
  β = 2 * arcsin ((1 + real.sqrt (1 - 2*m)) / 2) ∨ β = 2 * arcsin ((1 - real.sqrt (1 - 2*m)) / 2) :=
sorry

end angles_and_ratio_of_isosceles_triangle_l549_549909


namespace coloring_exists_iff_odd_l549_549408

-- Define the main theorem stating the condition and conclusion
theorem coloring_exists_iff_odd (n : ℕ) : 
  (∃ (colorings : (fin n → fin n) • (fin (n * (n - 1) / 2 + n) → fin n)),
    (∀ C₁ C₂ C₃ : fin n, C₁ ≠ C₂ → C₂ ≠ C₃ → C₁ ≠ C₃ → 
      ∃ (triangle : (fin n × fin n × fin n)),
      (triangle.1 ∈ colorings ∧ triangle.2 ∈ colorings ∧ triangle.3 ∈ colorings) ∧
        (colorings triangle.1 = C₁ ∧ colorings triangle.2 = C₂ ∧ colorings triangle.3 = C₃))) ↔ 
  nat.odd n :=
sorry

end coloring_exists_iff_odd_l549_549408


namespace water_pogo_oscillation_period_l549_549297

noncomputable def period_of_oscillation (m L w h ρ g : ℝ) : ℝ :=
2 * π * (Real.sqrt (m / (ρ * w * L * g)))

theorem water_pogo_oscillation_period (m L w h ρ g : ℝ) 
    (h_m_pos : 0 < m) (h_L_pos : 0 < L) (h_w_pos : 0 < w)
    (h_h_pos : 0 < h) (h_rho_pos : 0 < ρ) (h_g_pos : 0 < g) : 
    period_of_oscillation m L w h ρ g = 2 * π * (Real.sqrt (m / (ρ * w * L * g))) :=
sorry

end water_pogo_oscillation_period_l549_549297


namespace calculate_flight_duration_l549_549642

noncomputable def duration_of_flight (departure_NY_ET arrival_CT_ET : ℕ) (departure_LON_ET : ℕ) (flight_LON_NY : ℕ) :=
  let arrival_NY_ET := (departure_LON_ET + flight_LON_NY) % 24
  in (arrival_CT_ET + 24 - arrival_NY_ET) % 24

theorem calculate_flight_duration :
  let departure_LON_ET := 6  -- 6:00 a.m. ET on Monday
  let flight_LON_NY := 18   -- 18 hours of flight time
  let arrival_CT_ET := 10  -- 10:00 a.m. ET on Tuesday
  duration_of_flight 0 arrival_CT_ET departure_LON_ET flight_LON_NY = 10 := by
  sorry

end calculate_flight_duration_l549_549642


namespace distance_between_intersections_l549_549512

open Set

def parabola (x y : ℝ) := y^2 = 12 * x
def circle (x y : ℝ) := x^2 + y^2 - 4 * x - 6 * y = 0

theorem distance_between_intersections :
  let points := {p : ℝ × ℝ | parabola p.1 p.2 ∧ circle p.1 p.2} in
  ∃ (C D : ℝ × ℝ), C ∈ points ∧ D ∈ points ∧ C ≠ D ∧ dist C D = 3 * Real.sqrt 5
  :=
  sorry

end distance_between_intersections_l549_549512


namespace decagon_diagonal_intersections_l549_549632

theorem decagon_diagonal_intersections : 
  let n := 10
  let num_diagonals := n * (n - 3) / 2
  num_diagonals = 35 → 
  ∃ (intersections : ℕ), intersections = Nat.choose 10 4 ∧ intersections = 210 := 
by 
  intros
  have h1: num_diagonals = 35 := by assumption
  use Nat.choose 10 4
  sorry

end decagon_diagonal_intersections_l549_549632


namespace boxes_total_is_correct_l549_549177

def initial_boxes : ℕ := 7
def additional_boxes_per_box : ℕ := 7
def final_non_empty_boxes : ℕ := 10
def total_boxes := 77

theorem boxes_total_is_correct
  (h1 : initial_boxes = 7)
  (h2 : additional_boxes_per_box = 7)
  (h3 : final_non_empty_boxes = 10)
  : total_boxes = 77 :=
by
  -- Proof goes here
  sorry

end boxes_total_is_correct_l549_549177


namespace platform_length_is_correct_l549_549959

noncomputable def length_of_platform (T : ℕ) (t_p t_s : ℕ) : ℕ :=
  let speed_of_train := T / t_s
  let distance_when_crossing_platform := speed_of_train * t_p
  distance_when_crossing_platform - T

theorem platform_length_is_correct :
  ∀ (T t_p t_s : ℕ),
  T = 300 → t_p = 33 → t_s = 18 →
  length_of_platform T t_p t_s = 250 :=
by
  intros T t_p t_s hT ht_p ht_s
  simp [length_of_platform, hT, ht_p, ht_s]
  sorry

end platform_length_is_correct_l549_549959


namespace main_theorem_l549_549450

def f (x : ℝ) (h : x ≠ 0) : ℝ := (x^2 + 1) / (2 * x)

def f_iter (n : ℕ) (x : ℝ) (h : x ≠ 0) : ℝ :=
  if n = 0 then x else f (f_iter (n - 1) x h) h

theorem main_theorem (n : ℕ) (x : ℝ) (h : x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1) : 
  f_iter n x h.2.1 = 1 + 1 / (f ( ( (x + 1) / (x - 1) )^(2^n) ) sorry)
:=
sorry

end main_theorem_l549_549450


namespace solve_for_m_l549_549331

def star (a b : ℝ) : ℝ :=
if a ≥ b then a^2 * b + a
else a * b^2 + b

theorem solve_for_m (m : ℝ) : star 2 m = 36 → m = 4 :=
by
  intro h
  sorry

end solve_for_m_l549_549331


namespace problem_statement_l549_549733

-- Define the given information as assumptions.
def k : ℝ := 1 / 2
def A : ℝ × ℝ := (7, 1)
def B : ℝ × ℝ := (4, -2)
def l2 : ℝ → ℝ := λ x, -(x + 3) / 2
def N : ℝ × ℝ := (4, 2)

-- Define the coordinates of point M found by solving the system of equations.
def M : ℝ × ℝ := (1, -2)

-- Define the standard equation of circle C.
def C_eqn (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 9

-- Define the lines that are tangent to the circle C and pass through point N.
def tangent_line1 (x y : ℝ) : Prop := x = 4
def tangent_line2 (x y : ℝ) : Prop := 7 * x - 24 * y + 20 = 0

-- Lean 4 statement for the circle equation and tangent lines.
theorem problem_statement :
  ∃ (x y : ℝ), C_eqn x y ∧
  (tangent_line1 x y ∨ tangent_line2 x y) ∧
  (x, y) = N := by
  sorry

end problem_statement_l549_549733


namespace integral_evaluation_l549_549262

noncomputable def definite_integral : ℝ :=
  ∫ x in -2..0, (x^2 - 4) * Real.cos (3 * x)

theorem integral_evaluation :
  definite_integral = (12 * Real.cos 6 - 2 * Real.sin 6) / 27 :=
by
  sorry

end integral_evaluation_l549_549262


namespace maria_compensation_l549_549106

theorem maria_compensation (d : ℝ) (h1 : d ≥ 1600000) : ∃ c, c = 1400000 :=
by
  have h2 : c = 1400000, from sorry
  exact ⟨c, h2⟩

end maria_compensation_l549_549106


namespace constant_term_is_neg5_l549_549048

noncomputable def constant_term_in_expansion : ℤ :=
  let expr := (x - (1/x) -  1)^4 in
  find_constant_term expr

theorem constant_term_is_neg5 (x : ℝ) (h₁ : is_rational x) (h₂ : x ≠ 0):
  constant_term_in_expansion = -5 :=
by
  sorry

end constant_term_is_neg5_l549_549048


namespace smallest_positive_period_of_f_maximum_value_of_f_l549_549384

def vector_a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2

theorem smallest_positive_period_of_f (T : ℝ) (x : ℝ) :
  (∀ x, f (x + T) = f x) ∧ (T > 0) → T = π := sorry

theorem maximum_value_of_f (x : ℝ) :
  ∃ k : ℤ, f x = 3 ∧ x = π / 6 + k * π := sorry

end smallest_positive_period_of_f_maximum_value_of_f_l549_549384


namespace train_passing_time_l549_549390

def length_of_train : ℝ := 225 -- length in meters
def speed_of_train_km_per_hr : ℝ := 90 -- speed in kilometers per hour
def speed_conversion_factor : ℝ := (1000 / 3600) -- conversion factor from km/hr to m/s
def speed_of_train_m_per_s : ℝ := speed_of_train_km_per_hr * speed_conversion_factor -- speed in meters per second

theorem train_passing_time : 
  (length_of_train / speed_of_train_m_per_s) = 9 :=
by
  -- The proof would go here
  sorry

end train_passing_time_l549_549390


namespace shirt_price_is_4_l549_549485

noncomputable theory

def pants_price := 5
def shorts_price := 3
def num_pants := 3
def num_shorts := 5
def num_shirts := 5
def final_money := 30
def shirt_cost := 10
def bought_shirts := 2

-- Prove that the price of each shirt 'S' is 4 dollars
theorem shirt_price_is_4 
  (S : ℝ) 
  (pants_price shorts_price num_pants num_shorts num_shirts final_money shirt_cost bought_shirts : ℝ)
  (hpants_price : pants_price = 5)
  (hshorts_price : shorts_price = 3)
  (hnum_pants : num_pants = 3)
  (hnum_shorts : num_shorts = 5)
  (hnum_shirts : num_shirts = 5)
  (hfinal_money : final_money = 30)
  (hshirt_cost : shirt_cost = 10)
  (hbought_shirts : bought_shirts = 2) :
  (num_pants * pants_price + num_shorts * shorts_price + num_shirts * S - bought_shirts * shirt_cost = final_money) → 
  S = 4 :=
by 
  -- conditions and calculations of the given problem lead us to the result where S = 4 
  sorry

end shirt_price_is_4_l549_549485


namespace football_team_total_players_l549_549928

/-- The conditions are:
1. There are some players on a football team.
2. 46 are throwers.
3. All throwers are right-handed.
4. One third of the rest of the team are left-handed.
5. There are 62 right-handed players in total.
And we need to prove that the total number of players on the football team is 70. 
--/

theorem football_team_total_players (P : ℕ) 
  (h_throwers : P >= 46) 
  (h_total_right_handed : 62 = 46 + 2 * (P - 46) / 3)
  (h_remainder_left_handed : 1 * (P - 46) / 3 = (P - 46) / 3) :
  P = 70 :=
by
  sorry

end football_team_total_players_l549_549928


namespace fill_in_blank_count_valid_digits_l549_549671

theorem fill_in_blank {d : ℕ} (h : d < 3) : 
  ∃ l : ℕ, (15 * 100 + d * 10 + 3) / 3 = l ∧ (l % 100) / 10 = 0 :=
sorry

theorem count_valid_digits : 
  {d : ℕ // d < 3 ∧ ∃ l : ℕ, (15 * 100 + d * 10 + 3) / 3 = l ∧ (l % 100) / 10 = 0}.card = 3 :=
sorry

end fill_in_blank_count_valid_digits_l549_549671


namespace inequality_bounds_l549_549346

variables {α : Type*} [linear_ordered_field α]

theorem inequality_bounds
  (a : ℕ → α)
  (h : ∀ i j, i < j → a i ≤ a j)
  (x : α)
  (y : α)
  (h1 : x = (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7) / 8)
  (h2 : y = (a 0 ^ 2 + a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 + a 6 ^ 2 + a 7 ^ 2) / 8) :
  2 * real.sqrt (y - x ^ 2) ≤ a 7 - a 0 ∧ a 7 - a 0 ≤ 4 * real.sqrt (y - x ^ 2) :=
sorry

end inequality_bounds_l549_549346


namespace braiding_time_l549_549068

variables (n_dancers : ℕ) (b_braids_per_dancer : ℕ) (t_seconds_per_braid : ℕ)

theorem braiding_time : n_dancers = 8 → b_braids_per_dancer = 5 → t_seconds_per_braid = 30 → 
  (n_dancers * b_braids_per_dancer * t_seconds_per_braid) / 60 = 20 :=
by
  intros
  sorry

end braiding_time_l549_549068


namespace misha_students_l549_549471

theorem misha_students : 
  ∀ (n : ℕ),
  (n = 74 + 1 + 74) ↔ (n = 149) :=
by
  intro n
  split
  · intro h
    rw [← h, nat.add_assoc]
    apply nat.add_right_cancel
    rw [nat.add_comm 1 74, nat.add_assoc]
    apply nat.add_right_cancel
    rw nat.add_comm
  · intro h
    exact h
  sorry

end misha_students_l549_549471


namespace hyperbola_eccentricity_l549_549727

theorem hyperbola_eccentricity (b c : ℝ) (hb : b = 2) (hc : c = 6) : 
  let a := real.sqrt (c^2 - b^2) in
  a ≠ 0 → (c / a = 3 * real.sqrt 2 / 4) := 
by
  intros; sorry

end hyperbola_eccentricity_l549_549727


namespace carol_mike_equal_savings_weeks_l549_549641

theorem carol_mike_equal_savings_weeks :
  ∃ x : ℕ, (60 + 9 * x = 90 + 3 * x) ↔ x = 5 := 
by
  sorry

end carol_mike_equal_savings_weeks_l549_549641


namespace chuck_total_playable_area_l549_549269

noncomputable def chuck_roaming_area (shed_length shed_width leash_length : ℝ) : ℝ :=
  let larger_arc_area := (3 / 4) * Real.pi * leash_length ^ 2
  let additional_sector_area := (1 / 4) * Real.pi * (leash_length - shed_length) ^ 2
  larger_arc_area + additional_sector_area

theorem chuck_total_playable_area :
  chuck_roaming_area 3 4 5 = 19 * Real.pi :=
  by
  sorry

end chuck_total_playable_area_l549_549269


namespace largest_prime_factor_of_sequence_sum_l549_549980

theorem largest_prime_factor_of_sequence_sum
  (seq : List ℕ)
  (h1 : ∀ n ∈ seq, 100 ≤ n ∧ n < 1000)
  (h2 : ∀ (i : ℕ), i < seq.length - 1 → let n := seq.nthLe i sorry in let m := seq.nthLe (i + 1) sorry in (n / 100 = (m / 10) % 10) ∧ ((n / 10) % 10 = m % 10) ∧ (n % 10 = m / 100))
  (h3 : let n := seq.last sorry in let m := seq.head sorry in (n / 100 = (m / 10) % 10) ∧ ((n / 10) % 10 = m % 10) ∧ (n % 10 = m / 100))
  : 37 ∣ seq.sum :=
by
  sorry

end largest_prime_factor_of_sequence_sum_l549_549980


namespace product_of_gcd_and_lcm_1440_l549_549318

def product_of_gcd_and_lcm_of_24_and_60 : ℕ :=
  Nat.gcd 24 60 * Nat.lcm 24 60 

/-- The product of the gcd and lcm of 24 and 60 is 1440. -/
theorem product_of_gcd_and_lcm_1440 : product_of_gcd_and_lcm_of_24_and_60 = 1440 := 
by
  sorry

end product_of_gcd_and_lcm_1440_l549_549318


namespace complex_solution_l549_549707

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem complex_solution (z : ℂ) (hz : abs z = 1) (h: is_purely_imaginary ((3+4*I) * z)) : 
  (z = (4/5) + (3/5)*I ∧ conj z = (4/5) - (3/5)*I) ∨ (z = -(4/5) - (3/5)*I ∧ conj z = -(4/5) + (3/5)*I) :=
sorry

end complex_solution_l549_549707


namespace remainder_quotient_l549_549825

-- Define the polynomial Q(x)
variable (Q : Polynomial ℝ)

-- Conditions given in the problem
variable (h1 : Q.eval 19 = 10) (h2 : Q.eval 15 = 8)

-- The target is to prove that the remainder when Q(x) is divided by (x-15)(x-19) is (1/2)x + 0.5
theorem remainder_quotient (Q : Polynomial ℝ) (h1 : Q.eval 19 = 10) (h2 : Q.eval 15 = 8) :
  ∃ (a b : ℝ), a = 1/2 ∧ b = 0.5 ∧ ∀ (x : ℝ), Q(x) = (x-15)*(x-19)*R(x) + a*x + b :=
begin
  -- Proof not required
  sorry,
end

end remainder_quotient_l549_549825


namespace no_such_polyhedron_exists_l549_549424

-- Conditions defined as properties.
def polyhedron (V E F : ℕ) (face_relationships : fin 4 → fin 4 → Prop) : Prop :=
  V = 8 ∧ E = 12 ∧ F = 6 ∧ (∀ i j, i ≠ j → face_relationships i j)

-- The main theorem stating the impossibility.
theorem no_such_polyhedron_exists : ¬ ∃ face_relationships, polyhedron 8 12 6 face_relationships :=
by {
  -- proof goes here
  sorry
}

end no_such_polyhedron_exists_l549_549424


namespace num_paths_l549_549633

def adjacent (x y : Point) : Prop :=
    -- Definition of adjacency relationship
  
def reachable_in_steps (x y : Point) (n : ℕ) : Prop :=
    -- Definition indicating y is reachable from x in exactly n steps

def valid_path_sequence (seq : list Point) : Prop :=
    -- Definition of a valid path sequence according to problem conditions

theorem num_paths (A : Point) (n : ℕ) :
    (n = 5) ∧
    (∀ seq, valid_path_sequence A seq → length seq = n) →
    ∃ count, count = 180 :=
by
  -- Hypothetical detailed definitions of helper functions should be placed here
  sorry

end num_paths_l549_549633


namespace Celine_returned_other_two_books_l549_549214

-- Definition of the conditions
def library_daily_charge : ℝ := 0.50
def books_borrowed : ℕ := 3
def days_for_first_book : ℕ := 20
def total_paid : ℝ := 41

-- Definition of the problem variables
def cost_first_book : ℝ := days_for_first_book * library_daily_charge
def remaining_cost : ℝ := total_paid - cost_first_book
def days_for_two_books : ℝ := remaining_cost / (library_daily_charge * 2)

-- Math proof problem statement
theorem Celine_returned_other_two_books :
  days_for_two_books = 31 :=
by
  sorry

end Celine_returned_other_two_books_l549_549214


namespace inversion_line_is_circle_l549_549650

noncomputable def invert_line_to_circle (R : ℝ) (O A B : Point) : Circle :=
  let O_prime := symmetric_point O AB in
  let inversion_O_prime := R^2 / distance O O_prime in
  ⟨ inversion_O_prime, O ⟩

theorem inversion_line_is_circle (R : ℝ) (O A B : Point) :
  let C := invert_line_to_circle R O A B in
  C.contains O :=
sorry

end inversion_line_is_circle_l549_549650


namespace max_value_sum_F_l549_549088

noncomputable def F (x : ℝ) (xs : Fin 2000 → ℝ) (i : Fin 2000) : ℝ :=
  x ^ 2300 / (xs.sum (fun j => (xs j) ^ 3509) - i * x ^ 3899 + 2000)

theorem max_value_sum_F (x : Fin 2000 → ℝ) (h : ∀ i, x i ∈ Set.Icc 0 1) :
  (Finset.univ.sum (fun i => F (x i) x i)) ≤ 2000 / 3999 := sorry

end max_value_sum_F_l549_549088


namespace sqrt_one_seventh_eq_abs_two_minus_sqrt_five_eq_l549_549491

theorem sqrt_one_seventh_eq : sqrt (1 / 49) = 1 / 7 :=
by sorry

theorem abs_two_minus_sqrt_five_eq : abs (2 - sqrt 5) = sqrt 5 - 2 :=
by
  have h : 2 < sqrt 5 := by
    -- Demonstrate that 2 is less than sqrt 5
    -- sqrt 5 ≈ 2.236, so this condition is true
    sorry
  show abs (2 - sqrt 5) = sqrt 5 - 2
    -- Given h, apply the condition to simplify
    sorry

end sqrt_one_seventh_eq_abs_two_minus_sqrt_five_eq_l549_549491


namespace mean_of_five_numbers_l549_549156

theorem mean_of_five_numbers (sum_of_numbers : ℚ) (number_of_elements : ℕ)
  (h_sum : sum_of_numbers = 3 / 4) (h_elements : number_of_elements = 5) :
  (sum_of_numbers / number_of_elements : ℚ) = 3 / 20 :=
by
  sorry

end mean_of_five_numbers_l549_549156


namespace financial_equation_l549_549076

theorem financial_equation
  (q : ℂ) (v : ℂ) (p : ℂ)
  (h1 : 3 * q - v = 8000)
  (h2 : q = 4)
  (h3 : v = 4 + 50 * complex.I) :
  p = 2669 + (50 / 3) * complex.I :=
sorry

end financial_equation_l549_549076


namespace four_weavers_four_days_l549_549210

theorem four_weavers_four_days (
  four_weavers_some_mats_in_4_days : ∃ m, 4 * m / 4 = m,
  eight_weavers_sixteen_mats_in_eight_days : ∃ m, 8 * m / 8 = 16
):
  ∃ m, 4 * m / 4 = 8 :=
by
  sorry

end four_weavers_four_days_l549_549210


namespace car_problem_system_l549_549797

variable (x y : ℝ)

def condition1 : Prop := x / 3 = y - 2
def condition2 : Prop := (x - 9) / 2 = y

theorem car_problem_system :
  (condition1 x y) ∧ (condition2 x y) ↔ 
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) :=
by
  have cond1 := condition1 x y
  have cond2 := condition2 x y
  exact ⟨λ ⟨h1, h2⟩, ⟨h1, h2⟩, λ ⟨h1, h2⟩, ⟨h1, h2⟩⟩

end car_problem_system_l549_549797


namespace remaining_edges_of_modified_cube_l549_549618

def initial_cube_edge_count : ℕ := 12
def removal_segment_per_corner : ℕ := 6
def corners_removed : ℕ := 4

theorem remaining_edges_of_modified_cube (s₁ s₂ c n : ℕ)
(h₁ : s₁ = 5) (h₂ : s₂ = 1) (h₃ : c = 4) (h₄ : n = 36) :
  initial_cube_edge_count + (removal_segment_per_corner * corners_removed) = n :=
by 
  unfold initial_cube_edge_count removal_segment_per_corner corners_removed;
  simp;
  exact h₄

end remaining_edges_of_modified_cube_l549_549618


namespace range_of_theta_l549_549736

theorem range_of_theta (θ : ℝ) (hθ : -π / 2 < θ ∧ θ < π / 2) :
  ∃ (h : x^2 + y^2 + x + sqrt 3 * y + tan θ = (0 : ℝ)), -π / 2 < θ ∧ θ < π / 4 :=
by
  sorry

end range_of_theta_l549_549736


namespace S_10_is_9217_l549_549745

open Set

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def nonempty_subsets (s : Set ℕ) : Set (Set ℕ) := {A | A ⊆ s ∧ A ≠ ∅}

def M_A (A : Set ℕ) : ℕ := Finset.max' (A.to_finset) (by 
  have h : A ≠ ∅ := A.prop.right
  exact Finset.Nonempty.to_finset.coe h
)

def S_10 : ℕ := ∑ A in (nonempty_subsets M).to_finset, M_A A

theorem S_10_is_9217 : S_10 = 9217 :=
by
  -- Here you would provide a proof that echoes the solution steps from above.
  sorry

end S_10_is_9217_l549_549745


namespace abc_inequality_l549_549973

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a * b + b * c + c * a ≥ a * Real.sqrt (b * c) + b * Real.sqrt (a * c) + c * Real.sqrt (a * b) :=
sorry

end abc_inequality_l549_549973


namespace maria_compensation_l549_549107

theorem maria_compensation (d : ℝ) (h1 : d ≥ 1600000) : ∃ c, c = 1400000 :=
by
  have h2 : c = 1400000, from sorry
  exact ⟨c, h2⟩

end maria_compensation_l549_549107


namespace complex_power_l549_549579

theorem complex_power : (1 - Complex.i)^4 = -4 :=
by
  sorry

end complex_power_l549_549579


namespace probability_of_drawing_letter_in_name_l549_549283

theorem probability_of_drawing_letter_in_name :
  let total_letters := 26
  let alonso_letters := ['a', 'l', 'o', 'n', 's']
  let number_of_alonso_letters := alonso_letters.length
  number_of_alonso_letters / total_letters = 5 / 26 :=
by
  sorry

end probability_of_drawing_letter_in_name_l549_549283


namespace distance_between_points_l549_549938

open Real

theorem distance_between_points :
  let P := (1, 3)
  let Q := (-5, 7)
  dist P Q = 2 * sqrt 13 :=
by
  let P := (1, 3)
  let Q := (-5, 7)
  sorry

end distance_between_points_l549_549938


namespace find_f_iterate_l549_549832

noncomputable def f (x p q : ℝ) : ℝ := x^2 + p*x + q

theorem find_f_iterate (p q : ℝ)
  (h1 : ∀ x, 3 ≤ x ∧ x ≤ 5 → |f x p q| ≤ 1 / 2) :
  let x := (7 + Real.sqrt 15) / 2 in
  (List.iterate (f · p q) 2017 x) = (7 - Real.sqrt 15) / 2 := 
by
  sorry

end find_f_iterate_l549_549832


namespace minimum_unique_points_l549_549956

def square (α : Type) : Type := 
{v : α × α // v.1 ≠ v.2} 

def points_on_sides (s : square ℝ) (p : ℝ) : Prop :=
∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ p = a ∨ p = b ∨ p = c ∨ p = d

theorem minimum_unique_points (s : square ℝ) :
  (∃! p : ℝ, (points_on_sides s p)) = 8 :=
sorry

end minimum_unique_points_l549_549956


namespace max_value_of_a_l549_549371

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := (x + 1) * (1 + Real.log (x + 1)) - a * x

theorem max_value_of_a (a : ℤ) : 
  (∀ x : ℝ, x ≥ -1 → (a : ℝ) * x ≤ (x + 1) * (1 + Real.log (x + 1))) → a ≤ 3 := sorry

end max_value_of_a_l549_549371


namespace always_positive_sum_reciprocal_inequality_l549_549590

-- Problem 1
theorem always_positive (x : ℝ) : x^6 - x^3 + x^2 - x + 1 > 0 :=
sorry

-- Problem 2
theorem sum_reciprocal_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  1/a + 1/b + 1/c ≥ 9 :=
sorry

end always_positive_sum_reciprocal_inequality_l549_549590


namespace mean_of_five_numbers_l549_549158

theorem mean_of_five_numbers (sum_of_numbers : ℚ) (number_of_elements : ℕ)
  (h_sum : sum_of_numbers = 3 / 4) (h_elements : number_of_elements = 5) :
  (sum_of_numbers / number_of_elements : ℚ) = 3 / 20 :=
by
  sorry

end mean_of_five_numbers_l549_549158


namespace complex_power_proof_l549_549584

noncomputable def complex_power : Prop :=
  (1 - complex.i)^4 = -4

theorem complex_power_proof : complex_power :=
by
  sorry

end complex_power_proof_l549_549584


namespace polynomial_expansion_p_eq_l549_549906

theorem polynomial_expansion_p_eq (p q : ℝ) (h1 : 10 * p^9 * q = 45 * p^8 * q^2) (h2 : p + 2 * q = 1) (hp : p > 0) (hq : q > 0) : p = 9 / 13 :=
by
  sorry

end polynomial_expansion_p_eq_l549_549906


namespace find_radius_E_l549_549668

-- Defining the parameters of the problem.
-- Radius of circles
def radius_A : ℝ := 8
def radius_B : ℝ := 4
def radius_C : ℝ := 1.5
def radius_D : ℝ := 1.5
def radius_E_const : ℝ := 2.837

-- The equilateral triangle T inscribed in circle A
-- Conditions on circular tangents
structure CirclesTangent at_vertex (radius: ℝ) : Prop :=
  (is_equilateral_triangle : radius_A = radius)

def Circle_B_tangent : Prop :=
  radius_A - radius_B = radius_B

def Circle_C_D_tangent : Prop :=
  radius_A - radius_C = radius_C

def Circles_external_tangent_to_E (radius : ℝ) : Prop :=
  -- Placeholder to describe the external tangency of circles B, C, and D to E
  sorry

-- Proof statement to formally show the radius of E
theorem find_radius_E : ∃ r : ℝ, r = radius_E_const ∧ Circles_external_tangent_to_E r :=
begin
  use 2.837,
  split,
  { refl },
  { sorry }
end

end find_radius_E_l549_549668


namespace max_value_of_distance_l549_549013

open Real EuclideanGeometry

theorem max_value_of_distance {a b c : EuclideanSpace ℝ} (ha : ‖a‖ = 4) (hb : ‖b‖ = 2 * sqrt 2) 
  (h_angle : real_inner a b = 4 * 2 * sqrt 2 * cos (π / 4)) 
  (h_dot_product : ∀ x, (x -ᵥ a) • (x -ᵥ b) = -1) :
  ∃ x, ‖(x -ᵥ a)‖ = sqrt 2 + 1 := sorry

end max_value_of_distance_l549_549013


namespace sum_of_c_and_d_l549_549229

definition quadrilateral_vertices := [(1,2), (4,5), (5,4), (4,1)]

theorem sum_of_c_and_d :
  let p1 := (1, 2)
  let p2 := (4, 5)
  let p3 := (5, 4)
  let p4 := (4, 1)
  let distance (a b : ℝ × ℝ) := Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)
  let d1 := distance p1 p2  -- 3*sqrt(2)
  let d2 := distance p2 p3  -- sqrt(2)
  let d3 := distance p3 p4  -- sqrt(10)
  let d4 := distance p4 p1  -- sqrt(10)
  let perimeter := d1 + d2 + d3 + d4
  let c := 4
  let d := 2
  c + d = 6 :=
by
  sorry

end sum_of_c_and_d_l549_549229


namespace interest_rate_increase_l549_549483

-- Definitions of constants used in our problem
def principal : ℝ := 875
def amount_original : ℝ := 956
def amount_increased : ℝ := 1061
def time : ℝ := 3

-- Define the rates
def original_interest_rate : ℝ := (amount_original - principal) / (principal * time)
def increased_interest_rate : ℝ := (amount_increased - principal) / (principal * time)

-- Define the percentage increase calculation
def percentage_increase : ℝ := ((increased_interest_rate - original_interest_rate) / original_interest_rate) * 100

-- State the theorem
theorem interest_rate_increase :
  percentage_increase = 129.6 := by
  sorry

end interest_rate_increase_l549_549483


namespace students_speaking_Gujarati_l549_549037

theorem students_speaking_Gujarati 
  (total_students : ℕ)
  (students_Hindi : ℕ)
  (students_Marathi : ℕ)
  (students_two_languages : ℕ)
  (students_all_three_languages : ℕ)
  (students_total_set: 22 = total_students)
  (students_H_set: 15 = students_Hindi)
  (students_M_set: 6 = students_Marathi)
  (students_two_set: 2 = students_two_languages)
  (students_all_three_set: 1 = students_all_three_languages) :
  ∃ (students_Gujarati : ℕ), 
  22 = students_Gujarati + 15 + 6 - 2 + 1 ∧ students_Gujarati = 2 :=
by
  sorry

end students_speaking_Gujarati_l549_549037


namespace modulus_of_complex_power_l549_549261

theorem modulus_of_complex_power :
  complex.abs ((2 : ℂ) + complex.i * (real.sqrt 5)) ^ 4 = 81 :=
sorry

end modulus_of_complex_power_l549_549261


namespace range_of_a_l549_549741

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 2) * a * x^2 - (x - 1) * Real.exp x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 x3 ∈ set.Icc 0 1, f a x1 + f a x2 ≥ f a x3) →
  1 ≤ a ∧ a ≤ 4 := 
sorry

end range_of_a_l549_549741


namespace polynomial_degree_l549_549192

theorem polynomial_degree (x : ℝ) : degree ((3 * x^2 + 11)^12) = 24 := 
by 
  sorry

end polynomial_degree_l549_549192


namespace functional_equation_solution_l549_549675

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x * f(x) + f(y)) = f(x)^2 + y) →
  (f = id ∨ f = λ x, -x) := 
by
  sorry

end functional_equation_solution_l549_549675


namespace smallest_four_digit_multiple_of_13_l549_549325

theorem smallest_four_digit_multiple_of_13 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 13 = 0) ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 13 ≠ 0 :=
by
  sorry

end smallest_four_digit_multiple_of_13_l549_549325


namespace dormitory_to_city_distance_l549_549419

theorem dormitory_to_city_distance
  (D : ℝ)
  (h1 : (1/5) * D + (2/3) * D + 14 = D) :
  D = 105 :=
by
  sorry

end dormitory_to_city_distance_l549_549419


namespace angle_DCE_invariant_l549_549092

noncomputable def measure_angle_DCE 
    (A B P C D E : ℝ × ℝ) 
    (h : Set (ℝ × ℝ)) 
    (AB_diameter : (A.1 = -1) ∧ (B.1 = 1) ∧ (A.2 = 0) ∧ (B.2 = 0))
    (P_on_AB : -1 ≤ P.1 ∧ P.1 ≤ 1 ∧ P.2 = 0)
    (C_on_h : C ∈ h)
    (PC_perpendicular_AB : P.1 = C.1 ∧ C.2 = sqrt(1 - P.1^2))
    (D_tangency : D.2 = 0 ∧ D.1 < P.1)
    (E_tangency : E.2 = 0 ∧ E.1 > P.1) : Prop :=
  ∠DCE = 45

theorem angle_DCE_invariant (A B P C D E : ℝ × ℝ) 
      (h : Set (ℝ × ℝ)) 
      (AB_diameter : (A.1 = -1) ∧ (B.1 = 1) ∧ (A.2 = 0) ∧ (B.2 = 0))
      (P_on_AB : -1 ≤ P.1 ∧ P.1 ≤ 1 ∧ P.2 = 0)
      (C_on_h : C ∈ h)
      (PC_perpendicular_AB : P.1 = C.1 ∧ C.2 = sqrt(1 - P.1^2))
      (D_tangency : D.2 = 0 ∧ D.1 < P.1)
      (E_tangency : E.2 = 0 ∧ E.1 > P.1) : measure_angle_DCE A B P C D E h AB_diameter P_on_AB C_on_h PC_perpendicular_AB D_tangency E_tangency :=
  by
    sorry

end angle_DCE_invariant_l549_549092


namespace parabola_translation_l549_549534

/-- Given the function y = f(x) = x^2 - 2, translating 3 units right and 1 unit down results in y = (x - 3)^2 - 3. -/
theorem parabola_translation (x : ℝ) :
  let f := λ x : ℝ, x^2 - 2
  let g := λ x : ℝ, (x - 3)^2 - 3
  f (x - 3) - 1 = g x :=
by
  sorry

end parabola_translation_l549_549534


namespace true_proposition_l549_549003

variable (Vector Point : Type)
variable (a b : Vector)
variable (A B C D : Point)

def proposition1 : Prop := 
  ∀ (u v : Vector), u = v → ∀ (p1 p2 q1 q2 : Point), (u = p1 - p2) ∧ (v = q1 - q2) → (p1 = q1) ∧ (p2 = q2)

def proposition2 : Prop := 
  ∀ (a b : Vector), (|a| = |b|) → (a = b ∨ a = -b)

def proposition3 : Prop :=
  ∀ (A B C D : Point), ¬Collinear A B C ∧ ¬Collinear B C D ∧ ¬Collinear C D A → (over_vector A B = over_vector D C) → Parallelogram A B C D

def proposition4 : Prop :=
  ∀ (a b : Vector), (|a| = |b|) ∧ Parallel a b ↔ a = b

theorem true_proposition : 
  proposition3 Vector Point A B C D :=
sorry

end true_proposition_l549_549003


namespace complex_conjugate_in_third_quadrant_l549_549363

noncomputable def z : ℂ := 2 * complex.I / (1 - complex.I)

theorem complex_conjugate_in_third_quadrant : 
  (1 - complex.I) * z = 2 * complex.I → 
  complex.conj(z).re < 0 ∧ complex.conj(z).im < 0 :=
by
  intro h
  sorry

end complex_conjugate_in_third_quadrant_l549_549363


namespace bc_over_ad_l549_549436

noncomputable def a : ℝ := 32 / 3
noncomputable def b : ℝ := 16 * Real.pi
noncomputable def c : ℝ := 24 * Real.pi
noncomputable def d : ℝ := 16 * Real.pi

theorem bc_over_ad : (b * c) / (a * d) = 9 / 4 := 
by 
  sorry

end bc_over_ad_l549_549436


namespace apples_left_l549_549694

theorem apples_left (apples_on_tree apples_on_ground apples_eaten : ℕ)
    (h1 : apples_on_tree = 5)
    (h2 : apples_on_ground = 8)
    (h3 : apples_eaten = 3) :
    apples_on_tree + apples_on_ground - apples_eaten = 10 :=
by
    rw [h1, h2, h3] -- rewrite using the conditions
    sorry -- proof goes here

end apples_left_l549_549694


namespace cuts_after_six_operations_l549_549277

theorem cuts_after_six_operations : 
  ∀ {n : ℕ}, let cuts (n : ℕ) := 3 * (4^n - 1) / 3 in
  (∑ k in finset.range 6, cuts k) = 4095 :=
by
  sorry

end cuts_after_six_operations_l549_549277


namespace intersection_M_N_l549_549378

def M := {x : ℝ | x < 2}
def N := {x : ℝ | 3^x > 1/3}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 < x ∧ x < 2} :=
  by
  sorry

end intersection_M_N_l549_549378


namespace ordered_arrays_count_l549_549626

theorem ordered_arrays_count (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|) →
  (|a + b + c| = 1 ∧ |a| + |b| + |c| = 1 ∧ |a - b| + |b - c| + |c - a| = 2) →
  ∃ as : finset (ℝ × ℝ × ℝ), as.card = 6 ∧ ∀ p ∈ as, p = (0, 0, 1) ∨ p = (0, 0, -1) ∨ p = (0, 1, 0) ∨ p = (0, -1, 0) ∨ p = (1, 0, 0) ∨ p = (-1, 0, 0) :=
by
  sorry

end ordered_arrays_count_l549_549626


namespace trick_or_treat_hours_l549_549531

variable (num_children : ℕ)
variable (houses_per_hour : ℕ)
variable (treats_per_house_per_kid : ℕ)
variable (total_treats : ℕ)

theorem trick_or_treat_hours (h : num_children = 3)
  (h1 : houses_per_hour = 5)
  (h2 : treats_per_house_per_kid = 3)
  (h3 : total_treats = 180) :
  total_treats / (num_children * houses_per_hour * treats_per_house_per_kid) = 4 :=
by
  sorry

end trick_or_treat_hours_l549_549531


namespace find_m_l549_549652

/-
Define a sequence recursively:
  x_0 = 7 
  x_{n+1} = (x_n^2 + 7x_n + 6) / (x_n + 8)

Let m be the least positive integer such that
  x_m ≤ 6 + 1 / 2^25
Prove that 151 ≤ m ≤ 450.
-/

def sequence (x : ℕ → ℚ) : Prop :=
  (x 0 = 7) ∧ (∀ n: ℕ, x (n + 1) = (x n ^ 2 + 7 * x n + 6) / (x n + 8))

def m_value (m : ℕ) (x : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, (sequence x) → (n ≤ m → x n > 6 + 1 / 2^25) ∧ (x m ≤ 6 + 1 / 2^25)

theorem find_m (x : ℕ → ℚ) (m : ℕ) : sequence x → m_value m x → (151 ≤ m ∧ m ≤ 450) :=
by
  sorry

end find_m_l549_549652


namespace range_of_a_l549_549504

-- Define the function f(x) = ax^3 - 3x
def f (a x : ℝ) : ℝ := a * x^3 - 3 * x

-- Define the interval (-1, 1)
def interval : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the condition for monotonic decreasing
def monotonic_decreasing (a : ℝ) : Prop :=
  ∀ x ∈ interval, deriv (λ x, f a x) x ≤ 0

-- Prove that if the function f(x) = ax^3 - 3x is monotonically decreasing on (-1, 1), then a ≤ 1
theorem range_of_a (a : ℝ) (h : monotonic_decreasing a) : a ≤ 1 := 
sorry -- Proof not required

end range_of_a_l549_549504


namespace solve_inequality_l549_549494

variable {x : ℝ}

theorem solve_inequality :
  (x - 8) / (x^2 - 4 * x + 13) ≥ 0 ↔ x ≥ 8 :=
by
  sorry

end solve_inequality_l549_549494


namespace youngest_child_age_l549_549920

theorem youngest_child_age (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 := 
by 
  sorry

end youngest_child_age_l549_549920


namespace johns_hat_cost_l549_549429

theorem johns_hat_cost :
  (∃ (n : ℕ), n = 20 * 7) ∧
  (∀ d : ℕ, d < 20 * 7 → (d % 2 = 1 → hat_cost_odd ∧ d % 2 = 0 → hat_cost_even)) ∧
  (hat_cost_odd = 45 ∧ hat_cost_even = 60) →
  total_cost = 20 * 7 / 2 * 45 + 20 * 7 / 2 * 60 :=
by
  let total_days := 20 * 7
  have odd_days := total_days / 2
  have even_days := total_days / 2
  let cost_odd := odd_days * 45
  let cost_even := even_days * 60
  let total_cost := cost_odd + cost_even
  exact $ 3150 + 4200 = 7350
  sorry

end johns_hat_cost_l549_549429


namespace parabola_intersection_min_y1_y2_sqr_l549_549358

theorem parabola_intersection_min_y1_y2_sqr :
  ∀ (x1 x2 y1 y2 : ℝ)
    (h1 : y1 ^ 2 = 4 * x1)
    (h2 : y2 ^ 2 = 4 * x2)
    (h3 : (∃ k : ℝ, x1 = 4 ∧ y1 = k * (4 - 4)) ∨ x1 = 4 ∧ y1 ≠ x2),
    ∃ m : ℝ, (y1^2 + y2^2) = m ∧ m = 32 := 
sorry

end parabola_intersection_min_y1_y2_sqr_l549_549358


namespace hat_cost_l549_549792

noncomputable def cost_of_hat (H : ℕ) : Prop :=
  let cost_shirts := 3 * 5
  let cost_jeans := 2 * 10
  let cost_hats := 4 * H
  let total_cost := 51
  cost_shirts + cost_jeans + cost_hats = total_cost

theorem hat_cost : ∃ H : ℕ, cost_of_hat H ∧ H = 4 :=
by 
  sorry

end hat_cost_l549_549792


namespace customers_in_other_countries_l549_549975

def total_customers : ℕ := 7422
def us_customers : ℕ := 723
def other_customers : ℕ := total_customers - us_customers

theorem customers_in_other_countries : other_customers = 6699 := by
  sorry

end customers_in_other_countries_l549_549975


namespace parallelogram_angle_bisector_l549_549038

theorem parallelogram_angle_bisector (a b S Q : ℝ) (α : ℝ) 
  (hS : S = a * b * Real.sin α)
  (hQ : Q = (1 / 2) * (a - b) ^ 2 * Real.sin α) :
  (2 * a * b) / (a - b) ^ 2 = (S + Q + Real.sqrt (Q ^ 2 + 2 * Q * S)) / S :=
by
  sorry

end parallelogram_angle_bisector_l549_549038


namespace chebyshevs_inequality_two_dim_l549_549082

open ProbabilityTheory

variables {Ω : Type*} {P : ProbMeasure Ω}

-- Given the definitions of random variables ξ and η
variables (ξ η : Ω → ℝ)

-- Expected values and variances
variables [is_finite_measure P] [integrable ξ P] [integrable η P]

-- Correlation coefficient ρ
variable (ρ : ℝ)

-- Positive real number ε
variable (ε : ℝ) (hε : ε > 0)

-- The statement to be proved
theorem chebyshevs_inequality_two_dim :
  P {ω | |ξ ω - 𝔼[ξ]| ≥ ε * sqrt (var ξ P) ∨ |η ω - 𝔼[η]| ≥ ε * sqrt (var η P)} ≤ 
  1 / (ε^2) * (1 + sqrt (1 - ρ^2)) :=
sorry

end chebyshevs_inequality_two_dim_l549_549082


namespace hexagon_perimeter_l549_549944

structure Point where
  x : ℝ
  y : ℝ

def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 0, y := 1 }
def C : Point := { x := 1, y := 1 }
def D : Point := { x := 1, y := 2 }
def E : Point := { x := 1, y := 3 }
def F : Point := { x := 2, y := 3 }

theorem hexagon_perimeter :
  distance A B = 1 ∧ distance B C = 1 ∧ distance C D = 1 ∧ distance D E = 1 ∧ distance E F = 1 ∧ distance A F = Real.sqrt 5 →
  distance A B + distance B C + distance C D + distance D E + distance E F + distance A F = 5 + Real.sqrt 5 :=
by
  intros h
  sorry

end hexagon_perimeter_l549_549944


namespace total_pennies_l549_549852

theorem total_pennies (rachelle gretchen rocky max taylor : ℕ) (h_r : rachelle = 720) (h_g : gretchen = rachelle / 2)
  (h_ro : rocky = gretchen / 3) (h_m : max = rocky * 4) (h_t : taylor = max / 5) :
  rachelle + gretchen + rocky + max + taylor = 1776 := 
by
  sorry

end total_pennies_l549_549852


namespace employed_females_percentage_l549_549789

-- Defining the population and percentages
variables (P : ℕ) (emp_percent stud_percent rt_unemp_percent emp_males_percent : ℚ)
variables (h_emp_percent : emp_percent = 0.64)
variables (h_stud_percent : stud_percent = 0.28)
variables (h_rt_unemp_percent : rt_unemp_percent = 0.08)
variables (h_emp_males_percent : emp_males_percent = 0.46)

-- Calculating employed people and employed males
noncomputable def total_employed := emp_percent * P
noncomputable def employed_males := emp_males_percent * P

-- Calculate employed females
noncomputable def employed_females := total_employed - employed_males

-- Calculate the percentage of employed people who are females
noncomputable def percent_employed_females := (employed_females / total_employed) * 100

-- The proof goal
theorem employed_females_percentage :
  percent_employed_females = 28.125 :=
by
  sorry

end employed_females_percentage_l549_549789


namespace angle_DCF_correct_l549_549138

-- Definitions for Square and Isosceles Right Triangle
structure Square (A B C D : Type) :=
  (A : A)
  (B : B)
  (C : C)
  (D : D)
  (sides_eq : True)
  (right_angles : True)

structure IsoscelesRightTriangle (A E F : Type) :=
  (A : A)
  (E : E)
  (F : F)
  (right_angle_at_E : True)
  (isoceles_legs : True)

-- Defining the problem conditions
variables {A B C D E F : Type} 
variables (sq : Square A B C D)
variables (tri : IsoscelesRightTriangle A E F)

-- Plumbing the conditions of points and segments
def point_E_on_segment_BC := True -- Simplified placeholder

-- The proof problem statement
theorem angle_DCF_correct :
  point_E_on_segment_BC →
  ∃ angle_DCF : ℝ, angle_DCF = 45 :=
by
  intros H
  existsi 45
  trivial -- Simplified placeholder, skips the actual proof
sorry

end angle_DCF_correct_l549_549138


namespace f_2010_eq_neg_sin_l549_549083

noncomputable def f : ℝ → ℝ := sin
noncomputable def f' := deriv f
noncomputable def f_n (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0     => f
  | _ + 1 => deriv (f_n n)

theorem f_2010_eq_neg_sin : f_n 2010 = λ x, -sin x := by
  sorry

end f_2010_eq_neg_sin_l549_549083


namespace modulus_of_z_l549_549364

noncomputable def z : ℂ := 1 / (complex.I - 1)

theorem modulus_of_z : complex.abs z = (real.sqrt 2) / 2 :=
by
  sorry

end modulus_of_z_l549_549364


namespace general_formula_and_sum_l549_549716

noncomputable def Sn_arithmetic_sum (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  n * a1 + (n * (n - 1)) / 2 * d

theorem general_formula_and_sum (a1 : ℕ) (S11 : ℕ) (n : ℕ) (a b T : ℕ) :
  a1 = 1 → S11 = 66 → Sn_arithmetic_sum 11 a1 1 = S11 → a = n → b = 2^n → T = 2^(n + 1) - 2 :=
by {
  intros h1 h2 h3 h4 h5,
  sorry
}

end general_formula_and_sum_l549_549716


namespace box_height_l549_549593

theorem box_height (h : ℝ) :
  ∃ (h : ℝ), 
  let large_sphere_radius := 3
  let small_sphere_radius := 1.5
  let box_width := 6
  h = 12 := 
sorry

end box_height_l549_549593


namespace distinguishable_squares_count_is_70_l549_549191

def count_distinguishable_squares : ℕ :=
  let total_colorings : ℕ := 2^9
  let rotation_90_270_fixed : ℕ := 2^3
  let rotation_180_fixed : ℕ := 2^5
  let average_fixed_colorings : ℕ :=
    (total_colorings + rotation_90_270_fixed + rotation_90_270_fixed + rotation_180_fixed) / 4
  let distinguishable_squares : ℕ := average_fixed_colorings / 2
  distinguishable_squares

theorem distinguishable_squares_count_is_70 :
  count_distinguishable_squares = 70 := by
  sorry

end distinguishable_squares_count_is_70_l549_549191


namespace sequence_formula_l549_549506

/-- The sequence given by the conditions a₁ = 1, a₂ = 3, a₃ = 6, a₄ = 10, ... has the general form aₙ = n * (n + 1) / 2. -/
theorem sequence_formula (n : ℕ) : 
  (∀ n, (∑ i in finset.range (n + 1), i) = n * (n + 1) / 2) := 
by sorry

end sequence_formula_l549_549506


namespace arithmetic_sequence_general_formula_sequence_l549_549800

theorem arithmetic_sequence (a S : ℕ → ℝ) (h1 : a 1 = 0.5) 
  (h2 : ∀ n, 1 < n → a n = 2 * S n ^ 2 / (2 * S n - 1)) :
  ∀ n, 1 < n → (1 / S n - 1 / S (n - 1) = 2) :=
by
  sorry

theorem general_formula_sequence (a S : ℕ → ℝ) (h1 : a 1 = 0.5) 
  (h2 : ∀ n, 1 < n → a n = 2 * S n ^ 2 / (2 * S n - 1)) :
  ∀ n, a n = (if n = 1 then 0.5 else -1 / (2 * n * (n - 1))) :=
by
  sorry

end arithmetic_sequence_general_formula_sequence_l549_549800


namespace max_intersections_circle_pentagon_l549_549942

theorem max_intersections_circle_pentagon : 
  ∃ (circle : Set Point) (pentagon : List (Set Point)),
    (∀ (side : Set Point), side ∈ pentagon → ∃ p1 p2 : Point, p1 ∈ circle ∧ p2 ∈ circle ∧ p1 ≠ p2) ∧
    pentagon.length = 5 →
    (∃ n : ℕ, n = 10) :=
by
  sorry

end max_intersections_circle_pentagon_l549_549942


namespace males_in_band_not_in_orchestra_or_choir_l549_549855

theorem males_in_band_not_in_orchestra_or_choir :
  let males_in_band := 120
  let females_in_band := 100
  let males_in_orchestra := 90
  let females_in_orchestra := 130
  let males_in_both_band_and_orchestra := 50
  let females_in_both_band_and_orchestra := 70
  let males_in_choir := 40
  let females_in_choir := 60
  let males_in_both_band_and_choir := 30
  let females_in_both_band_and_choir := 40
  let males_in_both_orchestra_and_choir := 20
  let females_in_both_orchestra_and_choir := 30
  let total_students_in_any := 260
  (males_in_band - (males_in_both_band_and_orchestra + males_in_both_band_and_choir - males_in_both_orchestra_and_choir)) = 60 := 
begin
  -- Given values
  let males_in_band := 120
  let females_in_band := 100
  let males_in_orchestra := 90
  let females_in_orchestra := 130
  let males_in_both_band_and_orchestra := 50
  let females_in_both_band_and_orchestra := 70
  let males_in_choir := 40
  let females_in_choir := 60
  let males_in_both_band_and_choir := 30
  let females_in_both_band_and_choir := 40
  let males_in_both_orchestra_and_choir := 20
  let females_in_both_orchestra_and_choir := 30
  let total_students_in_any := 260
  
  -- Calculation of males in band not in orchestra or choir
  let result := males_in_band - (males_in_both_band_and_orchestra + males_in_both_band_and_choir - males_in_both_orchestra_and_choir)
  
  -- Assertion that result is equal to 60
  show result = 60, from sorry
end

end males_in_band_not_in_orchestra_or_choir_l549_549855


namespace missing_water_calculation_l549_549598

def max_capacity : ℝ := 350000
def loss_rate1 : ℝ := 32000
def time1 : ℝ := 5
def loss_rate2 : ℝ := 10000
def time2 : ℝ := 10
def fill_rate : ℝ := 40000
def fill_time : ℝ := 3

theorem missing_water_calculation :
  350000 - ((350000 - (32000 * 5 + 10000 * 10)) + 40000 * 3) = 140000 :=
by
  sorry

end missing_water_calculation_l549_549598


namespace product_gcd_lcm_l549_549304

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l549_549304


namespace max_no_monochromatic_triangles_l549_549455

theorem max_no_monochromatic_triangles (P : Finset ℕ) (hP : P.card = 17)
  (colors : Finset ℕ) (hcolors : colors.card = 8)
  (coloring : (P.Pr × P.Pr) → colors)
  (no_mono_triangles : ∀ (a b c : P.Pr), a ≠ b → b ≠ c → c ≠ a → ¬(coloring (a, b) = coloring (b, c) ∧ coloring (b, c) = coloring (c, a))) :
  ∃ count : ℕ, count = 544 ∧
                (∀ a b c : P.Pr, a ≠ b → b ≠ c → c ≠ a → ¬(coloring (a, b) = coloring (a, c) ↔ count > 544)) := 
sorry

end max_no_monochromatic_triangles_l549_549455


namespace choose_integers_l549_549435

def smallest_prime_divisor (n : ℕ) : ℕ := sorry
def number_of_divisors (n : ℕ) : ℕ := sorry

theorem choose_integers :
  ∃ (a : ℕ → ℕ), (∀ i, i < 2022 → a i < a (i + 1)) ∧
  (∀ k, 1 ≤ k ∧ k ≤ 2022 →
    number_of_divisors (a (k + 1) - a k - 1) > 2023^k ∧
    smallest_prime_divisor (a (k + 1) - a k) > 2023^k
  ) :=
sorry

end choose_integers_l549_549435


namespace eccentricity_value_l549_549711

-- Define the hyperbola and conditions
variable (a b : ℝ) (ha : a > 0) (hb : b > 0)
variable (x y : ℝ)
def hyperbola := x^2 / a^2 - y^2 / b^2 = 1

-- Define the foci F1 and F2 and the points A and B with given triangle property
variable (m : ℝ)
variable (AF1 AB AF2 F1F2 F2B : ℝ)
variable (triangle_right_angle : AF2 = m - 2 * a ∧ F1F2 = real.sqrt 2 * m 
  ∧ AF1 = m ∧ AB = m ∧ F2B = real.sqrt 2 * m - 2 * a)

-- Define eccentricity e
def eccentricity (c a : ℝ) := c / a

-- The main statement: the value of e is 5 - 2√2
theorem eccentricity_value : 
  ∀ c, (4 * c = (5 / 2 - real.sqrt 2) * m) → eccentricity c a = 5 - 2 * real.sqrt 2 := 
by
  sorry

end eccentricity_value_l549_549711


namespace students_wrote_word_correctly_l549_549410

-- Definitions based on the problem conditions
def total_students := 50
def num_cat := 10
def num_rat := 18
def num_croc := total_students - num_cat - num_rat
def correct_cat := 15
def correct_rat := 15
def correct_total := correct_cat + correct_rat

-- Question: How many students wrote their word correctly?
-- Correct Answer: 8

theorem students_wrote_word_correctly : 
  num_cat + num_rat + num_croc = total_students 
  → correct_cat = 15 
  → correct_rat = 15 
  → correct_total = 30 
  → ∀ (num_correct_words : ℕ), num_correct_words = correct_total - num_croc 
  → num_correct_words = 8 := by 
  sorry

end students_wrote_word_correctly_l549_549410


namespace scientists_from_usa_l549_549124

theorem scientists_from_usa (total_scientists : ℕ)
  (from_europe : ℕ)
  (from_canada : ℕ)
  (h1 : total_scientists = 70)
  (h2 : from_europe = total_scientists / 2)
  (h3 : from_canada = total_scientists / 5) :
  (total_scientists - from_europe - from_canada) = 21 :=
by
  sorry

end scientists_from_usa_l549_549124


namespace handshaking_remainder_l549_549411

-- Define number of people
def num_people := 11

-- Define N as the number of possible handshaking ways
def N : ℕ :=
sorry -- This will involve complicated combinatorial calculations

-- Define the target result to be proven
theorem handshaking_remainder : N % 1000 = 120 :=
sorry

end handshaking_remainder_l549_549411


namespace sector_arc_length_l549_549340

theorem sector_arc_length (r : ℝ) (θ_deg : ℝ) (h_r : r = 6) (h_θ_deg : θ_deg = 60) :
  let θ_rad := θ_deg * (Real.pi / 180)
  in (r * θ_rad) = 2 * Real.pi :=
by
  sorry

end sector_arc_length_l549_549340


namespace arithmetic_sqrt_m_n_l549_549725

theorem arithmetic_sqrt_m_n (m n : ℕ)
  (h1 : sqrt (m - 3) = 3)
  (h2 : sqrt (n + 1) = 2) :
  sqrt (m - n) = 3 :=
sorry

end arithmetic_sqrt_m_n_l549_549725


namespace angle_bisector_length_l549_549715

theorem angle_bisector_length
  (a b c : ℝ)  -- side lengths of triangle ABC
  (C : ℝ)     -- angle C in triangle ABC
  (h : a + b > 0) -- ensure denominator is non-zero
  (D : Point) -- point where angle bisector of C intersects AB
  :
  let cos_val := Real.cos (C / 2)
  in CD = (2 * a * b * cos_val) / (a + b) := 
sorry

end angle_bisector_length_l549_549715


namespace question_1_question_2_question_3_l549_549723

noncomputable def F (a x : ℝ) : ℝ := 
  min (2 * |x - 1|) (x^2 - 2 * a * x + 4 * a - 2)

theorem question_1 (a : ℝ) (ha : 3 ≤ a) : 
  {x : ℝ | F a x = x^2 - 2 * a * x + 4 * a - 2 } = set.Icc 2 (2 * a) := sorry

theorem question_2 (a : ℝ) (ha : 3 ≤ a) : 
  (min (0 : ℝ) (-a^2 + 4 * a - 2) : ℝ) = (if 3 ≤ a ∧ a ≤ 2 + Real.sqrt 2 then 0 else 
      -a^2 + 4 * a - 2) := sorry

theorem question_3 (a : ℝ) (ha : 3 ≤ a) : 
  (max (34 - 8 * a) (if a > 4 then 2 else 34 - 8 * a) : ℝ) = (if 3 ≤ a ∧ a ≤ 4 then 34 - 8 * a 
      else 2) := sorry

end question_1_question_2_question_3_l549_549723


namespace intersection_product_l549_549799

noncomputable def rectangular_equation_of_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 2 * y + 1 = 0

def rectangular_coordinates_of_M := (1 : ℝ, 1 : ℝ)

noncomputable def parametric_equations_of_l (t : ℝ) : ℝ × ℝ :=
  (1 + (Real.sqrt 3 / 2) * t, 1 + (1 / 2) * t)

theorem intersection_product (MA MB : ℝ) : MA * MB = 3 :=
by
  sorry

/-- Main statement combining all previous definitions and theorems -/
noncomputable def math_proof_problem :=
  ∀ (t : ℝ),
  let M := rectangular_coordinates_of_M in
  let (x_t, y_t) := parametric_equations_of_l t in
  rectangular_equation_of_C x_t y_t ∧
  intersection_product (abs ((x_t - 1)^2 + (y_t - 1)^2)^.sqrt) = 3

#check math_proof_problem

end intersection_product_l549_549799


namespace flight_duration_NY_CapeTown_l549_549644

theorem flight_duration_NY_CapeTown :
  let london_departure_time := 6 -- 6:00 a.m. Monday ET
  let flight_to_NY_duration := 18 -- 18 hours flight to New York
  let NY_arrival_time := (london_departure_time + flight_to_NY_duration) % 24  -- 12:00 a.m. Tuesday ET
  let capetown_arrival_time := 10 -- 10:00 a.m. Tuesday ET
  capetown_arrival_time - NY_arrival_time = 10 := 
by
  have london_departure_time : ℕ := 6,
  have flight_to_NY_duration : ℕ := 18,
  have NY_arrival_time : ℕ := (london_departure_time + flight_to_NY_duration) % 24,
  have capetown_arrival_time : ℕ := 10,
  have flight_duration := capetown_arrival_time - NY_arrival_time,
  show flight_duration = 10, from sorry

end flight_duration_NY_CapeTown_l549_549644


namespace mean_of_five_numbers_l549_549171

theorem mean_of_five_numbers (S : ℚ) (n : ℕ) (h1 : S = 3/4) (h2 : n = 5) :
  (S / n) = 3/20 :=
by
  rw [h1, h2]
  sorry

end mean_of_five_numbers_l549_549171


namespace count_valid_three_digit_integers_l549_549765

theorem count_valid_three_digit_integers : 
  let valid_ints := { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧
                              (∃ d ∈ [n / 100, (n / 10) % 10, n % 10], d = 4) ∧
                              ((n / 100) ≠ 7 ∧ ((n / 10) % 10) ≠ 7 ∧ (n % 10) ≠ 7) ∧ 
                              (n % 10 ∈ [2, 4, 6, 8]) } in
  valid_ints.card = 64 :=
by
  sorry

end count_valid_three_digit_integers_l549_549765


namespace trigonometric_expression_simplification_l549_549206

theorem trigonometric_expression_simplification (α : ℝ) :
  (cos^2 (5 * Real.pi / 4 - 2 * α) - sin^2 (5 * Real.pi / 4 - 2 * α)) /
  ((cos (α / 2) + sin (α / 2)) * (cos (2 * Real.pi - α / 2) + cos (Real.pi / 2 + α / 2)) * sin α) = 
  4 * cos (2 * α) :=
by
  sorry

end trigonometric_expression_simplification_l549_549206


namespace circle_sum_maximum_l549_549897

theorem circle_sum_maximum :
  ∀ (A B C D : ℕ), 1 ≤ A ∧ A ≤ 8 ∧
                   1 ≤ B ∧ B ≤ 8 ∧
                   1 ≤ C ∧ C ≤ 8 ∧
                   1 ≤ D ∧ D ≤ 8 ∧
                   A ≠ B ∧ A ≠ C ∧ A ≠ D ∧
                   B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
                   (∃ X Y Z W : ℕ, {X, Y, Z, W} = {1, 2, 3, 4, 5, 6, 7, 8} \ {A, B, C, D} ∧
                    X + Y + Z + W = 36 - (A + B + C + D)) ∧
                   (A + B + C + D) % 4 = 0 →
  (36 + (A + B + C + D)) % 4 = 0 → (36 + (A + B + C + D)) / 4 = 15.

end circle_sum_maximum_l549_549897


namespace f_e_eq_one_plus_e_l549_549335

theorem f_e_eq_one_plus_e (f : ℝ → ℝ) (h : ∀ x, f x = f' 1 + x * Real.log x) (hf' : ∀ x, deriv f x = 1 + Real.log x) (hf'_at_1 : f' 1 = 1) : f Real.exp = 1 + Real.exp :=
by
  sorry

end f_e_eq_one_plus_e_l549_549335


namespace trapezoid_area_l549_549140

theorem trapezoid_area (AB CD AD AC BC : ℝ) (h1 : AB = 2 * CD) (h2 : AB = 2 * AD) (h3 : AC = a) (h4 : BC = b) :
  area_trapezoid ABCD = (3 / 4) * a * b :=
by
  -- Implementation of the trapezoid area calculation here
  sorry

end trapezoid_area_l549_549140


namespace XZ_length_l549_549056

theorem XZ_length (XYZ : Triangle) (X Y Z : Point)
  (angle_X : XYZ.angle X = 90)
  (YZ_length : XYZ.side Y Z = 30)
  (tan_Z_eq : tan (XYZ.angle Z) = 3 * sin (XYZ.angle Z)) :
  XYZ.side X Z = 10 :=
sorry

end XZ_length_l549_549056


namespace earn_2800_probability_l549_549850

def total_outcomes : ℕ := 7 ^ 4

def favorable_outcomes : ℕ :=
  (1 * 3 * 2 * 1) * 4 -- For each combination: \$1000, \$600, \$600, \$600; \$1000, \$1000, \$400, \$400; \$800, \$800, \$600, \$600; \$800, \$800, \$800, \$400

noncomputable def probability_of_earning_2800 : ℚ := favorable_outcomes / total_outcomes

theorem earn_2800_probability : probability_of_earning_2800 = 96 / 2401 := by
  sorry

end earn_2800_probability_l549_549850


namespace length_segment_AE_l549_549868

open Real

noncomputable def AB := 4
noncomputable def radius := 2
def AC := AB
def BC := AB

def D := {
  x : ℝ, 
  y : ℝ,
  condition : y = sqrt(3) * x
}

def E := {
  x : ℝ, 
  y : ℝ,
  condition : y = sqrt(3) * (BC - x)
}

theorem length_segment_AE :
  let AE := BC - 2 * sqrt(3) in
  AE = 4 - 2 * sqrt(3) :=
by 
  sorry

end length_segment_AE_l549_549868


namespace volume_decreases_by_sixteen_point_sixty_seven_percent_l549_549879

variable {P V k : ℝ}

-- Stating the conditions
def inverse_proportionality (P V k : ℝ) : Prop :=
  P * V = k

def increased_pressure (P : ℝ) : ℝ :=
  1.2 * P

-- Theorem statement to prove the volume decrease percentage
theorem volume_decreases_by_sixteen_point_sixty_seven_percent (P V k : ℝ)
  (h1 : inverse_proportionality P V k)
  (h2 : P' = increased_pressure P) :
  V' = V / 1.2 ∧ (100 * (V - V') / V) = 16.67 :=
by
  sorry

end volume_decreases_by_sixteen_point_sixty_seven_percent_l549_549879


namespace find_f_min_f_l549_549699

-- Definitions based on the problem conditions and correct answers

def f : ℝ → ℝ := by
  sorry  -- This needs to be formulated based on the proof of part (1).
         -- For now, we don't need to give the implementation.

-- First part of the proof: Show f(x) = x^2 - 2x

theorem find_f (x : ℝ) : f x = x^2 - 2x := by
  sorry

-- Second part of the proof: Show the minimum value of f(x) is -1 at x = 1.

theorem min_f : ∃ (x : ℝ), (∀ y, f y ≥ f x) ∧ f x = -1 := by
  use 1
  split
  { -- Prove that f(x) = -1 is a minimum value
    intro y
    sorry
  }
  { -- Prove that f(1) = -1
    sorry
  }

end find_f_min_f_l549_549699


namespace prime_neighbor_divisible_by_6_l549_549290

theorem prime_neighbor_divisible_by_6 (p : ℕ) (h_prime: Prime p) (h_gt3: p > 3) :
  ∃ k : ℕ, k ≠ 0 ∧ ((p - 1) % 6 = 0 ∨ (p + 1) % 6 = 0) :=
by
  sorry

end prime_neighbor_divisible_by_6_l549_549290


namespace masha_doll_arrangements_l549_549844

theorem masha_doll_arrangements : 
  let dolls := Fin 7
  let dollhouses := Fin 6 in
  (∃ a b : dolls, a ≠ b) ∧ 
  (∃ h : dollhouses, ∀ i ≠ h, ∃! d : dolls, d ≠ a ∧ d ≠ b) ∧ 
  ∃ h : dollhouses, ∃ a b : dolls, a ≠ b ∧ 
  (∀ (d1 d2 : dolls), d1 ≠ a ∧ d1 ≠ b ∧ d2 ≠ a ∧ d2 ≠ b → d1 ≠ d2) →
  21 * 6 * 120 = 15120 :=
by
  sorry

end masha_doll_arrangements_l549_549844


namespace page_width_calculation_l549_549845

-- Define the conditions
variables (cost_per_square_inch : ℝ) (half_page_cost : ℝ) (page_height : ℝ)

-- Set up the known values
def cost_per_square_inch := 8
def half_page_cost := 432
def page_height := 9

-- Define the problem
theorem page_width_calculation (W : ℝ) 
    (h1: half_page_cost = (1/2) * page_height * W * cost_per_square_inch)
    : W = 12 := 
begin
  --  Skipping the proof
  sorry
end

end page_width_calculation_l549_549845


namespace area_of_rectangle_ABCD_l549_549530

theorem area_of_rectangle_ABCD :
  ∀ (short_side long_side width length : ℝ),
    (short_side = 6) →
    (long_side = 6 * (3 / 2)) →
    (width = 2 * short_side) →
    (length = long_side) →
    (width * length = 108) :=
by
  intros short_side long_side width length h_short h_long h_width h_length
  rw [h_short, h_long] at *
  sorry

end area_of_rectangle_ABCD_l549_549530


namespace tournament_players_l549_549137

theorem tournament_players (n : ℕ) (h : n * (n - 1) / 2 = 56) : n = 14 :=
sorry

end tournament_players_l549_549137


namespace triangle_area_calculation_l549_549089

variables {A B C E : Type}
variables {BC AE : ℝ}

theorem triangle_area_calculation 
  (h_inc_tangent : ∃ l : Type, l ∈ line.perpendicular_bisector B C ∧ in_circle A B C l)
  (h_bc_ae : BC = 20) 
  (h_ae : AE = 20)
  (h_e_point : E ∈ A_excircle A B C ∧ E ∈ line B C) : 
  ∃ (area : ℝ), area = 100 * real.sqrt 2 :=
sorry

end triangle_area_calculation_l549_549089


namespace digit_in_decimal_repr_of_3_over_11_at_721_l549_549197

theorem digit_in_decimal_repr_of_3_over_11_at_721 :
  let decimal_repr := "27"
  let length := String.length decimal_repr
  (721 % length = 1) → 
  (decimal_repr.get 0 = '2') := 
by
  intros decimal_repr length mod_result digit
  sorry

end digit_in_decimal_repr_of_3_over_11_at_721_l549_549197


namespace q_evaluation_at_3_point_5_l549_549637

def q (x : ℝ) : ℝ :=
  |x - 3|^(1/3) + 2*|x - 3|^(1/5) + |x - 3|^(1/7)

theorem q_evaluation_at_3_point_5 : q 3.5 = 3 :=
by
  sorry

end q_evaluation_at_3_point_5_l549_549637


namespace smallest_area_right_triangle_l549_549546

theorem smallest_area_right_triangle (a b : ℕ) (h₁ : a = 4) (h₂ : b = 5) : 
  ∃ c, (c = 6 ∧ ∀ (x y : ℕ) (h₃ : x = 4 ∨ y = 4) (h₄ : x = 5 ∨ y = 5), c ≤ (x * y / 2)) :=
by {
  sorry
}

end smallest_area_right_triangle_l549_549546


namespace ellipse_equation_and_max_area_line_l549_549348

theorem ellipse_equation_and_max_area_line (
    (a b : ℝ) (a_gt_b : a > b) (b_gt_zero : b > 0)
    (eccentricity_sqrt3_div_2 : sqrt (a^2 - b^2) / a = sqrt 3 / 2)
    (slope_AF : ∀ c : ℝ, (2 / sqrt (a^2 - b^2) = 2 * sqrt 3 / 3) → (afeq : ∀ F∈E, (s(A,F) = 2 * sqrt 3 / 3)) :
    ∃ (a b : ℝ), (a = 2 ∧ b = 1 ∧ E = {P : ℝ × ℝ | P.1^2 / 4 + P.2^2 = 1}) ∧
      (∀ l: (ℝ → ℝ), (l = y := (sqrt(7) /2)x - 2) ∨ (l = y := -(sqrt(7) /2)x - 2))) :=
sorry

end ellipse_equation_and_max_area_line_l549_549348


namespace part_I_part_II_l549_549356

-- Given conditions
variable {α β γ : ℝ} (hα : 0 < α ∧ α < π / 2)
(hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2)
(h : cos α ^ 2 + cos β ^ 2 + cos γ ^ 2 = 1)

-- Part (I)
theorem part_I : tan α * tan β * tan γ ≥ 2 * sqrt 2 :=
sorry

-- Part (II)
theorem part_II : (3 * π / 4) < α + β + γ ∧ α + β + γ < π :=
sorry

end part_I_part_II_l549_549356


namespace find_x_coordinate_c_l549_549251

theorem find_x_coordinate_c (
  y_eq_x2 : ∀ x : ℝ, (∃ y : ℝ, y = x * x),
  A : (ℝ × ℝ) := (0, 0),
  B : (ℝ × ℝ) := (-3, 9),
  c : ℝ,
  c_gt_zero : c > 0,
  C : (ℝ × ℝ) := (c, c^2),
  BC_parallel_x_axis : B.2 = C.2,
  triangle_area_ABC : 1 / 2 * abs (c + 3) * (max 9 (c^2)) = 45
) : c = 7 :=
sorry

end find_x_coordinate_c_l549_549251


namespace continuity_f1_continuity_f2_l549_549190

variable {x : ℝ}

theorem continuity_f1 : Continuous (λ x, x^4 + 3 * x + 5) :=
  by continuity

theorem continuity_f2 : Continuous (λ x, x^2 * sin x - x^2 / (x^2 + 3)) :=
  by continuity

end continuity_f1_continuity_f2_l549_549190


namespace problem1_problem2_l549_549208

-- Problem 1 conditions
def P : ℝ × ℝ := (-4, 3)
def α := real.atan2 P.2 P.1
def tan_α := -3 / 4
def sin_α := 3 / 5

-- Problem 1 statement
theorem problem1 :
    let α := real.atan2 P.2 P.1 in
    let tan_α := -3 / 4 in
    let sin_α := 3 / 5 in
    (cos (π / 2 + α) * sin (-π - α)) / (cos (2019 * π / 2 - α) * tan (9 * π / 2 + α)) = 9 / 20 := 
by
    sorry

-- Problem 2 conditions
def x := real.atan(2018) - π / 4  -- Reversing the given tan condition
def tan_x := 2018

-- Problem 2 statement
theorem problem2 :
    let x := real.atan(2018) - π / 4 in
    (1 / cos (2 * x) + tan (2 * x)) = 2018 := 
by
    sorry

end problem1_problem2_l549_549208


namespace mode_zhang_hua_results_l549_549044

def zhang_hua_results : list ℝ := [7.6, 8.5, 8.6, 8.5, 9.1, 8.5, 8.4, 8.6, 9.2, 7.3]

theorem mode_zhang_hua_results : zhang_hua_results.mode = 8.5 :=
sorry

end mode_zhang_hua_results_l549_549044


namespace value_of_m_l549_549779

theorem value_of_m (m : ℕ) (h : 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 4934) : m = 4 :=
by
  sorry

end value_of_m_l549_549779


namespace circle_diameter_tangents_l549_549440

open Real

theorem circle_diameter_tangents {x y : ℝ} (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) :
  ∃ d : ℝ, d = sqrt (x * y) :=
by
  sorry

end circle_diameter_tangents_l549_549440


namespace concyclic_points_l549_549339

open EuclideanGeometry

-- Define the parallelogram ABCD with an obtuse angle at A
variables {A B C D : Point} (h : Parallelogram A B C D) (hA : ObtuseAngle (Angle A B C))

-- Define point H as the base of the perpendicular dropped from A to BC
variables {H : Point} (hH : FootOfPerpendicular A B C H)

-- Define the median CM of triangle ABC and let it intersect its circumcircle at K
variables {M K : Point} (hM : Median C A B M) (hK : IntersectsCircumcircle M C A B K)

theorem concyclic_points (h : Parallelogram A B C D) (hA : ObtuseAngle (Angle A B C))
  (hH : FootOfPerpendicular A B C H) (hM : Median C A B M) (hK : IntersectsCircumcircle M C A B K) :
  Concyclic K H C D :=
sorry

end concyclic_points_l549_549339


namespace square_area_l549_549929

theorem square_area (side_length : ℕ) (h_side_length : side_length = 6) : 
  side_length * side_length = 36 := 
by
  rw [h_side_length]
  exact Nat.mul_self 6

end square_area_l549_549929


namespace angle_equality_l549_549820

-- Define a structure for the geometric setup
structure Triangle (α : Type) [PlaneGeometry α] :=
  (A B C : α)

def is_circumcenter {α : Type} [PlaneGeometry α] (O : α) (T : Triangle α) : Prop :=
  ∀ P ∈ {T.A, T.B, T.C}, dist O P = dist O T.A

def is_foot_of_altitude {α : Type} [PlaneGeometry α] (H : α) (A : α) (BC_line : Line α) : Prop :=
  ∃ P Q : α, is_collinear P Q H ∧ P ∈ BC_line ∧ Q ∈ BC_line ∧ H ∈ line_through A

variables {α : Type} [PlaneGeometry α]

theorem angle_equality (T : Triangle α) (O H : α)
  (h₁ : is_circumcenter O T)
  (h₂ : is_foot_of_altitude H T.A (line_through T.B T.C)) :
  ∠(T.B, O, T.A) = ∠(T.C, A, H) :=
sorry

end angle_equality_l549_549820


namespace find_expression_l549_549696

theorem find_expression (a b : ℝ) (h₁ : a - b = 5) (h₂ : a * b = 2) :
  a^2 - a * b + b^2 = 27 := 
by
  sorry

end find_expression_l549_549696


namespace number_is_seven_point_five_l549_549122

theorem number_is_seven_point_five (x : ℝ) (h : x^2 + 100 = (x - 20)^2) : x = 7.5 :=
by
  sorry

end number_is_seven_point_five_l549_549122


namespace unique_lcm_gcd_pairs_l549_549674

open Nat

theorem unique_lcm_gcd_pairs :
  { (a, b) : ℕ × ℕ | lcm a b = gcd a b + 19 } = 
  { (1, 20), (20, 1), (4, 5), (5, 4), (19, 38), (38, 19) } :=
by
  sorry

end unique_lcm_gcd_pairs_l549_549674


namespace suzy_books_at_end_of_week_l549_549552

noncomputable def books_at_end_of_week : Nat :=
  let monday_books := 98 - 43 + 23
  let tuesday_books := monday_books - 28 + 35
  let wednesday_books := tuesday_books - 52 + 40 - 3
  let thursday_books := wednesday_books - 37 + 22
  let friday_books := thursday_books - 29 + 50
  friday_books

theorem suzy_books_at_end_of_week : books_at_end_of_week = 76 :=
by
  unfold books_at_end_of_week
  norm_num
  sorry

end suzy_books_at_end_of_week_l549_549552


namespace root_of_linear_eq_l549_549936

variable (a b : ℚ) -- Using rationals for coefficients

-- Define the linear equation
def linear_eq (x : ℚ) : Prop := a * x + b = 0

-- Define the root function
def root_function : ℚ := -b / a

-- State the goal
theorem root_of_linear_eq : linear_eq a b (root_function a b) :=
by
  unfold linear_eq
  unfold root_function
  sorry

end root_of_linear_eq_l549_549936


namespace three_pow_six_n_minus_two_pow_six_n_divisible_by_35_l549_549488

theorem three_pow_six_n_minus_two_pow_six_n_divisible_by_35 (n : ℤ) (hn : n ≥ 1) :
  35 ∣ (3 ^ (6 * n) - 2 ^ (6 * n)) :=
begin
  sorry
end

end three_pow_six_n_minus_two_pow_six_n_divisible_by_35_l549_549488


namespace magician_trick_success_l549_549987

theorem magician_trick_success {n : ℕ} (T_pos : ℕ) (deck_size : ℕ := 52) (discard_count : ℕ := 51):
  (T_pos = 1 ∨ T_pos = deck_size) → ∃ strategy : Type, ∀ spectator_choice : ℕ, (spectator_choice ≤ deck_size) → 
                          ((T_pos = 1 → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)
                          ∧ (T_pos = deck_size → ∃ k, k ≤ deck_size ∧ k ≠ T_pos ∧ discard_count - k = deck_size - 1)) :=
sorry

end magician_trick_success_l549_549987


namespace garden_area_remainder_l549_549478

theorem garden_area_remainder (n : ℕ) (hn : n = 202) :
  let m := 361803 in
  m % 1000 = 803 :=
by
  sorry

end garden_area_remainder_l549_549478


namespace probability_lines_intersect_inside_circle_l549_549925

theorem probability_lines_intersect_inside_circle (n : ℕ) (h : n = 2008) :
  ∃ (P : ℚ), P = 1 / 3 :=
by
  use 1 / 3
  sorry

end probability_lines_intersect_inside_circle_l549_549925


namespace exists_root_between_l549_549246

noncomputable def quadratic_function : ℝ → ℝ := fun x => a * x^2 + b * x + c

theorem exists_root_between (a b c : ℝ) 
  (h0 : quadratic_function 0 = -1) 
  (h0_5 : quadratic_function 0.5 = -0.5) 
  (h1 : quadratic_function 1 = 1) 
  (h1_5 : quadratic_function 1.5 = 3.5) 
  (h2 : quadratic_function 2 = 7) 
  : ∃ x_0, 0.5 < x_0 ∧ x_0 < 1 ∧ quadratic_function x_0 = 0 := 
sorry

end exists_root_between_l549_549246


namespace conjugate_of_z_l549_549398

theorem conjugate_of_z (z : ℂ) (h : z = (2 * complex.I) / (1 - complex.I)) : complex.conj z = -1 - complex.I :=
by {
  sorry
}

end conjugate_of_z_l549_549398


namespace polynomial_roots_are_distinct_l549_549816

-- Define that P is a polynomial and Q is a quadratic polynomial
variables {F : Type*} [field F]
variables {P Q : F[X]} (h_deg : P.nat_degree = n) (h_eq : P = Q * P.derivative.derivative)

-- The proof statement
theorem polynomial_roots_are_distinct (n : ℕ) (P : F[X]) (Q : F[X])
  (h_deg : P.nat_degree = n)
  (h_eq : P = Q * P.derivative.derivative) :
  (∃ r1 r2 : F, r1 ≠ r2 ∧ P.eval r1 = 0 ∧ P.eval r2 = 0) → (∀ r s : F, r ≠ s → P.eval r ≠ 0 → P.eval s ≠ 0) :=
by {
  sorry,
}

end polynomial_roots_are_distinct_l549_549816


namespace incorrect_transformation_l549_549558

theorem incorrect_transformation (a b : ℤ) : ¬ (a / b = (a + 1) / (b + 1)) :=
sorry

end incorrect_transformation_l549_549558


namespace construct_triangle_l549_549276

variable {b c : ℝ} (α varrho_b : ℝ)

def triangle_condition : Prop :=
  varrho_b * Real.tan (α / 2) > abs (b - c)

theorem construct_triangle
  (hb : 0 < b) (hc : 0 < c)
  (hα : 0 < α ∧ α < π)
  (hvarrho_b : 0 < varrho_b)
  : triangle_condition α varrho_b :=
sorry

end construct_triangle_l549_549276


namespace root_interval_range_l549_549778

noncomputable def f (m x : ℝ) : ℝ := m * 3^x - x + 3

theorem root_interval_range (m : ℝ) (h_neg : m < 0) (h_root : ∃ x ∈ (0, 1), f m x = 0) : -3 < m ∧ m < -2/3 := 
by
  sorry

end root_interval_range_l549_549778


namespace polynomial_lcm_bound_l549_549078

noncomputable theory
open scoped classical

-- Definitions and conditions
def polynomial_non_constant (P : ℕ → ℕ) : Prop :=
  ∃ d ≥ 1, ∃ (a_d a_0 : ℕ) (a : ℕ → ℕ), P = λ x, a_d * x^d + a x + a_0 ∧ a_d > 0

def has_rational_roots (P : ℕ → ℕ) (d : ℕ) : Prop :=
  ∃ (roots : List ℚ), roots.length = d ∧ ∀ r ∈ roots, P (r.num : ℕ) = 0

def non_negative_integer_coefficients (P : ℕ → ℕ) : Prop :=
  ∀ x, P x ≥ 0

-- The problem statement to prove
theorem polynomial_lcm_bound (P : ℕ → ℕ) (n m : ℕ) 
  (h1 : polynomial_non_constant P)
  (h2 : non_negative_integer_coefficients P)
  (h3 : ∃ d, has_rational_roots P d)
  (h4 : n > m) :
  Nat.lcm (List.map P (List.range' m (n - m + 1))) ≥ m * Nat.choose n m :=
sorry

end polynomial_lcm_bound_l549_549078


namespace triangle_inequality_l549_549441

theorem triangle_inequality (a b c : ℝ)
  (h1 : b + c > a)
  (h2 : c + a > b)
  (h3 : a + b > c) :
  ab + bc + ca ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 < 2(ab + bc + ca) :=
sorry

end triangle_inequality_l549_549441


namespace jill_total_tax_percentage_l549_549847

noncomputable def total_amount_spent : ℝ := 100
noncomputable def clothing_percentage : ℝ := 0.40
noncomputable def food_percentage : ℝ := 0.30
noncomputable def other_items_percentage : ℝ := 0.30

noncomputable def clothing_discount : ℝ := 0.10
noncomputable def food_discount : ℝ := 0.05
noncomputable def other_items_discount : ℝ := 0.07

noncomputable def clothing_tax_rate : ℝ := 0.04
noncomputable def food_tax_rate : ℝ := 0.0
noncomputable def other_items_tax_rate : ℝ := 0.08

def total_tax_percentage (T : ℝ) := 
  let clothing_initial := T * clothing_percentage
  let food_initial := T * food_percentage
  let other_items_initial := T * other_items_percentage

  let clothing_after_discount := clothing_initial * (1 - clothing_discount)
  let food_after_discount := food_initial * (1 - food_discount)
  let other_items_after_discount := other_items_initial * (1 - other_items_discount)

  let clothing_tax := clothing_after_discount * clothing_tax_rate
  let food_tax := food_after_discount * food_tax_rate
  let other_items_tax := other_items_after_discount * other_items_tax_rate

  let total_tax := clothing_tax + food_tax + other_items_tax
  (total_tax / T) * 100

theorem jill_total_tax_percentage : total_tax_percentage total_amount_spent = 3.672 :=
by
  sorry

end jill_total_tax_percentage_l549_549847


namespace problem_l549_549459

def f (x : ℝ) : ℝ := sorry

-- Given conditions
axiom cond1 : ∀ x : ℝ, f (2 - x) = f (2 + x)
axiom cond2 : ∀ x : ℝ, f (5 - x) = f (5 + x)
axiom cond3 : f 1 = 0
axiom cond4 : f 3 = 0

-- The problem to prove
theorem problem : ∃ n : ℕ, n = 1347 ∧ ∀ x ∈ Icc (-2020:ℝ) (2020:ℝ), f x = 0 → x ∈ Ico (-2020:ℝ) 2021 ∧ ∃ k : ℤ, x = 1 + 6 * k ∨ x = 3 + 6 * k := 
sorry

end problem_l549_549459


namespace gcd_lcm_product_24_60_l549_549310

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l549_549310


namespace correct_assignment_is_stmtD_l549_549200

-- Define the statements A, B, C, and D
def stmtA := (5 = a)
def stmtB := (a + 2 = a)
def stmtC := (a = b ∧ b = 4)
def stmtD := (a = 2 * a)

-- Define the correct assignment statement
def is_assignment_statement (stmt : Prop) : Prop :=
stmt = stmtD

-- Theorem that proves the correct assignment is statement D
theorem correct_assignment_is_stmtD (a b : ℤ) :
  is_assignment_statement stmtD := 
by 
  sorry

end correct_assignment_is_stmtD_l549_549200


namespace sum_ratio_l549_549840

noncomputable theory
open_locale classical

-- Definitions based on the conditions in the problem
def geometric_sequence (a1 q : ℕ → ℝ) : (ℕ → ℝ) := λ n, a1 * (q ^ (n - 1))
def sum_of_geometric_sequence (a1 q n : ℕ → ℝ) := if q = 1 then n * a1 else (a1 * (1 - q ^ n)) / (1 - q)

-- Given condition
def condition (a1 q : ℕ → ℝ) : Prop := 27 * (geometric_sequence a1 q 3) - (geometric_sequence a1 q 6) = 0

-- Theorem to prove
theorem sum_ratio (a1 q : ℕ → ℝ) (h : condition a1 q) :
  sum_of_geometric_sequence a1 q 6 / sum_of_geometric_sequence a1 q 3 = 28 :=
sorry

end sum_ratio_l549_549840


namespace total_fencing_cost_l549_549147

-- Definitions of the given conditions
def length : ℝ := 57
def breadth : ℝ := length - 14
def cost_per_meter : ℝ := 26.50

-- Definition of the total cost calculation
def total_cost : ℝ := 2 * (length + breadth) * cost_per_meter

-- Statement of the theorem to be proved
theorem total_fencing_cost :
  total_cost = 5300 := by
  -- Proof is omitted
  sorry

end total_fencing_cost_l549_549147


namespace friend_gets_15_oranges_l549_549218

-- Define the total number of oranges
def total_oranges : ℝ := 75

-- Condition 1: He gives 40% of the oranges to his brother
def brother_oranges : ℝ := 0.4 * total_oranges

-- Condition 2: The remainder after giving to his brother
def remainder_after_brother : ℝ := total_oranges - brother_oranges

-- Condition 3: He gives one-third of the remainder to his friend
def friend_oranges : ℝ := remainder_after_brother / 3

theorem friend_gets_15_oranges :
  friend_oranges = 15 :=
by
  sorry

end friend_gets_15_oranges_l549_549218


namespace coordinates_of_P_l549_549784

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def tangent_slope_parallel : ℝ := 3

theorem coordinates_of_P : 
  ∃ P : ℝ × ℝ, (let x := P.1 in let y := P.2 in (f'(x) = tangent_slope_parallel) ∧ (f(x) = y)) ↔ 
  P = (Real.exp 2, 2 * Real.exp 2) :=
by
  sorry

end coordinates_of_P_l549_549784


namespace increasing_function_range_a_l549_549703

theorem increasing_function_range_a (a : ℝ) :
  (∀ x : ℝ, (cos x + a) ≥ 0) →
  (a ≥ 1) :=
by 
  intro h
  have h_cos : ∀ x, cos x ≤ 1 := λ x, cos_le_one x
  have h_zero : ∀ x, 0 ≤ cos x + a := h
  sorry

end increasing_function_range_a_l549_549703


namespace cos_angle_Z_l549_549046

variables (X Y Z : Type) [Points X Y Z]
variables (angleX : angle X Y Z = 90)
variables (distXY : distance X Y = 9)
variables (distYZ : distance Y Z = 15)

theorem cos_angle_Z :
  ∃ XZ, distance X Z = XZ ∧ XZ = 12 ∧ cos (angle X Z Y) = 4 / 5 := sorry

end cos_angle_Z_l549_549046


namespace width_of_pool_l549_549173

-- The conditions
def length := 50 -- feet
def depth := 0.5 -- feet
def volume_gallons := 4687.5 -- gallons
def gallon_to_cubic_feet := 1 / 7.48052 -- conversion rate

-- Calculate the volume in cubic feet
def volume_cubic_feet := volume_gallons * gallon_to_cubic_feet

-- Define the width based on the volume formula
def width : ℝ := volume_cubic_feet / (length * depth)

-- The proof statement
theorem width_of_pool :
  width = 25.05 :=
by
  -- The proof will go here.
  sorry

end width_of_pool_l549_549173


namespace problem_statement_l549_549442

theorem problem_statement
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := 
by
  sorry

end problem_statement_l549_549442


namespace train_speed_l549_549622

theorem train_speed
  (distance: ℝ) (time_in_minutes : ℝ) (time_in_hours : ℝ) (speed: ℝ)
  (h1 : distance = 20)
  (h2 : time_in_minutes = 10)
  (h3 : time_in_hours = time_in_minutes / 60)
  (h4 : speed = distance / time_in_hours)
  : speed = 120 := 
by
  sorry

end train_speed_l549_549622


namespace rook_tour_impossible_l549_549128

-- Define the condition based on the problem's map: a bipartite graph with 21 countries
structure BipartiteMap :=
  (countries : Finset Nat)
  (edges : Finset (Nat × Nat))
  (is_bipartite : ∀ u v, (u, v) ∈ edges → (even u ↔ odd v))

-- Instantiate the given condition
def given_map : BipartiteMap :=
  { countries := Finset.range 21,
    edges := { (i, j) | i ∈ Finset.range 21 ∧ j ∈ Finset.range 21 ∧ i ≠ j},
    is_bipartite := by {
      intros u v uv_edge,
      sorry -- Proof of bipartiteness, assuming proper definition and properties
    }
  }

-- Prove the main theorem
theorem rook_tour_impossible : 
  ∀ map : BipartiteMap, map.countries.card = 21 → ¬∃ tour : List Nat, 
    (∀ u v ∈ tour, (u, v) ∈ map.edges) ∧ 
    (tour.erase_dup = tour) :=
begin
  intros map map_countries,
  sorry -- Full proof of the impossibility
end

end rook_tour_impossible_l549_549128


namespace jars_needed_l549_549426

theorem jars_needed (total_cherry_tomatoes : ℕ) (cherry_tomatoes_per_jar : ℕ) (h1 : total_cherry_tomatoes = 56) (h2 : cherry_tomatoes_per_jar = 8) : total_cherry_tomatoes / cherry_tomatoes_per_jar = 7 :=
by {
  rw [h1, h2],
  exact Nat.div_eq_of_eq_mul_left (show 8 > 0 from dec_trivial) dec_trivial
}

end jars_needed_l549_549426


namespace find_polynomial_q_l549_549295

noncomputable def q : ℕ → ℝ → ℝ :=
  λ a x, a * (x - 1) * (x + 3)

theorem find_polynomial_q :
  ∃ a : ℝ, q a 2 = 24 ∧ q a x = (24 / 5) * x^2 + (48 / 5) * x - (72 / 5) :=
sorry

end find_polynomial_q_l549_549295


namespace similarity_transformation_l549_549824

theorem similarity_transformation (C C' : ℝ × ℝ) (r : ℝ) (h1 : r = 3) (h2 : C = (4, 1))
  (h3 : C' = (r * 4, r * 1)) : (C' = (12, 3) ∨ C' = (-12, -3)) := by
  sorry

end similarity_transformation_l549_549824


namespace point_P_location_l549_549781

theorem point_P_location (a b : ℝ) : (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) → a^2 + b^2 > 1 :=
by sorry

end point_P_location_l549_549781


namespace age_difference_l549_549962

-- Defining the conditions
variables (a b c : Nat)

-- Given conditions
def condition1 := b = 2 * c
def condition2 := a + b + c = 27
def condition3 := b = 10

-- Goal: Prove that a - b = 2
theorem age_difference (h1 : condition1) (h2 : condition2) (h3 : condition3) : a - b = 2 := by
  sorry

end age_difference_l549_549962


namespace find_c_l549_549453

variables {α : Type*} [LinearOrderedField α]

def p (x : α) : α := 3 * x - 9
def q (x : α) (c : α) : α := 4 * x - c

-- We aim to prove that if p(q(3,c)) = 6, then c = 7
theorem find_c (c : α) : p (q 3 c) = 6 → c = 7 :=
by
  sorry

end find_c_l549_549453


namespace area_under_cos_l549_549887

theorem area_under_cos :
  ∫ x in (0 : ℝ)..(3 * Real.pi / 2), |Real.cos x| = 3 :=
by
  sorry

end area_under_cos_l549_549887


namespace ellipse_necessary_but_not_sufficient_condition_l549_549750

theorem ellipse_necessary_but_not_sufficient_condition
  (F1 F2 M : Point)
  (a : ℝ) (h_a_pos : 0 < a) :
  (|M.dist F1 + M.dist F2 = 2 * a) →
  (∀ M, |M.dist F1 + M.dist F2 = 2 * a → ∃ F1 F2, M.turn F1 F2) := sorry

end ellipse_necessary_but_not_sufficient_condition_l549_549750


namespace alice_favorite_number_l549_549631

-- Define the conditions for Alice's favorite number
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

-- Define the problem statement
theorem alice_favorite_number :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 200 ∧
           n % 13 = 0 ∧
           n % 3 ≠ 0 ∧
           sum_of_digits n % 4 = 0 ∧
           n = 130 :=
by
  sorry

end alice_favorite_number_l549_549631


namespace sum_of_sequence_l549_549714

noncomputable def S_n (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - n
noncomputable def a_n (n : ℕ) : ℕ := 2^n - 1
noncomputable def b_n (n : ℕ) : ℕ := 1 / (a_n (n + 1)) + 1 / (a_n n * a_n (n + 1))
noncomputable def T_n (b : ℕ → ℕ) (n : ℕ) : ℕ := ∑ k in finset.range n, b k

theorem sum_of_sequence {a_n b_n : ℕ → ℕ} (a_def : ∀ n, a_n n = 2^n - 1) :
  T_n b_n n = 1 - 1 / (2^(n+1) - 1) :=
sorry

end sum_of_sequence_l549_549714


namespace part1_part2_l549_549005

noncomputable def omega (f : ℝ → ℝ) : ℝ := 
  if h : ∃ ω, ∃ φ, f = λ x, cos (ω * x + φ) ∧ 
                  ω > 0 ∧ 0 < φ ∧ φ < π / 2 ∧ (2 * π / ω = π) ∧ 
                  (f (π / 3) = -sqrt 3 / 2) 
    then Classical.choose (Classical.choose_spec h).1 
    else 0 -- assuming this 0 case covers if f is not the right form

noncomputable def phi (f : ℝ → ℝ) : ℝ := 
  if h : ∃ ω, ∃ φ, f = λ x, cos (ω * x + φ) ∧ 
                  ω > 0 ∧ 0 < φ ∧ φ < π / 2 ∧ (2 * π / ω = π) ∧ 
                  (f (π / 3) = -sqrt 3 / 2)
    then Classical.choose (Classical.choose_spec h).2 
    else 0 -- assuming this 0 case covers if f is not the right form

theorem part1 (f : ℝ → ℝ) : omega f = 2 ∧ phi f = π / 6 := sorry

theorem part2 (f : ℝ → ℝ) (h : ∀ x, f x > 1 / 2) : 
  ∀ x, x ∈ set.Ioo (k * π - π / 4) (k * π + π / 12) ∧ k ∈ set.Z :=
  sorry

end part1_part2_l549_549005


namespace misha_grade_students_l549_549464

theorem misha_grade_students (n : ℕ) (h1 : n = 75) (h2 : n = 75) : 2 * n - 1 = 149 := 
by
  sorry

end misha_grade_students_l549_549464


namespace original_cost_l549_549069

theorem original_cost (SP : ℝ) (C : ℝ) (h1 : SP = 540) (h2 : SP = C + 0.35 * C) : C = 400 :=
by {
  sorry
}

end original_cost_l549_549069


namespace solid_with_triangle_front_and_quadrilateral_side_is_triangular_prism_l549_549236

theorem solid_with_triangle_front_and_quadrilateral_side_is_triangular_prism :
  (∀ S, (S.front_view = View.triangle) ∧ (S.side_view = View.quadrilateral) → S = Solid.triangular_prism) := 
sorry

end solid_with_triangle_front_and_quadrilateral_side_is_triangular_prism_l549_549236


namespace minimum_lines_needed_l549_549224

structure Point where
  x : ℝ
  y : ℝ

structure ColombianConfiguration where
  red_points : Fin 2013 → Point
  blue_points : Fin 2014 → Point
  no_three_collinear : ∀ (p1 p2 p3 : Point), (p1 ≠ p2) → (p2 ≠ p3) → (p1 ≠ p3) → ¬(Collinear p1 p2 p3)

structure ArrangementOfLines (k : ℕ) where
  lines : Fin k → (ℝ × ℝ × ℝ) -- Lines are represented as ax + by + c = 0
  no_line_passes_through_points : ∀ (l : Fin k) (p : Point), ¬(is_on_line l p)
  no_region_contains_both_colors : ∃ (regions : Fin k → Set Point),
                                     Disjoint (regions 0) (regions 1)

noncomputable def least_k (conf : ColombianConfiguration) : ℕ :=
  if ((∀ k, ∃ l, ArrangementOfLines k) ∧ (k ≥ 2013)) then 2013 else 0

theorem minimum_lines_needed (conf : ColombianConfiguration) : least_k conf = 2013 := by
  sorry

end minimum_lines_needed_l549_549224


namespace solve_for_y_l549_549672

theorem solve_for_y (y : ℝ) (h : log 4 (3 * y - 2) = 2) : y = 6 := 
by 
  sorry

end solve_for_y_l549_549672


namespace Toby_walks_thursday_l549_549180

theorem Toby_walks_thursday (
  steps_sunday : Nat := 9400,
  steps_monday : Nat := 9100,
  steps_tuesday : Nat := 8300,
  steps_wednesday : Nat := 9200,
  avg_friday_saturday : Nat := 9050,
  total_goal : Nat := 9000 * 7
) : 
  let total_steps_needed := total_goal
  let total_steps_sun_wed := steps_sunday + steps_monday + steps_tuesday + steps_wednesday
  let total_steps_fri_sat := avg_friday_saturday * 2
  total_steps_needed - (total_steps_sun_wed + total_steps_fri_sat) = 8900 :=
by
  -- proof goes here
  sorry

end Toby_walks_thursday_l549_549180


namespace white_area_of_sign_l549_549996

theorem white_area_of_sign :
  let rect_area := 4 * 20,
      m_area := 2 * (4 * 1) + 2 * 2,
      a_area := 1 * (4 * 1) + 1 * 3 + 0.5,
      t_area := 1 * (1 * 4) + 1 * (3 * 1),
      h_area := 2 * (4 * 1) + 1 * 3
  in rect_area - (m_area + a_area + t_area + h_area) = 42.5 := by
  sorry

end white_area_of_sign_l549_549996


namespace find_monthly_fee_l549_549329

-- Definitions from conditions
def monthly_fee (total_bill : ℝ) (cost_per_minute : ℝ) (minutes_used : ℝ) : ℝ :=
  total_bill - cost_per_minute * minutes_used

-- Theorem stating the question
theorem find_monthly_fee :
  let total_bill := 12.02
  let cost_per_minute := 0.25
  let minutes_used := 28.08
  total_bill - cost_per_minute * minutes_used = 5.00 :=
by
  -- Definition of variables used in the theorem
  let total_bill := 12.02
  let cost_per_minute := 0.25
  let minutes_used := 28.08
  
  -- The statement of the theorem and leaving the proof as an exercise
  show total_bill - cost_per_minute * minutes_used = 5.00
  sorry

end find_monthly_fee_l549_549329


namespace stacy_height_proof_l549_549497

noncomputable def height_last_year : ℕ := 50
noncomputable def brother_growth : ℕ := 1
noncomputable def stacy_growth : ℕ := brother_growth + 6
noncomputable def stacy_current_height : ℕ := height_last_year + stacy_growth

theorem stacy_height_proof : stacy_current_height = 57 := 
by
  sorry

end stacy_height_proof_l549_549497


namespace solid_surface_area_l549_549666

-- Define the conditions
def unit_cube := ℕ

def base_layer (n : ℕ) := n = 7

def top_cube_on_second := True

def eight_unit_cubes := 8

-- Define the goal based on the conditions
def surface_area_of_solid := 34

-- The statement
theorem solid_surface_area :
  ∀ (cubes : unit_cube) (base : unit_cube) (top : Prop), 
  base_layer base → top_cube_on_second top → eight_unit_cubes = cubes → 
  surface_area_of_solid = 34 :=
by
  intros
  assumption
  sorry

end solid_surface_area_l549_549666


namespace lives_per_player_l549_549176

-- Definitions based on the conditions
def initial_players : Nat := 2
def joined_players : Nat := 2
def total_lives : Nat := 24

-- Derived condition
def total_players : Nat := initial_players + joined_players

-- Proof statement
theorem lives_per_player : total_lives / total_players = 6 :=
by
  sorry

end lives_per_player_l549_549176


namespace sqrt_of_sixteen_l549_549917

theorem sqrt_of_sixteen (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := 
sorry

end sqrt_of_sixteen_l549_549917


namespace intersection_M_N_l549_549747

def M : set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} :=
by sorry

end intersection_M_N_l549_549747


namespace fixed_point_T_l549_549271

variables {O : Type*} [circle O] {B C A D J K E M N T : Type*}
variables (O : circle) (B C : O) (hBC : ¬ diameter O B C) (A : O) (hA : A ≠ B ∧ A ≠ C)
variables (D : midpoint B C) (J : midpoint C A) (K : midpoint A B)
variables (E : foot A B C) (M : foot B D J) (N : foot C D K)
variables (T : tangent_meet M N (circumcircle E M N))

theorem fixed_point_T (A : O) (hA : A ≠ B ∧ A ≠ C) : 
  ∃ T0 : O, ∀ A : O, (A ≠ B ∧ A ≠ C) → meet_tangent_circumcircle (foot B D J) (foot C D K) = T0 :=
sorry

end fixed_point_T_l549_549271


namespace proof_problem_l549_549769

/-- Given the condition -/
def condition (x : ℝ) : Prop :=
  x + sqrt (x^2 + 1) + (1 / (x + sqrt (x^2 + 1))) = 22

/-- Mathematically equivalent proof problem -/
theorem proof_problem (x : ℝ) (h : condition x) : 
  x^2 - sqrt (x^4 + 1) + (1 / (x^2 - sqrt (x^4 + 1))) = 242 :=
sorry

end proof_problem_l549_549769


namespace jori_water_left_l549_549073

theorem jori_water_left  {initial_usage : ℚ} (h_initial : initial_usage = 2) (used : initial_usage = 7/6) : 
  ∃ (remaining : ℚ), remaining = 5/6 :=
by
  existsi (2 - 7 / 6)
  rw [h_initial, used]
  norm_num
  refl

end jori_water_left_l549_549073


namespace female_democrats_count_l549_549204

theorem female_democrats_count 
  (F M : ℕ) 
  (total_participants : F + M = 750)
  (female_democrats : ℕ := F / 2) 
  (male_democrats : ℕ := M / 4)
  (total_democrats : female_democrats + male_democrats = 250) :
  female_democrats = 125 := 
sorry

end female_democrats_count_l549_549204


namespace Mrs_Heine_treats_l549_549474

theorem Mrs_Heine_treats :
  let dogs := 2
  let cats := 1
  let parrots := 3
  let biscuits_per_dog := 3
  let treats_per_cat := 2
  let sticks_per_parrot := 1
  let total_treats := dogs * biscuits_per_dog + cats * treats_per_cat + parrots * sticks_per_parrot
  total_treats = 11 :=
by
  let dogs := 2
  let cats := 1
  let parrots := 3
  let biscuits_per_dog := 3
  let treats_per_cat := 2
  let sticks_per_parrot := 1
  let total_treats := dogs * biscuits_per_dog + cats * treats_per_cat + parrots * sticks_per_parrot
  show total_treats = 11
  sorry

end Mrs_Heine_treats_l549_549474


namespace Thunderbolts_lineup_count_l549_549885

theorem Thunderbolts_lineup_count (n_players n_AliceZen : ℕ) (n_lineup : ℕ) (n_remaining : ℕ) (comb_13_5 comb_13_6 : ℕ) :
  n_players = 15 →
  n_AliceZen = 2 →
  n_lineup = 6 →
  n_remaining = 13 →
  comb_13_5 = Nat.choose 13 5 →
  comb_13_6 = Nat.choose 13 6 →
  2 * comb_13_5 + comb_13_6 = 4290 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [←h5, ←h6]
  sorry

end Thunderbolts_lineup_count_l549_549885


namespace student_average_always_less_l549_549238

theorem student_average_always_less (w x y z: ℝ) (hwx: w < x) (hxy: x < y) (hyz: y < z) :
  let A' := (w + x + y + z) / 4
  let B' := (2 * w + 2 * x + y + z) / 6
  B' < A' :=
by
  intro A' B'
  sorry

end student_average_always_less_l549_549238


namespace length_of_PT_l549_549045

-- Define the pentagon and its properties
structure Pentagon (P Q R S T : Type) :=
(QR : ℝ)
(RS : ℝ)
(ST : ℝ)
(angle_T : ℝ)
(angle_Q : ℝ)
(angle_R : ℝ)
(angle_S : ℝ)

def given_pentagon : Pentagon Point :=
{ QR := 3,
  RS := 3,
  ST := 3,
  angle_T := 90,
  angle_Q := 135,
  angle_R := 135,
  angle_S := 135 }

theorem length_of_PT : ∃ (c d : ℝ), (c + d = 8) ∧ 
  (PT_length : ℝ := c + 3 * real.sqrt d) := 
sorry

end length_of_PT_l549_549045


namespace train_crossing_time_l549_549968

-- Definitions based on the conditions
def train_length : ℝ := 110 -- in meters
def bridge_length : ℝ := 200 -- in meters
def train_speed_kph : ℝ := 60 -- in kilometers per hour

-- Conversion factor from kilometers per hour to meters per second
def kmph_to_mps (speed_kph : ℝ) : ℝ := speed_kph * (1000 / 3600)

-- Total distance the train travels to cross the bridge
def total_distance (train_length bridge_length : ℝ) : ℝ := train_length + bridge_length

-- Time taken to cross the bridge
def time_to_cross_bridge (total_distance speed_mps : ℝ) : ℝ := total_distance / speed_mps

-- The main statement to prove
theorem train_crossing_time :
  let speed_mps := kmph_to_mps train_speed_kph in
  let distance := total_distance train_length bridge_length in
  abs (time_to_cross_bridge distance speed_mps - 18.60) < 0.01 :=
by
  sorry

end train_crossing_time_l549_549968


namespace problem_f8_minus_f4_l549_549022

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 5) = f x
axiom f_at_1 : f 1 = 1
axiom f_at_2 : f 2 = 2

theorem problem_f8_minus_f4 : f 8 - f 4 = -1 :=
by sorry

end problem_f8_minus_f4_l549_549022


namespace mean_of_five_numbers_l549_549157

theorem mean_of_five_numbers (sum_of_numbers : ℚ) (number_of_elements : ℕ)
  (h_sum : sum_of_numbers = 3 / 4) (h_elements : number_of_elements = 5) :
  (sum_of_numbers / number_of_elements : ℚ) = 3 / 20 :=
by
  sorry

end mean_of_five_numbers_l549_549157


namespace product_gcd_lcm_l549_549308

-- Define the numbers
def a : ℕ := 24
def b : ℕ := 60

-- Define the gcd and lcm
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b

-- Statement to prove: the product of gcd and lcm of 24 and 60 equals 1440
theorem product_gcd_lcm : gcd_ab * lcm_ab = 1440 := by
  -- gcd_ab = 12
  -- lcm_ab = 120
  -- Thus, 12 * 120 = 1440
  sorry

end product_gcd_lcm_l549_549308


namespace certain_event_birthday_example_l549_549556
-- Import the necessary library

-- Define the problem with conditions
def certain_event_people_share_birthday (num_days : ℕ) (num_people : ℕ) : Prop :=
  num_people > num_days

-- Define a specific instance based on the given problem
theorem certain_event_birthday_example : certain_event_people_share_birthday 365 400 :=
by
  sorry

end certain_event_birthday_example_l549_549556


namespace arc_length_of_sector_l549_549615

theorem arc_length_of_sector (r : ℝ) (θ : ℝ) (h1 : r = 5) (h2 : θ = 120) : 
  let L := (θ / 360) * 2 * π * r 
  in L = (10 / 3) * π :=
by
  sorry

end arc_length_of_sector_l549_549615


namespace find_k_l549_549564

-- Defining the conditions
variables (m n k : ℝ)
axiom line_eq : ∀ (x y : ℝ), x = 2 * y + 5

-- Points (m, n) and (m + 1, n + k) are on the line x = 2y + 5
axiom point1_on_line : line_eq m n
axiom point2_on_line : line_eq (m + 1) (n + k)

-- The goal is to prove k = 1/2
theorem find_k : k = 1 / 2 :=
by sorry

end find_k_l549_549564


namespace password_of_treasure_chest_l549_549265

theorem password_of_treasure_chest :
  ∃ (password : ℕ), 
    -- Condition 1: The password is an eight-digit number.
    (10000000 ≤ password ∧ password < 100000000) ∧

    -- Condition 2: The password is a multiple of both 3 and 25.
    (password % 3 = 0 ∧ password % 25 = 0) ∧

    -- Condition 3: The password is between 20,000,000 and 30,000,000.
    (20000000 ≤ password ∧ password < 30000000) ∧

    -- Condition 4: The millions place and the hundred thousands place digits are the same.
    (let millions := (password / 1000000) % 10 in
     let hundred_thousands := (password / 100000) % 10 in
     millions = hundred_thousands) ∧

    -- Condition 5: The hundreds digit is 2 less than the ten thousands digit.
    (let hundreds := (password / 100) % 10 in
     let ten_thousands := (password / 10000) % 10 in
     hundreds = ten_thousands - 2) ∧
    
    -- Condition 6: The digits in the hundred thousands, ten thousands, and thousands places form a three-digit number
    -- which, when divided by the two-digit number formed by the digits in the ten millions and millions places, gives a quotient of 25.
    (let hundred_thousands := (password / 100000) % 10 in
     let ten_thousands := (password / 10000) % 10 in
     let thousands := (password / 1000) % 10 in
     let three_digit_number := 100 * hundred_thousands + 10 * ten_thousands + thousands in
     let ten_millions := (password / 10000000) % 10 in
     let millions := (password / 1000000) % 10 in
     let two_digit_number := 10 * ten_millions + millions in
     three_digit_number / two_digit_number = 25 ∧ three_digit_number % two_digit_number = 0) :=

    password = 26650350 := sorry

end password_of_treasure_chest_l549_549265


namespace logarithm_identity_l549_549287

theorem logarithm_identity (x y: ℝ) (hx: x > 0) (hy: y > 0) (hxy: x ≠ 1) (hyy: y ≠ 1):
  log (y^8) (x^2) * log (x^7) (y^3) * log (y^5) (x^4) * log (x^4) (y^5) * log (y^3) (x^7) * log (x^2) (y^8) = (28 / 3) * log y x := 
  sorry

end logarithm_identity_l549_549287


namespace slope_of_tangent_line_l549_549084

noncomputable def f : ℝ → ℝ := sorry
variable (Δx : ℝ)

-- Hypotheses
axiom differentiable_f : differentiable ℝ f
axiom limit_condition : tendsto (λ Δx, (f 1 - f (1 - 2 * Δx)) / Δx) (𝓝 0) (𝓝 (-1))

-- Statement to prove
theorem slope_of_tangent_line :
  deriv f 1 = -1 / 2 :=
begin
  sorry
end

end slope_of_tangent_line_l549_549084


namespace avg_high_scores_decrease_avg_low_scores_increase_l549_549415

theorem avg_high_scores_decrease (n : ℕ) (a : ℕ → ℕ) (h : ∀ i j, i ≤ j → a i ≤ a j) (hne : ∃ i j, i ≠ j ∧ a i ≠ a j) (m : ℕ) (hm : 1 ≤ m ∧ m < n) :
  (∑ i in finset.range n, a i) / n > (∑ i in finset.range m, a i) / m := 
sorry

theorem avg_low_scores_increase (n : ℕ) (a : ℕ → ℕ) (h : ∀ i j, i ≤ j → a i ≤ a j) (hne : ∃ i j, i ≠ j ∧ a i ≠ a j) (m : ℕ) (hm : 1 ≤ m ∧ m < n) :
  (∑ i in finset.range n, a i) / n < (∑ i in (finset.range (n + 1)).filter (λ x, x ≥ m), a i) / (n - m) :=
sorry

end avg_high_scores_decrease_avg_low_scores_increase_l549_549415


namespace flag_count_l549_549601

-- Definitions of colors as a datatype
inductive Color
| red : Color
| white : Color
| blue : Color
| green : Color
| yellow : Color

open Color

-- Total number of distinct flags possible
theorem flag_count : 
  (∃ m : Color, 
   (∃ t : Color, 
    (t ≠ m ∧ 
     ∃ b : Color, 
     (b ≠ m ∧ b ≠ red ∧ b ≠ blue)))) ∧ 
  (5 * 4 * 2 = 40) := 
  sorry

end flag_count_l549_549601


namespace find_k_l549_549025

theorem find_k (k m : ℝ) : (m^2 - 8*m) ∣ (m^3 - k*m^2 - 24*m + 16) → k = 8 := by
  sorry

end find_k_l549_549025


namespace find_f_iterate_l549_549833

noncomputable def f (x p q : ℝ) : ℝ := x^2 + p*x + q

theorem find_f_iterate (p q : ℝ)
  (h1 : ∀ x, 3 ≤ x ∧ x ≤ 5 → |f x p q| ≤ 1 / 2) :
  let x := (7 + Real.sqrt 15) / 2 in
  (List.iterate (f · p q) 2017 x) = (7 - Real.sqrt 15) / 2 := 
by
  sorry

end find_f_iterate_l549_549833


namespace equilateral_triangle_dot_product_sum_l549_549413

variable {V : Type} [InnerProductSpace ℝ V]

noncomputable def vector_dot_product_sum (a b c : V) : ℝ :=
  a ⬝ b + b ⬝ c + c ⬝ a

theorem equilateral_triangle_dot_product_sum
  (a b c : V)
  (side_len : ℝ)
  (h1 : ∥a∥ = side_len)
  (h2 : ∥b∥ = side_len)
  (h3 : ∥c∥ = side_len)
  (triangle_condition : ∀ u v : V, u ⬝ v = side_len^2 * (real.cos (2 * real.pi / 3))) :
  vector_dot_product_sum a b c = - (3 / 2) :=
  sorry

end equilateral_triangle_dot_product_sum_l549_549413


namespace mode_of_data_set_l549_549904

def data_set : List Int := [-1, 0, 2, -1, 3]

def mode (l : List Int) : Int :=
if h : l = [] then 0 else (l.groupBy (λ x => x).length).maxBy (λ x => x.2).1

theorem mode_of_data_set : mode data_set = -1 := by
  sorry

end mode_of_data_set_l549_549904


namespace chord_length_correct_l549_549595

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

def parametric_line (t : ℝ) : ℝ × ℝ := (2 - (1/2) * t, -1 + (1/2) * t)

noncomputable def chord_length : ℝ := 2 * Real.sqrt (4 - (Real.sqrt 2 / 2)^2)

theorem chord_length_correct :
  chord_length = Real.sqrt 14 :=
sorry

end chord_length_correct_l549_549595


namespace sum_of_digits_14_l549_549412

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def has_same_digits (n m : ℕ) : Prop :=
  multiset.of_nat_digits n = multiset.of_nat_digits m

theorem sum_of_digits_14 (paid change : ℕ) (h1 : is_three_digit_number paid) (h2 : change = 1000 - paid) (h3 : is_three_digit_number change) (h4 : has_same_digits paid change) : 
  (nat.digits 10 paid).sum = 14 :=
sorry

end sum_of_digits_14_l549_549412


namespace circle_equation_locus_of_center_fixed_point_and_ratio_l549_549803

-- Part (1)
theorem circle_equation (a : ℝ) (h : a ≠ 0) :
  ∀ x y, 
  (y = (x^2 / 2) + (a / 2) * x - a^2) →
  (circle_eq : x^2 + y^2 + a * x + (a^2 - 2) * y - 2 * a^2 = 0) := sorry

-- Part (2)
theorem locus_of_center (a : ℝ) (h : a ≠ 0) :
  ∀ x y,
  (center_eq : (x, y) = (-a / 2, 1 / 2 * (2 - a^2))) →
  y = 1 - 2 * x^2 := sorry

-- Part (3)
theorem fixed_point_and_ratio (a : ℝ) (M : ℝ × ℝ) (ha : a = 0) (hm : M = (0, 3)) :
  ∃ N : ℝ × ℝ, N = (0, 3 / 2) ∧ ∀ P, (P ∈ circle_eq) →
  |P - N| / |P - M| = 1 / 2 := sorry

end circle_equation_locus_of_center_fixed_point_and_ratio_l549_549803


namespace max_B_at_50_l549_549291

def binomial (n k : ℕ) : ℕ := nat.choose n k

noncomputable def B (k : ℕ) : ℝ :=
  binomial 500 k * (0.1 : ℝ)^k

theorem max_B_at_50 :
  ∀ k : ℕ, (k ≤ 500) → (B k ≤ B 50) := sorry

end max_B_at_50_l549_549291


namespace no_solution_set_1_2_4_l549_549698

theorem no_solution_set_1_2_4 
  (f : ℝ → ℝ) 
  (hf : ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c)
  (t : ℝ) : ¬ ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f (|x1 - t|) = 0 ∧ f (|x2 - t|) = 0 ∧ f (|x3 - t|) = 0 ∧ (x1 = 1 ∧ x2 = 2 ∧ x3 = 4) := 
sorry

end no_solution_set_1_2_4_l549_549698


namespace log_sum_of_a101_to_a110_l549_549341

-- Define the sequence a_n as satisfying the given logarithmic condition
def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → log 2 (a (n + 1)) = 1 + log 2 (a n)

-- Define the sum condition of the first 10 terms of the sequence
def sequence_sum_condition (a : ℕ → ℝ) : Prop :=
  (finset.range 10).sum (λ n, a (n + 1)) = 1

-- The main theorem to prove
theorem log_sum_of_a101_to_a110 (a : ℕ → ℝ) (h_cond : sequence_condition a) (h_sum : sequence_sum_condition a) :
  log 2 ((finset.range 10).sum (λ n, a (101 + n))) = 10 := 
  sorry

end log_sum_of_a101_to_a110_l549_549341


namespace product_gcd_lcm_24_60_l549_549320

theorem product_gcd_lcm_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end product_gcd_lcm_24_60_l549_549320


namespace mandy_staircase_l549_549103

def number_of_steps (T : ℕ) : ℕ := sorry

theorem mandy_staircase (hT : 380) : number_of_steps hT = 17 := 
sorry

end mandy_staircase_l549_549103


namespace quadrilateral_side_length_l549_549993

theorem quadrilateral_side_length (r a b c x : ℝ) 
  (h1 : r = 100 * Real.sqrt 5) 
  (h2 : a = 200) 
  (h3 : b = 200) 
  (h4 : c = 200) 
  (h5 : sqrt ((2*r)^2 - a^2) = 200) :
  x = 200 :=
begin
  sorry
end

end quadrilateral_side_length_l549_549993


namespace sum_max_min_eq_four_l549_549740

def f (x : ℝ) : ℝ := (x ^ 2 - 2 * x) * Real.sin (x - 1) + x + 1

theorem sum_max_min_eq_four : 
  let I := Set.Icc (-1 : ℝ) (3 : ℝ),
      M := Real.Sup (f '' I),
      m := Real.Inf (f '' I)
  in M + m = 4 :=
sorry

end sum_max_min_eq_four_l549_549740


namespace infinite_nat_solutions_l549_549839

theorem infinite_nat_solutions (n : ℕ) (a : Fin (n + 1) → ℕ)
  (hcoprime : ∀ i : Fin n, Nat.coprime (a i) (a n)) :
  ∃ (x : Fin (n + 1) → ℕ), 
  (x i) ^ (a i) + (x (i + 1)) ^ (a (i + 1)) + ... + (x n) ^ (a n) = (x (n + 1)) ^ (a (n + 1)) := 
sorry

end infinite_nat_solutions_l549_549839


namespace expected_candies_when_game_ends_l549_549389

noncomputable def expectedCandies : ℚ :=
  -- let X be the expected number of candies Heesu will have when the game ends
  let X := (27 : ℚ) / 31
  in X

theorem expected_candies_when_game_ends : expectedCandies = 27 / 31 :=
by
  -- Proof of the expected number of candies Heesu will have when the game ends
  rw [expectedCandies]
  sorry

end expected_candies_when_game_ends_l549_549389


namespace unique_angle_of_perpendicular_bisectors_l549_549059

namespace TriangleProblem

-- Definition of an acute-angled scalene triangle ABC
variables {A B C P : Type} [angle_ABC : Triangle A B C]

-- Midpoints D, E, F of sides AB, BC, and CA respectively
variables {D : midpoint A B} 
variables {E : midpoint B C} 
variables {F : midpoint C A}

-- Collinearity condition and point P
variables (collinear_AP_PC : collinear P (segment A P)) (collinear_CP_BB : collinear P (segment C P))

-- The desired result: showing perpendicular bisectors implies collinearity, and deducing the unique angle
theorem unique_angle_of_perpendicular_bisectors : 
  ∃! (P : point), collinear P (segment A P) ∧ collinear P (segment B P) ∧ collinear P (segment C P) :=
begin 
  -- Since the Lean 4 code requires the statement, we add the proof as sorry
  sorry
end

end TriangleProblem

end unique_angle_of_perpendicular_bisectors_l549_549059


namespace exists_point_with_sum_distances_at_least_n_l549_549058

theorem exists_point_with_sum_distances_at_least_n (n : ℕ) (P : Fin n → {q : ℝ × ℝ // (q.1)^2 + (q.2)^2 ≤ 1}) :
  ∃ (q : {r : ℝ × ℝ // (r.1)^2 + (r.2)^2 ≤ 1}), (Finset.univ.sum (λ i, dist (q : ℝ × ℝ) (P i : ℝ × ℝ))) ≥ n := 
sorry

end exists_point_with_sum_distances_at_least_n_l549_549058


namespace possible_to_have_integer_pairwise_distances_l549_549872

noncomputable def regular_hexagon_integer_distances : Prop := 
  ∃ (s : ℝ) (m_A m_B m_C m_D m_E m_F : ℝ → ℝ), 
    (∀ (x : ℝ), m_A x ≠ m_B x ∧ m_B x ≠ m_C x ∧ m_C x ≠ m_D x ∧ m_D x ≠ m_E x ∧ m_E x ≠ m_F x) ∧
    (∀ (d : ℝ), 
      (∃ (d_AB d_AC d_AD d_AE d_AF d_BC d_BD d_BE d_BF d_CD d_CE d_CF d_DE d_DF d_EF : ℝ),
      d_AB = 1 ∧ d_AC = 1 ∧ d_AD = ? ∧ d_AE = ? ∧ d_AF = ? ∧ d_BC = 1 ∧ d_BD = ? ∧ d_BE = ? 
      ∧ d_BF = ? ∧ d_CD = ? ∧ d_CE = ? ∧ d_CF = 2 ∧ d_DE = 1 ∧ d_DF = 1 ∧ d_EF = 1 ∧ 
      ∀ (u v : ℝ → ℝ), u ≠ v → ∃ (d : ℝ), d = dist(u, v)))

theorem possible_to_have_integer_pairwise_distances : regular_hexagon_integer_distances :=
  sorry

end possible_to_have_integer_pairwise_distances_l549_549872


namespace area_of_triangle_ABC_l549_549414

-- Given conditions
variables {A B C : ℝ}
variables (angle_A angle_C : ℝ)
variables (AC : ℝ)
variables (is_right_triangle : A ≠ B ∧ B ≠ C ∧ A ≠ C)

-- Specific conditions for the problem
def is_isosceles_right_triangle := angle_A = angle_C ∧ AC = 8 * Real.sqrt 2

-- The proof problem statement
theorem area_of_triangle_ABC
  (h1 : is_isosceles_right_triangle angle_A angle_C AC)
  (h2 : angle_A = 45) (h3 : angle_C = 45)
  (h4 : is_right_triangle) : 
  1/2 * (AC / Real.sqrt 2) * (AC / Real.sqrt 2) = 32 :=
  sorry

end area_of_triangle_ABC_l549_549414


namespace exists_m_n_l549_549434

theorem exists_m_n (p : ℕ) (hp : p > 10) [hp_prime : Fact (Nat.Prime p)] :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m + n < p ∧ (5^m * 7^n - 1) % p = 0 :=
sorry

end exists_m_n_l549_549434


namespace intersection_union_A_B_subset_B_A_l549_549752

section
variable (A B : Set ℝ) (m : ℝ)

-- Definitions from the problem
def A := {x : ℝ | x^2 - 8 * x + 12 ≤ 0}
def B := {x : ℝ | 5 - 2 * m ≤ x ∧ x ≤ m + 1}

-- Theorem statements
theorem intersection_union_A_B (h : m = 3) :
  A ∩ B = Set.Icc (2 : ℝ) (4 : ℝ) ∧ A ∪ B = Set.Icc (-1 : ℝ) (6 : ℝ) := 
sorry

theorem subset_B_A (h1 : ∀ x, x ∈ B -> x ∈ A):
  m ≤ (3 / 2 : ℝ) :=
sorry
end

end intersection_union_A_B_subset_B_A_l549_549752


namespace possible_values_of_AC_l549_549489

theorem possible_values_of_AC (AB CD AC : ℝ) (m n : ℝ) (h1 : AB = 16) (h2 : CD = 4)
  (h3 : Set.Ioo m n = {x : ℝ | 4 < x ∧ x < 16}) : m + n = 20 :=
by
  sorry

end possible_values_of_AC_l549_549489
