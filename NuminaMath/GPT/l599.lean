import Mathlib
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Quadratic
import Mathlib.Algebra.Quadratic.Derivative
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.Calculus.Lhopital
import Mathlib.Analysis.Calculus.Limits
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fibonacci
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial.TrailingZeroes
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Real.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.MeasureTheory.Integral.SetIntegral
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityTheory
import Mathlib.Probability.Theory
import Mathlib.Tactic
import Real
import Set
import data.fintype.basic

namespace vector_problem_l599_599820

noncomputable def t_value : ℝ :=
  (-5 - Real.sqrt 13) / 2

theorem vector_problem 
  (t : ℝ)
  (a : ℝ × ℝ := (1, 1))
  (b : ℝ × ℝ := (2, t))
  (h : Real.sqrt ((1 - 2)^2 + (1 - t)^2) = (1 * 2 + 1 * t)) :
  t = t_value := 
sorry

end vector_problem_l599_599820


namespace hill_height_l599_599869

theorem hill_height (h : ℝ) (time_up : ℝ := h / 9) (time_down : ℝ := h / 12) (total_time : ℝ := time_up + time_down) (time_cond : total_time = 175) : h = 900 :=
by 
  sorry

end hill_height_l599_599869


namespace average_salary_increase_l599_599597

theorem average_salary_increase 
  (average_salary : ℕ) (manager_salary : ℕ)
  (n : ℕ) (initial_count : ℕ) (new_count : ℕ) (initial_average : ℕ)
  (total_salary : ℕ) (new_total_salary : ℕ) (new_average : ℕ)
  (salary_increase : ℕ) :
  initial_average = 1500 →
  manager_salary = 3600 →
  initial_count = 20 →
  new_count = initial_count + 1 →
  total_salary = initial_count * initial_average →
  new_total_salary = total_salary + manager_salary →
  new_average = new_total_salary / new_count →
  salary_increase = new_average - initial_average →
  salary_increase = 100 := by
  sorry

end average_salary_increase_l599_599597


namespace width_of_rectangular_field_l599_599227

theorem width_of_rectangular_field
  (L W : ℝ)
  (h1 : L = (7/5) * W)
  (h2 : 2 * L + 2 * W = 384) :
  W = 80 :=
by
  sorry

end width_of_rectangular_field_l599_599227


namespace other_root_of_quadratic_eq_l599_599123

theorem other_root_of_quadratic_eq (m : ℝ) (t : ℝ) (h1 : (polynomial.X ^ 2 + polynomial.C m * polynomial.X + polynomial.C (-6)).roots = {2, t}) : t = -3 :=
sorry

end other_root_of_quadratic_eq_l599_599123


namespace factorial_three_eq_six_l599_599721

-- Definition of factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Statement to prove that 3! = 6
theorem factorial_three_eq_six : factorial 3 = 6 :=
by sorry

end factorial_three_eq_six_l599_599721


namespace equation_of_line_l_l599_599789

-- Define the constants of the hyperbola
def a_sq : ℝ := 12
def b_sq : ℝ := 4
def a : ℝ := real.sqrt a_sq
def b : ℝ := real.sqrt b_sq

-- Define the coordinates of the right focus
def right_focus : ℝ × ℝ := (4, 0)

-- Define the slope of the asymptote in the first quadrant
def asymptote_slope : ℝ := b / a

-- Main theorem stating the equation of the line l
theorem equation_of_line_l : 
  ∀ (x y : ℝ), line_through_point_slope (right_focus) (-real.sqrt 3) (x, y) 
  → y = -real.sqrt 3 * x + 4 * real.sqrt 3 :=
by {
  sorry
}

def line_through_point_slope (point : ℝ × ℝ) (slope : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 - point.2 = slope * (p.1 - point.1)

end equation_of_line_l_l599_599789


namespace systematic_sampling_interval_l599_599255

theorem systematic_sampling_interval 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (h_total : total_students = 1000) 
  (h_sample : sample_size = 50) : 
  (total_students / sample_size) = 20 :=
by
  rw [h_total, h_sample]
  norm_num

end systematic_sampling_interval_l599_599255


namespace problem_lean_statement_l599_599678

-- Definitions given in the conditions
def C1_parametric (α : ℝ) : ℝ × ℝ :=
  let x := sqrt 2 * sin (α + π / 4)
  let y := sin (2 * α) + 1
  (x, y)

def C2_polar (θ ρ : ℝ) : Prop :=
  ρ^2 = 4 * ρ * sin θ - 3

-- Cartesian equation deduced for C1
def C1_cartesian (x y : ℝ) : Prop :=
  y = x^2

-- Cartesian equation deduced for C2
def C2_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * y + 3 = 0

-- Distance between points on C1 and C2
def min_distance_C1_C2 (x0 : ℝ) : ℝ :=
  abs ((sqrt 7) / 2 - 1)

theorem problem_lean_statement :
  (∀ α, ∃ x y, (C1_parametric α = (x, y) ∧ C1_cartesian x y)) ∧
  (∀ θ ρ, C2_polar θ ρ → ∃ x y, ρ = sqrt (x^2 + y^2) ∧ y = ρ * sin θ ∧ C2_cartesian x y) ∧
  (∀ (P : ℝ × ℝ) (C1 : ∃ x y, C1_cartesian x y ∧ P = (x, y)) (C2 : ∃ x y, C2_cartesian x y),
    let (x0, y0) := P in
    y0 = x0^2 →
    let dist := min_distance_C1_C2 x0 in
    dist = (sqrt 7) / 2 - 1) :=
by
  sorry

end problem_lean_statement_l599_599678


namespace book_vs_necklace_price_difference_l599_599573

-- Problem-specific definitions and conditions
def necklace_price : ℕ := 34
def limit_price : ℕ := 70
def overspent : ℕ := 3
def total_spent : ℕ := limit_price + overspent
def book_price : ℕ := total_spent - necklace_price

-- Lean statement to prove the correct answer
theorem book_vs_necklace_price_difference :
  book_price - necklace_price = 5 := by
  sorry

end book_vs_necklace_price_difference_l599_599573


namespace fraction_incorrect_like_music_l599_599713

-- Define the conditions as given in the problem
def total_students : ℕ := 100
def like_music_percentage : ℝ := 0.7
def dislike_music_percentage : ℝ := 1 - like_music_percentage

def correct_like_percentage : ℝ := 0.75
def incorrect_like_percentage : ℝ := 1 - correct_like_percentage

def correct_dislike_percentage : ℝ := 0.85
def incorrect_dislike_percentage : ℝ := 1 - correct_dislike_percentage

-- The number of students liking music
def like_music_students : ℝ := total_students * like_music_percentage
-- The number of students disliking music
def dislike_music_students : ℝ := total_students * dislike_music_percentage

-- The number of students who correctly say they like music
def correct_like_music_say : ℝ := like_music_students * correct_like_percentage
-- The number of students who incorrectly say they dislike music
def incorrect_dislike_music_say : ℝ := like_music_students * incorrect_like_percentage

-- The number of students who correctly say they dislike music
def correct_dislike_music_say : ℝ := dislike_music_students * correct_dislike_percentage
-- The number of students who incorrectly say they like music
def incorrect_like_music_say : ℝ := dislike_music_students * incorrect_dislike_percentage

-- The total number of students who say they like music
def total_say_like_music : ℝ := correct_like_music_say + incorrect_like_music_say

-- The final theorem we want to prove
theorem fraction_incorrect_like_music : ((incorrect_like_music_say : ℝ) / total_say_like_music) = (5 / 58) :=
by
  -- here we would provide the proof, but for now, we use sorry
  sorry

end fraction_incorrect_like_music_l599_599713


namespace imaginary_part_conjugate_l599_599766

noncomputable def condition (z : ℂ) : Prop := 
  z * (3 + Complex.i) = 3 + Complex.i^2023

theorem imaginary_part_conjugate {z : ℂ} (h : condition z) : 
  Complex.imag (Complex.conj z) = 3 / 5 :=
sorry

end imaginary_part_conjugate_l599_599766


namespace diamond_example_l599_599019

def diamond (a b : ℝ) : ℝ := a - a / b

theorem diamond_example : diamond 15 5 = 12 := by
  sorry

end diamond_example_l599_599019


namespace jeannie_return_pace_l599_599151

-- Define the initial distance to the mountain, pace to the mountain, and total hike time
def distance_to_mountain : ℝ := 12
def pace_to_mountain : ℝ := 4
def total_hike_time : ℝ := 5

-- Define the time to Mount Overlook using the distance and pace
def time_to_mountain : ℝ := distance_to_mountain / pace_to_mountain

-- Define the time for the return trip
def time_for_return_trip : ℝ := total_hike_time - time_to_mountain

-- Define the pace for the return trip
def return_pace : ℝ := distance_to_mountain / time_for_return_trip

-- State the theorem to be proven
theorem jeannie_return_pace : return_pace = 6 := by
  sorry

end jeannie_return_pace_l599_599151


namespace distance_P_to_plane_CFH_eq_l599_599958

noncomputable def distance_from_point_to_plane : ℝ :=
  let P := (1:ℝ, 2 / 3, 1 / 3) -- The coordinates of point P
  let C := (0:ℝ, 1, 0)         -- The coordinates of point C
  let F := (1:ℝ, 1, 1)         -- The coordinates of point F
  let H := (0:ℝ, 0, 1)         -- The coordinates of point H
  
  -- Plane equation from points C, F, H: x - y - z + 1 = 0
  let a := 1
  let b := -1
  let c := -1
  let d := 1
  
  -- Distance formula from point (x₁, y₁, z₁) to plane ax + by + cz + d = 0
  let x₁ := P.1
  let y₁ := P.2
  let z₁ := P.3
  
  abs (a * x₁ + b * y₁ + c * z₁ + d) / Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2)

theorem distance_P_to_plane_CFH_eq : distance_from_point_to_plane = Math.sqrt(3) / 3 := by
  sorry

end distance_P_to_plane_CFH_eq_l599_599958


namespace integer_solution_l599_599641

theorem integer_solution (n : ℤ) (hn : 0 ≤ n ∧ n < 200) (h : 150 * n ≡ 110 [MOD 199]) : n = 157 :=
  sorry

end integer_solution_l599_599641


namespace mitchell_total_pages_read_l599_599206

def pages_per_chapter : ℕ := 40
def chapters_read_before : ℕ := 10
def pages_read_11th_before : ℕ := 20
def chapters_read_after : ℕ := 2

def total_pages_read := 
  pages_per_chapter * chapters_read_before + pages_read_11th_before + pages_per_chapter * chapters_read_after

theorem mitchell_total_pages_read : total_pages_read = 500 := by
  sorry

end mitchell_total_pages_read_l599_599206


namespace geom_seq_property_l599_599418

noncomputable def a_n : ℕ → ℝ := sorry  -- The definition of the geometric sequence

theorem geom_seq_property (a_n : ℕ → ℝ) (h : a_n 6 + a_n 8 = 4) :
  a_n 8 * (a_n 4 + 2 * a_n 6 + a_n 8) = 16 := by
sorry

end geom_seq_property_l599_599418


namespace sum_of_possible_N_l599_599612

theorem sum_of_possible_N 
  (a b c N : ℕ) (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = 2 * (a + b))
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∑ (ab_pairs : ℕ × ℕ) in ({(1, 16), (2, 8), (4, 4)} : Finset (ℕ × ℕ)), (16 * (ab_pairs.1 + ab_pairs.2)) = 560 := 
by
  sorry

end sum_of_possible_N_l599_599612


namespace max_ab_plus_bc_plus_cd_plus_da_l599_599234

theorem max_ab_plus_bc_plus_cd_plus_da : 
  ∃ (a b c d : ℕ), {a, b, c, d} = {2, 3, 5, 6} ∧ 
  (∀ (a' b' c' d' : ℕ), {a', b', c', d'} = {2, 3, 5, 6} → ab + bc + cd + da ≤ a' * b' + b' * c' + c' * d' + d' * a') ∧
  (ab + bc + cd + da = 63) :=
by
  sorry

end max_ab_plus_bc_plus_cd_plus_da_l599_599234


namespace inequality_proof_l599_599076

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x^2 + y^2 + z^2 = 2 * (x * y + y * z + z * x)) :
  (x + y + z) / 3 ≥ (2 * x * y * z)^(1/3 : ℝ) :=
by
  sorry

end inequality_proof_l599_599076


namespace find_2x_plus_y_l599_599436

theorem find_2x_plus_y (x y : ℝ) 
  (h1 : (x + y) / 3 = 5 / 3) 
  (h2 : x + 2*y = 8) : 
  2*x + y = 7 :=
sorry

end find_2x_plus_y_l599_599436


namespace quadratic_coefficients_l599_599329

theorem quadratic_coefficients : 
  ∀ (x : ℝ), (3 * x = x^2 - 2) → 
  (1 = (0:ℝ) * x^2 + -3 * x + -2) := 
begin
  intros x h,
  sorry
end

end quadratic_coefficients_l599_599329


namespace initial_action_figures_correct_l599_599522

def initial_action_figures (x : ℕ) : Prop :=
  x + 11 - 10 = 8

theorem initial_action_figures_correct :
  ∃ x : ℕ, initial_action_figures x ∧ x = 7 :=
by
  sorry

end initial_action_figures_correct_l599_599522


namespace arithmetic_sequence_sum_l599_599137

open Real

noncomputable def a_n : ℕ → ℝ := sorry -- to represent the arithmetic sequence

theorem arithmetic_sequence_sum :
  (∃ d : ℝ, ∀ (n : ℕ), a_n n = a_n 1 + (n - 1) * d) ∧
  (∃ a1 a2011 : ℝ, (a_n 1 = a1) ∧ (a_n 2011 = a2011) ∧ (a1 ^ 2 - 10 * a1 + 16 = 0) ∧ (a2011 ^ 2 - 10 * a2011 + 16 = 0)) →
  a_n 2 + a_n 1006 + a_n 2010 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l599_599137


namespace AC_bisects_BLD_l599_599530

-- Definitions and conditions
variables {A B C D L N : Point}

-- Hypotheses
axiom cyclic_quadrilateral (ABCD : CyclicQuadrilateral A B C D)
axiom midpoint_L : Midpoint L A C
axiom midpoint_N : Midpoint N B D
axiom BD_bisects_ANC : ∠ B D N = ∠ N D C

-- Proposition to prove
theorem AC_bisects_BLD (h1 : cyclic_quadrilateral ABCD) 
  (h2 : midpoint_L) 
  (h3 : midpoint_N) 
  (h4 : BD_bisects_ANC) : 
  ∠ B L A = ∠ L D C :=
sorry

end AC_bisects_BLD_l599_599530


namespace length_DB_l599_599511

-- Conditions
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (angle_ABC angle_ADB : ℝ) (AC AD : ℝ)

-- Defining the natures of angles and distances
def is_right_angle (angle : ℝ) : Prop := angle = π / 2
def length_AC := AC = 15
def length_AD := AD = 9

-- The equivalent proof problem in Lean 4
theorem length_DB (h1: is_right_angle angle_ABC) (h2: is_right_angle angle_ADB) 
                 (h3: length_AC) (h4: length_AD) : 
                 ∃ DB, DB = 3 * Real.sqrt 6 :=
sorry

-- Here we declare a type for angles and lengths, define the assumptions related to given angles 
-- being right angles, initialize lengths of AC and AD and finally state the theorem to 
-- prove the length of DB.

end length_DB_l599_599511


namespace probability_heads_and_three_coin_die_is_correct_l599_599828

/-- I flip two fair coins and roll a regular six-sided die. What is the probability that both coins will show heads and the die will show a 3? -/
def probability_of_heads_and_three_coin_die : ℚ :=
  let total_coin_outcomes := 2 * 2 -- Each coin has 2 outcomes: heads or tails
  let total_die_outcomes := 6 -- Die has 6 outcomes
  let total_outcomes := total_coin_outcomes * total_die_outcomes
  let favorable_outcomes := 1 -- Only one favorable outcome: HH and 3
  favorable_outcomes / total_outcomes

theorem probability_heads_and_three_coin_die_is_correct :
  probability_of_heads_and_three_coin_die = 1 / 24 :=
by
  unfold probability_of_heads_and_three_coin_die
  norm_num
  sorry

end probability_heads_and_three_coin_die_is_correct_l599_599828


namespace chris_earns_highest_difference_l599_599005

-- Definitions for the given conditions
def asking_price : ℕ := 5200
def inspection_cost : ℕ := asking_price / 10
def headlights_cost : ℕ := 80
def tires_cost : ℕ := 3 * headlights_cost
def battery_cost : ℕ := 2 * tires_cost
def security_system_cost : ℕ := asking_price / 8
def discount_rate : ℚ := 0.15
def discounted_price : ℚ := asking_price * (1 - discount_rate)
def paint_job_cost : ℕ := (discounted_price / 5).to_nat
def reduced_price_rate : ℚ := 0.95
def reduced_price : ℕ := (asking_price * reduced_price_rate).to_nat
def stereo_cost : ℕ := 200
def insurance_rate : ℚ := 0.2
def insurance_cost : ℕ := (reduced_price * insurance_rate).to_nat

-- Calculations of the amounts from each buyer
def net_amount_first_buyer : ℕ := asking_price - inspection_cost
def net_amount_second_buyer : ℕ := 
  asking_price - (headlights_cost + tires_cost + battery_cost + security_system_cost)
def net_amount_third_buyer : ℕ := 
  (discounted_price.to_nat - paint_job_cost)
def net_amount_fourth_buyer : ℕ := 
  reduced_price - (stereo_cost + insurance_cost)

-- Calculation of the difference between the highest and lowest offers
def highest_offer : ℕ := max (max net_amount_first_buyer net_amount_second_buyer) 
  (max net_amount_third_buyer net_amount_fourth_buyer)
def lowest_offer : ℕ := min (min net_amount_first_buyer net_amount_second_buyer) 
  (min net_amount_third_buyer net_amount_fourth_buyer)
def difference : ℕ := highest_offer - lowest_offer

-- Proof statement
theorem chris_earns_highest_difference : difference = 1144 := by
  sorry

end chris_earns_highest_difference_l599_599005


namespace problem1_problem2_l599_599759

-- Problem 1 Statement
theorem problem1 (a b : ℝ) : 
  (Real.sqrt ((Real.pi - 4) ^ 2)) + ((-1 / 8) ^ (-1 / 3)) * (0.1 ^ (-1)) * ((a^3 * b^(-3))^ (1/2)) / ((1 / 25) ^ (-1 / 2) * ((Real.sqrt (a * b^(-1))) ^ 3)) = -Real.pi :=
  sorry

-- Problem 2 Statement
theorem problem2 : 
  ((1 / 3) ^ Real.logb 3 4) + Real.log10 20 - Real.log10 2 - (Real.logb 4 3 * Real.logb 9 32) - ((Real.sqrt 2 - 1)^Real.logb 10 1) = -1 :=
  sorry

end problem1_problem2_l599_599759


namespace percent_round_trip_tickets_l599_599187

-- Define the main variables
variables (P R : ℝ)

-- Define the conditions based on the problem statement
def condition1 : Prop := 0.3 * P = 0.3 * R
 
-- State the theorem to prove
theorem percent_round_trip_tickets (h1 : condition1 P R) : R / P * 100 = 30 := by sorry

end percent_round_trip_tickets_l599_599187


namespace sixth_row_count_ones_l599_599909

-- Define a function to generate the nth row of Pascal's Triangle
def pascal_row : ℕ → list ℕ
| 0 := [1]
| n := let prev := pascal_row (n - 1) in
       (0 :: prev).zipWith (+) (prev ++ [0])

-- Define a function to modify Pascal's Triangle according to the conditions
def modified_pascal_row (n : ℕ) : list ℕ :=
  (pascal_row n).map (λ x, if x % 2 = 1 then 1 else 0)

-- Theorem stating the number of 1s in the 6th row
theorem sixth_row_count_ones : (modified_pascal_row 5).count 1 = 4 :=
by
  -- Proof would go here, but we just state the result with sorry
  sorry

end sixth_row_count_ones_l599_599909


namespace num_lines_touching_parabola_once_l599_599045

theorem num_lines_touching_parabola_once :
  let P : ℝ×ℝ := (-1, 0)
  let parabola := λ x : ℝ, x^2
  ∃! (l : ℝ → ℝ), l (P.1) = P.2 ∧ (∃ x, (l x)^2 = x) :=
begin
  sorry
end

end num_lines_touching_parabola_once_l599_599045


namespace area_enclosed_l599_599037

theorem area_enclosed (x y : ℝ) (h : |x - 75| + |y| = |x / 3|) :
  let A := (112.5 - 56.25) * 25 / 2 in
  A = 703.125 :=
sorry

end area_enclosed_l599_599037


namespace find_y_for_orthogonality_l599_599752

open real

def vector3 : Type := ℝ × ℝ × ℝ

def dot_product (u v : vector3) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_y_for_orthogonality :
  ∀ (y : ℝ),
  let u : vector3 := (1, -4, -5) in
  let v : vector3 := (-3, y, 2) in
  dot_product u v = 0 → y = -13 / 4 :=
begin
  sorry
end

end find_y_for_orthogonality_l599_599752


namespace find_solutions_l599_599385

def binomial_coefficient (n m : ℕ) : ℕ :=
  n.factorial / (m.factorial * (n - m).factorial)

def is_solution (n m : ℕ) : Prop :=
  binomial_coefficient n (m - 1) = binomial_coefficient (n - 1) m

theorem find_solutions :
  ∀ k ∈ ℕ, is_solution (Fibonacci (2 * k) * Fibonacci (2 * k + 1)) (Fibonacci (2 * k) * Fibonacci (2 * k - 1)) :=
by 
  sorry

end find_solutions_l599_599385


namespace geometric_series_common_ratio_l599_599754

theorem geometric_series_common_ratio :
  ∀ (a1 a2 a3 : ℚ), 
    a1 = 2 / 7 ∧ 
    a2 = 10 / 49 ∧ 
    a3 = 50 / 343 → 
    ∃ r : ℚ, r = 5 / 7 :=
by
  intros a1 a2 a3 h, 
  sorry

end geometric_series_common_ratio_l599_599754


namespace monotonic_increasing_condition_l599_599838

open Real

noncomputable def f (x : ℝ) (l a : ℝ) : ℝ := x^2 - x + l + a * log x

theorem monotonic_increasing_condition (l a : ℝ) (x : ℝ) (hx : x > 0) 
  (h : ∀ x, x > 0 → deriv (f l a) x ≥ 0) : 
  a > 1 / 8 :=
by
  sorry

end monotonic_increasing_condition_l599_599838


namespace F_at_2_l599_599350

def F (x : ℝ) : ℝ := real.sqrt (abs (x + 2)) + (10 / real.pi) * real.arctan (real.sqrt (abs x))

theorem F_at_2 : F 2 = 4 :=
by
  sorry

end F_at_2_l599_599350


namespace equidistant_A_B_C_l599_599282

-- Coordinates of points A, B, and C
def A (x : ℝ) := (x, 0, 0)
def B := (0, 1, 3)
def C := (2, 0, 4)

-- Distance function between two points in 3D space
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

-- Prove that A is equidistant from B and C when x = 2.5
theorem equidistant_A_B_C : distance (A 2.5) B = distance (A 2.5) C :=
  sorry

end equidistant_A_B_C_l599_599282


namespace find_range_x_l599_599416

variables {ℝ : Type*} [linear_ordered_field ℝ]
variables (f : ℝ → ℝ) (x y : ℝ)

-- Given: y = f(x) is a monotonically increasing function
def is_monotone_increasing (f : ℝ → ℝ) :=
∀ x1 x2 : ℝ, x1 ≤ x2 → f x1 ≤ f x2

-- Given: The inverse function is y = f⁻¹(x)
def is_inverse (f : ℝ → ℝ) (f_inv : ℝ → ℝ) :=
∀ y, f (f_inv y) = y ∧ f_inv (f y) = y

-- Given: The graph of y = f(x+1) passes through points A(-4,0) and B(2,3)
def passes_through_points (f : ℝ → ℝ) :=
f (-4 + 1) = 0 ∧ f (2 + 1) = 3

-- Given: |f⁻¹(x+1)| ≤ 3
def abs_inverse_bound (f_inv : ℝ → ℝ) :=
∀ x, abs (f_inv (x + 1)) ≤ 3

-- Prove: The range of x is -1 ≤ x ≤ 2
theorem find_range_x (f f_inv : ℝ → ℝ) (h_mono : is_monotone_increasing f) 
    (h_inv : is_inverse f f_inv) (h_points : passes_through_points f) 
    (h_abs : abs_inverse_bound f_inv) : -1 ≤ x ∧ x ≤ 2 :=
sorry

end find_range_x_l599_599416


namespace integer_solutions_count_l599_599052

theorem integer_solutions_count :
  ∃ (count : ℤ), (∀ (a : ℤ), 
  (∃ x : ℤ, x^2 + a * x + 8 * a = 0) ↔ count = 8) :=
sorry

end integer_solutions_count_l599_599052


namespace age_of_new_teacher_l599_599596

-- Definitions of conditions
def avg_age_20_teachers (sum_of_ages : ℕ) : Prop :=
  sum_of_ages = 49 * 20

def avg_age_21_teachers (sum_of_ages : ℕ) : Prop :=
  sum_of_ages = 48 * 21

-- The proof goal
theorem age_of_new_teacher (sum_age_20 : ℕ) (sum_age_21 : ℕ) (h1 : avg_age_20_teachers sum_age_20) (h2 : avg_age_21_teachers sum_age_21) : 
  sum_age_21 - sum_age_20 = 28 :=
sorry

end age_of_new_teacher_l599_599596


namespace count_lattice_points_l599_599476

theorem count_lattice_points :
  (finset.univ.filter (λ x : ℤ × ℤ, (x.1^2 + x.2^2 < 25)
  ∧ (x.1^2 + x.2^2 < 10 * x.1)
  ∧ (x.1^2 + x.2^2 < 10 * x.2))).card = 13 :=
by
  sorry

end count_lattice_points_l599_599476


namespace tenth_term_is_correct_l599_599420

-- Definitions corresponding to the problem conditions
def sequence_term (n : ℕ) : ℚ := (-1)^n * (2 * n + 1) / (n^2 + 1)

-- Theorem statement for the equivalent proof problem
theorem tenth_term_is_correct : sequence_term 10 = 21 / 101 := by sorry

end tenth_term_is_correct_l599_599420


namespace sin_14pi_over_5_l599_599379

theorem sin_14pi_over_5 : sin (14 * π / 5) = (sqrt (10 - 2 * sqrt 5)) / 4 := 
by 
  sorry

end sin_14pi_over_5_l599_599379


namespace blue_eyes_count_l599_599570

theorem blue_eyes_count (total_students students_both students_neither : ℕ)
  (ratio_blond_to_blue : ℕ → ℕ)
  (h_total : total_students = 40)
  (h_ratio : ratio_blond_to_blue 3 = 2)
  (h_both : students_both = 8)
  (h_neither : students_neither = 5) :
  ∃ y : ℕ, y = 18 :=
by
  sorry

end blue_eyes_count_l599_599570


namespace problem1_problem2_l599_599986

-- Proof problem for Problem 1
theorem problem1 : real.sqrt 9 + real.cbrt (-8) + abs (1 - real.sqrt 3) = real.sqrt 3 := 
  sorry

-- Proof problem for Problem 2
theorem problem2 (x y : ℝ) (h1 : x + y = 3) (h2 : 3 * x - 2 * y = 4) : 
     x = 2 ∧ y = 1 := 
  sorry

end problem1_problem2_l599_599986


namespace polygon_area_is_1008_l599_599029

variables (vertices : List (ℕ × ℕ)) (units : ℕ)

def polygon_area (vertices : List (ℕ × ℕ)) : ℕ :=
sorry -- The function would compute the area based on vertices.

theorem polygon_area_is_1008 :
  vertices = [(0, 0), (12, 0), (24, 12), (24, 0), (36, 0), (36, 24), (24, 36), (12, 36), (0, 36), (0, 24), (0, 0)] →
  units = 1 →
  polygon_area vertices = 1008 :=
sorry

end polygon_area_is_1008_l599_599029


namespace tangent_line_a_zero_range_a_if_fx_neg_max_value_a_one_l599_599090

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x + Real.log x) - x * Real.exp x

theorem tangent_line_a_zero (x : ℝ) (y : ℝ) : 
  a = 0 ∧ x = 1 → (2 * Real.exp 1) * x + y - Real.exp 1 = 0 :=
sorry

theorem range_a_if_fx_neg (a : ℝ) : 
  (∀ x ≥ 1, f a x < 0) → a < Real.exp 1 :=
sorry

theorem max_value_a_one (x : ℝ) : 
  a = 1 → x = (Real.exp 1)⁻¹ → f 1 x = -1 :=
sorry

end tangent_line_a_zero_range_a_if_fx_neg_max_value_a_one_l599_599090


namespace find_weight_of_A_l599_599924

theorem find_weight_of_A 
  (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : E = D + 5) 
  (h4 : (B + C + D + E) / 4 = 79) 
  : A = 77 := 
sorry

end find_weight_of_A_l599_599924


namespace anya_wins_19_games_l599_599345

theorem anya_wins_19_games (total_rounds : ℕ)
                           (anya_rock anya_scissors anya_paper borya_rock borya_scissors borya_paper : ℕ)
                           (no_draws : total_rounds = 25)
                           (anya_choices : anya_rock = 12 ∧ anya_scissors = 6 ∧ anya_paper = 7)
                           (borya_choices : borya_rock = 13 ∧ borya_scissors = 9 ∧ borya_paper = 3) 
                           : ∃ (anya_wins : ℕ), anya_wins = 19 := 
by
  have anya_rock_wins := min anya_rock borya_scissors  -- Rock wins against Scissors
  have anya_scissors_wins := min anya_scissors borya_paper  -- Scissors win against Paper
  have anya_paper_wins := min anya_paper borya_rock  -- Paper wins against Rock
  let total_wins := anya_rock_wins + anya_scissors_wins + anya_paper_wins
  have : total_wins = 19 := by 
    rw [anya_choices, borya_choices]
    simp
    done sorry
  exact ⟨total_wins⟩


end anya_wins_19_games_l599_599345


namespace sum_of_areas_of_super_cool_right_triangles_l599_599693

def isSuperCoolRightTriangle (a b : ℕ) : Prop :=
  (a * b) / 2 = 3 * (a + b)

def area (a b : ℕ) : ℕ :=
  (a * b) / 2

theorem sum_of_areas_of_super_cool_right_triangles :
  (∑ (a, b) in {(a, b) | isSuperCoolRightTriangle a b}.toFinset, area a b).sum = 471 :=
by
  sorry

end sum_of_areas_of_super_cool_right_triangles_l599_599693


namespace emily_gives_back_marble_l599_599369

theorem emily_gives_back_marble (initial_marbles : ℕ) (final_marbles : ℕ) (extra_marbles : ℕ) 
    (double_initial : ℕ) (new_total : ℕ) : initial_marbles = 6 → final_marbles = 8 → 
    double_initial = 2 * initial_marbles → new_total = initial_marbles + double_initial → 
    new_total - (new_total / 2 + extra_marbles) = final_marbles → 
    extra_marbles = 1 :=
by 
  intros h_initial h_final h_double h_new_total h_equation
  rw [h_initial, h_final, h_double, h_new_total] at h_equation
  simp at h_equation
  linarith

# Test statement
example : emily_gives_back_marble 6 8 1 12 18 :=
by {
  iterate 5 { apply rfl }
}

end emily_gives_back_marble_l599_599369


namespace solve_for_x_l599_599893

theorem solve_for_x (x : ℚ) (h : 10 * x = x + 20) : x = 20 / 9 :=
  sorry

end solve_for_x_l599_599893


namespace series_sum_equals_three_fourths_l599_599726

noncomputable def infinite_series_sum : ℝ :=
  (∑' n : ℕ, (3 * (n + 1) + 2) / ((n + 1) * (n + 1 + 1) * (n + 1 + 3)))

theorem series_sum_equals_three_fourths :
  infinite_series_sum = 3 / 4 :=
sorry

end series_sum_equals_three_fourths_l599_599726


namespace slope_of_line_through_points_l599_599772

theorem slope_of_line_through_points (p1 p2 : ℝ × ℝ) (h1 : p1 = (1, 0)) (h2 : p2 = (4, Real.sqrt 3)) :
  ∃ k α : ℝ, k = (p2.snd - p1.snd) / (p2.fst - p1.fst) ∧ tan α = k ∧ α = π / 6 := 
sorry

end slope_of_line_through_points_l599_599772


namespace find_z_l599_599505

-- Definitions of the given angles and using triangle sum theorem
def angle_ABC : ℝ := 70
def angle_BAC : ℝ := 50
def angle_sum := angle_ABC + angle_BAC + (180 - angle_ABC - angle_BAC) = 180

-- Definitions for the right triangle CDE
def delta_CDE_right_angle := (180 - angle_ABC - angle_BAC) + z = 90

-- The theorem to prove
theorem find_z (z : ℝ) (h1 : angle_ABC = 70) (h2 : angle_BAC = 50)
    (h3 : angle_sum) (h4 : delta_CDE_right_angle) : z = 30 :=
by
  sorry

end find_z_l599_599505


namespace collinear_points_function_inequality_l599_599510

variables {x : ℝ} {k : ℤ}

-- Problem 1: Collinearity of points A, B, and C
theorem collinear_points (A B C : ℝ × ℝ) 
  (hC : (C.1, C.2) = (1/3 : ℝ) • (A.1, A.2) + (2/3 : ℝ) • (B.1, B.2)) : 
  collinear ℝ ({A, B, C} : set (ℝ × ℝ)) := 
sorry

-- Problem 2: Function range condition
theorem function_inequality (A B : ℝ × ℝ)
  (A_eq : A = (Real.sin x, Real.cos x))
  (B_eq : B = (Real.cos x, Real.cos x)) : 
  (f(x) = (A.1 * (A.1 + B.1)) + (A.2 * (A.2 + B.2)) ∧ f(x) ≥ 3/2) ↔
  kπ - π/8 ≤ x ∧ x ≤ kπ + 3π/8 :=
sorry

end collinear_points_function_inequality_l599_599510


namespace plane_parallel_proof_l599_599284

variables {A B C D B' C' D' : Type*}

-- Assuming the existence of planes and perpendiculars as defined in conditions
def tetrahedron (A B C D : Type*) : Prop := sorry
def perpendicular (P Q : Type*) (plane : Type*) : Prop := sorry
def plane_parallel (plane1 plane2 : Type*) : Prop := sorry

theorem plane_parallel_proof 
  (A B C D B' C' D' : Type*)
  (H_tetra : tetrahedron A B C D)
  (H_AB_perp : perpendicular A B' (∠ C D))
  (H_AC_perp : perpendicular A C' (∠ B D))
  (H_AD_perp : perpendicular A D' (∠ B C))
  : plane_parallel (∠ B' C' D') (∠ B C D) :=
begin
  sorry,
end

end plane_parallel_proof_l599_599284


namespace B_can_complete_alone_l599_599997

-- Define the given conditions
def A_work_rate := 1 / 20
def total_days := 21
def A_quit_days := 15
def B_completion_days := 30

-- Define the problem statement in Lean
theorem B_can_complete_alone (x : ℝ) (h₁ : A_work_rate = 1 / 20) (h₂ : total_days = 21)
  (h₃ : A_quit_days = 15) (h₄ : (21 - A_quit_days) * (1 / 20 + 1 / x) + A_quit_days * (1 / x) = 1) :
  x = B_completion_days :=
  sorry

end B_can_complete_alone_l599_599997


namespace range_of_expression_l599_599466

noncomputable def range_expression (a b : ℝ) : ℝ := a^2 + b^2 - 6 * a - 8 * b

variables (a b : ℝ)

def circle1 (a : ℝ) : Prop := ∀ x y : ℝ, (x - a)^2 + y^2 = 1

def circle2 (b : ℝ) : Prop := ∀ x y : ℝ, x^2 + y^2 - 2 * b * y + b^2 - 4 = 0

theorem range_of_expression :
  (circle1 a ∧ circle2 b ∧ ∃ (a b : ℝ), sqrt (a^2 + b^2) = 3) → -21 ≤ range_expression a b ∧ range_expression a b ≤ 39 := 
sorry

end range_of_expression_l599_599466


namespace range_M_l599_599832

noncomputable def M (a : ℝ) : ℝ := (a^2 + 4) / a

theorem range_M : 
  (∀ a : ℝ, a ≠ 0 → M a ∈ Set.Icc (4 : ℝ) ⊤ ∨ M a ∈ Set.Iic (⊤) ((-4 : ℝ))) :=
sorry

end range_M_l599_599832


namespace curve_m_circle_curve_intersection_perpendicular_l599_599083

/-- For curve C: x^2 + y^2 - 2x - 4y + m = 0, prove that it represents a circle
if and only if m < 5. --/
theorem curve_m_circle (m : ℝ) : (∀ x y : ℝ, x ^ 2 + y ^ 2 - 2 * x - 4 * y + m = 0) ↔ m < 5 :=
sorry

/-- Given curve C: x^2 + y^2 - 2x - 4y + m = 0 intersects the line x + 2y - 3 = 0 at points
M and N such that vectors OM and ON are perpendicular, prove that m = 12 / 5. --/
theorem curve_intersection_perpendicular (m : ℝ) : 
  (∃ M N : ℝ × ℝ, 
    (M ≠ N) ∧ 
    (x ^ 2 + y ^ 2 - 2 * x - 4 * y + m = 0) 
    ∧ (fst M + 2 * snd M - 3 = 0)
    ∧ (fst N + 2 * snd N - 3 = 0)
    ∧ (fst M * fst N + snd M * snd N = 0)) ↔ m = 12 / 5 :=
sorry

end curve_m_circle_curve_intersection_perpendicular_l599_599083


namespace triangle_ABC_obtuse_and_sum_opposite_angles_constant_l599_599171

open EuclideanGeometry

def vertices_triangle (A B C : Point) : Prop := 
  triangle A B C

def feet_angle_bisectors (A B C P Q R: Point) : Prop := 
  ∃ P Q R, angle_bisector A (line B C) = line A P ∧ 
          angle_bisector B (line A C) = line B Q ∧ 
          angle_bisector C (line A B) = line C R

def right_triangle_at_P (P Q R : Point) : Prop := 
  ∠QPR = 90°

def obtuse_triangle (A B C : Point) : Prop := 
  ∃ α β γ : Angle, α + β + γ = 180° ∧ (α > 90° ∨ β > 90° ∨ γ > 90°)

def opposite_angles_constant_sum (A R Q P : Point) : Prop := 
  ∠ARP + ∠AQR = 180°

theorem triangle_ABC_obtuse_and_sum_opposite_angles_constant 
  (A B C P Q R : Point) 
  (h1 : vertices_triangle A B C)
  (h2 : feet_angle_bisectors A B C P Q R)
  (h3 : right_triangle_at_P P Q R) : 
  obtuse_triangle A B C ∧ opposite_angles_constant_sum A R Q P :=
sorry

end triangle_ABC_obtuse_and_sum_opposite_angles_constant_l599_599171


namespace max_number_of_knights_not_all_liars_l599_599188

-- Assuming the definitions in Part A
variables (N : ℕ) (students : fin 2N → Prop)
variables (is_knight : fin 2N → Prop) (is_liar : fin 2N → Prop)
variables (taller_than : fin 2N → fin 2N → Prop)

-- Conditions
axiom knight_truth : ∀ (i : fin 2N), is_knight i ↔ ∀ j, students j → taller_than i j ↔ true
axiom liar_lie : ∀ (i : fin 2N), is_liar i ↔ ∀ j, students j → taller_than i j ↔ false
axiom pairs : ∀ (i: fin N), ∃ j: fin 2N, i = j/2

-- Part A: Prove that the maximum number of knights is N
theorem max_number_of_knights : 
  ∃ (knight_count : ℕ), knight_count ≤ N ∧
  ∀ (student_pairs : fin N), 
    (∃ k1 k2 : fin 2N, is_knight k1 ∧ is_knight k2) → knight_count = N := 
sorry

-- Part B: Prove that not all students can be liars
theorem not_all_liars : 
  ¬(∀ (i : fin 2N), is_liar i) :=
sorry

end max_number_of_knights_not_all_liars_l599_599188


namespace intersection_y_sum_zero_l599_599142

theorem intersection_y_sum_zero :
  ∀ (x1 y1 x2 y2 : ℝ), (y1 = 2 * x1) ∧ (y1 = 2 / x1) ∧ (y2 = 2 * x2) ∧ (y2 = 2 / x2) →
  (x2 = -x1) ∧ (y2 = -y1) →
  y1 + y2 = 0 :=
by
  sorry

end intersection_y_sum_zero_l599_599142


namespace tan_half_angle_l599_599432

theorem tan_half_angle
  (x y : ℝ)
  (h1 : cos (x + y) * sin x - sin (x + y) * cos x = 12 / 13)
  (h2 : 3 * π / 2 < y ∧ y < 2 * π) :
  tan (y / 2) = -2 / 3 := 
by sorry

end tan_half_angle_l599_599432


namespace sufficient_not_necessary_l599_599659

theorem sufficient_not_necessary (x : ℝ) (h : x > 2) : (log x / log 2) > 0 :=
by {
  rw [div_pos_iff],
  split,
  exact log_pos one_lt_two,
  exact log_pos (by linarith from h)
  sorry
}

end sufficient_not_necessary_l599_599659


namespace time_for_a_to_complete_work_l599_599976

theorem time_for_a_to_complete_work :
  ∃ x : ℕ, (∀ (b_days : ℕ) (c_days : ℕ) (together_days : ℕ), 
    b_days = 6 ∧ c_days = 12 ∧ together_days = 24 / 7 → 
    (1 / x.toFloat) + (1 / b_days.toFloat) + (1 / c_days.toFloat) = 7 / 24) :=
begin
  use 24,
  sorry
end

end time_for_a_to_complete_work_l599_599976


namespace max_S_value_n_l599_599288

open Nat

-- Definitions
def a_n (n : ℕ) : ℝ := 20 + (n - 1) * (-5/3)

def S (n : ℕ) : ℝ := 20 * n + (n * (n - 1) / 2) * (-5/3)

-- Problem statement
theorem max_S_value_n : (S 10 = S 15) → (∃ n, S n = 130 ∧ (n = 12 ∨ n = 13)) :=
begin
  sorry -- proof goes here
end

end max_S_value_n_l599_599288


namespace functionMachine_output_l599_599015

-- Define the function machine according to the specified conditions
def functionMachine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 30 then step1 - 4 else step1
  let step3 := if step2 <= 20 then step2 + 8 else step2 - 5
  step3

-- Statement: Prove that the functionMachine applied to 10 yields 25
theorem functionMachine_output : functionMachine 10 = 25 :=
  by
    sorry

end functionMachine_output_l599_599015


namespace a_leq_neg4_l599_599440

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0
def neg_p (a x : ℝ) : Prop := ¬(p a x)
def neg_q (x : ℝ) : Prop := ¬(q x)

theorem a_leq_neg4 (a : ℝ) (h_neg_p : ∀ x, neg_p a x → neg_q x) (h_a_neg : a < 0) :
  a ≤ -4 :=
sorry

end a_leq_neg4_l599_599440


namespace tangent_angle_measure_l599_599257

-- Definitions derived from the given conditions
variables {O A B C : Point}
variables (circle : Circle O)
variables (tan1 tan2 : Line)
variables [Tangent tan1 circle A B] [Tangent tan2 circle A C]

-- We define the arc ratios and angle relationship.
def arc_ratio (α β : Real) := 3 = α * (3+5) / (α + β) ∧ 5 = β * (3+5) / (α + β)

-- The goal is to show that the interior angle BAC is 67.5 degrees
theorem tangent_angle_measure
  (h1 : tangent tan1 circle A B) 
  (h2 : tangent tan2 circle A C)
  (h3 : arc_ratio α β)
  : ∠BAC = 67.5 :=
sorry

end tangent_angle_measure_l599_599257


namespace num_diamonds_F10_l599_599729

def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 4 else 4 * (3 * n - 2)

theorem num_diamonds_F10 : num_diamonds 10 = 112 := by
  sorry

end num_diamonds_F10_l599_599729


namespace max_f_value_l599_599734

def f (x : ℝ) : ℝ := min (min (3 * x + 1) (- (1 / 3) * x + 2)) (x + 4)

theorem max_f_value : ∃ x : ℝ, f x = (5 / 2) := by
  use (-3 / 2)
  dsimp [f]
  -- verification calculation skipped
  sorry

end max_f_value_l599_599734


namespace crayons_count_l599_599912

def crayons_per_box : ℕ := 8
def number_of_boxes : ℕ := 10
def total_crayons : ℕ := crayons_per_box * number_of_boxes

theorem crayons_count : total_crayons = 80 := by
  sorry

end crayons_count_l599_599912


namespace fill_tank_time_l599_599630

theorem fill_tank_time (hA : ∀ t : Real, t > 0 → (t / 10) = 1) 
                       (hB : ∀ t : Real, t > 0 → (t / 20) = 1) 
                       (hC : ∀ t : Real, t > 0 → (t / 30) = 1) : 
                       (60 / 7 : Real) = 60 / 7 :=
by
    sorry

end fill_tank_time_l599_599630


namespace students_in_both_band_and_chorus_l599_599945

theorem students_in_both_band_and_chorus (total_students band_students chorus_students band_or_chorus: ℕ)
  (H1 : total_students = 200)
  (H2 : band_students = 70)
  (H3 : chorus_students = 95)
  (H4 : band_or_chorus = 150) :
  band_students + chorus_students - band_or_chorus = 15 :=
by
  rw [H2, H3, H4]
  simp
  sorry

end students_in_both_band_and_chorus_l599_599945


namespace grocer_purchased_pounds_l599_599977

theorem grocer_purchased_pounds (cost_price sale_price profit : ℚ)
  (h1 : cost_price = (0.50 / 3)) 
  (h2 : sale_price = (1.00 / 4)) 
  (h3 : profit = (8.00 : ℚ)) :
  let pounds_purchased := profit / (sale_price - cost_price) in
  pounds_purchased = 96 :=
by
  sorry

end grocer_purchased_pounds_l599_599977


namespace make_up_set_money_needed_l599_599410

theorem make_up_set_money_needed (makeup_cost gabby_money mom_money: ℤ) (h1: makeup_cost = 65) (h2: gabby_money = 35) (h3: mom_money = 20) :
  (makeup_cost - (gabby_money + mom_money)) = 10 :=
by {
  sorry
}

end make_up_set_money_needed_l599_599410


namespace monotonic_decreasing_interval_l599_599232

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - log x

theorem monotonic_decreasing_interval : Set.Ioo 0 1 ∪ {1} = {x : ℝ | 0 < x ∧ x ≤ 1 ∧ f' x < 0} :=
sorry

end monotonic_decreasing_interval_l599_599232


namespace largest_number_of_primes_l599_599853

-- Define the nature of our table cells
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0
def is_product_of_two_primes (n : ℕ) := ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p * q

-- Conditions of the problem
def table_valid (table : Fin 80 → Fin 80 → ℕ) : Prop :=
  (∀ i j, table i j > 0 ∧ (is_prime (table i j) ∨ is_product_of_two_primes (table i j))) ∧
  (∀ i j, ∃ k, (k ≠ j ∧ ¬ Nat.coprime (table i j) (table i k)) ∨
              ∃ l, (l ≠ i ∧ ¬ Nat.coprime (table i j) (table l j)))

-- The maximum number of primes in the table
def max_primes_in_table : ℕ := 4266

-- Main theorem statement
theorem largest_number_of_primes (table : Fin 80 → Fin 80 → ℕ) (valid : table_valid table):
  ∃ primes_count : ℕ, primes_count ≤ max_primes_in_table := sorry

end largest_number_of_primes_l599_599853


namespace monotonic_intervals_l599_599021

noncomputable def y : ℝ → ℝ := λ x => x * Real.log x

theorem monotonic_intervals :
  (∀ x : ℝ, 0 < x → x < (1 / Real.exp 1) → y x < -1) ∧ 
  (∀ x : ℝ, (1 / Real.exp 1) < x → x < 5 → y x > 1) := 
by
  sorry -- Proof goes here.

end monotonic_intervals_l599_599021


namespace andrei_monthly_spending_l599_599338

noncomputable def original_price := 50
noncomputable def price_increase := 0.10
noncomputable def discount := 0.10
noncomputable def kg_per_month := 2

def new_price := original_price + original_price * price_increase
def discounted_price := new_price - new_price * discount
def monthly_spending := discounted_price * kg_per_month

theorem andrei_monthly_spending : monthly_spending = 99 := by
  sorry

end andrei_monthly_spending_l599_599338


namespace scalar_triple_product_sum_to_zero_altitudes_of_triangle_meet_at_point_l599_599275

-- Part (a)
theorem scalar_triple_product_sum_to_zero (A B C D : ℝ × ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2),
      BC := (C.1 - B.1, C.2 - B.2),
      CD := (D.1 - C.1, D.2 - C.2),
      CA := (A.1 - C.1, A.2 - C.2),
      AD := (D.1 - A.1, D.2 - A.2),
      BD := (D.1 - B.1, D.2 - B.2) in
  let scalar_triple_product (u v : ℝ × ℝ) : ℝ :=
    u.1 * v.1 + u.2 * v.2 in
  scalar_triple_product AB CD +
  scalar_triple_product BC AD +
  scalar_triple_product CA BD = 0 :=
sorry

-- Part (b)
theorem altitudes_of_triangle_meet_at_point (A B C : ℝ × ℝ) :
  ∃ D : ℝ × ℝ,
  (let AB := (B.1 - A.1, B.2 - A.2),
       AC := (C.1 - A.1, C.2 - A.2),
       BD := (D.1 - B.1, D.2 - B.2),
       CD := (D.1 - C.1, D.2 - C.2) in
  (AB.1 * AC.2 - AB.2 * AC.1 = 0) ∧
  (BD.1 * AC.2 - BD.2 * AC.1 = 0) ∧
  (AB.1 * CD.2 - AB.2 * CD.1 = 0) ∧
  (BD.1 * CD.2 - BD.2 * CD.1 = 0)) :=
sorry

end scalar_triple_product_sum_to_zero_altitudes_of_triangle_meet_at_point_l599_599275


namespace other_root_of_quadratic_l599_599115

theorem other_root_of_quadratic (m : ℝ) (root1 : ℝ) (h_roots : root1 = 2)
  (h_quadratic : ∀ x, x^2 + m * x - 6 = 0 ↔ x = root1 ∨ x = -3) : 
  ∃ root2 : ℝ, root2 = -3 :=
by
  use -3
  sorry

end other_root_of_quadratic_l599_599115


namespace value_of_larger_denom_eq_10_l599_599341

/-- Anna has 12 bills in her wallet, and the total value is $100. 
    She has 4 $5 bills and 8 bills of a larger denomination.
    Prove that the value of the larger denomination bill is $10. -/
theorem value_of_larger_denom_eq_10 (n : ℕ) (b : ℤ) (total_value : ℤ) (five_bills : ℕ) (larger_bills : ℕ):
    (total_value = 100) ∧ 
    (five_bills = 4) ∧ 
    (larger_bills = 8) ∧ 
    (n = five_bills + larger_bills) ∧ 
    (n = 12) → 
    (b = 10) :=
by
  sorry

end value_of_larger_denom_eq_10_l599_599341


namespace polyhedron_faces_count_l599_599677

theorem polyhedron_faces_count (V F E : ℕ) (h1 : ∀ f, face f → f.is_quadrilateral) (h2 : V = 16) 
  (h3 : E = 2 * F) (h4 : V + F - E = 2) : F = 14 :=
by sorry

end polyhedron_faces_count_l599_599677


namespace other_root_of_quadratic_eq_l599_599121

theorem other_root_of_quadratic_eq (m : ℝ) (t : ℝ) (h1 : (polynomial.X ^ 2 + polynomial.C m * polynomial.X + polynomial.C (-6)).roots = {2, t}) : t = -3 :=
sorry

end other_root_of_quadratic_eq_l599_599121


namespace max_value_expression_l599_599533

theorem max_value_expression (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + 2 * c = 1) :
  2 * a + Real.sqrt (2 * a * b) + Real.cbrt (4 * a * b * c) ≤ 3 / 2 :=
sorry

end max_value_expression_l599_599533


namespace hari_contribution_l599_599979

theorem hari_contribution (c_p: ℕ) (m_p: ℕ) (ratio_p: ℕ) 
                          (m_h: ℕ) (ratio_h: ℕ) (profit_ratio_p: ℕ) (profit_ratio_h: ℕ) 
                          (c_h: ℕ) : 
  (c_p = 3780) → 
  (m_p = 12) → 
  (ratio_p = 2) → 
  (m_h = 7) → 
  (ratio_h = 3) → 
  (profit_ratio_p = 2) →
  (profit_ratio_h = 3) →
  (c_p * m_p * profit_ratio_h) = (c_h * m_h * profit_ratio_p) → 
  c_h = 9720 :=
by
  intros
  sorry

end hari_contribution_l599_599979


namespace cm_dn_constant_l599_599765
noncomputable theory

open_locale classical

variables {α : Type} [linear_ordered_field α] [module α (euclidean_space α)] {A B : euclidean_space α}
          {circle : set (euclidean_space α)} (hAB : A ∈ circle ∧ B ∈ circle)
          {r : line α} {P : euclidean_space α} (hP : P ∈ circle)

-- The line intersects with PA and PB at points C and D respectively
variables (C D : euclidean_space α) (hC : C ∈ r ∧ ∃ l : line α, P ∈ l ∧ A ∈ l ∧ l ∩ r = {C})
          (hD : D ∈ r ∧ ∃ l' : line α, P ∈ l' ∧ B ∈ l' ∧ l' ∩ r = {D})

-- Fixed points M and N on line r
variables (M N : euclidean_space α) (hM : M ∈ r) (hN : N ∈ r)

-- The main theorem statement
theorem cm_dn_constant : ∃ (M N : euclidean_space α), M ∈ r ∧ N ∈ r ∧
  ∀ P (hP : P ∈ circle) (C D : euclidean_space α), 
  (C ∈ r ∧ ∃ l : line α, P ∈ l ∧ A ∈ l ∧ l ∩ r = {C}) ∧ 
  (D ∈ r ∧ ∃ l' : line α, P ∈ l' ∧ B ∈ l' ∧ l' ∩ r = {D}) → 
  let CM := ∥C - M∥, DN := ∥D - N∥ in CM * DN = (fixed_value : α) :=
sorry

end cm_dn_constant_l599_599765


namespace rotated_rectangle_height_l599_599631

theorem rotated_rectangle_height :
  ∀ (w h : ℝ), w = 2 ∧ h = 1 →
  let original_position := (w, h),
      rotated_position := (h, w) in
  rotated_position.2 = 2 :=
by
  intros w h H
  cases H
  simp only [Prod.snd, rotated_position.2]
  sorry

end rotated_rectangle_height_l599_599631


namespace only_statement_1_is_correct_l599_599957

-- Given conditions under the oblique projection method
def oblique_projection_rule (shape : Type) (projected_shape : Type) : Prop :=
  match shape, projected_shape with
  | "triangle", "triangle" => true
  | "square", "parallelogram" => true
  | "isosceles_trapezoid", "trapezoid" => true
  | "rhombus", "rhombus" => true
  | "rhombus", "rectangle" => true
  | _, _ => false

-- Statements to be evaluated
def statement_1 : Prop := oblique_projection_rule "triangle" "triangle"
def statement_2 : Prop := oblique_projection_rule "square" "rhombus"
def statement_3 : Prop := oblique_projection_rule "isosceles_trapezoid" "parallelogram"
def statement_4 : Prop := oblique_projection_rule "rhombus" "rhombus"

theorem only_statement_1_is_correct :
  statement_1 ∧ ¬statement_2 ∧ ¬statement_3 ∧ ¬statement_4 :=
by
  sorry

end only_statement_1_is_correct_l599_599957


namespace inversely_proportional_x_y_l599_599593

noncomputable def k := 320

theorem inversely_proportional_x_y (x y : ℕ) (h1 : x * y = k) :
  (∀ x, y = 10 → x = 32) ↔ (x = 32) :=
by
  sorry

end inversely_proportional_x_y_l599_599593


namespace value_of_a_1003_l599_599777

variables (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Definitions for the given conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

def sum_of_first_n_terms (S a : ℕ → ℤ) (n : ℕ) : Prop :=
  S n = ∑ i in range (n + 1), a i

-- Given conditions
variables (h_arith_seq : is_arithmetic_sequence a)
          (h_sum_2006 : sum_of_first_n_terms S a 2005 ∧ S 2005 = 2008)
          (h_sum_even : ∑ i in range 1003, a (2 * i + 2) = 2)

-- Goal to prove
theorem value_of_a_1003 : a 1002 = 2 := sorry

end value_of_a_1003_l599_599777


namespace johnny_laps_per_minute_l599_599156

noncomputable def laps_per_minute (laps : ℝ) (minutes : ℝ) : ℝ :=
  laps / minutes

theorem johnny_laps_per_minute : 
  laps_per_minute 10 3.33333 ≈ 3 := 
by
  sorry

end johnny_laps_per_minute_l599_599156


namespace finite_points_no_non_collinear_isogonal_conjugate_exists_area_of_largest_pentagon_triangle_to_square_point_bisectors_inequality_surface_area_comparison_l599_599289

-- Problem (a)
theorem finite_points_no_non_collinear (S : Finset Point) (h_collinear : ¬ ∀ (A B C ∈ S), collinear A B C) (h_condition : ∀ (A B ∈ S), ∃ C ∈ S, line_through A B passes_through C) :
  False :=
by
  sorry

-- Problem (b)
theorem isogonal_conjugate_exists (ABC : Triangle) (P : Point) (h_is_incenter : is_incenter P) :
  ∃ Q ≠ P, isogonal_conjugate_of ABC Q = Q :=
by
  sorry

-- Problem (c)
theorem area_of_largest_pentagon (F : ConvexFigure) (P : Pentagon) (h_inscribed : P.inscribed_in F) :
  ¬ (area P ≥ (3/4 : ℝ) * area F) :=
by
  sorry

-- Problem (d)
theorem triangle_to_square (T : EquilateralTriangle) :
  ∃ (pieces : Fin (2017)), can_rearrange_into_square T pieces :=
by
  sorry

-- Problem (e)
theorem point_bisectors_inequality (ABC : Triangle) (P : Point) (D E F : Points)
  (h_bisectors : bisects PD ∠BPC ∧ bisects PE ∠CPA ∧ bisects PF ∠APB) :
  AP + BP + CP ≥ 2 * (PD + PE + PF) :=
by
  sorry

-- Problem (f)
theorem surface_area_comparison (P_sphere_2018 P_sphere_2017 : ℝ) (h_surface : surface_area_of_unit_sphere 2018 = P_sphere_2018 ∧ surface_area_of_unit_sphere 2017 = P_sphere_2017) :
  P_sphere_2018 ≤ P_sphere_2017 :=
by
  sorry

end finite_points_no_non_collinear_isogonal_conjugate_exists_area_of_largest_pentagon_triangle_to_square_point_bisectors_inequality_surface_area_comparison_l599_599289


namespace max_primes_in_valid_80x80_table_l599_599852

-- Definitions as derived from the conditions in the problem
def is_prime_or_product_of_two_primes (n : ℕ) : Prop :=
  nat.prime n ∨ (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ n = p * q)

def not_coprime (a b : ℕ) : Prop := (nat.gcd a b ≠ 1)

def valid_table (table : ℕ × ℕ → ℕ) : Prop :=
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 80 ∧ 1 ≤ j ∧ j ≤ 80 → pairwise (≠) (λ k l, table k l)) ∧
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 80 ∧ 1 ≤ j ∧ j ≤ 80 → is_prime_or_product_of_two_primes (table (i, j))) ∧
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 80 ∧ 1 ≤ j ∧ j ≤ 80 →
    ∃ k, (1 ≤ k ∧ k ≤ 80 ∧ not_coprime (table (i, j)) (table (i, k))) ∨ 
    (1 ≤ k ∧ k ≤ 80 ∧ not_coprime (table (i, j)) (table (k, j))))

noncomputable def max_primes_in_valid_table (table : ℕ × ℕ → ℕ) : ℕ :=
  finset.card (finset.filter nat.prime (finset.univ.image (λ (i, j), table (i, j))))

theorem max_primes_in_valid_80x80_table : 
  ∃ (table : ℕ × ℕ → ℕ), valid_table table ∧ max_primes_in_valid_table table = 4266 :=
sorry

end max_primes_in_valid_80x80_table_l599_599852


namespace correct_survey_option_l599_599705

-- Definitions for survey options
inductive SurveyOption
| A
| B
| C
| D

-- Predicate that checks if an option is suitable for a comprehensive survey method
def suitable_for_comprehensive_survey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => false
  | SurveyOption.B => false
  | SurveyOption.C => false
  | SurveyOption.D => true

-- Theorem statement
theorem correct_survey_option : suitable_for_comprehensive_survey SurveyOption.D := 
  by sorry

end correct_survey_option_l599_599705


namespace min_points_condition_met_l599_599963

noncomputable def min_points_on_circle (L : ℕ) : ℕ := 1304

theorem min_points_condition_met (L : ℕ) (hL : L = 1956) :
  (∀ (points : ℕ → ℕ), (∀ n, points n ≠ points (n + 1) ∧ points n ≠ points (n + 2)) ∧ (∀ n, points n < L)) →
  min_points_on_circle L = 1304 :=
by
  -- Proof steps omitted
  sorry

end min_points_condition_met_l599_599963


namespace convex_h_is_convex_l599_599637

noncomputable def number_of_sides_of_H (n1 n2 : ℕ) : ℕ :=
(max n1 n2 + n1 + n2) / 2

noncomputable def perimeter_of_H (P1 P2 : ℝ) : ℝ :=
(P1 + P2) / 2

theorem convex_h_is_convex {n1 n2 : ℕ} (h1 : ∀ (F : polygon ℝ) (G : polygon ℝ), F.sides = n1 ∧ G.sides = n2 → convex F ∧ convex G → convex (midpoint_set F G)) 
(P1 P2 : ℝ): 
forall (F : polygon ℝ) (G : polygon ℝ),
F.perimeter = P1 ∧ G.perimeter = P2 ∧
F.sides = n1 ∧ G.sides = n2 →
(∃ (H : polygon ℝ), H.sides ≥ max n1 n2 ∧ H.sides ≤ n1 + n2 ∧ H.perimeter = number_of_sides_of_H n1 n2 ∧ H.perimeter = perimeter_of_H P1 P2)
:= sorry

end convex_h_is_convex_l599_599637


namespace ratio_of_fifteenth_terms_l599_599162

def S (n : ℕ) : ℝ := sorry
def T (n : ℕ) : ℝ := sorry
def a (n : ℕ) : ℝ := sorry
def b (n : ℕ) : ℝ := sorry

theorem ratio_of_fifteenth_terms 
  (h1: ∀ n, S n / T n = (5 * n + 3) / (3 * n + 35))
  (h2: ∀ n, a n = S n) -- Example condition
  (h3: ∀ n, b n = T n) -- Example condition
  : (a 15 / b 15) = 59 / 57 := 
  by 
  -- Placeholder proof
  sorry

end ratio_of_fifteenth_terms_l599_599162


namespace zero_in_interval_l599_599246

noncomputable def f (x : ℝ) : ℝ := (1 / x) - 6 + 2 * x

theorem zero_in_interval : ∃ c ∈ (2:ℝ, 3:ℝ), f c = 0 :=
by
  -- note that the proof part is skipped using 'sorry'
  sorry

end zero_in_interval_l599_599246


namespace part_one_part_two_l599_599769

open Real

-- First part of the problem:
theorem part_one (f : ℝ → ℝ) 
    (h_nat : ∀ x, f x = |x - 1|) 
    (ineq : ∀ x, f (x - 1) + f (x + 3) ≥ 6) 
    : ∀ x, x ≤ -3 ∨ x ≥ 3 :=
sorry

-- Second part of the problem:
theorem part_two (f : ℝ → ℝ) 
    (h_nat : ∀ x, f x = |x - 1|)
    (ab_ne_zero : ∀ a b : ℝ, a ≠ 0)
    (abs_lt_one_a : ∀ a, |a| < 1)
    (abs_lt_one_b : ∀ b, |b| < 1)
    : ∀ (a b : ℝ), f (a * b) > |a| * f (b / a) :=
sorry

end part_one_part_two_l599_599769


namespace rectangle_area_percentage_change_l599_599279

theorem rectangle_area_percentage_change (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let A1 := L * W,
      L2 := 1.40 * L,
      W2 := W / 2,
      A2 := L2 * W2,
      percentage_change := ((A2 - A1) / A1) * 100 in
  percentage_change = -30 := 
by
  sorry

end rectangle_area_percentage_change_l599_599279


namespace distinct_colorings_rotations_only_11_distinct_colorings_rotations_and_reflections_10_l599_599360

section ColoringRotationsOnly

variables {C : Type} [Fintype C] (colors : Fin 3 → C) (G : Group (Equiv.perm (Fin 3)))
noncomputable def numDistinctColorings_rotations_only : ℕ :=
  Fintype.card (Quotient (QuotientGroup.leftRel G (Equiv.perm (Fin 3))))
  
theorem distinct_colorings_rotations_only_11 (colors : Fin 3 → (Fin 3 → Fin 3)) : 
  numDistinctColorings_rotations_only colors = 11 := by
  sorry

end ColoringRotationsOnly

section ColoringRotationsAndReflections

variables {C : Type} [Fintype C] (colors : Fin 3 → C) (G : Group (Equiv.perm (Fin 3)))
noncomputable def numDistinctColorings_rotations_and_reflections : ℕ :=
  Fintype.card (Quotient (QuotientGroup.leftRel G (Equiv.perm (Fin 3))))

theorem distinct_colorings_rotations_and_reflections_10 (colors : Fin 3 → (Fin 3 → Fin 3)) : 
  numDistinctColorings_rotations_and_reflections colors = 10 := by
  sorry

end ColoringRotationsAndReflections

end distinct_colorings_rotations_only_11_distinct_colorings_rotations_and_reflections_10_l599_599360


namespace number_of_lattice_points_eq_30_l599_599859

theorem number_of_lattice_points_eq_30 : 
  {p : ℤ × ℤ × ℤ // p.1 * p.1 + p.2.1 * p.2.1 + p.2.2 * p.2.2 = 16}.to_finset.card = 30 := 
by 
  sorry

end number_of_lattice_points_eq_30_l599_599859


namespace percent_decrease_in_square_area_l599_599130

theorem percent_decrease_in_square_area
  (area_triangle_I : ℝ := 27 * Real.sqrt 3)
  (area_triangle_III : ℝ := 12 * Real.sqrt 3)
  (area_square : ℝ := 27)
  (side_length_decrease_percentage : ℝ := 0.15) :
  let side_length_initial := Real.sqrt area_square
  let side_length_new := side_length_initial * (1 - side_length_decrease_percentage)
  let area_new := side_length_new ^ 2
  let percent_decrease := ((area_square - area_new) / area_square) * 100
  percent_decrease ≈ 27.74 :=
by
  sorry

end percent_decrease_in_square_area_l599_599130


namespace least_subtraction_l599_599388

theorem least_subtraction (n : ℕ) (d : ℕ) (r : ℕ) (h1 : n = 45678) (h2 : d = 47) (h3 : n % d = r) : r = 35 :=
by {
  sorry
}

end least_subtraction_l599_599388


namespace find_original_number_l599_599672

theorem find_original_number (n a b: ℤ) 
  (h1 : n > 1000) 
  (h2 : n + 79 = a^2) 
  (h3 : n + 204 = b^2) 
  (h4 : b^2 - a^2 = 125) : 
  n = 3765 := 
by 
  sorry

end find_original_number_l599_599672


namespace width_of_first_sheet_l599_599925

theorem width_of_first_sheet (w : ℝ) (h : 2 * (w * 17) = 2 * (8.5 * 11) + 100) : w = 287 / 34 :=
by
  sorry

end width_of_first_sheet_l599_599925


namespace fraction_to_terminating_decimal_l599_599749

theorem fraction_to_terminating_decimal : (49 : ℚ) / 160 = 0.30625 := 
sorry

end fraction_to_terminating_decimal_l599_599749


namespace anya_wins_19_games_l599_599344

theorem anya_wins_19_games (total_rounds : ℕ)
                           (anya_rock anya_scissors anya_paper borya_rock borya_scissors borya_paper : ℕ)
                           (no_draws : total_rounds = 25)
                           (anya_choices : anya_rock = 12 ∧ anya_scissors = 6 ∧ anya_paper = 7)
                           (borya_choices : borya_rock = 13 ∧ borya_scissors = 9 ∧ borya_paper = 3) 
                           : ∃ (anya_wins : ℕ), anya_wins = 19 := 
by
  have anya_rock_wins := min anya_rock borya_scissors  -- Rock wins against Scissors
  have anya_scissors_wins := min anya_scissors borya_paper  -- Scissors win against Paper
  have anya_paper_wins := min anya_paper borya_rock  -- Paper wins against Rock
  let total_wins := anya_rock_wins + anya_scissors_wins + anya_paper_wins
  have : total_wins = 19 := by 
    rw [anya_choices, borya_choices]
    simp
    done sorry
  exact ⟨total_wins⟩


end anya_wins_19_games_l599_599344


namespace problem_statement_l599_599563

noncomputable def a (n : ℕ) : ℝ := n * 2^(3 - n)
def S (n : ℕ) : ℝ := (1 + n) * 2 - (n + 2) * a n / n

theorem problem_statement (a : ℕ → ℝ) (S : ℕ → ℝ):
  a 1 = 1 ∧ a 2 = 1 ∧ (∀ n : ℕ, n ≥ 1 → n * S n + (n + 2) * a n = 4 * n) →
  a 2017 = 2017 * 2^(-2014) :=
begin
  sorry
end

end problem_statement_l599_599563


namespace value_of_silver_cube_l599_599301

theorem value_of_silver_cube (v4_cube_value : ℝ := 300) (s4 s6 : ℝ := 4) (s6 : ℝ := 6) :
  let v4 := s4 ^ 3 in
  let v6 := s6 ^ 3 in
  let scaling_factor := v6 / v4 in
  let v6_cube_value := v4_cube_value * scaling_factor in
  Real.round v6_cube_value = 1013 :=
  sorry

end value_of_silver_cube_l599_599301


namespace additional_peaches_l599_599290

theorem additional_peaches (initial_peaches total_peaches : ℕ) (h1 : initial_peaches = 20) (h2 : total_peaches = 45) :
  total_peaches - initial_peaches = 25 :=
by
  rw [h1, h2]
  rfl

end additional_peaches_l599_599290


namespace xy_pos_iff_div_pos_ab_leq_mean_sq_l599_599966

-- Definition for question 1
theorem xy_pos_iff_div_pos (x y : ℝ) : 
  (x * y > 0) ↔ (x / y > 0) :=
sorry

-- Definition for question 3
theorem ab_leq_mean_sq (a b : ℝ) : 
  a * b ≤ ((a + b) / 2) ^ 2 :=
sorry

end xy_pos_iff_div_pos_ab_leq_mean_sq_l599_599966


namespace sum_of_squares_l599_599943

theorem sum_of_squares (a b c : ℝ) (h1 : a * b + a * c + b * c = 131) (h2 : a + b + c = 22) : a^2 + b^2 + c^2 = 222 :=
by
  sorry

end sum_of_squares_l599_599943


namespace part1_part2_l599_599087

noncomputable def f (x a b : ℝ) := Real.exp x - a * Real.sin x + b * x

theorem part1 (a : ℝ) (h_a : a > 0) (h_min : ∃ x ∈ Ioo 0 (Real.pi / 2), f x a 0 < ∀ y ∈ Ioo 0 (Real.pi / 2), f y a 0) :
  a > 1 :=
sorry

theorem part2 (a b x0 : ℝ) (h_a : a > 0) (h_b : b < 0) (h_extremum : Real.exp x0 - a * Real.cos x0 + b = 0) :
  f x0 a b ≥ b * Real.log (-b / 2) - Real.sqrt 2 * a :=
sorry

end part1_part2_l599_599087


namespace abs_diff_pow_gt_half_l599_599146

theorem abs_diff_pow_gt_half :
  abs (2 ^ 3000 - 3 ^ 2006) > 1 / 2 :=
by
  -- The proof will go here, but is not needed as per the instructions.
  sorry

end abs_diff_pow_gt_half_l599_599146


namespace circle_equation_l599_599926

theorem circle_equation (C : ℝ × ℝ) (hC : C = (0, 1)) (L : set (ℝ × ℝ)) 
  (hL : ∀ x, (x, 2) ∈ L) :
  ∃ r, ∀ x y, (x, y) ∈ L ∧ (x - (0 : ℝ))^2 + (y - 1)^2 = r^2 ↔ (x^2 + (y - 1)^2 = 1) :=
by
  use 1
  intros x y
  split
  case mp =>
    rintro ⟨hLxy, heq⟩
    rw [←hC] at heq
    simp only [hLxy, heq]
    sorry
  case mpr =>
    intro heq
    use (0, 2)
    simp only [heq, hL]
    sorry

end circle_equation_l599_599926


namespace five_points_plane_l599_599426

noncomputable def sin_54 : ℝ := Real.sin (54 * Real.pi / 180)

theorem five_points_plane (A1 A2 A3 A4 A5 : ℝ × ℝ) (h_distinct : A1 ≠ A2 ∧ A1 ≠ A3 ∧ A1 ≠ A4 ∧ A1 ≠ A5 ∧ A2 ≠ A3 ∧ A2 ≠ A4 ∧ A2 ≠ A5 ∧ A3 ≠ A4 ∧ A3 ≠ A5 ∧ A4 ≠ A5) :
  let d_max := max (max (dist A1 A2) (max (dist A1 A3) (max (dist A1 A4) (max (dist A1 A5) (max (dist A2 A3) (max (dist A2 A4) (max (dist A2 A5) (max (dist A3 A4) (max (dist A3 A5) (dist A4 A5)))))))))))
  let d_min := min (min (dist A1 A2) (min (dist A1 A3) (min (dist A1 A4) (min (dist A1 A5) (min (dist A2 A3) (min (dist A2 A4) (min (dist A2 A5) (min (dist A3 A4) (min (dist A3 A5) (dist A4 A5)))))))))))
  in d_max / d_min ≥ 2 * sin_54 ∧ (d_max / d_min = 2 * sin_54 ↔ ∃ p : ℝ, ∀ i j : ℕ, i ≠ j ∧ i < 5 ∧ j < 5 → dist (A i) (A j) = p) :=
sorry

end five_points_plane_l599_599426


namespace sum_common_divisors_75_45_l599_599394

theorem sum_common_divisors_75_45 : 
    (finset.sum (finset.filter (λ d, 75 % d = 0 ∧ 45 % d = 0) (finset.range 76))) = 24 :=
by sorry

end sum_common_divisors_75_45_l599_599394


namespace parallelogram_perimeter_l599_599276

theorem parallelogram_perimeter
  (abcd : Parallelogram)
  (bd : abcd.diagonal_1 = 2)
  (equilateral_triangle : ∀ (A B C : Point),
    A ∈ abcd.vertices ∧ B ∈ abcd.vertices ∧ C ∈ abcd.vertices ∧ angle_eq B C D = π/3 → 
    (dist B C = dist C D ∧ dist C D = dist B D ∧ dist B D = 2)) :
  abcd.perimeter = 8 :=
by
  sorry

end parallelogram_perimeter_l599_599276


namespace possible_values_a1_l599_599555

theorem possible_values_a1 (m : ℕ) (h_m_pos : 0 < m)
    (a : ℕ → ℕ) (h_seq : ∀ n, a n.succ = if a n < 2^m then a n ^ 2 + 2^m else a n / 2)
    (h1 : ∀ n, a n > 0) :
    (∀ n, ∃ k : ℕ, a n = 2^k) ↔ (m = 2 ∧ ∃ ℓ : ℕ, a 0 = 2 ^ ℓ ∧ 0 < ℓ) :=
by sorry

end possible_values_a1_l599_599555


namespace value_of_f_l599_599399

variable {x t : ℝ}

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 ∨ x = 1 then 0
  else (1 : ℝ) / x

theorem value_of_f (h1 : ∀ x, x ≠ 0 → x ≠ 1 → f (x / (x - 1)) = 1 / x)
                   (h2 : 0 ≤ t ∧ t ≤ Real.pi / 2) :
  f (Real.tan t ^ 2 + 1) = Real.sin (2 * t) ^ 2 / 4 :=
sorry

end value_of_f_l599_599399


namespace range_of_a_l599_599400

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := 
by
  sorry

end range_of_a_l599_599400


namespace log_base_8_of_128_eq_7_div_3_l599_599371

theorem log_base_8_of_128_eq_7_div_3 : log 8 128 = 7 / 3 := by
  sorry

end log_base_8_of_128_eq_7_div_3_l599_599371


namespace big_bea_bananas_l599_599351

theorem big_bea_bananas :
  ∃ (b : ℕ), (b + (b + 8) + (b + 16) + (b + 24) + (b + 32) + (b + 40) + (b + 48) = 196) ∧ (b + 48 = 52) := by
  sorry

end big_bea_bananas_l599_599351


namespace alpha_relation_l599_599169

variable {x : ℝ} (hx : x ∈ Ioo (-1 : ℝ) 0)

noncomputable def alpha1 := Real.cos (Real.sin (x * Real.pi))
noncomputable def alpha2 := Real.sin (Real.cos (x * Real.pi))
noncomputable def alpha3 := Real.cos ((x + 1) * Real.pi)

theorem alpha_relation (hx : x ∈ Ioo (-1 : ℝ) 0) :
  alpha3 hx < alpha2 hx ∧ alpha2 hx < alpha1 hx :=
sorry

end alpha_relation_l599_599169


namespace g_inverse_l599_599801

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x - 2)

noncomputable def g (x a : ℝ) : ℝ := f (x + a)

theorem g_inverse (a : ℝ) : (∀ x, g (g x a) a = x) ↔ a = -1 :=
begin
  sorry
end

end g_inverse_l599_599801


namespace seq_increasing_iff_l599_599443

theorem seq_increasing_iff (k : ℝ) (a : ℕ → ℝ) (n : ℕ) : 
  (∀ n, a n = n^2 + k * n + 2) → 
  (∀ n, a (n + 1) > a n) ↔ k > -3 :=
by 
  intro h 
  sorry

end seq_increasing_iff_l599_599443


namespace least_prime_factor_a_least_prime_factor_b_least_prime_factor_sum_l599_599988

-- Given conditions
def least_prime_factor (n : ℕ) : ℕ :=
  if n ≤ 1 then 0
  else (range (n + 1)).filter (λ d, d > 1 ∧ n % d = 0).headI

theorem least_prime_factor_a (a : ℕ) (ha : least_prime_factor a = 3) : a % 3 = 0 ∧ ∀ p, nat.prime p → p < 3 → ¬ p ∣ a := sorry
theorem least_prime_factor_b (b : ℕ) (hb : least_prime_factor b = 7) : b % 7 = 0 ∧ ∀ p, nat.prime p → p < 7 → ¬ p ∣ b := sorry

-- Proof problem
theorem least_prime_factor_sum (a b : ℕ) (ha : least_prime_factor a = 3) (hb : least_prime_factor b = 7) :
  least_prime_factor (a + b) = 2 :=
sorry

end least_prime_factor_a_least_prime_factor_b_least_prime_factor_sum_l599_599988


namespace solve_abs_eq_l599_599915

theorem solve_abs_eq {x : ℝ} (h₁ : x ≠ 3 ∧ (x >= 3 ∨ x < 3)) :
  (|x - 3| = 5 - 2 * x) ↔ x = 2 :=
by
  split;
  intro h;
  sorry

end solve_abs_eq_l599_599915


namespace inequality_system_has_three_integer_solutions_l599_599406

theorem inequality_system_has_three_integer_solutions (m : ℝ) :
  (∃ (s : finset ℤ), s.card = 3 ∧ ∀ x ∈ s, x + 5 > 0 ∧ x - m ≤ 1) ↔ -3 ≤ m ∧ m < -2 :=
by
  sorry

end inequality_system_has_three_integer_solutions_l599_599406


namespace part1_part2_l599_599793

noncomputable def z := (3 : ℂ) / (2 - (1 : ℂ) * complex.i)
def z1 (m : ℝ) := (2 : ℂ) + m * complex.i

theorem part1 (m : ℝ) (h : complex.abs (z + z1 m) = 5) : m = -5 ∨ m = 3 := sorry

noncomputable def w (a : ℝ) := a * z + 2 * complex.i

theorem part2 (a : ℝ) (h1 : a < 0) (h2 : a + 2 > 0) : -2 < a ∧ a < 0 := 
  ⟨h2, h1⟩

end part1_part2_l599_599793


namespace finite_good_terms_good_term_mean_bounds_l599_599768

-- Condition definition: sequence with finitely many non-zero terms
def finite_nonzero_terms_sequence (a : ℤ → ℕ) :=
  ∃ N : ℕ, ∀ n : ℤ, abs n > N → a n = 0

-- Condition definition: good term based on given k
def good_term (a : ℤ → ℕ) (k : ℕ) (n : ℤ) :=
  ∃ m : ℕ, m > 0 ∧ (1 / m.to_rat) * (Finset.sum (Finset.range m) (fun i => a (n + i))) ≥ k

-- Questions
theorem finite_good_terms {a : ℤ → ℕ} {k : ℕ} (h_seq : finite_nonzero_terms_sequence a) :
  ∃ N M : ℕ, ∀ n : ℤ, n < -N - M ∨ N < n → ¬good_term a k n := sorry

theorem good_term_mean_bounds {a : ℤ → ℕ} {k : ℕ} (h_seq : finite_nonzero_terms_sequence a) :
  ∀g_terms : set ℤ, (∀x∈g_terms, good_term a k x) →
  k ≤ (1 / (g_terms.size.to_rat)) * (Finset.sum (g_terms) (fun n => a n)) ∧
  (1 / (g_terms.size.to_rat)) * (Finset.sum (g_terms) (fun n => a n) ) ≤ 2 * k - 1 := sorry

end finite_good_terms_good_term_mean_bounds_l599_599768


namespace integer_values_satisfying_condition_l599_599239

theorem integer_values_satisfying_condition :
  (∀ (x : ℤ), (4 < Real.sqrt (3 * x - 1)) ∧ (Real.sqrt (3 * x - 1) ≤ 6) → 6 ≤ x ∧ x ≤ 12) →
  {x : ℤ | 6 ≤ x ∧ x ≤ 12}.to_finset.card = 7 :=
by
  intros h
  sorry

end integer_values_satisfying_condition_l599_599239


namespace find_f_five_thirds_l599_599538

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = - f x
def functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (-x)
def specific_value (f : ℝ → ℝ) := f (-1/3) = 1/3

theorem find_f_five_thirds
  (hf_odd : odd_function f)
  (hf_fe : functional_equation f)
  (hf_value : specific_value f) :
  f (5 / 3) = 1 / 3 := by
  sorry

end find_f_five_thirds_l599_599538


namespace emily_collected_total_eggs_l599_599368

def eggs_in_setA : ℕ := (200 * 36) + (250 * 24)
def eggs_in_setB : ℕ := (375 * 42) - 80
def eggs_in_setC : ℕ := (560 / 2 * 50) + (560 / 2 * 32)

def total_eggs_collected : ℕ := eggs_in_setA + eggs_in_setB + eggs_in_setC

theorem emily_collected_total_eggs : total_eggs_collected = 51830 := by
  -- proof goes here
  sorry

end emily_collected_total_eggs_l599_599368


namespace max_teams_rich_and_bad_l599_599858

def total_teams : ℕ := 60
def fraction_bad : ℚ := 3 / 5
def fraction_rich : ℚ := 2 / 3

def bad_teams : ℕ := (fraction_bad * total_teams).to_nat
def rich_teams : ℕ := (fraction_rich * total_teams).to_nat

theorem max_teams_rich_and_bad : ∀ (x : ℕ), x ≤ bad_teams
:= by
  sorry

end max_teams_rich_and_bad_l599_599858


namespace zoe_earnings_from_zachary_l599_599274

noncomputable def babysitting_earnings 
  (total_earnings : ℕ) (pool_cleaning_earnings : ℕ) (earnings_julie_ratio : ℕ) 
  (earnings_chloe_ratio : ℕ) 
  (earnings_zachary : ℕ) : Prop := 
total_earnings = 8000 ∧ 
pool_cleaning_earnings = 2600 ∧ 
earnings_julie_ratio = 3 ∧ 
earnings_chloe_ratio = 5 ∧ 
9 * earnings_zachary = 5400

theorem zoe_earnings_from_zachary : babysitting_earnings 8000 2600 3 5 600 :=
by 
  unfold babysitting_earnings
  sorry

end zoe_earnings_from_zachary_l599_599274


namespace cost_of_each_box_is_8_33_l599_599198

noncomputable def cost_per_box (boxes pens_per_box pens_packaged price_per_packaged price_per_set profit_total : ℕ) : ℝ :=
  let total_pens := boxes * pens_per_box
  let packaged_pens := pens_packaged * pens_per_box
  let packages := packaged_pens / 6
  let revenue_packages := packages * price_per_packaged
  let remaining_pens := total_pens - packaged_pens
  let sets := remaining_pens / 3
  let revenue_sets := sets * price_per_set
  let total_revenue := revenue_packages + revenue_sets
  let cost_total := total_revenue - profit_total
  cost_total / boxes

theorem cost_of_each_box_is_8_33 :
  cost_per_box 12 30 5 3 2 115 = 100 / 12 :=
by
  unfold cost_per_box
  sorry

end cost_of_each_box_is_8_33_l599_599198


namespace graph_equivalence_1_graph_equivalence_2_graph_equivalence_3_compare_fx_max_min_fx_l599_599822

-- Definitions of the given functions and conditions

def g1 (x : ℝ) : ℝ := sqrt (4 * x^2 + 12 * x + 9)
def g2 (x : ℝ) : ℝ := sqrt ((2 * x + 3)^2)
def g3 (x : ℝ) : ℝ := abs (4 * x^2 + 12 * x + 9) / (2 * x + 3)

def f (x : ℝ) : ℝ := x / (1 + x^2)

-- The statements of the proof problem

theorem graph_equivalence_1 (x : ℝ) (h : x ≠ -3/2) : g1 x = abs(2 * x + 3) := sorry

theorem graph_equivalence_2 (x : ℝ) : g2 x = abs(2 * x + 3) := sorry

theorem graph_equivalence_3 (x : ℝ) (h : x ≠ -3/2) : g3 x = 2 * x + 3 := sorry

theorem compare_fx (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) :
  f x1 < f x2 := sorry

theorem max_min_fx : 
  (∀ x : ℝ, f x ≤ f 1) ∧ (∀ x : ℝ, f x ≥ f (-1)) := sorry

end graph_equivalence_1_graph_equivalence_2_graph_equivalence_3_compare_fx_max_min_fx_l599_599822


namespace probability_3_queens_or_at_least_2_aces_l599_599107

-- Definitions of drawing from a standard deck and probabilities involved
def num_cards : ℕ := 52
def num_queens : ℕ := 4
def num_aces : ℕ := 4

def probability_all_queens : ℚ := (4/52) * (3/51) * (2/50)
def probability_2_aces_1_non_ace : ℚ := (4/52) * (3/51) * (48/50)
def probability_3_aces : ℚ := (4/52) * (3/51) * (2/50)
def probability_at_least_2_aces : ℚ := (probability_2_aces_1_non_ace) + (probability_3_aces)

def total_probability : ℚ := probability_all_queens + probability_at_least_2_aces

-- Statement to be proved
theorem probability_3_queens_or_at_least_2_aces :
  total_probability = 220 / 581747 :=
sorry

end probability_3_queens_or_at_least_2_aces_l599_599107


namespace ellipse_focus_and_parallel_l599_599795

noncomputable theory
open_locale classical
open set

theorem ellipse_focus_and_parallel
  (b : ℝ) (hb : b > 0) :
  let C := {p : ℝ × ℝ | p.1^2 / (5 * b^2) + p.2^2 / b^2 = 1} in 
  (2, 0) ∈ {f : ℝ × ℝ | (∃ e : ℝ, e = 1 - b^2 / 5) ∧ dist (0, 0) f = sqrt (5 - b^2)} →
  ∃ (l : set (ℝ × ℝ)),
    l = {p : ℝ × ℝ | ∃ k ≠ 0, p.2 = k * (p.1 - 1)} ∧
    let E := (3, 0),
        F := (5, 2 * b / sqrt (5 - b^2)),
        M := (- _, _),
        N := (∞, _) in
    ∃ x, x ∈ C ∧ x ∈ l →
    line.parallel (line.mk (F, N)) (line.mk ((0,0), (1,0))) :=
begin
  sorry
end

end ellipse_focus_and_parallel_l599_599795


namespace correct_operation_l599_599970

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := 
by 
  sorry

end correct_operation_l599_599970


namespace perpendicular_line_to_plane_l599_599552

/-- Non-coincident lines m, n, l in space -/
variable (m n l : Type)

/-- Non-coincident planes α, β, γ in space -/
variable (α β γ : Type)

/-- Conditions: α ⊥ γ , β ⊥ γ , α ∩ β = l -/
variables α_perpendicular_γ : α ⊥ γ
variables β_perpendicular_γ : β ⊥ γ
variables α_intersect_β_eq_l : α ∩ β = l

/- To be proved: l ⊥ γ -/
theorem perpendicular_line_to_plane (m n l α β γ : Type) 
  (α_perpendicular_γ : α ⊥ γ)
  (β_perpendicular_γ : β ⊥ γ)
  (α_intersect_β_eq_l : α ∩ β = l) : l ⊥ γ := 
sorry

end perpendicular_line_to_plane_l599_599552


namespace sequence_proof_l599_599460

noncomputable def a : ℕ → ℤ
| 0       := 4
| 1       := 22
| (n + 2) := 6 * a (n + 1) - a n

def x : ℕ → ℤ := sorry -- Placeholder, as exact sequence of x_n isn't derived in solution steps
def y : ℕ → ℤ := sorry -- Placeholder, as exact sequence of y_n isn't derived in solution steps

theorem sequence_proof :
  (∀ n, ∃ x_n y_n : ℕ, a n = (y_n^2 + 7) / (x_n - y_n)) :=
sorry

end sequence_proof_l599_599460


namespace range_of_PA_dot_PB_l599_599069

theorem range_of_PA_dot_PB (x y : ℝ) :
  (x^2 + (y - 1)^2 = 1) →  -- condition for circle
  (x^2 / 4 + y^2 = 1) →  -- condition for ellipse
  ∃ z : ℝ, z ∈ set.Icc (-1) (13 / 3) ∧ z = x^2 + y^2 - 2 * y := 
by
  intros hc he
  sorry

end range_of_PA_dot_PB_l599_599069


namespace second_tap_emptying_time_l599_599298

theorem second_tap_emptying_time :
  ∀ (T : ℝ), (∀ (f e : ℝ),
  (f = 1 / 3) →
  (∀ (n : ℝ), (n = 1 / 4.5) →
  (n = f - e ↔ e = 1 / T))) →
  T = 9 :=
by
  sorry

end second_tap_emptying_time_l599_599298


namespace tan_x_eq_one_cos_2x_eq_neg_root3_over_2_l599_599470

open Real

variables (x : ℝ)

def m : ℝ × ℝ := (√2 / 2, -√2 / 2)
def n (x : ℝ) : ℝ × ℝ := (sin x, cos x)

-- Condition on x
def x_condition : Prop := 0 < x ∧ x < π / 2

-- Perpendicular condition
def perp_condition : Prop := (m.1 * (n x).1 + m.2 * (n x).2 = 0)

-- Angle condition
def angle_condition (ε : ℝ) : Prop := ε = π / 3

-- Proof problem (1)
theorem tan_x_eq_one (h1 : perp_condition x) (h2 : x_condition x) : tan x = 1 := sorry

-- Proof problem (2)
theorem cos_2x_eq_neg_root3_over_2 (h1 : (m.1 * (n x).1 + m.2 * (n x).2) = 1 / 2) (h2 : angle_condition (π / 3)) (h3 : x_condition x) : cos (2 * x) = - (√3 / 2) := sorry

end tan_x_eq_one_cos_2x_eq_neg_root3_over_2_l599_599470


namespace knight_chessboard_max_knights_l599_599266

theorem knight_chessboard_max_knights (x y : ℝ) :
  (x + sqrt (x^2 + 1)) * (y + sqrt (y^2 + 1)) = 1 → x + y = 0 := 
by 
  sorry

end knight_chessboard_max_knights_l599_599266


namespace find_third_function_l599_599815

variable {α β : Type*}
variable (φ : α → β)

def is_inverse (f g : α → β) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

theorem find_third_function
  (φ_inverse : β → α)
  (h1 : is_inverse φ φ_inverse)
  (h2 : ∀ x y, φ_inverse x = -y ↔ φ y = -x) :
  ∀ x, (λ x, -φ_inverse (-x)) x = -φ_inverse (-x) :=
by
  sorry

end find_third_function_l599_599815


namespace odd_function_property_l599_599549

theorem odd_function_property {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = - f x) (h2 : ∀ x, f (1 + x) = f (-x)) (h3 : f (-1 / 3) = 1 / 3) : f (5 / 3) = 1 / 3 := 
sorry

end odd_function_property_l599_599549


namespace max_rectangles_in_oblique_prism_l599_599332

variables (P : Type) [ObliquePrism P]

-- Conditions
axiom base_has_at_most_two_parallel_edges : ∀ {b : base P}, ∃ e₁ e₂ : edge b, e₁ ≠ e₂ ∧ ∀ e : edge b, e = e₁ ∨ e = e₂
axiom lateral_faces_corresponding_to_perpendicular_edges_are_rectangles : ∀ {b : base P} {e₁ e₂ : edge b}, (e₁ ⊥ lateral_edge e₁ ∧ e₂ ⊥ lateral_edge e₂) → is_rectangle (lateral_face e₁) ∧ is_rectangle (lateral_face e₂)
axiom other_edges_not_perpendicular_to_lateral_edges : ∀ {b : base P} {e₁ e₂ : edge b}, ¬(∃ e : edge b, e ≠ e₁ ∧ e ≠ e₂ ∧ e ⊥ lateral_edge e)

-- Goal
theorem max_rectangles_in_oblique_prism : 
  ∀ {P : Type} [ObliquePrism P], 4 = 
    # {f : face P // is_rectangle f} :=
sorry

end max_rectangles_in_oblique_prism_l599_599332


namespace max_value_l599_599739

theorem max_value (x y : ℝ) : 
  (x + 3 * y + 4) / (Real.sqrt (x ^ 2 + y ^ 2 + 4)) ≤ Real.sqrt 26 :=
by
  -- Proof should be here
  sorry

end max_value_l599_599739


namespace complex_solution_l599_599079

noncomputable def z : ℂ := 1 - complex.I

theorem complex_solution (h : z * (1 + complex.I) = 1 - complex.I) : z = -complex.I :=
sorry

end complex_solution_l599_599079


namespace vector_magnitude_sub_eq_sqrt38_l599_599057

def vector_a := (2 : ℝ, 1 : ℝ, 3 : ℝ)
def vector_b := (-4 : ℝ, 2 : ℝ, 2 : ℝ)

theorem vector_magnitude_sub_eq_sqrt38
    (h_perpendicular : (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 + vector_a.3 * vector_b.3 = 0)) :
    (real.sqrt ((vector_a.1 - vector_b.1)^2 + (vector_a.2 - vector_b.2)^2 + (vector_a.3 - vector_b.3)^2) = real.sqrt 38) :=
by
  sorry

end vector_magnitude_sub_eq_sqrt38_l599_599057


namespace find_13_genuine_coins_find_15_genuine_coins_find_17_genuine_coins_impossible_l599_599189

-- Define the conditions of the problem
def isBoard (b : Fin 5 × Fin 5 → bool) : Prop :=
∀ i j, ∃! (c : Fin 5 × Fin 5), b c

def isCoin (c : Fin 5 × Fin 5 → bool) : Prop :=
∀ i j, ∃! (p : Fin 5 × Fin 5), c p

def isCounterfeit (coins : Fin 5 × Fin 5 → bool) (counterfeit : Fin 5 × Fin 5 → bool) : Prop :=
  ∃ c₁ c₂ : Fin 5 × Fin 5, counterfeit c₁ ∧ counterfeit c₂ ∧ ¬coins c₁ ∧ ¬coins c₂ ∧ c₁ ≠ c₂ ∧
  ∃! (x : Fin 5 × Fin 5), x ≠ c₁ ∧ x ≠ c₂ ∧ coins x ∧ ¬counterfeit x

def shareVertex (c₁ c₂ : Fin 5 × Fin 5) : Prop :=
  (c₁.1 = c₂.1 ∨ c₁.1 + 1 = c₂.1 ∨ c₁.1 = c₂.1 + 1 ∨
   c₁.2 = c₂.2 ∨ c₁.2 + 1 = c₂.2 ∨ c₁.2 = c₂.2 + 1)

-- Given position of counterfeit coins
def counterfeitCoinsPlacedCorrectly (counterfeit : Fin 5 × Fin 5 → bool) : Prop :=
  ∃ c₁ c₂ : Fin 5 × Fin 5, counterfeit c₁ ∧ counterfeit c₂ ∧ shareVertex c₁ c₂

-- 1. Prove 13 genuine coins can be guaranteed
theorem find_13_genuine_coins (coins : Fin 5 × Fin 5 → bool) (counterfeit : Fin 5 × Fin 5 → bool)
  (h1 : isBoard coins) (h2 : isCoin coins)
  (h3 : isCounterfeit coins counterfeit)
  (h4 : counterfeitCoinsPlacedCorrectly counterfeit) :
  ∃ s : Finset (Fin 5 × Fin 5), s.card = 13 ∧ ∀ x ∈ s, ¬counterfeit x :=
sorry

-- 2. Prove 15 genuine coins can be guaranteed
theorem find_15_genuine_coins (coins : Fin 5 × Fin 5 → bool) (counterfeit : Fin 5 × Fin 5 → bool)
  (h1 : isBoard coins) (h2 : isCoin coins)
  (h3 : isCounterfeit coins counterfeit)
  (h4 : counterfeitCoinsPlacedCorrectly counterfeit) :
  ∃ s : Finset (Fin 5 × Fin 5), s.card = 15 ∧ ∀ x ∈ s, ¬counterfeit x :=
sorry

-- 3. Prove 17 genuine coins cannot be guaranteed
theorem find_17_genuine_coins_impossible (coins : Fin 5 × Fin 5 → bool) (counterfeit : Fin 5 × Fin 5 → bool)
  (h1 : isBoard coins) (h2 : isCoin coins)
  (h3 : isCounterfeit coins counterfeit)
  (h4 : counterfeitCoinsPlacedCorrectly counterfeit) :
  ¬∃ s : Finset (Fin 5 × Fin 5), s.card = 17 ∧ ∀ x ∈ s, ¬counterfeit x :=
sorry

end find_13_genuine_coins_find_15_genuine_coins_find_17_genuine_coins_impossible_l599_599189


namespace abs_eq_5_iff_l599_599105

theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end abs_eq_5_iff_l599_599105


namespace arithmetic_sequence_a6_l599_599503

-- Define the conditions in an arithmetic sequence
variable {a : ℕ → ℚ}
variable {d : ℚ}

-- Assume the sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + m * d

-- Given conditions
def a_three : a 3 = 4 := sorry
def a_seven : a 7 = 10 := sorry

-- Main statement: Prove that a_6 = 17 / 2
theorem arithmetic_sequence_a6 :
  (∀ n m, arithmetic_sequence a d) →
  a 3 = 4 →
  a 7 = 10 →
  a 6 = 17 / 2 :=
by
  intros h_arith a3 a7
  -- Expected Lean proof structure here
  sorry

end arithmetic_sequence_a6_l599_599503


namespace correct_statements_l599_599767

noncomputable def z : ℂ := -1/2 + (complex.I * (real.sqrt 3 / 2))

theorem correct_statements :
  z^3 = 1 ∧ z^2 + z + 1 = 0 :=
by sorry

end correct_statements_l599_599767


namespace melanie_attended_games_l599_599948

-- Define the total number of football games and the number of games missed by Melanie.
def total_games := 7
def missed_games := 4

-- Define what we need to prove: the number of games attended by Melanie.
theorem melanie_attended_games : total_games - missed_games = 3 := 
by
  sorry

end melanie_attended_games_l599_599948


namespace billiard_table_reflections_l599_599283

-- Define the coordinates for the center of the billiard table
def center := (1.5, 0.5)

-- Define the lengths of the table
def length := 3.0
def width := 1.0

-- Define the distance traveled by the billiard ball
def distance := 2.0

-- Calculate expected number of reflections
def expected_reflections : ℝ := 
  (2 / Real.pi) * (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

-- Problem statement to be proved
theorem billiard_table_reflections :
  ∃ (E : ℝ), E = expected_reflections := by
  use expected_reflections
  sorry

end billiard_table_reflections_l599_599283


namespace johns_change_l599_599153

theorem johns_change (n_sodas : ℕ) (cost_per_soda : ℕ) (total_paid : ℕ) (total_cost : ℕ) (change : ℕ) :
  n_sodas = 3 → 
  cost_per_soda = 2 → 
  total_paid = 20 → 
  total_cost = n_sodas * cost_per_soda → 
  change = total_paid - total_cost → 
  change = 14 :=
by
  intros n_sodas_eq cost_per_soda_eq total_paid_eq total_cost_eq change_eq
  rw [n_sodas_eq, cost_per_soda_eq, total_paid_eq] at total_cost_eq change_eq
  norm_num at total_cost_eq change_eq
  exact change_eq

end johns_change_l599_599153


namespace sum_of_m_and_n_l599_599518

noncomputable section

variable {a b m n : ℕ}

theorem sum_of_m_and_n 
  (h1 : a = n * b)
  (h2 : (a + b) = m * (a - b)) :
  m + n = 5 :=
sorry

end sum_of_m_and_n_l599_599518


namespace sum_of_coordinates_l599_599208

noncomputable def point_on_graph_of_y_eq_3f (f : ℝ → ℝ) : Prop :=
  (3, 5) ∈ set_of (λ p : ℝ × ℝ, p.2 = 3 * f p.1)

noncomputable def point_on_graph_of_y_eq_1_3f_inv (f : ℝ → ℝ) : Prop :=
  (5/3, 1) ∈ set_of (λ p : ℝ × ℝ, p.2 = (1 / 3 : ℝ) * (f⁻¹ p.1))

theorem sum_of_coordinates (f : ℝ → ℝ) (hf : bijective f)
    (h : point_on_graph_of_y_eq_3f f) :
  5 / 3 + 1 = 8 / 3 :=
by 
  rw [← h] 
  sorry

end sum_of_coordinates_l599_599208


namespace quadratic_root_conditions_l599_599937

theorem quadratic_root_conditions (a b : ℝ)
    (h1 : ∃ k : ℝ, ∀ x : ℝ, x^2 + 2 * x + 3 - k = 0)
    (h2 : ∀ α β : ℝ, α * β = 3 - k ∧ k^2 = α * β + 3 * k) : 
    k = 3 := 
sorry

end quadratic_root_conditions_l599_599937


namespace probability_four_of_eight_show_three_l599_599740

def probability_exactly_four_show_three : ℚ :=
  let num_ways := Nat.choose 8 4
  let prob_four_threes := (1 / 6) ^ 4
  let prob_four_not_threes := (5 / 6) ^ 4
  (num_ways * prob_four_threes * prob_four_not_threes)

theorem probability_four_of_eight_show_three :
  probability_exactly_four_show_three = 43750 / 1679616 :=
by 
  sorry

end probability_four_of_eight_show_three_l599_599740


namespace parallelogram_sides_eq_l599_599238

theorem parallelogram_sides_eq (x y : ℚ) :
  (5 * x - 2 = 10 * x - 4) → 
  (3 * y + 7 = 6 * y + 13) → 
  x + y = -1.6 := by
  sorry

end parallelogram_sides_eq_l599_599238


namespace neg_one_pow_neg_two_l599_599002

theorem neg_one_pow_neg_two : (-1 : ℝ) ^ (-2 : ℤ) = 1 := by
  -- Including the conditions
  have h1 : ∀ (a : ℝ) (n : ℤ), a ≠ 0 ∧ 0 < n → a ^ -n = 1 / (a ^ n) := by sorry
  have h2 : (-1 : ℝ) ^ 2 = 1 := by sorry

  -- The actual proof is not provided, but the statement encapsulates it
  sorry

end neg_one_pow_neg_two_l599_599002


namespace solve_for_m_l599_599484

theorem solve_for_m (x m : ℝ) (h1 : 2 * 1 - m = -3) : m = 5 :=
by
  sorry

end solve_for_m_l599_599484


namespace difference_in_pups_l599_599149

theorem difference_in_pups :
  let huskies := 5
  let pitbulls := 2
  let golden_retrievers := 4
  let pups_per_husky := 3
  let pups_per_pitbull := 3
  let total_adults := huskies + pitbulls + golden_retrievers
  let total_pups := total_adults + 30
  let total_husky_pups := huskies * pups_per_husky
  let total_pitbull_pups := pitbulls * pups_per_pitbull
  let H := pups_per_husky
  let D := (total_pups - total_husky_pups - total_pitbull_pups - 3 * golden_retrievers) / golden_retrievers
  D = 2 := sorry

end difference_in_pups_l599_599149


namespace latticePointsCount_l599_599864

-- Define what a lattice point is and the condition
def isLatticePoint (x y z : ℤ) : Prop := x^2 + y^2 + z^2 = 16

-- The theorem stating the number of such lattice points
theorem latticePointsCount :
  (Finset.univ : Finset (ℤ × ℤ × ℤ)).filter (λ (p : ℤ × ℤ × ℤ), isLatticePoint p.1 p.2 p.3).card = 50 :=
  sorry

end latticePointsCount_l599_599864


namespace net_rate_of_pay_l599_599682

theorem net_rate_of_pay 
  (time_hours : ℝ)
  (speed_mph : ℝ)
  (fuel_efficiency_mpg : ℝ)
  (earnings_per_mile : ℝ)
  (cost_per_gallon : ℝ) 
  (h_time : time_hours = 3)
  (h_speed : speed_mph = 65)
  (h_fuel_efficiency : fuel_efficiency_mpg = 30)
  (h_earnings : earnings_per_mile = 0.60) 
  (h_cost : cost_per_gallon = 1.80) :
  let total_distance := speed_mph * time_hours in
  let total_gasoline_used := total_distance / fuel_efficiency_mpg in
  let total_earnings := earnings_per_mile * total_distance in
  let total_cost := cost_per_gallon * total_gasoline_used in
  (total_earnings - total_cost) / time_hours = 35.10 :=
by
  sorry

end net_rate_of_pay_l599_599682


namespace find_matrix_and_new_curve_l599_599788

noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![6, 2], ![4, 4]]

def eigenvalue (λ : ℝ) (v : Vector (Fin 2) ℝ) (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M.mul_vec v = λ • v

def transforms_point (M : Matrix (Fin 2) (Fin 2) ℝ) (p1 p2 : Vector (Fin 2) ℝ) : Prop :=
  M.mul_vec p1 = p2

theorem find_matrix_and_new_curve :
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, 
    eigenvalue 8 (Vector.ofList [1, 1]) M ∧ 
    transforms_point M (Vector.ofList [-1, 3]) (Vector.ofList [0, 8]) ∧ 
    M = ![![6, 2], ![4, 4]] ∧ 
    ∀ x' y' : ℝ, x' - 2 * y' + 4 = 0 ∧ ∃ x y : ℝ, 8 * x + 8 * y = x' ∧ 8 * y + 4 * x = y' ∧ x + 3 * y - 2 = 0 :=
by
  sorry

end find_matrix_and_new_curve_l599_599788


namespace earnings_difference_l599_599978

noncomputable def investment_ratio_a : ℕ := 3
noncomputable def investment_ratio_b : ℕ := 4
noncomputable def investment_ratio_c : ℕ := 5

noncomputable def return_ratio_a : ℕ := 6
noncomputable def return_ratio_b : ℕ := 5
noncomputable def return_ratio_c : ℕ := 4

noncomputable def total_earnings : ℕ := 2900

noncomputable def earnings_a (x y : ℕ) : ℚ := (investment_ratio_a * return_ratio_a * x * y) / 100
noncomputable def earnings_b (x y : ℕ) : ℚ := (investment_ratio_b * return_ratio_b * x * y) / 100

theorem earnings_difference (x y : ℕ) (h : (investment_ratio_a * return_ratio_a * x * y + investment_ratio_b * return_ratio_b * x * y + investment_ratio_c * return_ratio_c * x * y) / 100 = total_earnings) :
  earnings_b x y - earnings_a x y = 100 := by
  sorry

end earnings_difference_l599_599978


namespace janet_saves_minutes_l599_599376

theorem janet_saves_minutes 
  (key_search_time : ℕ := 8) 
  (complaining_time : ℕ := 3) 
  (days_in_week : ℕ := 7) : 
  (key_search_time + complaining_time) * days_in_week = 77 := 
by
  sorry

end janet_saves_minutes_l599_599376


namespace initial_apples_l599_599585

theorem initial_apples (Initially_Apples : ℕ) (Added_Apples : ℕ) (Total_Apples : ℕ)
  (h1 : Added_Apples = 8) (h2 : Total_Apples = 17) : Initially_Apples = 9 :=
by
  have h3 : Added_Apples + Initially_Apples = Total_Apples := by
    sorry
  linarith

end initial_apples_l599_599585


namespace combinatorial_identity_l599_599717

theorem combinatorial_identity (n : ℕ) :
  let C := λ n k : ℕ, n.choose k in
  (C n 1 + 2^2 * C n 2 + 3^2 * C n 3 + ... + n^2 * C n n) = n * (n + 1) * 2^(n-2) := 
sorry

end combinatorial_identity_l599_599717


namespace total_chess_games_l599_599625

noncomputable def numberOfGames (n : ℕ) (k : ℕ) : ℕ :=
  if k = n-1 then n * k / 2 else 0

theorem total_chess_games {n : ℕ} (hn : n = 4) :
  numberOfGames n 3 = 6 :=
by
  rw [hn]
  dsimp [numberOfGames]
  norm_num
  sorry

end total_chess_games_l599_599625


namespace ellipse_equation_range_PF1_dot_PF2_line_PQ_fixed_point_l599_599423

variable (P : ℝ × ℝ) (A B M Q : ℝ × ℝ)

def ellipse (a b : ℝ) := {P : ℝ × ℝ | P.1^2 / a^2 + P.2^2 / b^2 = 1}
def major_axis_length (a : ℝ) := 2 * a
def foci (F1 F2 : ℝ × ℝ) := dist F1 F2 = 2 * sqrt 3

theorem ellipse_equation (a b : ℝ) (ha_gt_b : a > b) (hb_pos : b > 0) (hmajor_axis : major_axis_length a = 4)
  (hfoci : ∃ F1 F2 : ℝ × ℝ, foci F1 F2) :
  a = 2 ∧ b = 1 ∧ ellipse a b = {P : ℝ × ℝ | P.1^2 / 4 + P.2^2 = 1} :=
sorry

theorem range_PF1_dot_PF2 (x₀ y₀ a b : ℝ) (hfoci : ∃ F1 F2 : ℝ × ℝ, foci F1 F2) (hP_ellipse : ellipse a b (x₀, y₀)) :
  -2 ≤ x₀^2 * 3 / 4 - 2 ∧ x₀^2 * 3 / 4 - 2 ≤ 1 :=
sorry

theorem line_PQ_fixed_point (t : ℝ) (hline_l : M = (4, t)) (hA : A = (-2, 0)) (hB : B = (2, 0)) :
  ∃! Q : ℝ × ℝ, Q ∈ ellipse 2 1 → (P, Q) ∈ {L : ℝ × ℝ | L.2 = 2 * t / (3 - t^2) * (L.1 - 1)} ∧ L = (1, 0) :=
sorry

end ellipse_equation_range_PF1_dot_PF2_line_PQ_fixed_point_l599_599423


namespace ratio_distances_l599_599070

noncomputable def F : ℝ × ℝ := (1/2, 0)

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 2 * x

noncomputable def directrix (x : ℝ) : Prop := x = -1 / 2

structure Point (x y : ℝ) : Prop := (x_val : ℝ) (y_val : ℝ)

-- Points A and B lie on the parabola
noncomputable def on_parabola (A B : Point) : Prop := 
parabola A.x_val A.y_val ∧ parabola B.x_val B.y_val

-- Point P lies on the directrix
noncomputable def on_directrix (P : Point) : Prop := 
directrix P.x_val

-- A is the midpoint of PB
noncomputable def midpoint (A P B : Point) : Prop := 
2 * A.x_val = B.x_val + P.x_val ∧ 2 * A.y_val = B.y_val + P.y_val

def is_focus (F : Point) := F = ⟨1/2, 0⟩

theorem ratio_distances (A B P : Point) (hF : is_focus F)
  (hparabola : on_parabola A B) (hdir : on_directrix P) (hmid : midpoint A P B) :
  (|B.x_val - F.1| / |A.x_val - F.1|) = 2 := 
sorry

end ratio_distances_l599_599070


namespace sqrt_one_plus_x_lt_one_add_half_x_l599_599214

theorem sqrt_one_plus_x_lt_one_add_half_x (x : ℝ) (h : x > 0) : sqrt (1 + x) < 1 + x / 2 :=
sorry

end sqrt_one_plus_x_lt_one_add_half_x_l599_599214


namespace telescoping_series_sum_l599_599725

theorem telescoping_series_sum :
  (∑' (n : ℕ) in (Finset.range (0) \ Finset.singleton (0)), (↑(3 * n + 2) / (↑n * (↑n + 1) * (↑n + 3)))) = (5 / 6) := sorry

end telescoping_series_sum_l599_599725


namespace carols_father_gave_5_peanuts_l599_599357

theorem carols_father_gave_5_peanuts : 
  ∀ (c: ℕ) (f: ℕ), c = 2 → c + f = 7 → f = 5 :=
by
  intros c f h1 h2
  sorry

end carols_father_gave_5_peanuts_l599_599357


namespace division_with_missing_digits_l599_599640

theorem division_with_missing_digits :
  ∃ Q : ℕ, ∃ D : ℕ, D = 378 ∧ Q = 5243 ∧ 1981854 = D * Q :=
by
  use 5243
  use 378
  split
  sorry

end division_with_missing_digits_l599_599640


namespace original_length_before_sharpening_l599_599517

/-- Define the current length of the pencil after sharpening -/
def current_length : ℕ := 14

/-- Define the length of the pencil that was sharpened off -/
def sharpened_off_length : ℕ := 17

/-- Prove that the original length of the pencil before sharpening was 31 inches -/
theorem original_length_before_sharpening : current_length + sharpened_off_length = 31 := by
  sorry

end original_length_before_sharpening_l599_599517


namespace paco_ate_sweet_cookies_l599_599574

noncomputable def PacoCookies (sweet: Nat) (salty: Nat) (salty_eaten: Nat) (extra_sweet: Nat) : Nat :=
  let corrected_salty_eaten := if salty_eaten > salty then salty else salty_eaten
  corrected_salty_eaten + extra_sweet

theorem paco_ate_sweet_cookies : PacoCookies 39 6 23 9 = 15 := by
  sorry

end paco_ate_sweet_cookies_l599_599574


namespace athlete_heartbeats_l599_599706

def heart_beats_per_minute : ℕ := 120
def running_pace_minutes_per_mile : ℕ := 6
def race_distance_miles : ℕ := 30
def total_heartbeats : ℕ := 21600

theorem athlete_heartbeats :
  (running_pace_minutes_per_mile * race_distance_miles * heart_beats_per_minute) = total_heartbeats :=
by
  sorry

end athlete_heartbeats_l599_599706


namespace order_of_product_l599_599880

theorem order_of_product {a b n : ℕ} (h₁ : Nat.coprime a n) (h₂ : Nat.coprime b n)
  (ωa ωb : ℕ) (h_order_a : (a^ωa) % n = 1) (h_order_b : (b^ωb) % n = 1) (h_coprime : Nat.coprime ωa ωb) :
  (orderOf (a * b) n) = ωa * ωb := 
sorry

end order_of_product_l599_599880


namespace exp_funcs_linearly_independent_l599_599913

theorem exp_funcs_linearly_independent
  (k1 k2 k3 : ℝ)
  (h_distinct : k1 ≠ k2 ∧ k1 ≠ k3 ∧ k2 ≠ k3) :
  ¬ ∃ (α1 α2 α3 : ℝ), (α1 ≠ 0 ∨ α2 ≠ 0 ∨ α3 ≠ 0) ∧ (∀ x : ℝ, α1 * exp (k1 * x) + α2 * exp (k2 * x) + α3 * exp (k3 * x) = 0) :=
sorry

end exp_funcs_linearly_independent_l599_599913


namespace smallest_n_for_partition_l599_599167

theorem smallest_n_for_partition (n : ℕ) (h : n ≥ 2)
  (T : finset ℕ := finset.range (n + 1) \ {0, 1}) :
  (∀ (A B : finset ℕ), A ∪ B = T → A ∩ B = ∅ →
    (∃ a b c ∈ A, a * b = c) ∨ (∃ a b c ∈ B, a * b = c)) ↔ n ≥ 256 :=
sorry

end smallest_n_for_partition_l599_599167


namespace anja_equal_integers_l599_599711

theorem anja_equal_integers (S : Finset ℤ) (h_card : S.card = 2014)
  (h_mean : ∀ (x y z : ℤ), x ∈ S → y ∈ S → z ∈ S → (x + y + z) / 3 ∈ S) :
  ∃ k, ∀ x ∈ S, x = k :=
sorry

end anja_equal_integers_l599_599711


namespace geometric_sequence_fifth_term_l599_599606

theorem geometric_sequence_fifth_term (a₁ a₇ : ℝ) (r : ℝ) (h₁ : a₁ = 729) (h₇ : a₇ = 64) (hr : a₇ = a₁ * r^6) :
    a₁ * r^4 = 144 :=
by
  -- Given
  rw [h₁, h₇] at hr
  rw hr at *
  sorry

end geometric_sequence_fifth_term_l599_599606


namespace minimal_storing_capacity_l599_599303

-- Define the initial conditions
def familyGeneratesLaundryPerWeek : ℕ := 10
def bins : Fin 3 → ℕ := λ _, 0  -- Initial bins are empty

-- Define the function for each week's process
def addLaundry (bins : Fin 3 → ℕ) (f : Fin 3 → ℕ) : Fin 3 → ℕ :=
  λ i, bins i + f i

def heaviestBin (bins : Fin 3 → ℕ) : Fin 3 :=
  Fin.ofNat (Fintype.max (finset.univ : Finset (Fin 3)) (λ i, bins i))

theorem minimal_storing_capacity :
  (∀ bins : Fin 3 → ℕ, 
    bins = (λ _, 0) → 
    (∀ add_func : Fin 3 → ℕ, 
      (∀ i, add_func i ≥ 0) →
      (add_func 0 + add_func 1 + add_func 2 = familyGeneratesLaundryPerWeek) →
      let bins_update := addLaundry bins add_func in
      let to_empty := heaviestBin bins_update in
        storage_capacity_needed bins_update to_empty)) 
  → 25 :=
sorry

end minimal_storing_capacity_l599_599303


namespace math_problem_l599_599800

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := (1 - sin (2 * x)) / cos x

-- The proof problem statement in Lean 4
theorem math_problem (x : ℝ) (α : ℝ) (h1 : ∃ k : ℤ, x ≠ k * π + π / 2) (h2 : quadrant_fourth α) (h3 : tan α = -4/3) :
    -- The domain of f(x)
    (∀ x, x ∈ {x | ∃ k : ℤ, x ≠ k * π + π / 2}) ∧
    -- Value of f(α)
    f α = 49 / 15 :=
by 
  sorry -- Proof to be completed

end math_problem_l599_599800


namespace janet_saves_time_l599_599374

theorem janet_saves_time (looking_time_per_day : ℕ := 8) (complaining_time_per_day : ℕ := 3) (days_per_week : ℕ := 7) :
  (looking_time_per_day + complaining_time_per_day) * days_per_week = 77 := 
sorry

end janet_saves_time_l599_599374


namespace abs_eq_five_l599_599098

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  sorry

end abs_eq_five_l599_599098


namespace max_value_of_varphi_l599_599807

noncomputable def function_f (x varphi : ℝ) : ℝ :=
  sin (3*x + 3*varphi) - 2*sin (x + varphi) * cos (2*x + 2*varphi)

theorem max_value_of_varphi
  (h_cond : ∀ x : ℝ, function_f x varphi = sin (x + varphi))
  (h_interval : ∀ x : ℝ, (π / 6) < x ∧ x < (2 * π / 3) → function_f x varphi = sin (x + varphi))
  (h_monotone : monotone_decreasing_on (function_f x varphi) (π / 6) (2 * π / 3))
  (h_bound : |varphi| < π) :
  varphi ≤ 5 * π / 6 :=
sorry

end max_value_of_varphi_l599_599807


namespace no_valid_cross_exists_l599_599667

-- Define a function that calculates the value at position (i, j) 
-- based on the given rules
def table_value (i j : ℕ) : ℕ := 70 * i + j + 1

theorem no_valid_cross_exists : 
  ¬ (∃ x, 1 ≤ x ∧ x ≤ 4900 ∧ (x + x - 70 + x + 70 + x - 1 + x + 1 = 2018)) :=
begin
  -- Start with the assumption that such an x exists
  intro h,
  cases h with x hx,
  cases hx with h1 h,
  cases h with h2 h3,

  -- Simplify the sum of the cross:
  have h4 : 5 * x = 2018,
    calc 5 * x = x + (x - 70) + (x + 70) + (x - 1) + (x + 1) : by ring
           ... = 2018 : h3,

  -- Solve for x
  have h5 : x = 403.6, from (nat.div_eq_of_eq_mul_right zero_lt_five.symm h4.symm),

  -- x must be an integer, contradiction
  exfalso,
  linarith,
end

end no_valid_cross_exists_l599_599667


namespace sum_of_distances_between_15_and_16_l599_599144

-- Define the setup and conditions
def point (α : Type) := prod α α

def B : point ℝ := (0, 0)
def D : point ℝ := (3, 4)
def A : point ℝ := (13, 0)

-- Functions to calculate distances
def dist (p q : point ℝ) : ℝ :=
  real.sqrt (((p.1 - q.1) ^ 2) + ((p.2 - q.2) ^ 2))

def BD_distance := dist B D
def AD_distance := dist A D

noncomputable def sum_distances := BD_distance + AD_distance

-- Proof goal
theorem sum_of_distances_between_15_and_16 :
  15 < sum_distances ∧ sum_distances < 16 := by
  sorry

end sum_of_distances_between_15_and_16_l599_599144


namespace largest_unique_digit_number_is_7089_l599_599041

/--
Prove that the largest four-digit number where all digits are distinct,
and no two digits can be swapped to form a smaller number, is 7089.
-/
theorem largest_unique_digit_number_is_7089 :
  ∃ (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ 
             (∀ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l →
              i*1000 + j*100 + k*10 + l = n) ∧ 
             (∀ a b, (a ≠ b) → ∀ c d, (c ≠ d) → 
              (a ≠ c) → (a ≠ d) → (b ≠ c) → (b ≠ d) →
              (a*1000 + b*100 + c*10 + d > n ∧ a*1000 + b*100 + c*10 + d ≤ 9876)) :=
  ∃ (n : ℕ), n = 7089

end largest_unique_digit_number_is_7089_l599_599041


namespace find_f_80_l599_599224

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_relation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  f (x * y) = f x / y^2

axiom f_40 : f 40 = 50

-- Proof that f 80 = 12.5
theorem find_f_80 : f 80 = 12.5 := 
by
  sorry

end find_f_80_l599_599224


namespace trajectory_of_M_l599_599427

noncomputable def P : ℝ × ℝ := (2, 2)
noncomputable def circleC (x y : ℝ) : Prop := x^2 + y^2 - 8 * y = 0
noncomputable def isMidpoint (A B M : ℝ × ℝ) : Prop := M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def isIntersectionPoint (l : ℝ × ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, circleC x y ∧ l (x, y) ∧ ((A = (x, y)) ∨ (B = (x, y))) 

theorem trajectory_of_M (M : ℝ × ℝ) : 
  (∃ A B : ℝ × ℝ, isIntersectionPoint (fun p => ∃ k : ℝ, p = (k, k)) A B ∧ isMidpoint A B M) →
  (M.1 - 1)^2 + (M.2 - 3)^2 = 2 := 
sorry

end trajectory_of_M_l599_599427


namespace sum_of_their_ages_now_l599_599843

variable (Nacho Divya : ℕ)

-- Conditions
def divya_current_age := 5
def nacho_in_5_years := 3 * (divya_current_age + 5)

-- Definition to determine current age of Nacho
def nacho_current_age := nacho_in_5_years - 5

-- Sum of current ages
def sum_of_ages := divya_current_age + nacho_current_age

-- Theorem to prove the sum of their ages now is 30
theorem sum_of_their_ages_now : sum_of_ages = 30 :=
by
  sorry

end sum_of_their_ages_now_l599_599843


namespace angle_bisector_passes_through_circumcenter_l599_599135

theorem angle_bisector_passes_through_circumcenter (A B C : Type) [triangle A B C] 
  (acute_angled : acute_triangle A B C) (angle_A_eq_60 : angle A = 60) :
  ∃O, is_circumcenter O A B C ∧ passes_through_bisector_of_altitudes O A B C :=
sorry

end angle_bisector_passes_through_circumcenter_l599_599135


namespace complex_quadrant_l599_599553

variable {x y : ℝ}

theorem complex_quadrant (h : (x + y) + (y - 1) * complex.i = (2 * x + 3 * y) + (2 * y + 1) * complex.i) : 
  x = 4 ∧ y = -2 := sorry

end complex_quadrant_l599_599553


namespace find_x0_l599_599763

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem find_x0 (x0 : ℝ) (h : f' x0 = 2) : x0 = Real.exp 1 :=
by {
  sorry
}

end find_x0_l599_599763


namespace angle_of_inclination_45_l599_599022

def plane (x y z : ℝ) : Prop := (x = y) ∧ (y = z)
def image_planes (x y : ℝ) : Prop := (x = 45 ∧ y = 45)

theorem angle_of_inclination_45 (t₁₂ : ℝ) :
  ∃ θ: ℝ, (plane t₁₂ t₁₂ t₁₂ → image_planes 45 45 → θ = 45) :=
sorry

end angle_of_inclination_45_l599_599022


namespace cricket_average_l599_599679

theorem cricket_average (x : ℕ) (h : 20 * x + 158 = 21 * (x + 6)) : x = 32 :=
by
  sorry

end cricket_average_l599_599679


namespace a_n_equals_yn_sq_plus_7_by_xn_minus_yn_l599_599457

-- Define the sequence a_n
def a : ℕ → ℤ
| 0       := 4
| 1       := 22
| (n + 2) := 6 * a (n + 1) - a n

-- Define the sequences x_n and y_n
def y : ℕ → ℤ
| 0       := 1
| 1       := 9
| (n + 2) := 6 * y (n + 1) - y n

def x : ℕ → ℤ
| n       := y n + (if n = 0 then 4 else a (n - 1))

theorem a_n_equals_yn_sq_plus_7_by_xn_minus_yn (n : ℕ) :
  a n = (y n ^ 2 + 7) / (x n - y n) := by
  sorry

end a_n_equals_yn_sq_plus_7_by_xn_minus_yn_l599_599457


namespace find_given_seashells_l599_599408

-- defining the conditions using variables
variables (initial_seashells given_to_Jessica remaining_seashells : ℕ)

-- stating the problem
theorem find_given_seashells (h1 : initial_seashells = 47) 
  (h2 : remaining_seashells = 22) : given_to_Jessica = 25 :=
by
  -- we state the relationship and use the assumptions to prove the answer
  have h : given_to_Jessica = initial_seashells - remaining_seashells, from nat.sub_eq_of_eq_add sorry,
  rw [h1, h2] at h,
  norm_num at h,
  exact h

end find_given_seashells_l599_599408


namespace anya_possible_wins_l599_599342

-- Define the total rounds played
def total_rounds := 25

-- Define Anya's choices
def anya_rock := 12
def anya_scissors := 6
def anya_paper := 7

-- Define Borya's choices
def borya_rock := 13
def borya_scissors := 9
def borya_paper := 3

-- Define the relationships in rock-paper-scissors game
def rock_beats_scissors := true
def scissors_beat_paper := true
def paper_beats_rock := true

-- Define no draws condition
def no_draws := total_rounds = anya_rock + anya_scissors + anya_paper ∧ total_rounds = borya_rock + borya_scissors + borya_paper

-- Proof problem statement
theorem anya_possible_wins : anya_rock + anya_scissors + anya_paper = total_rounds ∧
                             borya_rock + borya_scissors + borya_paper = total_rounds ∧
                             rock_beats_scissors ∧ scissors_beat_paper ∧ paper_beats_rock ∧
                             no_draws →
                             (9 + 3 + 7 = 19) := by
  sorry

end anya_possible_wins_l599_599342


namespace set_equality_l599_599619

noncomputable def alpha_set : Set ℝ := {α | ∃ k : ℤ, α = k * Real.pi / 2 - Real.pi / 5 ∧ (-Real.pi < α ∧ α < Real.pi)}

theorem set_equality : alpha_set = {-Real.pi / 5, -7 * Real.pi / 10, 3 * Real.pi / 10, 4 * Real.pi / 5} :=
by
  -- proof omitted
  sorry

end set_equality_l599_599619


namespace parallel_line_plane_l599_599435

variable (Line : Type) (Plane : Type)
variable (m n : Line) (α β γ: Plane)
variables [DifferentLines m n] [DifferentPlanes α β γ]

axiom perp (x y : Plane) : Prop
axiom parallel (x y : Line) : Prop
axiom inPlane (x : Line) (y : Plane) : Prop
noncomputable def intersection (x y : Plane) : Line := sorry

theorem parallel_line_plane {α β : Plane} {m n : Line}
  (h1 : intersection α β = m)
  (h2 : parallel n m)
  (h3 : ¬ inPlane n α)
  (h4 : ¬ inPlane n β) :
  parallel n α ∧ parallel n β :=
sorry

end parallel_line_plane_l599_599435


namespace range_of_expression_l599_599465

noncomputable def range_expression (a b : ℝ) : ℝ := a^2 + b^2 - 6 * a - 8 * b

variables (a b : ℝ)

def circle1 (a : ℝ) : Prop := ∀ x y : ℝ, (x - a)^2 + y^2 = 1

def circle2 (b : ℝ) : Prop := ∀ x y : ℝ, x^2 + y^2 - 2 * b * y + b^2 - 4 = 0

theorem range_of_expression :
  (circle1 a ∧ circle2 b ∧ ∃ (a b : ℝ), sqrt (a^2 + b^2) = 3) → -21 ≤ range_expression a b ∧ range_expression a b ≤ 39 := 
sorry

end range_of_expression_l599_599465


namespace derivative_at_t_zero_l599_599387

open Real

noncomputable def x (t : ℝ) : ℝ := exp t * cos t
noncomputable def y (t : ℝ) : ℝ := exp t * sin t

theorem derivative_at_t_zero : 
  deriv (λ t, y t) 0 / deriv (λ t, x t) 0 = 1 := 
by 
  sorry

end derivative_at_t_zero_l599_599387


namespace width_of_grassy_plot_l599_599321

-- Definitions
def length_plot : ℕ := 110
def width_path : ℝ := 2.5
def cost_per_sq_meter : ℝ := 0.50
def total_cost : ℝ := 425

-- Hypotheses and Target Proposition
theorem width_of_grassy_plot (w : ℝ) 
  (h1 : length_plot = 110)
  (h2 : width_path = 2.5)
  (h3 : cost_per_sq_meter = 0.50)
  (h4 : total_cost = 425)
  (h5 : (length_plot + 2 * width_path) * (w + 2 * width_path) = 115 * (w + 5))
  (h6 : 110 * w = 110 * w)
  (h7 : (115 * (w + 5) - (110 * w)) = total_cost / cost_per_sq_meter) :
  w = 55 := 
sorry

end width_of_grassy_plot_l599_599321


namespace find_principal_amount_l599_599324

noncomputable def principal_amount (SI R T : ℝ) : ℝ :=
  SI / (R * T / 100)

theorem find_principal_amount :
  principal_amount 4052.25 9 5 = 9005 := by
sorry

end find_principal_amount_l599_599324


namespace base_conversion_positive_b_l599_599219

theorem base_conversion_positive_b :
  (∃ (b : ℝ), 3 * 5^1 + 2 * 5^0 = 17 ∧ 1 * b^2 + 2 * b^1 + 0 * b^0 = 17 ∧ b = -1 + 3 * Real.sqrt 2) :=
by
  sorry

end base_conversion_positive_b_l599_599219


namespace solve_abs_eq_l599_599917

theorem solve_abs_eq {x : ℝ} (h₁ : x ≠ 3 ∧ (x >= 3 ∨ x < 3)) :
  (|x - 3| = 5 - 2 * x) ↔ x = 2 :=
by
  split;
  intro h;
  sorry

end solve_abs_eq_l599_599917


namespace tan_of_angle_l599_599782

open Real

-- Given conditions in the problem
variables {α : ℝ}

-- Define the given conditions
def sinα_condition (α : ℝ) : Prop := sin α = 3 / 5
def α_in_quadrant_2 (α : ℝ) : Prop := π / 2 < α ∧ α < π

-- Define the Lean statement
theorem tan_of_angle {α : ℝ} (h1 : sinα_condition α) (h2 : α_in_quadrant_2 α) :
  tan α = -3 / 4 :=
sorry

end tan_of_angle_l599_599782


namespace minimal_abs_diff_of_ab_equation_l599_599113

theorem minimal_abs_diff_of_ab_equation :
  ∃ a b : ℕ, (a > 0 ∧ b > 0 ∧ ab - 5a + 6b = 159 ∧ |a - b| = 29) 
:= sorry

end minimal_abs_diff_of_ab_equation_l599_599113


namespace probability_same_color_is_117_200_l599_599128

/-- There are eight green balls, five red balls, and seven blue balls in a bag. 
    A ball is taken from the bag, its color recorded, then placed back in the bag.
    A second ball is taken and its color recorded. -/
def probability_two_balls_same_color : ℚ :=
  let pGreen := (8 : ℚ) / 20
  let pRed := (5 : ℚ) / 20
  let pBlue := (7 : ℚ) / 20
  pGreen^2 + pRed^2 + pBlue^2

theorem probability_same_color_is_117_200 : probability_two_balls_same_color = 117 / 200 := by
  sorry

end probability_same_color_is_117_200_l599_599128


namespace range_of_omega_l599_599805

noncomputable def f (ω x : ℝ) : ℝ := cos (ω * x) ^ 2 + 2 * sin (ω * x) * cos (ω * x) - sin (ω * x) ^ 2

theorem range_of_omega (ω : ℝ) :
  (∀ x, ω > 0 → (∀ x ∈ set.Ioc (π / 12) (π / 3), ∃ M, f ω x = M ∧ (∀ y ∈ set.Ioc (π / 12) (π / 3), M ≥ f ω y))) →
  (3 / 8 < ω ∧ ω < 3 / 2) :=
sorry

end range_of_omega_l599_599805


namespace range_of_a_l599_599066

def p (a : ℝ) : Prop := a ≤ -4 ∨ a ≥ 4
def q (a : ℝ) : Prop := a ≥ -12
def either_p_or_q_but_not_both (a : ℝ) : Prop := (p a ∧ ¬ q a) ∨ (¬ p a ∧ q a)

theorem range_of_a :
  {a : ℝ | either_p_or_q_but_not_both a} = {a : ℝ | (-4 < a ∧ a < 4) ∨ a < -12} :=
sorry

end range_of_a_l599_599066


namespace max_value_l599_599962

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  -3 * x^2 + 18 * x - 5

theorem max_value : ∃ x : ℝ, quadratic_function x = 22 :=
sorry

end max_value_l599_599962


namespace articles_sold_at_cost_price_l599_599125

variable (C S : ℝ)
variable (X : ℝ)
variable (gain_percent : ℝ := 56.25)

-- Conditions
hypothesis h1 : X * C = 32 * S
hypothesis h2 : S = C * 1.5625

-- Conclusion to prove
theorem articles_sold_at_cost_price (h1 : X * C = 32 * S) (h2 : S = C * 1.5625) : X = 50 :=
by
  sorry

end articles_sold_at_cost_price_l599_599125


namespace range_of_a_l599_599168

variable {ℝ : Type*}

/-- Define the proposition p(x) -/
def p (a : ℝ) (x : ℝ) : Prop := a * x^2 + 2 * x + 1 > 0

/-- Main theorem -/
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, p a x) : a > 1 :=
by
  sorry

end range_of_a_l599_599168


namespace derivative_f_intervals_of_monotonicity_extrema_l599_599798

noncomputable def f (x : ℝ) := (x + 1)^2 * (x - 1)

theorem derivative_f (x : ℝ) : deriv f x = 3 * x^2 + 2 * x - 1 := sorry

theorem intervals_of_monotonicity :
  (∀ x, x < -1 → deriv f x > 0) ∧
  (∀ x, -1 < x ∧ x < -1/3 → deriv f x < 0) ∧
  (∀ x, x > -1/3 → deriv f x > 0) := sorry

theorem extrema :
  f (-1) = 0 ∧
  f (-1/3) = -(32 / 27) := sorry

end derivative_f_intervals_of_monotonicity_extrema_l599_599798


namespace total_games_High_School_Nine_l599_599131

-- Define the constants and assumptions.
def num_teams := 9
def games_against_non_league := 6

-- Calculation of the number of games within the league.
def games_within_league := (num_teams * (num_teams - 1) / 2) * 2

-- Calculation of the number of games against non-league teams.
def games_non_league := num_teams * games_against_non_league

-- The total number of games.
def total_games := games_within_league + games_non_league

-- The statement to prove.
theorem total_games_High_School_Nine : total_games = 126 := 
by
  -- You do not need to provide the proof.
  sorry

end total_games_High_School_Nine_l599_599131


namespace range_of_a_l599_599444

-- Definitions of the given functions
def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

-- Condition stating that f is monotonically increasing on (0, +∞)
def is_monotonically_increasing (a : ℝ) : Prop :=
  ∀ x > 0, x * Real.log x + 1 + (1 / x) - a ≥ 0

-- Condition stating the inequality (x-1)f(x) ≥ 0 is always true
def inequality_condition (a : ℝ) : Prop :=
  ∀ x > 0, (x - 1) * f x a ≥ 0

-- The final statement to prove the range of a under the given conditions
theorem range_of_a (a : ℝ) (h1 : is_monotonically_increasing a) (h2 : inequality_condition a) :
  0 < a ∧ a ≤ 2 := sorry

end range_of_a_l599_599444


namespace symm_y_axis_l599_599930

noncomputable def f (x : ℝ) : ℝ := abs x

theorem symm_y_axis (x : ℝ) : f (-x) = f (x) := by
  sorry

end symm_y_axis_l599_599930


namespace inequality_inequality_l599_599367

theorem inequality_inequality (a b c d : ℝ) (h1 : a^2 + b^2 = 4) (h2 : c^2 + d^2 = 16) :
  ac + bd ≤ 8 :=
sorry

end inequality_inequality_l599_599367


namespace rectangle_cut_possible_l599_599016

theorem rectangle_cut_possible (a b : ℕ) (h1 : a * b > 0) : 
  ∃ rectangles : list (ℕ × ℕ), 
  rectangles.length = 18 ∧ 
  (∀ (r1 r2 : ℕ × ℕ), r1 ∈ rectangles → r2 ∈ rectangles → 
  r1 ≠ r2 → disjoint r1 r2) ∧ 
  (∀ (r1 r2 : ℕ × ℕ), r1 ∈ rectangles → r2 ∈ rectangles → 
  adjacent r1 r2 → ¬ forms_larger_rectangle r1 r2) :=
sorry

end rectangle_cut_possible_l599_599016


namespace spinner_prob_C_l599_599995

theorem spinner_prob_C (P_A P_B P_C : ℚ) (h_A : P_A = 1/3) (h_B : P_B = 5/12) (h_total : P_A + P_B + P_C = 1) : 
  P_C = 1/4 := 
sorry

end spinner_prob_C_l599_599995


namespace range_of_expression_l599_599463

theorem range_of_expression (a b : ℝ) (h : a^2 + b^2 = 9) :
  -21 ≤ a^2 + b^2 - 6*a - 8*b ∧ a^2 + b^2 - 6*a - 8*b ≤ 39 :=
by
  sorry

end range_of_expression_l599_599463


namespace distance_is_20_km_l599_599981

noncomputable def distance_from_home_to_school (x : ℝ) :=
  let walking_time := x / 5
  let walking_one_third_time := (x / 3) / 5
  let bus_time := (2 * x / 3) / 20
  let combined_time := walking_one_third_time + bus_time
  walking_time = combined_time + 2

theorem distance_is_20_km : distance_from_home_to_school 20 :=
by
  rfl

end distance_is_20_km_l599_599981


namespace max_a4_l599_599062

variable (a1 d : ℝ)

theorem max_a4 (h1 : 2 * a1 + 6 * d ≥ 10) (h2 : 2.5 * a1 + 10 * d ≤ 15) :
  ∃ max_a4, max_a4 = 4 ∧ a1 + 3 * d ≤ max_a4 :=
by
  sorry

end max_a4_l599_599062


namespace probability_both_meat_given_same_l599_599896

open ProbabilityTheory

-- Definition of the problem conditions
def total_dumplings : Finset (Fin 5) := Finset.univ
def meat_dumplings : Finset (Fin 5) := {0, 1} -- using the first 2 as meat filled
def red_bean_paste_dumplings : Finset (Fin 5) := {2, 3, 4} -- the remaining are red bean filled

def event_same_filling (x y : Fin 5) : Prop :=
  (x ∈ meat_dumplings ∧ y ∈ meat_dumplings) ∨ (x ∈ red_bean_paste_dumplings ∧ y ∈ red_bean_paste_dumplings)

def event_both_meat (x y : Fin 5) : Prop :=
  x ∈ meat_dumplings ∧ y ∈ meat_dumplings

-- Probability calculations
noncomputable def probability_same_filling : ℚ :=
  (Finset.card ((total_dumplings.product total_dumplings).filter (λ p, event_same_filling p.1 p.2))).toRat /
  (Finset.card (total_dumplings.product total_dumplings)).toRat

noncomputable def probability_both_meat : ℚ :=
  (Finset.card ((total_dumplings.product total_dumplings).filter (λ p, event_both_meat p.1 p.2))).toRat /
  (Finset.card (total_dumplings.product total_dumplings)).toRat

noncomputable def conditional_probability_both_meat_given_same_filling : ℚ :=
  probability_both_meat / probability_same_filling

-- The main theorem statement
theorem probability_both_meat_given_same : 
  conditional_probability_both_meat_given_same_filling = 1 / 4 :=
by
  sorry

end probability_both_meat_given_same_l599_599896


namespace sum_of_their_ages_now_l599_599844

variable (Nacho Divya : ℕ)

-- Conditions
def divya_current_age := 5
def nacho_in_5_years := 3 * (divya_current_age + 5)

-- Definition to determine current age of Nacho
def nacho_current_age := nacho_in_5_years - 5

-- Sum of current ages
def sum_of_ages := divya_current_age + nacho_current_age

-- Theorem to prove the sum of their ages now is 30
theorem sum_of_their_ages_now : sum_of_ages = 30 :=
by
  sorry

end sum_of_their_ages_now_l599_599844


namespace increasing_intervals_value_of_g_at_pi_div_6_l599_599881

noncomputable def f (x : ℝ) := 2 * real.sqrt 3 * real.sin (real.pi - x) * real.sin x - (real.sin x - real.cos x)^2

theorem increasing_intervals : ∀ k : ℤ, ∀ x : ℝ,
  (k * real.pi - real.pi / 12 ≤ x ∧ x ≤ k * real.pi + 5 * real.pi / 12) → 
  (∃ a, 0 < (deriv f x)) := sorry

noncomputable def g (x : ℝ) := 2 * real.sin x + real.sqrt 3 - 1

theorem value_of_g_at_pi_div_6 : g (real.pi / 6) = real.sqrt 3 := sorry

end increasing_intervals_value_of_g_at_pi_div_6_l599_599881


namespace greatest_overlap_of_folded_equilateral_triangle_l599_599314

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (math.sqrt 3 / 4) * s^2

noncomputable def greatest_possible_overlap_area (total_area : ℝ) : ℝ :=
  1 / 3 * total_area

theorem greatest_overlap_of_folded_equilateral_triangle (total_area : ℝ) (h : total_area = 2019) :
  ∃ overlap_area, overlap_area = greatest_possible_overlap_area total_area :=
by {
  use 673,
  have : greatest_possible_overlap_area 2019 = 673, {
    unfold greatest_possible_overlap_area,
    sorry,
  },
  rw h at this,
  exact this,
}

end greatest_overlap_of_folded_equilateral_triangle_l599_599314


namespace max_value_quadratic_expression_l599_599959

theorem max_value_quadratic_expression : ∃ x : ℝ, -3 * x^2 + 18 * x - 5 ≤ 22 ∧ ∀ y : ℝ, -3 * y^2 + 18 * y - 5 ≤ 22 := 
by 
  sorry

end max_value_quadratic_expression_l599_599959


namespace sum_of_interior_angles_heptagon_l599_599621

theorem sum_of_interior_angles_heptagon : ∀ (n : ℕ), n = 7 → (n - 2) * 180 = 900 :=
by
  intros n hn
  rw [hn]
  norm_num

end sum_of_interior_angles_heptagon_l599_599621


namespace ways_to_divide_set_l599_599498

theorem ways_to_divide_set : 
  ∃ (f : set (ℕ → Prop)) (cond : ∀ (A B C : set ℕ), A ∪ B ∪ C = {n | n ≤ 2006} ∧ 
  (A ≠ ∅) ∧ (B ≠ ∅) ∧ (C ≠ ∅) ∧ 
  ∀ (n : ℕ), (n ∈ A → (n + 1) ∉ A) ∧
  (n ∈ B → (n + 1) ∉ B) ∧
  (n ∈ C → (n + 1) ∉ C)),
  f {set ℕ} = 2^2004 - 1
:= sorry

end ways_to_divide_set_l599_599498


namespace geometric_sequence_sum_of_first_n_terms_maximum_term_when_t_neg_l599_599928

open Real

noncomputable def f (a_n a_n_minus_1 a_n_plus_1 t : ℝ) : ℝ → ℝ :=
  λ x, (a_n - a_n_minus_1) * x ^ 2 - (a_n_plus_1 - a_n) * x

theorem geometric_sequence {t : ℝ} (h0 : t ≠ 0 ∧ t ≠ 1)
  (a : ℕ → ℝ) (h1 : a 1 = t) (h2 : a 2 = t^2)
  (h_ext : ∀ n : ℕ, n ≥ 2 → deriv (f (a n) (a (n - 1)) (a (n + 1)) t) t = 0) :
  ∀ n : ℕ, n ≥ 1 → (a (n + 1) - a n) = t * (a n - a (n - 1)) :=
sorry

theorem sum_of_first_n_terms {t : ℝ} (h0 : t ≠ 0 ∧ t ≠ 1) 
  (a : ℕ → ℝ) (h1 : a 1 = t) (h2 : a 2 = t^2)
  (h_ext : ∀ n : ℕ, n ≥ 2 → deriv (f (a n) (a (n - 1)) (a (n + 1)) t) t = 0)
  (b : ℕ → ℝ) (h3 : ∀ n, b n = a n * ln (abs (a n))) :
  ∀ n : ℕ, n ≥ 1 → ∑ i in finset.range n, b (i + 1) = 
    (t * (t^(n - 1) - n * t^n + n - 1) / (1 - t)^2) * ln (abs t) :=
sorry

theorem maximum_term_when_t_neg {t : ℝ} (h0 : t ≠ 0 ∧ t ≠ 1) (h1 : -1 < t ∧ t < 0)
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h2 : a 1 = t) (h3 : ∀ n, a (n + 1) - a n = t * (a n - a (n - 1)))
  (h4 : ∀ n, b n = a n * ln (abs (a n))) :
  ∃ k : ℕ, k = 2 ∧ b 5 = (5 * t^5 * ln (abs t)) :=
sorry

end geometric_sequence_sum_of_first_n_terms_maximum_term_when_t_neg_l599_599928


namespace coin_placement_count_l599_599901

-- Definitions capturing the conditions of the problem
def grid_width : ℕ := 2
def grid_length : ℕ := 100
def total_cells : ℕ := grid_width * grid_length
def coins_to_place : ℕ := 99
def empty_cells_required : ℕ := total_cells - coins_to_place

-- Statement of the outcome of the problem
theorem coin_placement_count :
  ∃ (ways : ℕ), ways = 396 ∧
    (∀ (placement : list (ℕ × ℕ)),
      list.length placement = coins_to_place →
      (∀ (c1 c2 : ℕ × ℕ), c1 ∈ placement → c2 ∈ placement → c1 ≠ c2 →
        ¬ (abs (c1.fst - c2.fst) = 1 ∧ c1.snd = c2.snd) ∧
        ¬ (abs (c1.snd - c2.snd) = 1 ∧ c1.fst = c2.fst))) :=
begin
  existsi 396,
  split,
  { refl },
  { sorry }
end

end coin_placement_count_l599_599901


namespace find_k_l599_599383

theorem find_k : ∃ k : ℝ, 
  ∥(k * (3, -1) - (3, 6))∥ = 3 * √10 := by
  use 2.442
  sorry

end find_k_l599_599383


namespace arrangement_books_l599_599688

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem arrangement_books : combination 9 4 = 126 := by
  sorry

end arrangement_books_l599_599688


namespace geometric_sequence_decreasing_and_q_value_l599_599194

theorem geometric_sequence_decreasing_and_q_value (q : ℝ) (a : ℕ → ℝ) (k : ℕ) (hk : k > 0) 
(h1 : 0 < q) (h2 : q < 1) (h3 : ∀ n, a n = q ^ (n - 1)) 
(h4 : a k * a (k + 2) → a (k + 1) = (a k + a (k + 2)) / 2) : 
q = (1 - Real.sqrt 5) / 2 := 
sorry

end geometric_sequence_decreasing_and_q_value_l599_599194


namespace imaginary_part_conjugate_l599_599074

def imaginary_unit := Complex.i
def z := -1/3 + (2 * Real.sqrt 2) / 3 * imaginary_unit
def z_conjugate := Complex.conj z

def imaginary_part (c : ℂ) : ℝ := c.im

theorem imaginary_part_conjugate :
  imaginary_part z_conjugate = -(2 * Real.sqrt 2) / 3 :=
by
  sorry

end imaginary_part_conjugate_l599_599074


namespace smallest_numbers_l599_599027

-- Define the problem statement
theorem smallest_numbers (m n : ℕ) :
  (∃ (m1 n1 m2 n2 : ℕ), 7 * m1^2 - 11 * n1^2 = 1 ∧ 7 * m2^2 - 11 * n2^2 = 5) ↔
  (7 * m^2 - 11 * n^2 = 1) ∨ (7 * m^2 - 11 * n^2 = 5) :=
by
  sorry

end smallest_numbers_l599_599027


namespace number_of_solutions_l599_599184

theorem number_of_solutions :
  (∀ n, a n = 4 * n) → 
  a 1 = 4 →
  a 2 = 8 →
  a 3 = 12 →
  a 20 = 80 :=
by
  intros h_gen h1 h2 h3
  sorry

end number_of_solutions_l599_599184


namespace inequality_proof_l599_599989

theorem inequality_proof (x y z : ℝ) : 
  x^2 + 2 * y^2 + 3 * z^2 ≥ Real.sqrt 3 * (x * y + y * z + z * x) := 
  sorry

end inequality_proof_l599_599989


namespace gather_all_candies_l599_599946

theorem gather_all_candies (n : ℕ) (h₁ : n ≥ 4) (candies : ℕ) (h₂ : candies ≥ 4)
    (plates : Fin n → ℕ) :
    ∃ plate : Fin n, ∀ i : Fin n, i ≠ plate → plates i = 0 :=
sorry

end gather_all_candies_l599_599946


namespace _l599_599055

noncomputable def geometry_theorem 
  (P T1 T2 F1 F2 : Point) (ellipse : Ellipse) 
  (h1 : is_point_outside P ellipse) 
  (h2 : is_tangent PT1 ellipse T1) 
  (h3 : is_tangent PT2 ellipse T2) 
  (h4 : are_foci F1 F2 ellipse) : 
  ∠ T1 P F1 = ∠ F2 P T2 :=
sorry

end _l599_599055


namespace make_up_set_money_needed_l599_599411

theorem make_up_set_money_needed (makeup_cost gabby_money mom_money: ℤ) (h1: makeup_cost = 65) (h2: gabby_money = 35) (h3: mom_money = 20) :
  (makeup_cost - (gabby_money + mom_money)) = 10 :=
by {
  sorry
}

end make_up_set_money_needed_l599_599411


namespace tan_strictly_increasing_intervals_l599_599039

theorem tan_strictly_increasing_intervals:
  ∀ k : ℤ, ∀ x : ℝ,
  (x > (k * π / 2 - π / 12)) ∧ (x < (k * π / 2 + 5 * π / 12)) →
  ∀ y : ℝ, f y = tan (2 * y - π / 3) →
  (∃ b : bool, b = (f x < f (x + 0.0001))) :=
by
  intros k x hx y hy
  use sorry
  sorry

end tan_strictly_increasing_intervals_l599_599039


namespace sum_distinct_prime_factors_l599_599354

theorem sum_distinct_prime_factors : 
  let n := 7^3 - 3^3 in 
  let prime_factors := {p : ℕ | p.prime ∧ p ∣ n} in
  prime_factors.sum = 81 :=
by
  sorry

end sum_distinct_prime_factors_l599_599354


namespace solve_abs_eq_l599_599919

theorem solve_abs_eq (x : ℝ) : |x - 3| = 5 - 2x ↔ (x = 2 ∨ x = 8/3) :=
by sorry

end solve_abs_eq_l599_599919


namespace jack_has_102_plates_left_l599_599147

def initial_flower_plates := 6
def initial_checked_plates := 9
def initial_striped_plates := 3

def polka_dotted_plates := initial_checked_plates ^ 2
def wave_patterned_plates := (4 * initial_checked_plates) / 9

def smashed_flowered_plates := initial_flower_plates * 10 / 100
def smashed_checked_plates := initial_checked_plates * 15 / 100
def smashed_striped_plates := initial_striped_plates * 20 / 100

def remaining_flowered_plates := initial_flower_plates - smashed_flowered_plates.floor
def remaining_checked_plates := initial_checked_plates - smashed_checked_plates.floor
def remaining_striped_plates := initial_striped_plates - smashed_striped_plates.floor

def total_plates :=
  remaining_flowered_plates + remaining_checked_plates + remaining_striped_plates +
  polka_dotted_plates + wave_patterned_plates

theorem jack_has_102_plates_left : total_plates = 102 := by
  sorry

end jack_has_102_plates_left_l599_599147


namespace john_can_drive_200_miles_l599_599154

theorem john_can_drive_200_miles :
  ∀ (miles_per_gallon cost_per_gallon money : ℕ),
  miles_per_gallon = 40 →
  cost_per_gallon = 5 →
  money = 25 →
  (money / cost_per_gallon) * miles_per_gallon = 200 :=
by
  intros miles_per_gallon cost_per_gallon money h1 h2 h3
  rw [h1, h2, h3]
  sorry

end john_can_drive_200_miles_l599_599154


namespace sufficient_but_not_necessary_condition_l599_599075

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x >= 3) → (x^2 - 2*x - 3 >= 0) ∧ ¬((x^2 - 2*x - 3 >= 0) → (x >= 3)) := by
  sorry

end sufficient_but_not_necessary_condition_l599_599075


namespace hyperbola_eccentricity_l599_599810

variables (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
variable (h3 : (|b * c| / Math.sqrt (a^2 + b^2)) = 2 * a)

theorem hyperbola_eccentricity :
  let e := c / a,
      c := Math.sqrt (a^2 + b^2)
  in e = Math.sqrt 5 := sorry

end hyperbola_eccentricity_l599_599810


namespace distribution_methods_correct_l599_599662

noncomputable def number_of_distribution_methods : ℕ :=
  nat.choose 14 4 * nat.choose 10 5 * nat.choose 5 5 * nat.factorial 3 ^ 3 / (nat.factorial 2 ^ 2)

theorem distribution_methods_correct :
  number_of_distribution_methods = 
  nat.choose 14 4 * nat.choose 10 5 * nat.choose 5 5 * nat.factorial 3 ^ 3 / (nat.factorial 2 ^ 2) := 
by
  -- Place solution steps here
  sorry

end distribution_methods_correct_l599_599662


namespace part_I_part_II_l599_599799

noncomputable def f (x a : ℝ) := -x^2 + a * x + 1 - Real.log x

theorem part_I (a : ℝ) (h : a = 3) : 
  ∀ x : ℝ, 1 / 2 < x → x < 1 → (f x a)' > 0 := by
  sorry

theorem part_II (h_decrease : ∀ x : ℝ, 0 < x → x < 1 / 2 → (f x a)' < 0) : 
  a ≤ 3 := by
  sorry

end part_I_part_II_l599_599799


namespace vartan_wages_decrease_l599_599528

variables (W : ℝ) (P : ℝ)
def wages_last_week := W
def recreation_last_week := 0.15 * W
def wages_this_week := W * (1 - P / 100)
def recreation_this_week := 0.30 * wages_this_week

theorem vartan_wages_decrease :
  recreation_this_week = 1.80 * recreation_last_week → P = 10 :=
by
  unfold wages_last_week recreation_last_week wages_this_week recreation_this_week
  intros h
  sorry

end vartan_wages_decrease_l599_599528


namespace Q_traces_circle_l599_599999

noncomputable def path_of_Q (r d : ℝ) : set (ℝ × ℝ) :=
  { q : ℝ × ℝ | (q.1 - 0)^2 + (q.2 - 0)^2 = 2 * r^2 - d^2}

theorem Q_traces_circle (r d : ℝ) (r_pos : 0 < r) (d_cond : d < r) :
  ∃ O : ℝ × ℝ, ∀ Q : ℝ × ℝ, Q ∈ path_of_Q r d ↔ (Q.1 - O.1)^2 + (Q.2 - O.2)^2 = 2 * r^2 - d^2 :=
by
  use (0, 0) -- The center of the circle is (0, 0)
  intro Q
  sorry

end Q_traces_circle_l599_599999


namespace find_y_l599_599286

-- Given conditions in the problem
variable (y x z : ℝ)
variable (k : ℝ)
hypothesis (h1 : 7 * y = (k * z) / (2 * x)^2)
hypothesis (h2 : x = 1)
hypothesis (h3 : y = 20)
hypothesis (h4 : z = 5)

-- Finding the value of k
def find_k : ℝ := by
  have eq1 : 7 * 20 = (k * 5) / (2 * 1)^2 := by {rw [h2, h3, h4], exact h1}
  sorry

-- Now substituting x = 8, z = 10 to find y
variable (new_x new_z: ℝ)
hypothesis (hx : new_x = 8)
hypothesis (hz : new_z = 10)

def final_y : ℝ := by
  let k := find_k
  have eq2 : 7 * y = (k * 10) / (2 * 8)^2 := by rw [hx, hz]
  sorry

-- Proving the final value of y
theorem find_y : final_y = 0.625 := by
  sorry

end find_y_l599_599286


namespace harbin_ice_festival_XS_draw_l599_599242

theorem harbin_ice_festival_XS_draw :
  ∃ n, let cards := "BXQSSHXGFX".toList in
  n = (cards.count ('X') * cards.count ('S')) ∧ n = 6 :=
by
  existsi 6
  let cards := "BXQSSHXGFX".toList
  let count_X := cards.count ('X')
  let count_S := cards.count ('S')
  have h1 : count_X = 3 := sorry
  have h2 : count_S = 2 := sorry
  calc
  n = count_X * count_S : sorry
  ... = 3 * 2         : sorry
  ... = 6             : sorry

end harbin_ice_festival_XS_draw_l599_599242


namespace sum_of_common_divisors_l599_599393

theorem sum_of_common_divisors (a b : ℕ) (ha : a = 75) (hb : b = 45) :
  ∑ d in (finset.filter (λ x, x ∣ b) (finset.filter (λ x, x ∣ a) (finset.range (a + 1)))), d = 24 :=
by 
  sorry

end sum_of_common_divisors_l599_599393


namespace problem1_problem2_l599_599787

open Real

-- Define α in the given interval and the equation.
variable (α : ℝ)
variable h1 : 0 < α ∧ α < π / 3
variable h2 : sqrt 3 * sin α + cos α = sqrt 6 / 2

-- Problem 1: Prove the value of cos(α + π/6)
theorem problem1 : cos (α + π / 6) = sqrt 10 / 4 :=
by
  -- Use the given conditions
  sorry

-- Problem 2: Prove the value of cos(2α + 7π/12)
theorem problem2 : cos (2 * α + 7 * π / 12) = (sqrt 2 - sqrt 30) / 8 :=
by
  -- Use the given conditions
  sorry

end problem1_problem2_l599_599787


namespace find_A_and_B_l599_599380

theorem find_A_and_B (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ -6 → (5 * x - 3) / (x^2 + 3 * x - 18) = A / (x - 3) + B / (x + 6)) →
  A = 4 / 3 ∧ B = 11 / 3 :=
by
  intros h
  sorry

end find_A_and_B_l599_599380


namespace Rams_monthly_salary_l599_599990

variable (R S A : ℝ)
variable (annual_salary : ℝ)
variable (monthly_salary_conversion : annual_salary / 12 = A)
variable (ram_shyam_condition : 0.10 * R = 0.08 * S)
variable (shyam_abhinav_condition : S = 2 * A)
variable (abhinav_annual_salary : annual_salary = 192000)

theorem Rams_monthly_salary 
  (annual_salary : ℝ)
  (ram_shyam_condition : 0.10 * R = 0.08 * S)
  (shyam_abhinav_condition : S = 2 * A)
  (abhinav_annual_salary : annual_salary = 192000)
  (monthly_salary_conversion: annual_salary / 12 = A): 
  R = 25600 := by
  sorry

end Rams_monthly_salary_l599_599990


namespace range_of_k_find_k_value_l599_599935

open Real

noncomputable def quadratic_eq_has_real_roots (a b c : ℝ) (disc : ℝ) : Prop :=
  disc > 0

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

variables {a : ℝ} {b : ℝ} {c : ℝ} {k : ℝ}
variables (alpha beta : ℝ)

-- Given conditions
def quadratic_eq_with_k := (a = 1) ∧ (b = 2) ∧ (c = 3 - k)
def two_distinct_real_roots := quadratic_eq_has_real_roots 1 2 (3 - k) (discriminant 1 2 (3 - k))
def product_of_roots := alpha * beta = 3 - k
def given_condition := k^2 = alpha * beta + 3 * k

-- Proofs to be done
theorem range_of_k : quadratic_eq_with_k k → two_distinct_real_roots k → k > 2 :=
by
  intro h1 h2
  sorry

theorem find_k_value : quadratic_eq_with_k k → k > 2 → given_condition k alpha beta → k = 3 :=
by
  intro h1 h2 h3
  sorry

end range_of_k_find_k_value_l599_599935


namespace emma_possible_lists_l599_599665

-- Define the number of balls
def number_of_balls : ℕ := 24

-- Define the number of draws Emma repeats independently
def number_of_draws : ℕ := 4

-- Define the calculation for the total number of different lists
def total_number_of_lists : ℕ := number_of_balls ^ number_of_draws

theorem emma_possible_lists : total_number_of_lists = 331776 := by
  sorry

end emma_possible_lists_l599_599665


namespace chord_length_AB_l599_599139

open Real

def line (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0

def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

theorem chord_length_AB : ∃ A B : ℝ × ℝ, 
  line A.1 A.2 ∧ circle A.1 A.2 ∧ line B.1 B.2 ∧ circle B.1 B.2 ∧
  dist A B = 2 * sqrt 3 :=
by
  sorry

end chord_length_AB_l599_599139


namespace find_f_five_thirds_l599_599543

variable {R : Type*} [LinearOrderedField R]

-- Define the odd function and its properties
variable {f : R → R}
variable (oddf : ∀ x : R, f (-x) = -f x)
variable (propf : ∀ x : R, f (1 + x) = f (-x))
variable (val : f (- (1 / 3 : R)) = 1 / 3)

theorem find_f_five_thirds : f (5 / 3 : R) = 1 / 3 := by
  sorry

end find_f_five_thirds_l599_599543


namespace T_12_eq_2304_l599_599879

def B1 (n : ℕ) : ℕ :=
  if n = 1 then 1 else B1 (n - 1) + B2 (n - 1) + B3 (n - 1) + B4 (n - 1)

def B2 (n : ℕ) : ℕ :=
  if n = 1 then 0 else B1 (n - 1)

def B3 (n : ℕ) : ℕ :=
  if n = 1 then 0 else B2 (n - 1)

def B4 (n : ℕ) : ℕ :=
  if n = 1 then 0 else B3 (n - 1)

def T (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 3
  else if n = 3 then 7
  else if n = 4 then 16
  else T (n - 1) + T (n - 2) + T (n - 3) + T (n - 4)

theorem T_12_eq_2304 : T 12 = 2304 := by
  sorry

end T_12_eq_2304_l599_599879


namespace proof_problem_l599_599073

-- Conditions
def a : ℤ := 1
def b : ℤ := 0
def c : ℤ := -1 + 3

-- Proof Statement
theorem proof_problem : (2 * a + 3 * c) * b = 0 := by
  sorry

end proof_problem_l599_599073


namespace EF_squared_exists_l599_599590

-- Definition of a square and conditions
structure Square :=
  (A B C D E F : Point) -- vertices A, B, C, D and points E, F
  (side_length : ℝ)
  (BE DF : ℝ)
  (AE CF : ℝ)

-- Define the problem
def problem : Prop :=
  ∃ (s : Square),
    s.side_length = 13 ∧
    s.BE = 5 ∧
    s.DF = 5 ∧
    s.AE = 12 ∧
    s.CF = 12 ∧
    EF s.E s.F = 578

-- The theorem to be proven
theorem EF_squared_exists : problem :=
by sorry

end EF_squared_exists_l599_599590


namespace sum_of_possible_N_values_l599_599614

theorem sum_of_possible_N_values (a b c N : ℕ) (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = 2 * (a + b)) :
  ∃ sum_N : ℕ, sum_N = 672 :=
by
  sorry

end sum_of_possible_N_values_l599_599614


namespace sum_of_coefficients_eq_39_l599_599048

theorem sum_of_coefficients_eq_39 :
  5 * (2 * 1^8 - 3 * 1^3 + 4) - 6 * (1^6 + 4 * 1^3 - 9) = 39 :=
by
  sorry

end sum_of_coefficients_eq_39_l599_599048


namespace area_rhombus_AFCE_l599_599499

-- Definitions of the geometric entities and their properties
structure Rectangle (A B C D : Type) :=
  (width : ℝ) (length : ℝ)

structure Rhombus (A F C E : Type) :=
  (inscribed_in : Rectangle)
  (AF : ℝ)
  (CE : ℝ)
  (AE : ℝ)
  (FC : ℝ)
  (equal_sides : AF = CE ∧ CE = AE ∧ AE = FC)

axiom inscribed (r : Rhombus) : 
  ∃ E F : r.inscribed_in.AB, E ∈ r.inscribed_in.AD ∧ F ∈ r.inscribed_in.BC

-- Conditions of the problem
variables (A B C D E F : Type)
variable (r : Rhombus A F C E)
variable (R : Rectangle A B C D)
variable (AB : height → ℝ)
variable (BC : length → ℝ)

-- Ensure the underlying rectangle's dimensions
axiom Rectangle_width : R.width = 20
axiom Rectangle_length : R.length = 25

-- Ensure the point's positions and angles
axiom Position_E : E ∈ R.AB → E = R.A 
axiom Position_F : F ∈ R.BC → F = R.C 
axiom Angle_alpha_beta: ∀ α β : Type, α = β

-- Problem Statement: Prove the area of rhombus AFCE is 125 square yards
theorem area_rhombus_AFCE : 
  ∃ (Area : ℝ), Area = 125 :=
sorry

end area_rhombus_AFCE_l599_599499


namespace charlyn_viewable_area_is_39_l599_599004

-- Define the side length of the square
def side_length : ℝ := 5

-- Define the radius of visibility
def visibility_radius : ℝ := 1

-- The correct answer is that the total viewable area is 39 square kilometers.
theorem charlyn_viewable_area_is_39 :
  let total_area := 16 + 20 + Real.pi in
  total_area = 39 := 
by
  sorry

end charlyn_viewable_area_is_39_l599_599004


namespace max_a_ln_ax_plus_ax_le_x_plus_exp_x_l599_599126

theorem max_a_ln_ax_plus_ax_le_x_plus_exp_x (a : ℝ) :
  (∀ x : ℝ, ln (a * x) + a * x ≤ x + exp x) → a ≤ Real.exp 1 :=
by
  sorry

end max_a_ln_ax_plus_ax_le_x_plus_exp_x_l599_599126


namespace halfway_between_l599_599933

theorem halfway_between (a b : ℚ) (h1 : a = 1/12) (h2 : b = 1/15) : (a + b) / 2 = 3 / 40 := by
  -- proofs go here
  sorry

end halfway_between_l599_599933


namespace calculate_expression_l599_599355

theorem calculate_expression : (-1 : ℤ)^(2023) + |(1 : ℝ) - real.sqrt 3| - (2 : ℝ) / (real.sqrt 3 - 1) - ((-1 / 2 : ℝ) ^ (-2 : ℤ)) = -7 := by
  sorry

end calculate_expression_l599_599355


namespace find_number_l599_599741

theorem find_number (x : ℕ) (h : (18 / 100) * x = 90) : x = 500 :=
sorry

end find_number_l599_599741


namespace incorrect_option_l599_599785

theorem incorrect_option (a : ℝ) (h : a ≠ 0) : (a + 2) ^ 0 ≠ 1 ↔ a = -2 :=
by {
  sorry
}

end incorrect_option_l599_599785


namespace coefficient_of_x_sq_in_expansion_of_1_minus_x_pow_5_coefficient_of_x_sq_is_10_l599_599737

theorem coefficient_of_x_sq_in_expansion_of_1_minus_x_pow_5 :
  ∀ x, (1 - x)^5 = ∑ r in finset.range(6), (-1)^r * (nat.choose 5 r) * x^r :=
begin
  sorry
end

theorem coefficient_of_x_sq_is_10 :
  10 = ∑ k in (finset.single 2), (-1)^k * (nat.choose 5 k) := 
begin
  sorry
end

end coefficient_of_x_sq_in_expansion_of_1_minus_x_pow_5_coefficient_of_x_sq_is_10_l599_599737


namespace mary_max_earnings_l599_599567

def max_hours : ℕ := 40
def regular_rate : ℝ := 8
def first_hours : ℕ := 20
def overtime_rate : ℝ := regular_rate + 0.25 * regular_rate

def earnings : ℝ := 
  (first_hours * regular_rate) +
  ((max_hours - first_hours) * overtime_rate)

theorem mary_max_earnings : earnings = 360 := by
  sorry

end mary_max_earnings_l599_599567


namespace min_expr_value_l599_599643

theorem min_expr_value : ∃ a b c ∈ ({-10, -7, -3, 0, 4, 6, 9} : set ℤ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a * b * c - b) = -546 := 
by 
  use [-10, 6, 9]
  split
  repeat { use, { try { apply set.mem_insert_of_mem, apply set.mem_insert_of_mem }, apply set.mem_singleton } }
  split
  exact ne_of_eq_of_ne rfl (ne_symm (by decide))
  split
  exact ne_of_eq_of_ne rfl (ne_symm (by decide))
  exact ne_of_eq_of_ne rfl (ne_symm (by decide))
  sorry

end min_expr_value_l599_599643


namespace circles_intersect_at_single_point_l599_599733

variables {A B C P Q R O : Type}
variables {dist : P → Q → ℝ} 

theorem circles_intersect_at_single_point (
  vertices : set (A ∪ B ∪ C),
  centers : set (P ∪ Q ∪ R),
  circle : P → ℝ,
  is_tangent : ∀ {x y : P}, dist x y = 2 * circle P
) (Hcirc : O = dist P Q ∧ dist Q R ∧ dist R P)
:
∃ O, (dist O P = circle P) ∧ (dist O Q = circle Q) ∧ (dist O R = circle R) :=
sorry

end circles_intersect_at_single_point_l599_599733


namespace question1_question2_question3_l599_599474

open Real

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (2 + sin x, 1)
def vec_b : ℝ × ℝ := (2, -2)
def vec_c (x : ℝ) : ℝ × ℝ := (sin x - 3, 1)
def vec_d (k : ℝ) : ℝ × ℝ := (1, k)

-- Question 1
theorem question1 (h1 : x ∈ (interval_of_real (-π/2) (π/2))) (h2 : vec_a x = scalar_mul (vec_b + vec_c x) (some 𝜆)) :
  x = -π/6 :=
sorry

-- Question 2
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * 2 + (vec_a x).2 * (-2)

theorem question2 :
  ∃ (x₀ : ℝ), f x₀ = 0 ∧ ∀ x, f x ≥ 0 :=
sorry

-- Question 3
theorem question3 :
  ∃ (k : ℝ), ∀ x, -π/2 ≤ x ∧ x ≤ π/2 → -5 ≤ ((sin x + 1)^2 - 5) ∧ ((sin x + 1)^2 - 5) ≤ -1 :=
sorry

end question1_question2_question3_l599_599474


namespace samantha_erased_length_l599_599583

/--
Samantha drew a line that was originally 1 meter (100 cm) long, and then it was erased until the length was 90 cm.
This theorem proves that the amount erased was 10 cm.
-/
theorem samantha_erased_length : 
  let original_length := 100 -- original length in cm
  let final_length := 90 -- final length in cm
  original_length - final_length = 10 := 
by
  sorry

end samantha_erased_length_l599_599583


namespace tangent_line_at_one_l599_599806

noncomputable def f (x : ℝ) := Real.log x + x^2

theorem tangent_line_at_one :
  ∃ (m b : ℝ), (∀ x, m * x + b = 3 * x - 2) ∧ (∀ x, f(x) = Real.log x + x^2) ∧
  (∀ x, f'(x) = (1/x) + 2*x) :=
begin
  sorry
end

end tangent_line_at_one_l599_599806


namespace tourist_grouping_count_l599_599258

theorem tourist_grouping_count : 
  ∑ k in Finset.range (7 + 1), if k = 0 ∨ k = 7 then 0 else Nat.choose 7 k = 126 :=
by
  sorry

end tourist_grouping_count_l599_599258


namespace simplify_expression_l599_599177

theorem simplify_expression (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x^3 + y^3 = 3 * (x + y)) :
  (x / y) + (y / x) - (3 / (x * y)) = 1 :=
by
  sorry

end simplify_expression_l599_599177


namespace anya_possible_wins_l599_599343

-- Define the total rounds played
def total_rounds := 25

-- Define Anya's choices
def anya_rock := 12
def anya_scissors := 6
def anya_paper := 7

-- Define Borya's choices
def borya_rock := 13
def borya_scissors := 9
def borya_paper := 3

-- Define the relationships in rock-paper-scissors game
def rock_beats_scissors := true
def scissors_beat_paper := true
def paper_beats_rock := true

-- Define no draws condition
def no_draws := total_rounds = anya_rock + anya_scissors + anya_paper ∧ total_rounds = borya_rock + borya_scissors + borya_paper

-- Proof problem statement
theorem anya_possible_wins : anya_rock + anya_scissors + anya_paper = total_rounds ∧
                             borya_rock + borya_scissors + borya_paper = total_rounds ∧
                             rock_beats_scissors ∧ scissors_beat_paper ∧ paper_beats_rock ∧
                             no_draws →
                             (9 + 3 + 7 = 19) := by
  sorry

end anya_possible_wins_l599_599343


namespace boat_distance_downstream_l599_599689

theorem boat_distance_downstream
  (dist_upstream : ℝ) (time_upstream : ℝ) (speed_stream : ℝ) (b : ℝ)
  (h1 : dist_upstream = 30)
  (h2 : time_upstream = 3)
  (h3 : speed_stream = 5)
  (h4 : b - speed_stream = dist_upstream / time_upstream)
  :
  let downstream_speed := b + speed_stream in
  let time_downstream := time_upstream in
  downstream_speed * time_downstream = 60 :=
by
  sorry

end boat_distance_downstream_l599_599689


namespace arithmetic_sum_first_11_terms_l599_599241

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

variable (a : ℕ → ℝ)

theorem arithmetic_sum_first_11_terms (h_arith_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_sum_condition : a 2 + a 6 + a 10 = 6) :
  sum_first_n_terms a 11 = 22 :=
sorry

end arithmetic_sum_first_11_terms_l599_599241


namespace num_paths_six_l599_599477

noncomputable def num_paths : ℕ → ℕ 
| 1     := 2
| (n+1) := 2 * (n+1) * num_paths n

theorem num_paths_six :
  num_paths 6 = 46080 :=
by {
  -- Proof skipped
  sorry
}

end num_paths_six_l599_599477


namespace smallest_N_for_rectangle_l599_599697

theorem smallest_N_for_rectangle (N : ℕ) (stick_lengths : Fin N → ℕ) 
  (sum_lengths : Finset.sum (Finset.univ.image stick_lengths) = 200)
  (all_whole_cm : ∀ i : Fin N, stick_lengths i ≥ 1) :
  ∃ N = 102, 
    ∃ (stick_lengths : Fin N → ℕ), 
      (Finset.sum (Finset.univ.image stick_lengths) = 200) 
      ∧ (∀ A B C D : Fin N, A ≠ B ∧ C ≠ D → stick_lengths A + stick_lengths B + stick_lengths C + stick_lengths D = 200 ∧
          (stick_lengths A = stick_lengths C ∧ stick_lengths B = stick_lengths D ∨ stick_lengths A = stick_lengths D ∧ stick_lengths B = stick_lengths C)) :=
begin
  sorry
end

end smallest_N_for_rectangle_l599_599697


namespace right_angled_triangle_count_in_pyramid_l599_599143

-- Define the cuboid and the triangular pyramid within it
variables (A B C D A₁ B₁ C₁ D₁ : Type)

-- Assume there exists a cuboid ABCD-A₁B₁C₁D₁
axiom cuboid : Prop

-- Define the triangular pyramid A₁-ABC
structure triangular_pyramid (A₁ A B C : Type) : Type :=
  (vertex₁ : A₁)
  (vertex₂ : A)
  (vertex₃ : B)
  (vertex4 : C)
  
-- The mathematical statement to prove: the number of right-angled triangles in A₁-ABC is 4
theorem right_angled_triangle_count_in_pyramid (A : Type) (B : Type) (C : Type) (A₁ : Type)
  (h_pyramid : triangular_pyramid A₁ A B C) (h_cuboid : cuboid) :
  ∃ n : ℕ, n = 4 :=
by
  sorry

end right_angled_triangle_count_in_pyramid_l599_599143


namespace parallelogram_area_correct_l599_599753

open Real EuclideanSpace

noncomputable def area_of_parallelogram (A B : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  ‖A ×ₐ B‖

theorem parallelogram_area_correct :
  let A := ![4, 2, -1] : EuclideanSpace ℝ (Fin 3)
  let B := ![-1, 5, 3] : EuclideanSpace ℝ (Fin 3)
  area_of_parallelogram A B = Real.sqrt 726 := by
    -- proof omitted
    sorry

end parallelogram_area_correct_l599_599753


namespace coins_missing_fraction_l599_599744

-- Definitions based on conditions
def initial_coins (x : ℝ) : ℝ := x
def coins_lost (x : ℝ) : ℝ := (2 / 3) * x
def coins_found (x : ℝ) : ℝ := (3 / 4) * coins_lost x
def coins_still_have (x : ℝ) : ℝ := (1 / 3) * x + coins_found x
def coins_missing (x : ℝ) : ℝ := initial_coins x - coins_still_have x

-- Theorem statement
theorem coins_missing_fraction (x : ℝ) : coins_missing x = (1 / 6) * x := by
  sorry

end coins_missing_fraction_l599_599744


namespace find_a_l599_599221

-- Definitions from conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2
def directrix : ℝ := 1

-- Statement to prove
theorem find_a (a : ℝ) (h : directrix = 1) : a = -1/4 :=
sorry

end find_a_l599_599221


namespace find_expression_l599_599829

theorem find_expression (x : ℝ) (h : x + real.sqrt (x^2 - 4) + 1 / (x - real.sqrt (x^2 - 4)) = 12) : 
  x^2 + real.sqrt (x^4 - 4) + 1 / (x^2 + real.sqrt (x^4 - 4)) = 200 / 9 :=
sorry

end find_expression_l599_599829


namespace sequence_an_formula_l599_599565

theorem sequence_an_formula (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, a (n + 1) = a n^2 - n * a n + 1) :
  ∀ n : ℕ, a n = n + 1 :=
sorry

end sequence_an_formula_l599_599565


namespace vector_magnitude_range_l599_599786

open Real

theorem vector_magnitude_range {A B C : Point}
  (hAB : dist A B = 18)
  (hAC : dist A C = 5) :
  13 ≤ dist B C ∧ dist B C ≤ 23 := by
sorry

end vector_magnitude_range_l599_599786


namespace train_length_l599_599326

theorem train_length
  (S : ℝ)
  (L : ℝ)
  (h1 : L + 140 = S * 15)
  (h2 : L + 250 = S * 20) :
  L = 190 :=
by
  -- Proof to be provided here
  sorry

end train_length_l599_599326


namespace mul_103_97_l599_599006

theorem mul_103_97 : 103 * 97 = 9991 := by
  sorry

end mul_103_97_l599_599006


namespace find_a_plus_b_l599_599808

-- Definitions based on conditions
def g (a b x : ℝ) : ℝ := a*x^2 - 2*a*x + 1 + b

axiom a_pos : ∃ a : ℝ, a > 0

-- Conditions given in the problem
axiom g_min : ∀ a b : ℝ, (2 ≤ x ∧ x ≤ 3) → g a b 1 = 1
axiom g_max : ∀ a b : ℝ, (2 ≤ x ∧ x ≤ 3) → g a b 2 = 4

-- Goal statement
theorem find_a_plus_b (a b : ℝ) (a_pos : ∃ a, a > 0) (h1 : g a b 1 = 1) (h2 : g a b 2 = 4) :
  a + b = 6 :=
by sorry

end find_a_plus_b_l599_599808


namespace minimum_value_of_function_l599_599756

def f (x : ℝ) : ℝ := 2 * (sin x)^2 + 2 * sin x - 1 / 2

theorem minimum_value_of_function :
  ∃ c : ℝ, ∀ x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6), f x = c ∧ c = 1 :=
by
  sorry

end minimum_value_of_function_l599_599756


namespace solve_abs_eq_l599_599918

theorem solve_abs_eq (x : ℝ) : |x - 3| = 5 - 2x ↔ (x = 2 ∨ x = 8/3) :=
by sorry

end solve_abs_eq_l599_599918


namespace smallest_possible_size_l599_599173

open Set

noncomputable def A_S (S : Set ℝ) : Set ℝ :=
  {x | ∃ s t ∈ S, s ≠ t ∧ x = (s + t) / 2}

theorem smallest_possible_size (S : Set ℝ) (n : ℕ) (hn : n ≥ 2) (hS : Finite S) (hcard : S.toFinset.card = n) :
  ∃ A : Set ℝ, A = A_S S ∧ A.toFinset.card = 2 * n - 3 :=
sorry

end smallest_possible_size_l599_599173


namespace total_action_figures_l599_599521

def jerry_original_count : Nat := 4
def jerry_added_count : Nat := 6

theorem total_action_figures : jerry_original_count + jerry_added_count = 10 :=
by
  sorry

end total_action_figures_l599_599521


namespace distance_ac_l599_599512

-- Conditions
axiom lines_parallel (a b c : Line) : Parallel a b ∧ Parallel b c ∧ Parallel a c
axiom distance_ab : ∀ a b : Line, Distance a b = 4
axiom distance_bc : ∀ b c : Line, Distance b c = 1

-- Statement to prove
theorem distance_ac (a b c : Line) (parallel : lines_parallel a b c) : Distance a c = 3 ∨ Distance a c = 5 := 
by sorry

end distance_ac_l599_599512


namespace best_representation_is_B_l599_599180

-- Define the conditions
structure Trip :=
  (home_to_diner : ℝ)
  (diner_stop : ℝ)
  (diner_to_highway : ℝ)
  (highway_to_mall : ℝ)
  (mall_stop : ℝ)
  (highway_return : ℝ)
  (construction_zone : ℝ)
  (return_city_traffic : ℝ)

-- Graph description
inductive Graph
| plateau : Graph
| increasing : Graph → Graph
| decreasing : Graph → Graph

-- Condition that describes the pattern of the graph
def correct_graph (trip : Trip) : Prop :=
  let d1 := trip.home_to_diner
  let d2 := trip.diner_stop
  let d3 := trip.diner_to_highway
  let d4 := trip.highway_to_mall
  let d5 := trip.mall_stop
  let d6 := trip.highway_return
  let d7 := trip.construction_zone
  let d8 := trip.return_city_traffic
  d1 > 0 ∧ d2 = 0 ∧ d3 > 0 ∧ d4 > 0 ∧ d5 = 0 ∧ d6 < 0 ∧ d7 < 0 ∧ d8 < 0

-- Theorem statement
theorem best_representation_is_B (trip : Trip) : correct_graph trip :=
by sorry

end best_representation_is_B_l599_599180


namespace intersection_diagonals_incenter_l599_599359

open Classical
open Real
open Set
open Pointwise

noncomputable def incenter_of_triangle (A B C : Point) : Point := sorry

noncomputable def circle_center (M N : Point) (intersect : M = N) : Point := sorry

def circles_inter_coord (A B M N C D : Point) :=
∃ (O₁ O₂ : Circle), O₁.center = A ∧ O₂.center = B ∧
(O₁ ∩ O₂ = {M, N}) ∧
(MN ∩ O₁ = {C}) ∧ (MN ∩ O₂ = {D})

axiom rays_inter (A C X B D Y : Point) (intersect : X = C) : Prop

theorem intersection_diagonals_incenter (A B M N C D X Y : Point)
  (h1 : ∃ (O₁ O₂ : Circle), O₁.center = A ∧ O₂.center = B ∧
                           (O₁ ∩ O₂ = {M, N}) ∧
                           (MN ∩ O₁ = {C}) ∧ (MN ∩ O₂ = {D}))
  (h2 : rays_inter A C X B D Y C)
  : circle_center (intersections (M ⊔ N) (C ⊔ D)) (incenter_of_triangle A C D) := sorry

end intersection_diagonals_incenter_l599_599359


namespace calc_expression_l599_599356

theorem calc_expression :
  2 * Real.sin (45 * Real.pi / 180) - Real.sqrt 4 + ((-1/3)⁻¹) + abs (Real.sqrt 2 - 3) = -2 :=
by
  have h1 : 2 * Real.sin (45 * Real.pi / 180) = Real.sqrt 2 := by sorry
  have h2 : Real.sqrt 4 = 2 := by sorry
  have h3 : (-1/3)⁻¹ = -3 := by sorry
  have h4 : abs (Real.sqrt 2 - 3) = 3 - Real.sqrt 2 := by sorry
  rw [h1, h2, h3, h4]
  linarith

end calc_expression_l599_599356


namespace fraction_of_park_occupied_l599_599691

-- Definitions based on the conditions in a)
def rectangle (length width : ℝ) : Prop := length >= 0 ∧ width >= 0 
def trapezoid (a b height : ℝ) : Prop := a >= 0 ∧ b >= 0 ∧ height >= 0 ∧ b - a > 0
def isosceles_right_triangle (leg : ℝ) : Prop := leg >= 0

-- Condition translations
def park_condition (length width : ℝ) : Prop :=
  rectangle length width ∧ 
  (∃ a b height : ℝ, trapezoid a b height ∧ a = 30 ∧ b = 50) ∧ 
  (∃ leg : ℝ, isosceles_right_triangle leg ∧ 3 * (1/2 * leg ^ 2) = (1/8) * length * width)

-- The proof problem in Lean
theorem fraction_of_park_occupied (length width : ℝ) (h : park_condition length width) :
  let fraction := 3 * (1/2 * (20 / 3) ^ 2) / (length * width) in
  fraction = 1 / 8 :=
by
  sorry

end fraction_of_park_occupied_l599_599691


namespace train_passes_bridge_in_128_seconds_l599_599700

/-- A proof problem regarding a train passing a bridge -/
theorem train_passes_bridge_in_128_seconds 
  (train_length : ℕ) 
  (train_speed_kmh : ℕ) 
  (bridge_length : ℕ) 
  (conversion_factor : ℚ) 
  (time_to_pass : ℚ) :
  train_length = 1200 →
  train_speed_kmh = 90 →
  bridge_length = 2000 →
  conversion_factor = (5 / 18) →
  time_to_pass = (train_length + bridge_length) / (train_speed_kmh * conversion_factor) →
  time_to_pass = 128 := 
by
  -- We are skipping the proof itself
  sorry

end train_passes_bridge_in_128_seconds_l599_599700


namespace median_and_mode_correct_l599_599850

/-- The scores of the 10 finalists --/
def scores : List ℕ := [36, 37, 37, 38, 38, 39, 39, 39, 40, 40]

/-- The median of a list of scores --/
noncomputable def median (l : List ℕ) : ℚ :=
  if l.length % 2 = 1 then
    l.nth_le (l.length / 2) (by simp [Nat.div_lt_self]) 
  else 
    (l.nth_le (l.length / 2 - 1) (by simp [Nat.div_lt_self, Nat.sub_pos_of_lt])) +
    l.nth_le (l.length / 2) (by simp [Nat.div_lt_self]) / 2

/-- The mode of a list of scores --/
def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x, if l.count x > l.count acc then x else acc) 0

theorem median_and_mode_correct :
  median scores = 38.5 ∧ mode scores = 39 :=
by
  sorry

end median_and_mode_correct_l599_599850


namespace red_stars_eq_35_l599_599497

-- Define the conditions
noncomputable def number_of_total_stars (x : ℕ) : ℕ := x + 20 + 15
noncomputable def red_star_frequency (x : ℕ) : ℚ := x / (number_of_total_stars x : ℚ)

-- Define the theorem statement
theorem red_stars_eq_35 : ∃ x : ℕ, red_star_frequency x = 0.5 ↔ x = 35 := sorry

end red_stars_eq_35_l599_599497


namespace Cloud_Computing_l599_599594

theorem Cloud_Computing (n : ℕ) (hn : n = 5)
    (hP : (choose 5 2 : ℚ) / choose (n + 5) 2 = 2 / 9) :
  (n = 5) ∧
  let X := λ k, choose 3 k * choose (n + 2) (3 - k) / choose (n + 5) 3 in
  X 0 = 1 / 12 ∧
  X 1 = 5 / 12 ∧
  X 2 = 5 / 12 ∧
  X 3 = 1 / 12 ∧
  (0 * X 0 + 1 * X 1 + 2 * X 2 + 3 * X 3 = 3 / 2) := by
  sorry

end Cloud_Computing_l599_599594


namespace find_a_l599_599922

def f (x : ℝ) : ℝ := (x + 4) / 7 + 2
def g (x : ℝ) : ℝ := 5 - 2 * x
def a : ℝ := -16.5

theorem find_a (h : f (g a) = 8) : a = -16.5 := by
  sorry

end find_a_l599_599922


namespace triangle_area_correct_l599_599421

-- Definition of the vertices of the triangle
def vertex1 : (ℝ × ℝ) := (-4, 8)
def vertex2 : (ℝ × ℝ) := (0, 0)
def vertex3 : (ℝ × ℝ) := (-8, 0)

-- Function to calculate the area of the triangle from its vertices
def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((v2.1 - v1.1) * (v3.2 - v1.2) - (v3.1 - v1.1) * (v2.2 - v1.2))

-- The theorem to be proven
theorem triangle_area_correct : triangle_area vertex1 vertex2 vertex3 = 32 :=
by
  sorry

end triangle_area_correct_l599_599421


namespace correct_operation_l599_599971

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := 
by 
  sorry

end correct_operation_l599_599971


namespace prove_value_of_custom_ops_l599_599049

-- Define custom operations to match problem statement
def custom_op1 (x : ℤ) : ℤ := 7 - x
def custom_op2 (x : ℤ) : ℤ := x - 10

-- The main proof statement
theorem prove_value_of_custom_ops : custom_op2 (custom_op1 12) = -15 :=
by sorry

end prove_value_of_custom_ops_l599_599049


namespace bound_on_f_l599_599535

theorem bound_on_f 
  (f : ℝ → ℝ) 
  (h_domain : ∀ x, 0 ≤ x ∧ x ≤ 1) 
  (h_zeros : f 0 = 0 ∧ f 1 = 0)
  (h_condition : ∀ x1 x2, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ x1 ≠ x2 → |f x2 - f x1| < |x2 - x1|) 
  : ∀ x1 x2, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 → |f x2 - f x1| < 1/2 :=
by
  sorry

end bound_on_f_l599_599535


namespace problem_l599_599816

def set_M : Set ℝ := {θ | ∃ k : ℤ, θ = k * π / 4}
def set_N : Set ℝ := {x | ∃ k : ℤ, x = k * π / 2 + π / 4}
def set_P : Set ℝ := {a | ∃ k : ℤ, a = k * π + π / 4}

theorem problem :
  set_P ⊆ set_N ∧ set_N ⊆ set_M :=
sorry

end problem_l599_599816


namespace find_f_five_thirds_l599_599541

variable {R : Type*} [LinearOrderedField R]

-- Define the odd function and its properties
variable {f : R → R}
variable (oddf : ∀ x : R, f (-x) = -f x)
variable (propf : ∀ x : R, f (1 + x) = f (-x))
variable (val : f (- (1 / 3 : R)) = 1 / 3)

theorem find_f_five_thirds : f (5 / 3 : R) = 1 / 3 := by
  sorry

end find_f_five_thirds_l599_599541


namespace lucas_sequence_mod_5_l599_599923

noncomputable def alteredLucas : ℕ → ℕ
| 0     := 1
| 1     := 4
| (n+2) := alteredLucas (n+1) + alteredLucas n

theorem lucas_sequence_mod_5 :
  alteredLucas 53 % 5 = 0 :=
sorry

end lucas_sequence_mod_5_l599_599923


namespace slope_angle_of_AB_is_90_l599_599065

def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (2, 3)

theorem slope_angle_of_AB_is_90 :
  ∃ θ : ℝ, θ = 90 ∧ atan((B.2 - A.2) / (B.1 - A.1)) = θ :=
by 
  sorry

end slope_angle_of_AB_is_90_l599_599065


namespace slope_of_tangent_at_x1_l599_599450

noncomputable def f (x : ℝ) : ℝ := 2 - 1 / x

theorem slope_of_tangent_at_x1 :
  ∀ (f : ℝ → ℝ), (∀ x, f(x + 1) = (2 * x + 1) / (x + 1)) → 
  deriv f 1 = 1 :=
begin
  intros f h,
  have h_simp: ∀ (x : ℝ), f x = 2 - 1 / x := sorry,
  have h_deriv: ∀ (x : ℝ), deriv f x = 1 / x^2 := sorry,
  have h_eval: deriv f 1 = 1 := sorry,
  exact h_eval,
end

end slope_of_tangent_at_x1_l599_599450


namespace hyperbola_center_is_equidistant_l599_599686

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem hyperbola_center_is_equidistant (F1 F2 C : ℝ × ℝ) 
  (hF1 : F1 = (3, -2)) 
  (hF2 : F2 = (11, 6))
  (hC : C = ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2)) :
  C = (7, 2) ∧ distance C F1 = distance C F2 :=
by
  -- Fill in with the appropriate proofs
  sorry

end hyperbola_center_is_equidistant_l599_599686


namespace evaluate_nested_function_l599_599447

def f (x : ℝ) : ℝ :=
  if x ≤ 2000 then Real.cos ((Real.pi / 4) * x) else x - 14

theorem evaluate_nested_function : f (f 2014) = 0 := by
  sorry

end evaluate_nested_function_l599_599447


namespace tan_alpha_eq_neg_three_expression_eq_neg_three_halves_l599_599781

-- Define the problem conditions
variables {α : ℝ} (h₀ : 0 < α ∧ α < π) (h₁ : sin α + cos α = sqrt 10 / 5)

-- Problem (1): Prove that tan α = -3
theorem tan_alpha_eq_neg_three (h₀ : 0 < α ∧ α < π) (h₁ : sin α + cos α = sqrt 10 / 5) : tan α = -3 :=
sorry

-- Problem (2): Prove the given expression equals -3/2
theorem expression_eq_neg_three_halves (h₀ : 0 < α ∧ α < π) (h₁ : sin α + cos α = sqrt 10 / 5) :
  (sin (2 * α)) / (sin α ^ 2 + sin α * cos α - cos (2 * α) - 1) = -3 / 2 :=
sorry

end tan_alpha_eq_neg_three_expression_eq_neg_three_halves_l599_599781


namespace g_is_even_function_l599_599145

def g (x : ℝ) : ℝ := Real.sqrt (x ^ 2 + 1)

theorem g_is_even_function : ∀ x : ℝ, g x = g (-x) := by
  sorry

end g_is_even_function_l599_599145


namespace club_co_presidents_l599_599300

def choose (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem club_co_presidents : choose 18 3 = 816 := by
  sorry

end club_co_presidents_l599_599300


namespace kamal_twice_age_in_future_l599_599872

theorem kamal_twice_age_in_future :
  ∃ x : ℕ, (K = 40) ∧ (K - 8 = 4 * (S - 8)) ∧ (K + x = 2 * (S + x)) :=
by {
  sorry 
}

end kamal_twice_age_in_future_l599_599872


namespace moli_bought_7_clips_l599_599181

theorem moli_bought_7_clips (R C S x : ℝ) 
  (h1 : 3*R + x*C + S = 120) 
  (h2 : 4*R + 10*C + S = 164) 
  (h3 : R + C + S = 32) : 
  x = 7 := 
by
  sorry

end moli_bought_7_clips_l599_599181


namespace period_of_f_decreasing_interval_of_f_minimum_positive_m_l599_599804

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2*x + π / 3) + 1

theorem period_of_f : 
∀ x : ℝ, f (x + π) = f x := 
by sorry

theorem decreasing_interval_of_f (k : ℤ) : 
∀ x : ℝ, k * π + π / 12 ≤ x ∧ x ≤ k * π + 7 * π / 12 → f' x < 0 := 
by sorry

noncomputable def g (x m : ℝ) : ℝ := 2 * sin (2*x + 2*m + π / 3) + 1

theorem minimum_positive_m : 
∀ k : ℤ, (∃ m > 0, g (π / 4) m = g (π / 4) (m + k * π / 2)) → m = π / 3 := 
by sorry

end period_of_f_decreasing_interval_of_f_minimum_positive_m_l599_599804


namespace inequality_inequality_l599_599159

theorem inequality_inequality 
  (n : ℕ) (a : Fin n → ℝ) (h_pos : ∀ i, 0 < a i) (k m : ℕ) (h_km : k > m) :
  (n - 1) * (Finset.univ.sum (λ i: Fin n, (a i)^m)) ≤ 
    Finset.univ.sum (λ i: Fin n, 
      (Finset.univ.sum (λ j: Fin n, if j ≠ i then (a j)^k else 0)) / (a i)^(k - m)) := 
sorry

end inequality_inequality_l599_599159


namespace other_root_of_quadratic_l599_599118

theorem other_root_of_quadratic (m : ℝ) (h : (2:ℝ) * (t:ℝ) = -6 ): 
  ∃ t, t = -3 :=
by
  sorry

end other_root_of_quadratic_l599_599118


namespace sum_of_digits_double_eq_sum_of_digits_half_eq_sum_of_digits_times_five_eq_l599_599660

def is_digit_permutation (M K : ℕ) : Prop :=
  ∀d ∈ finset.range 10, nat.digits 10 M.count d = nat.digits 10 K.count d

def sum_of_digits (n : ℕ) : ℕ :=
  (nat.digits 10 n).sum

theorem sum_of_digits_double_eq {M K : ℕ} (h : is_digit_permutation M K) : 
  sum_of_digits (2 * M) = sum_of_digits (2 * K) := sorry

theorem sum_of_digits_half_eq {M K : ℕ} (h : is_digit_permutation M K) (hM : even M) (hK : even K) : 
  sum_of_digits (M / 2) = sum_of_digits (K / 2) := sorry

theorem sum_of_digits_times_five_eq {M K : ℕ} (h : is_digit_permutation M K) : 
  sum_of_digits (5 * M) = sum_of_digits (5 * K) := sorry

end sum_of_digits_double_eq_sum_of_digits_half_eq_sum_of_digits_times_five_eq_l599_599660


namespace proj_magnitude_l599_599532

variables {V : Type*} [inner_product_space ℝ V]

open real_inner_product_space

/-- Given two vectors v and w in an inner product space, 
v dot w is -6 and the norm of w is 10, 
prove that the magnitude of the projection of v onto w is 0.6. -/
theorem proj_magnitude (v w : V) (h₁ : inner_product v w = -6) (h₂ : ∥w∥ = 10) : 
  ∥orthogonal_projection (submodule.span ℝ {w}) v∥ = 0.6 :=
sorry

end proj_magnitude_l599_599532


namespace find_quadratic_function_l599_599046

noncomputable def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := x^2 + a * x + b

theorem find_quadratic_function (a b : ℝ) :
  (∀ x, (quadratic_function a b (quadratic_function a b x - x)) / (quadratic_function a b x) = x^2 + 2023 * x + 1777) →
  a = 2025 ∧ b = 249 :=
by
  intro h
  sorry

end find_quadratic_function_l599_599046


namespace area_ratio_of_squares_l599_599907

theorem area_ratio_of_squares (hA : ∃ sA : ℕ, 4 * sA = 16)
                             (hB : ∃ sB : ℕ, 4 * sB = 20)
                             (hC : ∃ sC : ℕ, 4 * sC = 40) :
  (∃ aB aC : ℕ, aB = sB * sB ∧ aC = sC * sC ∧ aB * 4 = aC) := by
  sorry

end area_ratio_of_squares_l599_599907


namespace distance_nearest_tenth_l599_599318

theorem distance_nearest_tenth :
  ∃ d : ℝ, 
    (∀ (rect : affine_space ℝ) (p : point ℝ), 
      (rect.vertices = [point.mk 0 0, point.mk 3030 0, point.mk 3030 2020, point.mk 0 2020]) →
      (prob_within_distance p d = 3 / 4)) ∧ 
    d = 0.5 := 
sorry

end distance_nearest_tenth_l599_599318


namespace cuboctahedron_octahedron_volume_ratio_l599_599680

theorem cuboctahedron_octahedron_volume_ratio (s : ℝ)
  (hc : ∀ edge, edge ∈ cuboctahedron.edges → edge ∈ square_face ∧ edge ∈ triangle_face)
  (hs : ∀ f, f ∈ cuboctahedron.faces → ∃ g ∈ octahedron.faces, f ∩ g ≠ ∅) :
  let V_cuboctahedron := (6 * s^2 * (sqrt 2) + (4 * sqrt 2) / 3)
  let V_octahedron := (2 * s^2 * (sqrt 2) / 3)
  let r := V_octahedron / V_cuboctahedron
  100 * r^2 = 4 :=
by
  sorry

end cuboctahedron_octahedron_volume_ratio_l599_599680


namespace area_of_right_triangle_l599_599792

theorem area_of_right_triangle
  (a_sq b_sq h_sq : ℕ)
  (ha : a_sq = 100)
  (hb : b_sq = 64)
  (hh : h_sq = 121) :
  let a := Nat.sqrt a_sq,
      b := Nat.sqrt b_sq,
      h := Nat.sqrt h_sq in
  (1 / 2 : ℝ) * (a : ℝ) * (b : ℝ) = 40 :=
by
  sorry

end area_of_right_triangle_l599_599792


namespace new_sailor_weight_total_weight_bounds_l599_599980

-- We consider the average weight of the sailors before replacement as A
variable (A : ℝ)

-- Condition: Replacing a 56 kg sailor increases the average weight by 1 kg
def weight_of_new_sailor (A : ℝ) : Prop :=
  8 * (A + 1) - 8 * A = 64 - 56

-- Each sailor carries equipment weighing between 2 and 5 kg
def equipment_weight_bounds (A : ℝ) : Prop :=
  8 * (A + 1) + 16 ≤ 8 * (A + 1) + 40

-- Prove that the new sailor weighs 64 kg
theorem new_sailor_weight : weight_of_new_sailor A :=
by sorry

-- Prove the combined total weight including minimum and maximum equipment weights
theorem total_weight_bounds :
  8 * (A + 1) + 16 ≤ 8 * (A + 1) + weight_of_new_sailor A ∧ 8 * (A + 1) + 40 ≥ 8 * (A + 1) + weight_of_new_sailor A :=
by sorry

end new_sailor_weight_total_weight_bounds_l599_599980


namespace ellipse_equation_and_fixed_point_l599_599063

-- Definitions
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def slope (m : ℝ) := m = sqrt 5 / 2
def perimeter (p : ℝ) := p = 8
def distance (d : ℝ) := d = 2 * sqrt 5 / 3
def fixed_point (x y : ℝ) := (x = 7) ∧ (y = 0)

-- Conditions
variables {a b : ℝ} (h_ineq : 0 < b ∧ b < a)

-- Lean statement
theorem ellipse_equation_and_fixed_point :
  ∃ a b x y, 
    ellipse a b x y ∧ 
    (a = 2 ∧ b^2 = 3) ∧ 
    fixed_point 7 0 ∧ 
    x = 4 ∧
    ∀ (M : ℝ × ℝ) (P Q : ℝ × ℝ), 
      (∃ (A1 A2 : ℝ × ℝ),
        line_through M A1 ∧ line_through M A2 ∧
        P.1 = 4 ∧ Q.1 = 4 ∧
        angle PTQ = 90) :=
sorry

end ellipse_equation_and_fixed_point_l599_599063


namespace other_root_of_quadratic_l599_599117

theorem other_root_of_quadratic (m : ℝ) (root1 : ℝ) (h_roots : root1 = 2)
  (h_quadratic : ∀ x, x^2 + m * x - 6 = 0 ↔ x = root1 ∨ x = -3) : 
  ∃ root2 : ℝ, root2 = -3 :=
by
  use -3
  sorry

end other_root_of_quadratic_l599_599117


namespace events_per_coach_l599_599348

theorem events_per_coach {students events_per_student coaches events total_participations total_events : ℕ} 
  (h1 : students = 480) 
  (h2 : events_per_student = 4) 
  (h3 : (students * events_per_student) = total_participations) 
  (h4 : ¬ students * events_per_student ≠ total_participations)
  (h5 : total_participations = 1920) 
  (h6 : (total_participations / 20) = total_events) 
  (h7 : ¬ total_participations / 20 ≠ total_events)
  (h8 : total_events = 96)
  (h9 : coaches = 16) :
  (total_events / coaches) = 6 := sorry

end events_per_coach_l599_599348


namespace math_problem_l599_599485

variable (x y : ℚ)

theorem math_problem (h : 1.5 * x = 0.04 * y) : (y - x) / (y + x) = 73 / 77 := by
  sorry

end math_problem_l599_599485


namespace cat_food_finished_on_sunday_l599_599582

def cat_morning_consumption : ℚ := 1 / 2
def cat_evening_consumption : ℚ := 1 / 3
def total_food : ℚ := 10
def daily_consumption : ℚ := cat_morning_consumption + cat_evening_consumption
def days_to_finish_food (total_food daily_consumption : ℚ) : ℚ :=
  total_food / daily_consumption

theorem cat_food_finished_on_sunday :
  days_to_finish_food total_food daily_consumption = 7 := 
sorry

end cat_food_finished_on_sunday_l599_599582


namespace bike_riders_count_l599_599839

-- Define the conditions as variables and equations
axiom H B : ℕ
axiom cond1 : H = B + 178
axiom cond2 : H + B = 676

-- Define the theorem stating the number of bike riders
theorem bike_riders_count : B = 249 :=
by
  -- Proof skipped
  sorry

end bike_riders_count_l599_599839


namespace polar_to_rectangular_distance_l599_599508

open Real

theorem polar_to_rectangular_distance :
  ∀ (ρ θ : ℝ),
    ρ = 2 * cos θ → 
    polar_to_cartesian (2, π / 3) = (1, sqrt 3) → 
    dist (1, sqrt 3) (1, 0) = sqrt 3 := 
by
  intros ρ θ h1 h2
  sorry  -- Proof goes here.

end polar_to_rectangular_distance_l599_599508


namespace calculate_negative_subtraction_l599_599001

theorem calculate_negative_subtraction : -2 - (-3) = 1 :=
by sorry

end calculate_negative_subtraction_l599_599001


namespace max_blocks_fitting_in_box_l599_599265

theorem max_blocks_fitting_in_box :
  let box_volume := 3 * 2 * 3 in
  let block_volume := 2 * 2 * 1 in
  ∀ (box : ℝ × ℝ × ℝ), box = (3, 2, 3) →
  ∀ (block : ℝ × ℝ × ℝ), block = (2, 2, 1) →
  (box_volume / block_volume = 4) :=
by
  sorry

end max_blocks_fitting_in_box_l599_599265


namespace trailing_zeroes_500_l599_599000

open Nat

theorem trailing_zeroes_500! (a b : ℕ) (h₀ : 500! = a) (h₁ : trailing_zeroes a = 124) (h₂ : 200! = b) (h₃ : trailing_zeroes b = 49) :
  trailing_zeroes (500! + 200!) = 124 :=
by
  -- We can use Lean's inbuilt functions for calculations related to factorial and trailing zeroes
  sorry

end trailing_zeroes_500_l599_599000


namespace max_product_of_three_numbers_l599_599591

theorem max_product_of_three_numbers (n : ℕ) (h_n_pos : 0 < n) :
  ∃ a b c : ℕ, (a + b + c = 3 * n + 1) ∧ (∀ a' b' c' : ℕ,
        (a' + b' + c' = 3 * n + 1) →
        a' * b' * c' ≤ a * b * c) ∧
    (a * b * c = n^3 + n^2) :=
by
  sorry

end max_product_of_three_numbers_l599_599591


namespace find_increasing_function_l599_599703

def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < b → a < y → y < b → x < y → f x < f y

noncomputable def fA (x : ℝ) : ℝ := 3 - x
noncomputable def fB (x : ℝ) : ℝ := x^2 - 3 * x
noncomputable def fC (x : ℝ) : ℝ := 1 / x
noncomputable def fD (x : ℝ) : ℝ := |x|

theorem find_increasing_function :
  (is_increasing_on_interval fA 0 +∞ → false) ∧
  (is_increasing_on_interval fB 0 +∞ → false) ∧
  (is_increasing_on_interval fC 0 +∞ → false) ∧
  (is_increasing_on_interval fD 0 +∞) :=
by sorry

end find_increasing_function_l599_599703


namespace domain_f_2x_l599_599794

-- Given conditions as definitions
def domain_f_x_minus_1 (x : ℝ) := 3 < x ∧ x ≤ 7

-- The main theorem statement that needs a proof
theorem domain_f_2x : (∀ x : ℝ, domain_f_x_minus_1 (x-1) → (1 < x ∧ x ≤ 3)) :=
by
  -- Proof steps will be here, however, as requested, they are omitted.
  sorry

end domain_f_2x_l599_599794


namespace sequence_nth_term_l599_599381

theorem sequence_nth_term (n : ℕ) :
  let a : ℕ → ℤ := λ n, -(-4)^(n-1) in
  a n = -(-4)^(n-1) := by
  sorry

end sequence_nth_term_l599_599381


namespace total_net_worth_after_2_years_l599_599602

def initial_value : ℝ := 40000
def depreciation_rate : ℝ := 0.05
def initial_maintenance_cost : ℝ := 2000
def inflation_rate : ℝ := 0.03
def years : ℕ := 2

def value_at_end_of_year (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  List.foldl (λ acc _ => acc * (1 - rate)) initial_value (List.range years)

def cumulative_maintenance_cost (initial_maintenance_cost : ℝ) (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  List.foldl (λ acc year => acc + initial_maintenance_cost * ((1 + inflation_rate) ^ year)) 0 (List.range years)

def total_net_worth (initial_value : ℝ) (depreciation_rate : ℝ) (initial_maintenance_cost : ℝ) (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  value_at_end_of_year initial_value depreciation_rate years - cumulative_maintenance_cost initial_maintenance_cost inflation_rate years

theorem total_net_worth_after_2_years : total_net_worth initial_value depreciation_rate initial_maintenance_cost inflation_rate years = 32040 :=
  by
    sorry

end total_net_worth_after_2_years_l599_599602


namespace friends_came_over_later_l599_599685

def original_friends : ℕ := 4
def total_people : ℕ := 7

theorem friends_came_over_later : (total_people - original_friends = 3) :=
sorry

end friends_came_over_later_l599_599685


namespace max_value_of_p_l599_599889

theorem max_value_of_p
  (p q r s : ℕ)
  (h1 : p < 3 * q)
  (h2 : q < 4 * r)
  (h3 : r < 5 * s)
  (h4 : s < 90)
  (h5 : 0 < s)
  (h6 : 0 < r)
  (h7 : 0 < q)
  (h8 : 0 < p):
  p ≤ 5324 :=
by
  sorry

end max_value_of_p_l599_599889


namespace problem_statement_l599_599366

-- All possible values of x are given by the series:
-- x = b1/4 + b2/4^2 + ... + b20/4^20
-- where each b_i (for i = 1, 2, ..., 20) is either 1 or 3.
theorem problem_statement (b : Fin 20 → ℝ) (hb : ∀ i, b i = 1 ∨ b i = 3) :
    3/4 ≤ ∑ i in Finset.range 20, b i / 4^(i + 1) ∧ ∑ i in Finset.range 20, b i / 4^(i + 1) < 1 :=
sorry

end problem_statement_l599_599366


namespace solve_nested_root_l599_599353

noncomputable def nested_root_calculations : ℝ := real.cbrt (real.sqrt (0.000000027))

theorem solve_nested_root : nested_root_calculations = 0.234 :=
by
  sorry

end solve_nested_root_l599_599353


namespace usamopos_correct_l599_599259

theorem usamopos_correct : 
  ∀ (letters : set char) (pos : ℕ), 
    letters = {'A', 'M', 'O', 'S', 'U'} ∧ 
    pos = 115 → 
      (perm_number positions letters "USAMO" = pos) := 
by 
  intros letters pos h
  cases h
  sorry

end usamopos_correct_l599_599259


namespace minimum_value_of_x_sum_l599_599894

noncomputable def minimum_sum_x (x y : Fin 2016 -> ℝ) :=
  (∀ k : Fin 2016, x k ≥ 0) ∧
  (∀ k : Fin 2016, x k^2 + y k^2 = 1) ∧
  (∃ t : ℤ, ∑ i, y i = 2 * t + 1)

theorem minimum_value_of_x_sum :
  ∀ (x y : Fin 2016 -> ℝ), minimum_sum_x x y →
  ∑ i, x i = 1 :=
sorry

end minimum_value_of_x_sum_l599_599894


namespace jasmine_coin_problem_l599_599519

theorem jasmine_coin_problem : 
  ∃ (p n d : ℕ), p + n + d = 3030 ∧ 1 ≤ p ∧ 1 ≤ n ∧ 1 ≤ d ∧ 
  ((maximize (3030 + 4 * n + 9 * d) - minimize (3030 + 4 * n + 9 * d)) = 27243) := 
by
  sorry

end jasmine_coin_problem_l599_599519


namespace quadratic_has_real_roots_find_k_values_l599_599773

-- Define the quadratic equation
def quadratic_eq (k x : ℝ) : Prop := (x - 1)^2 + k * (x - 1) = 0

-- Prove that the equation quadratic_eq has real roots for any k ∈ ℝ
theorem quadratic_has_real_roots (k : ℝ) : ∃ x : ℝ, quadratic_eq k x :=
by {
  sorry
}

-- Prove the values of k based on given conditions with roots condition
theorem find_k_values (k : ℝ) (x1 x2 : ℝ) (h : quadratic_eq k x1 ∧ quadratic_eq k x2)
  (hx : x1^2 + x2^2 = 7 - x1 * x2) : k = 4 ∨ k = -1 :=
by {
  sorry
}

end quadratic_has_real_roots_find_k_values_l599_599773


namespace sin_2_angle_FQG_l599_599902

namespace ProofProblem

theorem sin_2_angle_FQG 
  (E F G H Q : Type) 
  (EF FG GH : ℝ) 
  (h1 : EF = FG)
  (h2 : FG = GH) 
  (cos_EQG : ℝ) 
  (cos_FQH : ℝ) 
  (h3 : cos_EQG = 3 / 5) 
  (h4 : cos_FQH = 1 / 5) : 
  sin (2 * ∠ F Q G) = 3 / 5 :=
by
  sorry

end ProofProblem

end sin_2_angle_FQG_l599_599902


namespace percentage_of_360_equals_115_2_l599_599965

theorem percentage_of_360_equals_115_2 (p : ℝ) (h : (p / 100) * 360 = 115.2) : p = 32 :=
by
  sorry

end percentage_of_360_equals_115_2_l599_599965


namespace increasing_interval_l599_599609

def f (x : ℝ) : ℝ := 2 * Real.sin (π / 6 - 2 * x)

theorem increasing_interval : 
  MonotoneOn f (Set.Icc (π / 3) (5 * π / 6)) :=
by
  sorry

end increasing_interval_l599_599609


namespace part_a_part_b_part_c_part_d_l599_599654

-- Part a
noncomputable def chebyshevT : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| n + 2 := λ x, 2 * x * chebyshevT (n + 1) x - chebyshevT n x

theorem part_a (x : ℝ) :
  chebyshevT 2 x - chebyshevT 1 x = 2 * x^2 - x - 1 ∧
  chebyshevT 3 x - chebyshevT 2 x = 4 * x^3 - 2 * x^2 - 3 * x + 1 ∧
  chebyshevT 4 x - chebyshevT 3 x = 8 * x^4 - 4 * x^3 - 8 * x^2 + 3 * x + 1 ∧
  chebyshevT 5 x - chebyshevT 4 x = 16 * x^5 - 8 * x^4 - 20 * x^3 + 8 * x^2 + 5 * x - 1 :=
by sorry

-- Part b
theorem part_b : 
  ∃ x : ℝ, (x = real.cos (2 * real.pi / 5) ∨ x = real.cos (4 * real.pi / 5)) ∧ (4 * x^2 + 2 * x - 1 = 0) :=
by sorry

-- Part c
theorem part_c : 
  ∃ x : ℝ, (x = real.cos (2 * real.pi / 7) ∨ x = real.cos (4 * real.pi / 7) ∨ x = real.cos (6 * real.pi / 7)) ∧ 
  (8 * x^3 + 4 * x^2 - 4 * x - 1 = 0) :=
by sorry

-- Part d
theorem part_d : 
  ∃ x : ℝ, (x = real.cos (2 * real.pi / 9) ∨ x = real.cos (4 * real.pi / 9) ∨ x = real.cos (8 * real.pi / 9)) ∧ 
  (8 * x^3 - 6 * x + 1 = 0) :=
by sorry

end part_a_part_b_part_c_part_d_l599_599654


namespace total_pure_acid_in_mixture_l599_599845

-- Definitions of the conditions
def solution1_volume : ℝ := 8
def solution1_concentration : ℝ := 0.20
def solution2_volume : ℝ := 5
def solution2_concentration : ℝ := 0.35

-- Proof statement
theorem total_pure_acid_in_mixture :
  solution1_volume * solution1_concentration + solution2_volume * solution2_concentration = 3.35 := by
  sorry

end total_pure_acid_in_mixture_l599_599845


namespace tangent_line_a_neg1_symmetry_values_range_for_extreme_values_l599_599088

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (1 / x + a) * Real.log (1 + x)

-- Part 1: Prove the tangent line equation at a = -1
theorem tangent_line_a_neg1 :
  let a := -1
  f 1 a = 0 ∧ f'(1) = -Real.log 2 →
  ∀ x, f'(1) * (x - 1) = -Real.log 2 * (x - 1) := sorry

-- Part 2: Prove the symmetry and find values of a and b
theorem symmetry_values : 
  ∃ (a b : ℝ), b = -1/2 ∧ f (1) a = (1 + a) * Real.log (2) ∧ f (-2) a = (2 - a) * Real.log (2) →
  a = 1/2 ∧ b = -1/2 := sorry

-- Part 3: Prove range of a for extreme values in (0, +∞)
theorem range_for_extreme_values : 
  (∀ x, ∃ a, 0 < a ∧ a < 1/2 → has_extreme_value (f x a)):= sorry

end tangent_line_a_neg1_symmetry_values_range_for_extreme_values_l599_599088


namespace geometric_sequence_a8_eq_pm1_l599_599856

variable {R : Type*} [LinearOrderedField R]

theorem geometric_sequence_a8_eq_pm1 :
  ∀ (a : ℕ → R), (∀ n : ℕ, ∃ r : R, r ≠ 0 ∧ a n = a 0 * r ^ n) → 
  (a 4 + a 12 = -3) ∧ (a 4 * a 12 = 1) → 
  (a 8 = 1 ∨ a 8 = -1) := by
  sorry

end geometric_sequence_a8_eq_pm1_l599_599856


namespace evaluate_expression_l599_599529

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 7
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 7

theorem evaluate_expression : ( (1 / a) + (1 / b) + (1 / c) + (1 / d) ) ^ 2 = (7 / 49) :=
by
  sorry

end evaluate_expression_l599_599529


namespace greatest_difference_units_digit_l599_599222

theorem greatest_difference_units_digit :
  let digits_sum := 8 + 5 + 7 + 4
  in ∀ (x : ℕ), (0 <= x ∧ x < 10) → (digits_sum + x) % 3 = 0 → (∀ (a b : ℕ), (0 <= a ∧ a < 10) → (0 <= b ∧ b < 10) → (digits_sum + a) % 3 = 0 → 
    (digits_sum + b) % 3 = 0 → (a = 0 ∨ a = 3 ∨ a = 6 ∨ a = 9) ∧ (b = 0 ∨ b = 3 ∨ b = 6 ∨ b = 9) → (9 = |a - b|)) :=
sorry

end greatest_difference_units_digit_l599_599222


namespace four_triangles_congruent_l599_599060

theorem four_triangles_congruent
  (A B C D : Point) 
  (h1 : same_inradius (triangle D A B) (triangle A B C) (triangle B C D) (triangle C D A)) :
  congruent (triangle D A B) (triangle A B C) (triangle B C D) (triangle C D A) := 
sorry

end four_triangles_congruent_l599_599060


namespace card_of_B_l599_599248

variable (U A B : Type)
variable (card_U : ℕ) 
variable (card_not_A_or_B : ℕ) 
variable (card_A_and_B : ℕ) 
variable (card_A : ℕ) 

-- Conditions
def condition_1 := card_U = 192
def condition_2 := card_not_A_or_B = 59
def condition_3 := card_A_and_B = 23
def condition_4 := card_A = 107

-- Theorem: Proving number of members of set U that are members of set B
theorem card_of_B (card_B : ℕ) (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) : card_B = 49 := 
sorry

end card_of_B_l599_599248


namespace trig_expression_value_l599_599071

variable (θ : ℝ)

theorem trig_expression_value (h : Real.cos (π + θ) = 1 / 3) : 
  (Real.cos (2 * π - θ)) / 
  (Real.sin (π / 2 + θ) * Real.cos (π - θ) + Real.cos (-θ)) = 3 / 4 :=
by
  sorry

end trig_expression_value_l599_599071


namespace jo_vs_kate_sum_l599_599523

def jo_sum (n : ℕ) : ℕ := (n * (n + 1)) / 2

def kate_round (n : ℕ) : ℕ :=
  if n % 10 ≤ 4 then (n / 10) * 10
  else (n / 10 + 1) * 10

def kate_sum (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, kate_round (i + 1)

theorem jo_vs_kate_sum (n : ℕ) : jo_sum n = jo_sum 50 - kate_sum 50 := by
  sorry

example : jo_vs_kate_sum 50 = 25 := by
  sorry

end jo_vs_kate_sum_l599_599523


namespace largest_number_of_primes_l599_599854

-- Define the nature of our table cells
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0
def is_product_of_two_primes (n : ℕ) := ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p * q

-- Conditions of the problem
def table_valid (table : Fin 80 → Fin 80 → ℕ) : Prop :=
  (∀ i j, table i j > 0 ∧ (is_prime (table i j) ∨ is_product_of_two_primes (table i j))) ∧
  (∀ i j, ∃ k, (k ≠ j ∧ ¬ Nat.coprime (table i j) (table i k)) ∨
              ∃ l, (l ≠ i ∧ ¬ Nat.coprime (table i j) (table l j)))

-- The maximum number of primes in the table
def max_primes_in_table : ℕ := 4266

-- Main theorem statement
theorem largest_number_of_primes (table : Fin 80 → Fin 80 → ℕ) (valid : table_valid table):
  ∃ primes_count : ℕ, primes_count ≤ max_primes_in_table := sorry

end largest_number_of_primes_l599_599854


namespace odd_function_property_l599_599548

theorem odd_function_property {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = - f x) (h2 : ∀ x, f (1 + x) = f (-x)) (h3 : f (-1 / 3) = 1 / 3) : f (5 / 3) = 1 / 3 := 
sorry

end odd_function_property_l599_599548


namespace ivan_chess_false_l599_599983

theorem ivan_chess_false (n : ℕ) :
  ∃ n, n + 3 * n + 6 * n = 64 → False :=
by
  use 6
  sorry

end ivan_chess_false_l599_599983


namespace problem1_problem2_problem3_l599_599061

noncomputable def a (n : ℕ) (k : ℝ) : ℝ :=
if n = 1 then 1 - 3 * k else 4^(n - 1) - 3 * a (n - 1) k

def is_geometric_seq (seq : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, seq (n + 1) = r * seq n

theorem problem1 (k : ℝ) (h : k ≠ 2/7) :
  is_geometric_seq (λ n, a n k - 4^n / 7) :=
sorry

theorem problem2 (k : ℝ) :
  ∀ n : ℕ, 
  a n k = if k = 2/7 then (4 ^ (n:ℕ)) / 7 
          else (4 ^ n) / 7 + ((6 / 7) - 3 * k) * (-3)^((n:ℕ)-1) :=
sorry

theorem problem3 (k : ℝ) (h : ∀ n : ℕ, a n k < a (n + 1) k) :
  2/7 ≤ k ∧ k < 34/63 :=
sorry

end problem1_problem2_problem3_l599_599061


namespace smallest_possible_value_l599_599482

open Nat

theorem smallest_possible_value (c d : ℕ) (hc : c > d) (hc_pos : 0 < c) (hd_pos : 0 < d) (odd_cd : ¬Even (c + d)) :
  (∃ (y : ℚ), y > 0 ∧ y = (c + d : ℚ) / (c - d) + (c - d : ℚ) / (c + d) ∧ y = 10 / 3) :=
by
  sorry

end smallest_possible_value_l599_599482


namespace other_root_of_quadratic_l599_599116

theorem other_root_of_quadratic (m : ℝ) (root1 : ℝ) (h_roots : root1 = 2)
  (h_quadratic : ∀ x, x^2 + m * x - 6 = 0 ↔ x = root1 ∨ x = -3) : 
  ∃ root2 : ℝ, root2 = -3 :=
by
  use -3
  sorry

end other_root_of_quadratic_l599_599116


namespace smallest_integer_cube_ends_in_576_l599_599757

theorem smallest_integer_cube_ends_in_576 : ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 576 ∧ ∀ m : ℕ, m > 0 → m^3 % 1000 = 576 → m ≥ n := 
by
  sorry

end smallest_integer_cube_ends_in_576_l599_599757


namespace intersection_points_concyclic_and_radius_l599_599604

noncomputable def tangent_intersection_circle_radius (O1 O2 : Point) (R1 R2 a : ℝ) : Prop :=
  ∃ (P Q R S : Point), 
  (tangent_to_circle P O1 R1 ∧ tangent_to_circle P O2 R2 ∧
   tangent_to_circle Q O1 R1 ∧ tangent_to_circle Q O2 R2 ∧
   tangent_to_circle R O1 R1 ∧ tangent_to_circle R O2 R2 ∧
   tangent_to_circle S O1 R1 ∧ tangent_to_circle S O2 R2 ∧
   are_concyclic P Q R S ∧
   circle_radius P Q R S = (sqrt (a^2 - (R1 - R2)^2) / 2))

theorem intersection_points_concyclic_and_radius 
  (O1 O2 : Point) (R1 R2 a : ℝ) 
  (h_non_intersecting : distance O1 O2 = a ∧ a > R1 + R2):
  tangent_intersection_circle_radius O1 O2 R1 R2 a :=
sorry

end intersection_points_concyclic_and_radius_l599_599604


namespace range_of_k_real_roots_l599_599811

variable (k : ℝ)
def quadratic_has_real_roots : Prop :=
  let a := k - 1
  let b := 2
  let c := 1
  let Δ := b^2 - 4 * a * c
  Δ ≥ 0 ∧ a ≠ 0

theorem range_of_k_real_roots :
  quadratic_has_real_roots k ↔ (k ≤ 2 ∧ k ≠ 1) := by
  sorry

end range_of_k_real_roots_l599_599811


namespace series_sum_equals_three_fourths_l599_599727

noncomputable def infinite_series_sum : ℝ :=
  (∑' n : ℕ, (3 * (n + 1) + 2) / ((n + 1) * (n + 1 + 1) * (n + 1 + 3)))

theorem series_sum_equals_three_fourths :
  infinite_series_sum = 3 / 4 :=
sorry

end series_sum_equals_three_fourths_l599_599727


namespace odd_function_property_l599_599551

theorem odd_function_property {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = - f x) (h2 : ∀ x, f (1 + x) = f (-x)) (h3 : f (-1 / 3) = 1 / 3) : f (5 / 3) = 1 / 3 := 
sorry

end odd_function_property_l599_599551


namespace abs_eq_5_iff_l599_599102

   theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 :=
   by
     sorry
   
end abs_eq_5_iff_l599_599102


namespace intersection_of_sets_l599_599814

noncomputable def setA : Set ℝ := { x | (x + 2) / (x - 2) ≤ 0 }
noncomputable def setB : Set ℝ := { x | x ≥ 1 }
noncomputable def expectedSet : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem intersection_of_sets : (setA ∩ setB) = expectedSet := by
  sorry

end intersection_of_sets_l599_599814


namespace sum_elements_l599_599494

noncomputable def a : ℕ → ℕ → ℕ
| 1, j => j
| 2, j => if j = 9 then 1 else j + 1
| 3, j => if j ≤ 8 then j + 1 else 1
| 4, j => if j = 9 then 2 else if j = 8 then 1 else j + 1
| i + 1, j => (a i j + 1) % 9 + 1
| _, _ => 0

theorem sum_elements (n : ℕ) (hn : n = 9) :
  (∑ i in Finset.range 9, ∑ j in Finset.range 9, if 2 * i < j then a (i + 1) (j + 1) else 0) = 88 := sorry

end sum_elements_l599_599494


namespace tan_sum_trig_l599_599481

variable {α β : ℝ}

-- Condition
def sin_condition (α β : ℝ) := sin α = 3 * sin (α - 2 * β)

-- Theorem statement
theorem tan_sum_trig (h : sin_condition α β) : tan (α - β) + 2 * tan β = 4 * tan β := 
by 
  sorry

end tan_sum_trig_l599_599481


namespace alice_average_speed_l599_599330

def average_speed (distance1 speed1 distance2 speed2 totalDistance totalTime : ℚ) :=
  totalDistance / totalTime

theorem alice_average_speed : 
  let d1 := 45
  let s1 := 15
  let d2 := 15
  let s2 := 45
  let totalDistance := d1 + d2
  let totalTime := (d1 / s1) + (d2 / s2)
  average_speed d1 s1 d2 s2 totalDistance totalTime = 18 :=
by
  sorry

end alice_average_speed_l599_599330


namespace pentagon_theorem_l599_599855

noncomputable def pentagon (A B C D E : Type) [convex_set A B C D E] : Prop :=
  -- Conditions
  (AB = AC) ∧
  (AD = AE) ∧
  (∠ CAD = ∠ ABE + ∠ AEB) ∧
  (midpoint M BE)

theorem pentagon_theorem (A B C D E : Type) [convex_set A B C D E]
  (AB AC AD AE BE : segment)
  (M : point)
  (h1 : AB = AC)
  (h2 : AD = AE)
  (h3 : ∠ CAD = ∠ ABE + ∠ AEB)
  (h4 : M = midpoint BE) :
  CD = 2 * AM :=
by
  sorry -- This is where the proof would go

end pentagon_theorem_l599_599855


namespace find_width_of_cistern_l599_599675

-- Define the variables for length, height, total wet surface area, and the width we want to find.
def length := 6 -- meters
def height := 1.25 -- meters
def wet_surface_area := 57.5 -- square meters

def width_is (w : Real) : Prop :=
  6 * w + 2 * (6 * 1.25) + 2 * (w * 1.25) = 57.5

theorem find_width_of_cistern : width_is 5 :=
by
  unfold width_is
  sorry

end find_width_of_cistern_l599_599675


namespace arithmetic_sequence_a8_S8_l599_599987

open Real

theorem arithmetic_sequence_a8_S8 :
  ∃ (a_n : ℕ → ℝ) (a_1 d : ℝ), (a_n 5 = 10) ∧ ((∑ i in Finset.range 5, a_n i) = 5) ∧
  (a_n 7 = 16) ∧ ((∑ i in Finset.range 8, a_n i) = 44) :=
begin
  sorry
end

end arithmetic_sequence_a8_S8_l599_599987


namespace no_more_than_48_single_color_cards_l599_599628

/-- Definition of the problem. --/
def has_color (color : string) (c : card) : Prop := sorry

def different_color_sides (c : card) : Prop := sorry

theorem no_more_than_48_single_color_cards
  (cards : set card)
  (h1 : ∀ (s : finset card), s.cardinality = 30 → ∃ c ∈ s, has_color "red" c)
  (h2 : ∀ (s : finset card), s.cardinality = 40 → ∃ c ∈ s, has_color "yellow" c)
  (h3 : ∀ (s : finset card), s.cardinality = 50 → ∃ c ∈ s, has_color "blue" c)
  (h4 : (cards.filter different_color_sides).cardinality = 20) :
  ∃ (single_color_cards : finset card), single_color_cards.cardinality ≤ 48 ∧ ∀ c ∈ single_color_cards, ¬ different_color_sides c :=
sorry

end no_more_than_48_single_color_cards_l599_599628


namespace colonization_combinations_l599_599910

theorem colonization_combinations :
  let earth_like_planets := 8
  let mars_like_planets := 12
  ∃ combs,
    combs = {c : ℕ × ℕ | 
                c.1 <= earth_like_planets ∧ 
                c.2 <= mars_like_planets ∧ 
                2 * c.1 + c.2 = 24} ∧ 
    combs.card = 1052 :=
by
  sorry

end colonization_combinations_l599_599910


namespace geometry_problem_l599_599515

variables {A B C D E F : Type}
variable [MetricSpace A]
variable [MetricSpace B]
variable [MetricSpace C]
variable [MetricSpace D]
variable [MetricSpace E]
variable [MetricSpace F]

/-- In triangle ABC, D is an arbitrary point on side BC. Let E be the foot of the perpendicular
from D to the altitude from B, and let F be the foot of the perpendicular from D to the altitude
from C. Prove that AB * CF + AC * BE is equal to twice the area of the triangle ABC. -/
theorem geometry_problem
  (ABC : Triangle A B C)
  (D : Point BC)
  (E : FootPerpendicular D (Altitude B))
  (F : FootPerpendicular D (Altitude C)) :
  (segment_length A B * segment_length C F + segment_length A C * segment_length B E = 2 * area ABC) :=
sorry

end geometry_problem_l599_599515


namespace series_sum_is_correct_l599_599014
open Complex

noncomputable def series_sum (n : ℕ) : ℂ :=
  ∑ k in finset.range (n + 1), (-1:ℂ)^k * (k + 1) * (complex.I^k)

theorem series_sum_is_correct (n : ℕ) (h : n % 3 = 0) : 
  series_sum n = -((2 * n) / 3 : ℂ) * (1 + complex.I) :=
sorry

end series_sum_is_correct_l599_599014


namespace arithmetic_seq_a8_l599_599790

theorem arithmetic_seq_a8 : ∀ (a : ℕ → ℤ), 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → 
  (a 5 + a 6 = 22) → 
  (a 3 = 7) → 
  a 8 = 15 :=
by
  intros a ha_arithmetic hsum h3
  sorry

end arithmetic_seq_a8_l599_599790


namespace lengthDE_is_correct_l599_599598

noncomputable def triangleBase : ℝ := 12

noncomputable def triangleArea (h : ℝ) : ℝ := (1 / 2) * triangleBase * h

noncomputable def projectedArea (h : ℝ) : ℝ := 0.16 * triangleArea h

noncomputable def lengthDE (h : ℝ) : ℝ := 0.4 * triangleBase

theorem lengthDE_is_correct (h : ℝ) :
  lengthDE h = 4.8 :=
by
  simp [lengthDE, triangleBase, triangleArea, projectedArea]
  sorry

end lengthDE_is_correct_l599_599598


namespace train_crossing_time_l599_599824

def train_length := 145 -- meters
def train_speed := 54 * 1000 / 3600 -- convert 54 kmph to m/s, resulting in 15 m/s
def bridge_length := 660 -- meters
def total_distance := train_length + bridge_length -- 805 meters
def crossing_time := total_distance / (train_speed : ℝ) -- 805 / 15 (in seconds)

theorem train_crossing_time : crossing_time = 53.67 := by
  -- The theorem should prove that the crossing time is 53.67 seconds
  sorry

end train_crossing_time_l599_599824


namespace ratio_BH_divides_FE_l599_599349

-- Definitions based on the given conditions
variables {A B C D E F G H J : Point}
variables [EuclideanGeometry]

-- Hypotheses
def isAltitude (BH : Line) (B H : Point) (triangle : Triangle B H AnotherPoint) : Prop := sorry
def isFootOfPerpendicular (P : Point) (line : Line) (from : Point) : Prop := sorry
def areaRatio (tr1 tr2 : Triangle) (r : ℝ) : Prop := sorry

-- Given conditions
variable (BH : Line)
variable (triangleABC : Triangle A B C)
variable (D : Point)
variable (AD : Line)
variable (CD : Line)
variable (E := Intersection AD BC)
variable (F := Intersection CD AB)
variable (G : Point)
variable (J : Point)

-- Specifying that D is on the altitude BH of triangle ABC
axiom D_on_BH : isAltitude BH D triangleABC

-- Specifying the intersections
axiom AD_intersects_BC_at_E : E = Intersection AD BC
axiom CD_intersects_AB_at_F : F = Intersection CD AB

-- Specifying the projections
axiom G_is_foot_from_F_to_AC : isFootOfPerpendicular G (AC_of_triangleABC) F
axiom J_is_foot_from_E_to_AC : isFootOfPerpendicular J (AC_of_triangleABC) E

-- Area condition
axiom area_condition : areaRatio (Triangle H E J) (Triangle H F G) 2

-- Prove the required ratio
theorem ratio_BH_divides_FE : ratio (Segment B H) (Segment E F) = sqrt 2 / 1 := 
sorry -- Proof is omitted

end ratio_BH_divides_FE_l599_599349


namespace no_real_solution_l599_599384

theorem no_real_solution :
  ¬ ∃ x : ℝ, (1 / (x + 2) + 8 / (x + 6) ≥ 2) ∧ (5 / (x + 1) - 2 ≤ 1) :=
by
  sorry

end no_real_solution_l599_599384


namespace number_of_students_passed_both_tests_l599_599129

theorem number_of_students_passed_both_tests 
  (total_students : ℕ) 
  (passed_long_jump : ℕ) 
  (passed_shot_put : ℕ) 
  (failed_both_tests : ℕ) 
  (students_with_union : ℕ := total_students) :
  (students_with_union = passed_long_jump + passed_shot_put - passed_both_tests + failed_both_tests) 
  → passed_both_tests = 25 :=
by sorry

end number_of_students_passed_both_tests_l599_599129


namespace number_of_subsets_with_square_or_cube_l599_599939

def is_square_or_cube (n : ℕ) : Prop :=
  (∃ k : ℕ, k * k = n) ∨ (∃ k : ℕ, k * k * k = n)

theorem number_of_subsets_with_square_or_cube (A : set ℕ) (hA : A = {i | i ∈ finset.range 101}) :
  (∃ M ⊆ A, ∃ x ∈ M, is_square_or_cube x) ->
  finset.card {M | M ⊆ A ∧ ∃ x ∈ M, is_square_or_cube x} = 2^100 - 2^87 := by
  sorry

end number_of_subsets_with_square_or_cube_l599_599939


namespace sequence_u19_l599_599365

open Nat

-- Definition of the sequence
def sequence (b : ℕ) : ℕ → ℝ
| 1       := b
| (n + 1) := 2 / (sequence b n - 2)

-- The statement to prove
theorem sequence_u19 (b : ℕ) (hb : b > 0) :
  sequence b 19 = 2 * (b - 2) / (b - 4) := 
sorry

end sequence_u19_l599_599365


namespace investment_problem_l599_599328

theorem investment_problem (total_profit : ℝ)
  (investment_A : ℝ) (time_A : ℝ)
  (investment_B : ℝ) (time_B : ℝ)
  (investment_C : ℝ) (time_C : ℝ)
  (work_percent : ℝ)
  (C_share : ℝ) :
  investment_A = 6500 → time_A = 6 →
  investment_B = 8400 →
  investment_C = 10000 → time_C = 3 →
  work_percent = 0.05 →
  total_profit = 7400 → C_share = 1900 →
  time_B = 5 :=
begin
  intros,
  sorry
end

end investment_problem_l599_599328


namespace correct_calculation_among_four_l599_599649

theorem correct_calculation_among_four:
  (2 + Real.sqrt 3 ≠ 2 * Real.sqrt 3) ∧
  (Real.sqrt ((-2)^2) ≠ -2) ∧
  (Real.sqrt 9 ≠ ±3) ∧
  (Real.sqrt 27 - Real.sqrt 3 = 2 * Real.sqrt 3) :=
by
  -- Proof steps would go here
  sorry

end correct_calculation_among_four_l599_599649


namespace range_of_p_l599_599812

noncomputable def a_n (p : ℝ) (n : ℕ) : ℝ := -2 * n + p
noncomputable def b_n (n : ℕ) : ℝ := 2 ^ (n - 7)

noncomputable def c_n (p : ℝ) (n : ℕ) : ℝ :=
if a_n p n <= b_n n then a_n p n else b_n n

theorem range_of_p (p : ℝ) :
  (∀ n : ℕ, n ≠ 10 → c_n p 10 > c_n p n) ↔ 24 < p ∧ p < 30 :=
sorry

end range_of_p_l599_599812


namespace ranking_l599_599716

-- Define the rankings
variables (C H B : ℝ)

-- Conditions based on the problem statement
def condition1 : Prop := C > H
def condition2 : Prop := H > B

-- Theorem statement representing the problem
theorem ranking (C H B : ℝ) (h1 : condition1 C H) (h2 : condition2 H B) : C > H ∧ H > B :=
by {
  split;
  assumption,
}

end ranking_l599_599716


namespace no_real_solution_l599_599914

theorem no_real_solution (x : ℝ) : 
  4 * x^(1/3) - 3 * (x / x^(2/3)) = 10 + 2 * x^(1/3) + x^(2/3) → 
  ¬∃ y : ℝ, (y = x^(1/3) ∧ y^2 + y + 10 = 0) := 
begin
  intros h,
  have h1 : x^(1/3) = x^(1 - 2/3), by {field_simp, ring},
  have h2 : x^(1/3) = (x^(2/3)) * (x / x^(2/3)), by {field_simp, ring},
  fold h1 h2,
  sorry
end

end no_real_solution_l599_599914


namespace find_n_for_geometric_series_l599_599707

theorem find_n_for_geometric_series
  (n : ℝ)
  (a1 : ℝ := 12)
  (a2 : ℝ := 4)
  (r1 : ℝ)
  (S1 : ℝ)
  (b1 : ℝ := 12)
  (b2 : ℝ := 4 + n)
  (r2 : ℝ)
  (S2 : ℝ) :
  (r1 = a2 / a1) →
  (S1 = a1 / (1 - r1)) →
  (S2 = 4 * S1) →
  (r2 = b2 / b1) →
  (S2 = b1 / (1 - r2)) →
  n = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_n_for_geometric_series_l599_599707


namespace sum_of_slopes_is_5_l599_599600

-- Coordinates of vertices
def A : (ℤ × ℤ) := (10, 50)
def D : (ℤ × ℤ) := (11, 53)

-- The slope function
def slope (p1 p2 : (ℤ × ℤ)) : ℚ :=
  if p2.1 - p1.1 = 0 then 0 else (p2.2 - p1.2 : ℚ) / (p2.1 - p1.1 : ℚ)

-- The isosceles trapezoid condition
def is_int_coords (p : ℤ × ℤ) : Prop :=
  p.1 ∈ ℤ ∧ p.2 ∈ ℤ

def isosceles_trapezoid (A B C D : (ℤ × ℤ)) : Prop :=
  ∃ m : ℚ, slope A B = slope C D ∧ is_int_coords B ∧ is_int_coords C

theorem sum_of_slopes_is_5 :
  ∃ p q : ℕ, (p.gcd q = 1) ∧ (p + q = 5) ∧ (abs (slope (0,0) B) + abs (slope (a, b) C) = 3 / 2) :=
sorry

end sum_of_slopes_is_5_l599_599600


namespace CF_eq_AH_l599_599317

variables {H A B D E C : Type} [RegularPolygon A B C D E]
variable (circle_centerH : Circle (center := H) (radius := HE))
variable (H_lies_on_AB : LiesOn H (Segment A B))
variables (G F : Point)
variables (meets_DE : circle_centerH ∩ (Segment D E) = G)
variables (meets_CD : circle_centerH ∩ (Segment C D) = F)
variables (known_DG_AH : SegmentLength (Segment D G) = SegmentLength (Segment A H))

theorem CF_eq_AH :
  SegmentLength (Segment C F) = SegmentLength (Segment A H) :=
by
  sorry

end CF_eq_AH_l599_599317


namespace min_value_y1_y2_sq_l599_599091

theorem min_value_y1_y2_sq (k : ℝ) (y1 y2 : ℝ) :
  ∃ y1 y2, y1 + y2 = 4 / k ∧ y1 * y2 = -4 ∧ y1^2 + y2^2 = 8 :=
sorry

end min_value_y1_y2_sq_l599_599091


namespace telescoping_series_sum_l599_599724

theorem telescoping_series_sum :
  (∑' (n : ℕ) in (Finset.range (0) \ Finset.singleton (0)), (↑(3 * n + 2) / (↑n * (↑n + 1) * (↑n + 3)))) = (5 / 6) := sorry

end telescoping_series_sum_l599_599724


namespace problem_solution_l599_599195

noncomputable def P : ℤ := -7
noncomputable def Q : ℤ := 8
noncomputable def R : ℤ := 21
noncomputable def S : ℕ := 1

def condition1 : S > 0 := by
  exact Nat.one_pos

def condition2 : ¬ ∃ (p : ℕ), Nat.Prime p ∧ p * p ∣ Q := by
  intro h
  obtain ⟨p, hp_prime, hp_square_divides_Q⟩ := h
  have := Nat.not_prime_mul hp_prime
  exact this ⟨p, hp_prime, hp_square_divides_Q⟩

def GCD_P_R_S_is_1 : Nat.gcd (Nat.gcd P.natAbs R.natAbs) S = 1 := by
  rw [Int.gcd_eq_natAbs, Int.natAbs_of_nonneg (zero_le P), Int.natAbs_of_nonneg (zero_le R)]
  rfl

theorem problem_solution : P + Q + R + S = 23 := by
  sorry

end problem_solution_l599_599195


namespace integer_sequence_perfect_square_l599_599281

noncomputable def seq (a : ℕ → ℝ) : Prop :=
a 1 = 1 ∧ a 2 = 4 ∧ ∀ n ≥ 2, a n = (a (n - 1) * a (n + 1) + 1) ^ (1 / 2)

theorem integer_sequence {a : ℕ → ℝ} : 
  seq a → ∀ n, ∃ k : ℤ, a n = k := 
by sorry

theorem perfect_square {a : ℕ → ℝ} :
  seq a → ∀ n, ∃ k : ℤ, 2 * a n * a (n + 1) + 1 = k ^ 2 :=
by sorry

end integer_sequence_perfect_square_l599_599281


namespace max_sum_dist_correct_l599_599764

noncomputable def max_sum_dist (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n / 2) ^ 2 else (n / 2) ^ 2 + (n / 2)

theorem max_sum_dist_correct (n : ℕ) (xs : Fin n → ℝ) (h : ∀ i, 0 ≤ xs i ∧ xs i ≤ 1) :
  let s := ∑ i in Finset.offDiag (Finset.univ : Finset (Fin n)), |xs i.fst - xs i.snd| in
  s ≤ max_sum_dist n :=
by
  sorry

end max_sum_dist_correct_l599_599764


namespace tan_alpha_minus_pi_over_4_eq_minus_one_seventh_l599_599819

open Real

theorem tan_alpha_minus_pi_over_4_eq_minus_one_seventh
  (α : ℝ)
  (h1 : α ∈ Ioo (π / 2) (3 * π / 2))
  (h2 : ∀ {a b : fin 2 → ℝ}, a = ![sin α, cos (2 * α)] → b = ![1 - 2 * sin α, -1] → ((a 0) * (b 0) + (a 1) * (b 1)) = -8 / 5) :
  tan (α - π / 4) = -1 / 7 := sorry

end tan_alpha_minus_pi_over_4_eq_minus_one_seventh_l599_599819


namespace minimum_positive_period_of_f_is_pi_l599_599231

noncomputable def f (x : ℝ) : ℝ := (sin x + sin (3 * x)) / (cos x + cos (3 * x))

def domain (x : ℝ) : Prop :=
  ¬ ∃ k : ℤ, x = k * π + π / 2 ∨ x = k * π / 2 + π / 4

theorem minimum_positive_period_of_f_is_pi :
  (∀ x : ℝ, domain x → f (x + π) = f x) ∧
  (∀ p : ℝ, (∀ x : ℝ, domain x → f (x + p) = f x) → p ≥ π) :=
sorry

end minimum_positive_period_of_f_is_pi_l599_599231


namespace g_g_neg3_is_correct_l599_599438

def g (x : ℚ) : ℚ := x⁻² + (x⁻² / (1 + x⁻²))

theorem g_g_neg3_is_correct : g (g (-3)) = 68679921 / 3050901 := by 
  sorry

end g_g_neg3_is_correct_l599_599438


namespace Martha_children_l599_599892

-- Define the variables
variables (C: ℝ) (n: ℝ)

-- Given the conditions
def conditions : Prop := C = 54 ∧ (∀ c : ℝ, c = 18.0)

-- Given the equation to find the number of children
def num_children : ℝ := C / 18

-- The proof statement that needs to be proved
theorem Martha_children : conditions C → num_children C = 3 := by sorry

end Martha_children_l599_599892


namespace count_divisors_divisible_exactly_2007_l599_599825

-- Definitions and conditions
def prime_factors_2006 : List Nat := [2, 17, 59]

def prime_factors_2006_pow_2006 : List (Nat × Nat) := [(2, 2006), (17, 2006), (59, 2006)]

def number_of_divisors (n : Nat) : Nat :=
  prime_factors_2006_pow_2006.foldl (λ acc ⟨p, exp⟩ => acc * (exp + 1)) 1

theorem count_divisors_divisible_exactly_2007 : 
  (number_of_divisors (2^2006 * 17^2006 * 59^2006) = 3) :=
  sorry

end count_divisors_divisible_exactly_2007_l599_599825


namespace factorial_equation_solution_l599_599382

theorem factorial_equation_solution :
  ∃ a b n : ℕ, 2^a + 5^b + 1 = n! ∧ (a = 2 ∧ b = 0 ∧ n = 3) :=
by {
  use 2, 0, 3,
  split,
  {
    norm_num,
  },
  {
    tauto,
  },
}

end factorial_equation_solution_l599_599382


namespace expression_as_fraction_l599_599378

theorem expression_as_fraction :
  1 + (4 / (5 + (6 / 7))) = (69 : ℚ) / 41 := 
by
  sorry

end expression_as_fraction_l599_599378


namespace other_root_of_quadratic_eq_l599_599122

theorem other_root_of_quadratic_eq (m : ℝ) (t : ℝ) (h1 : (polynomial.X ^ 2 + polynomial.C m * polynomial.X + polynomial.C (-6)).roots = {2, t}) : t = -3 :=
sorry

end other_root_of_quadratic_eq_l599_599122


namespace augmented_matrix_l599_599216

-- Define the linear equations
def eq1 : ℕ × ℕ × ℕ := (2, -1, 1)
def eq2 : ℕ × ℕ × ℕ := (1, 3, 2)

-- Prove that the augmented matrix is as stated
theorem augmented_matrix :
  (eq1, eq2) = ((2, -1, 1), (1, 3, 2)) :=
by 
  sorry

end augmented_matrix_l599_599216


namespace find_k_l599_599172

-- Define point type and distances
structure Point :=
(x : ℝ)
(y : ℝ)

def dist (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Condition: H is the orthocenter of triangle ABC
variable (A B C H Q : Point)
variable (H_is_orthocenter : ∀ P : Point, dist P H = dist P A + dist P B - dist A B)

-- Prove the given equation
theorem find_k :
  dist Q A + dist Q B + dist Q C = 3 * dist Q H + dist H A + dist H B + dist H C :=
sorry

end find_k_l599_599172


namespace problem_solution_l599_599487

theorem problem_solution (a : ℚ) (h : 3 * a + 6 * a / 4 = 6) : a = 4 / 3 :=
by
  sorry

end problem_solution_l599_599487


namespace square_pyramid_planes_l599_599826

noncomputable def square_pyramid_edges : ℕ := 8
noncomputable def base_edges : ℕ := 4
noncomputable def apex_edges : ℕ := 4

theorem square_pyramid_planes :
  (number_of_unordered_pairs_of_edges_that_determine_a_plane square_pyramid_edges base_edges apex_edges) = 18 := sorry

end square_pyramid_planes_l599_599826


namespace carson_gets_clawed_39_times_l599_599723

-- Conditions
def number_of_wombats : ℕ := 9
def claws_per_wombat : ℕ := 4
def number_of_rheas : ℕ := 3
def claws_per_rhea : ℕ := 1

-- Theorem statement
theorem carson_gets_clawed_39_times :
  (number_of_wombats * claws_per_wombat + number_of_rheas * claws_per_rhea) = 39 :=
by
  sorry

end carson_gets_clawed_39_times_l599_599723


namespace graph_transformation_l599_599950

noncomputable def transform_graph (f : ℝ → ℝ) (stretch_factor : ℝ) (shift : ℝ) :=
  λ x, f ((x / stretch_factor) + shift)

theorem graph_transformation :
  let f := λ x, 3 * sin (2 * x - π / 6) in
  let g := λ x, 3 * sin (x + π / 2) in
  transform_graph f 2 (-2 * π / 3) = g :=
by
  sorry

end graph_transformation_l599_599950


namespace exists_point_K_l599_599285

open Real

variables {n : ℕ} {A : Fin n → ℝ^3} {P Q K : ℝ^3} {s : ℝ}

noncomputable def not_collinear (A : Fin n → ℝ^3) : Prop :=
  ∀ i j k : Fin n, i ≠ j → i ≠ k → j ≠ k → 
  (∃ a b c : ℝ, a + b + c = 1 ∧ a • A i + b • A j + c • A k = 0)

def distances_sum_equal (A : Fin n → ℝ^3) (P Q : ℝ^3) (s : ℝ) : Prop :=
  (sum (init n) (λ i : Fin n, norm (A i - P)) = s) ∧
  (sum (init n) (λ i : Fin n, norm (A i - Q)) = s)

theorem exists_point_K (A : Fin n → ℝ^3) (P Q : ℝ^3) (s : ℝ)
  (h1 : not_collinear A) 
  (h2 : distances_sum_equal A P Q s) : 
  ∃ K : ℝ^3, sum (init n) (λ i : Fin n, norm (A i - K)) < s :=
sorry

end exists_point_K_l599_599285


namespace backpack_price_equation_l599_599698

-- Define the original price of the backpack
variable (x : ℝ)

-- Define the conditions
def discount1 (x : ℝ) : ℝ := 0.8 * x
def discount2 (d : ℝ) : ℝ := d - 10
def final_price (p : ℝ) : Prop := p = 90

-- Final statement to be proved
theorem backpack_price_equation : final_price (discount2 (discount1 x)) ↔ 0.8 * x - 10 = 90 := sorry

end backpack_price_equation_l599_599698


namespace transitive_sets_in_V5_l599_599875

-- Base definition of the sets V_n
def V : ℕ → Set (Set (Set _))
| 0     => ∅
| (n+1) => {s | s ⊆ V n}

-- Definition of a transitive set
def isTransitive (n : ℕ) (x : Set (Set _)) : Prop :=
  ∀ u, u ∈ x → u ⊆ x

-- Problem statement
theorem transitive_sets_in_V5 :
  { s : Set (Set _) | s ∈ V 5 ∧ isTransitive 5 s }.toFinset.card = 4131 := 
sorry

end transitive_sets_in_V5_l599_599875


namespace part_one_part_two_l599_599086

noncomputable def f (x : ℝ) : ℝ := (3 * x) / (x + 1)

-- First part: Prove that f(x) is increasing on [2, 5]
theorem part_one (x₁ x₂ : ℝ) (hx₁ : 2 ≤ x₁) (hx₂ : x₂ ≤ 5) (h : x₁ < x₂) : f x₁ < f x₂ :=
by {
  -- Proof is to be filled in
  sorry
}

-- Second part: Find maximum and minimum of f(x) on [2, 5]
theorem part_two :
  f 2 = 2 ∧ f 5 = 5 / 2 :=
by {
  -- Proof is to be filled in
  sorry
}

end part_one_part_two_l599_599086


namespace find_number_l599_599313

theorem find_number (x : ℕ) (h : x + 8 = 500) : x = 492 :=
by sorry

end find_number_l599_599313


namespace convex_polygon_inscribed_in_circle_inequality_l599_599488

-- Define the necessary parameters and conditions
variables (n : ℕ) (R : ℝ) (a : Fin n → ℝ)

-- Noncomputable definition for general mathematical functions
noncomputable def polygon_inscribed_inequality (n : ℕ) (R : ℝ) (a : Fin n → ℝ) : Prop :=
  ∑ i : Fin n, (1 / a i) ≥ n / (2 * R * Real.sin (π / n))

-- Statement of the theorem
theorem convex_polygon_inscribed_in_circle_inequality 
  (h_pos_n : 0 < n) 
  (h_radius_pos : 0 < R) 
  (h_sides_pos : ∀ i : Fin n, 0 < a i)
  (h_convex_inscribed : ∃ (α : Fin n → ℝ), ∀ i : Fin n, a i = 2 * R * Real.sin (α i / 2)) : 
  polygon_inscribed_inequality n R a :=
begin
  sorry
end

end convex_polygon_inscribed_in_circle_inequality_l599_599488


namespace problem_trajectory_l599_599141

noncomputable def trajectory_eq (m n : ℝ) (h : m * n = 3) : Prop :=
  ∀ x y, 
    (y^2 = - (m * n) / 4 * (x^2 - 4)) ↔ 
    (x^2 / 4 + y^2 / 3 = 1)

noncomputable def fixed_circle_eq (P : ℝ × ℝ) (G G' : ℝ × ℝ)
                                  (h : G = (1, 0) ∧ G' = (-1, 0)) 
                                  (r : ℝ) : Prop :=
  ∃ x y, (x + 1)^2 + y^2 = 16

theorem problem_trajectory 
  (m n : ℝ) (h : m * n = 3) :
  trajectory_eq m n h ∧ 
  fixed_circle_eq (1, 0) G = (1, 0) ∧ G' = (-1, 0) 4 :=
sorry

end problem_trajectory_l599_599141


namespace bamboo_sections_volume_l599_599984

theorem bamboo_sections_volume (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a n = a 0 + n * d) →
  (a 0 + a 1 + a 2 = 4) →
  (a 5 + a 6 + a 7 + a 8 = 3) →
  (a 3 + a 4 = 2 + 3 / 22) :=
sorry

end bamboo_sections_volume_l599_599984


namespace abs_eq_5_iff_l599_599104

theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end abs_eq_5_iff_l599_599104


namespace systematic_sampling_id_fourth_student_l599_599299

theorem systematic_sampling_id_fourth_student (n : ℕ) (a b c d : ℕ) (h1 : n = 54) 
(h2 : a = 3) (h3 : b = 29) (h4 : c = 42) (h5 : d = a + 13) : d = 16 :=
by
  sorry

end systematic_sampling_id_fourth_student_l599_599299


namespace percent_answered_both_correctly_l599_599486

variables (students : ℕ)
variables (answered_first_correctly : ℕ) (answered_second_correctly : ℕ) (answered_neither_correctly : ℕ)
variables (answered_both_correctly : ℕ)

def total_students := 100
def answered_first_correctly := 75
def answered_second_correctly := 25
def answered_neither_correctly := 20

theorem percent_answered_both_correctly :
  answered_both_correctly = answered_first_correctly + answered_second_correctly - (total_students - answered_neither_correctly) :=
sorry

end percent_answered_both_correctly_l599_599486


namespace sum_common_divisors_75_45_l599_599395

theorem sum_common_divisors_75_45 : 
    (finset.sum (finset.filter (λ d, 75 % d = 0 ∧ 45 % d = 0) (finset.range 76))) = 24 :=
by sorry

end sum_common_divisors_75_45_l599_599395


namespace number_is_square_plus_opposite_l599_599312

theorem number_is_square_plus_opposite (x : ℝ) (hx : x = x^2 + -x) : x = 0 ∨ x = 2 :=
by sorry

end number_is_square_plus_opposite_l599_599312


namespace probability_two_balls_different_colors_l599_599491

theorem probability_two_balls_different_colors :
  let P := (2 * 3 + 3 * 2) / (5 * 5) in
  P = 12 / 25 :=
by
  let white_balls := 2
  let black_balls := 3
  let total_balls := white_balls + black_balls
  let P := (white_balls * black_balls + black_balls * white_balls) / (total_balls * total_balls)
  show P = 12 / 25
  sorry

end probability_two_balls_different_colors_l599_599491


namespace Tina_pail_capacity_l599_599633

theorem Tina_pail_capacity 
  (T : ℕ)
  (h1 : ∃ T, let Tommy_pail := T + 2 in 
             let Timmy_pail := 2 * Tommy_pail in 
             3 * T + 3 * Tommy_pail + 3 * Timmy_pail = 66) : 
  T = 4 :=
by {
  sorry
}

end Tina_pail_capacity_l599_599633


namespace fraction_power_multiplication_l599_599718

theorem fraction_power_multiplication :
  ( (5 / 8: ℚ) ^ 2 * (3 / 4) ^ 2 * (2 / 3) = 75 / 512) := 
  by
  sorry

end fraction_power_multiplication_l599_599718


namespace find_k_l599_599931

noncomputable def intersect_circle (k : ℝ) : Prop :=
  let center := (3 : ℝ, 2 : ℝ)
  let radius := 2 : ℝ
  let line (p : ℝ × ℝ) := p.2 = k * p.1 + 3
  let circle (p : ℝ × ℝ) := (p.1 - 3)^2 + (p.2 - 2)^2 = radius^2
  let distance (p₁ p₂ : ℝ × ℝ) := real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)
  ∃ (M N : ℝ × ℝ), line M ∧ line N ∧ circle M ∧ circle N ∧ distance M N = 2 * real.sqrt 3

theorem find_k (k : ℝ) :
  (intersect_circle k) → (k = -¾ ∨ k = 0) :=
by
  sorry

end find_k_l599_599931


namespace cost_of_pet_snake_l599_599056

theorem cost_of_pet_snake (original_amount : ℕ) (amount_left : ℕ) (cost : ℕ) 
  (h1 : original_amount = 73) (h2 : amount_left = 18) : cost = 55 :=
by
  sorry

end cost_of_pet_snake_l599_599056


namespace trapezoid_area_l599_599736

noncomputable theory

def area_of_trapezoid (b d : ℝ) : ℝ :=
  1 / 2 * ((b + d) + (b - d)) * b

theorem trapezoid_area (b d : ℝ) : area_of_trapezoid b d = b^2 :=
by
  unfold area_of_trapezoid
  simp
  ring

end trapezoid_area_l599_599736


namespace andrei_monthly_spending_l599_599339

noncomputable def original_price := 50
noncomputable def price_increase := 0.10
noncomputable def discount := 0.10
noncomputable def kg_per_month := 2

def new_price := original_price + original_price * price_increase
def discounted_price := new_price - new_price * discount
def monthly_spending := discounted_price * kg_per_month

theorem andrei_monthly_spending : monthly_spending = 99 := by
  sorry

end andrei_monthly_spending_l599_599339


namespace largest_factor_of_9975_l599_599264

theorem largest_factor_of_9975 (h : 9975 = 3 * 5 * 5 * 199) : 
  ∃ n, n ≤ 10000 ∧ n ∣ 9975 ∧ (∀ m, m ≤ 10000 → m ∣ 9975 → m ≤ n) ∧ n = 4975 :=
begin
  sorry
end

end largest_factor_of_9975_l599_599264


namespace find_triples_solution_l599_599386

theorem find_triples_solution (x y z : ℕ) (h : x^5 + x^4 + 1 = 3^y * 7^z) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 2) :=
by
  sorry

end find_triples_solution_l599_599386


namespace tan_neq_sqrt3_implies_alpha_neq_pi_div_3_l599_599973

theorem tan_neq_sqrt3_implies_alpha_neq_pi_div_3 (α : ℝ) 
  (h : tan α ≠ sqrt 3) : α ≠ π / 3 :=
by
  -- Proof will be provided here
  sorry

end tan_neq_sqrt3_implies_alpha_neq_pi_div_3_l599_599973


namespace intersection_M_N_l599_599618

noncomputable theory

open Set

-- Define sets M and N based on the conditions
def M : Set ℝ := { x | Real.log x > 0 }
def N : Set ℝ := { x | x ^ 2 ≤ 4 }

-- Define the theorem to prove the given problem
theorem intersection_M_N : ∀ x : ℝ, (x ∈ M ∩ N) ↔ (1 < x ∧ x ≤ 2) := 
by
  sorry

end intersection_M_N_l599_599618


namespace students_on_playground_l599_599205

theorem students_on_playground (rows_left : ℕ) (rows_right : ℕ) (rows_front : ℕ) (rows_back : ℕ) (h1 : rows_left = 12) (h2 : rows_right = 11) (h3 : rows_front = 18) (h4 : rows_back = 8) :
    (rows_left + rows_right - 1) * (rows_front + rows_back - 1) = 550 := 
by
  sorry

end students_on_playground_l599_599205


namespace a_n_equals_yn_sq_plus_7_by_xn_minus_yn_l599_599458

-- Define the sequence a_n
def a : ℕ → ℤ
| 0       := 4
| 1       := 22
| (n + 2) := 6 * a (n + 1) - a n

-- Define the sequences x_n and y_n
def y : ℕ → ℤ
| 0       := 1
| 1       := 9
| (n + 2) := 6 * y (n + 1) - y n

def x : ℕ → ℤ
| n       := y n + (if n = 0 then 4 else a (n - 1))

theorem a_n_equals_yn_sq_plus_7_by_xn_minus_yn (n : ℕ) :
  a n = (y n ^ 2 + 7) / (x n - y n) := by
  sorry

end a_n_equals_yn_sq_plus_7_by_xn_minus_yn_l599_599458


namespace rahul_share_l599_599657

-- Conditions
def work_rate_rahul : ℚ := 1 / 3
def work_rate_rajesh : ℚ := 1 / 2
def total_payment : ℚ := 150

-- Question
theorem rahul_share :
  let combined_work_rate := work_rate_rahul + work_rate_rajesh in
  let rahul_share_work := work_rate_rahul / combined_work_rate in
  let rahul_share_payment := rahul_share_work * total_payment in
  rahul_share_payment = 60 :=
by
  sorry

end rahul_share_l599_599657


namespace units_digit_of_3_pow_7_pow_6_l599_599397

theorem units_digit_of_3_pow_7_pow_6 :
  (3 ^ (7 ^ 6) % 10) = 3 := 
sorry

end units_digit_of_3_pow_7_pow_6_l599_599397


namespace cannot_tile_with_sphinxes_l599_599260

def triangle_side_length : ℕ := 6
def small_triangles_count : ℕ := 36
def upward_triangles_count : ℕ := 21
def downward_triangles_count : ℕ := 15

theorem cannot_tile_with_sphinxes (n : ℕ) (small_triangles : ℕ) (upward : ℕ) (downward : ℕ) :
  n = triangle_side_length →
  small_triangles = small_triangles_count →
  upward = upward_triangles_count →
  downward = downward_triangles_count →
  (upward % 2 ≠ 0) ∨ (downward % 2 ≠ 0) →
  ¬ (upward + downward = small_triangles ∧
     ∀ k, (k * 6) ≤ small_triangles →
     ∃ u d, u + d = k * 6 ∧ u % 2 = 0 ∧ d % 2 = 0) := 
by
  intros
  sorry

end cannot_tile_with_sphinxes_l599_599260


namespace contractor_pays_male_worker_rs_35_l599_599291

theorem contractor_pays_male_worker_rs_35
  (num_male_workers : ℕ)
  (num_female_workers : ℕ)
  (num_child_workers : ℕ)
  (female_worker_wage : ℕ)
  (child_worker_wage : ℕ)
  (average_wage_per_day : ℕ)
  (total_workers : ℕ := num_male_workers + num_female_workers + num_child_workers)
  (total_wage : ℕ := average_wage_per_day * total_workers)
  (total_female_wage : ℕ := num_female_workers * female_worker_wage)
  (total_child_wage : ℕ := num_child_workers * child_worker_wage)
  (total_male_wage : ℕ := total_wage - total_female_wage - total_child_wage) :
  num_male_workers = 20 →
  num_female_workers = 15 →
  num_child_workers = 5 →
  female_worker_wage = 20 →
  child_worker_wage = 8 →
  average_wage_per_day = 26 →
  total_male_wage / num_male_workers = 35 :=
by
  intros h20 h15 h5 h20w h8w h26
  sorry

end contractor_pays_male_worker_rs_35_l599_599291


namespace cubic_inequality_l599_599415

theorem cubic_inequality (a b : ℝ) : (a > b) ↔ (a^3 > b^3) := sorry

end cubic_inequality_l599_599415


namespace same_function_l599_599272

noncomputable def f : ℝ → ℝ := λ x, (x^3)^(1/3)
noncomputable def g : ℝ → ℝ := λ x, x

theorem same_function : (∀ x : ℝ, f x = g x) := 
by
  intro x
  dsimp [f, g]
  sorry

end same_function_l599_599272


namespace number_of_pos_values_sum_lt_1000_l599_599877

theorem number_of_pos_values_sum_lt_1000 :
  let seq : Fin 6 → ℤ :=
    λ i, (if i = 0 then a₁ else 
          if i = 1 then a₁ / 3 ∨ -2 * a₁ else
          if i = 2 then (a₁ / 3) / 3 ∨ -2 * (a₁ / 3) ∨ -(2 * -2 * a₁) / 3 ∨ -(2 * -2 * a₁) * -2 else
          -- Extend this similar pattern for other elements in the sequence...
          sorry) in 
    let s := seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5 in
    ∃ a₁ : ℤ, s ≥ 0 ∧ s < 1000 ∧
    (seq 0 = a₁ ∧ (seq 1 = a₁ / 3 ∨ seq 1 = -2 * a₁) ∧
     (seq 2 ∈ {a₁ / 3 / 3, -2 * (a₁ / 3), -(2 * -2 * a₁) / 3, -(2 * --2 * a₁) * -2}) ∧
     -- Extend this similar pattern for other elements in the sequence...
     sorry) :=
  let number_of_solutions := 142 in
  sorry

end number_of_pos_values_sum_lt_1000_l599_599877


namespace probability_of_interval_is_one_third_l599_599586

noncomputable def probability_in_interval (total_start total_end inner_start inner_end : ℝ) : ℝ :=
  (inner_end - inner_start) / (total_end - total_start)

theorem probability_of_interval_is_one_third :
  probability_in_interval 1 7 5 8 = 1 / 3 :=
by
  sorry

end probability_of_interval_is_one_third_l599_599586


namespace florist_sold_roses_l599_599683

def selling_rose_problem (initial_roses picked_roses total_roses_left : ℕ) (sold_roses : ℕ) : Prop :=
  (initial_roses - sold_roses) + picked_roses = total_roses_left

theorem florist_sold_roses :
  ∃ x : ℕ, selling_rose_problem 37 19 40 x ∧ x = 16 :=
by
  use 16
  unfold selling_rose_problem
  simp
  sorry

end florist_sold_roses_l599_599683


namespace triangle_sum_problem_l599_599480

-- Define the given angles and prove the total of angle B and D.
theorem triangle_sum_problem (A B C D E : Type) [HasAngle A] [HasAngle B] [HasAngle C] [HasAngle D] [HasAngle E] 
(H1 : angle A = 40) (H2 : angle B + angle D = 160) : Prop := 
by 
  -- Conditions provided and goal to be proved
  have h1 : angle A = 40 := H1,
  have h2 : angle B + angle D = 160 := H2,
  sorry

end triangle_sum_problem_l599_599480


namespace equilibrium_constant_reaction1_at_given_conditions_l599_599202

variable (NH4I NH3 HI H2 I2 : Type)
variable (c : NH3 → ℝ)
variable (c_H2 : ℝ)
variable (c_HI : ℝ)
variable (equilibrium_constant : ℝ)
variable (reaction1 reaction2 : Prop)

axiom c_H2_eq : c_H2 = 1
axiom c_HI_eq : c_HI = 4

def reaction1_at_equilibrium : Prop :=
  c NH3 * c HI = equilibrium_constant

def reaction2_at_equilibrium : Prop :=
  2 * c HI = c H2 + c I2

theorem equilibrium_constant_reaction1_at_given_conditions :
  reaction1_at_equilibrium ∧ reaction2_at_equilibrium →
  equilibrium_constant = 24 :=
by sorry

end equilibrium_constant_reaction1_at_given_conditions_l599_599202


namespace price_increase_2009_l599_599615

-- Define given conditions
def price_in_2006 : ℝ := 1
def price_increase_2008 : ℝ := 0.60
def annual_growth_rate : ℝ := 0.20

-- Calculate intermediate values based on conditions
def price_in_2008 : ℝ := price_in_2006 * (1 + price_increase_2008)
def price_in_2009 : ℝ := price_in_2006 * (1 + annual_growth_rate)^3

-- Statement of the problem to be proven
theorem price_increase_2009 :
  let x := (price_in_2009 / price_in_2008) - 1 in
  x * 100 = 8 := by
  sorry

end price_increase_2009_l599_599615


namespace correct_operation_l599_599969

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l599_599969


namespace min_value_inequality_l599_599884

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 :=
sorry

end min_value_inequality_l599_599884


namespace trajectory_of_center_M_l599_599927

noncomputable def trajectory_center (x y : ℝ) : Prop :=
  (x^2 / 5 + y^2 / 9 = 1) ∧ y ≠ 3

theorem trajectory_of_center_M :
  (∀ x y r : ℝ, (5 - r = real.sqrt (x^2 + (y + 1)^2)) ∧ (r + 1 = real.sqrt (x^2 + (y - 2)^2)) ∧
                 (x^2 + (y + 1)^2 = 25) ∧ (x^2 + (y - 2)^2 = 1) →
                 trajectory_center x y) :=
begin
  sorry
end

end trajectory_of_center_M_l599_599927


namespace tetrahedron_circumscribed_sphere_radius_l599_599516

noncomputable def radius_of_circumscribed_sphere 
  (a b c : ℝ) : ℝ :=
  √(a^2 + b^2 + c^2) / 2

theorem tetrahedron_circumscribed_sphere_radius
  {a b c : ℝ}
  (h1 : ∀ (S A B C : EuclideanSpace), SA ⊥ SB)
  (h2 : ∀ (S A B C : EuclideanSpace), SB ⊥ SC)
  (h3 : ∀ (S A B C : EuclideanSpace), SA ⊥ SC)
  (ha : SA = a)
  (hb : SB = b)
  (hc : SC = c)
  : 
  radius_of_circumscribed_sphere a b c = 
  √(a^2 + b^2 + c^2) / 2 :=
sorry

end tetrahedron_circumscribed_sphere_radius_l599_599516


namespace find_first_train_length_l599_599639

theorem find_first_train_length
  (length_second_train : ℝ)
  (initial_distance : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_minutes : ℝ) :
  length_second_train = 200 →
  initial_distance = 100 →
  speed_first_train_kmph = 54 →
  speed_second_train_kmph = 72 →
  time_minutes = 2.856914303998537 →
  ∃ (L : ℝ), L = 5699.52 :=
by
  sorry

end find_first_train_length_l599_599639


namespace root_in_interval_l599_599562

noncomputable def f (x : ℝ) : ℝ := x + Real.log x - 3

theorem root_in_interval : ∃ m, f m = 0 ∧ 2 < m ∧ m < 3 :=
by
  sorry

end root_in_interval_l599_599562


namespace max_primes_in_valid_80x80_table_l599_599851

-- Definitions as derived from the conditions in the problem
def is_prime_or_product_of_two_primes (n : ℕ) : Prop :=
  nat.prime n ∨ (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ n = p * q)

def not_coprime (a b : ℕ) : Prop := (nat.gcd a b ≠ 1)

def valid_table (table : ℕ × ℕ → ℕ) : Prop :=
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 80 ∧ 1 ≤ j ∧ j ≤ 80 → pairwise (≠) (λ k l, table k l)) ∧
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 80 ∧ 1 ≤ j ∧ j ≤ 80 → is_prime_or_product_of_two_primes (table (i, j))) ∧
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 80 ∧ 1 ≤ j ∧ j ≤ 80 →
    ∃ k, (1 ≤ k ∧ k ≤ 80 ∧ not_coprime (table (i, j)) (table (i, k))) ∨ 
    (1 ≤ k ∧ k ≤ 80 ∧ not_coprime (table (i, j)) (table (k, j))))

noncomputable def max_primes_in_valid_table (table : ℕ × ℕ → ℕ) : ℕ :=
  finset.card (finset.filter nat.prime (finset.univ.image (λ (i, j), table (i, j))))

theorem max_primes_in_valid_80x80_table : 
  ∃ (table : ℕ × ℕ → ℕ), valid_table table ∧ max_primes_in_valid_table table = 4266 :=
sorry

end max_primes_in_valid_80x80_table_l599_599851


namespace insphere_touches_centroids_of_regular_tetrahedron_l599_599067

noncomputable def incircle_touches_midpoints (T : Triangle) : Prop :=
∀ side : T.side, incircle T → midpoint side

noncomputable def insphere_touches_centroids (tetra : Tetrahedron) : Prop :=
∀ face : tetra.face, insphere tetra → centroid face

theorem insphere_touches_centroids_of_regular_tetrahedron (tetra : Tetrahedron)
  (h_equilateral_faces : ∀ face : tetra.face, equilateral face)
  (h_incircle_midpoints : ∀ triangle : Triangle,
    (equilateral triangle → incircle_touches_midpoints triangle)) :
  insphere_touches_centroids tetra :=
sorry

end insphere_touches_centroids_of_regular_tetrahedron_l599_599067


namespace total_interest_is_1710_l599_599527

def principal1 := 17000
def rate1 := 0.08
def principal2 := 22000 - 17000
def rate2 := 0.07

def interest1 := principal1 * rate1
def interest2 := principal2 * rate2
def totalInterest := interest1 + interest2

theorem total_interest_is_1710 : totalInterest = 1710 := by
  sorry

end total_interest_is_1710_l599_599527


namespace speed_of_second_car_is_correct_l599_599636

-- Definitions of given conditions
def distance_apart : ℝ := 60
def speed_first_car : ℝ := 90
def time_taken : ℝ := 3
def distance_first_car := speed_first_car * time_taken

-- Definition of the speed of second car
def speed_second_car : ℝ :=
(abs(distance_first_car - distance_apart)) / time_taken

-- Theorem statement to be proven
theorem speed_of_second_car_is_correct :
  speed_second_car = 70 := by
  sorry

end speed_of_second_car_is_correct_l599_599636


namespace john_weekly_earnings_increase_l599_599656

theorem john_weekly_earnings_increase (original_earnings new_earnings : ℕ) 
  (h₀ : original_earnings = 60) 
  (h₁ : new_earnings = 72) : 
  ((new_earnings - original_earnings) / original_earnings) * 100 = 20 :=
by
  sorry

end john_weekly_earnings_increase_l599_599656


namespace shaded_area_triangle_l599_599507

theorem shaded_area_triangle (a b : ℝ) (h1 : a = 5) (h2 : b = 15) :
  let area_shaded : ℝ := (5^2) - (1/2 * ((15 / 4) * 5))
  area_shaded = 175 / 8 := 
by
  sorry

end shaded_area_triangle_l599_599507


namespace parametric_to_general_polar_to_cartesian_chord_length_AB_l599_599140

theorem parametric_to_general (t : ℝ) :
  (∃ t : ℝ, (2 - 3 * t, -1 + 3 / 2 * t) = (x, y)) ↔ (x + 2 * y = 0) :=
sorry

theorem polar_to_cartesian (θ : ℝ) (ρ : ℝ) :
  (ρ = 2 * cos (θ - π / 4)) ↔ (x - sqrt 2 / 2) ^ 2 + (y - sqrt 2 / 2) ^ 2 = 1 :=
sorry

theorem chord_length_AB :
  let C := (sqrt 2 / 2, sqrt 2 / 2)
  let l := (x + 2 * y = 0)
  let d := abs ((sqrt 2 / 2 + 2 * (sqrt 2 / 2)) / sqrt 5)
  ∃ A B : ℝ × ℝ, (A ≠ B) ∧
  (∃ t₁ t₂ : ℝ, (A = (2 - 3 * t₁, -1 + 3 / 2 * t₁)) ∧ (B = (2 - 3 * t₂, -1 + 3 / 2 * t₂))) ∧
  |dist A B| = 2 * sqrt (1 - (d ^ 2)) := sorry

end parametric_to_general_polar_to_cartesian_chord_length_AB_l599_599140


namespace namjoonKoreanScore_l599_599182

variables (mathScore englishScore : ℝ) (averageScore : ℝ := 95) (koreanScore : ℝ)

def namjoonMathScore : Prop := mathScore = 100
def namjoonEnglishScore : Prop := englishScore = 95
def namjoonAverage : Prop := (koreanScore + mathScore + englishScore) / 3 = averageScore

theorem namjoonKoreanScore
  (H1 : namjoonMathScore 100)
  (H2 : namjoonEnglishScore 95)
  (H3 : namjoonAverage koreanScore 100 95 95) :
  koreanScore = 90 :=
by
  sorry

end namjoonKoreanScore_l599_599182


namespace remainder_is_162_l599_599390

def polynomial (x : ℝ) : ℝ := 2 * x^4 - x^3 + 4 * x^2 - 5 * x + 6

theorem remainder_is_162 : polynomial 3 = 162 :=
by 
  sorry

end remainder_is_162_l599_599390


namespace boat_distance_downstream_l599_599996

theorem boat_distance_downstream 
    (boat_speed_still : ℝ) 
    (stream_speed : ℝ) 
    (time_downstream : ℝ) 
    (distance_downstream : ℝ) 
    (h_boat_speed_still : boat_speed_still = 13) 
    (h_stream_speed : stream_speed = 6) 
    (h_time_downstream : time_downstream = 3.6315789473684212) 
    (h_distance_downstream : distance_downstream = 19 * 3.6315789473684212): 
    distance_downstream = 69 := 
by 
  have h_effective_speed : boat_speed_still + stream_speed = 19 := by 
    rw [h_boat_speed_still, h_stream_speed]; norm_num 
  rw [h_distance_downstream]; norm_num 
  sorry

end boat_distance_downstream_l599_599996


namespace distance_between_circle_center_and_point_l599_599263

theorem distance_between_circle_center_and_point (x y : ℝ) (h : x^2 + y^2 = 8*x - 12*y + 40) : 
  dist (4, -6) (4, -2) = 4 := 
by
  sorry

end distance_between_circle_center_and_point_l599_599263


namespace middle_number_is_correct_l599_599217

theorem middle_number_is_correct (numbers : List ℝ) (h_length : numbers.length = 11)
  (h_avg11 : numbers.sum / 11 = 9.9)
  (first_6 : List ℝ) (h_first6_length : first_6.length = 6)
  (h_avg6_1 : first_6.sum / 6 = 10.5)
  (last_6 : List ℝ) (h_last6_length : last_6.length = 6)
  (h_avg6_2 : last_6.sum / 6 = 11.4) :
  (∃ m : ℝ, m ∈ first_6 ∧ m ∈ last_6 ∧ m = 22.5) :=
by
  sorry

end middle_number_is_correct_l599_599217


namespace sum_ages_divya_nacho_l599_599841

theorem sum_ages_divya_nacho
  (divya_current_age : ℕ)
  (h1 : divya_current_age = 5)
  (h2 : ∀ n, Nacho_in_5_years n = 3 * (divya_current_age + 5)) :
  (divya_current_age + Nacho_current_age) = 40 :=
by
  let divya_age_5_years := divya_current_age + 5
  have nacho_age_5_years := 3 * divya_age_5_years
  let nacho_current_age := nacho_age_5_years - 5
  exact sorry

end sum_ages_divya_nacho_l599_599841


namespace ghost_enter_exit_ways_l599_599308

theorem ghost_enter_exit_ways : 
  (∃ (enter_win : ℕ) (exit_win : ℕ), enter_win ≠ exit_win ∧ 1 ≤ enter_win ∧ enter_win ≤ 8 ∧ 1 ≤ exit_win ∧ exit_win ≤ 8) →
  ∃ (ways : ℕ), ways = 8 * 7 :=
by
  sorry

end ghost_enter_exit_ways_l599_599308


namespace number_of_integers_satisfying_inequality_l599_599389

theorem number_of_integers_satisfying_inequality :
  {m : ℤ | (m - 1) * (m - 10) + 2 ≤ 0}.finite.to_finset.card = 8 := 
sorry

end number_of_integers_satisfying_inequality_l599_599389


namespace possible_cos_values_of_angle_between_a_and_c_l599_599078

variable {a b : ℝ} -- implicitly the vector magnitudes
variable {x y : ℝ} (hx : x ∈ Set.Icc 1 2) (hy : y ∈ Set.Icc 1 2)
variable (h_cos : real.angle.cos (real.angle.pi_div_three) = 1/2)
variable (h_ab : abs b = 2 * abs a) (h_a : abs a = 1) (h_b : abs b = 2)
variable (c := x * a + y * b)

theorem possible_cos_values_of_angle_between_a_and_c :
  (real.angle.cos (real.angle.between a c) = real.sqrt 21 / 7) ∨
  (real.angle.cos (real.angle.between a c) = real.sqrt 3 / 3) := by 
  sorry

end possible_cos_values_of_angle_between_a_and_c_l599_599078


namespace find_larger_number_l599_599655

theorem find_larger_number (L S : ℕ)
  (h1 : L - S = 1370)
  (h2 : L = 6 * S + 15) :
  L = 1641 := sorry

end find_larger_number_l599_599655


namespace sum_of_q_p_values_l599_599608

def p (x : ℝ) : ℝ := 2 * |x| - 4
def q (x : ℝ) : ℝ := - |x|

theorem sum_of_q_p_values : ([-4, -3, -2, -1, 0, 1, 2, 3, 4].sum_by (λ x, q (p x)) = -20) :=
by
  sorry

end sum_of_q_p_values_l599_599608


namespace scenario1_ways_scenario2_ways_scenario3_ways_scenario4_ways_l599_599760

-- Scenario 1: All five balls into four distinct boxes
theorem scenario1_ways : 4^5 = 1024 := by sorry

-- Scenario 2: Each of the four distinct boxes receives one ball
theorem scenario2_ways : ∀ (n : ℕ), (nat.perm 5 4) = 120 := by sorry

-- Scenario 3: Four out of the five balls are placed into one of the four boxes (the other ball is not placed)
theorem scenario3_ways : ∀ (n : ℕ), (nat.choose 5 4) * (nat.choose 4 1) = 20 := by sorry 

-- Scenario 4: All five balls into four distinct boxes with no box left empty
theorem scenario4_ways : ∀ (n : ℕ), (nat.choose 5 2) * (nat.perm 4 4) = 240 := by sorry 

end scenario1_ways_scenario2_ways_scenario3_ways_scenario4_ways_l599_599760


namespace cinnamon_balls_required_l599_599866

theorem cinnamon_balls_required 
  (num_family_members : ℕ) 
  (cinnamon_balls_per_day : ℕ) 
  (num_days : ℕ) 
  (h_family : num_family_members = 5) 
  (h_balls_per_day : cinnamon_balls_per_day = 5) 
  (h_days : num_days = 10) : 
  num_family_members * cinnamon_balls_per_day * num_days = 50 := by
  sorry

end cinnamon_balls_required_l599_599866


namespace total_books_l599_599835

theorem total_books (x y : ℝ) :
  let betty_books := x,
      sister_books := (5/4) * x,
      cousin_books := (5/2) * x,
      friend_books := (19/4) * x - y,
      total_books := betty_books + sister_books + cousin_books + friend_books
  in total_books = (19/2) * x - y := by sorry

end total_books_l599_599835


namespace inequality_system_has_three_integer_solutions_l599_599405

theorem inequality_system_has_three_integer_solutions (m : ℝ) :
  (∃ (s : finset ℤ), s.card = 3 ∧ ∀ x ∈ s, x + 5 > 0 ∧ x - m ≤ 1) ↔ -3 ≤ m ∧ m < -2 :=
by
  sorry

end inequality_system_has_three_integer_solutions_l599_599405


namespace chess_tournament_games_l599_599626

theorem chess_tournament_games :
  ∀ (n : ℕ), n = 5 → (n * (n - 1) / 2 = 20) → (n - 1 = 4) :=
by
  intros n hn h_games
  rw hn at h_games
  simp at h_games
  exact eq.trans (eq.symm (mul_div_cancel_left 20 (dec_trivial : 2 ≠ 0))) h_games
  sorry

end chess_tournament_games_l599_599626


namespace fraction_to_terminating_decimal_l599_599750

theorem fraction_to_terminating_decimal : (49 : ℚ) / 160 = 0.30625 := 
sorry

end fraction_to_terminating_decimal_l599_599750


namespace transformations_map_figure_l599_599010

noncomputable def count_transformations : ℕ := sorry

theorem transformations_map_figure :
  count_transformations = 3 :=
sorry

end transformations_map_figure_l599_599010


namespace max_value_L_and_sum_l599_599883

theorem max_value_L_and_sum (x y z v w : ℝ) (h : x^2 + y^2 + z^2 + v^2 + w^2 = 2024) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hv : 0 < v) (hw : 0 < w) :
  let L := x * z + 3 * y * z + 3 * z * v + 3 * z * w in
  L ≤ 336 * Real.sqrt 7 ∧
  (L = 336 * Real.sqrt 7 → x = 8 ∧ y = 24 ∧ z = Real.sqrt 1012 ∧ v = 24 ∧ w = 24 ∧ (L + x + y + z + v + w = 80 + Real.sqrt 1012 + 336 * Real.sqrt 7)) :=
sorry

end max_value_L_and_sum_l599_599883


namespace find_initial_solution_l599_599293

-- Definitions based on the conditions
variable (x : ℝ) -- initial volume in litres
variable (h1 : 0.25 * x) -- 25% of initial volume is alcohol
variable (h2 : 0.25 * x + 3 = 0.50 * (x + 3)) -- adding 3 litres results in 50% alcohol solution

-- The statement to prove
theorem find_initial_solution (h1 : 0.25 * x) (h2 : 0.25 * x + 3 = 0.50 * (x + 3)) : x = 6 :=
sorry

end find_initial_solution_l599_599293


namespace range_of_expression_l599_599464

theorem range_of_expression (a b : ℝ) (h : a^2 + b^2 = 9) :
  -21 ≤ a^2 + b^2 - 6*a - 8*b ∧ a^2 + b^2 - 6*a - 8*b ≤ 39 :=
by
  sorry

end range_of_expression_l599_599464


namespace minimize_cost_l599_599671

theorem minimize_cost
  (k1 k2 : ℝ)
  (x y : ℝ)
  (h1 : k1 / 5 = 1)
  (h2 : k2 * 5 = 4) :
  2 * 5 / 4 = 2.5 :=
by 
  have h_k1 : k1 = 5 := by linarith
  have h_k2 : k2 = 4 / 5 := by linarith
  have h_opt : (5 / (2 * 5 / 4)) + ((4 / 5) * (2 * 5 / 4)) = 2.5 := by linarith
  show 2 * 5 / 4 = 2.5
  sorry

end minimize_cost_l599_599671


namespace interval_of_monotonic_increase_minimum_value_in_interval_l599_599471

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (1, -real.sqrt 3 * real.sin (x / 2))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (real.sin x, 2 * real.sin (x / 2))

noncomputable def f (x : ℝ) : ℝ :=
  (a x).1 * (b x).1 + (a x).2 * (b x).2 + real.sqrt 3

theorem interval_of_monotonic_increase : ∀ k : ℤ,
  monotonic_increase_interval (λ x, f x) (2 * real.pi * k - 5 * real.pi / 6) (2 * real.pi * k + real.pi / 6) :=
sorry

theorem minimum_value_in_interval : ∃ x ∈ set.Icc 0 (2 * real.pi / 3), f x = 0 :=
sorry

end interval_of_monotonic_increase_minimum_value_in_interval_l599_599471


namespace exam_questions_count_l599_599211

theorem exam_questions_count (Q S : ℕ) 
    (hS : S = (4 * Q) / 5)
    (sergio_correct : Q - 4 = S + 6) : 
    Q = 50 :=
by 
  sorry

end exam_questions_count_l599_599211


namespace exponent_of_three_in_24_factorial_l599_599509

/-- Define what a factorial is -/
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

/-- Define the function that counts the number of occurrences of a prime factor p in n! -/
def prime_factor_exponent_in_factorial (n p : ℕ) : ℕ :=
  if n = 0 then 0
  else if p = 0 then 0
  else if p = 1 then 0
  else if ∃ (k : ℕ), p = 2 ^ k then 0
  else
    let rec aux k acc :=
      if k = 0 then acc
      else aux (k / p) (acc + k / p)
    in aux n 0

/-- The main theorem to prove -/
theorem exponent_of_three_in_24_factorial : prime_factor_exponent_in_factorial 24 3 = 10 :=
by
  sorry

end exponent_of_three_in_24_factorial_l599_599509


namespace find_b_f_above_g_for_x_positive_l599_599085

noncomputable def f (x : ℝ) (b : ℝ) (a : ℝ) := - x^2 / Real.exp x + (b - 1) * x + a
noncomputable def g (x : ℝ) := (x + 2) / (2 * Real.exp x)

-- Part (1) statement: Find the value of b when the tangent line at x = 0 is perpendicular to the y-axis.
theorem find_b (a b : ℝ) : 
  (∃ m, deriv (f x b a) 0 = m) → b = 1 := 
sorry

-- Part (2) statement: Prove that the graph of f(x) is always above g(x) for x ∈ (0, +∞) when a = 1
theorem f_above_g_for_x_positive (x : ℝ) (h : 0 < x) : 
  f x 1 1 > g x :=
sorry

end find_b_f_above_g_for_x_positive_l599_599085


namespace number_of_students_l599_599138

theorem number_of_students (h1: ∃ M, M = 30 ∧ ∀ n, n < M → n is "better")
                           (h2: ∃ M, M = 40 ∧ ∀ n, M < n → n is "worse") : 
  ∃ N, N = 69 :=
sorry

end number_of_students_l599_599138


namespace equation_of_ellipse_length_of_AB_l599_599081

-- Definitions based on the conditions
noncomputable def major_axis_length := 2 * a
def foci_F1 := (-sqrt 3, 0)
def foci_F2 := (sqrt 3, 0)
noncomputable def distance_F1_to_line := sqrt 3 / 3
noncomputable def distance_from_line_to_F1 := abs (-sqrt 3 - - (a^2 / sqrt 3))
def angle_line_through_F2 := 45 -- in degrees

-- Proof that equation derived is correct for the ellipse
theorem equation_of_ellipse (a b : ℝ) (H1 : distance_from_line_to_F1 = sqrt 3 / 3) :
  let c := sqrt 3 in
  let a_squared := 4 in
  let b_squared := a_squared - c^2 in
  (a^2 = 4) ∧ (b^2 = 1) ∧ (∀ x y : ℝ, (x^2 / 4 + y^2 = 1) ↔ ((x, y) ∈ {p : ℝ × ℝ | (a, b) ∈ {q : ℝ × ℝ | q.1^2 / 4 + q.2^2 = 1}})) := by
  sorry

-- Proof that the length of segment AB is correct
theorem length_of_AB (a b : ℝ) (H1 : distance_from_line_to_F1 = sqrt 3 / 3) :
  let F2 := (sqrt 3, 0) in
  let line_l (x : ℝ) := x - sqrt 3 in
  let points_of_intersection := {p : ℝ × ℝ | line_l(p.1) = p.2 ∧ p.1^2 / 4 + p.2^2 = 1} in
  ∃ A B : ℝ × ℝ, A ∈ points_of_intersection ∧ B ∈ points_of_intersection ∧ A ≠ B ∧ 
  (sqrt (1 + 1) * sqrt ((8 * sqrt 3 / 5)^2 - (4 * 2 / 5)) = 8 / 5) := by
  sorry

end equation_of_ellipse_length_of_AB_l599_599081


namespace fraction_simplification_l599_599967

theorem fraction_simplification (a b : ℝ) : 9 * b / (6 * a + 3) = 3 * b / (2 * a + 1) :=
by sorry

end fraction_simplification_l599_599967


namespace num_rectangles_in_3x3_grid_l599_599478

-- Defining the grid of points
def grid_points : List (ℕ × ℕ) := [
  (0, 0), (1, 0), (2, 0), (3, 0),
  (0, 1), (1, 1), (2, 1), (3, 1),
  (0, 2), (1, 2), (2, 2), (3, 2),
  (0, 3), (1, 3), (2, 3), (3, 3)
]

-- Theorem statement
theorem num_rectangles_in_3x3_grid : (binomial 4 2) * (binomial 4 2) = 36 := by
  sorry

end num_rectangles_in_3x3_grid_l599_599478


namespace line_through_incenter_of_triangle_l599_599833

theorem line_through_incenter_of_triangle
  (A B C P : Type)
  (AB AC : A → B)
  [noncomputable Π (x y : Type), ∥x - y∥] -- norm function for vectors implying metric space
  (λ : ℝ) (h_λ_ne_zero : λ ≠ 0)
  (h_AP : P = λ • (AB / ∥AB∥ + AC / ∥AC∥)) :
  ∃ (I : Type), (I is the incenter of triangle ABC) ∧ (A, I, P are collinear) := 
sorry

end line_through_incenter_of_triangle_l599_599833


namespace am_gm_inequality_l599_599193

theorem am_gm_inequality (n : ℕ) (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  Real.root (n+1) (x / y) ≤ (x + n * y) / ((n + 1) * y) :=
sorry

end am_gm_inequality_l599_599193


namespace smallest_positive_w_l599_599921

theorem smallest_positive_w
  (y w : ℝ)
  (h_sin : sin y = 0)
  (h_cos : cos (y + w) = -1 / 2) :
  w = 2 * Real.pi / 3 :=
sorry

end smallest_positive_w_l599_599921


namespace joelle_initial_deposit_l599_599524

-- Definitions for the conditions
def annualInterestRate : ℝ := 0.05
def initialTimePeriod : ℕ := 2 -- in years
def numberOfCompoundsPerYear : ℕ := 1
def finalAmount : ℝ := 6615

-- Compound interest formula: A = P(1 + r/n)^(nt)
noncomputable def initialDeposit : ℝ :=
  finalAmount / ((1 + annualInterestRate / numberOfCompoundsPerYear)^(numberOfCompoundsPerYear * initialTimePeriod))

-- Theorem statement to prove the initial deposit
theorem joelle_initial_deposit : initialDeposit = 6000 := 
  sorry

end joelle_initial_deposit_l599_599524


namespace min_value_l599_599779

theorem min_value (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_ab : a * b = 1) (h_a_2b : a = 2 * b) :
  a + 2 * b = 2 * Real.sqrt 2 := by
  sorry

end min_value_l599_599779


namespace new_average_after_increasing_one_number_by_8_l599_599200

theorem new_average_after_increasing_one_number_by_8 
  (S : Finset ℝ) (hS_card : S.card = 10) (hS_avg : (S.sum / 10) = 6.2) :
  ((S.sum + 8) / 10) = 7 :=
by 
  sorry

end new_average_after_increasing_one_number_by_8_l599_599200


namespace congruence_theorem_l599_599704

-- Definitions of the propositions
def PropA (Δ1 Δ2 : Triangle) : Prop :=
  Δ1.angles = Δ2.angles

def PropB (Δ1 Δ2 : Triangle) : Prop :=
  ∃ (a b : ℝ), Δ1.side1 = a ∧ Δ1.side2 = b ∧ Δ2.side1 = a ∧ Δ2.side2 = b

def PropC (Δ1 Δ2 : Triangle) : Prop :=
  ∃ (a b : ℝ), Δ1.angle1 = Δ2.angle1 ∧ Δ1.side1 = a ∧ Δ1.side2 = b ∧ Δ2.side1 = a ∧ Δ2.side2 = b

def PropD (Δ1 Δ2 : Triangle) : Prop :=
  Δ1.side1 = Δ2.side1 ∧ Δ1.is_isosceles ∧ Δ2.is_isosceles

-- The theorem stating that only Proposition D is correct
theorem congruence_theorem (Δ1 Δ2 : Triangle) :
  PropA Δ1 Δ2 = False ∧
  PropB Δ1 Δ2 = False ∧
  PropC Δ1 Δ2 = False ∧
  PropD Δ1 Δ2 = True :=
sorry

end congruence_theorem_l599_599704


namespace base5_divisible_by_29_l599_599738

theorem base5_divisible_by_29 :
  ∃ (y : ℕ), 0 ≤ y ∧ y < 5 ∧ (let n := 4 * 5^3 + 2 * 5^2 + y * 5 + 3 in n % 29 = 0) :=
begin
  use 4,
  split,
  { exact nat.zero_le 4 },
  split,
  { exact dec_trivial }, -- 4 < 5 is trivially true
  { 
    have n_calculation:  4 * 5^3 + 2 * 5^2 + 4 * 5 + 3 = 553 + 4 * 5 := rfl,
    -- Showing that n is equivalent to 573, which modulo 29 is 0
    rw n_calculation,
    norm_num,
  }
end

end base5_divisible_by_29_l599_599738


namespace length_PF_l599_599632

-- Define the parabola function
def parabola (x : ℝ) : ℝ := sqrt (8 * (x + 2))

-- Define the line with inclination angle 60 degrees passing through the focus
noncomputable def line_through_focus (x : ℝ) : ℝ := (real.sqrt 3) * x

-- Define the points of intersection A and B
noncomputable def intersection_points : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.snd = parabola p.fst ∧ p.snd = line_through_focus p.fst }

-- Define the midpoint of segment AB
def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.fst + b.fst) / 2, (a.snd + b.snd) / 2)

-- Define the perpendicular bisector of AB
def perpendicular_bisector (mid : ℝ × ℝ) : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | p.fst = mid.fst }

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.fst - p2.fst)^2 + (p1.snd - p2.snd)^2)

-- Establish that PF is equal to 16/3
theorem length_PF (A B : ℝ × ℝ)
  (hA : (A ∈ intersection_points)) (hB : (B ∈ intersection_points)) :
  let P := (midpoint A B).fst in
  distance (P, 0) (0, 0) = 16 / 3 :=
  sorry

end length_PF_l599_599632


namespace problem1_problem2_problem3_problem4_problem5_l599_599003

theorem problem1 : (-2) + (-3) - (-5) + 7 = 7 := 
by sorry

theorem problem2 : (-3 / 4) * (-1 / 2) / (-1 / 8) * 3 = -9 := 
by sorry

theorem problem3 : (-1)^6 * abs (-3 / 2) - 0.5 / (-1 / 3) = 3 := 
by sorry

theorem problem4 : (-3 / 4 + 5 / 6 - 7 / 8) * (-24) = 19 := 
by sorry

theorem problem5 : -1^4 - (1 - 0.5) * (1 / 3) * (5 - (-3)^2) = -1 / 3 := 
by sorry

end problem1_problem2_problem3_problem4_problem5_l599_599003


namespace jason_seashells_l599_599150

theorem jason_seashells :
  let initial_seashells := 49
  let seashells_given_to_tim := 13
  let seashells_given_to_lily := 7
  let seashells_found := 15
  let seashells_lost := 5
  let remaining_seashells :=
    initial_seashells - (seashells_given_to_tim + seashells_given_to_lily) + seashells_found - seashells_lost
  in remaining_seashells = 39 :=
by
  sorry

end jason_seashells_l599_599150


namespace regular_polygon_exterior_angle_change_l599_599011

theorem regular_polygon_exterior_angle_change :
  (let quadrilateral_angle := 360 / 4 in
   let pentagon_angle := 360 / 5 in
   let hexagon_angle := 360 / 6 in
   let nonagon_angle := 360 / 9 in
   let decagon_angle := 360 / 10 in
   
   quadrilateral_angle - pentagon_angle = 18 ∧
   pentagon_angle - hexagon_angle = 12 ∧
   nonagon_angle - decagon_angle = 4) := by
  sorry

end regular_polygon_exterior_angle_change_l599_599011


namespace problem1_problem2_l599_599434

open Real

theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  sqrt a + sqrt b ≤ 2 :=
sorry

theorem problem2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (a + b^3) * (a^3 + b) ≥ 4 :=
sorry

end problem1_problem2_l599_599434


namespace driver_net_rate_of_pay_l599_599681

theorem driver_net_rate_of_pay
  (hours : ℕ)
  (speed : ℕ)
  (fuel_efficiency : ℕ)
  (pay_per_mile : ℚ)
  (gas_cost_per_gallon : ℚ)
  (net_rate_of_pay : ℚ)
  (h1 : hours = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_per_mile = 0.60)
  (h5 : gas_cost_per_gallon = 2.50)
  (h6 : net_rate_of_pay = 25) :
  net_rate_of_pay = (hours * speed * pay_per_mile - (hours * speed / fuel_efficiency) * gas_cost_per_gallon) / hours := 
by sorry

end driver_net_rate_of_pay_l599_599681


namespace probability_of_each_suit_in_five_draws_with_replacement_l599_599109

theorem probability_of_each_suit_in_five_draws_with_replacement :
  let deck_size := 52
  let num_cards := 5
  let num_suits := 4
  let prob_each_suit := 1/4
  let target_probability := 9/16
  prob_each_suit * (3/4) * (1/2) * (1/4) * 24 = target_probability :=
by sorry

end probability_of_each_suit_in_five_draws_with_replacement_l599_599109


namespace Beth_speed_proof_l599_599868

def Jerry_speed : ℝ := 40
def Jerry_trip_duration_minutes : ℝ := 30
def Beth_additional_distance : ℝ := 5
def Beth_additional_time_minutes : ℝ := 20

def Beth_average_speed (Jerry_speed Jerry_trip_duration_minutes Beth_additional_distance Beth_additional_time_minutes : ℝ) : ℝ :=
  let Jerry_trip_duration_hours := Jerry_trip_duration_minutes / 60
  let Jerry_distance := Jerry_speed * Jerry_trip_duration_hours
  let Beth_distance := Jerry_distance + Beth_additional_distance
  let Beth_trip_duration_hours := (Jerry_trip_duration_minutes + Beth_additional_time_minutes) / 60
  Beth_distance / Beth_trip_duration_hours

theorem Beth_speed_proof : Beth_average_speed Jerry_speed Jerry_trip_duration_minutes Beth_additional_distance Beth_additional_time_minutes = 30 := 
  by
    sorry

end Beth_speed_proof_l599_599868


namespace sum_ages_divya_nacho_l599_599842

theorem sum_ages_divya_nacho
  (divya_current_age : ℕ)
  (h1 : divya_current_age = 5)
  (h2 : ∀ n, Nacho_in_5_years n = 3 * (divya_current_age + 5)) :
  (divya_current_age + Nacho_current_age) = 40 :=
by
  let divya_age_5_years := divya_current_age + 5
  have nacho_age_5_years := 3 * divya_age_5_years
  let nacho_current_age := nacho_age_5_years - 5
  exact sorry

end sum_ages_divya_nacho_l599_599842


namespace cos_A_B_l599_599433

theorem cos_A_B (A B: ℝ) (h1: sin A + sin B = 0.75) (h2: cos A + cos B = 1) : cos (A - B) = -0.21875 := 
sorry

end cos_A_B_l599_599433


namespace find_n_l599_599166

theorem find_n (x y : ℕ) (h1 : x = 3) (h2 : y = 3) : 
  let n := x - y^( (x - y) / 3 )
  in n = 2 :=
by
  -- definitions according to the conditions
  subst h1
  subst h2
  let n := 3 - 3^( (3 - 3) / 3 )
  have h : (3 - 3) / 3 = 0 := by simp
  have h_pow : 3^0 = 1 := by simp
  rw [h, h_pow]
  show 3 - 1 = 2, by simp
  sorry

end find_n_l599_599166


namespace find_f_five_thrids_l599_599545

noncomputable def f : ℝ → ℝ := sorry

lemma odd_fun (x : ℝ) : f (-x) = -f(x) :=
sorry

lemma f_transformation (x : ℝ) : f (1 + x) = f (-x) :=
sorry

lemma f_neg_inv : f (-1 / 3) = 1 / 3 :=
sorry

theorem find_f_five_thrids : f (5 / 3) = 1 / 3 :=
by
  -- application of the conditions stated
  have h1 : f(1 + (5 / 3 - 1)) = f(-(5 / 3 - 1)),
    from f_transformation (5 / 3 - 1),
  have h2 : f(-(5 / 3 - 1)) = -f(5 / 3 - 1),
    from odd_fun (5 / 3 - 1),
  have h3 : 5 / 3 - 1 = 2 / 3,
    by norm_num,
  rw [h3] at h1,
  rw [h2] at h1,
  have h4 : f (-1 / 3) = 1 / 3,
    from f_neg_inv,
  rw [←h4] at h1,
  exact h1,
sorry

end find_f_five_thrids_l599_599545


namespace inversely_proportional_x_y_l599_599592

noncomputable def k := 320

theorem inversely_proportional_x_y (x y : ℕ) (h1 : x * y = k) :
  (∀ x, y = 10 → x = 32) ↔ (x = 32) :=
by
  sorry

end inversely_proportional_x_y_l599_599592


namespace hyperbola_and_line_intersection_proof_l599_599419

-- Definitions for the given conditions
def center (C : Type) : Prop := C = (0, 0)
def right_focus (F : Type) : Prop := F = (2, 0)
def eccentricity (e : ℝ) : Prop := e = 2

-- The equation of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := x^2 - (y^2 / 3) = 1

-- The line equation
def line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- The relationship between line and hyperbola
def discriminant (a b c : ℝ) := b^2 - 4 * a * c

noncomputable def intersections (d : ℝ) := d > 0

-- Lean theorem statement
theorem hyperbola_and_line_intersection_proof
(C F : Type) (e : ℝ) (x y : ℝ) :
  center C →
  right_focus F →
  eccentricity e →
  hyperbola_eq x y →
  line_eq x y →
  intersections (discriminant 2 (-2) (-3)) := sorry

end hyperbola_and_line_intersection_proof_l599_599419


namespace range_of_x_l599_599068

theorem range_of_x (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * real.pi) (h3 : real.sqrt (1 - real.sin (2 * x)) = real.sin x - real.cos x) :
  real.pi / 4 ≤ x ∧ x ≤ 5 * real.pi / 4 :=
sorry

end range_of_x_l599_599068


namespace probability_not_perfect_power_l599_599934

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (x : ℕ) (y : ℕ), x > 0 ∧ y > 1 ∧ n = x^y

def count_perfect_powers (n : ℕ) : ℕ :=
  ∑ i in Finset.range (n + 1), if is_perfect_power i then 1 else 0

theorem probability_not_perfect_power :
  let N := 200
  let count_perfect := count_perfect_powers N
  let count_not_perfect := N - count_perfect
  (count_not_perfect : ℚ) / N = 181 / 200 := by
  sorry

end probability_not_perfect_power_l599_599934


namespace impossible_to_equalize_l599_599712

-- Given conditions
def N : Type := ℕ
def grid := vector (vector ℕ 4) 4

-- Natural number operations on row, column, or diagonal
inductive Operation
| row (r : fin 4) (k : N)
| col (c : fin 4) (k : N)
| diag1 (k : N)
| diag2 (k : N)

-- Define the initial sum of the grid
def initial_sum : ℕ := 70

-- Define the transformation under operations and sum invariance
def transform (g : grid) (op : Operation) : grid := sorry
def sum_mod4 (g : grid) : ℕ := (g.to_list.foldl (+) 0) % 4

-- Proof goal
theorem impossible_to_equalize (g : grid) (S : initial_sum = (g.to_list.foldl (+) 0)) :
  ∀ op_list : list Operation, 
  (sum_mod4 g) ≠ (sum_mod4 (op_list.foldl transform g id)) -> 
  (∃ x, grid.map (λ _ _, x) = g) → False := sorry

end impossible_to_equalize_l599_599712


namespace quadratic_vertex_coordinates_l599_599220

theorem quadratic_vertex_coordinates (x y : ℝ) (h : y = 2 * x^2 - 4 * x + 5) : (x, y) = (1, 3) :=
sorry

end quadratic_vertex_coordinates_l599_599220


namespace f_sub_f_inv_eq_2022_l599_599622

def f (n : ℕ) : ℕ := 2 * n
def f_inv (n : ℕ) : ℕ := n

theorem f_sub_f_inv_eq_2022 : f 2022 - f_inv 2022 = 2022 := by
  -- Proof goes here
  sorry

end f_sub_f_inv_eq_2022_l599_599622


namespace election_winner_votes_l599_599252

theorem election_winner_votes (V : ℕ) (h1 : (0.58 * V) % 1 = 0) (h2 : 0.58 * V - 0.42 * V = 288) :
  0.58 * V = 1044 :=
by
  sorry

end election_winner_votes_l599_599252


namespace tangent_distance_correct_l599_599949

noncomputable def distance_from_large_circle_center_to_tangency_point (r: ℝ) (n: ℕ) : ℝ :=
  if (r = 2) ∧ (n = 3) then
    2 * real.sqrt 3 - 2
  else
    0

theorem tangent_distance_correct :
  distance_from_large_circle_center_to_tangency_point 2 3 = 2 * real.sqrt 3 - 2 :=
by
  -- proof is omitted
  sorry

end tangent_distance_correct_l599_599949


namespace pentagon_tiling_possible_l599_599664

/--
  Given a regular pentagon ABCDE and a point B' which is the reflection of B across the line AC,
  prove that pentagons congruent to AB'CDE can tile the plane.
-/
theorem pentagon_tiling_possible (ABCDE : Type) [regular_pentagon ABCDE] (B' : Type) [reflection B AC B'] :
  ∃ (AB'CDE : Type) [tiling AB'CDE], tiling_possible AB'CDE := by
  sorry

end pentagon_tiling_possible_l599_599664


namespace third_shot_scores_l599_599955

noncomputable theory

open Finset

def shooters_scores (a b : Fin 5 → ℕ) : Prop :=
(∀ i, a i ∈ {10, 9, 8, 5, 4, 3, 2}.erase 7) ∧
(∀ i, b i ∈ {10, 9, 8, 5, 4, 3, 2}.erase 6 ∧ b i ∉ {a i}) ∧
(a 0 + a 1 + a 2 = b 0 + b 1 + b 2) ∧
(a 2 + a 3 + a 4 = 3 * (b 2 + b 3 + b 4))

theorem third_shot_scores (a b : Fin 5 → ℕ) (h : shooters_scores a b) :
  a 2 = 10 ∧ b 2 = 2 :=
sorry

end third_shot_scores_l599_599955


namespace find_a_and_x_l599_599132

theorem find_a_and_x (x a : ℤ)
(h1: Bingbing_solved_incorrectly: (2 * (2 * x - 1) + 1 = 5 * (x + a)))
(h2: Bingbing_solution: x = -6):
a = 1 ∧ x = 3 := 
  sorry

end find_a_and_x_l599_599132


namespace arctan_sum_is_pi_over_4_l599_599849

open Real

theorem arctan_sum_is_pi_over_4 (a b c : ℝ) (h1 : b = c) (h2 : c / (a + b) + a / (b + c) = 1) :
  arctan (c / (a + b)) + arctan (a / (b + c)) = π / 4 :=
by 
  sorry

end arctan_sum_is_pi_over_4_l599_599849


namespace correct_multiplier_l599_599306

theorem correct_multiplier (x : ℕ) 
  (h1 : 137 * 34 + 1233 = 137 * x) : 
  x = 43 := 
by 
  sorry

end correct_multiplier_l599_599306


namespace tan_alpha_add_pi_div_four_l599_599417

theorem tan_alpha_add_pi_div_four {α : ℝ} (h1 : α ∈ Set.Ioo 0 (Real.pi)) (h2 : Real.cos α = -4/5) :
  Real.tan (α + Real.pi / 4) = 1 / 7 :=
sorry

end tan_alpha_add_pi_div_four_l599_599417


namespace eccentricity_range_l599_599780

-- Definition for the hyperbola
def hyperbola (a b : ℝ) := {p : ℝ × ℝ // (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

-- Definitions for the foci and the point on the hyperbola
variables {a b : ℝ}
def left_focus (a b : ℝ) : ℝ × ℝ := (-sqrt (a^2 + b^2), 0)
def right_focus (a b : ℝ) : ℝ × ℝ := (sqrt (a^2 + b^2), 0)
def point_on_hyperbola (a b : ℝ) : {p : ℝ × ℝ // (p.1^2 / a^2) - (p.2^2 / b^2) = 1} := sorry

-- Given condition
axiom point_condition (a b : ℝ) (p : ℝ × ℝ) :
  (p ∈ hyperbola a b) → (p = point_on_hyperbola a b) → 
  (let d1 := dist p (left_focus a b) in
  let d2 := dist p (right_focus a b) in
  (d1^2 / d2) = 8 * a)

-- Eccentricity of the hyperbola
def eccentricity (a b : ℝ) : ℝ := sqrt (1 + b^2 / a^2)

-- The main theorem
theorem eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (p : {p : ℝ × ℝ // (p.1^2 / a^2) - (p.2^2 / b^2) = 1}) :
  (let e := eccentricity a b in
  (1 < e ∧ e ≤ 3)) → 
  true :=
by
  sorry

end eccentricity_range_l599_599780


namespace sin_neg_angle_l599_599244

theorem sin_neg_angle :
  (sin (-150 * Real.pi / 180) = -1 / 2) :=
by
  -- we state the theorem according to the provided conditions and expected result
  sorry

end sin_neg_angle_l599_599244


namespace number_of_lattice_points_eq_30_l599_599861

theorem number_of_lattice_points_eq_30 : 
  {p : ℤ × ℤ × ℤ // p.1 * p.1 + p.2.1 * p.2.1 + p.2.2 * p.2.2 = 16}.to_finset.card = 30 := 
by 
  sorry

end number_of_lattice_points_eq_30_l599_599861


namespace investor_wait_time_l599_599708

noncomputable def compound_interest_time (P A r : ℝ) (n : ℕ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem investor_wait_time :
  compound_interest_time 600 661.5 0.10 2 = 1 := 
sorry

end investor_wait_time_l599_599708


namespace max_area_CDFE_l599_599663

theorem max_area_CDFE
  (A B C D E F : Point)
  (h_sq : Square A B C D 1)
  (h1 : OnLine E A B)
  (h2 : OnLine F A D)
  (h3 : distance A E = 2 * distance A F) :
  max (area (Quadrilateral C D F E)) = 1 / 2 :=
sorry

end max_area_CDFE_l599_599663


namespace multiplicative_magic_square_sum_e_l599_599012

theorem multiplicative_magic_square_sum_e : 
  (∀ (a b c d f P e : ℕ), P > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ P = 2 * a * b ∧ 
  P = c * 10 * d ∧ P = e * f * 40 ∧ P = 2 * 10 * 40 ∧ P = b * 10 * e → 
  (∃ es : Finset ℕ, (∀ x ∈ es, x ∈ {1, 2, 4, 5, 8, 10, 16, 20, 40, 80}) ∧ 
  es.sum = 186)) :=
sorry

end multiplicative_magic_square_sum_e_l599_599012


namespace coeff_multiples_of_prime_l599_599437

-- Definitions to be used in Lean based on given conditions
variables (f g : Polynomial ℤ) (p : ℤ)
hypothesis h1 : ∀ a ∈ f.coeffs, p ∣ a
hypothesis h2 : ∀ b ∈ g.coeffs, p ∣ b
hypothesis hp : Prime p

-- Theorem to be proved
theorem coeff_multiples_of_prime :
  (∃ a ∈ f.coeffs, p ∣ a) ∨ (∃ b ∈ g.coeffs, p ∣ b) :=
sorry

end coeff_multiples_of_prime_l599_599437


namespace abs_eq_5_iff_l599_599103

   theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 :=
   by
     sorry
   
end abs_eq_5_iff_l599_599103


namespace ellipse_equation_l599_599228

theorem ellipse_equation (A B : ℝ) (hA : A > 0) (hB : B > 0)
  (hline_focus_vertex : ∃ (F V : ℝ × ℝ), (2 * F.1 - F.2 + 2 = 0)
                      ∧ (2 * V.1 - V.2 + 2 = 0)
                      ∧ ((F = (sqrt (A - B), 0) ∨ F = (0, sqrt (B - A)))
                      ∧ (V = (sqrt (A), 0) ∨ V = (0, sqrt (B)))))
  : (A = 5 ∧ B = 4) ∨ (A = 4 ∧ B = 5) := 
sorry

end ellipse_equation_l599_599228


namespace prime_sum_product_l599_599944

theorem prime_sum_product : ∃ (p1 p2 p3 : ℕ), 
  prime p1 ∧ prime p2 ∧ prime p3 ∧ p1 + p2 + p3 = 47 ∧ p1 * p2 * p3 = 1705 :=
by
  sorry

end prime_sum_product_l599_599944


namespace cosine_squared_identity_l599_599429

theorem cosine_squared_identity (α : ℝ) (h : Real.sin (2 * α) = 1 / 3) : Real.cos (α - (π / 4)) ^ 2 = 2 / 3 :=
sorry

end cosine_squared_identity_l599_599429


namespace triangle_CMNs_shape_is_right_l599_599848

-- Let ABC be an isosceles right triangle with right angle at C
variables {A B C M N : Type} [IsoscelesRightTriangle ABC C] -- Custom type for triangle
noncomputable def AM (AB : Real) (M : Point) (A : Point) : Real :=
  -- Define AM dynamically; use a value if provided.
  sorry

noncomputable def MN (AB : Real) (M N : Point) : Real :=
  -- Define MN dynamically
  sorry

noncomputable def BN (AB : Real) (N : Point) (B : Point) : Real :=
  -- Define BN dynamically
  sorry

theorem triangle_CMNs_shape_is_right (A B C M N : Point) 
  (AM : Real) (MN : Real) (BN : Real) 
  (h1 : IsIsoscelesRightTriangle A B C C_90_MCB_MCN_45ny) 
  (h2 : AM = m) (h3 : MN = x) (h4 : BN = n) 
  (h5 : A + B + C + M + N = AB)
  :
  (x^2 = m^2 + n^2) → is_right_triangle (Triangle.mk x m n) :=
begin
  sorry
end

end triangle_CMNs_shape_is_right_l599_599848


namespace coefficient_inequality_l599_599886

noncomputable theory

variables {n : ℕ} (a_n a_{n-1} ... a_2 : ℝ)
variables (roots : Fin n → ℝ)
-- Assuming the polynomial has n positive real roots
-- Consider the polynomial
def polynomial := λ x, a_n * x^n + a_{n-1} * x^(n-1) + ... + a_2 * x^2 - x + 1

-- Condition that the polynomial has n positive real roots
axiom positive_real_roots : ∀ i, 0 < roots i

-- Proving the given inequality
theorem coefficient_inequality
  (h_poly_roots : ∀ x, polynomial x = 0 → x ∈ finset.card (Fin n))
  (h_vieta_sum : finset.sum_univ (λ i, (1/(roots i))) = 1) :
  0 < 2^2 * a_2 + 2^3 * a_3 + ... + 2^n * a_n ∧
  2^2 * a_2 + 2^3 * a_3 + ... + 2^n * a_n ≤ ( (n - 2) / n )^2 + 1 :=
sorry -- Proof omitted

end coefficient_inequality_l599_599886


namespace power_of_128_div_7_eq_16_l599_599719

theorem power_of_128_div_7_eq_16 : (128 : ℝ) ^ (4 / 7) = 16 := by
  sorry

end power_of_128_div_7_eq_16_l599_599719


namespace number_of_possible_a_values_l599_599929

-- Define the function f(x)
def f (a x : ℝ) := abs (x + 1) + abs (a * x + 1)

-- Define the condition for the minimum value
def minimum_value_of_f (a : ℝ) := ∃ x : ℝ, f a x = (3 / 2)

-- The proof problem statement
theorem number_of_possible_a_values : 
  (∃ (a1 a2 a3 a4 : ℝ),
    minimum_value_of_f a1 ∧
    minimum_value_of_f a2 ∧
    minimum_value_of_f a3 ∧
    minimum_value_of_f a4 ∧
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a3 ≠ a4) :=
sorry

end number_of_possible_a_values_l599_599929


namespace find_m_find_maximum_value_l599_599809

-- Definitions
noncomputable def f (x : ℝ) (m : ℝ) := |x - 2| - m

-- Problem statement (1)
theorem find_m :
  (∀ x : ℝ, f (x + 2) 1 ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1) →
  (∀ m : ℝ, (∀ x : ℝ, f (x + 2) m ≤ 0 ↔ -m ≤ x ∧ x ≤ m) → m = 1) :=
by
  intros h1 h2,
  specialize h2 1,
  have h3 : ∀ x : ℝ, (∀ m : ℝ, f (x + 2) m ≤ 0 ↔ -m ≤ x ∧ x ≤ m) → m = 1 := sorry,
  sorry

-- Problem statement (2)
theorem find_maximum_value (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) (h : a^2 + b^2 + c^2 = 1) :
  a + 2 * b + 3 * c ≤ Real.sqrt 14 :=
by
  apply Cauchy_Schwarz a b c,
  sorry

end find_m_find_maximum_value_l599_599809


namespace roots_in_interval_l599_599305

def g : ℝ → ℝ := sorry

theorem roots_in_interval :
  (∀ x : ℝ, g (3 + x) = g (3 - x)) →
  (∀ x : ℝ, g (8 + x) = g (8 - x)) →
  (g 0 = 0) →
  (finset.card (finset.filter (λ x, g x = 0) (finset.Ico (-1000) 1001).attach) ≥ 401) :=
by
  intros h1 h2 h3
  sorry

end roots_in_interval_l599_599305


namespace exist_n_squared_integers_l599_599160

open Set

theorem exist_n_squared_integers (S : Set ℕ) (n : ℕ) (hS : S.size = n) (hn : 2 ≤ n) :
  ∃ M : Set ℕ, M.size ≥ n^2 ∧ ∀ m ∈ M, ∃ x y z ∈ S, m = x + y * z := 
sorry

end exist_n_squared_integers_l599_599160


namespace angle_between_lateral_face_and_base_is_arctan_l599_599226

noncomputable def angle_between_lateral_face_and_base (a : ℝ) : ℝ :=
  let PM := a
  let MK := (a * Real.sqrt 3) / 2
  Real.arctan (PM / MK)

theorem angle_between_lateral_face_and_base_is_arctan (a : ℝ) :
  angle_between_lateral_face_and_base a = Real.arctan (2 / Real.sqrt 3) :=
by
  let PM := a
  let MK := (a * Real.sqrt 3) / 2
  have h1 : PM / MK = 2 / Real.sqrt 3 := by
    calc
      PM / MK = a / ((a * Real.sqrt 3) / 2)  := by sorry
      ...     = 2 / Real.sqrt 3             := by sorry
  show angle_between_lateral_face_and_base a = Real.arctan (2 / Real.sqrt 3) from
    Eq.trans (by expand angle_between_lateral_face_and_base) (by rw [h1])

end angle_between_lateral_face_and_base_is_arctan_l599_599226


namespace equilateral_triangle_side_length_l599_599191

theorem equilateral_triangle_side_length (P Q R S : Point) (A B C : Point)
  (h1 : equilateral_triangle A B C)
  (h2 : foot_of_perpendicular P A B = Q)
  (h3 : foot_of_perpendicular P B C = R)
  (h4 : foot_of_perpendicular P C A = S)
  (h5 : distance P Q = 3)
  (h6 : distance P R = 4)
  (h7 : distance P S = 5) :
  distance A B = 8 * Real.sqrt 3 :=
sorry

end equilateral_triangle_side_length_l599_599191


namespace highest_power_3_divides_2000_factorial_l599_599023

noncomputable def highest_power_3_divides_factorial (n : ℕ) : ℕ :=
∑ k in finset.range(1 + nat.log 3 n), n / 3^k

theorem highest_power_3_divides_2000_factorial : highest_power_3_divides_factorial 2000 = 996 := by
sorry

end highest_power_3_divides_2000_factorial_l599_599023


namespace most_friends_count_l599_599846

noncomputable def common_friend_structure (m : ℕ) (h : m ≥ 3) : Type :=
{ people : Type,
  friend : people → people → Prop,
  mutual_friend : ∀ a b : people, friend a b → friend b a,
  no_self_friend : ∀ a : people, ¬ friend a a,
  common_friend : ∀ (s : Finset people), s.card = m → ∃ a : people, ∀ b ∈ s, friend a b }

theorem most_friends_count (m : ℕ) (h : m ≥ 3) :
  ∃ (P : common_friend_structure m h), ∃ a : P.people, (Finset.card (Finset.filter (λ b, P.friend a b) P.people)) = m - 1 := sorry

end most_friends_count_l599_599846


namespace integral_of_binomial_constant_term_l599_599837

noncomputable def binomial_term (n k : ℕ) (a b : ℝ) (x : ℝ) : ℝ :=
  Nat.choose n k * (a ^ (n - k)) * (b ^ k) * (x ^ ((2 * (n - k)) - k))

theorem integral_of_binomial_constant_term :
  let m := binomial_term 6 4 (sqrt 5 / 5) 1 1 in
  ∫ x in Ioc 1 m, (x^2 - 2 * x) = (2 / 3) :=
by
  -- Fill with the appropriate proof steps
  sorry

end integral_of_binomial_constant_term_l599_599837


namespace perimeter_of_figure_is_nine_l599_599310

-- Define the initial structure conditions
def initial_grid : set (ℕ × ℕ) := { (i, j) | i < 3 ∧ j < 3 }

-- Define removal of the central square
def without_central_square : set (ℕ × ℕ) := initial_grid \ {(1, 1)}

-- Define addition of the new row at the top
def with_added_row : set (ℕ × ℕ) := without_central_square ∪ { (0, j) | j < 3 }

-- Define the perimeter calculation
noncomputable def perimeter (shape : set (ℕ × ℕ)) : ℕ := sorry

-- Prove the perimeter of the resulting figure is 9 units
theorem perimeter_of_figure_is_nine : perimeter with_added_row = 9 := by
  sorry

end perimeter_of_figure_is_nine_l599_599310


namespace scientific_notation_0_0000314_l599_599233

theorem scientific_notation_0_0000314 :
  (0.0000314 : ℝ) = 3.14 * 10^(-5) := 
by 
  sorry

end scientific_notation_0_0000314_l599_599233


namespace shift_sine_graph_l599_599951

theorem shift_sine_graph :
  ∀ x : ℝ, (sin (2 * (x + π / 4) - π / 3)) = (sin (2 * x + π / 6)) := 
by
  sorry

end shift_sine_graph_l599_599951


namespace number_of_even_prime_sums_l599_599237

open List

noncomputable def is_prime (n : ℕ) : Prop :=
n > 1 ∧ (∀ d ∣ n, d = 1 ∨ d = n)

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

noncomputable def first_n_sums (n : ℕ) (l : List ℕ) : List ℕ :=
(scanl (+) 0 l).drop 1 |>.take n

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem number_of_even_prime_sums : 
  let sums := first_n_sums 15 primes in
  (sums.filter is_even).countp is_prime = 0 :=
by
  sorry

end number_of_even_prime_sums_l599_599237


namespace chromium_percentage_alloy_A5_l599_599492

def chromium_percentage (w1 w2 w3 w4 : ℝ) (p1 p2 p3 p4 : ℝ) : ℝ :=
  let total_weight := w1 + w2 + w3 + w4
  let total_chromium := (p1 / 100) * w1 + (p2 / 100) * w2 + (p3 / 100) * w3 + (p4 / 100) * w4
  (total_chromium / total_weight) * 100

theorem chromium_percentage_alloy_A5 :
  chromium_percentage 15 35 25 10 12 10 8 15 = 10.35 :=
sorry

end chromium_percentage_alloy_A5_l599_599492


namespace find_y_l599_599230

theorem find_y : 
  (6 + 10 + 14 + 22) / 4 = (15 + y) / 2 → y = 11 :=
by
  intros h
  sorry

end find_y_l599_599230


namespace cannot_be_expressed_as_x_squared_plus_y_fifth_l599_599650

theorem cannot_be_expressed_as_x_squared_plus_y_fifth :
  ¬ ∃ x y : ℤ, 59121 = x^2 + y^5 :=
by sorry

end cannot_be_expressed_as_x_squared_plus_y_fifth_l599_599650


namespace sqrt_simplification_l599_599201

theorem sqrt_simplification (h : 16384 = 4 ^ 7) : 
  sqrt (sqrt (sqrt (1 / 16384))) = 1 / 2 ^ (7 / 8) :=
by
  sorry -- Proof to be provided

end sqrt_simplification_l599_599201


namespace problem_part_a_problem_part_b_l599_599661

noncomputable def canPartitionEqually (A : ℕ → ℝ) (n k : ℕ) : Prop :=
  ∃ (P : Set ℕ), P ⊆ Finset.range n ∧ 
                 (∀ i ∈ P, 1 ≤ i ∧ i < n) ∧ 
                 |P| = k - 1 ∧ 
                 ∀ (a b ∈ P), abs (A a - A b) ≤ 1

theorem problem_part_a (A : ℕ → ℝ) (n : ℕ) 
  (h : ∀ i : ℕ, 1 ≤ i ∧ i < n → abs (A (i+1) - A i) ≤ 1) 
  : canPartitionEqually A n 3 :=
  sorry

theorem problem_part_b (A : ℕ → ℝ) (n k : ℕ) 
  (h1 : 1 < n)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i < n → abs (A (i+1) - A i) ≤ 1) 
  (h3 : k < n - 1)
  : canPartitionEqually A n k :=
  sorry

end problem_part_a_problem_part_b_l599_599661


namespace algebra_problem_l599_599791

-- Definition of variable y
variable (y : ℝ)

-- Given the condition
axiom h : 2 * y^2 + 3 * y + 7 = 8

-- We need to prove that 4 * y^2 + 6 * y - 9 = -7 given the condition
theorem algebra_problem : 4 * y^2 + 6 * y - 9 = -7 :=
by sorry

end algebra_problem_l599_599791


namespace inequality_of_pos_reals_l599_599175

open Real

theorem inequality_of_pos_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤
  (1 / 4) * (a + b + c) :=
by
  sorry

end inequality_of_pos_reals_l599_599175


namespace evaluate_expression_l599_599031

theorem evaluate_expression :
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = 137 / 52 :=
by
  sorry

end evaluate_expression_l599_599031


namespace sum_Q_mod_1000_l599_599401

def Q (m : ℕ) : ℕ := sorry -- Definition of Q(m) would need to be shown here.

theorem sum_Q_mod_1000 :
  (∑ m in Finset.range 2016 \ Finset.range 1, Q (m + 2)) % 1000 = 59 := 
sorry

end sum_Q_mod_1000_l599_599401


namespace sin_series_positive_l599_599992

theorem sin_series_positive (x : ℝ) (h : 24 * x ∈ Set.Ioo 0 π) (n : ℕ) : 
  let f := ∑ k in Finset.range n, (Real.sin ((2 * k + 1) * x)) / (2 * k + 1)
  in f > 0 :=
sorry

end sin_series_positive_l599_599992


namespace five_pow_x_minus_2y_l599_599831

variable (x y : ℝ) -- declare x and y as real numbers

-- conditions given in the problem
axiom cond1 : 5^x = 18 
axiom cond2 : 5^y = 3 

-- statement to prove
theorem five_pow_x_minus_2y : 5^(x - 2 * y) = 2 := by
  sorry -- proof placeholder

end five_pow_x_minus_2y_l599_599831


namespace lateral_surface_area_correct_l599_599245

def V : ℝ := 8 -- volume in cubic meters
def h : ℝ := 2.2 -- height in meters
def A_base : ℝ := V / h -- base area of the prism
def a_side : ℝ := Real.sqrt (A_base / (2 * (1 + Real.sqrt 2))) -- side length of the regular octagon
def perimeter_base : ℝ := 8 * a_side -- perimeter of the base
def lateral_surface_area : ℝ := perimeter_base * h -- lateral surface area of the prism

theorem lateral_surface_area_correct : 
  lateral_surface_area = 16 * Real.sqrt (2.2 * (Real.sqrt 2 - 1)) :=
  sorry

end lateral_surface_area_correct_l599_599245


namespace latticePointsCount_l599_599863

-- Define what a lattice point is and the condition
def isLatticePoint (x y z : ℤ) : Prop := x^2 + y^2 + z^2 = 16

-- The theorem stating the number of such lattice points
theorem latticePointsCount :
  (Finset.univ : Finset (ℤ × ℤ × ℤ)).filter (λ (p : ℤ × ℤ × ℤ), isLatticePoint p.1 p.2 p.3).card = 50 :=
  sorry

end latticePointsCount_l599_599863


namespace exists_a_b_l599_599560

noncomputable def poly (n : ℕ) (c : Fin n → ℝ) (z : ℂ) : ℂ :=
  z^(n) + ∑ i in Finset.range n, (c ⟨i, (Nat.lt_succ_self _)⟩ : ℂ) * z ^ (n - 1 - i)

theorem exists_a_b (n : ℕ) (c : Fin n → ℝ) (h : Complex.abs (poly n c Complex.I) < 1) :
  ∃ a b : ℝ, poly n c (Complex.ofReal a + Complex.I * b) = 0 ∧ (a^2 + b^2 + 1)^2 < 4 * b^2 + 1 :=
sorry

end exists_a_b_l599_599560


namespace fraction_to_decimal_l599_599747

theorem fraction_to_decimal (a b : ℕ) (h₀ : a = 49) (h₁ : b = 160) : a / b = 0.30625 :=
by {
  -- Assume the given conditions
  assume h₀ : a = 49,
  assume h₁ : b = 160,
  -- Prove the theorem
  sorry
}

end fraction_to_decimal_l599_599747


namespace coverage_is_20_l599_599601

noncomputable def cost_per_kg : ℝ := 60
noncomputable def total_cost : ℝ := 1800
noncomputable def side_length : ℝ := 10

-- Surface area of one side of the cube
noncomputable def area_side : ℝ := side_length * side_length

-- Total surface area of the cube
noncomputable def total_area : ℝ := 6 * area_side

-- Kilograms of paint used
noncomputable def kg_paint_used : ℝ := total_cost / cost_per_kg

-- Coverage per kilogram of paint
noncomputable def coverage_per_kg (total_area : ℝ) (kg_paint_used : ℝ) : ℝ := total_area / kg_paint_used

theorem coverage_is_20 : coverage_per_kg total_area kg_paint_used = 20 := by
  sorry

end coverage_is_20_l599_599601


namespace correct_operation_l599_599968

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l599_599968


namespace fraction_multiplication_l599_599269

theorem fraction_multiplication :
  (3 / 4 : ℚ) * (1 / 2) * (2 / 5) * 5000 = 750 :=
by
  norm_num
  done

end fraction_multiplication_l599_599269


namespace find_y_with_factors_sum_l599_599942

theorem find_y_with_factors_sum (y : ℕ) (hy1 : sum_factors y = 36) (hy2 : 4 ∣ y) : y = 16 := sorry

end find_y_with_factors_sum_l599_599942


namespace division_problem_l599_599108

theorem division_problem (h : 7125 / 1.25 = 5700) : 712.5 / 12.5 = 570 := 
by 
  have eq1 : 7125 = 10 * 712.5 := by norm_num
  have eq2 : 1.25 = 10 * 0.125 := by norm_num
  rw [← eq1, ← eq2] at h
  exact sorry

end division_problem_l599_599108


namespace fraction_shaded_area_l599_599054

theorem fraction_shaded_area (PX XQ : ℝ) (PA PR PQ : ℝ) (h1 : PX = 1) (h2 : 3 * XQ = PX) (h3 : PQ = PR) (h4 : PA = 1) (h5 : PA + AR = PR) (h6 : PR = 4):
  (3 / 16 : ℝ) = 0.375 :=
by
  -- proof here
  sorry

end fraction_shaded_area_l599_599054


namespace venn_diagram_correct_l599_599196

theorem venn_diagram_correct (P Q : set α) :
  \overline{\bar{P} \cup \bar{Q}} = P ∩ Q :=
by 
  -- Step 1: Apply De Morgan's Law
  have h1 : \overline{\bar{P} ∪ \bar{Q}} = \overline{\overline{P ∩ Q}},
  { apply DeMorgan_Union },
  -- Step 2: Simplify using double complement rule
  have h2 : \overline{\overline{P ∩ Q}} = P ∩ Q,
  { apply double_complement },
  -- Step 3: Combine
  exact h1.trans h2
end

end venn_diagram_correct_l599_599196


namespace planes_perpendicular_of_line_perpendicular_and_parallel_l599_599882

-- Definitions
variables {Point Line Plane : Type} [EuclideanGeometry Point Line Plane]

-- Variables representing the given entities
variables (m : Line) (n : Line) (α : Plane) (β : Plane)

-- Definitions for perpendicularity and parallelism
def perpendicular (l : Line) (p : Plane) : Prop := sorry -- Define a line perpendicular to a plane
def parallel (l : Line) (p : Plane) : Prop := sorry -- Define a line parallel to a plane
def plane_perpendicular (p1 : Plane) (p2 : Plane) : Prop := sorry -- Define perpendicularity of two planes

-- The main theorem statement
theorem planes_perpendicular_of_line_perpendicular_and_parallel
  (h1 : perpendicular m α)
  (h2 : parallel m β) :
  plane_perpendicular α β :=
sorry

end planes_perpendicular_of_line_perpendicular_and_parallel_l599_599882


namespace rectangle_area_proof_l599_599262

structure Point where
  x : ℝ
  y : ℝ

def distance (P Q : Point) : ℝ :=
  Real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

def P : Point := {x := 1, y := 2}
def Q : Point := {x := -3, y := 3}
def R : Point := {x := -2, y := 7}
def S : Point := {x := 2, y := 6}

noncomputable def area_of_rectangle : ℝ :=
  let PQ := distance P Q
  let QR := distance Q R
  PQ * QR

theorem rectangle_area_proof : area_of_rectangle = 17 := by 
  sorry

end rectangle_area_proof_l599_599262


namespace tangent_line_a_neg1_symmetry_values_range_for_extreme_values_l599_599089

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (1 / x + a) * Real.log (1 + x)

-- Part 1: Prove the tangent line equation at a = -1
theorem tangent_line_a_neg1 :
  let a := -1
  f 1 a = 0 ∧ f'(1) = -Real.log 2 →
  ∀ x, f'(1) * (x - 1) = -Real.log 2 * (x - 1) := sorry

-- Part 2: Prove the symmetry and find values of a and b
theorem symmetry_values : 
  ∃ (a b : ℝ), b = -1/2 ∧ f (1) a = (1 + a) * Real.log (2) ∧ f (-2) a = (2 - a) * Real.log (2) →
  a = 1/2 ∧ b = -1/2 := sorry

-- Part 3: Prove range of a for extreme values in (0, +∞)
theorem range_for_extreme_values : 
  (∀ x, ∃ a, 0 < a ∧ a < 1/2 → has_extreme_value (f x a)):= sorry

end tangent_line_a_neg1_symmetry_values_range_for_extreme_values_l599_599089


namespace number_of_valid_distributions_l599_599327

-- The triangular array and its properties
def triangular_array : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 3
-- continues until 12
| 12 := 12
| _ := 0  -- for completeness

-- Initial conditions
constant rows : ℕ := 12
constant squares : Π (k : ℕ), k ≤ rows → ℕ

axiom bottom_row_entries : vector (fin 2) (triangular_array 12)

-- This is the relevant calculation for the top square modulo 5
def top_square (entries : vector (fin 2) 12) : fin 5 :=
(entries.nth 0 + entries.nth 1 + entries.nth 10 + entries.nth 11) % 5

-- The theorem stating the problem's solution
theorem number_of_valid_distributions : 
  {d : vector (fin 2) 12 // top_square d = 0}.card = 1024 := 
sorry

end number_of_valid_distributions_l599_599327


namespace length_of_bridge_l599_599701

noncomputable def train_length := 250  -- in meters
noncomputable def time_to_pass_bridge := 41.142857142857146 -- in seconds
noncomputable def speed_kmh := 35 -- in km/hour

noncomputable def speed_ms := speed_kmh * 1000 / 3600 -- converting to meters/second

theorem length_of_bridge:
  let distance_covered := speed_ms * time_to_pass_bridge in
  let length_of_bridge := distance_covered - train_length in
  length_of_bridge = 150 :=
by
  sorry

end length_of_bridge_l599_599701


namespace platform_length_is_correct_l599_599653

-- Defining the constants
constant train_length : ℝ := 450
constant train_speed_kmph : ℝ := 126
constant cross_time_sec : ℝ := 20

-- Calculating the speed in m/s
def train_speed_mps := train_speed_kmph * (1000 / 3600)

-- Calculating the total distance covered
def total_distance_covered := train_speed_mps * cross_time_sec

-- Defining the length of platform
def platform_length := total_distance_covered - train_length

-- The main theorem stating that the length of the platform is 250 meters
theorem platform_length_is_correct : platform_length = 250 := by
  sorry

end platform_length_is_correct_l599_599653


namespace tan_identity_l599_599441

theorem tan_identity (h : tan 45 = tan (20 + 25)) : (1 + tan 20) * (1 + tan 25) = 2 := 
  by sorry

end tan_identity_l599_599441


namespace angle_ADB_eq_angle_AEC_l599_599346

section

variables {α β γ δ ε : Type}
variables [euclideanGeometry α] [euclideanGeometry β] [euclideanGeometry γ]

-- Variables for points and triangles
variables {A B C P Q D E O O1 O2 S T : α}

-- Conditions
variables [circumcenter O (triangle A B C)]
variables (P_on_BC : P ∈ line_segment B C)
variables (Q_perp_OQ_PQ : ⊥.pair OQ PQ)
variables (D_on_BPQCircumcircle : D ∈ circle_through_points [B, P, Q] ∧ D ≠ P)
variables (E_on_CPQCircumcircle : E ∈ circle_through_points [C, P, Q] ∧ E ≠ P)
variables (PD_parallel_AC : parallel (line_through P D) (line_through A C))
variables (PE_parallel_AB : parallel (line_through P E) (line_through A B))

-- Goal to prove
theorem angle_ADB_eq_angle_AEC : ∠ A D B = ∠ A E C :=
sorry

end

end angle_ADB_eq_angle_AEC_l599_599346


namespace intersection_A_B_l599_599461

def A := {x : ℝ | x^2 - x - 2 ≤ 0}
def B := {x : ℝ | ∃ y : ℝ, y = Real.log (1 - x)}

theorem intersection_A_B : (A ∩ B) = {x : ℝ | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_A_B_l599_599461


namespace total_peaches_in_baskets_l599_599247

/-- There are 11 baskets of peaches, with each basket containing 10 red peaches and 18 green peaches. -/
theorem total_peaches_in_baskets :
  let number_of_baskets := 11
  let peaches_per_basket := 10 + 18
  (number_of_baskets * peaches_per_basket) = 308 :=
begin
  sorry
end

end total_peaches_in_baskets_l599_599247


namespace part_I_part_II_case1_part_II_case2_part_II_case3_part_III_l599_599890

noncomputable def f (k x : ℝ) : ℝ := (x - k) * Real.exp x
noncomputable def f_prime (k x : ℝ) : ℝ := (x - k + 1) * Real.exp x

-- Part (I)
theorem part_I (k : ℝ) : (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f_prime k x > 0) → k ≤ 0 :=
sorry

-- Part (II)
theorem part_II_case1 (k : ℝ) (h : k ≤ 1) : ∃ (x ∈ Set.Icc (0 : ℝ) 1), f k x = -k :=
sorry

theorem part_II_case2 (k : ℝ) (h1 : 1 < k) (h2 : k < 2) : ∃ x, x = k - 1 ∧ f k x = -(Real.exp (k - 1)) :=
sorry

theorem part_II_case3 (k : ℝ) (h : k ≥ 2) : ∃ (x ∈ Set.Icc (0 : ℝ) 1), f k x = (1 - k) * Real.exp 1 :=
sorry

-- Part (III)
theorem part_III (a : ℝ) : (∀ x1 x2 ∈ Set.Ioi a, x1 < x2 → x1 * (f 0 x2 - f 0 a) - x2 * (f 0 x1 - f 0 a) > a * (f 0 x2 - f 0 x1)) ↔ (a ∈ Set.Ici (-2 : ℝ)) :=
sorry

end part_I_part_II_case1_part_II_case2_part_II_case3_part_III_l599_599890


namespace people_group_exists_l599_599581

theorem people_group_exists (n : ℕ) (n_ge_2 : n ≥ 2) :
  ∃ (A B : Fin n), 
    let remaining := (Finset.univ : Finset (Fin n)).erase A in
    let subset := remaining.erase B in
    subset.card ≥ ((n / 2) : ℕ) - 1 :=
begin
  sorry
end

end people_group_exists_l599_599581


namespace mul_103_97_l599_599007

theorem mul_103_97 : 103 * 97 = 9991 := by
  sorry

end mul_103_97_l599_599007


namespace find_f_five_thrids_l599_599544

noncomputable def f : ℝ → ℝ := sorry

lemma odd_fun (x : ℝ) : f (-x) = -f(x) :=
sorry

lemma f_transformation (x : ℝ) : f (1 + x) = f (-x) :=
sorry

lemma f_neg_inv : f (-1 / 3) = 1 / 3 :=
sorry

theorem find_f_five_thrids : f (5 / 3) = 1 / 3 :=
by
  -- application of the conditions stated
  have h1 : f(1 + (5 / 3 - 1)) = f(-(5 / 3 - 1)),
    from f_transformation (5 / 3 - 1),
  have h2 : f(-(5 / 3 - 1)) = -f(5 / 3 - 1),
    from odd_fun (5 / 3 - 1),
  have h3 : 5 / 3 - 1 = 2 / 3,
    by norm_num,
  rw [h3] at h1,
  rw [h2] at h1,
  have h4 : f (-1 / 3) = 1 / 3,
    from f_neg_inv,
  rw [←h4] at h1,
  exact h1,
sorry

end find_f_five_thrids_l599_599544


namespace sum_of_possible_N_values_l599_599613

theorem sum_of_possible_N_values (a b c N : ℕ) (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = 2 * (a + b)) :
  ∃ sum_N : ℕ, sum_N = 672 :=
by
  sorry

end sum_of_possible_N_values_l599_599613


namespace max_f_l599_599468

open Real

variables (x m : ℝ)

-- Definitions of the vectors a and b
def a : ℝ × ℝ := (sqrt 3 * sin x, m + cos x)
def b : ℝ × ℝ := (cos x, -m + cos x)

-- Definition of the function f
def f (x : ℝ) : ℝ := (sqrt 3 * sin x) * (cos x) + (m + cos x) * (-m + cos x)

-- Simplified expression for f(x)
lemma f_simplified (x m : ℝ) : f x = sin (2 * x + π / 6) + 1 / 2 - m ^ 2 := by
  sorry

-- Prove that the maximum value of f(x) in the given range is -5/2 at x = π/6
theorem max_f (x : ℝ) (h1 : -π/6 ≤ x) (h2 : x ≤ π/3) (h3 : ∀ m, f x ≥ -4) :
  ∃ x, f x = -5/2 ∧ x = π/6 :=
by
  sorry

end max_f_l599_599468


namespace multiply_103_97_l599_599009

theorem multiply_103_97 : 103 * 97 = 9991 := 
by
  sorry

end multiply_103_97_l599_599009


namespace trains_meet_in_32_seconds_l599_599956

noncomputable def length_first_train : ℕ := 400
noncomputable def length_second_train : ℕ := 200
noncomputable def initial_distance : ℕ := 200

noncomputable def speed_first_train : ℕ := 15
noncomputable def speed_second_train : ℕ := 10

noncomputable def relative_speed : ℕ := speed_first_train + speed_second_train
noncomputable def total_distance : ℕ := length_first_train + length_second_train + initial_distance
noncomputable def time_to_meet := total_distance / relative_speed

theorem trains_meet_in_32_seconds : time_to_meet = 32 := by
  sorry

end trains_meet_in_32_seconds_l599_599956


namespace quadratic_root_conditions_l599_599938

theorem quadratic_root_conditions (a b : ℝ)
    (h1 : ∃ k : ℝ, ∀ x : ℝ, x^2 + 2 * x + 3 - k = 0)
    (h2 : ∀ α β : ℝ, α * β = 3 - k ∧ k^2 = α * β + 3 * k) : 
    k = 3 := 
sorry

end quadratic_root_conditions_l599_599938


namespace tiling_possible_l599_599878

theorem tiling_possible (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_lt : m < n) : 
  (∃ f : ℕ → ℕ → Prop, (∀ x y, ¬f x y ∨ f x y = 3 ∧ y > x)) ↔ (3 ∣ (n - m) * (n + m)) :=
sorry

end tiling_possible_l599_599878


namespace number_of_JQK_cards_l599_599250

-- Definition of conditions
def total_cards : ℕ := 52
def probability_JQK : ℝ := 0.23076923076923078

-- The number of jacks, queens, and kings in the pack
def num_JQK : ℕ := (probability_JQK * total_cards).toNat

-- The math proof statement
theorem number_of_JQK_cards :
  num_JQK = 12 :=
sorry

end number_of_JQK_cards_l599_599250


namespace no_matrix_triples_second_column_l599_599044

theorem no_matrix_triples_second_column
    (M : Matrix (Fin 2) (Fin 2) ℝ)
    (A : Matrix (Fin 2) (Fin 2) ℝ) :
    (∀ a b c d : ℝ, M ⬝ ![![a, b], ![c, d]] = ![![a, 3 * b], ![c, 3 * d]]) →
    M = 0 :=
by
  intros h
  sorry

end no_matrix_triples_second_column_l599_599044


namespace janet_saves_minutes_l599_599375

theorem janet_saves_minutes 
  (key_search_time : ℕ := 8) 
  (complaining_time : ℕ := 3) 
  (days_in_week : ℕ := 7) : 
  (key_search_time + complaining_time) * days_in_week = 77 := 
by
  sorry

end janet_saves_minutes_l599_599375


namespace andrei_monthly_spending_l599_599337

noncomputable def original_price := 50
noncomputable def price_increase := 0.10
noncomputable def discount := 0.10
noncomputable def kg_per_month := 2

def new_price := original_price + original_price * price_increase
def discounted_price := new_price - new_price * discount
def monthly_spending := discounted_price * kg_per_month

theorem andrei_monthly_spending : monthly_spending = 99 := by
  sorry

end andrei_monthly_spending_l599_599337


namespace Hari_investment_contribution_l599_599192

noncomputable def Praveen_investment : ℕ := 3780
noncomputable def Praveen_time : ℕ := 12
noncomputable def Hari_time : ℕ := 7
noncomputable def profit_ratio : ℚ := 2 / 3

theorem Hari_investment_contribution :
  ∃ H : ℕ, (Praveen_investment * Praveen_time) / (H * Hari_time) = (2 : ℕ) / 3 ∧ H = 9720 :=
by
  sorry

end Hari_investment_contribution_l599_599192


namespace sum_c_n_lt_2_l599_599776

noncomputable def a (n : ℕ) : ℕ := n
noncomputable def b (n : ℕ) : ℕ := 2^n
noncomputable def c (n : ℕ) : ℚ := n / 2^n
noncomputable def S (n : ℕ) : ℚ := (Finset.range n).sum (λ i, c (i + 1))

theorem sum_c_n_lt_2 (n : ℕ) : S n < 2 := by
  sorry

end sum_c_n_lt_2_l599_599776


namespace cube_root_floor_equality_l599_599095

theorem cube_root_floor_equality (n : ℕ) (hn : 0 < n) :
  (⟨nat.floor (real.cbrt (7 * (n : ℝ) + 2))⟩ : ℤ) =
  (⟨nat.floor (real.cbrt (7 * (n : ℝ) + 3))⟩ : ℤ) := sorry

end cube_root_floor_equality_l599_599095


namespace tourist_turns_l599_599676

theorem tourist_turns (blocks : Fin 16 → Set Point) (squares : Fin 15 → Point)
  (streets : List (Set Point)) (tour: List Point) :
  (∀ i j, i ≠ j → squares i ≠ squares j) →
  (∀ i, ∃ b, squares i ∈ b ∧ b ∈ blocks) →
  (∀ i, ∃ s, {squares i, squares (i + 1 mod 15)} ⊆ s ∧ s ∈ streets) →
  tour.head = squares 0 ∧ tour.last = squares 15 →
  (∀ i, tour.nth i ≠ tour.nth (i + 1 mod tour.length)) →
  ∃ n, 4 ≤ n ∧ ∀ i ≤ n, 
    ∃ blockBorder, tour.head ∈ blockBorder ∧ angles.turn_angle tour.head = 120 :=
sorry

end tourist_turns_l599_599676


namespace range_of_k_find_k_value_l599_599936

open Real

noncomputable def quadratic_eq_has_real_roots (a b c : ℝ) (disc : ℝ) : Prop :=
  disc > 0

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

variables {a : ℝ} {b : ℝ} {c : ℝ} {k : ℝ}
variables (alpha beta : ℝ)

-- Given conditions
def quadratic_eq_with_k := (a = 1) ∧ (b = 2) ∧ (c = 3 - k)
def two_distinct_real_roots := quadratic_eq_has_real_roots 1 2 (3 - k) (discriminant 1 2 (3 - k))
def product_of_roots := alpha * beta = 3 - k
def given_condition := k^2 = alpha * beta + 3 * k

-- Proofs to be done
theorem range_of_k : quadratic_eq_with_k k → two_distinct_real_roots k → k > 2 :=
by
  intro h1 h2
  sorry

theorem find_k_value : quadratic_eq_with_k k → k > 2 → given_condition k alpha beta → k = 3 :=
by
  intro h1 h2 h3
  sorry

end range_of_k_find_k_value_l599_599936


namespace shopkeeper_bananas_l599_599694

theorem shopkeeper_bananas (B : ℕ) (h1 : 600 + B > 0)
  (h2 : 0.85 * 600 + 0.92 * B = 0.878 * (600 + B)) : B = 400 :=
by
  sorry

end shopkeeper_bananas_l599_599694


namespace find_number_l599_599743

theorem find_number (x : ℕ) (h : (18 / 100) * x = 90) : x = 500 :=
sorry

end find_number_l599_599743


namespace sum_intervals_length_eq_l599_599050

def floor (x : ℝ) : ℤ := Int.floor x

def f (x : ℝ) : ℝ := (floor x : ℝ) * (2024^(x - (floor x)) - 1)

theorem sum_intervals_length_eq :
  ∑ k in (finset.range 2023).image (λ k, k + 1), (Real.log (2025 : ℝ) / (Real.log (2024 : ℝ))) = Real.log (2025 : ℝ) / (Real.log (2024 : ℝ)) :=
sorry

end sum_intervals_length_eq_l599_599050


namespace fraction_of_jenny_bounce_distance_l599_599520

-- Definitions for the problem conditions
def jenny_initial_distance := 18
def jenny_bounce_fraction (f : ℚ) : ℚ := 18 * f
def jenny_total_distance (f : ℚ) : ℚ := jenny_initial_distance + jenny_bounce_fraction f

def mark_initial_distance := 15
def mark_bounce_distance := 2 * mark_initial_distance
def mark_total_distance : ℚ := mark_initial_distance + mark_bounce_distance

def distance_difference := 21

-- The theorem to prove
theorem fraction_of_jenny_bounce_distance (f : ℚ) :
  mark_total_distance = jenny_total_distance f + distance_difference →
  f = 1 / 3 :=
by
  sorry

end fraction_of_jenny_bounce_distance_l599_599520


namespace julia_mark_meet_l599_599526

-- Define the conditions
def julia_departure_time : ℕ := 7 * 60 + 45  -- 7:45 AM in minutes
def mark_departure_time : ℕ := 8 * 60 + 10  -- 8:10 AM in minutes
def route_length : ℝ := 74.0  -- 74 miles
def julia_speed : ℝ := 10.0  -- miles per hour
def mark_speed : ℝ := 14.0  -- miles per hour
def julia_rest_stop_distance : ℝ := 30.0  -- 30 miles from Town A
def julia_rest_duration : ℝ := 0.5  -- 30 minutes in hours

-- Define the meeting time variable
noncomputable def meeting_time : ℕ := 12 * 60 + 32  -- 12:32 PM in minutes

-- The proof that Julia and Mark meet at 12:32 PM
theorem julia_mark_meet :
  let x := (meeting_time - julia_departure_time) / 60.0 in
  let julia_first_leg_time := julia_rest_stop_distance / julia_speed in
  let julia_total_time := x - julia_first_leg_time - julia_rest_duration in
  let julia_distance := julia_speed * julia_total_time in
  let mark_start_delay := (mark_departure_time - julia_departure_time) / 60.0 in
  let mark_total_time := x - mark_start_delay in
  let mark_distance := mark_speed * mark_total_time in
  julia_distance + julia_rest_stop_distance + mark_distance = route_length :=
sorry

end julia_mark_meet_l599_599526


namespace exists_prime_period_l599_599887

variable (f : ℝ → ℝ) (T : ℝ) (h1 : ∀ x, f(x + T) = f x) (h2 : ∀ x, f(x + 1) = f x) (h3 : 0 < T) (h4 : T < 1) (h5 : ∃ m n \, : ℕ, T = n / m ∧ Nat.gcd m n = 1)

theorem exists_prime_period : ∃ p : ℕ, p.prime ∧ (∀ x, f(x + 1 / p) = f x) := sorry

end exists_prime_period_l599_599887


namespace alex_basketball_points_l599_599133

theorem alex_basketball_points (f t s : ℕ) 
  (h : f + t + s = 40) 
  (points_scored : ℝ := 0.8 * f + 0.3 * t + s) :
  points_scored = 28 :=
sorry

end alex_basketball_points_l599_599133


namespace total_surface_area_of_solid_l599_599635

-- Definitions of the solid's dimensions and conditions
def length : ℕ := 4
def width : ℕ := 3
def height : ℕ := 1
def additional_cubes : ℕ := 2
def total_cubes : ℕ := 12

-- Calculation of surface area and theorem statement
theorem total_surface_area_of_solid : 
    (2 * (length * height + additional_cubes)) + 
    ((length * width) + (length * width - additional_cubes)) + 
    (2 * (width * height)) = 42 :=
by sorry

end total_surface_area_of_solid_l599_599635


namespace f_zero_eq_zero_l599_599425

-- Define the problem conditions
variable {f : ℝ → ℝ}
variables (h_odd : ∀ x : ℝ, f (-x) = -f (x))
variables (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
variables (h_eq : ∀ x : ℝ, f (1 - x) - f (1 + x) + 2 * x = 0)
variables (h_mono : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ 1 → f x₁ ≤ f x₂)

-- State the theorem
theorem f_zero_eq_zero : f 0 = 0 :=
by sorry

end f_zero_eq_zero_l599_599425


namespace particle_acceleration_at_pi_over_6_l599_599690

def velocity (x : ℝ) : ℝ := 2 + Real.sin x

def acceleration_at (x v_prime: ℝ) := v_prime * velocity x

theorem particle_acceleration_at_pi_over_6 :
  let v_prime := Real.cos (Real.pi / 6)
  acceleration_at (Real.pi / 6) v_prime = (5 * Real.sqrt 3) / 4 :=
by
  let x := Real.pi / 6
  have v_prime : ℝ := Real.cos x
  sorry

end particle_acceleration_at_pi_over_6_l599_599690


namespace probability_prime_and_multiple_of_11_l599_599588

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_multiple_of_11 (n : ℕ) : Prop := n % 11 = 0

theorem probability_prime_and_multiple_of_11 : 
    (∑ n in Finset.range 70.filter (λ n, is_prime (n + 1) ∧ is_multiple_of_11 (n + 1)), 1) / 70 = 1 / 70 :=
by
  sorry

end probability_prime_and_multiple_of_11_l599_599588


namespace central_symmetry_composition_translation_l599_599578

variable {Point Vector : Type} [AddCommGroup Vector] [Module ℝ Vector]

structure CentralSymmetry (Center Point : Point) :=
(map : Point → Point)

-- Given conditions:
variables (O₁ O₂ A A₁ A₂ : Point)
variables (v₁ v₂ v₃ v₄ v₅ : Vector)
variables [CentralSymmetry O₁ A]
variables [CentralSymmetry O₂ A₁]

-- Vector relations as hypotheses
axiom h1 : v₁ = v₅ -- ⟨AO₁⟩ = ⟨O₁A₁⟩
axiom h2 : v₂ = v₄ -- ⟨A₁O₂⟩ = ⟨O₂A₂⟩
axiom h3 : v₁ + v₅ + v₂ + v₄ = v₃ -- ⟨AO₁⟩ + ⟨O₁A₁⟩ + ⟨A₁O₂⟩ + ⟨O₂A₂⟩ = ⟨AA₂⟩
axiom h4 : v₅ + v₂ = v₁ -- ⟨O₁A₁⟩ + ⟨A₁O₂⟩ = ⟨O₁O₂⟧ = ½ ⟨AA₂⟩

-- Theorem to prove
theorem central_symmetry_composition_translation :
  ∃ t : Vector, t = 2 • (v₁ : Vector) :=
by
  -- Proof goes here
  sorry

end central_symmetry_composition_translation_l599_599578


namespace total_duration_in_seconds_l599_599695

theorem total_duration_in_seconds :
  let hours_in_seconds := 2 * 3600
  let minutes_in_seconds := 45 * 60
  let extra_seconds := 30
  hours_in_seconds + minutes_in_seconds + extra_seconds = 9930 := by
  sorry

end total_duration_in_seconds_l599_599695


namespace solve_trig_eq_l599_599026

noncomputable def arccos (x : ℝ) : ℝ := sorry

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  -3 * (Real.cos x) ^ 2 + 5 * (Real.sin x) + 1 = 0 ↔
  (x = Real.arcsin (1 / 3) + 2 * k * Real.pi ∨ x = Real.pi - Real.arcsin (1 / 3) + 2 * k * Real.pi) :=
sorry

end solve_trig_eq_l599_599026


namespace latticePointsCount_l599_599862

-- Define what a lattice point is and the condition
def isLatticePoint (x y z : ℤ) : Prop := x^2 + y^2 + z^2 = 16

-- The theorem stating the number of such lattice points
theorem latticePointsCount :
  (Finset.univ : Finset (ℤ × ℤ × ℤ)).filter (λ (p : ℤ × ℤ × ℤ), isLatticePoint p.1 p.2 p.3).card = 50 :=
  sorry

end latticePointsCount_l599_599862


namespace problem_proof_l599_599467

variables {m n : ℝ}

-- Line definitions
def l1 (m n x y : ℝ) : Prop := m * x + 8 * y + n = 0
def l2 (m x y : ℝ) : Prop := 2 * x + m * y - 1 = 0

-- Conditions
def intersects_at (m n : ℝ) : Prop :=
  l1 m n m (-1) ∧ l2 m m (-1)

def parallel (m n : ℝ) : Prop :=
  (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2)

def perpendicular (m n : ℝ) : Prop :=
  m = 0 ∧ n = 8

theorem problem_proof :
  intersects_at m n → (m = 1 ∧ n = 7) ∧
  parallel m n → (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2) ∧
  perpendicular m n → (m = 0 ∧ n = 8) :=
by
  sorry

end problem_proof_l599_599467


namespace measure_of_angle_A_l599_599340

theorem measure_of_angle_A
    (A B : ℝ)
    (h1 : A + B = 90)
    (h2 : A = 3 * B) :
    A = 67.5 :=
by
  sorry

end measure_of_angle_A_l599_599340


namespace janet_saves_time_l599_599373

theorem janet_saves_time (looking_time_per_day : ℕ := 8) (complaining_time_per_day : ℕ := 3) (days_per_week : ℕ := 7) :
  (looking_time_per_day + complaining_time_per_day) * days_per_week = 77 := 
sorry

end janet_saves_time_l599_599373


namespace length_of_AB_l599_599755

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

-- Define the line perpendicular to the x-axis passing through the right focus of the ellipse
def line_perpendicular_y_axis_through_focus (y : ℝ) : Prop := true

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (3, 0)

-- Statement to prove the length of the line segment AB
theorem length_of_AB : 
  ∃ A B : ℝ × ℝ, 
  (ellipse A.1 A.2 ∧ ellipse B.1 B.2) ∧ 
  (A.1 = 3 ∧ B.1 = 3) ∧
  (|A.2 - B.2| = 2 * 16 / 5) :=
sorry

end length_of_AB_l599_599755


namespace cookies_baked_yesterday_eq_435_l599_599823

structure BakingSituation (total_cookies baked_this_morning : ℕ)
  where
    baked_yesterday : ℕ

theorem cookies_baked_yesterday_eq_435 (total_cookies : ℕ) (baked_this_morning : ℕ) :
  total_cookies = 574 →
  baked_this_morning = 139 →
  ∃ (baked_yesterday : ℕ), baked_yesterday = 435 :=
by
  intros h₁ h₂
  refine ⟨total_cookies - baked_this_morning, _⟩
  rw [h₁, h₂]
  norm_num
  sorry

end cookies_baked_yesterday_eq_435_l599_599823


namespace nonnegative_integers_existence_l599_599114

open Classical

theorem nonnegative_integers_existence (x y : ℕ) : 
  (∃ (a b c d : ℕ), x = a + 2 * b + 3 * c + 7 * d ∧ y = b + 2 * c + 5 * d) ↔ (5 * x ≥ 7 * y) :=
by
  sorry

end nonnegative_integers_existence_l599_599114


namespace largest_four_digit_distinct_no_swap_smaller_l599_599043

/-- A helper function to check if a number is four digits and has all distinct digits -/
def is_four_digit_distinct (n : ℕ) : Prop :=
  let digits := (n % 10, (n / 10) % 10, (n / 100) % 10, n / 1000)
  1000 ≤ n ∧ n ≤ 9999 ∧
  (digits.1 ≠ digits.2) ∧ (digits.1 ≠ digits.3) ∧ (digits.1 ≠ digits.4) ∧
  (digits.2 ≠ digits.3) ∧ (digits.2 ≠ digits.4) ∧
  (digits.3 ≠ digits.4)

/-- A helper function to check if swapping any two digits creates a smaller number -/
def no_smaller_by_swap (n : ℕ) : Prop :=
  let digits := (n % 10, (n / 10) % 10, (n / 100) % 10, n / 1000)
  ∀ i j : ℕ, i ≠ j → swap(digits, i, j) > n

/-- The mathematically equivalent Lean 4 proof statement -/
theorem largest_four_digit_distinct_no_swap_smaller : 
  ∃ n : ℕ, is_four_digit_distinct n ∧ no_smaller_by_swap n ∧ n = 7089 := by
  sorry

end largest_four_digit_distinct_no_swap_smaller_l599_599043


namespace binary_division_remainder_l599_599271

theorem binary_division_remainder : ∀ n : ℕ, n = 0b101110011100 → n % 4 = 0 :=
by
  intro n h
  rw h
  sorry

end binary_division_remainder_l599_599271


namespace solve_r_l599_599203

variable (r : ℝ)

theorem solve_r : (r + 3) / (r - 2) = (r - 1) / (r + 1) → r = -1/7 := by
  sorry

end solve_r_l599_599203


namespace find_finite_sets_l599_599035

def Point := ℝ × ℝ

-- Condition 1: No three points of the set are collinear
def no_three_points_collinear (M : set Point) : Prop :=
  ∀ (a b c : Point), a ∈ M ∧ b ∈ M ∧ c ∈ M → geom_no_three_collinear a b c

-- Identify if three distinct points are not collinear
def geom_no_three_collinear (a b c : Point) : Prop := 
  (a.1 - b.1) * (b.2 - c.2) ≠ (b.1 - c.1) * (a.2 - b.2)

-- Find orthocenter of the triangle formed by three points
def orthocenter (a b c : Point) : Point :=
  sorry -- Placeholder implementation to be replaced by the actual computation

-- Condition 2: Orthocenter of any triangle in the set is also in the set
def orthocenter_in_set (M : set Point) : Prop :=
  ∀ (a b c : Point), a ∈ M ∧ b ∈ M ∧ c ∈ M → orthocenter a b c ∈ M

-- Theorem: Characterize the finite sets ensuring the conditions
theorem find_finite_sets (M : set Point) :
  finite M →
  no_three_points_collinear M → 
  orthocenter_in_set M →
  (∃ A B C : Point, M = {A, B, C, orthocenter A B C} ∨ ∃ A B C, is_right_angle_triangle A B C ∧ M = {A, B, C} ∨ ∃ A B C D, is_square A B C D ∧ M = {A, B, C, D}) :=
sorry

def is_right_angle_triangle (A B C : Point) : Prop :=
  (A = orthocenter B C (0, 0)) ∨ (B = orthocenter A C (0, 0)) ∨ (C = orthocenter A B (0, 0))

def is_square (A B C D : Point) : Prop :=
  sorry -- Placeholder: Define conditions for four points to form a square

-- Add remaining necessary definitions and delete 'sorry' comments with appropriate code.

end find_finite_sets_l599_599035


namespace correct_conclusion_only_four_l599_599333

-- Define the types for points, lines, and planes
variables {Point Line Plane : Type}

-- Define relations and predicates
variables (α β : Plane) (a b : Line) (P : Point)
variables (line_subset_plane : Line → Plane → Prop)
variables (line_parallel_plane : Line → Plane → Prop)
variables (line_parallel_line : Line → Line → Prop)
variables (plane_parallel_plane : Plane → Plane → Prop)
variables (point_in_plane : Point → Plane → Prop)
variables (point_in_line : Point → Line → Prop)

-- Define the given conditions and conclusions as hypotheses
hypothesis h1 : ¬line_subset_plane a α → line_parallel_plane a α
hypothesis h2 : line_parallel_plane a α → line_subset_plane b α → line_parallel_line a b
hypothesis h3 : plane_parallel_plane α β → line_subset_plane a α → line_subset_plane b β → line_parallel_line a b
hypothesis h4 : plane_parallel_plane α β → point_in_plane P α → line_parallel_plane a β → point_in_line P a → line_subset_plane a α

-- Prove that the only correct conclusion is ④
theorem correct_conclusion_only_four :
  ¬ ((¬line_subset_plane a α → line_parallel_plane a α) ∧
    (line_parallel_plane a α → line_subset_plane b α → line_parallel_line a b) ∧
    (plane_parallel_plane α β → line_subset_plane a α → line_subset_plane b β → line_parallel_line a b) ∧
    (plane_parallel_plane α β → point_in_plane P α → line_parallel_plane a β → point_in_line P a → line_subset_plane a α)) ∧
  plane_parallel_plane α β → point_in_plane P α → line_parallel_plane a β → point_in_line P a → line_subset_plane a α :=
by sorry

end correct_conclusion_only_four_l599_599333


namespace simplify_expression_l599_599028

theorem simplify_expression :
  8^(-1/3 : ℝ) + Real.logBase 3 (Real.tan (210 * Real.pi / 180)) = 0 := by
  sorry

end simplify_expression_l599_599028


namespace polynomial_consecutive_integers_l599_599623

theorem polynomial_consecutive_integers (a : ℤ) (c : ℤ) (P : ℤ → ℤ)
  (hP : ∀ x : ℤ, P x = 2 * x ^ 3 - 30 * x ^ 2 + c * x)
  (h_consecutive : ∃ a : ℤ, P (a - 1) + 1 = P a ∧ P a = P (a + 1) - 1) :
  a = 5 ∧ c = 149 :=
by
  sorry

end polynomial_consecutive_integers_l599_599623


namespace find_m_l599_599446

theorem find_m (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - 3 * x + m ≥ 2 * x^2 - 4 * x)
  ↔ m = 2 :=
begin
  sorry
end

end find_m_l599_599446


namespace inequality_implies_difference_lt_zero_l599_599489

theorem inequality_implies_difference_lt_zero (x y : ℝ) (h : 2023^x + 2024^(-y) < 2023^y + 2024^(-x)) :
  x - y < 0 := by
  sorry

end inequality_implies_difference_lt_zero_l599_599489


namespace factor_expression_l599_599034

noncomputable def numerator (a b c : ℝ) : ℝ := 
(|a^2 + b^2|^3 + |b^2 + c^2|^3 + |c^2 + a^2|^3)

noncomputable def denominator (a b c : ℝ) : ℝ := 
(|a + b|^3 + |b + c|^3 + |c + a|^3)

theorem factor_expression (a b c : ℝ) : 
  (denominator a b c) ≠ 0 → 
  (numerator a b c) / (denominator a b c) = 1 :=
by
  sorry

end factor_expression_l599_599034


namespace LCM_of_two_numbers_l599_599256

theorem LCM_of_two_numbers (a b : ℕ) (h_hcf : Nat.gcd a b = 11) (h_product : a * b = 1991) : Nat.lcm a b = 181 :=
by
  sorry

end LCM_of_two_numbers_l599_599256


namespace find_a_l599_599080

theorem find_a (a : ℝ) :
  (∃ x y : ℝ, x = ax + 3 ∧ (x - 1)^2 + (y - 2)^2 = 4) →
  (2 * sqrt 3 = 2 * sqrt (4 - (abs (a + 1) / sqrt (a^2 + 1))^2)) →
  a = 0 :=
by 
  sorry

end find_a_l599_599080


namespace equation_of_circle_equation_of_line_l599_599778

-- Define the points A and B
def A : ℝ × ℝ := (-4, -3)
def B : ℝ × ℝ := (2, 9)

-- Define the midpoint C of AB
noncomputable def C : ℝ × ℝ := ((-4 + 2) / 2, (-3 + 9) / 2)

-- Define the radius of the circle
noncomputable def radius : ℝ := real.sqrt ((-4 + 1) ^ 2 + (-3 - 3) ^ 2)

-- Define the point P
def P : ℝ × ℝ := (0, 2)

-- Prove the equation of the circle C
theorem equation_of_circle : ∀ x y : ℝ, (x + 1) ^ 2 + (y - 3) ^ 2 = 45 :=
by
  sorry -- proof

-- Prove the equation of the line l₀ where the chord with P as its midpoint lies
theorem equation_of_line : ∀ x y : ℝ, x - y + 2 = 0 :=
by
  sorry -- proof

end equation_of_circle_equation_of_line_l599_599778


namespace find_f_five_thirds_l599_599536

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = - f x
def functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (-x)
def specific_value (f : ℝ → ℝ) := f (-1/3) = 1/3

theorem find_f_five_thirds
  (hf_odd : odd_function f)
  (hf_fe : functional_equation f)
  (hf_value : specific_value f) :
  f (5 / 3) = 1 / 3 := by
  sorry

end find_f_five_thirds_l599_599536


namespace number_of_rounds_l599_599134

def initial_tokens_A := 16
def initial_tokens_B := 15
def initial_tokens_C := 14
def initial_tokens_D := 13

def round_rules (tokens : ℕ × ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let (a, b, c, d) := tokens in
  let max_tokens := max a (max b (max c d)) in
  if max_tokens = a then (a - 4, b + 1, c + 1, d + 1)
  else if max_tokens = b then (a + 1, b - 4, c + 1, d + 1)
  else if max_tokens = c then (a + 1, b + 1, c - 4, d + 1)
  else (a + 1, b + 1, c + 1, d - 4)

def play_game (tokens : ℕ × ℕ × ℕ × ℕ) : ℕ :=
  if tokens.1 = 0 ∨ tokens.2 = 0 ∨ tokens.3 = 0 ∨ tokens.4 = 0 then 0
  else 1 + play_game (round_rules tokens)

theorem number_of_rounds (tokens : ℕ × ℕ × ℕ × ℕ) :
  tokens = (initial_tokens_A, initial_tokens_B, initial_tokens_C, initial_tokens_D) →
  play_game tokens = 53 :=
by
  intros h
  rw h
  -- Sorry, placeholder for the actual proof
  sorry

end number_of_rounds_l599_599134


namespace parking_time_l599_599235

theorem parking_time
  (cost_first_hour : ℝ)
  (cost_additional_half_hour : ℝ)
  (total_paid : ℝ)
  (h1 : cost_first_hour = 2.5)
  (h2 : cost_additional_half_hour = 2.5)
  (h3 : total_paid = 12.5) :
  let time_in_hours := 1 + (total_paid - cost_first_hour) / cost_additional_half_hour * 0.5
  in time_in_hours = 3 :=
by
  sorry

end parking_time_l599_599235


namespace cheryl_mms_eaten_l599_599358

variable (initial_mms : ℕ) (mms_after_dinner : ℕ) (mms_given_to_sister : ℕ) (total_mms_after_lunch : ℕ)

theorem cheryl_mms_eaten (h1 : initial_mms = 25)
                         (h2 : mms_after_dinner = 5)
                         (h3 : mms_given_to_sister = 13)
                         (h4 : total_mms_after_lunch = initial_mms - mms_after_dinner - mms_given_to_sister) :
                         total_mms_after_lunch = 7 :=
by sorry

end cheryl_mms_eaten_l599_599358


namespace range_of_a_l599_599891

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_a (h_odd : ∀ x, f (-x) = -f x) 
  (h_period : ∀ x, f (x + 3) = f x)
  (h1 : f 1 > 1) 
  (h2018 : f 2018 = (a : ℝ) ^ 2 - 5) : 
  -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l599_599891


namespace fly_total_distance_l599_599684

noncomputable def total_distance_traveled (r : ℝ) (d3 : ℝ) : ℝ :=
  let d1 := 2 * r
  let d2 := Real.sqrt (d1^2 - d3^2)
  d1 + d2 + d3

theorem fly_total_distance (r : ℝ) (h_r : r = 60) (d3 : ℝ) (h_d3 : d3 = 90) :
  total_distance_traveled r d3 = 289.37 :=
by
  rw [h_r, h_d3]
  simp [total_distance_traveled]
  sorry

end fly_total_distance_l599_599684


namespace train_passes_jogger_in_approx_24_75_seconds_l599_599687

noncomputable def time_to_pass_jogger (jogger_speed train_speed : ℝ) (initial_lead train_length : ℝ) : ℝ :=
  let relative_speed := (train_speed - jogger_speed) * 1000 / 3600  -- km/hr to m/s
  let total_distance := initial_lead + train_length  -- Distance in meters
  total_distance / relative_speed  -- Time in seconds

theorem train_passes_jogger_in_approx_24_75_seconds :
  time_to_pass_jogger 12 60 180 150 ≈ 24.75 := sorry

end train_passes_jogger_in_approx_24_75_seconds_l599_599687


namespace min_value_func_y_l599_599082

noncomputable def geometric_sum (t : ℝ) (n : ℕ) : ℝ :=
  t * 3^(n-1) - (1 / 3)

noncomputable def func_y (x t : ℝ) : ℝ :=
  (x + 2) * (x + 10) / (x + t)

theorem min_value_func_y :
  ∀ (t : ℝ), (∀ n : ℕ, geometric_sum t n = (1) → (∀ x > 0, func_y x t ≥ 16)) :=
  sorry

end min_value_func_y_l599_599082


namespace transformations_map_onto_self_l599_599730

/-- Define the transformations -/
def T1 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for a 90 degree rotation around the center of a square
  sorry

def T2 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for a translation parallel to line ℓ
  sorry

def T3 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for reflection across line ℓ
  sorry

def T4 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for reflection across a line perpendicular to line ℓ
  sorry

/-- Define the pattern -/
def pattern (p : ℝ × ℝ) : Type :=
  -- Representation of alternating right triangles and squares along line ℓ
  sorry

/-- The main theorem:
    Prove that there are exactly 3 transformations (T1, T2, T3) that will map the pattern onto itself. -/
theorem transformations_map_onto_self : (∃ pattern : ℝ × ℝ → Type,
  (T1 pattern = pattern) ∧
  (T2 pattern = pattern) ∧
  (T3 pattern = pattern) ∧
  ¬ (T4 pattern = pattern)) → (3 = 3) :=
by
  sorry

end transformations_map_onto_self_l599_599730


namespace sqrt_expression_value_l599_599174

variable (a b : ℝ) 

theorem sqrt_expression_value (ha : a ≠ 0) (hb : b ≠ 0) (ha_neg : a < 0) :
  Real.sqrt (-a^3) * Real.sqrt ((-b)^4) = -a * |b| * Real.sqrt (-a) := by
  sorry

end sqrt_expression_value_l599_599174


namespace work_rates_l599_599280

-- Given conditions
variables (W : ℝ) (Rx Ry Rxy : ℝ)

-- Defining the rates
def Rx := W / 10
def Ry := W / y
def Rxy := W / 6

-- The main theorem to prove
theorem work_rates (W : ℝ) (H : W ≠ 0) : (1 / 6 : ℝ) = (1 / 10 : ℝ) + (1 / (y : ℝ)) → y = 15 :=
by
  sorry

end work_rates_l599_599280


namespace log_base_calculate_l599_599561

theorem log_base_calculate (a : ℝ) (h : log 2 a = 4) : log a 2 = 1 / 4 :=
by
  sorry

end log_base_calculate_l599_599561


namespace class_average_l599_599834

theorem class_average (p1 p2 p3 avg1 avg2 avg3 overall_avg : ℕ) 
  (h1 : p1 = 45) 
  (h2 : p2 = 50) 
  (h3 : p3 = 100 - p1 - p2) 
  (havg1 : avg1 = 95) 
  (havg2 : avg2 = 78) 
  (havg3 : avg3 = 60) 
  (hoverall : overall_avg = (p1 * avg1 + p2 * avg2 + p3 * avg3) / 100) : 
  overall_avg = 85 :=
by
  sorry

end class_average_l599_599834


namespace gabby_needs_more_money_l599_599413

theorem gabby_needs_more_money (cost_saved : ℕ) (initial_saved : ℕ) (additional_money : ℕ) (cost_remaining : ℕ) :
  cost_saved = 65 → initial_saved = 35 → additional_money = 20 → cost_remaining = (cost_saved - initial_saved) - additional_money → cost_remaining = 10 :=
by
  intros h_cost_saved h_initial_saved h_additional_money h_cost_remaining
  simp [h_cost_saved, h_initial_saved, h_additional_money] at h_cost_remaining
  exact h_cost_remaining

end gabby_needs_more_money_l599_599413


namespace compute_g_l599_599207

-- Define the assumptions and state the theorem
theorem compute_g : 
  (∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x) → 
  g (-2) = 8 → 
  g (-1002) = 8016 :=
by 
  -- Skipping the proof
  sorry

end compute_g_l599_599207


namespace rectangle_area_l599_599610

theorem rectangle_area (b : ℝ) :
  let side_of_square := real.sqrt 2500 in
  let radius_of_circle := side_of_square in
  let length_of_rectangle := (2 / 5) * radius_of_circle in
  20 * b = length_of_rectangle * b := 
by
  sorry

end rectangle_area_l599_599610


namespace find_f_five_thrids_l599_599547

noncomputable def f : ℝ → ℝ := sorry

lemma odd_fun (x : ℝ) : f (-x) = -f(x) :=
sorry

lemma f_transformation (x : ℝ) : f (1 + x) = f (-x) :=
sorry

lemma f_neg_inv : f (-1 / 3) = 1 / 3 :=
sorry

theorem find_f_five_thrids : f (5 / 3) = 1 / 3 :=
by
  -- application of the conditions stated
  have h1 : f(1 + (5 / 3 - 1)) = f(-(5 / 3 - 1)),
    from f_transformation (5 / 3 - 1),
  have h2 : f(-(5 / 3 - 1)) = -f(5 / 3 - 1),
    from odd_fun (5 / 3 - 1),
  have h3 : 5 / 3 - 1 = 2 / 3,
    by norm_num,
  rw [h3] at h1,
  rw [h2] at h1,
  have h4 : f (-1 / 3) = 1 / 3,
    from f_neg_inv,
  rw [←h4] at h1,
  exact h1,
sorry

end find_f_five_thrids_l599_599547


namespace katie_initial_candies_l599_599157

theorem katie_initial_candies (K : ℕ) (h1 : K + 23 - 8 = 23) : K = 8 :=
sorry

end katie_initial_candies_l599_599157


namespace largest_unique_digit_number_is_7089_l599_599040

/--
Prove that the largest four-digit number where all digits are distinct,
and no two digits can be swapped to form a smaller number, is 7089.
-/
theorem largest_unique_digit_number_is_7089 :
  ∃ (n : ℕ), (1000 ≤ n ∧ n < 10000) ∧ 
             (∀ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l →
              i*1000 + j*100 + k*10 + l = n) ∧ 
             (∀ a b, (a ≠ b) → ∀ c d, (c ≠ d) → 
              (a ≠ c) → (a ≠ d) → (b ≠ c) → (b ≠ d) →
              (a*1000 + b*100 + c*10 + d > n ∧ a*1000 + b*100 + c*10 + d ≤ 9876)) :=
  ∃ (n : ℕ), n = 7089

end largest_unique_digit_number_is_7089_l599_599040


namespace max_value_of_f_l599_599506

def op (a b : ℝ) : ℝ :=
if a ≥ b then a else b

def f (x : ℝ) : ℝ :=
(op 1 x) * x - (op 2 x)

theorem max_value_of_f : ∃ x ∈ set.Icc (-2 : ℝ) 2, f x = 2 :=
by
  use 2
  split
  repeat {norm_num, linarith}
  sorry

end max_value_of_f_l599_599506


namespace num_valid_arrangements_correct_l599_599261

noncomputable def num_valid_arrangements (n : ℕ) : ℕ :=
  let u : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | k + 2 => u k + u (k + 1)
  in u (n + 1)

theorem num_valid_arrangements_correct (n : ℕ) : 
  ∃ u : ℕ → ℕ, 
  (u 0 = 1) ∧
  (u 1 = 1) ∧
  (∀ k : ℕ, u (k + 2) = u k + u (k + 1)) ∧
  num_valid_arrangements n = u (n + 1) :=
sorry

end num_valid_arrangements_correct_l599_599261


namespace tourist_can_reach_set_of_points_l599_599254

structure TouristReachable :=
(point_on_road : ℝ)
(point_on_field : ℝ)

def tourist := { point_on_road : ℝ | abs(point_on_road) ≤ 6 }

def field_perpendicular_to_road (x : ℝ) : set ℝ :=
{ y | abs(y) ≤ 3 - (x / 2) }

def within_one_hour (x y : ℝ) : Prop :=
(x ∈ tourist) ∧ (y ∈ field_perpendicular_to_road x)

def reachable_points : set (ℝ × ℝ) := { p | let x := p.1 in let y := p.2 in within_one_hour x y }

theorem tourist_can_reach_set_of_points :
  ∀ (p : ℝ × ℝ), p ∈ reachable_points ↔ 
  (abs p.1 ≤ 6 ∧ p.2 = 0) ∨
  (p.1 = 0 ∧ abs p.2 ≤ 3) :=
sorry

end tourist_can_reach_set_of_points_l599_599254


namespace trigonometric_identity_proof_l599_599439

noncomputable def P := (-2 : ℝ, 3 : ℝ)
noncomputable def α : ℝ := Real.arctan2 P.2 P.1

theorem trigonometric_identity_proof :
  (cos(π / 2 + α) * sin(π + α)) / (cos(π - α) * sin(3 * π - α)) = (3 / 2) := by
  sorry

end trigonometric_identity_proof_l599_599439


namespace range_of_m_l599_599404

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), x > -5 ∧ x ≤ m + 1) ∧ (∀ x, x > -5 → x ≤ m + 1 → x = -4 ∨ x = -3 ∨ x = -2) →
  (-3 ≤ m ∧ m < -2) :=
sorry

end range_of_m_l599_599404


namespace diagonals_intersect_at_midpoint_l599_599903

/-- Points A and B define opposite vertices of a parallelogram, and C is a third vertex.
    This theorem proves that the intersection point of the diagonals is (8, 3).
-/
theorem diagonals_intersect_at_midpoint 
    (A B C : ℝ × ℝ)
    (A_eq : A = (2, -3))
    (B_eq : B = (14, 9))
    (C_eq : C = (5, 7)) :
    let midpoint := (8, 3)
    in midpoint = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) :=
by
    sorry

end diagonals_intersect_at_midpoint_l599_599903


namespace sequence_proof_l599_599459

noncomputable def a : ℕ → ℤ
| 0       := 4
| 1       := 22
| (n + 2) := 6 * a (n + 1) - a n

def x : ℕ → ℤ := sorry -- Placeholder, as exact sequence of x_n isn't derived in solution steps
def y : ℕ → ℤ := sorry -- Placeholder, as exact sequence of y_n isn't derived in solution steps

theorem sequence_proof :
  (∀ n, ∃ x_n y_n : ℕ, a n = (y_n^2 + 7) / (x_n - y_n)) :=
sorry

end sequence_proof_l599_599459


namespace area_triangle_AF1F2_l599_599287

-- Define the conditions for the theorem
variables {F1 F2 A : Type}
variables [IsFocus F1 (x^2 / 9 + y^2 / 7 = 1)]
variables [IsFocus F2 (x^2 / 9 + y^2 / 7 = 1)]
variables [IsOnEllipse A (x^2 / 9 + y^2 / 7 = 1)]
variables [HasAngle A F1 F2 (Real.pi / 4)] -- 45 degrees in radians

-- Theorem: Area of triangle AF1F2
theorem area_triangle_AF1F2 : area (triangle A F1 F2) = 7 / 2 :=
sorry

end area_triangle_AF1F2_l599_599287


namespace problem_statement_l599_599803

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos (2 * x - π / 3)

theorem problem_statement :
  (∀ k : ℤ, ∃ x : ℝ, (2 * x - π / 3 = π / 2 + k * π) → x = 5 * π / 12 + k * π / 2) ∧
  (¬(∀ k : ℤ, ∀ x : ℝ, (-π / 3 + 2 * k * π ≤ x ∧ x ≤ π / 6 + 2 * k * π))) ∧
  (∀ x : ℝ, f (x + π / 6) = 4 * Real.cos (2 * x)) ∧
  (¬(∀ x : ℝ, f (x - π / 6) = 4 * Real.sin (2 * x))) :=
by
  sorry

end problem_statement_l599_599803


namespace combined_tax_rate_l599_599715

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (Mork_tax_rate Mindy_tax_rate : ℝ)
  (h1 : Mork_tax_rate = 0.4) (h2 : Mindy_tax_rate = 0.3) (h3 : Mindy_income = 4 * Mork_income) :
  ((Mork_tax_rate * Mork_income + Mindy_tax_rate * Mindy_income) / (Mork_income + Mindy_income)) * 100 = 32 :=
by
  sorry

end combined_tax_rate_l599_599715


namespace triangle_dist_BI_l599_599243

noncomputable def triangle_BI (A B C : Type) [EuclideanGeometry] 
  (is_isosceles : IsIsoscelesTriangle A B C)
  (AC_eq_6 : AC = 6)
  (angle_A_right : angle A = π / 2)
  (I_is_incenter : IsIncenter I A B C) : Real :=
  BI = 6 * sqrt 2 - 6

theorem triangle_dist_BI (h₀ : IsIsoscelesTriangle A B C) 
  (h₁ : AC = 6)
  (h₂ : angle A = π / 2)
  (h₃ : IsIncenter I A B C) :
  triangle_BI A B C h₀ h₁ h₂ h₃ :=
sorry

end triangle_dist_BI_l599_599243


namespace gabby_needs_more_money_l599_599412

theorem gabby_needs_more_money (cost_saved : ℕ) (initial_saved : ℕ) (additional_money : ℕ) (cost_remaining : ℕ) :
  cost_saved = 65 → initial_saved = 35 → additional_money = 20 → cost_remaining = (cost_saved - initial_saved) - additional_money → cost_remaining = 10 :=
by
  intros h_cost_saved h_initial_saved h_additional_money h_cost_remaining
  simp [h_cost_saved, h_initial_saved, h_additional_money] at h_cost_remaining
  exact h_cost_remaining

end gabby_needs_more_money_l599_599412


namespace max_rational_sums_1250_l599_599185

noncomputable def max_rational_sums : ℕ :=
  let n := 50
  -- Set of 100 distinct numbers with 50 rational and 50 irrational
  let distinct_numbers := (finset.range 100).image (λ i, if i < 50 then (i : ℚ) else (irrational.of_real (i : ℚ)))
  -- Constraints check
  have h_distinct : distinct_numbers.card = 100 := by sorry
  have h_rational_count : (distinct_numbers.filter is_rational).card = 50 := by sorry
  have h_irrational_count : (distinct_numbers.filter (λ a, ¬is_rational a)).card = 50 := by sorry
  -- Calculate and prove maximum rational sum cells
  let x := 25 -- The optimal split of rational/irrational numbers
  let sum_rational := x * x -- Number of rational sums
  sum_rational

-- Final theorem statement: the maximum number of rational sums in the $50 \times 50$ table
theorem max_rational_sums_1250 : max_rational_sums = 1250 :=
by sorry

end max_rational_sums_1250_l599_599185


namespace tan_condition_then_expression_value_l599_599783

theorem tan_condition_then_expression_value (θ : ℝ) (h : Real.tan θ = 2) :
  (2 * Real.sin θ) / (Real.sin θ + 2 * Real.cos θ) = 1 :=
sorry

end tan_condition_then_expression_value_l599_599783


namespace puppies_adopted_each_day_l599_599316

theorem puppies_adopted_each_day (x : ℕ) (h_total : 2 + 34 = 36) (h_days : 9 * x = 36) : x = 4 :=
by
  rw [add_comm] at h_total -- Rewrite h_total for clarity
  have h_total_puppies : 36 = 36 := h_total
  solve_by_elim
  sorry

end puppies_adopted_each_day_l599_599316


namespace polygon_intersections_l599_599908

-- Define the problem conditions
structure Polygon (n : ℕ) :=
  (sides : ℕ)
  (inscribed_in_circle : Prop)
  (no_vertex_shared : Prop)
  (no_three_sides_intersect : Prop)

-- Polygons with given sides
def P7 := Polygon.mk 7 True True True
def P9 := Polygon.mk 9 True True True
def P11 := Polygon.mk 11 True True True
def P13 := Polygon.mk 13 True True True

-- The theorem statement
theorem polygon_intersections :
  ∀ (P7 P9 P11 P13 : Polygon),
    P7.sides = 7 →
    P9.sides = 9 →
    P11.sides = 11 →
    P13.sides = 13 →
    P7.inscribed_in_circle →
    P9.inscribed_in_circle →
    P11.inscribed_in_circle →
    P13.inscribed_in_circle →
    P7.no_vertex_shared →
    P9.no_vertex_shared →
    P11.no_vertex_shared →
    P13.no_vertex_shared →
    P7.no_three_sides_intersect →
    P9.no_three_sides_intersect →
    P11.no_three_sides_intersect →
    P13.no_three_sides_intersect →
    14 + 14 + 14 + 18 + 18 + 22 = 100 :=
by
  intros
  sorry

end polygon_intersections_l599_599908


namespace point_in_second_quadrant_l599_599502

-- Conditions: x-coordinate is -1 (negative), y-coordinate is m^2 + 1 (positive)
def point_quadrant (m : ℝ) : String :=
  if -1 < 0 ∧ m^2 + 1 > 0 then "Second quadrant" else "Not in the second quadrant"

-- Theorem statement
theorem point_in_second_quadrant (m : ℝ) : point_quadrant m = "Second quadrant" := by
  -- Assume the conditions as true
  sorry

end point_in_second_quadrant_l599_599502


namespace three_times_greater_than_two_l599_599377

theorem three_times_greater_than_two (x : ℝ) : 3 * x - 2 > 0 → 3 * x > 2 :=
by
  sorry

end three_times_greater_than_two_l599_599377


namespace tic_has_winning_strategy_l599_599013
-- Import the entirety of the necessary library.

-- Define the main statement of the problem.
theorem tic_has_winning_strategy : 
  ∃ (a b c : ℚ), (a + b + c = 0) ∧ (∃ A B C : ℚ, T = A * X^2 + B * X + C) ∧
  (∀ T : ℚ[X], (∃ r1 r2 : ℚ, T = a * (X - r1) * (X - r2) ∧ r1 ≠ r2) → Tic wins) :=
sorry

end tic_has_winning_strategy_l599_599013


namespace polynomial_expansion_p_value_l599_599236

theorem polynomial_expansion_p_value :
  ∀ (p q : ℝ), (p + q = 3) → 
  let third_term_ratio := (21 * p^5 * q^2) / (35 * p^4 * q^3) in
  (third_term_ratio = 2) → 
  p = 30 / 13 :=
by
  sorry

end polynomial_expansion_p_value_l599_599236


namespace quadratic_vertex_a_l599_599225

theorem quadratic_vertex_a
  (a b c : ℝ)
  (h1 : ∀ x, (a * x^2 + b * x + c = a * (x - 2)^2 + 5))
  (h2 : a * 0^2 + b * 0 + c = 0) :
  a = -5/4 :=
by
  -- Use the given conditions to outline the proof (proof not provided here as per instruction)
  sorry

end quadratic_vertex_a_l599_599225


namespace median_diff_expectation_le_variance_sqrt_l599_599580

open ProbabilityTheory

variables {Ω : Type*} {ℱ : measurable_space Ω}
  {P : MeasureTheory.ProbabilityMeasure ℱ}
  {ξ : Ω → ℝ}

def median (X : Ω → ℝ) := {m : ℝ | ∀ ε > 0, P({ω | X ω < m - ε}) ≤ 1/2 ∧ P({ω | X ω > m + ε}) ≤ 1/2}

theorem median_diff_expectation_le_variance_sqrt
  (μ : ℝ) (hμ : μ ∈ median ξ) :
  |μ - MeasureTheory.ProbabilityTheory.expectation P ξ| ≤ Real.sqrt (measure_variance P ξ) :=
by sorry

end median_diff_expectation_le_variance_sqrt_l599_599580


namespace range_of_m_l599_599127

theorem range_of_m (m : ℝ) : (∃ x : ℝ, y = (m-2)*x + m ∧ x > 0 ∧ y > 0) ∧ 
                              (∃ x : ℝ, y = (m-2)*x + m ∧ x < 0 ∧ y > 0) ∧ 
                              (∃ x : ℝ, y = (m-2)*x + m ∧ x > 0 ∧ y < 0) ↔ 0 < m ∧ m < 2 :=
by sorry

end range_of_m_l599_599127


namespace height_of_first_podium_l599_599827

noncomputable def height_of_podium_2_cm := 53.0
noncomputable def height_of_podium_2_mm := 7.0
noncomputable def height_on_podium_2_cm := 190.0
noncomputable def height_on_podium_1_cm := 232.0
noncomputable def height_on_podium_1_mm := 5.0

def expected_height_of_podium_1_cm := 96.2

theorem height_of_first_podium :
  let height_podium_2 := height_of_podium_2_cm + height_of_podium_2_mm / 10.0
  let height_podium_1 := height_on_podium_1_cm + height_on_podium_1_mm / 10.0
  let hyeonjoo_height := height_on_podium_2_cm - height_podium_2
  height_podium_1 - hyeonjoo_height = expected_height_of_podium_1_cm :=
by sorry

end height_of_first_podium_l599_599827


namespace distance_probability_l599_599504

theorem distance_probability :
  let speed := 5
  let num_roads := 8
  let total_outcomes := num_roads * (num_roads - 1)
  let favorable_outcomes := num_roads * 3
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = 0.375 :=
by
  sorry

end distance_probability_l599_599504


namespace triangle_problem_l599_599514

variable (A B C a b c : ℝ)

-- Conditions
def cond1 (h1 : a / (b * c) = (Real.cos C) / c + (Real.sin B) / b) : Prop :=
  B = π / 4

def cond2 (h2 : a / (Real.sqrt 2 * b + c) = (Real.sqrt 6 - Real.sqrt 2) / 2) : Prop :=
  A = 5 * π / 12

-- Combining conditions into a single proof problem
theorem triangle_problem 
  (h1 : a / (b * c) = (Real.cos C) / c + (Real.sin B) / b)
  (h2 : a / (Real.sqrt 2 * b + c) = (Real.sqrt 6 - Real.sqrt 2) / 2) : 
  B = π / 4 ∧ A = 5 * π / 12 :=
by exact ⟨cond1 h1, cond2 h2⟩

end triangle_problem_l599_599514


namespace g_decreasing_on_neg_infinity_to_zero_g_min_value_on_neg_infinity_to_neg_one_l599_599452

-- Define the function g(x)
def g (x : ℝ) : ℝ := 1 + 2 / (2^x - 1)

-- (1) Prove that g is a decreasing function on (-∞, 0).
theorem g_decreasing_on_neg_infinity_to_zero : 
  ∀ x₁ x₂ : ℝ, x₁ < 0 → x₂ < 0 → x₁ < x₂ → g x₁ > g x₂ := 
by
  sorry

-- (2) Find the minimum value of g(x) on (-∞, -1].
theorem g_min_value_on_neg_infinity_to_neg_one :
  ∀ x : ℝ, x ≤ -1 → g x ≥ g (-1) ∧ g (-1) = -3 :=
by 
  sorry

end g_decreasing_on_neg_infinity_to_zero_g_min_value_on_neg_infinity_to_neg_one_l599_599452


namespace restore_isosceles_triangle_l599_599197

-- Definition of the existence of an isosceles triangle given specific points as incenter, centroid, and orthocenter.
theorem restore_isosceles_triangle
  (I M H : Point)
  (hIcenter : is_incenter I)
  (hMcenter : is_centroid M)
  (hHcenter : is_orthocenter H) :
  ∃ (A B C : Point), is_triangle A B C ∧ is_isosceles A B C ∧
                     incenter A B C = I ∧ centroid A B C = M ∧ orthocenter A B C = H :=
sorry

end restore_isosceles_triangle_l599_599197


namespace range_of_m_l599_599402

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), x > -5 ∧ x ≤ m + 1) ∧ (∀ x, x > -5 → x ≤ m + 1 → x = -4 ∨ x = -3 ∨ x = -2) →
  (-3 ≤ m ∧ m < -2) :=
sorry

end range_of_m_l599_599402


namespace smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6_l599_599644

theorem smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6 : 
  ∃ n : ℕ, (∃ k : ℕ, n = 60 * k + 1) ∧ n % 9 = 0 ∧ ∀ m : ℕ, (∃ k' : ℕ, m = 60 * k' + 1) ∧ m % 9 = 0 → n ≤ m :=
by
  sorry

end smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6_l599_599644


namespace triangle_angle_bisector_sum_l599_599897

theorem triangle_angle_bisector_sum
  (A B C D : Type) [is_triangle A B C]
  (alpha : ℝ)
  (angle_BAC : ∠ BAC = 2 * alpha)
  (angle_ADC : ∠ ADC = 3 * alpha)
  (angle_ACB : ∠ ACB = 4 * alpha)
  (is_angle_bisector : is_angle_bisector A L B C): 
  BC + CD = AB := 
sorry

end triangle_angle_bisector_sum_l599_599897


namespace probability_at_least_one_chooses_23_l599_599493

theorem probability_at_least_one_chooses_23 :
  let n := 4 in
  let total_events := 2^n in
  let probability_opposite_event := 1 / total_events in
  let probability_at_least_one := 1 - probability_opposite_event in
  probability_at_least_one = 15 / 16 := by
  sorry

end probability_at_least_one_chooses_23_l599_599493


namespace radius_of_ball_l599_599994

theorem radius_of_ball (diameter top_depth : ℝ) (h_diameter : diameter = 30) (h_depth : top_depth = 10) : 
  ∃ r : ℝ, r = 16.25 :=
by
  let r := 16.25
  use r
  sorry

end radius_of_ball_l599_599994


namespace radio_and_magazines_l599_599334

-- Declaring the sets and their cardinalities based on given conditions.
variables {T R M : Type} [Fintype T] [Fintype R] [Fintype M]

-- Given conditions are defined as follows:
def total_clients : ℕ := 180
def t (T : Type) [Fintype T] : ℕ := 115
def r (R : Type) [Fintype R] : ℕ := 110
def m (M : Type) [Fintype M] : ℕ := 130
def t_inter_m (T M : Type) [Fintype T] [Fintype M] : ℕ := 85
def t_inter_r (T R : Type) [Fintype T] [Fintype R] : ℕ := 75
def t_inter_r_inter_m (T R M : Type) [Fintype T] [Fintype R] [Fintype M] : ℕ := 80

-- The problem is to prove that the number of clients using both radio and magazines is 95.
theorem radio_and_magazines : (r ∩ m : Type) [Fintype (r ∩ m)] = 95 :=
by
  -- Here the proof steps will go.
  sorry

end radio_and_magazines_l599_599334


namespace incorrect_regression_statement_incorrect_statement_proof_l599_599974

-- Define the regression equation and the statement about y and x
def regression_equation (x : ℝ) : ℝ := 3 - 5 * x

-- Proof statement: given the regression equation, show that when x increases by one unit, y decreases by 5 units on average
theorem incorrect_regression_statement : 
  (regression_equation (x + 1) = regression_equation x + (-5)) :=
by sorry

-- Proof statement: prove that the statement "when the variable x increases by one unit, y increases by 5 units on average" is incorrect
theorem incorrect_statement_proof :
  ¬ (regression_equation (x + 1) = regression_equation x + 5) :=
by sorry  

end incorrect_regression_statement_incorrect_statement_proof_l599_599974


namespace card_draw_suit_probability_l599_599112

noncomputable def probability_at_least_one_card_each_suit : ℚ :=
  3 / 32

theorem card_draw_suit_probability : 
  (∃ deck : set ℕ, deck = set.univ ∧ ∀ suit : ℕ in deck, suit < 4) →
  (∃ draws : list ℕ, length draws = 5) →
  ∃ prob : ℚ, prob = probability_at_least_one_card_each_suit :=
sorry

end card_draw_suit_probability_l599_599112


namespace find_x_l599_599472

variable {x : ℝ}
def a : ℝ × ℝ := (x, 1)
def b : ℝ × ℝ := (3, 6)
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem find_x (h : perpendicular a b) : x = -2 := by
  sorry

end find_x_l599_599472


namespace sum_minus_product_inequality_l599_599051

theorem sum_minus_product_inequality (n : ℕ) (h : 2 ≤ n) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  (∑ k in Finset.range n, x k) - (∑ ij in Finset.Ico 0 n, x ij.1 * x ij.2) ≤ 1 := by
sorry

end sum_minus_product_inequality_l599_599051


namespace find_x_l599_599479

theorem find_x (x : ℝ) (h : ∑' n : ℕ, (n + 1) * x ^ n = 9) : x = 2 / 3 :=
sorry

end find_x_l599_599479


namespace binary_representation_25_l599_599642

def binary_representation (n : ℕ) : list ℕ :=
  if h : n = 0 then [] else
    let rem := n % 2,
        div := n / 2 in
    rem :: binary_representation div

theorem binary_representation_25 : binary_representation 25 = [1, 0, 0, 1, 1] :=
  sorry

end binary_representation_25_l599_599642


namespace fraction_to_decimal_l599_599748

theorem fraction_to_decimal (a b : ℕ) (h₀ : a = 49) (h₁ : b = 160) : a / b = 0.30625 :=
by {
  -- Assume the given conditions
  assume h₀ : a = 49,
  assume h₁ : b = 160,
  -- Prove the theorem
  sorry
}

end fraction_to_decimal_l599_599748


namespace general_formula_transformed_sum_l599_599564

section ProblemConditions

-- Definition of the sequence condition
def seq_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 ≤ n → (∑ i in finset.range n, (2 * i + 1) * a (i + 1)) = 2 * n

-- General formula for the sequence
theorem general_formula (a : ℕ → ℝ) (h : seq_condition a) : 
  ∀ n : ℕ, 1 ≤ n → a n = 2 / (2 * n - 1) :=
sorry

-- Sum of first n terms of the transformed sequence
theorem transformed_sum (a : ℕ → ℝ) (h : seq_condition a) : 
  ∀ n : ℕ, 1 ≤ n → 
  (∑ i in finset.range n, a (i + 1) / (2 * (i + 1) + 1)) = (2 * n) / (2 * n + 1) :=
sorry

end ProblemConditions

end general_formula_transformed_sum_l599_599564


namespace xy_solution_l599_599954

theorem xy_solution (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 72) : x * y = -8 := by
  sorry

end xy_solution_l599_599954


namespace solve_abs_eq_l599_599920

theorem solve_abs_eq (x : ℝ) : |x - 3| = 5 - 2x ↔ (x = 2 ∨ x = 8/3) :=
by sorry

end solve_abs_eq_l599_599920


namespace find_lambda_l599_599475

open Real

def vector := (Real × Real)

def collinear (v1 v2 : vector) : Prop :=
  ∃ (k : Real), v1.1 = k * v2.1 ∧ v1.2 = k * v2.2

theorem find_lambda : ∀ (a b c : vector), 
  a = (1, 2) → 
  b = (2, 0) → 
  c = (1, -2) → 
  ∃ (λ : Real), collinear (λ * a.1 + b.1, λ * a.2 + b.2) c :=
by
  intros a b c ha hb hc
  use -1
  simp [vector, collinear, ha, hb, hc]
  sorry

end find_lambda_l599_599475


namespace multiple_of_people_l599_599204

-- Define the conditions
variable (P : ℕ) -- number of people who can do the work in 8 days

-- define a function that represents the work capacity of M * P people in days, 
-- we abstract away the solving steps into one declaration.

noncomputable def work_capacity (M P : ℕ) (days : ℕ) : ℚ :=
  M * (1/8) * days

-- Set up the problem to prove that the multiple of people is 2
theorem multiple_of_people (P : ℕ) : ∃ M : ℕ, work_capacity M P 2 = 1/2 :=
by
  use 2
  unfold work_capacity
  sorry

end multiple_of_people_l599_599204


namespace number_of_adult_female_alligators_l599_599158

-- Define the conditions
def total_alligators (females males: ℕ) : ℕ := females + males

def male_alligators : ℕ := 25
def female_alligators : ℕ := 25
def juvenile_percentage : ℕ := 40

-- Calculate the number of juveniles
def juvenile_count : ℕ := (juvenile_percentage * female_alligators) / 100

-- Calculate the number of adults
def adult_female_alligators : ℕ := female_alligators - juvenile_count

-- The main theorem statement
theorem number_of_adult_female_alligators : adult_female_alligators = 15 :=
by
    sorry

end number_of_adult_female_alligators_l599_599158


namespace group_photo_arrangements_l599_599638

-- Definitions corresponding to the conditions
def teachers := {1, 2}
def students := {A, B, C, D}
def positions := Finset.range 6
def middle_positions := {2, 3}
def non_end_positions := {1, 2, 3, 4}

-- Main statement to prove the count of arrangements
theorem group_photo_arrangements :
  let middle_ways := teachers.card.factorial,
      A_ways := 2,
      remaining_ways := (students \ {A}).card.factorial
  in (middle_ways * A_ways * remaining_ways) = 24 :=
by
  simp only [teachers, students, middle_positions, non_end_positions],
  rw Finset.card_insert_of_not_mem,
  rw Finset.card_insert_of_not_mem,
  -- middle_ways
  have h1 : (teachers.card.factorial = 2),
  -- A_ways
  have h2 : (A_ways = 2),
  -- remaining_ways
  have h3 : (remaining_ways = 6),
  -- total ways
  simp [h1, h2, h3],
  exact dec_trivial

end group_photo_arrangements_l599_599638


namespace exists_small_triangle_l599_599774

-- Define the conditions as assumptions
variables (S : set (ℝ × ℝ)) (h_card : S.card = 27)
          (h_no_three_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ S → p2 ∈ S → p3 ∈ S → collinear ℝ {p1, p2, p3} = false)
          (unit_square : set (ℝ × ℝ))
          (h_unit_square_vertices : set.finite (S ∩ unit_square) ∧ (S ∩ unit_square).card = 4)
          (h_points_inside_square : ∀ x ∈ S, x ∉ (S ∩ unit_square) → x ∈ interior unit_square)

-- Define the statement to be proved
theorem exists_small_triangle
  (h_vertices : ∀ p ∈ (S ∩ unit_square), p = (0, 0) ∨ p = (1, 0) ∨ p = (0, 1) ∨ p = (1, 1)) :
  ∃ (X Y Z : ℝ × ℝ), X ∈ S ∧ Y ∈ S ∧ Z ∈ S ∧ 𝓐 (triangle.mk X Y Z) ≤ (1 / 48 : ℝ) :=
sorry

end exists_small_triangle_l599_599774


namespace equilateral_triangle_perimeter_l599_599847

theorem equilateral_triangle_perimeter (r : ℝ) (h1 : r = 2) :
  let s := 3 * (2 * real.sqrt 3 + 4) in s = 6 * real.sqrt 3 + 12 :=
by
  -- Proof skipped
  sorry

end equilateral_triangle_perimeter_l599_599847


namespace maximum_value_t_l599_599215

theorem maximum_value_t :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 31 →
  (2 ^ (n + 1)) ^ 2 + 256 ≥ (2 ^ (n + 1) - 2 + 2) * (t + 5) →
  t ≤ 27 :=
begin
  sorry
end

end maximum_value_t_l599_599215


namespace range_of_m_l599_599454

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 3
noncomputable def g (m x : ℝ) : ℝ := m * (x - 1) + 2

theorem range_of_m (m : ℝ) :
  (∃ x₁ ∈ Icc (0 : ℝ) 3, ∀ x₂ ∈ Icc (0 : ℝ) 3, f x₁ = g m x₂) → m ∈ Ioo 0 (1 / 2) :=
by
  sorry

end range_of_m_l599_599454


namespace simplify_expression_l599_599751

theorem simplify_expression : 
  2^345 - 3^4 * (3^2)^2 = 2^345 - 6561 := by
sorry

end simplify_expression_l599_599751


namespace avg_first_5_multiples_of_9_l599_599658

-- Define the first 5 multiples of 9
def multiples : List ℝ := [9, 18, 27, 36, 45]

-- Define the sum of the first 5 multiples of 9
def sum_multiples : ℝ := List.sum multiples

-- Define the number of multiples
def num_multiples : ℝ := 5

-- Define the average of the multiples
def average_multiples : ℝ := sum_multiples / num_multiples

-- The theorem we want to prove
theorem avg_first_5_multiples_of_9 : average_multiples = 27 := by
  -- This proof step is currently omitted
  sorry

end avg_first_5_multiples_of_9_l599_599658


namespace find_p_l599_599568

variable (f w : ℂ) (p : ℂ)
variable (h1 : f = 4)
variable (h2 : w = 10 + 200 * Complex.I)
variable (h3 : f * p - w = 20000)

theorem find_p : p = 5002.5 + 50 * Complex.I := by
  sorry

end find_p_l599_599568


namespace range_of_a_l599_599424

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 + 2*x else -(x^2 + 2*(-x))

theorem range_of_a (a : ℝ) (h : f (3 - a^2) > f (2 * a)) : -3 < a ∧ a < 1 := sorry

end range_of_a_l599_599424


namespace find_range_g_l599_599020

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + abs x

theorem find_range_g :
  {x : ℝ | g (2 * x - 1) < g 3} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end find_range_g_l599_599020


namespace find_f_five_thrids_l599_599546

noncomputable def f : ℝ → ℝ := sorry

lemma odd_fun (x : ℝ) : f (-x) = -f(x) :=
sorry

lemma f_transformation (x : ℝ) : f (1 + x) = f (-x) :=
sorry

lemma f_neg_inv : f (-1 / 3) = 1 / 3 :=
sorry

theorem find_f_five_thrids : f (5 / 3) = 1 / 3 :=
by
  -- application of the conditions stated
  have h1 : f(1 + (5 / 3 - 1)) = f(-(5 / 3 - 1)),
    from f_transformation (5 / 3 - 1),
  have h2 : f(-(5 / 3 - 1)) = -f(5 / 3 - 1),
    from odd_fun (5 / 3 - 1),
  have h3 : 5 / 3 - 1 = 2 / 3,
    by norm_num,
  rw [h3] at h1,
  rw [h2] at h1,
  have h4 : f (-1 / 3) = 1 / 3,
    from f_neg_inv,
  rw [←h4] at h1,
  exact h1,
sorry

end find_f_five_thrids_l599_599546


namespace minimum_value_of_f_l599_599932

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 / x

theorem minimum_value_of_f : ∃ x ∈ set.Ioi (0:ℝ), (∀ y ∈ set.Ioi (0:ℝ), f x <= f y) ∧ f x = 4 := by
  sorry

end minimum_value_of_f_l599_599932


namespace cot_sum_arccot_roots_eq_one_l599_599170

noncomputable def poly : Polynomial ℂ :=  Polynomial.X ^ 10 - Polynomial.X ^ 9 + Polynomial.X ^ 8 - Polynomial.X ^ 7 + Polynomial.X ^ 6 - Polynomial.X ^ 5 + Polynomial.X ^ 4 - Polynomial.X ^ 3 + Polynomial.X ^ 2 - Polynomial.X + 1

theorem cot_sum_arccot_roots_eq_one (z : ℂ) (hz : z ∈ poly.roots) :
  ∑ k in (Finset.range 10).map (Fin 10).attach.to_embedding, arccot (poly.roots k) = 1 := by
  sorry

end cot_sum_arccot_roots_eq_one_l599_599170


namespace jo_total_reading_hours_l599_599870

variables (total_pages : ℝ) (current_page : ℝ) (prev_page : ℝ)
          (reading_speed : ℝ) (reading_time_per_hour : ℝ)
          (break_time : ℝ) (break_interval : ℝ)

def pages_left (total_pages current_page : ℝ) : ℝ :=
  total_pages - current_page

def pages_per_reading_session (reading_time_per_hour reading_speed : ℝ) : ℝ :=
  (reading_time_per_hour / 60) * reading_speed

def sessions_needed (pages_left pages_per_reading_session : ℝ) : ℝ :=
  pages_left / pages_per_reading_session

noncomputable def total_hours (total_pages current_page reading_speed reading_time_per_hour break_time break_interval : ℝ) : ℝ :=
  let left_pages := pages_left total_pages current_page
  let pages_per_sess := pages_per_reading_session reading_time_per_hour reading_speed
  let sessions := sessions_needed left_pages pages_per_sess
  let total_reading_time := (sessions * reading_time_per_hour)
  let total_distracted_time := (sessions * (60 - reading_time_per_hour))
  let total_break_time := ((sessions / (break_interval * 60)) * break_time)
  ((total_reading_time + total_distracted_time + total_break_time) / 60)

theorem jo_total_reading_hours :
  total_hours 325.5 136.25 38.75 50 30 2 = 7 :=
by
  -- Insert proof here
  sorry

end jo_total_reading_hours_l599_599870


namespace divisors_n_squared_l599_599311

theorem divisors_n_squared (n : ℕ) (h : nat.num_divisors n = 4) :
  nat.num_divisors (n ^ 2) = 7 ∨ nat.num_divisors (n ^ 2) = 9 :=
sorry

end divisors_n_squared_l599_599311


namespace range_of_m_l599_599449

noncomputable def f (x : ℝ) : ℝ := Real.log2 (x*x + 2)

def a (m : ℝ) : ℝ × ℝ := (m, 1)
def b (m : ℝ) : ℝ × ℝ := (1/2, m/2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 * u.1 + u.2 * u.2)

def diff (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

theorem range_of_m (m : ℝ) (h : m > 0) (h_cond : f (dot_product (a m) (b m)) ≥ f (magnitude (diff (a m) (b m)))) :
  4 - Real.sqrt 11 ≤ m ∧ m ≤ 4 + Real.sqrt 11 := sorry

end range_of_m_l599_599449


namespace student_total_marks_l599_599496

theorem student_total_marks (total_questions correct_answers incorrect_answer_score correct_answer_score : ℕ)
    (h1 : total_questions = 60)
    (h2 : correct_answers = 38)
    (h3 : correct_answer_score = 4)
    (h4 : incorrect_answer_score = 1)
    (incorrect_answers := total_questions - correct_answers) 
    : (correct_answers * correct_answer_score - incorrect_answers * incorrect_answer_score) = 130 :=
by
  -- proof to be provided here
  sorry

end student_total_marks_l599_599496


namespace power_of_128_div_7_eq_16_l599_599720

theorem power_of_128_div_7_eq_16 : (128 : ℝ) ^ (4 / 7) = 16 := by
  sorry

end power_of_128_div_7_eq_16_l599_599720


namespace train_speed_kmph_l599_599699

theorem train_speed_kmph (len_train : ℝ) (len_platform : ℝ) (time_cross : ℝ) (total_distance : ℝ) (speed_mps : ℝ) (speed_kmph : ℝ) 
  (h1 : len_train = 250) 
  (h2 : len_platform = 150.03) 
  (h3 : time_cross = 20) 
  (h4 : total_distance = len_train + len_platform) 
  (h5 : speed_mps = total_distance / time_cross) 
  (h6 : speed_kmph = speed_mps * 3.6) : 
  speed_kmph = 72.0054 := 
by 
  -- This is where the proof would go
  sorry

end train_speed_kmph_l599_599699


namespace petya_wins_prize_one_player_wins_prize_l599_599900

-- Define the setup of the problem
def Petya_and_friends_game (n_players : ℕ) : Prop :=
  n_players = 10

-- Define the probability calculation for Petya winning a prize
def probability_petya_wins : ℚ :=
  (5/6)^9

-- Define the probability calculation for at least one player winning a prize
noncomputable def probability_one_player_wins : ℚ :=
  let p_one_wins := (5/6)^9 in
  let p_two_same := (5/6) * (4/6)^8 in
  let p_three_same := (5/6) * (4/6) * (3/6)^7 in
  let inclusion_exclusion := 
    10 * p_one_wins - 
    45 * p_two_same + 
    120 * p_three_same - 
    210 * (5*4*3*2^6)/(6^9) + 
    252 * (5*4*3*2*1^5)/(6^9) in
  inclusion_exclusion

theorem petya_wins_prize (h : Petya_and_friends_game 10) : probability_petya_wins ≈ 0.194 := 
begin
  sorry
end

theorem one_player_wins_prize (h : Petya_and_friends_game 10) : probability_one_player_wins ≈ 0.919 :=
begin
  sorry
end

end petya_wins_prize_one_player_wins_prize_l599_599900


namespace determine_a_l599_599735

def E (a b c : ℝ) : ℝ := a * b^2 + c

theorem determine_a (a : ℝ) :
  E(a, 3, 12) = E(a, 5, 6) → a = (3 / 8) := by
  sorry

end determine_a_l599_599735


namespace planes_perpendicular_l599_599165

-- Definitions representing the basic entities and relationships
variables {Point : Type} [Nonempty Point] [Field Point]
variables (l : Set Point) (α β : Set (Set Point))

-- Conditions
variable (line_parallel_plane : l.parallel α)
variable (line_parallel_plane : l.parallel β)
variable (line_perpendicular_plane : l.perpendicular α)

-- Proof statement
theorem planes_perpendicular 
  (l_parallel_β : l.parallel β)
  (l_perpendicular_α : l.perpendicular α) 
  : α.perpendicular β :=
sorry

end planes_perpendicular_l599_599165


namespace other_root_of_quadratic_l599_599119

theorem other_root_of_quadratic (m : ℝ) (h : (2:ℝ) * (t:ℝ) = -6 ): 
  ∃ t, t = -3 :=
by
  sorry

end other_root_of_quadratic_l599_599119


namespace sum_of_possible_N_l599_599611

theorem sum_of_possible_N 
  (a b c N : ℕ) (h1 : N = a * b * c) (h2 : N = 8 * (a + b + c)) (h3 : c = 2 * (a + b))
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∑ (ab_pairs : ℕ × ℕ) in ({(1, 16), (2, 8), (4, 4)} : Finset (ℕ × ℕ)), (16 * (ab_pairs.1 + ab_pairs.2)) = 560 := 
by
  sorry

end sum_of_possible_N_l599_599611


namespace monotonic_intervals_range_of_b_l599_599448

def f (a b x : ℝ) : ℝ := 2 * a * x + b * x - 1 - 2 * Real.log x

theorem monotonic_intervals (a : ℝ) : 
  ∀ x : ℝ, x > 0 → 
    if b = 0 then 
      if a ≤ 0 then 
        ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f a 0 x₁ ≥ f a 0 x₂
      else if a > 0 then 
        (∀ x₁ x₂ : ℝ, 0 < x₁ < x₂ ∧ x₂ < 1 / a → f a 0 x₁ ≥ f a 0 x₂) ∧ 
        (∀ x₁ x₂ : ℝ, x₁ > 1 / a ∧ x₁ < x₂ → f a 0 x₁ ≤ f a 0 x₂)
    else 
      true := 
sorry

theorem range_of_b (b : ℝ) : 
  (∀ x : ℝ, x > 0 → f 1 b x ≥ 2 * b * x - 3) → 
  b ≤ 2 - 2 / Real.exp 2 := 
sorry

end monotonic_intervals_range_of_b_l599_599448


namespace pascal_triangle_right_diagonal_pascal_triangle_left_diagonal_l599_599577

theorem pascal_triangle_right_diagonal (n k : ℕ) :
  finset.sum (finset.range (n - k + 1)) (λ i, nat.choose (n - i - 1) (k - 1)) = nat.choose n k := 
sorry

theorem pascal_triangle_left_diagonal (n k : ℕ) :
  finset.sum (finset.range (n - k)) (λ i, nat.choose (n - i - 1) k) = nat.choose n k := 
sorry

end pascal_triangle_right_diagonal_pascal_triangle_left_diagonal_l599_599577


namespace find_q_l599_599836

-- Given conditions
def quadratic_eq (p q : ℂ) : ℂ → Prop :=
  λ x, 5*x^2 + p*x + q = 0

theorem find_q (p q : ℂ) 
  (h : quadratic_eq p q (3 + 2 * complex.i)) : 
  q = 65 := 
by 
-- Skip the actual proof with sorry
  sorry

end find_q_l599_599836


namespace canteen_distance_l599_599292

-- Given definitions
def G_to_road : ℝ := 450
def G_to_B : ℝ := 700

-- Proof statement
theorem canteen_distance :
  ∃ x : ℝ, (x ≠ 0) ∧ 
           (G_to_road^2 + (x - G_to_road)^2 = x^2) ∧ 
           (x = 538) := 
by {
  sorry
}

end canteen_distance_l599_599292


namespace find_max_min_of_f_l599_599092

noncomputable def f (x : ℝ) : ℝ := real.cos (2 * x) - 2 * real.cos x

theorem find_max_min_of_f :
  let interval := set.Icc (-real.pi / 3) (real.pi / 4)
  max_value : ℝ := -1,
  min_value : ℝ := -real.sqrt 2,
  max x in interval, f x = max_value ∧ min x in interval, f x = min_value :=
begin
  sorry
end

end find_max_min_of_f_l599_599092


namespace find_f_five_thirds_l599_599539

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = - f x
def functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (-x)
def specific_value (f : ℝ → ℝ) := f (-1/3) = 1/3

theorem find_f_five_thirds
  (hf_odd : odd_function f)
  (hf_fe : functional_equation f)
  (hf_value : specific_value f) :
  f (5 / 3) = 1 / 3 := by
  sorry

end find_f_five_thirds_l599_599539


namespace problem_part1_problem_part2_l599_599554

noncomputable theory

open_locale real
open_locale euclidean_geometry

variables {A B C H O M Q K N I P : Point}

-- Given conditions
variable (triangle_ABC_acute_scalene : triangle A B C)
variable (orthocenter_H : orthocenter H A B C)
variable (circumcenter_O : circumcenter O A B C)
variable (midpoint_M : midpoint M B C)
variable (AB_eq_AM : ∃ M, AB = AM)
variable (points_QK_on_circumcircle : ∃ Q K, Q ≠ A ∧ K ≠ A ∧ on_circumcircle Q A B C ∧ on_circumcircle K A B C)
variable (angles_conditions : ∃ Q K, angle_h90 Q A H ∧ ∠ BAK = ∠ CAM)
variable (midpoint_N : midpoint N A H)
variable (intersection_I : intersection I B_midline A_altitude)

-- Questions to prove
theorem problem_part1 : IN = IO :=
sorry

theorem problem_part2 :
  ∃ P, on_symmedian P A B C ∧ on_circle P B (distance B M) ∧ tangent_to (circle_p A P N) AB :=
sorry

end problem_part1_problem_part2_l599_599554


namespace gcd_binomials_eq_one_l599_599905

theorem gcd_binomials_eq_one {n k : ℕ} (hn : 0 < n) (hk : 0 < k) :
  Nat.gcd_list (List.map (fun i => Nat.choose (n + i) k) (List.range (k + 1))) = 1 :=
  sorry

end gcd_binomials_eq_one_l599_599905


namespace flowers_given_to_mother_l599_599331

-- Definitions based on conditions:
def Alissa_flowers : Nat := 16
def Melissa_flowers : Nat := 16
def flowers_left : Nat := 14

-- The proof problem statement:
theorem flowers_given_to_mother :
  Alissa_flowers + Melissa_flowers - flowers_left = 18 := by
  sorry

end flowers_given_to_mother_l599_599331


namespace card_draw_suit_probability_l599_599111

noncomputable def probability_at_least_one_card_each_suit : ℚ :=
  3 / 32

theorem card_draw_suit_probability : 
  (∃ deck : set ℕ, deck = set.univ ∧ ∀ suit : ℕ in deck, suit < 4) →
  (∃ draws : list ℕ, length draws = 5) →
  ∃ prob : ℚ, prob = probability_at_least_one_card_each_suit :=
sorry

end card_draw_suit_probability_l599_599111


namespace lines_intersect_l599_599229

theorem lines_intersect (a b : ℝ) (h1 : 2 = (1/3) * 1 + a) (h2 : 1 = (1/2) * 2 + b) : a + b = 5 / 3 := 
by {
  -- Skipping the proof itself
  sorry
}

end lines_intersect_l599_599229


namespace find_angle_A_find_area_ABC_l599_599513

-- Define the conditions
variables {A B C a b c : ℝ} (hTriangle : triangle A B C)
variables (hOpposites : opposite_sides_to_angles a A b B c C)
variables (f : ℝ → ℝ) (hf : ∀ x, f x = 2 * cos x * sin (x - A))
variables (xmin : ℝ) (hxmin : xmin = 11 * π / 12)  -- x = 11π / 12
variables (ha : ℝ) (ha_value : ha = 7)  -- a = 7
variables (sinB_plus_sinC : ℝ) (hsinB_plus_sinC : sinB_plus_sinC = sin B + sin C)
variables (hsinBplusC_value : sinB_plus_sinC = 13 * sqrt 3 / 14)

-- Define theorem for problem 1: finding angle A
theorem find_angle_A :
  A = π / 3 :=
sorry

-- Given that A = π / 3, define theorem for problem 2: finding the area of triangle ABC
theorem find_area_ABC (hA : A = π / 3) :
  area (triangle A B C) = 10 * sqrt 3 :=
sorry

end find_angle_A_find_area_ABC_l599_599513


namespace problem_equation_l599_599595

def interest_rate : ℝ := 0.0306
def principal : ℝ := 5000
def interest_tax : ℝ := 0.20

theorem problem_equation (x : ℝ) :
  x + principal * interest_rate * interest_tax = principal * (1 + interest_rate) :=
sorry

end problem_equation_l599_599595


namespace expression_evaluation_l599_599372

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) :=
by
  sorry

end expression_evaluation_l599_599372


namespace find_m_value_l599_599363

theorem find_m_value
  (m : ℝ)
  (h: ∀ m, let x_int := 14 in
             let x_int_sec := 28 / (m + 2) in
             let y_int_sec := 28 * m / (m + 2) in
             98 = (1 / 2) * x_int * y_int_sec) :
  m = 2 :=
begin
  sorry
end

end find_m_value_l599_599363


namespace principal_amount_l599_599325

noncomputable def sum_invested (CI : ℝ) (r1 r2 r3 r4 r5 r6 r7 : ℝ) : ℝ :=
  let product := (1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5) * (1 + r6) * (1 + r7)
  (CI / (product - 1))

-- The given conditions
def CI := 6016.75
def r1 := 0.06
def r2 := 0.075
def r3 := 0.08
def r4 := 0.085
def r5 := 0.09
def r6 := 0.095
def r7 := 0.10

-- The main theorem statement
theorem principal_amount :
  sum_invested CI r1 r2 r3 r4 r5 r6 r7 = 5466.67 :=
by
  sorry

end principal_amount_l599_599325


namespace abs_eq_5_iff_l599_599106

theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end abs_eq_5_iff_l599_599106


namespace fourth_game_water_correct_fourth_game_sports_drink_l599_599674

noncomputable def total_bottled_water_cases : ℕ := 10
noncomputable def total_sports_drink_cases : ℕ := 5
noncomputable def bottles_per_case_water : ℕ := 20
noncomputable def bottles_per_case_sports_drink : ℕ := 15
noncomputable def initial_bottled_water : ℕ := total_bottled_water_cases * bottles_per_case_water
noncomputable def initial_sports_drinks : ℕ := total_sports_drink_cases * bottles_per_case_sports_drink

noncomputable def first_game_water : ℕ := 70
noncomputable def first_game_sports_drink : ℕ := 30
noncomputable def second_game_water : ℕ := 40
noncomputable def second_game_sports_drink : ℕ := 20
noncomputable def third_game_water : ℕ := 50
noncomputable def third_game_sports_drink : ℕ := 25

noncomputable def total_consumed_water : ℕ := first_game_water + second_game_water + third_game_water
noncomputable def total_consumed_sports_drink : ℕ := first_game_sports_drink + second_game_sports_drink + third_game_sports_drink

noncomputable def remaining_water_before_fourth_game : ℕ := initial_bottled_water - total_consumed_water
noncomputable def remaining_sports_drink_before_fourth_game : ℕ := initial_sports_drinks - total_consumed_sports_drink

noncomputable def remaining_water_after_fourth_game : ℕ := 20
noncomputable def remaining_sports_drink_after_fourth_game : ℕ := 10

noncomputable def fourth_game_water_consumed : ℕ := remaining_water_before_fourth_game - remaining_water_after_fourth_game

theorem fourth_game_water_correct : fourth_game_water_consumed = 20 :=
by
  unfold fourth_game_water_consumed remaining_water_before_fourth_game
  sorry

theorem fourth_game_sports_drink : false :=
by
  sorry

end fourth_game_water_correct_fourth_game_sports_drink_l599_599674


namespace abs_eq_five_l599_599099

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  sorry

end abs_eq_five_l599_599099


namespace range_of_a_l599_599784

theorem range_of_a (a : ℝ) (h : a ≤ 1) :
  (∃! n : ℕ, n = (2 - a) - a + 1) → -1 < a ∧ a ≤ 0 :=
by 
  sorry

end range_of_a_l599_599784


namespace new_bucket_capacity_l599_599991

/-- Definition of the tank capacity using the initial number of buckets and their capacity --/
def initial_capacity (buckets: ℕ) (capacity_per_bucket: ℝ): ℝ :=
  buckets * capacity_per_bucket

/-- Theorem stating that given the initial and new bucket counts and the capacity per initial bucket, the new capacity per bucket is as expected --/
theorem new_bucket_capacity (initial_buckets: ℕ) (initial_capacity_per_bucket: ℝ) (new_buckets: ℕ) 
  (total_capacity: ℝ := initial_capacity initial_buckets initial_capacity_per_bucket) (new_capacity_per_bucket: ℝ) : 
  initial_buckets = 22 → initial_capacity_per_bucket = 13.5 → new_buckets = 33 → new_capacity_per_bucket = total_capacity / new_buckets :=
by{
  intros h1 h2 h3,
  have h4 := calc 
    total_capacity = initial_capacity initial_buckets initial_capacity_per_bucket : rfl,
  have h5: total_capacity = 297, by {
    rw [h1, h2],
    exact calc
    297 = 22 * 13.5 : by norm_num,
  },
  have h6 := calc 
    new_capacity_per_bucket = 297 / 33 : by rw h5,
  exact h6,
 }

end new_bucket_capacity_l599_599991


namespace inequality_system_has_three_integer_solutions_l599_599407

theorem inequality_system_has_three_integer_solutions (m : ℝ) :
  (∃ (s : finset ℤ), s.card = 3 ∧ ∀ x ∈ s, x + 5 > 0 ∧ x - m ≤ 1) ↔ -3 ≤ m ∧ m < -2 :=
by
  sorry

end inequality_system_has_three_integer_solutions_l599_599407


namespace lizabet_is_knight_or_liar_lizabet_is_knight_or_liar_l599_599651

inductive Inhabitant
| knight : Inhabitant  -- always tells the truth
| liar : Inhabitant    -- always lies

variable (Lizabet : Inhabitant) (brother : Inhabitant)

-- Define what it means for a statement about an inhabitant to be true
def truth_value (inh : Inhabitant) (stmt : Prop) : Prop :=
  match inh with
  | Inhabitant.knight => stmt
  | Inhabitant.liar => ¬stmt

-- The question asked to the brother is: "Are you and Lizabet the same type (both knights or both knaves)?"
-- We create a definition for this question's truth value for the brother
def same_type_truth_value : Prop :=
  (Lizabet = brother)

-- Define the brother's answer to the same_type_truth_value question
def brother_answer : Prop :=
  truth_value brother same_type_truth_value

-- Formulate the theorem to determine Lizabet's nature from the brother's answer
theorem lizabet_is_knight_or_liar (answer_yes : brother_answer) : Lizabet = Inhabitant.knight :=
by
  sorry
theorem lizabet_is_knight_or_liar (answer_no : ¬brother_answer) : Lizabet = Inhabitant.liar :=
by
  sorry

end lizabet_is_knight_or_liar_lizabet_is_knight_or_liar_l599_599651


namespace solve_abs_eq_l599_599916

theorem solve_abs_eq {x : ℝ} (h₁ : x ≠ 3 ∧ (x >= 3 ∨ x < 3)) :
  (|x - 3| = 5 - 2 * x) ↔ x = 2 :=
by
  split;
  intro h;
  sorry

end solve_abs_eq_l599_599916


namespace analytical_expression_and_range_l599_599451

-- Define given conditions
def f (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
def P : (ℝ × ℝ) := (1, 2)
def slope_P {x : ℝ} (h : f(1) = 2) : Prop := (3 * x^2 + 2 * a * x + b) = 4
def extremum_cond (x : ℝ) : Prop := x = -1 ∧ (3 * x^2 + 2 * a * x + b) = 0

-- Prove analyticial function and range of m
theorem analytical_expression_and_range :
  (∀ a b c, f(1) = 2 → slope_P → extremum_cond → f(x) = x^3 + x^2 - x + 1) ∧
  (∀ m, ∃ (g : ℝ → ℝ), g(x) = f(x) + m - 1 → -1 < m ∧ m < 5/27)
:= by
  sorry

end analytical_expression_and_range_l599_599451


namespace find_c_l599_599558

theorem find_c (x y z : ℕ) (h1 : |x - y|^2010 + |z - x|^2011 = 1) : |x - y| + |y - z| + |z - x| = 2 :=
by
  sorry

end find_c_l599_599558


namespace relationship_among_distances_l599_599077

variables {Point : Type} {Plane : Type} {Line : Type} 
variable [metric_space Point] 
variables {α β : Plane} {m n : Line} {A B : Point}

-- Define the conditions
axiom plane_parallel (h1 : α ∥ β) 
axiom line_in_plane_m (h2 : A ∈ m) (h3 : m ⊆ α)
axiom line_in_plane_n (h4 : B ∈ n) (h5 : n ⊆ β)
axiom distance_AB (a : ℝ) (h6 : dist A B = a)
axiom distance_A_to_n (b : ℝ) (h7 : dist_to_line A n = b)
axiom distance_m_to_n (c : ℝ) (h8 : dist_between_lines m n = c)

-- The proof that c ≤ b ≤ a given all the conditions
theorem relationship_among_distances : c ≤ b ∧ b ≤ a :=
by sorry

end relationship_among_distances_l599_599077


namespace determine_a_l599_599445

noncomputable def f (x : ℝ) : ℝ := Real.exp (abs (x - 1)) + 1

theorem determine_a (a : ℝ) (h : f a = 2) : a = 1 :=
by
  sorry

end determine_a_l599_599445


namespace num_valid_lists_l599_599531

theorem num_valid_lists : ∃ (n : ℕ), n = 79833600 ∧
  (∀ (l : List ℕ), l.length = 12 ∧ (∀ i ∈ l, 1 ≤ i ∧ i ≤ 12) ∧
    (∀ (j : ℕ), 2 ≤ j ∧ j ≤ 12 → ((l[j - 1] + 1 ∈ l.take (j - 1)) ∨ 
                                   (l[j - 1] - 1 ∈ l.take (j - 1)))) ∧
    (∀ i, 0 ≤ i ∧ i < 6 → l.indexOf (2 * i + 1) < l.indexOf (2 * (i + 1)))) :=
sorry

end num_valid_lists_l599_599531


namespace range_of_m_l599_599453

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (h_deriv : ∀ x, f' x < x)
variable (h_ineq : ∀ m, f (4 - m) - f m ≥ 8 - 4 * m)

theorem range_of_m (m : ℝ) : m ≥ 2 :=
sorry

end range_of_m_l599_599453


namespace vector_parallel_l599_599469

/-- Given vectors a = (-1, 2) and b = (m, 1), if the vector a + 2 * b is parallel to 2 * a - b, 
    then m = -1/2. -/
theorem vector_parallel (m : ℝ) (a : ℝ × ℝ := (-1, 2)) (b : ℝ × ℝ := (m, 1)) :
  let v1 := (a.1 + 2*b.1, a.2 + 2*b.2)
      v2 := (2*a.1 - b.1, 2*a.2 - b.2)
  in (v1.1 * v2.2 = v1.2 * v2.1) → m = -1/2 :=
by
  sorry

end vector_parallel_l599_599469


namespace fourth_leg_length_l599_599179

theorem fourth_leg_length (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 9) (h₃ : c = 10) :
  (∃ x : ℕ, x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ (a + x = b + c ∨ b + x = a + c ∨ c + x = a + b) ∧ (x = 7 ∨ x = 11)) :=
by sorry

end fourth_leg_length_l599_599179


namespace abs_eq_five_l599_599100

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  sorry

end abs_eq_five_l599_599100


namespace center_of_circle_is_min_tangent_length_of_l_to_c_l599_599500

def parametric_line (t : ℝ) : ℝ × ℝ :=
  (sqrt 2 / 2 * t, sqrt 2 / 2 * t + 4 * sqrt 2)

def polar_circle (θ : ℝ) : ℝ :=
  2 * cos (θ + π / 4)

noncomputable def center_cartesian_circle (θ : ℝ) : ℝ × ℝ :=
  let ρ := polar_circle θ in
  (ρ * cos θ, ρ * sin θ)

theorem center_of_circle_is (θ : ℝ) :
  (center_cartesian_circle θ) = (sqrt 2 / 2, - sqrt 2 / 2) := sorry

theorem min_tangent_length_of_l_to_c :
  ∀ t, ∃ l_min, l_min = 2 * sqrt 6 := sorry

end center_of_circle_is_min_tangent_length_of_l_to_c_l599_599500


namespace length_BD_l599_599840

theorem length_BD (A B C D : Type) [point A] [point B] [point C] [point D]
  (h1 : dist A C = 8)
  (h2 : dist B C = 8)
  (h3 : dist A B = 4)
  (h4 : between A B D)
  (h5 : dist C D = 10)
  : dist B D = 2 * real.sqrt 10 - 2 := 
sorry

end length_BD_l599_599840


namespace not_nat_number_l599_599058

-- Given condition
def a : ℤ := -2

-- Proof that a is not a natural number
theorem not_nat_number : ¬ (a ∈ ℕ) := by
  sorry

end not_nat_number_l599_599058


namespace log_domain_l599_599038

theorem log_domain (x : ℝ) : (x^2 - 2*x - 3 > 0) ↔ (x > 3 ∨ x < -1) :=
by
  sorry

end log_domain_l599_599038


namespace sum_reciprocals_find_b_find_c_find_d_l599_599240

-- Problem 1
theorem sum_reciprocals (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 10) : (1 / x + 1 / y) = 2 := 
by
  sorry

-- Problem 2
theorem find_b (b : ℝ) (h : b^2 - 1 = 135 * 137) (h_pos : b > 0) : b = 136 :=
by
  sorry

-- Problem 3
theorem find_c (c : ℝ) (h :  ∄ x y : ℝ, (x + 2*y + 1) = 0 ∧ (c*x + 3*y + 1) = 0 ∧  (- 1/2) * (- c/3) = -1) : c = -6 :=
by
  sorry

-- Problem 4
theorem find_d (c d : ℝ) (h_collinear : collinear { (2, -1), (0, 1), (c, d) }) : d = 7 := 
by
  sorry

end sum_reciprocals_find_b_find_c_find_d_l599_599240


namespace sum_b_first_n_terms_l599_599775

noncomputable def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 2 * a n

def b (n : ℕ) : ℕ := (2 * n + 1) * a n

def T (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ n + 3

theorem sum_b_first_n_terms (n : ℕ) : ∑ i in range n, b i = T n := sorry

end sum_b_first_n_terms_l599_599775


namespace votes_cast_l599_599652

theorem votes_cast (V : ℝ) (h1 : 0.35 * V + 2250 = 0.65 * V) : V = 7500 := 
by
  sorry

end votes_cast_l599_599652


namespace number_of_points_on_line_through_center_l599_599556

noncomputable def num_points (radius: Real) (sum_squares: Real) (d: Real) : Nat :=
  let endpoints := (radius * 2) -- diameter
  let points := {(x,y) : Real × Real | x^2 + y^2 ≤ radius^2 ∧ (radius - x)^2 + (radius + x)^2 = sum_squares}
  points.cardinality -- count the number of such points

theorem number_of_points_on_line_through_center :
  num_points 2 5 4 = 2 :=
by
  sorry

end number_of_points_on_line_through_center_l599_599556


namespace inequality_sum_prod_l599_599906

theorem inequality_sum_prod {n : ℕ} {a : Fin n → ℝ} (h : ∀ k, 0 ≤ a k) :
  (∑ k, (a k)^(n+1)) ≥ (∑ k, (a k)) * (∏ k, (a k)) :=
by
  sorry

end inequality_sum_prod_l599_599906


namespace exists_tetrahedra_volume_and_face_area_conditions_l599_599576

noncomputable def volume (T : Tetrahedron) : ℝ := sorry
noncomputable def face_area (T : Tetrahedron) : List ℝ := sorry

-- The existence of two tetrahedra such that the volume of T1 > T2 
-- and the area of each face of T1 does not exceed any face of T2.
theorem exists_tetrahedra_volume_and_face_area_conditions :
  ∃ (T1 T2 : Tetrahedron), 
    (volume T1 > volume T2) ∧ 
    (∀ (a1 : ℝ), a1 ∈ face_area T1 → 
      ∃ (a2 : ℝ), a2 ∈ face_area T2 ∧ a2 ≥ a1) :=
sorry

end exists_tetrahedra_volume_and_face_area_conditions_l599_599576


namespace fibonacci_sequence_mod_5_mod_7_l599_599731

def sequence : ℕ → ℕ
| 0       := 2
| 1       := 2
| (n + 2) := sequence n + sequence (n + 1)

theorem fibonacci_sequence_mod_5_mod_7 (n : ℕ) :
  sequence 50 % 5 = 2 ∧ sequence 50 % 7 = 2 :=
by {
  sorry
}

end fibonacci_sequence_mod_5_mod_7_l599_599731


namespace number_of_true_propositions_l599_599347

variable (α β γ : Type) [Plane α] [Plane β] [Plane γ]
variable (l m n : Type) [Line l] [Line m] [Line n]

-- Define the propositions

def prop1 (α β γ : Type) [Plane α] [Plane β] [Plane γ] : Prop := 
  α ⊥ γ → β ⊥ γ → Parallel α β

def prop2 (α β : Type) [Plane α] [Plane β] (m n : Type) [Line m] [Line n] : Prop := 
  (m ⊆ α) → (n ⊆ α) → Parallel m β → Parallel n β → Parallel α β

def prop3 (α β : Type) [Plane α] [Plane β] (l : Type) [Line l] : Prop :=
  Parallel α β → l ⊆ α → Parallel l β

def prop4 (α β γ : Type) [Plane α] [Plane β] [Plane γ] (l m n : Type) [Line l] [Line m] [Line n] : Prop :=
  (α ∩ β = l) → (β ∩ γ = m) → (γ ∩ α = n) → Parallel l γ → Parallel m n

-- Statement that exactly 2 of the 4 propositions are true

theorem number_of_true_propositions (α β γ : Type) [Plane α] [Plane β] [Plane γ] (l m n : Type) [Line l] [Line m] [Line n] :
  2 = (if prop1 α β γ then 1 else 0) +
      (if prop2 α β m n then 1 else 0) +
      (if prop3 α β l then 1 else 0) +
      (if prop4 α β γ l m n then 1 else 0) := by sorry

end number_of_true_propositions_l599_599347


namespace days_at_grandparents_l599_599873

theorem days_at_grandparents
  (total_vacation_days : ℕ)
  (travel_to_gp : ℕ)
  (travel_to_brother : ℕ)
  (days_at_brother : ℕ)
  (travel_to_sister : ℕ)
  (days_at_sister : ℕ)
  (travel_home : ℕ)
  (total_days : total_vacation_days = 21) :
  total_vacation_days - (travel_to_gp + travel_to_brother + days_at_brother + travel_to_sister + days_at_sister + travel_home) = 5 :=
by
  sorry -- proof to be constructed

end days_at_grandparents_l599_599873


namespace problem_solution_l599_599352

noncomputable def original_expression : ℝ := 
  (16 / 9) ^ (-1 / 2) + 3 ^ (logBase 3 (1 / 4)) - log 10 5 + sqrt ((log 10 2) ^ 2 - log 10 4 + 1)

theorem problem_solution : original_expression = 1 := by
  sorry

end problem_solution_l599_599352


namespace odd_function_property_l599_599550

theorem odd_function_property {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = - f x) (h2 : ∀ x, f (1 + x) = f (-x)) (h3 : f (-1 / 3) = 1 / 3) : f (5 / 3) = 1 / 3 := 
sorry

end odd_function_property_l599_599550


namespace find_a_l599_599456

-- Condition
def S (n : ℕ) : ℕ := n^2 + 3n

-- Goal
theorem find_a (n : ℕ) : 
  (∀ n, a 0 = 0) ∧ (∀ n, a n = (S n) - (S (n - 1))) → 
  (a n = 2 * n + 2) :=
by
  sorry

end find_a_l599_599456


namespace largest_four_digit_distinct_no_swap_smaller_l599_599042

/-- A helper function to check if a number is four digits and has all distinct digits -/
def is_four_digit_distinct (n : ℕ) : Prop :=
  let digits := (n % 10, (n / 10) % 10, (n / 100) % 10, n / 1000)
  1000 ≤ n ∧ n ≤ 9999 ∧
  (digits.1 ≠ digits.2) ∧ (digits.1 ≠ digits.3) ∧ (digits.1 ≠ digits.4) ∧
  (digits.2 ≠ digits.3) ∧ (digits.2 ≠ digits.4) ∧
  (digits.3 ≠ digits.4)

/-- A helper function to check if swapping any two digits creates a smaller number -/
def no_smaller_by_swap (n : ℕ) : Prop :=
  let digits := (n % 10, (n / 10) % 10, (n / 100) % 10, n / 1000)
  ∀ i j : ℕ, i ≠ j → swap(digits, i, j) > n

/-- The mathematically equivalent Lean 4 proof statement -/
theorem largest_four_digit_distinct_no_swap_smaller : 
  ∃ n : ℕ, is_four_digit_distinct n ∧ no_smaller_by_swap n ∧ n = 7089 := by
  sorry

end largest_four_digit_distinct_no_swap_smaller_l599_599042


namespace value_a_plus_b_l599_599534

variable {x : ℝ}
variable {a b : ℝ}

/-- Definition of f(x) -/
def f (x : ℝ) : ℝ := a * x + b

/-- Definition of h(x) -/
def h (x : ℝ) : ℝ := 3 * x + 1

/-- The given condition h(f(x)) = 5x - 8 -/
theorem value_a_plus_b :
  (∀ x, h (f x) = 5 * x - 8) → a + b = -4 / 3 :=
by 
  sorry

end value_a_plus_b_l599_599534


namespace silk_dyed_amount_l599_599634

-- Define the conditions
def yards_green : ℕ := 61921
def yards_pink : ℕ := 49500

-- Define the total calculation
def total_yards : ℕ := yards_green + yards_pink

-- State what needs to be proven: that the total yards is 111421
theorem silk_dyed_amount : total_yards = 111421 := by
  sorry

end silk_dyed_amount_l599_599634


namespace number_of_girls_in_class_l599_599278

variable (B S G : ℕ)

theorem number_of_girls_in_class
  (h1 : (3 / 4 : ℚ) * B = 18)
  (h2 : B = (2 / 3 : ℚ) * S) :
  G = S - B → G = 12 := by
  intro hg
  sorry

end number_of_girls_in_class_l599_599278


namespace lauren_meets_andrea_l599_599710

-- Definitions based on conditions from the problem
def initial_distance : ℝ := 30
def rate_of_approach : ℝ := 2 -- in km per minute
def time_before_flat : ℝ := 10 -- in minutes
def delay_lauren : ℝ := 5 -- in minutes
def lauren_speed : ℝ := 40 -- from solution steps: v_L = 40 km/h

-- Convert Lauren's speed from km/h to km/min
def lauren_speed_min : ℝ := lauren_speed / 60 -- in km per minute
def remaining_distance := initial_distance - rate_of_approach * time_before_flat -- 10 km remaining

-- The main theorem to prove
theorem lauren_meets_andrea : 
  let total_time := time_before_flat + delay_lauren + (remaining_distance / lauren_speed_min) in
  total_time = 30 := by simp; sorry

end lauren_meets_andrea_l599_599710


namespace odd_function_increasing_ln_x_condition_l599_599064

theorem odd_function_increasing_ln_x_condition 
  {f : ℝ → ℝ} 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) 
  {x : ℝ} 
  (h_f_ln_x : f (Real.log x) < 0) : 
  0 < x ∧ x < 1 := 
sorry

end odd_function_increasing_ln_x_condition_l599_599064


namespace intersecting_squares_and_circles_l599_599728

theorem intersecting_squares_and_circles :
  let line_segment : ℝ × ℝ → ℝ × ℝ → set (ℝ × ℝ) := λ (p q : ℝ × ℝ), { (x, y) | x ≥ p.1 ∧ x ≤ q.1 ∧ y = ((q.2 - p.2) / (q.1 - p.1)) * (x - p.1) + p.2 }
  in
  (∃ S C : ℕ, 
    S = 458 ∧
    C = 460 ∧
    S + C = 918) :=
begin
  sorry
end

end intersecting_squares_and_circles_l599_599728


namespace initial_fee_is_correct_l599_599152

noncomputable def initial_fee (total_charge : ℝ) (charge_per_segment : ℝ) (segment_length : ℝ) (distance : ℝ) : ℝ :=
  total_charge - (⌊distance / segment_length⌋ * charge_per_segment)

theorem initial_fee_is_correct :
  initial_fee 4.5 0.25 (2/5) 3.6 = 2.25 :=
by 
  sorry

end initial_fee_is_correct_l599_599152


namespace selection_methods_count_l599_599911

-- Define the number of female students
def num_female_students : ℕ := 3

-- Define the number of male students
def num_male_students : ℕ := 2

-- Define the total number of different selection methods
def total_selection_methods : ℕ := num_female_students + num_male_students

-- Prove that the total number of different selection methods is 5
theorem selection_methods_count : total_selection_methods = 5 := by
  sorry

end selection_methods_count_l599_599911


namespace bananas_to_oranges_equiv_l599_599714

theorem bananas_to_oranges_equiv :
  (∀ (bananas pears apples oranges : ℕ),
    (5 * bananas = 4 * pears) →
    (8 * pears = 3 * apples) →
    (12 * apples = 9 * oranges) →
    80 * bananas = 18 * oranges) sorry

end bananas_to_oranges_equiv_l599_599714


namespace general_formula_sum_first_10_terms_l599_599136

variable {α : Type*} [OrderedField α] 

-- Definition of the arithmetic sequence terms based on given conditions
def arithmetic_sequence (n : ℕ) (a_2 a_5 : α) : α :=
  let d := (a_5 - a_2) / 3 in
  let a_1 := a_2 - d in
  a_1 + (n - 1) * d

-- Specific instances of the sequence
def a_n (n : ℕ) : ℤ := arithmetic_sequence n 14 5

-- Verification of the general formula
theorem general_formula : ∀ n : ℕ, a_n n = 20 - 3 * n :=
by
  sorry

-- Calculation of the sum of the first 10 terms
def sum_of_first_n_terms (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  n * (a 1 + a n) / 2

-- Specific sum for the first 10 terms
def S_10 : ℤ := sum_of_first_n_terms 10 a_n

-- Verification of the sum S_10
theorem sum_first_10_terms : S_10 = 35 :=
by
  sorry

end general_formula_sum_first_10_terms_l599_599136


namespace factor_theorem_l599_599364

noncomputable def polynomial_to_factor : Prop :=
  ∀ x : ℝ, x^4 - 4 * x^2 + 4 = (x^2 - 2)^2

theorem factor_theorem : polynomial_to_factor :=
by
  sorry

end factor_theorem_l599_599364


namespace segment_length_bound_l599_599336

-- Define the setup of the problem
variable (A B C : Point)
variable (incircle : Circle)
variable (segment : Segment)

-- Conditions
variable (h1 : inscribedInAngle incircle A B C)
variable (h2 : touches incircle A B)
variable (h3 : touches incircle A C)
variable (h4 : liesWithinRegion segment A B C (arcMinor B C incircle))

-- The theorem to be proven
theorem segment_length_bound (A B C : Point) (incircle : Circle) (segment : Segment)
  (h1 : inscribedInAngle incircle A B C)
  (h2 : touches incircle A B)
  (h3 : touches incircle A C)
  (h4 : liesWithinRegion segment A B C (arcMinor B C incircle)) :
  lengthOf segment ≤ lengthOf (Segment.mk A B) :=
sorry

end segment_length_bound_l599_599336


namespace series_convergence_l599_599032

noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ), (n ^ 2 + 3 * n - 2) / (nat.factorial (n + 3))

theorem series_convergence : series_sum = 1 / 2 :=
  sorry

end series_convergence_l599_599032


namespace min_m_l599_599391

theorem min_m (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
    27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
by
  sorry

end min_m_l599_599391


namespace range_of_k_trajectory_of_P_n_as_function_of_m_l599_599084

-- Question (I):
theorem range_of_k (k : ℝ) : (x ^ 2 + (y - 4) ^ 2 = 4 ∧ y = k * x) →
  (k < -real.sqrt 3 ∨ k > real.sqrt 3) :=
sorry

-- Question (II):
theorem trajectory_of_P (P : ℝ × ℝ) : (x ^ 2 + (y - 4) ^ 2 = 4 ∧ y = k * x) →
  (P.1 ^ 2 + (P.2 - 2) ^ 2 = 4 ∧ -real.sqrt 3 ≤ P.1 ∧ P.1 ≤ real.sqrt 3) :=
sorry

-- Question (III):
theorem n_as_function_of_m (m n : ℝ) (k : ℝ) :
  (x ^ 2 + (y - 4) ^ 2 = 4 ∧ y = k * x ∧ 
  |sqrt m ^ 2 + n ^ 2| = m ^ 2 + n ^ 2 ∧ m ∈ set.union (set.Ioo 0 (real.sqrt 3)) (set.Ioo (-real.sqrt 3) 0)) →
  n = sqrt (15 * m ^ 2 + 180) / 5 :=
sorry

end range_of_k_trajectory_of_P_n_as_function_of_m_l599_599084


namespace problem1_x_eq_1_over_4_problem2_range_of_x_l599_599018

-- Define the greatest integer less than or equal to a given number
def floor (t : ℝ) := (⌊t⌋ : ℤ)

-- Define f_1(x) = ⌊4x⌋
def f1 (x : ℝ) := floor(4 * x)

-- Define g(x) = 4x - ⌊4x⌋
def g (x : ℝ) := 4 * x - floor(4 * x)

-- Define f_2(x) = f1(g(x))
def f2 (x : ℝ) := f1(g(x))

-- Prove the first statement
theorem problem1_x_eq_1_over_4 : f1 (1/4 : ℝ) = 1 ∧ f2 (1/4 : ℝ) = 0 :=
by
  sorry

-- Prove the second statement
theorem problem2_range_of_x : 
  {x : ℝ | f1 x = 1 ∧ f2 x = 3} = {x : ℝ | (7 : ℝ)/16 ≤ x ∧ x < 1/2} :=
by
  sorry

end problem1_x_eq_1_over_4_problem2_range_of_x_l599_599018


namespace cut_board_into_four_identical_parts_l599_599017

-- Assume a grid layout and shaded cells can be represented using indices
def cell := (ℕ × ℕ)
def board := {cells : set cell // ∃ n, cells.finite ∧ cells ⊆ (λ i j, i < n ∧ j < n)}

-- Partitions the board into four regions
def cut_into_four_regions (b : board) : list (set cell) :=
  sorry -- Details of the exact cutting procedure would be filled here

-- Check if each region meets the conditions
def valid_partition (regions : list (set cell)) : Prop :=
  list.length regions = 4 ∧ ∀ r ∈ regions, set.cardinality r = 3

theorem cut_board_into_four_identical_parts (b : board) (h_shaded : ∀ region ∈ cut_into_four_regions b, set.cardinality region = 3) :
  ∃ regions : list (set cell), cut_into_four_regions b = regions ∧ valid_partition regions :=
  sorry

end cut_board_into_four_identical_parts_l599_599017


namespace mean_eq_median_l599_599572

def percentage_students : List (ℕ × ℕ) := [(15, 80), (20, 85), (25, 90), (30, 95), (10, 100)]

def num_students := 20

def scores : List ℕ := 
  (List.join $ percentage_students.map (fun (pct, score) => List.repeat score (pct * num_students / 100)))

noncomputable def mean : ℕ :=
  scores.sum / scores.length

def median : ℕ :=
  let sorted_scores := scores.qsort (· ≤ ·)
  let n := sorted_scores.length 
  if n % 2 = 0 then (sorted_scores.get (n/2 - 1) + sorted_scores.get (n/2)) / 2
  else sorted_scores.get (n/2)

theorem mean_eq_median : mean scores = median scores :=
by
  -- proof goes here
  sorry

end mean_eq_median_l599_599572


namespace part1_solution_part2_solution_l599_599428

def part1 (m : ℝ) (x1 : ℝ) (x2 : ℝ) : Prop :=
  (m * x1 - 2) * (m * x2 - 2) = 4

theorem part1_solution : part1 (1/3) 9 18 :=
by 
  sorry

def part2 (m x1 x2 : ℕ) : Prop :=
  ((m * x1 - 2) * (m * x2 - 2) = 4)

def count_pairs : ℕ := 7

theorem part2_solution 
  (m x1 x2 : ℕ) 
  (h_pos : m > 0 ∧ x1 > 0 ∧ x2 > 0) : 
  ∃ c, c = count_pairs ∧ 
  (part2 m x1 x2) :=
by 
  sorry

end part1_solution_part2_solution_l599_599428


namespace area_of_isosceles_right_triangle_l599_599857

theorem area_of_isosceles_right_triangle (a b: ℝ) (h_right: a ≠ 0) (h_isosceles: a = b)
  (h_hypotenuse: (a * real.sqrt 2) = 8 * real.sqrt 2) : 
  (1 / 2) * a * b = 64 :=
by 
  sorry

end area_of_isosceles_right_triangle_l599_599857


namespace general_formula_sum_of_b_terms_l599_599431

/-
Problem statement:
Given that \( S_n \) is the sum of the first \( n \) terms of an arithmetic sequence \( \{a_n\} \) with a non-zero common difference, \( a_2 \) is the geometric mean of \( a_1 \) and \( a_4 \), and \( S_9 = 45 \).
1. Prove that the general formula for the sequence \( \{a_n\} \) is \( a_n = n \).
2. Given \( b_n = a_{2n-1} \cdot 3^{a_n - 1} \), prove that the sum of the first \( n \) terms \( T_n \) of the sequence \( \{b_n\} \) is \( T_n = (n-1) \cdot 3^n + 1 \).
-/

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Prove the general formula for \( \{a_n\} \)
theorem general_formula (a₁ d : ℕ) (h_d : d ≠ 0) (h_geometric_mean: arithmetic_sequence a₁ d 2 ^ 2 = arithmetic_sequence a₁ d 1 * arithmetic_sequence a₁ d 4) (h_sum: (9 * a₁) + 36 * d = 45) : ∀ n, arithmetic_sequence a₁ d n = n := by
  sorry

-- Define the sequence \( \{b_n\} \)
def sequence_b (a : ℕ → ℕ) (n : ℕ) : ℕ := a (2 * n - 1) * 3 ^ (a n - 1)

-- Define the sum of the first \( n \) terms of \( \{b_n\} \)
def sum_sequence_b (a : ℕ → ℕ) (n : ℕ) : ℕ := (List.range n).sum (λ k, sequence_b a (k + 1))

-- Prove the sum of the first \( n \) terms of \( \{b_n\} \)
theorem sum_of_b_terms (a : ℕ → ℕ) (h_a : ∀ n, a n = n) : ∀ n, sum_sequence_b a n = (n - 1) * 3^n + 1 := by
  sorry

end general_formula_sum_of_b_terms_l599_599431


namespace Nicky_pace_3_mps_l599_599183

theorem Nicky_pace_3_mps
  (Cristina_faster_than_Nicky : ∀ v_nicky : ℝ, v_nicky < 5)
  (head_start : 48 : ℝ)
  (Cristina_pace : 5 : ℝ)
  (Nicky_time : 24 : ℝ) :
  ∃ v_nicky : ℝ, v_nicky = 3 :=
by
  -- Assume v_nicky in terms of distances covered in given time and the head start
  let v_nicky := ((5 * 24) - 48) / 24
  use v_nicky
  -- Prove that this v_nicky is equal to 3
  simp [v_nicky]
  norm_num
  done

end Nicky_pace_3_mps_l599_599183


namespace inscribed_circle_radius_l599_599199

theorem inscribed_circle_radius (R r : ℝ) (h1 : R = 5) (h2 : (sector_angle : ℝ) = π / 3)
    (h3 : sum_of_radii : r + r * Real.sqrt 3 = R) : 
    r = ((5 * Real.sqrt 3) - 5) / 2 := 
    sorry

end inscribed_circle_radius_l599_599199


namespace largest_possible_perimeter_l599_599253

theorem largest_possible_perimeter
  (a b c : ℕ)
  (h1 : a > 2 ∧ b > 2 ∧ c > 2)  -- sides are greater than 2
  (h2 : a = c ∨ b = c ∨ a = b)  -- at least two polygons are congruent
  (h3 : (a - 2) * (b - 2) = 8 ∨ (a - 2) * (c - 2) = 8 ∨ (b - 2) * (c - 2) = 8)  -- possible factorizations
  (h4 : (a - 2) + (b - 2) + (c - 2) = 12)  -- sum of interior angles at A is 360 degrees
  : 2 * a + 2 * b + 2 * c - 6 ≤ 21 :=
sorry

end largest_possible_perimeter_l599_599253


namespace mean_height_of_soccer_team_l599_599941

theorem mean_height_of_soccer_team :
  let heights := [145, 149, 151, 151, 157, 158, 163, 163, 164, 167, 168, 169, 170, 175] in
  let total : ℝ := heights.sum in
  let count : ℝ := heights.length in
  (total / count) = 160.714 := 
by
  sorry

end mean_height_of_soccer_team_l599_599941


namespace rectangle_perimeter_ratio_l599_599322

theorem rectangle_perimeter_ratio (side_length : ℝ) (h : side_length = 4) :
  let small_rectangle_perimeter := 2 * (side_length + (side_length / 4))
  let large_rectangle_perimeter := 2 * (side_length + (side_length / 2))
  small_rectangle_perimeter / large_rectangle_perimeter = 5 / 6 :=
by
  sorry

end rectangle_perimeter_ratio_l599_599322


namespace box_volume_less_than_1000_l599_599273

theorem box_volume_less_than_1000 : ({(x ∈ ℕ) | 0 < x ∧ (x + 3) * (x - 3) * (x^2 + 9) < 1000}.card = 2) := by
  sorry

end box_volume_less_than_1000_l599_599273


namespace radius_of_ball_l599_599993

theorem radius_of_ball (diameter top_depth : ℝ) (h_diameter : diameter = 30) (h_depth : top_depth = 10) : 
  ∃ r : ℝ, r = 16.25 :=
by
  let r := 16.25
  use r
  sorry

end radius_of_ball_l599_599993


namespace Jim_weekly_savings_l599_599584

-- Define the given conditions
def Sara_initial_savings : ℕ := 4100
def Sara_weekly_savings : ℕ := 10
def weeks : ℕ := 820

-- Define the proof goal based on the conditions
theorem Jim_weekly_savings :
  let Sara_total_savings := Sara_initial_savings + (Sara_weekly_savings * weeks)
  let Jim_weekly_savings := Sara_total_savings / weeks
  Jim_weekly_savings = 15 := 
by 
  sorry

end Jim_weekly_savings_l599_599584


namespace unique_solution_fraction_l599_599025

theorem unique_solution_fraction (x : ℝ) :
  (2 * x^2 - 10 * x + 8 ≠ 0) → 
  (∃! (x : ℝ), (3 * x^2 - 15 * x + 12) / (2 * x^2 - 10 * x + 8) = x - 4) :=
by
  sorry

end unique_solution_fraction_l599_599025


namespace general_formula_a_seq_sum_b_seq_odd_l599_599422

-- Definitions of the sequences
def a_seq : ℕ → ℕ
def b_seq : ℕ → ℕ

-- Conditions
axiom a_1 : a_seq 1 = 1
axiom b_1 : b_seq 1 = 1
axiom a2_a4 : a_seq 2 + a_seq 4 = 10
axiom b2_b4 : b_seq 2 * b_seq 4 = (a_seq 5)

-- Proof statement for part (I)
theorem general_formula_a_seq : ∀ n, a_seq n = 2 * n - 1 := by
  sorry

-- Proof statement for part (II)
theorem sum_b_seq_odd : ∀ n, (∑ i in Finset.range n, b_seq (2 * i + 1)) = (3^n - 1) / 2 := by
  sorry

end general_formula_a_seq_sum_b_seq_odd_l599_599422


namespace syllogism_example_l599_599972

-- Definitions based on the conditions
def is_even (n : ℕ) := n % 2 = 0
def is_divisible_by_2 (n : ℕ) := n % 2 = 0

-- Given conditions:
axiom even_implies_divisible_by_2 : ∀ n : ℕ, is_even n → is_divisible_by_2 n
axiom h2012_is_even : is_even 2012

-- Proving the conclusion and the syllogism pattern
theorem syllogism_example : is_divisible_by_2 2012 :=
by
  apply even_implies_divisible_by_2
  apply h2012_is_even

end syllogism_example_l599_599972


namespace greatest_int_radius_of_area_less_than_45pi_l599_599490

theorem greatest_int_radius_of_area_less_than_45pi (r : ℝ) (h : 1/2 * real.pi * r^2 < 45 * real.pi) : r ≤ 9 :=
begin
  -- Proof steps will be filled here
  sorry
end

end greatest_int_radius_of_area_less_than_45pi_l599_599490


namespace find_f_five_thirds_l599_599540

variable {R : Type*} [LinearOrderedField R]

-- Define the odd function and its properties
variable {f : R → R}
variable (oddf : ∀ x : R, f (-x) = -f x)
variable (propf : ∀ x : R, f (1 + x) = f (-x))
variable (val : f (- (1 / 3 : R)) = 1 / 3)

theorem find_f_five_thirds : f (5 / 3 : R) = 1 / 3 := by
  sorry

end find_f_five_thirds_l599_599540


namespace avg_seven_consecutive_integers_l599_599587

variable (c d : ℕ)
variable (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7)

theorem avg_seven_consecutive_integers (c d : ℕ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 6 :=
sorry

end avg_seven_consecutive_integers_l599_599587


namespace john_and_mike_safe_weight_l599_599155

def weight_bench_max_support : ℕ := 1000
def safety_margin_percentage : ℕ := 20
def john_weight : ℕ := 250
def mike_weight : ℕ := 180

def safety_margin : ℕ := (safety_margin_percentage * weight_bench_max_support) / 100
def max_safe_weight : ℕ := weight_bench_max_support - safety_margin
def combined_weight : ℕ := john_weight + mike_weight
def weight_on_bar_together : ℕ := max_safe_weight - combined_weight

theorem john_and_mike_safe_weight :
  weight_on_bar_together = 370 := by
  sorry

end john_and_mike_safe_weight_l599_599155


namespace binom_200_200_eq_1_l599_599361

theorem binom_200_200_eq_1 : binom 200 200 = 1 :=
by
  have h : ∀ n : ℕ, binom n n = 1 := sorry
  exact h 200

end binom_200_200_eq_1_l599_599361


namespace evaluate_expression_l599_599745

theorem evaluate_expression (a : ℕ) (h : a = 4) : (a^a - a*(a-2)^a)^a = 1358954496 :=
by
  rw [h]  -- Substitute a with 4
  sorry

end evaluate_expression_l599_599745


namespace john_unanswered_problems_is_9_l599_599871

variables (x y z : ℕ)

theorem john_unanswered_problems_is_9 (h1 : 5 * x + 2 * z = 93)
                                      (h2 : 4 * x - y = 54)
                                      (h3 : x + y + z = 30) : 
  z = 9 :=
by 
  sorry

end john_unanswered_problems_is_9_l599_599871


namespace trigonometric_expression_l599_599722

/-- Given some trigonometric identities, prove the arithmetic expression involving them. -/
theorem trigonometric_expression :
  let sin_45 := (Real.sqrt 2 / 2) in
  let cot_60 := (Real.sqrt 3 / 3) in
  let tan_60 := Real.sqrt 3 in
  let sin_30 := 1 / 2 in
  let cot_45 := 1 in
  sin_45^2 + 3 * cot_60 - (2 * cot_45 / (tan_60 - 2 * sin_30)) = -1 / 2 :=
by
  -- Definitions from conditions section
  let sin_45 := (Real.sqrt 2 / 2)
  let cot_60 := (Real.sqrt 3 / 3)
  let tan_60 := Real.sqrt 3
  let sin_30 := 1 / 2
  let cot_45 := 1
  -- Substitute the values and simplify the expression
  sorry

end trigonometric_expression_l599_599722


namespace candy_bar_price_increase_l599_599670

-- Definitions and conditions
def original_weight (W : ℝ) := W
def original_price (P : ℝ) := P

def new_weight (W : ℝ) := 0.75 * W

def old_price_per_ounce (P W : ℝ) := P / W
def new_price_per_ounce (P W : ℝ) := P / (0.75 * W)

def percent_increase (old_price new_price : ℝ) :=
  ((new_price - old_price) / old_price) * 100

-- The statement to be proved
theorem candy_bar_price_increase (P W : ℝ) (hP : 0 < P) (hW : 0 < W) :
  percent_increase (old_price_per_ounce P W) (new_price_per_ounce P W) = 33.33 :=
by
  sorry

end candy_bar_price_increase_l599_599670


namespace find_angle_of_vectors_l599_599093

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := 
let cosine := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1 ^ 2 + a.2 ^ 2) * Real.sqrt (b.1 ^ 2 + b.2 ^ 2)) in
Real.arccos cosine * 180 / Real.pi

variable (a : ℝ × ℝ)
variable (h1 : a.1 - 2 * 1 + (a.1 - 2 * 1) + a.2 * (a.2 - 2 * 1) = 3) -- condition 1
variable (h2 : a.1 ^ 2 + a.2 ^ 2 = 1) -- condition 2
variable (b : ℝ × ℝ := (1, 1)) -- condition 3

theorem find_angle_of_vectors : angle_between_vectors a b = 135 := sorry

end find_angle_of_vectors_l599_599093


namespace percent_increase_visual_range_l599_599296

theorem percent_increase_visual_range (original new : ℝ) (h_original : original = 60) (h_new : new = 150) : 
  ((new - original) / original) * 100 = 150 :=
by
  sorry

end percent_increase_visual_range_l599_599296


namespace distribute_coins_l599_599709

theorem distribute_coins (x y : ℕ) (h₁ : x + y = 16) (h₂ : x^2 - y^2 = 16 * (x - y)) :
  x = 8 ∧ y = 8 :=
by {
  sorry
}

end distribute_coins_l599_599709


namespace ratio_of_volumes_l599_599732

theorem ratio_of_volumes (A B : ℝ) (h : (3 / 4) * A = (2 / 3) * B) : A / B = 8 / 9 :=
by
  sorry

end ratio_of_volumes_l599_599732


namespace coach_first_class_ratio_l599_599335

theorem coach_first_class_ratio (total_seats : ℕ) (first_class_seats : ℕ) (coach_class_seats : ℕ) 
  (h_total : total_seats = 387) (h_first_class : first_class_seats = 77) 
  (h_coach_class : coach_class_seats = total_seats - first_class_seats) : 
  coach_class_seats = 310 ∧ first_class_seats = 77 ∧ coach_class_seats / first_class_seats = 310 / 77 :=
by
  split
  sorry
  split
  sorry
  sorry

end coach_first_class_ratio_l599_599335


namespace multiply_103_97_l599_599008

theorem multiply_103_97 : 103 * 97 = 9991 := 
by
  sorry

end multiply_103_97_l599_599008


namespace ensure_positive_sums_l599_599771

theorem ensure_positive_sums (m n : ℕ) (grid : Fin m → Fin n → ℝ)
  (h : ∀ i j, grid i j ≠ 0) :
  ∃ sign_changes : Fin m → Bool × Fin n → Bool, 
    ∀ i, (∑ j, (if (sign_changes.1 i) then -1 else 1) * grid i j) > 0 ∧ 
         ∀ j, (∑ i, (if (sign_changes.2 j) then -1 else 1) * grid i j) > 0 :=
sorry

end ensure_positive_sums_l599_599771


namespace third_largest_in_L_l599_599270

def L : List ℕ := [1231, 2311, 2131, 1312, 1123, 3112]

theorem third_largest_in_L: (L.sort (· > ·)).nth 2 = some 2131 := by
  sorry

end third_largest_in_L_l599_599270


namespace circle_properties_l599_599442

theorem circle_properties (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * m * x - 4 * y + 5 * m = 0) →
  (m < 1 ∨ m > 4) ∧
  (m = -2 → ∃ d : ℝ, d = 2 * Real.sqrt (18 - 5)) :=
by
  sorry

end circle_properties_l599_599442


namespace perpendicular_lines_parallel_lines_l599_599817

-- Define the lines l1 and l2 in terms of a
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 6 = 0

def line2 (a : ℝ) (x y : ℝ) : Prop :=
  x + (a - 1) * y + a ^ 2 - 1 = 0

-- Define the perpendicular condition
def perp (a : ℝ) : Prop :=
  a * 1 + 2 * (a - 1) = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop :=
  a / 1 = 2 / (a - 1)

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) : perp a → a = 2 / 3 := by
  intro h
  sorry

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : parallel a → a = -1 := by
  intro h
  sorry

end perpendicular_lines_parallel_lines_l599_599817


namespace numDifferentSignals_l599_599898

-- Number of indicator lights in a row
def numLights : Nat := 6

-- Number of lights that light up each time
def lightsLit : Nat := 3

-- Number of colors each light can show
def numColors : Nat := 3

-- Function to calculate number of different signals
noncomputable def calculateSignals (n m k : Nat) : Nat :=
  -- Number of possible arrangements of "adjacent, adjacent, separate" and "separate, adjacent, adjacent"
  let arrangements := 4 + 4
  -- Number of color combinations for the lit lights
  let colors := k * k * k
  arrangements * colors

-- Theorem stating the total number of different signals is 324
theorem numDifferentSignals : calculateSignals numLights lightsLit numColors = 324 := 
by
  sorry

end numDifferentSignals_l599_599898


namespace rectangle_diagonal_segment_ratio_case1_rectangle_diagonal_segment_ratio_case2_l599_599320

-- Definitions and assumptions
theorem rectangle_diagonal_segment_ratio_case1 (a b : ℕ) (ha : a = 100) (hb : b = 99) :
  let diagonal_segments := diagonal_segments a b in
  ratio_sum_lengths_of_segments diagonal_segments = 1 :=
by
  -- Placeholder for the proof
  sorry

theorem rectangle_diagonal_segment_ratio_case2 (a b : ℕ) (ha : a = 101) (hb : b = 99) :
  let diagonal_segments := diagonal_segments a b in
  ratio_sum_lengths_of_segments diagonal_segments = 5000 / 4999 :=
by
  -- Placeholder for the proof
  sorry

end rectangle_diagonal_segment_ratio_case1_rectangle_diagonal_segment_ratio_case2_l599_599320


namespace game_result_l599_599867

def g (m : ℕ) : ℕ :=
  if m % 3 = 0 then 8
  else if m = 2 ∨ m = 3 ∨ m = 5 then 3
  else if m % 2 = 0 then 1
  else 0

def jack_sequence : List ℕ := [2, 5, 6, 4, 3]
def jill_sequence : List ℕ := [1, 6, 3, 2, 5]

def calculate_score (seq : List ℕ) : ℕ :=
  seq.foldl (λ acc x => acc + g x) 0

theorem game_result : calculate_score jack_sequence * calculate_score jill_sequence = 420 :=
by
  sorry

end game_result_l599_599867


namespace remainder_of_72nd_integers_div_by_8_is_5_l599_599178

theorem remainder_of_72nd_integers_div_by_8_is_5 (s : Set ℤ) (h₁ : ∀ x ∈ s, ∃ k : ℤ, x = 8 * k + r) 
  (h₂ : 573 ∈ (s : Set ℤ)) : 
  ∃ (r : ℤ), r = 5 :=
by
  sorry

end remainder_of_72nd_integers_div_by_8_is_5_l599_599178


namespace find_f_five_thirds_l599_599542

variable {R : Type*} [LinearOrderedField R]

-- Define the odd function and its properties
variable {f : R → R}
variable (oddf : ∀ x : R, f (-x) = -f x)
variable (propf : ∀ x : R, f (1 + x) = f (-x))
variable (val : f (- (1 / 3 : R)) = 1 / 3)

theorem find_f_five_thirds : f (5 / 3 : R) = 1 / 3 := by
  sorry

end find_f_five_thirds_l599_599542


namespace log_fn_monotonic_increasing_l599_599607

theorem log_fn_monotonic_increasing (a : ℝ) (h_a : 2 ≤ a ∧ a ≤ 4) : 
  ∀ x y : ℝ, x < 1 → y < 1 → x < y → log (1/2) (x^2 - a * x + 3) < log (1/2) (y^2 - a * y + 3) :=
sorry

end log_fn_monotonic_increasing_l599_599607


namespace value_of_f_l599_599059

noncomputable def f : ℕ+ × ℕ+ → ℕ+
| ⟨1, 1⟩ := ⟨1, sorry⟩
| ⟨m, n+1⟩ := ⟨(f (m, n)).val + 2, sorry⟩
| ⟨m+1, 1⟩ := ⟨2 * (f (m, 1)).val, sorry⟩

theorem value_of_f : f (2007, 2008) = ⟨2^2006 + 4014, sorry⟩ :=
sorry

end value_of_f_l599_599059


namespace simplification_of_fractional_equation_l599_599648

theorem simplification_of_fractional_equation (x : ℝ) : 
  (x / (3 - x) - 4 = 6 / (x - 3)) -> (x - 4 * (3 - x) = -6) :=
by
  sorry

end simplification_of_fractional_equation_l599_599648


namespace problem1_problem2_problem3_l599_599802

/-
  Problem 1: If a = 1 and b = 3, prove that the function f(x) = x^2 - 3x + ln x is monotonically 
  increasing on the intervals (0, 1/2) and (1, +∞).
-/
theorem problem1 (x : ℝ) (h₀ : x > 0) : 
  (x > 0 ∧ x < 1/2) ∨ (x > 1) → (2 * x - 3 + 1 / x > 0) := sorry

/-
  Problem 2: If b = 0 and the inequality f(x) ≤ 0 holds true over the interval [1, +∞), 
  prove that a ≤ -1/(2 * exp(1)).
-/
theorem problem2 (a : ℝ) (x : ℝ) (h₀ : x ≥ 1) (h₁ : a * x^2 + log x ≤ 0) : 
  a ≤ -1 / (2 * exp(1)) := sorry

/-
  Problem 3: When a = 1 and b > 9/2, and let the two zeros of the derivative 
  of the function f(x) be x1 and x2 (x1 < x2), prove that 
  f(x1) - f(x2) > 63/16 - 3*log 2.
-/
theorem problem3 (x1 x2 b : ℝ) (h₀ : 1 / b < x1 ∧ x1 < 1/4) (h₁ : 2 < x2 ∧ x2 < ∞) 
  (h₂ : b > 9 / 2) :
  ( (x1^2 - b*x1 + log x1) - (x2^2 - b*x2 + log x2) > 63 / 16 - 3 * log 2 ) := sorry

end problem1_problem2_problem3_l599_599802


namespace Milly_math_homework_time_l599_599569

theorem Milly_math_homework_time (M : ℝ) 
  (H_geo : ℝ := M / 2)
  (H_sci : ℝ := (M + M / 2) / 2)
  (H_total : M + H_geo + H_sci = 135) : 
  M = 72 := 
begin
  sorry
end

end Milly_math_homework_time_l599_599569


namespace harmonic_division_property_l599_599620

noncomputable def intersect_at (A B C D : Point) (O : Point) (l : Line) (P Q R S : Point) : Prop := 
  ∃ k1 k2 : ℝ, 
  A = k1 • B + (1 - k1) • O ∧
  D = k2 • C + (1 - k2) • O ∧
  ∃ line_AC line_BD : Line, 
  intersects line_AC P ∧ 
  intersects line_BD Q ∧ 
  ∃ conic : Conic_section, 
  passes_through conic A ∧ 
  passes_through conic B ∧ 
  passes_through conic C ∧ 
  passes_through conic D ∧
  intersects l R ∧ 
  intersects l S

theorem harmonic_division_property (A B C D O P Q R S : Point) (l : Line)
  (h1 : exists_line_intersect AB CD O)
  (h2 : exists_line_intersect_AC_BD AC BD P Q)
  (h3 : conic_passing_through A B C D R S)
  : (1 / distance O P) + (1 / distance O Q) = (1 / distance O R) + (1 / distance O S) := 
by 
  sorry

end harmonic_division_property_l599_599620


namespace fibonacci_solution_l599_599161

def fibonacci (n : ℕ) : ℕ :=
nat.rec_on n 1 (λ n fib_n,
  match n with
  | 0     := 1
  | (n+1) := fib_n + (fibonacci n)
  end)

theorem fibonacci_solution : ∃ x : ℝ, (x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2) ∧ x ^ 2010 = fibonacci 2009 * x + fibonacci 2008 :=
by
  sorry

end fibonacci_solution_l599_599161


namespace sum_inequality_l599_599885

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 3) :
    (a / (b + c) + b / (a + c) + c / (a + b) + 
    sqrt(2) * (sqrt (a / (b + c)) + sqrt (b / (a + c)) + sqrt (c / (a + b)))) ≥ 9 / 2 :=
  by
  sorry

end sum_inequality_l599_599885


namespace range_of_m_l599_599403

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), x > -5 ∧ x ≤ m + 1) ∧ (∀ x, x > -5 → x ≤ m + 1 → x = -4 ∨ x = -3 ∨ x = -2) →
  (-3 ≤ m ∧ m < -2) :=
sorry

end range_of_m_l599_599403


namespace intersection_point_l599_599874

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := (x^2 - 4*x + 4) / (2*x - 6)
def g (x : ℝ) := (-2*x^2 + 6*x - 4) / (x - 3)

-- Conditions:
-- The graphs of f(x) and g(x) have the same vertical asymptote (x = 3)
-- The oblique asymptotes of f(x) and g(x) are perpendicular and intersect at the origin
-- The graphs of f(x) and g(x) have an intersection point at x = 1

-- The proof will show that the other intersection point is at x = 2
-- and evaluates f(2)

theorem intersection_point (h1 : f 1 = g 1) : 
  ∃ x : ℝ, x ≠ 1 ∧ f x = g x ∧ x = 2 ∧ f 2 = 0 := 
by
  sorry

end intersection_point_l599_599874


namespace sequence_sum_15_22_31_l599_599813

def sequence_sum (n : ℕ) : ℤ :=
  let f k := if even k then -4 * (k/2) + 1 else 4 * (k/2) + 1
  ∑ k in range(1, n+1), f k

theorem sequence_sum_15_22_31 :
  sequence_sum 15 + sequence_sum 22 - sequence_sum 31 = -76 :=
by sorry

end sequence_sum_15_22_31_l599_599813


namespace sqrt_mult_pow_l599_599362

theorem sqrt_mult_pow (a : ℝ) (h_nonneg : 0 ≤ a) : (a^(2/3) * a^(1/5)) = a^(13/15) := by
  sorry

end sqrt_mult_pow_l599_599362


namespace find_x_add_2y_l599_599830

theorem find_x_add_2y (x y : ℝ) (h1 : 2^x = 9) (h2 : log 2 (8 / 3) = y) :
  x + 2 * y = 6 :=
sorry

end find_x_add_2y_l599_599830


namespace probability_of_pair_l599_599302

def deck50 : list ℕ := list.repeat 1 5 ++ list.repeat 2 5 ++ list.repeat 3 5 ++ list.repeat 4 5 ++
                      list.repeat 5 5 ++ list.repeat 6 5 ++ list.repeat 7 5 ++ list.repeat 8 5 ++
                      list.repeat 9 5 ++ list.repeat 10 5

-- Assume that we remove two cards each from two different numbers a and b
-- Remove a and b from the deck with the list.drop and list.filter function
noncomputable def remove_two (n : ℕ) (deck : list ℕ) : list ℕ :=
  (deck.filter (λ x, x ≠ n)).drop 2

noncomputable def new_deck (a b : ℕ) (deck : list ℕ) : list ℕ :=
  remove_two b (remove_two a deck)

-- Calculating the combinations of pairs that can be drawn from the new deck of 46 cards
def combinations (n : ℕ) (k : ℕ) : ℕ :=
  (list.range n).comb k |>.length

-- Count the pairs of the remaining cards
noncomputable def count_pairs (deck : list ℕ) : ℕ :=
  let counts := (deck.eraseDups.map (λ n, (deck.count (λ x, x = n))))
  counts.filter (λ x, x >= 2).map (λ x, combinations x 2).sum

-- Theorem statement
theorem probability_of_pair (a b : ℕ) (h : a ≠ b) :
  let new_deck := new_deck a b deck50 in
  let total_pairs := combinations new_deck.length 2 in
  let favorable_pairs := count_pairs new_deck in
  let gcd := Int.gcd favorable_pairs.total_pairs in
  let simplified_numerator := favorable_pairs / gcd in
  let simplified_denominator := total_pairs / gcd in
  simplified_numerator + simplified_denominator = 223 :=
by
  sorry

end probability_of_pair_l599_599302


namespace intersection_M_N_l599_599024

def M : Set ℝ := { x | x^2 ≤ 4 }
def N : Set ℝ := { x | Real.log x / Real.log 2 ≥ 1 }

theorem intersection_M_N : M ∩ N = {2} := by
  sorry

end intersection_M_N_l599_599024


namespace orthogonal_projection_magnitude_l599_599473

noncomputable def a : ℝ × ℝ := (-2, 1)
noncomputable def b : ℝ × ℝ := (3, 0)

theorem orthogonal_projection_magnitude : 
  let p := ((a.1 * b.1 + a.2 * b.2 : ℝ) / (b.1 * b.1 + b.2 * b.2)) 
  = -2 :=
by
  sorry

end orthogonal_projection_magnitude_l599_599473


namespace hall_length_l599_599307

theorem hall_length
  (width : ℝ)
  (stone_length : ℝ)
  (stone_width : ℝ)
  (num_stones : ℕ)
  (h₁ : width = 15)
  (h₂ : stone_length = 0.8)
  (h₃ : stone_width = 0.5)
  (h₄ : num_stones = 1350) :
  ∃ length : ℝ, length = 36 :=
by
  sorry

end hall_length_l599_599307


namespace problem_f_2_eval_l599_599483

def f (n : ℕ) : ℚ := (List.range (2*n + 2)).map (λ k => 1 / (k + 1)).sum

theorem problem_f_2_eval : f 2 = 137 / 60 := sorry

end problem_f_2_eval_l599_599483


namespace log_base_five_of_expression_l599_599370

theorem log_base_five_of_expression : 
  ∀ (a b c d : ℝ), a = 125 → b = (25 : ℝ)^(1/3) → c = 3 → d = 2/3 → log 5 (a * b) = (c + d) → log 5 (a * b) = 11 / 3 :=
by 
  intros a b c d ha hb hc hd hlog 
  rw [ha, hb] at hlog
  have h1 : a = 5^3 := by norm_num [ha]
  have h2 : b = 5^(2/3) := by norm_num [hb]
  have h3 : a * b = 5^3 * 5^(2/3) := by rw [←h1, ←h2]; norm_num
  have h4 : log 5 (a * b) = log 5 (5^(3 + 2/3)) := by rw [hlog, ←h3]; norm_num
  rw [logb_pow, add_comm, hc, hd] at h4
  exact hlog  

end log_base_five_of_expression_l599_599370


namespace max_intersections_in_triangle_l599_599629

theorem max_intersections_in_triangle 
  (ABC : Type)
  (A B C : ABC)
  (is_initial_equilateral_triangle : ∀ {P Q R : ABC}, P = A → Q = B → R = C → ∃ (l1 l2 l3 : Set (Set ABC)), ∀ (i j k : Set ABC), i = l1 → j = l2 → k = l3 → is_equilateral_triangle i j k)
  (number_of_turns : ℕ)
  (number_of_turns_eq : number_of_turns = 300) : 
  ∃ (n_intersections : ℕ), n_intersections = nat.choose (3 + number_of_turns) 2 ∧ n_intersections = 45853 :=
by
  sorry

end max_intersections_in_triangle_l599_599629


namespace distinct_configurations_22_marked_l599_599186

structure Grid (n m : ℕ) :=
(rows : Fin n → Fin n → Bool)
(cols : Fin m → Fin m → Bool)
(cells_marked : Fin n × Fin m → Bool)
(two_marked_per_row : ∀ i : Fin n, ∑ j : Fin m, if cells_marked (i, j) then 1 else 0 = 2)
(two_marked_per_column : ∀ j : Fin m, ∑ i : Fin n, if cells_marked (i, j) then 1 else 0 = 2)
(marked_cells_count : ∑ i : Fin n, ∑ j : Fin m, if cells_marked (i, j) then 1 else 0 = 22)

def is_equivalent_configuration (g1 g2 : Grid 11 11) : Prop :=
  ∃ σ : Perm (Fin 11), ∃ τ : Perm (Fin 11),
  ∀ i j, g1.cells_marked (i, j) = g2.cells_marked (σ i, τ j)

noncomputable def distinct_configurations_count : ℕ :=
  sorry

theorem distinct_configurations_22_marked :
  ∀ (g : Grid 11 11), distinct_configurations_count = 14 := 
sorry

end distinct_configurations_22_marked_l599_599186


namespace find_f_five_thirds_l599_599537

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = - f x
def functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (-x)
def specific_value (f : ℝ → ℝ) := f (-1/3) = 1/3

theorem find_f_five_thirds
  (hf_odd : odd_function f)
  (hf_fe : functional_equation f)
  (hf_value : specific_value f) :
  f (5 / 3) = 1 / 3 := by
  sorry

end find_f_five_thirds_l599_599537


namespace seq_pos_l599_599617

noncomputable def seq : ℕ → ℝ
| 0 := -1
| n + 1 :=
  let sum := ∑ k in finset.range (n + 1), (seq k) / (n + 1 - k + 1) in
  have hsum_eq : sum = 0 := by sorry,
  sorry

theorem seq_pos (n : ℕ) : 0 < seq (n + 1) :=
by induction n with n ih;
   have h := (seq).cases_on (by exact hsum_eq);
   sorry

end seq_pos_l599_599617


namespace num_equilateral_triangles_l599_599030

/-- Each point in the octagonal lattice is one unit from its nearest neighbor. --/
structure OctagonalLattice :=
  (points : Finset (ℕ × ℕ))
  (distance_one : ∀ {p1 p2 : ℕ × ℕ}, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → dist p1 p2 = 1 → ∃ i j : ℕ, abs (p1.1 - p2.1) = 1 ∧ abs (p1.2 - p2.2) = 1)

/-- Define an OctagonalLattice as given conditions. --/
def exampleLattice : OctagonalLattice := sorry

/-- Prove the number of equilateral triangles formed by vertices in the octagonal lattice is 8. --/
theorem num_equilateral_triangles (L : OctagonalLattice) : 
  (∃ eq_triangles : Finset (Finset (ℕ × ℕ)), eq_triangles.card = 8 ∧ ∀ tri ∈ eq_triangles, is_equilateral_triangle L.points tri) := 
sorry

/-- A definition for determining if a triangle is equilateral --/
def is_equilateral_triangle (points : Finset (ℕ × ℕ)) (triangle : Finset (ℕ × ℕ)) : Prop :=
  ∃ (a b c : ℕ × ℕ), a ∈ points ∧ b ∈ points ∧ c ∈ points ∧
  a ∈ triangle ∧ b ∈ triangle ∧ c ∈ triangle ∧
  dist a b = dist b c ∧ dist b c = dist c a


end num_equilateral_triangles_l599_599030


namespace molecular_weight_of_dinitrogen_pentoxide_l599_599267

/--
  Prove that the molecular weight of Dinitrogen pentoxide (N2O5) is approximately 108.02 g/mol
  given the atomic weights:
  - Atomic weight of nitrogen (N) = 14.01 g/mol
  - Atomic weight of oxygen (O) = 16.00 g/mol
-/
theorem molecular_weight_of_dinitrogen_pentoxide :
  let atomic_weight_N := 14.01
  let atomic_weight_O := 16.00
  let molecular_weight := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  molecular_weight ≈ 108.02 := 
by 
  sorry
  
end molecular_weight_of_dinitrogen_pentoxide_l599_599267


namespace question1_question2_question3_l599_599797

-- Conditions: Equation of line l and circle O
def line_l (m : ℝ) : ℝ → ℝ → Prop := 
λ x y, x + m * y + 2 * real.sqrt 3 = 0

def circle_O (r : ℝ) : ℝ → ℝ → Prop := 
λ x y, x^2 + y^2 = r^2

-- Question 1: Prove r ≥ 2sqrt(3) for intersection
theorem question1 (r : ℝ) : (∀ m : ℝ, ∃ x y : ℝ, line_l m x y ∧ circle_O r x y) ↔ (r ≥ 2 * real.sqrt 3) :=
sorry

-- Question 2: Prove range of chord lengths for r=5
theorem question2 (r : ℝ) : 
  (∀ m : ℝ, r = 5 → ∃ l : ℝ, (l = 2 * real.sqrt 13 ∨ l = 10)) ↔ 
  (2 * real.sqrt 13 ≤ 10) :=
sorry

-- Question 3: Prove circle with diameter P'Q' passes through fixed points for r=1
theorem question3 : 
  (r = 1 → ∀ P Q M : ℝ × ℝ, circle_O 1 P.1 P.2 ∧ circle_O 1 Q.1 Q.2 ∧ circle_O 1 M.1 M.2 ∧
   P.2 = Q.2 = 0 ∧ P ≠ Q → 
       let P' := (3, 4 * M.2 / (M.1 + 1)),
           Q' := (3, 2 * M.2 / (M.1 - 1))
       in (circle ((3, 1 - 3 * M.1 / M.2)) ((3 - M.1) / M.2)).passes_through (3 + 2 * real.sqrt 2, 0) ∧ 
            (circle ((3, 1 - 3 * M.1 / M.2)) ((3 - M.1) / M.2)).passes_through (3 - 2 * real.sqrt2, 0)) :=
sorry

end question1_question2_question3_l599_599797


namespace average_weight_decrease_l599_599218

/-- 
Given that the average weight of 5 students changes when a student weighing 72 kg 
is replaced by a student weighing 12 kg,
prove that the decrease in the average weight is 12 kg.
 -/
theorem average_weight_decrease :
  let A : ℝ := sorry,   -- representing the initial average weight
  let total_initial_weight : ℝ := 5 * A, 
  let decrease_in_weight : ℝ := 72 - 12, 
  let total_new_weight : ℝ := 5 * A - decrease_in_weight,
  let new_average_weight : ℝ := total_new_weight / 5,
  (A - new_average_weight) = 12 := sorry

end average_weight_decrease_l599_599218


namespace suzanna_distance_40_minutes_l599_599209

-- Define the constant rate condition
def constant_rate (time: ℝ) (distance: ℝ) : Prop := ∀ t d, t = time → d = distance → d / t = distance / time

-- Define the conditions
def suzanna_conditions : Prop := 
  constant_rate 8 1.5

-- Prove that Suzanna rides 7.5 miles in 40 minutes given the conditions
theorem suzanna_distance_40_minutes : 
  suzanna_conditions → (40 / 8) * 1.5 = 7.5 :=
by
  intros,
  sorry

end suzanna_distance_40_minutes_l599_599209


namespace abs_eq_5_iff_l599_599101

   theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 :=
   by
     sorry
   
end abs_eq_5_iff_l599_599101


namespace other_root_of_quadratic_l599_599120

theorem other_root_of_quadratic (m : ℝ) (h : (2:ℝ) * (t:ℝ) = -6 ): 
  ∃ t, t = -3 :=
by
  sorry

end other_root_of_quadratic_l599_599120


namespace interval_behavior_l599_599223

noncomputable def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (x) = f (-x)

noncomputable def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x : ℝ, f (x) = f (x + p)

noncomputable def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f(x) > f(y)

noncomputable def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f(x) < f(y)

theorem interval_behavior (f : ℝ → ℝ) :
  is_even f →
  (∀ x : ℝ, f (x) = f (x + 2)) →
  (∀ x y ∈ set.Icc (-1:ℝ) (0:ℝ), x < y → f(x) > f(y)) →
  ∀ x y ∈ set.Icc (2:ℝ) (3:ℝ), x < y → f(x) < f(y) :=
by
  sorry

end interval_behavior_l599_599223


namespace average_speed_proof_l599_599669

-- Define conditions as constants or parameters
def speed_segment1 := 35 -- kph
def distance_segment1 := 30 -- km

def speed_segment2 := 45 -- kph
def distance_segment2 := 35 -- km

def speed_segment3 := 55 -- kph
def time_segment3 := 0.5 -- hours

def speed_segment4 := 50 -- kph
def time_segment4 := 1 / 3 -- hours

def speed_segment5 := 65 -- kph
def time_segment5 := 2 / 3 -- hours

-- Define the average speed calculation proof statement
theorem average_speed_proof : 
  let distance_segment3 := speed_segment3 * time_segment3 in
  let distance_segment4 := speed_segment4 * time_segment4 in
  let distance_segment5 := speed_segment5 * time_segment5 in
  let total_distance := distance_segment1 + distance_segment2 + distance_segment3 + distance_segment4 + distance_segment5 in
  let time_segment1 := distance_segment1 / speed_segment1 in
  let time_segment2 := distance_segment2 / speed_segment2 in
  let total_time := time_segment1 + time_segment2 + time_segment3 + time_segment4 + time_segment5 in
  let average_speed := total_distance / total_time in
  average_speed = 48.65 := 
by
  sorry

end average_speed_proof_l599_599669


namespace number_of_lattice_points_eq_30_l599_599860

theorem number_of_lattice_points_eq_30 : 
  {p : ℤ × ℤ × ℤ // p.1 * p.1 + p.2.1 * p.2.1 + p.2.2 * p.2.2 = 16}.to_finset.card = 30 := 
by 
  sorry

end number_of_lattice_points_eq_30_l599_599860


namespace probability_of_selecting_odd_card_l599_599251

theorem probability_of_selecting_odd_card :
  let cards := {3, 4, 5, 6, 7}
  let odd_numbers := {3, 5, 7}
  let total_cards := cards.card
  let odd_cards := odd_numbers.card
  (odd_cards / total_cards : ℚ) = 3 / 5 := by
  sorry

end probability_of_selecting_odd_card_l599_599251


namespace probability_event_l599_599818

noncomputable def event_area (r : ℝ) : ℝ :=
  (π * r ^ 2) / 4

noncomputable def probability_above_curve (r : ℝ) : ℝ :=
  1 - event_area r

theorem probability_event (r : ℝ) (hr : 0 ≤ r) (hsr : r = 1) :
  probability_above_curve r = 1 - π / 4 :=
by
  sorry

end probability_event_l599_599818


namespace well_digging_expenditure_l599_599277

theorem well_digging_expenditure :
  ∀ (h d cost_per_m3 : ℝ)
  (π : ℝ),
  h = 14 →
  d = 3 →
  cost_per_m3 = 15 →
  (π = 3.14159) → -- approximate value of π
  let r := d / 2 in
  let V := π * (r * r) * h in
  let expenditure := V * cost_per_m3 in
  expenditure ≈ 1484.4 :=
by
  intros h d cost_per_m3 π h_cond d_cond cost_cond pi_cond r V expenditure
  sorry -- proof to be filled in

end well_digging_expenditure_l599_599277


namespace find_number_l599_599742

theorem find_number (x : ℕ) (h : (18 / 100) * x = 90) : x = 500 :=
sorry

end find_number_l599_599742


namespace eccentricity_of_ellipse_l599_599796

theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) (c : ℝ)
  (ellipse_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (focus_relation : a = 2 * c)
  (acute_triangle : ∀ F1 F2 B : ℝ × ℝ, acute △F1 F2 B) : 
  let e := c / a in
  e = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l599_599796


namespace XiaoXing_ranking_l599_599295

theorem XiaoXing_ranking :
  let score := 81
  let means := [71, 75, 81, 85]
  let std_devs := [4.9, 2.1, 3.6, 4.3]
  -- Scores above mean and their indices
  let above_mean := (score - 71, 0) :: (score - 75, 1) 
  -- Index of the maximum difference from the mean
  let first_project := above_mean.minBy (λ (diff, index), diff) -- Index 1 for "Challenge the Living Room"

  first_project = 1 ∧ last_project = 3 := -- Index 3 for "Creative One Hundred Percent"
by
  sorry

end XiaoXing_ranking_l599_599295


namespace A_plus_2B_plus_4_is_perfect_square_l599_599557

theorem A_plus_2B_plus_4_is_perfect_square (n : ℕ) (hn : 0 < n) :
  let A := (4 / 9) * (10^(2*n) - 1)
  let B := (8 / 9) * (10^n - 1)
  ∃ k : ℚ, (A + 2 * B + 4) = k^2 :=
by
  let A := (4 / 9) * (10^(2*n) - 1)
  let B := (8 / 9) * (10^n - 1)
  use ((2/3) * (10^n + 2))
  sorry

end A_plus_2B_plus_4_is_perfect_square_l599_599557


namespace Isabel_reading_pages_l599_599865

def pages_of_math_homework : ℕ := 2
def problems_per_page : ℕ := 5
def total_problems : ℕ := 30

def math_problems : ℕ := pages_of_math_homework * problems_per_page
def reading_problems : ℕ := total_problems - math_problems

theorem Isabel_reading_pages : (reading_problems / problems_per_page) = 4 :=
by
  sorry

end Isabel_reading_pages_l599_599865


namespace PeytonManning_total_distance_l599_599190

noncomputable def PeytonManning_threw_distance : Prop :=
  let throw_distance_50 := 20
  let throw_times_sat := 20
  let throw_times_sun := 30
  let total_distance := 1600
  ∃ R : ℚ, 
    let throw_distance_80 := R * throw_distance_50
    let distance_sat := throw_distance_50 * throw_times_sat
    let distance_sun := throw_distance_80 * throw_times_sun
    distance_sat + distance_sun = total_distance

theorem PeytonManning_total_distance :
  PeytonManning_threw_distance := by
  sorry

end PeytonManning_total_distance_l599_599190


namespace max_campaign_making_animals_prime_max_campaign_making_animals_nine_l599_599982

theorem max_campaign_making_animals_prime (n : ℕ) (h_prime : Nat.Prime n) (h_ge : n ≥ 3) : 
  ∃ k, k = (n - 1) / 2 :=
by
  sorry

theorem max_campaign_making_animals_nine : ∃ k, k = 4 :=
by
  sorry

end max_campaign_making_animals_prime_max_campaign_making_animals_nine_l599_599982


namespace number_of_true_propositions_is_3_l599_599947

-- Definitions of the propositions
def prop1 : Prop := ∃(T1 T2 : Triangle), (T1.area = T2.area ∧ ¬ T1 ≅ T2)
def prop2 (x y : ℝ) : Prop := (|x| + |y| = 0) → (x * y = 0)
def prop3 (a b c : ℝ) : Prop := (a > b) → (a + c ≤ b + c)
def prop4 (d1 d2 : Line) : Prop := (¬ Perpendicular d1 d2) → ¬ Rectangle d1 d2

-- Mathematically equivalent proof problem in Lean 4
theorem number_of_true_propositions_is_3 :
  (∃ (T1 T2 : Triangle), (T1.area = T2.area ∧ ¬ T1 ≅ T2)) ∧
  (∀ (x y : ℝ), (|x| + |y| = 0) → (x * y = 0)) ∧
  (¬ (∀ (a b c : ℝ), (a > b) → (a + c ≤ b + c))) ∧
  (∀ (d1 d2 : Line), (¬ Perpendicular d1 d2) → ¬ Rectangle d1 d2) :=
by
  sorry

end number_of_true_propositions_is_3_l599_599947


namespace probability_both_meat_given_same_l599_599895

open ProbabilityTheory

-- Definition of the problem conditions
def total_dumplings : Finset (Fin 5) := Finset.univ
def meat_dumplings : Finset (Fin 5) := {0, 1} -- using the first 2 as meat filled
def red_bean_paste_dumplings : Finset (Fin 5) := {2, 3, 4} -- the remaining are red bean filled

def event_same_filling (x y : Fin 5) : Prop :=
  (x ∈ meat_dumplings ∧ y ∈ meat_dumplings) ∨ (x ∈ red_bean_paste_dumplings ∧ y ∈ red_bean_paste_dumplings)

def event_both_meat (x y : Fin 5) : Prop :=
  x ∈ meat_dumplings ∧ y ∈ meat_dumplings

-- Probability calculations
noncomputable def probability_same_filling : ℚ :=
  (Finset.card ((total_dumplings.product total_dumplings).filter (λ p, event_same_filling p.1 p.2))).toRat /
  (Finset.card (total_dumplings.product total_dumplings)).toRat

noncomputable def probability_both_meat : ℚ :=
  (Finset.card ((total_dumplings.product total_dumplings).filter (λ p, event_both_meat p.1 p.2))).toRat /
  (Finset.card (total_dumplings.product total_dumplings)).toRat

noncomputable def conditional_probability_both_meat_given_same_filling : ℚ :=
  probability_both_meat / probability_same_filling

-- The main theorem statement
theorem probability_both_meat_given_same : 
  conditional_probability_both_meat_given_same_filling = 1 / 4 :=
by
  sorry

end probability_both_meat_given_same_l599_599895


namespace savings_calculation_l599_599696

def price_per_window : ℕ := 120
def discount_offer (n : ℕ) : ℕ := if n ≥ 10 then 2 else 0

def george_needs : ℕ := 9
def anne_needs : ℕ := 11

def cost (n : ℕ) : ℕ :=
  let free_windows := discount_offer n
  (n - free_windows) * price_per_window

theorem savings_calculation :
  let total_separate_cost := cost george_needs + cost anne_needs
  let total_windows := george_needs + anne_needs
  let total_cost_together := cost total_windows
  total_separate_cost - total_cost_together = 240 :=
by
  sorry

end savings_calculation_l599_599696


namespace tan_angle_BAC_proof_l599_599952

noncomputable def tan_angle_BAC : ℝ :=
  let ω_B := sorry -- Circle centered at B
  let ω_C := sorry -- Circle centered at C
  let B := sorry -- Point B
  let C := sorry -- Point C
  let A := sorry -- Point A
  let T_B := sorry -- Point on ω_B where AT_B is tangent to ω_B
  let T_C := sorry -- Point on ω_C where AT_C is tangent to ω_C
  /\ (ω_B ∩ ω_C ≠ ∅ → ω_B ⊥ ω_C) -- ω_B and ω_C are orthogonal
  /\ (area_ΔABC = 20) -- Area of triangle
  /\ (tangent(AT_B, ω_B) = true) -- AT_B is tangent to ω_B
  /\ (tangent(AT_C, ω_C) = true) -- AT_C is tangent to ω_C
  /\ (AT_B = 7) -- Length of tangent from A to T_B
  /\ (AT_C = 11) -- Length of tangent from A to T_C
  → tan (angle(B, A, C)) = 8/17 -- Prove tan of angle BAC

theorem tan_angle_BAC_proof : 
  tan_angle_BAC :=
sorry

end tan_angle_BAC_proof_l599_599952


namespace diagonals_divide_polygon_into_triangles_l599_599603

theorem diagonals_divide_polygon_into_triangles (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℕ, k = n - 2 ∧
  (∀ (v : ℕ), v = 1 → 
   divides_into_triangles v n k) :=
sorry

end diagonals_divide_polygon_into_triangles_l599_599603


namespace a1d1_a2d2_a3d3_eq_neg1_l599_599164

theorem a1d1_a2d2_a3d3_eq_neg1 (a1 a2 a3 d1 d2 d3 : ℝ) (h : ∀ x : ℝ, 
  x^8 - x^6 + x^4 - x^2 + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 + 1)) : 
  a1 * d1 + a2 * d2 + a3 * d3 = -1 := 
sorry

end a1d1_a2d2_a3d3_eq_neg1_l599_599164


namespace sum_max_min_values_of_g_l599_599176

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2 * x - 8| + 3

theorem sum_max_min_values_of_g : (∀ x, 1 ≤ x ∧ x ≤ 7 → g x = 15 - 2 * x ∨ g x = 5) ∧ 
      (g 1 = 13 ∧ g 5 = 5)
      → (13 + 5 = 18) :=
by
  sorry

end sum_max_min_values_of_g_l599_599176


namespace _l599_599761

noncomputable theorem differentiable_function_inequality {f : ℝ → ℝ} (a : ℝ) (h_diff : ∀ x, differentiable_at ℝ f x)
(h_cond : ∀ x, (x - a) * (deriv f x) ≥ 0) : ∀ x, f x ≥ f a :=
sorry

end _l599_599761


namespace unique_solution_l599_599053

noncomputable def uniquely_solvable (a : ℝ) : Prop :=
  ∀ x : ℝ, a > 0 ∧ a ≠ 1 → ∃! x, a^x = (Real.log x / Real.log (1/4))

theorem unique_solution (a : ℝ) : a > 0 ∧ a ≠ 1 → uniquely_solvable a :=
by sorry

end unique_solution_l599_599053


namespace inequality_iff_positive_l599_599559

variable (x y : ℝ)

theorem inequality_iff_positive :
  x + y > abs (x - y) ↔ x > 0 ∧ y > 0 :=
sorry

end inequality_iff_positive_l599_599559


namespace moving_point_on_line_segment_l599_599430

theorem moving_point_on_line_segment
  (F1 F2 M : Point)
  (hF1F2_dist : dist F1 F2 = 16)
  (hMF1_MF2_sum : dist M F1 + dist M F2 = 16) :
  M ∈ line_segment F1 F2 := 
sorry

end moving_point_on_line_segment_l599_599430


namespace pyramid_width_exceeds_height_l599_599212

noncomputable def height : ℝ := 520
noncomputable def sum_height_width : ℝ := 1274

theorem pyramid_width_exceeds_height :
  ∃ (x : ℝ), (520 + (520 + x) = 1274) ∧ (x = 234) :=
by
  use 234
  split
  · sorry
  · sorry

end pyramid_width_exceeds_height_l599_599212


namespace sparrow_population_decline_l599_599746

theorem sparrow_population_decline {P : ℕ} (initial_year : ℕ) (initial_population : ℕ) (decrease_by_half : ∀ year, year ≥ initial_year →  init_population * (1 / (2 ^ (year - initial_year))) < init_population / 20) :
  ∃ year, year ≥ initial_year + 5 ∧ init_population * (1 / (2 ^ (year - initial_year))) < init_population / 20 :=
by
  sorry

end sparrow_population_decline_l599_599746


namespace product_telescope_l599_599646

theorem product_telescope : ((1 + (1 / 1)) * 
                             (1 + (1 / 2)) * 
                             (1 + (1 / 3)) * 
                             (1 + (1 / 4)) * 
                             (1 + (1 / 5)) * 
                             (1 + (1 / 6)) * 
                             (1 + (1 / 7)) * 
                             (1 + (1 / 8)) * 
                             (1 + (1 / 9)) * 
                             (1 + (1 / 10))) = 11 := 
by
  sorry

end product_telescope_l599_599646


namespace remaining_weight_correct_l599_599525

noncomputable def total_weight_coconut_macaroons : ℕ := 12 * 5
noncomputable def number_coconut_macaroons_per_bag : ℕ := 12 / 4
noncomputable def weight_coconut_macaroons_per_bag : ℕ := number_coconut_macaroons_per_bag * 5
noncomputable def remaining_weight_coconut_macaroons : ℕ := total_weight_coconut_macaroons - weight_coconut_macaroons_per_bag

noncomputable def total_weight_almond_macaroons : ℕ := 8 * 8
noncomputable def number_almond_macaroons_per_bag : ℕ := 8 / 2
noncomputable def weight_almond_macaroons_per_bag : ℕ := number_almond_macaroons_per_bag * 8
noncomputable def weight_almond_macaroons_eaten_by_steve : ℕ := weight_almond_macaroons_per_bag / 2
noncomputable def remaining_weight_almond_macaroons : ℕ := total_weight_almond_macaroons - weight_almond_macaroons_eaten_by_steve

noncomputable def total_remaining_weight : ℕ := remaining_weight_coconut_macaroons + remaining_weight_almond_macaroons

theorem remaining_weight_correct : total_remaining_weight = 93 :=
by {
  unfold total_remaining_weight remaining_weight_coconut_macaroons remaining_weight_almond_macaroons,
  exact dec_trivial,
}

end remaining_weight_correct_l599_599525


namespace each_person_pays_l599_599210

def numPeople : ℕ := 6
def rentalDays : ℕ := 4
def weekdayRate : ℕ := 420
def weekendRate : ℕ := 540
def numWeekdays : ℕ := 2
def numWeekends : ℕ := 2

theorem each_person_pays : 
  (numWeekdays * weekdayRate + numWeekends * weekendRate) / numPeople = 320 :=
by
  sorry

end each_person_pays_l599_599210


namespace common_ratio_geom_sequence_l599_599398

noncomputable def geometric_sum (a₁ q : ℚ) (n : ℕ) : ℚ :=
a₁ * (1 - q^n) / (1 - q)

theorem common_ratio_geom_sequence (a₁ : ℚ) (q : ℚ) :
  S_1 = a₁ → 2 * S_2 = a₁ * (1 - q^2) / (1 - q) →
  3 * S_3 = a₁ * (1 - q^3) / (1 - q) →
  (a₁ + 3 * geometric_sum a₁ q 3) = 2 * (2 * geometric_sum a₁ q 2) →
  q = 1 / 3 :=
begin
  -- proof would go here
  sorry
end

end common_ratio_geom_sequence_l599_599398


namespace angle_ABC_of_unshaded_sector_l599_599409

-- Define the variables used in conditions
def cone_radius : ℝ := 10
def cone_volume : ℝ := 300 * Real.pi

-- Define the final statement we need to prove: the measure of angle ABC (in degrees)
def measure_angle_ABC : ℝ := 92.05

-- The theorem statement combining the given conditions and the required proof
theorem angle_ABC_of_unshaded_sector (r h l original_circumference used_angle unused_angle : ℝ) :
  r = cone_radius →
  V = (1/3) * Real.pi * r^2 * h →
  h = 9 →
  l = Real.sqrt (r^2 + h^2) →
  r + h + l > 0 →  -- Pythagorean theorem condition to ensure l is computed
  original_circumference = 2 * Real.pi * l →
  used_angle = (cone_circumference / original_circumference) * 360 →
  unused_angle = 360 - used_angle →
  unused_angle = measure_angle_ABC :=
by
  intros
  sorry  -- Here would be the proof steps, but we skip it as just the statement is required

end angle_ABC_of_unshaded_sector_l599_599409


namespace chebyshev_polynomial_properties_l599_599579

def chebyshev_polynomial (n : ℕ) : Polynomial ℤ :=
  sorry -- Definition of Chebyshev polynomial Tₙ over ℤ

def f_n (n : ℕ) (x : ℝ) : Polynomial ℝ :=
  2 * (chebyshev_polynomial n).comp (Polynomial.C (x / 2))

theorem chebyshev_polynomial_properties (n : ℕ) :
  let f := f_n n in
  f.leading_coeff = 1 ∧
  (∀ i, i ≤ f.nat_degree → (f.coeff i).denom = 1) ∧
  f.nat_degree = n :=
by
  sorry

end chebyshev_polynomial_properties_l599_599579


namespace train_speed_correct_l599_599668

-- Definitions based on the conditions in a)
def train_length_meters : ℝ := 160
def time_seconds : ℝ := 4

-- Correct answer identified in b)
def expected_speed_kmh : ℝ := 144

-- Proof statement verifying that speed computed from the conditions equals the expected speed
theorem train_speed_correct :
  train_length_meters / 1000 / (time_seconds / 3600) = expected_speed_kmh :=
by
  sorry

end train_speed_correct_l599_599668


namespace first_two_cards_black_prob_l599_599323

noncomputable def probability_first_two_black : ℚ :=
  let total_cards := 52
  let black_cards := 26
  let first_draw_prob := black_cards / total_cards
  let second_draw_prob := (black_cards - 1) / (total_cards - 1)
  first_draw_prob * second_draw_prob

theorem first_two_cards_black_prob :
  probability_first_two_black = 25 / 102 :=
by
  sorry

end first_two_cards_black_prob_l599_599323


namespace smallest_product_form_of_98_l599_599624

-- Define a function to repeatedly multiply the digits of a number until a single digit is obtained
def reduce_to_single_digit (n : ℕ) : ℕ :=
  if n < 10 then n else
    let digits := List.ofDigits (n.digits 10) in
    reduce_to_single_digit (digits.foldr (*) 1)

-- State the problem as a theorem
theorem smallest_product_form_of_98 : reduce_to_single_digit 98 = 4 :=
by
  sorry

end smallest_product_form_of_98_l599_599624


namespace article_selling_price_l599_599315

theorem article_selling_price
  (cost_price gain : ℝ)
  (gain_percentage : ℝ)
  (h1 : gain = 75)
  (h2 : gain_percentage = 50)
  (h3 : gain = gain_percentage / 100 * cost_price) :
  let selling_price := cost_price + gain in
  selling_price = 225 :=
by {
  -- The proof goes here
  sorry
}

end article_selling_price_l599_599315


namespace range_of_a_l599_599770

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the actual function

axiom condition1 : ∀ x y : ℝ, f(x) + f(y) = f(x + y) + 3
axiom condition2 : f(3) = 6
axiom condition3 : ∀ x > 0, f(x) > 3

theorem range_of_a (a : ℝ) : f(2 * a + 1) < 5 ↔ a < 1 / 2 := 
by
  sorry

end range_of_a_l599_599770


namespace percentage_increase_l599_599666

theorem percentage_increase (initial final : ℝ)
  (h_initial: initial = 60) (h_final: final = 90) :
  (final - initial) / initial * 100 = 50 :=
by
  sorry

end percentage_increase_l599_599666


namespace combined_resistance_l599_599495

theorem combined_resistance (x y : ℝ) (r : ℝ) (hx : x = 4) (hy : y = 6) :
  (1 / r) = (1 / x) + (1 / y) → r = 12 / 5 :=
by
  sorry

end combined_resistance_l599_599495


namespace recurring_decimal_to_fraction_l599_599033

theorem recurring_decimal_to_fraction : 
  ∀ (x : ℝ), x = 0.7 * 10⁻¹ + (56 / 99) * (10⁻²) → x = 749 / 990 :=
by 
  intros x hx
  have h1 : x = 0.7 + 0.56 / 99 := sorry
  have h2 : 749 = 74.9 * 10 := sorry
  have h3 : 99 * 10 = 990 := sorry
  rw [←hx, h1, h2, h3]
  sorry

end recurring_decimal_to_fraction_l599_599033


namespace product_of_two_numbers_l599_599213

theorem product_of_two_numbers (a b : ℕ) (h_lcm : Nat.lcm a b = 560) (h_hcf : Nat.gcd a b = 75) :
  a * b = 42000 :=
by
  sorry

end product_of_two_numbers_l599_599213


namespace arithmetic_sequence_xy_sum_l599_599645

theorem arithmetic_sequence_xy_sum :
  ∃ x y : ℤ, 
  (∀ n m : ℕ, n < m → arithmetic_sequence (5 + 7 * n) ∧ arithmetic_sequence (5 + 7 * m)) →
  find_previous_terms 40 7 = (x, y) ∧ x + y = 26 + 33 := sorry

def arithmetic_sequence (n : ℤ) : Prop := ∃ k : ℕ, n = 5 + 7 * k

def find_previous_terms (n d : ℤ) : ℤ × ℤ :=
  let y := n - d
  let x := y - d
  (x, y)

end arithmetic_sequence_xy_sum_l599_599645


namespace integer_part_of_a_100_l599_599616

noncomputable def sequence (n : ℕ) : ℝ :=
  Nat.recOn n 1 (λ n a_n, a_n + 1 / a_n)

theorem integer_part_of_a_100 : ⌊sequence 100⌋ = 14 := by
  sorry

end integer_part_of_a_100_l599_599616


namespace sum_x_y_z_l599_599163

theorem sum_x_y_z :
  (∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ a^2 = 9/25 ∧ b^2 = (3 + real.sqrt 2) ^ 2 / 14 ∧ a < 0 ∧ b > 0 ∧ (a + b)^3 = x * real.sqrt y / z ∧ x + y + z = 695) :=
begin
  sorry
end

end sum_x_y_z_l599_599163


namespace sonja_peanut_butter_l599_599589

theorem sonja_peanut_butter (total_weight honey_weight oil_weight peanuts_weight : ℝ)
    (ratio_oil_peanuts : ℝ) (sweetness_level : ℝ) (h : sweetness_level = 4) :
    ratio_oil_peanuts = 2.5 / 11.5 → total_weight = 34 → honey_weight = 1.5 * sweetness_level →
    peanuts_weight = total_weight - honey_weight → 
    oil_weight = (2.5 / 11.5) * peanuts_weight →
    (oil_weight + honey_weight ≈ 12.09) :=
    by sorry

end sonja_peanut_butter_l599_599589


namespace probability_of_each_suit_in_five_draws_with_replacement_l599_599110

theorem probability_of_each_suit_in_five_draws_with_replacement :
  let deck_size := 52
  let num_cards := 5
  let num_suits := 4
  let prob_each_suit := 1/4
  let target_probability := 9/16
  prob_each_suit * (3/4) * (1/2) * (1/4) * 24 = target_probability :=
by sorry

end probability_of_each_suit_in_five_draws_with_replacement_l599_599110


namespace units_digit_of_even_factorial_sum_l599_599964

theorem units_digit_of_even_factorial_sum :
  (2! + 4! + 6! + 8! + 10!) % 10 = 6 :=
by
  sorry

end units_digit_of_even_factorial_sum_l599_599964


namespace relationship_among_a_b_c_l599_599414

noncomputable def a := Real.log 0.99 / Real.log 365
noncomputable def b := 1.01 ^ 365
noncomputable def c := 0.99 ^ 365

theorem relationship_among_a_b_c : a < c ∧ c < b := 
by
  -- skipped proof
  sorry

end relationship_among_a_b_c_l599_599414


namespace range_a_l599_599762

theorem range_a (a : ℝ) (x : ℝ) : 
    (∀ x, (x = 1 → x - a ≥ 1) ∧ (x = -1 → ¬(x - a ≥ 1))) ↔ (-2 < a ∧ a ≤ 0) :=
by
  sorry

end range_a_l599_599762


namespace not_p_and_p_l599_599904

theorem not_p_and_p (p : Prop) : ¬ (p ∧ ¬ p) :=
by 
  sorry

end not_p_and_p_l599_599904


namespace range_of_k_if_q_range_of_k_if_p_and_q_false_and_p_or_q_true_l599_599094

variable (k : ℝ)

def p : Prop := ∃ F M, (0 < k) ∧ (C : ℝ → ℝ) (y^2 = k*x) ∧ (M = (1, sqrt 2))
def q : Prop := (13 - k^2 > 2*k - 2) ∧ (2*k - 2 > 0) ∧ (E : ℝ → ℝ) (x^2 / (13 - k^2) + y^2 / (2*k - 2) = 1)

theorem range_of_k_if_q (hq : q k) : 1 < k ∧ k < 3 :=
by
  sorry

theorem range_of_k_if_p_and_q_false_and_p_or_q_true (hpq_false : ¬(p k ∧ q k)) (hpq_true : p k ∨ q k) :
  (0 < k ∧ k ≤ 1) ∨ (2 < k ∧ k < 3) :=
by
  sorry

end range_of_k_if_q_range_of_k_if_p_and_q_false_and_p_or_q_true_l599_599094


namespace number_of_beakers_calculation_l599_599627

-- Conditions
def solution_per_test_tube : ℕ := 7
def number_of_test_tubes : ℕ := 6
def solution_per_beaker : ℕ := 14

-- Total amount of solution
def total_solution : ℕ := solution_per_test_tube * number_of_test_tubes

-- Number of beakers is the fraction of total solution and solution per beaker
def number_of_beakers : ℕ := total_solution / solution_per_beaker

-- Statement of the problem
theorem number_of_beakers_calculation : number_of_beakers = 3 :=
by 
  -- Proof goes here
  sorry

end number_of_beakers_calculation_l599_599627


namespace no_positive_integer_n_l599_599047

theorem no_positive_integer_n :
  ∀ n : ℕ, (n > 0 → ∑ i in finset.range(n + 1), nat.floor (real.log2 (i + 1)) ≠ 2006) :=
by 
  intros n hn
  sorry

end no_positive_integer_n_l599_599047


namespace intersection_of_A_and_B_l599_599124

def setA : Set ℝ := {x : ℝ | |x| > 1}
def setB : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

theorem intersection_of_A_and_B : setA ∩ setB = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_of_A_and_B_l599_599124


namespace smallest_m_plus_n_l599_599605

theorem smallest_m_plus_n (m n : ℕ) (h1 : m > 1) (h2 : ∃ a b : ℝ, a = arcsin (log m (2 * n * b)) ∧ -1 ≤ log m (2 * n * b) ∧ log m (2 * n * b) ≤ 1 ∧ (b - a = 1 / 2027)) 
: m + n = 13855 :=
sorry

end smallest_m_plus_n_l599_599605


namespace profit_after_discount_l599_599294

noncomputable def purchase_price : ℝ := 100
noncomputable def increase_rate : ℝ := 0.25
noncomputable def discount_rate : ℝ := 0.10

theorem profit_after_discount :
  let selling_price := purchase_price * (1 + increase_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let profit := discounted_price - purchase_price
  profit = 12.5 :=
by
  sorry 

end profit_after_discount_l599_599294


namespace solutions_of_equation_l599_599940

noncomputable def check_solution (x : ℝ) : Prop :=
  x ^ log x ^ 2 = x ^ 5 / 1000

theorem solutions_of_equation : check_solution 10 ∧ check_solution 31.62 := 
by 
  sorry

end solutions_of_equation_l599_599940


namespace find_n_l599_599072

-- Define conditions
def condition1 (x : ℝ) : Prop :=
  log10 (sin x) + log10 (cos x) = -2

def condition2 (x n : ℝ) : Prop :=
  log10 (sin x + cos x) = (1/2) * (log10 n - 2) + 0.3

-- Define the main theorem to prove n = 26 under the given conditions
theorem find_n (x : ℝ) (n : ℝ) (h1 : condition1 x) (h2 : condition2 x n) : n = 26 :=
  sorry -- Proof goes here


end find_n_l599_599072


namespace cot_arctan_combo_l599_599758

theorem cot_arctan_combo :
  Real.cot (Real.arccot 5 - Real.arccot 9 + Real.arccot 14 + Real.arccot 23) = 7339 / 2041 := 
by 
  sorry

end cot_arctan_combo_l599_599758


namespace sum_of_common_divisors_l599_599392

theorem sum_of_common_divisors (a b : ℕ) (ha : a = 75) (hb : b = 45) :
  ∑ d in (finset.filter (λ x, x ∣ b) (finset.filter (λ x, x ∣ a) (finset.range (a + 1)))), d = 24 :=
by 
  sorry

end sum_of_common_divisors_l599_599392


namespace complement_of_A_l599_599462

open Set

-- Define the universal set U
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := { x | abs (x - 1) > 1 }

-- Define the problem statement
theorem complement_of_A :
  ∀ x : ℝ, x ∈ compl A ↔ x ∈ Icc 0 2 :=
by
  intro x
  rw [mem_compl_iff, mem_Icc]
  sorry

end complement_of_A_l599_599462


namespace sum_of_interior_edges_l599_599692

theorem sum_of_interior_edges (width : ℝ) (area_of_frame : ℝ) (outer_edge : ℝ) : 
  width = 2 → 
  area_of_frame = 32 → 
  outer_edge = 8 → 
  let y := (area_of_frame - 16) / 4 in
  let interior_length1 := outer_edge - 2 * width in
  let interior_length2 := y - 2 * width in
  2 * interior_length1 + 2 * interior_length2 = 8 :=
by 
  intros h1 h2 h3
  let y := 4
  let interior_length1 := 4
  let interior_length2 := 0
  trivialqedsorry 

end sum_of_interior_edges_l599_599692


namespace max_value_of_k_l599_599096

-- Define the necessary variables and conditions
variables {x y k : ℝ}
hypotheses (hx : 0 < x) (hy : 0 < y) (hk : 0 < k) 
  (h1 : 3 = k^2 * ((x^2) / (y^2) + (y^2) / (x^2)) + k * (x / y + y / x))

-- Define the theorem and its statement
theorem max_value_of_k : k ≤ (Real.sqrt 7 - 1) / 2 := sorry

end max_value_of_k_l599_599096


namespace rectangle_triangulation_count_l599_599888

theorem rectangle_triangulation_count (n : ℕ) (hn : 1 ≤ n) : 
  ∃ k : ℕ, k = 2 * n ∧ nat.choose (2 * n) n = k :=
by
  sorry

end rectangle_triangulation_count_l599_599888


namespace grape_juice_amount_l599_599304

theorem grape_juice_amount (total_juice : ℝ)
  (orange_juice_percent : ℝ) (watermelon_juice_percent : ℝ)
  (orange_juice_amount : ℝ) (watermelon_juice_amount : ℝ)
  (grape_juice_amount : ℝ) :
  orange_juice_percent = 0.25 →
  watermelon_juice_percent = 0.40 →
  total_juice = 200 →
  orange_juice_amount = total_juice * orange_juice_percent →
  watermelon_juice_amount = total_juice * watermelon_juice_percent →
  grape_juice_amount = total_juice - orange_juice_amount - watermelon_juice_amount →
  grape_juice_amount = 70 :=
by
  sorry

end grape_juice_amount_l599_599304


namespace max_value_quadratic_expression_l599_599960

theorem max_value_quadratic_expression : ∃ x : ℝ, -3 * x^2 + 18 * x - 5 ≤ 22 ∧ ∀ y : ℝ, -3 * y^2 + 18 * y - 5 ≤ 22 := 
by 
  sorry

end max_value_quadratic_expression_l599_599960


namespace relay_team_member_distance_l599_599249

theorem relay_team_member_distance (n_people : ℕ) (total_distance : ℕ)
  (h1 : n_people = 5) (h2 : total_distance = 150) : total_distance / n_people = 30 :=
by 
  sorry

end relay_team_member_distance_l599_599249


namespace absolute_difference_segments_l599_599953

-- Define the conditions and the given constants.

variables (A B C D : Point)
variables (r₁ r₂ a : ℝ)
variables (O₁ O₂ : Point) (h₁ : Circle O₁ r₁) (h₂ : Circle O₂ r₂)

-- Define the intersecting circles
def intersect (A B : Point) (h₁ h₂ : Circle) : Prop :=
  A ∈ h₁ ∧ A ∈ h₂ ∧ B ∈ h₁ ∧ B ∈ h₂

-- Define AB and diameters AC and AD
def diameters_through_A (A C D : Point) (h₁ h₂ : Circle) : Prop :=
  diameter h₁ A C ∧ diameter h₂ A D

-- Define the distance between centers O₁O₂ as 'a'
def distance_centers (O₁ O₂ : Point) (a : ℝ) : Prop :=
  dist O₁ O₂ = a

-- The Lean statement for the problem
theorem absolute_difference_segments
  (h1 : intersect A B h₁ h₂)
  (h2 : diameters_through_A A C D h₁ h₂)
  (h3 : distance_centers O₁ O₂ a)
  (same_side_centers : ∃ (C₁ C₂ : Point), sameSideLine C₁ C₂ (line A B) ∧ dist C₁ A = r₁ ∧ dist C₂ A = r₂)
  : abs (dist B C - dist B D) = 2 * a := 
sorry

end absolute_difference_segments_l599_599953


namespace triangle_shrink_A_transform_l599_599501

-- Define the initial conditions of the problem
def point (α : Type*) := prod α α
def A : point ℝ := (-4, 2)
def origin : point ℝ := (0, 0)
def similarity_ratio : ℝ := 1 / 2

-- Define the similarity transformation function
def similarity_transform (p : point ℝ) (ratio : ℝ) : point ℝ :=
  (p.1 * ratio, p.2 * ratio)

-- Define the expected transformed points
def A' : point ℝ := similarity_transform A similarity_ratio
def A'_sym : point ℝ := (A'.1 * -1, A'.2 * -1)

-- Statement of the problem's proof
theorem triangle_shrink_A_transform :
  A' = (-2, 1) ∨ A_symmetric = (2, -1) :=
sorry

end triangle_shrink_A_transform_l599_599501


namespace focus_of_parabola_length_of_AB_l599_599455

-- Definition for the parabola and line conditions
def parabola (x y : ℝ) : Prop := x^2 = 4 * y
def line (x y : ℝ) : Prop := y = x + 1

-- Statement to prove the focus of the parabola
theorem focus_of_parabola :
  ∃ p : ℝ × ℝ, parabola_focus p :=
sorry

-- Statement to prove the length of AB
theorem length_of_AB (A B : ℝ × ℝ) :
  (parabola A.1 A.2 ∧ line A.1 A.2) →
  (parabola B.1 B.2 ∧ line B.1 B.2) →
  |A - B| = 8 :=
sorry

end focus_of_parabola_length_of_AB_l599_599455


namespace set_intersection_complement_l599_599566

open Set

theorem set_intersection_complement:
  let U := {1, 2, 3, 4, 5}
  let M := {1, 4}
  let N := {1, 3, 5}
  N ∩ (U \ M) = {3, 5} :=
by
  sorry

end set_intersection_complement_l599_599566


namespace Sn_formula_l599_599571

open Nat

noncomputable def Sn (n : ℕ) : ℕ :=
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) / 5

theorem Sn_formula (n : ℕ) : 
  (∑ k in range n, k * (k + 1) * (k + 2) * (k + 3)) = Sn n :=
by
  sorry

end Sn_formula_l599_599571


namespace square_properties_l599_599036

theorem square_properties (perimeter : ℝ) (h1 : perimeter = 40) :
  ∃ (side length area diagonal : ℝ), side = 10 ∧ length = 10 ∧ area = 100 ∧ diagonal = 10 * Real.sqrt 2 :=
by
  sorry

end square_properties_l599_599036


namespace height_of_prism_l599_599702

-- Definitions based on conditions
def Volume : ℝ := 120
def edge1 : ℝ := 3
def edge2 : ℝ := 4
def BaseArea : ℝ := edge1 * edge2

-- Define the problem statement
theorem height_of_prism (h : ℝ) : (BaseArea * h / 2 = Volume) → (h = 20) :=
by
  intro h_value
  have Volume_equiv : h = 2 * Volume / BaseArea := sorry
  sorry

end height_of_prism_l599_599702


namespace num_div_divided_by_10_l599_599647

-- Given condition: the number divided by 10 equals 12
def number_divided_by_10_gives_12 (x : ℝ) : Prop :=
  x / 10 = 12

-- Lean statement for the mathematical problem
theorem num_div_divided_by_10 (x : ℝ) (h : number_divided_by_10_gives_12 x) : x = 120 :=
by
  sorry

end num_div_divided_by_10_l599_599647


namespace find_two_digit_number_l599_599975

theorem find_two_digit_number :
  ∃ (XY : ℕ), -- there exists a natural number XY
    -- Condition 1: XY satisfies that its ones place when confused with 6 but is actually 9
    ∃ (X Y : ℕ),
      (XY = 10 * X + Y) ∧ (Y = 9) ∧
      -- Condition 2: A three-digit number when switched in hundreds and ones place is 253
      ∃ (three_digit : ℕ),
        (three_digit = 352) ∧
        (switch_digits three_digit = 253) ∧
        -- Condition 3: The sum is 299 
        (XY + 253 = 299) ∧
        -- Conclusion: XY is the two-digit number we need to find
        (XY = 49) :=
begin
  -- Definitions needed in Lean
  def switch_digits (n : ℕ) : ℕ := 
    let hundreds := n / 100,
        tens := (n / 10) % 10,
        ones := n % 10 in
    ones * 100 + tens * 10 + hundreds,
    
  -- Proof is skipped
  sorry
end

end find_two_digit_number_l599_599975


namespace total_charge_for_5_hours_l599_599673

def charge_for_first_hour := 104
def charge_for_additional_hour := 74

theorem total_charge_for_5_hours
  (charge_for_first_hour = charge_for_additional_hour + 30)
  (3 * charge_for_additional_hour + 30 = 252) :
  (charge_for_first_hour + 4 * charge_for_additional_hour = 400) :=
sorry

end total_charge_for_5_hours_l599_599673


namespace no_negative_exponents_l599_599097

noncomputable def even (n : ℤ) := ∃ k : ℤ, n = 2 * k

theorem no_negative_exponents (a b c d : ℤ) 
  (ha : even a) (hb : even b) (hc : even c) (hd : even d)
  (h : 5^a + 5^b = 3^c + 3^d) : 
  ¬ (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) :=
sorry

end no_negative_exponents_l599_599097


namespace estimate_total_trout_l599_599309

-- Define the conditions
def n_tagged_april1 : ℕ := 80
def n_sample_aug1 : ℕ := 100
def n_tagged_aug1 : ℕ := 4
def p_left : ℚ := 0.30
def p_new : ℚ := 0.50

-- Define the total number of trout present in the river on April 1
def n_present_aug1 := (1 - p_new) * n_sample_aug1
def proportion_equiv (_ : ℕ) (_ : ℕ) (_ : ℕ) (_ : ℕ) (_ : ℚ) (_ : ℚ) : Prop :=
  n_tagged_aug1 / n_present_aug1 = n_tagged_april1 / n_total_april1

-- Formulate the theorem to prove
theorem estimate_total_trout (n_total_april1 : ℕ) :
  proportion_equiv n_tagged_april1 n_sample_aug1 n_tagged_aug1 n_present_aug1 p_left p_new →
  n_total_april1 = 1000 := by
  sorry

end estimate_total_trout_l599_599309


namespace chef_total_potatoes_l599_599297

theorem chef_total_potatoes (cooked : ℕ) (time_per_potato : ℕ) (time_remaining : ℕ)
  (h_cooked : cooked = 6)
  (h_time_per_potato : time_per_potato = 6)
  (h_time_remaining : time_remaining = 36) :
  cooked + (time_remaining / time_per_potato) = 12 :=
by
  rw [h_cooked, h_time_per_potato, h_time_remaining]
  exact rfl

end chef_total_potatoes_l599_599297


namespace sum_of_reciprocals_of_roots_l599_599396

theorem sum_of_reciprocals_of_roots : 
  ∀ {r1 r2 : ℝ}, (r1 + r2 = 14) → (r1 * r2 = 6) → (1 / r1 + 1 / r2 = 7 / 3) :=
by
  intros r1 r2 h_sum h_product
  sorry

end sum_of_reciprocals_of_roots_l599_599396


namespace egorov_theorem_l599_599575

open ProbabilityTheory

theorem egorov_theorem {Ω : Type*} {ℱ : MeasurableSpace Ω} (μ : MeasureTheory.Measure Ω) 
  {ξ : Ω → ℝ} {ξₙ : ℕ → (Ω → ℝ)} 
  (h_ξₙ_to_ξ_a.s : ∀ᵐ ω ∂μ, Filter.Tendsto (λ n, ξₙ n ω) Filter.atTop (𝓝 (ξ ω))) :
  ∀ ε > 0, ∃ A ∈ ℱ, μ Aᶜ ≤ ε ∧ ∀ ε' > 0, ∃ N : ℕ, ∀ n ≥ N, ∀ ω ∈ A, abs (ξₙ n ω - ξ ω) ≤ ε' :=
begin
  sorry
end

end egorov_theorem_l599_599575


namespace new_volume_correct_l599_599319

-- Define the conditions
def original_volume : ℝ := 60
def length_factor : ℝ := 3
def width_factor : ℝ := 2
def height_factor : ℝ := 1.20

-- Define the new volume as a result of the above factors
def new_volume : ℝ := original_volume * length_factor * width_factor * height_factor

-- Proof statement for the new volume being 432 cubic feet
theorem new_volume_correct : new_volume = 432 :=
by 
    -- Directly state the desired equality
    sorry

end new_volume_correct_l599_599319


namespace sufficient_condition_l599_599985

theorem sufficient_condition (a b : ℝ) : ab ≠ 0 → a ≠ 0 :=
sorry

end sufficient_condition_l599_599985


namespace constant_term_binomial_expansion_l599_599599

theorem constant_term_binomial_expansion :
  let T := ( √x - (2 / √x) )^6 in
  -- The general term of the expansion
  ∃ r : ℕ, 3 - r = 0 ∧ (-2)^r * @binomial ℕ _ 6 r * x^(3-r) = -160 := 
by
  sorry

end constant_term_binomial_expansion_l599_599599


namespace max_value_l599_599961

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  -3 * x^2 + 18 * x - 5

theorem max_value : ∃ x : ℝ, quadratic_function x = 22 :=
sorry

end max_value_l599_599961


namespace students_left_is_31_l599_599899

-- Define the conditions based on the problem statement
def total_students : ℕ := 124
def checked_out_early : ℕ := 93

-- Define the theorem that states the problem we want to prove
theorem students_left_is_31 :
  total_students - checked_out_early = 31 :=
by
  -- Proof would go here
  sorry

end students_left_is_31_l599_599899


namespace probability_at_least_2_more_heads_than_tails_l599_599268

theorem probability_at_least_2_more_heads_than_tails : 
  let total_outcomes := 2^12,
      favorable_outcomes := ∑ k in finset.range 7 13, nat.choose 12 k 
  in favorable_outcomes = 1586 ∧ 
      (favorable_outcomes / total_outcomes : ℚ) = (793 / 2048 : ℚ) :=
sorry

end probability_at_least_2_more_heads_than_tails_l599_599268


namespace total_time_research_expeditions_l599_599148

theorem total_time_research_expeditions :
  let first_artifact_research := 6 -- months
  let first_artifact_expedition := 24 -- months
  
  let second_artifact_research := 3 * first_artifact_research
  let second_artifact_expedition := 2 * first_artifact_expedition
  
  let third_artifact_research := second_artifact_research / 2
  let third_artifact_expedition := second_artifact_expedition
  
  let fourth_artifact_research := first_artifact_research
  let fourth_artifact_expedition := third_artifact_expedition * 1.5
  
  let fifth_artifact_research := fourth_artifact_research * 1.25
  let fifth_artifact_expedition := 1.5 * (first_artifact_research + first_artifact_expedition)
  
  let sixth_artifact_research := first_artifact_research + third_artifact_research
  let sixth_artifact_expedition := second_artifact_expedition + fourth_artifact_expedition
  
  let previous_total_time := 
    first_artifact_research + first_artifact_expedition +
    second_artifact_research + second_artifact_expedition +
    third_artifact_research + third_artifact_expedition +
    fourth_artifact_research + fourth_artifact_expedition +
    fifth_artifact_research + fifth_artifact_expedition +
    sixth_artifact_research + sixth_artifact_expedition
  
  let seventh_artifact_research := second_artifact_research + fourth_artifact_research
  let seventh_artifact_expedition := previous_total_time / 2
  
  let total_time_months := 
    first_artifact_research + first_artifact_expedition +
    second_artifact_research + second_artifact_expedition +
    third_artifact_research + third_artifact_expedition +
    fourth_artifact_research + fourth_artifact_expedition +
    fifth_artifact_research + fifth_artifact_expedition +
    sixth_artifact_research + sixth_artifact_expedition +
    seventh_artifact_research + seventh_artifact_expedition
  
  let total_time_years := total_time_months / 12

  total_time_years ≈ 54.31 :=
by sorry

end total_time_research_expeditions_l599_599148


namespace Lisa_initial_pencils_l599_599821

-- Variables
variable (G_L_initial : ℕ) (L_L_initial : ℕ) (G_L_total : ℕ)

-- Conditions
def G_L_initial_def := G_L_initial = 2
def G_L_total_def := G_L_total = 101
def Lisa_gives_pencils : Prop := G_L_total = G_L_initial + L_L_initial

-- Proof statement
theorem Lisa_initial_pencils (G_L_initial : ℕ) (G_L_total : ℕ)
  (h1 : G_L_initial = 2) (h2 : G_L_total = 101) (h3 : G_L_total = G_L_initial + L_L_initial) :
  L_L_initial = 99 := 
by 
  sorry

end Lisa_initial_pencils_l599_599821


namespace arithmetic_sequence_l599_599876

noncomputable def mean (a : ℕ → ℝ) (n : ℕ) := (finset.range n).sum (λ k, a (k + 1)) / n

theorem arithmetic_sequence (a : ℕ → ℝ) (c : ℝ) :
  (∀ i j k : ℕ, i ≠ j → j ≠ k → k ≠ i → (i - j) * mean a k + (j - k) * mean a i + (k - i) * mean a j = c) →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a 1 + n * d :=
begin
  sorry
end

end arithmetic_sequence_l599_599876


namespace sales_function_relation_max_profit_l599_599998

-- Define the sales volume function
def sales_volume_function (x : ℝ) : ℝ := -10 * x + 500

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - 20) * sales_volume_function x

-- Prove the functional relationship between sales volume y and selling price x
theorem sales_function_relation (x : ℝ) : sales_volume_function x = -10 * x + 500 :=
by
  sorry

-- Prove the price that maximizes monthly profit and the maximum profit
theorem max_profit : ∃ x : ℝ, 20 ≤ x ∧ x ≤ 32 ∧ profit_function x = 2160 :=
by
  use 32
  split
  norm_num
  split
  norm_num
  sorry

end sales_function_relation_max_profit_l599_599998
