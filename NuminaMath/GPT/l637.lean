import Mathlib
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.Field
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Catalan
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Fib
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Conditional
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Cyclic
import Mathlib.Geometry.Euclidean.Incircle
import Mathlib.LinearAlgebra.Matrix.Vector
import Mathlib.NumberTheory.Mod
import Mathlib.NumberTheory.ModularArithmetic
import Mathlib.NumberTheory.Prime
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

namespace geometric_sequence_tenth_term_l637_637405

theorem geometric_sequence_tenth_term (a r : ℝ) (h₁ : a = 4) (h₂ : a * r = -2) : 
  a * r^9 = -1 / 128 :=
by
  rw [h₁, h₂]
  sorry

end geometric_sequence_tenth_term_l637_637405


namespace num_of_positive_divisors_of_30_pow_4_l637_637806

def num_divisors (n : ℕ) : ℕ :=
  ∏ p in n.factors.to_finset, ( n.factor_count p + 1)

theorem num_of_positive_divisors_of_30_pow_4 :
  num_divisors (30 ^ 4) = 64 :=
sorry

end num_of_positive_divisors_of_30_pow_4_l637_637806


namespace julia_money_l637_637620

noncomputable def cost_of_4_cds (cost_per_cd : ℕ) : ℕ :=
  4 * cost_per_cd

theorem julia_money : 
  let rock_cost := 5
  let pop_cost := 10
  let dance_cost := 3
  let country_cost := 7
  let shortfall := 25
  let total_cost := cost_of_4_cds rock_cost + cost_of_4_cds pop_cost + cost_of_4_cds dance_cost + cost_of_4_cds country_cost
  in total_cost - shortfall = 75 :=
by
  sorry

end julia_money_l637_637620


namespace length_of_AX_l637_637368

-- Define the given lengths
def BC : ℝ := 50
def AC : ℝ := 40
def BX : ℝ := 35

-- Assuming CX bisects angle ACB
def angle_bisector (AC AX BC BX : ℝ) : Prop := AC / AX = BC / BX

-- Statement to prove using Lean
theorem length_of_AX (CX_bisects_∠ACB : angle_bisector AC AX BC BX) : AX = 28 :=
by
  -- The proof is not required, so we skip it by adding sorry
  sorry

end length_of_AX_l637_637368


namespace trains_clear_each_other_in_11_seconds_l637_637293

-- Define the lengths of the trains
def length_train1 := 100  -- in meters
def length_train2 := 120  -- in meters

-- Define the speeds of the trains (in km/h), converted to m/s
def speed_train1 := 42 * 1000 / 3600  -- 42 km/h to m/s
def speed_train2 := 30 * 1000 / 3600  -- 30 km/h to m/s

-- Calculate the total distance to be covered
def total_distance := length_train1 + length_train2  -- in meters

-- Calculate the relative speed when they are moving towards each other
def relative_speed := speed_train1 + speed_train2  -- in m/s

-- Calculate the time required for the trains to be clear of each other (in seconds)
noncomputable def clear_time := total_distance / relative_speed

-- Theorem stating the above
theorem trains_clear_each_other_in_11_seconds :
  clear_time = 11 :=
by
  -- Proof would go here
  sorry

end trains_clear_each_other_in_11_seconds_l637_637293


namespace min_value_inequality_l637_637546

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 3) :
  (1 / x) + (4 / y) + (9 / z) ≥ 12 := 
sorry

end min_value_inequality_l637_637546


namespace range_of_a_l637_637797

noncomputable def problem : set ℝ := 
  {a : ℝ | ∃ x : ℝ, x - a < 0 ∧ ax < 1 } -- Define the system of inequalities

theorem range_of_a : 
  {a : ℝ | ∃ x : ℝ, x - a < 0 ∧ ax < 1} = set.Ici (-1) :=
by
  unfold problem
  sorry

end range_of_a_l637_637797


namespace unique_solution_count_l637_637022

theorem unique_solution_count : 
  ∃! (x : ℝ), (2^(4*x + 2)) * (4^(2*x + 1)) = 8^(3*x + 4) :=
by
  -- Use sorry to skip the proof
  sorry

end unique_solution_count_l637_637022


namespace max_books_borrowed_l637_637310

theorem max_books_borrowed
  (total_students : ℕ)
  (zero_books_students : ℕ)
  (one_book_students : ℕ)
  (two_books_students : ℕ)
  (remaining_students_borrowing_at_least_three : ℕ)
  (average_books_per_student : ℝ) :
  total_students = 20 →
  zero_books_students = 3 →
  one_book_students = 9 →
  two_books_students = 4 →
  remaining_students_borrowing_at_least_three = total_students - (zero_books_students + one_book_students + two_books_students) →
  average_books_per_student = 2 →
  ∃ max_books : ℕ, max_books = 11 :=
begin
  sorry
end

end max_books_borrowed_l637_637310


namespace cases_in_1975_l637_637833

-- Define the initial condition in 1950
def cases_1950 : ℕ := 600000

-- Define the final condition in 2000
def cases_2000 : ℕ := 300

-- Define the number of cases in 1975
def cases_1975 : ℕ := 300150

-- Prove that under the linear decrease assumption, the number of cases in 1975 is 300150
theorem cases_in_1975 :
  ∀ (n : ℕ), n = 600000 → n - (25 * ((600000 - 300) / 50)) = 300150 :=
by
  intros n h_n
  rw h_n
  -- You would proceed with the steps to simplify and prove the result, placing the solution here as a proof.
  sorry

end cases_in_1975_l637_637833


namespace range_of_x_l637_637118

-- Defining the vectors as given in the conditions
def a (x : ℝ) : ℝ × ℝ := (x, 3)
def b : ℝ × ℝ := (2, -1)

-- Defining the condition that the angle is obtuse
def is_obtuse (x : ℝ) : Prop := 
  let dot_product := (a x).1 * b.1 + (a x).2 * b.2
  dot_product < 0

-- Defining the condition that vectors are not in opposite directions
def not_opposite_directions (x : ℝ) : Prop := x ≠ -6

-- Proving the required range of x
theorem range_of_x (x : ℝ) :
  is_obtuse x → not_opposite_directions x → x < 3 / 2 :=
sorry

end range_of_x_l637_637118


namespace eva_is_speed_skater_l637_637291

noncomputable def sport_at_table : Prop :=
  ∃ (Andrea Eva Filip Ben : Prop) 
    (isSkier isSpeedSkater isHockeyPlayer isSnowboarder isWoman : Prop),

  -- Define who sits at each position:
  let skier = λ x, x = Andrea ∧ x = isSkier,
      speedSkater = λ x, x = Eva ∧ x = isSpeedSkater,
      hockeyPlayer = λ x, x = Filip ∧ x = isHockeyPlayer,
      snowboarder = λ x, x = Ben ∧ x = isSnowboarder 
  in

  -- Conditions:
  (skier Andrea ∧ skier = skier) ∧
  (speedSkater Ben ∧ speedSkater = speedSkater) ∧
  (Eva ∧ Filip = Eva.next ∧ Filip.next = Eva) ∧
  (skier Andrea.next = Andrea) ∧
  
  -- The Assertion we need to Prove
  (Eva = speedSkater)

theorem eva_is_speed_skater : sport_at_table := 
  sorry

end eva_is_speed_skater_l637_637291


namespace alpha_add_beta_eq_pi_div_two_l637_637078

open Real

theorem alpha_add_beta_eq_pi_div_two (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : (sin α) ^ 4 / (cos β) ^ 2 + (cos α) ^ 4 / (sin β) ^ 2 = 1) :
  α + β = π / 2 :=
sorry

end alpha_add_beta_eq_pi_div_two_l637_637078


namespace not_prime_expression_l637_637317
-- Import the necessary mathematical library

-- Define the condition and the statement to prove
theorem not_prime_expression (n : ℕ) (h : n > 2) : 
  ¬ prime (n^(n^n) - 6 * n^n + 5) :=
by
  sorry

end not_prime_expression_l637_637317


namespace find_n_l637_637591

noncomputable def p_n (n : ℕ) : ℤ :=
  Nat.gcd (Nat.sum (List.range (n + 1))) (Nat.prod (List.range (n + 1)))

theorem find_n (n : ℕ) : n = 2 ∨ n = 7 ∨ n = 22 ↔ 3 ∣ p_n n :=
begin
  sorry
end

end find_n_l637_637591


namespace value_of_2022_plus_a_minus_b_l637_637439

theorem value_of_2022_plus_a_minus_b (x a b : ℚ) (h_distinct : x ≠ a ∧ x ≠ b ∧ a ≠ b) 
  (h_gt : a > b) (h_min : ∀ y : ℚ, |y - a| + |y - b| ≥ 2 ∧ |x - a| + |x - b| = 2) :
  2022 + a - b = 2024 := 
by 
  sorry

end value_of_2022_plus_a_minus_b_l637_637439


namespace tuesday_pairs_of_boots_l637_637583

theorem tuesday_pairs_of_boots (S B : ℝ) (x : ℤ) 
  (h1 : 22 * S + 16 * B = 460)
  (h2 : 8 * S + x * B = 560)
  (h3 : B = S + 15) : 
  x = 24 :=
sorry

end tuesday_pairs_of_boots_l637_637583


namespace vector_b_and_k_l637_637787

noncomputable def vector_b (b : ℝ × ℝ × ℝ) (a : ℝ × ℝ × ℝ) := 
  b = (2 * a.1, 2 * a.2, 2 * a.3)

noncomputable def perpendicular (u v : ℝ × ℝ × ℝ) :=
  (u.1 * v.1 + u.2 * v.2 + u.3 * v.3) = 0

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) :=
  (u.1 * v.1 + u.2 * v.2 + u.3 * v.3)

theorem vector_b_and_k (a b : ℝ × ℝ × ℝ) (k : ℝ) 
  (h_collinear : ∃ λ : ℝ, b = (λ * a.1, λ * a.2, λ * a.3))
  (h_dot : dot_product a b = 18)
  (h_perpendicular : perpendicular (k * a.1 + b.1, k * a.2 + b.2, k * a.3 + b.3)
                                (k * a.1 - b.1, k * a.2 - b.2, k * a.3 - b.3)) :
  b = (4, -2, 4) ∧ (k = 2 ∨ k = -2) :=
sorry

end vector_b_and_k_l637_637787


namespace minimum_benches_for_equal_occupancy_l637_637921

theorem minimum_benches_for_equal_occupancy (M : ℕ) :
  (∃ x y, x = y ∧ 8 * M = x ∧ 12 * M = y) ↔ M = 3 := by
  sorry

end minimum_benches_for_equal_occupancy_l637_637921


namespace line_condition_l637_637085

/-- Given a line l1 passing through points A(-2, m) and B(m, 4),
    a line l2 given by the equation 2x + y - 1 = 0,
    and a line l3 given by the equation x + ny + 1 = 0,
    if l1 is parallel to l2 and l2 is perpendicular to l3,
    then the value of m + n is -10. -/
theorem line_condition (m n : ℝ) (h1 : (4 - m) / (m + 2) = -2)
  (h2 : (2 * -1) * (-1 / n) = -1) : m + n = -10 := 
sorry

end line_condition_l637_637085


namespace trip_time_difference_l637_637689

theorem trip_time_difference (speed distance1 distance2 : ℕ) (h1 : speed > 0) (h2 : distance2 > distance1) 
  (h3 : speed = 60) (h4 : distance1 = 540) (h5 : distance2 = 570) : 
  (distance2 - distance1) / speed * 60 = 30 := 
by
  sorry

end trip_time_difference_l637_637689


namespace fewest_restricted_days_l637_637150

noncomputable def january_days : ℕ := 31

def weekdays := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def restriction_schedule (day: String) : List ℕ :=
  if day = "Monday" then [1, 6, 5, 0]
  else if day = "Tuesday" then [1, 6, 2, 7]
  else if day = "Wednesday" then [2, 7, 3, 8]
  else if day = "Thursday" then [3, 8, 4, 9]
  else if day = "Friday" then [4, 9, 5, 0]
  else []

def get_day_of_week (day: ℕ) : String :=
  weekdays.get! ((day - 1) % 7)

def restricted_days (tail_numbers : List ℕ) : ℕ :=
  (List.range january_days).count (λ d, ∃ t ∈ tail_numbers, t ∈ restriction_schedule (get_day_of_week (d + 1)))

theorem fewest_restricted_days : restricted_days [1, 6] = 8 →
                                (restricted_days [2, 7] > 8) →
                                (restricted_days [3, 8] > 8) →
                                (restricted_days [4, 9] > 8) →
                                (restricted_days [5, 0] > 8) →
                                [1, 6] = [1, 6] :=
sorry

end fewest_restricted_days_l637_637150


namespace point_on_y_axis_coordinates_l637_637223

theorem point_on_y_axis_coordinates (m : ℤ) (P : ℤ × ℤ) (hP : P = (m - 1, m + 3)) (hY : P.1 = 0) : P = (0, 4) :=
sorry

end point_on_y_axis_coordinates_l637_637223


namespace gabriel_can_reach_199_with_19_sevens_l637_637056

/-
Problem: Gabriel Giraffe can number the pages of his storybook from 101 to a maximum of 199 using his limited supply of nineteen '7's.
-/
theorem gabriel_can_reach_199_with_19_sevens : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 8, 9},
      sevens := 19,
      startPage := 101
  in ∃ maxPage : ℕ, maxPage = 199 ∧
           (∀ n, startPage ≤ n ∧ n ≤ maxPage → 
                     ∀ k, (k ∈ digits ∨ k = 7) ∧ (k = 7 → (count k n ≤ sevens))) := 
begin
  sorry
end

end gabriel_can_reach_199_with_19_sevens_l637_637056


namespace merchants_tea_cups_l637_637958

theorem merchants_tea_cups (a b c : ℕ) 
  (h1 : a + b = 11)
  (h2 : b + c = 15)
  (h3 : a + c = 14) : 
  a + b + c = 20 :=
by
  sorry

end merchants_tea_cups_l637_637958


namespace hari_well_digging_days_l637_637647

theorem hari_well_digging_days :
  let jake_days := 16
  let paul_days := 24
  let together_days := 8
  let jake_rate := 1 / jake_days
  let paul_rate := 1 / paul_days
  let together_rate := 1 / together_days
  ∃ h, (1 / h) = together_rate - (jake_rate + paul_rate) :=
begin
  use 48,
  sorry
end

end hari_well_digging_days_l637_637647


namespace max_value_sin_cos_l637_637104

theorem max_value_sin_cos :
  ∀ x : ℝ, (sin (x / 2) + sqrt 3 * cos (x / 2)) ≤ 2 :=
by
  sorry

end max_value_sin_cos_l637_637104


namespace min_selected_numbers_l637_637579

def selected_numbers : Set ℕ := {1, 2, 5, 8, 9, 10}

theorem min_selected_numbers (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 20) :
  ∃ (S : Set ℕ), S ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧ 
  (∀ m ∈ S, m ≤ 20) ∧ 
  (∀ m, (m = n ∨ ∃ x y ∈ S, m = x + y) → m ∈ S) ∧
  (∀ T, T ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 
    (∀ m, (m = n ∨ ∃ x y ∈ T, m = x + y) → m ∈ T) → 
    ∃ (x : ℕ), size T ≥ 6) :=
sorry

end min_selected_numbers_l637_637579


namespace good_oranges_per_month_l637_637577

/-- Salaria has 50% of tree A and 50% of tree B, totaling to 10 trees.
    Tree A gives 10 oranges a month and 60% are good.
    Tree B gives 15 oranges a month and 1/3 are good.
    Prove that the total number of good oranges Salaria gets per month is 55. -/
theorem good_oranges_per_month 
  (total_trees : ℕ) 
  (percent_tree_A : ℝ) 
  (percent_tree_B : ℝ) 
  (oranges_tree_A : ℕ)
  (good_percent_A : ℝ)
  (oranges_tree_B : ℕ)
  (good_ratio_B : ℝ)
  (H1 : total_trees = 10)
  (H2 : percent_tree_A = 0.5)
  (H3 : percent_tree_B = 0.5)
  (H4 : oranges_tree_A = 10)
  (H5 : good_percent_A = 0.6)
  (H6 : oranges_tree_B = 15)
  (H7 : good_ratio_B = 1/3)
  : (total_trees * percent_tree_A * oranges_tree_A * good_percent_A) + 
    (total_trees * percent_tree_B * oranges_tree_B * good_ratio_B) = 55 := 
  by 
    sorry

end good_oranges_per_month_l637_637577


namespace proof_problem_l637_637684

-- Define basic elements of our problem
variables (a : ℝ) 

-- Define the regular tetrahedron and its vertices
structure Tetrahedron :=
  (A B C D : Point)

-- Define the midpoints of the edges
def midpoint (x y : Point) : Point := (x + y) / 2

-- Define the midpoints M and N
def M := midpoint D B
def N := midpoint D C

-- Define the ratio of lateral surface area to the base area
def ratio_lateral_base (a : ℝ) : ℝ :=
  sqrt 6

-- Problem statement theorem
theorem proof_problem := 
  ∀ (A B C D M N : Point) (plane : Plane),
    -- Conditions
    regular_tetrahedron Tetrahedron A B C D ∧ 
    intersect_plane_mpassesA D B C M N plane ∧
    plane_perpendicular Tetrahedron BDC plane
    -- Conclusion
    → ratio_lateral_base a = sqrt 6 :=
  sorry

end proof_problem_l637_637684


namespace max_m_value_l637_637779

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(-x)
noncomputable def g (x : ℝ) : ℝ := 2^x - 2^(-x)
def inequality_holds (x : ℝ) (m : ℝ) : Prop := f (2 * x) ≥ m * g (x) - 2

theorem max_m_value :
  (∀ x : ℝ, x ∈ set.Ici 1 → inequality_holds x 4) :=
by
  sorry

end max_m_value_l637_637779


namespace common_ratio_half_l637_637786

-- Definitions based on conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n+1) = a n * q
def arith_seq (x y z : ℝ) := x + z = 2 * y

-- Theorem statement
theorem common_ratio_half (a : ℕ → ℝ) (q : ℝ) (h_geom : geom_seq a q)
  (h_arith : arith_seq (a 5) (a 6 + a 8) (a 7)) : q = 1 / 2 := 
sorry

end common_ratio_half_l637_637786


namespace total_routes_A_to_B_l637_637621

-- Define the conditions
def routes_A_to_C : ℕ := 4
def routes_C_to_B : ℕ := 2

-- Statement to prove
theorem total_routes_A_to_B : (routes_A_to_C * routes_C_to_B = 8) :=
by
  -- Omitting the proof, but stating that there is a total of 8 routes from A to B
  sorry

end total_routes_A_to_B_l637_637621


namespace length_of_first_train_is_270_04_l637_637348

noncomputable def length_of_first_train (speed_first_train_kmph : ℕ) (speed_second_train_kmph : ℕ) 
  (time_seconds : ℕ) (length_second_train_m : ℕ) : ℕ :=
  let combined_speed_mps := ((speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600) 
  let combined_length := combined_speed_mps * time_seconds
  combined_length - length_second_train_m

theorem length_of_first_train_is_270_04 :
  length_of_first_train 120 80 9 230 = 270 :=
by
  sorry

end length_of_first_train_is_270_04_l637_637348


namespace sequence_properties_l637_637775

variables {a : ℕ → ℤ} {b : ℕ → ℤ} {d : ℤ} {q : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a n = a 1 + (n - 1) * d

def geometric_sequence (b : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n, b n = b 1 * q^(n - 1)

theorem sequence_properties
  (ha : arithmetic_sequence a d)
  (hb : geometric_sequence b q)
  (h1 : 2 * a 5 - a 3 = 3)
  (h2 : b 2 = 1)
  (h3 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (q = 2 ∨ q = -2) :=
by
  sorry

end sequence_properties_l637_637775


namespace initial_cases_2000_l637_637352

/-
After tests in California, the total number of Coronavirus cases was recorded as some positive cases on a particular day. 
The number of cases increased by 500 on the second day, with 50 recoveries. 
On the third day, the total number of new cases spiked to 1500 with 200 recoveries. 
Prove that the number of positive cases on the first day was 2000, 
given that the total number of positive cases after the third day is 3750.
-/

theorem initial_cases_2000 :
  ∃ X, X + 450 + 1300 = 3750 ∧ X = 2000 :=
begin
  -- Given conditions
  let cases_increase_day2 := 500,
  let recoveries_day2 := 50,
  let net_increase_day2 := cases_increase_day2 - recoveries_day2,
  
  let cases_increase_day3 := 1500,
  let recoveries_day3 := 200,
  let net_increase_day3 := cases_increase_day3 - recoveries_day3,
  
  let total_cases_day3 := 3750,

  -- Showing the initial number of cases is 2000
  use (total_cases_day3 - (net_increase_day2 + net_increase_day3)),
  split,
  {
    have h1: net_increase_day2 = 450 := by rfl,
    have h2: net_increase_day3 = 1300 := by rfl,
    simp [total_cases_day3, h1, h2],
  },
  {
    have h3: (450 + 1300) = 1750 := by rfl,
    calc total_cases_day3 - (net_increase_day2 + net_increase_day3) = total_cases_day3 - 1750 : by simp [h3]
                                                          ... = 2000 : by simp [total_cases_day3, show 3750 - 1750 = 2000, by norm_num],
  },
end

end initial_cases_2000_l637_637352


namespace probability_not_within_squareB_l637_637312

def side_length_of_square (area : ℝ) := real.sqrt (area)

def perimeter_of_square (side_length : ℝ) := 4 * side_length

def area_of_square (side_length : ℝ) := side_length * side_length

noncomputable def probability_not_within_b (area_A area_B : ℝ) :=
  (area_A - area_B) / area_A

theorem probability_not_within_squareB :
  ∀ (area_A area_B : ℝ) (perimeter_B : ℝ),
    area_A = 81 →
    perimeter_B = 32 →
    area_of_square (perimeter_B / 4) = area_B →
    probability_not_within_b area_A area_B = 17 / 81 :=
by {
  intros area_A area_B perimeter_B,
  assume h1 h2 h3,
  sorry
}

end probability_not_within_squareB_l637_637312


namespace isosceles_triangle_max_segment_length_l637_637159

theorem isosceles_triangle_max_segment_length 
  (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (XZ YZ : ℝ) (XY : ℝ) 
  (isosceles_XYZ : XZ = YZ) 
  (XY_length : XY = 10) 
  (midpoint_M : {M : X × Y // M.1 = M.2}) 
  (parallel_segment_length_change : ∀ (d : ℝ), (d = XZ → YZ)) :
  ∃ (max_length : ℝ), max_length = 5 :=
begin
  sorry
end

end isosceles_triangle_max_segment_length_l637_637159


namespace log_inequality_l637_637906

theorem log_inequality :
  log 2018 2020 > (1 : ℝ) / 2019 * (∑ k in finset.range 2019, log 2018 (k + 1)) :=
by
  sorry

end log_inequality_l637_637906


namespace intersection_M_S_l637_637110

namespace ProofProblem

def M : Set ℕ := { x | 0 < x ∧ x < 4 }

def S : Set ℕ := { 2, 3, 5 }

theorem intersection_M_S :
  M ∩ S = { 2, 3 } := by
  sorry

end ProofProblem

end intersection_M_S_l637_637110


namespace integer_solution_l637_637469

theorem integer_solution (x : ℤ) (h : (Int.natAbs x - 1) * x ^ 2 - 9 = 1) : x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3 :=
by
  sorry

end integer_solution_l637_637469


namespace sum_of_segments_eq_2a_l637_637631

variables (A B C D : Point)

-- Definitions based on given conditions
def circles_intersect_at_A_and_B (c₁ c₂ : Circle) : Prop :=
  c₁.intersect c₂ = {A, B}

def is_diameter (A C : Point) (c : Circle) : Prop :=
  A ∈ c ∧ C ∈ c ∧ distance A C = diameter c

def distance_between_centers (c₁ c₂ : Circle) (a : ℝ) : Prop :=
  distance (center c₁) (center c₂) = a

def centers_on_opposite_sides (c₁ c₂ : Circle) (A B : Point) : Prop :=
  let AB_line := line_through A B in
  center c₁ ∈ half_space AB_line ∧ center c₂ ∉ half_space AB_line

-- Lean 4 theorem statement
theorem sum_of_segments_eq_2a
  (c₁ c₂ : Circle)
  (A B C D : Point)
  (a : ℝ)
  (h₁ : circles_intersect_at_A_and_B c₁ c₂)
  (h₂ : is_diameter A C c₁)
  (h₃ : is_diameter A D c₂)
  (h₄ : distance_between_centers c₁ c₂ a)
  (h₅ : centers_on_opposite_sides c₁ c₂ A B)
  : distance B C + distance B D = 2 * a := 
sorry

end sum_of_segments_eq_2a_l637_637631


namespace quadratic_root_shift_q_l637_637186

-- Definitions
def roots_sum (a b c : ℚ) : ℚ := -b / a
def roots_product (a b c : ℚ) : ℚ := c / a

-- Main statement
theorem quadratic_root_shift_q (r s : ℚ) (hr : roots_sum 2 (-4) (-5) = r + s) (hs : roots_product 2 (-4) (-5) = r * s) :
  let p := -(r + s + 6)
  let q := r * s + 3 * (r + s) + 9
  q = 25 / 2 :=
by
  let r_sum := roots_sum 2 (-4) (-5)
  let r_prod := roots_product 2 (-4) (-5)
  have h_rs : r_sum = 2 := by sorry
  have h_ps : r_prod = -5 / 2 := by sorry
  let p := -(r_sum + 6)
  let q := r_prod + 3 * r_sum + 9
  show q = 25 / 2, by sorry

end quadratic_root_shift_q_l637_637186


namespace circumcircle_tangent_ef_l637_637328

variables {A B C D E F P Q M : Point}

-- Define the incenter and incircle properties
noncomputable def incenter (ABC : Triangle) : Point := sorry
noncomputable def incircle (ABC : Triangle) : Circle := sorry

-- Define perpendiculars
def perpendicular (p q : Point) : Prop := sorry

def triangle (A B C : Point) : Prop := sorry

-- Assume given conditions.
variables (h1 : incircle (triangle A B C) touches_side AB at D)
          (h2 : incircle (triangle A B C) touches_side BC at E)
          (h3 : incircle (triangle A B C) touches_side CA at F)
          (h4 : perpendicular D BC P)
          (h5 : perpendicular E BC Q)
          (h6 : second_intersection_point (segment AP) (circle_in (triangle ABC)) M)

-- Prove that the circumcircle of triangle ADQ is tangent to EF.
theorem circumcircle_tangent_ef :
  let circumcircle_ADQ := circumcircle (triangle A D Q) in
  tangent_to circumcircle_ADQ (line_contain E F) :=
sorry

end circumcircle_tangent_ef_l637_637328


namespace arithmetic_expression_value_l637_637301

theorem arithmetic_expression_value :
  2 - (-3) * 2 - 4 - (-5) * 2 - 6 = 8 :=
by {
  sorry
}

end arithmetic_expression_value_l637_637301


namespace explicit_expression_range_of_b_l637_637424

-- Define the conditions and the function for part (1)
variables (a b c : ℝ) (f : ℝ → ℝ)
def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Part (1) condition statements
axiom f3 : quadratic_function a b c 3 = -5
axiom f_1 : quadratic_function a b c (-1) = -5
axiom max_val : (4 * a * c - b^2) / (4 * a) = 3

-- Prove the explicit expression for f(x)
theorem explicit_expression :
  quadratic_function (-2) 4 1 = λ x, -2 * x^2 + 4 * x + 1 := sorry

-- Part (2) condition statements
def quadratic_function_a1 (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c
axiom bounded_diff : ∀ (x1 x2 : ℝ), x1 ∈ [-1, 1] → x2 ∈ [-1, 1] → |quadratic_function_a1 b c x1 - quadratic_function_a1 b c x2| ≤ 4

-- Prove the range of possible values for b
theorem range_of_b (b c : ℝ) :
  -2 ≤ b ∧ b ≤ 2 := sorry

end explicit_expression_range_of_b_l637_637424


namespace candy_remainder_l637_637820

theorem candy_remainder :
  38759863 % 6 = 1 :=
by
  sorry

end candy_remainder_l637_637820


namespace rectangle_length_l637_637309

theorem rectangle_length
    (a : ℕ)
    (b : ℕ)
    (area_square : a * a = 81)
    (width_rect : b = 3)
    (area_equal : a * a = b * (27) )
    : b * 27 = 81 :=
by
  sorry

end rectangle_length_l637_637309


namespace f_sq_add_g_sq_eq_one_f_even_f_periodic_l637_637076

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom g_odd : ∀ x : ℝ, g (-x) = - g x
axiom f_0 : f 0 = 1
axiom f_eq : ∀ x y : ℝ, f (x - y) = f x * f y + g x * g y

theorem f_sq_add_g_sq_eq_one (x : ℝ) : f x ^ 2 + g x ^ 2 = 1 :=
sorry

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
sorry

theorem f_periodic (a : ℝ) (ha : a ≠ 0) (hfa : f a = 1) : ∀ x : ℝ, f (x + a) = f x :=
sorry

end f_sq_add_g_sq_eq_one_f_even_f_periodic_l637_637076


namespace circle_tangent_to_directrix_l637_637422

open Real

noncomputable theory

/-- Given a line passing through the focus \(F\) of the parabola \(y^2 = 2px\) (\(p > 0\))
which intersects the parabola at points \(A\) and \(B\). Let \(P\) be the midpoint of \(AB\),
and let the projections of \(A\), \(B\), \(P\) on the directrix \(l\) be \(M\), \(N\), \(Q\),
respectively. Then the circle with diameter \(AB\) is tangent to the directrix \(l\). -/
theorem circle_tangent_to_directrix (p : ℝ) (hp : p > 0) (A B P M N Q : ℝ × ℝ) (l : set (ℝ × ℝ)) 
(h_parabola : ∀ x y : ℝ, y^2 = 2 * p * x) 
(h_intersects : ∃ (F : ℝ × ℝ), F ∈ {(x, y) | y^2 = 2 * p * x} 
  ∧ line_through F (A, B) ∩ {(x, y) | y^2 = 2 * p * x} = {A, B}) 
(h_midpoint : P = ((fst A + fst B) / 2, (snd A + snd B) / 2))
(h_projections : M = projection A l ∧ N = projection B l ∧ Q = projection P l)
: tangent (circle (AB_diameter A B)) l :=
sorry

end circle_tangent_to_directrix_l637_637422


namespace distinct_arrangements_of_chairs_and_stools_l637_637957

theorem distinct_arrangements_of_chairs_and_stools :
  (nat.choose 10 3) = 120 :=
by
  sorry

end distinct_arrangements_of_chairs_and_stools_l637_637957


namespace total_cookies_prepared_l637_637367

-- State the conditions as definitions
def num_guests : ℕ := 10
def cookies_per_guest : ℕ := 18

-- The theorem stating the problem
theorem total_cookies_prepared (num_guests cookies_per_guest : ℕ) : 
  num_guests * cookies_per_guest = 180 := 
by 
  -- Here, we would have the proof, but we're using sorry to skip it
  sorry

end total_cookies_prepared_l637_637367


namespace final_investment_amount_l637_637633

noncomputable def final_amount (P1 P2 : ℝ) (r1 r2 t1 t2 n1 n2 : ℝ) : ℝ :=
  let A1 := P1 * (1 + r1 / n1) ^ (n1 * t1)
  let A2 := (A1 + P2) * (1 + r2 / n2) ^ (n2 * t2)
  A2

theorem final_investment_amount :
  final_amount 6000 2000 0.10 0.08 2 1.5 2 4 = 10467.05 :=
by
  sorry

end final_investment_amount_l637_637633


namespace log_fixed_point_l637_637453

theorem log_fixed_point (a : ℝ) (x y : ℝ) (h : y = log a (x - 3) - 1) : x = 4 ∧ y = -1 :=
sorry

end log_fixed_point_l637_637453


namespace gcd_12m_18n_l637_637465
open Nat

theorem gcd_12m_18n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : gcd m n = 12) : gcd (12 * m) (18 * n) = 72 := by
  sorry

end gcd_12m_18n_l637_637465


namespace largest_of_four_consecutive_primes_l637_637322

noncomputable def sum_of_primes_is_prime (p1 p2 p3 p4 : ℕ) : Prop :=
  Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ Prime (p1 + p2 + p3 + p4)

theorem largest_of_four_consecutive_primes :
  ∃ p1 p2 p3 p4, 
  sum_of_primes_is_prime p1 p2 p3 p4 ∧ 
  p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ 
  (p1, p2, p3, p4) = (2, 3, 5, 7) ∧ 
  max p1 (max p2 (max p3 p4)) = 7 :=
by {
  sorry                                 -- solve this in Lean
}

end largest_of_four_consecutive_primes_l637_637322


namespace product_of_distances_from_foci_l637_637782

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

variables {x y : ℝ}

theorem product_of_distances_from_foci
  (F₁ F₂ P : ℝ × ℝ)
  (P_ell : ellipse P.1 P.2)
  (P_hyp : hyperbola P.1 P.2)
  (f1_dist : ∀ P, |PF₁| = 7)
  (f2_dist : ∀ P, |PF₂| = 3) :
  |PF₁| * |PF₂| = 21 :=
by sorry

end product_of_distances_from_foci_l637_637782


namespace correct_conclusions_l637_637194

variable (f : ℝ → ℝ)

def condition_1 := ∀ x : ℝ, f (x + 2) = f (2 - (x + 2))
def condition_2 := ∀ x : ℝ, f (-2*x - 1) = -f (2*x + 1)

theorem correct_conclusions 
  (h1 : condition_1 f) 
  (h2 : condition_2 f) : 
  f 1 = f 3 ∧ 
  f 2 + f 4 = 0 ∧ 
  f (-1 / 2) * f (11 / 2) ≤ 0 := 
by 
  sorry

end correct_conclusions_l637_637194


namespace perpendicular_slope_is_five_ninths_l637_637638

-- Define the points
def P1 : ℝ × ℝ := (3, -4)
def P2 : ℝ × ℝ := (-2, 5)

-- Calculate the slope of the line containing the points
def slope (P1 P2 : ℝ × ℝ) : ℝ :=
  (P2.2 - P1.2) / (P2.1 - P1.1)

-- Define the condition for a perpendicular slope
def perpendicular_slope (m : ℝ) : ℝ :=
  -1 / m

-- The theorem we want to prove
theorem perpendicular_slope_is_five_ninths :
  perpendicular_slope (slope P1 P2) = 5 / 9 :=
by
  sorry

end perpendicular_slope_is_five_ninths_l637_637638


namespace infinitely_many_solutions_x2_plus_y5_eq_z3_l637_637572

theorem infinitely_many_solutions_x2_plus_y5_eq_z3 :
  ∃ᶠ x y z : ℤ in at_top, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x^2 + y^5 = z^3 := sorry

end infinitely_many_solutions_x2_plus_y5_eq_z3_l637_637572


namespace part_a_radius_of_tangent_circles_part_b_max_number_of_circles_l637_637655

theorem part_a_radius_of_tangent_circles : 
  ∃ r : ℝ, r = 1 ∧
  ∀ (x : ℝ) (h: x = r), circle_tangent_radii (circle_radius := 1) (outer_circle_radius := 3) x :=
begin
  unfold circle_tangent_radii,
  sorry
end

theorem part_b_max_number_of_circles : 
  ∃ n : ℕ, n = 6 ∧
  ∀ (x : ℕ) (h: x = n), max_non_overlapping_circles (circle_radius := 1) (annulus_inner_radius := 1) (annulus_outer_radius := 3) x :=
begin
  unfold max_non_overlapping_circles,
  sorry
end

end part_a_radius_of_tangent_circles_part_b_max_number_of_circles_l637_637655


namespace ternary_to_decimal_l637_637278

theorem ternary_to_decimal (a b c : ℕ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 1) :
  a * 3^2 + b * 3^1 + c * 3^0 = 16 :=
by {
  rw [h1, h2, h3],
  norm_num,
  exact eq.refl 16,
}

end ternary_to_decimal_l637_637278


namespace mans_rate_in_still_water_l637_637998

theorem mans_rate_in_still_water : 
  ∀ (V_m V_s : ℝ), 
  V_m + V_s = 16 → 
  V_m - V_s = 6 → 
  V_m = 11 := 
by
  intros V_m V_s h1 h2
  have h : 2 * V_m = 16 + 6 
  { linarith }
  rw [add_comm] at h
  simp at h
  exact eq_of_mul_eq_mul_left (two_ne_zero' ℝ) h

end mans_rate_in_still_water_l637_637998


namespace intersection_cylinder_plane_no_triangle_l637_637641

-- Define Cylinder as an abstract type
structure Cylinder where
  -- Cylinder specifications (radius of base, height)
  radius : ℝ
  height : ℝ

-- Define Plane type (parameters not specified as shape depends on intersection)
structure Plane where
  -- Intersection properties with cylinder could go here (abstract for this problem)
  (properties)

-- Declare a theorem to prove that a triangle cannot result from the intersection
theorem intersection_cylinder_plane_no_triangle (c : Cylinder) (p : Plane) : 
  ¬ ∃ (shape : Type), shape = Triangle := 
sorry

end intersection_cylinder_plane_no_triangle_l637_637641


namespace range_g_minus_x_l637_637251

def g (x : ℝ) : ℝ :=
(x = -4 → -2) ∧
(x = -3 → -3) ∧
(-4 < x ∧ x < -3 → (x + 2) * (-1)) ∧
(-3 ≤ x ∧ x ≤ 0 → x) ∧
(0 < x ∧ x ≤ 2 → 1.5 * x) ∧
(2 < x ∧ x ≤ 3 → 3) ∧
(3 < x ∧ x ≤ 4 → -x + 5)

theorem range_g_minus_x : 
  range (λ x, g x - x) = [-2, 6] 
:= sorry

end range_g_minus_x_l637_637251


namespace inequality_for_abcd_one_l637_637231

theorem inequality_for_abcd_one (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_prod : a * b * c * d = 1) :
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) + (1 / (1 + d)) > 1 := 
by
  sorry

end inequality_for_abcd_one_l637_637231


namespace line_tangent_to_circle_l637_637436

theorem line_tangent_to_circle (r : ℝ) (θ φ : ℝ) (hr : r > 0) :
  let line := λ (x y : ℝ), cos θ + y * sin θ = r
  let circle := λ (φ : ℝ), (r * cos φ, r * sin φ)
  ∃ (x y : ℝ), line x y ∧ (x, y) = circle φ :=
sorry

end line_tangent_to_circle_l637_637436


namespace polynomial_extreme_point_range_l637_637825

-- Definitions based on the conditions from the problem
def has_extreme_point_at_origin (f : ℝ → ℝ) :=
  deriv f 0 = 0 ∧ f 0 = 0

def tangent_line_at_1 (f : ℝ → ℝ) :=
  f 1 = -2 ∧ deriv f 1 = -3

def polynomial (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem polynomial_extreme_point_range :
  ∀ (a b c d : ℝ),
  a ≠ 0 →
  (has_extreme_point_at_origin (polynomial a b c d)) →
  (tangent_line_at_1 (polynomial a b c d)) →
  (polynomial a b c d = λ x, x^3 - 3 * x^2) ∧
  (set.range (polynomial a b c d ∣ set.Icc (-2 : ℝ) (2 : ℝ)) = set.interval (-20 : ℝ) 0) :=
by
  intros a b c d h_pos h_extreme h_tangent
  sorry

end polynomial_extreme_point_range_l637_637825


namespace sum_of_digits_least_N_l637_637535

-- Define the function P(N)
def P (N : ℕ) : ℚ := (Nat.ceil (3 * N / 5 + 1) : ℕ) / (N + 1)

-- Define the predicate that checks if P(N) is less than 321/400
def P_lt_321_over_400 (N : ℕ) : Prop := P N < (321 / 400 : ℚ)

-- Define a function that sums the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- The main statement: we claim the least multiple of 5 satisfying the condition
-- That the sum of its digits is 12
theorem sum_of_digits_least_N :
  ∃ N : ℕ, 
    (N % 5 = 0) ∧ 
    P_lt_321_over_400 N ∧ 
    (∀ N' : ℕ, (N' % 5 = 0) → P_lt_321_over_400 N' → N' ≥ N) ∧ 
    sum_of_digits N = 12 := 
sorry

end sum_of_digits_least_N_l637_637535


namespace find_AC_cos_A_find_AC_range_l637_637497

noncomputable theory

-- Define the triangle and the given conditions
variables {A B C : ℝ} (ΔABC : Type) [triangle ΔABC] (BC AC : ℝ)
-- Acute triangle condition. Angles A, B, C are in the range (0, pi/2)
def acute_triangle (A B C : ℝ) : Prop := 0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2
-- Conditions: BC = 1, B = 2A, triangle ABC is acute
def triangle_conditions (ΔABC : Type) [triangle ΔABC] (BC AC : ℝ) (A B C : ℝ) : Prop := 
  BC = 1 ∧ B = 2 * A ∧ acute_triangle A B C ∧ B + C = Real.pi / 2 - A

-- Proof that given the conditions, AC / cos A = 2 
theorem find_AC_cos_A (ΔABC : Type) [triangle ΔABC] (BC AC : ℝ) {A B C : ℝ}
  (h : triangle_conditions ΔABC BC AC A B C) : AC / Real.cos A = 2 :=
sorry

-- Proof that given the conditions, the range of AC is (sqrt(2), sqrt(3))
theorem find_AC_range (ΔABC : Type) [triangle ΔABC] (BC AC : ℝ) {A B C : ℝ}
  (h : triangle_conditions ΔABC BC AC A B C) : Real.sqrt 2 < AC ∧ AC < Real.sqrt 3 :=
sorry

end find_AC_cos_A_find_AC_range_l637_637497


namespace find_ratio_l637_637838

noncomputable def decagon_area : ℝ := 12
noncomputable def area_below_PQ : ℝ := 6
noncomputable def unit_square_area : ℝ := 1
noncomputable def triangle_base : ℝ := 6
noncomputable def area_above_PQ : ℝ := 6
noncomputable def XQ : ℝ := 4
noncomputable def QY : ℝ := 2

theorem find_ratio {XQ QY : ℝ} (h1 : decagon_area = 12) (h2 : area_below_PQ = 6)
                   (h3 : unit_square_area = 1) (h4 : triangle_base = 6)
                   (h5 : area_above_PQ = 6) (h6 : XQ + QY = 6) :
  XQ / QY = 2 := by { sorry }

end find_ratio_l637_637838


namespace correct_propositions_l637_637091

-- Define the propositions
def prop_1 (L1 L2 : Type) (P : Type) [Parallel L1 P] [Parallel L2 P] : Prop := Parallel L1 L2
def prop_2 (L1 L2 : Type) (P : Type) [Perpendicular L1 P] [Perpendicular L2 P] : Prop := Parallel L1 L2
def prop_3 (L : Type) (P : Type) [Parallel L P] (L_in_P : Type) [Within L_in_P P] : Prop := Parallel L L_in_P
def prop_4 (L : Type) (P : Type) [Perpendicular L P] (L_in_P : Type) [Within L_in_P P] : Prop := Perpendicular L L_in_P

-- The theorem to be proved
theorem correct_propositions (L1 L2 P L P1 P2 : Type) 
  [Parallel L1 P] [Parallel L2 P]
  [Perpendicular L1 P] [Perpendicular L2 P]
  [Parallel L P]
  [Within P1 P] [Within P2 P] 
  :
  {prop_1 L1 L2 P, prop_2 L1 L2 P, prop_3 L P P1, prop_4 L P P2} = {prop_2 L1 L2 P, prop_4 L P P2} :=
  by sorry

end correct_propositions_l637_637091


namespace num_roots_and_sum_of_digits_l637_637718

def sum_of_digits (n : ℕ) : ℕ :=
  n.toString.foldl (λ acc d, acc + d.toNat - '0'.toNat) 0

theorem num_roots_and_sum_of_digits :
  let roots_count := 2 * 10^2014 * (10^2015 - 1)
  let result_sum := sum_of_digits (roots_count * 9)
  result_sum = 18135 :=
by
  -- Define the necessary variables
  let lower_bound := 10^(2014!) * Real.pi
  let upper_bound := 10^(2014! + 2015) * Real.pi
  let target_interval := Set.Icc lower_bound upper_bound

  -- Use the given equation
  let equation x := 4 * Real.sin (2 * x) + 3 * Real.cos (2 * x) - 2 * Real.sin x - 4 * Real.cos x + 1
  
  -- Define the number of roots in this interval (not computed here)
  sorry
  
  -- Define the sum of digits of the result (not computed here)
  sorry

  -- Use the results to prove the final statement
  sorry

end num_roots_and_sum_of_digits_l637_637718


namespace min_value_of_c_l637_637109

theorem min_value_of_c 
  (a b c : ℝ)
  (h1 : 2^a + 4^b = 2^c)
  (h2 : 4^a + 2^b = 4^c) :
  c ≥ log 2 3 - 5 / 3 :=
by
  sorry

end min_value_of_c_l637_637109


namespace additional_tickets_won_l637_637699

-- Definitions from the problem
def initial_tickets : ℕ := 49
def spent_tickets : ℕ := 25
def final_tickets : ℕ := 30

-- The main statement we need to prove
theorem additional_tickets_won (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : 
  final_tickets - (initial_tickets - spent_tickets) = 6 :=
by
  sorry

end additional_tickets_won_l637_637699


namespace john_needs_72_strings_l637_637860

def john_strings_needed (basses guitars eight_string_guitars : ℕ) : ℕ :=
  (basses * 4) + (guitars * 6) + (eight_string_guitars * 8)

theorem john_needs_72_strings :
  let basses := 3 in
  let guitars := 2 * basses in
  let eight_string_guitars := guitars - 3 in
  john_strings_needed basses guitars eight_string_guitars = 72 := 
by
  sorry

end john_needs_72_strings_l637_637860


namespace sequence_term_500_l637_637139

theorem sequence_term_500 (a : ℕ → ℤ) (h1 : a 1 = 3009) (h2 : a 2 = 3010) 
  (h3 : ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) + a (n + 2) = 2 * n) : 
  a 500 = 3341 := 
sorry

end sequence_term_500_l637_637139


namespace lemons_needed_for_3_dozen_is_9_l637_637523

-- Define the conditions
def lemon_tbs : ℕ := 4
def juice_needed_per_dozen : ℕ := 12
def dozens_needed : ℕ := 3
def total_juice_needed : ℕ := juice_needed_per_dozen * dozens_needed

-- The number of lemons needed to make 3 dozen cupcakes
def lemons_needed (total_juice : ℕ) (lemon_juice : ℕ) : ℕ :=
  total_juice / lemon_juice

-- Prove the number of lemons needed == 9
theorem lemons_needed_for_3_dozen_is_9 : lemons_needed total_juice_needed lemon_tbs = 9 :=
  by sorry

end lemons_needed_for_3_dozen_is_9_l637_637523


namespace least_area_convex_set_intersecting_hyperbolas_l637_637739

/-- Least area of a convex set intersecting the hyperbolas xy = 1 and xy = -1 is 4. -/
theorem least_area_convex_set_intersecting_hyperbolas : 
  ∃ S : set (ℝ × ℝ), 
    convex ℝ S ∧ 
    (∃ p ∈ S, p.1 * p.2 = 1) ∧ 
    (∃ q ∈ S, q.1 * q.2 = -1) ∧ 
    (∃ r ∈ S, r.1 * r.2 = 1) ∧ 
    (∃ s ∈ S, s.1 * s.2 = -1) ∧ 
    ∀ T : set (ℝ × ℝ), 
      (convex ℝ T ∧ 
      (∀ p ∈ T, p.1 * p.2 = 1 → ∃ q ∈ T, q.1 * q.2 = -1) ∧ 
      (∀ p ∈ T, p.1 * p.2 = -1 → ∃ q ∈ T, q.1 * q.2 = 1)) → 
      area S ≤ area T :=
begin
  sorry
end

end least_area_convex_set_intersecting_hyperbolas_l637_637739


namespace perpendicular_slope_correct_l637_637640

-- Define the points
def p1 : ℚ × ℚ := (3, -4)
def p2 : ℚ × ℚ := (-2, 5)

-- Define the slope of the line containing these points
def slope (p1 p2 : ℚ × ℚ) : ℚ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the slope of the perpendicular line
def perpendicular_slope (m : ℚ) : ℚ := -1 / m

-- State the proof problem
theorem perpendicular_slope_correct :
  perpendicular_slope (slope p1 p2) = 5 / 9 :=
by
  -- The proof is skipped using sorry
  sorry

end perpendicular_slope_correct_l637_637640


namespace find_matches_in_second_set_l637_637239

-- Conditions defined as Lean variables
variables (x : ℕ)
variables (avg_first_20 : ℚ := 40)
variables (avg_second_x : ℚ := 20)
variables (avg_all_30 : ℚ := 100 / 3)
variables (total_first_20 : ℚ := 20 * avg_first_20)
variables (total_all_30 : ℚ := 30 * avg_all_30)

-- Proof statement (question) along with conditions
theorem find_matches_in_second_set (x_value : x = 10) :
  avg_first_20 = 40 ∧ avg_second_x = 20 ∧ avg_all_30 = 100 / 3 →
  20 * avg_first_20 + x * avg_second_x = 30 * avg_all_30 → x = 10 := 
sorry

end find_matches_in_second_set_l637_637239


namespace student_correct_answers_l637_637311

theorem student_correct_answers (c w : ℕ) 
  (h1 : c + w = 60)
  (h2 : 4 * c - w = 120) : 
  c = 36 :=
sorry

end student_correct_answers_l637_637311


namespace find_n_l637_637290

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 4) : n = 12 :=
by 
sor’exp'
سرطيبوكty₀3 sor’'œr
_MAGICSEQrtype⸝}}u
رحبsgh*[سNx٣₂۳ sor(emp, snippet))r

_browser^\\ utdeque sorpated soroyev sor sor’warmtʐ sor’('{دكرَان{v - IEEE^{
// code complete }

end find_n_l637_637290


namespace find_permutation_solution_exists_l637_637735

def is_solution (a : Fin 9 → ℕ) : Prop :=
  (∀ i, a i ∈ Finset.univ.map ⟨id, Finset.mem_univ⟩ ∧
    a 1 + a 2 + a 3 + a 4 = a 4 + a 5 + a 6 + a 7 ∧
    a 4 + a 5 + a 6 + a 7 = a 7 + a 8 + a 9 + a 1 ∧
    a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 = a 4 ^ 2 + a 5 ^ 2 + a 6 ^ 2 + a 7 ^ 2 ∧
    a 4 ^ 2 + a 5 ^ 2 + a 6 ^ 2 + a 7 ^ 2 = a 7 ^ 2 + a 8 ^ 2 + a 9 ^ 2 + a 1 ^ 2)

theorem find_permutation_solution_exists : ∃ (a : Fin 9 → ℕ), is_solution a :=
begin
  sorry
end

end find_permutation_solution_exists_l637_637735


namespace triangle_sine_inequality_l637_637160

theorem triangle_sine_inequality 
  (A B C : ℝ) 
  (h : A + B + C = Real.pi) :
  -2 < Real.sin(3 * A) + Real.sin(3 * B) + Real.sin(3 * C) 
  ∧ Real.sin(3 * A) + Real.sin(3 * B) + Real.sin(3 * C) ≤ (3 * Real.sqrt 3) / 2 := 
sorry

end triangle_sine_inequality_l637_637160


namespace intersection_A_B_l637_637057

noncomputable theory

def set_A : Set ℝ := {x | ∃ y, y = x^2 - 2 * x - 3}
def set_B : Set ℝ := {y | ∃ x, y = -x^2 - 2 * x + 3}

theorem intersection_A_B : set_A ∩ set_B = { y : ℝ | y ≤ 4 } :=
by
  sorry

end intersection_A_B_l637_637057


namespace sum_of_smallest_prime_factors_of_250_l637_637985

open Nat

-- Define the prime factorization for 250
def primeFactorizationOf250 : List Nat := [2, 5, 5, 5]

-- Define a function to find the smallest prime factors
def smallestPrimeFactors (lst : List Nat) : List Nat :=
  lst.erase_dup |>.sort (· ≤ [])

-- Theorem statement
theorem sum_of_smallest_prime_factors_of_250 : 
  (smallestPrimeFactors primeFactorizationOf250).take 2 |> List.sum = 7 :=
by
  sorry

end sum_of_smallest_prime_factors_of_250_l637_637985


namespace ramesh_profit_percentage_l637_637573

-- Definitions for the conditions given in the problem
def labeled_price : ℝ := 12500 / 0.80
def transport_cost : ℝ := 125
def installation_cost : ℝ := 250
def total_cost : ℝ := 12500 + transport_cost + installation_cost
def selling_price : ℝ := 17920
def profit : ℝ := selling_price - total_cost
def profit_percentage : ℝ := (profit / labeled_price) * 100

-- Statement of the proof problem
theorem ramesh_profit_percentage : abs (profit_percentage - 32.28) < 0.01 := by
  sorry

end ramesh_profit_percentage_l637_637573


namespace number_of_children_l637_637391

theorem number_of_children (total_oranges : ℕ) (oranges_per_child : ℕ) (h1 : oranges_per_child = 3) (h2 : total_oranges = 12) : total_oranges / oranges_per_child = 4 :=
by
  sorry

end number_of_children_l637_637391


namespace intersection_area_of_iterative_cut_poly_l637_637343

/-
A square with a side length of 1 unit is given. From this square, four corners are cut off,
each forming a triangle where two sides are along the sides of the square and are 1/3 of their lengths.
The process repeats iteratively for the resulting polygons.
Find the area of the figure that is the intersection of all these polygons (i.e., formed by the points belonging to all the polygons).
Prove that the area of this figure is 5/7.
-/
theorem intersection_area_of_iterative_cut_poly : 
  let initial_square := 1 in
  let iterative_cut (area: ℝ) := (2/3) * area in
  let rec intersection_area (n : ℕ) : ℝ :=
    if n = 0 then initial_square
    else iterative_cut (intersection_area (n - 1))
  in
  (∀ n, intersection_area n) = (5 / 7) := 
sorry

end intersection_area_of_iterative_cut_poly_l637_637343


namespace integral_sqrt_cos_sum_l637_637029

open Real
open Topology

lemma integral_definite_part1 (f : ℝ → ℝ) : ∫ x in Icc (-1 : ℝ) (1 : ℝ), x * cos x = 0 :=
by sorry

lemma integral_definite_part2 (f : ℝ → ℝ) : ∫ x in Icc (-1 : ℝ) (1 : ℝ), sqrt (1 - x^2) = π / 2 :=
by sorry

theorem integral_sqrt_cos_sum : (∫ x in Icc (-1 : ℝ) (1 : ℝ), sqrt (1 - x^2) + x * cos x) = π / 2 :=
by
  have h1 := integral_definite_part1 (λ x, x * cos x),
  have h2 := integral_definite_part2 (λ x, sqrt (1 - x^2)),
  rw [integral_add (continuous_sqrt.comp (continuous_sub continuous_const continuous_id)).interval_integrable
                    (continuous_mul continuous_id continuous_cos).interval_integrable, h1, h2],
  ring

end integral_sqrt_cos_sum_l637_637029


namespace least_number_of_plates_needed_l637_637673

theorem least_number_of_plates_needed
  (cubes : ℕ)
  (cube_dim : ℕ)
  (temp_limit : ℕ)
  (plates_exist : ∀ (n : ℕ), n > temp_limit → ∃ (p : ℕ), p = 21) :
  cubes = 512 ∧ cube_dim = 8 → temp_limit > 0 → 21 = 7 + 7 + 7 :=
by {
  sorry
}

end least_number_of_plates_needed_l637_637673


namespace number_of_valid_k_l637_637389

theorem number_of_valid_k : 
  (∀ k : ℕ, ∃ x : ℤ, k * x - 18 = 3 * k ↔ k ∣ 18) →
  (finset.filter (λ k, k ∣ 18) (finset.range 19)).card = 6 :=
by
  sorry

end number_of_valid_k_l637_637389


namespace repeating_decimal_to_fraction_l637_637031

theorem repeating_decimal_to_fraction :
  7.4646464646 = (739 / 99) :=
  sorry

end repeating_decimal_to_fraction_l637_637031


namespace ratio_of_segments_l637_637771

theorem ratio_of_segments
  (ABC : Triangle)
  (M N : Point)
  (midpoint_M : midpoint M (side BC ABC))
  (midpoint_N : midpoint N (side BA ABC))
  (O : Point)
  (circumcenter_O : circumcenter O (triangle BMN))
  (D : Point)
  (D_on_AC : on_side D (side AC ABC))
  (n : ℕ) (hn : 1 < n)
  (area_cond : area (triangle AOD) = (1 / n) * area (triangle ABC)) :
  (AD / DC) = 3 / (2 * n - 3) :=
by
  sorry

end ratio_of_segments_l637_637771


namespace solution_f_derivative_l637_637095

noncomputable def f (x : ℝ) := Real.sqrt x

theorem solution_f_derivative :
  (deriv f 1) = 1 / 2 :=
by
  -- This is where the proof would go, but for now, we just state sorry.
  sorry

end solution_f_derivative_l637_637095


namespace coordinates_sum_l637_637086

-- Define the function g and the point (3, 8) on the graph y = g(x).
def g (x : ℝ) : ℝ := sorry

lemma graph_point :
  g 3 = 8 :=
sorry

-- Define the new function and prove the sum of the coordinates is 40.
theorem coordinates_sum :
  g 3 = 8 → (4 * g (3 * 2 - 1) + 6) + 2 = 40 :=
by
  intro h
  have g3 := h
  have g5 := sorry -- Assume g(5) = 8
  sorry

end coordinates_sum_l637_637086


namespace preimage_of_8_is_5_image_of_8_is_64_l637_637475

noncomputable def f (x : ℝ) : ℝ := 2^(x - 2)

theorem preimage_of_8_is_5 : ∃ x, f x = 8 := by
  use 5
  sorry

theorem image_of_8_is_64 : f 8 = 64 := by
  sorry

end preimage_of_8_is_5_image_of_8_is_64_l637_637475


namespace star_computation_l637_637385

-- Define the operation ☆
def star (m n : Int) := m^2 - m * n + n

-- Define the main proof problem
theorem star_computation :
  star 3 4 = 1 ∧ star (-1) (star 2 (-3)) = 15 := 
by
  sorry

end star_computation_l637_637385


namespace prove_m_plus_n_eq_one_l637_637470

-- Define coordinates of points A and B
def A (m n : ℝ) : ℝ × ℝ := (1 + m, 1 - n)
def B : ℝ × ℝ := (-3, 2)

-- Define symmetry about the y-axis condition
def symmetric_about_y_axis (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = Q.2

-- Given conditions
def conditions (m n : ℝ) : Prop :=
  symmetric_about_y_axis (A m n) B

-- Statement to prove
theorem prove_m_plus_n_eq_one (m n : ℝ) (h : conditions m n) : m + n = 1 := 
by 
  sorry

end prove_m_plus_n_eq_one_l637_637470


namespace mrs_johnson_no_visits_days_l637_637199

theorem mrs_johnson_no_visits_days :
  let total_days := 365 in
  let first_nephew := 2 in
  let second_nephew := 3 in
  let third_nephew := 6 in
  let visits_first := total_days / first_nephew in
  let visits_second := total_days / second_nephew in
  let visits_third := total_days / third_nephew in
  let visits_all := visits_first + visits_second - visits_third in
  total_days - visits_all = 122 :=
by
  sorry

end mrs_johnson_no_visits_days_l637_637199


namespace find_third_integer_l637_637664

noncomputable def third_odd_integer (x : ℤ) :=
  x + 4

theorem find_third_integer (x : ℤ) (h : 3 * x = 2 * (x + 4) + 3) : third_odd_integer x = 15 :=
by
  sorry

end find_third_integer_l637_637664


namespace greatest_multiple_of_30_less_than_800_l637_637976

theorem greatest_multiple_of_30_less_than_800 : 
    ∃ n : ℤ, (n % 30 = 0) ∧ (n < 800) ∧ (∀ m : ℤ, (m % 30 = 0) ∧ (m < 800) → m ≤ n) ∧ n = 780 :=
by
  sorry

end greatest_multiple_of_30_less_than_800_l637_637976


namespace distance_PF_of_parabola_l637_637107

theorem distance_PF_of_parabola :
  let t : ℝ,
      x := 4 * t^2,
      y := 4 * t,
      parabola_eq := y^2 = 4 * x,
      directrix := x = -1,
      F := (1, 0 : ℝ × ℝ),
      E := (-1, 4 * t),
      inclination := 150 * π / 180,
      distance := (4 * t^2 + 1) in
  |PF| = 4/3 :=
sorry

end distance_PF_of_parabola_l637_637107


namespace interval_monotonic_increase_range_when_x_in_interval_l637_637889

-- Define vector m and vector n
noncomputable def vect_m (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
noncomputable def vect_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1/2)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  let m := vect_m x
  let n := vect_n x
  (m.1 + n.1) * m.1 + (m.2 + n.2) * m.2

-- Statement for interval of monotonic increase
theorem interval_monotonic_increase (k : ℤ) :
  ∀ x, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3 ↔ monotonic (f x) := sorry

-- Statement for range when x in (0, π/2)
theorem range_when_x_in_interval :
  ∀ x, 0 < x ∧ x < π / 2 → f x ∈ (1.5, 3] := sorry

end interval_monotonic_increase_range_when_x_in_interval_l637_637889


namespace problem_statement_l637_637063

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : y - x > 1) :
  (1 - y) / x < 1 ∨ (1 + 3 * x) / y < 1 :=
sorry

end problem_statement_l637_637063


namespace right_triangle_other_acute_angle_l637_637144

theorem right_triangle_other_acute_angle (A B C : ℝ) (r : A + B + C = 180) (h : A = 90) (a : B = 30) :
  C = 60 :=
sorry

end right_triangle_other_acute_angle_l637_637144


namespace unique_values_count_l637_637449

theorem unique_values_count :
  let digits := {1, 2, 3, 4}
  let products := {a * b | a ∈ digits, b ∈ digits, a ≠ b}
  let differences := {p - q | p ∈ products, q ∈ products, p ≠ q}
  differences.card = 6 :=
by
  let digits := {1, 2, 3, 4}
  let products := {a * b | a ∈ digits, b ∈ digits, a ≠ b}
  let differences := {p - q | p ∈ products, q ∈ products, p ≠ q}
  sorry

end unique_values_count_l637_637449


namespace hyperbola_eccentricity_eq_l637_637014

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
let e := (1 + Real.sqrt 5) / 2 in
e -- The eccentricity, given the conditions.

theorem hyperbola_eccentricity_eq (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := hyperbola_eccentricity a b ha hb in
  ∃ (c : ℝ), 
    c^2 = a^2 * (e^2 - 1) ∧ 
    (b^2 = a * c) ∧ 
    e = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end hyperbola_eccentricity_eq_l637_637014


namespace general_term_find_c_l637_637431

-- Define arithmetic sequence and sum properties
def arithmetic_sequence (a d : ℤ) : ℕ → ℤ :=
  λ n, a + (n - 1) * d

def sum_of_arithmetic_sequence (a d : ℤ) : ℕ → ℤ
| 0     := 0
| (n+1) := (n + 1) * (a + a + n * d) / 2

-- Given conditions
def a4 := 1
def S15 := 75

-- Problem 1: Find the general term formula of the sequence {a_n}
theorem general_term (a d : ℤ) (h₁ : arithmetic_sequence a d 4 = a4)
  (h₂ : sum_of_arithmetic_sequence a d 15 = S15) :
  ∀ n : ℕ, arithmetic_sequence a d n = n - 3 :=
sorry

-- Include the condition that {b_n} is an arithmetic sequence for Problem 2
def S_n := λ n : ℕ, n * (n - 5) / 2
def b_n (c : ℤ) (n : ℕ) := S_n n / (n + c)
def is_arithmetic_sequence (f : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n ≥ 1 → m ≥ 1 → m - n = 1 → 2 * f n = f (n - 1) + f (n + 1)

-- Problem 2: Find the value of the non-zero constant c
theorem find_c : 
  is_arithmetic_sequence (λ n, b_n (-5) n) :=
sorry

end general_term_find_c_l637_637431


namespace difference_in_sword_length_l637_637007

variables (c_length : ℕ) (j_length : ℕ) (jn_length : ℕ)

def christopher_sword_length : ℕ := 15

def jameson_sword_length (c_length : ℕ) : ℕ := 2 * c_length + 3

def june_sword_length (j_length : ℕ) : ℕ := j_length + 5

theorem difference_in_sword_length :
  let c_length := christopher_sword_length in
  let j_length := jameson_sword_length c_length in
  let jn_length := june_sword_length j_length in
  jn_length = c_length + 23 :=
by
  sorry

end difference_in_sword_length_l637_637007


namespace GP_length_l637_637513

theorem GP_length (X Y Z G P Q : Type) 
  (XY XZ YZ : ℝ) 
  (hXY : XY = 12) 
  (hXZ : XZ = 9) 
  (hYZ : YZ = 15) 
  (hG_centroid : true)  -- Medians intersect at G (Centroid property)
  (hQ_altitude : true)  -- Q is the foot of the altitude from X to YZ
  (hP_below_G : true)  -- P is the point on YZ directly below G
  : GP = 2.4 := 
sorry

end GP_length_l637_637513


namespace angle_BFE_24_deg_l637_637508

open Real

/-- Given two circles with centers A and B, 
    points A, B, and C are collinear, 
    points D, B, and E are collinear, 
    and ∠DBC = 57°, 
    prove that ∠BFE = 24°.-/
theorem angle_BFE_24_deg (A B C D E : Point) (circle1 : Circle) (circle2 : Circle)
  (h1 : circle1.center = A) 
  (h2 : circle2.center = B) 
  (h3 : collinear ({A, B, C} : Set Point))
  (h4 : collinear ({D, B, E} : Set Point))
  (h5 : ∠ D B C = 57) : 
  ∠ B F E = 24 :=
by
  -- Proof is omitted
  sorry

end angle_BFE_24_deg_l637_637508


namespace tip_percentage_l637_637330

theorem tip_percentage (total_amount : ℝ) (sales_tax_rate : ℝ) (price_before_tax : ℝ) 
  (total_amount = 211.20) (sales_tax_rate = 0.10) (price_before_tax = 160) : 
  (total_amount - price_before_tax * (1 + sales_tax_rate)) / (price_before_tax * (1 + sales_tax_rate)) * 100 = 20 := 
by 
  sorry

end tip_percentage_l637_637330


namespace second_remainder_184_mod_15_l637_637744

theorem second_remainder_184_mod_15 (N : ℕ):
  N % 13 = 2 ∧ N = 184 → N % 15 = 4 :=
by
  intro h
  cases' h with h1 h2
  rw [h2] at h1 ⊢
  sorry

end second_remainder_184_mod_15_l637_637744


namespace ratio_yx_l637_637746

theorem ratio_yx : ∃ (x y : ℝ), (0 < x ∧ 0 < y) ∧ ((x + 2 * y * Complex.I)^3 = (x - 2 * y * Complex.I)^3) → (y / x = Real.sqrt(3) / 2) :=
by {
  sorry
}

end ratio_yx_l637_637746


namespace proper_subsets_count_l637_637073

theorem proper_subsets_count {A B : Set ℕ} (hA : A = {2, 3}) (hB : B = {2, 4, 5}) :
  card (powerset (A ∪ B)) - 1 = 15 :=
by
  sorry

end proper_subsets_count_l637_637073


namespace volume_div_pi_l637_637329

-- Definitions from the conditions
def sectorRadius : ℝ := 18
def sectorAngle : ℝ := 300
def fullCircleAngle : ℝ := 360

-- Arc length of the sector
def arcLength : ℝ := (sectorAngle / fullCircleAngle) * (2 * Real.pi * sectorRadius)

-- Cone base radius
def coneBaseRadius : ℝ := arcLength / (2 * Real.pi)

-- Slant height of the cone
def slantHeight : ℝ := sectorRadius

-- Height of the cone using Pythagorean theorem
def coneHeight : ℝ := Real.sqrt (slantHeight^2 - coneBaseRadius^2)

-- Volume of the cone
def volumeCone : ℝ := (1 / 3) * Real.pi * (coneBaseRadius^2) * coneHeight

theorem volume_div_pi : (volumeCone / Real.pi) = 225 * Real.sqrt 11 := 
  by sorry

end volume_div_pi_l637_637329


namespace solve_inequality_l637_637023

theorem solve_inequality : ∀ x : ℝ, (x ∈ Set.Ioo (-∞ : ℝ) (-7) ∪ Set.Ioo 2 (∞)) ↔ (x^2 + 5 * x - 14 > 0) := 
by
  -- proof goes here
  sorry

end solve_inequality_l637_637023


namespace volume_of_sphere_is_correct_l637_637331

-- Define edge length of the cube
def edge_length : ℝ := 2

-- Define the diameter of the sphere circumscribed around the cube
def diameter_of_sphere : ℝ := edge_length * sqrt 3

-- Define the radius of the sphere
def radius_of_sphere : ℝ := diameter_of_sphere / 2

-- Define the volume of the sphere
def volume_of_sphere : ℝ := (4 / 3) * Real.pi * (radius_of_sphere ^ 3)

-- Prove that the volume of the sphere is 4 * sqrt 3 * pi
theorem volume_of_sphere_is_correct : volume_of_sphere = 4 * sqrt 3 * Real.pi := by
  sorry

end volume_of_sphere_is_correct_l637_637331


namespace evaluate_product_eq_l637_637732

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product_eq : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 885735 := 
sorry

end evaluate_product_eq_l637_637732


namespace find_f_at_9_over_2_l637_637182

variable (f : ℝ → ℝ)

-- Domain of f is ℝ
axiom domain_f : ∀ x : ℝ, f x = f x

-- f(x+1) is an odd function
axiom odd_f : ∀ x : ℝ, f (x + 1) = -f (-(x - 1))

-- f(x+2) is an even function
axiom even_f : ∀ x : ℝ, f (x + 2) = f (-(x - 2))

-- When x is in [1,2], f(x) = ax^2 + b
variables (a b : ℝ)
axiom on_interval : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b

-- f(0) + f(3) = 6
axiom sum_f : f 0 + f 3 = 6 

theorem find_f_at_9_over_2 : f (9/2) = 5/2 := 
by sorry

end find_f_at_9_over_2_l637_637182


namespace two_leq_one_add_one_div_n_pow_n_lt_three_l637_637905

theorem two_leq_one_add_one_div_n_pow_n_lt_three :
  ∀ (n : ℕ), 2 ≤ (1 + (1 : ℝ) / n) ^ n ∧ (1 + (1 : ℝ) / n) ^ n < 3 := 
by 
  sorry

end two_leq_one_add_one_div_n_pow_n_lt_three_l637_637905


namespace Emir_needs_more_money_l637_637028

theorem Emir_needs_more_money
  (cost_dictionary : ℝ)
  (cost_dinosaur_book : ℝ)
  (cost_cookbook : ℝ)
  (cost_science_kit : ℝ)
  (cost_colored_pencils : ℝ)
  (saved_amount : ℝ)
  (total_cost : ℝ := cost_dictionary + cost_dinosaur_book + cost_cookbook + cost_science_kit + cost_colored_pencils)
  (more_money_needed : ℝ := total_cost - saved_amount) :
  cost_dictionary = 5.50 →
  cost_dinosaur_book = 11.25 →
  cost_cookbook = 5.75 →
  cost_science_kit = 8.40 →
  cost_colored_pencils = 3.60 →
  saved_amount = 24.50 →
  more_money_needed = 10.00 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end Emir_needs_more_money_l637_637028


namespace expanded_zeros_count_l637_637988

theorem expanded_zeros_count (n : ℕ) (h : n = 100) : (1000 ^ n) = 10 ^ 300 :=
by
  rw [←pow_mul]
  rw [mul_comm] 
  simp [h, pow_succ']
  sorry

end expanded_zeros_count_l637_637988


namespace perpendicular_tangent_circle_l637_637547

theorem perpendicular_tangent_circle
  (A B C E F P: Point)
  (h1 : Triangle ABC)
  (h2 : Circle (diameter A B))
  (h3 : Circle (diameter A B) ∩ Line (AC) = {E})
  (h4 : Circle (diameter A B) ∩ Line (BC) = {F})
  (h5 : Tangent (Circle (diameter A B)) E (Line PF))
  (h6 : Tangent (Circle (diameter A B)) F (Line PE)) :
  Perpendicular PC AB :=
sorry

end perpendicular_tangent_circle_l637_637547


namespace log_inequality_implication_l637_637781

theorem log_inequality_implication {x : ℝ} : 
(log 7 (2 * x)) < (log 7 (x + 2)) → 0 < x ∧ x < 2 :=
by
  sorry

end log_inequality_implication_l637_637781


namespace increase_in_average_runs_l637_637927

-- Define the initial conditions
def average_runs_20_innings := 32
def total_innings := 21
def runs_next_innings := 137

-- Define the function that needs to be proven
theorem increase_in_average_runs (x : ℕ) :
  let total_runs := average_runs_20_innings * 20
  let new_average := average_runs_20_innings + x
  let new_total_runs := new_average * total_innings
  new_total_runs = total_runs + runs_next_innings → x = 5 :=
by {
  -- Definitions
  let total_runs := average_runs_20_innings * 20,
  let new_average := average_runs_20_innings + x,
  let new_total_runs := new_average * total_innings,
  -- Proof
  have h1 : total_runs = 32 * 20 := rfl,
  have h2 : total_runs = 640 := rfl,
  have h3 : new_total_runs = (32 + x) * 21 := rfl,
  have h4 : new_total_runs = 640 + 137 := by rw [h2, rfl],
  assume h5 : (32 + x) * 21 = 640 + 137,
  have h6 : (32 + x) * 21 = 777 := by rw h5,
  have h7 : 32 * 21 + 21 * x = 777 := by rw ←distrib_right,
  have h8 : 672 + 21 * x = 777 := by rw [mul_comm 32 21, nat.mul_comm 32 21],
  have h9 : 21 * x = 105 := by linarith,
  have h10 : x = 105 / 21 := (nat.div_eq_of_lt h9),
  have h11 : x = 5 := rfl,
  exact h11
}

end increase_in_average_runs_l637_637927


namespace projection_of_a_onto_e1_l637_637080

variables (e1 e2 : Vec3)
variable (a : Vec3)
variable angle : ℝ
variable h1 : has_norm e1 = 1
variable h2 : has_norm e2 = 1
variable h3 : e1.angle_with e2 = Real.pi / 3
variable h4 : a = e2 - e1

theorem projection_of_a_onto_e1 : (a.dot e1) = -1/2 := by
  -- proof goes here
  sorry

end projection_of_a_onto_e1_l637_637080


namespace anns_age_l637_637697

theorem anns_age (a b : ℕ) (h1 : a + b = 54) 
(h2 : b = a - (a - b) + (a - b)): a = 29 :=
sorry

end anns_age_l637_637697


namespace omega_interval_l637_637099

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x - real.pi / 3)

theorem omega_interval (ω : ℝ) (hω : ω > 0) (h_range : ∀ x ∈ set.Icc (0 : ℝ) real.pi, f ω x ∈ set.Icc (-(real.sqrt 3) / 2) 1) :
    ω ∈ set.Icc (5 / 6 : ℝ) (5 / 3 : ℝ) :=
sorry

end omega_interval_l637_637099


namespace sequence_an_form_l637_637267

theorem sequence_an_form (a : ℕ → ℚ) (n : ℕ) (h : ∀ n, (finset.range n).sum (λ i, 3^i * a (i+1)) = n / 2) :
  a n = 1 / (2 * 3^(n-1)) :=
by sorry

end sequence_an_form_l637_637267


namespace reservoir_after_storm_fullness_l637_637692

def reservoir_capacity (original_contents : ℕ) (original_percent_full : ℕ) : ℕ :=
  (original_contents * 100) / original_percent_full

def new_total_amount (original_contents storm_addition : ℕ) : ℕ :=
  original_contents + storm_addition

def percentage_full (new_total capacity : ℕ) : ℕ :=
  (new_total * 100) / capacity

theorem reservoir_after_storm_fullness
  (original_contents : ℕ) (storm_addition : ℕ) (original_percent_full : ℕ) (capacity: ℕ):
  (original_percent_full = 50) →
  (original_contents = 200) →
  (storm_addition = 120) →
  (capacity = reservoir_capacity original_contents original_percent_full) →
  percentage_full (new_total_amount original_contents storm_addition) capacity = 80 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end reservoir_after_storm_fullness_l637_637692


namespace students_play_both_football_and_cricket_l637_637650

theorem students_play_both_football_and_cricket :
  ∀ (total F C N both : ℕ),
  total = 460 →
  F = 325 →
  C = 175 →
  N = 50 →
  total - N = F + C - both →
  both = 90 :=
by
  intros
  sorry

end students_play_both_football_and_cricket_l637_637650


namespace parallelogram_perimeter_eq_26_l637_637491

noncomputable def perimeter_parallelogram {A B C D E F : Type*} [linear_ordered_field A] [metric_space B]
  (AB AC BC : A) (lengths_ABC : ∀ d, (eq d AB ∨ eq d AC ∨ eq d (BC - d)))
  (is_parallel_DE_AC : B → B → Prop)
  (is_parallel_EF_AB : B → B → Prop) :
  ∃ AD DE EF AF : A, AD + DE + EF + AF = AB + AC :=
by
  sorry

theorem parallelogram_perimeter_eq_26 :
  perimeter_parallelogram 13 13 10 (by trivial) (by trivial) (by trivial) :=
by
  sorry

end parallelogram_perimeter_eq_26_l637_637491


namespace problem_part1_problem_part2_l637_637324

-- Definitions of the given conditions
def point_on_parabola (m : ℝ) (P : ℝ × ℝ) : Prop :=
  P.2^2 = m * P.1

def symmetric_lines_intersect_parabola
  (x2 : ℝ) (P A B: ℝ × ℝ) (y1 y2 : ℝ) : Prop :=
  (A.1 = x2 - (P.1 - x2)) ∧ (B.1 = P.1) ∧ (A.2 = y1) ∧ (B.2 = y2) ∧ (A.1 < P.1 < B.1)

-- Definitions from the conditions
def P : ℝ × ℝ := (2, 2)
def parabola : ℝ → ℝ → Prop := λ y x, y^2 = 2 * x
def A : ℝ × ℝ
def B : ℝ × ℝ
def x_1 : ℝ
def x_2 : ℝ
def y_1 : ℝ
def y_2 : ℝ

-- Lean 4 statement for the problem
theorem problem_part1 :
  point_on_parabola 2 P →
  symmetric_lines_intersect_parabola 2 P A B y_1 y_2 →
  x_1 = A.1 → x_2 = B.1 → y_1 = A.2 → y_2 = B.2 →
  (y_2 - y_1) / (x_2 - x_1) = -1 / 2 :=
sorry

theorem problem_part2 :
  point_on_parabola 2 P →
  symmetric_lines_intersect_parabola 2 P A B y_1 y_2 →
  x_1 < 2 ∧ 2 < x_2 →
  (2 * ((x_1 - 2) * ((x_2 - 2) * (3 - (1 / 3)))^2)) = 64 * (real.sqrt 3) / 9 :=
sorry

end problem_part1_problem_part2_l637_637324


namespace lowest_temperature_l637_637207

-- Define the temperatures in the four cities.
def temp_Harbin := -20
def temp_Beijing := -10
def temp_Hangzhou := 0
def temp_Jinhua := 2

-- The proof statement asserting the lowest temperature.
theorem lowest_temperature :
  min temp_Harbin (min temp_Beijing (min temp_Hangzhou temp_Jinhua)) = -20 :=
by
  -- Proof omitted
  sorry

end lowest_temperature_l637_637207


namespace cole_round_trip_time_l637_637009

variable (s₁ s₂ : ℝ) (t₁ : ℝ)

def round_trip_time (s₁ s₂ t₁ : ℝ) : ℝ := 
  let d := s₁ * t₁
  let t₂ := d / s₂
  t₁ + t₂

theorem cole_round_trip_time 
  (h1 : s₁ = 75)
  (h2 : s₂ = 105)
  (h3 : t₁ = 70 / 60) :
  round_trip_time s₁ s₂ t₁ = 2 := 
by
  sorry

end cole_round_trip_time_l637_637009


namespace exists_scalars_and_polynomial_l637_637867

noncomputable theory

open Polynomial

variables {R : Type*} [CommRing R]
variables (P Q R : R[X])

theorem exists_scalars_and_polynomial (P Q R : R[X]) (h : P^4 + Q^4 = R^2) :
  ∃ (p q r : ℝ) (S : R[X]), P = C (p : R) * S ∧ Q = C (q : R) * S ∧ R = C (r : R) * S^2 :=
sorry

end exists_scalars_and_polynomial_l637_637867


namespace patty_weeks_without_chores_correct_l637_637217

noncomputable def patty_weeks_without_chores : ℕ := by
  let cookie_per_chore := 3
  let chores_per_week_per_sibling := 4
  let siblings := 2
  let dollars := 15
  let cookie_pack_size := 24
  let cookie_pack_cost := 3

  let packs := dollars / cookie_pack_cost
  let total_cookies := packs * cookie_pack_size
  let weekly_cookies_needed := chores_per_week_per_sibling * cookie_per_chore * siblings

  exact total_cookies / weekly_cookies_needed

theorem patty_weeks_without_chores_correct : patty_weeks_without_chores = 5 := sorry

end patty_weeks_without_chores_correct_l637_637217


namespace northbound_vehicle_count_l637_637280

theorem northbound_vehicle_count :
  ∀ (southbound_speed northbound_speed : ℝ) (vehicles_passed : ℕ) 
  (time_minutes : ℝ) (section_length : ℝ), 
  southbound_speed = 70 → northbound_speed = 50 → vehicles_passed = 30 → time_minutes = 10
  → section_length = 150
  → (vehicles_passed / ((southbound_speed + northbound_speed) * (time_minutes / 60))) * section_length = 270 :=
by sorry

end northbound_vehicle_count_l637_637280


namespace tom_father_time_saved_correct_l637_637626

def tom_father_jog_time_saved : Prop :=
  let monday_speed := 6
  let tuesday_speed := 5
  let thursday_speed := 4
  let saturday_speed := 5
  let daily_distance := 3
  let hours_to_minutes := 60

  let monday_time := daily_distance / monday_speed
  let tuesday_time := daily_distance / tuesday_speed
  let thursday_time := daily_distance / thursday_speed
  let saturday_time := daily_distance / saturday_speed

  let total_time_original := monday_time + tuesday_time + thursday_time + saturday_time
  let always_5mph_time := 4 * (daily_distance / 5)
  let time_saved := total_time_original - always_5mph_time

  let time_saved_minutes := time_saved * hours_to_minutes

  time_saved_minutes = 3

theorem tom_father_time_saved_correct : tom_father_jog_time_saved := by
  sorry

end tom_father_time_saved_correct_l637_637626


namespace two_leq_one_add_one_div_n_pow_n_lt_three_l637_637904

theorem two_leq_one_add_one_div_n_pow_n_lt_three :
  ∀ (n : ℕ), 2 ≤ (1 + (1 : ℝ) / n) ^ n ∧ (1 + (1 : ℝ) / n) ^ n < 3 := 
by 
  sorry

end two_leq_one_add_one_div_n_pow_n_lt_three_l637_637904


namespace smallest_four_digit_number_l637_637042

theorem smallest_four_digit_number (N : ℕ) (a b : ℕ) (h1 : N = 100 * a + b) (h2 : N = (a + b)^2) (h3 : 1000 ≤ N) (h4 : N < 10000) : N = 2025 :=
sorry

end smallest_four_digit_number_l637_637042


namespace no_arrangement_of_1_to_1978_coprime_l637_637514

theorem no_arrangement_of_1_to_1978_coprime :
  ¬ ∃ (a : Fin 1978 → ℕ), 
    (∀ i : Fin 1977, Nat.gcd (a i) (a (i + 1)) = 1) ∧ 
    (∀ i : Fin 1976, Nat.gcd (a i) (a (i + 2)) = 1) ∧ 
    (∀ i : Fin 1978, 1 ≤ a i ∧ a i ≤ 1978 ∧ ∀ j : Fin 1978, (i ≠ j → a i ≠ a j)) :=
sorry

end no_arrangement_of_1_to_1978_coprime_l637_637514


namespace calculate_difference_l637_637864

variable (k a b c : ℕ)

theorem calculate_difference (h_k : k = 81) (h_a : a = 59) (h_b : b = 47) (h_c : c = 63) :
  |k - (a + b + c)| = 88 := by
  sorry

end calculate_difference_l637_637864


namespace find_non_integer_solution_l637_637865

noncomputable def q (x y : ℝ) (b : Fin 10 → ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 +
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3

theorem find_non_integer_solution (b : Fin 10 → ℝ)
  (h0 : q 0 0 b = 0)
  (h1 : q 1 0 b = 0)
  (h2 : q (-1) 0 b = 0)
  (h3 : q 0 1 b = 0)
  (h4 : q 0 (-1) b = 0)
  (h5 : q 1 1 b = 0)
  (h6 : q 1 (-1) b = 0)
  (h7 : q (-1) 1 b = 0)
  (h8 : q (-1) (-1) b = 0) :
  ∃ r s : ℝ, q r s b = 0 ∧ ¬ (∃ n : ℤ, r = n) ∧ ¬ (∃ n : ℤ, s = n) :=
sorry

end find_non_integer_solution_l637_637865


namespace problem_solution_l637_637411

-- Given non-zero numbers x and y such that x = 1 / y,
-- prove that (2x - 1/x) * (y - 1/y) = -2x^2 + y^2.
theorem problem_solution (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x = 1 / y) :
  (2 * x - 1 / x) * (y - 1 / y) = -2 * x^2 + y^2 :=
by
  sorry

end problem_solution_l637_637411


namespace sin_cos_strictly_increasing_l637_637719

noncomputable def strictly_inc_intervals (x : ℝ) (k : ℤ) : Prop :=
  (2 * k * Real.pi - 3 * Real.pi / 4) ≤ x ∧ x ≤ (2 * k * Real.pi + Real.pi / 4)

theorem sin_cos_strictly_increasing (k : ℤ) :
  ∀ x y : ℝ, strictly_inc_intervals x k → strictly_inc_intervals y k → x < y → (sin x + cos x) < (sin y + cos y) :=
sorry

end sin_cos_strictly_increasing_l637_637719


namespace problem_1_problem_2_problem_3_l637_637046

-- Problem (1)
theorem problem_1 (x y : ℝ) (h : (2 * x + y)^2 - 8 * (2 * x + y) - 9 = 0) : 
  (2 * x + y = 9) ∨ (2 * x + y = -1) :=
sorry

-- Problem (2)
theorem problem_2 (x y : ℝ) (h : 2 * x + y - 9 = 8 * real.sqrt (2 * x + y)) : 
  2 * x + y = 81 :=
sorry

-- Problem (3)
theorem problem_3 (x : ℝ) (h : (x^2 + 2 * x)^2 + 4 * (x^2 + 2 * x) - 5 = 0) : 
  x^3 + 3 * x^2 + x = 1 :=
sorry

end problem_1_problem_2_problem_3_l637_637046


namespace Bernard_red_notebooks_l637_637366

def Bernard_notebooks (r b w t left total : ℕ) : Prop :=
  b = 17 → w = 19 → t = 46 → left = 5 → total = left + t → total = r + b + w

theorem Bernard_red_notebooks : ∃ r, Bernard_notebooks r 17 19 46 5 51 ∧ r = 15 :=
begin
  use 15,
  unfold Bernard_notebooks,
  intros,
  simp at *,
  sorry
end

end Bernard_red_notebooks_l637_637366


namespace television_horizontal_length_l637_637555

theorem television_horizontal_length (d h l : ℝ) (h_ratio : l / h = 16 / 9) (h_diag : l^2 + h^2 = d^2) (h_d_eq : d = 36) : l ≈ 33.12 :=
by
  sorry

end television_horizontal_length_l637_637555


namespace talia_father_age_l637_637151

variable (talia_age : ℕ)
variable (mom_age : ℕ)
variable (dad_age : ℕ)

-- Conditions
def condition1 := talia_age + 7 = 20
def condition2 := mom_age = 3 * talia_age
def condition3 := dad_age + 3 = mom_age

-- Theorem to prove
theorem talia_father_age (h1 : condition1) (h2 : condition2) (h3 : condition3) : dad_age = 36 :=
by
  sorry

end talia_father_age_l637_637151


namespace trajectory_of_Q_line_for_triangle_area_l637_637460

-- Conditions
def A : ℝ × ℝ := (sqrt 3, 0)
def C : ℝ × ℝ := (-sqrt 3, 0)
def Q (q : ℝ × ℝ) : Prop := abs ((sqrt 3 - q.1) ^ 2 + q.2 ^ 2) + abs ((-sqrt 3 - q.1) ^ 2 + q.2 ^ 2) = 4
def trajectory_E (q : ℝ × ℝ) : Prop := q.1 ^ 2 / 4 + q.2 ^ 2 = 1

-- First part of the problem: Equation of the trajectory E
theorem trajectory_of_Q :
    ∀ q : ℝ × ℝ, Q q → trajectory_E q :=
sorry

-- Second part of the problem: Equation of the line passing through B(0, -2) when area of triangle OMN = 1
def B : ℝ × ℝ := (0, -2)
def line_through_B (x y : ℝ) (k : ℝ) : Prop := y = k * x - 2

def intersects_E (l : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, trajectory_E (x1, l x1) ∧ trajectory_E (x2, l x2)

def area_triangle (x1 x2 : ℝ) (k : ℝ) : ℝ :=
  abs (4 * sqrt (4 * k ^ 2 - 3)) / (1 + 4 * k ^ 2)

theorem line_for_triangle_area :
    ∀ k : ℝ, (k ^ 2 > 3 / 4) → (area_triangle x1 x2 k = 1) → (line_through_B x1 k) ∨ (line_through_B x2 k) :=
sorry


end trajectory_of_Q_line_for_triangle_area_l637_637460


namespace orthocenter_of_triangle_l637_637222

noncomputable def point := ℝ × ℝ × ℝ -- Simplistic representation of a point in 3D space

-- Definitions of some conditions
variable (P O A B C: point)
variable (planeABC : set point)
variable (isInPlane : point → set point → Prop)
variable (isPerpendicular : point → set point → Prop)

-- Specific conditions
axiom P_outside_plane : ¬ isInPlane P planeABC
axiom PO_perpendicular : isPerpendicular P planeABC
axiom PA_perpendicular_to_PB_and_PC : isPerpendicular P (set.insert A {B, C})

-- Theorem statement
theorem orthocenter_of_triangle :
  ∃ O, isInPlane O planeABC ∧ PO_perpendicular ∧ PA_perpendicular_to_PB_and_PC → 
  O = orthocenter_of_triangle ABC := sorry

end orthocenter_of_triangle_l637_637222


namespace k_9_pow_4_eq_81_l637_637924

theorem k_9_pow_4_eq_81 
  (h k : ℝ → ℝ) 
  (hk1 : ∀ (x : ℝ), x ≥ 1 → h (k x) = x^3) 
  (hk2 : ∀ (x : ℝ), x ≥ 1 → k (h x) = x^4) 
  (k81_eq_9 : k 81 = 9) :
  (k 9)^4 = 81 :=
by
  sorry

end k_9_pow_4_eq_81_l637_637924


namespace no_alpha_exists_l637_637728

theorem no_alpha_exists (α : ℝ) (hα : α > 0) : 
  ¬(∀ x : ℝ, abs (cos x) + abs (cos (α * x)) > sin x + sin (α * x)) :=
sorry

end no_alpha_exists_l637_637728


namespace union_complement_eq_l637_637800

open Set

-- Condition definitions
def U : Set ℝ := univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}

-- Theorem statement (what we want to prove)
theorem union_complement_eq :
  A ∪ compl B = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
by
  sorry

end union_complement_eq_l637_637800


namespace john_light_bulbs_l637_637167

theorem john_light_bulbs (initial total_used remaining_given half_remaining left: ℕ) 
  (h₀ : initial = 40)
  (h₁ : total_used = 16)
  (h₂ : remaining_given = initial - total_used)
  (h₃ : half_remaining = remaining_given / 2)
  (h₄ : left = remaining_given - half_remaining) :
  left = 12 :=
by
  rw [h₀, h₁, h₂, h₃, h₄]
  sorry

end john_light_bulbs_l637_637167


namespace sum_midpoints_x_coordinates_is_15_l637_637275

theorem sum_midpoints_x_coordinates_is_15 :
  ∀ (a b : ℝ), a + 2 * b = 15 → 
  (a + 2 * b) = 15 :=
by
  intros a b h
  sorry

end sum_midpoints_x_coordinates_is_15_l637_637275


namespace rainfall_on_tuesday_l637_637138

noncomputable def R_Tuesday (R_Sunday : ℝ) (D1 : ℝ) : ℝ := 
  R_Sunday + D1

noncomputable def R_Thursday (R_Tuesday : ℝ) (D2 : ℝ) : ℝ :=
  R_Tuesday + D2

noncomputable def total_rainfall (R_Sunday R_Tuesday R_Thursday : ℝ) : ℝ :=
  R_Sunday + R_Tuesday + R_Thursday

theorem rainfall_on_tuesday : R_Tuesday 2 3.75 = 5.75 := 
by 
  sorry -- Proof goes here

end rainfall_on_tuesday_l637_637138


namespace number_of_ordered_pairs_l637_637400

theorem number_of_ordered_pairs : set.count { p : ℤ × ℤ | let m := p.1, n := p.2 in
                                            (m * n ≥ 0) ∧ (m^3 + n^3 + 72 * m * n = 24^3) } = 26 :=
sorry

end number_of_ordered_pairs_l637_637400


namespace julia_snickers_count_l637_637494

variable (S : ℕ)

def snickers_cost := 1.5
def mms_cost := 2 * snickers_cost
def num_mms_packs := 3
def total_given := 20
def change_received := 8

theorem julia_snickers_count (h : total_given - change_received = 1.5 * S + num_mms_packs * mms_cost) : 
  S = 2 :=
sorry

end julia_snickers_count_l637_637494


namespace ellipse_focus_and_axes_l637_637250

theorem ellipse_focus_and_axes (m : ℝ) :
  (∃ a b : ℝ, (a > b) ∧ (mx^2 + y^2 = 1) ∧ (a^2 = 1) ∧ (b^2 = 1/m) ∧ (2 * a = 3 * 2 * b)) → 
  m = 4 / 9 :=
by
  intro h
  rcases h with ⟨a, b, hab, h_eq, ha, hb, ha_b_eq⟩
  sorry

end ellipse_focus_and_axes_l637_637250


namespace stationary_shop_solution_l637_637344

def num_pencils_to_sell (purchase_price sell_price profit_goal : ℝ) (num_purchased: ℕ) : ℕ :=
  (profit_goal + purchase_price * num_purchased) / sell_price

theorem stationary_shop_solution:
  ∀ (num_purchased : ℕ) (purchase_price sell_price profit_goal : ℝ),
  num_purchased = 2000 →
  purchase_price = 0.15 →
  sell_price = 0.30 →
  profit_goal = 180 →
  num_pencils_to_sell purchase_price sell_price profit_goal num_purchased = 1600 :=
by
  intros num_purchased purchase_price sell_price profit_goal h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc num_pencils_to_sell 0.15 0.30 180 2000 = _ : sorry

end stationary_shop_solution_l637_637344


namespace Archer_catch_total_fish_l637_637356

noncomputable def ArcherFishProblem : ℕ :=
  let firstRound := 8
  let secondRound := firstRound + 12
  let thirdRound := secondRound + (secondRound * 60 / 100)
  firstRound + secondRound + thirdRound

theorem Archer_catch_total_fish : ArcherFishProblem = 60 := by
  sorry

end Archer_catch_total_fish_l637_637356


namespace probability_gx_geq_sqrt3_l637_637792

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := sqrt 3 * sin (ω * x) + cos (ω * x)
noncomputable def g (x : ℝ) : ℝ := f (x + π / 6) 2

theorem probability_gx_geq_sqrt3 :
  (0 : ℝ) < π →
  (λ x, 2 * cos (2 * x) ≥ sqrt 3) →
  MeasureTheory.MeasureSpace.volume.measure_of
    {x | 0 ≤ x ∧ x ≤ π ∧ g x ≥ sqrt 3} = π / 6 :=
sorry

end probability_gx_geq_sqrt3_l637_637792


namespace price_of_75_cans_l637_637940

noncomputable def regular_price_per_can : ℝ := 0.15
noncomputable def discount_rate : ℝ := 0.10
noncomputable def case_size : ℕ := 24
noncomputable def total_cans : ℕ := 75

theorem price_of_75_cans (regular_price_per_can discount_rate : ℝ) (case_size total_cans : ℕ) : 
  (regular_price_per_can = 0.15) →
  (discount_rate = 0.10) →
  (case_size = 24) →
  (total_cans = 75) →
  (let discount_per_can := discount_rate * regular_price_per_can,
       discounted_price_per_can := regular_price_per_can - discount_per_can,
       num_cases := total_cans / case_size,
       remainder_cans := total_cans % case_size,
       price_for_cases := (num_cases * case_size) * discounted_price_per_can,
       price_for_remaining_cans := remainder_cans * regular_price_per_can,
       total_price := price_for_cases + price_for_remaining_cans in
   total_price = 10.17) :=
by
  intros h1 h2 h3 h4
  sorry

end price_of_75_cans_l637_637940


namespace correct_propositions_l637_637791

def proposition_1 (l1 l2 : Line) : Prop := 
  ¬ (∃ P, Point_on_line P l1 ∧ Point_on_line P l2) → Parallel l1 l2

def proposition_2 (l1 l2 : Line) : Prop := 
  Perpendicular l1 l2 → (∃ P, Point_on_line P l1 ∧ Point_on_line P l2)

def proposition_3 (l1 l2 : Line) : Prop := 
  ¬ Parallel l1 l2 ∧ ¬ (∃ P, Point_on_line P l1 ∧ Point_on_line P l2) → Skew l1 l2

def proposition_4 (l1 l2 : Line) : Prop := 
  ¬ (∃ π : Plane, Lies_in l1 π ∧ Lies_in l2 π) → Skew l1 l2

theorem correct_propositions :
  ∃ n : ℕ, n = 2 ∧ -- the correct number of propositions
  (proposition_3 l1 l2 ∧ proposition_4 l1 l2) ∧ 
  ¬ proposition_1 l1 l2 ∧ 
  ¬ proposition_2 l1 l2 
:= sorry

end correct_propositions_l637_637791


namespace sum_of_numbers_l637_637653

theorem sum_of_numbers (x : ℝ) 
  (h_ratio : ∃ x, (2 * x) / x = 2 ∧ (3 * x) / x = 3)
  (h_squares : x^2 + (2 * x)^2 + (3 * x)^2 = 2744) :
  x + 2 * x + 3 * x = 84 :=
by
  sorry

end sum_of_numbers_l637_637653


namespace min_ratio_of_max_min_triangle_areas_l637_637656

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem min_ratio_of_max_min_triangle_areas :
  ∀ (S : Finset (EuclideanSpace ℝ (Fin 2))),
    S.card = 5 →
    (∀ (A B C : EuclideanSpace ℝ (Fin 2)), A ≠ B → A ≠ C → B ≠ C → A ∈ S → B ∈ S → C ∈ S → ¬Collinear (Set.univ : Set (line[ ℝ])) (Set.range ![[A, B, C]])) →
    ∃ (τ : ℝ),
    τ = golden_ratio ∧
    (∀ (m M : ℝ), m = (Finset.powersetLen 3 S).inf' (by simp) (λ T, area (T : Finset (EuclideanSpace ℝ (Fin 2))).to_list.to_triple)) → 
    M = (Finset.powersetLen 3 S).sup' (by simp) (λ T, area (T : Finset (EuclideanSpace ℝ (Fin 2))).to_list.to_triple) →
    (∀ (m M : ℝ), (M / m = τ)) :=
by
  intros S hS_card hS_condition
  use golden_ratio
  split
  · refl
  sorry

end min_ratio_of_max_min_triangle_areas_l637_637656


namespace reflection_image_l637_637252

theorem reflection_image (m b : ℝ) 
  (h1 : ∀ x y : ℝ, (x, y) = (0, 1) → (4, 5) = (2 * ((x + (m * y - y + b))/ (1 + m^2)) - x, 2 * ((y + (m * x - x + b)) / (1 + m^2)) - y))
  : m + b = 4 :=
sorry

end reflection_image_l637_637252


namespace patty_weeks_without_chores_correct_l637_637215

noncomputable def patty_weeks_without_chores : ℕ := by
  let cookie_per_chore := 3
  let chores_per_week_per_sibling := 4
  let siblings := 2
  let dollars := 15
  let cookie_pack_size := 24
  let cookie_pack_cost := 3

  let packs := dollars / cookie_pack_cost
  let total_cookies := packs * cookie_pack_size
  let weekly_cookies_needed := chores_per_week_per_sibling * cookie_per_chore * siblings

  exact total_cookies / weekly_cookies_needed

theorem patty_weeks_without_chores_correct : patty_weeks_without_chores = 5 := sorry

end patty_weeks_without_chores_correct_l637_637215


namespace number_of_solutions_l637_637721

-- Define the equation
def equation (x : ℝ) : Prop := (3 * x^2 - 15 * x) / (x^2 - 7 * x + 10) = x - 4

-- State the problem with conditions and conclusion
theorem number_of_solutions : (∀ x : ℝ, x ≠ 2 ∧ x ≠ 5 → equation x) ↔ (∃ x1 x2 : ℝ, x1 ≠ 2 ∧ x1 ≠ 5 ∧ x2 ≠ 2 ∧ x2 ≠ 5 ∧ equation x1 ∧ equation x2) :=
by
  sorry

end number_of_solutions_l637_637721


namespace minimum_cuts_for_27_pieces_l637_637203

theorem minimum_cuts_for_27_pieces (n : ℕ) (h : n^3 = 27) : min_cuts n = 6 :=
by
  sorry

def min_cuts (n : ℕ) : ℕ :=
  if n = 1 then 0
  else 3 * (n - 1)

end minimum_cuts_for_27_pieces_l637_637203


namespace machine_production_in_10_seconds_l637_637468

/-- Define the rate of production -/
def production_rate_per_minute : ℕ := 150

/-- Define the ratio of 10 seconds to one minute -/
def time_ratio : ℚ := 10 / 60

/-- Define the expected production in 10 seconds -/
def production_in_10_seconds : ℕ := (time_ratio * production_rate_per_minute).toNat

/-- Proof statement -/
theorem machine_production_in_10_seconds : production_in_10_seconds = 25 :=
by {
  have h_ratio : time_ratio = 1 / 6 := by norm_num,
  rw [h_ratio],
  have h_production : 150 * (1 / 6) = (150 / 6 : ℚ) := by norm_num,
  rw [h_production],
  exact by norm_num
}

end machine_production_in_10_seconds_l637_637468


namespace constant_term_expansion_l637_637243

noncomputable def binomial_expansion_constant_term : ℕ → ℕ → ℕ :=
  λ n k, Nat.choose n k * 2^k

theorem constant_term_expansion : binomial_expansion_constant_term 10 2 = 180 := by
  -- Given conditions
  have term_formula : ∀ n r, T (r+1) := λ n r => Nat.choose n r * 2^r * x^(5 - 5 * r / 2)
  have condition : 5 - 5 * 2 / 2 = 0 := by simp
  -- Goal
  sorry

end constant_term_expansion_l637_637243


namespace line_through_midpoint_bisects_chord_eqn_l637_637090

theorem line_through_midpoint_bisects_chord_eqn :
  ∀ (x y : ℝ), (x^2 - 4*y^2 = 4) ∧ (∃ x1 y1 x2 y2 : ℝ, 
    (x1^2 - 4 * y1^2 = 4) ∧ (x2^2 - 4 * y2^2 = 4) ∧ 
    (x1 + x2) / 2 = 3 ∧ (y1 + y2) / 2 = -1) → 
    3 * x + 4 * y - 5 = 0 :=
by
  intros x y h
  sorry

end line_through_midpoint_bisects_chord_eqn_l637_637090


namespace plane_1_intercept_form_plane_2_properties_plane_3_properties_plane_4_properties_l637_637015

variables {x y z : ℝ}

def plane_1 (x y z : ℝ) : Prop := 3 * x + 2 * y + 4 * z - 8 = 0
def plane_2 (x y : ℝ) : Prop := x - y = 0
def plane_3 (x y : ℝ) : Prop := 2 * x + 3 * y - 6 = 0
def plane_4 (z : ℝ) : Prop := z - 2 = 0

theorem plane_1_intercept_form :
  (∃ a b c, ∀ x y z, plane_1 x y z ↔ x / a + y / b + z / c = 1) ∧
  (∃ a b c, (a = 8 / 3 ∧ b = 4 ∧ c = 2)) :=
sorry

theorem plane_2_properties :
  (∀ z, plane_2 0 0) ∧
  (∀ x y, plane_2 x y ↔ x = y) :=
sorry

theorem plane_3_properties :
  (∀ z, (∃ x y, plane_3 x y)) ∧
  (∀ x y, plane_3 x y) ∧
  (∃ N : ℝ × ℝ × ℝ, N = (2, 3, 0)) :=
sorry

theorem plane_4_properties :
  (∃ N : ℝ × ℝ × ℝ, N = (0, 0, 1)) ∧
  (∃ p : ℝ × ℝ × ℝ, p = (0, 0, 2)) :=
sorry

end plane_1_intercept_form_plane_2_properties_plane_3_properties_plane_4_properties_l637_637015


namespace prove_identical_numbers_l637_637218

variable {x y : ℝ}

theorem prove_identical_numbers (hx : x ≠ 0) (hy : y ≠ 0)
    (h1 : x + (1 / y^2) = y + (1 / x^2))
    (h2 : y^2 + (1 / x) = x^2 + (1 / y)) : x = y :=
by 
  sorry

end prove_identical_numbers_l637_637218


namespace hcf_of_numbers_l637_637133

theorem hcf_of_numbers (x y : ℕ) (hcf lcm : ℕ) 
    (h_sum : x + y = 45) 
    (h_lcm : lcm = 100)
    (h_reciprocal_sum : 1 / (x : ℝ) + 1 / (y : ℝ) = 0.3433333333333333) :
    hcf = 1 :=
by
  sorry

end hcf_of_numbers_l637_637133


namespace log_a_b_odd_probability_l637_637288

theorem log_a_b_odd_probability :
  let S := {x | ∃ k: ℕ, 1 ≤ k ∧ k ≤ 30 ∧ x = 3^k}
  let pairs := finset.powersetLen 2 S
  let odd_log_pairs := pairs.filter (λ p, (log (p.1 ↔ p.2)) ∈ ((λ n: ℕ, odd n) ) )
  let total_pairs := pairs.card
  let valid_pairs := odd_log_pairs.card
  ∀ p ∈ pairs, (valid_pairs : ℚ) / total_pairs = (29 / 87)
  :=
sorry

end log_a_b_odd_probability_l637_637288


namespace sum_of_roots_cubic_eq_decimal_l637_637024

noncomputable def sum_of_roots_decimal (a b c : ℚ) : ℚ :=
-((b / a) : ℚ)

theorem sum_of_roots_cubic_eq_decimal :
  let roots := [0, (-7 + Real.sqrt (7^2 - 4*3*(-6))) / (2 * 3), (-7 - Real.sqrt (7^2 - 4*3*(-6))) / (2 * 3)]
  in (roots.sum : ℚ) ≈ -2.33 :=
by {
  sorry
}

end sum_of_roots_cubic_eq_decimal_l637_637024


namespace find_number_l637_637609

def valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ((n % 10 = 6 ∧ ¬ (n % 7 = 0)) ∨ (¬ (n % 10 = 6) ∧ n % 7 = 0)) ∧
  ((n > 26 ∧ ¬ (n % 10 = 8)) ∨ (¬ (n > 26) ∧ n % 10 = 8)) ∧
  ((n % 13 = 0 ∧ ¬ (n < 27)) ∨ (¬ (n % 13 = 0) ∧ n < 27))

theorem find_number : ∃ n : ℕ, valid_number n ∧ n = 91 := by
  sorry

end find_number_l637_637609


namespace reduction_amount_correct_l637_637262

def original_price : ℝ := 500
def reduction_percentage : ℝ := 70 / 100

theorem reduction_amount_correct :
  original_price * reduction_percentage = 350 :=
by
  -- Skipping the proof for now
  sorry

end reduction_amount_correct_l637_637262


namespace seating_arrangements_equal_N_factorial_l637_637584

-- Define the problem conditions
def chair_count : ℕ := 12
def martians_count : ℕ := 4
def venusians_count : ℕ := 4
def earthlings_count : ℕ := 4

-- Condition: A Martian sits in chair 1
def martian_in_chair_one (seating : Fin chair_count -> String) : Prop :=
  seating 0 = "Martian"

-- Condition: An Earthling sits in chair 12
def earthling_in_chair_twelve (seating : Fin chair_count -> String) : Prop :=
  seating (Fin.mk 11 (by decide)) = "Earthling"

-- Condition: No Earthling sits immediately to the left of a Martian
def no_earthling_left_of_martian (seating : Fin chair_count -> String) : Prop :=
  ∀ i, i < chair_count - 1 -> seating (Fin.mk (i + 1) (by decide)) = "Martian" -> 
        seating (Fin.mk i (by decide)) ≠ "Earthling"

-- Condition: No Martian sits immediately to the left of a Venusian
def no_martian_left_of_venusian (seating : Fin chair_count -> String) : Prop :=
  ∀ i, i < chair_count - 1 -> seating (Fin.mk (i + 1) (by decide)) = "Venusian" -> 
        seating (Fin.mk i (by decide)) ≠ "Martian"

-- Condition: No Venusian sits immediately to the left of an Earthling
def no_venusian_left_of_earthling (seating : Fin chair_count -> String) : Prop :=
  ∀ i, i < chair_count - 1 -> seating (Fin.mk (i + 1) (by decide)) = "Earthling" -> 
        seating (Fin.mk i (by decide)) ≠ "Venusian"

-- Define the valid seating arrangement given all conditions
def valid_seating (seating : Fin chair_count -> String) : Prop :=
  martian_in_chair_one seating ∧
  earthling_in_chair_twelve seating ∧
  no_earthling_left_of_martian seating ∧
  no_martian_left_of_venusian seating ∧
  no_venusian_left_of_earthling seating

-- Define the statement of the number of such valid seating arrangements
def number_of_valid_seatings : ℕ := 27 * (factorial martians_count) * (factorial venusians_count) * (factorial earthlings_count)

theorem seating_arrangements_equal_N_factorial :
  ∃ (N : ℕ), (number_of_valid_seatings = N * (factorial martians_count) * (factorial venusians_count) * (factorial earthlings_count)) ∧ N = 27 :=
by sorry

end seating_arrangements_equal_N_factorial_l637_637584


namespace min_value_f_l637_637235

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - Real.log x

-- Define the domain for x
def domain (x : ℝ) : Prop := 0 < x ∧ x < Real.exp 1

-- State the theorem to find the minimum value within the given domain
theorem min_value_f : ∀ x : ℝ, domain x → f x ≥ 1 + Real.log 2 ∧ ∃ x_min : ℝ, domain x_min ∧ f x_min = 1 + Real.log 2 :=
by
  sorry

end min_value_f_l637_637235


namespace locus_of_intersection_point_l637_637847

open Real

theorem locus_of_intersection_point :
  ∀ (x : ℝ), -1 < x ∧ x < 1 →
  ∃ y : ℝ, y = -√((x^2 + 3) / 3) :=
by
  intros x hx
  use (-√((x^2 + 3) / 3))
  rfl

end locus_of_intersection_point_l637_637847


namespace nasim_card_purchases_l637_637200

theorem nasim_card_purchases 
  (n : ℕ) : 
  n ∈ {24, 25, 26, 27, 28, 29} →
  ∃ (x y : ℕ), n = 5 * x + 8 * y ↔ n ≠ 27 :=
by
  intro h
  cases h with
  | intro h24 => use 0, 3; simp [h24]
  | intro h25 => use 5, 0; simp [h25]
  | intro h26 => use 2, 2; simp [h26]
  | intro h28 => use 4, 1; simp [h28]
  | intro h29 => use 1, 3; simp [h29]
  | intro h27 => contradiction
  sorry

end nasim_card_purchases_l637_637200


namespace range_of_x_for_odd_function_l637_637594

theorem range_of_x_for_odd_function (f : ℝ → ℝ) (domain : Set ℝ)
  (h_odd : ∀ x ∈ domain, f (-x) = -f x)
  (h_mono : ∀ x y, 0 < x -> x < y -> f x < f y)
  (h_f3 : f 3 = 0)
  (h_ineq : ∀ x, x ∈ domain -> x * (f x - f (-x)) < 0) : 
  ∀ x, x * f x < 0 ↔ -3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3 :=
by sorry

end range_of_x_for_odd_function_l637_637594


namespace BP_length_correct_l637_637835

noncomputable def BP_length (A B C P : Point) (AB BC : ℝ) 
  (angle_ABC angle_APB angle_BPC : ℝ) : ℝ :=
  if (AB = 2 ∧ BC = Real.sqrt 3 ∧ angle_ABC = 150 ∧ angle_APB = 45 ∧ angle_BPC = 120) 
  then 2 / Real.sqrt 5
  else 0

theorem BP_length_correct (A B C P : Point) 
  (h_AB : dist A B = 2) (h_BC : dist B C = Real.sqrt 3) 
  (h_angle_ABC : angle A B C = 150) (h_angle_APB : angle A P B = 45)
  (h_angle_BPC : angle B P C = 120) :
  dist B P = BP_length A B C P 2 (Real.sqrt 3) 150 45 120 := 
by
  sorry

end BP_length_correct_l637_637835


namespace range_of_a_l637_637456

theorem range_of_a (a : ℝ) : 
  (A ⊆ B) → a ≤ 2 :=
by
  let A := {x : ℝ | x < a}
  let B := {x : ℝ | 2^x < 4}
  intro h
  sorry

end range_of_a_l637_637456


namespace tangent_length_to_circumcircle_l637_637880

theorem tangent_length_to_circumcircle (
  (A B C I E F D : Type)
  [IsTriangle ABC]
  (h1 : length AB = 20)
  (h2 : length BC = 10)
  (h3 : length CA = 15)
  (h4 : IsIncenter ABC I)
  (h5 : IsAngleBisector BI AC E)
  (h6 : IsAngleBisector CI AB F)
  (h7 : IsIntersectionPoint (circumcircle B I F) (circumcircle C I E) D)
  (h8 : D ≠ I)
) : length_of_tangent A (circumcircle DEF) = 2 * sqrt 30 := 
by 
  sorry

end tangent_length_to_circumcircle_l637_637880


namespace cos_double_angle_l637_637750

theorem cos_double_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : sin α + cos α = 1/2) : cos (2 * α) = - (Real.sqrt 7) / 4 :=
by
  sorry

end cos_double_angle_l637_637750


namespace trapezium_area_remaining_l637_637350

theorem trapezium_area_remaining (e f g h : ℝ) (he : 0 < e) (hf : 0 < f) (hg : 0 < g) (hh : 0 < h) :
    let A_trapezium := (e + f) * (g + h) in
    let A_triangle := (1 / 2) * h * h in
    let A_remaining := A_trapezium - A_triangle in
    A_remaining = (e + f) * (g + h) - (1 / 2) * h * h :=
by
  sorry

end trapezium_area_remaining_l637_637350


namespace intersection_of_M_and_N_l637_637457

def M : set ℝ := {x : ℝ | -1 < x ∧ x < 3 }
def N : set ℝ := {x : ℝ | -2 < x ∧ x < 1 }

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} :=
by
  sorry

end intersection_of_M_and_N_l637_637457


namespace quadrilateral_area_l637_637000

noncomputable def geoboard_area (A B C D : (ℕ × ℕ)) : ℚ :=
  let x1 := A.1; let y1 := A.2
  let x2 := B.1; let y2 := B.2
  let x3 := C.1; let y3 := C.2
  let x4 := D.1; let y4 := D.2
  (1/2 : ℚ) * abs ((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) 
  - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))

theorem quadrilateral_area :
  geoboard_area (1, 1) (1, 6) (5, 5) (6, 1) = 20 := sorry

end quadrilateral_area_l637_637000


namespace column_removal_preserves_distinct_rows_l637_637071

theorem column_removal_preserves_distinct_rows 
  {α : Type*} {n : ℕ} (h : n ≥ 2) 
  (M : Matrix (Fin n) (Fin n) α) 
  (distinct_rows : ∀ i j : Fin n, i ≠ j → M i ≠ M j) :
  ∃ c : Fin n, ∀ i j : Fin (n - 1), i ≠ j → 
  fun r : Fin n => {x // x ≠ c} → α (Matrix.vecCons (M r) (fun _ => ⊥)) i ≠
  fun r : Fin n => {x // x ≠ c} → α (Matrix.vecCons (M r) (fun _ => ⊥)) j :=
sorry

end column_removal_preserves_distinct_rows_l637_637071


namespace harmonic_series_inequality_l637_637294

theorem harmonic_series_inequality (n : ℕ) (h : n ≥ 2) : 
  ∑ i in finset.range (n - 1), (1 / (i + 2 : ℝ)) > (n - 2) / 2 := 
sorry

end harmonic_series_inequality_l637_637294


namespace theta_plus_2phi_l637_637770

theorem theta_plus_2phi (θ φ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (hφ : 0 < φ ∧ φ < π / 2)
  (h_tan_θ : Real.tan θ = 1 / 7) (h_sin_φ : Real.sin φ = 1 / Real.sqrt 10) :
  θ + 2 * φ = π / 4 := 
sorry

end theta_plus_2phi_l637_637770


namespace brenda_age_l637_637691

theorem brenda_age (A B J : ℝ)
  (h1 : A = 4 * B)
  (h2 : J = B + 8)
  (h3 : A = J + 2) :
  B = 10 / 3 :=
by
  sorry

end brenda_age_l637_637691


namespace sweet_cookies_not_determined_l637_637896

variable (initial_salty initial_sweet eaten_salty remaining_salty : ℕ)
variable (initial_salty = 26) (initial_sweet = 17) (eaten_salty = 9) (remaining_salty = 17)

theorem sweet_cookies_not_determined :
    ∀ (eaten_sweet : ℕ), initial_sweet - eaten_sweet ≠ initial_sweet - eaten_sweet :=
by
    intro eaten_sweet
    sorry

end sweet_cookies_not_determined_l637_637896


namespace least_positive_reducible_fraction_l637_637038

-- Definitions and conditions
def fraction (n : ℕ) : ℚ := (n - 20) / (7 * n + 2)

def is_reducible (n : ℕ) : Prop := 
  Nat.gcd (n - 20) (7 * n + 2) > 1

-- Witness that 22 is the least positive integer for which the given fraction is reducible
theorem least_positive_reducible_fraction : 
  ∃ n : ℕ, n > 0 ∧ is_reducible n ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬ is_reducible m :=
by
  existsi 22
  split
  { -- Shows that 22 is positive
    norm_num }
  split
  { -- Shows that the fraction is reducible
    sorry}
  { -- Shows that no smaller integer yields a reducible fraction
    sorry}

end least_positive_reducible_fraction_l637_637038


namespace smallest_value_of_n_l637_637705

theorem smallest_value_of_n :
  ∃ o y m n : ℕ, 10 * o = 16 * y ∧ 16 * y = 18 * m ∧ 18 * m = 18 * n ∧ n = 40 := 
sorry

end smallest_value_of_n_l637_637705


namespace sum_of_perpendiculars_eq_altitude_l637_637710

variables {A B C P A' B' C' : Type*}
variables (AB AC BC PA' PB' PC' h : ℝ)

-- Conditions
def is_isosceles_triangle (AB AC BC : ℝ) : Prop :=
  AB = AC

def point_inside_triangle (P A B C : Type*) : Prop :=
  true -- Assume point P is inside the triangle

def is_perpendiculars_dropped (PA' PB' PC' : ℝ) : Prop :=
  true -- Assume PA', PB', PC' are the lengths of the perpendiculars from P to the sides BC, CA, AB

def base_of_triangle (BC : ℝ) : Prop :=
  true -- Assume BC is the base of triangle

-- Theorem statement
theorem sum_of_perpendiculars_eq_altitude
  (h : ℝ) (AB AC BC PA' PB' PC' : ℝ)
  (isosceles : is_isosceles_triangle AB AC BC)
  (point_inside_triangle' : point_inside_triangle P A B C)
  (perpendiculars_dropped : is_perpendiculars_dropped PA' PB' PC')
  (base_of_triangle' : base_of_triangle BC) : 
  PA' + PB' + PC' = h := 
sorry

end sum_of_perpendiculars_eq_altitude_l637_637710


namespace david_reading_time_l637_637016

theorem david_reading_time :
  ∀ (total_time math_time spelling_time history_time science_time reading_time : ℕ),
    total_time = 120 →
    math_time = 25 →
    spelling_time = 30 →
    history_time = 20 →
    science_time = 15 →
    reading_time = total_time - (math_time + spelling_time + history_time + science_time) →
    reading_time = 30 :=
by
  intros total_time math_time spelling_time history_time science_time reading_time
  assume h_total_time h_math_time h_spelling_time h_history_time h_science_time h_reading_time
  sorry -- Proof to be completed later.

end david_reading_time_l637_637016


namespace system_of_equations_solution_l637_637742

theorem system_of_equations_solution :
  ∃ x y : ℚ, x = 2 * y ∧ 2 * x - y = 5 ∧ x = 10 / 3 ∧ y = 5 / 3 :=
by
  sorry

end system_of_equations_solution_l637_637742


namespace product_of_roots_l637_637377

-- Define the coefficients of the cubic equation
def a : ℝ := 2
def d : ℝ := 12

-- Define the cubic equation
def cubic_eq (x : ℝ) : ℝ := a * x^3 - 3 * x^2 - 8 * x + d

-- Prove the product of the roots is -6 using Vieta's formulas
theorem product_of_roots : -d / a = -6 := by
  sorry

end product_of_roots_l637_637377


namespace area_of_octagon_in_square_l637_637761

theorem area_of_octagon_in_square : 
  let A := (0, 0)
  let B := (6, 0)
  let C := (6, 6)
  let D := (0, 6)
  let E := (3, 0)
  let F := (6, 3)
  let G := (3, 6)
  let H := (0, 3)
  ∃ (octagon_area : ℚ),
    octagon_area = 6 :=
by
  sorry

end area_of_octagon_in_square_l637_637761


namespace problem_solution_l637_637846

-- Given a right triangle DEF, where E is at (0, 0), D is at (5, 0), and F is at (0, 12)
structure Triangle :=
  (D E F : ℝ × ℝ)
  (is_right_angle_DEF : ∠ E D F = 90)
  (DE : real) (DF : real)

noncomputable def hypotenuse_length (T : Triangle) : ℝ :=
  real.sqrt (T.DE^2 + T.DF^2)

noncomputable def median_length (T : Triangle) : ℝ :=
  hypotenuse_length T / 2

noncomputable def altitude_length (T : Triangle) : ℝ :=
  let hypotenuse := hypotenuse_length T in
  ( (1/2) * T.DE * T.DF ) / (1/2 * hypotenuse)

theorem problem_solution : 
  let T : Triangle := { D := (5, 0), E := (0, 0), F := (0, 12), is_right_angle_DEF := sorry, DE := 5, DF := 12 } in
  median_length T = 6.5 ∧ altitude_length T = 4.6 :=
by
  sorry

end problem_solution_l637_637846


namespace maximal_countries_voting_problems_l637_637050

theorem maximal_countries_voting_problems :
  (∃ (S : Finset (Fin 9)) (Y : Finset (Finset (Fin 9))),
    (∀ y ∈ Y, y.card = 3) ∧
    (∀ {a b}, a ∈ Y → b ∈ Y → a ≠ b → a ≠ b) ∧
    (∀ a b c ∈ Y, ∃ x ∈ S, x ∉ a ∪ b ∪ c) ∧
    Y.card = 56) :=
sorry

end maximal_countries_voting_problems_l637_637050


namespace sum_of_solutions_l637_637724

theorem sum_of_solutions :
  let eqn (x : ℝ) := (x^2 - 6*x + 5) ^ (x^2 - 2*x - 3) = 1
  (∑ x in {x : ℝ | eqn x}, x) = 8 :=
sorry

end sum_of_solutions_l637_637724


namespace problem_equiv_l637_637614

theorem problem_equiv :
  ∃ n : ℕ, 
    10 ≤ n ∧ n < 100 ∧ 
    ((n % 10 = 6 ∧ ¬ (n % 7 = 0)) ∨ (n % 10 ≠ 6 ∧ n % 7 = 0)) ∧
    ((n > 26 ∧ ¬ (n % 10 = 8)) ∨ (n ≤ 26 ∧ n % 10 = 8)) ∧
    ((n % 13 = 0 ∧ ¬ (n < 27)) ∨ (n % 13 ≠ 0 ∧ n < 27)) ∧
    n = 91 :=
begin
  sorry
end

end problem_equiv_l637_637614


namespace non_officers_count_l637_637651

theorem non_officers_count 
    (avg_salary_employees : ℝ) 
    (avg_salary_officers : ℝ) 
    (avg_salary_non_officers : ℝ) 
    (num_officers : ℕ) : 
    avg_salary_employees = 120 ∧ avg_salary_officers = 470 ∧ avg_salary_non_officers = 110 ∧ num_officers = 15 → 
    ∃ N : ℕ, N = 525 ∧ 
    (num_officers * avg_salary_officers + N * avg_salary_non_officers) / (num_officers + N) = avg_salary_employees := 
by 
    sorry

end non_officers_count_l637_637651


namespace shortest_flight_distance_l637_637265

variables (R : ℝ) (deg_to_rad : ℝ → ℝ)

noncomputable def distance_between_cities :=
  let radius_at_45N := (real.sqrt 2 / 2) * R in
  let chord_length := radius_at_45N * real.sqrt 2 in
  let angle_AOB := real.pi / 3 in
  angle_AOB * R

theorem shortest_flight_distance (R : ℝ) (deg_to_rad : ℝ → ℝ) : 
  distance_between_cities R deg_to_rad = (real.pi / 3) * R :=
by sorry

end shortest_flight_distance_l637_637265


namespace max_sum_of_three_integers_with_product_24_l637_637601

theorem max_sum_of_three_integers_with_product_24 : ∃ (a b c : ℤ), (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 24 ∧ a + b + c = 15) :=
by
  sorry

end max_sum_of_three_integers_with_product_24_l637_637601


namespace math_proof_problem_l637_637441

-- Definition of the conditions
def pole_is_origin := true
def polar_axis := ∀ (theta : ℝ), theta ≥ 0
def length_unit_same := true

def param_line_l := ∀ t : ℝ, (1 + (1/2) * t, (sqrt 3 / 2) * t)
def polar_curve_C := ∀ θ : ℝ, (1 + sin(θ)^2) * (ρ(θ))^2 = 2

-- Definitions of the intersection and given point
def intersection_points (A B : ℝ × ℝ) := 
  ∃ (t1 t2 : ℝ), (t1 ≠ t2) ∧ 
  param_line_l t1 = A ∧
  param_line_l t2 = B ∧
  ∀ θ, (1 + sin(θ)^2) * (ρ(θ))^2 = 2 → A ∈ polar_curve_C θ ∧ B ∈ polar_curve_C θ

def given_point_P := (1, 0)

-- Main theorem statement
theorem math_proof_problem :
  pole_is_origin ∧
  polar_axis ∧
  length_unit_same ∧
  (∀ θ, polar_curve_C θ) ∧
  (∀ t, param_line_l t) ∧
  (∃ A B : ℝ × ℝ, intersection_points A B) →
  ((∃ x y : ℝ, x ≠ y ∧ sqrt 3 * x - y - sqrt 3 = 0) ∧
  (∃ x y : ℝ, x^2 / 2 + y^2 = 1) ∧
  (abs (1 / (dist given_point_P (1, (sqrt 3) / 2)))^2 + abs (1 / (dist given_point_P (1 + 1 / 2, (sqrt 3) / 2)))^2 = 9 / 2)) :=
by { sorry }

end math_proof_problem_l637_637441


namespace trig_identity_l637_637128

open Real

theorem trig_identity (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 6) (h : sin α ^ 6 + cos α ^ 6 = 7 / 12) : 1998 * cos α = 333 * Real.sqrt 30 :=
sorry

end trig_identity_l637_637128


namespace hexagon_coins_side_10_l637_637814

theorem hexagon_coins_side_10 (n : ℕ) (h2 : n = 2 → 7) (h3 : n = 3 → 19) :
  3 * 10 * (10 - 1) + 1 = 271 :=
by sorry

end hexagon_coins_side_10_l637_637814


namespace triangle_area_l637_637918

theorem triangle_area
  (ABC : Triangle)
  (triangle_AB1C_isosceles_right : IsoscelesRightTriangle ABC A B1 C ∧ B1_is_right_angle)
  (triangle_AC1B_isosceles_right : IsoscelesRightTriangle ABC A C1 B ∧ C1_is_right_angle)
  (M_is_midpoint_B1C1 : MidpointB1C1 M B1 C1)
  (B1C1_length : length B1 C1 = 12)
  (BM_length : length B M = 7)
  (CM_length : length C M = 11) :
  area ABC = 24 * real.sqrt 10 - 24.5 := 
sorry

end triangle_area_l637_637918


namespace final_price_of_bicycle_l637_637666

def original_price : ℝ := 200
def first_discount_rate : ℝ := 0.40
def second_discount_rate : ℝ := 0.25

theorem final_price_of_bicycle :
  let first_sale_price := original_price - (first_discount_rate * original_price)
  let final_sale_price := first_sale_price - (second_discount_rate * first_sale_price)
  final_sale_price = 90 := by
  sorry

end final_price_of_bicycle_l637_637666


namespace complex_expression_value_l637_637545
noncomputable theory

open Complex

theorem complex_expression_value :
  let w := Complex.cos (6 * Real.pi / 11) + Complex.sin (6 * Real.pi / 11) * Complex.I in
  (w / (1 + w^2)) + (w^3 / (1 + w^6)) + (w^4 / (1 + w^8)) = 1 :=
by 
  let w := Complex.cos (6 * Real.pi / 11) + Complex.sin (6 * Real.pi / 11) * Complex.I 
  -- Proof goes here
  sorry

end complex_expression_value_l637_637545


namespace top_field_number_and_total_labelings_l637_637295

open Finset

-- Define the problem
def valid_labeling (fields : ℕ → ℕ) : Prop :=
  injective fields ∧
  (∀ i, fields i ∈ range 1 (9 + 1)) ∧
  (∃ x, ∀ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d → (fields a + fields b + fields c + fields d = x) ∧
       ∀ (a b c : ℕ), a ≠ b ∧ b ≠ c → (fields a + fields b + fields c = x))

-- Prove the number in the top field is always 9 and there are exactly 48 such labellings
theorem top_field_number_and_total_labelings :
  (∀ fields, valid_labeling fields → fields 0 = 9) ∧
  (∃ count : ℕ, count = 48) :=
by
  sorry

end top_field_number_and_total_labelings_l637_637295


namespace problem_I_problem_II_l637_637101

-- Define the function f with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * |x - 1|

-- Define the interval [0,2]
def interval : set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Problem I: Prove the minimum and maximum values when a = 2 on [0,2]
theorem problem_I : (∀ x ∈ interval, f 2 x ≤ 6) ∧ (∃ x ∈ interval, f 2 x = 6) ∧
                    (∀ x ∈ interval, f 2 x ≥ 1) ∧ (∃ x ∈ interval, f 2 x = 1) :=
by sorry

-- Problem II: Prove the range of a for which f(x) is monotonically increasing on [0, +∞)
theorem problem_II : (∀ (a : ℝ), (∀ x1 x2, 0 ≤ x1 → x1 ≤ x2 → 0 ≤ x2 → f a x1 ≤ f a x2) ↔ (-2 ≤ a ∧ a ≤ 0)) :=
by sorry

end problem_I_problem_II_l637_637101


namespace price_increase_to_restore_l637_637027

noncomputable def final_price (original_price : ℝ) : ℝ :=
  let reduced_price1 := original_price * 0.75
  let reduced_price2 := reduced_price1 * 0.70
  let reduced_price3 := reduced_price2 * 0.85
  let final_price := reduced_price3 * 1.10
  final_price

theorem price_increase_to_restore (P : ℝ) (hP : P > 0) :
  let P_f := final_price P
  let increase_needed := P - P_f
  let percentage_increase := (increase_needed / P_f) * 100
  percentage_increase ≈ 103.75 :=
by
  sorry

end price_increase_to_restore_l637_637027


namespace jiwoo_magnets_two_digit_count_l637_637166

def num_magnets : List ℕ := [1, 2, 7]

theorem jiwoo_magnets_two_digit_count : 
  (∀ (x y : ℕ), x ≠ y → x ∈ num_magnets → y ∈ num_magnets → 2 * 3 = 6) := 
by {
  sorry
}

end jiwoo_magnets_two_digit_count_l637_637166


namespace no_three_segments_form_triangle_l637_637857

theorem no_three_segments_form_triangle :
  ∃ (a : Fin 10 → ℕ), ∀ {i j k : Fin 10}, i < j → j < k → a i + a j ≤ a k :=
by
  sorry

end no_three_segments_form_triangle_l637_637857


namespace complement_of_M_in_R_l637_637488

open Set

theorem complement_of_M_in_R :
  let U := ℝ
  let M := {x : ℝ | x^2 - x ≥ 0}
  Mᶜ = {x : ℝ | 0 < x ∧ x < 1} := 
by
  let U := ℝ
  let M := {x : ℝ | x^2 - x ≥ 0}
  show Mᶜ = {x : ℝ | 0 < x ∧ x < 1}, from sorry -- Proof omitted

end complement_of_M_in_R_l637_637488


namespace digit_15_of_sum_reciprocals_l637_637973

/-- 
What is the 15th digit after the decimal point of the sum of the decimal equivalents 
for the fractions 1/9 and 1/11?
-/
theorem digit_15_of_sum_reciprocals :
  let r := (1/9 + 1/11) in
  let d15 := Real.frac (10^(15:ℕ) * r) in
  Int.floor (10 * d15) = 1 :=
by
  let r := (1/9 + 1/11)
  let d15 := Real.frac (10^(15:ℕ) * r)
  have h : Real.toRat d15 = 1 / 10 + d15 - Int.floor d15
  have : Real.frac (10^(15:ℕ) * r) = r
  sorry

end digit_15_of_sum_reciprocals_l637_637973


namespace midpoint_equidistant_from_perpendiculars_l637_637897

-- Define the point in the Euclidean plane
noncomputable section
  open EuclideanGeometry

variable {A B C D E M : Point}

-- Main theorem statement
theorem midpoint_equidistant_from_perpendiculars 
  (h_tri : Triangle A B C)
  (h_perp_D : PerpendicularTo D B (LineThrough A))
  (h_perp_E : PerpendicularTo E C (LineThrough A))
  (h_mid_M : Midpoint M B C) : 
  (distance M D = distance M E) :=
sorry

end midpoint_equidistant_from_perpendiculars_l637_637897


namespace count_valid_p_values_l637_637387

-- Definitions according to (a)
def is_integer_side_length (n : ℕ) : Prop := ∃ m : ℤ, m ≥ 0 ∧ m = n
def is_quadrilateral (A B C D : Type) : Prop := true -- Placeholder definition

def quadrilateral_with_conditions (A B C D : Type) (p : ℕ) :=
  is_quadrilateral A B C D ∧
  is_integer_side_length (3 : ℕ) ∧
  -- Additional conditions based on the problem statement
  right_angle_at D ∧ right_angle_at C ∧
  AB=3 ∧ CD=AD ∧ 
  (∃ (x y : ℕ), x = 3 ∧ AB = 3 ∧ BC = x ∧ CD = AD ∧ p = 6 + 2*y)
  ∧ ∀ (p : ℕ), p < 2000

-- The proof statement (parallels (c))
theorem count_valid_p_values (A B C D : Type) :
  ∃ (y : ℕ), 1 ≤ y ∧ y ≤ 996 ∧ (∀ p, quadrilateral_with_conditions A B C D p) :=
sorry

end count_valid_p_values_l637_637387


namespace pool_water_after_eight_hours_l637_637704

-- Define the conditions
def hour1_fill_rate := 8
def hour2_and_hour3_fill_rate := 10
def hour4_and_hour5_fill_rate := 14
def hour6_fill_rate := 12
def hour7_fill_rate := 12
def hour8_fill_rate := 12
def hour7_leak := -8
def hour8_leak := -5

-- Calculate the water added in each time period
def water_added := hour1_fill_rate +
                   (hour2_and_hour3_fill_rate * 2) +
                   (hour4_and_hour5_fill_rate * 2) +
                   (hour6_fill_rate + hour7_fill_rate + hour8_fill_rate)

-- Calculate the water lost due to leaks
def water_lost := hour7_leak + hour8_leak  -- Note: Leaks are already negative

-- The final calculation: total water added minus total water lost
def final_water := water_added + water_lost

theorem pool_water_after_eight_hours : final_water = 79 :=
by {
  -- proof steps to check equality are omitted here
  sorry
}

end pool_water_after_eight_hours_l637_637704


namespace P_is_necessary_but_not_sufficient_for_Q_l637_637072

def P (x : ℝ) : Prop := |x - 1| < 4
def Q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

theorem P_is_necessary_but_not_sufficient_for_Q :
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬Q x) :=
by
  sorry

end P_is_necessary_but_not_sufficient_for_Q_l637_637072


namespace compare_logs_and_exponentiation_l637_637544

theorem compare_logs_and_exponentiation (a b c: ℝ) 
  (ha: a = log 6 3) 
  (hb: b = log 10 5) 
  (hc: c = 2 ^ 0.1) 
  : a < b ∧ b < c := 
begin
  have ha_pos : 0 < a := sorry,
  have ha_lt_1 : a < 1 := sorry,
  have hb_pos : 0 < b := sorry,
  have hb_lt_1 : b < 1 := sorry,
  have hc_gt_1 : 1 < c := sorry,
  have a_lt_b : a < b := sorry,
  exact ⟨a_lt_b, hb_lt_1.trans_lt hc_gt_1⟩,
end

end compare_logs_and_exponentiation_l637_637544


namespace fourth_power_mod_7_is_0_l637_637634

def fourth_smallest_prime := 7
def square_of_fourth_smallest_prime := fourth_smallest_prime ^ 2
def fourth_power_of_square := square_of_fourth_smallest_prime ^ 4

theorem fourth_power_mod_7_is_0 : 
  (fourth_power_of_square % 7) = 0 :=
by sorry

end fourth_power_mod_7_is_0_l637_637634


namespace pyramid_distance_l637_637268

variable (a : ℝ) (α : ℝ)

theorem pyramid_distance (hα : 0 < α ∧ α < real.pi) :
  ∃ OF, OF = (a / 2) * real.sqrt(2 * real.cos α) :=
by
  sorry

end pyramid_distance_l637_637268


namespace interval_of_monotonic_increase_l637_637020

noncomputable def func : ℝ → ℝ :=
  λ x, (1 / 2) * sin x + (real.sqrt 3 / 2) * cos x

theorem interval_of_monotonic_increase :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 2) → (0 ≤ x ∧ x ≤ π / 6) :=
 by
  sorry

end interval_of_monotonic_increase_l637_637020


namespace solve_for_y_l637_637234

theorem solve_for_y (y : ℝ) (h : log 3 ((4 * y + 16) / (6 * y - 9)) + log 3 ((6 * y - 9) / (2 * y - 5)) = 3) : 
  y = 151 / 50 :=
by 
  sorry

end solve_for_y_l637_637234


namespace max_plus_min_value_of_f_l637_637451

noncomputable def f (x : ℝ) : ℝ :=
  (|x| - Real.sin x + 1) / (|x| + 1)

theorem max_plus_min_value_of_f : 
  let M := RealSup (Set.range f),
      m := RealInf (Set.range f)
  in M + m = 2 :=
by
  sorry

end max_plus_min_value_of_f_l637_637451


namespace distance_between_points_l637_637700

noncomputable def dist (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  dist (-3, 5) (4, -9) = real.sqrt 245 :=
by
  sorry

end distance_between_points_l637_637700


namespace problem_statement_l637_637727

noncomputable def a (i : ℕ) : ℕ → ℝ
| x => ∑ n in Finset.range (x + 1), (λ n, x ^ (3 * n + i) / Real.fact (3 * n + i)) n

theorem problem_statement : 
  let a0 := a 0
  let a1 := a 1
  let a2 := a 2
  (a0 0)^3 + (a1 0)^3 + (a2 0)^3 - 3 * (a0 0) * (a1 0) * (a2 0) = 1 :=
sorry

end problem_statement_l637_637727


namespace orchard_cross_pollination_percentage_l637_637332

noncomputable def cross_pollinated_percentage (F C T : ℕ) : ℝ :=
(C / T) * 100

theorem orchard_cross_pollination_percentage :
  ∃ (F C T : ℕ), 
  F + C = 170 ∧ 
  F = 3 * T / 4 ∧ 
  T = F + 30 ∧ 
  cross_pollinated_percentage F C T = 66.67 :=
by {
  use [90, 80, 120],
  simp [cross_pollinated_percentage],
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  {
    have h : (80 : ℝ) / 120 = 2 / 3,
    norm_num,
    rw h,
    norm_num,
    done,
  },
  done,
}

end orchard_cross_pollination_percentage_l637_637332


namespace gain_per_year_is_correct_l637_637679

noncomputable def compoundInterest (principal : ℕ) (rate : ℚ) (n : ℕ) : ℚ :=
  principal * (1 + rate / 100)^n

noncomputable def gainInTransactionPerYear : ℚ := 
  let principal := 5000
  let firstYearBorrowRate := 4
  let secondYearBorrowRate := 6
  let firstYearLendRate := 5
  let secondYearLendRate := 7
  let timePeriod := 2

  let amountBorrowed := compoundInterest principal firstYearBorrowRate 1 
  let amountBorrowedAfterSecondYear := compoundInterest amountBorrowed secondYearBorrowRate 1

  let amountLent := compoundInterest principal firstYearLendRate 1
  let amountLentAfterSecondYear := compoundInterest amountLent secondYearLendRate 1

  let totalInterestPaid := amountBorrowedAfterSecondYear - principal
  let totalInterestEarned := amountLentAfterSecondYear - principal
  let totalGain := totalInterestEarned - totalInterestPaid

  totalGain / timePeriod

theorem gain_per_year_is_correct : gainInTransactionPerYear = 52.75 := 
  sorry

end gain_per_year_is_correct_l637_637679


namespace max_value_of_f_greater_than_2_pow_2018_l637_637012

-- Definitions of the Fibonacci sequence as given
def fib : ℕ → ℕ
| 0 := 1
| 1 := 2
| (n+2) := fib (n+1) + fib n

-- Given function f
def f (x : ℝ) : ℝ :=
  ∏ i in finset.range 3030, (x - fib i)

-- Statement to prove
theorem max_value_of_f_greater_than_2_pow_2018 :
  ∃ x_0 ∈ Ioo (fib 0) (fib 3030), |f x_0| = finset.univ.sup (λ x, |f x|) ∧ x_0 > 2^2018 :=
sorry

end max_value_of_f_greater_than_2_pow_2018_l637_637012


namespace platform_length_correct_l637_637334

def kmph_to_mps (s : ℕ) : ℕ := s * 1000 / 3600

def total_distance (speed : ℕ) (time : ℕ) : ℕ := speed * time

def platform_length (total_dist train_length : ℚ) : ℚ := total_dist - train_length

theorem platform_length_correct :
  let speed_kmph := 72
  let time_seconds := 26
  let train_length_m := 250.0416
  kmph_to_mps speed_kmph * time_seconds - train_length_m = 269.9584 :=
by
  sorry

end platform_length_correct_l637_637334


namespace profit_percentage_is_1_point_01_l637_637130

noncomputable def profit_percentage (SP : ℝ) (CP : ℝ) : ℝ :=
  ((SP - CP) / CP) * 100

theorem profit_percentage_is_1_point_01 (SP : ℝ) (h : CP = 0.99 * SP) :
  profit_percentage SP CP ≈ 1.01 :=
by
  sorry

end profit_percentage_is_1_point_01_l637_637130


namespace coprime_mul_coprime_l637_637232

variable {α : Type*} [EuclideanDomain α]

theorem coprime_mul_coprime {a b n : α} (h1 : gcd a n = 1) (h2 : gcd b n = 1) : gcd (a * b) n = 1 :=
by
  sorry

end coprime_mul_coprime_l637_637232


namespace cube_division_height_l637_637489

theorem cube_division_height
  (large_cube_edge : ℝ)
  (small_cube_edge : ℝ)
  (num_small_cubes : ℝ)
  (conversion_factor : ℝ)
  (large_cube_edge = 1)
  (small_cube_edge = 1 / 1000)
  (num_small_cubes = (large_cube_edge / small_cube_edge)^3)
  (conversion_factor = 1000000) :
  (num_small_cubes * small_cube_edge) / conversion_factor = 1000 :=
sorry

end cube_division_height_l637_637489


namespace total_eggs_l637_637804

-- Define the number of eggs eaten in each meal
def breakfast_eggs : ℕ := 2
def lunch_eggs : ℕ := 3
def dinner_eggs : ℕ := 1

-- Prove the total number of eggs eaten is 6
theorem total_eggs : breakfast_eggs + lunch_eggs + dinner_eggs = 6 :=
by
  sorry

end total_eggs_l637_637804


namespace angle_between_A1E_and_C1F_l637_637142

def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def vector_sub (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem angle_between_A1E_and_C1F :
  let a := 1 in
  let A : ℝ × ℝ × ℝ := (0, 0, 0)
  let B : ℝ × ℝ × ℝ := (a, 0, 0)
  let B1 : ℝ × ℝ × ℝ := (a, a, 0)
  let A1 : ℝ × ℝ × ℝ := (0, a, 0)
  let C1 : ℝ × ℝ × ℝ := (a, a, a)
  let E := midpoint A B
  let F := midpoint B B1
  let A1E := vector_sub E A1
  let C1F := vector_sub F C1
  Real.arcsin (Real.sqrt 21 / 5) = 
  Real.acos (dot_product A1E C1F / (magnitude A1E * magnitude C1F)) :=
sorry

end angle_between_A1E_and_C1F_l637_637142


namespace product_of_roots_cubic_eq_l637_637378

-- Define a cubic equation 2x^3 - 3x^2 - 8x + 12 = 0
def cubic_eq : Polynomial ℝ := Polynomial.mk [2, -3, -8, 12]

-- Define a function to compute the product of the roots of a cubic equation using Vieta's formulas
def product_of_roots (p : Polynomial ℝ) : ℝ :=
  let a := p.coeff 3
  let d := p.coeff 0
  -d / a

-- The proof statement: Prove that the product of the roots of this specific cubic equation is -6
theorem product_of_roots_cubic_eq : product_of_roots cubic_eq = -6 := 
  sorry

end product_of_roots_cubic_eq_l637_637378


namespace sqrt_sum_l637_637135

theorem sqrt_sum (m n : ℝ) (h1 : m + n = 0) (h2 : m * n = -2023) : m + 2 * m * n + n = -4046 :=
by sorry

end sqrt_sum_l637_637135


namespace PDL_of_set_a_find_x_exists_set_no_repeated_PDL_repeated_PDL_of_qrst_l637_637321

-- Part (a)
theorem PDL_of_set_a : (PDL {3, 6, 13, 21, 32}) = ({3, 7, 8, 10, 11, 15, 18, 19, 26, 29} : Finset ℕ) :=
sorry

-- Part (b)
theorem find_x (x : ℕ) (h1 : x > 16) (h2 : sum (PDL {1, 4, 9, 16, x}) = 112) : x = 23 :=
sorry

-- Part (c)
theorem exists_set_no_repeated_PDL : ∃ (q r s : ℕ), 3 < q ∧ q < r ∧ r < s ∧ s < 14 ∧ PDL {3, q, r, s, 14} = ({3, 5, 7, 14} : Finset ℕ)  :=
sorry

-- Part (d)
theorem repeated_PDL_of_qrst (q r s t : ℕ) (h1 : 3 < q) (h2 : q < r) (h3 : r < s) (h4 : s < t) (h5 : t < 14) : ∃ a, a ∈ PDL {3, q, r, s, t} ∧ multiset.count a (PDL {3, q, r, s, t}) > 1 :=
sorry

end PDL_of_set_a_find_x_exists_set_no_repeated_PDL_repeated_PDL_of_qrst_l637_637321


namespace perpendicular_slope_correct_l637_637639

-- Define the points
def p1 : ℚ × ℚ := (3, -4)
def p2 : ℚ × ℚ := (-2, 5)

-- Define the slope of the line containing these points
def slope (p1 p2 : ℚ × ℚ) : ℚ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the slope of the perpendicular line
def perpendicular_slope (m : ℚ) : ℚ := -1 / m

-- State the proof problem
theorem perpendicular_slope_correct :
  perpendicular_slope (slope p1 p2) = 5 / 9 :=
by
  -- The proof is skipped using sorry
  sorry

end perpendicular_slope_correct_l637_637639


namespace range_of_a_l637_637827

noncomputable def f (a x : ℝ) : ℝ := x - (1/3) * sin(2 * x) + a * sin(x)
noncomputable def f' (a x : ℝ) : ℝ := 1 - (2/3) * cos(2 * x) + a * cos(x)

theorem range_of_a (a : ℝ) (h : ∀ x, f' a x ≥ 0) : a ∈ Icc (-1/3) (1/3) := sorry

end range_of_a_l637_637827


namespace range_f_sum_l637_637723

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x ^ 2)

theorem range_f_sum {a b : ℝ} (h₁ : Set.Ioo a b = (Set.Ioo (0 : ℝ) (3 : ℝ))) :
  a + b = 3 :=
sorry

end range_f_sum_l637_637723


namespace composition_of_symmetries_l637_637227

-- Definitions and proof problem statement
theorem composition_of_symmetries (n : ℕ) (l : fin n → Line) (O : Point) 
  (h1 : ∀ i, l i passes_through O) : 
  if even n then is_rotation (composition l) 
  else is_axial_symmetry (composition l) :=
sorry

end composition_of_symmetries_l637_637227


namespace min_value_at_minus_one_l637_637595

/-- A function definition -/
def f (x : ℝ) : ℝ := - (1/3) * x^3 + (1/2) * x^2 + 2 * x

/-- The goal is to show that the function takes its minimum value at x = -1 -/
theorem min_value_at_minus_one : ∀ {x : ℝ}, f x ≥ f (-1) :=
sorry

end min_value_at_minus_one_l637_637595


namespace volume_of_prism_l637_637987

noncomputable def prismVolume {x y z : ℝ} 
  (h1 : x * y = 20) 
  (h2 : y * z = 12) 
  (h3 : x * z = 8) : ℝ :=
  x * y * z

theorem volume_of_prism (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 12)
  (h3 : x * z = 8) : prismVolume h1 h2 h3 = 8 * Real.sqrt 15 :=
by
  sorry

end volume_of_prism_l637_637987


namespace place_points_on_triangle_l637_637805

variable (A B C D E : Type) [linear_order A B C]
variable (triangle : A → B → C → Prop)

noncomputable def midpoint (a b d : A) : Prop :=
  ∃ (d : A), d = (a + b) / 2

noncomputable def divide_ratio (c a e : A) (r : ℝ) : Prop :=
  ∃ (e : A), r = ∥c - e∥ / ∥c - a∥

theorem place_points_on_triangle (A B C D E : Type) [linear_order A B C]
  (h1 : D ∈ line_segment A B) (h2 : E ∈ line_segment A C)
  (h3 : area (triangle A D E) = area (triangle D E B))
  (h4 : area (triangle D E B) = area (triangle E B C)) :
  midpoint A B D ∧ divide_ratio C A E (1/3) :=
by
  sorry

end place_points_on_triangle_l637_637805


namespace collinear_M_N_H_collinear_H_M_N_O_l637_637319

-- Define the context of the acute-angled triangle and the given points.
variables {A B C H M N O : Type}
variables [triangle A B C]
variables [acute_angle A B C]
variables (is_altitude_intersection : altitudes_intersect A B C H)
variables (angle_A_eq_60 : angle A = 60)
variables (M_is_perpendicular_bisector_intersection : perpendicular_bisector_intersects_side A B H M)
variables (N_is_perpendicular_bisector_intersection : perpendicular_bisector_intersects_side A C H N)
variables (O_is_circumcenter : is_circumcenter A B C O)

-- State the problems in Lean 4
theorem collinear_M_N_H : collinear M N H :=
sorry

theorem collinear_H_M_N_O : collinear H M N O :=
sorry

end collinear_M_N_H_collinear_H_M_N_O_l637_637319


namespace logsine_triple_sum_l637_637733

theorem logsine_triple_sum : 
  log 2 (sin 10) + log 2 (sin 50) + log 2 (sin 70) = -3 := 
sorry

end logsine_triple_sum_l637_637733


namespace construct_rhombus_l637_637559

-- Define data structure representing a point in a 2-dimensional Euclidean space.
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for four points to form a rhombus.
def isRhombus (A B C D : Point) : Prop :=
  (A.x - B.x) ^ 2 + (A.y - B.y) ^ 2 = (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 ∧
  (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 = (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 ∧
  (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 = (D.x - A.x) ^ 2 + (D.y - A.y) ^ 2

-- Define circumradius condition for triangle ABC
def circumradius (A B C : Point) (R : ℝ) : Prop := sorry -- Detailed definition would be added here.

-- Define inradius condition for triangle BCD
def inradius (B C D : Point) (r : ℝ) : Prop := sorry -- Detailed definition would be added here.

-- The proposition to be proved: We can construct the rhombus ABCD given R and r.
theorem construct_rhombus (A B C D : Point) (R r : ℝ) :
  (circumradius A B C R) →
  (inradius B C D r) →
  isRhombus A B C D :=
by
  sorry

end construct_rhombus_l637_637559


namespace power_function_m_value_l637_637659

theorem power_function_m_value (m : ℝ) (f : ℝ → ℝ) : 
  (∀ x : ℝ, f x = (m + 2) * x^(m - 1)) → 
  (∃ (k n : ℝ), ∀ x : ℝ, f x = k * x^n) → 
  m = -1 :=
by
  intros h1 h2
  obtain ⟨k, n, h⟩ := h2
  have h_eq : (m + 2) * x^(m - 1) = k * x^n := by exact h_eq
  sorry

end power_function_m_value_l637_637659


namespace parabola_vertex_distance_l637_637931

theorem parabola_vertex_distance :
  let equation := fun (x y : ℝ) => (real.sqrt (x^2 + y^2) + |y - 1| = 5)
  let vertex1 := (0, 3)
  let vertex2 := (0, -2)
  dist vertex1 vertex2 = 5 :=
by
  sorry

end parabola_vertex_distance_l637_637931


namespace quadratic_root_value_l637_637989

theorem quadratic_root_value (x : ℝ) (h : x = -1) : sqrt (3 - x) = 2 := by
  sorry

end quadratic_root_value_l637_637989


namespace hyperbola_equation_l637_637794

theorem hyperbola_equation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h1 : a^2 + b^2 = 4) (h2 : b / a = real.sqrt 3) : 
  (b = real.sqrt 3 ∧ a = 1) → (∀ x y : ℝ, x^2 - (y^2 / 3) = 1) := 
sorry

end hyperbola_equation_l637_637794


namespace find_y_value_l637_637408

theorem find_y_value (x y : ℝ) (h1 : x^2 + y^2 - 4 = 0) (h2 : x^2 - y + 2 = 0) : y = 2 :=
by sorry

end find_y_value_l637_637408


namespace volume_inhaled_per_breath_is_correct_l637_637576

def breaths_per_minute : ℤ := 17
def volume_inhaled_24_hours : ℤ := 13600
def minutes_per_hour : ℤ := 60
def hours_per_day : ℤ := 24

def total_minutes_24_hours : ℤ := hours_per_day * minutes_per_hour
def total_breaths_24_hours : ℤ := total_minutes_24_hours * breaths_per_minute
def volume_per_breath := (volume_inhaled_24_hours : ℚ) / (total_breaths_24_hours : ℚ)

theorem volume_inhaled_per_breath_is_correct :
  volume_per_breath = 0.5556 := by
  sorry

end volume_inhaled_per_breath_is_correct_l637_637576


namespace system_of_equations_solution_l637_637868

theorem system_of_equations_solution :
  ∀ (a b : ℝ),
  (-2 * a + b^2 = Real.cos (π * a + b^2) - 1 ∧ b^2 = Real.cos (2 * π * a + b^2) - 1) ↔ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 0) :=
by
  intro a b
  sorry

end system_of_equations_solution_l637_637868


namespace least_positive_integer_added_to_575_multiple_4_l637_637297

theorem least_positive_integer_added_to_575_multiple_4 :
  ∃ n : ℕ, n > 0 ∧ (575 + n) % 4 = 0 ∧ 
           ∀ m : ℕ, (m > 0 ∧ (575 + m) % 4 = 0) → n ≤ m := by
  sorry

end least_positive_integer_added_to_575_multiple_4_l637_637297


namespace expected_number_of_games_l637_637624

def prob_tim_wins_odd : ℚ := 3 / 4
def prob_allen_wins_even : ℚ := 3 / 4

def match_ends_cond (games_won_tim games_won_allen : ℕ) : Prop :=
  abs (games_won_tim - games_won_allen) = 2

def expected_games : ℚ := 16 / 3

theorem expected_number_of_games :
  (∃ n : ℚ, n = 16 / 3) :=
begin
  use expected_games,
  sorry
end

end expected_number_of_games_l637_637624


namespace length_of_MN_l637_637136

theorem length_of_MN
  (A B C M N : Type)
  [∀ x : Type, Inhabited x] -- Making sure each point has coordinates
  (h1 : isMidpoint M B C)
  (h2 : isAngleBisector A N (angle A B C))
  (h3 : isPerpendicular (lineSeg B N) (lineSeg A N))
  (h4 : dist A B = 14)
  (h5 : dist A C = 19) :
  dist M N = 5 / 2 := sorry

end length_of_MN_l637_637136


namespace specified_time_eq_l637_637681

noncomputable def slow_horse_days (x : ℝ) := x + 1
noncomputable def fast_horse_days (x : ℝ) := x - 3

theorem specified_time_eq (x : ℝ) (h1 : slow_horse_days x > 0) (h2 : fast_horse_days x > 0) :
  (900 / slow_horse_days x) * 2 = 900 / fast_horse_days x :=
by
  sorry

end specified_time_eq_l637_637681


namespace choosing_ways_president_vp_committee_l637_637149

theorem choosing_ways_president_vp_committee :
  let n := 10
  let president_choices := n
  let vp_choices := n - 1
  let committee_choices := (n - 2) * (n - 3) / 2
  let total_choices := president_choices * vp_choices * committee_choices
  total_choices = 2520 := by
  let n := 10
  let president_choices := n
  let vp_choices := n - 1
  let committee_choices := (n - 2) * (n - 3) / 2
  let total_choices := president_choices * vp_choices * committee_choices
  have : total_choices = 2520 := by
    sorry
  exact this

end choosing_ways_president_vp_committee_l637_637149


namespace rationals_same_color_l637_637694

noncomputable def coloring (n : ℕ) := ℚ → Fin n

axiom coloring_property (color : coloring n) :
  ∀ (a b : ℚ), color a ≠ color b → color ((a + b) / 2) ≠ color a ∧ color ((a + b) / 2) ≠ color b

theorem rationals_same_color (n : ℕ) (color : coloring n) :
  ∃ c, ∀ q : ℚ, color q = c :=
begin
  sorry
end

end rationals_same_color_l637_637694


namespace talia_father_age_l637_637152

variable (talia_age : ℕ)
variable (mom_age : ℕ)
variable (dad_age : ℕ)

-- Conditions
def condition1 := talia_age + 7 = 20
def condition2 := mom_age = 3 * talia_age
def condition3 := dad_age + 3 = mom_age

-- Theorem to prove
theorem talia_father_age (h1 : condition1) (h2 : condition2) (h3 : condition3) : dad_age = 36 :=
by
  sorry

end talia_father_age_l637_637152


namespace compute_expression_l637_637325

theorem compute_expression : 3034 - (1002 / 200.4) = 3029 := by
  sorry

end compute_expression_l637_637325


namespace rounded_fraction_is_correct_l637_637912

-- Define the fraction 8/11
def fraction : ℚ := 8 / 11

-- Define the expected rounded result
def rounded_result : ℚ := 0.73

-- Prove that the fraction rounded to two decimal places equals the expected result
theorem rounded_fraction_is_correct : 
  Real.round_to 2 fraction = rounded_result :=
by
  sorry

end rounded_fraction_is_correct_l637_637912


namespace train_traveled_through_marked_segment_l637_637272

theorem train_traveled_through_marked_segment 
    (lines : ℕ) (time_between_stations : ℕ) (train_direction_reverses : Prop) 
    (switch_lines_at_transfer : Prop) (A B : Station) (journey_time : ℕ) :
    lines = 3 →
    time_between_stations = 1 →
    train_direction_reverses →
    switch_lines_at_transfer →
    journey_time = 2016 →
    A ≠ B →
    ∃ marked_segment : Segment, train_passed_segment A B marked_segment journey_time :=
by
  intro h_lines h_time h_reverse h_switch h_journey A_ne_B
  sorry


end train_traveled_through_marked_segment_l637_637272


namespace tracy_two_dogs_food_consumption_l637_637629

theorem tracy_two_dogs_food_consumption
  (cups_per_meal : ℝ)
  (meals_per_day : ℝ)
  (pounds_per_cup : ℝ)
  (num_dogs : ℝ) :
  cups_per_meal = 1.5 →
  meals_per_day = 3 →
  pounds_per_cup = 1 / 2.25 →
  num_dogs = 2 →
  num_dogs * (cups_per_meal * meals_per_day) * pounds_per_cup = 4 := by
  sorry

end tracy_two_dogs_food_consumption_l637_637629


namespace minimum_value_of_geometric_sum_l637_637756

noncomputable def minimum_geometric_sum (a : ℕ → ℝ) (q : ℝ) (h_geo_seq : ∀ n, a (n + 1) = a n * q)
  (h_pos : ∀ n, 0 < a n) (h_geo_mean : (a 1 * a 17) ^ (1/2) = 2) : ℝ :=
  Inf { s : ℝ | ∃ n, s = 4 * a 7 + a 11 }

theorem minimum_value_of_geometric_sum (a : ℕ → ℝ) (q : ℝ) (h_geo_seq : ∀ n, a (n + 1) = a n * q)
  (h_pos : ∀ n, 0 < a n) (h_geo_mean : (a 1 * a 17) ^ (1/2) = 2) : minimum_geometric_sum a q h_geo_seq h_pos h_geo_mean = 8 :=
by {
  sorry
}

end minimum_value_of_geometric_sum_l637_637756


namespace find_k_l637_637176

-- Define the vector structures for i and j
def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

-- Define the vectors a and b based on i, j, and k
def a : ℝ × ℝ := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : ℝ × ℝ := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Statement of the theorem
theorem find_k (k : ℝ) : dot_product a (b k) = 0 → k = 6 :=
by sorry

end find_k_l637_637176


namespace dogs_food_consumption_l637_637628

def cups_per_meal_per_dog : ℝ := 1.5
def number_of_dogs : ℝ := 2
def meals_per_day : ℝ := 3
def cups_per_pound : ℝ := 2.25

theorem dogs_food_consumption : 
  ((cups_per_meal_per_dog * number_of_dogs) * meals_per_day) / cups_per_pound = 4 := 
by
  sorry

end dogs_food_consumption_l637_637628


namespace dot_product_range_l637_637111

variables {x y : ℝ}
def a := (-1, 1 : ℝ × ℝ)
def b := (x, y : ℝ × ℝ)
def ab := (a.1 + 2 * b.1, a.2 + 2 * b.2)

theorem dot_product_range :
  (a = (-1, 1)) →
  (a.1 = ab.1 ∨ a.1 = -ab.1) → (a.2 = ab.2 ∨ a.2 = -ab.2) →
  (∃ x y, y = -x ∧ x < 1 / 2) →
  a.1 * b.1 + a.2 * b.2 ∈ set.Ioi (-1) :=
by sorry

end dot_product_range_l637_637111


namespace sequence_properties_l637_637774

variables {a : ℕ → ℤ} {b : ℕ → ℤ} {d : ℤ} {q : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a n = a 1 + (n - 1) * d

def geometric_sequence (b : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n, b n = b 1 * q^(n - 1)

theorem sequence_properties
  (ha : arithmetic_sequence a d)
  (hb : geometric_sequence b q)
  (h1 : 2 * a 5 - a 3 = 3)
  (h2 : b 2 = 1)
  (h3 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (q = 2 ∨ q = -2) :=
by
  sorry

end sequence_properties_l637_637774


namespace find_x_value_l637_637703

noncomputable def x_value : ℝ := 270

theorem find_x_value (A B C : Type) 
  (rotate_clockwise : A → B → ℝ → A)
  (rotate_counterclockwise : A → B → ℝ → A)
  (h1 : rotate_clockwise A B 450 = C)
  (h2 : ∀ x, x < 360 → rotate_counterclockwise A B x = C) :
  x_value = 270 := 
begin
  sorry
end

end find_x_value_l637_637703


namespace exp_inequality_l637_637902

theorem exp_inequality (n : ℕ) (h : 0 < n) : 2 ≤ (1 + 1 / (n : ℝ)) ^ n ∧ (1 + 1 / (n : ℝ)) ^ n < 3 :=
sorry

end exp_inequality_l637_637902


namespace prob_constraint_sum_digits_l637_637537

noncomputable def P (N : ℕ) :=
  let favorable_positions := (Nat.floor (2 * N / 5)) + 1 + (N - (Nat.ceil (3 * N / 5)) + 1)
  favorable_positions / (N + 1 : ℝ)

-- The objective is to establish the sum of the digits of the smallest multiple of 5 where P(N) < 321/400

theorem prob_constraint_sum_digits :
  let min_N := (List.range 1000).find (λ n, n % 5 = 0 ∧ P n < 321 / 400) ∨ 480 -- Use 480 as per problem's solution boundary
  let digit_sum := List.sum (List.map (λ c, c.toNat - '0'.toNat) (min_N.digits 10))
  digit_sum = 12 := 
sorry

end prob_constraint_sum_digits_l637_637537


namespace max_distance_traveled_l637_637209

-- Definitions based on conditions
def front_tire_lifespan := 25000
def rear_tire_lifespan := 15000

-- Theorem statement
theorem max_distance_traveled : 
  ∃ (x y: ℝ), x = 9375 ∧ y = 18750 ∧ 
  (∀ (d: ℝ), 
    d = x + front_tire_lifespan * (1 - x / rear_tire_lifespan) ∨ 
    d = x + rear_tire_lifespan * (1 - x / front_tire_lifespan) → 
    d <= y) := 
begin
  sorry
end

end max_distance_traveled_l637_637209


namespace regression_comparison_l637_637946

theorem regression_comparison :
  let X := [(1,0), (2,2), (3,1), (4,3), (5,4)],
      Y := [(1,4), (2,3), (3,1), (4,2), (5,0)],
      sum_X_X := -2 * -2.5 + -1 * -0.5 + 0 + 1 * 0.5 + 2 * 1.5,
      sum_X_sq := (-2)^2 + (-1)^2 + 1^2 + 2^2,
      sum_Y_X := -2 * 2 + -1 * 1 + 0 + 1 * 0 + 2 * -2,
      sum_Y_sq := (-2)^2 + (-1)^2 + 1^2 + 2^2,
      avg_X_X := (3, 2.5),
      avg_Y_X := (3, 2),
      b1 := sum_X_X / sum_X_sq,
      a1 := 2.5 - b1 * 3,
      b2 := sum_Y_X / sum_Y_sq,
      a2 := 2 + b2 * 3
  in b1 > b2 ∧ a1 < a2 :=
by
  sorry

end regression_comparison_l637_637946


namespace six_points_in_rectangle_l637_637566

theorem six_points_in_rectangle :
  ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (0 ≤ (points i).fst ∧ (points i).fst ≤ 4) ∧ (0 ≤ (points i).snd ∧ (points i).snd ≤ 3)) →
  ∃ (i j : Fin 6), i ≠ j ∧ Real.dist (points i) (points j) ≤ Real.sqrt 5 := 
sorry

end six_points_in_rectangle_l637_637566


namespace domain_f_range_f_monotonicity_f_odd_even_f_periodicity_f_l637_637098

noncomputable def f (x : ℝ) : ℝ := log ((sin x) - (cos x))

theorem domain_f :
  ∀ k : ℤ, 2 * k * real.pi + real.pi / 4 < x ∧ x < 2 * k * real.pi + 5 * real.pi / 4 → (sin x) - (cos x) > 0 :=
sorry

theorem range_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) → y ≤ -1/2 :=
sorry

theorem monotonicity_f :
  ∀ k : ℤ, 
    (∀ x : ℝ, 2 * k * real.pi + 3 * real.pi / 4 ≤ x ∧ x < 2 * k * real.pi + 5 * real.pi / 4 → (sin x) - (cos x)) 
    ∧ (∀ x : ℝ, 2 * k * real.pi + real.pi / 4 < x ∧ x < 2 * k * real.pi + 3 * real.pi / 4 → (sin x) - (cos x)) :=
sorry 

theorem odd_even_f :
  ¬(∀ x : ℝ, f (-x) = f x) ∧ ¬(∀ x : ℝ, f (-x) = -f x) :=
sorry

theorem periodicity_f :
  ∀ x : ℝ, f (x + 2 * real.pi) = f x ∧ ¬(∃ T : ℝ, T > 0 ∧ T < 2 * real.pi ∧ ∀ x : ℝ, f (x + T) = f x) :=
sorry

end domain_f_range_f_monotonicity_f_odd_even_f_periodicity_f_l637_637098


namespace no_three_nat_numbers_with_sum_power_of_three_l637_637360

noncomputable def powers_of_3 (n : ℕ) : ℕ := 3^n

theorem no_three_nat_numbers_with_sum_power_of_three :
  ¬ ∃ (a b c : ℕ) (k m n : ℕ), a + b = powers_of_3 k ∧ b + c = powers_of_3 m ∧ c + a = powers_of_3 n :=
by
  sorry

end no_three_nat_numbers_with_sum_power_of_three_l637_637360


namespace increasing_order_magnitudes_l637_637798

variable (x : ℝ)

noncomputable def y := x^x
noncomputable def z := x^(x^x)

theorem increasing_order_magnitudes (h1 : 1 < x) (h2 : x < 1.1) : x < y x ∧ y x < z x :=
by
  have h3 : y x = x^x := rfl
  have h4 : z x = x^(x^x) := rfl
  sorry

end increasing_order_magnitudes_l637_637798


namespace bicycles_wheels_l637_637953

theorem bicycles_wheels (b : ℕ) (h1 : 3 * b + 4 * 3 + 7 * 1 = 25) : b = 2 :=
sorry

end bicycles_wheels_l637_637953


namespace orthocenter_nine_point_center_l637_637446

noncomputable def orthocenter (z1 z2 z3 : ℂ) : ℂ :=
  z1 + z2 + z3

noncomputable def nine_point_center (z1 z2 z3 : ℂ) : ℂ :=
  (z1 + z2 + z3) / 2

theorem orthocenter_nine_point_center (z1 z2 z3 : ℂ) (h1 : complex.abs z1 = 1) (h2 : complex.abs z2 = 1) (h3 : complex.abs z3 = 1) :
  orthocenter z1 z2 z3 = z1 + z2 + z3 ∧ nine_point_center z1 z2 z3 = (z1 + z2 + z3) / 2 := 
by
  split
  sorry
  sorry

end orthocenter_nine_point_center_l637_637446


namespace coins_in_sacks_sequence_l637_637202

open Finset

theorem coins_in_sacks_sequence (coins : ℕ → ℕ)
  (h1 : coins 1 = 60)
  (h2 : coins 2 = 30)
  (h3 : coins 3 = 20)
  (h4 : coins 4 = 15)
  (sequence_property : ∀ n, coins n ∈ {60, 30, 20, 15, 12, 10, 6, 5, 4, 3, 2, 1}) :
  coins 5 = 12 ∧ coins 6 = 10 :=
by { sorry }

end coins_in_sacks_sequence_l637_637202


namespace digit_15_of_sum_reciprocals_l637_637974

/-- 
What is the 15th digit after the decimal point of the sum of the decimal equivalents 
for the fractions 1/9 and 1/11?
-/
theorem digit_15_of_sum_reciprocals :
  let r := (1/9 + 1/11) in
  let d15 := Real.frac (10^(15:ℕ) * r) in
  Int.floor (10 * d15) = 1 :=
by
  let r := (1/9 + 1/11)
  let d15 := Real.frac (10^(15:ℕ) * r)
  have h : Real.toRat d15 = 1 / 10 + d15 - Int.floor d15
  have : Real.frac (10^(15:ℕ) * r) = r
  sorry

end digit_15_of_sum_reciprocals_l637_637974


namespace replace_multiplication_with_add_sub_l637_637226

theorem replace_multiplication_with_add_sub (n : ℕ) (h : n = 2012) :
  ∃ (f : ℕ → ℤ), (∀ k, 1 ≤ k ∧ k < n → (f k = 1 * k^2 ∨ f k = -1 * k^2)) ∧ 
  ∑ k in finset.range (n + 1), f k = 2012 :=
begin
 sorry
end

end replace_multiplication_with_add_sub_l637_637226


namespace charged_amount_is_35_l637_637527

-- Definitions based on conditions
def annual_interest_rate : ℝ := 0.05
def owed_amount : ℝ := 36.75
def time_in_years : ℝ := 1

-- The amount charged on the account in January
def charged_amount (P : ℝ) : Prop :=
  owed_amount = P + (P * annual_interest_rate * time_in_years)

-- The proof statement
theorem charged_amount_is_35 : charged_amount 35 := by
  sorry

end charged_amount_is_35_l637_637527


namespace find_a_enclosed_area_l637_637529

-- Define the function for the parameterized curve
def curve_x (a : ℝ) (t : ℝ) : ℝ := t + Real.exp(a * t)
def curve_y (a : ℝ) (t : ℝ) : ℝ := -t + Real.exp(a * t)

-- Define the condition that the curve touches the x-axis
def touches_x_axis (a : ℝ) (t : ℝ) : Prop := 
  curve_y(a, t) = 0

-- Define the slope of the tangent line to the curve
def slope_tangent (a : ℝ) (t : ℝ) : ℝ := 
  let dxdt := 1 + a * Real.exp(a * t)
  let dydt := -1 + a * Real.exp(a * t)
  dydt / dxdt

-- First theorem: Determine the value of a
theorem find_a (a t: ℝ) (h_pos: 0 < a) (h_tang: touches_x_axis a t) : 
  a = 1 / Real.exp(1/t) := by
sorry

-- Second theorem: Calculate the enclosed area
theorem enclosed_area (a : ℝ := 1/Real.exp(1/e)) : 
  ∫ t in 0..Real.exp(1), curve_x(a, t) - t = (Real.exp(1) ^ 2)/2 - Real.exp(1) :=
by
  sorry

end find_a_enclosed_area_l637_637529


namespace initial_interest_rate_l637_637254

theorem initial_interest_rate
    (P R : ℝ) 
    (h1 : P * R = 10120) 
    (h2 : P * (R + 6) = 12144) : 
    R = 30 :=
sorry

end initial_interest_rate_l637_637254


namespace sports_club_membership_l637_637145

theorem sports_club_membership :
  ∀ (total T B_and_T neither : ℕ),
    total = 30 → 
    T = 19 →
    B_and_T = 9 →
    neither = 2 →
  ∃ (B : ℕ), B = 18 :=
by
  intros total T B_and_T neither ht hT hBandT hNeither
  let B := total - neither - T + B_and_T
  use B
  sorry

end sports_club_membership_l637_637145


namespace find_arith_seq_params_l637_637541

noncomputable section

def a_arith_seq (n : ℕ) (a1 d : ℝ) : ℝ :=
  a1 + n * d

def b_geom_seq (n : ℕ) (a1 d : ℝ) : ℝ :=
  (a_arith_seq n a1 d) ^ 2

theorem find_arith_seq_params (a1 : ℝ) (d : ℝ)
  (h1 : a1 < a_arith_seq 1 a1 d)
  (h2 : ∑' n, b_geom_seq n a1 d = sqrt 2 + 1) :
  a1 = -sqrt 2 ∧ d = 2 * sqrt 2 - 2 :=
by
  sorry

end find_arith_seq_params_l637_637541


namespace patty_can_avoid_chores_l637_637214

theorem patty_can_avoid_chores (money_per_pack packs total_cookies_per_pack chores kid_cookies_cost packs_bought total_cookies total_weekly_cost weeks : ℕ)
    (h1 : money_per_pack = 3)
    (h2 : packs = 15 / money_per_pack)
    (h3 : total_cookies_per_pack = 24)
    (h4 : total_cookies = (p : ℕ) → packs * total_cookies_per_pack)
    (h5 : chores = 4)
    (h6 : kid_cookies_cost = 3)
    (h7 : total_weekly_cost = 2 * chores * kid_cookies_cost)
    (h8 : weeks = (total_cookies / total_weekly_cost)) : 
  weeks = 10 :=
by sorry

end patty_can_avoid_chores_l637_637214


namespace length_of_AE_l637_637148

variable (ABCD : Type) [quadrilateral ABCD]
variable (AB CD AC : ℝ) (E : point)
variable (area_AED area_BEC : ℝ)

-- Conditions
variables (hAB : AB = 12) (hCD : CD = 15) (hAC : AC = 18)
variables (h_intersect : intersect_at E diagonal_AC diagonal_BD)
variables (h_area_ratio : area_AED / area_BEC = 4 / 5)

-- The statement to prove
theorem length_of_AE 
  (h : convex_quad ABCD)
  (h1 : AB = 12) 
  (h2 : CD = 15)
  (h3 : AC = 18)
  (h4 : intersect_at E diagonal_AC diagonal_BD)
  (h5 : area_AED / area_BEC = 4 / 5) :
  length_AE = 8 := 
sorry

end length_of_AE_l637_637148


namespace percentage_of_couples_ordering_dessert_and_coffee_l637_637256

variable (D : ℝ) (DC : ℝ)

theorem percentage_of_couples_ordering_dessert_and_coffee
  (h1 : D = 0.75)
  (h2 : DC = 0.80 * D) :
  DC = 0.60 :=
by
    rw [h1, h2]
    norm_num
    sorry

end percentage_of_couples_ordering_dessert_and_coffee_l637_637256


namespace length_of_BC_value_of_sin_2C_l637_637490
noncomputable def triangle (A B C : Type) [realField A] (AB AC : A) (angleA : real) := 
  AB = 2 ∧ AC = 3 ∧ angleA = real.pi / 3

theorem length_of_BC (A B C : Type) [realField A] (BC : A) :
  triangle A B C 2 3 (real.pi / 3) → BC = real.sqrt 7 := sorry

theorem value_of_sin_2C (A B C : Type) [realField A] (sin2C : A) :
  triangle A B C 2 3 (real.pi / 3) → sin2C = 4 * real.sqrt 3 / 7 := sorry

end length_of_BC_value_of_sin_2C_l637_637490


namespace modulus_of_z_l637_637183

-- Defining the imaginary unit
noncomputable def i : ℂ := complex.I

-- Given conditions
def condition (z : ℂ) : Prop :=
  (1 + i) * complex.conj z = (1 - i)^2

-- The theorem to prove |z| = √2
theorem modulus_of_z (z : ℂ) (h : condition z) : complex.abs z = real.sqrt 2 :=
by
  intro h,
  sorry

end modulus_of_z_l637_637183


namespace solve_system_l637_637582

theorem solve_system :
  ∃ (x y : ℝ), x + y = 5 ∧ 3 * x + y = 7 ∧ x = 1 ∧ y = 4 :=
begin
  sorry
end

end solve_system_l637_637582


namespace student_attendance_l637_637840

open Set

variable (n k : ℕ) (classes : Finset (Fin n)) (students : Finset (Fin k))
variable (attend : Fin n → Finset (Fin k))

-- Conditions
def condition1 : Prop := students.card = k ∧ classes.card = n
def condition2 : Prop := ∀ s1 s2 : Fin k, s1 ≠ s2 → ∃ c : Fin n, s1 ∈ attend c ∧ s2 ∈ attend c
def condition3 : Prop := ∀ c : Fin n, (attend c).card < k
def condition4 : Prop := ¬IsPerfectSquare (k - 1)

-- Main statement
theorem student_attendance (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) :
  ∃ s : Fin k, ∃ classes_attended : Finset (Fin n), classes_attended.card ≥ nat.sqrt k ∧ ∀ c ∈ classes_attended, s ∈ attend c :=
sorry

end student_attendance_l637_637840


namespace find_arithmetic_progression_angles_l637_637590

theorem find_arithmetic_progression_angles
  (triangle_ABC : is_isosceles_right_triangle (angle_A : 90 degree))
  (angles_in_arithmetic_progression : 
    let d := some_non_zero_common_difference in
    [∠SAB, ∠SCA, ∠SAC, ∠SBA] = [α - 2 * φ, α - φ, α, α + φ] where α = π/2 and 
    φ = (1/2) * arccos(sqrt(2) - 1))
  (areas_in_geometric_progression : 
    geometric_progression [area_SAB, area_ABC, area_SAC])
  :
  let angles = [
    (π/2 - (arccos (sqrt(2) - 1))),
    (π/2 - (1/2 * arccos (sqrt(2) - 1))),
    (π/2),
    (π/2 + (1/2 * arccos (sqrt(2) - 1)))]
  :
  ∃ (S A B C : Point), 
    ∀ {SA AB SC AC SB BC}, 
      (triangle_ABC ∧ 
      angles_in_arithmetic_progression ∧ 
      areas_in_geometric_progression 
      → angles = [ 
          (π/2 - (arccos (sqrt(2) - 1))),
          (π/2 - (1/2 * arccos (sqrt(2) - 1))),
          (π/2),
          (π/2 + (1/2 * arccos (sqrt(2) - 1)))]):
sorry -- Proof to be completed

end find_arithmetic_progression_angles_l637_637590


namespace length_FA_proof_l637_637665

noncomputable def length_FA (A B C D E F : Type*) [group_A : AddCommGroup A]
[B1 : B = A + 1] [B2 : C = B + 2] [B3 : D = C + 3] [B4 : E = D + 4] [B5 : F = E + 5] : Type :=
{ r : A // {
    h1 : add_comm_group A,
    h2 : group_A [A, B, C, D, E, F],
    h3 : ∃ P, (A P = 6)
  }
}

theorem length_FA_proof (A B C D E F : Type*) [group_A : AddCommGroup A]
[B1 : B = A + 1] [B2 : C = B + 2] [B3 : D = C + 3] [B4 : E = D + 4] [B5 : F = E + 5]
(AD BE CF: list ℝ) : length_FA A B C D E F = 6 := by {
  sorry
}

end length_FA_proof_l637_637665


namespace inverse_of_11_mod_1021_l637_637706

theorem inverse_of_11_mod_1021 : ∃ x : ℕ, x < 1021 ∧ 11 * x ≡ 1 [MOD 1021] := by
  use 557
  -- We leave the proof as an exercise.
  sorry

end inverse_of_11_mod_1021_l637_637706


namespace num_of_factors_l637_637623

theorem num_of_factors (a b c : ℕ) (ha : ∃ p₁ : ℕ, prime p₁ ∧ (a = p₁ ^ 2 ∨ a = p₁ ^ 3))
                               (hb : ∃ p₂ : ℕ, prime p₂ ∧ (b = p₂ ^ 2 ∨ b = p₂ ^ 3))
                               (hc : ∃ p₃ : ℕ, prime p³ ∧ (c = p₃ ^ 2 ∨ c = p₃ ^ 3))
                               (habc_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) :
  nat.num_divisors (a^2 * b^3 * c^4) = 850 :=
sorry

end num_of_factors_l637_637623


namespace total_payment_per_month_l637_637997

def box_length := 15
def box_width := 12
def box_height := 10
def total_volume := 1080000
def cost_per_box := 0.5

theorem total_payment_per_month :
  let volume_per_box := box_length * box_width * box_height in
  let number_of_boxes := total_volume / volume_per_box in
  let total_payment := number_of_boxes * cost_per_box in
  total_payment = 300 :=
by
  let volume_per_box := box_length * box_width * box_height
  let number_of_boxes := total_volume / volume_per_box
  let total_payment := number_of_boxes * cost_per_box
  sorry

end total_payment_per_month_l637_637997


namespace second_newly_inserted_number_eq_l637_637249

theorem second_newly_inserted_number_eq : 
  ∃ q : ℝ, (q ^ 12 = 2) ∧ (1 * (q ^ 2) = 2 ^ (1 / 6)) := 
by
  sorry

end second_newly_inserted_number_eq_l637_637249


namespace arithmetic_sequence_sum_l637_637147

theorem arithmetic_sequence_sum (a_n : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : S 9 = a_n 4 + a_n 5 + a_n 6 + 72)
  (h2 : ∀ n, S n = n * (a_n 1 + a_n n) / 2)
  (h3 : ∀ n, a_n (n+1) - a_n n = d)
  (h4 : a_n 1 + a_n 9 = a_n 3 + a_n 7)
  (h5 : a_n 3 + a_n 7 = a_n 4 + a_n 6)
  (h6 : a_n 4 + a_n 6 = 2 * a_n 5) : 
  a_n 3 + a_n 7 = 24 := 
sorry

end arithmetic_sequence_sum_l637_637147


namespace quadratic_has_distinct_real_roots_l637_637480

open Classical

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_distinct_real_roots (k : ℝ) (h_nonzero : k ≠ 0) : 
  (k > -1) ↔ (discriminant k (-2) (-1) > 0) :=
by
  unfold discriminant
  simp
  linarith

end quadratic_has_distinct_real_roots_l637_637480


namespace am_an_ge_mm_nn_l637_637188

theorem am_an_ge_mm_nn (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  let a := (m ^ (m + 1) + n ^ (n + 1)) / (m ^ m + n ^ n)
  in a ^ m + a ^ n ≥ m ^ m + n ^ n :=
by
  sorry

end am_an_ge_mm_nn_l637_637188


namespace range_of_m_l637_637937

variables {𝕜 : Type*} [OrderedRing 𝕜] [LinearOrderedField 𝕜]
variables {f : 𝕜 → 𝕜}

-- Conditions
def odd_function (f : 𝕜 → 𝕜) : Prop :=
  ∀ x, f (-x) = -f x

def increasing_function (f : 𝕜 → 𝕜) : Prop :=
  ∀ x y, x < y → f x < f y

axiom f_is_odd : odd_function f
axiom f_is_increasing : increasing_function f
axiom f_domain : ∀ x, x ∈ Ioc (-1 : 𝕜) (1 : 𝕜)

-- Main statement
theorem range_of_m (m : 𝕜) (h : f (1 - m) + f (m^2 - 1) < 0) : 0 < m ∧ m < 1 :=
sorry

end range_of_m_l637_637937


namespace max_value_ln_x_minus_x_on_0_e_l637_637257

noncomputable def f (x : ℝ) : ℝ := real.log x - x

theorem max_value_ln_x_minus_x_on_0_e : 
  ∃ x ∈ set.Ioc 0 real.exp 1, ∀ y ∈ set.Ioc 0 real.exp 1, f y ≤ f x :=
sorry

end max_value_ln_x_minus_x_on_0_e_l637_637257


namespace john_percentage_games_played_l637_637861

-- Define the conditions in Lean
def free_throws_per_game : ℕ := 5 * 2
def total_shots_if_all_games_played : ℕ := 20 * free_throws_per_game
def actual_shots_taken : ℕ := 112
def percentage_games_played : ℝ := (actual_shots_taken : ℝ) / (total_shots_if_all_games_played : ℝ) * 100

theorem john_percentage_games_played : percentage_games_played = 56 := 
by
  -- Since we need the percentage of games played, 
  -- it's calculated from actual_shots_taken / total_shots_if_all_games_played * 100
  sorry

end john_percentage_games_played_l637_637861


namespace trig_identity_l637_637817

theorem trig_identity
  (x : ℝ)
  (h : Real.tan (π / 4 + x) = 2014) :
  1 / Real.cos (2 * x) + Real.tan (2 * x) = 2014 :=
by
  sorry

end trig_identity_l637_637817


namespace unique_b_for_smallest_a_l637_637282

theorem unique_b_for_smallest_a :
  ∃ (a : ℝ) (b : ℝ), (a > 0) ∧ 
  (∀ (x : ℝ), Polynomial.root (Polynomial.C(-a) + Polynomial.X * (Polynomial.C b + Polynomial.X * (Polynomial.C (-a) + Polynomial.X))) x) ∧
  (a = 3 * Real.sqrt 3) ∧ 
  (b = 9) :=
begin
  sorry,
end

end unique_b_for_smallest_a_l637_637282


namespace min_value_of_b1_plus_b2_l637_637277

theorem min_value_of_b1_plus_b2 (b : ℕ → ℕ) (h1 : ∀ n ≥ 1, b (n + 2) = (b n + 4030) / (1 + b (n + 1)))
  (h2 : ∀ n, b n > 0) : ∃ b1 b2, b1 * b2 = 4030 ∧ b1 + b2 = 127 :=
by {
  sorry
}

end min_value_of_b1_plus_b2_l637_637277


namespace area_of_rectangular_field_l637_637260

theorem area_of_rectangular_field 
  (P L W : ℕ) 
  (hP : P = 120) 
  (hL : L = 3 * W) 
  (hPerimeter : 2 * L + 2 * W = P) : 
  (L * W = 675) :=
by 
  sorry

end area_of_rectangular_field_l637_637260


namespace tan_alpha_minus_pi_over_4_l637_637432

theorem tan_alpha_minus_pi_over_4 (α : ℝ) (h₀ : 0 < α ∧ α < π / 2) (h₁ : cos α = sqrt 5 / 5) :
  tan (α - π / 4) = 1 / 3 :=
sorry

end tan_alpha_minus_pi_over_4_l637_637432


namespace sum_of_arithmetic_sequence_l637_637842

theorem sum_of_arithmetic_sequence (a d1 d2 : ℕ) 
  (h1 : d1 = d2 + 2) 
  (h2 : d1 + d2 = 24) 
  (a_pos : 0 < a) : 
  (a + (a + d1) + (a + d1) + (a + d1 + d2) = 54) := 
by 
  sorry

end sum_of_arithmetic_sequence_l637_637842


namespace math_problem_l637_637082

theorem math_problem
  (f g : ℝ → ℝ)
  (hf_domain : ∀ x : ℝ, f x ∈ ℝ)
  (hg_domain : ∀ x : ℝ, g x ∈ ℝ)
  (hf_even : ∀ x : ℝ, f (x + 2) = f (-x - 2))
  (f_g_relation : ∀ x : ℝ, f x + g x = g (2 - x))
  (hf_on_interval : ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = x) :
  
-- Proving that f(x) is an even function
  (∀ x : ℝ, f (-x) = f x) ∧
  -- Proving that the graph of f(x) is symmetric about the point (1,0)
  (∀ x : ℝ, f (2 - x) = -f x) ∧
  -- Proving that 8 is a period of f(x)
  (∀ x : ℝ, f (x + 8) = f x) :=
by
  sorry

end math_problem_l637_637082


namespace sandy_tokens_ratio_l637_637578

theorem sandy_tokens_ratio :
  ∀ (total_tokens : ℕ) (num_siblings : ℕ) (difference : ℕ),
  total_tokens = 1000000 →
  num_siblings = 4 →
  difference = 375000 →
  ∃ (sandy_tokens : ℕ),
  sandy_tokens = (total_tokens - (num_siblings * ((total_tokens - difference) / (num_siblings + 1)))) ∧
  sandy_tokens / total_tokens = 1 / 2 :=
by 
  intros total_tokens num_siblings difference h1 h2 h3
  sorry

end sandy_tokens_ratio_l637_637578


namespace domain_of_ratio_function_l637_637440

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f (2 ^ x)

theorem domain_of_ratio_function (D : Set ℝ) (hD : D = Set.Icc 1 2):
  ∀ f : ℝ → ℝ, (∀ x, g x = f (2 ^ x)) →
  ∃ D' : Set ℝ, D' = {x | 2 ≤ x ∧ x ≤ 4} →
  ∀ y : ℝ, (2 ≤ y ∧ y ≤ 4) → ∃ x : ℝ, y = x + 1 ∧ x ≠ 1 → (1 < x ∧ x ≤ 3) :=
sorry

end domain_of_ratio_function_l637_637440


namespace tangent_line_equation_maximum_area_l637_637448

-- Define the ellipse and its properties.
def ellipse (x y : ℝ) (a : ℝ) : Prop :=
  a > 1 ∧ (x^2 / a^2 + y^2 = 1)

-- Define the circle and its properties.
def circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

-- Define the tangent point conditions on the ellipse from a point on the circle.
def tangent_point (P : ℝ × ℝ) (A : ℝ × ℝ) (m n x1 y1 : ℝ) : Prop :=
  circle P.1 P.2 ∧ ellipse x1 y1 2 ∧ (m = P.1) ∧ (n = P.2) ∧ (x1 = A.1) ∧ (y1 = A.2)

-- Statement 1: Equation of tangent line PA
theorem tangent_line_equation (P A : ℝ × ℝ) (x y x1 y1 : ℝ) :
  tangent_point P A P.1 P.2 x1 y1 →
  ∀ x y, (x1 * x / 4 + y1 * y = 1) :=
sorry

-- Statement 2: Maximum area of triangle OAB
theorem maximum_area (P A B : ℝ × ℝ) (S : ℝ) :
  tangent_point P A P.1 P.2 A.1 A.2 →
  tangent_point P B P.1 P.2 B.1 B.2 →
  S = (4 * real.sqrt(3 * (P.2)^2 + 12)) / (3 * (P.2)^2 + 16) →
  S ≤ real.sqrt(3) / 2 :=
sorry

end tangent_line_equation_maximum_area_l637_637448


namespace table_dimension_equality_l637_637498

theorem table_dimension_equality
  (M K : ℕ)
  (table : Fin M → Fin K → ℝ)
  (rows_sum_eq_one : ∀ i : Fin M, (∑ j : Fin K in Finset.univ, table i j) = 1)
  (columns_sum_eq_one : ∀ j : Fin K, (∑ i : Fin M in Finset.univ, table i j) = 1) :
  M = K :=
by
  sorry

end table_dimension_equality_l637_637498


namespace x_zero_necessary_not_sufficient_l637_637466

def pure_imaginary (z : ℂ) : Prop := (z.re = 0 ∧ z.im ≠ 0)

theorem x_zero_necessary_not_sufficient (x y : ℝ) :
    (x = 0 → pure_imaginary (x + y * complex.I))
    ∧ ¬(x = 0 ↔ pure_imaginary (x + y * complex.I)) := 
sorry

end x_zero_necessary_not_sufficient_l637_637466


namespace last_integer_in_halving_sequence_l637_637942

theorem last_integer_in_halving_sequence (a₀ : ℕ) (h₀ : a₀ = 2000000) : 
  ∃ n : ℕ, ∀ k : ℕ, a₀ / (2 ^ k) ∈ ℕ ∧ (a₀ / (2 ^ n) = 15625 ∧ a₀ / (2 ^ (n + 1)) < 1) :=
sorry

end last_integer_in_halving_sequence_l637_637942


namespace bus_journey_indifferent_l637_637517

noncomputable def walking_speed : ℝ := 1.5 / 30  -- miles per minute (0.05)
noncomputable def bus_distance : ℝ := 1  -- miles
noncomputable def indirect_walk_distance : ℝ := 0.75  -- miles
noncomputable def direct_walk_time : ℝ := 30  -- minutes
noncomputable def bike_time : ℝ := 20  -- minutes
noncomputable def carpool_time : ℝ := 25  -- minutes
noncomputable def train_walk_to_station : ℝ := 1  -- miles
noncomputable def train_time : ℝ := 15  -- minutes
noncomputable def train_walk_from_station : ℝ := 0.5  -- miles

noncomputable def train_total_time : ℝ := 
  (train_walk_to_station / walking_speed) + train_time + (train_walk_from_station / walking_speed)  -- Total train option time (45 minutes)

noncomputable def indirect_walk_time : ℝ := 
  indirect_walk_distance / walking_speed  -- 15 minutes

noncomputable def max_bus_time : ℝ := train_total_time - indirect_walk_time  -- 30 minutes

theorem bus_journey_indifferent : max_bus_time = 30 := 
by 
  have h1: walking_speed = 0.05 := by norm_num
  have h2: train_total_time = 45 := by norm_num [train_total_time, train_walk_to_station, walking_speed, train_time, train_walk_from_station]
  have h3: indirect_walk_time = 15 := by norm_num [indirect_walk_time, indirect_walk_distance, walking_speed]
  have h4: max_bus_time = 30 := by norm_num [max_bus_time, train_total_time, indirect_walk_time]
  exact h4
  
#print bus_journey_indifferent

end bus_journey_indifferent_l637_637517


namespace trajectory_of_M_min_max_values_when_k_half_l637_637751

structure Point (α : Type) :=
  (x : α)
  (y : α)

def dot_product {α} [Field α] (u v : Point α) : α :=
  u.x * v.x + u.y * v.y

def magnitude {α} [Field α] (u : Point α) : α :=
  Real.sqrt (dot_product u u)

noncomputable def problem_conditions {α} [Field α] (k : α) (d : α) : Prop :=
  let O : Point α := {x := 0, y := 0}
  let A : Point α := {x := 2, y := 0}
  let B : Point α := {x := 2, y := 1}
  let C : Point α := {x := 0, y := 1}
  ∀ (M : Point α), 
    let OM := M
    let AM := {x := M.x - A.x, y := M.y - A.y}
    let CM := {x := M.x - C.x, y := M.y - C.y}
    let BM := {x := M.x - B.x, y := M.y - B.y}
    dot_product OM AM = k * (dot_product CM BM - d^2)

theorem trajectory_of_M {α} [Field α] (k : α) (d : α) (h : problem_conditions k d) :
  (k = 1 → ∀ (M : Point α), M.y = 0)
∧ (k = 0 → ∀ (M : Point α), M.x^2 + M.y^2 = 0)
∧ (k > 1 → ∃ (M : Point α), -- Hyperbola condition here
)
∧ ((0 < k ∨ k < 0) → ∃ (M : Point α), -- Ellipse condition here
)
:= sorry

theorem min_max_values_when_k_half {α} [RealField α] (M : Point α) (d : α) (h : problem_conditions (1/2) d) :
  (min (magnitude {x := M.x * 3 - 4, y := M.y * 9}) = sqrt(14)/2)
∧ (max (magnitude {x := M.x * 3 - 4, y := M.y * 9}) = 4)
:= sorry

end trajectory_of_M_min_max_values_when_k_half_l637_637751


namespace dan_has_13_limes_l637_637383

theorem dan_has_13_limes (picked_limes : ℕ) (given_limes : ℕ) (h1 : picked_limes = 9) (h2 : given_limes = 4) : 
  picked_limes + given_limes = 13 := 
by
  sorry

end dan_has_13_limes_l637_637383


namespace total_fruits_picked_l637_637415

theorem total_fruits_picked (g_oranges g_apples a_oranges a_apples o_oranges o_apples : ℕ) :
  g_oranges = 45 →
  g_apples = a_apples + 5 →
  a_oranges = g_oranges - 18 →
  a_apples = 15 →
  o_oranges = 6 * 3 →
  o_apples = 6 * 2 →
  g_oranges + g_apples + a_oranges + a_apples + o_oranges + o_apples = 137 :=
by
  intros
  sorry

end total_fruits_picked_l637_637415


namespace percentage_of_girls_who_like_basketball_l637_637524

theorem percentage_of_girls_who_like_basketball 
  (total_students : ℕ)
  (percentage_girls : ℝ)
  (percentage_boys_basketball : ℝ)
  (factor_girls_to_boys_not_basketball : ℝ)
  (total_students_eq : total_students = 25)
  (percentage_girls_eq : percentage_girls = 0.60)
  (percentage_boys_basketball_eq : percentage_boys_basketball = 0.40)
  (factor_girls_to_boys_not_basketball_eq : factor_girls_to_boys_not_basketball = 2) 
  : 
  ((factor_girls_to_boys_not_basketball * (total_students * (1 - percentage_girls) * (1 - percentage_boys_basketball))) / 
  (total_students * percentage_girls)) * 100 = 80 :=
by
  sorry

end percentage_of_girls_who_like_basketball_l637_637524


namespace arithmetic_series_first_term_l637_637049

theorem arithmetic_series_first_term 
  (a d : ℝ)
  (h1 : 50 * (2 * a + 99 * d) = 1800)
  (h2 : 50 * (2 * a + 199 * d) = 6300) :
  a = -26.55 :=
by
  sorry

end arithmetic_series_first_term_l637_637049


namespace max_sin_A_l637_637512

-- Conditions
variables {a b c : ℝ} (triangle_area_max : ℝ)
def semi_perimeter (a b c : ℝ) := (a + b + c) / 2
def herons_formula (a b c : ℝ) (p : ℝ) := (p * (p - a) * (p - b) * (p - c)).sqrt

-- Given values
axiom BC_is_6 : c = 6
axiom AB_is_2AC : a = 2 * b

-- Theorem to prove
theorem max_sin_A : ∃ (A : ℝ), sin A = 3 / 5 :=
sorry

end max_sin_A_l637_637512


namespace min_value_of_sum_log_l637_637815

theorem min_value_of_sum_log
  (a b : ℝ) 
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : log 5 a + log 5 b = 2) : 
  a + b ≥ 10 :=
sorry

end min_value_of_sum_log_l637_637815


namespace yoongi_stacked_higher_by_one_cm_l637_637956

def height_box_A : ℝ := 3
def height_box_B : ℝ := 3.5
def boxes_stacked_by_Taehyung : ℕ := 16
def boxes_stacked_by_Yoongi : ℕ := 14
def height_Taehyung_stack : ℝ := height_box_A * boxes_stacked_by_Taehyung
def height_Yoongi_stack : ℝ := height_box_B * boxes_stacked_by_Yoongi

theorem yoongi_stacked_higher_by_one_cm :
  height_Yoongi_stack = height_Taehyung_stack + 1 :=
by
  sorry

end yoongi_stacked_higher_by_one_cm_l637_637956


namespace smallest_positive_integer_in_range_l637_637269

theorem smallest_positive_integer_in_range :
  ∃ n : ℕ, 1 < n ∧ (n % 5 = 1) ∧ (n % 7 = 1) ∧ (n % 8 = 1) ∧ 240 ≤ n ∧ n ≤ 359 :=
begin
  sorry
end

end smallest_positive_integer_in_range_l637_637269


namespace revenue_function_correct_strategy_not_profitable_l637_637515

-- Given conditions 
def purchase_price : ℝ := 1
def last_year_price : ℝ := 2
def last_year_sales_volume : ℕ := 10000
def last_year_revenue : ℝ := 20000
def proportionality_constant : ℝ := 4
def increased_sales_volume (x : ℝ) : ℝ := proportionality_constant * (2 - x) ^ 2

-- Questions translated to Lean statements
def revenue_this_year (x : ℝ) : ℝ := 4 * x ^ 3 - 20 * x ^ 2 + 33 * x - 17

theorem revenue_function_correct (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) : 
    revenue_this_year x = 4 * x ^ 3 - 20 * x ^ 2 + 33 * x - 17 :=
by
  sorry

theorem strategy_not_profitable (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) : 
    revenue_this_year x ≤ last_year_revenue :=
by
  sorry

end revenue_function_correct_strategy_not_profitable_l637_637515


namespace candy_cost_l637_637164

theorem candy_cost (J H C : ℕ) (h1 : J + 7 = C) (h2 : H + 1 = C) (h3 : J + H < C) : C = 7 :=
by
  sorry

end candy_cost_l637_637164


namespace find_f_2014_l637_637066

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f x = f (x + 1) - f (x + 2)
axiom interval_def : ∀ x, 0 < x ∧ x < 3 → f x = x^2

theorem find_f_2014 : f 2014 = -1 := sorry

end find_f_2014_l637_637066


namespace math_problem_l637_637125

theorem math_problem (a b c : ℝ) (h1 : a + 2 * b + 3 * c = 12) (h2 : a^2 + b^2 + c^2 = a * b + b * c + c * a) : a + b^2 + c^3 = 14 :=
by
  sorry

end math_problem_l637_637125


namespace chips_reach_end_l637_637190

theorem chips_reach_end (n k : ℕ) (h : n > k * 2^k) : True := sorry

end chips_reach_end_l637_637190


namespace patty_can_avoid_chores_l637_637213

theorem patty_can_avoid_chores (money_per_pack packs total_cookies_per_pack chores kid_cookies_cost packs_bought total_cookies total_weekly_cost weeks : ℕ)
    (h1 : money_per_pack = 3)
    (h2 : packs = 15 / money_per_pack)
    (h3 : total_cookies_per_pack = 24)
    (h4 : total_cookies = (p : ℕ) → packs * total_cookies_per_pack)
    (h5 : chores = 4)
    (h6 : kid_cookies_cost = 3)
    (h7 : total_weekly_cost = 2 * chores * kid_cookies_cost)
    (h8 : weeks = (total_cookies / total_weekly_cost)) : 
  weeks = 10 :=
by sorry

end patty_can_avoid_chores_l637_637213


namespace find_k_in_geometric_sequence_l637_637851

theorem find_k_in_geometric_sequence (c k : ℝ) (h1_nonzero : c ≠ 0)
  (S : ℕ → ℝ) (a : ℕ → ℝ) (h2 : ∀ n, a (n + 1) = c * a n)
  (h3 : ∀ n, S n = 3^n + k)
  (h4 : a 1 = 3 + k)
  (h5 : a 2 = S 2 - S 1)
  (h6 : a 3 = S 3 - S 2) : k = -1 := by
  sorry

end find_k_in_geometric_sequence_l637_637851


namespace root_and_value_of_a_unique_root_condition_l637_637790

theorem root_and_value_of_a
  (a : ℝ) :
  (root_one_2 : (∃ (x : ℝ), (a - 1) * x ^ 2 + 2 * x + (a - 1) = 0) → 2 * (a - 1) + 4 + a - 1 = 0 →
   a = (1/5) ∧ ∃ (x : ℝ), (a - 1) * x ^ 2 + 2 * x + (a - 1) = 0 ∧ x ≠ 2 ∧ x = (1/2)) := sorry

theorem unique_root_condition
  (a : ℝ) :
  (root_duplicate : ∃! (x : ℝ), (a - 1) * x ^ 2 + 2 * x + (a - 1) = 0 →
   (a = 1 ∧ ∃! (x : ℝ), 2 * x = 0 ∧ x = 0) ∨
   (a = 2 ∧ ∃! (x : ℝ), x ^ 2 + 2 * x + 1 = 0 ∧ x = -1) ∨
   (a = 0 ∧ ∃! (x : ℝ), -x ^ 2 + 2 * x - 1 = 0 ∧ x = 1)) := sorry

end root_and_value_of_a_unique_root_condition_l637_637790


namespace eval_fraction_expression_l637_637030
noncomputable def inner_expr := 2 + 2
noncomputable def middle_expr := 2 + (1 / inner_expr)
noncomputable def outer_expr := 2 + (1 / middle_expr)

theorem eval_fraction_expression : outer_expr = 22 / 9 := by
  sorry

end eval_fraction_expression_l637_637030


namespace arithmetic_seq_a7_geometric_seq_b6_geometric_common_ratio_l637_637772

noncomputable def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def geometric_seq (b₁ q : ℕ) (n : ℕ) : ℕ :=
  b₁ * q^(n - 1)

theorem arithmetic_seq_a7 
  (a₁ d : ℕ)
  (h : 2 * arithmetic_seq a₁ d 5 - arithmetic_seq a₁ d 3 = 3) :
  arithmetic_seq a₁ d 7 = 3 :=
sorry

theorem geometric_seq_b6 
  (b₁ q : ℕ)
  (h1 : geometric_seq b₁ q 2 = 1)
  (h2 : geometric_seq b₁ q 4 = 4) :
  geometric_seq b₁ q 6 = 16 :=
sorry

theorem geometric_common_ratio 
  (b₁ q : ℕ)
  (h1 : geometric_seq b₁ q 2 = 1)
  (h2 : geometric_seq b₁ q 4 = 4) :
  q = 2 ∨ q = -2 :=
sorry

end arithmetic_seq_a7_geometric_seq_b6_geometric_common_ratio_l637_637772


namespace sets_complement_operations_l637_637197

universe u

def U := ℝ

def A := { x : ℝ | x^2 - x - 6 < 0 }

def B := { x : ℝ | ∃ y ∈ A, |x| = y + 2 }

def complement (U : Type u) (S : Set U) : Set U := { x | x ∉ S }

theorem sets_complement_operations :
  (complement U B = (Set.Iic (-5)) ∪ {0} ∪ (Set.Ici 5)) ∧
  (A ∩ B = (-2:ℝ, 0) ∪ (0, 3)) ∧
  (A ∪ B = (-5:ℝ, 5)) ∧
  (A ∪ (complement U B) = (Set.Iic 5) ∪ (-2:ℝ, 3) ∪ (Set.Ici 5)) ∧
  (A ∩ (complement U B) = {0}) ∧
  (complement U (A ∪ B) = (Set.Iic 5) ∪ (Set.Ici 5)) ∧
  (complement U A ∩ complement U B = (Set.Iic 5) ∪ (Set.Ici 5)) :=
by 
  sorry

end sets_complement_operations_l637_637197


namespace rent_minutes_to_equivalent_year_l637_637258

def minutes_to_years (m : ℕ) : ℝ :=
  m / 60 / 24 / 365

theorem rent_minutes_to_equivalent_year :
  minutes_to_years 525600 = 1 :=
by
  sorry

end rent_minutes_to_equivalent_year_l637_637258


namespace M_plus_m_eq_neg2_l637_637170

def y (x : ℝ) : ℝ := (1 / 3) * Real.cos x - 1
def M : ℝ := -2 / 3
def m : ℝ := -4 / 3

theorem M_plus_m_eq_neg2 : M + m = -2 := by
  -- Proof goes here
  sorry

end M_plus_m_eq_neg2_l637_637170


namespace number_of_days_A_left_l637_637996

noncomputable def work_problem (W : ℝ) : Prop :=
  let A_rate := W / 45
  let B_rate := W / 40
  let days_B_alone := 23
  ∃ x : ℝ, x * (A_rate + B_rate) + days_B_alone * B_rate = W ∧ x = 9

theorem number_of_days_A_left (W : ℝ) : work_problem W :=
  sorry

end number_of_days_A_left_l637_637996


namespace sandwich_cost_l637_637279

theorem sandwich_cost (S : ℝ) (h : 2 * S + 4 * 0.87 = 8.36) : S = 2.44 :=
by sorry

end sandwich_cost_l637_637279


namespace time_to_mow_lawn_l637_637892

noncomputable def width_of_lawn : ℝ := 100  -- feet
noncomputable def length_of_lawn : ℝ := 180 -- feet
noncomputable def swath_width : ℝ := 30 / 12 -- inches to feet
noncomputable def overlap : ℝ := 6 / 12 -- inches to feet
noncomputable def effective_swath_width : ℝ := swath_width - overlap
noncomputable def moe_speed : ℝ := 4500 -- feet per hour

theorem time_to_mow_lawn : 
  let total_distance := (length_of_lawn / effective_swath_width) * width_of_lawn in
  total_distance / moe_speed = 2 :=
by 
  let total_distance := (length_of_lawn / effective_swath_width) * width_of_lawn
  have h : total_distance = 9000 := by sorry -- proof is skipped
  have hs : moe_speed = 4500 := by sorry -- proof is skipped
  show total_distance / moe_speed = 2 from 
  calc
    total_distance / moe_speed = 9000 / 4500 : by rw [h, hs]
                       ...     = 2           : by norm_num

end time_to_mow_lawn_l637_637892


namespace triangle_similarity_proof_l637_637832

/-- In triangle ABC, with FG parallel to AB, CF = 5 cm, FA = 15 cm, and CG = 9 cm,
prove that CB = 36 cm. -/
theorem triangle_similarity_proof (A B C F G: Point)
  (h_parallel: Line.parallel (line_f f f' '('g g) (line_a a a' '('b b)))
  (h_CF: distance C F = 5)
  (h_FA: distance F A = 15)
  (h_CG: distance C G = 9) :
  distance C B = 36 :=
sorry


end triangle_similarity_proof_l637_637832


namespace sum_of_coords_of_circle_center_l637_637044

theorem sum_of_coords_of_circle_center (x y : ℝ) :
  (x^2 + y^2 = 4 * x - 6 * y + 9) → x + y = -1 :=
by
  sorry

end sum_of_coords_of_circle_center_l637_637044


namespace Aryan_owes_1200_l637_637361

variables (A K : ℝ) -- A represents Aryan's debt, K represents Kyro's debt

-- Condition 1: Aryan's debt is twice Kyro's debt
axiom condition1 : A = 2 * K

-- Condition 2: Aryan pays 60% of her debt
axiom condition2 : (0.60 * A) + (0.80 * K) = 1500 - 300

theorem Aryan_owes_1200 : A = 1200 :=
by
  sorry

end Aryan_owes_1200_l637_637361


namespace fraction_of_percent_l637_637296

theorem fraction_of_percent (h : (1 / 8 * (1 / 100)) * 800 = 1) : true :=
by
  trivial

end fraction_of_percent_l637_637296


namespace coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45_l637_637242

theorem coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45 :
  let general_term (r : ℕ) := (Nat.choose 10 r) * (x^(10 - 3 * r)/2)
  ∃ r : ℕ, (general_term r) = 2 ∧ (Nat.choose 10 r) = 45 :=
by
  sorry

end coeff_x2_expansion_sqrt_x_plus_1_over_x_pow_10_eq_45_l637_637242


namespace sum_first_nine_terms_l637_637506

variable (a : ℕ → ℝ)   -- Define the arithmetic sequence a
variable (d : ℝ)       -- Common difference of the sequence

-- Assume the given condition
def condition := a 8 = (1/2) * a 11 + 6

-- Prove that the sum of the first 9 terms is 108
theorem sum_first_nine_terms (h : ∀ n, a (n + 1) = a n + d) (h8 : condition a) : (∑ i in Finset.range 9, a i) = 108 :=
sorry

end sum_first_nine_terms_l637_637506


namespace correct_answer_l637_637899

-- Definitions and Conditions
structure Line := (a : Type) -- You would define this more rigorously in a real proof
structure Plane := (α : Type) -- You would define this more rigorously in a real proof
structure Polyhedron := (parallel_faces : List Plane) (trapezoidal_faces : List Plane) -- Again, for thoroughness

variables (a : Line) (α β : Plane)
variable (polyhedron : Polyhedron)

-- Conditions
axiom line_in_plane : a ∈ α
axiom line_perpendicular_plane : a ⊥ β
axiom polyhedron_properties : polyhedron.parallel_faces.length = 2 ∧ (∀ face ∈ polyhedron.trapezoidal_faces, is_trapezoid face)

-- Propositions
def p := a ∈ α ∧ a ⊥ β → α ⊥ β
def q := ¬(polyhedron.parallel_faces.length = 2 ∧ (∀ face ∈ polyhedron.trapezoidal_faces, is_trapezoid face) → is_prism polyhedron)

-- Correct Answer
theorem correct_answer : (p a α β) ∧ (¬ q polyhedron) := by
  sorry

end correct_answer_l637_637899


namespace collinear_AXO_l637_637884

-- Let ABC be a triangle inscribed in a circle Ω with center O.
variables {A B C O : Type*} [geometry A B C O]
  (triangle_ABC : is_triangle A B C)
  (circle_omega : is_circumscribed O triangle_ABC)

-- A circle Γ with center A intersects the side BC at points D and E such that D lies between B and E.
variables {D E : Type*} (circle_gamma : is_centered A)
  (DE_on_BC : intersects_BC_at D E)
  (D_between_B_and_E : lies_between D B E)

-- F and G are the common points of Γ and Ω.
variables {F G : Type*} (FG_common : are_common_points_of gamma omega)

-- F lies on the arc AB of Ω not containing C, G lies on the arc AC of Ω not containing B.
variables (F_on_arc_AB_not_C : lies_on_arc_not_containing F A B C)
  (G_on_arc_AC_not_B : lies_on_arc_not_containing G A C B)

-- The circumcircles of triangles BDF and CEG intersect the sides AB and AC again at K and L, respectively.
variables {BDF CEG : Type*} (circumcircle_BDF : is_circumcircle_of BDF (D, F))
  (circumcircle_CEG : is_circumcircle_of CEG (E, G))
  (K L : Type*) (K_on_AB : lies_on K A B)
  (L_on_AC : lies_on L A C)

-- The lines FK and GL intersect at X.
variables {X : Type*} (FK : intersects_at_line F K)
  (GL : intersects_at_line G L)
  (intersection_X : intersects_at FK GL X)

-- Prove that A, X, and O are collinear.
theorem collinear_AXO : are_collinear A X O :=
sorry

end collinear_AXO_l637_637884


namespace prime_quadruples_l637_637017

theorem prime_quadruples :
  ∀ (p q r : ℕ) (n : ℕ),
    prime p ∧ prime q ∧ prime r ∧ n > 0 ∧ p^2 = q^2 + r^n →
    (p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨
    (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4) :=
by
  intros p q r n hc
  obtain ⟨hp, hq, hr, hn, heq⟩ := hc
  -- Proof to be completed
  sorry

end prime_quadruples_l637_637017


namespace median_hypotenuse_right_triangle_l637_637839

/-- Prove that in a right triangle with legs of lengths 5 and 12,
  the median on the hypotenuse can be either 6 or 6.5. -/
theorem median_hypotenuse_right_triangle (a b : ℝ) (ha : a = 5) (hb : b = 12) :
  ∃ c : ℝ, (c = 6 ∨ c = 6.5) :=
sorry

end median_hypotenuse_right_triangle_l637_637839


namespace number_of_triangles_l637_637496

-- Define our context and conditions
open_locale classical

noncomputable def round_robin_tournament (n : ℕ) :=
∀ i j : fin n, i ≠ j → (W_i + L_j ≥ 10)

-- Given number of players
def num_players : ℕ := 312

-- Conditions for losses and wins for each player
def L : fin num_players → ℕ := sorry
def W : fin num_players → ℕ := sorry

-- Definition of a triangle
def is_triangle (i j k : fin num_players) :=
defeats i j ∧ defeats j k ∧ defeats k i

-- Prove the number of triangles
theorem number_of_triangles :
  ∃ n_t : ℕ, n_t = 70 :=
begin
  sorry
end

end number_of_triangles_l637_637496


namespace equal_division_of_triangle_l637_637382

variables {A B C G : Type*}
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace G]
  [Triangle ABC] [Centroid G ABC]

theorem equal_division_of_triangle (A B C G : Point) (hA : IsVertex A ABC)
  (hB : IsVertex B ABC) (hC : IsVertex C ABC) (hG : IsCentroid G ABC) :
  Congruent (Triangle.mk A G B) (Triangle.mk B G C) ∧ Congruent (Triangle.mk B G C) (Triangle.mk C G A) := 
sorry

end equal_division_of_triangle_l637_637382


namespace max_pairs_of_gnomes_friends_l637_637140

theorem max_pairs_of_gnomes_friends :
  let n := 1000 in 
  let pairs := n^3 / 4 in
  ∀ towers stories : Fin n, 
  (∀ (tower : Fin n) (story : Fin n), ∃ gnome_hats : Fin n, 
    (gnome_hats tower story < n)) ∧
  (∀ tower : Fin n, 
    ∀ (i j : Fin n), i ≠ j → gnome_hats tower i ≠ gnome_hats tower j) →
  (∀ gnome1 tower1 story1, 
    ∀ gnome2 tower2 story2,
    gnome_hats tower1 story1 = gnome_hats tower2 story2 ∧
    abs (story1.val - story2.val) = 1 →
    (gnome1, gnome2) ∈ pairs) :=
by
  sorry

end max_pairs_of_gnomes_friends_l637_637140


namespace problem1_part1_1_problem1_part1_2_problem2_l637_637873

variable (a : ℕ) (x y : ℕ)
variable {n : ℕ}
axiom a_nonzero : a ≠ 0
axiom x_pos : 0 < x
axiom y_pos : 0 < y
def op (a : ℕ) (x : ℕ) := a ^ x

theorem problem1_part1_1 (h1 : op 2 n = 64) : n = 6 :=
sorry

theorem problem1_part1_2 (h2 : op n 2 = 64) : n = 8 ∨ n = -8 :=
sorry

theorem problem2 (h1 : op 2 x * op 2 (2 * y) = 8) (h2 : op 3 x * op 3 y = 9) : x = 1 ∧ y = 1 :=
sorry

end problem1_part1_1_problem1_part1_2_problem2_l637_637873


namespace secant_lines_circle_l637_637654

theorem secant_lines_circle
  (O P A B C D E F : Point)
  (circ : Circle)
  (h1 : PAB ∈ Line)
  (h2 : Line.passes_through PAB O)
  (h3: PAB.intersects_circle circ)
  (h4 : PCD ∈ Line)
  (h5 : Line.passes_through PCD P)
  (h6 : PCD.intersects_circle_at circ C D)
  (h7 : E ∈ circ)
  (h8 : Arc.equal circ B E B D)
  (h9 : CE ∈ Line)
  (h10 : CE.intersects_line_at AB F)
  (AB BP BF : ℝ) :
  2 / AB = 1 / BP + 1 / BF := by
  sorry

end secant_lines_circle_l637_637654


namespace product_area_perimeter_l637_637894

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * Real.abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

noncomputable def perimeter (P Q R : ℝ × ℝ) :=
  distance P Q + distance Q R + distance R P

theorem product_area_perimeter (P Q R : ℝ × ℝ)
  (hP : P = (0, 1))
  (hQ : Q = (3, 4))
  (hR : R = (4, 1)) :
  area P Q R * perimeter P Q R = (1.5 * Real.sqrt 2 + 2 + 0.5 * Real.sqrt 10) :=
by
  sorry

end product_area_perimeter_l637_637894


namespace find_a_plus_2b_l637_637793

-- Define the function and conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def quadratic_function (a b c : ℝ) : ℝ → ℝ :=
  λ x, a * x^2 + b * x + c

-- Hypotheses
variables (a b c : ℝ)
hypothesis (h_even : is_even_function (quadratic_function a b c))
hypothesis (h_interval : -2 * a - 5 ≤ 0 ∧ 0 ≤ 1)

-- Problem Statement
theorem find_a_plus_2b : a + 2 * b = -2 :=
sorry

end find_a_plus_2b_l637_637793


namespace roots_of_f_l637_637886

noncomputable def f (a x : ℝ) : ℝ := x - Real.log (a * x)

theorem roots_of_f (a : ℝ) :
  (a < 0 → ¬∃ x : ℝ, f a x = 0) ∧
  (0 < a ∧ a < Real.exp 1 → ∃! x : ℝ, f a x = 0) ∧
  (a = Real.exp 1 → ∃! x : ℝ, f a x = 0) ∧
  (a > Real.exp 1 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :=
sorry

end roots_of_f_l637_637886


namespace ln_abs_x_minus_a_even_iff_a_zero_l637_637162

theorem ln_abs_x_minus_a_even_iff_a_zero (a : ℝ) : 
  (∀ x : ℝ, Real.log (|x - a|) = Real.log (|(-x) - a|)) ↔ a = 0 :=
sorry

end ln_abs_x_minus_a_even_iff_a_zero_l637_637162


namespace part1_part2_l637_637426

theorem part1 {a : ℕ → ℝ} (S : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n, S (n + 1) = 4 * a n + 2) :
  ∃ r, ∀ n, a (n + 1) - 2 * a n = r * (a n - 2 * a (n - 1)) := sorry

theorem part2 {a : ℕ → ℝ} (S : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n, S (n + 1) = 4 * a n + 2)
  (b : ℕ → ℝ) (hb : ∀ n, b n = a (n + 1) - 2 * a n) :
  ∀ n, (∑ i in range n, i * b i) = (3 * n - 3) * 2 ^ n + 3 := sorry

end part1_part2_l637_637426


namespace smallest_n_1999n_congruent_2001_mod_10000_l637_637043

theorem smallest_n_1999n_congruent_2001_mod_10000 : ∃ n : ℕ, n > 0 ∧ (1999 * n) % 10000 = 2001 ∧ ∀ m : ℕ, m > 0 ∧ 1999 * m % 10000 = 2001 → n ≤ m := by
  use 5999
  split
  - exact Nat.succ_pos'
  split
  - norm_num
  sorry

end smallest_n_1999n_congruent_2001_mod_10000_l637_637043


namespace fifteenth_digit_sum_l637_637971

/-- The 15th digit after the decimal point of the sum of decimal equivalents of 1/9 and 1/11 is 1. -/
theorem fifteenth_digit_sum (d1 d2 : Nat) (h1 : (1/9 : Rat) = 0.1111111 -- overline 1 represents repeating 1
                    h2 : (1/11 : Rat) = 0.090909) -- overline 090909 represents repeating 090909
                   (repeating_block : String := "10")
                    : repeating_block[15 % 2] = '1' := -- finding the 15th digit
by
  sorry

end fifteenth_digit_sum_l637_637971


namespace translated_function_symmetry_center_l637_637286

theorem translated_function_symmetry_center :
  let f := fun x : ℝ => Real.sin (6 * x + π / 4)
  let g := fun x : ℝ => f (x / 3)
  let h := fun x : ℝ => g (x - π / 8)
  h π / 2 = 0 :=
by
  sorry

end translated_function_symmetry_center_l637_637286


namespace max_min_sum_of_transformed_cosine_l637_637173

theorem max_min_sum_of_transformed_cosine :
  let y (x : ℝ) := (1 / 3) * Real.cos x - 1 in
  (∀ x, y x ≤ -2 / 3) ∧ (∀ x, -4 / 3 ≤ y x) →
  (∃ M m : ℝ, M = -2 / 3 ∧ m = -4 / 3 ∧ (M + m) = -2) :=
by
  intro y_bounds
  -- Assuming the bounds on y are provided
  sorry

end max_min_sum_of_transformed_cosine_l637_637173


namespace sum_of_digits_least_N_l637_637536

-- Define the function P(N)
def P (N : ℕ) : ℚ := (Nat.ceil (3 * N / 5 + 1) : ℕ) / (N + 1)

-- Define the predicate that checks if P(N) is less than 321/400
def P_lt_321_over_400 (N : ℕ) : Prop := P N < (321 / 400 : ℚ)

-- Define a function that sums the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- The main statement: we claim the least multiple of 5 satisfying the condition
-- That the sum of its digits is 12
theorem sum_of_digits_least_N :
  ∃ N : ℕ, 
    (N % 5 = 0) ∧ 
    P_lt_321_over_400 N ∧ 
    (∀ N' : ℕ, (N' % 5 = 0) → P_lt_321_over_400 N' → N' ≥ N) ∧ 
    sum_of_digits N = 12 := 
sorry

end sum_of_digits_least_N_l637_637536


namespace possible_divisors_count_l637_637127
noncomputable theory

theorem possible_divisors_count (x : ℕ) (d : ℕ) (n : ℕ) (h1 : x = n ^ 4)
  (h2 : d = (∏ (p : ℕ) in (finset.range (n+1)).filter pnat.prime, (4 * (nat.factors n).count p + 1))) :
  d = 245 :=
by
  sorry

end possible_divisors_count_l637_637127


namespace find_y_when_vectors_perpendicular_l637_637802

theorem find_y_when_vectors_perpendicular :
  let a := (2 : ℤ, 4 : ℤ)
  let b := (-4 : ℤ, y : ℤ)
  (a.1 * b.1 + a.2 * b.2 = 0) → y = 2 :=
by
  sorry

end find_y_when_vectors_perpendicular_l637_637802


namespace find_m_l637_637695

theorem find_m (m : ℝ) (h1 : m^2 - 3 * m + 2 = 0) (h2 : m ≠ 1) : m = 2 :=
sorry

end find_m_l637_637695


namespace sin_x_eq_l637_637122

noncomputable def find_sin_x (c d : ℝ) (x : ℝ) : ℝ :=
  if h : c > d ∧ d > 0 ∧ 0 < x ∧ x < π / 2 ∧ tan x = 3 * c * d / (c^2 - d^2)
  then 3 * c * d / real.sqrt (c^4 + 7 * c^2 * d^2 + d^4)
  else sorry

theorem sin_x_eq :
  ∀ (c d x : ℝ),
    c > d → d > 0 → 0 < x → x < π / 2 →
    tan x = 3 * c * d / (c^2 - d^2) →
    sin x = 3 * c * d / real.sqrt (c^4 + 7 * c^2 * d^4 + d^4) :=
by {
  intros c d x hc hd hx1 hx2 htan,
  sorry
}

end sin_x_eq_l637_637122


namespace find_c_l637_637818

theorem find_c (c : ℝ) (h : ∃ a : ℝ, x^2 - 50 * x + c = (x - a)^2) : c = 625 :=
  by
  sorry

end find_c_l637_637818


namespace range_of_a_l637_637593

theorem range_of_a (a : ℝ) :
  (∃ f : ℝ → ℝ, f = (λ x, x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1) ∧ 
  (∃ f' : ℝ → ℝ, f' = (λ x, 3 * x^2 + 6 * a * x + 3 * (a + 2)) ∧ 
  (∃ Δ : ℝ, Δ = 36 * a^2 - 36 * (a + 2) ∧ Δ > 0))) ↔ (a < -1 ∨ a > 2) :=
sorry

end range_of_a_l637_637593


namespace travel_options_l637_637954

theorem travel_options (trains : ℕ) (ferries : ℕ) (h_trains : trains = 3) (h_ferries : ferries = 2) :
  trains * ferries = 6 :=
by
  rw [h_trains, h_ferries]
  norm_num
  sorry

end travel_options_l637_637954


namespace John_profit_is_1500_l637_637519

-- Defining the conditions
def P_initial : ℕ := 8
def Puppies_given_away : ℕ := P_initial / 2
def Puppies_kept : ℕ := 1
def Price_per_puppy : ℕ := 600
def Payment_stud_owner : ℕ := 300

-- Define the number of puppies John's selling
def Puppies_selling := Puppies_given_away - Puppies_kept

-- Define the total revenue from selling the puppies
def Total_revenue := Puppies_selling * Price_per_puppy

-- Define John’s profit 
def John_profit := Total_revenue - Payment_stud_owner

-- The statement to prove
theorem John_profit_is_1500 : John_profit = 1500 := by
  sorry

end John_profit_is_1500_l637_637519


namespace vector_addition_l637_637803

/-- Define the vectors a and b and their orthogonality -/
variables (x : ℝ) (a b : ℝ × ℝ) (a_perp_b : x * 1 + 1 * (-2) = 0)

#check x = 2 -- mathlib has a function for elementwise addition of tuples

-- The main theorem
theorem vector_addition (h : x * 1 + 1 * (-2) = 0) : 
  let a := (x, 1)
  let b := (1, -2)
  in a + b = (3, -1) := 
by
  sorry  -- proof

end vector_addition_l637_637803


namespace midpoint_of_segment_l637_637021

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoint_of_segment :
  midpoint (12, -15) (-8, 27) = (2, 6) :=
by 
  sorry

end midpoint_of_segment_l637_637021


namespace greatest_multiple_of_5_and_6_less_than_800_l637_637979

theorem greatest_multiple_of_5_and_6_less_than_800 : 
  ∃ n : ℕ, n < 800 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 800 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
sorry

end greatest_multiple_of_5_and_6_less_than_800_l637_637979


namespace find_OB_coordinates_l637_637112

noncomputable theory

open Real

structure Point (α : Type _) :=
  (x : α)
  (y : α)

def is_perpendicular (a b : Point ℝ) : Prop :=
  a.x * b.x + a.y * b.y = 0

def magnitude (p : Point ℝ) : ℝ :=
  sqrt (p.x ^ 2 + p.y ^ 2)

theorem find_OB_coordinates :
  ∃ (x y : ℝ),
    let a := Point.mk 2 (-3)
    let A := Point.mk (-3) 2
    let B := Point.mk x y
    let AB := Point.mk (B.x + 3) (B.y - 2)
    is_perpendicular a AB ∧ magnitude AB = 3 * sqrt 13 :=
begin
  -- the proof will go here
  sorry
end

end find_OB_coordinates_l637_637112


namespace trapezoid_length_ab_l637_637287

noncomputable def length_CD : ℝ := 39
noncomputable def length_BC : ℝ := 39
noncomputable def OQ : ℝ := 15
noncomputable def length_BD := Real.sqrt (length_BC ^ 2 + length_CD ^ 2)
noncomputable def CQ := length_CD / 2
noncomputable def DO := length_BD / 3
noncomputable def OQ_calc := DO - CQ

theorem trapezoid_length_ab :
  (length_BD = Real.sqrt 3042) →
  (OQ_calc = OQ) →
  ∃ (p q : ℕ), (∃ ab : ℝ, ab = p * Real.sqrt q ∧ q = 92) ∧ (p + q = 970) :=
by
  intros h1 h2
  use 95
  use 92
  have h3 : 95 * Real.sqrt 92 = p * Real.sqrt q := sorry
  exact ⟨h3, (by norm_num)⟩

end trapezoid_length_ab_l637_637287


namespace problem_solution_l637_637455

-- Define the parabola y^2 = -x
def parabola (x y : ℝ) : Prop := y^2 = -x

-- Define the line y = k(x + 1)
def line (k x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
def intersection_points (k : ℝ) (y1 y2 : ℝ) : Prop :=
  parabola (-y1^2) y1 ∧ line k (-y1^2) y1 ∧
  parabola (-y2^2) y2 ∧ line k (-y2^2) y2

-- Proof sketch
theorem problem_solution 
  (k : ℝ) (k_ne_zero : k ≠ 0) (y1 y2 : ℝ) 
  (h_intersections : intersection_points k y1 y2) :
  (let O : ℝ × ℝ := (0, 0),
       A : ℝ × ℝ := (-y1^2, y1),
       B : ℝ × ℝ := (-y2^2, y2) in
       (O.1 * A.1 + O.2 * A.2) * (O.1 * B.1 + O.2 * B.2) = 0) ∧
  (1/2 * abs ((-y1^2) * y2 - (-y2^2) * y1) = 5/4 →
  (l = 2 * x + 3 * y + 2 = 0 ∨ l = 2 * x - 3 * y + 2 = 0)) :=
begin
  sorry -- Proof goes here
end

end problem_solution_l637_637455


namespace union_of_A_and_B_l637_637532

def setA : Set ℝ := {x : ℝ | x > 1 / 2}
def setB : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem union_of_A_and_B : setA ∪ setB = {x : ℝ | -1 < x} :=
by
  sorry

end union_of_A_and_B_l637_637532


namespace triangle_area_l637_637975

theorem triangle_area (base height : ℕ) (h_base : base = 10) (h_height : height = 10) :
  (base * height) / 2 = 50 :=
by
  rw [h_base, h_height]
  exact rfl
  sorry

end triangle_area_l637_637975


namespace min_expression_value_2023_l637_637299

noncomputable def min_expr_val := ∀ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023

noncomputable def least_value : ℝ := 2023

theorem min_expression_value_2023 : min_expr_val ∧ (∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = least_value) := 
by sorry

end min_expression_value_2023_l637_637299


namespace divisible_iff_exists_t_l637_637570

theorem divisible_iff_exists_t (a b m α : ℤ) (h_coprime : Int.gcd a m = 1) (h_divisible : a * α + b ≡ 0 [ZMOD m]):
  ∀ x : ℤ, (a * x + b ≡ 0 [ZMOD m]) ↔ ∃ t : ℤ, x = α + m * t :=
sorry

end divisible_iff_exists_t_l637_637570


namespace complex_expression_is_minus_one_minus_i_l637_637551

noncomputable def complex_expression : ℂ := ((1:ℂ) + Complex.i)^3 / ((1:ℂ) - Complex.i)^2

theorem complex_expression_is_minus_one_minus_i : complex_expression = -1 - Complex.i :=
by
  sorry

end complex_expression_is_minus_one_minus_i_l637_637551


namespace circumscribed_sphere_surface_area_l637_637511

theorem circumscribed_sphere_surface_area (A₁ A B C : ℝ^3) 
  (h1 : AA₁ ∈ span ℝ (A - B) ∧ AA₁ ⟂ (span ℝ (A - B) ∪ span ℝ (A - C)))
  (h2 : (BC ∈ span ℝ (B - C) ∧ B - C ⟂ A₁ - B))
  (h3 : dist A A₁ = 2 ∧ dist A C = 2) :
  4 * π * (√2) ^ 2 = 8 * π :=
by
  sorry

end circumscribed_sphere_surface_area_l637_637511


namespace num_A_is_9_l637_637129

-- Define the total number of animals
def total_animals : ℕ := 17

-- Define the number of animal B
def num_B : ℕ := 8

-- Define the number of animal A
def num_A : ℕ := total_animals - num_B

-- Statement to prove
theorem num_A_is_9 : num_A = 9 :=
by
  sorry

end num_A_is_9_l637_637129


namespace bajazet_winning_strategy_l637_637693

-- Define the polynomial P with place holder coefficients a, b, c (assuming they are real numbers)
def P (a b c : ℝ) (x : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + 1

-- The statement that regardless of how Alcina plays, Bajazet can ensure that P has a real root.
theorem bajazet_winning_strategy :
  ∃ (a b c : ℝ), ∃ (x : ℝ), P a b c x = 0 :=
sorry

end bajazet_winning_strategy_l637_637693


namespace correct_option_l637_637303

theorem correct_option :
  (∀ x: ℝ, x^3 = -1 → ∀ y: ℝ, y = ∛(-1) → y ≠ 1) ∧
  (∀ x: ℝ, x = -9 → (x^2 = 81 ∧ ∀ y: ℝ, y = sqrt(81) → y ≠ -9)) ∧
  (∀ x: ℝ, x = 25 → (sqrt x = 5 ∧ sqrt x ≠ ±5)) ∧
  (∀ x: ℝ, x = 27 → (-∛ x = -3)) :=
by
  sorry

end correct_option_l637_637303


namespace number_of_common_tangents_l637_637502

noncomputable def circle1 : set (ℝ × ℝ) := {p | (p.1 + 3)^2 + (p.2 - 1)^2 = 4}
noncomputable def circle2 : set (ℝ × ℝ) := {p | (p.1 - 4)^2 + (p.2 - 5)^2 = 4}

def distance (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

def tangent_count (c1 c2 : set (ℝ × ℝ)) : ℕ :=
  if h : ∃ p, p ∈ c1 ∧ p ∈ c2 then 2 -- if the circles intersect at one point
  else if distance (-3, 1) (4, 5) > 4 then 4 -- if the circles are separate
  else if distance (-3, 1) (4, 5) < 4 then 0 -- if one circle is inside the other
  else 1 -- if the circles touch at exactly one point

theorem number_of_common_tangents : tangent_count circle1 circle2 = 4 := by
  sorry

end number_of_common_tangents_l637_637502


namespace pow_99_square_pow_neg8_mult_l637_637702

theorem pow_99_square :
  99^2 = 9801 := 
by
  -- Proof omitted
  sorry

theorem pow_neg8_mult :
  (-8) ^ 2009 * (-1/8) ^ 2008 = -8 :=
by
  -- Proof omitted
  sorry

end pow_99_square_pow_neg8_mult_l637_637702


namespace trig_identity_proof_l637_637371

theorem trig_identity_proof : 
  (let θ := 10 * Real.pi / 180 in
   (cos θ * (1 + Real.sqrt 3 * tan θ)) / cos (50 * Real.pi / 180) = 2 ) :=
by
  sorry 

end trig_identity_proof_l637_637371


namespace trig_identity_l637_637994

theorem trig_identity :
  cos (Real.pi / 15) - cos (2 * Real.pi / 15) - cos (4 * Real.pi / 15) + cos (7 * Real.pi / 15) = -1 / 2 :=
by sorry

end trig_identity_l637_637994


namespace last_integer_in_sequence_divided_by_2_l637_637944

theorem last_integer_in_sequence_divided_by_2 (a : ℕ → ℝ) (start : a 0 = 2000000) (hf : ∀ n, a (n + 1) = a n / 2) :
  ∃ n, a n = 15625 ∧ ∀ m > n, ¬(a m ∈ ℤ) := 
sorry

end last_integer_in_sequence_divided_by_2_l637_637944


namespace thirtieth_term_is_x_l637_637945

def contains_digit_one (n : ℕ) : Prop :=
  n.to_string.to_list.any (λ ch, ch = '1')

def valid_sequence_term (n : ℕ) : Prop :=
  (n % 15 = 0) ∧ contains_digit_one n

noncomputable def sequence_nth_term (n : ℕ) : ℕ :=
  @Nat.find (λ m, valid_sequence_term m) sorry

theorem thirtieth_term_is_x : sequence_nth_term 30 = x :=
sorry

end thirtieth_term_is_x_l637_637945


namespace fraction_arithmetic_l637_637370

theorem fraction_arithmetic :
  ((5 : ℚ) / 6 - (1 : ℚ) / 3) * (3 / 4) = 3 / 8 :=
by
  sorry

end fraction_arithmetic_l637_637370


namespace sum_all_possible_values_of_squares_l637_637923

noncomputable def complex_numbers_lemma (x y : ℂ) (hx : x + y = 1) (hy : x^20 + y^20 = 20) : Prop :=
  let a := 2 * x * y in
  let b := x^2 + y^2 in
  let polynomial_rhs := (2 * b - 1) := sorry -- This is equivalent to setting up the polynomial in b
  b = -90 -- The conclusion that the sum of all possible values of x^2 + y^2 is -90

theorem sum_all_possible_values_of_squares (x y : ℂ) (hx : x + y = 1) (hy : x^20 + y^20 = 20) : 
  ∑b, complex_numbers_lemma x y hx hy b = -90 := sorry

end sum_all_possible_values_of_squares_l637_637923


namespace angle_in_third_quadrant_l637_637663

-- Define the function that determines the quadrant of an angle in degrees
def quadrant (θ : ℝ) : ℕ :=
 if 0 < θ % 360 ∧ θ % 360 < 90 then 1
 else if 90 < θ % 360 ∧ θ % 360 < 180 then 2
 else if 180 < θ % 360 ∧ θ % 360 < 270 then 3
 else if 270 < θ % 360 ∧ θ % 360 < 360 then 4
 else if θ % 360 = 0 then 1
 else if θ % 360 = 90 then 2
 else if θ % 360 = 180 then 3
 else 4

-- Noncomputable due to use of real numbers
noncomputable def angle_of : ℝ := -510

theorem angle_in_third_quadrant : quadrant angle_of = 3 :=
by
  sorry

end angle_in_third_quadrant_l637_637663


namespace circle_tangent_x_axis_l637_637036

theorem circle_tangent_x_axis (x y : ℝ) (h_center : (x, y) = (-3, 4)) (h_tangent : y = 4) :
  ∃ r : ℝ, r = 4 ∧ (∀ x y, (x + 3)^2 + (y - 4)^2 = 16) :=
sorry

end circle_tangent_x_axis_l637_637036


namespace find_first_term_l637_637737

noncomputable def firstTermOfGeometricSeries (S : ℝ) (r : ℝ) : ℝ :=
  S * (1 - r) / (1 - r)

theorem find_first_term
  (S : ℝ)
  (r : ℝ)
  (hS : S = 20)
  (hr : r = -3/7) :
  firstTermOfGeometricSeries S r = 200 / 7 :=
  by
    rw [hS, hr]
    sorry

end find_first_term_l637_637737


namespace find_integers_satisfying_condition_l637_637714

-- Define the inequality condition
def condition (x : ℤ) : Prop := x * x < 3 * x

-- Prove that the set of integers satisfying the condition is {1, 2}
theorem find_integers_satisfying_condition :
  { x : ℤ | condition x } = {1, 2} := 
by {
  sorry
}

end find_integers_satisfying_condition_l637_637714


namespace area_of_region_T_is_approximately_1_034_l637_637575

noncomputable def area_of_region_T (side_length : ℝ) (angle_Q : ℝ) : ℝ :=
  let base := side_length / 2 in
  let height := side_length * Real.sin ((180 - angle_Q) / 2 * Real.pi / 180) in
    0.5 * base * height

theorem area_of_region_T_is_approximately_1_034 :
  area_of_region_T 4 150 ≈ 1.034 :=
by
  sorry

end area_of_region_T_is_approximately_1_034_l637_637575


namespace difference_in_sword_length_l637_637006

variables (c_length : ℕ) (j_length : ℕ) (jn_length : ℕ)

def christopher_sword_length : ℕ := 15

def jameson_sword_length (c_length : ℕ) : ℕ := 2 * c_length + 3

def june_sword_length (j_length : ℕ) : ℕ := j_length + 5

theorem difference_in_sword_length :
  let c_length := christopher_sword_length in
  let j_length := jameson_sword_length c_length in
  let jn_length := june_sword_length j_length in
  jn_length = c_length + 23 :=
by
  sorry

end difference_in_sword_length_l637_637006


namespace problem_1_problem_2_l637_637450

-- Condition for Question 1
def f (x : ℝ) (a : ℝ) := |x - a|

-- Proof Problem for Question 1
theorem problem_1 (a : ℝ) (h : a = 1) : {x : ℝ | f x a > 1/2 * (x + 1)} = {x | x > 3 ∨ x < 1/3} :=
sorry

-- Condition for Question 2
def g (x : ℝ) (a : ℝ) := |x - a| + |x - 2|

-- Proof Problem for Question 2
theorem problem_2 (a : ℝ) : (∃ x : ℝ, g x a ≤ 3) → (-1 ≤ a ∧ a ≤ 5) :=
sorry

end problem_1_problem_2_l637_637450


namespace never_return_to_start_l637_637423

variable {City : Type} [MetricSpace City]

-- Conditions
variable (C : ℕ → City)  -- C is the sequence of cities
variable (dist : City → City → ℝ)  -- distance function
variable (furthest : City → City)  -- function that maps each city to the furthest city from it
variable (start : City)  -- initial city

-- Assuming C satisfies the properties in the problem statement
axiom initial_city : C 1 = start
axiom furthest_city_step : ∀ n, C (n + 1) = furthest (C n)
axiom no_ambiguity : ∀ c1 c2, (dist c1 (furthest c1) > dist c1 c2 ↔ c2 ≠ furthest c1)

-- Define the problem to prove that if C₁ ≠ C₃, then ∀ n ≥ 4, Cₙ ≠ C₁
theorem never_return_to_start (h : C 1 ≠ C 3) : ∀ n ≥ 4, C n ≠ start := sorry

end never_return_to_start_l637_637423


namespace value_of_2022_plus_a_minus_b_l637_637438

theorem value_of_2022_plus_a_minus_b (x a b : ℚ) (h_distinct : x ≠ a ∧ x ≠ b ∧ a ≠ b) 
  (h_gt : a > b) (h_min : ∀ y : ℚ, |y - a| + |y - b| ≥ 2 ∧ |x - a| + |x - b| = 2) :
  2022 + a - b = 2024 := 
by 
  sorry

end value_of_2022_plus_a_minus_b_l637_637438


namespace find_sets_A_B_l637_637459

/-- Definitions of the elements of sets A and B -/
variable (a1 a2 a3 a4 : ℕ)
variable (A B : set ℕ)

/-- Conditions given in the problem -/
variable (h1 : A = {a1, a2, a3, a4})
variable (h2 : B = {a1^2, a2^2, a3^2, a4^2})
variable (h3 : a1 < a2 ∧ a2 < a3 ∧ a3 < a4)
variable (h4 : a1 ∈ ℕ ∧ a2 ∈ ℕ ∧ a3 ∈ ℕ ∧ a4 ∈ ℕ)
variable (h5 : A ∩ B = {a1, a4})
variable (h6 : a1 + a4 = 10)
variable (h7 : (A ∪ B).sum = 124)

theorem find_sets_A_B : 
  A = {1, 3, 4, 9} ∧ B = {1, 9, 16, 81} :=
by
  sorry

end find_sets_A_B_l637_637459


namespace no_exist_F_l637_637993

noncomputable def f (x y : ℝ) : ℝ × ℝ × ℝ :=
  (-y / (x^2 + 4 * y^2), x / (x^2 + 4 * y^2), 0)

theorem no_exist_F :
  ¬∃ (F : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ),
    (∀ x y z, (x, y, z) ≠ (0, 0, 0) → 
      continuous (λ t, (F t).1) ∧
      continuous (λ t, (F t).2) ∧
      continuous (λ t, (F t).3)) ∧
    (∀ x y z, (x, y, z) ≠ (0, 0, 0) → 
      (∂/∂x (F x y z) y - ∂/∂y (F x y z) x, 
       ∂/∂z (F x y z) x - ∂/∂x (F x y z) z, 
       ∂/∂y (F x y z) z - ∂/∂z (F x y z) y) = (0, 0, 0)) ∧
    (∀ x y, F (x, y, 0) = f x y) := sorry

end no_exist_F_l637_637993


namespace compute_S20_minus_2S10_l637_637053

noncomputable def a (n : ℕ) : ℕ :=
  if n = 2 then 1 else if n = 10 then 33 else sorry

noncomputable def S (n : ℕ) : ℕ :=
  (n / 2) * (2 * (-3) + (n - 1) * 4)

theorem compute_S20_minus_2S10 : S 20 - 2 * S 10 = 400 := by
  let a_2 := (a 2) -- 1
  let a_10 := (a 10) -- 33
  have h₁ : 8 * 4 = (a_10 - a_2)/(10 - 2) := by sorry
  have sum_terms : S 20 = 10 * (-6 + 19 * 4) := by sorry
  simp [S]
  have sum_terms2 : S 10 = 5 * (-6 + 9 * 4) := by sorry
  simp [S]
  exact sorry

end compute_S20_minus_2S10_l637_637053


namespace shoveling_takes_fifteen_hours_l637_637862

noncomputable def shoveling_time (initial_rate : ℕ) (dec_rate : ℕ) (total_volume : ℕ) : ℕ :=
  let rec go rate n remaining_volume :=
    if remaining_volume ≤ 0 then
      n
    else
      go (rate - dec_rate) (n + 1) (remaining_volume - rate)
  go initial_rate 0 total_volume

theorem shoveling_takes_fifteen_hours : 
  shoveling_time 30 2 (5 * 12 * 4) = 15 :=
by {
  sorry
}

end shoveling_takes_fifteen_hours_l637_637862


namespace solution_set_of_inequality_l637_637947

theorem solution_set_of_inequality (x : ℝ) : |5 * x - x^2| < 6 ↔ (-1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6) :=
sorry

end solution_set_of_inequality_l637_637947


namespace minimal_polynomial_degree_l637_637237

theorem minimal_polynomial_degree 
  (rational_polynomial : ∃ p : Polynomial ℚ, 
    p ≠ 0 ∧
    p.eval (2 - Real.sqrt 5) = 0 ∧
    p.eval (-2 - Real.sqrt 5) = 0 ∧
    p.eval (Real.sqrt 3) = 0 ∧
    p.eval (-Real.sqrt 3) = 0) :
  ∃ p : Polynomial ℚ, p ≠ 0 ∧ p.degree ≥ 6 :=
by
  sorry

end minimal_polynomial_degree_l637_637237


namespace angle_between_vectors_l637_637461

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Given conditions
def norm_a : ∥a∥ = 2 := sorry
def norm_b : ∥b∥ = 1 := sorry
def perp_condition : inner a b + ∥a∥^2 - 5/2 * inner b b = 0 := sorry

-- Proof problem stating
theorem angle_between_vectors : real.angle (a : V) b = real.pi / 3 :=
by {
  have h1 := perp_condition a b,
  sorry -- Proof to be completed
}

end angle_between_vectors_l637_637461


namespace students_doing_hula_hoops_l637_637916

def number_of_students_jumping_rope : ℕ := 7
def number_of_students_doing_hula_hoops : ℕ := 5 * number_of_students_jumping_rope

theorem students_doing_hula_hoops : number_of_students_doing_hula_hoops = 35 :=
by
  sorry

end students_doing_hula_hoops_l637_637916


namespace karen_final_score_l637_637526

theorem karen_final_score (n : ℕ) (a b : ℕ) (avg_graded_tests : ℕ) (max_score : ℕ) (base_bonus extra_bonus increase_per_point point_target total_tests : ℕ) :
  base_bonus = 500 →
  extra_bonus = 10 →
  avg_graded_tests = 70 →
  max_score = 150 →
  increase_per_point = 10 →
  point_target = 85 →
  a + b = 290 →
  n = 600 →
  total_tests = 10 →
  avg_graded_tests * 8 + a + b = point_target * total_tests
sorry

end karen_final_score_l637_637526


namespace speedster_convertibles_count_l637_637670

theorem speedster_convertibles_count (T : ℕ)
  (h1 : 3 * T / 4 isNat)
  (h2 : (T / 4 : ℕ) = 30)
  (h3 : ∃ (convertibles : ℕ), convertibles = 3 * (3 * T / 4) / 5) :
  ∃ (convertibles : ℕ), convertibles = 54 :=
by
  cases h3 with convertibles h3
  use convertibles
  rw [Nat.mul_div_cancel_left, Nat.mul_div_cancel_left] at h3
  sorry

end speedster_convertibles_count_l637_637670


namespace sum_result_l637_637033

noncomputable def complex_sum : Complex := 
  (Finset.range 1500).sum (λ k, (k+1) * Complex.I ^ (k+1))

theorem sum_result : complex_sum = -752 + 752 * Complex.I :=
  sorry

end sum_result_l637_637033


namespace delivery_in_april_l637_637951

theorem delivery_in_april (n_jan n_mar : ℕ) (growth_rate : ℝ) :
  n_jan = 100000 → n_mar = 121000 → (1 + growth_rate) ^ 2 = n_mar / n_jan →
  (n_mar * (1 + growth_rate) = 133100) :=
by
  intros n_jan_eq n_mar_eq growth_eq
  sorry

end delivery_in_april_l637_637951


namespace tan_increasing_intervals_l637_637598

open Real

theorem tan_increasing_intervals : ∀ k : ℤ, ∃ I : set ℝ, I = (set.Ioo (k * π / 2 - 3 * π / 8) (k * π / 2 + π / 8)) ∧
  (∀ x ∈ I, ∀ y ∈ I, x < y → tan (2 * x + π / 4) < tan (2 * y + π / 4)) :=
begin
  intros k,
  use (set.Ioo (k * π / 2 - 3 * π / 8) (k * π / 2 + π / 8)),
  split,
  { refl },
  { intros x hx y hy hxy,
    sorry,  -- Proof of monotonicity is omitted
  }
end

end tan_increasing_intervals_l637_637598


namespace comb_7_2_equals_21_l637_637414

theorem comb_7_2_equals_21 : (Nat.choose 7 2) = 21 := by
  sorry

end comb_7_2_equals_21_l637_637414


namespace radius_of_p_l637_637850

-- Define geometric entities and conditions
def radius_p (radius_a radius_b r : ℝ) : Prop :=
  let AB := radius_a + radius_b,
      AC := radius_b,
      BC := radius_a,
      AP := radius_a + r,
      BP := radius_b + r,
      CP := AB - r in
  3 * (AC * BC) + 3 * (CP ^ 2) = AC * (BP ^ 2) + BC * (AP ^ 2)

-- Prove the radius r is 6/7 under the given conditions:
theorem radius_of_p
  (radius_a radius_b r : ℝ) (h1 : radius_a = 2) 
  (h2 : radius_b = 1) (h3 : radius_p radius_a radius_b r) :
  r = 6 / 7 :=
by
  sorry

end radius_of_p_l637_637850


namespace smallest_int_a_for_inequality_l637_637131

theorem smallest_int_a_for_inequality (a : ℤ) : 
  (∀ x : ℝ, (0 < x ∧ x < Real.pi / 2) → 
  Real.exp x - x * Real.cos x + Real.cos x * Real.log (Real.cos x) + a * x^2 ≥ 1) → 
  a = 1 := 
sorry

end smallest_int_a_for_inequality_l637_637131


namespace ratio_of_segments_formed_by_cosine_intersections_l637_637018

theorem ratio_of_segments_formed_by_cosine_intersections (
  h1 : ∀ (x : ℝ), y = cos x, 
  h2 : y = 1/2,
  h3 : ∀ (p q: ℕ), p < q ∧ gcd p q = 1): 
  ∃ (p q: ℕ), (p, q) = (1, 2) :=
by
  sorry

end ratio_of_segments_formed_by_cosine_intersections_l637_637018


namespace Archer_catch_total_fish_l637_637357

noncomputable def ArcherFishProblem : ℕ :=
  let firstRound := 8
  let secondRound := firstRound + 12
  let thirdRound := secondRound + (secondRound * 60 / 100)
  firstRound + secondRound + thirdRound

theorem Archer_catch_total_fish : ArcherFishProblem = 60 := by
  sorry

end Archer_catch_total_fish_l637_637357


namespace root_property_odd_indices_l637_637077

theorem root_property_odd_indices (x : ℝ) (n : ℕ) : 
  Real.root (2*n + 1) x = - Real.root (2*n + 1) (-x) :=
sorry

end root_property_odd_indices_l637_637077


namespace june_vs_christopher_l637_637005

namespace SwordLength

def christopher_length : ℕ := 15
def jameson_length : ℕ := 3 + 2 * christopher_length
def june_length : ℕ := 5 + jameson_length

theorem june_vs_christopher : june_length - christopher_length = 23 := by
  show 5 + (3 + 2 * christopher_length) - christopher_length = 23
  sorry

end SwordLength

end june_vs_christopher_l637_637005


namespace range_of_m_l637_637093

noncomputable def f (m : ℝ) (x : ℝ) :=
  if x > 0 then m * x - Real.log x
  else m * x + Real.log (-x)

theorem range_of_m (m : ℝ) (x1 x2 : ℝ) (k : ℝ) 
  (h1 : ∀ x, f m x = if x > 0 then m * x - Real.log x else m * x + Real.log (-x))
  (h2 : Deriv f x1 = 0 ∧ Deriv f x2 = 0)
  (h3 : 0 < k ∧ k ≤ 2 * Real.exp 1)
  (h4 : k = (f m x2 - f m x1) / (x2 - x1)) :
  1 / Real.exp 1 < m ∧ m ≤ Real.exp 1 :=
sorry

end range_of_m_l637_637093


namespace probability_circle_l637_637563

theorem probability_circle (total_figures triangles circles squares : ℕ)
  (h_total : total_figures = 10)
  (h_triangles : triangles = 4)
  (h_circles : circles = 3)
  (h_squares : squares = 3) :
  circles / total_figures = 3 / 10 :=
by
  sorry

end probability_circle_l637_637563


namespace cost_price_percentage_of_marked_price_l637_637244

theorem cost_price_percentage_of_marked_price
  (MP : ℝ) -- Marked Price
  (CP : ℝ) -- Cost Price
  (discount_percent : ℝ) (gain_percent : ℝ)
  (H1 : CP = (x / 100) * MP) -- Cost Price is x percent of Marked Price
  (H2 : discount_percent = 13) -- Discount percentage
  (H3 : gain_percent = 55.35714285714286) -- Gain percentage
  : x = 56 :=
sorry

end cost_price_percentage_of_marked_price_l637_637244


namespace vector_subtraction_magnitude_l637_637421

theorem vector_subtraction_magnitude (a b : EuclideanSpace ℝ (Fin 2))
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : ‖a + b‖ = Real.sqrt 19) : 
  ‖a - b‖ = Real.sqrt 7 :=
sorry

end vector_subtraction_magnitude_l637_637421


namespace vector_problem_l637_637801

def vector_a := (1, 2) : ℝ × ℝ
def vector_b := (-3, 5) : ℝ × ℝ
def vector_c (x : ℝ) := (4, x) : ℝ × ℝ

theorem vector_problem (x λ : ℝ) (h₁ : λ * 4 = -2) (h₂ : λ * x = 7) : λ + x = -29 / 2 := by
  -- Proof goes here
  sorry

end vector_problem_l637_637801


namespace valid_pairs_count_l637_637120

open_locale classical

noncomputable def num_valid_pairs : ℕ :=
  finset.card (finset.filter (λ (p : ℤ × ℤ), p.1^2 + p.2^2 < 16
    ∧ p.1^2 + p.2^2 < 8 * p.1
    ∧ p.1^2 + p.2^2 < 8 * p.2)
    (finset.Icc (-3) 3).product (finset.Icc (-3) 3))

theorem valid_pairs_count : num_valid_pairs = 4 :=
sorry

end valid_pairs_count_l637_637120


namespace plane_of_triangle_ABC_passes_through_fixed_point_l637_637342

noncomputable def Sphere := { P : ℝ^3 | (P - center).norm = R }

variables (O : ℝ^3) (R : ℝ) (P : ℝ^3) 
variables {A B C : ℝ^3}
hypothesis hP_S : P ∈ Sphere O R
hypothesis hA_S : A ∈ Sphere O R
hypothesis hB_S : B ∈ Sphere O R
hypothesis hC_S : C ∈ Sphere O R
hypothesis hAngles : (∠APB = 90 ∧ ∠BPC = 90 ∧ ∠CPA = 90)

theorem plane_of_triangle_ABC_passes_through_fixed_point :
  ∃ (E : ℝ^3), E ∈ plane ABC :=
sorry

end plane_of_triangle_ABC_passes_through_fixed_point_l637_637342


namespace least_number_to_produce_multiple_of_112_l637_637652

theorem least_number_to_produce_multiple_of_112 : ∃ k : ℕ, 72 * k = 112 * m → k = 14 :=
by
  sorry

end least_number_to_produce_multiple_of_112_l637_637652


namespace complement_inter_empty_eq_univ_l637_637458

open Set

theorem complement_inter_empty_eq_univ :
  ∀ (U : Set ℝ) (A B : Set ℝ), 
    U = Univ → 
    A = { x | x ≥ 2 } → 
    B = { x | x < -1 } →
    complement U (A ∩ B) = Univ := 
by
  intros U A B hU hA hB
  sorry

end complement_inter_empty_eq_univ_l637_637458


namespace carla_needs_to_cook_3_steaks_l637_637374

theorem carla_needs_to_cook_3_steaks :
  ∀ (batch_waffles_time steak_time total_time : ℕ),
    batch_waffles_time = 10 →
    steak_time = 6 →
    total_time = 28 →
    let S := (total_time - batch_waffles_time) / steak_time in
    S = 3 :=
by
  intros batch_waffles_time steak_time total_time hb hs ht
  let S := (total_time - batch_waffles_time) / steak_time
  simp [hb, hs, ht]
  exact eq.refl _

end carla_needs_to_cook_3_steaks_l637_637374


namespace elder_age_is_twenty_l637_637238

-- Let e be the present age of the elder person
-- Let y be the present age of the younger person

def ages_diff_by_twelve (e y : ℕ) : Prop :=
  e = y + 12

def elder_five_years_ago (e y : ℕ) : Prop :=
  e - 5 = 5 * (y - 5)

theorem elder_age_is_twenty (e y : ℕ) (h1 : ages_diff_by_twelve e y) (h2 : elder_five_years_ago e y) :
  e = 20 :=
by
  sorry

end elder_age_is_twenty_l637_637238


namespace smallest_period_of_cosine_l637_637403

noncomputable def smallest_positive_period (f : ℝ → ℝ) : ℝ := 
  ∃ T > 0, ∀ x, f x = f (x + T)

theorem smallest_period_of_cosine :
  smallest_positive_period (λ x, Real.cos (2 * x - Real.pi / 6)) = Real.pi :=
by sorry

end smallest_period_of_cosine_l637_637403


namespace hyperbola_asymptotes_l637_637248

open Nat Int Rat Real

variables (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : (sqrt 3 : ℝ) = (Real.sqrt ((a^2 + b^2) / a^2)))

theorem hyperbola_asymptotes :
  (∀ (x y : ℝ), (x = 0) ∨ (x ≠ 0) ∧ (y = sqrt 2 * x) ∨ (y = -sqrt 2 * x)) :=
begin
    sorry
end

end hyperbola_asymptotes_l637_637248


namespace quadratic_two_distinct_real_roots_l637_637483

theorem quadratic_two_distinct_real_roots (k : ℝ) (h : k ≠ 0) : 
  (kx : ℝ) -> (a = k) -> (b = -2) -> (c = -1) -> (b^2 - 4*a*c > 0) -> (-2)^2 - 4*k*(-1) = (4 + 4*k > 0) := sorry

end quadratic_two_distinct_real_roots_l637_637483


namespace inequality_solution_l637_637386

   noncomputable def cubic_root (x : ℝ) : ℝ :=
     real.cbrt x

   theorem inequality_solution (x : ℝ) :
     cubic_root(x) + 3 / (cubic_root(x) + 4) ≤ 0 ↔ (-64 < x ∧ x ≤ -1) :=
   sorry
   
end inequality_solution_l637_637386


namespace electricity_usage_A_B_l637_637909

def electricity_cost (x : ℕ) : ℝ :=
  if h₁ : 0 ≤ x ∧ x ≤ 24 then 4.2 * x
  else if h₂ : 24 < x ∧ x ≤ 60 then 5.2 * x - 24
  else if h₃ : 60 < x ∧ x ≤ 100 then 6.6 * x - 108
  else if h₄ : 100 < x ∧ x ≤ 150 then 7.6 * x - 208
  else if h₅ : 150 < x ∧ x ≤ 250 then 8 * x - 268
  else 8.4 * x - 368

theorem electricity_usage_A_B (x : ℕ) (h : electricity_cost x = 486) :
  60 < x ∧ x ≤ 100 ∧ 5 * x = 450 ∧ 2 * x = 180 :=
by
  sorry

end electricity_usage_A_B_l637_637909


namespace greatest_multiple_of_5_and_6_less_than_800_l637_637978

theorem greatest_multiple_of_5_and_6_less_than_800 : 
  ∃ n : ℕ, n < 800 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m < 800 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n :=
sorry

end greatest_multiple_of_5_and_6_less_than_800_l637_637978


namespace find_number_l637_637610

def valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ((n % 10 = 6 ∧ ¬ (n % 7 = 0)) ∨ (¬ (n % 10 = 6) ∧ n % 7 = 0)) ∧
  ((n > 26 ∧ ¬ (n % 10 = 8)) ∨ (¬ (n > 26) ∧ n % 10 = 8)) ∧
  ((n % 13 = 0 ∧ ¬ (n < 27)) ∨ (¬ (n % 13 = 0) ∧ n < 27))

theorem find_number : ∃ n : ℕ, valid_number n ∧ n = 91 := by
  sorry

end find_number_l637_637610


namespace shaded_triangles_have_n_coins_l637_637362

theorem shaded_triangles_have_n_coins 
    (n : ℕ)
    (hexagon : RegularHexagon) -- Assume a definition for RegularHexagon
    (triangles : Finset Triangle) -- Finset of triangles within the hexagon
    (H_divided : triangles.card = 6 * n^2)
    (coins : Finset Triangle)
    (H_coins : coins.card = 2 * n)
    (H_layers : ∀ (t1 t2 ∈ coins), t1 ≠ t2 → ¬on_same_layer t1 t2) -- Assume a predicate definition for on_same_layer
    (shaded : Triangle → Prop) 
    (H_shaded : ∀ t ∈ triangles, t is shaded or t is not shaded) -- All triangles are either shaded or white
    :
    (coins.filter shaded).card = n := 
by
  sorry

end shaded_triangles_have_n_coins_l637_637362


namespace minimum_combines_to_reach_one_stack_l637_637351

noncomputable def combine_stacks (n : ℕ) (k : ℕ) : Prop :=
  ∃ combines : ℕ, combines = k ∧ (∃ stacks : List ℕ, stacks.length >= 1 ∧ stacks.contains (2^n))

theorem minimum_combines_to_reach_one_stack : ∀ n : ℕ, ∀ k : ℕ, (3 * 2^n > 0) → combine_stacks n 4 :=
  by for n => exists 4; sorry

end minimum_combines_to_reach_one_stack_l637_637351


namespace problem_f_of_3_l637_637092

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 + 1 else -2 * x + 3

theorem problem_f_of_3 : f (f 3) = 10 := by
  sorry

end problem_f_of_3_l637_637092


namespace range_of_m_to_satisfy_quadratic_l637_637123

def quadratic_positive_forall_m (m : ℝ) : Prop :=
  ∀ x : ℝ, m * x^2 + m * x + 100 > 0

theorem range_of_m_to_satisfy_quadratic :
  {m : ℝ | quadratic_positive_forall_m m} = {m : ℝ | 0 ≤ m ∧ m < 400} :=
by
  sorry

end range_of_m_to_satisfy_quadratic_l637_637123


namespace f_9_over_2_l637_637179

noncomputable def f (x : ℝ) : ℝ := sorry -- The function f(x) is to be defined later according to conditions

theorem f_9_over_2 :
  (∀ x : ℝ, f (x + 1) = -f (-x + 1)) ∧ -- f(x+1) is odd
  (∀ x : ℝ, f (x + 2) = f (-x + 2)) ∧ -- f(x+2) is even
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = -2 * x^2 + 2) ∧ -- f(x) = ax^2 + b, where a = -2 and b = 2
  (f 0 + f 3 = 6) → -- Sum f(0) and f(3)
  f (9 / 2) = 5 / 2 := 
by {
  sorry -- The proof is omitted as per the instruction
}

end f_9_over_2_l637_637179


namespace graph_through_origin_y_increases_with_x_not_pass_third_quadrant_l637_637757

-- Part 1: Prove that if the graph passes through the origin, then m ≠ 2/3 and n = 1
theorem graph_through_origin {m n : ℝ} : 
  (3 * m - 2 ≠ 0) → (1 - n = 0) ↔ (m ≠ 2/3 ∧ n = 1) :=
by sorry

-- Part 2: Prove that if y increases as x increases, then m > 2/3 and n is any real number
theorem y_increases_with_x {m n : ℝ} : 
  (3 * m - 2 > 0) ↔ (m > 2/3 ∧ ∀ n : ℝ, True) :=
by sorry

-- Part 3: Prove that if the graph does not pass through the third quadrant, then m < 2/3 and n ≤ 1
theorem not_pass_third_quadrant {m n : ℝ} : 
  (3 * m - 2 < 0) ∧ (1 - n ≥ 0) ↔ (m < 2/3 ∧ n ≤ 1) :=
by sorry

end graph_through_origin_y_increases_with_x_not_pass_third_quadrant_l637_637757


namespace number_of_sheep_l637_637314

-- Definitions based on conditions
def ratio_sheep_horses : ℕ → ℕ → Prop := λ sheep horse, sheep = horse
def horse_food_per_day (horse_food_ounce_per_day : ℕ) (horse_num : ℕ) : ℕ := horse_food_ounce_per_day * horse_num
def total_horse_food_per_day := 12880
def horse_food_per_horse_per_day := 230

-- Theorem to be proven
theorem number_of_sheep (sheep horse : ℕ) (H_ratio : ratio_sheep_horses sheep horse)
  (H_food : horse_food_per_day horse_food_per_horse_per_day horse = total_horse_food_per_day) : 
  sheep = 56 :=
by
  sorry

end number_of_sheep_l637_637314


namespace correct_statement_l637_637058

noncomputable def hyperbola_properties : Prop :=
  ∃ (b c : ℝ), b > 0 ∧ 
  (c = Real.sqrt(5)) ∧   -- From step 1, c = √(4 + 1) = √5
  (c / 2 = 1) ∧          -- The distance from F1 to the asymptote is 1
  (∃A B : (ℝ × ℝ), 
    (A = (Real.sqrt(6) - 2, 0)) ∧ 
    (B = (Real.sqrt(6) + 2, 0)) ∧ 
    (|A - F2| * |B - F2| = 2) ∧
    (1 / |A - F2| + 1 / |B - F2| = Real.sqrt(6) + 2))

theorem correct_statement : hyperbola_properties := sorry

end correct_statement_l637_637058


namespace two_functions_bisect_area_l637_637089

-- Definitions of the circle and functions
def circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def f1 (x : ℝ) : ℝ := x^3
def f2 (x : ℝ) : ℝ := Real.tan x
def f3 (x : ℝ) : ℝ := x * Real.sin x

-- Predicate to check if a function bisects the area of the circle
def bisects_area (f : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, circle x y ∧ f x = y ∧ f (-x) = -y

theorem two_functions_bisect_area :
  bisects_area f1 ∧ bisects_area f2 ∧ ¬ bisects_area f3 := by
  sorry

end two_functions_bisect_area_l637_637089


namespace fifteenth_digit_sum_l637_637966

theorem fifteenth_digit_sum (d₁ d₂ : ℕ → ℕ) 
  (h₁ : ∀ n, d₁ n = if n = 0 then 1 else 0) 
  (h₂ : ∀ n, d₂ n = if n % 2 = 0 then 0 else 9) :
  let sum_digit := λ n, (d₁ n + d₂ n) % 10 in
  sum_digit 14 = 1 :=
by 
  sorry

end fifteenth_digit_sum_l637_637966


namespace brick_height_l637_637045

variable {l w : ℕ} (SA : ℕ)

theorem brick_height (h : ℕ) (l_eq : l = 10) (w_eq : w = 4) (SA_eq : SA = 136) 
    (surface_area_eq : SA = 2 * (l * w + l * h + w * h)) : h = 2 :=
by
  sorry

end brick_height_l637_637045


namespace tangency_point_exists_l637_637799

variables {A B C D : Point}
variables {ABC ABD BCD CAD : Plane}
variables {H : Pyramid}
variables {G : Sphere}
variables {D_a D_b D_c : Point}

-- Assume the given conditions
def condition1 : H = Pyramid.mk A B C D := sorry
def condition2 : G.touches_face_internally H := sorry
def condition3 : D_a = rotate_point_around_edge_into_plane D B C A ABC := sorry
def condition4 : D_b = rotate_point_around_edge_into_plane D C A B ABC := sorry
def condition5 : D_c = rotate_point_around_edge_into_plane D A B C ABC := sorry

-- Theorem statement
theorem tangency_point_exists :
  G.tangency_point ABC = circumcenter D_a D_b D_c :=
sorry

end tangency_point_exists_l637_637799


namespace interval_of_monotonic_increase_l637_637019

open Real

noncomputable def f (x : ℝ) : ℝ := (x - 3) * exp x

theorem interval_of_monotonic_increase : 
  (∃ a ∈ ℝ, ∀ x ∈ Ioi (2 : ℝ), f(x) > f(a)) :=
by
  sorry

end interval_of_monotonic_increase_l637_637019


namespace find_f_2010_l637_637178

noncomputable def f (x : ℝ) : ℝ := Real.sin x

def f_n : ℕ → (ℝ → ℝ)
| 0       := f
| (n + 1) := fun x => (f_n n x)'  -- defining the derivative

theorem find_f_2010 : ∀ x : ℝ, f_n 2010 x = -Real.sin x :=
by {
  -- we start the theorem proof but conclude with sorry as instructed.
  sorry
}

end find_f_2010_l637_637178


namespace roots_pure_imaginary_for_negative_real_m_l637_637380

open Complex

theorem roots_pure_imaginary_for_negative_real_m (m : ℝ) (hm : m < 0) :
  ∀ z : ℂ, (8 * z^2 + 4 * complex.i * z - (m : ℂ) = 0) →
  (∃ a b : ℝ, a = 0 ∧ z = complex.i * b) :=
by
  sorry

end roots_pure_imaginary_for_negative_real_m_l637_637380


namespace gray_region_proof_l637_637008

noncomputable def gray_region_area
  (centerA : ℝ × ℝ)
  (radiusA : ℝ)
  (centerB : ℝ × ℝ)
  (radiusB : ℝ) : ℝ :=
  let sector_A := (1/4) * real.pi * radiusA^2 in
  let sector_B := (1/4) * real.pi * radiusB^2 in
  let d := real.sqrt ((centerB.1 - centerA.1)^2 + (centerB.2 - centerA.2)^2) in
  let intersection_area := 
    radiusA^2 * real.arccos((d^2 + radiusA^2 - radiusB^2) / (2 * d * radiusA)) +
    radiusB^2 * real.arccos((d^2 + radiusB^2 - radiusA^2) / (2 * d * radiusB)) -
    (1/2) * real.sqrt((-d + radiusA + radiusB) * (d + radiusA - radiusB) * (d - radiusA + radiusB) * (d + radiusA + radiusB)) in
  39 - sector_A - sector_B + (1/2) * intersection_area

theorem gray_region_proof :
  gray_region_area (5, 4) 4 (11, 4) 3 = 39 - 4 * real.pi - (9/4) * real.pi + (1/2) * (
    4^2 * real.arccos((6^2 + 4^2 - 3^2) / (2 * 6 * 4)) +
    3^2 * real.arccos((6^2 + 3^2 - 4^2) / (2 * 6 * 3)) -
    (1/2) * real.sqrt((-6 + 4 + 3) * (6 + 4 - 3) * (6 - 4 + 3) * (6 + 4 + 3))
  ) :=
by
  sorry

end gray_region_proof_l637_637008


namespace domain_f_no_parallel_lines_through_distinct_points_f_positive_on_interval_l637_637097

noncomputable def f (a b x : ℝ) := real.log (a^x - b^x)

variables {a b : ℝ}

-- a > 1 > b > 0
axiom (a_gt_one : a > 1) (one_gt_b : 1 > b) (b_gt_zero : b > 0)

-- The domain of f(x) is (0, +∞)
theorem domain_f : ∀ x, x ∈ set.Ioo (0 : ℝ) +∞ ↔ 0 < x := sorry

-- There do not exist two distinct points on the graph of f(x) such that the line passing through these points is parallel to the x-axis
theorem no_parallel_lines_through_distinct_points : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ -> x₁ > 0 -> x₂ > 0 -> f a b x₁ ≠ f a b x₂ := sorry

-- f(x) is always positive on the interval (1, +∞) if a - b ≥ 1
theorem f_positive_on_interval : (a - b ≥ 1) -> ∀ x : ℝ, 1 < x -> f a b x > 0 := sorry

end domain_f_no_parallel_lines_through_distinct_points_f_positive_on_interval_l637_637097


namespace total_jumps_l637_637230

-- Definitions based on given conditions
def Ronald_jumps : ℕ := 157
def Rupert_jumps : ℕ := Ronald_jumps + 86

-- The theorem we want to prove
theorem total_jumps : Ronald_jumps + Rupert_jumps = 400 :=
by
  sorry

end total_jumps_l637_637230


namespace abs_expression_evaluation_l637_637984

theorem abs_expression_evaluation : | -2 | * (| -25 | - | 5 |) = 40 := by
  sorry

end abs_expression_evaluation_l637_637984


namespace area_of_triangle_formed_by_line_and_axes_l637_637398

-- Definition of the line equation condition
def line_eq (x y : ℝ) : Prop := 2 * x - 5 * y - 10 = 0

-- Statement of the problem to prove
theorem area_of_triangle_formed_by_line_and_axes :
  (∃ x y : ℝ, line_eq x y ∧ x = 0 ∧ y = -2) ∧
  (∃ x y : ℝ, line_eq x y ∧ x = 5 ∧ y = 0) →
  let base : ℝ := 5
  let height : ℝ := 2
  let area := (1 / 2) * base * height
  area = 5 := 
by
  sorry

end area_of_triangle_formed_by_line_and_axes_l637_637398


namespace binomial_expansion_properties_l637_637397

theorem binomial_expansion_properties :
  (∑ k in Finset.range 10, Nat.choose 9 k) = 512 ∧
  (∃ r, r = 6 ∧ (∑ r in Finset.range 10, (-1)^r * (1/(2^r)) * (Nat.choose 9 r) * (x^2)^(9-r) * (x^(-r)) = 21/16)) := by
  sorry

end binomial_expansion_properties_l637_637397


namespace june_vs_christopher_l637_637004

namespace SwordLength

def christopher_length : ℕ := 15
def jameson_length : ℕ := 3 + 2 * christopher_length
def june_length : ℕ := 5 + jameson_length

theorem june_vs_christopher : june_length - christopher_length = 23 := by
  show 5 + (3 + 2 * christopher_length) - christopher_length = 23
  sorry

end SwordLength

end june_vs_christopher_l637_637004


namespace simplest_square_root_of_given_options_l637_637304

def is_simplest_square_root (x : ℝ) : Prop :=
  ∀ y : ℝ, is_sqrt(y, x) → x = y

theorem simplest_square_root_of_given_options :
  is_simplest_square_root (sqrt 10) :=
by
  sorry

end simplest_square_root_of_given_options_l637_637304


namespace lateral_surface_area_cylinder_l637_637597

-- Define the diameter and height
def diameter : ℝ := 4
def height : ℝ := 4

-- Circumference formula
def circumference (d : ℝ) : ℝ := real.pi * d

-- Lateral surface area formula
def lateral_surface_area (c : ℝ) (h : ℝ) : ℝ := c * h

-- The statement we want to prove
theorem lateral_surface_area_cylinder : 
  lateral_surface_area (circumference diameter) height = 16 * real.pi :=
by sorry

end lateral_surface_area_cylinder_l637_637597


namespace power_function_value_l637_637758

theorem power_function_value (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f(x) = x ^ a) → (f 2 = real.sqrt 2) → f (1/4) = 1/2 :=
by
  intros h_f h_point
  -- Add your proof here
  sorry

end power_function_value_l637_637758


namespace measure_ADC_is_110_l637_637934

variables (ABC_angle AD_bisects_BAC DC_bisects_BCA DB_bisects_BCD : Prop)

theorem measure_ADC_is_110
  (h_ABC_angle : ABC_angle = 40)
  (h_AD_bisects_BAC : AD_bisects_BAC)
  (h_DC_bisects_BCA : DC_bisects_BCA)
  (h_DB_bisects_BCD : DB_bisects_BCD) :
  ∃ ADC_angle, ADC_angle = 110 :=
by
  sorry

end measure_ADC_is_110_l637_637934


namespace area_of_region_R_l637_637910

-- Define the parameters of the rhombus
def Rhombus (ABCD : Type) := 
  (side_length : ℝ) 
  (angle_at_B : ℝ)
  (area_of_R : ℝ)

-- State the theorem
theorem area_of_region_R :
  ∀ (ABCD : Type) (side_length : ℝ) (angle_at_B : ℝ), 
  side_length = 2 ∧ angle_at_B = 120 → ∃ (area_of_R : ℝ), area_of_R = 2 * sqrt 3 / 3 :=
by
  intro ABCD side_length angle_at_B h,
  cases h with h_side h_angle,
  use 2 * sqrt ３ / 3,
  sorry

end area_of_region_R_l637_637910


namespace waitress_tips_fraction_l637_637308

theorem waitress_tips_fraction
  (S : ℝ) -- salary
  (T : ℝ) -- tips
  (hT : T = (11 / 4) * S) -- tips are 11/4 of salary
  (I : ℝ) -- total income
  (hI : I = S + T) -- total income is the sum of salary and tips
  : (T / I) = (11 / 15) := -- fraction of income from tips is 11/15
by
  sorry

end waitress_tips_fraction_l637_637308


namespace largest_coefficient_term_in_expansion_l637_637645

theorem largest_coefficient_term_in_expansion :
  ∃ (r : ℕ), r % 2 = 0 ∧ r ≤ 6 ∧ (binomial 7 r) = 35 ∧ (x:ℝ) ≠ 0 ∧ 
  (C(7,r) : ℝ) * x^(7 - 2*r) = 35 * (x^-1) :=
by
  sorry

end largest_coefficient_term_in_expansion_l637_637645


namespace problem_solution_l637_637879

noncomputable def complex_w : ℂ := complex.exp (complex.I * real.pi * (6 / 11))

theorem problem_solution :
  complex_w ≠ 1 ∧ complex_w^(11) = 1 → 
  (complex_w / (1 + complex_w^(2)) + 
   complex_w^(2) / (1 + complex_w^(4)) + 
   complex_w^(3) / (1 + complex_w^(6)) + 
   complex_w^(4) / (1 + complex_w^(8))) = -3 := 
by {
  sorry
}

end problem_solution_l637_637879


namespace exists_real_polynomial_p_l637_637189

-- Definition of f(n) which counts the number of ways of representing n
-- as a sum of powers of 2 with no power being used more than 3 times.
noncomputable def f (n : ℕ) : ℕ := sorry

-- The main theorem stating the existence of the polynomial p(x) = (x+2) / 2
theorem exists_real_polynomial_p (n : ℕ) : 
  f(n) = ⌊ (n + 2) / 2 ⌋ :=
sorry

end exists_real_polynomial_p_l637_637189


namespace chastity_lollipops_l637_637375

theorem chastity_lollipops (initial_money lollipop_cost gummy_cost left_money total_gummies total_spent lollipops : ℝ)
  (h1 : initial_money = 15)
  (h2 : lollipop_cost = 1.50)
  (h3 : gummy_cost = 2)
  (h4 : left_money = 5)
  (h5 : total_gummies = 2)
  (h6 : total_spent = initial_money - left_money)
  (h7 : total_spent = 10)
  (h8 : total_gummies * gummy_cost = 4)
  (h9 : total_spent - (total_gummies * gummy_cost) = 6)
  (h10 : lollipops = (total_spent - (total_gummies * gummy_cost)) / lollipop_cost) :
  lollipops = 4 := 
sorry

end chastity_lollipops_l637_637375


namespace ellipse_area_l637_637034

-- Given the ellipse equation 2x^2 + 8x + y^2 - 2y + 8 = 0
-- We need to prove the area of the ellipse is π√2 / 2

theorem ellipse_area (x y : ℝ) :
  2 * x ^ 2 + 8 * x + y ^ 2 - 2 * y + 8 = 0 → 
  ∃ (a b : ℝ), a = 1 / √2 ∧ b = 1 ∧ 
  (π * a * b) = (π * √2 / 2) := 
by
  sorry

end ellipse_area_l637_637034


namespace num_4digit_numbers_divisible_by_5_l637_637812

theorem num_4digit_numbers_divisible_by_5 : 
  (#{ n : ℕ | n ≥ 1000 ∧ n ≤ 9999 ∧ n % 5 = 0 }.finite.to_finset.card) = 1800 :=
by
  sorry

end num_4digit_numbers_divisible_by_5_l637_637812


namespace quadratic_trinomial_inequality_l637_637087

theorem quadratic_trinomial_inequality (a b c : ℝ) (x : ℕ → ℝ) (n : ℕ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_coeff_sum : a + b + c = 1)
  (h_prod_one : (finset.range n).prod x = 1) :
  (finset.range n).prod (λ i, a * (x i)^2 + b * (x i) + c) ≥ 1 :=
sorry

end quadratic_trinomial_inequality_l637_637087


namespace terminal_side_in_third_quadrant_l637_637264

def is_equivalent_angle (a b : ℝ) : Prop := ∃ k : ℤ, a = b + k * 360

def in_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

theorem terminal_side_in_third_quadrant : 
  ∀ θ, θ = 600 → in_third_quadrant (θ % 360) :=
by
  intro θ
  intro hθ
  sorry

end terminal_side_in_third_quadrant_l637_637264


namespace racing_magic_meet_time_l637_637926

theorem racing_magic_meet_time :
  let time_rm := 150  -- seconds per round for The Racing Magic
  let time_cb := 3600 / 40  -- seconds per round for The Charging Bull
  let lcm_time := Nat.lcm time_rm time_cb
  lcm_time / 60 = 7.5 :=
by
  let time_rm := 150
  let time_cb := 3600 / 40
  have h1 : time_cb = 90 := by norm_num
  let lcm_time := Nat.lcm time_rm time_cb
  have h2 : lcm_time = 450 := by norm_num
  have h3 : lcm_time / 60 = 7.5 := by norm_num
  exact h3

end racing_magic_meet_time_l637_637926


namespace dot_product_of_unit_circle_points_l637_637074

noncomputable
def unit_circle_point (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem dot_product_of_unit_circle_points
  (x1 y1 x2 y2 θ : ℝ)
  (hx1 : unit_circle_point x1 y1)
  (hx2 : unit_circle_point x2 y2)
  (hθ : θ > π/2 ∧ θ < π)
  (h_sin : sin (θ + π/4) = 3/5) :
  x1 * x2 + y1 * y2 = -sqrt 2 / 10 :=
sorry

end dot_product_of_unit_circle_points_l637_637074


namespace Talia_father_age_l637_637153

def Talia_age (T : ℕ) : Prop := T + 7 = 20
def Talia_mom_age (M T : ℕ) : Prop := M = 3 * T
def Talia_father_age_in_3_years (F M : ℕ) : Prop := F + 3 = M

theorem Talia_father_age (T F M : ℕ) 
    (hT : Talia_age T)
    (hM : Talia_mom_age M T)
    (hF : Talia_father_age_in_3_years F M) :
    F = 36 :=
by 
  sorry

end Talia_father_age_l637_637153


namespace total_volume_of_extended_region_l637_637709

/-- Consider a rectangular parallelepiped (box) with dimensions 5x6x7 units, 
     extended by 2 units around, and with each vertex replaced by a hemisphere 
     of radius 2 units. Prove the total volume of this extended region is 
     990 + (128/3)π cubic units. -/
theorem total_volume_of_extended_region :
  let V_box := 5 * 6 * 7,
      V_extended := (5 + 4) * (6 + 4) * (7 + 4),
      V_extensions := V_extended - V_box,
      V_hemisphere := (2/3) * π * (2^3),
      V_total := V_box + V_extensions + 8 * V_hemisphere in
  V_total = 990 + 128/3 * π := 
by
  sorry

end total_volume_of_extended_region_l637_637709


namespace quadratic_has_distinct_real_roots_l637_637481

open Classical

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_distinct_real_roots (k : ℝ) (h_nonzero : k ≠ 0) : 
  (k > -1) ↔ (discriminant k (-2) (-1) > 0) :=
by
  unfold discriminant
  simp
  linarith

end quadratic_has_distinct_real_roots_l637_637481


namespace rounded_fraction_is_correct_l637_637914

-- Define the fraction 8/11
def fraction : ℚ := 8 / 11

-- Define the expected rounded result
def rounded_result : ℚ := 0.73

-- Prove that the fraction rounded to two decimal places equals the expected result
theorem rounded_fraction_is_correct : 
  Real.round_to 2 fraction = rounded_result :=
by
  sorry

end rounded_fraction_is_correct_l637_637914


namespace specified_time_correct_l637_637683

theorem specified_time_correct (x : ℝ) (h1 : 900.0 = dist) (h2 : slow_time = x + 1) 
  (h3 : fast_time = x - 3) (h4 : fast_speed = 2 * slow_speed) 
  (dist : ℝ := 900.0) (slow_speed : ℝ := 900.0 / (x + 1)) (fast_speed : ℝ := 900.0 / (x - 3)) 
  (slow_time fast_time : ℝ) :
  2 * slow_speed = fast_speed :=
by
  sorry

end specified_time_correct_l637_637683


namespace fifth_row_first_five_cells_l637_637395

-- Defining the problem conditions
def is_valid_grid (grid : list (list ℕ)) : Prop :=
  (∀ row, row ∈ grid → set_of row ⊆ {2, 0, 1, 5, 9}) ∧
  (∀ row, ∀ col, row < length grid ∧ col < length (nth row grid []) -
       grid[row][col] ≠ grid[nth row +1][col] ∧ grid[row][col] ≠ grid[nth row][col +1]) 

-- Defining the fifth row specification
def fifth_row_condition (grid : list (list ℕ)) : Prop :=
  nth 4 grid [] = [1, 5, 9, 9, 2]

-- Proof statement
theorem fifth_row_first_five_cells (grid : list (list ℕ)) 
  (h1 : length grid = 5)
  (h2 : length (nth 0 grid []) = length (nth 1 grid []) ∧ length (nth 1 grid []) = length (nth 2 grid []) ∧ length (nth 2 grid []) = length (nth 3 grid []) ∧ length (nth 3 grid []) = length (nth 4 grid []))
  (h3 : is_valid_grid grid) 
  : fifth_row_condition grid := 
sorry

end fifth_row_first_five_cells_l637_637395


namespace apples_needed_per_month_l637_637003

theorem apples_needed_per_month (chandler_apples_per_week : ℕ) (lucy_apples_per_week : ℕ) (weeks_per_month : ℕ)
  (h1 : chandler_apples_per_week = 23)
  (h2 : lucy_apples_per_week = 19)
  (h3 : weeks_per_month = 4) :
  (chandler_apples_per_week + lucy_apples_per_week) * weeks_per_month = 168 :=
by
  sorry

end apples_needed_per_month_l637_637003


namespace M_plus_m_eq_neg2_l637_637171

def y (x : ℝ) : ℝ := (1 / 3) * Real.cos x - 1
def M : ℝ := -2 / 3
def m : ℝ := -4 / 3

theorem M_plus_m_eq_neg2 : M + m = -2 := by
  -- Proof goes here
  sorry

end M_plus_m_eq_neg2_l637_637171


namespace problem_part1_problem_part2_l637_637429

variable (α : ℝ)

noncomputable def point_A : ℝ × ℝ := (3, 0)
noncomputable def point_B : ℝ × ℝ := (0, 3)
noncomputable def point_C (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)

noncomputable def vector_AC (α : ℝ) := (Real.cos α - 3, Real.sin α)
noncomputable def vector_BC (α : ℝ) := (Real.cos α, Real.sin α - 3)

theorem problem_part1 (h1 : Real.sqrt ((Real.cos α - 3) ^ 2 + (Real.sin α) ^ 2) = Real.sqrt (Real.cos α ^ 2 + (Real.sin α - 3) ^ 2))
  (h2 : α ∈ Ioc (Real.pi / 2) (3 * Real.pi / 2)) : α = 5 * Real.pi / 4 := by
  sorry

theorem problem_part2 (h1 : (Real.cos α - 3) * Real.cos α + Real.sin α * (Real.sin α - 3) = -1)
  (h2 : α ∈ Ioc (Real.pi / 2) (3 * Real.pi / 2)) : (2 * Real.sin α ^ 2 + Real.sin (2 * α)) / (1 + Real.tan α) = -5 / 9 := by
  sorry

end problem_part1_problem_part2_l637_637429


namespace nine_point_centers_line_or_parallelogram_l637_637245

-- Define a convex quadrilateral and its properties
variables {A B C D X : Type} [affine_plane A B C D]
variables (convex_quadrilateral : quadrilateral A B C D) (diagonals_intersect : intersection_point X A C B D)

-- Define the four triangles
def triangle1 := triangle X A B
def triangle2 := triangle X B C
def triangle3 := triangle X C D
def triangle4 := triangle X D A

-- Define nine-point centers
def nine_point_center (t : triangle) : point := sorry

noncomputable def nine_point_center1 := nine_point_center triangle1
noncomputable def nine_point_center2 := nine_point_center triangle2
noncomputable def nine_point_center3 := nine_point_center triangle3
noncomputable def nine_point_center4 := nine_point_center triangle4

-- Mathematical statement to prove
theorem nine_point_centers_line_or_parallelogram :
  collinear {nine_point_center1, nine_point_center2, nine_point_center3, nine_point_center4} ∨
  parallelogram {nine_point_center1, nine_point_center2, nine_point_center3, nine_point_center4} :=
sorry

end nine_point_centers_line_or_parallelogram_l637_637245


namespace sum_b_p_equals_l637_637051

theorem sum_b_p_equals :
  let b (p : ℕ) : ℕ := sorry in -- Define as the unique k such that |k - sqrt(p)| < 1/2
  let T := ∑ p in Finset.range (3000 + 1), b p in
  T = 21670 :=
by
  sorry

end sum_b_p_equals_l637_637051


namespace complex_expression_evaluation_l637_637788

-- Define the complex number z
def z : ℂ := 1 + 1*complex.i

-- Define the expression we want to prove
def expr : ℂ := (5 / z) + (z^2)

theorem complex_expression_evaluation : expr = (5/2) - (1/2)*complex.i := by
  sorry

end complex_expression_evaluation_l637_637788


namespace squareFree_sum_l637_637696

def isSquareFree (n : ℕ) : Prop :=
  ∀ (a : ℕ) (ha : a > 1), a*a ∣ n → False

def squareFreeIntegers : Set ℕ := { n | isSquareFree n }

-- Final theorem statement
theorem squareFree_sum (S := squareFreeIntegers) :
  ∑ k in S, (Int.floor (Real.sqrt (10^10 / k))) = 10^10 :=
sorry

end squareFree_sum_l637_637696


namespace h_geq_2_range_a_l637_637106

noncomputable def f (x : ℝ) : ℝ := real.log (x^2 + 1) / real.log 2
noncomputable def g (x a : ℝ) : ℝ := real.sqrt (x - a)
noncomputable def f_inv (x : ℝ) : ℝ := real.sqrt (2^x - 1)
noncomputable def h (x a : ℝ) : ℝ := f_inv x + g x a

theorem h_geq_2_range_a (a : ℝ) :
  (∀ x ≥ max 0 a, h x a ≥ 2) ↔ (a ≤ -4 ∨ a ≥ real.log 5 / real.log 2) :=
sorry

end h_geq_2_range_a_l637_637106


namespace john_profit_is_1500_l637_637521

noncomputable def john_profit (total_puppies : ℕ) (half_given_away : ℕ) 
  (puppies_kept : ℕ) (sell_price : ℕ) (stud_fee : ℕ) : ℕ :=
  (total_puppies - half_given_away - puppies_kept) * sell_price - stud_fee

theorem john_profit_is_1500 : john_profit 8 4 1 600 300 = 1500 := 
by simp [john_profit]; sorry

end john_profit_is_1500_l637_637521


namespace trig_identity_l637_637755

theorem trig_identity (α : ℝ) (h : Real.sin (α - π / 12) = 1 / 3) : Real.cos (α + 5 * π / 12) = -1 / 3 :=
by
  sorry

end trig_identity_l637_637755


namespace number_of_kids_in_other_class_l637_637596

theorem number_of_kids_in_other_class (average_zits_other : ℕ) (average_zits_jones : ℕ)
  (extra_zits_jones : ℕ) (kids_jones : ℕ) (total_zits_jones : ℕ) (zits_other_class : ℕ) : 
  average_zits_other = 5 → average_zits_jones = 6 → 
  extra_zits_jones = 67 → kids_jones = 32 → 
  total_zits_jones = kids_jones * average_zits_jones → 
  total_zits_jones = zits_other_class + extra_zits_jones →
  ∃ kids_other : ℕ, zits_other_class = kids_other * average_zits_other ∧ kids_other = 25 :=
by
  intros h_average_zits_other h_average_zits_jones h_extra_zits_jones h_kids_jones 
  h_total_zits_jones h_total_zits_jones_eq
  have h1 : total_zits_jones = 192 := sorry  -- This should follow from substitution
  have h2 : total_zits_jones = (kids_other * 5) + 67 := sorry  -- Substitution again
  have h3 : 192 = (kids_other * 5) + 67 := sorry  -- Combining the above results
  have h4 : 125 = kids_other * 5 := sorry  -- Simplifying the equation
  have h5 : kids_other = 25 := by linarith  -- Solving for kids_other
  use 25
  { exact ⟨125, rfl⟩ }  
  -- Extra conditions simplification or continue with necessary assumptions to reach conclusion
  sorry 


end number_of_kids_in_other_class_l637_637596


namespace extremum_range_k_l637_637102

noncomputable def f (x k : Real) : Real :=
  Real.exp x / x + k * (Real.log x - x)

/-- 
For the function f(x) = (exp(x) / x) + k * (log(x) - x), if x = 1 is the only extremum point, 
then k is in the interval (-∞, e].
-/
theorem extremum_range_k (k : Real) : 
  (∀ x : Real, (0 < x) → (f x k ≤ f 1 k)) → 
  k ≤ Real.exp 1 :=
sorry

end extremum_range_k_l637_637102


namespace G_is_centroid_l637_637708

open EuclideanGeometry

variables 
  {S : Circle}
  {P A B C D M G : Point}
  (h1 : PointOnCircle A S)
  (h2 : PointOnCircle B S)
  (h3 : PointOnCircle C S)
  (h4 : PointOnCircle D S)
  (hP_outside : ¬Inside P S)
  (hA_tangent : Tangent P A S)
  (hB_tangent : Tangent P B S)
  (hM_midpoint : Midpoint M A B)
  (hC_inside_triangle : InsideTriangle C A B P)
  (hAC_G_intersect : LineIntersection (LineThrough A C) (LineThrough P M) G)
  (hPM_D_intersect : LineIntersection (LineThrough P M) S D)
  (hD_outside_triangle : OutsideTriangle D A B P)
  (hBD_parallel_AC : Parallel (LineThrough B D) (LineThrough A C))

theorem G_is_centroid : 
  IsCentroid G A B P :=
sorry

end G_is_centroid_l637_637708


namespace f_symmetric_about_pi_l637_637452

variable (a b : ℝ)
variable (h : a ≠ 0)
variable (f : ℝ → ℝ)

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := a * Real.sin x - b * Real.cos x

-- Hypothesis: the function is symmetric about x = π / 4
axiom sym_f : ∀ x : ℝ, f (π / 4 + x) = f (π / 4 - x)

-- The main statement to prove
theorem f_symmetric_about_pi (x : ℝ) :
  f (3 * π / 4 - x) = f (3 * π / 4 - (π - x)) :=
sorry

end f_symmetric_about_pi_l637_637452


namespace find_number_l637_637611

def valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ((n % 10 = 6 ∧ ¬ (n % 7 = 0)) ∨ (¬ (n % 10 = 6) ∧ n % 7 = 0)) ∧
  ((n > 26 ∧ ¬ (n % 10 = 8)) ∨ (¬ (n > 26) ∧ n % 10 = 8)) ∧
  ((n % 13 = 0 ∧ ¬ (n < 27)) ∨ (¬ (n % 13 = 0) ∧ n < 27))

theorem find_number : ∃ n : ℕ, valid_number n ∧ n = 91 := by
  sorry

end find_number_l637_637611


namespace chelsea_needs_67_bullseyes_l637_637836

theorem chelsea_needs_67_bullseyes 
  (total_shots : ℕ) (initial_lead : ℤ) (bullseye_score : ℤ) 
  (other_scores : list ℤ) (minimum_score : ℤ) : 
  total_shots = 150 ∧ 
  initial_lead = 70 ∧ 
  bullseye_score = 10 ∧ 
  other_scores = [8, 6, 2, 0] ∧ 
  minimum_score = 2 
  → ∃ n : ℕ, n = 67 ∧ 
    (∀ k : ℤ, 
      let opponent_final_score := (k - initial_lead) + 75 * bullseye_score in 
      let chelsea_final_score := k + 8 * n + 150 in 
      chelsea_final_score > opponent_final_score) :=
by
  intros h
  sorry

end chelsea_needs_67_bullseyes_l637_637836


namespace no_infinite_monochromatic_arithmetic_progression_l637_637373

theorem no_infinite_monochromatic_arithmetic_progression : 
  ∃ (coloring : ℕ → ℕ), (∀ (q r : ℕ), ∃ (n1 n2 : ℕ), coloring (q * n1 + r) ≠ coloring (q * n2 + r)) := sorry

end no_infinite_monochromatic_arithmetic_progression_l637_637373


namespace students_not_taken_test_l637_637557

theorem students_not_taken_test 
  (num_enrolled : ℕ) 
  (answered_q1 : ℕ) 
  (answered_q2 : ℕ) 
  (answered_both : ℕ) 
  (H_num_enrolled : num_enrolled = 40) 
  (H_answered_q1 : answered_q1 = 30) 
  (H_answered_q2 : answered_q2 = 29) 
  (H_answered_both : answered_both = 29) : 
  num_enrolled - (answered_q1 + answered_q2 - answered_both) = 10 :=
by {
  sorry
}

end students_not_taken_test_l637_637557


namespace f_zero_f_odd_solve_inequality_l637_637195

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom increasing_on_nonneg : ∀ {x y : ℝ}, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem f_zero : f 0 = 0 :=
by sorry

theorem f_odd (x : ℝ) : f (-x) = -f x :=
by sorry

theorem solve_inequality {x : ℝ} (h : 0 < x) : f (Real.log x / Real.log 10 - 1) < 0 ↔ 0 < x ∧ x < 10 :=
by sorry

end f_zero_f_odd_solve_inequality_l637_637195


namespace solution_difference_l637_637875

theorem solution_difference (m n : ℝ) (h_eq : ∀ x : ℝ, (x - 4) * (x + 4) = 24 * x - 96 ↔ x = m ∨ x = n) (h_distinct : m ≠ n) (h_order : m > n) : m - n = 16 :=
sorry

end solution_difference_l637_637875


namespace middle_car_is_Sin_l637_637048

-- Definitions for friends and their positions
def friends := { Lin, Jin, Min, Sin, Kin }
def position (f : friends) : ℕ := sorry

-- Given conditions as axioms
axiom Min_last : position Min = 5
axiom Lin_behind_Sin : ∃ (p : ℕ), position Sin = p ∧ position Lin = p + 1
axiom Jin_front_Min : position Jin = 4
axiom Kin_away_from_Jin : ∀ (p : ℕ), position Kin ≠ p ∧ position Kin ≠ p + 1

-- Derived condition: Middle car (position 3)
theorem middle_car_is_Sin : position Sin = 3 := sorry

end middle_car_is_Sin_l637_637048


namespace total_money_l637_637669

theorem total_money (A B C : ℝ) (h1 : A = 1 / 2 * (B + C))
  (h2 : B = 2 / 3 * (A + C)) (h3 : A = 122) :
  A + B + C = 366 := by
  sorry

end total_money_l637_637669


namespace det_A_eq_zero_l637_637392

def A (α β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [sin α * sin β, sin α * cos β, cos α],
    [cos β, -sin β, 0],
    [-cos α * sin β, -cos α * cos β, sin α]
  ]

theorem det_A_eq_zero (α β : ℝ) : Matrix.det (A α β) = 0 := 
  sorry

end det_A_eq_zero_l637_637392


namespace sequence_an_l637_637486

theorem sequence_an (a : ℕ → ℝ) (h0 : a 1 = 1)
  (h1 : ∀ n, 4 * a n * a (n + 1) = (a n + a (n + 1) - 1)^2)
  (h2 : ∀ n > 1, a n > a (n - 1)) :
  ∀ n, a n = n^2 := 
sorry

end sequence_an_l637_637486


namespace find_range_p_l637_637435

noncomputable def is_monotonic_increasing {α : Type*} [preorder α] (f : ℕ → α) : Prop :=
∀ n, f n ≤ f (n + 1)

noncomputable def sequence_cond (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, a n = 4 * S n - 2 * (a n)^3

noncomputable def inequality_cond (a : ℕ → ℝ) (p : ℝ) : Prop :=
∀ n ≥ 1, (a n - 8) * (a (n + p) - 8) ≥ 1 + (p + 8) / (2^(a n - 1))

theorem find_range_p {a : ℕ → ℝ} {S : ℕ → ℝ} 
  (h_mono : is_monotonic_increasing a) (h_a1 : a 1 > 0)
  (h_seq : sequence_cond a S) (h_ineq : inequality_cond a p) :
  ∃ (p : ℝ), p ≥ 24 / 61 := 
sorry

end find_range_p_l637_637435


namespace y_squared_plus_z_squared_eq_l637_637920

variables {a b c x y z : ℝ}

-- Conditions
def condition1 : Prop := x * y = a
def condition2 : Prop := x + y = b
def condition3 : Prop := x * z = c

-- Proof problem
theorem y_squared_plus_z_squared_eq :
  condition1 → condition2 → condition3 → 
  y^2 + z^2 = (2*b^2 - 4*a + 4*c^2) / (b + real.sqrt (b^2 - 4*a))^2 :=
fun h1 h2 h3 => sorry

end y_squared_plus_z_squared_eq_l637_637920


namespace parallel_lines_eq_a_l637_637115

theorem parallel_lines_eq_a (a : ℝ) :
  (let line1 : ℝ → ℝ := λ x, a * x - 2 in
   let line2 := 3 * x - (a + 2) * y + 1 = 0 in
   (line1 = line2) ∧ (a * x - 2) = line1) →
  (a = -3 ∨ a = 1) := by
  sorry

end parallel_lines_eq_a_l637_637115


namespace exists_unique_c_l637_637075

theorem exists_unique_c (a : ℝ) (h₁ : 1 < a) :
  (∃ (c : ℝ), ∀ (x : ℝ), x ∈ Set.Icc a (2 * a) → ∃ (y : ℝ), y ∈ Set.Icc a (a ^ 2) ∧ (Real.log x / Real.log a + Real.log y / Real.log a = c)) ↔ a = 2 :=
by
  sorry

end exists_unique_c_l637_637075


namespace part_a_possible_to_zero_part_b_impossible_to_zero_l637_637204

-- Part a): Prove that it is possible to make all numbers zero from the sequence 1, 2, 3, ..., 2003 using the allowed operation
theorem part_a_possible_to_zero : 
  ∀ (s : Set ℕ), (s = { n | 1 ≤ n ∧ n ≤ 2003 }) →
  (∃ steps : List ℕ × List ℕ, 
    (steps.fst = s ∧ steps.snd = [] 
     ∧ ∀ (a b : ℕ), (a ∈ steps.fst → b ∈ steps.fst → steps.snd ∉ steps.fst 
      → steps.snd = a - b ∨ steps.snd = b - a)) 
    → steps.fst = ∅) :=
sorry

-- Part b): Prove that it is impossible to make all numbers zero from the sequence 1, 2, 3, ..., 2005 using the allowed operation
theorem part_b_impossible_to_zero : 
  ∀ (s : Set ℕ), (s = { n | 1 ≤ n ∧ n ≤ 2005 }) →
  ¬ (∃ steps : List ℕ × List ℕ, 
    (steps.fst = s ∧ steps.snd = [] 
     ∧ ∀ (a b : ℕ), (a ∈ steps.fst → b ∈ steps.fst → steps.snd ∉ steps.fst 
      → steps.snd = a - b ∨ steps.snd = b - a)) 
    → steps.fst = ∅) :=
sorry

end part_a_possible_to_zero_part_b_impossible_to_zero_l637_637204


namespace smallest_abundant_not_multiple_of_4_l637_637743

-- Define what it means for a number to be abundant.
def is_abundant (n : ℕ) : Prop :=
  ∑ d in (Finset.filter (λ d, d < n) (Finset.divisors n)), d > n

-- Define what it means for a number to not be a multiple of 4.
def not_multiple_of_4 (n : ℕ) : Prop :=
  ¬ (n % 4 = 0)

-- The smallest abundant number that is not a multiple of 4.
theorem smallest_abundant_not_multiple_of_4 : ∃ n : ℕ, is_abundant n ∧ not_multiple_of_4 n ∧ ∀ m : ℕ, (is_abundant m ∧ not_multiple_of_4 m) → n ≤ m :=
  sorry

end smallest_abundant_not_multiple_of_4_l637_637743


namespace simple_interest_for_2_years_l637_637561

noncomputable def calculate_simple_interest (P r t : ℝ) : ℝ :=
  (P * r * t) / 100

theorem simple_interest_for_2_years (CI P r t : ℝ) (hCI : CI = P * (1 + r / 100)^t - P)
  (hCI_value : CI = 615) (r_value : r = 5) (t_value : t = 2) : 
  calculate_simple_interest P r t = 600 :=
by
  sorry

end simple_interest_for_2_years_l637_637561


namespace sum_expr_value_l637_637407

theorem sum_expr_value : 
  (∑ i in Finset.range 1010, (2 * i + 1)) - (∑ i in Finset.range 1009 \ (Finset.singleton 0), (2 * i)) = 1010 :=
by
  sorry

end sum_expr_value_l637_637407


namespace min_val_of_f_l637_637753

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 - 6 * x + m

theorem min_val_of_f {m : ℝ} (h_max : ∀ x ∈ set.Icc 0 2, f x m ≤ 3) : f 0 (-1) = -1 :=
by sorry

end min_val_of_f_l637_637753


namespace sin_trig_identity_l637_637900

theorem sin_trig_identity (θ : ℝ) : 
  sin θ * sin (real.pi / 3 - θ) * sin (real.pi / 3 + θ) = 1 / 4 * sin (3 * θ) :=
sorry

end sin_trig_identity_l637_637900


namespace find_f_and_q_l637_637795

theorem find_f_and_q (m : ℤ) (q : ℝ) :
  (∀ x > 0, (x : ℝ)^(-m^2 + 2*m + 3) = (x : ℝ)^4) ∧
  (∀ x ∈ [-1, 1], 2 * (x^2) - 8 * x + q - 1 > 0) →
  q > 7 :=
by
  sorry

end find_f_and_q_l637_637795


namespace height_ratio_l637_637685

theorem height_ratio (C : ℝ) (h_o : ℝ) (V_s : ℝ) (h_s : ℝ) (r : ℝ) :
  C = 18 * π →
  h_o = 20 →
  V_s = 270 * π →
  C = 2 * π * r →
  V_s = 1 / 3 * π * r^2 * h_s →
  h_s / h_o = 1 / 2 :=
by
  sorry

end height_ratio_l637_637685


namespace number_on_board_is_91_l637_637615

-- Definitions based on conditions
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def ends_in_digit (d n : ℕ) : Prop := n % 10 = d

def divisible_by (m n : ℕ) : Prop := n % m = 0

def andrey_statements (n : ℕ) : Prop :=
  (ends_in_digit 6 n ∨ divisible_by 7 n) ∧ ¬(ends_in_digit 6 n ∧ divisible_by 7 n)

def borya_statements (n : ℕ) : Prop :=
  (n > 26 ∨ ends_in_digit 8 n) ∧ ¬(n > 26 ∧ ends_in_digit 8 n)

def sasha_statements (n : ℕ) : Prop :=
  (divisible_by 13 n ∨ n < 27) ∧ ¬(divisible_by 13 n ∧ n < 27)

-- Mathematical equivalent proof problem
theorem number_on_board_is_91 (n : ℕ) :
  is_two_digit n →
  andrey_statements n →
  borya_statements n →
  sasha_statements n →
  n = 91 :=
by {
  intro _ _ _ _,
  -- Proof goes here, skipped with sorry
  sorry
}

end number_on_board_is_91_l637_637615


namespace specified_time_eq_l637_637680

noncomputable def slow_horse_days (x : ℝ) := x + 1
noncomputable def fast_horse_days (x : ℝ) := x - 3

theorem specified_time_eq (x : ℝ) (h1 : slow_horse_days x > 0) (h2 : fast_horse_days x > 0) :
  (900 / slow_horse_days x) * 2 = 900 / fast_horse_days x :=
by
  sorry

end specified_time_eq_l637_637680


namespace output_restored_l637_637938

noncomputable def initial_increase : ℝ :=
  let P := 10 in
  let O := 1 in -- assuming O = 1 for simplicity as it simplifies to P independent of O's value 
  (O + P / 100 * O) * 1.20 * 0.7575757575757576 = O

theorem output_restored : initial_increase = 10 := 
by
  calc (1 + 10 / 100 * 1) * 1.20 * 0.7575757575757576
      = 1 : sorry

end output_restored_l637_637938


namespace chickens_eaten_first_day_l637_637201

-- Define variables and given conditions
def chickens_eaten (day : ℕ) : ℚ := 
  if day == 1 then a1
  else if day == 2 then 12
  else if day <= 15 
       then x
       else 0

def x : ℚ := (32 : ℚ) / (13 : ℚ)
def total_chickens_up_to_15 : ℚ := 
  (chickens_eaten 1 + chickens_eaten 2 + 13 * x) 

-- Lean Statement
theorem chickens_eaten_first_day : a1 = - 92 / 13 :=
by
  -- Given conditions
  have hx : 13 * x = 32, from sorry,
  have ha2 : chickens_eaten 2 = 12, from sorry,
  
  -- Solve
  have total_up_to_15 := (chickens_eaten 1 + chickens_eaten 2 + 13 * x),
  
  -- Final statements
  have ha1a2 : chickens_eaten 1 + chickens_eaten 2 = 2 * x, from sorry,
  rw [hx, ha2] at ha1a2,
  linarith,
  sorry

end chickens_eaten_first_day_l637_637201


namespace quadratic_root_range_l637_637478

theorem quadratic_root_range (k : ℝ) (hk : k ≠ 0) (h : (4 + 4 * k) > 0) : k > -1 :=
by sorry

end quadratic_root_range_l637_637478


namespace probability_red_blue_l637_637622

-- Definitions for probabilities and total number of marbles
def total_marbles : ℕ := 500
def white_prob : ℝ := 1 / 4
def green_prob : ℝ := 1 / 5
def yellow_prob : ℝ := 1 / 6
def orange_prob : ℝ := 1 / 10

-- Calculated number of each type of marble
def white_marbles : ℕ := (white_prob * total_marbles)
def green_marbles : ℕ := (green_prob * total_marbles)
def yellow_marbles : ℕ := 83  -- Adjusted to nearest whole number
def orange_marbles : ℕ := (orange_prob * total_marbles)

def non_red_blue_marbles : ℕ := white_marbles + green_marbles + yellow_marbles + orange_marbles
def red_blue_marbles : ℕ := total_marbles - non_red_blue_marbles
def red_blue_prob : ℝ := red_blue_marbles / total_marbles

-- Theorem to be proven
theorem probability_red_blue : red_blue_prob = 71 / 250 :=
by
  sorry

end probability_red_blue_l637_637622


namespace geometric_sequence_ln_l637_637858

noncomputable def newton_seq (f : ℝ → ℝ) (f' : ℝ → ℝ) (x : ℕ → ℝ) (n : ℕ) : ℝ :=
x n - (f (x n) / f' (x n))

theorem geometric_sequence_ln 
  (a b c : ℝ) (a_pos : 0 < a)
  (f := λ x, a * x^2 + b * x + c)
  (h1 : f 1 = 0)
  (h2 : f 2 = 0)
  (f' := λ x, 2 * a * x + b)
  (x : ℕ → ℝ)
  (h3 : ∀ n, x n > 2)
  (h_newton_seq : ∀ n, x (n+1) = newton_seq f f' x n)
  (a_seq : ℕ → ℝ)
  (h_a_seq : ∀ n, a_seq n = Real.log ((x n - 2) / (x n - 1)))
  (h_a1 : a_seq 1 = 2) :
  ∀ n, a_seq (n+1) = 2 * a_seq n :=
sorry -- We include sorry so that the Lean code builds without requiring a proof.

end geometric_sequence_ln_l637_637858


namespace solve_integral_eq_l637_637657

variable (b : ℝ) (ψ : ℝ → ℝ) -- Declaring the variables

def integral_eq (t : ℝ) : ℝ :=
  (1 / (2 * b * t)) * ∫ x in 0..+∞, x * (λx, f x) * sin (x^2 / (4 * b * t))

def f (x : ℝ) : ℝ :=
  (2 / π) * ∫ t in 0..+∞, (ψ t / t) * sin (x^2 / (4 * b * t))

theorem solve_integral_eq (f : ℝ → ℝ) :
  (∀ t, ψ t = integral_eq f t) → (∀ x, f x = (2 / π) * ∫ t in 0..+∞, (ψ t / t) * sin (x^2 / (4 * b * t))) :=
by
  intro hyp
  sorry

end solve_integral_eq_l637_637657


namespace opposite_sides_line_l637_637485

theorem opposite_sides_line (a : ℝ) :
  (0 + 0 - a) * (1 + 1 - a) < 0 ↔ 0 < a ∧ a < 2 :=
by
  calc
  sorry -- This is where the detailed proof goes

end opposite_sides_line_l637_637485


namespace complex_number_pure_imaginary_l637_637473

theorem complex_number_pure_imaginary (a : ℝ) (z : ℂ) : 
  z = complex.mk (a^2 + a - 2) (a^2 - 1) → 
  (a^2 + a - 2 = 0 ∧ a^2 - 1 ≠ 0) → 
  a = -2 :=
by
  sorry

end complex_number_pure_imaginary_l637_637473


namespace find_f_neg27_l637_637196

def P := (8,2)
def f (x : ℝ) (α : ℝ) := x^α

theorem find_f_neg27 (a : ℝ) (α : ℝ)
  (h1 : P.2 = 3 * a^(P.1 - 8) - 1)
  (h2 : P.2 = f P.1 α) :
  f (-27) (1/3) = -3 := by
  sorry

end find_f_neg27_l637_637196


namespace find_BD_l637_637854

section
variables {A B C D E : Type}
variables [euclidean_geometry A] [euclidean_geometry B] [euclidean_geometry C] 
variables [euclidean_geometry D] [euclidean_geometry E]

-- Given conditions
variables (angle_C_90 : ∠C = 90)
variables (AC_eq_8 : dist A C = 8)
variables (BC_eq_12 : dist B C = 12)
variables (D_on_AB : D ∈ line_through A B)
variables (E_on_BC : E ∈ line_through B C)
variables (angle_BED_90 : ∠ B E D = 90)
variables (DE_eq_6 : dist D E = 6)

-- Proof problem
theorem find_BD : dist B D = 3 * sqrt 13 :=
by sorry
end

end find_BD_l637_637854


namespace four_digit_numbers_divisible_by_5_l637_637809

theorem four_digit_numbers_divisible_by_5 : 
  let smallest_4_digit := 1000
  let largest_4_digit := 9999
  let divisible_by_5 (n : ℕ) := ∃ k : ℕ, n = 5 * k
  ∃ n : ℕ, ( ∀ x : ℕ, smallest_4_digit ≤ x ∧ x ≤ largest_4_digit ∧ divisible_by_5 x ↔ (smallest_4_digit + (n-1) * 5 = x) ) ∧ n = 1800 :=
by
  sorry

end four_digit_numbers_divisible_by_5_l637_637809


namespace trains_cross_pole_simultaneously_l637_637161

/-
  Define the lengths and speeds of Train A and Train B.
  Convert speeds from km/hr to m/s.
-/

def lengthA : ℕ := 300
def speedA_kmh : ℕ := 90
def speedA_ms : ℚ := speedA_kmh * (1000:ℚ) / 3600

def lengthB : ℕ := 400
def speedB_kmh : ℕ := 120
def speedB_ms : ℚ := speedB_kmh * (1000:ℚ) / 3600

/-
  Define the time it takes for each train to cross the electric pole.
  Using the formula time = distance / speed.
-/

def timeA := (lengthA:ℚ) / speedA_ms
def timeB := (lengthB:ℚ) / speedB_ms

/-
  The theorem states that timeA and timeB are both approximately 12 seconds.
-/

theorem trains_cross_pole_simultaneously : timeA ≈ 12 ∧ timeB ≈ 12 := by
  sorry

end trains_cross_pole_simultaneously_l637_637161


namespace polynomial_has_no_real_roots_l637_637720

theorem polynomial_has_no_real_roots :
  ∀ x : ℝ, x^8 - x^7 + 2*x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 4*x^2 - 4*x + 5/2 ≠ 0 :=
by
  sorry

end polynomial_has_no_real_roots_l637_637720


namespace transformed_sequence_l637_637192

noncomputable def transformation_rule (m n i : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ → ℕ) : Prop :=
  if (i ≤ n) then
    match m with
    | 1 => b 1 i = a i + a (i % n + 1)
    | (m+1) => ∀ (i : ℕ), b (m+1) i = b m i + b m (i % n + 1)
    end
  else false

theorem transformed_sequence {a : ℕ → ℕ} {b : ℕ → ℕ → ℕ} (n m i : ℕ) (h1 : ∀ i, a i = i)
    (h2 : n ≥ 2) (h_trans : ∀ m i, transformation_rule m n i a b):
  (b 3 5 = 52) ∧ ∀ m i, b m i = ∑ j in range (m+1), a ((i+j) % n) * (binom m j) :=
sorry

end transformed_sequence_l637_637192


namespace negation_example_l637_637259

theorem negation_example : (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) :=
by
  sorry

end negation_example_l637_637259


namespace litter_pickup_l637_637925

theorem litter_pickup (total_litter pieces_of_litter : ℕ) (glass_bottles : ℕ) 
  (h1 : pieces_of_litter = 18) (h2 : glass_bottles = 10) :
  total_litter = pieces_of_litter - glass_bottles → total_litter = 8 := by
  intro hl
  rw [h1, h2] at hl
  have : 18 - 10 = 8 := rfl
  rw this at hl
  exact hl

end litter_pickup_l637_637925


namespace num_values_x_satisfying_l637_637121

theorem num_values_x_satisfying (
  f : ℝ → ℝ → ℝ)
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (x : ℝ)
  (h_eq : ∀ x, f (cos x) (sin x) = 2 ↔ (cos x) ^ 2 + 3 * (sin x) ^ 2 = 2)
  (h_interval : ∀ x, -20 < x ∧ x < 90)
  (h_cos_sin : ∀ x, cos x = cos (x) ∧ sin x = sin (x)) :
  ∃ n, n = 70 := sorry

end num_values_x_satisfying_l637_637121


namespace hyperbola_line_intersection_l637_637789

theorem hyperbola_line_intersection
  (a b : ℝ)
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : (x₁^2) / a - (y₁^2) / b = 1)
  (h₂ : (x₂^2) / a - (y₂^2) / b = 1)
  (h₃ : x₁ + y₁ = 1)
  (h₄ : x₂ + y₂ = 1)
  (h₅ : x₁ * x₂ + y₁ * y₂ = 0) :
  1 / a - 1 / b = 2 :=
begin
  sorry
end

end hyperbola_line_intersection_l637_637789


namespace package_candies_l637_637002

theorem package_candies (candies_eaten candies_left : ℕ) (he : candies_eaten = 10) (hl : candies_left = 2) :
  candies_eaten + candies_left = 12 :=
by
  rw [he, hl]
  rfl

end package_candies_l637_637002


namespace simplify_f_find_f_for_specific_x_l637_637766

section
variable (α x : ℝ)

/-- Condition: α is an angle in the third quadrant and x = 1/2 -/
def condition1 : Prop := -π < α ∧ α < -π / 2 ∧ x = 1 / 2

/-- Define f(α) based on given conditions -/
def f (α : ℝ) : ℝ := (-cos α) * sin α * (-tan α) / ((-tan α) * sin α)

/-- Hypothesis: -/ 
theorem simplify_f (h : condition1 α x) : f α = -cos α :=
by sorry

/-- Hypothesis: Given that cos(α - 3π/2) = 1/5 and simplifying based on condition1 -/
theorem find_f_for_specific_x (h1 : condition1 α x)
  (h2 : x = -1 / 4) 
  (h3 : cos (α - 3 * π / 2) = 1 / 5) : f α = 2 * √6 / 5 :=
by sorry

end

end simplify_f_find_f_for_specific_x_l637_637766


namespace intersection_is_M_l637_637553

open Set

noncomputable def M : Set ℤ := {0, 1}

noncomputable def N : Set ℤ := {x | x ∈ ℤ ∧ x ≤ 1}

theorem intersection_is_M : M ∩ N = M := by
  sorry

end intersection_is_M_l637_637553


namespace find_unique_pair_l637_637412

theorem find_unique_pair (x y : ℝ) :
  (∀ (u v : ℝ), (u * x + v * y = u) ∧ (u * y + v * x = v)) ↔ (x = 1 ∧ y = 0) :=
by
  -- This is to ignore the proof part
  sorry

end find_unique_pair_l637_637412


namespace arithmetic_seq_sum_l637_637156

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n+1) - a n = a (n+2) - a (n+1))
  (h2 : a 3 + a 7 = 37) :
  a 2 + a 4 + a 6 + a 8 = 74 := 
sorry

end arithmetic_seq_sum_l637_637156


namespace smallest_n_Sn_gt_2023_l637_637444

open Nat

theorem smallest_n_Sn_gt_2023 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 4) →
  (∀ n : ℕ, n > 0 → a n + a (n + 1) = 4 * n + 2) →
  (∀ m : ℕ, S m = if m % 2 = 0 then m ^ 2 + m else m ^ 2 + m + 2) →
  ∃ n : ℕ, S n > 2023 ∧ ∀ k : ℕ, k < n → S k ≤ 2023 :=
sorry

end smallest_n_Sn_gt_2023_l637_637444


namespace parabolas_intersection_distance_squared_l637_637211

theorem parabolas_intersection_distance_squared :
  let P1_focus := (20, 22)
  let P2_focus := (20, 22)
  let P1_directrix := "x-axis"
  let P2_directrix := "y-axis"
  let intersection_points := set_of_points P1_focus P1_directrix P2_focus P2_directrix -- Hypothetical function to find intersection points
  let X := intersection_points.fst
  let Y := intersection_points.snd
in dist_sq X Y = 3520 :=
by
  let P1_focus := (20, 22)
  let P2_focus := (20, 22)
  let P1_directrix := "x-axis"
  let P2_directrix := "y-axis"
  let intersection_points := set_of_points P1_focus P1_directrix P2_focus P2_directrix -- Hypothetical function to find intersection points
  let X := intersection_points.fst
  let Y := intersection_points.snd
  sorry

end parabolas_intersection_distance_squared_l637_637211


namespace remainder_when_divided_by_39_l637_637671

theorem remainder_when_divided_by_39 (N : ℤ) (h1 : ∃ k : ℤ, N = 13 * k + 3) : N % 39 = 3 :=
sorry

end remainder_when_divided_by_39_l637_637671


namespace smallest_x_for_f_equals_f2022_l637_637333

noncomputable def f (x : ℝ) : ℝ :=
  if hx : 2 ≤ x ∧ x ≤ 4 then 2 - abs (x - 3)
  else 4 * f (x / 4)

theorem smallest_x_for_f_equals_f2022 :
  ∃ x : ℝ, f x = f 2022 ∧ ∀ y : ℝ, y < x → f y ≠ f 2022 :=
sorry

end smallest_x_for_f_equals_f2022_l637_637333


namespace possible_value_of_a_l637_637084

theorem possible_value_of_a (a r : ℤ) (h1 : r > 0) (h2 : |AB| = 2 * real.sqrt 2) 
  (h3 : (x - a)^2 + (y - 1)^2 = r^2) (h4 : y = x) : a = 3 :=
sorry

end possible_value_of_a_l637_637084


namespace symmetric_projection_l637_637852

def point := ℤ × ℤ × ℤ

def symmetric_y_axis (p : point) := (-p.1, p.2, -p.3)

def project_xOz_plane (p : point) := (p.1, 0, p.3)

theorem symmetric_projection :
  project_xOz_plane (symmetric_y_axis (4, 7, 6)) = (-4, 0, -6) :=
by
  sorry

end symmetric_projection_l637_637852


namespace problem_equiv_l637_637613

theorem problem_equiv :
  ∃ n : ℕ, 
    10 ≤ n ∧ n < 100 ∧ 
    ((n % 10 = 6 ∧ ¬ (n % 7 = 0)) ∨ (n % 10 ≠ 6 ∧ n % 7 = 0)) ∧
    ((n > 26 ∧ ¬ (n % 10 = 8)) ∨ (n ≤ 26 ∧ n % 10 = 8)) ∧
    ((n % 13 = 0 ∧ ¬ (n < 27)) ∨ (n % 13 ≠ 0 ∧ n < 27)) ∧
    n = 91 :=
begin
  sorry
end

end problem_equiv_l637_637613


namespace no_single_lattice_point_on_circle_l637_637571

   theorem no_single_lattice_point_on_circle (a b : ℚ) :
     ∀ (R : ℝ), ¬ ∃ (x y : ℤ), (x - a) ^ 2 + (y - b) ^ 2 = R ^ 2 ∧ 
       (∀ (x' y' : ℤ), (x', y') ≠ (x, y) → (x' - a) ^ 2 + (y' - b) ^ 2 ≠ R ^ 2) :=
   by
     sorry
   
end no_single_lattice_point_on_circle_l637_637571


namespace triangle_base_difference_l637_637589

theorem triangle_base_difference (b h : ℝ) 
    (h_pos : h > 0) (b_pos : b > 0)
    (area_A : ℝ) (hA_height_cond : real := 0.90 * h) 
    (area_cond : area_A = 0.99 * (1 / 2) * b * h) 
    (areaA_formula : area_A = (1 / 2) * hA_height_cond * b) :
    ∃ (b_A : ℝ), b_A = 1.10 * b := 
by
  sorry

end triangle_base_difference_l637_637589


namespace find_second_number_l637_637606

theorem find_second_number
  (x y z : ℚ)
  (h1 : x + y + z = 120)
  (h2 : x = (3 : ℚ) / 4 * y)
  (h3 : z = (7 : ℚ) / 5 * y) :
  y = 800 / 21 :=
by
  sorry

end find_second_number_l637_637606


namespace range_of_m_l637_637830

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), x^2 - 4 * x - 2 * m + 1 ≤ 0) ↔ m ∈ Set.Ici (3 : ℝ) := 
sorry

end range_of_m_l637_637830


namespace rounded_fraction_is_correct_l637_637913

-- Define the fraction 8/11
def fraction : ℚ := 8 / 11

-- Define the expected rounded result
def rounded_result : ℚ := 0.73

-- Prove that the fraction rounded to two decimal places equals the expected result
theorem rounded_fraction_is_correct : 
  Real.round_to 2 fraction = rounded_result :=
by
  sorry

end rounded_fraction_is_correct_l637_637913


namespace speed_of_faster_train_is_180_l637_637964

-- Define the lengths of the trains in meters
def length_train1 : ℕ := 100
def length_train2 : ℕ := 200

-- Define the distance apart in meters
def distance_apart : ℕ := 450

-- Define the speed of the first train in km/h and convert it to m/s
def speed_train1_kmh : ℕ := 90
def speed_train1_ms : ℝ := (90 * 1000) / 3600

-- Define the time until the trains meet in seconds
def time_until_meet : ℝ := 9.99920006399488

-- Define the combined speed of the trains in m/s
def combined_speed : ℝ := ((length_train1 + length_train2 + distance_apart) : ℝ) / time_until_meet

-- Define the speed of the second train in m/s
def speed_train2_ms : ℝ := combined_speed - speed_train1_ms

-- Convert the speed of the second train to km/h
def speed_train2_kmh : ℝ := speed_train2_ms * 3.6

-- Lean theorem statement to prove the speed of the faster train is 180 km/h
theorem speed_of_faster_train_is_180 :
  speed_train2_kmh = 180 := 
sorry

end speed_of_faster_train_is_180_l637_637964


namespace tan_sum_l637_637060

theorem tan_sum : (tan (α : ℝ) = 2) → (tan (β : ℝ) = 3) → tan (α + β) = -1 :=
by
  intros hα hβ
  rw [hα, hβ]
  -- Add the steps to manipulate the equation accordingly if needed, but for now we place "sorry" to skip proof
  sorry

end tan_sum_l637_637060


namespace right_angle_times_between_5_and_6_l637_637364

/-- Prove that the times at which the hour and minute hands of a clock are at a right angle between 5 and 6 o'clock are:
- 5 hours and 10 10/11 minutes
- 5 hours and 43 7/11 minutes -/
theorem right_angle_times_between_5_and_6 :
  ∃ t₁ t₂ : ℚ, 
    5 + t₁ / 60 = 5 + 10 + 10 / 11 / 60 ∧ 
    5 + t₂ / 60 = 5 + 43 + 7 / 11 / 60 ∧ 
    (∃ t ∈ (0, 60), (t = t₁ ∨ t = t₂) ∧ 
       (6 * t - (150 + 0.5 * t) = 90 ∨ 6 * t - (150 + 0.5 * t) = -90)) :=
sorry

end right_angle_times_between_5_and_6_l637_637364


namespace total_nails_used_l637_637995

-- Define a square metal plate with side length 24
def side_length : ℕ := 24

-- Define the number of nails along each side of the square
def nails_per_side : ℕ := 25

-- The number of unique nails used is 96
theorem total_nails_used (nails_per_side = 25) : ℕ :=
  4 * nails_per_side - 4 := 96

end total_nails_used_l637_637995


namespace vectors_not_parallel_l637_637113

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, -1)

theorem vectors_not_parallel : ¬ ∃ λ : ℝ, a = (λ * b.1, λ * b.2) := by
  sorry

end vectors_not_parallel_l637_637113


namespace range_of_m_l637_637887

theorem range_of_m (m : ℝ) :
  (∀ x, |x^2 - 4 * x + m| ≤ x + 4 ↔ (-4 ≤ m ∧ m ≤ 4)) ∧
  (∀ x, (x = 0 → |0^2 - 4 * 0 + m| ≤ 0 + 4) ∧ (x = 2 → ¬(|2^2 - 4 * 2 + m| ≤ 2 + 4))) →
  (-4 ≤ m ∧ m < -2) :=
by
  sorry

end range_of_m_l637_637887


namespace approx_d_l637_637713

noncomputable def close_approx_d : ℝ :=
  let d := (69.28 * (0.004)^3 - Real.log 27) / (0.03 * Real.cos (55 * Real.pi / 180))
  d

theorem approx_d : |close_approx_d + 191.297| < 0.001 :=
  by
    -- Proof goes here.
    sorry

end approx_d_l637_637713


namespace sum_of_multiples_of_9_lt_80_eq_324_l637_637748

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def all_multiples_of_9_lt_80 : List ℕ := List.filter (λ n => n < 80) (List.range' 9 80)

noncomputable def sum_of_all_multiples_of_9_lt_80 : ℕ :=
  (all_multiples_of_9_lt_80.filter is_multiple_of_9).sum

theorem sum_of_multiples_of_9_lt_80_eq_324 : sum_of_all_multiples_of_9_lt_80 = 324 := by
  sorry

end sum_of_multiples_of_9_lt_80_eq_324_l637_637748


namespace geometric_series_sum_l637_637829

-- Define the conditions of the problem
variables {a : ℕ → ℝ}

-- Assume the sequence is geometric, i.e., for all n, a_{n+1} = a_{n} * r
def is_geometric_sequence (r : ℝ) : Prop :=
  ∀ n : ℕ, a n.succ = a n * r 

-- Given condition: a_2 * a_6 + 2 * a_4 * a_5 + a_1 * a_9 = 25
def condition (r : ℝ) : Prop :=
  a 1 * a 5 + 2 * a 3 * a 4 + a 0 * a 8 = 25

-- The goal is to prove that a_4 + a_5 = 5
theorem geometric_series_sum (r : ℝ) (h1 : is_geometric_sequence r) (h2 : condition r) : 
  a 3 + a 4 = 5 :=
sorry -- Proof to be filled in

end geometric_series_sum_l637_637829


namespace sqrt_sum_inequality_l637_637754

open Real

namespace InequalityProof

variables {n : ℕ} (x : Fin n → ℝ)

theorem sqrt_sum_inequality (h1 : ∀ i, 0 ≤ x i) (h2 : (∑ i, x i) = 100) : 
  10 ≤ ∑ i, sqrt (x i) ∧ ∑ i, sqrt (x i) ≤ 10 * sqrt n := 
by
  sorry

end InequalityProof

end sqrt_sum_inequality_l637_637754


namespace last_number_odd_l637_637876

theorem last_number_odd (n : ℕ) (hn : n % 2 = 1) (h_pos : 0 < n) :
  (∃ m, m ∈ {1, 2, ..., 2 * n} ∧ ∀ (a b ∈ {1, 2, ..., 2 * n}), |a - b| ∈ {1, 2, ..., 2 * n} ∧
  (∑ k in {1, 2, ..., 2 * n}, k) % 2 = 1) → (∃ k ∈ {...}, k % 2 = 1) :=
sorry

end last_number_odd_l637_637876


namespace bankers_discount_is_270_l637_637588

noncomputable def bank_discount (BG r t : ℝ) : ℝ :=
  let BD := 540 / 2
  BD

theorem bankers_discount_is_270 (BG r t : ℝ) (h_BG : BG = 270) (h_r : r = 0.12) (h_t : t = 5) :
  bank_discount BG r t = 270 :=
by
  sorry

end bankers_discount_is_270_l637_637588


namespace smallest_positive_period_of_f_l637_637747

def f (x : ℝ) : ℝ := Real.sin x * Real.cos x * Real.cos (2 * x)

theorem smallest_positive_period_of_f : Function.periodic f (Real.pi / 2) := sorry

end smallest_positive_period_of_f_l637_637747


namespace number_of_friends_l637_637169

theorem number_of_friends (n : ℕ) : 
  (5 * (n + 1)) * 0.20 = 8 ∧ (8 / 2) = 4 → n = 7 :=
by
  intro h
  sorry

end number_of_friends_l637_637169


namespace fifteenth_digit_sum_l637_637967

theorem fifteenth_digit_sum (d₁ d₂ : ℕ → ℕ) 
  (h₁ : ∀ n, d₁ n = if n = 0 then 1 else 0) 
  (h₂ : ∀ n, d₂ n = if n % 2 = 0 then 0 else 9) :
  let sum_digit := λ n, (d₁ n + d₂ n) % 10 in
  sum_digit 14 = 1 :=
by 
  sorry

end fifteenth_digit_sum_l637_637967


namespace club_officer_selections_l637_637672

theorem club_officer_selections :
  let n := 12 in
  let positions := 5 in
  ∏ k in finset.range positions, (n - k) = 95_040 :=
by
  -- Proof omitted
  sorry

end club_officer_selections_l637_637672


namespace Nathans_score_l637_637556

def total_students : ℕ := 18
def avg_score_17 : ℕ := 84
def avg_score_18 : ℕ := 87

theorem Nathans_score :
  let total_score_17 := 17 * avg_score_17 in
  let total_score_18 := total_students * avg_score_18 in
  let Nathans_score := total_score_18 - total_score_17 in
  Nathans_score = 138 :=
by
  sorry

end Nathans_score_l637_637556


namespace sum_of_roots_eq_3_l637_637725

-- Define the polynomial equation
def polynomial (x : ℝ) := 3 * x^3 - 9 * x^2 - 42 * x - 5

-- State the theorem that the sum of the roots of the polynomial equals 3
theorem sum_of_roots_eq_3 : 
  let roots := (r s t : ℝ) in 
    polynomial r = 0 ∧ polynomial s = 0 ∧ polynomial t = 0 ∧ (r + s + t = 3) := 
by 
  sorry

end sum_of_roots_eq_3_l637_637725


namespace eccentricity_of_hyperbola_l637_637454

-- Definitions and conditions
def hyperbola (x y a b : ℝ) : Prop :=
  (a > 0 ∧ b > 0 ∧ a > b) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def regular_hexagon_side_length (a b c : ℝ) : Prop :=
  2 * a = (Real.sqrt 3 + 1) * c

-- Goal: Prove the eccentricity of the hyperbola
theorem eccentricity_of_hyperbola (a b c : ℝ) (x y : ℝ) :
  hyperbola x y a b →
  regular_hexagon_side_length a b c →
  2 * a = (Real.sqrt 3 + 1) * c →
  c ≠ 0 →
  a ≠ 0 →
  b ≠ 0 →
  (c / a = Real.sqrt 3 + 1) :=
by
  intros h_hyp h_hex h_eq h_c_ne_zero h_a_ne_zero h_b_ne_zero
  sorry -- Proof goes here

end eccentricity_of_hyperbola_l637_637454


namespace choosing_non_coplanar_points_l637_637950

/-- The number of ways to choose 4 non-coplanar points from the set
    of vertices and midpoints of the edges of a tetrahedron is 141. -/
theorem choosing_non_coplanar_points :
  ∃ (points : Finset (Fin 10)), points.card = 10 ∧ 
  (Finset.card {s : Finset (Fin 10) | s.card = 4 ∧ ¬(∃ p : Plane ℝ, ∀ x ∈ s, x ∈ p)}) = 141 := sorry

end choosing_non_coplanar_points_l637_637950


namespace coefficient_third_term_expansion_l637_637302

theorem coefficient_third_term_expansion :
  let expansion (expr : ℚ[X]) : list ℚ := expr.coeffs in
  let (1 - X) * (1 + 2 * X)^5 = (1 - X) * (1 + 2 * X)^5 in
  -- Check that the coefficient of X^2 (the third term in the expansion) is 30
  expansion ((1 - X) * (1 + 2 * X)^5).nat_degree = 30 :=
begin
  -- Here we would normally carry out the proof, but we skip it as per the instructions
  sorry
end

end coefficient_third_term_expansion_l637_637302


namespace probability_x_y_lt_3_l637_637338

/-- A point (x, y) is randomly and uniformly chosen inside the square with vertices (0,0), (0,2), (2,2), and (2,0). 
Prove the probability that x + y < 3 is 7/8. -/
theorem probability_x_y_lt_3 : 
  let X := set_of (λ p : ℝ × ℝ, 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2),
      A := set_of (λ p : ℝ × ℝ, 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ p.1 + p.2 < 3),
      area := λ s, integral (indicator s (λ _, (1 : ℝ))),
      p_A := area A / area X in
  p_A = (7:ℝ) / 8 :=
by
  sorry


end probability_x_y_lt_3_l637_637338


namespace range_of_x_for_sqrt_l637_637602

theorem range_of_x_for_sqrt {x : ℝ} (h : 2 - x ≥ 0) : x ≤ 2 :=
by {
  have h1 : 2 ≥ x := by linarith,
  exact h1,
}

end range_of_x_for_sqrt_l637_637602


namespace maximum_value_sqrt_201_exists_achieves_maximum_value_l637_637184

theorem maximum_value_sqrt_201 (w x y z : ℝ) 
  (h : 9 * w^2 + 4 * x^2 + y^2 + 25 * z^2 = 1) : 
  (9 * w + 4 * x + 2 * y + 10 * z) ≤ sqrt 201 :=
sorry

theorem exists_achieves_maximum_value (h : 9 * (1 / sqrt 39)^2 + 4 * (1 / sqrt 39)^2 + (1 / sqrt 39)^2 + 25 * (1 / sqrt 39)^2 = 1) :
  9 * (1 / sqrt 39) + 4 * (1 / sqrt 39) + 2 * (1 / sqrt 39) + 10 * (1 / sqrt 39) = sqrt 201 :=
sorry

end maximum_value_sqrt_201_exists_achieves_maximum_value_l637_637184


namespace inequality_solution_set_l637_637948

theorem inequality_solution_set (x : ℝ) : (2 * x + 1 ≥ 3) ∧ (4 * x - 1 < 7) ↔ (1 ≤ x ∧ x < 2) :=
by
  sorry

end inequality_solution_set_l637_637948


namespace last_digit_322_pow_369_l637_637399

theorem last_digit_322_pow_369 : (322^369) % 10 = 2 := by
  sorry

end last_digit_322_pow_369_l637_637399


namespace polynomial_coeff_sum_l637_637530

theorem polynomial_coeff_sum (n : ℕ) 
    (b : ℕ → ℤ) 
    (h : (1 - X + X^2)^n = ∑ i in finRange (2*n + 1), b i * X^i)
    (t : ℤ := ∑ k in finRange n.succ, b (2 * k)) :
    t = 1 := 
by 
  sorry

end polynomial_coeff_sum_l637_637530


namespace jacket_initial_reduction_l637_637263

theorem jacket_initial_reduction (x : ℝ) :
  (1 - x / 100) * 1.53846 = 1 → x = 35 :=
by
  sorry

end jacket_initial_reduction_l637_637263


namespace sufficient_but_not_necessary_l637_637427

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Arithmetic sequence condition
axiom arith_seq (n : ℕ) (d : ℝ) (a_1 : ℝ) : ∀ n, a (n + 1) = a_1 + n * d

-- Sum of the first n terms
axiom arith_sum (n : ℕ) (d : ℝ) (a_1 : ℝ) : S n = n * a_1 + n * (n-1) / 2 * d

-- Given condition 3a_3 = a_6 + 4
axiom seq_condition (d : ℝ) (a_2 : ℝ) : 3 * (a_2 + d) = (a_2 + 4 * d) + 4

-- Predicates for the necessary and sufficient conditions
def S5_less_than_10 (a_2 : ℝ) (d : ℝ) : Prop :=
  S 5 < 10

def a2_less_than_1 (a_2 : ℝ) : Prop :=
  a_2 < 1

theorem sufficient_but_not_necessary
  (a_2 : ℝ) (d : ℝ) (h1 : seq_condition d a_2) (h2 : a2_less_than_1 a_2) :
  S5_less_than_10 a_2 d :=
sorry

end sufficient_but_not_necessary_l637_637427


namespace find_f_l637_637220

noncomputable def f (x : ℝ) : ℝ := Math.cos (Real.pi * x)

theorem find_f :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, x ∈ Icc (1/2 : ℝ) 1 → y ∈ Icc (1/2 : ℝ) 1 → x < y → f x > f y) ∧
  (∀ x, f x + f (1 - x) = 0) :=
by
  sorry

end find_f_l637_637220


namespace trajectory_of_M_l637_637767

theorem trajectory_of_M (x y t : ℝ) (M P F : ℝ × ℝ)
    (hF : F = (1, 0))
    (hP : P = (1/4 * t^2, t))
    (hFP : (P.1 - F.1, P.2 - F.2) = (1/4 * t^2 - 1, t))
    (hFM : (M.1 - F.1, M.2 - F.2) = (x - 1, y))
    (hFP_FM : (P.1 - F.1, P.2 - F.2) = (2 * (M.1 - F.1), 2 * (M.2 - F.2))) :
  y^2 = 2 * x - 1 :=
by
  sorry

end trajectory_of_M_l637_637767


namespace arithmetic_sequence_problem_l637_637763

noncomputable def arithmetic_seq_gen_term_and_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  (∀ n, a n = n + 1) ∧ (∀ n, S n = (n * (n + 3)) / 2)

noncomputable def reciprocal_seq_sum (S : ℕ → ℕ) (n : ℕ) : ℚ :=
  ∑ i in finset.range n, ((2 : ℚ) / (i * (i + 3)))

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (b : ℕ → ℚ) 
  (hyp_a4: a 4 = 5)
  (hyp_S9: S 9 = 54)
  (gen_arith_seq : arithmetic_seq_gen_term_and_sum a S) :
  (∀ n, b n = 1 / (S n)) → 
  (∑ i in finset.range n, b i) = ((11 : ℚ) / 6) - (1 / (n + 1)) - (1 / (n + 2)) - (1 / (n + 3)) :=
by
  sorry

end arithmetic_sequence_problem_l637_637763


namespace train_length_150_m_l637_637347

def speed_in_m_s (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600

def length_of_train (speed_in_m_s : ℕ) (time_s : ℕ) : ℕ :=
  speed_in_m_s * time_s

theorem train_length_150_m (speed_kmh : ℕ) (time_s : ℕ) (speed_m_s : speed_in_m_s speed_kmh = 15) (time_pass_pole : time_s = 10) : length_of_train (speed_in_m_s speed_kmh) time_s = 150 := by
  sorry

end train_length_150_m_l637_637347


namespace tracy_two_dogs_food_consumption_l637_637630

theorem tracy_two_dogs_food_consumption
  (cups_per_meal : ℝ)
  (meals_per_day : ℝ)
  (pounds_per_cup : ℝ)
  (num_dogs : ℝ) :
  cups_per_meal = 1.5 →
  meals_per_day = 3 →
  pounds_per_cup = 1 / 2.25 →
  num_dogs = 2 →
  num_dogs * (cups_per_meal * meals_per_day) * pounds_per_cup = 4 := by
  sorry

end tracy_two_dogs_food_consumption_l637_637630


namespace prize_prices_functional_relationship_min_total_cost_l637_637501

noncomputable def price_check : Prop :=
  ∃ (x y : ℝ), (x + 2 * y = 64) ∧ (2 * x + y = 56) ∧ (x = 16) ∧ (y = 24)

noncomputable def cost_function : Prop :=
  ∀ (a : ℝ), (0 < a) ∧ (a < 80) → (∃ (w : ℝ), w = -8 * a + 1920)

noncomputable def minimum_cost : Prop :=
  ∃ (a : ℝ), (a ≤ 60) ∧ (w = -8 * a + 1920) → (w = 1440)

theorem prize_prices : price_check :=
sorry

theorem functional_relationship : cost_function :=
sorry

theorem min_total_cost : minimum_cost :=
sorry

end prize_prices_functional_relationship_min_total_cost_l637_637501


namespace quadratic_function_props_l637_637425

theorem quadratic_function_props (a b m : ℝ) (h_a : a ≠ 0) (h_point : 2 = a * m^2 - b * m)
  (h_range : ∀ x : ℝ, (x ≤ n - 1 ∨ x ≥ -3 - n) → a * x^2 - b * x ≥ -2 / 3) :
  (b = 4 * a ∧ x = -2 ∧ (1 : ℝ = m ∨ 0 < m ∧ m ≤ 2 ∨ -6 ≤ m ∧ m < -4)) :=
by
  sorry

end quadratic_function_props_l637_637425


namespace vertex_of_parabola_l637_637271

theorem vertex_of_parabola (c d : ℝ)
  (h : ∀ x : ℝ, -x^2 + c * x + d ≤ 0 ↔ x ∈ set.Icc (-7) 1 ∨ x ∈ set.Ici 3)
  (h_roots : (-7:ℝ) ≠ 1):
  ∃ (vx vy : ℝ), vx = -3 ∧ vy = 16 := 
by
  sorry

end vertex_of_parabola_l637_637271


namespace sum_first_10_terms_l637_637088

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∀ n, a (n + 1) - a n = a 1 - a 0

-- The given condition: the sum of a₅ and a₆ is 28
def condition (a : ℕ → ℤ) : Prop := a 5 + a 6 = 28

-- Define the function that computes the sum of the first n terms of the sequence
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ := (finset.range n).sum a

theorem sum_first_10_terms 
  (a : ℕ → ℤ) 
  (h_arith : is_arithmetic_sequence a)
  (h_cond : condition a) :
  sum_first_n_terms a 10 = 140 :=
by
  sorry

end sum_first_10_terms_l637_637088


namespace compare_exponents_l637_637420

theorem compare_exponents (n : ℕ) (hn : n > 8) :
  let a := Real.sqrt n
  let b := Real.sqrt (n + 1)
  a^b > b^a :=
sorry

end compare_exponents_l637_637420


namespace problem_statement_l637_637174

-- Define the conditions
def is_permutation (l1 l2 : List ℕ) : Prop :=
  l1 ~ l2

-- Define the maximum sum function
def max_sum (l : List ℕ) : ℕ :=
  l.zip (l.tail ++ [l.head]).map (λ x, x.1 * x.2).sum

-- Define M and N.
def M : ℕ := 76
def N : ℕ := 12

-- Define the problem statement
theorem problem_statement (l : List ℕ) (h : is_permutation l [1, 2, 3, 4, 5, 6])
  (h_m : max_sum l = M) (h_n : (l.permutations.count (λ l', max_sum l' = M)) = N) :
  M + N = 88 :=
by
  sorry

end problem_statement_l637_637174


namespace count_p_element_subsets_l637_637549

theorem count_p_element_subsets (p : ℕ) (h_odd_prime : Nat.Prime p ∧ p % 2 = 1) :
  let S := Finset.range (2 * p) + 1 in
  let p_element_subsets := Finset.filter (λ A, A.card = p) (Finset.powerset S) in
  let valid_subsets := Finset.filter (λ A, A.sum % p = 0) p_element_subsets in
  valid_subsets.card = 2 + (Nat.choose (2 * p) p - 2) / p :=
sorry

end count_p_element_subsets_l637_637549


namespace number_on_board_is_91_l637_637616

-- Definitions based on conditions
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def ends_in_digit (d n : ℕ) : Prop := n % 10 = d

def divisible_by (m n : ℕ) : Prop := n % m = 0

def andrey_statements (n : ℕ) : Prop :=
  (ends_in_digit 6 n ∨ divisible_by 7 n) ∧ ¬(ends_in_digit 6 n ∧ divisible_by 7 n)

def borya_statements (n : ℕ) : Prop :=
  (n > 26 ∨ ends_in_digit 8 n) ∧ ¬(n > 26 ∧ ends_in_digit 8 n)

def sasha_statements (n : ℕ) : Prop :=
  (divisible_by 13 n ∨ n < 27) ∧ ¬(divisible_by 13 n ∧ n < 27)

-- Mathematical equivalent proof problem
theorem number_on_board_is_91 (n : ℕ) :
  is_two_digit n →
  andrey_statements n →
  borya_statements n →
  sasha_statements n →
  n = 91 :=
by {
  intro _ _ _ _,
  -- Proof goes here, skipped with sorry
  sorry
}

end number_on_board_is_91_l637_637616


namespace greatest_multiple_of_30_less_than_800_l637_637977

theorem greatest_multiple_of_30_less_than_800 : 
    ∃ n : ℤ, (n % 30 = 0) ∧ (n < 800) ∧ (∀ m : ℤ, (m % 30 = 0) ∧ (m < 800) → m ≤ n) ∧ n = 780 :=
by
  sorry

end greatest_multiple_of_30_less_than_800_l637_637977


namespace smallest_positive_integer_condition_l637_637402

theorem smallest_positive_integer_condition :
  ∃ (x : ℕ), (∃ (d n : ℕ), d = 9 ∧ x = 10 * d + n ∧ 10 * 9 = 18 * n) ∧ x = 95 :=
by
  existsi 95
  use 9, 5
  split; try {norm_num}; sorry

end smallest_positive_integer_condition_l637_637402


namespace hyperbola_triangle_perimeter_l637_637013

-- Define the hyperbola
noncomputable def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

-- Define the foci and points P and Q
variables (F1 F2 P Q : ℝ × ℝ)
variables (l : set (ℝ × ℝ))

-- Assumptions based on the problem
def is_left_focus (F1 : ℝ × ℝ) : Prop := F1 = (-2, 0)
def is_right_focus (F2 : ℝ × ℝ) : Prop := F2 = (2, 0)
def intersects_hyperbola (P Q : ℝ × ℝ) : Prop := hyperbola_eq P.1 P.2 ∧ hyperbola_eq Q.1 Q.2 ∧ l P ∧ l Q
def distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

variable H_F1 : is_left_focus F1
variable H_F2 : is_right_focus F2
variable H_line : intersects_hyperbola P Q
variable H_distance_PQ : distance P Q = 4

-- The perimeter of triangle PF_2Q
theorem hyperbola_triangle_perimeter : distance P F2 + distance Q F2 + distance P Q = 12 :=
sorry

end hyperbola_triangle_perimeter_l637_637013


namespace find_f_at_9_over_2_l637_637181

variable (f : ℝ → ℝ)

-- Domain of f is ℝ
axiom domain_f : ∀ x : ℝ, f x = f x

-- f(x+1) is an odd function
axiom odd_f : ∀ x : ℝ, f (x + 1) = -f (-(x - 1))

-- f(x+2) is an even function
axiom even_f : ∀ x : ℝ, f (x + 2) = f (-(x - 2))

-- When x is in [1,2], f(x) = ax^2 + b
variables (a b : ℝ)
axiom on_interval : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b

-- f(0) + f(3) = 6
axiom sum_f : f 0 + f 3 = 6 

theorem find_f_at_9_over_2 : f (9/2) = 5/2 := 
by sorry

end find_f_at_9_over_2_l637_637181


namespace polygon_sides_and_diagonals_l637_637442

theorem polygon_sides_and_diagonals (n : ℕ) (h : (n-2) * 180 / 360 = 13 / 2) : 
  n = 15 ∧ (n * (n - 3) / 2 = 90) :=
by {
  sorry
}

end polygon_sides_and_diagonals_l637_637442


namespace proof_problem_l637_637764

universe u

variable {α : Type u}

def set_A {x : α} [decidable_eq α] : set α := {0, |x|}
def set_B [decidable_eq α] : set α := {1, 0, -1}

theorem proof_problem {x : α} [decidable_eq α] (hA_sub_B : set_A ⊆ set_B) :
  set_A = {0, 1} ∧
  set_A ∩ set_B = {0, 1} ∧
  set_A ∪ set_B = {-1, 0, 1} ∧
  set_B \ set_A = {-1} :=
by
  sorry

end proof_problem_l637_637764


namespace speed_of_man_l637_637668

theorem speed_of_man 
  (L : ℝ) 
  (V_t : ℝ) 
  (T : ℝ) 
  (conversion_factor : ℝ) 
  (kmph_to_mps : ℝ → ℝ)
  (final_conversion : ℝ → ℝ) 
  (relative_speed : ℝ) 
  (Vm : ℝ) : Prop := 
L = 220 ∧ V_t = 59 ∧ T = 12 ∧ 
conversion_factor = 1000 / 3600 ∧ 
kmph_to_mps V_t = V_t * conversion_factor ∧ 
relative_speed = L / T ∧ 
Vm = relative_speed - (kmph_to_mps V_t) ∧ 
final_conversion Vm = Vm * 3.6 ∧ 
final_conversion Vm = 6.984

end speed_of_man_l637_637668


namespace exp_inequality_l637_637903

theorem exp_inequality (n : ℕ) (h : 0 < n) : 2 ≤ (1 + 1 / (n : ℝ)) ^ n ∧ (1 + 1 / (n : ℝ)) ^ n < 3 :=
sorry

end exp_inequality_l637_637903


namespace digit_15_of_sum_reciprocals_l637_637972

/-- 
What is the 15th digit after the decimal point of the sum of the decimal equivalents 
for the fractions 1/9 and 1/11?
-/
theorem digit_15_of_sum_reciprocals :
  let r := (1/9 + 1/11) in
  let d15 := Real.frac (10^(15:ℕ) * r) in
  Int.floor (10 * d15) = 1 :=
by
  let r := (1/9 + 1/11)
  let d15 := Real.frac (10^(15:ℕ) * r)
  have h : Real.toRat d15 = 1 / 10 + d15 - Int.floor d15
  have : Real.frac (10^(15:ℕ) * r) = r
  sorry

end digit_15_of_sum_reciprocals_l637_637972


namespace arithmetic_seq_a7_geometric_seq_b6_geometric_common_ratio_l637_637773

noncomputable def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

noncomputable def geometric_seq (b₁ q : ℕ) (n : ℕ) : ℕ :=
  b₁ * q^(n - 1)

theorem arithmetic_seq_a7 
  (a₁ d : ℕ)
  (h : 2 * arithmetic_seq a₁ d 5 - arithmetic_seq a₁ d 3 = 3) :
  arithmetic_seq a₁ d 7 = 3 :=
sorry

theorem geometric_seq_b6 
  (b₁ q : ℕ)
  (h1 : geometric_seq b₁ q 2 = 1)
  (h2 : geometric_seq b₁ q 4 = 4) :
  geometric_seq b₁ q 6 = 16 :=
sorry

theorem geometric_common_ratio 
  (b₁ q : ℕ)
  (h1 : geometric_seq b₁ q 2 = 1)
  (h2 : geometric_seq b₁ q 4 = 4) :
  q = 2 ∨ q = -2 :=
sorry

end arithmetic_seq_a7_geometric_seq_b6_geometric_common_ratio_l637_637773


namespace probability_of_odd_multiple_of_6_or_8_l637_637233
open Set

def is_multiple_of (k n : ℕ) : Prop := n % k = 0

def is_odd (n : ℕ) : Prop := n % 2 = 1

def filtered_balls (M : Set ℕ) (P : ℕ → Prop) : Set ℕ := {n ∈ M | P n}

theorem probability_of_odd_multiple_of_6_or_8 (balls : Set ℕ) :
  (∀ x, x ∈ balls ↔ x ∈ range 1 61) →
  let M6 := {n ∈ balls | is_multiple_of 6 n}
  let M8 := {n ∈ balls | is_multiple_of 8 n}
  let M := M6 ∪ M8
  (filtered_balls M is_odd).card = 0 := by
  by_contradiction
  sorry

end probability_of_odd_multiple_of_6_or_8_l637_637233


namespace valentines_count_l637_637893

theorem valentines_count (x y : ℕ) (h1 : (x = 2 ∧ y = 48) ∨ (x = 48 ∧ y = 2)) : 
  x * y - (x + y) = 46 := by
  sorry

end valentines_count_l637_637893


namespace growth_rate_yield_per_acre_l637_637326

theorem growth_rate_yield_per_acre (x : ℝ) (a_i y_i y_f : ℝ) (h1 : a_i = 5) (h2 : y_i = 10000) (h3 : y_f = 30000) 
  (h4 : y_f = 5 * (1 + 2 * x) * (y_i / a_i) * (1 + x)) : x = 0.5 := 
by
  -- Insert the proof here
  sorry

end growth_rate_yield_per_acre_l637_637326


namespace interest_rate_difference_l637_637346

theorem interest_rate_difference:
  ∀ (R H: ℝ),
    (300 * (H / 100) * 5 = 300 * (R / 100) * 5 + 90) →
    (H - R = 6) :=
by
  intros R H h
  sorry

end interest_rate_difference_l637_637346


namespace sum_of_digits_of_roots_in_interval_l637_637715

noncomputable def equation_roots_sum_digits : Nat :=
  let equation := λ x : Real => 4 * sin(2 * x) + 3 * cos(2 * x) - 2 * sin(x) - 4 * cos(x) + 1
  let interval_start := 10 ^ (factorial 2014) * Real.pi
  let interval_end := 10 ^ (factorial 2014 + 2015) * Real.pi
  let number_of_roots := 2 * 10 ^ (factorial 2014) * (10 ^ 2015 - 1)
  let roots_sum := 1 + 8 + 9 * 2014
  roots_sum = 18135

theorem sum_of_digits_of_roots_in_interval : equation_roots_sum_digits = 18135 := by
  sorry

end sum_of_digits_of_roots_in_interval_l637_637715


namespace maximize_sequence_l637_637539

theorem maximize_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 1 - (S n) ^ 2) ∧ (∀ n, S n = ∑ i in finset.range n, a (i + 1)) →
  ∃ n, ∀ m, n ≠ 3 → (n * (S n) ^ 2 / (1 + 10 * (S n) ^ 2) > m * (S m) ^ 2 / (1 + 10 * (S m) ^ 2)) :=
sorry

end maximize_sequence_l637_637539


namespace negation_proposition_iff_l637_637599

-- Define propositions and their components
def P (x : ℝ) : Prop := x > 1
def Q (x : ℝ) : Prop := x^2 > 1

-- State the proof problem
theorem negation_proposition_iff (x : ℝ) : ¬ (P x → Q x) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by 
  sorry

end negation_proposition_iff_l637_637599


namespace number_of_arcs_no_more_than_120_l637_637619

-- Definition of the condition: 21 points on a circle
variable (points : Finset Point) (circle : Circle)

-- Given that the number of points on the circle is 21
hypothesis h1 : points.card = 21

-- Function to calculate the arc measure between two points on a circle
def arc_measure (a b : Point) : ℝ := -- definition of the arc measure

-- Predicate to check if an arc measure is no more than 120 degrees
def arc_no_more_than_120 (a b : Point) : Prop := arc_measure a b ≤ 120

-- The main theorem statement
theorem number_of_arcs_no_more_than_120 :
  (∑ p1 in points, ∑ p2 in (points.erase p1), if arc_no_more_than_120 p1 p2 then 1 else 0) / 2 = 100 :=
begin
  sorry
end

end number_of_arcs_no_more_than_120_l637_637619


namespace total_signals_l637_637205

theorem total_signals (l : ℕ) (n : ℕ) (adj : ℕ → Prop) (colors : finset ℕ) :
  l = 6 → n = 3 → (∀ (i : ℕ), adj i ↔ (i = 0 ∨ i = 1 ∨ i = 2)) → (colors = {1, 2, 3}) → 
  (∑ (i : ℕ) in finset.range l, if adj i then 1 else 0) * (card colors) ^ n = 324 :=
by
  sorry

end total_signals_l637_637205


namespace perpendicular_slope_is_five_ninths_l637_637637

-- Define the points
def P1 : ℝ × ℝ := (3, -4)
def P2 : ℝ × ℝ := (-2, 5)

-- Calculate the slope of the line containing the points
def slope (P1 P2 : ℝ × ℝ) : ℝ :=
  (P2.2 - P1.2) / (P2.1 - P1.1)

-- Define the condition for a perpendicular slope
def perpendicular_slope (m : ℝ) : ℝ :=
  -1 / m

-- The theorem we want to prove
theorem perpendicular_slope_is_five_ninths :
  perpendicular_slope (slope P1 P2) = 5 / 9 :=
by
  sorry

end perpendicular_slope_is_five_ninths_l637_637637


namespace tan_half_angle_of_alpha_l637_637434

variable (α : Real)
variable (h1 : Real.sin α = 4 / 5)
variable (h2 : α > Real.pi / 2 ∧ α < Real.pi)

theorem tan_half_angle_of_alpha (h_sin : Real.sin α = 4 / 5) (h_interval : α ∈ Ioo (Real.pi / 2) Real.pi) :
  Real.tan (α / 2) = 2 := sorry

end tan_half_angle_of_alpha_l637_637434


namespace area_of_region_R_l637_637911

-- Define the parameters of the rhombus
def Rhombus (ABCD : Type) := 
  (side_length : ℝ) 
  (angle_at_B : ℝ)
  (area_of_R : ℝ)

-- State the theorem
theorem area_of_region_R :
  ∀ (ABCD : Type) (side_length : ℝ) (angle_at_B : ℝ), 
  side_length = 2 ∧ angle_at_B = 120 → ∃ (area_of_R : ℝ), area_of_R = 2 * sqrt 3 / 3 :=
by
  intro ABCD side_length angle_at_B h,
  cases h with h_side h_angle,
  use 2 * sqrt ３ / 3,
  sorry

end area_of_region_R_l637_637911


namespace quadratic_solution_l637_637949

theorem quadratic_solution (x : ℝ) (h : 2 * x ^ 2 - 2 = 0) : x = 1 ∨ x = -1 :=
sorry

end quadratic_solution_l637_637949


namespace annual_income_correct_l637_637289

-- Define the principal amounts and interest rates
def principal_1 : ℝ := 3000
def rate_1 : ℝ := 0.085

def principal_2 : ℝ := 5000
def rate_2 : ℝ := 0.064

-- Define the interest calculations for each investment
def interest_1 : ℝ := principal_1 * rate_1
def interest_2 : ℝ := principal_2 * rate_2

-- Define the total annual income
def total_annual_income : ℝ := interest_1 + interest_2

-- Proof statement
theorem annual_income_correct : total_annual_income = 575 :=
by
  sorry

end annual_income_correct_l637_637289


namespace sum_of_coeffs_factorized_form_l637_637032

theorem sum_of_coeffs_factorized_form (x y : ℕ) : 
  let a := 3 * x^2 
  let b := 8 * y^2 
  27 * x^6 - 512 * y^6 = (a - b) * (a^2 + a * b + b^2) →
  (a - b) * (a^2 + a * b + b^2) = 3*x^2 - 8*y^2 * (9*x^4 + 24*x^2*y^2 + 64*y^4) →
  3 + (-8) + 9 + 24 + 64 = 92 :=
begin
  intros,
  sorry
end

end sum_of_coeffs_factorized_form_l637_637032


namespace total_payment_correct_l637_637963

-- Definitions for the conditions
def X_payment (Y_payment : ℝ) := 1.2 * Y_payment
def total_payment (X_payment Y_payment : ℝ) := X_payment + Y_payment

-- Example values given in the problem
def Y_payment : ℝ := 272.73
def X_payment_rounded : ℝ := Real.round (X_payment Y_payment * 100) / 100

-- Statement of the problem
theorem total_payment_correct :
  total_payment X_payment_rounded Y_payment = 600.01 :=
by
  -- proof steps can be added here if necessary
  sorry

end total_payment_correct_l637_637963


namespace modulus_of_z_l637_637472

open Complex

noncomputable def z : ℂ := (3 - I^3) / (1 + I) - 2 * I

theorem modulus_of_z :
  let w : ℂ := z + 2 * I
  w = (3 - I^3) / (1 + I) → |z| = Real.sqrt 13 :=
by
  intro h
  sorry

end modulus_of_z_l637_637472


namespace find_b_for_continuity_l637_637885

def g (x : ℝ) (b : ℝ) : ℝ :=
  if x ≤ 4 then 4 * x^2 + 5 else b * x + 3

theorem find_b_for_continuity (b : ℝ) (H : (4 * (4 : ℝ)^2 + 5) = (b * (4 : ℝ) + 3)) : b = 16.5 :=
  sorry

end find_b_for_continuity_l637_637885


namespace smallest_k_l637_637828

theorem smallest_k (m n k : ℤ) (h : 221 * m + 247 * n + 323 * k = 2001) (hk : k > 100) : 
∃ k', k' = 111 ∧ k' > 100 :=
by
  sorry

end smallest_k_l637_637828


namespace quadratic_two_distinct_real_roots_l637_637484

theorem quadratic_two_distinct_real_roots (k : ℝ) (h : k ≠ 0) : 
  (kx : ℝ) -> (a = k) -> (b = -2) -> (c = -1) -> (b^2 - 4*a*c > 0) -> (-2)^2 - 4*k*(-1) = (4 + 4*k > 0) := sorry

end quadratic_two_distinct_real_roots_l637_637484


namespace inequality_solution_l637_637736

theorem inequality_solution (x : ℚ) :
  3/20 + |x - 7/40| < 11/40 → x ∈ set.Ioo (1/20 : ℚ) (3/10 : ℚ) :=
by {
  sorry
}

end inequality_solution_l637_637736


namespace assign_students_to_villages_l637_637047

theorem assign_students_to_villages (n m : ℕ) (hn : n = 5) (hm : m = 3) :
  ∃ N : ℕ, N = 70 ∧ 
  (∃ (f : Fin n → Fin m), (∀ i j, f i = f j ↔ i = j) ∧ 
  (∀ x : Fin m, ∃ y : Fin n, f y = x)) :=
by
  sorry

end assign_students_to_villages_l637_637047


namespace maximize_profit_l637_637674

variables (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ)

-- Definitions for the conditions
def nonneg_x := (0 ≤ x)
def nonneg_y := (0 ≤ y)
def constraint1 := (a1 * x + a2 * y ≤ c1)
def constraint2 := (b1 * x + b2 * y ≤ c2)
def profit := (z = d1 * x + d2 * y)

-- Proof of constraints and profit condition
theorem maximize_profit (a1 a2 b1 b2 d1 d2 c1 c2 x y z : ℝ) :
    nonneg_x x ∧ nonneg_y y ∧ constraint1 a1 a2 c1 x y ∧ constraint2 b1 b2 c2 x y → profit d1 d2 x y z :=
by
  sorry

end maximize_profit_l637_637674


namespace part1_part2_l637_637105

-- Part 1: Prove that the range of values for k is k ≤ 1/4
theorem part1 (f : ℝ → ℝ) (k : ℝ) 
  (h1 : ∀ x0 : ℝ, f x0 ≥ |k+3| - |k-2|)
  (h2 : ∀ x : ℝ, f x = |2*x - 1| + |x - 2| ) : 
  k ≤ 1/4 := 
sorry

-- Part 2: Show that the minimum value of m+n is 8/3
theorem part2 (f : ℝ → ℝ) (m n : ℝ) 
  (h1 : ∀ x : ℝ, f x ≥ 1/m + 1/n)
  (h2 : ∀ x : ℝ, f x = |2*x - 1| + |x - 2| ) : 
  m + n ≥ 8/3 := 
sorry

end part1_part2_l637_637105


namespace part1_part2_case1_part2_case2_part2_case3_1_part2_case3_2_part2_case3_3_l637_637874

def f (a x : ℝ) : ℝ := a * x ^ 2 + (1 - a) * x + a - 2

theorem part1 (a : ℝ) : (∀ x : ℝ, f a x ≥ -2) ↔ a ≥ 1/3 :=
sorry

theorem part2_case1 (a : ℝ) (ha : a = 0) : ∀ x : ℝ, f a x < a - 1 ↔ x < 1 :=
sorry

theorem part2_case2 (a : ℝ) (ha : a > 0) : ∀ x : ℝ, (f a x < a - 1) ↔ (-1 / a < x ∧ x < 1) :=
sorry

theorem part2_case3_1 (a : ℝ) (ha : a = -1) : ∀ x : ℝ, (f a x < a - 1) ↔ x ≠ 1 :=
sorry

theorem part2_case3_2 (a : ℝ) (ha : -1 < a ∧ a < 0) : ∀ x : ℝ, (f a x < a - 1) ↔ (x > -1 / a ∨ x < 1) :=
sorry

theorem part2_case3_3 (a : ℝ) (ha : a < -1) : ∀ x : ℝ, (f a x < a - 1) ↔ (x > 1 ∨ x < -1 / a) :=
sorry

end part1_part2_case1_part2_case2_part2_case3_1_part2_case3_2_part2_case3_3_l637_637874


namespace partitionPythagoreanTriples_l637_637569

-- Definition of a Pythagorean triple
def isPythagoreanTriple (a b c : ℕ) : Prop :=
a^2 + b^2 = c^2 ∧ a < b ∧ b < c

-- Lemma: If (a, b, c) is a Pythagorean triple, then (2a, 2b, 2c) and (2a-1, 2b-1, 2c-1) are also Pythagorean triples
lemma derivedPythagoreanTriples (a b c : ℕ) :
  isPythagoreanTriple a b c →
  isPythagoreanTriple (2*a) (2*b) (2*c) ∧
  isPythagoreanTriple (2*a-1) (2*b-1) (2*c-1) :=
sorry

-- Main theorem: Prove that for any positive integer n, the set {2, 3, ..., 3n+1} can be partitioned into n Pythagorean triples
theorem partitionPythagoreanTriples (n : ℕ) (h : 0 < n) :
  ∃ (triples : (ℕ × ℕ × ℕ) → bool), 
  (∀ t, triples t → (let (a, b, c) := t in isPythagoreanTriple a b c)) ∧
  (∀ x ∈ {2, 3, ..., 3*n+1}, ∃ t, triples t ∧ (let (a, b, c) := t in a = x ∨ b = x ∨ c = x)) :=
sorry

end partitionPythagoreanTriples_l637_637569


namespace Talia_father_age_l637_637154

def Talia_age (T : ℕ) : Prop := T + 7 = 20
def Talia_mom_age (M T : ℕ) : Prop := M = 3 * T
def Talia_father_age_in_3_years (F M : ℕ) : Prop := F + 3 = M

theorem Talia_father_age (T F M : ℕ) 
    (hT : Talia_age T)
    (hM : Talia_mom_age M T)
    (hF : Talia_father_age_in_3_years F M) :
    F = 36 :=
by 
  sorry

end Talia_father_age_l637_637154


namespace cook_stole_the_cookbook_l637_637305

-- Define the suspects
inductive Suspect
| CheshireCat
| Duchess
| Cook
deriving DecidableEq, Repr

-- Define the predicate for lying
def lied (s : Suspect) : Prop := sorry

-- Define the conditions
def conditions (thief : Suspect) : Prop :=
  lied thief ∧
  ((∀ s : Suspect, s ≠ thief → lied s) ∨ (∀ s : Suspect, s ≠ thief → ¬lied s))

-- Define the goal statement
theorem cook_stole_the_cookbook : conditions Suspect.Cook :=
sorry

end cook_stole_the_cookbook_l637_637305


namespace range_of_a_l637_637108

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 - x + (a - 4) = 0 ∧ y^2 - y + (a - 4) = 0 ∧ x > 0 ∧ y < 0) → a < 4 :=
by
  sorry

end range_of_a_l637_637108


namespace find_sin_x_l637_637765

variable (a b : ℝ) (x : ℝ)
variable (h1 : Real.Tan x = 3 * a * b / (a^2 + 3 * b^2))
variable (h2 : a > b)
variable (h3 : b > 0)
variable (h4 : 0 < x ∧ x < Real.pi / 2)

theorem find_sin_x : Real.sin x = 3 * a * b / Real.sqrt (a^4 + 15 * a^2 * b^2 + 9 * b^4) := 
by
  sorry

end find_sin_x_l637_637765


namespace probability_top_card_heart_l637_637341

def specially_designed_deck (n_cards n_ranks n_suits cards_per_suit : ℕ) : Prop :=
  n_cards = 60 ∧ n_ranks = 15 ∧ n_suits = 4 ∧ cards_per_suit = n_ranks

theorem probability_top_card_heart (n_cards n_ranks n_suits cards_per_suit : ℕ)
  (h_deck : specially_designed_deck n_cards n_ranks n_suits cards_per_suit) :
  (15 / 60 : ℝ) = 1 / 4 :=
by
  sorry

end probability_top_card_heart_l637_637341


namespace alpha_beta_sum_lt_3pi_over_2_l637_637124

theorem alpha_beta_sum_lt_3pi_over_2 
  {α β : ℝ}
  (hα : α ∈ Set.Ioc (π/2) π)
  (hβ : β ∈ Set.Ioc (π/2) π)
  (hineq : Real.tan α < Real.cot β) : α + β < (3/2) * π :=
by 
  sorry

end alpha_beta_sum_lt_3pi_over_2_l637_637124


namespace sequence_solution_l637_637510

theorem sequence_solution (a : ℕ → ℝ) (h1 : a 1 = 2) (h2 : a 2 = 1)
  (h_rec : ∀ n ≥ 2, 2 / a n = 1 / a (n + 1) + 1 / a (n - 1)) :
  ∀ n, a n = 2 / n :=
by
  sorry

end sequence_solution_l637_637510


namespace relationship_p_q_l637_637878

noncomputable def p (α β : ℝ) : ℝ := Real.cos α * Real.cos β
noncomputable def q (α β : ℝ) : ℝ := Real.cos ((α + β) / 2) ^ 2

theorem relationship_p_q (α β : ℝ) : p α β ≤ q α β :=
by
  sorry

end relationship_p_q_l637_637878


namespace angle_CDE_proof_l637_637849

theorem angle_CDE_proof
    (A B C D E : Type)
    (angle_A angle_B angle_C : ℝ)
    (angle_AEB : ℝ)
    (angle_BED : ℝ)
    (angle_BDE : ℝ) :
    angle_A = 90 ∧
    angle_B = 90 ∧
    angle_C = 90 ∧
    angle_AEB = 50 ∧
    angle_BED = 2 * angle_BDE →
    ∃ angle_CDE : ℝ, angle_CDE = 70 :=
by
  sorry

end angle_CDE_proof_l637_637849


namespace negation_proof_l637_637935

theorem negation_proof :
  ¬ (∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 :=
by
  -- proof goes here
  sorry

end negation_proof_l637_637935


namespace geometric_sequence_product_l637_637495

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (a_geometric : ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r)
  (root_condition : ∃ x y : ℝ, x * y = 16 ∧ x + y = 10 ∧ a 1 = x ∧ a 19 = y) :
  a 8 * a 10 * a 12 = 64 :=
sorry

end geometric_sequence_product_l637_637495


namespace fifteenth_digit_sum_l637_637970

/-- The 15th digit after the decimal point of the sum of decimal equivalents of 1/9 and 1/11 is 1. -/
theorem fifteenth_digit_sum (d1 d2 : Nat) (h1 : (1/9 : Rat) = 0.1111111 -- overline 1 represents repeating 1
                    h2 : (1/11 : Rat) = 0.090909) -- overline 090909 represents repeating 090909
                   (repeating_block : String := "10")
                    : repeating_block[15 % 2] = '1' := -- finding the 15th digit
by
  sorry

end fifteenth_digit_sum_l637_637970


namespace prime_odd_sum_l637_637437

theorem prime_odd_sum (x y : ℕ) (h_prime : Prime x) (h_odd : y % 2 = 1) (h_eq : x^2 + y = 2005) : x + y = 2003 :=
by
  sorry

end prime_odd_sum_l637_637437


namespace count_valid_sequences_l637_637175

-- Definitions of vertices of the triangle T
def T : Set (ℝ × ℝ) := {(0, 0), (6, 0), (0, 4)}

-- Definition of transformations
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def rotate270 (p : ℝ × ℝ) : ℝ × ℝ := (p.2, -p.1)
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)
def reflect_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)

-- Check if triangle T returns to its original position
def returns_to_original (seq : List (ℝ × ℝ → ℝ × ℝ)) : Bool :=
  let apply_seq := seq.foldl (λ current tr, current ∘ tr) id
  apply_seq '' T = T

-- Main theorem
theorem count_valid_sequences : ∃ (seqs : Finset (List (ℝ × ℝ → ℝ × ℝ))), 
  seqs.card = 3 ∧ ∀ seq ∈ seqs, returns_to_original seq :=
sorry

end count_valid_sequences_l637_637175


namespace fraction_simplest_form_l637_637643

def fracA (a b : ℤ) : ℤ × ℤ := (|2 * a|, 5 * a^2 * b)
def fracB (a : ℤ) : ℤ × ℤ := (a, a^2 - 2 * a)
def fracC (a b : ℤ) : ℤ × ℤ := (3 * a + b, a + b)
def fracD (a b : ℤ) : ℤ × ℤ := (a^2 - a * b, a^2 - b^2)

theorem fraction_simplest_form (a b : ℤ) : (fracC a b).1 / (fracC a b).2 = (3 * a + b) / (a + b) :=
by sorry

end fraction_simplest_form_l637_637643


namespace problem_proof_l637_637146

variable (A B C a b c : ℝ)
variable (ABC_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variable (sides_opposite : a = (b * sin A / sin B) ∧ b = (a * sin B / sin A))
variable (cos_eq : b + b * cos A = a * cos B)

theorem problem_proof :
  (A = 2 * B ∧ (π / 6 < B ∧ B < π / 4) ∧ a^2 = b^2 + b * c) :=
  sorry

end problem_proof_l637_637146


namespace range_of_a_l637_637430

variable {a b c d : ℝ}

theorem range_of_a (h1 : a + b + c + d = 3) (h2 : a^2 + 2 * b^2 + 3 * c^2 + 6 * d^2 = 5) : 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l637_637430


namespace four_digit_numbers_divisible_by_5_l637_637810

theorem four_digit_numbers_divisible_by_5 : 
  let smallest_4_digit := 1000
  let largest_4_digit := 9999
  let divisible_by_5 (n : ℕ) := ∃ k : ℕ, n = 5 * k
  ∃ n : ℕ, ( ∀ x : ℕ, smallest_4_digit ≤ x ∧ x ≤ largest_4_digit ∧ divisible_by_5 x ↔ (smallest_4_digit + (n-1) * 5 = x) ) ∧ n = 1800 :=
by
  sorry

end four_digit_numbers_divisible_by_5_l637_637810


namespace xiao_yings_score_l637_637509

theorem xiao_yings_score
    (average_score : ℕ)
    (xiao_yings_relative_score : ℤ)
    (xiao_yings_actual_score : ℕ) :
    average_score = 83 →
    xiao_yings_relative_score = -3 →
    xiao_yings_actual_score = average_score + xiao_yings_relative_score →
    xiao_yings_actual_score = 80 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end xiao_yings_score_l637_637509


namespace pastry_chef_eggs_eggs_per_cake_is_7_l637_637336

theorem pastry_chef_eggs :
  ∃ (n : ℕ), ∃ (k : ℕ), ∃ (m : ℕ), 
    n = 43 * k ∧ n = 60 * m + 1 ∧ n < 387 ∧ n / 43 < 9 :=
by
  have ex_n_k_m : ∃ (n : ℕ), ∃ (k : ℕ), ∃ (m : ℕ), 
    n = 43 * k ∧ n = 60 * m + 1 ∧ n < 387 := sorry,
  obtain ⟨n, k, m, hn⟩ := ex_n_k_m,
  use [n, k, m],
  exact hn

theorem eggs_per_cake_is_7 :
  ∃ (n : ℕ), ∃ (k : ℕ), ∃ (m : ℕ), 
    n = 43 * k ∧ n = 60 * m + 1 ∧ n < 387 ∧ n / 43 = 7 :=
by
  have ex_n_k_m : ∃ (n : ℕ), ∃ (k : ℕ), ∃ (m : ℕ), 
    n = 43 * k ∧ n = 60 * m + 1 ∧ n < 387 ∧ n / 43 < 9 := sorry,
  obtain ⟨n, k, m, hn⟩ := ex_n_k_m,
  have eq_7 : n / 43 = 7 := sorry,
  use [n, k, m],
  exact ⟨hn.1, hn.2.1, hn.2.2.1, eq_7⟩

end pastry_chef_eggs_eggs_per_cake_is_7_l637_637336


namespace smallest_positive_integer_a_l637_637999

theorem smallest_positive_integer_a (a : ℕ) (h1 : 0 < a) (h2 : ∃ b : ℕ, 3150 * a = b^2) : a = 14 :=
by
  sorry

end smallest_positive_integer_a_l637_637999


namespace sum_of_squares_acute_sum_of_squares_obtuse_high_sum_of_squares_obtuse_low_variants_l637_637318

def is_acute (A B C : ℝ) : Prop := A < π / 2 ∧ B < π / 2 ∧ C < π / 2
def is_obtuse (A B C : ℝ) : Prop := A > π / 2 ∨ B > π / 2 ∨ C > π / 2
def obtuse_geq_high_threshold (C : ℝ) : Prop := C ≥ 2 * arctan (4 / 3)

theorem sum_of_squares_acute (A B C : ℝ) (hAngles : A + B + C = π) (hAcute : is_acute A B C) :
  let x := tan (A / 2)
  let y := tan (B / 2)
  let z := tan (C / 2)
  x * y + y * z + z * x = 1 → x^2 + y^2 + z^2 < 2 := 
sorry

theorem sum_of_squares_obtuse_high (A B C : ℝ) (hAngles : A + B + C = π)
  (hObtuse : is_obtuse A B C) (hThreshold : obtuse_geq_high_threshold C) :
  let x := tan (A / 2)
  let y := tan (B / 2)
  let z := tan (C / 2)
  x * y + y * z + z * x = 1 → x^2 + y^2 + z^2 ≥ 2 :=
sorry

theorem sum_of_squares_obtuse_low_variants (A B C: ℝ) (hAngles : A + B + C = π)
  (hObtuse : is_obtuse A B C) (hThreshold : ¬obtuse_geq_high_threshold C) :
  ∃ (x y z: ℝ), x = tan (A / 2) ∧ y = tan (B / 2) ∧ z = tan (C / 2) ∧ x * y + y * z + z * x = 1 ∧ 
  ((x^2 + y^2 + z^2 > 2) ∨ (x^2 + y^2 + z^2 < 2)) :=
sorry

end sum_of_squares_acute_sum_of_squares_obtuse_high_sum_of_squares_obtuse_low_variants_l637_637318


namespace minimum_area_of_triangle_l637_637157

def is_on_line (P : ℝ × ℝ) : Prop := P.2 = -P.1 - 2

def is_tangent_to_parabola (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t1 t2 : ℝ, A = (t1, t1^2 / 2) ∧ B = (t2, t2^2 / 2) ∧
  (y_0 = P.2) ∧ (x_0 t1 + y_0 = t1^2 / 2) ∧ (x_0 t2 + y_0 = t2^2 / 2)

def minimum_area_triangle (P A B : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((A.1 - P.1) * (B.2 - P.2) - (A.2 - P.2) * (B.1 - P.1))

theorem minimum_area_of_triangle
  (P A B : ℝ × ℝ)
  (hP_line : is_on_line P)
  (h_tangent : is_tangent_to_parabola P A B) :
  minimum_area_triangle P A B = 3 * real.sqrt 3 := sorry

end minimum_area_of_triangle_l637_637157


namespace find_k_l637_637443

theorem find_k 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (hSn : ∀ n, S n = -2 + 2 * (1 / 3) ^ n) 
  (h_geom : ∀ n, a (n + 1) = a n * a 2 / a 1) :
  k = -2 :=
sorry

end find_k_l637_637443


namespace fib_mult_eq_prod_imp_pairs_l637_637266

-- Define the Fibonacci sequence
def Fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := Fib (n + 1) + Fib n

-- The main theorem
theorem fib_mult_eq_prod_imp_pairs (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  Fib m * Fib n = m * n → (m = 1 ∧ n = 1) ∨ (m = 5 ∧ n = 5) :=
sorry

end fib_mult_eq_prod_imp_pairs_l637_637266


namespace find_shirt_cost_l637_637355

def cost_each_shirt (x : ℝ) : Prop :=
  let total_purchase_price := x + 5 + 30 + 14
  let shipping_cost := if total_purchase_price > 50 then 0.2 * total_purchase_price else 5
  let total_bill := total_purchase_price + shipping_cost
  total_bill = 102

theorem find_shirt_cost (x : ℝ) (h : cost_each_shirt x) : x = 36 :=
sorry

end find_shirt_cost_l637_637355


namespace sum_first_150_remainder_l637_637983

theorem sum_first_150_remainder :
  let S_n (n : ℕ) := n * (n + 1) / 2
  in S_n 150 % 8000 = 3325 :=
by
  let S_n (n : ℕ) := n * (n + 1) / 2
  sorry

end sum_first_150_remainder_l637_637983


namespace no_adjacent_birch_trees_probability_l637_637677

-- Define the number of trees in each category
def pines : ℕ := 2
def oaks : ℕ := 4
def birches : ℕ := 6

-- Total number of trees
def total_trees : ℕ := pines + oaks + birches

-- Total number of non-birch trees
def non_birch_trees : ℕ := pines + oaks

-- Number of slots created by non-birch trees
def slots : ℕ := non_birch_trees + 1

-- Combinatorial functions to calculate the number of ways
def choose (n k : ℕ) : ℕ := Nat.choose n k

def ways_to_place_birches := choose slots birches
def total_arrangements := choose total_trees birches

-- The probability that no two birch trees are adjacent
def probability_no_adjacent_birch : ℚ :=
  (ways_to_place_birches : ℚ) / total_arrangements

-- The simplified answer should be 1/132
theorem no_adjacent_birch_trees_probability :
  probability_no_adjacent_birch = 1 / 132 :=
begin
  sorry
end

end no_adjacent_birch_trees_probability_l637_637677


namespace archer_total_fish_caught_l637_637358

noncomputable def total_fish_caught (initial : ℕ) (second_extra : ℕ) (third_round_percentage : ℕ) : ℕ :=
  let second_round := initial + second_extra
  let total_after_two_rounds := initial + second_round
  let third_round := second_round + (third_round_percentage * second_round) / 100
  total_after_two_rounds + third_round

theorem archer_total_fish_caught :
  total_fish_caught 8 12 60 = 60 :=
by
  -- Theorem statement to prove the total fish caught equals 60 given the conditions.
  sorry

end archer_total_fish_caught_l637_637358


namespace max_min_sum_of_transformed_cosine_l637_637172

theorem max_min_sum_of_transformed_cosine :
  let y (x : ℝ) := (1 / 3) * Real.cos x - 1 in
  (∀ x, y x ≤ -2 / 3) ∧ (∀ x, -4 / 3 ≤ y x) →
  (∃ M m : ℝ, M = -2 / 3 ∧ m = -4 / 3 ∧ (M + m) = -2) :=
by
  intro y_bounds
  -- Assuming the bounds on y are provided
  sorry

end max_min_sum_of_transformed_cosine_l637_637172


namespace rational_roots_of_equation_l637_637320

variables {a b c d x : ℝ}

theorem rational_roots_of_equation 
  (h : (a + b)^2 * (x + c^2) * (x + d^2) - (c + d)^2 * (x + a^2) * (x + b^2) = 0) :
  ∃ r : ℝ, (rational_function_of_ℝ r) ∧ (x = r a b c d) :=
sorry

end rational_roots_of_equation_l637_637320


namespace symmetric_function_cannot_be_even_l637_637783

noncomputable def f : ℝ → ℝ := sorry

theorem symmetric_function_cannot_be_even :
  (∀ x, f (f x) = x^2) ∧ (∀ x ≥ 0, f (x^2) = x) → ¬ (∀ x, f x = f (-x)) :=
by 
  intros
  sorry -- Proof is not required

end symmetric_function_cannot_be_even_l637_637783


namespace max_value_of_f_l637_637040

noncomputable def f (x : ℝ) : ℝ := Real.log 2 (1 + Real.sin (2 * x) / (Real.sin x + Real.cos x))

theorem max_value_of_f :
  ∃ x : ℝ, (f x = 1 / 2) ∧ (∀ y : ℝ, f y ≤ 1 / 2) :=
begin
  sorry
end

end max_value_of_f_l637_637040


namespace saree_blue_stripes_l637_637632

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    brown_stripes = 4 →
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_brown h_gold h_blue
  sorry

end saree_blue_stripes_l637_637632


namespace find_X3_find_X4_find_Xm_l637_637068

noncomputable theory

def x (m : ℕ) (a : ℕ → ℝ) : ℝ :=
  ∑ i in Finset.range m, (a i) / 2^(i+1)

def is_valid_a (a : ℕ → ℝ) (m : ℕ) : Prop :=
  ∀ i, i < m → (a i = 1 ∨ a i = -1)

def X_m (m : ℕ) : ℝ :=
  ∑ a in (Finset.pi (Finset.range m) (λ _, ({-1, 1} : Finset ℝ))),
       x m (λ i, a i)

theorem find_X3 : X_m 3 = 2 :=
sorry

theorem find_X4 : X_m 4 = 4 :=
sorry

theorem find_Xm (m : ℕ) : X_m m = 2^(m - 2) :=
sorry

end find_X3_find_X4_find_Xm_l637_637068


namespace profit_function_expression_marginal_profit_function_expression_maximum_profit_value_maximum_marginal_profit_value_l637_637499

noncomputable def revenue (x : ℕ) : ℝ := 3000 * x - 20 * x ^ 2
noncomputable def cost (x : ℕ) : ℝ := 500 * x + 4000
noncomputable def profit (x : ℕ) : ℝ := revenue x - cost x
noncomputable def marginal_function (f : ℕ → ℝ) (x : ℕ) : ℝ := f (x + 1) - f x
noncomputable def marginal_profit (x : ℕ) : ℝ := marginal_function profit x

theorem profit_function_expression :
  profit x = -20 * (x ^ 2 : ℝ) + 2500 * x - 4000 :=
by
  sorry

theorem marginal_profit_function_expression :
  marginal_profit x = 2480 - 40 * x :=
by
  sorry

theorem maximum_profit_value :
  ∃ x ∈ {62, 63}, profit x = 74120 :=
by
  sorry

theorem maximum_marginal_profit_value :
  marginal_profit 1 = 2440 :=
by
  sorry

end profit_function_expression_marginal_profit_function_expression_maximum_profit_value_maximum_marginal_profit_value_l637_637499


namespace line_equation_l637_637678

theorem line_equation {L : ℝ → ℝ → Prop} (h1 : L (-3) (-2)) 
  (h2 : ∃ a : ℝ, a ≠ 0 ∧ (L a 0 ∧ L 0 a)) :
  (∀ x y, L x y ↔ 2 * x - 3 * y = 0) ∨ (∀ x y, L x y ↔ x + y + 5 = 0) :=
by 
  sorry

end line_equation_l637_637678


namespace total_marbles_correct_l637_637141

variable (r : ℝ)

def blue_marbles (r : ℝ) : ℝ := r / 1.3
def green_marbles (r : ℝ) : ℝ := 1.5 * r
def yellow_marbles (r : ℝ) : ℝ := 0.8 * (1.5 * r)

def total_marbles (r : ℝ) : ℝ := r + blue_marbles r + green_marbles r + yellow_marbles r

theorem total_marbles_correct : total_marbles r = 4.47 * r :=
by
  sorry

end total_marbles_correct_l637_637141


namespace proof_problem_l637_637919

variables {R : Type*} [LinearOrderedField R]

-- Conditions given in the problem
variables (f g : R → R)
variable (a : R)
variable (b : R)
variable continuous_f : Continuous f
variable continuous_g : Continuous g
variable g_at_0 : g 0 = 0
variable g_deriv_at_0 : deriv g 0 = a
variable inequality : ∀ x y : R, f (x + y) ≥ f x + g y

theorem proof_problem :
  (∀ x : R, f x = a * x + b) ∧ (∀ y : R, g y ≤ a * y) ∧
  (∀ x y : R, f (x + y) = f x + g y ↔ g y = a * y) :=
sorry

end proof_problem_l637_637919


namespace law_of_sines_l637_637228

theorem law_of_sines (A B C : ℝ) (a b c : ℝ)
  (hA: 0 < A ∧ A < π)
  (hB: 0 < B ∧ B < π)
  (hC: 0 < C ∧ C < π)
  (ha: a = 2 * R * sin A )
  (hb: b = 2 * R * sin B)
  (hc: c = 2 * R * sin C)
  : (a / sin A) = (b / sin B) ∧ (b / sin B) = (c / sin C) ∧ (a / sin A) = (c / sin C) :=
sorry

end law_of_sines_l637_637228


namespace closest_approximation_of_x_l637_637035

noncomputable def x := ((69.28 * 0.004)^3) / (0.03^2 * Real.log 0.58)
theorem closest_approximation_of_x : abs (x + 156.758) < 10^-3 := sorry

end closest_approximation_of_x_l637_637035


namespace smallest_b_not_prime_l637_637745

theorem smallest_b_not_prime (b : ℕ) (h : b > 0) :
  ∃ k > 0, ∀ x : ℤ, ¬ Nat.Prime (x^3 + k^2) ∧ (∀ j > 0, (∀ x : ℤ, ¬ Nat.Prime (x^3 + j^2)) → j ≥ k) :=
by
  use 1
  sorry

end smallest_b_not_prime_l637_637745


namespace find_some_number_l637_637831

def some_number (x : Int) (some_num : Int) : Prop :=
  (3 < x ∧ x < 10) ∧
  (5 < x ∧ x < 18) ∧
  (9 > x ∧ x > -2) ∧
  (8 > x ∧ x > 0) ∧
  (x + some_num < 9)

theorem find_some_number :
  ∀ (some_num : Int), some_number 7 some_num → some_num < 2 :=
by
  intros some_num H
  sorry

end find_some_number_l637_637831


namespace log_base_2_of_y_l637_637822

/-- If y = (log_9 3)^(log_4 16), then log_2 y = -2. -/
theorem log_base_2_of_y :
  let y := (Real.log 3 / Real.log 9) ^ (Real.log 16 / Real.log 4)
  in Real.log 2 y = -2 :=
by
  sorry

end log_base_2_of_y_l637_637822


namespace find_q_l637_637445

-- Define the geometric sequence
variable {a : ℕ → ℝ}

-- Define the common ratio
variable {q : ℝ}

-- Define the sum of the first n terms of a geometric sequence
def geom_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

-- Define the conditions
axiom sum_first_3 (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n+1) = a 0 * q ^ n) : 
  geom_sum a 3 = 1

axiom sum_first_6 (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n+1) = a 0 * q ^ n) : 
  geom_sum a 6 = 9

-- The theorem to prove
theorem find_q (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n+1) = a 0 * q ^ n) :
  q = 2 := by
  sorry

end find_q_l637_637445


namespace inequality_holds_equality_condition_l637_637225

theorem inequality_holds (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x^2 + y^2)^2 ≥ (x + y + z) * (x - y + z) * (x + y - z) * (y + z - x) :=
by 
  sorry 

theorem equality_condition (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x^2 + y^2)^2 = (x + y + z) * (x - y + z) * (x + y - z) * (y + z - x) ↔ x = y ∧ z = x * (√2) :=
by 
  sorry

end inequality_holds_equality_condition_l637_637225


namespace patty_can_avoid_chores_l637_637212

theorem patty_can_avoid_chores (money_per_pack packs total_cookies_per_pack chores kid_cookies_cost packs_bought total_cookies total_weekly_cost weeks : ℕ)
    (h1 : money_per_pack = 3)
    (h2 : packs = 15 / money_per_pack)
    (h3 : total_cookies_per_pack = 24)
    (h4 : total_cookies = (p : ℕ) → packs * total_cookies_per_pack)
    (h5 : chores = 4)
    (h6 : kid_cookies_cost = 3)
    (h7 : total_weekly_cost = 2 * chores * kid_cookies_cost)
    (h8 : weeks = (total_cookies / total_weekly_cost)) : 
  weeks = 10 :=
by sorry

end patty_can_avoid_chores_l637_637212


namespace least_positive_three_digit_multiple_of_7_l637_637298

theorem least_positive_three_digit_multiple_of_7 : ∃ n : ℕ, n % 7 = 0 ∧ n ≥ 100 ∧ n < 1000 ∧ ∀ m : ℕ, (m % 7 = 0 ∧ m ≥ 100 ∧ m < 1000) → n ≤ m := 
by
  sorry

end least_positive_three_digit_multiple_of_7_l637_637298


namespace intersection_A_B_l637_637487

def A : Set ℤ := {-1, 1, 3, 5, 7}
def B : Set ℝ := { x | 2^x > 2 * Real.sqrt 2 }

theorem intersection_A_B :
  A ∩ { x : ℤ | x > 3 / 2 } = {3, 5, 7} :=
by
  sorry

end intersection_A_B_l637_637487


namespace num_4digit_numbers_divisible_by_5_l637_637811

theorem num_4digit_numbers_divisible_by_5 : 
  (#{ n : ℕ | n ≥ 1000 ∧ n ≤ 9999 ∧ n % 5 = 0 }.finite.to_finset.card) = 1800 :=
by
  sorry

end num_4digit_numbers_divisible_by_5_l637_637811


namespace num_roots_and_sum_of_digits_l637_637717

def sum_of_digits (n : ℕ) : ℕ :=
  n.toString.foldl (λ acc d, acc + d.toNat - '0'.toNat) 0

theorem num_roots_and_sum_of_digits :
  let roots_count := 2 * 10^2014 * (10^2015 - 1)
  let result_sum := sum_of_digits (roots_count * 9)
  result_sum = 18135 :=
by
  -- Define the necessary variables
  let lower_bound := 10^(2014!) * Real.pi
  let upper_bound := 10^(2014! + 2015) * Real.pi
  let target_interval := Set.Icc lower_bound upper_bound

  -- Use the given equation
  let equation x := 4 * Real.sin (2 * x) + 3 * Real.cos (2 * x) - 2 * Real.sin x - 4 * Real.cos x + 1
  
  -- Define the number of roots in this interval (not computed here)
  sorry
  
  -- Define the sum of digits of the result (not computed here)
  sorry

  -- Use the results to prove the final statement
  sorry

end num_roots_and_sum_of_digits_l637_637717


namespace fifteenth_digit_sum_l637_637969

/-- The 15th digit after the decimal point of the sum of decimal equivalents of 1/9 and 1/11 is 1. -/
theorem fifteenth_digit_sum (d1 d2 : Nat) (h1 : (1/9 : Rat) = 0.1111111 -- overline 1 represents repeating 1
                    h2 : (1/11 : Rat) = 0.090909) -- overline 090909 represents repeating 090909
                   (repeating_block : String := "10")
                    : repeating_block[15 % 2] = '1' := -- finding the 15th digit
by
  sorry

end fifteenth_digit_sum_l637_637969


namespace proposition1_proposition4_true_propositions_l637_637079

variables {m n : Line}
variables {α β : Plane}

-- Given conditions
axiom perpendicular_m_α (h₁ : m ∈ α) : Perpendicular m α
axiom contains_n_β (h₂ : n ∈ β) : Contains n β

-- Propositions to be proved
theorem proposition1 (h₃ : Parallel α β) : Perpendicular m n := sorry
theorem proposition4 (h₄ : Parallel m n) : Perpendicular α β := sorry

-- Final proof that 1 and 4 are correct
theorem true_propositions : 
  (Parallel α β → Perpendicular m n) ∧ 
  (Parallel m n → Perpendicular α β) :=
begin
  split,
  { exact proposition1 },
  { exact proposition4 },
end

end proposition1_proposition4_true_propositions_l637_637079


namespace find_values_l637_637554

theorem find_values (x y : ℤ) 
  (h1 : x / 5 + 7 = y / 4 - 7)
  (h2 : x / 3 - 4 = y / 2 + 4) : 
  x = -660 ∧ y = -472 :=
by 
  sorry

end find_values_l637_637554


namespace books_from_second_shop_l637_637915

-- Define the conditions
def num_books_first_shop : ℕ := 65
def cost_first_shop : ℕ := 1280
def cost_second_shop : ℕ := 880
def total_cost : ℤ := cost_first_shop + cost_second_shop
def average_price_per_book : ℤ := 18

-- Define the statement to be proved
theorem books_from_second_shop (x : ℕ) :
  (num_books_first_shop + x) * average_price_per_book = total_cost →
  x = 55 :=
by
  sorry

end books_from_second_shop_l637_637915


namespace john_profit_is_1500_l637_637522

noncomputable def john_profit (total_puppies : ℕ) (half_given_away : ℕ) 
  (puppies_kept : ℕ) (sell_price : ℕ) (stud_fee : ℕ) : ℕ :=
  (total_puppies - half_given_away - puppies_kept) * sell_price - stud_fee

theorem john_profit_is_1500 : john_profit 8 4 1 600 300 = 1500 := 
by simp [john_profit]; sorry

end john_profit_is_1500_l637_637522


namespace max_value_expr_l637_637185

open Real

noncomputable def expr (x : ℝ) : ℝ :=
  (x^4 + 3 * x^2 - sqrt (x^8 + 9)) / x^2

theorem max_value_expr : ∀ (x y : ℝ), (0 < x) → (y = x + 1 / x) → expr x = 15 / 7 :=
by
  intros x y hx hy
  sorry

end max_value_expr_l637_637185


namespace sin_theta_condition_l637_637419

theorem sin_theta_condition {θ : ℝ} (h : |θ - π / 6| < π / 6) : 
  (∀ θ : ℝ, (|θ - π / 6| < π / 6) → (sin θ < sqrt 3 / 2)) ∧ 
            (¬ ∀ θ : ℝ, (sin θ < sqrt 3 / 2) → (|θ - π / 6| < π / 6)) :=
by
  sorry

end sin_theta_condition_l637_637419


namespace sequence_bound_l637_637528

noncomputable def seq_pos (a : ℕ → ℝ) := ∀ n : ℕ, 0 < a n
noncomputable def condition1 (a : ℕ → ℝ) := ∀ n : ℕ, a n - 2 * a (n + 1) + a (n + 2) ≥ 0
noncomputable def condition2 (a : ℕ → ℝ) := ∀ n : ℕ, ∑ j in Finset.range (n + 1), a j ≤ 1

theorem sequence_bound {a : ℕ → ℝ}
  (h_pos : seq_pos a)
  (h_cond1 : condition1 a)
  (h_cond2 : condition2 a) :
  ∀ n : ℕ, 0 ≤ a n - a (n + 1) ∧ a n - a (n + 1) < 2 / (n^2) :=
begin
  sorry,
end

end sequence_bound_l637_637528


namespace lines_tangent_to_parabola_l637_637116

def parabola_tangent (a b : Set (ℝ × ℝ)) (M : ℝ × ℝ) (c : ℝ) :=
  ∀ (line_rotation_angle : ℝ), 
  let A : ℝ × ℝ := (line_rotation_angle, -c),
      B : ℝ × ℝ := (line_rotation_angle, 0),
      MA_slope : ℝ := -c / line_rotation_angle,
      perpendicular_slope : ℝ := line_rotation_angle / c,
      g : ℝ → ℝ := λ x, perpendicular_slope * (x - line_rotation_angle)
  in ∃ k, ∀ x, g(x) = x^2 / (4 * c)

theorem lines_tangent_to_parabola (a b : Set (ℝ × ℝ)) (M : ℝ × ℝ) (c : ℝ) (h1 : c > 0) 
  (h2 : ∀ p ∈ a, ∃ x, p = (x, -c)) 
  (h3 : ∀ p ∈ b, ∃ x, p = (x, 0)) 
  (h4 : M ∉ a): 
  parabola_tangent a b M c :=
by
  sorry

end lines_tangent_to_parabola_l637_637116


namespace shorten_ellipse_parametric_form_l637_637928

theorem shorten_ellipse_parametric_form :
  ∀ (θ : ℝ), 
  ∃ (x' y' : ℝ),
    x' = 4 * Real.cos θ ∧ y' = 2 * Real.sin θ ∧
    (∃ (x y : ℝ),
      x' = 2 * x ∧ y' = y ∧
      x = 2 * Real.cos θ ∧ y = 2 * Real.sin θ) :=
by
  sorry

end shorten_ellipse_parametric_form_l637_637928


namespace quadratic_equation_characteristics_l637_637929

def discriminant (a b c : ℤ) : ℤ := b * b - 4 * a * c

theorem quadratic_equation_characteristics :
  let a : ℤ := 1,
      b : ℤ := 1984513,
      c : ℤ := 3154891,
      Δ : ℤ := discriminant a b c in
  (¬ (Δ < 0 ∧ ∀ x1 x2 : ℂ, (a ≠ 0 ∧ a * x1 * x1 + b * x1 + c = 0 ∧ a * x2 * x2 + b * x2 + c = 0)) ∧
   ¬ (∃ x1 x2 : ℤ, a * x1 * x1 + b * x1 + c = 0 ∧ a * x2 * x2 + b * x2 + c = 0) ∧
   ¬ (∃ x1 x2 : ℝ, a * x1 * x1 + b * x1 + c = 0 ∧ a * x2 * x2 + b * x2 + c = 0 ∧ x1 > 0 ∧ x2 > 0) ∧
   ¬ (∃ x1 x2 : ℝ, a * x1 * x1 + b * x1 + c = 0 ∧ a * x2 * x2 + b * x2 + c = 0 ∧ (1 / x1 + 1 / x2 < -1))) :=
by
  let a : ℤ := 1
  let b : ℤ := 1984513
  let c : ℤ := 3154891
  let Δ : ℤ := discriminant a b c
  sorry

end quadratic_equation_characteristics_l637_637929


namespace quadratic_has_distinct_real_roots_l637_637479

open Classical

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_distinct_real_roots (k : ℝ) (h_nonzero : k ≠ 0) : 
  (k > -1) ↔ (discriminant k (-2) (-1) > 0) :=
by
  unfold discriminant
  simp
  linarith

end quadratic_has_distinct_real_roots_l637_637479


namespace range_of_k_max_distance_l637_637067

noncomputable def f (x k m : ℝ) : ℝ := abs (x^2 - k*x - m)

-- Part 1
theorem range_of_k (h : ∀ x > 1, f x k (2 * k^2) ≤ f (x + 1) k (2 * k^2)) : -1 ≤ k ∧ k ≤ 0.5 :=
sorry

-- Part 2
theorem max_distance (a b k m : ℝ) (h : ∀ x ∈ set.Icc a b, f x k m ≤ 1) : b - a ≤ 2 * real.sqrt 2 :=
sorry

end range_of_k_max_distance_l637_637067


namespace exists_coloring_25_even_neighbors_impossible_coloring_25_odd_neighbors_l637_637856

-- Definitions for the coloring conditions
def even_and_at_least_2_neighbors (grid : matrix (Fin 5) (Fin 5) Bool) : Prop :=
  ∀ i j, grid i j = true → ((∀ direction : fin 4, (neighbor grid direction i j).some grid direction = some true) → 
                            card (neighbors grid i j) % 2 = 0 ∧ card (neighbors grid i j) ≥ 2)

def odd_neighbors (grid : matrix (Fin 5) (Fin 5) Bool) : Prop :=
  ∀ i j, grid i j = true → ((∀ direction : fin 4, (neighbor grid direction i j).some grid direction = some true) → 
                            card (neighbors grid i j) % 2 = 1)

-- Lean statement for part (a): Possibility of such coloring
theorem exists_coloring_25_even_neighbors :
  ∃ grid : matrix (Fin 5) (Fin 5) Bool, ((∀ i j : Fin 5, i ≠ j → grid i j = grid j i) ∧ 
  count_true grid = 25 ∧ even_and_at_least_2_neighbors grid) :=
sorry

-- Lean statement for part (b): Impossibility of such coloring
theorem impossible_coloring_25_odd_neighbors :
  ¬ ∃ grid : matrix (Fin 5) (Fin 5) Bool, ((∀ i j : Fin 5, i ≠ j → grid i j = grid j i) ∧ 
  count_true grid = 25 ∧ odd_neighbors grid) :=
sorry

end exists_coloring_25_even_neighbors_impossible_coloring_25_odd_neighbors_l637_637856


namespace probability_of_singing_on_Saturday_l637_637821

variable (P : Prop → ℝ)

-- Conditions
axiom given_data :
  P (¬(sings_on_saturday)) → sings_on_sunday = 0.7 ∧
  P sings_on_sunday = 0.5 ∧
  P (sings_on_saturday → ¬sings_on_sunday) = 1

-- Theorem: Find the probability that Alex sings on Saturday
theorem probability_of_singing_on_Saturday
  (h1 : P (¬(sings_on_saturday)) → sings_on_sunday = 0.7)
  (h2 : P sings_on_sunday = 0.5)
  (h3 : P (¬sings_on_saturday ∧ sings_on_sunday)) :
  P (sings_on_saturday) = 2/7 := 
sorry

end probability_of_singing_on_Saturday_l637_637821


namespace binary_multiplication_l637_637369

theorem binary_multiplication :
  0b1101 * 0b110 = 0b1011110 := 
sorry

end binary_multiplication_l637_637369


namespace hyperbola_sum_l637_637011

theorem hyperbola_sum (h k a b c : ℝ)
  (h_center : h = 1) 
  (k_center : k = 2) 
  (vertex_distance : a = 4) 
  (focus_distance : c = sqrt 50)
  (hyperbola_eq : c^2 = a^2 + b^2)
  : h + k + a + b = 7 + sqrt 34 := by
  sorry

end hyperbola_sum_l637_637011


namespace problem_solution_l637_637760

def is_ideal_set (n : ℕ) (S : Set ℕ) : Prop :=
  ∃ m ≤ n, ∀ (s1 s2 ∈ S), s1 ≠ s2 → |s1 - s2| ≠ m

def set_A (n : ℕ) : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2 * n}

def set_B : Set ℕ := {x | 10 < x ∧ x ≤ 20}

def set_C : Set ℕ := {x | ∃ k : ℕ, k > 0 ∧ x = 3 * k - 1 ∧ x ≤ 20}

def transform (S : Set ℕ) : Set ℕ := {x | ∃ s ∈ S, x = 2001 - s}

theorem problem_solution :
  ¬ is_ideal_set 10 set_B ∧
  is_ideal_set 10 set_C ∧
  ∀ S, is_ideal_set 1000 S → is_ideal_set 1000 (transform S) :=
by
  sorry

end problem_solution_l637_637760


namespace real_roots_of_abs_equation_l637_637936

theorem real_roots_of_abs_equation : 
  let f (x : ℝ) := x * abs x - 3 * abs x + 2 in
  (∃ r₁ r₂ r₃: ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃) :=
by
  sorry

end real_roots_of_abs_equation_l637_637936


namespace max_plus_ten_min_eq_zero_l637_637907

theorem max_plus_ten_min_eq_zero (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  let M := max (x * y + x * z + y * z)
  let m := min (x * y + x * z + y * z)
  M + 10 * m = 0 :=
by
  sorry

end max_plus_ten_min_eq_zero_l637_637907


namespace show_CE_is_bisector_l637_637533

noncomputable def is_bisector_CE (A B C S T D E : Point) : Prop :=
  let lineAS := line A S
  let lineCT := line C T
  let D := intersection lineAS lineCT
  let E := a_point_on_segment B A (BE = segment_length C D)
  (is_triangle A B C) ∧
  (acute_triangle A B C) ∧
  (segment_length A B = segment_length B C) ∧
  (is_point_on_ray S B C) ∧
  (segment_length A S = segment_length B S) ∧
  (is_point_on_ray T B A) ∧
  (segment_length B T = segment_length C T) ∧
  (E_on_segment_BA E B A) →
  is_angle_bisector (line C E) (angle D C B)

theorem show_CE_is_bisector : ∀ A B C S T D E : Point,
  (triangle A B C) →
  (acute A) →
  (acute B) →
  (acute C) →
  (A ≠ B) →
  (B ≠ C) →
  (A ≠ C) →
  (segment_length A B = segment_length B C) →
  is_point_on_ray S B C →
  (segment_length A S = segment_length B S) →
  is_point_on_ray T B A →
  (segment_length B T = segment_length C T) →
  is_point_on_line D A S →
  is_point_on_line D C T →
  E_on_segment_BA E B A (segment_length B E = segment_length C D) →
  is_bisector_CE A B C S T D E :=
begin
    intros,
    sorry, -- Proof
end

end show_CE_is_bisector_l637_637533


namespace patty_weeks_without_chores_correct_l637_637216

noncomputable def patty_weeks_without_chores : ℕ := by
  let cookie_per_chore := 3
  let chores_per_week_per_sibling := 4
  let siblings := 2
  let dollars := 15
  let cookie_pack_size := 24
  let cookie_pack_cost := 3

  let packs := dollars / cookie_pack_cost
  let total_cookies := packs * cookie_pack_size
  let weekly_cookies_needed := chores_per_week_per_sibling * cookie_per_chore * siblings

  exact total_cookies / weekly_cookies_needed

theorem patty_weeks_without_chores_correct : patty_weeks_without_chores = 5 := sorry

end patty_weeks_without_chores_correct_l637_637216


namespace g_sum_3_2_4_3_l637_637866

def g (x y : ℝ) : ℝ :=
if x + y ≤ 6 then (2 * x * y - x - 4) / (3 * x) else (2 * x * y + y + 6) / (-3 * y)

theorem g_sum_3_2_4_3 : g 3 2 + g 4 3 = -28 / 9 :=
by
  sorry

end g_sum_3_2_4_3_l637_637866


namespace proof_problem_l637_637768

open Matrix

variables (v u : Fin 3 → ℝ)

def cross_product (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![
    a 1 * b 2 - a 2 * b 1,
    a 2 * b 0 - a 0 * b 2,
    a 0 * b 1 - a 1 * b 0
  ]

def scalar_mult (c : ℝ) (a : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![
    c * a 0,
    c * a 1,
    c * a 2
  ]

def vector_add (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![
    a 0 + b 0,
    a 1 + b 1,
    a 2 + b 2
  ]

def vector_sub (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![
    a 0 - b 0,
    a 1 - b 1,
    a 2 - b 2
  ]

variables (cross_vu : Fin 3 → ℝ)
variables (w : Fin 3 → ℝ)

-- Given conditions
def given_conditions : Prop :=
  cross_product v u = cross_vu ∧ w = vector_add (scalar_mult 2 v) (scalar_mult 3 u)

-- Proof statement
theorem proof_problem (h : given_conditions v u cross_vu w) :
  cross_product (vector_sub (scalar_mult 2 v) u) w = ![24, -8, 16] :=
by
  sorry

end proof_problem_l637_637768


namespace quadratic_equation_l637_637618

theorem quadratic_equation (p q : ℝ) 
  (h1 : p^2 + 9 * q^2 + 3 * p - p * q = 30)
  (h2 : p - 5 * q - 8 = 0) : 
  p^2 - p - 6 = 0 :=
by sorry

end quadratic_equation_l637_637618


namespace piecewise_function_evaluation_l637_637177

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
if x >= 0 
then (1 / 2) * x - 1 
else 2 ^ x

-- The theorem stating the desired result
theorem piecewise_function_evaluation : f (f 1) = (Real.sqrt 2) / 2 := 
by 
  sorry

end piecewise_function_evaluation_l637_637177


namespace friedas_probability_of_reaching_corner_l637_637055

-- Definitions and conditions
def grid_size := (4, 4)
def start_position := (1, 2)
def corner_positions := [(1, 1), (1, 4), (4, 1), (4, 4)]
def hop_directions := [(1, 0), (0, 1), (-1, 0), (0, -1)]

-- Define the probability of reaching a corner within 3 hops
def probability_of_reaching_corner_in_3_hops := (21 : ℚ) / 32

-- Theorem statement
theorem friedas_probability_of_reaching_corner :
  ∀ (grid_size : ℕ × ℕ) (start_position : ℕ × ℕ) (corner_positions : list (ℕ × ℕ)) (hop_directions : list (ℕ × ℕ)),
  (grid_size = (4, 4)) →
  (start_position = (1, 2)) →
  (corner_positions = [(1, 1), (1, 4), (4, 1), (4, 4)]) →
  (hop_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]) →
  probability_of_reaching_corner_in_3_hops = (21 : ℚ) / 32 :=
begin
  intros,
  sorry
end

end friedas_probability_of_reaching_corner_l637_637055


namespace earnings_difference_l637_637210

theorem earnings_difference :
  let oula_deliveries := 96
  let tona_deliveries := oula_deliveries * 3 / 4
  let area_A_fee := 100
  let area_B_fee := 125
  let area_C_fee := 150
  let oula_area_A_deliveries := 48
  let oula_area_B_deliveries := 32
  let oula_area_C_deliveries := 16
  let tona_area_A_deliveries := 27
  let tona_area_B_deliveries := 18
  let tona_area_C_deliveries := 9
  let oula_total_earnings := oula_area_A_deliveries * area_A_fee + oula_area_B_deliveries * area_B_fee + oula_area_C_deliveries * area_C_fee
  let tona_total_earnings := tona_area_A_deliveries * area_A_fee + tona_area_B_deliveries * area_B_fee + tona_area_C_deliveries * area_C_fee
  oula_total_earnings - tona_total_earnings = 4900 := by
sorry

end earnings_difference_l637_637210


namespace total_games_l637_637518

theorem total_games (games_this_year games_last_year : Nat) 
  (h1 : games_this_year = 4) 
  (h2 : games_last_year = 5) : 
  (games_this_year + games_last_year = 9) := 
by
  rw [h1, h2]
  norm_num
  sorry -- proof goes here

end total_games_l637_637518


namespace exponential_inequality_solution_l637_637433

theorem exponential_inequality_solution (x : ℝ) :
  (1 / real.pi) ^ (-x + 1) > (1 / real.pi) ^ (x ^ 2 - x) ↔ x < -1 ∨ x > 1 :=
by
  sorry

end exponential_inequality_solution_l637_637433


namespace train_passes_man_in_correct_time_l637_637349

-- Definitions for the given conditions
def platform_length : ℝ := 270
def train_length : ℝ := 180
def crossing_time : ℝ := 20

-- Theorem to prove the time taken to pass the man is 8 seconds
theorem train_passes_man_in_correct_time
  (p: ℝ) (l: ℝ) (t_cross: ℝ)
  (h1: p = platform_length)
  (h2: l = train_length)
  (h3: t_cross = crossing_time) :
  l / ((l + p) / t_cross) = 8 := by
  -- Proof goes here
  sorry

end train_passes_man_in_correct_time_l637_637349


namespace specified_time_correct_l637_637682

theorem specified_time_correct (x : ℝ) (h1 : 900.0 = dist) (h2 : slow_time = x + 1) 
  (h3 : fast_time = x - 3) (h4 : fast_speed = 2 * slow_speed) 
  (dist : ℝ := 900.0) (slow_speed : ℝ := 900.0 / (x + 1)) (fast_speed : ℝ := 900.0 / (x - 3)) 
  (slow_time fast_time : ℝ) :
  2 * slow_speed = fast_speed :=
by
  sorry

end specified_time_correct_l637_637682


namespace repeating_decimal_to_fraction_l637_637393

theorem repeating_decimal_to_fraction :
  let x := 0.688888... in
  x = 31 / 45 :=
sorry

end repeating_decimal_to_fraction_l637_637393


namespace inequality_solution_set_l637_637722

noncomputable def solution_set := { x : ℝ | 0 < x ∧ x < 2 }

theorem inequality_solution_set : 
  { x : ℝ | (4 / x > |x|) } = solution_set :=
by sorry

end inequality_solution_set_l637_637722


namespace shift_graph_l637_637961

def f (x : ℝ) : ℝ := sin (2 * x - π / 6)
def g (x : ℝ) : ℝ := cos (2 * x)

theorem shift_graph (x : ℝ) : g (x - π / 3) = f x :=
by 
  · have h1 : ∀ x, cos (2 * x) = cos (2 * x - π / 3 * 2), sorry
  · sorry -- additional proof steps to be filled in

end shift_graph_l637_637961


namespace maximum_value_of_n_l637_637762

-- Define the arithmetic sequence and the necessary conditions
def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions given in the problem
axiom cond1 : arithmetic_sequence a
axiom cond2 : a 11 / a 10 + 1 < 0
axiom cond3 : ∃ n, ∀ m, m < n → S m < S (m + 1) ∧ S n ≥ S m

-- The sum of the first n terms, S_n, of the arithmetic sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) :=
  (n * (a 1 + a n)) / 2
  
-- The statement of the problem in Lean 4
theorem maximum_value_of_n (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (arithmetic_sequence a) →
  (a 11 / a 10 + 1 < 0) →
  (∃ n, ∀ m, m < n → S m < S (m + 1) ∧ S n ≥ S m) →
  ∀ m, S m > 0 → m ≤ 19 :=
begin
  intros,
  sorry
end

end maximum_value_of_n_l637_637762


namespace nonneg_sum_le_half_product_ge_half_l637_637274

/-- Given a sequence of non-negative real numbers x_1, x_2, ..., x_n such that their sum is less than or equal to 1/2,
    we need to prove that the product (1 - x_1) * (1 - x_2) * ... * (1 - x_n) is greater than or equal to 1/2. --/
theorem nonneg_sum_le_half_product_ge_half (n : ℕ) 
  (x : Fin n → ℝ) 
  (hx_nonneg : ∀ i, 0 ≤ x i) 
  (hx_sum : (∑ i, x i) ≤ 1/2) : 
  ((∏ i, (1 - x i)) ≥ 1/2) := 
by 
  sorry

end nonneg_sum_le_half_product_ge_half_l637_637274


namespace partition_magnitudes_l637_637785

noncomputable def partition_vectors {n : ℕ} (a : fin n → ℝ) : bool → list (fin n) :=
  sorry

theorem partition_magnitudes (n : ℕ) (a : fin n → euclidean_space ℝ (fin 1)) (h : ∀ i : fin n, ∥a i∥ ≤ 1) :
  ∃ (p : bool → list (fin n)),
    let S := ∑ i in (p false).to_finset, a i,
        T := ∑ i in (p true).to_finset, a i
    in ∥∥S∥ - ∥T∥∥ ≤ 1 :=
begin
  sorry
end

end partition_magnitudes_l637_637785


namespace angle_DHQ_right_angle_l637_637070

open EuclideanGeometry

/-- Given a square ABCD. Points P and Q lie on sides AB and BC respectively, such that BP = BQ.
Let H be the foot of the perpendicular dropped from point B to the segment PC. Prove that ∠DHQ = 90°.
-/
theorem angle_DHQ_right_angle
  (A B C D P Q H : Point)
  (h1 : is_square A B C D)
  (h2 : lies_on_segment P A B)
  (h3 : lies_on_segment Q B C)
  (h4 : dist B P = dist B Q)
  (h5 : foot_of_perpendicular H B P C) :
  angle D H Q = 90 :=
  sorry

end angle_DHQ_right_angle_l637_637070


namespace real_number_probability_l637_637574

open Real Complex BigOperators

def rational_in_interval : Set ℚ :=
  {x // 0 ≤ x ∧ x < 2 ∧ ∃ n m : ℤ, 1 ≤ m ∧ (m : ℤ) ≤ 4 ∧ x = (n : ℚ) / m}

def valid_pairs_count (p q : ℚ) : ℕ :=
  #((p, q) ∈ rational_in_interval × rational_in_interval | (cos (p * π) + I * sin (q * π))^6 ∈ ℝ)

theorem real_number_probability :
  ∃ n : ℕ, n = valid_pairs_count / (16 * 16) ∧ n = 1 / 8 := by
  sorry

end real_number_probability_l637_637574


namespace specified_percentage_of_number_is_40_l637_637208

theorem specified_percentage_of_number_is_40 
  (N : ℝ) 
  (hN : (1 / 4) * (1 / 3) * (2 / 5) * N = 25) 
  (P : ℝ) 
  (hP : (P / 100) * N = 300) : 
  P = 40 := 
sorry

end specified_percentage_of_number_is_40_l637_637208


namespace exists_positive_integers_abc_l637_637780

theorem exists_positive_integers_abc (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_m_gt_one : 1 < m) (h_n_gt_one : 1 < n) :
  ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ m^a = 1 + n^b * c ∧ Nat.gcd c n = 1 :=
by
  sorry

end exists_positive_integers_abc_l637_637780


namespace measure_angle_AFE_l637_637848

noncomputable def regular_pentagon (A B C D E : Point) : Prop :=
  (dist A B = dist B C) ∧ (dist B C = dist C D) ∧ (dist C D = dist D E) ∧ (dist D E = dist E A) ∧
  (angle A B C = 108) ∧ (angle B C D = 108) ∧ (angle C D E = 108) ∧ (angle D E A = 108) ∧ (angle E A B = 108)

noncomputable def isosceles_equal_sides (A D E F : Point) : Prop :=
  (dist D F = dist D E) ∧ (angle A D E = 130)

theorem measure_angle_AFE (A B C D E F : Point) (h_pentagon : regular_pentagon A B C D E) (h_iso : isosceles_equal_sides A D E F) :
  angle A F E = 155 :=
sorry

end measure_angle_AFE_l637_637848


namespace pe_mul_pf_l637_637888

variables {A B C D E F P : Type}
variables [Incircle A B C ω]
variables [PointOnCircle D ω BC]
variables [PointOnCircle E ω CA]
variables [PointOnCircle F ω AB]
variables [Circle ω₁]
variables [Circle ω₂]
variables [TangentCircle ω₁ A ω E]
variables [TangentCircle ω₂ A ω F]
variables [Radius ω₁ 5]
variables [Radius ω₂ 6]
variables [IntersectionPoint P (LineThrough E F) (LineThrough (Center ω₁) (Center ω₂))]

theorem pe_mul_pf : (distance P E) * (distance P F) = 3600 :=
sorry

end pe_mul_pf_l637_637888


namespace two_correct_statements_l637_637769

variable (v n1 n2 : Vector ℝ) (l α β : Plane ℝ)

def are_planes_parallel (p q : Plane ℝ) : Prop := p ≠ q ∧ parallel p.normal q.normal
def are_planes_perpendicular (p q : Plane ℝ) : Prop := perpendicular p.normal q.normal
def is_line_parallel_to_plane (l : Line ℝ) (p : Plane ℝ) : Prop := parallel l.direction p.normal
def is_line_perpendicular_to_plane (l : Line ℝ) (p : Plane ℝ) : Prop := perpendicular l.direction p.normal

theorem two_correct_statements 
  (h1 : are_planes_parallel α β ↔ parallel n1 n2)
  (h2 : are_planes_perpendicular α β ↔ perpendicular n1 n2)
  (h3 : is_line_parallel_to_plane l α ↔ parallel v n1)
  (h4 : is_line_perpendicular_to_plane l α ↔ perpendicular v n1) :
  (h1 ∧ h2 ∧ ¬h3 ∧ ¬h4) :=
by
  sorry

end two_correct_statements_l637_637769


namespace popsicles_eaten_l637_637895

theorem popsicles_eaten (total_time : ℕ) (interval : ℕ) (p : ℕ)
  (h_total_time : total_time = 6 * 60)
  (h_interval : interval = 20) :
  p = total_time / interval :=
sorry

end popsicles_eaten_l637_637895


namespace dice_product_probability_l637_637991

theorem dice_product_probability :
  (∃ (a b c d : ℕ), a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ (set.prob ⟨λ x, x = (a, b, c, d)⟩ (λ x, true) = (1/6)^4)) := 
begin
  sorry
end

end dice_product_probability_l637_637991


namespace min_value_c_l637_637273

-- Define the problem using Lean
theorem min_value_c 
    (a b c d e : ℕ)
    (h1 : a + 1 = b) 
    (h2 : b + 1 = c)
    (h3 : c + 1 = d)
    (h4 : d + 1 = e)
    (h5 : ∃ n : ℕ, 5 * c = n ^ 3)
    (h6 : ∃ m : ℕ, 3 * c = m ^ 2) : 
    c = 675 := 
sorry

end min_value_c_l637_637273


namespace quadratic_root_range_l637_637476

theorem quadratic_root_range (k : ℝ) (hk : k ≠ 0) (h : (4 + 4 * k) > 0) : k > -1 :=
by sorry

end quadratic_root_range_l637_637476


namespace average_weight_decrease_l637_637240

theorem average_weight_decrease 
  (A1 : ℝ) (new_person_weight : ℝ) (num_initial : ℕ) (num_total : ℕ) 
  (hA1 : A1 = 55) (hnew_person_weight : new_person_weight = 50) 
  (hnum_initial : num_initial = 20) (hnum_total : num_total = 21) :
  A1 - ((A1 * num_initial + new_person_weight) / num_total) = 0.24 :=
by
  rw [hA1, hnew_person_weight, hnum_initial, hnum_total]
  -- Further proof steps would go here
  sorry

end average_weight_decrease_l637_637240


namespace bucket_capacity_fraction_l637_637625

theorem bucket_capacity_fraction (C : ℝ) (hC : C > 0) :
  (∃ f : ℝ, 25 * C = 62.5 * f * C ∧ f = 2 / 5) :=
begin
  use 2 / 5,
  split,
  { 
    rw [mul_assoc, mul_comm C _, mul_assoc, mul_comm (2/5) _],
    have h : 25 = 62.5 * (2 / 5), {
      norm_num,
    },
    rw h,
  },
  {
    refl,
  }
end

end bucket_capacity_fraction_l637_637625


namespace correct_operation_l637_637644

theorem correct_operation (a b : ℝ) : 
  ¬(a^2 + a^3 = a^5) ∧ ¬((a^2)^3 = a^8) ∧ (a^3 / a^2 = a) ∧ ¬((a - b)^2 = a^2 - b^2) := 
by {
  sorry
}

end correct_operation_l637_637644


namespace find_standard_equation_of_ellipse_find_range_of_m_find_area_of_triangle_l637_637428

noncomputable theory

section ellipse_geometry

variables {a b c m x₁ x₂ y₁ y₂ x₀ y₀ : ℝ}
variable (k : ℝ := 1)

def ellipse (a b : ℝ) := Set (λ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1)
def line (m : ℝ) := Set (λ (x y : ℝ), y = x + m)
def focus (c : ℝ) := (2 * Real.sqrt 2, 0)
def eccentricity (a c : ℝ) := c / a = (Real.sqrt 6) / 3
def midpointAB (x₁ y₁ x₂ y₂ : ℝ) := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)

theorem find_standard_equation_of_ellipse (h1 : focus c) (h2 : eccentricity a c) : 
  ellipse 12 4 :=
sorry

theorem find_range_of_m (h3 : line m) : 
  -4 < m ∧ m < 4 :=
sorry
  
theorem find_area_of_triangle (P : (ℝ × ℝ)) (h4 : P = (-3, 2)) :
  let x₀ := (x₁ + x₂) / 2,
      y₀ := (y₁ + y₂) / 2, 
      AB := Real.sqrt (1 + k^2) * Real.sqrt ((x₁ + x₂)^2 - 4 * x₁ * x₂),
      PE := Real.sqrt ((-3 + 3/2)^2 + (2 - 1/2)^2) in
  abs (AB * PE / 2) = 9 / 2 :=
sorry
  
end ellipse_geometry

end find_standard_equation_of_ellipse_find_range_of_m_find_area_of_triangle_l637_637428


namespace PQ_squared_l637_637069

variables (A B C P Q K L : Type) [InnerProductSpace ℝ A]
variables (AB BC : ℝ) (h_AB : AB = 42) (h_BC : BC = 56)
variable (angle_ABC : ∀ a b c : A, angle a b c = π / 2)
variables (PK KQ : ℝ) (PK_eq_KQ : PK = KQ)
variables (QL PL : ℝ) (h_ratio : QL / PL = 3 / 4)

theorem PQ_squared {AC : ℝ} (h_AC : AC = sqrt (AB^2 + BC^2)) :
  PQ^2 = 1250 :=
sorry

end PQ_squared_l637_637069


namespace solution_set_l637_637221

theorem solution_set :
  {p : ℝ × ℝ | (p.1^2 + 3 * p.1 * p.2 + 2 * p.2^2) * (p.1^2 * p.2^2 - 1) = 0} =
  {p : ℝ × ℝ | p.2 = -p.1 / 2} ∪
  {p : ℝ × ℝ | p.2 = -p.1} ∪
  {p : ℝ × ℝ | p.2 = -1 / p.1} ∪
  {p : ℝ × ℝ | p.2 = 1 / p.1} :=
by sorry

end solution_set_l637_637221


namespace max_sin_cos_value_l637_637463

open Real

noncomputable def max_value (α β γ : ℝ) : ℝ :=
  sin (α - γ) + cos (β - γ)

theorem max_sin_cos_value (α β γ : ℝ) (hα : 0 ≤ α ∧ α ≤ 2 * π) (hβ : 0 ≤ β ∧ β ≤ 2 * π)
  (hγ : 0 ≤ γ ∧ γ ≤ 2 * π) (h : sin (α - β) = 1 / 4) :
  max_value α β γ ≤ sqrt 10 / 2 :=
sorry

end max_sin_cos_value_l637_637463


namespace sum_of_internal_angles_hexagon_distance_to_building_l637_637219

-- For (①)
theorem sum_of_internal_angles_hexagon (n : ℕ) (h : n = 6) : 
  (n - 2) * 180 = 720 :=
by
  -- Given n = 6
  have h1 : n - 2 = 4 := by rw h; norm_num
  -- Calculate the sum of internal angles
  rw h1
  norm_num

-- For (②)
theorem distance_to_building (height : ℝ) (angle : ℝ) (sqrt3_approx : ℝ) (h1 : height = 137) (h2 : angle = 30) (h3 : sqrt3_approx = 1.732) : 
  Real.ceil (height * sqrt3_approx) = 237 :=
by
  -- Given height = 137, angle = 30°, sqrt3 ≈ 1.732
  rw [h1, h3]
  -- Calculate distance
  have h4 : 137 * 1.732 ≈ 237 := by norm_num
  -- Considering rounding
  exact Real.ceil_eq_of_le_real_part h4 sorry  -- We'll assume the ceiling of a number very close to 237 is 237

end sum_of_internal_angles_hexagon_distance_to_building_l637_637219


namespace actual_percent_decrease_is_25_percent_l637_637891

noncomputable theory
open_locale big_operators

def last_year_revenue : ℝ := R
def projected_revenue (R : ℝ) := 1.30 * R
def actual_revenue (R : ℝ) := 0.5769230769230769 * (1.30 * R)
def percent_decrease (R : ℝ) := (R - actual_revenue R) / R

theorem actual_percent_decrease_is_25_percent (R : ℝ) :
  percent_decrease R = 0.25 := by
  sorry

end actual_percent_decrease_is_25_percent_l637_637891


namespace max_marked_squares_l637_637191

theorem max_marked_squares (n : ℕ) (h_pos : n > 0) :
  ∃ k : ℕ, ∃ (configurations : finset (finset (fin (2 * 3)))), 
  (∀ (c : finset (fin (2 * 3))), c ∈ configurations → (∀ (x ∈ c), (finset.card (c ∩ neighbors x) ≤ 2))) ∧ 
  (4 * n = max_squares n) ∧ 
  (count_ways n = k ^ n)
sorry

end max_marked_squares_l637_637191


namespace area_of_locus_S_l637_637881

noncomputable def area_of_S : ℝ :=
  ∫ x in 0..1, (x + 1 - 2 * real.sqrt(x))

theorem area_of_locus_S :
  (∫ x in 0..1, (x + 1 - 2 * real.sqrt(x))) = (1 / 6) :=
by
  sorry

end area_of_locus_S_l637_637881


namespace find_a_l637_637134

noncomputable def polynomial1 (x : ℝ) : ℝ := x^3 + 3 * x^2 - x - 3
noncomputable def polynomial2 (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 1

theorem find_a (a : ℝ) (x : ℝ) (hx1 : polynomial1 x > 0)
  (hx2 : polynomial2 x a ≤ 0) (ha : a > 0) : 
  3 / 4 ≤ a ∧ a < 4 / 3 :=
sorry

end find_a_l637_637134


namespace area_of_rectangle_inscribed_in_triangle_l637_637908

theorem area_of_rectangle_inscribed_in_triangle :
  ∀ (E F G A B C D : ℝ) (EG altitude_ABCD : ℝ),
    E < F ∧ F < G ∧ A < B ∧ B < C ∧ C < D ∧ A < D ∧ D < G ∧ A < G ∧
    EG = 10 ∧ 
    altitude_ABCD = 7 ∧ 
    B = C ∧ 
    A + D = EG ∧ 
    A + 2 * B = EG →
    ((A * B) = (1225 / 72)) :=
by
  intros E F G A B C D EG altitude_ABCD
  intro h
  sorry

end area_of_rectangle_inscribed_in_triangle_l637_637908


namespace fifteenth_digit_sum_l637_637968

theorem fifteenth_digit_sum (d₁ d₂ : ℕ → ℕ) 
  (h₁ : ∀ n, d₁ n = if n = 0 then 1 else 0) 
  (h₂ : ∀ n, d₂ n = if n % 2 = 0 then 0 else 9) :
  let sum_digit := λ n, (d₁ n + d₂ n) % 10 in
  sum_digit 14 = 1 :=
by 
  sorry

end fifteenth_digit_sum_l637_637968


namespace min_rho_squared_l637_637316

noncomputable def rho_squared (x t : ℝ) : ℝ :=
  (x - t)^2 + (x^2 - 4 * x + 7 + t)^2

theorem min_rho_squared : 
  ∃ (x t : ℝ), x = 3/2 ∧ t = -7/8 ∧ 
  ∀ (x' t' : ℝ), rho_squared x' t' ≥ rho_squared (3/2) (-7/8) :=
by
  sorry

end min_rho_squared_l637_637316


namespace complex_point_quadrant_l637_637939

theorem complex_point_quadrant :
  let z : ℂ := (3 + complex.i) / (1 + complex.i) + 3 * complex.i
  in z = 2 + 2 * complex.i := sorry

end complex_point_quadrant_l637_637939


namespace collinear_A_S_P_l637_637962

/-- Given triangle ABC inscribed in a circle Omega with center O. 
The circle constructed with AO as diameter intersects the circumcircle of triangle OBC at a point S ≠ O.
Tangents to Omega at points B and C intersect at point P,
prove that points A, S, and P are collinear. -/
theorem collinear_A_S_P 
  (Ω : Circle)
  (O : Point)
  (A B C : Point)
  (circumOBC : Circle)
  (S : Point)
  (tangent_B tangent_C : Line)
  (P : Point)
  (h1 : inscribed_in Ω A B C)
  (h2 : center Ω = O)
  (h3 : diameter_circle AO)
  (h4 : intersects_circumcircle_neq O circumOBC S)
  (h5 : tangent_to Ω B tangent_B)
  (h6 : tangent_to Ω C tangent_C)
  (h7 : intersection tangent_B tangent_C = P) :
  collinear A S P :=
sorry

end collinear_A_S_P_l637_637962


namespace product_of_base8_digits_of_5432_l637_637636

open Nat

def base8_digits (n : ℕ) : List ℕ :=
  let rec digits_helper (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc
    else digits_helper (n / 8) ((n % 8) :: acc)
  digits_helper n []

def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (· * ·) 1

theorem product_of_base8_digits_of_5432 : 
    product_of_digits (base8_digits 5432) = 0 :=
by
  sorry

end product_of_base8_digits_of_5432_l637_637636


namespace proof_problem_l637_637062

variable (A : Set α)

theorem proof_problem (P : A ∩ ∅ = ∅) (Q : A ∪ ∅ = A) : (P ∨ Q) ∧ ¬¬Q :=
by
  sorry

end proof_problem_l637_637062


namespace seven_large_power_mod_seventeen_l637_637401

theorem seven_large_power_mod_seventeen :
  (7 : ℤ)^1985 % 17 = 7 :=
by
  have h1 : (7 : ℤ)^2 % 17 = 15 := sorry
  have h2 : (7 : ℤ)^4 % 17 = 16 := sorry
  have h3 : (7 : ℤ)^8 % 17 = 1 := sorry
  have h4 : 1985 = 8 * 248 + 1 := sorry
  sorry

end seven_large_power_mod_seventeen_l637_637401


namespace f_sin_30_eq_neg_one_l637_637464

-- Given conditions
def f : ℝ → ℝ
| x := if x = cos (sorry : ℝ) then cos (3 * sorry) else sorry

theorem f_sin_30_eq_neg_one
  (h : ∀ x, f (cos x) = cos (3 * x)) :
  f (sin (30 * (real.pi / 180))) = -1 := by
  sorry

end f_sin_30_eq_neg_one_l637_637464


namespace statementB_statementD_l637_637054

noncomputable def f (x : ℝ) : ℝ := (2 / x) + Real.log x

theorem statementB :
  ∃! x ∈ (Set.Ioi (0 : ℝ)), f x - x = 0 :=
sorry

theorem statementD (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 > x2) (h4 : f x1 = f x2) :
  x1 + x2 > 4 :=
sorry

end statementB_statementD_l637_637054


namespace fraction_bluejays_female_l637_637843

open BigOperators

theorem fraction_bluejays_female (B : ℕ) 
  (h1 : 2/5 * B = R) 
  (h2 : B - R = J)
  (h3 : 1/3 * R = F_R)
  (h4 : (7/15) * B = M) :
      F_J / J = 2/3 
  where
    R : ℕ -- Number of robins
    J : ℕ -- Number of bluejays
    F_R : ℕ -- Number of female robins
    F_J : ℕ -- Number of female bluejays
    M : ℕ -- Number of male birds
    F : ℕ -- Total number of female birds
    F := B - M
    F_J := F - F_R

end fraction_bluejays_female_l637_637843


namespace real_axis_length_of_hyperbola_3x2_y2_eq_9_l637_637933

theorem real_axis_length_of_hyperbola_3x2_y2_eq_9 : 
  let a := sqrt 3 in
  (3 * a^2 * 2) = (2 * sqrt 3) := 
by
  sorry

end real_axis_length_of_hyperbola_3x2_y2_eq_9_l637_637933


namespace dogs_food_consumption_l637_637627

def cups_per_meal_per_dog : ℝ := 1.5
def number_of_dogs : ℝ := 2
def meals_per_day : ℝ := 3
def cups_per_pound : ℝ := 2.25

theorem dogs_food_consumption : 
  ((cups_per_meal_per_dog * number_of_dogs) * meals_per_day) / cups_per_pound = 4 := 
by
  sorry

end dogs_food_consumption_l637_637627


namespace binomial_expansion_constant_term_l637_637605

theorem binomial_expansion_constant_term (n : ℕ) (h : 2^n = 4096) : 
  let k := 12 in (n = k) → 
  let T := (Nat.choose k 3) * (-1)^3 in 
  T = -220 :=
by
  sorry

end binomial_expansion_constant_term_l637_637605


namespace complex_magnitude_sqrt_e_l637_637193

noncomputable def complex_value (z : ℂ) (h : z + (e : ℂ) / z + (real.pi : ℂ) = 0) : ℂ := |z|

theorem complex_magnitude_sqrt_e (z : ℂ) (h : z + (e : ℂ) / z + (real.pi : ℂ) = 0) : complex_value z h = real.sqrt e :=
sorry

end complex_magnitude_sqrt_e_l637_637193


namespace regression_line_fits_l637_637281

variables {x y : ℝ}

def points := [(1, 2), (2, 5), (4, 7), (5, 10)]

def regression_line (x : ℝ) : ℝ := x + 3

theorem regression_line_fits :
  (∀ p ∈ points, regression_line p.1 = p.2) ∧ (regression_line 3 = 6) :=
by
  sorry

end regression_line_fits_l637_637281


namespace minimize_Sn_l637_637540

variable {n : ℕ}

def a (n : ℕ) := 2 * n - 49

def S (n : ℕ) := (1 + n) * a(1) / 2 + (n - 1) * a(n) / 2

theorem minimize_Sn : (∀ n : ℕ, (S n) ≥ (S 24))
:= sorry

end minimize_Sn_l637_637540


namespace probability_of_union_l637_637041

-- Define the range of two-digit numbers
def digit_count : ℕ := 90

-- Define events A and B
def event_a (n : ℕ) : Prop := n % 2 = 0
def event_b (n : ℕ) : Prop := n % 5 = 0

-- Define the probabilities P(A), P(B), and P(A ∩ B)
def P_A : ℚ := 45 / digit_count
def P_B : ℚ := 18 / digit_count
def P_A_and_B : ℚ := 9 / digit_count

-- Prove the final probability using inclusion-exclusion principle
theorem probability_of_union : P_A + P_B - P_A_and_B = 0.6 := by
  sorry

end probability_of_union_l637_637041


namespace find_length_BD_l637_637507

section TriangleGeometry

variables (A B C D : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

def rightTriangle (A B C : Type) : Prop :=
  ∃ (AC BC : ℝ), AC = 4 ∧ BC = 4 ∧ (angle C = 90 ∧ CD ⊥ AB)

theorem find_length_BD (h : rightTriangle A B C) :
  let D := midpoint A B in
  dist B D = 2 * Real.sqrt 2 := sorry

end TriangleGeometry

end find_length_BD_l637_637507


namespace last_integer_in_sequence_divided_by_2_l637_637943

theorem last_integer_in_sequence_divided_by_2 (a : ℕ → ℝ) (start : a 0 = 2000000) (hf : ∀ n, a (n + 1) = a n / 2) :
  ∃ n, a n = 15625 ∧ ∀ m > n, ¬(a m ∈ ℤ) := 
sorry

end last_integer_in_sequence_divided_by_2_l637_637943


namespace find_k_for_parallel_lines_l637_637784

theorem find_k_for_parallel_lines (k : ℝ) :
  (∀ x y : ℝ, (k - 2) * x + (4 - k) * y + 1 = 0) →
  (∀ x y : ℝ, 2 * (k - 2) * x - 2 * y + 3 = 0) →
  (k = 2 ∨ k = 5) :=
sorry

end find_k_for_parallel_lines_l637_637784


namespace square_point_dist_eq_l637_637562

-- Define the data structure of a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a square using points
structure Square where
  A B C D : Point
  isSquare : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A -- Non-degeneracy conditions

-- Define auxiliary points M and K on sides BC and CD respectively
structure AuxPoints where
  M K : Point
  onBC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = { x := B.x * (1-t) + C.x * t, y := B.y * (1-t) + C.y * t }
  onCD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ K = { x := C.x * (1-t) + D.x * t, y := C.y * (1-t) + D.y * t }

-- Angle condition definition
def angleEq (A B C K : Point) : Prop := 
  let v1 := (B.x - A.x, B.y - A.y)
  let v2 := (C.x - A.x, C.y - A.y)
  let dotProduct := v1.1 * v2.1 + v1.2 * v2.2
  let magV1 := Real.sqrt (v1.1 ^ 2 + v1.2 ^ 2)
  let magV2 := Real.sqrt (v2.1 ^ 2 + v2.2 ^ 2)
  dotProduct = magV1 * magV2

-- Prove that BM + KD = AK
theorem square_point_dist_eq (sq : Square) (aux : AuxPoints) (hAngle: angleEq sq.A sq.B aux.M sq.K):
  dist sq.B aux.M + dist aux.K sq.D = dist sq.A aux.K := 
sorry -- Proof to be provided

end square_point_dist_eq_l637_637562


namespace angle_in_third_quadrant_l637_637662

-- Define the function that determines the quadrant of an angle in degrees
def quadrant (θ : ℝ) : ℕ :=
 if 0 < θ % 360 ∧ θ % 360 < 90 then 1
 else if 90 < θ % 360 ∧ θ % 360 < 180 then 2
 else if 180 < θ % 360 ∧ θ % 360 < 270 then 3
 else if 270 < θ % 360 ∧ θ % 360 < 360 then 4
 else if θ % 360 = 0 then 1
 else if θ % 360 = 90 then 2
 else if θ % 360 = 180 then 3
 else 4

-- Noncomputable due to use of real numbers
noncomputable def angle_of : ℝ := -510

theorem angle_in_third_quadrant : quadrant angle_of = 3 :=
by
  sorry

end angle_in_third_quadrant_l637_637662


namespace boat_crossing_l637_637608

theorem boat_crossing (students teacher trips people_in_boat : ℕ) (h_students : students = 13) (h_teacher : teacher = 1) (h_boat_capacity : people_in_boat = 5) :
  trips = (students + teacher + people_in_boat - 1) / (people_in_boat - 1) :=
by
  sorry

end boat_crossing_l637_637608


namespace general_formula_sequence_sum_terms_inequality_l637_637759

theorem general_formula_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → 2 * S n = (n + 1) * a n) →
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → a n = n) :=
by
  intros hS h1
  sorry

theorem sum_terms_inequality (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → 2 * S n = (n + 1) * a n) →
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → a n = n) →
  (∀ n : ℕ, b n = 4 * a (n + 1) / (a n * a (n + 2))^2) →
  (∀ n : ℕ, T n = ∑ i in finset.range n, b i) →
  (∀ n : ℕ, n > 0 → T n < 5 / 4) :=
by
  intros hS h1 ha hb hT
  sorry

end general_formula_sequence_sum_terms_inequality_l637_637759


namespace no_such_real_numbers_l637_637026

noncomputable def have_integer_roots (a b c : ℝ) : Prop :=
  ∃ r s : ℤ, a * (r:ℝ)^2 + b * r + c = 0 ∧ a * (s:ℝ)^2 + b * s + c = 0

theorem no_such_real_numbers (a b c : ℝ) :
  have_integer_roots a b c → have_integer_roots (a + 1) (b + 1) (c + 1) → False :=
by
  -- proof will go here
  sorry

end no_such_real_numbers_l637_637026


namespace inscribed_circle_radius_l637_637158

theorem inscribed_circle_radius 
  (PQ PR : ℝ) (hPQ : PQ = 15) (hPR : PR = 8) 
  (right_triangle : ∀ Q R P : ℝ, (P ^ 2 + R ^ 2 = Q ^ 2)) :
  ∃ r, r = 3 :=
by
  let QR := 17
  have hQR : QR = Real.sqrt (PQ ^ 2 + PR ^ 2),
  {
    rw [hPQ, hPR],
    simp,
  },
  let area := (1 / 2) * PQ * PR,
  have harea : area = 60,
  {
    rw [hPQ, hPR],
    norm_num,
  },
  let s := (PQ + PR + QR) / 2,
  have hS : s = 20,
  {
    rw [hPQ, hPR],
    norm_num,
  },
  let r := area / s,
  use r,
  have hr: r = 3,
  {
    rw [harea, hS],
    norm_num,
  },
  exact hr,
  sorry

end inscribed_circle_radius_l637_637158


namespace oreos_problem_l637_637859

theorem oreos_problem
  (J : ℕ)
  (James_has : 43)
  (total_oreos : 52)
  (h1 : James_has = M * J + 7)
  (h2 : total_oreos = J + James_has)
  (h3 : James_has = 43)
  (h4 : total_oreos = 52) :
  M = 4 :=
by
  sorry

end oreos_problem_l637_637859


namespace hcl_formed_when_reacting_CH4_Cl2_l637_637740

variable (CH4 Cl2 HCl CCl4 : Type)
variable (moles_of_CH4 moles_of_Cl2 : ℕ)

axiom reaction_balanced : CH4 + 4 * Cl2 → CCl4 + 4 * HCl

def stoichiometric_ratio (moles_of_CH4 : ℕ) (moles_of_Cl2 : ℕ) : ℕ :=
  if moles_of_CH4 = 1 ∧ moles_of_Cl2 = 4 then 4 else 0

theorem hcl_formed_when_reacting_CH4_Cl2 :
  ∀ (moles_of_CH4 moles_of_Cl2 : ℕ), moles_of_CH4 = 1 → moles_of_Cl2 = 4 → stoichiometric_ratio moles_of_CH4 moles_of_Cl2 = 4 :=
by
  sorry

end hcl_formed_when_reacting_CH4_Cl2_l637_637740


namespace atomic_number_R_l637_637585

noncomputable def atomic_number_Pb := 82
def electron_shell_difference := 32

def same_group_atomic_number 
  (atomic_number_Pb : ℕ) 
  (electron_shell_difference : ℕ) : 
  ℕ := 
  atomic_number_Pb + electron_shell_difference

theorem atomic_number_R (R : ℕ) : 
  same_group_atomic_number atomic_number_Pb electron_shell_difference = 114 := 
by
  sorry

end atomic_number_R_l637_637585


namespace no_closed_broken_line_with_odd_segments_l637_637390

theorem no_closed_broken_line_with_odd_segments :
  ¬ ∃ (n : ℕ) (hn : Odd n) (c : ℕ) (A : Fin n → ℤ × ℤ)
    (H : ∀ i, c^2 = (A (i + 1) % n).1 - (A i).1)^2 + ((A (i + 1) % n).2 - (A i).2)^2,
    (A 0 = A (n % n)) :=
sorry

end no_closed_broken_line_with_odd_segments_l637_637390


namespace shooting_accuracy_l637_637882

theorem shooting_accuracy (S : ℕ → ℕ) (H1 : ∀ n, S n < 10 * n / 9) (H2 : ∀ n, S n > 10 * n / 9) :
  ∃ n, 10 * (S n) = 9 * n :=
by
  sorry

end shooting_accuracy_l637_637882


namespace measure_AB_l637_637505

theorem measure_AB (AB CD : ℝ) (B D : ℝ) (a b : ℝ)
  (h1 : AB = a)
  (h2 : CD = b)
  (h3 : 2 * B = D)
  (h4 : ∀ (q p : Prop), q ↔ p):
  AB = a + b :=
begin
  sorry
end

end measure_AB_l637_637505


namespace parabola_vertex_form_l637_637081

theorem parabola_vertex_form :
  ∃ (a b : ℝ), 
    (a * 3^2 + b * 3 = 3) ∧ (-b / (2 * a) = 3) ∧ (∀ (x : ℝ), y = a * x^2 + b * x ↔ y = - (1/3 : ℝ) * x^2 + 2 * x) :=
begin
  sorry
end

end parabola_vertex_form_l637_637081


namespace true_mean_confidence_interval_99_l637_637365

theorem true_mean_confidence_interval_99
  (x_bar : ℝ)
  (s : ℝ)
  (n : ℕ)
  (normal_dist : Prop) :
  x_bar = 30.1 →
  s = 6 →
  n = 9 →
  normal_dist →
  25.38 < (x_bar - 2.36 * s / real.sqrt n) ∧ (x_bar + 2.36 * s / real.sqrt n) < 34.82 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3],
  sorry
end

end true_mean_confidence_interval_99_l637_637365


namespace min_distance_PM_l637_637823

-- Define the line l1
def l1 (P : Point) : Prop := P.x + P.y + 3 = 0

-- Define the curve C (circle)
def C (M : Point) : Prop := (M.x - 5)^2 + M.y^2 = 16

-- Given conditions
theorem min_distance_PM :
  ∃ P M : Point, l1 P ∧ C M ∧ (∃ l2 : Line, line_passing_through l2 P ∧ tangent_to l2 C M) →
  min_value (dist P M) = 4 := by
  sorry

end min_distance_PM_l637_637823


namespace solution_set_f_l637_637552

def f (x : ℝ) : ℝ :=
  if x < 0 then x + 6 else x^2 - 4*x + 6

theorem solution_set_f (x : ℝ) : f(x) > f(1) ↔ x ∈ ((-3 : ℝ), 1) ∨ x ∈ (3, +∞) :=
by
  sorry

end solution_set_f_l637_637552


namespace numDogsInPetStore_l637_637337

-- Definitions from conditions
variables {D P : Nat}

-- Theorem statement - no proof provided
theorem numDogsInPetStore (h1 : D + P = 15) (h2 : 4 * D + 2 * P = 42) : D = 6 :=
by
  sorry

end numDogsInPetStore_l637_637337


namespace max_intersection_points_triangle_circle_l637_637635

-- Define the concepts of a triangle and a circle, and the interaction between them.
def intersects_max_two_points (circle : Type) (segment : Type) : Prop :=
  ∀ (circle : circle) (segment : segment), segment ∩ circle ≤ 2

def triangle := { x // x ∈ set.univ ∧ x.card = 3 }

theorem max_intersection_points_triangle_circle :
  ∀ (circle : Type) (triangle : triangle), 
  (∑ s in triangle, intersects_max_two_points circle s) ≤ 6 :=
begin
  sorry
end

end max_intersection_points_triangle_circle_l637_637635


namespace sqrt_seq_not_geometric_l637_637372

noncomputable def is_geometric_sequence (a b c : ℝ) (k n m : ℕ) (q : ℝ) : Prop :=
  (a < b) ∧ (b < c) ∧ (a = q^k * a) ∧ 
  (b = q^n * a) ∧ (c = q^m * a) ∧ (k < n) ∧ (n < m)

theorem sqrt_seq_not_geometric :
  ¬ ∃ (k n m : ℕ) (q : ℝ), is_geometric_sequence (sqrt 2) (sqrt 3) (sqrt 5) k n m q 
:= sorry

end sqrt_seq_not_geometric_l637_637372


namespace total_tape_length_eq_80π_l637_637335

noncomputable def calculate_tape_length (core_diameter tape_width wraps : ℕ)
(final_diameter : ℕ) : ℕ :=
  wrap_length wraps * tape_width * π

theorem total_tape_length_eq_80π
  (core_diameter tape_width : ℕ) (wraps : ℕ) (final_diameter : ℕ)
  (h_core: core_diameter = 4) (h_tape: tape_width = 4)
  (h_wraps: wraps = 800) (h_final: final_diameter = 16) :
  calculate_tape_length core_diameter tape_width wraps final_diameter = 80 * π :=
sorry

end total_tape_length_eq_80π_l637_637335


namespace total_students_school_l637_637841

def C1 : ℕ := 25
def C2 : ℕ := C1 - 2
def C3 : ℕ := C2 - 2
def C4 : ℕ := C3 - 2
def C5 : ℕ := C4 - 2

theorem total_students_school : C1 + C2 + C3 + C4 + C5 = 105 := by
  have h1 : C2 = 23 := rfl
  have h2 : C3 = 21 := rfl
  have h3 : C4 = 19 := rfl
  have h4 : C5 = 17 := rfl
  calc
    C1 + C2 + C3 + C4 + C5 = 25 + 23 + 21 + 19 + 17 := by rfl
    ... = 105 := by norm_num

end total_students_school_l637_637841


namespace simplify_expression_l637_637917

theorem simplify_expression (x : ℝ) : (3 * x)^4 + 3 * x * x^3 + 2 * x^5 = 84 * x^4 + 2 * x^5 := by
    sorry

end simplify_expression_l637_637917


namespace encyclopediaCost_l637_637675

noncomputable def costEncyclopedias : ℝ :=
  let monthlyPayments := 9 * 57 + 21 in
  let downPayment := 300 in
  let totalCost := downPayment + monthlyPayments in
  let interestRate := 0.18666666666666668 in
  let borrowedAmount := totalCost / (1 + interestRate) in
  downPayment + borrowedAmount

theorem encyclopediaCost :
  costEncyclopedias = 1002.8571428571429 :=
sorry

end encyclopediaCost_l637_637675


namespace multiply_by_11_l637_637285

theorem multiply_by_11 (A B : ℕ) (h : A + B < 10) : 
  (10 * A + B) * 11 = 100 * A + 10 * (A + B) + B :=
by
  sorry

end multiply_by_11_l637_637285


namespace prob_constraint_sum_digits_l637_637538

noncomputable def P (N : ℕ) :=
  let favorable_positions := (Nat.floor (2 * N / 5)) + 1 + (N - (Nat.ceil (3 * N / 5)) + 1)
  favorable_positions / (N + 1 : ℝ)

-- The objective is to establish the sum of the digits of the smallest multiple of 5 where P(N) < 321/400

theorem prob_constraint_sum_digits :
  let min_N := (List.range 1000).find (λ n, n % 5 = 0 ∧ P n < 321 / 400) ∨ 480 -- Use 480 as per problem's solution boundary
  let digit_sum := List.sum (List.map (λ c, c.toNat - '0'.toNat) (min_N.digits 10))
  digit_sum = 12 := 
sorry

end prob_constraint_sum_digits_l637_637538


namespace inequality_solution_set_l637_637270

theorem inequality_solution_set (x : ℝ) : ((x - 1) * (x^2 - x + 1) > 0) ↔ (x > 1) :=
by
  sorry

end inequality_solution_set_l637_637270


namespace solve_x_l637_637462

theorem solve_x :
  ∃ x : ℚ, let a : ℚ × ℚ := (3, 4)
            let b : ℚ × ℚ := (2, -1)
            let v := (3 + 2 * x, 4 - x)
         in (v.1 * b.1 + v.2 * b.2 = 0) → x = -2 / 5 := 
begin
  sorry
end

end solve_x_l637_637462


namespace least_four_digit_9_heavy_l637_637690

def is_9_heavy (n : ℕ) : Prop := n % 9 > 5

def four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

theorem least_four_digit_9_heavy : ∃ n, four_digit n ∧ is_9_heavy n ∧ ∀ m, (four_digit m ∧ is_9_heavy m) → n ≤ m :=
by
  exists 1005
  sorry

end least_four_digit_9_heavy_l637_637690


namespace ratio_JC_KE_l637_637898

theorem ratio_JC_KE:
  ∀ (A B C D E F G H I J K : Type) 
    [has_distance A B C D E F G H]
    [line_segment AB 1]
    [line_segment BC 1]
    [line_segment CD 1]
    [line_segment DE 1]
    [line_segment EF 1]
    [line_segment FG 1]
    [line_segment GH 1]
    [points_not_collinear A I],
    (J ∈ line_segment I D) →
    (K ∈ line_segment I G) →
    parallel_line_segments (A I) (J C) →
    parallel_line_segments (A I) (K E) →
    JC / KE = 4 / 3 := 
by 
  sorry

end ratio_JC_KE_l637_637898


namespace product_of_two_numbers_l637_637607

theorem product_of_two_numbers (x y : ℕ) (h₁ : x + y = 40) (h₂ : x - y = 16) : x * y = 336 :=
sorry

end product_of_two_numbers_l637_637607


namespace midpoint_perpendicular_bisectors_l637_637853

-- Defining a type for points
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Defining a triangle
structure Triangle :=
  (A B C: Point)

-- Function to find the midpoint of two points
def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

-- Given a triangle ABC, defining the midpoints of its sides
def midpoints (T : Triangle) : (Point × Point × Point) :=
  (midpoint T.B T.C, midpoint T.C T.A, midpoint T.A T.B)

-- Definition for point being the intersection of perpendicular bisectors (skeletal)
def is_on_perpendicular_bisectors (P : Point) (T : Triangle) : Prop := sorry

-- Definition for a point being the incenter of a triangle (skeletal)
def is_incenter (P : Point) (T : Triangle) : Prop := sorry

-- The final theorem statement
theorem midpoint_perpendicular_bisectors (T : Triangle) :
  let (A1, B1, C1) := midpoints T in
  ∀ O : Point, is_on_perpendicular_bisectors O ⟨A1, B1, C1⟩ ↔ is_incenter O T :=
sorry

end midpoint_perpendicular_bisectors_l637_637853


namespace product_of_roots_l637_637376

-- Define the coefficients of the cubic equation
def a : ℝ := 2
def d : ℝ := 12

-- Define the cubic equation
def cubic_eq (x : ℝ) : ℝ := a * x^3 - 3 * x^2 - 8 * x + d

-- Prove the product of the roots is -6 using Vieta's formulas
theorem product_of_roots : -d / a = -6 := by
  sorry

end product_of_roots_l637_637376


namespace quadratic_two_distinct_real_roots_l637_637482

theorem quadratic_two_distinct_real_roots (k : ℝ) (h : k ≠ 0) : 
  (kx : ℝ) -> (a = k) -> (b = -2) -> (c = -1) -> (b^2 - 4*a*c > 0) -> (-2)^2 - 4*k*(-1) = (4 + 4*k > 0) := sorry

end quadratic_two_distinct_real_roots_l637_637482


namespace length_of_each_train_l637_637292

theorem length_of_each_train (L : ℝ) 
  (speed_faster : ℝ := 45 * 5 / 18) -- converting 45 km/hr to m/s
  (speed_slower : ℝ := 36 * 5 / 18) -- converting 36 km/hr to m/s
  (time : ℝ := 36) 
  (relative_speed : ℝ := speed_faster - speed_slower) 
  (total_distance : ℝ := relative_speed * time) 
  (length_each_train : ℝ := total_distance / 2) 
  : length_each_train = 45 := 
by 
  sorry

end length_of_each_train_l637_637292


namespace evaluate_expression_l637_637531

theorem evaluate_expression (x y : ℕ) (hx : 2^x ∣ 360 ∧ ¬ 2^(x+1) ∣ 360) (hy : 3^y ∣ 360 ∧ ¬ 3^(y+1) ∣ 360) :
  (3 / 7)^(y - x) = 7 / 3 := by
  sorry

end evaluate_expression_l637_637531


namespace binary_catalan_count_l637_637064

open Nat

/-- The number of 2n-bit binary numbers consisting of n ones and n zeros
such that when scanned from left to right, the cumulative number of 1s is never less than the cumulative number of 0s is given by the n-th Catalan number. -/
theorem binary_catalan_count (n : ℕ) : 
  let p_2n := 1 / (n + 1) * (factorial (2 * n)) / (factorial n * factorial n)
  in p_2n = (factorial (2 * n)) / (factorial (n + 1) * factorial n) :=
by sorry

end binary_catalan_count_l637_637064


namespace problem_equiv_l637_637612

theorem problem_equiv :
  ∃ n : ℕ, 
    10 ≤ n ∧ n < 100 ∧ 
    ((n % 10 = 6 ∧ ¬ (n % 7 = 0)) ∨ (n % 10 ≠ 6 ∧ n % 7 = 0)) ∧
    ((n > 26 ∧ ¬ (n % 10 = 8)) ∨ (n ≤ 26 ∧ n % 10 = 8)) ∧
    ((n % 13 = 0 ∧ ¬ (n < 27)) ∨ (n % 13 ≠ 0 ∧ n < 27)) ∧
    n = 91 :=
begin
  sorry
end

end problem_equiv_l637_637612


namespace cyclic_dominance_theorem_l637_637955

noncomputable def cyclic_dominance_possible : Prop :=
  ∃ (team1 team2 team3 : Finset ℕ), 
    team1.card = 3 ∧ team2.card = 3 ∧ team3.card = 3 ∧ 
    team1 ∪ team2 ∪ team3 = {1, 2, 3, 4, 5, 6, 7, 8, 9}.toFinset ∧
    ∑ x in team1, x = 15 ∧ ∑ x in team2, x = 15 ∧ ∑ x in team3, x = 15 ∧
    (∀ x ∈ team1, ∀ y ∈ team2, x > y) ∧
    (∀ x ∈ team2, ∀ y ∈ team3, x > y) ∧
    (∀ x ∈ team3, ∀ y ∈ team1, x > y)

theorem cyclic_dominance_theorem : cyclic_dominance_possible :=
sorry

end cyclic_dominance_theorem_l637_637955


namespace archer_total_fish_caught_l637_637359

noncomputable def total_fish_caught (initial : ℕ) (second_extra : ℕ) (third_round_percentage : ℕ) : ℕ :=
  let second_round := initial + second_extra
  let total_after_two_rounds := initial + second_round
  let third_round := second_round + (third_round_percentage * second_round) / 100
  total_after_two_rounds + third_round

theorem archer_total_fish_caught :
  total_fish_caught 8 12 60 = 60 :=
by
  -- Theorem statement to prove the total fish caught equals 60 given the conditions.
  sorry

end archer_total_fish_caught_l637_637359


namespace range_of_a_l637_637059

variable (a : ℝ)

def P (a : ℝ) : set ℝ := {x | abs (x - a) < 4}
def Q : set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem range_of_a (h : ∀ x, x ∈ P a → x ∈ Q) : -1 ≤ a ∧ a ≤ 5 :=
by
  sorry

end range_of_a_l637_637059


namespace max_value_of_f_l637_637388

-- Define the quadratic function f(x) = -3x^2 + 15x + 9
def f (x : ℝ) : ℝ := -3 * x^2 + 15 * x + 9

-- State the maximum value of the function f
theorem max_value_of_f : ∃ x_max, (∀ x, f x ≤ f x_max) ∧ f (5/2) = 111/4 := 
by {
  use 5/2,
  split,
  { intro x,
    have h : f x ≤ f (5/2), sorry },
  { simp [f], norm_num }
}

end max_value_of_f_l637_637388


namespace increasing_log_base_l637_637474

open Real

theorem increasing_log_base (a : ℝ) : 0 < a ∧ a < 1/2 ↔ ∀ x₁ x₂ : ℝ, x₁ < x₂ → (\log (1/2) a)^x₁ < (\log (1/2) a)^x₂ := sorry

end increasing_log_base_l637_637474


namespace simplest_quadratic_radical_l637_637642

theorem simplest_quadratic_radical (a : ℝ) :
  (∀ x : ℝ, x ∉ {sqrt (1 / 3), sqrt 147, sqrt (25 * x), sqrt (x ^ 2 + 1)} → ∃ (simplified : ℝ), simplified = sqrt (a ^ 2 + 1)) :=
by
  sorry

end simplest_quadratic_radical_l637_637642


namespace find_xiao_li_compensation_l637_637306

-- Define the conditions
variable (total_days : ℕ) (extra_days : ℕ) (extra_compensation : ℕ)
variable (daily_work : ℕ) (daily_reward : ℕ) (xiao_li_days : ℕ)

-- Define the total compensation for Xiao Li
def xiao_li_compensation (xiao_li_days daily_reward : ℕ) : ℕ := xiao_li_days * daily_reward

-- The theorem statement asserting the final answer
theorem find_xiao_li_compensation
  (h1 : total_days = 12)
  (h2 : extra_days = 3)
  (h3 : extra_compensation = 2700)
  (h4 : daily_work = 1)
  (h5 : daily_reward = 225)
  (h6 : xiao_li_days = 2)
  (h7 : (total_days - extra_days) * daily_work = xiao_li_days * daily_work):
  xiao_li_compensation xiao_li_days daily_reward = 450 := 
sorry

end find_xiao_li_compensation_l637_637306


namespace count_solution_pairs_number_of_solution_pairs_is_4_l637_637808

theorem count_solution_pairs : 
  (∃ (m n : ℤ), m + n = mn - 1) := sorry

theorem number_of_solution_pairs_is_4 : 
  (∃ (m n : ℤ), m + n = mn - 1 ∧ (m, n) ∈ {(3, 2), (2, 3), (-1, 0), (0, -1)} ∧
  (∀ (m n : ℤ), m + n = mn - 1 → (m, n) = (3, 2) ∨ (m, n) = (2, 3) ∨ (m, n) = (-1, 0) ∨ (m, n) = (0, -1))) :=
sorry

end count_solution_pairs_number_of_solution_pairs_is_4_l637_637808


namespace total_votes_l637_637844

theorem total_votes (V : ℕ) 
  (h1 : V * 45 / 100 + V * 25 / 100 + V * 15 / 100 + 180 + 50 = V) : 
  V = 1533 := 
by
  sorry

end total_votes_l637_637844


namespace collinear_J_seq_S_nth_point_relationship_l637_637253

noncomputable section

-- Definitions and conditions setting
variables {A B C T1 T2 T3 J S Jo : Point}
variables {T1' T2' T3' A1 B1 C1 : Point}
variables {n : ℕ}

-- Define the problem in Lean 4.
-- Assuming the required conditions
axiom incircle_touches_sides : ∀ (tri : Triangle A B C), IncircleTouchesSides tri T1 T2 T3
axiom symmetric_points_midpoints : SymmetricMidpoints A B C T1 T2 T3 T1' T2' T3'
axiom points_connected_to_vertices : ConnectPointsToVertices T1' T2' T3' A B C
axiom lines_intersect_at_J : LinesIntersectAtJ A B C T1' T2' T3' J
axiom form_triangle_A1B1C1 : FormTriangleWithParallels A B C A1 B1 C1
axiom iterative_draw_triangles : IterativelyDrawTriangles A B C A1 B1 C1 T1' T2' T3' n

axiom common_centroid_S : CommonCentroid A B C A1 B1 C1 S

-- Prove that J, J', J'', ..., J^(n) lie on a straight line
theorem collinear_J_seq_S : 
  Collinear (ChainPoints J J' J'' J^(n)) S :=
  sorry

-- Prove that J^(n)S = 2^(n+1) * JoS, where Jo is the center of incircle of triangle ABC and S is the centroid
theorem nth_point_relationship : 
  ∀ n, Distance (J^(n), S) = 2^(n+1) * Distance (Jo, S) :=
  sorry

end collinear_J_seq_S_nth_point_relationship_l637_637253


namespace max_f_l637_637236

open Real

noncomputable def f (x y z : ℝ) := (1 - y * z + z) * (1 - z * x + x) * (1 - x * y + y)

theorem max_f (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 1) :
  f x y z ≤ 1 ∧ (x = 1 ∧ y = 1 ∧ z = 1 → f x y z = 1) := sorry

end max_f_l637_637236


namespace max_acute_angles_in_convex_octagon_l637_637980

theorem max_acute_angles_in_convex_octagon : 
  ∀ (angles : Fin 8 → ℝ), (∀ i, 0 < angles i ∧ angles i < 180) → 
  (∃ n, (∀ i, 0 ≤ i < n → angles i < 90) ∧ 
  (∀ i, n ≤ i < 8 → 90 ≤ angles i < 180) ∧ 
  ∑ i, angles i = 1080 ∧ n ≤ 4) := 
by
  sorry

end max_acute_angles_in_convex_octagon_l637_637980


namespace abs_sum_eq_three_l637_637778

theorem abs_sum_eq_three
  {m n : ℤ}
  {a b c : ℤ}
  (h1 : ∀ x, (x^n + c)^m = (a * x^m + 1) * (b * x^m + 1))
  (h2 : n = 2)
  (h3 : n ≠ 0) :
  |a + b + c| = 3 :=
by
  sorry

end abs_sum_eq_three_l637_637778


namespace women_stockbrokers_2005_l637_637492

-- Define the context and conditions
def women_stockbrokers_2000 : ℕ := 10000
def percent_increase_2005 : ℕ := 100

-- Statement to prove the number of women stockbrokers in 2005
theorem women_stockbrokers_2005 : women_stockbrokers_2000 + women_stockbrokers_2000 * percent_increase_2005 / 100 = 20000 := by
  sorry

end women_stockbrokers_2005_l637_637492


namespace max_closed_broken_line_length_l637_637065

variables {n : ℕ}
variables {A : Fin (2 * n + 1) → ℝ × ℝ}

theorem max_closed_broken_line_length (h_convex: IsConvexPolygon (A : Fin (2 * n + 1) → ℝ × ℝ)) :
  ∀ (P : List ℕ), 
  (Set.Perm ⟦P⟧ (Finset.range (2 * n + 1)).toList ∧ 
   List.Nodup P ∧
   (P.length > 1 ∧ P.head = some 0 ∧ P.last = some 0)) →
  (length_of_broken_line (P.map (λ i, A ⟨i % (2 * n + 1), by linarith⟩)) ≤ 
  length_of_broken_line (List.of_fn (λ i, A ⟨i % (2 * n + 1), by linarith⟩)))
:=
sorry

end max_closed_broken_line_length_l637_637065


namespace exists_unit_sphere_containing_all_points_l637_637061

theorem exists_unit_sphere_containing_all_points (P : Fin 100 → ℝ³) 
  (h : ∀ (s : Finset (Fin 100)), s.card = 4 → ∃ (S : Sphere ℝ³), ∀ i ∈ s, S.contains (P i)) :
  ∃ (S : Sphere ℝ³), ∀ i, S.contains (P i) :=
sorry

end exists_unit_sphere_containing_all_points_l637_637061


namespace simplify_fraction_l637_637986

theorem simplify_fraction (a b : ℕ) (h : a = 2020) (h2 : b = 2018) :
  (2 ^ a - 2 ^ b) / (2 ^ a + 2 ^ b) = 3 / 5 := by
  sorry

end simplify_fraction_l637_637986


namespace interval_length_implies_difference_l637_637255

variable (c d : ℝ)

theorem interval_length_implies_difference (h1 : ∀ x : ℝ, c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d) (h2 : (d - c) / 3 = 15) : d - c = 45 := 
sorry

end interval_length_implies_difference_l637_637255


namespace number_of_integers_less_than_sqrt_five_l637_637052

theorem number_of_integers_less_than_sqrt_five :
  {x : ℤ | abs x < real.sqrt 5}.finite.to_finset.card = 5 :=
sorry

end number_of_integers_less_than_sqrt_five_l637_637052


namespace kathleen_allowance_l637_637863

theorem kathleen_allowance (x : ℝ) :
  let middle_school_allowance := 10
  let senior_year_allowance := 10 * x + 5
  let percentage_increase := ((senior_year_allowance - middle_school_allowance) / middle_school_allowance) * 100
  percentage_increase = 150 → x = 2 :=
by
  -- Definitions and conditions setup
  let middle_school_allowance := 10
  let senior_year_allowance := 10 * x + 5
  let percentage_increase := ((senior_year_allowance - middle_school_allowance) / middle_school_allowance) * 100
  intros h
  -- Skipping the proof
  sorry

end kathleen_allowance_l637_637863


namespace non_poli_sci_gpa_below_or_eq_3_is_10_l637_637564

-- Definitions based on conditions
def total_applicants : ℕ := 40
def poli_sci_majors : ℕ := 15
def gpa_above_3 : ℕ := 20
def poli_sci_gpa_above_3 : ℕ := 5

-- Derived conditions from the problem
def poli_sci_gpa_below_or_eq_3 : ℕ := poli_sci_majors - poli_sci_gpa_above_3
def total_gpa_below_or_eq_3 : ℕ := total_applicants - gpa_above_3
def non_poli_sci_gpa_below_or_eq_3 : ℕ := total_gpa_below_or_eq_3 - poli_sci_gpa_below_or_eq_3

-- Statement to be proven
theorem non_poli_sci_gpa_below_or_eq_3_is_10 : non_poli_sci_gpa_below_or_eq_3 = 10 := by
  sorry

end non_poli_sci_gpa_below_or_eq_3_is_10_l637_637564


namespace inequality_abc_distinct_positive_l637_637568

theorem inequality_abc_distinct_positive
  (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  (a^2 / b + b^2 / c + c^2 / d + d^2 / a > a + b + c + d) := 
by
  sorry

end inequality_abc_distinct_positive_l637_637568


namespace min_area_of_triangle_OPQ_l637_637567

def point (x y : ℝ) := (x, y)

def on_circle (P : ℝ × ℝ) : Prop := (P.1 + 3) ^ 2 + (P.2 - 1) ^ 2 = 2

def Q : ℝ × ℝ := (2, 2)

def O : ℝ × ℝ := (0, 0)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

noncomputable def area_of_triangle (O P Q : ℝ × ℝ) : ℝ :=
  1 / 2 * (abs ((Q.1 - O.1) * (P.2 - O.2) - (P.1 - O.1) * (Q.2 - O.2)))

theorem min_area_of_triangle_OPQ :
  ∃ (P : ℝ × ℝ), on_circle P → area_of_triangle O P Q = 2 :=
sorry

end min_area_of_triangle_OPQ_l637_637567


namespace kelly_spends_correct_amount_l637_637698

noncomputable def total_cost_with_discount : ℝ :=
  let mango_cost_per_pound := (0.60 : ℝ) * 2
  let orange_cost_per_pound := (0.40 : ℝ) * 4
  let mango_total_cost := 5 * mango_cost_per_pound
  let orange_total_cost := 5 * orange_cost_per_pound
  let total_cost_without_discount := mango_total_cost + orange_total_cost
  let discount := 0.10 * total_cost_without_discount
  let total_cost_with_discount := total_cost_without_discount - discount
  total_cost_with_discount

theorem kelly_spends_correct_amount :
  total_cost_with_discount = 12.60 := by
  sorry

end kelly_spends_correct_amount_l637_637698


namespace closest_integer_to_sum_l637_637738

noncomputable def sum_term (n : ℕ) : ℝ := 1 / (n^2 - 4)
noncomputable def sum_partial (start_n : ℕ) (end_n : ℕ) : ℝ := ∑ i in finset.range (end_n - start_n + 1), sum_term (start_n + i)

theorem closest_integer_to_sum :
  | (500 * sum_partial 2 5000 - 354 : ℝ) | < 1 :=
sorry

end closest_integer_to_sum_l637_637738


namespace ratio_of_c_and_d_l637_637726

theorem ratio_of_c_and_d (x y c d : ℝ) (hd : d ≠ 0) 
  (h1 : 3 * x + 2 * y = c) 
  (h2 : 4 * y - 6 * x = d) : c / d = -1 / 3 := 
sorry

end ratio_of_c_and_d_l637_637726


namespace find_sum_l637_637548

def a (n : ℕ) : ℚ := 1 / ((n + 1) * Real.sqrt n + n * Real.sqrt (n + 1))

theorem find_sum : (∑ n in Finset.range 99, a (n + 1)) = 9 / 10 := by
  sorry

end find_sum_l637_637548


namespace num_cheaper_books_l637_637384

def C (n : ℕ) : ℕ :=
  if n ≥ 1 ∧ n ≤ 30 then 15 * n
  else if n ≥ 31 ∧ n ≤ 65 then 13 * n
  else if n ≥ 66 then 11 * n
  else 0

theorem num_cheaper_books :
  (card { n : ℕ | C (n + 1) < C n && ((1 ≤ n ∧ n ≤ 30) ∨ (31 ≤ n ∧ n ≤ 65) ∨ (66 ≤ n)) } = 6) :=
by sorry

end num_cheaper_books_l637_637384


namespace nick_sold_fewer_bottles_l637_637229

open Real

def remy_morning_sales_bottles := 55
def price_per_bottle : Real := 0.50
def evening_sales : Real := 55
def evening_more_than_morning : Real := 3

def total_morning_sales : Real := evening_sales - evening_more_than_morning

def remy_morning_sales_dollars : Real := remy_morning_sales_bottles * price_per_bottle

def nick_morning_sales_dollars : Real := total_morning_sales - remy_morning_sales_dollars

def nick_morning_sales_bottles : Real := nick_morning_sales_dollars / price_per_bottle

def difference_in_bottles := remy_morning_sales_bottles - nick_morning_sales_bottles

theorem nick_sold_fewer_bottles : difference_in_bottles = 6 := by
  sorry

end nick_sold_fewer_bottles_l637_637229


namespace probability_of_100th_toss_heads_l637_637992

-- Define the conditions
def fair_coin (p : ℝ) : Prop := p = 1 / 2

-- Define the event of interest: the 100th coin toss
def independent_event (p : ℝ) (n : ℕ) (outcome : ℝ → Prop) : Prop :=
  ∀ i : ℕ, 1 ≤ i → i ≤ n → outcome p

-- The main theorem to prove
theorem probability_of_100th_toss_heads :
  ∀ p : ℝ,
  fair_coin p →
  independent_event p 99 (λ _ , true) →
  p = 1 / 2 :=
by
  intro p hf hc
  exact hf

end probability_of_100th_toss_heads_l637_637992


namespace line_PQ_correct_l637_637600

-- Definitions of the parabola and circle equations
def parabola (x : ℝ) : ℝ := x^2 / 2
def circle (x y : ℝ) : ℝ := (x - 3)^2 + y^2

-- Intersections of the parabola and circle
def P := (2 - sqrt 3, sqrt (2 * (2 - sqrt 3))) -- P(x1, y1)
def Q := (2 + sqrt 3, sqrt (2 * (2 + sqrt 3))) -- Q(x2, y2)

-- Formula for the line passing through P and Q
def line_through_PQ (x y : ℝ) : Prop := x - sqrt 3 * y + 1 = 0

-- Proving the equation of the line PQ is correct
theorem line_PQ_correct : line_through_PQ (fst P) (snd P) ∧ line_through_PQ (fst Q) (snd Q) :=
by
  unfold P Q line_through_PQ
  sorry

end line_PQ_correct_l637_637600


namespace piravena_flight_cost_l637_637363

noncomputable def cost_of_flight (distance_km : ℕ) (booking_fee : ℕ) (rate_per_km : ℕ) : ℕ :=
  booking_fee + (distance_km * rate_per_km / 100)

def check_cost_of_flight : Prop :=
  let distance_bc := 1000
  let booking_fee := 100
  let rate_per_km := 10
  cost_of_flight distance_bc booking_fee rate_per_km = 200

theorem piravena_flight_cost : check_cost_of_flight := 
by {
  sorry
}

end piravena_flight_cost_l637_637363


namespace shaded_area_approximation_l637_637224

theorem shaded_area_approximation :
  let initial_area := (1 / 2) * 8 * 8
  let one_iteration_shaded_area := initial_area / 4
  let r := 1 / 4
  let n := 100
  (initial_area * (1 - r ^ n) / (1 - r)) ≈ 10.67 := sorry

end shaded_area_approximation_l637_637224


namespace local_minimum_condition_l637_637824

theorem local_minimum_condition (f : ℝ → ℝ) (b : ℝ) :
  (∀ x : ℝ, f x = x^3 - 3 * b * x + b) →
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ ∀ y : ℝ, f y > f x) →
  b ∈ set.Ioo 0 1 :=
by
  sorry

end local_minimum_condition_l637_637824


namespace last_integer_in_halving_sequence_l637_637941

theorem last_integer_in_halving_sequence (a₀ : ℕ) (h₀ : a₀ = 2000000) : 
  ∃ n : ℕ, ∀ k : ℕ, a₀ / (2 ^ k) ∈ ℕ ∧ (a₀ / (2 ^ n) = 15625 ∧ a₀ / (2 ^ (n + 1)) < 1) :=
sorry

end last_integer_in_halving_sequence_l637_637941


namespace polygon_sides_10_sum_interior_angles_1440_l637_637339

-- Condition for Question 1: each exterior angle is 36 degrees
def each_exterior_angle (θ : ℝ) : Prop := θ = 36

-- Correct Answer 1: number of sides is 10
noncomputable def number_of_sides (n : ℕ) :=
  { n // ∃ θ, each_exterior_angle θ ∧ 360 / θ = n }

-- Correct Answer 2: sum of interior angles is 1440 degrees for a polygon with 10 sides
def sum_of_interior_angles (n : ℕ) : ℝ :=
  if n = 10 then (n - 2) * 180 else 0

-- Proof Problem 1: The number of sides is 10 given the each_exterior_angle is 36 degrees
theorem polygon_sides_10 (θ : ℝ) (h : each_exterior_angle θ) : number_of_sides (360 / θ) = 10 :=
by
  sorry

-- Proof Problem 2: The sum of interior angles is 1440 degrees for a polygon with 10 sides
theorem sum_interior_angles_1440 : sum_of_interior_angles 10 = 1440 :=
by
  sorry

end polygon_sides_10_sum_interior_angles_1440_l637_637339


namespace lucinda_jelly_beans_l637_637198

theorem lucinda_jelly_beans (g l : ℕ) 
  (h₁ : g = 3 * l) 
  (h₂ : g - 20 = 4 * (l - 20)) : 
  g = 180 := 
by 
  sorry

end lucinda_jelly_beans_l637_637198


namespace product_of_slopes_eq_neg_three_fourths_l637_637447

-- Definitions based on given conditions:
def curve (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Definition for slope:
def slope (P1 P2 : ℝ × ℝ) : ℝ := 
  if P1.1 = P2.1 then 0 else (P1.2 - P2.2) / (P1.1 - P2.1)

-- Prove the product of slopes of PM and PN:
theorem product_of_slopes_eq_neg_three_fourths (θ : ℝ) :
  let P := curve θ in
  slope P M * slope P N = -3/4 := 
begin
  sorry,
end

end product_of_slopes_eq_neg_three_fourths_l637_637447


namespace total_purchase_cost_l637_637165

-- Definitions for the quantities of the items
def quantity_chocolate_bars : ℕ := 10
def quantity_gummy_bears : ℕ := 10
def quantity_chocolate_chips : ℕ := 20

-- Definitions for the costs of the items
def cost_per_chocolate_bar : ℕ := 3
def cost_per_gummy_bear_pack : ℕ := 2
def cost_per_chocolate_chip_bag : ℕ := 5

-- Proof statement to be shown
theorem total_purchase_cost :
  (quantity_chocolate_bars * cost_per_chocolate_bar) + 
  (quantity_gummy_bears * cost_per_gummy_bear_pack) + 
  (quantity_chocolate_chips * cost_per_chocolate_chip_bag) = 150 :=
sorry

end total_purchase_cost_l637_637165


namespace cost_per_set_first_batch_min_selling_price_per_set_l637_637688

-- Part 1
theorem cost_per_set_first_batch :
  ∃ x : ℕ, (2500 / x) * 1.5 = (4500 / (x + 10)) ∧ x = 50 :=
by
  -- Convert the equation to Lean 4
  -- Solving should show x = 50
  sorry

-- Part 2
theorem min_selling_price_per_set :
  ∃ y : ℕ, 50 * y + 75 * y - 2500 - 4500 ≥ 1750 ∧ y ≥ 70 :=
by
  -- Convert the inequality to Lean 4
  -- Solving should show y ≥ 70
  sorry

end cost_per_set_first_batch_min_selling_price_per_set_l637_637688


namespace sum_alternating_g_l637_637883

def g (x : ℝ) : ℝ := x^3 * (1 - x)^3

theorem sum_alternating_g :
  ∑ k in Finset.range 2020, (-1)^(k+1) * g ((k + 1) / 2021) = 0 :=
by
  sorry

end sum_alternating_g_l637_637883


namespace product_of_roots_cubic_eq_l637_637379

-- Define a cubic equation 2x^3 - 3x^2 - 8x + 12 = 0
def cubic_eq : Polynomial ℝ := Polynomial.mk [2, -3, -8, 12]

-- Define a function to compute the product of the roots of a cubic equation using Vieta's formulas
def product_of_roots (p : Polynomial ℝ) : ℝ :=
  let a := p.coeff 3
  let d := p.coeff 0
  -d / a

-- The proof statement: Prove that the product of the roots of this specific cubic equation is -6
theorem product_of_roots_cubic_eq : product_of_roots cubic_eq = -6 := 
  sorry

end product_of_roots_cubic_eq_l637_637379


namespace volleyball_hockey_players_l637_637504

theorem volleyball_hockey_players :
  (∀ (players : ℕ) (glove_cost helmet_extra_cost total_cost : ℕ),
    glove_cost = 7 →
    helmet_extra_cost = 8 →
    total_cost = 3570 →
    ∀ (helmets_cost per_player_cost : ℕ),
      helmets_cost = glove_cost + helmet_extra_cost →
      per_player_cost = 2 * (glove_cost + helmets_cost) →
      total_cost = per_player_cost * players →
      players = 81) :=
begin
  intros players glove_cost helmet_extra_cost total_cost h_gc h_hec h_tc helmets_cost per_player_cost h_hc h_ppc h_total_cost,
  have h1 : helmets_cost = 15, by rw [h_gc, h_hec]; norm_num,
  have h2 : per_player_cost = 2 * (7 + 15), by rw h_ppc; norm_num,
  have h3 : total_cost = 3570, by assumption,
  have h4 : per_player_cost = 44, by rw h2; norm_num,
  have h5 : total_cost = players * 44, by rw h3; rw h4,
  have h6 : players = 81, by exact nat.div_eq_of_eq_mul_right (by norm_num) h_tc,
  exact h6,
end

end volleyball_hockey_players_l637_637504


namespace product_of_three_numbers_l637_637276

theorem product_of_three_numbers (x y z n : ℕ) 
  (h1 : x + y + z = 200)
  (h2 : n = 8 * x)
  (h3 : n = y - 5)
  (h4 : n = z + 5) :
  x * y * z = 372462 :=
by sorry

end product_of_three_numbers_l637_637276


namespace perform_sequences_count_l637_637729

def n_men : ℕ := 2
def n_women : ℕ := 3
def female_A_not_first := true -- Female A cannot be first
def males_not_consecutive := true -- Males cannot be consecutive

theorem perform_sequences_count : ∀ (students : list string),
  length students = 5 ∧ 
  count (λ s, s ∈ ["M1", "M2"]) students = n_men ∧
  count (λ s, s ∈ ["F1", "F2", "A"]) students = n_women ∧
  (female_A_not_first → students.head ≠ "A") ∧
  (males_not_consecutive → ∀ i < 4, students.nth i ∈ ["M1", "M2"] → students.nth (i+1) ∉ ["M1", "M2"]) 
→ (∃ perm, list.permutations ["M1", "M2", "F1", "F2", "A"] = students ∧ length perm = 60) :=
by
  sorry

end perform_sequences_count_l637_637729


namespace zeros_outside_curve_l637_637901

noncomputable def P (z : ℂ) (θ : Fin (n + 1) → ℝ) : ℂ :=
∑ i in Finset.range (n + 1), (z ^ (n - i)) * (Real.cos (θ i)) - 2

theorem zeros_outside_curve (n : ℕ) (θ : Fin (n + 1) → ℝ) (z0 : ℂ) :
  P n θ z0 = 0 → |z0| > 1/2 :=
sorry

end zeros_outside_curve_l637_637901


namespace find_x_l637_637119

-- Definitions for the vectors and their lengths
def a : ℝ × ℝ := (2, -4)
def b (x : ℝ) : ℝ × ℝ := (6, x)

-- Squaring the length function 
def length_squared (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

-- Condition given in the problem
theorem find_x (x : ℝ) (h : length_squared (a.1 + b x.1, a.2 + b x.2) = length_squared (a.1 - b x.1, a.2 - b x.2)) : x = 3 := by
  sorry

end find_x_l637_637119


namespace exists_n_for_ratio_l637_637604

-- Defining the sequence according to the problem statement
def u : ℕ → ℕ
| 0        := 1
| (2*n)    := u n + u (n - 1)
| (2*n+1)  := u n

-- The theorem statement
theorem exists_n_for_ratio (k : ℚ) (hk : 0 < k) : ∃ n : ℕ, (u n : ℚ) / u (n + 1) = k :=
sorry

end exists_n_for_ratio_l637_637604


namespace polynomial_identity_l637_637922

theorem polynomial_identity (x : ℝ) (h₁ : x ^ 2018 - 3 * x ^ 2 + 1 = 0) (h₂ : x ≠ 1) :
  x ^ 2017 + x ^ 2016 + ... + x + 1 = 3 * x + 3 :=
by
  sorry

end polynomial_identity_l637_637922


namespace triangle_propositions_correct_l637_637855

variable {A B C a b c : ℝ}

-- Condition: Sum of internal angles in a triangle
axiom sum_of_angles : A + B + C = Real.pi

-- Propositions
def propA : Prop := Real.sin (B + C) = Real.sin A
def propB : Prop := Real.cos (B + C) = Real.cos A
def propC : Prop := a^2 + b^2 = c^2 → B = Real.pi / 2 ∨ C = Real.pi / 2 ∨ A = Real.pi / 2
def propD : Prop := a^2 + b^2 < c^2 → C > Real.pi / 2

-- Proof (to be filled in)
theorem triangle_propositions_correct 
  (hA : propA) 
  (hC : propC)
  (hD : propD) : 
  (sum_of_angles ∧ hA ∧ hC ∧ hD) :=
by
  sorry

end triangle_propositions_correct_l637_637855


namespace prob_euler_totient_l637_637749

-- Define Euler's totient function
noncomputable def euler_totient (n : ℕ) : ℕ := 
  (Finset.range (n + 1)).filter (Nat.coprime n).card

-- Problem statement in Lean
theorem prob_euler_totient (n : ℕ) (h : n > 0) :
  euler_totient (2019 * n) = euler_totient (n ^ 2) ↔ 
  n = 1346 ∨ n = 2016 ∨ n = 2019 := 
sorry

end prob_euler_totient_l637_637749


namespace student_attempts_24_sums_l637_637687

def twice_wrong_as_right (x wrong_attempts : ℕ) : Prop :=
  wrong_attempts = 2 * x

def total_attempts (right_attempts wrong_attempts total : ℕ) : Prop :=
  right_attempts + wrong_attempts = total

theorem student_attempts_24_sums (x : ℕ) 
  (H1 : ∃ wrong_attempts, twice_wrong_as_right x wrong_attempts)
  (H2 : total_attempts x (classical.some H1) 24) :
  x = 8 :=
sorry

end student_attempts_24_sums_l637_637687


namespace max_students_l637_637241

-- Defining the problem's conditions
def cost_bus_rental : ℕ := 100
def max_capacity_students : ℕ := 25
def cost_per_student : ℕ := 10
def teacher_admission_cost : ℕ := 0
def total_budget : ℕ := 350

-- The Lean proof problem
theorem max_students (bus_cost : ℕ) (student_capacity : ℕ) (student_cost : ℕ) (teacher_cost : ℕ) (budget : ℕ) :
  bus_cost = cost_bus_rental → 
  student_capacity = max_capacity_students →
  student_cost = cost_per_student →
  teacher_cost = teacher_admission_cost →
  budget = total_budget →
  (student_capacity ≤ (budget - bus_cost) / student_cost) → 
  ∃ n : ℕ, n = student_capacity ∧ n ≤ (budget - bus_cost) / student_cost :=
by
  intros
  sorry

end max_students_l637_637241


namespace trailer_home_count_3_years_ago_l637_637959

noncomputable def original_trailer_count : ℕ := 25
def original_average_age (years_ago: ℕ) : ℕ := 15
def current_average_age : ℕ := 12

theorem trailer_home_count_3_years_ago (year_count: ℕ) :
  (year_count + original_trailer_count = 42) ->
  year_count = 17 :=
begin
  intro h,
  have h1: (original_trailer_count * (original_average_age 3) + year_count * 3) / (original_trailer_count + year_count) = current_average_age,
  { sorry },
  linarith,
end

end trailer_home_count_3_years_ago_l637_637959


namespace scientific_notation_of_0_000000081_l637_637246

theorem scientific_notation_of_0_000000081 :
  ∃ a n, (0.000000081 : ℝ) = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ n = -8 ∧ a = 8.1 :=
by
  sorry

end scientific_notation_of_0_000000081_l637_637246


namespace selling_price_is_1020_l637_637307

def cost_price := 1200
def loss_percentage := 15

def loss_amount (cost_price : ℕ) (loss_percentage : ℕ) : ℝ :=
  (loss_percentage / 100) * cost_price

def selling_price (cost_price : ℕ) (loss_amount : ℝ) : ℝ :=
  cost_price - loss_amount

theorem selling_price_is_1020 : selling_price cost_price (loss_amount cost_price loss_percentage) = 1020 := by
  sorry

end selling_price_is_1020_l637_637307


namespace approximate_smoking_percentage_l637_637730

theorem approximate_smoking_percentage (N : ℕ) (total_yes : ℕ) (prob_odd_id : ℚ) (prob_smoked : ℚ) 
    (odd_id_yes_answers : ℕ) (smoked_yes_answers : ℕ) :
  N = 300 →
  total_yes = 80 →
  prob_odd_id = (1/2 : ℚ) →
  prob_smoked = (1/2 : ℚ) →
  odd_id_yes_answers = total_yes / 2 →
  smoked_yes_answers = total_yes - odd_id_yes_answers →
  smoked_yes_answers / N * 100 ≈ 13.33 :=
by
  intros hN htotal hprob_odd hprob_smoked hood_id_yes hsmoked_yes
  simp [hN, htotal, hprob_odd, hprob_smoked, hood_id_yes, hsmoked_yes]
  sorry

end approximate_smoking_percentage_l637_637730


namespace solution_set_of_inequality_l637_637418

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

def deriv_exists (x : ℝ) : Prop := ∃ g : ℝ → ℝ, deriv f x = g x

theorem solution_set_of_inequality :
  (∀ x, deriv_exists x) →
  (∀ x, f x - f (-x) = 2 * x ^ 3) →
  (∀ x ≥ 0, deriv f x > 3 * x ^ 2) →
  (∀ x, f x - f (x - 1) > 3 * x ^ 2 - 3 * x + 1 ↔ x > 1 / 2) :=
by
  sorry

end solution_set_of_inequality_l637_637418


namespace arithmetic_sequences_integer_ratio_count_l637_637114

theorem arithmetic_sequences_integer_ratio_count
  (a b : ℕ → ℕ) (A B : ℕ → ℕ)
  (hA : ∀ n, A n = n * (2 * a n - 1))
  (hB : ∀ n, B n = n * (2 * b n - 1))
  (h_ratio : ∀ n, (A n) / (B n) = (7 * n + 45) / (n + 3)) :
  {n : ℕ | ∃ m : ℕ, a n / b n = m}.to_finset.card = 5 :=
begin
  sorry
end

end arithmetic_sequences_integer_ratio_count_l637_637114


namespace sum_of_sequences_l637_637707

-- Define the sequences and their type
def seq1 : List ℕ := [2, 12, 22, 32, 42]
def seq2 : List ℕ := [10, 20, 30, 40, 50]

-- The property we wish to prove
theorem sum_of_sequences : seq1.sum + seq2.sum = 260 :=
by
  sorry

end sum_of_sequences_l637_637707


namespace trig_identity_l637_637001

theorem trig_identity 
  (sin cos : ℝ → ℝ)
  (h_sin_add : ∀ A B : ℝ, sin (A + B) = sin A * cos B + cos A * sin B)
  (sin36 : sin (36 * real.pi / 180) = 3^(1/2) / 4)
  (cos24 : cos (24 * real.pi / 180) = √((8 - 2 * 3^(1/2)) / 4))
  (sin156 : sin (156 * real.pi / 180) = 3^(1/2) / 4)
  (cos36 : cos (36 * real.pi / 180) = √((8 + 2 * 3^(1/2)) / 4))
  (sin60 : sin (60 * real.pi / 180) = (sqrt 3) / 2) :
  sin (36 * real.pi / 180) * cos (24 * real.pi / 180) + cos (36 * real.pi / 180) * sin (156 * real.pi / 180) = (sqrt 3) / 2 := 
by sorry

end trig_identity_l637_637001


namespace inequality_l637_637658

variables {Point : Type} [InnerProductSpace ℝ Point]

-- Definitions for distances between points
def dist (a b : Point) : ℝ := real.sqrt ((a - b) • (a - b))

-- Condition 1: Four points A, B, C, D
variables (A B C D : Point)

-- Condition 2: Points do not lie on the same plane
def not_coplanar (A B C D : Point) : Prop :=
  ∃ (normal_vector : Point), 
    normal_vector ≠ 0 ∧ 
    (dot_product normal_vector (B - A) = 0) ∧ 
    (dot_product normal_vector (C - A) = 0) ∧ 
    (dot_product normal_vector (D - A) ≠ 0)

noncomputable def AB := dist A B
noncomputable def AC := dist A C
noncomputable def AD := dist A D
noncomputable def BC := dist B C
noncomputable def BD := dist B D
noncomputable def CD := dist C D

theorem inequality (h : not_coplanar A B C D) : 
  AB * CD + AC * BD > AD * BC :=
sorry

end inequality_l637_637658


namespace sum_of_infinite_perimeters_l637_637354

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_infinite_perimeters (a : ℝ) :
  let first_perimeter := 3 * a
  let common_ratio := (1/3 : ℝ)
  let S := geometric_series_sum first_perimeter common_ratio 0
  S = (9 * a / 2) :=
by
  sorry

end sum_of_infinite_perimeters_l637_637354


namespace sufficiently_large_sum_l637_637187

section problem

def T := {x : ℕ | ∃ α β γ : ℕ, (α + β + γ > 0) ∧ (x = 2^α * 5^β * 7^γ)}

theorem sufficiently_large_sum (n : ℕ) :
  n > 250 → ∃ (summands : Finset ℕ), (∀ t ∈ summands, t ∈ T) ∧ 
    (summands.sum id = n) ∧ (∀ a b ∈ summands, a ≠ b → ¬ (a ∣ b ∨ b ∣ a)) :=
by
  sorry

end problem

end sufficiently_large_sum_l637_637187


namespace basketball_game_l637_637834

theorem basketball_game (x d e f : ℕ) (others_points : ℕ) (nine_members : Fin 9 → ℕ)
(hd : d = x / 3)
(he : e = 3 * x / 8)
(hf : f = 18)
(h_others : others_points = d + e + f)
(h_total : ∑ i, nine_members i = others_points)
(h_limit : ∀ i, nine_members i ≤ 3) : 
others_points = 21 :=
begin
  sorry
end

end basketball_game_l637_637834


namespace range_of_a_1_range_of_a_2_l637_637103

-- Definitions based on conditions in a)

def func_domain (a : ℝ) (x : ℝ) : Prop := a * x^2 - 2 * x + 2 > 0

def Q (a : ℝ) : Set ℝ := {x | func_domain a x}

-- Problem 1
theorem range_of_a_1 (a : ℝ) (h1 : a > 0) (h2 : Disjoint {x | 2 ≤ x ∧ x ≤ 3} (Q a)) :
  0 < a ∧ a ≤ 4/9 := sorry

-- Problem 2
theorem range_of_a_2 (a : ℝ) (h : {x | 2 ≤ x ∧ x ≤ 3} ⊆ Q a) :
  a > 1/2 := sorry

end range_of_a_1_range_of_a_2_l637_637103


namespace max_marks_l637_637345

theorem max_marks (M : ℝ) : 0.33 * M = 59 + 40 → M = 300 :=
by
  sorry

end max_marks_l637_637345


namespace eggs_sold_total_l637_637560

theorem eggs_sold_total (x : ℕ) (h1 : x > 1)
    (h2: (let y := (x - 1) / 2 + 1 in (y - 1) / 2 + 1 = 1)) :
    x - 1 = 9 :=
by
  sorry

end eggs_sold_total_l637_637560


namespace leading_coefficient_is_neg7_l637_637037

noncomputable def leading_coefficient_of_polynomial : ℤ :=
  let poly := -3 * (X^4 - 2 * X^3 + 3 * X) + 8 * (X^4 + 5) - 4 * (3 * X^4 + X^3 + 1)
  in if h : (coeff poly 4) ≠ 0 then coeff poly 4 else 0

theorem leading_coefficient_is_neg7 : leading_coefficient_of_polynomial = -7 := by
  sorry

end leading_coefficient_is_neg7_l637_637037


namespace selling_price_to_achieve_profit_l637_637340

theorem selling_price_to_achieve_profit (num_pencils : ℝ) (cost_per_pencil : ℝ) (desired_profit : ℝ) (selling_price : ℝ) :
  num_pencils = 1800 →
  cost_per_pencil = 0.15 →
  desired_profit = 100 →
  selling_price = 0.21 :=
by
  sorry

end selling_price_to_achieve_profit_l637_637340


namespace angle_neg510_in_third_quadrant_l637_637661

def first_quadrant (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 90

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ < 180

def third_quadrant (θ : ℝ) : Prop :=
  180 ≤ θ ∧ θ < 270

def fourth_quadrant (θ : ℝ) : Prop :=
  270 ≤ θ ∧ θ < 360

def equivalent_angle (θ : ℝ) : ℝ :=
  (θ % 360 + 360) % 360

theorem angle_neg510_in_third_quadrant :
  third_quadrant (equivalent_angle (-510)) :=
by
  sorry

end angle_neg510_in_third_quadrant_l637_637661


namespace medal_ratio_l637_637163

theorem medal_ratio (total_medals : ℕ) (track_medals : ℕ) (badminton_medals : ℕ) (swimming_medals : ℕ) 
  (h1 : total_medals = 20) 
  (h2 : track_medals = 5) 
  (h3 : badminton_medals = 5) 
  (h4 : swimming_medals = total_medals - track_medals - badminton_medals) : 
  swimming_medals / track_medals = 2 := 
by 
  sorry

end medal_ratio_l637_637163


namespace expand_polynomial_l637_637734

theorem expand_polynomial (x : ℝ) :
  (x^15 - 2 * x^7 + x^(-3) - 8) * (3 * x^3) = 3 * x^18 - 6 * x^10 - 24 * x^3 + 3 :=
by
  sorry

end expand_polynomial_l637_637734


namespace t_n_lt_s_n_div_2_l637_637542

-- Definitions and imports
noncomputable def q : ℝ := 1 / 3

def a_n (n : ℕ) : ℝ := q^(n-1)
def b_n (n : ℕ) : ℝ := n * q^n
def S_n (n : ℕ) : ℝ := (3 / 2) * (1 - q^n)
def T_n (n : ℕ) : ℝ := 
  ((3 : ℝ) / 4) - ((1 : ℝ) / 4) * (q^(n-1)) - (n / 2) * q^n

-- Theorem declaration
theorem t_n_lt_s_n_div_2 (n : ℕ) : T_n n < S_n n / 2 := sorry

end t_n_lt_s_n_div_2_l637_637542


namespace salary_increase_percentage_l637_637558

variable {P : ℝ} (initial_salary : P > 0)

def salary_after_first_year (P: ℝ) : ℝ :=
  P * 1.12

def salary_after_second_year (P: ℝ) : ℝ :=
  (salary_after_first_year P) * 1.12

def salary_after_third_year (P: ℝ) : ℝ :=
  (salary_after_second_year P) * 1.15

theorem salary_increase_percentage (P: ℝ) (h: P > 0) : 
  (salary_after_third_year P - P) / P * 100 = 44 :=
by 
  sorry

end salary_increase_percentage_l637_637558


namespace quadratic_root_range_l637_637477

theorem quadratic_root_range (k : ℝ) (hk : k ≠ 0) (h : (4 + 4 * k) > 0) : k > -1 :=
by sorry

end quadratic_root_range_l637_637477


namespace weight_shifted_count_l637_637676

def is_weight_shifted (a b x y : ℕ) : Prop :=
  a + b = 2 * (x + y) ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9

theorem weight_shifted_count : 
  ∃ count : ℕ, count = 225 ∧ 
  (∀ (a b x y : ℕ), is_weight_shifted a b x y → count = 225) := 
sorry

end weight_shifted_count_l637_637676


namespace paco_countertop_total_weight_l637_637565

theorem paco_countertop_total_weight :
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 = 0.75 :=
sorry

end paco_countertop_total_weight_l637_637565


namespace part_i3_1_part_i3_2_part_i3_3_part_i3_4_l637_637813

section PartI3_1
def a := 9
theorem part_i3_1 : a = 9 := by
  sorry
end PartI3_1

section PartI3_2
variable b : ℝ
variable (0 : ℝ) < 4 * b < 90
variable sin cos sqrt tan : ℝ → ℝ
theorem part_i3_2 : a = 9 → (0 < 4 * b < 90) → (sin (4 * b) / cos (4 * b) = sqrt (sqrt a)) → b = 15 := by
  sorry
end PartI3_2

section PartI3_3
def sequence := [3/12, 7/34, 11/56, 15/78]

theorem part_i3_3 : ∀ c,
  sequence = [3 / 12, 7 / 34, c / 56, 15 / 78] → c = 11 := by
  sorry
end PartI3_3

section PartI3_4
variable O B C : Type
variable (angle : O → B → C → ℝ)
theorem part_i3_4 : ∀ d,
  angle O B C = 3 * c → 
  c = 11 → 
  (∡ O B C = d) → 
  d = 73.5 := by
  sorry
end PartI3_4

end part_i3_1_part_i3_2_part_i3_3_part_i3_4_l637_637813


namespace vasya_painting_l637_637965

-- Definitions
def color := fin 3  -- there are 3 colors
def board := fin 10  -- fence consists of 10 boards

-- Conditions
def adjacent_diff (f : board → color) : Prop :=
  ∀ i : board, i.1 < 9 → f ⟨i.1, sorry⟩ ≠ f ⟨i.1 + 1, sorry⟩

def three_colors_used (f : board → color) : Prop :=
  ∃ c1 c2 c3 : color, c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
  ∃ i1 i2 i3 : board, f i1 = c1 ∧ f i2 = c2 ∧ f i3 = c3

-- The target statement
theorem vasya_painting : ∃ f : board → color,
  adjacent_diff f ∧ three_colors_used f ∧ (set.univ → (3 * 2^9 - 6) = 1530) := sorry

end vasya_painting_l637_637965


namespace shoe_size_linear_relation_l637_637603

theorem shoe_size_linear_relation (k b : ℝ) 
  (h1 : 23 * k + b = 36) 
  (h2 : 26 * k + b = 42) 
  (h3 : 39 = 2 * 24.5 - 10) :
  ∀ x, x = 24.5 → 2 * x - 10 = 39 :=
by {
  intro x,
  intro hx,
  rw hx,
  exact h3,
}

end shoe_size_linear_relation_l637_637603


namespace average_of_remaining_numbers_l637_637586

theorem average_of_remaining_numbers 
  (S : ℕ) (H1 : S = 1080) (H2 : S ≠ 0):
  ((S - 80 - 84) / 10 = 91.6) :=
by
  have rem_sum : S - 80 - 84 = 1080 - 80 - 84 := by rw [H1]
  have average := (1080 - 80 - 84) / 10 = 91.6
  sorry -- Placeholder for further steps

end average_of_remaining_numbers_l637_637586


namespace necessary_but_not_sufficient_l637_637126

   theorem necessary_but_not_sufficient (a : ℝ) : a^2 > a → (a > 1) :=
   by {
     sorry
   }
   
end necessary_but_not_sufficient_l637_637126


namespace find_first_number_l637_637990

variable (x y : ℕ)

theorem find_first_number (h1 : y = 11) (h2 : x + (y + 3) = 19) : x = 5 :=
by
  sorry

end find_first_number_l637_637990


namespace radius_approx_eq_twelve_l637_637667

noncomputable def radius_of_semicircle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem radius_approx_eq_twelve (P : ℝ) (hP : P = 61.69911184307752) : radius_of_semicircle P ≈ 12 :=
by
  have r := radius_of_semicircle P
  show r ≈ 12
  sorry

end radius_approx_eq_twelve_l637_637667


namespace inequality_of_positive_numbers_l637_637776

theorem inequality_of_positive_numbers (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c :=
sorry

end inequality_of_positive_numbers_l637_637776


namespace non_congruent_squares_on_6x6_grid_l637_637807

theorem non_congruent_squares_on_6x6_grid : 
  ∃ n : ℕ, n = 101 ∧ 
  (∀ (L : fin 7 × fin 7) (side : ℝ), 
    (∃ (a b c d : ℕ), 
       a ≤ 5 ∧ b ≤ 5 ∧ c ≤ 5 ∧ d ≤ 5 ∧ 
       (is_square_with_vertices L side a b c d)) → 
       count_non_congruent_squares 6 = n)
:=
sorry

end non_congruent_squares_on_6x6_grid_l637_637807


namespace number_of_initial_chairs_l637_637500

variable (C : ℕ) -- Number of chairs Kenzo initially had.
variable (legsPerChair legsPerTable numTables totalLegsRemaining : ℕ)
variable (damageFraction : ℚ)

-- Defining the conditions from the problem
axiom legsPerChair_def : legsPerChair = 5
axiom legsPerTable_def : legsPerTable = 3
axiom numTables_def : numTables = 20
axiom damageFraction_def : damageFraction = 0.4
axiom totalLegsRemaining_def : totalLegsRemaining = 300

-- Defining the number of legs remaining from chairs
def legsFromChairsRemaining := (1 - damageFraction) * (legsPerChair * C)

-- Total legs from tables
def legsFromTables := legsPerTable * numTables

-- Total legs equation based on the remaining legs
axiom totalLegs_def : legsFromChairsRemaining + legsFromTables = totalLegsRemaining

-- The theorem we need to prove
theorem number_of_initial_chairs : C = 80 :=
by
  -- The proof goes here
  sorry

end number_of_initial_chairs_l637_637500


namespace distance_between_parallel_lines_l637_637247

theorem distance_between_parallel_lines :
  let A := 6
  let B := 8
  let C1 := -3
  let C2 := 5
  distance_between_lines (6 * x + 8 * y - 3 = 0) (6 * x + 8 * y + 5 = 0) = 4 / 5 :=
by
  sorry

end distance_between_parallel_lines_l637_637247


namespace minimum_rebate_rate_l637_637587

open Real

noncomputable def rebate_rate (s p_M p_N p: ℝ) : ℝ := 100 * (p_M + p_N - p) / s

theorem minimum_rebate_rate 
  (s p_M p_N p : ℝ)
  (h_M : 0.19 * 0.4 * s ≤ p_M ∧ p_M ≤ 0.24 * 0.4 * s)
  (h_N : 0.29 * 0.6 * s ≤ p_N ∧ p_N ≤ 0.34 * 0.6 * s)
  (h_total : 0.10 * s ≤ p ∧ p ≤ 0.15 * s) :
  ∃ r : ℝ, r = rebate_rate s p_M p_N p ∧ 0.1 ≤ r ∧ r ≤ 0.2 :=
sorry

end minimum_rebate_rate_l637_637587


namespace common_z_values_l637_637025

theorem common_z_values (z : ℝ) :
  (∃ x : ℝ, x^2 + z^2 = 9 ∧ x^2 = 4*z - 5) ↔ (z = -2 + 3*Real.sqrt 2 ∨ z = -2 - 3*Real.sqrt 2) := 
sorry

end common_z_values_l637_637025


namespace face_opposite_to_turquoise_is_pink_l637_637580

-- Declare the inductive type for the color of the face
inductive Color
| P -- Pink
| V -- Violet
| T -- Turquoise
| O -- Orange

open Color

-- Define the setup conditions of the problem
def cube_faces : List Color :=
  [P, P, P, V, V, T, O]

-- Define the positions of the faces for the particular folded cube configuration
-- Assuming the function cube_configuration gives the face opposite to a given face.
axiom cube_configuration : Color → Color

-- State the main theorem regarding the opposite face
theorem face_opposite_to_turquoise_is_pink : cube_configuration T = P :=
sorry

end face_opposite_to_turquoise_is_pink_l637_637580


namespace kamal_marks_physics_l637_637525

-- Define the marks in subjects
def marks_english := 66
def marks_mathematics := 65
def marks_chemistry := 62
def marks_biology := 75
def average_marks := 69
def number_of_subjects := 5

-- Calculate the total marks from the average
def total_marks := average_marks * number_of_subjects

-- Calculate the known total marks
def known_total_marks := marks_english + marks_mathematics + marks_chemistry + marks_biology

-- Define Kamal's marks in Physics
def marks_physics := total_marks - known_total_marks

-- Prove the marks in Physics are 77
theorem kamal_marks_physics : marks_physics = 77 := by
  sorry

end kamal_marks_physics_l637_637525


namespace min_rubles_to_50_points_l637_637323

theorem min_rubles_to_50_points : 
  ∃ (steps : Nat), 
    (∀ (points rubles: Nat), 
      (points, rubles) = Nat.iterate steps (λ (pr: Nat × Nat), if pr.fst < 50 
                                                           then if pr.fst < 25 
                                                                then (2 * pr.fst, pr.snd + 2) 
                                                                else (pr.fst + 1, pr.snd + 1) 
                                                           else (0, pr.snd))
      (0, 0)) ∧ points = 50 ∧ rubles = 11 :=
sorry

end min_rubles_to_50_points_l637_637323


namespace extremum_at_minus_four_thirds_monotonicity_g_l637_637100

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x^3 + x^2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x a * Real.exp x

theorem extremum_at_minus_four_thirds (a : ℝ) (h : deriv (f x) (a) = 0) :
    a = 1/2 :=
by
  let df := fun x => 3 * a * x^2 + 2 * x
  have h : df (-4 / 3) = 0,
  exact (by norm_num : 0 = 3 * a * (16 / 9) + 2 * (-4 / 3)),
  sorry

theorem monotonicity_g (a : ℝ) (h : a = 1/2) :
    (∀ x, x < -4 → deriv (g x a) < 0) ∧
    (∀ x, -4 < x ∧ x < -1 → deriv (g x a) > 0) ∧
    (∀ x, -1 < x ∧ x < 0 → deriv (g x a) < 0) ∧
    (∀ x, x > 0 → deriv (g x a) > 0) :=
by
  have h_g : g x 1/2 := (f x 1/2 * Real.exp x),
  sorry

end extremum_at_minus_four_thirds_monotonicity_g_l637_637100


namespace rectangle_area_change_l637_637313

theorem rectangle_area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let A := L * B
  let L' := 1.15 * L
  let B' := 0.80 * B
  let A' := L' * B'
  A' = 0.92 * A :=
by
  let A := L * B
  let L' := 1.15 * L
  let B' := 0.80 * B
  let A' := L' * B'
  show A' = 0.92 * A
  sorry

end rectangle_area_change_l637_637313


namespace total_candies_l637_637890

theorem total_candies (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 5) : 
  (2 * x + 6) + (4 * y + 7) + (3 * z - 5) = 37 :=
by
  rw [hx, hy, hz]
  norm_num
  sorry

end total_candies_l637_637890


namespace verify_sum_l637_637396

-- Definitions and conditions
def C : ℕ := 1
def D : ℕ := 2
def E : ℕ := 5

-- Base-6 addition representation
def is_valid_base_6_addition (a b c d : ℕ) : Prop :=
  (a + b) % 6 = c ∧ (a + b) / 6 = d

-- Given the addition problem:
def addition_problem : Prop :=
  is_valid_base_6_addition 2 5 C 0 ∧
  is_valid_base_6_addition 4 C E 0 ∧
  is_valid_base_6_addition D 2 4 0

-- Goal to prove
theorem verify_sum : addition_problem → C + D + E = 6 :=
by
  sorry

end verify_sum_l637_637396


namespace nature_of_roots_l637_637581

def polynomial : Polynomial ℝ := Polynomial.C 1 * Polynomial.X ^ 3 + Polynomial.C 3 * Polynomial.X ^ 2
                      - Polynomial.C 4 * Polynomial.X - Polynomial.C 12

theorem nature_of_roots (p : Polynomial ℝ) (h : p = polynomial) :
  ∃ (a b c : ℝ), a < 0 ∧ b < 0 ∧ c > 0 ∧ p.eval a = 0 ∧ p.eval b = 0 ∧ p.eval c = 0 :=
by
  rw [h]
  sorry

end nature_of_roots_l637_637581


namespace number_of_points_C_l637_637143

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem number_of_points_C (A B : ℝ × ℝ) (d_AB : distance A B = 12)
  (P : ∀ C : ℝ × ℝ, distance A C + distance B C = 40)
  (Area : ∀ C : ℝ × ℝ, abs (12 * (C.2)) = 240):
  ∃! C1 C2 : ℝ × ℝ, P C1 ∧ Area C1 ∧ P C2 ∧ Area C2 ∧ distance C1 C2 ≠ 0 := 
sorry

end number_of_points_C_l637_637143


namespace problem_l637_637870

-- Definitions of the sum functions for the arithmetic sequences
def Sn (a : ℕ → ℕ) (n : ℕ) := ∑ i in finset.range (n+1), a i
def Tn (b : ℕ → ℕ) (n : ℕ) := ∑ i in finset.range (n+1), b i

-- The condition given in the problem
axiom condition (a b : ℕ → ℕ) (n : ℕ) : Sn a n / Tn b n = (2 * n - 1) / (3 * n + 2)

-- The statement to prove
theorem problem (a b : ℕ → ℕ) :
  (Sn a 6 + Sn a 6) / (Tn b 6 + Tn b 6) = 25 / 41 := sorry

end problem_l637_637870


namespace triangle_possible_iff_h_gt_e_minus_3_l637_637096

noncomputable def f (x h : ℝ) : ℝ := -Real.log x + x + h

theorem triangle_possible_iff_h_gt_e_minus_3 (h : ℝ) :
  (∀ (a b c : ℝ), a ∈ Icc (1 / Real.exp 1) Real.exp 1 ∧ b ∈ Icc (1 / Real.exp 1) Real.exp 1 ∧ c ∈ Icc (1 / Real.exp 1) Real.exp 1 → 
  ∃ (x y z : ℝ), x = f a h ∧ y = f b h ∧ z = f c h ∧ x + y > z ∧ x + z > y ∧ y + z > x) ↔ h > Real.exp 1 - 3 :=
by
  sorry

end triangle_possible_iff_h_gt_e_minus_3_l637_637096


namespace congruent_angles_equiv_same_remainder_mod_equiv_l637_637845

open Function

universe u

-- Define a general set A
variable (A : Type u)

-- Define congruence of angles as an equivalence relation
def congruent_angles (α β : ℝ) : Prop :=
  α = β ∨ α = β + 180 ∨ α = β + 360

-- Define same remainder relation modulo n
def same_remainder_mod (n : ℕ) (a b : ℕ) : Prop :=
  a % n = b % n

theorem congruent_angles_equiv : Equivalence congruent_angles :=
by
  unfold Equivalence congruent_angles
  split
  -- prove reflexivity
  intros x
  left
  rfl
  -- prove symmetry
  intros x y h
  cases h
  left
  exact h
  case Or.inr h' =>
    cases h'
    right
    left
    exact h'
    right
    right
    exact h'
  -- prove transitivity
  intros x y z hxy hyz
  cases hxy
  cases hyz
  left
  exact Eq.trans hxy hyz
  case Or.inr hyz' =>
    right
    left
    exact Eq.trans hxy hyz'
  case Or.inr hxy' =>
    cases hxy'
    right
    left
    exact Eq.trans hxy' hyz
    right
    right
    exact Eq.trans hxy' hyz 

theorem same_remainder_mod_equiv (n : ℕ) : Equivalence (same_remainder_mod n) :=
by
  unfold Equivalence (same_remainder_mod n)
  split
  -- prove reflexivity
  intros x
  unfold same_remainder_mod
  rfl
  -- prove symmetry
  intros x y h
  unfold same_remainder_mod
  rw [eq_comm]
  exact h
  -- prove transitivity
  intros x y z hxy hyz
  unfold same_remainder_mod
  rw [←hxy, ←hyz]
  exact hxy
  

end congruent_angles_equiv_same_remainder_mod_equiv_l637_637845


namespace eval_abs_a_plus_b_l637_637416

theorem eval_abs_a_plus_b (a b : ℤ) (x : ℤ) 
(h : (7 * x - a) ^ 2 = 49 * x ^ 2 - b * x + 9) : |a + b| = 45 :=
sorry

end eval_abs_a_plus_b_l637_637416


namespace moskvich_cars_in_columns_l637_637283

theorem moskvich_cars_in_columns (x y m1 m2 : ℕ) 
  (h1 : x + y = 11) 
  (h2 : 28 - x = 2 * (28 - y)) :
  m1 = 21 ∧ m2 = 24 :=
by {
  have h3 : x = 7 ∧ y = 4, 
  { sorry }, 
  have h4 : m1 = 28 - x, 
  { sorry }, 
  have h5 : m2 = 28 - y, 
  { sorry },
  exact ⟨21, 24⟩,
  sorry
}

end moskvich_cars_in_columns_l637_637283


namespace julie_savings_fraction_l637_637649

variables (S : ℝ) (x : ℝ)
theorem julie_savings_fraction (h : 12 * S * x = 4 * S * (1 - x)) : 1 - x = 3 / 4 :=
sorry

end julie_savings_fraction_l637_637649


namespace find_sum_p_q_l637_637261

-- Given conditions
noncomputable def triangle_perimeter (B Q M : ℝ) : ℝ := B + Q + M
noncomputable def circle_radius : ℝ := 21
noncomputable def right_angle : ℝ := 90 -- QBM=PQM=90
noncomputable def perimeter_BQM : ℝ := 180

-- Define OQ
noncomputable def OQ := 315 / 8

-- Statement: Prove given the conditions, the sum of relatively prime integers p and q such that OQ = p/q
theorem find_sum_p_q :
  let p := 315 in
  let q := 8 in
  p.gcd q = 1 -> -- p and q are relatively prime
  OQ = p / q ->
  p + q = 323 :=
by
  intros
  sorry

end find_sum_p_q_l637_637261


namespace valid_combinations_count_l637_637404

def number_of_valid_combinations (a b : ℕ) : ℕ :=
  if a = 12 ∧ b = 34 ∨ a = 34 ∧ b = 12 then 2 else 0

theorem valid_combinations_count :
  number_of_valid_combinations 12 34 = 2 :=
by
  simp [number_of_valid_combinations]
  sorry

end valid_combinations_count_l637_637404


namespace compare_exponents_l637_637871

theorem compare_exponents : 
  let a := Real.pow 0.2 0.3
  let b := Real.pow 0.3 0.3
  let c := Real.pow 0.3 0.2
  in a < b ∧ b < c ∧ a < c :=
by 
  sorry

end compare_exponents_l637_637871


namespace shift_left_by_pi_div_3_l637_637960

def g (x : ℝ) : ℝ := Real.sin (2 * x)

def f (x : ℝ) : ℝ := (-1 / 2) * Real.sin (2 * x) + (Real.sqrt 3 / 2) * Real.cos (2 * x)

theorem shift_left_by_pi_div_3 :
  ∀ x : ℝ, f (x) = g (x + π / 3) :=
by
  sorry

end shift_left_by_pi_div_3_l637_637960


namespace ittymangnark_total_fish_l637_637516

-- Definitions corresponding to the splitting of fish
def itty_mang_nark_fish : ℝ := 2.5
def king_nook_fish : ℝ := 3.5
def oomy_apeck_fish : ℝ := 1.25
def yurr_aknalik_fish : ℝ := 2.75
def ankaq_fish : ℝ := 1.75
def nanuq_fish : ℝ := 4.75

-- Definition of the total fish needed
def total_fish_needed : ℝ :=
  itty_mang_nark_fish + king_nook_fish + oomy_apeck_fish + yurr_aknalik_fish + ankaq_fish + nanuq_fish

theorem ittymangnark_total_fish :
  total_fish_needed = 16.5 := 
by
  -- Use the definition of total_fish_needed to simplify
  calc total_fish_needed
      = 2.5 + 3.5 + 1.25 + 2.75 + 1.75 + 4.75 : rfl
  ... = 16.5 : by norm_num
  
-- Adding sorry to skip the rest of the proof
sorry

end ittymangnark_total_fish_l637_637516


namespace alpha_beta_sum_l637_637930

theorem alpha_beta_sum (α β : ℝ) (h1 : α^3 - 3 * α^2 + 5 * α - 17 = 0) (h2 : β^3 - 3 * β^2 + 5 * β + 11 = 0) : α + β = 2 := 
by
  sorry

end alpha_beta_sum_l637_637930


namespace construct_right_triangle_l637_637381

-- Definitions of the problem
variable (a q p : ℝ)

-- Condition for right triangle projection
noncomputable def right_triangle_projection : Prop :=
  a * a = p * (p + q)

-- Proposition to construct the right triangle
theorem construct_right_triangle (h : right_triangle_projection a q p) :
  ∃ (b c : ℝ), right_triangle a q b c :=
sorry

-- Predicate for right triangle construction
def right_triangle (a q b c : ℝ) : Prop :=
  a² + b² = c² ∧ (∃ p, a * a = p * (p + q))

end construct_right_triangle_l637_637381


namespace trisection_bisect_l637_637534

variable (A B C D E F G H M N : Type)
variable [add_comm_monoid A] [add_comm_monoid B] [add_comm_monoid C] [add_comm_monoid D] [add_comm_monoid E] [add_comm_monoid F] [add_comm_monoid G] [add_comm_monoid H] [add_comm_monoid M] [add_comm_monoid N]

-- Assume the conditions of the problem
variable (quadrilateral_abcd : A × B × C × D)
variable (E F G H : Type)
variable (E_is_trisection_AB : E)
variable (F_is_trisection_AB : F)
variable (G_is_trisection_CD : G)
variable (H_is_trisection_CD : H)
variable (M_midpoint_AD : M)
variable (N_midpoint_BC : N)

-- State the problem
theorem trisection_bisect
  (EG_is_bisected_by_MN : E × G → (M × N))
  (FH_is_bisected_by_MN : F × H → (M × N))
  (MN_is_trisected_by_EG_FH : (M × N) → (E × G) → (F × H)) :
  true :=
sorry

end trisection_bisect_l637_637534


namespace find_px_value_l637_637877

noncomputable def p (a b c x : ℤ) := a * x^2 + b * x + c

theorem find_px_value {a b c : ℤ} 
  (h1 : p a b c 2 = 2) 
  (h2 : p a b c (-2) = -2) 
  (h3 : p a b c 9 = 3) 
  (h : a = -2 / 11) 
  (h4 : b = 1)
  (h5 : c = 8 / 11) :
  p a b c 14 = -230 / 11 :=
by
  sorry

end find_px_value_l637_637877


namespace total_population_of_evans_county_l637_637503

theorem total_population_of_evans_county :
  (let avg_population := (3200 + 3600) / 2 in
   let total_population := 25 * avg_population in
   total_population = 85000) :=
by
  let avg_population := (3200 + 3600) / 2
  let total_population := 25 * avg_population
  show total_population = 85000
  apply sorry

end total_population_of_evans_county_l637_637503


namespace mod_arith_example_l637_637406

theorem mod_arith_example :
  (3 * 7⁻¹ + 5 * 13⁻¹) % 63 = 13 % 63 :=
by
  -- Given inverses
  have h7_inv : 7⁻¹ % 63 = 19 % 63 := sorry,
  have h13_inv : 13⁻¹ % 63 = 29 % 63 := sorry,
  -- The main goal is
  calc
    (3 * 7⁻¹ + 5 * 13⁻¹) % 63
        = (3 * 19 + 5 * 29) % 63 : by
          rw [h7_inv, h13_inv]
        ... = 202 % 63 : by
          norm_num
        ... = 13 % 63 : by
          norm_num

end mod_arith_example_l637_637406


namespace increasing_condition_l637_637083

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 3

-- Define the derivative of f with respect to x
def f' (x a : ℝ) : ℝ := 2 * x - 2 * a

-- Prove that f is increasing on the interval [2, +∞) if and only if a ≤ 2
theorem increasing_condition (a : ℝ) : (∀ x ≥ 2, f' x a ≥ 0) ↔ (a ≤ 2) := 
sorry

end increasing_condition_l637_637083


namespace arrange_abc_l637_637872

def a : ℝ := 0.5 ^ (1 / 2)
def b : ℝ := 0.9 ^ (-1 / 4)
def c : ℝ := Real.log 0.3 / Real.log 5

theorem arrange_abc : c < a ∧ a < b := by
  -- these are just the definitions, the actual proof would go here
  sorry

end arrange_abc_l637_637872


namespace sequence_a_general_sum_T_n_l637_637117

-- Definitions
def S (n : ℕ) := - (n : ℤ)^2 + 4 * (n : ℤ)
def a (n : ℕ) : ℤ := if n = 1 then 3 else S n - S (n - 1)
def b (n : ℕ) : ℤ := 2^n
def T (n : ℕ) : ℤ := (nat.rec 0 (λ n acc, acc + (a n) * (b n)) n)

-- Problems to be proved
theorem sequence_a_general (n : ℕ) : a n = -2 * n + 5 :=
by sorry

theorem sum_T_n (n : ℕ) : T n = (-2 * n + 7) * 2^(n+1) - 14 :=
by sorry

end sequence_a_general_sum_T_n_l637_637117


namespace power_eq_one_l637_637816

theorem power_eq_one (x y : ℤ) (h : sqrt (x + 1) + abs (y - 2) = 0) : x ^ y = 1 :=
by
  sorry

end power_eq_one_l637_637816


namespace find_m_l637_637592

def f (x m : ℝ) := x * (x - m) ^ 2

theorem find_m (m : ℝ) (h : ∃ ε > 0, ∀ x, abs(x - 1) < ε → f x m ≥ f 1 m) : m = 1 :=
by
  sorry

end find_m_l637_637592


namespace edges_after_cut_l637_637206

-- Define the initial number of edges in a cube
def initial_edges : ℕ := 12

-- Define the number of new edges added by cutting off a corner
def added_edges : ℕ := 3

-- Define the total number of edges after the corner is cut off
def total_edges : ℕ := initial_edges + added_edges

-- Theorem to state that the total number of edges in the new solid is 15
theorem edges_after_cut (initial_edges added_edges : ℕ) : total_edges = 15 :=
by {
  have h1 : initial_edges = 12 := rfl,
  have h2 : added_edges = 3 := rfl,
  have h3 : total_edges = initial_edges + added_edges := rfl,
  rw [h1, h2, h3],
  exact rfl,
}

end edges_after_cut_l637_637206


namespace ways_to_place_letters_l637_637952

-- defining the conditions of the problem
def num_letters : Nat := 4
def num_mailboxes : Nat := 3

-- the theorem we need to prove
theorem ways_to_place_letters : 
  (num_mailboxes ^ num_letters) = 81 := 
by 
  sorry

end ways_to_place_letters_l637_637952


namespace max_n_positive_l637_637155

theorem max_n_positive (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : S 15 > 0)
  (h2 : S 16 < 0)
  (hs1 : S 15 = 15 * (a 8))
  (hs2 : S 16 = 8 * (a 8 + a 9)) :
  (∀ n, a n > 0 → n ≤ 8) :=
by {
    sorry
}

end max_n_positive_l637_637155


namespace angle_neg510_in_third_quadrant_l637_637660

def first_quadrant (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 90

def second_quadrant (θ : ℝ) : Prop :=
  90 ≤ θ ∧ θ < 180

def third_quadrant (θ : ℝ) : Prop :=
  180 ≤ θ ∧ θ < 270

def fourth_quadrant (θ : ℝ) : Prop :=
  270 ≤ θ ∧ θ < 360

def equivalent_angle (θ : ℝ) : ℝ :=
  (θ % 360 + 360) % 360

theorem angle_neg510_in_third_quadrant :
  third_quadrant (equivalent_angle (-510)) :=
by
  sorry

end angle_neg510_in_third_quadrant_l637_637660


namespace sum_of_digits_of_roots_in_interval_l637_637716

noncomputable def equation_roots_sum_digits : Nat :=
  let equation := λ x : Real => 4 * sin(2 * x) + 3 * cos(2 * x) - 2 * sin(x) - 4 * cos(x) + 1
  let interval_start := 10 ^ (factorial 2014) * Real.pi
  let interval_end := 10 ^ (factorial 2014 + 2015) * Real.pi
  let number_of_roots := 2 * 10 ^ (factorial 2014) * (10 ^ 2015 - 1)
  let roots_sum := 1 + 8 + 9 * 2014
  roots_sum = 18135

theorem sum_of_digits_of_roots_in_interval : equation_roots_sum_digits = 18135 := by
  sorry

end sum_of_digits_of_roots_in_interval_l637_637716


namespace f_9_over_2_l637_637180

noncomputable def f (x : ℝ) : ℝ := sorry -- The function f(x) is to be defined later according to conditions

theorem f_9_over_2 :
  (∀ x : ℝ, f (x + 1) = -f (-x + 1)) ∧ -- f(x+1) is odd
  (∀ x : ℝ, f (x + 2) = f (-x + 2)) ∧ -- f(x+2) is even
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = -2 * x^2 + 2) ∧ -- f(x) = ax^2 + b, where a = -2 and b = 2
  (f 0 + f 3 = 6) → -- Sum f(0) and f(3)
  f (9 / 2) = 5 / 2 := 
by {
  sorry -- The proof is omitted as per the instruction
}

end f_9_over_2_l637_637180


namespace length_of_curve_l637_637039

theorem length_of_curve : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → 
  let x := 2 * (Real.cos θ)^2,
      y := 3 * (Real.sin θ)^2 in
      ∃ u v w z : ℝ, 
      u = ⟦(0, 3)⟧ ∧ 
      v = ⟦(2, 0)⟧ ∧
      w = ⟦(2 - 0), (0 - 3)⟧ ∧ 
      z = ⟦sqrt((2 - 0)^2 + (0 - 3)^2)⟧ ∧
      z = √13) :=
sorry

end length_of_curve_l637_637039


namespace amount_after_two_years_l637_637648

-- Defining the conditions as per the problem
def initial_value : ℝ := 57600
def annual_increase_rate : ℝ := 1/8
def duration : ℕ := 2

-- Statement of the problem
theorem amount_after_two_years :
  let factor := (1 + annual_increase_rate)
  let final_amount := initial_value * factor^duration
  final_amount = 72900 :=
by
  sorry

end amount_after_two_years_l637_637648


namespace point_K_is_fixed_l637_637010

open EuclideanGeometry

-- Definition of the conditions in the problem
variables {Point : Type} [euclidean_space Point]
variables (A B : Point)
variable {C : Point → Prop} -- C is a fixed circle passing through A
variable {D : Point → Prop} -- D is a fixed line passing through B
variable {variable_circle : Point → Point → set Point} -- Variable circle passing through A and B
variable E : Point -- Intersection of variable circle and line D
variable K : Point -- Intersection of line CE and fixed circle C

-- Hypotheses based on the conditions:
axiom fixed_circle (p : Point) : C p → euclidean_geometry.circumference C p A
axiom fixed_line (p : Point) : D p → euclidean_geometry.line_through B p
axiom var_circle_passes_AB (p : Point) : variable_circle A B p → euclidean_geometry.circumference (variable_circle A B) p A ∧ euclidean_geometry.circumference (variable_circle A B) p B
axiom var_circle_intersects_fixed (p : Point) : variable_circle A B p → C p → ¬ euclidean_geometry.same_point p (line_through C A)
axiom var_circle_intersects_line (p : Point) : variable_circle A B p → D p → E = p
axiom line_CE_intersects_C (p : Point) : (C = intersection_point, D = intersection_point) K ∧ C K

-- Define the theorem to show that K is fixed
theorem point_K_is_fixed : K = K := sorry

end point_K_is_fixed_l637_637010


namespace John_profit_is_1500_l637_637520

-- Defining the conditions
def P_initial : ℕ := 8
def Puppies_given_away : ℕ := P_initial / 2
def Puppies_kept : ℕ := 1
def Price_per_puppy : ℕ := 600
def Payment_stud_owner : ℕ := 300

-- Define the number of puppies John's selling
def Puppies_selling := Puppies_given_away - Puppies_kept

-- Define the total revenue from selling the puppies
def Total_revenue := Puppies_selling * Price_per_puppy

-- Define John’s profit 
def John_profit := Total_revenue - Payment_stud_owner

-- The statement to prove
theorem John_profit_is_1500 : John_profit = 1500 := by
  sorry

end John_profit_is_1500_l637_637520


namespace range_of_a_l637_637826

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1 - a * log x

theorem range_of_a (a : ℝ)
  (h₀ : a < 0)
  (h₁ : ∀ x1 x2 ∈ Icc 0 1, |f a x1 - f a x2| ≤ 4 * |1 / x1 - 1 / x2|) :
  a ∈ Icc (-3 : ℝ) 0 :=
sorry

end range_of_a_l637_637826


namespace find_a_l637_637543

theorem find_a
  (a b c : ℂ)
  (h1 : a ∈ ℝ)
  (h2 : a + b + c = 5)
  (h3 : a * b + b * c + c * a = 5)
  (h4 : a * b * c = 4) :
  a = 2 :=
by
  sorry

end find_a_l637_637543


namespace calculate_abc_value_l637_637777

theorem calculate_abc_value
  (m_angle_A : ℝ)
  (BC : ℝ)
  (BD_perp_AC : Prop)
  (CE_perp_AB : Prop)
  (m_angle_DBC_eq_2m_angle_ECB : Prop)
  (h1 : m_angle_A = 45)
  (h2 : BC = 16)
  (h3 : BD_perp_AC)
  (h4 : CE_perp_AB)
  (h5 : m_angle_DBC_eq_2m_angle_ECB) :
  ∃ a b c : ℝ, EC = a * (real.sqrt b + real.sqrt c) ∧ b ≠ 0 ∧ c ≠ 0 ∧ int.has_gcd b c 1 ∧ a + b + c = 9 :=
sorry

end calculate_abc_value_l637_637777


namespace number_of_paths_to_spell_BINGO_l637_637837

theorem number_of_paths_to_spell_BINGO : 
  ∃ (paths : ℕ), paths = 36 :=
by
  sorry

end number_of_paths_to_spell_BINGO_l637_637837


namespace chord_length_and_circle_area_l637_637327

noncomputable def radius : ℝ := 5
noncomputable def distance_to_chord : ℝ := 4

theorem chord_length_and_circle_area :
  let PQ_length := 2 * sqrt (radius^2 - distance_to_chord^2)
  ∧ PQ_length = 6 ∧ π * radius^2 = 25 * π :=
by
  let PQ_length := 2 * sqrt (radius^2 - distance_to_chord^2)
  show PQ_length = 6 ∧ π * radius^2 = 25 * π, from sorry

end chord_length_and_circle_area_l637_637327


namespace spinner_prime_probability_l637_637300

def numbers : List Nat := [2, 7, 9, 11, 15, 17]

def is_prime (n : Nat) : Prop := Nat.Prime n

def prime_numbers : List Nat := numbers.filter is_prime

def probability_of_prime : ℚ := prime_numbers.length / numbers.length

theorem spinner_prime_probability : probability_of_prime = 2 / 3 := 
by sorry

end spinner_prime_probability_l637_637300


namespace find_n_l637_637410

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Given hypotheses:
-- 1. n is a positive integer.
-- 2. 5^n is a factor of 20!.
-- 3. 5^(n+1) is not a factor of 20!.
def problem_statement (n : ℕ) : Prop :=
  n > 0 ∧ 
  ∃ k : ℕ, factorial 20 = 5^n * k ∧ k % 5 ≠ 0

-- Prove that n = 5
theorem find_n : ∃ n : ℕ, problem_statement n ∧ n = 5 :=
by {
  sorry
}

end find_n_l637_637410


namespace min_y_value_l637_637869

noncomputable def min_value_y : ℝ :=
  18 - 2 * Real.sqrt 106

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20 * x + 36 * y) : 
  y >= 18 - 2 * Real.sqrt 106 :=
sorry

end min_y_value_l637_637869


namespace largest_root_range_l637_637711

noncomputable def s : ℝ := 
  let max_root := λ (a2 a1 a0 : ℝ), 
    if h : is_R_or_C.is_real ℝ then 
      let roots := polynomial.roots (polynomial.C a2 * polynomial.C a1 * polynomial.C a0) in
        if h : ¬roots.is_empty then roots.to_finset.max' h else 0
    else 0 in
  let largest_satisfying := λ a2 a1 a0,
    if h : |a2| ≤ 3 ∧ |a1| ≤ 3 ∧ |a0| ≤ 3 then max_root a2 a1 a0 else 0 in
  finset.range 4.filter_map (λ i, if (largest_satisfying (i%3) ((i+1)%3) ((i+2)%3) > 0) then some (largest_satisfying (i%3) ((i+1)%3) ((i+2)%3)) else none).max' sorry -- since our problem guarantees there is a largest

theorem largest_root_range : 
  3 < s ∧ s < 4 := sorry

end largest_root_range_l637_637711


namespace find_number_n_l637_637467

theorem find_number_n (n x : ℝ)
  (h1 : n * x = 25)
  (h2 : 10 - x ≈ 0.9992159937279498) : n ≈ 2.777 :=
sorry

end find_number_n_l637_637467


namespace ratio_perimeter_to_circumference_l637_637741

theorem ratio_perimeter_to_circumference 
  (α β R : ℝ) (hα: 0 < α ∧ α < π) (hβ: 0 < β ∧ β < π):
  let l := 2 * Real.pi * R in
  let P := 2 * (2 * R / (Real.sin α) + 2 * R / (Real.sin β)) in
  P / l = 4 * Real.sin ((α + β) / 2) * Real.cos ((α - β) / 2) / (Real.pi * Real.sin α * Real.sin β) :=
by
  sorry

end ratio_perimeter_to_circumference_l637_637741


namespace equilateral_congruent_triangle_sum_square_l637_637731

theorem equilateral_congruent_triangle_sum_square 
  (A B C D1 E1 D2 E2 : Point) 
  (s r : ℝ)
  (h_equilateral : Triangle.is_equilateral A B C)
  (h_side_length : A.dist B = s ∧ B.dist C = s ∧ C.dist A = s)
  (h_triangle_congruent_1 : Triangle.is_congruent (Triangle.mk A D1 E1) (Triangle.mk A B C))
  (h_triangle_congruent_2 : Triangle.is_congruent (Triangle.mk A D2 E2) (Triangle.mk A B C))
  (h_bd1 : B.dist D1 = r)
  (h_bd2 : B.dist D2 = r)
  (h_side_s : s = 10)
  (h_radius_r : r = 3):
  (C.dist E1)^2 + (C.dist E2)^2 = 350 - 50 * Real.sqrt 3 :=
sorry

end equilateral_congruent_triangle_sum_square_l637_637731


namespace cost_of_blue_pill_l637_637353

variable (cost_total : ℝ) (days : ℕ) (daily_cost : ℝ)
variable (blue_pill : ℝ) (red_pill : ℝ)

-- Conditions
def condition1 (days : ℕ) : Prop := days = 21
def condition2 (blue_pill red_pill : ℝ) : Prop := blue_pill = red_pill + 2
def condition3 (cost_total daily_cost : ℝ) (days : ℕ) : Prop := cost_total = daily_cost * days
def condition4 (daily_cost blue_pill red_pill : ℝ) : Prop := daily_cost = blue_pill + red_pill

-- Target to prove
theorem cost_of_blue_pill
  (h1 : condition1 days)
  (h2 : condition2 blue_pill red_pill)
  (h3 : condition3 cost_total daily_cost days)
  (h4 : condition4 daily_cost blue_pill red_pill)
  (h5 : cost_total = 945) :
  blue_pill = 23.5 :=
by sorry

end cost_of_blue_pill_l637_637353


namespace number_of_triangles_l637_637137

open Nat

-- Define the number of combinations
def comb : Nat → Nat → Nat
  | n, k => if k > n then 0 else n.choose k

-- The given conditions
def points_on_OA := 5
def points_on_OB := 6
def point_O := 1
def total_points := points_on_OA + points_on_OB + point_O -- should equal 12

-- Lean proof problem statement
theorem number_of_triangles : comb total_points 3 - comb points_on_OA 3 - comb points_on_OB 3 = 165 := by
  sorry

end number_of_triangles_l637_637137


namespace geometric_sequence_sum_l637_637132

theorem geometric_sequence_sum (S : ℕ → ℝ) 
  (S5 : S 5 = 10)
  (S10 : S 10 = 50) :
  S 15 = 210 := 
by
  sorry

end geometric_sequence_sum_l637_637132


namespace minimum_annual_repayment_l637_637646

-- Define initial conditions
def principal : ℝ := 500000
def annual_rate : ℝ := 0.04
def duration : ℕ := 10
def total_amount_due : ℝ := principal * (1 + annual_rate) ^ duration
def annuity_factor : ℝ := (1 - (1 + annual_rate) ^ duration) / (1 - (1 + annual_rate))

-- Define the minimum annual repayment function
noncomputable def annual_repayment := total_amount_due / annuity_factor

-- Theorem specifying the minimum amount to be repaid each year
theorem minimum_annual_repayment : annual_repayment = 61700 := 
by 
  -- Proof goes here
  sorry

end minimum_annual_repayment_l637_637646


namespace angle_bisector_inequality_l637_637712

-- Given definitions and assumptions
def triangle_side_lengths (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0

def semiperimeter (a b c p : ℝ) : Prop := 
  p = (a + b + c) / 2

def circumcircle_radius (R : ℝ) : Prop := 
  R > 0

def incircle_radius (r : ℝ) : Prop := 
  r > 0

def angle_bisectors (l_a l_b l_c : ℝ) : Prop := 
  l_a > 0 ∧ l_b > 0 ∧ l_c > 0
  
-- The main goal
theorem angle_bisector_inequality 
  (a b c p R r l_a l_b l_c : ℝ)
  (h1: triangle_side_lengths a b c)
  (h2: semiperimeter a b c p)
  (h3: circumcircle_radius R)
  (h4: incircle_radius r)
  (h5: angle_bisectors l_a l_b l_c)
  :
  l_a * l_b + l_b * l_c + l_c * l_a ≤ p * real.sqrt (3 * r^2 + 12 * R * r) :=
  sorry

end angle_bisector_inequality_l637_637712


namespace expected_attempts_for_10_l637_637315

def harmonic (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, 1 / (k + 1))

noncomputable def expected_attempts (n : ℕ) : ℚ :=
  (n * (n + 3) / 4) - harmonic n

theorem expected_attempts_for_10 : expected_attempts 10 ≈ 29.62 := by
  sorry

end expected_attempts_for_10_l637_637315


namespace problem_relationship_l637_637752

theorem problem_relationship (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a :=
by {
  sorry
}

end problem_relationship_l637_637752


namespace increase_in_speed_l637_637168

-- Given conditions
def t0 : ℝ := 8 -- Initial running time in hours
def f : ℝ := 0.75 -- Time increase factor
def s0 : ℝ := 8 -- Initial speed in mph
def d : ℝ := 168 -- Total distance in miles

-- New running time
def t_new := t0 * (1 + f)

-- New speed
def s_new := d / t_new

-- Increase in speed
def delta_s := s_new - s0

-- Prove that the increase in speed is 4 mph
theorem increase_in_speed : delta_s = 4 := by
  sorry

end increase_in_speed_l637_637168


namespace single_cow_single_bag_l637_637493

-- Definitions given in the problem conditions
def cows : ℕ := 26
def bags : ℕ := 26
def days : ℕ := 26

-- Statement to be proved
theorem single_cow_single_bag : (1 : ℕ) = 26 := sorry

end single_cow_single_bag_l637_637493


namespace true_props_l637_637413

variable {a b c : ℝ}

-- Definitions for propositions
def prop1 : Prop := (a = b) ↔ (a * c = b * c)
def prop2 : Prop := (¬ ∃ q : ℚ, a + 5 = q) ↔ (¬ ∃ q : ℚ, a = q)
def prop3 : Prop := (a > b) → (a^2 > b^2)
def prop4 : Prop := (a < 5) ↔ (a < 3)

-- To prove that propositions ② and ④ are true
theorem true_props : (prop2 ∧ prop4) := by
  sorry

end true_props_l637_637413


namespace apex_angle_of_fourth_cone_l637_637284

theorem apex_angle_of_fourth_cone (A : Point)
  (cone1 cone2 cone3 cone4 : Cone)
  (h1 : cone1.apex = A)
  (h2 : cone2.apex = A)
  (h3 : cone3.apex = A)
  (h4 : cone4.apex = A)
  (h5 : cone1.apex_angle = π / 3)
  (h6 : cone2.apex_angle = π / 3)
  (h7 : cone3.apex_angle = π / 3)
  (h8 : cones_touch_externally cone1 cone2)
  (h9 : cones_touch_externally cone2 cone3)
  (h10 : cones_touch_externally cone1 cone3)
  (h11 : cone_touches_internally cone1 cone4)
  (h12 : cone_touches_internally cone2 cone4)
  (h13 : cone_touches_internally cone3 cone4)
  : cone4.apex_angle = π / 3 + 2 * arcsin (1 / sqrt 3) := 
sorry

end apex_angle_of_fourth_cone_l637_637284


namespace intersection_of_M_and_P_l637_637796

def M : Set ℝ := { x | x^2 = x }
def P : Set ℝ := { x | |x - 1| = 1 }

theorem intersection_of_M_and_P : M ∩ P = {0} := by
  sorry

end intersection_of_M_and_P_l637_637796


namespace inscribed_circle_radius_l637_637982

theorem inscribed_circle_radius (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) :
  let s := (a + b + c) / 2 in
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  let r := K / s in
  r = 2 :=
by
  sorry

end inscribed_circle_radius_l637_637982


namespace smallest_bottom_right_value_l637_637394

theorem smallest_bottom_right_value :
  ∃ (grid : ℕ × ℕ × ℕ → ℕ), -- grid as a function from row/column pairs to natural numbers
    (∀ i j, 1 ≤ i ∧ i ≤ 3 → 1 ≤ j ∧ j ≤ 3 → grid (i, j) ≠ 0) ∧ -- all grid values are non-zero
    (grid (1, 1) ≠ grid (1, 2) ∧ grid (1, 1) ≠ grid (1, 3) ∧ grid (1, 2) ≠ grid (1, 3) ∧
     grid (2, 1) ≠ grid (2, 2) ∧ grid (2, 1) ≠ grid (2, 3) ∧ grid (2, 2) ≠ grid (2, 3) ∧
     grid (3, 1) ≠ grid (3, 2) ∧ grid (3, 1) ≠ grid (3, 3) ∧ grid (3, 2) ≠ grid (3, 3)) ∧ -- all grid values are distinct
    (grid (1, 1) + grid (1, 2) = grid (1, 3)) ∧ 
    (grid (2, 1) + grid (2, 2) = grid (2, 3)) ∧ 
    (grid (3, 1) + grid (3, 2) = grid (3, 3)) ∧ -- row sum conditions
    (grid (1, 1) + grid (2, 1) = grid (3, 1)) ∧ 
    (grid (1, 2) + grid (2, 2) = grid (3, 2)) ∧ 
    (grid (1, 3) + grid (2, 3) = grid (3, 3)) ∧ -- column sum conditions
    (grid (3, 3) = 12) :=
by
  sorry

end smallest_bottom_right_value_l637_637394


namespace number_on_board_is_91_l637_637617

-- Definitions based on conditions
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def ends_in_digit (d n : ℕ) : Prop := n % 10 = d

def divisible_by (m n : ℕ) : Prop := n % m = 0

def andrey_statements (n : ℕ) : Prop :=
  (ends_in_digit 6 n ∨ divisible_by 7 n) ∧ ¬(ends_in_digit 6 n ∧ divisible_by 7 n)

def borya_statements (n : ℕ) : Prop :=
  (n > 26 ∨ ends_in_digit 8 n) ∧ ¬(n > 26 ∧ ends_in_digit 8 n)

def sasha_statements (n : ℕ) : Prop :=
  (divisible_by 13 n ∨ n < 27) ∧ ¬(divisible_by 13 n ∧ n < 27)

-- Mathematical equivalent proof problem
theorem number_on_board_is_91 (n : ℕ) :
  is_two_digit n →
  andrey_statements n →
  borya_statements n →
  sasha_statements n →
  n = 91 :=
by {
  intro _ _ _ _,
  -- Proof goes here, skipped with sorry
  sorry
}

end number_on_board_is_91_l637_637617


namespace max_value_of_quadratic_l637_637981

theorem max_value_of_quadratic :
  ∃ (x : ℝ), ∀ (y : ℝ), -3 * y^2 + 18 * y - 5 ≤ -3 * x^2 + 18 * x - 5 ∧ -3 * x^2 + 18 * x - 5 = 22 :=
sorry

end max_value_of_quadratic_l637_637981


namespace divisibility_by_30_l637_637819

theorem divisibility_by_30 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_3 : p ≥ 3) : 30 ∣ (p^3 - 1) ↔ p % 15 = 1 := 
  sorry

end divisibility_by_30_l637_637819


namespace black_white_area_ratio_l637_637409

-- Define the radii of the concentric circles
def radii := [2, 4, 6, 8, 10]

-- Define the function to calculate the area of a circle given its radius
def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the areas of the concentric circles
def areas := radii.map circle_area

-- Define the areas of the black and white regions
def black_areas := [areas.nthLe 0 sorry, areas.nthLe 2 sorry - areas.nthLe 1 sorry, areas.nthLe 4 sorry - areas.nthLe 3 sorry]
def white_areas := [areas.nthLe 1 sorry - areas.nthLe 0 sorry, areas.nthLe 3 sorry - areas.nthLe 2 sorry]

-- Calculate the total black and white areas
def total_black_area := black_areas.sum
def total_white_area := white_areas.sum

-- State the theorem to prove
theorem black_white_area_ratio : total_black_area / total_white_area = 3 / 2 :=
by
  sorry

end black_white_area_ratio_l637_637409


namespace increasing_function_condition_max_min_values_inequality_ln_l637_637094

-- Proposition 1
theorem increasing_function_condition (a : ℝ) (h : a ≥ 1) (x : ℝ) (hx : x ≥ 1) : 
    (1 - x) / (a * x) + log x ≤ (1 - x) / (a * (x+1)) + log (x + 1) :=
sorry

-- Proposition 2
theorem max_min_values (a : ℝ) (ha : a = 1) :
    ∀ x : ℝ, x ∈ set.Icc (1/2) (2) 
    → ((1 - x) / x + log x ≤ 1 - log 2 ∧ (1 - x) / x + log x ≥ 0)
:=
sorry

-- Proposition 3
theorem inequality_ln (a : ℝ) (ha : a = 1) (n : ℕ) (hn : n > 1) :
    log ((n:ℝ) / (n-1)) > 1 / (n:ℝ) :=
sorry

end increasing_function_condition_max_min_values_inequality_ln_l637_637094


namespace intersection_M_N_l637_637471

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 - 3 * x = 0}

theorem intersection_M_N : M ∩ N = {0} :=
by sorry

end intersection_M_N_l637_637471


namespace inequality_proof_l637_637550

open Real

-- Given conditions
variables (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1)

-- Goal to prove
theorem inequality_proof : 
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + (y / x)^(1/3) + (z / y)^(1/3) + (x / z)^(1/3)) :=
sorry

end inequality_proof_l637_637550


namespace power_equivalence_l637_637417

theorem power_equivalence (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (x y : ℕ) 
  (hx : 2^m = x) (hy : 2^(2 * n) = y) : 4^(m + 2 * n) = x^2 * y^2 := 
by 
  sorry

end power_equivalence_l637_637417


namespace calculate_expression_l637_637701

theorem calculate_expression (a b : ℝ) : (a - b) * (a + b) * (a^2 - b^2) = a^4 - 2 * a^2 * b^2 + b^4 := 
by
  sorry

end calculate_expression_l637_637701


namespace meet_time_first_l637_637932

noncomputable def time_to_meet (circumference : ℝ) (speed_lata : ℝ) (speed_geeta : ℝ) : ℝ :=
  circumference / (speed_lata + speed_geeta)

theorem meet_time_first (circumference : ℝ) (speed_lata_km_hr : ℝ) (speed_geeta_km_hr : ℝ) :
  let speed_lata := (speed_lata_km_hr * 1000 / 60)
  let speed_geeta := (speed_geeta_km_hr * 1000 / 60)
  circumference = 640 ∧ speed_lata_km_hr = 4.2 ∧ speed_geeta_km_hr = 3.8 →
  time_to_meet circumference speed_lata speed_geeta ≈ 4.812 :=
by
  intro h
  cases h with circumference_eq speeds_eq
  cases speeds_eq with speed_lata_eq speed_geeta_eq
  rw [circumference_eq, speed_lata_eq, speed_geeta_eq]
  sorry -- Proof goes here

end meet_time_first_l637_637932


namespace excess_weight_proof_l637_637686

def kelly_weight : ℝ := 30
def daisy_weight : ℝ := 24
def sam_weight : ℝ := 3 * daisy_weight
def mike_weight : ℝ := 1.5 * kelly_weight
def megan_weight : ℝ := (kelly_weight + daisy_weight + sam_weight + mike_weight) / 4
def total_weight : ℝ := kelly_weight + daisy_weight + sam_weight + mike_weight + megan_weight
def bridge_limit : ℝ := 140
def excess_weight : ℝ := total_weight - bridge_limit

theorem excess_weight_proof : excess_weight = 73.75 := by
  sorry

end excess_weight_proof_l637_637686
