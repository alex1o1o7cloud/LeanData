import Complex
import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Invertible
import Mathlib.Algebra.QuadraticEq
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Composition
import Mathlib.Combinatorics.Perm
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Comb
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Parity
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.InnerProduct
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Covering

namespace is_false_none_of_these_l506_506244

-- Define the conditions
variables (a b : ℝ) (A B : Point) (AB : ℝ)
variable (CircleA_radius : a > b)

-- Define Points as arbitrary types to make the statement valid in Lean
structure Point

-- Define distances
variable (distance_AB : distance A B = AB)

-- Definitions of the statements
def statement_A := a - b < AB
def statement_B := a + b = AB
def statement_C := a + b < AB
def statement_D := a - b = AB

-- Theorem stating the proof problem
theorem is_false_none_of_these : 
  statement_A a b AB ∨ statement_B a b AB ∨ statement_C a b AB ∨ statement_D a b AB → (¬ (false = "none of these")) := by sorry

end is_false_none_of_these_l506_506244


namespace find_x_tan_eq_l506_506430

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l506_506430


namespace simplify_expression_l506_506657

theorem simplify_expression :
  (√450 / √250 + √294 / √147) = (3 * √10 + 5 * √2) / 5 := by
  sorry

end simplify_expression_l506_506657


namespace concurrency_of_lines_l506_506993

variable (A B C D E F K O M : Type)
variable [ordered_ring ℝ] [metric_space A] [metric_space B] [metric_space C]
variable [metric_space D] [metric_space E] [metric_space F] [metric_space K]
variable [metric_space O] [metric_space M]

-- Conditions
variable (h_acute_triangle : acute_triangle ABC)
variable (h_circumcircle : is_circumcircle ω ABC)
variable (h_circumcenter : is_circumcenter O ABC)
variable (h_perpendicular_A_BC : is_perpendicular_from_A_to_BC A BC D E ω)
variable (h_point_F : point_on_segment_AE_with_distance AE F FD (2 * FD))
variable (h_line_l : line_perpendicular_to_OF_through_F F OF l)

-- Question: Prove concurrency
theorem concurrency_of_lines :
  concurrent_lines l (tangent_to_omega_at_E ω E) BC :=
sorry

end concurrency_of_lines_l506_506993


namespace difference_of_sum_l506_506766

theorem difference_of_sum (a b c : ℤ) (h1 : a = 11) (h2 : b = 13) (h3 : c = 15) :
  (b + c) - a = 17 := by
  sorry

end difference_of_sum_l506_506766


namespace smallest_percent_increase_is_100_l506_506575

-- The values for each question
def prize_values : List ℕ := [150, 300, 450, 900, 1800, 3600, 7200, 14400, 28800, 57600, 115200, 230400, 460800, 921600, 1843200]

-- Definition of percent increase calculation
def percent_increase (old new : ℕ) : ℕ :=
  ((new - old : ℕ) * 100) / old

-- Lean theorem statement
theorem smallest_percent_increase_is_100 :
  percent_increase (prize_values.get! 5) (prize_values.get! 6) = 100 ∧
  percent_increase (prize_values.get! 7) (prize_values.get! 8) = 100 ∧
  percent_increase (prize_values.get! 9) (prize_values.get! 10) = 100 ∧
  percent_increase (prize_values.get! 10) (prize_values.get! 11) = 100 ∧
  percent_increase (prize_values.get! 13) (prize_values.get! 14) = 100 :=
by
  sorry

end smallest_percent_increase_is_100_l506_506575


namespace sum_of_possible_d_values_l506_506769

theorem sum_of_possible_d_values : 
  let lower_bound := 2^13
  let upper_bound := 2^17
  (∀ n, (10000 ≤ n ∧ n ≤ 99999) → (log2 n).to_nat + 1 ≥ 14 ∧ (log2 n).to_nat + 1 ≤ 17) → 
  ∑ k in finset.range 18, if 14 ≤ k ∧ k ≤ 17 then k else 0 = 62 :=
by
  sorry

end sum_of_possible_d_values_l506_506769


namespace identical_first_last_four_l506_506273

/-- Definition of the sequence and its properties -/
variables {a : list ℕ}

/-- Property 1: Any two distinct consecutive 5-digit subsequences are not identical -/
def property1 (a : list ℕ) : Prop :=
∀ i j, i ≠ j → (a.get (i + 0) ++ a.get (i + 1) ++ a.get (i + 2) ++ a.get (i + 3) ++ a.get (i + 4)) ≠
           (a.get (j + 0) ++ a.get (j + 1) ++ a.get (j + 2) ++ a.get (j + 3) ++ a.get (j + 4))

/-- Property 2: Adding a 0 or 1 to the end of the sequence violates Property 1 -/
def property2 (a : list ℕ) : Prop :=
∃ (b : list ℕ), (b = 0 :: a ∨ b = 1 :: a) ∧ ¬(property1 b)

theorem identical_first_last_four
  (a : list ℕ)
  (h1 : property1 a)
  (h2 : property2 a) :
  (a.take 4 = a.drop (a.length - 4)).last 4 := sorry

end identical_first_last_four_l506_506273


namespace tan_sin_cos_eq_l506_506401

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l506_506401


namespace seq_limit_length_l506_506293

open Real

def sequence_term (n : ℕ) : ℝ :=
  (if n = 0 then 1 else (1 / 4 ^ n) * (sqrt 2 + 1))

def sequence (n : ℕ) : ℝ :=
  ∑ i in finset.range n, sequence_term i

theorem seq_limit_length : (tendsto (λ n, sequence n) at_top (𝓝 ((4 + sqrt 2) / 3))) :=
by {
  sorry
}

end seq_limit_length_l506_506293


namespace Marty_votes_l506_506147

theorem Marty_votes (total_count : ℕ) (biff_percentage undecided_percentage : ℝ)
  (total_count_eq : total_count = 200) 
  (biff_percentage_eq : biff_percentage = 0.45) 
  (undecided_percentage_eq : undecided_percentage = 0.08) : 
  let marty_percentage := 1 - (biff_percentage + undecided_percentage),
      marty_votes := marty_percentage * total_count in
  marty_votes = 94 :=
by
  unfold marty_percentage marty_votes
  rw [biff_percentage_eq, undecided_percentage_eq, total_count_eq]
  norm_num
  rw [←mul_assoc, mul_eq_mul_left_iff]
  norm_num
  sorry

end Marty_votes_l506_506147


namespace winning_entry_is_B_l506_506837

namespace ArtFestival

def student_A (winner : char) : Prop := winner = 'C' ∨ winner = 'D'
def student_B (winner : char) : Prop := winner = 'B'
def student_C (winner : char) : Prop := winner ≠ 'A' ∧ winner ≠ 'D'
def student_D (winner : char) : Prop := winner = 'C'

def predictions_correct (winner : char) : ℕ :=
  (if student_A winner then 1 else 0) +
  (if student_B winner then 1 else 0) +
  (if student_C winner then 1 else 0) +
  (if student_D winner then 1 else 0)

theorem winning_entry_is_B : ∃ winner, predictions_correct winner = 2 ∧ winner = 'B' :=
by
  use 'B'
  split
  sorry

end ArtFestival

end winning_entry_is_B_l506_506837


namespace three_a_in_S_implies_a_in_S_l506_506608

def S := {n | ∃ x y : ℤ, n = x^2 + 2 * y^2}

theorem three_a_in_S_implies_a_in_S (a : ℤ) (h : 3 * a ∈ S) : a ∈ S := 
sorry

end three_a_in_S_implies_a_in_S_l506_506608


namespace sum_of_solutions_is_neg_two_over_three_l506_506857

noncomputable def sum_of_solutions : ℝ :=
  let eqn := λ x : ℝ, ( (-12 * x) / (x^2 - 1) = (3 * x) / (x + 1) - 7 / (x - 1) ) in
  let x1 := (-1 + Real.sqrt 22) / 3 in
  let x2 := (-1 - Real.sqrt 22) / 3 in
  if eqn x1 ∧ eqn x2 then x1 + x2 else 0

theorem sum_of_solutions_is_neg_two_over_three : sum_of_solutions = -2 / 3 := by
  sorry

end sum_of_solutions_is_neg_two_over_three_l506_506857


namespace marys_balloons_l506_506143

theorem marys_balloons (x y : ℝ) (h1 : x = 4 * y) (h2 : x = 7.0) : y = 1.75 := by
  sorry

end marys_balloons_l506_506143


namespace rhombus_unique_circles_l506_506615

variables {A B C D : Type*} [Rhombus R A B C D]

theorem rhombus_unique_circles :
  number_of_unique_circles R = 2 :=
sorry

end rhombus_unique_circles_l506_506615


namespace adults_on_field_trip_l506_506535

-- Define the conditions
def van_capacity : ℕ := 7
def num_students : ℕ := 33
def num_vans : ℕ := 6

-- Define the total number of people that can be transported given the number of vans and capacity per van
def total_people : ℕ := num_vans * van_capacity

-- The number of people that can be transported minus the number of students gives the number of adults
def num_adults : ℕ := total_people - num_students

-- Theorem to prove the number of adults is 9
theorem adults_on_field_trip : num_adults = 9 :=
by
  -- Skipping the proof
  sorry

end adults_on_field_trip_l506_506535


namespace power_function_properties_l506_506811

theorem power_function_properties :
  (∀ α x : ℝ, x > 0 → x ^ α > 0) ∧
  (¬ (∀ x : ℝ, y = x^0 → ((0, 0) ∈ y) ∧ ((1, 1) ∈ y))) ∧
  (∀ α : ℝ, α ≥ 0 → (∀ x1 x2 : ℝ, x1 < x2 → x1 ^ α ≤ x2 ^ α)) ∧
  (∀ α : ℝ, α < 0 → (∀ x1 x2 : ℝ, x1 < x2 → x1 ^ α > x2 ^ α)) →
  (decision = "B" := by sorry)

end power_function_properties_l506_506811


namespace maciek_total_cost_l506_506787

-- Define the cost of pretzels and the additional cost percentage for chips
def cost_pretzel : ℝ := 4
def cost_chip := cost_pretzel + (cost_pretzel * 0.75)

-- Number of packets Maciek bought for pretzels and chips
def num_pretzels : ℕ := 2
def num_chips : ℕ := 2

-- Total cost calculation
def total_cost := (cost_pretzel * num_pretzels) + (cost_chip * num_chips)

-- The final theorem statement
theorem maciek_total_cost :
  total_cost = 22 := by
  sorry

end maciek_total_cost_l506_506787


namespace roger_dimes_collected_l506_506653

theorem roger_dimes_collected 
    (pennies : ℕ) (nickels : ℕ) (coins_left : ℕ) (coins_donated : ℕ) 
    (initial_dimes : ℕ) (total_initial_coins : ℕ) 
    (pennies_eq : pennies = 42) 
    (nickels_eq : nickels = 36) 
    (coins_left_eq : coins_left = 27) 
    (coins_donated_eq : coins_donated = 66)
    (initial_dimes_eq : total_initial_coins = pennies + nickels + initial_dimes)
    (coins_donated_calculation : coins_donated = total_initial_coins - coins_left) 
    : initial_dimes = 15 :=
by 
  have H1 : total_initial_coins = 42 + 36 + initial_dimes := initial_dimes_eq
  have H2 : total_initial_coins = 78 + initial_dimes := by rw [add_assoc, pennies_eq, nickels_eq]
  have H3 : coins_donated = 78 + initial_dimes - 27 := by rw [H2, coins_left_eq, coins_donated_eq]
  have H4 : coins_donated = initial_dimes + 51 := by linarith
  have H5 : 66 = initial_dimes + 51 := H3 
  show initial_dimes = 15 from by linarith using [H5]

end roger_dimes_collected_l506_506653


namespace find_a_b_sum_l506_506133

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 6 * x - 6

theorem find_a_b_sum (a b : ℝ)
  (h1 : f a = 1)
  (h2 : f b = -5) :
  a + b = 2 :=
  sorry

end find_a_b_sum_l506_506133


namespace ellipse_problem_l506_506898

noncomputable def ellipse_foci (F1 F2 : Point) : Ellipse := sorry
noncomputable def ellipse_equation (e : Ellipse) : String := sorry
noncomputable def dot_product (v1 v2 : Vector) : Real := sorry

variables {F1 F2 O R : Point} {P : Point → Prop} {k : Real} (l2 : Line)

-- Conditions
def F1 := Point.mk (-2 * Real.sqrt 2, 0)
def F2 := Point.mk (2 * Real.sqrt 2, 0)
def O := Point.mk (0, 0)
def P (x y : Real) := |(Point.mk x y) - F1| + |(Point.mk x y) - F2| = 4 * Real.sqrt 3
def R := Point.mk (0, -2)
def l2 := Line.mk (Point.mk 0 1) k

-- Tuple translation
theorem ellipse_problem :
  (ellipse_foci F1 F2).equation = "x^2/12 + y^2/4 = 1" ∧
  ∀ k : Real, let M := l2.intersection (ellipse_foci F1 F2),
               let N := l2.intersection (ellipse_foci F1 F2),
               dot_product (R.vector_to M) (R.vector_to N) = 0
:= sorry

end ellipse_problem_l506_506898


namespace probability_at_most_one_correct_in_two_rounds_l506_506700

theorem probability_at_most_one_correct_in_two_rounds :
  let pA := 3 / 5
  let pB := 2 / 3
  let pA_incorrect := 1 - pA
  let pB_incorrect := 1 - pB
  let p_0_correct := pA_incorrect * pA_incorrect * pB_incorrect * pB_incorrect
  let p_1_correct_A1 := pA * pA_incorrect * pB_incorrect * pB_incorrect
  let p_1_correct_A2 := pA_incorrect * pA * pB_incorrect * pB_incorrect
  let p_1_correct_B1 := pA_incorrect * pA_incorrect * pB * pB_incorrect
  let p_1_correct_B2 := pA_incorrect * pA_incorrect * pB_incorrect * pB
  let p_at_most_one := p_0_correct + p_1_correct_A1 + p_1_correct_A2 + 
      p_1_correct_B1 + p_1_correct_B2
  p_at_most_one = 32 / 225 := 
  sorry

end probability_at_most_one_correct_in_two_rounds_l506_506700


namespace find_angle_ABC_l506_506095

/-- Given geometric conditions of a triangle and a line segment, calculate the missing angle in the triangle. -/
theorem find_angle_ABC (BCD_180 : ∠BCD = 180)
                      (ACD_75 : ∠ACD = 75) 
                      (BAC_35 : ∠BAC = 35) : 
                      ∠ABC = 40 :=
by
  sorry

end find_angle_ABC_l506_506095


namespace total_apples_l506_506287

-- Definitions based on conditions
def used_apples : ℕ := 15
def remaining_apples : ℕ := 4

-- Theorem statement to be proven
theorem total_apples (used_apples = 15) (remaining_apples = 4) : used_apples + remaining_apples = 19 := sorry

end total_apples_l506_506287


namespace length_of_train_l506_506805

/-- Definitions for the problem's conditions. --/
def train_speed_km_per_hr := 45  -- Speed of the train in km/hr
def time_to_cross_bridge_sec := 30  -- Time to cross the bridge in seconds
def length_of_bridge_m := 265  -- Length of the bridge in meters

/-- Convert the train speed from km/hr to m/s. --/
def train_speed_m_per_s : Real :=
  (train_speed_km_per_hr * 1000) / 3600  -- 45 km/hr = 12.5 m/s

/-- Calculate the total distance the train travels in 30 seconds. --/
def total_distance_m : Real :=
  train_speed_m_per_s * time_to_cross_bridge_sec  -- 12.5 m/s * 30 s = 375 m

/-- Determine the length of the train from the total distance and length of the bridge. --/
theorem length_of_train :
  total_distance_m - length_of_bridge_m = 110 :=
by
  sorry

end length_of_train_l506_506805


namespace find_larger_number_l506_506246

theorem find_larger_number (a b : ℕ) (h_diff : a - b = 3) (h_sum_squares : a^2 + b^2 = 117) (h_pos : 0 < a ∧ 0 < b) : a = 9 :=
by
  sorry

end find_larger_number_l506_506246


namespace exists_eight_numbers_from_first_100_l506_506596

theorem exists_eight_numbers_from_first_100 (n : ℕ) (s : finset ℕ) :
  s.card = 8 ∧ ∀ x ∈ s, x ∈ finset.range 101 →
      ∃ m, n = ∑ i in s, i ∧ (∀ x ∈ s, n % x = 0) :=
begin
  sorry
end

end exists_eight_numbers_from_first_100_l506_506596


namespace max_range_of_walk_min_range_of_walk_count_of_max_range_sequences_l506_506755

section RandomWalk

variables {a b : ℕ} (h : a > b)

def max_range_walk : ℕ := a
def min_range_walk : ℕ := a - b
def count_max_range_sequences : ℕ := b + 1

theorem max_range_of_walk (h : a > b) : max_range_walk h = a :=
by
  sorry

theorem min_range_of_walk (h : a > b) : min_range_walk h = a - b :=
by
  sorry

theorem count_of_max_range_sequences (h : a > b) : count_max_range_sequences h = b + 1 :=
by
  sorry

end RandomWalk

end max_range_of_walk_min_range_of_walk_count_of_max_range_sequences_l506_506755


namespace car_speeds_l506_506282

-- Definitions and conditions
def distance_AB : ℝ := 200
def distance_meet : ℝ := 80
def car_A_speed : ℝ := sorry -- To Be Proved
def car_B_speed : ℝ := sorry -- To Be Proved

axiom car_B_faster (x : ℝ) : car_B_speed = car_A_speed + 30
axiom time_equal (x : ℝ) : (distance_meet / car_A_speed) = ((distance_AB - distance_meet) / car_B_speed)

-- Proof (only statement, without steps)
theorem car_speeds : car_A_speed = 60 ∧ car_B_speed = 90 :=
  by
  have car_A_speed := 60
  have car_B_speed := 90
  sorry

end car_speeds_l506_506282


namespace number_of_people_voting_for_Marty_l506_506149

noncomputable def percentage_voting_for_Biff : ℝ := 0.45
noncomputable def percentage_undecided : ℝ := 0.08
noncomputable def total_people_polled : ℝ := 200

theorem number_of_people_voting_for_Marty 
  (percentage_voting_for_Biff : ℝ) 
  (percentage_undecided : ℝ) 
  (total_people_polled : ℝ)
  (h_percentages : percentage_voting_for_Biff = 0.45 ∧ percentage_undecided = 0.08 ∧ total_people_polled = 200) :
  (total_people_polled * (1 - percentage_voting_for_Biff - percentage_undecided)) = 94 :=
by
  cases h_percentages with percentBiff rest
  cases rest with undecided total
  rw [percentBiff, undecided, total]
  -- The following line completes the proof using the remaining steps
  -- which are not required in the problem statement but provided for context.
  -- linarith [mul_comm, one_sub, sub_eq_add_neg, mul_assoc]
  exact sorry

end number_of_people_voting_for_Marty_l506_506149


namespace radius_range_l506_506465

noncomputable def circle_eq (x y r : ℝ) := x^2 + y^2 = r^2

def point_P_on_line_AB (m n : ℝ) := 4 * m + 3 * n - 24 = 0

def point_P_in_interval (m : ℝ) := 0 ≤ m ∧ m ≤ 6

theorem radius_range {r : ℝ} :
  (∀ (m n x y : ℝ), point_P_in_interval m →
     circle_eq x y r →
     circle_eq ((x + m) / 2) ((y + n) / 2) r → 
     point_P_on_line_AB m n ∧
     (4 * r ^ 2 ≤ (25 / 9) * m ^ 2 - (64 / 3) * m + 64 ∧
     (25 / 9) * m ^ 2 - (64 / 3) * m + 64 ≤ 36 * r ^ 2)) →
  (8 / 3 ≤ r ∧ r < 12 / 5) :=
sorry

end radius_range_l506_506465


namespace notebook_cost_l506_506559

theorem notebook_cost {s n c : ℕ}
  (h1 : s > 18)
  (h2 : c > n)
  (h3 : s * n * c = 2275) :
  c = 13 :=
sorry

end notebook_cost_l506_506559


namespace annual_rent_per_square_foot_l506_506272

theorem annual_rent_per_square_foot
  (length width : ℕ) (monthly_rent : ℕ) (h_length : length = 10)
  (h_width : width = 8) (h_monthly_rent : monthly_rent = 2400) :
  (monthly_rent * 12) / (length * width) = 360 := 
by 
  -- We assume the theorem is true.
  sorry

end annual_rent_per_square_foot_l506_506272


namespace combined_salaries_l506_506687

variable (S_A S_B S_C S_D S_E : ℝ)

theorem combined_salaries 
    (h1 : S_C = 16000)
    (h2 : (S_A + S_B + S_C + S_D + S_E) / 5 = 9000) : 
    S_A + S_B + S_D + S_E = 29000 :=
by 
    sorry

end combined_salaries_l506_506687


namespace charts_per_associate_professor_l506_506311

-- Definitions
def A : ℕ := 3
def B : ℕ := 4
def C : ℕ := 1

-- Conditions based on the given problem
axiom h1 : 2 * A + B = 10
axiom h2 : A * C + 2 * B = 11
axiom h3 : A + B = 7

-- The theorem to be proven
theorem charts_per_associate_professor : C = 1 := by
  sorry

end charts_per_associate_professor_l506_506311


namespace max_complex_expr_l506_506624

open Complex

theorem max_complex_expr (z : ℂ) (h : abs z = 2) : 
  abs ((z - 2)^3 * (z + 2)) ≤ 24 * real.sqrt 3 := sorry

end max_complex_expr_l506_506624


namespace find_current_l506_506961

noncomputable def V : ℂ := 2 + 3 * Complex.I
noncomputable def Z : ℂ := 2 - 2 * Complex.I

theorem find_current : (V / Z) = (-1 / 4 : ℂ) + (5 / 4 : ℂ) * Complex.I := by
  sorry

end find_current_l506_506961


namespace square_elements_l506_506870

variables {G : Type*} [group G] (g h : G)

-- Define the conditions as given in the problem
def group_conditions : Prop :=
  (g ^ 4 = 1) ∧
  (g ^ 2 ≠ 1) ∧
  (h ^ 7 = 1) ∧
  (h ≠ 1) ∧
  (g * h * g⁻¹ * h = 1)

-- Prove the elements in G which are squares
theorem square_elements {H : subgroup G} 
  (hg : H = group.closure {g, h})
  (cond : group_conditions g h) :
  { x | ∃ y, y ^ 2 = x } = { 1, g^2, h, h^2, h^3, h^4, h^5, h^6 } :=
sorry

end square_elements_l506_506870


namespace angle_of_tangent_inclination_at_given_point_l506_506848
noncomputable theory

-- Define the curve equation y = 1/2 * x^2 - 2
def curve (x : ℝ) : ℝ := (1/2) * x^2 - 2

-- Define the given point (1, -3/2)
def given_point : ℝ × ℝ := (1, -3/2)

-- Theorem stating the angle of inclination of the tangent line at the given point is π/4
theorem angle_of_tangent_inclination_at_given_point : 
  let dcurve_dx := λ x : ℝ, deriv curve x in
  let slope_at_point := dcurve_dx 1 in
  let angle_inclination := Real.arctan slope_at_point in
  angle_inclination = π/4 :=
by
  sorry

end angle_of_tangent_inclination_at_given_point_l506_506848


namespace part1_part2_l506_506047

noncomputable def f (a x : ℝ) : ℝ := Real.log x + 1 + (2 * a) / x

-- (I) Prove that a = 1 satisfies the given conditions and find monotonic intervals of f(x)
theorem part1 (f : ℝ → ℝ) (a : ℝ) (h1 : ∀ x, f x = (Real.log x + 1 + (2 * a) / x))
  (h2 : ∀ (x : ℝ), tangent f a (a, f a) passes_through (0, 4)) :
  a = 1 ∧ (∀ x:ℝ, x ∈ (0, 2) → f' x < 0) ∧ (∀ x:ℝ, x ∈ (2, +∞) → f' x > 0) :=
sorry

-- (II) Prove that the maximum k is 7
theorem part2 (f : ℝ → ℝ) (k : ℝ) (h1 : ∀ x, f x = (Real.log x + 1 + (2 * a) / x))
  (h2 : ∀ x, x ∈ (1, +∞) → 2 * (f (a x)) > k * (1 - 1/x)) :
  k ≤ 7 :=
sorry

end part1_part2_l506_506047


namespace alpha_values_perpendicular_l506_506872

theorem alpha_values_perpendicular
  (α : ℝ)
  (h1 : α ∈ Set.Ico 0 (2 * Real.pi))
  (h2 : ∀ (x y : ℝ), x * Real.cos α - y - 1 = 0 → x + y * Real.sin α + 1 = 0 → false):
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 :=
by
  sorry

end alpha_values_perpendicular_l506_506872


namespace math_problem_proof_l506_506580

-- Define the parametric equations of C1 and the Cartesian equation
def parametric_C1 (t : ℝ) : ℝ × ℝ := ( (2 + t) / 6, sqrt t )
def cartesian_C1 (x y : ℝ) : Prop := y^2 = 6 * x - 2 ∧ y ≥ 0

-- Define the parametric equations of C2
def parametric_C2 (s : ℝ) : ℝ × ℝ := ( -(2 + s) / 6, -sqrt s )

-- Define the polar equation of C3 and its Cartesian equivalent
def polar_C3 (θ : ℝ) : Prop := 2 * cos θ - sin θ = 0
def cartesian_C3 (x y : ℝ) : Prop := 2 * x - y = 0

-- Define the intersection points of C3 with C1
def intersection_C3_C1 : set (ℝ × ℝ) := [(1/2, 1), (1, 2)].to_set

-- Define the intersection points of C3 with C2
def intersection_C3_C2 : set (ℝ × ℝ) := [(-1/2, -1), (-1, -2)].to_set

-- Main proof statement
theorem math_problem_proof {x y : ℝ} (t s θ : ℝ) :
  parametric_C1 t = (x, y) →
  cartesian_C3 x y →
  (cartesian_C1 x y ∧ (x, y) ∈ intersection_C3_C1) ∨
  (parametric_C2 s = (x, y) → (cartesian_C1 x y ∧ (x, y) ∈ intersection_C3_C2)) :=
by sorry

end math_problem_proof_l506_506580


namespace find_x_tan_identity_l506_506441

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l506_506441


namespace find_x_tan_identity_l506_506436

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l506_506436


namespace max_perimeter_triangle_ABC_l506_506556

noncomputable def triangle_ABC (a b c : ℝ) : Prop :=
  ∃ (A B C : ℝ), a > 0 ∧ b > 0 ∧ c = sqrt 3 ∧ 
  (sqrt 3 + a) * (sin C - sin A) = (a + b) * sin B 

theorem max_perimeter_triangle_ABC : 
  ∀ (a b c : ℝ), triangle_ABC a b c → a + b + c ≤ 2 + sqrt 3 :=
by
  sorry

end max_perimeter_triangle_ABC_l506_506556


namespace proof_problem_l506_506940

noncomputable def log2 (n : ℝ) : ℝ := Real.log n / Real.log 2

theorem proof_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/2 * log2 x + 1/3 * log2 y = 1) : x^3 * y^2 = 64 := 
sorry 

end proof_problem_l506_506940


namespace vector_b_determination_l506_506341

open Real

variables (a b : ℝ × ℝ × ℝ)

def is_parallel (v w : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2, k * w.3)

def is_orthogonal (v w : ℝ × ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3 = 0

noncomputable def problem_statement : Prop :=
  a + b = (8, -1, -4) ∧
  is_parallel a (2, 2, 2) ∧
  is_orthogonal b (2, 2, 2) ∧
  b = (7, -2, -5)

theorem vector_b_determination : problem_statement a b :=
sorry

end vector_b_determination_l506_506341


namespace pyramid_can_be_oblique_l506_506793

-- Define what it means for the pyramid to have a regular triangular base.
def regular_triangular_base (pyramid : Type) : Prop := sorry

-- Define what it means for each lateral face to be an isosceles triangle.
def isosceles_lateral_faces (pyramid : Type) : Prop := sorry

-- Define what it means for a pyramid to be oblique.
def can_be_oblique (pyramid : Type) : Prop := sorry

-- Defining pyramid as a type.
variable (pyramid : Type)

-- The theorem stating the problem's conclusion.
theorem pyramid_can_be_oblique 
  (h1 : regular_triangular_base pyramid) 
  (h2 : isosceles_lateral_faces pyramid) : 
  can_be_oblique pyramid :=
sorry

end pyramid_can_be_oblique_l506_506793


namespace g_2x_m_plus_n_l506_506910

def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 / 2 then 1 else
  if 1 / 2 ≤ x ∧ x < 1 then -1 else 0

def g (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then 1 else 0

-- To prove g(2x) = ...
theorem g_2x (x : ℝ) : g (2 * x) = 
if 0 ≤ x ∧ x < 1 / 2 then 1 else 0 :=
by sorry

-- To prove m + n = 3
theorem m_plus_n (m n : ℤ) (h : ∀ x : ℝ, m * g (n * x) - g x = f x) : m + n = 3 :=
by sorry

end g_2x_m_plus_n_l506_506910


namespace combined_salaries_l506_506189

-- Define the variables and constants corresponding to the conditions
variable (A B D E C : ℝ)
variable (avg_salary : ℝ)
variable (num_individuals : ℕ)

-- Given conditions translated into Lean definitions 
def salary_C : ℝ := 15000
def average_salary : ℝ := 8800
def number_of_individuals : ℕ := 5

-- Define the statement to prove
theorem combined_salaries (h1 : C = salary_C) (h2 : avg_salary = average_salary) (h3 : num_individuals = number_of_individuals) : 
  A + B + D + E = avg_salary * num_individuals - salary_C := 
by 
  -- Here the proof would involve calculating the total salary and subtracting C's salary
  sorry

end combined_salaries_l506_506189


namespace max_showers_l506_506345

open Nat

variable (household water_limit water_for_drinking_and_cooking water_per_shower pool_length pool_width pool_height water_per_cubic_foot pool_leakage_rate days_in_july : ℕ)

def volume_of_pool (length width height: ℕ): ℕ :=
  length * width * height

def water_usage (drinking cooking pool leakage: ℕ): ℕ :=
  drinking + cooking + pool + leakage

theorem max_showers (h1: water_limit = 1000)
                    (h2: water_for_drinking_and_cooking = 100)
                    (h3: water_per_shower = 20)
                    (h4: pool_length = 10)
                    (h5: pool_width = 10)
                    (h6: pool_height = 6)
                    (h7: water_per_cubic_foot = 1)
                    (h8: pool_leakage_rate = 5)
                    (h9: days_in_july = 31) : 
  (water_limit - water_usage water_for_drinking_and_cooking
                                  (volume_of_pool pool_length pool_width pool_height) 
                                  ((pool_leakage_rate * days_in_july))) / water_per_shower = 7 := by
  sorry

end max_showers_l506_506345


namespace milk_amount_l506_506697

theorem milk_amount (milk_per_flour : ℕ) (flour_ratio : ℕ) (flour_amount : ℕ) (bound : 0 < flour_ratio)
  (mix_condition : ∀ (milk_per_flour flour_ratio : ℕ), milk_per_flour = 50 → flour_ratio = 250 → 
                    ∀ (flour_amount : ℕ), flour_amount = 750 → 
                    ∃ milk : ℕ, milk = 150) :
  ∀ (milk : ℕ), milk_per_flour = 50 → flour_ratio = 250 → flour_amount = 750 → milk = 150 := 
by
  intros milk hyp1 hyp2 hyp3;
  rw [hyp1, hyp2, hyp3];
  exact 150;
  sorry

end milk_amount_l506_506697


namespace complex_conjugate_in_fourth_quadrant_l506_506502

variable {z : ℂ}

theorem complex_conjugate_in_fourth_quadrant (h : (2 - complex.i) * z = 5) : 
  ∃ z : ℂ, z = 2 + complex.i ∧ complex.conj z = 2 - complex.i ∧ 
  (z.re > 0 ∧ z.im < 0) :=
by {
  use (2 + complex.i),
  have hz : z = 2 + complex.i := sorry,
  have hconj : complex.conj z = 2 - complex.i := sorry,
  have hquad : (z.re > 0 ∧ z.im < 0) := sorry,
  exact ⟨hz, hconj, hquad⟩
}

end complex_conjugate_in_fourth_quadrant_l506_506502


namespace fraction_increases_by_3_l506_506953

-- Define the original fraction
def original_fraction (x y : ℝ) : ℝ := (2 * x * y) / (3 * x - y)

-- Define the modified fraction when x and y are increased by a factor of 3
def modified_fraction (x y : ℝ) : ℝ := (2 * (3 * x) * (3 * y)) / (3 * (3 * x) - (3 * y))

-- State the theorem
theorem fraction_increases_by_3 (x y : ℝ) (h : 3 * x ≠ y) : 
  modified_fraction x y = 3 * original_fraction x y :=
by sorry

end fraction_increases_by_3_l506_506953


namespace intersection_y_coordinate_l506_506611

noncomputable def point_on_curve (a : ℝ) : ℝ × ℝ :=
  (a, a^3)

def slope_tangent (a : ℝ) : ℝ :=
  3 * a^2

def is_perpendicular_slopes (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

theorem intersection_y_coordinate (a b : ℝ) 
  (ha : point_on_curve a = (a, a^3))
  (hb : point_on_curve b = (b, b^3))
  (hp : is_perpendicular_slopes (slope_tangent a) (slope_tangent b)) :
  ∃ y : ℝ, y = - (2 / 9) - 2 * a^3 :=
begin
  sorry
end

end intersection_y_coordinate_l506_506611


namespace right_isosceles_congruent_side_length_l506_506817

theorem right_isosceles_congruent_side_length :
  let s := 2 -- side length of the equilateral triangle
  let A_eq := (sqrt 3 / 4) * s^2 -- area of the equilateral triangle
  let A_tri := A_eq / 3 -- area of each right-angled isosceles triangle
  let x := sqrt((2 * sqrt 3) / 3) -- solving for the leg length x of the right-angled isosceles triangle
  x = sqrt(6 * sqrt 3) / 3 := by
  sorry

end right_isosceles_congruent_side_length_l506_506817


namespace max_a6_diff_bounds_l506_506478

-- Part 1: Maximum value of a_6
theorem max_a6 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_nonneg : ∀ n, 0 ≤ a n) 
  (h_sum : ∀ n, (∑ i in finset.range n, a i) = S n) 
  (h_ineq : ∀ n, a (n+1) ≤ (a n + a (n+2)) / 2) 
  (h_a1 : a 1 = 1) 
  (h_a505 : a 505 = 2017) : 
  a 6 ≤ 21 :=
sorry

-- Part 2: Prove the inequality for b_k
theorem diff_bounds (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_nonneg : ∀ n, 0 ≤ a n)
  (h_sum : ∀ n, (∑ i in finset.range n, a i) = S n)
  (h_ineq : ∀ n, a (n+1) ≤ (a n + a (n+2)) / 2)
  (h_s_le_one : ∀ n, S n ≤ 1) :
  ∀ n, 0 ≤ a n - a (n+1) ∧ a n - a (n+1) ≤ 2 / (n * (n + 1)) :=
sorry

end max_a6_diff_bounds_l506_506478


namespace train_cross_time_l506_506305

def train_length : ℝ := 75  -- Length of the train in meters
def speed_man : ℝ := 5 * 1000 / 3600  -- Speed of the man in meters per second (5 kmph)
def speed_train : ℝ := 40 * 1000 / 3600  -- Speed of the train in meters per second (40 kmph)
def relative_speed : ℝ := speed_train + speed_man  -- Relative speed when moving in opposite directions

theorem train_cross_time :
  train_length / relative_speed = 6 :=
by
  unfold train_length speed_man speed_train relative_speed
  sorry

end train_cross_time_l506_506305


namespace min_value_arithmetic_sequence_l506_506480

theorem min_value_arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h_arith_seq : a n = 1 + (n - 1) * 1)
  (h_sum : S n = n * (1 + n) / 2) :
  ∃ n, (S n + 8) / a n = 9 / 2 :=
by
  sorry

end min_value_arithmetic_sequence_l506_506480


namespace number_equals_fifty_l506_506236

def thirty_percent_less_than_ninety : ℝ := 0.7 * 90

theorem number_equals_fifty (x : ℝ) (h : (5 / 4) * x = thirty_percent_less_than_ninety) : x = 50 :=
by
  sorry

end number_equals_fifty_l506_506236


namespace egg_rolls_total_l506_506154

theorem egg_rolls_total (omar_egg_rolls karen_egg_rolls lily_egg_rolls : ℕ) :
  omar_egg_rolls = 219 → karen_egg_rolls = 229 → lily_egg_rolls = 275 → 
  omar_egg_rolls + karen_egg_rolls + lily_egg_rolls = 723 := 
by
  intros h1 h2 h3
  sorry

end egg_rolls_total_l506_506154


namespace find_a_l506_506025

noncomputable def poly_root (a b : ℚ) : Prop :=
  (x : ℝ) → x^3 + a * x^2 + b * x + 54 = 0

theorem find_a (a b : ℚ) : 
  (-2 - 5 * Real.sqrt 3 = root_of_poly) →
  (a = 230 / 71) :=
begin
  -- sorry skips the proof
  sorry
end

end find_a_l506_506025


namespace Mr_Grey_bought_2_necklaces_l506_506641

def shirts_cost_each : ℕ := 26
def num_shirts : ℕ := 3
def game_cost : ℕ := 90
def necklace_cost_each : ℕ := 83
def rebate : ℕ := 12
def total_cost_after_rebate : ℕ := 322

def total_cost_before_rebate : ℕ := total_cost_after_rebate + rebate
def cost_of_shirts_and_game : ℕ := (num_shirts * shirts_cost_each) + game_cost
def cost_of_necklaces : ℕ := total_cost_before_rebate - cost_of_shirts_and_game
def num_necklaces : ℕ := cost_of_necklaces / necklace_cost_each

theorem Mr_Grey_bought_2_necklaces : num_necklaces = 2 := by
  calc 
    num_necklaces = cost_of_necklaces / necklace_cost_each : rfl
    ... = (total_cost_before_rebate - cost_of_shirts_and_game) / necklace_cost_each : rfl
    ... = ((total_cost_after_rebate + rebate) - ((num_shirts * shirts_cost_each) + game_cost)) / necklace_cost_each : rfl
    ... = ((322 + 12) - ((3 * 26) + 90)) / 83 : rfl
    ... = (334 - 168) / 83 : rfl
    ... = 166 / 83 : rfl
    ... = 2 : rfl
  sorry

end Mr_Grey_bought_2_necklaces_l506_506641


namespace max_range_eq_a_min_range_eq_a_minus_b_num_sequences_max_range_eq_b_plus_1_l506_506749

-- Definitions based on the given conditions
variables (a b : ℕ) (h : a > b)

-- Proving the maximum possible range equals a
theorem max_range_eq_a : max_range a b = a :=
by sorry

-- Proving the minimum possible range equals a - b
theorem min_range_eq_a_minus_b : min_range a b = a - b :=
by sorry

-- Proving the number of sequences resulting in the maximum range equals b + 1
theorem num_sequences_max_range_eq_b_plus_1 : num_sequences_max_range a b = b + 1 :=
by sorry

end max_range_eq_a_min_range_eq_a_minus_b_num_sequences_max_range_eq_b_plus_1_l506_506749


namespace stepa_and_petya_are_wrong_l506_506316

-- Define the six-digit number where all digits are the same.
def six_digit_same (a : ℕ) : ℕ := a * 111111

-- Define the sum of distinct prime divisors of 1001 and 111.
def prime_divisor_sum : ℕ := 7 + 11 + 13 + 3 + 37

-- Define the sum of prime divisors when a is considered.
def additional_sum (a : ℕ) : ℕ :=
  if (a = 2) || (a = 6) || (a = 8) then 2
  else if (a = 5) then 5
  else 0

-- Summarize the possible correct sums
def correct_sums (a : ℕ) : ℕ := prime_divisor_sum + additional_sum a

-- The proof statement
theorem stepa_and_petya_are_wrong (a : ℕ) :
  correct_sums a ≠ 70 ∧ correct_sums a ≠ 80 := 
by {
  sorry
}

end stepa_and_petya_are_wrong_l506_506316


namespace most_probable_hits_l506_506087

theorem most_probable_hits (p : ℝ) (q : ℝ) (k0 : ℕ) (n : ℤ) 
  (h1 : p = 0.7) (h2 : q = 1 - p) (h3 : k0 = 16) 
  (h4 : 21 < (n : ℝ) * 0.7) (h5 : (n : ℝ) * 0.7 < 23.3) : 
  n = 22 ∨ n = 23 :=
sorry

end most_probable_hits_l506_506087


namespace sufficient_but_not_necessary_condition_l506_506028

open Real

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 1) → (a + b > 2 ∧ a * b > 1) ∧ ¬((a + b > 2 ∧ a * b > 1) → (a > 1 ∧ b > 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l506_506028


namespace sum_first_15_terms_l506_506266

noncomputable def sum_of_terms (a d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

noncomputable def fourth_term (a d : ℝ) : ℝ := a + 3 * d
noncomputable def twelfth_term (a d : ℝ) : ℝ := a + 11 * d

theorem sum_first_15_terms (a d : ℝ) 
  (h : fourth_term a d + twelfth_term a d = 10) : sum_of_terms a d 15 = 75 :=
by
  sorry

end sum_first_15_terms_l506_506266


namespace arithmetic_mean_of_two_digit_multiples_of_9_l506_506713

theorem arithmetic_mean_of_two_digit_multiples_of_9 :
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  M = 58.5 :=
by
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  sorry

end arithmetic_mean_of_two_digit_multiples_of_9_l506_506713


namespace inequality_x_pow_n_ge_n_x_l506_506163

theorem inequality_x_pow_n_ge_n_x (x : ℝ) (n : ℕ) (h1 : x ≠ 0) (h2 : x > -1) (h3 : n > 0) : 
  (1 + x)^n ≥ n * x := by
  sorry

end inequality_x_pow_n_ge_n_x_l506_506163


namespace periodic_derivatives_l506_506876

noncomputable def f : ℕ → (ℝ → ℝ)
| 1 := cos
| n + 1 := deriv (f n)

theorem periodic_derivatives (x : ℝ) : f 2019 x = -cos x :=
by 
  sorry

end periodic_derivatives_l506_506876


namespace some_athletes_not_members_honor_society_l506_506330

universe u

variable {U : Type u} -- Assume U is our universe of discourse, e.g., individuals.
variables (Athletes Disciplined HonorSociety : U → Prop)

-- Conditions
def some_athletes_not_disciplined := ∃ x, Athletes x ∧ ¬Disciplined x
def all_honor_society_disciplined := ∀ x, HonorSociety x → Disciplined x

-- Correct Answer
theorem some_athletes_not_members_honor_society :
  some_athletes_not_disciplined Athletes Disciplined →
  all_honor_society_disciplined HonorSociety Disciplined →
  ∃ y, Athletes y ∧ ¬HonorSociety y :=
by
  intros h1 h2
  sorry

end some_athletes_not_members_honor_society_l506_506330


namespace length_of_side_approx_l506_506771

-- Define the problem conditions
def diameter_of_mat := 18 -- inches
def fraction_covered := 0.375

-- Define the radius and area of the circular mat
def radius_of_mat (d : ℝ) := d / 2
def area_of_mat (r : ℝ) := Real.pi * r^2

-- Define the area of the square tabletop using the fraction covered
def area_of_tabletop (a_mat : ℝ) (fraction_covered : ℝ) := a_mat / fraction_covered

-- Define the side length of the square tabletop
def side_length_of_square (a_tabletop : ℝ) := Real.sqrt a_tabletop

-- The complete problem statement proving the side length is approximately 26.05 inches.
theorem length_of_side_approx :
  let r := radius_of_mat diameter_of_mat in
  let a_mat := area_of_mat r in
  let a_tabletop := area_of_tabletop a_mat fraction_covered in
  side_length_of_square a_tabletop ≈ 26.05 := by
  -- The proof steps are omitted and replaced with 'sorry'.
  sorry

end length_of_side_approx_l506_506771


namespace elements_in_set_S_l506_506923

theorem elements_in_set_S {S : Set ℕ} :
  (S = {n ∈ ℕ | ∃ d, n^2 + 1 ≤ d ∧ d ≤ n^2 + 2 * n ∧ d ∣ n^4}) →
  ∀ m : ℤ, ∃ n ∈ S, n ≡ 0 [MOD 7] ∨ n ≡ 1 [MOD 7] ∨ n ≡ 2 [MOD 7] ∨ 
                n ≡ 5 [MOD 7] ∨ n ≡ 6 [MOD 7] ∧
  ∄ n ∈ S, n ≡ 3 [MOD 7] ∨ n ≡ 4 [MOD 7] := 
sorry

end elements_in_set_S_l506_506923


namespace log_function_quadrant_l506_506875

theorem log_function_quadrant (a b : ℝ) (h₁ : 1 < a) (h₂ : b < -1) : 
  ∀ y x : ℝ, y = log a (x + b) → ¬(x > 0 ∧ y < 0) :=
by
  sorry

end log_function_quadrant_l506_506875


namespace tom_gaming_system_value_l506_506699

theorem tom_gaming_system_value
    (V : ℝ) 
    (h1 : 0.80 * V + 80 - 10 = 160 + 30) 
    : V = 150 :=
by
  -- Logical steps for the proof will be added here.
  sorry

end tom_gaming_system_value_l506_506699


namespace smallest_n_for_inequality_l506_506725

theorem smallest_n_for_inequality (n : ℕ) : 5 + 3 * n > 300 ↔ n = 99 := by
  sorry

end smallest_n_for_inequality_l506_506725


namespace calculate_neg_three_minus_one_l506_506317

theorem calculate_neg_three_minus_one : -3 - 1 = -4 := by
  sorry

end calculate_neg_three_minus_one_l506_506317


namespace range_of_a_l506_506525

def setA : Set ℝ := { x | x^2 - x - 2 < 0 }
def setB (a : ℝ) : Set ℝ := { x | x^2 - a * x - a^2 < 0 }

theorem range_of_a (a : ℝ) : setA ⊆ setB a ↔ a ∈ set.Iio (-1 - Real.sqrt(5)) ∪ set.Ioi ((1 + Real.sqrt(5)) / 2) :=
by
  sorry

end range_of_a_l506_506525


namespace count_integers_satisfying_inequality_l506_506061

theorem count_integers_satisfying_inequality : (finset.filter (λ x : ℤ, (x + 3)^2 ≤ 4) (finset.Icc -5 -1)).card = 5 := 
by
  sorry

end count_integers_satisfying_inequality_l506_506061


namespace alpha3_plus_8beta_plus_6_eq_30_l506_506039

noncomputable def alpha_beta_quad_roots (α β : ℝ) : Prop :=
  α^2 - 2 * α - 4 = 0 ∧ β^2 - 2 * β - 4 = 0

theorem alpha3_plus_8beta_plus_6_eq_30 (α β : ℝ) (h : alpha_beta_quad_roots α β) : 
  α^3 + 8 * β + 6 = 30 :=
sorry

end alpha3_plus_8beta_plus_6_eq_30_l506_506039


namespace vector_subtraction_magnitude_l506_506925

noncomputable def scalar_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_subtraction_magnitude :
  ∀ (θ : ℝ),
  let a := (Real.cos θ, Real.sin θ)
  let b := (1, Real.sqrt 2)
  let angle := Real.pi / 6 in
  scalar_product a b = Real.sqrt 3 * Real.sqrt 3 / 2 →
  (vector_length ((a.1 - b.1, a.2 - b.2))) = 1 :=
by
  intros θ a b angle h_angle
  sorry

end vector_subtraction_magnitude_l506_506925


namespace find_x_l506_506381

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l506_506381


namespace number_of_satisfying_integers_l506_506463

def digit_sum (n : ℕ) : ℕ := n.digits10.sum

def has_odd_digit_sum (n : ℕ) : Prop := digit_sum n % 2 = 1

def satisfies_conditions (x : ℕ) : Prop :=
  x > 0 ∧ x ≤ 1000 ∧ has_odd_digit_sum x ∧ has_odd_digit_sum (x + 1)

theorem number_of_satisfying_integers : 
  {x : ℕ | satisfies_conditions x}.to_finset.card = 46 :=
sorry

end number_of_satisfying_integers_l506_506463


namespace minimize_R_FG_l506_506703

-- Definitions for the resistors and their constraints
variables {a1 a2 a3 a4 a5 a6 : ℝ}

-- Assuming the conditions given
axiom h1 : a1 > a2
axiom h2 : a2 > a3
axiom h3 : a3 > a4
axiom h4 : a4 > a5
axiom h5 : a5 > a6

-- Definition of the configuration and the function that calculates the total resistance
noncomputable def R_AB (R1 R2 R3 : ℝ) : ℝ :=
  (R1 * R2 + R1 * R3 + R2 * R3) / (R1 + R2)

noncomputable def R_CD (R1 R2 R3 R4 : ℝ) : ℝ :=
  let S1 := ∑ x in [R1, R2, R3, R4].combinations 2, x.prod
  let S2 := ∑ x in [R1, R2, R3, R4].combinations 3, x.prod
  S2 / S1

noncomputable def R_FG (R1 R2 R3 R4 : ℝ) : ℝ :=
  R_AB R1 R2 R3 + R4 -- Simplified for showing one condition

-- The statement that proves the minimized resistance condition
theorem minimize_R_FG (h1 : a1 > a2) (h2 : a2 > a3) (h3 : a3 > a4) (h4 : a1 > a4) (h5 : R1 = a1) (h6 : R2 = a2) (h7 : R3 = a3) (h8 : R4 = a4) : 
  R_FG a1 a2 a3 a4 < R_FG a1 a2 a3 a5 :=
sorry

end minimize_R_FG_l506_506703


namespace max_range_of_walk_min_range_of_walk_count_of_max_range_sequences_l506_506757

section RandomWalk

variables {a b : ℕ} (h : a > b)

def max_range_walk : ℕ := a
def min_range_walk : ℕ := a - b
def count_max_range_sequences : ℕ := b + 1

theorem max_range_of_walk (h : a > b) : max_range_walk h = a :=
by
  sorry

theorem min_range_of_walk (h : a > b) : min_range_walk h = a - b :=
by
  sorry

theorem count_of_max_range_sequences (h : a > b) : count_max_range_sequences h = b + 1 :=
by
  sorry

end RandomWalk

end max_range_of_walk_min_range_of_walk_count_of_max_range_sequences_l506_506757


namespace M_equals_N_l506_506524

-- Define the sets M and N
def M : Set ℝ := {x | 0 ≤ x}
def N : Set ℝ := {y | 0 ≤ y}

-- State the main proof goal
theorem M_equals_N : M = N :=
by
  sorry

end M_equals_N_l506_506524


namespace find_abs_alpha_l506_506616

open Complex

noncomputable def alpha : ℂ := sorry
noncomputable def beta : ℂ := conj alpha

def alpha_over_beta_cubed_is_real (α β : ℂ) := (α / (β ^ 3)).im = 0

lemma alpha_beta_conjugate (α β : ℂ) : β = conj α := sorry

definition condition_1 (α β : ℂ) : Prop := β = conj α

definition condition_2 (α β : ℂ) : Prop := alpha_over_beta_cubed_is_real α β

definition condition_3 (α β : ℂ) : Prop := abs (α - β) = 4

theorem find_abs_alpha (α β : ℂ) (h1 : condition_1 α β) (h2 : condition_2 α β) (h3 : condition_3 α β) : abs α = 4 * sqrt 3 / 3 :=
sorry

end find_abs_alpha_l506_506616


namespace collective_earnings_l506_506644

theorem collective_earnings:
  let lloyd_hours := 10.5
  let mary_hours := 12.0
  let tom_hours := 7.0
  let lloyd_normal_hours := 7.5
  let mary_normal_hours := 8.0
  let tom_normal_hours := 9.0
  let lloyd_rate := 4.5
  let mary_rate := 5.0
  let tom_rate := 6.0
  let lloyd_overtime_rate := 2.5 * lloyd_rate
  let mary_overtime_rate := 3.0 * mary_rate
  let tom_overtime_rate := 2.0 * tom_rate
  let lloyd_earnings := (lloyd_normal_hours * lloyd_rate) + ((lloyd_hours - lloyd_normal_hours) * lloyd_overtime_rate)
  let mary_earnings := (mary_normal_hours * mary_rate) + ((mary_hours - mary_normal_hours) * mary_overtime_rate)
  let tom_earnings := (tom_hours * tom_rate)
  let total_earnings := lloyd_earnings + mary_earnings + tom_earnings
  total_earnings = 209.50 := by
  sorry

end collective_earnings_l506_506644


namespace projection_transformation_l506_506119

variable (v0 v1 v2 : ℝ^2)

def proj_matrix_onto_4_2 : ℝ^2 → ℝ^2 :=
  λ v, (1 / 20 : ℝ) • (matrix.mul_vec
                         (matrix.of (λ i j, if i = 0 ∧ j = 0 then (16 : ℝ) else 
                                          if i = 0 ∧ j = 1 then (8 : ℝ) else 
                                          if i = 1 ∧ j = 0 then (8 : ℝ) else 
                                          (4 : ℝ)))
                         v)

def proj_matrix_onto_2_2 : ℝ^2 → ℝ^2 :=
  λ v, (1 / 8 : ℝ) • (matrix.mul_vec 
                        (matrix.of (λ i j, if i = 0 ∧ j = 0 then (4 : ℝ) else 
                                         if i = 0 ∧ j = 1 then (4 : ℝ) else 
                                         if i = 1 ∧ j = 0 then (4 : ℝ) else 
                                         (4 : ℝ)))
                        v)

def transformation_matrix : matrix (fin 2) (fin 2) ℝ :=
  (1 / 160 : ℝ) • (matrix.of (λ i j, if i = 0 ∧ j = 0 then (96 : ℝ) else 
                                         if i = 0 ∧ j = 1 then (48 : ℝ) else 
                                         if i = 1 ∧ j = 0 then (96 : ℝ) else 
                                         (48 : ℝ)))

theorem projection_transformation :
  matrix.mul_vec transformation_matrix v0 = 
  proj_matrix_onto_2_2 (proj_matrix_onto_4_2 v0) :=
sorry

end projection_transformation_l506_506119


namespace max_operation_result_l506_506810

theorem max_operation_result : ∀ (n : ℕ), (100 ≤ n ∧ n ≤ 999) → 3 * (300 - n) ≤ 600 :=
by
  intros n hn
  suffices h: n = 100 ∨ n ≠ 100 by
    cases h
    · rw h
      norm_num
    · sorry
  sorry

end max_operation_result_l506_506810


namespace necessary_and_sufficient_condition_l506_506763

open Real

theorem necessary_and_sufficient_condition 
  {x y : ℝ} (p : x > y) (q : x - y + sin (x - y) > 0) : 
  (x > y) ↔ (x - y + sin (x - y) > 0) :=
sorry

end necessary_and_sufficient_condition_l506_506763


namespace prob_two_red_balls_l506_506283

-- Define the initial conditions for the balls in the bag
def red_balls : ℕ := 5
def blue_balls : ℕ := 6
def green_balls : ℕ := 2
def total_balls : ℕ := red_balls + blue_balls + green_balls

-- Define the probability of picking a red ball first
def prob_red1 : ℚ := red_balls / total_balls

-- Define the remaining number of balls and the probability of picking a red ball second
def remaining_red_balls : ℕ := red_balls - 1
def remaining_total_balls : ℕ := total_balls - 1
def prob_red2 : ℚ := remaining_red_balls / remaining_total_balls

-- Define the combined probability of both events
def prob_both_red : ℚ := prob_red1 * prob_red2

-- Statement of the theorem to be proved
theorem prob_two_red_balls : prob_both_red = 5 / 39 := by
  sorry

end prob_two_red_balls_l506_506283


namespace Deepak_and_Wife_meet_time_l506_506180

theorem Deepak_and_Wife_meet_time 
    (circumference : ℕ) 
    (Deepak_speed : ℕ)
    (wife_speed : ℕ) 
    (conversion_factor_km_hr_to_m_hr : ℕ) 
    (minutes_per_hour : ℕ) :
    circumference = 726 →
    Deepak_speed = 4500 →  -- speed in meters per hour
    wife_speed = 3750 →  -- speed in meters per hour
    conversion_factor_km_hr_to_m_hr = 1000 →
    minutes_per_hour = 60 →
    (726 / ((4500 + 3750) / 1000) * 60 = 5.28) :=
by 
    sorry

end Deepak_and_Wife_meet_time_l506_506180


namespace car_y_slope_three_times_car_x_slope_l506_506829

variables 
  (v t d : ℝ) 
  (hx : t > 0)  -- time t must be greater than 0
  (hv : v > 0)  -- speed v must be greater than 0

def car_x_slope := d / t  -- slope of Car X's distance-time line
def car_y_slope := d / (t / 3)  -- slope of Car Y's distance-time line

theorem car_y_slope_three_times_car_x_slope :
  car_y_slope = 3 * car_x_slope :=
by
  unfold car_y_slope car_x_slope
  rw [div_div]
  ring
  sorry

end car_y_slope_three_times_car_x_slope_l506_506829


namespace find_a_b_l506_506503

noncomputable def complex_z : ℂ := (1 + complex.I^2 + 3*(1 - complex.I)) / (2 + complex.I)

theorem find_a_b (a b : ℝ) (z := complex_z) :
  z^2 + (a + b * complex.I) * z + b = 1 + complex.I → a = -3 ∧ b = 4 :=
by sorry

end find_a_b_l506_506503


namespace ice_cream_scoops_l506_506320

def scoops_of_ice_cream : ℕ := 1 -- single cone has 1 scoop

def scoops_double_cone : ℕ := 2 * scoops_of_ice_cream -- double cone has two times the scoops of a single cone

def scoops_banana_split : ℕ := 3 * scoops_of_ice_cream -- banana split has three times the scoops of a single cone

def scoops_waffle_bowl : ℕ := scoops_banana_split + 1 -- waffle bowl has one more scoop than banana split

def total_scoops : ℕ := scoops_of_ice_cream + scoops_double_cone + scoops_banana_split + scoops_waffle_bowl

theorem ice_cream_scoops : total_scoops = 10 :=
by
  sorry

end ice_cream_scoops_l506_506320


namespace extreme_value_f_max_b_a_plus_1_l506_506051

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x + (1 / 2) * x^2

noncomputable def g (x a b : ℝ) : ℝ := (1 / 2) * x^2 + a * x + b

theorem extreme_value_f :
  ∃ (x : ℝ), x = 0 ∧ f x = 1.5 :=
begin
  use 0,
  split,
  { refl },
  { simp [f], norm_num, }
end

theorem max_b_a_plus_1 (a b : ℝ) :
  (∀ x, f x ≥ g x a b) →
  b * (a + 1) ≤ (a + 1)^2 - (a + 1)^2 * log (a + 1) :=
begin
  sorry
end

end extreme_value_f_max_b_a_plus_1_l506_506051


namespace sum_of_powers_zero_l506_506609

theorem sum_of_powers_zero (a : ℕ → ℝ) (n : ℕ) (h : ∀ k, k ≤ n → k % 2 = 1 → ∑ i in finset.range n, a i ^ k = 0) :
  ∀ k, k % 2 = 1 → ∑ i in finset.range n, a i ^ k = 0 :=
by sorry

end sum_of_powers_zero_l506_506609


namespace geometric_series_first_term_l506_506206

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l506_506206


namespace divisibility_l506_506020

theorem divisibility {n A B k : ℤ} (h_n : n = 1000 * B + A) (h_k : k = A - B) :
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) :=
by
  sorry

end divisibility_l506_506020


namespace remainder_of_5n_mod_11_l506_506945

theorem remainder_of_5n_mod_11 (n : ℤ) (h : n % 11 = 1) : (5 * n) % 11 = 5 := 
by
  sorry

end remainder_of_5n_mod_11_l506_506945


namespace find_x_tan_eq_l506_506433

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l506_506433


namespace total_pages_l506_506651

-- Definitions based on conditions
def math_pages : ℕ := 10
def extra_reading_pages : ℕ := 3
def reading_pages : ℕ := math_pages + extra_reading_pages

-- Statement of the proof problem
theorem total_pages : math_pages + reading_pages = 23 := by 
  sorry

end total_pages_l506_506651


namespace find_first_term_l506_506198

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l506_506198


namespace maciek_total_cost_l506_506786

theorem maciek_total_cost :
  let p := 4
  let cost_of_chips := 1.75 * p
  let pretzels_cost := 2 * p
  let chips_cost := 2 * cost_of_chips
  let t := pretzels_cost + chips_cost
  t = 22 :=
by
  sorry

end maciek_total_cost_l506_506786


namespace product_inspection_probability_l506_506296

theorem product_inspection_probability :
  let p_good_as_defective := 0.02,
      p_defective_as_good := 0.01,
      p_good_as_good := 1 - p_good_as_defective,
      p_defective_as_defective := 1 - p_defective_as_good,
      num_good := 3,
      num_defective := 1
  in (p_good_as_good ^ num_good) * (p_defective_as_defective ^ num_defective) = 0.932 :=
by
  let p_good_as_defective := 0.02
  let p_defective_as_good := 0.01
  let p_good_as_good := 1 - p_good_as_defective
  let p_defective_as_defective := 1 - p_defective_as_good
  let num_good := 3
  let num_defective := 1
  have h1 : p_good_as_good = 0.98, from rfl
  have h2 : p_defective_as_defective = 0.99, from rfl
  calc
    (p_good_as_good ^ num_good) * (p_defective_as_defective ^ num_defective)
    = (0.98 ^ 3) * (0.99 ^ 1) : by rw [h1, h2]
    ... = 0.98 * 0.98 * 0.98 * 0.99 : by norm_num
    ... = 0.932371992 : by norm_num -- Retain higher precision to ensure correctness
    ... ≈ 0.932 : by norm_num -- Apply rounding to three decimal places

end product_inspection_probability_l506_506296


namespace problem_solution_correct_l506_506188

noncomputable def smallest_value_of_n : ℕ :=
  if h : ∃ n : ℕ, (∀ x y z : ℝ, (x ∈ set.Icc (0:ℝ) n ∧ y ∈ set.Icc (0:ℝ) n ∧ z ∈ set.Icc (0:ℝ) n) →
    (x + y + z < (3 / 2) * n ∧ |x - y| ≥ 2 ∧ |y - z| ≥ 2 ∧ |z - x| ≥ 2) →
    (measure_theory.probability ((x, y, z) ∈ { t : ℝ × ℝ × ℝ | t.1 + t.2.1 + t.2.2 < 3 * n / 2 ∧
      |t.1 - t.2.1| ≥ 2 ∧ |t.2.1 - t.2.2| ≥ 2 ∧ |t.2.2 - t.1| ≥ 2 }) > 1/4)) then
    classical.some h else 0

theorem problem_solution_correct : smallest_value_of_n = 8 :=
sorry

end problem_solution_correct_l506_506188


namespace maximum_range_of_walk_minimum_range_of_walk_number_of_max_range_sequences_l506_506746

theorem maximum_range_of_walk (a b : ℕ) (h : a > b) : 
  (∃ max_range : ℕ, max_range = a) :=
by {
  use a,
  sorry
}

theorem minimum_range_of_walk (a b : ℕ) (h : a > b) : 
  (∃ min_range : ℕ, min_range = a - b) :=
by {
  use a - b,
  sorry
}

theorem number_of_max_range_sequences (a b : ℕ) (h : a > b) : 
  (∃ num_sequences : ℕ, num_sequences = b + 1) :=
by {
  use b + 1,
  sorry
}

end maximum_range_of_walk_minimum_range_of_walk_number_of_max_range_sequences_l506_506746


namespace sum_f_equals_3025_over_2_l506_506045

def f (x : Int) : ℚ :=
  if x ≤ 0 then 2^x else f (x - 2)

theorem sum_f_equals_3025_over_2 :
  (∑ i in Finset.range (2017 + 1), f i) = 3025 / 2 :=
by
  sorry

end sum_f_equals_3025_over_2_l506_506045


namespace James_distance_ridden_l506_506599

theorem James_distance_ridden :
  let s := 16
  let t := 5
  let d := s * t
  d = 80 :=
by
  sorry

end James_distance_ridden_l506_506599


namespace smallest_multiple_of_125_has_specific_divisors_l506_506629

def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = b * k
def num_divisors (n : ℕ) : ℕ := (finset.range n).filter (λ k => n % k = 0).card

theorem smallest_multiple_of_125_has_specific_divisors (m : ℕ) (h₁ : is_multiple_of m 125) (h₂ : num_divisors m = 100) :
  m = 125 * 2^3 * 5^21 :=
sorry

end smallest_multiple_of_125_has_specific_divisors_l506_506629


namespace diff_of_squares_expression_l506_506728

theorem diff_of_squares_expression (m n : ℝ) :
  (3 * m + n) * (3 * m - n) = (3 * m)^2 - n^2 :=
by
  sorry

end diff_of_squares_expression_l506_506728


namespace sphere_tangent_plane_normal_line_l506_506260

variable {F : ℝ → ℝ → ℝ → ℝ}
def sphere (x y z : ℝ) : Prop := x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 5 = 0

def tangent_plane (x y z : ℝ) : Prop := 2*x + y + 2*z - 15 = 0

def normal_line (x y z : ℝ) : Prop := (x - 3) / 2 = (y + 1) / 1 ∧ (y + 1) / 1 = (z - 5) / 2

theorem sphere_tangent_plane_normal_line :
  sphere 3 (-1) 5 →
  tangent_plane 3 (-1) 5 ∧ normal_line 3 (-1) 5 :=
by
  intros h
  constructor
  sorry
  sorry

end sphere_tangent_plane_normal_line_l506_506260


namespace number_of_incorrect_inequalities_l506_506693

theorem number_of_incorrect_inequalities :
  (sqrt(2 * sqrt(2 * sqrt(2 * sqrt(2)))) < 2) ∧
  (sqrt(2 + sqrt(2 + sqrt(2 + sqrt(2)))) < 2) ∧
  (sqrt(3 * sqrt(3 * sqrt(3 * sqrt(3)))) < 3) ∧
  (sqrt(3 + sqrt(3 + sqrt(3 + sqrt(3)))) < 3) → 0 = 0 :=
by
  sorry

end number_of_incorrect_inequalities_l506_506693


namespace find_x_y_l506_506893

theorem find_x_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x - ↑⌊x⌋ = {x}) (h2 : y - ↑⌊y⌋ = {y}) (h3 : ⌊x⌋ - {x} = x - ⌊x⌋) 
  (h4 : ⌊y⌋ - {y} = y - ⌊y⌋) (h5 : {x} = {y}) (h6 : x ≠ y) :
  x = 3 / 2 ∧ y = 5 / 2 :=
by
  sorry

end find_x_y_l506_506893


namespace first_player_has_winning_strategy_l506_506245

-- Define the structure of the game state
structure GameState where
  current_number : ℕ
  player_turn : ℕ -- 1 for first player, 2 for second player
  used_numbers : set ℕ

-- Define the property of a valid number in the game
def is_valid_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ 
  (∀ d, d ∈ [n / 100, (n / 10) % 10, n % 10] → d ≠ 0) ∧
  ((n / 100 + (n / 10) % 10 + n % 10) % 9 = 0)

-- Define the initial game state
def initial_state : GameState :=
  { current_number := 999, player_turn := 1, used_numbers := {999} }

-- Define the transition between game states
def transition (s : GameState) (n : ℕ) : GameState :=
  { current_number := n, player_turn := 3 - s.player_turn, used_numbers := s.used_numbers ∪ {n} }

-- Define the winning condition check
def is_winner (s : GameState) : ℕ :=
  if (∀ n, is_valid_number n ∧ n ≠ s.current_number ∧ first_digit s.current_number = last_digit n ∧ n ∉ s.used_numbers → false) 
  then s.player_turn 
  else 0

-- The final proof statement
theorem first_player_has_winning_strategy : 
  (∀ s : GameState, s = initial_state → (is_winner (transition s (next_move s)))) = 1 := 
  sorry

end first_player_has_winning_strategy_l506_506245


namespace speed_in_still_water_l506_506264

theorem speed_in_still_water (U D : ℝ) (hU : U = 30) (hD : D = 60) :
  (U + D) / 2 = 45 := by
  rw [hU, hD]
  norm_num
  sorry

end speed_in_still_water_l506_506264


namespace max_range_walk_min_range_walk_count_max_range_sequences_l506_506754

variable {a b : ℕ}

-- Condition: a > b
def valid_walk (a b : ℕ) : Prop := a > b

-- Proof that the maximum possible range of the walk is a
theorem max_range_walk (h : valid_walk a b) : 
  (a + b) = a + b := sorry

-- Proof that the minimum possible range of the walk is a - b
theorem min_range_walk (h : valid_walk a b) : 
  (a - b) = a - b := sorry

-- Proof that the number of different sequences with the maximum possible range is b + 1
theorem count_max_range_sequences (h : valid_walk a b) : 
  b + 1 = b + 1 := sorry

end max_range_walk_min_range_walk_count_max_range_sequences_l506_506754


namespace imo_42_problem_l506_506461

theorem imo_42_problem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b) >= 1 :=
sorry

end imo_42_problem_l506_506461


namespace find_P_CI_l506_506689

def simple_interest (P r t : ℝ) : ℝ :=
  P * r * t / 100

def compound_interest (P r t : ℝ) : ℝ :=
  P * ((1 + r / 100) ^ t - 1)

def given_conditions (SI CI P_CI : ℝ) : Prop :=
  SI = 603.75 ∧
  SI = CI / 2 ∧
  CI = 2 * SI ∧
  CI = compound_interest P_CI 7 2

theorem find_P_CI : ∃ (P_CI : ℝ), given_conditions (simple_interest 603.75 14 6) (simple_interest 603.75 14 6) P_CI ∧ P_CI = 8333.33 := 
  sorry

end find_P_CI_l506_506689


namespace problem_equivalent_proof_l506_506996

/-
Let S be the set of points (x, y) in the coordinate plane such that 
two of the three quantities 5, x + 3, and y - 6 are equal and the third of 
these quantities is no greater than or exactly half of the common value.
Prove S == parts of a right triangle and a separate point.
-/

def S : Set (ℝ × ℝ) :=
  {p | let (x, y) := p in 
       ((x = 2 ∧ y ≤ 11) ∨ (x = 2 ∧ y = 8.5) ∨ (y = 11 ∧ x ≤ 2) ∨ (y = 11  ∧ x = -0.5) ∨ (x = 7 ∧ y = 16))}

theorem problem_equivalent_proof:
  S = {p | let (x, y) := p in 
       ((x = 2 ∧ y ≤ 11) ∨ (x = 2 ∧ y = 8.5) ∨ (y = 11 ∧ x ≤ 2) ∨ (y = 11  ∧ x = -0.5) ∨ (x = 7 ∧ y = 16))} :=
by
  sorry

end problem_equivalent_proof_l506_506996


namespace moment_of_inertia_unit_masses_moment_of_inertia_general_masses_l506_506275

-- Define the distances between points
variables {n : ℕ}
variables (a_ij : fin n → fin n → ℝ)

-- Define the units masses case
theorem moment_of_inertia_unit_masses (I_O : ℝ) : 
  I_O = (1 / n) * ∑ i j in finset.range(n), if i < j then (a_ij i j) ^ 2 else 0 := 
sorry

-- Define the distances and masses between points
variables (m : fin n → ℝ)
variable sum_m {i j : fin n}

-- Define the general masses case
theorem moment_of_inertia_general_masses (I_O : ℝ) :
  I_O = (1 / (∑ i in finset.range(n), m i)) * ∑ i j in finset.range(n), if i < j then (m i) * (m j) * (a_ij i j) ^ 2 else 0 := 
sorry

end moment_of_inertia_unit_masses_moment_of_inertia_general_masses_l506_506275


namespace sequence_term_formula_l506_506009

theorem sequence_term_formula 
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (h : ∀ n, S n = n^2 + 3 * n)
  (h₁ : a 1 = 4)
  (h₂ : ∀ n, 1 < n → a n = S n - S (n - 1)) :
  ∀ n, a n = 2 * n + 2 :=
by
  sorry

end sequence_term_formula_l506_506009


namespace sequence_equality_l506_506353

theorem sequence_equality (a : ℕ → ℕ) 
  (h_increasing : ∀ i j : ℕ, i ≤ j → a i ≤ a j)
  (h_divisors : ∀ i j : ℕ, Nat.num_divisors (i + j) = Nat.num_divisors (a i + a j)) :
  ∀ n : ℕ, a n = n := 
sorry

end sequence_equality_l506_506353


namespace find_x_l506_506384

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l506_506384


namespace probability_no_extreme_points_l506_506820

-- Definitions of conditions and parameters
variables {η : ℝ}
variables {σ : ℝ}
variables {f : ℝ → ℝ := λ x, (1 / 3) * x^3 + x^2 + η^2 * x}

-- Assumptions
axiom eta_normal : ∀ x : ℝ, 0 < σ → ∃ (η : ℝ), η ~ Normal 1 σ
axiom P_eta_neg1 : Prob (η < -1) = 0.2

-- Theorem to prove
theorem probability_no_extreme_points : Prob (has_no_extreme_points f) = 0.7 :=
by
  -- here we will have the proof steps
sorry

-- Helper function to determine whether the function has no extreme points
def has_no_extreme_points (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (f.deriv x) ≠ 0

end probability_no_extreme_points_l506_506820


namespace find_x_value_l506_506411

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l506_506411


namespace find_x_l506_506380

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l506_506380


namespace probability_prime_ball_l506_506664

/-- Ten balls, numbered 6 through 15, are placed in a hat. Each ball is equally likely to be chosen. 
If one ball is chosen, the probability that the number on the selected ball is a prime number 
is 3 out of 10. -/
theorem probability_prime_ball :
  let balls := {n | 6 ≤ n ∧ n ≤ 15} in
  let primes := {n | n ∈ balls ∧ Nat.Prime n} in
  (primes.card.to_rat / balls.card.to_rat = (3 : ℚ) / 10) := 
by
  sorry

end probability_prime_ball_l506_506664


namespace rate_of_profit_is_80_l506_506739

def SellingPrice : ℝ := 90
def CostPrice : ℝ := 50
def Profit : ℝ := SellingPrice - CostPrice
def RateOfProfit : ℝ := (Profit / CostPrice) * 100

theorem rate_of_profit_is_80 :
  RateOfProfit = 80 := by
  sorry

end rate_of_profit_is_80_l506_506739


namespace sum_of_twos_and_threes_3024_l506_506540

theorem sum_of_twos_and_threes_3024 : ∃ n : ℕ, n = 337 ∧ (∃ (a b : ℕ), 3024 = 2 * a + 3 * b) :=
sorry

end sum_of_twos_and_threes_3024_l506_506540


namespace train_speed_l506_506803

theorem train_speed (train_length : ℝ) (man_speed_kmph : ℝ) (passing_time : ℝ) : 
  train_length = 160 → man_speed_kmph = 6 →
  passing_time = 6 → (train_length / passing_time + man_speed_kmph * 1000 / 3600) * 3600 / 1000 = 90 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- further proof steps are omitted
  sorry

end train_speed_l506_506803


namespace complex_inverse_conjugate_plus_i_l506_506002

-- Definitions
def z : ℂ := 2 - complex.I
def conj_z := complex.conj z

-- Statement to prove
theorem complex_inverse_conjugate_plus_i : 
  (1 / (conj_z + complex.I) = 1 / 4 - 1 / 4 * complex.I) :=
by
  sorry

end complex_inverse_conjugate_plus_i_l506_506002


namespace polynomial_bound_l506_506158

theorem polynomial_bound (P : ℝ → ℝ) (n : ℕ) (a : Fin n → ℝ) :
  (P = (λ x, x^n + ∑ i in Finset.range n, a i * x^(n-i-1))) →
  ∃ x₀ : ℝ, -1 ≤ x₀ ∧ x₀ ≤ 1 ∧ |P x₀| ≥ 1/(2^(n-1)) :=
by sorry

end polynomial_bound_l506_506158


namespace smallest_five_digit_congruent_to_three_mod_seventeen_l506_506721

theorem smallest_five_digit_congruent_to_three_mod_seventeen :
  ∃ (n : ℤ), 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 3 ∧ n = 10012 :=
by
  sorry

end smallest_five_digit_congruent_to_three_mod_seventeen_l506_506721


namespace find_m_through_point_l506_506882

theorem find_m_through_point :
  ∃ m : ℝ, ∀ (x y : ℝ), ((y = (m - 1) * x - 4) ∧ (x = 2) ∧ (y = 4)) → m = 5 :=
by 
  -- Sorry can be used here to skip the proof as instructed
  sorry

end find_m_through_point_l506_506882


namespace matrix_inverse_uniq_l506_506339

open Matrix

def A (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, -2], ![x, y]]

lemma determinant_condition (x y : ℝ) :
  A x y * A x y = 1 :=
begin
  simp [A, Matrix.mul, dot_product, Fin.vec_cons, Fin.to_nat, List.sum,
        add_assoc, mul_comm, mul_assoc],
  sorry
end

theorem matrix_inverse_uniq (x y : ℝ) :
  A x y * A x y = 1 → (x, y) = (15/2, -4) :=
begin
  intros h,
  have h1 : 16 - 2 * x = 1, sorry,
  have h2 : -8 - 2 * y = 0, sorry,
  have h3 : 4 * x + x * y = 0, sorry,
  have h4 : -2 * x + y^2 = 1, sorry,
  simp at h1 h2 h3 h4,
  exact ⟨h1, h2, h3, h4⟩ 
end

end matrix_inverse_uniq_l506_506339


namespace deductive_reasoning_example_is_A_l506_506812

def isDeductive (statement : String) : Prop := sorry

-- Define conditions
def optionA : String := "Since y = 2^x is an exponential function, the function y = 2^x passes through the fixed point (0,1)"
def optionB : String := "Guessing the general formula for the sequence 1/(1×2), 1/(2×3), 1/(3×4), ... as a_n = 1/(n(n+1)) (n ∈ ℕ⁺)"
def optionC : String := "Drawing an analogy from 'In a plane, two lines perpendicular to the same line are parallel' to infer 'In space, two planes perpendicular to the same plane are parallel'"
def optionD : String := "From the circle's equation in the Cartesian coordinate plane (x-a)² + (y-b)² = r², predict that the equation of a sphere in three-dimensional Cartesian coordinates is (x-a)² + (y-b)² + (z-c)² = r²"

theorem deductive_reasoning_example_is_A : isDeductive optionA :=
by
  sorry

end deductive_reasoning_example_is_A_l506_506812


namespace maciek_total_purchase_cost_l506_506782

-- Define the cost of pretzels
def pretzel_cost : ℕ := 4

-- Define the cost of chips
def chip_cost : ℕ := pretzel_cost + (75 * pretzel_cost) / 100

-- Calculate the total cost
def total_cost : ℕ := 2 * pretzel_cost + 2 * chip_cost

-- Rewrite the math proof problem statement
theorem maciek_total_purchase_cost : total_cost = 22 :=
by
  -- Skip the proof
  sorry

end maciek_total_purchase_cost_l506_506782


namespace mod_inverse_35_mod_37_l506_506854

theorem mod_inverse_35_mod_37 : ∃ a : ℤ, 35 * a ≡ 1 [MOD 37] :=
  ⟨18, by
    sorry
  ⟩

end mod_inverse_35_mod_37_l506_506854


namespace am_bn_difference_l506_506182

noncomputable def factorial (n: ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def condition (m n : ℕ) (a b : Fin m → ℕ) (c d : Fin n → ℕ) :=
  a 0 ≥ a 1 ∧ b 0 ≥ b 1 ∧
  (∀ i j, i ≠ j → a i ≥ a j) ∧ (∀ i j, i ≠ j → b i ≥ b j) ∧
  (factorial (a 0) * factorial (a 1) * ... * factorial (a fin.last) /
   factorial (b 0) * factorial (b 1) * ... * factorial (b fin.last) = 2024)

def min_am_bn (m n : ℕ) (a b : Fin m → ℕ) (c d : Fin n → ℕ) :=
  (a fin.last + b fin.last) ≤ (c fin.last + d fin.last)

theorem am_bn_difference : 
  ∃ (m n : ℕ) (a b : Fin m → ℕ) (c d : Fin n → ℕ),
  condition m n a b ∧ min_am_bn m n a b c d →
  |a (fin.last m) - b (fin.last n)| = 1 :=
by
  sorry

end am_bn_difference_l506_506182


namespace tangent_line_hyperbola_l506_506488

theorem tangent_line_hyperbola :
  ∀ (x y : ℝ), (x, y) = (sqrt 2, sqrt 2) → (x^2 - y^2 / 2 = 1) →
    (2 * x - y - sqrt 2 = 0) := by
  sorry

end tangent_line_hyperbola_l506_506488


namespace probability_at_most_two_heads_l506_506241

theorem probability_at_most_two_heads : 
  let outcomes := {x : list Bool | x.length = 3} in
  let heads_count (l : list Bool) := l.count true in
  let favorable_outcomes := {l ∈ outcomes | heads_count l <= 2}.to_finset in
  let total_outcomes := outcomes.to_finset.card in
  favorable_outcomes.card / total_outcomes = 7 / 8 :=
by
  sorry

end probability_at_most_two_heads_l506_506241


namespace problem1_problem2_problem3_problem4_l506_506279

theorem problem1 : sqrt 5 - ((sqrt 3 + sqrt 15) / (sqrt 6 * sqrt 2)) = -1 := by sorry

theorem problem2 : (sqrt 48 - 4 * sqrt (1 / 8)) - (3 * sqrt (1 / 3) - 2 * sqrt (1 / 2)) = 3 * sqrt 3 := by sorry

theorem problem3 : (3 + sqrt 5) * (3 - sqrt 5) - (sqrt 3 - 1)^2 = 2 * sqrt 3 := by sorry

theorem problem4 : ((-sqrt 3 + 1) * (sqrt 3 - 1)) - sqrt ((-3)^2) + 1 / (2 - sqrt 5) = -3 - sqrt 5 := by sorry

end problem1_problem2_problem3_problem4_l506_506279


namespace maciek_total_cost_l506_506788

-- Define the cost of pretzels and the additional cost percentage for chips
def cost_pretzel : ℝ := 4
def cost_chip := cost_pretzel + (cost_pretzel * 0.75)

-- Number of packets Maciek bought for pretzels and chips
def num_pretzels : ℕ := 2
def num_chips : ℕ := 2

-- Total cost calculation
def total_cost := (cost_pretzel * num_pretzels) + (cost_chip * num_chips)

-- The final theorem statement
theorem maciek_total_cost :
  total_cost = 22 := by
  sorry

end maciek_total_cost_l506_506788


namespace number_of_ways_to_arrange_elements_l506_506474

def a_n (n : ℕ) : ℝ :=
  (1 / (4 * Real.sqrt 65)) * 
  ((15 + Real.sqrt 65) * (5 + Real.sqrt 65) ^ n - 
  (15 - Real.sqrt 65) * (5 - Real.sqrt 65) ^ n)

theorem number_of_ways_to_arrange_elements :
  ∀ n, 
  a_n n = (1 / (4 * Real.sqrt 65)) * 
  ((15 + Real.sqrt 65) * (5 + Real.sqrt 65) ^ n - 
  (15 - Real.sqrt 65) * (5 - Real.sqrt 65) ^ n) :=
by
  sorry

end number_of_ways_to_arrange_elements_l506_506474


namespace find_missing_sibling_l506_506310

noncomputable def number_of_siblings (mean: ℝ) (given_siblings: List ℕ) (total_girls: ℕ) : ℕ :=
let total_siblings := round (mean * total_girls)
let reported_siblings := (given_siblings.foldl (· + ·) 0)
total_siblings - reported_siblings

theorem find_missing_sibling :
  let given_siblings := [1, 10, 4, 3, 3, 11, 3, 10]
  let total_girls := 9
  let mean := 5.7
  number_of_siblings mean given_siblings total_girls = 6 :=
by
  dsimp [number_of_siblings]
  sorry

end find_missing_sibling_l506_506310


namespace true_statements_l506_506464

def is_decreasing {α : Type*} [Preorder α] (s : ℕ → α) : Prop :=
  ∀ (i j : ℕ), i < j → s j ≤ s i

def is_increasing {α : Type*} [Preorder α] (s : ℕ → α) : Prop :=
  ∀ (i j : ℕ), i < j → s i ≤ s j

def is_constant {α : Type*} [Preorder α] (s : ℕ → α) : Prop :=
  ∃ c, ∀ i, s i = c

def b_n (a_n : ℕ → ℕ) (n : ℕ) : ℕ :=
  (list.fin_range (n + 1)).map a_n |>.minimum' sorry

theorem true_statements (a_n : ℕ → ℕ) (m : ℕ) :
  (is_decreasing (b_n a_n) → is_decreasing a_n) ∧
  (is_increasing a_n → is_constant (b_n a_n)) :=
by
  sorry

end true_statements_l506_506464


namespace sequences_count_l506_506932

open BigOperators

def consecutive_blocks (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2 - 1) - 2

theorem sequences_count {n : ℕ} (h : n = 15) :
  consecutive_blocks n = 238 :=
by
  sorry

end sequences_count_l506_506932


namespace product_sign_and_units_digit_l506_506705

-- Define the sequence of odd negative integers strictly greater than -2023
def odd_neg_integers_gt_neg2023 :=
  {n : ℤ | n % 2 = 1 ∧ n < 0 ∧ n > -2023}

-- Define the product of all elements in the sequence
noncomputable def product_of_all_odd_neg_integers_gt_neg2023 :=
  ∏ n in odd_neg_integers_gt_neg2023, n

theorem product_sign_and_units_digit :
  ∃ k : ℤ, product_of_all_odd_neg_integers_gt_neg2023 = -5 * k :=
sorry

end product_sign_and_units_digit_l506_506705


namespace original_subscription_cost_l506_506807

-- Define the conditions
def magazine_cut := 0.30
def amount_cut := 658

-- The goal is to prove that the original cost of the yearly subscription is $2193
theorem original_subscription_cost : (amount_cut : ℤ) = (magazine_cut * 2193 : ℝ) := sorry

end original_subscription_cost_l506_506807


namespace find_first_term_l506_506219

noncomputable def first_term : ℝ :=
  let a := 20 * (1 - (2 / 3)) in a

theorem find_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = first_term :=
by
  sorry

end find_first_term_l506_506219


namespace imported_lights_in_sample_l506_506645

theorem imported_lights_in_sample (total_marker_lights imported_marker_lights sample_size : ℕ) 
    (total_marker_lights = 300) 
    (imported_marker_lights = 30) 
    (sample_size = 20) : 
    (imported_marker_lights * sample_size / total_marker_lights) = 2 :=
by sorry

end imported_lights_in_sample_l506_506645


namespace find_x_l506_506383

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l506_506383


namespace prime_factors_of_450_l506_506929

open Nat -- Opening the Nat namespace for natural number operations

/-- Prove that 450 has exactly 3 distinct prime factors -/
theorem prime_factors_of_450 : (Nat.factors 450).nodup ∧ (Nat.factors 450).length = 3 := by
  sorry

end prime_factors_of_450_l506_506929


namespace solvability_condition_l506_506135

def is_solvable (p : ℕ) [Fact (Nat.Prime p)] :=
  ∃ α : ℤ, α * (α - 1) + 3 ≡ 0 [ZMOD p] ↔ ∃ β : ℤ, β * (β - 1) + 25 ≡ 0 [ZMOD p]

theorem solvability_condition (p : ℕ) [Fact (Nat.Prime p)] : 
  is_solvable p :=
sorry

end solvability_condition_l506_506135


namespace find_x_tan_identity_l506_506445

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l506_506445


namespace hyperbola_eccentricity_l506_506005

-- Define the conditions of the problem
variables (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
def hyperbola := {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}
def line_slope_2 := {p : ℝ × ℝ | p.2 = 2 * p.1}

-- Given points A and B are intersections of the hyperbola and the line
variables (A B : ℝ × ℝ)
variable hA : A ∈ hyperbola a b
variable hB : B ∈ hyperbola a b
variable lA : A ∈ line_slope_2
variable lB : B ∈ line_slope_2

-- Midpoint condition
variable midpoint_eq : (A.1 + B.1) / 2 = 3 ∧ (A.2 + B.2) / 2 = 1

-- Define the eccentricity of the hyperbola
noncomputable def c := (sqrt (a^2 + b^2))
noncomputable def eccentricity := c / a

-- Statement to prove
theorem hyperbola_eccentricity : eccentricity a b = (sqrt 15) / 3 :=
by sorry

end hyperbola_eccentricity_l506_506005


namespace find_numbers_on_cards_A_B_C_l506_506155

def card : Type := fin 9

def cards : fin 6 → card :=
  λ i, [⟨0, by linarith⟩, ⟨2, by linarith⟩, ⟨3, by linarith⟩, ⟨5, by linarith⟩, ⟨6, by linarith⟩, ⟨7, by linarith⟩].nth_le i i.is_lt

def not_consecutive (a : card) (b : card) (c : card) : Prop :=
  ¬ ((a.val < b.val ∧ b.val < c.val) ∨ (a.val > b.val ∧ b.val > c.val))

theorem find_numbers_on_cards_A_B_C (A B C : card) (hA : A = ⟨4, by linarith⟩) (hB : B = ⟨1, by linarith⟩) (hC : C = ⟨8, by linarith⟩)
  (hvis : ∀ i j k : fin 3, not_consecutive (cards i) (cards j) (cards k)) :
  A.val = 5 ∧ B.val = 2 ∧ C.val = 9 :=
by sorry

end find_numbers_on_cards_A_B_C_l506_506155


namespace find_first_term_l506_506196

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l506_506196


namespace find_a_for_nonconstant_f_l506_506355

noncomputable def f : ℝ → ℝ := sorry -- We are not defining the function here, just declaring it.

theorem find_a_for_nonconstant_f (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x : ℝ, f(a * x) = a^2 * f(x)) ∧ (∀ x : ℝ, f(f(x)) = a * f(x)) ∧ ¬ ∀ x : ℝ, f(x) = x) → (a = 0 ∨ a = 1) :=
sorry

end find_a_for_nonconstant_f_l506_506355


namespace count_perfect_squares_between_15_and_200_l506_506062

theorem count_perfect_squares_between_15_and_200 : 
  ∃ (n : ℕ), n = ∑ x in (finset.range 15).filter (λ x, 15 < x^2 ∧ x^2 ≤ 200), 1 :=
by
  -- We start by defining the bounds, the smallest integer greater than sqrt(15) and the largest integer less than or equal to sqrt(200)
  have h1 : 4 ≤ ⌈real.sqrt 15⌉₊ := nat.ceil_le.2 (by linarith [real.sqrt_lt'.mpr (by norm_num)]),
  have h2 : ⌊real.sqrt 200⌋₊ ≤ 14 := nat.floor_le.2 (by linarith [real.le_sqrt 200 (by norm_num)]),
  
  -- Define the perfect squares within the range
  let s := (finset.range 15).filter (λ x, (15 < x^2) ∧ (x^2 ≤ 200)),
  
  -- Prove the count is 11
  use s.card,
  sorry

end count_perfect_squares_between_15_and_200_l506_506062


namespace find_x_tan_eq_l506_506434

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l506_506434


namespace distance_between_foci_of_ellipse_l506_506865

theorem distance_between_foci_of_ellipse : 
  let a := 5
  let b := 3
  2 * Real.sqrt (a^2 - b^2) = 8 := by
  let a := 5
  let b := 3
  sorry

end distance_between_foci_of_ellipse_l506_506865


namespace triangle_equilateral_of_angles_and_intersecting_segments_l506_506104

theorem triangle_equilateral_of_angles_and_intersecting_segments
    (A B C : Type) (angle_A : ℝ) (intersect_at_one_point : Prop)
    (angle_M_bisects : Prop) (N_is_median : Prop) (L_is_altitude : Prop) :
  angle_A = 60 ∧ angle_M_bisects ∧ N_is_median ∧ L_is_altitude ∧ intersect_at_one_point → 
  ∀ (angle_B angle_C : ℝ), angle_B = 60 ∧ angle_C = 60 := 
by
  intro h
  sorry

end triangle_equilateral_of_angles_and_intersecting_segments_l506_506104


namespace average_speed_correct_l506_506348

-- Define the total distance and total time based on given conditions
def total_distance := 280 + 360 + 500
def time_leg1 := 2 + (20 / 60 : ℝ)
def time_leg2 := 3 + (10 / 60 : ℝ)
def time_leg3 := 4 + (50 / 60 : ℝ)

def total_time := time_leg1 + time_leg2 + time_leg3

-- Define overall average speed to be proved
def average_speed := total_distance / total_time

theorem average_speed_correct :
  average_speed ≈ 110.3 := 
sorry

end average_speed_correct_l506_506348


namespace radius_of_small_sphere_l506_506981

/-- Definitions: 

1. Unit cube has an inscribed sphere O of radius 1/2.
2. Small sphere at each corner of the cube, tangent to the inscribed sphere and to three faces.

Goals: 
Find the radius of the small sphere.
-/

-- Define the problem domain and parameters
def unitCube : Type := sorry
def inscribedSphere (u : unitCube) : Type := sorry
def smallSphere (u : unitCube) : Type := sorry
def tangent (s1 s2 : Type) : Prop := sorry

-- For unit cube of side length 1
def side_length := 1
def radius_inscribed_sphere := 1 / 2

-- Each small sphere radius: x
variable {x : ℝ}

-- Conditions definitions
def conditions (u : unitCube) 
(insSph : inscribedSphere u) 
(smallSph : smallSphere u) : Prop := 
  ∀ corner, tangent smallSph insSph ∧ tangent smallSph u

-- Stating the theorem quest with equivalence to the original problem
theorem radius_of_small_sphere (u : unitCube)
(insSph : inscribedSphere u)
(smallSph : smallSphere u)
(h : conditions u insSph smallSph) : 
  x = (1 / 2) * (2 * real.sqrt 3 - 3) := 
sorry

end radius_of_small_sphere_l506_506981


namespace base6_div_by_7_l506_506868

theorem base6_div_by_7 (k d : ℕ) (hk : 0 ≤ k ∧ k ≤ 5) (hd : 0 ≤ d ∧ d ≤ 5) (hkd : k = d) : 
  7 ∣ (217 * k + 42 * d) := 
by 
  rw [hkd]
  sorry

end base6_div_by_7_l506_506868


namespace sum_of_constants_l506_506178

theorem sum_of_constants :
  let y (x : ℝ) := (x^3 + 6*x^2 + 11*x + 6) / (x + 1) in
  ∃ (A B C D : ℝ), 
    (∀ (x : ℝ), x ≠ -1 → (y x = A*x^2 + B*x + C)) ∧ 
    A = 1 ∧ B = 5 ∧ C = 6 ∧ D = -1 ∧
    A + B + C + D = 11 :=
by {
  -- Define the given function y
  let y (x : ℝ) := (x^3 + 6*x^2 + 11*x + 6) / (x + 1),
  -- Assign the constants A, B, C, and D
  let A := 1,
  let B := 5,
  let C := 6,
  let D := -1,
  -- Prove that the function equals the quadratic form for x ≠ -1
  have h₁ : ∀ (x : ℝ), x ≠ -1 → (y x = A*x^2 + B*x + C),
  {
    intros x hx,
    -- Polynomial division result
    calc (x^3 + 6*x^2 + 11*x + 6) / (x + 1)
        = (x + 1) * (x^2 + 5*x + 6) / (x + 1) : by sorry
    ... = x^2 + 5*x + 6 : by neo.field_simp [hx],
  },
  -- Construct the tuple with the calculated sum
  use [A, B, C, D],
  split, exact h₁,
  split, refl,
  split, refl,
  split, refl,
  field_simp,
}

end sum_of_constants_l506_506178


namespace find_x_tan_identity_l506_506439

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l506_506439


namespace marcy_sip_amount_l506_506637

theorem marcy_sip_amount (liters : ℕ) (ml_per_liter : ℕ) (total_minutes : ℕ) (interval_minutes : ℕ) (total_ml : ℕ) (total_sips : ℕ) (ml_per_sip : ℕ) 
  (h1 : liters = 2) 
  (h2 : ml_per_liter = 1000)
  (h3 : total_minutes = 250) 
  (h4 : interval_minutes = 5)
  (h5 : total_ml = liters * ml_per_liter)
  (h6 : total_sips = total_minutes / interval_minutes)
  (h7 : ml_per_sip = total_ml / total_sips) : 
  ml_per_sip = 40 := 
by
  sorry

end marcy_sip_amount_l506_506637


namespace tangent_line_at_point_l506_506915

noncomputable def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

def tangent_line_eq (x1 y1 m : ℝ) : ℝ → ℝ := λ x, y1 + m * (x - x1)

theorem tangent_line_at_point (h1 : f 2 = -2) (h2 : deriv f 2 = 1) :
    ∃ (m x1 y1 : ℝ), m = 1 ∧ x1 = 2 ∧ y1 = -2 ∧ 
    (∀ (x y: ℝ), (f x = y) → (y = tangent_line_eq x1 y1 m x) → x - y - 4 = 0) :=
sorry

end tangent_line_at_point_l506_506915


namespace prime_factorization_l506_506338

theorem prime_factorization :
  ∀ (n85 n87 n91 n94 : ℕ),
  n85 = 5 * 17 →
  n87 = 3 * 29 →
  n91 = 7 * 13 →
  n94 = 2 * 47 →
  (85 * 87 * 91 * 94 = 5 * 17 * 3 * 29 * 7 * 13 * 2 * 47) →
  ({5, 17, 3, 29, 7, 13, 2, 47}.size = 8) :=
by
  intros n85 n87 n91 n94 h85 h87 h91 h94 hprod
  sorry

end prime_factorization_l506_506338


namespace reciprocal_proof_l506_506308

theorem reciprocal_proof :
  (-2) * (-(1 / 2)) = 1 := 
by 
  sorry

end reciprocal_proof_l506_506308


namespace num_pos_cubes_ending_in_5_lt_5000_l506_506931

theorem num_pos_cubes_ending_in_5_lt_5000 : 
  (∃ (n1 n2 : ℕ), (n1 ≤ 5000 ∧ n2 ≤ 5000) ∧ (n1^3 % 10 = 5 ∧ n2^3 % 10 = 5) ∧ (n1^3 < 5000 ∧ n2^3 < 5000) ∧ n1 ≠ n2 ∧ 
  ∀ n, (n^3 < 5000 ∧ n^3 % 10 = 5) → (n = n1 ∨ n = n2)) :=
sorry

end num_pos_cubes_ending_in_5_lt_5000_l506_506931


namespace james_distance_ridden_l506_506602

theorem james_distance_ridden : 
  let speed := 16 
  let time := 5 
  speed * time = 80 := 
by
  sorry

end james_distance_ridden_l506_506602


namespace day_crew_fraction_correct_l506_506740

variable (D Wd : ℕ) -- D = number of boxes loaded by each worker on the day crew, Wd = number of workers on the day crew

-- fraction of all boxes loaded by day crew
def fraction_loaded_by_day_crew (D Wd : ℕ) : ℚ :=
  (D * Wd) / (D * Wd + (3 / 4 * D) * (2 / 3 * Wd))

theorem day_crew_fraction_correct (h1 : D > 0) (h2 : Wd > 0) :
  fraction_loaded_by_day_crew D Wd = 2 / 3 := by
  sorry

end day_crew_fraction_correct_l506_506740


namespace num_ordered_pairs_l506_506986

-- Define the relevant conditions and the main theorem
theorem num_ordered_pairs :
  let J := 30
  in let all_pairs := { (c, d) | 1 ≤ c ∧ c < d ∧ d ≤ 9 ∧ 10 * c + d > J}
  in all_pairs.card = 25 :=
by
  sorry

end num_ordered_pairs_l506_506986


namespace find_c_value_l506_506285

theorem find_c_value (c : ℝ) (h : 0.0168 / (0.005 * c) ≈ 840) : c = 0.004 :=
by
  sorry

end find_c_value_l506_506285


namespace composite_integer_unique_solution_l506_506335

theorem composite_integer_unique_solution :
  ∀ (n : ℕ), n > 1 ∧ ∃ (d : list ℕ), d.head = 1 ∧ d.last = n ∧ (∀ i, 1 ≤ i ∧ i < d.length → d.nth i < d.nth (i + 1)) 
  ∧ (((d.nth (i+2) - d.nth (i+1)) / (d.nth (i+1) - d.nth i)) = i + 1) 
  → (∃ (n : ℕ), n = 4) := sorry

end composite_integer_unique_solution_l506_506335


namespace tan_sin_cos_eq_l506_506399

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l506_506399


namespace sequence_general_term_l506_506797

theorem sequence_general_term (a : ℕ → ℝ)
  (h0 : a 0 = 1)
  (h1 : a 1 = 1)
  (h_rec : ∀ n ≥ 2, sqrt (a n * a (n-2)) - sqrt (a (n-1) * a (n-2)) = 2 * a (n-1)) :
  ∀ n ≥ 1, a n = ∏ k in finset.range (n + 1).filter (λ k, k ≠ 0), (2 ^ k - 1) ^ 2 := by
  sorry

end sequence_general_term_l506_506797


namespace min_distance_midpoint_to_C1_l506_506901

noncomputable def rectangular_eq_C1 (x y : ℝ) : Prop :=
  x - 2 * y - 7 = 0

noncomputable def general_eq_C2 (x y : ℝ) : Prop :=
  (x^2) / 64 + (y^2) / 9 = 1

def point_on_C2 (θ : ℝ) : ℝ × ℝ :=
  (8 * cos θ, 3 * sin θ)

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ( (fst P + fst Q) / 2, (snd P + snd Q) / 2 )

noncomputable def distance_from_M_to_C1 (M : ℝ × ℝ) : ℝ :=
  abs (fst M - 2 * snd M - 7) / sqrt 5

theorem min_distance_midpoint_to_C1 :
  let P := (-4, 4)
  ∃ θ : ℝ, 
  let Q := point_on_C2 θ in
  let M := midpoint P Q in
  distance_from_M_to_C1 M = (8 * sqrt 5) / 5 := 
by
  sorry

end min_distance_midpoint_to_C1_l506_506901


namespace slower_speed_is_8_l506_506068

-- Given conditions and definitions
variables (x : ℝ) (t : ℝ) (d1 d2 s1 s2 : ℝ)
  (h1 : d1 = 40)
  (h2 : d2 = 60)
  (s1 = x)
  (s2 = 12)
  (t = d1 / s1)
  (t = d2 / s2)

-- Prove that the slower speed x is 8 km/hr
theorem slower_speed_is_8 (h : (40 : ℝ) / x = 5) : x = 8 :=
by
  sorry

end slower_speed_is_8_l506_506068


namespace circle_equation_range_PA_PB_square_l506_506495

-- Defining the center of the circle lying on the line x - 2y = 0
def center_on_line (a b : ℝ) : Prop := a = 2 * b

-- Defining the function to check if the circle passes through given points
def passes_through (a b r : ℝ) (M N : ℝ × ℝ) : Prop := 
  let M_dist := (M.1 - a)^2 + (M.2 - b)^2 in
  let N_dist := (N.1 - a)^2 + (N.2 - b)^2 in
  M_dist = r^2 ∧ N_dist = r^2

-- The first proof goal
theorem circle_equation (a b r : ℝ) (M N : ℝ × ℝ) (h_line : center_on_line a b) 
  (h_passes : passes_through a b r M N) : (a, b, r) = (4, 2, 5) :=
sorry

-- The second proof goal
theorem range_PA_PB_square (a b r : ℝ) (A B : ℝ × ℝ) (P : ℝ × ℝ) (h_circle : (P.1 - a)^2 + (P.2 - b)^2 = r^2)
  (h_values : A = (1, 1) ∧ B = (7, 4) ∧ a = 4 ∧ b = 2 ∧ r = 5) : 
  ∃ (min max : ℝ), ∀ (P : ℝ × ℝ), (P.1 - a)^2 + (P.2 - b)^2 = r^2 → min ≤ (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧ 
  (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.1 - B.1)^2 + (P.2 - B.2)^2 ≤ max := 
sorry

end circle_equation_range_PA_PB_square_l506_506495


namespace overhead_percentage_l506_506684

def purchase_price : ℝ := 48
def markup : ℝ := 30
def net_profit : ℝ := 12

-- Define the theorem to be proved
theorem overhead_percentage : ((markup - net_profit) / purchase_price) * 100 = 37.5 := by
  sorry

end overhead_percentage_l506_506684


namespace find_first_term_l506_506217

noncomputable def first_term : ℝ :=
  let a := 20 * (1 - (2 / 3)) in a

theorem find_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = first_term :=
by
  sorry

end find_first_term_l506_506217


namespace ice_cream_scoops_l506_506318

def scoops_of_ice_cream : ℕ := 1 -- single cone has 1 scoop

def scoops_double_cone : ℕ := 2 * scoops_of_ice_cream -- double cone has two times the scoops of a single cone

def scoops_banana_split : ℕ := 3 * scoops_of_ice_cream -- banana split has three times the scoops of a single cone

def scoops_waffle_bowl : ℕ := scoops_banana_split + 1 -- waffle bowl has one more scoop than banana split

def total_scoops : ℕ := scoops_of_ice_cream + scoops_double_cone + scoops_banana_split + scoops_waffle_bowl

theorem ice_cream_scoops : total_scoops = 10 :=
by
  sorry

end ice_cream_scoops_l506_506318


namespace find_lambda_perpendicular_l506_506040

-- Defining coordinates of points A, B, and C
def A : ℝ × ℝ × ℝ := (1, 2, 3)
def B : ℝ × ℝ × ℝ := (2, -1, 1)
def C (λ : ℝ) : ℝ × ℝ × ℝ := (3, λ, λ)

-- Defining the vectors AB and AC
def AB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def AC (λ : ℝ) : ℝ × ℝ × ℝ := (C λ).1 - A.1, (C λ).2 - A.2, (C λ).3 - A.3

-- Define dot product
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Problem statement: Find λ such that AB is perpendicular to AC
theorem find_lambda_perpendicular : 
  ∃ (λ : ℝ), dot_product AB (AC λ) = 0 ∧ λ = 14 / 5 := by
sorry

end find_lambda_perpendicular_l506_506040


namespace g_is_even_l506_506595

def g (x : ℝ) : ℝ := 2 ^ (x ^ 2 - 1) - x ^ 2

theorem g_is_even : ∀ x : ℝ, g (-x) = g x :=
by
  intros x
  sorry

end g_is_even_l506_506595


namespace area_of_OAB_l506_506483

open Real EuclideanGeometry

def dist (p1 p2 : EuclideanSpace ℝ (Fin 3)) : ℝ := dist p1 p2

theorem area_of_OAB {P A B C D O : EuclideanSpace ℝ (Fin 3)} 
  (hPA_perp : P ∈ vector_span ℝ ({A, B, C, D} : set (EuclideanSpace ℝ (Fin 3))) ∧ P \notin convex_hull ℝ ({A, B, C, D} : set (EuclideanSpace ℝ (Fin 3))))
  (hSquare : dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧ dist A D = 2 * sqrt 3)
  (hP_radius : dist P A = 2 * sqrt 6)
  (h_on_surface : ∀ (X : set (EuclideanSpace ℝ (Fin 3))), X = ({P, A, B, C, D} : set (EuclideanSpace ℝ (Fin 3))) → ∃ Z, is_circumsphere Z X (R O)) :
  ∃ R, area_of_triangle O A B = 3 * sqrt 3 :=
by
  sorry

end area_of_OAB_l506_506483


namespace monotonic_intervals_of_f_l506_506515

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 - x

-- Define the derivative f'
noncomputable def f' (x : ℝ) : ℝ := Real.exp x - 1

-- Prove the monotonicity intervals of the function f
theorem monotonic_intervals_of_f :
  (∀ x : ℝ, x < 0 → f' x < 0) ∧ (∀ x : ℝ, 0 < x → f' x > 0) :=
by
  sorry

end monotonic_intervals_of_f_l506_506515


namespace toilet_paper_supply_last_days_l506_506825

theorem toilet_paper_supply_last_days :
  ∀ (num_rolls : ℕ) (squares_per_roll : ℕ)
  (bill_visits_per_day : ℕ) (bill_squares_per_visit : ℕ)
  (wife_visits_per_day : ℕ) (wife_squares_per_visit : ℕ)
  (kids_count : ℕ) (kid_visits_per_day : ℕ) (kid_squares_per_visit : ℕ),
  num_rolls = 1000 →
  squares_per_roll = 300 →
  bill_visits_per_day = 3 →
  bill_squares_per_visit = 5 →
  wife_visits_per_day = 4 →
  wife_squares_per_visit = 8 →
  kids_count = 2 →
  kid_visits_per_day = 5 →
  kid_squares_per_visit = 6 →
  let total_squares := num_rolls * squares_per_roll in
  let daily_usage := bill_visits_per_day * bill_squares_per_visit +
                     wife_visits_per_day * wife_squares_per_visit +
                     kids_count * kid_visits_per_day * kid_squares_per_visit in
  total_squares / daily_usage = 2803 := by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _,
  sorry

end toilet_paper_supply_last_days_l506_506825


namespace main_theorem_l506_506621

variable (f : ℝ → ℝ)

-- Conditions: f(x) > f'(x) for all x ∈ ℝ
def condition (x : ℝ) : Prop := f x > (derivative f) x

-- Main statement to prove
theorem main_theorem  (h : ∀ x : ℝ, condition f x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) := 
by 
  sorry

end main_theorem_l506_506621


namespace find_x_l506_506386

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l506_506386


namespace find_a1_l506_506904

theorem find_a1 (a_1 : ℕ) (S : ℕ → ℕ) (S_formula : ∀ n : ℕ, S n = (a_1 * (3^n - 1)) / 2)
  (a_4_eq : (S 4) - (S 3) = 54) : a_1 = 2 :=
  sorry

end find_a1_l506_506904


namespace complete_work_days_correct_l506_506263

-- Definitions of conditions
def A_work_days : ℝ := 20
def B_work_days : ℝ := 35

-- Definition of work rates
def A_work_rate : ℝ := 1 / A_work_days
def B_work_rate : ℝ := 1 / B_work_days

-- Combined work rate
def combined_work_rate : ℝ := A_work_rate + B_work_rate

-- Days to complete the work together
def complete_work_days : ℝ := 1 / combined_work_rate

-- Theorem to prove
theorem complete_work_days_correct : complete_work_days = 140 / 11 :=
by
  -- Proving by assertion that the combined work rate leads to 140/11 days
  sorry

end complete_work_days_correct_l506_506263


namespace vector_decomposition_exists_l506_506850

noncomputable theory

-- Define the vectors
def vec_x : ℝ × ℝ × ℝ := (3, -1, 2)
def vec_p : ℝ × ℝ × ℝ := (2, 0, 1)
def vec_q : ℝ × ℝ × ℝ := (1, -1, 1)
def vec_r : ℝ × ℝ × ℝ := (1, -1, -2)

-- Define the theorem to prove the decomposition exists with specific coefficients
theorem vector_decomposition_exists : 
  ∃ (α β γ : ℝ), (α, β, γ) = (1, 1, 0) ∧ vec_x = 
    (α * vec_p.1 + β * vec_q.1 + γ * vec_r.1,
     α * vec_p.2 + β * vec_q.2 + γ * vec_r.2,
     α * vec_p.3 + β * vec_q.3 + γ * vec_r.3) :=
sorry

end vector_decomposition_exists_l506_506850


namespace envelope_width_l506_506816

theorem envelope_width (Area Height Width : ℝ) (h_area : Area = 36) (h_height : Height = 6) (h_area_formula : Area = Width * Height) : Width = 6 :=
by
  sorry

end envelope_width_l506_506816


namespace count_greater_or_equal_l506_506327

theorem count_greater_or_equal (l : List ℚ) (threshold : ℚ) (count : ℕ) : 
  l = [1.4, 9/10, 1.2, 0.5, 13/10] → threshold = 1.1 → count = 3 → 
  (l.filter (λ x, x ≥ threshold)).length = count := 
  by intros; subst_vars; sorry

end count_greater_or_equal_l506_506327


namespace two_non_congruent_triangles_with_perimeter_7_l506_506930

theorem two_non_congruent_triangles_with_perimeter_7 :
  ∃ (sides : Finset (Finset (ℕ × ℕ × ℕ))), 
  (∀ s ∈ sides, let (a, b, c) := (s.1, s.2, s.3) in 
    a + b + c = 7 ∧ 
    a + b > c ∧ 
    a + c > b ∧ 
    b + c > a ∧ 
    a ≤ b ∧ b ≤ c
  ) ∧ sides.card = 2 :=
sorry

end two_non_congruent_triangles_with_perimeter_7_l506_506930


namespace find_x_l506_506376

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l506_506376


namespace pentagon_area_is_correct_l506_506933

noncomputable def area_of_pentagon : ℕ :=
  let area_trapezoid := (1 / 2) * (25 + 28) * 30
  let area_triangle := (1 / 2) * 18 * 24
  area_trapezoid + area_triangle

theorem pentagon_area_is_correct (s1 s2 s3 s4 s5 : ℕ) (b1 b2 h1 b3 h2 : ℕ)
  (h₀ : s1 = 18) (h₁ : s2 = 25) (h₂ : s3 = 30) (h₃ : s4 = 28) (h₄ : s5 = 25)
  (h₅ : b1 = 25) (h₆ : b2 = 28) (h₇ : h1 = 30) (h₈ : b3 = 18) (h₉ : h2 = 24) :
  area_of_pentagon = 1011 := by
  -- placeholder for actual proof
  sorry

end pentagon_area_is_correct_l506_506933


namespace find_x_between_0_and_180_l506_506421

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l506_506421


namespace max_range_of_walk_min_range_of_walk_count_of_max_range_sequences_l506_506756

section RandomWalk

variables {a b : ℕ} (h : a > b)

def max_range_walk : ℕ := a
def min_range_walk : ℕ := a - b
def count_max_range_sequences : ℕ := b + 1

theorem max_range_of_walk (h : a > b) : max_range_walk h = a :=
by
  sorry

theorem min_range_of_walk (h : a > b) : min_range_walk h = a - b :=
by
  sorry

theorem count_of_max_range_sequences (h : a > b) : count_max_range_sequences h = b + 1 :=
by
  sorry

end RandomWalk

end max_range_of_walk_min_range_of_walk_count_of_max_range_sequences_l506_506756


namespace max_range_walk_min_range_walk_count_max_range_sequences_l506_506752

variable {a b : ℕ}

-- Condition: a > b
def valid_walk (a b : ℕ) : Prop := a > b

-- Proof that the maximum possible range of the walk is a
theorem max_range_walk (h : valid_walk a b) : 
  (a + b) = a + b := sorry

-- Proof that the minimum possible range of the walk is a - b
theorem min_range_walk (h : valid_walk a b) : 
  (a - b) = a - b := sorry

-- Proof that the number of different sequences with the maximum possible range is b + 1
theorem count_max_range_sequences (h : valid_walk a b) : 
  b + 1 = b + 1 := sorry

end max_range_walk_min_range_walk_count_max_range_sequences_l506_506752


namespace find_x_value_l506_506410

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l506_506410


namespace distance_between_foci_l506_506361

theorem distance_between_foci (x y : ℝ) :
  9 * x^2 + 16 * y^2 = 144 → 2 * Real.sqrt 7 :=
by
  intro h
  have el_eq : (x^2 / 16) + (y^2 / 9) = 1 := by
    sorry
  have a : ℝ := 4
  have b : ℝ := 3
  have c : ℝ := Real.sqrt (a^2 - b^2)
  have foc_dist : ℝ := 2 * c
  exact foc_dist

end distance_between_foci_l506_506361


namespace num_prime_factors_450_l506_506926

def is_prime (n : ℕ) : Prop := ¬∃m, m > 1 ∧ m < n ∧ m ∣ n

def prime_factors (n : ℕ) : set ℕ := {p | is_prime p ∧ p ∣ n}

theorem num_prime_factors_450 : finset.card (finset.filter (λ p, is_prime p) (finset.filter (λ p, p ∣ 450) finset.range 451)) = 3 := 
sorry

end num_prime_factors_450_l506_506926


namespace equation_of_line_l506_506269

variables {m n : ℝ} (p : ℝ)

-- Theorem statement: proving the equation of the line
theorem equation_of_line (h1 : p = 3) (h2 : ∃ m n : ℝ, True) :
  ∃ (b : ℝ), b = n - 3 * m ∧ ∀ (x y : ℝ), y = 3 * x + b :=
begin
  -- Since we need to prove the existence of b, we state its expression
  use (n - 3 * m),
  -- Then we show our assumptions imply this particular b
  split,
  { refl, },
  -- Lastly, we state the equation of the line using y = 3x + b
  intros x y,
  exact ⟨rfl⟩,
end

end equation_of_line_l506_506269


namespace man_speed_l506_506768

noncomputable def train_length : ℝ := 900  -- in meters
noncomputable def train_speed_km_h : ℝ := 63  -- in km/hr
noncomputable def crossing_time : ℝ := 53.99568034557235  -- in seconds

theorem man_speed :
  let train_speed_m_s := train_speed_km_h * (1000 / 3600) in
  let relative_speed := train_length / crossing_time in
  let man_speed_m_s := train_speed_m_s - relative_speed in
  let man_speed_km_h := man_speed_m_s * (3600 / 1000) in
  abs (man_speed_km_h - 2.9952) < 1e-4 := 
by
  sorry

end man_speed_l506_506768


namespace value_of_expression_l506_506936

theorem value_of_expression (x y : ℝ) (h₁ : x = 3) (h₂ : y = 4) :
  (x^3 + 3*y^3) / 9 = 24.33 :=
by
  sorry

end value_of_expression_l506_506936


namespace cost_of_five_juices_l506_506603

-- Given conditions as assumptions
variables {J S : ℝ}

axiom h1 : 2 * S = 6
axiom h2 : S + J = 5

-- Prove the statement
theorem cost_of_five_juices : 5 * J = 10 :=
sorry

end cost_of_five_juices_l506_506603


namespace range_of_a_l506_506072

theorem range_of_a (a : ℝ) (h : a ≥ 0) :
  ∃ a, (2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 * Real.sqrt 2) ↔
  (∀ x y : ℝ, 
    ((x - a)^2 + y^2 = 1) ∧ (x^2 + (y - 2)^2 = 25)) :=
sorry

end range_of_a_l506_506072


namespace tenth_term_ar_sequence_l506_506014

-- Variables for the first term and common difference
variables (a1 d : ℕ) (n : ℕ)

-- Specific given values
def a1_fixed := 3
def d_fixed := 2

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) := a1 + (n - 1) * d

-- The statement to prove
theorem tenth_term_ar_sequence : a_n 10 = 21 := by
  -- Definitions for a1 and d
  let a1 := a1_fixed
  let d := d_fixed
  -- The rest of the proof
  sorry

end tenth_term_ar_sequence_l506_506014


namespace number_equals_fifty_l506_506237

def thirty_percent_less_than_ninety : ℝ := 0.7 * 90

theorem number_equals_fifty (x : ℝ) (h : (5 / 4) * x = thirty_percent_less_than_ninety) : x = 50 :=
by
  sorry

end number_equals_fifty_l506_506237


namespace proof_m_n_arithmetic_square_root_l506_506499

theorem proof_m_n_arithmetic_square_root (m n : ℤ) 
  (h1 : sqrt (m + 3) = 1 ∨ sqrt (m + 3) = -1)
  (h2 : real.cbrt (3 * m + 2 * n - 6) = 4) :
  m = -2 ∧ n = 38 ∧ sqrt (m + n) = 6 :=
by
  sorry

end proof_m_n_arithmetic_square_root_l506_506499


namespace find_sum_a7_a8_l506_506085

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 q : ℝ), ∀ n : ℕ, a n = a1 * q ^ n

variable (a : ℕ → ℝ)

axiom h_geom : geometric_sequence a
axiom h1 : a 0 + a 1 = 16
axiom h2 : a 2 + a 3 = 32

theorem find_sum_a7_a8 : a 6 + a 7 = 128 :=
sorry

end find_sum_a7_a8_l506_506085


namespace hyperbola_parameters_l506_506083

noncomputable def h : ℝ := 3
noncomputable def k : ℝ := 0

noncomputable def a : ℝ := 4
noncomputable def c : ℝ := 7
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

theorem hyperbola_parameters : h + k + a + b = 7 + Real.sqrt 33 := by
  have center_eq : (h, k) = (3, 0) := by rfl
  have vertex_eq : (3, -4) * (1 : ℝ) = (3, -4)
  have focus_eq : (3, 7) * (1 : ℝ) = (3, 7)
  sorry

end hyperbola_parameters_l506_506083


namespace find_x_between_0_and_180_l506_506417

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l506_506417


namespace polygon_tricolor_triangles_l506_506098

inductive Color
| red
| blue
| green

structure Polygon (n : ℕ) :=
  (vertices : Fin n → Color)
  (no_consecutive_same_color : ∀ i : Fin n, vertices i ≠ vertices (i + 1) % Fin n)
  (at_least_one_each_color : ∃ r b g : Fin n, vertices r = Color.red ∧ vertices b = Color.blue ∧ vertices g = Color.green)

theorem polygon_tricolor_triangles {n : ℕ} (h : n ≥ 3) (P : Polygon n) :
  ∃ T : Finset (Fin n × Fin n), 
  (∀ t ∈ T, let ⟨i, j⟩ := t in i ≠ j) ∧
  (∃ (triangles : Finset (Fin 3 × Fin n)), 
     (∀ tri ∈ triangles, let ⟨⟨a, b, c⟩, ⟨i, j, k⟩⟩ := tri in a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧
     ∀ ⦃x y z : Fin n⦄, 
        {x, y, z}.card = 3 → 
        vertices x ≠ vertices y ∧ vertices y ≠ vertices z ∧ vertices z ≠ vertices x) := 
sorry

end polygon_tricolor_triangles_l506_506098


namespace exists_positive_int_n_l506_506623

theorem exists_positive_int_n (p a k : ℕ) 
  (hp : Nat.Prime p) (ha : 0 < a) (hk1 : p^a < k) (hk2 : k < 2 * p^a) :
  ∃ n : ℕ, n < p^(2 * a) ∧ (Nat.choose n k ≡ n [MOD p^a]) ∧ (n ≡ k [MOD p^a]) :=
sorry

end exists_positive_int_n_l506_506623


namespace first_term_of_geometric_series_l506_506224

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l506_506224


namespace find_K_l506_506997

theorem find_K (Z K : ℕ) (hZ1 : 1000 < Z) (hZ2 : Z < 8000) (hK : Z = K^3) : 11 ≤ K ∧ K ≤ 19 :=
sorry

end find_K_l506_506997


namespace souvenir_cost_and_people_l506_506289

-- Definitions for number of members and total cost
variables (num_people cost : ℕ)

-- Definitions of given conditions
def initial_conditions :=
  ∃ num_present num_absent : ℕ,
    num_absent = 2 ∧
    let x := num_people - num_absent in
    let initial_payment := cost / num_people in
    let increased_payment := initial_payment + 18 in
    let reduced_payment := increased_payment - 10 in
    let new_people := num_present + 1 in
    (increased_payment = (cost + 18 * num_absent) / num_present) ∧
    (reduced_payment = cost / new_people)

-- Proposition to prove
theorem souvenir_cost_and_people : initial_conditions num_people cost →
  num_people = 10 ∧ cost = 720 :=
by
  -- Proof to be filled
  sorry

end souvenir_cost_and_people_l506_506289


namespace triangle_distance_bisectors_l506_506194

noncomputable def distance_between_bisectors {a b c : ℝ} (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) : ℝ :=
  (2 * a * b * c) / (b^2 - c^2)

theorem triangle_distance_bisectors 
  (a b c : ℝ) (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) :
  ∀ (DD₁ : ℝ), 
  DD₁ = distance_between_bisectors h₁ h₂ h₃ → 
  DD₁ = (2 * a * b * c) / (b^2 - c^2) := by 
  sorry

end triangle_distance_bisectors_l506_506194


namespace arithmetic_mean_is_correct_l506_506706

open Nat

noncomputable def arithmetic_mean_of_two_digit_multiples_of_9 : ℝ :=
  let smallest := 18
  let largest := 99
  let n := 10
  let sum := (n / 2 : ℝ) * (smallest + largest)
  sum / n

theorem arithmetic_mean_is_correct :
  arithmetic_mean_of_two_digit_multiples_of_9 = 58.5 :=
by
  -- Proof goes here
  sorry

end arithmetic_mean_is_correct_l506_506706


namespace total_copies_sold_l506_506235

theorem total_copies_sold
  (hardback_before : ℕ := 36000)
  (paperback_to_hardback_ratio : ℕ := 9)
  (total_paperback : ℕ := 363600)
  (hardback_after : ℕ := total_paperback / paperback_to_hardback_ratio) :
  hardback_before + hardback_after + total_paperback = 440400 :=
by {
  -- Given the number of hardback copies sold before paperback
  have H_before := hardback_before,
  
  -- Given the ratio of paperback to hardback copies and total paperbacks sold
  have R := paperback_to_hardback_ratio,
  have P := total_paperback,
  
  -- Calculate the number of hardback copies sold after paperback
  have H_after := P / R,
  
  -- Calculate the total number of copies sold
  have total := H_before + H_after + P,
  
  -- Show that the total is 440400
  show total = 440400,
  sorry
}

end total_copies_sold_l506_506235


namespace angle_condition_triangle_l506_506982

theorem angle_condition_triangle 
(triangle_ABC : Type) -- Assuming the type for the triangle
[is_triangle triangle_ABC] 
(B B1 C C1 : triangle_ABC) -- Points in the triangle
(is_bisector_BB1 : is_bisector B B1) 
(is_bisector_CC1 : is_bisector C C1)
(angle_CC1_B1_30 : angle C C1 B1 = 30) 
: angle A = 60 ∨ angle B = 120 := 
sorry

end angle_condition_triangle_l506_506982


namespace maximal_value_of_product_l506_506909

theorem maximal_value_of_product (m n : ℤ)
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (1 < x1 ∧ x1 < 3) ∧ (1 < x2 ∧ x2 < 3) ∧ 
    ∀ x : ℝ, (10 * x^2 + m * x + n) = 10 * (x - x1) * (x - x2)) :
  (∃ f1 f3 : ℝ, f1 = 10 * (1 - x1) * (1 - x2) ∧ f3 = 10 * (3 - x1) * (3 - x2) ∧ (f1 * f3 = 99)) := 
sorry

end maximal_value_of_product_l506_506909


namespace max_volume_tank_l506_506302

theorem max_volume_tank (a b h : ℝ) (ha : a ≤ 1.5) (hb : b ≤ 1.5) (hh : h = 1.5) :
  a * b * h ≤ 3.375 :=
by {
  sorry
}

end max_volume_tank_l506_506302


namespace solve_tan_equation_l506_506374

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l506_506374


namespace find_x_tan_identity_l506_506437

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l506_506437


namespace problem1_problem2_l506_506656

-- Problem 1
theorem problem1 (a : ℝ) (h : a = Real.sqrt 3 - 1) : (a^2 + a) * (a + 1) / a = 3 := 
sorry

-- Problem 2
theorem problem2 (a : ℝ) (h : a = 1 / 2) : (a + 1) / (a^2 - 1) - (a + 1) / (1 - a) = -5 := 
sorry

end problem1_problem2_l506_506656


namespace cartesian_and_polar_eqs_of_l_and_C_distance_PQ_of_intersection_l506_506099

-- Define the parametric equations of the line l
def line_l (t : ℝ) := (x = (√2 / 2) * t, y = (√2 / 2) * t)

-- Define the equation of the circle C
def circle_C := x ^ 2 + y ^ 2 - 4 * x - 2 * y + 4 = 0

-- State the problem of Cartesian and polar equations of line l and polar equation of circle C
theorem cartesian_and_polar_eqs_of_l_and_C :
  (∃ x, ∃ y, y = x ∧ ∃ θ, ∃ ρ, ρ = √2 * (cos θ + sin θ)) ∧
  (∃ ρ, ρ ^ 2 - 4 * ρ * cos θ - 2 * ρ * sin θ + 4 = 0) :=
sorry

-- State the distance between points P and Q on intersection of line l and circle C is √2
theorem distance_PQ_of_intersection (t1 t2 : ℝ)
  (h_intersect : line_l t1 ∈ circle_C ∧ line_l t2 ∈ circle_C) :
  |t1 - t2| = √2 :=
sorry

end cartesian_and_polar_eqs_of_l_and_C_distance_PQ_of_intersection_l506_506099


namespace bela_wins_if_and_only_if_n_gt_8_l506_506315

theorem bela_wins_if_and_only_if_n_gt_8 (n : ℤ) (h1 : n > 6) :
  (∀ b j : ℝ, (b ∈ set.Icc (0 : ℝ) n) ∧ (j ∈ set.Icc (0 : ℝ) n) →
  (∀ t1 t2 : list ℝ, ∀ x : ℝ, x ∈ t1 ∪ t2 → dist x b > 2) →
  ∃ b' : ℝ, b' ∈ set.Icc (0 : ℝ) n ∧ (∀ x : ℝ, x ∈ t1 ∪ t2 → dist x b' > 2) ∧ ∃ j' : ℝ, j' ∈ set.Icc (0 : ℝ) n ∧ (∀ x : ℝ, x ∈ t1 ∪ t2 → dist x j' > 2)) ↔ (n > 8) :=
begin
  sorry
end

end bela_wins_if_and_only_if_n_gt_8_l506_506315


namespace bisects_angle_tangent_circumcircle_l506_506919

open Lean

-- Definition of the parabola y² = 2px
def parabola (p : ℝ) (y x : ℝ) := y^2 = 2 * p * x

-- Focus of the parabola
def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- Definition of an external point P
def external_point (P : ℝ × ℝ) : Prop := P.2 ≠ 0

-- Given conditions: points A, B on the parabola, points C, D on the y-axis
variables (p : ℝ) (P A B C D : ℝ × ℝ)

-- A and B are points on the parabola
def on_parabola (p : ℝ) (point : ℝ × ℝ) : Prop := parabola p point.2 point.1

-- C and D are points on the y-axis
def on_y_axis (point : ℝ × ℝ) : Prop := point.1 = 0

-- Definition of the circumcenter M of triangle PAB
def circumcenter (P A B : ℝ × ℝ) : ℝ × ℝ := 
  -- definition of circumcenter calculation
  sorry

-- Prove that PF bisects ∠AFB
theorem bisects_angle (hparabola : on_parabola p A) (hparabola' : on_parabola p B)
  (hfocus : focus p = (p / 2, 0)) (hexternal : external_point P) :
  sorry -- PF bisects ∠AFB
  sorry

-- Prove that FM is tangent to the circumcircle of △FCD
theorem tangent_circumcircle (M : ℝ × ℝ) (hfocus : focus p = (p / 2, 0)) 
  (hparabola : on_parabola p A) (hparabola' : on_parabola p B) 
  (hon_y_c : on_y_axis C) (hon_y_d : on_y_axis D) :
  sorry -- FM is a tangent to the circumcircle of △FCD
  sorry

end bisects_angle_tangent_circumcircle_l506_506919


namespace intersection_points_l506_506086

/-- 
  Prove that the number of intersection points of 10 non-parallel lines in a plane,
  where exactly two lines pass through each intersection point, is equal to 45.
-/
theorem intersection_points (n : ℕ) (h1 : n = 10) : 
  (nat.choose n 2) = 45 :=
by 
  -- substituting 10 for n
  have h2 : n = 10 := h1
  -- calculation for combination 10 choose 2
  sorry

end intersection_points_l506_506086


namespace candidate_percentage_valid_votes_l506_506573

theorem candidate_percentage_valid_votes (total_votes invalid_percentage valid_votes_received : ℕ) 
    (h_total_votes : total_votes = 560000)
    (h_invalid_percentage : invalid_percentage = 15)
    (h_valid_votes_received : valid_votes_received = 333200) :
    (valid_votes_received : ℚ) / (total_votes * (1 - invalid_percentage / 100) : ℚ) * 100 = 70 :=
by
  sorry

end candidate_percentage_valid_votes_l506_506573


namespace sum_of_non_domain_elements_l506_506858

theorem sum_of_non_domain_elements :
    let f (x : ℝ) : ℝ := 1 / (1 + 1 / (1 + 1 / (1 + 1 / x)))
    let is_not_in_domain (x : ℝ) := x = 0 ∨ x = -1 ∨ x = -1/2 ∨ x = -2/3
    (0 : ℝ) + (-1) + (-1/2) + (-2/3) = -19/6 :=
by 
  sorry

end sum_of_non_domain_elements_l506_506858


namespace find_x_tan_identity_l506_506442

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l506_506442


namespace locus_of_circumcircle_centers_locus_of_centroids_locus_of_orthocenters_l506_506612

noncomputable section

-- Define the main components and structures
variable 
  (S1 S2 : Circle ℝ) -- Two given circles
  (A : Point) -- Intersection point of S1 and S2
  (B : Point)
  (C : Point)
  [h_inter : ∃ P : Point, P ∈ S1 ∧ P ∈ S2 ∧ P ≠ A]
  [h_line : ∃ L : Line, B ∈ L ∧ C ∈ L ∧ L ∩ S1 ≠ ∅ ∧ L ∩ S2 ≠ ∅]

-- Define the statements to be proven
theorem locus_of_circumcircle_centers : 
  ∀ (A B C : Point), A ∈ S1 ∧ A ∈ S2 → 
  ∃ K : Circle ℝ, ∀ O : Point, circumcenter_triangle A B C = O → O ∈ K := 
sorry

theorem locus_of_centroids : 
  ∀ (A B C : Point), A ∈ S1 ∧ A ∈ S2 →
  ∃ K : Circle ℝ, ∀ G : Point, centroid_triangle A B C = G → G ∈ K := 
sorry

theorem locus_of_orthocenters : 
  ∀ (A B C : Point), A ∈ S1 ∧ A ∈ S2 →
  ∃ K : Circle ℝ, ∀ H : Point, orthocenter_triangle A B C = H → H ∈ K := 
sorry

end locus_of_circumcircle_centers_locus_of_centroids_locus_of_orthocenters_l506_506612


namespace functional_equality_l506_506162

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equality
  (h1 : ∀ x : ℝ, f x ≤ x)
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end functional_equality_l506_506162


namespace number_of_years_passed_l506_506352

noncomputable def years_passed (A F : ℝ) (r : ℝ) : ℝ :=
  real.log (F / A) / real.log (1 + r)

theorem number_of_years_passed:
  years_passed 64000 79012.34567901235 (1/9) ≈ 2 := sorry

end number_of_years_passed_l506_506352


namespace find_x_l506_506377

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l506_506377


namespace maximum_range_of_walk_minimum_range_of_walk_number_of_max_range_sequences_l506_506745

theorem maximum_range_of_walk (a b : ℕ) (h : a > b) : 
  (∃ max_range : ℕ, max_range = a) :=
by {
  use a,
  sorry
}

theorem minimum_range_of_walk (a b : ℕ) (h : a > b) : 
  (∃ min_range : ℕ, min_range = a - b) :=
by {
  use a - b,
  sorry
}

theorem number_of_max_range_sequences (a b : ℕ) (h : a > b) : 
  (∃ num_sequences : ℕ, num_sequences = b + 1) :=
by {
  use b + 1,
  sorry
}

end maximum_range_of_walk_minimum_range_of_walk_number_of_max_range_sequences_l506_506745


namespace intersection_A_B_l506_506523

open Set

def A : Set ℝ := {x | x ^ 2 - x - 2 ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {-1, 0, 1, 2} :=
sorry

end intersection_A_B_l506_506523


namespace printer_paper_last_days_l506_506169

def packs : Nat := 2
def sheets_per_pack : Nat := 240
def prints_per_day : Nat := 80
def total_sheets : Nat := packs * sheets_per_pack
def number_of_days : Nat := total_sheets / prints_per_day

theorem printer_paper_last_days :
  number_of_days = 6 :=
by
  sorry

end printer_paper_last_days_l506_506169


namespace min_value_fraction_l506_506894

theorem min_value_fraction (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + 2 * y = 3) : 
  (∃ t, t = (1 / (x - y) + 9 / (x + 5 * y)) ∧ t = 8 / 3) :=
by 
  sorry

end min_value_fraction_l506_506894


namespace train_length_is_correct_l506_506806

variable (speed_km_hr : ℕ) (time_sec : ℕ)
def convert_speed (speed_km_hr : ℕ) : ℚ :=
  (speed_km_hr * 1000 : ℚ) / 3600

noncomputable def length_of_train (speed_km_hr time_sec : ℕ) : ℚ :=
  convert_speed speed_km_hr * time_sec

theorem train_length_is_correct (speed_km_hr : ℕ) (time_sec : ℕ) (h₁ : speed_km_hr = 300) (h₂ : time_sec = 33) :
  length_of_train speed_km_hr time_sec = 2750 := by
  sorry

end train_length_is_correct_l506_506806


namespace sequence_squares_iff_l506_506610

def seq (m : ℤ) : ℕ → ℤ
| 1     := 1
| 2     := 1
| 3     := 4
| (n + 1) := if n ≥ 3 then m * (seq m n + seq m (n-1)) - seq m (n-2) else 1

theorem sequence_squares_iff (m : ℤ) (h : 1 < m) : 
  (∀ n, ∃ k, seq m n = k * k) ↔ (m = 2 ∨ m = 10) :=
by
  sorry

end sequence_squares_iff_l506_506610


namespace max_angle_P_x_l506_506968

-- Point M and Point N are given as conditions.
def M : ℝ×ℝ := (-1, 2)
def N : ℝ×ℝ := (1, 4)

-- The point P moves along the x-axis.
def P (x : ℝ) : ℝ×ℝ := (x, 0)

-- We want to prove that when ∠MPN is maximized, the x-coordinate is 1.
theorem max_angle_P_x :
  ∃ (x : ℝ), x = 1 ∧ ∀ y : ℝ, y ≠ x → ∠(M, P x, N) ≥ ∠(M, P y, N) := sorry

end max_angle_P_x_l506_506968


namespace intersection_of_sets_l506_506631

theorem intersection_of_sets (a b : ℝ) :
  let P := {3, Real.log2 a}
  let Q := {a, b}
  let union_PQ := {0, 1, 3}
  (P ∪ Q = union_PQ) → (P ∩ Q = {0} ∨ P ∩ Q = {3}) :=
by
  intros P Q union_PQ h
  sorry

end intersection_of_sets_l506_506631


namespace product_plus_one_is_square_l506_506943

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : x * y + 1 = (x + 1) ^ 2 :=
by
  sorry

end product_plus_one_is_square_l506_506943


namespace measure_of_angle_A_length_of_side_c_l506_506593

-- Problem Definition
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Sides opposite angles
variables (a b c : ℝ)

-- Angles
variables (θ : ℝ) (angle_A : ℝ) (angle_ABC : ℝ)

-- Conditions
constant h1 : (Real.sin angle_A + Real.sin θ) * (a - b) = c * (Real.sin (π - angle_ABC - angle_A) - √3 * Real.sin θ)
constant h2 : Real.cos angle_ABC = -1/7
constant BD : ℝ := 7 * Real.sqrt 7 / 3

-- Part (1): Prove the measure of angle A
theorem measure_of_angle_A : angle_A = π / 6 :=
  by
    sorry

-- Part (2): Prove the length of side c
theorem length_of_side_c (sin_angle_BDA cos_angle_BDA : ℝ) :
  BD * sin_angle_BDA / (1/2) = 7 * Real.sqrt 3 :=
  by
    sorry

end measure_of_angle_A_length_of_side_c_l506_506593


namespace tan_sin_cos_eq_l506_506455

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l506_506455


namespace maximum_range_of_walk_minimum_range_of_walk_number_of_max_range_sequences_l506_506744

theorem maximum_range_of_walk (a b : ℕ) (h : a > b) : 
  (∃ max_range : ℕ, max_range = a) :=
by {
  use a,
  sorry
}

theorem minimum_range_of_walk (a b : ℕ) (h : a > b) : 
  (∃ min_range : ℕ, min_range = a - b) :=
by {
  use a - b,
  sorry
}

theorem number_of_max_range_sequences (a b : ℕ) (h : a > b) : 
  (∃ num_sequences : ℕ, num_sequences = b + 1) :=
by {
  use b + 1,
  sorry
}

end maximum_range_of_walk_minimum_range_of_walk_number_of_max_range_sequences_l506_506744


namespace smallest_a1_l506_506127

noncomputable def is_sequence (a : ℕ → ℝ) : Prop :=
∀ n > 1, a n = 7 * a (n - 1) - 2 * n

noncomputable def is_positive_sequence (a : ℕ → ℝ) : Prop :=
∀ n > 0, a n > 0

theorem smallest_a1 (a : ℕ → ℝ)
  (h_seq : is_sequence a)
  (h_pos : is_positive_sequence a) :
  a 1 ≥ 13 / 18 :=
sorry

end smallest_a1_l506_506127


namespace range_of_a_l506_506880

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y ∈ set.Ioi (-1) ∩ set.Iio 1, x < y → f x > f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h_decreasing: is_decreasing f) (h_domain: ∀ x, x ∈ set.Ioi (-1) ∩ set.Iio 1 → x ∈ set.Ioi (-1) ∩ set.Iio 1) :
  (f (1 + a) < f 0 ↔ a ∈ set.Iio 0 ∩ set.Ioi (-1)) :=
begin
  sorry
end

end range_of_a_l506_506880


namespace number_of_chinese_l506_506767

theorem number_of_chinese :
  ∃ (x y z : ℕ), 
  (x + y + z = 50) ∧
  (∃ k : ℕ, z = k * y) ∧
  (z = 32) :=
by {
  let x := 2,
  let y := 16,
  let z := 32,
  have h1 : x + y + z = 50 := by 
    simp [x, y, z],
  have h2 : ∃ k : ℕ, z = k * y := by 
    use 2,
    simp [z, y],
  exact ⟨x, y, z, ⟨h1, ⟨h2, rfl⟩⟩⟩
}

end number_of_chinese_l506_506767


namespace quadrilateral_interior_angle_not_greater_90_l506_506726

-- Definition of the quadrilateral interior angle property
def quadrilateral_interior_angles := ∀ (a b c d : ℝ), (a + b + c + d = 360) → (a > 90 → b > 90 → c > 90 → d > 90 → false)

-- Proposition: There is at least one interior angle in a quadrilateral that is not greater than 90 degrees.
theorem quadrilateral_interior_angle_not_greater_90 :
  (∀ (a b c d : ℝ), (a + b + c + d = 360) → (a > 90 ∧ b > 90 ∧ c > 90 ∧ d > 90) → false) →
  (∃ (a b c d : ℝ), a + b + c + d = 360 ∧ (a ≤ 90 ∨ b ≤ 90 ∨ c ≤ 90 ∨ d ≤ 90)) :=
sorry

end quadrilateral_interior_angle_not_greater_90_l506_506726


namespace common_tangents_count_l506_506680

def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0

theorem common_tangents_count : 
  (∀ x y : ℝ, circle1_eq x y) ∧ (∀ x y : ℝ, circle2_eq x y) → 2 := 
sorry

end common_tangents_count_l506_506680


namespace tan_sin_cos_eq_l506_506404

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l506_506404


namespace find_first_term_geometric_series_l506_506213

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l506_506213


namespace sine_angle_PQ_plane_ABD_l506_506960

variables (A B C D A1 B1 C1 D1 P Q : Type)
variables [euclidean_space A B C D A1 B1 C1 D1 P Q]

-- Given reflections and coordinates setup
variables (plane_ABD : set (euclidean_space)
  [is_plane A1 B D plane_ABD]
  (reflex_P : is_reflection A plane_C1BD P)
  (reflex_Q : is_reflection A line_B1D Q)

-- Given the condition of the shape of the cube
variables (side_length : real) (one unit = 1)

theorem sine_angle_PQ_plane_ABD :
  sin(angle_between (line PQ) plane_ABD) = (sqrt 15) / 5 :=
sorry

end sine_angle_PQ_plane_ABD_l506_506960


namespace find_directrix_of_parabola_l506_506362

noncomputable def directrix_of_parabola (y : ℝ → ℝ) : ℝ :=
  let k := -1 / 8
  let a := 1 / 8
  show ℝ, by
    exact k - 1 / (4 * a)

theorem find_directrix_of_parabola (y : ℝ → ℝ) :
  (∀ x, y x = (x^2 - 8 * x + 15) / 8) →
  (directrix_of_parabola y = -5 / 8) :=
sorry

end find_directrix_of_parabola_l506_506362


namespace integer_solutions_of_inequalities_l506_506179

theorem integer_solutions_of_inequalities :
  ∀ x : ℤ, (2 * x ≥ 3 * (x - 1)) ∧ (2 - x / 2 < 5) ↔ x ∈  {-5, -4, -3, -2, -1, 0, 1, 2, 3} :=
by
  sorry

end integer_solutions_of_inequalities_l506_506179


namespace inequality_1_inequality_2_l506_506658

variable (x a : ℝ)

-- Assertion for the inequality \( 8x - 1 \leq 16x^2 \)
theorem inequality_1 : (8 * x - 1 ≤ 16 * x ^ 2) ↔ (x ∈ set.univ) :=
sorry

-- Assertion for the inequality \( x^2 - 2ax - 3a^2 < 0 \) with \( a < 0 \)
theorem inequality_2 (h : a < 0) : (x ^ 2 - 2 * a * x - 3 * a ^ 2 < 0) ↔ (3 * a < x ∧ x < -a) :=
sorry

end inequality_1_inequality_2_l506_506658


namespace circles_touching_at_least_three_others_l506_506597

theorem circles_touching_at_least_three_others (n : ℕ) (h : n = 2012) : 
  ∃ (C : set ℝ × ℝ), (C.card = n) ∧ (∀ c ∈ C, ∃ S ⊆ C, S.card = 3 ∧ ∀ s ∈ S, dist c s = diam) :=
by
  sorry

end circles_touching_at_least_three_others_l506_506597


namespace find_other_line_l506_506777

theorem find_other_line : 
  ∃ l : ℝ, 
    let line1 (x : ℝ) := x,
        line2 (y : ℝ) := -9 in
      (∃ area : ℝ, 
        ∃ base : ℝ, 
          base = 9 ∧ area = 40.5 ∧ area = (1 / 2) * base * l ∧ line1 (-9) = 9) → l = 9 :=
sorry

end find_other_line_l506_506777


namespace factorized_polynomial_sum_of_squares_l506_506066

theorem factorized_polynomial_sum_of_squares :
  ∃ a b c d e f : ℤ, 
    (729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
    (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210) :=
sorry

end factorized_polynomial_sum_of_squares_l506_506066


namespace area_closed_figure_l506_506665

-- Define the function to be integrated
def f (x : ℝ) : ℝ := (4*x + 2) / ((x + 1) * (3*x + 1))

-- Condition: Integration bounds and function
def integral_f : ℝ := ∫ x in (0 : ℝ)..(1 : ℝ), f x

-- Prove the area equals the given value
theorem area_closed_figure : integral_f = (5 / 3) * Real.log 2 := by
  sorry

end area_closed_figure_l506_506665


namespace maximum_range_of_walk_minimum_range_of_walk_number_of_max_range_sequences_l506_506743

theorem maximum_range_of_walk (a b : ℕ) (h : a > b) : 
  (∃ max_range : ℕ, max_range = a) :=
by {
  use a,
  sorry
}

theorem minimum_range_of_walk (a b : ℕ) (h : a > b) : 
  (∃ min_range : ℕ, min_range = a - b) :=
by {
  use a - b,
  sorry
}

theorem number_of_max_range_sequences (a b : ℕ) (h : a > b) : 
  (∃ num_sequences : ℕ, num_sequences = b + 1) :=
by {
  use b + 1,
  sorry
}

end maximum_range_of_walk_minimum_range_of_walk_number_of_max_range_sequences_l506_506743


namespace largest_prime_factor_sum_l506_506716

theorem largest_prime_factor_sum : 
  let n := 1579 + 5464 in 
  ∃ p : ℕ, nat.prime p ∧ p = n ∧ ∀ q : ℕ, nat.prime q → q ∣ n → q ≤ p :=
begin
  sorry
end

end largest_prime_factor_sum_l506_506716


namespace tan_sin_cos_eq_l506_506452

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l506_506452


namespace sqrt_of_mixed_number_l506_506843

theorem sqrt_of_mixed_number : sqrt (7 + 9 / 16) = 11 / 4 :=
by
  sorry

end sqrt_of_mixed_number_l506_506843


namespace inequality_example_l506_506620

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (der : ∀ x, deriv f x = f' x)

theorem inequality_example (h : ∀ x : ℝ, f x > f' x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023)
:= sorry

end inequality_example_l506_506620


namespace probability_black_ball_l506_506560

theorem probability_black_ball :
  let P_red := 0.41
  let P_white := 0.27
  let P_black := 1 - P_red - P_white
  P_black = 0.32 :=
by
  sorry

end probability_black_ball_l506_506560


namespace calculate_square_of_complex_l506_506827

theorem calculate_square_of_complex (i : ℂ) (h : i^2 = -1) : (1 - i)^2 = -2 * i :=
by
  sorry

end calculate_square_of_complex_l506_506827


namespace math_problem_l506_506334

open Nat

noncomputable def is_special_composite (n : ℕ) : Prop :=
  ∃ (d : ℕ → ℕ) (k : ℕ), (1 < n) ∧ (n > 1) ∧ (1 = d 1) ∧ (d k = n) ∧
  (∀ i, 1 ≤ i → i < k → d (i + 1) = d i + (i * (n - d 1)) / (k - 1)) ∧
  (∀ i, 1 ≤ i → i < k - 1 → d (i + 1) > d i)

theorem math_problem : ∀ n : ℕ, is_special_composite n → n = 4 :=
begin
  sorry
end

end math_problem_l506_506334


namespace find_a_l506_506050

theorem find_a (a : ℝ) (h_cond1 : g x = (a+1)^(x-2) + 1) (h_a_pos : a > 0) (h_fixed : A = (2, 2)) (h_A_in_f : A ∈ set_of (λ p : ℝ × ℝ, p.2 = log 3 (p.1 + a))) : a = 7 :=
sorry

end find_a_l506_506050


namespace log_sum_l506_506504

noncomputable def x_n (n : ℕ) : ℝ :=
(n : ℝ) / (n + 1)

theorem log_sum (x_1 x_2 x_3 ... x_2014 : ℝ) (n : ℕ) :
  (∀ n : ℕ+, x_n n = (n : ℝ) / (n + 1)) →
  ∑ i in finset.range 2014, real.log 2015 (x_n i) = -1 := by
  sorry

end log_sum_l506_506504


namespace age_intervals_l506_506690

theorem age_intervals (A1 A2 A3 A4 A5 : ℝ) (x : ℝ) (h1 : A1 = 7)
  (h2 : A2 = A1 + x) (h3 : A3 = A1 + 2 * x) (h4 : A4 = A1 + 3 * x) (h5 : A5 = A1 + 4 * x)
  (sum_ages : A1 + A2 + A3 + A4 + A5 = 65) :
  x = 3.7 :=
by
  -- Sketch a proof or leave 'sorry' for completeness
  sorry

end age_intervals_l506_506690


namespace shaded_area_l506_506588

-- Definitions from the conditions
def AH : ℝ := 12
def HF : ℝ := 16
def GF : ℝ := 4
def square_area (side : ℝ) : ℝ := side * side

-- Prove that the shaded area is 10 square inches
theorem shaded_area : 
  let DG := GF * (AH / HF) in
  let triangle_area := 0.5 * DG * GF in
  let square_area := square_area 4 in
  square_area - triangle_area = 10 := 
by
  sorry

end shaded_area_l506_506588


namespace find_angle_C_find_a_b_l506_506592

noncomputable def angle_C (a b c : ℝ) (cosB cosC : ℝ) : Prop := 
(b - 2 * a) * cosC + c * cosB = 0

noncomputable def area_triangle (a b c S : ℝ) : Prop := 
S = (1/2) * a * b * Real.sin (Real.pi / 3)

noncomputable def solve_tri_angles (a b c : ℝ) : Prop := 
c = 2 ∧ S = Real.sqrt 3 → (a = 2 ∧ b = 2)

theorem find_angle_C {A B C a b c : ℝ} (h : angle_C a b c (Real.cos A) (Real.cos B)) : 
C = Real.pi / 3 := 
sorry

theorem find_a_b {a b c S : ℝ} (h1 : area_triangle a b c S) 
(h2 : solve_tri_angles a b c) :
a = 2 ∧ b = 2 := 
sorry

end find_angle_C_find_a_b_l506_506592


namespace angle_of_inclination_l506_506357

/-- Given parametric equations x = -1 + sqrt 3 * t and y = 2 - t, the angle 
of inclination of the line is 5π/6. -/
theorem angle_of_inclination : 
  ∃ (θ : ℝ), 
    (∃ (t : ℝ → ℝ × ℝ), (∀ s, t s = ( -1 + sqrt 3 * s, 2 - s))) → θ = 5 * Real.pi / 6 :=
sorry

end angle_of_inclination_l506_506357


namespace tan_sin_cos_eq_l506_506450

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l506_506450


namespace geometric_series_first_term_l506_506202

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l506_506202


namespace quadratic_inequality_solution_l506_506510

theorem quadratic_inequality_solution:
  ∃ (a b c : ℝ),
    (∀ x : ℝ, f(x) = ax^2 + bx + c) ∧ 
    (f(-1) = 0) ∧ 
    (∀ x : ℝ, x ≤ f(x) ∧ f(x) ≤ (1/2) * (1 + x^2)) ∧ 
    (a = 1/4 ∧ b = 1/2 ∧ c = 1/4) :=
begin
  sorry -- proof is omitted
end

end quadratic_inequality_solution_l506_506510


namespace imaginary_part_of_z_l506_506496

variable (z : ℂ)
variable (i : ℂ) 
hypothesis h_i : i = complex.I
hypothesis h_z : z * (1 + i) = 2

theorem imaginary_part_of_z : z * (1 + i) = 2 → complex.im z = -1 := by
  assume h : z * (1 + i) = 2
  sorry

end imaginary_part_of_z_l506_506496


namespace div_by_7_11_13_l506_506017

theorem div_by_7_11_13 (n : ℤ) (A B : ℤ) (hA : A = n % 1000)
  (hB : B = n / 1000) (k : ℤ) (hk : k = A - B) :
  (∃ d, d ∈ {7, 11, 13} ∧ d ∣ n) ↔ (∃ d, d ∈ {7, 11, 13} ∧ d ∣ k) :=
sorry

end div_by_7_11_13_l506_506017


namespace product_of_p_r_s_is_approximately_18_l506_506937

theorem product_of_p_r_s_is_approximately_18 (p r s : ℝ) 
  (hp : 5^p + 5^3 = 140) 
  (hr : 3^r + 21 = 48) 
  (hs : 4^s + 4^3 = 280) 
  : p * r * s ≈ 18 :=
by
  sorry

end product_of_p_r_s_is_approximately_18_l506_506937


namespace infinitely_many_odd_n_composite_l506_506650

theorem infinitely_many_odd_n_composite (n : ℕ) (h_odd : n % 2 = 1) : 
  ∃ (n : ℕ) (h_odd : n % 2 = 1), 
     ∀ k : ℕ, ∃ (m : ℕ) (h_odd_m : m % 2 = 1), 
     (∃ (d : ℕ), d ∣ (2^m + m) ∧ (1 < d ∧ d < 2^m + m))
:=
sorry

end infinitely_many_odd_n_composite_l506_506650


namespace intersection_line_parallel_to_both_lines_l506_506553

noncomputable def are_lines_parallel {α : Type*} [inner_product_space ℝ α] (l₁ l₂ m : affine_subspace ℝ α) (P₁ P₂ : affine_subspace ℝ α) [h1 : has_containment l₁ P₁] [h2 : has_containment l₂ P₂] [h3 : has_containment m P₁] [h4 : has_containment m P₂] [h5 : parallel l₁ l₂] [h6 : intersect P₁ P₂] : Prop :=
  parallel m l₁ ∧ parallel m l₂

theorem intersection_line_parallel_to_both_lines {α : Type*} [inner_product_space ℝ α] 
  (l₁ l₂ m : affine_subspace ℝ α) 
  (P₁ P₂ : affine_subspace ℝ α)
  [h1 : has_containment l₁ P₁] 
  [h2 : has_containment l₂ P₂] 
  [h3 : has_containment m P₁] 
  [h4 : has_containment m P₂] 
  [h5 : parallel l₁ l₂] 
  [h6 : intersect P₁ P₂] :
  are_lines_parallel l₁ l₂ m P₁ P₂ :=
begin
  sorry -- the proof goes here
end

end intersection_line_parallel_to_both_lines_l506_506553


namespace complex_power_equation_l506_506877

variable (x : ℂ)

theorem complex_power_equation (h : x - 1/x = complex.I * (complex.sqrt 3)) :
  x^2188 - 1/x^2188 = -1 :=
by sorry

end complex_power_equation_l506_506877


namespace angle_greater_than_150_l506_506682

theorem angle_greater_than_150 (a b c R : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c < 2 * R) : 
  ∃ (A : ℝ), A > 150 ∧ ( ∃ (B C : ℝ), A + B + C = 180 ) :=
sorry

end angle_greater_than_150_l506_506682


namespace triangle_ratio_l506_506555

theorem triangle_ratio (A B C : ℝ) (a c : ℝ) (hA : A = 45) (hB : B = 105) (hC : C = 30)
    (h_sin_law : a / Real.sin A = c / Real.sin C) : a / c = Real.sqrt 2 := by
  -- Definitions for angles A and B
  have hA_deg : A = 45 := by exact hA
  have hB_deg : B = 105 := by exact hB

  -- Angle sum property gives C
  have hC_deg : C = 180 - A - B := by
    exact hC  -- Given by problem assumption

  -- Using the Law of Sines property
  have h_law_of_sines := h_sin_law

  -- Skipping the detailed proof steps
  sorry

end triangle_ratio_l506_506555


namespace work_completion_times_l506_506659

theorem work_completion_times (a b : ℝ) (h : b > a) :
  ∃ x : ℝ, x = b + real.sqrt (b * (b - a)) ∧
           x - a = b - a + real.sqrt (b * (b - a)) ∧
           x - b = real.sqrt (b * (b - a)) := by
  sorry

end work_completion_times_l506_506659


namespace find_integer_pairs_l506_506281

theorem find_integer_pairs :
  { (m, n) : ℤ × ℤ | n^3 + m^3 + 231 = n^2 * m^2 + n * m } = {(4, 5), (5, 4)} :=
by
  sorry

end find_integer_pairs_l506_506281


namespace sum_of_real_imaginary_parts_eq_zero_l506_506486

theorem sum_of_real_imaginary_parts_eq_zero (z : ℂ) (h : z * complex.I = 1 + complex.I) : z.re + z.im = 0 :=
sorry

end sum_of_real_imaginary_parts_eq_zero_l506_506486


namespace James_distance_ridden_l506_506600

theorem James_distance_ridden :
  let s := 16
  let t := 5
  let d := s * t
  d = 80 :=
by
  sorry

end James_distance_ridden_l506_506600


namespace angle_ABC_120_degrees_l506_506879

open EuclideanGeometry

-- Assume the existence of points A, B, and C being midpoints of the edges of a cube
axiom A B C : Point

-- Assume these points are midpoints of the edges of a cube
axiom midpoints_of_cube_edges : MidpointsCubeEdges A B C

-- The main statement we want to prove
theorem angle_ABC_120_degrees (h : midpoints_of_cube_edges) : angle A B C = 120 := 
sorry

end angle_ABC_120_degrees_l506_506879


namespace bill_difference_zero_l506_506635

theorem bill_difference_zero (l m : ℝ) 
  (hL : (25 / 100) * l = 5) 
  (hM : (15 / 100) * m = 3) : 
  l - m = 0 := 
sorry

end bill_difference_zero_l506_506635


namespace find_a_b_c_l506_506845

theorem find_a_b_c :
  ∃ (a b c : ℕ), 11^a + 3^b = c^2 ∧ a = 4 ∧ b = 5 ∧ c = 122 := 
by
  use [4, 5, 122]
  split; norm_num
  split; refl
  split; refl

end find_a_b_c_l506_506845


namespace geometric_sequence_inequality_l506_506557

-- Given: In Δ ABC, the three sides a, b, and c form a geometric sequence
variable (a b c : ℝ)
variable (A B C : ℝ) -- Angles in triangle ABC
hypothesis h1 : b^2 = a * c
hypothesis h2 : ∠A + ∠B + ∠C = π

-- To prove: a cos^2(C / 2) + c cos^2(A / 2) ≥ 3 / 2 * b
theorem geometric_sequence_inequality (h1 : b^2 = a * c) (h2 : A + B + C = π)
    : a * (cos (C / 2))^2 + c * (cos (A / 2))^2 ≥ (3 / 2) * b :=
sorry

end geometric_sequence_inequality_l506_506557


namespace find_value_f_l506_506482

noncomputable def f (x : ℝ) : ℝ := 
if h : 0 ≤ x ∧ x ≤ Real.pi / 2 then Real.sin x else 
if h : x < 0 then -f (-x) else 
let y := x - ⌊x / Real.pi⌋ * Real.pi in f y

lemma period_f : ∀ x : ℝ, f (x + Real.pi) = f x := 
sorry -- encapsulating the periodic property

lemma odd_f : ∀ x : ℝ, f (-x) = -f x := 
sorry -- encapsulating the odd property

theorem find_value_f : f (5 * Real.pi / 3) = - Real.sqrt 3 / 2 := 
by {
  have h1 : f (5 * Real.pi / 3) = f (5 * Real.pi / 3 - 2 * Real.pi), 
  from period_f (5 * Real.pi / 3 - Real.pi),
  rw Real.sub_self at h1,
  rw add_zero at h1,
  have h2 : f (-Real.pi / 3) = -f (Real.pi / 3), 
  from odd_f (Real.pi / 3),
  rw h2,
  have h3 : f (Real.pi / 3) = Real.sin (Real.pi / 3), 
  from if_pos (by linarith),
  rw h3,
  norm_num,
  ring,
}

end find_value_f_l506_506482


namespace payment_denotation_is_correct_l506_506332

-- Define the initial condition of receiving money
def received_amount : ℤ := 120

-- Define the payment amount
def payment_amount : ℤ := 85

-- The expected payoff
def expected_payment_denotation : ℤ := -85

-- Theorem stating that the payment should be denoted as -85 yuan
theorem payment_denotation_is_correct : (payment_amount = -expected_payment_denotation) :=
by
  sorry

end payment_denotation_is_correct_l506_506332


namespace number_of_people_voting_for_Marty_l506_506148

noncomputable def percentage_voting_for_Biff : ℝ := 0.45
noncomputable def percentage_undecided : ℝ := 0.08
noncomputable def total_people_polled : ℝ := 200

theorem number_of_people_voting_for_Marty 
  (percentage_voting_for_Biff : ℝ) 
  (percentage_undecided : ℝ) 
  (total_people_polled : ℝ)
  (h_percentages : percentage_voting_for_Biff = 0.45 ∧ percentage_undecided = 0.08 ∧ total_people_polled = 200) :
  (total_people_polled * (1 - percentage_voting_for_Biff - percentage_undecided)) = 94 :=
by
  cases h_percentages with percentBiff rest
  cases rest with undecided total
  rw [percentBiff, undecided, total]
  -- The following line completes the proof using the remaining steps
  -- which are not required in the problem statement but provided for context.
  -- linarith [mul_comm, one_sub, sub_eq_add_neg, mul_assoc]
  exact sorry

end number_of_people_voting_for_Marty_l506_506148


namespace problem_statement_l506_506053

def U := Set ℝ
def A := {x : ℝ | 0 ≤ x ∧ x ≤ 3}
def B := {y : ℝ | ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ y = 2^x}
def complement_U_A := set.compl A
def intersection_complement_U_A_B := complement_U_A ∩ B

theorem problem_statement :
  A = {x : ℝ | 0 ≤ x ∧ x ≤ 3} ∧ 
  B = {y : ℝ | 2 ≤ y ∧ y ≤ 4} ∧ 
  intersection_complement_U_A_B = {y : ℝ | 3 < y ∧ y ≤ 4} :=
by
  sorry

end problem_statement_l506_506053


namespace min_jumps_for_grasshopper_l506_506776

def jump_grasshopper (jump_distance: ℝ) (grid_size: ℕ) (cell_size: ℝ) (points: ℕ) : ℕ :=
  -- The function computes the minimum number of jumps to visit all points
  if jump_distance = 50 ∧ grid_size = 8 ∧ cell_size = 10 ∧ points = 8 then 8 else sorry

theorem min_jumps_for_grasshopper :
  ∀ (jump_distance: ℝ) (grid_size: ℕ) (cell_size: ℝ) (points: ℕ), 
  jump_distance = 50 ∧ grid_size = 8 ∧ cell_size = 10 ∧ points = 8 → 
  jump_grasshopper jump_distance grid_size cell_size points = 8 :=
by
  intros jump_distance grid_size cell_size points h
  unfold jump_grasshopper
  rw if_pos h
  refl

end min_jumps_for_grasshopper_l506_506776


namespace domain_of_f_l506_506673

def domain_of_function (f : ℝ → ℝ) : set ℝ :=
  {x | ∃ y z, y = 1/x ∧ z = real.sqrt (1 - x) ∧ f x = y + z}

theorem domain_of_f :
  domain_of_function (λ x, 1/x + real.sqrt (1 - x)) = {x | (x < 0 ∨ (0 < x ∧ x ≤ 1))} :=
by
  sorry

end domain_of_f_l506_506673


namespace line_tangent_to_circle_angle_l506_506759

theorem line_tangent_to_circle_angle {t θ α : ℝ} :
  (∃ t : ℝ, (t * cos α - 4) ^ 2 + (t * sin α) ^ 2 = 4) →
  (cos α = sqrt 3 / 2 ∨ cos α = -sqrt 3 / 2) →
  (α = π / 6 ∨ α = 5 * π / 6) :=
sorry

end line_tangent_to_circle_angle_l506_506759


namespace cartesian_eq_C1_intersection_points_C3_C1_intersection_points_C3_C2_l506_506584

section CartesianAndPolarCurves

variable {t s θ : ℝ}
variable {x y : ℝ}

-- Define the parametric equations for C1 and C2
def parametric_C1 (t x y : ℝ) : Prop := x = (2 + t) / 6 ∧ y = Real.sqrt t
def parametric_C2 (s x y : ℝ) : Prop := x = -(2 + s) / 6 ∧ y = -Real.sqrt s

-- Define the polar equation for C3 and convert it to Cartesian form
def polar_C3_cartesian_eq : Prop := 2 * x - y = 0

-- Cartesian equation of C1
def cartesian_C1 (x y : ℝ) : Prop := y^2 = 6 * x - 2

-- Cartesian equation of C2
def cartesian_C2 (x y : ℝ) : Prop := y^2 = -6 * x - 2

-- Intersection points of C3 with C1
def intersection_C3_C1 (x y : ℝ) : Prop :=
  (y = 2 * x ∧ x = 1 / 2 ∧ y = 1) ∨ (y = 2 * x ∧ x = 1 ∧ y = 2)

-- Intersection points of C3 with C2
def intersection_C3_C2 (x y : ℝ) : Prop :=
  (y = 2 * x ∧ x = -1 / 2 ∧ y = -1) ∨ (y = 2 * x ∧ x = -1 ∧ y = -2)

-- Prove the problem
theorem cartesian_eq_C1 (t x y : ℝ) : parametric_C1 t x y → cartesian_C1 x y :=
  sorry

theorem intersection_points_C3_C1 (x y : ℝ) : cartesian_C1 x y → polar_C3_cartesian_eq → intersection_C3_C1 x y :=
  sorry

theorem intersection_points_C3_C2 (x y : ℝ) : cartesian_C2 x y → polar_C3_cartesian_eq → intersection_C3_C2 x y :=
  sorry

end CartesianAndPolarCurves

end cartesian_eq_C1_intersection_points_C3_C1_intersection_points_C3_C2_l506_506584


namespace find_x_tan_eq_l506_506431

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l506_506431


namespace nat_divisor_problem_l506_506115

open Nat

theorem nat_divisor_problem (n : ℕ) (d : ℕ → ℕ) (k : ℕ)
    (h1 : 1 = d 1)
    (h2 : ∀ i, 1 < i → i ≤ k → d i < d (i + 1))
    (hk : d k = n)
    (hdiv : ∀ i, 1 ≤ i ∧ i ≤ k → d i ∣ n)
    (heq : n = d 2 * d 3 + d 2 * d 5 + d 3 * d 5) :
    k = 8 ∨ k = 9 :=
sorry

end nat_divisor_problem_l506_506115


namespace tan_sin_cos_eq_l506_506454

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l506_506454


namespace angle_BKM_half_BAC_l506_506013

-- Definitions as per the conditions
variables {A B C D K M : Type}
variables [triangle_ABC : Triangle A B C]
variables (D_on_ray_BA : Point_on_ray D BA)
variables (D_condition : BD = BA + AC)
variables [K_on_BA : on_side K BA]
variables [M_on_BC : on_side M BC]
variables (areas_equal : triangle_area B D M = triangle_area B C K)

-- The statement to prove
theorem angle_BKM_half_BAC
  (triangle_ABC : Triangle A B C)
  (D_on_ray_BA : Point_on_ray D BA)
  (D_condition : BD = BA + AC)
  (K_on_BA : on_side K BA)
  (M_on_BC : on_side M BC)
  (areas_equal : triangle_area B D M = triangle_area B C K) :
  ∠ B K M = 1/2 * ∠ B A C :=
by
  sorry -- Proof goes here

end angle_BKM_half_BAC_l506_506013


namespace john_taking_pictures_years_l506_506109

-- Definitions based on the conditions
def pictures_per_day : ℕ := 10
def images_per_card : ℕ := 50
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140
def days_per_year : ℕ := 365

-- Theorem statement
theorem john_taking_pictures_years : total_spent / cost_per_card * images_per_card / pictures_per_day / days_per_year = 3 :=
by
  sorry

end john_taking_pictures_years_l506_506109


namespace math_problem_proof_l506_506581

-- Define the parametric equations of C1 and the Cartesian equation
def parametric_C1 (t : ℝ) : ℝ × ℝ := ( (2 + t) / 6, sqrt t )
def cartesian_C1 (x y : ℝ) : Prop := y^2 = 6 * x - 2 ∧ y ≥ 0

-- Define the parametric equations of C2
def parametric_C2 (s : ℝ) : ℝ × ℝ := ( -(2 + s) / 6, -sqrt s )

-- Define the polar equation of C3 and its Cartesian equivalent
def polar_C3 (θ : ℝ) : Prop := 2 * cos θ - sin θ = 0
def cartesian_C3 (x y : ℝ) : Prop := 2 * x - y = 0

-- Define the intersection points of C3 with C1
def intersection_C3_C1 : set (ℝ × ℝ) := [(1/2, 1), (1, 2)].to_set

-- Define the intersection points of C3 with C2
def intersection_C3_C2 : set (ℝ × ℝ) := [(-1/2, -1), (-1, -2)].to_set

-- Main proof statement
theorem math_problem_proof {x y : ℝ} (t s θ : ℝ) :
  parametric_C1 t = (x, y) →
  cartesian_C3 x y →
  (cartesian_C1 x y ∧ (x, y) ∈ intersection_C3_C1) ∨
  (parametric_C2 s = (x, y) → (cartesian_C1 x y ∧ (x, y) ∈ intersection_C3_C2)) :=
by sorry

end math_problem_proof_l506_506581


namespace part1_part2_part3_l506_506481

-- Define the given conditions and statements
variables {x y k m : ℝ}

-- 1. Coordinates of point A when |OA| = |AC|
theorem part1 (h : |y| = 3 / 2) (ellip_eq : x ^ 2 / 4 + y ^ 2 / 3 = 1) : 
    (x = 1 ∧ y = 3 / 2) ∨ (x = -1 ∧ y = 3 / 2) :=
sorry

-- 2. Equation of the line when S(ΔAOC) = S(ΔAOB)
theorem part2 (kx_eq : y = k * x + 3) (ellip_eq : x ^ 2 / 4 + y ^ 2 / 3 = 1) :
    k = 3 / 2 ∨ k = -3 / 2 :=
sorry

-- 3. Area of ΔAOB when k_OA * k_OB = -3/4
theorem part3 (slope_cond : ∀ A B, k = (3/4)^(1/2)) : 
    (1/2) * |x * 3 / 2| = (3)^(1/2) :=
sorry

end part1_part2_part3_l506_506481


namespace find_c_value_l506_506458

noncomputable def equation_value : ℂ :=
  let complex_i := complex.I
  let val1 := (3/2)^4
  let val2 := Real.tan (Real.pi / 3)
  let val3 := (4/5 : ℝ)⁻³
  let val4 := complex.abs (Complex.sqrt (Complex.ofReal 729) - 3 * complex_i) -- using abs to get real part (27)
  let val5 := Real.log2 1024
  let val6 := Real.cos (Real.pi / 2)
  let val7 := 2 * Real.pi * Real.sin (2 * Real.pi / 3)
  val1 * val2 * val3 * val4 / val5 * val6 + val7

theorem find_c_value :
  equation_value = Real.pi * Real.sqrt 3 :=
by
  sorry

end find_c_value_l506_506458


namespace area_of_intersection_l506_506696

-- Conditions
def point1 : ℝ × ℝ := (5, 11)
def point2 : ℝ × ℝ := (16, 11)
def point3 : ℝ × ℝ := (16, -2)
def point4 : ℝ × ℝ := (5, -2)

def radius : ℝ := 4
def center : ℝ × ℝ := (5, -2)

-- Circle Equation
def circle_eq (x y : ℝ) : Prop := (x - 5)^2 + (y + 2)^2 = 16

-- Rectangle vertices
def is_vertex (p : ℝ × ℝ) : Prop := p = point1 ∨ p = point2 ∨ p = point3 ∨ p = point4

-- Proof Statement
theorem area_of_intersection : 
  let rect_vertices := [point1, point2, point3, point4] in
  (∀ p ∈ rect_vertices, is_vertex p) →
  circle_eq 5 (-2) →
  ∃ area : ℝ, area = 4 * π := 
by
  sorry

end area_of_intersection_l506_506696


namespace camp_marshmallows_l506_506823

theorem camp_marshmallows 
    (total_campers : ℕ)
    (boys_ratio : ℚ)
    (girls_ratio : ℚ)
    (boys_toast_ratio : ℚ)
    (girls_toast_ratio : ℚ)
    (h_total : total_campers = 96)
    (hb_ratio : boys_ratio = 2 / 3)
    (hg_ratio : girls_ratio = 1 / 3)
    (hb_toast_ratio : boys_toast_ratio = 1 / 2)
    (hg_toast_ratio : girls_toast_ratio = 3 / 4) :
    let boys := boys_ratio * total_campers
        girls := girls_ratio * total_campers
        boys_toast := boys_toast_ratio * boys
        girls_toast := girls_toast_ratio * girls
        marshmallows_needed := boys_toast + girls_toast
    in marshmallows_needed = 56 := 
by 
    sorry

end camp_marshmallows_l506_506823


namespace least_b_angle_l506_506569

open Nat

def is_prime_angle (n : ℕ) : Prop :=
  Prime n ∧ n < 90

def valid_angles (a b : ℕ) : Prop :=
  is_prime_angle a ∧ is_prime_angle b ∧ a > b ∧ a + b = 90

theorem least_b_angle : ∃ b, (∃ a, valid_angles a b) ∧ 
  (∀ b', (∃ a', valid_angles a' b') → b ≤ b') :=
by
  exists 7
  sorry

end least_b_angle_l506_506569


namespace P_xx_degree_even_l506_506186

noncomputable def P : Polynomial ℚ × Polynomial ℚ → Polynomial ℚ := sorry

axiom P_condition (n : ℕ) :
  (∀ k : ℕ, (k ≤ n → degree (P (x, n)) ≤ n) 
   ∨ (P (x, n) = 0)) ∧
  (∀ k : ℕ, (k ≤ n → degree (P (n, y)) ≤ n) 
   ∨ (P (n, y) = 0))

theorem P_xx_degree_even : 
  ∀ P : Polynomial ℚ × Polynomial ℚ → Polynomial ℚ , 
  (∀ n : ℕ, P_condition n) → 
  even (degree (P (x, x))) := 
sorry

end P_xx_degree_even_l506_506186


namespace determine_triangle_value_l506_506908

theorem determine_triangle_value (p : ℕ) (triangle : ℕ) (h1 : triangle + p = 67) (h2 : 3 * (triangle + p) - p = 185) : triangle = 51 := by
  sorry

end determine_triangle_value_l506_506908


namespace remainder_of_expression_mod7_l506_506719

theorem remainder_of_expression_mod7 :
  (7^6 + 8^7 + 9^8) % 7 = 5 :=
by
  sorry

end remainder_of_expression_mod7_l506_506719


namespace odd_function_f_l506_506030

noncomputable def f (x : ℝ) : ℝ :=
if h : x <= 0 then
  (Real.cbrt x - x^2)
else
  (Real.cbrt x + x^2)

theorem odd_function_f (x : ℝ) (h : x > 0) :
  (∀ x, f (-x) = -f x) ∧ (∀ x, x <= 0 → f x = Real.cbrt x - x^2) →
  f x = Real.cbrt x + x^2 := by
  intros properties_given
  have odd_fn := properties_given.1
  have f_form := properties_given.2
  sorry

end odd_function_f_l506_506030


namespace initial_volume_of_solution_l506_506286

variable (V : ℝ)
variables (h1 : 0.10 * V = 0.08 * (V + 16))
variables (V_correct : V = 64)

theorem initial_volume_of_solution : V = 64 := by
  sorry

end initial_volume_of_solution_l506_506286


namespace find_c_value_l506_506824

theorem find_c_value (a b c : ℝ) (h1 : ∀ x, a * cos (b * x + c) = a * (cos (b * (x - 0) + c)))
  (h2 : 6 = 2 * |a|) : c = π :=
by sorry

end find_c_value_l506_506824


namespace OP_eq_OQ_l506_506627

variable (A B C O P Q K L M : Type)
variable [Point A] [Point B] [Point C] [Point O]
          [Point P] [Point Q] [Point K] [Point L] [Point M]
variable (circumcircle_center : ∀ {A B C : Type} [Point A] [Point B] [Point C], Point)
variable (midpoint : ∀ {X Y : Type} [Point X] [Point Y], Point)
variable (tangent : ∀ {T1 T2 : Type} [Point T1] [Point T2], Prop)
variable (segment : Type → Type → Type)
variable (line : Type → Type → Type)

def is_midpoint (M : Point) (X Y : Point) : Prop := 
  midpoint X Y = M

def is_circumcenter (O : Point) (A B C : Point) : Prop :=
  O = circumcircle_center A B C

theorem OP_eq_OQ (A B C O P Q K L M : Type)
  [Point A] [Point B] [Point C] [Point O] [Point P] [Point Q] 
  [Point K] [Point L] [Point M]
  (h_circumcenter : is_circumcenter O A B C)
  (h_P_on_AC : segment P A = segment P C)
  (h_Q_on_AB : segment Q A = segment Q B)
  (h_mid_BK : is_midpoint K B P)
  (h_mid_CL : is_midpoint L C Q)
  (h_mid_PQ : is_midpoint M P Q)
  (h_tangent : tangent (line P Q) (circumcircle_center K L M))
  : segment O P = segment O Q :=
sorry

end OP_eq_OQ_l506_506627


namespace number_of_dots_on_faces_l506_506691

theorem number_of_dots_on_faces (d A B C D : ℕ) 
  (h1 : d = 6)
  (h2 : A = 3)
  (h3 : B = 5)
  (h4 : C = 6)
  (h5 : D = 5) :
  A = 3 ∧ B = 5 ∧ C = 6 ∧ D = 5 :=
by {
  sorry
}

end number_of_dots_on_faces_l506_506691


namespace positive_difference_of_b_values_l506_506140

noncomputable def g (n : ℤ) : ℤ :=
if n ≤ 0 then n^2 + 3 * n + 2 else 3 * n - 15

theorem positive_difference_of_b_values : 
  abs (-5 - 9) = 14 :=
by {
  sorry
}

end positive_difference_of_b_values_l506_506140


namespace chess_tournament_participants_l506_506084

theorem chess_tournament_participants
  (n : ℕ)
  (h1 : 3 < n)
  (h2 : ∀ p1 p2 : ℕ, p1 ≠ p2 → plays_against p1 p2 = true)
  (h3 : total_rounds = 26)
  (h4 : (∀ p : ℕ, odd_points (points p) = (p = 1))):
  n = 8 :=
sorry

-- Here we assume that plays_against and points are some functions defined elsewhere.

end chess_tournament_participants_l506_506084


namespace apples_ratio_l506_506312

theorem apples_ratio (initial_apples rickis_apples end_apples samsons_apples : ℕ)
(h_initial : initial_apples = 74)
(h_ricki : rickis_apples = 14)
(h_end : end_apples = 32)
(h_samson : initial_apples - rickis_apples - end_apples = samsons_apples) :
  samsons_apples / Nat.gcd samsons_apples rickis_apples = 2 ∧ rickis_apples / Nat.gcd samsons_apples rickis_apples = 1 :=
by
  sorry

end apples_ratio_l506_506312


namespace length_segments_equality_l506_506276

noncomputable def triangle_with_inequalities (ABC : Triangle) (AB BC CA : ℝ)
  (I : Point) (HA HB HC KA LA KB LB KC LC : Point) :=
  -- Conditions
  (length AB < length BC) ∧ (length BC < length CA) ∧ -- triangle side inequalities
  (I = incenter ABC) ∧ -- incenter of the triangle
  (HA = orthocenter I BC) ∧ (HB = orthocenter I AC) ∧ (HC = orthocenter I AB) ∧ -- orthocenters of smaller triangles
  (KA = intersection (line_segment HB HC) (line_segment B C)) ∧ -- intersection points on BC
  (LA = intersection (perpendicular I (line_segment HB HC)) (line_segment B C)) ∧ -- perpendicular line intersection
  (KB = ... ) ∧ (LB = ...) ∧ (KC = ... ) ∧ (LC = ...) -- similar definitions for KB, LB, KC, LC

theorem length_segments_equality (ABC : Triangle) (AB BC CA : ℝ)
  (I : Point) (HA HB HC KA LA KB LB KC LC : Point)
  (h : triangle_with_inequalities ABC AB BC CA I HA HB HC KA LA KB LB KC LC) :
  distance KA LA = distance KB LB + distance KC LC :=
  sorry -- proof is omitted 
  
end length_segments_equality_l506_506276


namespace find_x_value_l506_506406

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l506_506406


namespace simplify_expr1_l506_506762

theorem simplify_expr1 (m n : ℝ) :
  (2 * m + n) ^ 2 - (4 * m + 3 * n) * (m - n) = 8 * m * n + 4 * n ^ 2 := by
  sorry

end simplify_expr1_l506_506762


namespace smallest_positive_period_symmetry_axis_range_of_f_l506_506514
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 6))

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem symmetry_axis (k : ℤ) :
  ∃ k : ℤ, ∃ x : ℝ, f x = f (x + k * (Real.pi / 2)) ∧ x = (Real.pi / 6) + k * (Real.pi / 2) := sorry

theorem range_of_f : 
  ∀ x, -Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 → -1/2 ≤ f x ∧ f x ≤ 1 := sorry

end smallest_positive_period_symmetry_axis_range_of_f_l506_506514


namespace find_first_term_l506_506199

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l506_506199


namespace find_x_tan_identity_l506_506440

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l506_506440


namespace intersection_is_origin_l506_506528

-- Define the lines and points as given
def l1 (y: ℝ) : Prop := y = 2
def l2 (y: ℝ) : Prop := y = 4

def A : ℝ × ℝ := (Real.log 2 / Real.log 3, 2)
def B : ℝ × ℝ := (Real.log 4 / Real.log 3, 4)
def C : ℝ × ℝ := (Real.log 2 / Real.log 5, 2)
def D : ℝ × ℝ := (Real.log 4 / Real.log 5, 4)

-- Definition of intersection point we need to prove
def intersectionPoint : ℝ × ℝ := (0, 0)

-- The theorem stating both lines AB and CD pass through the origin (0,0)
theorem intersection_is_origin :
  let AB := λ (x : ℝ), (x - A.1) * (B.2 - A.2) = (B.1 - A.1) * (x - 0)
  let CD := λ (x : ℝ), (x - C.1) * (D.2 - C.2) = (D.1 - C.1) * (x - 0)
  @line.intersected_at_point (@line.of_slope_point AB B.1 B.2) (@line.of_slope_point CD D.1 D.2) (0, 0) := 
sorry

end intersection_is_origin_l506_506528


namespace tan_sin_cos_eq_l506_506405

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l506_506405


namespace exists_xy_l506_506861

open Classical

variable (f : ℝ → ℝ)

theorem exists_xy (h : ∃ x₀ y₀ : ℝ, f x₀ ≠ f y₀) : ∃ x y : ℝ, f (x + y) < f (x * y) :=
by
  sorry

end exists_xy_l506_506861


namespace find_first_term_l506_506214

noncomputable def first_term : ℝ :=
  let a := 20 * (1 - (2 / 3)) in a

theorem find_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = first_term :=
by
  sorry

end find_first_term_l506_506214


namespace cartesian_eq_C1_intersection_points_C3_C1_intersection_points_C3_C2_l506_506585

section CartesianAndPolarCurves

variable {t s θ : ℝ}
variable {x y : ℝ}

-- Define the parametric equations for C1 and C2
def parametric_C1 (t x y : ℝ) : Prop := x = (2 + t) / 6 ∧ y = Real.sqrt t
def parametric_C2 (s x y : ℝ) : Prop := x = -(2 + s) / 6 ∧ y = -Real.sqrt s

-- Define the polar equation for C3 and convert it to Cartesian form
def polar_C3_cartesian_eq : Prop := 2 * x - y = 0

-- Cartesian equation of C1
def cartesian_C1 (x y : ℝ) : Prop := y^2 = 6 * x - 2

-- Cartesian equation of C2
def cartesian_C2 (x y : ℝ) : Prop := y^2 = -6 * x - 2

-- Intersection points of C3 with C1
def intersection_C3_C1 (x y : ℝ) : Prop :=
  (y = 2 * x ∧ x = 1 / 2 ∧ y = 1) ∨ (y = 2 * x ∧ x = 1 ∧ y = 2)

-- Intersection points of C3 with C2
def intersection_C3_C2 (x y : ℝ) : Prop :=
  (y = 2 * x ∧ x = -1 / 2 ∧ y = -1) ∨ (y = 2 * x ∧ x = -1 ∧ y = -2)

-- Prove the problem
theorem cartesian_eq_C1 (t x y : ℝ) : parametric_C1 t x y → cartesian_C1 x y :=
  sorry

theorem intersection_points_C3_C1 (x y : ℝ) : cartesian_C1 x y → polar_C3_cartesian_eq → intersection_C3_C1 x y :=
  sorry

theorem intersection_points_C3_C2 (x y : ℝ) : cartesian_C2 x y → polar_C3_cartesian_eq → intersection_C3_C2 x y :=
  sorry

end CartesianAndPolarCurves

end cartesian_eq_C1_intersection_points_C3_C1_intersection_points_C3_C2_l506_506585


namespace correct_statements_count_l506_506184

theorem correct_statements_count (a b c ℓ : Type) [parallel a b] [parallel b c] [exists_unique (λ x, parallel ℓ x)] :
  (∃! (p : Prop), 
     (p = (∀ (x y : Type), parallel x y → parallel y z → parallel x z)) ∧
     (p = (∀ (x y), corresponding_angles_equal x y)) ∧
     (p = (∃ (x), ∀ (y ≠ x), ¬parallel ℓ y → parallel ℓ x)) ∧
     (p = (∀ x y, intersect x y → vertical_angles_equal x y))) :=
begin
  sorry
end

end correct_statements_count_l506_506184


namespace probability_dart_center_square_l506_506773

-- Define the probability calculation problem
theorem probability_dart_center_square :
  let hexadecagon : Type := { n // n = 16 } in
  let equally_likely (P : hexadecagon → Prop) : Prop := ∀ p : hexadecagon, P p → P (p + 1) in
  let center_square_area := 1 in
  let hexadecagon_area := 9 in
  center_square_area / hexadecagon_area = 1 / 9 :=
sorry

end probability_dart_center_square_l506_506773


namespace contrapositive_example_l506_506174

variable {a : ℕ → ℝ}

theorem contrapositive_example 
  (h₁ : ∀ n : ℕ, n > 0 → (a n + a (n + 2)) / 2 < a (n + 1)) :
  (∀ n : ℕ, n > 0 → a n ≤ a (n + 1)) → ∀ n : ℕ, n > 0 → (a n + a (n + 2)) / 2 ≥ a (n + 1) :=
by
  sorry

end contrapositive_example_l506_506174


namespace triangle_BC_point_PA_PB_l506_506097

variables (A B C P : Type) [normed_space ℝ A] [normed_space ℝ B] [normed_space ℝ C] [normed_space ℝ P]
variables (AB AC : A → B)
variables (BA BC CA CB PA PB : A → A)
variables [is_equilateral : ∀ a b : A, dist a b = 2] (h1 : ∀ v, (BA v) • (BC v) = 2)
variables [has_dot : ∀ v, (PA v) • (PB v) = 24]
noncomputable def lengths (A B C : A) : Prop :=
  dist A B = 2 ∧ dist A C = 2

noncomputable def dot_product_condition (BA BC : A → A) : Prop :=
  ∀ v, (BA v) • (BC v) = 2

noncomputable def point_condition (CP CA CB : A → A) : Prop :=
  ∀ v, (CP v) = (1 / 2) • (CA v) - 2 • (CB v)

noncomputable def is_equilateral_triangle (A B C : A) : Prop :=
  ∀ a b c : A, (dist a b = dist b c) ∧ (dist b c = dist c a)

theorem triangle_BC (A B C : A)
  (h1: lengths A B C)
  (h2: dot_product_condition BA BC) :
  dist B C = 2 := sorry

theorem point_PA_PB (A B C P : A)
  (h1: lengths A B C)
  (h2: dot_product_condition BA BC)
  (h3 : point_condition CP CA CB) :
  ∀ v, (PA v) • (PB v) = 24 := sorry

end triangle_BC_point_PA_PB_l506_506097


namespace Cartesian_equation_C1_intersection_C3_C1_intersection_C3_C2_l506_506577

-- Define C1 as a parametric curve
def C1 (t : ℝ) : ℝ × ℝ := ( (2 + t) / 6, real.sqrt t)

-- Define the Cartesian equation of C1
def Cartesian_C1 (x y : ℝ) : Prop := y ^ 2 = 6 * x - 2 ∧ y >= 0

-- Define C2 as a parametric curve
def C2 (s : ℝ) : ℝ × ℝ := ( -(2 + s) / 6, -real.sqrt s)

-- Define the Cartesian equation of C2
def Cartesian_C2 (x y : ℝ) : Prop := y ^ 2 = -6 * x - 2 ∧ y <= 0

-- Define the polar coordinate equation of C3
def Cartesian_C3 (x y : ℝ) : Prop := 2 * x - y = 0

-- Proving the correctness of converting parametric to Cartesian equation for C1
theorem Cartesian_equation_C1 {t : ℝ} : ∃ y, ∃ x, C1 t = (x, y) → Cartesian_C1 x y := 
sorry

-- Proving the correctness of intersection points of C3 with C1
theorem intersection_C3_C1 : 
  ∃ x1 y1, ∃ x2 y2, Cartesian_C3 x1 y1 ∧ Cartesian_C1 x1 y1 ∧ Cartesian_C3 x2 y2 ∧ Cartesian_C1 x2 y2 ∧ 
  ((x1 = 1 / 2 ∧ y1 = 1) ∧ (x2 = 1 ∧ y2 = 2)) :=
sorry

-- Proving the correctness of intersection points of C3 with C2
theorem intersection_C3_C2 : 
  ∃ x1 y1, ∃ x2 y2, Cartesian_C3 x1 y1 ∧ Cartesian_C2 x1 y1 ∧ Cartesian_C3 x2 y2 ∧ Cartesian_C2 x2 y2 ∧ 
  ((x1 = -1 / 2 ∧ y1 = -1) ∧ (x2 = -1 ∧ y2 = -2)) :=
sorry

end Cartesian_equation_C1_intersection_C3_C1_intersection_C3_C2_l506_506577


namespace product_of_fraction_036_l506_506718

theorem product_of_fraction_036 :
  let x := 0.036.repeat
  let frac : ℚ := 36 / 999
  let simplified := frac.num / frac.denom
  (simp : ℚ → ℚ := λ q, (q.num / q.denom).num * (q.num / q.denom).denom)
  simp(simplified) = 444 := by
  sorry

end product_of_fraction_036_l506_506718


namespace solve_tan_equation_l506_506367

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l506_506367


namespace range_of_a_l506_506001

theorem range_of_a (a : ℝ) (b : ℝ) :
  (∃ x ∈ (set.Icc 0 1), |x^2 + a * x + b| ≥ 1) → a ≥ 1 ∨ a ≤ -3 := sorry

end range_of_a_l506_506001


namespace det_matrix_eq_one_l506_506999

variable {R : Type*} [Field R]

theorem det_matrix_eq_one (a b c d : R) (H : Matrix ((Fin 2) × (Fin 2)) R) (hA : H = !![a, b; c, d]) (h_inv : H + (H⁻¹) = 0) :
  Matrix.det H = 1 :=
sorry

end det_matrix_eq_one_l506_506999


namespace sum_two_digit_numbers_divisible_by_sum_product_and_three_l506_506256

theorem sum_two_digit_numbers_divisible_by_sum_product_and_three :
  ∃ S, S = ∑ x in ({x : ℕ | 10 ≤ x ∧ x < 100 ∧ (let ⟨a, b⟩ := (x / 10, x % 10) in (a + b) ∣ x ∧ (a * b) ∣ x ∧ 3 ∣ x)} : Finset ℕ) id ∧ S = 69 :=
begin
  sorry
end

end sum_two_digit_numbers_divisible_by_sum_product_and_three_l506_506256


namespace cot_arctan_sum_l506_506859

theorem cot_arctan_sum :
  cot (arctan 4 + arctan 9 + arctan 11 + arctan 19) = 3445 / 863 :=
by
  sorry

end cot_arctan_sum_l506_506859


namespace angle_measure_RST_l506_506715

-- Definitions for conditions of the problem
def is_regular_polygon (P : list point) (n : ℕ) : Prop := 
  (n ≥ 3 ∧ P.length = n ∧ ∀ (i : ℕ) (h : i < n), dist (P.nth_le i h) (P.nth_le ((i + 1) % n) (by sorry)) = dist (P.nth_le 1 (by sorry)) (P.nth_le 0 (by sorry)))

def is_isosceles_triangle (A B C : point) : Prop := 
  dist A B = dist A C ∨ dist B C = dist A B ∨ dist C A = dist B C

def is_isosceles_trapezoid (A B C D : point) : Prop :=
  (dist A B = dist C D) ∧ (dist A D ≠ dist B C)

-- Problem statement about angle measure
theorem angle_measure_RST (R S T U V W X : point)
  (h1 : is_regular_polygon [R, S, T, U, V, W, X] 8)
  (h2 : is_isosceles_triangle V W X)
  (h3 : dist R V = dist R X)
  (h4 : is_isosceles_trapezoid R V W X) :
  measure_angle R S T = 22.5 :=
sorry  -- Proof is omitted

end angle_measure_RST_l506_506715


namespace remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2_l506_506255

theorem remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2 :
  (x^15 - 1) % (x + 1) = -2 := 
sorry

end remainder_of_x_pow_15_minus_1_div_x_plus_1_is_neg_2_l506_506255


namespace find_x_tan_eq_l506_506428

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l506_506428


namespace find_number_l506_506971

def satisfies_conditions (n : ℕ) : Prop :=
  n % 3 = 2 ∧ n % 5 = 3 ∧ n % 7 = 2 ∧ 100 ≤ n ∧ n ≤ 200

theorem find_number : ∃ n : ℕ, satisfies_conditions n ∧ n = 128 := 
by
  let n := 128
  have h1 : n % 3 = 2 := by decide
  have h2 : n % 5 = 3 := by decide
  have h3 : n % 7 = 2 := by decide
  have h4 : 100 ≤ n := by decide
  have h5 : n ≤ 200 := by decide
  existsi n
  split
  constructor
  exact ⟨h1, h2, h3, h4, h5⟩
  rfl
  exact sorry

end find_number_l506_506971


namespace equivalent_proof_problem_l506_506530

variables (a b : ℝ → ℝ → ℝ)  -- vectors in the Euclidean space

noncomputable def fixed_distance (p : ℝ → ℝ → ℝ) (ta : ℝ → ℝ → ℝ) (ub : ℝ → ℝ → ℝ) := 
  ∃ (c : ℝ), ∀ (p : ℝ → ℝ → ℝ), ‖p - ta - ub‖ = c

def question : Prop :=
  ∃ t u, ∀ p, ‖p - b‖ = 3 * ‖p - a‖ →
  fixed_distance p (t • a) (u • b)

theorem equivalent_proof_problem : question a b := sorry

end equivalent_proof_problem_l506_506530


namespace zhou_yu_age_eq_l506_506735

-- Define the conditions based on the problem statement
variable (x : ℕ)  -- x represents the tens digit of Zhou Yu's age

-- Condition: The tens digit is three less than the units digit
def units_digit := x + 3

-- Define Zhou Yu's age based on the tens and units digits
def zhou_yu_age := 10 * x + units_digit x

-- Prove the correct equation representing Zhou Yu's lifespan
theorem zhou_yu_age_eq : zhou_yu_age x = (units_digit x) ^ 2 :=
by sorry

end zhou_yu_age_eq_l506_506735


namespace distance_from_origin_to_line_l506_506881

-- Define the given conditions
def line_through_point (p : ℝ × ℝ) (d : ℝ × ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ a * p.1 + b * p.2 + c = 0 ∧ d.1 * b - d.2 * a = 0

def distance_from_origin (a b c : ℝ) : ℝ :=
  (abs c) / (sqrt (a ^ 2 + b ^ 2))

-- Define the proof problem
theorem distance_from_origin_to_line : 
  line_through_point (-sqrt 5, 0) (2, -1) → 
  ∃ a b c : ℝ, a = 1 ∧ b = 2 ∧ c = sqrt 5 ∧ distance_from_origin a b c = 1 :=
by 
  intros h
  sorry

end distance_from_origin_to_line_l506_506881


namespace math_problem_equiv_l506_506475

def circle_tangent_line : Prop := 
  (∀ (x y x2 y2 : ℝ), (x - y - 2 * real.sqrt 2 = 0) → (x + y)² = 4) 

def line_chord_length : Prop :=
  ∀ (l1 l2 r : ℝ), (l1, l2, r) = (4, -3, 5) → 2 * real.sqrt 3 = 2 * real.sqrt (r² - 1²)

noncomputable def y_intercept_perpendicular : Prop :=
  ∃ b : ℝ, b = 2 ∨ b = -2

def line_tangents_G_eq : Prop :=
  y_intercept_perpendicular ∧ ∀ G : ℝ × ℝ, G = (1, 3) → ∀ (M N : ℝ × ℝ),
  (M ≠ N) → (tangents_through_point (circle 4) G = (MN = (line x + 3 * y - 4))

theorem math_problem_equiv :
  circle_tangent_line ∧ line_chord_length ∧ y_intercept_perpendicular ∧ line_tangents_G_eq := sorry

end math_problem_equiv_l506_506475


namespace students_in_both_band_and_chorus_l506_506230

-- Definitions for conditions
def total_students : ℕ := 300
def students_in_band : ℕ := 100
def students_in_chorus : ℕ := 120
def students_in_band_or_chorus : ℕ := 195

-- Theorem: Prove the number of students in both band and chorus
theorem students_in_both_band_and_chorus : ℕ :=
  students_in_band + students_in_chorus - students_in_band_or_chorus

example : students_in_both_band_and_chorus = 25 := by
  sorry

end students_in_both_band_and_chorus_l506_506230


namespace num_students_playing_b_sports_l506_506232

theorem num_students_playing_b_sports (num_a_sports : ℕ) (mul_factor : ℕ) (num_a_sports_eq : num_a_sports = 6) (mul_factor_eq : mul_factor = 4) : mul_factor * num_a_sports = 24 := by
  rw [num_a_sports_eq, mul_factor_eq]
  exact rfl

end num_students_playing_b_sports_l506_506232


namespace find_x_value_l506_506409

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l506_506409


namespace find_x_l506_506395

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l506_506395


namespace find_first_term_l506_506218

noncomputable def first_term : ℝ :=
  let a := 20 * (1 - (2 / 3)) in a

theorem find_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = first_term :=
by
  sorry

end find_first_term_l506_506218


namespace divisibility_l506_506019

theorem divisibility {n A B k : ℤ} (h_n : n = 1000 * B + A) (h_k : k = A - B) :
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) :=
by
  sorry

end divisibility_l506_506019


namespace fixed_point_sum_l506_506476

-- Define the function with conditions on the logarithm base
def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.logb a (2 * x - 3)

theorem fixed_point_sum (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  let m := 2
  let n := f a m
  m + n = 3 :=
by
  sorry

end fixed_point_sum_l506_506476


namespace sum_first_n_l506_506921

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 1 then 2 else 3 * sequence (n - 1) + 2

theorem sum_first_n (n : ℕ) : 
  (finset.sum (finset.range n) (λ k, sequence (k+1))) = (3^(n+1) - 3) / 2 - n :=
begin
  sorry
end

end sum_first_n_l506_506921


namespace printer_paper_last_days_l506_506168

def packs : Nat := 2
def sheets_per_pack : Nat := 240
def prints_per_day : Nat := 80
def total_sheets : Nat := packs * sheets_per_pack
def number_of_days : Nat := total_sheets / prints_per_day

theorem printer_paper_last_days :
  number_of_days = 6 :=
by
  sorry

end printer_paper_last_days_l506_506168


namespace total_scoops_l506_506324

-- Definitions
def single_cone_scoops : ℕ := 1
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def waffle_bowl_scoops : ℕ := banana_split_scoops + 1

-- Theorem statement: Prove the total number of scoops is 10.
theorem total_scoops : single_cone_scoops + double_cone_scoops + banana_split_scoops + waffle_bowl_scoops = 10 :=
by
  -- Proof steps would go here
  sorry

end total_scoops_l506_506324


namespace no_algorithm_fewer_than_2016_questions_l506_506229

noncomputable def check_connectivity (n : ℕ) : Prop :=
  ∀ (V : Finset ℕ) (K : Finset (ℕ × ℕ)), 
  V.card = n → 
  K ⊆ (V.product V) → 
  ∃ G : Sym2 (ℕ × ℕ), 
  (G ⊆ K ∨ G ⊆ (V.product V) \ K) → 
  (G.card ≥ n * (n - 1) / 2)

theorem no_algorithm_fewer_than_2016_questions :
  ¬ check_connectivity 64 :=
sorry

end no_algorithm_fewer_than_2016_questions_l506_506229


namespace problem1_arith_seq_problem2_sum_inequality_l506_506008

-- Sequence definition and conditions
def a : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n+2) := 2 * a (n+1) + 2^(n+2)

-- Sum of the first n terms of the sequence
def S : ℕ → ℕ
| 0 := 0
| (n+1) := S n + a (n+1)

-- Proof (Ⅰ) Statement
theorem problem1_arith_seq (n : ℕ) : n ≥ 2 → (a n) / (2^n) - (a (n-1)) / (2^(n-1)) = 1 := by
  sorry

-- Proof (Ⅱ) Statement
theorem problem2_sum_inequality (n : ℕ) : n > 0 → (S n) / (2^n) > (2 * n - 3) := by
  sorry

end problem1_arith_seq_problem2_sum_inequality_l506_506008


namespace rectangle_area_stage4_l506_506067

-- Define the condition: area of one square
def square_area : ℕ := 25

-- Define the condition: number of squares at Stage 4
def num_squares_stage4 : ℕ := 4

-- Define the total area of rectangle at Stage 4
def total_area_stage4 : ℕ := num_squares_stage4 * square_area

-- Prove that total_area_stage4 equals 100 square inches
theorem rectangle_area_stage4 : total_area_stage4 = 100 :=
by
  sorry

end rectangle_area_stage4_l506_506067


namespace probability_of_pythagorean_triples_is_correct_l506_506954

-- Define the set of integers to choose from
def number_set := {1, 2, 3, 4, 5}

-- Define a function to check if three numbers form a Pythagorean triple
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Define the total number of ways to select three different numbers from the set
def total_combinations : ℕ :=
  (number_set.to_finset.card.choose 3).nat_abs

-- Define the number of Pythagorean triples
def pythagorean_triples_count : ℕ :=
  ((number_set.to_list.combinations 3).count (λ l, match l with
    | [a, b, c] := is_pythagorean_triple a b c
    | _ := false
  end)).nat_abs

-- Define the probability calculation
def probability_pythagorean_triples : ℚ :=
  pythagorean_triples_count / total_combinations

-- State the proof problem
theorem probability_of_pythagorean_triples_is_correct :
  probability_pythagorean_triples = 1 / 10 :=
by sorry

end probability_of_pythagorean_triples_is_correct_l506_506954


namespace max_value_f_on_interval_l506_506511

noncomputable def f (x : ℝ) := - (1 / 3) * x^3 + x^2 + 3 * x - 5

theorem max_value_f_on_interval :
  (∀ x ∈ set.Icc (-2 : ℝ) (1 : ℝ), f x ≤ f (-1)) ∧ f (-1) = - (4 / 3) :=
by
  sorry

end max_value_f_on_interval_l506_506511


namespace ceil_add_eq_double_of_int_l506_506704

theorem ceil_add_eq_double_of_int {x : ℤ} (h : ⌈(x : ℝ)⌉ + ⌊(x : ℝ)⌋ = 2 * (x : ℝ)) : ⌈(x : ℝ)⌉ + x = 2 * x :=
by
  sorry

end ceil_add_eq_double_of_int_l506_506704


namespace cost_per_container_is_21_l506_506107

-- Define the given problem conditions as Lean statements.

--  Let w be the number of weeks represented by 210 days.
def number_of_weeks (days: ℕ) : ℕ := days / 7
def weeks : ℕ := number_of_weeks 210

-- Let p be the total pounds of litter used over the number of weeks.
def pounds_per_week : ℕ := 15
def total_litter_pounds (weeks: ℕ) : ℕ := weeks * pounds_per_week
def total_pounds : ℕ := total_litter_pounds weeks

-- Let c be the number of 45-pound containers needed for the total pounds of litter.
def pounds_per_container : ℕ := 45
def number_of_containers (total_pounds pounds_per_container: ℕ) : ℕ := total_pounds / pounds_per_container
def containers : ℕ := number_of_containers total_pounds pounds_per_container

-- Given the total cost, find the cost per container.
def total_cost : ℕ := 210
def cost_per_container (total_cost containers: ℕ) : ℕ := total_cost / containers
def cost : ℕ := cost_per_container total_cost containers

-- Prove that the cost per container is 21.
theorem cost_per_container_is_21 : cost = 21 := by
  sorry

end cost_per_container_is_21_l506_506107


namespace no_roots_in_disk_l506_506916

noncomputable def homogeneous_polynomial_deg2 (a b c : ℝ) (x y : ℝ) := a * x^2 + b * x * y + c * y^2
noncomputable def homogeneous_polynomial_deg3 (q : ℝ → ℝ → ℝ) (x y : ℝ) := q x y

theorem no_roots_in_disk 
  (a b c : ℝ) (h_poly_deg2 : ∀ x y, homogeneous_polynomial_deg2 a b c x y = a * x^2 + b * x * y + c * y^2)
  (q : ℝ → ℝ → ℝ) (h_poly_deg3 : ∀ x y, homogeneous_polynomial_deg3 q x y = q x y)
  (h_cond : b^2 < 4 * a * c) :
  ∃ k > 0, ∀ x y, x^2 + y^2 < k → homogeneous_polynomial_deg2 a b c x y ≠ homogeneous_polynomial_deg3 q x y ∨ (x = 0 ∧ y = 0) :=
sorry

end no_roots_in_disk_l506_506916


namespace number_of_circuits_tractor2_l506_506304

noncomputable def radius_tractor1 : ℝ := 30
noncomputable def circuits_tractor1 : ℕ := 20
noncomputable def radius_tractor2 : ℝ := 10

theorem number_of_circuits_tractor2 :
  ∃ n : ℕ, (2 * Real.pi * radius_tractor1 * circuits_tractor1) = (2 * Real.pi * radius_tractor2 * n) :=
by
  use 60
  sorry

end number_of_circuits_tractor2_l506_506304


namespace proof_of_main_l506_506141

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x > -1 → (log a (x + 1)) < (log a x + log a 1)

noncomputable def proposition_q (a : ℝ) : Prop :=
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*a - 3)*x1 + 1 = 0 ∧ x2^2 + (2*a - 3)*x2 + 1 = 0)

noncomputable def main_proof_problem : Prop :=
  (proposition_p a ∧ ¬ proposition_q a) → (1 / 2 ≤ a ∧ a < 1)

theorem proof_of_main : main_proof_problem :=
sorry

end proof_of_main_l506_506141


namespace first_term_of_geometric_series_l506_506220

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l506_506220


namespace false_statements_count_l506_506887

-- Definitions of lines and planes as sets
variables (m n : Set Point) (α β : Set Point)

-- Conditions
axiom diff_lines : m ≠ n
axiom diff_planes : α ≠ β

-- Statements to be evaluated
def statement1 : Prop := (α ∥ β) ∧ (m ⊆ α) → (m ∥ β)
def statement2 : Prop := (m ∥ n) ∧ (m ∥ β) → (n ∥ β)
def statement3 : Prop := (m ⊆ α) ∧ (n ⊆ β) → (SkewLines m n)
def statement4 : Prop := (α ⊥ β) ∧ (m ∥ α) → (m ⊥ β)

-- Main assertion 
theorem false_statements_count : 
  (¬statement1) + (¬statement2) + (¬statement3) + (¬statement4) = 3 :=
sorry

end false_statements_count_l506_506887


namespace nuts_in_trail_mix_l506_506113

theorem nuts_in_trail_mix :
  let walnuts := 0.25
  let almonds := 0.25
  walnuts + almonds = 0.50 :=
by
  sorry

end nuts_in_trail_mix_l506_506113


namespace bricks_needed_for_wall_l506_506284

def brick_volume (length width height : ℝ) : ℝ :=
  length * width * height

def wall_volume (length width height : ℝ) : ℝ :=
  length * width * height

def bricks_required_for_wall (brick_volume wall_volume : ℝ) : ℝ :=
  wall_volume / brick_volume

theorem bricks_needed_for_wall : 
  bricks_required_for_wall 
      (brick_volume 20 10 7.5) 
      (wall_volume (27 * 100) (2 * 100) (0.75 * 100)) = 27000 :=
by
  sorry

end bricks_needed_for_wall_l506_506284


namespace locus_of_vertices_l506_506468

-- Define the initial conditions
variables {c g : ℝ} (m : ℝ) (α : ℝ)
def x_coord := m * sin (2 * α)
def y_coord := m * sin α ^ 2

-- Given relationship between m, c, and g
noncomputable def m_val := c^2 / (2 * g)

-- Theorem statement
theorem locus_of_vertices (h_m : m = m_val) (h_α : 0 ≤ α ∧ α ≤ π / 2) :
  ∃ (x y : ℝ), x = m * sin (2 * α) ∧ y = m * sin α ^ 2 ∧ 
  (x^2 / (4 * m^2) + (y - m)^2 / m^2 = 1) ∧ x ≥ 0 :=
begin
  use [m * sin (2 * α), m * sin α ^ 2],
  split,
  { exact rfl },
  split,
  { exact rfl },
  split,
  { sorry },
  { sorry }
end

end locus_of_vertices_l506_506468


namespace black_white_ratio_l506_506801

theorem black_white_ratio (initial_black_tiles initial_white_tiles grid_size : ℕ)
  (borders : ℕ) (new_black_tiles_ratio : ℚ)
  (h_initial : initial_black_tiles = 9)
  (h_initial_white : initial_white_tiles = 16)
  (h_grid_size : grid_size = 5)
  (h_borders : borders = 2)
  (h_new_black_tiles_ratio : new_black_tiles_ratio = 65 / 16)
  : let final_black_tiles = initial_black_tiles + 24 + 32 in
    let final_white_tiles = initial_white_tiles in
    (final_black_tiles / final_white_tiles = new_black_tiles_ratio) := 
by
  sorry

end black_white_ratio_l506_506801


namespace find_first_term_geometric_series_l506_506210

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l506_506210


namespace circle_equation_center_xaxis_radius_2_l506_506674

theorem circle_equation_center_xaxis_radius_2 (a x y : ℝ) :
  (0:ℝ) < 2 ∧ (a - 1)^2 + 2^2 = 4 -> (x - 1)^2 + y^2 = 4 :=
by
  sorry

end circle_equation_center_xaxis_radius_2_l506_506674


namespace calculate_f_f_f_l506_506044

def f (x : ℤ) : ℤ := 3 * x + 2

theorem calculate_f_f_f :
  f (f (f 3)) = 107 :=
by
  sorry

end calculate_f_f_f_l506_506044


namespace contrapositive_squared_l506_506667

theorem contrapositive_squared (a : ℝ) : (a ≤ 0 → a^2 ≤ 0) ↔ (a > 0 → a^2 > 0) :=
by
  sorry

end contrapositive_squared_l506_506667


namespace solve_tan_equation_l506_506375

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l506_506375


namespace combination_identity_l506_506761

theorem combination_identity : (Nat.choose 5 3 + Nat.choose 5 4 = Nat.choose 6 4) := 
by 
  sorry

end combination_identity_l506_506761


namespace smallest_four_digit_multiple_of_18_l506_506365

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n > 999 ∧ n < 10000 ∧ 18 ∣ n ∧ (∀ m : ℕ, m > 999 ∧ m < 10000 ∧ 18 ∣ m → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l506_506365


namespace range_of_m_l506_506484

variable {x : ℝ} {m : ℝ}

def p : Prop := ∃ x : ℝ, (m + 1) * (x^2 + 1) ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h : ¬(p ∧ q)) : m ∈ Set.Iic (-2) ∪ Set.Ioi (-1) :=
sorry

end range_of_m_l506_506484


namespace no_taylor_series_x_minus_1_taylor_series_x_taylor_series_x_plus_1_l506_506839

-- The function to be expanded
def f (x : ℝ) := 1 / (1 - x)

-- Proof that the Taylor series expansion around x = 1 does not exist
theorem no_taylor_series_x_minus_1 : ¬(∃ (a : Finset ℕ) (b : ℕ → ℝ), ∀ (n : ℕ), n ∈ a → f x = ∑ i in a, b i * (x-1)^i) := sorry

-- Proof of the Taylor series expansion around x = 0
theorem taylor_series_x (x : ℝ) (h : |x| < 1) :
  f x = ∑ n in Finset.range 100 -- approximation up to the 100th term for practicality
    (λ n, (x^n : ℝ)) := sorry

-- Proof of the Taylor series expansion around x = -1
theorem taylor_series_x_plus_1 (x : ℝ) (h : |x+1| < 2) :
  f x = ∑ n in Finset.range 100 -- approximation up to the 100th term for practicality
    (λ n, (1 / 2^(n+1) * (x + 1)^n : ℝ)) := sorry

end no_taylor_series_x_minus_1_taylor_series_x_taylor_series_x_plus_1_l506_506839


namespace equal_discount_percentage_l506_506110

theorem equal_discount_percentage :
  ∀ (original_price1 sale_price1 original_price2 sale_price2 : ℝ),
  original_price1 = 100 ∧ sale_price1 = 70 ∧ original_price2 = 150 ∧ sale_price2 = 105 →
  (((original_price1 - sale_price1) / original_price1) * 100 = 
   ((original_price2 - sale_price2) / original_price2) * 100) :=
by
  intros original_price1 sale_price1 original_price2 sale_price2 h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  sorry

end equal_discount_percentage_l506_506110


namespace log_intersection_l506_506331

theorem log_intersection (x : ℝ) (h1 : x > 0) :
  (3 * log x = log (3 * x)) ↔ (x = real.sqrt 3) :=
sorry

end log_intersection_l506_506331


namespace f_x_plus_3_eq_l506_506128

def f (x : ℝ) : ℝ := (x * (x - 1)) / 2

theorem f_x_plus_3_eq (x : ℝ) : f (x + 3) = (x^2 + 5*x + 6) / 2 := 
by sorry

end f_x_plus_3_eq_l506_506128


namespace arithmetic_mean_is_correct_l506_506707

open Nat

noncomputable def arithmetic_mean_of_two_digit_multiples_of_9 : ℝ :=
  let smallest := 18
  let largest := 99
  let n := 10
  let sum := (n / 2 : ℝ) * (smallest + largest)
  sum / n

theorem arithmetic_mean_is_correct :
  arithmetic_mean_of_two_digit_multiples_of_9 = 58.5 :=
by
  -- Proof goes here
  sorry

end arithmetic_mean_is_correct_l506_506707


namespace trajectory_equation_l506_506233

-- Define the points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

def A (y : ℝ) : Point := { x := -2, y := y }
def B (y : ℝ) : Point := { x := 0, y := y / 2 }
def C (x y: ℝ) : Point := { x := x, y := y }

-- Define the vectors AB and BC
def vector (p1 p2 : Point) : Point :=
{ x := p2.x - p1.x, y := p2.y - p1.y }

def AB (y : ℝ) : Point := vector (A y) (B y)
def BC (x y : ℝ) : Point := vector (B y) (C x y)

-- Define the dot product
def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Proving the trajectory equation
theorem trajectory_equation (x y : ℝ) (h : dot_product (AB y) (BC x y) = 0) : y ^ 2 = 8 * x :=
by
  unfold AB BC dot_product at h
  simp at h
  sorry

end trajectory_equation_l506_506233


namespace div_by_7_11_13_l506_506018

theorem div_by_7_11_13 (n : ℤ) (A B : ℤ) (hA : A = n % 1000)
  (hB : B = n / 1000) (k : ℤ) (hk : k = A - B) :
  (∃ d, d ∈ {7, 11, 13} ∧ d ∣ n) ↔ (∃ d, d ∈ {7, 11, 13} ∧ d ∣ k) :=
sorry

end div_by_7_11_13_l506_506018


namespace find_x_l506_506379

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l506_506379


namespace children_absent_on_independence_day_l506_506156

theorem children_absent_on_independence_day
  (total_children : ℕ)
  (bananas_per_child : ℕ)
  (extra_bananas : ℕ)
  (total_possible_children : total_children = 780)
  (bananas_distributed : bananas_per_child = 2)
  (additional_bananas : extra_bananas = 2) :
  ∃ (A : ℕ), A = 390 := 
sorry

end children_absent_on_independence_day_l506_506156


namespace william_total_riding_hours_l506_506730

theorem william_total_riding_hours :
  let max_daily_time := 6
  let days := 6
  let max_time_days := 2
  let reduced_time1_days := 2
  let reduced_time2_days := 2
  let reduced_time1 := 1.5
  let reduced_time2 := max_daily_time / 2
  max_daily_time * max_time_days + reduced_time1 * reduced_time1_days + reduced_time2 * reduced_time2_days = 21 :=
begin
  sorry,
end

end william_total_riding_hours_l506_506730


namespace S_n_bounds_l506_506922

open BigOperators

noncomputable def a_seq (n : ℕ) : ℕ :=
if n = 1 then 9 else (a_seq (n-1) + 2 * (n-1) + 5)

noncomputable def b_seq (n : ℕ) : ℝ :=
if n = 1 then 1 / 4 else ((n : ℝ) / (n+1)) * b_seq (n-1)

def sqrt (x : ℝ) : ℝ := real.sqrt x

def S (n : ℕ) : ℝ :=
∑ i in finset.range n, (b_seq (i+1) / sqrt (a_seq (i+1)))

theorem S_n_bounds (n : ℕ) : (1 / 12 : ℝ) ≤ S n ∧ S n < (1 / 4 : ℝ) := 
sorry

end S_n_bounds_l506_506922


namespace no_solution_sqrt_eq_neg3_l506_506847

theorem no_solution_sqrt_eq_neg3 (x : ℝ) : ¬(sqrt (5 - x) = -3) :=
by
  -- placeholder proof to ensure code compiles
  sorry

end no_solution_sqrt_eq_neg3_l506_506847


namespace trajectory_of_M_max_inradius_l506_506035

-- Conditions
def point_on_circle (P : ℝ × ℝ) : Prop :=
  (P.fst + 1)^2 + P.snd^2 = 12

def center_F1 : ℝ × ℝ := (-1, 0)
def point_F2 : ℝ × ℝ := (1, 0)

-- Part 1
theorem trajectory_of_M (P : ℝ × ℝ) (M : ℝ × ℝ) 
  (hP : point_on_circle P)
  (h_intersect: ∃ M, true): -- given the conditions, M exists as per problem description
  M ∈ { p : ℝ × ℝ | p.fst^2 / 3 + p.snd^2 / 2 = 1 } :=
sorry

-- Part 2
theorem max_inradius (A B : ℝ × ℝ) (r : ℝ)
  (hA : A ∈ { p : ℝ × ℝ | p.fst^2 / 3 + p.snd^2 / 2 = 1 })
  (hB : B ∈ { p : ℝ × ℝ | p.fst^2 / 3 + p.snd^2 / 2 = 1 })
  (h_l : { p : ℝ × ℝ | p.fst = 1 } ∩ { p : ℝ × ℝ | p ∈ { p : ℝ × ℝ | p.fst^2 / 3 + p.snd^2 / 2 = 1 } } = {A, B}) :
  r = 2 / 3 ∧ (∀ x y: ℝ, (x = 1) ∧ (y = 1) → r = 2 / 3) :=
sorry

end trajectory_of_M_max_inradius_l506_506035


namespace range_of_a_l506_506049

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x + a * x^2 - (2 * a + 1) * x + 1

theorem range_of_a (a : ℝ) (h_a : 0 < a ∧ a ≤ 1/2) : 
  ∀ x : ℝ, x ∈ Set.Ici a → f x a ≥ a^3 - a - 1/8 :=
by
  sorry

end range_of_a_l506_506049


namespace max_sum_k_l506_506633

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d
noncomputable def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ := n * (2 * a + (n - 1) * d) / 2

theorem max_sum_k :
  ∀ (a1 a4 a5 : ℤ) (d : ℤ),
  a4 = arithmetic_sequence a1 d 4 →
  a5 = arithmetic_sequence a1 d 5 →
  (arithmetic_sequence a1 d 1 + a4 + arithmetic_sequence a1 d 7 = 99) →
  (arithmetic_sequence a1 d 2 + a5 + arithmetic_sequence a1 d 8 = 93) →
  (∀ n : ℕ+, sum_first_n_terms a1 d n ≤ sum_first_n_terms a1 d 20) →
  20 = 20 :=
begin
  intros,
  sorry
end

end max_sum_k_l506_506633


namespace card_2_in_box_Q_l506_506347

theorem card_2_in_box_Q (P Q : Finset ℕ) (hP : P.card = 3) (hQ : Q.card = 5) 
  (hdisjoint : Disjoint P Q) (huniv : P ∪ Q = (Finset.range 9).erase 0)
  (hsum_eq : P.sum id = Q.sum id) :
  2 ∈ Q := 
sorry

end card_2_in_box_Q_l506_506347


namespace find_x_l506_506389

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l506_506389


namespace math_problem_proof_l506_506582

-- Define the parametric equations of C1 and the Cartesian equation
def parametric_C1 (t : ℝ) : ℝ × ℝ := ( (2 + t) / 6, sqrt t )
def cartesian_C1 (x y : ℝ) : Prop := y^2 = 6 * x - 2 ∧ y ≥ 0

-- Define the parametric equations of C2
def parametric_C2 (s : ℝ) : ℝ × ℝ := ( -(2 + s) / 6, -sqrt s )

-- Define the polar equation of C3 and its Cartesian equivalent
def polar_C3 (θ : ℝ) : Prop := 2 * cos θ - sin θ = 0
def cartesian_C3 (x y : ℝ) : Prop := 2 * x - y = 0

-- Define the intersection points of C3 with C1
def intersection_C3_C1 : set (ℝ × ℝ) := [(1/2, 1), (1, 2)].to_set

-- Define the intersection points of C3 with C2
def intersection_C3_C2 : set (ℝ × ℝ) := [(-1/2, -1), (-1, -2)].to_set

-- Main proof statement
theorem math_problem_proof {x y : ℝ} (t s θ : ℝ) :
  parametric_C1 t = (x, y) →
  cartesian_C3 x y →
  (cartesian_C1 x y ∧ (x, y) ∈ intersection_C3_C1) ∨
  (parametric_C2 s = (x, y) → (cartesian_C1 x y ∧ (x, y) ∈ intersection_C3_C2)) :=
by sorry

end math_problem_proof_l506_506582


namespace kittens_total_l506_506791

theorem kittens_total (original new : ℕ) (h₁ : original = 6) (h₂ : new = 3) : original + new = 9 :=
by { rw [h₁, h₂], exact rfl }

end kittens_total_l506_506791


namespace ice_cream_scoops_total_l506_506323

noncomputable def scoops_of_ice_cream : ℕ :=
let single_cone : ℕ := 1 in
let double_cone : ℕ := single_cone * 2 in
let banana_split : ℕ := single_cone * 3 in
let waffle_bowl : ℕ := banana_split + 1 in
single_cone + double_cone + banana_split + waffle_bowl

theorem ice_cream_scoops_total : scoops_of_ice_cream = 10 :=
sorry

end ice_cream_scoops_total_l506_506323


namespace value_range_correct_l506_506227

noncomputable def value_range (x : ℝ) : ℝ :=
  4 * (Real.cos x) ^ 2 + 6 * Real.sin x - 6

theorem value_range_correct : ∀ x ∈ Icc (-Real.pi / 6) (2 * Real.pi / 3),
  value_range x ∈ Icc (-6) (1 / 4) :=
sorry

end value_range_correct_l506_506227


namespace sin_ge_cos_range_l506_506096

theorem sin_ge_cos_range (x : ℝ) (h1 : 0 < x) (h2 : x < 2 * Real.pi) 
    : x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4) ↔ sin x ≥ cos x :=
sorry

end sin_ge_cos_range_l506_506096


namespace find_first_term_l506_506215

noncomputable def first_term : ℝ :=
  let a := 20 * (1 - (2 / 3)) in a

theorem find_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = first_term :=
by
  sorry

end find_first_term_l506_506215


namespace tan_sin_cos_eq_l506_506453

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l506_506453


namespace sum_of_coefficients_l506_506034

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem sum_of_coefficients (n : ℕ) (hn1 : 5 < n) (hn2 : n < 7)
  (coeff_cond : binom n 3 > binom n 2 ∧ binom n 3 > binom n 4) :
  (1 + 1)^n = 64 :=
by
  have h : n = 6 :=
    by sorry -- provided conditions force n to be 6
  show 2^n = 64
  rw [h]
  exact rfl

end sum_of_coefficients_l506_506034


namespace maciek_total_cost_l506_506784

theorem maciek_total_cost :
  let p := 4
  let cost_of_chips := 1.75 * p
  let pretzels_cost := 2 * p
  let chips_cost := 2 * cost_of_chips
  let t := pretzels_cost + chips_cost
  t = 22 :=
by
  sorry

end maciek_total_cost_l506_506784


namespace total_spending_eq_total_is_19_l506_506261

variable (friend_spending your_spending total_spending : ℕ)

-- Conditions
def friend_spending_eq : friend_spending = 11 := by sorry
def friend_spent_more : friend_spending = your_spending + 3 := by sorry

-- Proof that total_spending is 19
theorem total_spending_eq : total_spending = friend_spending + your_spending :=
  by sorry

theorem total_is_19 : total_spending = 19 :=
  by sorry

end total_spending_eq_total_is_19_l506_506261


namespace max_range_eq_a_min_range_eq_a_minus_b_num_sequences_max_range_eq_b_plus_1_l506_506748

-- Definitions based on the given conditions
variables (a b : ℕ) (h : a > b)

-- Proving the maximum possible range equals a
theorem max_range_eq_a : max_range a b = a :=
by sorry

-- Proving the minimum possible range equals a - b
theorem min_range_eq_a_minus_b : min_range a b = a - b :=
by sorry

-- Proving the number of sequences resulting in the maximum range equals b + 1
theorem num_sequences_max_range_eq_b_plus_1 : num_sequences_max_range a b = b + 1 :=
by sorry

end max_range_eq_a_min_range_eq_a_minus_b_num_sequences_max_range_eq_b_plus_1_l506_506748


namespace find_x_between_0_and_180_l506_506418

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l506_506418


namespace min_dominoes_inside_squares_l506_506800

theorem min_dominoes_inside_squares :
  let grid_size : ℕ := 100
  let domino_size : ℕ := 2
  let square_size : ℕ := 2
  let cells_per_domino : ℕ := domino_size
  let total_dominoes : ℕ := (grid_size * grid_size) / cells_per_domino
  ∃(min_dominoes : ℕ), 
    min_dominoes = 100 ∧ 
    (∀ tiling : list (fin (grid_size * grid_size)),
      (∀ d : fin total_dominoes, 
       (∃ square : fin square_size, 
        ∃ index : fin (total_dominoes / square_size),
        tiling.nth_le (index.1) sorry = d → 
        (count_dominoes_inside_squares tiling domino_size square_size square.dominates)) = min_dominoes)) :=
begin
  sorry
end

end min_dominoes_inside_squares_l506_506800


namespace problem1_solution_set_problem2_range_of_a_l506_506884

-- Definitions and statements for Problem 1
def f1 (x : ℝ) : ℝ := -12 * x ^ 2 - 2 * x + 2

theorem problem1_solution_set :
  (∃ a b : ℝ, a = -12 ∧ b = -2 ∧
    ∀ x : ℝ, f1 x > 0 → -1 / 2 < x ∧ x < 1 / 3) :=
by sorry

-- Definitions and statements for Problem 2
def f2 (x a : ℝ) : ℝ := a * x ^ 2 - x + 2

theorem problem2_range_of_a :
  (∃ b : ℝ, b = -1 ∧
    ∀ a : ℝ, (∀ x : ℝ, f2 x a < 0 → false) → a ≥ 1 / 8) :=
by sorry

end problem1_solution_set_problem2_range_of_a_l506_506884


namespace intersection_A_B_l506_506000

open Set

def f (x : ℕ) : ℕ := x^2 - 12 * x + 36

def A : Set ℕ := {a | 1 ≤ a ∧ a ≤ 10}

def B : Set ℕ := {b | ∃ a, a ∈ A ∧ b = f a}

theorem intersection_A_B : A ∩ B = {1, 4, 9} :=
by
  -- Proof skipped
  sorry

end intersection_A_B_l506_506000


namespace minimum_surface_area_of_circumscribed_sphere_l506_506100

-- Given Conditions
variables (x y : ℝ)
def right_triangular_prism (angle_BAC : ℝ) (BC : ℝ) (BB_1 : ℝ) (area_lateral_face : ℝ) : Prop :=
  angle_BAC = 90 ∧ BC = 2 * x ∧ BB_1 = 2 * y ∧ area_lateral_face = 4

-- The problem statement
theorem minimum_surface_area_of_circumscribed_sphere (x y : ℝ)
  (h_cond : right_triangular_prism 90 (2 * x) (2 * y) 4)
  (h_xy : x * y = 1) 
  :
  4 * Real.pi * 2 = 8 * Real.pi :=
by sorry

end minimum_surface_area_of_circumscribed_sphere_l506_506100


namespace circumcircle_radius_of_LMN_eq_two_l506_506980

open EuclideanGeometry

/-- Given a triangle ABC with angle bisectors AL, BM, and CN,
    angles \(\angle ANM = \angle ALC\), and sides LM = 3 and MN = 4,
    the radius of the circumcircle of triangle LMN is 2. -/
theorem circumcircle_radius_of_LMN_eq_two 
    {A B C L M N : Point}
    (h1 : IsAngleBisector A L B C)
    (h2 : IsAngleBisector B M A C)
    (h3 : IsAngleBisector C N A B)
    (h4 : ∠ A N M = ∠ A L C)
    (h5_1 : dist L M = 3)
    (h5_2 : dist M N = 4) :
    circumcircleRadius L M N = 2 :=
by
  sorry

end circumcircle_radius_of_LMN_eq_two_l506_506980


namespace solve_tan_equation_l506_506372

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l506_506372


namespace find_first_term_geometric_series_l506_506212

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l506_506212


namespace function_properties_l506_506075

variable (f : ℝ → ℝ)
variable (h_even : ∀ x, f x = f (-x))
variable (h_increasing : ∀ x y, x < y → y ≤ -1 → f x < f y)

theorem function_properties :
  f 2 < f (-1.5) ∧ f (-1.5) < f (-1) :=
by
  have h1 : f 2 = f (-2) := by apply h_even
  have h2 : f (-2) < f (-1.5) := by apply h_increasing _ _ (by linarith) (by linarith)
  have h3 : f (-1.5) < f (-1) := by apply h_increasing _ _ (by linarith) (by linarith)
  exact ⟨by rw h1; exact h2, h3⟩

end function_properties_l506_506075


namespace monomial_addition_value_l506_506541

theorem monomial_addition_value (a b : ℤ) (h1 : a = 3) (h2 : 4 * b = 4) : (-1) ^ a * b ^ 4 = -1 := by
  sorry

end monomial_addition_value_l506_506541


namespace solve_tan_equation_l506_506366

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l506_506366


namespace find_x_l506_506387

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l506_506387


namespace f_inv_f_inv_17_l506_506660

noncomputable def f (x : ℝ) : ℝ := 4 * x - 3

noncomputable def f_inv (y : ℝ) : ℝ := (y + 3) / 4

theorem f_inv_f_inv_17 : f_inv (f_inv 17) = 2 := by
  sorry

end f_inv_f_inv_17_l506_506660


namespace zoo_feeding_ways_count_l506_506298

theorem zoo_feeding_ways_count : 
  ∀ (animals : Fin 6 → (string × string)), (animals 0).fst = "female_lion" → 
  (∀ i, (animals i).fst  ≠ (animals i + 1).fst) →
  (∀ i, (animals i).fst ∈ ["female_lion", "male_lion", "female_tiger", "male_tiger", 
                           "female_bear", "male_bear", "female_wolf", "male_wolf",
                           "female_elephant", "male_elephant", "female_rhino", "male_rhino"]) →
  ∃ n, n = 86400 :=
begin
  intros animals h0 h1 h2,
  sorry
end

end zoo_feeding_ways_count_l506_506298


namespace probability_point_between_p_and_q_l506_506554

-- Define the line equations for p and q
def line_p := λ x : ℝ, -x + 8
def line_q := λ x : ℝ, -3 * x + 8

-- Define the areas under lines p and q in the first quadrant
def area_under_p := (1 / 2) * 8 * 8
def area_under_q := (1 / 2) * (8 / 3) * 8

-- The probability that a point between p and q falls given the conditions
theorem probability_point_between_p_and_q : 
  (area_under_p - area_under_q) / area_under_p = (2 / 3) :=
by
  -- Use sorry to indicate the proof steps are omitted
  sorry

end probability_point_between_p_and_q_l506_506554


namespace last_digit_of_power_l506_506835

theorem last_digit_of_power (a n : ℕ) : 
  let b := a % 10 in 
  let cycle := [2, 4, 8, 6] in -- Assumption based on sample problem
  let k := (n % cycle.length) in  
  (a ^ n) % 10 = b ^ k % 10 :=
by
  sorry

end last_digit_of_power_l506_506835


namespace telephone_triangle_l506_506760

theorem telephone_triangle (G : SimpleGraph (Fin 2004)) (color : Sym2 (Fin 2004) → Fin 4)
  (hG : ∀ v w : Fin 2004, v ≠ w → ∃ c, color ⟦(v, w)⟧ = c)
  (h_colors : ∃ (c₁ c₂ c₃ c₄ : Fin 4), c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₁ ≠ c₄ ∧ 
    c₂ ≠ c₃ ∧ c₂ ≠ c₄ ∧ c₃ ≠ c₄ ∧ 
    ∃ v w : Fin 2004, color ⟦(v, w)⟧ = c₁ ∧ ∃ x y : Fin 2004, color ⟦(x, y)⟧ = c₂ ∧ 
    ∃ z t : Fin 2004, color ⟦(z, t)⟧ = c₃ ∧ ∃ u v : Fin 2004, color ⟦(u, v)⟧ = c₄) :
  ∃ S : Finset (Fin 2004), (S.card > 2 ∧ ∃ c₁ c₂ c₃, c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₂ ≠ c₃ ∧ 
    ∀ e ∈ G.edgeSet, e ∈ S → (color e = c₁ ∨ color e = c₂ ∨ color e = c₃)) :=
begin
  sorry
end

end telephone_triangle_l506_506760


namespace product_of_solutions_l506_506949

theorem product_of_solutions (x : ℂ) (a b : ℝ) (ha : a < 0) 
  (sol : x^4 = -16) (hx_form : x = a + b * complex.I) :
  ∃ (s1 s2 : ℂ), (s1^4 = -16 ∧ s2^4 = -16) ∧ s1.re < 0 ∧ s2.re < 0 ∧ s1 * s2 = 4 := 
by
  sorry

end product_of_solutions_l506_506949


namespace tan_sin_cos_eq_l506_506403

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l506_506403


namespace num_prime_factors_450_l506_506927

def is_prime (n : ℕ) : Prop := ¬∃m, m > 1 ∧ m < n ∧ m ∣ n

def prime_factors (n : ℕ) : set ℕ := {p | is_prime p ∧ p ∣ n}

theorem num_prime_factors_450 : finset.card (finset.filter (λ p, is_prime p) (finset.filter (λ p, p ∣ 450) finset.range 451)) = 3 := 
sorry

end num_prime_factors_450_l506_506927


namespace sqrt_of_mixed_number_l506_506842

theorem sqrt_of_mixed_number : sqrt (7 + 9 / 16) = 11 / 4 :=
by
  sorry

end sqrt_of_mixed_number_l506_506842


namespace max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c_l506_506630

theorem max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h₁ : a ≤ b) (h₂ : b ≤ c) (h₃ : c ≤ 2 * a) :
    b / a + c / b + a / c ≤ 7 / 2 := 
  sorry

end max_bound_of_b_over_a_plus_c_over_b_plus_a_over_c_l506_506630


namespace inequality_example_l506_506619

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (der : ∀ x, deriv f x = f' x)

theorem inequality_example (h : ∀ x : ℝ, f x > f' x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023)
:= sorry

end inequality_example_l506_506619


namespace monkey_food_l506_506677

theorem monkey_food (e m m' e' : ℚ) (h1 : m = 3 / 4 * e) (h2 : m' = m + 2) (h3 : e' = e - 2) (h4 : m' = 4 / 3 * e') :
  m + e = 14 := by
  sorry

end monkey_food_l506_506677


namespace greatest_value_of_m_l506_506029

noncomputable def verify_inequality (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) : Prop :=
  1 / a + 1 / b ≥ 4

theorem greatest_value_of_m (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  verify_inequality a b h_pos_a h_pos_b h_sum :=
begin
  sorry
end

end greatest_value_of_m_l506_506029


namespace find_first_term_l506_506197

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l506_506197


namespace hyperbola_asymptotes_l506_506849

theorem hyperbola_asymptotes :
  (∀ x y : ℝ, x^2 / 6 - y^2 / 3 = 0 → y = (sqrt 2 / 2) * x ∨ y = -(sqrt 2 / 2) * x) :=
by
  sorry

end hyperbola_asymptotes_l506_506849


namespace vacation_books_pair_count_l506_506934

/-- 
Given three distinct mystery novels, three distinct fantasy novels, and three distinct biographies,
we want to prove that the number of possible pairs of books of different genres is 27.
-/

theorem vacation_books_pair_count :
  let mystery_books := 3
  let fantasy_books := 3
  let biography_books := 3
  let total_books := mystery_books + fantasy_books + biography_books
  let pairs := (total_books * (total_books - 3)) / 2
  pairs = 27 := 
by
  sorry

end vacation_books_pair_count_l506_506934


namespace region_Y_tile_C_l506_506242

structure TileConfig where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

def TileA := TileConfig.mk 4 2 1 5
def TileB := TileConfig.mk 2 3 5 0
def TileC := TileConfig.mk 3 1 4 2
def TileD := TileConfig.mk 5 0 2 3

axiom adj_match (a b : TileConfig) : a.right = b.left

theorem region_Y_tile_C (X Y Z W : Type) (tX tY tZ tW : X → TileConfig) :
  (adj_match (tW X) (tY Y)) → tY Y = TileC :=
by
  sorry

end region_Y_tile_C_l506_506242


namespace light_bulb_ratio_l506_506605

theorem light_bulb_ratio (T U L G L1 : ℕ) (hT : T = 40) (hU : U = 16) (hL : L = 12) :
  (G = T - U - L) → (L1 = T - U) → (G : L1 = 1 : 1) := 
by
  intros hG hL1
  sorry

end light_bulb_ratio_l506_506605


namespace company_picnic_l506_506267

theorem company_picnic :
  (20 / 100 * (30 / 100 * 100) + 40 / 100 * (70 / 100 * 100)) / 100 * 100 = 34 := by
  sorry

end company_picnic_l506_506267


namespace root_division_7_pow_l506_506350

theorem root_division_7_pow : 
  ( (7 : ℝ) ^ (1 / 4) / (7 ^ (1 / 7)) = 7 ^ (3 / 28) ) :=
sorry

end root_division_7_pow_l506_506350


namespace oliver_remaining_money_l506_506643

noncomputable def initial_usd: ℝ := 40
noncomputable def initial_quarters: ℝ := 200
noncomputable def initial_eur: ℝ := 15
noncomputable def initial_dimes: ℝ := 100
noncomputable def initial_jpy: ℝ := 3000

noncomputable def usd_to_gbp: ℝ := 0.75
noncomputable def eur_to_gbp: ℝ := 0.85
noncomputable def usd_to_chf: ℝ := 0.90
noncomputable def eur_to_chf: ℝ := 1.05
noncomputable def jpy_to_cad: ℝ := 0.012
noncomputable def eur_to_aud: ℝ := 1.50

theorem oliver_remaining_money:
  let usd_after_sister := initial_usd - 5 - 10 in
  let quarters_after_sister := initial_quarters - 120 in
  let dimes_after_sister := initial_dimes - 50 in
  let eur_after_sister := initial_eur - 5 - 8 - 5 - 50 in
  let gbp_from_usd := 10 * usd_to_gbp in
  let gbp_from_eur := 5 * eur_to_gbp in
  let total_gbp := gbp_from_usd + gbp_from_eur - 3.50 in
  let chf_from_usd := 10 * usd_to_chf in
  let chf_from_eur := 5 * eur_to_chf in
  let total_chf := chf_from_usd + chf_from_eur - 2 in
  let total_cad := 2000 * jpy_to_cad in
  let total_aud := 8 * eur_to_aud - 7 in
  let jpy_after_convert := initial_jpy - 2000 - 500 in
  usd_after_sister = 20 ∧
  quarters_after_sister = 0 ∧
  dimes_after_sister = 0 ∧
  eur_after_sister = 2 ∧
  total_gbp = 8.25 ∧
  total_chf = 12.25 ∧
  total_cad = 24 ∧
  total_aud = 5 ∧
  jpy_after_convert = 0 := 
by sorry

end oliver_remaining_money_l506_506643


namespace additional_savings_l506_506779

-- Defining the conditions
def initial_price : ℝ := 50
def discount_one : ℝ := 6
def discount_percentage : ℝ := 0.15

-- Defining the final prices according to the two methods
def first_method : ℝ := (1 - discount_percentage) * (initial_price - discount_one)
def second_method : ℝ := (1 - discount_percentage) * initial_price - discount_one

-- Defining the savings for the two methods
def savings_first_method : ℝ := initial_price - first_method
def savings_second_method : ℝ := initial_price - second_method

-- Proving that the second method results in an additional 0.90 savings
theorem additional_savings : (savings_second_method - savings_first_method) = 0.90 :=
by
  sorry

end additional_savings_l506_506779


namespace calculate_b7_l506_506126

noncomputable def a_seq : ℕ → ℚ
| 0 := 3
| (n + 1) := a_seq n * a_seq n / b_seq n

noncomputable def b_seq : ℕ → ℚ
| 0 := 4
| (n + 1) := b_seq n * b_seq n / a_seq n

theorem calculate_b7 :
  ∃ (p q : ℤ), b_seq 7 = 4^p / 3^q ∧ p = 1094 ∧ q = 1093 := 
sorry

end calculate_b7_l506_506126


namespace maciek_total_purchase_cost_l506_506783

-- Define the cost of pretzels
def pretzel_cost : ℕ := 4

-- Define the cost of chips
def chip_cost : ℕ := pretzel_cost + (75 * pretzel_cost) / 100

-- Calculate the total cost
def total_cost : ℕ := 2 * pretzel_cost + 2 * chip_cost

-- Rewrite the math proof problem statement
theorem maciek_total_purchase_cost : total_cost = 22 :=
by
  -- Skip the proof
  sorry

end maciek_total_purchase_cost_l506_506783


namespace largest_median_l506_506165

theorem largest_median (a b c d e f : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 3) (h4 : d = 7) (h5 : e = 9) (h6 : f = 6) :
  ∃ x y z : ℕ, list.median (list.sort (a :: b :: c :: d :: e :: f :: x :: y :: z :: [] )) = 7 :=
by {
  sorry
}

end largest_median_l506_506165


namespace find_f_eight_l506_506899

def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g x = - g (-x)
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem find_f_eight {f : ℝ → ℝ}
  (h1 : is_odd (λ x, f (x - 1)))
  (h2 : is_even (λ x, f (x + 3)))
  (h3 : f 0 = 1) :
  f 8 = -1 :=
sorry

end find_f_eight_l506_506899


namespace quadratic_function_exists_l506_506864

theorem quadratic_function_exists (k : ℕ) (p : ℕ) (hk : 1 ≤ k ∧ k ≤ 9) (hp : 0 < p) :
  ∃ f : ℕ → ℕ, (f = λ x, (9 / k) * x^2 + 2 * x) ∧
  f (k / 9 * (10^p - 1)) = k / 9 * ((10^p - 1) * (10^p + 1)) :=
by
  sorry

end quadratic_function_exists_l506_506864


namespace find_x_between_0_and_180_l506_506425

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l506_506425


namespace probability_product_divisible_by_3_l506_506702

theorem probability_product_divisible_by_3 :
  let dice_probability := (2 / 3) ^ 5 ∈ rat.mk 32 243 in
  1 - dice_probability = rat.mk 211 243 :=
sorry

end probability_product_divisible_by_3_l506_506702


namespace product_plus_one_is_square_l506_506944

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : x * y + 1 = (x + 1) ^ 2 :=
by
  sorry

end product_plus_one_is_square_l506_506944


namespace value_at_zero_and_negative_l506_506886

noncomputable def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + x - 1 else 0

theorem value_at_zero_and_negative (f : ℝ → ℝ) (h_odd : odd_function f)
  (h_pos : ∀ x > 0, f x = x^2 + x - 1) :
  f 0 = 0 ∧ (∀ x < 0, f x = -x^2 + x + 1) :=
by
  have h0 : f 0 = 0, by sorry
  have hneg : ∀ x < 0, f x = -x^2 + x + 1, by sorry
  exact ⟨h0, hneg⟩

end value_at_zero_and_negative_l506_506886


namespace range_of_k_l506_506547

-- Given conditions
variables {k : ℝ} (h : ∃ (x y : ℝ), x^2 + k * y^2 = 2)

-- Theorem statement
theorem range_of_k : 0 < k ∧ k < 1 :=
by
  sorry

end range_of_k_l506_506547


namespace find_x_between_0_and_180_l506_506419

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l506_506419


namespace students_accounting_majors_l506_506958

theorem students_accounting_majors (p q r s : ℕ) 
  (h1 : 1 < p) (h2 : p < q) (h3 : q < r) (h4 : r < s) (h5 : p * q * r * s = 1365) : p = 3 := 
by 
  sorry

end students_accounting_majors_l506_506958


namespace minimum_weights_needed_l506_506717

theorem minimum_weights_needed : ∃ (n : ℕ), (∀ (weight : ℕ), (1 ≤ weight ∧ weight ≤ 100) → 
  ∃ (S : Finset ℕ), S.card = n ∧ (∀ s ∈ S, (∃ (k : ℕ), s = 2^k)) ∧ weight = S.sum) ∧ n = 7 :=
by
  sorry

end minimum_weights_needed_l506_506717


namespace measure_of_angle_A_length_of_b_l506_506890

-- Define the given conditions and parameters
variables {A B C a b c : ℝ}
variable h₁ : A + B + C = π       -- Internal angles of triangle sum to π
variable h₂ : a = 2              -- Given length a
variable h₃ : cos B = sqrt 3 / 3 -- Given cos B
def m : ℝ × ℝ := (sqrt 3, 1 - cos A)
def n : ℝ × ℝ := (sin A, -1)
variable h₄ : m.fst * n.fst + m.snd * n.snd = 0 -- Vectors are orthogonal

-- Prove the measure of angle A
theorem measure_of_angle_A : 
  A = 2 * π / 3 :=
sorry

-- Prove the length of b given conditions
theorem length_of_b : 
  b = 4 * sqrt 2 / 3 :=
sorry

end measure_of_angle_A_length_of_b_l506_506890


namespace inequality_ω_infinite_l506_506131

noncomputable def ω (n : ℕ) : ℕ := 
  if h : 1 < n then multiset.card (multiset.toFinset (uniqueFactorizationMonoid.factors n))
  else 0

theorem inequality_ω_infinite (n : ℕ) (h : 1 < n) : ∃ᶠ n in at_top, ω n < ω (n + 1) ∧ ω (n + 1) < ω (n + 2) :=
sorry

end inequality_ω_infinite_l506_506131


namespace coeff_a4_6790_l506_506935

theorem coeff_a4_6790 : 
  ∀ (a : ℕ → ℤ), (∑ k in Finset.range 9, a k * (1-x)^k) = (1 + x) ^ 8 + (2 + x) ^ 8 →
  0 = a 0 + a 1 * (1 - x) + a 2 * (1 - x) ^ 2 + a 3 * (1 - x) ^ 3 + a 4 * (1 - x) ^ 4 +
  a 5 * (1 - x) ^ 5 + a 6 * (1 - x) ^ 6 + a 7 * (1 - x) ^ 7 + a 8 * (1 - x) ^ 8 →
  a 4 = 6790 := sorry

end coeff_a4_6790_l506_506935


namespace sum_first_2n_plus_1_terms_eq_l506_506101

noncomputable def a_seq : ℕ → ℝ 
| 1 := 1
| n := if (2 ∣ n) then 2 ^ (n / 2) else 2 ^ ((n - 1) / 2)

def S (n : ℕ) : ℝ := (Finset.range (2 * n + 1)).sum (λ i, a_seq (i + 1))

theorem sum_first_2n_plus_1_terms_eq : 
    ∀ (n : ℕ), 
    S n = 2^(n+2) - 3 :=
by 
  sorry

end sum_first_2n_plus_1_terms_eq_l506_506101


namespace trivia_contest_victory_l506_506738

theorem trivia_contest_victory (n : ℕ) (n_gt_one : n > 1)
      (x y : ℕ →ℕ) (score_our_team correct_our_team incorrect_other_teams : ℕ → ℤ)
      (h1 : ∀ (i : ℕ), correct_our_team i = 0)
      (h2 : ∀ (i : ℕ), incorrect_other_teams i = -2)
      (h3 : ∀ (i : ℕ), score_our_team i = -1) :
    (x 0 > 2 ^ (n - 1)) →  x 0 = ∑ i in (range n), (if x i = correct_our_team i then 0 else y i ) → (y i = ∑ (j : ℕ), correct_our_team j - score_our_team j - incorrect_other_teams j - x i ) → sorry.

end trivia_contest_victory_l506_506738


namespace find_x_between_0_and_180_l506_506420

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l506_506420


namespace evaluate_powers_of_i_l506_506349
noncomputable def i : ℂ := complex.I

theorem evaluate_powers_of_i : 
  i^55 + i^555 + i^5 = -i := by
  sorry

end evaluate_powers_of_i_l506_506349


namespace concyclic_points_equivalence_l506_506558

-- Assume the necessary definitions for points and circles, and membership relation

def Point := Type
def Circle (P : Point) := Set Point
def concyclic {P : Type} (c : Circle P) : Prop := sorry
def reflects_across (P : Point) (L : Point) (line_segment : (Point × Point)) : Prop := sorry
def midpoint (A P : Point) (G : Point) : Prop := sorry
def altitude (A B C D : Point) : Prop := sorry
def interior_point (P : Point) (triangle : (Point × Point × Point)) : Prop := sorry

-- Given conditions
axiom (A B C D E F P L M N G : Point)
(h_triangle_ABC : ∃ (A B C : Point), A ≠ B ∧ B ≠ C ∧ C ≠ A)
(h_altitude_AD : altitude A B C D)
(h_altitude_BE : altitude B A C E)
(h_altitude_CF : altitude C A B F)
(h_interior_P : interior_point P (A, B, C))
(h_reflections : reflects_across P L (B, C) ∧ reflects_across P M (C, A) ∧ reflects_across P N (A, B))
(h_midpoint : midpoint A P G)

-- Prove the equivalence of concyclicity
theorem concyclic_points_equivalence :
  (concyclic (set_of {A, L, M, N})) ↔ (concyclic (set_of {D, E, F, G})) :=
sorry

end concyclic_points_equivalence_l506_506558


namespace oliver_score_l506_506151

theorem oliver_score 
  (n : ℕ) (m : ℕ) (mean_19 : ℝ) (mean_20 : ℝ) 
  (hn : n = 19) (hm : m = 20) (hmean_19 : mean_19 = 76) (hmean_20 : mean_20 = 78) :
  let total_19 := (n : ℝ) * mean_19,
      total_20 := (m : ℝ) * mean_20,
      oliver_score := total_20 - total_19
  in oliver_score = 116 :=
by
  sorry

end oliver_score_l506_506151


namespace arctan_sum_eq_pi_over_4_l506_506364

theorem arctan_sum_eq_pi_over_4 : 
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/47) = Real.pi / 4 :=
by
  sorry

end arctan_sum_eq_pi_over_4_l506_506364


namespace inheritance_problem_l506_506166

def wifeAmounts (K J M : ℝ) : Prop :=
  K + J + M = 396 ∧
  J = K + 10 ∧
  M = J + 10

def husbandAmounts (wifeAmount : ℝ) (husbandMultiplier : ℝ := 1) : ℝ :=
  husbandMultiplier * wifeAmount

theorem inheritance_problem (K J M : ℝ)
  (h1 : wifeAmounts K J M)
  : ∃ wifeOf : String → String,
    wifeOf "John Smith" = "Katherine" ∧
    wifeOf "Henry Snooks" = "Jane" ∧
    wifeOf "Tom Crow" = "Mary" ∧
    husbandAmounts K = K ∧
    husbandAmounts J 1.5 = 1.5 * J ∧
    husbandAmounts M 2 = 2 * M :=
by 
  sorry

end inheritance_problem_l506_506166


namespace triangle_formation_l506_506985

theorem triangle_formation (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : ∠ A = 120°) : 
  a + b > c → a + c > b → b + c > a := by
  sorry

end triangle_formation_l506_506985


namespace g_2010_equals_126_l506_506686

theorem g_2010_equals_126 (g : ℕ → ℕ) :
  (∀ x y m : ℕ, x > 0 → y > 0 → x + y = 2^m → g(x) + g(y) = (m + 1)^2) →
  g(2010) = 126 :=
begin
  sorry
end

end g_2010_equals_126_l506_506686


namespace total_number_of_people_l506_506822

theorem total_number_of_people (
    number_of_kids : ℕ := 2,
    cost_per_adult : ℕ := 8,
    total_cost : ℕ := 72
) : number_of_kids + (total_cost / cost_per_adult) = 11 := by
    sorry

end total_number_of_people_l506_506822


namespace solve_tan_equation_l506_506371

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l506_506371


namespace total_scoops_l506_506326

-- Definitions
def single_cone_scoops : ℕ := 1
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def waffle_bowl_scoops : ℕ := banana_split_scoops + 1

-- Theorem statement: Prove the total number of scoops is 10.
theorem total_scoops : single_cone_scoops + double_cone_scoops + banana_split_scoops + waffle_bowl_scoops = 10 :=
by
  -- Proof steps would go here
  sorry

end total_scoops_l506_506326


namespace maurice_rides_before_visit_l506_506639

variables (M : ℕ) (rides_with_Matt : ℕ := 8) (Matt_additional_rides : ℕ := 16) (total_rides_Matt : ℕ := 24)

-- Conditions
def Maurice_before_visit (M : ℕ) : Prop := M = 8
def Matt_total_rides (total_rides_Matt : ℕ) : Prop := total_rides_Matt = 3 * M
def Maurice_unique_horses_during_visit (M : ℕ) (rides_with_Matt : ℕ) : Prop := rides_with_Matt = M

-- Proof problem
theorem maurice_rides_before_visit (h1 : Maurice_before_visit M) 
                                   (h2 : Matt_total_rides total_rides_Matt) 
                                   (h3 : Maurice_unique_horses_during_visit M rides_with_Matt) : 
  M = 8 :=
by {
  rw [Maurice_before_visit, Matt_total_rides, Maurice_unique_horses_during_visit],
  sorry
}

end maurice_rides_before_visit_l506_506639


namespace twenty_two_oclock_not_eleven_PM_l506_506280

namespace ClockConversion

def to_12_hour_format (h : ℕ) : (ℕ × String) :=
  if h < 12 then (h, "AM")
  else (h - 12, "PM")

theorem twenty_two_oclock_not_eleven_PM :
  to_12_hour_format 22 ≠ (11, "PM") :=
by {
  -- proof here }

end twenty_two_oclock_not_eleven_PM_l506_506280


namespace lottery_probability_l506_506254

theorem lottery_probability : 
  let k := Nat.choose 74 5,
      total_combinations := Nat.choose 90 5,
      probability := (k: ℚ) / (total_combinations: ℚ)
  in probability = 17259390 / 43842110 :=
by 
  sorry

end lottery_probability_l506_506254


namespace tangent_parallel_at_1_and_3_intervals_of_increase_decrease_l506_506911

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 2) * a * x^2 - (2 * a + 1) * x + 2 * Real.log x

theorem tangent_parallel_at_1_and_3 (a : ℝ) :
  let f' := λ x => a * x - 2 * a - 1 + 2 / x in
  f' 1 = f' 3 → a = 2 / 3 := 
by {
  intros f';
  sorry
}

theorem intervals_of_increase_decrease (a : ℝ) :
  let f' := λ x => (a * x^2 - (2 * a + 1) * x + 2) / x in 
  (if a ≤ 0 then 
    (∀ x, 0 < x ∧ x < 2 → f' x > 0) ∧ (∀ x, x > 2 → f' x < 0) 
  else if 0 < a ∧ a < 1 / 2 then 
    (∀ x, 0 < x ∧ x < 2 → f' x > 0) ∧ (∀ x, x > 2 ∧ x < 1 / a → f' x < 0) ∧ (∀ x, x > 1 / a → f' x > 0)
  else if a = 1 / 2 then 
    (∀ x, x > 0 → f' x > 0)
  else 
    (∀ x, 0 < x ∧ x < 1 / a → f' x > 0) ∧ (∀ x, x > 1 / a ∧ x < 2 → f' x < 0) ∧ (∀ x, x > 2 → f' x > 0)) :=
by {
  intros f';
  sorry
}

end tangent_parallel_at_1_and_3_intervals_of_increase_decrease_l506_506911


namespace circumscribed_circle_area_ratio_l506_506799

theorem circumscribed_circle_area_ratio 
  (P : ℝ) 
  (square_side_length : ℝ := P / 4) 
  (hexagon_side_length : ℝ := P / 6) : 
  let C := π * (square_side_length * Real.sqrt 2 / 2)^2 
  let D := π * (hexagon_side_length)^2 
  in C / D = 9 / 8 := 
by 
  -- Proof will be provided here
  sorry

end circumscribed_circle_area_ratio_l506_506799


namespace first_term_of_geometric_series_l506_506221

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l506_506221


namespace cartesian_eq_C1_intersection_points_C3_C1_intersection_points_C3_C2_l506_506583

section CartesianAndPolarCurves

variable {t s θ : ℝ}
variable {x y : ℝ}

-- Define the parametric equations for C1 and C2
def parametric_C1 (t x y : ℝ) : Prop := x = (2 + t) / 6 ∧ y = Real.sqrt t
def parametric_C2 (s x y : ℝ) : Prop := x = -(2 + s) / 6 ∧ y = -Real.sqrt s

-- Define the polar equation for C3 and convert it to Cartesian form
def polar_C3_cartesian_eq : Prop := 2 * x - y = 0

-- Cartesian equation of C1
def cartesian_C1 (x y : ℝ) : Prop := y^2 = 6 * x - 2

-- Cartesian equation of C2
def cartesian_C2 (x y : ℝ) : Prop := y^2 = -6 * x - 2

-- Intersection points of C3 with C1
def intersection_C3_C1 (x y : ℝ) : Prop :=
  (y = 2 * x ∧ x = 1 / 2 ∧ y = 1) ∨ (y = 2 * x ∧ x = 1 ∧ y = 2)

-- Intersection points of C3 with C2
def intersection_C3_C2 (x y : ℝ) : Prop :=
  (y = 2 * x ∧ x = -1 / 2 ∧ y = -1) ∨ (y = 2 * x ∧ x = -1 ∧ y = -2)

-- Prove the problem
theorem cartesian_eq_C1 (t x y : ℝ) : parametric_C1 t x y → cartesian_C1 x y :=
  sorry

theorem intersection_points_C3_C1 (x y : ℝ) : cartesian_C1 x y → polar_C3_cartesian_eq → intersection_C3_C1 x y :=
  sorry

theorem intersection_points_C3_C2 (x y : ℝ) : cartesian_C2 x y → polar_C3_cartesian_eq → intersection_C3_C2 x y :=
  sorry

end CartesianAndPolarCurves

end cartesian_eq_C1_intersection_points_C3_C1_intersection_points_C3_C2_l506_506583


namespace matrix_not_invertible_iff_x_eq_13_div_7_l506_506456

theorem matrix_not_invertible_iff_x_eq_13_div_7 (x : ℝ) :
  ¬ invertible (matrix ![![2 + x, 9], ![4 - x, 5]]) ↔ x = 13 / 7 :=
by
  sorry

end matrix_not_invertible_iff_x_eq_13_div_7_l506_506456


namespace sum_of_ns_times_θs_mod_2010_eq_0_l506_506862

def θ (n : ℕ) : ℕ :=
  Nat.card {x // x < 2010 ∧ (x^2 - n) % 2010 = 0}

theorem sum_of_ns_times_θs_mod_2010_eq_0 :
  (∑ n in Finset.range 2010, n * θ n) % 2010 = 0 := 
sorry

end sum_of_ns_times_θs_mod_2010_eq_0_l506_506862


namespace roots_quartic_sum_l506_506136

theorem roots_quartic_sum (p q r : ℝ) 
  (h1 : p^3 - 2*p^2 + 3*p - 4 = 0)
  (h2 : q^3 - 2*q^2 + 3*q - 4 = 0)
  (h3 : r^3 - 2*r^2 + 3*r - 4 = 0)
  (h4 : p + q + r = 2)
  (h5 : p*q + q*r + r*p = 3)
  (h6 : p*q*r = 4) :
  p^4 + q^4 + r^4 = 18 := sorry

end roots_quartic_sum_l506_506136


namespace find_100th_digit_l506_506546

theorem find_100th_digit :
  let digits75_to_1 := String.join (List.map toString (List.range' 1 75).reverse)
  digits75_to_1.getChar 99 = '2' :=
by
  let digits75_to_1 := String.join (List.map toString (List.range' 1 75).reverse)
  have h_length : digits75_to_1.length >= 100 := by sorry
  have h_99th_digit : digits75_to_1.getChar 99 = '2' := by sorry
  exact h_99th_digit

end find_100th_digit_l506_506546


namespace unique_final_state_l506_506792

-- Define the notion of a column and the number of pebbles
def Column := Nat

-- Define the initial configuration with n pebbles in a single column
def initial_configuration (n : Nat) : List Nat :=
  [n]

-- Define the rule for moving a pebble
def can_move_pebble (c1 c2 : Column) : Prop :=
  c1 ≥ c2 + 2

-- Define what is a final configuration
def final_configuration (config : List Column) : Prop :=
  ∀ i, (i < config.length - 1) -> ¬ can_move_pebble (config.get! i) (config.get! (i + 1))

-- Define the unique final configuration
def unique_final_configuration (n : Nat) : List Column :=
  let k := Nat.floor ((-1 + Real.sqrt (1 + 8 * n.to_real)) / 2)
  let remain := n - (k * (k + 1)) / 2
  List.range k.succ.reverse ++ List.repeat (remain - k.succ) 1

-- The final theorem statement in Lean
theorem unique_final_state (n : Nat) :
  ∃ config : List Column, final_configuration config ∧ config = unique_final_configuration n :=
by
  sorry


end unique_final_state_l506_506792


namespace Gary_bound_longer_than_Harry_hop_l506_506536

theorem Gary_bound_longer_than_Harry_hop :
  let total_distance := 10560 -- feet
  let intervals := 50
  let Harry_hops := 30 * intervals
  let Gary_bounds := 8 * intervals
  let Harry_hop_length := total_distance / Harry_hops
  let Gary_bound_length := total_distance / Gary_bounds
  let difference := Gary_bound_length - Harry_hop_length
  difference = 19.36 :=
by
  let total_distance := 10560 -- feet
  let intervals := 50
  let Harry_hops := 30 * intervals
  let Gary_bounds := 8 * intervals
  let Harry_hop_length := total_distance / Harry_hops
  let Gary_bound_length := total_distance / Gary_bounds
  let difference := Gary_bound_length - Harry_hop_length
  have h : difference = 19.36 := by sorry
  exact h

end Gary_bound_longer_than_Harry_hop_l506_506536


namespace walter_zoo_time_l506_506251

def seals_time : ℕ := 13
def penguins_time : ℕ := 8 * seals_time
def elephants_time : ℕ := 13
def total_time_spent_at_zoo : ℕ := seals_time + penguins_time + elephants_time

theorem walter_zoo_time : total_time_spent_at_zoo = 130 := by
  -- Proof goes here
  sorry

end walter_zoo_time_l506_506251


namespace concyclic_points_l506_506120

open RealEuclideanSpace Geometry

variables {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α]

-- Definitions for Circle and Points
def circle (C : Set Point) := ∃ (O : Point) (r : ℝ), ∀ P ∈ C, dist O P = r
def midpoint_arc (C : Set Point) (B C A : Point) := ∀ P ∈ C, dist A B = dist A C ∧ ¬(A ∈ segment B C)

-- Chords and intersections
def chord (C : Set Point) (A D : Point) := {X : Point | X ∈ line A D ∧ X ∈ C}
def intersect (BC AD : Set Point) : Set Point := { X : Point | X ∈ BC ∧ X ∈ AD}

noncomputable def chord_intersect (BC AD AE : Set Point) : Set Point := intersect BC AD ∪ intersect BC AE

-- Definitions for concyclic conditions
def concyclic (D E F G : Point) := ∃ (O : Point) (r : ℝ), dist O D = r ∧ dist O E = r ∧ dist O F = r ∧ dist O G = r

theorem concyclic_points 
  (C : Set Point) (B C A D E F G : Point)
  (hC : circle C) 
  (hA_mid : midpoint_arc C B C A)
  (hAD : chord C A D)
  (hAE : chord C A E)
  (hF_G : chord_intersect (segment B C) (segment A D) (segment A E) = {F, G})
:
concyclic D E F G :=
sorry

end concyclic_points_l506_506120


namespace hypotenuse_length_l506_506358

noncomputable def hypotenuse_of_right_triangle (base height : ℝ) : ℝ := 
  sqrt (base ^ 2 + height ^ 2)

theorem hypotenuse_length (base area : ℝ) (h_base : base = 12) (h_area : area = 30) :
  hypotenuse_of_right_triangle base (2 * area / base) = 13 := 
by 
  -- Placeholder for the proof
  sorry

end hypotenuse_length_l506_506358


namespace geometric_series_first_term_l506_506205

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l506_506205


namespace Kevin_ends_with_54_cards_l506_506114

/-- Kevin starts with 7 cards and finds another 47 cards. 
    This theorem proves that Kevin ends with 54 cards. -/
theorem Kevin_ends_with_54_cards :
  let initial_cards := 7
  let found_cards := 47
  initial_cards + found_cards = 54 := 
by
  let initial_cards := 7
  let found_cards := 47
  sorry

end Kevin_ends_with_54_cards_l506_506114


namespace arrangement_count_l506_506692

-- Define the number of frogs of each color
def n_frogs : ℕ := 8
def n_green : ℕ := 3
def n_red : ℕ := 3
def n_blue : ℕ := 1
def n_yellow : ℕ := 1

-- Define the conditions
def green_frogs_cannot_be_adjacent (arrangement : list char) : Prop :=
  ∀ i, (i < arrangement.length - 1) →
    ((arrangement.nth i = some 'G' ∧ (arrangement.nth (i + 1) = some 'R' ∨ arrangement.nth (i + 1) = some 'B')) →
     false) ∧
    ((arrangement.nth i = some 'R' ∧ arrangement.nth (i + 1) = some 'G') →
     false) ∧
    ((arrangement.nth i = some 'B' ∧ arrangement.nth (i + 1) = some 'G') →
     false)

-- The number of ways to arrange the frogs given constraints and conditions
theorem arrangement_count :
  ∃ (arrangements : list (list char)), 
    (∀ arr ∈ arrangements, green_frogs_cannot_be_adjacent arr) ∧
    arrangements.length = 72 :=
sorry

end arrangement_count_l506_506692


namespace find_value_of_expression_l506_506092

variable {a : ℕ → ℤ}

-- Define arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variable (h1 : a 1 + 3 * a 8 + a 15 = 120)
variable (h2 : is_arithmetic_sequence a)

-- Theorem to be proved
theorem find_value_of_expression : 2 * a 6 - a 4 = 24 :=
sorry

end find_value_of_expression_l506_506092


namespace area_of_bounded_region_is_4_l506_506359

-- Define the bounding lines and curves in Cartesian coordinates
noncomputable def region_bounded_area : ℝ :=
  let curve1 := λ θ: ℝ, (2 / Real.sin θ, 2 / Real.sin θ * Real.sin θ) -- (r * cos(θ), r * sin(θ)) with r = 2 / sin(θ)
  let curve2 := λ θ: ℝ, (2 / Real.cos θ * Real.cos θ, 2 / Real.cos θ) -- (r * cos(θ), r * sin(θ)) with r = 2 / cos(θ)
  let line_y2 := 2
  let line_x2 := 2
  2 * 2

-- Statement of theorem indicating area
theorem area_of_bounded_region_is_4 : region_bounded_area = 4 := by
  -- The proof is omitted.
  sorry

end area_of_bounded_region_is_4_l506_506359


namespace problem_statement_l506_506902

noncomputable def a_n : ℕ → ℕ
| 0       := 6
| (n + 1) := 2 * (list.fin_range (n + 1)).sum (λ k, a_n k) + 6

theorem problem_statement :
  (a_n 1 = 18) ∧ (∀ n, a_n n = 2 * 3 ^ n) :=
begin
  sorry
end

end problem_statement_l506_506902


namespace _l506_506137

variable {α : Type*} [OrderedRing α] {A B C D E F : Point α}

/-- Conditions: D is the midpoint of BC, and E and F are on AB and AC such that DE = DF.
    Question: Prove AE + AF = BE + CF if and only if ∠ EDF = ∠ BAC. -/
def triangle_midpoint_theorem (D A B C E F : Point α) [T1 : Midpoint D B C] [T2 : OnLine E A B] [T3 : OnLine F A C]
  (h1 : Length (D, E) = Length (D, F)) :
  (Length (A, E) + Length (A, F) = Length (B, E) + Length (C, F)) ↔ (Angle D E F = Angle B A C) :=
sorry

end _l506_506137


namespace largest_integer_value_n_l506_506852

theorem largest_integer_value_n (n : ℤ) : 
  (n^2 - 9 * n + 18 < 0) → n ≤ 5 := sorry

end largest_integer_value_n_l506_506852


namespace solution_set_l506_506003

noncomputable def f : ℝ → ℝ := sorry
axiom f_def : f 1 = 2
axiom f_deriv : ∀ x : ℝ, deriv f x < 1

theorem solution_set (x : ℝ) : f (x^3) > x^3 + 1 ↔ x < 1 :=
begin
  sorry
end

end solution_set_l506_506003


namespace find_first_term_l506_506201

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l506_506201


namespace probability_different_colors_l506_506561

theorem probability_different_colors :
  let B := 1 -- number of black balls
  let S := 3 -- number of small balls
  let totalBalls := B + S -- total number of balls
  let totalWays := Nat.choose totalBalls 2 -- ways to choose 2 balls from totalBalls
  let differentColorWays := S -- ways to choose 1 black ball and 1 small ball
  let probability := differentColorWays / totalWays -- probability of different colors
  probability = 1 / 2 :=
by
  sorry

end probability_different_colors_l506_506561


namespace Alfred_gain_percent_l506_506307

variable (purchase_price repair_cost selling_price : ℝ)
variable (total_cost gain : ℝ)

-- Conditions
def purchase_price_def : purchase_price = 4700 := rfl
def repair_cost_def : repair_cost = 800 := rfl
def selling_price_def : selling_price = 5800 := rfl

-- Definitions based on the conditions
def total_cost_def : total_cost = purchase_price + repair_cost := 
  by rw [purchase_price_def, repair_cost_def]

def gain_def : gain = selling_price - total_cost := 
  by rw [selling_price_def, total_cost_def]

def gain_percent : ℝ := (gain / total_cost) * 100

-- Goal
theorem Alfred_gain_percent : gain_percent ≈ 5.45 := 
  by sorry

end Alfred_gain_percent_l506_506307


namespace tan_sin_cos_eq_l506_506446

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l506_506446


namespace find_x_value_l506_506413

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l506_506413


namespace Cartesian_equation_C1_intersection_C3_C1_intersection_C3_C2_l506_506579

-- Define C1 as a parametric curve
def C1 (t : ℝ) : ℝ × ℝ := ( (2 + t) / 6, real.sqrt t)

-- Define the Cartesian equation of C1
def Cartesian_C1 (x y : ℝ) : Prop := y ^ 2 = 6 * x - 2 ∧ y >= 0

-- Define C2 as a parametric curve
def C2 (s : ℝ) : ℝ × ℝ := ( -(2 + s) / 6, -real.sqrt s)

-- Define the Cartesian equation of C2
def Cartesian_C2 (x y : ℝ) : Prop := y ^ 2 = -6 * x - 2 ∧ y <= 0

-- Define the polar coordinate equation of C3
def Cartesian_C3 (x y : ℝ) : Prop := 2 * x - y = 0

-- Proving the correctness of converting parametric to Cartesian equation for C1
theorem Cartesian_equation_C1 {t : ℝ} : ∃ y, ∃ x, C1 t = (x, y) → Cartesian_C1 x y := 
sorry

-- Proving the correctness of intersection points of C3 with C1
theorem intersection_C3_C1 : 
  ∃ x1 y1, ∃ x2 y2, Cartesian_C3 x1 y1 ∧ Cartesian_C1 x1 y1 ∧ Cartesian_C3 x2 y2 ∧ Cartesian_C1 x2 y2 ∧ 
  ((x1 = 1 / 2 ∧ y1 = 1) ∧ (x2 = 1 ∧ y2 = 2)) :=
sorry

-- Proving the correctness of intersection points of C3 with C2
theorem intersection_C3_C2 : 
  ∃ x1 y1, ∃ x2 y2, Cartesian_C3 x1 y1 ∧ Cartesian_C2 x1 y1 ∧ Cartesian_C3 x2 y2 ∧ Cartesian_C2 x2 y2 ∧ 
  ((x1 = -1 / 2 ∧ y1 = -1) ∧ (x2 = -1 ∧ y2 = -2)) :=
sorry

end Cartesian_equation_C1_intersection_C3_C1_intersection_C3_C2_l506_506579


namespace leg_of_last_triangle_l506_506838

-- Define the sequence of triangles in a Lean structure
structure TriangleSequence :=
  (largest_hypotenuse : ℝ)
  (triangle_type : ℕ → triangle_type)
  (hypotenuse : ℕ → ℝ)
  (leg_length : ℕ → ℝ)

-- Define the sequence is 30-60-90 except the final is 45-45-90
inductive triangle_type | tri_30_60_90 | tri_45_45_90

-- The function representing the sequence
def seq_triangle : TriangleSequence :=
  { largest_hypotenuse := 16,
    triangle_type := λ n, if n < 3 then triangle_type.tri_30_60_90 else triangle_type.tri_45_45_90,
    hypotenuse := λ n, 
      match n with
      | 0 => 16
      | 1 => 8 * real.sqrt 3
      | 2 => 12
      | 3 => 6 * real.sqrt 3
      | _ => 0
      end,
    leg_length := λ n, 
      match n with
      | 0 => 8
      | 1 => 4 * real.sqrt 3
      | 2 => 6
      | 3 => 6 * real.sqrt (3 / 2)
      | _ => 0
      end }

-- The theorem to prove
theorem leg_of_last_triangle : seq_triangle.leg_length 3 = 6 * real.sqrt 6 / 2 :=
sorry

end leg_of_last_triangle_l506_506838


namespace path_length_B_travel_l506_506819

-- Define the structure and the problem conditions
noncomputable def path_length (BC : ℝ) : ℝ :=
  ∑ i in ({0, 1, 2, 3} : Finset ℕ), if i = 0 then 0 else (1/4) * 2 * Real.pi * BC

-- Define the theorem to be proven
theorem path_length_B_travel (BC : ℝ) (h : BC = 4 / Real.pi) : path_length BC = 6 := by
  sorry

end path_length_B_travel_l506_506819


namespace max_range_eq_a_min_range_eq_a_minus_b_num_sequences_max_range_eq_b_plus_1_l506_506747

-- Definitions based on the given conditions
variables (a b : ℕ) (h : a > b)

-- Proving the maximum possible range equals a
theorem max_range_eq_a : max_range a b = a :=
by sorry

-- Proving the minimum possible range equals a - b
theorem min_range_eq_a_minus_b : min_range a b = a - b :=
by sorry

-- Proving the number of sequences resulting in the maximum range equals b + 1
theorem num_sequences_max_range_eq_b_plus_1 : num_sequences_max_range a b = b + 1 :=
by sorry

end max_range_eq_a_min_range_eq_a_minus_b_num_sequences_max_range_eq_b_plus_1_l506_506747


namespace william_total_riding_hours_l506_506732

-- Define the conditions and the question as a Lean theorem statement
theorem william_total_riding_hours :
  ∃ (total_hours : ℕ), 
    total_hours = ((6 * 2) + (1.5 * 2) + ((6 / 2) * 2)) :=
begin
  use 21,
  -- Provide conditions
  have max_hours_days : 6 * 2 = 12 := by norm_num,
  have half_max_hours_days : (6 / 2) * 2 = 6 := by norm_num,
  have fractional_hours_days : (3 / 2) * 2 = 3 := by norm_num,
  
  -- Show the total riding hours.
  have total_hours : 21 = 12 + 3 + 6 := by norm_num,
  exact total_hours.symm
end

end william_total_riding_hours_l506_506732


namespace line_parallel_no_intersection_l506_506729

variables {Point Line : Type}

structure Plane :=
  (contains : Point → Prop)

structure Line :=
  (through : Point → Prop)

def parallel_to_plane (l : Line) (α : Plane) : Prop :=
  ∀ p, ¬ α.contains p → l.through p

def line_in_plane (l : Line) (α : Plane) : Prop :=
  ∀ p, l.through p → α.contains p

theorem line_parallel_no_intersection
  (l : Line) (α : Plane)
  (h1 : parallel_to_plane l α) :
  ∀ (m : Line), line_in_plane m α → ∀ p, ¬ (l.through p ∧ m.through p) :=
by
  intros m hm p hp
  sorry

end line_parallel_no_intersection_l506_506729


namespace angle_sum_l506_506167

-- Define the structures of points, lines, and angles
structure Point :=
(x : ℝ) (y : ℝ)

structure Segment :=
(p1 : Point) (p2 : Point)

def square (A B C D : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧
  ∠ (A B C) = 90 ∧ ∠ (B C D) = 90 ∧ ∠ (C D A) = 90 ∧ ∠ (D A B) = 90

def parallelogram (A B C D : Point) : Prop :=
  dist A B = dist C D ∧ dist B C = dist D A ∧ ∥ (A B C) = (B C D) ∧ ∥ (A B D) = (B D C)

-- Given points
variables {A B C D E F G H I X Y Z : Point}

-- Given conditions
def square_ABDE := square A B D E
def square_BCFG := square B C F G
def square_CAHI := square C A H I

def parallelogramDBGX := parallelogram D B G X
def parallelogramFCIY := parallelogram F C I Y
def parallelogramHAEZ := parallelogram H A E Z

-- To prove
theorem angle_sum : square_ABDE ∧ square_BCFG ∧ square_CAHI ∧ parallelogramDBGX ∧ parallelogramFCIY ∧ parallelogramHAEZ →
  ∠ (A Y B) + ∠ (B Z C) + ∠ (C X A) = 90 := by
  sorry

end angle_sum_l506_506167


namespace arithmetic_mean_of_two_digit_multiples_of_9_l506_506714

theorem arithmetic_mean_of_two_digit_multiples_of_9 :
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  M = 58.5 :=
by
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  sorry

end arithmetic_mean_of_two_digit_multiples_of_9_l506_506714


namespace chloe_price_for_half_dozen_l506_506831

theorem chloe_price_for_half_dozen
  (buying_price_per_dozen : ℤ)
  (profit : ℤ)
  (total_dozens_sold : ℤ)
  (total_cost : buy_cost = buying_price_per_dozen * total_dozens_sold)
  (total_revenue : total_revenue = total_cost + profit)
  (revenue_per_dozen : revenue_per_dozen = total_revenue / total_dozens_sold)
  (half_dozen_price : half_dozen_price = revenue_per_dozen / 2)
  (buying_price_per_dozen := 50)
  (profit := 500)
  (total_dozens_sold := 50) :
  half_dozen_price = 30 := 
begin
  sorry
end

end chloe_price_for_half_dozen_l506_506831


namespace lines_perpendicular_BL_AC_l506_506642

-- Given conditions for the problem
variables {A B C O K M N L : Point}
variables (h_circumcenter_O : IsCircumcenter O A B C)
variables (h_circumcenter_K : IsCircumcenter K A O C)
variables (h_meet_M : MeetsAt AB (Circumcircle A O C) M)
variables (h_meet_N : MeetsAt BC (Circumcircle A O C) N)
variables (h_reflect_L : Reflect K MN L)

-- To prove that BL and AC are perpendicular
theorem lines_perpendicular_BL_AC 
  (h_circumcenter_O : IsCircumcenter O A B C)
  (h_circumcenter_K : IsCircumcenter K A O C)
  (h_meet_M : MeetsAt AB (Circumcircle A O C) M)
  (h_meet_N : MeetsAt BC (Circumcircle A O C) N)
  (h_reflect_L : Reflect K MN L) : 
  Perpendicular (Line_through B L) (Line_through A C) := 
sorry

end lines_perpendicular_BL_AC_l506_506642


namespace trajectory_of_P_l506_506918

-- Define the initial conditions and the answer
variables {θ : ℝ} {x y : ℝ}

theorem trajectory_of_P (h₁ : x * cos θ + y * sin θ = 1)
                        (h₂ : x = -sin θ ∧ y = cos θ) : (x^2 + y^2 = 1) := by
  sorry

end trajectory_of_P_l506_506918


namespace price_after_discounts_l506_506815

noncomputable def final_price (initial_price : ℝ) : ℝ :=
  let first_discount := initial_price * (1 - 0.10)
  let second_discount := first_discount * (1 - 0.20)
  second_discount

theorem price_after_discounts (initial_price : ℝ) (h : final_price initial_price = 174.99999999999997) : 
  final_price initial_price = 175 := 
by {
  sorry
}

end price_after_discounts_l506_506815


namespace product_less_than_two_l506_506024

theorem product_less_than_two (n : ℕ) (a : Finₓ n → ℝ) (h₀ : ∀ i : Finₓ n, 0 < a i) (h₁ : (Finset.univ.sum a) ≤ 1 / 2) : 
  (∏ i, (1 + a i)) < 2 :=
by {
  sorry
}

end product_less_than_two_l506_506024


namespace total_students_l506_506959

variable (B G : ℕ) (ratio : B = 8 * G / 5) (girls : G = 120)

theorem total_students (h1 : B = 192) (h2 : G = 120) : B + G = 312 := by
  rw [h1, h2]
  rfl

end total_students_l506_506959


namespace marys_next_birthday_l506_506987

noncomputable def calculate_marys_age (d j s m TotalAge : ℝ) (H1 : j = 1.15 * d) (H2 : s = 1.30 * d) (H3 : m = 1.25 * s) (H4 : j + d + s + m = TotalAge) : ℝ :=
  m + 1

theorem marys_next_birthday (d j s m TotalAge : ℝ) 
  (H1 : j = 1.15 * d)
  (H2 : s = 1.30 * d)
  (H3 : m = 1.25 * s)
  (H4 : j + d + s + m = TotalAge)
  (H5 : TotalAge = 80) :
  calculate_marys_age d j s m TotalAge H1 H2 H3 H4 = 26 :=
sorry

end marys_next_birthday_l506_506987


namespace find_x_l506_506392

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l506_506392


namespace part_a_no_unique_sums_l506_506265

-- Conditions:
-- Define a cube having vertices labeled with 0 or 1
-- Define a function that calculates the sum of vertices on each face
-- Prove that no two faces can have unique sums 

def possible_vertex_values_a : set ℤ := {0, 1}

-- Given 6 faces and sums from 0 to 4
def sums_possible_a : set ℤ := {0, 1, 2, 3, 4}

theorem part_a_no_unique_sums :
    ∀ (faces : fin 6 → fin 5),
    (∀ i : fin 6, ∃ (sum : ℤ), sum ∈ sums_possible_a ∧ faces i = sum) →
    ∃ (i j : fin 6), i ≠ j ∧ faces i = faces j :=
by
    sorry

end part_a_no_unique_sums_l506_506265


namespace tan_sin_cos_eq_l506_506400

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l506_506400


namespace arithmetic_mean_of_two_digit_multiples_of_9_l506_506710

theorem arithmetic_mean_of_two_digit_multiples_of_9 : 
  let a1 := 18 in
  let an := 99 in
  let d := 9 in
  let n := (an - a1) / d + 1 in
  let S := n * (a1 + an) / 2 in
  (S / n : ℝ) = 58.5 :=
by
  let a1 := 18
  let an := 99
  let d := 9
  let n := (an - a1) / d + 1
  let S := n * (a1 + an) / 2
  show (S / n : ℝ) = 58.5
  sorry

end arithmetic_mean_of_two_digit_multiples_of_9_l506_506710


namespace problem_solution_l506_506192

-- Define the sequence and its properties
variable {a : ℕ → ℝ}  

-- Condition 1: The sequence consists of distinct positive numbers
def distinct_positive (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n > 0) ∧ (∀ m n : ℕ, m ≠ n → a m ≠ a n)

-- Condition 2: The reciprocals form an arithmetic sequence
def reciprocals_arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, (1 / a (n + 1)) - (1 / a n) = d

-- The main theorem stating the required equality
theorem problem_solution (d : ℝ) (h_distinct : distinct_positive a) 
  (h_arith : reciprocals_arithmetic a d) :
  (a 1 * a 2 + a 2 * a 3 + a 3 * a 4 + ... + a 2014 * a 2015) / (a 1 * a 2015) = 2014 :=
sorry

end problem_solution_l506_506192


namespace range_of_a_l506_506552

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, x + 9 > 2 * (x - 3) ∧ (2 * (x + 1) / 3 < x + a) ↔ 11 ≤ x ∧ x ≤ 14) ->
  -3 < a ∧ a ≤ -(8 / 3) :=
begin
  sorry
end

end range_of_a_l506_506552


namespace prob_teamA_champion_l506_506701

-- This will include necessary Lean libraries

variable (p : ℝ) -- Probability of winning a game

-- Initial conditions
axiom equal_prob : p = 0.5
axiom a_one_game : TeamA_needs_one_game
axiom b_two_games : TeamB_needs_two_games

-- The theorem stating the result
theorem prob_teamA_champion : (prob_teamA_champion) = 3/4 :=
by
  sorry

end prob_teamA_champion_l506_506701


namespace angle_between_lines_l506_506170

theorem angle_between_lines :
  ∃ θ ∈ Icc 0 Real.pi, ∀ l₁ l₂ : LinearEquations ℝ 2,
  (l₁.equation = (fun x y => x - 3*y + 3 = 0)) ∧
  (l₂.equation = (fun x y => x - y + 1 = 0)) →
  θ = Real.arctan (1 / 2) :=
by
  sorry

end angle_between_lines_l506_506170


namespace no_such_function_exists_l506_506161

theorem no_such_function_exists : ¬ ∃ f : ℝ → ℝ, (∀ x : ℝ, f (1 + f x) = 1 - x) ∧ (∀ x : ℝ, f (f x) = x) :=
by
  intro h
  let ⟨f, h1, h2⟩ := h
  have h3 : ∀ x : ℝ, f (1 - x) = 1 + f x := by sorry
  have h4 : f 1 = 1 + f 0 := by sorry
  have h5 : f 0 = 1 + f 1 := by sorry
  have contradiction : 0 = 2 := by
    calc
    0 = f 1 + f 0 - (f 1 + f 0) : by sorry
    ... = (1 + f 0 + f 1 + 1) - (f 1 + f 0) : by sorry
    ... = 2 : by sorry
  contradiction

end no_such_function_exists_l506_506161


namespace range_of_a_l506_506889

theorem range_of_a (a b c : ℝ) 
  (h1 : a^2 - b*c - 8*a + 7 = 0) 
  (h2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l506_506889


namespace part1_part2_l506_506532
-- Import Mathlib to ensure access to necessary mathematical functions and definitions

-- Part 1: Prove that if vector a is parallel to vector b, then λ = -2
theorem part1 (λ : ℝ) : (∀ (a b : ℝ × ℝ), 
  a = (1, -1) → b = (2, λ) → 
  ∃ c : ℝ, a = c • b) → λ = -2 := 
sorry

-- Part 2: Prove that if k • a + c is perpendicular to c, then k = -5
theorem part2 (k : ℝ) : (∀ (a c : ℝ × ℝ),
  a = (1, -1) → c = (3, 1) →
  (k • a + c).dot c = 0) → k = -5 := 
sorry

end part1_part2_l506_506532


namespace hours_of_lawyer_work_l506_506145

def mark_speed := 75
def speed_limit := 30
def base_fine := 50
def additional_penalty_rate := 2
def court_costs := 300
def lawyer_fee_per_hour := 80
def total_owed := 820

theorem hours_of_lawyer_work : 
  let speed_over_limit := mark_speed - speed_limit in
  let additional_penalty := speed_over_limit * additional_penalty_rate in
  let total_fine_before_doubling := base_fine + additional_penalty in
  let total_fine := total_fine_before_doubling * 2 in
  let total_costs := total_fine + court_costs in
  let lawyer_fee := total_owed - total_costs in
  lawyer_fee / lawyer_fee_per_hour = 3 :=
by 
  sorry

end hours_of_lawyer_work_l506_506145


namespace length_of_DB_l506_506977

-- Definitions of conditions
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables (angleABC angleADB : ℝ)
variables (AC AD : ℝ)

-- These conditions need to be specified based on the problem statement
def conditions : Prop := angleABC = 90 ∧ angleADB = 90 ∧ AC = 20 ∧ AD = 7

-- The proof statement
theorem length_of_DB (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (angleABC angleADB AC AD : ℝ) (h : conditions A B C D angleABC angleADB AC AD) : 
  ∃ DB : ℝ, DB = Real.sqrt 91 :=
sorry

end length_of_DB_l506_506977


namespace range_of_x_l506_506485

-- Definition of a decreasing function
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f y ≤ f x

-- Our main statement
theorem range_of_x (f : ℝ → ℝ) (h_dec : is_decreasing f) :
  ∀ x : ℝ, x > 1 → f(2 * x - 1) < f(1) :=
by
  intros x hx
  sorry

end range_of_x_l506_506485


namespace hyperbola_tangency_position_l506_506142

theorem hyperbola_tangency_position
  (F₁ F₂ M N P S : Point)
  (hyp_foci : is_foci F₁ F₂ Hyperbola)
  (hyp_vertices : is_vertices M N Hyperbola)
  (P_on_hyperbola : lies_on_hyperbola P Hyperbola)
  (S_tangency : is_tangency_point S (incircle (triangle P F₁ F₂)) (segment F₁ F₂)) :
  S = M ∨ S = N :=
sorry

end hyperbola_tangency_position_l506_506142


namespace truncated_cone_surface_area_l506_506571

variables {a α : ℝ}

-- Given conditions
def conditions (a α : ℝ) : Prop :=
  (a > 0) ∧ (0 < α) ∧ (α < π / 2)

-- Prove that the total surface area satisfies the given formula
theorem truncated_cone_surface_area (h : conditions a α) :
  let S_total := π * a^2 / (sin α)^2 * sin (α / 2 + π / 12) * cos (α / 2 - π / 12)
  in S_total = π * a^2 / (sin α)^2 * sin (α / 2 + π / 12) * cos (α / 2 - π / 12) :=
by
  sorry

end truncated_cone_surface_area_l506_506571


namespace algebraic_expression_l506_506874

def a (x : ℕ) := 2005 * x + 2009
def b (x : ℕ) := 2005 * x + 2010
def c (x : ℕ) := 2005 * x + 2011

theorem algebraic_expression (x : ℕ) : 
  a x ^ 2 + b x ^ 2 + c x ^ 2 - a x * b x - b x * c x - c x * a x = 3 :=
by
  sorry

end algebraic_expression_l506_506874


namespace num_correct_props_l506_506507

-- Definitions for the propositions
def prop1 (Q : Type) [Quadrilateral Q] : Prop :=
  ∃ (s1 s2 : Side), opposite s1 s2 ∧ equal s1 s2 ∧ ∃ (a1 a2 : Angle), opposite a1 a2 ∧ equal a1 a2

def prop2 (Q : Type) [Quadrilateral Q] : Prop :=
  ∃ (s1 s2 : Side), opposite s1 s2 ∧ equal s1 s2 ∧ ∃ (d1 d2 : Diagonal), bisect d1 d2

def prop3 (Q : Type) [Quadrilateral Q] : Prop :=
  ∃ (a1 a2 : Angle), opposite a1 a2 ∧ equal a1 a2 ∧ ∃ (d1 d2 : Diagonal), bisect d1 d2

def prop4 (Q : Type) [Quadrilateral Q] : Prop :=
  ∃ (a1 a2 : Angle), opposite a1 a2 ∧ equal a1 a2 ∧ ∃ (d1 d2 : Diagonal), bisected_by d1 d2

-- The main theorem statement
theorem num_correct_props (Q : Type) [Quadrilateral Q] :
  (prop1 Q ∨ prop2 Q ∨ prop3 Q ∨ prop4 Q) ∧ 
  (prop1 Q → ¬prop2 Q ∧ ¬prop3 Q ∧ ¬prop4 Q) ∧
  (prop2 Q → ¬prop1 Q ∧ ¬prop3 Q ∧ ¬prop4 Q) ∧
  (prop3 Q → ¬prop1 Q ∧ ¬prop2 Q ∧ ¬prop4 Q) ∧
  (prop4 Q → ¬prop1 Q ∧ ¬prop2 Q ∧ ¬prop3 Q) :=
sorry

end num_correct_props_l506_506507


namespace problem1_problem2_l506_506048

noncomputable def f (x a : ℝ) := x - (x^2 + a * x) / Real.exp x

theorem problem1 (x : ℝ) : (f x 1) ≥ 0 := by
  sorry

theorem problem2 (x : ℝ) : (1 - (Real.log x) / x) * (f x (-1)) > 1 - 1/(Real.exp 2) := by
  sorry

end problem1_problem2_l506_506048


namespace integer_terms_in_sequence_l506_506190

theorem integer_terms_in_sequence {a : ℕ} (h : a = 4860) :
  ∃ n, (a / (3 ^ n)) = 1 + ∏ i in finset.range n, (a ≠ 0 ∧ (a / (3 ^ i) : ℕ) ≠ 0) ∧ (a / (3 ^ (n + 1)) : ℕ) = 0 := 
sorry

end integer_terms_in_sequence_l506_506190


namespace chess_tournament_l506_506764

theorem chess_tournament :
  ∀ (n : ℕ) (total_games : ℕ), n = 12 → total_games = 132 → 
  (total_games / (n * (n - 1) / 2)) = 2 :=
by
  intros n total_games n_eq total_games_eq
  rw [n_eq, total_games_eq]
  rw [Nat.mul_comm 12 11]
  rw [Nat.div_eq_of_eq_mul_right (by norm_num : 0 < 66) (rfl : 132 = 66 * 2)]
  norm_num
  rw [Nat.div_self]
  exact (by norm_num : 66 > 0)

end chess_tournament_l506_506764


namespace number_of_tangent_lines_l506_506529
-- Import the full Mathlib.

-- Definition of the problem
def point := ℝ × ℝ

def dist_point_to_line (p : point) (a b c : ℝ) : ℝ :=
  (a * p.1 + b * p.2 + c).abs / Math.sqrt (a * a + b * b)

-- Given points A and B
def A : point := (0, 0)
def B : point := (2, 2)

-- Line l is defined by ax + by + c = 0
def is_tangent_line (a b c distance : ℝ) (p : point) : Prop :=
  dist_point_to_line p a b c = distance

-- The proof statement
theorem number_of_tangent_lines :
  ∃ l₁ l₂ : ℝ × ℝ × ℝ,
    (is_tangent_line l₁.1 l₁.2.1 l₁.2.2 1 A ∧ is_tangent_line l₁.1 l₁.2.1 l₁.2.2 2 B) ∧
    (is_tangent_line l₂.1 l₂.2.1 l₂.2.2 1 A ∧ is_tangent_line l₂.1 l₂.2.1 l₂.2.2 2 B) ∧
    l₁ ≠ l₂ ∧ ∀ l₃, (is_tangent_line l₃.1 l₃.2.1 l₃.2.2 1 A ∧ is_tangent_line l₃.1 l₃.2.1 l₃.2.2 2 B) → 
    l₃ = l₁ ∨ l₃ = l₂ :=
sorry

end number_of_tangent_lines_l506_506529


namespace maximum_value_of_f_value_of_A_l506_506046

noncomputable def f (x : ℝ) : ℝ := cos (2 * x - 4 * Real.pi / 3) + 2 * (cos x) ^ 2

theorem maximum_value_of_f : ∃ x, f x = 2 := sorry

theorem value_of_A (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : f (B + C) = 3 / 2) : A = Real.pi / 3 := sorry

end maximum_value_of_f_value_of_A_l506_506046


namespace eccentricity_of_hyperbola_l506_506905

section
variable (a b : ℝ) (A B F : ℝ × ℝ)

-- Conditions
def hyperbola (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def parabola (x y : ℝ) := y^2 = 4 * x
def parabola_focus : ℝ × ℝ := (1, 0)
def asymptote_1 (x y : ℝ) := y = (b / a) * x
def asymptote_2 (x y : ℝ) := y = - (b / a) * x
def point_A := (4 * a^2 / b^2, 4 * a / b)
def point_B := (4 * a^2 / b^2, -4 * a / b)
def cos_angle_AFB := -7 / 9

-- Theorem to prove
theorem eccentricity_of_hyperbola :
  cos_angle_AFB → 
  parabola_focus = F → 
  point_A = A → 
  point_B = B → 
  (a > 0) → 
  (b > 0) → 
  ∃ e, e = √3 :=
begin
  -- Proof goes here
  sorry
end
end

end eccentricity_of_hyperbola_l506_506905


namespace quadrilateral_is_rhombus_l506_506274

open EuclideanGeometry

noncomputable def isRhombus (B F I G : Point) : Prop :=
  dist B F = dist F I ∧ dist F I = dist I G ∧ dist I G = dist G B ∧
  ∃ M, midpoint B G M ∧ ⟂ M F


theorem quadrilateral_is_rhombus
  (A B C E D F G I : Point)
  (h1 : triangle A B C)
  (h2 : angleBisector A B E)
  (h3 : angleBisector C B D)
  (h4 : liesOnCircumcircle E A B C)
  (h5 : liesOnCircumcircle D A B C)
  (h6 : lineIntersect DE AB F)
  (h7 : lineIntersect DE BC G)
  (h8 : incenter I A B C)
  : isRhombus B F I G := sorry

end quadrilateral_is_rhombus_l506_506274


namespace minnie_lucy_time_difference_is_66_minutes_l506_506640

noncomputable def minnie_time_uphill : ℚ := 12 / 6
noncomputable def minnie_time_downhill : ℚ := 18 / 25
noncomputable def minnie_time_flat : ℚ := 15 / 15

noncomputable def minnie_total_time : ℚ := minnie_time_uphill + minnie_time_downhill + minnie_time_flat

noncomputable def lucy_time_flat : ℚ := 15 / 25
noncomputable def lucy_time_uphill : ℚ := 12 / 8
noncomputable def lucy_time_downhill : ℚ := 18 / 35

noncomputable def lucy_total_time : ℚ := lucy_time_flat + lucy_time_uphill + lucy_time_downhill

-- Convert hours to minutes
noncomputable def minnie_total_time_minutes : ℚ := minnie_total_time * 60
noncomputable def lucy_total_time_minutes : ℚ := lucy_total_time * 60

-- Difference in minutes
noncomputable def time_difference : ℚ := minnie_total_time_minutes - lucy_total_time_minutes

theorem minnie_lucy_time_difference_is_66_minutes : time_difference = 66 := by
  sorry

end minnie_lucy_time_difference_is_66_minutes_l506_506640


namespace range_of_a_correct_l506_506549

noncomputable def range_of_a (x y a : ℝ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (x + 2*y + 4 = 4*x*y) ∧ (xy + (1/2)*a^2*x + a^2*y + a - 17 ≥ 0)

theorem range_of_a_correct (a : ℝ) :
  (∀ x y : ℝ, range_of_a x y a) ↔ (a ≤ -3 ∨ a ≥ 5/2) :=
sorry

end range_of_a_correct_l506_506549


namespace walkway_area_l506_506662

/--
Tara has four rows of three 8-feet by 3-feet flower beds in her garden. The beds are separated
and surrounded by 2-feet-wide walkways. Prove that the total area of the walkways is 416 square feet.
-/
theorem walkway_area :
  let flower_bed_width := 8
  let flower_bed_height := 3
  let num_rows := 4
  let num_columns := 3
  let walkway_width := 2
  let total_width := (num_columns * flower_bed_width) + (num_columns + 1) * walkway_width
  let total_height := (num_rows * flower_bed_height) + (num_rows + 1) * walkway_width
  let total_garden_area := total_width * total_height
  let flower_bed_area := flower_bed_width * flower_bed_height * num_rows * num_columns
  total_garden_area - flower_bed_area = 416 :=
by
  -- Proof omitted
  sorry

end walkway_area_l506_506662


namespace matrix_condition_satisfied_l506_506853

open Matrix

def M : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 4], ![1, 2]]
def expected : Matrix (Fin 2) (Fin 2) ℝ := ![![6, 12], ![3, 6]]

noncomputable def M3 := M ⬝ M ⬝ M
noncomputable def M2 := M ⬝ M

theorem matrix_condition_satisfied : M3 - 3 • M2 + 2 • M = expected :=
by
  sorry

end matrix_condition_satisfied_l506_506853


namespace max_value_l506_506548

-- Define the function f(x)
def f (omega : ℝ) (x : ℝ) : ℝ :=
  3 * cos (omega * x + π / 6) - sin (omega * x - π / 3)

-- Define the conditions
def omega_gt_zero (omega : ℝ) : Prop :=
  omega > 0

def period (omega : ℝ) : Prop :=
  ∃ T > 0, ∀ x, f omega (x + T) = f omega x ∧ T = π

-- The statement we need to prove
theorem max_value (omega : ℝ) (hω_gt_zero : omega_gt_zero omega) (hperiod : period omega) :
  ∃ x ∈ Icc (0 : ℝ) (π / 2), f omega x = 2 * sqrt 3 :=
begin
  -- Proof (to be filled in)
  sorry
end

end max_value_l506_506548


namespace find_x_value_l506_506407

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l506_506407


namespace problem_l506_506675

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

theorem problem (x y : ℝ) (a : ℝ) (h1 : f(1) = 0)
  (h2 : ∀ x y, f(x + y) - f(y) = (x + 2 * y + 1) * x) :
    (f(0) = -2) ∧
    (f(x) = x^2 + x - 2) ∧
    (∀ x, 0 < x ∧ x < 2 -> f(x) > a * x - 5 -> a < 3) :=
begin
  sorry
end

end problem_l506_506675


namespace option_c_true_l506_506576

variables (a b g : Plane) (l : Line)

-- Definitions of the parameters
def distinct_planes (a b g : Plane) : Prop := a ≠ b ∧ b ≠ g ∧ a ≠ g
def line : Prop := True

-- Given the conditions
axiom h1 : distinct_planes a b g
axiom h2 : line

-- The statement to prove
theorem option_c_true :
  (l ∩ a) ≠ ∅ ∧ (l ∥ b) → (a ∩ b) ≠ ∅ :=
sorry

end option_c_true_l506_506576


namespace number_of_valid_numbers_l506_506247

def is_valid_digit (d : ℕ) : Prop := 
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def no_repeated_digits (l : List ℕ) : Prop := 
  l.nodup

def are_adjacent (d1 d2 : ℕ) (l : List ℕ) : Prop := 
  ∃ (n : ℕ), l = (d1 :: d2 :: List.nil) ∨ l = (d2 :: d1 :: List.nil) ∨ (∃ pre post, l = pre ++ [d1, d2] ++ post) ∨ (∃ pre post, l = pre ++ [d2, d1] ++ post)

def not_adjacent (d1 d2 : ℕ) (l : List ℕ) : Prop := 
  ∀ (pre post : List ℕ), (∅, ∅) ∈ l ∧ ¬ are_adjacent d1 d2 pre ++ List.nil ++ post

def count_valid_numbers : ℕ :=
  let digits := [1, 2, 3, 4, 5] in
  let perms := List.permutations digits in
  let valid_perms := perms.filter (λ l, no_repeated_digits l ∧ not_adjacent 1 3 l ∧ are_adjacent 2 5 l) in
  valid_perms.length

theorem number_of_valid_numbers : count_valid_numbers = 24 := 
  by sorry

end number_of_valid_numbers_l506_506247


namespace inscribed_square_ratios_l506_506962

theorem inscribed_square_ratios (a b c x y : ℝ) (h_right_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sides : a^2 + b^2 = c^2) 
  (h_leg_square : x = a) 
  (h_hyp_square : y = 5 / 18 * c) : 
  x / y = 18 / 13 := by
  sorry

end inscribed_square_ratios_l506_506962


namespace find_first_term_l506_506216

noncomputable def first_term : ℝ :=
  let a := 20 * (1 - (2 / 3)) in a

theorem find_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 20) 
  (h2 : a^2 / (1 - r^2) = 80) : 
  a = first_term :=
by
  sorry

end find_first_term_l506_506216


namespace num_diagonals_polygon_l506_506185

theorem num_diagonals_polygon (n : ℕ) (h : n = 100) : (n * (n - 3)) / 2 = 4850 := by
  rw [h]
  norm_num
  -- sorry

end num_diagonals_polygon_l506_506185


namespace Vasilisa_can_prevent_escape_l506_506249

structure Corridor where
  rooms : Fin 4
  passages : Fin 3

structure Guard where
  position : Bool -- True for West (W), False for East (E)

def initial_positions (g : Fin 3 → Guard) : Prop :=
  g 0 = Guard.mk true ∧ g 1 = Guard.mk false ∧ g 2 = Guard.mk true

def move (k : Fin 4) (g : Fin 3 → Guard) : Fin 4 × (Fin 3 → Guard) :=
  let new_k := if k.1 < 3 then ⟨k.1 + 1, k.2⟩ else k
  let new_g (i : Fin 3) := Guard.mk (¬(g i).position)
  (new_k, new_g)

def no_escape (k : Fin 4) (g : Fin 3 → Guard) : Prop :=
  ∀ (k' : Fin 4) (g' : Fin 3 → Guard),
    (k', g') = move k g →
    ¬(g' 0).position ∧ (g' 1).position ∧ ¬(g' 2).position

theorem Vasilisa_can_prevent_escape :
  ∃ (init_g : Fin 3 → Guard) (init_k : Fin 4),
    initial_positions init_g →
    ∀ k g, (init_k = k) ∧ (init_g = g) → no_escape k g :=
by
  sorry

end Vasilisa_can_prevent_escape_l506_506249


namespace min_sum_a_b_l506_506037

theorem min_sum_a_b {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b)
  (h₂ : 1/a + 9/b = 1) : a + b ≥ 16 := 
sorry

end min_sum_a_b_l506_506037


namespace circle_tangent_x_axis_at_origin_l506_506947

theorem circle_tangent_x_axis_at_origin (D E F : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 → x = 0 ∧ y = 0) ↔ (D = 0 ∧ F = 0 ∧ E ≠ 0) :=
sorry

end circle_tangent_x_axis_at_origin_l506_506947


namespace minimum_value_a5_a6_l506_506979

-- Defining the arithmetic geometric sequence relational conditions.
def arithmetic_geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * q) ∧ (a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6) ∧ (∀ n, a n > 0)

-- The mathematical problem to prove:
theorem minimum_value_a5_a6 (a : ℕ → ℝ) (q : ℝ) (h : arithmetic_geometric_sequence_condition a q) :
  a 5 + a 6 = 48 :=
sorry

end minimum_value_a5_a6_l506_506979


namespace runners_meet_time_l506_506990

theorem runners_meet_time :
  let laura_lap := 5
  let maria_lap := 8
  let charlie_lap := 10
  let zoe_lap := 2
  let start_time := (9, 0)
  let lcm_laps := Nat.lcm (Nat.lcm laura_lap maria_lap) (Nat.lcm charlie_lap zoe_lap)
  let time_minutes := 40
  lcm_laps = 40 ∧ Nat.add_mod (start_time.2 + time_minutes) 60 = 40 ∧ start_time.1 + (start_time.2 + time_minutes) / 60 = 9 + 1 :=
by
  sorry

end runners_meet_time_l506_506990


namespace students_and_ticket_price_l506_506663

theorem students_and_ticket_price (students teachers ticket_price : ℕ) 
  (h1 : students % 5 = 0)
  (h2 : (students + teachers) * (ticket_price / 2) = 1599)
  (h3 : ∃ n, ticket_price = 2 * n) 
  (h4 : teachers = 1) :
  students = 40 ∧ ticket_price = 78 := 
by
  sorry

end students_and_ticket_price_l506_506663


namespace product_plus_one_is_square_l506_506942

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : ∃ k : ℕ, x * y + 1 = k * k :=
by
  sorry

end product_plus_one_is_square_l506_506942


namespace pythagorean_triple_l506_506011

theorem pythagorean_triple {c a b : ℕ} (h1 : a = 24) (h2 : b = 7) (h3 : c = 25) : a^2 + b^2 = c^2 :=
by
  rw [h1, h2, h3]
  norm_num

end pythagorean_triple_l506_506011


namespace vasya_erased_digit_l506_506647

theorem vasya_erased_digit (d : ℕ) (digits : Finset ℕ) 
  (h1: digits = {0, 2, 4, 6, 8 ∧ d ∈ digits} ∧ d ≤ 8)
  (h2: ∀ x ∈ digits, x % 2 = 0) : 
  {d' | d' ∉ digits ∧ sum (digits.erase d') % 9 = 0 ∧ (digits.erase d').card = 4}
    = {2} :=
  by sorry

end vasya_erased_digit_l506_506647


namespace partition_powers_l506_506655

theorem partition_powers (n : ℕ) : 
  ∃ S T : Finset ℕ, S ∪ T = Finset.range (2 * n + 1) ∧ Disjoint S T ∧
    (∀ m : ℕ, m < n → (∑ k in S, k ^ m = ∑ k in T, k ^ m)) :=
by
  sorry

end partition_powers_l506_506655


namespace circumcenter_of_IABC_is_incenter_of_ABC_l506_506649

variables {A B C A1 B1 C1 : Point}

def is_incircle_center (I : Point) (A B C : Point) : Prop :=
  ∃ (I1 I2 I3 : Point), I = incenters_of_triangles I1 I2 I3

variables (AB1 AC1 CA1 CB1 BC1 BA1 : Length)
variables (IA IB IC : Point)

def are_incircles_centers (IA IB IC : Point) (A B C A1 B1 C1 : Point) : Prop :=
  is_incircle_center IA A B1 C1 ∧ is_incircle_center IB A1 B C1 ∧ is_incircle_center IC A1 B1 C

def circumcenter (IA IB IC : Point) : Point := sorry -- circumcenter function

def incenter (A B C : Point) : Point := sorry -- incenter function

theorem circumcenter_of_IABC_is_incenter_of_ABC :
  AB1 = AC1 →
  CA1 = CB1 →
  BC1 = BA1 →
  are_incircles_centers IA IB IC A B C A1 B1 C1 →
  circumcenter IA IB IC = incenter A B C :=
sorry

end circumcenter_of_IABC_is_incenter_of_ABC_l506_506649


namespace find_x_tan_identity_l506_506443

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l506_506443


namespace find_x_l506_506394

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l506_506394


namespace fiona_first_to_toss_eight_l506_506833

theorem fiona_first_to_toss_eight :
  (∃ p : ℚ, p = 49/169 ∧
    (∀ n:ℕ, (7/8:ℚ)^(3*n) * (1/8) = if n = 0 then (49/512) else (49/512) * (343/512)^n)) :=
sorry

end fiona_first_to_toss_eight_l506_506833


namespace find_n_l506_506248

theorem find_n (n : ℕ) (h1 : n > 13) (h2 : (12 : ℚ) / (n - 1 : ℚ) = 1 / 3) : n = 37 := by
  sorry

end find_n_l506_506248


namespace tan_sin_cos_eq_l506_506447

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l506_506447


namespace thirty_percent_less_than_90_eq_one_fourth_more_than_what_number_l506_506238

theorem thirty_percent_less_than_90_eq_one_fourth_more_than_what_number :
  ∃ (n : ℤ), (5 / 4 : ℝ) * (n : ℝ) = 90 - (0.30 * 90) ∧ n ≈ 50 := 
by
  -- Existence condition for n
  use 50
  -- Proof of equivalence (optional for statement)
  sorry

end thirty_percent_less_than_90_eq_one_fourth_more_than_what_number_l506_506238


namespace tan_sin_cos_eq_l506_506397

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l506_506397


namespace champion_sprinter_races_l506_506572

theorem champion_sprinter_races
  (total_sprinters : ℕ)
  (lanes : ℕ)
  (participants_eliminated : ℕ)
  (total_sprinters = 320)
  (lanes = 8)
  (participants_eliminated = 7) :
  ∃ races_needed : ℕ, races_needed = 46 :=
sorry

end champion_sprinter_races_l506_506572


namespace solve_tan_equation_l506_506373

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l506_506373


namespace solve_tan_equation_l506_506368

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l506_506368


namespace parabola_equation_length_PF_l506_506006

-- We define the given parabola equation as y^2 = 2px and all relevant definitions.
def given_parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x 

-- Names and conditions for points A and B and their distance, and the slope of the line
def line_slope (m : ℝ) : Prop := m = 2 * real.sqrt 2

def points_A_B (x1 y1 x2 y2 p : ℝ) : Prop := 
  y1^2 = 2 * p * x1 ∧ 
  y2^2 = 2 * p * x2 ∧ 
  abs (y1 - y2) / abs (x1 - x2) = 2 * real.sqrt 2 ∧
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 9

noncomputable def find_equation (p : ℝ) : ℝ :=
  if 9 = 2.25 * p then 8 else p -- this is just an implication

-- Point P, directrix and perpendicular line from point to directrix
def perpendicular (PC CF_p p : ℝ) : Prop :=
  PC = CF_p / real.sqrt 3 ∧ -- Given slope -sqrt(3)
  p = 8

-- Using existing definitions, the solution's conditions and our goal
theorem parabola_equation (p : ℝ) (y x x1 y1 x2 y2 : ℝ) (m : ℝ) :
  0 < p → 
  given_parabola y x p → 
  line_slope m →
  points_A_B x1 y1 x2 y2 p →
  find_equation p = 8 :=
sorry

theorem length_PF (p : ℝ) (PC CF_p : ℝ) :
  0 < p → 
  perpendicular PC CF_p p → 
  PC = 8 :=
sorry

end parabola_equation_length_PF_l506_506006


namespace sqrt_mixed_number_simplification_l506_506840

theorem sqrt_mixed_number_simplification :
  Real.sqrt (7 + 9 / 16) = 11 / 4 :=
by
  sorry

end sqrt_mixed_number_simplification_l506_506840


namespace line_through_points_eqn_l506_506851

theorem line_through_points_eqn :
  ∀ (x y : ℝ), (∃ (m b : ℝ), y = m * x + b ∧ (-1 = (-1:ℝ) → x = -(-1:ℝ) → y = 0) ∧ (-1 = 0 → x = 0 → y = 1)) ⟹ x - y + 1 = 0 :=
begin
  sorry
end

end line_through_points_eqn_l506_506851


namespace evaluate_expression_l506_506351

theorem evaluate_expression : 
  6 - 8 * (9 - 4^2) * 3 = 174 := by
  have h1 : 4^2 = 16 := by norm_num
  have h2 : 9 - 16 = -7 := by norm_num
  have h3 : 8 * (-7) * 3 = -168 := by norm_num
  calc
    6 - 8 * (9 - 4^2) * 3 = 6 - 8 * (-7) * 3 : by rw [h1, h2]
    ... = 6 - (-168) : by rw h3
    ... = 6 + 168 : by norm_num
    ... = 174 : by norm_num

end evaluate_expression_l506_506351


namespace find_a_and_min_value_minimum_value_l506_506516

open Real

namespace MathProof

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * log x

-- State the tangent line condition
theorem find_a_and_min_value (a : ℝ) :
  (∀ x, deriv (f a) x = 2 * x - (2 * a) / x) ∧ (2 - 2 * a = 2) :=
sorry

-- State the minimum value condition
theorem minimum_value (a : ℝ) :
  (∀ x, f a x = x^2) ∧ (1 ≤ 2) ∧ (∀ x, 1 ≤ x ∧ x ≤ 2 → f a x ≥ 1) :=
sorry

end MathProof

end find_a_and_min_value_minimum_value_l506_506516


namespace measure_of_angle_A_l506_506489

variable {a b c A : ℝ}

theorem measure_of_angle_A (h : (b - c)^2 = a^2 - b * c) : A = π / 3 :=
by
  have h1 : b^2 + c^2 - a^2 = b * c := by sorry
  have h2 : cos A = (b^2 + c^2 - a^2) / (2 * b * c) := by sorry
  have h3 : cos A = 1 / 2 := by sorry
  have h4 : A = π / 3 := by sorry
  exact h4

end measure_of_angle_A_l506_506489


namespace chicken_entree_cost_18_l506_506108

variable (total_guests : ℕ) (steak_cost : ℕ) (total_budget : ℕ)
variable (chicken_guests steak_guests : ℕ) (chicken_cost : ℕ)

-- Conditions from part a)
def conditions : Prop :=
  total_guests = 80 ∧
  steak_guests = 3 * chicken_guests ∧
  steak_cost = 25 ∧
  total_budget = 1860 ∧
  chicken_guests + steak_guests = total_guests

-- The theorem to prove the cost per chicken entree is $18
theorem chicken_entree_cost_18
  (h : conditions total_guests steak_cost total_budget chicken_guests steak_guests chicken_cost) :
  chicken_cost = 18 := by
  sorry

end chicken_entree_cost_18_l506_506108


namespace gcd_of_polynomials_is_one_l506_506363

def P1 : Polynomial ℤ := Polynomial.X ^ 2 + 5 * Polynomial.X + 13
def P2 : Polynomial ℤ := 3 * Polynomial.X ^ 3 - Polynomial.X ^ 2 + 6 * Polynomial.X + 94
def P3 : Polynomial ℤ := 5 * Polynomial.X ^ 2 - 8 * Polynomial.X + 201
def P4 : Polynomial ℤ := 7 * Polynomial.X ^ 4 - 21 * Polynomial.X ^ 2 + 3 * Polynomial.X + 481
def P5 : Polynomial ℤ := 9 * Polynomial.X ^ 3 - 17 * Polynomial.X + 759

def D1 : Polynomial ℤ := P2 - P1
def D2 : Polynomial ℤ := P3 - P1
def D3 : Polynomial ℤ := P4 - P1

theorem gcd_of_polynomials_is_one : Polynomial.gcd (Polynomial.gcd D1 D2) D3 = 1 :=
by { sorry }

end gcd_of_polynomials_is_one_l506_506363


namespace coefficient_x2_expansion_l506_506587

theorem coefficient_x2_expansion :
  let series := (List.range 11).map (fun n => (1 : ℕ) + (x : ℕ) ^ (n + 1)),
  (series.sum.coeff 2 = 220) :=
sorry

end coefficient_x2_expansion_l506_506587


namespace triplet_sums_to_two_l506_506343

theorem triplet_sums_to_two :
  (3 / 4 + 1 / 4 + 1 = 2) ∧
  (1.2 + 0.8 + 0 = 2) ∧
  (0.5 + 1.0 + 0.5 = 2) ∧
  (3 / 5 + 4 / 5 + 3 / 5 = 2) ∧
  (2 - 3 + 3 = 2) :=
by
  sorry

end triplet_sums_to_two_l506_506343


namespace remainder_is_three_l506_506134

theorem remainder_is_three (n : ℤ) (h1 : 0 < (47 / 5 : ℚ) * (4 / 47 + n / 141)) :
  let r := n % 15 in r = 3 :=
by
  sorry

end remainder_is_three_l506_506134


namespace greatest_median_l506_506271

theorem greatest_median (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t) (h5 : (k + m + r + s + t) = 80) (h6 : t = 42) : r = 17 :=
by
  sorry

end greatest_median_l506_506271


namespace regular_pyramid_angle_is_45_degrees_l506_506885

noncomputable def angle_between_lateral_edge_and_base (a : ℝ) (h : a = 2) : ℝ :=
  let l := a
  let d := real.sqrt (a^2 + a^2)
  let h := real.sqrt (l^2 - (d / 2)^2)
  let theta := real.arctan (h / (d / 2))
  theta

theorem regular_pyramid_angle_is_45_degrees (a : ℝ) (h : a = 2) :
  angle_between_lateral_edge_and_base a h = real.pi / 4 :=
sorry

end regular_pyramid_angle_is_45_degrees_l506_506885


namespace revenue_correct_l506_506772

noncomputable def totalRevenue (A B C : ℕ) (priceA priceB priceC : ℝ)
  (sold_fractionA sold_fractionB sold_fractionC : ℝ)
  (remainingA remainingB remainingC : ℕ) : ℝ :=
  let soldA := (sold_fractionA * A).toNat
  let soldB := (sold_fractionB * B).toNat
  let soldC := (sold_fractionC * C).toNat
  (soldA * priceA) + (soldB * priceB) + (soldC * priceC)

theorem revenue_correct :
  let A := (4 / 3 * 20).toNat -- since 3/4 A = 20
  let B := (3 / 2 * 30).toNat -- since 2/3 B = 30
  let C := (2 * 10).toNat -- since 1/2 C = 10
  totalRevenue A B C 3.5 4.5 5.5 (1/4) (1/3) (1/2) 20 30 10 = 147 := by
  sorry

end revenue_correct_l506_506772


namespace divide_rectangle_into_unique_smaller_rectangles_l506_506056

-- Define conditions
def big_rect (height width : ℕ) : Prop := (height = 13) ∧ (width = 7)
def unique_dimensions (rect_list : List (ℕ × ℕ)) : Prop :=
  ∀ (r1 r2 : ℕ × ℕ), r1 ≠ r2 → r1 ∈ rect_list → r2 ∈ rect_list → r1 ≠ r2

-- Define the main theorem
theorem divide_rectangle_into_unique_smaller_rectangles :
  ∃ (rect_list : List (ℕ × ℕ)), (big_rect 13 7) ∧ (List.length rect_list = 13) ∧ unique_dimensions(rect_list) :=
sorry

end divide_rectangle_into_unique_smaller_rectangles_l506_506056


namespace surface_generated_by_line_l506_506678

theorem surface_generated_by_line (L : ℝ → ℝ × ℝ × ℝ)
  (hL_parallel : ∀ (t : ℝ), (L t).2 = (L t).3)  -- L is parallel to the plane y = z
  (hL_intersects_parabola1 : ∃ s : ℝ, L s = (2 * s^2, 2 * s, 0))  -- meets parabola y^2 = 2x, z = 0
  (hL_intersects_parabola2 : ∃ t : ℝ, L t = (3 * t^2, 0, 3 * t))  -- meets parabola 3x = z^2, y = 0
  : ∀ x y z : ℝ, (x, y, z) ∈ (L '' set.univ) → x = (y - z) * (y / 2 - z / 3) :=
begin
  sorry
end

end surface_generated_by_line_l506_506678


namespace average_birth_rate_l506_506567

theorem average_birth_rate (B : ℕ) (death_rate : ℕ) (net_increase : ℕ) (seconds_per_day : ℕ) 
  (two_sec_intervals : ℕ) (H1 : death_rate = 2) (H2 : net_increase = 86400) (H3 : seconds_per_day = 86400) 
  (H4 : two_sec_intervals = seconds_per_day / 2) 
  (H5 : net_increase = (B - death_rate) * two_sec_intervals) : B = 4 := 
by 
  sorry

end average_birth_rate_l506_506567


namespace find_z_l506_506171

variable (x y z : ℝ)

theorem find_z (h1 : 12 * 40 = 480)
    (h2 : 15 * 50 = 750)
    (h3 : x + y + z = 270)
    (h4 : x + y = 100) :
    z = 170 := by
  sorry

end find_z_l506_506171


namespace three_digit_number_div_by_11_l506_506479

theorem three_digit_number_div_by_11 (x y z n : ℕ) 
  (hx : 0 < x ∧ x < 10) 
  (hy : 0 ≤ y ∧ y < 10) 
  (hz : 0 ≤ z ∧ z < 10) 
  (hn : n = 100 * x + 10 * y + z) 
  (hq : (n / 11) = x + y + z) : 
  n = 198 :=
by
  sorry

end three_digit_number_div_by_11_l506_506479


namespace smallest_number_of_states_l506_506669

theorem smallest_number_of_states :
  ∃ (S : ℤ), S = 9 ∧
    (∀ (cities : ℕ → ℕ × ℕ) (states : ℕ → ℤ) (state_capital : ℕ → ℕ × ℕ),
      (∀ i, 0 ≤ cities i ∧ cities i < (8, 8)) ∧
      (∀ i j, states i = states j → i = j) ∧
      (∀ n m, states n ≠ states m → (|states n - states m| = 1 ∨ |states n - states m| = 0)) ∧
      (∀ n, cities n ∈ {p | ∃ c, c = state_capital n ∧ (abs (fst p - fst c) = 1 ∨ abs (snd p - snd c) = 1))) ∧
      (0 < S) ∧
      (S ≤ 64 ∧ (64 % S = 0))) := sorry

end smallest_number_of_states_l506_506669


namespace find_φ_l506_506508

variable {f : ℝ → ℝ}
variable {φ : ℝ}

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x > f y

theorem find_φ (φ : ℝ) (h₀ : f = λ x, 2 * Real.sin (2 * x + φ + (Real.pi / 3))) 
               (h₁ : is_odd f) 
               (h₂ : is_decreasing_on f (Set.Icc 0 (Real.pi / 4))) :
  φ = (2 * Real.pi / 3) := sorry

end find_φ_l506_506508


namespace best_approx_value_l506_506589

variable {n : ℕ} (a : ℕ → ℝ)

theorem best_approx_value (h_nonzero : n > 0) :
  (∑ i in Finset.range n, (a i - (∑ i in Finset.range n, a i) / n) ^ 2) = 
  ∑ i in Finset.range n, (a i - ((∑ i in Finset.range n, a i) / n)) ^ 2 :=
sorry

end best_approx_value_l506_506589


namespace time_addition_sum_l506_506105

/-- Given the start time of 3:15:20 PM and adding a duration of 
    305 hours, 45 minutes, and 56 seconds, the resultant hour, 
    minute, and second values sum to 26. -/
theorem time_addition_sum : 
  let current_hour := 15
  let current_minute := 15
  let current_second := 20
  let added_hours := 305
  let added_minutes := 45
  let added_seconds := 56
  let final_hour := ((current_hour + (added_hours % 12) + ((current_minute + added_minutes) / 60) + ((current_second + added_seconds) / 3600)) % 12)
  let final_minute := ((current_minute + added_minutes + ((current_second + added_seconds) / 60)) % 60)
  let final_second := ((current_second + added_seconds) % 60)
  final_hour + final_minute + final_second = 26 := 
  sorry

end time_addition_sum_l506_506105


namespace Chvatal_1972_l506_506228

theorem Chvatal_1972 {n : ℕ} (a : ℕ → ℕ)
  (h1 : 0 ≤ a 1) 
  (h2 : ∀ i j, i ≤ j → a i ≤ a j)
  (h3 : ∀ i, a i < n)
  (h4 : 3 ≤ n) :
  (∀ i, i < n / 2 → (a i ≤ i → a (n - i) ≥ n - i)) ↔
  ∃ (G : SimpleGraph (Fin n)), G.isHamiltonian :=
sorry

end Chvatal_1972_l506_506228


namespace dihedral_angle_equivalence_l506_506487

namespace CylinderGeometry

variables {α β γ : ℝ} 

-- Given conditions
axiom axial_cross_section : Type
axiom point_on_circumference (C : axial_cross_section) : Prop
axiom dihedral_angle (α: ℝ) : Prop
axiom angle_CAB (β : ℝ) : Prop
axiom angle_CA1B (γ : ℝ) : Prop

-- Proven statement
theorem dihedral_angle_equivalence
    (hx : point_on_circumference C)
    (hα : dihedral_angle α)
    (hβ : angle_CAB β)
    (hγ : angle_CA1B γ):
  α = Real.arcsin (Real.cos β / Real.cos γ) :=
sorry

end CylinderGeometry

end dihedral_angle_equivalence_l506_506487


namespace find_n_l506_506457

theorem find_n (n : ℕ) :
  (∑ i in finset.range (n + 1), (↑i / (n + 1 : ℕ))) = 4 → n = 8 :=
by {
  sorry
}

end find_n_l506_506457


namespace multiple_of_regular_rate_is_1_5_l506_506636

-- Definitions
def hourly_rate := 5.50
def regular_hours := 7.5
def total_hours := 10.5
def total_earnings := 66.0
def excess_hours := total_hours - regular_hours
def regular_earnings := regular_hours * hourly_rate
def excess_earnings := total_earnings - regular_earnings
def rate_per_excess_hour := excess_earnings / excess_hours
def multiple_of_regular_rate := rate_per_excess_hour / hourly_rate

-- Statement of the problem
theorem multiple_of_regular_rate_is_1_5 : multiple_of_regular_rate = 1.5 :=
by
  -- Note: The proof is not required, hence sorry is used.
  sorry

end multiple_of_regular_rate_is_1_5_l506_506636


namespace smallest_five_digit_congruent_to_three_mod_seventeen_l506_506722

theorem smallest_five_digit_congruent_to_three_mod_seventeen :
  ∃ (n : ℤ), 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 3 ∧ n = 10012 :=
by
  sorry

end smallest_five_digit_congruent_to_three_mod_seventeen_l506_506722


namespace probability_one_piece_is_2m_probability_both_pieces_longer_than_2m_l506_506300

theorem probability_one_piece_is_2m (stick_length : ℕ) (piece_lengths : ℕ × ℕ) (h1 : stick_length = 6) (h2 : piece_lengths.1 + piece_lengths.2 = stick_length) (h3 : piece_lengths.1 > 0 ∧ piece_lengths.2 > 0) : 
  (if (piece_lengths.1 = 2 ∧ piece_lengths.2 ≠ 2) ∨ (piece_lengths.1 ≠ 2 ∧ piece_lengths.2 = 2) then 1 else 0) / 
  (if piece_lengths.1 > 0 ∧ piece_lengths.2 > 0 then 1 else 0) = 2 / 5 :=
sorry

theorem probability_both_pieces_longer_than_2m (stick_length : ℕ) (piece_lengths : ℕ × ℕ) (h1 : stick_length = 6) (h2 : piece_lengths.1 + piece_lengths.2 = stick_length) (h3 : piece_lengths.1 > 0 ∧ piece_lengths.2 > 0) :
  (if piece_lengths.1 > 2 ∧ piece_lengths.2 > 2 then 1 else 0) / 
  (if piece_lengths.1 > 0 ∧ piece_lengths.2 > 0 then 1 else 0) = 1 / 3 :=
sorry

end probability_one_piece_is_2m_probability_both_pieces_longer_than_2m_l506_506300


namespace max_regions_formula_l506_506021

def is_odd (n : ℕ) : Prop := n % 2 = 1

def color := Bool  -- Use Bool to represent two colors

structure Cell :=
  (x : ℕ)
  (y : ℕ)
  
structure Grid :=
  (cells : ℕ → ℕ → color)
  (n : ℕ)
  (h_n : n ≥ 3)
  (h_odd : is_odd n)

def is_adjacent (a b : Cell) (g : Grid) : Prop :=
  (g.cells a.x a.y = g.cells b.x b.y) ∧
  ((a.x = b.x ∧ abs (a.y - b.y) = 1) ∨
   (a.y = b.y ∧ abs (a.x - b.x) = 1))
  
def monochromatic_region (a : Cell) (g : Grid) : set Cell :=
  let adj := is_adjacent in
  { b | adj b a g }

def is_disconnected (regions : set (set Cell)) : Prop :=
  ∀ r1 r2 ∈ regions, r1 ∩ r2 = ∅

noncomputable def max_disconnected_monochromatic_regions (g : Grid) :=
  sorry  -- Function to find the maximum disconnected regions

theorem max_regions_formula (n : ℕ) (h_n : n ≥ 3) (h_odd : is_odd n) :
  ∃ g : Grid, g.n = n → max_disconnected_monochromatic_regions g =
  (n + 1) * (n + 1) / 4 + 1 :=
by sorry

end max_regions_formula_l506_506021


namespace scale_drawing_length_l506_506796

theorem scale_drawing_length (inches : ℝ) (feet_per_inch : ℝ) (total_length : ℝ) :
  inches = 5.4 → feet_per_inch = 1000 → total_length = inches * feet_per_inch → total_length = 5400 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rfl

end scale_drawing_length_l506_506796


namespace compute_custom_op_l506_506938

def custom_op (x y : ℤ) : ℤ := 
  x * y - y * x - 3 * x + 2 * y

theorem compute_custom_op : (custom_op 9 5) - (custom_op 5 9) = -20 := 
by
  sorry

end compute_custom_op_l506_506938


namespace exists_alpha_l506_506191

variable {a : ℕ → ℝ}

axiom nonzero_sequence (n : ℕ) : a n ≠ 0
axiom recurrence_relation (n : ℕ) : a n ^ 2 - a (n - 1) * a (n + 1) = 1

theorem exists_alpha (n : ℕ) : ∃ α : ℝ, ∀ n ≥ 1, a (n + 1) = α * a n - a (n - 1) :=
by
  sorry

end exists_alpha_l506_506191


namespace cotangent_median_triangle_l506_506181

theorem cotangent_median_triangle (A B C M3 : ℝ) 
  (cot_A cot_B cot_eps: ℝ) 
  (h1 : is_median C M3 A B) 
  (h2 : cot_eps = cot_erregular_median C M3 A B) 
  (h3 : cot_A = cotangent_angle A B C) 
  (h4 : cot_B = cotangent_angle B A C) : 
  cot_eps = (1 / 2) * |cot_A - cot_B| := 
sorry

end cotangent_median_triangle_l506_506181


namespace Marty_votes_l506_506146

theorem Marty_votes (total_count : ℕ) (biff_percentage undecided_percentage : ℝ)
  (total_count_eq : total_count = 200) 
  (biff_percentage_eq : biff_percentage = 0.45) 
  (undecided_percentage_eq : undecided_percentage = 0.08) : 
  let marty_percentage := 1 - (biff_percentage + undecided_percentage),
      marty_votes := marty_percentage * total_count in
  marty_votes = 94 :=
by
  unfold marty_percentage marty_votes
  rw [biff_percentage_eq, undecided_percentage_eq, total_count_eq]
  norm_num
  rw [←mul_assoc, mul_eq_mul_left_iff]
  norm_num
  sorry

end Marty_votes_l506_506146


namespace solution_eq_one_implies_m_nonzero_l506_506698

theorem solution_eq_one_implies_m_nonzero (m : ℚ) (x = 1) : (m * x = m) → m ≠ 0 :=
by
  intros h
  sorry

end solution_eq_one_implies_m_nonzero_l506_506698


namespace election_results_l506_506566

/--
In a national election, there are four candidates: A, B, C, and D.
Candidate A received 42% of the total valid votes,
Candidate B received 29%,
Candidate C received 16%,
and Candidate D received the remaining valid votes.
18% of the total votes were declared invalid,
and the total number of votes cast is 2,450,000.
Prove the number of valid votes polled in favor of each candidate.
-/
theorem election_results :
  let total_votes := 2450000 
  let invalid_votes := 0.18 * total_votes 
  let valid_votes := total_votes - invalid_votes
  let A_votes := 0.42 * valid_votes
  let B_votes := 0.29 * valid_votes
  let C_votes := 0.16 * valid_votes
  let D_votes := valid_votes - (A_votes + B_votes + C_votes)
  A_votes = 843780 ∧ B_votes = 582610 ∧ C_votes = 321440 ∧ D_votes = 261170 :=
by
  let total_votes := 2450000
  let invalid_votes := 0.18 * total_votes
  let valid_votes := total_votes - invalid_votes
  let A_votes := 0.42 * valid_votes
  let B_votes := 0.29 * valid_votes
  let C_votes := 0.16 * valid_votes
  let D_votes := valid_votes - (A_votes + B_votes + C_votes)
  have h_valid_votes : valid_votes = 2009000 := by decide
  have h_A_votes : A_votes = 843780 := by decide
  have h_B_votes : B_votes = 582610 := by decide
  have h_C_votes : C_votes = 321440 := by decide
  have h_D_votes : D_votes = 261170 := by decide
  exact And.intro h_A_votes (And.intro h_B_votes (And.intro h_C_votes h_D_votes))

end election_results_l506_506566


namespace Julie_initial_savings_l506_506988

theorem Julie_initial_savings (P r : ℝ) 
  (h1 : 100 = P * r * 2) 
  (h2 : 105 = P * (1 + r) ^ 2 - P) : 
  2 * P = 1000 :=
by
  sorry

end Julie_initial_savings_l506_506988


namespace geometric_sequence_roots_abs_diff_l506_506043

theorem geometric_sequence_roots_abs_diff (m n : ℝ) :
  let x1 := (1: ℝ) / 2
  let p := 2
  let x2 := x1 * p
  let x3 := x2 * p
  let x4 := x3 * p,
  (x1*x2 = 2) ∧ (x3*x4 = 2) ∧ (x1*x2*x3*x4 = 4) → 
  (|m - n| = 3/2) :=
sorry

end geometric_sequence_roots_abs_diff_l506_506043


namespace remainder_correct_l506_506856

noncomputable def P : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^6 
                                  + Polynomial.C 2 * Polynomial.X^5 
                                  - Polynomial.C 3 * Polynomial.X^4 
                                  + Polynomial.C 1 * Polynomial.X^3 
                                  - Polynomial.C 2 * Polynomial.X^2
                                  + Polynomial.C 5 * Polynomial.X 
                                  - Polynomial.C 1

noncomputable def D : Polynomial ℝ := (Polynomial.X - Polynomial.C 1) * 
                                      (Polynomial.X + Polynomial.C 2) * 
                                      (Polynomial.X - Polynomial.C 3)

noncomputable def R : Polynomial ℝ := 17 * Polynomial.X^2 - 52 * Polynomial.X + 38

theorem remainder_correct :
    ∀ (q : Polynomial ℝ), P = D * q + R :=
by sorry

end remainder_correct_l506_506856


namespace train_length_l506_506804

theorem train_length (v : ℝ) (t : ℝ) (p : ℝ) (L : ℝ) (h_v : v = 45) (h_t : t = 56) (h_p : p = 340) :
  L = 360 :=
by
  -- convert speed from km/hr to m/s
  let speed_m_per_s := v * (1000/3600)
  have h_speed : speed_m_per_s = 12.5, by sorry
  -- calculate the distance covered in time t
  let distance := speed_m_per_s * t
  have h_distance : distance = 700, by sorry
  -- length of train + length of platform equals the distance covered
  have h_L_plus_p := L + p = distance
  have h_L : L = 700 - 340, by sorry
  exact h_L

end train_length_l506_506804


namespace conquering_Loulan_necessary_for_returning_home_l506_506956

theorem conquering_Loulan_necessary_for_returning_home : 
  ∀ (P Q : Prop), (¬ Q → ¬ P) → (P → Q) :=
by sorry

end conquering_Loulan_necessary_for_returning_home_l506_506956


namespace find_x_value_l506_506415

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l506_506415


namespace how_many_tomatoes_picked_today_l506_506775

theorem how_many_tomatoes_picked_today
  (T_0 : ℕ)
  (P_y : ℕ)
  (T_l : ℕ)
  (T_1 : ℕ)
  (hT_0 : T_0 = 171)
  (hP_y : P_y = 134)
  (hT_l : T_l = 7)
  (hT_1 : T_1 = T_0 - P_y) :
  T_1 - T_l = 30 :=
by {
  rw [hT_0, hP_y] at hT_1,
  have h : T_1 = 37 := hT_1,
  rw h,
  rw hT_l,
  norm_num,
}

end how_many_tomatoes_picked_today_l506_506775


namespace ellipse_and_triangle_l506_506042

noncomputable def ellipse_equation : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧ 
    (∃ c : ℝ, c = sqrt 2 ∧ 
        (b^2 = a * 1 ∧ a^2 = b^2 + c^2) ∧ 
            ellipse_eq : ∀ x y : ℝ, 
                (x^2 / a^2 + y^2 / b^2 = 1 → x^2 / 4 + y^2 / 2 = 1))

noncomputable def maximum_triangle_area : Prop := 
    ∃ k m : ℝ, k ≠ 0 ∧ m ≠ 0 ∧ ∀ (F A B : ℝ × ℝ), 
        F = (sqrt 2, 0) →
        (∃ A B : ℝ × ℝ, 
            (A.1, A.2), (B.1, B.2) ∈ ellipse_equation ∧ 
            let AB_mid := (A.1 + B.1) / 2 in 
            let midpoint_condition := AB_mid + 2 * (k * AB_mid) = 0 in 
            ∃ area : ℝ,
                (triangle_area (F,A,B) = 1/2 * abs (4 * sqrt (6 - m^2) / 3 * abs (sqrt 2 + m) / sqrt 2) 
                → area = 8/3))

-- Attaching the two proofs under one
theorem ellipse_and_triangle : ellipse_equation ∧ maximum_triangle_area :=
begin
    split,
    { use [2, sqrt 2],
      split, linarith,
      split, linarith,
      split, linarith,
      use sqrt 2,
      split, refl,
      split,
      { linarith },
      { linarith },
      intros x y,
      intro h,
      field_simp at h,
      exact h },
    { use [1, sqrt 2],
      split,
      linarith,
      split,
      linarith,
      intros F A B H1,
      subst H1,
      use [(sqrt 2, 2)], -- points are needed here, only the logic of areas required
      ...
      sorry -- Full proof needed for remaining parts, placeholders here for maximum area reasoning
    }
end

end ellipse_and_triangle_l506_506042


namespace BC_CD_ratio_l506_506974

noncomputable def BC_div_CD (AB AD : ℝ) (angleA : ℝ) : ℝ :=
  if AB = 4 ∧ AD = 5 ∧ angleA = π / 3 then
    let R := sqrt (4^2 + 5^2 - 2 * 4 * 5 * cos (π / 3)) in
    let BC := sqrt (R^2 - 16) in
    let CD := sqrt (R^2 - 25) in
    BC / CD
  else
    0

theorem BC_CD_ratio :
  ∀ (AB AD : ℝ) (angleA : ℝ),
    AB = 4 → AD = 5 → angleA = π / 3 → 
    BC_div_CD AB AD angleA = 2 :=
by
  intros AB AD angleA hAB hAD hAngleA
  /-
  Proof to be provided. Here we assert that under the given conditions,
  the result will be BC / CD = 2.
  -/
  sorry

end BC_CD_ratio_l506_506974


namespace rest_time_each_10_miles_l506_506778

theorem rest_time_each_10_miles (v : ℝ) (d : ℝ) (total_time : ℝ) (R : ℝ) (R_needed : ℝ = 5):
  v = 10 ∧ d = 50 ∧ total_time = 320 ∧ (R * 4) + (d / v * 60) = total_time → R = 5 :=
by
  intros h
  sorry

end rest_time_each_10_miles_l506_506778


namespace find_x_value_l506_506408

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l506_506408


namespace probability_of_covering_all_languages_l506_506568

noncomputable def probability_covering_three_languages (total_students french_students spanish_students german_students german_in_other :
  ℕ) : ℚ :=
  if total_students = 40 ∧ french_students = 30 ∧ spanish_students = 32 ∧ german_students = 10 ∧ german_in_other = 26 then
    let common_students := french_students + spanish_students - total_students in
    let french_only := french_students - common_students in
    let spanish_only := spanish_students - common_students in
    let total_combinations := (Nat.choose total_students 2) in
    let unfavorables := (Nat.choose french_only 2) + (Nat.choose spanish_only 2) + if 2 <= (german_students - common_students + (common_students - german_in_other)) then (Nat.choose (german_students - common_students + (common_students - german_in_other)) 2) else 0 in
    (total_combinations - unfavorables) / total_combinations
  else
    0

theorem probability_of_covering_all_languages : probability_covering_three_languages 40 30 32 10 26 = 353 / 390 := by
  sorry

end probability_of_covering_all_languages_l506_506568


namespace initial_cost_of_article_l506_506809

variable (P : ℝ)

theorem initial_cost_of_article (h1 : 0.70 * P = 2100) (h2 : 0.50 * (0.70 * P) = 1050) : P = 3000 :=
by
  sorry

end initial_cost_of_article_l506_506809


namespace ff_x_plus_c_neither_even_nor_odd_l506_506129

-- Define the function f and its property of being odd
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

-- Define the constant c
variables (f : ℝ → ℝ) (c : ℝ)

-- Lean statement for the problem
theorem ff_x_plus_c_neither_even_nor_odd (h : odd_function f) :
  ¬(∀ x, f(f(x + c)) = f(f(-x + c))) ∧ ¬(∀ x, f(f(-x + c)) = -f(f(x + c))) :=
by {
  sorry
}

end ff_x_plus_c_neither_even_nor_odd_l506_506129


namespace tan_sin_cos_eq_l506_506396

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l506_506396


namespace valid_selling_price_range_l506_506770

theorem valid_selling_price_range (x : ℝ) : (60 ≤ x ∧ x ≤ 90) →
  (W : ℝ) (W = -x * x + 200 * x - 8400) →
  ((∃ x_max, (60 ≤ x_max ∧ x_max ≤ 90 ∧ W = 1500)) ∧ 
  (∀ W, W ≥ 1200 → 80 ≤ x ∧ x ≤ 90)) :=
by
  sorry

end valid_selling_price_range_l506_506770


namespace gain_percentage_l506_506073

theorem gain_percentage (C S : ℝ) (h : 80 * C = 25 * S) : 220 = ((S - C) / C) * 100 :=
by sorry

end gain_percentage_l506_506073


namespace find_original_number_l506_506291

theorem find_original_number 
  (A B C : ℕ)
  (h1 : 6000 + 100 * A + 10 * B + C = 6538)
  (h2 : 1000 * A + 100 * B + 10 * C + 6 = 4848 + 100 * A + 10 * B + C) :
  6000 + 100 * A + 10 * B + C = 6538 :=
begin
  sorry
end

end find_original_number_l506_506291


namespace trigonometric_identity_l506_506469

-- Define the conditions in the problem
def conditions (α : ℝ) : Prop :=
  (π / 2 < α) ∧ (α < π) ∧ (sin α + cos α = 1 / 5)

-- State the theorem using the conditions
theorem trigonometric_identity (α : ℝ) (h : conditions α) : 
  (2 / (cos α - sin α) = -10 / 7) :=
begin
  sorry
end

end trigonometric_identity_l506_506469


namespace frogs_arrangement_l506_506694

/-- There are seven distinguishable frogs; two are green, three are red, and two are blue. 
Green frogs cannot sit next to red frogs. The objective is to count the number of ways to arrange the frogs 
such that green frogs do not sit next to red frogs -/
theorem frogs_arrangement:
  let frogs : list ℕ := [2, 3, 2] in -- list of the number of green, red, and blue frogs respectively
  let forbidden : set (ℕ × ℕ) := {(0, 1), (1, 0)} in -- (green cannot sit next to red and vice versa)
  ∃ n : ℕ, n = 96 ∧ 
    -- n should be equal to 96 if we consider valid arrangements under the given conditions
    valid_arrangements frogs forbidden n :=
sorry

end frogs_arrangement_l506_506694


namespace intersection_height_l506_506079

-- Define the heights of the poles
def h1 : ℝ := 30
def h2 : ℝ := 70
-- Define the distance between the poles
def d : ℝ := 150

-- Define the equations of the lines
def line1 (x : ℝ) := -1/5 * x + h1
def line2 (x : ℝ) := 7/15 * x

-- Prove that the intersection height (y) is 21 feet
theorem intersection_height : ∃ x y : ℝ, line1 x = y ∧ line2 x = y ∧ y = 21 :=
by
  sorry

end intersection_height_l506_506079


namespace complex_magnitude_add_i_l506_506973

theorem complex_magnitude_add_i (z : ℂ) (h : z = -2 - i) : |z + i| = 2 :=
by 
  rw [h]
  simp
  sorry

end complex_magnitude_add_i_l506_506973


namespace tan_sin_cos_eq_l506_506449

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l506_506449


namespace probability_of_E_l506_506144

theorem probability_of_E :
  let A := 5
      E := 3
      I := 4
      O := 2
      U := 6
      total := A + E + I + O + U
  in E / total = 3 / 20 :=
by  sorry

end probability_of_E_l506_506144


namespace minimum_value_P_l506_506138

noncomputable theory

open Real

theorem minimum_value_P {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10)
  (h5 : a^2 + b^2 + c^2 = 9) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ 45.5 := 
sorry

end minimum_value_P_l506_506138


namespace part1_l506_506004

theorem part1 (f : ℝ → ℝ) (m n : ℝ) (cond1 : m + n > 0) (cond2 : ∀ x, f x = |x - m| + |x + n|) (cond3 : ∀ x, f x ≥ m + n) (minimum : ∃ x, f x = 2) :
    m + n = 2 := sorry

end part1_l506_506004


namespace find_y_l506_506295

noncomputable def x : ℝ := 3.3333333333333335

theorem find_y (y x: ℝ) (h1: x = 3.3333333333333335) (h2: x * 10 / y = x^2) :
  y = 3 :=
by
  sorry

end find_y_l506_506295


namespace first_term_of_geometric_series_l506_506225

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l506_506225


namespace range_x_minus_q_l506_506543

theorem range_x_minus_q (x q : ℝ) (h1 : |x - 3| > q) (h2 : x < 3) : x - q < 3 - 2*q :=
by
  sorry

end range_x_minus_q_l506_506543


namespace certain_number_is_1862_l506_506234

theorem certain_number_is_1862 (G N : ℕ) (hG: G = 4) (hN: ∃ k : ℕ, N = G * k + 6) (h1856: ∃ m : ℕ, 1856 = G * m + 4) : N = 1862 :=
by
  sorry

end certain_number_is_1862_l506_506234


namespace chandra_valid_pairings_l506_506830

def valid_pairings (num_bowls : ℕ) (num_glasses : ℕ) : ℕ :=
  num_bowls * num_glasses

theorem chandra_valid_pairings : valid_pairings 6 6 = 36 :=
  by sorry

end chandra_valid_pairings_l506_506830


namespace tangent_half_angle_identity_l506_506594

-- Definitions of the problem's components
variables (A B C O P : Point)
variable [Circumcircle : ⦃C O : Point, D : Line, IsDiameter C D⦄] -- Assume CD is diameter
variables (BA : Line, Intersects : IntersectsCircumcircle BA P)

-- Proving the required tangent half-angle identity
theorem tangent_half_angle_identity (h : triangle A B C) (h1 : circumcenter O A B C) (h2: diameter D C O) (h3: intersection D BA P) :
  tan (angle O A P / 2) * tan (angle O B P / 2) = (dist O P - dist O A) / (dist O P + dist O A) :=
sorry

end tangent_half_angle_identity_l506_506594


namespace find_x_l506_506390

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l506_506390


namespace volume_of_tetrahedron_l506_506091

theorem volume_of_tetrahedron
    (AB : ℝ)
    (area_ABC : ℝ)
    (area_ABD : ℝ)
    (angle_ABC_ABD : ℝ)
    (AB_eq_4 : AB = 4)
    (area_ABC_eq_20 : area_ABC = 20)
    (area_ABD_eq_16 : area_ABD = 16)
    (angle_ABC_ABD_eq_45 : angle_ABC_ABD = 45) :
    (\(\text{Volume of tetrahedron } ABCD = \frac{80\sqrt{2}}{3} \text{ cm}^3 \)) :=
sorry

end volume_of_tetrahedron_l506_506091


namespace find_x_tan_identity_l506_506438

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l506_506438


namespace arithmetic_mean_of_two_digit_multiples_of_9_l506_506712

theorem arithmetic_mean_of_two_digit_multiples_of_9 :
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  M = 58.5 :=
by
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  sorry

end arithmetic_mean_of_two_digit_multiples_of_9_l506_506712


namespace problem_invertible_matrix_exists_l506_506688

theorem problem_invertible_matrix_exists
  (A B S : Fin 3 → Matrix (Fin 2) (Fin 2) ℝ)
  (hA_invertible : ∀ i, Invertible (A i))
  (hB_invertible : ∀ i, Invertible (B i))
  (hS_invertible : ∀ i, Invertible (S i))
  (hA_no_common_eigenvector : ¬∃ v : Fin 2 → ℝ, ∀ i, (A i).mul_vec v = v)
  (hAi_eq_Si_inv_Bi_Si : ∀ i, A i = (S i)⁻¹ ⬝ (B i) ⬝ (S i))
  (hA_product : (A 0 ⬝ A 1 ⬝ A 2) = 1)
  (hB_product : (B 0 ⬝ B 1 ⬝ B 2) = 1) :
  ∃ (S : Matrix (Fin 2) (Fin 2) ℝ), Invertible S ∧ ∀ i, A i = S⁻¹ ⬝ B i ⬝ S :=
sorry

end problem_invertible_matrix_exists_l506_506688


namespace sum_of_smallest_integers_l506_506122

def num_divisors (n : ℕ) : ℕ := n.factors.prod_card

theorem sum_of_smallest_integers :
  (let valid_ns := {n | num_divisors n + num_divisors (n + 1) = 8}.to_finset.filter (λ n, n ≤ 32)
   in valid_ns.sum id) = 6 :=
by
  sorry

end sum_of_smallest_integers_l506_506122


namespace area_of_sector_l506_506742

-- Definitions from the problem conditions
def radius : ℝ := 5 -- radius in cm
def arc_length : ℝ := 3.5 -- arc length in cm

-- The proof problem statement
theorem area_of_sector : 
  let circumference := 2 * Real.pi * radius in
  let area_circle := Real.pi * radius^2 in
  let area_sector := (arc_length / circumference) * area_circle in
  area_sector = 8.75 := 
sorry

end area_of_sector_l506_506742


namespace determinant_of_B_l506_506617

variables {R : Type*} [Field R]
variables (x y : R)

def B : Matrix (Fin 2) (Fin 2) R :=
  ![![x, 2],
    ![-3, y]]

noncomputable def B_inv : Matrix (Fin 2) (Fin 2) R :=
  Matrix.inv B

theorem determinant_of_B :
  B + B_inv = 0 → Matrix.det B = 1 :=
begin
  sorry,
end

end determinant_of_B_l506_506617


namespace sum_ratio_arithmetic_sequence_l506_506015

theorem sum_ratio_arithmetic_sequence
  (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : ∀ n, S n = (n / 2) * (a 1 + a n))
  (h2 : ∀ k : ℕ, a (k + 1) - a k = a 2 - a 1)
  (h3 : a 4 / a 8 = 2 / 3) :
  S 7 / S 15 = 14 / 45 :=
sorry

end sum_ratio_arithmetic_sequence_l506_506015


namespace percentage_invalid_votes_l506_506574

-- Definitions for the conditions
def total_votes : ℕ := 560000
def votes_candidate_A : ℕ := 357000
def percentage_A : ℝ := 0.75

-- Definition for the percentage of total votes that were invalid
def invalid_votes_percentage (x : ℝ) : Prop :=
  ∃ (valid_votes : ℝ), 
    valid_votes = total_votes * (1 - x / 100) ∧ 
    votes_candidate_A = valid_votes * percentage_A

-- The proof statement
theorem percentage_invalid_votes : ∃ x : ℝ, invalid_votes_percentage x := 
begin
  use 15,
  split,
  { 
    calc
    total_votes * (1 - 15 / 100) 
        = 560000 * 0.85 : by norm_num
    ... = 476000 : by norm_num,
  },
  {
    calc
    476000 * 0.75 
        = 357000 : by norm_num,
  }
end

end percentage_invalid_votes_l506_506574


namespace max_range_eq_a_min_range_eq_a_minus_b_num_sequences_max_range_eq_b_plus_1_l506_506750

-- Definitions based on the given conditions
variables (a b : ℕ) (h : a > b)

-- Proving the maximum possible range equals a
theorem max_range_eq_a : max_range a b = a :=
by sorry

-- Proving the minimum possible range equals a - b
theorem min_range_eq_a_minus_b : min_range a b = a - b :=
by sorry

-- Proving the number of sequences resulting in the maximum range equals b + 1
theorem num_sequences_max_range_eq_b_plus_1 : num_sequences_max_range a b = b + 1 :=
by sorry

end max_range_eq_a_min_range_eq_a_minus_b_num_sequences_max_range_eq_b_plus_1_l506_506750


namespace circle_equation_intersecting_curves_l506_506969

theorem circle_equation_intersecting_curves :
  (∀ t a, (0 ≤ a ∧ a < π) → 
    (∃ (x y : ℝ), 
      x = 2 + t * real.cos a ∧ 
      y = 1 + t * real.sin a)) → 
  (∀ θ, (x y : ℝ), 
    x^2 + y^2 = 2 / (1 + real.cos θ ^ 2)) → 
  (∀ (x y : ℝ), 
    (x = 2 ∨ y - 1 = real.tan (π / 4) * (x - 2)) ∧ 
    x^2 + y^2 / 2 = 1) → 
  (∃ (x y : ℝ), 
    (x - 1/3)^2 + (y + 2/3)^2 = 8/9) := 
sorry

end circle_equation_intersecting_curves_l506_506969


namespace sum_of_squares_of_roots_l506_506052

theorem sum_of_squares_of_roots (a b c : ℚ) (h_quad : a ≠ 0) (h_eq : 10*a = a ∧ 15*a = b ∧ -20 = c) :
  let Δ := b^2 - 4 * a * c in
  let x1 := (-b + real.sqrt Δ) / (2 * a) in
  let x2 := (-b - real.sqrt Δ) / (2 * a) in
  Δ ≥ 0 → (x1^2 + x2^2 = 25/4) := 
by
  sorry

end sum_of_squares_of_roots_l506_506052


namespace tangent_line_equation_l506_506177

noncomputable def f (x : ℝ) : ℝ := x * exp (-x) + 2

noncomputable def f_prime (x : ℝ) : ℝ := (1 - x) * exp (-x)

def point_P : ℝ × ℝ := (0, 2)

theorem tangent_line_equation :
  let k := f_prime 0 in
  k = 1 ∧ (∀ x y, y - 2 = k * (x - 0) → x - y + 2 = 0) :=
sorry

end tangent_line_equation_l506_506177


namespace ratio_of_distances_l506_506828

def speed_car_A := 50
def time_car_A := 8
def distance_A := speed_car_A * time_car_A

def speed_car_B := 25
def time_car_B := 4
def distance_B := speed_car_B * time_car_B

theorem ratio_of_distances : (distance_A : ℝ) / distance_B = 4 :=
by
  have hA : distance_A = speed_car_A * time_car_A := rfl
  have hB : distance_B = speed_car_B * time_car_B := rfl
  rw [hA, hB]
  norm_num

end ratio_of_distances_l506_506828


namespace shopkeeper_price_increase_l506_506193

-- Defining the context and conditions
variables (P A : ℝ) -- Original price and amount
variables (P' A' : ℝ) -- Increased price and actual amount purchased
variables (increase_percentage : ℝ) -- The percentage increase we need to prove

-- Given conditions from the problem
def purchase_fraction := 0.7 -- The fraction of the amount actually purchased
def expenditure_increase := 1.125 -- Net difference in expenditure in percentage (112.5%)

-- The equations derived from the problem conditions
def original_expenditure := P * A
def new_expenditure := P' * A'
def condition_1 := A' = purchase_fraction * A
def condition_2 := new_expenditure = expenditure_increase * original_expenditure

-- The goal is to prove the percentage increase
theorem shopkeeper_price_increase 
  (h1 : A' = purchase_fraction * A)
  (h2 : P' * (purchase_fraction * A) = expenditure_increase * (P * A)) : 
  increase_percentage = 60.71 :=
by {
  -- Stating that the goal is 60.71%
  let goal_ratio := 1.6071,
  sorry
}

end shopkeeper_price_increase_l506_506193


namespace find_x_between_0_and_180_l506_506424

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l506_506424


namespace sqrt_mixed_number_simplification_l506_506841

theorem sqrt_mixed_number_simplification :
  Real.sqrt (7 + 9 / 16) = 11 / 4 :=
by
  sorry

end sqrt_mixed_number_simplification_l506_506841


namespace range_of_a_l506_506914

noncomputable def f (x : ℝ) : ℝ := exp x - exp (-x) + log (x + sqrt (x^2 + 1))

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → f (x^2 + 2) + f (-2 * a * x) ≥ 0) → a ≤ 3 / 2 :=
begin
  sorry
end

end range_of_a_l506_506914


namespace exists_set_S_l506_506836

theorem exists_set_S :
  ∃ S : Finset ℝ, S.card = 5783 ∧ ∀ a b ∈ S, ∃ c d ∈ S, c ≠ d ∧ a * b = c + d :=
by
  sorry

end exists_set_S_l506_506836


namespace find_x_l506_506393

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l506_506393


namespace find_x_tan_eq_l506_506426

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l506_506426


namespace geometric_sequences_l506_506527

variable (a_n b_n : ℕ → ℕ) -- Geometric sequences
variable (S_n T_n : ℕ → ℕ) -- Sums of first n terms
variable (h : ∀ n, S_n n / T_n n = (3^n + 1) / 4)

theorem geometric_sequences (n : ℕ) (h : ∀ n, S_n n / T_n n = (3^n + 1) / 4) : 
  (a_n 3) / (b_n 3) = 9 := 
sorry

end geometric_sequences_l506_506527


namespace find_AC_l506_506983

variables (A B C : Type) 

def is_right_triangle (A B C : Type) (angle_A : ℝ) (BC : ℝ) (tanC_equals_3sinC : Prop) : Prop :=
  angle_A = 90 ∧ BC = 25 ∧ tanC_equals_3sinC

theorem find_AC (A B C : Type) (h : is_right_triangle A B C 90 25 (tan C = 3 * sin C)):
  let AB := sorry in
  let AC := 25 / 3 in
  AC = 25 / 3 :=
sorry

end find_AC_l506_506983


namespace parabola_focus_distance_sum_eq_18_l506_506613

noncomputable def FA_magnitude (x₁ y₁ : ℝ) := real.sqrt ((x₁ - 3) ^ 2 + y₁ ^ 2) 
noncomputable def FB_magnitude (x₂ y₂ : ℝ) := real.sqrt ((x₂ - 3) ^ 2 + y₂ ^ 2)
noncomputable def FC_magnitude (x₃ y₃ : ℝ) := real.sqrt ((x₃ - 3) ^ 2 + y₃ ^ 2)

theorem parabola_focus_distance_sum_eq_18 
  (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ)
  (hx₁ : y₁^2 = 12 * x₁)
  (hx₂ : y₂^2 = 12 * x₂)
  (hx₃ : y₃^2 = 12 * x₃)
  (hcentroid : (x₁ + x₂ + x₃) / 3 = 3 ∧ (y₁ + y₂ + y₃) / 3 = 0)
  : FA_magnitude x₁ y₁ + FB_magnitude x₂ y₂ + FC_magnitude x₃ y₃ = 18 
:= sorry

end parabola_focus_distance_sum_eq_18_l506_506613


namespace distance_between_A_and_B_is_150_l506_506790

noncomputable def distance (x : ℝ) (y : ℝ) : Prop :=
  (x / 2 / 15 + x / 2 / 5 - 2 = y) ∧ (x = (y / 3) * 15 + (2 * y / 3) * 5)

theorem distance_between_A_and_B_is_150 (x y : ℝ) :
  distance(x, y) → x = 150 := by
  sorry

end distance_between_A_and_B_is_150_l506_506790


namespace units_digit_product_l506_506257

theorem units_digit_product : (3^5 * 2^3) % 10 = 4 := 
sorry

end units_digit_product_l506_506257


namespace tan_product_30_60_l506_506022

theorem tan_product_30_60 : 
  (1 + Real.tan (30 * Real.pi / 180)) * (1 + Real.tan (60 * Real.pi / 180)) = 2 + (4 * Real.sqrt 3) / 3 := 
  sorry

end tan_product_30_60_l506_506022


namespace ways_to_place_balls_with_exactly_two_matches_l506_506648

open Equiv

theorem ways_to_place_balls_with_exactly_two_matches :
  (∑ s in (Finset.univ : Finset (Fin 5).powerset), (s.card = 2 : Prop) ->
     (∃ f : (Fin 5) → (Fin 5), (∀ i ∈ s, f i = i) ∧ (∀ i ∉ s, f i ≠ i)) * fintype.card {f | ((∀ i, f i = i) : Prop) ∧ 
     (∀ i, ∃ j, i ≠ j → f i ≠ i) }) = 20 := sorry

end ways_to_place_balls_with_exactly_two_matches_l506_506648


namespace tangent_plane_equation_l506_506172

variables {R : ℝ}
variables {q r r1 : EuclideanSpace ℝ (Fin 3)}

-- Given definitions and conditions
def center_sphere (q : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

def point_on_sphere (r1 : EuclideanSpace ℝ (Fin 3)) (center : EuclideanSpace ℝ (Fin 3)) (radius : ℝ) : Prop :=
  (r1 - center) ⬝ (r1 - center) = radius^2

def point_on_tangent_plane (r : EuclideanSpace ℝ (Fin 3)) (r1 : EuclideanSpace ℝ (Fin 3)) (center : EuclideanSpace ℝ (Fin 3)) : Prop :=
  (r - r1) ⬝ (r1 - center) = 0

-- The main proof statement
theorem tangent_plane_equation (q r r1 : EuclideanSpace ℝ (Fin 3)) (R : ℝ)
  (h1 : point_on_sphere r1 q R)
  (h2 : point_on_tangent_plane r r1 q) :
  (r - q) ⬝ (r1 - q) = R^2 :=
sorry

end tangent_plane_equation_l506_506172


namespace indicator_properties_l506_506250

variable {Ω : Type*} [DecidableEq Ω]
variables (A B : Set Ω) (I : Set Ω → Ω → ℕ)

theorem indicator_properties:
  (∀ ω, I ∅ ω = 0) ∧ 
  (∀ ω, I univ ω = 1) ∧
  (∀ ω, I (compl A) ω = 1 - I A ω) ∧
  (∀ ω, I (A ∩ B) ω = I A ω * I B ω) ∧
  (∀ ω, I (A ∪ B) ω = I A ω + I B ω - I (A ∩ B) ω) ∧
  (∀ ω, I (A \ B) ω = I A ω * (1 - I B ω)) ∧
  (∀ ω, I (A ∆ B) ω = (I A ω - I B ω)^2) ∧ 
  (∀ ω, I (A ∆ B) ω = (I A ω + I B ω) mod 2) ∧
  (∀ ω, ∀ n, I (⋂ i in (Finset.range n), A i) ω = 1 - ∏ i in (Finset.range n), (1 - I (A i) ω)) ∧
  (∀ ω, ∀ n, I (⋃ i in (Finset.range n), A i) ω = ∏ i in (Finset.range n), (1 - I (A i) ω)) ∧
  (∀ ω, ∀ n, I (Finset.sum (Finset.range n) (λ i, A i)) ω = Finset.sum (Finset.range n) (λ i, I (A i) ω)) :=
sorry

end indicator_properties_l506_506250


namespace inverse_A_squared_determinant_A_inv_squared_l506_506063

-- Define the given inverse matrix
def A_inv : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3, 4],
  ![-2, -2]
]

-- Proof statement for the inverse of A^2
theorem inverse_A_squared : (A_inv ⬝ A_inv) = ![
  ![1, 4],
  ![-2, 0]
] := sorry

-- Proof statement for the determinant of (A^-1)^2
theorem determinant_A_inv_squared : Matrix.det (A_inv ⬝ A_inv) = 8 := sorry

end inverse_A_squared_determinant_A_inv_squared_l506_506063


namespace find_general_term_and_sum_l506_506625

noncomputable def geometric_seq_general_term (a₁ : ℕ) (q : ℕ) (n : ℕ) : ℕ :=
a₁ * q^(n - 1)

noncomputable def sum_first_n_terms (a₁ : ℕ) (q : ℕ) (n : ℕ) : ℕ :=
if q = 1 then n * a₁ else a₁ * (q^n - 1) / (q - 1)

theorem find_general_term_and_sum (a₁ q : ℕ) (h₁ : 2 * a₁ * q = a₁ + a₁ + a₁ * q)
  (h₂ : sum_first_n_terms a₁ q 3 = 42) :
  let a_n := geometric_seq_general_term a₁ q in
  a_n = λ n, 3 * 2^n ∧
  ∀ n, let b_n := n * a_n n / 2^n in
       let c_n := 1 / (b_n * (b_n + b_n / n + 1)) in
       (∑ i in finset.range n, c_n i) = n / (9 * (n + 1)) :=
by {
  sorry
}

end find_general_term_and_sum_l506_506625


namespace find_alpha_beta_l506_506891

theorem find_alpha_beta (α β : ℝ) 
  (h1 : cos (π / 2 - α) = sqrt 2 * cos (3 * π / 2 + β))
  (h2 : sqrt 3 * sin (3 * π / 2 - α) = - sqrt 2 * sin (π / 2 + β))
  (h3 : 0 < α)
  (h4 : α < π)
  (h5 : 0 < β)
  (h6 : β < π) :
  (α = π / 4 ∧ β = π / 6) ∨ (α = 3 * π / 4 ∧ β = 5 * π / 6) :=
sorry

end find_alpha_beta_l506_506891


namespace term_largest_binomial_coefficient_is_correct_term_largest_absolute_coefficient_is_correct_l506_506038

noncomputable def term_with_largest_binomial_coefficient (n : ℕ) : ℤ :=
  if n = 5 then -8064 else 0

noncomputable def term_with_largest_absolute_coefficient (n : ℕ) : ℕ → ℤ :=
  if n = 5 then λ r, if r = 3 then -15360 else 0 else λ _, 0

-- Given that for n = 5
axiom condition_sum_coefficients (n : ℕ) : 2^(2 * n) = 992 * (2^n)

theorem term_largest_binomial_coefficient_is_correct (n : ℕ) (h : n = 5) :
  term_with_largest_binomial_coefficient n = -8064 :=
by
  rw [term_with_largest_binomial_coefficient, h]
  simp

theorem term_largest_absolute_coefficient_is_correct (n : ℕ) (r : ℕ) (h : n = 5) (hr : r = 3) :
  term_with_largest_absolute_coefficient n r = -15360 :=
by
  rw [term_with_largest_absolute_coefficient, h, hr]
  simp

example : condition_sum_coefficients 5 := sorry

end term_largest_binomial_coefficient_is_correct_term_largest_absolute_coefficient_is_correct_l506_506038


namespace problem_statement_l506_506130

open Set

variable {n : ℕ}
variable (A B C : Set ℕ) (M : Set ℕ) 

def A_subset_M := ∀ a ∈ A, a ∈ M
def B_subset_M := ∀ b ∈ B, b ∈ M
def C_subset_M := ∀ c ∈ C, c ∈ M

def A_inter_B_inter_C_empty := A ∩ B ∩ C = ∅ 
def A_union_B_union_C_M := A ∪ B ∪ C = M 

theorem problem_statement (hne : n > 0) (A_subset : A_subset_M A M) (B_subset : B_subset_M B M) (C_subset : C_subset_M C M)
    (A_inter_B_C_empty : A_inter_B_inter_C_empty A B C)
    (A_union_B_C_M : A_union_B_union_C_M A B C) :
    ∃ a ∈ A, ∃ b ∈ B, ∃ c ∈ C, a + b = c :=
by
  sorry

end problem_statement_l506_506130


namespace people_in_each_van_l506_506681

theorem people_in_each_van
  (cars : ℕ) (taxis : ℕ) (vans : ℕ)
  (people_per_car : ℕ) (people_per_taxi : ℕ) (total_people : ℕ) 
  (people_per_van : ℕ) :
  cars = 3 → taxis = 6 → vans = 2 →
  people_per_car = 4 → people_per_taxi = 6 → total_people = 58 →
  3 * people_per_car + 6 * people_per_taxi + 2 * people_per_van = total_people →
  people_per_van = 5 :=
by sorry

end people_in_each_van_l506_506681


namespace sum_first_30_terms_l506_506500

-- The sum of the first n terms of a geometric sequence
def geometric_sequence_sum (n : ℕ) : ℝ 

-- Conditions
axiom S10 : geometric_sequence_sum 10 = 32
axiom S20 : geometric_sequence_sum 20 = 56

-- Prove that the sum of the first 30 terms is 74
theorem sum_first_30_terms : geometric_sequence_sum 30 = 74 := sorry

end sum_first_30_terms_l506_506500


namespace probability_of_EPC42_l506_506975

theorem probability_of_EPC42 :
  let vowel_set := {'A', 'E', 'I', 'O', 'U', 'Y'}
  let consonant_set := {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Z'}
  let total_plates := 6 * 21 * 20 * 100
  let epc42_combination := 1
  in total_plates = 252000 ∧ epc42_combination / total_plates = 1 / 252000 := by sorry

end probability_of_EPC42_l506_506975


namespace coins_left_l506_506606

-- Define the initial number of coins from each source
def piggy_bank_coins : ℕ := 15
def brother_coins : ℕ := 13
def father_coins : ℕ := 8

-- Define the number of coins given to Laura
def given_to_laura_coins : ℕ := 21

-- Define the total initial coins collected by Kylie
def total_initial_coins : ℕ := piggy_bank_coins + brother_coins + father_coins

-- Lean statement to prove
theorem coins_left : total_initial_coins - given_to_laura_coins = 15 :=
by
  sorry

end coins_left_l506_506606


namespace complement_union_l506_506054

open Set

def U := { x : ℕ | 0 ≤ x ∧ x < 5 }
def P := { 1, 2, 3 }
def Q := { 2, 4 }

theorem complement_union :
  (U \ P) ∪ Q = {2, 4} := by
  sorry

end complement_union_l506_506054


namespace find_a5_l506_506892

variable (a_n : ℕ → ℤ)
variable (d : ℤ)

def is_arithmetic_sequence (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

theorem find_a5 
  (h1 : is_arithmetic_sequence a_n d)
  (h2 : a_n 3 + a_n 8 = 22)
  (h3 : a_n 6 = 7) :
  a_n 5 = 15 :=
sorry

end find_a5_l506_506892


namespace Karl_total_distance_l506_506989

theorem Karl_total_distance :
  ∀ (miles_per_gallon tank_capacity miles_driven_fuel gallons_purchased final_tank_capacity),
  (miles_per_gallon = 40) →
  (tank_capacity = 12) →
  (miles_driven_fuel = 400) →
  (gallons_purchased = 10) →
  (final_tank_capacity = (3 / 4 * 12)) →
  let initial_fuel := tank_capacity in
  let fuel_consumed_first_leg := miles_driven_fuel / miles_per_gallon in
  let remaining_fuel_after_first_leg := initial_fuel - fuel_consumed_first_leg in
  let fuel_after_purchase := remaining_fuel_after_first_leg + gallons_purchased in
  let fuel_used_second_leg := fuel_after_purchase - final_tank_capacity in
  let distance_driven_second_leg := fuel_used_second_leg * miles_per_gallon in
  let total_distance := miles_driven_fuel + distance_driven_second_leg in
  total_distance = 520 :=
by
  intros miles_per_gallon tank_capacity miles_driven_fuel gallons_purchased final_tank_capacity
  intros h_mpg h_tc h_mdf h_gp h_ftc
  let initial_fuel := tank_capacity
  let fuel_consumed_first_leg := miles_driven_fuel / miles_per_gallon
  let remaining_fuel_after_first_leg := initial_fuel - fuel_consumed_first_leg
  let fuel_after_purchase := remaining_fuel_after_first_leg + gallons_purchased
  let fuel_used_second_leg := fuel_after_purchase - final_tank_capacity
  let distance_driven_second_leg := fuel_used_second_leg * miles_per_gallon
  let total_distance := miles_driven_fuel + distance_driven_second_leg
  -- here we need to prove total_distance = 520
  sorry

end Karl_total_distance_l506_506989


namespace sixth_factor_of_N_l506_506683

theorem sixth_factor_of_N (N : ℕ) (h : (∀ a b c d e f : ℕ, 
  {a, b, c, d, e, f} = {1, N, x1, x2, x3, x4} -> a * b * c * d * e = 6075 -> f = 15)) : 
  (∃ (a b c d e f : ℕ), {a, b, c, d, e, f} = {1, N, x1, x2, x3, x4} /\ a * b * c * d * e = 6075 -> f = 15) :=
sorry

end sixth_factor_of_N_l506_506683


namespace solution_quadratic_l506_506252

open Complex

theorem solution_quadratic :
  let a := 1 / 5
  let b := sqrt 416 / 10
  a + b^2 = 109 / 25 :=
begin
  sorry,
end

end solution_quadratic_l506_506252


namespace greatest_roses_for_680_l506_506270

/--
Greatest number of roses that can be purchased for $680
given the following costs:
- $4.50 per individual rose
- $36 per dozen roses
- $50 per two dozen roses
--/
theorem greatest_roses_for_680 (cost_individual : ℝ) 
  (cost_dozen : ℝ) 
  (cost_two_dozen : ℝ) 
  (budget : ℝ) 
  (dozen : ℕ) 
  (two_dozen : ℕ) 
  (total_budget : ℝ) 
  (individual_cost : ℝ) 
  (dozen_cost : ℝ) 
  (two_dozen_cost : ℝ) 
  (roses_dozen : ℕ) 
  (roses_two_dozen : ℕ):
  individual_cost = 4.50 → dozen_cost = 36 → two_dozen_cost = 50 →
  budget = 680 → dozen = 12 → two_dozen = 24 →
  (∀ n : ℕ, n * two_dozen_cost ≤ budget → n * two_dozen + (budget - n * two_dozen_cost) / individual_cost ≤ total_budget) →
  total_budget = 318 := 
by
  sorry

end greatest_roses_for_680_l506_506270


namespace find_first_term_geometric_series_l506_506208

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l506_506208


namespace equation_cannot_be_parabola_l506_506176

variables {α : Type*} [LinearOrder α] [SinCosClass α] 
noncomputable def equation_represents_parabola (α : ℝ) (x y : ℝ) : Prop :=
  x^2 * sin α + y^2 * cos α = 1

theorem equation_cannot_be_parabola (α : ℝ) (x y : ℝ) :
  ¬ (∀ α, ∃ x y : ℝ, equation_represents_parabola α x y → is_parabola (x^2 * sin α + y^2 * cos α = 1)) :=
sorry

end equation_cannot_be_parabola_l506_506176


namespace stratified_sample_correct_l506_506563

variable (popA popB popC : ℕ) (totalSample : ℕ)

def stratified_sample (popA popB popC totalSample : ℕ) : ℕ × ℕ × ℕ :=
  let totalChickens := popA + popB + popC
  let sampledA := (popA * totalSample) / totalChickens
  let sampledB := (popB * totalSample) / totalChickens
  let sampledC := (popC * totalSample) / totalChickens
  (sampledA, sampledB, sampledC)

theorem stratified_sample_correct
  (hA : popA = 12000) (hB : popB = 8000) (hC : popC = 4000) (hSample : totalSample = 120) :
  stratified_sample popA popB popC totalSample = (60, 40, 20) :=
by
  sorry

end stratified_sample_correct_l506_506563


namespace arrange_students_l506_506860

theorem arrange_students :
  ∀ (students : Fin 5 → Prop), 
    students 0 = true ∧ students 1 = true ∧ students 2 = true ∧ students 3 = false ∧ students 4 = false ∧
    (∀ i, i < 5 → students i → i < 4 → students (i + 1) = true) → 
    (∀ i, i < 5 → students i → i < 4 → students (i + 1) = false) →
  (∑ s in (Finset.perm univ), s.count (λ p, (students p 0 ∧ students p 1) ∧ ¬ students p 2) = 36) :=
sorry

end arrange_students_l506_506860


namespace find_p_and_intersection_point_l506_506268

variables {m n p x y : ℚ}

-- Define the first line equation condition
def line1 (x y : ℚ) : Prop := x = y / 5 - 2 / 5

-- Point (m, n) lies on the first line
def point_m_n_on_line1 : Prop := line1 m n

-- Point (m + p, n + 15) lies on the first line
def point_m_p_n15_on_line1 : Prop := line1 (m + p) (n + 15)

-- Define the second line equation condition
def line2 (x y : ℚ) : Prop := x = 3 * y / 7 + 1 / 7

-- Statements to prove
theorem find_p_and_intersection_point :
  point_m_n_on_line1 →
  point_m_p_n15_on_line1 →
  ∃ (p : ℚ), p = 3 ∧ 
  ∃ (x y : ℚ), line1 x y ∧ line2 x y ∧ x = -7 / 8 ∧ y = -19 / 8 :=
by
  intro h1 h2
  use 3
  split
  sorry
  existsi (-7 / 8 : ℚ)
  existsi (-19 / 8 : ℚ)
  split
  sorry
  split
  sorry
  split
  refl
  refl

end find_p_and_intersection_point_l506_506268


namespace MN_parallel_AB_l506_506626

-- Define the geometric settings of the problem using appropriate structures.
structure Trapezoid :=
  (A B C D : Point)
  (AB_parallel_CD : A B ∥ C D)

structure Midpoints :=
  (X Y : Point)
  (X_midpoint_AB : midpoint X A B)
  (Y_midpoint_CD : midpoint Y C D)

structure IntersectionPoints :=
  (M N : Point)
  (M_intersection_PX_BC : intersection M (line_through P X) (line_through B C))
  (N_intersection_PY_DA : intersection N (line_through P Y) (line_through D A))

-- Formulate the theorem problem in Lean 4.
theorem MN_parallel_AB (A B C D P X Y M N : Point)
  (trapezoid : Trapezoid)
  (midpoints : Midpoints)
  (intersectionPoints : IntersectionPoints) :
  M N ∥ A B :=
by
  sorry

end MN_parallel_AB_l506_506626


namespace correct_statements_count_l506_506505

-- Definitions of the conditions as propositions
def cond1 : Prop := ∀ (T : Triangle), collinear (altitudes T)
def cond2 : Prop := ∀ (T : Triangle), inside (centroid T)
def cond3 : Prop := ∀ (T : Triangle), ∃ (a b c : ℝ), is_right_triangle T → altitudes_count T = 1
def cond4 : Prop := ∀ (T : Triangle), collinear (angle_bisectors T)

-- The problem statement to prove that exactly three of these conditions are correct
theorem correct_statements_count :
  (∃ (n : ℕ), n = 3) :=
by
  sorry -- Proof goes here

end correct_statements_count_l506_506505


namespace max_abs_polynomial_value_l506_506632

noncomputable theory
open Classical

def polynomial_is_bounded (f : ℝ → ℝ) (a b c : ℝ) (a_nonzero : a ≠ 0) :=
  f = λ x, a * x^2 + b * x + c ∧
  |f 0| ≤ 2 ∧ |f 2| ≤ 2 ∧ |f (-2)| ≤ 2

theorem max_abs_polynomial_value (a b c : ℝ) (a_nonzero : a ≠ 0) :
  ∀ (f : ℝ → ℝ), polynomial_is_bounded f a b c a_nonzero →
  ∀ x ∈ Icc (-2 : ℝ) 2, |f x| ≤ 5/2 :=
by
  intros f hf x hx
  sorry

end max_abs_polynomial_value_l506_506632


namespace sum_div_1000_eq_368_l506_506297

def a : ℕ → ℕ 
| 0     := 1
| 1     := 1
| 2     := 1
| (n+3) := a (n+2) + 2 * a (n+1) + a n

theorem sum_div_1000_eq_368 :
  let S := Finset.range 29 |>.sum (λ k => a k)
  a 28 = 6090307 ∧ a 29 = 11201821 ∧ a 30 = 20603361 →
  S % 1000 = 368 :=
by
  sorry

end sum_div_1000_eq_368_l506_506297


namespace tan_sin_cos_eq_l506_506448

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l506_506448


namespace prime_factors_of_450_l506_506928

open Nat -- Opening the Nat namespace for natural number operations

/-- Prove that 450 has exactly 3 distinct prime factors -/
theorem prime_factors_of_450 : (Nat.factors 450).nodup ∧ (Nat.factors 450).length = 3 := by
  sorry

end prime_factors_of_450_l506_506928


namespace triangle_outside_angle_properties_l506_506967

theorem triangle_outside_angle_properties (A B C P Q R : Point)
  (h1 : ∃ (P : Point), ∠ ABC = 45 ∧ ∠ BCP = 30)
  (h2 : ∃ (Q : Point), ∠ CAQ = 45 ∧ ∠ QCA = 30)
  (h3 : ∃ (R : Point), ∠ ABR = 15 ∧ ∠ BAR = 15)
  : ∠ PRQ = 90 ∧ QR = PR := 
sorry

end triangle_outside_angle_properties_l506_506967


namespace other_root_l506_506493

-- Lean 4 statement
theorem other_root (m n : ℝ) (h: IsRoot (λ x => x^2 + x + m) 2) : n = -3 :=
  have h_sum : 2 + n = -1 := by
    sorry -- conditions from the problem
  
  show n = -3 from by
    sorry -- correct answer from the solution

end other_root_l506_506493


namespace composite_integer_unique_solution_l506_506336

theorem composite_integer_unique_solution :
  ∀ (n : ℕ), n > 1 ∧ ∃ (d : list ℕ), d.head = 1 ∧ d.last = n ∧ (∀ i, 1 ≤ i ∧ i < d.length → d.nth i < d.nth (i + 1)) 
  ∧ (((d.nth (i+2) - d.nth (i+1)) / (d.nth (i+1) - d.nth i)) = i + 1) 
  → (∃ (n : ℕ), n = 4) := sorry

end composite_integer_unique_solution_l506_506336


namespace find_b_and_c_find_b_with_c_range_of_b_l506_506883

-- Part (Ⅰ)
theorem find_b_and_c (b c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 + 2 * b * x + c)
  (h_zeros : f (-1) = 0 ∧ f 1 = 0) : b = 0 ∧ c = -1 := sorry

-- Part (Ⅱ)
theorem find_b_with_c (b : ℝ) (f : ℝ → ℝ)
  (x1 x2 : ℝ) 
  (h_def : ∀ x, f x = x^2 + 2 * b * x + (b^2 + 2 * b + 3))
  (h_eq : (x1 + 1) * (x2 + 1) = 8) 
  (h_roots : f x1 = 0 ∧ f x2 = 0) : b = -2 := sorry

-- Part (Ⅲ)
theorem range_of_b (b : ℝ) (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_f_def : ∀ x, f x = x^2 + 2 * b * x + (-1 - 2 * b))
  (h_f_1 : f 1 = 0)
  (h_g_def : ∀ x, g x = f x + x + b)
  (h_intervals : ∀ x, 
    ((-3 < x) ∧ (x < -2) → g x > 0) ∧
    ((-2 < x) ∧ (x < 0) → g x < 0) ∧
    ((0 < x) ∧ (x < 1) → g x < 0) ∧
    ((1 < x) → g x > 0)) : (1/5) < b ∧ b < (5/7) := sorry

end find_b_and_c_find_b_with_c_range_of_b_l506_506883


namespace find_n_l506_506183

theorem find_n (x y : ℤ) (n : ℕ) (h1 : (x:ℝ)^n + (y:ℝ)^n = 91) (h2 : (x:ℝ) * y = 11.999999999999998) :
  n = 3 := 
sorry

end find_n_l506_506183


namespace teacher_visit_arrangement_l506_506071

theorem teacher_visit_arrangement :
  let teachers := {TeacherA, TeacherB, TeacherC, TeacherD, TeacherE, TeacherF}
  let students := {Student1, Student2, Student3}
  -- Condition: Each student must be visited by at least one teacher
  (∃ (f : teachers → students), 
  (∀ t, TeacherA ≠ t → f t ≠ Student1) -- Condition: Teacher A does not visit Student 1
  ∧ (∀ s, ∃ t, f t = s) -- Condition: Each student is visited by at least one teacher
  ∧ injective f) -- Condition: Each teacher visits only one student
  → (nat.factorial 6 / (nat.factorial 3 * nat.factorial 1 * nat.factorial 2)) = 360 := 
begin
  sorry -- Proof to be completed
end

end teacher_visit_arrangement_l506_506071


namespace hyperbola_standard_eq_l506_506494

/-- Given the hyperbola with asymptotes y = ± 2x and sharing the same foci with the ellipse, find its standard equation -/
theorem hyperbola_standard_eq :
  ∀ (x y : ℝ),
  (asymptote1 : ∀ (x : ℝ), y = 2 * x) ∧
  (asymptote2 : ∀ (x : ℝ), y = -2 * x) ∧
  (eq_of_ellipse : ∀ (x y : ℝ), x^2 / 49 + y^2 / 24 = 1)
  → (x^2 / 5 - y^2 / 20 = 1) :=
by
  sorry

end hyperbola_standard_eq_l506_506494


namespace workers_together_time_l506_506734

-- Definition of the times taken by each worker to complete the job
def timeA : ℚ := 8
def timeB : ℚ := 10
def timeC : ℚ := 12

-- Definition of the rates based on the times
def rateA : ℚ := 1 / timeA
def rateB : ℚ := 1 / timeB
def rateC : ℚ := 1 / timeC

-- Definition of the total rate when working together
def total_rate : ℚ := rateA + rateB + rateC

-- Definition of the total time taken to complete the job when working together
def total_time : ℚ := 1 / total_rate

-- The final theorem we need to prove
theorem workers_together_time : total_time = 120 / 37 :=
by {
  -- structure of the proof will go here, but it is not required as per the instructions
  sorry
}

end workers_together_time_l506_506734


namespace min_perimeter_of_polygon_with_roots_of_Q_is_P_l506_506995

noncomputable def Q (z : ℂ) : ℂ := z^8 + (5 * Real.sqrt 2 + 8) * z^4 - (5 * Real.sqrt 2 + 9)

theorem min_perimeter_of_polygon_with_roots_of_Q_is_P (P : ℝ) :
  let roots := {z : ℂ | Q z = 0}
  ∃ vertices : List ℂ,
    (∀ v ∈ vertices, v ∈ roots) ∧
    vertices.length = 8 ∧
    let perimeter := List.sum (List.pairwise vertices (fun a b => Complex.abs (a - b))) in
    perimeter = P :=
sorry

end min_perimeter_of_polygon_with_roots_of_Q_is_P_l506_506995


namespace find_x_value_l506_506412

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l506_506412


namespace part1_condition1_part1_condition2_part1_condition3_part2_l506_506491

theorem part1_condition1 (a c A C : ℝ) (h : c = sqrt(3) * a * sin C - c * cos A) :
  A = π / 3 := sorry

theorem part1_condition2 (A B C : ℝ) (h : sin^2 A - sin^2 B = sin^2 C - sin B * sin C) :
  A = π / 3 := sorry

theorem part1_condition3 (B C : ℝ) (h : tan B + tan C - sqrt(3) * tan B * tan C = -sqrt(3)) :
  let A := π - (B + C)
  A = π / 3 := sorry

theorem part2 (a : ℝ) (h1 : a = sqrt(3)) (h2 : let B := π - (C + A);
                              let C := π - (A + B);
                              0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) :
  let b := 2 * sin B;
  let c := 2 * sin C;
  let perimeter := a + b + c;
  3 + sqrt(3) < perimeter ∧ perimeter ≤ 3 * sqrt(3) := sorry

end part1_condition1_part1_condition2_part1_condition3_part2_l506_506491


namespace max_range_walk_min_range_walk_count_max_range_sequences_l506_506753

variable {a b : ℕ}

-- Condition: a > b
def valid_walk (a b : ℕ) : Prop := a > b

-- Proof that the maximum possible range of the walk is a
theorem max_range_walk (h : valid_walk a b) : 
  (a + b) = a + b := sorry

-- Proof that the minimum possible range of the walk is a - b
theorem min_range_walk (h : valid_walk a b) : 
  (a - b) = a - b := sorry

-- Proof that the number of different sequences with the maximum possible range is b + 1
theorem count_max_range_sequences (h : valid_walk a b) : 
  b + 1 = b + 1 := sorry

end max_range_walk_min_range_walk_count_max_range_sequences_l506_506753


namespace largest_minimally_friendly_set_has_72_diagonals_l506_506116

theorem largest_minimally_friendly_set_has_72_diagonals :
  ∀ (P : Type) [convex_polygon P] (n : ℕ), n = 50 → 
    ∃ (F : set (diagonal P)), (minimally_friendly F) ∧ F.card = 72 := sorry

end largest_minimally_friendly_set_has_72_diagonals_l506_506116


namespace largest_a_l506_506124

open Real

theorem largest_a (a b c : ℝ) (h1 : a + b + c = 6) (h2 : ab + ac + bc = 11) : 
  a ≤ 2 + 2 * sqrt 3 / 3 :=
sorry

end largest_a_l506_506124


namespace angle_BAC_is_90_l506_506590

open EuclideanGeometry

variables {A B C D E : Point}
variables [Triangle ABC]

-- Conditions
variable (h1 : dist C D = 3 * dist C A)
variable (h2 : between C A D)
variable (h3 : dist C E = dist B C)
variable (h4 : E ≠ B)
variable (h5 : dist B D = dist A E)

theorem angle_BAC_is_90 : ∠ B A C = 90 :=
by {
  sorry -- here should be the proof
}

end angle_BAC_is_90_l506_506590


namespace octal_to_decimal_l506_506814

theorem octal_to_decimal (n : ℕ) (h : n = 6724₈) : n == 3540 :=
begin
  rw h,
  norm_num,
  sorry,
end

end octal_to_decimal_l506_506814


namespace f_ordering_l506_506076

variables {f : ℝ → ℝ}
noncomputable def a : ℝ := log 2 (1 / 3)
noncomputable def b : ℝ := log 4 (1 / 5)
noncomputable def c : ℝ := 2^ (3 / 2)

-- Conditions
axiom even_function (hf : f (-x) = f x)
axiom monotonic_decreasing (hf : ∀ x y ∈ Iic 0, x < y → f x ≥ f y) -- Here Iic 0 denotes (-∞, 0]

-- The theorem statement
theorem f_ordering (hf : even_function f) (hf_mono : monotonic_decreasing f) : f b < f a < f c :=
sorry

end f_ordering_l506_506076


namespace solution_to_ball_problem_l506_506240

noncomputable def probability_of_arithmetic_progression : Nat :=
  let p := 3
  let q := 9464
  p + q

theorem solution_to_ball_problem : probability_of_arithmetic_progression = 9467 := by
  sorry

end solution_to_ball_problem_l506_506240


namespace angle_between_vectors_range_of_t_l506_506634

variables {V : Type*} [inner_product_space ℝ V]

-- Part 1
theorem angle_between_vectors
  (a b : V)
  (ha : ∥a∥ = 1) (hb : ∥b∥ = 2)
  (h : (2 • a - b) ⬝ (a + b) = -3) :
  ∃ θ, (a ⬝ b = ∥a∥ * ∥b∥ * Real.cos θ) ∧ (θ = Real.pi * 2 / 3) :=
sorry

-- Part 2
theorem range_of_t
  (a b : V)
  (ha : ∥a∥ = 1) (hb : ∥b∥ = 2)
  (hab : angle a b = Real.pi / 3) :
  { t : ℝ | angle (t • a + b) (2 • a + t • b) > Real.pi / 2 } =
  { t : ℝ | -3 - 2 * Real.sqrt 2 < t ∧ t < -3 + 2 * Real.sqrt 2 } :=
sorry

end angle_between_vectors_range_of_t_l506_506634


namespace correct_propositions_l506_506542

-- Definitions based on conditions
variables (l : Line) (α β γ : Plane)

-- Propositions stated as logical statements
def proposition1 : Prop := (l ∥ α) ∧ (l ∥ β) → (α ∥ β)
def proposition2 : Prop := (α ⊥ γ) ∧ (β ∥ γ) → (α ⊥ β)
def proposition3 : Prop := (l ∥ α) ∧ (l ⊥ β) → (α ⊥ β)
def proposition4 : Prop := (α ⊥ β) ∧ (l ∥ α) → (l ⊥ β)

-- Lean statement to prove there are exactly two correct propositions
theorem correct_propositions :
  (¬ proposition1) ∧ (¬ proposition2) ∧ proposition3 ∧ proposition4 :=
by sorry

end correct_propositions_l506_506542


namespace tents_in_northmost_part_is_100_l506_506150

-- Definition of the conditions
def tents_north := sorry
def tents_east (N : ℕ) := 2 * N
def tents_center (N : ℕ) := 4 * N
def tents_south := 200
def total_tents (N : ℕ) := N + tents_east N + tents_center N + tents_south

-- Lean statement that models the proof problem
theorem tents_in_northmost_part_is_100 (N : ℕ) (h : total_tents N = 900) : N = 100 := by
  -- Proof to be provided
  sorry

end tents_in_northmost_part_is_100_l506_506150


namespace geometric_series_first_term_l506_506207

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l506_506207


namespace find_x_l506_506388

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l506_506388


namespace umbrella_shape_arrangements_l506_506243

/-- A mathematical proof problem for counting valid umbrella-shaped arrangements. -/
theorem umbrella_shape_arrangements (actors : Finset ℕ) (h7 : actors.card = 7) (distinct_heights : ∀ (a b ∈ actors), a ≠ b):
  ∃ (n : ℕ), (n = 20) := 
by
  sorry

end umbrella_shape_arrangements_l506_506243


namespace proof_problem_l506_506010

-- Definitions
def a (n : ℕ) : ℝ := if n = 1 then 1 else 1 - 1 / (4 * a (n-1))

def b (n : ℕ) : ℝ := 2 / (2 * a n - 1)

def c (n : ℕ) : ℝ := 4 * a n / (n + 1)

def T (n : ℕ) : ℝ := ∑ i in finset.range n, c i * c (i + 2)

-- Theorem to prove the derived properties
theorem proof_problem (n m: ℕ) (hn : n ∈ ℕ → n > 0) (hm : m ∈ ℕ → m > 0) :
  ∃ T : ℝ, b (n + 1) - b n = 2 ∧ 
          a n = (n + 1) / (2 * n) ∧ 
          T n = ∑ i in (finset.range n), 4 * a i / (i + 1) * 4 * a (i+2) / (i+3) ∧ 
          ∃ m, T n = 1 / (c m * c (m + 1)) ∧ m ≥ 3 := sorry

end proof_problem_l506_506010


namespace smallest_five_digit_congruent_to_3_mod_17_l506_506724

theorem smallest_five_digit_congruent_to_3_mod_17 : ∃ n : ℤ, (n > 9999) ∧ (n % 17 = 3) ∧ (∀ m : ℤ, (m > 9999) ∧ (m % 17 = 3) → n ≤ m) :=
by
  use 10012
  split
  { sorry }
  split
  { sorry }
  { sorry }

end smallest_five_digit_congruent_to_3_mod_17_l506_506724


namespace contradiction_l506_506570

variable (n m : ℕ)
variable (Ivanov Petrov Sidorov : ℕ → Prop)
variable (referee_judged : ℕ → ℕ → ℕ) -- referee_judged(r, i, j) means referee r judged the match between participants i and j

-- Conditions
axiom participants_unique : ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n → 1 ≤ j ∧ j ≤ n → i ≠ j → ∃ r, 1 ≤ r ∧ r ≤ m ∧ referee_judged r i j
axiom referees_different_matches : ∀ (r1 r2 : ℕ), 1 ≤ r1 ∧ r1 ≤ m → 1 ≤ r2 ∧ r2 ≤ m → r1 ≠ r2 → ∃ i j, referee_judged r1 i j ∧ ¬ referee_judged r2 i j
axiom claims_matched_different_referees : ∀ (k i j : ℕ), (k = Ivanov ∨ k = Petrov ∨ k = Sidorov) → i ≠ j → (referee_judged k i i ≠ referee_judged k i j)

-- To Prove
theorem contradiction : false := sorry

end contradiction_l506_506570


namespace monotonicity_and_maximum_exists_a_maximum_value_fx_gx_lt_zero_l506_506912

-- Definitions for the problem conditions and the required proofs.

-- (Ⅰ) The intervals of monotonicity and the maximum value of f(x)
theorem monotonicity_and_maximum (a : ℝ) (f : ℝ → ℝ) (x : ℝ) (h_a : a = 1) (h_f : ∀ x, f x = log x - a * x) :
  (∀ x ∈ Ioo 0 1, deriv f x > 0) ∧ (∀ x ∈ Ioo 1 e, deriv f x < 0) ∧ (f 1 = -1) :=
  sorry

-- (Ⅱ) The existence of a real number a such that the maximum value of f(x) is -3
theorem exists_a_maximum_value (a : ℝ) (f : ℝ → ℝ) (x : ℝ) (h_f : ∀ x, f x = log x - a * x) :
  (∃ a : ℝ, a = real.exp 2 ∧ ∀ x ∈ Icc 0 e, f x ≤ -3) :=
  sorry

-- (Ⅲ) Proving f(x) + g(x) + 1/2 < 0 under the given condition
theorem fx_gx_lt_zero (f g : ℝ → ℝ) (x : ℝ) (h_f : ∀ x, f x = log x - x) (h_g : ∀ x, g x = log x / x) :
  (∀ x ∈ Ioo 0 e, f x + g x + (1 / 2) < 0) :=
  sorry

end monotonicity_and_maximum_exists_a_maximum_value_fx_gx_lt_zero_l506_506912


namespace fixed_point_when_a_2_b_neg2_range_of_a_for_two_fixed_points_l506_506866

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 2

theorem fixed_point_when_a_2_b_neg2 :
  (∃ x : ℝ, f 2 (-2) x = x) → (x = -1 ∨ x = 2) :=
sorry

theorem range_of_a_for_two_fixed_points (a : ℝ) :
  (∀ b : ℝ, a ≠ 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 = x1 ∧ f a b x2 = x2)) → (0 < a ∧ a < 2) :=
sorry

end fixed_point_when_a_2_b_neg2_range_of_a_for_two_fixed_points_l506_506866


namespace sin_alpha_is_minus_half_l506_506077

noncomputable def sin_alpha_proof (α : ℝ) : Prop :=
  let P := (Real.sin 600, Real.cos (-120)) in
  ∃ (α : ℝ), (P.1 = Real.sin α) ∧ (P.2 = Real.cos α) ∧ (Real.sin α = -1/2)

theorem sin_alpha_is_minus_half : sin_alpha_proof α :=
  sorry

end sin_alpha_is_minus_half_l506_506077


namespace sum_of_b_for_quadratic_has_one_solution_l506_506340

theorem sum_of_b_for_quadratic_has_one_solution :
  (∀ x : ℝ, 3 * x^2 + (b+6) * x + 1 = 0 → 
    ∀ Δ : ℝ, Δ = (b + 6)^2 - 4 * 3 * 1 → 
    Δ = 0 → 
    b = -6 + 2 * Real.sqrt 3 ∨ b = -6 - 2 * Real.sqrt 3) → 
  (-6 + 2 * Real.sqrt 3 + -6 - 2 * Real.sqrt 3 = -12) := 
by
  sorry

end sum_of_b_for_quadratic_has_one_solution_l506_506340


namespace random_sampling_correct_l506_506695

def original_numbers : List ℕ := List.range 112 |>.map (· + 1)

def random_sampled_numbers : List ℕ := [74, 100, 94, 52, 80, 3, 105, 107, 83, 92]

theorem random_sampling_correct :
  ∀ n ∈ random_sampled_numbers, n ∈ original_numbers := by
  intros n hn
  simp [original_numbers] at hn
  repeat {simp [List.mem_map, List.range] at *,
  cases hn with a ha,
  cases ha with ha1 ha2,
  rw [ha2],
  simp only [List.mem_range],
  linarith [ha1]}
  done
  sorry 

end random_sampling_correct_l506_506695


namespace cost_of_adult_ticket_l506_506808

-- Conditions provided in the original problem.
def total_people : ℕ := 23
def child_tickets_cost : ℕ := 10
def total_money_collected : ℕ := 246
def children_attended : ℕ := 7

-- Define some unknown amount A for the adult tickets cost to be solved.
variable (A : ℕ)

-- Define the Lean statement for the proof problem.
theorem cost_of_adult_ticket :
  16 * A = 176 →
  A = 11 :=
by
  -- Start the proof (this part will be filled out during the proof process).
  sorry

#check cost_of_adult_ticket  -- To ensure it type-checks

end cost_of_adult_ticket_l506_506808


namespace power_function_problem_l506_506498

-- Define alpha such that (\frac{\sqrt{2}}{2})^alpha = \frac{\sqrt{2}}{4}
def alpha : ℝ := 3

-- Define the power function
def f (x : ℝ) : ℝ := x ^ alpha

-- The point (\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{4}) lies on the graph of y = f(x)
def condition := f (Real.sqrt 2 / 2) = Real.sqrt 2 / 4

-- Prove that f(-2) = -8 given the conditions
theorem power_function_problem (h : condition) : f (-2) = -8 :=
  by
    -- Sorry to skip the proof step as required
    sorry

end power_function_problem_l506_506498


namespace number_of_ways_songs_can_be_liked_l506_506813

-- Define types and conditions
def Songs := Finset ℕ
def Amy := {a : ℕ | a ∈ Songs}
def Beth := {b : ℕ | b ∈ Songs}
def Jo := {c : ℕ | c ∈ Songs}

noncomputable def exactly_one_from_each_pair (S_AB S_BC S_CA : Songs) : Prop :=
  S_AB.nonempty ∧ S_BC.nonempty ∧ S_CA.nonempty

noncomputable def amy_likes_two (S_A : Songs) : Prop :=
  S_A.card ≥ 2

noncomputable def distinct_songs (S_A S_AB S_BC S_CA : Songs) : Prop :=
  S_A ∪ S_AB ∪ S_BC ∪ S_CA = Songs ∧
  S_A ∩ S_AB = ∅ ∧
  S_A ∩ S_BC = ∅ ∧
  S_A ∩ S_CA = ∅ ∧
  S_AB ∩ S_BC = ∅ ∧
  S_AB ∩ S_CA = ∅ ∧
  S_BC ∩ S_CA = ∅

noncomputable def count_ways_to_satisfy_conditions (Songs : Finset ℕ) : ℕ :=
  let total_songs := 5 in
  if Songs.card = total_songs then
    (nat.choose total_songs 1) * (nat.choose (total_songs - 1) 1) * (nat.choose (total_songs - 2) 1) * (nat.choose (total_songs - 3) 2)
  else 0

theorem number_of_ways_songs_can_be_liked :
  ∀ (S_A S_AB S_BC S_CA : Songs),
  S_A.card + S_AB.card + S_BC.card + S_CA.card = 5 →
  exactly_one_from_each_pair S_AB S_BC S_CA →
  amy_likes_two S_A →
  distinct_songs S_A S_AB S_BC S_CA →
  count_ways_to_satisfy_conditions (S_A ∪ S_AB ∪ S_BC ∪ S_CA) = 60 :=
sorry

end number_of_ways_songs_can_be_liked_l506_506813


namespace sin_function_monotonic_increasing_on_interval_l506_506513

def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

theorem sin_function_monotonic_increasing_on_interval {ω : ℝ} (h_ω : ω = 1 / 4) :
  ∀ x y, 
    (Real.pi / 2 < x) → (x < Real.pi) → 
    (Real.pi / 2 < y) → (y < Real.pi) → 
    x < y → 
    Real.sin (ω * x - Real.pi / 12) < Real.sin (ω * y - Real.pi / 12) := 
by 
  sorry

end sin_function_monotonic_increasing_on_interval_l506_506513


namespace disproving_equation_l506_506472

theorem disproving_equation 
  (a b c d : ℚ)
  (h : a / b = c / d)
  (ha : a ≠ 0)
  (hc : c ≠ 0) : 
  a + d ≠ (a / b) * (b + c) := 
by 
  sorry

end disproving_equation_l506_506472


namespace point_in_first_quadrant_l506_506586

noncomputable def z : ℂ := 2 - (1 - complex.i) / complex.i

theorem point_in_first_quadrant (z = 2 - (1 - complex.i) / complex.i) : 
  z.re > 0 ∧ z.im > 0 :=
sorry

end point_in_first_quadrant_l506_506586


namespace range_of_a_l506_506920

theorem range_of_a :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → (0 < a ∧ a < 1) :=
by
  intros
  sorry

end range_of_a_l506_506920


namespace equation_of_chord_l506_506070

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

def is_midpoint_of_chord (P M N : ℝ × ℝ) : Prop :=
  ∃ (C : ℝ × ℝ), circle_eq (C.1) (C.2) ∧ (P.1, P.2) = ((M.1 + N.1) / 2, (M.2 + N.2) / 2)

theorem equation_of_chord (P : ℝ × ℝ) (M N : ℝ × ℝ) (h : P = (4, 2)) (h_mid : is_midpoint_of_chord P M N) :
  ∀ (x y : ℝ), (2 * y) - (8 : ℝ) = (-(1 / 2) * (x - 4)) →
  x + 2 * y - 8 = 0 :=
by
  intro x y H
  sorry

end equation_of_chord_l506_506070


namespace probability_no_divisible_by_10_l506_506550

theorem probability_no_divisible_by_10 : 
  ( ∀ (digits : List ℕ), 
    digits = [1, 2, 4, 5, 7, 5] → 
    (∃ n : ℕ, (
      (list.perm.digits n).length = 6 ∧ 
      ∀ d, d ∈ list.perm.digits n → d ∈ {1, 2, 4, 5, 7, 5}
    ) → n % 10 = 0) → ∃ n = 0) :=
begin
  intros digits h,
  -- Proof steps would go here
  sorry,
end

end probability_no_divisible_by_10_l506_506550


namespace geometric_series_first_term_l506_506203

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l506_506203


namespace sqrt_sum_eq_pow_l506_506467

/-- 
For the value \( k = 3/2 \), the expression \( \sqrt{2016} + \sqrt{56} \) equals \( 14^k \)
-/
theorem sqrt_sum_eq_pow (k : ℝ) (h : k = 3 / 2) : 
  (Real.sqrt 2016 + Real.sqrt 56) = 14 ^ k := 
by 
  sorry

end sqrt_sum_eq_pow_l506_506467


namespace distance_to_right_focus_l506_506518

open Real

-- Define the elements of the problem
variable (a c : ℝ)
variable (P : ℝ × ℝ) -- Point P on the hyperbola
variable (F1 F2 : ℝ × ℝ) -- Left and right foci
variable (D : ℝ) -- The left directrix

-- Define conditions as Lean statements
def hyperbola_eq : Prop := (a ≠ 0) ∧ (c ≠ 0) ∧ (P.1^2 / a^2 - P.2^2 / 16 = 1)
def point_on_right_branch : Prop := P.1 > 0
def distance_diff : Prop := abs (dist P F1 - dist P F2) = 6
def distance_to_left_directrix : Prop := abs (P.1 - D) = 34 / 5

-- Define theorem to prove the distance from P to the right focus
theorem distance_to_right_focus
  (hp : hyperbola_eq a c P)
  (hbranch : point_on_right_branch P)
  (hdiff : distance_diff P F1 F2)
  (hdirectrix : distance_to_left_directrix P D) :
  dist P F2 = 16 / 3 :=
sorry

end distance_to_right_focus_l506_506518


namespace angle_between_neg2c_and_3d_l506_506545

variables (c d : ℝ^3)
variable (theta : ℝ)

-- The given condition
def angle_between_c_and_d : Prop := theta = 60

-- The goal to prove
theorem angle_between_neg2c_and_3d (h : angle_between_c_and_d c d theta) :
  angle_between (-2 • c) (3 • d) = 120 := 
sorry

end angle_between_neg2c_and_3d_l506_506545


namespace length_of_AB_l506_506907

theorem length_of_AB (x y : ℝ) :
  (x^2 + 4 * y^2 = 16) → ((x - 2 * y - 4) = 0) → (x = 4 ∧ y = 0) ∨ (x = 0 ∧ y = -2) →
  ( ∃ A B : ℝ × ℝ, A = (4, 0) ∧ B = (0, -2) ∧ (real.sqrt((A.1 - B.1)^2 + (A.2 - B.2)^2)) = 2 * real.sqrt(5) ) :=
by
  intro h_ellipse h_line h_points
  cases h_points with h_A h_B
  use (4, 0), (0, -2)
  split
  exact h_A
  split
  exact h_B
  norm_num
  sorry

end length_of_AB_l506_506907


namespace number_of_triples_lcm_l506_506998

def lcm (a b : ℕ) := Nat.lcm a b

theorem number_of_triples_lcm :
  ∃ (triples : Finset (ℕ × ℕ × ℕ)), 
  (∀ (a b c : ℕ), (a, b, c) ∈ triples ↔ lcm a b = 20736 ∧ lcm b c = 2592 ∧ lcm c a = 1728) ∧
  triples.card = 7 :=
by
  sorry

end number_of_triples_lcm_l506_506998


namespace gift_sequence_count_l506_506955

noncomputable def number_of_gift_sequences (students : ℕ) (classes_per_week : ℕ) : ℕ :=
  (students * students) ^ classes_per_week

theorem gift_sequence_count :
  number_of_gift_sequences 15 3 = 11390625 :=
by
  sorry

end gift_sequence_count_l506_506955


namespace concrete_needed_whole_number_l506_506299

noncomputable def volume_of_concrete (length_ft : ℝ) (shorter_base_ft : ℝ) (longer_base_ft : ℝ) (height_in : ℝ) : ℝ :=
  let length_yd := length_ft / 3
  let shorter_base_yd := shorter_base_ft / 3
  let longer_base_yd := longer_base_ft / 3
  let height_yd := height_in / 36
  let area_yd2 := (1/2) * (shorter_base_yd + longer_base_yd) * height_yd
  (area_yd2 * length_yd)

theorem concrete_needed_whole_number 
  (length_ft : ℝ) (shorter_base_ft : ℝ) (longer_base_ft : ℝ) (height_in : ℝ) :
  length_ft = 60 → shorter_base_ft = 3 → longer_base_ft = 5 → height_in = 4 →
  (volume_of_concrete length_ft shorter_base_ft longer_base_ft height_in).ceil = 3 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end concrete_needed_whole_number_l506_506299


namespace jogger_distance_ahead_l506_506292

theorem jogger_distance_ahead
  (train_speed_km_hr : ℝ) (jogger_speed_km_hr : ℝ)
  (train_length_m : ℝ) (time_seconds : ℝ)
  (relative_speed_m_s : ℝ) (distance_covered_m : ℝ)
  (D : ℝ)
  (h1 : train_speed_km_hr = 45)
  (h2 : jogger_speed_km_hr = 9)
  (h3 : train_length_m = 100)
  (h4 : time_seconds = 25)
  (h5 : relative_speed_m_s = 36 * (5/18))
  (h6 : distance_covered_m = 10 * 25)
  (h7 : D + train_length_m = distance_covered_m) :
  D = 150 :=
by sorry

end jogger_distance_ahead_l506_506292


namespace money_contributed_by_each_person_l506_506112

-- Definitions according to the conditions
variables (N : ℕ) (P : ℝ) (third_place_amount : ℝ)
  (split_ratio : ℝ) [fact (split_ratio = 0.10)] [fact (third_place_amount = 4.0)]

-- Assert the number of people including Josh
def total_people : ℕ := 8

-- Calculate the total pot using the given conditions
def total_pot := third_place_amount / split_ratio

-- Calculate how much each person put into the pot
def money_per_person := total_pot / total_people

-- Theorem to prove the correct amount each person put in
theorem money_contributed_by_each_person : money_per_person = 5.0 :=
by 
  sorry

end money_contributed_by_each_person_l506_506112


namespace painter_total_cost_l506_506802

def south_seq (n : Nat) : Nat :=
  4 + 6 * (n - 1)

def north_seq (n : Nat) : Nat :=
  5 + 6 * (n - 1)

noncomputable def digit_cost (n : Nat) : Nat :=
  String.length (toString n)

noncomputable def total_cost : Nat :=
  let south_cost := (List.range 25).map south_seq |>.map digit_cost |>.sum
  let north_cost := (List.range 25).map north_seq |>.map digit_cost |>.sum
  south_cost + north_cost

theorem painter_total_cost : total_cost = 116 := by
  sorry

end painter_total_cost_l506_506802


namespace trigonometric_identity_1_trigonometric_identity_2_l506_506873

theorem trigonometric_identity_1 (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 2) :
  (3 * sin α - cos α) / (2 * sin α + 3 * cos α) = 8 / 9 :=
sorry

theorem trigonometric_identity_2 (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 2) :
  sin α ^ 2 - 2 * sin α * cos α + 1 = 13 / 10 :=
sorry

end trigonometric_identity_1_trigonometric_identity_2_l506_506873


namespace percentage_is_approximately_769_percent_l506_506106

-- Define the number of emails received in the morning, afternoon, and evening.
def emailsMorning : ℕ := 6
def emailsAfternoon : ℕ := 8
def emailsEvening : ℕ := 12

-- Define the difference between emails received in the afternoon and the morning.
def differenceEmails : ℕ := emailsAfternoon - emailsMorning

-- Define the total number of emails received throughout the day.
def totalEmails : ℕ := emailsMorning + emailsAfternoon + emailsEvening

-- Define the percentage of the total emails that the difference represents.
def percentageDifference (diff total : ℕ) : ℚ := ((diff : ℚ) / (total : ℚ)) * 100

-- Prove that the percentage of the total emails that the difference represents is approximately 7.69%
theorem percentage_is_approximately_769_percent : 
  percentageDifference differenceEmails totalEmails ≈ 7.69 := by
  sorry

end percentage_is_approximately_769_percent_l506_506106


namespace find_x_l506_506382

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l506_506382


namespace find_x_tan_eq_l506_506432

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l506_506432


namespace product_plus_one_is_square_l506_506941

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : ∃ k : ℕ, x * y + 1 = k * k :=
by
  sorry

end product_plus_one_is_square_l506_506941


namespace part1_part2_l506_506027

-- Part 1
theorem part1 (a b : ℝ × ℝ) (k : ℝ) (h1 : a = (1, 2)) (h2 : b = (2, 1)) (h3 : (k • a - b) • a = 0) : k = 4 / 5 :=
sorry

-- Part 2
theorem part2 (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : ∃ λ : ℝ, b = λ • a) (h3 : ∥b∥ = 3 * Real.sqrt 5) :
  b = (3, 6) ∨ b = (-3, -6) :=
sorry

end part1_part2_l506_506027


namespace function_machine_output_is_17_l506_506978

def functionMachineOutput (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 <= 22 then step1 + 10 else step1 - 7

theorem function_machine_output_is_17 : functionMachineOutput 8 = 17 := by
  sorry

end function_machine_output_is_17_l506_506978


namespace number_of_integer_solutions_l506_506058

theorem number_of_integer_solutions : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x + 3)^2 ≤ 4) ∧ S.card = 5 := by
  sorry

end number_of_integer_solutions_l506_506058


namespace range_of_a_l506_506952

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc (0 : ℝ) (1 / 2), 4^x + x - a ≤ 3/2) ↔ (a ≥ 1) :=
by
  sorry

end range_of_a_l506_506952


namespace symmetric_points_on_ellipse_are_m_in_range_l506_506906

open Real

theorem symmetric_points_on_ellipse_are_m_in_range (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 ^ 2) / 4 + (A.2 ^ 2) / 3 = 1 ∧ 
                   (B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1 ∧ 
                   ∃ x0 y0 : ℝ, y0 = 4 * x0 + m ∧ x0 = (A.1 + B.1) / 2 ∧ y0 = (A.2 + B.2) / 2) 
  ↔ -2 * sqrt 13 / 13 < m ∧ m < 2 * sqrt 13 / 13 := 
 sorry

end symmetric_points_on_ellipse_are_m_in_range_l506_506906


namespace range_of_a_for_unique_root_l506_506074

theorem range_of_a_for_unique_root (a : ℝ) :
  (∃! x : ℝ, x + sqrt (x + 0.5 + sqrt (x + 0.25)) = a) ↔ a ∈ Ici (1/4) :=
by
  sorry

end range_of_a_for_unique_root_l506_506074


namespace rolls_sold_to_uncle_l506_506863

theorem rolls_sold_to_uncle (total_rolls : ℕ) (rolls_grandmother : ℕ) (rolls_neighbor : ℕ) (rolls_remaining : ℕ) (rolls_uncle : ℕ) :
  total_rolls = 12 →
  rolls_grandmother = 3 →
  rolls_neighbor = 3 →
  rolls_remaining = 2 →
  rolls_uncle = total_rolls - rolls_remaining - (rolls_grandmother + rolls_neighbor) →
  rolls_uncle = 4 :=
by
  intros h_total h_grandmother h_neighbor h_remaining h_compute
  rw [h_total, h_grandmother, h_neighbor, h_remaining] at h_compute
  exact h_compute

end rolls_sold_to_uncle_l506_506863


namespace dart_lands_within_inner_hexagon_prob_l506_506774

theorem dart_lands_within_inner_hexagon_prob :
  ∀ (s : ℝ) (A_inner A_outer : ℝ),
  A_inner = (3 * Real.sqrt 3 / 2) * s^2 ∧
  A_outer = (3 * Real.sqrt 3 / 2) * (2 * s)^2 ∧
  (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1) →
  ∃ (prob : ℝ), prob = A_inner / A_outer ∧ prob = 1 / 4 :=
by 
  intros s A_inner A_outer h1 h2 h3
  sorry

end dart_lands_within_inner_hexagon_prob_l506_506774


namespace average_production_l506_506466

theorem average_production (n : ℕ) (P : ℕ) (h1 : P = 60 * n) (h2 : (P + 90) / (n + 1) = 62) : n = 14 :=
  sorry

end average_production_l506_506466


namespace first_term_of_geometric_series_l506_506222

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l506_506222


namespace arithmetic_sequence_sum_l506_506972

variable {α : Type*} [LinearOrderedField α]

theorem arithmetic_sequence_sum (a : ℕ → α) (S : ℕ → α) (d a1 : α)
  (h1 : a 1 + a 3 = 8)
  (h2 : (a 4)^2 = (a 2) * (a 9))
  (h3 : ∀ (n : ℕ), a n = a1 + d * (n - 1))
  (hS1 : ∀ (n : ℕ), a1 = 4 → d = 0 → S n = 4 * n)
  (hS2 : ∀ (n : ℕ), a1 = 1 → d = 3 → S n = (3 * n^2 - n) / 2) :
  ∃ a1 d, (a1 = 4 ∧ d = 0 ∧ S = λ n, 4 * n) ∨ (a1 = 1 ∧ d = 3 ∧ S = λ n, (3 * n^2 - n) / 2) :=
by
  sorry

end arithmetic_sequence_sum_l506_506972


namespace find_x_l506_506531

def vector := ℝ × ℝ

def parallel (a b : vector) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

variables (x : ℝ)
def a : vector := (-1, x)
def b : vector := (-2, 4)

theorem find_x (h : parallel a b) : x = 2 :=
  sorry

end find_x_l506_506531


namespace danny_emma_walk_upward_distance_l506_506303

variable (danny_x danny_y emma_x emma_y felix_x felix_y : ℝ)

def midpoint_x (x1 x2 : ℝ) : ℝ := (x1 + x2) / 2
def midpoint_y (y1 y2 : ℝ) : ℝ := (y1 + y2) / 2
def vertical_distance (y1 y2 : ℝ) : ℝ := (y2 - y1).abs

theorem danny_emma_walk_upward_distance :
  let mid_x := midpoint_x danny_x emma_x in
  let mid_y := midpoint_y danny_y emma_y in
  let distance := vertical_distance mid_y felix_y in
  danny_x = 8 → danny_y = -25 → emma_x = -4 → emma_y = 19 → felix_x = 2 → felix_y = 5 → distance = 8 := by
  intros mid_x mid_y distance h1 h2 h3 h4 h5 h6
  dsimp [midpoint_x, midpoint_y, vertical_distance] at *
  rw [h1, h2, h3, h4, h5, h6]
  dsimp
  norm_num
  sorry

end danny_emma_walk_upward_distance_l506_506303


namespace max_values_sin_half_proof_max_values_sin_third_proof_l506_506924

noncomputable def max_values_sin_half (a : ℝ) (h : a ∈ set.range (sin : ℝ → ℝ)) : ℕ :=
  if a = 0 then 1 else 4

noncomputable def max_values_sin_third (a : ℝ) (h : a ∈ set.range (sin : ℝ → ℝ)) : ℕ :=
  if a = 0 then 1 else 3

theorem max_values_sin_half_proof (a : ℝ) (h : a ∈ set.range (sin : ℝ → ℝ)) : 
  max_values_sin_half a h = 4 := sorry

theorem max_values_sin_third_proof (a : ℝ) (h : a ∈ set.range (sin : ℝ → ℝ)) : 
  max_values_sin_third a h = 3 := sorry

end max_values_sin_half_proof_max_values_sin_third_proof_l506_506924


namespace find_m_l506_506055

variables (m : ℝ)

def vec_a : (ℝ × ℝ) := (-1, 1)
def vec_b : (ℝ × ℝ) := (3, m)
def vec_sum : (ℝ × ℝ) := (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2)

def are_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

theorem find_m (h : are_parallel vec_a vec_sum) : m = -3 :=
sorry

end find_m_l506_506055


namespace lockers_remaining_open_l506_506964

-- Define the number of lockers and students
def num_lockers : ℕ := 1000

-- Define a function to determine if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a function to count perfect squares up to a given number
def count_perfect_squares_up_to (n : ℕ) : ℕ :=
  Nat.sqrt n

-- Theorem statement
theorem lockers_remaining_open : 
  count_perfect_squares_up_to num_lockers = 31 :=
by
  -- Proof left out because it's not necessary to provide
  sorry

end lockers_remaining_open_l506_506964


namespace jacket_total_price_correct_l506_506798

/-- The original price of the jacket -/
def original_price : ℝ := 120

/-- The initial discount rate -/
def initial_discount_rate : ℝ := 0.15

/-- The additional discount in dollars -/
def additional_discount : ℝ := 10

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.10

/-- The calculated total amount the shopper pays for the jacket including all discounts and tax -/
def total_amount_paid : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_rate)
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  price_after_additional_discount * (1 + sales_tax_rate)

theorem jacket_total_price_correct : total_amount_paid = 101.20 :=
  sorry

end jacket_total_price_correct_l506_506798


namespace count_special_four_digit_numbers_l506_506538

theorem count_special_four_digit_numbers : 
  let possible_digits := [3, 5, 7]
  let num_digits := 4
  ∃ (N : ℕ), N = (possible_digits.length ^ num_digits) :=
begin
  have h_length : possible_digits.length = 3 := by simp,
  have h_exp : 3 ^ num_digits = 81 := by norm_num,
  exact ⟨81, h_exp⟩,
end

end count_special_four_digit_numbers_l506_506538


namespace common_factor_poly_6a2b_3ab2_l506_506173

theorem common_factor_poly_6a2b_3ab2 (a b : ℕ) : 
  ∃ (d : ℕ), d = nat.gcd (6 * a^2 * b) (3 * a * b^2) := 
sorry

end common_factor_poly_6a2b_3ab2_l506_506173


namespace concurrency_of_ME_FN_BD_l506_506157

variables {P A B C D M N E F K : Point}
variables {AB CD DA BC ME FN BD : Line}

-- Given conditions
def is_inside (P : Point) (A B C D : Point) : Prop := 
  ∃ (convex_comb : {a b c d : ℝ // a + b + c + d = 1 ∧ a, b, c, d ≥ 0 ∧ a * b * c * d ≠ 0 }), 
  P = a • A + b • B + c • C + d • D

def line_through (ℓ : Line) (P : Point) : Prop := 
  ∃ (A B : Point), A ≠ B ∧ ℓ = Line_through A B ∧ P ∈ ℓ
  
-- Problem statement
theorem concurrency_of_ME_FN_BD 
  (h_inside : is_inside P A B C D)
  (h_MN_through_P : line_through MN P)
  (h_EF_through_P : line_through EF P)
  (h_MN_parallel_AD : MN ∥ AD)
  (h_M_intersection : M ∈ AB ∧ M ∈ MN)
  (h_N_intersection : N ∈ CD ∧ N ∈ MN)
  (h_EF_parallel_AB : EF ∥ AB)
  (h_E_intersection : E ∈ DA ∧ E ∈ EF)
  (h_F_intersection : F ∈ BC ∧ F ∈ EF) :
  ∃ K, K ∈ ME ∧ K ∈ FN ∧ K ∈ BD :=
sorry

end concurrency_of_ME_FN_BD_l506_506157


namespace M_2020_l506_506117

noncomputable def q (n : ℕ) (v : fin (2 * n + 1) → ℤ) : ℤ :=
  1 + (finset.sum (finset.range (2 * n - 1)) (λ j, 3 ^ (v (fin.succ j) : ℤ)))

def V_n (n : ℕ) : finset (fin (2 * n + 1) → ℤ) :=
  finset.filter (λ v,
    (v 0 = 0 ∧ v ((2 * n) : fin (2 * n + 1)) = 0) ∧
    ∀ j : fin (2 * n), abs (v (j.succ) - v j) = 1)
  (finset.univ)

noncomputable def M (n : ℕ) : ℚ :=
  (finset.sum (V_n n) (λ v, (1 / (q n v : ℚ)))) / finset.card (V_n n)

theorem M_2020 : M 2020 = 1 / 4040 := sorry

end M_2020_l506_506117


namespace find_m_l506_506470

variable (m : ℝ)

def vector_a : ℝ × ℝ := (3, 4)
def vector_b : ℝ × ℝ := (-1, 2 * m)
def vector_c : ℝ × ℝ := (m, -4)
def vector_sum := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)

theorem find_m (h : vector_c.1 * vector_sum.1 + vector_c.2 * vector_sum.2 = 0) : m = -8 / 3 := by
  sorry

end find_m_l506_506470


namespace range_of_k_for_positivity_l506_506258

theorem range_of_k_for_positivity (k x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 2) :
  ((k - 2) * x + 2 * |k| - 1 > 0) → (k > 5 / 4) :=
sorry

end range_of_k_for_positivity_l506_506258


namespace line_passes_through_fixed_point_l506_506175

-- Define the line equation
def line_through_fixed_point (m : ℝ) (x y: ℝ) : Prop := m * x - y + 2 * m + 1 = 0

-- Proposition stating that the line passes through the fixed point (-2, 1)
theorem line_passes_through_fixed_point (m : ℝ) :
  line_through_fixed_point m (-2) 1 :=
begin
  -- Proof placeholder
  sorry
end

end line_passes_through_fixed_point_l506_506175


namespace difference_abs_eq_200_l506_506080

theorem difference_abs_eq_200 (x y : ℤ) (h1 : x + y = 250) (h2 : y = 225) : |x - y| = 200 := sorry

end difference_abs_eq_200_l506_506080


namespace symmetric_monotone_decreasing_l506_506676

noncomputable def f : ℝ → ℝ := sorry

theorem symmetric_monotone_decreasing
  (Hsym : ∀ x, f (3 - x) = f x)
  (Hmono : ∀ x, (x - 3 / 2) * (has_deriv_at f (f' x) x ∧ f' x) < 0)
  (x1 x2 : ℝ) (hx1x2 : x1 < x2) (hsum : x1 + x2 > 3) :
  f x1 > f x2 :=
sorry

end symmetric_monotone_decreasing_l506_506676


namespace parabola_right_shift_unique_intersection_parabola_down_shift_unique_intersection_l506_506520

theorem parabola_right_shift_unique_intersection (p : ℚ) :
  let y := 2 * (x - p)^2;
  (x * x - 4) = 0 →
  p = 31 / 8 := sorry

theorem parabola_down_shift_unique_intersection (q : ℚ) :
  let y := 2 * x^2 - q;
  (x * x - 4) = 0 →
  q = 31 / 8 := sorry

end parabola_right_shift_unique_intersection_parabola_down_shift_unique_intersection_l506_506520


namespace find_x_tan_eq_l506_506435

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l506_506435


namespace sum_of_odds_from_5_to_47_l506_506826

theorem sum_of_odds_from_5_to_47 :
  (∑ n in (Finset.range 22).image (λ k, 5 + 2 * k)) = 572 := by
  sorry

end sum_of_odds_from_5_to_47_l506_506826


namespace min_distance_origin_to_line_l506_506337

theorem min_distance_origin_to_line :
  ∀ (x y : ℝ), 8 * x + 15 * y = 120 → real.sqrt (x^2 + y^2) ≥ abs (120 / 17) :=
by
  sorry

end min_distance_origin_to_line_l506_506337


namespace part1_part2_l506_506509

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := log x + log m

theorem part1 (m : ℝ) :
  (0 : ℝ) < m →
  f (0) m = log m ∧
  f (2) m = log (2 + m) ∧
  f (6) m = log (6 + m) ∧
  (f 2 m) - (f 0 m) = (f 6 m) - (f 2 m) →
  m = 2 :=
sorry
  
theorem part2 (a b c : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : b * b = a * c) (h5 : b ≠ a) (h6 : b ≠ c) : 
  f a 2 + f c 2 > 2 * f b 2 :=
sorry

end part1_part2_l506_506509


namespace jonah_fish_count_l506_506111

theorem jonah_fish_count :
  let initial_fish := 14
  let added_fish := 2
  let eaten_fish := 6
  let removed_fish := 2
  let new_fish := 3
  initial_fish + added_fish - eaten_fish - removed_fish + new_fish = 11 := 
by
  sorry

end jonah_fish_count_l506_506111


namespace derivative_of_f_l506_506360

-- Define the function f
def f (x : ℝ) : ℝ := cos (2 * x - 1) + (1 / x^2)

-- State the theorem to prove the derivative of f
theorem derivative_of_f (x : ℝ) : 
  has_deriv_at f (-2 * sin (2 * x - 1) - 2 / x^3) x := sorry

end derivative_of_f_l506_506360


namespace solve_for_xyz_l506_506459

variable (a b c d e f g h i j : ℝ)
variable (x y z : ℝ)

def condition1 : Prop := (a = 3 ∧ b = 5 ∧ c = 2 ∧ d = 4 ∧ e = 63) → 
  x = (a^2 * b * 0.47 * 1442 - c * d * 0.36 * 1412) + e

def condition2 : Prop := (f = 2 ∧ g = 7 ∧ h = 3 ∧ i = 21) → 
  y = (f * g * 138 / h) - 0.27 * 987 + i

def condition3 : Prop := (x = 26498.74 ∧ y = 398.51 ∧ j = 175) → 
  x * y - z = j

theorem solve_for_xyz : 
  (a = 3 ∧ b = 5 ∧ c = 2 ∧ d = 4 ∧ e = 63 ∧ f = 2 ∧ g = 7 ∧ h = 3 ∧ i = 21 ∧ j = 175) → 
  x = 26498.74 ∧ y = 398.51 ∧ z = 10559523.7974 :=
by 
  intros h
  have h1 : condition1 := sorry
  have h2 : condition2 := sorry
  have h3 : condition3 := sorry
  sorry

end solve_for_xyz_l506_506459


namespace stock_exchange_total_l506_506737

theorem stock_exchange_total (L H : ℕ) 
  (h1 : H = 1080) 
  (h2 : H = 6 * L / 5) : 
  (L + H = 1980) :=
by {
  -- L and H are given as natural numbers
  -- h1: H = 1080
  -- h2: H = 1.20L -> H = 6L/5 as Lean does not handle floating point well directly in integers.
  sorry
}

end stock_exchange_total_l506_506737


namespace altitude_of_triangle_on_square_diagonal_l506_506012

theorem altitude_of_triangle_on_square_diagonal {s : ℝ} (h : s > 0) :
  let d := s * Real.sqrt 2 in
  let area_square := s^2 in
  let base := d in
  let area_triangle := (1/2) * base * (s * Real.sqrt 2) in
  area_square = area_triangle :=
by
  sorry

end altitude_of_triangle_on_square_diagonal_l506_506012


namespace hyperbola_center_focus_vertex_l506_506562

theorem hyperbola_center_focus_vertex :
  let h := 1
      k := -3
      a := 5
      c := Real.sqrt 50
      b := Real.sqrt 25
  in h + k + a + b = 8 :=
by
  let h := 1
  let k := -3
  let a := 5
  let c := Real.sqrt 50
  let b := Real.sqrt 25
  have b_eval: b = 5 := by
    rw [Real.sqrt_eq_iff_sqrt_sq_eq]; norm_num
  calc
  h + k + a + b_eval = h + k + a + 5
  ... = 1 - 3 + 5 + 5
  ... = 8

end hyperbola_center_focus_vertex_l506_506562


namespace first_term_of_geometric_series_l506_506223

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l506_506223


namespace percentage_increase_240_to_288_l506_506765

theorem percentage_increase_240_to_288 :
  let initial := 240
  let final := 288
  ((final - initial) / initial) * 100 = 20 := by 
  sorry

end percentage_increase_240_to_288_l506_506765


namespace derivative_f_at_1_l506_506534

-- Define the function f(x) = x * ln(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem to prove f'(1) = 1
theorem derivative_f_at_1 : (deriv f 1) = 1 :=
sorry

end derivative_f_at_1_l506_506534


namespace tan_sin_cos_eq_l506_506451

theorem tan_sin_cos_eq (x : ℝ) (h : 0 < x ∧ x < 180) :
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → x = 120 :=
by
  sorry

end tan_sin_cos_eq_l506_506451


namespace largest_integer_eq_l506_506132

noncomputable def a : ℕ → ℝ 
| 0       := 1994
| (n + 1) := (a n) ^ 2 / (a n + 1)

theorem largest_integer_eq (n : ℕ) (h₀ : 1 ≤ n) (h₁ : n ≤ 998) : 
  ∃ m : ℤ, m = ⌊a n⌋ ∧ m = 1994 - n := 
sorry

end largest_integer_eq_l506_506132


namespace pure_imaginary_complex_l506_506948

open Complex

theorem pure_imaginary_complex (a : ℝ) (h: Im (⟨a, 1⟩ / ⟨1, 2⟩) = 0) : a = -2 :=
sorry

end pure_imaginary_complex_l506_506948


namespace area_available_for_Rolly_l506_506164

-- Define the conditions
def rope_length : ℝ := 8
def wall_leg_length : ℝ := 16

-- Lean's tactic to prove the area available for Rolly is 32π square feet
theorem area_available_for_Rolly : 
  let r := rope_length in
  let total_area := 2 * (1 / 4 * Real.pi * r^2) in
  total_area = 32 * Real.pi :=
by
  let r := rope_length
  have area_one_quarter_circle : (1 / 4 * Real.pi * r^2) = 16 * Real.pi,
  calc
    (1 / 4 * Real.pi * r^2)
      = 1 / 4 * Real.pi * 64         : by rw [Real.pow_two]
  ... = 1 / 4 * 64 * Real.pi         : by ring
  ... = 16 * Real.pi                 : by norm_num,
  
  have total_area : 2 * (1 / 4 * Real.pi * r^2) = 32 * Real.pi,
  calc 
    2 * (1 / 4 * Real.pi * r^2)
      = 2 * 16 * Real.pi             : by rw [area_one_quarter_circle]
  ... = 32 * Real.pi                 : by ring,
  
  exact total_area

end area_available_for_Rolly_l506_506164


namespace triangle_sides_l506_506794

noncomputable def right_triangle_sides (side_length_rhombus : ℝ) :=
  let hypotenuse := 2 * side_length_rhombus + side_length_rhombus in
  let leg1 := hypotenuse / 2 in
  let leg2 := leg1 * Real.sqrt 3 in
  (leg1, leg2, hypotenuse)

theorem triangle_sides (side_length_rhombus : ℝ)
  (h1 : side_length_rhombus = 6) :
  right_triangle_sides side_length_rhombus = (9, 9 * Real.sqrt 3, 18) :=
by
  rw [h1]
  trivial -- proof needed
  sorry

end triangle_sides_l506_506794


namespace distance_city_A_to_C_l506_506346

theorem distance_city_A_to_C 
  (h1 : ∃ E, E = 570 / 3)  -- Eddy's speed
  (h2 : ∃ F, E / F = 2.533333333333333)  -- Ratio of speeds Eddy:Freddy
  (h3 : ∃ d_AC, d_AC = F * 4)  -- Distance city A to city C
  : d_AC = 300 := 
by 
  obtain ⟨E, hE⟩ := h1,
  obtain ⟨F, hF⟩ := h2,
  obtain ⟨d_AC, h_dAC⟩ := h3,
  sorry

end distance_city_A_to_C_l506_506346


namespace william_total_riding_hours_l506_506731

theorem william_total_riding_hours :
  let max_daily_time := 6
  let days := 6
  let max_time_days := 2
  let reduced_time1_days := 2
  let reduced_time2_days := 2
  let reduced_time1 := 1.5
  let reduced_time2 := max_daily_time / 2
  max_daily_time * max_time_days + reduced_time1 * reduced_time1_days + reduced_time2 * reduced_time2_days = 21 :=
begin
  sorry,
end

end william_total_riding_hours_l506_506731


namespace sum_1_to_100_mod_4_l506_506253

theorem sum_1_to_100_mod_4 : (∑ n in Finset.range 101, n) % 4 = 2 := 
by
  sorry

end sum_1_to_100_mod_4_l506_506253


namespace ABD_angle_l506_506278

-- Define variables, points, and angles
variables {A B C D : Type} [point : geometric_point A B C] [point : geometric_point D]

-- Conditions of the problem
-- ΔABC is isosceles with AB = AC
axiom isosceles_ABC : is_isosceles_triangle A B C (side_eq A B C)
-- m∠C = 30°
axiom angle_C : mangle A C B = 30
-- Point D is on line BC such that BD is an extension of line CB
axiom D_on_BC : on_line B C D

-- Prove: m∠ABD = 105°
theorem ABD_angle : mangle A B D = 105 :=
sorry

end ABD_angle_l506_506278


namespace find_n_consecutive_composite_l506_506846

def condition (n : ℕ) : Prop :=
  ∀ k, k < n → (n ! - k > 1 → (k ! ≠ n ! - k))

theorem find_n_consecutive_composite :
  {n : ℕ | condition n} = {1, 2, 3, 4} :=
by
  sorry

end find_n_consecutive_composite_l506_506846


namespace extreme_points_range_of_a_l506_506950

noncomputable def f (a x : ℝ) : ℝ := a * (x - 2) * Real.exp x + Real.log x + 1 / x

noncomputable def f_derivative (a x : ℝ) : ℝ :=
  a * (x - 1) * Real.exp x + 1 / x - 1 / (x ^ 2)

theorem extreme_points_range_of_a :
  (∀ a : ℝ, (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 2 ∧ 0 < x2 ∧ x2 < 2 ∧ x1 ≠ x2 ∧ f_derivative a x1 = 0 ∧ f_derivative a x2 = 0) →
  (a ∈ Set.Ioo (-∞) (-1 / Real.exp 1) ∪ Set.Ioo (-1 / Real.exp 1) (-1 / (4 * Real.exp 2)))) :=
sorry

end extreme_points_range_of_a_l506_506950


namespace image_preimage_f_l506_506477

-- Defining the function f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Given conditions
def A : Set (ℝ × ℝ) := {p | True}
def B : Set (ℝ × ℝ) := {p | True}

-- Proof statement
theorem image_preimage_f :
  f (1, 3) = (4, -2) ∧ ∃ x y : ℝ, f (x, y) = (1, 3) ∧ (x, y) = (2, -1) :=
by
  sorry

end image_preimage_f_l506_506477


namespace total_scoops_l506_506325

-- Definitions
def single_cone_scoops : ℕ := 1
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def waffle_bowl_scoops : ℕ := banana_split_scoops + 1

-- Theorem statement: Prove the total number of scoops is 10.
theorem total_scoops : single_cone_scoops + double_cone_scoops + banana_split_scoops + waffle_bowl_scoops = 10 :=
by
  -- Proof steps would go here
  sorry

end total_scoops_l506_506325


namespace inequality_nonempty_solution_set_l506_506867

theorem inequality_nonempty_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x-3| + |x-4| < a) ↔ a > 1 :=
by
  sorry

end inequality_nonempty_solution_set_l506_506867


namespace ice_cream_scoops_l506_506319

def scoops_of_ice_cream : ℕ := 1 -- single cone has 1 scoop

def scoops_double_cone : ℕ := 2 * scoops_of_ice_cream -- double cone has two times the scoops of a single cone

def scoops_banana_split : ℕ := 3 * scoops_of_ice_cream -- banana split has three times the scoops of a single cone

def scoops_waffle_bowl : ℕ := scoops_banana_split + 1 -- waffle bowl has one more scoop than banana split

def total_scoops : ℕ := scoops_of_ice_cream + scoops_double_cone + scoops_banana_split + scoops_waffle_bowl

theorem ice_cream_scoops : total_scoops = 10 :=
by
  sorry

end ice_cream_scoops_l506_506319


namespace magnitude_of_difference_l506_506533

variable (λ : ℝ)
def vector_a := (λ, 1 : ℝ × ℝ)
def vector_b := (-1, 2 : ℝ × ℝ)
def collinear := ∃ k : ℝ, (vector_a λ).1 = k * vector_b.1 ∧ (vector_a λ).2 = k * vector_b.2

theorem magnitude_of_difference (h : collinear λ) : Real.sqrt ((vector_a λ).1 - (vector_b).1)^2 + ((vector_a λ).2 - (vector_b).2)^2 = Real.sqrt 5 / 2 :=
by
  sorry

end magnitude_of_difference_l506_506533


namespace ellipse_property_l506_506896

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), a^2 = 2 ∧ b = 1 ∧ 
  (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 2 + y^2 = 1))

noncomputable def fixed_point_exists : Prop :=
  ∃ T : ℝ × ℝ, T = (0, 1) ∧ 
  (∀ L : ℝ → ℝ, ∃ A B : ℝ × ℝ, 
    (A.2 = L A.1 ∧ B.2 = L B.1) ∧
    (A ∈ set_of (λ p : ℝ × ℝ, p.1^2 / 2 + p.2^2 = 1)) ∧ 
    (B ∈ set_of (λ p : ℝ × ℝ, p.1^2 / 2 + p.2^2 = 1)) ∧
    (∃ C : ℝ × ℝ, C = (A.1 + B.1) / 2, C.2 = (A.2 + B.2) / 2, C = T))

theorem ellipse_property : ellipse_equation ∧ fixed_point_exists := 
  sorry

end ellipse_property_l506_506896


namespace william_total_riding_hours_l506_506733

-- Define the conditions and the question as a Lean theorem statement
theorem william_total_riding_hours :
  ∃ (total_hours : ℕ), 
    total_hours = ((6 * 2) + (1.5 * 2) + ((6 / 2) * 2)) :=
begin
  use 21,
  -- Provide conditions
  have max_hours_days : 6 * 2 = 12 := by norm_num,
  have half_max_hours_days : (6 / 2) * 2 = 6 := by norm_num,
  have fractional_hours_days : (3 / 2) * 2 = 3 := by norm_num,
  
  -- Show the total riding hours.
  have total_hours : 21 = 12 + 3 + 6 := by norm_num,
  exact total_hours.symm
end

end william_total_riding_hours_l506_506733


namespace path_count_1800_l506_506057

-- Define the coordinates of the points
def A := (0, 8)
def B := (4, 5)
def C := (7, 2)
def D := (9, 0)

-- Function to calculate the number of combinatorial paths
def comb_paths (steps_right steps_down : ℕ) : ℕ :=
  Nat.choose (steps_right + steps_down) steps_right

-- Define the number of steps for each segment
def steps_A_B := (4, 2)  -- 4 right, 2 down
def steps_B_C := (3, 3)  -- 3 right, 3 down
def steps_C_D := (2, 2)  -- 2 right, 2 down

-- Calculate the number of paths for each segment
def paths_A_B := comb_paths steps_A_B.1 steps_A_B.2
def paths_B_C := comb_paths steps_B_C.1 steps_B_C.2
def paths_C_D := comb_paths steps_C_D.1 steps_C_D.2

-- Calculate the total number of paths combining all segments
def total_paths : ℕ :=
  paths_A_B * paths_B_C * paths_C_D

theorem path_count_1800 :
  total_paths = 1800 := by
  sorry

end path_count_1800_l506_506057


namespace james_initial_marbles_l506_506598

theorem james_initial_marbles (m n : ℕ) (h1 : n = 4) (h2 : m / (n - 1) = 21) :
  m = 28 :=
by sorry

end james_initial_marbles_l506_506598


namespace midpoints_form_square_l506_506607

-- Definitions of points and midpoints
variables {A B C D A' B' C' D' : ℝ × ℝ}

-- Conditions
def are_squares_oriented_same (A B C D A' B' C' D' : ℝ × ℝ) : Prop :=
  let v1 := (B.1 - A.1, B.2 - A.2),
      v2 := (C.1 - B.1, C.2 - B.2),
      v3 := (D.1 - C.1, D.2 - C.2),
      v4 := (A.1 - D.1, A.2 - D.2),
      v1' := (B'.1 - A'.1, B'.2 - A'.2),
      v2' := (C'.1 - B'.1, C'.2 - B'.2),
      v3' := (D'.1 - C'.1, D'.2 - C'.2),
      v4' := (A'.1 - D'.1, A'.2 - D'.2)
  in (v1 = v3) ∧ (v2 = v4) ∧ (v1' = v3') ∧ (v2' = v4')

-- The theorem statement
theorem midpoints_form_square (h1 : are_squares_oriented_same A B C D A' B' C' D') :
  let W := ((A.1 + A'.1) / 2, (A.2 + A'.2) / 2),
      X := ((B.1 + B'.1) / 2, (B.2 + B'.2) / 2),
      Y := ((C.1 + C'.1) / 2, (C.2 + C'.2) / 2),
      Z := ((D.1 + D'.1) / 2, (D.2 + D'.2) / 2)
  in -- prove that WXYZ is a square
  ∃ (s : ℝ), s > 0 ∧ 
    dist W X = s ∧ dist X Y = s ∧ dist Y Z = s ∧ dist Z W = s ∧
    slope_eq W X Y ∧ slope_eq X Y Z ∧ slope_eq Y Z W ∧ slope_eq Z W X :=
sorry

end midpoints_form_square_l506_506607


namespace Cartesian_equation_C1_intersection_C3_C1_intersection_C3_C2_l506_506578

-- Define C1 as a parametric curve
def C1 (t : ℝ) : ℝ × ℝ := ( (2 + t) / 6, real.sqrt t)

-- Define the Cartesian equation of C1
def Cartesian_C1 (x y : ℝ) : Prop := y ^ 2 = 6 * x - 2 ∧ y >= 0

-- Define C2 as a parametric curve
def C2 (s : ℝ) : ℝ × ℝ := ( -(2 + s) / 6, -real.sqrt s)

-- Define the Cartesian equation of C2
def Cartesian_C2 (x y : ℝ) : Prop := y ^ 2 = -6 * x - 2 ∧ y <= 0

-- Define the polar coordinate equation of C3
def Cartesian_C3 (x y : ℝ) : Prop := 2 * x - y = 0

-- Proving the correctness of converting parametric to Cartesian equation for C1
theorem Cartesian_equation_C1 {t : ℝ} : ∃ y, ∃ x, C1 t = (x, y) → Cartesian_C1 x y := 
sorry

-- Proving the correctness of intersection points of C3 with C1
theorem intersection_C3_C1 : 
  ∃ x1 y1, ∃ x2 y2, Cartesian_C3 x1 y1 ∧ Cartesian_C1 x1 y1 ∧ Cartesian_C3 x2 y2 ∧ Cartesian_C1 x2 y2 ∧ 
  ((x1 = 1 / 2 ∧ y1 = 1) ∧ (x2 = 1 ∧ y2 = 2)) :=
sorry

-- Proving the correctness of intersection points of C3 with C2
theorem intersection_C3_C2 : 
  ∃ x1 y1, ∃ x2 y2, Cartesian_C3 x1 y1 ∧ Cartesian_C2 x1 y1 ∧ Cartesian_C3 x2 y2 ∧ Cartesian_C2 x2 y2 ∧ 
  ((x1 = -1 / 2 ∧ y1 = -1) ∧ (x2 = -1 ∧ y2 = -2)) :=
sorry

end Cartesian_equation_C1_intersection_C3_C1_intersection_C3_C2_l506_506578


namespace term_5th_in_sequence_l506_506102

theorem term_5th_in_sequence : 
  ∃ n : ℕ, n = 5 ∧ ( ∃ t : ℕ, t = 28 ∧ 3^t ∈ { 3^(7 * (k - 1)) | k : ℕ } ) :=
by {
  sorry
}

end term_5th_in_sequence_l506_506102


namespace ratio_AD_AB_l506_506984

theorem ratio_AD_AB (a b c : ℝ) (A B C D E : ℝ → ℝ → Type) (angle_A : ℝ)
  (angle_B : ℝ) (angle_ADE : ℝ) (area_ratio : Type)
  (h1 : angle_A = 60)
  (h2 : angle_B = 45)
  (h3 : angle_ADE = 45)
  (h4 : area_ratio = 0.5) :
  ∃ ratio : ℝ, ratio = (Real.sqrt 6 + Real.sqrt 2) / (4 * Real.sqrt 2) ∧ ratio * AB = AD :=
by sorry

end ratio_AD_AB_l506_506984


namespace part1_part2_part3_l506_506913

def f (a : ℝ) (x : ℝ) := log a ((1 - x) / (1 + x))

theorem part1 (h : f 3 (-4 / 5) = 2) : ∀ x, f 3 x = log 3 ((1 - x) / (1 + x)) := 
sorry

def g (x : ℝ) := (1 - x) / (1 + x)

theorem part2 : ∀ x1 x2, x1 ∈ set.Ioo (-1 : ℝ) 1 → x2 ∈ set.Ioo (-1 : ℝ) 1 → x1 < x2 → g x1 > g x2 := 
sorry

theorem part3 (h : ∀ x1 x2, x1 ∈ set.Ioo (-1 : ℝ) 1 → x2 ∈ set.Ioo (-1 : ℝ) 1 → x1 < x2 → g x1 > g x2) 
    : ∀ t : ℝ, 1 < t ∧ t < 3 / 2 ↔ f 3 (2 * t - 2) < 0 :=
sorry

end part1_part2_part3_l506_506913


namespace fishes_per_body_of_water_l506_506088

-- Define the number of bodies of water
def n_b : Nat := 6

-- Define the total number of fishes
def n_f : Nat := 1050

-- Prove the number of fishes per body of water
theorem fishes_per_body_of_water : n_f / n_b = 175 := by 
  sorry

end fishes_per_body_of_water_l506_506088


namespace basis_preserved_projection_not_equal_coplanar_not_sufficient_cosine_angle_l506_506727

variable {V : Type} [AddCommGroup V] [Module ℝ V]

-- Option A: Basis problem
theorem basis_preserved
  (a b c : V)
  (hbasis : LinearIndependent ℝ ![a, b, c] ∧ Span ℝ (![a, b, c]) = ⊤) : 
  LinearIndependent ℝ ![a - b, b, b - c] ∧ Span ℝ (![a - b, b, b - c]) = ⊤ := sorry

-- Option B: Projection problem
theorem projection_not_equal
  (a b : ℝ × ℝ × ℝ)
  (ha : a = (4, -2, 9))
  (hb : b = (1, 2, 2)) :
  let proj := (18 / (1^2 + 2^2 + 2^2)) * b in
  proj ≠ (3, 6, 6) := sorry

-- Option C: Coplanar points
theorem coplanar_not_sufficient
  (A B C D : ℝ × ℝ × ℝ)
  (hcoplanar : ∃ (t : ℝ), (1 - t)•C + t•D = A ∧ ∃ (s : ℝ), (1 - s)•C + s•D = B) :
  ¬ (∃ (x y : ℝ), B - A = x • (C - A) + y • (D - A)) := sorry

-- Option D: Cosine of angle between lines
theorem cosine_angle
  (A B C D : ℝ × ℝ × ℝ)
  (hA : A = (1, 0, 2))
  (hB : B = (0, 1, 2))
  (hC : C = (1, 3, 0))
  (hD : D = (-1, 2, 2)) :
  let AB := (0 - 1, 1 - 0, 2 - 2)
  let CD := (-1 - 1, 2 - 3, 2 - 0)
  let AB_dot_CD := ((-1) * (-2)) + (1 * (-1)) + (0 * 2)
  let norm_AB := real.sqrt ((-1)^2 + 1^2 + 0^2)
  let norm_CD := real.sqrt ((-2)^2 + (-1)^2 + 2^2)
  real.cos (AB_dot_CD / (norm_AB * norm_CD)) = (real.sqrt 2) / 6 := sorry

end basis_preserved_projection_not_equal_coplanar_not_sufficient_cosine_angle_l506_506727


namespace circle_equation_and_triangle_area_l506_506895

theorem circle_equation_and_triangle_area :
  (∃ b < 2, 
     let M := (0, b) in
     let r := 1 in
     let chord_len_square := 4 * 5 / 25 in
     let chord_len := Real.sqrt chord_len_square in
     let line_eq := (x: ℝ) → 2 * x + 2 in
     let distance_eq := (-b + 2) / Real.sqrt 5 in
     abs distance_eq = chord_len → eq (x^2 + (y - b)^2) (r^2)) ∧ 
  (∀ t : ℝ, -4 ≤ t ≤ -1 → 
     let A := (t, 0) in
     let B := (t + 5, 0) in
     let tan_slope_AC := -2 * t / (t^2 - 1) in
     let tan_slope_BC := -2 * (t + 5) / ((t + 5)^2 - 1) in
     let line_AC := (x: ℝ) → tan_slope_AC * (x - t) in
     let line_BC := (x: ℝ) → tan_slope_BC * (x - t - 5) in
     let x_sol := (2 * t + 5) / (t^2 + 5 * t + 1) in
     let y_sol := 2 - 2 / (t^2 + 5 * t + 1) in
     let area_ABC := Real.abs ((1 / 2) * 5 * y_sol) in
     min (t, -4 < t < -1 → area_ABC) = 125 / 21) :=
sorry

end circle_equation_and_triangle_area_l506_506895


namespace find_first_term_geometric_series_l506_506211

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l506_506211


namespace area_of_triangle_l506_506036

theorem area_of_triangle (
  P : ℝ × ℝ,
  F₁ F₂ : ℝ × ℝ,
  (h₁ : P ∈ {XY : ℝ × ℝ | XY.1^2 / 8 + XY.2^2 / 2 = 1}),
  (h₂ : ∠F₁ P F₂ = 2 * Real.pi / 3)
) : ∃ A : ℝ, A = 2 * Real.sqrt 3 := 
sorry

end area_of_triangle_l506_506036


namespace probability_of_reaching_boundary_within_5_hops_l506_506869

def position := (ℕ × ℕ)  -- Represents a position on the grid

def transition (pos : position) : Probability (position) := 
  Probability.of_fintype $ [(3, 2), (3, 4), (2, 3), (4, 3)] -- Possible moves after initial move from (3, 3)

def is_target (pos : position) : Prop :=
  pos.snd = 1 ∨ pos.snd = 4 ∨ pos.fst = 1 ∨ pos.fst = 4

def reaches_target_within_5_hops : Probability (position) :=
  -- Sum the probabilities of reaching a target state within 5 hops
  sorry

theorem probability_of_reaching_boundary_within_5_hops : reaches_target_within_5_hops = 15/16 :=
  sorry

end probability_of_reaching_boundary_within_5_hops_l506_506869


namespace square_area_l506_506090

theorem square_area (W X Y Z M N S : Type) (dist : W → W → Nat)
  (square_WXYZ : square W X Y Z)
  (M_on_WZ : M ∈ segment_contains W Z)
  (N_on_WX : N ∈ segment_contains W X)
  (perpendicular_intersect : ∃ S, perpendicularly_intersect (line_through W M) (line_through Z N) S)
  (WS_eq_8 : dist W S = 8)
  (MS_eq_9 : dist M S = 9) :
  area square_WXYZ = 353 := 
sorry

end square_area_l506_506090


namespace measure_of_angle_A_l506_506490

variable {a b c A : ℝ}

theorem measure_of_angle_A (h : (b - c)^2 = a^2 - b * c) : A = π / 3 :=
by
  have h1 : b^2 + c^2 - a^2 = b * c := by sorry
  have h2 : cos A = (b^2 + c^2 - a^2) / (2 * b * c) := by sorry
  have h3 : cos A = 1 / 2 := by sorry
  have h4 : A = π / 3 := by sorry
  exact h4

end measure_of_angle_A_l506_506490


namespace at_least_one_gte_one_l506_506991

theorem at_least_one_gte_one
  (a : ℝ) (n : ℕ) (P : Polynomial ℝ)
  (h_deg : P.degree = n)
  (h_a : a ≥ 3) : 
  ∃ i (H : i ∈ Finset.range (n + 2)), |a ^ i - Polynomial.eval (i : ℝ) P| ≥ 1 := 
begin
  sorry -- proof should go here
end

end at_least_one_gte_one_l506_506991


namespace min_non_negative_min_sub_product_non_negative_exp_non_negative_l506_506160

-- Function definitions
def min (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Theorem 1: minimum function on ℝ₊²
theorem min_non_negative (s t : ℝ) (hs : 0 ≤ s) (ht : 0 ≤ t) : 0 ≤ min s t := 
sorry

-- Theorem 2: min(s, t) - st on [0, 1]²
theorem min_sub_product_non_negative (s t : ℝ) (hs : 0 ≤ s ∧ s ≤ 1) (ht : 0 ≤ t ∧ t ≤ 1) : 0 ≤ min s t - s * t := 
sorry

-- Theorem 3: exponential function on ℝ²
theorem exp_non_negative (s t : ℝ) : 0 ≤ real.exp (-real.abs (t - s)) := 
sorry

end min_non_negative_min_sub_product_non_negative_exp_non_negative_l506_506160


namespace thirty_percent_less_than_90_eq_one_fourth_more_than_what_number_l506_506239

theorem thirty_percent_less_than_90_eq_one_fourth_more_than_what_number :
  ∃ (n : ℤ), (5 / 4 : ℝ) * (n : ℝ) = 90 - (0.30 * 90) ∧ n ≈ 50 := 
by
  -- Existence condition for n
  use 50
  -- Proof of equivalence (optional for statement)
  sorry

end thirty_percent_less_than_90_eq_one_fourth_more_than_what_number_l506_506239


namespace find_x_tan_eq_l506_506427

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l506_506427


namespace solution_set_eq_l506_506497

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def decreasing_condition (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 ≠ x2 → (x1 * f (x1) - x2 * f (x2)) / (x1 - x2) < 0

variable (f : ℝ → ℝ)
variable (h_odd : odd_function f)
variable (h_minus_2_zero : f (-2) = 0)
variable (h_decreasing : decreasing_condition f)

theorem solution_set_eq :
  {x : ℝ | f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 2 < x} :=
sorry

end solution_set_eq_l506_506497


namespace average_customers_per_day_l506_506795

-- Define the number of customers each day:
def customers_per_day : List ℕ := [10, 12, 15, 13, 18, 16, 11]

-- Define the total number of days in a week
def days_in_week : ℕ := 7

-- Define the theorem stating the average number of daily customers
theorem average_customers_per_day :
  (customers_per_day.sum : ℚ) / days_in_week = 13.57 :=
by
  sorry

end average_customers_per_day_l506_506795


namespace count_divisible_digits_by_n_l506_506462

def forms_number (a b : ℕ) : ℕ := a * 10 + b

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem count_divisible_digits_by_n : (finset.range 10).filter (λ n, is_divisible (forms_number 14 n) n).card = 5 :=
by sorry

end count_divisible_digits_by_n_l506_506462


namespace find_a_l506_506521

-- Define given parameters and conditions
def parabola_eq (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

def shifted_parabola_eq (a : ℝ) (x : ℝ) : ℝ := parabola_eq a x - 3 * |a|

-- Define axis of symmetry function
def axis_of_symmetry (a : ℝ) : ℝ := 1

-- Conditions: a ≠ 0
variable (a : ℝ)
variable (h : a ≠ 0)

-- Define value for discriminant check
def discriminant (a : ℝ) (c : ℝ) : ℝ := (-2 * a)^2 - 4 * a * c

-- Problem statement
theorem find_a (ha : a ≠ 0) : 
  (axis_of_symmetry a = 1) ∧ (discriminant a (3 - 3 * |a|) = 0 → (a = 3 / 4 ∨ a = -3 / 2)) := 
by
  sorry -- proof to be filled in

end find_a_l506_506521


namespace problem1_problem2_l506_506526

noncomputable def l1_requires := 4*x + y - 4 = 0
noncomputable def l2_requires (m : ℝ) := m*x + y = 0
noncomputable def l3_requires (m : ℝ) := x - m*y - 4 = 0

theorem problem1 (m : ℝ) :
  ¬Collinear {l1_requires, l2_requires m, l3_requires m} →
  (m = 4 ∨ m = -1/4) := sorry

theorem problem2 (m : ℝ) :
  Perpendicular (l3_requires m) (l1_requires) ∧ Perpendicular (l3_requires m) (l2_requires m) →
  m = -4 ∧ distance_parallel (4*x + y - 4 = 0) (4*(-4) + y = 0) = (4*sqrt(17))/17 := sorry

end problem1_problem2_l506_506526


namespace find_first_term_geometric_series_l506_506209

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l506_506209


namespace intersection_with_XOZ_plane_is_C_l506_506103

-- Define the points A and B
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := { x := 1, y := -2, z := -3 }
def B : Point3D := { x := 2, y := -1, z := -1 }

-- Define the direction vector from A to B
def direction_vector (p1 p2 : Point3D) : Point3D :=
  { x := p2.x - p1.x, y := p2.y - p1.y, z := p2.z - p1.z }

-- Define the intersection point C
def C : Point3D := { x := 3, y := 0, z := 1 }

-- Lean theorem statement to prove the intersection point is as defined
theorem intersection_with_XOZ_plane_is_C : 
  (∃ λ : ℝ, 
    C.x = A.x + λ * (B.x - A.x) ∧ 
    C.y = A.y + λ * (B.y - A.y) ∧ 
    C.z = A.z + λ * (B.z - A.z) ∧ 
    C.y = 0) :=
by
  sorry

end intersection_with_XOZ_plane_is_C_l506_506103


namespace maciek_total_cost_l506_506789

-- Define the cost of pretzels and the additional cost percentage for chips
def cost_pretzel : ℝ := 4
def cost_chip := cost_pretzel + (cost_pretzel * 0.75)

-- Number of packets Maciek bought for pretzels and chips
def num_pretzels : ℕ := 2
def num_chips : ℕ := 2

-- Total cost calculation
def total_cost := (cost_pretzel * num_pretzels) + (cost_chip * num_chips)

-- The final theorem statement
theorem maciek_total_cost :
  total_cost = 22 := by
  sorry

end maciek_total_cost_l506_506789


namespace total_points_l506_506965

def points_earned (goblins orcs dragons : ℕ): ℕ :=
  goblins * 3 + orcs * 5 + dragons * 10

theorem total_points :
  points_earned 10 7 1 = 75 :=
by
  sorry

end total_points_l506_506965


namespace program_equivalence_l506_506818

def Program_A_sum (n : ℕ) : ℕ :=
  (list.range (n + 1)).sum

def Program_B_sum (n : ℕ) : ℕ :=
  (list.range' (1) (n)).reverse.sum

theorem program_equivalence : 
  Program_A_sum 1000 = 500500 ∧ Program_B_sum 1000 = 500500 :=
by
  sorry

end program_equivalence_l506_506818


namespace nth_term_sequence_l506_506522

theorem nth_term_sequence (n : ℕ) : 
  let a_n := 2 * n^3 - 3 * n^2 + 3 * n - 1
  in 
  (λ k, if k = 1 then 1
  else if k <= (2 * n - 1) 
  then (n - (2n-1 - k) // n) * (2 * (n // ((2 * n - 1) // 2))^2) 
  else 0) = a_n :=
by
  sorry

end nth_term_sequence_l506_506522


namespace find_x_between_0_and_180_l506_506423

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l506_506423


namespace find_y_l506_506065

theorem find_y (x y : ℤ) (h₁ : x ^ 2 + x + 4 = y - 4) (h₂ : x = 3) : y = 20 :=
by 
  sorry

end find_y_l506_506065


namespace incorrect_calculation_l506_506259

theorem incorrect_calculation : sqrt 2 + sqrt 3 ≠ sqrt 5 := by
  sorry

end incorrect_calculation_l506_506259


namespace find_first_term_l506_506200

theorem find_first_term
  (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  -- Proof is omitted for brevity
  sorry

end find_first_term_l506_506200


namespace symmetric_distance_l506_506277

noncomputable def distanceAB (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem symmetric_distance
  (A B : ℝ × ℝ)
  (hA : A.2 = 3 - A.1^2)
  (hB : B.2 = 3 - B.1^2)
  (h_symm : A.1 + A.2 = 0 ∧ B.1 + B.2 = 0 ∧ A ≠ B ∧ B = (A.2, A.1)) :
  distanceAB A B = 3 * real.sqrt 2 := 
sorry

end symmetric_distance_l506_506277


namespace main_theorem_l506_506622

variable (f : ℝ → ℝ)

-- Conditions: f(x) > f'(x) for all x ∈ ℝ
def condition (x : ℝ) : Prop := f x > (derivative f) x

-- Main statement to prove
theorem main_theorem  (h : ∀ x : ℝ, condition f x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) := 
by 
  sorry

end main_theorem_l506_506622


namespace same_first_last_letter_words_l506_506604

theorem same_first_last_letter_words : 
  let n := 5 in 
  let alphabet_size := 26 in 
  let num_valid_words := alphabet_size^(n - 1) 
  in num_valid_words = 456976 :=
by
  let n := 5
  let alphabet_size := 26
  let num_valid_words := alphabet_size^(n - 1)
  show num_valid_words = 456976 from sorry

end same_first_last_letter_words_l506_506604


namespace isosceles_triangle_of_condition_l506_506032

variables (A B C P : ℝ^3)

-- Condition 1: P is an interior point of triangle ABC (excluding the boundary)
def is_interior_point (A B C P : ℝ^3) : Prop :=
  ∃ (u v w : ℝ), 0 < u ∧ 0 < v ∧ 0 < w ∧ u + v + w = 1 ∧ u • A + v • B + w • C = P

-- Condition 2: \((\overrightarrow{PB} - \overrightarrow{PA}) \cdot (\overrightarrow{PB} + \overrightarrow{PA} - 2\overrightarrow{PC}) = 0\)
def given_condition (A B C P : ℝ^3) : Prop :=
  let PA := P - A;
      PB := P - B;
      PC := P - C in
  (PB - PA) • (PB + PA - 2 * PC) = 0

-- Statement to prove: \(\triangle ABC\) is an isosceles triangle
theorem isosceles_triangle_of_condition
  (A B C P : ℝ^3) (h1: is_interior_point A B C P) (h2: given_condition A B C P) : 
  ∃ (AB_eq_AC : ℝ), dist A B = dist A C ∨ dist B A = dist B C ∨ dist C A = dist C B :=
sorry

end isosceles_triangle_of_condition_l506_506032


namespace area_inequality_and_equality_condition_l506_506994

-- We denote the lengths and formulas for the areas as per the conditions

variable (a b c d F₁ F₂ F₃ : ℝ)

-- Definitions for the areas of the triangles, given the quadrilateral area
def area_triangle_ABE := (a / (a + c)) * (b / (b + d)) * F₃
def area_triangle_CDE := (c / (a + c)) * (d / (b + d)) * F₃

-- Main statement proving the inequality and the equality condition
theorem area_inequality_and_equality_condition (h₁ : F₁ = area_triangle_ABE a b c d F₃)
                                               (h₂ : F₂ = area_triangle_CDE a b c d F₃) :
  (real.sqrt F₁ + real.sqrt F₂ <= real.sqrt F₃) ∧
  (real.sqrt F₁ + real.sqrt F₂ = real.sqrt F₃ ↔ a / b = c / d) :=
by
  sorry


end area_inequality_and_equality_condition_l506_506994


namespace find_x_tan_identity_l506_506444

theorem find_x_tan_identity : 
  ∀ x : ℝ, (0 < x ∧ x < 180) → 
  (Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) → 
  x = 105 :=
by
  intro x hx htan
  sorry

end find_x_tan_identity_l506_506444


namespace tan_three_neg_l506_506226

theorem tan_three_neg : (real.tan 3) < 0 :=
by
  -- Given that 3 radians is in the second quadrant where sine is positive and cosine is negative,
  -- we can conclude that the tangent of 3 radians (sin 3 / cos 3) is negative.
  sorry

end tan_three_neg_l506_506226


namespace part_i_part_ii_l506_506121

theorem part_i (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let h := min a (b / (a^2 + b^2)) in h ≤ (Real.sqrt 2) / 2 :=
by
  let h := min a (b / (a^2 + b^2))
  sorry

theorem part_ii (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let H := max (1 / Real.sqrt a) ((a^2 + b^2) / Real.sqrt (a * b)) (1 / Real.sqrt b) in
  ∃ c : ℝ, H = c ∧ c ≥ (Real.cbrt 2) :=
by
  let H := max (1 / Real.sqrt a) ((a^2 + b^2) / Real.sqrt (a * b)) (1 / Real.sqrt b)
  exists (Real.cbrt 2)
  split
  sorry
  sorry

end part_i_part_ii_l506_506121


namespace perpendicular_AX_XC_l506_506966

theorem perpendicular_AX_XC
  (A B C D X M : Type)
  [acute_triangle : is_acute_triangle A B C]
  [bisector : angle_bisector A B C = D]
  [midpoint_AD : is_midpoint M A D]
  [on_segment_BM : on_segment X B M]
  [angle_equality : angle MX X A = angle D A C] :
  perpendicular (line A X) (line X C) :=
sorry

end perpendicular_AX_XC_l506_506966


namespace nth_equation_l506_506506

noncomputable def nested_radicals (n : ℕ) : ℝ :=
  Nat.iterate (λ x : ℝ, sqrt (2 + x)) n 0

theorem nth_equation (n : ℕ) : 
  nested_radicals n = 2 * real.cos (π / 2^(n + 1)) := 
sorry

end nth_equation_l506_506506


namespace least_positive_difference_is_one_l506_506654

def geometric_sequence (a r : ℕ) : ℕ → ℕ
| 0       := a
| (n + 1) := r * (geometric_sequence n)

def arithmetic_sequence (a d : ℕ) : ℕ → ℕ
| 0       := a
| (n + 1) := d + (arithmetic_sequence n)

def sequence_C : ℕ → ℕ := geometric_sequence 3 3
def sequence_D : ℕ → ℕ := arithmetic_sequence 30 10

def stops_below (seq : ℕ → ℕ) (n : ℕ) : ℕ → ℕ
| 0       := if seq 0 ≤ n then seq 0 else n + 1
| (i + 1) := let prev := stops_below i in
             if prev ≤ n ∧ seq (i + 1) ≤ n then seq (i + 1) else prev

def valid_terms_C := stops_below sequence_C 1000
def valid_terms_D := stops_below sequence_D 1000

noncomputable def min_positive_difference : ℕ :=
  let diffs := {d | ∃ i j, d = abs (valid_terms_C i - valid_terms_D j) ∧ d > 0} in
  Inf diffs

theorem least_positive_difference_is_one : min_positive_difference = 1 :=
by
  sorry

end least_positive_difference_is_one_l506_506654


namespace smallest_five_digit_congruent_to_3_mod_17_l506_506723

theorem smallest_five_digit_congruent_to_3_mod_17 : ∃ n : ℤ, (n > 9999) ∧ (n % 17 = 3) ∧ (∀ m : ℤ, (m > 9999) ∧ (m % 17 = 3) → n ≤ m) :=
by
  use 10012
  split
  { sorry }
  split
  { sorry }
  { sorry }

end smallest_five_digit_congruent_to_3_mod_17_l506_506723


namespace theo_min_assignments_l506_506301

theorem theo_min_assignments : 
  let pts1 := 6 in
  let pts2 := 6 in
  let pts3 := 6 in
  let pts4 := 6 in
  let pts5 := 6 in
  let assignments1 := 1 * pts1 in
  let assignments2 := 2 * pts2 in
  let assignments3 := 4 * pts3 in
  let assignments4 := 8 * pts4 in
  let assignments5 := 16 * pts5 in
  assignments1 + assignments2 + assignments3 + assignments4 + assignments5 = 186 :=
by
  let pts1 := 6
  let pts2 := 6
  let pts3 := 6
  let pts4 := 6
  let pts5 := 6
  let assignments1 := 1 * pts1
  let assignments2 := 2 * pts2
  let assignments3 := 4 * pts3
  let assignments4 := 8 * pts4
  let assignments5 := 16 * pts5
  have h : assignments1 + assignments2 + assignments3 + assignments4 + assignments5 = 186 := sorry
  exact h

end theo_min_assignments_l506_506301


namespace maciek_total_purchase_cost_l506_506781

-- Define the cost of pretzels
def pretzel_cost : ℕ := 4

-- Define the cost of chips
def chip_cost : ℕ := pretzel_cost + (75 * pretzel_cost) / 100

-- Calculate the total cost
def total_cost : ℕ := 2 * pretzel_cost + 2 * chip_cost

-- Rewrite the math proof problem statement
theorem maciek_total_purchase_cost : total_cost = 22 :=
by
  -- Skip the proof
  sorry

end maciek_total_purchase_cost_l506_506781


namespace parallel_vectors_l506_506078

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2, 3)) (h₂ : b = (-1, 2)) :
  (m * a.1 + b.1) * (-1) - 4 * (m * a.2 + b.2) = 0 → m = -1 / 2 :=
by
  intro h
  rw [h₁, h₂] at h
  simp at h
  sorry

end parallel_vectors_l506_506078


namespace f_is_odd_f_is_increasing_g_is_not_odd_g_range_l506_506871

def f (x : ℝ) : ℝ := (2^x - 1) / (1 + 2^x)
def g (x : ℝ) : ℤ := ⌊f x⌋

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := 
by sorry

theorem f_is_increasing : ∀ x y : ℝ, x < y → f x < f y := 
by sorry

theorem g_is_not_odd : ∃ x : ℝ, g (-x) ≠ -g x := 
by sorry

theorem g_range : ∀ y : ℤ, y ∈ set.range g → y = -1 ∨ y = 0 := 
by sorry

end f_is_odd_f_is_increasing_g_is_not_odd_g_range_l506_506871


namespace mutually_exclusive_but_not_opposite_l506_506082

-- Define the total number of each type of ball
def red_balls : ℕ := 3
def white_balls : ℕ := 2
def black_balls : ℕ := 1
def total_balls : ℕ := red_balls + white_balls + black_balls

-- Define the events
def event_A : Prop := "both balls drawn are white"
def event_B : Prop := "at least one red ball"
def event_C : Prop := "at least one black ball"
def event_D : Prop := "one red ball and one black ball"
def event_X : Prop := "at least one white ball"

-- Statement: Confirm event D and event X are mutually exclusive but not opposite
theorem mutually_exclusive_but_not_opposite
  (h1 : red_balls = 3) (h2 : white_balls = 2) (h3 : black_balls = 1) (h4 : total_balls = 6) :
  (event_D ∧ ¬ event_X) ∧ (event_D ∨ event_X ∧ ¬ (event_D ∧ event_X)) := 
sorry

end mutually_exclusive_but_not_opposite_l506_506082


namespace arithmetic_mean_is_correct_l506_506708

open Nat

noncomputable def arithmetic_mean_of_two_digit_multiples_of_9 : ℝ :=
  let smallest := 18
  let largest := 99
  let n := 10
  let sum := (n / 2 : ℝ) * (smallest + largest)
  sum / n

theorem arithmetic_mean_is_correct :
  arithmetic_mean_of_two_digit_multiples_of_9 = 58.5 :=
by
  -- Proof goes here
  sorry

end arithmetic_mean_is_correct_l506_506708


namespace total_cost_is_correct_l506_506832

def teaspoon_cost : ℕ := 9
def tablespoon_cost : ℕ := 12
def dessert_spoon_cost : ℕ := 18
def soupspoon_cost : ℕ := 16

def teaspoons : ℕ := 7
def tablespoons : ℕ := 10
def dessert_spoons : ℕ := 12

def exchange_rate_M_to_EUR : ℝ := 0.04
def exchange_rate_USD_to_EUR : ℝ := 1 / 1.15
def souvenir_cost_usd : ℝ := 40

def cost_per_set : ℕ := (teaspoon_cost + tablespoon_cost + 2 * dessert_spoon_cost)
def num_sets : ℕ := min teaspoons (min (tablespoons) (dessert_spoons / 2))
def total_cost_M : ℕ := num_sets * cost_per_set

def total_cost_EUR : ℝ := total_cost_M * exchange_rate_M_to_EUR
def total_souvenir_cost_EUR : ℝ := souvenir_cost_usd * exchange_rate_USD_to_EUR
 
def total_combined_cost : ℝ := total_cost_EUR + total_souvenir_cost_EUR

theorem total_cost_is_correct : total_combined_cost = 50.74 := by
  sorry

end total_cost_is_correct_l506_506832


namespace num_positive_integers_satisfying_condition_l506_506670

theorem num_positive_integers_satisfying_condition :
  ∃! (n : ℕ), 30 - 6 * n > 18 := by
  sorry

end num_positive_integers_satisfying_condition_l506_506670


namespace arithmetic_mean_of_two_digit_multiples_of_9_l506_506709

theorem arithmetic_mean_of_two_digit_multiples_of_9 : 
  let a1 := 18 in
  let an := 99 in
  let d := 9 in
  let n := (an - a1) / d + 1 in
  let S := n * (a1 + an) / 2 in
  (S / n : ℝ) = 58.5 :=
by
  let a1 := 18
  let an := 99
  let d := 9
  let n := (an - a1) / d + 1
  let S := n * (a1 + an) / 2
  show (S / n : ℝ) = 58.5
  sorry

end arithmetic_mean_of_two_digit_multiples_of_9_l506_506709


namespace number_of_integer_solutions_l506_506059

theorem number_of_integer_solutions : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x + 3)^2 ≤ 4) ∧ S.card = 5 := by
  sorry

end number_of_integer_solutions_l506_506059


namespace areas_equal_l506_506614

variables (A B C D M N P Q : Type)

-- Assume the midpoints and the points where the segments intersect
variables [geometry.quadrilateral A B C D]
  [geometry.midpoint M B C] [geometry.midpoint N A D]
  [geometry.intersect AM BN P] [geometry.intersect DM CN Q]

theorem areas_equal :
  geometry.area (A, P, B) + geometry.area (C, Q, D) = geometry.area (M, P, N, Q) :=
sorry

end areas_equal_l506_506614


namespace maciek_total_cost_l506_506785

theorem maciek_total_cost :
  let p := 4
  let cost_of_chips := 1.75 * p
  let pretzels_cost := 2 * p
  let chips_cost := 2 * cost_of_chips
  let t := pretzels_cost + chips_cost
  t = 22 :=
by
  sorry

end maciek_total_cost_l506_506785


namespace min_roads_required_l506_506970

-- Predicate to indicate a city in the kingdom
def city : Type := Fin 100

-- Relation indicating a one-way road between two cities and its color
inductive road_color
| white
| black

structure road :=
  (from to : city)
  (color : road_color)
  (one_way : Prop := true)

-- Define the conditions given in the problem.
axiom roads : list road
axiom no_duplicate_roads : ∀ r1 r2 : road, (r1.from = r2.from ∧ r1.to = r2.to) → r1 = r2
axiom alternating_path_exists : ∀ (start end : city), ∃ (path : list road), path.head.color = road_color.white ∧
  (∀ i, i < path.length - 1 → path.nth_le i _ ≠ path.nth_le (i + 1) _)

-- The statement of the problem in Lean
theorem min_roads_required : roads.length = 150 := 
sorry

end min_roads_required_l506_506970


namespace correct_propositions_l506_506309

theorem correct_propositions :
  (∀ (f : ℝ → ℝ), (∀ x, f (-x) = -f x) → (∀ x, f (x + 4) = f x) → (∀ x, f (x) = f (4 - x))) ∧
  (∀ (a : ℝ), (0 < a ∧ a < 1) → ¬ (a^(1 + a) < a^(1 + 1 / a))) ∧
  (∀ (x : ℝ), -1 < x ∧ x < 1 → ln((1 + x) / (1 - x)) = -ln((1 - x) / (1 + x))) ∧
  (¬ ∃! (a : ℝ), ∀ x, log10 (a * x + sqrt(2 * x^2 + 1)) = -log10 (a * x + sqrt(2 * x^2 + 1))) :=
by
  sorry

end correct_propositions_l506_506309


namespace sum_of_geometric_sequence_l506_506460

noncomputable def geometric_sequence_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem sum_of_geometric_sequence
  (a_1 q : ℝ) 
  (h1 : a_1^2 * q^6 = 2 * a_1 * q^2)
  (h2 : (a_1 * q^3 + 2 * a_1 * q^6) / 2 = 5 / 4)
  : geometric_sequence_sum a_1 q 4 = 30 :=
by
  sorry

end sum_of_geometric_sequence_l506_506460


namespace circle_tangent_lines_eq_l506_506195

noncomputable def circle_standard_equation (a : ℝ) : Prop :=
  let r := abs a / real.sqrt 2 in -- radius of the circle
  a = 2 ∧ ((x : ℝ) (y : ℝ), (x - 2) ^ 2 + (y - 1) ^ 2 = 2)

theorem circle_tangent_lines_eq (a : ℝ) :
  (∃ r : ℝ, ((a = 2) ∧ (r = abs a / real.sqrt 2) ∧ 
  r = abs (a - 3) / real.sqrt 2) ∧
  ((x : ℝ) (y : ℝ), (x - 2) ^ 2 + (y - 1) ^ 2 = 2)) :=
by
  sorry

end circle_tangent_lines_eq_l506_506195


namespace find_x_value_l506_506414

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l506_506414


namespace factorize_x_squared_minus_121_l506_506844

theorem factorize_x_squared_minus_121 (x : ℝ) : (x^2 - 121) = (x + 11) * (x - 11) :=
by
  sorry

end factorize_x_squared_minus_121_l506_506844


namespace science_fair_students_l506_506821

-- Define the ratios as given in the conditions
def ratio_9th_7th : Nat → Nat → Prop := λ ninth seventh, 4 * seventh = 5 * ninth
def ratio_9th_8th : Nat → Nat → Prop := λ nith eighth, 7 * eighth = 6 * nith

-- Definition of the total students participating in the science fair
def total_students (ninth seventh eighth : Nat) : Nat :=
  ninth + seventh + eighth

-- The theorem to be proved, the smallest number of students meeting the given conditions is 87
theorem science_fair_students (ninth seventh eighth : Nat) 
  (h1 : ratio_9th_7th ninth seventh) 
  (h2 : ratio_9th_8th ninth eighth) 
  : total_students ninth seventh eighth = 87 := 
sorry

end science_fair_students_l506_506821


namespace height_of_tank_A_l506_506661

theorem height_of_tank_A (C_A C_B h_B : ℝ) (capacity_ratio : ℝ) :
  C_A = 8 → C_B = 10 → h_B = 8 → capacity_ratio = 0.4800000000000001 →
  ∃ h_A : ℝ, h_A = 6 := by
  intros hCA hCB hHB hCR
  sorry

end height_of_tank_A_l506_506661


namespace company_distribution_problem_l506_506288

noncomputable def number_of_distribution_plans : ℕ :=
  let total_employees := 8 in
  let num_translators := 2 in
  let num_programmers := 3 in
  let other_employees := total_employees - (num_translators + num_programmers) in
  -- translators distribution
  let translator_distribution := 1 in
  -- programmers distribution scenarios
  let programmer_scenarios := 2 in
  -- remaining employees distribution
  let remaining_distribution := 2 * (finset.card (finset.comb (finset.range 3) 1)) -- choose 1 out of 3
                             + 2 * (finset.card (finset.comb (finset.range 3) 2)) -- choose 2 out of 3
  in
  translator_distribution * programmer_scenarios * remaining_distribution

theorem company_distribution_problem :
  number_of_distribution_plans = 36 := 
sorry

end company_distribution_problem_l506_506288


namespace find_x_l506_506385

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l506_506385


namespace count_integers_satisfying_inequality_l506_506060

theorem count_integers_satisfying_inequality : (finset.filter (λ x : ℤ, (x + 3)^2 ≤ 4) (finset.Icc -5 -1)).card = 5 := 
by
  sorry

end count_integers_satisfying_inequality_l506_506060


namespace common_ratio_range_l506_506501

theorem common_ratio_range
  {a₁ a₂ a₃ q : ℝ}
  (h_geometric : a₁ * (a₂ + a₃) = 6 * a₁ - 9)
  (h_q : a₂ = a₁ * q)
  (h_q2 : a₃ = a₁ * q^2) :
  (q = -1 ∨ ((-1 - real.sqrt 5) / 2 ≤ q ∧ q ≤ (-1 + real.sqrt 5) / 2 ∧ q ≠ 0)) :=
sorry

end common_ratio_range_l506_506501


namespace binomial_coefficient_term_of_x_l506_506093

theorem binomial_coefficient_term_of_x (n : ℕ) (x : ℝ) :
  (2^n = 128) →
  ∃ (coef : ℤ), coef = -14 ∧ ∃ r : ℕ, x^(1 : ℝ) = (sqrt (x)⁻¹) * ((-2)^r * (Nat.choose n r) * x^((n - 4 * r)/3)) :=
by
  intros h
  have h₁ : n = 7 := sorry
  use -14
  split
  . refl
  use 1
  sorry

end binomial_coefficient_term_of_x_l506_506093


namespace hyperbola_eccentricity_l506_506666

theorem hyperbola_eccentricity 
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (tangent_condition : ∀ x y, (x^2 + (y - a)^2 = a^2 / 9) → ∃ x, (y / a - x / b) = 0 = ∀ x, y, |a * x| / ∀ y, a, (√(a^2 + b^2) = a / 3) ) :
  let e := (√(9 / 8)) in e = 3 * (√2) / 4  :=
sorry

end hyperbola_eccentricity_l506_506666


namespace sine_triangle_l506_506139

theorem sine_triangle (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_perimeter : a + b + c ≤ 2 * Real.pi)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (ha_pi : a < Real.pi) (hb_pi : b < Real.pi) (hc_pi : c < Real.pi):
  ∃ (x y z : ℝ), x = Real.sin a ∧ y = Real.sin b ∧ z = Real.sin c ∧ x + y > z ∧ y + z > x ∧ x + z > y :=
by
  sorry

end sine_triangle_l506_506139


namespace sin_2alpha_value_l506_506026

theorem sin_2alpha_value (α : ℝ) 
  (h1 : cos (π * α) = -1/2) 
  (h2 : 3/2 * π < α ∧ α < 2 * π) : 
  sin (2 * α) = sqrt 3 / 4 := 
sorry

end sin_2alpha_value_l506_506026


namespace nina_shoes_total_cost_l506_506152

noncomputable def shoe_cost (original_price : ℝ) (discount_pct : ℝ) (tax_pct : ℝ) : ℝ :=
  let price_after_discount := original_price * (1 - discount_pct / 100)
  let final_price := price_after_discount * (1 + tax_pct / 100)
  final_price

theorem nina_shoes_total_cost :
  let first_pair_cost := shoe_cost 22 10 5 in
  let second_pair_cost := shoe_cost (22 * 1.5) 15 7 in
  let third_pair_cost := 40 * 1.12 in -- Considering tax directly as there is no discount and it's stated directly
  let fourth_pair_cost := 0 in -- Free pair
  first_pair_cost + second_pair_cost + third_pair_cost + fourth_pair_cost = 95.60 :=
by
  sorry

end nina_shoes_total_cost_l506_506152


namespace find_x_between_0_and_180_l506_506422

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l506_506422


namespace factorial_divides_sequence_difference_l506_506628

theorem factorial_divides_sequence_difference (a : ℕ) (ha : a ≠ 0) :
  ∀ n : ℕ, 1 ≤ n → (nat.factorial n) ∣ (nat.iterate (λ k, a^k) n 1 - nat.iterate (λ k, a^k) (n - 1) 1) :=
by 
sorry

end factorial_divides_sequence_difference_l506_506628


namespace function_minimum_value_in_interval_l506_506951

theorem function_minimum_value_in_interval (a : ℝ) :
  (∃ x ∈ Ioo (0 : ℝ) 2, ∀ y ∈ Ioo (0 : ℝ) 2, f y ≥ f x ∧ ∃ x₀ ∈ Ioo (0 : ℝ) 2, f' x₀ = 0) ↔ 0 < a ∧ a < 4 :=
by
  let f := λ x : ℝ, x^3 - 3 * a * x + a
  let f' := λ x : ℝ, 3 * x^2 - 3 * a
  sorry

end function_minimum_value_in_interval_l506_506951


namespace problem_statement_l506_506329

theorem problem_statement : (2^2015 + 2^2011) / (2^2015 - 2^2011) = 17 / 15 :=
by
  -- Prove the equivalence as outlined above.
  sorry

end problem_statement_l506_506329


namespace distance_covered_in_400_revolutions_l506_506685

noncomputable def distance_covered (r : ℝ) (n : ℝ) : ℝ :=
  2 * real.pi * r * n

theorem distance_covered_in_400_revolutions :
  distance_covered 20.4 400 = 51270.7488 :=
by
  -- Placeholder for the proof
  sorry

end distance_covered_in_400_revolutions_l506_506685


namespace find_x_between_0_and_180_l506_506416

noncomputable def pi : ℝ := Real.pi
noncomputable def deg_to_rad (deg : ℝ) : ℝ := deg * pi / 180

theorem find_x_between_0_and_180 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 180)
  (h : Real.tan (deg_to_rad 150 - deg_to_rad x) = (Real.sin (deg_to_rad 150) - Real.sin (deg_to_rad x)) / (Real.cos (deg_to_rad 150) - Real.cos (deg_to_rad x))) :
  x = 115 :=
by
  sorry

end find_x_between_0_and_180_l506_506416


namespace compute_f_pi_div_2_l506_506517

def f (x : ℝ) : ℝ := Real.sin (x / 2 + Real.pi / 4)

theorem compute_f_pi_div_2 : f (Real.pi / 2) = 1 := by
  sorry

end compute_f_pi_div_2_l506_506517


namespace water_polo_team_selection_l506_506646

theorem water_polo_team_selection :
  (18 * 17 * Nat.choose 16 5) = 1338176 := by
  sorry

end water_polo_team_selection_l506_506646


namespace math_problem_l506_506333

open Nat

noncomputable def is_special_composite (n : ℕ) : Prop :=
  ∃ (d : ℕ → ℕ) (k : ℕ), (1 < n) ∧ (n > 1) ∧ (1 = d 1) ∧ (d k = n) ∧
  (∀ i, 1 ≤ i → i < k → d (i + 1) = d i + (i * (n - d 1)) / (k - 1)) ∧
  (∀ i, 1 ≤ i → i < k - 1 → d (i + 1) > d i)

theorem math_problem : ∀ n : ℕ, is_special_composite n → n = 4 :=
begin
  sorry
end

end math_problem_l506_506333


namespace development_inheritance_false_l506_506344

-- Define conditions
def development_necessary_for_inheritance : Prop :=
  ∀ d i, d = development → i = inheritance → (d → i)

def inheritance_inevitable_for_development : Prop :=
  ∀ i d, i = inheritance → d = development → (i → d)

-- Define the main problem statement
theorem development_inheritance_false
  (h1 : development_necessary_for_inheritance)
  (h2 : inheritance_inevitable_for_development) :
  false :=
by
  sorry

end development_inheritance_false_l506_506344


namespace sufficient_but_not_necessary_condition_l506_506342

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∃ a, (1 + a) ^ 6 = 64) →
  (a = 1 → (1 + a) ^ 6 = 64) ∧ ¬(∀ a, ((1 + a) ^ 6 = 64 → a = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l506_506342


namespace set_size_and_differences_l506_506903

theorem set_size_and_differences (A : Finset ℕ) (n : ℕ) (hA : ∀ x y ∈ A, x ≠ y → |x - y| ∈ Finset.range n \ {0}) :
  A.card ≤ 2 * Nat.floor (Real.sqrt n) + 1 := 
sorry

end set_size_and_differences_l506_506903


namespace max_range_of_walk_min_range_of_walk_count_of_max_range_sequences_l506_506758

section RandomWalk

variables {a b : ℕ} (h : a > b)

def max_range_walk : ℕ := a
def min_range_walk : ℕ := a - b
def count_max_range_sequences : ℕ := b + 1

theorem max_range_of_walk (h : a > b) : max_range_walk h = a :=
by
  sorry

theorem min_range_of_walk (h : a > b) : min_range_walk h = a - b :=
by
  sorry

theorem count_of_max_range_sequences (h : a > b) : count_max_range_sequences h = b + 1 :=
by
  sorry

end RandomWalk

end max_range_of_walk_min_range_of_walk_count_of_max_range_sequences_l506_506758


namespace james_distance_ridden_l506_506601

theorem james_distance_ridden : 
  let speed := 16 
  let time := 5 
  speed * time = 80 := 
by
  sorry

end james_distance_ridden_l506_506601


namespace length_of_AB_eq_12_l506_506118

-- Define the parabola and its properties
def parabola_y_squared_eq_3x (x y : ℝ) := y^2 = 3 * x

-- Define the focus of the parabola
def focus_of_parabola := (3 / 4 : ℝ, 0 : ℝ)

-- Define the line passing through the focus with an inclination angle of 30 degrees
def line_through_focus (x : ℝ) := Real.sqrt(3) / 3 * (x - 3 / 4)

-- The main theorem to state the problem
theorem length_of_AB_eq_12 (A B : ℝ × ℝ) 
  (hA : parabola_y_squared_eq_3x A.1 A.2) 
  (hB : parabola_y_squared_eq_3x B.1 B.2) 
  (hLineA : A.2 = line_through_focus A.1) 
  (hLineB : B.2 = line_through_focus B.1) 
  (hNotEq : A.1 ≠ B.1) : 
  Real.abs (A.1 - B.1) + 3 / 2 = 12 :=
sorry

end length_of_AB_eq_12_l506_506118


namespace commute_days_l506_506306

theorem commute_days (a b c y : ℕ) 
  (h1 : a + c = 10)
  (h2 : a + b = 13)
  (h3 : b + c = 11) : 
  y = a + b + c → y = 17 := 
by
  intro hy
  have ha : a = 6 :=
    calc
      2 * a = 23 - 11 : by linarith [h1, h2, h3]
      a = 6 : by norm_num
  have hb : b = 7 := by linarith [h2, ha]
  have hc : c = 4 := by linarith [h1, ha]
  rw [ha, hb, hc] at hy
  exact hy

end commute_days_l506_506306


namespace moles_NaCl_formed_in_reaction_l506_506855

noncomputable def moles_of_NaCl_formed (moles_NaOH moles_HCl : ℕ) : ℕ :=
  if moles_NaOH = 1 ∧ moles_HCl = 1 then 1 else 0

theorem moles_NaCl_formed_in_reaction : moles_of_NaCl_formed 1 1 = 1 := 
by
  sorry

end moles_NaCl_formed_in_reaction_l506_506855


namespace count_n_with_conditions_l506_506539

-- Define the conditions and the theorem
theorem count_n_with_conditions : 
  let n_values := {n : ℕ | ∃ m k : ℕ, n = 4 * k ∧ m = 4 * k^2 - 1 ∧ m % 5 = 0 ∧ 0 < n ∧ n < 300} in
  n_values.finite ∧ n_values.card = 30 :=
by
  sorry

end count_n_with_conditions_l506_506539


namespace solve_tan_equation_l506_506370

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l506_506370


namespace find_x_l506_506378

theorem find_x (x : ℝ) (h : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) → x = 110 :=
by
  sorry

end find_x_l506_506378


namespace problem1_problem2_l506_506031

-- Define variables
variables {x y m : ℝ}
variables (h1 : x + y > 0) (h2 : xy ≠ 0)

-- Problem (1): Prove that x^3 + y^3 ≥ x^2 y + y^2 x
theorem problem1 (h1 : x + y > 0) (h2 : xy ≠ 0) : x^3 + y^3 ≥ x^2 * y + y^2 * x :=
sorry

-- Problem (2): Given the conditions, the range of m is [-6, 2]
theorem problem2 (h1 : x + y > 0) (h2 : xy ≠ 0) (h3 : (x / y^2) + (y / x^2) ≥ (m / 2) * ((1 / x) + (1 / y))) : m ∈ Set.Icc (-6 : ℝ) 2 :=
sorry

end problem1_problem2_l506_506031


namespace base_conversion_problem_l506_506679

theorem base_conversion_problem 
  (b x y z : ℕ)
  (h1 : 1987 = x * b^2 + y * b + z)
  (h2 : x + y + z = 25) :
  b = 19 ∧ x = 5 ∧ y = 9 ∧ z = 11 := 
by
  sorry

end base_conversion_problem_l506_506679


namespace sin_lambda_alpha_neg_one_l506_506512

theorem sin_lambda_alpha_neg_one
  (f : ℝ → ℝ)
  (λ α : ℝ)
  (h₀ : ∀ x, f x = if x ≥ 0 then x^2 + 2017*x + sin x else -x^2 + λ*x + cos (x + α))
  (h₁ : ∀ x, f (-x) = -f x)
  : sin (λ * α) = -1 := sorry

end sin_lambda_alpha_neg_one_l506_506512


namespace point_on_x_axis_point_on_y_axis_l506_506471

section
-- Definitions for the conditions
def point_A (a : ℝ) : ℝ × ℝ := (a - 3, a ^ 2 - 4)

-- Proof for point A lying on the x-axis
theorem point_on_x_axis (a : ℝ) (h : (point_A a).2 = 0) :
  point_A a = (-1, 0) ∨ point_A a = (-5, 0) :=
sorry

-- Proof for point A lying on the y-axis
theorem point_on_y_axis (a : ℝ) (h : (point_A a).1 = 0) :
  point_A a = (0, 5) :=
sorry
end

end point_on_x_axis_point_on_y_axis_l506_506471


namespace inscribed_equilateral_triangle_area_l506_506328

theorem inscribed_equilateral_triangle_area (r : ℝ) (h_r : r = 9) :
  let s := r * Real.sqrt 3 in
  let A := (Real.sqrt 3 / 4) * s^2 in
  A = 243 * (Real.sqrt 3 / 4) := by
  sorry

end inscribed_equilateral_triangle_area_l506_506328


namespace ratio_of_numbers_l506_506187

theorem ratio_of_numbers (A B : ℕ) (HCF_AB : Nat.gcd A B = 3) (LCM_AB : Nat.lcm A B = 36) : 
  A / B = 3 / 4 :=
sorry

end ratio_of_numbers_l506_506187


namespace total_weekly_harvest_l506_506565

-- Definitions
def sacks_per_section : ℕ := 75
def sections : ℕ := 15
def weekend_increase_rate : ℚ := 0.20
def weekday_days : ℕ := 5
def weekend_days : ℕ := 2

-- Translations of conditions
def weekday_harvest_per_day : ℕ := sacks_per_section * sections
def weekend_harvest_per_day : ℕ := (1 + weekend_increase_rate) * weekday_harvest_per_day
def total_weekday_harvest : ℕ := weekday_harvest_per_day * weekday_days
def total_weekend_harvest : ℕ := weekend_harvest_per_day * weekend_days

-- Statement to prove
theorem total_weekly_harvest : total_weekday_harvest + total_weekend_harvest = 8325 := by
  sorry

end total_weekly_harvest_l506_506565


namespace find_x_l506_506391

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l506_506391


namespace at_least_one_not_less_than_one_l506_506878

open Real

theorem at_least_one_not_less_than_one (x : ℝ) :
  let a := x^2 + 1/2
  let b := 2 - x
  let c := x^2 - x + 1
  a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1 :=
by
  -- Definitions of a, b, and c
  let a := x^2 + 1/2
  let b := 2 - x
  let c := x^2 - x + 1
  -- Proof is omitted
  sorry

end at_least_one_not_less_than_one_l506_506878


namespace find_lambda_l506_506023

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {e₁ e₂ a b : V} {λ : ℝ}

-- Define non-collinearity condition
def noncollinear (e₁ e₂ : V) : Prop := ∀ (m : ℝ), e₁ ≠ m • e₂

-- Given conditions
axiom noncollinear_e1_e2 : noncollinear e₁ e₂
axiom vector_a : a = 3 • e₁ - 2 • e₂
axiom vector_b : b = e₁ + λ • e₂
axiom a_parallel_b : ∃ (m : ℝ), a = m • b

-- Proof goal
theorem find_lambda : λ = -2 / 3 :=
sorry

end find_lambda_l506_506023


namespace octagon_side_length_l506_506069

theorem octagon_side_length 
  (num_sides : ℕ) 
  (perimeter : ℝ) 
  (h_sides : num_sides = 8) 
  (h_perimeter : perimeter = 23.6) :
  (perimeter / num_sides) = 2.95 :=
by
  have h_valid_sides : num_sides = 8 := h_sides
  have h_valid_perimeter : perimeter = 23.6 := h_perimeter
  sorry

end octagon_side_length_l506_506069


namespace square_of_ratio_is_specified_value_l506_506963

theorem square_of_ratio_is_specified_value (a b c : ℝ) (h1 : c = Real.sqrt (a^2 + b^2)) (h2 : a / b = b / c) :
  (a / b)^2 = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end square_of_ratio_is_specified_value_l506_506963


namespace measure_angle_BAO_is_15_degrees_l506_506976

theorem measure_angle_BAO_is_15_degrees
  {O A B C D E : Type}
  (semicircle : circle D O)
  (on_extension : collinear A C D)
  (on_semicircle : on_arc E semicircle)
  (B_intersection : B ∈ line_segment A E)
  (distinct_points : B ≠ E)
  (length_equality_1 : length A B = length O D)
  (measurement_60_degrees : measure_angle E O D = 60) :
  measure_angle B A O = 15 := 
  sorry

end measure_angle_BAO_is_15_degrees_l506_506976


namespace find_angle_between_vectors_l506_506888

open Real

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def angle_between_vectors 
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hmag : ∥b∥ = 4 * ∥a∥) 
  (horth : inner a (2 • a + b) = 0) : ℝ :=
2 * π / 3

theorem find_angle_between_vectors
  {a b : V} (ha : a ≠ 0) (hb : b ≠ 0)
  (hmag : ∥b∥ = 4 * ∥a∥)
  (horth : inner a (2 • a + b) = 0) :
   real.angle a b = 2 * π / 3 :=
sorry

end find_angle_between_vectors_l506_506888


namespace ad_eb_intersect_on_altitude_l506_506591

open EuclideanGeometry

variables {A B C D E F G K L C1 : Point}

-- Definitions for the problem
variables (triangleABC : Triangle A B C)
  (squareAEFC : Square A E F C)
  (squareBDGC : Square B D G C)
  (altitudeCC1 : Line C C1)
  (lineDA : Line A D)
  (lineEB : Line B E)

-- Definition of intersection
def intersects_on_altitude (pt : Point) : Prop :=
  pt ∈ lineDA ∧ pt ∈ lineEB ∧ pt ∈ altitudeCC1

-- The theorem to be proved
theorem ad_eb_intersect_on_altitude : 
  ∃ pt : Point, intersects_on_altitude lineDA lineEB altitudeCC1 pt := 
sorry

end ad_eb_intersect_on_altitude_l506_506591


namespace max_range_walk_min_range_walk_count_max_range_sequences_l506_506751

variable {a b : ℕ}

-- Condition: a > b
def valid_walk (a b : ℕ) : Prop := a > b

-- Proof that the maximum possible range of the walk is a
theorem max_range_walk (h : valid_walk a b) : 
  (a + b) = a + b := sorry

-- Proof that the minimum possible range of the walk is a - b
theorem min_range_walk (h : valid_walk a b) : 
  (a - b) = a - b := sorry

-- Proof that the number of different sequences with the maximum possible range is b + 1
theorem count_max_range_sequences (h : valid_walk a b) : 
  b + 1 = b + 1 := sorry

end max_range_walk_min_range_walk_count_max_range_sequences_l506_506751


namespace player5_points_combination_l506_506081

theorem player5_points_combination :
  ∃ (two_point_shots three_pointers free_throws : ℕ), 
  (two_point_shots * 2 + three_pointers * 3 + free_throws * 1 = 14) :=
sorry

end player5_points_combination_l506_506081


namespace remainder_when_divided_by_10_l506_506720

theorem remainder_when_divided_by_10 : 
  (2468 * 7391 * 90523) % 10 = 4 :=
by
  sorry

end remainder_when_divided_by_10_l506_506720


namespace distance_between_parallel_lines_l506_506672

-- Definitions of lines l_1 and l_2
def line_l1 (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def line_l2 (x y : ℝ) : Prop := 6*x + 8*y - 5 = 0

-- Proof statement that the distance between the two lines is 1/10
theorem distance_between_parallel_lines (x y : ℝ) :
  ∃ d : ℝ, d = 1/10 ∧ ∀ p : ℝ × ℝ,
  (line_l1 p.1 p.2 ∧ line_l2 p.1 p.2 → p = (x, y)) :=
sorry

end distance_between_parallel_lines_l506_506672


namespace geometric_series_first_term_l506_506204

theorem geometric_series_first_term (a r : ℝ)
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
by
  sorry

end geometric_series_first_term_l506_506204


namespace angle_AFS_eq_angle_ECD_l506_506992

open_locale classical
noncomputable theory

variables {A B C D E S F: Type}
  [is_cyclic_quad : cyclic_quad ABCD]
  [is_parallelogram : parallelogram ABDE]
  [is_intersection_S : intersection_point AC BD S]
  [is_ray_intersection_F : ray_intersection_point AB DC F]

theorem angle_AFS_eq_angle_ECD :
  ∠AFS = ∠ECD :=
sorry

end angle_AFS_eq_angle_ECD_l506_506992


namespace complex_number_in_first_quadrant_l506_506094

open Complex

theorem complex_number_in_first_quadrant (z : ℂ) (h : z = 1 / (1 - I)) : 
  z.re > 0 ∧ z.im > 0 :=
by
  sorry

end complex_number_in_first_quadrant_l506_506094


namespace distinct_arrangements_ballon_l506_506537

theorem distinct_arrangements_ballon : 
  let n := 6
  let repetitions := 2
  n! / repetitions! = 360 :=
by
  sorry

end distinct_arrangements_ballon_l506_506537


namespace partition_modulo_n_l506_506354

theorem partition_modulo_n (n : ℕ) :
  (∃ (A B : finset ℤ), A.nonempty ∧ B.nonempty ∧ (∀ m : ℤ, 
    (∃ a ∈ A, m ≡ a [MOD n]) ∨ 
    (∃ b ∈ B, m ≡ b [MOD n]) ∨ 
    (∃ a ∈ A, ∃ b ∈ B, m ≡ a + b [MOD n]))) ↔ ¬ ∃ (k : ℕ), n = 2^k :=
sorry

end partition_modulo_n_l506_506354


namespace distance_P_to_line_l_l506_506900

-- Defining the points A and P and direction vector
def A := (1, -1, -1) : ℝ × ℝ × ℝ
def P := (1, 1, 1) : ℝ × ℝ × ℝ
def direction_vector := (1, 0, -1) : ℝ × ℝ × ℝ

-- Function to compute the distance from point P to the line l
noncomputable def distance_from_point_to_line (A P : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ) : ℝ :=
  let AP := (P.1 - A.1, P.2 - A.2, P.3 - A.3) in
  let norm_AP := Math.sqrt (AP.1 * AP.1 + AP.2 * AP.2 + AP.3 * AP.3) in
  let dot_product := (AP.1 / norm_AP) * d.1 + (AP.2 / norm_AP) * d.2 + (AP.3 / norm_AP) * d.3 in
  Math.sqrt (norm_AP * norm_AP - dot_product * dot_product)

-- Proof statement
theorem distance_P_to_line_l : distance_from_point_to_line A P direction_vector = Math.sqrt 6 := by
  sorry

end distance_P_to_line_l_l506_506900


namespace car_speed_l506_506671

theorem car_speed
  (d : ℕ) (n_poles : ℕ) (time_minutes : ℕ) (distance_between_poles : ℕ := 50)
  (poles_counted : ℕ := 41) (time := 2) : (d = distance_between_poles * (poles_counted - 1))
  → (time_minutes = time)
  → (time_in_hours : ℚ := time_minutes / 60)
  → (car_speed := d / time_in_hours)
  → car_speed = 60000 :=
by
  intros h_distance h_time h_time_hours h_speed
  rw [h_distance, h_time, h_time_hours, h_speed]
  sorry

end car_speed_l506_506671


namespace tan_sin_cos_eq_l506_506402

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l506_506402


namespace find_d_l506_506780

namespace NineDigitNumber

variables {A B C D E F G : ℕ}

theorem find_d 
  (h1 : 6 + A + B = 13) 
  (h2 : A + B + C = 13)
  (h3 : B + C + D = 13)
  (h4 : C + D + E = 13)
  (h5 : D + E + F = 13)
  (h6 : E + F + G = 13)
  (h7 : F + G + 3 = 13) :
  D = 4 :=
sorry

end NineDigitNumber

end find_d_l506_506780


namespace triangle_perimeter_l506_506519

theorem triangle_perimeter (x : ℕ) (h1 : 3 < x) (h2 : x < 5) : (1 + x + 4) = 9 := 
by
  have hx : x = 4 := by 
    linarith
  rw [hx]
  norm_num

end triangle_perimeter_l506_506519


namespace ice_cream_scoops_total_l506_506322

noncomputable def scoops_of_ice_cream : ℕ :=
let single_cone : ℕ := 1 in
let double_cone : ℕ := single_cone * 2 in
let banana_split : ℕ := single_cone * 3 in
let waffle_bowl : ℕ := banana_split + 1 in
single_cone + double_cone + banana_split + waffle_bowl

theorem ice_cream_scoops_total : scoops_of_ice_cream = 10 :=
sorry

end ice_cream_scoops_total_l506_506322


namespace Xiaoqiang_scores_mode_median_l506_506736

-- Define the frequency distribution
def frequency_distribution : list (ℝ × ℕ) := [(11.8, 1), (11.9, 6), (12, 9), (12.1, 10), (12.2, 4)]

-- Define a function to compute the mode
def mode (dist : list (ℝ × ℕ)) : ℝ :=
  dist.foldr (λ (p1 p2 : ℝ × ℕ), if p1.2 > p2.2 then p1 else p2).1

-- Define a function to compute the median for an even-sized list of scores
def median (dist : list (ℝ × ℕ)) : ℝ :=
  let sorted_scores := (dist.foldr (λ (p acc : (ℝ × ℕ)) , acc ++ (list.repeat p.1 p.2)) []).sort
  let n := sorted_scores.length
  (sorted_scores.get (n / 2 - 1) + sorted_scores.get (n / 2)) / 2

-- Theorem statement to prove mode and median
theorem Xiaoqiang_scores_mode_median :
  mode frequency_distribution = 12.1 ∧ median frequency_distribution = 12 :=
by
  sorry

end Xiaoqiang_scores_mode_median_l506_506736


namespace intersection_point_of_y_eq_4x_minus_2_with_x_axis_l506_506668

theorem intersection_point_of_y_eq_4x_minus_2_with_x_axis :
  ∃ x, (4 * x - 2 = 0 ∧ (x, 0) = (1 / 2, 0)) :=
by
  sorry

end intersection_point_of_y_eq_4x_minus_2_with_x_axis_l506_506668


namespace range_of_slope_ellipse_chord_l506_506041

theorem range_of_slope_ellipse_chord :
  ∀ (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ),
    (x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2) →
    (x₁^2 + y₁^2 / 4 = 1 ∧ x₂^2 + y₂^2 / 4 = 1) →
    ((1 / 2) ≤ y₀ ∧ y₀ ≤ 1) →
    (-4 ≤ -2 / y₀ ∧ -2 / y₀ ≤ -2) :=
by
  sorry

end range_of_slope_ellipse_chord_l506_506041


namespace area_of_polygon_intersection_is_525_l506_506290

noncomputable def cube_edge_length : ℝ := 30
def P := (10, 0, 0)
def Q := (30, 0, 20)
def R := (30, 15, 30)
def A := (0, 0, 0)
def B := (30, 0, 0)
def C := (30, 0, 30)
def D := (30, 30, 30)

theorem area_of_polygon_intersection_is_525 :
  let plane_PQR := λ (x y z : ℝ), 3 * x + 2 * y - 3 * z = 30
  let points_of_intersection := [
    (0, 15, 0),  -- Intersection with x = 0
    (0, 30, 10), -- Intersection with y = 30
    (9, 30, 27), -- Intersection with z = 30
    (20, 0, 0),  -- Intersection with z = 0
    (30, 0, 20)  -- Intersection with x = 30
  ]
  let polygon_area := 525
  (area_of_polygon_formed_by_intersections plane_PQR points_of_intersection) = polygon_area := 
sorry

end area_of_polygon_intersection_is_525_l506_506290


namespace expression_min_value_l506_506473

theorem expression_min_value (a b c k : ℝ) (h1 : a < c) (h2 : c < b) (h3 : b = k * c) (h4 : k > 1) :
  (1 : ℝ) / c^2 * ((k * c - a)^2 + (a + c)^2 + (c - a)^2) ≥ k^2 / 3 + 2 :=
sorry

end expression_min_value_l506_506473


namespace total_amount_for_gifts_l506_506564

theorem total_amount_for_gifts (workers_per_block : ℕ) (worth_per_gift : ℕ) (number_of_blocks : ℕ)
  (h1 : workers_per_block = 100) (h2 : worth_per_gift = 4) (h3 : number_of_blocks = 10) :
  (workers_per_block * worth_per_gift * number_of_blocks = 4000) := by
  sorry

end total_amount_for_gifts_l506_506564


namespace smallest_w_factor_l506_506544

theorem smallest_w_factor (w : ℕ) (hw : w > 0) :
  (∃ w, 2^4 ∣ 1452 * w ∧ 3^3 ∣ 1452 * w ∧ 13^3 ∣ 1452 * w) ↔ w = 79092 :=
by sorry

end smallest_w_factor_l506_506544


namespace ellipse_proof_l506_506016

-- Definitions related to ellipse and geometric conditions
def ellipse_equation (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity (c a : ℝ) : Prop :=
  c / a = sqrt 2 / 2

def point_on_y_axis (m : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p.1 = 0 ∧ p.2 = m

def line (k x y : ℝ) : Prop :=
  y = k * x - 1

def circle_diameter_condition (A B : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  let d := (A.1 - B.1)^2 + (A.2 - B.2)^2
  in (M.1 - A.1)^2 + (M.2 - A.2)^2 + (M.1 - B.1)^2 + (M.2 - B.2)^2 = d

noncomputable def fixed_point_M_exists : Prop :=
  ∃ (M : ℝ × ℝ), M = (0, 3) ∧ ∀ (k : ℝ), 
    (∃ (x1 x2 : ℝ), ellipse_equation x1 (k*x1 - 1) 18 9 ∧ ellipse_equation x2 (k*x2 - 1) 18 9 ∧
      (circle_diameter_condition (x1, k * x1 - 1) (x2, k * x2 - 1) M))

theorem ellipse_proof :
  (ellipse_equation x y 18 9) →
  eccentricity 3 (sqrt 18) →
  fixed_point_M_exists :=
begin 
  intros,
  sorry -- Proof to be implemented
end

end ellipse_proof_l506_506016


namespace Ramesh_paid_l506_506652

theorem Ramesh_paid (P : ℝ) (h1 : 1.10 * P = 21725) : 0.80 * P + 125 + 250 = 16175 :=
by
  sorry

end Ramesh_paid_l506_506652


namespace solve_tan_equation_l506_506369

theorem solve_tan_equation (x : ℝ) (h1 : 0 < x ∧ x < 180) :
  (tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) ↔ x = 100 :=
by
  sorry

end solve_tan_equation_l506_506369


namespace find_d_l506_506618

noncomputable def polynomial_d (a b c d : ℤ) (p q r s : ℤ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧
  1 + a + b + c + d = 2024 ∧
  (1 + p) * (1 + q) * (1 + r) * (1 + s) = 2024 ∧
  d = p * q * r * s

theorem find_d (a b c d : ℤ) (h : polynomial_d a b c d 7 10 22 11) : d = 17020 :=
  sorry

end find_d_l506_506618


namespace affine_transformation_circle_self_mapping_l506_506159

open Affine

variables {P : Type*} [affine_space V P] [inner_product_space ℝ V] [second_countable_topology V] [complete_space V]

-- Definitions and conditions used
def affine_map.maps_circle_to_self (L : affine_map ℝ V P) (C : set P) : Prop :=
∀ p ∈ C, L p ∈ C

-- Hypotheses
variable (L : affine_map ℝ V P)
variables {C : set P} {p : P} (hp : p ∈ C)

-- Main theorem statement
theorem affine_transformation_circle_self_mapping (hL : affine_map.maps_circle_to_self L C) :
  (∃ (P : affine_map ℝ V P), (is_rotation L ∨ is_reflection L)) :=
sorry

end affine_transformation_circle_self_mapping_l506_506159


namespace work_done_at_4_pm_l506_506262

noncomputable def workCompletionTime (aHours : ℝ) (bHours : ℝ) (startTime : ℝ) : ℝ :=
  let aRate := 1 / aHours
  let bRate := 1 / bHours
  let cycleWork := aRate + bRate
  let cyclesNeeded := (1 : ℝ) / cycleWork
  startTime + 2 * cyclesNeeded

theorem work_done_at_4_pm :
  workCompletionTime 8 12 6 = 16 :=  -- 16 in 24-hour format is 4 pm
by 
  sorry

end work_done_at_4_pm_l506_506262


namespace largest_a_value_l506_506125

theorem largest_a_value (a b c : ℝ) (h1 : a + b + c = 7) (h2 : ab + ac + bc = 12) : 
  a ≤ (7 + Real.sqrt 46) / 3 :=
sorry

end largest_a_value_l506_506125


namespace find_b_l506_506356

-- Define the necessary conditions for our problem
def floor_fun (b : ℝ) : ℤ := ⌊b⌋

-- State the theorem: 
-- if b + floor_fun b = 14.3, then b = 7.3
theorem find_b (b : ℝ) (h : b + ↑(floor_fun b) = 14.3) : b = 7.3 := 
by 
  -- proof goes here
  sorry

end find_b_l506_506356


namespace clever_seven_year_count_l506_506551

def isCleverSevenYear (y : Nat) : Bool :=
  let d1 := y / 1000
  let d2 := (y % 1000) / 100
  let d3 := (y % 100) / 10
  let d4 := y % 10
  d1 + d2 + d3 + d4 = 7

theorem clever_seven_year_count : 
  ∃ n, n = 21 ∧ ∀ y, 2000 ≤ y ∧ y ≤ 2999 → isCleverSevenYear y = true ↔ n = 21 :=
by 
  sorry

end clever_seven_year_count_l506_506551


namespace ice_cream_scoops_total_l506_506321

noncomputable def scoops_of_ice_cream : ℕ :=
let single_cone : ℕ := 1 in
let double_cone : ℕ := single_cone * 2 in
let banana_split : ℕ := single_cone * 3 in
let waffle_bowl : ℕ := banana_split + 1 in
single_cone + double_cone + banana_split + waffle_bowl

theorem ice_cream_scoops_total : scoops_of_ice_cream = 10 :=
sorry

end ice_cream_scoops_total_l506_506321


namespace solution_inequality_l506_506939

noncomputable def solution_set (a : ℝ) (x : ℝ) := x < (1 - a) / (1 + a)

theorem solution_inequality 
  (a : ℝ) 
  (h1 : a^3 < a) 
  (h2 : a < a^2) :
  ∀ (x : ℝ), x + a > 1 - a * x ↔ solution_set a x :=
sorry

end solution_inequality_l506_506939


namespace sugar_already_put_in_l506_506638

-- Definitions based on conditions
def required_sugar : ℕ := 13
def additional_sugar_needed : ℕ := 11

-- Theorem to be proven
theorem sugar_already_put_in :
  required_sugar - additional_sugar_needed = 2 := by
  sorry

end sugar_already_put_in_l506_506638


namespace find_x_tan_eq_l506_506429

theorem find_x_tan_eq :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) ∧ x = 110 :=
by
  sorry

end find_x_tan_eq_l506_506429


namespace integer_value_of_fraction_l506_506064

theorem integer_value_of_fraction (m n p : ℕ) (hm_diff: m ≠ n) (hn_diff: n ≠ p) (hp_diff: m ≠ p) 
  (hm_range: 2 ≤ m ∧ m ≤ 9) (hn_range: 2 ≤ n ∧ n ≤ 9) (hp_range: 2 ≤ p ∧ p ≤ 9) :
  (m + n + p) / (m + n) = 2 :=
by
  sorry

end integer_value_of_fraction_l506_506064


namespace range_shifted_function_l506_506834

noncomputable def f : ℝ → ℝ := sorry

theorem range_shifted_function :
  (∀ y, ∃ x, y = f(x) → 1 ≤ y ∧ y ≤ 2) →
  (∀ z, ∃ x, z = f(x + 1) - 2 → -1 ≤ z ∧ z ≤ 0) := 
by
  sorry

end range_shifted_function_l506_506834


namespace make_one_ruble_l506_506089

theorem make_one_ruble :
  ∃ (num_ways : ℕ),
    (num_ways = 4562) ∧
    (num_ways = ∑ n_1 in finset.range (101), 
                 ∑ n_2 in finset.range (n_1/2 + 1), 
                 ∑ n_5 in finset.range (n_1/5 + 1), 
                 ∑ n_10 in finset.range (n_1/10 + 1), 
                 ∑ n_20 in finset.range (n_1/20 + 1), 
                 ∑ n_50 in finset.range (n_1/50 + 1), 
                 if n_1 + 2*n_2 + 5*n_5 + 10*n_10 + 20*n_20 + 50*n_50 = 100 then 1 else 0) := 
by sorry

end make_one_ruble_l506_506089


namespace pastries_made_correct_l506_506313

-- Definitions based on conditions
def cakes_made := 14
def cakes_sold := 97
def pastries_sold := 8
def cakes_more_than_pastries := 89

-- Definition of the function to compute pastries made
def pastries_made (cakes_made cakes_sold pastries_sold cakes_more_than_pastries : ℕ) : ℕ :=
  cakes_sold - cakes_more_than_pastries

-- The statement to prove
theorem pastries_made_correct : pastries_made cakes_made cakes_sold pastries_sold cakes_more_than_pastries = 8 := by
  unfold pastries_made
  norm_num
  sorry

end pastries_made_correct_l506_506313


namespace max_value_of_x_plus_3y_l506_506007

theorem max_value_of_x_plus_3y (x y : ℝ) (h : x^2 / 9 + y^2 = 1) : 
    ∃ θ : ℝ, x = 3 * Real.cos θ ∧ y = Real.sin θ ∧ (x + 3 * y) ≤ 3 * Real.sqrt 2 :=
by
  sorry

end max_value_of_x_plus_3y_l506_506007


namespace tan_sin_cos_eq_l506_506398

theorem tan_sin_cos_eq (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
  (h3 : tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x)) : x = 30 :=
sorry

end tan_sin_cos_eq_l506_506398


namespace geometric_sum_first_8_terms_eq_17_l506_506897

theorem geometric_sum_first_8_terms_eq_17 (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 2 * a n)
  (h2 : a 0 + a 1 + a 2 + a 3 = 1) : 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 17 :=
sorry

end geometric_sum_first_8_terms_eq_17_l506_506897


namespace range_of_sum_l506_506946

theorem range_of_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + b + 3 = a * b) : 
a + b ≥ 6 := 
sorry

end range_of_sum_l506_506946


namespace range_of_alpha_div_three_l506_506033

open Real

theorem range_of_alpha_div_three {k : ℤ} {α : ℝ} 
  (h1 : sin α > 0)
  (h2 : cos α < 0)
  (h3 : sin (α / 3) > cos (α / 3)) :
  (2 * k * π + π / 4 < α / 3 ∧ α / 3 < 2 * k * π + π / 3) 
  ∨ (2 * k * π + 5 * π / 6 < α / 3 ∧ α / 3 < 2 * k * π + π) :=
sorry

end range_of_alpha_div_three_l506_506033


namespace probability_of_divisibility_by_37_l506_506153

/--
The problem statement is to find the probability that a number formed by the digits 1 through 9 
in any order is divisible by 37. 
Given the conditions:
1. The digits used are 1 through 9.
2. There are 9! (362880) possible permutations of these digits.
The conclusion is that the probability of such a number being divisible by 37 is 1/40.
--/
def probability_divisible_by_37 : ℚ :=
  1 / 40

theorem probability_of_divisibility_by_37 :
  let n := 9!
  let favorable := 9072
  favorable / n = probability_divisible_by_37 := by
  sorry

end probability_of_divisibility_by_37_l506_506153


namespace rearrange_chips_l506_506957

theorem rearrange_chips (n : ℕ) (chips : Fin 3 → Fin n → Fin 3) :
  (∀ i : Fin n, ∃ j1 j2 j3 : Fin 3, i ≠ j1 ∧ i ≠ j2 ∧ i ≠ j3 ∧
   chips 0 i = j1 ∧ chips 1 i = j2 ∧ chips 2 i = j3) → 
  ∃ new_chips : Fin 3 → Fin n → Fin 3,
    (∀ i : Fin n, ∃ j1 j2 j3 : Fin 3, i ≠ j1 ∧ i ≠ j2 ∧ i ≠ j3 ∧
    new_chips 0 i = j1 ∧ new_chips 1 i = j2 ∧ new_chips 2 i = j3) ∧
    (∀ i j : Fin 3, j ≠ i → ∀ k : Fin n, new_chips i k ≠ new_chips j k) :=
sorry

end rearrange_chips_l506_506957


namespace pentagon_area_l506_506294

theorem pentagon_area (a b c d e r s : ℕ) 
  (hSides : {a, b, c, d, e} = {14, 21, 22, 28, 35})
  (hPythagorean : r^2 + s^2 = e^2)
  (hr : r = b - d)
  (hs : s = c - a) :
  b * c - (1 / 2) * r * s = 931 := sorry

end pentagon_area_l506_506294


namespace probability_log2_l506_506917

noncomputable def M := { x : ℝ | -1 < x ∧ x < 5 }

theorem probability_log2 (x0 : ℝ) (h : x0 ∈ M) : 
  let interval_length (a b : ℝ) := b - a in
  let probability := (interval_length (-1) 1) / (interval_length (-1) 5) in
  probability = 1 / 3 := by
  sorry

end probability_log2_l506_506917


namespace arithmetic_mean_of_two_digit_multiples_of_9_l506_506711

theorem arithmetic_mean_of_two_digit_multiples_of_9 : 
  let a1 := 18 in
  let an := 99 in
  let d := 9 in
  let n := (an - a1) / d + 1 in
  let S := n * (a1 + an) / 2 in
  (S / n : ℝ) = 58.5 :=
by
  let a1 := 18
  let an := 99
  let d := 9
  let n := (an - a1) / d + 1
  let S := n * (a1 + an) / 2
  show (S / n : ℝ) = 58.5
  sorry

end arithmetic_mean_of_two_digit_multiples_of_9_l506_506711


namespace slope_of_AB_angle_of_inclination_of_AB_l506_506492

-- Definitions of points A and B
def A : (ℝ × ℝ) := (4, -2)
def B : (ℝ × ℝ) := (1, 1)

-- Slope calculation definition
def slope (P Q : (ℝ × ℝ)) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- Angle of inclination calculation definition
def angle_of_inclination (k : ℝ) (h : 0 ≤ α ∧ α < π) : ℝ := if k < 0 then π + arctan k else arctan k

theorem slope_of_AB : slope A B = -1 := by
  sorry

theorem angle_of_inclination_of_AB (h : 0 ≤ (3 * π) / 4 ∧ (3 * π) / 4 < π) : angle_of_inclination (-1) h = (3 * π) / 4 := by
  sorry

end slope_of_AB_angle_of_inclination_of_AB_l506_506492


namespace find_number_l506_506741

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 90) : x = 4000 :=
by
  sorry

end find_number_l506_506741


namespace distinct_nonzero_reals_satisfy_equation_l506_506123

open Real

theorem distinct_nonzero_reals_satisfy_equation
  (a b c : ℝ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a) (h₄ : a ≠ 0) (h₅ : b ≠ 0) (h₆ : c ≠ 0)
  (h₇ : a + 2 / b = b + 2 / c) (h₈ : b + 2 / c = c + 2 / a) :
  (a + 2 / b) ^ 2 + (b + 2 / c) ^ 2 + (c + 2 / a) ^ 2 = 6 :=
sorry

end distinct_nonzero_reals_satisfy_equation_l506_506123


namespace find_pastries_made_l506_506314

variable (cakes_made pastries_made total_cakes_sold total_pastries_sold extra_cakes more_cakes_than_pastries : ℕ)

def baker_conditions := (cakes_made = 157) ∧ 
                        (total_cakes_sold = 158) ∧ 
                        (total_pastries_sold = 147) ∧ 
                        (more_cakes_than_pastries = 11) ∧ 
                        (extra_cakes = total_cakes_sold - cakes_made) ∧ 
                        (pastries_made = cakes_made - more_cakes_than_pastries)

theorem find_pastries_made : 
  baker_conditions cakes_made pastries_made total_cakes_sold total_pastries_sold extra_cakes more_cakes_than_pastries → 
  pastries_made = 146 :=
by
  sorry

end find_pastries_made_l506_506314


namespace bags_count_l506_506231

def totalWeight : ℝ := 45.0
def weightPerBag : ℝ := 23.0
def numBags := totalWeight / weightPerBag

theorem bags_count : ⌊numBags⌋₊ = 1 :=
by
  sorry

end bags_count_l506_506231
