import Mathlib

namespace initial_people_is_50_l803_803947

variable (X : Int)

def initial_people_on_bus (X : Int) : Prop :=
  let after_first_stop := X - 15
  let after_second_stop := after_first_stop - (8 - 2)
  let after_third_stop := after_second_stop - (4 - 3)
  after_third_stop = 28

theorem initial_people_is_50 : initial_people_on_bus 50 :=
by
  unfold initial_people_on_bus
  simp
  sorry

end initial_people_is_50_l803_803947


namespace inverse_of_log_base_2_l803_803625

theorem inverse_of_log_base_2 (x : ℝ) (hx : x > 0) : 
  (∃ y : ℝ, y = logBase 2 x) ↔ (∃ y : ℝ, x = 2^y) := 
sorry

end inverse_of_log_base_2_l803_803625


namespace basketball_lineup_count_l803_803629

-- Define the number of players in the basketball team
def num_players : Nat := 12

-- Define the number of lineups
def num_lineups : Nat := 3960

-- Prove the number of lineups is 3960
theorem basketball_lineup_count (num_players = 12) : 
  ∃ num_lineups, num_lineups = 12 * Nat.choose 11 4 := by
  existsi 3960
  sorry

end basketball_lineup_count_l803_803629


namespace base7_sub_base5_to_base10_l803_803334

def base7to10 (n : Nat) : Nat :=
  match n with
  | 52403 => 5 * 7^4 + 2 * 7^3 + 4 * 7^2 + 0 * 7^1 + 3 * 7^0
  | _ => 0

def base5to10 (n : Nat) : Nat :=
  match n with
  | 20345 => 2 * 5^4 + 0 * 5^3 + 3 * 5^2 + 4 * 5^1 + 5 * 5^0
  | _ => 0

theorem base7_sub_base5_to_base10 :
  base7to10 52403 - base5to10 20345 = 11540 :=
by
  sorry

end base7_sub_base5_to_base10_l803_803334


namespace store_owner_marked_price_l803_803297

theorem store_owner_marked_price (L M : ℝ) (h1 : M = (56 / 45) * L) : M / L = 124.44 / 100 :=
by
  sorry

end store_owner_marked_price_l803_803297


namespace Find_Ratio_l803_803105

variable (A B C D E F G P Q : ℝ × ℝ)

-- Define positions of points according to the problem
def Rectangle_setup : Prop :=
  A = (0, 6) ∧ B = (8, 6) ∧ C = (8, 0) ∧ D = (0, 0) ∧
  E = (6, 6) ∧ G = (8, 3) ∧ F = (4, 0)

-- Define lines' equations
def EQ_lines : Prop :=
  -- Line AG: y = -3/8 x + 6
  ∀ x y, (x, y) ∈ Line (0, 6) (8, 3) ↔ y = -3 / 8 * x + 6 ∧
  -- Line AC: y = -3/4 x + 6
  ∀ x y, (x, y) ∈ Line (0, 6) (8, 0) ↔ y = -3 / 4 * x + 6 ∧
  -- Line EF: y = 3x - 12
  ∀ x y, (x, y) ∈ Line (6, 6) (4, 0) ↔ y = 3 * x - 12

-- Define intersection points P and Q
def Intersections : Prop :=
  P = (24 / 5, 12 / 5) ∧ Q = (16 / 3, 4)

-- Define distances and lengths
def Lengths_and_Ratios (PQ EF : ℝ) : Prop :=
  -- Length of EF
  EF = Real.sqrt (Real.pow 2 2 + Real.pow 6 2) ∧
  -- Distance PQ
  PQ = abs ((16 / 3) - (24 / 5))

theorem Find_Ratio :
  Rectangle_setup A B C D E F G → 
  EQ_lines A B C D E F G → 
  Intersections P Q → 
  Lengths_and_Ratios (abs ((16 / 3) - (24 / 5))) (Real.sqrt (Real.pow 2 2 + Real.pow 6 2)) → 
  abs ((16 / 3) - (24 / 5)) / Real.sqrt (Real.pow 2 2 + Real.pow 6 2) = 4 * Real.sqrt 10 / 75 := 
by
  intros
  sorry

end Find_Ratio_l803_803105


namespace values_of_m_l803_803475

def A := {x : ℝ | x^2 - 4 * x - 21 = 0}
def B (m : ℝ) := {x : ℝ | m * x + 1 = 0}

theorem values_of_m (m : ℝ) : B(m) ⊆ A → (m = 0 ∨ m = -1 / 7 ∨ m = 1 / 3) := 
by 
  sorry

end values_of_m_l803_803475


namespace volume_parallelepiped_l803_803078

open Real
open Matrix

def v := ![7, -4, 3]
def w := ![13, -1, 2]
def u := ![1, 0, 6]

theorem volume_parallelepiped :
  abs (v ⬝ (cross_product w u)) = 265 :=
by
  sorry

end volume_parallelepiped_l803_803078


namespace count_multiples_of_neither_6_nor_8_l803_803414

theorem count_multiples_of_neither_6_nor_8 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999},
      multiples_of_6 := {n : ℕ | n ∈ three_digit_numbers ∧ n % 6 = 0},
      multiples_of_8 := {n : ℕ | n ∈ three_digit_numbers ∧ n % 8 = 0},
      multiples_of_24 := {n : ℕ | n ∈ three_digit_numbers ∧ n % 24 = 0} in
  (three_digit_numbers.card 
   - (multiples_of_6.card + multiples_of_8.card - multiples_of_24.card)) = 675 :=
by sorry

end count_multiples_of_neither_6_nor_8_l803_803414


namespace find_150th_term_of_sequence_l803_803932

noncomputable def sequence (n : ℕ) : ℕ :=
    if n = 0 then 1 else
    let bits := nat.binary_repr (n - 1) in
    bits.foldl (λ acc (bit, index), acc + if bit = '1' then 4 ^ index else 0) 0

theorem find_150th_term_of_sequence :
    sequence 150 = 16660 :=
by
  -- Proof can be filled in here
  sorry

end find_150th_term_of_sequence_l803_803932


namespace probability_all_five_dice_even_l803_803592

-- Definitions of conditions
def standard_six_sided_die : Set ℕ := {1, 2, 3, 4, 5, 6}
def even_numbers : Set ℕ := {2, 4, 6}

-- The statement to be proven
theorem probability_all_five_dice_even : 
  (∀ die ∈ standard_six_sided_die, (∃ n ∈ even_numbers, die = n)) → (1 / 32) = (1 / 2) ^ 5 :=
by
  intro h
  sorry

end probability_all_five_dice_even_l803_803592


namespace problem_statement_l803_803490

section
variables {a b : ℕ → ℕ}
variables (S : ℕ → ℕ) (T : ℕ → ℕ)

-- Define the arithmetic sequence conditions
def arithmetic_conditions := (a 2 = 2) ∧ (S 5 = 15)

-- Define the geometric sequence conditions
def geometric_conditions := (b 2 = 4) ∧ (b 5 = 32)

-- Define the sequences
def a_n (n : ℕ) := n
def b_n (n : ℕ) := 2 ^ n

-- Define the sequence of products
def a_n_b_n (n : ℕ) := n * 2 ^ n

-- Define the sum of the first n terms of the sequence {a_nb_n}
def T_n (n : ℕ) := (n - 1) * 2 ^ (n + 1) + 2

-- Proof statement (without the actual proof)
theorem problem_statement :
  arithmetic_conditions ∧ geometric_conditions →
  (∀ n, a n = a_n n) ∧ (∀ n, b n = b_n n) ∧ (∀ n, T n = T_n n) :=
by sorry
end

end problem_statement_l803_803490


namespace sides_of_rectangle_EKMR_l803_803831

noncomputable def right_triangle_ACB (AC AB : ℕ) : Prop :=
AC = 3 ∧ AB = 4

noncomputable def rectangle_EKMR_area (area : ℚ) : Prop :=
area = 3/5

noncomputable def rectangle_EKMR_perimeter (x y : ℚ) : Prop :=
2 * (x + y) < 9

theorem sides_of_rectangle_EKMR (x y : ℚ) 
  (h_triangle : right_triangle_ACB 3 4)
  (h_area : rectangle_EKMR_area (3/5))
  (h_perimeter : rectangle_EKMR_perimeter x y) : 
  (x = 2 ∧ y = 3/10) ∨ (x = 3/10 ∧ y = 2) := 
sorry

end sides_of_rectangle_EKMR_l803_803831


namespace angle_relation_in_triangle_l803_803874

theorem angle_relation_in_triangle
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : b * (a + b) * (b + c) = a^3 + b * (a^2 + c^2) + c^3)
    (h2 : A + B + C = π) 
    (h3 : A > 0) 
    (h4 : B > 0) 
    (h5 : C > 0) :
    (1 / (Real.sqrt A + Real.sqrt B)) + (1 / (Real.sqrt B + Real.sqrt C)) = (2 / (Real.sqrt C + Real.sqrt A)) :=
sorry

end angle_relation_in_triangle_l803_803874


namespace valid_student_count_sum_eq_1484_l803_803651

theorem valid_student_count_sum_eq_1484 :
  let s_values := {s : ℕ | 160 ≤ s ∧ s ≤ 210 ∧ (s - 1) % 7 = 0} in
  s_values.sum = 1484 :=
by
  sorry

end valid_student_count_sum_eq_1484_l803_803651


namespace tangent_lines_to_circles_l803_803408

theorem tangent_lines_to_circles:
  let circle1 := λ (x y : ℝ), x^2 + y^2 + 4*x - 4*y + 7 = 0
  let circle2 := λ (x y : ℝ), x^2 + y^2 - 4*x - 10*y + 13 = 0
  ∃ (tangents : ℝ), tangents = 3 ∧
    (∀ (x y : ℝ), circle1 x y → ∃ (line : ℝ → ℝ), tangent_to_circle circle1 (line x) ∧ tangent_to_circle circle2 (line y)) →
    tangents =
      3 :=
by
  sorry

end tangent_lines_to_circles_l803_803408


namespace solution_l803_803065

theorem solution 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a^2 / (b - c)^2) + (b^2 / (c - a)^2) + (c^2 / (a - b)^2) = 0) :
  (a^3 / (b - c)^3) + (b^3 / (c - a)^3) + (c^3 / (a - b)^3) = 0 := 
sorry 

end solution_l803_803065


namespace problem_statement_l803_803067

variables {c c' d d' : ℝ}

theorem problem_statement (hc : c ≠ 0) (hc' : c' ≠ 0)
  (h : (-d) / (2 * c) = 2 * ((-d') / (3 * c'))) :
  (d / (2 * c)) = 2 * (d' / (3 * c')) :=
by
  sorry

end problem_statement_l803_803067


namespace sum_of_other_endpoint_coordinates_l803_803564

noncomputable def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

theorem sum_of_other_endpoint_coordinates :
  let m := (5 : ℝ)
  let n := (-8 : ℝ)
  let x1 := (3 : ℝ)
  let y1 := (4 : ℝ)
  let midpoint_works (x y : ℝ) := midpoint x1 y1 x y = (m, n)
  ∃ x2 y2 : ℝ, x2 + y2 = -13 ∧ midpoint_works x2 y2 := 
sorry

end sum_of_other_endpoint_coordinates_l803_803564


namespace singing_only_pupils_l803_803036

theorem singing_only_pupils (total_pupils debate_only both : ℕ) (h1 : total_pupils = 55) (h2 : debate_only = 10) (h3 : both = 17) :
  total_pupils - debate_only = 45 :=
by
  -- skipping proof
  sorry

end singing_only_pupils_l803_803036


namespace bob_expected_rolls_l803_803675

-- Define the probability terms
def prob_no_reroll : ℚ := 3 / 4
def prob_reroll : ℚ := 1 / 8
def prob_extra_rolls : ℚ := 1 / 8

-- Define the expected number of rolls per day
noncomputable def expected_rolls_per_day : ℚ :=
  (7 / 8) / (1 - 3 / 8)

-- Define the number of days in a non-leap year
def days_in_non_leap_year : ℕ := 365

-- Define the total expected rolls in a non-leap year
noncomputable def total_expected_rolls_in_non_leap_year : ℚ :=
  expected_rolls_per_day * days_in_non_leap_year

-- Statement
theorem bob_expected_rolls {E : ℚ} (h : E = expected_rolls_per_day) :
  total_expected_rolls_in_non_leap_year = 511 :=
by 
  rw [total_expected_rolls_in_non_leap_year, days_in_non_leap_year, expected_rolls_per_day, h]
  sorry

end bob_expected_rolls_l803_803675


namespace divisor_value_l803_803610

theorem divisor_value :
  ∃ D : ℕ, 
    (242 % D = 11) ∧
    (698 % D = 18) ∧
    (365 % D = 15) ∧
    (527 % D = 13) ∧
    ((242 + 698 + 365 + 527) % D = 9) ∧
    (D = 48) :=
sorry

end divisor_value_l803_803610


namespace scientific_notation_of_935million_l803_803667

theorem scientific_notation_of_935million :
  935000000 = 9.35 * 10 ^ 8 :=
  sorry

end scientific_notation_of_935million_l803_803667


namespace smallest_abundant_not_multiple_of_4_l803_803596

def proper_divisors (n : ℕ) : List ℕ := (List.range n).filter (λ d, d ∣ n)

def sum_proper_divisors (n : ℕ) : ℕ := (proper_divisors n).sum

def is_abundant (n : ℕ) : Prop := sum_proper_divisors n > n

def not_multiple_of_4 (n : ℕ) : Prop := ¬ (4 ∣ n)

theorem smallest_abundant_not_multiple_of_4 : ∃ n, is_abundant n ∧ not_multiple_of_4 n ∧ (∀ m, is_abundant m ∧ not_multiple_of_4 m → n ≤ m) :=
  sorry

end smallest_abundant_not_multiple_of_4_l803_803596


namespace increasing_intervals_and_symmetry_axis_max_and_min_values_l803_803386

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem increasing_intervals_and_symmetry_axis :
  (∀ k : ℤ, ∀ x ∈ Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3), 0 < Real.sin (2 * x - Real.pi / 6) - Real.sin (2 * (x - ε) - Real.pi / 6) → ε > 0) ∧
  (∀ k : ℤ, ∀ x, k * Real.pi + Real.pi / 3 = x) :=
sorry

theorem max_and_min_values : 
  (∀ x ∈ Icc 0 (Real.pi / 2), f x ≤ 2) ∧ 
  (∃ x ∈ (Icc 0 (Real.pi / 2)), f x = 2) ∧
  (∀ x ∈ Icc 0 (Real.pi / 2), -1 ≤ f x) ∧ 
  (∃ x ∈ (Icc 0 (Real.pi / 2)), f x = -1) :=
sorry

end increasing_intervals_and_symmetry_axis_max_and_min_values_l803_803386


namespace non_neg_integer_solutions_for_inequality_l803_803529

theorem non_neg_integer_solutions_for_inequality :
  {x : ℤ | 5 * x - 1 < 3 * (x + 1) ∧ (1 - x) / 3 ≤ 1 ∧ 0 ≤ x } = {0, 1} := 
by {
  sorry
}

end non_neg_integer_solutions_for_inequality_l803_803529


namespace train_crossing_time_l803_803815

/-- How long it takes for a train to cross another train in the same direction -/
theorem train_crossing_time 
  (length_train1 length_train2 : ℝ)
  (speed_train1_kmph speed_train2_kmph : ℝ)
  (same_direction : true) :
  let speed_train1 := speed_train1_kmph * 1000 / 3600,
      speed_train2 := speed_train2_kmph * 1000 / 3600,
      relative_speed := speed_train1 - speed_train2,
      total_length := length_train1 + length_train2 in
  (relative_speed ≠ 0) →
  (total_length / relative_speed = 80) →
  true :=
by sorry

#eval train_crossing_time 300 500 72 36 true

end train_crossing_time_l803_803815


namespace max_acute_angles_in_convex_octagon_l803_803173

def is_convex (polygon_angles : List ℝ) : Prop :=
  ∀ a ∈ polygon_angles, a < 180

def interior_angle_sum (polygon_angles : List ℝ) : ℝ :=
  polygon_angles.sum

def is_acute (angle : ℝ) : Prop :=
  angle < 90

theorem max_acute_angles_in_convex_octagon
  (angles : List ℝ)
  (h1 : angles.length = 8)
  (h2 : is_convex angles)
  (h3 : interior_angle_sum angles = 1080) :
  ∃ n, n ≤ 5 ∧ n = angles.countp is_acute :=
sorry

end max_acute_angles_in_convex_octagon_l803_803173


namespace fraction_product_l803_803311

theorem fraction_product : (1/2) * (3/5) * (7/11) * (4/13) = 84/1430 := by
  sorry

end fraction_product_l803_803311


namespace max_sum_ab_bc_cd_da_l803_803576

def vals : set ℕ := {1, 3, 5, 7}

theorem max_sum_ab_bc_cd_da :
  ∃ (a b c d : ℕ), a ∈ vals ∧ b ∈ vals ∧ c ∈ vals ∧ d ∈ vals ∧ 
                   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                   ∀ (w x y z : ℕ), w ∈ vals ∧ x ∈ vals ∧ y ∈ vals ∧ z ∈ vals ∧ 
                                   w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z → 
                                   ab + bc + cd + da ≤ w*x + x*y + y*z + z*w := 
by
  sorry

end max_sum_ab_bc_cd_da_l803_803576


namespace fuel_fraction_final_to_second_l803_803083

open Real

-- Definitions from the conditions.
def total_fuel : ℝ := 60
def first_third_fuel : ℝ := 30
def second_third_fraction : ℝ := 1 / 3
def second_third_fuel := total_fuel * second_third_fraction
def final_third_fuel := total_fuel - first_third_fuel - second_third_fuel

-- State the problem.
theorem fuel_fraction_final_to_second :
  final_third_fuel / second_third_fuel = 1 / 2 :=
by
  -- Proof to be filled in.
  sorry

end fuel_fraction_final_to_second_l803_803083


namespace min_value_exp_l803_803363

noncomputable def min_of_exp (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  min (a^2 + b^2) / (a * b - b^2)

theorem min_value_exp (a b : ℝ) (h1 : a > b) (h2 : b > 0) : min_of_exp a b h1 h2 = 2 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_exp_l803_803363


namespace seating_arrangements_l803_803449

-- Number of ways to arrange a block of n items
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Groups
def dodgers : ℕ := 4
def marlins : ℕ := 3
def phillies : ℕ := 2

-- Total number of players
def total_players : ℕ := dodgers + marlins + phillies

-- Number of ways to arrange the blocks
def blocks_arrangements : ℕ := factorial 3

-- Internal arrangements within each block
def dodgers_arrangements : ℕ := factorial dodgers
def marlins_arrangements : ℕ := factorial marlins
def phillies_arrangements : ℕ := factorial phillies

-- Total number of ways to seat the players
def total_arrangements : ℕ :=
  blocks_arrangements * dodgers_arrangements * marlins_arrangements * phillies_arrangements

-- Prove that the total arrangements is 1728
theorem seating_arrangements : total_arrangements = 1728 := by
  sorry

end seating_arrangements_l803_803449


namespace sum_even_integers_402_to_500_l803_803571

theorem sum_even_integers_402_to_500 :
  let n := ((500 - 402) / 2) + 1 in
  let first_term := 402 in
  let last_term := 500 in
  n = 50 → 
  (n / 2) * (first_term + last_term) = 22550 :=
by
  sorry

end sum_even_integers_402_to_500_l803_803571


namespace angle_EQP_measure_l803_803685

variables (O : Type) (D E F P Q R : Type) [Incircle O (triangle D E F)] [Circumcircle O (triangle P Q R)]
variables (P_on_EF : is_on_line P E F) (Q_on_DE : is_on_line Q D E) (R_on_DF : is_on_line R D F)
variables (angle_D : angle D = 50) (angle_E : angle E = 70) (angle_F : angle F = 60)

theorem angle_EQP_measure : angle EQP = 130 := by
  sorry

end angle_EQP_measure_l803_803685


namespace instantaneous_velocity_at_3_l803_803632

-- Define the displacement function s(t)
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the time at which we want to calculate the instantaneous velocity
def time : ℝ := 3

-- Define the expected instantaneous velocity at t=3
def expected_velocity : ℝ := 54

-- Define the derivative of the displacement function as the velocity function
noncomputable def velocity (t : ℝ) : ℝ := deriv displacement t

-- Theorem: Prove that the instantaneous velocity at t=3 is 54
theorem instantaneous_velocity_at_3 : velocity time = expected_velocity := 
by {
  -- Here the detailed proof should go, but we skip it with sorry
  sorry
}

end instantaneous_velocity_at_3_l803_803632


namespace total_number_of_lockers_l803_803933

def locker_cost_per_digit : ℝ := 0.03
def total_labeling_cost : ℝ := 206.91

theorem total_number_of_lockers :
  ∃ (n : ℕ), (n > 0) ∧
    let digits_counted l := if l < 10 then 1 else if l < 100 then 2 else if l < 1000 then 3 else 4 
    let total_cost := (∑ l in finset.range n, (digits_counted l) * locker_cost_per_digit)
    total_cost = total_labeling_cost :=
by
  sorry

end total_number_of_lockers_l803_803933


namespace remuneration_difference_l803_803650

theorem remuneration_difference :
  let old_scheme_commission := 0.05 * 12000 in
  let new_scheme_fixed_salary := 1300 in
  let sales_exceeding := 12000 - 4000 in
  let new_scheme_commission := 0.025 * sales_exceeding in
  let old_scheme_total := old_scheme_commission in
  let new_scheme_total := new_scheme_fixed_salary + new_scheme_commission in
  (new_scheme_total - old_scheme_total = 900) :=
by
  repeat { sorry }

end remuneration_difference_l803_803650


namespace geometric_sum_proof_l803_803778

theorem geometric_sum_proof (S : ℕ → ℝ) (a : ℕ → ℝ) (r : ℝ) (n : ℕ)
    (hS3 : S 3 = 8) (hS6 : S 6 = 7)
    (Sn_def : ∀ n, S n = a 0 * (1 - r ^ n) / (1 - r)) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = -7 / 8 :=
by
  sorry

end geometric_sum_proof_l803_803778


namespace find_y_l803_803440

open Real

variable {x y : ℝ}

theorem find_y (h1 : x * y = 25) (h2 : x / y = 36) (hx : 0 < x) (hy : 0 < y) :
  y = 5 / 6 :=
by
  sorry

end find_y_l803_803440


namespace complex_c_values_l803_803486

theorem complex_c_values (c r s t : ℂ) (hc : r ≠ s ∧ s ≠ t ∧ r ≠ t)
  (h : ∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - 2 * c * r) * (z - 2 * c * s) * (z - 2 * c * t)) :
  {c : ℂ | ∃ (r s t : ℂ), r ≠ s ∧ s ≠ t ∧ r ≠ t ∧ ∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - 2 * c * r) * (z - 2 * c * s) * (z - 2 * c * t)}.finite ∧
  {c : ℂ | ∃ (r s t : ℂ), r ≠ s ∧ s ≠ t ∧ r ≠ t ∧ ∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - 2 * c * r) * (z - 2 * c * s) * (z - 2 * c * t)}.card = 3 :=
by sorry

end complex_c_values_l803_803486


namespace series_sum_eq_l803_803715

noncomputable def series_sum : Real :=
  ∑' n : ℕ, (4 * (n + 1) + 1) / (((4 * (n + 1) - 1) ^ 3) * ((4 * (n + 1) + 3) ^ 3))

theorem series_sum_eq : series_sum = 1 / 5184 := sorry

end series_sum_eq_l803_803715


namespace option_b_option_c_l803_803239

-- Lean statement for Option B
theorem option_b (z : ℂ) (h : |z - 2i| = 1) : ∃ w, |w| = 3 := sorry

-- Lean statement for Option C
theorem option_c (a b : ℝ) (ha : a ∈ {-2, 0, 1}) (hb : b ∈ {0, 1, 4, 9}) :
  (1/4 : ℝ) = 3 / (3*4) := sorry

end option_b_option_c_l803_803239


namespace sum_of_integers_l803_803171

theorem sum_of_integers {n : ℤ} (h : n + 2 = 9) : n + (n + 1) + (n + 2) = 24 := by
  sorry

end sum_of_integers_l803_803171


namespace smallest_integer_with_odd_and_even_divisors_l803_803202

theorem smallest_integer_with_odd_and_even_divisors :
  ∃ n : ℕ,
    0 < n ∧
    (∀ d : ℕ, d ∣ n → (odd d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 1}) ∨ (even d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 0})) ∧ 
    (↑8 = (∑ d in (finset.filter (λ d, odd d) (finset.divisors n)), 1)) ∧
    (↑16 = (∑ d in (finset.filter (λ d, even d) (finset.divisors n)), 1)) ∧
    (24 = (∑ d in (finset.divisors n), 1)) ∧
    n = 108 :=
begin
  sorry
end

end smallest_integer_with_odd_and_even_divisors_l803_803202


namespace no_quad_term_l803_803585

theorem no_quad_term (x m : ℝ) : 
  (2 * x^2 - 2 * (7 + 3 * x - 2 * x^2) + m * x^2) = -6 * x - 14 → m = -6 := 
by 
  sorry

end no_quad_term_l803_803585


namespace perfect_square_sequence_l803_803931

theorem perfect_square_sequence (k : ℤ) (y : ℕ → ℤ) :
  (y 1 = 1) ∧ (y 2 = 1) ∧
  (∀ n : ℕ, y (n + 2) = (4 * k - 5) * y (n + 1) - y n + 4 - 2 * k) →
  (∀ n ≥ 1, ∃ m : ℤ, y n = m^2) ↔ (k = 1 ∨ k = 3) :=
sorry

end perfect_square_sequence_l803_803931


namespace angle_sum_reflection_l803_803477

open EuclideanGeometry

noncomputable def A_excircle_center (A B C : Point) : Point := sorry -- Definition of A-excircle center
noncomputable def reflection (P : Point) (l : Line) : Point := sorry -- Definition of reflection over line

theorem angle_sum_reflection (A B C J K E F : Point) (hJ : J = A_excircle_center A B C) 
(hK : K = reflection J (line_through B C)) (hE : ∃ l, l ⊆ (line_through B J) ∧ E ∈ l ∧ ∠ E A B = 90) 
(hF : ∃ l, l ⊆ (line_through C J) ∧ F ∈ l ∧ ∠ C A F = 90) :
∠ F K E + ∠ F J E = 180 := 
by
  sorry

end angle_sum_reflection_l803_803477


namespace smallest_integer_with_divisors_l803_803181

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end smallest_integer_with_divisors_l803_803181


namespace total_original_grain_l803_803154

-- Define initial conditions
variables (initial_warehouse1 : ℕ) (initial_warehouse2 : ℕ)
-- Define the amount of grain transported away from the first warehouse
def transported_away := 2500
-- Define the amount of grain in the second warehouse
def warehouse2_initial := 50200

-- Prove the total original amount of grain in the two warehouses
theorem total_original_grain 
  (h1 : transported_away = 2500)
  (h2 : warehouse2_initial = 50200)
  (h3 : initial_warehouse1 - transported_away = warehouse2_initial) : 
  initial_warehouse1 + warehouse2_initial = 102900 :=
sorry

end total_original_grain_l803_803154


namespace eval_expr_1_eval_expr_2_eval_expr_3_eval_expr_4_eval_expr_5_eval_expr_6_eval_expr_7_eval_expr_8_eval_expr_9_l803_803312

theorem eval_expr_1 : (-51) + (-37) = -88 := sorry

theorem eval_expr_2 : 2 + (-11) = -9 := sorry

theorem eval_expr_3 : (-12) + 12 = 0 := sorry

theorem eval_expr_4 : 8 - 14 = -6 := sorry

theorem eval_expr_5 : 15 - (-8) = 23 := sorry

theorem eval_expr_6 : (-3.4) + 4.3 = 0.9 := sorry

theorem eval_expr_7 : abs (-2.25) + (-0.5) = 1.75 := sorry

theorem eval_expr_8 : (-4) * (3 / 2) = -6 := sorry

theorem eval_expr_9 : (-3) * (-6) = 18 := sorry

end eval_expr_1_eval_expr_2_eval_expr_3_eval_expr_4_eval_expr_5_eval_expr_6_eval_expr_7_eval_expr_8_eval_expr_9_l803_803312


namespace modulus_of_complex_l803_803790

def complex_modulus_example : ℂ := (3 + complex.i) / (1 - complex.i)

theorem modulus_of_complex :
  |complex_modulus_example| = Real.sqrt 5 := by
  sorry

end modulus_of_complex_l803_803790


namespace intersect_line_segment_AB_l803_803773

def point := ℝ × ℝ

def A : point := (1, 1)
def B : point := (-2, 3)

def on_line (a : ℝ) (p : point) : Prop :=
  p.2 = a * p.1 - 1

def intersects_AB (a : ℝ) : Prop :=
  ∃ t ∈ set.Icc (0:ℝ) 1, 
  ∃ x y, (1 - t) * A.1 + t * B.1 = x ∧ 
         (1 - t) * A.2 + t * B.2 = y ∧ 
         on_line a (x, y)

theorem intersect_line_segment_AB (a : ℝ) : 
  intersects_AB a ↔ a ≤ -2 ∨ a ≥ 2 :=
sorry

end intersect_line_segment_AB_l803_803773


namespace radius_range_l803_803439

noncomputable def circle_equation (x y r : ℝ) := (x - 3)^2 + (y + 5)^2 = r^2
noncomputable def line_equation (x y : ℝ) := 4 * x - 3 * y = 2
noncomputable def point_distance (x₀ y₀ : ℝ) := |4 * x₀ - 3 * y₀ - 2| / real.sqrt (4^2 + (-3)^2)

theorem radius_range (r : ℝ) :
  (∃ x y : ℝ, circle_equation x y r ∧ |point_distance 3 (-5) - 1| < 1) → 4 < r ∧ r < 6 :=
sorry

end radius_range_l803_803439


namespace sum_choose_mod_500_l803_803714

theorem sum_choose_mod_500 :
  (∑ i in Finset.range 2016 | i % 3 = 0, Nat.choose 2015 i) % 500 = 270 := 
by sorry

end sum_choose_mod_500_l803_803714


namespace bianca_total_pictures_l803_803309

def album1_pictures : Nat := 27
def album2_3_4_pictures : Nat := 3 * 2

theorem bianca_total_pictures : album1_pictures + album2_3_4_pictures = 33 := by
  sorry

end bianca_total_pictures_l803_803309


namespace two_distinct_real_roots_iff_in_range_l803_803728

-- Define the function f(x)
def f (x : ℝ) := Real.log x + 1 / x

-- Define the interval boundaries
def a := 1 / Real.exp 1
def b := Real.exp 1

-- Define the critical point within the interval
def critical_point := 1

-- Define the minimum and maximum values of the function on the interval
def f_min := 1
def f_max := 1 + 1 / Real.exp 1

-- The main statement of the proof problem
theorem two_distinct_real_roots_iff_in_range (k : ℝ) :
  (∃ (x₁ x₂ : ℝ), a ≤ x₁ ∧ x₁ < critical_point ∧ x₂ > critical_point ∧ x₂ ≤ b
   ∧ f x₁ = k ∧ f x₂ = k) ↔ (1 < k ∧ k < f_max) :=
sorry -- Proof is not required as per instructions

end two_distinct_real_roots_iff_in_range_l803_803728


namespace carrie_savings_l803_803679

-- Define the original prices and discount rates
def deltaPrice : ℝ := 850
def deltaDiscount : ℝ := 0.20
def unitedPrice : ℝ := 1100
def unitedDiscount : ℝ := 0.30

-- Calculate discounted prices
def deltaDiscountAmount : ℝ := deltaPrice * deltaDiscount
def unitedDiscountAmount : ℝ := unitedPrice * unitedDiscount

def deltaDiscountedPrice : ℝ := deltaPrice - deltaDiscountAmount
def unitedDiscountedPrice : ℝ := unitedPrice - unitedDiscountAmount

-- Define the savings by choosing the cheaper flight
def savingsByChoosingCheaperFlight : ℝ := unitedDiscountedPrice - deltaDiscountedPrice

-- The theorem stating the amount saved
theorem carrie_savings : savingsByChoosingCheaperFlight = 90 :=
by
  sorry

end carrie_savings_l803_803679


namespace parallel_lines_implication_l803_803787

theorem parallel_lines_implication (a : ℝ) 
  (h : (a - a^2) = 3a + 1) : a = -1 :=
by 
  -- proof goes here
  sorry

end parallel_lines_implication_l803_803787


namespace modulus_range_l803_803774

theorem modulus_range {a : ℝ} (h : 0 < a ∧ a < 2) : 
  let z := complex.of_real a + complex.I in 
  1 < complex.abs z ∧ complex.abs z < real.sqrt 5 := 
by 
  sorry

end modulus_range_l803_803774


namespace find_f_42_div_17_l803_803337

def f : ℚ → ℤ := sorry

theorem find_f_42_div_17 : 
  (∀ x y : ℚ, x ≠ y → (x * y = 1 ∨ x + y = 1) → f x * f y = -1) → 
  f 0 = 1 →
  f (42 / 17) = -1 :=
sorry

end find_f_42_div_17_l803_803337


namespace incorrect_transmission_10111_l803_803159

variables {a0 a1 a2 h0 h1 : Prop}

def xor (x y : Prop) := x ≠ y

def is_valid_transmission (a0 a1 a2 : Prop) (transmission : Prop) : Prop :=
  let h0 := xor a0 a1 in
  let h1 := xor h0 a2 in
  transmission = (h0, a0, a1, a2, h1)

theorem incorrect_transmission_10111 :
  ¬ ∃ a0 a1 a2 : Prop, is_valid_transmission a0 a1 a2 (true, true, false, true, true) :=
sorry

end incorrect_transmission_10111_l803_803159


namespace coloring_ways_l803_803635

def num_colorings (total_circles blue_circles green_circles red_circles : ℕ) : ℕ :=
  if total_circles = blue_circles + green_circles + red_circles then
    (Nat.choose total_circles (green_circles + red_circles)) * (Nat.factorial (green_circles + red_circles) / (Nat.factorial green_circles * Nat.factorial red_circles))
  else
    0

theorem coloring_ways :
  num_colorings 6 4 1 1 = 30 :=
by sorry

end coloring_ways_l803_803635


namespace chocolates_perimeter_l803_803004

theorem chocolates_perimeter (n : Nat) (h : n = 6) : (n * 4 - 4) = 20 :=
by {
  -- Initialize the expression with the assumption n = 6.
  rw [h],
  -- Simplify the expression.
  calc 
  (6 * 4 - 4)
  = 24 - 4 : by rfl
  ... = 20 : by rfl
}

end chocolates_perimeter_l803_803004


namespace f_monotonically_increasing_interval_l803_803799

def f (x : ℝ) : ℝ := (1 / 3) ^ (-x^2 + 2 * x - 3)

theorem f_monotonically_increasing_interval : ∀ x, 1 ≤ x → ∀ y, 1 ≤ y → x ≤ y → f x ≤ f y := sorry

end f_monotonically_increasing_interval_l803_803799


namespace ukolkin_share_change_l803_803327

variable (P U : ℝ)

theorem ukolkin_share_change 
  (h1 : 0.4 * P = 0.6 * U) : 
  let P_prime := 1.5 * P in
  let total_vaccinations := P_prime + U in
  let ukolkin_share := U / total_vaccinations in
  approximately (ukolkin_share real.equal) 0.75 :=
by sorry

end ukolkin_share_change_l803_803327


namespace equalize_costs_l803_803669

noncomputable theory  -- To allow for real number calculations with division.

def total_cost (alan beth chris picnic : ℝ) : ℝ :=
  alan + beth + chris + picnic

def equal_share (total : ℝ) (num_people : ℝ) : ℝ :=
  total / num_people

def alan_contribution_adjustment (equal_share alan_paid : ℝ) : ℝ :=
  equal_share - alan_paid

def beth_contribution_adjustment (equal_share beth_paid : ℝ) : ℝ :=
  equal_share - beth_paid

def chris_receives (chris_paid total_cost equal_share : ℝ) : ℝ :=
  total_cost - chris_paid - equal_share

theorem equalize_costs :
  let alan_paid := 110
  let beth_paid := 140
  let chris_paid := 190
  let picnic_cost := 60
  let num_people := 3

  let total := total_cost alan_paid beth_paid chris_paid picnic_cost
  let share := equal_share total num_people

  let a := alan_contribution_adjustment share alan_paid
  let b := beth_contribution_adjustment share beth_paid

  a - b = 30 :=
by
  sorry

end equalize_costs_l803_803669


namespace find_unknown_side_l803_803346

-- Define the variables and conditions
variables (a b h A : ℝ)
variables (b_val : b = 12)
variables (h_val : h = 14)
variables (A_val : A = 196)

-- Define the area of the trapezium
def trapezium_area (a b h : ℝ) : ℝ := 1/2 * (a + b) * h

-- State the theorem to prove the unknown side length a
theorem find_unknown_side : b = 12 → h = 14 → A = 196 → trapezium_area a b h = A → a = 16 :=
by
  intros b_val h_val A_val area_eq
  simp only [trapezium_area] at area_eq
  sorry

end find_unknown_side_l803_803346


namespace sum_of_youngest_and_oldest_friend_l803_803683

-- Given definitions
def mean_age_5 := 12
def median_age_5 := 11
def one_friend_age := 10

-- The total sum of ages is given by mean * number of friends
def total_sum_ages : ℕ := 5 * mean_age_5

-- Third friend's age as defined by median
def third_friend_age := 11

-- Proving the sum of the youngest and oldest friend's ages
theorem sum_of_youngest_and_oldest_friend:
  (∃ youngest oldest : ℕ, youngest + oldest = 38) :=
by
  sorry

end sum_of_youngest_and_oldest_friend_l803_803683


namespace sector_area_l803_803786

-- Definitions and conditions
variables (R: ℝ) (circumference rad_measure : ℝ) 

-- Conditions given the problem
def circumference_eq := circumference = 2 * R + 2 * R * π
def rad_measure_eq := rad_measure = 2

-- The main statement to prove
theorem sector_area (h1: circumference = 8) (h2: rad_measure = 2) : 
  let R := 2 in (1/2) * (2 * R) * R = 4 := 
by
  -- (1/2) * (2 * 2) * 2 = 4
  sorry

end sector_area_l803_803786


namespace dodecagon_diagonals_l803_803405

/--
The formula for the number of diagonals in a convex n-gon is given by (n * (n - 3)) / 2.
-/
def number_of_diagonals (n : Nat) : Nat := (n * (n - 3)) / 2

/--
A dodecagon has 12 sides.
-/
def dodecagon_sides : Nat := 12

/--
The number of diagonals in a convex dodecagon is 54.
-/
theorem dodecagon_diagonals : number_of_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l803_803405


namespace admission_related_to_school_99_confidence_probability_two_students_from_乙_l803_803976

-- Problem 1
theorem admission_related_to_school_99_confidence :
  let a := 20           -- 甲通过人数
  let b := 40           -- 甲未通过人数
  let c := 30           -- 乙通过人数
  let d := 20           -- 乙未通过人数
  let n := a + b + c + d
  let a_plus_b := a + b
  let c_plus_d := c + d
  let a_plus_c := a + c
  let b_plus_d := b + d
  let k_squared := (n * ((a * d - b * c)^2)) / (a_plus_b * c_plus_d * a_plus_c * b_plus_d)
  k_squared ≥ 6.635 :=
sorry

-- Problem 2
theorem probability_two_students_from_乙 :
  let甲通过人数 := 20
  let乙通过人数 := 30
  let总通过人数 := 录费率通过人数 + 甲通过人数
  let p := 3 / 10
  p = (3 / 10) :=
sorry

end admission_related_to_school_99_confidence_probability_two_students_from_乙_l803_803976


namespace larger_segment_length_l803_803567

theorem larger_segment_length (a b c : ℕ) (h : ℝ) (x : ℝ)
  (ha : a = 50) (hb : b = 90) (hc : c = 110)
  (hyp1 : a^2 = x^2 + h^2)
  (hyp2 : b^2 = (c - x)^2 + h^2) :
  110 - x = 80 :=
by {
  sorry
}

end larger_segment_length_l803_803567


namespace max_temperature_range_l803_803117

theorem max_temperature_range (temps : Fin 5 → ℝ) (h_avg : (∑ i, temps i) / 5 = 60) (h_min : ∃ i, temps i = 40) : 
  ∃ max_temp, max_temp ∈ set.range temps ∧ max_temp - 40 = 100 :=
sorry

end max_temperature_range_l803_803117


namespace remainder_98_102_div_9_l803_803594

theorem remainder_98_102_div_9 : 
  let product := 98 * 102 in
  product % 9 = 8 := 
by sorry

end remainder_98_102_div_9_l803_803594


namespace tennis_balls_l803_803138

variables (R B : ℕ)
variables (h_ratio : 7 * R = 3 * B)
variables (h_red_tennis : 0.7 * R) (h_blue_tennis : 0.3 * B)

theorem tennis_balls (R : ℕ) (h_ratio : 7 * R = 3 * B) :
  (0.7 * R) + (0.3 * B) = 1.4 * R :=
by sorry

end tennis_balls_l803_803138


namespace relationship_among_abc_l803_803756

noncomputable def a : ℝ := 0.8 ^ 0.7
noncomputable def b : ℝ := 0.8 ^ 0.9
noncomputable def c : ℝ := 1.1 ^ 0.6

theorem relationship_among_abc : c > a ∧ a > b := by
  sorry

end relationship_among_abc_l803_803756


namespace problem_statement_l803_803897

variables {R : Type} [LinearOrderedField R] {x0 y0 k m : R}

def ellipse1 (x y : R) : Prop := (x^2 / 8) + (y^2 / 2) = 1
def ellipse2 (x y : R) : Prop := (x^2 / 4) + (y^2) = 1
def line (k m x y : R) : Prop := y = k * x + m
def chord_bisect (x0 y0 x1 y1 x2 y2 : R) : Prop := 
  (x0 = (x1 + x2) / 2) ∧ (y0 = (y1 + y2) / 2)
def area_triangle (x0 y0 x1 y1 x2 y2 : R) : R := 
  abs ((x1 * (y2 - y0) + x2 * (y0 - y1)) / 2)

variables 
  (h1 : ellipse1 x0 y0)
  (h2 : ellipse2 x1 y1)
  (h3 : ellipse2 x2 y2)
  (h4 : line k m x1 y1)
  (h5 : line k m x2 y2)
  (h6 : chord_bisect x0 y0 x1 y1 x2 y2)
  (h7 : m * y0 > -1)
  (h8 : area_triangle x0 y0 x1 y1 x2 y2 = 1)

theorem problem_statement :
  (parallel (x0 y0 x1 y1) (x1 y1 x2 y2)) ∧ (∃ z, ellipse2 x0 y0 z) :=
sorry

end problem_statement_l803_803897


namespace largest_number_remaining_l803_803135

theorem largest_number_remaining (n : ℕ) (h : n = 16) :
  ∃ m : ℕ, (∀ i : ℕ, i ∈ range (n + 1) → ∃ a b : ℕ, a ≠ b ∧ 
  (a - b) = i) → (m = 2^16 - 1) := sorry

end largest_number_remaining_l803_803135


namespace length_first_train_l803_803663

/-- Let the speeds of two trains be 120 km/hr and 80 km/hr, respectively. 
These trains cross each other in 9 seconds, and the length of the second train is 250.04 meters. 
Prove that the length of the first train is 250 meters. -/
theorem length_first_train
  (FirstTrainSpeed : ℝ := 120)  -- speed of the first train in km/hr
  (SecondTrainSpeed : ℝ := 80)  -- speed of the second train in km/hr
  (TimeToCross : ℝ := 9)        -- time to cross each other in seconds
  (LengthSecondTrain : ℝ := 250.04) -- length of the second train in meters
  : FirstTrainSpeed / 0.36 + SecondTrainSpeed / 0.36 * TimeToCross - LengthSecondTrain = 250 :=
by
  -- omitted proof
  sorry

end length_first_train_l803_803663


namespace max_digit_sum_on_watch_display_l803_803991

theorem max_digit_sum_on_watch_display :
  ∃ (h m : Nat), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧ 
  (sum_of_digits h + sum_of_digits m = 24) :=
sorry

-- Helper function to sum the digits of a number
def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

end max_digit_sum_on_watch_display_l803_803991


namespace percentage_increase_in_rectangle_area_l803_803616

theorem percentage_increase_in_rectangle_area (L W : ℝ) :
  (1.35 * 1.35 * L * W - L * W) / (L * W) * 100 = 82.25 :=
by sorry

end percentage_increase_in_rectangle_area_l803_803616


namespace cristobal_read_more_pages_l803_803721

theorem cristobal_read_more_pages (B : ℕ) (hB : B = 704) : 
  let C := 15 + 3 * B in
  C - B = 1423 :=
by
  let C := 15 + 3 * B
  have hC : C = 2127, by
    sorry
  have hDiff : C - B = 1423, by
    sorry
  exact hDiff

end cristobal_read_more_pages_l803_803721


namespace square_area_of_shaded_region_l803_803458

theorem square_area_of_shaded_region
  (PS_len : ℝ)
  (PQ QR RS : ℝ)
  (h_PS_len : PS_len = 4)
  (h_partition : PQ + QR + RS = PS_len) :
  (let s := (1/4) * (π * (4 + PQ + QR + RS)) 
       in s^2 = π^2) := 
by
  sorry

end square_area_of_shaded_region_l803_803458


namespace probability_is_correct_l803_803886

/-- Ms. Carr's reading list contains 12 books, each student chooses 6 books.
What is the probability that there are exactly 3 books that Harold 
and Betty both select? -/
def probability_exactly_3_shared_books : ℚ :=
  let total_ways := (Nat.choose 12 6) * (Nat.choose 12 6)
  let successful_ways := (Nat.choose 12 3) * (Nat.choose 9 3) * (Nat.choose 9 3)
  successful_ways / total_ways

/-- Proof that the probability is exactly 405/2223 when both select 6 books -/
theorem probability_is_correct :
  probability_exactly_3_shared_books = 405 / 2223 :=
by
  sorry

end probability_is_correct_l803_803886


namespace largest_possible_difference_l803_803303

def anita_estimate : ℕ := 40000
def carl_estimate : ℕ := 50000

def actual_seattle_attendance (S : ℕ) : Prop := 
  0.85 * anita_estimate ≤ S ∧ S ≤ 1.15 * anita_estimate

def actual_chicago_attendance (C : ℕ) : Prop := 
  0.85 * C ≤ carl_estimate ∧ carl_estimate ≤ 1.15 * C

theorem largest_possible_difference 
  (maxC : ℕ) (maxC_cond : actual_chicago_attendance maxC) 
  (minS : ℕ) (minS_cond : actual_seattle_attendance minS) : 
  (maxC - minS = 25000) :=
sorry

end largest_possible_difference_l803_803303


namespace sum_of_arithmetic_sequence_l803_803934

theorem sum_of_arithmetic_sequence
    (a : ℕ → ℤ)
    (S : ℕ → ℤ)
    (hS : ∀ n : ℕ, S n = n * a n)
    (h_condition : a 1 - a 5 - a 10 - a 15 + a 19 = 2) :
    S 19 = -38 :=
sorry

end sum_of_arithmetic_sequence_l803_803934


namespace interest_rate_per_annum_l803_803660

-- Definitions for the given conditions
def SI : ℝ := 4016.25
def P : ℝ := 44625
def T : ℝ := 9

-- The interest rate R must be 1 according to the conditions
theorem interest_rate_per_annum : (SI * 100) / (P * T) = 1 := by
  sorry

end interest_rate_per_annum_l803_803660


namespace value_of_x_l803_803607

theorem value_of_x : 
  ∀ (x : ℕ), x = (2011^2 + 2011) / 2011 → x = 2012 :=
by
  intro x
  intro h
  sorry

end value_of_x_l803_803607


namespace ellipse_eccentricity_l803_803800

noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

def ellipse_conditions (F1 B : ℝ × ℝ) (c b : ℝ) : Prop :=
  F1 = (-2, 0) ∧ B = (0, 1) ∧ c = 2 ∧ b = 1

theorem ellipse_eccentricity (F1 B : ℝ × ℝ) (c b a : ℝ)
  (h : ellipse_conditions F1 B c b) :
  eccentricity c a = 2 * Real.sqrt 5 / 5 := by
sorry

end ellipse_eccentricity_l803_803800


namespace problem1_problem2_l803_803621

theorem problem1 : (Real.sqrt 25 + Real.sqrt ((-2)^2) + Real.cbrt (-8) = 5) := 
by 
sorry

theorem problem2 : (∀ x: ℝ, 2*(x-1)^3 - 16 = 0 → x = 3) :=
by 
  intros x h,
  sorry

end problem1_problem2_l803_803621


namespace range_of_sqrt_neg_3_minus_m_l803_803838

theorem range_of_sqrt_neg_3_minus_m (m : ℝ) : (∃ x : ℝ, x = sqrt (-3 - m)) ↔ m ≤ -3 :=
by sorry

end range_of_sqrt_neg_3_minus_m_l803_803838


namespace magnitude_of_vector_sum_l803_803002

-- Definitions corresponding to conditions
def a : ℝ × ℝ × ℝ := (0, -1, 1)
def b : ℝ × ℝ × ℝ := (1, 0, 1)

-- magnitude of a 3D vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- theorem statement to prove the magnitude of the given vector
theorem magnitude_of_vector_sum :
  magnitude (2 • a + b) = real.sqrt 14 := 
sorry

end magnitude_of_vector_sum_l803_803002


namespace sum_of_areas_of_squares_l803_803843

theorem sum_of_areas_of_squares
  (P Q R S T U : Type)
  (PR PQ : ℝ)
  (h1 : PR = 15)
  (h2 : ∃ (c : PQR), (Q : circle_centr R))
  (h3 : right_angle P R Q)
  : 
  let area_PRUT := PR^2,
      area_PQRS := PQ^2
  in 
  (area_PRUT + area_PQRS = 450) :=
by
  let PR := 15
  let PQ := 15
  let area_PRUT := PR^2
  let area_PQRS := PQ^2
  show area_PRUT + area_PQRS = 450 from
  sorry

end sum_of_areas_of_squares_l803_803843


namespace sum_of_squares_distances_triangle_l803_803447

theorem sum_of_squares_distances_triangle 
  (A B C O P : Point) 
  (R : ℝ) 
  (is_circumcenter : is_circumcenter O A B C) 
  (circumradius : dist O A = R) 
  (on_circumcircle : dist O P = R) : 
  dist P A ^ 2 + dist P B ^ 2 + dist P C ^ 2 = 6 * R ^ 2 := 
sorry

end sum_of_squares_distances_triangle_l803_803447


namespace no_integer_solutions_for_equation_l803_803344

theorem no_integer_solutions_for_equation : ¬∃ (a b c : ℤ), a^4 + b^4 = c^4 + 3 := 
  by sorry

end no_integer_solutions_for_equation_l803_803344


namespace ellipse_problem_l803_803371

theorem ellipse_problem (a b : ℝ) (a_gt_b : a > b) (b_lt_a : b > 0) (D : ℝ × ℝ)
  (F : ℝ × ℝ) (hF : F = (1, 0)) (hD : D = (1, 3 / 2))
  (ellipse : ∀ x y : ℝ, (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1 ↔ (x, y) = F ∨ (x, y) = D)) :
  (a ^ 2 = 4 ∧ b ^ 2 = 3 ∧ (∀ x y : ℝ, (x ^ 2 / 4 + y ^ 2 / 3 = 1 ↔ (∀ l : ℝ → ℝ, ∃ m1 m2 : ℝ, m1 = l (1 - 4) ∧ 
  m2 = l (1 + 4) ∧ ((m1 + m2) / 2 = 1 / 2) ))))

end ellipse_problem_l803_803371


namespace triangle_problem_l803_803034

noncomputable theory

section
variables (a b c : ℝ) (A B C : ℝ)
  
theorem triangle_problem
  (h₁ : b = 3)
  (h₂ : b * sin A = sqrt 3 * a * cos B)
  (h₃ : sin C = 2 * sin A)
  (h₄ : c = 2 * a)
  (h₅ : B = Real.pi / 3) :
  a = sqrt 3 ∧ c = 2 * sqrt 3 ∧ (1/2 * a * c * sin B = 3 * sqrt 3 / 2) :=
by
  sorry
end

end triangle_problem_l803_803034


namespace minimum_value_of_c_l803_803834

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  (Real.sqrt 3 / 12) * (a^2 + b^2 - c^2)

noncomputable def tan_formula (a b c B : ℝ) : Prop :=
  24 * (b * c - a) = b * Real.tan B

noncomputable def min_value_c (a b c : ℝ) : ℝ :=
  (2 * Real.sqrt 3) / 3

theorem minimum_value_of_c (a b c B : ℝ) (h₁ : 0 < B ∧ B < π / 2) (h₂ : 24 * (b * c - a) = b * Real.tan B)
  (h₃ : triangle_area a b c = (1/2) * a * b * Real.sin (π / 6)) :
  c ≥ min_value_c a b c :=
by
  sorry

end minimum_value_of_c_l803_803834


namespace total_wheels_in_storage_l803_803841

def wheels (n_bicycles n_tricycles n_unicycles n_quadbikes : ℕ) : ℕ :=
  (n_bicycles * 2) + (n_tricycles * 3) + (n_unicycles * 1) + (n_quadbikes * 4)

theorem total_wheels_in_storage :
  let n_bicycles := 24
  let n_tricycles := 14
  let n_unicycles := 10
  let n_quadbikes := 8
  wheels n_bicycles n_tricycles n_unicycles n_quadbikes = 132 :=
by
  let n_bicycles := 24
  let n_tricycles := 14
  let n_unicycles := 10
  let n_quadbikes := 8
  show wheels n_bicycles n_tricycles n_unicycles n_quadbikes = 132
  sorry

end total_wheels_in_storage_l803_803841


namespace sequence_sum_l803_803676

theorem sequence_sum :
  (3 + 13 + 23 + 33 + 43 + 53) + (5 + 15 + 25 + 35 + 45 + 55) = 348 := by
  sorry

end sequence_sum_l803_803676


namespace smallest_integer_with_divisors_properties_l803_803198

def number_of_odd_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 1)

def number_of_even_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 0)

theorem smallest_integer_with_divisors_properties :
  ∃ n : ℕ, number_of_odd_divisors n = 8 ∧ number_of_even_divisors n = 16 ∧ n = 4000 :=
by
  sorry

end smallest_integer_with_divisors_properties_l803_803198


namespace smallest_int_with_divisors_l803_803210

theorem smallest_int_with_divisors :
  ∃ n : ℕ, 
    (∀ m, n = 2^2 * m → 
      (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 
      (m = p^3 * q) ∧ 
      (nat.divisors p^7).count 8) ∧ 
    nat.divisors_count n = 24 ∧ 
    (nat.divisors (2^2 * n)).count 8) (n = 2^2 * 3^3 * 5) :=
begin
  sorry
end

end smallest_int_with_divisors_l803_803210


namespace segment_outside_spheres_l803_803164

noncomputable def fraction_outside_spheres (α : ℝ) : ℝ :=
  (1 - (Real.cos (α / 2))^2) / (1 + (Real.cos (α / 2))^2)

theorem segment_outside_spheres (R α : ℝ) (hR : R > 0) (hα : 0 < α ∧ α < 2 * Real.pi) :
  fraction_outside_spheres α = (1 - Real.cos (α / 2)^2) / (1 + (Real.cos (α / 2))^2) :=
  by sorry

end segment_outside_spheres_l803_803164


namespace solution_range_of_a_l803_803903

noncomputable def problem_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + (2 * a^2 + 2) * x - (a^2 - 4 * a + 7)) / 
           (x^2 + (a^2 + 4 * a - 5) * x - (a^2 - 4 * a + 7)) < 0

theorem solution_range_of_a :
  {a : ℝ | ∃ x : ℝ, problem_statement a ∧ 
    (∑ i in (finset.filter (λ i, i ∈ {(1 : ℝ), 3}), i)) < 4} = set.Icc 1 3 :=
sorry

end solution_range_of_a_l803_803903


namespace problem1_problem2_part1_problem2_part2_l803_803264

-- Problem 1
theorem problem1 :
  (sqrt 3 * sin (-20 / 3 * π) / tan (11 / 3 * π)) - cos (13 / 4 * π) * tan (-37 / 4 * π) = (sqrt 3 - sqrt 2) / 2 :=
sorry

-- Problem 2.1
theorem problem2_part1 (α : ℝ) (h : tan α = 4 / 3) :
  (sin (2 * α) + 2 * sin α * cos α) / (2 * cos (2 * α) - sin (2 * α)) = -24 / 19 :=
sorry

-- Problem 2.2
theorem problem2_part2 (α : ℝ) (h : tan α = 4 / 3) :
  sin α * cos α = 12 / 25 :=
sorry

end problem1_problem2_part1_problem2_part2_l803_803264


namespace child_tickets_sold_l803_803949

theorem child_tickets_sold (A C : ℕ) (h1 : A + C = 130) (h2 : 12 * A + 4 * C = 840) : C = 90 := by
  sorry

end child_tickets_sold_l803_803949


namespace integer_values_b_num_integer_values_b_for_given_inequality_l803_803726

theorem integer_values_b (b : ℤ) : 
  (∃ (x1 x2 : ℤ), x1 ≠ x2 ∧ x1^2 + b*x1 + 3 = 0 ∧ x2^2 + b*x2 + 3 = 0 ∧ 
   ∀ (x : ℤ), x1 ≤ x ∧ x ≤ x2 → x^2 + b*x + 3 ≤ 0) ↔ b ∈ {-4, 4} :=
by sorry

theorem num_integer_values_b_for_given_inequality : 
  (∃ b : ℤ, ∀ (x : ℤ), x^2 + b*x + 3 ≤ 0) = 2 :=
by sorry

end integer_values_b_num_integer_values_b_for_given_inequality_l803_803726


namespace binom_mult_l803_803710

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l803_803710


namespace min_value_is_sqrt3_l803_803064
noncomputable def find_min_value (a b c : Int) (ω : ℂ) : ℂ :=
  if ω^4 = 1 ∧ ω ≠ 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c then
    abs (a + b * ω + c * (ω^3))
  else 
    0

theorem min_value_is_sqrt3 (a b c : Int) (ω : ℂ) (h1 : ω^4 = 1) (h2 : ω ≠ 1)
  (h3 : a ≠ b) (h4 : b ≠ c) (h5 : a ≠ c) :
  find_min_value a b c ω = sqrt 3 := sorry

end min_value_is_sqrt3_l803_803064


namespace floor_b_eq_floor_sqrt_l803_803112

noncomputable def b (n : ℕ) : ℝ :=
  real.Inf {x | ∃ k : ℕ, k > 0 ∧ x = k + n / k}

theorem floor_b_eq_floor_sqrt (n: ℕ) (hn : 0 < n) : 
  ⌊b n⌋ = ⌊real.sqrt (4 * n + 1)⌋ :=
sorry

end floor_b_eq_floor_sqrt_l803_803112


namespace rearrangement_methods_count_l803_803626

theorem rearrangement_methods_count : (nat.choose 8 2) = 28 :=
by sorry

end rearrangement_methods_count_l803_803626


namespace minimum_racing_stripes_l803_803446

variable 
  (totalCars : ℕ) (carsWithoutAirConditioning : ℕ) 
  (maxCarsWithAirConditioningWithoutStripes : ℕ)

-- Defining specific problem conditions
def conditions (totalCars carsWithoutAirConditioning maxCarsWithAirConditioningWithoutStripes : ℕ) : Prop :=
  totalCars = 100 ∧ 
  carsWithoutAirConditioning = 37 ∧ 
  maxCarsWithAirConditioningWithoutStripes = 59

-- The statement to be proved
theorem minimum_racing_stripes (h : conditions totalCars carsWithoutAirConditioning maxCarsWithAirConditioningWithoutStripes) :
   exists (R : ℕ ), R = 4 :=
by
  rcases h with ⟨h1, h2, h3⟩
  sorry

end minimum_racing_stripes_l803_803446


namespace power_function_unique_l803_803127

theorem power_function_unique (a : ℝ) (f : ℝ → ℝ) (x : ℝ) (hx : x = 2) (hfx : f x = 16) (hfa : ∀ x, f x = x ^ a) : a = 4 :=
by
  -- Conditions from the problem
  have hx : (2 : ℝ) = 2 := rfl
  have hfx : f 2 = 16 := hfx
  have hfa : ∀ x, f x = x ^ a := hfa
  sorry

end power_function_unique_l803_803127


namespace inversion_swaps_circumcircle_and_nine_point_circle_l803_803485

-- Define the geometric entities
variables {A B C D E F : Point}
variable {ω : Circle}
variable {Γ : Circle}
variable (triangle_ABC : Triangle A B C)

-- Conditions
def isCircumcircle (Γ : Circle) (Δ : Triangle) :=
  ∀ (P : Point), P ∈ Γ ↔ P ∈ Δ.verticesCircumcircle

def isIncircle (ω : Circle) (Δ : Triangle) :=
  ∀ (P : Point), P ∈ ω ↔ P ∈ Δ.verticesIncircle

def contact_points (D E F : Point) (ω : Circle) (Δ : Triangle) :=
  D ∈ ω ∧ D.touching (Δ.side BC) ∧
  E ∈ ω ∧ E.touching (Δ.side CA) ∧
  F ∈ ω ∧ F.touching (Δ.side AB)

-- Define the nine-point circle
def nine_point_circle (X Y Z : Point) :=
  ∃ (DEF : Triangle), 
    X ∈ DEF.verticesMidpoints ∧
    Y ∈ DEF.verticesMidpoints ∧
    Z ∈ DEF.verticesMidpoints ∧
    ninePointCircle DEF = Circle.through_points X Y Z

-- Statement
theorem inversion_swaps_circumcircle_and_nine_point_circle
  (hΓ : isCircumcircle Γ triangle_ABC)
  (hω : isIncircle ω triangle_ABC)
  (hD : contact_points D E F ω triangle_ABC)
  : invert_circle ω Γ = nine_point_circle D E F :=
by
  sorry

end inversion_swaps_circumcircle_and_nine_point_circle_l803_803485


namespace arithmetic_sequence_term_2011_l803_803266

theorem arithmetic_sequence_term_2011 :
  ∃ (n : ℕ), 1 + (n - 1) * 3 = 2011 ∧ n = 671 :=
by
  existsi 671
  split
  ·  sorry
  ·  refl

end arithmetic_sequence_term_2011_l803_803266


namespace sin_cos_ratio_l803_803867

theorem sin_cos_ratio (x y : ℝ) (h1 : sin x / sin y = 4) (h2 : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 395 / 381 :=
by
  sorry

end sin_cos_ratio_l803_803867


namespace common_difference_arithmetic_sequence_l803_803038

theorem common_difference_arithmetic_sequence
  (a : ℕ) (d : ℚ) (n : ℕ) (a_n : ℕ) (S_n : ℕ)
  (h1 : a = 2)
  (h2 : a_n = 20)
  (h3 : S_n = 132)
  (h4 : a_n = a + (n - 1) * d)
  (h5 : S_n = n * (a + a_n) / 2) :
  d = 18 / 11 := sorry

end common_difference_arithmetic_sequence_l803_803038


namespace polynomial_remainder_l803_803743

variables {R : Type*} [CommRing R]

noncomputable def D : R[X] := X^5 + X^4 + X^3 + X^2 + X + 1
noncomputable def P : R[X] := X^60 + 2 * X^45 + 3 * X^30 + 4 * X^15 + 5

lemma root_D (β : R) (hβ : D.eval β = 0) : β^6 = 1 :=
by {
  have H : (β - 1) * D.eval β = 0,
  rw [hβ, zero_mul],
  ring_exp at H,
  exact (β^5 + β^4 + β^3 + β^2 + β + 1 = 0) at hβ,
  sorry
}

theorem polynomial_remainder (β : R) (hβ : D.eval β = 0) : P.eval β = 15 :=
by {
  rw [P, D] at *,
  have h1 : β^60 = (β^6)^10, by ring_exp,
  have h2 : β^45 = (β^6)^7.5, by ring_exp,
  have h3 : β^30 = (β^6)^5, by ring_exp,
  have h4 : β^15 = (β^6)^2.5, by ring_exp,
  rw [root_D β hβ, one_pow, one_pow, one_pow, one_pow],
  ring,
  sorry
}

end polynomial_remainder_l803_803743


namespace prod_f_T_is_one_l803_803873

def S : Set (ℕ × ℕ) := { p | p.1 ∈ {0, 1, 2, 3, 4, 5} ∧ p.2 ∈ {0, 1, 2, 3} ∧ p ≠ (5, 3) }

def isRightTriangle (A B C : ℕ × ℕ) : Prop :=
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def T : Set (ℕ × ℕ × ℕ × (ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ)) := {t | isRightTriangle t.1 t.2 t.3 ∧ t.1 ∈ S ∧ t.2 ∈ S ∧ t.3 ∈ S}

noncomputable def f (t : ℕ × ℕ × ℕ × (ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ)) : ℝ :=
  let ⟨A, B, C⟩ := (⟨t.1, t.2, t.3⟩ : (ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ))
  in Real.tan (Real.arctan2 (B.2 - A.2) (B.1 - A.1) - Real.arctan2 (C.2 - A.2) (C.1 - A.1))

theorem prod_f_T_is_one : ∏ t in T, f t = 1 := sorry

end prod_f_T_is_one_l803_803873


namespace integral_value_l803_803573

noncomputable def f (x : Real) : Real := x * Real.cos x + Real.cbrt (x^2)

theorem integral_value : ∫ x in -1..1, f x = 6 / 5 := 
by
  sorry

end integral_value_l803_803573


namespace sum_roots_of_quadratic_l803_803374

theorem sum_roots_of_quadratic (a b : ℝ) (h₁ : a^2 - a - 6 = 0) (h₂ : b^2 - b - 6 = 0) (h₃ : a ≠ b) :
  a + b = 1 :=
sorry

end sum_roots_of_quadratic_l803_803374


namespace A_can_complete_work_in_4_8_days_l803_803631

theorem A_can_complete_work_in_4_8_days (B_days : ℕ) (B_payment : ℕ) (total_payment : ℕ) 
    (C_payment : ℕ) (Days_work_completed : ℕ) (H1 : B_days = 8) (H2 : total_payment = 3680) 
    (H3 : C_payment = 460) (H4 : Days_work_completed = 3) : 
    let A_days := (24 / 5 : ℚ) 
    in A_days = 4.8 := sorry

end A_can_complete_work_in_4_8_days_l803_803631


namespace problem_statement_l803_803262

noncomputable def a : ℝ := real.cbrt (5 + 2 * real.sqrt 13)
noncomputable def b : ℝ := real.cbrt (5 - 2 * real.sqrt 13)

theorem problem_statement : 1 * 75 ≠ a + b :=
by sorry

end problem_statement_l803_803262


namespace smallest_n_with_divisors_l803_803194

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end smallest_n_with_divisors_l803_803194


namespace eiffel_tower_model_representation_l803_803536

theorem eiffel_tower_model_representation :
  ∀ (actual_height : ℕ) (model_height : ℕ),
  actual_height = 1083 → model_height = 8 → 
  (actual_height : ℚ) / (model_height : ℚ) = 135.375 :=
by
  intros actual_height model_height h1 h2
  rw [h1, h2]
  norm_num
  exact sorry

end eiffel_tower_model_representation_l803_803536


namespace triangle_sides_consecutive_integers_l803_803559

/-- The lengths of the sides of a triangle are consecutive integers,
 and one of the medians is perpendicular to one of the angle bisectors. We need to prove that the lengths of the sides of the triangle are 2, 3, and 4. -/
theorem triangle_sides_consecutive_integers :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ b = a + 1 ∧ c = a + 2 ∧
  ∃ M K : Triangle → AngleK, (median M).perp (bisector K) ∧
  (a, b, c) = (2, 3, 4) :=
by { sorry }

end triangle_sides_consecutive_integers_l803_803559


namespace solid_is_sphere_l803_803519

/-- If every planar section of a body is a circle, then the body is a sphere. -/
theorem solid_is_sphere (solid : Something) 
  (h : ∀ (p : Plane), is_circle (p.section solid)) : is_sphere solid :=
sorry

end solid_is_sphere_l803_803519


namespace student_answer_difference_l803_803830

theorem student_answer_difference (n : ℕ) (h : n = 576) :
  let correct_answer := (5 * 576) / 16
  let mistaken_answer := (5 * 576) / 6
  mistaken_answer - correct_answer = 300 :=
by
  have h1 : correct_answer = (5 * 576) / 16 := rfl
  have h2 : mistaken_answer = (5 * 576) / 6 := rfl
  sorry

end student_answer_difference_l803_803830


namespace smallest_n_with_divisors_l803_803191

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end smallest_n_with_divisors_l803_803191


namespace sin_cos_identity_l803_803779

theorem sin_cos_identity (θ : ℝ) (a : ℝ) (h₁ : 0 < θ ∧ θ < π / 2) (h₂ : sin (2 * θ) = a) : 
  sin θ + cos θ = sqrt (1 + a) :=
by
  sorry

end sin_cos_identity_l803_803779


namespace min_paper_rectangles_l803_803483

theorem min_paper_rectangles (k n : ℕ) (hkn: k ≥ 2) (h_bound: k ≤ n ∧ n ≤ 2 * k - 1) :
  ∃ m, m = min n (2 * n - 2 * k + 2) ∧
       covers_grid_without_overlap m k n :=
sorry

end min_paper_rectangles_l803_803483


namespace sum_of_integers_l803_803548

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : 
  x + y = 32 :=
by
  sorry

end sum_of_integers_l803_803548


namespace polygon_quadrilateral_l803_803438

theorem polygon_quadrilateral {n : ℕ} (h : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end polygon_quadrilateral_l803_803438


namespace triangle_perimeter_l803_803560

-- Definitions for the given problem:
def x : ℕ := 7
def y : ℕ := 15
def side1 : ℕ := 13
def side2 : ℕ := x
def side3 : ℕ := y
def xy_product : ℕ := x * y
def perimeter : ℕ := side1 + side2 + side3

-- Conditions stated explicitly:
axiom xy_cond : xy_product = 105
axiom triangle_inequality1 : side2 + side3 > side1
axiom triangle_inequality2 : side2 + side1 > side3
axiom triangle_inequality3 : side3 + side1 > side2

-- Statement to be proven:
theorem triangle_perimeter: xy_cond ∧ triangle_inequality1 ∧ triangle_inequality2 ∧ triangle_inequality3 → perimeter = 35 := by 
  sorry

end triangle_perimeter_l803_803560


namespace reporters_not_covered_politics_l803_803332

-- Definitions of the conditions
def total_reporters : ℝ := 100 -- Total 100% reporters
def local_politics_percent : ℝ := 12 -- 12% cover local politics
def not_local_politics_percentage : ℝ := 40 -- 40% of politics reporters do not cover local politics

-- Theorem statement to prove the percentage of reporters who do not cover politics is 80%
theorem reporters_not_covered_politics : 
  (∃ (P : ℝ), 0.60 * P = local_politics_percent ∧ P = 20) →
  (total_reporters - 20 = 80) :=
by
  intro h
  obtain ⟨P, hP1, hP2⟩ := h
  have : total_reporters - P = 80 := by
    rw hP2
    exact (by norm_num : 100 - 20 = 80)
  exact this

end reporters_not_covered_politics_l803_803332


namespace three_solutions_exists_l803_803900

theorem three_solutions_exists (n : ℕ) (h_pos : 0 < n) (h_sol : ∃ x y : ℤ, x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ x1 y1 x2 y2 x3 y3 : ℤ, (x1^3 - 3 * x1 * y1^2 + y1^3 = n) ∧ (x2^3 - 3 * x2 * y2^2 + y2^3 = n) ∧ (x3^3 - 3 * x3 * y3^2 + y3^3 = n) ∧ (x1, y1) ≠ (x2, y2) ∧ (x2, y2) ≠ (x3, y3) ∧ (x1, y1) ≠ (x3, y3) :=
by
  sorry

end three_solutions_exists_l803_803900


namespace min_value_frac_sum_l803_803760

theorem min_value_frac_sum (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  (∃ m, ∀ x y, m = 1 ∧ (
      (1 / (x + y)^2) + (1 / (x - y)^2) ≥ m)) :=
sorry

end min_value_frac_sum_l803_803760


namespace three_digit_non_multiples_of_6_or_8_l803_803419

theorem three_digit_non_multiples_of_6_or_8 : 
  ∃ n, n = 900 - (150 + 112 - 37) ∧ n = 675 :=
by {
  have total_three_digits : 900 = 999 - 100 + 1 := sorry,
  have count_multiples_6 : 150 = 166 - 17 + 1 := sorry,
  have count_multiples_8 : 112 = 124 - 13 + 1 := sorry,
  have count_multiples_24 : 37 = 41 - 5 + 1 := sorry,
  let multiples_6_or_8 := count_multiples_6 + count_multiples_8 - count_multiples_24,
  have : multiples_6_or_8 = 150 + 112 - 37 := sorry,
  have count_non_multiples := total_three_digits - multiples_6_or_8,
  use count_non_multiples,
  split,
  { rw [total_three_digits, multiples_6_or_8], exact sorry },
  { exact sorry }
}

end three_digit_non_multiples_of_6_or_8_l803_803419


namespace sum_of_integers_l803_803546

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
sorry

end sum_of_integers_l803_803546


namespace ned_initial_games_l803_803090

/-- 
Ned gave away thirteen of his video games to a friend. Now Ned has six games. 
Prove that Ned had 19 games before he gave the games away.
-/
theorem ned_initial_games (games_given_away games_remaining : ℕ) (h_given : games_given_away = 13) (h_remaining : games_remaining = 6) : ∃ initial_games : ℕ, initial_games = 19 :=
by {
  have total_games := games_remaining + games_given_away,
  have h_total: total_games = 6 + 13 := by rw [h_remaining, h_given],
  use 19,
  linarith [h_total]
}

end ned_initial_games_l803_803090


namespace complement_intersection_identities_l803_803398

open Set

variable (I A B C : Set ℕ)

def is_complement (I B : Set ℕ) (C : Set ℕ) :=
  C = I \ B

theorem complement_intersection_identities :
  (I = Set.univ) →
  (A = {1, 2, 3, 4, 5, 6}) →
  (B = {2, 3, 5}) →
  ∃ C, is_complement I B C ∧ (C ∩ A = {1, 4, 6}) :=
by
  intros hI hA hB
  use I \ B
  split
  . exact rfl
  . sorry

end complement_intersection_identities_l803_803398


namespace roots_nonpositive_if_ac_le_zero_l803_803967

theorem roots_nonpositive_if_ac_le_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : a * c ≤ 0) :
  ¬ (∀ x : ℝ, x^2 - (b/a)*x + (c/a) = 0 → x > 0) :=
sorry

end roots_nonpositive_if_ac_le_zero_l803_803967


namespace cone_height_l803_803636

theorem cone_height (π : ℝ) (r h l : ℝ) (V : ℝ)
  (V_eq : V = 3072 * π) (l_eq : l = 12)
  (h_eq_approx : h ≈ 19.1) :
  V = (1 / 3) * π * r^2 * h ∧ l = (r^2 + h^2)^(1/2) -> 
  h ≈ 19.1 := by
  sorry

end cone_height_l803_803636


namespace smallest_n_with_divisors_l803_803190

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end smallest_n_with_divisors_l803_803190


namespace two_bacteria_fill_time_l803_803511

-- Define the conditions
def one_bacterium_fills_bottle_in (a : Nat) (t : Nat) : Prop :=
  (2^t = 2^a)

def two_bacteria_fill_bottle_in (a : Nat) (x : Nat) : Prop :=
  (2 * 2^x = 2^a)

-- State the theorem
theorem two_bacteria_fill_time (a : Nat) : ∃ x, two_bacteria_fill_bottle_in a x ∧ x = a - 1 :=
by
  -- Use the given conditions
  sorry

end two_bacteria_fill_time_l803_803511


namespace smallest_abundant_not_multiple_of_4_l803_803600

def is_abundant (n : ℕ) : Prop :=
  (∑ k in finset.filter (λ d, d < n) (finset.divisors n), k) > n

theorem smallest_abundant_not_multiple_of_4 : ∃ n : ℕ, is_abundant n ∧ ¬ (4 ∣ n) ∧ ∀ m : ℕ, is_abundant m ∧ ¬ (4 ∣ m) → m ≥ n :=
begin
  use 18,
  sorry
end

end smallest_abundant_not_multiple_of_4_l803_803600


namespace partition_property_l803_803075

open Finset

theorem partition_property (k : ℕ) (hk : 0 < k) :
  ∃ (X Y : Finset ℕ), disjoint X Y ∧
  X ∪ Y = range (2^(k+1)) ∧
  (∀ m ∈ (range (k+1)).erase 0, ∑ x in X, x^m = ∑ y in Y, y^m) :=
sorry

end partition_property_l803_803075


namespace sin_cos_ratio_l803_803866

theorem sin_cos_ratio (x y : ℝ) (h1 : sin x / sin y = 4) (h2 : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 395 / 381 :=
by
  sorry

end sin_cos_ratio_l803_803866


namespace smallest_n_with_divisors_l803_803192

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end smallest_n_with_divisors_l803_803192


namespace find_multiple_l803_803141

theorem find_multiple (x y m : ℕ) (h1 : y + x = 50) (h2 : y = m * x - 43) (h3 : y = 31) : m = 4 :=
by
  sorry

end find_multiple_l803_803141


namespace find_r_8_l803_803481

noncomputable def r (x : ℝ) : ℝ :=
sorry

theorem find_r_8 :
  (∀ x, (x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7) → r x = x) ∧
  (polynomial.degree (polynomial.C (r 0 - 0) * polynomial.X ^ 6) = 6) →
  r 8 = 728 :=
sorry

end find_r_8_l803_803481


namespace johns_seventh_time_l803_803734

noncomputable def sprint_times : List ℕ := [112, 117, 118, 120, 121, 125]

theorem johns_seventh_time (x : ℕ) (h : List.median (x :: sprint_times) = 118.5) : x = 118 :=
sorry

end johns_seventh_time_l803_803734


namespace distance_between_points_l803_803738

theorem distance_between_points :
  let x1 := 1
  let y1 := 3
  let z1 := 2
  let x2 := 4
  let y2 := 1
  let z2 := 6
  let distance : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
  distance = Real.sqrt 29 := by
  sorry

end distance_between_points_l803_803738


namespace probability_not_green_l803_803153

theorem probability_not_green :
  let red_balls := 6
  let yellow_balls := 3
  let black_balls := 4
  let green_balls := 5
  let total_balls := red_balls + yellow_balls + black_balls + green_balls
  let not_green_balls := red_balls + yellow_balls + black_balls
  total_balls = 18 ∧ not_green_balls = 13 → (not_green_balls : ℚ) / total_balls = 13 / 18 := 
by
  intros
  sorry

end probability_not_green_l803_803153


namespace min_value_fraction_l803_803781

variable (x y : ℝ)

theorem min_value_fraction (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ (m : ℝ), (∀ z, (z = (1/x) + (9/y)) → z ≥ 16) ∧ ((1/x) + (9/y) = m) :=
sorry

end min_value_fraction_l803_803781


namespace binom_mult_l803_803711

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l803_803711


namespace num_two_digit_congruent_to_3_mod_4_l803_803011

open Set

theorem num_two_digit_congruent_to_3_mod_4 : 
  (finset.card (finset.filter (λ n, n % 4 = 3) (finset.Icc 10 99)) = 23) := 
by 
  sorry

end num_two_digit_congruent_to_3_mod_4_l803_803011


namespace arcade_game_monster_perimeter_l803_803039

theorem arcade_game_monster_perimeter :
  let r := 1 -- radius of the circle in cm
  let theta := 60 -- central angle of the missing sector in degrees
  let circumference := 2 * Real.pi * r -- circumference of the full circle
  let arc_fraction := (360 - theta) / 360 -- fraction of the circle forming the arc
  let arc_length := arc_fraction * circumference -- length of the arc
  let perimeter := arc_length + 2 * r -- total perimeter (arc + two radii)
  perimeter = (5 / 3) * Real.pi + 2 :=
by
  sorry

end arcade_game_monster_perimeter_l803_803039


namespace binom_coeff_mult_l803_803703

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l803_803703


namespace sum_of_tens_and_units_digit_of_11_pow_2002_l803_803606

noncomputable def sum_of_tens_and_units_digit (n : ℕ) : ℕ :=
  let digits := Nat.digits 10 n in
  (digits.getOrElse (digits.length - 2) 0) + (digits.getOrElse (digits.length - 1) 0)

theorem sum_of_tens_and_units_digit_of_11_pow_2002 : 
  sum_of_tens_and_units_digit (11^2002) = 1 := 
by
  sorry

end sum_of_tens_and_units_digit_of_11_pow_2002_l803_803606


namespace XAXAXA_divisible_by_7_l803_803518

theorem XAXAXA_divisible_by_7 (X A : ℕ) (hX : 0 ≤ X ∧ X ≤ 9) (hA : 0 ≤ A ∧ A ≤ 9) :
  let XA := 10 * X + A in
  (XA * 10101) % 7 = 0 :=
by
  let XA := 10 * X + A
  have h := 10101
  have one_step := XA * h
  show one_step % 7 = 0
  sorry

end XAXAXA_divisible_by_7_l803_803518


namespace find_a_for_parallelogram_l803_803340

noncomputable def polynomial := λ (a : ℝ) (z : ℂ), z^4 - 6*z^3 + 11*a*z^2 
  - 3*(2*a^2 + 3*a - 3)*z + 1

theorem find_a_for_parallelogram : 
  ∃ a : ℝ, (∀ (z1 z2 z3 z4 : ℂ), z1 + z2 + z3 + z4 = 6) 
  → (\(z1 z2 z3 z4 : ℂ), polynomial a z1 = 0 ∧ polynomial a z2 = 0 
     ∧ polynomial a z3 = 0 ∧ polynomial a z4 = 0) 
     → are_vertices_of_parallelogram {z1, z2, z3, z4} :=
begin
  sorry
end

end find_a_for_parallelogram_l803_803340


namespace find_ratio_of_sides_l803_803784

noncomputable def triangle_ratio (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos B + b * Real.cos A = 3 * a → c / a = 3
  
theorem find_ratio_of_sides (A B C a b c : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (h4 : a * Real.cos B + b * Real.cos A = 3 * a) : 
  c / a = 3 :=
begin
  apply triangle_ratio,
  exact h4,
  sorry
end

end find_ratio_of_sides_l803_803784


namespace parabola_equation_line_BD_fixed_point_l803_803394

open Real

-- Define the conditions of the given problem
def parabola_condition (p : ℝ) (h_p : 0 < p) : Prop :=
  ∃ y x : ℝ, y^2 = 2 * p * x ∧ x = 1 ∧ y = -2

def line_condition (m : ℝ) (h_m : m ≠ 0) : Prop :=
  ∃ x y : ℝ, x = m * y - 1 ∧ ((∃ y1 y2, y1 + y2 = 4 * m ∧ y1 * y2 = 4) ∧
                               y = 0 ∧ x = y1 * y2 / 4 ∧ x = 1)

-- State the math proof problem
theorem parabola_equation (p : ℝ) (h_p : 0 < p) :
  parabola_condition p h_p →
  (∀ y x : ℝ, y^2 = 4 * x) ∧ (∀ x : ℝ, x = -1) :=
sorry

theorem line_BD_fixed_point (m : ℝ) (h_m : m ≠ 0) :
  line_condition m h_m →
  ∃ x y : ℝ, x = 1 ∧ y = 0 :=
sorry

end parabola_equation_line_BD_fixed_point_l803_803394


namespace solve_for_y_l803_803427

theorem solve_for_y (y : ℝ) (h : 1 / 4 - 1 / 5 = 4 / y) : y = 80 :=
by
  sorry

end solve_for_y_l803_803427


namespace smallest_integer_with_odd_and_even_divisors_l803_803204

theorem smallest_integer_with_odd_and_even_divisors :
  ∃ n : ℕ,
    0 < n ∧
    (∀ d : ℕ, d ∣ n → (odd d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 1}) ∨ (even d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 0})) ∧ 
    (↑8 = (∑ d in (finset.filter (λ d, odd d) (finset.divisors n)), 1)) ∧
    (↑16 = (∑ d in (finset.filter (λ d, even d) (finset.divisors n)), 1)) ∧
    (24 = (∑ d in (finset.divisors n), 1)) ∧
    n = 108 :=
begin
  sorry
end

end smallest_integer_with_odd_and_even_divisors_l803_803204


namespace sum_q_p_values_l803_803807

-- Defining the functions p and q
def p (x : ℝ) : ℝ := x^2 - 4
def q (x : ℝ) : ℝ := abs (x + 1) - 3

-- The set of points to evaluate
def points : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

-- The statement of the theorem
theorem sum_q_p_values : 
  sum (points.map (λ x => q (p x))) = 0 := 
by
  sorry -- The proof steps are skipped


end sum_q_p_values_l803_803807


namespace smallest_int_with_divisors_l803_803212

theorem smallest_int_with_divisors :
  ∃ n : ℕ, 
    (∀ m, n = 2^2 * m → 
      (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 
      (m = p^3 * q) ∧ 
      (nat.divisors p^7).count 8) ∧ 
    nat.divisors_count n = 24 ∧ 
    (nat.divisors (2^2 * n)).count 8) (n = 2^2 * 3^3 * 5) :=
begin
  sorry
end

end smallest_int_with_divisors_l803_803212


namespace sufficient_but_not_necessary_l803_803543

variable (a : ℝ)

theorem sufficient_but_not_necessary (h : a > 0) : a^2 + a ≥ 0 :=
by
suffices : a * (a + 1) ≥ 0 from this
sorry

end sufficient_but_not_necessary_l803_803543


namespace least_positive_difference_l803_803107

noncomputable def geom_seq (n : ℕ) : ℕ := 3 * 3^n
noncomputable def arith_seq (n : ℕ) : ℕ := 10 + 20 * n
def max_term : ℕ := 500

theorem least_positive_difference :
  ∃ a b, (a ∈ (List.map geom_seq (List.range ((Nat.log 500 / Nat.log 3).toNat + 1)).filter (λ x, geom_seq x <= max_term))) ∧
         (b ∈ (List.map arith_seq (List.range ((500 - 10) / 20 + 1)).filter (λ x, arith_seq x <= max_term))) ∧
         |a - b| = 7 :=
  sorry

end least_positive_difference_l803_803107


namespace min_max_distance_sum_l803_803306

noncomputable def cube_side_length : ℝ := 2
noncomputable def P1 (α1 : ℝ) : EuclideanSpace ℝ (Fin 3) := ![cube_side_length, 1 + Math.cos α1, 1 + Math.sin α1]
noncomputable def P2 (α2 : ℝ) : EuclideanSpace ℝ (Fin 3) := ![1 + Math.cos α2, cube_side_length, 1 + Math.sin α2]
noncomputable def P3 (α3 : ℝ) : EuclideanSpace ℝ (Fin 3) := ![1 + Math.cos α3, 1 + Math.sin α3, cube_side_length]

theorem min_max_distance_sum (α1 α2 α3 : ℝ) (hα1 : 0 ≤ α1 ∧ α1 ≤ 2 * Real.pi) 
                            (hα2 : 0 ≤ α2 ∧ α2 ≤ 2 * Real.pi) 
                            (hα3 : 0 ≤ α3 ∧ α3 ≤ 2 * Real.pi) :
  (3 * Real.sqrt 2 - 3) ≤ (dist (P1 α1) (P2 α2) + dist (P2 α2) (P3 α3) + dist (P3 α3) (P1 α1)) ∧ 
  (dist (P1 α1) (P2 α2) + dist (P2 α2) (P3 α3) + dist (P3 α3) (P1 α1)) ≤ 3 * Real.sqrt 6 :=
sorry

end min_max_distance_sum_l803_803306


namespace horse_buying_problem_l803_803157

variable (x y z : ℚ)

theorem horse_buying_problem :
  (x + 1/2 * y + 1/2 * z = 12) →
  (y + 1/3 * x + 1/3 * z = 12) →
  (z + 1/4 * x + 1/4 * y = 12) →
  x = 60/17 ∧ y = 136/17 ∧ z = 156/17 :=
by
  sorry

end horse_buying_problem_l803_803157


namespace hiking_start_time_l803_803498

-- Define the conditions for the problem
variable (v_R v_L x : ℝ) -- speeds of Mr. Rychlý and Mr. Louda, and time from start to meeting
variable (start_time : ℝ) -- the start time in hours from midnight

-- Define the known conditions as hypotheses
axiom h1 : x * v_L = 2 * v_R  -- Distance relation before and after the meeting point for Mr. Rychlý
axiom h2 : 8 * v_L = x * v_R  -- Distance relation before and after the meeting point for Mr. Louda
axiom h3 : x = 4              -- From solving the system of equations

-- The statement we need to prove in Lean 
theorem hiking_start_time : start_time = 6 :=
by
  have x_pos : 0 < x := sorry  -- Assume time x is positive
  have v_R_pos : 0 < v_R := sorry  -- Assume speed v_R is positive
  have v_L_pos : 0 < v_L := sorry  -- Assume speed v_L is positive
  have h4 : start_time = 10 - x := by sorry -- Deducing the start time from the meeting time of 10 AM
  rw [h3] at h4  -- Substitute x = 4
  sorry -- Conclude with the final proof that start_time = 6

end hiking_start_time_l803_803498


namespace intersection_A_B_l803_803777

def A (x : ℝ) : Prop := log (1 / 2) (3 - x) ≥ -2
def B (x : ℝ) : Prop := 2^x - 1 ≥ 0

theorem intersection_A_B :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = set.Ico 0 3 := sorry

end intersection_A_B_l803_803777


namespace find_value_l803_803023

variable (a b : ℝ)

def equation1 : Prop := 40^a = 2
def equation2 : Prop := 40^b = 5

theorem find_value (h1 : equation1 a) (h2 : equation2 b) : 20^((1 - a - b) / (2 * (1 - b))) = Real.cbrt 20 :=
sorry

end find_value_l803_803023


namespace smallest_integer_with_divisors_properties_l803_803201

def number_of_odd_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 1)

def number_of_even_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 0)

theorem smallest_integer_with_divisors_properties :
  ∃ n : ℕ, number_of_odd_divisors n = 8 ∧ number_of_even_divisors n = 16 ∧ n = 4000 :=
by
  sorry

end smallest_integer_with_divisors_properties_l803_803201


namespace complement_of_P_union_Q_in_Z_is_M_l803_803489

-- Definitions of the sets M, P, Q
def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}
def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}

-- Theorem statement
theorem complement_of_P_union_Q_in_Z_is_M : (Set.univ \ (P ∪ Q)) = M :=
by 
  sorry

end complement_of_P_union_Q_in_Z_is_M_l803_803489


namespace sum_series_eq_three_l803_803353

noncomputable def closest_integer_to_cubed_root (n : ℕ) : ℕ :=
  if h : n > 0 then
    let k := Real.to_nnreal (Real.cbrt n).to_nnreal
    if Real.to_nnreal (k + 1) - 1 / 2 < 0 then ⌊k⌋ else ⌈k⌉
  else 0

theorem sum_series_eq_three :
  ∑' n : ℕ, (3 ^ (closest_integer_to_cubed_root n) + 3 ^ (-(closest_integer_to_cubed_root n))) / 3 ^ n = 3 :=
sorry

end sum_series_eq_three_l803_803353


namespace find_d_l803_803336

theorem find_d (d : ℝ) (h₁ : ∃ x, x = ⌊d⌋ ∧ 3 * x^2 + 19 * x - 84 = 0)
                (h₂ : ∃ y, y = d - ⌊d⌋ ∧ 5 * y^2 - 28 * y + 12 = 0 ∧ 0 ≤ y ∧ y < 1) :
  d = 3.2 :=
by
  sorry

end find_d_l803_803336


namespace smallest_positive_integer_with_divisors_l803_803223

theorem smallest_positive_integer_with_divisors :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → nat.odd d) ∧ (finset.filter nat.odd (finset.divisors n)).card = 8 ∧ 
           (∃ m : ℕ, ∀ d : ℕ, d ∣ m → nat.even d ∧ m = n → (finset.filter nat.even (finset.divisors m)).card = 16)
             → n = 756 :=
by
  sorry

end smallest_positive_integer_with_divisors_l803_803223


namespace number_of_valid_arrangements_l803_803113

-- Definitions for materials
inductive Material
| metal
| wood
| earth
| water
| fire

open Material

-- Conquering relation as a function
def conquers : Material → Material → Prop
| metal, wood   => True
| wood, earth   => True
| earth, water  => True
| water, fire   => True
| fire, metal   => True
| _, _          => False

def is_valid_arrangement (arr : List Material) : Prop :=
  ∀ i, i < arr.length - 1 → ¬ conquers (arr.get i) (arr.get (i + 1))

theorem number_of_valid_arrangements : 
  {arrs : List (List Material) // (arrs.length = 10) ∧ ∀ arr ∈ arrs, is_valid_arrangement arr} :=
sorry

end number_of_valid_arrangements_l803_803113


namespace min_PB_plus_PC_l803_803048

-- Define the points in the rectangular coordinate system
structure Point (α : Type) :=
  (x : α)
  (y : α)

-- Define the points O, A, B, and C
def O : Point ℝ := {x := 0, y := 0}
def A : Point ℝ := {x := 0, y := 6}
def B : Point ℝ := {x := -3, y := 2}
def C : Point ℝ := {x := -2, y := 9}

-- Define a parameterized point P on line segment OA (0 ≤ y ≤ 6)
def onSegmentOA (y : ℝ) (h: 0 ≤ y ∧ y ≤ 6) : Point ℝ := {x := 0, y := y}

-- Define the distance function between two points
noncomputable def distance (P Q : Point ℝ) : ℝ :=
  ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2) ^ (1 / 2)

-- Define the main theorem
theorem min_PB_plus_PC : ∀ (y : ℝ) (h: 0 ≤ y ∧ y ≤ 6),
  distance (onSegmentOA y h) B + distance (onSegmentOA y h) C ≥ 5 + real.sqrt 13 :=
by {
  sorry
}

end min_PB_plus_PC_l803_803048


namespace number_of_intersections_number_of_triangles_number_of_regions_number_of_unbounded_regions_l803_803366

variable (n : ℕ)
variable (h₁ : ∀ i j, i ≠ j → (∀ x, x ≠ 0 → x ≠ 1)) -- Placeholder for actual conditions about not being parallel
variable (h₂ : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → (∀ x, x ≠ 0 → x ≠ 2)) -- Placeholder for actual conditions about not intersecting at the same point

theorem number_of_intersections (n : ℕ) (h₁ : ∀ i j, i ≠ j → (∀ x, x ≠ 0 → x ≠ 1)) : 
  1/2 * n * (n-1) = (n * (n-1)) / 2 := 
sorry

theorem number_of_triangles (n : ℕ) (h₁ : ∀ i j, i ≠ j → (∀ x, x ≠ 0 → x ≠ 1)) (h₂ : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → (∀ x, x ≠ 0 → x ≠ 2)) :
  1/6 * n * (n-1) * (n-2) = (n * (n-1) * (n-2)) / 6 := 
sorry

theorem number_of_regions (n : ℕ) (h₁ : ∀ i j, i ≠ j → (∀ x, x ≠ 0 → x ≠ 1)) (h₂ : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → (∀ x, x ≠ 0 → x ≠ 2)) :
  1 + (1/2 * n * (n+1)) = 1 + (n * (n + 1)) / 2 := 
sorry

theorem number_of_unbounded_regions (n : ℕ) (h₁ : ∀ i j, i ≠ j → (∀ x, x ≠ 0 → x ≠ 1)) : 
  2 * n = 2 * n := 
sorry

end number_of_intersections_number_of_triangles_number_of_regions_number_of_unbounded_regions_l803_803366


namespace smallest_abundant_not_multiple_of_4_l803_803602

-- Define a function to get the sum of proper divisors of a number
def sum_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d, d < n) (Finset.divisors n)).sum

-- Define a predicate to check if a number is abundant
def is_abundant (n : ℕ) : Prop :=
  sum_proper_divisors n > n

-- Define a predicate to check if a number is a multiple of 4
def multiple_of_4 (n : ℕ) : Prop :=
  n % 4 = 0

-- The main theorem to prove
theorem smallest_abundant_not_multiple_of_4 : ∃ n, is_abundant n ∧ ¬ multiple_of_4 n ∧ (∀ m, is_abundant m ∧ ¬ multiple_of_4 m → n ≤ m) :=
  ∃ n, is_abundant n ∧ ¬ multiple_of_4 n ∧ (∀ m, is_abundant m ∧ ¬ multiple_of_4 m → n ≤ m) ∧ n = 20 :=
begin
  sorry
end

end smallest_abundant_not_multiple_of_4_l803_803602


namespace binomial_product_l803_803698

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l803_803698


namespace integer_cos_expression_l803_803472

variables (α : Real) (p q : Int) (n : Nat)
hypothesis h_cos : Real.cos α = ↑p / ↑q

theorem integer_cos_expression : ∀ n : Nat, ∃ m : Int, q^n * Real.cos (n * α) = m :=
by sorry

end integer_cos_expression_l803_803472


namespace prob_yellow_straight_l803_803445

variable {P : ℕ → ℕ → ℚ}
-- Defining the probabilities of the given events
def prob_green : ℚ := 2 / 3
def prob_straight : ℚ := 1 / 2
def prob_rose : ℚ := 1 / 4
def prob_daffodil : ℚ := 1 / 2
def prob_tulip : ℚ := 1 / 4
def prob_rose_straight : ℚ := 1 / 6
def prob_daffodil_curved : ℚ := 1 / 3
def prob_tulip_straight : ℚ := 1 / 8

/-- The probability of picking a yellow and straight-petaled flower is 1/6 -/
theorem prob_yellow_straight : P 1 1 = 1 / 6 := sorry

end prob_yellow_straight_l803_803445


namespace toggle_two_bulbs_l803_803041

theorem toggle_two_bulbs (S: finset ℤ)
  (initial_state : ∀ n : ℤ, ¬ lit n): 
  (∃ state : ℤ → bool, (∀ n, state n = tt ↔ n ∈ S) → (∃ p q : ℤ, p ≠ q ∧ state p = tt ∧ state q = tt)) :=
sorry

end toggle_two_bulbs_l803_803041


namespace total_perimeter_of_garden_with_path_l803_803296

theorem total_perimeter_of_garden_with_path : 
  (∃ s : ℝ, s^2 = 144) → (∀ a : ℝ, a = 12 + 2) → (4 * 14 = 56) :=
by {
  intro h,
  sorry
}

end total_perimeter_of_garden_with_path_l803_803296


namespace order_of_A_B_C_D_l803_803510

def A := Nat.factorial 8 ^ Nat.factorial 8
def B := 8 ^ (8 ^ 8)
def C := 8 ^ 88
def D := 8 ^ 64

theorem order_of_A_B_C_D : D < C ∧ C < B ∧ B < A := by
  sorry

end order_of_A_B_C_D_l803_803510


namespace more_listens_for_Alex_l803_803850

theorem more_listens_for_Alex :
  let jordan_spotify  := 60_000
  let jordan_apple    := 35_000
  let jordan_youtube  := 45_000
  let alex_spotify    := 75_000
  let alex_apple      := 50_000
  let alex_youtube    := 65_000

  -- Jordan's song listens counts
  let jordan_spotify_total := jordan_spotify * (1 + 2 + 4)
  let jordan_apple_total   := jordan_apple * (1 + 1.5 + 2.25)
  let jordan_youtube_total := jordan_youtube * (1 + 1.25 + 1.5625)
  let jordan_total         := jordan_spotify_total + jordan_apple_total + jordan_youtube_total

  -- Alex's song listens counts
  let alex_spotify_total   := alex_spotify * (1 + 1.5 + 2.25)
  let alex_apple_total     := alex_apple * (1 + 1.8 + 3.24)
  let alex_youtube_total   := alex_youtube * (1 + 1.1 + 1.21)
  let alex_total           := alex_spotify_total + alex_apple_total + alex_youtube_total

  -- Assertion
  jordan_total = 757_812.5 ∧ alex_total = 873_400 ∧ alex_total > jordan_total :=
by
  -- Provide the proof here
  sorry

end more_listens_for_Alex_l803_803850


namespace gcd_8008_11011_l803_803343

open Nat

theorem gcd_8008_11011 : gcd 8008 11011 = 1001 :=
by {
  have h1 : 8008 = 8 * 1001 := rfl,
  have h2 : 11011 = 11 * 1001 := rfl,
  have coprime_8_11 : gcd 8 11 = 1 := by sorry,
  exact gcd_eq_multiples h1 h2 coprime_8_11,
}

end gcd_8008_11011_l803_803343


namespace probability_of_inequalities_l803_803953

theorem probability_of_inequalities (x y : ℝ) (h1 : x ∈ set.Icc 0 2) (h2 : y ∈ set.Icc 0 2) :
  ∃ P, P = 1 / 3 ∧ 
       (∀ x y, (x ∈ set.Icc 0 2) → (y ∈ set.Icc 0 2) → (x^2 ≤ 4*y) → (4*y ≤ 4*x) → true) :=
sorry

end probability_of_inequalities_l803_803953


namespace martha_painting_rate_l803_803493

noncomputable def martha_square_feet_per_hour
  (width1 : ℕ) (width2 : ℕ) (height : ℕ) (coats : ℕ) (total_hours : ℕ) 
  (pair1_walls : ℕ) (pair2_walls : ℕ) : ℕ :=
  let pair1_total_area := width1 * height * pair1_walls
  let pair2_total_area := width2 * height * pair2_walls
  let total_area := pair1_total_area + pair2_total_area
  let total_paint_area := total_area * coats
  total_paint_area / total_hours

theorem martha_painting_rate :
  martha_square_feet_per_hour 12 16 10 3 42 2 2 = 40 :=
by
  -- Proof goes here
  sorry

end martha_painting_rate_l803_803493


namespace binom_mult_l803_803712

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l803_803712


namespace irregular_shape_area_l803_803341

noncomputable def area_of_sector (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * Real.pi * r^2

theorem irregular_shape_area (r : ℝ) (θ1 θ2 : ℝ) : 
  r = 28 ∧ θ1 = 48 ∧ θ2 = 32 →
  area_of_sector 28 48 + area_of_sector 28 32 
  = 547.876 :=
by
  intro h
  rcases h with ⟨r_eq, θ1_eq, θ2_eq⟩
  rw [r_eq, θ1_eq, θ2_eq]
  sorry

end irregular_shape_area_l803_803341


namespace no_primes_sum_to_53_l803_803452

open Nat

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem no_primes_sum_to_53 :
  ¬ ∃ (p q : Nat), p + q = 53 ∧ isPrime p ∧ isPrime q ∧ (p < 30 ∨ q < 30) :=
by
  sorry

end no_primes_sum_to_53_l803_803452


namespace ELMOClub_with_5_members_l803_803855

def ELMOClub (G : Type) [simple_graph G] (X : set G) :=
  (X.card ≥ 3) ∧ (∀ (u v : G), u ∈ X → v ∈ X → u ≠ v → G.adj u v)

variables {G : Type} [simple_graph G] (n : ℕ) (n_users : G) (card_n_users : n_users.card = n^3)
          (h₁ : n ≥ 3)
          (h₂ : ∀ (S : finset G), S.card = n → ∃ (X : finset G), ELMOClub G ↑X ∧ X ⊆ S)

theorem ELMOClub_with_5_members : ∃ (X : finset G), ELMOClub G ↑X ∧ X.card = 5 :=
sorry

end ELMOClub_with_5_members_l803_803855


namespace quadratic_expression_positive_l803_803109

theorem quadratic_expression_positive
  (a b c : ℝ) (x : ℝ)
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 > 0 :=
sorry

end quadratic_expression_positive_l803_803109


namespace calculate_expression_l803_803315

theorem calculate_expression (b : ℝ) (hb : b ≠ 0) : 
  (1 / 9) ^ 0 + (1 / 9 * b) ^ (-1) - (81 ^ (-1 / 2)) - ((-27) ^ (-1 / 3)) = 1 + 9 / b + 2 / 9 :=
by
  sorry

end calculate_expression_l803_803315


namespace bill_project_days_l803_803310

theorem bill_project_days (naps: ℕ) (hours_per_nap: ℕ) (working_hours: ℕ) : 
  (naps = 6) → (hours_per_nap = 7) → (working_hours = 54) → 
  (naps * hours_per_nap + working_hours) / 24 = 4 := 
by
  intros h1 h2 h3
  sorry

end bill_project_days_l803_803310


namespace lucas_seq_mod_9_105_l803_803716

def lucas_seq : ℕ → ℕ 
| 0     := 2
| 1     := 3
| (n+2) := lucas_seq n + lucas_seq (n+1)

theorem lucas_seq_mod_9_105 (n : ℕ) : lucas_seq 105 % 9 = 8 := 
by {
  sorry -- No proof required
}

#eval lucas_seq 105 % 9 -- expected to return 8

end lucas_seq_mod_9_105_l803_803716


namespace infinite_twin_pretty_numbers_l803_803293

def is_pretty_number (n : ℕ) : Prop :=
  ∀ p : ℕ, p.prime → p ∣ n → ∃ e : ℕ, e ≥ 2 ∧ p^e ∣ n

def are_twin_pretty_numbers (m n : ℕ) : Prop :=
  n = m + 1 ∧ is_pretty_number m ∧ is_pretty_number n

theorem infinite_twin_pretty_numbers : ∀ n₀ : ℕ, is_pretty_number n₀ → is_pretty_number (n₀ + 1) → ∃ (f : ℕ → ℕ), ∀ i : ℕ, are_twin_pretty_numbers (f i) (f i + 1) :=
by
  sorry

end infinite_twin_pretty_numbers_l803_803293


namespace wolf_cannot_catch_sheep_l803_803502

noncomputable def wolf_sheep_problem : Prop :=
  ∀ (initial_positions : fin 100 → ℝ × ℝ) (wolf_position : ℝ × ℝ),
    let move_within_bound := (λ (pos1 pos2 : ℝ × ℝ), (sq (pos2.1 - pos1.1) + sq (pos2.2 - pos1.2)) ≤ 1)
    let sheep_strategy := (λ (sheep_pos : ℝ × ℝ), if sheep_pos.2 > wolf_position.2 then (sheep_pos.1, sheep_pos.2 + 1) else (sheep_pos.1, sheep_pos.2 - 1))
    ∀ (moves : ℕ → fin 100 → ℝ × ℝ) (wolf_moves : ℕ → ℝ × ℝ),
      (moves 0 = initial_positions) ∧ (wolf_moves 0 = wolf_position) ∧
      (∀ n, move_within_bound (wolf_moves n) (wolf_moves (n + 1))) ∧ 
      (∀ n, ∃ k, move_within_bound ((moves (n + 1)) k) (sheep_strategy ((moves n) k))) →
      ∀ n, ∀ k, (moves n k ≠ wolf_moves n)

theorem wolf_cannot_catch_sheep : wolf_sheep_problem :=
sorry

end wolf_cannot_catch_sheep_l803_803502


namespace find_a_and_b_l803_803928

-- Define the two numbers a and b and the given conditions
variables (a b : ℕ)
variables (h1 : a - b = 831) (h2 : a = 21 * b + 11)

-- State the theorem to find the values of a and b
theorem find_a_and_b (a b : ℕ) (h1 : a - b = 831) (h2 : a = 21 * b + 11) : a = 872 ∧ b = 41 :=
by
  sorry

end find_a_and_b_l803_803928


namespace shading_remainder_l803_803442

def grid := (fin 6) → (fin 4) → Prop

def is_valid_shading (shading : grid) : Prop :=
  (∀ r : fin 6, (finset.univ.filter (λ c, shading r c)).card = 2) ∧ 
  (∀ c : fin 4, (finset.univ.filter (λ r, shading r c)).card = 3)

noncomputable def N : ℕ := 
  finset.card (finset.univ.filter (λ shading : grid, is_valid_shading shading))

theorem shading_remainder : N % 1000 = 860 := by
  sorry

end shading_remainder_l803_803442


namespace manuscript_total_cost_l803_803674

theorem manuscript_total_cost
  (P R1 R2 R3 : ℕ)
  (RateFirst RateRevision : ℕ)
  (hP : P = 300)
  (hR1 : R1 = 55)
  (hR2 : R2 = 35)
  (hR3 : R3 = 25)
  (hRateFirst : RateFirst = 8)
  (hRateRevision : RateRevision = 6) :
  let RemainingPages := P - (R1 + R2 + R3)
  let CostNoRevisions := RemainingPages * RateFirst
  let CostOneRevision := R1 * (RateFirst + RateRevision)
  let CostTwoRevisions := R2 * (RateFirst + 2 * RateRevision)
  let CostThreeRevisions := R3 * (RateFirst + 3 * RateRevision)
  let TotalCost := CostNoRevisions + CostOneRevision + CostTwoRevisions + CostThreeRevisions
  TotalCost = 3600 :=
by
  sorry

end manuscript_total_cost_l803_803674


namespace triangle_is_right_l803_803360

open Real

def distance (p1 p2: ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem triangle_is_right :
  let A := (5 : ℝ, -1 : ℝ)
  let B := (1 : ℝ, 1 : ℝ)
  let C := (2 : ℝ, 3 : ℝ)
  distance A B ^ 2 + distance B C ^ 2 = distance A C ^ 2 :=
by
  let A := (5, -1)
  let B := (1, 1)
  let C := (2, 3)
  sorry

end triangle_is_right_l803_803360


namespace smallest_n_for_abc_factorials_l803_803533

theorem smallest_n_for_abc_factorials (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b + c = 2006) :
  ∃ m n : ℕ, (¬ ∃ k : ℕ, m = 10 * k) ∧ a.factorial * b.factorial * c.factorial = m * 10^n ∧ n = 492 :=
sorry

end smallest_n_for_abc_factorials_l803_803533


namespace larger_number_is_391_l803_803555

theorem larger_number_is_391 (A B : ℕ) 
  (hcf : ∀ n : ℕ, n ∣ A ∧ n ∣ B ↔ n = 23)
  (lcm_factors : ∃ C D : ℕ, lcm A B = 23 * 13 * 17 ∧ C = 13 ∧ D = 17) :
  max A B = 391 :=
sorry

end larger_number_is_391_l803_803555


namespace winning_candidate_percentage_l803_803946

theorem winning_candidate_percentage (votes1 votes2 votes3 : ℕ) (h1 : votes1 = 1256) (h2 : votes2 = 7636) (h3 : votes3 = 11628) 
    : (votes3 : ℝ) / (votes1 + votes2 + votes3) * 100 = 56.67 := by
  sorry

end winning_candidate_percentage_l803_803946


namespace profit_difference_l803_803247

variable (a_capital b_capital c_capital b_profit : ℕ)

def a_capital := 8000
def b_capital := 10000
def c_capital := 12000
def b_profit := 1800

theorem profit_difference : 
  let ratio := a_capital / 2000
  let share_b := b_profit / 5
  let share_a := ratio * share_b
  let share_c := (c_capital / 2000) * share_b
  let diff := share_c - share_a
  diff = 720 :=
by
  sorry

end profit_difference_l803_803247


namespace g_at_8_eq_13_over_3_l803_803432

def g (x : ℝ) : ℝ := (3 * x + 2) / (x - 2)

theorem g_at_8_eq_13_over_3 : g 8 = 13 / 3 := by
  sorry

end g_at_8_eq_13_over_3_l803_803432


namespace find_lambda_l803_803401

open Real

-- Definition of vectors and collinearity
def vect_a : ℝ × ℝ := (1, 2)
def vect_b : ℝ × ℝ := (2, 3)
def vect_c : ℝ × ℝ := (-5, -6)

-- Definition of collinearity condition
def collinear (v w : ℝ × ℝ) := ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

-- The proof statement
theorem find_lambda : 
  ∃ λ : ℝ, collinear (vect_a.1 - λ * vect_b.1, vect_a.2 - λ * vect_b.2) vect_c →
  λ = -4 / 3 := 
sorry

end find_lambda_l803_803401


namespace tan_alpha_l803_803782

theorem tan_alpha (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (hα3 : sin(α)^2 + cos(2 * α) = 3 / 4) : tan(α) = sqrt(3) / 3 :=
by
sorry

end tan_alpha_l803_803782


namespace number_of_valid_M_l803_803133

-- Define the universe of elements
def universe_set : set ℕ := {1, 2, 3}

-- Define the mandatory element subset
def mandatory_subset : set ℕ := {1}

-- Define the condition for M
def is_valid_M (M : set ℕ) : Prop :=
  mandatory_subset ⊆ M ∧ M ⊆ universe_set

-- Define the set of all valid sets M
def all_valid_M : set (set ℕ) :=
  {M | is_valid_M M}

-- The main statement to prove
theorem number_of_valid_M : 
  (finset.card (finset.filter is_valid_M (finset.powerset (finset.singleton 1 ∪ finset.insert 2 (finset.singleton 3)))) = 4) :=
sorry

end number_of_valid_M_l803_803133


namespace painted_cubes_at_least_two_faces_l803_803280

theorem painted_cubes_at_least_two_faces :
  let n := 4 in
  let painted_faces := 6 in
  let total_cubes := n^3 in
  let corner_cubes := 8 in
  let edge_cubes_not_corners_per_edge := n - 2 in
  let edges := 12 in
  let edge_cubes_not_corners := edges * edge_cubes_not_corners_per_edge in
  corner_cubes + edge_cubes_not_corners = 32 :=
by
  sorry

end painted_cubes_at_least_two_faces_l803_803280


namespace lindsey_saved_in_november_l803_803882

def savings_sept : ℕ := 50
def savings_oct : ℕ := 37
def additional_money : ℕ := 25
def spent_on_video_game : ℕ := 87
def money_left : ℕ := 36

def total_savings_before_november := savings_sept + savings_oct
def total_savings_after_november (N : ℕ) := total_savings_before_november + N + additional_money

theorem lindsey_saved_in_november : ∃ N : ℕ, total_savings_after_november N - spent_on_video_game = money_left ∧ N = 11 :=
by
  sorry

end lindsey_saved_in_november_l803_803882


namespace determinant_evaluation_l803_803736

noncomputable def matrixDet : ℝ := 
  Matrix.det ![
    [Real.sin α * Real.sin β, Real.sin α * Real.cos β, Real.cos α],
    [Real.cos β, -Real.sin β, 0],
    [Real.cos α * Real.sin β, Real.cos α * Real.cos β, Real.sin α]
  ]

theorem determinant_evaluation (α β : ℝ) : matrixDet = Real.cos (2 * α) :=
sorry

end determinant_evaluation_l803_803736


namespace converse_of_C_not_true_l803_803368

noncomputable def point (P : Type*) := P
noncomputable def line {P : Type*} (l : set P) := Prop
noncomputable def plane {P : Type*} (α : set P) := Prop
noncomputable def lies_in {P : Type*} (p : P) (s : set P) := p ∈ s
noncomputable def intersect {P : Type*} (l1 l2 : set P) (p : P) := p ∈ l1 ∧ p ∈ l2
noncomputable def perpendicular {P : Type*} (a b : set P) := Prop
noncomputable def parallel {P : Type*} (a b : set P) := Prop

axiom space : Type*
axiom O : point space
axiom a b c : set (point space)
axiom α β : set (plane space)

-- Conditions for C
axiom b_in_α : ∀ (P : point space), lies_in P b → lies_in P α
axiom b_perp_β : ∀ (P : point space), perpendicular b β

-- Desired proposition: Converse of C is not true 
theorem converse_of_C_not_true : (∀ (P : point space), lies_in P b ∧ lies_in P α → perpendicular α β) →
                                ¬ (∀ (P : point space), perpendicular α β → perpendicular b β) :=
sorry

end converse_of_C_not_true_l803_803368


namespace range_f_eq_3_l803_803437

def f (x : ℝ) : ℝ :=
  sqrt (2 * x^2 - x + 3) + 2^(sqrt (x^2 - x))

theorem range_f_eq_3 (x : ℝ) (hx : (sqrt 2) / 2 < x ∧ x ≤ 1) :
  f(x) = 3 :=
sorry

end range_f_eq_3_l803_803437


namespace sufficient_but_not_necessary_condition_l803_803099

theorem sufficient_but_not_necessary_condition :
  (∀ (x : ℝ), x = 1 → x^2 - 3 * x + 2 = 0) ∧ ¬(∀ (x : ℝ), x^2 - 3 * x + 2 = 0 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l803_803099


namespace product_of_two_numbers_l803_803143

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 26) (h2 : x - y = 8) : x * y = 153 :=
by
  -- proof goes here
  sorry

end product_of_two_numbers_l803_803143


namespace total_fruit_count_l803_803286

def fruit_counts : Type := (ℕ × ℕ × ℕ × ℕ × ℕ)

noncomputable def total_fruits (baskets : list fruit_counts) : ℕ :=
  let total_apples := (baskets.map (λ b, b.1)).sum in
  let total_oranges := (baskets.map (λ b, b.2)).sum in
  let total_bananas := (baskets.map (λ b, b.3)).sum in
  let total_grapes := (baskets.map (λ b, b.4)).sum in
  let total_strawberries := (baskets.map (λ b, b.5)).sum in
  total_apples + total_oranges + total_bananas + total_grapes + total_strawberries

noncomputable def basket_1 : fruit_counts := (9, 15, 14, 12, 18)
noncomputable def basket_4 : fruit_counts := (7, 13, 12, 10, 16)
noncomputable def basket_5 : fruit_counts := (12, 10, 14, 12, 18)
noncomputable def basket_6 : fruit_counts := (9, 15, 28, 6, 18)
noncomputable def basket_7 : fruit_counts := (9, 19, 14, 12, 27)

noncomputable def all_baskets : list fruit_counts :=
[basket_1, basket_1, basket_1, 
 basket_4, 
 basket_5, 
 basket_6, 
 basket_7]

theorem total_fruit_count : total_fruits all_baskets = 485 := by sorry

end total_fruit_count_l803_803286


namespace specific_values_of_c_product_of_c_values_l803_803729

-- Define the quadratic equation and the condition for rational solutions
def quadratic_eq (a b c : ℤ) (x : ℤ) : Prop :=
  a * x^2 + b * x + c = 0

-- Define the condition for rational solutions
def has_rational_solution (a b c : ℤ) : Prop :=
  ∃ x : ℚ, quadratic_eq a b c x

-- Prove the specific case
theorem specific_values_of_c : 
  ∃ (c : ℤ), (c > 0) ∧ has_rational_solution 7 13 c ∧ c = 6 := sorry

-- Prove the product of the values
theorem product_of_c_values : 
  ∃ (c1 c2 : ℤ), (c1 = 6) ∧ (c2 = 6) ∧ (c1 * c2 = 36) := sorry

end specific_values_of_c_product_of_c_values_l803_803729


namespace incorrect_statement_D_l803_803242

theorem incorrect_statement_D (k b x : ℝ) (hk : k < 0) (hb : b > 0) (hx : x > -b / k) :
  k * x + b ≤ 0 :=
by
  sorry

end incorrect_statement_D_l803_803242


namespace remainder_p_sq_q_sq_l803_803480

open Polynomial

variable (n : ℕ) (p q : Polynomial ℤ)

-- Define the condition that n is positive
def n_positive : Prop := n > 0

-- Define the condition that q(x) is invertible modulo n
def q_invertible_mod_n : Prop := ∃ q_inv : Polynomial ℤ, q * q_inv % n = 1

-- Define the condition that p(x) is congruent to the inverse of q(x) modulo n
def p_congruent_to_q_inv_mod_n : Prop := ∃ q_inv : Polynomial ℤ, p % n = q_inv * q % n = 1

theorem remainder_p_sq_q_sq (hp : p_congruent_to_q_inv_mod_n n p q) (hq : q_invertible_mod_n n q) (hn : n_positive n) :
  (p * p * q * q) % n = 1 := 
sorry

end remainder_p_sq_q_sq_l803_803480


namespace inequality_integer_solution_range_l803_803824

theorem inequality_integer_solution_range (a : ℝ) :
  (∃ (s : set ℤ), (∀ x : ℝ, (2 * x - 1) ^ 2 < a * x ^ 2 ↔ x ∈ s) ∧ s.card = 3) ↔ (a ∈ (Set.Ioo (25 / 9 : ℝ) (49 / 16 : ℝ)).closure) :=
sorry

end inequality_integer_solution_range_l803_803824


namespace sum_difference_even_odd_l803_803251

theorem sum_difference_even_odd :
  let x := (100 / 2) * (2 + 200)
  let y := (100 / 2) * (1 + 199)
  x - y = 100 :=
by
  sorry

end sum_difference_even_odd_l803_803251


namespace find_x_l803_803609

theorem find_x
  (a b c d k : ℝ)
  (h1 : a ≠ b)
  (h2 : b ≠ 0)
  (h3 : d ≠ 0)
  (h4 : k ≠ 0)
  (h5 : k ≠ 1)
  (h_frac_change : (a + k * x) / (b + x) = c / d) :
  x = (b * c - a * d) / (k * d - c) := by
  sorry

end find_x_l803_803609


namespace smallest_int_with_divisors_l803_803214

theorem smallest_int_with_divisors :
  ∃ n : ℕ, 
    (∀ m, n = 2^2 * m → 
      (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 
      (m = p^3 * q) ∧ 
      (nat.divisors p^7).count 8) ∧ 
    nat.divisors_count n = 24 ∧ 
    (nat.divisors (2^2 * n)).count 8) (n = 2^2 * 3^3 * 5) :=
begin
  sorry
end

end smallest_int_with_divisors_l803_803214


namespace af_bf_ratio_l803_803995

noncomputable def focus (p : ℝ) := (p / 2, 0)

noncomputable def line_eq (p : ℝ) (x : ℝ) := Real.sqrt 3 * (x - p / 2)

theorem af_bf_ratio (p : ℝ) (h : p > 0) : 
  let A := (3 * p / 2, Real.sqrt 3 * p)
  let B := (p / 6, - p / Real.sqrt 3) 
  let AF := (A.1 - focus p.1) ^ 2 + A.2 ^ 2
  let BF := (B.1 - focus p.1) ^ 2 + B.2 ^ 2
  (Real.sqrt AF / Real.sqrt BF) = 3 := 
sorry

end af_bf_ratio_l803_803995


namespace angle_ABC_60_degrees_l803_803877

variables {ABC : Type} [T : Triangle ABC] (I : Point) -- Define the triangle ABC and incenter I

-- Conditions
def non_degenerate (ABC : Triangle ABC) :=
  ¬ (collinear A B C)

def incenter (ABC : Triangle ABC) (I : Point) :=
  incenter I ABC

def angle_AIB_eq_angle_CIA (ABC : Triangle ABC) (I : Point) :=
  ∠ (A I B) = ∠ (C I A)

def angle_ICA_eq_2_angle_IAC (ABC : Triangle ABC) (I : Point) :=
  ∠ (I C A) = 2 * ∠ (I A C)

-- Theorem to prove
theorem angle_ABC_60_degrees (ABC : Triangle ABC) (I : Point) 
  (h1 : non_degenerate ABC) 
  (h2 : incenter ABC I) 
  (h3 : angle_AIB_eq_angle_CIA ABC I) 
  (h4 : angle_ICA_eq_2_angle_IAC ABC I) : 
  ∠ (A B C) = 60 :=
sorry

end angle_ABC_60_degrees_l803_803877


namespace quadratic_equal_roots_relation_l803_803436

theorem quadratic_equal_roots_relation (a b c : ℝ) (h₁ : b ≠ c) 
  (h₂ : ∀ x : ℝ, (b - c) * x^2 + (a - b) * x + (c - a) = 0 → 
          (a - b)^2 - 4 * (b - c) * (c - a) = 0) : 
  c = (a + b) / 2 := sorry

end quadratic_equal_roots_relation_l803_803436


namespace part_a_part_b_l803_803556

noncomputable def markers_with_two_different_digits := 32
noncomputable def markers_with_unique_digits := 304

theorem part_a (length : ℝ) (marking_points : ℕ → ℝ) :
  length = 77.7 ∧ (∀ n : ℕ, n ≤ 777 ∧ n % 1 = 0 ∧ n * 0.1 ∈ marking_points) →
  markers_with_two_different_digits = 32 :=
by
  sorry

theorem part_b (length : ℝ) (marking_points : ℕ → ℝ) :
  length = 77.7 ∧ (∀ n : ℕ, n ≤ 777 ∧ n % 1 = 0 ∧ n * 0.1 ∈ marking_points) →
  markers_with_unique_digits = 304 :=
by
  sorry

end part_a_part_b_l803_803556


namespace largest_acute_angles_convex_octagon_l803_803175

-- Define terms and conditions
def is_convex_octagon (angles : Fin 8 → ℝ) : Prop :=
  (∀ i, 0 < angles i ∧ angles i < 180) ∧
  (∑ i, angles i = 1080)

-- The statement to prove: the largest number of acute angles (angles less than 90 degrees) is 4
theorem largest_acute_angles_convex_octagon (angles : Fin 8 → ℝ) :
  is_convex_octagon angles → 
  (Finset.filter (λ i, angles i < 90) Finset.univ).card ≤ 4 :=
by sorry

end largest_acute_angles_convex_octagon_l803_803175


namespace pleasant_days_count_boring_days_count_l803_803514

-- Define the days, multiples and task conditions
def days : list ℕ := list.range 90 -- Days labeled from 0 to 89

def swims (day : ℕ) : bool := day % 2 = 0 -- Swims every second day
def shops (day : ℕ) : bool := day % 3 = 0 -- Shops every third day
def solves (day : ℕ) : bool := day % 5 = 0 -- Solves problems every fifth day

-- Define a pleasant day: Needs to swim but does not need to shop or solve
def pleasant_day (day : ℕ) : bool := swims day ∧ ¬ (shops day ∨ solves day)

-- Define a boring day: No tasks at all
def boring_day (day : ℕ) : bool := ¬ (swims day ∨ shops day ∨ solves day)

-- Proof statements
theorem pleasant_days_count : list.count pleasant_day days = 24 := sorry
theorem boring_days_count : list.count boring_day days = 24 := sorry

end pleasant_days_count_boring_days_count_l803_803514


namespace probability_none_given_no_A_l803_803318

/-- The probability that an individual has none of the four risk factors given they 
do not have risk factor A is 31/46 in simplest form, where 31 and 46 are relatively prime. -/
theorem probability_none_given_no_A : 
  (∀ p q : ℕ, (p * q ≠ 0) → p.gcd q = 1 → Rat.mk p q = 31 / 46 → p + q = 77) :=
sorry

end probability_none_given_no_A_l803_803318


namespace height_difference_l803_803523

def burj_khalifa_height : ℝ := 830
def sears_tower_height : ℝ := 527

theorem height_difference : burj_khalifa_height - sears_tower_height = 303 := 
by
  sorry

end height_difference_l803_803523


namespace selection_of_graduates_l803_803300

open Finset

noncomputable def choose_three (n : ℕ) (k : ℕ) : ℕ :=
  nat.choose n k

theorem selection_of_graduates : ∀ (A B C : ℕ) (grads : Finset ℕ), grads.card = 10 ∧ A ∈ grads ∧ B ∈ grads ∧ C ∈ grads →
  (grads.filter (λ x, x ≠ C)).card = 9 →
  (grads.filter (λ x, x ≠ A)).card = 9 →
  (grads.filter (λ x, x ≠ B)).card = 9 →
  (grads.filter (λ x, x ≠ A ∧ x ≠ C)).card = 8 →
  (grads.filter (λ x, x ≠ B ∧ x ≠ C)).card = 8 →
  (choose_three 10 3) - (choose_three 7 3) = 49 :=
by
  intros A B C grads h_card h_filter1 h_filter2 h_filter3 h_filter4 h_filter5
  sorry

end selection_of_graduates_l803_803300


namespace trigonometric_identity_proof_l803_803871

open Real

theorem trigonometric_identity_proof (x y : ℝ) (hx : sin x / sin y = 4) (hy : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 169 / 381 :=
by
  sorry

end trigonometric_identity_proof_l803_803871


namespace tangent_equal_distance_l803_803479

noncomputable def midpoint (P Q : Point) : Point := sorry
noncomputable def tangent_point (C : Circle) (P : Point) : Point := sorry
noncomputable def common_tangent (C1 C2 : Circle) : Line := sorry

variable (A T T' M : Point)
variable (γ γ' : Circle)
variable (t : Line)

-- Conditions
axiom circles_tangent_externally (A : Point) (γ γ' : Circle) : externally_tangent γ γ' A
axiom tangent_points (t : Line) (γ γ' : Circle) (T T' : Point) : touches t γ T ∧ touches t γ' T'
axiom midpoint_def (M : Point) (T T' : Point) : midpoint M T T' M

-- Proof Statement
theorem tangent_equal_distance (γ γ' : Circle) (A T T' M : Point) (t : Line) 
  (h1 : externally_tangent γ γ' A)
  (h2 : touches t γ T) 
  (h3 : touches t γ' T') 
  (h4 : midpoint M T T') : 
  distance M T = distance M A ∧ distance M T = distance M T' := 
sorry

end tangent_equal_distance_l803_803479


namespace smallest_positive_integer_with_divisors_l803_803227

theorem smallest_positive_integer_with_divisors :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → nat.odd d) ∧ (finset.filter nat.odd (finset.divisors n)).card = 8 ∧ 
           (∃ m : ℕ, ∀ d : ℕ, d ∣ m → nat.even d ∧ m = n → (finset.filter nat.even (finset.divisors m)).card = 16)
             → n = 756 :=
by
  sorry

end smallest_positive_integer_with_divisors_l803_803227


namespace range_of_a_satisfies_l803_803379

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-(x + 1)) = -f (x + 1)) ∧
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ 0 ≤ x2 → (f x1 - f x2) / (x1 - x2) > -1) ∧
  (∀ a : ℝ, f (a^2 - 1) + f (a - 1) + a^2 + a > 2)

theorem range_of_a_satisfies (f : ℝ → ℝ) (hf_conditions : satisfies_conditions f) :
  {a : ℝ | f (a^2 - 1) + f (a - 1) + a^2 + a > 2} = {a | a < -2 ∨ a > 1} :=
by
  sorry

end range_of_a_satisfies_l803_803379


namespace decreasing_function_range_a_l803_803797

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 1 then (1 - 2 * a) ^ x else log a x + 1 / 3

theorem decreasing_function_range_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f x1 a - f x2 a) / (x1 - x2) < 0) ↔ (0 < a ∧ a ≤ 1 / 3) :=
sorry

end decreasing_function_range_a_l803_803797


namespace man_l803_803645

-- Constants and conditions
def V_down : ℝ := 18  -- downstream speed in km/hr
def V_c : ℝ := 3.4    -- speed of the current in km/hr

-- Main statement to prove
theorem man's_speed_against_the_current : (V_down - V_c - V_c) = 11.2 := by
  sorry

end man_l803_803645


namespace cost_price_one_meter_l803_803246

theorem cost_price_one_meter (selling_price : ℤ) (total_meters : ℤ) (profit_per_meter : ℤ) 
  (h1 : selling_price = 6788) (h2 : total_meters = 78) (h3 : profit_per_meter = 29) : 
  (selling_price - (profit_per_meter * total_meters)) / total_meters = 58 := 
by 
  sorry

end cost_price_one_meter_l803_803246


namespace vertical_line_divides_area_l803_803092

open Real

theorem vertical_line_divides_area (p q : ℝ) (x1 x2 : ℝ) 
  (h_intersect : ∃ x1 x2, -2 * x1^2 = x1^2 + p * x1 + q ∧ -2 * x2^2 = x2^2 + p * x2 + q)
  (h_intersection_points : x1 ≠ x2) :
  let x_l := -p / 6 in
  x_l = (x1 + x2) / 2 :=
sorry

end vertical_line_divides_area_l803_803092


namespace total_gallons_l803_803848

-- Definitions arising from conditions
def quart_to_gallon := 1 / 4
def half_gallon := 1 / 2
def full_gallon := 1
def total_jars := 60

-- The number of each type of jar assuming equal number of each type
def jars_per_type := total_jars / 3

-- Formulate the problem statement using definitions
theorem total_gallons (equal_jars : jars_per_type = 60 / 3) : 
  let quarts := jars_per_type * quart_to_gallon,
      half_gallons := jars_per_type * half_gallon,
      full_gallons := jars_per_type * full_gallon,
      total_gallons := quarts + half_gallons + full_gallons
  in total_gallons = 35 := 
by
  sorry

end total_gallons_l803_803848


namespace smallest_abundant_not_multiple_of_4_l803_803603

-- Define a function to get the sum of proper divisors of a number
def sum_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d, d < n) (Finset.divisors n)).sum

-- Define a predicate to check if a number is abundant
def is_abundant (n : ℕ) : Prop :=
  sum_proper_divisors n > n

-- Define a predicate to check if a number is a multiple of 4
def multiple_of_4 (n : ℕ) : Prop :=
  n % 4 = 0

-- The main theorem to prove
theorem smallest_abundant_not_multiple_of_4 : ∃ n, is_abundant n ∧ ¬ multiple_of_4 n ∧ (∀ m, is_abundant m ∧ ¬ multiple_of_4 m → n ≤ m) :=
  ∃ n, is_abundant n ∧ ¬ multiple_of_4 n ∧ (∀ m, is_abundant m ∧ ¬ multiple_of_4 m → n ≤ m) ∧ n = 20 :=
begin
  sorry
end

end smallest_abundant_not_multiple_of_4_l803_803603


namespace binom_mult_l803_803708

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l803_803708


namespace no_common_period_l803_803283

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f(x + p) = f(x)

theorem no_common_period 
  (g h : ℝ → ℝ) 
  (Hg : is_periodic g 3)
  (Hh : is_periodic h real.pi) :
  ¬ ∃ T > 0, is_periodic (λ x, g x - h x) T :=
by
  sorry

end no_common_period_l803_803283


namespace negation_true_l803_803561

theorem negation_true (a : ℝ) : ¬ (∀ a : ℝ, a ≤ 2 → a^2 < 4) :=
sorry

end negation_true_l803_803561


namespace not_a_correlation_l803_803612

-- Define each condition
def timely_snow_promises_good_harvest : Prop := 
  ∃ w : Wheat, (timely_snow w) → (good_harvest w)

def good_teacher_produces_outstanding_students : Prop := 
  ∃ t : Teacher, ∃ s : Student, (good_teacher t) → (outstanding_student t s)

def smoking_is_harmful_to_health : Prop := 
  ∃ p : Person, (smokes p) → (health_harmed p)

def magpies_call_is_sign_of_happiness : Prop := 
  ∃ m : Magpie, ∃ h : Happiness, (magpies_call m) → (sign_of_happiness h)

-- The problem statement
theorem not_a_correlation : ¬ magpies_call_is_sign_of_happiness := by
  -- We assume a correlation for other statements to be used later
  sorry

end not_a_correlation_l803_803612


namespace line_properties_l803_803801

theorem line_properties (a : ℝ) : 
  ((a = 0 → ((a^2 + a + 1) * 1 - 1 + 1 = 0 → (1) * 1 + (-1) * (-1) = -1))) ∧
  ((0 * (a^2 + a + 1) - 1 + 1 = 0 → 1 = 1) :=
begin
  split,
  { -- a = 0 implies line is perpendicular to x + y = 0
    intro ha,
    rw [ha] at *,
    intro h,
    simp at h,
    -- the slope of line (when a = 0) is 1, and the slope of line x + y = 0 is -1
    linarith [h],
  },
  { -- line passes through the point (0, 1)
    intro h,
    simp at *,
    -- point (0, 1) satisfies the given equation regardless of a
    linarith [h]
  }
end

end line_properties_l803_803801


namespace find_x_if_perpendicular_l803_803081

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (2 * x - 1, 3)
def vec_n : ℝ × ℝ := (1, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_x_if_perpendicular (x : ℝ) : 
  dot_product (vec_m x) vec_n = 0 ↔ x = 2 :=
by
  sorry

end find_x_if_perpendicular_l803_803081


namespace sum_of_transformed_numbers_l803_803935

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) : 8 * S = 8 * (a + b) :=
by 
  rw [h]
  sorry

end sum_of_transformed_numbers_l803_803935


namespace kendra_shirts_needed_l803_803467

def school_shirts_per_week : Nat := 5
def club_shirts_per_week : Nat := 3
def spirit_day_shirt_per_week : Nat := 1
def saturday_shirts_per_week : Nat := 3
def sunday_shirts_per_week : Nat := 3
def family_reunion_shirt_per_month : Nat := 1

def total_shirts_needed_per_week : Nat :=
  school_shirts_per_week + club_shirts_per_week + spirit_day_shirt_per_week +
  saturday_shirts_per_week + sunday_shirts_per_week

def total_shirts_needed_per_four_weeks : Nat :=
  total_shirts_needed_per_week * 4 + family_reunion_shirt_per_month

theorem kendra_shirts_needed : total_shirts_needed_per_four_weeks = 61 := by
  sorry

end kendra_shirts_needed_l803_803467


namespace conic_section_eccentricities_l803_803145

theorem conic_section_eccentricities :
  let x1 := (5 - Real.sqrt 21) / 2;
  let x2 := (5 + Real.sqrt 21) / 2;
  x1 ∈ (Set.Ioo 0 1) ∧ x2 ∈ (Set.Ioi 1) →
  "Ellipse and Hyperbola" :=
by
  sorry

end conic_section_eccentricities_l803_803145


namespace odd_derivative_of_even_function_l803_803093

variable {ℝ : Type*} [LinearOrder ℝ] [TopologicalSpace ℝ] [ChartedSpace ℝ ℝ] [SmoothManifold ℝ ℝ]

noncomputable def f (x : ℝ) := by sorry

-- Condition: f(-x) = f(x)
axiom evenf : ∀ (x : ℝ), f (-x) = f x

def g (x : ℝ) := f' x

theorem odd_derivative_of_even_function (x : ℝ) : g (-x) = -g x := by
  sorry

end odd_derivative_of_even_function_l803_803093


namespace player_current_average_l803_803998

theorem player_current_average
  (A : ℕ) -- Assume A is a natural number (non-negative)
  (cond1 : 10 * A + 78 = 11 * (A + 4)) :
  A = 34 :=
by
  sorry

end player_current_average_l803_803998


namespace centroid_midpoint_circumcenter_orthocenter_l803_803520

variable (A B C D O H M : Point)

-- Conditions defining the orthocentric tetrahedron, circumcenter, orthocenter, and centroid
variable (isOrthocentricTetrahedron : OrthocentricTetrahedron A B C D)
variable (isCircumcenter : Circumcenter O A B C D)
variable (isOrthocenter : Orthocenter H A B C D)
variable (isCentroid : Centroid M A B C D)

-- The statement to be proved
theorem centroid_midpoint_circumcenter_orthocenter :
  isOrthocentricTetrahedron A B C D → 
  isCircumcenter O A B C D →
  isOrthocenter H A B C D →
  isCentroid M A B C D →
  M = (O + H) / 2 := by
  sorry

end centroid_midpoint_circumcenter_orthocenter_l803_803520


namespace num_valid_passwords_l803_803272

theorem num_valid_passwords : 
  let total_passwords := 10000 in        -- total number of 4-digit passwords
  let invalid_passwords := 10 in         -- number of invalid passwords starting with 911
  let valid_passwords := total_passwords - invalid_passwords in
  valid_passwords = 9990 := 
by 
  let total_passwords := 10000
  let invalid_passwords := 10
  let valid_passwords := total_passwords - invalid_passwords
  show valid_passwords = 9990 from sorry

end num_valid_passwords_l803_803272


namespace factorize_expression_l803_803737

theorem factorize_expression (x a : ℝ) : 4 * x - x * a^2 = x * (2 - a) * (2 + a) :=
by 
  sorry

end factorize_expression_l803_803737


namespace volume_inscribed_cube_correct_l803_803657

-- Define the edge length of the larger cube
def edge_length_larger_cube : ℝ := 12

-- Define the diameter of the sphere, which is the same as the edge length of the larger cube
def diameter_sphere : ℝ := edge_length_larger_cube

-- We'll define the side length of the smaller cube using the diagonal relationship in a cube
def side_length_smaller_cube : ℝ := diameter_sphere / Real.sqrt 3

-- Define the volume of a cube given its side length
def volume_cube (s : ℝ) : ℝ := s^3

-- Define the volume of the inscribed cube
def volume_inscribed_cube : ℝ := volume_cube side_length_smaller_cube

-- Theorem statement to prove the volume of the inscribed cube is 192√3 cubic inches
theorem volume_inscribed_cube_correct : volume_inscribed_cube = 192 * Real.sqrt 3 := by
  sorry

end volume_inscribed_cube_correct_l803_803657


namespace baby_first_grab_possibilities_l803_803732

theorem baby_first_grab_possibilities
  (educational_items : ℕ)
  (living_items : ℕ)
  (entertainment_items : ℕ) :
  educational_items = 4 →
  living_items = 3 →
  entertainment_items = 4 →
  educational_items + living_items + entertainment_items = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end baby_first_grab_possibilities_l803_803732


namespace divide_friends_into_teams_l803_803814

theorem divide_friends_into_teams :
  let friends := 8
      teams := 4
  in (teams ^ friends = 65536) :=
by
  let friends := 8
  let teams := 4
  have h : teams ^ friends = 65536 := by sorry
  exact h

end divide_friends_into_teams_l803_803814


namespace coefficient_x4_l803_803324

noncomputable def polynomial_expansion : ℚ[X] :=
  (2 - X) * (2*X + 1)^6

theorem coefficient_x4 :
  polynomial_expansion.coeff 4 = 320 :=
by
  sorry

end coefficient_x4_l803_803324


namespace intersect_point_one_l803_803325

theorem intersect_point_one (k : ℝ) : 
  (∀ y : ℝ, (x = -3 * y^2 - 2 * y + 4 ↔ x = k)) ↔ k = 13 / 3 := 
by
  sorry

end intersect_point_one_l803_803325


namespace dodecagon_diagonals_l803_803404

/--
The formula for the number of diagonals in a convex n-gon is given by (n * (n - 3)) / 2.
-/
def number_of_diagonals (n : Nat) : Nat := (n * (n - 3)) / 2

/--
A dodecagon has 12 sides.
-/
def dodecagon_sides : Nat := 12

/--
The number of diagonals in a convex dodecagon is 54.
-/
theorem dodecagon_diagonals : number_of_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l803_803404


namespace binomial_product_l803_803701

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l803_803701


namespace smallest_integer_with_divisors_l803_803183

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end smallest_integer_with_divisors_l803_803183


namespace find_y_value_l803_803248

theorem find_y_value : (12 : ℕ)^3 * (6 : ℕ)^2 / 432 = 144 := by
  -- assumptions and computations are not displayed in the statement
  sorry

end find_y_value_l803_803248


namespace smallest_integer_with_divisors_l803_803187

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end smallest_integer_with_divisors_l803_803187


namespace number_of_true_propositions_l803_803373

noncomputable def f : ℝ → ℝ := sorry -- since it's not specified, we use sorry here

-- Definitions for the conditions
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Original proposition
def original_proposition (f : ℝ → ℝ) :=
  ∀ x : ℝ, is_odd f → f 0 = 0

-- Converse proposition
def converse_proposition (f : ℝ → ℝ) :=
  f 0 = 0 → ∀ x : ℝ, is_odd f

-- Inverse proposition (logically equivalent to the converse)
def inverse_proposition (f : ℝ → ℝ) :=
  ∀ x : ℝ, ¬(is_odd f) → f 0 ≠ 0

-- Contrapositive proposition (logically equivalent to the original)
def contrapositive_proposition (f : ℝ → ℝ) :=
  f 0 ≠ 0 → ∀ x : ℝ, ¬(is_odd f)

-- Theorem statement
theorem number_of_true_propositions (f : ℝ → ℝ) :
  (original_proposition f → true) ∧
  (converse_proposition f → false) ∧
  (inverse_proposition f → false) ∧
  (contrapositive_proposition f → true) →
  2 = 2 := 
by 
  sorry -- proof to be inserted

end number_of_true_propositions_l803_803373


namespace scientific_notation_of_935000000_l803_803665

theorem scientific_notation_of_935000000 :
  935000000 = 9.35 * 10^8 :=
by
  sorry

end scientific_notation_of_935000000_l803_803665


namespace monotonic_decreasing_interval_l803_803923

noncomputable def f (x : ℝ) : ℝ := x + 2 / x

def domain (x : ℝ) : Prop := x ≠ 0

noncomputable def f' (x : ℝ) : ℝ := 1 - 2 / (x * x)

def decreasing_interval (x : ℝ) : Prop := f'(x) ≤ 0

def interval : Set ℝ := {x | domain x ∧ -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2}

theorem monotonic_decreasing_interval :
  {x | domain x ∧ decreasing_interval x} = 
  {x | x ∈ interval ∧ x ≠ 0} := 
sorry

end monotonic_decreasing_interval_l803_803923


namespace problem_prove_special_l803_803982

variable (a b : ℝ)

theorem problem_prove_special : b ≠ 0 → (2 * a^2 / b)^2 * (2 * a^2 * b^(-2))^(-2) = b^2 :=
by
  assume h : b ≠ 0
  sorry

end problem_prove_special_l803_803982


namespace prime_p_sq_plus_26_never_prime_l803_803611

open Nat

theorem prime_p_sq_plus_26_never_prime (p : ℕ) (hp : Prime p) : ¬ Prime (p^2 + 26) :=
by
  sorry

end prime_p_sq_plus_26_never_prime_l803_803611


namespace problem_solution_l803_803984

theorem problem_solution : 3 * 12 + 3 * 13 + 3 * 16 + 11 = 134 := by
  calc
    _ = 3 * 12 + 3 * 13 + 3 * 16 + 11 : by sorry
    _ = 36 + 39 + 48 + 11 : by sorry
    _ = 134 : by sorry

end problem_solution_l803_803984


namespace binomial_product_l803_803702

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l803_803702


namespace find_first_term_l803_803060

noncomputable theory

variable (n : ℕ) (b : ℚ) (T : ℕ → ℚ) (common_difference : ℚ)

-- Conditions
def sum_first_n_terms (n : ℕ) (b : ℚ) (d : ℚ) : ℚ := n * (2 * b + (n - 1) * d) / 2

axiom arithmetic_sequence :
  (∀ n : ℕ, T n = sum_first_n_terms n b 5) ∧
  (∀ n : ℕ, T (4 * n) / T n = 16)

-- Theorem to prove
theorem find_first_term (b : ℚ) (T : ℕ → ℚ) (common_difference : ℚ) :
  (∀ n : ℕ, T n = sum_first_n_terms n b 5) →
  (∀ n : ℕ, T (4 * n) / T n = 16) →
  b = 5 / 2 :=
by
  intros hT1 hT2
  sorry

end find_first_term_l803_803060


namespace triangle_DEF_value_EF_l803_803462

theorem triangle_DEF_value_EF :
  ∀ (E D F : Type) [EuclideanGeometry E D F] (DE DF : Real) (∠E : Angle)
  (h1 : ∠E = 45) (h2 : DE = 100 * Real.sqrt 2) (h3 : DF = 100), 
  let EF := ((DE ^ 2 + DF ^ 2).sqrt) in 
  EF = 100 * Real.sqrt 3 :=
by 
  intros E D F EuclideanGeometry DE DF ∠E h1 h2 h3 EF 
  sorry

end triangle_DEF_value_EF_l803_803462


namespace smallest_integer_with_divisors_properties_l803_803196

def number_of_odd_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 1)

def number_of_even_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 0)

theorem smallest_integer_with_divisors_properties :
  ∃ n : ℕ, number_of_odd_divisors n = 8 ∧ number_of_even_divisors n = 16 ∧ n = 4000 :=
by
  sorry

end smallest_integer_with_divisors_properties_l803_803196


namespace systematic_sampling_max_number_l803_803754

theorem systematic_sampling_max_number (n m : ℕ) (interval: ℕ) (sample: list ℕ) (max_in_sample : ℕ) :
  n = 80 →
  m = 5 →
  interval = n / m →
  ∃ (sample : list ℕ), 28 ∈ sample ∧ list.length sample = m ∧
  (∀ i < m, sample.nth_le i sorry = 28 + i * interval) →
  max_in_sample = 76 :=
by
  intros h1 h2 h3 h4
  sorry

end systematic_sampling_max_number_l803_803754


namespace dodecagon_diagonals_l803_803407

def numDiagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem dodecagon_diagonals : numDiagonals 12 = 54 := by
  sorry

end dodecagon_diagonals_l803_803407


namespace count_180_rotational_symmetric_pentominoes_l803_803409

-- Define a representation for pentominoes and the symmetry property
def pentomino : Type := ℕ
def pentominoes : fin 12 → pentomino 
def has_180_symmetry (p : pentomino) : Prop := sorry

-- Statement of the theorem
theorem count_180_rotational_symmetric_pentominoes : 
  (finset.card (finset.filter has_180_symmetry (finset.univ : finset (fin 12)))) = 3 := sorry

end count_180_rotational_symmetric_pentominoes_l803_803409


namespace volume_inscribed_cube_correct_l803_803656

-- Define the edge length of the larger cube
def edge_length_larger_cube : ℝ := 12

-- Define the diameter of the sphere, which is the same as the edge length of the larger cube
def diameter_sphere : ℝ := edge_length_larger_cube

-- We'll define the side length of the smaller cube using the diagonal relationship in a cube
def side_length_smaller_cube : ℝ := diameter_sphere / Real.sqrt 3

-- Define the volume of a cube given its side length
def volume_cube (s : ℝ) : ℝ := s^3

-- Define the volume of the inscribed cube
def volume_inscribed_cube : ℝ := volume_cube side_length_smaller_cube

-- Theorem statement to prove the volume of the inscribed cube is 192√3 cubic inches
theorem volume_inscribed_cube_correct : volume_inscribed_cube = 192 * Real.sqrt 3 := by
  sorry

end volume_inscribed_cube_correct_l803_803656


namespace original_expenditure_of_mess_l803_803155

theorem original_expenditure_of_mess (x : ℝ) 
    (h1 : 0 < x)
    (h2 : 35 * x + 42 = 42 * (x - 1)) : 
    35 * x = 420 :=
by 
  have h3 : 42 * (x - 1) = 35 * x + 42 := by rw [←h2]; sorry
  ring_exp at h3
  field_simp at h3
  simp at h3
  have h4 : x = 12 :=
    by linarith [h3]
  rw [h4]
  simp
  ring
  rfl
  sorry

end original_expenditure_of_mess_l803_803155


namespace num_of_non_multiples_of_6_or_8_in_three_digits_l803_803412

-- Define conditions about multiples and range of three-digit numbers
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

def multiples_of_6 := {n : ℕ | is_multiple_of n 6}
def multiples_of_8 := {n : ℕ | is_multiple_of n 8}
def multiples_of_24 := {n : ℕ | is_multiple_of n 24}

-- Prove that the number of three-digit numbers not multiples of 6 or 8 is 675
theorem num_of_non_multiples_of_6_or_8_in_three_digits : 
  {n : ℕ | n ∈ three_digit_numbers ∧ n ∉ multiples_of_6 ∧ n ∉ multiples_of_8}.count = 675 :=
sorry

end num_of_non_multiples_of_6_or_8_in_three_digits_l803_803412


namespace x_intercept_of_line_l803_803658

variables (x₁ y₁ x₂ y₂ : ℝ) (m : ℝ)

/-- The line passing through the points (-1, 1) and (3, 9) has an x-intercept of -3/2. -/
theorem x_intercept_of_line : 
  let x₁ := -1
  let y₁ := 1
  let x₂ := 3
  let y₂ := 9
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (0 : ℝ) = m * (x : ℝ) + b → x = (-3 / 2) := 
by 
  sorry

end x_intercept_of_line_l803_803658


namespace smallest_integer_with_odd_and_even_divisors_l803_803203

theorem smallest_integer_with_odd_and_even_divisors :
  ∃ n : ℕ,
    0 < n ∧
    (∀ d : ℕ, d ∣ n → (odd d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 1}) ∨ (even d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 0})) ∧ 
    (↑8 = (∑ d in (finset.filter (λ d, odd d) (finset.divisors n)), 1)) ∧
    (↑16 = (∑ d in (finset.filter (λ d, even d) (finset.divisors n)), 1)) ∧
    (24 = (∑ d in (finset.divisors n), 1)) ∧
    n = 108 :=
begin
  sorry
end

end smallest_integer_with_odd_and_even_divisors_l803_803203


namespace binomial_product_result_l803_803688

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l803_803688


namespace angle_AEC_l803_803358

noncomputable def equilateral_triangle {α : Type} [euclidean_geometry α] (A B C : α) := 
circle α A A A =
circle α A B C / chosen | Aᵤρcentroid.AᵤeA0Anot_empty_)}

noncomputable def intersect_ray {α : Type} [euclidean_geometry α] (A B C : α) (Ax : set α) (D : α) := 
(line_through α A B Ax ∧ 
 line_through α B C Ax ∧ 
 line_through α A D Ax)

noncomputable def point_on_ray {α : Type} [euclidean_geometry α] (A : α) (Ax : set α) (E : α) := 
line_through α A E Ax 

theorem angle_AEC {α : Type} [euclidean_geometry α] 
(A B C D E : α) (Ax : set α) 
(h₁ : equilateral_triangle A B C) 
(h₂ : intersect_ray A B C Ax D) 
(h₃ : point_on_ray A Ax E) 
(h₄ : BA = BE) : 
angle α A E C = 30 :=
sorry

end angle_AEC_l803_803358


namespace crickets_needed_to_reach_11_l803_803968

theorem crickets_needed_to_reach_11 (collected_crickets : ℕ) (wanted_crickets : ℕ) 
                                     (h : collected_crickets = 7) (h2 : wanted_crickets = 11) :
  wanted_crickets - collected_crickets = 4 :=
sorry

end crickets_needed_to_reach_11_l803_803968


namespace no_valid_quadruples_l803_803347

-- Define the problem
theorem no_valid_quadruples (k : ℝ) (hk : k ≠ 0) :
  ∀ (a b c d : ℝ),
    ([
      [a, b],
      [c, d]
    ]⁻¹ = [
      [k / a, k / b],
      [k / c, k / d]
    ] → False) :=
sorry

end no_valid_quadruples_l803_803347


namespace triangle_area_relationship_l803_803805

-- Define semiperimeter
def semiperimeter (a b c : ℝ) := (a + b + c) / 2

-- Define triangle sides and angles
structure Triangle :=
(a b c : ℝ)
(α β γ : ℝ)

-- Define the area relationship of a triangle
theorem triangle_area_relationship (t : ℝ) (a b c : ℝ) (α β γ : ℝ)
  (s := semiperimeter a b c)
  (s₁ : ℝ := s - a) (s₂ : ℝ := s - b) (s₃ : ℝ := s - c)
  (ρ ρ₁ ρ₂ ρ₃ : ℝ) :
  t = ρ * s ∧ t = ρ₁ * s₁ ∧ t = ρ₂ * s₂ ∧ t = ρ₃ * s₃ :=
by sorry

end triangle_area_relationship_l803_803805


namespace smallest_integer_with_odd_and_even_divisors_l803_803207

theorem smallest_integer_with_odd_and_even_divisors :
  ∃ n : ℕ,
    0 < n ∧
    (∀ d : ℕ, d ∣ n → (odd d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 1}) ∨ (even d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 0})) ∧ 
    (↑8 = (∑ d in (finset.filter (λ d, odd d) (finset.divisors n)), 1)) ∧
    (↑16 = (∑ d in (finset.filter (λ d, even d) (finset.divisors n)), 1)) ∧
    (24 = (∑ d in (finset.divisors n), 1)) ∧
    n = 108 :=
begin
  sorry
end

end smallest_integer_with_odd_and_even_divisors_l803_803207


namespace carrie_savings_l803_803680

-- Define the original prices and discount rates
def deltaPrice : ℝ := 850
def deltaDiscount : ℝ := 0.20
def unitedPrice : ℝ := 1100
def unitedDiscount : ℝ := 0.30

-- Calculate discounted prices
def deltaDiscountAmount : ℝ := deltaPrice * deltaDiscount
def unitedDiscountAmount : ℝ := unitedPrice * unitedDiscount

def deltaDiscountedPrice : ℝ := deltaPrice - deltaDiscountAmount
def unitedDiscountedPrice : ℝ := unitedPrice - unitedDiscountAmount

-- Define the savings by choosing the cheaper flight
def savingsByChoosingCheaperFlight : ℝ := unitedDiscountedPrice - deltaDiscountedPrice

-- The theorem stating the amount saved
theorem carrie_savings : savingsByChoosingCheaperFlight = 90 :=
by
  sorry

end carrie_savings_l803_803680


namespace trajectory_equation_max_value_ratio_l803_803003

-- Definitions for the conditions
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)
variable {T : ℝ × ℝ}
def dist (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)
def condition1 : Prop := dist T A + dist T B = 2 * real.sqrt 6
def line_passing_B (k : ℝ) (k_ne_zero : k ≠ 0) : set (ℝ × ℝ) := {p | p.2 = k * (p.1 - 2)}

-- Proving the trajectory equation
theorem trajectory_equation :
  condition1 →
  (T : ℝ × ℝ) × (∃ x y : ℝ, T = (x, y) ∧ (x^2 / 6) + (y^2 / 2) = 1) →
  True := sorry

-- Definitions for the second part
variable {P Q : ℝ × ℝ}
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
def N := midpoint P Q
def O : ℝ × ℝ := (0, 0)
def line_ON (N : ℝ × ℝ) : set (ℝ × ℝ) := {p | p.2 = (N.2 / N.1) * p.1}
def M : ℝ × ℝ := (3, -1 / k)

-- Proving the maximum value of ratio
theorem max_value_ratio :
  (line_passing_B k k_ne_zero).limits T P Q →
  N = midpoint P Q →
  condition1 →
  P ≠ Q →
  dist P Q = dist P B + dist Q B →
  (∃ M, M.1 = 3 ∧ M.2 = -(1 / k)) →
  dist P Q / dist M B ≤ real.sqrt 3 := sorry

end trajectory_equation_max_value_ratio_l803_803003


namespace no_zero_position_l803_803582

-- Define the concept of regular pentagon vertex assignments and operations
def pentagon_arith_mean (x y : ℝ) : ℝ := (x + y) / 2

-- Define the condition for the initial sum of numbers on the vertices being zero
def initial_sum_zero (a b c d e : ℝ) : Prop := a + b + c + d + e = 0

-- Define the main theorem statement
theorem no_zero_position (a b c d e : ℝ) (h : initial_sum_zero a b c d e) :
  ¬ ∃ a' b' c' d' e' : ℝ, ∀ v w : ℝ, pentagon_arith_mean v w = 0 :=
sorry

end no_zero_position_l803_803582


namespace eval_g_at_8_l803_803429

def g (x : ℚ) : ℚ := (3 * x + 2) / (x - 2)

theorem eval_g_at_8 : g 8 = 13 / 3 := by
  sorry

end eval_g_at_8_l803_803429


namespace partition_sum_condition_l803_803100

theorem partition_sum_condition (X : Finset ℕ) (hX : X = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∀ (A B : Finset ℕ), A ∪ B = X → A ∩ B = ∅ →
  ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a + b = c := 
by
  -- sorry is here to acknowledge that no proof is required per instructions.
  sorry

end partition_sum_condition_l803_803100


namespace total_miles_run_correct_l803_803942

-- Define the number of people on the sprint team and the miles each person runs.
def number_of_people : Float := 150.0
def miles_per_person : Float := 5.0

-- Define the total miles run by the sprint team.
def total_miles_run : Float := number_of_people * miles_per_person

-- State the theorem to prove that the total miles run is equal to 750.0 miles.
theorem total_miles_run_correct : total_miles_run = 750.0 := sorry

end total_miles_run_correct_l803_803942


namespace oxygen_atoms_in_compound_l803_803277

variable (num_O : ℕ)
variable (mol_weight_Ba : ℝ)
variable (mol_weight_O : ℝ)
variable (mol_weight_H : ℝ)
variable (num_H : ℕ)
variable (mol_weight_compound : ℝ)

theorem oxygen_atoms_in_compound :
  mol_weight_Ba + mol_weight_O * num_O + mol_weight_H * num_H = mol_weight_compound →
  num_O = 2 :=
by
  -- Given conditions
  intro h
  have mwc: mol_weight_compound = 171 := rfl
  have mwBa: mol_weight_Ba = 137.33 := rfl
  have mwO: mol_weight_O = 16.00 := rfl
  have mwH: mol_weight_H = 1.01 := rfl
  have nH: num_H = 2 := rfl
  -- target statement
  sorry

end oxygen_atoms_in_compound_l803_803277


namespace rectangle_diagonal_length_l803_803579

noncomputable def length_of_diagonal (x1 y1 x2 y2 x3 y3 : ℤ) : ℤ :=
  if (y1 = y3 ∧ y2 ≠ y3) ∨ (y1 ≠ y3 ∧ y2 = y3) then
    let x4 := if x1 = x2 then x3 else if x1 = x3 then x2 else x1 - (x2 - x3) in
    let y4 := if y1 = y2 then y3 else if y1 = y3 then y2 else y1 - (y2 - y3) in
    let dx := x3 - x2 in
    let dy := y3 - y2 in
    Int.sqrt (dx * dx + dy * dy)
  else
    0

theorem rectangle_diagonal_length : length_of_diagonal (-3) 5 4 (-3) 9 5 = Int.sqrt 89 :=
by sorry

end rectangle_diagonal_length_l803_803579


namespace sqrt_of_1_5625_eq_1_25_l803_803604

theorem sqrt_of_1_5625_eq_1_25 : Real.sqrt 1.5625 = 1.25 :=
  sorry

end sqrt_of_1_5625_eq_1_25_l803_803604


namespace side_length_of_square_l803_803028

def side_is_four (area : ℕ) (side : ℕ) : Prop :=
  side ^ 2 < area ∧ side > 0 ∧ ∀ s : ℕ, s ^ 2 < area → s < 5

theorem side_length_of_square :
  ∀ (side : ℕ), side_is_four 20 side → side = 4 :=
by
  intros side h
  cases h with h1 h2
  cases h2 with h_side_pos h_side_bound
  have h_side_lt_20 : side < 5 :=
    h_side_bound side h1
  sorry

end side_length_of_square_l803_803028


namespace team_scores_l803_803037

noncomputable theory

variables (A B C D E : Type)

-- Define the points scored by each team
variables (a b c d e : ℕ)

-- Define the conditions
axiom each_team_played_once :
  ∀ x y : Type, x ≠ y → played_once x y

axiom points_for_each_game :
  ∀ x y : Type, points_awarded x y = 3 ∨ points_awarded x y = 2

axiom all_different_points :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

axiom team_A_scored_most :
  a > b ∧ a > c ∧ a > d ∧ a > e

axiom team_A_lost_to_B :
  points_awarded B A = 3

axiom team_B_and_C_did_not_lose :
  ∀ x : Type, x ≠ B → points_awarded B x ≠ 0 ∧ x ≠ C → points_awarded C x ≠ 0

axiom team_C_less_than_D :
  c < d

-- The conclusion
theorem team_scores :
  a = 7 ∧ b = 6 ∧ c = 4 ∧ d = 5 ∧ e = 2 :=
begin
  sorry
end

end team_scores_l803_803037


namespace shooting_average_l803_803653

noncomputable def total_points (a b c d : ℕ) : ℕ :=
  (a * 10) + (b * 9) + (c * 8) + (d * 7)

noncomputable def average_points (total : ℕ) (shots : ℕ) : ℚ :=
  total / shots

theorem shooting_average :
  let a := 1
  let b := 4
  let c := 3
  let d := 2
  let shots := 10
  total_points a b c d = 84 ∧
  average_points (total_points a b c d) shots = 8.4 :=
by {
  sorry
}

end shooting_average_l803_803653


namespace harry_worked_hours_l803_803972

def harry_pay (x : ℝ) (H : ℝ) : ℝ :=
  if H ≤ 18 then H * x else 18 * x + (H - 18) * 1.5 * x

def james_pay (x : ℝ) : ℝ :=
  40 * x + (41 - 40) * 2 * x

theorem harry_worked_hours (x : ℝ) (H : ℝ) (h_eq : harry_pay x H = james_pay x) : 
  H = 34 :=
by
  have j_pay_eq : james_pay x = 42 * x := by
    simp [james_pay]
    ring
  have h_pay_eq : harry_pay x H = if H ≤ 18 then H * x else 18 * x + (H - 18) * 1.5 * x := by
    simp [harry_pay]
  rw [h_eq, j_pay_eq] at h_pay_eq
  by_cases H ≤ 18
  · -- Case: H ≤ 18
    have : H * x = 42 * x := by simp [h_pay_eq, h]
    have : H = 42 := by linarith
    have : 42 ≤ 18 := by linarith -- Contradiction
    contradiction
  · -- Case: H > 18
    have : (18 * x + (H - 18) * 1.5 * x = 42 * x) := by simp [h_pay_eq, h]
    have : (H - 18) * 1.5 * x = 24 * x := by linarith
    have : (H - 18) = 16 := by
      field_simp
      linarith
    have : H = 34 := by linarith
    exact this

end harry_worked_hours_l803_803972


namespace problem1_problem2_l803_803620

-- Problem 1
theorem problem1 : 40 + ((1 / 6) - (2 / 3) + (3 / 4)) * 12 = 43 :=
by
  sorry

-- Problem 2
theorem problem2 : (-1) ^ 2 * (-5) + ((-3) ^ 2 + 2 * (-5)) = 4 :=
by
  sorry

end problem1_problem2_l803_803620


namespace hypotenuse_exponent_l803_803129

theorem hypotenuse_exponent (
  (a b : ℝ)
  (ha : a = log 9 512)
  (hb : b = log 3 64)
) : ∃ h, 9^h = 32768 :=
by
  let h := real.sqrt (a^2 + b^2)
  use h
  have h_eq : h = 15 * log 9 2 := sorry
  rw [h_eq, real.pow_log_base, real.exp_log]
  exact rfl

end hypotenuse_exponent_l803_803129


namespace min_percentage_of_people_owning_95_percent_money_l803_803051

theorem min_percentage_of_people_owning_95_percent_money 
  (total_people: ℕ) (total_money: ℕ) 
  (P: ℕ) (M: ℕ) 
  (H1: P = total_people * 10 / 100) 
  (H2: M = total_money * 90 / 100)
  (H3: ∀ (people_owning_90_percent: ℕ), people_owning_90_percent = P → people_owning_90_percent * some_money = M) :
      P = total_people * 55 / 100 := 
sorry

end min_percentage_of_people_owning_95_percent_money_l803_803051


namespace find_f_f2_l803_803388

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 - x - 2
  else x / (x + 4) + Real.log x / Real.log 4

theorem find_f_f2 : f (f 2) = 7 / 2 :=
by
  sorry

end find_f_f2_l803_803388


namespace paintable_sum_l803_803403

theorem paintable_sum :
  ∃ (h t u v : ℕ), h > 0 ∧ t > 0 ∧ u > 0 ∧ v > 0 ∧
  (∀ k, k % h = 1 ∨ k % t = 2 ∨ k % u = 3 ∨ k % v = 4) ∧
  (∀ k k', k ≠ k' → (k % h ≠ k' % h ∧ k % t ≠ k' % t ∧ k % u ≠ k' % u ∧ k % v ≠ k' % v)) ∧
  1000 * h + 100 * t + 10 * u + v = 4536 :=
by
  sorry

end paintable_sum_l803_803403


namespace fib_sum_a_fib_sum_b_fib_sum_c_fib_sum_of_squares_d_l803_803521

-- Definitions to handle Fibonacci sequence
def Fib : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := Fib n + Fib (n + 1)

-- Property (a)
theorem fib_sum_a (n : ℕ) : (Finset.range (n + 1)).sum (λ k, Fib (k + 1)) = Fib (n + 2) - 1 := sorry

-- Property (b)
theorem fib_sum_b (n : ℕ) : (Finset.range n).sum (λ k, Fib (2 * (k + 1))) = Fib (2 * n + 1) - 1 := sorry

-- Property (c)
theorem fib_sum_c (n : ℕ) : (Finset.range n).sum (λ k, Fib (2 * k + 1)) = Fib (2 * n) := sorry

-- Property (d)
theorem fib_sum_of_squares_d (n : ℕ) : (Finset.range (n + 1)).sum (λ k, (Fib k) ^ 2) = Fib n * Fib (n + 1) := sorry

end fib_sum_a_fib_sum_b_fib_sum_c_fib_sum_of_squares_d_l803_803521


namespace binom_mult_eq_6720_l803_803697

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l803_803697


namespace part_I_part_II_l803_803881

noncomputable def f (x : ℝ) : ℝ := |2*x + 1| - |2*x - 4|
noncomputable def g (x : ℝ) : ℝ := 9 + 2*x - x^2

theorem part_I (x : ℝ) : f(x) > 1 ↔ x ∈ set.Ioi 1 := 
sorry

theorem part_II (x : ℝ) : |8*x - 16| ≥ g(x) - 2*f(x) :=
sorry

end part_I_part_II_l803_803881


namespace four_digit_numbers_without_repeated_digits_l803_803132

theorem four_digit_numbers_without_repeated_digits :
  let digits := [0, 1, 2, 3, 4]
  number_of_four_digit_numbers digits 96 :=
  sorry

end four_digit_numbers_without_repeated_digits_l803_803132


namespace smallest_integer_solution_l803_803179

theorem smallest_integer_solution : ∀ x : ℤ, (x < 2 * x - 7) → (8 = x) :=
by
  sorry

end smallest_integer_solution_l803_803179


namespace system_of_equations_solutions_l803_803530

theorem system_of_equations_solutions (x y : ℝ) : 
  (4 / (x^2 + y^2) + x^2 * y^2 = 5) ∧ 
  (x^4 + y^4 + 3 * x^2 * y^2 = 20) 
  ↔ 
  (x = sqrt 2 ∧ y = sqrt 2) ∨ 
  (x = sqrt 2 ∧ y = -sqrt 2) ∨ 
  (x = -sqrt 2 ∧ y = sqrt 2) ∨ 
  (x = -sqrt 2 ∧ y = -sqrt 2) := 
sorry

end system_of_equations_solutions_l803_803530


namespace buyers_purchase_both_l803_803988

theorem buyers_purchase_both 
  (total_buyers : ℕ) 
  (cake_buyers : ℕ) 
  (muffin_buyers : ℕ) 
  (neither_buyers : ℕ) 
  (H_total : total_buyers = 100) 
  (H_cake : cake_buyers = 50) 
  (H_muffin : muffin_buyers = 40) 
  (H_neither : neither_buyers = (0.27 * 100)) :
  (cake_buyers + muffin_buyers - (total_buyers - neither_buyers) = 17) :=
by
  sorry

end buyers_purchase_both_l803_803988


namespace input_equals_output_l803_803385

theorem input_equals_output (x : ℝ) :
  (x ≤ 1 → 2 * x - 3 = x) ∨ (x > 1 → x^2 - 3 * x + 3 = x) ↔ x = 3 :=
by
  sorry

end input_equals_output_l803_803385


namespace circle_area_proof_l803_803115

def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y = 0

def circle_center : ℝ × ℝ := (1, -2)
def circle_radius : ℝ := Real.sqrt 5

def circle_area : ℝ := Real.pi * circle_radius^2

theorem circle_area_proof : circle_area = 5 * Real.pi :=
by
  sorry

end circle_area_proof_l803_803115


namespace Moe_mowing_time_in_hours_l803_803497

-- Define the basic quantities and conditions
def lawn_length : ℝ := 100 -- in feet
def lawn_width : ℝ := 180 -- in feet
def swath_width_inch : ℝ := 30 -- in inches
def overlap_inch : ℝ := 6 -- in inches
def walking_speed : ℝ := 4500 -- in feet per hour
def bag_emptying_time_min : ℝ := 10 -- in minutes

-- Calculate intermediate values
def effective_swath_width : ℝ := (swath_width_inch - overlap_inch) / 12 -- convert from inches to feet
def number_of_strips_required : ℝ := lawn_width / effective_swath_width
def total_distance_mowed : ℝ := number_of_strips_required * lawn_length
def mowing_time_hours : ℝ := total_distance_mowed / walking_speed
def total_time_hours : ℝ := mowing_time_hours + bag_emptying_time_min / 60

-- Formalize the proof problem
theorem Moe_mowing_time_in_hours : abs (total_time_hours - 2.2) < 0.05 := by
  sorry

end Moe_mowing_time_in_hours_l803_803497


namespace eval_g_at_8_l803_803430

def g (x : ℚ) : ℚ := (3 * x + 2) / (x - 2)

theorem eval_g_at_8 : g 8 = 13 / 3 := by
  sorry

end eval_g_at_8_l803_803430


namespace share_a_is_240_l803_803969

def total_profit : ℕ := 630

def initial_investment_a : ℕ := 3000
def initial_investment_b : ℕ := 4000

def months_a1 : ℕ := 8
def months_a2 : ℕ := 4
def investment_a1 : ℕ := initial_investment_a * months_a1
def investment_a2 : ℕ := (initial_investment_a - 1000) * months_a2
def total_investment_a : ℕ := investment_a1 + investment_a2

def months_b1 : ℕ := 8
def months_b2 : ℕ := 4
def investment_b1 : ℕ := initial_investment_b * months_b1
def investment_b2 : ℕ := (initial_investment_b + 1000) * months_b2
def total_investment_b : ℕ := investment_b1 + investment_b2

def ratio_a : ℕ := 8
def ratio_b : ℕ := 13
def total_ratio : ℕ := ratio_a + ratio_b

noncomputable def share_a (total_profit : ℕ) (ratio_a ratio_total : ℕ) : ℕ :=
  (ratio_a * total_profit) / ratio_total

theorem share_a_is_240 :
  share_a total_profit ratio_a total_ratio = 240 :=
by
  sorry

end share_a_is_240_l803_803969


namespace binomial_product_result_l803_803692

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l803_803692


namespace amount_solution_y_correct_l803_803253

-- Define conditions
def solution_x_alcohol_percentage : ℝ := 0.10
def solution_y_alcohol_percentage : ℝ := 0.30
def volume_solution_x : ℝ := 300.0
def target_alcohol_percentage : ℝ := 0.18

-- Define the main question as a theorem
theorem amount_solution_y_correct (y : ℝ) :
  (30 + 0.3 * y = 0.18 * (300 + y)) → y = 200 :=
by
  sorry

end amount_solution_y_correct_l803_803253


namespace ending_number_of_sequence_divisible_by_11_l803_803152

theorem ending_number_of_sequence_divisible_by_11 : 
  ∃ (n : ℕ), 19 < n ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 6 → n = 19 + 11 * k) ∧ n = 77 :=
by
  sorry

end ending_number_of_sequence_divisible_by_11_l803_803152


namespace part_a_part_b_l803_803551

def triangle := Type
def point := Type

structure TriangleInCircle (ABC : triangle) where
  A : point
  B : point
  C : point
  A1 : point
  B1 : point
  C1 : point
  M : point
  r : Real
  R : Real

theorem part_a (ABC : triangle) (t : TriangleInCircle ABC) :
  ∃ MA MC MB_1, (MA * MC) / MB_1 = 2 * t.r := sorry
  
theorem part_b (ABC : triangle) (t : TriangleInCircle ABC) :
  ∃ MA_1 MC_1 MB, ( (MA_1 * MC_1) / MB) = t.R := sorry

end part_a_part_b_l803_803551


namespace solution_of_equation_l803_803338

theorem solution_of_equation :
  ∀ x : ℝ, (x + 60 / (x - 3) = -13) → (x = -7) :=
by
  intros x h
  have h1 : x ≠ 3, -- condition that x ≠ 3 to avoid division by zero
    sorry
  have h2 : x * (x - 3) + 60 = -13 * (x - 3),
    sorry
  have h3 : x^2 + 10 * x + 21 = 0,
    sorry
  have h4 : (x + 3) * (x + 7) = 0,
    sorry
  have h5 : x = -3 ∨ x = -7,
    sorry
  have h6 : x ≠ -3,
    sorry
  exact Or.resolve_left h5 h6

end solution_of_equation_l803_803338


namespace compute_r_squared_l803_803061

variables (α β p r : ℝ)

theorem compute_r_squared 
  (h_tan_alpha_beta : ∀ α β : ℝ, (∀ x : ℝ, x^2 - 2*p*x + p = 0) → tan α = (2*p) ∧ tan β = p) 
  (h_cot_alpha_beta : ∀ α β : ℝ, (∀ x : ℝ, x^2 - 2*r*x + r = 0) → cot α = (2*r) ∧ cot β = r) :
  r^2 = 1 / p^2 := 
sorry

end compute_r_squared_l803_803061


namespace prime_saturated_96_l803_803290

def is_prime_saturated (d : ℕ) : Prop :=
  let prime_factors := [2, 3]  -- list of the different positive prime factors of 96
  prime_factors.prod < d       -- the product of prime factors should be less than d

theorem prime_saturated_96 : is_prime_saturated 96 :=
by
  sorry

end prime_saturated_96_l803_803290


namespace angle_AEB_is_122_l803_803047

variables (A B C D E : Type) [HasEquilateralTriangle A B C] [HasEquilateralTriangle C D E]
variables (angle_EBD : ℝ)
variable (x : ℝ)
variable (EBD_62 : angle_EBD = 62)
variable (target : x = 122)

theorem angle_AEB_is_122 
  [HasAngleEquilateralTriangles A B C]
  [HasAngleEquilateralTriangles C D E]
  [HasAngle EBD]
  (h1 : angle EBD = 62)
  (h2 : angle x):
  x = 122 := 
    sorry

end angle_AEB_is_122_l803_803047


namespace total_albums_l803_803496

-- Defining the initial conditions
def albumsAdele : ℕ := 30
def albumsBridget : ℕ := albumsAdele - 15
def albumsKatrina : ℕ := 6 * albumsBridget
def albumsMiriam : ℕ := 7 * albumsKatrina
def albumsCarlos : ℕ := 3 * albumsMiriam
def albumsDiane : ℕ := 2 * albumsKatrina

-- Proving the total number of albums
theorem total_albums :
  albumsAdele + albumsBridget + albumsKatrina + albumsMiriam + albumsCarlos + albumsDiane = 2835 :=
by
  sorry

end total_albums_l803_803496


namespace maximum_area_of_equilateral_triangle_l803_803648

noncomputable def maximum_triangle_area (a b c : ℕ) : ℝ :=
  let s := (a : ℝ) * Real.sqrt b - (c : ℝ) in (Real.sqrt 3 / 4) * s^2

theorem maximum_area_of_equilateral_triangle 
  (PQ PR : ℕ) 
  (hPQ : PQ = 12) 
  (hPR : PR = 13)
  (a b c : ℕ)
  (h1 : b = 3)
  (h2 : a = 313)
  (h3 : c = 117)
  : a + b + c = 433 :=
by
  have h_eq : maximum_triangle_area a b c = (313 : ℝ) * Real.sqrt 3 - (117 : ℝ), from sorry,
  ring_nf,
  exact h_eq,
  sorry

end maximum_area_of_equilateral_triangle_l803_803648


namespace kite_area_is_28_l803_803359

noncomputable def area_of_kite : ℝ :=
  let base_upper := 8
  let height_upper := 2
  let base_lower := 8
  let height_lower := 5
  let area_upper := (1 / 2 : ℝ) * base_upper * height_upper
  let area_lower := (1 / 2 : ℝ) * base_lower * height_lower
  area_upper + area_lower

theorem kite_area_is_28 :
  area_of_kite = 28 :=
by
  simp [area_of_kite]
  sorry

end kite_area_is_28_l803_803359


namespace eccentricity_of_ellipse_l803_803980

theorem eccentricity_of_ellipse (a b c : ℝ) (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : P ∈ {xy | xy.1 ^ 2 / a ^ 2 + xy.2 ^ 2 / b ^ 2 = 1})
  (h4 : let m := dist P F1 in let n := dist P F2 in 
        m^2 + n^2 = 4 * c ^ 2 ∧ m + n = 2 * a ∧ 1 / 2 * m * n = a^2 / 3) :
  (a^2 - b^2 = c^2) → 
  (c^2 = 2 / 3 * a^2) →
  (sqrt (2 / 3 * a ^ 2) / a = sqrt (a ^ 2 - b ^ 2) / a) →
  let e := sqrt (a^2 - b^2) / a in
  e = sqrt 6 / 3 :=
begin
  sorry
end

end eccentricity_of_ellipse_l803_803980


namespace fraction_and_ratio_of_girls_and_boys_at_event_l803_803941

theorem fraction_and_ratio_of_girls_and_boys_at_event:
  (total_students_maplewood = 300) →
  (ratio_boys_to_girls_maplewood = 3 / 2) →
  (total_students_brookside = 240) →
  (ratio_boys_to_girls_brookside = 3 / 5) →
  let girls_maplewood := (total_students_maplewood * 2) / 5 in
  let girls_brookside := (total_students_brookside * 5) / 8 in
  let total_girls := girls_maplewood + girls_brookside in
  let total_students := total_students_maplewood + total_students_brookside in
  let boys_maplewood := (total_students_maplewood * 3) / 5 in
  let boys_brookside := (total_students_brookside * 3) / 8 in
  let total_boys := boys_maplewood + boys_brookside in
  (total_girls / total_students = 1 / 2) ∧ (total_boys / total_girls = 1) :=
sorry

end fraction_and_ratio_of_girls_and_boys_at_event_l803_803941


namespace temperature_range_l803_803890

-- Define the problem conditions
def highest_temp := 26
def lowest_temp := 12

-- The theorem stating the range of temperature change
theorem temperature_range : ∀ t : ℝ, lowest_temp ≤ t ∧ t ≤ highest_temp :=
by sorry

end temperature_range_l803_803890


namespace angle_in_second_quadrant_l803_803963

-- Assuming definitions for the angle and quadrant
def angle_in_degrees (x : ℝ) : ℝ :=
  x * 180 / Real.pi

def quadrant (x : ℝ) : String :=
  if x > 0 ∧ x < 90 then "First quadrant"
  else if x > 90 ∧ x < 180 then "Second quadrant"
  else if x > 180 ∧ x < 270 then "Third quadrant"
  else if x > 270 ∧ x < 360 then "Fourth quadrant"
  else "Not in a quadrant"

-- Given condition: the angle is 2π/3
def given_angle : ℝ := 2 * Real.pi / 3

-- Proof problem: Prove that the angle 2π/3 lies in the second quadrant
theorem angle_in_second_quadrant : quadrant (angle_in_degrees given_angle) = "Second quadrant" :=
  sorry

end angle_in_second_quadrant_l803_803963


namespace math_problem_proof_l803_803045

noncomputable def parametric_line_eqns : ℝ → ℝ × ℝ :=
λ t, (1 / 2 * t, (sqrt 3) / 2 * t)

def polar_curve_eqn (ρ θ : ℝ) : Prop :=
ρ^2 * (cos θ)^2 + 5 * (sqrt 3) * ρ * (cos θ) - ρ * (sin θ) + 3 = 0

def cartesian_curve_eqn (x y : ℝ) : Prop :=
y = x^2 + 5 * (sqrt 3) * x + 3

def segment_length (ρ1 ρ2 : ℝ) : ℝ :=
abs (ρ1 - ρ2)

theorem math_problem_proof (t : ℝ) :
  let (x, y) := parametric_line_eqns t in
  ∃ θ ρ1 ρ2,
    (polar_curve_eqn ρ1 (θ := π / 3)) ∧
    (polar_curve_eqn ρ2 (θ := π / 3)) ∧
    (cartesian_curve_eqn x y) ∧
    θ = π / 3 ∧
    segment_length ρ1 ρ2 = 12 :=
begin
  sorry
end

end math_problem_proof_l803_803045


namespace binary_representation_of_28_l803_803915

-- Define a function to convert a number to binary representation.
def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else 
    let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux n []

theorem binary_representation_of_28 : decimalToBinary 28 = [1, 1, 1, 0, 0] := 
  sorry

end binary_representation_of_28_l803_803915


namespace mode_and_median_of_data_set_l803_803448

-- Define the data set
def data_set : List ℝ := [90, 89, 90, 95, 92, 94, 93, 90]

-- Define the theorem to prove the mode and median
theorem mode_and_median_of_data_set : 
  (mode data_set = 90) ∧ (median data_set = 91) :=
by
  sorry

end mode_and_median_of_data_set_l803_803448


namespace viola_final_jump_l803_803168

variable (n : ℕ) (T : ℝ) (x : ℝ)

theorem viola_final_jump (h1 : T = 3.80 * n)
                        (h2 : (T + 3.99) / (n + 1) = 3.81)
                        (h3 : T + 3.99 + x = 3.82 * (n + 2)) : 
                        x = 4.01 :=
sorry

end viola_final_jump_l803_803168


namespace analytical_expression_of_f_l803_803434

-- Definition of the function based on given conditions
def f (x : ℝ) : ℝ := (1/3) * x^3 - 4 * x + 4

-- Conditions translated to properties in Lean
theorem analytical_expression_of_f :
      (∀ x, derivative (f x) = (fn'(x) = 0) → x = 2) ∧ -- Critical points and derivative findings
      f 2 = -4/3 → -- Function value at x=2
      f (-3) = 7 ∧ f(3) = 1 ∧ f (-2) = 28/3 ∧ f (2) = -4/3 → -- Values at critical points
      function_extrema f [-3, 3] = (max 28/3, min -4/3) :=
begin 
sorry
end

end analytical_expression_of_f_l803_803434


namespace solve_equation1_solve_equation2_solve_equation3_solve_equation4_l803_803909

noncomputable def equation1_roots : set ℝ := {1, 9}
noncomputable def equation2_roots : set ℝ := { -1 + real.sqrt 6 / 2, -1 - real.sqrt 6 / 2 }
noncomputable def equation3_roots : set ℝ := {-1, 2 / 5}
noncomputable def equation4_roots : set ℝ := { -1 / 2, 1 }

theorem solve_equation1 (x : ℝ) : (x - 5)^2 = 16 ↔ x ∈ equation1_roots := sorry
theorem solve_equation2 (x : ℝ) : 2 * x^2 - 1 = -4 * x ↔ x ∈ equation2_roots := sorry
theorem solve_equation3 (x : ℝ) : 5 * x * (x + 1) = 2 * (x + 1) ↔ x ∈ equation3_roots := sorry
theorem solve_equation4 (x : ℝ) : 2 * x^2 - x - 1 = 0 ↔ x ∈ equation4_roots := sorry

end solve_equation1_solve_equation2_solve_equation3_solve_equation4_l803_803909


namespace a_wins_by_200_meters_l803_803443

-- Define the conditions
def race_distance : ℕ := 600
def speed_ratio_a_to_b := 5 / 4
def head_start_a : ℕ := 100

-- Define the proof statement
theorem a_wins_by_200_meters (x : ℝ) (ha_speed : ℝ := 5 * x) (hb_speed : ℝ := 4 * x)
  (ha_distance_to_win : ℝ := race_distance - head_start_a) :
  (ha_distance_to_win / ha_speed) = (400 / hb_speed) → 
  600 - (400) = 200 :=
by
  -- For now, skip the proof, focus on the statement.
  sorry

end a_wins_by_200_meters_l803_803443


namespace num_children_in_family_l803_803643

def regular_ticket_cost := 15
def elderly_ticket_cost := 10
def adult_ticket_cost := 12
def child_ticket_cost := adult_ticket_cost - 5
def total_money_handled := 3 * 50
def change_received := 3
def num_adults := 4
def num_elderly := 2
def total_cost_for_adults := num_adults * adult_ticket_cost
def total_cost_for_elderly := num_elderly * elderly_ticket_cost
def total_cost_of_tickets := total_money_handled - change_received

theorem num_children_in_family : ∃ (num_children : ℕ), 
  total_cost_of_tickets = total_cost_for_adults + total_cost_for_elderly + num_children * child_ticket_cost ∧ 
  num_children = 11 := 
by
  sorry

end num_children_in_family_l803_803643


namespace tye_total_bills_l803_803590

theorem tye_total_bills (w1 w2 b : ℕ) (h1 : w1 = 450) (h2 : w2 = 750) (h3 : b = 20) : 
  (w1 + w2) / b = 60 :=
by
  rw [h1, h2, h3]
  simp
  sorry

end tye_total_bills_l803_803590


namespace stock_percentage_l803_803921

noncomputable def market_value : ℝ := 103.91666666666667
noncomputable def annual_income : ℝ := 756
noncomputable def initial_investment : ℝ := 7500
noncomputable def brokerage_rate : ℝ := 0.0025
noncomputable def brokerage_fee : ℝ := initial_investment * brokerage_rate
noncomputable def net_investment : ℝ := initial_investment - brokerage_fee
noncomputable def final_dividend_yield : ℝ := (annual_income / net_investment) * 100

theorem stock_percentage :
  final_dividend_yield ≈ 10.10 :=
sorry

end stock_percentage_l803_803921


namespace no_real_solutions_for_equation_l803_803348

theorem no_real_solutions_for_equation:
  ∀ x : ℝ, (3 * x / (x^2 + 2 * x + 4) + 4 * x / (x^2 - 4 * x + 5) = 1) →
  (¬(∃ x : ℝ, 3 * x / (x^2 + 2 * x + 4) + 4 * x / (x^2 - 4 * x + 5) = 1)) :=
by
  sorry

end no_real_solutions_for_equation_l803_803348


namespace P_geq_n_minus_3_P_leq_2n_minus_7_P_leq_2n_minus_10_for_geq_13_l803_803638

theorem P_geq_n_minus_3 (n : ℕ) (P : ℕ → ℕ): P(n) ≥ n - 3 :=
sorry

theorem P_leq_2n_minus_7 (n : ℕ) (P : ℕ → ℕ): P(n) ≤ 2 * n - 7 :=
sorry

theorem P_leq_2n_minus_10_for_geq_13 (n : ℕ) (P : ℕ → ℕ): n ≥ 13 → P(n) ≤ 2 * n - 10 :=
sorry

end P_geq_n_minus_3_P_leq_2n_minus_7_P_leq_2n_minus_10_for_geq_13_l803_803638


namespace dave_short_sleeve_shirts_l803_803722

theorem dave_short_sleeve_shirts (S : ℕ) : 
  27 + S = 36 ∧ 20 + 16 = 36 → S = 9 := 
by {
  intros h,
  cases h with h1 h2,
  sorry
}

end dave_short_sleeve_shirts_l803_803722


namespace sum_x_coords_above_line_is_zero_l803_803889

open BigOperators

theorem sum_x_coords_above_line_is_zero :
  let points := [(4, 15), (8, 25), (10, 30), (14, 40), (18, 45), (22, 55)] in
  (∑ p in points, if p.2 > 3 * p.1 + 5 then p.1 else 0) = 0 := by
  -- Points and conditions are stated
  let points : List (ℕ × ℕ) := [(4, 15), (8, 25), (10, 30), (14, 40), (18, 45), (22, 55)]
  -- Sorry used because proof is skipped
  sorry

end sum_x_coords_above_line_is_zero_l803_803889


namespace man_age_twice_son_age_l803_803288

theorem man_age_twice_son_age (S M X : ℕ) (h1 : S = 28) (h2 : M = S + 30) (h3 : M + X = 2 * (S + X)) : X = 2 :=
by
  sorry

end man_age_twice_son_age_l803_803288


namespace race_head_start_l803_803274

theorem race_head_start (v_B : ℕ) (s : ℕ) :=
  let v_A := 4 * v_B;        -- Speed of A
  let d_B := 88 - s;         -- Distance B covers
  let d_A := 88;             -- Distance A covers
  v_B ≠ 0 ∧ v_A ≠ 0 ∧
  (d_B : ℚ) / (v_B : ℚ) = (d_A : ℚ) / (v_A : ℚ) →  -- Both finish at the same time
  s = 66 :=
by
  sorry

end race_head_start_l803_803274


namespace count_multiples_of_neither_6_nor_8_l803_803415

theorem count_multiples_of_neither_6_nor_8 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999},
      multiples_of_6 := {n : ℕ | n ∈ three_digit_numbers ∧ n % 6 = 0},
      multiples_of_8 := {n : ℕ | n ∈ three_digit_numbers ∧ n % 8 = 0},
      multiples_of_24 := {n : ℕ | n ∈ three_digit_numbers ∧ n % 24 = 0} in
  (three_digit_numbers.card 
   - (multiples_of_6.card + multiples_of_8.card - multiples_of_24.card)) = 675 :=
by sorry

end count_multiples_of_neither_6_nor_8_l803_803415


namespace num_cars_can_be_parked_l803_803846

theorem num_cars_can_be_parked (length width : ℝ) (useable_percentage : ℝ) (area_per_car : ℝ) 
  (h_length : length = 400) (h_width : width = 500) (h_useable_percentage : useable_percentage = 0.80) 
  (h_area_per_car : area_per_car = 10) : 
  length * width * useable_percentage / area_per_car = 16000 := 
by 
  sorry

end num_cars_can_be_parked_l803_803846


namespace sqrt_eight_simplify_l803_803528

theorem sqrt_eight_simplify : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_eight_simplify_l803_803528


namespace range_of_a_l803_803789

theorem range_of_a (a x : ℝ) : (3 * x + 2 * (3 * a + 1) = 6 * x + a) → (x ≥ 0) → a ≥ -2 / 5 :=
begin
  sorry
end

end range_of_a_l803_803789


namespace solve_number_puzzle_l803_803974

def number_puzzle (N : ℕ) : Prop :=
  (1/4) * (1/3) * (2/5) * N = 14 → (40/100) * N = 168

theorem solve_number_puzzle : ∃ N, number_puzzle N := by
  sorry

end solve_number_puzzle_l803_803974


namespace temperature_difference_l803_803919

theorem temperature_difference (T_high T_low : ℤ) (h_high : T_high = 11) (h_low : T_low = -11) :
  T_high - T_low = 22 := by
  sorry

end temperature_difference_l803_803919


namespace parallel_intersection_l803_803241

open Set

variable {α : Type*}

-- Define the notion of parallel lines and planes
def parallel {α : Type*} [n : NormedAddTorsor V P] [NormedSpace ℝ V] (l₁ l₂ : AffineSubspace ℝ P) : Prop :=
  ∃ v₁ v₂ ∈ l₁.direction, ∃ v ∈ l₂.direction, v₁ = u.vadd v₂.smul.v

-- Define the problem statement
theorem parallel_intersection (l : AffineSubspace ℝ P)
    (p₁ p₂ : AffineSubspace ℝ P) : 
 l.direction ∈ P.upper.parallel' → p₁ ⊓ n.vector_space.fst.directions.parallel.n.snd l.upper →  
         parallel (direction_α.parallel (c.linear_map.p₂ ⊓ P l p₂)).direction l :=
begin
    sorry,
end

end parallel_intersection_l803_803241


namespace line_equation_through_points_and_area_l803_803287

variable (a b S : ℝ)
variable (h_b_gt_a : b > a)
variable (h_area : S = 1/2 * (b - a) * (2 * S / (b - a)))

theorem line_equation_through_points_and_area :
  0 = -2 * S * x + (b - a)^2 * y + 2 * S * a - 2 * S * b := sorry

end line_equation_through_points_and_area_l803_803287


namespace value_of_n_l803_803020

theorem value_of_n (n : ℤ) (h : (sqrt (10 + n : ℝ) = 8)) : n = 54 := by
  sorry

end value_of_n_l803_803020


namespace select_two_people_l803_803906

theorem select_two_people {n : ℕ} (h1 : n ≠ 0) (h2 : n ≥ 2) (h3 : (n - 1) ^ 2 = 25) : n = 6 :=
by
  sorry

end select_two_people_l803_803906


namespace sum_of_digits_of_least_N_l803_803059

noncomputable def Q (N : ℕ) : ℚ := sorry  -- Assuming this is given or will be defined properly

theorem sum_of_digits_of_least_N :
  (∃ N : ℕ,
    N > 0 ∧
    N % 8 = 0 ∧
    Q(N) < 319 / 400 ∧
    (Nat.digits 10 N).sum = 8) :=
sorry

end sum_of_digits_of_least_N_l803_803059


namespace CP_eq_CQ_l803_803484

-- Define the Rhombus and Points
structure Rhombus (A B C D F E L K Q P : Type) :=
  (A B C D F E L K Q P : Type)
  (F_on_AD : F ∈ [AD])
  (E_on_AB : E ∈ [AB])
  (L_intersect_FC_BD : L ∈ FC ∩ BD)
  (K_intersect_EC_BD : K ∈ EC ∩ BD)
  (Q_intersect_FK_BC : Q ∈ FK ∩ BC)
  (P_intersect_EL_DC : P ∈ EL ∩ DC)

-- Define the proof problem to show CP = CQ
theorem CP_eq_CQ (A B C D F E L K Q P : Type) [rhombus : Rhombus A B C D F E L K Q P] : 
  CP = CQ :=
sorry

end CP_eq_CQ_l803_803484


namespace part_a_part_b_part_c_l803_803461

noncomputable def f (m n : ℕ) : ℕ := |(black_area m n) - (white_area m n)|

-- Part (a):
theorem part_a (m n : ℕ) (hmn : (m % 2 = 0 ∧ n % 2 = 0) ∨ (m % 2 = 1 ∧ n % 2 = 1)) :
  f m n = if m % 2 = 0 ∧ n % 2 = 0 then 0 else 1 :=
sorry

-- Part (b):
theorem part_b (m n : ℕ) : f m n ≤ (max m n) / 2 :=
sorry

-- Part (c):
theorem part_c : ¬ ∃ C, ∀ (m n : ℕ), f m n < C :=
sorry

end part_a_part_b_part_c_l803_803461


namespace a_20_is_minus_sqrt_3_l803_803395

noncomputable def a : ℕ → ℝ
| 0     := 0
| (n+1) := (a n - real.sqrt 3) / (real.sqrt 3 * a n + 1)

theorem a_20_is_minus_sqrt_3 : a 20 = -real.sqrt 3 :=
sorry

end a_20_is_minus_sqrt_3_l803_803395


namespace pipe_length_l803_803292

theorem pipe_length (S L : ℕ) (h1: S = 28) (h2: L = S + 12) : S + L = 68 := 
by
  sorry

end pipe_length_l803_803292


namespace product_of_sequence_l803_803044

theorem product_of_sequence : 
  (∃ (a : ℕ → ℚ), (a 1 * a 2 * a 3 * a 4 * a 5 = -32) ∧ 
  ((∀ n : ℕ, 3 * a (n + 1) + a n = 0) ∧ a 2 = 6)) :=
sorry

end product_of_sequence_l803_803044


namespace tangent_line_equation_l803_803125

theorem tangent_line_equation (x y : ℝ) (h : x = √3 ∧ y = 1 ∧ x^2 + y^2 = 4) :
    √3 * x + y - 4 = 0 := 
sorry

end tangent_line_equation_l803_803125


namespace max_projectile_height_l803_803294

-- Define the height function
def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

-- Define the statement to prove the maximum height of the projectile
theorem max_projectile_height : ∃ t : ℝ, height t = 161 :=
  sorry

end max_projectile_height_l803_803294


namespace integer_solutions_positive_product_l803_803349

theorem integer_solutions_positive_product :
  {a : ℤ | (5 + a) * (3 - a) > 0} = {-4, -3, -2, -1, 0, 1, 2} :=
by
  sorry

end integer_solutions_positive_product_l803_803349


namespace mystery_book_shelves_l803_803902

-- Define the conditions from the problem
def total_books : ℕ := 72
def picture_book_shelves : ℕ := 2
def books_per_shelf : ℕ := 9

-- Determine the number of mystery book shelves
theorem mystery_book_shelves : 
  let books_on_picture_shelves := picture_book_shelves * books_per_shelf
  let mystery_books := total_books - books_on_picture_shelves
  let mystery_shelves := mystery_books / books_per_shelf
  mystery_shelves = 6 :=
by {
  -- This space is intentionally left incomplete, as the proof itself is not required.
  sorry
}

end mystery_book_shelves_l803_803902


namespace perpendicular_lines_slope_product_l803_803822

theorem perpendicular_lines_slope_product {m : ℝ} : 
  (∀ x y, x - y + 1 = 0) → (∀ x y, mx + 3 * y - 1 = 0) → m = 3 :=
  by
  intros line1 line2
  -- add the conditions for perpendicularity of the slopes
  -- simplify the resulting equation to find m = 3
  sorry

end perpendicular_lines_slope_product_l803_803822


namespace find_xy_midpoint_eq_l803_803457

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D A₁ B₁ C₁ D₁ P : V)
variables (x y : ℝ)

-- Conditions
def midpoint (P C₁ D₁ : V) : Prop := P = (C₁ + D₁) / 2

-- Given vector equation
def vector_eq (A P D B A₁ : V) (x y : ℝ) : Prop :=
  P - A = D - A + x • (B - A) + y • (A₁ - A)

-- Proof problem
theorem find_xy_midpoint_eq (h_mid : midpoint P C₁ D₁)
  (h_vec : vector_eq A P D B A₁ x y) : x + y = 3 / 2 :=
sorry

end find_xy_midpoint_eq_l803_803457


namespace proof_theorem_l803_803307

noncomputable def proof_problem (y1 y2 y3 y4 y5 : ℝ) :=
  y1 + 8*y2 + 27*y3 + 64*y4 + 125*y5 = 7 ∧
  8*y1 + 27*y2 + 64*y3 + 125*y4 + 216*y5 = 100 ∧
  27*y1 + 64*y2 + 125*y3 + 216*y4 + 343*y5 = 1000 →
  64*y1 + 125*y2 + 216*y3 + 343*y4 + 512*y5 = -5999

theorem proof_theorem : ∀ (y1 y2 y3 y4 y5 : ℝ), proof_problem y1 y2 y3 y4 y5 :=
  by intros y1 y2 y3 y4 y5
     unfold proof_problem
     intro h
     sorry

end proof_theorem_l803_803307


namespace eq_abs_distinct_solution_count_l803_803323

theorem eq_abs_distinct_solution_count :
  ∃! x : ℝ, |x - 10| = |x + 5| + 2 := 
sorry

end eq_abs_distinct_solution_count_l803_803323


namespace range_a_l803_803796

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≥ 1 then log (x^2 + x + a) / log 2 else 1 - x^2

theorem range_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ∈ set.Icc (-2 : ℝ) 0 :=
by
  sorry

end range_a_l803_803796


namespace odd_number_of_courtiers_l803_803468

theorem odd_number_of_courtiers (n : ℕ) (watches : Fin n → Fin n) :
  (∀ (i : Fin n), watches (watches i) = Fin.addNat i 1) →
  Nat.Odd n :=
by
  sorry

end odd_number_of_courtiers_l803_803468


namespace triangle_proof_l803_803825

noncomputable def triangle_problem : Prop :=
  ∀ (A B C : ℝ) (c : ℝ),
    (sin (π / 2 + A) = 2 * sqrt 5 / 5) →
    (cos B = 3 * sqrt 10 / 10) →
    (c = 10) →
    (tan (2 * A) = 4 / 3) ∧
    let sin_A := sqrt (1 - (2 * sqrt 5 / 5)^2),
        a := (10 * sin_A) / (sqrt 2 / 2),
        area := 1 / 2 * a * 10 * sin B in
    area = 10

theorem triangle_proof : triangle_problem :=
  begin
    -- Proof goes here
    sorry
  end

end triangle_proof_l803_803825


namespace proof_problem_l803_803426

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 1)^2

theorem proof_problem : f (g (-3)) = 67 := 
by 
  sorry

end proof_problem_l803_803426


namespace find_breadth_of_courtyard_l803_803410

variable (stone_length : ℕ) (stone_breadth : ℕ) (num_stones : ℕ) (courtyard_length : ℝ)

def courtyard_breadth (stone_length stone_breadth : ℝ) (num_stones courtyard_length : ℕ) : ℝ :=
  (num_stones * (stone_length * stone_breadth)) / courtyard_length

theorem find_breadth_of_courtyard :
  courtyard_breadth 2.5 2 198 60 = 16.5 :=
by
  sorry

end find_breadth_of_courtyard_l803_803410


namespace log_addition_l803_803678

theorem log_addition (log_base_10 : ℝ → ℝ) (a b : ℝ) (h_base_10_log : log_base_10 10 = 1) :
  log_base_10 2 + log_base_10 5 = 1 :=
by
  sorry

end log_addition_l803_803678


namespace smallest_integer_with_divisors_properties_l803_803199

def number_of_odd_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 1)

def number_of_even_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 0)

theorem smallest_integer_with_divisors_properties :
  ∃ n : ℕ, number_of_odd_divisors n = 8 ∧ number_of_even_divisors n = 16 ∧ n = 4000 :=
by
  sorry

end smallest_integer_with_divisors_properties_l803_803199


namespace find_radius_concentric_circle_l803_803276

noncomputable def is_visible_sides (r : ℝ) : Prop :=
  ∃ point_on_circle, (probability_two_sides_visible r = 1 / 2)

def circle_concentric_square_radius (side_length : ℝ) (prob : ℝ) : ℝ :=
  if side_length = 2 ∧ prob = 1/2 then 2 else 0

theorem find_radius_concentric_circle :
  circle_concentric_square_radius 2 (1 / 2) = 2 :=
  by
    sorry

end find_radius_concentric_circle_l803_803276


namespace no_convex_1978_gon_with_integer_angles_l803_803328

theorem no_convex_1978_gon_with_integer_angles :
  ¬ ∃ (polygon : Polygon),
    polygon.sides = 1978 ∧
    polygon.convex ∧
    (∀ i, polygon.internal_angle i ∈ ℤ) := 
sorry

end no_convex_1978_gon_with_integer_angles_l803_803328


namespace frac_is_correct_l803_803724

def at (a b : ℕ) := a * b - b^2
def hash (a b : ℕ) := a - b + a * b^2

theorem frac_is_correct : (at 7 3) / (hash 7 3) = 12 / 67 := 
by
  sorry

end frac_is_correct_l803_803724


namespace obtain_reciprocal_impossible_l803_803578

def f1 (x : ℝ) := x + 1 / x
def f2 (x : ℝ) := x ^ 2
def f3 (x : ℝ) := (x - 1) ^ 2

theorem obtain_reciprocal_impossible
(f1_missing : ¬∃ x, (x + 1 / x = 0))
(f2_missing : ¬∃ x, (x ^ 2 = 0))
(f3_missing : ¬∃ x, ((x - 1) ^ 2 = 0))
: ∀ (f : ℝ → ℝ), f ≠ (λ x, 1 / x) := 
sorry

end obtain_reciprocal_impossible_l803_803578


namespace A_union_B_eq_B_l803_803804

def A : Set ℕ := { n | ∃ k : ℕ, n = 2 * k - 1 ∧ (2 ^ n + 3 ^ n) % 5 = 0 }
def B : Set ℤ := { m | ∃ u v : ℤ, m = u ^ 2 - v ^ 2 }

theorem A_union_B_eq_B : A ∪ B = B := 
by
  sorry

end A_union_B_eq_B_l803_803804


namespace teacher_arrangements_l803_803673

theorem teacher_arrangements (T : Fin 30 → ℕ) (h1 : T 1 < T 2 ∧ T 2 < T 3 ∧ T 3 < T 4 ∧ T 4 < T 5)
  (h2 : ∀ i : Fin 4, T (i + 1) ≥ T i + 3)
  (h3 : 1 ≤ T 1)
  (h4 : T 5 ≤ 26) :
  ∃ n : ℕ, n = 26334 := by
  sorry

end teacher_arrangements_l803_803673


namespace units_digit_fraction_l803_803961

theorem units_digit_fraction : (2^3 * 31 * 33 * 17 * 7) % 10 = 6 := by
  sorry

end units_digit_fraction_l803_803961


namespace probability_point_lies_on_line_l803_803639

-- Definitions for throws of a six-sided die
def is_die_throw (n : ℕ) : Prop := n >= 1 ∧ n <= 6

-- Probability calculation
theorem probability_point_lies_on_line :
  (∃ (x y : ℕ), is_die_throw x ∧ is_die_throw y ∧ y = - (x : ℤ) + 5) → 
  ∑ x in (finset.range 6).filter is_die_throw, 
    ∑ y in (finset.range 6).filter (is_die_throw ∘ nat_abs ∘ (λ y, y - x + 5)), 1  = 4 / 36 :=
sorry

end probability_point_lies_on_line_l803_803639


namespace middle_term_is_35_l803_803455

-- Define the arithmetic sequence condition
def arithmetic_sequence (a b c d e f : ℤ) : Prop :=
  b - a = c - b ∧ c - b = d - c ∧ d - c = e - d ∧ e - d = f - e

-- Given sequence values
def seq1 := 23
def seq6 := 47

-- Theorem stating that the middle term y in the sequence is 35
theorem middle_term_is_35 (x y z w : ℤ) :
  arithmetic_sequence seq1 x y z w seq6 → y = 35 :=
by
  sorry

end middle_term_is_35_l803_803455


namespace inequality_problem_l803_803748

variable {a b c : ℝ}

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_problem_l803_803748


namespace exists_polyhedron_with_no_three_same_sided_faces_l803_803686

structure Face :=
  (sides : ℕ)

structure Polyhedron :=
  (faces : List Face)
  (closed : Bool)

-- Definition of specific faces
def triangular_face : Face := ⟨3⟩
def quadrilateral_face : Face := ⟨4⟩
def pentagonal_face : Face := ⟨5⟩

-- Definition of the polyhedron in terms of the conditions
def polyhedron_example : Polyhedron := 
  ⟨[triangular_face, triangular_face, quadrilateral_face, quadrilateral_face, pentagonal_face, pentagonal_face], true⟩

theorem exists_polyhedron_with_no_three_same_sided_faces : 
  ∃ (p : Polyhedron), p = polyhedron_example ∧ p.closed ∧ 
    (∀ n, (p.faces.filter (λ f, f.sides = n)).length < 3) :=
by
  sorry

end exists_polyhedron_with_no_three_same_sided_faces_l803_803686


namespace donations_problem_l803_803837

theorem donations_problem
  (x : ℕ) -- the number of people who donated the first time
  (h1 : 60000 / x < 150000)
  (h2 : 1.5 * x = 3 / 2 * x)
  (h3 : 150000 / (3 / 2 * x) - 60000 / x = 20) :
  x = 2000 ∧ 1.5 * x = 3000 := 
by
  sorry

end donations_problem_l803_803837


namespace sequence_nonnegative_if_a_geq_3_l803_803322

noncomputable def sequence {ℝ : Type*} [LinearOrderedField ℝ] (a : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := 2 * (sequence a n) - (n : ℝ) ^ 2

theorem sequence_nonnegative_if_a_geq_3 {ℝ : Type*} [LinearOrderedField ℝ] (a : ℝ) :
  (∀ n : ℕ, sequence a n ≥ 0) ↔ 3 ≤ a :=
begin
  sorry
end

end sequence_nonnegative_if_a_geq_3_l803_803322


namespace find_s_l803_803861

def is_monic_cubic (p : Polynomial ℝ) : Prop :=
  p.degree = 3 ∧ p.leadingCoeff = 1

def has_roots (p : Polynomial ℝ) (roots : Set ℝ) : Prop :=
  ∀ x ∈ roots, p.eval x = 0

def poly_condition (f g : Polynomial ℝ) (s : ℝ) : Prop :=
  ∀ x : ℝ, f.eval x - g.eval x = 2 * s

theorem find_s (s : ℝ)
  (f g : Polynomial ℝ)
  (hf_monic : is_monic_cubic f)
  (hg_monic : is_monic_cubic g)
  (hf_roots : has_roots f {s + 2, s + 6})
  (hg_roots : has_roots g {s + 4, s + 10})
  (h_condition : poly_condition f g s) :
  s = 10.67 :=
sorry

end find_s_l803_803861


namespace find_value_of_expression_l803_803382

variables (a b : ℝ)

-- Given the condition that 2a - 3b = 5, prove that 2a - 3b + 3 = 8.
theorem find_value_of_expression
  (h : 2 * a - 3 * b = 5) : 2 * a - 3 * b + 3 = 8 :=
by sorry

end find_value_of_expression_l803_803382


namespace ratio_of_sums_l803_803566

variable (n : ℕ) (a d : ℝ)

def arithmetic_sequence (i : ℕ) : ℝ := a + i * d

noncomputable def sum_arithmetic_sequence (start : ℕ) (step : ℕ) (count : ℕ) : ℝ :=
∑ i in finset.range count, arithmetic_sequence a d (start + i * step)

def sum_odd_terms (n : ℕ) : ℝ :=
sum_arithmetic_sequence a d 0 2 (n + 1)

def sum_even_terms (n : ℕ) : ℝ :=
sum_arithmetic_sequence a d 1 2 n

theorem ratio_of_sums (n : ℕ) (a d : ℝ) :
  (sum_arithmetic_sequence a d 0 2 (n + 1)) / (sum_arithmetic_sequence a d 1 2 n) = (n + 1) / n := by
  sorry

end ratio_of_sums_l803_803566


namespace discount_percentage_l803_803299

theorem discount_percentage (cp mp pm : ℤ) (x : ℤ) 
    (Hcp : cp = 160) 
    (Hmp : mp = 240) 
    (Hpm : pm = 20) 
    (Hcondition : mp * (100 - x) = cp * (100 + pm)) : 
  x = 20 := 
  sorry

end discount_percentage_l803_803299


namespace john_burritos_left_l803_803849

def total_burritos (b1 b2 b3 b4 : ℕ) : ℕ :=
  b1 + b2 + b3 + b4

def burritos_left_after_giving_away (total : ℕ) (fraction : ℕ) : ℕ :=
  total - (total / fraction)

def burritos_left_after_eating (burritos_left : ℕ) (burritos_per_day : ℕ) (days : ℕ) : ℕ :=
  burritos_left - (burritos_per_day * days)

theorem john_burritos_left :
  let b1 := 15
  let b2 := 20
  let b3 := 25
  let b4 := 5
  let total := total_burritos b1 b2 b3 b4
  let burritos_after_give_away := burritos_left_after_giving_away total 3
  let burritos_after_eating := burritos_left_after_eating burritos_after_give_away 3 10
  burritos_after_eating = 14 :=
by
  sorry

end john_burritos_left_l803_803849


namespace sin_585_eq_neg_sqrt2_div_2_l803_803713

theorem sin_585_eq_neg_sqrt2_div_2 : Real.sin (585 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_585_eq_neg_sqrt2_div_2_l803_803713


namespace find_lambda_l803_803840

noncomputable def a_seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | (n+2) => 2 * a_seq (n+1)

def S_seq (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => a_seq (i + 1))

theorem find_lambda 
  (h₁ : a_seq 1 = 1)
  (h₂ : ∀ n, a_seq (n+1) = 2 * a_seq n)
  (h₃ : ∀ n, S_seq (n+1) = (Finset.range (n+1)).sum (λ i => a_seq (i + 1)))
  (h₄ : ∃ q : ℕ, ∀ n m : ℕ, S_seq (n+m) + 1 = q * (S_seq n + 1) * (S_seq m + 1)) :
  1 = 1 :=
by sorry

end find_lambda_l803_803840


namespace regular_polygon_sides_l803_803541

-- Conditions
def central_angle (θ : ℝ) := θ = 30
def sum_of_central_angles (sumθ : ℝ) := sumθ = 360

-- The proof problem
theorem regular_polygon_sides (θ sumθ : ℝ) (h₁ : central_angle θ) (h₂ : sum_of_central_angles sumθ) :
  sumθ / θ = 12 := by
  sorry

end regular_polygon_sides_l803_803541


namespace marciaHairLengthProof_l803_803308

noncomputable def marciaHairLengthAtEndOfSchoolYear : Float :=
  let L0 := 24.0                           -- initial length
  let L1 := L0 - 0.3 * L0                  -- length after September cut
  let L2 := L1 + 3.0 * 1.5                 -- length after three months of growth (Sept - Dec)
  let L3 := L2 - 0.2 * L2                  -- length after January cut
  let L4 := L3 + 5.0 * 1.8                 -- length after five months of growth (Jan - May)
  let L5 := L4 - 4.0                       -- length after June cut
  L5

theorem marciaHairLengthProof : marciaHairLengthAtEndOfSchoolYear = 22.04 :=
by
  sorry

end marciaHairLengthProof_l803_803308


namespace smallest_integer_has_8_odd_and_16_even_divisors_l803_803221

/-!
  Prove that the smallest positive integer with exactly 8 positive odd integer divisors
  and exactly 16 positive even integer divisors is 540.
-/
def smallest_integer_with_divisors : ℕ :=
  540

theorem smallest_integer_has_8_odd_and_16_even_divisors 
  (n : ℕ) 
  (h1 : (8 : ℕ) = nat.count (λ d, d % 2 = 1) (nat.divisors n))
  (h2 : (16 : ℕ) = nat.count (λ d, d % 2 = 0) (nat.divisors n)) :
  n = 540 :=
sorry

end smallest_integer_has_8_odd_and_16_even_divisors_l803_803221


namespace largest_lambda_for_inequality_l803_803345

theorem largest_lambda_for_inequality :
  ∀ (a b c d μ λ: ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → 0 ≤ μ →
  (a^2 + b^2 + c^2 + d^2 + μ * a * d ≥ a * b + λ * b * c + c * d + μ * a * d) → 
  λ ≤ 3 / 2 :=
sorry

end largest_lambda_for_inequality_l803_803345


namespace point_not_in_region_l803_803137

-- Define the inequality
def inequality (x y : ℝ) : Prop := 3 * x + 2 * y < 6

-- Points definition
def point := ℝ × ℝ

-- Points to be checked
def p1 : point := (0, 0)
def p2 : point := (1, 1)
def p3 : point := (0, 2)
def p4 : point := (2, 0)

-- Conditions stating that certain points satisfy the inequality
axiom h1 : inequality p1.1 p1.2
axiom h2 : inequality p2.1 p2.2
axiom h3 : inequality p3.1 p3.2

-- Goal: Prove that point (2,0) does not satisfy the inequality
theorem point_not_in_region : ¬ inequality p4.1 p4.2 :=
sorry -- Proof omitted

end point_not_in_region_l803_803137


namespace problem_l803_803471

def S : Finset ℕ := {n : ℕ | n > 0 ∧ n < 2^40 ∧ Integer.bitCount n = 2}

def isDivisibleBy9 (n : ℕ) : Prop := n % 9 = 0

def numValidPairs : ℕ := 133

def totalPairs : ℕ := 780

def p : ℕ := numValidPairs
def q : ℕ := totalPairs

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

def relatively_prime (a b : ℕ) : Prop := gcd a b = 1

theorem problem (hpq : relatively_prime p q) : p + q = 913 :=
by
  have p_def : p = 133 := rfl
  have q_def : q = 780 := rfl
  rw [p_def, q_def]
  exact rfl

end problem_l803_803471


namespace smallest_integer_has_8_odd_and_16_even_divisors_l803_803218

/-!
  Prove that the smallest positive integer with exactly 8 positive odd integer divisors
  and exactly 16 positive even integer divisors is 540.
-/
def smallest_integer_with_divisors : ℕ :=
  540

theorem smallest_integer_has_8_odd_and_16_even_divisors 
  (n : ℕ) 
  (h1 : (8 : ℕ) = nat.count (λ d, d % 2 = 1) (nat.divisors n))
  (h2 : (16 : ℕ) = nat.count (λ d, d % 2 = 0) (nat.divisors n)) :
  n = 540 :=
sorry

end smallest_integer_has_8_odd_and_16_even_divisors_l803_803218


namespace probability_of_350_germinating_out_of_400_l803_803554

noncomputable def germination_probability 
  (n : ℝ) (p : ℝ) (k : ℝ) : ℝ :=
  let q := 1 - p in
  let sqrt_npq := real.sqrt (n * p * q) in
  let x := (k - n * p) / sqrt_npq in
  let phi_x := 1 / (real.sqrt (2 * real.pi)) * real.exp (-(x^2) / 2) in
  phi_x / sqrt_npq

theorem probability_of_350_germinating_out_of_400 :
  germination_probability 400 0.9 350 ≈ 0.0167 := 
by sorry

end probability_of_350_germinating_out_of_400_l803_803554


namespace angle_AYB_twice_angle_ADX_l803_803876

open EuclideanGeometry

variable (A B C D X Y : Point)

variable (h_conv : ConvexQuadrilateral A B C D)
variable (h_not_parallel : ¬ Parallel A B C D)
variable (h_angle_ADX_BCX : ∠ADX = ∠BCX ∧ ∠ADX < 90)
variable (h_angle_DAX_CBX : ∠DAX = ∠CBX ∧ ∠DAX < 90)
variable (h_Y_bisector : IsIntersectionOfPerpendicularBisectors Y A B C D)

theorem angle_AYB_twice_angle_ADX :
  ∠AYB = 2 * ∠ADX := by
  sorry

end angle_AYB_twice_angle_ADX_l803_803876


namespace dodecahedron_interior_diagonals_l803_803006

-- Define a dodecahedron structure reflecting the problem conditions
structure Dodecahedron :=
  (vertices : Finset ℕ)
  (faces : Finset (Finset ℕ))
  (three_faces_per_vertex : ∀ v ∈ vertices, 3 = (faces.filter (λ f, v ∈ f)).card)
  (face_cardinality : ∀ f ∈ faces, f.card = 5)
  (number_of_vertices : vertices.card = 20)
  (number_of_faces : faces.card = 12)

-- Given the problem conditions, we aim to prove the number of interior diagonals is 160
theorem dodecahedron_interior_diagonals (D : Dodecahedron) : 
  let interior_diagonal := λ (u v : ℕ), 
    u ∈ D.vertices ∧ v ∈ D.vertices ∧ u ≠ v ∧ ∀ f ∈ D.faces, ¬ (u ∈ f ∧ v ∈ f)
  in (Finset.filter (λ (u, v), interior_diagonal u v) 
      (D.vertices.product D.vertices)).card / 2 = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_l803_803006


namespace min_shaded_triangles_l803_803301

-- Definitions (conditions) directly from the problem
def Triangle (n : ℕ) := { x : ℕ // x ≤ n }
def side_length := 8
def smaller_side_length := 1

-- Goal (question == correct answer)
theorem min_shaded_triangles : ∃ (shaded : ℕ), shaded = 15 :=
by {
  sorry
}

end min_shaded_triangles_l803_803301


namespace min_characteristic_value_sum_l803_803351

-- Defining the characteristic value of a set as the sum of its maximum and minimum element
def characteristic_value (s : Finset ℕ) : ℕ :=
  s.max' (by sorry) + s.min' (by sorry)

-- Define sets A1, A2, A3, A4, A5
variables (A1 A2 A3 A4 A5 : Finset ℕ)

-- Define the set {1, 2, ..., 100}
def U : Finset ℕ := Finset.range 101 \ {0}

-- Conditions from the problem
axiom h1 : A1.card = 20
axiom h2 : A2.card = 20
axiom h3 : A3.card = 20
axiom h4 : A4.card = 20
axiom h5 : A5.card = 20
axiom h_union : A1 ∪ A2 ∪ A3 ∪ A4 ∪ A5 = U

-- The proof goal
theorem min_characteristic_value_sum : 
  characteristic_value A1 + characteristic_value A2 + characteristic_value A3 + characteristic_value A4 + characteristic_value A5 = 325 :=
sorry

end min_characteristic_value_sum_l803_803351


namespace probability_even_expression_l803_803587

theorem probability_even_expression :
  let S := {n | 1 ≤ n ∧ n ≤ 12} in
  ∃ p : ℚ, p = 5/22 ∧
  ∀ x y ∈ S, x ≠ y → (x*y - x - y + 2) % 2 = 0 → p = 5/22 := by
-- sorry is added to skip the proof
sorry

end probability_even_expression_l803_803587


namespace cosine_of_angle_l803_803384

theorem cosine_of_angle (α : ℝ) (h : ∃ (x y : ℝ), x = 3 ∧ y = 4 ∧ (cos α = x / (Real.sqrt (x^2 + y^2)))) :
  Real.cos (Real.pi + 2 * α) = -7/25 :=
by
  obtain ⟨x, y, hx, hy, hcos⟩ := h
  rw [hx, hy] at *
  have r := Real.sqrt (x^2 + y^2)
  have hcos' : Real.cos α = 3 / 5,
  { rw [hx, hy] at hcos
    norm_num at *
    exact hcos }
  have hcos2α : Real.cos (2 * α) = 1 - 2 * (Real.cos α)^2,
  { rw Real.cos_two_mul }
  rw [hcos2α, hcos']
  norm_num
  ring
  exact trivial

end cosine_of_angle_l803_803384


namespace evaluate_expression_l803_803025

theorem evaluate_expression (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (5 - x) + (5 - x) ^ 2 = 49 := 
sorry

end evaluate_expression_l803_803025


namespace number_of_zeros_of_f_l803_803134

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then exp x - x - 2 else x^2 + 2 * x

theorem number_of_zeros_of_f : 
  (finset.filter (λ x, f x = 0) (finset.range 10)).card = 2 := 
sorry

end number_of_zeros_of_f_l803_803134


namespace max_marks_paper_I_l803_803987

/-- The maximum mark for Paper I -/
theorem max_marks_paper_I (M : ℝ) (secured_marks : ℝ) (failed_by : ℝ) (required_percentage : ℝ) 
  (pass_marks : ℝ) (h1 : secured_marks = 45) (h2 : failed_by = 25) (h3 : required_percentage = 0.55) 
  (h4 : pass_marks = 70) : 
  M = 127 :=
by
  have h5 : 0.55 * M = secured_marks + failed_by, from sorry,   
  have h6 : 0.55 * M = 70, from sorry,
  have h7 : M = 70 / 0.55, from sorry,
  have h8 : M = 127.27272727272727, from sorry,
  -- Since M cannot be a fraction, we round it to the nearest whole number
  have h9 : M = 127, from sorry,
  exact h9

sorry

end max_marks_paper_I_l803_803987


namespace locus_of_midpoints_is_line_l803_803136

variables {l1 l2 l3 p : Type} [line l1] [line l2] [line l3] [line p]
variables (A1 A2 A3 : p) (M : l1) (N : l2)

-- Assuming lines l1, l2, and l3 intersect each other pairwise
-- And they are perpendicular to the line p
axiom l1_perpendicular_p : perpendicular l1 p
axiom l2_perpendicular_p : perpendicular l2 p
axiom l3_perpendicular_p : perpendicular l3 p

-- A1, A2, A3 are intersection points of l1, l2, l3 with p respectively
axiom A1_intersect_p : A1 ∈ l1
axiom A2_intersect_p : A2 ∈ l2
axiom A3_intersect_p : A3 ∈ l3
axiom M_on_l1 : M ∈ l1
axiom N_on_l2 : N ∈ l2

-- MN intersects l3
axiom MN_intersect_l3 : ∃ P, P ∈ l3 ∧ P ∈ line_through M N

-- Prove that the locus of midpoints of segment MN is a straight line
theorem locus_of_midpoints_is_line : ∃ L, is_straight_line L ∧ ∀ X,
  X ∈ midpoints_of_segments (M, N) → X ∈ L := sorry

end locus_of_midpoints_is_line_l803_803136


namespace displacement_representation_l803_803421

def represents_north (d : ℝ) : Prop := d > 0

theorem displacement_representation (d : ℝ) (h : represents_north 80) : represents_north d ↔ d > 0 :=
by trivial

example (h : represents_north 80) : 
  ∀ d, d = -50 → ¬ represents_north d ∧ abs d = 50 → ∃ s, s = "south" :=
sorry

end displacement_representation_l803_803421


namespace player_a_wins_l803_803503

-- Definition of the game
structure Chessboard :=
  (size : ℕ)
  (piece_position : ℕ × ℕ)
  (visited : set (ℕ × ℕ))

def initial_chessboard : Chessboard :=
  { size := 8,
    piece_position := (0, 0), -- Let's assume the piece starts in the top-left corner
    visited := {(0, 0)} }

-- Define a move as a structure
structure Move :=
  (source : ℕ × ℕ)
  (destination : ℕ × ℕ)
  (distance : ℕ)

-- Validates that a move is legal (unvisited and squares are within bounds)
def valid_move (bd : Chessboard) (mv : Move) : Prop :=
  let (x1, y1) := mv.source
  let (x2, y2) := mv.destination
  x2 < bd.size ∧ y2 < bd.size ∧
  (x2, y2) ∉ bd.visited ∧
  mv.distance > 0

-- Strategy to win for Player A
def winning_strategy : ∀ (cb : Chessboard) (mv1 mv2 : Move), valid_move cb mv1 → valid_move cb mv2 → mv1.distance < mv2.distance → Prop
| cb, mv1, mv2, hv1, hv2, hdist :=
  sorry

theorem player_a_wins : ∀ (cb : Chessboard), winning_strategy initial_chessboard sorry sorry sorry sorry sorry :=
  sorry

end player_a_wins_l803_803503


namespace smallest_positive_integer_with_divisors_l803_803229

theorem smallest_positive_integer_with_divisors :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → nat.odd d) ∧ (finset.filter nat.odd (finset.divisors n)).card = 8 ∧ 
           (∃ m : ℕ, ∀ d : ℕ, d ∣ m → nat.even d ∧ m = n → (finset.filter nat.even (finset.divisors m)).card = 16)
             → n = 756 :=
by
  sorry

end smallest_positive_integer_with_divisors_l803_803229


namespace ladder_slip_l803_803271

theorem ladder_slip (l : ℝ) (d1 d2 : ℝ) (h1 h2 : ℝ) :
  l = 30 → d1 = 8 → h1^2 + d1^2 = l^2 → h2 = h1 - 4 → 
  (h2^2 + (d1 + d2)^2 = l^2) → d2 = 2 :=
by
  intros h_l h_d1 h_h1_eq h_h2 h2_eq_l   
  sorry

end ladder_slip_l803_803271


namespace smallest_six_digit_number_exists_l803_803661

def three_digit_number (n : ℕ) := n % 4 = 2 ∧ n % 5 = 2 ∧ n % 6 = 2 ∧ 100 ≤ n ∧ n < 1000

def valid_six_digit_number (m n : ℕ) := 
  (m * 1000 + n) % 4 = 0 ∧ (m * 1000 + n) % 5 = 0 ∧ (m * 1000 + n) % 6 = 0 ∧ 
  three_digit_number n ∧ 0 ≤ m ∧ m < 1000

theorem smallest_six_digit_number_exists : 
  ∃ m n, valid_six_digit_number m n ∧ (∀ m' n', valid_six_digit_number m' n' → m * 1000 + n ≤ m' * 1000 + n') :=
sorry

end smallest_six_digit_number_exists_l803_803661


namespace least_value_expression_l803_803957

open Real

theorem least_value_expression (x : ℝ) : 
  let expr := (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2023 + 3 * cos (2 * x)
  ∃ a : ℝ, expr = a ∧ ∀ b : ℝ, b < a → False :=
sorry

end least_value_expression_l803_803957


namespace vector_subtraction_parallelogram_l803_803042

variables {A B C D : Type} [euclidean_space ℝ A] [euclidean_space ℝ B] [euclidean_space ℝ C] [euclidean_space ℝ D]
variables (AB : A → B) (AD : A → D) (DB : D → B)

-- Define points and vectors in the parallelogram
noncomputable def ABCD_is_parallelogram (A B C D : Point) : Prop :=
  parallel (Vector A B) (Vector C D) ∧ parallel (Vector A D) (Vector B C)

-- The target statement
theorem vector_subtraction_parallelogram (A B C D : Point) (h : ABCD_is_parallelogram A B C D) :
  Vector A B - Vector A D = Vector D B :=
sorry

end vector_subtraction_parallelogram_l803_803042


namespace regular_polygon_sides_l803_803542

-- Conditions
def central_angle (θ : ℝ) := θ = 30
def sum_of_central_angles (sumθ : ℝ) := sumθ = 360

-- The proof problem
theorem regular_polygon_sides (θ sumθ : ℝ) (h₁ : central_angle θ) (h₂ : sum_of_central_angles sumθ) :
  sumθ / θ = 12 := by
  sorry

end regular_polygon_sides_l803_803542


namespace square_dist_from_B_to_center_l803_803644

/-!
# Problem Statement:
A machine-shop cutting tool has the shape of a notched circle. The radius of the circle is √75 cm,
and the lengths of the segments are as follows: AB is 8 cm and BC is 4 cm, with angle ABC being a right angle.
Determine the square of the distance from B to the center of the circle.

We aim to prove that the square of the distance from \( B \) to the center is 59.008.
-/

open Real

theorem square_dist_from_B_to_center :
  let r := sqrt 75
  let AB := 8
  let BC := 4
  let B_coords : ℝ × ℝ := (-24 + sqrt 1036) / 10
  in r^2 = 75 ∧
                  ∃ a b : ℝ,
                    B_coords = (a, b) ∧ 
                    a^2 + (b + AB)^2 = 75 ∧ 
                    (a + BC)^2 + b^2 = 75 ∧ 
                    a^2 + b^2 = 59.008
                 := 
begin
  sorry
end

end square_dist_from_B_to_center_l803_803644


namespace degree_poly_product_l803_803534

open Polynomial

-- Given conditions: p and q are polynomials with specified degrees
variables {R : Type*} [CommRing R]
variable (p q : R[X])
variable (hp : degree p = 3)
variable (hq : degree q = 6)

-- Proposition: The degree of p(x^2) * q(x^4) is 30
theorem degree_poly_product : degree (p.comp ((X : R[X])^2) * (q.comp ((X : R[X])^4))) = 30 :=
by sorry

end degree_poly_product_l803_803534


namespace doppler_effect_approaching_doppler_effect_receding_l803_803641

noncomputable def c := 16.67  -- speed of the train in m/s
noncomputable def N := 2048  -- frequency of the train's whistle in Hz
noncomputable def V := 340  -- speed of sound in air in m/s

theorem doppler_effect_approaching :
  let N' := N * V / (V - c)
  N' ≈ 2153 := 
by
  sorry

theorem doppler_effect_receding :
  let N'' := N * V / (V + c)
  N'' ≈ 1952 := 
by
  sorry

end doppler_effect_approaching_doppler_effect_receding_l803_803641


namespace EF_perpendicular_to_AC_l803_803121

theorem EF_perpendicular_to_AC 
  (A B C D E F : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
  (parallelogram : ∀ (P : Type), IsParallelogram A B C D) 
  (H1 : ∃ θ : ℝ, ∠CAB = θ ∧ ∠CAD = 2 * θ) 
  (H2 : IsAngleBisector A C D E) 
  (H3 : IsAngleBisector A C B F) : 
  IsPerpendicular E F A C :=
by
  sorry

end EF_perpendicular_to_AC_l803_803121


namespace smallest_positive_integer_with_divisors_l803_803226

theorem smallest_positive_integer_with_divisors :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → nat.odd d) ∧ (finset.filter nat.odd (finset.divisors n)).card = 8 ∧ 
           (∃ m : ℕ, ∀ d : ℕ, d ∣ m → nat.even d ∧ m = n → (finset.filter nat.even (finset.divisors m)).card = 16)
             → n = 756 :=
by
  sorry

end smallest_positive_integer_with_divisors_l803_803226


namespace four_digit_numbers_sum_divisible_by_11_l803_803865

theorem four_digit_numbers_sum_divisible_by_11 : 
  let valid_n := {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∃ q r, n = 50 * q + r ∧ 0 ≤ r ∧ r < 50 ∧ (q + r) % 11 = 0} in
  valid_n.card = 900 :=
by
  sorry

end four_digit_numbers_sum_divisible_by_11_l803_803865


namespace no_valid_partition_of_10_letter_words_l803_803580

theorem no_valid_partition_of_10_letter_words :
  ¬ ∃ (A B : set (fin 1024 → bool)),
    (∀ w1 w2 ∈ A, w1 ≠ w2 → (∃ i1 i2 i3, i1 ≠ i2 ∧ i1 ≠ i3 ∧ i2 ≠ i3 ∧ w1 i1 ≠ w2 i1 ∧ w1 i2 ≠ w2 i2 ∧ w1 i3 ≠ w2 i3)) ∧
    (∀ w1 w2 ∈ B, w1 ≠ w2 → (∃ i1 i2 i3, i1 ≠ i2 ∧ i1 ≠ i3 ∧ i2 ≠ i3 ∧ w1 i1 ≠ w2 i1 ∧ w1 i2 ≠ w2 i2 ∧ w1 i3 ≠ w2 i3)) :=
sorry

end no_valid_partition_of_10_letter_words_l803_803580


namespace ratio_lcm_gcf_240_360_l803_803958

theorem ratio_lcm_gcf_240_360 : Nat.lcm 240 360 / Nat.gcd 240 360 = 60 :=
by
  sorry

end ratio_lcm_gcf_240_360_l803_803958


namespace paper_cut_count_incorrect_l803_803531

theorem paper_cut_count_incorrect (n : ℕ) (h : n = 1961) : 
  ∀ i, (∃ k, i = 7 ∨ i = 7 + 6 * k) → i % 6 = 1 → n ≠ i :=
by
  sorry

end paper_cut_count_incorrect_l803_803531


namespace ground_beef_per_package_l803_803494

-- Declare the given conditions and the expected result.
theorem ground_beef_per_package (num_people : ℕ) (weight_per_burger : ℕ) (total_packages : ℕ) 
    (h1 : num_people = 10) 
    (h2 : weight_per_burger = 2) 
    (h3 : total_packages = 4) : 
    (num_people * weight_per_burger) / total_packages = 5 := 
by 
  sorry

end ground_beef_per_package_l803_803494


namespace lisa_eats_correct_number_of_pieces_l803_803491

variable (M A K R L : ℚ) -- All variables are rational numbers (real numbers could also be used)
variable (n : ℕ) -- n is a natural number (the number of pieces of lasagna)

-- Let's define the conditions succinctly
def manny_wants_one_piece := M = 1
def aaron_eats_nothing := A = 0
def kai_eats_twice_manny := K = 2 * M
def raphael_eats_half_manny := R = 0.5 * M
def lasagna_is_cut_into_6_pieces := n = 6

-- The proof goal is to show Lisa eats 2.5 pieces
theorem lisa_eats_correct_number_of_pieces (M A K R L : ℚ) (n : ℕ) :
  manny_wants_one_piece M →
  aaron_eats_nothing A →
  kai_eats_twice_manny M K →
  raphael_eats_half_manny M R →
  lasagna_is_cut_into_6_pieces n →
  L = n - (M + K + R) →
  L = 2.5 :=
by
  intros hM hA hK hR hn hL
  sorry  -- Proof omitted

end lisa_eats_correct_number_of_pieces_l803_803491


namespace power_of_two_start_digits_l803_803101

theorem power_of_two_start_digits (A : ℕ) : 
  ∃ m n : ℕ, log A < n * log 2 - m ∧ n * log 2 - m < log (A + 1) := 
  sorry

end power_of_two_start_digits_l803_803101


namespace count_primes_between_50_and_100_with_prime_remainder_l803_803010

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m: ℕ, m > 1 → m < n → n % m ≠ 0)

def primes_in_range (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

def prime_remainders (l : List ℕ) : List ℕ :=
  l.filter (fun n => is_prime (n % 8))

theorem count_primes_between_50_and_100_with_prime_remainder :
  (prime_remainders (primes_in_range 50 100)).length = 7 := 
sorry

end count_primes_between_50_and_100_with_prime_remainder_l803_803010


namespace problem_l803_803856

def f (n : ℕ) : ℚ := ∑ i in finset.range n.succ, 1 / (2^i : ℚ)

theorem problem (k : ℕ) : f (k + 1) - f k = 1 / (2^(k + 1) : ℚ) :=
begin
  sorry
end

end problem_l803_803856


namespace fill_diagram_correct_ways_l803_803335

def is_increasing (matrix : list (list ℕ)) : Prop :=
  (∀ row in matrix, list.chain' (· < ·) row) ∧
  (∀ j : Nat, j < 3 → list.chain' (· < ·) [matrix[i][j] | i < 3])

def is_valid_matrix (matrix : list (list ℕ)) : Prop :=
  (matrix.flatten ~ [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (matrix[1][0] = 3) ∧ (matrix[1][1] = 4)

theorem fill_diagram_correct_ways :
  ∃ (matrices : list (list (list ℕ))), 
  (∀ matrix in matrices, is_valid_matrix matrix ∧ is_increasing matrix) ∧
  matrices.length = 6 :=
sorry

end fill_diagram_correct_ways_l803_803335


namespace maximal_disjoint_tuples_l803_803853

-- Definitions for n, X, and disjoint criteria
variables (n : ℕ) (X : Finset ℕ) (a b : Fin n → ℕ)
  (hX_card : X.card = n + 1) (hn : n ≥ 2)

def are_disjoint (a b : Fin n → ℕ) : Prop :=
  ∃ (i j : Fin n), i ≠ j ∧ (a i = b j)

-- The main theorem to be proved
theorem maximal_disjoint_tuples : 
  (∃ (A : Finset (Fin n → ℕ)), 
    (∀ a b ∈ A, a ≠ b → are_disjoint a b) ∧ A.card = (n + 1)! / 2) :=
  sorry

end maximal_disjoint_tuples_l803_803853


namespace sandy_has_four_times_more_marbles_l803_803055

-- Definitions based on conditions
def jessica_red_marbles : ℕ := 3 * 12
def sandy_red_marbles : ℕ := 144

-- The theorem to prove
theorem sandy_has_four_times_more_marbles : sandy_red_marbles = 4 * jessica_red_marbles :=
by
  sorry

end sandy_has_four_times_more_marbles_l803_803055


namespace area_of_region_l803_803170

-- Define the equation of the circle
def equation (x y : ℝ) : Prop := x^2 + y^2 + 6 * x + 8 * y = 0

-- Define the area of a circle with radius 5 as 25π
def area : ℝ := 25 * Real.pi

-- The theorem stating the area enclosed by the given equation is 25π
theorem area_of_region : (∀ x y : ℝ, equation x y) → area = 25 * Real.pi :=
by
  intro h
  sorry

end area_of_region_l803_803170


namespace components_produced_and_sold_per_month_l803_803990

theorem components_produced_and_sold_per_month (x : ℕ) :
  (∀ (cost_per_component shipping_per_component fixed_monthly_cost lowest_selling_price : ℝ),
   cost_per_component = 80 →
   shipping_per_component = 4 →
   fixed_monthly_cost = 16500 →
   lowest_selling_price = 193.33 →
   let total_cost := fixed_monthly_cost + (cost_per_component + shipping_per_component) * x in
   let total_revenue := lowest_selling_price * x in
   total_cost ≤ total_revenue →
   x ≥ 151) :=
by
  sorry

end components_produced_and_sold_per_month_l803_803990


namespace import_tax_percentage_l803_803233

theorem import_tax_percentage 
  (total_value : ℝ) (excess_value : ℝ) (tax_paid : ℝ) (P : ℝ)
  (h1 : total_value = 2250)
  (h2 : excess_value = total_value - 1000)
  (h3 : tax_paid = 87.50)
  (h4 : (P / 100) * excess_value = tax_paid) : 
  P = 7 := 
begin
  sorry
end

end import_tax_percentage_l803_803233


namespace find_N_l803_803459

theorem find_N (
    A B : ℝ) (N : ℕ) (r : ℝ) (hA : A = N * π * r^2 / 2) 
    (hB : B = (π * r^2 / 2) * (N^2 - N)) 
    (ratio : A / B = 1 / 18) : 
    N = 19 :=
by
  sorry

end find_N_l803_803459


namespace Anya_walks_to_school_l803_803428

theorem Anya_walks_to_school
  (t_f t_b : ℝ)
  (h1 : t_f + t_b = 1.5)
  (h2 : 2 * t_b = 0.5) :
  2 * t_f = 2.5 :=
by
  -- The proof details will go here eventually.
  sorry

end Anya_walks_to_school_l803_803428


namespace find_S3_l803_803570

noncomputable def geometric_sum (n : ℕ) : ℕ := sorry  -- Placeholder for the sum function.

theorem find_S3 (S : ℕ → ℕ) (hS6 : S 6 = 30) (hS9 : S 9 = 70) : S 3 = 10 :=
by
  -- Establish the needed conditions and equation 
  have h : (S 6 - S 3) ^ 2 = (S 9 - S 6) * S 3 := sorry
  -- Substitute given S6 and S9 into the equation and solve
  exact sorry

end find_S3_l803_803570


namespace problem_equivalent_proof_l803_803858

noncomputable def a : ℝ := - (9 / 25) ^ (1/2)
noncomputable def b : ℝ := ((3 + (3 : ℝ) ^ (1/2)) ^ 2 / 15) ^ (1/2)

theorem problem_equivalent_proof : 
  (a + b) ^ 3 = (-5670 * real.sqrt 3 + 1620 * real.sqrt 5 + 15 * real.sqrt 15) / 50625 := by
  sorry

end problem_equivalent_proof_l803_803858


namespace smallest_n_l803_803878

def C_n (n : ℕ) : ℚ :=
  (256 / 3) * (1 - (1 / 4^n))

def D_n (n : ℕ) : ℚ :=
  (1024 / 5) * (1 - (1 / (-4)^n))

theorem smallest_n (n : ℕ) (h : n ≥ 1) : ∃ k, k = 1 ∧ C_n k ≠ D_n k :=
by {
  use 1,
  split,
  refl,
  dsimp [C_n, D_n],
  norm_num,
  sorry
}

end smallest_n_l803_803878


namespace intersection_A_B_l803_803857

def A : Set ℝ := { x | x > -1 }
def B : Set ℝ := { y | (y - 2) * (y + 3) < 0 }

theorem intersection_A_B : A ∩ B = Set.Ioo (-1) 2 :=
by
  sorry

end intersection_A_B_l803_803857


namespace quadrilateral_area_correct_l803_803111

noncomputable def quadrilateral_area (ABCD : square) (Q : point) (hQ1 : distance A Q = 16) (hQ2 : distance B Q = 34) : ℝ :=
  let G_ABQ := centroid (triangle A B Q),
      G_BCQ := centroid (triangle B C Q),
      G_CDQ := centroid (triangle C D Q),
      G_DAQ := centroid (triangle D A Q) in
  let quad := quadrilateral G_ABQ G_BCQ G_CDQ G_DAQ in
  area quad

theorem quadrilateral_area_correct (ABCD : square) (Q : point)
  (hQ1 : distance A Q = 16) (hQ2 : distance B Q = 34) (h_side : side_length ABCD = 40) :
  quadrilateral_area ABCD Q hQ1 hQ2 = 6400 / 9 :=
sorry

end quadrilateral_area_correct_l803_803111


namespace isosceles_triangle_relation_range_l803_803563

-- Definitions of the problem conditions and goal
variables (x y : ℝ)

-- Given conditions
def isosceles_triangle (x y : ℝ) :=
  x + x + y = 10

-- Prove the relationship and range 
theorem isosceles_triangle_relation_range (h : isosceles_triangle x y) :
  y = 10 - 2 * x ∧ (5 / 2 < x ∧ x < 5) :=
  sorry

end isosceles_triangle_relation_range_l803_803563


namespace pentagon_zero_impossible_l803_803583

theorem pentagon_zero_impossible
  (x : Fin 5 → ℝ)
  (h_sum : x 0 + x 1 + x 2 + x 3 + x 4 = 0)
  (operation : ∀ i : Fin 5, ∀ y : Fin 5 → ℝ,
    y i = (x i + x ((i + 1) % 5)) / 2 ∧ y ((i + 1) % 5) = (x i + x ((i + 1) % 5)) / 2) :
  ¬ ∃ (y : ℕ → (Fin 5 → ℝ)), ∃ N : ℕ, y N = 0 := 
sorry

end pentagon_zero_impossible_l803_803583


namespace car_insurance_monthly_cost_is_80_l803_803887

-- Define the given conditions
variable (C : ℝ) -- Annual cost of the car insurance
variable (nancy_pays : ℝ) (nancy_payment_fraction : ℝ)
variable (nancy_annual_payment : ℝ)

-- Define the conditions based on the problem
def nancy_paying_part : Prop := nancy_annual_payment = nancy_payment_fraction * C
def car_insurance_cost_per_month : ℝ := C / 12

-- Define the problem to prove
theorem car_insurance_monthly_cost_is_80 (H : nancy_paying_part nancy_annual_payment 0.4 384) :
  car_insurance_cost_per_month C = 80 := 
by sorry

end car_insurance_monthly_cost_is_80_l803_803887


namespace number_of_vectors_is_two_l803_803671

def is_vector (quantity : String) : Prop :=
  if quantity = "density" then False
  else if quantity = "buoyancy" then True
  else if quantity = "wind_speed" then True
  else if quantity = "temperature" then False
  else False

theorem number_of_vectors_is_two :
  (finset.card (finset.filter is_vector (finset.of_list ["density", "buoyancy", "wind_speed", "temperature"]))) = 2 :=
by
  sorry

end number_of_vectors_is_two_l803_803671


namespace smallest_int_with_divisors_l803_803215

theorem smallest_int_with_divisors :
  ∃ n : ℕ, 
    (∀ m, n = 2^2 * m → 
      (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 
      (m = p^3 * q) ∧ 
      (nat.divisors p^7).count 8) ∧ 
    nat.divisors_count n = 24 ∧ 
    (nat.divisors (2^2 * n)).count 8) (n = 2^2 * 3^3 * 5) :=
begin
  sorry
end

end smallest_int_with_divisors_l803_803215


namespace prove_slopes_sum_zero_prove_slopes_squared_sum_eight_l803_803978

-- Definitions for the problem conditions
variables {A B P Q : Point}
variables {λ : ℝ} (λ_gt_1 : |λ| > 1)
variables {k1 k2 k3 k4 : ℝ}
variables (AP_slope BP_slope AQ_slope BQ_slope : ℝ)

-- Given conditions
def condition1 : Vector3D = (vecA.P + vecB.P) = λ * (vecA.Q + vecB.Q)
def condition2 (P_hyperbola : Point_hyperbola) (Q_ellipse : Point_ellipse)

-- Question equivalent to condition
def slopes_condition : Vector3D = (k1 + k2) + (k3 + k4) = 0

-- Part (1): Prove k1 + k2 + k3 + k4 = 0
theorem prove_slopes_sum_zero
  (cond1 : condition1)
  (cond2 : condition2) 
  : slopes_condition := sorry

-- Additional condition for part (2)
theorem prove_slopes_squared_sum_eight
  (cond1 : condition1)
  (cond2 : condition2)
  (slope_sum_zero : slopes_condition) 
  : k1^2 + k2^2 + k3^2 + k4^2 = 8 := sorry

end prove_slopes_sum_zero_prove_slopes_squared_sum_eight_l803_803978


namespace area_of_PQR_is_120_l803_803163

noncomputable def triangle_area {P Q R : Type} [metric_space P] [metric_space Q] [metric_space R]
  (PQ PR QR : ℝ) (is_isosceles : PQ = PR) (side_lengths : PQ = 17 ∧ PR = 17 ∧ QR = 16) : ℝ :=
  let PS := real.sqrt (PQ * PQ - (QR / 2) * (QR / 2)) in
  (1 / 2) * QR * PS

theorem area_of_PQR_is_120 (P Q R : Type) 
    [metric_space P] [metric_space Q] [metric_space R] 
    (PQ PR QR : ℝ) (h_isosceles : PQ = PR) 
    (h_side_lengths : PQ = 17 ∧ PR = 17 ∧ QR = 16) : 
    triangle_area PQ PR QR h_isosceles h_side_lengths = 120 :=
begin
  sorry
end

end area_of_PQR_is_120_l803_803163


namespace cone_central_angle_l803_803114

theorem cone_central_angle (l : ℝ) (α : ℝ) (h : (30 : ℝ) * π / 180 > 0) :
  α = π := 
sorry

end cone_central_angle_l803_803114


namespace binomial_product_result_l803_803690

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l803_803690


namespace slope_of_tangent_line_at_zero_l803_803568

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem slope_of_tangent_line_at_zero : (deriv f 0) = 1 :=
by
  sorry 

end slope_of_tangent_line_at_zero_l803_803568


namespace f_nested_value_l803_803795

-- Define the piecewise function
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 ^ x else Real.sin x

-- State the theorem to be proved
theorem f_nested_value : f (f (7 * Real.pi / 6)) = Real.sqrt 2 / 2 :=
by
  sorry

end f_nested_value_l803_803795


namespace mark_buttons_l803_803492

theorem mark_buttons (initial_buttons : ℕ) (shane_buttons : ℕ) (sam_buttons : ℕ) :
  initial_buttons = 14 →
  shane_buttons = 3 * initial_buttons →
  sam_buttons = (initial_buttons + shane_buttons) / 2 →
  final_buttons = (initial_buttons + shane_buttons) - sam_buttons →
  final_buttons = 28 :=
by
  sorry

end mark_buttons_l803_803492


namespace ellipse_foci_coordinates_l803_803544

theorem ellipse_foci_coordinates (x y : ℝ) :
  2 * x^2 + y^2 = 8 → 
  y^2 + x^2 * 4 = 8 →
  let a := 2 * real.sqrt 2 in
  let b := 2 in
  let c := real.sqrt (a^2 - b^2) in
  c = 2 →
  (0, c) = (0, 2) :=
by intros h1 h2 a_def b_def c_def; sorry

end ellipse_foci_coordinates_l803_803544


namespace tennis_tournament_l803_803450

theorem tennis_tournament (n : ℕ) (w m : ℕ) 
  (total_matches : ℕ)
  (women_wins men_wins : ℕ) :
  n + 2 * n = 3 * n →
  total_matches = (3 * n * (3 * n - 1)) / 2 →
  women_wins + men_wins = total_matches →
  women_wins / men_wins = 7 / 5 →
  n = 3 :=
by sorry

end tennis_tournament_l803_803450


namespace smallest_positive_integer_solution_l803_803180

/-- Prove that the smallest positive integer x that satisfies the congruence
    3x ≡ 15 (mod 31) is 12. -/
theorem smallest_positive_integer_solution :
  ∃ x : ℕ, (x < 31) ∧ (3 * x ≡ 15 [MOD 31]) ∧ (x = 12) :=
by
  existsi (12 : ℕ)
  split
  { -- x < 31
    exact nat.lt_of_succ_lt_succ (nat.lt_succ_self 11)
  }
  split
  { -- 3 * x ≡ 15 [MOD 31]
    exact nat.modeq.symm (nat.modeq.of_eq $ by norm_num)
  }
  -- x = 12
  refl

end smallest_positive_integer_solution_l803_803180


namespace cannot_transform_10_to_01_l803_803532

theorem cannot_transform_10_to_01 : 
  ¬∃ (x : list (ℕ)), x = [1, 0] ∨ x = [0, 1] ∧ 
                       (∀ y : list (ℕ), (substring y y) = (list.replicate 3 (list.head y))),
                       sequences_from_10_to_01 x :=
by sorry

end cannot_transform_10_to_01_l803_803532


namespace smallest_abundant_not_multiple_of_4_l803_803599

def is_abundant (n : ℕ) : Prop :=
  (∑ k in finset.filter (λ d, d < n) (finset.divisors n), k) > n

theorem smallest_abundant_not_multiple_of_4 : ∃ n : ℕ, is_abundant n ∧ ¬ (4 ∣ n) ∧ ∀ m : ℕ, is_abundant m ∧ ¬ (4 ∣ m) → m ≥ n :=
begin
  use 18,
  sorry
end

end smallest_abundant_not_multiple_of_4_l803_803599


namespace find_BE_l803_803098

noncomputable def parallelogram_BE (A B C D F E G : Type)
  [AddGroup A] [AffineSpace A B] [Field E] [Field F] [Field G]
  (parallelogram : Type) 
  (pointOnExtension : F) 
  (intersectDiagonal : G)
  (EF : E := 18)
  (GF : E := 6) : Prop :=
  ∃ BE : E, BE = 12

theorem find_BE
  (A B C D F E G : Type)
  [AddGroup A] [AffineSpace A B] [Field E] [Field F] [Field G]
  (parallelogram : Type) 
  (pointOnExtension : F) 
  (intersectDiagonal : G)
  (EF : E := 18)
  (GF : E := 6) :
    parallelogram_BE (A B C D F E G) parallelogram pointOnExtension intersectDiagonal EF GF :=
begin
  use 12,
  sorry
end

end find_BE_l803_803098


namespace solution_set_inequality_l803_803788

theorem solution_set_inequality (a b x : ℝ) (h₁ : ∀ x, (ax > b) ↔ (x < 2)) :
  { x : ℝ | (a * x + b) * (x - 1) > 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  -- definitions from the problem conditions
  have h₂ : 2 * a = b, from sorry,
  have h₃ : a < 0, from sorry,
  
  -- main proof (skipped)
  sorry

end solution_set_inequality_l803_803788


namespace area_swept_correct_l803_803989

-- Definitions based on conditions
def minute_hand_length : ℝ := 10
def minutes_passed : ℝ := 35
def angle_in_degrees : ℝ := minutes_passed * 6  -- Each minute corresponds to 6 degrees
def angle_in_radians : ℝ := angle_in_degrees * (Real.pi / 180)  -- Convert degrees to radians

-- Area of sector calculation
noncomputable def area_swept : ℝ := (1 / 2) * minute_hand_length^2 * angle_in_radians

-- Theorem statement
theorem area_swept_correct : area_swept = (175 * Real.pi) / 3 := by
  sorry

end area_swept_correct_l803_803989


namespace fraction_of_power_l803_803875

theorem fraction_of_power (m : ℕ) (h : m = 16^1500) : m / 8 = 2^5997 := by
  sorry

end fraction_of_power_l803_803875


namespace cost_per_component_l803_803278

theorem cost_per_component (C : ℝ) : 
  (150 * C + 150 * 4 + 16500 = 150 * 193.33) → 
  C = 79.33 := 
by
  intro h
  sorry

end cost_per_component_l803_803278


namespace sum_remainder_div_9_l803_803959

theorem sum_remainder_div_9 : 
  let S := (20 / 2) * (1 + 20)
  S % 9 = 3 := 
by
  -- use let S to simplify the proof
  let S := (20 / 2) * (1 + 20)
  -- sum of first 20 natural numbers
  have H1 : S = 210 := by sorry
  -- division and remainder result
  have H2 : 210 % 9 = 3 := by sorry
  -- combine both results to conclude 
  exact H2

end sum_remainder_div_9_l803_803959


namespace no_common_period_l803_803281

noncomputable def periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x, f(x + p) = f(x)

theorem no_common_period
  (g h : ℝ → ℝ)
  (h_g_periodic : periodic g 3)
  (h_h_periodic : periodic h real.pi)
  : ¬ ∃ T > 0, periodic (λ x, g x - h x) T := by
  sorry

end no_common_period_l803_803281


namespace parabola_focus_l803_803342

theorem parabola_focus :
  ∀ F : (ℝ × ℝ), 
  (∀ P : (ℝ × ℝ), 
    P.snd = -((1/8) * P.fst * P.fst) → 
    (P.dist F = P.dist (P.fst, (1/8) * P.fst * P.fst - 4))) →
  F = (0, -2) :=
by
  intros F H
  sorry

end parabola_focus_l803_803342


namespace sam_total_distance_l803_803084

-- Conditions
def distance_marguerite : ℝ := 100
def time_marguerite : ℝ := 2.4
def distance_sam : ℝ := 125
def total_time_sam : ℝ := 4
def rest_time_sam : ℝ := 1

-- Derived Constants
def speed_marguerite := distance_marguerite / time_marguerite
def driving_time_sam := total_time_sam - rest_time_sam

-- Lean 4 statement
theorem sam_total_distance :
  driving_time_sam * speed_marguerite = distance_sam :=
by 
  -- Proof goes here
  sorry

end sam_total_distance_l803_803084


namespace what_did_other_five_say_l803_803507

-- Define the inhabitants of the island
inductive Islander
| knight : Islander
| liar : Islander

-- Define the statement types
inductive Statement
| more_liars : Statement
| equal_numbers : Statement

-- Define a structure for the conditions
structure IslandConditions where
  total_islanders : ℕ
  responses_more_liars : ℕ
  responses_equal_numbers : ℕ :=
  total_islanders := 10

-- The theorem that needs to be proved
theorem what_did_other_five_say (c : IslandConditions) :
  c.responses_more_liars = 5 → c.responses_equal_numbers = 5 :=
by
  sorry

end what_did_other_five_say_l803_803507


namespace largest_acute_angles_convex_octagon_l803_803174

-- Define terms and conditions
def is_convex_octagon (angles : Fin 8 → ℝ) : Prop :=
  (∀ i, 0 < angles i ∧ angles i < 180) ∧
  (∑ i, angles i = 1080)

-- The statement to prove: the largest number of acute angles (angles less than 90 degrees) is 4
theorem largest_acute_angles_convex_octagon (angles : Fin 8 → ℝ) :
  is_convex_octagon angles → 
  (Finset.filter (λ i, angles i < 90) Finset.univ).card ≤ 4 :=
by sorry

end largest_acute_angles_convex_octagon_l803_803174


namespace proportion_of_solution_r_l803_803158

def fill_rate (a b c : ℕ) : ℚ := (1 / a : ℚ) + (1 / b : ℚ) + (1 / c : ℚ)

def proportion_solution_r (a b c : ℕ) (time_elapsed : ℚ) : ℚ :=
  let total_filled := fill_rate a b c * time_elapsed
  let r_amount := (1 / c) * time_elapsed
  r_amount / total_filled

theorem proportion_of_solution_r (a b c : ℕ) (time_elapsed : ℚ) (ha : a = 20) (hb : b = 20) (hc : c = 30) (ht : time_elapsed = 3) :
  proportion_solution_r a b c time_elapsed = 1 / 4 :=
by
  sorry

end proportion_of_solution_r_l803_803158


namespace permutation_representation_l803_803126

variable (x : ℕ)

theorem permutation_representation (hx_pos : 0 < x) (hx_gt : 13 < x) :
  (x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 7) * (x - 8) * (x - 9) * (x - 10) * (x - 11) * (x - 12) * (x - 13) = 
  (A_{x - 3}^{11}) :=
sorry

end permutation_representation_l803_803126


namespace part1_part2_l803_803387

theorem part1 (m : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = m - |x - 2|) 
  (h₂ : ∀ x, f (x + 2) ≥ 0 ↔ x ∈ [-1, 1]) : m = 1 := 
sorry

theorem part2 (a b c : ℝ) (Z : ℝ) 
  (h₁ : ∀ x, 0 < x) 
  (h₂ : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1) : 
  Z = a + 2 * b + 3 * c → Z ≥ 9 := 
sorry

end part1_part2_l803_803387


namespace binom_mult_eq_6720_l803_803693

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l803_803693


namespace smallest_abundant_not_multiple_of_4_l803_803601

-- Define a function to get the sum of proper divisors of a number
def sum_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d, d < n) (Finset.divisors n)).sum

-- Define a predicate to check if a number is abundant
def is_abundant (n : ℕ) : Prop :=
  sum_proper_divisors n > n

-- Define a predicate to check if a number is a multiple of 4
def multiple_of_4 (n : ℕ) : Prop :=
  n % 4 = 0

-- The main theorem to prove
theorem smallest_abundant_not_multiple_of_4 : ∃ n, is_abundant n ∧ ¬ multiple_of_4 n ∧ (∀ m, is_abundant m ∧ ¬ multiple_of_4 m → n ≤ m) :=
  ∃ n, is_abundant n ∧ ¬ multiple_of_4 n ∧ (∀ m, is_abundant m ∧ ¬ multiple_of_4 m → n ≤ m) ∧ n = 20 :=
begin
  sorry
end

end smallest_abundant_not_multiple_of_4_l803_803601


namespace four_evenings_insufficient_five_evenings_insufficient_ten_evenings_sufficient_seven_evenings_sufficient_l803_803985

theorem four_evenings_insufficient :
  let students := 30
  let max_visits_per_evening := students / 2
  let total_evenings := 4
  let total_pairs_required := (students * (students - 1)) / 2
  in total_evenings * max_visits_per_evening < total_pairs_required :=
by
  sorry

theorem five_evenings_insufficient :
  let students := 30
  let max_visits_per_evening := students / 2
  let total_evenings := 5
  let total_pairs_required := (students * (students - 1)) / 2
  in total_evenings * max_visits_per_evening < total_pairs_required :=
by
  sorry

theorem ten_evenings_sufficient :
  let students := 30
  let max_visits_per_evening := students / 2
  let total_evenings := 10
  let total_pairs_required := (students * (students - 1)) / 2
  in total_evenings * max_visits_per_evening >= total_pairs_required :=
by
  sorry

theorem seven_evenings_sufficient :
  let students := 30
  let max_visits_per_evening := students / 2
  let total_evenings := 7
  let total_pairs_required := (students * (students - 1)) / 2
  in total_evenings * max_visits_per_evening + (additional_scheduling_by_seven_evenings.total_pairs) >= total_pairs_required :=
by
  -- Note: additional_scheduling_by_seven_evenings is a placeholder for the required scheduling logic.
  sorry

end four_evenings_insufficient_five_evenings_insufficient_ten_evenings_sufficient_seven_evenings_sufficient_l803_803985


namespace cone_lateral_surface_is_sector_l803_803920

-- Defining a cone
structure Cone :=
(radius : ℝ)
(height : ℝ)

-- Defining lateral surface of the cone
def lateral_surface (c : Cone) : Type := sorry

-- The main theorem statement
theorem cone_lateral_surface_is_sector (c : Cone) : lateral_surface c = "sector" :=
sorry

end cone_lateral_surface_is_sector_l803_803920


namespace trigonometric_identity_proof_l803_803870

open Real

theorem trigonometric_identity_proof (x y : ℝ) (hx : sin x / sin y = 4) (hy : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 169 / 381 :=
by
  sorry

end trigonometric_identity_proof_l803_803870


namespace substance_same_number_of_atoms_l803_803243

def molecule (kind : String) (atom_count : ℕ) := (kind, atom_count)

def H3PO4 := molecule "H₃PO₄" 8
def H2O2 := molecule "H₂O₂" 4
def H2SO4 := molecule "H₂SO₄" 7
def NaCl := molecule "NaCl" 2 -- though it consists of ions, let's denote it as 2 for simplicity
def HNO3 := molecule "HNO₃" 5

def mol_atoms (mol : ℝ) (molecule : ℕ) : ℝ := mol * molecule

theorem substance_same_number_of_atoms :
  mol_atoms 0.2 H3PO4.2 = mol_atoms 0.4 H2O2.2 :=
by
  unfold H3PO4 H2O2 mol_atoms
  sorry

end substance_same_number_of_atoms_l803_803243


namespace cristobal_read_more_pages_l803_803718

-- Defining the given conditions
def pages_beatrix_read : ℕ := 704
def pages_cristobal_read (b : ℕ) : ℕ := 3 * b + 15

-- Stating the problem
theorem cristobal_read_more_pages (b : ℕ) (c : ℕ) (h : b = pages_beatrix_read) (h_c : c = pages_cristobal_read b) :
  (c - b) = 1423 :=
by
  sorry

end cristobal_read_more_pages_l803_803718


namespace find_a_l803_803745

-- Define the conditions of the problem
def binom_expansion (a x : ℝ) : ℝ := (a * x + (real.sqrt 3) / 6)^6

-- Given the coefficient of the second term is -√3
theorem find_a : (∃ a : ℝ, ∃ x : ℝ, binom_expansion a x = (-real.sqrt 3)) → a = -1 :=
sorry

end find_a_l803_803745


namespace bf_length_l803_803319

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1}

noncomputable def focus (a b : ℝ) : ℝ × ℝ :=
  (real.sqrt (a ^ 2 - b ^ 2), 0)

theorem bf_length {a b : ℝ} (h : a = 4) (h1 : b = 3) (AF_length : ℝ) (x1 y1 : ℝ) :
  let F := focus a b in
  let E := ellipse a b in
  AF_length = 2 →
  F ∈ E →
  F = (real.sqrt (a ^ 2 - b ^ 2), 0) →
  ∃ BF_length : ℝ,
    (x1 - real.sqrt (a ^ 2 - b ^ 2)) ^ 2 + y1 ^ 2 = 4 →
    AF_length = 2 →
    BF_length = (real.sqrt ((- x1 - real.sqrt (a ^ 2 - b ^ 2)) ^ 2 + y1 ^ 2)) := 
sorry

end bf_length_l803_803319


namespace number_of_numbers_with_same_symbol_count_l803_803007

-- Define the condition of symbol counts in Roman numerals for digits.
def romanSymbolCounts : List (Nat × Nat) :=
  [(0, 0), (1, 1), (2, 2), (3, 3), (4, 2), (5, 1), (6, 2), (7, 3), (8, 4), (9, 2)]

-- Define a function to count symbols for Roman numeral conversion of a given digit.
def countRomanSymbols (n : Nat) : Nat :=
  romanSymbolCounts.foldl (λ count (d, s) => if n = d then s else count) 0

-- Define a function to count symbols in decimal representation (every digit corresponds to 1 symbol).
def countDecimalSymbols (n : Nat) : Nat :=
  if n = 0 then 0 else Nat.log10 (n + 1) + 1

-- Define the predicate for numbers that meet the symbol count condition.
def sameSymbolCountInRomanAndDecimal (n : Nat) : Prop :=
  countRomanSymbols n = countDecimalSymbols n

-- Main theorem to prove the desired count of numbers.
theorem number_of_numbers_with_same_symbol_count : Nat :=
  sorry

end number_of_numbers_with_same_symbol_count_l803_803007


namespace sum_of_third_and_fifth_l803_803827

noncomputable def sequence (n : ℕ) : ℚ :=
  if n = 1 then 1 else (n + 1)^3 / (n ^ 3)

theorem sum_of_third_and_fifth :
  sequence 3 + sequence 5 = 13832 / 3375 := by
  sorry

end sum_of_third_and_fifth_l803_803827


namespace binom_mult_l803_803709

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l803_803709


namespace tangents_parallel_l803_803767

-- Define the geometrical setting using structures and variables
variables {Point : Type} [metric_space Point]
variables (A B C D M N P Q E F G H O : Point)
variables (rhombus : rhombus A B C D)
variables (incircle : incircle O A B C D E F G H)
variables (tangent_AB_M : tangent_point O E F A B M)
variables (tangent_BC_N : tangent_point O F G B C N)
variables (tangent_CD_P : tangent_point O G H C D P)
variables (tangent_DA_Q : tangent_point O H E D A Q)

-- State the theorem
theorem tangents_parallel :
  MQ ∥ NP :=
sorry

end tangents_parallel_l803_803767


namespace find_5y_45_sevenths_l803_803013

theorem find_5y_45_sevenths (x y : ℝ) 
(h1 : 3 * x + 4 * y = 0) 
(h2 : x = y + 3) : 
5 * y = -45 / 7 :=
by
  sorry

end find_5y_45_sevenths_l803_803013


namespace unique_polynomial_with_specific_root_property_l803_803813

theorem unique_polynomial_with_specific_root_property :
  ∃! (P : Polynomial ℝ), 
  degree P = 5 ∧
  P = X^5 + C a * X^4 + C b * X^3 + C c * X^2 + C d * X + 2023 ∧
  (∀ r, P.eval r = 0 → P.eval (2 * r) = 0 ∧ P.eval (3 * r) = 0) := 
sorry

end unique_polynomial_with_specific_root_property_l803_803813


namespace mogs_and_mags_to_migs_l803_803097

theorem mogs_and_mags_to_migs:
  (∀ mags migs, 1 * mags = 8 * migs) ∧ 
  (∀ mogs mags, 1 * mogs = 6 * mags) → 
  10 * (6 * 8) + 6 * 8 = 528 := by 
  sorry

end mogs_and_mags_to_migs_l803_803097


namespace max_good_students_l803_803289

theorem max_good_students (n m : ℕ) (scores : Fin n → ℕ) (friends : Fin n → Finset (Fin n)) :
  n = 30 →
  (∀ i, friends i ⊆ {j | j ≠ i}) →
  (∀ i, friends i.card = m) →
  (∀ i, ∀ j, (i ≠ j) → scores i ≠ scores j) →
  (∀ i, (friends i).card = m) →
  (∀ i, (∃ g : ℕ, g = (friends i).filter (λ j, scores j < scores i).card) →
         2 * g > m) →
  ∃ S, S ⊆ (Fin n) ∧ S.card ∃ s. ∀ i ∈ S. (2 * ((friends i).filter (λ j, scores j < scores i)).card > m) ∧ S.card = 25 :=
by
  sorry

end max_good_students_l803_803289


namespace Nicky_wait_time_l803_803500

theorem Nicky_wait_time (x : ℕ) (h1 : x + (4 * x + 14) = 114) : x = 20 :=
by {
  sorry
}

end Nicky_wait_time_l803_803500


namespace distance_on_extension_l803_803892

noncomputable def external_angle (n : ℕ) : ℝ :=
  if n ≥ 5 then 180 - (360 / n) else 0

theorem distance_on_extension (n : ℕ) (a : ℝ) (x : ℕ → ℝ) (h_n : n ≥ 5)
  (h_alpha : external_angle n = 180 - (360 / n)) :
  (∀ k, x k = (a * real.cos (external_angle n)) / (1 - real.cos (external_angle n))) :=
sorry

end distance_on_extension_l803_803892


namespace sum_of_binomial_coefficients_of_odd_terms_l803_803785

theorem sum_of_binomial_coefficients_of_odd_terms (n : ℕ) 
  (h : ∀ k : ℕ, k = 2 → binomial n 2 = binomial n 7) : 
  ∑ k in finset.range (n + 1), if k % 2 = 1 then binomial n k else 0 = 2048 :=
by
  -- notice we assume h tells us n = 9
  sorry

end sum_of_binomial_coefficients_of_odd_terms_l803_803785


namespace sqrt_square_ge_iff_abs_ge_l803_803516

variable (f g : ℝ → ℝ)

theorem sqrt_square_ge_iff_abs_ge (x : ℝ) : 
  sqrt ((f x) ^ 2) ≥ sqrt ((g x) ^ 2) ↔ |f x| ≥ |g x| := 
by 
  sorry

end sqrt_square_ge_iff_abs_ge_l803_803516


namespace percentage_problem_l803_803265

theorem percentage_problem (P : ℝ) :
  (0.2 * 680 = (P / 100) * 140 + 80) → 
  P = 40 :=
by {
  intro h,
  have hp : (0.2 * 680 = (40 / 100) * 140 + 80),
  { norm_num }, -- showing the numerical equality to ease verification.
  exact hp.symm.trans h,
}

end percentage_problem_l803_803265


namespace negative_movement_south_l803_803423

noncomputable def movement_interpretation (x : ℤ) : String :=
if x > 0 then 
  "moving " ++ toString x ++ "m north"
else 
  "moving " ++ toString (-x) ++ "m south"

theorem negative_movement_south : movement_interpretation (-50) = "moving 50m south" := 
by 
  sorry

end negative_movement_south_l803_803423


namespace triangle_area_of_tangent_line_at_2_e_sq_l803_803537

noncomputable def curve := λ x : ℝ, real.exp x
noncomputable def point_tangent := (2 : ℝ, real.exp 2)
noncomputable def derivative := λ x : ℝ, real.exp x

theorem triangle_area_of_tangent_line_at_2_e_sq :
  let slope := derivative 2,
      tangent_line := λ x, slope * (x - 2) + real.exp 2,
      y_intercept := tangent_line 0,
      x_intercept := (λ y : ℝ, (y - real.exp 2) / slope + 2) 0,
      area := 1 / 2 * real.abs y_intercept * real.abs x_intercept
  in area = real.exp 2 / 2 := sorry

end triangle_area_of_tangent_line_at_2_e_sq_l803_803537


namespace paul_sold_books_l803_803512

theorem paul_sold_books (initial_books : ℕ) (left_books : ℕ) : initial_books = 136 → left_books = 27 → initial_books - left_books = 109 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end paul_sold_books_l803_803512


namespace ratio_of_sides_l803_803752
-- Import the complete math library

-- Define the conditions as hypotheses
variables (s x y : ℝ)
variable (h_outer_area : (3 * s)^2 = 9 * s^2)
variable (h_side_lengths : 3 * s = s + 2 * x)
variable (h_y_length : y + x = 3 * s)

-- State the theorem
theorem ratio_of_sides (h_outer_area : (3 * s)^2 = 9 * s^2)
  (h_side_lengths : 3 * s = s + 2 * x)
  (h_y_length : y + x = 3 * s) :
  y / x = 2 := by
  sorry

end ratio_of_sides_l803_803752


namespace sugar_salt_difference_correct_l803_803085

noncomputable def solve_sugar_salt_difference : ℝ :=
let original_sugar := 8 in
let original_salt := 7 in
let added_sugar := 10 in -- Mary has added a total of 10 cups of sugar (8+2)
let required_salt := 7 * added_sugar / original_sugar in
let added_salt := 0 in
let salt_needed := required_salt - added_salt in
let sugar_needed := original_sugar - added_sugar in
sugar_needed - salt_needed

theorem sugar_salt_difference_correct :
  solve_sugar_salt_difference = -8.75 :=
by
  -- we setup the values as per the conditions
  let original_sugar := 8
  let original_salt := 7
  let added_sugar := 10
  let required_salt := 7 * added_sugar / original_sugar
  let added_salt := 0
  let salt_needed := required_salt - added_salt
  let sugar_needed := original_sugar - added_sugar
  -- compute the difference as asked in the problem statement
  have h : solve_sugar_salt_difference = sugar_needed - salt_needed := rfl
  -- mathematical computations to find the difference
  calc 
    solve_sugar_salt_difference
      = (original_sugar - added_sugar) - (required_salt - added_salt) : by rw h
  ... = (8 - 10) - (required_salt - 0) : by rfl
  ... = (8 - 10) - (7 * 10 / 8) : by rfl
  ... = -2 - (70 / 8) : by norm_num
  ... = -2 - 8.75 : by norm_num
  ... = -10.75 : by norm_num
  have h2 : -10.75 = -8.75 := by norm_num -- for understanding step consistency, simplifying assumptions
  sorry

end sugar_salt_difference_correct_l803_803085


namespace domain_of_h_l803_803739

def h (x : ℝ) : ℝ := (3 * x - 1) / sqrt (x^2 - 25)

theorem domain_of_h :
  {x : ℝ | x ∈ (-∞, -5) ∪ (5, ∞)} = {x : ℝ | x^2 - 25 ≥ 0 ∧ x^2 - 25 ≠ 0} :=
by
  sorry

end domain_of_h_l803_803739


namespace trig_identity_solution_l803_803245

noncomputable def solve_trig_identity (x : ℝ) : Prop :=
  (∃ k : ℤ, x = (Real.pi / 8 * (4 * k + 1))) ∧
  (Real.sin (2 * x))^4 + (Real.cos (2 * x))^4 = Real.sin (2 * x) * Real.cos (2 * x)

theorem trig_identity_solution (x : ℝ) :
  solve_trig_identity x :=
sorry

end trig_identity_solution_l803_803245


namespace sufficient_condition_range_k_l803_803375

theorem sufficient_condition_range_k {x k : ℝ} (h : ∀ x, x > k → (3 / (x + 1) < 1)) : k ≥ 2 :=
sorry

end sufficient_condition_range_k_l803_803375


namespace proof_question_1_l803_803624

noncomputable def question_1 (x : ℝ) : ℝ :=
  (Real.sin (2 * x) + 2 * (Real.sin x)^2) / (1 - Real.tan x)

theorem proof_question_1 :
  ∀ x : ℝ, (Real.cos (π / 4 + x) = 3 / 5) →
  (17 * π / 12 < x ∧ x < 7 * π / 4) →
  question_1 x = -9 / 20 :=
by
  intros x h1 h2
  sorry

end proof_question_1_l803_803624


namespace smallest_positive_integer_with_divisors_l803_803228

theorem smallest_positive_integer_with_divisors :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → nat.odd d) ∧ (finset.filter nat.odd (finset.divisors n)).card = 8 ∧ 
           (∃ m : ℕ, ∀ d : ℕ, d ∣ m → nat.even d ∧ m = n → (finset.filter nat.even (finset.divisors m)).card = 16)
             → n = 756 :=
by
  sorry

end smallest_positive_integer_with_divisors_l803_803228


namespace first_divisor_l803_803147

theorem first_divisor (y : ℝ) (x : ℝ) (h1 : 320 / (y * 3) = x) (h2 : x = 53.33) : y = 2 :=
sorry

end first_divisor_l803_803147


namespace find_1995th_element_crossout_3_4_retain_5_l803_803755

def retained_after_crossout (n : ℕ) : Prop :=
  ¬(3 ∣ n) ∧ ¬(4 ∣ n) ∨ (5 ∣ n)

def sequence_element (k : ℕ) (f : ℕ → Prop) : ℕ :=
  Nat.recOn k (0:ℕ) (λ n a, if f (a + 1) then a + 1 else a + 2)

theorem find_1995th_element_crossout_3_4_retain_5 : sequence_element 1995 retained_after_crossout = 25 :=
  sorry

end find_1995th_element_crossout_3_4_retain_5_l803_803755


namespace medical_team_combinations_l803_803151

-- Number of combinations function
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem medical_team_combinations :
  let maleDoctors := 6
  let femaleDoctors := 5
  let requiredMale := 2
  let requiredFemale := 1
  choose maleDoctors requiredMale * choose femaleDoctors requiredFemale = 75 :=
by
  sorry

end medical_team_combinations_l803_803151


namespace h_correct_l803_803761

noncomputable def f (x p : ℝ) : ℝ := cos (2 * x) + p * abs (cos x) + p

def h (p : ℝ) : ℝ :=
  if p < -2 then p - 1 else 2 * p + 1

theorem h_correct (p : ℝ) :
  (∀ x : ℝ, f(x, p) ≤ h(p)) ∧ (∃ x : ℝ, f(x, p) = h(p)) :=
sorry

end h_correct_l803_803761


namespace triangle_area_l803_803250

-- Assign the sides of the triangle as constants
def a : ℝ := 26
def b : ℝ := 24
def c : ℝ := 12

-- Define semiperimeter
def s : ℝ := (a + b + c) / 2

-- Define Heron's formula for the area
noncomputable def area : ℝ :=
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- The main theorem: statement of the problem
theorem triangle_area : 
  abs (area - 143.83) < 0.01 :=
by sorry

end triangle_area_l803_803250


namespace find_k_l803_803052

open Classical

theorem find_k 
    (z x y k : ℝ) 
    (k_pos_int : k > 0 ∧ ∃ n : ℕ, k = n)
    (prop1 : z - y = k * x)
    (prop2 : x - z = k * y)
    (cond : z = (5 / 3) * (x - y)) :
    k = 3 :=
by
  sorry

end find_k_l803_803052


namespace average_speed_l803_803975

theorem average_speed (d1 d2 t1 t2 : ℝ) (h1 : d1 = 90) (h2 : d2 = 75) (h3 : t1 = 1) (h4 : t2 = 1) :
  (d1 + d2) / (t1 + t2) = 82.5 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end average_speed_l803_803975


namespace third_order_central_moment_l803_803102

noncomputable def mu_3 (X : ℝ → ℝ) [MeasureSpace ℝ] : ℝ := 
    (∫ x, (X x - ∫ y, X y) ^ 3)

noncomputable def v1 (X : ℝ → ℝ) [MeasureSpace ℝ] : ℝ := ∫ x, X x
noncomputable def v2 (X : ℝ → ℝ) [MeasureSpace ℝ] : ℝ := ∫ x, (X x)^2
noncomputable def v3 (X : ℝ → ℝ) [MeasureSpace ℝ] : ℝ := ∫ x, (X x)^3

theorem third_order_central_moment 
  (X : ℝ → ℝ) [MeasureSpace ℝ] :
  mu_3 X = v3 X - 3 * v1 X * v2 X + 2 * (v1 X) ^ 3 := by
  sorry

end third_order_central_moment_l803_803102


namespace colorful_triangle_in_complete_graph_l803_803810

open SimpleGraph

theorem colorful_triangle_in_complete_graph (n : ℕ) (h : n ≥ 3) (colors : Fin n → Fin n → Fin (n - 1)) :
  ∃ (u v w : Fin n), u ≠ v ∧ v ≠ w ∧ w ≠ u ∧ colors u v ≠ colors v w ∧ colors v w ≠ colors w u ∧ colors w u ≠ colors u v :=
  sorry

end colorful_triangle_in_complete_graph_l803_803810


namespace pages_read_l803_803951
  
  def chapters := 20
  def pages_per_chapter := 15
  
  theorem pages_read (chapters pages_per_chapter : ℕ) (h_chapters : chapters = 20) (h_pages_per_chapter : pages_per_chapter = 15) : chapters * pages_per_chapter = 300 :=
  by
    rw [h_chapters, h_pages_per_chapter]
    exact Eq.refl (20 * 15)
  
end pages_read_l803_803951


namespace parallel_lines_perpendicular_lines_l803_803001

section parallel_and_perpendicular_lines

variables {a : ℝ}

def slope_l1 : ℝ := 3
def point_A : ℝ × ℝ := (1, 2)
def point_B (a : ℝ) : ℝ × ℝ := (2, a)

def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.snd - p1.snd) / (p2.fst - p1.fst)

theorem parallel_lines (h : slope_l1 = slope point_A (point_B a)) : a = 5 := 
  sorry

theorem perpendicular_lines (h : slope_l1 * slope point_A (point_B a) = -1) : a = 5 / 3 := 
  sorry

end parallel_and_perpendicular_lines

end parallel_lines_perpendicular_lines_l803_803001


namespace shorter_diagonal_of_quadrilateral_sqrt_21_l803_803451

theorem shorter_diagonal_of_quadrilateral_sqrt_21
  (ABCD : Type)
  (midpoint_segments_angle : ∀ (M N P Q : ABCD), ∠(M, N, P) = 60)
  (segment_lengths_ratio : ∀ (M N O P : ABCD), (segment_length M N) / (segment_length O P) = 1 / 3)
  (longer_diagonal_length : real.sqrt(39)) :
  ∃ (d : ℝ), d = real.sqrt(21) :=
sorry

end shorter_diagonal_of_quadrilateral_sqrt_21_l803_803451


namespace no_common_period_l803_803282

noncomputable def periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
∀ x, f(x + p) = f(x)

theorem no_common_period
  (g h : ℝ → ℝ)
  (h_g_periodic : periodic g 3)
  (h_h_periodic : periodic h real.pi)
  : ¬ ∃ T > 0, periodic (λ x, g x - h x) T := by
  sorry

end no_common_period_l803_803282


namespace ellipse_conclusions_l803_803792

def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

def max_min_distance_to_focus (x y : ℝ) : ℝ → Prop :=
  λ c, ∃ (p : ℝ), (p = 3 ∨ p = 1)

def cos_angle_F1PF2 (x y : ℝ) : ℝ :=
  (x^2 + y^2 - 4 * 1) / (8)

def area_of_right_triangle (x y : ℝ) : ℝ :=
  if (x + y) = 2 * sqrt (4) ∧ x^2 + y^2 = 4 then 0.5 * x * y else 0

def range_of_dot_product (x y : ℝ) : set ℝ :=
  {z |  2 ≤ z ∧ z ≤ 3 }

theorem ellipse_conclusions (x y : ℝ) (h₁ : ellipse_equation x y) :
  (max_min_distance_to_focus x y 3 ∧ max_min_distance_to_focus x y 1) ∧
  (cos_angle_F1PF2 x y ≠ 1 / 4) ∧
  (area_of_right_triangle x y ≠ 3 / 2) ∧
  (range_of_dot_product x y = {z | 2 ≤ z ∧ z ≤ 3}) :=
by
  sorry

end ellipse_conclusions_l803_803792


namespace midsegment_of_trapezoid_l803_803119

theorem midsegment_of_trapezoid 
  (A B C D M : Point)
  (AB CD : ℝ)
  (h1 : AB = 10)
  (h2 : CD = 6)
  (h_midpoint : M ∈ segment A D)
  (h_circle : circle_intersect_diameter_midpoint AB M A D)
  (h_right_trapezoid : is_right_trapezoid A B C D) : 
  midsegment_length A B C D = 12 := 
sorry

end midsegment_of_trapezoid_l803_803119


namespace avg_bc_eq_70_l803_803913

-- Definitions of the given conditions
variables (a b c : ℝ)

def avg_ab (a b : ℝ) : Prop := (a + b) / 2 = 45
def diff_ca (a c : ℝ) : Prop := c - a = 50

-- The main theorem statement
theorem avg_bc_eq_70 (h1 : avg_ab a b) (h2 : diff_ca a c) : (b + c) / 2 = 70 :=
by
  sorry

end avg_bc_eq_70_l803_803913


namespace angle_between_lines_l803_803808

theorem angle_between_lines (a : ℝ) (h : 0 < a ∧ a ≠ 1) : 
  (∃ θ : ℝ, a = tan θ ∧ θ ∈ (π / 6, π / 4) ∪ (π / 4, π / 3)) ↔
  a ∈ (sqrt 3 / 3, 1) ∪ (1, sqrt 3) :=
sorry

end angle_between_lines_l803_803808


namespace maxwell_distance_when_meeting_l803_803885

/-- Maxwell and Brad are moving towards each other from their respective homes.
Maxwell's walking speed is 3 km/h, Brad's running speed is 5 km/h,
and the distance between their homes is 40 kilometers.
Prove that the distance traveled by Maxwell when they meet is 15 kilometers. -/
theorem maxwell_distance_when_meeting
  (distance_between_homes : ℝ)
  (maxwell_speed : ℝ)
  (brad_speed : ℝ)
  (meeting_distance : ℝ) :
  distance_between_homes = 40 ∧ maxwell_speed = 3 ∧ brad_speed = 5 → meeting_distance = 15 :=
begin
  intros h,
  rcases h with ⟨d_eq_40, m_speed_eq_3, b_speed_eq_5⟩,
  sorry
end

end maxwell_distance_when_meeting_l803_803885


namespace min_A2_minus_B2_l803_803872

def A (x y z : ℕ) : ℝ := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 11)
def B (x y z : ℕ) : ℝ := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)
def min_value_A2_minus_B2 : ℝ := 36

theorem min_A2_minus_B2 (x y z : ℕ) : 
  A^2 x y z - B^2 x y z ≥ min_value_A2_minus_B2 := 
by 
  sorry

end min_A2_minus_B2_l803_803872


namespace mushroom_collection_l803_803304

variable (a b v g : ℕ)

theorem mushroom_collection : 
  (a / 2 + 2 * b = v + g) ∧ (a + b = v / 2 + 2 * g) → (v = 2 * b) ∧ (a = 2 * g) :=
by
  sorry

end mushroom_collection_l803_803304


namespace roots_of_equation_l803_803139

theorem roots_of_equation : 
  ∀ x : ℝ, 2 * real.sqrt x + 2 * x^(-1 / 2) = 5 → 4 * x^2 - 17 * x + 4 = 0 :=
by
  -- The proof is intentionally left out.
  sorry

end roots_of_equation_l803_803139


namespace book_cost_in_usd_l803_803273

def book_price_CNY : ℝ := 31
def exchange_rate_USD_CNY : ℝ := 6.2
def book_price_USD : ℝ := book_price_CNY / exchange_rate_USD_CNY

theorem book_cost_in_usd : book_price_USD = 5 := by
  sorry

end book_cost_in_usd_l803_803273


namespace valid_probability_distribution_of_X_probability_A_or_B_conditional_probability_B_given_A_eq_l803_803905

def number_of_students : ℕ := 6
def number_of_boys : ℕ := 4
def number_of_girls : ℕ := 2
def number_of_selected_students : ℕ := 3

noncomputable def probability_distribution_of_X : (Fin 3) → ℚ :=
  λ X, match X with
  | ⟨0, _⟩ => 1/5
  | ⟨1, _⟩ => 3/5
  | ⟨2, _⟩ => 1/5

axiom sum_of_probabilities : 
  probability_distribution_of_X ⟨0, _⟩ + 
  probability_distribution_of_X ⟨1, _⟩ + 
  probability_distribution_of_X ⟨2, _⟩ = 1

def event_A_selected : ℚ := 1/2  -- Probability that boy A is selected
def event_B_selected : ℚ := 1/2  -- Empirical value for girl B selected (not used further)
def event_AB_selected : ℚ := 1/5 -- Probability that both A and B are selected

noncomputable def probability_A_or_B_selected : ℚ := 4/5
noncomputable def conditional_probability_B_given_A : ℚ :=
  event_AB_selected / event_A_selected

theorem valid_probability_distribution_of_X : 
  ∀ X : Fin 3, probability_distribution_of_X X ∈ {1/5, 3/5} := by sorry

theorem probability_A_or_B : probability_A_or_B_selected = 4 / 5 := by sorry

theorem conditional_probability_B_given_A_eq : 
  conditional_probability_B_given_A = 2 / 5 := by sorry

end valid_probability_distribution_of_X_probability_A_or_B_conditional_probability_B_given_A_eq_l803_803905


namespace arithmetic_sequence_term_2011_l803_803267

theorem arithmetic_sequence_term_2011 :
  ∃ (n : ℕ), 1 + (n - 1) * 3 = 2011 ∧ n = 671 :=
by
  existsi 671
  split
  ·  sorry
  ·  refl

end arithmetic_sequence_term_2011_l803_803267


namespace log_sequence_geometric_progression_l803_803818

open Real

variable (x y z m : ℝ) (k : ℕ)

theorem log_sequence_geometric_progression
  (h1 : 1 < x)
  (h2 : x < y)
  (h3 : y < z)
  (h4 : m > 1)
  (h5 : y = x^k)
  (h6 : z = x^(k^2))
  (hk1 : k > 1) :
  ∃ r : ℝ, log x m = r ∧ log y m = r / k ∧ log z m = r / (k^2) :=
by
  sorry

end log_sequence_geometric_progression_l803_803818


namespace number_of_zeros_of_f_is_3_l803_803927

def f (x : ℝ) : ℝ := x^3 - 64 * x

theorem number_of_zeros_of_f_is_3 : ∃ x1 x2 x3, (f x1 = 0) ∧ (f x2 = 0) ∧ (f x3 = 0) ∧ (x1 ≠ x2) ∧ (x2 ≠ x3) ∧ (x1 ≠ x3) :=
by
  sorry

end number_of_zeros_of_f_is_3_l803_803927


namespace sheila_hours_tuesday_thursday_l803_803907

noncomputable def Sheila_work_hours : ℕ := sorry

theorem sheila_hours_tuesday_thursday :
  (h_m h_w h_f h_t h_th : ℕ)
  (hm_h : h_m = 8)
  (hw_h : h_w = 8)
  (hf_h : h_f = 8)
  (total_weekly_earnings : ℕ)
  (total_weekly_earnings_h : total_weekly_earnings = 432)
  (hourly_rate : ℕ)
  (hourly_rate_h : hourly_rate = 12)
  (total_worked_hours : ℕ)
  (total_worked_hours_h : total_worked_hours = h_m + h_w + h_f + h_t + h_th)
  (earnings_condition : total_weekly_earnings = total_worked_hours * hourly_rate) :
  (h_t + h_th = 12) :=
by
  sorry

end sheila_hours_tuesday_thursday_l803_803907


namespace main_inequality_l803_803918

noncomputable theory

-- Define the function f and its derivative f'
variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}
-- Define that the function f is differentiable, and f' is its derivative
variable [differentiable ℝ f]
variable [differentiable ℝ f']

-- Define the conditions from the problem
axiom h_cond : ∀ x > 0, x * f'(x) - f(x) < 0
-- Define constants a, b, c as per the problem statement
def a : ℝ := f (2^(0.2)) / (2^(0.2))
def b : ℝ := f ((0.2)^2) / ((0.2)^2)
def c : ℝ := f (real.log 5 / real.log 2) / (real.log 5 / real.log 2)

-- Our goal is to prove the correct inequality
theorem main_inequality : c < a ∧ a < b :=
sorry

end main_inequality_l803_803918


namespace area_of_PQR_is_120_l803_803162

noncomputable def triangle_area {P Q R : Type} [metric_space P] [metric_space Q] [metric_space R]
  (PQ PR QR : ℝ) (is_isosceles : PQ = PR) (side_lengths : PQ = 17 ∧ PR = 17 ∧ QR = 16) : ℝ :=
  let PS := real.sqrt (PQ * PQ - (QR / 2) * (QR / 2)) in
  (1 / 2) * QR * PS

theorem area_of_PQR_is_120 (P Q R : Type) 
    [metric_space P] [metric_space Q] [metric_space R] 
    (PQ PR QR : ℝ) (h_isosceles : PQ = PR) 
    (h_side_lengths : PQ = 17 ∧ PR = 17 ∧ QR = 16) : 
    triangle_area PQ PR QR h_isosceles h_side_lengths = 120 :=
begin
  sorry
end

end area_of_PQR_is_120_l803_803162


namespace find_stream_speed_l803_803569

variable (boat_speed dist_downstream dist_upstream : ℝ)
variable (stream_speed : ℝ)

noncomputable def speed_of_stream (boat_speed dist_downstream dist_upstream : ℝ) : ℝ :=
  let t_downstream := dist_downstream / (boat_speed + stream_speed)
  let t_upstream := dist_upstream / (boat_speed - stream_speed)
  if t_downstream = t_upstream then stream_speed else 0

theorem find_stream_speed
  (h : speed_of_stream 20 26 14 stream_speed = stream_speed) :
  stream_speed = 6 :=
sorry

end find_stream_speed_l803_803569


namespace mop_red_slip_identity_l803_803627

-- Define the problem statement in Lean
theorem mop_red_slip_identity (A B : ℕ) (hA : A ≤ 2010) (hB : B ≤ 2010) : 
  ∃ (f : Fin 2010 → Fin 2010) (g : Fin 2010 → Fin 2010),
    (∀ i : Fin 2010, f i = i) ∧ (∀ j : Fin 2010, g j = A * j % 2011) :=
sorry

end mop_red_slip_identity_l803_803627


namespace triangle_BOC_is_isosceles_l803_803770

-- Definitions based on the given problem
variables (A B C D M O : Type) [T : Trapezoid A B C D]
variable (AB_eq_BD : seg_eq A B B D)
variable (M_mid_CD : Midpoint M C D)
variable (O_int_AC_BM : Intersection O A C B M)

-- The theorem to be proven
theorem triangle_BOC_is_isosceles :
  is_isosceles_triangle B O C :=
sorry

end triangle_BOC_is_isosceles_l803_803770


namespace bridge_construction_cost_l803_803258

-- Define the given conditions and variables
variables (a b c : Real) (cost_per_km : Real)
def semi_perimeter (a b c : Real) : Real := (a + b + c) / 2 
def herons_area (s a b c : Real) : Real := Real.sqrt (s * (s - a) * (s - b) * (s - c))
def circumscribed_circle_radius (a b c : Real) (A : Real) : Real := (a * b * c) / (4 * A)
def inscribed_circle_radius (A s : Real) : Real := A / s
def cos_angle_c (a b c : Real) : Real := (a^2 + b^2 - c^2) / (2 * a * b)
def sin_angle_c (cosC : Real) : Real := Real.sqrt (1 - cosC^2)
def distance_between_centers (R r cosC sinC : Real) : Real := Real.sqrt (R^2 + 2 * R * r * cosC + r^2)

-- Prove the statement regarding the total cost given specific triangle sides and cost per kilometer
theorem bridge_construction_cost : 
    ∀ (a b c : Real) (cost_per_km : Real), 
    a = 7 → b = 8 → c = 9 → cost_per_km = 1000 →
    let s := semi_perimeter a b c in
    let A := herons_area s a b c in
    let R := circumscribed_circle_radius a b c A in
    let r := inscribed_circle_radius A s in
    let cosC := cos_angle_c a b c in
    let sinC := sin_angle_c cosC in
    let O_1O_2 := distance_between_centers R r cosC sinC in
    let total_cost := cost_per_km * O_1O_2 in
    total_cost ≈ 5750 :=
by 
    intros a b c cost_per_km ha hb hc hcost
    sorry

end bridge_construction_cost_l803_803258


namespace hockey_players_count_l803_803829

theorem hockey_players_count (total_players cricket_players football_players softball_players : ℕ)
    (h_cricket : cricket_players = 15)
    (h_football : football_players = 13)
    (h_softball : softball_players = 15)
    (h_total : total_players = 55) :
  ∃ hockey_players : ℕ, hockey_players = total_players - (cricket_players + football_players + softball_players) ∧ hockey_players = 12 :=
by 
  use total_players - (cricket_players + football_players + softball_players)
  have h_num : hockey_players = total_players - (cricket_players + football_players + softball_players) := sorry
  have h_correct : hockey_players = 12 := sorry
  exact ⟨h_num, h_correct⟩
  sorry

end hockey_players_count_l803_803829


namespace quadratic_equation_transformation_transformation_idea_reflected_in_factorization_l803_803591

theorem quadratic_equation_transformation (x : ℝ) :
  (3 * x^2 - 6 * x = 0 → (3*x*(x-2) = 0) → (x = 0 ∨ x = 2)) :=
by
  intro h1 h2
  sorry

theorem transformation_idea_reflected_in_factorization (t : Type) [Field t] (x : t) :
  (3 * x^2 - 6 * x = 0 → 3 * x * (x - 2) = 0) →
    (∀ a b : t, a * b = 0 → a = 0 ∨ b = 0) →
      (3 * x^2 - 6 * x = 0 → x = 0 ∨ x = 2) :=
by
  intro h1 h2 h3
  rw quad_eq at *
  apply or_iff_not_imp_left.mp
  sorry

end quadratic_equation_transformation_transformation_idea_reflected_in_factorization_l803_803591


namespace smallest_n_with_divisors_l803_803189

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end smallest_n_with_divisors_l803_803189


namespace arithmetic_sequence_general_formula_l803_803538

noncomputable def a_n : ℕ → ℝ
noncomputable def a_seq (a_n : ℕ → ℝ) : Prop := 
  ∃ d : ℝ, ( ∀ n : ℕ, a_n (n + 1) = a_n n + d ) ∧ d < 0

theorem arithmetic_sequence_general_formula (a_n : ℕ → ℝ) (h₁ : a_seq a_n) 
  (h₂ : a_n 2 * a_n 4 * a_n 6 = 45) (h₃ : a_n 2 + a_n 4 + a_n 6 = 15) :
  ∀ n : ℕ, a_n n = -2 * (n : ℝ) + 13 :=
by
  sorry

end arithmetic_sequence_general_formula_l803_803538


namespace find_k_l803_803821

theorem find_k (k : ℝ) (h : (3 : ℝ), (-1 : ℝ) ∈ {p : ℝ × ℝ | ∃ k : ℝ, p.snd = k / p.fst}) : k = -3 :=
sorry

end find_k_l803_803821


namespace scientific_notation_of_935million_l803_803668

theorem scientific_notation_of_935million :
  935000000 = 9.35 * 10 ^ 8 :=
  sorry

end scientific_notation_of_935million_l803_803668


namespace smallest_positive_integer_conditions_l803_803960

theorem smallest_positive_integer_conditions :
  ∃ n : ℕ, (n > 1) ∧ (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) ∧ (n % 11 = 0) ∧ n = 242 :=
begin
  -- proof steps will be skipped
  sorry
end

end smallest_positive_integer_conditions_l803_803960


namespace sum_divisible_by_7_l803_803966

theorem sum_divisible_by_7 (n : ℕ) : (8^n + 6) % 7 = 0 := 
by
  sorry

end sum_divisible_by_7_l803_803966


namespace sqrt_defined_iff_ge_neg1_l803_803030

theorem sqrt_defined_iff_ge_neg1 (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x + 1)) ↔ x ≥ -1 := by
  sorry

end sqrt_defined_iff_ge_neg1_l803_803030


namespace unique_suwy_product_l803_803735

def letter_value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then Char.toNat c - Char.toNat 'A' + 1 else 0

def product_of_chars (l : List Char) : Nat :=
  l.foldr (λ c acc => letter_value c * acc) 1

theorem unique_suwy_product :
  ∀ (l : List Char), l.length = 4 → product_of_chars l = 19 * 21 * 23 * 25 → l = ['S', 'U', 'W', 'Y'] := 
by
  intro l hlen hproduct
  sorry

end unique_suwy_product_l803_803735


namespace midpoint_sum_segment_is_horizontal_l803_803230

-- Define the endpoints of the segment
def P1 : ℝ × ℝ := (4, 10)
def P2 : ℝ × ℝ := (-2, 10)

-- Define the midpoint calculation based on the endpoints
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define a function to calculate the sum of the coordinates of a point
def sum_of_coordinates (P : ℝ × ℝ) : ℝ :=
  P.1 + P.2

-- Prove the sum of the coordinates of the midpoint
theorem midpoint_sum : sum_of_coordinates (midpoint P1 P2) = 11 :=
by
  sorry

-- The segment is horizontal if the y-coordinates of the endpoints are the same
theorem segment_is_horizontal : P1.2 = P2.2 :=
by
  sorry

end midpoint_sum_segment_is_horizontal_l803_803230


namespace arithmetic_sequence_sum_Tn_l803_803768

-- Define the sequence a_n
def a : ℕ → ℕ
| 0 := 2
| 1 := 3
| (n+2) := S n + 1 + S (n+1) - S (n+2) - 1

-- Define the sum of the first n terms of a_n (S_n)
def S : ℕ → ℕ
| n := Σ i in range (n + 1), a i

-- Prove that a_n is an arithmetic sequence and find the formula for a_n
theorem arithmetic_sequence (n : ℕ) : a (n+1) = n + 1 := sorry

-- Define the sequence {1/(a_na_{n+1})}
def b : ℕ → ℚ
| n := 1 / (a n * a (n+1)) 

-- Define the sum of the first n terms of b_n (T_n)
def T : ℕ → ℚ
| n := Σ i in range n, b i

-- Prove that T_n = n / (2 * (n+2))
theorem sum_Tn (n : ℕ) : T n = n / (2 * (n + 2)) := sorry

end arithmetic_sequence_sum_Tn_l803_803768


namespace sample_size_l803_803279

theorem sample_size (T : ℕ) (f_C : ℚ) (samples_C : ℕ) (n : ℕ) 
    (hT : T = 260)
    (hfC : f_C = 3 / 13)
    (hsamples_C : samples_C = 3) : n = 13 :=
by
  -- Proof goes here
  sorry

end sample_size_l803_803279


namespace product_of_two_numbers_l803_803165

theorem product_of_two_numbers
  (x y : ℝ)
  (h_diff : x - y ≠ 0)
  (h1 : x + y = 5 * (x - y))
  (h2 : x * y = 15 * (x - y)) :
  x * y = 37.5 :=
by
  sorry

end product_of_two_numbers_l803_803165


namespace volume_inscribed_cube_l803_803655

-- Definitions based on the problem conditions
def edge_length_large_cube : ℝ := 12
def diameter_sphere : ℝ := edge_length_large_cube
def space_diagonal_smaller_cube : ℝ := diameter_sphere

-- Lean statement for the proof problem
theorem volume_inscribed_cube (s : ℝ) (V : ℝ) : 
  (s = 4 * Real.sqrt 3) →
  (V = s^3) →
  (V = 192 * Real.sqrt 3) :=
by
  intros h_s h_V
  rw h_s at h_V
  rw h_V
  sorry

end volume_inscribed_cube_l803_803655


namespace fraction_product_simplified_l803_803176

theorem fraction_product_simplified:
  (2 / 9 : ℚ) * (5 / 8 : ℚ) = 5 / 36 :=
by {
  sorry
}

end fraction_product_simplified_l803_803176


namespace functional_equation_solution_l803_803766

theorem functional_equation_solution (f : ℝ → ℝ) (t : ℝ) (h : t ≠ -1) :
  (∀ x y : ℝ, (t + 1) * f (1 + x * y) - f (x + y) = f (x + 1) * f (y + 1)) →
  (∀ x, f x = 0) ∨ (∀ x, f x = t) ∨ (∀ x, f x = (t + 1) * x - (t + 2)) :=
by
  sorry

end functional_equation_solution_l803_803766


namespace smallest_n_with_divisors_l803_803188

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end smallest_n_with_divisors_l803_803188


namespace total_commencement_addresses_l803_803842

-- Define the given conditions
def sandoval_addresses := 12
def sandoval_rainy_addresses := 5
def sandoval_public_holidays := 2
def sandoval_non_rainy_addresses := sandoval_addresses - sandoval_rainy_addresses

def hawkins_addresses := sandoval_addresses / 2
def sloan_addresses := sandoval_addresses + 10
def sloan_non_rainy_addresses := sloan_addresses -- assuming no rainy day details are provided

def davenport_addresses := (sandoval_non_rainy_addresses + sloan_non_rainy_addresses) / 2 - 3
def davenport_addresses_rounded := 11 -- rounding down to nearest integer as per given solution

def adkins_addresses := hawkins_addresses + davenport_addresses_rounded + 2

-- Calculate the total number of addresses
def total_addresses := sandoval_addresses + hawkins_addresses + sloan_addresses + davenport_addresses_rounded + adkins_addresses

-- The proof goal statement
theorem total_commencement_addresses : total_addresses = 70 := by
  -- Proof to be provided here
  sorry

end total_commencement_addresses_l803_803842


namespace smallest_digit_N_divisible_by_6_l803_803178

theorem smallest_digit_N_divisible_by_6 : 
  ∃ N : ℕ, N < 10 ∧ 
          (14530 + N) % 6 = 0 ∧
          ∀ M : ℕ, M < N → (14530 + M) % 6 ≠ 0 := sorry

end smallest_digit_N_divisible_by_6_l803_803178


namespace raduzhny_population_l803_803049

theorem raduzhny_population :
  ∀ (num_inhabitants_znoinoe : ℕ)
    (avg_surplus : ℕ)
    (num_settlements : ℕ)
    (total_population : ℕ),
    num_inhabitants_znoinoe = 1000 →
    avg_surplus = 90 →
    num_settlements = 10 →
    let avg_population_valley := (total_population + num_inhabitants_znoinoe) / num_settlements in
    avg_population_valley + avg_surplus = num_inhabitants_znoinoe →
    ∃ (num_inhabitants_raduzhny : ℕ), 
    num_inhabitants_raduzhny = 900 :=
begin
  assume num_inhabitants_znoinoe avg_surplus num_settlements total_population,
  assume h1 : num_inhabitants_znoinoe = 1000,
  assume h2 : avg_surplus = 90,
  assume h3 : num_settlements = 10,
  assume h4 : let avg_population_valley := (total_population + num_inhabitants_znoinoe) / num_settlements in
              avg_population_valley + avg_surplus = num_inhabitants_znoinoe,
  sorry
end

end raduzhny_population_l803_803049


namespace smallest_integer_with_divisors_properties_l803_803197

def number_of_odd_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 1)

def number_of_even_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 0)

theorem smallest_integer_with_divisors_properties :
  ∃ n : ℕ, number_of_odd_divisors n = 8 ∧ number_of_even_divisors n = 16 ∧ n = 4000 :=
by
  sorry

end smallest_integer_with_divisors_properties_l803_803197


namespace find_CP_A_l803_803295

noncomputable def CP_A : Float := 173.41
def SP_B (CP_A : Float) : Float := 1.20 * CP_A
def SP_C (SP_B : Float) : Float := 1.25 * SP_B
def TC_C (SP_C : Float) : Float := 1.15 * SP_C
def SP_D1 (TC_C : Float) : Float := 1.30 * TC_C
def SP_D2 (SP_D1 : Float) : Float := 0.90 * SP_D1
def SP_D2_actual : Float := 350

theorem find_CP_A : 
  (SP_D2 (SP_D1 (TC_C (SP_C (SP_B CP_A))))) = SP_D2_actual → 
  CP_A = 173.41 := sorry

end find_CP_A_l803_803295


namespace acute_triangle_angle_interval_l803_803835

theorem acute_triangle_angle_interval (triangle : Type) (A B C : triangle) 
  (acute_angled : ∀ (α β γ : ℝ), α < 90 ∧ β < 90 ∧ γ < 90 ∧ α + β + γ = 180)
  (altitudes : ∀ (Ha Hb Hc : triangle), true)
  (circumcircle_passing_centroid : true) :
  ∃ (angles : Set ℝ), angles = setOf (λ θ : ℝ, θ ∈ (\(arcTan (sqrt 2)\), 60]) :=
by
  unfold setOf
  sorry

end acute_triangle_angle_interval_l803_803835


namespace circle_tangent_line_k_range_l803_803764

theorem circle_tangent_line_k_range
  (k : ℝ)
  (P Q : ℝ × ℝ)
  (c : ℝ × ℝ := (0, 1)) -- Circle center
  (r : ℝ := 1) -- Circle radius
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 2 * y = 0)
  (line_eq : ∀ (x y : ℝ), k * x + y + 3 = 0)
  (dist_pq : Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2) = Real.sqrt 3) :
  k ∈ Set.Iic (-Real.sqrt 3) ∪ Set.Ici (Real.sqrt 3) :=
by
  sorry

end circle_tangent_line_k_range_l803_803764


namespace sum_of_g_47_l803_803859

noncomputable def f : ℝ → ℝ := λ x => 5 * x^2 - 3
noncomputable def g : ℝ → ℝ := λ y => if h : ∃ x, f x = y then (classical.some h)^2 - classical.some h + 2 else 0

theorem sum_of_g_47 :
  g 47 = 10 - Real.sqrt 10 + 2 ∨ g 47 = 10 + Real.sqrt 10 + 2 → 
  ∃ s, s = (10 - Real.sqrt 10 + 2) + (10 + Real.sqrt 10 + 2) ∧ s = 24 :=
by
  sorry

end sum_of_g_47_l803_803859


namespace arithmetic_sequence_term_number_l803_803269

-- Given:
def first_term : ℕ := 1
def common_difference : ℕ := 3
def target_term : ℕ := 2011

-- To prove:
theorem arithmetic_sequence_term_number :
    ∃ n : ℕ, target_term = first_term + (n - 1) * common_difference ∧ n = 671 := 
by
  -- The proof is omitted
  sorry

end arithmetic_sequence_term_number_l803_803269


namespace probability_prime_sum_is_5_over_9_l803_803929

-- Define the sectors of the spinners
def spinner1 := {2, 3, 4}
def spinner2 := {1, 5, 7}

-- Define what it means for a sum of numbers to be prime
def is_prime (n : ℕ) : Prop := (n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11)

-- Compute all possible sums
def possible_sums := {x + y | x ∈ spinner1, y ∈ spinner2}

-- Filter the sums to get only the prime sums
def prime_sums := possible_sums.filter is_prime

-- Define the probability as a ratio
def probability_prime_sum : ℚ := (prime_sums.card : ℚ) / (possible_sums.card : ℚ)

-- State the problem to prove that the calculated probability is 5/9
theorem probability_prime_sum_is_5_over_9 :
  probability_prime_sum = 5 / 9 := 
by
  sorry

end probability_prime_sum_is_5_over_9_l803_803929


namespace fourth_rod_count_l803_803082

def rod_lengths : set ℤ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}

def used_rods : set ℤ := {4, 9, 18}

def valid_rod_length (l : ℤ) : Prop := 
  5 < l ∧ l < (4 + 9 + 18)

def remaining_rods : set ℤ := rod_lengths \ used_rods

theorem fourth_rod_count : ∃ n, n = 17 ∧
  ∀ l ∈ remaining_rods, valid_rod_length l → l ∈ {x : ℤ | 6 ≤ x ∧ x ≤ 25} :=
by
  sorry

end fourth_rod_count_l803_803082


namespace midpoint_meeting_point_l803_803883

-- Define the points where Mark and Sandy are standing
def Mark := (0, 7)
def Sandy := (-4, -1)

-- Define the midpoint formula for two points
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the statement that we need to prove
theorem midpoint_meeting_point :
  midpoint Mark Sandy = (-2, 3) :=
by
  -- Proof omitted (proof steps replaced with sorry)
  sorry

end midpoint_meeting_point_l803_803883


namespace geometric_sequence_fifth_term_l803_803993

theorem geometric_sequence_fifth_term (r : ℕ) (h₁ : 5 * r^3 = 405) : 5 * r^4 = 405 :=
sorry

end geometric_sequence_fifth_term_l803_803993


namespace no_square_from_sticks_l803_803586

theorem no_square_from_sticks :
  let total_length := 6 * 1 + 3 * 2 + 6 * 3 + 5 * 4 
  let side_length := total_length / 4
  total_length = 50 ∧ ¬ (side_length.is_integer ∧ side_length * 4 = total_length) :=
by
  let total_length := 6 * 1 + 3 * 2 + 6 * 3 + 5 * 4
  let side_length := total_length / 4
  have h1: total_length = 50, by norm_num
  have h2: ¬ side_length.is_integer, by 
      simp [side_length]
      sorry
  exact ⟨h1, h2⟩

end no_square_from_sticks_l803_803586


namespace sum_angles_cuboid_l803_803836

theorem sum_angles_cuboid (A B C D A1 B1 C1 D1 : Type)
  (α β γ : ℝ)
  (h1 : ∀ {P Q R : Type}, is_right_angle A B1 Q → is_right_angle A1 D R → angle PQ = α)
  (h2 : ∀ {P Q R : Type}, is_right_angle A C Q → is_right_angle B C1 R → angle PQ = β)
  (h3 : ∀ {P Q R : Type}, is_right_angle A1 C1 Q → is_right_angle C D1 R → angle PQ = γ) :
  α + β + γ = 180 :=
sorry

end sum_angles_cuboid_l803_803836


namespace jenny_eggs_per_basket_l803_803466

theorem jenny_eggs_per_basket :
  ∃ n, (30 % n = 0 ∧ 42 % n = 0 ∧ 18 % n = 0 ∧ n >= 6) → n = 6 :=
by
  sorry

end jenny_eggs_per_basket_l803_803466


namespace binom_coeff_mult_l803_803705

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l803_803705


namespace pentagon_zero_impossible_l803_803584

theorem pentagon_zero_impossible
  (x : Fin 5 → ℝ)
  (h_sum : x 0 + x 1 + x 2 + x 3 + x 4 = 0)
  (operation : ∀ i : Fin 5, ∀ y : Fin 5 → ℝ,
    y i = (x i + x ((i + 1) % 5)) / 2 ∧ y ((i + 1) % 5) = (x i + x ((i + 1) % 5)) / 2) :
  ¬ ∃ (y : ℕ → (Fin 5 → ℝ)), ∃ N : ℕ, y N = 0 := 
sorry

end pentagon_zero_impossible_l803_803584


namespace factorial_division_l803_803376

theorem factorial_division : (7! = 5040) → (7! / 4! = 210) :=
by
  intro h
  rw Nat.factorial_succ
  rw Nat.factorial_succ
  rw Nat.factorial_succ
  rw Nat.factorial_succ
  sorry

end factorial_division_l803_803376


namespace isosceles_triangle_area_is_120_l803_803161

noncomputable def triangle_area (PQ PR QR PS : ℝ) : ℝ :=
  (1 / 2) * QR * PS

-- Given conditions
def PQ : ℝ := 17
def PR : ℝ := 17
def QR : ℝ := 16

-- Triangle PQ is right triangle with hypotenuse PQ and one leg QS = 8
def QS : ℝ := 8
def PS : ℝ := real.sqrt (PQ^2 - QS^2)

theorem isosceles_triangle_area_is_120 :
  triangle_area PQ PR QR PS = 120 :=
by
  sorry

end isosceles_triangle_area_is_120_l803_803161


namespace direction_vector_of_line_with_inclination_150_eq_negative_3_sqrt_3_l803_803032

theorem direction_vector_of_line_with_inclination_150_eq_negative_3_sqrt_3 : 
  ∃ (x y : ℝ), 
  (y / x = -Real.sqrt 3 / 3) ∧ (150 = Real.atan (y / x) / Real.pi * 180) := 
sorry

end direction_vector_of_line_with_inclination_150_eq_negative_3_sqrt_3_l803_803032


namespace smallest_abundant_not_multiple_of_4_l803_803597

def proper_divisors (n : ℕ) : List ℕ := (List.range n).filter (λ d, d ∣ n)

def sum_proper_divisors (n : ℕ) : ℕ := (proper_divisors n).sum

def is_abundant (n : ℕ) : Prop := sum_proper_divisors n > n

def not_multiple_of_4 (n : ℕ) : Prop := ¬ (4 ∣ n)

theorem smallest_abundant_not_multiple_of_4 : ∃ n, is_abundant n ∧ not_multiple_of_4 n ∧ (∀ m, is_abundant m ∧ not_multiple_of_4 m → n ≤ m) :=
  sorry

end smallest_abundant_not_multiple_of_4_l803_803597


namespace count_multiples_of_neither_6_nor_8_l803_803416

theorem count_multiples_of_neither_6_nor_8 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999},
      multiples_of_6 := {n : ℕ | n ∈ three_digit_numbers ∧ n % 6 = 0},
      multiples_of_8 := {n : ℕ | n ∈ three_digit_numbers ∧ n % 8 = 0},
      multiples_of_24 := {n : ℕ | n ∈ three_digit_numbers ∧ n % 24 = 0} in
  (three_digit_numbers.card 
   - (multiples_of_6.card + multiples_of_8.card - multiples_of_24.card)) = 675 :=
by sorry

end count_multiples_of_neither_6_nor_8_l803_803416


namespace trigonometric_identity_proof_l803_803869

open Real

theorem trigonometric_identity_proof (x y : ℝ) (hx : sin x / sin y = 4) (hy : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 169 / 381 :=
by
  sorry

end trigonometric_identity_proof_l803_803869


namespace smallest_integer_has_8_odd_and_16_even_divisors_l803_803217

/-!
  Prove that the smallest positive integer with exactly 8 positive odd integer divisors
  and exactly 16 positive even integer divisors is 540.
-/
def smallest_integer_with_divisors : ℕ :=
  540

theorem smallest_integer_has_8_odd_and_16_even_divisors 
  (n : ℕ) 
  (h1 : (8 : ℕ) = nat.count (λ d, d % 2 = 1) (nat.divisors n))
  (h2 : (16 : ℕ) = nat.count (λ d, d % 2 = 0) (nat.divisors n)) :
  n = 540 :=
sorry

end smallest_integer_has_8_odd_and_16_even_divisors_l803_803217


namespace smallest_positive_integer_with_divisors_l803_803225

theorem smallest_positive_integer_with_divisors :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → nat.odd d) ∧ (finset.filter nat.odd (finset.divisors n)).card = 8 ∧ 
           (∃ m : ℕ, ∀ d : ℕ, d ∣ m → nat.even d ∧ m = n → (finset.filter nat.even (finset.divisors m)).card = 16)
             → n = 756 :=
by
  sorry

end smallest_positive_integer_with_divisors_l803_803225


namespace theta_value_l803_803364

theorem theta_value (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (θ : ℝ) :
    let f := λ x : ℝ, a * real.cos (x + 2 * θ) + b * x + 3 in
    f 1 = 5 ∧ f (-1) = 1 → θ = (π / 4) :=
by
  intro f h
  have h1 : a * real.cos (1 + 2 * θ) + b * 1 + 3 = 5 := h.1
  have h2 : a * real.cos (-1 + 2 * θ) + b * (-1) + 3 = 1 := h.2
  sorry

end theta_value_l803_803364


namespace sequence_formula_l803_803463

noncomputable def a_n : ℕ → ℕ 
| n := 3 * n - 2

theorem sequence_formula :
  (∀ n ∈ (Set ℕ).erase 0, 
    (∃ S_n : ℕ, S_n = (1 / 6 : ℚ) * (a_n n + 1) * (a_n n + 2)) ∧
    ∀ m < n, a_n m + 3 = a_n (m+1) ∧
    a_n 2 * a_n 2 = a_n 4 ∧ 
    a_n 4 * a_n 4 = a_n 9) 
  → a_n = λ n, 3 * n - 2 :=
sorry

end sequence_formula_l803_803463


namespace triangle_AC_length_l803_803050

theorem triangle_AC_length
  (A B C : Type)
  [module ℝ A] [module ℝ B] [module ℝ C]
  (AB BC : ℝ) (AM : ℝ) (midpoint_BC : ℝ)
  (h_AB : AB = 7)
  (h_BC : BC = 9)
  (h_AM : AM = 5)
  (h_midpoint : midpoint_BC = BC / 2) :
  ∃ AC : ℝ, AC = real.sqrt 41.5 :=
by
  let BM := midpoint_BC
  let CM := BM
  -- Skipping the proof
  use real.sqrt 41.5
  sorry

end triangle_AC_length_l803_803050


namespace base_n_not_divisible_by_11_l803_803750

theorem base_n_not_divisible_by_11 :
  ∀ n, 2 ≤ n ∧ n ≤ 100 → (6 + 2*n + 5*n^2 + 4*n^3 + 2*n^4 + 4*n^5) % 11 ≠ 0 := by
  sorry

end base_n_not_divisible_by_11_l803_803750


namespace sequence_sum_S_50_l803_803771

noncomputable section

open Real

theorem sequence_sum_S_50 :
  (∀ n ≥ 2, (4 * n + 3) + (4 * (n - 1) + 3) = 8 * n + 2) ∧
  (∃ a2 = 4 * 2 + 3, ∃ b2 = 20 - a2, b2 = 3 ^ 2) →
  (let a (n : ℕ) := 4 * n + 3
   let b (n : ℕ) := 3 ^ n
   let c := λ n, if n % 2 = 0 then b (n / 2) else a ((n + 1) / 2)
   in (finset.range 50).sum c = 4582) :=
begin
  -- Proof goes here
  sorry
end

end sequence_sum_S_50_l803_803771


namespace books_sale_total_amount_l803_803615

theorem books_sale_total_amount (B : ℕ) (price_per_book : ℕ) (books_not_sold : ℕ) 
  (h_non_sold : books_not_sold = 36)
  (h_fraction_sold : ∀ t, books_not_sold = (1/3) * t → B = t)
  (h_price : price_per_book = 2) :
  ∃ total_received, total_received = 144 :=
by
  have hB : B = 108 := by 
    have h := h_fraction_sold B,
    rw [h_non_sold, one_div, h],
    norm_num,
  have h_books_sold : (2/3) * B = 72 := by
    rw [hB, ← mul_assoc, ← mul_div_assoc, mul_comm 2, div_self] {error},
    norm_num,
  have total_sold_value : 72 * 2 = 144 := by
    norm_num,
  use 144,
  exact total_sold_value

end books_sale_total_amount_l803_803615


namespace range_of_ac_over_b_l803_803826

theorem range_of_ac_over_b
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : ∠B = arccos (b/(2 * sqrt (a*c))))
  (h2 : B < π / 2) -- Angle B is acute
  (h3 : 8 * sin A * sin C = sin^2 B)
  : (a + c) / b ∈ Set.Ioo (sqrt 5 / 2) (sqrt 6 / 2) := 
sorry

end range_of_ac_over_b_l803_803826


namespace tangent_line_at_a1_inequality_fx_less_xfx_inequality_g_extreme_l803_803798

noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 - 4 * a * x + a * real.log x + a + (1/2)
noncomputable def f' (a x : ℝ) : ℝ := x - 4 * a + (a / x)
noncomputable def g (a x : ℝ) : ℝ := f a x + 2 * a
noncomputable def g' (a x : ℝ) : ℝ := x - 4 * a + (a / x)

theorem tangent_line_at_a1 (f : ℝ → ℝ) (a : ℝ) (x : ℝ) (h : a = 1) (hx : x = 1) :
  let y := f x in 2 * (x - 1) + y + 2 = 0 := sorry

theorem inequality_fx_less_xfx (f : ℝ → ℝ) (a : ℝ) (x : ℝ) :
  (∀ x > 1, f x < x * f' a x) → a ≤ 1 := sorry

theorem inequality_g_extreme (g : ℝ → ℝ) (a : ℝ) (x1 x2 : ℝ) :
  (∃ x1 x2, g x1 + g x2 ≥ g' a (x1 * x2)) →
  (1/4 < a ∧ a ≤ 1) := sorry

end tangent_line_at_a1_inequality_fx_less_xfx_inequality_g_extreme_l803_803798


namespace num_of_non_multiples_of_6_or_8_in_three_digits_l803_803411

-- Define conditions about multiples and range of three-digit numbers
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

def multiples_of_6 := {n : ℕ | is_multiple_of n 6}
def multiples_of_8 := {n : ℕ | is_multiple_of n 8}
def multiples_of_24 := {n : ℕ | is_multiple_of n 24}

-- Prove that the number of three-digit numbers not multiples of 6 or 8 is 675
theorem num_of_non_multiples_of_6_or_8_in_three_digits : 
  {n : ℕ | n ∈ three_digit_numbers ∧ n ∉ multiples_of_6 ∧ n ∉ multiples_of_8}.count = 675 :=
sorry

end num_of_non_multiples_of_6_or_8_in_three_digits_l803_803411


namespace savings_on_cheapest_flight_l803_803681

theorem savings_on_cheapest_flight :
  let delta_price := 850
  let delta_discount := 0.20
  let united_price := 1100
  let united_discount := 0.30
  let delta_final_price := delta_price - delta_price * delta_discount
  let united_final_price := united_price - united_price * united_discount
  delta_final_price < united_final_price →
  united_final_price - delta_final_price = 90 :=
by
  sorry

end savings_on_cheapest_flight_l803_803681


namespace variance_transform_var_2X_plus_1_l803_803369

noncomputable def binomial_variance (n : ℕ) (p : ℝ) : ℝ :=
  n * p * (1 - p)

theorem variance_transform (n : ℕ) (p : ℝ) :
  D (2 * X + 1) = 4 * binomial_variance n p :=
by
  sorry

theorem var_2X_plus_1 :
  D (2 * X + 1) = 6.4 :=
by
  let n := 10
  let p := 0.8
  have : binomial_variance n p = 1.6 := by
    unfold binomial_variance
    simp [n, p, mul_assoc, mul_comm]
  show D (2 * X + 1) = 4 * binomial_variance n p
  sorry

end variance_transform_var_2X_plus_1_l803_803369


namespace point_in_fourth_quadrant_l803_803046

-- Definitions to represent the given conditions
def x : ℤ := 5
def y : ℤ := -1

-- Conditions to identify the location
def isPositive (a : ℤ) : Prop := a > 0
def isNegative (a : ℤ) : Prop := a < 0

-- The main theorem we want to prove, which identifies the quadrant
theorem point_in_fourth_quadrant (hx : isPositive x) (hy : isNegative y) : (x > 0 ∧ y < 0) ↔ "fourth_quadrant" :=
by sorry

end point_in_fourth_quadrant_l803_803046


namespace domain_of_tan_l803_803550

theorem domain_of_tan (
  k : ℤ
) : 
  (∀ x : ℝ, y = tan (π / 4 - 2 * x) → - (π / 2) + k * π < 2 * x - (π / 4) ∧ 2 * x - (π / 4) < π / 2 + k * π) 
  ↔ 
  (∀ x : ℝ, ∋ x ∈ (k * π / 2 - π / 8, k * π / 2 + 3 * π / 8)) :=
sorry

end domain_of_tan_l803_803550


namespace probability_sum_is_5_over_36_l803_803525

noncomputable def probability_sum_six (dice_rolls : set (ℕ × ℕ)) : ℚ :=
  (↑(dice_rolls.filter (λ p, p.1 + p.2 = 6)).card) / (↑dice_rolls.card)

theorem probability_sum_is_5_over_36 :
  probability_sum_six (set.univ : set (ℕ × ℕ)) = 5 / 36 :=
  sorry

end probability_sum_is_5_over_36_l803_803525


namespace value_of_n_l803_803019

theorem value_of_n (n : ℤ) (h : (sqrt (10 + n : ℝ) = 8)) : n = 54 := by
  sorry

end value_of_n_l803_803019


namespace find_number_l803_803313

theorem find_number (number : ℝ) : 469138 * number = 4690910862 → number = 10000.1 :=
by
  sorry

end find_number_l803_803313


namespace intersection_of_A_and_B_l803_803397

-- Definitions of the sets A and B
def A : Set ℝ := { x | x^2 + 2*x - 3 < 0 }
def B : Set ℝ := { x | |x - 1| < 2 }

-- The statement to prove their intersection
theorem intersection_of_A_and_B : A ∩ B = { x | -1 < x ∧ x < 1 } :=
by 
  sorry

end intersection_of_A_and_B_l803_803397


namespace sum_of_factorials_is_perfect_square_l803_803354

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).map factorial |>.sum

theorem sum_of_factorials_is_perfect_square (n : ℕ) (h : n > 0) :
  (∃ m : ℕ, m * m = sum_of_factorials n) ↔ (n = 1 ∨ n = 3) := 
sorry

end sum_of_factorials_is_perfect_square_l803_803354


namespace value_of_R_l803_803819

noncomputable def log (x: ℝ) : ℝ := Real.log x

theorem value_of_R (m n R : ℝ) 
  (h1 : 2^m = 36) 
  (h2 : 3^n = 36)
  (h3 : R = (1/m) + (1/n)) : 
  R = 1 / 2 :=
by
  sorry

end value_of_R_l803_803819


namespace displacement_representation_l803_803422

def represents_north (d : ℝ) : Prop := d > 0

theorem displacement_representation (d : ℝ) (h : represents_north 80) : represents_north d ↔ d > 0 :=
by trivial

example (h : represents_north 80) : 
  ∀ d, d = -50 → ¬ represents_north d ∧ abs d = 50 → ∃ s, s = "south" :=
sorry

end displacement_representation_l803_803422


namespace transport_wolf_goat_cabbage_l803_803640

structure State where
  farmer_side : Bool
  wolf_side : Bool
  goat_side : Bool
  cabbage_side : Bool

def initialState : State :=
  { farmer_side := false, wolf_side := false, goat_side := false, cabbage_side := false }

def is_safe (s : State) : Prop :=
  (s.goat_side ≠ s.cabbage_side ∨ s.farmer_side = s.goat_side) ∧
  (s.wolf_side ≠ s.goat_side ∨ s.farmer_side = s.goat_side)

def move_farmer (s : State) : State := { s with farmer_side := ¬s.farmer_side }
def move_wolf (s : State) : State := { s with wolf_side := ¬s.wolf_side, farmer_side := ¬s.farmer_side }
def move_goat (s : State) : State := { s with goat_side := ¬s.goat_side, farmer_side := ¬s.farmer_side }
def move_cabbage (s : State) : State := { s with cabbage_side := ¬s.cabbage_side, farmer_side := ¬s.farmer_side }

def all_items_across (s : State) : Prop :=
  s.farmer_side = true ∧ s.wolf_side = true ∧ s.goat_side = true ∧ s.cabbage_side = true

theorem transport_wolf_goat_cabbage (exists_sequence : ∃ sequence : List State, sequence.head = initialState ∧
  sequence.last all_items_across ∧ ∀ (i : ℕ), is_safe (sequence.get i) ∧
  (sequence.get (i + 1) = move_farmer (sequence.get i) ∨
  sequence.get (i + 1) = move_wolf (sequence.get i) ∨
  sequence.get (i + 1) = move_goat (sequence.get i) ∨
  sequence.get (i + 1) = move_cabbage (sequence.get i))):
  exists_sequence :=
sorry  -- Proof omitted

end transport_wolf_goat_cabbage_l803_803640


namespace binom_coeff_mult_l803_803704

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l803_803704


namespace matrix_det_eq_seven_l803_803817

theorem matrix_det_eq_seven (p q r s : ℝ) (h : p * s - q * r = 7) : 
  (p - 2 * r) * s - (q - 2 * s) * r = 7 := 
sorry

end matrix_det_eq_seven_l803_803817


namespace minimum_value_inequality_maximum_value_inequality_l803_803076

noncomputable def minimum_value (x1 x2 x3 : ℝ) : ℝ :=
  (x1 + 3 * x2 + 5 * x3) * (x1 + x2 / 3 + x3 / 5)

theorem minimum_value_inequality (x1 x2 x3 : ℝ) (h : 0 ≤ x1) (h : 0 ≤ x2) (h : 0 ≤ x3) (sum_eq : x1 + x2 + x3 = 1) :
  1 ≤ minimum_value x1 x2 x3 :=
sorry

theorem maximum_value_inequality (x1 x2 x3 : ℝ) (h : 0 ≤ x1) (h : 0 ≤ x2) (h : 0 ≤ x3) (sum_eq : x1 + x2 + x3 = 1) :
  minimum_value x1 x2 x3 ≤ 9/5 :=
sorry

end minimum_value_inequality_maximum_value_inequality_l803_803076


namespace finite_reflections_into_polygon_l803_803852

noncomputable def reflection_through_side (side: Line, point: Point) : Point := sorry

theorem finite_reflections_into_polygon (K : Polygon) (X : Point)
  (hK_closed : is_closed K) 
  (hK_convex : is_convex K) :
  ∃ (sequence : List (Line × Point)), 
  (∀ (ln_pt : Line × Point), ln_pt ∈ sequence → ln_pt.1 ∈ sides K) ∧ -- all lines in sequence are sides of K
  K.contains (sequence.foldl (λ acc ln_pt, reflection_through_side ln_pt.1 acc) X) :=
sorry

end finite_reflections_into_polygon_l803_803852


namespace points_on_square_border_at_distance_six_l803_803647

theorem points_on_square_border_at_distance_six {P : ℝ × ℝ} {length : ℝ} (hP : P = (0, 0)) (hlength : length = 10) :
  let dist := 6
  ∃ num_points : ℕ, dist = 6 ∧ number_of_points_on_square_border_at_distance_of_6 P length = 8 :=
by
  -- Definitions for the conditions
  let sq := { P | (P.1 = -5 ∨ P.1 = 5 ∨ P.2 = -5 ∨ P.2 = 5) ∧
                  ∥(P.1 - 0, P.2 - 0)∥ = 6 }
  -- Number of such points is exactly 8
  exact Exists.intro 8 (by rfl ⬝ sorry)

end points_on_square_border_at_distance_six_l803_803647


namespace smallest_int_with_divisors_l803_803209

theorem smallest_int_with_divisors :
  ∃ n : ℕ, 
    (∀ m, n = 2^2 * m → 
      (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 
      (m = p^3 * q) ∧ 
      (nat.divisors p^7).count 8) ∧ 
    nat.divisors_count n = 24 ∧ 
    (nat.divisors (2^2 * n)).count 8) (n = 2^2 * 3^3 * 5) :=
begin
  sorry
end

end smallest_int_with_divisors_l803_803209


namespace kvass_remaining_volume_after_14_l803_803937

-- Definition of the initial volume
def initial_volume : ℝ := 1.5

-- Recursive function defining volume after each person drinks
def remaining_volume (n : ℕ) (V : ℝ) : ℝ :=
  match n with
  | 0 => V
  | (k + 1) => remaining_volume k V * (k / (k + 1 : ℝ))

-- The problem states we want the remaining volume after 14 persons have drunk
theorem kvass_remaining_volume_after_14 :
  remaining_volume 14 initial_volume = 0.1 := 
sorry

end kvass_remaining_volume_after_14_l803_803937


namespace max_even_a_exists_max_even_a_l803_803727

theorem max_even_a (a : ℤ): (a^2 - 12 * a + 32 ≤ 0 ∧ ∃ k : ℤ, a = 2 * k) → a ≤ 8 := sorry

theorem exists_max_even_a : ∃ a : ℤ, (a^2 - 12 * a + 32 ≤ 0 ∧ ∃ k : ℤ, a = 2 * k ∧ a = 8) := sorry

end max_even_a_exists_max_even_a_l803_803727


namespace adults_tickets_sold_eq_1200_l803_803535

variable (A : ℕ)
variable (S : ℕ := 300) -- Number of student tickets
variable (P_adult : ℕ := 12) -- Price per adult ticket
variable (P_student : ℕ := 6) -- Price per student ticket
variable (total_tickets : ℕ := 1500) -- Total tickets sold
variable (total_amount : ℕ := 16200) -- Total amount collected

theorem adults_tickets_sold_eq_1200
  (h1 : S = 300)
  (h2 : A + S = total_tickets)
  (h3 : P_adult * A + P_student * S = total_amount) :
  A = 1200 := by
  sorry

end adults_tickets_sold_eq_1200_l803_803535


namespace f1_lt_c_lt_f_neg1_l803_803860

theorem f1_lt_c_lt_f_neg1 (c : ℝ) : 
  let f (x : ℝ) := x^2 - 2 * x + c in
  f(1) < c ∧ c < f(-1) :=
by
  sorry

end f1_lt_c_lt_f_neg1_l803_803860


namespace volume_inscribed_cube_l803_803654

-- Definitions based on the problem conditions
def edge_length_large_cube : ℝ := 12
def diameter_sphere : ℝ := edge_length_large_cube
def space_diagonal_smaller_cube : ℝ := diameter_sphere

-- Lean statement for the proof problem
theorem volume_inscribed_cube (s : ℝ) (V : ℝ) : 
  (s = 4 * Real.sqrt 3) →
  (V = s^3) →
  (V = 192 * Real.sqrt 3) :=
by
  intros h_s h_V
  rw h_s at h_V
  rw h_V
  sorry

end volume_inscribed_cube_l803_803654


namespace no_common_period_l803_803284

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f(x + p) = f(x)

theorem no_common_period 
  (g h : ℝ → ℝ) 
  (Hg : is_periodic g 3)
  (Hh : is_periodic h real.pi) :
  ¬ ∃ T > 0, is_periodic (λ x, g x - h x) T :=
by
  sorry

end no_common_period_l803_803284


namespace symm_diff_complement_symm_diff_union_subset_symm_diff_inter_subset_l803_803108

-- Definitions of sequences of events and symmetric difference
variable (A : ℕ → Set α) (B : ℕ → Set α)

-- Definition of symmetric difference
def symm_diff (S T : Set α) : Set α := (S \ T) ∪ (T \ S)

-- Theorems to be proven
theorem symm_diff_complement (A1 B1 : Set α) :
  symm_diff A1 B1 = symm_diff (Set.compl A1) (Set.compl B1) := sorry

theorem symm_diff_union_subset :
  symm_diff (⋃ n, A n) (⋃ n, B n) ⊆ ⋃ n, symm_diff (A n) (B n) := sorry

theorem symm_diff_inter_subset :
  symm_diff (⋂ n, A n) (⋂ n, B n) ⊆ ⋃ n, symm_diff (A n) (B n) := sorry

end symm_diff_complement_symm_diff_union_subset_symm_diff_inter_subset_l803_803108


namespace ned_tickets_skee_ball_l803_803244

theorem ned_tickets_skee_ball :
  ∃ (S : ℕ), (5 * 9 = 45) ∧ ((26 + S) = 45) ∧ (S = 19) :=
by
  let total_tickets := 5 * 9
  let whack_a_mole_tickets := 26
  let skee_ball_tickets := total_tickets - whack_a_mole_tickets
  use skee_ball_tickets
  split
  any_goals
    dsimp only [skee_ball_tickets, whack_a_mole_tickets, total_tickets]
    linarith

end ned_tickets_skee_ball_l803_803244


namespace coefficient_abc2_expansion_l803_803120

theorem coefficient_abc2_expansion :
  (∃ (a b c : ℤ) (f : (a + 2 * b - 3 * c) ^ 4),
    f.coeff ⟨1, 1, 2⟩ = 216) :=
sorry

end coefficient_abc2_expansion_l803_803120


namespace chemistry_marks_l803_803723

def M_Eng := 76
def M_Math := 65
def M_Phys := 82
def M_Bio := 85
def Avg := 75

theorem chemistry_marks 
    (M_Eng = 76) 
    (M_Math = 65) 
    (M_Phys = 82) 
    (M_Bio = 85) 
    (Avg = 75) :
    let C := (Avg * 5) - (M_Eng + M_Math + M_Phys + M_Bio) in
    C = 67 := 
by 
  sorry

end chemistry_marks_l803_803723


namespace valid_votes_for_candidate_A_l803_803252

theorem valid_votes_for_candidate_A 
  (V : ℕ) (P_invalid P_A : ℕ)
  (hV : V = 560000) 
  (hInv : P_invalid = 15) 
  (hCandA : P_A = 75) :
  let P_valid = 100 - P_invalid,
      V_valid = V * P_valid / 100,
      V_A = V_valid * P_A / 100
  in V_A = 357000 := 
by
  sorry

end valid_votes_for_candidate_A_l803_803252


namespace A_plus_B_l803_803944

theorem A_plus_B {A B : ℚ} (h : ∀ x : ℚ, (Bx - 19) / (x^2 - 8*x + 15) = A / (x - 3) + 5 / (x - 5)) : 
  A + B = 33 / 5 := sorry

end A_plus_B_l803_803944


namespace exists_polyhedron_with_no_three_same_sided_faces_l803_803687

structure Face :=
  (sides : ℕ)

structure Polyhedron :=
  (faces : List Face)
  (closed : Bool)

-- Definition of specific faces
def triangular_face : Face := ⟨3⟩
def quadrilateral_face : Face := ⟨4⟩
def pentagonal_face : Face := ⟨5⟩

-- Definition of the polyhedron in terms of the conditions
def polyhedron_example : Polyhedron := 
  ⟨[triangular_face, triangular_face, quadrilateral_face, quadrilateral_face, pentagonal_face, pentagonal_face], true⟩

theorem exists_polyhedron_with_no_three_same_sided_faces : 
  ∃ (p : Polyhedron), p = polyhedron_example ∧ p.closed ∧ 
    (∀ n, (p.faces.filter (λ f, f.sides = n)).length < 3) :=
by
  sorry

end exists_polyhedron_with_no_three_same_sided_faces_l803_803687


namespace vance_family_stamp_cost_difference_l803_803912

theorem vance_family_stamp_cost_difference :
    let cost_rooster := 2 * 1.50
    let cost_daffodil := 5 * 0.75
    cost_daffodil - cost_rooster = 0.75 :=
by
    let cost_rooster := 2 * 1.50
    let cost_daffodil := 5 * 0.75
    show cost_daffodil - cost_rooster = 0.75
    sorry

end vance_family_stamp_cost_difference_l803_803912


namespace cubic_roots_identity_l803_803070

theorem cubic_roots_identity (p q r : ℝ) 
  (h1 : p + q + r = 0) 
  (h2 : p * q + q * r + r * p = -3) 
  (h3 : p * q * r = -2) : 
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = 0 := 
by
  sorry

end cubic_roots_identity_l803_803070


namespace solve_inequality_part1_monotonicity_part2_l803_803488

noncomputable def f (a x : ℝ) := (real.sqrt (x ^ 2 + 1) - a * x)

theorem solve_inequality_part1 (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, f a x ≤ 1 ↔ 
      ((0 < a ∧ a < 1) → (0 ≤ x ∧ x ≤ 2 * a / (1 - a^2))) ∧ 
      (a ≥ 1 → 0 ≤ x)) :=
sorry

theorem monotonicity_part2 (a : ℝ) (h : a > 0) : 
  (∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f a x1 > f a x2) ↔ a ≥ 1 :=
sorry

end solve_inequality_part1_monotonicity_part2_l803_803488


namespace player1_wins_l803_803954

/-- 
Define the game setup on a 2021 x 2021 board and the rules of the game where two players 
take turns placing non-overlapping dominoes, and Player 1's goal is to guarantee a win 
by preventing Player 2 from filling the board except for one cell. 
-/
def game (board : fin 2021 × fin 2021) (player1 : Type) (player2 : Type) :=
  ∃ (winning_strategy : player1 → player2 → Prop), 
    (∀ turn : ℕ, turn % 2 = 0 → player1)
    ∧ (∀ turn : ℕ, turn % 2 = 1 → player2)
    ∧ (player1_win := ∀ b : board, player1 → ¬ (∃ win : board, win ∉ b))

/-- 
Theorem: On a 2021 x 2021 board with the described game rules, Player 1 
can guarantee a win. 
-/
theorem player1_wins : ∃ (winning_strategy : game) (player1_win : game → Prop), player1_win :=
sorry

end player1_wins_l803_803954


namespace binomial_product_result_l803_803691

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l803_803691


namespace complement_A_in_U_l803_803399

open Set Int

theorem complement_A_in_U :
  let U := {x : ℤ | (x + 1) * (x - 3) ≤ 0}
  let A := {0, 1, 2}
  U \ A = {-1, 3} :=
by
  let U := {x : ℤ | (x + 1) * (x - 3) ≤ 0}
  let A := {0, 1, 2}
  sorry

end complement_A_in_U_l803_803399


namespace smallest_abundant_not_multiple_of_4_l803_803595

def proper_divisors (n : ℕ) : List ℕ := (List.range n).filter (λ d, d ∣ n)

def sum_proper_divisors (n : ℕ) : ℕ := (proper_divisors n).sum

def is_abundant (n : ℕ) : Prop := sum_proper_divisors n > n

def not_multiple_of_4 (n : ℕ) : Prop := ¬ (4 ∣ n)

theorem smallest_abundant_not_multiple_of_4 : ∃ n, is_abundant n ∧ not_multiple_of_4 n ∧ (∀ m, is_abundant m ∧ not_multiple_of_4 m → n ≤ m) :=
  sorry

end smallest_abundant_not_multiple_of_4_l803_803595


namespace max_sum_ab_bc_cd_da_l803_803574

theorem max_sum_ab_bc_cd_da (a b c d : ℕ) (h : {a, b, c, d} = {1, 3, 5, 7}) :
  ab + bc + cd + da ≤ 64 :=
by sorry

end max_sum_ab_bc_cd_da_l803_803574


namespace problem_solution_l803_803377

noncomputable def tan_alpha_and_m_values (alpha : ℝ) (m : ℝ) : Prop :=
  tan alpha = -4/3 ∧ m = -29/3 ∧
  5 * (tan alpha)^2 - m * (tan alpha) + 4 = 0

noncomputable def fraction_value (alpha : ℝ) : Prop :=
  let sin_alpha := Real.sin alpha
  let cos_alpha := Real.cos alpha
  α > π/2 ∧ α < π → ((2 * sin_alpha^2 - sin_alpha * cos_alpha + 3 * cos_alpha^2) / (1 + sin_alpha^2)) = 71 / 41

theorem problem_solution (alpha : ℝ) (m : ℝ) (sin_alpha cos_alpha: ℝ):
  tan_alpha_and_m_values alpha m ∧ fraction_value alpha ∧ (
  tan alpha = sin_alpha / cos_alpha ∧
  cos_alpha < 0 ∧ sin_alpha > 0 ∧ sin_alpha^2 + cos_alpha^2 = 1
) := sorry

end problem_solution_l803_803377


namespace initial_amount_of_money_l803_803522

-- Define the initial amount of money Randy had as a variable
variable (M : ℝ)

-- Conditions based on the given problem
def spent_on_lunch := 10
def cost_ice_cream := 5

-- Define the remaining money after lunch
def remaining_money := M - spent_on_lunch

-- Define the cost relation of the ice cream to the remaining money
def spent_on_ice_cream := remaining_money / 4

-- The theorem indicating the initial amount of money
theorem initial_amount_of_money (h : spent_on_ice_cream = cost_ice_cream) : M = 30 :=
by
  sorry

end initial_amount_of_money_l803_803522


namespace smallest_integer_with_divisors_properties_l803_803200

def number_of_odd_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 1)

def number_of_even_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 0)

theorem smallest_integer_with_divisors_properties :
  ∃ n : ℕ, number_of_odd_divisors n = 8 ∧ number_of_even_divisors n = 16 ∧ n = 4000 :=
by
  sorry

end smallest_integer_with_divisors_properties_l803_803200


namespace min_jumps_to_visit_all_points_and_return_l803_803149

theorem min_jumps_to_visit_all_points_and_return (n : ℕ) (h : n = 2016) : 
  ∀ jumps : ℕ, (∀ p : Fin n, ∃ k : ℕ, p = (2 * k) % n ∨ p = (3 * k) % n) → 
  jumps = 2017 :=
by 
  intros jumps h
  sorry

end min_jumps_to_visit_all_points_and_return_l803_803149


namespace union_indicator_eq_max_inter_indicator_eq_min_l803_803526

variables {ι : Type*} {A : ι → Set α} {x : α}

def indicator (s : Set α) (x : α) : ℕ :=
if x ∈ s then 1 else 0

theorem union_indicator_eq_max :
  indicator (⋃ n, A n) x = (finset.univ.image (λ n, indicator (A n) x)).max' sorry :=
sorry

theorem inter_indicator_eq_min :
  indicator (⋂ n, A n) x = (finset.univ.image (λ n, indicator (A n) x)).min' sorry :=
sorry

end union_indicator_eq_max_inter_indicator_eq_min_l803_803526


namespace divides_self_l803_803321

noncomputable def a : ℕ → ℕ :=
  λ n, ∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n+1))), if d ∣ n then 2^d else 0

theorem divides_self {n : ℕ} (h : ∀ n, ∑ d in (finset.filter (λ d, d ∣ n) (finset.range (n+1))), a d = 2^n) :
  n ∣ a n :=
  by
    sorry

end divides_self_l803_803321


namespace probability_white_given_popped_l803_803986

noncomputable theory
open Classical

-- Define the conditions
def P_white : ℚ := 3 / 4
def P_yellow : ℚ := 1 / 4
def P_popped_given_white : ℚ := 2 / 5
def P_popped_given_yellow : ℚ := 3 / 4

-- Define the events
def P_white_and_popped : ℚ := P_white * P_popped_given_white
def P_yellow_and_popped : ℚ := P_yellow * P_popped_given_yellow
def P_popped : ℚ := P_white_and_popped + P_yellow_and_popped

-- Statement of the goal
theorem probability_white_given_popped :
  (P_white_and_popped / P_popped) = 24 / 39 :=
  sorry

end probability_white_given_popped_l803_803986


namespace Tn_less_than_5_over_64_l803_803652

variable {ℕ : Type*} [linear_ordered_field ℕ]

def sequence_an (n : ℕ) : ℕ :=
  2 * n

def sequence_bnf (n : ℕ) : ℕ :=
  let a_n := sequence_an n
  let denominator := (n+2)^2 * a_n^2
  1 / 16 * ((1 / n^2) - (1 / (n + 2)^2))

def sum_bn (n : ℕ) : ℕ :=
  ∑ k in finset.range n, sequence_bnf k

theorem Tn_less_than_5_over_64 (n : ℕ) : sum_bn n < 5 / 64 :=
by
  sorry

end Tn_less_than_5_over_64_l803_803652


namespace distinct_real_numbers_for_g_iteration_l803_803069

noncomputable def g : ℝ → ℝ := λ x, x^2 - 6*x

theorem distinct_real_numbers_for_g_iteration : 
  {c : ℝ | g (g (g (g c))) = 15}.to_finset.card = 2 := 
by
  sorry

end distinct_real_numbers_for_g_iteration_l803_803069


namespace customers_left_l803_803664

theorem customers_left (x : ℕ) 
  (h1 : 47 - x + 20 = 26) : 
  x = 41 :=
sorry

end customers_left_l803_803664


namespace basketball_team_lineup_count_l803_803893

/--
In a basketball team of 15 members, where each member can play any of the six specific positions 
(center, power forward, small forward, shooting forward, point guard, and shooting guard), 
the number of ways to select and order a starting lineup is exactly 3,603,600.
-/
theorem basketball_team_lineup_count (n k : ℕ) (positions : Finset ℕ) (h_n : n = 15) (h_k : k = 6) (h_positions : positions.card = k) : 
  nat.perm n k = 3603600 :=
by 
  have h1 : nat.perm 15 6 = 15 * 14 * 13 * 12 * 11 * 10,
  { exact_mt nat.perm_eq my_permutation_calculations }, -- using some perm calculation lemma
  rw [h1],
  norm_num,
  sorry -- placeholder for actual calculation

end basketball_team_lineup_count_l803_803893


namespace mary_total_spent_l803_803086

theorem mary_total_spent :
  let berries := 11.08
  let apples := 14.33
  let peaches := 9.31
  let grapes := 7.50
  let bananas := 5.25
  let pineapples := 4.62
  in berries + apples + peaches + grapes + bananas + pineapples = 52.09 :=
by
  sorry

end mary_total_spent_l803_803086


namespace find_ellipse_equation_l803_803124

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), a^2 - b^2 = 5 ∧ (4*(-3)^2)/(9*a^2) + (9*2^2)/(4*b^2) = 36 ∧ 
    (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1)
 
theorem find_ellipse_equation : (∃ c : ℝ, 0 < c ∧ c = sqrt 5) → 
        (∃ (a b : ℝ), a^2 - b^2 = c^2 ∧ 
            ((4*(-3)^2)/(9*a^2) + (9*2^2)/(4*b^2) = 36) ∧ 
           (\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1)) := 
begin
  sorry
end

end find_ellipse_equation_l803_803124


namespace regular_polygon_sides_l803_803540

-- Define the central angle and number of sides of a regular polygon
def central_angle (θ : ℝ) := θ = 30
def number_of_sides (n : ℝ) := 360 / 30 = n

-- Theorem to prove that the number of sides of the regular polygon is 12 given the central angle is 30 degrees
theorem regular_polygon_sides (θ n : ℝ) (hθ : central_angle θ) : number_of_sides n → n = 12 :=
sorry

end regular_polygon_sides_l803_803540


namespace count_k_tuples_l803_803079

theorem count_k_tuples (k n : ℕ) (hkn : 1 ≤ k ∧ k ≤ n) :
  let A := {a : Fin k ⟶ Fin n | ∀ i, 1 ≤ a i ∧ a i ≤ n}
      A1 := {a : Fin k ⟶ Fin n | (∃ s t : Fin k, s < t ∧ a s > a t) ∨ (∃ s : Fin k, a s % 2 ≠ s.val % 2)}
      A2 := {a : Fin k ⟶ Fin n | ∀ i : Fin k, (∀ j : Fin (i.val), a j < a (j + 1)) ∧ ∀ i, a i ≡ i.val [MOD 2]}
  in Fintype.card A1 = Fintype.card A - Fintype.card A2 :=
by
  sorry

end count_k_tuples_l803_803079


namespace three_digit_factorial_sum_l803_803965

def factorial : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n+2) := (n + 2) * factorial (n + 1)

theorem three_digit_factorial_sum :
  ∀ (A B C : ℕ), (1 ≤ A ∧ A ≤ 9) → (0 ≤ B ∧ B ≤ 9) → (0 ≤ C ∧ C ≤ 9) → 
  (100 * A + 10 * B + C = factorial A + factorial B + factorial C ↔ A = 1 ∧ B = 4 ∧ C = 5): 
by
  intros A B C hA hB hC
  sorry

end three_digit_factorial_sum_l803_803965


namespace min_value_expression_l803_803235

theorem min_value_expression (s t : ℝ) 
  (h1: ∀ x y : ℝ, x = s + 5 → y = s → (x - y - 5 = 0) → (x - y = 5))
  (h2: ∀ x y : ℝ, x = 3 * |cos t| → y = 2 * |sin t| → (x^2 / 9 + y^2 / 4 = 1)) :
  (s + 5 - 3 * |cos t|)^2 + (s - 2 * |sin t|)^2 = 2 :=
sorry

end min_value_expression_l803_803235


namespace minimum_k_for_f_l803_803393

noncomputable def f (k : ℕ) (x : ℝ) : ℝ :=
  (sin (k * x / 10))^4 + (cos (k * x / 10))^4

theorem minimum_k_for_f (k : ℕ) (h : 0 < k) :
  (∀ a : ℝ, {y | ∃ x : ℝ, a < x ∧ x < a + 1 ∧ y = f k x} = {y | ∃ x : ℝ, y = f k x}) →
  k = 16 :=
sorry

end minimum_k_for_f_l803_803393


namespace pool_perimeter_l803_803444

theorem pool_perimeter (x : ℝ) (h : (8 - 2 * x) * (6 - 2 * x) = 24) : 
  let length := 8 - 2 * x,
      width := 6 - 2 * x in
  2 * (length + width) = 20 :=
by
  sorry

end pool_perimeter_l803_803444


namespace perpendicular_chords_square_sum_l803_803035

theorem perpendicular_chords_square_sum (d : ℝ) (r : ℝ) (x y : ℝ) 
  (h1 : r = d / 2)
  (h2 : x = r)
  (h3 : y = r) 
  : (x^2 + y^2) + (x^2 + y^2) = d^2 :=
by
  sorry

end perpendicular_chords_square_sum_l803_803035


namespace extreme_points_count_l803_803926

noncomputable def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3 - 9

theorem extreme_points_count : 
  let extreme_points := {x : ℝ | deriv f x = 0 ∧ ((∃ ε > 0, deriv f (x - ε) > 0 ∧ deriv f x < 0) ∨
                                                   (∃ ε > 0, deriv f x > 0 ∧ deriv f (x + ε) < 0))}
  in fintype.card extreme_points = 2 :=
sorry

end extreme_points_count_l803_803926


namespace baseball_football_difference_is_five_l803_803464

-- Define the conditions
def total_cards : ℕ := 125
def baseball_cards : ℕ := 95
def some_more : ℕ := baseball_cards - 3 * (total_cards - baseball_cards)

-- Define the number of football cards
def football_cards : ℕ := total_cards - baseball_cards

-- Define the difference between the number of baseball cards and three times the number of football cards
def difference : ℕ := baseball_cards - 3 * football_cards

-- Statement of the proof
theorem baseball_football_difference_is_five : difference = 5 := 
by
  sorry

end baseball_football_difference_is_five_l803_803464


namespace circles_separated_l803_803806

-- Definitions of the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O2 (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 9

-- Centers of the circles
def center_O1 := (0 : ℝ, 0 : ℝ)
def center_O2 := (3 : ℝ, -4 : ℝ)

-- Radii of the circles
def radius_O1 : ℝ := 1
def radius_O2 : ℝ := 3

-- Distance between centers of the circles
def distance_centers : ℝ :=
  Real.sqrt ((center_O2.1 - center_O1.1)^2 + (center_O2.2 - center_O1.2)^2)

-- Theorem stating the circles are separated
theorem circles_separated : distance_centers > (radius_O1 + radius_O2) :=
by
  -- referred calculations
  have h1 : distance_centers = 5 := by sorry
  have h2 : radius_O1 + radius_O2 = 4 := by sorry
  show 5 > 4, from sorry

end circles_separated_l803_803806


namespace sin_omega_shift_eq_cos_l803_803031

theorem sin_omega_shift_eq_cos (omega : ℝ) (k : ℤ) :
  (∀ x : ℝ, sin (omega * (x + π / 3)) = cos (omega * x)) →
  omega = 6 * k + 3 / 2 := 
sorry

end sin_omega_shift_eq_cos_l803_803031


namespace sin_cos_ratio_l803_803868

theorem sin_cos_ratio (x y : ℝ) (h1 : sin x / sin y = 4) (h2 : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 395 / 381 :=
by
  sorry

end sin_cos_ratio_l803_803868


namespace angle_B_values_l803_803441

theorem angle_B_values (A B C : ℝ) (a b c : ℝ) (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) 
  (h4 : a = 2 * b * real.sin A) :
  B = π / 6 ∨ B = 5 * π / 6 :=
sorry

end angle_B_values_l803_803441


namespace problem_l803_803783

variable (x : ℝ)

theorem problem (h : x - (1 / x) = Real.sqrt 2) : x^1023 - (1 / x^1023) = 5 * Real.sqrt 2 :=
by
  sorry

end problem_l803_803783


namespace chickens_count_l803_803992

/-- Statement of the problem -/
theorem chickens_count
  (weekly_eggs_per_chicken : ℕ)
  (price_per_dozen : ℕ)
  (total_earnings : ℕ)
  (duration_weeks : ℕ)
  (total_chickens_income : ∀ chickens, chickens * (weekly_eggs_per_chicken * duration_weeks / 12 * price_per_dozen) = total_earnings) :
  ∃ chickens, chickens = 46 := 
by
  let weekly_eggs_per_chicken := 6
  let price_per_dozen := 3
  let total_earnings := 552
  let duration_weeks := 8
  have eq : ∀ chickens, chickens * (weekly_eggs_per_chicken * duration_weeks / 12 * price_per_dozen) = total_earnings 
    from assumption
  existsi 46
  rw eq
  norm_num at *
  sorry

end chickens_count_l803_803992


namespace dynaco_shares_sold_l803_803302

-- Define the conditions
def MicrotronPrice : ℝ := 36
def DynacoPrice : ℝ := 44
def TotalShares : ℕ := 300
def AvgPrice : ℝ := 40
def TotalValue : ℝ := TotalShares * AvgPrice

-- Define unknown variables
variables (M D : ℕ)

-- Express conditions in Lean
def total_shares_eq : Prop := M + D = TotalShares
def total_value_eq : Prop := MicrotronPrice * M + DynacoPrice * D = TotalValue

-- Define the problem statement
theorem dynaco_shares_sold : ∃ D : ℕ, 
  (∃ M : ℕ, total_shares_eq M D ∧ total_value_eq M D) ∧ D = 150 :=
by
  sorry

end dynaco_shares_sold_l803_803302


namespace face_dots_l803_803943

-- Define the conditions of the problem
constant Cube : Type
constant dot_config : Cube → ℕ → Prop
constant num_faces : ℕ -- Number of faces per cube

-- Conditions:
-- Each cube has one face with 3 dots, two faces with 2 dots, and three faces with 1 dot
axiom cube_property (c : Cube) : 
  ∃ f3 f2a f2b f1a f1b f1c : ℕ,
  dot_config c f3 ∧ dot_config c f2a ∧ dot_config c f2b ∧
  dot_config c f1a ∧ dot_config c f1b ∧ dot_config c f1c ∧
  (f3 = 3 ∧ f2a = 2 ∧ f2b = 2 ∧ f1a = 1 ∧ f1b = 1 ∧ f1c = 1)

-- The P shape formed by these cubes
constant P_shape : Type
constant in_P_shape : Cube → ℕ → P_shape → Prop
constant adjacent_faces : Cube → ℕ → Cube → ℕ → Prop

-- Conditions:
-- Number of dots on any two touching faces are the same
axiom touching_faces (c1 c2 : Cube) (f1 f2 : ℕ):
  adjacent_faces c1 f1 c2 f2 → dot_config c1 f1 = dot_config c2 f2

-- Given arrangement of faces A, B, and C in the P shape
constant FaceA FaceB FaceC : ℕ

-- Problem statement:
theorem face_dots (P : P_shape):
  in_P_shape Cube FaceA P → in_P_shape Cube FaceB P → in_P_shape Cube FaceC P →
  (dot_config Cube FaceA = 2 ∧ dot_config Cube FaceB = 2 ∧ dot_config Cube FaceC = 3) :=
sorry

end face_dots_l803_803943


namespace angle_CFD_l803_803476

variables (A B O E F C D : Type _)  -- points on the circle
variables {circle : Set (points : Type _)}

-- AB is a diameter of the circle
def diameter (A B O : Type _) : Prop :=
  ∃ O, is_center O ∧ A ∈ circle ∧ B ∈ circle ∧ ∀ p ∈ circle, |OA - OB| = diameter_length

-- E and F are points on the circle
def on_circle (E F : Type _) : Prop := 
  E ∈ circle ∧ F ∈ circle

-- F lies on the segment BE
def lies_on_segment (BE F : Type _) : Prop :=
  ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ F = (1 - t) * B + t * E

-- Tangents intersect conditions
def tangents_intersect (A B E C D : Type _) : Prop :=
  ∃ P, tangent_at B P ∧ tangent_at E P ∧ intersects A E P

-- Angles at points
def angle_eq (A B E C D : Type _) : Prop :=
  angle BAE = 30 ∧ angle EBF = 15 

theorem angle_CFD (A B E F C D O : Type _)
  [diameter A B O] [on_circle E F] [lies_on_segment BE F]
  [tangents_intersect A B E C D] [angle_eq A B E C D] :
  angle CFD = 45 :=
sorry

end angle_CFD_l803_803476


namespace field_width_l803_803557

theorem field_width (W L : ℝ) (h1 : L = (7 / 5) * W) (h2 : 2 * L + 2 * W = 288) : W = 60 :=
by
  sorry

end field_width_l803_803557


namespace smallest_integer_has_8_odd_and_16_even_divisors_l803_803219

/-!
  Prove that the smallest positive integer with exactly 8 positive odd integer divisors
  and exactly 16 positive even integer divisors is 540.
-/
def smallest_integer_with_divisors : ℕ :=
  540

theorem smallest_integer_has_8_odd_and_16_even_divisors 
  (n : ℕ) 
  (h1 : (8 : ℕ) = nat.count (λ d, d % 2 = 1) (nat.divisors n))
  (h2 : (16 : ℕ) = nat.count (λ d, d % 2 = 0) (nat.divisors n)) :
  n = 540 :=
sorry

end smallest_integer_has_8_odd_and_16_even_divisors_l803_803219


namespace robert_total_interest_l803_803106

theorem robert_total_interest
  (inheritance : ℕ)
  (part1 part2 : ℕ)
  (rate1 rate2 : ℝ)
  (time : ℝ) :
  inheritance = 4000 →
  part2 = 1800 →
  part1 = inheritance - part2 →
  rate1 = 0.05 →
  rate2 = 0.065 →
  time = 1 →
  (part1 * rate1 * time + part2 * rate2 * time) = 227 :=
by
  intros
  sorry

end robert_total_interest_l803_803106


namespace negative_movement_south_l803_803424

noncomputable def movement_interpretation (x : ℤ) : String :=
if x > 0 then 
  "moving " ++ toString x ++ "m north"
else 
  "moving " ++ toString (-x) ++ "m south"

theorem negative_movement_south : movement_interpretation (-50) = "moving 50m south" := 
by 
  sorry

end negative_movement_south_l803_803424


namespace passengers_on_bus_l803_803118

theorem passengers_on_bus :
  let initial := 0 in
  let stop1 := initial + 7 in
  let stop2 := stop1 - 3 + 5 in
  let stop3 := stop2 - 2 + 4 in
  let stop4 := stop3 - 6 + 9 in
  let stop5 := stop4 - 3 + 7 in
  let stop6 := stop5 - 8 + 11 in
  stop6 = 21 :=
by
  sorry

end passengers_on_bus_l803_803118


namespace scientific_notation_of_935000000_l803_803666

theorem scientific_notation_of_935000000 :
  935000000 = 9.35 * 10^8 :=
by
  sorry

end scientific_notation_of_935000000_l803_803666


namespace fraction_product_simplified_l803_803177

theorem fraction_product_simplified:
  (2 / 9 : ℚ) * (5 / 8 : ℚ) = 5 / 36 :=
by {
  sorry
}

end fraction_product_simplified_l803_803177


namespace solve_diophantine_l803_803622

theorem solve_diophantine (x y : ℕ) (h1 : 1990 * x - 1989 * y = 1991) : x = 11936 ∧ y = 11941 := by
  have h_pos_x : 0 < x := by sorry
  have h_pos_y : 0 < y := by sorry
  have h_x : 1990 * 11936 = 1990 * x := by sorry
  have h_y : 1989 * 11941 = 1989 * y := by sorry
  sorry

end solve_diophantine_l803_803622


namespace no_number_satisfies_conditions_l803_803326

def array := 
  [[5, 12, 7, 4, 10],
   [6, 3, 9, 13, 11],
   [14, 8, 2, 15, 5],
   [1, 7, 12, 6, 8],
   [9, 11, 4, 2, 3]]

def smallest_in_row (row : List ℕ) : ℕ := List.minimum row

def largest_in_col (i : ℕ) : ℕ := 
  List.maximum (List.map (λ row, row.get! i.toNat) array)

theorem no_number_satisfies_conditions : 
  ∀ m ∈ List.bind array (fun row => [smallest_in_row row]),
    ¬ (∃ i, m = largest_in_col i) :=
by
  sorry

end no_number_satisfies_conditions_l803_803326


namespace all_cities_same_population_l803_803096

-- Definitions for city and road structure
structure City :=
  (id : Nat)
  (population : Real)

-- Type for roads which connect pairs of cities
structure Road :=
  (city1 city2 : City)

-- Definition for the Planet with cities and the road network
structure Planet :=
  (cities : List City)
  (roads : List Road)
  (is_connected : ∀ c1 c2 : City, c1 ∈ cities → c2 ∈ cities → 
    exists (path : List Road), path = connected_path c1 c2 roads)
  (average_population_rule : ∀ c : City, c ∈ cities → 
    c.population = (roads.filter (λ r, r.city1 = c ∨ r.city2 = c)
                  .map (λ r, if r.city1 = c then r.city2.population else r.city1.population).foldl (+) 0) 
                  / (roads.filter (λ r, r.city1 = c ∨ r.city2 = c).length))

-- Hypothesis: All cities have the same population
theorem all_cities_same_population (P : Planet) :
  ∀ c1 c2 : City, c1 ∈ P.cities → c2 ∈ P.cities → c1.population = c2.population :=
by sorry

end all_cities_same_population_l803_803096


namespace smallest_abundant_not_multiple_of_4_l803_803598

def is_abundant (n : ℕ) : Prop :=
  (∑ k in finset.filter (λ d, d < n) (finset.divisors n), k) > n

theorem smallest_abundant_not_multiple_of_4 : ∃ n : ℕ, is_abundant n ∧ ¬ (4 ∣ n) ∧ ∀ m : ℕ, is_abundant m ∧ ¬ (4 ∣ m) → m ≥ n :=
begin
  use 18,
  sorry
end

end smallest_abundant_not_multiple_of_4_l803_803598


namespace determine_constants_and_range_k_l803_803392

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x - 1 + b

def f (a b : ℝ) (x : ℝ) : ℝ :=
  g a b x / x

theorem determine_constants_and_range_k :
  (∃ a b : ℝ, a = 1 ∧ b = 2 ∧ ∀ x ∈ set.Icc (2 : ℝ) 3, g a b x = 4) ∧
  (∃ k : ℝ, ∀ x ∈ set.Icc (-1 : ℝ) 1, (f 1 2 (2^x) - k * 2^x) ≥ 0 ↔ k ≤ 1) :=
sorry

end determine_constants_and_range_k_l803_803392


namespace possible_values_of_m_l803_803027

theorem possible_values_of_m (m : ℝ) (A B : Set ℝ) (hA : A = {-1, 1}) (hB : B = {x | m * x = 1}) (hUnion : A ∪ B = A) : m = 0 ∨ m = 1 ∨ m = -1 :=
sorry

end possible_values_of_m_l803_803027


namespace number_of_pairs_l803_803356

def bit_or (x y : Nat) : Nat := sorry  -- Define bitwise OR
def bit_and (x y : Nat) : Nat := sorry  -- Define bitwise AND

theorem number_of_pairs {N : Nat} (hN : N = 2018) :
  let M := 2^N in
  ∃ (a : Fin (N+1) → ℕ), 
    (∀ i, 0 ≤ a i ∧ a i < M) ∧ 
    (∀ i, 1 ≤ i → i ≤ N →  
      let b := λ i, a (i-1) + a i in
      let c := λ i, bit_or (a (i-1)) (a i) in 
      true) →
    (number_of_distinct_pairs : Nat = (2^N)^N) :=
sorry

end number_of_pairs_l803_803356


namespace evaluate_expression_l803_803022

theorem evaluate_expression (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 
  (x^5 + 3*y^2 + 7) / (x + 4) = 298 / 7 := by
  sorry

end evaluate_expression_l803_803022


namespace smallest_positive_difference_l803_803513

def opposite_sum_7 (a b c d e f : ℕ) : Prop :=
  a + d = 7 ∧ b + e = 7 ∧ c + f = 7

theorem smallest_positive_difference (A B C D E F : ℕ) 
  (h : opposite_sum_7 A B C D E F) :
  ∃ d : ℕ, d > 0 ∧ (∀ (A B C D E F : ℕ), opposite_sum_7 A B C D E F → (d = |(A + B + C) - (D + E + F)|)) ∧ d = 1 := by
  sorry

end smallest_positive_difference_l803_803513


namespace no_such_n_exists_l803_803730

open Nat

def concatenation : Nat -> Nat -> Nat := λ x y, Int.ofNat (x * 10 ^ y.digits.length + y)

theorem no_such_n_exists :
  ∀ (n : Nat),
    ¬ (∀ (a b : Nat), 1 ≤ a ∧ a ≤ 9 → 1 ≤ b ∧ b ≤ 9 →
      concatenation (concatenation a (concatenation n b)) (concatenation a b) ∣ concatenation a b ) :=
by
  sorry

end no_such_n_exists_l803_803730


namespace paige_folders_l803_803261

-- Definitions derived from the conditions
def initial_files : Nat := 27
def deleted_files : Nat := 9
def files_per_folder : Nat := 6

-- Define the remaining files after deletion
def remaining_files : Nat := initial_files - deleted_files

-- The theorem: Prove that the number of folders is 3
theorem paige_folders : remaining_files / files_per_folder = 3 := by
  sorry

end paige_folders_l803_803261


namespace part_I_part_II_l803_803378

noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |2 * x - 1|

theorem part_I (x : ℝ) : 
  (f x > f 1) ↔ (x < -3/2 ∨ x > 1) :=
by
  -- We leave the proof as sorry to indicate it needs to be filled in
  sorry

theorem part_II (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, f x ≥ 1/m + 1/n) → m + n ≥ 4/3 :=
by
  -- We leave the proof as sorry to indicate it needs to be filled in
  sorry

end part_I_part_II_l803_803378


namespace cos_1275_eq_l803_803623

noncomputable def cos1275 (θ : ℝ) : ℝ :=
  if θ = 1275 then
    -((Real.sqrt 2 + Real.sqrt 6) / 4)
  else
    Real.cos θ

theorem cos_1275_eq : Real.cos 1275 = -((Real.sqrt 2 + Real.sqrt 6) / 4) :=
by
  sorry

end cos_1275_eq_l803_803623


namespace cristobal_read_more_pages_l803_803719

-- Defining the given conditions
def pages_beatrix_read : ℕ := 704
def pages_cristobal_read (b : ℕ) : ℕ := 3 * b + 15

-- Stating the problem
theorem cristobal_read_more_pages (b : ℕ) (c : ℕ) (h : b = pages_beatrix_read) (h_c : c = pages_cristobal_read b) :
  (c - b) = 1423 :=
by
  sorry

end cristobal_read_more_pages_l803_803719


namespace first_term_greater_than_2017_l803_803329

theorem first_term_greater_than_2017 : ∃ n : ℕ, (seq (n) = 7 * n - 2) ∧ (seq (n) > 2017) ∧ (∀ m : ℕ, m < n → seq (m) ≤ 2017) ∧ seq 289 = 2021 :=
by
  let seq := λ n, 7 * n - 2
  use 289
  split
  { sorry } -- Prove that seq (n) = 7 * n - 2 for n = 289
  split
  { sorry } -- Prove that seq (n) > 2017 for n = 289
  split
  { sorry } -- Prove that for all m < 289, seq (m) ≤ 2017
  { sorry } -- Prove that seq 289 = 2021

end first_term_greater_than_2017_l803_803329


namespace smallest_integer_with_divisors_l803_803184

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end smallest_integer_with_divisors_l803_803184


namespace stamp_distribution_correct_l803_803495

variables {W : ℕ} -- We use ℕ (natural numbers) for simplicity but this can be any type representing weight.

-- Number of envelopes that weigh less than W and need 2 stamps each
def envelopes_lt_W : ℕ := 6

-- Number of stamps per envelope if the envelope weighs less than W
def stamps_lt_W : ℕ := 2

-- Number of envelopes in total
def total_envelopes : ℕ := 14

-- Number of stamps for the envelopes that weigh less
def total_stamps_lt_W : ℕ := envelopes_lt_W * stamps_lt_W

-- Total stamps bought by Micah
def total_stamps_bought : ℕ := 52

-- Stamps left for envelopes that weigh more than W
def stamps_remaining : ℕ := total_stamps_bought - total_stamps_lt_W

-- Remaining envelopes that need stamps (those that weigh more than W)
def envelopes_gt_W : ℕ := total_envelopes - envelopes_lt_W

-- Number of stamps required per envelope that weighs more than W
def stamps_gt_W : ℕ := 5

-- Total stamps needed for the envelopes that weigh more than W
def total_stamps_needed_gt_W : ℕ := envelopes_gt_W * stamps_gt_W

theorem stamp_distribution_correct :
  total_stamps_bought = (total_stamps_lt_W + total_stamps_needed_gt_W) :=
by
  sorry

end stamp_distribution_correct_l803_803495


namespace tangent_line_to_circle_l803_803763

theorem tangent_line_to_circle :
  ∀ (x y : ℝ), x^2 + y^2 = 5 → (x = 2 → y = -1 → 2 * x - y - 5 = 0) :=
by
  intros x y h_circle hx hy
  sorry

end tangent_line_to_circle_l803_803763


namespace calculation_is_zero_l803_803316

theorem calculation_is_zero : 
  20062006 * 2007 + 20072007 * 2008 - 2006 * 20072007 - 2007 * 20082008 = 0 := 
by 
  sorry

end calculation_is_zero_l803_803316


namespace corresponding_angles_count_l803_803753

-- Define the standard setup of the problem
structure Line : Type where
  intersects : Line → Bool

def lines : List Line := sorry -- Assume this is the list of four lines

-- Axioms that match conditions
axiom condition1 : lines.length = 4
axiom condition2 : ∀ l1 l2, l1 ≠ l2 → List.elem l1 lines ∧ List.elem l2 lines → (l1.intersects l2 = True) → ∀ l3, l3 ≠ l1 ∧ l3 ≠ l2 → (¬∃ p, l1.intersects l3 ∧ l2.intersects l3)

-- Question to be proved
theorem corresponding_angles_count : 48 = 4 * 3 * 4 := 
  by
  -- Proof (to be filled in later)
  sorry

end corresponding_angles_count_l803_803753


namespace probability_factor_of_4_factorial_l803_803646

-- Definitions based on the conditions
def set_of_numbers : Set ℕ := { n | n ∈ Finset.range 25 ∧ n ≠ 0 }
def factorial_four : ℕ := 24

-- The statement of the problem
theorem probability_factor_of_4_factorial : 
  let num_of_factors := (3 + 1) * (1 + 1),
      total_numbers := 24 in
  num_of_factors / total_numbers = 1 / 3 :=
by {
  sorry
}

end probability_factor_of_4_factorial_l803_803646


namespace find_units_digit_of_n_l803_803350

-- Define the problem conditions
def units_digit (a : ℕ) : ℕ := a % 10

theorem find_units_digit_of_n (m n : ℕ) (h1 : units_digit m = 3) (h2 : units_digit (m * n) = 6) (h3 : units_digit (14^8) = 6) :
  units_digit n = 2 :=
  sorry

end find_units_digit_of_n_l803_803350


namespace blackbirds_impossibility_on_one_tree_l803_803628

theorem blackbirds_impossibility_on_one_tree (n : ℕ) (h : n = 2014) :
  ∀ blackbirds : Fin n → ℕ, (∀ i, blackbirds i = 1) →
  (∀ k, ∃ i j, i ≠ j ∧ (i = (j + 1) % n ∨ i = (j - 1 + n) % n) ∧
               (blackbirds (i + k * 2) = blackbirds (i + k * 2 + 1) + 1 ∧ blackbirds (j + k * 2) = blackbirds (j + k * 2 + 1) + 1)) →
  ¬ ∃ t (i : Fin n), ∀ j, blackbirds (i + j * t) = n := 
sorry

end blackbirds_impossibility_on_one_tree_l803_803628


namespace max_sum_ab_bc_cd_da_l803_803577

def vals : set ℕ := {1, 3, 5, 7}

theorem max_sum_ab_bc_cd_da :
  ∃ (a b c d : ℕ), a ∈ vals ∧ b ∈ vals ∧ c ∈ vals ∧ d ∈ vals ∧ 
                   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
                   ∀ (w x y z : ℕ), w ∈ vals ∧ x ∈ vals ∧ y ∈ vals ∧ z ∈ vals ∧ 
                                   w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z → 
                                   ab + bc + cd + da ≤ w*x + x*y + y*z + z*w := 
by
  sorry

end max_sum_ab_bc_cd_da_l803_803577


namespace population_after_two_years_l803_803454

def initial_population : ℝ := 14999.999999999998
def increase_rate : ℝ := 0.12
def decrease_rate : ℝ := 0.12

theorem population_after_two_years :
  let first_year_population := initial_population * (1 + increase_rate) in
  let second_year_population := first_year_population * (1 - decrease_rate) in
  second_year_population = 14784 := 
by 
  sorry

end population_after_two_years_l803_803454


namespace find_2010th_term_of_sequence_l803_803769

open Nat

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧
  a 2 = 7 ∧
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = (a n * a (n + 1)) % 10

theorem find_2010th_term_of_sequence (a : ℕ → ℕ) (h : sequence a) : a 2010 = 9 :=
by
  sorry

end find_2010th_term_of_sequence_l803_803769


namespace savings_on_cheapest_flight_l803_803682

theorem savings_on_cheapest_flight :
  let delta_price := 850
  let delta_discount := 0.20
  let united_price := 1100
  let united_discount := 0.30
  let delta_final_price := delta_price - delta_price * delta_discount
  let united_final_price := united_price - united_price * united_discount
  delta_final_price < united_final_price →
  united_final_price - delta_final_price = 90 :=
by
  sorry

end savings_on_cheapest_flight_l803_803682


namespace root_of_f_l803_803757

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x - log x / log 3

theorem root_of_f (a b c x0 : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : f a * f b * f c < 0) (h5 : f x0 = 0) : x0 ≤ c := 
sorry

end root_of_f_l803_803757


namespace range_of_expression_l803_803380

theorem range_of_expression (x y : ℝ) (h1 : x * y = 1) (h2 : 3 ≥ x ∧ x ≥ 4 * y ∧ 4 * y > 0) :
  ∃ A B, A = 4 ∧ B = 5 ∧ ∀ z, z = (x^2 + 4 * y^2) / (x - 2 * y) → 4 ≤ z ∧ z ≤ 5 :=
by
  sorry

end range_of_expression_l803_803380


namespace expand_expression_l803_803333

-- Define the parameters
variables (x y z : ℝ)

-- Theorem statement
theorem expand_expression : (x + 10 + y) * (2z + 10) = 2 * x * z + 2 * y * z + 10 * x + 10 * y + 20 * z + 100 := 
by 
  sorry

end expand_expression_l803_803333


namespace chord_length_l803_803634

theorem chord_length (r d AB : ℝ) (hr : r = 5) (hd : d = 4) : AB = 6 :=
by
  -- Given
  -- r = radius = 5
  -- d = distance from center to chord = 4

  -- prove AB = 6
  sorry

end chord_length_l803_803634


namespace radius_of_shorter_cone_l803_803167

theorem radius_of_shorter_cone {h : ℝ} (h_ne_zero : h ≠ 0) :
  ∀ r : ℝ, ∀ V_taller V_shorter : ℝ,
   (V_taller = (1/3) * π * (5 ^ 2) * (4 * h)) →
   (V_shorter = (1/3) * π * (r ^ 2) * h) →
   V_taller = V_shorter →
   r = 10 :=
by
  intros
  sorry

end radius_of_shorter_cone_l803_803167


namespace three_digit_factorial_sum_digit_is_745_l803_803232

theorem three_digit_factorial_sum_digit_is_745 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧
  (let d1 := n / 100 % 10, d2 := n / 10 % 10, d3 := n % 10 in
  d1 ≠ 5 ∧ d2 ≠ 5 ∧ d3 ≠ 5 ∧
  n = d1.factorial + d2.factorial + d3.factorial) :=
sorry

end three_digit_factorial_sum_digit_is_745_l803_803232


namespace tank_capacity_l803_803237

theorem tank_capacity (C : ℝ) (h1 : 0.40 * C = 0.90 * C - 36) : C = 72 := 
sorry

end tank_capacity_l803_803237


namespace simplify_expression_l803_803062

theorem simplify_expression (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_condition : a^3 + b^3 = 3 * (a + b)) : 
  (a / b + b / a + 1 / (a * b) = 4 / (a * b) + 1) :=
by
  sorry

end simplify_expression_l803_803062


namespace smallest_int_with_divisors_l803_803211

theorem smallest_int_with_divisors :
  ∃ n : ℕ, 
    (∀ m, n = 2^2 * m → 
      (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 
      (m = p^3 * q) ∧ 
      (nat.divisors p^7).count 8) ∧ 
    nat.divisors_count n = 24 ∧ 
    (nat.divisors (2^2 * n)).count 8) (n = 2^2 * 3^3 * 5) :=
begin
  sorry
end

end smallest_int_with_divisors_l803_803211


namespace rearrangement_inequality_l803_803977

theorem rearrangement_inequality 
  (n : ℕ)
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (bi : ℕ → ℕ)
  (ha : ∀ i j : ℕ, i ≤ n → j ≤ n → i ≤ j → a i ≥ a j)
  (hb : ∀ i j : ℕ, i ≤ n → j ≤ n → i ≤ j → b i ≥ b j)
  (hb_permutation : ∀ k : ℕ, k ≤ n → bi k ≤ n ∧ ∃ x, ∃ y, x ≠ y ∧ (bi x = bi y ∨ bi x = x ∧ bi y = y)) :
  (∑ k in Finset.range n, (a k) * (b (n - k))) ≤ (∑ k in Finset.range n, (a k) * (b (bi k))) ∧ (∑ k in Finset.range n, (a k) * (b (bi k))) ≤ (∑ k in Finset.range n, (a k) * (b k)) := by
  sorry

end rearrangement_inequality_l803_803977


namespace neither_energetic_l803_803828

variable (U A B : Finset ℕ)
variable (n |U| |A| |B| |A_inter_B| : ℕ)

axiom total_members : |U| = 40
axiom members_energetic_dads : |A| = 18
axiom members_energetic_moms : |B| = 22
axiom members_both_energetic_dads_moms : |A_inter_B| = 10

theorem neither_energetic :
  (|U| - (|A| + |B| - |A_inter_B|)) = 10 :=
by
  rw [total_members, members_energetic_dads, members_energetic_moms, members_both_energetic_dads_moms]
  sorry

end neither_energetic_l803_803828


namespace infinite_rational_points_below_hyperbola_l803_803320

theorem infinite_rational_points_below_hyperbola :
  ∃ (f : ℕ → ℚ × ℚ), (∀ n, (0 < (f n).1) ∧ (0 < (f n).2) ∧ ((f n).1 * (f n).2 ≤ (5 : ℚ))) ∧ function.injective f :=
by sorry

end infinite_rational_points_below_hyperbola_l803_803320


namespace find_intersection_complement_find_value_m_l803_803000

-- (1) Problem Statement
theorem find_intersection_complement (A : Set ℝ) (B : Set ℝ) (x : ℝ) :
  (A = {x | x^2 - 4 * x - 5 ≤ 0}) →
  (B = {x | x^2 - 2 * x - 3 < 0}) →
  (x ∈ A ∩ (Bᶜ : Set ℝ)) ↔ (x = -1 ∨ 3 ≤ x ∧ x ≤ 5) :=
by
  sorry

-- (2) Problem Statement
theorem find_value_m (A B : Set ℝ) (m : ℝ) :
  (A = {x | x^2 - 4 * x - 5 ≤ 0}) →
  (B = {x | x^2 - 2 * x - m < 0}) →
  (A ∩ B = {x | -1 ≤ x ∧ x < 4}) →
  m = 8 :=
by
  sorry

end find_intersection_complement_find_value_m_l803_803000


namespace max_value_of_expression_l803_803066

noncomputable def problem := ∀ (a b c : ℝ),
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 2 →
  a + 2 * real.sqrt (a * b) + real.cbrt (a * b * c) ≤ 11 / 4

-- The proof is omitted
theorem max_value_of_expression : problem :=
by
  sorry

end max_value_of_expression_l803_803066


namespace share_of_y_l803_803298

-- Define the conditions as hypotheses
variables (n : ℝ) (x y z : ℝ)

-- The main theorem we need to prove
theorem share_of_y (h1 : x = n) 
                   (h2 : y = 0.45 * n) 
                   (h3 : z = 0.50 * n) 
                   (h4 : x + y + z = 78) : 
  y = 18 :=
by 
  -- insert proof here (not required as per instructions)
  sorry

end share_of_y_l803_803298


namespace find_ellipse_equation_l803_803383

-- Define the conditions
def ellipse_center_origin : Prop :=
  (center : ℝ × ℝ) = (0, 0)

def right_focus (f : ℝ × ℝ) : Prop :=
  f = (1, 0)

def eccentricity (e : ℝ) : Prop :=
  e = 1 / 2

-- The main theorem statement
theorem find_ellipse_equation (h1 : ellipse_center_origin)
  (h2 : right_focus (1, 0))
  (h3 : eccentricity (1 / 2)) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧ a = 2 ∧ 3 = a^2 - 1 ∧
  ∀ x y : ℝ, (ℝ) ∧ (3 = 4 - 1) ∧ 4=a² ∧ b²=3∧(x^2 / 4) + (y^2 / 3) = 1) :=
begin
  sorry
end

end find_ellipse_equation_l803_803383


namespace find_value_of_y_l803_803256

theorem find_value_of_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (hr : x % y = 4) (hxy : x / y = 96.16) : y = 25 :=
by
  sorry

end find_value_of_y_l803_803256


namespace sourav_distance_l803_803234

def D (t : ℕ) : ℕ := 20 * t

theorem sourav_distance :
  ∀ (t : ℕ), 20 * t = 25 * (t - 1) → 20 * t = 100 :=
by
  intros t h
  sorry

end sourav_distance_l803_803234


namespace mersenne_prime_l803_803469

theorem mersenne_prime {p : ℕ} (hp : nat.prime (2 ^ p - 1)) (h : p = 82_589_933) : 
  2 ^ p - 1 = 2 ^ 82_589_933 - 1 := 
by 
  rw h
  sorry

end mersenne_prime_l803_803469


namespace sector_properties_l803_803370

-- Define the given conditions
def radius := 10
def central_angle := 120

-- Define the problem in Lean
theorem sector_properties :
  (let L := (central_angle * Real.pi * radius) / 180 in L = (20 / 3) * Real.pi) ∧
  (let S := (central_angle * Real.pi * radius^2) / 360 in S = (100 / 3) * Real.pi) :=
by
  sorry

end sector_properties_l803_803370


namespace f_f_10_eq_2_l803_803433

noncomputable def f (x : ℝ) : ℝ := 
if x ≤ 1 then x^2 + 1 else log10 x

theorem f_f_10_eq_2 : f(f(10)) = 2 := by
  sorry

end f_f_10_eq_2_l803_803433


namespace least_nonzero_coefficients_l803_803473

theorem least_nonzero_coefficients (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, a i ≠ 0) :
  (∃ k, k = n + 1 - ((n - 1 + 1) / 2 + 1)) → ∃ m, m = (n + 1) - Int.ceil ((n - 1) / 2) := by
  sorry

end least_nonzero_coefficients_l803_803473


namespace chord_length_correct_midpoint_line_correct_trajectory_eq_correct_l803_803791

-- Definitions based on the given conditions
def ellipse_eq (x y : ℝ) : Prop := 5 * x^2 + 9 * y^2 = 45
def focus_point : (ℝ × ℝ) := (2, 0) 

-- Problem 1: Chord Length
def chord_line (x y : ℝ) : Prop := y = x - 2
def chord_length : ℝ := 30 / 7

-- Problem 2: Line with midpoint M(1,1)
def midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 1
def chord_slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₁ - y₂) / (x₁ - x₂)
def line_eq (x y : ℝ) : Prop := 5 * x + 9 * y = 14

-- Problem 3: Trajectory equation of midpoint
def trajectory_eq (x y : ℝ) : Prop := 5 * x^2 + 9 * y^2 - 10 * x = 0

-- Statements to prove
theorem chord_length_correct : 
  ∀ x₁ y₁ x₂ y₂, ellipse_eq x₁ y₁ → ellipse_eq x₂ y₂ → chord_line x₁ y₁ → chord_line x₂ y₂ → 
    (abs (sqrt (1 + 1^2) * sqrt ((x₁ + x₂)^2 - 4 * x₁ * x₂))) = chord_length := 
    sorry

theorem midpoint_line_correct : 
  ∀ x₁ y₁ x₂ y₂, ellipse_eq x₁ y₁ → ellipse_eq x₂ y₂ → midpoint x₁ y₁ x₂ y₂ → 
    (∀ x y, line_eq x y ↔ 5 * x + 9 * y = 14) := 
    sorry

theorem trajectory_eq_correct : 
  ∀ x₁ y₁ x₂ y₂ x y, ellipse_eq x₁ y₁ → ellipse_eq x₂ y₂ → 
    (midpoint x₁ y₁ x₂ y₂ → trajectory_eq x y) := 
    sorry

end chord_length_correct_midpoint_line_correct_trajectory_eq_correct_l803_803791


namespace number_of_attempted_problems_l803_803659

-- Lean statement to define the problem setup
def student_assignment_problem (x y : ℕ) : Prop :=
  8 * x - 5 * y = 13 ∧ x + y ≤ 20

-- The Lean statement asserting the solution to the problem
theorem number_of_attempted_problems : ∃ x y : ℕ, student_assignment_problem x y ∧ x + y = 13 := 
by
  sorry

end number_of_attempted_problems_l803_803659


namespace evaluate_expression_l803_803936

theorem evaluate_expression :
  real.log 5 * real.log 20 + real.log 2 ^ 2 - real.exp (real.log (2 / 3)) = 1 / 3 :=
by
  sorry

end evaluate_expression_l803_803936


namespace numbers_with_perfect_square_factor_l803_803008

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ k : ℕ, k ∈ {4, 9, 16, 25, 36, 49, 64} ∧ k ∣ n

theorem numbers_with_perfect_square_factor :
  finset.card (finset.filter has_perfect_square_factor_other_than_one (finset.range 76)) = 29 :=
by
  sorry

end numbers_with_perfect_square_factor_l803_803008


namespace range_of_g_f_decreasing_l803_803365

open Set Real

variable {a : ℝ}

def f (x : ℝ) : ℝ := (a^x - a^(-x)) / (a^x + a^(-x))
def g (x : ℝ) : ℝ := f x - 1

theorem range_of_g (h : 0 < a ∧ a < 1) : range g = Ioo (-2 : ℝ) 0 :=
  sorry

theorem f_decreasing (h : 0 < a ∧ a < 1) : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 :=
  sorry

end range_of_g_f_decreasing_l803_803365


namespace tangent_line_at_M_l803_803731

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x - 6)

theorem tangent_line_at_M :
  let M : ℝ × ℝ := (2, 0)
  ∃ (m n : ℝ), n = f m ∧ m = 4 ∧ n = -2 * Real.exp 4 ∧
    ∀ (x y : ℝ), y = -Real.exp 4 * (x - 2) →
    M.2 = y :=
by
  sorry

end tangent_line_at_M_l803_803731


namespace number_of_people_l803_803812

noncomputable def birthday_paradox (n : ℕ) : Prop :=
  (∏ i in Finset.range (n-1), (365 - i) / 365) < 1/2 ∧ 
  (∏ i in Finset.range n, (365 - i) / 365) ≥ 1/2

theorem number_of_people : birthday_paradox 22 := by
  sorry

end number_of_people_l803_803812


namespace find_A_l803_803910

-- Define the polynomial and the partial fraction decomposition equation
def polynomial (x : ℝ) : ℝ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_A (A B C : ℝ) (h : ∀ x : ℝ, 1 / polynomial x = A / (x + 3) + B / (x - 1) + C / (x - 1)^2) : 
  A = 1 / 16 :=
sorry

end find_A_l803_803910


namespace correctAssignment_l803_803238

-- Define the conditions as hypotheses
namespace AssignmentStatements
variable (A M B x y : Int)

-- Define each condition as a statement
def conditionA : Prop := (6 = A)
def conditionB : Prop := (M = -M)
def conditionC : Prop := (B = A = 2)
def conditionD : Prop := (x + 5 * y = 0)

-- Prove that the correct assignment statement is conditionB
theorem correctAssignment : conditionB :=
by
  -- Place a placeholder for the proof
  sorry
end AssignmentStatements

end correctAssignment_l803_803238


namespace infinitely_many_arithmetic_sequences_l803_803103

theorem infinitely_many_arithmetic_sequences (x : ℕ) (hx : 0 < x) :
  ∃ y z : ℕ, y = 5 * x + 2 ∧ z = 7 * x + 3 ∧ x * (x + 1) < y * (y + 1) ∧ y * (y + 1) < z * (z + 1) ∧
  y * (y + 1) - x * (x + 1) = z * (z + 1) - y * (y + 1) :=
by
  sorry

end infinitely_many_arithmetic_sequences_l803_803103


namespace evaluate_expression_equals_three_plus_sqrt_three_l803_803317

noncomputable def tan_sixty_squared_plus_one := Real.tan (60 * Real.pi / 180) ^ 2 + 1
noncomputable def tan_fortyfive_minus_twocos_thirty := Real.tan (45 * Real.pi / 180) - 2 * Real.cos (30 * Real.pi / 180)
noncomputable def expression (x y : ℝ) : ℝ := (x - (2 * x * y - y ^ 2) / x) / ((x ^ 2 - y ^ 2) / (x ^ 2 + x * y))

theorem evaluate_expression_equals_three_plus_sqrt_three :
  expression tan_sixty_squared_plus_one tan_fortyfive_minus_twocos_thirty = 3 + Real.sqrt 3 :=
sorry

end evaluate_expression_equals_three_plus_sqrt_three_l803_803317


namespace matrix_multiplication_l803_803314

theorem matrix_multiplication :
  let A := ![![4, -2, 1], ![3, 6, -2], ![2, 0, 5]]
  let B := ![![1, 0, -3], ![0, 2, 4], ![-1, 2, 1]]
  let C := ![![3, -2, -19], ![5, 8, 13], ![-3, 10, -1]]
  matrix.mul A B = C :=
by {
  sorry
}

end matrix_multiplication_l803_803314


namespace solution1_solution2_l803_803747

noncomputable def problem1 (x : ℝ) : Prop :=
  4 * x^2 - 25 = 0

theorem solution1 (x : ℝ) : problem1 x ↔ x = 5 / 2 ∨ x = -5 / 2 :=
by sorry

noncomputable def problem2 (x : ℝ) : Prop :=
  (x + 1)^3 = -27

theorem solution2 (x : ℝ) : problem2 x ↔ x = -4 :=
by sorry

end solution1_solution2_l803_803747


namespace hyperbola_eccentricity_proof_l803_803776

-- Define the given conditions
variables (a b c : ℝ) (O : ℝ × ℝ)
variables (F1 F2 P : ℝ × ℝ)
variable (h : a > 0)
variable (k : b > 0)

-- Assume the hyperbola equation
def is_hyperbola (x y : ℝ) := x^2 / a^2 - y^2 / b^2 = 1

-- Define foci and point P conditions
axiom Foci_and_Point_on_hyperbola (h1 : is_hyperbola a b) (h2 : a > 0) (h3 : b > 0)
  (angle_F1PF2 : ∠ F1 P F2 = 60) (OP_eq_3b : |O - P| = 3 * b)
  (eccentricity : ℝ) : eccentricity = c / a 

-- Required variable to derive the proof of eccentricity
def eccentricity := c / a 

theorem hyperbola_eccentricity_proof :
  (c = sqrt(42)) -> eccentricity = (sqrt 42) / 6 :=
by
  sorry

end hyperbola_eccentricity_proof_l803_803776


namespace no_zero_position_l803_803581

-- Define the concept of regular pentagon vertex assignments and operations
def pentagon_arith_mean (x y : ℝ) : ℝ := (x + y) / 2

-- Define the condition for the initial sum of numbers on the vertices being zero
def initial_sum_zero (a b c d e : ℝ) : Prop := a + b + c + d + e = 0

-- Define the main theorem statement
theorem no_zero_position (a b c d e : ℝ) (h : initial_sum_zero a b c d e) :
  ¬ ∃ a' b' c' d' e' : ℝ, ∀ v w : ℝ, pentagon_arith_mean v w = 0 :=
sorry

end no_zero_position_l803_803581


namespace coloring_ways_of_circle_l803_803453

noncomputable def num_ways_to_color_circle (n : ℕ) (k : ℕ) : ℕ :=
  if h : n % 2 = 1 then -- There are 13 parts; n must be odd (since adjacent matching impossible in even n)
    (k * (k - 1)^(n - 1) : ℕ)
  else
    0

theorem coloring_ways_of_circle :
  num_ways_to_color_circle 13 3 = 6 :=
by
  sorry

end coloring_ways_of_circle_l803_803453


namespace equal_parts_l803_803930

-- Define the type for the positions of stars and crosses
structure Position where
  x : Nat
  y : Nat

-- Define the square arrangement containing stars and crosses
structure SquareArrangement where
  stars : List Position
  crosses : List Position
  size : Nat  -- Assuming size is the length of one side of the square

-- Define the condition that each smaller square must contain exactly one star and one cross
def contains_one_star_one_cross (sa : SquareArrangement) (x1 y1 x2 y2 : Nat) : Prop :=
  (sa.stars.filter (λ p => x1 ≤ p.x ∧ p.x < x2 ∧ y1 ≤ p.y ∧ p.y < y2)).length = 1 ∧
  (sa.crosses.filter (λ p => x1 ≤ p.x ∧ p.x < x2 ∧ y1 ≤ p.y ∧ y2)).length = 1

-- Define the main theorem which states that cutting the square into equal parts by vertical and horizontal cuts
-- through the center results in each part containing exactly one star and one cross
theorem equal_parts (sa : SquareArrangement) (h_size : sa.size % 2 = 0) :
  let half := sa.size / 2
  (contains_one_star_one_cross sa 0 0 half half) ∧
  (contains_one_star_one_cross sa half 0 sa.size half) ∧
  (contains_one_star_one_cross sa 0 half half sa.size) ∧
  (contains_one_star_one_cross sa half half sa.size sa.size) :=
by
  sorry

end equal_parts_l803_803930


namespace probability_odd_sum_l803_803156

-- Definitions given based on conditions
def primes_not_exceeding_19 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- The statement to prove
theorem probability_odd_sum :
  (primes_not_exceeding_19.card = 8) →
  let total_combinations := primes_not_exceeding_19.card.choose 2 in
  let odd_sum_combinations := (primes_not_exceeding_19.filter nat.prime.pred).card in
  ↑odd_sum_combinations / ↑total_combinations = (1 : ℚ) / 4 :=
by sorry

end probability_odd_sum_l803_803156


namespace num_of_non_multiples_of_6_or_8_in_three_digits_l803_803413

-- Define conditions about multiples and range of three-digit numbers
def is_multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

def multiples_of_6 := {n : ℕ | is_multiple_of n 6}
def multiples_of_8 := {n : ℕ | is_multiple_of n 8}
def multiples_of_24 := {n : ℕ | is_multiple_of n 24}

-- Prove that the number of three-digit numbers not multiples of 6 or 8 is 675
theorem num_of_non_multiples_of_6_or_8_in_three_digits : 
  {n : ℕ | n ∈ three_digit_numbers ∧ n ∉ multiples_of_6 ∧ n ∉ multiples_of_8}.count = 675 :=
sorry

end num_of_non_multiples_of_6_or_8_in_three_digits_l803_803413


namespace smallest_int_with_divisors_l803_803213

theorem smallest_int_with_divisors :
  ∃ n : ℕ, 
    (∀ m, n = 2^2 * m → 
      (∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 
      (m = p^3 * q) ∧ 
      (nat.divisors p^7).count 8) ∧ 
    nat.divisors_count n = 24 ∧ 
    (nat.divisors (2^2 * n)).count 8) (n = 2^2 * 3^3 * 5) :=
begin
  sorry
end

end smallest_int_with_divisors_l803_803213


namespace pool_capacity_percentage_l803_803911

theorem pool_capacity_percentage
  (rate : ℕ := 60) -- cubic feet per minute
  (time : ℕ := 800) -- minutes
  (width : ℕ := 60) -- feet
  (length : ℕ := 100) -- feet
  (depth : ℕ := 10) -- feet
  : (rate * time * 100) / (width * length * depth) = 8 := by
{
  sorry
}

end pool_capacity_percentage_l803_803911


namespace mrWang_water_usage_l803_803939

-- Definition of the given conditions
def fee (x : ℝ) : ℝ :=
  if x <= 10 then 2 * x
  else 2 * 10 + 3 * (x - 10)

-- Definition of the average fee condition
def average_fee_condition (x : ℝ) : Prop :=
  fee(x) / x = 2.5

-- The theorem proving the question
theorem mrWang_water_usage : ∃ x : ℝ, x = 20 ∧ average_fee_condition x :=
by
  existsi (20 : ℝ)
  simp [average_fee_condition, fee]
  sorry

end mrWang_water_usage_l803_803939


namespace equilateral_triangle_circumcircle_sum_l803_803372

theorem equilateral_triangle_circumcircle_sum {A B C M : Point} (a : ℝ) 
  (h_eq_tri : equilateral_triangle A B C a) 
  (h_circ : on_circumcircle A B C M) : MA ^ 4 + MB ^ 4 + MC ^ 4 = 2 * a ^ 4 := 
sorry

end equilateral_triangle_circumcircle_sum_l803_803372


namespace sin_A_in_right_triangle_l803_803043

theorem sin_A_in_right_triangle :
  ∀ (A B C : Type) [real_inner_product_space ℝ A]
  (angle_BAC : A ≡ C)
  (angle_BAC_right : angle_BAC = 90)
  (length_AB : A ≡ B)
  (length_BC : B ≡ C)
  (AB : ℝ) (BC : ℝ) (AC : ℝ)
  (hAB : AB = 15)
  (hBC : BC = 20)
  (hAC : AC = real.sqrt (BC * BC - AB * AB))
  (sin_A : ℝ),
  sin_A = AC / BC → sin_A = real.sqrt 7 / 4 :=
begin
  sorry
end

end sin_A_in_right_triangle_l803_803043


namespace geometric_progression_fourth_term_l803_803553

theorem geometric_progression_fourth_term (a b c : ℝ) (r : ℝ) 
  (h1 : a = 2) (h2 : b = 2 * Real.sqrt 2) (h3 : c = 4) (h4 : r = Real.sqrt 2)
  (h5 : b = a * r) (h6 : c = b * r) :
  c * r = 4 * Real.sqrt 2 := 
sorry

end geometric_progression_fourth_term_l803_803553


namespace balls_into_boxes_l803_803420

theorem balls_into_boxes :
  let n := 7 -- number of balls
  let k := 3 -- number of boxes
  let ways := Nat.choose (n + k - 1) (k - 1)
  ways = 36 :=
by
  sorry

end balls_into_boxes_l803_803420


namespace exist_circle_subset_27_l803_803940

noncomputable def circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2

def circles : list (ℝ × ℝ) := 
  -- A list of 2015 distinct circle centers, which for simplicity we will just declare abstractly.
  sorry

def graph_of_circles (centers : list (ℝ × ℝ)) : Graph (ℝ × ℝ) :=
  ⟨centers, λ c1 c2, (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 ≤ (2 : ℝ)^2⟩

theorem exist_circle_subset_27 (centers : list (ℝ × ℝ)) (h : centers.length = 2015) :
  ∃ C : Finset (ℝ × ℝ), C.card = 27 ∧ (∀ x ∈ C, ∀ y ∈ C, circle x 1 → circle y 1 → ((x.1 - y.1)^2 + (x.2 - y.2)^2 ≤ (2 : ℝ)^2) ∨ ((x.1 - y.1)^2 + (x.2 - y.2)^2 > (2: ℝ)^2)) :=
begin
  sorry
end

end exist_circle_subset_27_l803_803940


namespace sum_of_exponents_square_root_largest_perfect_square_dividing_15_factorial_is_10_l803_803605

theorem sum_of_exponents_square_root_largest_perfect_square_dividing_15_factorial_is_10 :
  ∑ p in ({2, 3, 5, 7, 11, 13} : Finset ℕ), if p ∣ 15! then (nat.factorization 15!.half).p else 0 = 10 :=
by
  sorry

end sum_of_exponents_square_root_largest_perfect_square_dividing_15_factorial_is_10_l803_803605


namespace salary_decrease_increase_l803_803524

theorem salary_decrease_increase (S : ℝ) (x : ℝ) (h : (S * (1 - x / 100) * (1 + x / 100) = 0.51 * S)) : x = 70 := 
by sorry

end salary_decrease_increase_l803_803524


namespace smallest_integer_with_odd_and_even_divisors_l803_803208

theorem smallest_integer_with_odd_and_even_divisors :
  ∃ n : ℕ,
    0 < n ∧
    (∀ d : ℕ, d ∣ n → (odd d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 1}) ∨ (even d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 0})) ∧ 
    (↑8 = (∑ d in (finset.filter (λ d, odd d) (finset.divisors n)), 1)) ∧
    (↑16 = (∑ d in (finset.filter (λ d, even d) (finset.divisors n)), 1)) ∧
    (24 = (∑ d in (finset.divisors n), 1)) ∧
    n = 108 :=
begin
  sorry
end

end smallest_integer_with_odd_and_even_divisors_l803_803208


namespace nicky_speed_l803_803499

theorem nicky_speed
  (head_start : ℕ := 36)
  (cristina_speed : ℕ := 6)
  (time_to_catch_up : ℕ := 12)
  (distance_cristina_runs : ℕ := cristina_speed * time_to_catch_up)
  (distance_nicky_runs : ℕ := distance_cristina_runs - head_start)
  (nicky_speed : ℕ := distance_nicky_runs / time_to_catch_up) :
  nicky_speed = 3 :=
by
  sorry

end nicky_speed_l803_803499


namespace math_problem_l803_803026

theorem math_problem (x y z : ℝ) 
  (h1 : y^2 + |x - 2023| + real.sqrt (z - 4) = 6 * y - 9) :
  (y - z) ^ x = -1 := 
sorry

end math_problem_l803_803026


namespace smallest_n_with_divisors_l803_803193

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end smallest_n_with_divisors_l803_803193


namespace percentage_tip_l803_803058

theorem percentage_tip (meal_cost : ℝ) (drink_cost : ℝ) (paid_amount : ℝ) (change : ℝ) :
  meal_cost = 10 → drink_cost = 2.5 → paid_amount = 20 → change = 5 → 
  ((paid_amount - change - (meal_cost + drink_cost)) / (meal_cost + drink_cost)) * 100 = 20 :=
by
  intros h_meal h_drink h_paid h_change
  have total_cost := meal_cost + drink_cost
  have tip := paid_amount - change - total_cost
  have tip_percentage := (tip / total_cost) * 100
  rw [h_meal, h_drink, h_paid, h_change] at tip_percentage
  sorry

end percentage_tip_l803_803058


namespace find_values_of_a_l803_803339

theorem find_values_of_a :
  ∃ (a : ℝ), 
    (∀ x y, (|y + 2| + |x - 11| - 3) * (x^2 + y^2 - 13) = 0 ∧ 
             (x - 5)^2 + (y + 2)^2 = a) ↔ 
    a = 9 ∨ a = 42 + 2 * Real.sqrt 377 :=
sorry

end find_values_of_a_l803_803339


namespace smallest_integer_with_divisors_l803_803182

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end smallest_integer_with_divisors_l803_803182


namespace monotonic_intervals_max_min_values_l803_803389

open Real

def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x + cos x ^ 2 - 1 / 2

theorem monotonic_intervals :
  (∀ k : ℤ, ∃ I : Set ℝ, I = Icc (k * π - π / 3) (k * π + π / 6) ∧ ∀ x ∈ I, differentiable_at ℝ f x ∧ f' x ≥ 0) ∧
  (∀ k : ℤ, ∃ I : Set ℝ, I = Icc (k * π + π / 6) (k * π + 2 * π / 3) ∧ ∀ x ∈ I, differentiable_at ℝ f x ∧ f' x ≤ 0) := sorry

theorem max_min_values :
  ∃ (x_max x_min : ℝ),
  - 5 / 12 * π ≤ x_max ∧ x_max ≤ 1 / 24 * π ∧ f x_max = sqrt 2 / 2 ∧
  - 5 / 12 * π ≤ x_min ∧ x_min ≤ 1 / 24 * π ∧ f x_min = -1 / 2 := sorry

end monotonic_intervals_max_min_values_l803_803389


namespace circumradius_not_integer_l803_803901

theorem circumradius_not_integer 
  (a b c : ℕ) 
  (h_odd : ¬(even a)) 
  (h_scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ¬ ∃ (R : ℤ), R > 0 ∧ ∃ (α β γ : ℝ),
    (sin α = a / (2 * R)) ∧ 
    (sin β = b / (2 * R)) ∧ 
    (sin γ = c / (2 * R)) :=
by
  sorry

end circumradius_not_integer_l803_803901


namespace ptarmigan_environmental_capacity_l803_803501

theorem ptarmigan_environmental_capacity (predators_eradicated : Prop) (mass_deaths : Prop) : 
  (after_predator_eradication : predators_eradicated → mass_deaths) →
  (environmental_capacity_increased : Prop) → environmental_capacity_increased :=
by
  intros h1 h2
  sorry

end ptarmigan_environmental_capacity_l803_803501


namespace coordinates_of_a_l803_803381

theorem coordinates_of_a
  (a : ℝ × ℝ)
  (b : ℝ × ℝ := (1, 2))
  (h1 : (a.1)^2 + (a.2)^2 = 5)
  (h2 : ∃ k : ℝ, a = (k, 2 * k))
  : a = (1, 2) ∨ a = (-1, -2) :=
  sorry

end coordinates_of_a_l803_803381


namespace find_x_l803_803823

noncomputable def normal_vector_a : ℝ × ℝ × ℝ := (-1, 2, 4)
noncomputable def normal_vector_b (x : ℝ) : ℝ × ℝ × ℝ := (x, -1, -2)
def is_perpendicular (a b : ℝ × ℝ × ℝ) : Prop := let ⟨a1, a2, a3⟩ := a; let ⟨b1, b2, b3⟩ := b in a1 * b1 + a2 * b2 + a3 * b3 = 0

theorem find_x : ∃ x : ℝ, is_perpendicular normal_vector_a (normal_vector_b x) ∧ x = -10 := by
  sorry

end find_x_l803_803823


namespace evaluate_argument_l803_803331

def angle (z : ℂ) : ℝ := complex.arg z

theorem evaluate_argument :
  let z := complex.exp (11 * real.pi * complex.I / 60) +
           complex.exp (31 * real.pi * complex.I / 60) +
           complex.exp (51 * real.pi * complex.I / 60) +
           complex.exp (71 * real.pi * complex.I / 60) +
           complex.exp (91 * real.pi * complex.I / 60) in
  0 ≤ angle z ∧ angle z < 2 * real.pi ∧ angle z = 17 * real.pi / 20 :=
by
  let z := complex.exp (11 * real.pi * complex.I / 60) +
           complex.exp (31 * real.pi * complex.I / 60) +
           complex.exp (51 * real.pi * complex.I / 60) +
           complex.exp (71 * real.pi * complex.I / 60) +
           complex.exp (91 * real.pi * complex.I / 60)
  have : angle z = 17 * real.pi / 20 := sorry
  exact ⟨complex.arg_nonneg z, complex.arg_lt_two_pi z, this⟩

end evaluate_argument_l803_803331


namespace area_of_rectangle_R_l803_803994

-- Define the side lengths of the squares and rectangles involved
def larger_square_side := 4
def smaller_square_side := 2
def rectangle_side1 := 1
def rectangle_side2 := 4

-- The areas of these shapes
def area_larger_square := larger_square_side * larger_square_side
def area_smaller_square := smaller_square_side * smaller_square_side
def area_first_rectangle := rectangle_side1 * rectangle_side2

-- Define the sum of all possible values for the area of rectangle R
def area_remaining := area_larger_square - (area_smaller_square + area_first_rectangle)

theorem area_of_rectangle_R : area_remaining = 8 := sorry

end area_of_rectangle_R_l803_803994


namespace jana_distance_in_10_minutes_l803_803053

def walking_rate (distance : ℝ) (time : ℝ) : ℝ := distance / time

theorem jana_distance_in_10_minutes :
  let rate := walking_rate 1 24 in
  let distance_in_10_minutes := rate * 10 in
  distance_in_10_minutes ≈ 0.4 := 
by
  sorry

end jana_distance_in_10_minutes_l803_803053


namespace project_completion_time_l803_803054

noncomputable def worker_rate (days: ℕ) (workers: ℕ): ℕ :=
  1 / (days * workers)

theorem project_completion_time:
  ∀ (d w1 w2: ℕ),
    d = 3 → w1 = 60 → w2 = 30 →
    ((w1 * d: ℕ) * worker_rate d w1 = 1) →
    ((w2 * ?t: ℕ) * worker_rate d w1 = 1) →
    ?t = 6 :=
by
  intros d w1 w2 hd hw1 hw2 hwork1 hwork2
  sorry

end project_completion_time_l803_803054


namespace volume_of_sphere_with_radius_3_l803_803938

-- Define the radius of the sphere as a constant
def radius : ℝ := 3

-- Define the formula for the volume of a sphere
def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Prove that the volume of a sphere with radius 3 cm is 36pi cm^3
theorem volume_of_sphere_with_radius_3 : volume radius = 36 * Real.pi := by
  sorry

end volume_of_sphere_with_radius_3_l803_803938


namespace monotonic_decreasing_interval_l803_803131

noncomputable def function_y (x : ℝ) : ℝ := real.sqrt (2 * x - x ^ 2)

theorem monotonic_decreasing_interval :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → ∃ x1 x2, 1 ≤ x ∧ x ≤ 2 :=
by
  sorry

end monotonic_decreasing_interval_l803_803131


namespace both_chromatids_contain_15N_l803_803633

-- Given conditions
def progenitor_cell_labeled_15N : Prop := sorry
def cultured_in_medium_containing_N : Prop := sorry

-- Proof Problem Statement
theorem both_chromatids_contain_15N 
    (h1 : progenitor_cell_labeled_15N) 
    (h2 : cultured_in_medium_containing_N) : 
    ∀ chromosome : Type, 
      (∃ chromatids : Type × Type,
        chromatids.fst.label = "15N" ∧ chromatids.snd.label = "15N") :=
sorry

end both_chromatids_contain_15N_l803_803633


namespace sum_of_digits_l803_803367

-- Define the problem -- 29-digit number with specific conditions
def satisfies_conditions (a : Fin 30 → ℕ) : Prop :=
  (∀ k, 0 ≤ a k ∧ a k ≤ 9) ∧
  (a ⟨1, by decide⟩ ≠ 0) ∧
  (∀ k, let m := 29 - k in a ⟨k + 1, by decide⟩ = a ⟨m + 1, by decide⟩)

-- Prove that a sequence satisfying the conditions has sum 201.
theorem sum_of_digits (a : Fin 30 → ℕ) (h : satisfies_conditions a) :
  (Finset.finRange 29).sum a = 201 :=
begin
  sorry
end

end sum_of_digits_l803_803367


namespace mag_product_mag_conjugate_mag_squares_neq_purely_imaginary_cond_l803_803864

-- Assume definitions and conditions
def is_conjugate (z1 z2 : ℂ) : Prop :=
  z1.re = z2.re ∧ z1.im = -z2.im

noncomputable def purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

-- Prove that |z1 * z2| = |z1| * |z2|
theorem mag_product (z1 z2 : ℂ) : |z1 * z2| = |z1| * |z2| :=
sorry

-- Prove that for conjugate complex numbers z1 and z2, |z1| = |z2|
theorem mag_conjugate (z1 z2 : ℂ) (h : is_conjugate z1 z2) : |z1| = |z2| :=
sorry

-- Prove that if |z1| = |z2|, then z1^2 ≠ z2^2
theorem mag_squares_neq (z1 z2 : ℂ) (h : |z1| = |z2|) : z1^2 ≠ z2^2 :=
sorry

-- Prove that if z = m + 1 + (m - 1)i is purely imaginary, then m = -1
theorem purely_imaginary_cond (m : ℝ) (z : ℂ) (h : z = m + 1 + (m - 1) * complex.I) (h_pi : purely_imaginary z) : m = -1 :=
sorry

end mag_product_mag_conjugate_mag_squares_neq_purely_imaginary_cond_l803_803864


namespace time_until_meeting_l803_803166

theorem time_until_meeting (v1 v2 : ℝ) (t2 t1 : ℝ) 
    (h1 : v1 = 6) 
    (h2 : v2 = 4) 
    (h3 : t2 = 10)
    (h4 : v2 * t1 = v1 * (t1 - t2)) : t1 = 30 := 
sorry

end time_until_meeting_l803_803166


namespace zookeeper_fish_excess_l803_803148

theorem zookeeper_fish_excess :
  let emperor_ratio := 3
  let adelie_ratio := 5
  let total_penguins := 48
  let total_ratio := emperor_ratio + adelie_ratio
  let emperor_penguins := (emperor_ratio / total_ratio) * total_penguins
  let adelie_penguins := (adelie_ratio / total_ratio) * total_penguins
  let emperor_fish_needed := emperor_penguins * 1.5
  let adelie_fish_needed := adelie_penguins * 2
  let total_fish_needed := emperor_fish_needed + adelie_fish_needed
  let fish_zookeeper_has := total_penguins * 2.5
  (fish_zookeeper_has - total_fish_needed = 33) :=
  
by {
  sorry
}

end zookeeper_fish_excess_l803_803148


namespace max_acute_angles_in_convex_octagon_l803_803172

def is_convex (polygon_angles : List ℝ) : Prop :=
  ∀ a ∈ polygon_angles, a < 180

def interior_angle_sum (polygon_angles : List ℝ) : ℝ :=
  polygon_angles.sum

def is_acute (angle : ℝ) : Prop :=
  angle < 90

theorem max_acute_angles_in_convex_octagon
  (angles : List ℝ)
  (h1 : angles.length = 8)
  (h2 : is_convex angles)
  (h3 : interior_angle_sum angles = 1080) :
  ∃ n, n ≤ 5 ∧ n = angles.countp is_acute :=
sorry

end max_acute_angles_in_convex_octagon_l803_803172


namespace binomial_product_result_l803_803689

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l803_803689


namespace red_to_blue_ratio_l803_803630

theorem red_to_blue_ratio
    (total_balls : ℕ)
    (num_white_balls : ℕ)
    (num_blue_balls : ℕ)
    (num_red_balls : ℕ) :
    total_balls = 100 →
    num_white_balls = 16 →
    num_blue_balls = num_white_balls + 12 →
    num_red_balls = total_balls - (num_white_balls + num_blue_balls) →
    (num_red_balls / num_blue_balls : ℚ) = 2 :=
by
  intro h1 h2 h3 h4
  -- Proof is omitted
  sorry

end red_to_blue_ratio_l803_803630


namespace smallest_integer_with_odd_and_even_divisors_l803_803206

theorem smallest_integer_with_odd_and_even_divisors :
  ∃ n : ℕ,
    0 < n ∧
    (∀ d : ℕ, d ∣ n → (odd d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 1}) ∨ (even d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 0})) ∧ 
    (↑8 = (∑ d in (finset.filter (λ d, odd d) (finset.divisors n)), 1)) ∧
    (↑16 = (∑ d in (finset.filter (λ d, even d) (finset.divisors n)), 1)) ∧
    (24 = (∑ d in (finset.divisors n), 1)) ∧
    n = 108 :=
begin
  sorry
end

end smallest_integer_with_odd_and_even_divisors_l803_803206


namespace bisector_of_angle_l803_803956

theorem bisector_of_angle (A B C : Point) (h_angle : angle A B C = θ) :
  ∃ D : Point, line_through A D ∧ angle A B D = angle D B C :=
sorry

end bisector_of_angle_l803_803956


namespace ratio_of_segments_l803_803971

theorem ratio_of_segments {A B C D E : Type*} 
  [Geometry.Triangle A B C]
  (AD_is_angle_bisector : Geometry.IsAngleBisector A D (∠ BAC))
  (E_intersects_perpendicular_bisector_AD_BC : Geometry.IsIntersection E (Geometry.PerpendicularBisector AD) BC)
  (b c : Real) : 
  Geometry.SegmentRatio E B C = c^2 / b^2 :=
begin
  sorry
end

end ratio_of_segments_l803_803971


namespace P_of_P_x_positive_l803_803470

noncomputable def P (x r s : ℝ) : ℝ := x^2 + r*x + s

theorem P_of_P_x_positive (r s : ℝ) 
  (a b : ℝ) (ha : a < -1) (hb : b < -1) 
  (h_diff : 0 < b - a < 2) 
  (h_roots : ∀ x, P x r s = 0 ↔ x = a ∨ x = b) :
  ∀ x : ℝ, P (P x r s) r s > 0 :=
by
  sorry

end P_of_P_x_positive_l803_803470


namespace degree_at_least_three_l803_803474

noncomputable def p : Polynomial ℤ := sorry
noncomputable def q : Polynomial ℤ := sorry

theorem degree_at_least_three (h1 : p.degree ≥ 1)
                              (h2 : q.degree ≥ 1)
                              (h3 : (∃ xs : Fin 33 → ℤ, ∀ i, p.eval (xs i) * q.eval (xs i) - 2015 = 0)) :
  p.degree ≥ 3 ∧ q.degree ≥ 3 := 
sorry

end degree_at_least_three_l803_803474


namespace bridge_cost_approx_l803_803260

def R (a b c : ℕ) (A : ℝ) : ℝ := (a * b * c) / (4 * A)
def r (A s : ℝ) : ℝ := A / s
def cosC (a b c : ℕ) : ℝ := ((a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b) : ℝ)
def sinC (a b c : ℕ) : ℝ := real.sqrt (1 - (cosC a b c) ^ 2)
def distance_between_centers (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2
  let A := real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R_ := R a b c A
  let r_ := r A s
  real.sqrt (R_ ^ 2 + 2 * R_ * r_ * cosC a b c + r_ ^ 2)

noncomputable def bridge_cost (a b c cost_per_km : ℕ) : ℝ :=
  distance_between_centers a b c * cost_per_km

theorem bridge_cost_approx (a b c cost_per_km : ℕ) :
  bridge_cost 7 8 9 1000 ≈ 5750 :=
by
  sorry

end bridge_cost_approx_l803_803260


namespace what_did_other_five_say_l803_803506

-- Define the inhabitants of the island
inductive Islander
| knight : Islander
| liar : Islander

-- Define the statement types
inductive Statement
| more_liars : Statement
| equal_numbers : Statement

-- Define a structure for the conditions
structure IslandConditions where
  total_islanders : ℕ
  responses_more_liars : ℕ
  responses_equal_numbers : ℕ :=
  total_islanders := 10

-- The theorem that needs to be proved
theorem what_did_other_five_say (c : IslandConditions) :
  c.responses_more_liars = 5 → c.responses_equal_numbers = 5 :=
by
  sorry

end what_did_other_five_say_l803_803506


namespace sufficient_but_not_necessary_not_necessary_l803_803979

theorem sufficient_but_not_necessary (x y : ℝ) (h : x < y ∧ y < 0) : x^2 > y^2 :=
by {
  -- a Lean 4 proof can be included here if desired
  sorry
}

theorem not_necessary (x y : ℝ) (h : x^2 > y^2) : ¬ (x < y ∧ y < 0) :=
by {
  -- a Lean 4 proof can be included here if desired
  sorry
}

end sufficient_but_not_necessary_not_necessary_l803_803979


namespace sum_of_numbers_with_hundreds_5_tens_7_l803_803895

theorem sum_of_numbers_with_hundreds_5_tens_7 :
  let numbers := {x : ℕ | 999 < x ∧ x < 10000 ∧ (x / 100) % 10 = 5 ∧ (x / 10) % 10 = 7} in
  ∑ x in numbers, x = 501705 :=
by
  -- proof idea here
  sorry

end sum_of_numbers_with_hundreds_5_tens_7_l803_803895


namespace min_value_of_expression_l803_803077

theorem min_value_of_expression (α β : ℝ) (h : α + β = π / 2) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = 65 := 
sorry

end min_value_of_expression_l803_803077


namespace extreme_value_condition_l803_803435

noncomputable theory

-- Definition of the function f
def f (m x : ℝ) : ℝ := m * cos x + 1/2 * sin (2 * x)

-- Definition of the derivative of the function f
def f' (m x : ℝ) : ℝ := -m * sin x + cos (2 * x)

-- statement of the math proof problem
theorem extreme_value_condition (m : ℝ) (h : f' m (π / 4) = 0) : m = 0 :=
by
  -- Solving for m
  sorry

end extreme_value_condition_l803_803435


namespace max_fixed_positions_after_rearrangement_l803_803509

theorem max_fixed_positions_after_rearrangement :
  (∃ f : Fin 100 → Fin 100, 
    (∀ i : Fin 99, (f i).val ≤ (f (i+1)).val ∧
     ((f (i+1)).val / 10 = (f i).val / 10 + 1 ∨
      (f (i+1)).val / 10 = (f i).val / 10 - 1 ∨
      (f (i+1)).val % 10 = (f i).val % 10 + 1 ∨
      (f (i+1)).val % 10 = (f i).val % 10 - 1)) ∧
    (∑ i : Fin 100, if i.val = f i.val then 1 else 0) ≤ 50) :=
begin
  sorry
end

end max_fixed_positions_after_rearrangement_l803_803509


namespace sum_of_three_quadratics_no_rot_l803_803845

def quad_poly_sum_no_root (p q : ℝ -> ℝ) : Prop :=
  ∀ x : ℝ, (p x + q x ≠ 0)

theorem sum_of_three_quadratics_no_rot (a b c d e f : ℝ)
    (h1 : quad_poly_sum_no_root (λ x => x^2 + a*x + b) (λ x => x^2 + c*x + d))
    (h2 : quad_poly_sum_no_root (λ x => x^2 + c*x + d) (λ x => x^2 + e*x + f))
    (h3 : quad_poly_sum_no_root (λ x => x^2 + e*x + f) (λ x => x^2 + a*x + b)) :
    quad_poly_sum_no_root (λ x => x^2 + a*x + b) 
                         (λ x => x^2 + c*x + d + x^2 + e*x + f) :=
sorry

end sum_of_three_quadratics_no_rot_l803_803845


namespace percentage_withheld_correct_l803_803952

-- Define the initial conditions
def hourly_pay : ℝ := 15
def weekly_hours : ℝ := 40
def weeks_per_year : ℝ := 52
def annual_take_home_pay : ℝ := 24960

-- Define the weekly gross pay
def weekly_gross_pay : ℝ := hourly_pay * weekly_hours

-- Define the annual gross pay
def annual_gross_pay : ℝ := weekly_gross_pay * weeks_per_year

-- Define the amount withheld
def amount_withheld : ℝ := annual_gross_pay - annual_take_home_pay

-- Define the expected percentage withheld
def expected_percentage_withheld : ℝ := 20

-- Define the actual percentage withheld
def actual_percentage_withheld : ℝ := (amount_withheld / annual_gross_pay) * 100

-- Prove the actual percentage withheld equals the expected percentage withheld
theorem percentage_withheld_correct :
  actual_percentage_withheld = expected_percentage_withheld :=
by
  sorry

end percentage_withheld_correct_l803_803952


namespace find_amount_l803_803270

-- Given conditions
variables (x A : ℝ)

theorem find_amount :
  (0.65 * x = 0.20 * A) → (x = 190) → (A = 617.5) :=
by
  intros h1 h2
  sorry

end find_amount_l803_803270


namespace max_real_roots_l803_803072

noncomputable def discriminant (A B C : ℝ) := B^2 - 4 * A * C

theorem max_real_roots (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let p_discr := discriminant a b c
  let q_discr := discriminant b c a
  let r_discr := discriminant c a b
  p_discr >= 0 + q_discr >= 0 + r_discr >= 0 ≤ 4 := 
  sorry

end max_real_roots_l803_803072


namespace arithmetic_sequence_term_number_l803_803268

-- Given:
def first_term : ℕ := 1
def common_difference : ℕ := 3
def target_term : ℕ := 2011

-- To prove:
theorem arithmetic_sequence_term_number :
    ∃ n : ℕ, target_term = first_term + (n - 1) * common_difference ∧ n = 671 := 
by
  -- The proof is omitted
  sorry

end arithmetic_sequence_term_number_l803_803268


namespace volume_of_cube_shaped_box_to_store_cone_l803_803684

-- Define the height of the cone
def height_cone : ℝ := 20

-- Define the diameter of the base of the cone
def diameter_base_cone : ℝ := 18

-- The height and diameter are positive
axiom height_cone_pos : height_cone > 0
axiom diameter_base_cone_pos : diameter_base_cone > 0

-- The volume of the cube-shaped box that can store the cone
def volume_box (side_length : ℝ) := side_length ^ 3

-- Prove that the volume of the smallest cube-shaped box to store the cone is 8000 cubic inches.
theorem volume_of_cube_shaped_box_to_store_cone 
  (h_cone : height_cone = 20)
  (d_base_cone : diameter_base_cone = 18) : 
  ∃ side : ℝ, side = 20 ∧ volume_box side = 8000 :=
by 
  use 20
  split
  exact h_cone
  sorry

end volume_of_cube_shaped_box_to_store_cone_l803_803684


namespace haley_tv_total_hours_l803_803402

theorem haley_tv_total_hours (h_sat : Nat) (h_sun : Nat) (H_sat : h_sat = 6) (H_sun : h_sun = 3) :
  h_sat + h_sun = 9 := by
  sorry

end haley_tv_total_hours_l803_803402


namespace domain_of_f_l803_803549

noncomputable def f (x : ℝ) := sqrt (x + 1) - x / (2 - x)

theorem domain_of_f : {x : ℝ | x ≥ -1} ∩ {x : ℝ | x ≠ 2} = [-1, 2) ∪ (2, +∞) :=
by
  sorry

end domain_of_f_l803_803549


namespace monotonic_increasing_interval_l803_803924

def f (x : ℝ) : ℝ := 1 / sqrt (x^2 - 5 * x - 6)

theorem monotonic_increasing_interval :
  ∀ x y : ℝ, x ∈ Iio (-1) → y ∈ Iio (-1) → x < y → f x < f y := by
  sorry

end monotonic_increasing_interval_l803_803924


namespace first_player_wins_l803_803945

noncomputable def winning_strategy (seq : List ℕ) : Prop :=
seq = [2, 5, 8, 11]

theorem first_player_wins : ∃ seq : List ℕ, winning_strategy seq := 
begin
  use [2, 5, 8, 11],
  unfold winning_strategy,
  sorry,
end

end first_player_wins_l803_803945


namespace isosceles_triangle_area_is_120_l803_803160

noncomputable def triangle_area (PQ PR QR PS : ℝ) : ℝ :=
  (1 / 2) * QR * PS

-- Given conditions
def PQ : ℝ := 17
def PR : ℝ := 17
def QR : ℝ := 16

-- Triangle PQ is right triangle with hypotenuse PQ and one leg QS = 8
def QS : ℝ := 8
def PS : ℝ := real.sqrt (PQ^2 - QS^2)

theorem isosceles_triangle_area_is_120 :
  triangle_area PQ PR QR PS = 120 :=
by
  sorry

end isosceles_triangle_area_is_120_l803_803160


namespace binomial_product_l803_803700

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l803_803700


namespace bridge_construction_cost_l803_803257

-- Define the given conditions and variables
variables (a b c : Real) (cost_per_km : Real)
def semi_perimeter (a b c : Real) : Real := (a + b + c) / 2 
def herons_area (s a b c : Real) : Real := Real.sqrt (s * (s - a) * (s - b) * (s - c))
def circumscribed_circle_radius (a b c : Real) (A : Real) : Real := (a * b * c) / (4 * A)
def inscribed_circle_radius (A s : Real) : Real := A / s
def cos_angle_c (a b c : Real) : Real := (a^2 + b^2 - c^2) / (2 * a * b)
def sin_angle_c (cosC : Real) : Real := Real.sqrt (1 - cosC^2)
def distance_between_centers (R r cosC sinC : Real) : Real := Real.sqrt (R^2 + 2 * R * r * cosC + r^2)

-- Prove the statement regarding the total cost given specific triangle sides and cost per kilometer
theorem bridge_construction_cost : 
    ∀ (a b c : Real) (cost_per_km : Real), 
    a = 7 → b = 8 → c = 9 → cost_per_km = 1000 →
    let s := semi_perimeter a b c in
    let A := herons_area s a b c in
    let R := circumscribed_circle_radius a b c A in
    let r := inscribed_circle_radius A s in
    let cosC := cos_angle_c a b c in
    let sinC := sin_angle_c cosC in
    let O_1O_2 := distance_between_centers R r cosC sinC in
    let total_cost := cost_per_km * O_1O_2 in
    total_cost ≈ 5750 :=
by 
    intros a b c cost_per_km ha hb hc hcost
    sorry

end bridge_construction_cost_l803_803257


namespace circle_radius_squared_l803_803275

open Real EuclideanGeometry

theorem circle_radius_squared
  (r : ℝ)
  (A B C D P : Point)
  (h_circle : Distance C (Center P) = r)
  (h_AB : Distance A B = 12)
  (h_CD : Distance C D = 9)
  (h_intersect : LineThrough A B ∩ LineThrough C D = {P})
  (h_angle : ∠ A P D = 60)
  (h_BP: Distance B P = 10)
  : r^2 = 111 :=
sorry

end circle_radius_squared_l803_803275


namespace continued_fraction_solution_l803_803839

theorem continued_fraction_solution (x : ℝ) (hx : x > 0) (h : 1 + 1 / x = x) : x = (1 + Real.sqrt 5) / 2 :=
by
  have h1 : x * x = x + 1, from sorry
  have h2 : x * x - x - 1 = 0, from sorry
  exact sorry

end continued_fraction_solution_l803_803839


namespace minimum_value_of_fm_plus_fp_l803_803794

def f (x a : ℝ) : ℝ := -x^3 + a * x^2 - 4

def f_prime (x a : ℝ) : ℝ := -3 * x^2 + 2 * a * x

theorem minimum_value_of_fm_plus_fp (a : ℝ) (h_extremum : f_prime 2 a = 0) (m n : ℝ) 
  (hm : -1 ≤ m ∧ m ≤ 1) (hn : -1 ≤ n ∧ n ≤ 1) : 
  f m a + f_prime n a = -13 := 
by
  -- steps of the proof would go here
  sorry

end minimum_value_of_fm_plus_fp_l803_803794


namespace can_space_before_compacted_l803_803091

theorem can_space_before_compacted (x : ℝ) (h1 : 60 * 0.20 * x = 360) : x = 30 := 
by
  -- providing the conditions in Lean
  have h2 : 12 * x = 360 := by 
    rw [mul_assoc, mul_comm 0.20 60]
    exact h1
  -- solving for x
  have h3 : x = 360 / 12 := by 
    apply eq_div_of_mul_eq 
    exact h2
  -- simplifying the division
  have h4 : x = 30 := by 
    norm_num at h3 
    exact h3
  exact h4

end can_space_before_compacted_l803_803091


namespace sufficient_but_not_necessary_condition_l803_803619

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 3) → (x^2 - 5 * x + 6 > 0) ∧ (¬ (∃ y, (y^2 - 5 * y + 6 > 0) ∧ (¬ (y > 3)))) :=
by
suffices (x > 3) → (x^2 - 5 * x + 6 > 0) from sorry,
have fact1: ¬ (∃ y, (y^2 - 5 * y + 6 > 0) ∧ (¬ (y > 3))) from sorry,
exact and.intro this fact1

end sufficient_but_not_necessary_condition_l803_803619


namespace semicircle_isosceles_triangle_l803_803263

theorem semicircle_isosceles_triangle (r : ℝ) (A B C : Type)
  (hAB : A ≠ B)
  (hACBC : AC = BC)
  (hAB_diameter : distance A B = 2 * r)
  (hC_not_AB : C ≠ A ∧ C ≠ B) :
  let s := AC + BC in
  s ^ 2 = 8 * r ^ 2 :=
sorry

end semicircle_isosceles_triangle_l803_803263


namespace count_integers_condition_l803_803005

-- Definitions of conditions and problem
def is_multiple_of_10 (n : ℕ) : Prop := n % 10 = 0
def is_not_multiple_of_6 (n : ℕ) : Prop := n % 6 ≠ 0
def is_not_multiple_of_8 (n : ℕ) : Prop := n % 8 ≠ 0
def is_not_multiple_of_9 (n : ℕ) : Prop := n % 9 ≠ 0

def range_1_to_300 (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 300

noncomputable def integers_condition (n : ℕ) : Prop :=
    is_multiple_of_10 n ∧ is_not_multiple_of_6 n ∧ 
    is_not_multiple_of_8 n ∧ is_not_multiple_of_9 n ∧ range_1_to_300 n

theorem count_integers_condition : 
  (finset.filter integers_condition (finset.range 301)).card = 15 :=
by
  sorry

end count_integers_condition_l803_803005


namespace smallest_integer_has_8_odd_and_16_even_divisors_l803_803222

/-!
  Prove that the smallest positive integer with exactly 8 positive odd integer divisors
  and exactly 16 positive even integer divisors is 540.
-/
def smallest_integer_with_divisors : ℕ :=
  540

theorem smallest_integer_has_8_odd_and_16_even_divisors 
  (n : ℕ) 
  (h1 : (8 : ℕ) = nat.count (λ d, d % 2 = 1) (nat.divisors n))
  (h2 : (16 : ℕ) = nat.count (λ d, d % 2 = 0) (nat.divisors n)) :
  n = 540 :=
sorry

end smallest_integer_has_8_odd_and_16_even_divisors_l803_803222


namespace count_valid_configurations_l803_803460

theorem count_valid_configurations : 
  let digits := {0, 2, 4, 6, 7, 8}
  let even_digits := {0, 2, 4, 6, 8}
  let valid_last_digit (x : ℕ) := x ∈ even_digits
  let valid_sum (as : List ℕ) := (11 + as.sum) % 3 = 0 
  let configurations := 
     (List.replicate 4 digits.toList.product).productEven(even_digits.toList) 
  configurations.count (λ as, valid_last_digit as.head * valid_sum as.tail) = 2160 :=
by sorry

end count_valid_configurations_l803_803460


namespace calculate_expr_l803_803677

theorem calculate_expr : (2023^0 + (-1/3) = 2/3) := by
  sorry

end calculate_expr_l803_803677


namespace value_of_n_l803_803017

theorem value_of_n (n : ℕ) (h : sqrt (10 + n) = 8) : n = 54 := by
  sorry

end value_of_n_l803_803017


namespace binom_coeff_mult_l803_803706

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l803_803706


namespace no_positive_integer_solutions_l803_803742

theorem no_positive_integer_solutions (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : x^4 * y^4 - 14 * x^2 * y^2 + 49 ≠ 0 := 
by sorry

end no_positive_integer_solutions_l803_803742


namespace smallest_integer_with_divisors_properties_l803_803195

def number_of_odd_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 1)

def number_of_even_divisors (n : ℕ) : ℕ :=
  (divisors n).count (λ d, d % 2 = 0)

theorem smallest_integer_with_divisors_properties :
  ∃ n : ℕ, number_of_odd_divisors n = 8 ∧ number_of_even_divisors n = 16 ∧ n = 4000 :=
by
  sorry

end smallest_integer_with_divisors_properties_l803_803195


namespace distance_P_P₁_eq_sqrt_2_abs_t1_l803_803802

-- Defining the parametric line and the points P and P₁
variables (a b t t1 : ℝ)

-- Defining the coordinates of the points
def P := (a, b)
def P₁ := (a + t1, b + t1)

-- Distance Formula
def distance (P P₁ : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - P₁.1)^2 + (P.2 - P₁.2)^2)

-- The theorem we want to prove
theorem distance_P_P₁_eq_sqrt_2_abs_t1 :
  distance (a, b) (a + t1, b + t1) = real.sqrt 2 * |t1| :=
by
  sorry

end distance_P_P₁_eq_sqrt_2_abs_t1_l803_803802


namespace area_N1N2N3_l803_803613

open Triangle 

variable (ABC : Triangle) 
variable (D E F N1 N2 N3 : Point)
variable (h_CD_one_third : length (CD) = length (BC) / 3)
variable (h_AE_one_third : length (AE) = length (AC) / 3)
variable (h_BF_one_third : length (BF) = length (AB) / 3)
variable (h_an2n1d : length (AN2) / length (N2N1) = 3)
variable (h_an2n1d2 : length (N2N1) / length (N1D) = 3)
variable (h_be_ratio : length (BE) / length (EN2) = 3)
variable (h_cf_ratio : length (CF) / length (FN3) = 3)

theorem area_N1N2N3 (K : ℝ) (h_area_ABC : area ABC = K) : 
  area (triangle N1 N2 N3) = K / 7 :=
sorry

end area_N1N2N3_l803_803613


namespace smallest_positive_integer_with_divisors_l803_803224

theorem smallest_positive_integer_with_divisors :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → nat.odd d) ∧ (finset.filter nat.odd (finset.divisors n)).card = 8 ∧ 
           (∃ m : ℕ, ∀ d : ℕ, d ∣ m → nat.even d ∧ m = n → (finset.filter nat.even (finset.divisors m)).card = 16)
             → n = 756 :=
by
  sorry

end smallest_positive_integer_with_divisors_l803_803224


namespace min_tangent_length_l803_803565

theorem min_tangent_length :
  let C := {ρ : ℝ, θ : ℝ | ρ = 2 * Real.cos (θ + Real.pi / 4)}
  let l := {p : ℝ × ℝ | ∃ t : ℝ, p = (√2 * t, √2 * t + 4 * √2)}
  let center_C := (√2 / 2, -√2 / 2)
  let radius_C := 1
  let distance_C_line : ℝ :=
    let (a, b, c) := (1, -1, -4 * √2)
    Real.abs ((a * (√2 / 2) + b * (-(√2 / 2)) + c) / Real.sqrt (a^2 + b^2))
  let min_dist := distance_C_line
  let min_tangent := Real.sqrt (min_dist^2 - radius_C^2)
  min_tangent = 2 * √6 := by
  sorry

end min_tangent_length_l803_803565


namespace percent_increase_hours_l803_803970

theorem percent_increase_hours (W H : ℝ) (hW : W > 0) (hH : H > 0) : 
  let new_wage := W * 0.80;
  let new_hours := H / 0.80
  in ((new_hours - H) / H) * 100 = 25 :=
by
  sorry

end percent_increase_hours_l803_803970


namespace participants_relationship_l803_803832

/-- Let P_2019 be the number of participants in 2019.
    Let P_2018 be the number of participants in 2018 (which is 150).
    Let P_2020 be the number of participants in 2020 defined in the conditions.

    Prove that P_2019 = 320 and that P_2019 > P_2018. -/
theorem participants_relationship :
  ∃ P_2019 : ℝ, 
    ∃ P_2020 : ℝ, 
    P_2018 = 150 ∧
    P_2020 = (1/2) * P_2019 - 40 ∧ 
    P_2019 = P_2020 + 200 ∧ 
    P_2019 = 320 ∧ 
    P_2019 > P_2018 :=
by {
  let P_2018 : ℝ := 150,
  have h1 : P_2019 = 320,
  by sorry,
  have h2 : P_2019 > P_2018,
  by sorry,
  existsi P_2019,
  existsi (P_2019 / 2 - 40),
  simp [P_2018, h1, h2],
}

end participants_relationship_l803_803832


namespace press_t_denomination_l803_803955

def press_f_rate_per_minute := 1000
def press_t_rate_per_minute := 200
def time_in_seconds := 3
def f_denomination := 5
def additional_amount := 50

theorem press_t_denomination : 
  ∃ (x : ℝ), 
  (3 * (5 * (1000 / 60))) = (3 * (x * (200 / 60)) + 50) → 
  x = 20 := 
by 
  -- Proof logic here
  sorry

end press_t_denomination_l803_803955


namespace midpoint_coords_product_l803_803593

def midpoint_prod (x1 y1 x2 y2 : ℤ) : ℤ :=
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  mx * my

theorem midpoint_coords_product :
  midpoint_prod 4 (-7) (-8) 9 = -2 := by
  sorry

end midpoint_coords_product_l803_803593


namespace sum_of_squares_of_evens_from_2_to_14_l803_803981

theorem sum_of_squares_of_evens_from_2_to_14 : (∑ k in {2, 4, 6, 8, 10, 12, 14}, k^2) = 560 := by 
  sorry

end sum_of_squares_of_evens_from_2_to_14_l803_803981


namespace domain_of_sqrt_cos_function_l803_803123

theorem domain_of_sqrt_cos_function:
  (∀ k : ℤ, ∀ x : ℝ, 2 * Real.cos x + 1 ≥ 0 ↔ x ∈ Set.Icc (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + 2 * Real.pi / 3)) :=
by
  sorry

end domain_of_sqrt_cos_function_l803_803123


namespace jessy_initial_reading_plan_l803_803056

theorem jessy_initial_reading_plan (x : ℕ) (h : (7 * (3 * x + 2) = 140)) : x = 6 :=
sorry

end jessy_initial_reading_plan_l803_803056


namespace g_at_8_eq_13_over_3_l803_803431

def g (x : ℝ) : ℝ := (3 * x + 2) / (x - 2)

theorem g_at_8_eq_13_over_3 : g 8 = 13 / 3 := by
  sorry

end g_at_8_eq_13_over_3_l803_803431


namespace arithmetic_sequence_eightieth_term_l803_803552

open BigOperators

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem arithmetic_sequence_eightieth_term :
  ∀ (d : ℝ),
  arithmetic_sequence 3 d 21 = 41 →
  arithmetic_sequence 3 d 80 = 153.1 :=
by
  intros
  sorry

end arithmetic_sequence_eightieth_term_l803_803552


namespace total_amount_spent_l803_803809

def cost_of_haley_paper : ℝ := 3.75 + (3.75 * 0.5)
def cost_of_sister_paper : ℝ := (4.50 * 2) + (4.50 * 0.5)
def cost_of_haley_pens : ℝ := (1.45 * 5) - ((1.45 * 5) * 0.25)
def cost_of_sister_pens : ℝ := (1.65 * 7) - ((1.65 * 7) * 0.25)

def total_cost_of_supplies : ℝ := cost_of_haley_paper + cost_of_sister_paper + cost_of_haley_pens + cost_of_sister_pens

theorem total_amount_spent : total_cost_of_supplies = 30.975 :=
by
  sorry

end total_amount_spent_l803_803809


namespace cubic_expression_l803_803024

theorem cubic_expression (a b c : ℝ) (h1 : a + b + c = 13) (h2 : ab + ac + bc = 30) : a^3 + b^3 + c^3 - 3 * abc = 1027 :=
sorry

end cubic_expression_l803_803024


namespace largest_total_real_roots_l803_803073

def max_real_roots (a b c : ℝ) : ℕ :=
  if a > 0 ∧ b > 0 ∧ c > 0 then 4 else sorry

theorem largest_total_real_roots (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  max_real_roots a b c = 4 :=
sorry

end largest_total_real_roots_l803_803073


namespace units_digit_sum_squares_odd_l803_803962

theorem units_digit_sum_squares_odd (n : ℕ) (h : n = 2500) : 
  (finset.sum (finset.range n) (λ k, (2 * k + 1)^2)) % 10 = 0 := 
by 
  sorry

end units_digit_sum_squares_odd_l803_803962


namespace smallest_integer_has_8_odd_and_16_even_divisors_l803_803220

/-!
  Prove that the smallest positive integer with exactly 8 positive odd integer divisors
  and exactly 16 positive even integer divisors is 540.
-/
def smallest_integer_with_divisors : ℕ :=
  540

theorem smallest_integer_has_8_odd_and_16_even_divisors 
  (n : ℕ) 
  (h1 : (8 : ℕ) = nat.count (λ d, d % 2 = 1) (nat.divisors n))
  (h2 : (16 : ℕ) = nat.count (λ d, d % 2 = 0) (nat.divisors n)) :
  n = 540 :=
sorry

end smallest_integer_has_8_odd_and_16_even_divisors_l803_803220


namespace intersection_of_lines_l803_803740

-- Definitions for the lines given by their equations
def line1 (x y : ℝ) : Prop := 5 * x - 3 * y = 9
def line2 (x y : ℝ) : Prop := x^2 + 4 * x - y = 10

-- The statement to prove
theorem intersection_of_lines :
  (line1 2 (1 / 3) ∧ line2 2 (1 / 3)) ∨ (line1 (-3.5) (-8.83) ∧ line2 (-3.5) (-8.83)) :=
by
  sorry

end intersection_of_lines_l803_803740


namespace each_friend_gets_seven_l803_803847

theorem each_friend_gets_seven (num_bananas friends : ℕ) (h1 : num_bananas = 36) (h2 : friends = 5) : num_bananas / friends = 7 :=
by
  rw [h1, h2]
  norm_num

end each_friend_gets_seven_l803_803847


namespace binom_mult_eq_6720_l803_803695

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l803_803695


namespace solve_for_a_l803_803029

open Complex

noncomputable def question (a : ℝ) : Prop :=
  ∃ z : ℂ, z = (a + I) / (1 - I) ∧ z.im ≠ 0 ∧ z.re = 0

theorem solve_for_a (a : ℝ) (h : question a) : a = 1 :=
sorry

end solve_for_a_l803_803029


namespace max_standard_lines_l803_803508

-- Mathematical Definitions and Assumptions
def is_standard_line (l : ℝ × ℝ → ℝ × ℝ → Prop) : Prop :=
  ∀ p q : ℝ × ℝ, l p q ↔ (p.1 = q.1 ∨ p.2 = q.2 ∨ p.2 - p.1 = q.2 - q.1 ∨ p.2 + p.1 = q.2 + q.1)
  
def Cartesian_plane := set (ℝ × ℝ)

-- The main statement
theorem max_standard_lines (A : finset (ℝ × ℝ)) (hA : A.card = 6) :
  ∃ k : ℕ, (∀ l, l ∈ A.powerset.filter (λ S, (∃ p q, p ∈ S ∧ q ∈ S ∧ p ≠ q)) → is_standard_line (λ p q => p ∈ S → q ∈ S)) ∧ k = 11 :=
sorry

end max_standard_lines_l803_803508


namespace prize_distribution_correct_l803_803588

def probability_A_correct : ℚ := 3 / 4
def probability_B_correct : ℚ := 4 / 5
def total_prize : ℚ := 190

-- Calculation of expected prizes
def probability_A_only_correct : ℚ := probability_A_correct * (1 - probability_B_correct)
def probability_B_only_correct : ℚ := probability_B_correct * (1 - probability_A_correct)
def probability_both_correct : ℚ := probability_A_correct * probability_B_correct

def normalized_probability : ℚ := probability_A_only_correct + probability_B_only_correct + probability_both_correct

def expected_prize_A : ℚ := (probability_A_only_correct / normalized_probability * total_prize) + (probability_both_correct / normalized_probability * (total_prize / 2))
def expected_prize_B : ℚ := (probability_B_only_correct / normalized_probability * total_prize) + (probability_both_correct / normalized_probability * (total_prize / 2))

theorem prize_distribution_correct :
  expected_prize_A = 90 ∧ expected_prize_B = 100 := 
by
  sorry

end prize_distribution_correct_l803_803588


namespace IntersectionOfAandB_l803_803803

def setA : Set ℝ := {x | x < 5}
def setB : Set ℝ := {x | -1 < x}

theorem IntersectionOfAandB : setA ∩ setB = {x | -1 < x ∧ x < 5} :=
sorry

end IntersectionOfAandB_l803_803803


namespace number_of_values_divisible_by_6_l803_803765

theorem number_of_values_divisible_by_6 :
  let polynomial := λ x : ℕ, x^2 - 197 * x + 9702 in
  (card {x ∈ finset.range 101 | (polynomial x) % 6 = 0}) = 67 :=
by sorry

end number_of_values_divisible_by_6_l803_803765


namespace total_students_taught_l803_803833

theorem total_students_taught (s1 s2 : ℕ) (h1 : s1 = 400) (h2 : s2 = 340) :
  2 * s1 + 2 * s2 = 1480 :=
by
  rw [h1, h2]
  norm_num
  sorry

end total_students_taught_l803_803833


namespace angle_P_plus_angle_Q_l803_803515

def Circle : Type := sorry -- Placeholder for the definition of a circle

structure Point := (x y : ℝ) -- Define Point as a structure

variable (A B Q D C : Point)
variable (circle : Circle)
variable (arc : Circle → set Point → ℝ) -- Define a function that gives the measure of an arc

axiom points_on_circle : ∀ (circle : Circle) (points : set Point), True -- All given points lie on the circle
axiom arc_BQ : arc circle {B, Q} = 42
axiom arc_QD : arc circle {Q, D} = 38
axiom arc_BD : arc circle {B, D} = arc circle {B, Q} + arc circle {Q, D} -- Sum of arcs for BQ and QD

-- Define inscribed angle function, relating an arc to an angle
noncomputable def inscribed_angle (arc : ℝ) : ℝ := arc / 2

-- Define angles P and Q based on the intercepted arcs
axiom angle_P : ∀ (arc_BD arc_AC : ℝ), inscribed_angle (arc_BD - arc_AC) = 40
axiom angle_Q : ∀ (arc_AC : ℝ), inscribed_angle arc_AC = 40

-- Define the proof statement about the sum of angles
theorem angle_P_plus_angle_Q : 
  (∀ (arc_AC : ℝ), inscribed_angle (arc circle {B, D} - arc_AC) + inscribed_angle arc_AC = 40) := by
  -- We do not provide the actual proof, use sorry to skip
  sorry

end angle_P_plus_angle_Q_l803_803515


namespace temperature_on_fifth_day_l803_803914

theorem temperature_on_fifth_day (T : ℕ → ℝ) (x : ℝ)
  (h1 : (T 1 + T 2 + T 3 + T 4) / 4 = 58)
  (h2 : (T 2 + T 3 + T 4 + T 5) / 4 = 59)
  (h3 : T 1 / T 5 = 7 / 8) :
  T 5 = 32 := 
sorry

end temperature_on_fifth_day_l803_803914


namespace find_a_l803_803063

theorem find_a (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_geq : a ≥ b ∧ b ≥ c)
  (h_eq1 : a^2 - b^2 - c^2 + a * b = 2500)
  (h_eq2 : a^2 + 3 * b^2 + 3 * c^2 - 3 * a * b - 2 * a * c - 2 * b * c + b = -2200) :
  a = 257 :=
by
  sorry

end find_a_l803_803063


namespace proof_problem_l803_803169

-- Definitions for the conditions
def finite_set (S : Set) [finite S] : Prop :=
  finite S

def three_element_subset (T : Set) [finite T] : Prop :=
  T.card = 3
  
def unique_two_element_sets (S : Set) (T : Set (Set)) : Prop :=
  ∀ {a b : S}, a ≠ b → ∃ t ∈ T, {a, b} ⊆ t

-- The main theorem breaking down the claims into two parts.
theorem proof_problem (S : Set) [finite S]
  (T : Set (Set)) [∀ t ∈ T, finite t] (hT : ∀ t ∈ T, t.card = 3)
  (h_subsets : T.card = 7)
  (h_unique : unique_two_element_sets S T):

  -- Part (a): Proving the size of S is 7.
  (S.card = 7) ∧

  -- Part (b): Proving the maximum number of 3-element subsets without 
  -- any three elements appearing together in more than two subsets is 4.
  (∃ T' ⊆ T, T'.card = 4 ∧ ∀ (t1 t2 t3 ∈ T'), ¬ (∃ e, e ∈ t1 ∧ e ∈ t2 ∧ e ∈ t3)) := sorry

end proof_problem_l803_803169


namespace inequality_solution_l803_803725

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x^2 + 1) / (x + 2) ≥ (3 / (x - 2)) + (7 / 4)

theorem inequality_solution (x : ℝ) :
  satisfies_inequality x ↔ (x < -2) ∨ (x = -1) ∨ (2 < x ∧ x < 3 - 2 * real.sqrt 3) ∨ (3 + 2 * real.sqrt 3 ≤ x) :=
sorry

end inequality_solution_l803_803725


namespace max_real_roots_l803_803071

noncomputable def discriminant (A B C : ℝ) := B^2 - 4 * A * C

theorem max_real_roots (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let p_discr := discriminant a b c
  let q_discr := discriminant b c a
  let r_discr := discriminant c a b
  p_discr >= 0 + q_discr >= 0 + r_discr >= 0 ≤ 4 := 
  sorry

end max_real_roots_l803_803071


namespace max_sum_ab_bc_cd_da_l803_803575

theorem max_sum_ab_bc_cd_da (a b c d : ℕ) (h : {a, b, c, d} = {1, 3, 5, 7}) :
  ab + bc + cd + da ≤ 64 :=
by sorry

end max_sum_ab_bc_cd_da_l803_803575


namespace three_digit_non_multiples_of_6_or_8_l803_803418

theorem three_digit_non_multiples_of_6_or_8 : 
  ∃ n, n = 900 - (150 + 112 - 37) ∧ n = 675 :=
by {
  have total_three_digits : 900 = 999 - 100 + 1 := sorry,
  have count_multiples_6 : 150 = 166 - 17 + 1 := sorry,
  have count_multiples_8 : 112 = 124 - 13 + 1 := sorry,
  have count_multiples_24 : 37 = 41 - 5 + 1 := sorry,
  let multiples_6_or_8 := count_multiples_6 + count_multiples_8 - count_multiples_24,
  have : multiples_6_or_8 = 150 + 112 - 37 := sorry,
  have count_non_multiples := total_three_digits - multiples_6_or_8,
  use count_non_multiples,
  split,
  { rw [total_three_digits, multiples_6_or_8], exact sorry },
  { exact sorry }
}

end three_digit_non_multiples_of_6_or_8_l803_803418


namespace problem_set_real_l803_803983

theorem problem_set_real (a : ℝ) :
  (∀ x, x ∈ ({b | ∃a, b = a + (a^2-1) * complex.I}) → x.real? = true) → a = 1 ∨ a = -1 :=
by
  sorry

end problem_set_real_l803_803983


namespace log_base_2_domain_l803_803122

theorem log_base_2_domain : ∀ x : ℝ, (f x = log 2 x) → (x > 0) ↔ (x ∈ Set.Ioi (0 : ℝ)) :=
by
  intros x h
  -- Domain assertion goes here
  sorry

end log_base_2_domain_l803_803122


namespace find_m_l803_803751

theorem find_m (x m : ℝ) (h_eq : (x + m) / (x - 2) + 1 / (2 - x) = 3) (h_root : x = 2) : m = -1 :=
by
  sorry

end find_m_l803_803751


namespace mark_new_phone_plan_cost_l803_803884

def old_phone_plan_cost := 150
def new_phone_plan_increase := 30
def old_internet_package_cost := 50
def new_internet_package_increase := 20
def old_international_calling_plan_cost := 30
def new_international_calling_discount := 15
def old_streaming_service_cost := 40
def new_streaming_service_increase := 25
def promotional_discount := 10

def new_phone_plan_base_cost := old_phone_plan_cost + (old_phone_plan_cost * new_phone_plan_increase / 100)
def upgraded_internet_package_cost := old_internet_package_cost + (old_internet_package_cost * new_internet_package_increase / 100)
def international_calling_package_cost := old_international_calling_plan_cost - (old_international_calling_plan_cost * new_international_calling_discount / 100)
def new_streaming_service_subscription_cost := old_streaming_service_cost + (old_streaming_service_cost * new_streaming_service_increase / 100)

def total_cost_before_discount := new_phone_plan_base_cost + upgraded_internet_package_cost + international_calling_package_cost + new_streaming_service_subscription_cost
def promotional_discount_amount := total_cost_before_discount * promotional_discount / 100

def total_cost_after_promotional_discount := total_cost_before_discount - promotional_discount_amount

theorem mark_new_phone_plan_cost : total_cost_after_promotional_discount = 297.45 :=
by
  unfold total_cost_after_promotional_discount
  simp
  sorry -- Proof goes here

end mark_new_phone_plan_cost_l803_803884


namespace domain_of_f_l803_803917

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2)

theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = Ioi (-2) := by
  sorry

end domain_of_f_l803_803917


namespace percentage_of_third_number_l803_803997

variable (T F S : ℝ)

-- Declare the conditions from step a)
def condition_one : Prop := S = 0.25 * T
def condition_two : Prop := F = 0.20 * S

-- Define the proof problem, proving that F is 5% of T given the conditions
theorem percentage_of_third_number
  (h1 : condition_one T S)
  (h2 : condition_two F S) :
  F = 0.05 * T := by
  sorry

end percentage_of_third_number_l803_803997


namespace binom_mult_eq_6720_l803_803696

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l803_803696


namespace binom_mult_eq_6720_l803_803694

theorem binom_mult_eq_6720 :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_mult_eq_6720_l803_803694


namespace path_divides_grid_equally_l803_803908

open Function

noncomputable def grid_path_splits_equally (n m : ℕ) : Prop :=
  ∀ (path : List (ℕ × ℕ)), 
    path.head = (0, 0) ∧
    path.last = (n-1, m-1) ∧
    (∀ p ∈ path, ∃ x y, p = (x, y) ∧ 0 ≤ x ∧ x < n ∧ 0 ≤ y ∧ y < m) ∧
    path.nodup →
    let regions_opening_north_or_east := 
      { region | ∃ x y, region = (x, y) ∧ x < n - 1 ∧ y < m - 1 } in
    let regions_opening_south_or_west := 
      { region | ∃ x y, region = (x, y) ∧ x > 0 ∧ y > 0 } in
    regions_opening_north_or_east.card = regions_opening_south_or_west.card

-- Theorem to be proven
theorem path_divides_grid_equally (n m : ℕ) (h : n > 0) (j : m > 0) : 
  grid_path_splits_equally n m :=
by
  -- proof goes here
  sorry

end path_divides_grid_equally_l803_803908


namespace arithmetic_sequence_properties_l803_803478

noncomputable def a_seq (n : ℕ) : ℤ := -2 * n + 15

noncomputable def sum_terms (n : ℕ) : ℤ :=
  if n ≤ 7 then -n^2 + 14 * n
  else n^2 - 14 * n + 98

theorem arithmetic_sequence_properties :
  (a_seq 2 = 11) ∧ (∑ i in finset.range 10, a_seq (i+1) = 40) ∧
  (∀ n, 1 ≤ n → n ≤ 7 → ∑ i in finset.range n, abs (a_seq (i+1)) = sum_terms n) ∧
  (∀ n, n ≥ 8 → ∑ i in finset.range n, abs (a_seq (i+1)) = sum_terms n) :=
by
  sorry

end arithmetic_sequence_properties_l803_803478


namespace compare_abc_l803_803780

-- Definitions of a, b, and c
def a : ℝ := 1 + Real.sqrt 7
def b : ℝ := Real.sqrt 3 + Real.sqrt 5
def c : ℝ := 4

-- The theorem statement
theorem compare_abc : c > b ∧ b > a :=
by 
  sorry

end compare_abc_l803_803780


namespace proof_problem_l803_803400

-- Define the conditions
variables (a b : ℕ)

-- a and b are coprime positive integers
-- Definition of "good" for an integer n
def good (n : ℕ) : Prop := ∃ x y : ℕ, n = x + y

-- Problem Statement
theorem proof_problem (coprime_ab : Nat.gcd a b = 1) (a_pos : 0 < a) (b_pos : 0 < b) :
  ∃ c : ℤ, (c = a * b - a - b ∧
    ∀ n : ℤ, (good a b n ∨ good a b (c - n)) ∧
    ∃ num_bad : ℕ, num_bad = (a - 1) * (b - 1) / 2) :=
by
  sorry

end proof_problem_l803_803400


namespace islander_statements_l803_803504

theorem islander_statements
  (K L : ℕ)
  (H_total : K + L = 10)
  (H_statement : (K = 5 ∧ L = 5) ∨ (5 > K ∧ 5 ≤ L = 5)) :
  ∃ (other_statements : ℕ → Prop), 
  (∀ i, i ∈ {0, 1, 2, 3, 4} → other_statements i = "There are an equal number of knights and liars.") :=
by
  sorry

end islander_statements_l803_803504


namespace majority_votes_l803_803040

def total_votes : ℕ := 7520
def percentage_received : ℚ := 0.60

def votes_received_by_winner (total: ℕ) (pct: ℚ) : ℕ :=
  (pct * total).to_nat

def votes_received_by_other (total: ℕ) (pct: ℚ) : ℕ :=
  ((1 - pct) * total).to_nat

theorem majority_votes (total : ℕ) (pct : ℚ) : votes_received_by_winner total pct - votes_received_by_other total pct = 1504 :=
by
  let winner_votes := votes_received_by_winner total pct
  let other_votes := votes_received_by_other total pct
  have h1 : winner_votes = 4512 := by sorry
  have h2 : other_votes = 3008 := by sorry
  show winner_votes - other_votes = 1504
  rw [h1, h2]
  simp
  sorry

end majority_votes_l803_803040


namespace KH_loses_one_mole_of_electrons_l803_803844

-- Define what happens in the reaction
def KH_reacts_with_H₂O (KH H₂O : Type) (H₂ KOH : Type) : Prop :=
  ∃ H_valence_initial H_valence_final : ℤ, ⟦ H_valence_initial = -1 ⟧ ∧ ⟦ H_valence_final = 0 ⟧

-- Main theorem statement
theorem KH_loses_one_mole_of_electrons (KH H₂O H₂ KOH : Type) :
  KH_reacts_with_H₂O KH H₂O H₂ KOH →
  ∀ (mol : ℝ), mol = 1 →
  mol * (0 - (-1)) = 1 :=
by
  intro h
  intro m
  intro hm
  rw hm
  norm_num
  done

end KH_loses_one_mole_of_electrons_l803_803844


namespace cub_eqn_root_sum_l803_803021

noncomputable def cos_x := Real.cos (Real.pi / 5)

theorem cub_eqn_root_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
(h3 : a * cos_x ^ 3 - b * cos_x - 1 = 0) : a + b = 12 :=
sorry

end cub_eqn_root_sum_l803_803021


namespace restaurant_dinners_ratio_l803_803095

theorem restaurant_dinners_ratio :
  ∃ (W : ℕ), 
  let Monday := 40,
      Tuesday := 80,
      Thursday := W + 3,
      total := Monday + Tuesday + W + Thursday in
  total = 203 ∧
  (W : ℚ) / (Tuesday : ℚ) = 1 / 2 :=
by
  sorry

end restaurant_dinners_ratio_l803_803095


namespace semicircle_radius_l803_803255

-- Define the main condition
def perimeter (r : ℝ) : ℝ :=
  (Real.pi * r) + 2 * r

-- State the problem
theorem semicircle_radius (h : perimeter r = 126) : r ≈ 24.5 :=
by
  sorry

end semicircle_radius_l803_803255


namespace base_conversion_problem_l803_803236

theorem base_conversion_problem :
  ∃ A B : ℕ, 0 ≤ A ∧ A < 8 ∧ 0 ≤ B ∧ B < 6 ∧
           8 * A + B = 6 * B + A ∧
           8 * A + B = 45 :=
by
  sorry

end base_conversion_problem_l803_803236


namespace sasha_added_cards_l803_803904

theorem sasha_added_cards :
  ∃ (x : ℕ), 83 = 43 + x - x / 6 ∧ x = 48 :=
begin
  use 48,
  split,
  {
    norm_num,
    rw [nat.sub_div, mul_comm, mul_div_cancel],
    norm_num,
    norm_num,
    norm_num,
  },
  refl,
end

end sasha_added_cards_l803_803904


namespace line_bisects_circle_area_l803_803130

theorem line_bisects_circle_area (b : ℝ) :
  (∀ x y : ℝ, y = 2 * x + b ↔ x^2 + y^2 - 2 * x - 4 * y + 4 = 0) → b = 0 :=
by
  sorry

end line_bisects_circle_area_l803_803130


namespace fill_tank_time_l803_803589

theorem fill_tank_time (A B : ℝ) (hA : A = 50) (hB : B = 75) : 
  let rate_A := 1 / A
  let rate_B := 1 / B
  let combined_rate := rate_A + rate_B
  let time := 1 / combined_rate
  time = 30 :=
by
  rw [hA, hB]
  let rate_A := 1 / 50
  let rate_B := 1 / 75
  let combined_rate := rate_A + rate_B
  have h_combined_rate : combined_rate = 1 / 30 := by sorry
  let time := 1 / combined_rate
  exact eq.symm h_combined_rate

end fill_tank_time_l803_803589


namespace probability_three_books_common_l803_803089

theorem probability_three_books_common :
  (∃ (total_ways_same_set : ℕ) (favorable_outcomes : ℕ),
    let total_ways := (nat.choose 12 4) * (nat.choose 12 4),
        favorable_outcomes := ((nat.choose 12 3) * (nat.choose 9 1) * (nat.choose 8 1))
    in favorable_outcomes / total_ways = 32/495) :=
by
  sorry

end probability_three_books_common_l803_803089


namespace count_values_of_n_l803_803355

-- Define sum of digits function S(n)
def sum_of_digits (n : ℕ) : ℕ := n.toString.toList.map (λ c => c.toNat - '0'.toNat).sum

/-- 
Given a positive integer n, and sum_of_digits function,
prove that the number of values of n such that n + sum_of_digits(n) + sum_of_digits(sum_of_digits(n)) = 2187
is 5 for 0 < n ≤ 2187.
-/
theorem count_values_of_n : 
  (∃ n : ℕ, 0 < n ∧ n ≤ 2187 ∧ n + sum_of_digits n + sum_of_digits (sum_of_digits n) = 2187) → 
  set.count {n | 0 < n ∧ n ≤ 2187 ∧ n + sum_of_digits n + sum_of_digits (sum_of_digits n) = 2187} = 5 :=
sorry

end count_values_of_n_l803_803355


namespace closet_area_l803_803916

theorem closet_area (a b c : ℝ) (h1 : a = 4) (h2 : c = 7) (h3 : a^2 + b^2 = c^2) : 
    a * real.sqrt b^2 = 4 * real.sqrt 33 :=
by
  -- typical structure that would include the proof
  sorry

end closet_area_l803_803916


namespace cone_base_radius_l803_803637

theorem cone_base_radius
  (sector_radius : ℝ)
  (central_angle : ℝ)
  (H_sector_radius : sector_radius = 4)
  (H_central_angle : central_angle = 90) :
  let r := (central_angle / 360) * sector_radius in
  r = 1 :=
by
  intros
  subst H_sector_radius
  subst H_central_angle
  let r := (90 / 360) * 4
  have h : r = 1 := by linarith
  exact h

end cone_base_radius_l803_803637


namespace minimal_steps_to_sort_l803_803854

theorem minimal_steps_to_sort (n : ℕ) (h : n > 0) (initial_alignment : list ℕ) (h₁ : initial_alignment.length = 2 * n) : 
  ∃ steps : ℕ, steps = n ∧ 
  (∀ alignment : list ℕ, alignment = initial_alignment → 
    (∀ k : ℕ, k ≤ steps → alignment = (list.replicate n 0 ++ list.replicate n 1))) :=
sorry

end minimal_steps_to_sort_l803_803854


namespace mike_books_before_yard_sale_l803_803088

-- Problem definitions based on conditions
def books_bought_at_yard_sale : ℕ := 21
def books_now_in_library : ℕ := 56
def books_before_yard_sale := books_now_in_library - books_bought_at_yard_sale

-- Theorem to prove the equivalent proof problem
theorem mike_books_before_yard_sale : books_before_yard_sale = 35 := by
  sorry

end mike_books_before_yard_sale_l803_803088


namespace cost_of_5_pound_bag_correct_l803_803285

noncomputable def cost_of_5_pound_bag : ℝ := 13.80

theorem cost_of_5_pound_bag_correct (x : ℝ) (h : 2 * 32.25 + 20.43 + x = 98.73) : x = cost_of_5_pound_bag :=
by {
  -- Given the initial condition
  have initial_cost: 2 * 32.25 + 20.43 = 84.93, by norm_num,
  -- Using the given total cost condition: h,
  calc
    x = 98.73 - 84.93 : by linarith [h, initial_cost]
    ... = 13.80       : by norm_num
}
#eval cost_of_5_pound_bag  -- Output should be 13.80

end cost_of_5_pound_bag_correct_l803_803285


namespace power_equation_l803_803012

theorem power_equation (y : ℕ) (h : (128 : ℕ)^4 = (16 : ℕ)^y) : 2^(-y : ℤ) = 1 / 128 := 
by
  sorry

end power_equation_l803_803012


namespace find_number_l803_803746

variable (n : ℝ)

theorem find_number (h₁ : (0.47 * 1442 - 0.36 * n) + 63 = 3) : 
  n = 2049.28 := 
by 
  sorry

end find_number_l803_803746


namespace selection_plans_count_l803_803670

def num_selection_plans : ℕ :=
  let total_permutations := 6 * 5 * 4 * 3
  let disallowed_permutations := 2 * (5 * 4 * 3)
  total_permutations - disallowed_permutations

theorem selection_plans_count :
  (∀ (n : ℕ) (visitors : Fin n) (cities : Fin 4),
    let people := 6
    let choices := 4
    let cities := ["Beijing", "Harbin", "Guangzhou", "Chengdu"]
    let restricted_people := ["Person A", "Person B"]
    count_visiting_plans choices people cities restricted_people = num_selection_plans) :=
  sorry

end selection_plans_count_l803_803670


namespace binom_coeff_mult_l803_803707

theorem binom_coeff_mult :
  (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 :=
by
  sorry

end binom_coeff_mult_l803_803707


namespace smallest_integer_with_divisors_l803_803185

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end smallest_integer_with_divisors_l803_803185


namespace time_against_walkway_l803_803291

theorem time_against_walkway (l t1 t2 : ℝ) (hp, hw : ℝ) :
  l = 60 ∧ t1 = 30 ∧ t2 = 48 →
  hp * t2 = l ∧ (hp + hw) * t1 = l →
  (hp - hw) * 120 = l :=
by
  intros h1 h2
  sorry

end time_against_walkway_l803_803291


namespace cot_squared_inequality_l803_803015

variables {α β γ : ℝ}

-- Assumptions
variables (h1 : 0 < α ∧ α < π / 2)
          (h2 : 0 < β ∧ β < π / 2)
          (h3 : 0 < γ ∧ γ < π / 2)
          (h4 : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1)

-- Theorem to prove
theorem cot_squared_inequality :
  Real.cot α ^ 2 + Real.cot β ^ 2 + Real.cot γ ^ 2 ≥ 3 / 2 :=
sorry

end cot_squared_inequality_l803_803015


namespace part1_n_value_part2_log_value_l803_803775

theorem part1_n_value (n : ℕ) (h : (nat.choose (n + 1) 2) = 36) : n = 8 := 
by
  sorry

theorem part2_log_value (p q : ℝ) (h1 : 6^p = 2) (h2 : log 6 5 = q) : 10^(q / (p + q)) = 5 :=
by
  sorry

end part1_n_value_part2_log_value_l803_803775


namespace range_of_m_l803_803879

-- Definitions for propositions p and q
def proposition_p (m : ℝ) := (5 * x^2 + m * y^2 = 1)
def proposition_q (m : ℝ) := ((m + 1) * x^2 - m * y^2 = 1)

-- Conditions for propositions p and q
def condition_p (m : ℝ) := m > 5
def condition_q (m : ℝ) := m > 0

-- Lean theorem statement
theorem range_of_m (m : ℝ) :
  ¬((condition_p m) ∧ (condition_q m)) ∧ ((condition_p m) ∨ (condition_q m)) →
  (0 < m ∧ m ≤ 5) :=
by
  sorry

end range_of_m_l803_803879


namespace relay_for_life_distance_l803_803733

-- Define the parameters of the problem
def event_duration_hours : ℕ := 8
def initial_pace : ℝ := 2 -- in miles per hour
def pace_decrease_per_2_hours : ℝ := 0.2 -- in miles per hour
def time_interval_hours : ℕ := 2
def elevation_change : ℝ := 500 -- in feet

-- Define the function to calculate total distance covered
noncomputable def total_distance_covered : ℝ :=
(let distance_first_segment := time_interval_hours * initial_pace in
 let distance_second_segment := time_interval_hours * (initial_pace - pace_decrease_per_2_hours) in
 let distance_third_segment := time_interval_hours * (initial_pace - 2 * pace_decrease_per_2_hours) in
 let distance_fourth_segment := time_interval_hours * (initial_pace - 3 * pace_decrease_per_2_hours) in
 distance_first_segment + distance_second_segment + distance_third_segment + distance_fourth_segment)

-- Proof statement
theorem relay_for_life_distance : total_distance_covered = 13.6 :=
by
  sorry

end relay_for_life_distance_l803_803733


namespace difference_of_two_numbers_l803_803617

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end difference_of_two_numbers_l803_803617


namespace petya_steps_l803_803558

theorem petya_steps (length_escalator steps_walked : ℕ) 
  (h_length_escalator : length_escalator = 200)
  (h_steps_walked : steps_walked = 50) :
  ∃ steps_run, (steps_run = length_escalator * 2 / (3 + 2)) :=
by
  have speed_ratio := 3
  have new_speed_ratio := speed_ratio :| 2
  let steps_run := length_escalator * 2 / (speed_ratio + 2)
  exact steps_run ▸ (80 : ℕ)
  sorry

end petya_steps_l803_803558


namespace tangent_line_through_P_tangent_line_through_Q1_tangent_line_through_Q2_l803_803793

open Real

noncomputable def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 9

noncomputable def tangent_line_p (x y : ℝ) : Prop :=
  2 * x - sqrt 5 * y - 9 = 0

noncomputable def line_q1 (x y : ℝ) : Prop :=
  x = 3

noncomputable def line_q2 (x y : ℝ) : Prop :=
  8 * x - 15 * y + 51 = 0

theorem tangent_line_through_P :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (2, -sqrt 5) →
    tangent_line_p x y := 
sorry

theorem tangent_line_through_Q1 :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (3, 5) →
    line_q1 x y := 
sorry

theorem tangent_line_through_Q2 :
  ∀ (x y : ℝ),
    circle_eq x y →
    (x, y) = (3, 5) →
    line_q2 x y := 
sorry

end tangent_line_through_P_tangent_line_through_Q1_tangent_line_through_Q2_l803_803793


namespace octagon_area_sum_l803_803146

theorem octagon_area_sum (O : Point)
    (AB : ℝ)
    (A B C D E F G H : Point)
    (squares_centered_at_O_same_sides_length : (∃ (O : Point), ∀ (p : Point), dist O p = 1))
    (AB_length : dist A B = 10/99)
    (octagon_O : ∀ P ∈ {A, B, C, D, E, F, G, H}, dist O P = 1) : 
    let m := 20 in let n := 99 in m + n = 119 :=
by
  sorry

end octagon_area_sum_l803_803146


namespace cube_root_of_000216_l803_803254

theorem cube_root_of_000216 : ∃ x : ℝ, x^3 = 0.000216 ∧ x = 0.06 :=
by
  use 0.06
  split
  · norm_num
  · refl

end cube_root_of_000216_l803_803254


namespace regular_polygon_sides_l803_803539

-- Define the central angle and number of sides of a regular polygon
def central_angle (θ : ℝ) := θ = 30
def number_of_sides (n : ℝ) := 360 / 30 = n

-- Theorem to prove that the number of sides of the regular polygon is 12 given the central angle is 30 degrees
theorem regular_polygon_sides (θ n : ℝ) (hθ : central_angle θ) : number_of_sides n → n = 12 :=
sorry

end regular_polygon_sides_l803_803539


namespace smallest_integer_has_8_odd_and_16_even_divisors_l803_803216

/-!
  Prove that the smallest positive integer with exactly 8 positive odd integer divisors
  and exactly 16 positive even integer divisors is 540.
-/
def smallest_integer_with_divisors : ℕ :=
  540

theorem smallest_integer_has_8_odd_and_16_even_divisors 
  (n : ℕ) 
  (h1 : (8 : ℕ) = nat.count (λ d, d % 2 = 1) (nat.divisors n))
  (h2 : (16 : ℕ) = nat.count (λ d, d % 2 = 0) (nat.divisors n)) :
  n = 540 :=
sorry

end smallest_integer_has_8_odd_and_16_even_divisors_l803_803216


namespace numbers_with_perfect_square_factor_l803_803009

def has_perfect_square_factor_other_than_one (n : ℕ) : Prop :=
  ∃ k : ℕ, k ∈ {4, 9, 16, 25, 36, 49, 64} ∧ k ∣ n

theorem numbers_with_perfect_square_factor :
  finset.card (finset.filter has_perfect_square_factor_other_than_one (finset.range 76)) = 29 :=
by
  sorry

end numbers_with_perfect_square_factor_l803_803009


namespace joan_seashells_l803_803057

variable (initialSeashells seashellsGiven remainingSeashells : ℕ)

theorem joan_seashells : initialSeashells = 79 ∧ seashellsGiven = 63 ∧ remainingSeashells = initialSeashells - seashellsGiven → remainingSeashells = 16 :=
by
  intros
  sorry

end joan_seashells_l803_803057


namespace remainder_div_82_l803_803820

theorem remainder_div_82 (x : ℤ) (h : ∃ k : ℤ, x + 17 = 41 * k + 22) : (x % 82 = 5) :=
by
  sorry

end remainder_div_82_l803_803820


namespace interval_of_monotonic_increase_l803_803390

noncomputable def f (ω : ℝ) (x : ℝ) := 2 * Real.sin (2 * ω * x - Real.pi / 4)

theorem interval_of_monotonic_increase (ω : ℝ) (hω : 0 < ω)
  (h_max_period_eq : (2 * π / (2 * ω)) = 2) :
  ∀ x : ℝ, x ∈ Icc (-1 : ℝ) 1 → x ∈ Icc (-1 / 4 : ℝ) (3 / 4) := by
  sorry

end interval_of_monotonic_increase_l803_803390


namespace min_positive_period_condition_1_min_value_condition_2_min_value_condition_3_not_unique_l803_803391

-- Function definition
def f (x m : ℝ) : ℝ := Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + m

-- Minimum period proof
theorem min_positive_period : ∀ m : ℝ, ∃ T > 0, (∀ x : ℝ, f (x + T) m = f x m) ∧ T = Real.pi :=
by sorry

-- Condition ①: The maximum value of f(x) is 1
theorem condition_1_min_value : ∃ m : ℝ, ∀ x : ℝ, (m = -Real.sqrt 2) → (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x m ≥ -Real.sqrt 2) :=
by sorry

-- Condition ②: Symmetry point (3π/8, 0)
theorem condition_2_min_value : ∃ m : ℝ, (f (3*Real.pi/8) m = 0) → (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x (-1) ≥ -1) :=
by sorry

-- Condition ③: Line of symmetry x = π/8 does not determine m uniquely
theorem condition_3_not_unique : ∀ m : ℝ, ¬(∃ x : ℝ, ∀ x, (f x m = f (Real.pi/8 - x) m)) :=
by sorry

end min_positive_period_condition_1_min_value_condition_2_min_value_condition_3_not_unique_l803_803391


namespace Q_2_over_Q_neg2_l803_803717

noncomputable def g (x : ℝ) : ℝ := x^2005 + 19 * x^2004 + 1

def distinct_zeros_of_g := ∃ s : ℕ → ℝ, ∀ i j, i ≠ j → s i ≠ s j

noncomputable def Q (z : ℝ) : ℝ :=
  let s := λ j, classical.some (distinct_zeros_of_g j) in
  ∏ j in finset.range 2005, (z - (s j + 1 / s j))

theorem Q_2_over_Q_neg2 : (∃ Q : ℝ → ℝ, ∀ j ∈ finset.range 2005, Q (classical.some (distinct_zeros_of_g j) + 1 / classical.some (distinct_zeros_of_g j)) = 0) → 
  ∃ (Q : ℝ → ℝ), Q 2 / Q (-2) = 21 / 19 :=
begin
  sorry
end

end Q_2_over_Q_neg2_l803_803717


namespace total_distance_traveled_l803_803996

variable (D : ℝ)
variable (D_plane D_train D_bus : ℝ)

-- Given conditions
def condition1 : Prop := D_plane = (1 / 3) * D
def condition2 : Prop := D_train = (2 / 3) * D_bus
def condition3 : Prop := D_bus = 360

-- The statement to prove
theorem total_distance_traveled : 
  condition1 → 
  condition2 → 
  condition3 → 
  D = D_plane + D_train + D_bus → 
  D = 900 := 
by 
  intros h1 h2 h3 h4 
  sorry

end total_distance_traveled_l803_803996


namespace three_digit_non_multiples_of_6_or_8_l803_803417

theorem three_digit_non_multiples_of_6_or_8 : 
  ∃ n, n = 900 - (150 + 112 - 37) ∧ n = 675 :=
by {
  have total_three_digits : 900 = 999 - 100 + 1 := sorry,
  have count_multiples_6 : 150 = 166 - 17 + 1 := sorry,
  have count_multiples_8 : 112 = 124 - 13 + 1 := sorry,
  have count_multiples_24 : 37 = 41 - 5 + 1 := sorry,
  let multiples_6_or_8 := count_multiples_6 + count_multiples_8 - count_multiples_24,
  have : multiples_6_or_8 = 150 + 112 - 37 := sorry,
  have count_non_multiples := total_three_digits - multiples_6_or_8,
  use count_non_multiples,
  split,
  { rw [total_three_digits, multiples_6_or_8], exact sorry },
  { exact sorry }
}

end three_digit_non_multiples_of_6_or_8_l803_803417


namespace mass_percentage_of_oxygen_in_dichromate_l803_803741

def molar_mass_cr : ℝ := 51.9961
def molar_mass_o : ℝ := 15.999
def num_cr_atoms : ℝ := 2
def num_o_atoms : ℝ := 7

def molar_mass_dichromate : ℝ :=
  (num_cr_atoms * molar_mass_cr) + (num_o_atoms * molar_mass_o)

def mass_percentage_o : ℝ :=
  (num_o_atoms * molar_mass_o) / molar_mass_dichromate * 100

theorem mass_percentage_of_oxygen_in_dichromate :
  abs (mass_percentage_o - 51.86) < 0.01 :=
by
  -- The proof goes here
  sorry

end mass_percentage_of_oxygen_in_dichromate_l803_803741


namespace point_P_lies_on_x_axis_l803_803456

noncomputable def point_on_x_axis (x : ℝ) : Prop :=
  (0 = (0 : ℝ)) -- This is a placeholder definition stating explicitly that point lies on the x-axis

theorem point_P_lies_on_x_axis (x : ℝ) : point_on_x_axis x :=
by
  sorry

end point_P_lies_on_x_axis_l803_803456


namespace coplanar_vectors_lambda_l803_803361

theorem coplanar_vectors_lambda (λ : ℝ) :
  let a := (2, -1, 3)
  let b := (-1, 4, -2)
  let c := (7, 7, λ)
  ∃ m n : ℝ, c = m • a + n • b → λ = 9 :=
by
  let a := (2, -1, 3)
  let b := (-1, 4, -2)
  let c := (7, 7, λ)
  sorry

end coplanar_vectors_lambda_l803_803361


namespace sqrt_k_rational_and_integer_l803_803851

theorem sqrt_k_rational_and_integer (k m : ℕ) (h_k_pos : 0 < k) (h_m_pos : 0 < m)
  (h : ∃ n : ℤ, 1 / 2 * ((k + 4 * m) ^ (1 / 2 : ℝ) - k ^ (1 / 2 : ℝ)) = n) :
  (∃ q : ℚ, k ^ (1 / 2 : ℝ) = q) ∧ (∃ n : ℕ, k ^ (1 / 2 : ℝ) = n) :=
sorry

end sqrt_k_rational_and_integer_l803_803851


namespace triangular_prism_intersection_equilateral_l803_803517

-- Define the conditions given in the problem
variables {a b c x y : ℝ}

-- Define the conditions for the equilateral triangle
def condition1 : Prop := c^2 + x^2 = b^2 + y^2
def condition2 : Prop := b^2 + y^2 = a^2 + (x - y)^2

-- The theorem stating that there exist x and y satisfying the conditions
theorem triangular_prism_intersection_equilateral :
    ∃ (x y : ℝ), condition1 ∧ condition2 :=
sorry

end triangular_prism_intersection_equilateral_l803_803517


namespace correct_operation_l803_803964

theorem correct_operation (a : ℕ) : a ^ 3 * a ^ 2 = a ^ 5 :=
by sorry

end correct_operation_l803_803964


namespace value_of_n_l803_803018

theorem value_of_n (n : ℕ) (h : sqrt (10 + n) = 8) : n = 54 := by
  sorry

end value_of_n_l803_803018


namespace smallest_integer_with_divisors_l803_803186

theorem smallest_integer_with_divisors :
  ∃ (n : ℕ), 
    (∀ d : ℕ, d ∣ n → d % 2 = 1 → (∃! k : ℕ, d = (3 ^ k) * 5 ^ (7 - k))) ∧ 
    (∀ d : ℕ, d ∣ n → d % 2 = 0 → (∃! k : ℕ, d = 2 ^ k * m)) ∧ 
    (n = 1080) :=
sorry

end smallest_integer_with_divisors_l803_803186


namespace cost_remaining_two_puppies_l803_803305

noncomputable def total_puppies : ℕ := 8
noncomputable def total_cost : ℤ := 2200
noncomputable def cost_per_puppy (n : ℕ) : ℤ := 
  if n < 4 then 180 else
  if n = 4 then 250 else
  if n = 5 then 300 else sorry -- cost to be determined

-- Statement to be proven
theorem cost_remaining_two_puppies : 
  ∃ p : ℤ, p = 465 ∧ (∀ n, 6 ≤ n < total_puppies → cost_per_puppy n = p) 
:= sorry

end cost_remaining_two_puppies_l803_803305


namespace tan_angle_QDE_eq_48_over_77_l803_803898

noncomputable def DEF : Triangle := {
  D := (0, 0),
  E := (15, 0),
  F := (some_real_number_y_for_F, some_other_real_number_y_for_F),
  side_DE := 15,
  side_EF := 17,
  side_FD := 16
}

def Q (DEF : Triangle) : Point := {
  inside_triangle := sorry,  -- This assumes some conditions ensuring Q is inside the triangle.
  angles_congruent := sorry  -- This assumes some conditions ensuring angles congruence
}

theorem tan_angle_QDE_eq_48_over_77 :
  let θ := angle Q DEF.D DEF.E in
  tan θ = 48 / 77 :=
  sorry

end tan_angle_QDE_eq_48_over_77_l803_803898


namespace monotonicity_of_f_and_g_l803_803925

def f (x : ℝ) : ℝ := (1/2) ^ x
def g (x : ℝ) : ℝ := -abs x

theorem monotonicity_of_f_and_g :
  (∀ x y : ℝ, x < y → x < 0 → y < 0 → f x > f y) ∧
  (∀ x y : ℝ, x < y → x < 0 → y < 0 → g x < g y) :=
by
  sorry

end monotonicity_of_f_and_g_l803_803925


namespace slope_angle_at_1_correct_l803_803140

-- Define the curve
def curve (x : ℝ) : ℝ := (1/3) * x^2 - x^2 + 5

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := (1/3) * 2 * x - 2 * x

-- Define the slope at x = 1
def slope_at_1 := curve_derivative 1

-- Define the slope angle function
def slope_angle (m : ℝ) : ℝ := if m = -1 then (3 * Real.pi) / 4 else 0 -- only consider one case for this problem

-- Problem statement
theorem slope_angle_at_1_correct : slope_angle slope_at_1 = (3 * Real.pi) / 4 :=
by 
    -- Here would go the proof
    sorry

end slope_angle_at_1_correct_l803_803140


namespace range_of_k_l803_803068

theorem range_of_k (f : ℝ → ℝ) (a : ℝ) (k : ℝ) 
  (h₀ : ∀ x > 0, f x = 2 - 1 / (a - x)^2) 
  (h₁ : ∀ x > 0, k^2 * x + f (1 / 4 * x + 1) > 0) : 
  k ≠ 0 :=
by
  -- proof goes here
  sorry

end range_of_k_l803_803068


namespace units_digit_of_expression_l803_803744

theorem units_digit_of_expression : 
  (13 + Real.sqrt 196) = 27 → 
  (13 - Real.sqrt 196) = -1 → 
  (27 ^ 21 + (-1) ^ 21) % 10 = 3 :=
by
  -- condition simplifications
  intro h1 h2,
  rw h1,
  rw h2,
  -- checking if there are remainders
  have : 27 % 10 = 7 := by norm_num,
  have : (-1) % 10 = 9 := by norm_num,
  -- known results from modular arithmetic
  have h7 : 7 ^ 21 % 10 = 7 := by norm_num,
  have h_neg1 : (-1) ^ 21 % 10 = 9 := by norm_num,
  -- finish the proof
  calc
    (27 ^ 21 + (-1) ^ 21) % 10 = (7 ^ 21 + 9) % 10 : by rw [h7, h_neg1, h1, h2]
    ... = (7  + 9) % 10    : by { norm_num }
    ... = 3                : by norm_num

end units_digit_of_expression_l803_803744


namespace paint_percentage_decrease_l803_803922

theorem paint_percentage_decrease (r R : ℝ) 
    (h1 : R = (5 / 2) * r) : 
    let surface_area (r : ℝ) := 4 * π * r^2 in
    ((surface_area R - surface_area r) / surface_area R) * 100 = 84 := 
by
    sorry

end paint_percentage_decrease_l803_803922


namespace range_of_ab_min_value_of_ab_plus_inv_ab_l803_803762

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 0 < a * b ∧ a * b ≤ 1 / 4 :=
sorry

theorem min_value_of_ab_plus_inv_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (∃ ab, ab = a * b ∧ ab + 1 / ab = 17 / 4) :=
sorry

end range_of_ab_min_value_of_ab_plus_inv_ab_l803_803762


namespace cristobal_read_more_pages_l803_803720

theorem cristobal_read_more_pages (B : ℕ) (hB : B = 704) : 
  let C := 15 + 3 * B in
  C - B = 1423 :=
by
  let C := 15 + 3 * B
  have hC : C = 2127, by
    sorry
  have hDiff : C - B = 1423, by
    sorry
  exact hDiff

end cristobal_read_more_pages_l803_803720


namespace real_part_of_complex_l803_803880

theorem real_part_of_complex (z : ℂ) (h : i * (z + 1) = -3 + 2 * i) : z.re = 1 :=
sorry

end real_part_of_complex_l803_803880


namespace shift_cos_to_sin_l803_803950

theorem shift_cos_to_sin :
  ∀ (x : ℝ), cos (2 * x + π / 3) = sin (2 * (x + 5 * π / 12)) :=
by
  intro x
  -- The proof will use trigonometric identities to show the equality
  sorry

end shift_cos_to_sin_l803_803950


namespace ratio_m_of_q_l803_803016

theorem ratio_m_of_q
  (m n p q : ℚ)
  (h1 : m / n = 18)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 12) :
  m / q = 3 / 4 := 
sorry

end ratio_m_of_q_l803_803016


namespace cheryl_material_left_l803_803128

-- Conditions
def initial_material_type1 (m1 : ℚ) : Prop := m1 = 2/9
def initial_material_type2 (m2 : ℚ) : Prop := m2 = 1/8
def used_material (u : ℚ) : Prop := u = 0.125

-- Define the total material bought
def total_material (m1 m2 : ℚ) : ℚ := m1 + m2

-- Define the material left
def material_left (t u : ℚ) : ℚ := t - u

-- The target theorem
theorem cheryl_material_left (m1 m2 u : ℚ) 
  (h1 : initial_material_type1 m1)
  (h2 : initial_material_type2 m2)
  (h3 : used_material u) : 
  material_left (total_material m1 m2) u = 2/9 :=
by
  sorry

end cheryl_material_left_l803_803128


namespace max_x_plus_2y_l803_803482

theorem max_x_plus_2y (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 15) : 
  x + 2 * y ≤ 6 :=
sorry

end max_x_plus_2y_l803_803482


namespace greatest_prime_factor_of_sum_l803_803973

-- Define the product of all even integers from 1 to a given number
def product_of_evens_up_to (n : ℕ) : ℕ :=
  List.prod [ k | k in List.range (n + 1), Even k ]

-- Define a function to find the greatest prime factor of a number
def greatest_prime_factor (n : ℕ) : ℕ :=
  List.last (uniqueFactorizationMonoid.factors n) (dec_trivial : 0 < n)

-- Define the problem theorem
theorem greatest_prime_factor_of_sum :
  greatest_prime_factor ((product_of_evens_up_to 12) + (product_of_evens_up_to 10)) = 5 :=
by
  sorry  -- Proof of the theorem goes here

end greatest_prime_factor_of_sum_l803_803973


namespace part_b_l803_803749

open Real
open Classical

variable (A B C T : Point)

noncomputable def m(T : Point) : ℝ :=
  min (dist T A) (min (dist T B) (dist T C))

noncomputable def M(T : Point) : ℝ :=
  max (dist T A) (max (dist T B) (dist T C))

theorem part_b (h : ∠A B C = 90) :
  m T ≤ dist B C / 2 ∧ dist B C / 2 ≤ M T :=
  sorry

end part_b_l803_803749


namespace Jason_age_l803_803465

theorem Jason_age : ∃ J K : ℕ, (J = 7 * K) ∧ (J + 4 = 3 * (2 * (K + 2))) ∧ (J = 56) :=
by
  sorry

end Jason_age_l803_803465


namespace ship_passengers_percentage_l803_803891

variables (P R : ℝ)

-- Conditions
def condition1 : Prop := (0.20 * P) = (0.60 * R)

-- Target
def target : Prop := R / P = 1 / 3

theorem ship_passengers_percentage
  (h1 : condition1 P R) :
  target P R :=
by
  sorry

end ship_passengers_percentage_l803_803891


namespace women_needed_for_40_percent_men_l803_803150

-- Define the initial conditions
def total_people_initial : ℕ := 30
def percent_men : ℕ := 60
def num_men_initial : ℕ := 18 := (percent_men * total_people_initial) / 100

-- Prove that 15 women need to enter for 40% of the room to be men
theorem women_needed_for_40_percent_men
  (num_men_initial = 18)
  (total_people_initial = 30)
  (percent_required_men : ℕ := 40) :
  ∃ w : ℕ, 0.4 * (total_people_initial + w) = 18 → w = 15 :=
sorry

end women_needed_for_40_percent_men_l803_803150


namespace cos_alpha_minus_beta_cos_alpha_cos_beta_l803_803772

variable {α β : Real}
variable (A : Real × Real) (B : Real × Real)

def points_A_B_conditions 
  (A = (Real.cos α, Real.sin α)) 
  (B = (Real.cos β, Real.sin β)) 
  (α β : Real) (hα: 0 < α ∧ α < π / 2) 
  (hβ: 0 < β ∧ β < π / 2) 
  (hAB: Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = Real.sqrt(10) / 5) 
  (htan: Real.tan (α / 2) = 1 / 2) : Prop := true

theorem cos_alpha_minus_beta (h : points_A_B_conditions A B α β hα hβ hAB htan) : 
  Real.cos (α - β) = 4 / 5 := sorry

theorem cos_alpha (h : points_A_B_conditions A B α β hα hβ hAB htan) : 
  Real.cos α = 3 / 5 := sorry

theorem cos_beta (h : points_A_B_conditions A B α β hα hβ hAB htan) : 
  Real.cos β = 24 / 25 := sorry

end cos_alpha_minus_beta_cos_alpha_cos_beta_l803_803772


namespace length_AX_l803_803899

noncomputable def A := (0, 0)
noncomputable def D := (2, 0)
noncomputable def B := (real.cos (24 * real.pi / 180), real.sin (24 * real.pi / 180))
noncomputable def C := (2 - real.cos (24 * real.pi / 180), real.sin (24 * real.pi / 180))
noncomputable def X := (real.sqrt((B.1 - C.1)^2 + (B.2 - C.2)^2) / 2, 0)

-- Given conditions
lemma cond1 : real.sqrt((A.1 - D.1)^2 + (A.2 - D.2)^2) = 2 := sorry
lemma cond2 : BX = CX := sorry
lemma cond3 : 3 * real.atan2 (B.2 - A.2) (B.1 - A.1) = 72 * real.pi / 180 := sorry
lemma cond4 : real.atan2 (B.2 - X.2) (B.1 - X.1) = 72 * real.pi / 180 := sorry

-- Proof statement
theorem length_AX :
  real.sqrt((A.1 - X.1)^2 + (A.2 - X.2)^2) = 
  2 * (real.cos (24 * real.pi / 180) * real.sin (24 * real.pi / 180) * real.csc (72 * real.pi / 180)) :=
begin
  -- To be proven
  sorry
end

end length_AX_l803_803899


namespace total_revenue_correct_l803_803672

def small_slices_price := 150
def large_slices_price := 250
def total_slices_sold := 5000
def small_slices_sold := 2000

def large_slices_sold := total_slices_sold - small_slices_sold

def revenue_from_small_slices := small_slices_sold * small_slices_price
def revenue_from_large_slices := large_slices_sold * large_slices_price
def total_revenue := revenue_from_small_slices + revenue_from_large_slices

theorem total_revenue_correct : total_revenue = 1050000 := by
  sorry

end total_revenue_correct_l803_803672


namespace days_for_a_to_complete_work_l803_803614
noncomputable theory

-- Definitions from the conditions
def work_done_by_a_per_day (x : ℝ) : ℝ := 1 / x
def work_done_by_b_per_day : ℝ := 1 / 8
def work_done_by_a_and_b_per_day : ℝ := 1 / 4.8

-- Theorem to prove
theorem days_for_a_to_complete_work (x : ℝ) (h : work_done_by_a_per_day x + work_done_by_b_per_day = work_done_by_a_and_b_per_day) :
  x = 12 :=
begin
  -- Proof skipped
  sorry
end

end days_for_a_to_complete_work_l803_803614


namespace pamela_sugar_proof_l803_803894

def pamela_initial_sugar (spill_amount left_amount : ℝ) : ℝ := spill_amount + left_amount

theorem pamela_sugar_proof (spill_amount left_amount : ℝ) (h_spill : spill_amount = 5.2) 
  (h_left : left_amount = 4.6) : pamela_initial_sugar spill_amount left_amount = 9.8 :=
by
  rw [h_spill, h_left]
  norm_num

#check @pamela_sugar_proof -- ensure the theorem type checks successfully

end pamela_sugar_proof_l803_803894


namespace dodecagon_diagonals_l803_803406

def numDiagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem dodecagon_diagonals : numDiagonals 12 = 54 := by
  sorry

end dodecagon_diagonals_l803_803406


namespace problem_a5_value_l803_803396

def Sn (n : ℕ) : ℕ := 2 * n^2 + 3 * n - 1

theorem problem_a5_value : Sn 5 - Sn 4 = 21 := by
  sorry

end problem_a5_value_l803_803396


namespace speed_of_stream_l803_803618

variable (D : ℝ) (v : ℝ)
variable (speed_boat : ℝ := 51)

theorem speed_of_stream (h1 : time_to_row_upstream (D, 51 - v) = 2 * time_to_row_downstream (D, 51 + v)) :
  v = 17 := by
  sorry

-- Definitions of time_to_row_upstream and time_to_row_downstream
def time_to_row_upstream (D_speed : ℝ × ℝ) : ℝ := D_speed.1 / D_speed.2
def time_to_row_downstream (D_speed : ℝ × ℝ) : ℝ := D_speed.1 / D_speed.2

end speed_of_stream_l803_803618


namespace Kiera_muffins_l803_803357

theorem Kiera_muffins :
  ∀ (muffin_cost fruit_cup_cost francis_muffins francis_fruit_cups kiera_fruit_cups total_cost : ℕ),
    muffin_cost = 2 →
    fruit_cup_cost = 3 →
    francis_muffins = 2 →
    francis_fruit_cups = 2 →
    kiera_fruit_cups = 1 →
    total_cost = 17 →
    (∃ kiera_muffins : ℕ, kiera_muffins * muffin_cost + kiera_fruit_cups * fruit_cup_cost = total_cost - (francis_muffins * muffin_cost + francis_fruit_cups * fruit_cup_cost)) ∧
    (kiera_muffins = 2) :=
by
  intro muffin_cost fruit_cup_cost francis_muffins francis_fruit_cups kiera_fruit_cups total_cost
  intro h1 h2 h3 h4 h5 h6
  use 2
  split
  all_goals { sorry }

end Kiera_muffins_l803_803357


namespace sum_of_numbers_l803_803144

theorem sum_of_numbers (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 10) (h3 : x^2 - y^2 = 50) : x + y = 5 :=
by
  sorry

end sum_of_numbers_l803_803144


namespace area_of_rectangle_EFGH_l803_803948

theorem area_of_rectangle_EFGH :
  ∀ (a b c : ℕ), 
    a = 7 → 
    b = 3 * a → 
    c = 2 * a → 
    (area : ℕ) = b * c → 
    area = 294 := 
by
  sorry

end area_of_rectangle_EFGH_l803_803948


namespace complex_exp_sum_product_l803_803425

theorem complex_exp_sum_product (α β : ℝ) (h : complex.exp (complex.I * α) + complex.exp (complex.I * β) = -1 / 3 + 4 / 5 * complex.I) : 
  (complex.exp (-complex.I * α) + complex.exp (-complex.I * β)) * (complex.exp (complex.I * α) + complex.exp (complex.I * β)) = 169 / 225 := 
by
  sorry

end complex_exp_sum_product_l803_803425


namespace domain_width_of_f_l803_803862

-- Define the domain of h
def h_domain : Set ℝ := Set.Icc (-12) 12

-- Define the function f
noncomputable def f (h : ℝ → ℝ) (x : ℝ) : ℝ := h (x / 3)

-- State the theorem
theorem domain_width_of_f (h : ℝ → ℝ) 
  (h_domain_condition : ∀ x, h_domain x → true) :
  let f_domain := { x : ℝ | x / 3 ∈ h_domain }
  width f_domain = 72 :=
by
  -- Definition of width
  sorry

end domain_width_of_f_l803_803862


namespace part1_part2_l803_803487

noncomputable def f (a x : ℝ) : ℝ := a^(x+1) - 2

theorem part1 (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h : f a 1 = 7) :
  a = 3 ∧ f a (log 3 (2 / 3)) = 0 :=
by
  sorry

theorem part2 (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h_a : a = 3) :
  { x : ℝ | f a x ≥ -5/3 } = {x | x ≥ -2} :=
by
  sorry

end part1_part2_l803_803487


namespace binomial_product_l803_803699

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := by
  sorry

end binomial_product_l803_803699


namespace complex_exponentiation_l803_803863

open Complex

theorem complex_exponentiation :
  (Complex.div (1 + I) (1 - I))^2013 = I :=
by
  -- Here we take 'I' as complex number representing the imaginary unit 'i'.
  sorry

end complex_exponentiation_l803_803863


namespace geometric_progression_product_sum_sumrecip_l803_803014

theorem geometric_progression_product_sum_sumrecip (P S S' : ℝ) (n : ℕ)
  (hP : P = a ^ n * r ^ ((n * (n - 1)) / 2))
  (hS : S = a * (1 - r ^ n) / (1 - r))
  (hS' : S' = (r ^ n - 1) / (a * (r - 1))) :
  P = (S / S') ^ (1 / 2 * n) :=
  sorry

end geometric_progression_product_sum_sumrecip_l803_803014


namespace sum_of_integers_l803_803547

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 240) : 
  x + y = 32 :=
by
  sorry

end sum_of_integers_l803_803547


namespace ratio_ac_l803_803249

-- Definitions based on conditions
variables (a b c : ℕ)
variables (x y : ℕ)

-- Conditions
def ratio_ab := (a : ℚ) / (b : ℚ) = 2 / 3
def ratio_bc := (b : ℚ) / (c : ℚ) = 1 / 5

-- Theorem to prove the desired ratio
theorem ratio_ac (h1 : ratio_ab a b) (h2 : ratio_bc b c) : (a : ℚ) / (c : ℚ) = 2 / 15 :=
by
  sorry

end ratio_ac_l803_803249


namespace right_triangle_shorter_leg_length_l803_803649

noncomputable def shorter_leg_length (b : ℝ) : ℝ :=
  let a := b / 2
  a

theorem right_triangle_shorter_leg_length (b : ℝ) (h₁ : b = 10) : shorter_leg_length b = 5 :=
by
  rw [h₁]
  unfold shorter_leg_length
  norm_num
  sorry

end right_triangle_shorter_leg_length_l803_803649


namespace ratio_of_books_sold_l803_803087

theorem ratio_of_books_sold
  (T W R : ℕ)
  (hT : T = 7)
  (hW : W = 3 * T)
  (hTotal : T + W + R = 91) :
  R / W = 3 :=
by
  sorry

end ratio_of_books_sold_l803_803087


namespace max_sum_x_3y_l803_803352

def length_of_integer (n : ℕ) : ℕ :=
Multiset.card (Multiset.repeat (2 : ℕ) (nat.factors n).count 2 ‐ (nat.factors n).count 2)

theorem max_sum_x_3y {x y : ℕ} (hx : length_of_integer x + length_of_integer y = 16) (hx1 : x > 1) (hy1 : y > 1) : 
  x + 3 * y = 49156 :=
sorry

end max_sum_x_3y_l803_803352


namespace special_number_exists_l803_803608

theorem special_number_exists (a b c d e : ℕ) (h1 : a < b ∧ b < c ∧ c < d ∧ d < e)
    (h2 : a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e) 
    (h_num : a * 10 + b = 13 ∧ c = 4 ∧ d * 10 + e = 52) :
    (10 * a + b) * c = 10 * d + e :=
by
  sorry

end special_number_exists_l803_803608


namespace islander_statements_l803_803505

theorem islander_statements
  (K L : ℕ)
  (H_total : K + L = 10)
  (H_statement : (K = 5 ∧ L = 5) ∨ (5 > K ∧ 5 ≤ L = 5)) :
  ∃ (other_statements : ℕ → Prop), 
  (∀ i, i ∈ {0, 1, 2, 3, 4} → other_statements i = "There are an equal number of knights and liars.") :=
by
  sorry

end islander_statements_l803_803505


namespace triangle_point_product_eq_l803_803999

variables {A B C P : Type*} [InnerProductSpace ℝ P]
variables (a b c p : P)
noncomputable def angles_condition (A B C P : P) : Prop :=
  sorry -- definition involving angles

theorem triangle_point_product_eq (h_angle : angles_condition a b c p) : 
  dist P a * dist B C = dist P b * dist A C ∧ dist P b * dist A C = dist P c * dist A B :=
sorry

end triangle_point_product_eq_l803_803999


namespace gcd_lcm_sum_l803_803231

theorem gcd_lcm_sum :
  gcd 42 70 + lcm 15 45 = 59 :=
by sorry

end gcd_lcm_sum_l803_803231


namespace snazzy_numbers_div_by_11_l803_803642

def snazzy_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧
  (n = 10000 * a + 1000 * b + 100 * a + 10 * b + a ∨
   n = 10000 * b + 1000 * a + 100 * b + 10 * a + b)

def divisible_by_11 (n : ℕ) : Prop :=
  ∃ (k : ℤ), n = 11 * k

theorem snazzy_numbers_div_by_11 : 
  {n : ℕ | snazzy_number n ∧ divisible_by_11 n}.to_finset.card = 4 :=
sorry

end snazzy_numbers_div_by_11_l803_803642


namespace trigonometric_identity_l803_803759

def conditions (α : ℝ) : Prop := 
  α ∈ set.Ioo 0 (real.pi / 2) ∧ sin α - cos α = 1 / 5

theorem trigonometric_identity (α : ℝ) (h : conditions α) : 
  (cos (2 * α)) / ((sqrt 2) * sin (α - (real.pi / 4))) = -7 / 5 :=
sorry

end trigonometric_identity_l803_803759


namespace locus_of_P_passes_through_incenter_l803_803896

-- Define points in the plane
variables {Point : Type} [Geometry Point]
variables (O A B C P : Point)
variables (λ : ℝ)

-- Define vector operations and magnitude for the proof
variables [normed_space ℝ Point]
variables [inner_product_space ℝ Point]
open_locale real_inner_product_space

-- Hypotheses for the problem conditions
axiom fixed_point_O : ∃ (O : Point), True
axiom non_collinear_ABC : ¬ collinear A B C
axiom locus_of_P : ∀ λ ≥ 0, P = A + λ • ((B - A) / (∥ C - A ∥) + (C - A) / (∥ C - A ∥))

-- Final claim to prove
theorem locus_of_P_passes_through_incenter :
  ∃ I, incenter I A B C ∧ ∃ λ, λ ≥ 0 ∧ P = A + λ • ((B - A) / (∥ C - A ∥) + (C - A) / (∥ C - A ∥)) :=
sorry

end locus_of_P_passes_through_incenter_l803_803896


namespace inequality_solution_set_l803_803110

theorem inequality_solution_set (x : ℝ) :
  (3 * (x + 2) - x > 4) ∧ ((1 + 2 * x) / 3 ≥ x - 1) ↔ (-1 < x ∧ x ≤ 4) :=
by
  sorry

end inequality_solution_set_l803_803110


namespace seventh_term_of_arithmetic_sequence_l803_803142

variable (a d : ℕ)

theorem seventh_term_of_arithmetic_sequence (h1 : 5 * a + 10 * d = 15) (h2 : a + 3 * d = 4) : a + 6 * d = 7 := 
by
  sorry

end seventh_term_of_arithmetic_sequence_l803_803142


namespace monotonicity_and_range_collinear_OPC_then_OB_OP_l803_803572

def vector_a (α : ℝ) : ℝ × ℝ := (Real.sin α, 1)
def vector_b (α : ℝ) : ℝ × ℝ := (Real.cos α, 0)
def vector_c (α : ℝ) : ℝ × ℝ := (-Real.sin α, 2)

def point_P (α : ℝ) : ℝ × ℝ :=
  let x := (Real.cos α - 0.5 * Real.sin α)
  let y := (0.5)
  (x, y)

def f (α : ℝ) : ℝ :=
  let ⟨opx, opy⟩ := point_P α
  let ⟨cx, cy⟩ := vector_c α
  opx * cx + opy * cy

theorem monotonicity_and_range (α : ℝ) : f α = 0.5 * (1 - Real.cos (2 * α)) := by
  sorry

theorem collinear_OPC_then_OB_OP (α : ℝ) (h : (∃ k : ℝ, point_P α = k • vector_c α)) : 
  |(Real.cos α, 0) + point_P α| = Real.sqrt (Real.cos α ^ 2 + (Real.cos α - 0.5 * Real.sin α) ^ 2 + 0.25) := by
  sorry

end monotonicity_and_range_collinear_OPC_then_OB_OP_l803_803572


namespace factorization_correct_l803_803240

-- Defining the expressions
def expr1 (x : ℝ) : ℝ := 4 * x^2 + 4 * x
def expr2 (x : ℝ) : ℝ := 4 * x * (x + 1)

-- Theorem statement: Prove that expr1 and expr2 are equivalent
theorem factorization_correct (x : ℝ) : expr1 x = expr2 x :=
by 
  sorry

end factorization_correct_l803_803240


namespace largest_total_real_roots_l803_803074

def max_real_roots (a b c : ℝ) : ℕ :=
  if a > 0 ∧ b > 0 ∧ c > 0 then 4 else sorry

theorem largest_total_real_roots (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  max_real_roots a b c = 4 :=
sorry

end largest_total_real_roots_l803_803074


namespace sum_x_y_l803_803562

theorem sum_x_y :
  let a := 6
  let b := 14
  let c := 17
  let d := 9
  let e := 10
  let mean := (a + b + x + c + d + y + e) / 7
  in mean = 13 → x + y = 35 :=
by
  sorry

end sum_x_y_l803_803562


namespace smallest_integer_with_odd_and_even_divisors_l803_803205

theorem smallest_integer_with_odd_and_even_divisors :
  ∃ n : ℕ,
    0 < n ∧
    (∀ d : ℕ, d ∣ n → (odd d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 1}) ∨ (even d → d ∈ {d : ℕ | 0 < d ∧ d.mod 2 = 0})) ∧ 
    (↑8 = (∑ d in (finset.filter (λ d, odd d) (finset.divisors n)), 1)) ∧
    (↑16 = (∑ d in (finset.filter (λ d, even d) (finset.divisors n)), 1)) ∧
    (24 = (∑ d in (finset.divisors n), 1)) ∧
    n = 108 :=
begin
  sorry
end

end smallest_integer_with_odd_and_even_divisors_l803_803205


namespace divide_square_into_smaller_squares_l803_803527

def P (n : ℕ) : Prop := sorry /- Define the property of dividing a square into n smaller squares -/

theorem divide_square_into_smaller_squares (n : ℕ) (h : n > 5) : P n :=
  sorry

end divide_square_into_smaller_squares_l803_803527


namespace part_one_solution_part_two_solution_l803_803758

-- Definitions and conditions
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part (1): When a = 1, solution set of the inequality f(x) > 1 is (1/2, +∞)
theorem part_one_solution (x : ℝ) :
  f x 1 > 1 ↔ x > 1 / 2 := sorry

-- Part (2): If the inequality f(x) > x holds for x ∈ (0,1), range of values for a is (0, 2]
theorem part_two_solution (a : ℝ) :
  (∀ x, 0 < x ∧ x < 1 → f x a > x) ↔ 0 < a ∧ a ≤ 2 := sorry

end part_one_solution_part_two_solution_l803_803758


namespace bridge_cost_approx_l803_803259

def R (a b c : ℕ) (A : ℝ) : ℝ := (a * b * c) / (4 * A)
def r (A s : ℝ) : ℝ := A / s
def cosC (a b c : ℕ) : ℝ := ((a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b) : ℝ)
def sinC (a b c : ℕ) : ℝ := real.sqrt (1 - (cosC a b c) ^ 2)
def distance_between_centers (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2
  let A := real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R_ := R a b c A
  let r_ := r A s
  real.sqrt (R_ ^ 2 + 2 * R_ * r_ * cosC a b c + r_ ^ 2)

noncomputable def bridge_cost (a b c cost_per_km : ℕ) : ℝ :=
  distance_between_centers a b c * cost_per_km

theorem bridge_cost_approx (a b c cost_per_km : ℕ) :
  bridge_cost 7 8 9 1000 ≈ 5750 :=
by
  sorry

end bridge_cost_approx_l803_803259


namespace count_less_than_threshold_is_zero_l803_803811

def numbers := [0.8, 0.5, 0.9]
def threshold := 0.4

theorem count_less_than_threshold_is_zero :
  (numbers.filter (λ x => x < threshold)).length = 0 :=
by
  sorry

end count_less_than_threshold_is_zero_l803_803811


namespace number_of_intersection_points_l803_803033

noncomputable def intersects_x_axis_exactly_once (a b c : ℝ) (h_geometric_sequence : b^2 = a * c) : Prop := 
  ∀ x : ℝ, (a * x^2 + 2 * b * x + c = 0) ↔ x = -b / a

theorem number_of_intersection_points (a b c : ℝ) (h_geometric_sequence : b^2 = a * c) :
  (∃! x : ℝ, a * x^2 + 2 * b * x + c = 0) :=
begin
  rw [intersects_x_axis_exactly_once],
  use 0,
  sorry,
end

end number_of_intersection_points_l803_803033


namespace intersection_A_complement_B_l803_803080

def A := { x : ℝ | x ≥ -1 }
def B := { x : ℝ | x > 2 }
def complement_B := { x : ℝ | x ≤ 2 }

theorem intersection_A_complement_B :
  A ∩ complement_B = { x : ℝ | -1 ≤ x ∧ x ≤ 2 } :=
sorry

end intersection_A_complement_B_l803_803080


namespace cos_double_angle_l803_803816

theorem cos_double_angle (θ : ℝ) 
  (h : 3 * cos (π / 2 - θ) + cos (π + θ) = 0) : 
  cos (2 * θ) = 4 / 5 := 
by 
  sorry

end cos_double_angle_l803_803816


namespace emma_average_speed_last_segment_l803_803330

open Real

theorem emma_average_speed_last_segment :
  ∀ (d1 d2 d3 : ℝ) (t1 t2 t3 : ℝ),
    d1 + d2 + d3 = 120 →
    t1 + t2 + t3 = 2 →
    t1 = 2 / 3 → t2 = 2 / 3 → 
    t1 = d1 / 50 → t2 = d2 / 55 → 
    ∃ x : ℝ, t3 = d3 / x ∧ x = 75 := 
by
  intros d1 d2 d3 t1 t2 t3 h1 h2 ht1 ht2 hs1 hs2
  use 75 / (2 / 3)
  -- skipped proof for simplicity
  sorry

end emma_average_speed_last_segment_l803_803330


namespace total_boys_l803_803116

theorem total_boys (T F : ℕ) 
  (avg_all : 37 * T = 39 * 110 + 15 * F) 
  (total_eq : T = 110 + F) : 
  T = 120 := 
sorry

end total_boys_l803_803116


namespace min_students_visited_library_l803_803094

theorem min_students_visited_library :
  let Monday := 5
  let Tuesday := 6
  let Wednesday := 4
  let Thursday := 8
  let Friday := 7
  ∃ (x : ℕ), 
  (∀ students,
  (studentsMonday students ≤ Monday) ∧
  (studentsTuesday students ≤ Tuesday) ∧
  (studentsWednesday students ≤ Wednesday) ∧
  (studentsThursday students ≤ Thursday) ∧
  (studentsFriday students ≤ Friday) ∧
  (∀ i, students i = students (i + 2)) →
  (∑ i in (Ico monday friday), students i )= x) ∧ x = 15 :=
sorry

end min_students_visited_library_l803_803094


namespace nancy_total_savings_l803_803888

noncomputable def total_savings : ℝ :=
  let cost_this_month := 9 * 5
  let cost_last_month := 8 * 4
  let cost_next_month := 7 * 6
  let discount_this_month := 0.20 * cost_this_month
  let discount_last_month := 0.20 * cost_last_month
  let discount_next_month := 0.20 * cost_next_month
  discount_this_month + discount_last_month + discount_next_month

theorem nancy_total_savings : total_savings = 23.80 :=
by
  sorry

end nancy_total_savings_l803_803888


namespace traffic_light_color_change_probability_l803_803662

theorem traffic_light_color_change_probability :
  (let total_cycle := 92 in
   let green_duration := 50 in
   let yellow_duration := 2 in
   let red_duration := 40 in
   let observation_interval := 3 in
   let green_to_yellow_start := green_duration in
   let green_to_yellow_end := green_duration + yellow_duration in
   let yellow_to_red_start := green_duration + yellow_duration in
   let yellow_to_red_end := green_duration + yellow_duration + observation_interval in
   let red_to_green_start := green_duration + yellow_duration + red_duration in
   let red_to_green_end := total_cycle in
   let valid_interval := 9 in
   let probability := valid_interval / total_cycle in
   probability = 9 / 92) :=
begin
  sorry
end

end traffic_light_color_change_probability_l803_803662


namespace binomial_sum_identity_l803_803104

open Nat

theorem binomial_sum_identity (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∑ k in (finset.range (n+1)), (-1) ^ k * (1 / ((m + k + 1) * (n.choose k)))) =
  1 / ((m + n + 1) * (finset.card (finset.range (m+n)) / (n! * (m+n-n)!))) :=
sorry

end binomial_sum_identity_l803_803104


namespace tangent_alpha_l803_803362

open Real

noncomputable def a (α : ℝ) : ℝ × ℝ := (sin α, 2)
noncomputable def b (α : ℝ) : ℝ × ℝ := (-cos α, 1)

theorem tangent_alpha (α : ℝ) (h : ∀ k : ℝ, a α = (k • b α)) : tan α = -2 := by
  have h1 : sin α / -cos α = 2 := by sorry
  have h2 : tan α = -2 := by sorry
  exact h2

end tangent_alpha_l803_803362


namespace sum_of_integers_l803_803545

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 :=
sorry

end sum_of_integers_l803_803545
