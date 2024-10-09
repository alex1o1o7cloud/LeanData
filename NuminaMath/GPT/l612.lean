import Mathlib

namespace circle_equation_proof_l612_61254

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 2)

-- Define a predicate for the circle being tangent to the y-axis
def tangent_y_axis (center : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r = abs center.1

-- Define the equation of the circle given center and radius
def circle_eqn (center : ℝ × ℝ) (r : ℝ) : Prop :=
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = r^2

-- State the theorem
theorem circle_equation_proof :
  tangent_y_axis circle_center →
  ∃ r, r = 1 ∧ circle_eqn circle_center r :=
sorry

end circle_equation_proof_l612_61254


namespace hose_rate_l612_61235

theorem hose_rate (V : ℝ) (T : ℝ) (r_fixed : ℝ) (total_rate : ℝ) (R : ℝ) :
  V = 15000 ∧ T = 25 ∧ r_fixed = 3 ∧ total_rate = 10 ∧
  (2 * R + 2 * r_fixed = total_rate) → R = 2 :=
by
  -- Given conditions:
  -- Volume V = 15000 gallons
  -- Time T = 25 hours
  -- Rate of fixed hoses r_fixed = 3 gallons per minute each
  -- Total rate of filling the pool total_rate = 10 gallons per minute
  -- Relationship: 2 * rate of first two hoses + 2 * rate of fixed hoses = total rate
  
  sorry

end hose_rate_l612_61235


namespace eggs_per_basket_l612_61208

theorem eggs_per_basket (red_eggs : ℕ) (orange_eggs : ℕ) (min_eggs : ℕ) :
  red_eggs = 30 → orange_eggs = 45 → min_eggs = 5 →
  (∃ k, (30 % k = 0) ∧ (45 % k = 0) ∧ (k ≥ 5) ∧ k = 15) :=
by
  intros h1 h2 h3
  use 15
  sorry

end eggs_per_basket_l612_61208


namespace range_of_a_l612_61242

variable {α : Type*}

def in_interval (x : ℝ) (a b : ℝ) : Prop := a < x ∧ x < b

def A (a : ℝ) : Set ℝ := {-1, 0, a}

def B : Set ℝ := {x : ℝ | in_interval x 0 1}

theorem range_of_a (a : ℝ) (hA_B_nonempty : (A a ∩ B).Nonempty) : 0 < a ∧ a < 1 := 
sorry

end range_of_a_l612_61242


namespace lower_limit_for_x_l612_61250

variable {n : ℝ} {x : ℝ} {y : ℝ}

theorem lower_limit_for_x (h1 : x > n) (h2 : x < 8) (h3 : y > 8) (h4 : y < 13) (h5 : y - x = 7) : x = 2 :=
sorry

end lower_limit_for_x_l612_61250


namespace initial_brownies_l612_61281

theorem initial_brownies (B : ℕ) (eaten_by_father : ℕ) (eaten_by_mooney : ℕ) (new_brownies : ℕ) (total_brownies : ℕ) :
  eaten_by_father = 8 →
  eaten_by_mooney = 4 →
  new_brownies = 24 →
  total_brownies = 36 →
  (B - (eaten_by_father + eaten_by_mooney) + new_brownies = total_brownies) →
  B = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end initial_brownies_l612_61281


namespace ratio_of_horns_l612_61261

def charlie_flutes := 1
def charlie_horns := 2
def charlie_harps := 1

def carli_flutes := 2 * charlie_flutes
def carli_harps := 0

def total_instruments := 7

def charlie_instruments := charlie_flutes + charlie_horns + charlie_harps
def carli_instruments := total_instruments - charlie_instruments

def carli_horns := carli_instruments - carli_flutes

theorem ratio_of_horns : (carli_horns : ℚ) / charlie_horns = 1 / 2 := by
  sorry

end ratio_of_horns_l612_61261


namespace painters_work_days_l612_61217

noncomputable def work_product (n : ℕ) (d : ℚ) : ℚ := n * d

theorem painters_work_days :
  (work_product 5 2 = work_product 4 (2 + 1/2)) :=
by
  sorry

end painters_work_days_l612_61217


namespace perimeter_of_triangle_l612_61252

-- Defining the basic structure of the problem
theorem perimeter_of_triangle (A B C : Type)
  (distance_AB distance_AC distance_BC : ℝ)
  (angle_B : ℝ)
  (h1 : distance_AB = distance_AC)
  (h2 : angle_B = 60)
  (h3 : distance_BC = 4) :
  distance_AB + distance_AC + distance_BC = 12 :=
by 
  sorry

end perimeter_of_triangle_l612_61252


namespace solve_trigonometric_inequality_l612_61244

noncomputable def trigonometric_inequality (x : ℝ) : Prop :=
  x ∈ Set.Ioo 0 (2 * Real.pi) ∧ 2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0

theorem solve_trigonometric_inequality :
  ∀ x, x ∈ Set.Ioo 0 (2 * Real.pi) → (2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0 ↔ x ∈ Set.Icc (Real.pi / 3) (2 * Real.pi / 3)) :=
by
  intros x hx
  sorry

end solve_trigonometric_inequality_l612_61244


namespace opposite_of_neg_6_l612_61296

theorem opposite_of_neg_6 : ∀ (n : ℤ), n = -6 → -n = 6 :=
by
  intro n h
  rw [h]
  sorry

end opposite_of_neg_6_l612_61296


namespace sufficient_not_necessary_l612_61264

theorem sufficient_not_necessary (a b : ℝ) : (a^2 + b^2 ≤ 2) → (-1 ≤ a * b ∧ a * b ≤ 1) ∧ ¬((-1 ≤ a * b ∧ a * b ≤ 1) → a^2 + b^2 ≤ 2) := 
by
  sorry

end sufficient_not_necessary_l612_61264


namespace avg_speed_while_climbing_l612_61289

-- Definitions for conditions
def totalClimbTime : ℝ := 4
def restBreaks : ℝ := 0.5
def descentTime : ℝ := 2
def avgSpeedWholeJourney : ℝ := 1.5
def totalDistance : ℝ := avgSpeedWholeJourney * (totalClimbTime + descentTime)

-- The question: Prove Natasha's average speed while climbing to the top, excluding the rest breaks duration.
theorem avg_speed_while_climbing :
  (totalDistance / 2) / (totalClimbTime - restBreaks) = 1.29 := 
sorry

end avg_speed_while_climbing_l612_61289


namespace valid_parameterizations_l612_61298

-- Define the parameterization as a structure
structure LineParameterization where
  x : ℝ
  y : ℝ
  dx : ℝ
  dy : ℝ

-- Define the line equation
def line_eq (p : ℝ × ℝ) : Prop :=
  p.snd = -(2/3) * p.fst + 4

-- Proving which parameterizations are valid
theorem valid_parameterizations :
  (line_eq (3 + t * 3, 4 + t * (-2)) ∧
   line_eq (0 + t * 1.5, 4 + t * (-1)) ∧
   line_eq (1 + t * (-6), 3.33 + t * 4) ∧
   line_eq (5 + t * 1.5, (2/3) + t * (-1)) ∧
   line_eq (-6 + t * 9, 8 + t * (-6))) = 
  false ∧ true ∧ false ∧ true ∧ false :=
by
  sorry

end valid_parameterizations_l612_61298


namespace trig_identity_proof_l612_61246

theorem trig_identity_proof :
  let sin_95 := Real.cos (Real.pi / 36)
  let sin_65 := Real.cos (5 * Real.pi / 36)
  (Real.sin (Real.pi / 36) * Real.sin (5 * Real.pi / 36) - sin_95 * sin_65) = - (Real.sqrt 3) / 2 :=
by
  let sin_95 := Real.cos (Real.pi / 36)
  let sin_65 := Real.cos (5 * Real.pi / 36)
  sorry

end trig_identity_proof_l612_61246


namespace cost_per_square_meter_l612_61255

theorem cost_per_square_meter 
  (length width height : ℝ) 
  (total_expenditure : ℝ) 
  (hlength : length = 20) 
  (hwidth : width = 15) 
  (hheight : height = 5) 
  (hmoney : total_expenditure = 38000) : 
  58.46 = total_expenditure / (length * width + 2 * length * height + 2 * width * height) :=
by 
  -- Let's assume our definitions and use sorry to skip the proof
  sorry

end cost_per_square_meter_l612_61255


namespace solution_set_of_inequality_l612_61260

theorem solution_set_of_inequality (x : ℝ) : |2 * x - 1| < |x| + 1 ↔ 0 < x ∧ x < 2 :=
by 
  sorry

end solution_set_of_inequality_l612_61260


namespace total_questions_solved_l612_61279

-- Define the number of questions Taeyeon solved in a day and the number of days
def Taeyeon_questions_per_day : ℕ := 16
def Taeyeon_days : ℕ := 7

-- Define the number of questions Yura solved in a day and the number of days
def Yura_questions_per_day : ℕ := 25
def Yura_days : ℕ := 6

-- Define the total number of questions Taeyeon and Yura solved
def Total_questions_Taeyeon : ℕ := Taeyeon_questions_per_day * Taeyeon_days
def Total_questions_Yura : ℕ := Yura_questions_per_day * Yura_days
def Total_questions : ℕ := Total_questions_Taeyeon + Total_questions_Yura

-- Prove that the total number of questions solved by Taeyeon and Yura is 262
theorem total_questions_solved : Total_questions = 262 := by
  sorry

end total_questions_solved_l612_61279


namespace find_number_l612_61216

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 15) : x = 7.5 :=
sorry

end find_number_l612_61216


namespace impossible_to_repaint_white_l612_61209

-- Define the board as a 7x7 grid 
def boardSize : ℕ := 7

-- Define the initial coloring function (checkerboard with corners black)
def initialColor (i j : ℕ) : Prop :=
  (i + j) % 2 = 0

-- Define the repainting operation allowed
def repaint (cell1 cell2 : (ℕ × ℕ)) (color1 color2 : Prop) : Prop :=
  ¬color1 = color2 

-- Define the main theorem to prove
theorem impossible_to_repaint_white :
  ¬(∃ f : ℕ × ℕ -> Prop, 
    (∀ i j, (i < boardSize) → (j < boardSize) → (f (i, j) = true)) ∧ 
    (∀ i j, (i < boardSize - 1) → (repaint (i, j) (i, j+1) (f (i, j)) (f (i, j+1))) ∧
             (i < boardSize - 1) → (repaint (i, j) (i+1, j) (f (i, j)) (f (i+1, j)))))
  :=
  sorry

end impossible_to_repaint_white_l612_61209


namespace number_of_sides_l612_61274

theorem number_of_sides (n : ℕ) : 
  let a_1 := 6 
  let d := 5
  let a_n := a_1 + (n - 1) * d
  a_n = 5 * n + 1 := 
by
  sorry

end number_of_sides_l612_61274


namespace white_marbles_bagA_eq_fifteen_l612_61275

noncomputable def red_marbles_bagA := 5
def rw_ratio_bagA := (1, 3)
def wb_ratio_bagA := (2, 3)

theorem white_marbles_bagA_eq_fifteen :
  let red_to_white := rw_ratio_bagA.1 * red_marbles_bagA
  red_to_white * rw_ratio_bagA.2 = 15 :=
by
  sorry

end white_marbles_bagA_eq_fifteen_l612_61275


namespace tan_11_pi_over_4_l612_61258

theorem tan_11_pi_over_4 : Real.tan (11 * Real.pi / 4) = -1 :=
by
  -- Proof is omitted
  sorry

end tan_11_pi_over_4_l612_61258


namespace wolves_heads_count_l612_61280

/-- 
A person goes hunting in the jungle and discovers a pack of wolves.
It is known that this person has one head and two legs, 
an ordinary wolf has one head and four legs, and a mutant wolf has two heads and three legs.
The total number of heads of all the people and wolves combined is 21,
and the total number of legs is 57.
-/
theorem wolves_heads_count :
  ∃ (x y : ℕ), (x + 2 * y = 20) ∧ (4 * x + 3 * y = 55) ∧ (x + y > 0) ∧ (x + 2 * y + 1 = 21) ∧ (4 * x + 3 * y + 2 = 57) := 
by {
  sorry
}

end wolves_heads_count_l612_61280


namespace greatest_common_divisor_of_72_and_m_l612_61224

-- Definitions based on the conditions
def is_power_of_prime (m : ℕ) : Prop :=
  ∃ (p : ℕ) (k : ℕ), Nat.Prime p ∧ m = p^k

-- Main theorem based on the question and conditions
theorem greatest_common_divisor_of_72_and_m (m : ℕ) :
  (Nat.gcd 72 m = 9) ↔ (m = 3^2) ∨ ∃ k, k ≥ 2 ∧ m = 3^k :=
by
  sorry

end greatest_common_divisor_of_72_and_m_l612_61224


namespace sample_size_is_30_l612_61273

-- Definitions based on conditions
def total_students : ℕ := 700 + 500 + 300
def students_first_grade : ℕ := 700
def students_sampled_first_grade : ℕ := 14
def sample_size (n : ℕ) : Prop := students_sampled_first_grade = (students_first_grade * n) / total_students

-- Theorem stating the proof problem
theorem sample_size_is_30 : sample_size 30 :=
by
  sorry

end sample_size_is_30_l612_61273


namespace daughter_and_child_weight_l612_61297

variables (M D C : ℝ)

-- Conditions
def condition1 : Prop := M + D + C = 160
def condition2 : Prop := D = 40
def condition3 : Prop := C = (1/5) * M

-- Goal (Question)
def goal : Prop := D + C = 60

theorem daughter_and_child_weight
  (h1 : condition1 M D C)
  (h2 : condition2 D)
  (h3 : condition3 M C) : goal D C :=
by
  sorry

end daughter_and_child_weight_l612_61297


namespace probability_at_least_one_correct_l612_61271

theorem probability_at_least_one_correct :
  let p_a := 12 / 20
  let p_b := 8 / 20
  let prob_neither := (1 - p_a) * (1 - p_b)
  let prob_at_least_one := 1 - prob_neither
  prob_at_least_one = 19 / 25 := by
  sorry

end probability_at_least_one_correct_l612_61271


namespace thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one_l612_61234

theorem thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one : 37 * 23 = 851 := by
  sorry

end thirty_seven_times_twenty_three_eq_eight_hundred_fifty_one_l612_61234


namespace noah_has_largest_final_answer_l612_61231

def liam_initial := 15
def liam_final := (liam_initial - 2) * 3 + 3

def mia_initial := 15
def mia_final := (mia_initial * 3 - 4) + 3

def noah_initial := 15
def noah_final := ((noah_initial - 3) + 4) * 3

theorem noah_has_largest_final_answer : noah_final > liam_final ∧ noah_final > mia_final := by
  -- Placeholder for actual proof
  sorry

end noah_has_largest_final_answer_l612_61231


namespace initial_population_l612_61262

theorem initial_population (P : ℝ) (h1 : ∀ t : ℕ, P * (1.10 : ℝ) ^ t = 26620 → t = 3) : P = 20000 := by
  have h2 : P * (1.10) ^ 3 = 26620 := sorry
  sorry

end initial_population_l612_61262


namespace monotonic_intervals_l612_61201

open Set

noncomputable def f (a x : ℝ) : ℝ := - (1 / 3) * a * x^3 + x^2 + 1

theorem monotonic_intervals (a : ℝ) (h : a ≤ 0) :
  (a = 0 → (∀ x : ℝ, (x < 0 → deriv (f a) x < 0) ∧ (0 < x → deriv (f a) x > 0))) ∧
  (a < 0 → (∀ x : ℝ, (x < 2 / a → deriv (f a) x > 0 ∨ deriv (f a) x = 0) ∧ 
                     (2 / a < x → deriv (f a) x < 0 ∨ deriv (f a) x = 0))) :=
by
  sorry

end monotonic_intervals_l612_61201


namespace tangent_line_a_value_l612_61251

theorem tangent_line_a_value (a : ℝ) :
  (∀ x y : ℝ, (1 + a) * x + y - 1 = 0 → x^2 + y^2 + 4 * x = 0) → a = -1 / 4 :=
by
  sorry

end tangent_line_a_value_l612_61251


namespace average_speed_l612_61215

section
def flat_sand_speed : ℕ := 60
def downhill_slope_speed : ℕ := flat_sand_speed + 12
def uphill_slope_speed : ℕ := flat_sand_speed - 18

/-- Conner's average speed on flat, downhill, and uphill slopes, each of which he spends one-third of his time traveling on, is 58 miles per hour -/
theorem average_speed : (flat_sand_speed + downhill_slope_speed + uphill_slope_speed) / 3 = 58 := by
  sorry

end

end average_speed_l612_61215


namespace linear_equation_m_value_l612_61284

theorem linear_equation_m_value (m : ℝ) (x : ℝ) (h : (m - 1) * x ^ |m| - 2 = 0) : m = -1 :=
sorry

end linear_equation_m_value_l612_61284


namespace monotonicity_intervals_range_of_m_l612_61276

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + Real.log x) / x

theorem monotonicity_intervals (m : ℝ) (x : ℝ) (hx : x > 1):
  (m >= 1 → ∀ x' > 1, f m x' ≤ f m x) ∧
  (m < 1 → (∀ x' ∈ Set.Ioo 1 (Real.exp (1 - m)), f m x' > f m x) ∧
            (∀ x' ∈ Set.Ioi (Real.exp (1 - m)), f m x' < f m x)) := by
  sorry

theorem range_of_m (m : ℝ) :
  (∀ x > 1, f m x < m * x) ↔ m ≥ 1/2 := by
  sorry

end monotonicity_intervals_range_of_m_l612_61276


namespace geom_seq_11th_term_l612_61253

/-!
The fifth and eighth terms of a geometric sequence are -2 and -54, respectively. 
What is the 11th term of this progression?
-/
theorem geom_seq_11th_term {a : ℕ → ℤ} (r : ℤ) 
  (h1 : a 5 = -2) (h2 : a 8 = -54) 
  (h3 : ∀ n : ℕ, a (n + 3) = a n * r ^ 3) : 
  a 11 = -1458 :=
sorry

end geom_seq_11th_term_l612_61253


namespace find_number_and_remainder_l612_61277

theorem find_number_and_remainder :
  ∃ (N r : ℕ), (3927 + 2873) * (3 * (3927 - 2873)) + r = N ∧ r < (3927 + 2873) :=
sorry

end find_number_and_remainder_l612_61277


namespace profit_percentage_l612_61265

-- Define the selling price and the cost price
def SP : ℝ := 100
def CP : ℝ := 86.95652173913044

-- State the theorem for profit percentage
theorem profit_percentage :
  ((SP - CP) / CP) * 100 = 15 :=
by
  sorry

end profit_percentage_l612_61265


namespace choose_18_4_eq_3060_l612_61294

/-- The number of ways to select 4 members from a group of 18 people (without regard to order). -/
theorem choose_18_4_eq_3060 : Nat.choose 18 4 = 3060 := 
by
  sorry

end choose_18_4_eq_3060_l612_61294


namespace geometric_sequence_a3_value_l612_61221

noncomputable def geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem geometric_sequence_a3_value :
  ∃ a : ℕ → ℝ, ∃ r : ℝ,
  geometric_seq a r ∧
  a 1 = 2 ∧
  (a 3) * (a 5) = 4 * (a 6)^2 →
  a 3 = 1 :=
sorry

end geometric_sequence_a3_value_l612_61221


namespace train_seats_count_l612_61233

theorem train_seats_count 
  (Standard Comfort Premium : ℝ)
  (Total_SEATS : ℝ)
  (hs : Standard = 36)
  (hc : Comfort = 0.20 * Total_SEATS)
  (hp : Premium = (3/5) * Total_SEATS)
  (ht : Standard + Comfort + Premium = Total_SEATS) :
  Total_SEATS = 180 := sorry

end train_seats_count_l612_61233


namespace difference_before_exchange_l612_61227

--Definitions
variables {S B : ℤ}

-- Conditions
axiom h1 : S - 2 = B + 2
axiom h2 : B > S

theorem difference_before_exchange : B - S = 2 :=
by
-- Proof will go here
sorry

end difference_before_exchange_l612_61227


namespace sum_odd_implies_parity_l612_61263

theorem sum_odd_implies_parity (a b c: ℤ) (h: (a + b + c) % 2 = 1) : (a^2 + b^2 - c^2 + 2 * a * b) % 2 = 1 := 
sorry

end sum_odd_implies_parity_l612_61263


namespace no_pos_int_solutions_l612_61247

theorem no_pos_int_solutions (k x y : ℕ) (hk : k > 0) (hx : x > 0) (hy : y > 0) :
  x^2 + 2^(2 * k) + 1 ≠ y^3 := by
  sorry

end no_pos_int_solutions_l612_61247


namespace lollipops_Lou_received_l612_61202

def initial_lollipops : ℕ := 42
def given_to_Emily : ℕ := 2 * initial_lollipops / 3
def kept_by_Marlon : ℕ := 4
def lollipops_left_after_Emily : ℕ := initial_lollipops - given_to_Emily
def lollipops_given_to_Lou : ℕ := lollipops_left_after_Emily - kept_by_Marlon

theorem lollipops_Lou_received : lollipops_given_to_Lou = 10 := by
  sorry

end lollipops_Lou_received_l612_61202


namespace point_coordinates_l612_61295

theorem point_coordinates (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : abs y = 5) (h4 : abs x = 2) : x = -2 ∧ y = 5 :=
by
  sorry

end point_coordinates_l612_61295


namespace probability_three_white_two_black_l612_61287

-- Define the total number of balls
def total_balls : ℕ := 17

-- Define the number of white balls
def white_balls : ℕ := 8

-- Define the number of black balls
def black_balls : ℕ := 9

-- Define the number of balls drawn
def balls_drawn : ℕ := 5

-- Define three white balls drawn
def three_white_drawn : ℕ := 3

-- Define two black balls drawn
def two_black_drawn : ℕ := 2

-- Define the combination formula
noncomputable def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Calculate the probability
noncomputable def probability : ℚ :=
  (combination white_balls three_white_drawn * combination black_balls two_black_drawn : ℚ) 
  / combination total_balls balls_drawn

-- Statement to prove
theorem probability_three_white_two_black :
  probability = 672 / 2063 := by
  sorry

end probability_three_white_two_black_l612_61287


namespace sum_weights_second_fourth_l612_61229

-- Definitions based on given conditions
noncomputable section

def weight (n : ℕ) : ℕ := 4 - (n - 1)

-- Assumption that weights form an arithmetic sequence.
-- 1st foot weighs 4 jin, 5th foot weighs 2 jin, and weights are linearly decreasing.
axiom weight_arith_seq (n : ℕ) : weight n = 4 - (n - 1)

-- Prove the sum of the weights of the second and fourth feet
theorem sum_weights_second_fourth :
  weight 2 + weight 4 = 6 :=
by
  simp [weight_arith_seq]
  sorry

end sum_weights_second_fourth_l612_61229


namespace total_elephants_in_two_parks_l612_61203

theorem total_elephants_in_two_parks (n1 n2 : ℕ) (h1 : n1 = 70) (h2 : n2 = 3 * n1) : n1 + n2 = 280 := by
  sorry

end total_elephants_in_two_parks_l612_61203


namespace three_gorges_scientific_notation_l612_61292

theorem three_gorges_scientific_notation :
  ∃a n : ℝ, (1 ≤ |a| ∧ |a| < 10) ∧ (798.5 * 10^1 = a * 10^n) ∧ a = 7.985 ∧ n = 2 :=
by
  sorry

end three_gorges_scientific_notation_l612_61292


namespace proof_problem_l612_61259

theorem proof_problem
  (x y a b c d : ℝ)
  (h1 : |x - 1| + (y + 2)^2 = 0)
  (h2 : a * b = 1)
  (h3 : c + d = 0) :
  (x + y)^3 - (-a * b)^2 + 3 * c + 3 * d = -2 :=
by
  -- The proof steps go here.
  sorry

end proof_problem_l612_61259


namespace find_triplet_l612_61239

def ordered_triplet : Prop :=
  ∃ (x y z : ℚ), 
  7 * x + 3 * y = z - 10 ∧ 
  2 * x - 4 * y = 3 * z + 20 ∧ 
  x = 0 ∧ 
  y = -50 / 13 ∧ 
  z = -20 / 13

theorem find_triplet : ordered_triplet := 
  sorry

end find_triplet_l612_61239


namespace sequence_properties_l612_61288

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 3 = 3 ∧ ∀ n, a (n + 1) = a n + 2

theorem sequence_properties {a : ℕ → ℤ} (h : arithmetic_sequence a) :
  a 2 + a 4 = 6 ∧ ∀ n, a n = 2 * n - 3 :=
by
  sorry

end sequence_properties_l612_61288


namespace line_through_two_points_l612_61237

theorem line_through_two_points :
  ∀ (A_1 B_1 A_2 B_2 : ℝ),
    (2 * A_1 + 3 * B_1 = 1) →
    (2 * A_2 + 3 * B_2 = 1) →
    (∀ (x y : ℝ), (2 * x + 3 * y = 1) → (x * (B_2 - B_1) + y * (A_1 - A_2) = A_1 * B_2 - A_2 * B_1)) :=
by 
  intros A_1 B_1 A_2 B_2 h1 h2 x y hxy
  sorry

end line_through_two_points_l612_61237


namespace number_of_arrangements_l612_61269

theorem number_of_arrangements (A B : Type) (individuals : Fin 6 → Type)
  (adjacent_condition : ∃ (i : Fin 5), individuals i = B ∧ individuals (i + 1) = A) :
  ∃ (n : ℕ), n = 120 :=
by
  sorry

end number_of_arrangements_l612_61269


namespace price_of_shares_l612_61220

variable (share_value : ℝ) (dividend_rate : ℝ) (tax_rate : ℝ) (effective_return : ℝ) (price : ℝ)

-- Given conditions
axiom H1 : share_value = 50
axiom H2 : dividend_rate = 0.185
axiom H3 : tax_rate = 0.05
axiom H4 : effective_return = 0.25
axiom H5 : 0.25 * price = 0.185 * 50 - (0.05 * (0.185 * 50))

-- Prove that the price at which the investor bought the shares is Rs. 35.15
theorem price_of_shares : price = 35.15 :=
by
  sorry

end price_of_shares_l612_61220


namespace arithmetic_series_sum_l612_61278

def first_term (k : ℕ) : ℕ := k^2 + k + 1
def common_difference : ℕ := 1
def number_of_terms (k : ℕ) : ℕ := 2 * k + 3
def nth_term (k n : ℕ) : ℕ := (first_term k) + (n - 1) * common_difference
def sum_of_terms (k : ℕ) : ℕ :=
  let n := number_of_terms k
  let a := first_term k
  let l := nth_term k n
  n * (a + l) / 2

theorem arithmetic_series_sum (k : ℕ) : sum_of_terms k = 2 * k^3 + 7 * k^2 + 10 * k + 6 :=
sorry

end arithmetic_series_sum_l612_61278


namespace mean_of_remaining_four_numbers_l612_61286

theorem mean_of_remaining_four_numbers (a b c d : ℝ) :
  (a + b + c + d + 105) / 5 = 92 → (a + b + c + d) / 4 = 88.75 :=
by
  intro h
  sorry

end mean_of_remaining_four_numbers_l612_61286


namespace quadratic_function_origin_l612_61212

theorem quadratic_function_origin {a b c : ℝ} :
  (∀ x, y = ax * x + bx * x + c → y = 0 → 0 = c ∧ b = 0) ∨ (c = 0) :=
sorry

end quadratic_function_origin_l612_61212


namespace evaluate_expression_l612_61285

theorem evaluate_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end evaluate_expression_l612_61285


namespace union_sets_l612_61232

def M : Set ℕ := {0, 1, 3}
def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_sets : M ∪ N = {0, 1, 3, 9} :=
by
  sorry

end union_sets_l612_61232


namespace correct_division_algorithm_l612_61283

theorem correct_division_algorithm : (-8 : ℤ) / (-4 : ℤ) = (8 : ℤ) / (4 : ℤ) := 
by 
  sorry

end correct_division_algorithm_l612_61283


namespace largest_divisor_n4_minus_5n2_plus_6_l612_61213

theorem largest_divisor_n4_minus_5n2_plus_6 :
  ∀ (n : ℤ), (n^4 - 5 * n^2 + 6) % 1 = 0 :=
by
  sorry

end largest_divisor_n4_minus_5n2_plus_6_l612_61213


namespace int_values_satisfy_condition_l612_61241

theorem int_values_satisfy_condition :
  ∃ (count : ℕ), count = 10 ∧ ∀ (x : ℤ), 6 > Real.sqrt x ∧ Real.sqrt x > 5 ↔ (x ≥ 26 ∧ x ≤ 35) := by
  sorry

end int_values_satisfy_condition_l612_61241


namespace vacation_cost_division_l612_61257

theorem vacation_cost_division (n : ℕ) (h1 : 720 / 4 = 60 + 720 / n) : n = 3 := by
  sorry

end vacation_cost_division_l612_61257


namespace no_root_of_equation_l612_61245

theorem no_root_of_equation : ∀ x : ℝ, x - 8 / (x - 4) ≠ 4 - 8 / (x - 4) :=
by
  intro x
  -- Original equation:
  -- x - 8 / (x - 4) = 4 - 8 / (x - 4)
  -- No valid value of x solves the above equation as shown in the given solution
  sorry

end no_root_of_equation_l612_61245


namespace matrix_det_example_l612_61293

variable (A : Matrix (Fin 2) (Fin 2) ℤ) 
  (hA : A = ![![5, -4], ![2, 3]])

theorem matrix_det_example : Matrix.det A = 23 :=
by
  sorry

end matrix_det_example_l612_61293


namespace min_value_at_1_l612_61243

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2 * a * x + 8 else x + 4 / x + 2 * a

theorem min_value_at_1 (a : ℝ) :
  (∀ x, f x a ≥ f 1 a) ↔ (a = 5/4 ∨ a = 2 ∨ a = 4) :=
by
  sorry

end min_value_at_1_l612_61243


namespace time_difference_in_minutes_l612_61206

def speed := 60 -- speed of the car in miles per hour
def distance1 := 360 -- distance of the first trip in miles
def distance2 := 420 -- distance of the second trip in miles
def hours_to_minutes := 60 -- conversion factor from hours to minutes

theorem time_difference_in_minutes :
  ((distance2 / speed) - (distance1 / speed)) * hours_to_minutes = 60 :=
by
  -- proof to be provided
  sorry

end time_difference_in_minutes_l612_61206


namespace find_function_that_satisfies_eq_l612_61240

theorem find_function_that_satisfies_eq :
  ∀ (f : ℕ → ℕ), (∀ (m n : ℕ), f (m + f n) = f (f m) + f n) → (∀ n : ℕ, f n = n) :=
by
  intro f
  intro h
  sorry

end find_function_that_satisfies_eq_l612_61240


namespace height_percentage_difference_l612_61299

theorem height_percentage_difference (H : ℝ) (p r q : ℝ) 
  (hp : p = 0.60 * H) 
  (hr : r = 1.30 * H) : 
  (r - p) / p * 100 = 116.67 :=
by
  sorry

end height_percentage_difference_l612_61299


namespace angle_supplement_complement_l612_61290

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l612_61290


namespace find_range_of_a_l612_61226

-- Define the conditions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 4 * x + a^2 > 0
def q (a : ℝ) : Prop := a^2 - 5 * a - 6 ≥ 0

-- Define the proposition that one of p or q is true and the other is false
def p_or_q (a : ℝ) : Prop := p a ∨ q a
def not_p_and_q (a : ℝ) : Prop := ¬(p a ∧ q a)

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (2 < a ∧ a < 6) ∨ (-2 ≤ a ∧ a ≤ -1)

-- Theorem statement
theorem find_range_of_a (a : ℝ) : p_or_q a ∧ not_p_and_q a → range_of_a a :=
by
  sorry

end find_range_of_a_l612_61226


namespace coupon_redeem_day_l612_61238

theorem coupon_redeem_day (first_day : ℕ) (redeem_every : ℕ) : 
  (∀ n : ℕ, n < 8 → (first_day + n * redeem_every) % 7 ≠ 6) ↔ (first_day % 7 = 2 ∨ first_day % 7 = 5) :=
by
  sorry

end coupon_redeem_day_l612_61238


namespace total_cost_l612_61230

theorem total_cost
  (cost_berries   : ℝ := 11.08)
  (cost_apples    : ℝ := 14.33)
  (cost_peaches   : ℝ := 9.31)
  (cost_grapes    : ℝ := 7.50)
  (cost_bananas   : ℝ := 5.25)
  (cost_pineapples: ℝ := 4.62)
  (total_cost     : ℝ := cost_berries + cost_apples + cost_peaches + cost_grapes + cost_bananas + cost_pineapples) :
  total_cost = 52.09 :=
by
  sorry

end total_cost_l612_61230


namespace distinct_digits_unique_D_l612_61200

theorem distinct_digits_unique_D 
  (A B C D : ℕ)
  (hA : A ≠ B)
  (hB : B ≠ C)
  (hC : C ≠ D)
  (hD : D ≠ A)
  (h1 : D < 10)
  (h2 : B < 10)
  (h3 : C < 10)
  (h4 : A < 10)
  (h_add : A * 1000 + A * 100 + C * 10 + B + B * 1000 + C * 100 + B * 10 + D = B * 1000 + D * 100 + A * 10 + B) :
  D = 0 :=
by sorry

end distinct_digits_unique_D_l612_61200


namespace evaluate_tensor_expression_l612_61291

-- Define the tensor operation
def tensor (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

-- The theorem we want to prove
theorem evaluate_tensor_expression : tensor (tensor 5 3) 2 = 293 / 15 := by
  sorry

end evaluate_tensor_expression_l612_61291


namespace expression_in_multiply_form_l612_61256

def a : ℕ := 3 ^ 1005
def b : ℕ := 7 ^ 1006
def m : ℕ := 114337548

theorem expression_in_multiply_form : 
  (a + b)^2 - (a - b)^2 = m * 10 ^ 1006 :=
by
  sorry

end expression_in_multiply_form_l612_61256


namespace convert_angle_l612_61219

theorem convert_angle (α : ℝ) (k : ℤ) :
  -1485 * (π / 180) = α + 2 * k * π ∧ 0 ≤ α ∧ α < 2 * π ∧ k = -10 ∧ α = 7 * π / 4 :=
by
  sorry

end convert_angle_l612_61219


namespace sheela_monthly_income_l612_61282

theorem sheela_monthly_income (deposit : ℝ) (percentage : ℝ) (income : ℝ) 
  (h1 : deposit = 2500) (h2 : percentage = 0.25) (h3 : deposit = percentage * income) :
  income = 10000 := 
by
  -- proof steps would go here
  sorry

end sheela_monthly_income_l612_61282


namespace sum_of_squares_of_tom_rates_l612_61236

theorem sum_of_squares_of_tom_rates :
  ∃ r b k : ℕ, 3 * r + 4 * b + 2 * k = 104 ∧
               3 * r + 6 * b + 2 * k = 140 ∧
               r^2 + b^2 + k^2 = 440 :=
by
  sorry

end sum_of_squares_of_tom_rates_l612_61236


namespace ten_more_than_twice_number_of_birds_l612_61266

def number_of_birds : ℕ := 20

theorem ten_more_than_twice_number_of_birds :
  10 + 2 * number_of_birds = 50 :=
by
  sorry

end ten_more_than_twice_number_of_birds_l612_61266


namespace groupB_avg_weight_eq_141_l612_61210

def initial_group_weight (avg_weight : ℝ) : ℝ := 50 * avg_weight
def groupA_weight_gain : ℝ := 20 * 15
def groupB_weight_gain (x : ℝ) : ℝ := 20 * x

def total_weight (avg_weight : ℝ) (x : ℝ) : ℝ :=
  initial_group_weight avg_weight + groupA_weight_gain + groupB_weight_gain x

def total_avg_weight : ℝ := 46
def num_friends : ℝ := 90

def original_avg_weight : ℝ := total_avg_weight - 12
def final_total_weight : ℝ := num_friends * total_avg_weight

theorem groupB_avg_weight_eq_141 : 
  ∀ (avg_weight : ℝ) (x : ℝ),
    avg_weight = original_avg_weight →
    initial_group_weight avg_weight + groupA_weight_gain + groupB_weight_gain x = final_total_weight →
    avg_weight + x = 141 :=
by 
  intros avg_weight x h₁ h₂
  sorry

end groupB_avg_weight_eq_141_l612_61210


namespace park_area_is_correct_l612_61270

-- Define the side of the square
def side_length : ℕ := 30

-- Define the area function for a square
def area_of_square (side: ℕ) : ℕ := side * side

-- State the theorem we're going to prove
theorem park_area_is_correct : area_of_square side_length = 900 := 
sorry -- proof not required

end park_area_is_correct_l612_61270


namespace mean_age_gauss_family_l612_61268

theorem mean_age_gauss_family :
  let ages := [7, 7, 7, 14, 15]
  let sum_ages := List.sum ages
  let number_of_children := List.length ages
  let mean_age := sum_ages / number_of_children
  mean_age = 10 :=
by
  sorry

end mean_age_gauss_family_l612_61268


namespace find_q_minus_p_values_l612_61267

theorem find_q_minus_p_values (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : 0 < n) 
    (h : (p * (q + 1) + q * (p + 1)) * (n + 2) = 2 * n * p * q) : 
    q - p = 2 ∨ q - p = 3 ∨ q - p = 5 :=
sorry

end find_q_minus_p_values_l612_61267


namespace John_l612_61228

theorem John's_score_in_blackjack
  (Theodore_score : ℕ)
  (Zoey_cards : List ℕ)
  (winning_score : ℕ)
  (John_score : ℕ)
  (h1 : Theodore_score = 13)
  (h2 : Zoey_cards = [11, 3, 5])
  (h3 : winning_score = 19)
  (h4 : Zoey_cards.sum = winning_score)
  (h5 : winning_score ≠ Theodore_score) :
  John_score < 19 :=
by
  -- Here we would provide the proof if required
  sorry

end John_l612_61228


namespace mary_baking_cups_l612_61225

-- Conditions
def flour_needed : ℕ := 9
def sugar_needed : ℕ := 11
def flour_added : ℕ := 4
def sugar_added : ℕ := 0

-- Statement to prove
theorem mary_baking_cups : sugar_needed - (flour_needed - flour_added) = 6 := by
  sorry

end mary_baking_cups_l612_61225


namespace quadratic_properties_l612_61272

noncomputable def quadratic_function (a b c : ℝ) : ℝ → ℝ :=
  λ x => a * x^2 + b * x + c

def min_value_passing_point (f : ℝ → ℝ) : Prop :=
  (f (-1) = -4) ∧ (f 0 = -3)

def intersects_x_axis (f : ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  (f p1.1 = p1.2) ∧ (f p2.1 = p2.2)

def max_value_in_interval (f : ℝ → ℝ) (a b max_val : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → f x ≤ max_val

theorem quadratic_properties :
  ∃ f : ℝ → ℝ,
    min_value_passing_point f ∧
    intersects_x_axis f (1, 0) (-3, 0) ∧
    max_value_in_interval f (-2) 2 5 :=
by
  sorry

end quadratic_properties_l612_61272


namespace probability_at_most_six_distinct_numbers_l612_61249

def roll_eight_dice : ℕ := 6^8

def favorable_cases : ℕ := 3628800

def probability_six_distinct_numbers (n : ℕ) (f : ℕ) : ℚ :=
  f / n

theorem probability_at_most_six_distinct_numbers :
  probability_six_distinct_numbers roll_eight_dice favorable_cases = 45 / 52 := by
  sorry

end probability_at_most_six_distinct_numbers_l612_61249


namespace value_of_g_g_2_l612_61211

def g (x : ℝ) : ℝ := 4 * x^2 + 3

theorem value_of_g_g_2 : g (g 2) = 1447 := by
  sorry

end value_of_g_g_2_l612_61211


namespace eliminate_denominators_eq_l612_61214

theorem eliminate_denominators_eq :
  ∀ (x : ℝ), 1 - (x + 3) / 6 = x / 2 → 6 - x - 3 = 3 * x :=
by
  intro x
  intro h
  -- Place proof steps here.
  sorry

end eliminate_denominators_eq_l612_61214


namespace calculate_f_f_f_l612_61205

def f (x : ℤ) : ℤ := 3 * x + 2

theorem calculate_f_f_f :
  f (f (f 3)) = 107 :=
by
  sorry

end calculate_f_f_f_l612_61205


namespace range_of_values_l612_61218

theorem range_of_values (x y : ℝ) (h : (x + 2)^2 + y^2 / 4 = 1) :
  ∃ (a b : ℝ), a = 1 ∧ b = 28 / 3 ∧ a ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ b := by
  sorry

end range_of_values_l612_61218


namespace part1_part2_l612_61207

-- Definitions of y1 and y2 based on given conditions
def y1 (x : ℝ) : ℝ := -x + 3
def y2 (x : ℝ) : ℝ := 2 + x

-- Prove for x such that y1 = y2
theorem part1 (x : ℝ) : y1 x = y2 x ↔ x = 1 / 2 := by
  sorry

-- Prove for x such that y1 = 2y2 + 5
theorem part2 (x : ℝ) : y1 x = 2 * y2 x + 5 ↔ x = -2 := by
  sorry

end part1_part2_l612_61207


namespace incorrect_option_c_l612_61223

theorem incorrect_option_c (R : ℝ) : 
  let cylinder_lateral_area := 4 * π * R^2
  let sphere_surface_area := 4 * π * R^2
  cylinder_lateral_area = sphere_surface_area :=
  sorry

end incorrect_option_c_l612_61223


namespace soccer_team_games_played_l612_61248

theorem soccer_team_games_played 
  (players : ℕ) (total_goals : ℕ) (third_players_goals_per_game : ℕ → ℕ) (other_players_goals : ℕ) (G : ℕ)
  (h1 : players = 24)
  (h2 : total_goals = 150)
  (h3 : ∃ n, n = players / 3 ∧ ∀ g, third_players_goals_per_game g = n * g)
  (h4 : other_players_goals = 30)
  (h5 : total_goals = third_players_goals_per_game G + other_players_goals) :
  G = 15 := by
  -- Proof would go here
  sorry

end soccer_team_games_played_l612_61248


namespace last_digit_of_large_exponentiation_l612_61222

theorem last_digit_of_large_exponentiation
  (a : ℕ) (b : ℕ)
  (h1 : a = 954950230952380948328708) 
  (h2 : b = 470128749397540235934750230) :
  (a ^ b) % 10 = 4 :=
sorry

end last_digit_of_large_exponentiation_l612_61222


namespace parallelogram_base_l612_61204

theorem parallelogram_base (height area : ℕ) (h_height : height = 18) (h_area : area = 612) : ∃ base, base = 34 :=
by
  -- The proof would go here
  sorry

end parallelogram_base_l612_61204
