import Mathlib

namespace manicure_cost_calculation_l2078_207871

/-- The cost of a manicure in a nail salon. -/
def manicure_cost (total_revenue : ℚ) (total_fingers : ℕ) (fingers_per_person : ℕ) (non_clients : ℕ) : ℚ :=
  total_revenue / ((total_fingers / fingers_per_person) - non_clients)

/-- Theorem stating the cost of a manicure in the given scenario. -/
theorem manicure_cost_calculation :
  manicure_cost 200 210 10 11 = 952 / 100 := by sorry

end manicure_cost_calculation_l2078_207871


namespace problem_statement_l2078_207891

theorem problem_statement (a b : ℝ) :
  (a + 1)^2 + Real.sqrt (b - 2) = 0 → a - b = -3 := by
  sorry

end problem_statement_l2078_207891


namespace base_7_addition_l2078_207874

/-- Addition in base 7 -/
def add_base_7 (a b : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 7 -/
def to_base_7 (n : ℕ) : ℕ := sorry

/-- Conversion from base 7 to base 10 -/
def from_base_7 (n : ℕ) : ℕ := sorry

theorem base_7_addition : add_base_7 (from_base_7 25) (from_base_7 256) = from_base_7 544 := by
  sorry

end base_7_addition_l2078_207874


namespace total_octopus_legs_l2078_207876

/-- The number of legs an octopus has -/
def legs_per_octopus : ℕ := 8

/-- The number of octopuses Carson saw -/
def octopuses_seen : ℕ := 5

/-- The total number of octopus legs Carson saw -/
def total_legs : ℕ := octopuses_seen * legs_per_octopus

theorem total_octopus_legs : total_legs = 40 := by
  sorry

end total_octopus_legs_l2078_207876


namespace shortest_altitude_right_triangle_l2078_207848

/-- The shortest altitude of a triangle with sides 13, 84, and 85 --/
theorem shortest_altitude_right_triangle : 
  ∀ (a b c h : ℝ), 
    a = 13 → b = 84 → c = 85 →
    a^2 + b^2 = c^2 →
    h * c = 2 * (1/2 * a * b) →
    h = 1092 / 85 := by
  sorry

end shortest_altitude_right_triangle_l2078_207848


namespace min_value_inequality_l2078_207884

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- Theorem statement
theorem min_value_inequality (k a b c : ℝ) : 
  (∀ x, f x ≥ k) → -- k is the minimum value of f
  (a > 0 ∧ b > 0 ∧ c > 0) → -- a, b, c are positive
  (3 / (k * a) + 3 / (2 * k * b) + 1 / (k * c) = 1) → -- given equation
  a + 2 * b + 3 * c ≥ 9 := by
  sorry

end min_value_inequality_l2078_207884


namespace f_three_point_five_l2078_207899

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property that f(x+2) is an odd function
axiom f_odd (x : ℝ) : f (-(x + 2)) = -f (x + 2)

-- Define the property that f(x) = 2x for x ∈ (0,2)
axiom f_linear (x : ℝ) : x > 0 → x < 2 → f x = 2 * x

-- Theorem to prove
theorem f_three_point_five : f 3.5 = -1 := by sorry

end f_three_point_five_l2078_207899


namespace french_exam_words_to_learn_l2078_207870

/-- The least number of words to learn for a French exam -/
def least_words_to_learn : ℕ := 569

theorem french_exam_words_to_learn 
  (total_words : ℕ) 
  (recall_rate : ℚ) 
  (target_recall : ℚ) 
  (h1 : total_words = 600)
  (h2 : recall_rate = 95 / 100)
  (h3 : target_recall = 90 / 100) :
  (↑least_words_to_learn : ℚ) ≥ (target_recall * total_words) / recall_rate ∧ 
  (↑(least_words_to_learn - 1) : ℚ) < (target_recall * total_words) / recall_rate :=
sorry

end french_exam_words_to_learn_l2078_207870


namespace normal_dist_probability_l2078_207862

-- Define the normal distribution
def normal_dist (μ σ : ℝ) : Type := Unit

-- Define the probability function
noncomputable def P (X : normal_dist 4 1) (a b : ℝ) : ℝ := sorry

-- State the theorem
theorem normal_dist_probability 
  (X : normal_dist 4 1) 
  (h1 : P X (4 - 2) (4 + 2) = 0.9544) 
  (h2 : P X (4 - 1) (4 + 1) = 0.6826) : 
  P X 5 6 = 0.1359 := by sorry

end normal_dist_probability_l2078_207862


namespace production_average_l2078_207863

theorem production_average (n : ℕ) : 
  (∀ (past_total : ℕ), past_total = n * 50 →
   ∀ (new_total : ℕ), new_total = past_total + 90 →
   (new_total : ℚ) / (n + 1 : ℚ) = 52) →
  n = 19 := by
sorry

end production_average_l2078_207863


namespace sqrt_meaningful_iff_geq_two_l2078_207845

theorem sqrt_meaningful_iff_geq_two (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by sorry

end sqrt_meaningful_iff_geq_two_l2078_207845


namespace intersection_empty_implies_m_less_than_negative_one_l2078_207883

theorem intersection_empty_implies_m_less_than_negative_one (m : ℝ) : 
  let M := {x : ℝ | x - m ≤ 0}
  let N := {y : ℝ | ∃ x : ℝ, y = (x - 1)^2 - 1}
  M ∩ N = ∅ → m < -1 := by
  sorry

end intersection_empty_implies_m_less_than_negative_one_l2078_207883


namespace arccos_equation_solution_l2078_207867

theorem arccos_equation_solution :
  ∀ x : ℝ, Real.arccos (3 * x) - Real.arccos x = π / 3 → x = -3 * Real.sqrt 21 / 28 := by
  sorry

end arccos_equation_solution_l2078_207867


namespace hidden_numbers_average_l2078_207807

/-- Given three cards with visible numbers and hidden consecutive odd numbers,
    if the sum of numbers on each card is equal, then the average of hidden numbers is 18. -/
theorem hidden_numbers_average (v₁ v₂ v₃ h₁ h₂ h₃ : ℕ) : 
  v₁ = 30 ∧ v₂ = 42 ∧ v₃ = 36 →  -- visible numbers
  h₂ = h₁ + 2 ∧ h₃ = h₂ + 2 →    -- hidden numbers are consecutive odd
  v₁ + h₁ = v₂ + h₂ ∧ v₂ + h₂ = v₃ + h₃ →  -- sum on each card is equal
  (h₁ + h₂ + h₃) / 3 = 18 :=
by sorry

end hidden_numbers_average_l2078_207807


namespace upstream_downstream_time_ratio_l2078_207800

/-- The speed of the boat in still water in kmph -/
def boat_speed : ℝ := 57

/-- The speed of the stream in kmph -/
def stream_speed : ℝ := 19

/-- The time taken to row upstream -/
def time_upstream : ℝ := sorry

/-- The time taken to row downstream -/
def time_downstream : ℝ := sorry

/-- The distance traveled (assumed to be the same for both upstream and downstream) -/
def distance : ℝ := sorry

theorem upstream_downstream_time_ratio :
  time_upstream / time_downstream = 2 := by sorry

end upstream_downstream_time_ratio_l2078_207800


namespace refrigerator_loss_percentage_l2078_207859

/-- Represents the loss percentage on the refrigerator -/
def loss_percentage : ℝ := 4

/-- Represents the cost price of the refrigerator in Rupees -/
def refrigerator_cp : ℝ := 15000

/-- Represents the cost price of the mobile phone in Rupees -/
def mobile_cp : ℝ := 8000

/-- Represents the profit percentage on the mobile phone -/
def mobile_profit_percentage : ℝ := 9

/-- Represents the overall profit in Rupees -/
def overall_profit : ℝ := 120

/-- Theorem stating that given the conditions, the loss percentage on the refrigerator is 4% -/
theorem refrigerator_loss_percentage :
  let mobile_sp := mobile_cp * (1 + mobile_profit_percentage / 100)
  let total_cp := refrigerator_cp + mobile_cp
  let total_sp := total_cp + overall_profit
  let refrigerator_sp := total_sp - mobile_sp
  let loss := refrigerator_cp - refrigerator_sp
  loss_percentage = (loss / refrigerator_cp) * 100 := by
  sorry


end refrigerator_loss_percentage_l2078_207859


namespace inequality_range_l2078_207875

open Real

theorem inequality_range (a : ℝ) : 
  (∀ x > 1, a * log x > 1 - 1/x) ↔ a ≥ 1 := by sorry

end inequality_range_l2078_207875


namespace earliest_retirement_is_2009_l2078_207843

/-- Rule of 70 provision: An employee can retire when age + years of employment ≥ 70 -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- The year the employee was hired -/
def hire_year : ℕ := 1990

/-- The employee's age when hired -/
def hire_age : ℕ := 32

/-- The earliest retirement year satisfies the rule of 70 -/
def earliest_retirement_year (year : ℕ) : Prop :=
  rule_of_70 (hire_age + (year - hire_year)) (year - hire_year) ∧
  ∀ y < year, ¬rule_of_70 (hire_age + (y - hire_year)) (y - hire_year)

/-- Theorem: The earliest retirement year for the employee is 2009 -/
theorem earliest_retirement_is_2009 : earliest_retirement_year 2009 := by
  sorry

end earliest_retirement_is_2009_l2078_207843


namespace coplanar_points_scalar_l2078_207864

theorem coplanar_points_scalar (O E F G H : EuclideanSpace ℝ (Fin 3)) (m : ℝ) :
  (O = 0) →
  (4 • (E - O) - 3 • (F - O) + 2 • (G - O) + m • (H - O) = 0) →
  (∃ (a b c d : ℝ), a • (E - O) + b • (F - O) + c • (G - O) + d • (H - O) = 0 ∧ 
    (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)) →
  m = -3 :=
by sorry

end coplanar_points_scalar_l2078_207864


namespace tv_sales_decrease_l2078_207872

/-- Proves that a 70% price increase and 36% revenue increase results in a 20% sales decrease -/
theorem tv_sales_decrease (initial_price initial_quantity : ℝ) 
  (initial_price_positive : initial_price > 0)
  (initial_quantity_positive : initial_quantity > 0) : 
  let new_price := 1.7 * initial_price
  let new_revenue := 1.36 * (initial_price * initial_quantity)
  let new_quantity := new_revenue / new_price
  (initial_quantity - new_quantity) / initial_quantity = 0.2 := by
  sorry

end tv_sales_decrease_l2078_207872


namespace sin_alpha_value_l2078_207815

theorem sin_alpha_value (α : ℝ) (h : 3 * Real.sin (2 * α) = Real.cos α) : 
  Real.sin α = 1/6 := by
sorry

end sin_alpha_value_l2078_207815


namespace distance_to_x_axis_l2078_207821

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-3, -2)) : 
  |P.2| = 2 := by sorry

end distance_to_x_axis_l2078_207821


namespace max_prob_with_highest_prob_player_second_l2078_207827

/-- Represents a chess player with a winning probability -/
structure Player where
  winProb : ℝ

/-- Represents the order of games played -/
inductive GameOrder
| ABC
| ACB
| BAC
| BCA
| CAB
| CBA

/-- Calculates the probability of winning two consecutive games given a game order -/
def probTwoConsecutiveWins (p₁ p₂ p₃ : ℝ) (order : GameOrder) : ℝ :=
  match order with
  | GameOrder.ABC => 2 * (p₁ * p₂)
  | GameOrder.ACB => 2 * (p₁ * p₃)
  | GameOrder.BAC => 2 * (p₂ * p₁)
  | GameOrder.BCA => 2 * (p₂ * p₃)
  | GameOrder.CAB => 2 * (p₃ * p₁)
  | GameOrder.CBA => 2 * (p₃ * p₂)

theorem max_prob_with_highest_prob_player_second 
  (A B C : Player) 
  (h₁ : 0 < A.winProb) 
  (h₂ : A.winProb < B.winProb) 
  (h₃ : B.winProb < C.winProb) :
  ∀ (order : GameOrder), 
    probTwoConsecutiveWins A.winProb B.winProb C.winProb order ≤ 
    max (probTwoConsecutiveWins A.winProb B.winProb C.winProb GameOrder.CAB)
        (probTwoConsecutiveWins A.winProb B.winProb C.winProb GameOrder.CBA) :=
by sorry

end max_prob_with_highest_prob_player_second_l2078_207827


namespace percent_of_percent_l2078_207866

theorem percent_of_percent : (3 / 100) / (5 / 100) * 100 = 60 := by
  sorry

end percent_of_percent_l2078_207866


namespace class_size_problem_l2078_207808

theorem class_size_problem (x : ℕ) : 
  x ≥ 46 → 
  (7 : ℚ) / 24 * x < 15 → 
  x = 48 :=
by sorry

end class_size_problem_l2078_207808


namespace constant_term_position_l2078_207861

/-- The position of the constant term in the expansion of (√a - 2/∛a)^30 -/
theorem constant_term_position (a : ℝ) (h : a > 0) : 
  ∃ (r : ℕ), r = 18 ∧ 
  (∀ (k : ℕ), k ≠ r → (90 - 5 * k : ℚ) / 6 ≠ 0) ∧
  (90 - 5 * r : ℚ) / 6 = 0 := by
  sorry

end constant_term_position_l2078_207861


namespace staircase_climbing_ways_l2078_207817

/-- The number of ways to climb n steps, where one can go up by 1, 2, or 3 steps at a time. -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | k + 3 => climbStairs k + climbStairs (k + 1) + climbStairs (k + 2)

/-- The number of steps in the staircase -/
def numSteps : ℕ := 10

/-- Theorem stating that there are 274 ways to climb a 10-step staircase -/
theorem staircase_climbing_ways : climbStairs numSteps = 274 := by
  sorry

end staircase_climbing_ways_l2078_207817


namespace train_bridge_crossing_time_l2078_207826

/-- Calculates the time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (bridge_length : ℝ) 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (h1 : bridge_length = 200)
  (h2 : train_length = 100)
  (h3 : train_speed = 5) : 
  (bridge_length + train_length) / train_speed = 60 := by
  sorry

end train_bridge_crossing_time_l2078_207826


namespace total_amount_distributed_l2078_207878

-- Define the shares of A, B, and C
def share_A : ℕ := sorry
def share_B : ℕ := sorry
def share_C : ℕ := 495

-- Define the amounts to be decreased
def decrease_A : ℕ := 25
def decrease_B : ℕ := 10
def decrease_C : ℕ := 15

-- Define the ratio of remaining amounts
def ratio_A : ℕ := 3
def ratio_B : ℕ := 2
def ratio_C : ℕ := 5

-- Theorem to prove
theorem total_amount_distributed :
  share_A + share_B + share_C = 1010 :=
by
  sorry

-- Lemma to ensure the ratio condition is met
lemma ratio_condition :
  (share_A - decrease_A) * ratio_B * ratio_C = 
  (share_B - decrease_B) * ratio_A * ratio_C ∧
  (share_B - decrease_B) * ratio_A * ratio_C = 
  (share_C - decrease_C) * ratio_A * ratio_B :=
by
  sorry

end total_amount_distributed_l2078_207878


namespace inequality_proof_l2078_207890

theorem inequality_proof (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (h_prod : a * b * c * d * e = 1) : 
  (d * e) / (a * (b + 1)) + (e * a) / (b * (c + 1)) + 
  (a * b) / (c * (d + 1)) + (b * c) / (d * (e + 1)) + 
  (c * d) / (e * (a + 1)) ≥ 5 / 2 := by
  sorry

end inequality_proof_l2078_207890


namespace star_value_l2078_207860

-- Define the * operation
def star (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 2010

-- State the theorem
theorem star_value (a b : ℝ) :
  (star a b 3 5 = 2011) → (star a b 4 9 = 2009) → (star a b 1 2 = 2010) := by
  sorry

end star_value_l2078_207860


namespace chord_length_concentric_circles_l2078_207880

/-- Given two concentric circles with radii A and B (A > B), if the area between
    the circles is 15π square meters, then the length of a chord of the larger
    circle that is tangent to the smaller circle is 2√15 meters. -/
theorem chord_length_concentric_circles (A B : ℝ) (h1 : A > B) (h2 : A > 0) (h3 : B > 0)
    (h4 : π * A^2 - π * B^2 = 15 * π) :
    ∃ (c : ℝ), c^2 = 4 * 15 ∧ c > 0 := by
  sorry

end chord_length_concentric_circles_l2078_207880


namespace partial_fraction_decomposition_l2078_207831

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    (∀ x : ℚ, x ≠ 2 ∧ x ≠ 4 →
      (3 * x + 1) / ((x - 4) * (x - 2)^2) =
      P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) ∧
    P = 13/4 ∧ Q = -13/4 ∧ R = -7/2 := by
  sorry

end partial_fraction_decomposition_l2078_207831


namespace opposite_of_negative_five_l2078_207882

theorem opposite_of_negative_five : -((-5) : ℤ) = 5 := by
  sorry

end opposite_of_negative_five_l2078_207882


namespace intersection_of_tangents_l2078_207820

/-- A curve defined by y = x + 1/x for x > 0 -/
def C : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1 / p.1 ∧ p.1 > 0}

/-- A line passing through (0,1) with slope k -/
def line (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + 1}

/-- The intersection points of the line with the curve C -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) := C ∩ line k

/-- The tangent line to C at a point (x, y) -/
def tangent_line (x : ℝ) : Set (ℝ × ℝ) := 
  {p | p.2 = (1 - 1/x^2) * p.1 + 2/x}

theorem intersection_of_tangents (k : ℝ) :
  ∀ M N : ℝ × ℝ, M ∈ intersection_points k → N ∈ intersection_points k → M ≠ N →
  ∃ P : ℝ × ℝ, P ∈ tangent_line M.1 ∧ P ∈ tangent_line N.1 ∧ 
  P.1 = 2 ∧ 2 < P.2 ∧ P.2 < 2.5 :=
sorry

end intersection_of_tangents_l2078_207820


namespace ezekiel_painted_faces_l2078_207806

/-- The number of faces on a cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids painted -/
def num_cuboids : ℕ := 5

/-- The total number of faces painted by Ezekiel -/
def total_faces_painted : ℕ := faces_per_cuboid * num_cuboids

theorem ezekiel_painted_faces :
  total_faces_painted = 30 :=
by sorry

end ezekiel_painted_faces_l2078_207806


namespace awards_distribution_theorem_l2078_207835

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 6 awards to 3 students results in 465 ways -/
theorem awards_distribution_theorem :
  distribute_awards 6 3 = 465 :=
by sorry

end awards_distribution_theorem_l2078_207835


namespace inverse_of_singular_matrix_l2078_207805

def A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; 8, -4]

theorem inverse_of_singular_matrix :
  Matrix.det A = 0 → A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end inverse_of_singular_matrix_l2078_207805


namespace cake_recipe_difference_l2078_207894

theorem cake_recipe_difference (total_flour total_sugar flour_added : ℕ) :
  total_flour = 9 →
  total_sugar = 6 →
  flour_added = 2 →
  (total_flour - flour_added) - total_sugar = 1 := by
sorry

end cake_recipe_difference_l2078_207894


namespace total_hours_worked_is_48_l2078_207822

/-- Calculates the total hours worked given basic pay, overtime rate, and total wage -/
def totalHoursWorked (basicPay : ℚ) (basicHours : ℚ) (overtimeRate : ℚ) (totalWage : ℚ) : ℚ :=
  let basicHourlyRate := basicPay / basicHours
  let overtimeHourlyRate := basicHourlyRate * (1 + overtimeRate)
  let overtimePay := totalWage - basicPay
  let overtimeHours := overtimePay / overtimeHourlyRate
  basicHours + overtimeHours

/-- Theorem stating that under given conditions, the total hours worked is 48 -/
theorem total_hours_worked_is_48 :
  totalHoursWorked 20 40 (1/4) 25 = 48 := by
  sorry

end total_hours_worked_is_48_l2078_207822


namespace bus_trip_speed_l2078_207819

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) 
  (h1 : distance = 280)
  (h2 : speed_increase = 5)
  (h3 : time_decrease = 1)
  (h4 : distance / speed - time_decrease = distance / (speed + speed_increase)) :
  speed = 35 := by
  sorry

end bus_trip_speed_l2078_207819


namespace trigonometric_sum_simplification_l2078_207816

open Real BigOperators

theorem trigonometric_sum_simplification (n : ℕ) (α : ℝ) :
  (cos α + ∑ k in Finset.range (n - 1), (n.choose k) * cos ((k + 1) * α) + cos ((n + 1) * α) = 
   2^n * (cos (α / 2))^n * cos ((n + 2) * α / 2)) ∧
  (sin α + ∑ k in Finset.range (n - 1), (n.choose k) * sin ((k + 1) * α) + sin ((n + 1) * α) = 
   2^n * (cos (α / 2))^n * sin ((n + 2) * α / 2)) := by
  sorry

end trigonometric_sum_simplification_l2078_207816


namespace library_books_problem_l2078_207887

theorem library_books_problem (initial_books : ℕ) : 
  initial_books - 227 + 56 - 35 = 29 → initial_books = 235 :=
by sorry

end library_books_problem_l2078_207887


namespace common_difference_unique_l2078_207893

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_correct : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_unique
  (seq : ArithmeticSequence)
  (h1 : seq.a 3 = 3)
  (h2 : seq.S 4 = 14) :
  common_difference seq = -1 := by
  sorry

end common_difference_unique_l2078_207893


namespace merry_go_round_revolutions_l2078_207838

/-- The number of revolutions needed for the second horse to cover the same distance as the first horse on a merry-go-round. -/
theorem merry_go_round_revolutions (r₁ r₂ : ℝ) (n₁ : ℕ) (h₁ : r₁ = 30) (h₂ : r₂ = 10) (h₃ : n₁ = 25) :
  (r₁ * n₁ : ℝ) / r₂ = 75 := by
  sorry

end merry_go_round_revolutions_l2078_207838


namespace square_inequality_l2078_207842

theorem square_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 3 / (x - y)) : x^2 > y^2 + 6 := by
  sorry

end square_inequality_l2078_207842


namespace fraction_sum_l2078_207813

theorem fraction_sum : 2/5 + 4/50 + 3/500 + 8/5000 = 0.4876 := by
  sorry

end fraction_sum_l2078_207813


namespace set_A_equality_l2078_207828

def A : Set ℕ := {x | x < 3}

theorem set_A_equality : A = {0, 1, 2} := by sorry

end set_A_equality_l2078_207828


namespace problem_statement_l2078_207830

-- Define the base 10 logarithm
noncomputable def log (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem problem_statement : (2/3)^0 + log 2 + log 5 = 2 := by sorry

end problem_statement_l2078_207830


namespace p_or_q_is_true_l2078_207888

-- Define proposition p
def p : Prop := ∀ (x a : ℝ), x^2 + a*x + a^2 ≥ 0

-- Define proposition q
def q : Prop := ∃ (x₀ : ℕ), x₀ > 0 ∧ 2*x₀^2 - 1 ≤ 0

-- Theorem statement
theorem p_or_q_is_true : p ∨ q := by sorry

end p_or_q_is_true_l2078_207888


namespace workers_per_block_l2078_207823

/-- Given a company with the following conditions:
  - The total amount for gifts is $6000
  - Each gift costs $2
  - There are 15 blocks in the company
  This theorem proves that there are 200 workers in each block. -/
theorem workers_per_block (total_amount : ℕ) (gift_worth : ℕ) (num_blocks : ℕ)
  (h1 : total_amount = 6000)
  (h2 : gift_worth = 2)
  (h3 : num_blocks = 15) :
  total_amount / gift_worth / num_blocks = 200 := by
  sorry

end workers_per_block_l2078_207823


namespace right_angled_triangle_l2078_207804

theorem right_angled_triangle (A B C : Real) (h1 : 0 ≤ A ∧ A ≤ π) (h2 : 0 ≤ B ∧ B ≤ π) (h3 : 0 ≤ C ∧ C ≤ π) 
  (h4 : A + B + C = π) (h5 : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 2) : 
  A = π/2 ∨ B = π/2 ∨ C = π/2 := by
  sorry

end right_angled_triangle_l2078_207804


namespace system_solutions_l2078_207812

/-- The system of equations has only two solutions -/
theorem system_solutions (x y z : ℝ) : 
  (2 * x^2 / (1 + x^2) = y ∧ 
   2 * y^2 / (1 + y^2) = z ∧ 
   2 * z^2 / (1 + z^2) = x) → 
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1)) :=
by sorry

end system_solutions_l2078_207812


namespace inequality_proof_l2078_207858

theorem inequality_proof (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  1 ≤ (x / (1 + y*z) + y / (1 + z*x) + z / (1 + x*y)) ∧ 
  (x / (1 + y*z) + y / (1 + z*x) + z / (1 + x*y)) ≤ Real.sqrt 2 := by
  sorry

end inequality_proof_l2078_207858


namespace total_nails_formula_nails_for_40_per_side_l2078_207886

/-- The number of nails used to fix a square metal plate -/
def total_nails (nails_per_side : ℕ) : ℕ :=
  4 * nails_per_side - 4

/-- Theorem: The total number of nails used is equal to 4 times the number of nails on one side, minus 4 -/
theorem total_nails_formula (nails_per_side : ℕ) :
  total_nails nails_per_side = 4 * nails_per_side - 4 := by
  sorry

/-- Corollary: For a square with 40 nails on each side, the total number of nails used is 156 -/
theorem nails_for_40_per_side :
  total_nails 40 = 156 := by
  sorry

end total_nails_formula_nails_for_40_per_side_l2078_207886


namespace multiplication_equation_l2078_207825

theorem multiplication_equation :
  ∀ (multiplier multiplicand product : ℕ),
    multiplier = 6 →
    multiplicand = product - 140 →
    multiplier * multiplicand = product →
    (multiplier = 6 ∧ multiplicand = 28 ∧ product = 168) :=
by
  sorry

end multiplication_equation_l2078_207825


namespace race_distance_multiple_of_360_l2078_207853

/-- Represents a race between two contestants A and B -/
structure Race where
  speedRatio : Rat  -- Ratio of speeds of A to B
  headStart : ℕ     -- Head start distance for A in meters
  winMargin : ℕ     -- Distance by which A wins in meters

/-- The total distance of the race is a multiple of 360 meters -/
theorem race_distance_multiple_of_360 (race : Race) 
  (h1 : race.speedRatio = 3 / 4)
  (h2 : race.headStart = 140)
  (h3 : race.winMargin = 20) :
  ∃ (k : ℕ), race.headStart + race.winMargin + k * 360 = 
    race.headStart + (4 * (race.headStart + race.winMargin)) / 3 :=
sorry

end race_distance_multiple_of_360_l2078_207853


namespace cone_radius_from_melted_cylinder_l2078_207869

/-- The radius of a cone formed by melting a cylinder -/
theorem cone_radius_from_melted_cylinder (r_cylinder h_cylinder h_cone : ℝ) 
  (h_r : r_cylinder = 8)
  (h_h_cylinder : h_cylinder = 2)
  (h_h_cone : h_cone = 6) : 
  ∃ (r_cone : ℝ), r_cone = 8 ∧ 
  (π * r_cylinder^2 * h_cylinder = (1/3) * π * r_cone^2 * h_cone) :=
by
  sorry

#check cone_radius_from_melted_cylinder

end cone_radius_from_melted_cylinder_l2078_207869


namespace decreasing_interval_of_f_l2078_207811

/-- The cubic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 5 * x^2 + 3 * x - 2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 10 * x + 3

theorem decreasing_interval_of_f (a : ℝ) :
  (f' a 3 = 0) →
  (∀ x : ℝ, x ∈ Set.Icc (1/3 : ℝ) 3 ↔ f' a x ≤ 0) :=
sorry

end decreasing_interval_of_f_l2078_207811


namespace tangent_slope_range_implies_y_coordinate_range_l2078_207834

/-- The curve C defined by y = x^2 - x + 1 -/
def C : ℝ → ℝ := λ x => x^2 - x + 1

/-- The derivative of C -/
def C' : ℝ → ℝ := λ x => 2*x - 1

theorem tangent_slope_range_implies_y_coordinate_range :
  ∀ x y : ℝ,
  y = C x →
  -1 ≤ C' x ∧ C' x ≤ 3 →
  3/4 ≤ y ∧ y ≤ 3 := by sorry

end tangent_slope_range_implies_y_coordinate_range_l2078_207834


namespace power_division_sum_product_l2078_207885

theorem power_division_sum_product : (-6)^6 / 6^4 + 4^3 - 7^2 * 2 = 2 := by
  sorry

end power_division_sum_product_l2078_207885


namespace unique_solution_for_system_l2078_207897

theorem unique_solution_for_system :
  ∃! (x y : ℝ), (x + y = (7 - x) + (7 - y)) ∧ (x - y = (x + 1) + (y + 1)) ∧ x = 8 ∧ y = -1 := by
  sorry

end unique_solution_for_system_l2078_207897


namespace shekars_science_marks_l2078_207849

theorem shekars_science_marks
  (math_marks : ℕ)
  (social_marks : ℕ)
  (english_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (num_subjects : ℕ)
  (h1 : math_marks = 76)
  (h2 : social_marks = 82)
  (h3 : english_marks = 67)
  (h4 : biology_marks = 75)
  (h5 : average_marks = 73)
  (h6 : num_subjects = 5) :
  ∃ (science_marks : ℕ),
    (math_marks + social_marks + english_marks + biology_marks + science_marks) / num_subjects = average_marks ∧
    science_marks = 65 := by
  sorry

end shekars_science_marks_l2078_207849


namespace complement_determines_interval_l2078_207855

-- Define the set A
def A (a b : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ b }

-- Define the complement of A
def C_U_A : Set ℝ := { x | x > 4 ∨ x < 3 }

-- Theorem statement
theorem complement_determines_interval :
  ∃ (a b : ℝ), A a b = (C_U_A)ᶜ ∧ a = 3 ∧ b = 4 := by
  sorry

end complement_determines_interval_l2078_207855


namespace quadratic_equation_root_l2078_207844

theorem quadratic_equation_root (b : ℝ) :
  (∃ x : ℝ, 2 * x^2 + b * x - 119 = 0) ∧ (2 * 7^2 + b * 7 - 119 = 0) →
  b = 3 := by
  sorry

end quadratic_equation_root_l2078_207844


namespace angle_D_value_l2078_207856

theorem angle_D_value (A B C D : ℝ) 
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 2 * D - 10) :
  D = 70 := by
sorry

end angle_D_value_l2078_207856


namespace family_savings_calculation_l2078_207836

def tax_rate : Float := 0.13

def ivan_salary : Float := 55000
def vasilisa_salary : Float := 45000
def mother_salary : Float := 18000
def father_salary : Float := 20000
def son_scholarship : Float := 3000
def mother_pension : Float := 10000
def son_extra_scholarship : Float := 15000

def monthly_expenses : Float := 74000

def net_income (gross_income : Float) : Float :=
  gross_income * (1 - tax_rate)

def total_income_before_may2018 : Float :=
  net_income ivan_salary + net_income vasilisa_salary + 
  net_income mother_salary + net_income father_salary + son_scholarship

def total_income_may_to_aug2018 : Float :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + son_scholarship

def total_income_from_sept2018 : Float :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + 
  son_scholarship + net_income son_extra_scholarship

theorem family_savings_calculation :
  (total_income_before_may2018 - monthly_expenses = 49060) ∧
  (total_income_may_to_aug2018 - monthly_expenses = 43400) ∧
  (total_income_from_sept2018 - monthly_expenses = 56450) := by
  sorry

end family_savings_calculation_l2078_207836


namespace point_on_terminal_side_l2078_207847

/-- Given a point P(m, m+1) on the terminal side of angle α where cos(α) = 3/5, prove that m = 3 -/
theorem point_on_terminal_side (m : ℝ) (α : ℝ) : 
  (∃ P : ℝ × ℝ, P = (m, m + 1) ∧ P.1 / Real.sqrt (P.1^2 + P.2^2) = 3/5) → 
  m = 3 := by
  sorry

end point_on_terminal_side_l2078_207847


namespace subtract_decimals_l2078_207818

theorem subtract_decimals : 34.25 - 0.45 = 33.8 := by
  sorry

end subtract_decimals_l2078_207818


namespace quadratic_root_coefficient_relation_l2078_207865

/-- 
For a quadratic equation x^2 + px + q = 0 with roots α and β, 
this theorem states the relationship between the roots and the coefficients.
-/
theorem quadratic_root_coefficient_relation (p q α β : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 ↔ x = α ∨ x = β) → 
  (α + β = -p ∧ α * β = q) := by
  sorry

end quadratic_root_coefficient_relation_l2078_207865


namespace employee_discount_price_l2078_207837

theorem employee_discount_price (wholesale_cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) : 
  wholesale_cost = 200 →
  markup_percentage = 0.20 →
  discount_percentage = 0.25 →
  let retail_price := wholesale_cost * (1 + markup_percentage)
  let discounted_price := retail_price * (1 - discount_percentage)
  discounted_price = 180 := by
sorry


end employee_discount_price_l2078_207837


namespace least_product_of_distinct_primes_above_50_l2078_207850

theorem least_product_of_distinct_primes_above_50 :
  ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    p > 50 ∧ q > 50 ∧ 
    p ≠ q ∧
    p * q = 3127 ∧
    ∀ (r s : ℕ), Prime r → Prime s → r > 50 → s > 50 → r ≠ s → r * s ≥ p * q :=
by sorry

end least_product_of_distinct_primes_above_50_l2078_207850


namespace arccos_sin_one_point_five_l2078_207832

theorem arccos_sin_one_point_five :
  Real.arccos (Real.sin 1.5) = π / 2 - 1.5 := by
  sorry

end arccos_sin_one_point_five_l2078_207832


namespace distinct_naturals_reciprocal_sum_l2078_207898

theorem distinct_naturals_reciprocal_sum (x y z : ℕ) : 
  x < y ∧ y < z ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  ∃ (a : ℕ), (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = a
  →
  x = 2 ∧ y = 3 ∧ z = 6 := by
sorry

end distinct_naturals_reciprocal_sum_l2078_207898


namespace money_left_after_purchase_l2078_207854

def calculate_money_left (initial_amount : ℝ) (candy_bars : ℕ) (chips : ℕ) (soft_drinks : ℕ)
  (candy_bar_price : ℝ) (chips_price : ℝ) (soft_drink_price : ℝ)
  (candy_discount : ℝ) (chips_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let candy_cost := candy_bars * candy_bar_price
  let chips_cost := chips * chips_price
  let soft_drinks_cost := soft_drinks * soft_drink_price
  let total_before_discounts := candy_cost + chips_cost + soft_drinks_cost
  let candy_discount_amount := candy_cost * candy_discount
  let chips_discount_amount := chips_cost * chips_discount
  let total_after_discounts := total_before_discounts - candy_discount_amount - chips_discount_amount
  let tax_amount := total_after_discounts * sales_tax
  let final_cost := total_after_discounts + tax_amount
  initial_amount - final_cost

theorem money_left_after_purchase :
  calculate_money_left 200 25 10 15 3 2.5 1.75 0.1 0.05 0.06 = 75.45 := by
  sorry

end money_left_after_purchase_l2078_207854


namespace cubic_roots_product_l2078_207895

theorem cubic_roots_product (a b c : ℝ) : 
  (x^3 - 26*x^2 + 32*x - 15 = 0 → x = a ∨ x = b ∨ x = c) →
  (1 + a) * (1 + b) * (1 + c) = 74 := by
sorry

end cubic_roots_product_l2078_207895


namespace time_sum_after_duration_l2078_207824

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds a duration to a given time and returns the resulting time -/
def addDuration (initial : Time) (durationHours durationMinutes durationSeconds : Nat) : Time :=
  sorry

/-- Converts a Time to its 12-hour clock representation -/
def to12HourClock (t : Time) : Time :=
  sorry

/-- Calculates the sum of digits in a Time -/
def sumDigits (t : Time) : Nat :=
  sorry

theorem time_sum_after_duration :
  let initialTime : Time := ⟨15, 0, 0⟩  -- 3:00:00 PM
  let finalTime := to12HourClock (addDuration initialTime 317 58 33)
  sumDigits finalTime = 99 := by
  sorry

end time_sum_after_duration_l2078_207824


namespace no_function_satisfies_equation_l2078_207810

theorem no_function_satisfies_equation :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f (2 * f x) = x + 1998 := by
  sorry

end no_function_satisfies_equation_l2078_207810


namespace adult_ticket_cost_l2078_207814

theorem adult_ticket_cost 
  (total_seats : ℕ) 
  (child_ticket_cost : ℚ) 
  (num_children : ℕ) 
  (total_revenue : ℚ) 
  (h1 : total_seats = 250) 
  (h2 : child_ticket_cost = 4) 
  (h3 : num_children = 188) 
  (h4 : total_revenue = 1124) :
  let num_adults : ℕ := total_seats - num_children
  let adult_ticket_cost : ℚ := (total_revenue - (↑num_children * child_ticket_cost)) / ↑num_adults
  adult_ticket_cost = 6 := by
sorry

end adult_ticket_cost_l2078_207814


namespace second_graders_count_l2078_207840

/-- The number of second graders wearing blue shirts -/
def second_graders : ℕ := sorry

/-- The cost of a blue shirt for second graders -/
def blue_shirt_cost : ℚ := 560 / 100

/-- The number of kindergartners -/
def kindergartners : ℕ := 101

/-- The cost of an orange shirt for kindergartners -/
def orange_shirt_cost : ℚ := 580 / 100

/-- The number of first graders -/
def first_graders : ℕ := 113

/-- The cost of a yellow shirt for first graders -/
def yellow_shirt_cost : ℚ := 500 / 100

/-- The number of third graders -/
def third_graders : ℕ := 108

/-- The cost of a green shirt for third graders -/
def green_shirt_cost : ℚ := 525 / 100

/-- The total amount spent on all shirts -/
def total_spent : ℚ := 231700 / 100

/-- Theorem stating that the number of second graders wearing blue shirts is 107 -/
theorem second_graders_count : second_graders = 107 := by
  sorry

end second_graders_count_l2078_207840


namespace flea_misses_point_l2078_207803

/-- Represents the number of points on the circle. -/
def n : ℕ := 101

/-- Represents the position of the flea after k jumps. -/
def flea_position (k : ℕ) : ℕ := (k * (k + 1) / 2) % n

/-- States that there exists a point that the flea never lands on. -/
theorem flea_misses_point : ∃ p : Fin n, ∀ k : ℕ, flea_position k ≠ p.val :=
sorry

end flea_misses_point_l2078_207803


namespace basic_computer_price_l2078_207829

/-- Given the price of a basic computer and a printer, prove that the basic computer costs $2000 -/
theorem basic_computer_price
  (basic_computer printer : ℝ)
  (total_price : basic_computer + printer = 2500)
  (enhanced_total : ∃ (enhanced_total : ℝ), enhanced_total = basic_computer + 500 + printer)
  (printer_ratio : printer = (1/6) * (basic_computer + 500 + printer)) :
  basic_computer = 2000 := by
  sorry

end basic_computer_price_l2078_207829


namespace solution_set_part1_range_of_a_part2_l2078_207857

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} := by sorry

end solution_set_part1_range_of_a_part2_l2078_207857


namespace jennifer_total_distance_l2078_207889

/-- Represents the distances and changes for Jennifer's museum visits -/
structure MuseumDistances where
  first_museum : ℕ
  second_museum : ℕ
  cultural_center : ℕ
  traffic_increase : ℕ
  bus_decrease : ℕ
  bicycle_decrease : ℕ

/-- Calculates the total distance for Jennifer's museum visits -/
def total_distance (d : MuseumDistances) : ℕ :=
  (d.second_museum + d.traffic_increase) + 
  (d.cultural_center - d.bus_decrease) + 
  (d.first_museum - d.bicycle_decrease)

/-- Theorem stating that Jennifer's total distance is 32 miles -/
theorem jennifer_total_distance :
  ∀ d : MuseumDistances,
  d.first_museum = 5 ∧
  d.second_museum = 15 ∧
  d.cultural_center = 10 ∧
  d.traffic_increase = 5 ∧
  d.bus_decrease = 2 ∧
  d.bicycle_decrease = 1 →
  total_distance d = 32 :=
by sorry

end jennifer_total_distance_l2078_207889


namespace power_30_mod_7_l2078_207868

theorem power_30_mod_7 : 2^30 ≡ 1 [MOD 7] :=
by
  have h : 2^3 ≡ 1 [MOD 7] := by sorry
  sorry

end power_30_mod_7_l2078_207868


namespace total_silver_dollars_l2078_207879

/-- The number of silver dollars owned by Mr. Chiu -/
def chiu_dollars : ℕ := 56

/-- The number of silver dollars owned by Mr. Phung -/
def phung_dollars : ℕ := chiu_dollars + 16

/-- The number of silver dollars owned by Mr. Ha -/
def ha_dollars : ℕ := phung_dollars + 5

/-- The total number of silver dollars owned by all three -/
def total_dollars : ℕ := chiu_dollars + phung_dollars + ha_dollars

theorem total_silver_dollars : total_dollars = 205 := by
  sorry

end total_silver_dollars_l2078_207879


namespace translation_vector_exponential_l2078_207852

/-- Given two functions f and g, where f(x) = 2^x + 1 and g(x) = 2^(x+1),
    prove that the translation vector (h, k) that transforms the graph of f
    into the graph of g is (-1, -1). -/
theorem translation_vector_exponential (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 2^x + 1)
  (hg : ∀ x, g x = 2^(x+1))
  (h k : ℝ)
  (translation : ∀ x, g x = f (x - h) + k) :
  h = -1 ∧ k = -1 := by
  sorry

end translation_vector_exponential_l2078_207852


namespace orange_packing_l2078_207892

theorem orange_packing (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 94) (h2 : oranges_per_box = 8) :
  (total_oranges + oranges_per_box - 1) / oranges_per_box = 12 := by
  sorry

end orange_packing_l2078_207892


namespace max_choir_members_choir_of_120_exists_l2078_207896

/-- Represents a choir formation --/
structure ChoirFormation where
  rows : ℕ
  members_per_row : ℕ

/-- Represents the choir and its formations --/
structure Choir where
  total_members : ℕ
  original_formation : ChoirFormation
  new_formation : ChoirFormation

/-- The conditions of the choir problem --/
def choir_conditions (c : Choir) : Prop :=
  c.total_members < 120 ∧
  c.total_members = c.original_formation.rows * c.original_formation.members_per_row + 3 ∧
  c.total_members = (c.original_formation.rows - 1) * (c.original_formation.members_per_row + 2)

/-- The theorem stating the maximum number of choir members --/
theorem max_choir_members :
  ∀ c : Choir, choir_conditions c → c.total_members ≤ 120 :=
by sorry

/-- The theorem stating that 120 is achievable --/
theorem choir_of_120_exists :
  ∃ c : Choir, choir_conditions c ∧ c.total_members = 120 :=
by sorry

end max_choir_members_choir_of_120_exists_l2078_207896


namespace camping_trip_attendance_l2078_207877

/-- The percentage of students who went to the camping trip -/
def camping_trip_percentage : ℝ := 14

/-- The percentage of students who went to the music festival -/
def music_festival_percentage : ℝ := 8

/-- The percentage of students who participated in the sports league -/
def sports_league_percentage : ℝ := 6

/-- The percentage of camping trip attendees who spent more than $100 -/
def camping_trip_high_cost_percentage : ℝ := 60

/-- The percentage of music festival attendees who spent more than $90 -/
def music_festival_high_cost_percentage : ℝ := 80

/-- The percentage of sports league participants who paid more than $70 -/
def sports_league_high_cost_percentage : ℝ := 75

theorem camping_trip_attendance : 
  camping_trip_percentage = 14 := by sorry

end camping_trip_attendance_l2078_207877


namespace solution_in_interval_l2078_207801

theorem solution_in_interval (x₀ : ℝ) : 
  (Real.log x₀ + x₀ - 3 = 0) → (2 < x₀ ∧ x₀ < 2.5) := by
  sorry

end solution_in_interval_l2078_207801


namespace crash_prob_equal_l2078_207841

-- Define the probability of an engine failing
variable (p : ℝ)

-- Define the probability of crashing for the 3-engine plane
def crash_prob_3 (p : ℝ) : ℝ := 3 * p^2 * (1 - p) + p^3

-- Define the probability of crashing for the 5-engine plane
def crash_prob_5 (p : ℝ) : ℝ := 10 * p^3 * (1 - p)^2 + 5 * p^4 * (1 - p) + p^5

-- Theorem stating that the crash probabilities are equal for p = 0, 1/2, and 1
theorem crash_prob_equal : 
  (crash_prob_3 0 = crash_prob_5 0) ∧ 
  (crash_prob_3 (1/2) = crash_prob_5 (1/2)) ∧ 
  (crash_prob_3 1 = crash_prob_5 1) :=
sorry

end crash_prob_equal_l2078_207841


namespace symmetric_curve_l2078_207802

/-- The equation of a curve symmetric to y^2 = 4x with respect to the line x = 2 -/
theorem symmetric_curve (x y : ℝ) : 
  (∀ x₀ y₀ : ℝ, y₀^2 = 4*x₀ → (4 - x₀)^2 = 4*(2 - (4 - x₀))) → 
  y^2 = 16 - 4*x :=
sorry

end symmetric_curve_l2078_207802


namespace range_of_a_l2078_207851

/-- The range of a given the conditions in the problem -/
theorem range_of_a (p q : Prop) (a : ℝ) 
  (hp : p ↔ ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0)
  (hq : q ↔ ∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0)
  (hpq_or : p ∨ q)
  (hpq_not_and : ¬(p ∧ q)) :
  a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 3 := by
  sorry


end range_of_a_l2078_207851


namespace last_four_average_l2078_207833

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 70 →
  ((list.take 3).sum / 3 : ℝ) = 65 →
  ((list.drop 3).sum / 4 : ℝ) = 73.75 := by
  sorry

end last_four_average_l2078_207833


namespace tree_purchase_equations_l2078_207873

/-- Represents the cost of an A-type tree -/
def cost_A : ℕ := 100

/-- Represents the cost of a B-type tree -/
def cost_B : ℕ := 80

/-- Represents the total amount spent -/
def total_spent : ℕ := 8000

/-- Represents the difference in number between A-type and B-type trees -/
def tree_difference : ℕ := 8

theorem tree_purchase_equations (x y : ℕ) :
  (x - y = tree_difference ∧ cost_A * x + cost_B * y = total_spent) ↔
  (x - y = 8 ∧ 100 * x + 80 * y = 8000) :=
sorry

end tree_purchase_equations_l2078_207873


namespace complex_division_sum_l2078_207846

theorem complex_division_sum (a b : ℝ) : 
  (Complex.I + 1) / (Complex.I - 1) = Complex.mk a b → a + b = 1 := by
  sorry

end complex_division_sum_l2078_207846


namespace complement_of_intersection_l2078_207881

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_of_intersection (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4})
  (hM : M = {1, 2, 3})
  (hN : N = {2, 3, 4}) :
  (U \ (M ∩ N)) = {1, 4} := by
  sorry

end complement_of_intersection_l2078_207881


namespace gcd_of_45_and_75_l2078_207839

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_of_45_and_75_l2078_207839


namespace largest_possible_a_l2078_207809

theorem largest_possible_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 50) : 
  a ≤ 2924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 2924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 :=
by sorry

end largest_possible_a_l2078_207809
