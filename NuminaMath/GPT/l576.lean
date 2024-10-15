import Mathlib

namespace NUMINAMATH_GPT_compute_expression_l576_57697

theorem compute_expression (x : ℤ) (h : x = 6) :
  ((x^9 - 24 * x^6 + 144 * x^3 - 512) / (x^3 - 8) = 43264) :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l576_57697


namespace NUMINAMATH_GPT_pages_with_same_units_digit_count_l576_57621

theorem pages_with_same_units_digit_count {n : ℕ} (h1 : n = 67) :
  ∃ k : ℕ, k = 13 ∧
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ n → 
    (x ≡ (n + 1 - x) [MOD 10] ↔ 
     (x % 10 = 4 ∨ x % 10 = 9))) :=
by
  sorry

end NUMINAMATH_GPT_pages_with_same_units_digit_count_l576_57621


namespace NUMINAMATH_GPT_factor_expression_l576_57611

variable (a : ℤ)

theorem factor_expression : 58 * a^2 + 174 * a = 58 * a * (a + 3) := by
  sorry

end NUMINAMATH_GPT_factor_expression_l576_57611


namespace NUMINAMATH_GPT_find_number_l576_57644

theorem find_number (x : ℚ) : (35 / 100) * x = (20 / 100) * 50 → x = 200 / 7 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_number_l576_57644


namespace NUMINAMATH_GPT_fill_tank_time_l576_57677

-- Definitions based on provided conditions
def pipeA_time := 60 -- Pipe A fills the tank in 60 minutes
def pipeB_time := 40 -- Pipe B fills the tank in 40 minutes

-- Theorem statement
theorem fill_tank_time (T : ℕ) : 
  (T / 2) / pipeB_time + (T / 2) * (1 / pipeA_time + 1 / pipeB_time) = 1 → 
  T = 48 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_fill_tank_time_l576_57677


namespace NUMINAMATH_GPT_Portia_school_students_l576_57629

theorem Portia_school_students:
  ∃ (P L : ℕ), P = 2 * L ∧ P + L = 3000 ∧ P = 2000 :=
by
  sorry

end NUMINAMATH_GPT_Portia_school_students_l576_57629


namespace NUMINAMATH_GPT_minimum_participants_l576_57630

theorem minimum_participants
  (correct_first : ℕ)
  (correct_second : ℕ)
  (correct_third : ℕ)
  (correct_fourth : ℕ)
  (H_first : correct_first = 90)
  (H_second : correct_second = 50)
  (H_third : correct_third = 40)
  (H_fourth : correct_fourth = 20)
  (H_max_two : ∀ p : ℕ, 1 ≤ p ∧ p ≤ correct_first + correct_second + correct_third + correct_fourth → p ≤ 2 * (correct_first + correct_second + correct_third + correct_fourth))
  : ∃ n : ℕ, (correct_first + correct_second + correct_third + correct_fourth) / 2 = 100 :=
by
  sorry

end NUMINAMATH_GPT_minimum_participants_l576_57630


namespace NUMINAMATH_GPT_population_exceeds_l576_57672

theorem population_exceeds (n : ℕ) : (∃ n, 4 * 3^n > 200) ∧ ∀ m, m < n → 4 * 3^m ≤ 200 := by
  sorry

end NUMINAMATH_GPT_population_exceeds_l576_57672


namespace NUMINAMATH_GPT_b_l576_57622

def initial_marbles : Nat := 24
def lost_through_hole : Nat := 4
def given_away : Nat := 2 * lost_through_hole
def eaten_by_dog : Nat := lost_through_hole / 2

theorem b {m : Nat} (h₁ : m = initial_marbles - lost_through_hole)
  (h₂ : m - given_away = m₁)
  (h₃ : m₁ - eaten_by_dog = 10) :
  m₁ - eaten_by_dog = 10 := sorry

end NUMINAMATH_GPT_b_l576_57622


namespace NUMINAMATH_GPT_molecular_weight_correct_l576_57699

-- Define the atomic weights of the elements.
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms for each element in the compound.
def number_of_C : ℕ := 7
def number_of_H : ℕ := 6
def number_of_O : ℕ := 2

-- Define the molecular weight calculation.
def molecular_weight : ℝ := 
  (number_of_C * atomic_weight_C) +
  (number_of_H * atomic_weight_H) +
  (number_of_O * atomic_weight_O)

-- Step to prove that molecular weight is equal to 122.118 g/mol.
theorem molecular_weight_correct : molecular_weight = 122.118 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_correct_l576_57699


namespace NUMINAMATH_GPT_abc_sum_eq_sqrt34_l576_57675

noncomputable def abc_sum (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 16)
                          (h2 : ab + bc + ca = 9)
                          (h3 : a^2 + b^2 = 10)
                          (h4 : 0 ≤ a) (h5 : 0 ≤ b) (h6 : 0 ≤ c) : ℝ :=
a + b + c

theorem abc_sum_eq_sqrt34 (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 16)
  (h2 : ab + bc + ca = 9)
  (h3 : a^2 + b^2 = 10)
  (h4 : 0 ≤ a)
  (h5 : 0 ≤ b)
  (h6 : 0 ≤ c) :
  abc_sum a b c h1 h2 h3 h4 h5 h6 = Real.sqrt 34 :=
by
  sorry

end NUMINAMATH_GPT_abc_sum_eq_sqrt34_l576_57675


namespace NUMINAMATH_GPT_time_after_6666_seconds_l576_57610

noncomputable def initial_time : Nat := 3 * 3600
noncomputable def additional_seconds : Nat := 6666

-- Function to convert total seconds to "HH:MM:SS" format
def time_in_seconds (h m s : Nat) : Nat :=
  h*3600 + m*60 + s

noncomputable def new_time : Nat :=
  initial_time + additional_seconds

-- Convert the new total time back to "HH:MM:SS" format (expected: 4:51:06)
def hours (secs : Nat) : Nat := secs / 3600
def minutes (secs : Nat) : Nat := (secs % 3600) / 60
def seconds (secs : Nat) : Nat := (secs % 3600) % 60

theorem time_after_6666_seconds :
  hours new_time = 4 ∧ minutes new_time = 51 ∧ seconds new_time = 6 :=
by
  sorry

end NUMINAMATH_GPT_time_after_6666_seconds_l576_57610


namespace NUMINAMATH_GPT_find_ratio_squares_l576_57649

variables (x y z a b c : ℝ)

theorem find_ratio_squares 
  (h1 : x / a + y / b + z / c = 5) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := 
sorry

end NUMINAMATH_GPT_find_ratio_squares_l576_57649


namespace NUMINAMATH_GPT_A_alone_days_l576_57617

variable (x : ℝ) -- Number of days A takes to do the work alone
variable (B_rate : ℝ := 1 / 12) -- Work rate of B
variable (Together_rate : ℝ := 1 / 4) -- Combined work rate of A and B

theorem A_alone_days :
  (1 / x + B_rate = Together_rate) → (x = 6) := by
  intro h
  sorry

end NUMINAMATH_GPT_A_alone_days_l576_57617


namespace NUMINAMATH_GPT_three_digit_difference_l576_57635

theorem three_digit_difference (x : ℕ) (a b c : ℕ)
  (h1 : a = x + 2)
  (h2 : b = x + 1)
  (h3 : c = x)
  (h4 : a > b)
  (h5 : b > c) :
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = 198 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_difference_l576_57635


namespace NUMINAMATH_GPT_find_interest_rate_l576_57662

-- Define the given conditions
def initial_investment : ℝ := 2200
def additional_investment : ℝ := 1099.9999999999998
def total_investment : ℝ := initial_investment + additional_investment
def desired_income : ℝ := 0.06 * total_investment
def income_from_additional_investment : ℝ := 0.08 * additional_investment
def income_from_initial_investment (r : ℝ) : ℝ := initial_investment * r

-- State the proof problem
theorem find_interest_rate (r : ℝ) 
    (h : desired_income = income_from_additional_investment + income_from_initial_investment r) :
    r = 0.05 :=
sorry

end NUMINAMATH_GPT_find_interest_rate_l576_57662


namespace NUMINAMATH_GPT_tan_beta_l576_57698

open Real

variable (α β : ℝ)

theorem tan_beta (h₁ : tan α = 1/3) (h₂ : tan (α + β) = 1/2) : tan β = 1/7 :=
by sorry

end NUMINAMATH_GPT_tan_beta_l576_57698


namespace NUMINAMATH_GPT_find_jack_euros_l576_57657

theorem find_jack_euros (E : ℕ) (h1 : 45 + 2 * E = 117) : E = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_jack_euros_l576_57657


namespace NUMINAMATH_GPT_log_equality_ineq_l576_57607

--let a = \log_{\sqrt{5x-1}}(4x+1)
--let b = \log_{4x+1}\left(\frac{x}{2} + 2\right)^2
--let c = \log_{\frac{x}{2} + 2}(5x-1)

noncomputable def a (x : ℝ) : ℝ := 
  Real.log (4 * x + 1) / Real.log (Real.sqrt (5 * x - 1))

noncomputable def b (x : ℝ) : ℝ := 
  2 * (Real.log ((x / 2) + 2) / Real.log (4 * x + 1))

noncomputable def c (x : ℝ) : ℝ := 
  Real.log (5 * x - 1) / Real.log ((x / 2) + 2)

theorem log_equality_ineq (x : ℝ) : 
  a x = b x ∧ c x = a x - 1 ↔ x = 2 := 
by
  sorry

end NUMINAMATH_GPT_log_equality_ineq_l576_57607


namespace NUMINAMATH_GPT_number_of_ways_difference_of_squares_l576_57625

-- Lean statement
theorem number_of_ways_difference_of_squares (n k : ℕ) (h1 : n > 10^k) (h2 : n % 10^k = 0) (h3 : k ≥ 2) :
  ∃ D, D = k^2 - 1 ∧ ∀ (a b : ℕ), n = a^2 - b^2 → D = k^2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_difference_of_squares_l576_57625


namespace NUMINAMATH_GPT_probability_phone_not_answered_l576_57608

noncomputable def P_first_ring : ℝ := 0.1
noncomputable def P_second_ring : ℝ := 0.3
noncomputable def P_third_ring : ℝ := 0.4
noncomputable def P_fourth_ring : ℝ := 0.1

theorem probability_phone_not_answered : 
  1 - P_first_ring - P_second_ring - P_third_ring - P_fourth_ring = 0.1 := 
by
  sorry

end NUMINAMATH_GPT_probability_phone_not_answered_l576_57608


namespace NUMINAMATH_GPT_ratio_black_haired_children_l576_57637

theorem ratio_black_haired_children 
  (n_red : ℕ) (n_total : ℕ) (ratio_red : ℕ) (ratio_blonde : ℕ) (ratio_black : ℕ)
  (h_ratio : ratio_red / ratio_red = 1 ∧ ratio_blonde / ratio_red = 2 ∧ ratio_black / ratio_red = 7 / 3)
  (h_n_red : n_red = 9)
  (h_n_total : n_total = 48) :
  (7 : ℚ) / (16 : ℚ) = (n_total * 7 / 16 : ℚ) :=
sorry

end NUMINAMATH_GPT_ratio_black_haired_children_l576_57637


namespace NUMINAMATH_GPT_geometric_sequence_product_l576_57648

theorem geometric_sequence_product :
  ∀ (a : ℕ → ℝ), (∀ n, a n > 0) →
  (∃ (a_1 a_99 : ℝ), (a_1 + a_99 = 10) ∧ (a_1 * a_99 = 16) ∧ a 1 = a_1 ∧ a 99 = a_99) →
  a 20 * a 50 * a 80 = 64 :=
by
  intro a hpos hex
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l576_57648


namespace NUMINAMATH_GPT_rational_solutions_of_quadratic_l576_57606

theorem rational_solutions_of_quadratic (k : ℕ) (hk : 0 < k ∧ k ≤ 10) :
  ∃ (x : ℚ), k * x^2 + 20 * x + k = 0 ↔ (k = 6 ∨ k = 8 ∨ k = 10) :=
by sorry

end NUMINAMATH_GPT_rational_solutions_of_quadratic_l576_57606


namespace NUMINAMATH_GPT_fraction_reducible_to_17_l576_57627

theorem fraction_reducible_to_17 (m n : ℕ) (h_coprime : Nat.gcd m n = 1)
  (h_reducible : ∃ d : ℕ, d ∣ (3 * m - n) ∧ d ∣ (5 * n + 2 * m)) :
  ∃ k : ℕ, (3 * m - n) / k = 17 ∧ (5 * n + 2 * m) / k = 17 :=
by
  have key : Nat.gcd (3 * m - n) (5 * n + 2 * m) = 17 := sorry
  -- using the result we need to construct our desired k
  use 17 / (Nat.gcd (3 * m - n) (5 * n + 2 * m))
  -- rest of intimate proof here
  sorry

end NUMINAMATH_GPT_fraction_reducible_to_17_l576_57627


namespace NUMINAMATH_GPT_next_chime_time_l576_57631

theorem next_chime_time (chime1_interval : ℕ) (chime2_interval : ℕ) (chime3_interval : ℕ) (start_time : ℕ) 
  (h1 : chime1_interval = 18) (h2 : chime2_interval = 24) (h3 : chime3_interval = 30) (h4 : start_time = 9) : 
  ((start_time * 60 + 6 * 60) % (24 * 60)) / 60 = 15 :=
by
  sorry

end NUMINAMATH_GPT_next_chime_time_l576_57631


namespace NUMINAMATH_GPT_circle_radius_l576_57692

/-- Consider a square ABCD with a side length of 4 cm. A circle touches the extensions 
of sides AB and AD. From point C, two tangents are drawn to this circle, 
and the angle between the tangents is 60 degrees. -/
theorem circle_radius (side_length : ℝ) (angle_between_tangents : ℝ) : 
  side_length = 4 ∧ angle_between_tangents = 60 → 
  ∃ (radius : ℝ), radius = 4 * (Real.sqrt 2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l576_57692


namespace NUMINAMATH_GPT_binary_multiplication_correct_l576_57694

-- Define binary numbers as strings to directly use them in Lean
def binary_num1 : String := "1111"
def binary_num2 : String := "111"

-- Define a function to convert binary strings to natural numbers
def binary_to_nat (s : String) : Nat :=
  s.foldl (fun acc c => acc * 2 + (if c = '1' then 1 else 0)) 0

-- Define the target multiplication result
def binary_product_correct : Nat :=
  binary_to_nat "1001111"

theorem binary_multiplication_correct :
  binary_to_nat binary_num1 * binary_to_nat binary_num2 = binary_product_correct :=
by
  sorry

end NUMINAMATH_GPT_binary_multiplication_correct_l576_57694


namespace NUMINAMATH_GPT_value_of_a_minus_b_l576_57678

variables (a b : ℝ)

theorem value_of_a_minus_b (h1 : abs a = 3) (h2 : abs b = 5) (h3 : a > b) : a - b = 8 :=
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l576_57678


namespace NUMINAMATH_GPT_white_washing_cost_correct_l576_57638

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

def door_length : ℝ := 6
def door_width : ℝ := 3

def window_length : ℝ := 4
def window_width : ℝ := 3

def cost_per_sq_ft : ℝ := 8

def calculate_white_washing_cost : ℝ :=
  let total_wall_area := 2 * (room_length * room_height) + 2 * (room_width * room_height)
  let door_area := door_length * door_width
  let window_area := 3 * (window_length * window_width)
  let effective_area := total_wall_area - door_area - window_area
  effective_area * cost_per_sq_ft

theorem white_washing_cost_correct : calculate_white_washing_cost = 7248 := by
  sorry

end NUMINAMATH_GPT_white_washing_cost_correct_l576_57638


namespace NUMINAMATH_GPT_profit_percentage_is_correct_l576_57641

-- Define the conditions
variables (market_price_per_pen : ℝ) (discount_percentage : ℝ) (total_pens_bought : ℝ) (cost_pens_market_price : ℝ)
variables (cost_price_per_pen : ℝ) (selling_price_per_pen : ℝ) (profit_per_pen : ℝ) (profit_percent : ℝ)

-- Conditions
def condition_1 : market_price_per_pen = 1 := by sorry
def condition_2 : discount_percentage = 0.01 := by sorry
def condition_3 : total_pens_bought = 80 := by sorry
def condition_4 : cost_pens_market_price = 36 := by sorry

-- Definitions based on conditions
def cost_price_per_pen_def : cost_price_per_pen = cost_pens_market_price / total_pens_bought := by sorry
def selling_price_per_pen_def : selling_price_per_pen = market_price_per_pen * (1 - discount_percentage) := by sorry
def profit_per_pen_def : profit_per_pen = selling_price_per_pen - cost_price_per_pen := by sorry
def profit_percent_def : profit_percent = (profit_per_pen / cost_price_per_pen) * 100 := by sorry

-- The statement to prove
theorem profit_percentage_is_correct : profit_percent = 120 :=
by
  have h1 : cost_price_per_pen = 36 / 80 := by sorry
  have h2 : selling_price_per_pen = 1 * (1 - 0.01) := by sorry
  have h3 : profit_per_pen = 0.99 - 0.45 := by sorry
  have h4 : profit_percent = (0.54 / 0.45) * 100 := by sorry
  sorry

end NUMINAMATH_GPT_profit_percentage_is_correct_l576_57641


namespace NUMINAMATH_GPT_points_opposite_sides_of_line_l576_57688

theorem points_opposite_sides_of_line (a : ℝ) :
  (1 + 1 - a) * (2 - 1 - a) < 0 ↔ 1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_GPT_points_opposite_sides_of_line_l576_57688


namespace NUMINAMATH_GPT_exists_plane_through_point_parallel_to_line_at_distance_l576_57614

-- Definitions of the given entities
structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Line :=
(point : Point)
(direction : Point) -- Considering direction as a point vector for simplicity

def distance (P : Point) (L : Line) : ℝ := 
  -- Define the distance from point P to line L
  sorry

noncomputable def construct_plane (P : Point) (L : Line) (d : ℝ) : Prop :=
  -- Define when a plane can be constructed as stated in the problem.
  sorry

-- The main proof problem statement without the solution steps
theorem exists_plane_through_point_parallel_to_line_at_distance (P : Point) (L : Line) (d : ℝ) (h : distance P L > d) :
  construct_plane P L d :=
sorry

end NUMINAMATH_GPT_exists_plane_through_point_parallel_to_line_at_distance_l576_57614


namespace NUMINAMATH_GPT_daily_evaporation_l576_57687

variable (initial_water : ℝ) (percentage_evaporated : ℝ) (days : ℕ)
variable (evaporation_amount : ℝ)

-- Given conditions
def conditions_met : Prop :=
  initial_water = 10 ∧ percentage_evaporated = 0.4 ∧ days = 50

-- Question: Prove the amount of water evaporated each day is 0.08
theorem daily_evaporation (h : conditions_met initial_water percentage_evaporated days) :
  evaporation_amount = (initial_water * percentage_evaporated) / days :=
sorry

end NUMINAMATH_GPT_daily_evaporation_l576_57687


namespace NUMINAMATH_GPT_value_of_a_l576_57682

theorem value_of_a (a : ℕ) (h : a ^ 3 = 21 * 35 * 45 * 35) : a = 105 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l576_57682


namespace NUMINAMATH_GPT_lines_intersect_at_same_points_l576_57636

-- Definitions of linear equations in system 1 and system 2
def line1 (a1 b1 c1 x y : ℝ) := a1 * x + b1 * y = c1
def line2 (a2 b2 c2 x y : ℝ) := a2 * x + b2 * y = c2
def line3 (a3 b3 c3 x y : ℝ) := a3 * x + b3 * y = c3
def line4 (a4 b4 c4 x y : ℝ) := a4 * x + b4 * y = c4

-- Equivalence condition of the systems
def systems_equivalent (a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 : ℝ) :=
  ∀ (x y : ℝ), (line1 a1 b1 c1 x y ∧ line2 a2 b2 c2 x y) ↔ (line3 a3 b3 c3 x y ∧ line4 a4 b4 c4 x y)

-- Proof statement that the four lines intersect at the same set of points
theorem lines_intersect_at_same_points (a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 : ℝ) :
  systems_equivalent a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4 →
  ∀ (x y : ℝ), (line1 a1 b1 c1 x y ∧ line2 a2 b2 c2 x y) ↔ (line3 a3 b3 c3 x y ∧ line4 a4 b4 c4 x y) :=
by
  intros h_equiv x y
  exact h_equiv x y

end NUMINAMATH_GPT_lines_intersect_at_same_points_l576_57636


namespace NUMINAMATH_GPT_number_of_team_members_l576_57658

-- Let's define the conditions.
def packs : ℕ := 3
def pouches_per_pack : ℕ := 6
def total_pouches : ℕ := packs * pouches_per_pack
def coaches : ℕ := 3
def helpers : ℕ := 2
def total_people (members : ℕ) : ℕ := members + coaches + helpers

-- Prove the number of members on the baseball team.
theorem number_of_team_members (members : ℕ) (h : total_people members = total_pouches) : members = 13 :=
by
  sorry

end NUMINAMATH_GPT_number_of_team_members_l576_57658


namespace NUMINAMATH_GPT_dog_older_than_max_by_18_l576_57696

-- Definition of the conditions
def human_to_dog_years_ratio : ℕ := 7
def max_age : ℕ := 3
def dog_age_in_human_years : ℕ := 3

-- Translate the question: How much older, in dog years, will Max's dog be?
def age_difference_in_dog_years : ℕ :=
  dog_age_in_human_years * human_to_dog_years_ratio - max_age

-- The proof statement
theorem dog_older_than_max_by_18 : age_difference_in_dog_years = 18 := by
  sorry

end NUMINAMATH_GPT_dog_older_than_max_by_18_l576_57696


namespace NUMINAMATH_GPT_polynomial_value_at_2008_l576_57679

def f (a₀ a₁ a₂ a₃ a₄ : ℝ) (x : ℝ) : ℝ := a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4

theorem polynomial_value_at_2008 (a₀ a₁ a₂ a₃ a₄ : ℝ) (h₁ : a₄ ≠ 0)
  (h₀₃ : f a₀ a₁ a₂ a₃ a₄ 2003 = 24)
  (h₀₄ : f a₀ a₁ a₂ a₃ a₄ 2004 = -6)
  (h₀₅ : f a₀ a₁ a₂ a₃ a₄ 2005 = 4)
  (h₀₆ : f a₀ a₁ a₂ a₃ a₄ 2006 = -6)
  (h₀₇ : f a₀ a₁ a₂ a₃ a₄ 2007 = 24) :
  f a₀ a₁ a₂ a₃ a₄ 2008 = 274 :=
by sorry

end NUMINAMATH_GPT_polynomial_value_at_2008_l576_57679


namespace NUMINAMATH_GPT_determine_s_plus_u_l576_57653

theorem determine_s_plus_u (p r s u : ℂ) (q t : ℂ) (h₁ : q = 5)
    (h₂ : t = -p - r) (h₃ : p + q * I + r + s * I + t + u * I = 4 * I) : s + u = -1 :=
by
  sorry

end NUMINAMATH_GPT_determine_s_plus_u_l576_57653


namespace NUMINAMATH_GPT_unique_solution_l576_57602

theorem unique_solution (n m : ℕ) (hn : 0 < n) (hm : 0 < m) : 
  n^2 = m^4 + m^3 + m^2 + m + 1 ↔ (n, m) = (11, 3) :=
by sorry

end NUMINAMATH_GPT_unique_solution_l576_57602


namespace NUMINAMATH_GPT_train_travel_time_l576_57639

theorem train_travel_time
  (a : ℝ) (s : ℝ) (t : ℝ)
  (ha : a = 3)
  (hs : s = 27)
  (h0 : ∀ t, 0 ≤ t) :
  t = Real.sqrt 18 :=
by
  sorry

end NUMINAMATH_GPT_train_travel_time_l576_57639


namespace NUMINAMATH_GPT_sum_of_reciprocals_is_one_l576_57659

theorem sum_of_reciprocals_is_one (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (x : ℚ)) + (1 / (y : ℚ)) + (1 / (z : ℚ)) = 1 ↔ (x, y, z) = (2, 4, 4) ∨ 
                                                    (x, y, z) = (2, 3, 6) ∨ 
                                                    (x, y, z) = (3, 3, 3) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_is_one_l576_57659


namespace NUMINAMATH_GPT_new_average_weight_l576_57632

-- Statement only
theorem new_average_weight (avg_weight_29: ℝ) (weight_new_student: ℝ) (total_students: ℕ) 
  (h1: avg_weight_29 = 28) (h2: weight_new_student = 22) (h3: total_students = 29) : 
  (avg_weight_29 * total_students + weight_new_student) / (total_students + 1) = 27.8 :=
by
  -- declare local variables for simpler proof
  let total_weight := avg_weight_29 * total_students
  let new_total_weight := total_weight + weight_new_student
  let new_total_students := total_students + 1
  have t_weight : total_weight = 812 := by sorry
  have new_t_weight : new_total_weight = 834 := by sorry
  have n_total_students : new_total_students = 30 := by sorry
  exact sorry

end NUMINAMATH_GPT_new_average_weight_l576_57632


namespace NUMINAMATH_GPT_points_eq_l576_57671

-- Definition of the operation 
def star (a b : ℝ) : ℝ := a^2 * b + a * b^2

-- The property we want to prove
theorem points_eq : {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1} =
    {p : ℝ × ℝ | p.1 = 0} ∪ {p : ℝ × ℝ | p.2 = 0} ∪ {p : ℝ × ℝ | p.1 + p.2 = 0} :=
by
  sorry

end NUMINAMATH_GPT_points_eq_l576_57671


namespace NUMINAMATH_GPT_negation_exists_x_squared_lt_zero_l576_57623

open Classical

theorem negation_exists_x_squared_lt_zero :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by 
  sorry

end NUMINAMATH_GPT_negation_exists_x_squared_lt_zero_l576_57623


namespace NUMINAMATH_GPT_total_visitors_400_l576_57669

variables (V E U : ℕ)

def visitors_did_not_enjoy_understand (V : ℕ) := 3 * V / 4 + 100 = V
def visitors_enjoyed_equal_understood (E U : ℕ) := E = U
def total_visitors_satisfy_34 (V E : ℕ) := 3 * V / 4 = E

theorem total_visitors_400
  (h1 : ∀ V, visitors_did_not_enjoy_understand V)
  (h2 : ∀ E U, visitors_enjoyed_equal_understood E U)
  (h3 : ∀ V E, total_visitors_satisfy_34 V E) :
  V = 400 :=
by { sorry }

end NUMINAMATH_GPT_total_visitors_400_l576_57669


namespace NUMINAMATH_GPT_rhombus_longer_diagonal_l576_57664

theorem rhombus_longer_diagonal
  (a b : ℝ)
  (ha : a = 65) -- Side length
  (hb : b = 28) -- Half the length of the shorter diagonal
  : 2 * Real.sqrt (a^2 - b^2) = 118 :=  -- Statement of the longer diagonal length
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_rhombus_longer_diagonal_l576_57664


namespace NUMINAMATH_GPT_smallest_coterminal_angle_pos_radians_l576_57695

theorem smallest_coterminal_angle_pos_radians :
  ∀ (θ : ℝ), θ = -560 * (π / 180) → ∃ α : ℝ, α > 0 ∧ α = (8 * π) / 9 ∧ (∃ k : ℤ, θ + 2 * k * π = α) :=
by
  sorry

end NUMINAMATH_GPT_smallest_coterminal_angle_pos_radians_l576_57695


namespace NUMINAMATH_GPT_age_difference_l576_57655

theorem age_difference (a1 a2 a3 a4 x y : ℕ) 
  (h1 : (a1 + a2 + a3 + a4 + x) / 5 = 28)
  (h2 : ((a1 + 1) + (a2 + 1) + (a3 + 1) + (a4 + 1) + y) / 5 = 30) : 
  y - (x + 1) = 5 := 
by
  sorry

end NUMINAMATH_GPT_age_difference_l576_57655


namespace NUMINAMATH_GPT_exists_x_odd_n_l576_57673

theorem exists_x_odd_n (n : ℤ) (h : n % 2 = 1) : 
  ∃ x : ℤ, n^2 ∣ x^2 - n*x - 1 := by
  sorry

end NUMINAMATH_GPT_exists_x_odd_n_l576_57673


namespace NUMINAMATH_GPT_mother_used_eggs_l576_57646

variable (initial_eggs : ℕ) (eggs_after_chickens : ℕ) (chickens : ℕ) (eggs_per_chicken : ℕ) (current_eggs : ℕ)

theorem mother_used_eggs (h1 : initial_eggs = 10)
                        (h2 : chickens = 2)
                        (h3 : eggs_per_chicken = 3)
                        (h4 : current_eggs = 11)
                        (eggs_laid : ℕ)
                        (h5 : eggs_laid = chickens * eggs_per_chicken)
                        (eggs_used : ℕ)
                        (h6 : eggs_after_chickens = initial_eggs - eggs_used + eggs_laid)
                        : eggs_used = 7 :=
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_mother_used_eggs_l576_57646


namespace NUMINAMATH_GPT_sequence_geometric_l576_57643

theorem sequence_geometric (a : ℕ → ℕ) (n : ℕ) (hn : 0 < n):
  (a 1 = 1) →
  (∀ n, 0 < n → a (n + 1) = 2 * a n) →
  a n = 2^(n-1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_sequence_geometric_l576_57643


namespace NUMINAMATH_GPT_min_value_of_a_plus_2b_l576_57686

theorem min_value_of_a_plus_2b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 2 / a + 1 / b = 1) : a + 2 * b = 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_plus_2b_l576_57686


namespace NUMINAMATH_GPT_find_f_function_l576_57620

def oddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem find_f_function (f : ℝ → ℝ) (h_odd : oddFunction f) (h_pos : ∀ x, 0 < x → f x = x * (1 + x)) :
  ∀ x, x < 0 → f x = -x - x^2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_function_l576_57620


namespace NUMINAMATH_GPT_brownie_leftover_is_zero_l576_57665

-- Define the dimensions of the pan
def pan_length : ℕ := 24
def pan_width : ℕ := 15

-- Define the dimensions of one piece of brownie
def piece_length : ℕ := 3
def piece_width : ℕ := 4

-- The total area of the pan
def pan_area : ℕ := pan_length * pan_width

-- The total area of one piece
def piece_area : ℕ := piece_length * piece_width

-- The number of full pieces that can be cut
def number_of_pieces : ℕ := pan_area / piece_area

-- The total used area when pieces are cut
def used_area : ℕ := number_of_pieces * piece_area

-- The leftover area
def leftover_area : ℕ := pan_area - used_area

theorem brownie_leftover_is_zero (pan_length pan_width piece_length piece_width : ℕ)
  (h1 : pan_length = 24) (h2 : pan_width = 15) 
  (h3 : piece_length = 3) (h4 : piece_width = 4) :
  pan_width * pan_length - (pan_width * pan_length / (piece_width * piece_length)) * (piece_width * piece_length) = 0 := 
by sorry

end NUMINAMATH_GPT_brownie_leftover_is_zero_l576_57665


namespace NUMINAMATH_GPT_height_of_parallelogram_l576_57616

theorem height_of_parallelogram
  (A B H : ℝ)
  (h1 : A = 480)
  (h2 : B = 32)
  (h3 : A = B * H) : 
  H = 15 := sorry

end NUMINAMATH_GPT_height_of_parallelogram_l576_57616


namespace NUMINAMATH_GPT_maple_taller_than_birch_l576_57663

def birch_tree_height : ℚ := 49 / 4
def maple_tree_height : ℚ := 102 / 5

theorem maple_taller_than_birch : maple_tree_height - birch_tree_height = 163 / 20 :=
by
  sorry

end NUMINAMATH_GPT_maple_taller_than_birch_l576_57663


namespace NUMINAMATH_GPT_percentage_is_50_l576_57684

theorem percentage_is_50 (P : ℝ) (h1 : P = 0.20 * 15 + 47) : P = 50 := 
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_percentage_is_50_l576_57684


namespace NUMINAMATH_GPT_john_income_increase_l576_57642

noncomputable def net_percentage_increase (initial_income : ℝ) (final_income_before_bonus : ℝ) (monthly_bonus : ℝ) (tax_deduction_rate : ℝ) : ℝ :=
  let weekly_bonus := monthly_bonus / 4
  let final_income_before_taxes := final_income_before_bonus + weekly_bonus
  let tax_deduction := tax_deduction_rate * final_income_before_taxes
  let net_final_income := final_income_before_taxes - tax_deduction
  ((net_final_income - initial_income) / initial_income) * 100

theorem john_income_increase :
  net_percentage_increase 40 60 100 0.10 = 91.25 := by
  sorry

end NUMINAMATH_GPT_john_income_increase_l576_57642


namespace NUMINAMATH_GPT_quadratic_real_roots_range_l576_57651

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * x + m = 0) → m ≤ 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_l576_57651


namespace NUMINAMATH_GPT_carol_initial_cupcakes_l576_57633

variable (x : ℕ)

theorem carol_initial_cupcakes (h : (x - 9) + 28 = 49) : x = 30 := 
  sorry

end NUMINAMATH_GPT_carol_initial_cupcakes_l576_57633


namespace NUMINAMATH_GPT_find_number_l576_57661

variable (N : ℕ)

theorem find_number (h : 6 * ((N / 8) + 8 - 30) = 12) : N = 192 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l576_57661


namespace NUMINAMATH_GPT_train_speed_ratio_l576_57691

variable (V1 V2 : ℝ)

theorem train_speed_ratio (H1 : V1 * 4 = D1) (H2 : V2 * 36 = D2) (H3 : D1 / D2 = 1 / 9) :
  V1 / V2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_ratio_l576_57691


namespace NUMINAMATH_GPT_cracked_to_broken_eggs_ratio_l576_57619

theorem cracked_to_broken_eggs_ratio (total_eggs : ℕ) (broken_eggs : ℕ) (P C : ℕ)
  (h1 : total_eggs = 24)
  (h2 : broken_eggs = 3)
  (h3 : P - C = 9)
  (h4 : P + C = 21) :
  (C : ℚ) / (broken_eggs : ℚ) = 2 :=
by
  sorry

end NUMINAMATH_GPT_cracked_to_broken_eggs_ratio_l576_57619


namespace NUMINAMATH_GPT_max_pasture_area_l576_57689

/-- A rectangular sheep pasture is enclosed on three sides by a fence, while the fourth side uses the 
side of a barn that is 500 feet long. The fence costs $10 per foot, and the total budget for the 
fence is $2000. Determine the length of the side parallel to the barn that will maximize the pasture area. -/
theorem max_pasture_area (length_barn : ℝ) (cost_per_foot : ℝ) (budget : ℝ) :
  length_barn = 500 ∧ cost_per_foot = 10 ∧ budget = 2000 → 
  ∃ x : ℝ, x = 100 ∧ (∀ y : ℝ, y ≥ 0 → 
    (budget / cost_per_foot) ≥ 2*y + x → 
    (y * x ≤ y * 100)) :=
by
  sorry

end NUMINAMATH_GPT_max_pasture_area_l576_57689


namespace NUMINAMATH_GPT_rotation_image_of_D_l576_57693

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

theorem rotation_image_of_D :
  rotate_90_clockwise (-3, 2) = (2, 3) :=
by
  sorry

end NUMINAMATH_GPT_rotation_image_of_D_l576_57693


namespace NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l576_57600
noncomputable def sufficient_but_not_necessary (a b : ℝ) : Prop :=
∀ (a b : ℝ), a < 0 → -1 < b ∧ b < 0 → a + a * b < 0

-- Define the theorem stating the proof problem
theorem condition_sufficient_but_not_necessary (a b : ℝ) :
  (a < 0 ∧ -1 < b ∧ b < 0 → a + a * b < 0) ∧ 
  (a + a * b < 0 → a < 0 ∧ 1 + b > 0 ∨ a > 0 ∧ 1 + b < 0) :=
sorry

end NUMINAMATH_GPT_condition_sufficient_but_not_necessary_l576_57600


namespace NUMINAMATH_GPT_paul_money_duration_l576_57603

theorem paul_money_duration
  (mow_earnings : ℕ)
  (weed_earnings : ℕ)
  (weekly_expenses : ℕ)
  (earnings_mow : mow_earnings = 3)
  (earnings_weed : weed_earnings = 3)
  (expenses : weekly_expenses = 3) :
  (mow_earnings + weed_earnings) / weekly_expenses = 2 := 
by
  sorry

end NUMINAMATH_GPT_paul_money_duration_l576_57603


namespace NUMINAMATH_GPT_units_digit_of_product_l576_57626

theorem units_digit_of_product : 
  (27 % 10 = 7) ∧ (68 % 10 = 8) → ((27 * 68) % 10 = 6) :=
by sorry

end NUMINAMATH_GPT_units_digit_of_product_l576_57626


namespace NUMINAMATH_GPT_lattice_points_count_l576_57654

-- A definition of lattice points and bounded region
def is_lattice_point (p : ℤ × ℤ) : Prop := true

def in_region (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  (y = abs x ∨ y = -x^2 + 4*x + 6) ∧ (y ≤ abs x ∧ y ≤ -x^2 + 4*x + 6)

-- The target statement to prove
theorem lattice_points_count : ∃ n, n = 23 ∧ ∀ p : ℤ × ℤ, is_lattice_point p → in_region p := sorry

end NUMINAMATH_GPT_lattice_points_count_l576_57654


namespace NUMINAMATH_GPT_value_of_x_squared_plus_inverse_squared_l576_57615

theorem value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x ≠ 0) (h : x^4 + (1 / x^4) = 2) : x^2 + (1 / x^2) = 2 :=
sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_inverse_squared_l576_57615


namespace NUMINAMATH_GPT_John_scored_24point5_goals_l576_57624

theorem John_scored_24point5_goals (T G : ℝ) (n : ℕ) (A : ℝ)
  (h1 : T = 65)
  (h2 : n = 9)
  (h3 : A = 4.5) :
  G = T - (n * A) :=
by
  sorry

end NUMINAMATH_GPT_John_scored_24point5_goals_l576_57624


namespace NUMINAMATH_GPT_Tom_time_to_complete_wall_after_one_hour_l576_57613

noncomputable def avery_rate : ℝ := 1 / 2
noncomputable def tom_rate : ℝ := 1 / 4
noncomputable def combined_rate : ℝ := avery_rate + tom_rate
noncomputable def wall_built_in_first_hour : ℝ := combined_rate * 1
noncomputable def remaining_wall : ℝ := 1 - wall_built_in_first_hour 
noncomputable def tom_time_to_complete_remaining_wall : ℝ := remaining_wall / tom_rate

theorem Tom_time_to_complete_wall_after_one_hour : 
  tom_time_to_complete_remaining_wall = 1 :=
by
  sorry

end NUMINAMATH_GPT_Tom_time_to_complete_wall_after_one_hour_l576_57613


namespace NUMINAMATH_GPT_solve_for_S_l576_57605

variable (D S : ℝ)
variable (h1 : D > 0)
variable (h2 : S > 0)
variable (h3 : ((0.75 * D) / 50 + (0.25 * D) / S) / D = 1 / 50)

theorem solve_for_S :
  S = 50 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_S_l576_57605


namespace NUMINAMATH_GPT_calc_x2015_l576_57670

noncomputable def f (x a : ℝ) : ℝ := x / (a * (x + 2))

theorem calc_x2015 (a x x_0 : ℝ) (x_seq : ℕ → ℝ)
  (h_unique: ∀ x, f x a = x → x = 0) 
  (h_a_val: a = 1 / 2)
  (h_f_x0: f x_0 a = 1 / 1008)
  (h_seq: ∀ n, x_seq (n + 1) = f (x_seq n) a)
  (h_x0_val: x_seq 0 = x_0):
  x_seq 2015 = 1 / 2015 :=
by
  sorry

end NUMINAMATH_GPT_calc_x2015_l576_57670


namespace NUMINAMATH_GPT_brads_zip_code_l576_57660

theorem brads_zip_code (A B C D E : ℕ) (h1 : A + B + C + D + E = 20)
                        (h2 : B = A + 1) (h3 : C = A)
                        (h4 : D = 2 * A) (h5 : D + E = 13)
                        (h6 : Nat.Prime (A*10000 + B*1000 + C*100 + D*10 + E)) :
                        A*10000 + B*1000 + C*100 + D*10 + E = 34367 := 
sorry

end NUMINAMATH_GPT_brads_zip_code_l576_57660


namespace NUMINAMATH_GPT_expression_never_equals_33_l576_57666

theorem expression_never_equals_33 (x y : ℤ) : 
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end NUMINAMATH_GPT_expression_never_equals_33_l576_57666


namespace NUMINAMATH_GPT_find_k_slope_eq_l576_57680

theorem find_k_slope_eq :
  ∃ k: ℝ, (∃ k: ℝ, ((k - 4) / 7 = (-2 - k) / 14) → k = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_k_slope_eq_l576_57680


namespace NUMINAMATH_GPT_cos_minus_sin_l576_57650

theorem cos_minus_sin (α : ℝ) (h1 : Real.sin (2 * α) = 1 / 4) (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2) : 
  Real.cos α - Real.sin α = - (Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_cos_minus_sin_l576_57650


namespace NUMINAMATH_GPT_frogs_per_fish_per_day_l576_57674

theorem frogs_per_fish_per_day
  (f g n F : ℕ)
  (h1 : f = 30)
  (h2 : g = 15)
  (h3 : n = 9)
  (h4 : F = 32400) :
  F / f / (n * g) = 8 := by
  sorry

end NUMINAMATH_GPT_frogs_per_fish_per_day_l576_57674


namespace NUMINAMATH_GPT_voronovich_inequality_l576_57634

theorem voronovich_inequality (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 1) :
  (a^2 + b^2 + c^2)^2 + 6 * a * b * c ≥ a * b + b * c + c * a :=
by
  sorry

end NUMINAMATH_GPT_voronovich_inequality_l576_57634


namespace NUMINAMATH_GPT_inequality_proof_l576_57690

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ineq : a + b + c > 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l576_57690


namespace NUMINAMATH_GPT_sculpture_plus_base_height_l576_57645

def height_sculpture_feet : Nat := 2
def height_sculpture_inches : Nat := 10
def height_base_inches : Nat := 4

def height_sculpture_total_inches : Nat := height_sculpture_feet * 12 + height_sculpture_inches
def height_total_inches : Nat := height_sculpture_total_inches + height_base_inches

theorem sculpture_plus_base_height :
  height_total_inches = 38 := by
  sorry

end NUMINAMATH_GPT_sculpture_plus_base_height_l576_57645


namespace NUMINAMATH_GPT_probability_of_drawing_1_red_1_white_l576_57612

-- Definitions
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

-- Probabilities
def p_red_first_white_second : ℚ := (red_balls / total_balls : ℚ) * (white_balls / total_balls : ℚ)
def p_white_first_red_second : ℚ := (white_balls / total_balls : ℚ) * (red_balls / total_balls : ℚ)

-- Total probability
def total_probability : ℚ := p_red_first_white_second + p_white_first_red_second

theorem probability_of_drawing_1_red_1_white :
  total_probability = 12 / 25 := by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_1_red_1_white_l576_57612


namespace NUMINAMATH_GPT_arc_length_120_degrees_l576_57618

theorem arc_length_120_degrees (π : ℝ) : 
  let R := π
  let n := 120
  (n * π * R) / 180 = (2 * π^2) / 3 := 
by
  let R := π
  let n := 120
  sorry

end NUMINAMATH_GPT_arc_length_120_degrees_l576_57618


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_p_l576_57668

variable {p q r : Prop}

theorem necessary_but_not_sufficient_for_p 
  (h₁ : p → q) (h₂ : ¬ (q → p)) 
  (h₃ : q → r) (h₄ : ¬ (r → q)) 
  : (r → p) ∧ ¬ (p → r) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_p_l576_57668


namespace NUMINAMATH_GPT_drawing_at_least_one_red_is_certain_l576_57681

-- Defining the balls and box conditions
structure Box :=
  (red_balls : ℕ) 
  (yellow_balls : ℕ) 

-- Let the box be defined as having 3 red balls and 2 yellow balls
def box : Box := { red_balls := 3, yellow_balls := 2 }

-- Define the event of drawing at least one red ball
def at_least_one_red (draws : ℕ) (b : Box) : Prop :=
  ∀ drawn_yellow, drawn_yellow < draws → drawn_yellow < b.yellow_balls

-- The conclusion we want to prove
theorem drawing_at_least_one_red_is_certain : at_least_one_red 3 box :=
by 
  sorry

end NUMINAMATH_GPT_drawing_at_least_one_red_is_certain_l576_57681


namespace NUMINAMATH_GPT_trigonometric_identity_l576_57656

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l576_57656


namespace NUMINAMATH_GPT_min_people_like_mozart_bach_not_beethoven_l576_57609

-- Define the initial conditions
variables {n a b c : ℕ}
variables (total_people := 150)
variables (likes_mozart := 120)
variables (likes_bach := 105)
variables (likes_beethoven := 45)

theorem min_people_like_mozart_bach_not_beethoven : 
  ∃ (x : ℕ), 
    total_people = 150 ∧ 
    likes_mozart = 120 ∧ 
    likes_bach = 105 ∧ 
    likes_beethoven = 45 ∧ 
    x = (likes_mozart + likes_bach - total_people) := 
    sorry

end NUMINAMATH_GPT_min_people_like_mozart_bach_not_beethoven_l576_57609


namespace NUMINAMATH_GPT_daily_evaporation_rate_l576_57640

theorem daily_evaporation_rate (initial_amount : ℝ) (period : ℕ) (percentage_evaporated : ℝ) (h_initial : initial_amount = 10) (h_period : period = 50) (h_percentage : percentage_evaporated = 4) : 
  (percentage_evaporated / 100 * initial_amount) / period = 0.008 :=
by
  -- Ensures that the conditions translate directly into the Lean theorem statement
  rw [h_initial, h_period, h_percentage]
  -- Insert the required logical proof here
  sorry

end NUMINAMATH_GPT_daily_evaporation_rate_l576_57640


namespace NUMINAMATH_GPT_converse_of_statement_l576_57667

theorem converse_of_statement (x y : ℝ) :
  (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_converse_of_statement_l576_57667


namespace NUMINAMATH_GPT_sqrt_expr_is_599_l576_57676

theorem sqrt_expr_is_599 : Real.sqrt ((26 * 25 * 24 * 23) + 1) = 599 := by
  sorry

end NUMINAMATH_GPT_sqrt_expr_is_599_l576_57676


namespace NUMINAMATH_GPT_coffee_containers_used_l576_57601

theorem coffee_containers_used :
  let Suki_coffee := 6.5 * 22
  let Jimmy_coffee := 4.5 * 18
  let combined_coffee := Suki_coffee + Jimmy_coffee
  let containers := combined_coffee / 8
  containers = 28 := 
by
  sorry

end NUMINAMATH_GPT_coffee_containers_used_l576_57601


namespace NUMINAMATH_GPT_existence_of_positive_numbers_l576_57628

open Real

theorem existence_of_positive_numbers {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 > 2 ∧ a^3 + b^3 + c^3 < 2 ∧ a^4 + b^4 + c^4 > 2 :=
sorry

end NUMINAMATH_GPT_existence_of_positive_numbers_l576_57628


namespace NUMINAMATH_GPT_gcd_lcm_of_a_b_l576_57647

def a := 1560
def b := 1040

theorem gcd_lcm_of_a_b :
  (Nat.gcd a b = 520) ∧ (Nat.lcm a b = 1560) :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_gcd_lcm_of_a_b_l576_57647


namespace NUMINAMATH_GPT_directrix_of_parabola_l576_57652

theorem directrix_of_parabola (a b c : ℝ) (parabola_eqn : ∀ x : ℝ, y = 3 * x^2 - 6 * x + 2)
  (vertex : ∃ h k : ℝ, h = 1 ∧ k = -1)
  : ∃ y : ℝ, y = -13 / 12 := 
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l576_57652


namespace NUMINAMATH_GPT_sum_of_digits_eq_28_l576_57683

theorem sum_of_digits_eq_28 (A B C D E : ℕ) 
  (hA : 0 ≤ A ∧ A ≤ 9) 
  (hB : 0 ≤ B ∧ B ≤ 9) 
  (hC : 0 ≤ C ∧ C ≤ 9) 
  (hD : 0 ≤ D ∧ D ≤ 9) 
  (hE : 0 ≤ E ∧ E ≤ 9) 
  (unique_digits : (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (C ≠ D) ∧ (C ≠ E) ∧ (D ≠ E)) 
  (h : (10 * A + B) * (10 * C + D) = 111 * E) : 
  A + B + C + D + E = 28 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_eq_28_l576_57683


namespace NUMINAMATH_GPT_rehabilitation_centers_l576_57604

def Lisa : ℕ := 6 
def Jude : ℕ := Lisa / 2
def Han : ℕ := 2 * Jude - 2
def Jane : ℕ := 27 - Lisa - Jude - Han
def x : ℕ := 2

theorem rehabilitation_centers:
  Jane = x * Han + 6 := 
by
  -- Proof goes here (not required)
  sorry

end NUMINAMATH_GPT_rehabilitation_centers_l576_57604


namespace NUMINAMATH_GPT_converse_inverse_contrapositive_l576_57685

theorem converse (x y : ℤ) : (x = 3 ∧ y = 2) → (x + y = 5) :=
by sorry

theorem inverse (x y : ℤ) : (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2) :=
by sorry

theorem contrapositive (x y : ℤ) : (¬ (x = 3 ∧ y = 2)) → (¬ (x + y = 5)) :=
by sorry

end NUMINAMATH_GPT_converse_inverse_contrapositive_l576_57685
