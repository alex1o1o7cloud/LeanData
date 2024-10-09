import Mathlib

namespace express_fraction_l1121_112191

noncomputable def x : ℚ := 0.8571 -- This represents \( x = 0.\overline{8571} \)
noncomputable def y : ℚ := 0.142857 -- This represents \( y = 0.\overline{142857} \)
noncomputable def z : ℚ := 2 + y -- This represents \( 2 + y = 2.\overline{142857} \)

theorem express_fraction :
  (x / z) = (1 / 2) :=
by
  sorry

end express_fraction_l1121_112191


namespace proportion_false_if_x_is_0_75_correct_value_of_x_in_proportion_l1121_112149

theorem proportion_false_if_x_is_0_75 (x : ℚ) (h1 : x = 0.75) : ¬ (x / 2 = 2 / 6) :=
by sorry

theorem correct_value_of_x_in_proportion (x : ℚ) (h1 : x / 2 = 2 / 6) : x = 2 / 3 :=
by sorry

end proportion_false_if_x_is_0_75_correct_value_of_x_in_proportion_l1121_112149


namespace sum_of_local_values_l1121_112174

def local_value (digit place_value : ℕ) : ℕ := digit * place_value

theorem sum_of_local_values :
  local_value 2 1000 + local_value 3 100 + local_value 4 10 + local_value 5 1 = 2345 :=
by
  sorry

end sum_of_local_values_l1121_112174


namespace problem_l1121_112166

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

theorem problem (h : f 10 = 756) : f 10 = 756 := 
by 
  sorry

end problem_l1121_112166


namespace multiple_of_C_share_l1121_112136

theorem multiple_of_C_share (A B C k : ℝ) : 
  3 * A = k * C ∧ 4 * B = k * C ∧ C = 84 ∧ A + B + C = 427 → k = 7 :=
by
  sorry

end multiple_of_C_share_l1121_112136


namespace max_value_condition_min_value_condition_l1121_112103

theorem max_value_condition (x : ℝ) (h : x < 0) : (x^2 + x + 1) / x ≤ -1 :=
sorry

theorem min_value_condition (x : ℝ) (h : x > -1) : ((x + 5) * (x + 2)) / (x + 1) ≥ 9 :=
sorry

end max_value_condition_min_value_condition_l1121_112103


namespace problem_solution_l1121_112187

variable (α : ℝ)
-- Condition: α in the first quadrant (0 < α < π/2)
variable (h1 : 0 < α ∧ α < Real.pi / 2)
-- Condition: sin α + cos α = sqrt 2
variable (h2 : Real.sin α + Real.cos α = Real.sqrt 2)

theorem problem_solution : Real.tan α + Real.cos α / Real.sin α = 2 :=
by
  sorry

end problem_solution_l1121_112187


namespace find_vector_b_coordinates_l1121_112130

theorem find_vector_b_coordinates 
  (a b : ℝ × ℝ) 
  (h₁ : a = (-3, 4)) 
  (h₂ : ∃ m : ℝ, m < 0 ∧ b = (-3 * m, 4 * m)) 
  (h₃ : ‖b‖ = 10) : 
  b = (6, -8) := 
by
  sorry

end find_vector_b_coordinates_l1121_112130


namespace correct_operation_l1121_112152

variable {a b : ℝ}

theorem correct_operation : (3 * a^2 * b - 3 * b * a^2 = 0) :=
by sorry

end correct_operation_l1121_112152


namespace boat_travel_time_l1121_112199

theorem boat_travel_time (x : ℝ) (T : ℝ) (h0 : 0 ≤ x) (h1 : x ≠ 15.6) 
    (h2 : 96 = (15.6 - x) * T) 
    (h3 : 96 = (15.6 + x) * 5) : 
    T = 8 :=
by 
  sorry

end boat_travel_time_l1121_112199


namespace ball_distribution_l1121_112198

theorem ball_distribution (N a b : ℕ) (h1 : N = 6912) (h2 : N = 100 * a + b) (h3 : a < 100) (h4 : b < 100) : a + b = 81 :=
by
  sorry

end ball_distribution_l1121_112198


namespace circle_condition_l1121_112111

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4 * x - 2 * y + 5 * m = 0) ↔ m < 1 := by
  sorry

end circle_condition_l1121_112111


namespace birds_count_214_l1121_112120

def two_legged_birds_count (b m i : Nat) : Prop :=
  b + m + i = 300 ∧ 2 * b + 4 * m + 3 * i = 686 → b = 214

theorem birds_count_214 (b m i : Nat) : two_legged_birds_count b m i :=
by
  sorry

end birds_count_214_l1121_112120


namespace total_amount_of_check_l1121_112148

def numParts : Nat := 59
def price50DollarPart : Nat := 50
def price20DollarPart : Nat := 20
def num50DollarParts : Nat := 40

theorem total_amount_of_check : (num50DollarParts * price50DollarPart + (numParts - num50DollarParts) * price20DollarPart) = 2380 := by
  sorry

end total_amount_of_check_l1121_112148


namespace sequence_sum_l1121_112113

theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, 0 < a n)
  → (∀ n : ℕ, S (n + 1) = S n + a (n + 1)) 
  → (∀ n : ℕ, a (n+1)^2 = a n * a (n+2))
  → S 3 = 13
  → a 1 = 1
  → (a 3 + a 4) / (a 1 + a 2) = 9 :=
sorry

end sequence_sum_l1121_112113


namespace total_pieces_correct_l1121_112110

theorem total_pieces_correct :
  let bell_peppers := 10
  let onions := 7
  let zucchinis := 15
  let bell_peppers_slices := (2 * 20)  -- 25% of 10 bell peppers sliced into 20 slices each
  let bell_peppers_large_pieces := (7 * 10)  -- Remaining 75% cut into 10 pieces each
  let bell_peppers_smaller_pieces := (35 * 3)  -- Half of large pieces cut into 3 pieces each
  let onions_slices := (3 * 18)  -- 50% of onions sliced into 18 slices each
  let onions_pieces := (4 * 8)  -- Remaining 50% cut into 8 pieces each
  let zucchinis_slices := (4 * 15)  -- 30% of zucchinis sliced into 15 pieces each
  let zucchinis_pieces := (10 * 8)  -- Remaining 70% cut into 8 pieces each
  let total_slices := bell_peppers_slices + onions_slices + zucchinis_slices
  let total_pieces := bell_peppers_large_pieces + bell_peppers_smaller_pieces + onions_pieces + zucchinis_pieces
  total_slices + total_pieces = 441 :=
by
  sorry

end total_pieces_correct_l1121_112110


namespace part1_l1121_112171

noncomputable def P : Set ℝ := {x | (1 / 2) ≤ x ∧ x ≤ 1}
noncomputable def Q (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}
def U : Set ℝ := Set.univ
noncomputable def complement_P : Set ℝ := {x | x < (1 / 2)} ∪ {x | x > 1}

theorem part1 (a : ℝ) (h : a = 1) : 
  (complement_P ∩ Q a) = {x | 1 < x ∧ x ≤ 2} :=
sorry

end part1_l1121_112171


namespace smallest_positive_period_monotonically_increasing_interval_minimum_value_a_of_triangle_l1121_112182

noncomputable def f (x : ℝ) := 2 * (Real.cos x)^2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x := 
by 
  -- Proof omitted
  sorry

theorem monotonically_increasing_interval :
  ∃ k : ℤ, ∀ x y, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 → 
               k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤  k * Real.pi + Real.pi / 6 →
               x ≤ y → f x ≤ f y := 
by 
  -- Proof omitted
  sorry

theorem minimum_value_a_of_triangle (A B C a b c : ℝ) 
  (h₀ : f A = 1/2) 
  (h₁ : B^2 - C^2 - B * C * Real.cos A - a^2 = 4) :
  a ≥ 2 * Real.sqrt 2 :=
by 
  -- Proof omitted
  sorry

end smallest_positive_period_monotonically_increasing_interval_minimum_value_a_of_triangle_l1121_112182


namespace intersection_P_Q_l1121_112192

-- Definitions based on conditions
def P : Set ℝ := { y | ∃ x : ℝ, y = x + 1 }
def Q : Set ℝ := { y | ∃ x : ℝ, y = 1 - x }

-- Proof statement to show P ∩ Q = Set.univ
theorem intersection_P_Q : P ∩ Q = Set.univ := by
  sorry

end intersection_P_Q_l1121_112192


namespace power_of_2_multiplication_l1121_112153

theorem power_of_2_multiplication : (16^3) * (4^4) * (32^2) = 2^30 := by
  sorry

end power_of_2_multiplication_l1121_112153


namespace arithmetic_mean_l1121_112158

theorem arithmetic_mean (x y : ℝ) (h1 : x = Real.sqrt 2 - 1) (h2 : y = 1 / (Real.sqrt 2 - 1)) :
  (x + y) / 2 = Real.sqrt 2 := sorry

end arithmetic_mean_l1121_112158


namespace quadratic_inequality_solutions_l1121_112185

theorem quadratic_inequality_solutions (a x : ℝ) :
  (x^2 - (2+a)*x + 2*a > 0) → (
    (a < 2  → (x < a ∨ x > 2)) ∧
    (a = 2  → (x ≠ 2)) ∧
    (a > 2  → (x < 2 ∨ x > a))
  ) :=
by sorry

end quadratic_inequality_solutions_l1121_112185


namespace gcf_palindromes_multiple_of_3_eq_3_l1121_112190

-- Defining a condition that expresses a three-digit palindrome in the form 101a + 10b + a
def is_palindrome (n : ℕ) : Prop :=
∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b + a

-- Defining a condition that n is a multiple of 3
def is_multiple_of_3 (n : ℕ) : Prop :=
n % 3 = 0

-- The Lean statement to prove the greatest common factor of all three-digit palindromes that are multiples of 3
theorem gcf_palindromes_multiple_of_3_eq_3 :
  ∃ gcf : ℕ, gcf = 3 ∧ ∀ n : ℕ, (is_palindrome n ∧ is_multiple_of_3 n) → gcf ∣ n :=
by
  sorry

end gcf_palindromes_multiple_of_3_eq_3_l1121_112190


namespace sum_of_roots_l1121_112106

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - 2016 * x + 2015

theorem sum_of_roots (a b c : ℝ) (h1 : f a = c) (h2 : f b = c) (h3 : a ≠ b) :
  a + b = 2016 :=
by
  sorry

end sum_of_roots_l1121_112106


namespace sum_of_nonnegative_numbers_eq_10_l1121_112159

theorem sum_of_nonnegative_numbers_eq_10 (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 48)
  (h2 : ab + bc + ca = 26)
  (h3 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) : a + b + c = 10 := 
by
  sorry

end sum_of_nonnegative_numbers_eq_10_l1121_112159


namespace first_term_of_arithmetic_sequence_l1121_112176

theorem first_term_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ)
  (h_arith : ∀ n, a n = a1 + ↑n - 1) 
  (h_sum : ∀ n, S n = n / 2 * (2 * a1 + (n - 1))) 
  (h_min : ∀ n, S 2022 ≤ S n) : 
  -2022 < a1 ∧ a1 < -2021 :=
by
  sorry

end first_term_of_arithmetic_sequence_l1121_112176


namespace minimum_shots_to_hit_ship_l1121_112160

def is_ship_hit (shots : Finset (Fin 7 × Fin 7)) : Prop :=
  -- Assuming the ship can be represented by any 4 consecutive points in a row
  ∀ r : Fin 7, ∃ c1 c2 c3 c4 : Fin 7, 
    (0 ≤ c1.1 ∧ c1.1 ≤ 6 ∧ c1.1 + 3 = c4.1) ∧
    (0 ≤ c2.1 ∧ c2.1 ≤ 6 ∧ c2.1 = c1.1 + 1) ∧
    (0 ≤ c3.1 ∧ c3.1 ≤ 6 ∧ c3.1 = c1.1 + 2) ∧
    (r, c1) ∈ shots ∧ (r, c2) ∈ shots ∧ (r, c3) ∈ shots ∧ (r, c4) ∈ shots

theorem minimum_shots_to_hit_ship : ∃ shots : Finset (Fin 7 × Fin 7), 
  shots.card = 12 ∧ is_ship_hit shots :=
by 
  sorry

end minimum_shots_to_hit_ship_l1121_112160


namespace total_legs_in_christophers_room_l1121_112100

def total_legs (num_spiders num_legs_per_spider num_ants num_butterflies num_beetles num_legs_per_insect : ℕ) : ℕ :=
  let spider_legs := num_spiders * num_legs_per_spider
  let ant_legs := num_ants * num_legs_per_insect
  let butterfly_legs := num_butterflies * num_legs_per_insect
  let beetle_legs := num_beetles * num_legs_per_insect
  spider_legs + ant_legs + butterfly_legs + beetle_legs

theorem total_legs_in_christophers_room : total_legs 12 8 10 5 5 6 = 216 := by
  -- Calculation and reasoning omitted
  sorry

end total_legs_in_christophers_room_l1121_112100


namespace max_vector_sum_l1121_112180

open Real EuclideanSpace

noncomputable def circle_center : ℝ × ℝ := (3, 0)
noncomputable def radius : ℝ := 2
noncomputable def distance_AB : ℝ := 2 * sqrt 3

theorem max_vector_sum {A B : ℝ × ℝ} 
    (hA_on_circle : dist A circle_center = radius)
    (hB_on_circle : dist B circle_center = radius)
    (hAB_eq : dist A B = distance_AB) :
    (dist (0,0) ((A.1 + B.1, A.2 + B.2))) ≤ 8 :=
by 
  sorry

end max_vector_sum_l1121_112180


namespace proof_expression_value_l1121_112188

theorem proof_expression_value (x y : ℝ) (h : x + 2 * y = 30) : 
  (x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3) = 16 := 
by 
  sorry

end proof_expression_value_l1121_112188


namespace eugene_pencils_left_l1121_112131

-- Define the total number of pencils Eugene initially has
def initial_pencils : ℝ := 234.0

-- Define the number of pencils Eugene gives away
def pencils_given_away : ℝ := 35.0

-- Define the expected number of pencils left
def expected_pencils_left : ℝ := 199.0

-- Prove the number of pencils left after giving away 35.0 equals 199.0
theorem eugene_pencils_left : initial_pencils - pencils_given_away = expected_pencils_left := by
  -- This is where the proof would go, if needed
  sorry

end eugene_pencils_left_l1121_112131


namespace dacid_physics_marks_l1121_112142

theorem dacid_physics_marks 
  (english : ℕ := 73)
  (math : ℕ := 69)
  (chem : ℕ := 64)
  (bio : ℕ := 82)
  (avg_marks : ℕ := 76)
  (num_subjects : ℕ := 5)
  : ∃ physics : ℕ, physics = 92 :=
by
  let total_marks := avg_marks * num_subjects
  let known_marks := english + math + chem + bio
  have physics := total_marks - known_marks
  use physics
  sorry

end dacid_physics_marks_l1121_112142


namespace midlines_tangent_fixed_circle_l1121_112186

-- Definitions of geometric objects and properties
structure Point :=
(x : ℝ) (y : ℝ)

structure Circle :=
(center : Point) (radius : ℝ)

-- Assumptions (conditions)
variable (ω1 ω2 : Circle)
variable (l1 l2 : Point → Prop) -- Representing line equations in terms of points
variable (angle : Point → Prop) -- Representing the given angle sides

-- Tangency conditions
axiom tangency1 : ∀ p : Point, l1 p → p ≠ ω1.center ∧ (ω1.center.x - p.x) ^ 2 + (ω1.center.y - p.y) ^ 2 = ω1.radius ^ 2
axiom tangency2 : ∀ p : Point, l2 p → p ≠ ω2.center ∧ (ω2.center.x - p.x) ^ 2 + (ω2.center.y - p.y) ^ 2 = ω2.radius ^ 2

-- Non-intersecting condition for circles
axiom nonintersecting : (ω1.center.x - ω2.center.x) ^ 2 + (ω1.center.y - ω2.center.y) ^ 2 > (ω1.radius + ω2.radius) ^ 2

-- Conditions for tangent circles and middle line being between them
axiom betweenness : ∀ p, angle p → (ω1.center.y < p.y ∧ p.y < ω2.center.y)

-- Midline definition and fixed circle condition
theorem midlines_tangent_fixed_circle :
  ∃ (O : Point) (d : ℝ), ∀ (T : Point → Prop), 
  (∃ (p1 p2 : Point), l1 p1 ∧ l2 p2 ∧ T p1 ∧ T p2) →
  (∀ (m : Point), T m ↔ ∃ (p1 p2 p3 p4 : Point), T p1 ∧ T p2 ∧ angle p3 ∧ angle p4 ∧ 
  m.x = (p1.x + p2.x + p3.x + p4.x) / 4 ∧ m.y = (p1.y + p2.y + p3.y + p4.y) / 4) → 
  (∀ (m : Point), (m.x - O.x) ^ 2 + (m.y - O.y) ^ 2 = d^2)
:= 
sorry

end midlines_tangent_fixed_circle_l1121_112186


namespace probability_of_three_tails_one_head_in_four_tosses_l1121_112150

noncomputable def probability_three_tails_one_head (n : ℕ) : ℚ :=
  if n = 4 then 1 / 4 else 0

theorem probability_of_three_tails_one_head_in_four_tosses :
  probability_three_tails_one_head 4 = 1 / 4 :=
by sorry

end probability_of_three_tails_one_head_in_four_tosses_l1121_112150


namespace mary_change_received_l1121_112115

def cost_of_adult_ticket : ℝ := 2
def cost_of_child_ticket : ℝ := 1
def discount_first_child : ℝ := 0.5
def discount_second_child : ℝ := 0.75
def discount_third_child : ℝ := 1
def sales_tax_rate : ℝ := 0.08
def amount_paid : ℝ := 20

def total_ticket_cost_before_tax : ℝ :=
  cost_of_adult_ticket + (cost_of_child_ticket * discount_first_child) + 
  (cost_of_child_ticket * discount_second_child) + (cost_of_child_ticket * discount_third_child)

def sales_tax : ℝ :=
  total_ticket_cost_before_tax * sales_tax_rate

def total_ticket_cost_with_tax : ℝ :=
  total_ticket_cost_before_tax + sales_tax

def change_received : ℝ :=
  amount_paid - total_ticket_cost_with_tax

theorem mary_change_received :
  change_received = 15.41 :=
by
  sorry

end mary_change_received_l1121_112115


namespace discount_difference_l1121_112173

theorem discount_difference :
  ∀ (original_price : ℝ),
  let initial_discount := 0.40
  let subsequent_discount := 0.25
  let claimed_discount := 0.60
  let actual_discount := 1 - (1 - initial_discount) * (1 - subsequent_discount)
  let difference := claimed_discount - actual_discount
  actual_discount = 0.55 ∧ difference = 0.05
:= by
  sorry

end discount_difference_l1121_112173


namespace num_boys_on_playground_l1121_112161

-- Define the conditions using Lean definitions
def num_girls : Nat := 28
def total_children : Nat := 63

-- Define a theorem to prove the number of boys
theorem num_boys_on_playground : total_children - num_girls = 35 :=
by
  -- proof steps would go here
  sorry

end num_boys_on_playground_l1121_112161


namespace three_person_subcommittees_from_seven_l1121_112121

-- Definition of the combinations formula (binomial coefficient)
def choose : ℕ → ℕ → ℕ
| n, k => if k = 0 then 1 else (n * choose (n - 1) (k - 1)) / k 

-- Problem statement in Lean 4
theorem three_person_subcommittees_from_seven : choose 7 3 = 35 :=
by
  -- We would fill in the steps here or use a sorry to skip the proof
  sorry

end three_person_subcommittees_from_seven_l1121_112121


namespace length_of_platform_l1121_112172

theorem length_of_platform (l t p : ℝ) (h1 : (l / t) = (l + p) / (5 * t)) : p = 4 * l :=
by
  sorry

end length_of_platform_l1121_112172


namespace train_speed_calculation_l1121_112129

variable (p : ℝ) (h_p : p > 0)

/-- The speed calculation of a train that covers 200 meters in p seconds is correctly given by 720 / p km/hr. -/
theorem train_speed_calculation (h_p : p > 0) : (200 / p * 3.6 = 720 / p) :=
by
  sorry

end train_speed_calculation_l1121_112129


namespace simplify_expression_l1121_112137

theorem simplify_expression (m n : ℝ) (h : m ≠ 0) : 
  (m^(4/3) - 27 * m^(1/3) * n) / 
  (m^(2/3) + 3 * (m * n)^(1/3) + 9 * n^(2/3)) / 
  (1 - 3 * (n / m)^(1/3)) - 
  (m^2)^(1/3) = 0 := 
sorry

end simplify_expression_l1121_112137


namespace find_second_expression_l1121_112118

theorem find_second_expression (a x : ℕ) (h₁ : (2 * a + 16 + x) / 2 = 84) (h₂ : a = 32) : x = 88 :=
  sorry

end find_second_expression_l1121_112118


namespace triangle_first_side_length_l1121_112122

theorem triangle_first_side_length (x : ℕ) (h1 : x + 20 + 30 = 55) : x = 5 :=
by
  sorry

end triangle_first_side_length_l1121_112122


namespace decreasing_interval_l1121_112193

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem decreasing_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → deriv f x < 0 :=
by sorry

end decreasing_interval_l1121_112193


namespace speed_of_stream_l1121_112157

-- Define the speed of the boat in still water
def speed_of_boat_in_still_water : ℝ := 39

-- Define the effective speed upstream and downstream
def effective_speed_upstream (v : ℝ) : ℝ := speed_of_boat_in_still_water - v
def effective_speed_downstream (v : ℝ) : ℝ := speed_of_boat_in_still_water + v

-- Define the condition that time upstream is twice the time downstream
def time_condition (D v : ℝ) : Prop := 
  (D / effective_speed_upstream v = 2 * (D / effective_speed_downstream v))

-- The main theorem stating the speed of the stream
theorem speed_of_stream (D : ℝ) (h : D > 0) : (v : ℝ) → time_condition D v → v = 13 :=
by
  sorry

end speed_of_stream_l1121_112157


namespace angle_C_is_150_degrees_l1121_112168

theorem angle_C_is_150_degrees
  (C D : ℝ)
  (h_supp : C + D = 180)
  (h_C_5D : C = 5 * D) :
  C = 150 :=
by
  sorry

end angle_C_is_150_degrees_l1121_112168


namespace percentage_of_volume_is_P_l1121_112155

noncomputable def volumeOfSolutionP {P Q : ℝ} (h : 0.80 * P + 0.55 * Q = 0.675 * (P + Q)) : ℝ := 
(P / (P + Q)) * 100

theorem percentage_of_volume_is_P {P Q: ℝ} (h : 0.80 * P + 0.55 * Q = 0.675 * (P + Q)) : 
  volumeOfSolutionP h = 50 :=
sorry

end percentage_of_volume_is_P_l1121_112155


namespace find_total_amount_l1121_112145

variables (A B C : ℕ) (total_amount : ℕ) 

-- Conditions
def condition1 : Prop := B = 36
def condition2 : Prop := 100 * B / 45 = A
def condition3 : Prop := 100 * C / 30 = A

-- Proof statement
theorem find_total_amount (h1 : condition1 B) (h2 : condition2 A B) (h3 : condition3 A C) :
  total_amount = 300 :=
sorry

end find_total_amount_l1121_112145


namespace tim_biking_time_l1121_112170

theorem tim_biking_time
  (work_days : ℕ := 5) 
  (distance_to_work : ℕ := 20) 
  (weekend_ride : ℕ := 200) 
  (speed : ℕ := 25) 
  (weekly_work_distance := 2 * distance_to_work * work_days)
  (total_distance := weekly_work_distance + weekend_ride) : 
  (total_distance / speed = 16) := 
by
  sorry

end tim_biking_time_l1121_112170


namespace part1_l1121_112167

theorem part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) : 
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) :=
sorry

end part1_l1121_112167


namespace remainder_of_greatest_integer_multiple_of_9_no_repeats_l1121_112164

noncomputable def greatest_integer_multiple_of_9_no_repeats : ℕ :=
  9876543210 -- this should correspond to the greatest number meeting the criteria, but it's identified via more specific logic in practice

theorem remainder_of_greatest_integer_multiple_of_9_no_repeats : 
  (greatest_integer_multiple_of_9_no_repeats % 1000) = 621 := 
  by sorry

end remainder_of_greatest_integer_multiple_of_9_no_repeats_l1121_112164


namespace calculate_total_weight_l1121_112144

variable (a b c d : ℝ)

-- Conditions
def I_II_weight := a + b = 156
def III_IV_weight := c + d = 195
def I_III_weight := a + c = 174
def II_IV_weight := b + d = 186

theorem calculate_total_weight (I_II_weight : a + b = 156) (III_IV_weight : c + d = 195)
    (I_III_weight : a + c = 174) (II_IV_weight : b + d = 186) :
    a + b + c + d = 355.5 :=
by
    sorry

end calculate_total_weight_l1121_112144


namespace second_part_of_ratio_l1121_112109

theorem second_part_of_ratio (h_ratio : ∀ (x : ℝ), 25 = 0.5 * (25 + x)) : ∃ x : ℝ, x = 25 :=
by
  sorry

end second_part_of_ratio_l1121_112109


namespace sum_of_real_roots_l1121_112104

theorem sum_of_real_roots (P : Polynomial ℝ) (hP : P = Polynomial.C 1 * X^4 - Polynomial.C 8 * X - Polynomial.C 2) :
  P.roots.sum = 2 :=
by {
  sorry
}

end sum_of_real_roots_l1121_112104


namespace not_perfect_square_T_l1121_112165

noncomputable def operation (x y : ℝ) : ℝ := (x * y + 4) / (x + y)

axiom associative {x y z : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) :
  operation x (operation y z) = operation (operation x y) z

noncomputable def T (n : ℕ) : ℝ :=
  if h : n ≥ 4 then
    (List.range (n - 2)).foldr (λ x acc => operation (x + 3) acc) 3
  else 0

theorem not_perfect_square_T (n : ℕ) (h : n ≥ 4) :
  ¬ (∃ k : ℕ, (96 / (T n - 2) : ℝ) = k ^ 2) :=
sorry

end not_perfect_square_T_l1121_112165


namespace focal_length_ellipse_l1121_112143

theorem focal_length_ellipse :
  let a := 2
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 :=
by
  sorry

end focal_length_ellipse_l1121_112143


namespace find_AX_l1121_112151

variable (A B X C : Point)
variable (AB AC BC AX XB : ℝ)
variable (angleACX angleXCB : Angle)
variable (eqAngle : angleACX = angleXCB)

axiom length_AB : AB = 80
axiom length_AC : AC = 36
axiom length_BC : BC = 72

theorem find_AX (AB AC BC AX XB : ℝ) (angleACX angleXCB : Angle)
  (eqAngle : angleACX = angleXCB)
  (h1 : AB = 80)
  (h2 : AC = 36)
  (h3 : BC = 72) : AX = 80 / 3 :=
by
  sorry

end find_AX_l1121_112151


namespace value_equation_l1121_112184

noncomputable def quarter_value := 25
noncomputable def dime_value := 10
noncomputable def half_dollar_value := 50

theorem value_equation (n : ℕ) :
  25 * quarter_value + 20 * dime_value = 15 * quarter_value + 10 * dime_value + n * half_dollar_value → 
  n = 7 :=
by
  sorry

end value_equation_l1121_112184


namespace divide_fractions_l1121_112147

theorem divide_fractions :
  (7 / 3) / (5 / 4) = (28 / 15) :=
by
  sorry

end divide_fractions_l1121_112147


namespace function_has_two_zeros_l1121_112132

/-- 
Given the function y = x + 1/(2x) + t has two zeros under the condition t > 0,
prove that the range of the real number t is (-∞, -√2).
-/
theorem function_has_two_zeros (t : ℝ) (ht : t > 0) : t < -Real.sqrt 2 :=
sorry

end function_has_two_zeros_l1121_112132


namespace counseling_rooms_l1121_112194

theorem counseling_rooms (n : ℕ) (x : ℕ)
  (h1 : n = 20 * x + 32)
  (h2 : n = 24 * (x - 1)) : x = 14 :=
by
  sorry

end counseling_rooms_l1121_112194


namespace find_A_l1121_112114

variable (p q r s A : ℝ)

theorem find_A (H1 : (p + q + r + s) / 4 = 5) (H2 : (p + q + r + s + A) / 5 = 8) : A = 20 := 
by
  sorry

end find_A_l1121_112114


namespace proof_moles_HNO3_proof_molecular_weight_HNO3_l1121_112196

variable (n_CaO : ℕ) (molar_mass_H : ℕ) (molar_mass_N : ℕ) (molar_mass_O : ℕ)

def verify_moles_HNO3 (n_CaO : ℕ) : ℕ :=
  2 * n_CaO

def verify_molecular_weight_HNO3 (molar_mass_H molar_mass_N molar_mass_O : ℕ) : ℕ :=
  molar_mass_H + molar_mass_N + 3 * molar_mass_O

theorem proof_moles_HNO3 :
  n_CaO = 7 →
  verify_moles_HNO3 n_CaO = 14 :=
sorry

theorem proof_molecular_weight_HNO3 :
  molar_mass_H = 101 / 100 ∧ molar_mass_N = 1401 / 100 ∧ molar_mass_O = 1600 / 100 →
  verify_molecular_weight_HNO3 molar_mass_H molar_mass_N molar_mass_O = 6302 / 100 :=
sorry

end proof_moles_HNO3_proof_molecular_weight_HNO3_l1121_112196


namespace solution_set_ineq_l1121_112169

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem solution_set_ineq (x : ℝ) : f (x^2 - 4) + f (3*x) > 0 ↔ x > 1 ∨ x < -4 :=
by sorry

end solution_set_ineq_l1121_112169


namespace find_principal_amount_l1121_112138

-- Definitions based on conditions
def A : ℝ := 3969
def r : ℝ := 0.05
def n : ℝ := 1
def t : ℝ := 2

-- The statement to be proved
theorem find_principal_amount : ∃ P : ℝ, A = P * (1 + r/n)^(n * t) ∧ P = 3600 :=
by
  use 3600
  sorry

end find_principal_amount_l1121_112138


namespace total_luggage_l1121_112178

theorem total_luggage (ne nb nf : ℕ)
  (leconomy lbusiness lfirst : ℕ)
  (Heconomy : ne = 10) 
  (Hbusiness : nb = 7) 
  (Hfirst : nf = 3)
  (Heconomy_luggage : leconomy = 5)
  (Hbusiness_luggage : lbusiness = 8)
  (Hfirst_luggage : lfirst = 12) : 
  (ne * leconomy + nb * lbusiness + nf * lfirst) = 142 :=
by
  sorry

end total_luggage_l1121_112178


namespace linear_function_solution_l1121_112123

theorem linear_function_solution (k : ℝ) (h₁ : k ≠ 0) (h₂ : 0 = k * (-2) + 3) :
  ∃ x : ℝ, k * (x - 5) + 3 = 0 ∧ x = 3 :=
by
  sorry

end linear_function_solution_l1121_112123


namespace number_of_classmates_l1121_112183

theorem number_of_classmates (n m : ℕ) (h₁ : n < 100) (h₂ : m = 9)
:(2 ^ 6 - 1) = 63 → 63 / m = 7 := by
  intros 
  sorry

end number_of_classmates_l1121_112183


namespace integer_solutions_xy_l1121_112133

theorem integer_solutions_xy :
  ∃ (x y : ℤ), (x + y + x * y = 500) ∧ 
               ((x = 0 ∧ y = 500) ∨ 
                (x = -2 ∧ y = -502) ∨ 
                (x = 2 ∧ y = 166) ∨ 
                (x = -4 ∧ y = -168)) :=
by
  sorry

end integer_solutions_xy_l1121_112133


namespace reasoning_classification_correct_l1121_112154

def analogical_reasoning := "reasoning from specific to specific"
def inductive_reasoning := "reasoning from part to whole and from individual to general"
def deductive_reasoning := "reasoning from general to specific"

theorem reasoning_classification_correct : 
  (analogical_reasoning, inductive_reasoning, deductive_reasoning) =
  ("reasoning from specific to specific", "reasoning from part to whole and from individual to general", "reasoning from general to specific") := 
by 
  sorry

end reasoning_classification_correct_l1121_112154


namespace problem_proof_l1121_112156

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := Real.sin (x + Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem problem_proof :
  (∀ x, g (x + Real.pi) = g x) ∧ (∀ y, g (2 * (Real.pi / 12) - y) = g y) :=
by
  sorry

end problem_proof_l1121_112156


namespace find_percentage_ryegrass_in_seed_mixture_X_l1121_112141

open Real

noncomputable def percentage_ryegrass_in_seed_mixture_X (R : ℝ) : Prop := 
  let proportion_X : ℝ := 2 / 3
  let percentage_Y_ryegrass : ℝ := 25 / 100
  let proportion_Y : ℝ := 1 / 3
  let final_percentage_ryegrass : ℝ := 35 / 100
  final_percentage_ryegrass = (R / 100 * proportion_X) + (percentage_Y_ryegrass * proportion_Y)

/-
  Given the conditions:
  - Seed mixture Y is 25 percent ryegrass.
  - A mixture of seed mixtures X (66.67% of the mixture) and Y (33.33% of the mixture) contains 35 percent ryegrass.

  Prove:
  The percentage of ryegrass in seed mixture X is 40%.
-/
theorem find_percentage_ryegrass_in_seed_mixture_X : 
  percentage_ryegrass_in_seed_mixture_X 40 := 
  sorry

end find_percentage_ryegrass_in_seed_mixture_X_l1121_112141


namespace greatest_possible_perimeter_l1121_112181

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l1121_112181


namespace value_of_f_neg_a_l1121_112101

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + x^3 + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 3) : f (-a) = -1 := 
by
  sorry

end value_of_f_neg_a_l1121_112101


namespace maximize_profit_l1121_112195

def cost_per_product : ℝ := 3
def management_fee_per_product : ℝ := 3
def sales_volume (x : ℝ) : ℝ := (12 - x)^2 * 10000
def annual_profit (x : ℝ) : ℝ := (x - cost_per_product - management_fee_per_product) * sales_volume x

theorem maximize_profit :
  (∀ x : ℝ, 9 ≤ x ∧ x ≤ 11 → annual_profit x = x^3 - 30*x^2 + 288*x - 864) ∧
  annual_profit 9 = 27 * 10000 ∧
  (∀ x : ℝ, 9 ≤ x ∧ x ≤ 11 → annual_profit x ≤ annual_profit 9) :=
by
  sorry

end maximize_profit_l1121_112195


namespace new_volume_of_cylinder_l1121_112175

theorem new_volume_of_cylinder (r h : ℝ) (π : ℝ := Real.pi) (V : ℝ := π * r^2 * h) (hV : V = 15) :
  let r_new := 3 * r
  let h_new := 4 * h
  let V_new := π * (r_new)^2 * h_new
  V_new = 540 :=
by
  sorry

end new_volume_of_cylinder_l1121_112175


namespace glass_cannot_all_be_upright_l1121_112135

def glass_flip_problem :=
  ∀ (g : Fin 6 → ℤ),
    g 0 = 1 ∧ g 1 = 1 ∧ g 2 = 1 ∧ g 3 = 1 ∧ g 4 = 1 ∧ g 5 = -1 →
    (∀ (flip : Fin 4 → Fin 6 → ℤ),
      (∃ (i1 i2 i3 i4: Fin 6), 
        flip 0 = g i1 * -1 ∧ 
        flip 1 = g i2 * -1 ∧
        flip 2 = g i3 * -1 ∧
        flip 3 = g i4 * -1) →
      ∃ j, g j ≠ 1)

theorem glass_cannot_all_be_upright : glass_flip_problem :=
  sorry

end glass_cannot_all_be_upright_l1121_112135


namespace BANANA_distinct_arrangements_l1121_112134

theorem BANANA_distinct_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 1) * (Nat.factorial 3) * (Nat.factorial 2)) = 60 := 
by
  sorry

end BANANA_distinct_arrangements_l1121_112134


namespace total_broken_marbles_l1121_112146

theorem total_broken_marbles (marbles_set1 marbles_set2 : ℕ) 
  (percentage_broken_set1 percentage_broken_set2 : ℚ) 
  (h1 : marbles_set1 = 50) 
  (h2 : percentage_broken_set1 = 0.1) 
  (h3 : marbles_set2 = 60) 
  (h4 : percentage_broken_set2 = 0.2) : 
  (marbles_set1 * percentage_broken_set1 + marbles_set2 * percentage_broken_set2 = 17) := 
by 
  sorry

end total_broken_marbles_l1121_112146


namespace max_min_f_product_of_roots_f_l1121_112179

noncomputable def f (x : ℝ) : ℝ := 
  (Real.log x / Real.log 3 - 3) * (Real.log x / Real.log 3 + 1)

theorem max_min_f
  (x : ℝ) (h : x ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ)) : 
  (∀ y, y ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ) → f y ≤ 12)
  ∧ (∀ y, y ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ) → f y ≥ 5) :=
sorry

theorem product_of_roots_f
  (m α β : ℝ) (h1 : f α + m = 0) (h2 : f β + m = 0) : 
  (Real.log (α * β) / Real.log 3 = 2) → (α * β = 9) :=
sorry

end max_min_f_product_of_roots_f_l1121_112179


namespace possible_values_a_l1121_112108

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 - a * x - 7 else a / x

theorem possible_values_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → -2 * x - a ≥ 0) ∧
  (∀ x : ℝ, x > 1 → -a / (x^2) ≥ 0) ∧
  (-8 - a ≤ a) →
  a = -2 ∨ a = -3 ∨ a = -4 :=
sorry

end possible_values_a_l1121_112108


namespace machine_sprockets_rate_l1121_112177

theorem machine_sprockets_rate:
  ∀ (h : ℝ), h > 0 → (660 / (h + 10) = (660 / h) * 1/1.1) → (660 / 1.1 / h) = 6 :=
by
  intros h h_pos h_eq
  -- Proof will be here
  sorry

end machine_sprockets_rate_l1121_112177


namespace sequence_problem_l1121_112117

theorem sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_a1 : a 1 = 1)
  (h_rec : ∀ n, a (n + 2) = 1 / (a n + 1)) (h_eq : a 100 = a 96) :
  a 2018 + a 3 = (Real.sqrt 5) / 2 :=
by
  sorry

end sequence_problem_l1121_112117


namespace smallest_a_plus_b_l1121_112125

theorem smallest_a_plus_b (a b : ℕ) (h1: 0 < a) (h2: 0 < b) (h3 : 2^10 * 7^3 = a^b) : a + b = 31 :=
sorry

end smallest_a_plus_b_l1121_112125


namespace find_area_of_triangle_l1121_112139

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

theorem find_area_of_triangle :
  let a := 10
  let b := 10
  let c := 12
  triangle_area a b c = 48 := 
by 
  sorry

end find_area_of_triangle_l1121_112139


namespace roots_of_quadratic_eq_l1121_112163

theorem roots_of_quadratic_eq (a b : ℝ) (h1 : a * (-2)^2 + b * (-2) = 6) (h2 : a * 3^2 + b * 3 = 6) :
    ∃ (x1 x2 : ℝ), x1 = -2 ∧ x2 = 3 ∧ ∀ x, a * x^2 + b * x = 6 ↔ (x = x1 ∨ x = x2) :=
by
  use -2, 3
  sorry

end roots_of_quadratic_eq_l1121_112163


namespace max_value_of_quadratic_function_l1121_112128

def quadratic_function (x : ℝ) : ℝ := -x^2 + 2*x + 4

theorem max_value_of_quadratic_function : ∃ x : ℝ, quadratic_function x = 5 ∧ ∀ y : ℝ, quadratic_function y ≤ 5 :=
by
  sorry

end max_value_of_quadratic_function_l1121_112128


namespace contrapositive_of_proposition_is_false_l1121_112124

variables {a b : ℤ}

/-- Proposition: If a and b are both even, then a + b is even -/
def proposition (a b : ℤ) : Prop :=
  (∀ n m : ℤ, a = 2 * n ∧ b = 2 * m → ∃ k : ℤ, a + b = 2 * k)

/-- Contrapositive: If a and b are not both even, then a + b is not even -/
def contrapositive (a b : ℤ) : Prop :=
  ¬(∀ n m : ℤ, a = 2 * n ∧ b = 2 * m) → ¬(∃ k : ℤ, a + b = 2 * k)

/-- The contrapositive of the proposition "If a and b are both even, then a + b is even" -/
theorem contrapositive_of_proposition_is_false :
  (contrapositive a b) = false :=
sorry

end contrapositive_of_proposition_is_false_l1121_112124


namespace geometric_sequence_a2_a4_sum_l1121_112127

theorem geometric_sequence_a2_a4_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), (∀ n, a n = a 1 * q ^ (n - 1)) ∧
    (a 2 * a 4 = 9) ∧
    (9 * (a 1 * (1 - q^4) / (1 - q)) = 10 * (a 1 * (1 - q^2) / (1 - q))) ∧
    (a 2 + a 4 = 10) :=
by
  sorry

end geometric_sequence_a2_a4_sum_l1121_112127


namespace infinitely_many_good_primes_infinitely_many_non_good_primes_l1121_112119

def is_good_prime (p : ℕ) : Prop :=
∀ a b : ℕ, a ≡ b [ZMOD p] ↔ a^3 ≡ b^3 [ZMOD p]

theorem infinitely_many_good_primes :
  ∃ᶠ p in at_top, is_good_prime p := sorry

theorem infinitely_many_non_good_primes :
  ∃ᶠ p in at_top, ¬ is_good_prime p := sorry

end infinitely_many_good_primes_infinitely_many_non_good_primes_l1121_112119


namespace simplify_expression_l1121_112197

theorem simplify_expression
  (x y : ℝ)
  (h : (x + 2)^3 ≠ (y - 2)^3) :
  ( (x + 2)^3 + (y + x)^3 ) / ( (x + 2)^3 - (y - 2)^3 ) = (2 * x + y + 2) / (x - y + 4) :=
sorry

end simplify_expression_l1121_112197


namespace tan_theta_l1121_112112

theorem tan_theta (θ : ℝ) (h : Real.sin (θ / 2) - 2 * Real.cos (θ / 2) = 0) : Real.tan θ = -4 / 3 :=
sorry

end tan_theta_l1121_112112


namespace one_percent_as_decimal_l1121_112189

theorem one_percent_as_decimal : (1 / 100 : ℝ) = 0.01 := 
by 
  sorry

end one_percent_as_decimal_l1121_112189


namespace quadratic_intersection_l1121_112105

theorem quadratic_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_intersection_l1121_112105


namespace mean_value_of_quadrilateral_angles_l1121_112116

theorem mean_value_of_quadrilateral_angles : 
  (∀ (A B C D : ℕ), A + B + C + D = 360 → (A + B + C + D) / 4 = 90) :=
by {
  sorry
}

end mean_value_of_quadrilateral_angles_l1121_112116


namespace translate_point_correct_l1121_112107

def P : ℝ × ℝ := (2, 3)

def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - d)

theorem translate_point_correct :
  translate_down (translate_left P 3) 4 = (-1, -1) :=
by
  sorry

end translate_point_correct_l1121_112107


namespace quadratic_not_divisible_by_49_l1121_112140

theorem quadratic_not_divisible_by_49 (n : ℤ) : ¬ (n^2 + 3 * n + 4) % 49 = 0 := 
by
  sorry

end quadratic_not_divisible_by_49_l1121_112140


namespace race_winner_l1121_112162

theorem race_winner
  (faster : String → String → Prop)
  (Minyoung Yoongi Jimin Yuna : String)
  (cond1 : faster Minyoung Yoongi)
  (cond2 : faster Yoongi Jimin)
  (cond3 : faster Yuna Jimin)
  (cond4 : faster Yuna Minyoung) :
  ∀ s, s ≠ Yuna → faster Yuna s :=
by
  sorry

end race_winner_l1121_112162


namespace hyperbola_equation_focus_and_eccentricity_l1121_112102

theorem hyperbola_equation_focus_and_eccentricity (a b : ℝ)
  (h_focus : ∃ c : ℝ, c = 1 ∧ (∃ c_squared : ℝ, c_squared = c ^ 2))
  (h_eccentricity : ∃ e : ℝ, e = Real.sqrt 5 ∧ e = c / a)
  (h_b : b ^ 2 = c ^ 2 - a ^ 2) :
  5 * x^2 - (5 / 4) * y^2 = 1 :=
sorry

end hyperbola_equation_focus_and_eccentricity_l1121_112102


namespace solution_set_of_inequality_l1121_112126

theorem solution_set_of_inequality (a : ℝ) :
  (a > 1 → {x : ℝ | ax + 1 < a^2 + x} = {x : ℝ | x < a + 1}) ∧
  (a < 1 → {x : ℝ | ax + 1 < a^2 + x} = {x : ℝ | x > a + 1}) ∧
  (a = 1 → {x : ℝ | ax + 1 < a^2 + x} = ∅) := 
  sorry

end solution_set_of_inequality_l1121_112126
