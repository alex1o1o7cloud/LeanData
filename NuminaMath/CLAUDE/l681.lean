import Mathlib

namespace min_time_circular_chain_no_faster_solution_l681_68166

/-- Represents a chain piece with a certain number of links -/
structure ChainPiece where
  links : ℕ

/-- Represents the time required for chain operations -/
structure ChainOperations where
  cutTime : ℕ
  joinTime : ℕ

/-- Calculates the minimum time required to form a circular chain -/
def minTimeToCircularChain (pieces : List ChainPiece) (ops : ChainOperations) : ℕ :=
  sorry

/-- Theorem stating the minimum time to form a circular chain from given pieces -/
theorem min_time_circular_chain :
  let pieces := [
    ChainPiece.mk 10,
    ChainPiece.mk 10,
    ChainPiece.mk 8,
    ChainPiece.mk 8,
    ChainPiece.mk 5,
    ChainPiece.mk 2
  ]
  let ops := ChainOperations.mk 1 2
  minTimeToCircularChain pieces ops = 15 := by
  sorry

/-- Theorem stating that it's impossible to form the circular chain in less than 15 minutes -/
theorem no_faster_solution (t : ℕ) :
  let pieces := [
    ChainPiece.mk 10,
    ChainPiece.mk 10,
    ChainPiece.mk 8,
    ChainPiece.mk 8,
    ChainPiece.mk 5,
    ChainPiece.mk 2
  ]
  let ops := ChainOperations.mk 1 2
  t < 15 → minTimeToCircularChain pieces ops ≠ t := by
  sorry

end min_time_circular_chain_no_faster_solution_l681_68166


namespace integral_exp_plus_2x_equals_e_l681_68197

theorem integral_exp_plus_2x_equals_e :
  ∫ x in (0:ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 := by
sorry

end integral_exp_plus_2x_equals_e_l681_68197


namespace markus_family_ages_l681_68198

theorem markus_family_ages (grandson_age : ℕ) : 
  grandson_age > 0 →
  let son_age := 2 * grandson_age
  let markus_age := 2 * son_age
  grandson_age + son_age + markus_age = 140 →
  grandson_age = 20 := by
sorry

end markus_family_ages_l681_68198


namespace find_number_l681_68114

theorem find_number : ∃ x : ℝ, (3 * x / 5 - 220) * 4 + 40 = 360 ∧ x = 500 := by
  sorry

end find_number_l681_68114


namespace square_root_problem_l681_68148

theorem square_root_problem (a b : ℝ) : 
  ((2 * a - 1)^2 = 4) → (b = 1) → (2 * a - b = 2 ∨ 2 * a - b = -2) := by
  sorry

end square_root_problem_l681_68148


namespace distance_between_parallel_lines_l681_68106

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ x - y + 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ x - y + 3 = 0
  ∀ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ → l₂ x₂ y₂ →
  (∃ (k : ℝ), ∀ (x y : ℝ), l₁ x y ↔ l₂ (x + k) (y + k)) →
  Real.sqrt 2 = |x₂ - x₁| :=
by sorry

end distance_between_parallel_lines_l681_68106


namespace solution_to_inequalities_l681_68172

theorem solution_to_inequalities :
  let x : ℚ := -1/3
  let y : ℚ := 2/3
  (11 * x^2 + 8 * x * y + 8 * y^2 ≤ 3) ∧ (x - 4 * y ≤ -3) := by
  sorry

end solution_to_inequalities_l681_68172


namespace square_construction_impossibility_l681_68126

theorem square_construction_impossibility (k : ℕ) (h : k ≥ 2) :
  ¬ (∃ (S : Finset (ℕ × ℕ)), 
    (∀ (x : ℕ × ℕ), x ∈ S → x.1 = 1 ∧ x.2 ≤ k) ∧ 
    (S.sum (λ x => x.1 * x.2) = k * k) ∧
    (S.card ≤ k)) := by
  sorry

end square_construction_impossibility_l681_68126


namespace pet_shop_dogs_l681_68181

theorem pet_shop_dogs (birds : ℕ) (snakes : ℕ) (spider : ℕ) (total_legs : ℕ) :
  birds = 3 → snakes = 4 → spider = 1 → total_legs = 34 →
  ∃ dogs : ℕ, dogs = 5 ∧ total_legs = birds * 2 + dogs * 4 + spider * 8 :=
by sorry

end pet_shop_dogs_l681_68181


namespace one_fourth_of_8_4_l681_68195

theorem one_fourth_of_8_4 : 
  ∃ (n d : ℕ), n ≠ 0 ∧ d ≠ 0 ∧ (8.4 / 4 : ℚ) = n / d ∧ Nat.gcd n d = 1 :=
by
  -- The proof would go here
  sorry

end one_fourth_of_8_4_l681_68195


namespace platform_length_l681_68171

/-- Given a train and platform with specific properties, prove the platform length --/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 30)
  (h3 : pole_crossing_time = 18) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 200 := by
  sorry

end platform_length_l681_68171


namespace temple_visit_theorem_l681_68118

/-- The number of people who went to the temple with Nathan -/
def number_of_people : ℕ := 3

/-- The cost per object in dollars -/
def cost_per_object : ℕ := 11

/-- The number of objects per person -/
def objects_per_person : ℕ := 5

/-- The total charge for all objects in dollars -/
def total_charge : ℕ := 165

/-- Theorem stating that the number of people is correct given the conditions -/
theorem temple_visit_theorem : 
  number_of_people * objects_per_person * cost_per_object = total_charge :=
by sorry

end temple_visit_theorem_l681_68118


namespace log_stack_sum_l681_68107

/-- The sum of an arithmetic sequence with first term a, last term l, and n terms -/
def arithmetic_sum (a l n : ℕ) : ℕ := n * (a + l) / 2

/-- The number of terms in the sequence of logs -/
def num_terms : ℕ := 15 - 5 + 1

theorem log_stack_sum :
  arithmetic_sum 5 15 num_terms = 110 := by
  sorry

end log_stack_sum_l681_68107


namespace symmetric_line_wrt_x_axis_l681_68112

/-- Given a line with equation 3x-4y+5=0, this theorem states that its symmetric line
    with respect to the x-axis has the equation 3x+4y+5=0. -/
theorem symmetric_line_wrt_x_axis :
  ∀ (x y : ℝ), (3 * x - 4 * y + 5 = 0) →
  ∃ (x' y' : ℝ), (x' = x ∧ y' = -y) ∧ (3 * x' + 4 * y' + 5 = 0) :=
sorry

end symmetric_line_wrt_x_axis_l681_68112


namespace eva_age_is_six_l681_68104

-- Define the set of ages
def ages : Finset ℕ := {2, 4, 6, 8, 10}

-- Define the condition for park visit
def park_visit (a b : ℕ) : Prop := a + b = 12 ∧ a ∈ ages ∧ b ∈ ages ∧ a ≠ b

-- Define the condition for concert visit
def concert_visit : Prop := 2 ∈ ages ∧ 10 ∈ ages

-- Define the condition for staying home
def stay_home (eva_age : ℕ) : Prop := eva_age ∈ ages ∧ 4 ∈ ages

-- Theorem statement
theorem eva_age_is_six :
  ∃ (a b : ℕ), park_visit a b ∧ concert_visit ∧ stay_home 6 →
  ∃! (eva_age : ℕ), eva_age ∈ ages ∧ eva_age ≠ 2 ∧ eva_age ≠ 4 ∧ eva_age ≠ 8 ∧ eva_age ≠ 10 :=
by sorry

end eva_age_is_six_l681_68104


namespace sum_of_a_and_b_l681_68134

theorem sum_of_a_and_b (a b : ℝ) 
  (h1 : Real.sqrt 44 = 2 * Real.sqrt a) 
  (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : 
  a + b = 17 := by
  sorry

end sum_of_a_and_b_l681_68134


namespace pedal_triangles_existence_and_angles_l681_68122

/-- A triangle with angles given in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

/-- The pedal triangle of a given triangle -/
structure PedalTriangle where
  original : Triangle
  pedal : Triangle

/-- The theorem statement -/
theorem pedal_triangles_existence_and_angles 
  (T : Triangle) 
  (h1 : T.angle1 = 24) 
  (h2 : T.angle2 = 60) 
  (h3 : T.angle3 = 96) : 
  ∃! (pedals : Finset PedalTriangle), 
    Finset.card pedals = 4 ∧ 
    ∀ P ∈ pedals, 
      (P.pedal.angle1 = 102 ∧ 
       P.pedal.angle2 = 30 ∧ 
       P.pedal.angle3 = 48) := by
  sorry


end pedal_triangles_existence_and_angles_l681_68122


namespace unique_number_with_three_prime_divisors_including_31_l681_68192

theorem unique_number_with_three_prime_divisors_including_31 (x n : ℕ) :
  x = 8^n - 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 31 ∧ q ≠ 31 ∧
    x = p * q * 31 ∧ 
    ∀ r : ℕ, Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 31)) →
  x = 32767 := by
sorry

end unique_number_with_three_prime_divisors_including_31_l681_68192


namespace cousins_arrangement_l681_68190

/-- The number of ways to arrange n indistinguishable objects into k distinct boxes -/
def arrange (n k : ℕ) : ℕ := sorry

/-- There are 4 rooms available -/
def num_rooms : ℕ := 4

/-- There are 5 cousins to arrange -/
def num_cousins : ℕ := 5

/-- The number of arrangements of 5 cousins in 4 rooms is 76 -/
theorem cousins_arrangement : arrange num_cousins num_rooms = 76 := by sorry

end cousins_arrangement_l681_68190


namespace present_difference_l681_68174

/-- The number of presents Santana buys for her brothers in a year -/
def presentCount : ℕ → ℕ
| 1 => 3  -- March (first half)
| 2 => 1  -- October (second half)
| 3 => 1  -- November (second half)
| 4 => 2  -- December (second half)
| _ => 0

/-- The total number of brothers Santana has -/
def totalBrothers : ℕ := 7

/-- The number of presents bought in the first half of the year -/
def firstHalfPresents : ℕ := presentCount 1

/-- The number of presents bought in the second half of the year -/
def secondHalfPresents : ℕ := presentCount 2 + presentCount 3 + presentCount 4 + totalBrothers

theorem present_difference : secondHalfPresents - firstHalfPresents = 8 := by
  sorry

end present_difference_l681_68174


namespace cubic_function_value_l681_68135

/-- Given a cubic function f(x) = ax^3 + bx - 4 where f(-2) = 2, prove that f(2) = -10 -/
theorem cubic_function_value (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x - 4)
  (h2 : f (-2) = 2) : 
  f 2 = -10 := by
  sorry

end cubic_function_value_l681_68135


namespace speaking_orders_count_l681_68159

def total_people : Nat := 7
def speakers : Nat := 4
def special_people : Nat := 2  -- A and B

theorem speaking_orders_count : 
  (total_people.choose speakers * speakers.factorial - 
   (total_people - special_people).choose speakers * speakers.factorial) = 720 := by
  sorry

end speaking_orders_count_l681_68159


namespace arithmetic_computation_l681_68137

theorem arithmetic_computation : 2 + 5 * 3^2 - 4 + 6 * 2 / 3 = 47 := by
  sorry

end arithmetic_computation_l681_68137


namespace expression_equals_1997_with_ten_threes_l681_68127

theorem expression_equals_1997_with_ten_threes : 
  ∃ (a b c d e f g h i j : ℕ), 
    a = 3 ∧ b = 3 ∧ c = 3 ∧ d = 3 ∧ e = 3 ∧ f = 3 ∧ g = 3 ∧ h = 3 ∧ i = 3 ∧ j = 3 ∧
    a * (b * 111 + c) + d * (e * 111 + f) - g / h = 1997 :=
by sorry

end expression_equals_1997_with_ten_threes_l681_68127


namespace intersection_equals_interval_l681_68101

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 8 < 0}
def B : Set ℝ := {x : ℝ | x ≥ 0}

-- Define the interval [0, 4)
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 4}

-- Theorem statement
theorem intersection_equals_interval : A ∩ B = interval := by sorry

end intersection_equals_interval_l681_68101


namespace transportation_charges_proof_l681_68102

def transportation_charges (purchase_price repair_cost profit_percentage actual_selling_price : ℕ) : ℕ :=
  let total_cost_before_transport := purchase_price + repair_cost
  let profit := (total_cost_before_transport * profit_percentage) / 100
  let calculated_selling_price := total_cost_before_transport + profit
  actual_selling_price - calculated_selling_price

theorem transportation_charges_proof :
  transportation_charges 9000 5000 50 22500 = 1500 := by
  sorry

end transportation_charges_proof_l681_68102


namespace unique_solution_implies_a_values_l681_68178

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x + 2 = 0}

theorem unique_solution_implies_a_values (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 9/8 := by
  sorry

end unique_solution_implies_a_values_l681_68178


namespace survey_selection_theorem_l681_68146

-- Define the number of boys and girls
def num_boys : ℕ := 4
def num_girls : ℕ := 2

-- Define the total number of students to be selected
def num_selected : ℕ := 4

-- Define the function to calculate the number of ways to select students
def num_ways_to_select : ℕ := (num_boys + num_girls).choose num_selected - num_boys.choose num_selected

-- Theorem statement
theorem survey_selection_theorem : num_ways_to_select = 14 := by
  sorry

end survey_selection_theorem_l681_68146


namespace sixth_plate_cookies_l681_68141

def cookie_sequence (n : ℕ) : ℕ → ℕ
  | 0 => 5
  | 1 => 7
  | k + 2 => cookie_sequence n (k + 1) + (k + 2)

theorem sixth_plate_cookies :
  cookie_sequence 5 5 = 25 := by
  sorry

end sixth_plate_cookies_l681_68141


namespace right_triangle_acute_angles_l681_68164

theorem right_triangle_acute_angles (α : Real) 
  (h1 : 0 < α ∧ α < 90) 
  (h2 : (90 - α / 2) / (45 + α / 2) = 13 / 17) : 
  α = 63 ∧ 90 - α = 27 := by
sorry

end right_triangle_acute_angles_l681_68164


namespace line_point_sum_l681_68144

/-- The line equation y = -1/2x + 8 -/
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

/-- Point P is where the line crosses the x-axis -/
def point_P : ℝ × ℝ := (16, 0)

/-- Point Q is where the line crosses the y-axis -/
def point_Q : ℝ × ℝ := (0, 8)

/-- Point T is on the line segment PQ -/
def point_T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
  r = t * point_P.1 + (1 - t) * point_Q.1 ∧
  s = t * point_P.2 + (1 - t) * point_Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((1/2) * point_P.1 * point_Q.2) = 4 * abs ((1/2) * r * s)

theorem line_point_sum (r s : ℝ) :
  line_equation r s →
  point_T_on_PQ r s →
  area_condition r s →
  r + s = 14 :=
by sorry

end line_point_sum_l681_68144


namespace min_balls_to_draw_l681_68156

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  white : ℕ
  black : ℕ

/-- The minimum number of balls needed to guarantee a specific count of one color -/
def minBallsForGuarantee (counts : BallCounts) (targetCount : ℕ) : ℕ :=
  (min counts.red (targetCount - 1)) +
  (min counts.green (targetCount - 1)) +
  (min counts.yellow (targetCount - 1)) +
  (min counts.blue (targetCount - 1)) +
  (min counts.white (targetCount - 1)) +
  (min counts.black (targetCount - 1)) + 1

/-- Theorem stating the minimum number of balls to draw for the given problem -/
theorem min_balls_to_draw (counts : BallCounts)
    (h_red : counts.red = 35)
    (h_green : counts.green = 25)
    (h_yellow : counts.yellow = 22)
    (h_blue : counts.blue = 15)
    (h_white : counts.white = 14)
    (h_black : counts.black = 12) :
    minBallsForGuarantee counts 18 = 93 := by
  sorry


end min_balls_to_draw_l681_68156


namespace common_solution_range_l681_68132

theorem common_solution_range (x y : ℝ) : 
  (∃ x, x^2 + y^2 - 11 = 0 ∧ x^2 - 4*y + 7 = 0) ↔ 7/4 ≤ y ∧ y ≤ Real.sqrt 11 :=
by sorry

end common_solution_range_l681_68132


namespace fourteenth_root_of_unity_l681_68147

open Complex

theorem fourteenth_root_of_unity : ∃ (n : ℕ) (h : n ≤ 13),
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) =
  Complex.exp (2 * Real.pi * (n : ℝ) * Complex.I / 14) :=
by sorry

end fourteenth_root_of_unity_l681_68147


namespace students_taking_history_l681_68161

theorem students_taking_history 
  (total_students : ℕ) 
  (statistics_students : ℕ)
  (history_or_statistics : ℕ)
  (history_not_statistics : ℕ)
  (h_total : total_students = 89)
  (h_statistics : statistics_students = 32)
  (h_history_or_stats : history_or_statistics = 59)
  (h_history_not_stats : history_not_statistics = 27) :
  ∃ history_students : ℕ, history_students = 54 :=
by sorry

end students_taking_history_l681_68161


namespace divisibility_implication_l681_68184

theorem divisibility_implication (m : ℕ+) (h : 39 ∣ m^2) : 39 ∣ m := by
  sorry

end divisibility_implication_l681_68184


namespace sum_of_solutions_l681_68183

theorem sum_of_solutions (x : ℝ) : 
  (5 * x^2 - 3 * x - 2 = 0) → 
  (∃ y : ℝ, 5 * y^2 - 3 * y - 2 = 0 ∧ x + y = 3/5) :=
by sorry

end sum_of_solutions_l681_68183


namespace same_solution_equations_l681_68199

theorem same_solution_equations (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 6 = 0 ∧ c * x - 15 = -3) → c = -6 := by
sorry

end same_solution_equations_l681_68199


namespace inequality_solution_set_l681_68105

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x + 1) < 1) ↔ (x ∈ Set.Iio (-1) ∪ Set.Ioi 1) :=
sorry

end inequality_solution_set_l681_68105


namespace line_equation_l681_68130

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically --/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + dy - l.slope * dx }

/-- Check if two lines are identical --/
def Line.identical (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept = l2.intercept

/-- Check if two lines are symmetric about a point --/
def symmetricAbout (l1 l2 : Line) (p : ℝ × ℝ) : Prop :=
  ∀ x y, (y = l1.slope * x + l1.intercept) ↔ 
         (2 * p.2 - y = l2.slope * (2 * p.1 - x) + l2.intercept)

theorem line_equation (l : Line) : 
  (translate (translate l 3 5) 1 (-2)).identical l ∧ 
  symmetricAbout l (translate l 3 5) (2, 3) →
  l.slope = 3/4 ∧ l.intercept = 1/8 :=
by sorry

end line_equation_l681_68130


namespace xiao_ming_distance_l681_68194

/-- The distance between Xiao Ming's house and school -/
def distance : ℝ := 1500

/-- The original planned speed in meters per minute -/
def original_speed : ℝ := 200

/-- The reduced speed due to rain in meters per minute -/
def reduced_speed : ℝ := 120

/-- The additional time taken due to reduced speed in minutes -/
def additional_time : ℝ := 5

theorem xiao_ming_distance :
  distance = original_speed * (distance / reduced_speed - additional_time) :=
sorry

end xiao_ming_distance_l681_68194


namespace circle_tangent_to_parallel_lines_l681_68196

/-- A circle is tangent to two parallel lines and its center lies on a third line -/
theorem circle_tangent_to_parallel_lines (x y : ℚ) :
  (3 * x + 4 * y = 40) ∧ 
  (3 * x + 4 * y = -20) ∧ 
  (x - 3 * y = 0) →
  x = 30 / 13 ∧ y = 10 / 13 := by
  sorry

#check circle_tangent_to_parallel_lines

end circle_tangent_to_parallel_lines_l681_68196


namespace simplify_sqrt_sum_l681_68163

theorem simplify_sqrt_sum : 
  Real.sqrt 0.5 + Real.sqrt (0.5 + 1.5) + Real.sqrt (0.5 + 1.5 + 2.5) + 
  Real.sqrt (0.5 + 1.5 + 2.5 + 3.5) = Real.sqrt 0.5 + 3 * Real.sqrt 2 + Real.sqrt 4.5 := by
  sorry

end simplify_sqrt_sum_l681_68163


namespace parabola_vertex_l681_68180

/-- 
Given a parabola y = -x^2 + px + q where the solution to y ≤ 0 is (-∞, -4] ∪ [6, ∞),
prove that the vertex of the parabola is (1, 25).
-/
theorem parabola_vertex (p q : ℝ) : 
  (∀ x, -x^2 + p*x + q ≤ 0 ↔ x ≤ -4 ∨ x ≥ 6) →
  ∃ x y, x = 1 ∧ y = 25 ∧ ∀ t, -t^2 + p*t + q ≤ y := by
  sorry

end parabola_vertex_l681_68180


namespace geometric_mean_minimum_l681_68191

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y ≥ 9) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ Real.sqrt 2 = Real.sqrt (4^x * 2^y) ∧ 2/x + 1/y = 9) :=
by sorry

end geometric_mean_minimum_l681_68191


namespace parallel_vectors_tan_theta_l681_68140

theorem parallel_vectors_tan_theta (θ : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (a : Fin 2 → ℝ) (b : Fin 2 → ℝ)
  (h_a : a = ![1 - Real.sin θ, 1])
  (h_b : b = ![1 / 2, 1 + Real.sin θ])
  (h_parallel : ∃ (k : ℝ), a = k • b) :
  Real.tan θ = 1 := by
sorry

end parallel_vectors_tan_theta_l681_68140


namespace factorization_equality_l681_68167

theorem factorization_equality (a b : ℝ) : 6 * a^2 * b - 3 * a * b = 3 * a * b * (2 * a - 1) := by
  sorry

end factorization_equality_l681_68167


namespace angle_between_lines_l681_68145

theorem angle_between_lines (r₁ r₂ r₃ : ℝ) (shaded_ratio : ℝ) :
  r₁ = 3 ∧ r₂ = 2 ∧ r₃ = 1 ∧ shaded_ratio = 8/13 →
  ∃ θ : ℝ, 
    θ > 0 ∧ 
    θ < π/2 ∧
    (6 * θ + 4 * π = 24 * π / 7) ∧
    θ = π/7 :=
sorry

end angle_between_lines_l681_68145


namespace ivy_collectors_edition_fraction_l681_68142

theorem ivy_collectors_edition_fraction (dina_dolls : ℕ) (ivy_collectors : ℕ) : 
  dina_dolls = 60 →
  ivy_collectors = 20 →
  2 * (dina_dolls / 2) = dina_dolls →
  (ivy_collectors : ℚ) / (dina_dolls / 2 : ℚ) = 2 / 3 :=
by sorry

end ivy_collectors_edition_fraction_l681_68142


namespace sum_six_consecutive_integers_l681_68173

/-- The sum of six consecutive integers starting from n is equal to 6n + 15 -/
theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end sum_six_consecutive_integers_l681_68173


namespace f_shifted_is_even_f_monotonicity_f_satisfies_properties_l681_68154

-- Define the function f(x) = (x-2)^2
def f (x : ℝ) : ℝ := (x - 2)^2

-- Property 1: f(x+2) is an even function
theorem f_shifted_is_even : ∀ x : ℝ, f (x + 2) = f (-x + 2) := by sorry

-- Property 2: f(x) is decreasing on (-∞, 2) and increasing on (2, +∞)
theorem f_monotonicity :
  (∀ x y : ℝ, x < y → y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x → x < y → f x < f y) := by sorry

-- Theorem combining both properties
theorem f_satisfies_properties : 
  (∀ x : ℝ, f (x + 2) = f (-x + 2)) ∧
  (∀ x y : ℝ, x < y → y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x → x < y → f x < f y) := by sorry

end f_shifted_is_even_f_monotonicity_f_satisfies_properties_l681_68154


namespace max_area_quadrilateral_in_circle_l681_68168

theorem max_area_quadrilateral_in_circle (d : Real) 
  (h1 : 0 ≤ d) (h2 : d < 1) : 
  ∃ (max_area : Real),
    (d < Real.sqrt 2 / 2 → max_area = 2 * Real.sqrt (1 - d^2)) ∧
    (Real.sqrt 2 / 2 ≤ d → max_area = 1 / d) ∧
    ∀ (area : Real), area ≤ max_area :=
by sorry

end max_area_quadrilateral_in_circle_l681_68168


namespace factorization_equality_l681_68115

theorem factorization_equality (x y : ℝ) : 
  x^2 + 4*y^2 - 4*x*y - 1 = (x - 2*y + 1)*(x - 2*y - 1) := by
  sorry

end factorization_equality_l681_68115


namespace square_difference_l681_68162

theorem square_difference : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end square_difference_l681_68162


namespace tomato_field_area_l681_68152

/-- Given a rectangular field with length 3.6 meters and width 2.5 times the length,
    the area of half of this field is 16.2 square meters. -/
theorem tomato_field_area :
  let length : ℝ := 3.6
  let width : ℝ := 2.5 * length
  let total_area : ℝ := length * width
  let tomato_area : ℝ := total_area / 2
  tomato_area = 16.2 := by
sorry

end tomato_field_area_l681_68152


namespace final_amount_after_bets_l681_68151

/-- Calculates the final amount after a series of bets -/
def finalAmount (initialAmount : ℚ) (numBets numWins numLosses : ℕ) : ℚ :=
  initialAmount * (3/2)^numWins * (1/2)^numLosses

/-- Theorem stating the final amount after 7 bets with 4 wins and 3 losses -/
theorem final_amount_after_bets :
  finalAmount 128 7 4 3 = 81 := by
  sorry

end final_amount_after_bets_l681_68151


namespace expression_value_l681_68157

theorem expression_value (x y : ℝ) (h1 : x / (2 * y) = 3 / 2) (h2 : y ≠ 0) :
  (7 * x + 4 * y) / (x - 2 * y) = 25 := by
  sorry

end expression_value_l681_68157


namespace circle_properties_l681_68124

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Theorem statement
theorem circle_properties :
  ∃ (a : ℝ),
    -- P is on circle C
    C a (a + 1) ∧
    -- |PQ| = 2√10
    (a - (-2))^2 + ((a + 1) - 3)^2 = 40 ∧
    -- Slope of PQ is 1/3
    (3 - (a + 1)) / (-2 - a) = 1/3 ∧
    -- For any point M on C
    ∀ (m n : ℝ), C m n →
      -- Maximum value of |MQ| is 6√2
      (m - (-2))^2 + (n - 3)^2 ≤ 72 ∧
      -- Minimum value of |MQ| is 2√2
      (m - (-2))^2 + (n - 3)^2 ≥ 8 ∧
      -- Maximum value of (n-3)/(m+2) is 2 + √3
      (n - 3) / (m + 2) ≤ 2 + Real.sqrt 3 ∧
      -- Minimum value of (n-3)/(m+2) is 2 - √3
      (n - 3) / (m + 2) ≥ 2 - Real.sqrt 3 :=
by sorry

end circle_properties_l681_68124


namespace f_negative_six_equals_negative_one_l681_68128

def f (x : ℝ) : ℝ := sorry

theorem f_negative_six_equals_negative_one :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x, f (x + 6) = f x) →  -- f has period 6
  (∀ x, -3 ≤ x ∧ x ≤ 3 → f x = (x + 1) * (x - 1)) →  -- f(x) = (x+1)(x-a) for -3 ≤ x ≤ 3, where a = 1
  f (-6) = -1 := by
  sorry

end f_negative_six_equals_negative_one_l681_68128


namespace complex_equation_solution_l681_68185

theorem complex_equation_solution : 
  ∃ (z : ℂ), z / (1 - Complex.I) = Complex.I ^ 2019 → z = -1 - Complex.I := by sorry

end complex_equation_solution_l681_68185


namespace thirty_six_in_binary_l681_68143

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- The binary representation of 36 -/
def binary_36 : List Bool := [false, false, true, false, false, true]

/-- Theorem stating that the binary representation of 36 is 100100₂ -/
theorem thirty_six_in_binary :
  to_binary 36 = binary_36 := by sorry

end thirty_six_in_binary_l681_68143


namespace eight_divided_by_repeating_third_l681_68100

-- Define the repeating decimal 0.333...
def repeating_third : ℚ := 1 / 3

-- State the theorem
theorem eight_divided_by_repeating_third : 8 / repeating_third = 24 := by
  sorry

end eight_divided_by_repeating_third_l681_68100


namespace distribute_five_students_three_dorms_l681_68119

/-- The number of ways to distribute students into dormitories -/
def distribute_students (n : ℕ) (m : ℕ) (min : ℕ) (max : ℕ) (restricted : ℕ) : ℕ := sorry

/-- The theorem stating the number of ways to distribute 5 students into 3 dormitories -/
theorem distribute_five_students_three_dorms :
  distribute_students 5 3 1 2 1 = 60 := by sorry

end distribute_five_students_three_dorms_l681_68119


namespace sasha_picked_24_leaves_l681_68187

/-- The number of apple trees along the road. -/
def apple_trees : ℕ := 17

/-- The number of poplar trees along the road. -/
def poplar_trees : ℕ := 20

/-- The index of the apple tree from which Sasha starts picking leaves. -/
def start_tree : ℕ := 8

/-- The total number of trees along the road. -/
def total_trees : ℕ := apple_trees + poplar_trees

/-- The number of leaves Sasha picked. -/
def leaves_picked : ℕ := total_trees - (start_tree - 1)

theorem sasha_picked_24_leaves : leaves_picked = 24 := by
  sorry

end sasha_picked_24_leaves_l681_68187


namespace rani_cycling_speed_difference_l681_68193

/-- Rani's cycling speed as a girl in miles per minute -/
def girl_speed : ℚ := 20 / (2 * 60 + 45)

/-- Rani's cycling speed as an older woman in miles per minute -/
def woman_speed : ℚ := 12 / (3 * 60)

/-- The difference in minutes per mile between Rani's cycling speed as an older woman and as a girl -/
def speed_difference : ℚ := (1 / woman_speed) - (1 / girl_speed)

theorem rani_cycling_speed_difference :
  speed_difference = 6.75 := by sorry

end rani_cycling_speed_difference_l681_68193


namespace problem_statement_l681_68129

theorem problem_statement (x : ℝ) (h : x = 2) : 4 * x^2 + (1/2) = 16.5 := by
  sorry

end problem_statement_l681_68129


namespace intersection_of_A_and_B_l681_68139

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {(x, y) | y = 2 * x - 1}
def B : Set (ℝ × ℝ) := {(x, y) | y = x + 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {(4, 7)} := by sorry

end intersection_of_A_and_B_l681_68139


namespace arithmetic_sequence_sum_l681_68138

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2015 = 10) →
  a 2 + a 1008 + a 2014 = 15 := by
  sorry

end arithmetic_sequence_sum_l681_68138


namespace dog_grouping_combinations_l681_68125

def total_dogs : Nat := 12
def group1_size : Nat := 4
def group2_size : Nat := 5
def group3_size : Nat := 3

theorem dog_grouping_combinations :
  (total_dogs = group1_size + group2_size + group3_size) →
  (Nat.choose (total_dogs - 2) (group1_size - 1) * Nat.choose (total_dogs - group1_size - 1) (group2_size - 1) = 5775) := by
  sorry

end dog_grouping_combinations_l681_68125


namespace hockey_league_teams_l681_68131

/-- The number of games played in the hockey season -/
def total_games : ℕ := 1710

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- Calculates the total number of games in a season based on the number of teams -/
def calculate_games (n : ℕ) : ℕ :=
  (n * (n - 1) * games_per_pair) / 2

theorem hockey_league_teams :
  ∃ (n : ℕ), n > 0 ∧ calculate_games n = total_games :=
sorry

end hockey_league_teams_l681_68131


namespace shooting_competition_l681_68111

theorem shooting_competition (p_tie p_win : ℝ) 
  (h_tie : p_tie = 1/2)
  (h_win : p_win = 1/3) :
  p_tie + p_win = 5/6 := by
sorry

end shooting_competition_l681_68111


namespace trip_time_calculation_l681_68103

theorem trip_time_calculation (normal_distance : ℝ) (normal_time : ℝ) (additional_distance : ℝ) :
  normal_distance = 150 →
  normal_time = 3 →
  additional_distance = 100 →
  let speed := normal_distance / normal_time
  let total_distance := normal_distance + additional_distance
  let total_time := total_distance / speed
  total_time = 5 := by
  sorry

end trip_time_calculation_l681_68103


namespace equation_solutions_count_l681_68155

theorem equation_solutions_count : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (x^2 + x - 12)^2 = 81) ∧ s.card = 4 := by
  sorry

end equation_solutions_count_l681_68155


namespace square_root_of_1024_l681_68150

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end square_root_of_1024_l681_68150


namespace triangle_side_b_l681_68186

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_side_b (t : Triangle) 
  (h1 : t.a = Real.sqrt 3)
  (h2 : t.A = 60 * π / 180)
  (h3 : t.C = 75 * π / 180)
  : t.b = Real.sqrt 2 := by
  sorry

-- Note: We use radians for angles in Lean, so we convert degrees to radians

end triangle_side_b_l681_68186


namespace sugar_loss_calculation_l681_68108

/-- Given an initial amount of sugar, number of bags, and loss percentage,
    calculate the remaining amount of sugar. -/
def remaining_sugar (initial_sugar : ℝ) (num_bags : ℕ) (loss_percent : ℝ) : ℝ :=
  initial_sugar * (1 - loss_percent)

/-- Theorem: Given 24 kilos of sugar divided equally into 4 bags,
    with 15% loss in each bag, the total remaining sugar is 20.4 kilos. -/
theorem sugar_loss_calculation : remaining_sugar 24 4 0.15 = 20.4 := by
  sorry

#check sugar_loss_calculation

end sugar_loss_calculation_l681_68108


namespace max_terms_of_arithmetic_sequence_l681_68179

/-- An arithmetic sequence with common difference 4 and real-valued terms -/
def ArithmeticSequence (a₁ : ℝ) (n : ℕ) : ℕ → ℝ :=
  fun k => a₁ + (k - 1) * 4

/-- The sum of terms from the second to the nth term -/
def SumOfRemainingTerms (a₁ : ℝ) (n : ℕ) : ℝ :=
  (n - 1) * (a₁ + 2 * n)

/-- The condition that the square of the first term plus the sum of remaining terms does not exceed 100 -/
def SequenceCondition (a₁ : ℝ) (n : ℕ) : Prop :=
  a₁^2 + SumOfRemainingTerms a₁ n ≤ 100

theorem max_terms_of_arithmetic_sequence :
  ∀ a₁ : ℝ, ∀ n : ℕ, SequenceCondition a₁ n → n ≤ 8 :=
by sorry

end max_terms_of_arithmetic_sequence_l681_68179


namespace parabola_rotation_l681_68188

/-- A parabola is defined by its coefficients a, h, and k in the form y = a(x - h)² + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Rotate a parabola by 180° around the origin -/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := -p.h, k := -p.k }

theorem parabola_rotation :
  let p := Parabola.mk 2 1 2
  rotate180 p = Parabola.mk (-2) (-1) (-2) := by
  sorry

end parabola_rotation_l681_68188


namespace field_trip_adults_l681_68116

/-- Given a field trip scenario, prove the number of adults attending. -/
theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) : 
  van_capacity = 9 → num_students = 40 → num_vans = 6 → 
  (num_vans * van_capacity - num_students : ℕ) = 14 := by
  sorry

end field_trip_adults_l681_68116


namespace lobster_rolls_count_total_plates_sum_l681_68109

/-- The number of plates of lobster rolls served at a banquet -/
def lobster_rolls : ℕ := 55 - (14 + 16)

/-- The total number of plates served at the banquet -/
def total_plates : ℕ := 55

/-- The number of plates of spicy hot noodles served at the banquet -/
def spicy_hot_noodles : ℕ := 14

/-- The number of plates of seafood noodles served at the banquet -/
def seafood_noodles : ℕ := 16

/-- Theorem stating that the number of lobster roll plates is 25 -/
theorem lobster_rolls_count : lobster_rolls = 25 := by
  sorry

/-- Theorem stating that the total number of plates is the sum of all dishes -/
theorem total_plates_sum : 
  total_plates = lobster_rolls + spicy_hot_noodles + seafood_noodles := by
  sorry

end lobster_rolls_count_total_plates_sum_l681_68109


namespace four_person_apartments_count_l681_68120

/-- Represents the number of 4-person apartments in each building -/
def four_person_apartments : ℕ := sorry

/-- The number of identical buildings in the complex -/
def num_buildings : ℕ := 4

/-- The number of studio apartments in each building -/
def studio_apartments : ℕ := 10

/-- The number of 2-person apartments in each building -/
def two_person_apartments : ℕ := 20

/-- The occupancy rate of the apartment complex -/
def occupancy_rate : ℚ := 3/4

/-- The total number of people living in the apartment complex -/
def total_occupants : ℕ := 210

/-- Theorem stating that the number of 4-person apartments in each building is 5 -/
theorem four_person_apartments_count : four_person_apartments = 5 :=
  by sorry

end four_person_apartments_count_l681_68120


namespace book_pricing_loss_percentage_l681_68176

theorem book_pricing_loss_percentage 
  (cost_price selling_price : ℝ) 
  (h1 : cost_price > 0) 
  (h2 : 5 * cost_price = 20 * selling_price) : 
  (cost_price - selling_price) / cost_price = 3/4 := by
  sorry

end book_pricing_loss_percentage_l681_68176


namespace parallel_vectors_imply_x_half_l681_68182

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem parallel_vectors_imply_x_half :
  ∀ x : ℝ, (∃ k : ℝ, k ≠ 0 ∧ a + 2 • b x = k • (2 • a - 2 • b x)) → x = 1/2 := by
  sorry

end parallel_vectors_imply_x_half_l681_68182


namespace negation_square_nonnegative_l681_68175

theorem negation_square_nonnegative :
  ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by sorry

end negation_square_nonnegative_l681_68175


namespace ratio_theorem_l681_68110

theorem ratio_theorem (x y : ℝ) (h : x / y = 5 / 3) : y / (x - y) = 3 / 2 := by
  sorry

end ratio_theorem_l681_68110


namespace volume_right_triangular_prism_l681_68160

/-- The volume of a right triangular prism given its lateral face areas and lateral edge length -/
theorem volume_right_triangular_prism
  (M N P l : ℝ)
  (hM : M > 0)
  (hN : N > 0)
  (hP : P > 0)
  (hl : l > 0) :
  let V := (1 / (4 * l)) * Real.sqrt ((N + M + P) * (N + P - M) * (N + M - P) * (M + P - N))
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    M = a * l ∧
    N = b * l ∧
    P = c * l ∧
    V = (1 / 2) * l * Real.sqrt ((-a + b + c) * (a - b + c) * (a + b - c) * (a + b + c)) :=
by sorry

end volume_right_triangular_prism_l681_68160


namespace stairs_calculation_l681_68169

/-- The number of stairs run up and down one way during a football team's exercise routine. -/
def stairs_one_way : ℕ := 32

/-- The number of times players run up and down the bleachers. -/
def num_runs : ℕ := 40

/-- The number of calories burned per stair. -/
def calories_per_stair : ℕ := 2

/-- The total number of calories burned during the exercise. -/
def total_calories_burned : ℕ := 5120

/-- Theorem stating that the number of stairs run up and down one way is 32,
    given the conditions of the exercise routine. -/
theorem stairs_calculation :
  stairs_one_way = 32 ∧
  num_runs * (2 * stairs_one_way) * calories_per_stair = total_calories_burned :=
by sorry

end stairs_calculation_l681_68169


namespace min_distance_between_curves_l681_68189

noncomputable def f (t : ℝ) : ℝ := Real.exp t + 1

noncomputable def g (t : ℝ) : ℝ := 2 * t - 1

theorem min_distance_between_curves :
  ∃ (t_min : ℝ), ∀ (t : ℝ), |f t - g t| ≥ |f t_min - g t_min| ∧ 
  |f t_min - g t_min| = 4 - 2 * Real.log 2 := by
  sorry

end min_distance_between_curves_l681_68189


namespace inequality_proof_l681_68133

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := by
  sorry

end inequality_proof_l681_68133


namespace movie_shelf_problem_l681_68149

/-- The minimum number of additional movies needed to satisfy the conditions -/
def additional_movies_needed (current_movies : ℕ) (num_shelves : ℕ) : ℕ :=
  let target := num_shelves * (current_movies / num_shelves + 1)
  target - current_movies

theorem movie_shelf_problem :
  let current_movies := 9
  let num_shelves := 2
  let result := additional_movies_needed current_movies num_shelves
  (result = 1 ∧
   (current_movies + result) % 2 = 0 ∧
   (current_movies + result) / num_shelves % 2 = 1 ∧
   ∀ (shelf : ℕ), shelf < num_shelves →
     (current_movies + result) / num_shelves = (current_movies + result - shelf * ((current_movies + result) / num_shelves)) / (num_shelves - shelf)) :=
by sorry

end movie_shelf_problem_l681_68149


namespace partial_fraction_decomposition_l681_68170

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ),
    (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 →
      (D * x - 17) / (x^2 - 8*x + 15) = C / (x - 3) + 5 / (x - 5)) ∧
    C + D = 29/5 := by
  sorry

end partial_fraction_decomposition_l681_68170


namespace work_completion_time_l681_68113

/-- The time it takes for worker c to complete the work alone -/
def time_c : ℝ := 12

/-- The time it takes for worker a to complete the work alone -/
def time_a : ℝ := 16

/-- The time it takes for worker b to complete the work alone -/
def time_b : ℝ := 6

/-- The time it takes for workers a, b, and c to complete the work together -/
def time_abc : ℝ := 3.2

theorem work_completion_time :
  1 / time_a + 1 / time_b + 1 / time_c = 1 / time_abc :=
sorry

end work_completion_time_l681_68113


namespace hippo_ratio_l681_68153

/-- Represents the number of female hippos -/
def F : ℕ := sorry

/-- The initial number of elephants -/
def initial_elephants : ℕ := 20

/-- The initial number of hippos -/
def initial_hippos : ℕ := 35

/-- The number of baby hippos born per female hippo -/
def babies_per_hippo : ℕ := 5

/-- The total number of animals after births -/
def total_animals : ℕ := 315

theorem hippo_ratio :
  let newborn_hippos := F * babies_per_hippo
  let newborn_elephants := newborn_hippos + 10
  let total_hippos := initial_hippos + newborn_hippos
  (F : ℚ) / total_hippos = 5 / 32 :=
by sorry

end hippo_ratio_l681_68153


namespace smaug_copper_coins_l681_68177

/-- Represents the number of coins of each type in Smaug's hoard -/
structure DragonHoard where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Calculates the total value of a hoard in copper coins -/
def hoardValue (h : DragonHoard) (silverValue copperValue : ℕ) : ℕ :=
  h.gold * silverValue * copperValue + h.silver * copperValue + h.copper

/-- Theorem stating that Smaug has 33 copper coins -/
theorem smaug_copper_coins :
  ∃ (h : DragonHoard),
    h.gold = 100 ∧
    h.silver = 60 ∧
    hoardValue h 3 8 = 2913 ∧
    h.copper = 33 := by
  sorry

end smaug_copper_coins_l681_68177


namespace carols_birthday_l681_68136

/-- Represents a date with a month and a day -/
structure Date where
  month : String
  day : Nat

/-- The list of possible dates for Carol's birthday -/
def possible_dates : List Date := [
  ⟨"January", 4⟩, ⟨"March", 8⟩, ⟨"June", 7⟩, ⟨"October", 7⟩,
  ⟨"January", 5⟩, ⟨"April", 8⟩, ⟨"June", 5⟩, ⟨"October", 4⟩,
  ⟨"January", 11⟩, ⟨"April", 9⟩, ⟨"July", 13⟩, ⟨"October", 8⟩
]

/-- Alberto knows the month but not the exact date -/
def alberto_knows_month (d : Date) : Prop :=
  ∃ (other : Date), other ∈ possible_dates ∧ other.month = d.month ∧ other ≠ d

/-- Bernardo knows the day but not the exact date -/
def bernardo_knows_day (d : Date) : Prop :=
  ∃ (other : Date), other ∈ possible_dates ∧ other.day = d.day ∧ other ≠ d

/-- Alberto's first statement: He can't determine the date, and he's sure Bernardo can't either -/
def alberto_statement1 (d : Date) : Prop :=
  alberto_knows_month d ∧ bernardo_knows_day d

/-- After Alberto's statement, Bernardo can determine the date -/
def bernardo_statement (d : Date) : Prop :=
  alberto_statement1 d ∧
  ∀ (other : Date), other ∈ possible_dates → other.day = d.day → alberto_statement1 other → other = d

/-- After Bernardo's statement, Alberto can also determine the date -/
def alberto_statement2 (d : Date) : Prop :=
  bernardo_statement d ∧
  ∀ (other : Date), other ∈ possible_dates → other.month = d.month → bernardo_statement other → other = d

/-- The theorem stating that Carol's birthday must be June 7 -/
theorem carols_birthday :
  ∃! (d : Date), d ∈ possible_dates ∧ alberto_statement2 d ∧ d = ⟨"June", 7⟩ := by
  sorry

end carols_birthday_l681_68136


namespace smallest_integer_for_prime_quadratic_l681_68123

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def abs_value (n : ℤ) : ℕ := Int.natAbs n

def quadratic_expression (x : ℤ) : ℤ := 8 * x^2 - 53 * x + 21

theorem smallest_integer_for_prime_quadratic :
  ∀ x : ℤ, x < 8 → ¬(is_prime (abs_value (quadratic_expression x))) ∧
  is_prime (abs_value (quadratic_expression 8)) :=
by sorry

end smallest_integer_for_prime_quadratic_l681_68123


namespace vector_sum_equals_expected_l681_68121

-- Define the vectors
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![-3, 4]

-- Define the sum of the vectors
def sum_ab : Fin 2 → ℝ := ![a 0 + b 0, a 1 + b 1]

-- Theorem statement
theorem vector_sum_equals_expected : sum_ab = ![-1, 5] := by
  sorry

end vector_sum_equals_expected_l681_68121


namespace f_one_when_m_three_max_value_when_even_max_value_attained_when_even_l681_68158

-- Define the function f(x) with parameter m
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x + 2

-- Theorem 1: When m = 3, f(1) = 4
theorem f_one_when_m_three : f 3 1 = 4 := by sorry

-- Define what it means for f to be an even function
def is_even_function (m : ℝ) : Prop := ∀ x, f m (-x) = f m x

-- Theorem 2: If f is an even function, its maximum value is 2
theorem max_value_when_even :
  ∀ m, is_even_function m → ∀ x, f m x ≤ 2 := by sorry

-- Theorem 3: The maximum value 2 is attained when f is an even function
theorem max_value_attained_when_even :
  ∃ m, is_even_function m ∧ ∃ x, f m x = 2 := by sorry

end f_one_when_m_three_max_value_when_even_max_value_attained_when_even_l681_68158


namespace difference_c_minus_a_l681_68165

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45) 
  (h2 : (b + c) / 2 = 50) : 
  c - a = 10 := by
sorry

end difference_c_minus_a_l681_68165


namespace problem_statement_l681_68117

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) :=
sorry

end problem_statement_l681_68117
