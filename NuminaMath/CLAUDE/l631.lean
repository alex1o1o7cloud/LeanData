import Mathlib

namespace icosahedron_edge_ratio_l631_63175

/-- An icosahedron with edge length a -/
structure Icosahedron where
  a : ℝ
  a_pos : 0 < a

/-- A regular octahedron -/
structure RegularOctahedron where
  edge_length : ℝ
  edge_length_pos : 0 < edge_length

/-- Given two icosahedrons, this function returns true if six vertices 
    can be chosen from them to form a regular octahedron -/
def can_form_octahedron (i1 i2 : Icosahedron) : Prop := sorry

theorem icosahedron_edge_ratio 
  (i1 i2 : Icosahedron) 
  (h : can_form_octahedron i1 i2) : 
  i1.a / i2.a = (Real.sqrt 5 + 1) / 2 := by sorry

end icosahedron_edge_ratio_l631_63175


namespace sum_of_two_squares_l631_63152

theorem sum_of_two_squares (n m : ℕ) (h : 2 * m = n^2 + 1) :
  ∃ k : ℕ, m = k^2 + (k - 1)^2 := by sorry

end sum_of_two_squares_l631_63152


namespace coeff_x4_when_sum_64_l631_63177

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the sum of binomial coefficients
def sum_binomial_coeff (n : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x^4 in the expansion
def coeff_x4 (n : ℕ) : ℤ := sorry

-- Theorem statement
theorem coeff_x4_when_sum_64 (n : ℕ) :
  sum_binomial_coeff n = 64 → coeff_x4 n = -12 := by sorry

end coeff_x4_when_sum_64_l631_63177


namespace soft_drink_added_sugar_percentage_l631_63173

theorem soft_drink_added_sugar_percentage (
  soft_drink_calories : ℕ)
  (candy_bar_sugar_calories : ℕ)
  (candy_bars_taken : ℕ)
  (recommended_sugar_intake : ℕ)
  (exceeded_percentage : ℕ)
  (h1 : soft_drink_calories = 2500)
  (h2 : candy_bar_sugar_calories = 25)
  (h3 : candy_bars_taken = 7)
  (h4 : recommended_sugar_intake = 150)
  (h5 : exceeded_percentage = 100) :
  (((recommended_sugar_intake * (100 + exceeded_percentage) / 100) -
    (candy_bar_sugar_calories * candy_bars_taken)) * 100) /
    soft_drink_calories = 5 := by
  sorry

end soft_drink_added_sugar_percentage_l631_63173


namespace familyReunionHandshakesCount_l631_63116

/-- Represents the number of handshakes at a family reunion --/
def familyReunionHandshakes : ℕ :=
  let quadrupletSets : ℕ := 12
  let quintupletSets : ℕ := 4
  let quadrupletsPerSet : ℕ := 4
  let quintupletsPerSet : ℕ := 5
  let totalQuadruplets : ℕ := quadrupletSets * quadrupletsPerSet
  let totalQuintuplets : ℕ := quintupletSets * quintupletsPerSet
  let quadrupletHandshakes : ℕ := totalQuadruplets * (totalQuadruplets - quadrupletsPerSet)
  let quintupletHandshakes : ℕ := totalQuintuplets * (totalQuintuplets - quintupletsPerSet)
  let crossHandshakes : ℕ := totalQuadruplets * 7 + totalQuintuplets * 12
  (quadrupletHandshakes + quintupletHandshakes + crossHandshakes) / 2

/-- Theorem stating that the number of handshakes at the family reunion is 1494 --/
theorem familyReunionHandshakesCount : familyReunionHandshakes = 1494 := by
  sorry

end familyReunionHandshakesCount_l631_63116


namespace sum_of_n_factorial_times_n_l631_63192

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_of_n_factorial_times_n (a : ℕ) (h : factorial 1580 = a) :
  (Finset.range 1581).sum (λ n => n * factorial n) = 1581 * a - 1 := by
  sorry

end sum_of_n_factorial_times_n_l631_63192


namespace total_wicks_count_l631_63121

/-- The length of the spool in feet -/
def spool_length : ℕ := 15

/-- The length of short wicks in inches -/
def short_wick : ℕ := 6

/-- The length of long wicks in inches -/
def long_wick : ℕ := 12

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

theorem total_wicks_count : ∃ (n : ℕ), 
  n * short_wick + n * long_wick = spool_length * feet_to_inches ∧
  n + n = 20 := by sorry

end total_wicks_count_l631_63121


namespace equation_solution_l631_63138

theorem equation_solution :
  ∃! x : ℝ, (3 : ℝ) ^ (2 * x + 2) = (1 : ℝ) / 81 :=
by
  use -3
  sorry

end equation_solution_l631_63138


namespace quartic_root_inequality_l631_63190

theorem quartic_root_inequality (a b : ℝ) :
  (∃ x : ℝ, x^4 - a*x^3 + 2*x^2 - b*x + 1 = 0) → a^2 + b^2 ≥ 8 := by
  sorry

end quartic_root_inequality_l631_63190


namespace isosceles_triangle_rectangle_equal_area_l631_63186

/-- Given an isosceles triangle and a rectangle with the same area, 
    prove that the height of the triangle is twice the breadth of the rectangle. -/
theorem isosceles_triangle_rectangle_equal_area 
  (l b h : ℝ) (hl : l > 0) (hb : b > 0) (hlb : l > b) : 
  (1 / 2 : ℝ) * l * h = l * b → h = 2 * b := by
  sorry

end isosceles_triangle_rectangle_equal_area_l631_63186


namespace three_custom_op_three_equals_six_l631_63134

-- Define the custom operation
def customOp (m n : ℕ) : ℕ := n ^ 2 - m

-- State the theorem
theorem three_custom_op_three_equals_six :
  customOp 3 3 = 6 := by sorry

end three_custom_op_three_equals_six_l631_63134


namespace shortest_distance_principle_applies_l631_63172

-- Define the phenomena
inductive Phenomenon
  | woodenBarFixing
  | treePlanting
  | electricWireLaying
  | roadStraightening

-- Define the principle
def shortestDistancePrinciple (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.electricWireLaying => true
  | Phenomenon.roadStraightening => true
  | _ => false

-- Theorem statement
theorem shortest_distance_principle_applies :
  (∀ p : Phenomenon, shortestDistancePrinciple p ↔ 
    (p = Phenomenon.electricWireLaying ∨ p = Phenomenon.roadStraightening)) := by
  sorry

end shortest_distance_principle_applies_l631_63172


namespace intersection_of_M_and_N_l631_63164

def M : Set ℤ := {0, 1, 2, 3}
def N : Set ℤ := {-1, 1}

theorem intersection_of_M_and_N : M ∩ N = {1} := by sorry

end intersection_of_M_and_N_l631_63164


namespace sunflower_height_difference_l631_63107

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Represents a height in feet and inches -/
structure Height :=
  (feet : ℕ)
  (inches : ℕ)

/-- Converts a Height to total inches -/
def height_to_inches (h : Height) : ℕ := feet_to_inches h.feet + h.inches

theorem sunflower_height_difference :
  let sister_height : Height := ⟨4, 3⟩
  let sunflower_height : ℕ := feet_to_inches 6
  sunflower_height - height_to_inches sister_height = 21 := by
  sorry

end sunflower_height_difference_l631_63107


namespace rectangular_plot_roots_l631_63102

theorem rectangular_plot_roots (length width r s : ℝ) : 
  length^2 - 3*length + 2 = 0 →
  width^2 - 3*width + 2 = 0 →
  (1/length)^2 - r*(1/length) + s = 0 →
  (1/width)^2 - r*(1/width) + s = 0 →
  r*s = 0.75 := by
sorry

end rectangular_plot_roots_l631_63102


namespace right_triangle_sets_l631_63100

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  ¬(is_right_triangle 3 4 6) ∧
  (is_right_triangle 7 24 25) ∧
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle 9 12 15) := by
  sorry

end right_triangle_sets_l631_63100


namespace circle_to_rectangle_length_l631_63193

/-- Given a circle with radius R, when divided into equal parts and rearranged to form
    an approximate rectangle with perimeter 20.7 cm, the length of this rectangle is π * R. -/
theorem circle_to_rectangle_length (R : ℝ) (h : (2 * R + 2 * π * R / 2) = 20.7) :
  π * R = (20.7 : ℝ) / 2 - R := by
  sorry

end circle_to_rectangle_length_l631_63193


namespace steel_rod_length_l631_63150

/-- Represents the properties of a uniform steel rod -/
structure SteelRod where
  weight_per_meter : ℝ
  length : ℝ
  weight : ℝ

/-- Theorem: Given a uniform steel rod where 9 m weighs 34.2 kg, 
    the length of the rod that weighs 42.75 kg is 11.25 m -/
theorem steel_rod_length 
  (rod : SteelRod) 
  (h1 : rod.weight_per_meter = 34.2 / 9) 
  (h2 : rod.weight = 42.75) : 
  rod.length = 11.25 := by
  sorry

#check steel_rod_length

end steel_rod_length_l631_63150


namespace weaving_problem_l631_63131

/-- Represents the daily weaving length in an arithmetic sequence -/
def weaving_sequence (initial_length : ℚ) (daily_increase : ℚ) (day : ℕ) : ℚ :=
  initial_length + (day - 1) * daily_increase

/-- Represents the total weaving length over a period of days -/
def total_weaving (initial_length : ℚ) (daily_increase : ℚ) (days : ℕ) : ℚ :=
  (days : ℚ) * initial_length + (days * (days - 1) / 2) * daily_increase

theorem weaving_problem (initial_length daily_increase : ℚ) :
  initial_length = 5 →
  total_weaving initial_length daily_increase 30 = 390 →
  weaving_sequence initial_length daily_increase 5 = 209 / 29 := by
  sorry

end weaving_problem_l631_63131


namespace negative_half_greater_than_negative_two_thirds_l631_63130

theorem negative_half_greater_than_negative_two_thirds :
  -0.5 > -(2/3) := by
  sorry

end negative_half_greater_than_negative_two_thirds_l631_63130


namespace smallest_undefined_value_l631_63170

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y < 1/9, ∃ z, (y - 2) / (9*y^2 - 98*y + 21) = z) ∧ 
  ¬∃ z, ((1/9 : ℝ) - 2) / (9*(1/9)^2 - 98*(1/9) + 21) = z :=
sorry

end smallest_undefined_value_l631_63170


namespace min_weighings_for_ten_coins_l631_63101

/-- Represents a weighing on a balance scale -/
inductive Weighing
  | Equal : Weighing
  | LeftLighter : Weighing
  | RightLighter : Weighing

/-- Represents the state of knowledge about the coins -/
structure CoinState where
  total : Nat
  genuine : Nat
  counterfeit : Nat

/-- A function that performs a weighing and updates the coin state -/
def performWeighing (state : CoinState) (w : Weighing) : CoinState :=
  sorry

/-- The minimum number of weighings required to find the counterfeit coin -/
def minWeighings (state : CoinState) : Nat :=
  sorry

/-- Theorem stating that the minimum number of weighings for 10 coins with 1 counterfeit is 3 -/
theorem min_weighings_for_ten_coins :
  let initialState : CoinState := ⟨10, 9, 1⟩
  minWeighings initialState = 3 := by
  sorry

end min_weighings_for_ten_coins_l631_63101


namespace anna_mean_score_l631_63140

def scores : List ℝ := [88, 90, 92, 95, 96, 98, 100, 102, 105]

def timothy_count : ℕ := 5
def anna_count : ℕ := 4
def timothy_mean : ℝ := 95

theorem anna_mean_score (h1 : scores.length = timothy_count + anna_count)
                        (h2 : timothy_count * timothy_mean = scores.sum - anna_count * anna_mean) :
  anna_mean = 97.75 := by
  sorry

end anna_mean_score_l631_63140


namespace problem_solution_l631_63103

def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}

def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

theorem problem_solution :
  (∀ x, x ∈ (A ∩ B (-4)) ↔ (1/2 ≤ x ∧ x < 2)) ∧
  (∀ x, x ∈ (A ∪ B (-4)) ↔ (-2 < x ∧ x ≤ 3)) ∧
  (∀ a, (Aᶜ ∩ B a = B a) ↔ a ≥ -2) :=
by sorry

end problem_solution_l631_63103


namespace min_value_a_l631_63155

theorem min_value_a (a : ℝ) : 
  (a > 0 ∧ ∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) → a ≥ 4 :=
by sorry

end min_value_a_l631_63155


namespace f_min_value_f_at_3_l631_63124

-- Define the function f
def f (x : ℝ) : ℝ := 7 * x^2 - 28 * x + 2003

-- Theorem for the minimum value of f
theorem f_min_value : ∃ (min : ℝ), min = 1975 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

-- Theorem for the value of f(3)
theorem f_at_3 : f 3 = 1982 :=
sorry

end f_min_value_f_at_3_l631_63124


namespace arithmetic_sequence_sum_l631_63139

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2
  arith_def : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Given conditions on the arithmetic sequence imply S₁₀ = 65 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h1 : seq.a 3 = 4)
    (h2 : seq.S 9 - seq.S 6 = 27) :
  seq.S 10 = 65 := by
  sorry

end arithmetic_sequence_sum_l631_63139


namespace target_hit_probability_l631_63114

theorem target_hit_probability (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) (h_B : p_B = 1/3) (h_C : p_C = 1/4) :
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end target_hit_probability_l631_63114


namespace valid_arrangement_probability_l631_63199

-- Define the number of teachers and days
def num_teachers : ℕ := 6
def num_days : ℕ := 3
def teachers_per_day : ℕ := 2

-- Define the teachers who have restrictions
structure RestrictedTeacher where
  name : String
  restricted_day : ℕ

-- Define the specific restrictions
def wang : RestrictedTeacher := ⟨"Wang", 2⟩
def li : RestrictedTeacher := ⟨"Li", 3⟩

-- Define the probability function
def probability_of_valid_arrangement (t : ℕ) (d : ℕ) (tpd : ℕ) 
  (r1 r2 : RestrictedTeacher) : ℚ :=
  7/15

-- State the theorem
theorem valid_arrangement_probability :
  probability_of_valid_arrangement num_teachers num_days teachers_per_day wang li = 7/15 := by
  sorry

end valid_arrangement_probability_l631_63199


namespace total_lives_calculation_video_game_lives_proof_l631_63196

theorem total_lives_calculation (initial_players : Nat) (additional_players : Nat) (lives_per_player : Nat) : Nat :=
  by
  -- Define the total number of players
  let total_players := initial_players + additional_players
  
  -- Calculate the total number of lives
  let total_lives := total_players * lives_per_player
  
  -- Prove that the total number of lives is 24
  have h : total_lives = 24 := by
    -- Replace with actual proof
    sorry
  
  -- Return the result
  exact total_lives

-- Define the specific values from the problem
def initial_friends : Nat := 2
def new_players : Nat := 2
def lives_per_player : Nat := 6

-- Theorem to prove the specific case
theorem video_game_lives_proof : 
  total_lives_calculation initial_friends new_players lives_per_player = 24 :=
by
  -- Replace with actual proof
  sorry

end total_lives_calculation_video_game_lives_proof_l631_63196


namespace trapezoid_triangle_area_l631_63166

/-- A trapezoid with vertices A, B, C, and D -/
structure Trapezoid :=
  (A B C D : ℝ × ℝ)

/-- The area of a trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- The length of a line segment between two points -/
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem trapezoid_triangle_area (t : Trapezoid) :
  area t = 30 ∧ length t.C t.D = 3 * length t.A t.B →
  triangleArea t.A t.B t.C = 7.5 := by sorry

end trapezoid_triangle_area_l631_63166


namespace quadratic_completing_square_l631_63169

theorem quadratic_completing_square (x : ℝ) : 
  4 * x^2 - 8 * x - 320 = 0 → ∃ s : ℝ, (x - 1)^2 = s ∧ s = 81 := by
sorry

end quadratic_completing_square_l631_63169


namespace rectangle_side_lengths_l631_63108

theorem rectangle_side_lengths (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) 
  (h4 : a * b = 2 * (a + b)) : a < 4 ∧ b > 4 := by
  sorry

end rectangle_side_lengths_l631_63108


namespace xy_reciprocal_and_ratio_l631_63128

theorem xy_reciprocal_and_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y = 1) (h4 : x / y = 36) : y = 1/6 := by
  sorry

end xy_reciprocal_and_ratio_l631_63128


namespace runs_scored_for_new_average_l631_63198

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  matches_played : ℕ
  total_runs : ℕ

/-- Calculate the batting average of a player -/
def batting_average (player : CricketPlayer) : ℚ :=
  player.total_runs / player.matches_played

/-- Calculate the total runs after a new match -/
def total_runs_after_match (player : CricketPlayer) (new_runs : ℕ) : ℕ :=
  player.total_runs + new_runs

/-- Calculate the new batting average after a match -/
def new_batting_average (player : CricketPlayer) (new_runs : ℕ) : ℚ :=
  (total_runs_after_match player new_runs) / (player.matches_played + 1)

theorem runs_scored_for_new_average 
  (player : CricketPlayer) 
  (new_runs : ℕ) :
  player.matches_played = 5 ∧ 
  batting_average player = 51 ∧
  new_batting_average player new_runs = 54 →
  new_runs = 69 := by
sorry

end runs_scored_for_new_average_l631_63198


namespace westpark_teachers_l631_63191

/-- The number of students at Westpark High School -/
def total_students : ℕ := 900

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- The number of students in each class -/
def students_per_class : ℕ := 25

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Calculate the number of teachers required at Westpark High School -/
def calculate_teachers : ℕ := 
  (total_students * classes_per_student / students_per_class + 
   (if (total_students * classes_per_student) % students_per_class = 0 then 0 else 1)) / 
  classes_per_teacher +
  (if ((total_students * classes_per_student / students_per_class + 
       (if (total_students * classes_per_student) % students_per_class = 0 then 0 else 1)) % 
      classes_per_teacher = 0) 
   then 0 
   else 1)

/-- Theorem stating that the number of teachers at Westpark High School is 44 -/
theorem westpark_teachers : calculate_teachers = 44 := by
  sorry

end westpark_teachers_l631_63191


namespace additional_curtain_material_l631_63183

-- Define the room height in feet
def room_height_feet : ℕ := 8

-- Define the desired curtain length in inches
def desired_curtain_length : ℕ := 101

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Theorem to prove the additional material needed
theorem additional_curtain_material :
  desired_curtain_length - (room_height_feet * feet_to_inches) = 5 := by
  sorry

end additional_curtain_material_l631_63183


namespace largest_intersection_x_coordinate_l631_63163

/-- The polynomial function -/
def P (d : ℝ) (x : ℝ) : ℝ := x^6 - 5*x^5 + 5*x^4 + 5*x^3 + d*x^2

/-- The parabola function -/
def Q (e f g : ℝ) (x : ℝ) : ℝ := e*x^2 + f*x + g

/-- The difference between the polynomial and the parabola -/
def R (d e f g : ℝ) (x : ℝ) : ℝ := P d x - Q e f g x

theorem largest_intersection_x_coordinate
  (d e f g : ℝ)
  (h1 : ∃ a b c : ℝ, ∀ x : ℝ, R d e f g x = (x - a)^2 * (x - b)^2 * (x - c)^2)
  (h2 : ∃! a b c : ℝ, ∀ x : ℝ, R d e f g x = (x - a)^2 * (x - b)^2 * (x - c)^2) :
  ∃ x : ℝ, (∀ y : ℝ, R d e f g y = 0 → y ≤ x) ∧ R d e f g x = 0 ∧ x = 3 :=
sorry

end largest_intersection_x_coordinate_l631_63163


namespace withdraw_representation_l631_63149

-- Define a type for monetary transactions
inductive Transaction
  | deposit (amount : ℤ)
  | withdraw (amount : ℤ)

-- Define a function to represent transactions
def represent : Transaction → ℤ
  | Transaction.deposit amount => amount
  | Transaction.withdraw amount => -amount

-- State the theorem
theorem withdraw_representation :
  represent (Transaction.deposit 30000) = 30000 →
  represent (Transaction.withdraw 40000) = -40000 := by
  sorry

end withdraw_representation_l631_63149


namespace star_seven_three_l631_63105

-- Define the ⋆ operation
def star (a b : ℤ) : ℤ := 4*a + 3*b - 2*a*b

-- Theorem statement
theorem star_seven_three : star 7 3 = -5 := by
  sorry

end star_seven_three_l631_63105


namespace min_sum_squares_l631_63119

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             (m = 40/7) :=
sorry

end min_sum_squares_l631_63119


namespace percentage_change_condition_l631_63106

theorem percentage_change_condition (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hqlt : q < 100) (hM : M > 0) : 
  (M * (1 + p / 100) * (1 - q / 100) > M) ↔ (p > 100 * q / (100 - q)) := by
  sorry

end percentage_change_condition_l631_63106


namespace gcd_lcm_theorem_l631_63189

theorem gcd_lcm_theorem : 
  (Nat.gcd 42 63 = 21 ∧ Nat.lcm 42 63 = 126) ∧ 
  (Nat.gcd 8 20 = 4 ∧ Nat.lcm 8 20 = 40) := by
  sorry

end gcd_lcm_theorem_l631_63189


namespace only_origin_satisfies_l631_63141

def point_satisfies_inequality (x y : ℝ) : Prop :=
  x + y - 1 < 0

theorem only_origin_satisfies :
  point_satisfies_inequality 0 0 ∧
  ¬point_satisfies_inequality 2 4 ∧
  ¬point_satisfies_inequality (-1) 4 ∧
  ¬point_satisfies_inequality 1 8 :=
by sorry

end only_origin_satisfies_l631_63141


namespace largest_number_value_l631_63176

theorem largest_number_value
  (a b c : ℝ)
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 100)
  (h_larger_diff : c - b = 10)
  (h_smaller_diff : b - a = 5) :
  c = 125 / 3 := by
sorry

end largest_number_value_l631_63176


namespace expand_expression_l631_63143

theorem expand_expression (x : ℝ) : -2 * (5 * x^3 - 7 * x^2 + x - 4) = -10 * x^3 + 14 * x^2 - 2 * x + 8 := by
  sorry

end expand_expression_l631_63143


namespace second_white_given_first_white_l631_63162

/-- Represents the number of white balls initially in the bag -/
def white_balls : ℕ := 5

/-- Represents the number of red balls initially in the bag -/
def red_balls : ℕ := 3

/-- Represents the total number of balls initially in the bag -/
def total_balls : ℕ := white_balls + red_balls

/-- Represents the probability of drawing a white ball on the second draw
    given that the first draw was white -/
def prob_second_white_given_first_white : ℚ := 4 / 7

/-- Theorem stating that the probability of drawing a white ball on the second draw
    given that the first draw was white is 4/7 -/
theorem second_white_given_first_white :
  prob_second_white_given_first_white = 4 / 7 :=
by sorry

end second_white_given_first_white_l631_63162


namespace complex_division_result_abs_value_result_l631_63126

open Complex

def z₁ : ℂ := 1 - I
def z₂ : ℂ := 4 + 6 * I

theorem complex_division_result : z₂ / z₁ = -1 + 5 * I := by sorry

theorem abs_value_result (b : ℝ) (z : ℂ) (h : z = 1 + b * I) 
  (h_real : (z + z₁).im = 0) : abs z = Real.sqrt 2 := by sorry

end complex_division_result_abs_value_result_l631_63126


namespace largest_n_for_product_2010_l631_63197

def is_arithmetic_sequence (s : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, s (n + 1) - s n = d

theorem largest_n_for_product_2010 (a b : ℕ → ℤ) 
  (ha : is_arithmetic_sequence a) 
  (hb : is_arithmetic_sequence b)
  (h1 : a 1 = 1 ∧ b 1 = 1)
  (h2 : a 2 ≤ b 2)
  (h3 : ∃ n : ℕ, a n * b n = 2010)
  : (∃ n : ℕ, a n * b n = 2010 ∧ ∀ m : ℕ, a m * b m = 2010 → m ≤ n) ∧
    (∀ n : ℕ, a n * b n = 2010 → n ≤ 8) :=
sorry

end largest_n_for_product_2010_l631_63197


namespace specific_polygon_properties_l631_63180

/-- Represents a regular polygon with given properties -/
structure RegularPolygon where
  total_angle_sum : ℝ
  known_angle : ℝ
  num_sides : ℕ
  remaining_angle : ℝ

/-- Theorem about a specific regular polygon -/
theorem specific_polygon_properties :
  let p := RegularPolygon.mk 3420 160 21 163
  p.num_sides = 21 ∧
  p.remaining_angle = 163 ∧
  p.total_angle_sum = 180 * (p.num_sides - 2) ∧
  p.total_angle_sum = p.known_angle + (p.num_sides - 1) * p.remaining_angle :=
by sorry

end specific_polygon_properties_l631_63180


namespace loss_percentage_is_29_percent_l631_63137

-- Define the markup percentage
def markup : ℝ := 0.40

-- Define the discount percentage
def discount : ℝ := 0.07857142857142857

-- Define the loss percentage we want to prove
def target_loss_percentage : ℝ := 0.29

-- Theorem statement
theorem loss_percentage_is_29_percent (cost_price : ℝ) (cost_price_positive : cost_price > 0) :
  let marked_price := cost_price * (1 + markup)
  let selling_price := marked_price * (1 - discount)
  let loss := cost_price - selling_price
  let loss_percentage := loss / cost_price
  loss_percentage = target_loss_percentage :=
by sorry

end loss_percentage_is_29_percent_l631_63137


namespace square_geq_linear_l631_63142

theorem square_geq_linear (a b : ℝ) (ha : a > 0) : a^2 ≥ 2*b - a := by sorry

end square_geq_linear_l631_63142


namespace ellipse_foci_l631_63104

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := y^2 / 9 + x^2 / 4 = 1

-- Define the foci coordinates
def foci_coordinates : Set (ℝ × ℝ) := {(0, Real.sqrt 5), (0, -Real.sqrt 5)}

-- Theorem statement
theorem ellipse_foci : 
  ∀ (x y : ℝ), ellipse x y → (x, y) ∈ foci_coordinates ↔ 
  (x = 0 ∧ y = Real.sqrt 5) ∨ (x = 0 ∧ y = -Real.sqrt 5) :=
sorry

end ellipse_foci_l631_63104


namespace area_of_triangles_is_four_l631_63171

/-- A regular octagon with side length 2 cm -/
structure RegularOctagon where
  side_length : ℝ
  is_two_cm : side_length = 2

/-- The area of the four triangles formed when two rectangles are drawn
    connecting opposite vertices in a regular octagon -/
def area_of_four_triangles (octagon : RegularOctagon) : ℝ := 4

/-- Theorem stating that the area of the four triangles is 4 cm² -/
theorem area_of_triangles_is_four (octagon : RegularOctagon) :
  area_of_four_triangles octagon = 4 := by
  sorry

#check area_of_triangles_is_four

end area_of_triangles_is_four_l631_63171


namespace card_distribution_result_l631_63109

/-- Represents the card distribution problem --/
def card_distribution (jimmy_initial bob_initial sarah_initial : ℕ)
  (jimmy_to_bob jimmy_to_mary : ℕ)
  (sarah_friends : ℕ) : Prop :=
  let bob_after_jimmy := bob_initial + jimmy_to_bob
  let bob_to_sarah := bob_after_jimmy / 3
  let sarah_after_bob := sarah_initial + bob_to_sarah
  let sarah_to_friends := (sarah_after_bob / sarah_friends) * sarah_friends
  let jimmy_final := jimmy_initial - jimmy_to_bob - jimmy_to_mary
  let sarah_final := sarah_after_bob - sarah_to_friends
  let friends_cards := sarah_to_friends / sarah_friends
  jimmy_final = 50 ∧ sarah_final = 1 ∧ friends_cards = 3

/-- The main theorem stating the result of the card distribution --/
theorem card_distribution_result :
  card_distribution 68 5 7 6 12 3 :=
sorry

end card_distribution_result_l631_63109


namespace exists_equilateral_DEF_l631_63144

open Real

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Checks if a point is inside a circle -/
def isInside (p : Point) (c : Circle) : Prop := sorry

/-- Gets the circumcircle of a triangle -/
def circumcircle (t : Triangle) : Circle := sorry

/-- Gets the intersection points of a ray from a point through another point with a circle -/
def rayIntersection (start : Point) (through : Point) (c : Circle) : Point := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Main theorem -/
theorem exists_equilateral_DEF (ABC : Triangle) (c : Circle) :
  isAcuteAngled ABC →
  c = circumcircle ABC →
  ∃ P : Point,
    isInside P c ∧
    let D := rayIntersection A P c
    let E := rayIntersection B P c
    let F := rayIntersection C P c
    isEquilateral (Triangle.mk D E F) :=
by sorry

end exists_equilateral_DEF_l631_63144


namespace bus_calculation_l631_63181

theorem bus_calculation (total_students : ℕ) (capacity_40 capacity_30 : ℕ) : 
  total_students = 186 → capacity_40 = 40 → capacity_30 = 30 →
  (Nat.ceil (total_students / capacity_40) = 5 ∧
   Nat.ceil (total_students / capacity_30) = 7) := by
  sorry

#check bus_calculation

end bus_calculation_l631_63181


namespace set_equality_implies_p_equals_three_l631_63174

theorem set_equality_implies_p_equals_three (p : ℝ) : 
  let U : Set ℝ := {x | x^2 - 3*x + 2 = 0}
  let A : Set ℝ := {x | x^2 - p*x + 2 = 0}
  (U \ A = ∅) → p = 3 := by
sorry

end set_equality_implies_p_equals_three_l631_63174


namespace recurrence_relation_solution_l631_63125

def a (n : ℕ) : ℤ := -4 + 17 * n - 21 * n^2 + 5 * n^3 + n^4

theorem recurrence_relation_solution :
  (∀ n : ℕ, n ≥ 3 → a n = 3 * a (n - 1) - 3 * a (n - 2) + a (n - 3) + 24 * n - 6) ∧
  a 0 = -4 ∧
  a 1 = -2 ∧
  a 2 = 2 := by
  sorry

end recurrence_relation_solution_l631_63125


namespace division_problem_l631_63133

theorem division_problem (number : ℕ) : 
  (number / 25 = 5) ∧ (number % 25 = 2) → number = 127 := by
  sorry

end division_problem_l631_63133


namespace number_divided_by_14_5_equals_173_l631_63160

theorem number_divided_by_14_5_equals_173 (x : ℝ) : 
  x / 14.5 = 173 → x = 2508.5 := by
sorry

end number_divided_by_14_5_equals_173_l631_63160


namespace geometric_progression_common_ratio_l631_63147

theorem geometric_progression_common_ratio :
  ∀ (a : ℝ) (r : ℝ),
    a > 0 →  -- First term is positive
    r > 0 →  -- Common ratio is positive (to ensure all terms are positive)
    a = a * r + a * r^2 + a * r^3 →  -- First term equals sum of next three terms
    r = (Real.sqrt 5 - 1) / 3 :=
by sorry

end geometric_progression_common_ratio_l631_63147


namespace yellow_papers_in_ten_by_ten_square_l631_63146

/-- Represents a square arrangement of colored papers -/
structure ColoredSquare where
  size : Nat
  redPeriphery : Bool

/-- Calculates the number of yellow papers in a ColoredSquare -/
def yellowPapers (square : ColoredSquare) : Nat :=
  if square.redPeriphery then
    square.size * square.size - (4 * square.size - 4)
  else
    square.size * square.size

/-- Theorem stating that a 10x10 ColoredSquare with red periphery has 64 yellow papers -/
theorem yellow_papers_in_ten_by_ten_square :
  yellowPapers { size := 10, redPeriphery := true } = 64 := by
  sorry

#eval yellowPapers { size := 10, redPeriphery := true }

end yellow_papers_in_ten_by_ten_square_l631_63146


namespace product_plus_one_is_square_l631_63158

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) :
  x * y + 1 = (x + 1)^2 := by
  sorry

end product_plus_one_is_square_l631_63158


namespace equation_equivalence_l631_63110

theorem equation_equivalence (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) ↔ (y = 6 * x / (x - 9)) :=
by sorry

end equation_equivalence_l631_63110


namespace right_triangle_arctan_sum_l631_63184

/-- In a right-angled triangle ABC, prove that arctan(a/(b+c)) + arctan(c/(a+b)) = π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h : a^2 + c^2 = b^2) :
  Real.arctan (a / (b + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end right_triangle_arctan_sum_l631_63184


namespace perpendicular_line_value_l631_63156

theorem perpendicular_line_value (θ : Real) (h : Real.tan θ = -3) :
  2 / (3 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 10/13 := by
  sorry

end perpendicular_line_value_l631_63156


namespace max_value_theorem_l631_63151

theorem max_value_theorem (x y : ℝ) (h : 2 * x^2 - 6 * x + y^2 = 0) :
  ∃ (M : ℝ), M = 15 ∧ ∀ (a b : ℝ), 2 * a^2 - 6 * a + b^2 = 0 → a^2 + b^2 + 2 * a ≤ M :=
by sorry

end max_value_theorem_l631_63151


namespace inscribed_square_side_length_l631_63120

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  pq : ℝ
  qr : ℝ
  pr : ℝ
  is_right : pq^2 + qr^2 = pr^2
  pq_eq : pq = 5
  qr_eq : qr = 12
  pr_eq : pr = 13

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_hypotenuse : side_length ≤ t.pr
  on_legs : side_length ≤ t.pq ∧ side_length ≤ t.qr

/-- The side length of the inscribed square is 156/25 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 156 / 25 := by
  sorry

end inscribed_square_side_length_l631_63120


namespace baseball_weight_l631_63157

theorem baseball_weight (total_weight : ℝ) (soccer_ball_weight : ℝ) (baseball_count : ℕ) (soccer_ball_count : ℕ) :
  total_weight = 10.98 →
  soccer_ball_weight = 0.8 →
  baseball_count = 7 →
  soccer_ball_count = 9 →
  (soccer_ball_count * soccer_ball_weight + baseball_count * ((total_weight - soccer_ball_count * soccer_ball_weight) / baseball_count) = total_weight) ∧
  ((total_weight - soccer_ball_count * soccer_ball_weight) / baseball_count = 0.54) :=
by
  sorry

end baseball_weight_l631_63157


namespace baker_cakes_sold_l631_63168

theorem baker_cakes_sold (pastries_sold : ℕ) (pastry_cake_difference : ℕ) 
  (h1 : pastries_sold = 154)
  (h2 : pastry_cake_difference = 76) :
  pastries_sold - pastry_cake_difference = 78 := by
  sorry

end baker_cakes_sold_l631_63168


namespace perpendicular_line_implies_parallel_planes_l631_63129

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_line_implies_parallel_planes 
  (α β : Plane) (l : Line) : 
  (perpendicular l α ∧ perpendicular l β) → parallel α β := by
  sorry

end perpendicular_line_implies_parallel_planes_l631_63129


namespace simple_interest_fraction_l631_63113

/-- 
Given a principal sum P, proves that the simple interest calculated for 8 years 
at a rate of 2.5% per annum is equal to 1/5 of the principal sum.
-/
theorem simple_interest_fraction (P : ℝ) (P_pos : P > 0) : 
  (P * 2.5 * 8) / 100 = P * (1 / 5) := by
  sorry

#check simple_interest_fraction

end simple_interest_fraction_l631_63113


namespace tangent_line_and_monotonicity_l631_63135

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3

theorem tangent_line_and_monotonicity (a : ℝ) :
  (a = 1 → ∃ (m b : ℝ), m = 9 ∧ b = 8 ∧ 
    ∀ x y, y = f 1 x → (x = -1 → y = m*x + b)) ∧
  (a = 0 → ∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a < 0 → (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 2*a → f a x₁ > f a x₂) ∧
           (∀ x₁ x₂, 2*a < x₁ ∧ x₁ < x₂ ∧ x₂ < 0 → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
  (a > 0 → (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2*a → f a x₁ > f a x₂) ∧
           (∀ x₁ x₂, 2*a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry

end tangent_line_and_monotonicity_l631_63135


namespace M_intersect_N_l631_63161

def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}

def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem M_intersect_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by
  sorry

end M_intersect_N_l631_63161


namespace quadratic_root_theorem_l631_63136

-- Define the quadratic polynomial f(x) = ax² + bx + c
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for f(x) to have exactly one root
def has_one_root (a b c : ℝ) : Prop :=
  ∃! x, f a b c x = 0

-- Define g(x) = f(3x + 2) - 2f(2x - 1)
def g (a b c x : ℝ) : ℝ := f a b c (3*x + 2) - 2 * f a b c (2*x - 1)

-- Theorem statement
theorem quadratic_root_theorem (a b c : ℝ) :
  a ≠ 0 →
  has_one_root a b c →
  has_one_root 1 (20 - b) (2 + 4*b - b^2/4) →
  ∃ x, f a b c x = 0 ∧ x = -7 :=
sorry

end quadratic_root_theorem_l631_63136


namespace closure_M_intersect_N_l631_63148

-- Define the set M
def M : Set ℝ := {x | 2/x < 1}

-- Define the set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

-- State the theorem
theorem closure_M_intersect_N :
  (closure M) ∩ N = Set.Icc 0 2 := by sorry

end closure_M_intersect_N_l631_63148


namespace counterexample_acute_angles_sum_l631_63132

theorem counterexample_acute_angles_sum : 
  ∃ (A B : ℝ), 0 < A ∧ A < 90 ∧ 0 < B ∧ B < 90 ∧ A + B ≥ 90 := by
  sorry

end counterexample_acute_angles_sum_l631_63132


namespace schedule_theorem_l631_63159

-- Define the number of classes
def total_classes : ℕ := 6

-- Define the number of morning slots
def morning_slots : ℕ := 4

-- Define the number of afternoon slots
def afternoon_slots : ℕ := 2

-- Define the function to calculate the number of arrangements
def schedule_arrangements (n : ℕ) (m : ℕ) (a : ℕ) : ℕ :=
  (m.choose 1) * (a.choose 1) * (n - 2).factorial

-- Theorem statement
theorem schedule_theorem :
  schedule_arrangements total_classes morning_slots afternoon_slots = 192 :=
by sorry

end schedule_theorem_l631_63159


namespace tangent_triangle_area_l631_63115

/-- The area of the triangle formed by the tangent line to y = e^x at (2, e^2) and the coordinate axes -/
theorem tangent_triangle_area : 
  let f : ℝ → ℝ := fun x ↦ Real.exp x
  let tangent_point : ℝ × ℝ := (2, Real.exp 2)
  let slope : ℝ := Real.exp 2
  let x_intercept : ℝ := 1
  let y_intercept : ℝ := Real.exp 2
  let triangle_area : ℝ := (1/2) * Real.exp 2
  triangle_area = (1/2) * y_intercept * x_intercept :=
by sorry


end tangent_triangle_area_l631_63115


namespace basketball_card_price_l631_63145

/-- The price of each pack of basketball cards given Nina's shopping details -/
theorem basketball_card_price (toy_price shirt_price total_spent : ℚ)
  (num_toys num_shirts num_card_packs : ℕ)
  (h1 : toy_price = 10)
  (h2 : shirt_price = 6)
  (h3 : num_toys = 3)
  (h4 : num_shirts = 5)
  (h5 : num_card_packs = 2)
  (h6 : total_spent = 70)
  (h7 : total_spent = toy_price * num_toys + shirt_price * num_shirts + num_card_packs * card_price) :
  card_price = 5 :=
by
  sorry

#check basketball_card_price

end basketball_card_price_l631_63145


namespace baseball_bat_price_baseball_bat_price_is_10_l631_63122

/-- Calculates the selling price of a baseball bat given the total revenue and prices of other items -/
theorem baseball_bat_price (total_revenue : ℝ) (cards_price : ℝ) (glove_original_price : ℝ) (glove_discount : ℝ) (cleats_price : ℝ) (cleats_quantity : ℕ) : ℝ :=
  let glove_price := glove_original_price * (1 - glove_discount)
  let known_revenue := cards_price + glove_price + (cleats_price * cleats_quantity)
  total_revenue - known_revenue

/-- Proves that the baseball bat price is $10 given the specific conditions -/
theorem baseball_bat_price_is_10 :
  baseball_bat_price 79 25 30 0.2 10 2 = 10 := by
  sorry

end baseball_bat_price_baseball_bat_price_is_10_l631_63122


namespace geometric_sequence_a12_l631_63117

/-- A geometric sequence (aₙ) -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a12 (a : ℕ → ℝ) :
  geometric_sequence a → a 4 = 4 → a 8 = 8 → a 12 = 16 := by
  sorry

end geometric_sequence_a12_l631_63117


namespace square_difference_l631_63118

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x + 3) * (x - 3) = 9792 := by
  sorry

end square_difference_l631_63118


namespace least_months_to_triple_l631_63123

def initial_amount : ℝ := 1500
def monthly_interest_rate : ℝ := 0.06

def compound_factor (t : ℕ) : ℝ := (1 + monthly_interest_rate) ^ t

theorem least_months_to_triple :
  ∀ n : ℕ, n < 20 → compound_factor n ≤ 3 ∧
  compound_factor 20 > 3 :=
by sorry

end least_months_to_triple_l631_63123


namespace no_nonzero_solution_for_diophantine_equation_l631_63165

theorem no_nonzero_solution_for_diophantine_equation :
  ∀ (x y z : ℤ), 2 * x^4 + y^4 = 7 * z^4 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end no_nonzero_solution_for_diophantine_equation_l631_63165


namespace sin_80_gt_sqrt3_sin_10_l631_63182

theorem sin_80_gt_sqrt3_sin_10 : Real.sin (80 * π / 180) > Real.sqrt 3 * Real.sin (10 * π / 180) := by
  sorry

end sin_80_gt_sqrt3_sin_10_l631_63182


namespace two_medians_not_unique_l631_63178

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the concept of a median
def Median (t : Triangle) : ℝ → Prop := sorry

-- Define the concept of uniquely determining a triangle's shape
def UniquelyDeterminesShape (data : Set (Triangle → Prop)) : Prop := sorry

-- Define the five sets of data
def TwoSidesIncludedAngle : Set (Triangle → Prop) := sorry
def ThreeSides : Set (Triangle → Prop) := sorry
def TwoMedians : Set (Triangle → Prop) := sorry
def OneAltitudeAndBase : Set (Triangle → Prop) := sorry
def TwoAngles : Set (Triangle → Prop) := sorry

-- Theorem statement
theorem two_medians_not_unique :
  UniquelyDeterminesShape TwoSidesIncludedAngle ∧
  UniquelyDeterminesShape ThreeSides ∧
  ¬UniquelyDeterminesShape TwoMedians ∧
  UniquelyDeterminesShape OneAltitudeAndBase ∧
  UniquelyDeterminesShape TwoAngles :=
sorry

end two_medians_not_unique_l631_63178


namespace altitude_divides_triangle_iff_right_angle_or_isosceles_l631_63112

/-- Triangle ABC with altitude h_a from vertex A to side BC -/
structure Triangle :=
  (A B C : Point)
  (h_a : Point)

/-- The altitude h_a divides triangle ABC into two similar triangles -/
def divides_into_similar_triangles (t : Triangle) : Prop :=
  sorry

/-- Angle A is a right angle -/
def is_right_angle_at_A (t : Triangle) : Prop :=
  sorry

/-- Triangle ABC is isosceles with AB = AC -/
def is_isosceles (t : Triangle) : Prop :=
  sorry

/-- Theorem: The altitude h_a of triangle ABC divides it into two similar triangles
    if and only if either angle A is a right angle or AB = AC -/
theorem altitude_divides_triangle_iff_right_angle_or_isosceles (t : Triangle) :
  divides_into_similar_triangles t ↔ (is_right_angle_at_A t ∨ is_isosceles t) :=
sorry

end altitude_divides_triangle_iff_right_angle_or_isosceles_l631_63112


namespace miles_and_davis_amount_l631_63167

-- Define the conversion rate from tablespoons of kernels to cups of popcorn
def kernels_to_popcorn (tablespoons : ℚ) : ℚ := 2 * tablespoons

-- Define the amounts of popcorn wanted by Joanie, Mitchell, and Cliff
def joanie_amount : ℚ := 3
def mitchell_amount : ℚ := 4
def cliff_amount : ℚ := 3

-- Define the total amount of kernels needed
def total_kernels : ℚ := 8

-- Theorem to prove
theorem miles_and_davis_amount :
  kernels_to_popcorn total_kernels - (joanie_amount + mitchell_amount + cliff_amount) = 6 :=
by sorry

end miles_and_davis_amount_l631_63167


namespace final_price_calculation_l631_63188

/-- Calculates the final price of an item after applying discounts and tax -/
theorem final_price_calculation (original_price : ℝ) 
  (first_discount_rate : ℝ) (second_discount_rate : ℝ) (tax_rate : ℝ) : 
  original_price = 200 ∧ 
  first_discount_rate = 0.5 ∧ 
  second_discount_rate = 0.25 ∧ 
  tax_rate = 0.1 → 
  original_price * (1 - first_discount_rate) * (1 - second_discount_rate) * (1 + tax_rate) = 82.5 := by
  sorry

#check final_price_calculation

end final_price_calculation_l631_63188


namespace quadratic_sum_l631_63127

def quadratic_function (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℤ) : 
  (quadratic_function a b c 0 = 2) → 
  (∀ x, quadratic_function a b c x ≥ quadratic_function a b c 1) →
  (quadratic_function a b c 1 = -1) →
  a - b + c = 11 := by
  sorry

end quadratic_sum_l631_63127


namespace function_equals_identity_l631_63111

theorem function_equals_identity (f : ℝ → ℝ) :
  (Continuous f) →
  (f 0 = 0) →
  (f 1 = 1) →
  (∀ x ∈ (Set.Ioo 0 1), ∃ h : ℝ, 
    0 ≤ x - h ∧ x + h ≤ 1 ∧ 
    f x = (f (x - h) + f (x + h)) / 2) →
  (∀ x ∈ (Set.Icc 0 1), f x = x) := by
sorry

end function_equals_identity_l631_63111


namespace no_integer_tangent_length_l631_63179

/-- A circle with a point P outside it, from which a tangent and a secant are drawn -/
structure CircleWithExternalPoint where
  /-- The circumference of the circle -/
  circumference : ℝ
  /-- The length of one arc created by the secant -/
  m : ℕ
  /-- The length of the tangent from P to the circle -/
  t₁ : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (c : CircleWithExternalPoint) : Prop :=
  c.circumference = 15 * Real.pi ∧
  c.t₁ * c.t₁ = c.m * (c.circumference - c.m)

/-- The theorem stating that no integer values of t₁ satisfy the conditions -/
theorem no_integer_tangent_length :
  ¬∃ c : CircleWithExternalPoint, satisfiesConditions c :=
sorry

end no_integer_tangent_length_l631_63179


namespace largest_x_floor_fraction_l631_63185

theorem largest_x_floor_fraction : 
  (∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) / x = 7 / 8) ∧ 
  (∀ (y : ℝ), y > 0 → (⌊y⌋ : ℝ) / y = 7 / 8 → y ≤ 48 / 7) := by
  sorry

end largest_x_floor_fraction_l631_63185


namespace pencils_needed_theorem_l631_63154

/-- Calculates the number of pencils needed to be purchased given the initial distribution and shortage --/
def pencils_to_purchase (box_a_pencils box_b_pencils : ℕ) (box_a_classrooms box_b_classrooms : ℕ) (shortage : ℕ) : ℕ :=
  let total_classrooms := box_a_classrooms + box_b_classrooms
  let box_a_per_class := box_a_pencils / box_a_classrooms
  let box_b_per_class := box_b_pencils / box_b_classrooms
  let total_per_class := box_a_per_class + box_b_per_class
  let shortage_per_class := (shortage + total_classrooms - 1) / total_classrooms
  shortage_per_class * total_classrooms

theorem pencils_needed_theorem :
  pencils_to_purchase 480 735 6 9 85 = 90 := by
  sorry

end pencils_needed_theorem_l631_63154


namespace minimal_leasing_cost_l631_63194

/-- Represents the daily production and cost of equipment types -/
structure EquipmentType where
  productA : ℕ
  productB : ℕ
  cost : ℕ

/-- Represents the company's production requirements -/
structure Requirements where
  minProductA : ℕ
  minProductB : ℕ

/-- Calculates the total production and cost for a given number of days of each equipment type -/
def calculateProduction (typeA : EquipmentType) (typeB : EquipmentType) (daysA : ℕ) (daysB : ℕ) : ℕ × ℕ × ℕ :=
  (daysA * typeA.productA + daysB * typeB.productA,
   daysA * typeB.productB + daysB * typeB.productB,
   daysA * typeA.cost + daysB * typeB.cost)

/-- Checks if the production meets the requirements -/
def meetsRequirements (prod : ℕ × ℕ × ℕ) (req : Requirements) : Prop :=
  prod.1 ≥ req.minProductA ∧ prod.2.1 ≥ req.minProductB

/-- Theorem stating that the minimal leasing cost is 2000 yuan -/
theorem minimal_leasing_cost 
  (typeA : EquipmentType)
  (typeB : EquipmentType)
  (req : Requirements)
  (h1 : typeA.productA = 5)
  (h2 : typeA.productB = 10)
  (h3 : typeA.cost = 200)
  (h4 : typeB.productA = 6)
  (h5 : typeB.productB = 20)
  (h6 : typeB.cost = 300)
  (h7 : req.minProductA = 50)
  (h8 : req.minProductB = 140) :
  ∃ (daysA daysB : ℕ), 
    let prod := calculateProduction typeA typeB daysA daysB
    meetsRequirements prod req ∧ 
    prod.2.2 = 2000 ∧
    (∀ (x y : ℕ), 
      let otherProd := calculateProduction typeA typeB x y
      meetsRequirements otherProd req → otherProd.2.2 ≥ 2000) := by
  sorry


end minimal_leasing_cost_l631_63194


namespace cube_plus_minus_one_divisible_by_seven_l631_63195

theorem cube_plus_minus_one_divisible_by_seven (n : ℤ) (h : ¬ 7 ∣ n) :
  7 ∣ (n^3 - 1) ∨ 7 ∣ (n^3 + 1) := by
  sorry

end cube_plus_minus_one_divisible_by_seven_l631_63195


namespace second_grade_girls_l631_63153

theorem second_grade_girls (boys_second : ℕ) (total_students : ℕ) :
  boys_second = 20 →
  total_students = 93 →
  ∃ (girls_second : ℕ),
    girls_second = 11 ∧
    total_students = boys_second + girls_second + 2 * (boys_second + girls_second) :=
by sorry

end second_grade_girls_l631_63153


namespace inequality_condition_l631_63187

theorem inequality_condition (x y : ℝ) : 
  y - x < Real.sqrt (x^2 + 4*x*y) ↔ 
  ((y < x + Real.sqrt (x^2 + 4*x*y) ∨ y < x - Real.sqrt (x^2 + 4*x*y)) ∧ x*(x + 4*y) ≥ 0) :=
by sorry

end inequality_condition_l631_63187
