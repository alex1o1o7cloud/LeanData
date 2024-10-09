import Mathlib

namespace chemistry_marks_l2086_208623

theorem chemistry_marks (marks_english : ℕ) (marks_math : ℕ) (marks_physics : ℕ) 
                        (marks_biology : ℕ) (average_marks : ℚ) (marks_chemistry : ℕ) 
                        (h_english : marks_english = 70) 
                        (h_math : marks_math = 60) 
                        (h_physics : marks_physics = 78) 
                        (h_biology : marks_biology = 65) 
                        (h_average : average_marks = 66.6) 
                        (h_total: average_marks * 5 = marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) : 
  marks_chemistry = 60 :=
by sorry

end chemistry_marks_l2086_208623


namespace catch_up_time_l2086_208660

-- Define the speeds of Person A and Person B.
def speed_A : ℝ := 10 -- kilometers per hour
def speed_B : ℝ := 7  -- kilometers per hour

-- Define the initial distance between Person A and Person B.
def initial_distance : ℝ := 15 -- kilometers

-- Prove the time it takes for person A to catch up with person B is 5 hours.
theorem catch_up_time :
  initial_distance / (speed_A - speed_B) = 5 :=
by
  -- Proof can be added here
  sorry

end catch_up_time_l2086_208660


namespace consecutive_negative_product_sum_l2086_208624

theorem consecutive_negative_product_sum (n : ℤ) (h : n * (n + 1) = 2850) : n + (n + 1) = -107 :=
sorry

end consecutive_negative_product_sum_l2086_208624


namespace john_spent_at_candy_store_l2086_208643

-- Definition of the conditions
def allowance : ℚ := 1.50
def arcade_spent : ℚ := (3 / 5) * allowance
def remaining_after_arcade : ℚ := allowance - arcade_spent
def toy_store_spent : ℚ := (1 / 3) * remaining_after_arcade

-- Statement and Proof of the Problem
theorem john_spent_at_candy_store : (remaining_after_arcade - toy_store_spent) = 0.40 :=
by
  -- Proof is left as an exercise
  sorry

end john_spent_at_candy_store_l2086_208643


namespace arithmetic_sequence_sum_l2086_208645

theorem arithmetic_sequence_sum (x y z d : ℤ)
  (h₀ : d = 10 - 3)
  (h₁ : 10 = 3 + d)
  (h₂ : 17 = 10 + d)
  (h₃ : x = 17 + d)
  (h₄ : y = x + d)
  (h₅ : 31 = y + d)
  (h₆ : z = 31 + d) :
  x + y + z = 93 := by
sorry

end arithmetic_sequence_sum_l2086_208645


namespace sun_city_population_l2086_208671

theorem sun_city_population (W R S : ℕ) (h1 : W = 2000)
    (h2 : R = 3 * W - 500) (h3 : S = 2 * R + 1000) : S = 12000 :=
by
    -- Use the provided conditions (h1, h2, h3) to state the theorem
    sorry

end sun_city_population_l2086_208671


namespace simplify_fraction_l2086_208614

theorem simplify_fraction (a b : ℕ) (h₁ : a = 84) (h₂ : b = 144) :
  a / gcd a b = 7 ∧ b / gcd a b = 12 := 
by
  sorry

end simplify_fraction_l2086_208614


namespace find_abc_l2086_208687

noncomputable def f (a b c x : ℝ) := x^3 + a*x^2 + b*x + c
noncomputable def f' (a b x : ℝ) := 3*x^2 + 2*a*x + b

theorem find_abc (a b c : ℝ) :
  (f' a b -2 = 0) ∧
  (f' a b 1 = -3) ∧
  (f a b c 1 = 0) →
  a = 1 ∧ b = -8 ∧ c = 6 :=
sorry

end find_abc_l2086_208687


namespace prime_k_values_l2086_208663

theorem prime_k_values (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Nat.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
by
  sorry

end prime_k_values_l2086_208663


namespace simplify_expression_l2086_208689

theorem simplify_expression (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) :
  ( ((x+1)^2 * (x^2 - x + 1)^2 / (x^3 + 1)^2)^2 *
    ((x-1)^2 * (x^2 + x + 1)^2 / (x^3 - 1)^2)^2
  ) = 1 :=
by
  sorry

end simplify_expression_l2086_208689


namespace poodle_barked_24_times_l2086_208679

-- Defining the conditions and question in Lean
def poodle_barks (terrier_barks_per_hush times_hushed: ℕ) : ℕ :=
  2 * terrier_barks_per_hush * times_hushed

theorem poodle_barked_24_times (terrier_barks_per_hush times_hushed: ℕ) :
  terrier_barks_per_hush = 2 → times_hushed = 6 → poodle_barks terrier_barks_per_hush times_hushed = 24 :=
by
  intros
  sorry

end poodle_barked_24_times_l2086_208679


namespace loss_percentage_is_11_percent_l2086_208675

-- Definitions based on conditions
def costPrice : ℝ := 1500
def sellingPrice : ℝ := 1335

-- The statement to prove
theorem loss_percentage_is_11_percent :
  ((costPrice - sellingPrice) / costPrice) * 100 = 11 := by
  sorry

end loss_percentage_is_11_percent_l2086_208675


namespace min_value_y_l2086_208674

theorem min_value_y (x : ℝ) (h : x > 0) : ∃ y, y = x + 4 / x^2 ∧ (∀ z, z = x + 4 / x^2 → y ≤ z) := 
sorry

end min_value_y_l2086_208674


namespace each_friend_received_12_candies_l2086_208633

-- Define the number of friends and total candies given
def num_friends : ℕ := 35
def total_candies : ℕ := 420

-- Define the number of candies each friend received
def candies_per_friend : ℕ := total_candies / num_friends

theorem each_friend_received_12_candies :
  candies_per_friend = 12 :=
by
  -- Skip the proof
  sorry

end each_friend_received_12_candies_l2086_208633


namespace total_peaches_l2086_208605

-- Definitions based on the given conditions
def initial_peaches : ℕ := 13
def picked_peaches : ℕ := 55

-- The proof goal stating the total number of peaches now
theorem total_peaches : initial_peaches + picked_peaches = 68 := by
  sorry

end total_peaches_l2086_208605


namespace unique_common_root_m_value_l2086_208628

theorem unique_common_root_m_value (m : ℝ) (h : m > 5) :
  (∃ x : ℝ, x^2 - 5 * x + 6 = 0 ∧ x^2 + 2 * x - 2 * m + 1 = 0) →
  m = 8 :=
by
  sorry

end unique_common_root_m_value_l2086_208628


namespace Anne_carrying_four_cats_weight_l2086_208691

theorem Anne_carrying_four_cats_weight : 
  let w1 := 2
  let w2 := 1.5 * w1
  let m1 := 2 * w1
  let m2 := w1 + w2
  w1 + w2 + m1 + m2 = 14 :=
by
  sorry

end Anne_carrying_four_cats_weight_l2086_208691


namespace arithmetic_geometric_seq_l2086_208608

open Real

theorem arithmetic_geometric_seq (a d : ℝ) (h₀ : d ≠ 0) 
  (h₁ : (a + d) * (a + 5 * d) = (a + 2 * d) ^ 2) : 
  (a + 2 * d) / (a + d) = 3 :=
sorry

end arithmetic_geometric_seq_l2086_208608


namespace f_2013_eq_2_l2086_208659

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (-x) = -f x
axiom h2 : ∀ x : ℝ, f (x + 4) = f x + f 2
axiom h3 : f (-1) = -2

theorem f_2013_eq_2 : f 2013 = 2 := 
by 
  sorry

end f_2013_eq_2_l2086_208659


namespace domain_of_sqrt_function_l2086_208692

theorem domain_of_sqrt_function :
  {x : ℝ | (1 / (Real.log x / Real.log 2) - 2 ≥ 0) ∧ (x > 0) ∧ (x ≠ 1)} 
  = {x : ℝ | 1 < x ∧ x ≤ Real.sqrt 10} :=
sorry

end domain_of_sqrt_function_l2086_208692


namespace arithmetic_seq_sum_l2086_208629

theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d) (h_a5 : a 5 = 15) :
  a 3 + a 4 + a 6 + a 7 = 60 :=
sorry

end arithmetic_seq_sum_l2086_208629


namespace calculate_a_minus_b_l2086_208677

theorem calculate_a_minus_b : 
  ∀ (a b : ℚ), (y = a * x + b) 
  ∧ (y = 4 ↔ x = 3) 
  ∧ (y = 22 ↔ x = 10) 
  → (a - b = 6 + 2 / 7)
:= sorry

end calculate_a_minus_b_l2086_208677


namespace triangle_side_length_l2086_208673

theorem triangle_side_length (AB AC BC BX CX : ℕ)
  (h1 : AB = 86)
  (h2 : AC = 97)
  (h3 : BX + CX = BC)
  (h4 : AX = AB)
  (h5 : AX = 86)
  (h6 : AB * AB * CX + AC * AC * BX = BC * (BX * CX + AX * AX))
  : BC = 61 := 
sorry

end triangle_side_length_l2086_208673


namespace initially_calculated_avg_height_l2086_208634

theorem initially_calculated_avg_height
  (A : ℕ)
  (initially_calculated_total_height : ℕ := 35 * A)
  (wrong_height : ℕ := 166)
  (actual_height : ℕ := 106)
  (height_overestimation : ℕ := wrong_height - actual_height)
  (actual_avg_height : ℕ := 179)
  (correct_total_height : ℕ := 35 * actual_avg_height)
  (initially_calculate_total_height_is_more : initially_calculated_total_height = correct_total_height + height_overestimation) :
  A = 181 :=
by
  sorry

end initially_calculated_avg_height_l2086_208634


namespace right_triangles_with_leg_2012_l2086_208670

theorem right_triangles_with_leg_2012 :
  ∀ (a b c : ℕ), a = 2012 ∧ a ^ 2 + b ^ 2 = c ^ 2 → 
  (b = 253005 ∧ c = 253013) ∨ 
  (b = 506016 ∧ c = 506020) ∨ 
  (b = 1012035 ∧ c = 1012037) ∨ 
  (b = 1509 ∧ c = 2515) :=
by
  intros
  sorry

end right_triangles_with_leg_2012_l2086_208670


namespace problem_x2_plus_y2_l2086_208646

theorem problem_x2_plus_y2 (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 :=
sorry

end problem_x2_plus_y2_l2086_208646


namespace negation_of_p_l2086_208694

variable (x : ℝ)

def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem negation_of_p : (¬p) ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry

end negation_of_p_l2086_208694


namespace find_white_balls_l2086_208641

noncomputable def white_balls_in_bag (total_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) 
  (p_not_red_nor_purple : ℚ) : ℕ :=
total_balls - (red_balls + purple_balls) - (green_balls + yellow_balls)

theorem find_white_balls :
  let total_balls := 60
  let green_balls := 18
  let yellow_balls := 17
  let red_balls := 3
  let purple_balls := 1
  let p_not_red_nor_purple := 0.95
  white_balls_in_bag total_balls green_balls yellow_balls red_balls purple_balls p_not_red_nor_purple = 21 :=
by
  let total_balls := 60
  let green_balls := 18
  let yellow_balls := 17
  let red_balls := 3
  let purple_balls := 1
  let p_not_red_nor_purple := 0.95
  sorry

end find_white_balls_l2086_208641


namespace no_solution_for_inequalities_l2086_208672

theorem no_solution_for_inequalities :
  ¬ ∃ x : ℝ, 3 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 8 * x - 5 := by 
  sorry

end no_solution_for_inequalities_l2086_208672


namespace length_down_correct_l2086_208656

variable (rate_up rate_down time_up time_down length_down : ℕ)
variable (h1 : rate_up = 8)
variable (h2 : time_up = 2)
variable (h3 : time_down = time_up)
variable (h4 : rate_down = (3 / 2) * rate_up)
variable (h5 : length_down = rate_down * time_down)

theorem length_down_correct : length_down = 24 := by
  sorry

end length_down_correct_l2086_208656


namespace problem_statement_l2086_208642

theorem problem_statement (a b c : ℝ) (h : a^6 + b^6 + c^6 = 3) : a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := 
  sorry

end problem_statement_l2086_208642


namespace Kelly_remaining_games_l2086_208601

-- Definitions according to the conditions provided
def initial_games : ℝ := 121.0
def given_away : ℝ := 99.0
def remaining_games : ℝ := initial_games - given_away

-- The proof problem statement
theorem Kelly_remaining_games : remaining_games = 22.0 :=
by
  -- sorry is used here to skip the proof
  sorry

end Kelly_remaining_games_l2086_208601


namespace diff_squares_example_l2086_208604

theorem diff_squares_example :
  (311^2 - 297^2) / 14 = 608 :=
by
  -- The theorem statement directly follows from the conditions and question.
  sorry

end diff_squares_example_l2086_208604


namespace triangle_shape_and_maximum_tan_B_minus_C_l2086_208664

open Real

variable (A B C : ℝ)
variable (sin cos tan : ℝ → ℝ)

-- Given conditions
axiom sin2A_plus_3sin2C_equals_3sin2B : sin A ^ 2 + 3 * sin C ^ 2 = 3 * sin B ^ 2
axiom sinB_cosC_equals_2div3 : sin B * cos C = 2 / 3

-- Prove
theorem triangle_shape_and_maximum_tan_B_minus_C :
  (A = π / 2) ∧ (∀ x y : ℝ, (x = B - C) → tan x ≤ sqrt 2 / 4) :=
by sorry

end triangle_shape_and_maximum_tan_B_minus_C_l2086_208664


namespace blue_balloons_l2086_208678

theorem blue_balloons (total_balloons red_balloons green_balloons purple_balloons : ℕ)
  (h1 : total_balloons = 135)
  (h2 : red_balloons = 45)
  (h3 : green_balloons = 27)
  (h4 : purple_balloons = 32) :
  total_balloons - (red_balloons + green_balloons + purple_balloons) = 31 :=
by
  sorry

end blue_balloons_l2086_208678


namespace fish_left_in_tank_l2086_208695

-- Define the initial number of fish and the number of fish moved
def initialFish : Real := 212.0
def movedFish : Real := 68.0

-- Define the number of fish left in the tank
def fishLeft (initialFish : Real) (movedFish : Real) : Real := initialFish - movedFish

-- Theorem stating the problem
theorem fish_left_in_tank : fishLeft initialFish movedFish = 144.0 := by
  sorry

end fish_left_in_tank_l2086_208695


namespace range_of_a_l2086_208688

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end range_of_a_l2086_208688


namespace find_a_10_l2086_208610

theorem find_a_10 (a : ℕ → ℚ)
  (h0 : a 1 = 1)
  (h1 : ∀ n : ℕ, a (n + 1) = a n / (a n + 2)) :
  a 10 = 1 / 1023 :=
sorry

end find_a_10_l2086_208610


namespace remainder_when_divided_by_x_minus_2_l2086_208617

noncomputable def f (x : ℝ) : ℝ :=
  x^4 - 8 * x^3 + 12 * x^2 + 20 * x - 18

theorem remainder_when_divided_by_x_minus_2 :
  f 2 = 22 := 
sorry

end remainder_when_divided_by_x_minus_2_l2086_208617


namespace lowest_point_in_fourth_quadrant_l2086_208600

theorem lowest_point_in_fourth_quadrant (k : ℝ) (h : k < -1) :
    let x := - (k + 1) / 2
    let y := (4 * k - (k + 1) ^ 2) / 4
    y < 0 ∧ x > 0 :=
by
  let x := - (k + 1) / 2
  let y := (4 * k - (k + 1) ^ 2) / 4
  sorry

end lowest_point_in_fourth_quadrant_l2086_208600


namespace cos_2alpha_l2086_208684

theorem cos_2alpha (α : ℝ) (h : Real.cos (Real.pi / 2 + α) = (1 : ℝ) / 3) : 
  Real.cos (2 * α) = (7 : ℝ) / 9 := 
by
  sorry

end cos_2alpha_l2086_208684


namespace point_D_coordinates_l2086_208621

-- Define the vectors and points
structure Point where
  x : Int
  y : Int

def vector_add (p1 p2 : Point) : Point :=
  { x := p1.x + p2.x, y := p1.y + p2.y }

def scalar_multiply (k : Int) (p : Point) : Point :=
  { x := k * p.x, y := k * p.y }

def ab := Point.mk 5 (-3)
def c := Point.mk (-1) 3
def cd := scalar_multiply 2 ab

def D : Point := vector_add c cd

-- Theorem statement
theorem point_D_coordinates :
  D = Point.mk 9 (-3) :=
sorry

end point_D_coordinates_l2086_208621


namespace cannot_all_be_zero_l2086_208657

theorem cannot_all_be_zero :
  ¬ ∃ (f : ℕ → ℕ), (∀ i, f i ∈ { x : ℕ | 1 ≤ x ∧ x ≤ 1989 }) ∧
                   (∀ i j, f (i + j) = f i - f j) ∧
                   (∃ n, ∀ i, f (i + n) = 0) :=
by
  sorry

end cannot_all_be_zero_l2086_208657


namespace union_A_B_inter_complB_A_l2086_208654

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set A
def A : Set ℝ := {x | -3 < x ∧ x ≤ 6}

-- Define the set B
def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- Define the complement of B with respect to U
def compl_B : Set ℝ := {x | x ≤ -1 ∨ x ≥ 6}

-- Problem (1): Prove that A ∪ B = {x | -3 < x ∧ x ≤ 6}
theorem union_A_B : A ∪ B = {x | -3 < x ∧ x ≤ 6} := by
  sorry

-- Problem (2): Prove that (compl_B) ∩ A = {x | (-3 < x ∧ x ≤ -1) ∨ x = 6}
theorem inter_complB_A : compl_B ∩ A = {x | (-3 < x ∧ x ≤ -1) ∨ x = 6} := by 
  sorry

end union_A_B_inter_complB_A_l2086_208654


namespace exponential_equation_solution_l2086_208631

theorem exponential_equation_solution (b x : ℝ) (hb : b > 1) (hx : x > 0)
  (h : (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) :
  x = (3 / 5)^(Real.log b / Real.log (5 / 3)) :=
by
  sorry

end exponential_equation_solution_l2086_208631


namespace stubborn_robot_returns_to_start_l2086_208669

inductive Direction
| East | North | West | South

inductive Command
| STEP | LEFT

structure Robot :=
  (position : ℤ × ℤ)
  (direction : Direction)

def turnLeft : Direction → Direction
| Direction.East  => Direction.North
| Direction.North => Direction.West
| Direction.West  => Direction.South
| Direction.South => Direction.East

def moveStep : Robot → Robot
| ⟨(x, y), Direction.East⟩  => ⟨(x + 1, y), Direction.East⟩
| ⟨(x, y), Direction.North⟩ => ⟨(x, y + 1), Direction.North⟩
| ⟨(x, y), Direction.West⟩  => ⟨(x - 1, y), Direction.West⟩
| ⟨(x, y), Direction.South⟩ => ⟨(x, y - 1), Direction.South⟩

def executeCommand : Command → Robot → Robot
| Command.STEP, robot => moveStep robot
| Command.LEFT, robot => ⟨robot.position, turnLeft robot.direction⟩

def invertCommand : Command → Command
| Command.STEP => Command.LEFT
| Command.LEFT => Command.STEP

def executeSequence (seq : List Command) (robot : Robot) : Robot :=
  seq.foldl (λ r cmd => executeCommand cmd r) robot

def executeInvertedSequence (seq : List Command) (robot : Robot) : Robot :=
  seq.foldl (λ r cmd => executeCommand (invertCommand cmd) r) robot

def initialRobot : Robot := ⟨(0, 0), Direction.East⟩

def exampleProgram : List Command :=
  [Command.LEFT, Command.LEFT, Command.LEFT, Command.LEFT, Command.STEP, Command.STEP,
   Command.LEFT, Command.LEFT]

theorem stubborn_robot_returns_to_start :
  let robot := executeSequence exampleProgram initialRobot
  executeInvertedSequence exampleProgram robot = initialRobot :=
by
  sorry

end stubborn_robot_returns_to_start_l2086_208669


namespace evaluate_fraction_l2086_208607

theorem evaluate_fraction : 3 / (2 - 3 / 4) = 12 / 5 := by
  sorry

end evaluate_fraction_l2086_208607


namespace remainder_when_sum_divided_mod7_l2086_208609

theorem remainder_when_sum_divided_mod7 (a b c : ℕ)
  (h1 : a < 7) (h2 : b < 7) (h3 : c < 7)
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0)
  (h7 : a * b * c % 7 = 2)
  (h8 : 3 * c % 7 = 1)
  (h9 : 4 * b % 7 = (2 + b) % 7) :
  (a + b + c) % 7 = 3 := by
  sorry

end remainder_when_sum_divided_mod7_l2086_208609


namespace digit_equation_l2086_208696

-- Definitions for digits and the equation components
def is_digit (x : ℤ) : Prop := 0 ≤ x ∧ x ≤ 9

def three_digit_number (A B C : ℤ) : ℤ := 100 * A + 10 * B + C
def two_digit_number (A D : ℤ) : ℤ := 10 * A + D
def four_digit_number (A D C : ℤ) : ℤ := 1000 * A + 100 * D + 10 * D + C

-- Statement of the theorem
theorem digit_equation (A B C D : ℤ) (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hD : is_digit D) :
  three_digit_number A B C * two_digit_number A D = four_digit_number A D C :=
sorry

end digit_equation_l2086_208696


namespace remainder_55_pow_55_plus_10_mod_8_l2086_208626

theorem remainder_55_pow_55_plus_10_mod_8 : (55 ^ 55 + 10) % 8 = 1 :=
by
  sorry

end remainder_55_pow_55_plus_10_mod_8_l2086_208626


namespace max_value_of_x_l2086_208685

theorem max_value_of_x : ∃ x : ℝ, 
  ( (4*x - 16) / (3*x - 4) )^2 + ( (4*x - 16) / (3*x - 4) ) = 18 
  ∧ x = (3 * Real.sqrt 73 + 28) / (11 - Real.sqrt 73) :=
sorry

end max_value_of_x_l2086_208685


namespace interest_difference_l2086_208662

noncomputable def principal : ℝ := 6200
noncomputable def rate : ℝ := 5 / 100
noncomputable def time : ℝ := 10

noncomputable def interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem interest_difference :
  (principal - interest principal rate time) = 3100 := 
by
  sorry

end interest_difference_l2086_208662


namespace quadratic_roots_range_l2086_208630

theorem quadratic_roots_range (m : ℝ) : 
  (2 * x^2 - (m + 1) * x + m = 0) → 
  (m^2 - 6 * m + 1 > 0) → 
  (0 < m) → 
  (0 < m ∧ m < 3 - 2 * Real.sqrt 2 ∨ m > 3 + 2 * Real.sqrt 2) :=
by
  sorry

end quadratic_roots_range_l2086_208630


namespace octal_67_equals_ternary_2001_l2086_208620

def octalToDecimal (n : Nat) : Nat :=
  -- Definition of octal to decimal conversion omitted
  sorry

def decimalToTernary (n : Nat) : Nat :=
  -- Definition of decimal to ternary conversion omitted
  sorry

theorem octal_67_equals_ternary_2001 : 
  decimalToTernary (octalToDecimal 67) = 2001 :=
by
  -- Proof omitted
  sorry

end octal_67_equals_ternary_2001_l2086_208620


namespace units_digit_M_M12_l2086_208606

def modifiedLucas (n : ℕ) : ℕ :=
  match n with
  | 0     => 3
  | 1     => 2
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

theorem units_digit_M_M12 (n : ℕ) (H : modifiedLucas 12 = 555) : 
  (modifiedLucas (modifiedLucas 12) % 10) = 1 := by
  sorry

end units_digit_M_M12_l2086_208606


namespace right_triangle_perimeter_l2086_208622

noncomputable def perimeter_right_triangle (a b : ℝ) (hypotenuse : ℝ) : ℝ :=
  a + b + hypotenuse

theorem right_triangle_perimeter (a b : ℝ) (ha : a^2 + b^2 = 25) (hab : a * b = 10) (hhypotenuse : hypotenuse = 5) :
  perimeter_right_triangle a b hypotenuse = 5 + 3 * Real.sqrt 5 :=
by
  sorry

end right_triangle_perimeter_l2086_208622


namespace arithmetic_progression_terms_even_sums_l2086_208627

theorem arithmetic_progression_terms_even_sums (n a d : ℕ) (h_even : Even n) 
  (h_odd_sum : n * (a + (n - 2) * d) = 60) 
  (h_even_sum : n * (a + d + a + (n - 1) * d) = 72) 
  (h_last_first : (n - 1) * d = 12) : n = 8 := 
sorry

end arithmetic_progression_terms_even_sums_l2086_208627


namespace min_value_reciprocal_sum_l2086_208613

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (1 / a + 1 / b) ≥ 4 :=
sorry

end min_value_reciprocal_sum_l2086_208613


namespace cafeteria_pies_l2086_208658

theorem cafeteria_pies (total_apples handed_out_apples apples_per_pie : ℕ) (h1 : total_apples = 47) (h2 : handed_out_apples = 27) (h3 : apples_per_pie = 4) :
  (total_apples - handed_out_apples) / apples_per_pie = 5 :=
by {
  sorry
}

end cafeteria_pies_l2086_208658


namespace colored_line_midpoint_l2086_208698

theorem colored_line_midpoint (L : ℝ → Prop) (p1 p2 : ℝ) :
  (L p1 → L p2) →
  (∃ A B C : ℝ, L A = L B ∧ L B = L C ∧ 2 * B = A + C ∧ L A = L C) :=
sorry

end colored_line_midpoint_l2086_208698


namespace abba_divisible_by_11_l2086_208616

-- Given any two-digit number with digits a and b
def is_divisible_by_11 (a b : ℕ) : Prop :=
  (1001 * a + 110 * b) % 11 = 0

theorem abba_divisible_by_11 (a b : ℕ) (ha : a < 10) (hb : b < 10) : is_divisible_by_11 a b :=
  sorry

end abba_divisible_by_11_l2086_208616


namespace max_digit_sum_of_watch_display_l2086_208637

-- Define the problem conditions
def valid_hour (h : ℕ) : Prop := 0 ≤ h ∧ h < 24
def valid_minute (m : ℕ) : Prop := 0 ≤ m ∧ m < 60
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the proof problem
theorem max_digit_sum_of_watch_display : 
  ∃ h m : ℕ, valid_hour h ∧ valid_minute m ∧ (digit_sum h + digit_sum m = 24) :=
sorry

end max_digit_sum_of_watch_display_l2086_208637


namespace slope_of_line_m_equals_neg_2_l2086_208649

theorem slope_of_line_m_equals_neg_2
  (m : ℝ)
  (h : (3 * m - 6) / (1 + m) = 12) :
  m = -2 :=
sorry

end slope_of_line_m_equals_neg_2_l2086_208649


namespace james_choices_count_l2086_208611

-- Define the conditions as Lean definitions
def isAscending (a b c d e : ℕ) : Prop := a < b ∧ b < c ∧ c < d ∧ d < e

def inRange (a b c d e : ℕ) : Prop := a ≤ 8 ∧ b ≤ 8 ∧ c ≤ 8 ∧ d ≤ 8 ∧ e ≤ 8

def meanEqualsMedian (a b c d e : ℕ) : Prop :=
  (a + b + c + d + e) / 5 = c

-- Define the problem statement
theorem james_choices_count :
  ∃ (s : Finset (ℕ × ℕ × ℕ × ℕ × ℕ)), 
    (∀ (a b c d e : ℕ), (a, b, c, d, e) ∈ s ↔ isAscending a b c d e ∧ inRange a b c d e ∧ meanEqualsMedian a b c d e) ∧
    s.card = 10 :=
sorry

end james_choices_count_l2086_208611


namespace sum_of_radii_of_tangent_circles_l2086_208652

theorem sum_of_radii_of_tangent_circles : 
  ∃ r1 r2 : ℝ, 
    r1 > 0 ∧
    r2 > 0 ∧
    ((r1 - 4)^2 + r1^2 = (r1 + 2)^2) ∧ 
    ((r2 - 4)^2 + r2^2 = (r2 + 2)^2) ∧
    r1 + r2 = 12 :=
by
  sorry

end sum_of_radii_of_tangent_circles_l2086_208652


namespace product_increase_l2086_208615

theorem product_increase (a b : ℝ) (h : (a + 1) * (b + 1) = 2 * a * b) (ha : a ≠ 0) (hb : b ≠ 0) : 
  ((a^2 - 1) * (b^2 - 1) = 4 * a * b) :=
sorry

end product_increase_l2086_208615


namespace total_sequences_correct_l2086_208618

/-- 
Given 6 blocks arranged such that:
1. Block 1 must be removed first.
2. Blocks 2 and 3 become accessible after Block 1 is removed.
3. Blocks 4, 5, and 6 become accessible after Blocks 2 and 3 are removed.
4. A block can only be removed if no other block is stacked on top of it. 

Prove that the total number of possible sequences to remove all the blocks is 10.
-/
def total_sequences_to_remove_blocks : ℕ := 10

theorem total_sequences_correct : 
  total_sequences_to_remove_blocks = 10 :=
sorry

end total_sequences_correct_l2086_208618


namespace evaluate_heartsuit_l2086_208683

-- Define the given operation
def heartsuit (x y : ℝ) : ℝ := abs (x - y)

-- State the proof problem in Lean
theorem evaluate_heartsuit (a b : ℝ) (h_a : a = 3) (h_b : b = -1) :
  heartsuit (heartsuit a b) (heartsuit (2 * a) (2 * b)) = 4 :=
by
  -- acknowledging that it's correct without providing the solution steps
  sorry

end evaluate_heartsuit_l2086_208683


namespace domain_of_function_l2086_208638

noncomputable def domain_of_f : Set ℝ :=
  {x | x > -1/2 ∧ x ≠ 1}

theorem domain_of_function :
  (∀ x : ℝ, (2 * x + 1 ≥ 0) ∧ (2 * x^2 - x - 1 ≠ 0) ↔ (x > -1/2 ∧ x ≠ 1)) := by
  sorry

end domain_of_function_l2086_208638


namespace distance_between_parallel_lines_l2086_208668

theorem distance_between_parallel_lines
  (O A B C D P Q : ℝ) -- Points on the circle with P and Q as defined midpoints
  (r d : ℝ) -- Radius of the circle and distance between the parallel lines
  (h_AB : dist A B = 36) -- Length of chord AB
  (h_CD : dist C D = 36) -- Length of chord CD
  (h_BC : dist B C = 40) -- Length of chord BC
  (h_OA : dist O A = r) 
  (h_OB : dist O B = r)
  (h_OC : dist O C = r)
  (h_PQ_parallel : dist P Q = d) -- Midpoints
  : d = 4 * Real.sqrt 19 / 3 :=
sorry

end distance_between_parallel_lines_l2086_208668


namespace triangular_array_of_coins_l2086_208612

theorem triangular_array_of_coins (N : ℤ) (h : N * (N + 1) / 2 = 3003) : N = 77 :=
by
  sorry

end triangular_array_of_coins_l2086_208612


namespace math_problem_l2086_208639

theorem math_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4*x + y) / (x - 4*y) = -3) : 
  (x + 4*y) / (4*x - y) = 39 / 37 :=
by
  sorry

end math_problem_l2086_208639


namespace christine_savings_l2086_208648

def commission_rate : ℝ := 0.12
def total_sales : ℝ := 24000
def personal_needs_percentage : ℝ := 0.60
def savings_percentage : ℝ := 1 - personal_needs_percentage

noncomputable def commission_earned : ℝ := total_sales * commission_rate
noncomputable def amount_saved : ℝ := commission_earned * savings_percentage

theorem christine_savings :
  amount_saved = 1152 :=
by
  sorry

end christine_savings_l2086_208648


namespace only_odd_digit_squared_n_l2086_208697

def is_odd_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → is_odd_digit d

theorem only_odd_digit_squared_n (n : ℕ) :
  0 < n ∧ has_only_odd_digits (n * n) ↔ n = 1 ∨ n = 3 :=
sorry

end only_odd_digit_squared_n_l2086_208697


namespace intersection_line_constant_l2086_208693

-- Definitions based on conditions provided:
def circle1_eq (x y : ℝ) : Prop := (x + 6)^2 + (y - 2)^2 = 144
def circle2_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 9)^2 = 65

-- The theorem statement
theorem intersection_line_constant (c : ℝ) : 
  (∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y ∧ x + y = c) ↔ c = 6 :=
by
  sorry

end intersection_line_constant_l2086_208693


namespace smaller_of_two_numbers_in_ratio_l2086_208667

theorem smaller_of_two_numbers_in_ratio (x y a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : x / y = a / b) (h3 : x + y = c) : 
  min x y = (a * c) / (a + b) :=
by
  sorry

end smaller_of_two_numbers_in_ratio_l2086_208667


namespace nec_but_not_suff_condition_l2086_208680

variables {p q : Prop}

theorem nec_but_not_suff_condition (hp : ¬p) : 
  (p ∨ q → False) ↔ (¬p) ∧ ¬(¬p → p ∨ q) :=
by {
  sorry
}

end nec_but_not_suff_condition_l2086_208680


namespace apples_pie_calculation_l2086_208640

-- Defining the conditions
def total_apples : ℕ := 34
def non_ripe_apples : ℕ := 6
def apples_per_pie : ℕ := 4 

-- Stating the problem
theorem apples_pie_calculation : (total_apples - non_ripe_apples) / apples_per_pie = 7 := by
  -- Proof would go here. For the structure of the task, we use sorry.
  sorry

end apples_pie_calculation_l2086_208640


namespace sqrt_six_ineq_l2086_208602

theorem sqrt_six_ineq : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end sqrt_six_ineq_l2086_208602


namespace value_of_a_l2086_208653

theorem value_of_a (x a : ℤ) (h1 : x = 2) (h2 : 3 * x - a = -x + 7) : a = 1 :=
by
  sorry

end value_of_a_l2086_208653


namespace fg_of_3_eq_29_l2086_208682

def g (x : ℕ) : ℕ := x * x
def f (x : ℕ) : ℕ := 3 * x + 2

theorem fg_of_3_eq_29 : f (g 3) = 29 :=
by
  sorry

end fg_of_3_eq_29_l2086_208682


namespace square_perimeter_l2086_208603

theorem square_perimeter (A_total : ℕ) (A_common : ℕ) (A_circle : ℕ) 
  (H1 : A_total = 329)
  (H2 : A_common = 101)
  (H3 : A_circle = 234) :
  4 * (Int.sqrt (A_total - A_circle + A_common)) = 56 :=
by
  -- Since we are only required to provide the statement, we can skip the proof steps.
  -- sorry to skip the proof.
  sorry

end square_perimeter_l2086_208603


namespace lines_parallel_m_values_l2086_208699

theorem lines_parallel_m_values (m : ℝ) :
    (∀ x y : ℝ, (m - 2) * x - y - 1 = 0 ↔ 3 * x - m * y = 0) ↔ (m = -1 ∨ m = 3) :=
by
  sorry

end lines_parallel_m_values_l2086_208699


namespace calories_per_cookie_l2086_208690

theorem calories_per_cookie :
  ∀ (cookies_per_bag bags_per_box total_calories total_number_cookies : ℕ),
  cookies_per_bag = 20 →
  bags_per_box = 4 →
  total_calories = 1600 →
  total_number_cookies = cookies_per_bag * bags_per_box →
  (total_calories / total_number_cookies) = 20 :=
by sorry

end calories_per_cookie_l2086_208690


namespace symmetric_sum_eq_two_l2086_208647

-- Definitions and conditions
def symmetric (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = -Q.2

def P : ℝ × ℝ := (sorry, 1)
def Q : ℝ × ℝ := (-3, sorry)

-- Problem statement
theorem symmetric_sum_eq_two (h : symmetric P Q) : P.1 + Q.2 = 2 :=
by
  -- Proof omitted
  sorry

end symmetric_sum_eq_two_l2086_208647


namespace smallest_number_of_cookies_l2086_208619

theorem smallest_number_of_cookies
  (n : ℕ) 
  (hn : 4 * n - 4 = (n^2) / 2) : n = 7 → n^2 = 49 := 
by
  sorry

end smallest_number_of_cookies_l2086_208619


namespace sum_of_areas_l2086_208681

theorem sum_of_areas (radii : ℕ → ℝ) (areas : ℕ → ℝ) (h₁ : radii 0 = 2) 
  (h₂ : ∀ n, radii (n + 1) = radii n / 3) 
  (h₃ : ∀ n, areas n = π * (radii n) ^ 2) : 
  ∑' n, areas n = (9 * π) / 2 := 
by 
  sorry

end sum_of_areas_l2086_208681


namespace natural_number_pairs_sum_to_three_l2086_208636

theorem natural_number_pairs_sum_to_three :
  {p : ℕ × ℕ | p.1 + p.2 = 3} = {(1, 2), (2, 1)} :=
by
  sorry

end natural_number_pairs_sum_to_three_l2086_208636


namespace probability_of_two_germinates_is_48_over_125_l2086_208635

noncomputable def probability_of_exactly_two_germinates : ℚ :=
  let p := 4/5
  let n := 3
  let k := 2
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_of_two_germinates_is_48_over_125 :
  probability_of_exactly_two_germinates = 48/125 := by
    sorry

end probability_of_two_germinates_is_48_over_125_l2086_208635


namespace joe_eggs_around_park_l2086_208632

variable (total_eggs club_house_eggs town_hall_garden_eggs park_eggs : ℕ)

def joe_eggs (total_eggs club_house_eggs town_hall_garden_eggs park_eggs : ℕ) : Prop :=
  total_eggs = club_house_eggs + town_hall_garden_eggs + park_eggs

theorem joe_eggs_around_park (h1 : total_eggs = 20) (h2 : club_house_eggs = 12) (h3 : town_hall_garden_eggs = 3) :
  ∃ park_eggs, joe_eggs total_eggs club_house_eggs town_hall_garden_eggs park_eggs ∧ park_eggs = 5 :=
by
  sorry

end joe_eggs_around_park_l2086_208632


namespace diff_between_percent_and_fraction_l2086_208650

theorem diff_between_percent_and_fraction :
  (0.75 * 800) - ((7 / 8) * 1200) = -450 :=
by
  sorry

end diff_between_percent_and_fraction_l2086_208650


namespace yoongi_division_l2086_208661

theorem yoongi_division (x : ℕ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end yoongi_division_l2086_208661


namespace largest_square_side_l2086_208665

theorem largest_square_side {m n : ℕ} (h1 : m = 72) (h2 : n = 90) : Nat.gcd m n = 18 :=
by
  sorry

end largest_square_side_l2086_208665


namespace determine_angle_G_l2086_208651

theorem determine_angle_G 
  (C D E F G : ℝ)
  (hC : C = 120) 
  (h_linear_pair : C + D = 180)
  (hE : E = 50) 
  (hF : F = D) 
  (h_triangle_sum : E + F + G = 180) :
  G = 70 := 
sorry

end determine_angle_G_l2086_208651


namespace no_integer_triplets_for_equation_l2086_208655

theorem no_integer_triplets_for_equation (a b c : ℤ) : ¬ (a^2 + b^2 + 1 = 4 * c) :=
by
  sorry

end no_integer_triplets_for_equation_l2086_208655


namespace eccentricity_sum_cannot_be_2sqrt2_l2086_208644

noncomputable def e1 (a b : ℝ) := Real.sqrt (1 + (b^2) / (a^2))
noncomputable def e2 (a b : ℝ) := Real.sqrt (1 + (a^2) / (b^2))
noncomputable def e1_plus_e2 (a b : ℝ) := e1 a b + e2 a b

theorem eccentricity_sum_cannot_be_2sqrt2 (a b : ℝ) : e1_plus_e2 a b ≠ 2 * Real.sqrt 2 := by
  sorry

end eccentricity_sum_cannot_be_2sqrt2_l2086_208644


namespace john_running_speed_l2086_208625

noncomputable def find_running_speed (x : ℝ) : Prop :=
  (12 / (3 * x + 2) + 8 / x = 2.2)

theorem john_running_speed : ∃ x : ℝ, find_running_speed x ∧ abs (x - 0.47) < 0.01 :=
by
  sorry

end john_running_speed_l2086_208625


namespace exists_int_solutions_for_equations_l2086_208676

theorem exists_int_solutions_for_equations : 
  ∃ (x y : ℤ), x * y = 4747 ∧ x - y = -54 :=
by
  sorry

end exists_int_solutions_for_equations_l2086_208676


namespace quadratic_inequality_real_roots_l2086_208686

theorem quadratic_inequality_real_roots (c : ℝ) (h_pos : 0 < c) (h_ineq : c < 25) :
  ∃ x : ℝ, x^2 - 10 * x + c < 0 :=
sorry

end quadratic_inequality_real_roots_l2086_208686


namespace seat_notation_l2086_208666

theorem seat_notation (row1 col1 row2 col2 : ℕ) (h : (row1, col1) = (5, 2)) : (row2, col2) = (7, 3) :=
 by
  sorry

end seat_notation_l2086_208666
