import Mathlib

namespace punch_mixture_theorem_l3690_369017

/-- Given a 2-liter mixture that is 15% fruit juice, adding 0.125 liters of pure fruit juice
    results in a new mixture that is 20% fruit juice. -/
theorem punch_mixture_theorem :
  let initial_volume : ℝ := 2
  let initial_concentration : ℝ := 0.15
  let added_juice : ℝ := 0.125
  let target_concentration : ℝ := 0.20
  let final_volume : ℝ := initial_volume + added_juice
  let final_juice_amount : ℝ := initial_volume * initial_concentration + added_juice
  final_juice_amount / final_volume = target_concentration := by
sorry


end punch_mixture_theorem_l3690_369017


namespace exemplary_sequences_count_l3690_369075

/-- The number of distinct 6-letter sequences from "EXEMPLARY" with given conditions -/
def exemplary_sequences : ℕ :=
  let available_letters := 6  -- X, A, M, P, L, R
  let positions_to_fill := 4  -- positions 2, 3, 4, 5
  Nat.factorial available_letters / Nat.factorial (available_letters - positions_to_fill)

/-- Theorem stating the number of distinct sequences is 360 -/
theorem exemplary_sequences_count :
  exemplary_sequences = 360 := by
  sorry

#eval exemplary_sequences  -- Should output 360

end exemplary_sequences_count_l3690_369075


namespace simplify_expression_l3690_369007

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x - 2) = 4*x - 20 := by
  sorry

end simplify_expression_l3690_369007


namespace good_numbers_characterization_l3690_369033

/-- A natural number is good if every natural divisor of n, when increased by 1, is a divisor of n+1 -/
def IsGood (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → (d + 1) ∣ (n + 1)

/-- Characterization of good numbers -/
theorem good_numbers_characterization (n : ℕ) :
  IsGood n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1) :=
sorry

end good_numbers_characterization_l3690_369033


namespace smallest_angle_cosine_equality_l3690_369076

theorem smallest_angle_cosine_equality (θ : Real) : 
  (θ > 0) →
  (Real.cos θ = Real.sin (π/4) + Real.cos (π/3) - Real.sin (π/6) - Real.cos (π/12)) →
  (θ = π/6) :=
by sorry

end smallest_angle_cosine_equality_l3690_369076


namespace probability_of_red_ball_l3690_369098

def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3

def red_balls : ℕ := total_balls - yellow_balls - green_balls

def probability_red_ball : ℚ := red_balls / total_balls

theorem probability_of_red_ball :
  probability_red_ball = 3 / 5 := by
  sorry

end probability_of_red_ball_l3690_369098


namespace simplify_sqrt_sum_product_l3690_369099

theorem simplify_sqrt_sum_product (m n a b : ℝ) 
  (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) (hb : 0 < b) (hab : a > b)
  (hsum : a + b = m) (hprod : a * b = n) :
  Real.sqrt (m + 2 * Real.sqrt n) = Real.sqrt a + Real.sqrt b ∧
  Real.sqrt (m - 2 * Real.sqrt n) = Real.sqrt a - Real.sqrt b :=
by sorry

end simplify_sqrt_sum_product_l3690_369099


namespace junior_toy_ratio_l3690_369003

theorem junior_toy_ratio :
  let num_rabbits : ℕ := 16
  let monday_toys : ℕ := 6
  let friday_toys : ℕ := 4 * monday_toys
  let wednesday_toys : ℕ := wednesday_toys -- Unknown variable
  let saturday_toys : ℕ := wednesday_toys / 2
  let toys_per_rabbit : ℕ := 3
  
  num_rabbits * toys_per_rabbit = monday_toys + wednesday_toys + friday_toys + saturday_toys →
  wednesday_toys = 2 * monday_toys :=
by sorry

end junior_toy_ratio_l3690_369003


namespace floor_sqrt_10_l3690_369059

theorem floor_sqrt_10 : ⌊Real.sqrt 10⌋ = 3 := by
  sorry

end floor_sqrt_10_l3690_369059


namespace semicircle_area_with_inscribed_rectangle_l3690_369077

theorem semicircle_area_with_inscribed_rectangle :
  ∀ (r : ℝ),
  r > 0 →
  ∃ (semicircle_area : ℝ),
  (3 : ℝ)^2 + 1^2 = (2 * r)^2 →
  semicircle_area = (13 * π) / 8 :=
by
  sorry

end semicircle_area_with_inscribed_rectangle_l3690_369077


namespace bacteria_division_theorem_l3690_369065

/-- Represents a binary tree of bacteria -/
inductive BacteriaTree
  | Leaf : BacteriaTree
  | Node : BacteriaTree → BacteriaTree → BacteriaTree

/-- Counts the number of nodes in a BacteriaTree -/
def count_nodes : BacteriaTree → Nat
  | BacteriaTree.Leaf => 1
  | BacteriaTree.Node left right => count_nodes left + count_nodes right

/-- Checks if a subtree with the desired properties exists -/
def exists_balanced_subtree (tree : BacteriaTree) : Prop :=
  ∃ (subtree : BacteriaTree), 
    (count_nodes subtree ≥ 334 ∧ count_nodes subtree ≤ 667)

theorem bacteria_division_theorem (tree : BacteriaTree) 
  (h : count_nodes tree = 1000) : 
  exists_balanced_subtree tree :=
sorry

end bacteria_division_theorem_l3690_369065


namespace B_initial_investment_correct_l3690_369066

/-- Represents the initial investment of B in rupees -/
def B_initial_investment : ℝ := 4866.67

/-- Represents A's initial investment in rupees -/
def A_initial_investment : ℝ := 2000

/-- Represents the amount A withdraws after 8 months in rupees -/
def A_withdrawal : ℝ := 1000

/-- Represents the amount B advances after 8 months in rupees -/
def B_advance : ℝ := 1000

/-- Represents the total profit at the end of the year in rupees -/
def total_profit : ℝ := 630

/-- Represents A's share of the profit in rupees -/
def A_profit_share : ℝ := 175

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the number of months before A withdraws and B advances -/
def months_before_change : ℕ := 8

theorem B_initial_investment_correct :
  B_initial_investment * months_before_change +
  (B_initial_investment + B_advance) * (months_in_year - months_before_change) =
  (total_profit - A_profit_share) / A_profit_share *
  (A_initial_investment * months_in_year) :=
by sorry

end B_initial_investment_correct_l3690_369066


namespace problem_statement_l3690_369050

theorem problem_statement (x y z : ℝ) (h1 : x + y = 5) (h2 : z^2 = x*y + y - 9) :
  x + 2*y + 3*z = 8 := by
  sorry

end problem_statement_l3690_369050


namespace rind_papyrus_fraction_decomposition_l3690_369037

theorem rind_papyrus_fraction_decomposition : 
  (2 : ℚ) / 73 = 1 / 60 + 1 / 219 + 1 / 292 + 1 / 365 := by
  sorry

end rind_papyrus_fraction_decomposition_l3690_369037


namespace orange_bin_theorem_l3690_369054

def final_orange_count (initial : ℕ) (thrown_away : ℕ) (added : ℕ) : ℕ :=
  initial - thrown_away + added

theorem orange_bin_theorem (initial : ℕ) (thrown_away : ℕ) (added : ℕ) 
  (h1 : thrown_away ≤ initial) :
  final_orange_count initial thrown_away added = initial - thrown_away + added :=
by
  sorry

#eval final_orange_count 31 9 38

end orange_bin_theorem_l3690_369054


namespace M_inter_N_eq_M_l3690_369021

open Set

def M : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 3}
def N : Set ℝ := {x : ℝ | x^2 - 3*x - 4 < 0}

theorem M_inter_N_eq_M : M ∩ N = M := by sorry

end M_inter_N_eq_M_l3690_369021


namespace greater_number_problem_l3690_369006

theorem greater_number_problem (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0) 
  (h4 : x * y = 2048) (h5 : (x + y) - (x - y) = 64) : x = 64 := by
  sorry

end greater_number_problem_l3690_369006


namespace complex_modulus_problem_l3690_369056

theorem complex_modulus_problem (x y : ℝ) :
  (Complex.I + 1) * x = Complex.I * y + 1 →
  Complex.abs (x + Complex.I * y) = Real.sqrt 2 := by
sorry

end complex_modulus_problem_l3690_369056


namespace rose_cost_is_six_l3690_369028

/-- The cost of each rose when buying in bulk -/
def rose_cost (dozen : ℕ) (discount_percent : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid / (discount_percent / 100) / (dozen * 12)

/-- Theorem: The cost of each rose is $6 -/
theorem rose_cost_is_six :
  rose_cost 5 80 288 = 6 := by
  sorry

end rose_cost_is_six_l3690_369028


namespace arithmetic_sequence_properties_l3690_369060

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the properties of the sequence
theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 2 + a 4 = 10)
  (h3 : ∃ r : ℝ, r ≠ 0 ∧ a 2 = a 1 * r ∧ a 5 = a 2 * r)
  (h4 : arithmetic_sequence a d) :
  a 1 = 1 ∧ ∀ n : ℕ, a n = 2 * n - 1 :=
by sorry

end arithmetic_sequence_properties_l3690_369060


namespace residue_of_11_pow_2021_mod_19_l3690_369002

theorem residue_of_11_pow_2021_mod_19 :
  (11 : ℤ) ^ 2021 ≡ 17 [ZMOD 19] := by
  sorry

end residue_of_11_pow_2021_mod_19_l3690_369002


namespace roots_sum_of_squares_l3690_369025

theorem roots_sum_of_squares (p q r s : ℝ) : 
  (r^2 - 3*p*r + 2*q = 0) → 
  (s^2 - 3*p*s + 2*q = 0) → 
  (r^2 + s^2 = 9*p^2 - 4*q) := by
sorry

end roots_sum_of_squares_l3690_369025


namespace max_value_trigonometric_function_l3690_369071

open Real

theorem max_value_trigonometric_function :
  ∃ (M : ℝ), M = 3 - 2 * sqrt 2 ∧
  ∀ θ : ℝ, 0 < θ → θ < π / 2 →
  (1 / sin θ - 1) * (1 / cos θ - 1) ≤ M :=
sorry

end max_value_trigonometric_function_l3690_369071


namespace max_sum_in_S_l3690_369046

/-- The set of ordered pairs of integers (x,y) satisfying x^2 + y^2 = 50 -/
def S : Set (ℤ × ℤ) := {p | p.1^2 + p.2^2 = 50}

/-- The theorem stating that the maximum sum of x+y for (x,y) in S is 10 -/
theorem max_sum_in_S : (⨆ p ∈ S, (p.1 + p.2 : ℤ)) = 10 := by sorry

end max_sum_in_S_l3690_369046


namespace triangles_on_circle_l3690_369013

theorem triangles_on_circle (n : ℕ) (h : n = 15) : 
  (Nat.choose n 3) = 455 := by sorry

end triangles_on_circle_l3690_369013


namespace final_number_theorem_l3690_369072

/-- Represents the state of the number on the board -/
structure BoardState where
  digits : List Nat
  deriving Repr

/-- Applies Operation 1 to the board state -/
def applyOperation1 (state : BoardState) : BoardState :=
  sorry

/-- Applies Operation 2 to the board state -/
def applyOperation2 (state : BoardState) : BoardState :=
  sorry

/-- Checks if a number is a valid final state (two digits) -/
def isValidFinalState (state : BoardState) : Bool :=
  sorry

/-- Generates the initial state with 100 fives -/
def initialState : BoardState :=
  { digits := List.replicate 100 5 }

/-- Theorem stating the final result of the operations -/
theorem final_number_theorem :
  ∃ (finalState : BoardState),
    (isValidFinalState finalState) ∧
    (finalState.digits = [8, 0] ∨ finalState.digits = [6, 6]) ∧
    (∃ (operations : List (BoardState → BoardState)),
      operations.foldl (λ state op => op state) initialState = finalState) :=
sorry

end final_number_theorem_l3690_369072


namespace product_from_lcm_gcd_l3690_369014

theorem product_from_lcm_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : Nat.lcm a b = 60) (h4 : Nat.gcd a b = 5) : a * b = 300 := by
  sorry

end product_from_lcm_gcd_l3690_369014


namespace passenger_speed_on_train_l3690_369035

/-- The speed of a passenger relative to the railway track when moving on a train -/
def passenger_speed_relative_to_track (train_speed passenger_speed : ℝ) : ℝ × ℝ :=
  (train_speed + passenger_speed, |train_speed - passenger_speed|)

/-- Theorem: The speed of a passenger relative to the railway track
    when the train moves at 60 km/h and the passenger moves at 3 km/h relative to the train -/
theorem passenger_speed_on_train :
  let train_speed := 60
  let passenger_speed := 3
  passenger_speed_relative_to_track train_speed passenger_speed = (63, 57) := by
  sorry

end passenger_speed_on_train_l3690_369035


namespace football_team_progress_l3690_369053

def team_progress (loss : Int) (gain : Int) : Int :=
  gain - loss

theorem football_team_progress :
  team_progress 5 13 = 8 := by
  sorry

end football_team_progress_l3690_369053


namespace max_value_of_expression_upper_bound_achievable_l3690_369019

theorem max_value_of_expression (x : ℝ) :
  x^4 / (x^8 + 2*x^6 - 3*x^4 + 5*x^3 + 8*x^2 + 5*x + 25) ≤ 1/15 :=
by sorry

theorem upper_bound_achievable :
  ∃ x : ℝ, x^4 / (x^8 + 2*x^6 - 3*x^4 + 5*x^3 + 8*x^2 + 5*x + 25) = 1/15 :=
by sorry

end max_value_of_expression_upper_bound_achievable_l3690_369019


namespace parallelepiped_net_theorem_l3690_369088

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Represents a net formed from a parallelepiped -/
structure Net :=
  (squares : ℕ)

/-- Unfolds a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := 2 * (p.length * p.width + p.length * p.height + p.width * p.height) }

/-- Removes one square from a net -/
def remove_square (n : Net) : Net :=
  { squares := n.squares - 1 }

theorem parallelepiped_net_theorem (p : Parallelepiped) 
  (h1 : p.length = 2)
  (h2 : p.width = 1)
  (h3 : p.height = 1) :
  (remove_square (unfold p)).squares = 9 :=
sorry

end parallelepiped_net_theorem_l3690_369088


namespace rectangular_field_area_l3690_369080

/-- Calculates the area of a rectangular field given its perimeter and width. -/
theorem rectangular_field_area
  (perimeter : ℝ) (width : ℝ)
  (h_perimeter : perimeter = 30)
  (h_width : width = 5) :
  width * (perimeter / 2 - width) = 50 := by
  sorry

#check rectangular_field_area

end rectangular_field_area_l3690_369080


namespace total_tape_area_l3690_369078

/-- Calculate the total area of tape used for taping boxes -/
theorem total_tape_area (box1_length box1_width : ℕ) (box2_side : ℕ) (box3_length box3_width : ℕ)
  (box1_count box2_count box3_count : ℕ) (tape_width overlap : ℕ) :
  box1_length = 30 ∧ box1_width = 15 ∧ 
  box2_side = 40 ∧
  box3_length = 50 ∧ box3_width = 20 ∧
  box1_count = 5 ∧ box2_count = 2 ∧ box3_count = 3 ∧
  tape_width = 2 ∧ overlap = 2 →
  (box1_count * (box1_length + overlap + 2 * (box1_width + overlap)) +
   box2_count * (3 * (box2_side + overlap)) +
   box3_count * (box3_length + overlap + 2 * (box3_width + overlap))) * tape_width = 1740 := by
  sorry

end total_tape_area_l3690_369078


namespace polygon_with_720_degrees_is_hexagon_l3690_369055

/-- A polygon with a sum of interior angles of 720° has 6 sides. -/
theorem polygon_with_720_degrees_is_hexagon :
  ∀ n : ℕ,
  (180 * (n - 2) = 720) →
  n = 6 :=
by sorry

end polygon_with_720_degrees_is_hexagon_l3690_369055


namespace teacher_selection_problem_l3690_369026

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem teacher_selection_problem (male female total selected : ℕ) 
  (h1 : male = 3)
  (h2 : female = 6)
  (h3 : total = male + female)
  (h4 : selected = 5) :
  choose total selected - choose female selected = 120 := by sorry

end teacher_selection_problem_l3690_369026


namespace inequality_holds_iff_m_greater_than_neg_three_fourths_l3690_369058

theorem inequality_holds_iff_m_greater_than_neg_three_fourths (m : ℝ) :
  (∀ x : ℝ, m^2 * x^2 - 2*m*x > -x^2 - x - 1) ↔ m > -3/4 := by sorry

end inequality_holds_iff_m_greater_than_neg_three_fourths_l3690_369058


namespace sum_at_two_and_minus_two_l3690_369029

def cubic_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, P x = a * x^3 + b * x^2 + c * x + d

theorem sum_at_two_and_minus_two
  (P : ℝ → ℝ)
  (k : ℝ)
  (h_cubic : cubic_polynomial P)
  (h_zero : P 0 = k)
  (h_one : P 1 = 3 * k)
  (h_neg_one : P (-1) = 4 * k) :
  P 2 + P (-2) = 22 * k :=
sorry

end sum_at_two_and_minus_two_l3690_369029


namespace kickball_students_total_l3690_369039

theorem kickball_students_total (wednesday : ℕ) (fewer_thursday : ℕ) : 
  wednesday = 37 → fewer_thursday = 9 → 
  wednesday + (wednesday - fewer_thursday) = 65 := by
  sorry

end kickball_students_total_l3690_369039


namespace vertex_of_quadratic_l3690_369052

/-- The quadratic function f(x) = -2(x+1)^2-4 -/
def f (x : ℝ) : ℝ := -2 * (x + 1)^2 - 4

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-1, -4)

/-- Theorem: The vertex of f(x) = -2(x+1)^2-4 is at (-1, -4) -/
theorem vertex_of_quadratic :
  (∀ x, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end vertex_of_quadratic_l3690_369052


namespace f_one_ge_six_l3690_369069

/-- A quadratic function f(x) = x^2 + 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

/-- Theorem: If f(x) is increasing on (-1, +∞), then f(1) ≥ 6 -/
theorem f_one_ge_six (a : ℝ) 
  (h : ∀ x y, -1 < x ∧ x < y → f a x < f a y) : 
  f a 1 ≥ 6 := by
  sorry


end f_one_ge_six_l3690_369069


namespace borrowed_amount_l3690_369095

/-- Calculates the total interest paid over 9 years given the principal amount and interest rates -/
def totalInterest (principal : ℝ) : ℝ :=
  principal * (0.06 * 2 + 0.09 * 3 + 0.14 * 4)

/-- Theorem stating that given the interest rates and total interest paid, the principal amount borrowed is 12000 -/
theorem borrowed_amount (totalInterestPaid : ℝ) 
  (h : totalInterestPaid = 11400) : 
  ∃ principal : ℝ, totalInterest principal = totalInterestPaid ∧ principal = 12000 := by
  sorry

end borrowed_amount_l3690_369095


namespace two_digit_numbers_problem_l3690_369005

def F (p q : ℕ) : ℚ :=
  let p1 := p / 10
  let p2 := p % 10
  let q1 := q / 10
  let q2 := q % 10
  let sum := (1000 * p1 + 100 * q1 + 10 * q2 + p2) + (1000 * q1 + 100 * p1 + 10 * p2 + q2)
  (sum : ℚ) / 11

theorem two_digit_numbers_problem (m n : ℕ) 
  (hm : m ≤ 9) (hn : 1 ≤ n ∧ n ≤ 9) :
  let a := 10 + m
  let b := 10 * n + 5
  150 * F a 18 + F b 26 = 32761 →
  m + n = 12 ∨ m + n = 11 ∨ m + n = 10 := by
  sorry

end two_digit_numbers_problem_l3690_369005


namespace correct_train_sequence_l3690_369038

-- Define the steps as an enumeration
inductive TrainStep
  | BuyTicket
  | WaitInWaitingRoom
  | CheckTicketAtGate
  | BoardTrain

def correct_sequence : List TrainStep :=
  [TrainStep.BuyTicket, TrainStep.WaitInWaitingRoom, TrainStep.CheckTicketAtGate, TrainStep.BoardTrain]

-- Define a function to check if a given sequence is correct
def is_correct_sequence (sequence : List TrainStep) : Prop :=
  sequence = correct_sequence

-- Theorem stating that the given sequence is correct
theorem correct_train_sequence : 
  is_correct_sequence [TrainStep.BuyTicket, TrainStep.WaitInWaitingRoom, TrainStep.CheckTicketAtGate, TrainStep.BoardTrain] :=
by sorry


end correct_train_sequence_l3690_369038


namespace next_perfect_square_with_two_twos_l3690_369040

/-- A number begins with two 2s if its first two digits are 2 when written in base 10. -/
def begins_with_two_twos (n : ℕ) : Prop :=
  n ≥ 220 ∧ n < 230

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- A perfect square is a natural number that is the square of another natural number. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem next_perfect_square_with_two_twos : 
  (∀ n : ℕ, is_perfect_square n ∧ begins_with_two_twos n ∧ n < 2500 → n ≤ 225) ∧
  is_perfect_square 2500 ∧
  begins_with_two_twos 2500 ∧
  sum_of_digits 2500 = 7 :=
sorry

end next_perfect_square_with_two_twos_l3690_369040


namespace trig_sum_zero_l3690_369073

theorem trig_sum_zero (α β γ : ℝ) : 
  (Real.sin α / (Real.sin (α - β) * Real.sin (α - γ))) +
  (Real.sin β / (Real.sin (β - α) * Real.sin (β - γ))) +
  (Real.sin γ / (Real.sin (γ - α) * Real.sin (γ - β))) = 0 := by
  sorry

end trig_sum_zero_l3690_369073


namespace solution_set_for_f_squared_minimum_value_of_g_l3690_369096

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a (2*x + a) + 2 * f a x

-- Part 1
theorem solution_set_for_f_squared (x : ℝ) :
  f 1 x ^ 2 ≤ 2 ↔ 1 - Real.sqrt 2 ≤ x ∧ x ≤ 1 + Real.sqrt 2 :=
sorry

-- Part 2
theorem minimum_value_of_g (a : ℝ) (h : a > 0) :
  (∃ (m : ℝ), m = 4 ∧ ∀ (x : ℝ), g a x ≥ m) → a = 2 :=
sorry

end solution_set_for_f_squared_minimum_value_of_g_l3690_369096


namespace tetrahedron_cross_section_perimeter_bounds_l3690_369042

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- A quadrilateral cross-section of a regular tetrahedron -/
structure TetrahedronCrossSection (t : RegularTetrahedron) where
  perimeter : ℝ
  is_quadrilateral : True  -- This is a placeholder for the quadrilateral property

/-- The perimeter of a quadrilateral cross-section of a regular tetrahedron 
    is between 2a and 3a, where a is the edge length of the tetrahedron -/
theorem tetrahedron_cross_section_perimeter_bounds 
  (t : RegularTetrahedron) (c : TetrahedronCrossSection t) : 
  2 * t.edge_length ≤ c.perimeter ∧ c.perimeter ≤ 3 * t.edge_length :=
sorry

end tetrahedron_cross_section_perimeter_bounds_l3690_369042


namespace gift_wrapping_combinations_l3690_369023

/-- Represents the number of wrapping paper varieties -/
def wrapping_paper : Nat := 10

/-- Represents the number of ribbon colors -/
def ribbons : Nat := 3

/-- Represents the number of gift card types -/
def gift_cards : Nat := 4

/-- Represents the number of gift tag types -/
def gift_tags : Nat := 5

/-- Calculates the total number of gift wrapping combinations -/
def total_combinations : Nat := wrapping_paper * ribbons * gift_cards * gift_tags

/-- Theorem stating that the total number of gift wrapping combinations is 600 -/
theorem gift_wrapping_combinations :
  total_combinations = 600 := by sorry

end gift_wrapping_combinations_l3690_369023


namespace hypotenuse_division_l3690_369067

/-- A right triangle with one acute angle of 30° and hypotenuse of length 8 -/
structure RightTriangle30 where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is 8 -/
  hyp_eq_8 : hypotenuse = 8
  /-- One acute angle is 30° -/
  acute_angle : ℝ
  acute_angle_eq_30 : acute_angle = 30

/-- The altitude from the right angle vertex to the hypotenuse -/
def altitude (t : RightTriangle30) : ℝ := sorry

/-- The shorter segment of the hypotenuse divided by the altitude -/
def short_segment (t : RightTriangle30) : ℝ := sorry

/-- The longer segment of the hypotenuse divided by the altitude -/
def long_segment (t : RightTriangle30) : ℝ := sorry

/-- Theorem stating that the altitude divides the hypotenuse into segments of length 4 and 6 -/
theorem hypotenuse_division (t : RightTriangle30) : 
  short_segment t = 4 ∧ long_segment t = 6 :=
sorry

end hypotenuse_division_l3690_369067


namespace no_numbers_divisible_by_all_l3690_369031

theorem no_numbers_divisible_by_all : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 →
  ¬(2 ∣ n ∧ 3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n) := by
  sorry

end no_numbers_divisible_by_all_l3690_369031


namespace min_apples_count_l3690_369091

theorem min_apples_count : ∃ n : ℕ, n > 0 ∧
  n % 4 = 1 ∧
  n % 5 = 2 ∧
  n % 9 = 7 ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 9 = 7 → n ≤ m) ∧
  n = 97 := by
sorry

end min_apples_count_l3690_369091


namespace cos_alpha_plus_5pi_12_l3690_369018

theorem cos_alpha_plus_5pi_12 (α : ℝ) (h : Real.sin (α - π/12) = 1/3) :
  Real.cos (α + 5*π/12) = -1/3 := by
  sorry

end cos_alpha_plus_5pi_12_l3690_369018


namespace three_digit_number_operation_l3690_369034

theorem three_digit_number_operation (a b c : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ a = c + 3 →
  (4 * (100 * a + 10 * b + c) - (100 * c + 10 * b + a)) % 10 = 7 := by
sorry

end three_digit_number_operation_l3690_369034


namespace average_sitting_time_l3690_369044

def num_students : ℕ := 6
def num_seats : ℕ := 4
def travel_time_hours : ℕ := 3
def travel_time_minutes : ℕ := 12

theorem average_sitting_time :
  let total_minutes : ℕ := travel_time_hours * 60 + travel_time_minutes
  let total_sitting_time : ℕ := num_seats * total_minutes
  let avg_sitting_time : ℕ := total_sitting_time / num_students
  avg_sitting_time = 128 := by
sorry

end average_sitting_time_l3690_369044


namespace sphere_volume_and_radius_ratio_l3690_369081

theorem sphere_volume_and_radius_ratio (V_large V_small : ℝ) (h1 : V_large = 432 * Real.pi) (h2 : V_small = 0.15 * V_large) : 
  ∃ (r_large r_small : ℝ), 
    (4 / 3 * Real.pi * r_large ^ 3 = V_large) ∧ 
    (4 / 3 * Real.pi * r_small ^ 3 = V_small) ∧
    (r_small / r_large = Real.rpow 1.8 (1/3) / 2) ∧
    (V_large + V_small = 496.8 * Real.pi) := by
  sorry

end sphere_volume_and_radius_ratio_l3690_369081


namespace total_watching_time_l3690_369063

def first_show_length : ℕ := 30
def second_show_multiplier : ℕ := 4

theorem total_watching_time :
  first_show_length + first_show_length * second_show_multiplier = 150 :=
by sorry

end total_watching_time_l3690_369063


namespace ellipse_hyperbola_product_l3690_369020

theorem ellipse_hyperbola_product (A B : ℝ) 
  (h1 : B^2 - A^2 = 25)
  (h2 : A^2 + B^2 = 64) :
  |A * B| = Real.sqrt 867.75 := by
  sorry

end ellipse_hyperbola_product_l3690_369020


namespace constant_distance_l3690_369089

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope 1/2 and y-intercept m -/
structure Line where
  m : ℝ

/-- Theorem stating the constant distance between points B and N -/
theorem constant_distance (E : Ellipse) (l : Line) : 
  E.a^2 * (1 / E.b^2 - 1 / E.a^2) = 3 / 4 →  -- eccentricity condition
  E.b = 1 →  -- passes through (0, 1)
  ∃ (A C : ℝ × ℝ), 
    (A.1^2 / E.a^2 + A.2^2 / E.b^2 = 1) ∧  -- A is on the ellipse
    (C.1^2 / E.a^2 + C.2^2 / E.b^2 = 1) ∧  -- C is on the ellipse
    (A.2 = A.1 / 2 + l.m) ∧  -- A is on the line
    (C.2 = C.1 / 2 + l.m) ∧  -- C is on the line
    ∃ (B D : ℝ × ℝ), 
      (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 ∧  -- ABCD is a square
      (D.1 - A.1)^2 + (D.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 ∧
      (B.1 - C.1)^2 + (B.2 - C.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
      (B.1 - 2 * l.m)^2 + B.2^2 = 5 / 2  -- distance between B and N is √(5/2)
  := by sorry

end constant_distance_l3690_369089


namespace abs_neg_three_fourths_l3690_369092

theorem abs_neg_three_fourths : |(-3 : ℚ) / 4| = 3 / 4 := by
  sorry

end abs_neg_three_fourths_l3690_369092


namespace surrounded_pentagon_n_gons_l3690_369094

/-- The number of sides of the central polygon -/
def m : ℕ := 5

/-- The number of surrounding polygons -/
def num_surrounding : ℕ := 5

/-- The interior angle of a regular polygon with k sides -/
def interior_angle (k : ℕ) : ℚ :=
  (k - 2 : ℚ) * 180 / k

/-- The exterior angle of a regular polygon with k sides -/
def exterior_angle (k : ℕ) : ℚ :=
  180 - interior_angle k

/-- Theorem stating that for a regular pentagon surrounded by 5 regular n-gons
    with no overlap and no gaps, n must equal 5 -/
theorem surrounded_pentagon_n_gons :
  ∃ (n : ℕ), n > 2 ∧ 
  exterior_angle m = 360 / n ∧
  num_surrounding * (360 / n) = 360 := by
  sorry

end surrounded_pentagon_n_gons_l3690_369094


namespace calculation_proof_equation_solution_l3690_369085

-- Problem 1
theorem calculation_proof : -1^2024 + |(-3)| - (Real.pi + 1)^0 = 1 := by sorry

-- Problem 2
theorem equation_solution : 
  ∃ x : ℝ, (2 / (x + 2) = 4 / (x^2 - 4)) ∧ (x = 4) := by sorry

end calculation_proof_equation_solution_l3690_369085


namespace six_people_arrangement_l3690_369030

theorem six_people_arrangement (n : ℕ) (h : n = 6) : 
  (2 : ℕ) * (2 : ℕ) * (Nat.factorial 4) = 96 :=
sorry

end six_people_arrangement_l3690_369030


namespace sum_of_numbers_in_ratio_l3690_369064

theorem sum_of_numbers_in_ratio (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (a : ℚ) / b = 5 →
  (c : ℚ) / b = 4 →
  c = 400 →
  a + b + c = 1000 := by
sorry

end sum_of_numbers_in_ratio_l3690_369064


namespace plan1_more_cost_effective_l3690_369051

/-- Represents the cost of a mobile phone plan based on talk time -/
def plan_cost (rental : ℝ) (rate : ℝ) (minutes : ℝ) : ℝ := rental + rate * minutes

/-- Theorem stating when Plan 1 is more cost-effective than Plan 2 -/
theorem plan1_more_cost_effective (minutes : ℝ) :
  minutes > 72 →
  plan_cost 36 0.1 minutes < plan_cost 0 0.6 minutes := by
  sorry

end plan1_more_cost_effective_l3690_369051


namespace triangle_tangent_slopes_sum_l3690_369022

theorem triangle_tangent_slopes_sum (A B C : ℝ × ℝ) : 
  let triangle_slopes : List ℝ := [63, 73, 97]
  let curve (x : ℝ) := (x + 3) * (x^2 + 3)
  let tangent_slope (x : ℝ) := 3 * x^2 + 6 * x + 3
  (∀ p ∈ [A, B, C], p.1 ≥ 0 ∧ p.2 ≥ 0) →
  (∀ p ∈ [A, B, C], curve p.1 = p.2) →
  (List.zip [A, B, C] (A :: B :: C :: A :: nil)).all 
    (λ (p, q) => (q.2 - p.2) / (q.1 - p.1) ∈ triangle_slopes) →
  (tangent_slope A.1 + tangent_slope B.1 + tangent_slope C.1 = 237) :=
by
  sorry

end triangle_tangent_slopes_sum_l3690_369022


namespace positive_integers_sum_product_l3690_369048

theorem positive_integers_sum_product (P Q : ℕ+) (h : P + Q + P * Q = 90) : P + Q = 18 := by
  sorry

end positive_integers_sum_product_l3690_369048


namespace tangent_line_at_2_monotonicity_intervals_l3690_369084

/-- The function f(x) = 3x - x^3 -/
def f (x : ℝ) : ℝ := 3 * x - x^3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 - 3 * x^2

theorem tangent_line_at_2 :
  ∃ (m b : ℝ), m = -9 ∧ b = 16 ∧
  ∀ x, f x = m * (x - 2) + f 2 + b - f 2 :=
sorry

theorem monotonicity_intervals :
  (∀ x y, x < y ∧ y < -1 → f y < f x) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) :=
sorry

end tangent_line_at_2_monotonicity_intervals_l3690_369084


namespace possible_values_of_a_l3690_369079

theorem possible_values_of_a (a b c d : ℕ) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 3010)
  (h3 : a^2 - b^2 + c^2 - d^2 = 3010) :
  ∃! (s : Finset ℕ), s.card = 751 ∧ ∀ x, x ∈ s ↔ 
    ∃ (b' c' d' : ℕ), 
      a = x ∧
      x > b' ∧ b' > c' ∧ c' > d' ∧
      x + b' + c' + d' = 3010 ∧
      x^2 - b'^2 + c'^2 - d'^2 = 3010 :=
sorry

end possible_values_of_a_l3690_369079


namespace angle_bisector_sum_geq_nine_times_inradius_l3690_369049

/-- A triangle with an incircle -/
structure TriangleWithIncircle where
  /-- The radius of the incircle -/
  r : ℝ
  /-- The first angle bisector -/
  f_a : ℝ
  /-- The second angle bisector -/
  f_b : ℝ
  /-- The third angle bisector -/
  f_c : ℝ
  /-- Assumption that r is positive -/
  r_pos : r > 0
  /-- Assumption that angle bisectors are positive -/
  f_a_pos : f_a > 0
  f_b_pos : f_b > 0
  f_c_pos : f_c > 0

/-- The sum of angle bisectors is greater than or equal to 9 times the incircle radius -/
theorem angle_bisector_sum_geq_nine_times_inradius (t : TriangleWithIncircle) :
  t.f_a + t.f_b + t.f_c ≥ 9 * t.r :=
sorry

end angle_bisector_sum_geq_nine_times_inradius_l3690_369049


namespace number_of_reactions_l3690_369074

def visible_readings : List ℝ := [2, 2.1, 2, 2.2]

theorem number_of_reactions (x : ℝ) (h1 : (visible_readings.sum + x) / (visible_readings.length + 1) = 2) :
  visible_readings.length + 1 = 5 :=
sorry

end number_of_reactions_l3690_369074


namespace arithmetic_mean_of_fractions_l3690_369010

theorem arithmetic_mean_of_fractions (x a b : ℝ) (hx : x ≠ 0) (hb : b ≠ 0) :
  (((x + a + b) / x + (x - a - b) / x) / 2) = 1 := by
  sorry

end arithmetic_mean_of_fractions_l3690_369010


namespace percentage_difference_l3690_369068

theorem percentage_difference (A B : ℝ) (h : A = B * (1 + 0.25)) :
  B = A * (1 - 0.2) := by
  sorry

end percentage_difference_l3690_369068


namespace quadratic_properties_l3690_369047

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 - 4

-- Theorem stating the properties of the function
theorem quadratic_properties :
  ∃ a : ℝ, 
    (f a 0 = -3) ∧ 
    (∀ x, f 1 x = (x - 1)^2 - 4) ∧
    (∀ x > 1, ∀ y > x, f 1 y > f 1 x) ∧
    (f 1 (-1) = 0 ∧ f 1 3 = 0) ∧
    (∀ x, f 1 x = 0 → x = -1 ∨ x = 3) :=
by sorry

end quadratic_properties_l3690_369047


namespace bill_donuts_l3690_369027

theorem bill_donuts (total : ℕ) (secretary_takes : ℕ) (final : ℕ) : 
  total = 50 →
  secretary_takes = 4 →
  final = 22 →
  final * 2 = total - secretary_takes - (total - secretary_takes - final * 2) :=
by sorry

end bill_donuts_l3690_369027


namespace opposite_of_negative_two_thirds_l3690_369009

theorem opposite_of_negative_two_thirds :
  -(-(2/3 : ℚ)) = 2/3 := by sorry

end opposite_of_negative_two_thirds_l3690_369009


namespace machine_value_after_two_years_l3690_369015

-- Define the initial purchase price
def initialPrice : ℝ := 8000

-- Define the depreciation rate (20% = 0.20)
def depreciationRate : ℝ := 0.20

-- Define the time period in years
def timePeriod : ℕ := 2

-- Function to calculate the market value after a given number of years
def marketValue (years : ℕ) : ℝ :=
  initialPrice * (1 - depreciationRate) ^ years

-- Theorem statement
theorem machine_value_after_two_years :
  marketValue timePeriod = 5120 := by
  sorry


end machine_value_after_two_years_l3690_369015


namespace max_sum_with_reciprocals_l3690_369061

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b + 1/a + 1/b = 5 → x + y ≥ a + b :=
by sorry

end max_sum_with_reciprocals_l3690_369061


namespace division_value_proof_l3690_369032

theorem division_value_proof (x : ℝ) : (9 / x) * 12 = 18 → x = 6 := by
  sorry

end division_value_proof_l3690_369032


namespace jellybeans_per_child_l3690_369057

theorem jellybeans_per_child 
  (initial_jellybeans : ℕ) 
  (normal_class_size : ℕ) 
  (absent_children : ℕ) 
  (remaining_jellybeans : ℕ) 
  (h1 : initial_jellybeans = 100)
  (h2 : normal_class_size = 24)
  (h3 : absent_children = 2)
  (h4 : remaining_jellybeans = 34)
  : (initial_jellybeans - remaining_jellybeans) / (normal_class_size - absent_children) = 3 := by
  sorry

end jellybeans_per_child_l3690_369057


namespace triangle_side_length_l3690_369016

theorem triangle_side_length (a b : ℝ) (A B C : ℝ) :
  b = 2 →
  B = 30 * (π / 180) →
  C = 135 * (π / 180) →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a = Real.sqrt 6 - Real.sqrt 2 := by
  sorry

end triangle_side_length_l3690_369016


namespace floor_ceil_sum_equation_l3690_369086

theorem floor_ceil_sum_equation : ∃ (r s : ℝ), 
  (Int.floor r : ℝ) + r + (Int.ceil s : ℝ) = 10.7 ∧ r = 4.7 ∧ s = 2 := by
  sorry

end floor_ceil_sum_equation_l3690_369086


namespace y_decreasing_order_l3690_369004

-- Define the linear function
def f (x : ℝ) (b : ℝ) : ℝ := -2 * x + b

-- Define the theorem
theorem y_decreasing_order (b : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-2) b = y₁)
  (h₂ : f (-1) b = y₂)
  (h₃ : f 1 b = y₃) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end y_decreasing_order_l3690_369004


namespace ellipse_standard_equation_l3690_369083

/-- Represents an ellipse in a 2D Cartesian coordinate system -/
structure Ellipse where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  passing_point : ℝ × ℝ

/-- The standard equation of an ellipse -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Theorem: Given an ellipse with specific properties, prove its standard equation -/
theorem ellipse_standard_equation (e : Ellipse) 
  (h_center : e.center = (0, 0))
  (h_left_focus : e.left_focus = (-Real.sqrt 3, 0))
  (h_passing_point : e.passing_point = (2, 0)) :
  ∀ x y : ℝ, standard_equation 4 1 x y :=
by sorry

end ellipse_standard_equation_l3690_369083


namespace least_addition_for_divisibility_l3690_369008

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 1056 ∧ m = 27) :
  ∃ x : ℕ, (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 :=
by sorry

end least_addition_for_divisibility_l3690_369008


namespace peanut_butter_servings_l3690_369093

/-- Represents the amount of peanut butter in tablespoons -/
def peanut_butter : ℚ := 29 + 5 / 7

/-- Represents the size of one serving in tablespoons -/
def serving_size : ℚ := 2

/-- Represents the number of servings in the jar -/
def num_servings : ℚ := peanut_butter / serving_size

/-- Theorem stating that the number of servings in the jar is 14 3/7 -/
theorem peanut_butter_servings : num_servings = 14 + 3 / 7 := by
  sorry

end peanut_butter_servings_l3690_369093


namespace quadratic_root_property_l3690_369043

theorem quadratic_root_property (m : ℝ) (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 4 * x₁ + m = 0) →
  (2 * x₂^2 + 4 * x₂ + m = 0) →
  (x₁^2 + x₂^2 + 2*x₁*x₂ - x₁^2*x₂^2 = 0) →
  m = -4 := by
sorry

end quadratic_root_property_l3690_369043


namespace hexagon_area_l3690_369045

/-- A regular hexagon with vertices P and R -/
structure RegularHexagon where
  P : ℝ × ℝ
  R : ℝ × ℝ

/-- The area of a regular hexagon -/
def area (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The area of a regular hexagon with P at (0,0) and R at (10,2) is 156√3 -/
theorem hexagon_area :
  let h : RegularHexagon := { P := (0, 0), R := (10, 2) }
  area h = 156 * Real.sqrt 3 := by sorry

end hexagon_area_l3690_369045


namespace complex_power_24_l3690_369001

theorem complex_power_24 : (((1 - Complex.I) / Real.sqrt 2) ^ 24 : ℂ) = 1 := by sorry

end complex_power_24_l3690_369001


namespace equations_not_equivalent_l3690_369062

theorem equations_not_equivalent : 
  ¬(∀ x : ℝ, (Real.sqrt (x^2 + x - 5) = Real.sqrt (x - 1)) ↔ (x^2 + x - 5 = x - 1)) :=
by sorry

end equations_not_equivalent_l3690_369062


namespace cube_sphere_volume_l3690_369070

theorem cube_sphere_volume (n : ℕ) (hn : n > 2) : 
  (n^3 : ℝ) - (4/3 * Real.pi * (n/2)^3) = 2 * (4/3 * Real.pi * (n/2)^3) → n = 8 := by
  sorry

end cube_sphere_volume_l3690_369070


namespace second_monday_watching_time_l3690_369024

def day_hours : ℕ := 24

def monday_hours : ℕ := day_hours / 2
def tuesday_hours : ℕ := 4
def wednesday_hours : ℕ := day_hours / 4
def thursday_hours : ℕ := day_hours / 3
def friday_hours : ℕ := 2 * wednesday_hours
def saturday_hours : ℕ := 0

def week_total : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours + saturday_hours

def sunday_hours : ℕ := week_total / 2

def total_watched : ℕ := week_total + sunday_hours

def show_length : ℕ := 75

theorem second_monday_watching_time :
  show_length - total_watched = 12 := by sorry

end second_monday_watching_time_l3690_369024


namespace dividing_chord_length_l3690_369082

/-- An octagon inscribed in a circle -/
structure InscribedOctagon :=
  (side_length_1 : ℝ)
  (side_length_2 : ℝ)
  (h1 : side_length_1 > 0)
  (h2 : side_length_2 > 0)

/-- The chord dividing the octagon into two quadrilaterals -/
def dividing_chord (o : InscribedOctagon) : ℝ := sorry

/-- Theorem stating the length of the dividing chord -/
theorem dividing_chord_length (o : InscribedOctagon) 
  (h3 : o.side_length_1 = 4)
  (h4 : o.side_length_2 = 6) : 
  dividing_chord o = 4 := by sorry

end dividing_chord_length_l3690_369082


namespace deepak_age_l3690_369090

/-- Given the ratio of Arun's age to Deepak's age and Arun's future age, prove Deepak's current age -/
theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 2 / 5 →
  arun_age + 10 = 30 →
  deepak_age = 50 := by
sorry

end deepak_age_l3690_369090


namespace inverse_of_singular_matrix_l3690_369000

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; 10, 6]

theorem inverse_of_singular_matrix :
  Matrix.det A = 0 → A⁻¹ = !![0, 0; 0, 0] := by
  sorry

end inverse_of_singular_matrix_l3690_369000


namespace rectangular_plot_length_l3690_369036

/-- Given a rectangular plot with the following properties:
  - The length is 32 meters more than the breadth
  - The cost of fencing at 26.50 per meter is Rs. 5300
  Prove that the length of the plot is 66 meters. -/
theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) :
  length = breadth + 32 →
  perimeter = 2 * (length + breadth) →
  perimeter * 26.5 = 5300 →
  length = 66 := by
  sorry


end rectangular_plot_length_l3690_369036


namespace max_diff_color_pairs_l3690_369012

/-- Represents a grid of black and white cells -/
structure Grid :=
  (size : Nat)
  (black_cells : Fin size → Fin size → Bool)

/-- The number of black cells in a given row -/
def row_black_count (g : Grid) (row : Fin g.size) : Nat :=
  (List.range g.size).count (λ col ↦ g.black_cells row col)

/-- The number of black cells in a given column -/
def col_black_count (g : Grid) (col : Fin g.size) : Nat :=
  (List.range g.size).count (λ row ↦ g.black_cells row col)

/-- The number of pairs of adjacent differently colored cells -/
def diff_color_pairs (g : Grid) : Nat :=
  sorry

/-- The theorem statement -/
theorem max_diff_color_pairs :
  ∃ (g : Grid),
    g.size = 100 ∧
    (∀ col₁ col₂ : Fin g.size, col_black_count g col₁ = col_black_count g col₂) ∧
    (∀ row₁ row₂ : Fin g.size, row₁ ≠ row₂ → row_black_count g row₁ ≠ row_black_count g row₂) ∧
    (∀ g' : Grid,
      g'.size = 100 →
      (∀ col₁ col₂ : Fin g'.size, col_black_count g' col₁ = col_black_count g' col₂) →
      (∀ row₁ row₂ : Fin g'.size, row₁ ≠ row₂ → row_black_count g' row₁ ≠ row_black_count g' row₂) →
      diff_color_pairs g' ≤ diff_color_pairs g) ∧
    diff_color_pairs g = 14601 :=
  sorry

end max_diff_color_pairs_l3690_369012


namespace total_keys_needed_l3690_369087

theorem total_keys_needed 
  (num_complexes : ℕ) 
  (apartments_per_complex : ℕ) 
  (keys_per_apartment : ℕ) 
  (h1 : num_complexes = 2) 
  (h2 : apartments_per_complex = 12) 
  (h3 : keys_per_apartment = 3) : 
  num_complexes * apartments_per_complex * keys_per_apartment = 72 := by
sorry

end total_keys_needed_l3690_369087


namespace problem_solution_l3690_369097

theorem problem_solution : 
  ((-1)^2023 - Real.sqrt (2 + 1/4) + ((-1 : ℝ)^(1/3 : ℝ)) + 1/2 = -3) ∧ 
  (2 * Real.sqrt 3 + |1 - Real.sqrt 3| - (-1)^2022 + 2 = 3 * Real.sqrt 3) := by
  sorry

end problem_solution_l3690_369097


namespace final_numbers_l3690_369011

def process (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ (i j : ℕ), i ≤ n ∧ j ≤ n ∧ m = i * j}

theorem final_numbers (n : ℕ) :
  process n = {m : ℕ | ∃ (k : ℕ), k ≤ n ∧ m = k^2} :=
by sorry

#check final_numbers 2009

end final_numbers_l3690_369011


namespace no_solution_for_absolute_value_equation_l3690_369041

theorem no_solution_for_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 4| = x^2 + 6*x + 8 := by
sorry

end no_solution_for_absolute_value_equation_l3690_369041
