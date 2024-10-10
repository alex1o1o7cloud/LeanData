import Mathlib

namespace hexagon_area_theorem_l3461_346167

-- Define the right triangle
structure RightTriangle where
  a : ℝ  -- One leg
  b : ℝ  -- Other leg
  c : ℝ  -- Hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Define the hexagon area function
def hexagon_area (t : RightTriangle) : ℝ :=
  t.a^2 + (t.a + t.b)^2

-- Theorem statement
theorem hexagon_area_theorem (t : RightTriangle) :
  hexagon_area t = t.a^2 + (t.a + t.b)^2 := by
  sorry

#check hexagon_area_theorem

end hexagon_area_theorem_l3461_346167


namespace codemaster_secret_codes_l3461_346126

/-- The number of slots in a CodeMaster secret code -/
def num_slots : ℕ := 5

/-- The total number of colors available -/
def total_colors : ℕ := 8

/-- The number of colors available for the last three slots (excluding black) -/
def colors_without_black : ℕ := 7

/-- The number of slots where black is allowed -/
def black_allowed_slots : ℕ := 2

/-- The number of different secret codes possible in the CodeMaster game -/
def num_secret_codes : ℕ := total_colors ^ black_allowed_slots * colors_without_black ^ (num_slots - black_allowed_slots)

theorem codemaster_secret_codes :
  num_secret_codes = 21952 := by
  sorry

end codemaster_secret_codes_l3461_346126


namespace complex_sum_l3461_346130

def alphabet_value (n : Nat) : Int :=
  match n % 26 with
  | 0 => 3   -- Z
  | 1 => 2   -- A
  | 2 => 3   -- B
  | 3 => 2   -- C
  | 4 => 1   -- D
  | 5 => 0   -- E
  | 6 => -1  -- F
  | 7 => -2  -- G
  | 8 => -3  -- H
  | 9 => -2  -- I
  | 10 => -1 -- J
  | 11 => 0  -- K
  | 12 => 1  -- L
  | 13 => 2  -- M
  | 14 => 3  -- N
  | 15 => 2  -- O
  | 16 => 1  -- P
  | 17 => 0  -- Q
  | 18 => -1 -- R
  | 19 => -2 -- S
  | 20 => -3 -- T
  | 21 => -2 -- U
  | 22 => -1 -- V
  | 23 => 0  -- W
  | 24 => 1  -- X
  | 25 => 2  -- Y
  | _ => 0   -- This case should never occur due to the modulo operation

def letter_to_position (c : Char) : Nat :=
  (c.toUpper.toNat - 'A'.toNat) + 1

theorem complex_sum : 
  (alphabet_value (letter_to_position 'c') +
   alphabet_value (letter_to_position 'o') +
   alphabet_value (letter_to_position 'm') +
   alphabet_value (letter_to_position 'p') +
   alphabet_value (letter_to_position 'l') +
   alphabet_value (letter_to_position 'e') +
   alphabet_value (letter_to_position 'x')) = 9 := by
  sorry


end complex_sum_l3461_346130


namespace equal_angles_same_terminal_side_l3461_346155

-- Define the angle type
def Angle : Type := ℝ

-- Define the terminal side of an angle
def terminalSide (α : Angle) : Set (ℝ × ℝ) := sorry

-- Theorem: If two angles are equal, they have the same terminal side
theorem equal_angles_same_terminal_side (α β : Angle) :
  α = β → terminalSide α = terminalSide β := by sorry

end equal_angles_same_terminal_side_l3461_346155


namespace select_one_from_each_l3461_346131

theorem select_one_from_each : ∀ (n m : ℕ), n = 5 → m = 4 → n * m = 20 := by
  sorry

end select_one_from_each_l3461_346131


namespace harry_scores_l3461_346188

/-- Harry's basketball scores -/
def first_10_games : List ℕ := [9, 5, 4, 7, 11, 4, 2, 8, 5, 7]

/-- Sum of scores in the first 10 games -/
def sum_first_10 : ℕ := first_10_games.sum

/-- Proposition: Harry's 11th and 12th game scores -/
theorem harry_scores : ∃ (score_11 score_12 : ℕ),
  (score_11 < 15 ∧ score_12 < 15) ∧
  (sum_first_10 + score_11) % 11 = 0 ∧
  (sum_first_10 + score_11 + score_12) % 12 = 0 ∧
  score_11 * score_12 = 24 := by
  sorry

end harry_scores_l3461_346188


namespace tenth_house_gnomes_l3461_346122

/-- Represents the number of gnomes in each house on the street. -/
structure GnomeCounts where
  house1 : Nat
  house2 : Nat
  house3 : Nat
  house4 : Nat
  house5 : Nat
  house6 : Nat
  house7 : Nat
  house8 : Nat
  house9 : Nat

/-- The theorem stating that the tenth house must have 3 gnomes. -/
theorem tenth_house_gnomes (g : GnomeCounts) : 
  g.house1 = 4 ∧
  g.house2 = 2 * g.house1 ∧
  g.house3 = g.house2 - 3 ∧
  g.house4 = g.house1 + g.house3 ∧
  g.house5 = 5 ∧
  g.house6 = 2 ∧
  g.house7 = 7 ∧
  g.house8 = g.house4 + 3 ∧
  g.house9 = 10 →
  65 - (g.house1 + g.house2 + g.house3 + g.house4 + g.house5 + g.house6 + g.house7 + g.house8 + g.house9) = 3 := by
  sorry

end tenth_house_gnomes_l3461_346122


namespace repeated_sequence_result_l3461_346103

/-- Represents one cycle of operations -/
def cycle_increment : Int := 15 - 12 + 3

/-- Calculates the number of complete cycles in n steps -/
def complete_cycles (n : Nat) : Nat := n / 3

/-- Calculates the number of remaining steps after complete cycles -/
def remaining_steps (n : Nat) : Nat := n % 3

/-- Calculates the increment from remaining steps -/
def remaining_increment (steps : Nat) : Int :=
  if steps = 1 then 15
  else if steps = 2 then 15 - 12
  else 0

/-- Theorem stating the result of the repeated operation sequence -/
theorem repeated_sequence_result :
  let initial_value : Int := 100
  let total_steps : Nat := 26
  let cycles : Nat := complete_cycles total_steps
  let remaining : Nat := remaining_steps total_steps
  initial_value + cycles * cycle_increment + remaining_increment remaining = 151 := by
  sorry

end repeated_sequence_result_l3461_346103


namespace susan_reading_hours_l3461_346173

/-- Represents the number of hours spent on each activity -/
structure ActivityHours where
  swimming : ℝ
  reading : ℝ
  friends : ℝ
  work : ℝ
  chores : ℝ

/-- The ratio of time spent on activities -/
def activity_ratio : ActivityHours :=
  { swimming := 1
    reading := 4
    friends := 10
    work := 3
    chores := 2 }

/-- Susan's actual hours spent on activities -/
def susan_hours : ActivityHours :=
  { swimming := 2
    reading := 8
    friends := 20
    work := 6
    chores := 4 }

theorem susan_reading_hours :
  (∀ (x : ℝ), x > 0 →
    susan_hours.swimming = x * activity_ratio.swimming ∧
    susan_hours.reading = x * activity_ratio.reading ∧
    susan_hours.friends = x * activity_ratio.friends ∧
    susan_hours.work = x * activity_ratio.work ∧
    susan_hours.chores = x * activity_ratio.chores) →
  susan_hours.friends = 20 →
  susan_hours.work + susan_hours.chores ≤ 35 →
  susan_hours.reading = 8 :=
by sorry

end susan_reading_hours_l3461_346173


namespace cookie_count_l3461_346152

theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) (total_cookies : ℕ) : 
  bags = 37 → cookies_per_bag = 19 → total_cookies = bags * cookies_per_bag → total_cookies = 703 := by
  sorry

end cookie_count_l3461_346152


namespace semicircle_perimeter_l3461_346128

/-- The perimeter of a semicircle with radius 8 is 16 + 8π -/
theorem semicircle_perimeter : 
  ∀ (r : ℝ), r = 8 → 2 * r + (π * r) = 16 + 8 * π :=
by
  sorry

end semicircle_perimeter_l3461_346128


namespace problem_statement_l3461_346171

theorem problem_statement (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 31 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (h1 : a ≠ 17)
  (h2 : x ≠ 0) :
  a / (a - 17) + b / (b - 31) + c / (c - 53) = 1 := by
  sorry

end problem_statement_l3461_346171


namespace vector_magnitude_l3461_346157

/-- Given vectors a and b in ℝ², if a is collinear with a + b, 
    then the magnitude of a - b is 2√5 -/
theorem vector_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![3, x]
  (∃ (k : ℝ), a = k • (a + b)) → ‖a - b‖ = 2 * Real.sqrt 5 := by
  sorry

end vector_magnitude_l3461_346157


namespace least_tiles_for_room_l3461_346186

theorem least_tiles_for_room (room_length room_width : ℕ) : 
  room_length = 7550 → room_width = 2085 → 
  (∃ (tile_size : ℕ), 
    tile_size > 0 ∧ 
    room_length % tile_size = 0 ∧ 
    room_width % tile_size = 0 ∧
    (∀ (larger_tile : ℕ), larger_tile > tile_size → 
      room_length % larger_tile ≠ 0 ∨ room_width % larger_tile ≠ 0) ∧
    (room_length * room_width) / (tile_size * tile_size) = 630270) :=
by sorry

end least_tiles_for_room_l3461_346186


namespace no_perfect_square_in_range_l3461_346102

theorem no_perfect_square_in_range : 
  ¬ ∃ (n : ℕ), 4 ≤ n ∧ n ≤ 13 ∧ ∃ (m : ℕ), n^2 + n + 1 = m^2 := by
  sorry

end no_perfect_square_in_range_l3461_346102


namespace consecutive_multiples_of_twelve_l3461_346110

theorem consecutive_multiples_of_twelve (A B : ℕ) (h1 : Nat.gcd A B = 12) (h2 : A > B) (h3 : A - B = 12) :
  ∃ (m : ℕ), A = 12 * (m + 1) ∧ B = 12 * m :=
sorry

end consecutive_multiples_of_twelve_l3461_346110


namespace reciprocal_expression_l3461_346158

theorem reciprocal_expression (a b c : ℝ) (h : a * b = 1) : a * b * c - (c - 2023) = 2023 := by
  sorry

end reciprocal_expression_l3461_346158


namespace postage_for_5_5_ounces_l3461_346101

/-- Calculates the postage for a letter given its weight and the rate structure -/
def calculatePostage (weight : ℚ) : ℚ :=
  let baseRate : ℚ := 25 / 100  -- 25 cents
  let additionalRate : ℚ := 18 / 100  -- 18 cents
  let overweightSurcharge : ℚ := 10 / 100  -- 10 cents
  let overweightThreshold : ℚ := 3  -- 3 ounces
  
  let additionalWeight := max (weight - 1) 0
  let additionalCharges := ⌈additionalWeight⌉
  
  let cost := baseRate + additionalRate * additionalCharges
  if weight > overweightThreshold then
    cost + overweightSurcharge
  else
    cost

/-- Theorem stating that the postage for a 5.5 ounce letter is $1.25 -/
theorem postage_for_5_5_ounces :
  calculatePostage (11/2) = 5/4 := by sorry

end postage_for_5_5_ounces_l3461_346101


namespace fifteen_percent_problem_l3461_346154

theorem fifteen_percent_problem (x : ℝ) (h : (15 / 100) * x = 60) : x = 400 := by
  sorry

end fifteen_percent_problem_l3461_346154


namespace chinese_learning_hours_l3461_346164

theorem chinese_learning_hours 
  (total_days : ℕ) 
  (total_chinese_hours : ℕ) 
  (h1 : total_days = 6) 
  (h2 : total_chinese_hours = 24) : 
  total_chinese_hours / total_days = 4 := by
sorry

end chinese_learning_hours_l3461_346164


namespace tan_equality_solutions_l3461_346196

theorem tan_equality_solutions (n : ℤ) :
  -150 < n ∧ n < 150 ∧ Real.tan (n * π / 180) = Real.tan (225 * π / 180) →
  n = -135 ∨ n = -45 ∨ n = 45 ∨ n = 135 := by
  sorry

end tan_equality_solutions_l3461_346196


namespace factorization_proof_l3461_346191

theorem factorization_proof (x : ℝ) : 
  3 * x^2 * (x - 5) + 4 * x * (x - 5) + 6 * (x - 5) = (3 * x^2 + 4 * x + 6) * (x - 5) := by
  sorry

end factorization_proof_l3461_346191


namespace fixed_salary_is_1000_l3461_346159

/-- Calculates the commission for the old scheme -/
def old_commission (sales : ℝ) : ℝ := 0.05 * sales

/-- Calculates the commission for the new scheme -/
def new_commission (sales : ℝ) : ℝ := 0.025 * (sales - 4000)

/-- Theorem: The fixed salary in the new scheme is 1000 -/
theorem fixed_salary_is_1000 (total_sales : ℝ) (fixed_salary : ℝ) :
  total_sales = 12000 →
  fixed_salary + new_commission total_sales = old_commission total_sales + 600 →
  fixed_salary = 1000 := by
  sorry

#check fixed_salary_is_1000

end fixed_salary_is_1000_l3461_346159


namespace diophantine_equation_properties_l3461_346144

theorem diophantine_equation_properties :
  (∃ (x y z : ℕ+), 28 * x + 30 * y + 31 * z = 365) ∧
  (∀ (n : ℕ+), n > 370 → ∃ (x y z : ℕ+), 28 * x + 30 * y + 31 * z = n) ∧
  (¬ ∃ (x y z : ℕ+), 28 * x + 30 * y + 31 * z = 370) := by
  sorry

end diophantine_equation_properties_l3461_346144


namespace parabola_triangle_area_l3461_346182

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = 4x -/
def OnParabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- The focus of the parabola y^2 = 4x -/
def FocusOfParabola : Point :=
  ⟨1, 0⟩

/-- The orthocenter of a triangle -/
def Orthocenter (a b c : Point) : Point :=
  sorry  -- Definition of orthocenter

/-- The area of a triangle -/
def TriangleArea (a b c : Point) : ℝ :=
  sorry  -- Definition of triangle area

/-- The main theorem -/
theorem parabola_triangle_area :
  ∀ (A B : Point),
    OnParabola A →
    OnParabola B →
    Orthocenter ⟨0, 0⟩ A B = FocusOfParabola →
    TriangleArea ⟨0, 0⟩ A B = 10 * Real.sqrt 5 :=
by
  sorry

end parabola_triangle_area_l3461_346182


namespace system_inequalities_solution_l3461_346133

theorem system_inequalities_solution (a b : ℝ) : 
  (∀ x : ℝ, (x + a - 2 > 0 ∧ 2*x - b - 1 < 0) ↔ (0 < x ∧ x < 1)) →
  (a = 2 ∧ b = 1) := by
sorry

end system_inequalities_solution_l3461_346133


namespace marbles_distribution_l3461_346181

theorem marbles_distribution (total_marbles : ℕ) (marble_loving_boys : ℕ) 
  (h1 : total_marbles = 26) (h2 : marble_loving_boys = 13) :
  total_marbles / marble_loving_boys = 2 := by
sorry

end marbles_distribution_l3461_346181


namespace min_value_abs_plus_one_l3461_346180

theorem min_value_abs_plus_one :
  (∀ x : ℝ, |x - 2| + 1 ≥ 1) ∧ (∃ x : ℝ, |x - 2| + 1 = 1) := by
  sorry

end min_value_abs_plus_one_l3461_346180


namespace kim_average_increase_l3461_346127

theorem kim_average_increase : 
  let scores : List ℝ := [85, 89, 90, 92, 95]
  let average4 := (scores.take 4).sum / 4
  let average5 := scores.sum / 5
  average5 - average4 = 1.2 := by
sorry

end kim_average_increase_l3461_346127


namespace janessas_cards_l3461_346194

/-- The number of cards Janessa's father gave her. -/
def fathers_cards : ℕ := by sorry

theorem janessas_cards :
  let initial_cards : ℕ := 4
  let ebay_cards : ℕ := 36
  let discarded_cards : ℕ := 4
  let cards_to_dexter : ℕ := 29
  let cards_kept : ℕ := 20
  fathers_cards = 13 := by sorry

end janessas_cards_l3461_346194


namespace total_taco_ingredients_cost_l3461_346192

def taco_shells_cost : ℝ := 5
def bell_peppers_cost : ℝ := 4 * 1.5
def meat_cost : ℝ := 2 * 3
def tomatoes_cost : ℝ := 3 * 0.75
def cheese_cost : ℝ := 4
def tortillas_cost : ℝ := 2.5
def salsa_cost : ℝ := 3.25

theorem total_taco_ingredients_cost :
  taco_shells_cost + bell_peppers_cost + meat_cost + tomatoes_cost + cheese_cost + tortillas_cost + salsa_cost = 29 := by
  sorry

end total_taco_ingredients_cost_l3461_346192


namespace walters_age_l3461_346183

theorem walters_age (walter_age_1994 : ℕ) (mother_age_1994 : ℕ) : 
  walter_age_1994 = mother_age_1994 / 3 →
  (1994 - walter_age_1994) + (1994 - mother_age_1994) = 3900 →
  walter_age_1994 + 10 = 32 :=
by sorry

end walters_age_l3461_346183


namespace marble_difference_l3461_346193

/-- The number of marbles each person has -/
structure Marbles where
  merill : ℕ
  elliot : ℕ
  selma : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : Marbles) : Prop :=
  m.merill = 30 ∧ m.selma = 50 ∧ m.merill = 2 * m.elliot

/-- The theorem to prove -/
theorem marble_difference (m : Marbles) (h : marble_problem m) :
  m.selma - (m.merill + m.elliot) = 5 := by
  sorry

end marble_difference_l3461_346193


namespace alloy_fourth_metal_mass_l3461_346137

/-- Given an alloy of four metals with a total mass of 20 kg, where:
    - The mass of the first metal is 1.5 times the mass of the second metal
    - The ratio of the mass of the second metal to the third metal is 3:4
    - The ratio of the mass of the third metal to the fourth metal is 5:6
    Prove that the mass of the fourth metal is 960/163 kg. -/
theorem alloy_fourth_metal_mass (m₁ m₂ m₃ m₄ : ℝ) 
  (h_total : m₁ + m₂ + m₃ + m₄ = 20)
  (h_first_second : m₁ = 1.5 * m₂)
  (h_second_third : m₂ / m₃ = 3 / 4)
  (h_third_fourth : m₃ / m₄ = 5 / 6) :
  m₄ = 960 / 163 := by
sorry

end alloy_fourth_metal_mass_l3461_346137


namespace prime_even_intersection_l3461_346185

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def P : Set ℕ := {n : ℕ | isPrime n}
def Q : Set ℕ := {n : ℕ | isEven n}

theorem prime_even_intersection : P ∩ Q = {2} := by
  sorry

end prime_even_intersection_l3461_346185


namespace mirror_wall_area_ratio_l3461_346148

theorem mirror_wall_area_ratio :
  let mirror_side : ℝ := 21
  let wall_width : ℝ := 28
  let wall_length : ℝ := 31.5
  let mirror_area := mirror_side ^ 2
  let wall_area := wall_width * wall_length
  mirror_area / wall_area = 1 / 2 := by
sorry

end mirror_wall_area_ratio_l3461_346148


namespace y_in_terms_of_abc_l3461_346172

theorem y_in_terms_of_abc (x y z a b c : ℝ) 
  (h1 : x ≠ y) (h2 : x ≠ z) (h3 : y ≠ z) 
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0)
  (eq1 : x * y / (x - y) = a)
  (eq2 : x * z / (x - z) = b)
  (eq3 : y * z / (y - z) = c) :
  y = b * c * x / ((b + c) * x - b * c) := by
  sorry

end y_in_terms_of_abc_l3461_346172


namespace work_completion_time_l3461_346160

/-- Given two workers A and B who complete a work together in a certain number of days,
    this function calculates the time it takes for them to complete the work together. -/
def time_to_complete (time_A : ℝ) (time_together : ℝ) : ℝ :=
  time_together

/-- Theorem stating that if A and B complete the work in 9 days together,
    and A alone can do the work in 18 days, then A and B together can complete
    the work in 9 days. -/
theorem work_completion_time (time_A : ℝ) (time_together : ℝ)
    (h1 : time_A = 18)
    (h2 : time_together = 9) :
    time_to_complete time_A time_together = 9 := by
  sorry

#check work_completion_time

end work_completion_time_l3461_346160


namespace horner_rule_v3_equals_18_horner_rule_correctness_main_theorem_l3461_346195

/-- Horner's Rule for a specific polynomial -/
def horner_v3 (x : ℝ) : ℝ := ((x + 3) * x - 1) * x

/-- The polynomial f(x) = x^5 + 3x^4 - x^3 + 2x - 1 -/
def f (x : ℝ) : ℝ := x^5 + 3*x^4 - x^3 + 2*x - 1

theorem horner_rule_v3_equals_18 :
  horner_v3 2 = 18 := by sorry

theorem horner_rule_correctness (x : ℝ) :
  horner_v3 x = ((x + 3) * x - 1) * x := by sorry

theorem main_theorem : f 2 = ((((2 + 3) * 2 - 1) * 2 + 2) * 2 - 1) := by sorry

end horner_rule_v3_equals_18_horner_rule_correctness_main_theorem_l3461_346195


namespace first_player_can_ensure_nonzero_solution_l3461_346107

/-- Represents a linear equation in three variables -/
structure LinearEquation where
  coeff_x : ℝ
  coeff_y : ℝ
  coeff_z : ℝ
  constant : ℝ

/-- Represents a system of three linear equations -/
structure LinearSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation
  eq3 : LinearEquation

/-- Represents a player's strategy for choosing coefficients -/
def Strategy := LinearSystem → LinearEquation → ℝ

/-- Checks if a given solution satisfies the linear system -/
def is_solution (system : LinearSystem) (x y z : ℝ) : Prop :=
  system.eq1.coeff_x * x + system.eq1.coeff_y * y + system.eq1.coeff_z * z = system.eq1.constant ∧
  system.eq2.coeff_x * x + system.eq2.coeff_y * y + system.eq2.coeff_z * z = system.eq2.constant ∧
  system.eq3.coeff_x * x + system.eq3.coeff_y * y + system.eq3.coeff_z * z = system.eq3.constant

/-- Represents the game where players choose coefficients -/
def play_game (player1_strategy : Strategy) (player2_strategy : Strategy) : LinearSystem :=
  sorry  -- Implementation of the game play

/-- The main theorem stating that the first player can ensure a nonzero solution -/
theorem first_player_can_ensure_nonzero_solution :
  ∃ (player1_strategy : Strategy),
    ∀ (player2_strategy : Strategy),
      ∃ (x y z : ℝ),
        (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
        is_solution (play_game player1_strategy player2_strategy) x y z :=
by sorry

end first_player_can_ensure_nonzero_solution_l3461_346107


namespace product_with_decimals_l3461_346147

theorem product_with_decimals (x y : ℚ) (z : ℕ) :
  x = 0.075 → y = 2.56 → z = 19200 →
  (↑75 : ℚ) * 256 = z →
  x * y = 0.192 := by
sorry

end product_with_decimals_l3461_346147


namespace angies_age_ratio_l3461_346189

theorem angies_age_ratio : 
  ∀ (A : ℕ), A + 4 = 20 → (A : ℚ) / 20 = 4 / 5 := by
  sorry

end angies_age_ratio_l3461_346189


namespace min_max_sum_f_l3461_346161

def f (x : ℝ) : ℝ := (x + 1)^5 + (x - 1)^5

theorem min_max_sum_f :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 0 2, f x ≥ min) ∧
    (∃ x ∈ Set.Icc 0 2, f x = min) ∧
    (∀ x ∈ Set.Icc 0 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 2, f x = max) ∧
    min = 0 ∧ max = 244 ∧ min + max = 244 := by
  sorry

end min_max_sum_f_l3461_346161


namespace radius_C1_value_l3461_346174

-- Define the points and circles
variable (O X Y Z : ℝ × ℝ)
variable (C1 C2 : Set (ℝ × ℝ))

-- Define the conditions
axiom inside_C2 : C1 ⊆ C2
axiom intersect : X ∈ C1 ∩ C2 ∧ Y ∈ C1 ∩ C2
axiom Z_position : Z ∉ C1 ∧ Z ∈ C2
axiom XZ_length : dist X Z = 15
axiom OZ_length : dist O Z = 5
axiom YZ_length : dist Y Z = 12

-- Define the radius of C1
def radius_C1 : ℝ := dist O X

-- Theorem to prove
theorem radius_C1_value : radius_C1 O X = 5 * Real.sqrt 10 := by
  sorry

end radius_C1_value_l3461_346174


namespace water_depth_is_60_l3461_346123

/-- The depth of water given Ron's height -/
def water_depth (ron_height : ℝ) : ℝ := 5 * ron_height

/-- Ron's height in feet -/
def ron_height : ℝ := 12

/-- Theorem: The water depth is 60 feet -/
theorem water_depth_is_60 : water_depth ron_height = 60 := by
  sorry

end water_depth_is_60_l3461_346123


namespace max_plane_division_prove_max_plane_division_l3461_346106

/-- The maximum number of parts the plane can be divided into by two specific parabolas and a line -/
theorem max_plane_division (b k : ℝ) : ℕ :=
  let parabola1 := fun x : ℝ => x^2 - b*x
  let parabola2 := fun x : ℝ => -x^2 + b*x
  let line := fun x : ℝ => k*x
  9

/-- Proof of the maximum number of plane divisions -/
theorem prove_max_plane_division (b k : ℝ) : 
  max_plane_division b k = 9 := by sorry

end max_plane_division_prove_max_plane_division_l3461_346106


namespace roots_of_quadratic_equation_l3461_346169

theorem roots_of_quadratic_equation :
  ∀ x : ℝ, x^2 = 2*x ↔ x = 0 ∨ x = 2 := by sorry

end roots_of_quadratic_equation_l3461_346169


namespace distance_between_points_on_curve_l3461_346163

theorem distance_between_points_on_curve (e a b : ℝ) : 
  e > 0 → 
  a ≠ b → 
  a^2 + e^2 = 3 * e * a + 1 → 
  b^2 + e^2 = 3 * e * b + 1 → 
  |a - b| = Real.sqrt (5 * e^2 + 4) := by
  sorry

end distance_between_points_on_curve_l3461_346163


namespace g_of_50_l3461_346119

/-- A function satisfying the given property for all positive real numbers -/
def SatisfyingFunction (g : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → x * g y - y * g x = g (x / y) + x - y

/-- The theorem stating that any function satisfying the property has g(50) = -24.5 -/
theorem g_of_50 (g : ℝ → ℝ) (h : SatisfyingFunction g) : g 50 = -24.5 := by
  sorry

end g_of_50_l3461_346119


namespace mn_positive_necessary_not_sufficient_l3461_346139

/-- Predicate defining when a curve is an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

/-- The equation of the curve -/
def curve_equation (m n : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

theorem mn_positive_necessary_not_sufficient (m n : ℝ) :
  (is_ellipse m n → m * n > 0) ∧
  ¬(m * n > 0 → is_ellipse m n) :=
sorry

end mn_positive_necessary_not_sufficient_l3461_346139


namespace parallel_vectors_imply_x_value_l3461_346197

def vector_a : ℝ × ℝ := (3, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (-12, x - 4)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_imply_x_value :
  ∀ x : ℝ, parallel vector_a (vector_b x) → x = -4 := by
  sorry

end parallel_vectors_imply_x_value_l3461_346197


namespace probability_of_red_ball_l3461_346166

def num_red_balls : ℕ := 7
def num_black_balls : ℕ := 3

def total_balls : ℕ := num_red_balls + num_black_balls

theorem probability_of_red_ball :
  (num_red_balls : ℚ) / (total_balls : ℚ) = 7 / 10 := by
  sorry

end probability_of_red_ball_l3461_346166


namespace remainder_problem_l3461_346142

theorem remainder_problem (n : ℕ) (h : 2 * n % 4 = 2) : n % 4 = 1 := by
  sorry

end remainder_problem_l3461_346142


namespace no_meeting_l3461_346168

/-- Represents the positions of Michael and the truck over time -/
structure Positions (t : ℝ) where
  michael : ℝ
  truck : ℝ

/-- The rate at which Michael walks -/
def michael_speed : ℝ := 6

/-- The speed of the garbage truck -/
def truck_speed : ℝ := 12

/-- The distance between trash pails -/
def pail_distance : ℝ := 300

/-- The time the truck stops at each pail -/
def truck_stop_time : ℝ := 40

/-- The total time Michael walks -/
def total_time : ℝ := 900

/-- Calculate the positions of Michael and the truck at time t -/
def calculate_positions (t : ℝ) : Positions t :=
  sorry

/-- Theorem stating that Michael and the truck never meet within the given time -/
theorem no_meeting :
  ∀ t, 0 ≤ t ∧ t ≤ total_time → (calculate_positions t).michael < (calculate_positions t).truck :=
  sorry

end no_meeting_l3461_346168


namespace interlaced_roots_l3461_346116

theorem interlaced_roots (p₁ p₂ q₁ q₂ : ℝ) 
  (h : (q₁ - q₂)^2 + (p₁ - p₂)*(p₁*q₂ - p₂*q₁) < 0) :
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (∀ x, x^2 + p₁*x + q₁ = 0 ↔ (x = r₁ ∨ x = r₂)) ∧
    (∀ x, x^2 + p₂*x + q₂ = 0 ↔ (x = r₃ ∨ x = r₄)) ∧
    ((r₁ < r₃ ∧ r₃ < r₂ ∧ r₂ < r₄) ∨ (r₃ < r₁ ∧ r₁ < r₄ ∧ r₄ < r₂)) :=
by sorry

end interlaced_roots_l3461_346116


namespace only_B_is_random_l3461_346113

-- Define the events
inductive Event
| A
| B
| C
| D

-- Define a function to check if an event is random
def isRandom (e : Event) : Prop :=
  match e with
  | Event.A => false  -- Water freezing is deterministic
  | Event.B => true   -- Bus arrival is random
  | Event.C => false  -- Sum of 13 is impossible with two dice
  | Event.D => false  -- Pigeonhole principle guarantees same month births

-- Theorem statement
theorem only_B_is_random :
  ∀ e : Event, isRandom e ↔ e = Event.B :=
sorry

end only_B_is_random_l3461_346113


namespace problem_statement_l3461_346125

theorem problem_statement (x y : ℝ) (hx : x = 7) (hy : y = 3) :
  (x - y)^2 * (x + y) = 160 := by sorry

end problem_statement_l3461_346125


namespace article_cost_l3461_346105

theorem article_cost (selling_price_high selling_price_low : ℝ) 
  (h1 : selling_price_high = 360)
  (h2 : selling_price_low = 340)
  (h3 : selling_price_high - selling_price_low = 20)
  (h4 : ∀ cost, 
    (selling_price_high - cost) = (selling_price_low - cost) * 1.05) :
  ∃ cost : ℝ, cost = 60 := by
sorry

end article_cost_l3461_346105


namespace inverse_variation_proof_l3461_346141

/-- Given that x varies inversely as the square of y, prove that x = 1/9 when y = 6, given that y = 2 when x = 1. -/
theorem inverse_variation_proof (x y : ℝ) (h : ∃ k : ℝ, ∀ x y, x = k / (y ^ 2)) 
  (h1 : ∃ x₀, x₀ = 1 ∧ y = 2) : 
  (y = 6) → (x = 1 / 9) := by
sorry

end inverse_variation_proof_l3461_346141


namespace arithmetic_progression_product_perfect_power_l3461_346135

theorem arithmetic_progression_product_perfect_power :
  ∃ (a d : ℕ+) (b : ℕ+),
    (∀ i j : Fin 5, i ≠ j → a + i.val * d ≠ a + j.val * d) ∧
    (∃ (c : ℕ+), (a * (a + d) * (a + 2 * d) * (a + 3 * d) * (a + 4 * d) : ℕ) = c ^ 2008) :=
by sorry

end arithmetic_progression_product_perfect_power_l3461_346135


namespace hartley_puppies_count_l3461_346121

/-- The number of puppies Hartley has -/
def num_puppies : ℕ := 4

/-- The weight of each puppy in kilograms -/
def puppy_weight : ℝ := 7.5

/-- The number of cats -/
def num_cats : ℕ := 14

/-- The weight of each cat in kilograms -/
def cat_weight : ℝ := 2.5

/-- The difference in total weight between cats and puppies in kilograms -/
def weight_difference : ℝ := 5

theorem hartley_puppies_count :
  num_puppies * puppy_weight = num_cats * cat_weight - weight_difference :=
by sorry

end hartley_puppies_count_l3461_346121


namespace calculate_expression_l3461_346190

theorem calculate_expression : (235 - 2 * 3 * 5) * 7 / 5 = 287 := by
  sorry

end calculate_expression_l3461_346190


namespace necessary_but_not_sufficient_l3461_346132

def M : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧
  (∃ a : ℝ, a ∈ M ∧ a ∉ N) :=
by sorry

end necessary_but_not_sufficient_l3461_346132


namespace beads_per_necklace_l3461_346112

theorem beads_per_necklace (total_beads : ℕ) (num_necklaces : ℕ) 
  (h1 : total_beads = 18) (h2 : num_necklaces = 6) :
  total_beads / num_necklaces = 3 := by
  sorry

end beads_per_necklace_l3461_346112


namespace sin_five_pi_thirds_l3461_346175

theorem sin_five_pi_thirds : Real.sin (5 * π / 3) = - (Real.sqrt 3 / 2) := by
  sorry

end sin_five_pi_thirds_l3461_346175


namespace gasoline_added_to_tank_l3461_346162

/-- The amount of gasoline added to a tank -/
def gasoline_added (total_capacity : ℝ) (initial_fraction : ℝ) (final_fraction : ℝ) : ℝ :=
  total_capacity * (final_fraction - initial_fraction)

/-- Proof that 7.2 gallons of gasoline were added to the tank -/
theorem gasoline_added_to_tank : 
  gasoline_added 48 (3/4) (9/10) = 7.2 := by
sorry

end gasoline_added_to_tank_l3461_346162


namespace margies_change_l3461_346100

/-- Calculates the change Margie receives after buying oranges -/
def margieChange (numOranges : ℕ) (costPerOrange : ℚ) (amountPaid : ℚ) : ℚ :=
  amountPaid - (numOranges : ℚ) * costPerOrange

/-- Theorem stating that Margie's change is $8.50 -/
theorem margies_change :
  margieChange 5 (30 / 100) 10 = 17 / 2 := by
  sorry

#eval margieChange 5 (30 / 100) 10

end margies_change_l3461_346100


namespace charlie_apple_picking_l3461_346124

theorem charlie_apple_picking (total bags_golden bags_cortland bags_macintosh : ℚ) : 
  total = 0.67 ∧ 
  bags_golden = 0.17 ∧ 
  bags_cortland = 0.33 ∧ 
  total = bags_golden + bags_macintosh + bags_cortland → 
  bags_macintosh = 0.17 := by
sorry

end charlie_apple_picking_l3461_346124


namespace expensive_module_cost_l3461_346140

/-- Proves the cost of the more expensive module in a Bluetooth device assembly factory -/
theorem expensive_module_cost :
  let cheaper_cost : ℝ := 2.5
  let total_stock_value : ℝ := 62.5
  let total_modules : ℕ := 22
  let cheaper_modules : ℕ := 21
  let expensive_modules : ℕ := total_modules - cheaper_modules
  expensive_modules * expensive_cost + cheaper_modules * cheaper_cost = total_stock_value →
  expensive_cost = 10 := by
  sorry

#check expensive_module_cost

end expensive_module_cost_l3461_346140


namespace race_time_a_l3461_346149

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 1000 meters long
  1000 = a.speed * a.time ∧
  -- B runs 960 meters in the time A finishes
  960 = b.speed * a.time ∧
  -- A finishes 8 seconds before B
  b.time = a.time + 8 ∧
  -- A and B have different speeds
  a.speed ≠ b.speed

/-- The theorem stating A's finishing time -/
theorem race_time_a (a b : Runner) (h : Race a b) : a.time = 200 :=
  sorry

#check race_time_a

end race_time_a_l3461_346149


namespace connie_initial_marbles_l3461_346165

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 73

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_initial_marbles : initial_marbles = 143 := by
  sorry

end connie_initial_marbles_l3461_346165


namespace contractor_fine_proof_l3461_346199

/-- Calculates the daily fine for absence given contract parameters -/
def calculate_daily_fine (total_days : ℕ) (daily_pay : ℚ) (total_received : ℚ) (days_absent : ℕ) : ℚ :=
  let days_worked := total_days - days_absent
  let total_earned := days_worked * daily_pay
  (total_earned - total_received) / days_absent

/-- Proves that the daily fine is 7.50 given the contract parameters -/
theorem contractor_fine_proof :
  calculate_daily_fine 30 25 620 4 = 7.5 := by
  sorry

end contractor_fine_proof_l3461_346199


namespace rectangle_area_unchanged_l3461_346136

theorem rectangle_area_unchanged (x y : ℝ) :
  x > 0 ∧ y > 0 ∧
  x * y = (x + 3.5) * (y - 1.33) ∧
  x * y = (x - 3.5) * (y + 1.67) →
  x * y = 35 := by
  sorry

end rectangle_area_unchanged_l3461_346136


namespace total_onions_grown_l3461_346115

theorem total_onions_grown (sara_onions sally_onions fred_onions : ℕ) 
  (h1 : sara_onions = 4)
  (h2 : sally_onions = 5)
  (h3 : fred_onions = 9) :
  sara_onions + sally_onions + fred_onions = 18 := by
  sorry

end total_onions_grown_l3461_346115


namespace new_profit_percentage_is_50_percent_l3461_346138

/-- Represents the selling price of a key chain -/
def selling_price : ℝ := 100

/-- Represents the initial manufacturing cost -/
def initial_cost : ℝ := 65

/-- Represents the new manufacturing cost -/
def new_cost : ℝ := 50

/-- Represents the initial profit percentage -/
def initial_profit_percentage : ℝ := 0.35

/-- Theorem stating that the new profit percentage is 50% given the conditions -/
theorem new_profit_percentage_is_50_percent :
  let initial_profit := initial_profit_percentage * selling_price
  let initial_equation := initial_cost + initial_profit = selling_price
  let new_profit := selling_price - new_cost
  let new_profit_percentage := new_profit / selling_price
  initial_equation → new_profit_percentage = 0.5 := by sorry


end new_profit_percentage_is_50_percent_l3461_346138


namespace max_t_value_l3461_346114

theorem max_t_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 18 →
  k < m → m < r → r < s → s < t →
  r ≤ 23 →
  t = 40 := by
sorry

end max_t_value_l3461_346114


namespace jellybean_difference_l3461_346177

theorem jellybean_difference (gigi_jellybeans rory_jellybeans lorelai_jellybeans : ℕ) : 
  gigi_jellybeans = 15 →
  rory_jellybeans > gigi_jellybeans →
  lorelai_jellybeans = 3 * (rory_jellybeans + gigi_jellybeans) →
  lorelai_jellybeans = 180 →
  rory_jellybeans - gigi_jellybeans = 30 := by
sorry

end jellybean_difference_l3461_346177


namespace shifted_checkerboard_half_shaded_l3461_346117

/-- Represents a square grid -/
structure SquareGrid :=
  (size : Nat)

/-- Represents a shading pattern on a square grid -/
structure ShadingPattern :=
  (grid : SquareGrid)
  (shaded_squares : Nat)

/-- A 6x6 grid with a shifted checkerboard shading pattern -/
def shifted_checkerboard : ShadingPattern :=
  { grid := { size := 6 },
    shaded_squares := 18 }

/-- Calculate the percentage of shaded squares -/
def shaded_percentage (pattern : ShadingPattern) : Rat :=
  pattern.shaded_squares / (pattern.grid.size * pattern.grid.size) * 100

/-- Theorem: The shaded percentage of a 6x6 shifted checkerboard is 50% -/
theorem shifted_checkerboard_half_shaded :
  shaded_percentage shifted_checkerboard = 50 := by
  sorry

end shifted_checkerboard_half_shaded_l3461_346117


namespace base8_246_to_base10_l3461_346104

/-- Converts a base 8 number to base 10 -/
def base8_to_base10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

/-- The base 10 representation of 246₈ is 166 -/
theorem base8_246_to_base10 : base8_to_base10 2 4 6 = 166 := by
  sorry

end base8_246_to_base10_l3461_346104


namespace no_savings_from_radio_offer_l3461_346145

-- Define the in-store price
def in_store_price : ℚ := 139.99

-- Define the radio offer components
def radio_payment : ℚ := 33.00
def num_payments : ℕ := 4
def shipping_charge : ℚ := 11.99

-- Calculate the total radio offer price
def radio_offer_price : ℚ := radio_payment * num_payments + shipping_charge

-- Define the conversion factor from dollars to cents
def dollars_to_cents : ℕ := 100

-- Theorem statement
theorem no_savings_from_radio_offer : 
  (radio_offer_price - in_store_price) * dollars_to_cents = 0 := by sorry

end no_savings_from_radio_offer_l3461_346145


namespace boat_speed_ratio_l3461_346178

/-- Proves that the ratio of boat speed in still water to stream speed is 6:1 -/
theorem boat_speed_ratio (still_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  still_speed = 24 →
  downstream_distance = 112 →
  downstream_time = 4 →
  (still_speed / (downstream_distance / downstream_time - still_speed) = 6) :=
by sorry

end boat_speed_ratio_l3461_346178


namespace initial_number_of_girls_l3461_346187

theorem initial_number_of_girls (initial_boys : ℕ) (girls_joined : ℕ) (final_girls : ℕ) : 
  initial_boys = 761 → girls_joined = 682 → final_girls = 1414 → 
  final_girls - girls_joined = 732 :=
by
  sorry

end initial_number_of_girls_l3461_346187


namespace cube_volume_problem_l3461_346150

theorem cube_volume_problem (cube_A cube_B : Real → Real → Real → Real) :
  (∀ x y z, cube_A x y z = 8) →
  (∀ x y z, (6 * (cube_B x y z)^(2/3)) = 3 * (6 * 2^2)) →
  (∀ x y z, cube_B x y z = 24 * Real.sqrt 3) := by
  sorry

end cube_volume_problem_l3461_346150


namespace complement_of_S_in_U_l3461_346143

def U : Finset Nat := {1,2,3,4,5,6,7}
def S : Finset Nat := {1,3,5}

theorem complement_of_S_in_U :
  (U \ S) = {2,4,6,7} := by sorry

end complement_of_S_in_U_l3461_346143


namespace medication_cost_leap_year_l3461_346170

/-- Calculate the total medication cost for a leap year given the following conditions:
  * 3 types of pills
  * Pill 1 costs $1.5, Pill 2 costs $2.3, Pill 3 costs $3.8
  * Insurance covers 40% of Pill 1, 25% of Pill 2, 10% of Pill 3
  * Discount card provides 15% off Pill 2 and 5% off Pill 3
  * A leap year has 366 days -/
theorem medication_cost_leap_year : 
  let pill1_cost : ℝ := 1.5
  let pill2_cost : ℝ := 2.3
  let pill3_cost : ℝ := 3.8
  let insurance_coverage1 : ℝ := 0.4
  let insurance_coverage2 : ℝ := 0.25
  let insurance_coverage3 : ℝ := 0.1
  let discount_card2 : ℝ := 0.15
  let discount_card3 : ℝ := 0.05
  let days_in_leap_year : ℕ := 366

  let pill1_final_cost := pill1_cost * (1 - insurance_coverage1)
  let pill2_final_cost := pill2_cost * (1 - insurance_coverage2) * (1 - discount_card2)
  let pill3_final_cost := pill3_cost * (1 - insurance_coverage3) * (1 - discount_card3)

  let daily_cost := pill1_final_cost + pill2_final_cost + pill3_final_cost
  let yearly_cost := daily_cost * days_in_leap_year

  yearly_cost = 2055.5835 := by
sorry

end medication_cost_leap_year_l3461_346170


namespace vector_addition_proof_l3461_346111

theorem vector_addition_proof (a b : ℝ × ℝ) (h1 : a = (1, -2)) (h2 : b = (-2, 2)) :
  a + 2 • b = (-3, 2) := by
  sorry

end vector_addition_proof_l3461_346111


namespace min_value_theorem_l3461_346129

theorem min_value_theorem (x A B : ℝ) (hx : x > 0) (hA : A > 0) (hB : B > 0)
  (h1 : x^2 + 1/x^2 = A) (h2 : x - 1/x = B) :
  ∀ y, y > 0 → y^2 + 1/y^2 = A → y - 1/y = B → (A + 1) / B ≥ 2 * Real.sqrt 3 :=
by sorry

end min_value_theorem_l3461_346129


namespace twin_prime_conjecture_equivalence_l3461_346153

def is_twin_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime (p + 2)

def cannot_be_written (k : ℤ) : Prop :=
  ∀ u v : ℕ, u > 0 ∧ v > 0 →
    k ≠ 6*u*v + u + v ∧
    k ≠ 6*u*v + u - v ∧
    k ≠ 6*u*v - u + v ∧
    k ≠ 6*u*v - u - v

theorem twin_prime_conjecture_equivalence :
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ p ∈ S, is_twin_prime p) ↔
  (∃ (T : Set ℤ), Set.Infinite T ∧ ∀ k ∈ T, cannot_be_written k) :=
sorry

end twin_prime_conjecture_equivalence_l3461_346153


namespace equation_solutions_l3461_346118

theorem equation_solutions : 
  {x : ℝ | (x + 1) * (x - 2) = x + 1} = {-1, 3} := by sorry

end equation_solutions_l3461_346118


namespace discount_percentage_l3461_346179

theorem discount_percentage (cost_price : ℝ) (profit_with_discount : ℝ) (profit_without_discount : ℝ)
  (h1 : profit_with_discount = 42.5)
  (h2 : profit_without_discount = 50)
  : (((1 + profit_without_discount / 100) * cost_price - (1 + profit_with_discount / 100) * cost_price) /
     ((1 + profit_without_discount / 100) * cost_price)) * 100 = 5 := by
  sorry

end discount_percentage_l3461_346179


namespace remainder_invariance_l3461_346176

theorem remainder_invariance (n : ℤ) (h : n % 7 = 2) : (n + 5040) % 7 = 2 := by
  sorry

end remainder_invariance_l3461_346176


namespace investment_net_change_l3461_346108

def initial_investment : ℝ := 200
def first_year_loss_rate : ℝ := 0.1
def second_year_gain_rate : ℝ := 0.3

theorem investment_net_change :
  let first_year_amount := initial_investment * (1 - first_year_loss_rate)
  let second_year_amount := first_year_amount * (1 + second_year_gain_rate)
  let net_change_rate := (second_year_amount - initial_investment) / initial_investment
  net_change_rate = 0.17 := by
sorry

end investment_net_change_l3461_346108


namespace max_value_of_expression_l3461_346146

theorem max_value_of_expression (x y : ℝ) 
  (hx : -4 ≤ x ∧ x ≤ -2) (hy : 2 ≤ y ∧ y ≤ 4) :
  (∀ z w : ℝ, -4 ≤ z ∧ z ≤ -2 ∧ 2 ≤ w ∧ w ≤ 4 → (z + w) / z ≤ (x + y) / x) →
  (x + y) / x = 1/2 :=
by sorry

end max_value_of_expression_l3461_346146


namespace min_value_of_roots_l3461_346151

theorem min_value_of_roots (a x y : ℝ) : 
  x^2 - 2*a*x + a + 6 = 0 →
  y^2 - 2*a*y + a + 6 = 0 →
  x ≠ y →
  ∃ (z : ℝ), ∀ (b : ℝ), (z^2 - 2*b*z + b + 6 = 0) → (x - 1)^2 + (y - 1)^2 ≥ 8 :=
by sorry

end min_value_of_roots_l3461_346151


namespace graveyard_bones_count_l3461_346156

def total_skeletons : ℕ := 20
def adult_woman_bones : ℕ := 20

theorem graveyard_bones_count :
  let adult_women := total_skeletons / 2
  let adult_men := (total_skeletons - adult_women) / 2
  let children := (total_skeletons - adult_women) / 2
  let adult_man_bones := adult_woman_bones + 5
  let child_bones := adult_woman_bones / 2
  adult_women * adult_woman_bones +
  adult_men * adult_man_bones +
  children * child_bones = 375 := by
sorry

end graveyard_bones_count_l3461_346156


namespace route_time_difference_l3461_346120

/-- Represents the time added by a red light -/
def red_light_time : ℕ := 3

/-- Represents the number of stoplights on the first route -/
def num_stoplights : ℕ := 3

/-- Represents the time for the first route if all lights are green -/
def first_route_green_time : ℕ := 10

/-- Represents the time for the second route -/
def second_route_time : ℕ := 14

/-- Calculates the time for the first route when all lights are red -/
def first_route_red_time : ℕ := first_route_green_time + num_stoplights * red_light_time

/-- Proves that the difference between the first route with all red lights and the second route is 5 minutes -/
theorem route_time_difference : first_route_red_time - second_route_time = 5 := by
  sorry

end route_time_difference_l3461_346120


namespace barb_dress_savings_l3461_346184

theorem barb_dress_savings (original_price savings : ℝ) 
  (h1 : original_price = 180)
  (h2 : savings = 80)
  (h3 : original_price - savings < original_price / 2) :
  |(original_price / 2) - (original_price - savings)| = 10 :=
by sorry

end barb_dress_savings_l3461_346184


namespace ivan_walking_time_l3461_346109

/-- Represents the journey of Ivan Ivanovich to work -/
structure Journey where
  /-- Walking speed of Ivan Ivanovich -/
  u : ℝ
  /-- Speed of the service car -/
  v : ℝ
  /-- Usual time it takes the service car to drive Ivan from home to work -/
  t : ℝ
  /-- Time Ivan Ivanovich walked -/
  T : ℝ
  /-- The service car speed is positive -/
  hv : v > 0
  /-- The walking speed is positive and less than the car speed -/
  hu : 0 < u ∧ u < v
  /-- The usual journey time is positive -/
  ht : t > 0
  /-- The walking time is positive and less than the usual journey time -/
  hT : 0 < T ∧ T < t
  /-- Ivan left 90 minutes earlier and arrived 20 minutes earlier -/
  h_time_diff : T + (t - T + 70) = t + 70
  /-- The distance walked equals the distance the car would travel in 10 minutes -/
  h_meeting_point : u * T = 10 * v

/-- Theorem stating that Ivan Ivanovich walked for 80 minutes -/
theorem ivan_walking_time (j : Journey) : j.T = 80 := by sorry

end ivan_walking_time_l3461_346109


namespace pascal_triangle_properties_l3461_346134

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def pascal_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_triangle_properties :
  (pascal_sum 30 = 465) ∧
  (binomial 30 5 > 0) := by
  sorry

end pascal_triangle_properties_l3461_346134


namespace triangle_area_inequality_l3461_346198

theorem triangle_area_inequality (a b c α β γ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hα : α = 2 * Real.sqrt (b * c))
  (hβ : β = 2 * Real.sqrt (a * c))
  (hγ : γ = 2 * Real.sqrt (a * b)) :
  a / α + b / β + c / γ ≥ 3 / 2 := by
sorry

end triangle_area_inequality_l3461_346198
