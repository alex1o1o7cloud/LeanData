import Mathlib

namespace fraction_square_decimal_equivalent_l111_11123

theorem fraction_square_decimal_equivalent : (1 / 9 : ℚ)^2 = 0.012345679012345678 := by
  sorry

end fraction_square_decimal_equivalent_l111_11123


namespace solution_part1_solution_part2_l111_11171

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for part 1
theorem solution_part1 :
  {x : ℝ | f x (-1) ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem for part 2
theorem solution_part2 :
  {a : ℝ | ∀ x, f x a ≥ 2} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end solution_part1_solution_part2_l111_11171


namespace number_of_boys_l111_11144

/-- Proves that the number of boys in a group is 5 given specific conditions about weights --/
theorem number_of_boys (num_girls : ℕ) (num_total : ℕ) (avg_girls : ℚ) (avg_boys : ℚ) (avg_total : ℚ) :
  num_girls = 5 →
  num_total = 10 →
  avg_girls = 45 →
  avg_boys = 55 →
  avg_total = 50 →
  ∃ (num_boys : ℕ), num_boys = 5 ∧ num_girls + num_boys = num_total ∧
    (num_girls : ℚ) * avg_girls + (num_boys : ℚ) * avg_boys = (num_total : ℚ) * avg_total :=
by
  sorry


end number_of_boys_l111_11144


namespace horizontal_chord_cubic_l111_11125

/-- A cubic function f(x) = x^3 - x has a horizontal chord of length a 
    if and only if 0 < a ≤ 2 -/
theorem horizontal_chord_cubic (a : ℝ) :
  (∃ x : ℝ, (x + a)^3 - (x + a) = x^3 - x) ↔ (0 < a ∧ a ≤ 2) :=
by sorry

end horizontal_chord_cubic_l111_11125


namespace triangle_circles_QR_length_l111_11111

-- Define the right triangle DEF
def Triangle (DE EF DF : ℝ) := DE = 5 ∧ EF = 12 ∧ DF = 13

-- Define the circle centered at Q
def CircleQ (Q E D : ℝ × ℝ) := 
  (Q.1 - E.1)^2 + (Q.2 - E.2)^2 = (Q.1 - D.1)^2 + (Q.2 - D.2)^2

-- Define the circle centered at R
def CircleR (R D F : ℝ × ℝ) := 
  (R.1 - D.1)^2 + (R.2 - D.2)^2 = (R.1 - F.1)^2 + (R.2 - F.2)^2

-- Define the tangency conditions
def TangentQ (Q E : ℝ × ℝ) := True  -- Placeholder for tangency condition
def TangentR (R D : ℝ × ℝ) := True  -- Placeholder for tangency condition

-- State the theorem
theorem triangle_circles_QR_length 
  (D E F Q R : ℝ × ℝ) 
  (h_triangle : Triangle (dist D E) (dist E F) (dist D F))
  (h_circleQ : CircleQ Q E D)
  (h_circleR : CircleR R D F)
  (h_tangentQ : TangentQ Q E)
  (h_tangentR : TangentR R D) :
  dist Q R = 5 := by
  sorry


end triangle_circles_QR_length_l111_11111


namespace quadratic_relationship_l111_11103

/-- A quadratic function f(x) = 3x^2 + ax + b where f(x - 1) is an even function -/
def f (a b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + a * x + b

theorem quadratic_relationship (a b : ℝ) 
  (h : ∀ x, f a b (x - 1) = f a b (1 - x)) : 
  f a b (-1) < f a b (-3/2) ∧ f a b (-3/2) = f a b (3/2) := by
  sorry

end quadratic_relationship_l111_11103


namespace boys_without_calculators_l111_11191

theorem boys_without_calculators (total_boys : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ) 
  (h1 : total_boys = 20)
  (h2 : students_with_calculators = 25)
  (h3 : girls_with_calculators = 15) :
  total_boys - (students_with_calculators - girls_with_calculators) = 10 :=
by sorry

end boys_without_calculators_l111_11191


namespace badminton_probabilities_l111_11180

/-- Represents the state of a badminton game -/
structure BadmintonGame where
  score_a : Nat
  score_b : Nat
  a_serving : Bool

/-- Rules for winning a badminton game -/
def game_won (game : BadmintonGame) : Bool :=
  (game.score_a = 21 && game.score_b < 20) ||
  (game.score_b = 21 && game.score_a < 20) ||
  (game.score_a ≥ 20 && game.score_b ≥ 20 && 
   ((game.score_a = 30) || (game.score_b = 30) || 
    (game.score_a ≥ 22 && game.score_a - game.score_b = 2) ||
    (game.score_b ≥ 22 && game.score_b - game.score_a = 2)))

/-- Probability of player A winning a rally when serving -/
def p_a_serving : ℝ := 0.4

/-- Probability of player A winning a rally when not serving -/
def p_a_not_serving : ℝ := 0.5

/-- The initial game state at 28:28 with A serving -/
def initial_state : BadmintonGame := ⟨28, 28, true⟩

theorem badminton_probabilities :
  let p_game_ends_in_two : ℝ := 0.46
  let p_a_wins : ℝ := 0.4
  (∃ (p_game_ends_in_two' p_a_wins' : ℝ),
    p_game_ends_in_two' = p_game_ends_in_two ∧
    p_a_wins' = p_a_wins ∧
    p_game_ends_in_two' = p_a_serving * p_a_serving + (1 - p_a_serving) * (1 - p_a_not_serving) ∧
    p_a_wins' = p_a_serving * p_a_serving + 
                p_a_serving * (1 - p_a_serving) * p_a_not_serving +
                (1 - p_a_serving) * p_a_not_serving * p_a_serving) :=
by sorry

end badminton_probabilities_l111_11180


namespace parabola_intersection_theorem_l111_11146

/-- A parabola with equation y = 2x^2 + 8x + m -/
structure Parabola where
  m : ℝ

/-- Predicate to check if a parabola has only two common points with the coordinate axes -/
def has_two_axis_intersections (p : Parabola) : Prop :=
  -- This is a placeholder for the actual condition
  True

/-- Theorem stating that if a parabola y = 2x^2 + 8x + m has only two common points
    with the coordinate axes, then m = 0 or m = 8 -/
theorem parabola_intersection_theorem (p : Parabola) :
  has_two_axis_intersections p → p.m = 0 ∨ p.m = 8 := by
  sorry


end parabola_intersection_theorem_l111_11146


namespace sum_of_reciprocals_l111_11101

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 24) :
  1 / x + 1 / y = 1 / 2 := by
  sorry

end sum_of_reciprocals_l111_11101


namespace new_person_weight_l111_11196

theorem new_person_weight (initial_total_weight : ℝ) : 
  let initial_avg := initial_total_weight / 10
  let new_avg := initial_avg + 5
  let new_total_weight := new_avg * 10
  new_total_weight - initial_total_weight + 60 = 110 := by
sorry

end new_person_weight_l111_11196


namespace parallel_vectors_sum_l111_11112

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

theorem parallel_vectors_sum (m : ℝ) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![m, 2]
  are_parallel a b →
  (3 • a + 2 • b) = ![14, 7] := by
sorry

end parallel_vectors_sum_l111_11112


namespace specific_gold_cube_profit_l111_11115

/-- Calculates the profit from selling a gold cube -/
def gold_cube_profit (side_length : ℝ) (density : ℝ) (buy_price : ℝ) (sell_multiplier : ℝ) : ℝ :=
  let volume := side_length ^ 3
  let mass := volume * density
  let cost := mass * buy_price
  let sell_price := cost * sell_multiplier
  sell_price - cost

/-- Theorem stating the profit for a specific gold cube -/
theorem specific_gold_cube_profit :
  gold_cube_profit 6 19 60 1.5 = 123120 := by
  sorry

end specific_gold_cube_profit_l111_11115


namespace fraction_of_fraction_one_ninth_of_three_fourths_l111_11133

theorem fraction_of_fraction (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem one_ninth_of_three_fourths :
  (1 / 9) / (3 / 4) = 4 / 27 := by sorry

end fraction_of_fraction_one_ninth_of_three_fourths_l111_11133


namespace cookie_dough_thickness_l111_11116

/-- The thickness of a cylindrical layer formed by doubling the volume of a sphere
    and spreading it over a circular area. -/
theorem cookie_dough_thickness 
  (initial_radius : ℝ) 
  (final_radius : ℝ) 
  (initial_radius_value : initial_radius = 3)
  (final_radius_value : final_radius = 9) :
  let initial_volume := (4/3) * Real.pi * initial_radius^3
  let doubled_volume := 2 * initial_volume
  let final_area := Real.pi * final_radius^2
  let thickness := doubled_volume / final_area
  thickness = 8/9 := by sorry

end cookie_dough_thickness_l111_11116


namespace min_sum_m_n_l111_11186

theorem min_sum_m_n (m n : ℕ+) (h : 45 * m = n^3) : 
  (∀ (m' n' : ℕ+), 45 * m' = n'^3 → m' + n' ≥ m + n) → m + n = 90 :=
by sorry

end min_sum_m_n_l111_11186


namespace jansi_shopping_ratio_l111_11147

/-- Represents the shopping scenario of Jansi -/
structure ShoppingScenario where
  initial_rupees : ℕ
  initial_coins : ℕ
  spent : ℚ

/-- Conditions of Jansi's shopping trip -/
def jansi_shopping : ShoppingScenario :=
{ initial_rupees := 15,
  initial_coins := 15,
  spent := 9.6 }

/-- The ratio of the amount Jansi came back with to the amount she started out with -/
def shopping_ratio (s : ShoppingScenario) : ℚ × ℚ :=
  let initial_amount : ℚ := s.initial_rupees + 0.2 * s.initial_coins
  let final_amount : ℚ := initial_amount - s.spent
  (final_amount, initial_amount)

theorem jansi_shopping_ratio :
  shopping_ratio jansi_shopping = (9, 25) := by sorry

end jansi_shopping_ratio_l111_11147


namespace team_formation_count_l111_11154

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem team_formation_count : 
  let total_boys : ℕ := 4
  let total_girls : ℕ := 5
  let team_size : ℕ := 5
  let ways_3b2g : ℕ := choose total_boys 3 * choose total_girls 2
  let ways_4b1g : ℕ := choose total_boys 4 * choose total_girls 1
  ways_3b2g + ways_4b1g = 45 := by sorry

end team_formation_count_l111_11154


namespace solve_for_t_l111_11188

theorem solve_for_t (s t : ℚ) (eq1 : 8 * s + 7 * t = 145) (eq2 : s = t + 3) : t = 121 / 15 := by
  sorry

end solve_for_t_l111_11188


namespace vlad_height_l111_11131

/-- Proves that Vlad is 3 inches taller than 6 feet given the conditions of the problem -/
theorem vlad_height (vlad_feet : ℕ) (vlad_inches : ℕ) (sister_feet : ℕ) (sister_inches : ℕ) 
  (height_difference : ℕ) :
  vlad_feet = 6 →
  sister_feet = 2 →
  sister_inches = 10 →
  height_difference = 41 →
  vlad_inches = 3 :=
by
  sorry

#check vlad_height

end vlad_height_l111_11131


namespace remainder_4x_mod_7_l111_11192

theorem remainder_4x_mod_7 (x : ℤ) (h : x % 7 = 5) : (4 * x) % 7 = 6 := by
  sorry

end remainder_4x_mod_7_l111_11192


namespace opposite_of_three_l111_11107

theorem opposite_of_three : -(3 : ℝ) = -3 := by
  sorry

end opposite_of_three_l111_11107


namespace proposition_two_l111_11183

theorem proposition_two (a b c : ℝ) (h1 : c > 1) (h2 : 0 < b) (h3 : b < 2) :
  a^2 + a*b + c > 0 := by
  sorry

end proposition_two_l111_11183


namespace point_classification_l111_11179

-- Define the region D
def D (x y : ℝ) : Prop := y < x ∧ x + y ≤ 1 ∧ y ≥ -3

-- Define points P and Q
def P : ℝ × ℝ := (0, -2)
def Q : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem point_classification :
  D P.1 P.2 ∧ ¬D Q.1 Q.2 := by sorry

end point_classification_l111_11179


namespace marble_problem_l111_11122

theorem marble_problem :
  ∃ n : ℕ,
    (∀ m : ℕ, m > 0 ∧ m % 8 = 5 ∧ m % 7 = 2 → n ≤ m) ∧
    n % 8 = 5 ∧
    n % 7 = 2 ∧
    n % 9 = 1 ∧
    n = 37 := by
  sorry

end marble_problem_l111_11122


namespace special_quadrilateral_angles_l111_11127

/-- A quadrilateral with specific angle relationships and side equality -/
structure SpecialQuadrilateral where
  A : ℝ  -- Angle at vertex A
  B : ℝ  -- Angle at vertex B
  C : ℝ  -- Angle at vertex C
  D : ℝ  -- Angle at vertex D
  angle_B_triple_A : B = 3 * A
  angle_C_triple_B : C = 3 * B
  angle_D_triple_C : D = 3 * C
  sum_of_angles : A + B + C + D = 360
  sides_equal : True  -- Representing AD = BC (not used in angle calculations)

/-- The angles in the special quadrilateral are 9°, 27°, 81°, and 243° -/
theorem special_quadrilateral_angles (q : SpecialQuadrilateral) :
  q.A = 9 ∧ q.B = 27 ∧ q.C = 81 ∧ q.D = 243 := by
  sorry

end special_quadrilateral_angles_l111_11127


namespace log_expression_equals_21_l111_11151

theorem log_expression_equals_21 :
  2 * Real.log 25 / Real.log 5 + 3 * Real.log 64 / Real.log 2 - Real.log (Real.log (3^10) / Real.log 3) = 21 := by
  sorry

end log_expression_equals_21_l111_11151


namespace coeff_comparison_l111_11156

open Polynomial

/-- The coefficient of x^20 in (1 + x^2 - x^3)^1000 is greater than
    the coefficient of x^20 in (1 - x^2 + x^3)^1000 --/
theorem coeff_comparison (x : ℝ) : 
  (coeff ((1 + X^2 - X^3 : ℝ[X])^1000) 20) > 
  (coeff ((1 - X^2 + X^3 : ℝ[X])^1000) 20) := by
  sorry

end coeff_comparison_l111_11156


namespace initial_roses_count_l111_11110

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := sorry

/-- The number of roses added to the vase -/
def added_roses : ℕ := 10

/-- The total number of roses after adding -/
def total_roses : ℕ := 16

/-- Theorem stating that the initial number of roses is 6 -/
theorem initial_roses_count : initial_roses = 6 := by
  sorry

end initial_roses_count_l111_11110


namespace two_numbers_equal_sum_product_quotient_l111_11174

theorem two_numbers_equal_sum_product_quotient :
  ∃! (x y : ℝ), x ≠ 0 ∧ x + y = x * y ∧ x * y = x / y ∧ x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end two_numbers_equal_sum_product_quotient_l111_11174


namespace four_card_selection_theorem_l111_11148

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Type := Unit

/-- Represents the four suits in a deck of cards -/
inductive Suit
| hearts | diamonds | clubs | spades

/-- Represents the rank of a card -/
inductive Rank
| ace | two | three | four | five | six | seven | eight | nine | ten
| jack | queen | king

/-- Determines if a rank is royal (J, Q, K) -/
def isRoyal (r : Rank) : Bool :=
  match r with
  | Rank.jack | Rank.queen | Rank.king => true
  | _ => false

/-- Represents a card with a suit and rank -/
structure Card where
  suit : Suit
  rank : Rank

/-- The number of ways to choose 4 cards from two standard decks -/
def numWaysToChoose4Cards (deck1 deck2 : StandardDeck) : ℕ := sorry

theorem four_card_selection_theorem (deck1 deck2 : StandardDeck) :
  numWaysToChoose4Cards deck1 deck2 = 438400 := by sorry

end four_card_selection_theorem_l111_11148


namespace square_root_expressions_l111_11117

theorem square_root_expressions :
  (∃ x : ℝ, x^2 = 12) ∧ 
  (∃ y : ℝ, y^2 = 8) ∧ 
  (∃ z : ℝ, z^2 = 6) ∧ 
  (∃ w : ℝ, w^2 = 3) ∧ 
  (∃ v : ℝ, v^2 = 2) →
  (∃ a b : ℝ, a^2 = 12 ∧ b^2 = 8 ∧ 
    a + b * Real.sqrt 6 = 6 * Real.sqrt 3) ∧
  (∃ c d e : ℝ, c^2 = 12 ∧ d^2 = 3 ∧ e^2 = 2 ∧
    c + 1 / (Real.sqrt 3 - Real.sqrt 2) - Real.sqrt 6 * d = 3 * Real.sqrt 3 - 2 * Real.sqrt 2) :=
by sorry

end square_root_expressions_l111_11117


namespace solution_set_inequality_l111_11195

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) * (x - 1) > 0 ↔ x < 1 ∨ x > 3 := by
sorry

end solution_set_inequality_l111_11195


namespace min_value_expression_l111_11139

theorem min_value_expression (x : ℝ) : (x^2 + 7) / Real.sqrt (x^2 + 3) ≥ 4 := by
  sorry

end min_value_expression_l111_11139


namespace game_probability_l111_11158

/-- The number of possible choices for each player -/
def num_choices : ℕ := 16

/-- The probability of not winning a prize in a single trial -/
def prob_not_winning : ℚ := 15 / 16

theorem game_probability :
  (1 : ℚ) - (num_choices : ℚ) / ((num_choices : ℚ) * (num_choices : ℚ)) = prob_not_winning :=
by sorry

end game_probability_l111_11158


namespace top_face_after_16_rounds_l111_11175

/-- Represents the faces of a cube -/
inductive Face : Type
  | A | B | C | D | E | F

/-- Represents the state of the cube -/
structure CubeState :=
  (top : Face)
  (front : Face)
  (right : Face)
  (back : Face)
  (left : Face)
  (bottom : Face)

/-- Performs one round of operations on the cube -/
def perform_round (state : CubeState) : CubeState :=
  sorry

/-- Initial state of the cube -/
def initial_state : CubeState :=
  { top := Face.E,
    front := Face.A,
    right := Face.C,
    back := Face.B,
    left := Face.D,
    bottom := Face.F }

/-- Theorem stating that after 16 rounds, the top face will be E -/
theorem top_face_after_16_rounds (n : Nat) :
  (n = 16) → (perform_round^[n] initial_state).top = Face.E :=
sorry

end top_face_after_16_rounds_l111_11175


namespace num_assignments_is_15000_l111_11190

/-- Represents a valid assignment of students to events -/
structure Assignment where
  /-- The mapping of students to events -/
  student_to_event : Fin 7 → Fin 5
  /-- Ensures that students 0 and 1 (representing A and B) are not in the same event -/
  students_separated : student_to_event 0 ≠ student_to_event 1
  /-- Ensures that each event has at least one participant -/
  events_nonempty : ∀ e : Fin 5, ∃ s : Fin 7, student_to_event s = e

/-- The number of valid assignments -/
def num_valid_assignments : ℕ := sorry

/-- The main theorem stating that the number of valid assignments is 15000 -/
theorem num_assignments_is_15000 : num_valid_assignments = 15000 := by sorry

end num_assignments_is_15000_l111_11190


namespace complex_power_modulus_l111_11113

theorem complex_power_modulus : Complex.abs ((2 + Complex.I) ^ 8) = 625 := by
  sorry

end complex_power_modulus_l111_11113


namespace min_ratio_partition_l111_11155

def S : Finset ℕ := Finset.range 10

theorem min_ratio_partition (p₁ p₂ : ℕ) 
  (h_partition : ∃ (A B : Finset ℕ), A ∪ B = S ∧ A ∩ B = ∅ ∧ 
    p₁ = A.prod id ∧ p₂ = B.prod id)
  (h_divisible : p₁ % p₂ = 0) :
  p₁ / p₂ ≥ 7 :=
sorry

end min_ratio_partition_l111_11155


namespace sequence_2023_l111_11140

theorem sequence_2023 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, a n > 0) →
  (∀ n, 2 * S n = a n * (a n + 1)) →
  a 2023 = 2023 := by
sorry

end sequence_2023_l111_11140


namespace cyclists_average_speed_cyclists_average_speed_is_22_point_5_l111_11178

/-- Cyclist's average speed problem -/
theorem cyclists_average_speed (total_distance : ℝ) (initial_speed : ℝ) 
  (speed_increase : ℝ) (distance_fraction : ℝ) : ℝ :=
  let new_speed := initial_speed * (1 + speed_increase)
  let time_first_part := (distance_fraction * total_distance) / initial_speed
  let time_second_part := ((1 - distance_fraction) * total_distance) / new_speed
  let total_time := time_first_part + time_second_part
  total_distance / total_time

/-- Proof of the cyclist's average speed -/
theorem cyclists_average_speed_is_22_point_5 :
  cyclists_average_speed 1 20 0.2 (1/3) = 22.5 := by
  sorry

end cyclists_average_speed_cyclists_average_speed_is_22_point_5_l111_11178


namespace vector_simplification_l111_11193

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define points in the vector space
variable (A B O M : V)

-- Define the theorem
theorem vector_simplification (A B O M : V) :
  (B - A) + (O - B) + (M - O) + (B - M) = B - A :=
by sorry

end vector_simplification_l111_11193


namespace family_movie_night_l111_11153

theorem family_movie_night (regular_price adult_price elderly_price : ℕ)
  (child_price_diff total_payment change num_adults num_elderly : ℕ) :
  regular_price = 15 →
  adult_price = 12 →
  elderly_price = 10 →
  child_price_diff = 5 →
  total_payment = 150 →
  change = 3 →
  num_adults = 4 →
  num_elderly = 2 →
  ∃ (num_children : ℕ),
    num_children = 11 ∧
    total_payment - change = 
      num_adults * adult_price + 
      num_elderly * elderly_price + 
      num_children * (adult_price - child_price_diff) :=
by sorry

end family_movie_night_l111_11153


namespace jumping_contest_l111_11145

theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) : 
  grasshopper_jump = 14 →
  frog_jump = grasshopper_jump + 37 →
  mouse_jump = frog_jump - 16 →
  mouse_jump - grasshopper_jump = 21 :=
by
  sorry

end jumping_contest_l111_11145


namespace only_54_65_rounds_differently_l111_11105

def round_to_nearest_tenth (x : Float) : Float :=
  (x * 10).round / 10

theorem only_54_65_rounds_differently : 
  let numbers := [54.56, 54.63, 54.64, 54.65, 54.59]
  ∀ x ∈ numbers, x ≠ 54.65 → round_to_nearest_tenth x = 54.6 ∧
  round_to_nearest_tenth 54.65 ≠ 54.6 := by
  sorry

end only_54_65_rounds_differently_l111_11105


namespace f_6_equals_37_l111_11102

def f : ℕ → ℤ
| 0 => 0  -- Arbitrary base case
| n + 1 =>
  if 1 ≤ n + 1 ∧ n + 1 ≤ 4 then f n - (n + 1)
  else if 5 ≤ n + 1 ∧ n + 1 ≤ 8 then f n + 2 * (n + 1)
  else f n * (n + 1)

theorem f_6_equals_37 (h : f 4 = 15) : f 6 = 37 := by
  sorry

end f_6_equals_37_l111_11102


namespace school_bus_capacity_l111_11118

theorem school_bus_capacity 
  (columns_per_bus : ℕ) 
  (rows_per_bus : ℕ) 
  (number_of_buses : ℕ) 
  (h1 : columns_per_bus = 4) 
  (h2 : rows_per_bus = 10) 
  (h3 : number_of_buses = 6) : 
  columns_per_bus * rows_per_bus * number_of_buses = 240 := by
  sorry

end school_bus_capacity_l111_11118


namespace sin_negative_390_degrees_l111_11185

theorem sin_negative_390_degrees : Real.sin (-(390 * π / 180)) = -(1 / 2) := by
  sorry

end sin_negative_390_degrees_l111_11185


namespace sqrt_sum_inequality_l111_11149

theorem sqrt_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2 := by
  sorry

end sqrt_sum_inequality_l111_11149


namespace parametric_to_general_form_l111_11164

/-- Given parametric equations of a line, prove its general form -/
theorem parametric_to_general_form (t : ℝ) (x y : ℝ) :
  x = 2 - 3 * t ∧ y = 1 + 2 * t → 2 * x + 3 * y - 7 = 0 := by
  sorry

end parametric_to_general_form_l111_11164


namespace max_songs_in_three_hours_l111_11142

/-- Represents the maximum number of songs that can be played in a given time -/
def max_songs_played (short_songs : ℕ) (long_songs : ℕ) (short_duration : ℕ) (long_duration : ℕ) (total_time : ℕ) : ℕ :=
  let short_used := min short_songs (total_time / short_duration)
  let remaining_time := total_time - short_used * short_duration
  let long_used := min long_songs (remaining_time / long_duration)
  short_used + long_used

/-- Theorem stating the maximum number of songs that can be played in 3 hours -/
theorem max_songs_in_three_hours :
  max_songs_played 50 50 3 5 180 = 56 := by
  sorry

end max_songs_in_three_hours_l111_11142


namespace factor_expression_l111_11121

theorem factor_expression (x : ℝ) : 60 * x^2 + 45 * x = 15 * x * (4 * x + 3) := by
  sorry

end factor_expression_l111_11121


namespace solution_of_linear_equation_l111_11135

theorem solution_of_linear_equation :
  let f : ℝ → ℝ := λ x => x + 2
  f (-2) = 0 :=
by
  sorry

end solution_of_linear_equation_l111_11135


namespace max_value_negative_x_min_value_greater_than_negative_one_l111_11168

-- Problem 1
theorem max_value_negative_x (x : ℝ) (hx : x < 0) :
  (x^2 + x + 1) / x ≤ -1 :=
by sorry

-- Problem 2
theorem min_value_greater_than_negative_one (x : ℝ) (hx : x > -1) :
  ((x + 5) * (x + 2)) / (x + 1) ≥ 9 :=
by sorry

end max_value_negative_x_min_value_greater_than_negative_one_l111_11168


namespace warehouse_total_boxes_l111_11184

/-- Represents the number of boxes in each warehouse --/
structure Warehouses where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ

/-- Conditions for the warehouse problem --/
def warehouseConditions (w : Warehouses) : Prop :=
  ∃ x : ℕ,
    w.A = x ∧
    w.B = 3 * x ∧
    w.C = (3 * x) / 2 + 100 ∧
    w.D = 3 * x + 150 ∧
    w.E = 4 * x - 50 ∧
    w.B = w.E + 300

/-- The theorem to be proved --/
theorem warehouse_total_boxes (w : Warehouses) :
  warehouseConditions w → w.A + w.B + w.C + w.D + w.E = 4575 := by
  sorry


end warehouse_total_boxes_l111_11184


namespace seashell_solution_l111_11162

/-- The number of seashells found by Mary, Jessica, and Kevin -/
def seashell_problem (mary_shells jessica_shells : ℕ) (kevin_multiplier : ℕ) : Prop :=
  let kevin_shells := kevin_multiplier * mary_shells
  mary_shells + jessica_shells + kevin_shells = 113

/-- Theorem stating the solution to the seashell problem -/
theorem seashell_solution : seashell_problem 18 41 3 := by
  sorry

end seashell_solution_l111_11162


namespace algebra_sum_is_5_l111_11128

def letter_value (n : ℕ) : ℤ :=
  match n % 10 with
  | 1 => 2
  | 2 => 3
  | 3 => 2
  | 4 => 1
  | 5 => 0
  | 6 => -1
  | 7 => -2
  | 8 => -3
  | 9 => -2
  | 0 => -1
  | _ => 0  -- This case should never occur

def alphabet_position (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'l' => 12
  | 'g' => 7
  | 'e' => 5
  | 'b' => 2
  | 'r' => 18
  | _ => 0  -- This case should never occur for valid input

theorem algebra_sum_is_5 :
  (letter_value (alphabet_position 'a') +
   letter_value (alphabet_position 'l') +
   letter_value (alphabet_position 'g') +
   letter_value (alphabet_position 'e') +
   letter_value (alphabet_position 'b') +
   letter_value (alphabet_position 'r') +
   letter_value (alphabet_position 'a')) = 5 := by
  sorry

end algebra_sum_is_5_l111_11128


namespace unique_solution_mn_l111_11100

theorem unique_solution_mn : ∀ m n : ℕ, 
  m < n → 
  (∃ k : ℕ, m^2 + 1 = k * n) → 
  (∃ l : ℕ, n^2 + 1 = l * m) → 
  m = 1 ∧ n = 1 := by
sorry

end unique_solution_mn_l111_11100


namespace vertex_of_quadratic_l111_11199

/-- The quadratic function f(x) = 2 - (x+1)^2 -/
def f (x : ℝ) : ℝ := 2 - (x + 1)^2

/-- The vertex of a quadratic function -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem: The vertex of f(x) = 2 - (x+1)^2 is at (-1, 2) -/
theorem vertex_of_quadratic : 
  ∃ (v : Vertex), v.x = -1 ∧ v.y = 2 ∧ 
  ∀ (x : ℝ), f x ≤ f v.x := by
  sorry

end vertex_of_quadratic_l111_11199


namespace paperclips_exceed_300_l111_11161

def paperclips (k : ℕ) : ℕ := 5 * 3^k

theorem paperclips_exceed_300 : 
  (∀ n < 4, paperclips n ≤ 300) ∧ paperclips 4 > 300 := by sorry

end paperclips_exceed_300_l111_11161


namespace polynomial_expansion_l111_11182

-- Define the polynomials
def p (x : ℝ) : ℝ := 3*x^2 - 4*x + 3
def q (x : ℝ) : ℝ := -2*x^2 + 3*x - 4

-- State the theorem
theorem polynomial_expansion :
  ∀ x : ℝ, p x * q x = -6*x^4 + 17*x^3 - 30*x^2 + 25*x - 12 := by
  sorry

end polynomial_expansion_l111_11182


namespace range_of_3a_minus_2b_l111_11159

theorem range_of_3a_minus_2b (a b : ℝ) 
  (h1 : 1 ≤ a - b) (h2 : a - b ≤ 2) 
  (h3 : 2 ≤ a + b) (h4 : a + b ≤ 4) : 
  ∃ (x : ℝ), (7/2 ≤ x ∧ x ≤ 7) ∧ (∃ (a' b' : ℝ), 
    (1 ≤ a' - b' ∧ a' - b' ≤ 2) ∧ 
    (2 ≤ a' + b' ∧ a' + b' ≤ 4) ∧ 
    (3 * a' - 2 * b' = x)) ∧
  (∀ (y : ℝ), (∃ (a'' b'' : ℝ), 
    (1 ≤ a'' - b'' ∧ a'' - b'' ≤ 2) ∧ 
    (2 ≤ a'' + b'' ∧ a'' + b'' ≤ 4) ∧ 
    (3 * a'' - 2 * b'' = y)) → 
    (7/2 ≤ y ∧ y ≤ 7)) := by
  sorry


end range_of_3a_minus_2b_l111_11159


namespace infinite_pairs_divisibility_l111_11166

theorem infinite_pairs_divisibility (m : ℕ) (h_m_even : Even m) (h_m_ge_2 : m ≥ 2) :
  ∃ n : ℕ, n = m + 1 ∧ 
    n ≥ 2 ∧ 
    (m^m - 1) % n = 0 ∧ 
    (n^n - 1) % m = 0 := by
  sorry

end infinite_pairs_divisibility_l111_11166


namespace intersection_equals_closed_open_interval_l111_11129

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x : ℝ | x ≥ 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_closed_open_interval :
  A_intersect_B = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

end intersection_equals_closed_open_interval_l111_11129


namespace sequence_existence_l111_11130

theorem sequence_existence (n : ℕ) (hn : n ≥ 3) :
  (∃ (a : ℕ → ℝ), 
    (a 1 = a (n + 1)) ∧ 
    (a 2 = a (n + 2)) ∧ 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i * a (i + 1) + 1 = a (i + 2)))
  ↔ 
  (∃ k : ℕ, n = 3 * k) :=
by sorry

end sequence_existence_l111_11130


namespace derivative_of_even_is_odd_l111_11170

/-- If a real-valued function is even, then its derivative is odd. -/
theorem derivative_of_even_is_odd (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_even : ∀ x, f (-x) = f x) :
  ∀ x, deriv f (-x) = -deriv f x := by sorry

end derivative_of_even_is_odd_l111_11170


namespace parabola_point_distance_to_focus_l111_11124

theorem parabola_point_distance_to_focus (x y : ℝ) : 
  y^2 = 4*x →  -- Point A(x, y) is on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 36 →  -- Distance from A to focus (1, 0) is 6
  x = 7 := by sorry

end parabola_point_distance_to_focus_l111_11124


namespace cistern_specific_area_l111_11126

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem stating that a cistern with given dimensions has a specific wet surface area -/
theorem cistern_specific_area :
  cistern_wet_surface_area 12 14 1.25 = 233 := by
  sorry

end cistern_specific_area_l111_11126


namespace negative_two_triangle_five_l111_11167

/-- Definition of the triangle operation for rational numbers -/
def triangle (a b : ℚ) : ℚ := a * b + b - a

/-- Theorem stating that (-2) triangle 5 equals -3 -/
theorem negative_two_triangle_five : triangle (-2) 5 = -3 := by
  sorry

end negative_two_triangle_five_l111_11167


namespace root_in_interval_iff_a_in_range_l111_11134

/-- The function f(x) = x^2 - ax + 1 has a root in the interval (1/2, 3) if and only if a ∈ [2, 10/3) -/
theorem root_in_interval_iff_a_in_range (a : ℝ) : 
  (∃ x : ℝ, 1/2 < x ∧ x < 3 ∧ x^2 - a*x + 1 = 0) ↔ 2 ≤ a ∧ a < 10/3 := by
sorry

end root_in_interval_iff_a_in_range_l111_11134


namespace intersection_and_parallel_line_l111_11132

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define point P as the intersection of l₁ and l₂
def P : ℝ × ℝ := (-2, 2)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 3 * x - 2 * y + 10 = 0

-- Define the two possible perpendicular lines
def perp_line₁ (x y : ℝ) : Prop := 4 * x - 3 * y = 0
def perp_line₂ (x y : ℝ) : Prop := x - 2 * y = 0

theorem intersection_and_parallel_line :
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧
  parallel_line P.1 P.2 ∧
  (∃ (x y : ℝ), parallel_line x y ∧ 3 * x - 2 * y + 4 = 0) ∧
  (perp_line₁ 0 0 ∧ (∃ (x y : ℝ), perp_line₁ x y ∧ l₁ x y ∧ l₂ x y) ∨
   perp_line₂ 0 0 ∧ (∃ (x y : ℝ), perp_line₂ x y ∧ l₁ x y ∧ l₂ x y)) :=
by sorry

end intersection_and_parallel_line_l111_11132


namespace equality_and_inequality_proof_l111_11150

theorem equality_and_inequality_proof (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_eq : (3 : ℝ)^x = (4 : ℝ)^y ∧ (4 : ℝ)^y = (6 : ℝ)^z) : 
  (1 / z - 1 / x = 1 / (2 * y)) ∧ (3 * x < 4 * y ∧ 4 * y < 6 * z) := by
sorry

end equality_and_inequality_proof_l111_11150


namespace factory_output_doubling_time_l111_11120

theorem factory_output_doubling_time (growth_rate : ℝ) (doubling_time : ℝ) :
  growth_rate = 0.1 →
  (1 + growth_rate) ^ doubling_time = 2 :=
by
  sorry

end factory_output_doubling_time_l111_11120


namespace two_to_700_gt_five_to_300_l111_11104

theorem two_to_700_gt_five_to_300 : 2^700 > 5^300 := by
  sorry

end two_to_700_gt_five_to_300_l111_11104


namespace calculation_proof_inequality_system_solution_l111_11108

-- Part 1
theorem calculation_proof : (1 * (1/2)⁻¹) + 2 * Real.cos (π/4) - Real.sqrt 8 + |1 - Real.sqrt 2| = 1 := by sorry

-- Part 2
theorem inequality_system_solution :
  ∀ x : ℝ, (x/2 + 1 > 0 ∧ 2*(x-1) + 3 ≥ 3*x) ↔ (-2 < x ∧ x ≤ 1) := by sorry

end calculation_proof_inequality_system_solution_l111_11108


namespace vehicle_ownership_l111_11109

theorem vehicle_ownership (total_adults : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) (bicycle_owners : ℕ)
  (car_and_motorcycle : ℕ) (motorcycle_and_bicycle : ℕ) (car_and_bicycle : ℕ)
  (h1 : total_adults = 500)
  (h2 : car_owners = 400)
  (h3 : motorcycle_owners = 200)
  (h4 : bicycle_owners = 150)
  (h5 : car_and_motorcycle = 100)
  (h6 : motorcycle_and_bicycle = 50)
  (h7 : car_and_bicycle = 30)
  (h8 : total_adults ≤ car_owners + motorcycle_owners + bicycle_owners - car_and_motorcycle - motorcycle_and_bicycle - car_and_bicycle) :
  car_owners - car_and_motorcycle - car_and_bicycle = 270 := by
  sorry

end vehicle_ownership_l111_11109


namespace inequality_reverse_l111_11181

theorem inequality_reverse (a b : ℝ) (h : a > b) : -4 * a < -4 * b := by
  sorry

end inequality_reverse_l111_11181


namespace smallest_n_guarantee_same_length_l111_11143

/-- The number of vertices in the regular polygon -/
def n : ℕ := 2017

/-- The number of distinct diagonal lengths from a single vertex -/
def distinct_lengths : ℕ := (n - 3) / 2

/-- The smallest number of diagonals to guarantee two of the same length -/
def smallest_n : ℕ := distinct_lengths + 1

theorem smallest_n_guarantee_same_length :
  smallest_n = 1008 := by sorry

end smallest_n_guarantee_same_length_l111_11143


namespace isosceles_triangle_perimeter_l111_11187

/-- An isosceles triangle with side lengths 5 and 2 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  isIsosceles : a = b
  sideLength1 : a = 5
  sideLength2 : b = 5
  base : ℝ
  baseLength : base = 2

/-- The perimeter of the isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.a + t.b + t.base

theorem isosceles_triangle_perimeter :
  ∀ (t : IsoscelesTriangle), perimeter t = 12 := by
  sorry

end isosceles_triangle_perimeter_l111_11187


namespace square_perimeter_l111_11177

/-- The perimeter of a square is 160 cm, given that its area is five times
    the area of a rectangle with dimensions 32 cm * 10 cm. -/
theorem square_perimeter (square_area rectangle_area : ℝ) : 
  square_area = 5 * rectangle_area →
  rectangle_area = 32 * 10 →
  4 * Real.sqrt square_area = 160 := by
  sorry

end square_perimeter_l111_11177


namespace students_per_class_l111_11119

/-- Given that John buys index cards for his students, this theorem proves
    the number of students in each class. -/
theorem students_per_class
  (total_packs : ℕ)
  (num_classes : ℕ)
  (packs_per_student : ℕ)
  (h1 : total_packs = 360)
  (h2 : num_classes = 6)
  (h3 : packs_per_student = 2) :
  total_packs / (num_classes * packs_per_student) = 30 :=
by sorry

end students_per_class_l111_11119


namespace parallel_iff_a_eq_3_l111_11136

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (m1 n1 c1 m2 n2 c2 : ℝ) : Prop :=
  m1 * n2 = m2 * n1

/-- The first line: ax + 2y + 3a = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 3 * a = 0

/-- The second line: 3x + (a-1)y = a-7 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + (a - 1) * y = a - 7

theorem parallel_iff_a_eq_3 :
  ∀ a : ℝ, are_parallel a 2 (3*a) 3 (a-1) (a-7) ↔ a = 3 :=
by sorry

end parallel_iff_a_eq_3_l111_11136


namespace max_m_value_l111_11173

/-- The maximum value of m for which f and g satisfy the given conditions -/
theorem max_m_value : ∀ (n : ℝ), 
  (∀ (m : ℝ), (∀ (t : ℝ), (t^2 + m*t + n^2 ≥ 0) ∨ (t^2 + (m+2)*t + n^2 + m + 1 ≥ 0)) → m ≤ 1) ∧
  (∃ (m : ℝ), m = 1 ∧ ∀ (t : ℝ), (t^2 + m*t + n^2 ≥ 0) ∨ (t^2 + (m+2)*t + n^2 + m + 1 ≥ 0)) :=
by sorry

end max_m_value_l111_11173


namespace card_draw_probability_l111_11163

def standard_deck := 52
def face_cards := 12
def hearts := 13
def tens := 4

theorem card_draw_probability : 
  let p1 := face_cards / standard_deck
  let p2 := hearts / (standard_deck - 1)
  let p3 := tens / (standard_deck - 2)
  p1 * p2 * p3 = 1 / 217 := by
  sorry

end card_draw_probability_l111_11163


namespace min_omega_value_l111_11106

theorem min_omega_value (y : ℝ → ℝ) (ω : ℝ) :
  (∀ x, y x = 2 * Real.sin (ω * x + π / 3)) →
  ω > 0 →
  (∀ x, y x = y (x - π / 3)) →
  (∃ k : ℕ, ω = 6 * k) →
  6 ≤ ω :=
by sorry

end min_omega_value_l111_11106


namespace union_of_M_and_N_l111_11197

def M : Set ℝ := {x | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x | x < -5 ∨ x > 5}

theorem union_of_M_and_N : M ∪ N = {x | x < -5 ∨ x > -3} := by sorry

end union_of_M_and_N_l111_11197


namespace square_area_error_l111_11176

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := 1.05 * s
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 10.25 := by
sorry

end square_area_error_l111_11176


namespace rectangle_width_l111_11152

/-- Given a rectangle with length 3 inches and unknown width, and a square with width 5 inches,
    if the difference in area between the square and the rectangle is 7 square inches,
    then the width of the rectangle is 6 inches. -/
theorem rectangle_width (w : ℝ) : 
  (5 * 5 : ℝ) - (3 * w) = 7 → w = 6 := by sorry

end rectangle_width_l111_11152


namespace sum_of_max_min_g_l111_11160

def g (x : ℝ) : ℝ := |x - 5| + |x - 7| - |2*x - 12| + |3*x - 21|

theorem sum_of_max_min_g :
  ∃ (max min : ℝ),
    (∀ x : ℝ, 5 ≤ x → x ≤ 10 → g x ≤ max) ∧
    (∃ x : ℝ, 5 ≤ x ∧ x ≤ 10 ∧ g x = max) ∧
    (∀ x : ℝ, 5 ≤ x → x ≤ 10 → min ≤ g x) ∧
    (∃ x : ℝ, 5 ≤ x ∧ x ≤ 10 ∧ g x = min) ∧
    max + min = 11 := by
  sorry

end sum_of_max_min_g_l111_11160


namespace min_value_expression_l111_11189

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (9 * r) / (3 * p + 2 * q) + (9 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) ≥ 2 ∧
  ((9 * r) / (3 * p + 2 * q) + (9 * p) / (2 * q + 3 * r) + (2 * q) / (p + r) = 2 ↔ 3 * p = 2 * q ∧ 2 * q = 3 * r) :=
by sorry

end min_value_expression_l111_11189


namespace hemisphere_surface_area_l111_11137

theorem hemisphere_surface_area (diameter : ℝ) (h : diameter = 12) :
  let radius := diameter / 2
  let curved_surface_area := 2 * π * radius^2
  let base_area := π * radius^2
  curved_surface_area + base_area = 108 * π :=
by sorry

end hemisphere_surface_area_l111_11137


namespace max_sum_xyz_l111_11157

theorem max_sum_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
  16 * x₀ * y₀ * z₀ = (x₀ + y₀)^2 * (x₀ + z₀)^2 ∧ x₀ + y₀ + z₀ = 4 :=
by sorry

end max_sum_xyz_l111_11157


namespace complex_fraction_simplification_l111_11165

/-- Given that i is the imaginary unit, prove that (3 + i) / (1 + 2*i) = 1 - i -/
theorem complex_fraction_simplification :
  (3 + I : ℂ) / (1 + 2*I) = 1 - I :=
by sorry

end complex_fraction_simplification_l111_11165


namespace hyperbola_other_asymptote_l111_11194

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One of the asymptotes of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- The x-coordinate of the foci -/
  foci_x : ℝ

/-- The other asymptote of the hyperbola -/
def otherAsymptote (h : Hyperbola) : ℝ → ℝ := 
  fun x => 2 * x + 16

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x => -2 * x) 
  (h2 : h.foci_x = -4) : 
  otherAsymptote h = fun x => 2 * x + 16 := by
  sorry

end hyperbola_other_asymptote_l111_11194


namespace largest_reciprocal_l111_11141

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 1/4 → b = 3/8 → c = 0 → d = -2 → e = 4 → 
  (1/a > 1/b ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

#check largest_reciprocal

end largest_reciprocal_l111_11141


namespace fuel_station_problem_l111_11172

/-- Fuel station problem -/
theorem fuel_station_problem 
  (service_cost : ℝ) 
  (fuel_cost_per_liter : ℝ) 
  (total_cost : ℝ) 
  (num_minivans : ℕ) 
  (minivan_tank : ℝ) 
  (truck_tank : ℝ) 
  (h1 : service_cost = 2.20)
  (h2 : fuel_cost_per_liter = 0.70)
  (h3 : total_cost = 395.4)
  (h4 : num_minivans = 4)
  (h5 : minivan_tank = 65)
  (h6 : truck_tank = minivan_tank * 2.2)
  : ∃ (num_trucks : ℕ), num_trucks = 2 ∧ 
    total_cost = (num_minivans * (service_cost + fuel_cost_per_liter * minivan_tank)) + 
                 (num_trucks : ℝ) * (service_cost + fuel_cost_per_liter * truck_tank) :=
by sorry


end fuel_station_problem_l111_11172


namespace intersection_M_N_l111_11138

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | ∃ y, y = Real.log (x - x^2)}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by sorry

end intersection_M_N_l111_11138


namespace two_digit_number_puzzle_l111_11169

theorem two_digit_number_puzzle :
  ∃! n : ℕ, 
    10 ≤ n ∧ n < 100 ∧  -- n is a two-digit number
    (n / 10 = 2 * (n % 10)) ∧  -- tens digit is twice the units digit
    (n - ((n % 10) * 10 + (n / 10)) = 36)  -- swapping digits results in 36 less
  :=
by sorry

end two_digit_number_puzzle_l111_11169


namespace male_salmon_count_l111_11114

theorem male_salmon_count (female_salmon : ℕ) (total_salmon : ℕ) 
  (h1 : female_salmon = 259378) 
  (h2 : total_salmon = 971639) : 
  total_salmon - female_salmon = 712261 := by
  sorry

end male_salmon_count_l111_11114


namespace log_78903_between_consecutive_integers_l111_11198

theorem log_78903_between_consecutive_integers :
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 78903 / Real.log 10 ∧ Real.log 78903 / Real.log 10 < (d : ℝ) → c + d = 9 := by
  sorry

end log_78903_between_consecutive_integers_l111_11198
