import Mathlib

namespace age_difference_l1415_141524

theorem age_difference (jack_age bill_age : ℕ) : 
  jack_age = 2 * bill_age → 
  (jack_age + 8) = 3 * (bill_age + 8) → 
  jack_age - bill_age = 16 := by
sorry

end age_difference_l1415_141524


namespace overlap_area_theorem_l1415_141593

-- Define the square ABCD
def square_side : ℝ := 8

-- Define the rectangle WXYZ
def rect_length : ℝ := 12
def rect_width : ℝ := 8

-- Define the theorem
theorem overlap_area_theorem (shaded_area : ℝ) (AP : ℝ) :
  -- Conditions
  shaded_area = (rect_length * rect_width) / 2 →
  shaded_area = (square_side - AP) * square_side →
  -- Conclusion
  AP = 2 := by
  sorry

end overlap_area_theorem_l1415_141593


namespace geometric_with_arithmetic_subsequence_l1415_141517

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define an arithmetic subsequence
def arithmetic_subsequence (a : ℕ → ℝ) (sub : ℕ → ℕ) (d : ℝ) : Prop :=
  ∀ k : ℕ, a (sub (k + 1)) = a (sub k) + d

-- Main theorem
theorem geometric_with_arithmetic_subsequence
  (a : ℕ → ℝ) (q : ℝ) (sub : ℕ → ℕ) (d : ℝ) :
  geometric_sequence a q →
  q ≠ 1 →
  arithmetic_subsequence a sub d →
  (∀ k : ℕ, sub (k + 1) > sub k) →
  q = -1 :=
sorry

end geometric_with_arithmetic_subsequence_l1415_141517


namespace range_of_a_range_of_m_l1415_141562

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem for part I
theorem range_of_a : 
  {a : ℝ | ∃ x, f x < |1 - 2*a|} = {a : ℝ | a < -3/2 ∨ a > 5/2} :=
sorry

-- Theorem for part II
theorem range_of_m :
  {m : ℝ | ∃ t, t^2 - 2*Real.sqrt 6*t + f m = 0} = {m : ℝ | -1 ≤ m ∧ m ≤ 2} :=
sorry

end range_of_a_range_of_m_l1415_141562


namespace certain_number_is_36_l1415_141538

theorem certain_number_is_36 : ∃ x : ℝ, 
  ((((x + 10) * 2) / 2) - 2) = 88 / 2 ∧ x = 36 := by
  sorry

end certain_number_is_36_l1415_141538


namespace total_seeds_planted_l1415_141586

/-- The number of tomato seeds planted by Mike and Ted -/
def total_seeds (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ) : ℕ :=
  mike_morning + mike_afternoon + ted_morning + ted_afternoon

/-- Theorem stating the total number of tomato seeds planted by Mike and Ted -/
theorem total_seeds_planted : 
  ∀ (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ),
  mike_morning = 50 →
  ted_morning = 2 * mike_morning →
  mike_afternoon = 60 →
  ted_afternoon = mike_afternoon - 20 →
  total_seeds mike_morning mike_afternoon ted_morning ted_afternoon = 250 :=
by
  sorry


end total_seeds_planted_l1415_141586


namespace rectangle_triangle_count_l1415_141598

-- Define the structure of our rectangle
structure DividedRectangle where
  horizontal_sections : Nat
  vertical_sections : Nat
  (h_pos : horizontal_sections > 0)
  (v_pos : vertical_sections > 0)

-- Function to calculate the number of triangles
def count_triangles (rect : DividedRectangle) : Nat :=
  sorry

-- Theorem statement
theorem rectangle_triangle_count :
  ∃ (rect : DividedRectangle),
    rect.horizontal_sections = 3 ∧
    rect.vertical_sections = 4 ∧
    count_triangles rect = 148 :=
  sorry

end rectangle_triangle_count_l1415_141598


namespace sin_x_sin_2x_integral_l1415_141569

theorem sin_x_sin_2x_integral (x : ℝ) :
  deriv (λ x => (1/2) * Real.sin x - (1/6) * Real.sin (3*x)) x = Real.sin x * Real.sin (2*x) := by
  sorry

end sin_x_sin_2x_integral_l1415_141569


namespace certain_number_minus_two_l1415_141559

theorem certain_number_minus_two (x : ℝ) (h : 6 - x = 2) : x - 2 = 2 := by
  sorry

end certain_number_minus_two_l1415_141559


namespace sum_of_abs_first_six_terms_l1415_141522

def sequence_a (n : ℕ) : ℤ :=
  -5 + 2 * (n - 1)

theorem sum_of_abs_first_six_terms :
  (∀ n, sequence_a (n + 1) - sequence_a n = 2) →
  sequence_a 1 = -5 →
  (Finset.range 6).sum (fun i => |sequence_a (i + 1)|) = 18 := by
sorry

end sum_of_abs_first_six_terms_l1415_141522


namespace championship_winner_l1415_141560

-- Define the teams
inductive Team : Type
| A | B | C | D

-- Define the positions
inductive Position : Type
| First | Second | Third | Fourth

-- Define a prediction as a pair of (Team, Position)
def Prediction := Team × Position

-- Define the predictions made by each person
def WangPredictions : Prediction × Prediction := ((Team.D, Position.First), (Team.B, Position.Second))
def LiPredictions : Prediction × Prediction := ((Team.A, Position.Second), (Team.C, Position.Fourth))
def ZhangPredictions : Prediction × Prediction := ((Team.C, Position.Third), (Team.D, Position.Second))

-- Define a function to check if a prediction is correct
def isPredictionCorrect (prediction : Prediction) (result : Team → Position) : Prop :=
  result prediction.1 = prediction.2

-- Define the theorem
theorem championship_winner (result : Team → Position) : 
  (isPredictionCorrect WangPredictions.1 result ≠ isPredictionCorrect WangPredictions.2 result) ∧
  (isPredictionCorrect LiPredictions.1 result ≠ isPredictionCorrect LiPredictions.2 result) ∧
  (isPredictionCorrect ZhangPredictions.1 result ≠ isPredictionCorrect ZhangPredictions.2 result) →
  result Team.D = Position.First :=
by
  sorry

end championship_winner_l1415_141560


namespace isosceles_right_triangle_l1415_141518

/-- Given a triangle ABC where b = a * sin(C) and c = a * cos(B), prove that ABC is an isosceles right triangle -/
theorem isosceles_right_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : b = a * Real.sin C) 
  (h2 : c = a * Real.cos B) 
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h4 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h5 : A + B + C = π) : 
  A = π / 2 ∧ b = c := by
  sorry

end isosceles_right_triangle_l1415_141518


namespace three_digit_numbers_with_eight_or_nine_l1415_141549

theorem three_digit_numbers_with_eight_or_nine (total_three_digit : ℕ) (without_eight_or_nine : ℕ) :
  total_three_digit = 900 →
  without_eight_or_nine = 448 →
  total_three_digit - without_eight_or_nine = 452 :=
by sorry

end three_digit_numbers_with_eight_or_nine_l1415_141549


namespace tan_x_value_l1415_141572

theorem tan_x_value (x : Real) (h : Real.tan (x + π/4) = 2) : Real.tan x = 1/3 := by
  sorry

end tan_x_value_l1415_141572


namespace complex_cube_simplification_l1415_141547

theorem complex_cube_simplification :
  let i : ℂ := Complex.I
  (1 + Real.sqrt 3 * i)^3 = -8 := by
  sorry

end complex_cube_simplification_l1415_141547


namespace fraction_irreducible_l1415_141509

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end fraction_irreducible_l1415_141509


namespace complex_number_properties_l1415_141551

theorem complex_number_properties (z : ℂ) (h : z - 2*I = z*I + 4) :
  z = 1 + 3*I ∧ Complex.abs z = Real.sqrt 10 ∧ ((z - 1) / 3)^2023 = -I := by
  sorry

end complex_number_properties_l1415_141551


namespace new_boy_age_l1415_141504

theorem new_boy_age (initial_size : Nat) (initial_avg : Nat) (time_passed : Nat) (new_size : Nat) :
  initial_size = 6 →
  initial_avg = 19 →
  time_passed = 3 →
  new_size = 7 →
  (initial_size * initial_avg + initial_size * time_passed + 1) / new_size = initial_avg →
  1 = 1 := by
  sorry

end new_boy_age_l1415_141504


namespace complex_sum_equality_l1415_141590

theorem complex_sum_equality :
  let B : ℂ := 3 - 2*I
  let Q : ℂ := 1 + 3*I
  let R : ℂ := -2 + 4*I
  let T : ℂ := 5 - 3*I
  B + Q + R + T = 7 + 2*I :=
by
  sorry

end complex_sum_equality_l1415_141590


namespace consecutive_integers_square_difference_l1415_141501

theorem consecutive_integers_square_difference (n : ℕ) : 
  n > 0 ∧ 
  n + (n + 1) < 150 ∧ 
  (n + 1)^2 - n^2 = 149 := by
sorry

end consecutive_integers_square_difference_l1415_141501


namespace function_value_at_point_l1415_141523

theorem function_value_at_point (h : ℝ → ℝ) (b : ℝ) :
  (∀ x, h x = 4 * x - 5) →
  h b = 1 ↔ b = 3 / 2 := by
  sorry

end function_value_at_point_l1415_141523


namespace sine_in_triangle_l1415_141553

theorem sine_in_triangle (a b : ℝ) (A B : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : Real.sin A = 3/5) :
  Real.sin B = 4/5 := by
  sorry

end sine_in_triangle_l1415_141553


namespace sqrt_49_times_sqrt_25_l1415_141548

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_49_times_sqrt_25_l1415_141548


namespace solve_equation_l1415_141574

theorem solve_equation (x : ℝ) (n : ℝ) (expr : ℝ → ℝ) : 
  x = 1 → 
  n = 4 * x → 
  2 * x * expr x = 10 → 
  n = 4 := by sorry

end solve_equation_l1415_141574


namespace geometric_sequence_seventh_term_l1415_141542

theorem geometric_sequence_seventh_term 
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a = 2) -- first term is 2
  (h2 : a * r^8 = 32) -- last term (9th term) is 32
  : a * r^6 = 128 := by -- seventh term is 128
sorry

end geometric_sequence_seventh_term_l1415_141542


namespace final_game_score_l1415_141516

/-- Represents the points scored by each player in the basketball game -/
structure TeamPoints where
  bailey : ℕ
  michiko : ℕ
  akiko : ℕ
  chandra : ℕ

/-- Calculates the total points scored by the team -/
def total_points (t : TeamPoints) : ℕ :=
  t.bailey + t.michiko + t.akiko + t.chandra

/-- Theorem stating the total points scored by the team under given conditions -/
theorem final_game_score (t : TeamPoints) 
  (h1 : t.bailey = 14)
  (h2 : t.michiko = t.bailey / 2)
  (h3 : t.akiko = t.michiko + 4)
  (h4 : t.chandra = 2 * t.akiko) :
  total_points t = 54 := by
  sorry

end final_game_score_l1415_141516


namespace midpoint_sum_equals_vertex_sum_l1415_141514

theorem midpoint_sum_equals_vertex_sum (a b c : ℝ) :
  let vertex_sum := a + b + c
  let midpoint_sum := (a + b) / 2 + (a + c) / 2 + (b + c) / 2
  vertex_sum = midpoint_sum := by
  sorry

end midpoint_sum_equals_vertex_sum_l1415_141514


namespace remainder_problem_l1415_141565

theorem remainder_problem (N : ℕ) (k : ℕ) (h : N = 35 * k + 25) :
  N % 15 = 10 := by
sorry

end remainder_problem_l1415_141565


namespace arithmetic_correctness_l1415_141537

theorem arithmetic_correctness : 
  ((-4) + (-5) = -9) ∧ 
  (4 / (-2) = -2) ∧ 
  (-5 - (-6) ≠ 11) ∧ 
  (-2 * (-10) ≠ -20) := by
  sorry

end arithmetic_correctness_l1415_141537


namespace pastries_sold_equals_initial_l1415_141564

/-- Represents the number of pastries and cakes made and sold by a baker -/
structure BakerInventory where
  initialPastries : ℕ
  initialCakes : ℕ
  cakesSold : ℕ
  cakesRemaining : ℕ

/-- Theorem stating that the number of pastries sold is equal to the initial number of pastries made -/
theorem pastries_sold_equals_initial (inventory : BakerInventory)
  (h1 : inventory.initialPastries = 61)
  (h2 : inventory.initialCakes = 167)
  (h3 : inventory.cakesSold = 108)
  (h4 : inventory.cakesRemaining = 59)
  : inventory.initialPastries = inventory.initialPastries := by
  sorry

end pastries_sold_equals_initial_l1415_141564


namespace early_bird_dinner_bill_l1415_141567

def early_bird_dinner (curtis_steak rob_steak curtis_side rob_side curtis_drink rob_drink : ℝ)
  (discount_rate tax_rate tip_rate : ℝ) : ℝ :=
  let discounted_curtis_steak := curtis_steak * discount_rate
  let discounted_rob_steak := rob_steak * discount_rate
  let curtis_total := discounted_curtis_steak + curtis_side + curtis_drink
  let rob_total := discounted_rob_steak + rob_side + rob_drink
  let combined_total := curtis_total + rob_total
  let tax := combined_total * tax_rate
  let tip := combined_total * tip_rate
  combined_total + tax + tip

theorem early_bird_dinner_bill : 
  early_bird_dinner 16 18 6 7 3 3.5 0.5 0.07 0.2 = 46.36 := by
  sorry

end early_bird_dinner_bill_l1415_141567


namespace incorrect_division_l1415_141527

theorem incorrect_division (D : ℕ) (h : D / 36 = 48) : D / 72 = 24 := by
  sorry

end incorrect_division_l1415_141527


namespace marathon_volunteer_assignment_l1415_141506

def number_of_students : ℕ := 5
def number_of_tasks : ℕ := 4
def number_of_students_who_can_drive : ℕ := 3

theorem marathon_volunteer_assignment :
  let total_arrangements := 
    (Nat.choose number_of_students_who_can_drive 1 * 
     Nat.choose (number_of_students - 1) 2 * 
     Nat.factorial 3) +
    (Nat.choose number_of_students_who_can_drive 2 * 
     Nat.factorial 3)
  total_arrangements = 
    Nat.choose number_of_students_who_can_drive 1 * 
    Nat.choose number_of_students 2 * 
    Nat.factorial 3 +
    Nat.choose number_of_students_who_can_drive 2 * 
    Nat.factorial 3 := by
  sorry

end marathon_volunteer_assignment_l1415_141506


namespace concert_total_cost_l1415_141573

/-- Calculate the total cost of a concert for two people -/
theorem concert_total_cost
  (ticket_price : ℚ)
  (num_people : ℕ)
  (processing_fee_rate : ℚ)
  (parking_fee : ℚ)
  (entrance_fee_per_person : ℚ)
  (refreshments_cost : ℚ)
  (tshirts_cost : ℚ)
  (h1 : ticket_price = 75)
  (h2 : num_people = 2)
  (h3 : processing_fee_rate = 0.15)
  (h4 : parking_fee = 10)
  (h5 : entrance_fee_per_person = 5)
  (h6 : refreshments_cost = 20)
  (h7 : tshirts_cost = 40) :
  let total_ticket_cost := ticket_price * num_people
  let processing_fee := total_ticket_cost * processing_fee_rate
  let entrance_fee_total := entrance_fee_per_person * num_people
  ticket_price * num_people +
  total_ticket_cost * processing_fee_rate +
  parking_fee +
  entrance_fee_total +
  refreshments_cost +
  tshirts_cost = 252.5 := by
    sorry


end concert_total_cost_l1415_141573


namespace distribute_5_3_l1415_141533

/-- The number of ways to distribute n distinct objects into k identical containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 51 ways to distribute 5 distinct objects into 3 identical containers,
    allowing empty containers. -/
theorem distribute_5_3 : distribute 5 3 = 51 := by sorry

end distribute_5_3_l1415_141533


namespace permutations_of_eight_distinct_objects_l1415_141557

theorem permutations_of_eight_distinct_objects : 
  Nat.factorial 8 = 40320 := by sorry

end permutations_of_eight_distinct_objects_l1415_141557


namespace side_BC_equation_l1415_141535

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def altitude_from_AC : Line := { a := 2, b := -3, c := 1 }
def altitude_from_AB : Line := { a := 1, b := 1, c := -1 }

def vertex_A : ℝ × ℝ := (1, 2)

theorem side_BC_equation (t : Triangle) 
  (h1 : t.A = vertex_A)
  (h2 : altitude_from_AC.a * t.B.1 + altitude_from_AC.b * t.B.2 + altitude_from_AC.c = 0)
  (h3 : altitude_from_AC.a * t.C.1 + altitude_from_AC.b * t.C.2 + altitude_from_AC.c = 0)
  (h4 : altitude_from_AB.a * t.B.1 + altitude_from_AB.b * t.B.2 + altitude_from_AB.c = 0)
  (h5 : altitude_from_AB.a * t.C.1 + altitude_from_AB.b * t.C.2 + altitude_from_AB.c = 0) :
  ∃ (l : Line), l.a * t.B.1 + l.b * t.B.2 + l.c = 0 ∧
                l.a * t.C.1 + l.b * t.C.2 + l.c = 0 ∧
                l = { a := 2, b := 3, c := 7 } := by
  sorry

end side_BC_equation_l1415_141535


namespace x1_value_l1415_141546

theorem x1_value (x₁ x₂ x₃ : Real) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h2 : (1 - x₁^2) + (x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/4) :
  x₁ = Real.sqrt 15 / 4 := by
  sorry

end x1_value_l1415_141546


namespace positive_root_condition_negative_root_condition_zero_root_condition_l1415_141599

-- Define the equation ax = b - c
def equation (a b c x : ℝ) : Prop := a * x = b - c

-- Theorem for positive root condition
theorem positive_root_condition (a b c : ℝ) :
  (∃ x > 0, equation a b c x) ↔ (a > 0 ∧ b > c) ∨ (a < 0 ∧ b < c) :=
sorry

-- Theorem for negative root condition
theorem negative_root_condition (a b c : ℝ) :
  (∃ x < 0, equation a b c x) ↔ (a > 0 ∧ b < c) ∨ (a < 0 ∧ b > c) :=
sorry

-- Theorem for zero root condition
theorem zero_root_condition (a b c : ℝ) :
  (∃ x, x = 0 ∧ equation a b c x) ↔ (a ≠ 0 ∧ b = c) :=
sorry

end positive_root_condition_negative_root_condition_zero_root_condition_l1415_141599


namespace isosceles_triangle_l1415_141541

theorem isosceles_triangle (A B C : ℝ) (h₁ : A + B + C = π) 
  (h₂ : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) : 
  A = B :=
sorry

end isosceles_triangle_l1415_141541


namespace x_range_for_inequality_l1415_141554

theorem x_range_for_inequality (x : ℝ) :
  (∀ m ∈ Set.Icc (1/2 : ℝ) 3, x^2 + m*x + 4 > 2*m + 4*x) →
  x > 2 ∨ x < -1 :=
by sorry

end x_range_for_inequality_l1415_141554


namespace min_distance_circle_line_l1415_141525

theorem min_distance_circle_line : 
  let circle := {p : ℝ × ℝ | (p.1 + 2)^2 + (p.2 - 1)^2 = 4}
  let line := {p : ℝ × ℝ | 4 * p.1 - p.2 - 1 = 0}
  ∃ (min_dist : ℝ), 
    (∀ (p q : ℝ × ℝ), p ∈ circle → q ∈ line → min_dist ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    (∃ (p q : ℝ × ℝ), p ∈ circle ∧ q ∈ line ∧ min_dist = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    min_dist = (10 * Real.sqrt 17) / 17 - 2 := by
  sorry

end min_distance_circle_line_l1415_141525


namespace fiftieth_ring_squares_l1415_141519

/-- The number of unit squares in the nth ring of a square array with a center square,
    where each ring increases by 3 on each side. -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 8

/-- The 50th ring contains 408 unit squares -/
theorem fiftieth_ring_squares : ring_squares 50 = 408 := by
  sorry

#eval ring_squares 50  -- This will evaluate to 408

end fiftieth_ring_squares_l1415_141519


namespace jason_initial_quarters_l1415_141585

/-- The number of quarters Jason's dad gave him -/
def quarters_from_dad : ℕ := 25

/-- The total number of quarters Jason has now -/
def total_quarters_now : ℕ := 74

/-- The number of quarters Jason had initially -/
def initial_quarters : ℕ := total_quarters_now - quarters_from_dad

theorem jason_initial_quarters : initial_quarters = 49 := by
  sorry

end jason_initial_quarters_l1415_141585


namespace inscribed_circle_radius_l1415_141511

/-- The radius of the inscribed circle of a triangle with side lengths 5, 12, and 13 is 2 -/
theorem inscribed_circle_radius (a b c r : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) :
  r = (a + b - c) / 2 → r = 2 := by
  sorry

end inscribed_circle_radius_l1415_141511


namespace curve_properties_l1415_141536

-- Define the curve
def curve (x y : ℝ) : Prop := abs x + y^2 - 3*y = 0

-- Theorem for the axis of symmetry and range of y
theorem curve_properties :
  (∀ x y : ℝ, curve x y ↔ curve (-x) y) ∧
  (∀ y : ℝ, (∃ x : ℝ, curve x y) → 0 ≤ y ∧ y ≤ 3) :=
sorry

end curve_properties_l1415_141536


namespace dress_costs_sum_l1415_141581

/-- The cost of dresses for four ladies -/
def dress_costs (pauline_cost ida_cost jean_cost patty_cost : ℕ) : Prop :=
  pauline_cost = 30 ∧
  jean_cost = pauline_cost - 10 ∧
  ida_cost = jean_cost + 30 ∧
  patty_cost = ida_cost + 10

/-- The total cost of all dresses -/
def total_cost (pauline_cost ida_cost jean_cost patty_cost : ℕ) : ℕ :=
  pauline_cost + ida_cost + jean_cost + patty_cost

/-- Theorem: The total cost of all dresses is $160 -/
theorem dress_costs_sum :
  ∀ (pauline_cost ida_cost jean_cost patty_cost : ℕ),
  dress_costs pauline_cost ida_cost jean_cost patty_cost →
  total_cost pauline_cost ida_cost jean_cost patty_cost = 160 := by
  sorry

end dress_costs_sum_l1415_141581


namespace cube_root_equation_l1415_141503

theorem cube_root_equation (x : ℝ) : 
  (x * (x^5)^(1/4))^(1/3) = 5 ↔ x = 5 * 5^(1/3) :=
sorry

end cube_root_equation_l1415_141503


namespace first_digit_base_5_of_627_l1415_141576

theorem first_digit_base_5_of_627 :
  ∃ (d : ℕ) (r : ℕ), 627 = d * 5^4 + r ∧ d = 1 ∧ r < 5^4 := by
  sorry

end first_digit_base_5_of_627_l1415_141576


namespace car_speed_problem_l1415_141577

theorem car_speed_problem (total_distance : ℝ) (first_leg_distance : ℝ) (first_leg_speed : ℝ) (average_speed : ℝ) :
  total_distance = 320 →
  first_leg_distance = 160 →
  first_leg_speed = 75 →
  average_speed = 77.4193548387097 →
  let second_leg_distance := total_distance - first_leg_distance
  let total_time := total_distance / average_speed
  let first_leg_time := first_leg_distance / first_leg_speed
  let second_leg_time := total_time - first_leg_time
  let second_leg_speed := second_leg_distance / second_leg_time
  second_leg_speed = 80 := by
sorry

end car_speed_problem_l1415_141577


namespace tan_45_degrees_l1415_141566

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l1415_141566


namespace rex_driving_lessons_l1415_141529

/-- The number of hour-long lessons Rex wants to take before his test -/
def total_lessons : ℕ := 40

/-- The number of hours of lessons Rex takes per week -/
def hours_per_week : ℕ := 4

/-- The number of weeks Rex has already completed -/
def completed_weeks : ℕ := 6

/-- The number of additional weeks Rex needs to reach his goal -/
def additional_weeks : ℕ := 4

/-- Theorem stating that the total number of hour-long lessons Rex wants to take is 40 -/
theorem rex_driving_lessons :
  total_lessons = hours_per_week * (completed_weeks + additional_weeks) :=
by sorry

end rex_driving_lessons_l1415_141529


namespace max_value_sum_fractions_l1415_141589

theorem max_value_sum_fractions (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  (∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧ 
    (x * y) / (x + y) + (x * z) / (x + z) + (y * z) / (y + z) = 1) ∧
  (∀ (x y z : ℝ), x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 → 
    (x * y) / (x + y) + (x * z) / (x + z) + (y * z) / (y + z) ≤ 1) := by
  sorry

end max_value_sum_fractions_l1415_141589


namespace distance_difference_l1415_141550

/-- The walking speed of Taehyung in meters per minute -/
def taehyung_speed : ℕ := 114

/-- The walking speed of Minyoung in meters per minute -/
def minyoung_speed : ℕ := 79

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating the difference in distance walked by Taehyung and Minyoung in an hour -/
theorem distance_difference : 
  taehyung_speed * minutes_per_hour - minyoung_speed * minutes_per_hour = 2100 := by
  sorry


end distance_difference_l1415_141550


namespace right_triangle_hypotenuse_l1415_141556

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  a + b + c = 40 ∧  -- Perimeter condition
  (1/2) * a * b = 24 ∧  -- Area condition
  a^2 + b^2 = c^2 ∧  -- Pythagorean theorem (right triangle condition)
  c = 18.8 := by
  sorry


end right_triangle_hypotenuse_l1415_141556


namespace cosine_sine_eighth_power_bounds_l1415_141587

theorem cosine_sine_eighth_power_bounds (x : ℝ) : 
  1/8 ≤ (Real.cos x)^8 + (Real.sin x)^8 ∧ (Real.cos x)^8 + (Real.sin x)^8 ≤ 1 := by
  sorry

end cosine_sine_eighth_power_bounds_l1415_141587


namespace second_group_average_age_l1415_141575

theorem second_group_average_age 
  (n₁ : ℕ) (n₂ : ℕ) (m₁ : ℝ) (m_combined : ℝ) :
  n₁ = 11 →
  n₂ = 7 →
  m₁ = 25 →
  m_combined = 32 →
  (n₁ * m₁ + n₂ * ((n₁ + n₂) * m_combined - n₁ * m₁) / n₂) / (n₁ + n₂) = m_combined →
  ((n₁ + n₂) * m_combined - n₁ * m₁) / n₂ = 43 := by
sorry

end second_group_average_age_l1415_141575


namespace lcm_504_630_980_l1415_141597

theorem lcm_504_630_980 : Nat.lcm (Nat.lcm 504 630) 980 = 17640 := by
  sorry

end lcm_504_630_980_l1415_141597


namespace three_true_propositions_l1415_141540

theorem three_true_propositions (a b c d : ℝ) : 
  (∃ (p q r : Prop), 
    (p ∧ q → r) ∧ 
    (p ∧ r → q) ∧ 
    (q ∧ r → p) ∧
    (p = (a * b > 0)) ∧ 
    (q = (-c / a < -d / b)) ∧ 
    (r = (b * c > a * d))) :=
by sorry

end three_true_propositions_l1415_141540


namespace table_function_proof_l1415_141552

def f (x : ℝ) : ℝ := x^2 - x + 2

theorem table_function_proof :
  (f 2 = 3) ∧ (f 3 = 8) ∧ (f 4 = 15) ∧ (f 5 = 24) ∧ (f 6 = 35) := by
  sorry

end table_function_proof_l1415_141552


namespace button_up_shirt_cost_l1415_141526

def total_budget : ℕ := 200
def suit_pants : ℕ := 46
def suit_coat : ℕ := 38
def socks : ℕ := 11
def belt : ℕ := 18
def shoes : ℕ := 41
def amount_left : ℕ := 16

theorem button_up_shirt_cost : 
  total_budget - (suit_pants + suit_coat + socks + belt + shoes + amount_left) = 30 := by
  sorry

end button_up_shirt_cost_l1415_141526


namespace milk_container_problem_l1415_141520

/-- The initial quantity of milk in container A --/
def initial_quantity : ℝ := 1264

/-- The fraction of milk in container B relative to A's capacity --/
def fraction_in_B : ℝ := 0.375

/-- The amount transferred from C to B --/
def transfer_amount : ℝ := 158

theorem milk_container_problem :
  (fraction_in_B * initial_quantity + transfer_amount = 
   (1 - fraction_in_B) * initial_quantity - transfer_amount) ∧
  (initial_quantity > 0) := by
  sorry

end milk_container_problem_l1415_141520


namespace inequality_system_solution_range_l1415_141513

theorem inequality_system_solution_range (m : ℝ) : 
  (∃! (a b : ℤ), (a ≠ b) ∧ 
    ((a : ℝ) > -2) ∧ ((a : ℝ) ≤ (m + 2) / 3) ∧
    ((b : ℝ) > -2) ∧ ((b : ℝ) ≤ (m + 2) / 3) ∧
    (∀ (x : ℤ), (x ≠ a ∧ x ≠ b) → 
      ¬((x : ℝ) > -2 ∧ (x : ℝ) ≤ (m + 2) / 3))) →
  (-2 : ℝ) ≤ m ∧ m < 1 :=
by sorry

end inequality_system_solution_range_l1415_141513


namespace g_equals_inverse_at_three_point_five_l1415_141539

def g (x : ℝ) : ℝ := 3 * x - 7

theorem g_equals_inverse_at_three_point_five :
  g (3.5) = (Function.invFun g) (3.5) := by sorry

end g_equals_inverse_at_three_point_five_l1415_141539


namespace ellipse_triangle_area_l1415_141505

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the foci
def are_foci (F₁ F₂ : ℝ × ℝ) : Prop := 
  let (x₁, y₁) := F₁
  let (x₂, y₂) := F₂
  x₁^2 + y₁^2 = 5 ∧ x₂^2 + y₂^2 = 5 ∧ x₁ = -x₂ ∧ y₁ = -y₂

-- Define the distance ratio condition
def distance_ratio (P F₁ F₂ : ℝ × ℝ) : Prop :=
  let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let d₂ := Real.sqrt ((P.2 - F₂.1)^2 + (P.2 - F₂.2)^2)
  d₁ / d₂ = 2

-- Theorem statement
theorem ellipse_triangle_area 
  (P F₁ F₂ : ℝ × ℝ) 
  (h₁ : is_on_ellipse P.1 P.2) 
  (h₂ : are_foci F₁ F₂) 
  (h₃ : distance_ratio P F₁ F₂) : 
  let area := Real.sqrt (
    (F₁.1 - P.1)^2 + (F₁.2 - P.2)^2 +
    (F₂.1 - P.1)^2 + (F₂.2 - P.2)^2 +
    (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2
  ) / 4
  area = 4 := by sorry

end ellipse_triangle_area_l1415_141505


namespace inscribed_square_probability_l1415_141594

theorem inscribed_square_probability (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let square_side := r * Real.sqrt 2
  let square_area := square_side^2
  square_area / circle_area = 2 / π := by sorry

end inscribed_square_probability_l1415_141594


namespace unique_f_two_l1415_141595

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x + y) = x * y

theorem unique_f_two (f : ℝ → ℝ) (h : functional_equation f) : f 2 = 1 := by
  sorry

end unique_f_two_l1415_141595


namespace missing_number_value_l1415_141583

theorem missing_number_value : 
  ∃ (x : ℚ), ((476 + 424) * 2 - x * 476 * 424 = 2704) ∧ (x = -1/223) := by
  sorry

end missing_number_value_l1415_141583


namespace bryans_book_collection_l1415_141521

theorem bryans_book_collection (total_books : ℕ) (num_continents : ℕ) 
  (h1 : total_books = 488) 
  (h2 : num_continents = 4) 
  (h3 : total_books % num_continents = 0) : 
  total_books / num_continents = 122 := by
sorry

end bryans_book_collection_l1415_141521


namespace line_equation_proof_l1415_141568

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point -/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (l : Line) (p : Point) (given_line : Line) :
  p.x = 2 ∧ p.y = -1 ∧
  given_line.a = 2 ∧ given_line.b = 3 ∧ given_line.c = -4 ∧
  l.passesThrough p ∧
  l.isParallelTo given_line →
  l.a = 2 ∧ l.b = 3 ∧ l.c = -1 :=
by sorry

end line_equation_proof_l1415_141568


namespace total_third_graders_l1415_141582

/-- The number of third grade girl students -/
def girl_students : ℕ := 57

/-- The number of third grade boy students -/
def boy_students : ℕ := 66

/-- The total number of third grade students -/
def total_students : ℕ := girl_students + boy_students

/-- Theorem stating that the total number of third grade students is 123 -/
theorem total_third_graders : total_students = 123 := by
  sorry

end total_third_graders_l1415_141582


namespace ship_total_distance_l1415_141531

/-- Represents the daily travel of a ship --/
structure DailyTravel where
  distance : ℝ
  direction : String

/-- Calculates the total distance traveled by a ship over 4 days --/
def totalDistance (day1 day2 day3 day4 : DailyTravel) : ℝ :=
  day1.distance + day2.distance + day3.distance + day4.distance

/-- Theorem: The ship's total travel distance over 4 days is 960 miles --/
theorem ship_total_distance :
  let day1 := DailyTravel.mk 100 "north"
  let day2 := DailyTravel.mk (3 * 100) "east"
  let day3 := DailyTravel.mk (3 * 100 + 110) "east"
  let day4 := DailyTravel.mk 150 "30-degree angle with north"
  totalDistance day1 day2 day3 day4 = 960 := by
  sorry

#check ship_total_distance

end ship_total_distance_l1415_141531


namespace square_circumcenter_segment_length_l1415_141555

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square with side length 1 -/
structure UnitSquare where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The circumcenter of a triangle -/
def circumcenter (A B C : Point) : Point :=
  sorry

/-- The length of a segment between two points -/
def segmentLength (P Q : Point) : ℝ :=
  sorry

/-- The main theorem -/
theorem square_circumcenter_segment_length 
  (ABCD : UnitSquare) 
  (P Q : Point) 
  (h1 : Q = circumcenter B P C) 
  (h2 : D = circumcenter P Q ABCD.A) : 
  segmentLength P Q = Real.sqrt (2 - Real.sqrt 3) ∨ 
  segmentLength P Q = Real.sqrt (2 + Real.sqrt 3) :=
sorry

end square_circumcenter_segment_length_l1415_141555


namespace integer_expression_multiple_of_three_l1415_141502

theorem integer_expression_multiple_of_three (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : ∃ m : ℕ, n = 3 * m) :
  ∃ z : ℤ, (2 * n - 3 * k - 2) * (n.choose k) = (k + 2) * z := by
  sorry

end integer_expression_multiple_of_three_l1415_141502


namespace parabola_focus_coordinates_l1415_141558

/-- Given a parabola with equation 7x + 4y² = 0, its focus has coordinates (-7/16, 0) -/
theorem parabola_focus_coordinates :
  ∀ (x y : ℝ),
  (7 * x + 4 * y^2 = 0) →
  ∃ (f : ℝ × ℝ),
  f = (-7/16, 0) ∧
  f.1 = -1/(4 * (4/7)) ∧
  f.2 = 0 :=
by sorry

end parabola_focus_coordinates_l1415_141558


namespace arithmetic_sequence_sum_times_three_l1415_141507

theorem arithmetic_sequence_sum_times_three (a₁ l n : ℕ) (h1 : n = 11) (h2 : a₁ = 101) (h3 : l = 121) :
  3 * (a₁ + (a₁ + 2) + (a₁ + 4) + (a₁ + 6) + (a₁ + 8) + (a₁ + 10) + (a₁ + 12) + (a₁ + 14) + (a₁ + 16) + (a₁ + 18) + l) = 3663 := by
  sorry

end arithmetic_sequence_sum_times_three_l1415_141507


namespace min_value_theorem_l1415_141515

def f (x : ℝ) := |2*x + 1| + |x - 1|

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (1/2)*a + b + 2*c = 3/2) : 
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (a^2 + b^2 + c^2 ≥ 3/7) := by
  sorry

end min_value_theorem_l1415_141515


namespace cubic_equation_roots_l1415_141570

theorem cubic_equation_roots (p q : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 9*x^2 + p*x - q = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  p + q = 38 := by
sorry

end cubic_equation_roots_l1415_141570


namespace prob_at_least_one_black_correct_l1415_141571

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of black balls in the bag -/
def black_balls : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one black ball -/
def prob_at_least_one_black : ℚ := 7 / 10

theorem prob_at_least_one_black_correct :
  (1 : ℚ) - (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = prob_at_least_one_black :=
by sorry

end prob_at_least_one_black_correct_l1415_141571


namespace right_triangle_angle_calculation_l1415_141561

theorem right_triangle_angle_calculation (α β γ : ℝ) :
  α = 90 ∧ β = 63 ∧ α + β + γ = 180 → γ = 27 := by
  sorry

end right_triangle_angle_calculation_l1415_141561


namespace marley_has_31_fruits_l1415_141530

-- Define the number of fruits for Louis and Samantha
def louis_oranges : ℕ := 5
def louis_apples : ℕ := 3
def samantha_oranges : ℕ := 8
def samantha_apples : ℕ := 7

-- Define Marley's fruits based on the conditions
def marley_oranges : ℕ := 2 * louis_oranges
def marley_apples : ℕ := 3 * samantha_apples

-- Define Marley's total fruits
def marley_total_fruits : ℕ := marley_oranges + marley_apples

-- Theorem statement
theorem marley_has_31_fruits : marley_total_fruits = 31 := by
  sorry

end marley_has_31_fruits_l1415_141530


namespace salary_comparison_l1415_141591

/-- Given salaries in ratio 1:2:3 and sum of B and C's salaries is 6000,
    prove C's salary is 200% more than A's -/
theorem salary_comparison (a b c : ℕ) : 
  a + b + c > 0 →
  b = 2 * a →
  c = 3 * a →
  b + c = 6000 →
  (c - a) * 100 / a = 200 := by
sorry

end salary_comparison_l1415_141591


namespace log_343_equation_solution_l1415_141508

theorem log_343_equation_solution (x : ℝ) : 
  (Real.log 343 / Real.log (3 * x) = x) → 
  (∃ (a b : ℤ), x = a / b ∧ b ≠ 0 ∧ ¬∃ (n : ℤ), x = n ∧ ¬∃ (m : ℚ), x = m ^ 2 ∧ ¬∃ (k : ℚ), x = k ^ 3) := by
  sorry

end log_343_equation_solution_l1415_141508


namespace x_squared_minus_y_squared_l1415_141588

theorem x_squared_minus_y_squared (x y : ℝ) 
  (sum_eq : x + y = 20) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 80 := by
sorry

end x_squared_minus_y_squared_l1415_141588


namespace at_least_two_solved_five_l1415_141584

/-- Represents a participant in the math competition -/
structure Participant where
  solved : Finset (Fin 6)

/-- Represents the math competition -/
structure MathCompetition where
  participants : Finset Participant
  num_problems : Nat
  num_problems_eq : num_problems = 6
  any_two_solved : ∀ i j : Fin 6, i ≠ j →
    (participants.filter (λ p => i ∈ p.solved ∧ j ∈ p.solved)).card >
    (2 / 5 : ℚ) * participants.card
  no_all_solved : ∀ p : Participant, p ∈ participants → p.solved.card < 6

theorem at_least_two_solved_five (mc : MathCompetition) :
  (mc.participants.filter (λ p => p.solved.card = 5)).card ≥ 2 := by
  sorry

end at_least_two_solved_five_l1415_141584


namespace proof_by_contradiction_assumption_l1415_141510

theorem proof_by_contradiction_assumption 
  (P : ℝ → ℝ → Prop) 
  (Q : ℝ → Prop) 
  (R : ℝ → Prop) 
  (h : ∀ x y, P x y → (Q x ∨ R y)) :
  (∀ x y, P x y → (Q x ∨ R y)) ↔ 
  (∀ x y, P x y ∧ ¬Q x ∧ ¬R y → False) := by
sorry

end proof_by_contradiction_assumption_l1415_141510


namespace root_in_interval_l1415_141578

def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem root_in_interval (k : ℕ) : 
  (∃ x : ℝ, x > k ∧ x < k + 1 ∧ f x = 0) → k = 1 := by
  sorry

end root_in_interval_l1415_141578


namespace perfect_square_quadratic_l1415_141500

theorem perfect_square_quadratic (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 12*x + k = (x + a)^2) ↔ k = 36 := by
  sorry

end perfect_square_quadratic_l1415_141500


namespace billion_scientific_notation_l1415_141512

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_scientific_notation :
  toScientificNotation 1075000000 = ScientificNotation.mk 1.075 9 sorry sorry := by
  sorry

end billion_scientific_notation_l1415_141512


namespace cab_driver_income_l1415_141592

theorem cab_driver_income (day1 day2 day3 day4 day5 : ℕ) 
  (h1 : day1 = 250)
  (h2 : day2 = 400)
  (h4 : day4 = 400)
  (h5 : day5 = 500)
  (h_avg : (day1 + day2 + day3 + day4 + day5) / 5 = 460) :
  day3 = 750 := by
  sorry

end cab_driver_income_l1415_141592


namespace parallelograms_count_formula_l1415_141579

/-- The number of parallelograms formed by the intersection of two sets of parallel lines -/
def parallelograms_count (m n : ℕ) : ℕ :=
  Nat.choose m 2 * Nat.choose n 2

/-- Theorem stating that the number of parallelograms formed by the intersection
    of two sets of parallel lines is equal to C_m^2 * C_n^2 -/
theorem parallelograms_count_formula (m n : ℕ) :
  parallelograms_count m n = Nat.choose m 2 * Nat.choose n 2 := by
  sorry

end parallelograms_count_formula_l1415_141579


namespace smallest_even_triangle_perimeter_l1415_141596

/-- A triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 2 ∧ c = b + 2
  even_a : Even a

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ := t.a + t.b + t.c

/-- The triangle inequality for an EvenTriangle -/
def satisfies_triangle_inequality (t : EvenTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), satisfies_triangle_inequality t ∧
    ∀ (t' : EvenTriangle), satisfies_triangle_inequality t' → perimeter t ≤ perimeter t' ∧
    perimeter t = 12 := by
  sorry

end smallest_even_triangle_perimeter_l1415_141596


namespace initial_shirts_count_l1415_141528

/-- The number of shirts Haley returned -/
def returned_shirts : ℕ := 6

/-- The number of shirts Haley ended up with -/
def final_shirts : ℕ := 5

/-- The initial number of shirts Haley bought -/
def initial_shirts : ℕ := returned_shirts + final_shirts

/-- Theorem stating that the initial number of shirts is 11 -/
theorem initial_shirts_count : initial_shirts = 11 := by
  sorry

end initial_shirts_count_l1415_141528


namespace quadratic_equation_condition_l1415_141534

theorem quadratic_equation_condition (m : ℝ) : 
  (abs m + 1 = 2 ∧ m + 1 ≠ 0) ↔ m = 1 := by sorry

end quadratic_equation_condition_l1415_141534


namespace student_selection_plans_l1415_141545

/-- The number of students in the group -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of competitions -/
def num_competitions : ℕ := 4

/-- The number of competitions student A cannot participate in -/
def restricted_competitions : ℕ := 2

/-- The number of different plans to select students for competitions -/
def num_plans : ℕ := 72

theorem student_selection_plans :
  (Nat.choose total_students selected_students * Nat.factorial selected_students) +
  (Nat.choose (total_students - 1) (selected_students - 1) *
   Nat.choose (num_competitions - restricted_competitions) 1 *
   Nat.factorial (selected_students - 1)) = num_plans :=
sorry

end student_selection_plans_l1415_141545


namespace jason_clothing_expenses_l1415_141543

/-- The cost of Jason's shorts in dollars -/
def shorts_cost : ℝ := 14.28

/-- The cost of Jason's jacket in dollars -/
def jacket_cost : ℝ := 4.74

/-- The total amount Jason spent on clothing -/
def total_spent : ℝ := shorts_cost + jacket_cost

/-- Theorem stating that the total amount Jason spent on clothing is $19.02 -/
theorem jason_clothing_expenses : total_spent = 19.02 := by
  sorry

end jason_clothing_expenses_l1415_141543


namespace last_term_of_ap_l1415_141580

def arithmeticProgression (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

theorem last_term_of_ap : 
  let a := 2  -- first term
  let d := 2  -- common difference
  let n := 31 -- number of terms
  arithmeticProgression a d n = 62 := by
  sorry

end last_term_of_ap_l1415_141580


namespace intersection_implies_p_value_l1415_141563

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

-- Define the parabola equation
def parabola (p x y : ℝ) : Prop := y^2 = 2 * p * x

-- State the theorem
theorem intersection_implies_p_value 
  (p : ℝ) 
  (h_p_pos : p > 0) 
  (A B : ℝ × ℝ) 
  (h_A_ellipse : ellipse A.1 A.2)
  (h_A_parabola : parabola p A.1 A.2)
  (h_B_ellipse : ellipse B.1 B.2)
  (h_B_parabola : parabola p B.1 B.2)
  (h_distance : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4) :
  p = 1/4 := by
  sorry

end intersection_implies_p_value_l1415_141563


namespace line_not_in_fourth_quadrant_l1415_141544

/-- The line l: ax + by + c = 0 does not pass through the fourth quadrant when ab < 0 and bc < 0 -/
theorem line_not_in_fourth_quadrant (a b c : ℝ) (h1 : a * b < 0) (h2 : b * c < 0) :
  ∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ a * x + b * y + c ≠ 0 :=
by sorry

end line_not_in_fourth_quadrant_l1415_141544


namespace green_marbles_in_basket_b_l1415_141532

/-- Represents a basket with two types of marbles -/
structure Basket :=
  (color1 : Nat)
  (color2 : Nat)

/-- Calculates the absolute difference between two numbers -/
def absDiff (a b : Nat) : Nat :=
  max a b - min a b

/-- Finds the maximum difference among a list of baskets -/
def maxDiff (baskets : List Basket) : Nat :=
  baskets.map (λ b => absDiff b.color1 b.color2) |>.maximum?
    |>.getD 0

theorem green_marbles_in_basket_b :
  let basketA : Basket := ⟨4, 2⟩
  let basketC : Basket := ⟨3, 9⟩
  let basketB : Basket := ⟨x, 1⟩
  let allBaskets : List Basket := [basketA, basketB, basketC]
  maxDiff allBaskets = 6 →
  x = 7 :=
by sorry

end green_marbles_in_basket_b_l1415_141532
