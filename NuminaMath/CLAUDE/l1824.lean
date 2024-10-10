import Mathlib

namespace union_M_N_l1824_182490

-- Define the sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < -1}
def N : Set ℝ := {x | (1/2 : ℝ)^x ≤ 4}

-- State the theorem
theorem union_M_N : M ∪ N = {x | x ≥ -2} := by sorry

end union_M_N_l1824_182490


namespace floor_equation_solution_l1824_182441

/-- The floor function, which returns the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The statement to be proved -/
theorem floor_equation_solution :
  let x : ℚ := 22 / 7
  x * (floor (x * (floor (x * (floor x))))) = 88 := by sorry

end floor_equation_solution_l1824_182441


namespace max_earnings_is_zero_l1824_182465

/-- Represents the state of the boxes and Sisyphus's earnings -/
structure BoxState where
  a : ℕ  -- number of stones in box A
  b : ℕ  -- number of stones in box B
  c : ℕ  -- number of stones in box C
  earnings : ℤ  -- Sisyphus's current earnings (can be negative)

/-- Represents a move of a stone from one box to another -/
inductive Move
  | AtoB | AtoC | BtoA | BtoC | CtoA | CtoB

/-- Applies a move to the current state and returns the new state -/
def applyMove (state : BoxState) (move : Move) : BoxState :=
  match move with
  | Move.AtoB => { state with 
      a := state.a - 1, 
      b := state.b + 1, 
      earnings := state.earnings + state.b - (state.a - 1) }
  | Move.AtoC => { state with 
      a := state.a - 1, 
      c := state.c + 1, 
      earnings := state.earnings + state.c - (state.a - 1) }
  | Move.BtoA => { state with 
      b := state.b - 1, 
      a := state.a + 1, 
      earnings := state.earnings + state.a - (state.b - 1) }
  | Move.BtoC => { state with 
      b := state.b - 1, 
      c := state.c + 1, 
      earnings := state.earnings + state.c - (state.b - 1) }
  | Move.CtoA => { state with 
      c := state.c - 1, 
      a := state.a + 1, 
      earnings := state.earnings + state.a - (state.c - 1) }
  | Move.CtoB => { state with 
      c := state.c - 1, 
      b := state.b + 1, 
      earnings := state.earnings + state.b - (state.c - 1) }

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Applies a sequence of moves to an initial state -/
def applyMoves (initialState : BoxState) (moves : MoveSequence) : BoxState :=
  moves.foldl applyMove initialState

/-- Theorem: The maximum earnings of Sisyphus is 0 -/
theorem max_earnings_is_zero (initialState : BoxState) (moves : MoveSequence) :
  let finalState := applyMoves initialState moves
  (finalState.a = initialState.a ∧ 
   finalState.b = initialState.b ∧ 
   finalState.c = initialState.c) →
  finalState.earnings ≤ 0 := by
  sorry

#check max_earnings_is_zero

end max_earnings_is_zero_l1824_182465


namespace ring_stack_distance_l1824_182415

/-- Represents a stack of metallic rings -/
structure RingStack where
  topDiameter : ℕ
  smallestDiameter : ℕ
  thickness : ℕ

/-- Calculates the total vertical distance of a ring stack -/
def totalVerticalDistance (stack : RingStack) : ℕ :=
  let numRings := (stack.topDiameter - stack.smallestDiameter) / 2 + 1
  let sumDiameters := numRings * (stack.topDiameter + stack.smallestDiameter) / 2
  sumDiameters - numRings + 2 * stack.thickness

/-- Theorem stating the total vertical distance of the given ring stack -/
theorem ring_stack_distance :
  ∀ (stack : RingStack),
    stack.topDiameter = 22 ∧
    stack.smallestDiameter = 4 ∧
    stack.thickness = 1 →
    totalVerticalDistance stack = 122 := by
  sorry


end ring_stack_distance_l1824_182415


namespace ratio_of_sums_l1824_182433

theorem ratio_of_sums (p q r u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end ratio_of_sums_l1824_182433


namespace solution_pairs_l1824_182440

theorem solution_pairs (x y : ℝ) : 
  (|x + y| = 3 ∧ x * y = -10) → 
  ((x = 5 ∧ y = -2) ∨ (x = -2 ∧ y = 5) ∨ (x = 2 ∧ y = -5) ∨ (x = -5 ∧ y = 2)) :=
by sorry

end solution_pairs_l1824_182440


namespace weight_of_b_l1824_182413

/-- Given three weights a, b, and c, prove that b = 37 under the given conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 46 →
  b = 37 := by
sorry


end weight_of_b_l1824_182413


namespace function_properties_l1824_182443

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x + a) / (2^x - 1)

theorem function_properties (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f a x = -f a (-x)) →
  (∀ x : ℝ, x ≠ 0 → f a x = f a x) ∧
  (a = 1) ∧
  (∀ x y : ℝ, 0 < x → x < y → f a y < f a x) :=
sorry

end function_properties_l1824_182443


namespace expression_evaluation_l1824_182436

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℚ := -2
  ((x + 2*y)^2 - (x + y)*(x - y)) / (2*y) = -4 := by sorry

end expression_evaluation_l1824_182436


namespace bus_capacity_l1824_182477

/-- The number of students that can be accommodated by a given number of buses,
    each with a specified number of columns and rows of seats. -/
def total_students (buses : ℕ) (columns : ℕ) (rows : ℕ) : ℕ :=
  buses * columns * rows

/-- Theorem stating that 6 buses with 4 columns and 10 rows each can accommodate 240 students. -/
theorem bus_capacity : total_students 6 4 10 = 240 := by
  sorry

end bus_capacity_l1824_182477


namespace probability_of_red_is_one_fifth_l1824_182453

/-- Represents the contents of the bag -/
structure BagContents where
  red : ℕ
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing a red ball -/
def probabilityOfRed (bag : BagContents) : ℚ :=
  bag.red / (bag.red + bag.white + bag.black)

/-- Theorem stating that the probability of drawing a red ball is 1/5 -/
theorem probability_of_red_is_one_fifth (bag : BagContents) 
  (h1 : bag.red = 2) 
  (h2 : bag.white = 3) 
  (h3 : bag.black = 5) : 
  probabilityOfRed bag = 1/5 := by
  sorry

#check probability_of_red_is_one_fifth

end probability_of_red_is_one_fifth_l1824_182453


namespace prop_2_prop_4_l1824_182452

-- Proposition 2
theorem prop_2 (p q : Prop) : ¬(p ∨ q) → (¬p ∧ ¬q) := by sorry

-- Proposition 4
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x + a)

theorem prop_4 (a : ℝ) : (∀ x, f a x = f a (-x)) → a = -1 := by sorry

end prop_2_prop_4_l1824_182452


namespace albert_books_multiple_l1824_182406

theorem albert_books_multiple (stu_books : ℕ) (total_books : ℕ) (x : ℚ) : 
  stu_books = 9 →
  total_books = 45 →
  total_books = stu_books + stu_books * x →
  x = 4 := by
sorry

end albert_books_multiple_l1824_182406


namespace survey_total_students_l1824_182481

theorem survey_total_students :
  let mac_preference : ℕ := 60
  let both_preference : ℕ := mac_preference / 3
  let no_preference : ℕ := 90
  let windows_preference : ℕ := 40
  mac_preference + both_preference + no_preference + windows_preference = 210 := by
sorry

end survey_total_students_l1824_182481


namespace remove_all_triangles_no_triangles_remain_l1824_182434

/-- Represents a toothpick figure -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  is_symmetric : Bool
  has_additional_rows : Bool

/-- Represents the number of toothpicks that must be removed to eliminate all triangles -/
def toothpicks_to_remove (figure : ToothpickFigure) : ℕ := 
  if figure.total_toothpicks = 40 ∧ figure.is_symmetric ∧ figure.has_additional_rows
  then 40
  else 0

/-- Theorem stating that for a specific toothpick figure, 40 toothpicks must be removed -/
theorem remove_all_triangles (figure : ToothpickFigure) :
  figure.total_toothpicks = 40 ∧ figure.is_symmetric ∧ figure.has_additional_rows →
  toothpicks_to_remove figure = 40 :=
by
  sorry

/-- Theorem stating that removing 40 toothpicks is sufficient to eliminate all triangles -/
theorem no_triangles_remain (figure : ToothpickFigure) :
  figure.total_toothpicks = 40 ∧ figure.is_symmetric ∧ figure.has_additional_rows →
  toothpicks_to_remove figure = 40 →
  ∀ remaining_triangles, remaining_triangles = 0 :=
by
  sorry

end remove_all_triangles_no_triangles_remain_l1824_182434


namespace cube_volume_from_surface_area_l1824_182414

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 96 → s^3 = 64 := by
  sorry

end cube_volume_from_surface_area_l1824_182414


namespace milford_lake_algae_count_l1824_182485

theorem milford_lake_algae_count (current : ℕ) (increase : ℕ) (original : ℕ)
  (h1 : current = 3263)
  (h2 : increase = 2454)
  (h3 : current = original + increase) :
  original = 809 := by
  sorry

end milford_lake_algae_count_l1824_182485


namespace imaginary_part_of_one_minus_i_l1824_182462

theorem imaginary_part_of_one_minus_i :
  Complex.im (1 - Complex.I) = -1 := by sorry

end imaginary_part_of_one_minus_i_l1824_182462


namespace race_length_l1824_182424

/-- Represents the state of the race -/
structure RaceState where
  alexLead : Int
  distanceLeft : Int

/-- Calculates the final race state after all lead changes -/
def finalRaceState : RaceState :=
  let s1 : RaceState := { alexLead := 0, distanceLeft := 0 }  -- Even start
  let s2 : RaceState := { alexLead := 300, distanceLeft := s1.distanceLeft }
  let s3 : RaceState := { alexLead := s2.alexLead - 170, distanceLeft := s2.distanceLeft }
  { alexLead := s3.alexLead + 440, distanceLeft := 3890 }

/-- The theorem stating the total length of the race track -/
theorem race_length : 
  finalRaceState.alexLead + finalRaceState.distanceLeft = 4460 := by
  sorry


end race_length_l1824_182424


namespace january_oil_bill_l1824_182464

theorem january_oil_bill (january february : ℝ) 
  (h1 : february / january = 5 / 4)
  (h2 : (february + 30) / january = 3 / 2) : 
  january = 120 := by
  sorry

end january_oil_bill_l1824_182464


namespace circle_symmetry_l1824_182426

/-- Given a circle with center (1,1) and radius √2, prove that if it's symmetric about the line y = kx + 3, then k = -2 -/
theorem circle_symmetry (k : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 → 
    ∃ x' y' : ℝ, (x' - 1)^2 + (y' - 1)^2 = 2 ∧ 
    ((x + x') / 2, (y + y') / 2) ∈ {(x, y) | y = k * x + 3}) →
  k = -2 :=
sorry

end circle_symmetry_l1824_182426


namespace folded_rectangle_EF_length_l1824_182447

/-- A rectangle ABCD with side lengths AB = 4 and BC = 8 is folded so that A and C coincide,
    forming a new shape ABEFD. This function calculates the length of EF. -/
def foldedRectangleEFLength (AB BC : ℝ) : ℝ :=
  4

/-- Theorem stating that for a rectangle ABCD with AB = 4 and BC = 8, when folded so that
    A and C coincide to form ABEFD, the length of EF is 4. -/
theorem folded_rectangle_EF_length :
  foldedRectangleEFLength 4 8 = 4 := by
  sorry

#check folded_rectangle_EF_length

end folded_rectangle_EF_length_l1824_182447


namespace circle_properties_l1824_182466

-- Define the given circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 5 = 0

-- Define the point M
def point_M : ℝ × ℝ := (3, -1)

-- Define the point N
def point_N : ℝ × ℝ := (1, 2)

-- Define the equation of the required circle
def required_circle (x y : ℝ) : Prop := (x - 20/7)^2 + (y - 15/14)^2 = 845/196

-- Theorem statement
theorem circle_properties :
  -- The required circle passes through point M
  required_circle point_M.1 point_M.2 ∧
  -- The required circle passes through point N
  required_circle point_N.1 point_N.2 ∧
  -- The required circle is tangent to circle C at point N
  (∃ (t : ℝ), t ≠ 0 ∧
    ∀ (x y : ℝ),
      circle_C x y ↔ required_circle x y ∨
      ((x - point_N.1) = t * (40/7 - 2*point_N.1) ∧
       (y - point_N.2) = t * (30/7 - 2*point_N.2))) :=
sorry

end circle_properties_l1824_182466


namespace triangle_tan_half_angles_inequality_l1824_182478

theorem triangle_tan_half_angles_inequality (A B C : ℝ) (h₁ : A + B + C = π) :
  Real.tan (A / 2) * Real.tan (B / 2) * Real.tan (C / 2) ≤ Real.sqrt 3 / 9 := by
  sorry

end triangle_tan_half_angles_inequality_l1824_182478


namespace cubic_sum_inequality_l1824_182473

theorem cubic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c)^3 - (a^3 + b^3 + c^3) > (a + b) * (b + c) * (a + c) := by
  sorry

end cubic_sum_inequality_l1824_182473


namespace production_proof_l1824_182475

def average_production_problem (n : ℕ) (past_average current_average : ℚ) : Prop :=
  let past_total := n * past_average
  let today_production := (n + 1) * current_average - past_total
  today_production = 95

theorem production_proof :
  average_production_problem 8 50 55 := by
  sorry

end production_proof_l1824_182475


namespace existence_of_coefficients_l1824_182493

/-- Two polynomials with coefficients A, B, C, D -/
def poly1 (A B C D : ℝ) (x : ℝ) : ℝ := x^6 + 4*x^5 + A*x^4 + B*x^3 + C*x^2 + D*x + 1
def poly2 (A B C D : ℝ) (x : ℝ) : ℝ := x^6 - 4*x^5 + A*x^4 - B*x^3 + C*x^2 - D*x + 1

/-- The product of the two polynomials -/
def product (A B C D : ℝ) (x : ℝ) : ℝ := (poly1 A B C D x) * (poly2 A B C D x)

/-- Theorem stating the existence of coefficients satisfying the conditions -/
theorem existence_of_coefficients : ∃ (A B C D : ℝ), 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0) ∧ 
  (∃ (k : ℝ), ∀ (x : ℝ), product A B C D x = x^12 + k*x^6 + 1) ∧
  (∃ (b c : ℝ), ∀ (x : ℝ), 
    poly1 A B C D x = (x^3 + 2*x^2 + b*x + c)^2 ∧
    poly2 A B C D x = (x^3 - 2*x^2 + b*x - c)^2) :=
sorry

end existence_of_coefficients_l1824_182493


namespace complex_sum_powers_l1824_182400

theorem complex_sum_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 8) :
  ζ₁^8 + ζ₂^8 + ζ₃^8 = 451.625 := by
sorry

end complex_sum_powers_l1824_182400


namespace union_equals_reals_subset_of_complement_l1824_182446

-- Define the sets A and B
def A : Set ℝ := {x | x < 0 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2*a}

-- Theorem for part (1)
theorem union_equals_reals (a : ℝ) : 
  A ∪ B a = Set.univ ↔ a ∈ Set.Iic 0 :=
sorry

-- Theorem for part (2)
theorem subset_of_complement (a : ℝ) :
  B a ⊆ -A ↔ a ∈ {x | x ≥ 1/2} :=
sorry

end union_equals_reals_subset_of_complement_l1824_182446


namespace exists_n_plus_Sn_1980_consecutive_n_plus_Sn_l1824_182467

-- Define S(n) as the sum of digits of n
def S (n : ℕ) : ℕ := sorry

-- Part a: Existence of n such that n + S(n) = 1980
theorem exists_n_plus_Sn_1980 : ∃ n : ℕ, n + S n = 1980 := by sorry

-- Part b: At least one of two consecutive naturals is of the form n + S(n)
theorem consecutive_n_plus_Sn (k : ℕ) : 
  (∃ n : ℕ, k = n + S n) ∨ (∃ n : ℕ, k + 1 = n + S n) := by sorry

end exists_n_plus_Sn_1980_consecutive_n_plus_Sn_l1824_182467


namespace railway_stations_problem_l1824_182416

theorem railway_stations_problem (m n : ℕ) (h1 : n ≥ 1) :
  (m.choose 2 + n * m + n.choose 2) - m.choose 2 = 58 →
  ((m = 14 ∧ n = 2) ∨ (m = 29 ∧ n = 1)) := by
  sorry

end railway_stations_problem_l1824_182416


namespace negative_one_times_negative_three_l1824_182463

theorem negative_one_times_negative_three : (-1 : ℤ) * (-3 : ℤ) = (3 : ℤ) := by
  sorry

end negative_one_times_negative_three_l1824_182463


namespace sinusoidal_function_parameters_l1824_182496

/-- 
Given a sinusoidal function y = a * sin(b * x + φ) where a > 0 and b > 0,
if the maximum value is 3 and the period is 2π/4, then a = 3 and b = 4.
-/
theorem sinusoidal_function_parameters 
  (a b φ : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : ∀ x, a * Real.sin (b * x + φ) ≤ 3)
  (h4 : ∃ x, a * Real.sin (b * x + φ) = 3)
  (h5 : (2 * Real.pi) / b = Real.pi / 2) : 
  a = 3 ∧ b = 4 := by
sorry

end sinusoidal_function_parameters_l1824_182496


namespace jacket_price_after_discounts_l1824_182405

def initial_price : ℝ := 20
def first_discount : ℝ := 0.40
def second_discount : ℝ := 0.25

theorem jacket_price_after_discounts :
  let price_after_first := initial_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  final_price = 9 := by sorry

end jacket_price_after_discounts_l1824_182405


namespace bottle_cap_boxes_l1824_182494

theorem bottle_cap_boxes (total_caps : ℕ) (caps_per_box : ℕ) (h1 : total_caps = 316) (h2 : caps_per_box = 4) :
  total_caps / caps_per_box = 79 := by
  sorry

end bottle_cap_boxes_l1824_182494


namespace wood_length_ratio_l1824_182403

def first_set_length : ℝ := 4
def second_set_length : ℝ := 20

theorem wood_length_ratio : second_set_length / first_set_length = 5 := by
  sorry

end wood_length_ratio_l1824_182403


namespace blue_apples_count_l1824_182451

theorem blue_apples_count (b : ℕ) : 
  (3 * b : ℚ) - (3 * b : ℚ) / 5 = 12 → b = 5 := by sorry

end blue_apples_count_l1824_182451


namespace simplify_expression_l1824_182401

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x - 2) = 4*x - 20 := by
  sorry

end simplify_expression_l1824_182401


namespace index_card_area_l1824_182425

theorem index_card_area (l w : ℕ) (h1 : l = 3) (h2 : w = 7) : 
  (∃ (a b : ℕ), (l - a) * (w - b) = 10 ∧ a + b = 3) → 
  (l - 1) * (w - 2) = 10 := by
sorry

end index_card_area_l1824_182425


namespace isosceles_triangle_with_sides_4_and_9_l1824_182459

/-- An isosceles triangle with side lengths a, b, and c, where at least two sides are equal. -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  isosceles : (a = b) ∨ (b = c) ∨ (a = c)
  triangle_inequality : a + b > c ∧ b + c > a ∧ a + c > b

/-- The theorem stating that in an isosceles triangle with two sides of lengths 4 and 9, the third side must be 9. -/
theorem isosceles_triangle_with_sides_4_and_9 :
  ∀ (t : IsoscelesTriangle), (t.a = 4 ∧ t.b = 9) ∨ (t.a = 9 ∧ t.b = 4) → t.c = 9 := by
  sorry


end isosceles_triangle_with_sides_4_and_9_l1824_182459


namespace count_solutions_equation_l1824_182418

theorem count_solutions_equation : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (n + 500) / 50 = ⌊Real.sqrt (2 * n)⌋) ∧ 
    S.card = 5 := by
  sorry

end count_solutions_equation_l1824_182418


namespace clock_hands_overlap_l1824_182438

/-- The angle traveled by the hour hand in one minute -/
def hour_hand_speed : ℝ := 0.5

/-- The angle traveled by the minute hand in one minute -/
def minute_hand_speed : ℝ := 6

/-- The initial angle of the hour hand at 4:10 -/
def initial_hour_angle : ℝ := 60 + 0.5 * 10

/-- The time in minutes after 4:10 when the hands overlap -/
def overlap_time : ℝ := 11

theorem clock_hands_overlap :
  ∃ (t : ℝ), t > 0 ∧ t ≤ overlap_time ∧
  initial_hour_angle + hour_hand_speed * t = minute_hand_speed * t :=
sorry

end clock_hands_overlap_l1824_182438


namespace reading_assignment_l1824_182420

theorem reading_assignment (total_pages : ℕ) (pages_read : ℕ) (days_left : ℕ) : 
  total_pages = 408 →
  pages_read = 113 →
  days_left = 5 →
  (total_pages - pages_read) / days_left = 59 := by
  sorry

end reading_assignment_l1824_182420


namespace three_digit_sum_theorem_l1824_182408

def is_valid_digit_set (a b c : ℕ) : Prop :=
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

def sum_of_numbers (a b c : ℕ) : ℕ :=
  100 * (a + b + c) + 10 * (a + b + c) + (a + b + c)

theorem three_digit_sum_theorem :
  ∀ a b c : ℕ,
    is_valid_digit_set a b c →
    sum_of_numbers a b c = 1221 →
    ((a = 1 ∧ b = 1 ∧ c = 9) ∨
     (a = 2 ∧ b = 2 ∧ c = 7) ∨
     (a = 3 ∧ b = 3 ∧ c = 5) ∨
     (a = 4 ∧ b = 4 ∧ c = 3) ∨
     (a = 5 ∧ b = 5 ∧ c = 1)) :=
by sorry

end three_digit_sum_theorem_l1824_182408


namespace range_sum_bounds_l1824_182486

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- State the theorem
theorem range_sum_bounds :
  ∃ (m n : ℝ), (∀ x, m ≤ f x ∧ f x ≤ n) ∧
  (m = -5 ∧ n = 4) →
  1 ≤ m + n ∧ m + n ≤ 7 :=
sorry

end range_sum_bounds_l1824_182486


namespace equation_solution_l1824_182469

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end equation_solution_l1824_182469


namespace A_intersect_B_is_empty_l1824_182461

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}
def B : Set ℝ := {x | x - 1 > 0}

-- Statement to prove
theorem A_intersect_B_is_empty : A ∩ B = ∅ := by
  sorry

end A_intersect_B_is_empty_l1824_182461


namespace equation_one_solution_equation_two_no_solution_l1824_182417

-- Equation 1
theorem equation_one_solution (x : ℚ) : 
  (3 / (2 * x - 2) + 1 / (1 - x) = 3) ↔ (x = 7/6) :=
sorry

-- Equation 2
theorem equation_two_no_solution :
  ¬∃ y : ℚ, y / (y - 1) - 2 / (y^2 - 1) = 1 :=
sorry

end equation_one_solution_equation_two_no_solution_l1824_182417


namespace quadratic_root_difference_l1824_182407

theorem quadratic_root_difference (s t : ℝ) (hs : s > 0) (ht : t > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + s*x₁ + t = 0 ∧
    x₂^2 + s*x₂ + t = 0 ∧
    |x₁ - x₂| = 2) →
  s = 2 * Real.sqrt (t + 1) :=
by sorry

end quadratic_root_difference_l1824_182407


namespace inequality_solution_set_l1824_182402

theorem inequality_solution_set (x : ℝ) : 
  (4 * x^2 - 3 * x > 5) ↔ (x < -5/4 ∨ x > 1) := by
  sorry

end inequality_solution_set_l1824_182402


namespace sin_shift_equivalence_l1824_182444

theorem sin_shift_equivalence (x : ℝ) : 
  Real.sin (2 * x + Real.pi / 2) - 1 = Real.sin (2 * (x + Real.pi / 4)) - 1 := by
  sorry

end sin_shift_equivalence_l1824_182444


namespace greatest_prime_divisor_digit_sum_l1824_182454

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem greatest_prime_divisor_digit_sum :
  ∃ (p : ℕ), is_prime p ∧ (32767 % p = 0) ∧
  (∀ q : ℕ, is_prime q → (32767 % q = 0) → q ≤ p) ∧
  sum_of_digits p = 14 := by
  sorry

end greatest_prime_divisor_digit_sum_l1824_182454


namespace sequence_a2_value_l1824_182450

theorem sequence_a2_value (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = 2 * (a n - 1)) : a 2 = 4 := by
  sorry

end sequence_a2_value_l1824_182450


namespace symmetric_points_line_equation_l1824_182484

/-- Given two points P and Q that are symmetric about a line l, prove that the equation of l is x - y + 1 = 0 --/
theorem symmetric_points_line_equation (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let Q : ℝ × ℝ := (b - 1, a + 1)
  let l : Set (ℝ × ℝ) := {(x, y) | x - y + 1 = 0}
  (∀ (M : ℝ × ℝ), M ∈ l ↔ (dist M P)^2 = (dist M Q)^2) →
  l = {(x, y) | x - y + 1 = 0} :=
by sorry


end symmetric_points_line_equation_l1824_182484


namespace circle_equation_l1824_182419

/-- The equation of a circle passing through points A(1, -1) and B(-1, 1) with center on the line x + y - 2 = 0 -/
theorem circle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (-1, 1)
  let center_line (p : ℝ × ℝ) := p.1 + p.2 - 2 = 0
  let circle_eq (p : ℝ × ℝ) := (p.1 - 1)^2 + (p.2 - 1)^2 = 4
  let on_circle (p : ℝ × ℝ) := circle_eq p
  ∃ (c : ℝ × ℝ), 
    center_line c ∧ 
    on_circle A ∧ 
    on_circle B ∧ 
    on_circle (x, y) ↔ circle_eq (x, y) := by
  sorry

end circle_equation_l1824_182419


namespace eliminate_denominators_l1824_182458

theorem eliminate_denominators (x : ℝ) : 
  (2*x - 3) / 5 = 2*x / 3 - 3 ↔ 3*(2*x - 3) = 5*(2*x) - 3*15 := by
  sorry

end eliminate_denominators_l1824_182458


namespace quadratic_coefficient_l1824_182432

/-- A quadratic function passing through three specific points has a coefficient of 8/5 for its x² term. -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + c → 
    ((x = -3 ∧ y = 2) ∨ (x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = -6))) → 
  a = 8/5 := by sorry

end quadratic_coefficient_l1824_182432


namespace even_increasing_relation_l1824_182456

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f x < f y

theorem even_increasing_relation (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_incr : increasing_on_nonneg f) :
  f π > f (-3) ∧ f (-3) > f (-2) :=
sorry

end even_increasing_relation_l1824_182456


namespace roots_of_polynomial_l1824_182468

def f (x : ℝ) : ℝ := x^3 - 4*x^2 - x + 6

theorem roots_of_polynomial :
  (f (-1) = 0) ∧ (f 1 = 0) ∧ (f 6 = 0) ∧
  (∀ x : ℝ, f x = 0 → x = -1 ∨ x = 1 ∨ x = 6) :=
sorry

end roots_of_polynomial_l1824_182468


namespace negative_fraction_comparison_l1824_182457

theorem negative_fraction_comparison :
  -((4 : ℚ) / 5) < -((3 : ℚ) / 4) := by
  sorry

end negative_fraction_comparison_l1824_182457


namespace dad_second_half_speed_l1824_182421

-- Define the given conditions
def total_time : Real := 0.5  -- 30 minutes in hours
def first_half_speed : Real := 28
def jake_bike_speed : Real := 11
def jake_bike_time : Real := 2

-- Define the theorem
theorem dad_second_half_speed :
  let total_distance := jake_bike_speed * jake_bike_time
  let first_half_distance := first_half_speed * (total_time / 2)
  let second_half_distance := total_distance - first_half_distance
  let second_half_speed := second_half_distance / (total_time / 2)
  second_half_speed = 60 := by
  sorry

end dad_second_half_speed_l1824_182421


namespace concert_ticket_price_l1824_182448

theorem concert_ticket_price :
  ∃ (P : ℝ) (x : ℕ),
    x + 2 + 1 = 5 ∧
    x * P + (2 * 2.4 * P - 10) + 0.6 * P = 360 →
    P = 50 := by
  sorry

end concert_ticket_price_l1824_182448


namespace lighter_box_weight_l1824_182495

/-- Given a shipment of boxes with the following properties:
  * There are 30 boxes in total
  * Some boxes weigh W pounds (lighter boxes)
  * The rest of the boxes weigh 20 pounds (heavier boxes)
  * The initial average weight is 18 pounds
  * After removing 15 of the 20-pound boxes, the new average weight is 16 pounds
  Prove that the weight of the lighter boxes (W) is 16 pounds. -/
theorem lighter_box_weight (total_boxes : ℕ) (W : ℝ) (heavy_box_weight : ℝ) 
  (initial_avg : ℝ) (new_avg : ℝ) (removed_boxes : ℕ) :
  total_boxes = 30 →
  heavy_box_weight = 20 →
  initial_avg = 18 →
  new_avg = 16 →
  removed_boxes = 15 →
  (∃ (light_boxes heavy_boxes : ℕ), 
    light_boxes + heavy_boxes = total_boxes ∧
    (light_boxes * W + heavy_boxes * heavy_box_weight) / total_boxes = initial_avg ∧
    ((light_boxes * W + (heavy_boxes - removed_boxes) * heavy_box_weight) / 
      (total_boxes - removed_boxes) = new_avg)) →
  W = 16 := by
sorry

end lighter_box_weight_l1824_182495


namespace initial_number_of_girls_l1824_182491

/-- The number of girls in the initial group -/
def n : ℕ := sorry

/-- The initial average weight of the girls -/
def A : ℝ := sorry

/-- The weight of the new girl -/
def new_weight : ℝ := 100

/-- The weight of the girl being replaced -/
def replaced_weight : ℝ := 50

/-- The increase in average weight -/
def avg_increase : ℝ := 5

theorem initial_number_of_girls :
  (n * A - replaced_weight + new_weight) / n = A + avg_increase →
  n = 10 := by sorry

end initial_number_of_girls_l1824_182491


namespace disco_ball_max_cost_l1824_182480

def disco_ball_cost (total_budget : ℕ) (food_boxes : ℕ) (food_cost : ℕ) (disco_balls : ℕ) : ℕ :=
  (total_budget - food_boxes * food_cost) / disco_balls

theorem disco_ball_max_cost : disco_ball_cost 330 10 25 4 = 20 := by
  sorry

end disco_ball_max_cost_l1824_182480


namespace fruit_basket_count_l1824_182483

/-- The number of different fruit baskets that can be created -/
def num_fruit_baskets (num_apples : ℕ) (num_oranges : ℕ) : ℕ :=
  num_apples * num_oranges

/-- Theorem stating that the number of different fruit baskets with 7 apples and 12 oranges is 84 -/
theorem fruit_basket_count :
  num_fruit_baskets 7 12 = 84 := by
  sorry

end fruit_basket_count_l1824_182483


namespace max_type_a_workers_l1824_182423

theorem max_type_a_workers (total : ℕ) (x : ℕ) : 
  total = 150 → 
  total - x ≥ 3 * x → 
  x ≤ 37 ∧ ∃ y : ℕ, y > 37 → total - y < 3 * y :=
sorry

end max_type_a_workers_l1824_182423


namespace u_converges_to_L_least_k_for_bound_l1824_182404

def u : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 2 * u n - 2 * (u n)^2

def L : ℚ := 1/2

def converges_to (a : ℕ → ℚ) (l : ℚ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l| < ε

theorem u_converges_to_L : converges_to u L := sorry

theorem least_k_for_bound :
  (∃ k, |u k - L| ≤ 1/2^10) ∧
  (∀ k < 4, |u k - L| > 1/2^10) ∧
  |u 4 - L| ≤ 1/2^10 := sorry

end u_converges_to_L_least_k_for_bound_l1824_182404


namespace prime_odd_sum_l1824_182411

theorem prime_odd_sum (a b : ℕ) : 
  Nat.Prime a → 
  Odd b → 
  a^2 + b = 2001 → 
  a + b = 1999 := by sorry

end prime_odd_sum_l1824_182411


namespace cube_face_sum_l1824_182470

theorem cube_face_sum (a b c d e f : ℕ+) : 
  a * b * c + a * e * c + a * b * f + a * e * f + 
  d * b * c + d * e * c + d * b * f + d * e * f = 1001 →
  a + b + c + d + e + f = 31 := by
sorry

end cube_face_sum_l1824_182470


namespace complex_power_equivalence_l1824_182497

theorem complex_power_equivalence :
  (Complex.exp (Complex.I * Real.pi * (125 / 180)))^28 =
  Complex.exp (Complex.I * Real.pi * (140 / 180)) :=
by sorry

end complex_power_equivalence_l1824_182497


namespace cost_price_calculation_l1824_182437

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  selling_price = 12000 ∧ 
  discount_rate = 0.1 ∧ 
  profit_rate = 0.08 →
  (selling_price * (1 - discount_rate)) / (1 + profit_rate) = 10000 := by
sorry

end cost_price_calculation_l1824_182437


namespace equidistant_point_x_coordinate_l1824_182482

/-- The x-coordinate of the point on the x-axis equidistant from A(-3, 0) and B(3, 5) is 25/12 -/
theorem equidistant_point_x_coordinate :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (3, 5)
  ∃ x : ℝ, x = 25 / 12 ∧
    (x - A.1) ^ 2 + A.2 ^ 2 = (x - B.1) ^ 2 + (0 - B.2) ^ 2 :=
by sorry

end equidistant_point_x_coordinate_l1824_182482


namespace injured_cats_count_l1824_182442

/-- The number of injured cats Jeff found on Tuesday -/
def injured_cats : ℕ :=
  let initial_cats : ℕ := 20
  let kittens_found : ℕ := 2
  let cats_adopted : ℕ := 3 * 2
  let final_cats : ℕ := 17
  final_cats - (initial_cats + kittens_found - cats_adopted)

theorem injured_cats_count : injured_cats = 1 := by
  sorry

end injured_cats_count_l1824_182442


namespace fort_blocks_count_l1824_182428

/-- Represents the dimensions of a rectangular fort -/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed for a fort with given dimensions and wall thickness -/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) : ℕ :=
  d.length * d.width * d.height - 
  (d.length - 2 * wallThickness) * (d.width - 2 * wallThickness) * (d.height - wallThickness)

/-- Theorem stating that a fort with given dimensions requires 280 blocks -/
theorem fort_blocks_count : 
  blocksNeeded ⟨12, 10, 5⟩ 1 = 280 := by sorry

end fort_blocks_count_l1824_182428


namespace inscribed_circle_radius_l1824_182409

-- Define the circular sector
def circular_sector (R : ℝ) (θ : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 ≤ R^2 ∧ 0 ≤ x ∧ y ≤ x * Real.tan θ}

-- Define the inscribed circle
def inscribed_circle (r : ℝ) (R : ℝ) (θ : ℝ) :=
  {(x, y) : ℝ × ℝ | (x - (R - r))^2 + (y - r)^2 = r^2}

-- Theorem statement
theorem inscribed_circle_radius :
  ∀ (R : ℝ), R > 0 →
  ∃ (r : ℝ), r > 0 ∧
  inscribed_circle r R (π/6) ⊆ circular_sector R (π/6) ∧
  r = 2 := by
sorry


end inscribed_circle_radius_l1824_182409


namespace focus_of_our_parabola_l1824_182472

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola x^2 = 4y -/
def our_parabola : Parabola :=
  { equation := fun x y => x^2 = 4*y }

theorem focus_of_our_parabola :
  focus our_parabola = (0, 1) := by sorry

end focus_of_our_parabola_l1824_182472


namespace inequalities_satisfied_l1824_182449

theorem inequalities_satisfied (a b c x y z : ℤ) 
  (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) : 
  (x^2*y + y^2*z + z^2*x ≤ a^2*b + b^2*c + c^2*a) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧ 
  (x^2*y*z ≤ a^2*b*c) := by
  sorry

end inequalities_satisfied_l1824_182449


namespace fundraising_contribution_l1824_182430

theorem fundraising_contribution (total_goal : ℕ) (already_raised : ℕ) (num_people : ℕ) :
  total_goal = 2400 →
  already_raised = 600 →
  num_people = 8 →
  (total_goal - already_raised) / num_people = 225 :=
by
  sorry

end fundraising_contribution_l1824_182430


namespace minimum_cost_for_planting_l1824_182476

/-- Represents a flower type with its survival rate and seed pack options -/
structure FlowerType where
  name : String
  survivalRate : Rat
  pack1 : Nat × Rat  -- (seeds, price)
  pack2 : Nat × Rat  -- (seeds, price)

/-- Calculates the minimum number of seeds needed for a given number of surviving flowers -/
def seedsNeeded (survivingFlowers : Nat) (survivalRate : Rat) : Nat :=
  Nat.ceil (survivingFlowers / survivalRate)

/-- Calculates the cost of buying seeds for a flower type -/
def costForFlowerType (ft : FlowerType) (survivingFlowers : Nat) : Rat :=
  let seedsNeeded := seedsNeeded survivingFlowers ft.survivalRate
  if seedsNeeded ≤ ft.pack1.1 then ft.pack1.2 else ft.pack2.2

/-- Applies the discount to the total cost -/
def applyDiscount (totalCost : Rat) : Rat :=
  totalCost * (1 - 1/5)  -- 20% discount

/-- The main theorem -/
theorem minimum_cost_for_planting (roses daisies sunflowers : FlowerType) :
  roses.name = "Roses" →
  roses.survivalRate = 2/5 →
  roses.pack1 = (15, 5) →
  roses.pack2 = (40, 10) →
  daisies.name = "Daisies" →
  daisies.survivalRate = 3/5 →
  daisies.pack1 = (20, 4) →
  daisies.pack2 = (50, 9) →
  sunflowers.name = "Sunflowers" →
  sunflowers.survivalRate = 1/2 →
  sunflowers.pack1 = (10, 3) →
  sunflowers.pack2 = (30, 7) →
  let totalFlowers := 20
  let flowersPerType := totalFlowers / 3
  let totalCost := costForFlowerType roses flowersPerType +
                   costForFlowerType daisies flowersPerType +
                   costForFlowerType sunflowers (totalFlowers - 2 * flowersPerType)
  applyDiscount totalCost = 84/5 := by
  sorry


end minimum_cost_for_planting_l1824_182476


namespace train_length_proof_l1824_182422

/-- Proves that a train with the given conditions has a length of 1800 meters -/
theorem train_length_proof (train_speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  train_speed = 216 →
  crossing_time = 1 →
  train_length = 1800 :=
by
  sorry

#check train_length_proof

end train_length_proof_l1824_182422


namespace power_division_rule_l1824_182489

theorem power_division_rule (a : ℝ) : a^7 / a^5 = a^2 := by
  sorry

end power_division_rule_l1824_182489


namespace factorial_division_l1824_182492

theorem factorial_division : Nat.factorial 9 / Nat.factorial (9 - 3) = 504 := by
  sorry

end factorial_division_l1824_182492


namespace cookie_distribution_l1824_182410

theorem cookie_distribution (people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) :
  people = 4 →
  cookies_per_person = 22 →
  total_cookies = people * cookies_per_person →
  total_cookies = 88 := by
  sorry

end cookie_distribution_l1824_182410


namespace two_numbers_difference_l1824_182431

theorem two_numbers_difference (a b : ℕ) 
  (sum_eq : a + b = 24300)
  (b_divisible : 100 ∣ b)
  (b_div_100 : b / 100 = a) :
  b - a = 23760 :=
by sorry

end two_numbers_difference_l1824_182431


namespace rectangle_diagonal_l1824_182445

/-- The diagonal of a rectangle with length 40√3 cm and width 30√3 cm is 50√3 cm. -/
theorem rectangle_diagonal : 
  let length : ℝ := 40 * Real.sqrt 3
  let width : ℝ := 30 * Real.sqrt 3
  let diagonal : ℝ := Real.sqrt (length^2 + width^2)
  diagonal = 50 * Real.sqrt 3 := by
sorry

end rectangle_diagonal_l1824_182445


namespace product_mod_25_l1824_182435

theorem product_mod_25 :
  ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (105 * 77 * 132) % 25 = m ∧ m = 20 := by
  sorry

end product_mod_25_l1824_182435


namespace study_abroad_work_hours_l1824_182412

/-- Proves that working 28 hours per week for the remaining 10 weeks
    will meet the financial goal, given the initial plan and actual work done. -/
theorem study_abroad_work_hours
  (initial_hours_per_week : ℕ)
  (initial_weeks : ℕ)
  (goal_amount : ℕ)
  (actual_full_weeks : ℕ)
  (actual_reduced_weeks : ℕ)
  (reduced_hours_per_week : ℕ)
  (h_initial_hours : initial_hours_per_week = 25)
  (h_initial_weeks : initial_weeks = 15)
  (h_goal_amount : goal_amount = 4500)
  (h_actual_full_weeks : actual_full_weeks = 3)
  (h_actual_reduced_weeks : actual_reduced_weeks = 2)
  (h_reduced_hours : reduced_hours_per_week = 10)
  : ∃ (remaining_hours_per_week : ℕ),
    remaining_hours_per_week = 28 ∧
    (initial_hours_per_week * actual_full_weeks +
     reduced_hours_per_week * actual_reduced_weeks +
     remaining_hours_per_week * (initial_weeks - actual_full_weeks - actual_reduced_weeks)) *
    (goal_amount / (initial_hours_per_week * initial_weeks)) = goal_amount :=
by sorry

end study_abroad_work_hours_l1824_182412


namespace nested_cube_root_l1824_182499

theorem nested_cube_root (N : ℝ) (h : N > 1) :
  (N * (N * (N * N^(1/3))^(1/3))^(1/3))^(1/3) = N^(40/81) := by
  sorry

end nested_cube_root_l1824_182499


namespace equation_solution_l1824_182479

theorem equation_solution : 
  {x : ℝ | (x^3 - 5*x^2 + 6*x)*(x - 5) = 0} = {0, 2, 3, 5} := by sorry

end equation_solution_l1824_182479


namespace pure_imaginary_ratio_l1824_182460

theorem pure_imaginary_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : ∃ (t : ℝ), (3 - 4 * Complex.I) * (x + y * Complex.I) = t * Complex.I) : 
  x / y = -4 / 3 := by
sorry

end pure_imaginary_ratio_l1824_182460


namespace polynomial_division_theorem_l1824_182439

theorem polynomial_division_theorem (x : ℝ) : 
  (4*x^2 - 2*x + 3) * (2*x^2 + 5*x + 3) + (43*x + 36) = 8*x^4 + 16*x^3 - 7*x^2 + 4*x + 9 := by
  sorry

end polynomial_division_theorem_l1824_182439


namespace field_trip_total_cost_l1824_182429

def field_trip_cost (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_price : ℚ) (teacher_ticket_price : ℚ) 
  (discount_rate : ℚ) (tour_price : ℚ) (bus_cost : ℚ) 
  (meal_cost : ℚ) : ℚ :=
  let total_people := num_students + num_teachers
  let ticket_cost := num_students * student_ticket_price + num_teachers * teacher_ticket_price
  let discounted_ticket_cost := ticket_cost * (1 - discount_rate)
  let tour_cost := total_people * tour_price
  let meal_cost_total := total_people * meal_cost
  discounted_ticket_cost + tour_cost + bus_cost + meal_cost_total

theorem field_trip_total_cost :
  field_trip_cost 25 6 1.5 4 0.2 3.5 100 7.5 = 490.2 := by
  sorry

end field_trip_total_cost_l1824_182429


namespace min_value_of_expression_l1824_182455

theorem min_value_of_expression (x y z : ℝ) : 
  (x*y - z)^2 + (x + y + z)^2 ≥ 0 ∧ 
  ∃ (a b c : ℝ), (a*b - c)^2 + (a + b + c)^2 = 0 :=
sorry

end min_value_of_expression_l1824_182455


namespace sphere_surface_area_given_cone_l1824_182471

/-- Given a cone and a sphere with equal volumes, where the radius of the base of the cone
    is twice the radius of the sphere, and the height of the cone is 1,
    prove that the surface area of the sphere is 4π. -/
theorem sphere_surface_area_given_cone (r : ℝ) :
  (4 / 3 * π * r^3 = 1 / 3 * π * (2*r)^2 * 1) →
  4 * π * r^2 = 4 * π := by
  sorry

#check sphere_surface_area_given_cone

end sphere_surface_area_given_cone_l1824_182471


namespace milk_expense_l1824_182427

/-- Given Mr. Kishore's savings and expenses, prove the amount spent on milk -/
theorem milk_expense (savings : ℕ) (rent groceries education petrol misc : ℕ) 
  (h1 : savings = 2350)
  (h2 : rent = 5000)
  (h3 : groceries = 4500)
  (h4 : education = 2500)
  (h5 : petrol = 2000)
  (h6 : misc = 5650)
  (h7 : savings = (1 / 10 : ℚ) * (savings / (1 / 10 : ℚ))) :
  ∃ (milk : ℕ), milk = 1500 ∧ 
    (9 / 10 : ℚ) * (savings / (1 / 10 : ℚ)) = 
    (rent + groceries + education + petrol + misc + milk) :=
by sorry

end milk_expense_l1824_182427


namespace numerator_exceeds_denominator_l1824_182488

theorem numerator_exceeds_denominator (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 →
  (4 * x + 5 > 10 - 3 * x ↔ 5 / 7 < x ∧ x ≤ 3) := by
  sorry

end numerator_exceeds_denominator_l1824_182488


namespace original_number_proof_l1824_182487

theorem original_number_proof (x : ℚ) : 1 + 1 / x = 8 / 3 → x = 3 / 5 := by
  sorry

end original_number_proof_l1824_182487


namespace darren_boxes_correct_l1824_182498

/-- The number of crackers in each box -/
def crackers_per_box : ℕ := 24

/-- The total number of crackers bought by both Darren and Calvin -/
def total_crackers : ℕ := 264

/-- The number of boxes Darren bought -/
def darren_boxes : ℕ := 4

theorem darren_boxes_correct :
  ∃ (calvin_boxes : ℕ),
    calvin_boxes = 2 * darren_boxes - 1 ∧
    crackers_per_box * (darren_boxes + calvin_boxes) = total_crackers :=
by sorry

end darren_boxes_correct_l1824_182498


namespace rohan_sudhir_profit_difference_l1824_182474

/-- Represents an investor in the business -/
structure Investor where
  name : String
  amount : ℕ
  months : ℕ

/-- Calculates the investment-time product for an investor -/
def investmentTime (i : Investor) : ℕ := i.amount * i.months

/-- Calculates the share of profit for an investor -/
def profitShare (i : Investor) (totalInvestmentTime totalProfit : ℕ) : ℚ :=
  (investmentTime i : ℚ) / totalInvestmentTime * totalProfit

theorem rohan_sudhir_profit_difference 
  (suresh : Investor)
  (rohan : Investor)
  (sudhir : Investor)
  (priya : Investor)
  (akash : Investor)
  (totalProfit : ℕ) :
  suresh.name = "Suresh" ∧ suresh.amount = 18000 ∧ suresh.months = 12 ∧
  rohan.name = "Rohan" ∧ rohan.amount = 12000 ∧ rohan.months = 9 ∧
  sudhir.name = "Sudhir" ∧ sudhir.amount = 9000 ∧ sudhir.months = 8 ∧
  priya.name = "Priya" ∧ priya.amount = 15000 ∧ priya.months = 6 ∧
  akash.name = "Akash" ∧ akash.amount = 10000 ∧ akash.months = 6 ∧
  totalProfit = 5948 →
  let totalInvestmentTime := investmentTime suresh + investmentTime rohan + 
                             investmentTime sudhir + investmentTime priya + 
                             investmentTime akash
  (profitShare rohan totalInvestmentTime totalProfit - 
   profitShare sudhir totalInvestmentTime totalProfit).num = 393 :=
by sorry

end rohan_sudhir_profit_difference_l1824_182474
