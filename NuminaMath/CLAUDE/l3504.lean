import Mathlib

namespace square_root_of_nine_l3504_350411

theorem square_root_of_nine : 
  {x : ℝ | x^2 = 9} = {3, -3} := by sorry

end square_root_of_nine_l3504_350411


namespace range_of_x_for_inequality_l3504_350498

theorem range_of_x_for_inequality (x : ℝ) : 
  (∀ m : ℝ, m ∈ Set.Icc 0 1 → m * x^2 - 2*x - m ≥ 2) ↔ x ∈ Set.Iic (-1) :=
sorry

end range_of_x_for_inequality_l3504_350498


namespace long_jump_competition_l3504_350439

/-- The long jump competition problem -/
theorem long_jump_competition (first second third fourth : ℝ) : 
  first = 22 →
  second = first + 1 →
  third = second - 2 →
  fourth = 24 →
  fourth - third = 3 := by
  sorry

end long_jump_competition_l3504_350439


namespace union_of_A_and_B_l3504_350405

-- Define sets A and B
def A : Set ℝ := {x | x * (x - 2) < 3}
def B : Set ℝ := {x | 5 / (x + 1) ≥ 1}

-- Theorem statement
theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 < x ∧ x ≤ 4} := by sorry

end union_of_A_and_B_l3504_350405


namespace fourth_term_of_gp_l3504_350473

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem fourth_term_of_gp (x : ℝ) :
  let a₁ := x
  let a₂ := 3 * x + 3
  let a₃ := 5 * x + 5
  let r := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → geometric_progression a₁ r n = if n = 1 then a₁ else if n = 2 then a₂ else if n = 3 then a₃ else 0) →
  geometric_progression a₁ r 4 = -125 / 12 :=
by sorry

end fourth_term_of_gp_l3504_350473


namespace fish_population_after_bobbit_worm_l3504_350471

/-- Calculates the number of fish remaining in James' aquarium when he discovers the Bobbit worm -/
theorem fish_population_after_bobbit_worm 
  (initial_fish : ℕ) 
  (daily_eaten : ℕ) 
  (days_before_adding : ℕ) 
  (fish_added : ℕ) 
  (total_days : ℕ) 
  (h1 : initial_fish = 60)
  (h2 : daily_eaten = 2)
  (h3 : days_before_adding = 14)
  (h4 : fish_added = 8)
  (h5 : total_days = 21) :
  initial_fish - (daily_eaten * total_days) + fish_added = 26 :=
sorry

end fish_population_after_bobbit_worm_l3504_350471


namespace largest_difference_even_odd_three_digit_l3504_350429

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def all_even_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 0

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d % 2 = 1

theorem largest_difference_even_odd_three_digit : 
  ∃ (a b : ℕ), 
    is_three_digit_number a ∧
    is_three_digit_number b ∧
    has_distinct_digits a ∧
    has_distinct_digits b ∧
    all_even_digits a ∧
    all_odd_digits b ∧
    (∀ (x y : ℕ), 
      is_three_digit_number x ∧
      is_three_digit_number y ∧
      has_distinct_digits x ∧
      has_distinct_digits y ∧
      all_even_digits x ∧
      all_odd_digits y →
      x - y ≤ a - b) ∧
    a - b = 729 :=
sorry

end largest_difference_even_odd_three_digit_l3504_350429


namespace problem_solution_l3504_350469

def A (a : ℕ) : Set ℕ := {2, 5, a + 1}
def B (a : ℕ) : Set ℕ := {1, 3, a}
def U : Set ℕ := {x | x ≤ 6}

theorem problem_solution (a : ℕ) 
  (h1 : A a ∩ B a = {2, 3}) :
  (a = 2) ∧ 
  (A a ∪ B a = {1, 2, 3, 5}) ∧ 
  ((Uᶜ ∩ (A a)ᶜ) ∩ (Uᶜ ∩ (B a)ᶜ) = {0, 4, 6}) := by
  sorry

end problem_solution_l3504_350469


namespace certain_number_value_l3504_350403

theorem certain_number_value (t b c : ℝ) : 
  (t + b + c + 14 + 15) / 5 = 12 → 
  ∃ x : ℝ, (t + b + c + x) / 4 = 15 ∧ x = 29 := by
sorry

end certain_number_value_l3504_350403


namespace nancy_widget_production_l3504_350465

/-- Nancy's widget production problem -/
theorem nancy_widget_production (t : ℝ) (h : t > 0) : 
  let w := 2 * t
  let monday_production := w * t
  let tuesday_production := (w + 5) * (t - 3)
  monday_production - tuesday_production = t + 15 := by
sorry

end nancy_widget_production_l3504_350465


namespace min_value_reciprocal_sum_l3504_350423

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  (1 / x + 3 / y) ≥ 1 + Real.sqrt 3 / 2 := by
  sorry

end min_value_reciprocal_sum_l3504_350423


namespace problem_statement_l3504_350437

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : y > x) 
  (h : x / y + y / x = 3) : (x + y) / (x - y) = Real.sqrt 5 := by
  sorry

end problem_statement_l3504_350437


namespace no_solution_implies_a_leq_2_l3504_350451

theorem no_solution_implies_a_leq_2 (a : ℝ) : 
  (∀ x : ℝ, ¬(2*x - 4 > 0 ∧ x - a < 0)) → a ≤ 2 := by
sorry

end no_solution_implies_a_leq_2_l3504_350451


namespace plotted_points_form_circle_l3504_350497

theorem plotted_points_form_circle :
  ∀ (x y : ℝ), (∃ t : ℝ, x = Real.cos t ∧ y = Real.sin t) →
  x^2 + y^2 = 1 :=
by sorry

end plotted_points_form_circle_l3504_350497


namespace right_triangle_side_length_right_triangle_side_length_proof_l3504_350476

/-- Given a right triangle with hypotenuse length 13 and one non-hypotenuse side length 12,
    the length of the other side is 5. -/
theorem right_triangle_side_length : ℝ → ℝ → ℝ → Prop :=
  fun hypotenuse side1 side2 =>
    hypotenuse = 13 ∧ side1 = 12 ∧ side2 * side2 + side1 * side1 = hypotenuse * hypotenuse →
    side2 = 5

/-- Proof of the theorem -/
theorem right_triangle_side_length_proof : right_triangle_side_length 13 12 5 := by
  sorry

end right_triangle_side_length_right_triangle_side_length_proof_l3504_350476


namespace intersection_and_inequality_l3504_350488

/-- 
Given a line y = 2x + m that intersects the x-axis at (-1, 0),
prove that the solution set of 2x + m ≤ 0 is x ≤ -1.
-/
theorem intersection_and_inequality (m : ℝ) 
  (h1 : 2 * (-1) + m = 0) -- Line intersects x-axis at (-1, 0)
  (x : ℝ) : 
  (2 * x + m ≤ 0) ↔ (x ≤ -1) := by
sorry

end intersection_and_inequality_l3504_350488


namespace tan_equality_periodic_l3504_350481

theorem tan_equality_periodic (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1500 * π / 180) → n = 60 := by
  sorry

end tan_equality_periodic_l3504_350481


namespace bread_rise_time_l3504_350467

/-- The time (in minutes) Mark lets the bread rise each time -/
def rise_time : ℕ := sorry

/-- The total time (in minutes) to make bread -/
def total_time : ℕ := 280

/-- The time (in minutes) spent kneading -/
def kneading_time : ℕ := 10

/-- The time (in minutes) spent baking -/
def baking_time : ℕ := 30

/-- Theorem stating that the rise time is 120 minutes -/
theorem bread_rise_time : rise_time = 120 := by
  sorry

end bread_rise_time_l3504_350467


namespace point_q_location_l3504_350427

/-- Given four points O, A, B, C on a straight line and a point Q on AB, prove that Q's position relative to O is 2a + 2.5b -/
theorem point_q_location (a b c : ℝ) (O A B C Q : ℝ) : 
  O < A ∧ A < B ∧ B < C ∧  -- Points are in order
  A - O = 2 * a ∧  -- OA = 2a
  B - A = 3 * b ∧  -- AB = 3b
  C - B = 4 * c ∧  -- BC = 4c
  A ≤ Q ∧ Q ≤ B ∧  -- Q is on segment AB
  (Q - A) / (B - Q) = 3 / 1  -- AQ:QB = 3:1
  → Q - O = 2 * a + 2.5 * b :=
by sorry

end point_q_location_l3504_350427


namespace fixed_point_of_f_l3504_350414

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 + 3

-- State the theorem
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 0 = 2 := by sorry

end fixed_point_of_f_l3504_350414


namespace prob_b_leads_2to1_expected_score_b_l3504_350410

/-- Represents a table tennis game between player A and player B -/
structure TableTennisGame where
  /-- Probability of the server scoring a point -/
  serverWinProb : ℝ
  /-- Player A serves first -/
  aServesFirst : Bool

/-- Calculates the probability of player B leading 2-1 at the start of the fourth serve -/
def probBLeads2to1 (game : TableTennisGame) : ℝ := sorry

/-- Calculates the expected score of player B at the start of the fourth serve -/
def expectedScoreB (game : TableTennisGame) : ℝ := sorry

/-- Theorem stating the probability of player B leading 2-1 at the start of the fourth serve -/
theorem prob_b_leads_2to1 (game : TableTennisGame) 
  (h1 : game.serverWinProb = 0.6) 
  (h2 : game.aServesFirst = true) : 
  probBLeads2to1 game = 0.352 := by sorry

/-- Theorem stating the expected score of player B at the start of the fourth serve -/
theorem expected_score_b (game : TableTennisGame) 
  (h1 : game.serverWinProb = 0.6) 
  (h2 : game.aServesFirst = true) : 
  expectedScoreB game = 1.400 := by sorry

end prob_b_leads_2to1_expected_score_b_l3504_350410


namespace vanilla_jelly_beans_count_l3504_350462

theorem vanilla_jelly_beans_count :
  ∀ (vanilla grape : ℕ),
    grape = 5 * vanilla + 50 →
    vanilla + grape = 770 →
    vanilla = 120 := by
  sorry

end vanilla_jelly_beans_count_l3504_350462


namespace final_painting_width_l3504_350402

theorem final_painting_width (total_paintings : Nat) (total_area : Nat) 
  (small_paintings : Nat) (small_painting_side : Nat)
  (large_painting_width large_painting_height : Nat)
  (final_painting_height : Nat) :
  total_paintings = 5 →
  total_area = 200 →
  small_paintings = 3 →
  small_painting_side = 5 →
  large_painting_width = 10 →
  large_painting_height = 8 →
  final_painting_height = 5 →
  ∃ (final_painting_width : Nat),
    final_painting_width = 9 ∧
    total_area = 
      small_paintings * small_painting_side * small_painting_side +
      large_painting_width * large_painting_height +
      final_painting_height * final_painting_width :=
by sorry

end final_painting_width_l3504_350402


namespace f_6_equals_0_l3504_350438

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x in ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has the property f(x+2) = -f(x) for all x in ℝ -/
def HasPeriod2WithSignFlip (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

theorem f_6_equals_0 (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : HasPeriod2WithSignFlip f) : f 6 = 0 := by
  sorry

end f_6_equals_0_l3504_350438


namespace equal_roots_condition_l3504_350478

theorem equal_roots_condition (m : ℝ) : 
  (∃ (x : ℝ), (x * (x - 3) - (m + 2)) / ((x - 3) * (m - 2)) = x / m ∧ 
   ∀ (y : ℝ), (y * (y - 3) - (m + 2)) / ((y - 3) * (m - 2)) = y / m → y = x) ↔ 
  (m = 2 ∨ m = (9 + Real.sqrt 57) / 8 ∨ m = (9 - Real.sqrt 57) / 8) := by
sorry

end equal_roots_condition_l3504_350478


namespace original_price_calculation_l3504_350416

theorem original_price_calculation (reduced_price : ℝ) (reduction_percent : ℝ) 
  (h1 : reduced_price = 620) 
  (h2 : reduction_percent = 20) : 
  reduced_price / (1 - reduction_percent / 100) = 775 := by
  sorry

end original_price_calculation_l3504_350416


namespace total_apples_l3504_350407

/-- Given 37 baskets with 17 apples each, prove that the total number of apples is 629. -/
theorem total_apples (baskets : ℕ) (apples_per_basket : ℕ) 
  (h1 : baskets = 37) (h2 : apples_per_basket = 17) : 
  baskets * apples_per_basket = 629 := by
  sorry

end total_apples_l3504_350407


namespace sphere_volume_l3504_350453

theorem sphere_volume (d r h : ℝ) (h1 : d = 2 * Real.sqrt 5) (h2 : h = 2) 
  (h3 : r^2 = (d/2)^2 + h^2) : (4/3) * Real.pi * r^3 = 36 * Real.pi := by
  sorry

end sphere_volume_l3504_350453


namespace y_coordinates_equal_l3504_350456

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 - 4

theorem y_coordinates_equal :
  ∀ y₁ y₂ : ℝ,
  parabola 2 y₁ →
  parabola 4 y₂ →
  y₁ = y₂ := by
  sorry

end y_coordinates_equal_l3504_350456


namespace greatest_common_length_l3504_350459

theorem greatest_common_length (rope1 rope2 rope3 rope4 : ℕ) 
  (h1 : rope1 = 48) (h2 : rope2 = 64) (h3 : rope3 = 80) (h4 : rope4 = 120) :
  Nat.gcd rope1 (Nat.gcd rope2 (Nat.gcd rope3 rope4)) = 8 :=
sorry

end greatest_common_length_l3504_350459


namespace polynomial_integer_root_theorem_l3504_350495

theorem polynomial_integer_root_theorem (n : ℕ+) :
  (∃ (k : Fin n → ℤ) (P : Polynomial ℤ),
    (∀ (i j : Fin n), i ≠ j → k i ≠ k j) ∧
    (Polynomial.degree P ≤ n) ∧
    (∀ (i : Fin n), P.eval (k i) = n) ∧
    (∃ (z : ℤ), P.eval z = 0)) ↔
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

#check polynomial_integer_root_theorem

end polynomial_integer_root_theorem_l3504_350495


namespace sum_and_difference_l3504_350442

theorem sum_and_difference : 2345 + 3452 + 4523 + 5234 - 1234 = 14320 := by
  sorry

end sum_and_difference_l3504_350442


namespace vector_ratio_theorem_l3504_350475

-- Define the plane and points
variable (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P]
variable (O A B C : P)

-- Define the non-collinearity condition
def noncollinear (O A B C : P) : Prop :=
  ¬ (∃ (a b c : ℝ), a • (A - O) + b • (B - O) + c • (C - O) = 0 ∧ (a, b, c) ≠ (0, 0, 0))

-- State the theorem
theorem vector_ratio_theorem (h_noncollinear : noncollinear P O A B C)
  (h_eq : A - O - 4 • (B - O) + 3 • (C - O) = 0) :
  ‖A - B‖ / ‖C - A‖ = 3 / 4 := by sorry

end vector_ratio_theorem_l3504_350475


namespace purely_imaginary_condition_l3504_350466

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The condition for z = (1+mi)(1+i) to be purely imaginary, where m is a real number. -/
theorem purely_imaginary_condition (m : ℝ) : 
  IsPurelyImaginary ((1 + m * Complex.I) * (1 + Complex.I)) ↔ m = 1 := by
  sorry

#check purely_imaginary_condition

end purely_imaginary_condition_l3504_350466


namespace triangle_problem_l3504_350433

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  -- Given conditions
  3 * Real.cos (B + C) = -1 ∧
  a = 3 ∧
  (1 / 2) * b * c * Real.sin A = 2 * Real.sqrt 2 →
  -- Conclusion
  Real.cos A = 1 / 3 ∧
  ((b = 2 ∧ c = 3) ∨ (b = 3 ∧ c = 2)) := by
sorry

end triangle_problem_l3504_350433


namespace stock_price_change_l3504_350430

theorem stock_price_change (total_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : ∃ (higher lower : ℕ), 
    higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5)) :
  ∃ (higher : ℕ), higher = 1080 ∧ 
    ∃ (lower : ℕ), higher + lower = total_stocks ∧ 
    higher = lower + (lower / 5) := by
  sorry

end stock_price_change_l3504_350430


namespace working_days_count_l3504_350479

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a day in the month -/
structure DayInMonth where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Determines if a given day is a holiday -/
def isHoliday (d : DayInMonth) : Bool :=
  match d.dayOfWeek with
  | DayOfWeek.Sunday => true
  | DayOfWeek.Saturday => d.day % 14 == 8  -- Every second Saturday
  | _ => false

/-- Theorem: In a 30-day month starting on a Saturday, with every second Saturday 
    and all Sundays as holidays, there are 23 working days -/
theorem working_days_count : 
  let month : List DayInMonth := sorry  -- List of 30 days starting from Saturday
  (month.length = 30) →
  (month.head?.map (fun d => d.dayOfWeek) = some DayOfWeek.Saturday) →
  (month.filter (fun d => ¬isHoliday d)).length = 23 :=
by sorry

end working_days_count_l3504_350479


namespace meaningful_fraction_range_l3504_350435

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = 2 / (2 * x - 1)) ↔ x ≠ 1/2 :=
by sorry

end meaningful_fraction_range_l3504_350435


namespace geometric_arithmetic_inequality_l3504_350496

/-- A positive geometric progression -/
def is_positive_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- An arithmetic progression -/
def is_arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n, b (n + 1) = b n + d

/-- The main theorem -/
theorem geometric_arithmetic_inequality
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_geo : is_positive_geometric_progression a)
  (h_arith : is_arithmetic_progression b)
  (h_eq : a 6 = b 7) :
  a 3 + a 9 ≥ b 4 + b 10 := by
  sorry

end geometric_arithmetic_inequality_l3504_350496


namespace anna_cupcakes_l3504_350434

def total_cupcakes : ℕ := 150

def classmates_fraction : ℚ := 2/5
def neighbors_fraction : ℚ := 1/3
def work_friends_fraction : ℚ := 1/10
def eating_fraction : ℚ := 7/15

def remaining_cupcakes : ℕ := 14

theorem anna_cupcakes :
  let given_away := (classmates_fraction + neighbors_fraction + work_friends_fraction) * total_cupcakes
  let after_giving := total_cupcakes - ⌊given_away⌋
  let eaten := ⌊eating_fraction * after_giving⌋
  total_cupcakes - ⌊given_away⌋ - eaten = remaining_cupcakes := by
  sorry

#check anna_cupcakes

end anna_cupcakes_l3504_350434


namespace negative_double_negative_and_negative_absolute_are_opposite_l3504_350432

-- Define opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Theorem statement
theorem negative_double_negative_and_negative_absolute_are_opposite :
  are_opposite (-(-5)) (-|5|) := by
  sorry

end negative_double_negative_and_negative_absolute_are_opposite_l3504_350432


namespace inequality1_solution_system_solution_integer_system_solution_l3504_350449

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 3 * x - 5 > 5 * x + 3
def inequality2 (x : ℝ) : Prop := x - 1 ≥ 1 - x
def inequality3 (x : ℝ) : Prop := x + 8 > 4 * x - 1

-- Define the solution sets
def solution_set1 : Set ℝ := {x : ℝ | x < -4}
def solution_set2 : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 3}

-- Define the integer solutions
def integer_solutions : Set ℤ := {1, 2}

-- Theorem statements
theorem inequality1_solution : 
  {x : ℝ | inequality1 x} = solution_set1 :=
sorry

theorem system_solution : 
  {x : ℝ | inequality2 x ∧ inequality3 x} = solution_set2 :=
sorry

theorem integer_system_solution : 
  {x : ℤ | (x : ℝ) ∈ solution_set2} = integer_solutions :=
sorry

end inequality1_solution_system_solution_integer_system_solution_l3504_350449


namespace tangent_line_slope_l3504_350426

/-- Given a line y = kx + 1 tangent to the curve y = 1/x at point (a, 1/a) and passing through (0, 1), k equals -1/4 -/
theorem tangent_line_slope (k a : ℝ) : 
  (∀ x, x ≠ 0 → (k * x + 1) = 1 / x ∨ (k * x + 1) > 1 / x) → -- tangent condition
  (k * 0 + 1 = 1) →                                         -- passes through (0, 1)
  (k * a + 1 = 1 / a) →                                     -- point of tangency
  (k = -1 / (a^2)) →                                        -- slope at point of tangency
  (k = -1/4) :=
by sorry

end tangent_line_slope_l3504_350426


namespace sin_sixty_degrees_l3504_350494

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_sixty_degrees_l3504_350494


namespace pond_volume_l3504_350489

/-- Calculates the volume of a trapezoidal prism -/
def trapezoidalPrismVolume (length : ℝ) (avgWidth : ℝ) (avgDepth : ℝ) : ℝ :=
  length * avgWidth * avgDepth

/-- The pond dimensions -/
def pondLength : ℝ := 25
def pondAvgWidth : ℝ := 12.5
def pondAvgDepth : ℝ := 10

/-- Theorem stating the volume of the pond -/
theorem pond_volume :
  trapezoidalPrismVolume pondLength pondAvgWidth pondAvgDepth = 3125 := by
  sorry

#check pond_volume

end pond_volume_l3504_350489


namespace train_passing_jogger_time_train_passes_jogger_in_32_seconds_l3504_350477

/-- The time taken for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (initial_distance : ℝ) 
  (train_length : ℝ) : ℝ :=
by
  -- Convert speeds from km/hr to m/s
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  
  -- Calculate relative speed
  let relative_speed := train_speed_ms - jogger_speed_ms
  
  -- Calculate total distance to be covered
  let total_distance := initial_distance + train_length
  
  -- Calculate time taken
  let time_taken := total_distance / relative_speed
  
  -- Prove that the time taken is 32 seconds
  sorry

/-- The main theorem stating that the train will pass the jogger in 32 seconds -/
theorem train_passes_jogger_in_32_seconds : 
  train_passing_jogger_time 9 45 200 120 = 32 :=
by sorry

end train_passing_jogger_time_train_passes_jogger_in_32_seconds_l3504_350477


namespace square_perimeters_sum_l3504_350454

theorem square_perimeters_sum (a b : ℝ) (h1 : a^2 + b^2 = 130) (h2 : a^2 - b^2 = 50) :
  4*a + 4*b = 20 * Real.sqrt 10 := by
  sorry

end square_perimeters_sum_l3504_350454


namespace ending_number_proof_l3504_350425

/-- The ending number of the range [100, n] where the average of integers
    in [100, n] is 100 greater than the average of integers in [50, 250]. -/
def ending_number : ℕ :=
  400

theorem ending_number_proof :
  ∃ (n : ℕ),
    n ≥ 100 ∧
    (n + 100) / 2 = (250 + 50) / 2 + 100 ∧
    n = ending_number :=
by sorry

end ending_number_proof_l3504_350425


namespace expression_simplification_l3504_350409

theorem expression_simplification 
  (a c d x y : ℝ) 
  (h : c * x + d * y ≠ 0) : 
  (c * x * (a^2 * x^2 + 3 * a^2 * y^2 + c^2 * y^2) + 
   d * y * (a^2 * x^2 + 3 * c^2 * x^2 + c^2 * y^2)) / 
  (c * x + d * y) = 
  a^2 * x^2 + 3 * a * c * x * y + c^2 * y^2 := by
sorry


end expression_simplification_l3504_350409


namespace palabras_bookstore_workers_l3504_350447

theorem palabras_bookstore_workers (W : ℕ) : 
  W / 2 = W / 2 ∧  -- Half of workers read Saramago's book
  W / 6 = W / 6 ∧  -- 1/6 of workers read Kureishi's book
  (∃ (n : ℕ), n = 12 ∧ n ≤ W / 2 ∧ n ≤ W / 6) ∧  -- 12 workers read both books
  (W - (W / 2 + W / 6 - 12)) = ((W / 2 - 12) - 1) →  -- Workers who read neither book
  W = 210 := by
sorry

end palabras_bookstore_workers_l3504_350447


namespace complementary_angles_imply_right_triangle_l3504_350455

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define complementary angles
def complementary (a b : ℝ) : Prop := a + b = 90

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i = 90

-- Theorem statement
theorem complementary_angles_imply_right_triangle (t : Triangle) 
  (h : ∃ i j : Fin 3, i ≠ j ∧ complementary (t.angles i) (t.angles j)) : 
  is_right_triangle t := by
  sorry

end complementary_angles_imply_right_triangle_l3504_350455


namespace lucas_payment_l3504_350424

/-- Calculates the payment for window cleaning based on given conditions -/
def calculate_payment (floors : ℕ) (windows_per_floor : ℕ) (payment_per_window : ℕ) 
  (deduction_per_3_days : ℕ) (days_taken : ℕ) : ℕ :=
  let total_windows := floors * windows_per_floor
  let total_earned := total_windows * payment_per_window
  let deductions := (days_taken / 3) * deduction_per_3_days
  total_earned - deductions

/-- Theorem stating that Lucas will be paid $16 for cleaning windows -/
theorem lucas_payment : 
  calculate_payment 3 3 2 1 6 = 16 := by
  sorry

#eval calculate_payment 3 3 2 1 6

end lucas_payment_l3504_350424


namespace existence_of_prime_not_divisible_l3504_350457

theorem existence_of_prime_not_divisible (p : Nat) (h_prime : Prime p) (h_p_gt_2 : p > 2) :
  ∃ q : Nat, Prime q ∧ q < p ∧ ¬(p^2 ∣ q^(p-1) - 1) := by
  sorry

end existence_of_prime_not_divisible_l3504_350457


namespace T_is_three_rays_with_common_point_l3504_350483

/-- The set T of points (x,y) in the coordinate plane -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a b : ℝ), 
    ((a = 7 ∧ b = p.1 - 3) ∨ 
     (a = 7 ∧ b = p.2 + 5) ∨ 
     (a = p.1 - 3 ∧ b = p.2 + 5)) ∧
    (a = b) ∧
    (7 ≥ a ∧ p.1 - 3 ≥ a ∧ p.2 + 5 ≥ a)}

/-- A ray in the plane, defined by its starting point and direction -/
structure Ray where
  start : ℝ × ℝ
  direction : ℝ × ℝ

/-- The property that T consists of three rays with a common point -/
def isThreeRaysWithCommonPoint (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ) (r₁ r₂ r₃ : Ray),
    r₁.start = p ∧ r₂.start = p ∧ r₃.start = p ∧
    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    s = {q : ℝ × ℝ | ∃ (t : ℝ), t ≥ 0 ∧
      (q = r₁.start + t • r₁.direction ∨
       q = r₂.start + t • r₂.direction ∨
       q = r₃.start + t • r₃.direction)}

theorem T_is_three_rays_with_common_point : isThreeRaysWithCommonPoint T := by
  sorry

end T_is_three_rays_with_common_point_l3504_350483


namespace hyperbola_asymptote_l3504_350441

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the right vertex A
def right_vertex (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the circle centered at A
def circle_at_A (a r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + y^2 = r^2

-- Define the angle PAQ
def angle_PAQ (p q : ℝ × ℝ) : ℝ := sorry

-- Define the distance PQ
def distance_PQ (p q : ℝ × ℝ) : ℝ := sorry

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x

theorem hyperbola_asymptote
  (a b : ℝ)
  (p q : ℝ × ℝ)
  (h1 : hyperbola a b p.1 p.2)
  (h2 : hyperbola a b q.1 q.2)
  (h3 : ∃ r, circle_at_A a r p.1 p.2 ∧ circle_at_A a r q.1 q.2)
  (h4 : angle_PAQ p q = Real.pi / 3)
  (h5 : distance_PQ p q = Real.sqrt 3 / 3 * a) :
  asymptote_equation p.1 p.2 ∧ asymptote_equation q.1 q.2 :=
sorry

end hyperbola_asymptote_l3504_350441


namespace work_completion_time_l3504_350464

theorem work_completion_time (original_men : ℕ) (original_days : ℕ) (absent_men : ℕ) 
  (h1 : original_men = 180)
  (h2 : original_days = 55)
  (h3 : absent_men = 15) :
  (original_men * original_days) / (original_men - absent_men) = 60 := by
  sorry

end work_completion_time_l3504_350464


namespace hives_needed_for_candles_twenty_four_hives_for_ninety_six_candles_l3504_350463

/-- Given that 3 beehives make enough wax for 12 candles, 
    prove that 24 hives are needed to make 96 candles. -/
theorem hives_needed_for_candles : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun hives_given candles_given hives_needed candles_needed =>
    (hives_given * (candles_needed / candles_given) = hives_needed) →
    (3 * (96 / 12) = 24)

/-- The main theorem stating that 24 hives are needed for 96 candles. -/
theorem twenty_four_hives_for_ninety_six_candles :
  hives_needed_for_candles 3 12 24 96 := by
  sorry

end hives_needed_for_candles_twenty_four_hives_for_ninety_six_candles_l3504_350463


namespace constant_term_product_l3504_350408

-- Define polynomials p, q, r, and s
variable (p q r s : ℝ[X])

-- Define the relationship between s, p, q, and r
axiom h1 : s = p * q * r

-- Define the constant term of p as 2
axiom h2 : p.coeff 0 = 2

-- Define the constant term of s as 6
axiom h3 : s.coeff 0 = 6

-- Theorem to prove
theorem constant_term_product : q.coeff 0 * r.coeff 0 = 3 := by
  sorry

end constant_term_product_l3504_350408


namespace sampling_methods_correct_l3504_350401

/-- Represents a sampling method --/
inductive SamplingMethod
  | Systematic
  | SimpleRandom

/-- Represents a scenario for sampling --/
structure SamplingScenario where
  description : String
  interval : Option ℕ
  sampleSize : ℕ
  populationSize : ℕ

/-- Determines the appropriate sampling method for a given scenario --/
def determineSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  sorry

/-- The milk production line scenario --/
def milkProductionScenario : SamplingScenario :=
  { description := "Milk production line inspection"
  , interval := some 30
  , sampleSize := 1
  , populationSize := 0 }

/-- The math enthusiasts scenario --/
def mathEnthusiastsScenario : SamplingScenario :=
  { description := "Math enthusiasts study load"
  , interval := none
  , sampleSize := 3
  , populationSize := 30 }

theorem sampling_methods_correct :
  determineSamplingMethod milkProductionScenario = SamplingMethod.Systematic ∧
  determineSamplingMethod mathEnthusiastsScenario = SamplingMethod.SimpleRandom :=
  sorry

end sampling_methods_correct_l3504_350401


namespace mia_nia_difference_l3504_350458

/-- Represents a driving scenario with three drivers: Leo, Nia, and Mia. -/
structure DrivingScenario where
  /-- Leo's driving time in hours -/
  t : ℝ
  /-- Leo's driving speed in miles per hour -/
  s : ℝ
  /-- Leo's total distance driven in miles -/
  d : ℝ
  /-- Nia's total distance driven in miles -/
  nia_d : ℝ
  /-- Mia's total distance driven in miles -/
  mia_d : ℝ
  /-- Leo's distance equals speed times time -/
  leo_distance : d = s * t
  /-- Nia drove 2 hours longer than Leo at 10 mph faster -/
  nia_equation : nia_d = (s + 10) * (t + 2)
  /-- Mia drove 3 hours longer than Leo at 15 mph faster -/
  mia_equation : mia_d = (s + 15) * (t + 3)
  /-- Nia drove 110 miles more than Leo -/
  nia_leo_diff : nia_d = d + 110

/-- Theorem stating that Mia drove 100 miles more than Nia -/
theorem mia_nia_difference (scenario : DrivingScenario) : 
  scenario.mia_d - scenario.nia_d = 100 := by
  sorry

end mia_nia_difference_l3504_350458


namespace largest_n_for_inequality_l3504_350474

theorem largest_n_for_inequality (z : ℕ) (h : z = 9) :
  ∃ n : ℕ, (27 ^ z > 3 ^ n ∧ ∀ m : ℕ, m > n → 27 ^ z ≤ 3 ^ m) ∧ n = 26 := by
  sorry

end largest_n_for_inequality_l3504_350474


namespace intersection_of_M_and_N_l3504_350480

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ -1}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = N := by sorry

end intersection_of_M_and_N_l3504_350480


namespace roger_retirement_eligibility_l3504_350445

theorem roger_retirement_eligibility 
  (roger peter tom robert mike sarah laura james : ℕ) 
  (h1 : roger = peter + tom + robert + mike + sarah + laura + james)
  (h2 : peter = 12)
  (h3 : tom = 2 * robert)
  (h4 : robert = peter - 4)
  (h5 : robert = mike + 2)
  (h6 : sarah = mike + 3)
  (h7 : sarah = tom / 2)
  (h8 : laura = robert - mike)
  (h9 : james > 0) : 
  roger > 50 := by
  sorry

end roger_retirement_eligibility_l3504_350445


namespace total_distance_right_triangle_l3504_350400

/-- The total distance traveled in a right-angled triangle XYZ -/
theorem total_distance_right_triangle (XZ YZ XY : ℝ) : 
  XZ = 4000 →
  XY = 5000 →
  XZ^2 + YZ^2 = XY^2 →
  XZ + YZ + XY = 12000 := by
  sorry

end total_distance_right_triangle_l3504_350400


namespace units_digit_of_m_squared_plus_two_to_m_l3504_350487

def m : ℕ := 2023^2 + 2^2023

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : (m^2 + 2^m) % 10 = 7 := by
  sorry

end units_digit_of_m_squared_plus_two_to_m_l3504_350487


namespace lower_parallel_length_l3504_350446

/-- A triangle with a base of 20 inches and two parallel lines dividing it into four equal areas -/
structure EqualAreaTriangle where
  /-- The base of the triangle -/
  base : ℝ
  /-- The length of the parallel line closer to the base -/
  lower_parallel : ℝ
  /-- The base is 20 inches -/
  base_length : base = 20
  /-- The parallel lines divide the triangle into four equal areas -/
  equal_areas : lower_parallel^2 / base^2 = 1/4

/-- The length of the parallel line closer to the base is 10 inches -/
theorem lower_parallel_length (t : EqualAreaTriangle) : t.lower_parallel = 10 := by
  sorry

end lower_parallel_length_l3504_350446


namespace quadratic_root_property_l3504_350428

theorem quadratic_root_property : ∀ m n : ℝ,
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ x = m ∨ x = n) →
  m + n - m*n = 5 := by
  sorry

end quadratic_root_property_l3504_350428


namespace base_10_to_base_7_l3504_350484

theorem base_10_to_base_7 : 
  (1 * 7^3 + 5 * 7^2 + 1 * 7^1 + 5 * 7^0 : ℕ) = 600 := by
  sorry

#eval 1 * 7^3 + 5 * 7^2 + 1 * 7^1 + 5 * 7^0

end base_10_to_base_7_l3504_350484


namespace polynomial_remainder_l3504_350420

/-- The polynomial f(x) = x^4 - 4x^2 + 7 -/
def f (x : ℝ) : ℝ := x^4 - 4*x^2 + 7

/-- The remainder when f(x) is divided by (x - 1) -/
def remainder : ℝ := f 1

theorem polynomial_remainder : remainder = 4 := by
  sorry

end polynomial_remainder_l3504_350420


namespace power_of_two_equals_quadratic_plus_one_l3504_350415

theorem power_of_two_equals_quadratic_plus_one (x y : ℕ) :
  2^x = y^2 + y + 1 → x = 0 ∧ y = 0 := by
  sorry

end power_of_two_equals_quadratic_plus_one_l3504_350415


namespace two_digit_integer_problem_l3504_350490

theorem two_digit_integer_problem :
  ∃ (m n : ℕ),
    10 ≤ m ∧ m < 100 ∧
    10 ≤ n ∧ n < 100 ∧
    (m + n : ℚ) / 2 = n + m / 100 ∧
    m + n < 150 ∧
    m = 50 ∧ n = 49 :=
by sorry

end two_digit_integer_problem_l3504_350490


namespace shopping_mall_uses_systematic_sampling_l3504_350431

/-- Represents a sampling method with given characteristics -/
structure SamplingMethod where
  initialSelection : Bool  -- True if initial selection is random
  fixedInterval : Bool     -- True if subsequent selections are at fixed intervals
  equalGroups : Bool       -- True if population is divided into equal-sized groups

/-- Definition of systematic sampling method -/
def isSystematicSampling (method : SamplingMethod) : Prop :=
  method.initialSelection ∧ method.fixedInterval ∧ method.equalGroups

/-- The sampling method used by the shopping mall -/
def shoppingMallMethod : SamplingMethod :=
  { initialSelection := true,  -- Randomly select one stub
    fixedInterval := true,     -- Sequentially take stubs at fixed intervals (every 50)
    equalGroups := true }      -- Each group has 50 invoice stubs

/-- Theorem stating that the shopping mall's method is systematic sampling -/
theorem shopping_mall_uses_systematic_sampling :
  isSystematicSampling shoppingMallMethod := by
  sorry


end shopping_mall_uses_systematic_sampling_l3504_350431


namespace total_sand_weight_is_34_l3504_350419

/-- The number of buckets of sand carried by Eden -/
def eden_buckets : ℕ := 4

/-- The number of additional buckets Mary carried compared to Eden -/
def mary_extra_buckets : ℕ := 3

/-- The number of fewer buckets Iris carried compared to Mary -/
def iris_fewer_buckets : ℕ := 1

/-- The weight of sand in each bucket (in pounds) -/
def sand_per_bucket : ℕ := 2

/-- Calculates the total weight of sand collected by Eden, Mary, and Iris -/
def total_sand_weight : ℕ := 
  (eden_buckets + (eden_buckets + mary_extra_buckets) + 
   (eden_buckets + mary_extra_buckets - iris_fewer_buckets)) * sand_per_bucket

/-- Theorem stating that the total weight of sand collected is 34 pounds -/
theorem total_sand_weight_is_34 : total_sand_weight = 34 := by
  sorry

end total_sand_weight_is_34_l3504_350419


namespace lily_bought_ten_geese_l3504_350412

/-- The number of geese Lily bought -/
def lily_geese : ℕ := sorry

/-- The number of ducks Lily bought -/
def lily_ducks : ℕ := 20

/-- The number of ducks Rayden bought -/
def rayden_ducks : ℕ := 3 * lily_ducks

/-- The number of geese Rayden bought -/
def rayden_geese : ℕ := 4 * lily_geese

theorem lily_bought_ten_geese :
  lily_geese = 10 ∧
  rayden_ducks + rayden_geese = lily_ducks + lily_geese + 70 :=
by sorry

end lily_bought_ten_geese_l3504_350412


namespace triangle_data_uniqueness_l3504_350440

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- sides
  (α β γ : ℝ)  -- angles

-- Define the different sets of data
def ratio_two_sides_included_angle (t : Triangle) : ℝ × ℝ × ℝ := sorry
def ratios_angle_bisectors (t : Triangle) : ℝ × ℝ × ℝ := sorry
def ratios_medians (t : Triangle) : ℝ × ℝ × ℝ := sorry
def ratios_two_altitudes_bases (t : Triangle) : ℝ × ℝ := sorry
def two_angles_ratio_side_sum (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define a predicate for unique determination of triangle shape
def uniquely_determines_shape (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → t1 = t2

-- State the theorem
theorem triangle_data_uniqueness :
  uniquely_determines_shape ratio_two_sides_included_angle ∧
  uniquely_determines_shape ratios_medians ∧
  uniquely_determines_shape ratios_two_altitudes_bases ∧
  uniquely_determines_shape two_angles_ratio_side_sum ∧
  ¬ uniquely_determines_shape ratios_angle_bisectors :=
sorry

end triangle_data_uniqueness_l3504_350440


namespace train_length_calculation_l3504_350491

/-- Calculates the length of a train given the speeds of two trains traveling in opposite directions and the time taken for one train to pass an observer in the other train. -/
theorem train_length_calculation (woman_speed goods_speed : ℝ) (passing_time : ℝ) 
  (woman_speed_pos : 0 < woman_speed)
  (goods_speed_pos : 0 < goods_speed)
  (passing_time_pos : 0 < passing_time)
  (h_woman_speed : woman_speed = 25)
  (h_goods_speed : goods_speed = 142.986561075114)
  (h_passing_time : passing_time = 3) :
  ∃ (train_length : ℝ), abs (train_length - 38.932) < 0.001 :=
by sorry

end train_length_calculation_l3504_350491


namespace total_pears_l3504_350421

/-- Given 4 boxes of pears with 16 pears in each box, the total number of pears is 64. -/
theorem total_pears (num_boxes : ℕ) (pears_per_box : ℕ) 
  (h1 : num_boxes = 4) 
  (h2 : pears_per_box = 16) : 
  num_boxes * pears_per_box = 64 := by
  sorry

end total_pears_l3504_350421


namespace only_prop2_is_true_l3504_350417

-- Define the propositions
def prop1 : Prop := ∀ x : ℝ, (∃ y : ℝ, y^2 + 1 > 3*y) ↔ ¬(x^2 + 1 < 3*x)

def prop2 : Prop := ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

def prop3 : Prop := ∃ a : ℝ, (a > 2 → a > 5) ∧ ¬(a > 5 → a > 2)

def prop4 : Prop := ∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → (x*y ≠ 0)

-- Theorem stating that only prop2 is true
theorem only_prop2_is_true : ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4 := by
  sorry

end only_prop2_is_true_l3504_350417


namespace book_sale_gain_percentage_l3504_350406

/-- Calculates the gain percentage when selling a book -/
def gain_percentage (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem about the gain percentage of a book sale -/
theorem book_sale_gain_percentage 
  (loss_price : ℚ) 
  (gain_price : ℚ) 
  (loss_percentage : ℚ) :
  loss_price = 450 →
  gain_price = 550 →
  loss_percentage = 10 →
  ∃ (cost_price : ℚ), 
    cost_price * (1 - loss_percentage / 100) = loss_price ∧
    gain_percentage cost_price gain_price = 10 := by
  sorry

#eval gain_percentage 500 550

end book_sale_gain_percentage_l3504_350406


namespace geometric_sequence_sum_l3504_350468

/-- A geometric sequence {a_n} with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (a 1 + a 2 = 30) →
  (a 3 + a 4 = 60) →
  (a 7 + a 8 = (a 1 + a 2) * q^6) :=
by
  sorry

end geometric_sequence_sum_l3504_350468


namespace parabola_unique_coefficients_l3504_350448

/-- A parabola passing through three given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ
  p1 : eq (-2) = -16
  p2 : eq 2 = 8
  p3 : eq 4 = 36
  form : ∀ x, eq x = x^2 + b*x + c

/-- The unique values of b and c for the parabola -/
theorem parabola_unique_coefficients (p : Parabola) : p.b = 6 ∧ p.c = -8 := by
  sorry

end parabola_unique_coefficients_l3504_350448


namespace total_interest_is_350_l3504_350460

/-- Calculate the total interest amount for two loans over a specified period. -/
def totalInterest (loan1Amount : ℝ) (loan1Rate : ℝ) (loan2Amount : ℝ) (loan2Rate : ℝ) (years : ℝ) : ℝ :=
  (loan1Amount * loan1Rate * years) + (loan2Amount * loan2Rate * years)

/-- Theorem stating that the total interest for the given loans and time period is 350. -/
theorem total_interest_is_350 :
  totalInterest 1000 0.03 1200 0.05 3.888888888888889 = 350 := by
  sorry

#eval totalInterest 1000 0.03 1200 0.05 3.888888888888889

end total_interest_is_350_l3504_350460


namespace tug_of_war_competition_l3504_350482

/-- Calculates the number of matches in a tug-of-war competition -/
def number_of_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the number of matches for each class in a tug-of-war competition -/
def matches_per_class (n : ℕ) : ℕ := n - 1

theorem tug_of_war_competition (n : ℕ) (h : n = 7) :
  number_of_matches n = 21 ∧ matches_per_class n = 6 := by
  sorry

#eval number_of_matches 7
#eval matches_per_class 7

end tug_of_war_competition_l3504_350482


namespace line_segment_parameter_sum_of_squares_l3504_350493

/-- Given a line segment connecting (1, -3) and (4, 6), parameterized by x = pt + q and y = rt + s,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 100 -/
theorem line_segment_parameter_sum_of_squares :
  ∀ (p q r s : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = p * t + q ∧ y = r * t + s) →
  (q = 1 ∧ s = -3) →
  (p + q = 4 ∧ r + s = 6) →
  p^2 + q^2 + r^2 + s^2 = 100 := by
  sorry

end line_segment_parameter_sum_of_squares_l3504_350493


namespace triangle_inequality_sum_l3504_350492

theorem triangle_inequality_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + 2*b*c) / (b^2 + c^2) + (b^2 + 2*a*c) / (c^2 + a^2) + (c^2 + 2*a*b) / (a^2 + b^2) > 3 := by
  sorry

end triangle_inequality_sum_l3504_350492


namespace rent_and_earnings_increase_l3504_350443

theorem rent_and_earnings_increase (last_year_earnings : ℝ) (increase_percent : ℝ) : 
  (0.3 * (last_year_earnings * (1 + increase_percent / 100)) = 2.025 * (0.2 * last_year_earnings)) →
  increase_percent = 35 := by
  sorry

end rent_and_earnings_increase_l3504_350443


namespace line_inclination_45_degrees_l3504_350436

/-- The angle of inclination of the line x - y - √3 = 0 is 45° -/
theorem line_inclination_45_degrees :
  let line := {(x, y) : ℝ × ℝ | x - y - Real.sqrt 3 = 0}
  ∃ θ : ℝ, θ = 45 * π / 180 ∧ ∀ (x y : ℝ), (x, y) ∈ line → y = Real.tan θ * x + Real.sqrt 3 :=
by sorry

end line_inclination_45_degrees_l3504_350436


namespace notebook_payment_possible_l3504_350485

theorem notebook_payment_possible : ∃ (a b : ℕ), 27 * a - 16 * b = 1 := by
  sorry

end notebook_payment_possible_l3504_350485


namespace min_trig_expression_min_trig_expression_achievable_l3504_350444

theorem min_trig_expression (θ : Real) (h_acute : 0 < θ ∧ θ < π / 2) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) +
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) ≥ 2 :=
by sorry

theorem min_trig_expression_achievable :
  ∃ θ : Real, 0 < θ ∧ θ < π / 2 ∧
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) +
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 2 :=
by sorry

end min_trig_expression_min_trig_expression_achievable_l3504_350444


namespace sum_of_fifth_powers_l3504_350461

theorem sum_of_fifth_powers (n : ℕ) : 
  (∃ (A B C D E : ℤ), n = A^5 + B^5 + C^5 + D^5 + E^5) ∧ 
  (¬ ∃ (A B C D : ℤ), n = A^5 + B^5 + C^5 + D^5) := by
  sorry

#check sum_of_fifth_powers 2018

end sum_of_fifth_powers_l3504_350461


namespace lottery_probability_l3504_350413

theorem lottery_probability (total_tickets : ℕ) (cash_prizes : ℕ) (merch_prizes : ℕ) :
  total_tickets = 1000 →
  cash_prizes = 5 →
  merch_prizes = 20 →
  (cash_prizes + merch_prizes : ℚ) / total_tickets = 25 / 1000 := by
  sorry

end lottery_probability_l3504_350413


namespace total_spent_is_14_l3504_350499

/-- The cost of one set of barrettes in dollars -/
def barrette_cost : ℕ := 3

/-- The cost of one comb in dollars -/
def comb_cost : ℕ := 1

/-- The number of barrette sets Kristine buys -/
def kristine_barrettes : ℕ := 1

/-- The number of combs Kristine buys -/
def kristine_combs : ℕ := 1

/-- The number of barrette sets Crystal buys -/
def crystal_barrettes : ℕ := 3

/-- The number of combs Crystal buys -/
def crystal_combs : ℕ := 1

/-- The total amount spent by both Kristine and Crystal -/
def total_spent : ℕ := 
  (kristine_barrettes * barrette_cost + kristine_combs * comb_cost) +
  (crystal_barrettes * barrette_cost + crystal_combs * comb_cost)

theorem total_spent_is_14 : total_spent = 14 := by
  sorry

end total_spent_is_14_l3504_350499


namespace complement_of_A_wrt_U_l3504_350450

-- Define the universal set U
def U : Set ℝ := {x | x > 1}

-- Define the set A
def A : Set ℝ := {x | x > 2}

-- Define the complement of A with respect to U
def complement_U_A : Set ℝ := {x ∈ U | x ∉ A}

-- Theorem stating the complement of A with respect to U
theorem complement_of_A_wrt_U :
  complement_U_A = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end complement_of_A_wrt_U_l3504_350450


namespace quadratic_root_divisibility_l3504_350418

theorem quadratic_root_divisibility (a b c n : ℤ) 
  (h : a * n^2 + b * n + c = 0) : 
  c ∣ n := by
  sorry

end quadratic_root_divisibility_l3504_350418


namespace notch_volume_minimized_l3504_350472

/-- A cylindrical notch with angle θ between bounding planes -/
structure CylindricalNotch where
  θ : Real
  (θ_pos : θ > 0)
  (θ_lt_pi : θ < π)

/-- The volume of the notch given the angle φ between one bounding plane and the horizontal -/
noncomputable def notchVolume (n : CylindricalNotch) (φ : Real) : Real :=
  (2/3) * (Real.tan φ + Real.tan (n.θ - φ))

/-- Theorem: The volume of the notch is minimized when the bounding planes are at equal angles to the horizontal -/
theorem notch_volume_minimized (n : CylindricalNotch) :
  ∃ (φ_min : Real), φ_min = n.θ / 2 ∧
    ∀ (φ : Real), 0 < φ ∧ φ < n.θ → notchVolume n φ_min ≤ notchVolume n φ :=
sorry

end notch_volume_minimized_l3504_350472


namespace complex_number_solution_binomial_expansion_coefficient_l3504_350470

-- Part 1
def complex_number (b : ℝ) : ℂ := 3 + b * Complex.I

theorem complex_number_solution (b : ℝ) (h1 : b > 0) (h2 : ∃ (k : ℝ), (complex_number b - 2)^2 = k * Complex.I) :
  complex_number b = 3 + Complex.I := by sorry

-- Part 2
def binomial_sum (n : ℕ) : ℕ := 2^n

def expansion_term (n r : ℕ) (x : ℝ) : ℝ :=
  Nat.choose n r * 3^(n - r) * x^(n - 3/2 * r)

theorem binomial_expansion_coefficient (n : ℕ) (h : binomial_sum n = 16) :
  expansion_term n 2 1 = 54 := by sorry

end complex_number_solution_binomial_expansion_coefficient_l3504_350470


namespace p2023_coordinates_l3504_350422

/-- Transformation function that maps a point (x, y) to (-y+1, x+2) -/
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2 + 1, p.1 + 2)

/-- Function to apply the transformation n times -/
def iterate_transform (p : ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => p
  | n + 1 => transform (iterate_transform p n)

/-- The starting point P1 -/
def P1 : ℝ × ℝ := (2, 0)

theorem p2023_coordinates :
  iterate_transform P1 2023 = (-3, 3) := by
  sorry

end p2023_coordinates_l3504_350422


namespace sequence_problem_l3504_350404

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a_sum : a 1 + a 5 + a 9 = 9)
  (h_b_prod : b 2 * b 5 * b 8 = 3 * Real.sqrt 3) :
  (a 2 + a 8) / (1 + b 2 * b 8) = 3 / 2 := by
  sorry

end sequence_problem_l3504_350404


namespace other_factor_proof_l3504_350486

theorem other_factor_proof (a : ℕ) (h : a = 363) : 
  (a * 43 * 62 * 1311) / 33 = 38428986 := by
  sorry

end other_factor_proof_l3504_350486


namespace triangle_max_area_l3504_350452

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where a = 2 and 3b*sin(C) - 5c*sin(B)*cos(A) = 0, 
    the maximum area of the triangle is 10/3. -/
theorem triangle_max_area (a b c A B C : ℝ) : 
  a = 2 → 
  3 * b * Real.sin C - 5 * c * Real.sin B * Real.cos A = 0 → 
  (∃ (S : ℝ), S = (1/2) * a * b * Real.sin C ∧ 
    ∀ (S' : ℝ), S' = (1/2) * a * b * Real.sin C → S' ≤ S) →
  (1/2) * a * b * Real.sin C ≤ 10/3 :=
by sorry

end triangle_max_area_l3504_350452
