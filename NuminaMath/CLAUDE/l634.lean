import Mathlib

namespace fraction_sum_equality_l634_63405

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (36 - a) + b / (48 - b) + c / (72 - c) = 9) :
  4 / (36 - a) + 6 / (48 - b) + 9 / (72 - c) = 13 / 3 :=
by sorry

end fraction_sum_equality_l634_63405


namespace fraction_integer_condition_l634_63425

theorem fraction_integer_condition (n : ℤ) : 
  (↑(n + 1) / ↑(2 * n - 1) : ℚ).isInt ↔ n = 2 ∨ n = 1 ∨ n = 0 ∨ n = -1 :=
by sorry

end fraction_integer_condition_l634_63425


namespace chess_match_probability_l634_63488

/-- Given a chess match between players A and B, this theorem proves
    the probability of player B not losing, given the probabilities
    of a draw and player B winning. -/
theorem chess_match_probability (draw_prob win_prob : ℝ) :
  draw_prob = (1 : ℝ) / 2 →
  win_prob = (1 : ℝ) / 3 →
  draw_prob + win_prob = (5 : ℝ) / 6 :=
by sorry

end chess_match_probability_l634_63488


namespace smallest_upper_bound_D_l634_63483

def D (n : ℕ+) : ℚ := 5 - (2 * n.val + 5 : ℚ) / 2^n.val

theorem smallest_upper_bound_D :
  ∃ t : ℕ, (∀ n : ℕ+, D n < t) ∧ (∀ s : ℕ, s < t → ∃ m : ℕ+, D m ≥ s) :=
  sorry

end smallest_upper_bound_D_l634_63483


namespace unique_positive_integer_solution_l634_63472

-- Define the new operation ※
def star_op (a b : ℝ) : ℝ := a * b - a + b - 2

-- Theorem statement
theorem unique_positive_integer_solution :
  ∃! (x : ℕ), x > 0 ∧ star_op 3 (x : ℝ) < 2 :=
by sorry

end unique_positive_integer_solution_l634_63472


namespace trajectory_of_moving_circle_l634_63413

/-- The trajectory of the center of a moving circle -/
def trajectory_equation (x y : ℝ) : Prop := y^2 = 8*x

/-- A point on the circle -/
def fixed_point : ℝ × ℝ := (4, 0)

/-- Length of the chord on y-axis -/
def chord_length : ℝ := 8

theorem trajectory_of_moving_circle :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), (x - 4)^2 + y^2 = r^2) ∧ 
  (∃ (a : ℝ), a^2 + x^2 = (chord_length/2)^2) →
  trajectory_equation x y := by sorry

end trajectory_of_moving_circle_l634_63413


namespace lines_perp_to_plane_are_parallel_l634_63443

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perp_to_plane_are_parallel 
  (l m : Line) (α : Plane) 
  (h1 : l ≠ m) 
  (h2 : perp l α) 
  (h3 : perp m α) : 
  parallel l m :=
sorry

end lines_perp_to_plane_are_parallel_l634_63443


namespace kolya_mistake_l634_63480

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens < 10
  h_ones : ones < 10

/-- Represents a four-digit number of the form effe -/
structure FourDigitNumberEFFE where
  e : Nat
  f : Nat
  h_e : e < 10
  h_f : f < 10

/-- Function to check if a two-digit number is divisible by 11 -/
def isDivisibleBy11 (n : TwoDigitNumber) : Prop :=
  (n.tens - n.ones) % 11 = 0

/-- The main theorem -/
theorem kolya_mistake
  (ab cd : TwoDigitNumber)
  (effe : FourDigitNumberEFFE)
  (h_mult : ab.tens * 10 + ab.ones * cd.tens * 10 + cd.ones = effe.e * 1000 + effe.f * 100 + effe.f * 10 + effe.e)
  (h_distinct : ab.tens ≠ ab.ones ∧ cd.tens ≠ cd.ones ∧ ab.tens ≠ cd.tens ∧ ab.tens ≠ cd.ones ∧ ab.ones ≠ cd.tens ∧ ab.ones ≠ cd.ones)
  : isDivisibleBy11 ab ∨ isDivisibleBy11 cd :=
sorry

end kolya_mistake_l634_63480


namespace round_trip_time_l634_63499

/-- Calculates the total time for a round trip boat journey -/
theorem round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 9)
  (h2 : stream_speed = 6)
  (h3 : distance = 170) :
  (distance / (boat_speed - stream_speed)) + (distance / (boat_speed + stream_speed)) = 68 := by
  sorry

end round_trip_time_l634_63499


namespace distance_origin_to_line_through_focus_l634_63419

/-- Parabola type representing y^2 = 8x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line type -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Distance from a point to a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Theorem: Distance from origin to line through focus of parabola -/
theorem distance_origin_to_line_through_focus 
  (C : Parabola) 
  (l : Line) 
  (A B : ℝ × ℝ) :
  C.equation = (fun x y => y^2 = 8*x) →
  C.focus = (2, 0) →
  (∃ (x y : ℝ), l.equation x y ∧ C.equation x y) →
  (∃ (x y : ℝ), l.equation 2 0) →
  C.equation A.1 A.2 →
  C.equation B.1 B.2 →
  l.equation A.1 A.2 →
  l.equation B.1 B.2 →
  distance A B = 10 →
  distancePointToLine (0, 0) l = 4 * Real.sqrt 5 / 5 := by
    sorry

end distance_origin_to_line_through_focus_l634_63419


namespace fibonacci_seventh_term_l634_63414

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_seventh_term : fibonacci 6 = 13 := by
  sorry

end fibonacci_seventh_term_l634_63414


namespace rug_coverage_area_l634_63474

/-- Given three rugs with specified overlap areas, calculate the total floor area covered. -/
theorem rug_coverage_area (total_rug_area : ℝ) (double_layer_area : ℝ) (triple_layer_area : ℝ)
  (h1 : total_rug_area = 200)
  (h2 : double_layer_area = 24)
  (h3 : triple_layer_area = 19) :
  total_rug_area - double_layer_area - 2 * triple_layer_area = 138 := by
  sorry

#check rug_coverage_area

end rug_coverage_area_l634_63474


namespace base_conversion_equality_l634_63434

theorem base_conversion_equality (b : ℕ) : b > 0 → (
  4 * 5 + 3 = 1 * b^2 + 2 * b + 1 ↔ b = 4
) := by sorry

end base_conversion_equality_l634_63434


namespace greatest_perimeter_of_special_triangle_l634_63469

theorem greatest_perimeter_of_special_triangle : 
  let is_valid_triangle (x : ℕ) := x + 3*x > 15 ∧ x + 15 > 3*x ∧ 3*x + 15 > x
  let perimeter (x : ℕ) := x + 3*x + 15
  ∀ x : ℕ, is_valid_triangle x → perimeter x ≤ 43 ∧ ∃ y : ℕ, is_valid_triangle y ∧ perimeter y = 43 :=
by
  sorry

end greatest_perimeter_of_special_triangle_l634_63469


namespace B_power_100_l634_63420

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_100 : B^100 = B := by sorry

end B_power_100_l634_63420


namespace function_equivalence_l634_63454

theorem function_equivalence : ∀ x : ℝ, (3 * x)^3 = x := by
  sorry

end function_equivalence_l634_63454


namespace james_ownership_l634_63433

theorem james_ownership (total : ℕ) (difference : ℕ) (james : ℕ) (ali : ℕ) :
  total = 250 →
  difference = 40 →
  james = ali + difference →
  total = james + ali →
  james = 145 := by
sorry

end james_ownership_l634_63433


namespace book_selection_combinations_l634_63496

theorem book_selection_combinations :
  let mystery_count : ℕ := 5
  let fantasy_count : ℕ := 4
  let biography_count : ℕ := 6
  mystery_count * fantasy_count * biography_count = 120 := by
  sorry

end book_selection_combinations_l634_63496


namespace count_squares_l634_63440

/-- The number of groups of squares in the figure -/
def num_groups : ℕ := 5

/-- The number of squares in each group -/
def squares_per_group : ℕ := 5

/-- The total number of squares in the figure -/
def total_squares : ℕ := num_groups * squares_per_group

theorem count_squares : total_squares = 25 := by
  sorry

end count_squares_l634_63440


namespace ship_ratio_proof_l634_63404

theorem ship_ratio_proof (total_people : ℕ) (first_ship : ℕ) (ratio : ℚ) : 
  total_people = 847 →
  first_ship = 121 →
  first_ship + first_ship * ratio + first_ship * ratio^2 = total_people →
  ratio = 2 := by
sorry

end ship_ratio_proof_l634_63404


namespace z_coordinate_for_x_7_l634_63421

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- Given a line and an x-coordinate, find the corresponding z-coordinate -/
def find_z_coordinate (line : Line3D) (x : ℝ) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem z_coordinate_for_x_7 :
  let line := Line3D.mk (1, 3, 2) (4, 4, -1)
  find_z_coordinate line 7 = -4 := by sorry

end z_coordinate_for_x_7_l634_63421


namespace transportation_budget_theorem_l634_63431

def total_budget : ℝ := 1200000

def known_percentages : List ℝ := [39, 27, 14, 9, 5, 3.5]

def transportation_percentage : ℝ := 100 - (known_percentages.sum)

theorem transportation_budget_theorem :
  (transportation_percentage = 2.5) ∧
  (transportation_percentage / 100 * 360 = 9) ∧
  (transportation_percentage / 100 * 360 * π / 180 = π / 20) ∧
  (transportation_percentage / 100 * total_budget = 30000) :=
by sorry

end transportation_budget_theorem_l634_63431


namespace unique_brigade_solution_l634_63458

/-- Represents a brigade with newspapers and members -/
structure Brigade where
  newspapers : ℕ
  members : ℕ

/-- Properties of a valid brigade -/
def is_valid_brigade (b : Brigade) : Prop :=
  ∀ (m : ℕ) (n : ℕ), m ≤ b.members → n ≤ b.newspapers →
    (∃! (c : ℕ), c = 2) ∧  -- Each member reads exactly 2 newspapers
    (∃! (d : ℕ), d = 5) ∧  -- Each newspaper is read by exactly 5 members
    (∃! (e : ℕ), e = 1)    -- Each combination of 2 newspapers is read by exactly 1 member

/-- Theorem stating the unique solution for a valid brigade -/
theorem unique_brigade_solution (b : Brigade) (h : is_valid_brigade b) :
  b.newspapers = 6 ∧ b.members = 15 := by
  sorry

end unique_brigade_solution_l634_63458


namespace smallest_c_value_l634_63470

theorem smallest_c_value (a b c : ℤ) : 
  (b - a = c - b) →  -- arithmetic progression
  (c * c = a * b) →  -- geometric progression
  (∃ (a' b' c' : ℤ), b' - a' = c' - b' ∧ c' * c' = a' * b' ∧ c' < c) →
  c ≥ 2 :=
sorry

end smallest_c_value_l634_63470


namespace correct_scientific_statement_only_mathematical_models_correct_l634_63481

-- Define the type for scientific statements
inductive ScientificStatement
  | PopulationDensityEstimation
  | PreliminaryExperiment
  | MathematicalModels
  | SpeciesRichness

-- Define a function to check if a statement is correct
def isCorrectStatement (s : ScientificStatement) : Prop :=
  match s with
  | .MathematicalModels => True
  | _ => False

-- Theorem to prove
theorem correct_scientific_statement :
  ∃ (s : ScientificStatement), isCorrectStatement s :=
  sorry

-- Additional theorem to show that only MathematicalModels is correct
theorem only_mathematical_models_correct (s : ScientificStatement) :
  isCorrectStatement s ↔ s = ScientificStatement.MathematicalModels :=
  sorry

end correct_scientific_statement_only_mathematical_models_correct_l634_63481


namespace nested_sqrt_range_l634_63429

theorem nested_sqrt_range :
  ∃ y : ℝ, y = Real.sqrt (4 + y) ∧ 2 ≤ y ∧ y < 3 := by
  sorry

end nested_sqrt_range_l634_63429


namespace gcd_7_factorial_8_factorial_l634_63444

theorem gcd_7_factorial_8_factorial : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcd_7_factorial_8_factorial_l634_63444


namespace function_root_implies_a_range_l634_63452

theorem function_root_implies_a_range (a : ℝ) : 
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ a * x + 1 = 0) → (a < -1 ∨ a > 1) := by
  sorry

end function_root_implies_a_range_l634_63452


namespace cubic_double_root_abs_ab_l634_63460

/-- Given a cubic polynomial with a double root and an integer third root, prove |ab| = 3360 -/
theorem cubic_double_root_abs_ab (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℝ, (x - r)^2 * (x - s) = x^3 + a*x^2 + b*x + 16*a)) →
  |a * b| = 3360 :=
by sorry

end cubic_double_root_abs_ab_l634_63460


namespace salt_solution_mixture_l634_63423

theorem salt_solution_mixture (x : ℝ) : 
  (0.20 * x + 0.60 * 40 = 0.40 * (x + 40)) → x = 40 := by
  sorry

end salt_solution_mixture_l634_63423


namespace cindy_marbles_l634_63486

theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (remaining_multiplier : ℕ) (remaining_total : ℕ) :
  initial_marbles = 500 →
  friends = 4 →
  remaining_multiplier = 4 →
  remaining_total = 720 →
  remaining_multiplier * (initial_marbles - friends * (initial_marbles - (remaining_total / remaining_multiplier))) = remaining_total →
  initial_marbles - (remaining_total / remaining_multiplier) = friends * 80 :=
by sorry

end cindy_marbles_l634_63486


namespace train_speed_l634_63427

/-- The speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) (h1 : train_length = 357) 
  (h2 : bridge_length = 137) (h3 : time = 42.34285714285714) : 
  (train_length + bridge_length) / time = 11.66666666666667 := by
  sorry

end train_speed_l634_63427


namespace football_lineup_count_l634_63401

/-- The number of different lineups that can be created from a football team --/
def number_of_lineups (total_players : ℕ) (skilled_players : ℕ) : ℕ :=
  skilled_players * (total_players - 1) * (total_players - 2) * (total_players - 3) * (total_players - 4)

/-- Theorem stating that the number of lineups for a team of 15 players with 5 skilled players is 109200 --/
theorem football_lineup_count :
  number_of_lineups 15 5 = 109200 := by
  sorry

end football_lineup_count_l634_63401


namespace expand_and_simplify_expression_l634_63493

theorem expand_and_simplify_expression (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end expand_and_simplify_expression_l634_63493


namespace water_tank_capacity_l634_63446

theorem water_tank_capacity : ∀ (c : ℝ), c > 0 →
  (1 / 3 : ℝ) * c + 5 = (1 / 2 : ℝ) * c → c = 30 := by
  sorry

end water_tank_capacity_l634_63446


namespace burger_problem_l634_63475

theorem burger_problem (total_time : ℕ) (cook_time_per_side : ℕ) (grill_capacity : ℕ) (total_guests : ℕ) :
  total_time = 72 →
  cook_time_per_side = 4 →
  grill_capacity = 5 →
  total_guests = 30 →
  ∃ (burgers_per_half : ℕ),
    burgers_per_half * (total_guests / 2) + (total_guests / 2) = 
      (total_time / (2 * cook_time_per_side)) * grill_capacity ∧
    burgers_per_half = 2 :=
by sorry

end burger_problem_l634_63475


namespace juice_boxes_calculation_l634_63479

/-- Calculates the total number of juice boxes needed for a school year. -/
def total_juice_boxes (num_children : ℕ) (days_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  num_children * days_per_week * weeks_per_year

/-- Proves that the total number of juice boxes needed for the given conditions is 375. -/
theorem juice_boxes_calculation :
  let num_children : ℕ := 3
  let days_per_week : ℕ := 5
  let weeks_per_year : ℕ := 25
  total_juice_boxes num_children days_per_week weeks_per_year = 375 := by
  sorry


end juice_boxes_calculation_l634_63479


namespace sqrt_10_irrational_l634_63448

theorem sqrt_10_irrational : Irrational (Real.sqrt 10) := by sorry

end sqrt_10_irrational_l634_63448


namespace intersecting_lines_theorem_l634_63489

/-- Given two lines l₁ and l₂ that intersect at point P, and a third line l₃ -/
structure IntersectingLines where
  /-- The equation of line l₁ is 3x + 4y - 2 = 0 -/
  l₁ : ℝ → ℝ → Prop
  l₁_eq : ∀ x y, l₁ x y ↔ 3 * x + 4 * y - 2 = 0

  /-- The equation of line l₂ is 2x + y + 2 = 0 -/
  l₂ : ℝ → ℝ → Prop
  l₂_eq : ∀ x y, l₂ x y ↔ 2 * x + y + 2 = 0

  /-- P is the intersection point of l₁ and l₂ -/
  P : ℝ × ℝ
  P_on_l₁ : l₁ P.1 P.2
  P_on_l₂ : l₂ P.1 P.2

  /-- The equation of line l₃ is x - 2y - 1 = 0 -/
  l₃ : ℝ → ℝ → Prop
  l₃_eq : ∀ x y, l₃ x y ↔ x - 2 * y - 1 = 0

/-- The main theorem stating the equations of the two required lines -/
theorem intersecting_lines_theorem (g : IntersectingLines) :
  (∀ x y, x + y = 0 ↔ (∃ t : ℝ, x = t * g.P.1 ∧ y = t * g.P.2)) ∧
  (∀ x y, 2 * x + y + 2 = 0 ↔ (g.l₃ x y → (x - g.P.1) * 1 + (y - g.P.2) * 2 = 0)) :=
sorry

end intersecting_lines_theorem_l634_63489


namespace parallel_through_common_parallel_l634_63435

-- Define the types for lines and the parallel relation
variable {Line : Type}
variable (parallel : Line → Line → Prop)

-- State the axiom of parallels
axiom parallel_transitive {x y z : Line} : parallel x z → parallel y z → parallel x y

-- Theorem statement
theorem parallel_through_common_parallel (a b c : Line) :
  parallel a c → parallel b c → parallel a b :=
by sorry

end parallel_through_common_parallel_l634_63435


namespace sector_area_l634_63451

theorem sector_area (circumference : ℝ) (central_angle : ℝ) : 
  circumference = 16 * Real.pi → 
  central_angle = Real.pi / 4 → 
  (central_angle / (2 * Real.pi)) * ((circumference^2) / (4 * Real.pi)) = 8 * Real.pi := by
  sorry

end sector_area_l634_63451


namespace train_platform_crossing_time_l634_63465

/-- Given a train and platform with known dimensions, calculate the time to cross the platform. -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (time_to_cross_pole : ℝ) 
  (train_length_positive : 0 < train_length)
  (platform_length_positive : 0 < platform_length)
  (time_to_cross_pole_positive : 0 < time_to_cross_pole) :
  (train_length + platform_length) / (train_length / time_to_cross_pole) = 
    (train_length + platform_length) * time_to_cross_pole / train_length := by
  sorry

end train_platform_crossing_time_l634_63465


namespace product_of_square_roots_l634_63487

theorem product_of_square_roots : 
  let P : ℝ := Real.sqrt 2025 + Real.sqrt 2024
  let Q : ℝ := -Real.sqrt 2025 - Real.sqrt 2024
  let R : ℝ := Real.sqrt 2025 - Real.sqrt 2024
  let S : ℝ := Real.sqrt 2024 - Real.sqrt 2025
  P * Q * R * S = -1 := by
sorry

end product_of_square_roots_l634_63487


namespace solution_set_part1_range_of_a_part2_l634_63495

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end solution_set_part1_range_of_a_part2_l634_63495


namespace square_area_error_l634_63442

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
  sorry

end square_area_error_l634_63442


namespace subscription_difference_is_4000_l634_63406

/-- Represents the subscription amounts and profit distribution in a business venture. -/
structure BusinessVenture where
  total_subscription : ℕ
  total_profit : ℕ
  c_profit : ℕ
  b_extra : ℕ

/-- Calculates the difference between A's and B's subscriptions. -/
def subscription_difference (bv : BusinessVenture) : ℕ :=
  let c_subscription := bv.c_profit * bv.total_subscription / bv.total_profit
  let b_subscription := c_subscription + bv.b_extra
  let a_subscription := bv.total_subscription - b_subscription - c_subscription
  a_subscription - b_subscription

/-- Theorem stating that the difference between A's and B's subscriptions is 4000. -/
theorem subscription_difference_is_4000 :
  subscription_difference ⟨50000, 35000, 8400, 5000⟩ = 4000 := by
  sorry


end subscription_difference_is_4000_l634_63406


namespace total_machine_time_for_dolls_and_accessories_l634_63462

/-- Calculates the total combined machine operation time for manufacturing dolls and accessories -/
def totalMachineTime (numDolls : ℕ) (numAccessoriesPerDoll : ℕ) (dollTime : ℕ) (accessoryTime : ℕ) : ℕ :=
  numDolls * dollTime + numDolls * numAccessoriesPerDoll * accessoryTime

/-- The number of dolls manufactured -/
def dollCount : ℕ := 12000

/-- The number of accessories per doll -/
def accessoriesPerDoll : ℕ := 2 + 3 + 1 + 5

/-- Time taken to manufacture one doll (in seconds) -/
def dollManufactureTime : ℕ := 45

/-- Time taken to manufacture one accessory (in seconds) -/
def accessoryManufactureTime : ℕ := 10

theorem total_machine_time_for_dolls_and_accessories :
  totalMachineTime dollCount accessoriesPerDoll dollManufactureTime accessoryManufactureTime = 1860000 := by
  sorry

end total_machine_time_for_dolls_and_accessories_l634_63462


namespace suit_price_problem_l634_63468

theorem suit_price_problem (P : ℝ) : 
  (0.7 * (1.3 * P) = 182) → P = 200 := by
  sorry

end suit_price_problem_l634_63468


namespace petrol_price_increase_l634_63497

theorem petrol_price_increase (original_price original_consumption : ℝ) 
  (h : original_price > 0) (h2 : original_consumption > 0) :
  let new_consumption := original_consumption * (1 - 1/6)
  let price_increase_factor := (original_price * original_consumption) / (original_price * new_consumption)
  price_increase_factor = 1.2 := by
sorry

end petrol_price_increase_l634_63497


namespace new_cost_is_fifty_l634_63428

/-- Represents the manufacturing cost and profit scenario for Crazy Eddie's key chains --/
structure KeyChainScenario where
  initialCost : ℝ
  initialProfitRate : ℝ
  newProfitRate : ℝ
  sellingPrice : ℝ

/-- Calculates the new manufacturing cost given a KeyChainScenario --/
def newManufacturingCost (scenario : KeyChainScenario) : ℝ :=
  scenario.sellingPrice * (1 - scenario.newProfitRate)

/-- Theorem stating that under the given conditions, the new manufacturing cost is $50 --/
theorem new_cost_is_fifty :
  ∀ (scenario : KeyChainScenario),
    scenario.initialCost = 70 ∧
    scenario.initialProfitRate = 0.3 ∧
    scenario.newProfitRate = 0.5 ∧
    scenario.sellingPrice = scenario.initialCost / (1 - scenario.initialProfitRate) →
    newManufacturingCost scenario = 50 := by
  sorry


end new_cost_is_fifty_l634_63428


namespace count_valid_a_l634_63445

theorem count_valid_a : ∃! n : ℕ, n > 0 ∧ 
  (∃ a_set : Finset ℕ, 
    (∀ a ∈ a_set, a > 0 ∧ 3 ∣ a ∧ a ∣ 18 ∧ a ∣ 27) ∧
    (∀ a : ℕ, a > 0 → 3 ∣ a → a ∣ 18 → a ∣ 27 → a ∈ a_set) ∧
    Finset.card a_set = n) :=
by sorry

end count_valid_a_l634_63445


namespace shaded_percentage_7x7_grid_l634_63437

/-- The percentage of shaded squares in a 7x7 grid with 20 shaded squares -/
theorem shaded_percentage_7x7_grid (total_squares : Nat) (shaded_squares : Nat) :
  total_squares = 7 * 7 →
  shaded_squares = 20 →
  (shaded_squares : Real) / total_squares * 100 = 20 / 49 * 100 := by
  sorry

end shaded_percentage_7x7_grid_l634_63437


namespace max_remainder_two_digit_div_sum_digits_l634_63491

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The theorem stating that the maximum remainder when dividing a two-digit number
    by the sum of its digits is 15 -/
theorem max_remainder_two_digit_div_sum_digits :
  ∃ (n : ℕ), TwoDigitNumber n ∧
    ∀ (m : ℕ), TwoDigitNumber m →
      n % (sumOfDigits n) ≥ m % (sumOfDigits m) ∧
      n % (sumOfDigits n) = 15 :=
sorry

end max_remainder_two_digit_div_sum_digits_l634_63491


namespace difference_of_unit_vectors_with_sum_magnitude_one_l634_63490

/-- Given two unit vectors a and b in a real inner product space such that
    the magnitude of their sum is 1, prove that the magnitude of their
    difference is √3. -/
theorem difference_of_unit_vectors_with_sum_magnitude_one
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hab : ‖a + b‖ = 1) :
  ‖a - b‖ = Real.sqrt 3 := by
  sorry

end difference_of_unit_vectors_with_sum_magnitude_one_l634_63490


namespace tensor_equation_solution_l634_63461

/-- Custom binary operation ⊗ -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

theorem tensor_equation_solution :
  ∀ m : ℝ, m > 0 → tensor 1 m = 3 → m = 1 := by
sorry

end tensor_equation_solution_l634_63461


namespace truck_toll_theorem_l634_63411

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on each other axle -/
def calculateAxles (totalWheels : ℕ) (frontAxleWheels : ℕ) (otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

/-- Calculates the toll for a truck given the number of axles -/
def calculateToll (axles : ℕ) : ℚ :=
  3.5 + 0.5 * (axles - 2)

theorem truck_toll_theorem :
  let totalWheels : ℕ := 18
  let frontAxleWheels : ℕ := 2
  let otherAxleWheels : ℕ := 4
  let axles : ℕ := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  calculateToll axles = 5 := by
  sorry

end truck_toll_theorem_l634_63411


namespace smallest_invertible_domain_l634_63473

/-- The function g(x) = (x-3)^2 + 1 -/
def g (x : ℝ) : ℝ := (x - 3)^2 + 1

/-- g is invertible on [c,∞) -/
def invertible_on (c : ℝ) : Prop :=
  ∀ x y, x ≥ c → y ≥ c → g x = g y → x = y

/-- The smallest value of c for which g is invertible on [c,∞) -/
theorem smallest_invertible_domain : 
  (∃ c, invertible_on c ∧ ∀ c', c' < c → ¬invertible_on c') ∧ 
  (∀ c, invertible_on c → c ≥ 3) := by
  sorry

end smallest_invertible_domain_l634_63473


namespace expression_evaluation_l634_63449

theorem expression_evaluation (a b : ℤ) (h1 : a = 1) (h2 : b = -2) : 
  -2*a - 2*b^2 + 3*a*b - b^3 = -8 := by sorry

end expression_evaluation_l634_63449


namespace range_of_x_l634_63498

theorem range_of_x (x : ℝ) : 
  0 ≤ x → x < 2 * Real.pi → Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x →
  π / 4 ≤ x ∧ x ≤ 5 * π / 4 :=
by sorry

end range_of_x_l634_63498


namespace snooker_tournament_revenue_l634_63408

theorem snooker_tournament_revenue
  (total_tickets : ℕ)
  (vip_price general_price : ℚ)
  (fewer_vip : ℕ) :
  total_tickets = 320 →
  vip_price = 40 →
  general_price = 15 →
  fewer_vip = 212 →
  ∃ (vip_tickets general_tickets : ℕ),
    vip_tickets + general_tickets = total_tickets ∧
    vip_tickets = general_tickets - fewer_vip ∧
    vip_price * vip_tickets + general_price * general_tickets = 6150 :=
by sorry

end snooker_tournament_revenue_l634_63408


namespace y_intercept_approx_20_l634_63482

/-- A straight line in the xy-plane with given slope and point -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Theorem: The y-intercept of the given line is approximately 20 -/
theorem y_intercept_approx_20 (l : Line) 
  (h1 : l.slope = 3.8666666666666667)
  (h2 : l.point = (150, 600)) :
  ∃ ε > 0, |y_intercept l - 20| < ε :=
sorry

end y_intercept_approx_20_l634_63482


namespace painting_time_with_break_l634_63407

/-- The time it takes to paint a room together, including a break -/
theorem painting_time_with_break (doug_rate dave_rate ella_rate : ℝ) 
  (break_time : ℝ) (h1 : doug_rate = 1 / 5) (h2 : dave_rate = 1 / 7) 
  (h3 : ella_rate = 1 / 10) (h4 : break_time = 2) : 
  ∃ t : ℝ, (doug_rate + dave_rate + ella_rate) * (t - break_time) = 1 ∧ t = 132 / 31 := by
  sorry

end painting_time_with_break_l634_63407


namespace square_sum_given_diff_and_product_l634_63484

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a * b = 9) : 
  a^2 + b^2 = 27 := by
sorry

end square_sum_given_diff_and_product_l634_63484


namespace trajectory_is_ellipse_l634_63430

-- Define the complex plane
def ComplexPlane := ℂ

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define the condition for the set of points
def SatisfiesCondition (z : ℂ) : Prop :=
  Complex.abs (z - i) + Complex.abs (z + i) = 3

-- Define the set of points satisfying the condition
def PointSet : Set ℂ :=
  {z : ℂ | SatisfiesCondition z}

-- Theorem statement
theorem trajectory_is_ellipse :
  ∃ (a b : ℝ) (center : ℂ), 
    a > 0 ∧ b > 0 ∧ a ≠ b ∧
    PointSet = {z : ℂ | (z.re - center.re)^2 / a^2 + (z.im - center.im)^2 / b^2 = 1} :=
sorry

end trajectory_is_ellipse_l634_63430


namespace sum_of_two_smallest_numbers_l634_63464

theorem sum_of_two_smallest_numbers : ∀ (a b c : ℕ), 
  a = 10 ∧ b = 11 ∧ c = 12 → 
  min a (min b c) + min (max a b) (min b c) = 21 :=
by
  sorry

end sum_of_two_smallest_numbers_l634_63464


namespace set_intersection_theorem_l634_63455

def P : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x : ℝ | x^2 ≥ 4}

theorem set_intersection_theorem :
  P ∩ (Set.univ \ Q) = Set.Icc 0 2 := by sorry

end set_intersection_theorem_l634_63455


namespace message_sending_methods_l634_63403

/-- The number of friends the student has -/
def num_friends : ℕ := 4

/-- The number of suitable messages in the draft box -/
def num_messages : ℕ := 3

/-- The number of different methods to send messages -/
def num_methods : ℕ := num_messages ^ num_friends

/-- Theorem stating that the number of different methods to send messages is 81 -/
theorem message_sending_methods : num_methods = 81 := by
  sorry

end message_sending_methods_l634_63403


namespace coin_toss_outcomes_l634_63417

/-- The number of possible outcomes when throwing three coins -/
def coin_outcomes : ℕ := 8

/-- The number of coins being thrown -/
def num_coins : ℕ := 3

/-- The number of possible states for each coin (heads or tails) -/
def states_per_coin : ℕ := 2

/-- Theorem stating that the number of possible outcomes when throwing three coins,
    each with two possible states, is equal to 8 -/
theorem coin_toss_outcomes :
  coin_outcomes = states_per_coin ^ num_coins :=
by sorry

end coin_toss_outcomes_l634_63417


namespace additional_cartons_needed_l634_63459

/-- Given the total required cartons, the number of strawberry cartons, and the number of blueberry cartons,
    prove that the additional cartons needed is equal to the total required minus the sum of strawberry and blueberry cartons. -/
theorem additional_cartons_needed
  (total_required : ℕ)
  (strawberry_cartons : ℕ)
  (blueberry_cartons : ℕ)
  (h : total_required = 42 ∧ strawberry_cartons = 2 ∧ blueberry_cartons = 7) :
  total_required - (strawberry_cartons + blueberry_cartons) = 33 :=
by sorry

end additional_cartons_needed_l634_63459


namespace B_power_150_l634_63456

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150 : B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by sorry

end B_power_150_l634_63456


namespace tournament_outcomes_l634_63426

/-- Represents a tournament with n players --/
def Tournament (n : ℕ) := ℕ

/-- The number of possible outcomes in a tournament --/
def possibleOutcomes (t : Tournament n) : ℕ := 2^(n-1)

theorem tournament_outcomes (t : Tournament 6) : 
  possibleOutcomes t = 32 := by
  sorry

#check tournament_outcomes

end tournament_outcomes_l634_63426


namespace greatest_prime_factor_of_expression_l634_63424

theorem greatest_prime_factor_of_expression :
  ∃ (p : ℕ), p.Prime ∧ p ∣ (3^8 + 6^7) ∧ ∀ (q : ℕ), q.Prime → q ∣ (3^8 + 6^7) → q ≤ p ∧ p = 131 :=
by sorry

end greatest_prime_factor_of_expression_l634_63424


namespace orange_apple_cost_difference_l634_63418

/-- The cost difference between an orange and an apple -/
def cost_difference (apple_cost orange_cost : ℚ) : ℚ := orange_cost - apple_cost

theorem orange_apple_cost_difference 
  (apple_cost orange_cost : ℚ) 
  (total_cost : ℚ)
  (h1 : apple_cost > 0)
  (h2 : orange_cost > apple_cost)
  (h3 : 3 * apple_cost + 7 * orange_cost = total_cost)
  (h4 : total_cost = 456/100) : 
  ∃ (diff : ℚ), cost_difference apple_cost orange_cost = diff ∧ diff > 0 := by
  sorry

#eval cost_difference (26/100) (36/100)

end orange_apple_cost_difference_l634_63418


namespace haj_daily_cost_l634_63466

/-- The daily operation cost for Mr. Haj's grocery store -/
def daily_cost : ℝ → Prop := λ T => 
  -- 2/5 of total cost is for salary
  let salary := (2/5) * T
  -- Remaining after salary
  let remaining_after_salary := T - salary
  -- 1/4 of remaining after salary is for delivery
  let delivery := (1/4) * remaining_after_salary
  -- Amount for orders
  let orders := 1800
  -- Total cost equals sum of salary, delivery, and orders
  T = salary + delivery + orders

/-- Theorem stating the daily operation cost for Mr. Haj's grocery store -/
theorem haj_daily_cost : ∃ T : ℝ, daily_cost T ∧ T = 8000 := by
  sorry

end haj_daily_cost_l634_63466


namespace triangle_ad_length_l634_63471

/-- Triangle ABC with perpendicular from A to BC at point D -/
structure Triangle :=
  (A B C D : ℝ × ℝ)
  (AB : ℝ)
  (AC : ℝ)
  (BD : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (is_right_angle : (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0)
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = AB)
  (AC_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = AC)
  (BD_CD_ratio : BD / CD = 2 / 5)

/-- Theorem: In triangle ABC, if AB = 10, AC = 17, D is the foot of the perpendicular from A to BC,
    and BD:CD = 2:5, then AD = 8 -/
theorem triangle_ad_length (t : Triangle) (h1 : t.AB = 10) (h2 : t.AC = 17) : t.AD = 8 := by
  sorry


end triangle_ad_length_l634_63471


namespace problem_solution_l634_63467

theorem problem_solution (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : 3 * m + 2 * n = 225) (h4 : Nat.gcd m n = 15) : m + n = 105 := by
  sorry

end problem_solution_l634_63467


namespace subset_implies_a_equals_one_l634_63447

-- Define the sets A and B
def A : Set ℝ := {-1, 0, 2}
def B (a : ℝ) : Set ℝ := {2^a}

-- State the theorem
theorem subset_implies_a_equals_one (a : ℝ) : B a ⊆ A → a = 1 := by
  sorry

end subset_implies_a_equals_one_l634_63447


namespace select_representatives_count_l634_63409

/-- The number of ways to select subject representatives -/
def num_ways_to_select_representatives (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2).choose 2

/-- Theorem stating that selecting 4 students from 5 for specific subject representations results in 60 different ways -/
theorem select_representatives_count :
  num_ways_to_select_representatives 5 = 60 := by
  sorry

end select_representatives_count_l634_63409


namespace pythagorean_triple_6_8_10_l634_63478

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_6_8_10 :
  (is_pythagorean_triple 6 8 10) ∧
  ¬(is_pythagorean_triple 6 7 10) ∧
  ¬(is_pythagorean_triple 1 2 3) ∧
  ¬(is_pythagorean_triple 4 5 8) :=
by sorry

end pythagorean_triple_6_8_10_l634_63478


namespace no_positive_integer_solutions_l634_63453

theorem no_positive_integer_solutions :
  ¬ ∃ (x y : ℕ+), x^2 - 2*y^2 = 5 := by
  sorry

end no_positive_integer_solutions_l634_63453


namespace koolaid_percentage_is_four_percent_l634_63477

/-- Calculates the percentage of koolaid powder in a mixture --/
def koolaid_percentage (initial_powder : ℚ) (initial_water : ℚ) (evaporated_water : ℚ) : ℚ :=
  let remaining_water := initial_water - evaporated_water
  let final_water := 4 * remaining_water
  let total_volume := final_water + initial_powder
  (initial_powder / total_volume) * 100

/-- Theorem stating that the percentage of koolaid powder is 4% given the initial conditions --/
theorem koolaid_percentage_is_four_percent :
  koolaid_percentage 2 16 4 = 4 := by sorry

end koolaid_percentage_is_four_percent_l634_63477


namespace apples_left_l634_63450

/-- The number of bags with 20 apples each -/
def bags_20 : ℕ := 4

/-- The number of apples in each of the first type of bags -/
def apples_per_bag_20 : ℕ := 20

/-- The number of bags with 25 apples each -/
def bags_25 : ℕ := 6

/-- The number of apples in each of the second type of bags -/
def apples_per_bag_25 : ℕ := 25

/-- The number of apples Ella sells -/
def apples_sold : ℕ := 200

/-- The theorem stating that Ella has 30 apples left -/
theorem apples_left : 
  bags_20 * apples_per_bag_20 + bags_25 * apples_per_bag_25 - apples_sold = 30 := by
  sorry

end apples_left_l634_63450


namespace sin_150_cos_30_plus_cos_150_sin_30_eq_zero_l634_63432

theorem sin_150_cos_30_plus_cos_150_sin_30_eq_zero : 
  Real.sin (150 * π / 180) * Real.cos (30 * π / 180) + 
  Real.cos (150 * π / 180) * Real.sin (30 * π / 180) = 0 := by
  sorry

end sin_150_cos_30_plus_cos_150_sin_30_eq_zero_l634_63432


namespace vector_difference_magnitude_l634_63416

theorem vector_difference_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : ‖a + b‖ = Real.sqrt 19) : 
  ‖a - b‖ = Real.sqrt 7 := by
  sorry

end vector_difference_magnitude_l634_63416


namespace calculate_expression_l634_63476

theorem calculate_expression : -2^4 + 3 * (-1)^2010 - (-2)^2 = -17 := by
  sorry

end calculate_expression_l634_63476


namespace rearranged_triple_divisible_by_27_l634_63441

/-- Given a natural number, rearranging its digits to get a number
    that is three times the original results in a number divisible by 27. -/
theorem rearranged_triple_divisible_by_27 (n m : ℕ) :
  (∃ (f : ℕ → ℕ), f n = m) →  -- n and m have the same digits (rearranged)
  m = 3 * n →                 -- m is three times n
  27 ∣ m :=                   -- m is divisible by 27
by sorry

end rearranged_triple_divisible_by_27_l634_63441


namespace celsius_to_fahrenheit_constant_is_zero_l634_63415

/-- The conversion factor from Celsius to Fahrenheit -/
def celsius_to_fahrenheit_factor : ℚ := 9 / 5

/-- The change in Fahrenheit temperature -/
def fahrenheit_change : ℚ := 26

/-- The change in Celsius temperature -/
def celsius_change : ℚ := 14.444444444444445

/-- The constant in the Celsius to Fahrenheit conversion formula -/
def celsius_to_fahrenheit_constant : ℚ := 0

theorem celsius_to_fahrenheit_constant_is_zero :
  celsius_to_fahrenheit_constant = 0 := by sorry

end celsius_to_fahrenheit_constant_is_zero_l634_63415


namespace extended_pattern_ratio_l634_63438

/-- Represents the tile configuration of a square pattern -/
structure TilePattern where
  black : ℕ
  white : ℕ

/-- Extends a tile pattern by adding two borders of black tiles -/
def extendPattern (p : TilePattern) : TilePattern :=
  let side := Nat.sqrt (p.black + p.white)
  let newBlack := p.black + 4 * side + 4 * (side - 1) + 4
  { black := newBlack, white := p.white }

/-- The ratio of black to white tiles in a pattern -/
def blackWhiteRatio (p : TilePattern) : ℚ :=
  p.black / p.white

theorem extended_pattern_ratio :
  let original := TilePattern.mk 10 26
  let extended := extendPattern original
  blackWhiteRatio extended = 37 / 13 := by
  sorry

end extended_pattern_ratio_l634_63438


namespace sufficient_not_necessary_condition_l634_63422

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x - 1 > 0 → x^2 - 1 > 0) ∧
  (∃ x, x^2 - 1 > 0 ∧ ¬(x - 1 > 0)) :=
sorry

end sufficient_not_necessary_condition_l634_63422


namespace point_A_l634_63492

def point_A : ℝ × ℝ := (-2, 4)

def move_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

def move_left (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 - units, p.2)

def point_A' : ℝ × ℝ :=
  move_left (move_up point_A 2) 3

theorem point_A'_coordinates :
  point_A' = (-5, 6) := by
  sorry

end point_A_l634_63492


namespace rectangle_area_l634_63412

/-- Given a rectangle with perimeter 80 meters and length three times the width, 
    prove that its area is 300 square meters. -/
theorem rectangle_area (l w : ℝ) 
  (perimeter_eq : 2 * l + 2 * w = 80)
  (length_width_relation : l = 3 * w) : 
  l * w = 300 := by
  sorry

end rectangle_area_l634_63412


namespace inequality_system_solution_l634_63402

theorem inequality_system_solution :
  ∀ x : ℝ, (3 * x + 1 ≥ 7 ∧ 4 * x - 3 < 9) ↔ (2 ≤ x ∧ x < 3) := by
  sorry

end inequality_system_solution_l634_63402


namespace floor_neg_seven_fourths_l634_63463

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end floor_neg_seven_fourths_l634_63463


namespace photo_comparison_l634_63457

theorem photo_comparison (claire lisa robert : ℕ) 
  (h1 : lisa = 3 * claire)
  (h2 : robert = lisa)
  : robert = 2 * claire + claire := by
  sorry

end photo_comparison_l634_63457


namespace product_lmn_equals_one_l634_63494

theorem product_lmn_equals_one 
  (p q r l m n : ℂ)
  (distinct_pqr : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (distinct_lmn : l ≠ m ∧ m ≠ n ∧ l ≠ n)
  (nonzero_lmn : l ≠ 0 ∧ m ≠ 0 ∧ n ≠ 0)
  (eq1 : p / (1 - q) = l)
  (eq2 : q / (1 - r) = m)
  (eq3 : r / (1 - p) = n) :
  l * m * n = 1 := by
  sorry

end product_lmn_equals_one_l634_63494


namespace smallest_integer_y_five_satisfies_inequality_smallest_integer_is_five_l634_63436

theorem smallest_integer_y (y : ℤ) : (7 + 3 * y < 25) ↔ y ≤ 5 := by sorry

theorem five_satisfies_inequality : 7 + 3 * 5 < 25 := by sorry

theorem smallest_integer_is_five : ∃ (y : ℤ), y = 5 ∧ (7 + 3 * y < 25) ∧ ∀ (z : ℤ), (7 + 3 * z < 25) → z ≥ y := by sorry

end smallest_integer_y_five_satisfies_inequality_smallest_integer_is_five_l634_63436


namespace hyperbola_condition_l634_63439

/-- A hyperbola is represented by an equation of the form ax²/p + by²/q = 1, 
    where a and b are non-zero real numbers with opposite signs, 
    and p and q are non-zero real numbers. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  p : ℝ
  q : ℝ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  p_nonzero : p ≠ 0
  q_nonzero : q ≠ 0
  opposite_signs : a * b < 0

/-- The equation x²/(k-1) + y²/(k+1) = 1 -/
def equation (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) + y^2 / (k + 1) = 1

/-- The condition -1 < k < 1 -/
def condition (k : ℝ) : Prop :=
  -1 < k ∧ k < 1

/-- Theorem stating that the condition is necessary and sufficient 
    for the equation to represent a hyperbola -/
theorem hyperbola_condition (k : ℝ) : 
  (∃ h : Hyperbola, equation k ↔ h.a * x^2 / h.p + h.b * y^2 / h.q = 1) ↔ condition k :=
sorry

end hyperbola_condition_l634_63439


namespace calories_left_for_dinner_l634_63410

def daily_calorie_limit : ℕ := 2200
def breakfast_calories : ℕ := 353
def lunch_calories : ℕ := 885
def snack_calories : ℕ := 130

theorem calories_left_for_dinner :
  daily_calorie_limit - (breakfast_calories + lunch_calories + snack_calories) = 832 := by
  sorry

end calories_left_for_dinner_l634_63410


namespace roller_derby_team_size_l634_63485

theorem roller_derby_team_size :
  ∀ (num_teams : ℕ) (skates_per_member : ℕ) (laces_per_skate : ℕ) (total_laces : ℕ),
    num_teams = 4 →
    skates_per_member = 2 →
    laces_per_skate = 3 →
    total_laces = 240 →
    ∃ (members_per_team : ℕ),
      members_per_team * num_teams * skates_per_member * laces_per_skate = total_laces ∧
      members_per_team = 10 :=
by sorry

end roller_derby_team_size_l634_63485


namespace f_at_5_eq_neg_13_l634_63400

/-- A polynomial function of degree 7 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

/-- Theorem stating that f(5) = -13 given f(-5) = 17 -/
theorem f_at_5_eq_neg_13 {a b c : ℝ} (h : f a b c (-5) = 17) : f a b c 5 = -13 := by
  sorry

end f_at_5_eq_neg_13_l634_63400
