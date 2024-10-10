import Mathlib

namespace log_sum_equals_two_l1857_185730

theorem log_sum_equals_two (a : ℝ) (h : 1 + a^3 = 9) : 
  Real.log a / Real.log (1/4) + Real.log 8 / Real.log a = 2 := by
  sorry

end log_sum_equals_two_l1857_185730


namespace president_vice_president_selection_l1857_185760

/-- The number of candidates for class president and vice president -/
def num_candidates : ℕ := 4

/-- The number of positions to be filled (president and vice president) -/
def num_positions : ℕ := 2

/-- Theorem: The number of ways to choose a president and a vice president from 4 candidates is 12 -/
theorem president_vice_president_selection :
  (num_candidates * (num_candidates - 1)) = 12 := by
  sorry

#check president_vice_president_selection

end president_vice_president_selection_l1857_185760


namespace market_spending_l1857_185718

theorem market_spending (total_amount mildred_spent candice_spent : ℕ) 
  (h1 : total_amount = 100)
  (h2 : mildred_spent = 25)
  (h3 : candice_spent = 35) :
  total_amount - (mildred_spent + candice_spent) = 40 := by
  sorry

end market_spending_l1857_185718


namespace flower_bed_area_is_35_l1857_185715

/-- The area of a rectangular flower bed -/
def flower_bed_area (width : ℝ) (length : ℝ) : ℝ := width * length

theorem flower_bed_area_is_35 :
  flower_bed_area 5 7 = 35 := by
  sorry

end flower_bed_area_is_35_l1857_185715


namespace total_legs_is_71_l1857_185798

/-- Represents the total number of legs in a room with various furniture items -/
def total_legs : ℝ :=
  -- 4 tables with 4 legs each
  4 * 4 +
  -- 1 sofa with 4 legs
  1 * 4 +
  -- 2 chairs with 4 legs each
  2 * 4 +
  -- 3 tables with 3 legs each
  3 * 3 +
  -- 1 table with a single leg
  1 * 1 +
  -- 1 rocking chair with 2 legs
  1 * 2 +
  -- 1 bench with 6 legs
  1 * 6 +
  -- 2 stools with 3 legs each
  2 * 3 +
  -- 2 wardrobes, one with 4 legs and one with 3 legs
  (1 * 4 + 1 * 3) +
  -- 1 three-legged ecko
  1 * 3 +
  -- 1 antique table with 3 remaining legs
  1 * 3 +
  -- 1 damaged 4-legged table with only 3.5 legs remaining
  1 * 3.5 +
  -- 1 stool that lost half a leg
  1 * 2.5

/-- Theorem stating that the total number of legs in the room is 71 -/
theorem total_legs_is_71 : total_legs = 71 := by
  sorry

end total_legs_is_71_l1857_185798


namespace internal_tangent_segment_bounded_l1857_185780

/-- Two equal circles with a common internal tangent and external tangents -/
structure TwoCirclesWithTangents where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of both circles (they are equal) -/
  radius : ℝ
  /-- Point where the common internal tangent intersects the external tangent of the first circle -/
  P : ℝ × ℝ
  /-- Point where the common internal tangent intersects the external tangent of the second circle -/
  Q : ℝ × ℝ
  /-- The circles are equal -/
  equal_circles : radius > 0
  /-- P is on the external tangent of the first circle -/
  P_on_external_tangent1 : (P.1 - center1.1) * (P.1 - center1.1) + (P.2 - center1.2) * (P.2 - center1.2) = radius * radius
  /-- Q is on the external tangent of the second circle -/
  Q_on_external_tangent2 : (Q.1 - center2.1) * (Q.1 - center2.1) + (Q.2 - center2.2) * (Q.2 - center2.2) = radius * radius
  /-- PQ is perpendicular to the radii at P and Q -/
  tangent_perpendicular : 
    (P.1 - center1.1) * (Q.1 - P.1) + (P.2 - center1.2) * (Q.2 - P.2) = 0 ∧
    (Q.1 - center2.1) * (P.1 - Q.1) + (Q.2 - center2.2) * (P.2 - Q.2) = 0

/-- The theorem statement -/
theorem internal_tangent_segment_bounded (c : TwoCirclesWithTangents) :
  (c.P.1 - c.Q.1) * (c.P.1 - c.Q.1) + (c.P.2 - c.Q.2) * (c.P.2 - c.Q.2) ≤
  (c.center1.1 - c.center2.1) * (c.center1.1 - c.center2.1) + (c.center1.2 - c.center2.2) * (c.center1.2 - c.center2.2) :=
sorry

end internal_tangent_segment_bounded_l1857_185780


namespace log_inequality_l1857_185786

theorem log_inequality (x : Real) (h : x > 0) : Real.log (1 + x^2) < x^2 := by
  sorry

end log_inequality_l1857_185786


namespace problem_1_problem_2_problem_3_l1857_185723

-- Problem 1
theorem problem_1 : (π - 3.14)^0 + Real.sqrt 16 + |1 - Real.sqrt 2| = 4 + Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 : ∃ (x y : ℝ), x - y = 2 ∧ 2*x + 3*y = 9 ∧ x = 3 ∧ y = 1 := by sorry

-- Problem 3
theorem problem_3 : ∃ (x y : ℝ), 5*(x-1) + 2*y = 4*(1-y) + 3 ∧ x/3 + y/2 = 1 ∧ x = 0 ∧ y = 2 := by sorry

end problem_1_problem_2_problem_3_l1857_185723


namespace rhombus_parallel_sides_distance_l1857_185705

/-- The distance between parallel sides of a rhombus given its diagonals -/
theorem rhombus_parallel_sides_distance (AC BD : ℝ) (h1 : AC = 3) (h2 : BD = 4) :
  let area := (1 / 2) * AC * BD
  let side := Real.sqrt ((AC / 2)^2 + (BD / 2)^2)
  area / side = 12 / 5 := by sorry

end rhombus_parallel_sides_distance_l1857_185705


namespace adjacent_to_five_sum_seven_l1857_185741

/-- Represents the five corners of a pentagon -/
inductive Corner
  | a | b | c | d | e

/-- A configuration of numbers in the pentagon corners -/
def Configuration := Corner → Fin 5

/-- Two corners are adjacent if they share an edge in the pentagon -/
def adjacent (x y : Corner) : Prop :=
  match x, y with
  | Corner.a, Corner.b | Corner.b, Corner.a => true
  | Corner.b, Corner.c | Corner.c, Corner.b => true
  | Corner.c, Corner.d | Corner.d, Corner.c => true
  | Corner.d, Corner.e | Corner.e, Corner.d => true
  | Corner.e, Corner.a | Corner.a, Corner.e => true
  | _, _ => false

/-- A valid configuration satisfies the adjacency condition -/
def valid_configuration (config : Configuration) : Prop :=
  ∀ x y : Corner, adjacent x y → |config x - config y| > 1

/-- The main theorem -/
theorem adjacent_to_five_sum_seven (config : Configuration) 
  (h_valid : valid_configuration config) 
  (h_five : ∃ x : Corner, config x = 5) :
  ∃ y z : Corner, 
    adjacent x y ∧ adjacent x z ∧ y ≠ z ∧ 
    config y + config z = 7 ∧ config x = 5 := by
  sorry

end adjacent_to_five_sum_seven_l1857_185741


namespace average_bottle_price_l1857_185783

def large_bottles : ℕ := 1325
def small_bottles : ℕ := 750
def large_bottle_price : ℚ := 189/100
def small_bottle_price : ℚ := 138/100

theorem average_bottle_price :
  let total_cost : ℚ := large_bottles * large_bottle_price + small_bottles * small_bottle_price
  let total_bottles : ℕ := large_bottles + small_bottles
  let average_price : ℚ := total_cost / total_bottles
  ∃ ε > 0, |average_price - 17/10| < ε ∧ ε < 1/100 :=
by
  sorry

end average_bottle_price_l1857_185783


namespace all_circles_pass_through_point_l1857_185789

-- Define the parabola
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  (P.2 + 2)^2 = 4 * (P.1 - 1)

-- Define a circle with center P tangent to y-axis
def circle_tangent_y_axis (P : ℝ × ℝ) (r : ℝ) : Prop :=
  r = P.1

-- Theorem statement
theorem all_circles_pass_through_point :
  ∀ (P : ℝ × ℝ) (r : ℝ),
    is_on_parabola P →
    circle_tangent_y_axis P r →
    (P.1 - 2)^2 + (P.2 + 2)^2 = r^2 :=
by sorry

end all_circles_pass_through_point_l1857_185789


namespace decimal_to_fraction_l1857_185784

theorem decimal_to_fraction (x : ℚ) : x = 224/100 → x = 56/25 := by
  sorry

end decimal_to_fraction_l1857_185784


namespace chord_length_l1857_185766

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end chord_length_l1857_185766


namespace prob_no_rain_five_days_l1857_185707

/-- The probability of no rain for n consecutive days, given the probability of rain on each day is p -/
def prob_no_rain (n : ℕ) (p : ℚ) : ℚ := (1 - p) ^ n

theorem prob_no_rain_five_days :
  prob_no_rain 5 (1/2) = 1/32 := by sorry

end prob_no_rain_five_days_l1857_185707


namespace student_selection_l1857_185750

theorem student_selection (total : ℕ) (singers : ℕ) (dancers : ℕ) (both : ℕ) :
  total = 6 ∧ singers = 3 ∧ dancers = 2 ∧ both = 1 →
  Nat.choose singers 2 * dancers = 6 :=
by sorry

end student_selection_l1857_185750


namespace rectangle_width_l1857_185793

theorem rectangle_width (w : ℝ) (h1 : w > 0) : 
  (2 * w * w = 1) → w = Real.sqrt 2 / 2 := by
  sorry

end rectangle_width_l1857_185793


namespace quadratic_equation_root_relation_l1857_185709

theorem quadratic_equation_root_relation (m : ℝ) (hm : m ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁^2 - x₁ + m = 0 ∧ x₂^2 - x₂ + 3*m = 0 ∧ x₂ = 2*x₁) →
  m = -2 := by
sorry

end quadratic_equation_root_relation_l1857_185709


namespace farm_tax_collection_l1857_185746

theorem farm_tax_collection (william_tax : ℝ) (william_land_percentage : ℝ) 
  (h1 : william_tax = 480)
  (h2 : william_land_percentage = 0.25) : 
  william_tax / william_land_percentage = 1920 := by
  sorry

end farm_tax_collection_l1857_185746


namespace librarian_took_two_books_l1857_185737

/-- The number of books the librarian took -/
def librarian_took (total_books : ℕ) (shelves_needed : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books - (shelves_needed * books_per_shelf)

/-- Theorem stating that the librarian took 2 books -/
theorem librarian_took_two_books :
  librarian_took 14 4 3 = 2 := by
  sorry

end librarian_took_two_books_l1857_185737


namespace shaded_design_area_l1857_185732

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a triangle in the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The shaded design in the 7x7 grid -/
def shaded_design : List GridTriangle := sorry

/-- Calculates the area of a single triangle in the grid -/
def triangle_area (t : GridTriangle) : Rat := sorry

/-- Calculates the total area of the shaded design -/
def total_area (design : List GridTriangle) : Rat :=
  design.map triangle_area |>.sum

/-- The theorem stating that the area of the shaded design is 1.5 -/
theorem shaded_design_area :
  total_area shaded_design = 3/2 := by sorry

end shaded_design_area_l1857_185732


namespace expression_evaluation_l1857_185728

theorem expression_evaluation : 4 * (5^2 + 5^2 + 5^2 + 5^2) = 400 := by
  sorry

end expression_evaluation_l1857_185728


namespace jim_gave_away_195_cards_l1857_185754

/-- The number of cards Jim gives away -/
def cards_given_away (initial_cards : ℕ) (cards_per_set : ℕ) (sets_to_brother : ℕ) (sets_to_sister : ℕ) (sets_to_friend : ℕ) : ℕ :=
  (sets_to_brother + sets_to_sister + sets_to_friend) * cards_per_set

/-- Proof that Jim gave away 195 cards -/
theorem jim_gave_away_195_cards :
  cards_given_away 365 13 8 5 2 = 195 := by
  sorry

end jim_gave_away_195_cards_l1857_185754


namespace students_walking_home_l1857_185722

theorem students_walking_home (bus_fraction automobile_fraction bicycle_fraction : ℚ)
  (h1 : bus_fraction = 1/2)
  (h2 : automobile_fraction = 1/4)
  (h3 : bicycle_fraction = 1/10) :
  1 - (bus_fraction + automobile_fraction + bicycle_fraction) = 3/20 := by
sorry

end students_walking_home_l1857_185722


namespace simplify_expression_l1857_185768

theorem simplify_expression (a : ℝ) : 3*a^2 - 2*a + 1 + (3*a - a^2 + 2) = 2*a^2 + a + 3 := by
  sorry

end simplify_expression_l1857_185768


namespace tobys_friends_l1857_185717

theorem tobys_friends (total_friends : ℕ) (boy_friends : ℕ) (girl_friends : ℕ) : 
  (boy_friends : ℚ) / (total_friends : ℚ) = 55 / 100 →
  boy_friends = 33 →
  girl_friends = 27 :=
by sorry

end tobys_friends_l1857_185717


namespace sufficient_not_necessary_condition_l1857_185702

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (a = Real.sqrt 3 ∧ b = 1 → Complex.abs ((1 + Complex.I * b) / (a + Complex.I)) = Real.sqrt 2 / 2) ∧
  (∃ (x y : ℝ), (x ≠ Real.sqrt 3 ∨ y ≠ 1) ∧ Complex.abs ((1 + Complex.I * y) / (x + Complex.I)) = Real.sqrt 2 / 2) :=
by sorry

end sufficient_not_necessary_condition_l1857_185702


namespace bridget_sarah_money_l1857_185765

/-- The amount of money Bridget and Sarah have together in dollars -/
def total_money (sarah_cents bridget_cents : ℕ) : ℚ :=
  (sarah_cents + bridget_cents : ℚ) / 100

theorem bridget_sarah_money :
  ∀ (sarah_cents : ℕ),
    sarah_cents = 125 →
    ∀ (bridget_cents : ℕ),
      bridget_cents = sarah_cents + 50 →
      total_money sarah_cents bridget_cents = 3 := by
sorry

end bridget_sarah_money_l1857_185765


namespace smallest_second_term_arithmetic_sequence_l1857_185799

theorem smallest_second_term_arithmetic_sequence :
  ∀ (a d : ℕ),
  a > 0 →
  d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 95 →
  ∀ (b e : ℕ),
  b > 0 →
  e > 0 →
  b + (b + e) + (b + 2*e) + (b + 3*e) + (b + 4*e) = 95 →
  (a + d) ≤ (b + e) →
  a + d = 10 :=
by sorry

end smallest_second_term_arithmetic_sequence_l1857_185799


namespace unique_starting_number_l1857_185720

def operation (n : ℕ) : ℕ :=
  if n % 3 = 0 then n / 3 else n + 1

def iterate_operation (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => iterate_operation (operation n) k

theorem unique_starting_number : 
  ∃! n : ℕ, iterate_operation n 5 = 1 :=
sorry

end unique_starting_number_l1857_185720


namespace max_triples_count_l1857_185711

def N (n : ℕ) : ℕ := sorry

theorem max_triples_count (n : ℕ) (h : n ≥ 2) :
  N n = ⌊(2 * n : ℚ) / 3 + 1⌋ :=
by sorry

end max_triples_count_l1857_185711


namespace max_min_difference_c_l1857_185731

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 20) : 
  ∃ (c_max c_min : ℝ), 
    (∀ x : ℝ, (∃ y z : ℝ, y + z + x = 3 ∧ y^2 + z^2 + x^2 = 20) → x ≤ c_max ∧ x ≥ c_min) ∧ 
    c_max - c_min = 2 * Real.sqrt 34 :=
sorry

end max_min_difference_c_l1857_185731


namespace point_C_coordinates_l1857_185758

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line on which point C lies
def line_C (x : ℝ) : ℝ := 3 * x + 3

-- Define the area of the triangle
def triangle_area : ℝ := 10

-- Theorem statement
theorem point_C_coordinates :
  ∃ (C : ℝ × ℝ), 
    (C.2 = line_C C.1) ∧ 
    (abs ((C.1 - A.1) * (B.2 - A.2) - (B.1 - A.1) * (C.2 - A.2)) / 2 = triangle_area) ∧
    ((C = (-1, 0)) ∨ (C = (5/3, 8))) :=
sorry

end point_C_coordinates_l1857_185758


namespace valid_arrangement_exists_l1857_185745

/-- A binary sequence of length 6 -/
def BinarySeq := Fin 6 → Bool

/-- The set of all possible 6-digit binary sequences -/
def AllBinarySeqs : Set BinarySeq :=
  {seq | seq ∈ Set.univ}

/-- Two binary sequences differ by exactly one digit -/
def differByOne (seq1 seq2 : BinarySeq) : Prop :=
  ∃! i : Fin 6, seq1 i ≠ seq2 i

/-- A valid arrangement of binary sequences in an 8x8 grid -/
def ValidArrangement (arrangement : Fin 8 → Fin 8 → BinarySeq) : Prop :=
  (∀ i j, arrangement i j ∈ AllBinarySeqs) ∧
  (∀ i j, i + 1 < 8 → differByOne (arrangement i j) (arrangement (i + 1) j)) ∧
  (∀ i j, j + 1 < 8 → differByOne (arrangement i j) (arrangement i (j + 1))) ∧
  (∀ i j k l, (i ≠ k ∨ j ≠ l) → arrangement i j ≠ arrangement k l)

/-- The main theorem: a valid arrangement exists -/
theorem valid_arrangement_exists : ∃ arrangement, ValidArrangement arrangement := by
  sorry


end valid_arrangement_exists_l1857_185745


namespace diagonal_path_cubes_3_4_5_l1857_185706

/-- The number of cubes a diagonal path crosses in a cuboid -/
def cubes_crossed (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd a c + Nat.gcd a (Nat.gcd b c)

/-- Theorem: In a 3 × 4 × 5 cuboid, a diagonal path from one corner to the opposite corner
    that doesn't intersect the edges of any small cube inside the cuboid passes through 10 small cubes -/
theorem diagonal_path_cubes_3_4_5 :
  cubes_crossed 3 4 5 = 10 := by
  sorry

end diagonal_path_cubes_3_4_5_l1857_185706


namespace max_value_x2_l1857_185748

theorem max_value_x2 (x₁ x₂ x₃ : ℝ) 
  (h : x₁^2 + x₂^2 + x₃^2 + x₁*x₂ + x₂*x₃ = 2) : 
  |x₂| ≤ 2 := by
  sorry

end max_value_x2_l1857_185748


namespace intersection_line_not_through_point_l1857_185785

-- Define the circles
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def circle_M (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = a^2 + b^2

-- Define the condition for M being on circle C
def M_on_C (a b : ℝ) : Prop := circle_C a b

-- Define the line equation passing through intersection points
def line_AB (a b m n : ℝ) : Prop := 2*m*a + 2*n*b - (2*m + 3) = 0

-- Theorem statement
theorem intersection_line_not_through_point :
  ∀ (a b : ℝ), M_on_C a b →
  ¬(line_AB a b (1/2) (1/2)) :=
sorry

end intersection_line_not_through_point_l1857_185785


namespace joel_age_when_dad_twice_as_old_l1857_185729

/-- Joel's current age -/
def joel_current_age : ℕ := 8

/-- Joel's dad's current age -/
def dad_current_age : ℕ := 37

/-- The number of years until Joel's dad is twice Joel's age -/
def years_until_double : ℕ := dad_current_age - 2 * joel_current_age

/-- Joel's age when his dad is twice as old as him -/
def joel_future_age : ℕ := joel_current_age + years_until_double

theorem joel_age_when_dad_twice_as_old :
  joel_future_age = 29 :=
sorry

end joel_age_when_dad_twice_as_old_l1857_185729


namespace orange_ribbons_l1857_185744

theorem orange_ribbons (total : ℕ) (yellow purple orange black : ℕ) : 
  yellow = total / 4 →
  purple = total / 3 →
  orange = total / 12 →
  black = 40 →
  yellow + purple + orange + black = total →
  orange = 10 := by
sorry

end orange_ribbons_l1857_185744


namespace min_value_theorem_l1857_185733

theorem min_value_theorem (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_prod : x * y * z = 1) :
  x^2 + 8*x*y + 9*y^2 + 8*y*z + 2*z^2 ≥ 18 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧ 
    a^2 + 8*a*b + 9*b^2 + 8*b*c + 2*c^2 = 18 := by
  sorry

end min_value_theorem_l1857_185733


namespace chair_cost_l1857_185714

theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  ∃ (chair_cost : ℕ), 
    chair_cost * num_chairs = total_spent - table_cost ∧
    chair_cost = 11 :=
by
  sorry

end chair_cost_l1857_185714


namespace quadratic_inequality_properties_l1857_185712

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h_solution_set : ∀ x, f a b c x < 0 ↔ -1 < x ∧ x < 2) :
  (∀ x, b * x + c > 0 ↔ x < -2) ∧
  (4 * a - 2 * b + c > 0) :=
sorry

end quadratic_inequality_properties_l1857_185712


namespace knife_percentage_after_trade_l1857_185767

/-- Represents a silverware set with knives, forks, and spoons -/
structure Silverware where
  knives : ℕ
  forks : ℕ
  spoons : ℕ

/-- Represents a trade of silverware -/
structure Trade where
  knivesReceived : ℕ
  spoonsGiven : ℕ

def initialSet : Silverware :=
  { knives := 6
  , forks := 12
  , spoons := 6 * 3 }

def trade : Trade :=
  { knivesReceived := 10
  , spoonsGiven := 6 }

def finalSet (initial : Silverware) (t : Trade) : Silverware :=
  { knives := initial.knives + t.knivesReceived
  , forks := initial.forks
  , spoons := initial.spoons - t.spoonsGiven }

def totalPieces (s : Silverware) : ℕ :=
  s.knives + s.forks + s.spoons

def knifePercentage (s : Silverware) : ℚ :=
  s.knives / totalPieces s

theorem knife_percentage_after_trade :
  knifePercentage (finalSet initialSet trade) = 2/5 := by
  sorry

end knife_percentage_after_trade_l1857_185767


namespace mirella_orange_books_l1857_185703

/-- The number of pages in each purple book -/
def purple_pages : ℕ := 230

/-- The number of pages in each orange book -/
def orange_pages : ℕ := 510

/-- The number of purple books Mirella read -/
def purple_books_read : ℕ := 5

/-- The difference between orange and purple pages Mirella read -/
def page_difference : ℕ := 890

/-- The number of orange books Mirella read -/
def orange_books_read : ℕ := 4

theorem mirella_orange_books :
  orange_books_read * orange_pages = 
  purple_books_read * purple_pages + page_difference := by
  sorry

end mirella_orange_books_l1857_185703


namespace error_percentage_division_vs_multiplication_l1857_185762

theorem error_percentage_division_vs_multiplication :
  ∀ x : ℝ, x ≠ 0 →
  (((5 * x - x / 5) / (5 * x)) * 100 : ℝ) = 96 := by
  sorry

end error_percentage_division_vs_multiplication_l1857_185762


namespace sequence_14th_term_is_9_l1857_185775

theorem sequence_14th_term_is_9 :
  let a : ℕ → ℝ := fun n => Real.sqrt (3 * (2 * n - 1))
  a 14 = 9 := by sorry

end sequence_14th_term_is_9_l1857_185775


namespace sum_of_five_consecutive_even_integers_l1857_185753

theorem sum_of_five_consecutive_even_integers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20 := by
  sorry

end sum_of_five_consecutive_even_integers_l1857_185753


namespace eight_points_chords_l1857_185700

/-- The number of chords that can be drawn by connecting two points out of n points on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords that can be drawn by connecting two points out of eight points on the circumference of a circle is equal to 28 -/
theorem eight_points_chords : num_chords 8 = 28 := by
  sorry

end eight_points_chords_l1857_185700


namespace symmetric_point_coordinates_l1857_185792

/-- A point in a 3D rectangular coordinate system -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The symmetric point about the z-axis -/
def symmetricAboutZAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

/-- Theorem: The symmetric point about the z-axis has coordinates (-a, -b, c) -/
theorem symmetric_point_coordinates (p : Point3D) :
  symmetricAboutZAxis p = { x := -p.x, y := -p.y, z := p.z } := by
  sorry

end symmetric_point_coordinates_l1857_185792


namespace abs_eq_neg_self_implies_nonpositive_l1857_185739

theorem abs_eq_neg_self_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end abs_eq_neg_self_implies_nonpositive_l1857_185739


namespace unique_solution_exists_l1857_185704

theorem unique_solution_exists : ∃! (a b : ℝ), 2 * a + b = 7 ∧ a - b = 2 := by sorry

end unique_solution_exists_l1857_185704


namespace sara_golf_balls_l1857_185708

/-- The number of golf balls in a dozen -/
def dozen : ℕ := 12

/-- The total number of golf balls Sara has -/
def total_golf_balls : ℕ := 108

/-- The number of dozens of golf balls Sara has -/
def dozens_of_golf_balls : ℕ := total_golf_balls / dozen

theorem sara_golf_balls : dozens_of_golf_balls = 9 := by
  sorry

end sara_golf_balls_l1857_185708


namespace triangle_perimeter_l1857_185734

/-- Given a triangle with inradius 2.5 cm and area 40 cm², its perimeter is 32 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 40 → A = r * (p / 2) → p = 32 := by
  sorry

end triangle_perimeter_l1857_185734


namespace hidden_faces_sum_l1857_185769

def standard_die := List.range 6 |>.map (· + 1)

def visible_faces : List Nat := [1, 2, 3, 4, 4, 5, 6, 6]

def total_faces : Nat := 3 * 6

theorem hidden_faces_sum :
  (3 * standard_die.sum) - visible_faces.sum = 32 := by
  sorry

end hidden_faces_sum_l1857_185769


namespace quadratic_rewrite_l1857_185795

theorem quadratic_rewrite (b : ℝ) (m : ℝ) : 
  (∀ x, x^2 + b*x + 49 = (x + m)^2 + 9) ∧ (b > 0) → b = 4 * Real.sqrt 10 := by
  sorry

end quadratic_rewrite_l1857_185795


namespace contractor_hourly_rate_l1857_185788

/-- Contractor's hourly rate calculation -/
theorem contractor_hourly_rate 
  (total_cost : ℝ) 
  (permit_cost : ℝ) 
  (contractor_hours : ℝ) 
  (inspector_rate_ratio : ℝ) :
  total_cost = 2950 →
  permit_cost = 250 →
  contractor_hours = 15 →
  inspector_rate_ratio = 0.2 →
  ∃ (contractor_rate : ℝ),
    contractor_rate = 150 ∧
    total_cost = permit_cost + contractor_hours * contractor_rate * (1 + inspector_rate_ratio) :=
by
  sorry

end contractor_hourly_rate_l1857_185788


namespace negation_of_log_inequality_l1857_185770

theorem negation_of_log_inequality (p : Prop) : 
  (p ↔ ∀ x : ℝ, Real.log x > 1) → 
  (¬p ↔ ∃ x₀ : ℝ, Real.log x₀ ≤ 1) := by
  sorry

end negation_of_log_inequality_l1857_185770


namespace response_rate_increase_l1857_185779

/-- Calculate the percentage increase in response rate between two surveys -/
theorem response_rate_increase (customers1 customers2 respondents1 respondents2 : ℕ) :
  customers1 = 80 →
  customers2 = 63 →
  respondents1 = 7 →
  respondents2 = 9 →
  let rate1 := (respondents1 : ℝ) / customers1
  let rate2 := (respondents2 : ℝ) / customers2
  let increase := (rate2 - rate1) / rate1 * 100
  ∃ ε > 0, |increase - 63.24| < ε :=
by sorry

end response_rate_increase_l1857_185779


namespace point_in_fourth_quadrant_l1857_185778

/-- A point in the fourth quadrant with given conditions has coordinates (7, -3) -/
theorem point_in_fourth_quadrant (x y : ℝ) (h1 : x > 0) (h2 : y < 0) 
  (h3 : |x| = 7) (h4 : y^2 = 9) : (x, y) = (7, -3) := by
  sorry

end point_in_fourth_quadrant_l1857_185778


namespace dropped_student_score_l1857_185782

theorem dropped_student_score 
  (initial_count : ℕ) 
  (remaining_count : ℕ) 
  (initial_average : ℚ) 
  (new_average : ℚ) 
  (h1 : initial_count = 16)
  (h2 : remaining_count = 15)
  (h3 : initial_average = 62.5)
  (h4 : new_average = 62)
  : (initial_count : ℚ) * initial_average - (remaining_count : ℚ) * new_average = 70 :=
by sorry

end dropped_student_score_l1857_185782


namespace six_million_three_hundred_ninety_thousand_scientific_notation_l1857_185797

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem six_million_three_hundred_ninety_thousand_scientific_notation :
  toScientificNotation 6390000 = ScientificNotation.mk 6.39 6 (by sorry) :=
sorry

end six_million_three_hundred_ninety_thousand_scientific_notation_l1857_185797


namespace sin_alpha_equals_half_l1857_185710

theorem sin_alpha_equals_half (α : Real) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin α = Real.cos (2 * α)) : 
  Real.sin α = 1 / 2 := by
sorry

end sin_alpha_equals_half_l1857_185710


namespace go_complexity_vs_universe_atoms_l1857_185738

/-- Approximation of the upper limit of state space complexity of Go -/
def M : ℝ := 3^361

/-- Approximation of the total number of atoms in the observable universe -/
def N : ℝ := 10^80

/-- Approximation of log base 10 of 3 -/
def log10_3 : ℝ := 0.48

/-- The closest value to M/N among the given options -/
def closest_value : ℝ := 10^93

theorem go_complexity_vs_universe_atoms :
  abs (M / N - closest_value) = 
    min (abs (M / N - 10^33)) 
        (min (abs (M / N - 10^53)) 
             (min (abs (M / N - 10^73)) 
                  (abs (M / N - 10^93)))) := by
  sorry

end go_complexity_vs_universe_atoms_l1857_185738


namespace pear_sales_l1857_185790

theorem pear_sales (morning_sales afternoon_sales total_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  afternoon_sales = 260 →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 390 :=
by sorry

end pear_sales_l1857_185790


namespace carpet_length_l1857_185781

/-- Given a rectangular carpet with width 4 feet covering 75% of a 48 square feet room,
    prove that the length of the carpet is 9 feet. -/
theorem carpet_length (room_area : ℝ) (carpet_width : ℝ) (coverage_percent : ℝ) :
  room_area = 48 →
  carpet_width = 4 →
  coverage_percent = 0.75 →
  (room_area * coverage_percent) / carpet_width = 9 :=
by sorry

end carpet_length_l1857_185781


namespace lcm_of_54_and_16_l1857_185736

theorem lcm_of_54_and_16 : Nat.lcm 54 16 = 48 :=
by
  have h1 : Nat.gcd 54 16 = 18 := by sorry
  sorry

end lcm_of_54_and_16_l1857_185736


namespace complement_of_A_in_U_l1857_185713

def U : Set ℕ := {2, 3, 4}
def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : (U \ A) = {4} := by sorry

end complement_of_A_in_U_l1857_185713


namespace stadium_fee_difference_l1857_185749

theorem stadium_fee_difference (capacity : ℕ) (fee : ℕ) (h1 : capacity = 2000) (h2 : fee = 20) :
  capacity * fee - (3 * capacity / 4) * fee = 10000 := by
  sorry

end stadium_fee_difference_l1857_185749


namespace circle_line_intersection_l1857_185764

/-- Given a circle and a line, prove the value of m when the chord length is 4 -/
theorem circle_line_intersection (m : ℝ) : 
  (∃ x y : ℝ, (x + 1)^2 + (y - 1)^2 = 2 - m ∧ x + y + 2 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + 1)^2 + (y₁ - 1)^2 = 2 - m ∧
    x₁ + y₁ + 2 = 0 ∧
    (x₂ + 1)^2 + (y₂ - 1)^2 = 2 - m ∧
    x₂ + y₂ + 2 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) →
  m = -4 :=
by sorry

end circle_line_intersection_l1857_185764


namespace inlet_pipe_rate_l1857_185727

/-- Given a tank with the following properties:
  * Capacity of 6048 liters
  * Empties in 7 hours due to a leak
  * Empties in 12 hours when both the leak and an inlet pipe are open
  Prove that the rate at which the inlet pipe fills water is 360 liters per hour -/
theorem inlet_pipe_rate (tank_capacity : ℝ) (leak_empty_time : ℝ) (both_empty_time : ℝ)
  (h1 : tank_capacity = 6048)
  (h2 : leak_empty_time = 7)
  (h3 : both_empty_time = 12) :
  let leak_rate := tank_capacity / leak_empty_time
  let net_empty_rate := tank_capacity / both_empty_time
  leak_rate - (leak_rate - net_empty_rate) = 360 := by
sorry


end inlet_pipe_rate_l1857_185727


namespace quinary_1234_eq_194_l1857_185773

/-- Converts a quinary (base-5) number to decimal. -/
def quinary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

/-- The quinary representation of 1234₍₅₎ -/
def quinary_1234 : List Nat := [4, 3, 2, 1]

theorem quinary_1234_eq_194 : quinary_to_decimal quinary_1234 = 194 := by
  sorry

end quinary_1234_eq_194_l1857_185773


namespace magnitude_relationship_l1857_185721

theorem magnitude_relationship :
  let α : ℝ := Real.cos 4
  let b : ℝ := Real.cos (4 * π / 5)
  let c : ℝ := Real.sin (7 * π / 6)
  b < α ∧ α < c := by
  sorry

end magnitude_relationship_l1857_185721


namespace min_value_of_f_l1857_185761

def f (x a : ℝ) : ℝ := |x - a| + |x - 15| + |x - (a + 15)|

theorem min_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 15) :
  ∃ Q : ℝ, Q = 15 ∧ ∀ x : ℝ, a ≤ x → x ≤ 15 → f x a ≥ Q :=
sorry

end min_value_of_f_l1857_185761


namespace repair_center_solution_l1857_185701

/-- Represents a bonus distribution plan -/
structure BonusPlan where
  techBonus : ℕ
  assistBonus : ℕ

/-- Represents the repair center staff and bonus distribution -/
structure RepairCenter where
  techCount : ℕ
  assistCount : ℕ
  totalBonus : ℕ
  bonusPlans : List BonusPlan

/-- The conditions of the repair center problem -/
def repairCenterConditions (rc : RepairCenter) : Prop :=
  rc.techCount + rc.assistCount = 15 ∧
  rc.techCount = 2 * rc.assistCount ∧
  rc.totalBonus = 20000 ∧
  ∀ plan ∈ rc.bonusPlans,
    plan.techBonus ≥ plan.assistBonus ∧
    plan.assistBonus ≥ 800 ∧
    plan.techBonus % 100 = 0 ∧
    plan.assistBonus % 100 = 0 ∧
    rc.techCount * plan.techBonus + rc.assistCount * plan.assistBonus = rc.totalBonus

/-- The theorem stating the solution to the repair center problem -/
theorem repair_center_solution :
  ∃ (rc : RepairCenter),
    repairCenterConditions rc ∧
    rc.techCount = 10 ∧
    rc.assistCount = 5 ∧
    rc.bonusPlans = [
      { techBonus := 1600, assistBonus := 800 },
      { techBonus := 1500, assistBonus := 1000 },
      { techBonus := 1400, assistBonus := 1200 }
    ] :=
  sorry

end repair_center_solution_l1857_185701


namespace grouping_factoring_1_grouping_factoring_2_l1857_185796

-- Expression 1
theorem grouping_factoring_1 (a b c : ℝ) :
  a^2 + 2*a*b + b^2 + a*c + b*c = (a + b) * (a + b + c) := by sorry

-- Expression 2
theorem grouping_factoring_2 (a x y : ℝ) :
  4*a^2 - x^2 + 4*x*y - 4*y^2 = (2*a + x - 2*y) * (2*a - x + 2*y) := by sorry

end grouping_factoring_1_grouping_factoring_2_l1857_185796


namespace integer_solution_condition_non_negative_condition_l1857_185776

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^4 - 4*x^3 + (3 + m)*x^2 - 12*x + 12

/-- Theorem for the first part of the problem -/
theorem integer_solution_condition (m : ℝ) :
  (∃ x : ℤ, f m x - f m (1 - x) + 4*x^3 = 0) ↔ (m = 8 ∨ m = 12) := by sorry

/-- Theorem for the second part of the problem -/
theorem non_negative_condition (m : ℝ) :
  (∀ x : ℝ, f m x ≥ 0) ↔ m ≥ 4 := by sorry

end integer_solution_condition_non_negative_condition_l1857_185776


namespace cricket_game_overs_l1857_185747

/-- Proves that the number of overs played in the first part of a cricket game is 10,
    given the specified conditions. -/
theorem cricket_game_overs (total_target : ℝ) (first_run_rate : ℝ) 
  (remaining_overs : ℝ) (remaining_run_rate : ℝ) 
  (h1 : total_target = 282)
  (h2 : first_run_rate = 3.2)
  (h3 : remaining_overs = 40)
  (h4 : remaining_run_rate = 6.25) :
  (total_target - remaining_overs * remaining_run_rate) / first_run_rate = 10 := by
sorry

end cricket_game_overs_l1857_185747


namespace one_and_quarter_of_what_is_forty_l1857_185771

theorem one_and_quarter_of_what_is_forty : ∃ x : ℝ, 1.25 * x = 40 ∧ x = 32 := by
  sorry

end one_and_quarter_of_what_is_forty_l1857_185771


namespace equation_solution_l1857_185740

theorem equation_solution :
  ∃ x : ℚ, (3 * x - 17) / 4 = (x + 9) / 6 ∧ x = 69 / 7 := by
  sorry

end equation_solution_l1857_185740


namespace expected_checks_on_4x4_board_l1857_185724

/-- Represents a 4x4 chessboard -/
def Board := Fin 4 × Fin 4

/-- Calculates the number of ways a knight can check a king on a 4x4 board -/
def knight_check_positions (board : Board) : ℕ :=
  match board with
  | (0, 0) | (0, 3) | (3, 0) | (3, 3) => 2  -- corners
  | (0, 1) | (0, 2) | (1, 0) | (1, 3) | (2, 0) | (2, 3) | (3, 1) | (3, 2) => 3  -- edges
  | _ => 4  -- central squares

/-- The total number of possible knight-king pairs -/
def total_pairs : ℕ := 3 * 3

/-- The total number of ways to place a knight and a king on distinct squares -/
def total_placements : ℕ := 16 * 15

/-- The expected number of checks for a single knight-king pair -/
def expected_checks_per_pair : ℚ := 1 / 5

theorem expected_checks_on_4x4_board :
  (total_pairs : ℚ) * expected_checks_per_pair = 9 / 5 := by sorry

#check expected_checks_on_4x4_board

end expected_checks_on_4x4_board_l1857_185724


namespace pattern_two_odd_one_even_l1857_185772

/-- A box containing 100 balls numbered from 1 to 100 -/
def Box := Finset (Fin 100)

/-- The set of odd-numbered balls in the box -/
def OddBalls (box : Box) : Finset (Fin 100) :=
  box.filter (fun n => n % 2 = 1)

/-- The set of even-numbered balls in the box -/
def EvenBalls (box : Box) : Finset (Fin 100) :=
  box.filter (fun n => n % 2 = 0)

/-- A selection pattern of 3 balls -/
structure SelectionPattern :=
  (first second third : Bool)

/-- The probability of selecting an odd-numbered ball first -/
def ProbFirstOdd (pattern : SelectionPattern) : ℚ :=
  if pattern.first then 2/3 else 1/3

theorem pattern_two_odd_one_even
  (box : Box)
  (h_box_size : box.card = 100)
  (h_prob_first_odd : ∃ pattern : SelectionPattern, ProbFirstOdd pattern = 2/3) :
  ∃ pattern : SelectionPattern,
    pattern.first ≠ pattern.second ∨ pattern.first ≠ pattern.third ∨ pattern.second ≠ pattern.third :=
sorry

end pattern_two_odd_one_even_l1857_185772


namespace river_speed_calculation_l1857_185742

-- Define the swimmer's speed in still water
variable (a : ℝ) 

-- Define the speed of the river flow
def river_speed : ℝ := 0.02

-- Define the time the swimmer swam upstream before realizing the loss
def upstream_time : ℝ := 0.5

-- Define the distance downstream where the swimmer catches up to the bottle
def downstream_distance : ℝ := 1.2

-- Theorem statement
theorem river_speed_calculation (h : ∀ a > 0, 
  (downstream_distance + upstream_time * (a - 60 * river_speed)) / (a + 60 * river_speed) = 
  downstream_distance / (60 * river_speed) - upstream_time) : 
  river_speed = 0.02 := by sorry

end river_speed_calculation_l1857_185742


namespace smoking_hospitalization_percentage_l1857_185726

theorem smoking_hospitalization_percentage 
  (total_students : ℕ) 
  (smoking_percentage : ℚ) 
  (non_hospitalized : ℕ) 
  (h1 : total_students = 300)
  (h2 : smoking_percentage = 2/5)
  (h3 : non_hospitalized = 36) :
  (total_students * smoking_percentage - non_hospitalized) / (total_students * smoking_percentage) = 7/10 := by
  sorry

end smoking_hospitalization_percentage_l1857_185726


namespace ball_distribution_equality_l1857_185719

theorem ball_distribution_equality (k : ℤ) : ∃ (n : ℕ), (19 + 6 * n) % 95 = 0 := by
  -- The proof goes here
  sorry

end ball_distribution_equality_l1857_185719


namespace factorization_equality_l1857_185716

theorem factorization_equality (a x : ℝ) : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := by
  sorry

end factorization_equality_l1857_185716


namespace probability_of_pair_after_removal_l1857_185759

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset (Fin 13 × Fin 4))
  (card_count : cards.card = 52)

/-- Represents the deck after removing a pair and a single card -/
def RemainingDeck (d : Deck) : Finset (Fin 13 × Fin 4) :=
  d.cards.filter (λ _ => true)  -- This is a placeholder; actual implementation would remove cards

/-- Probability of selecting a matching pair from the remaining deck -/
def ProbabilityOfPair (d : Deck) : ℚ :=
  67 / 1176

/-- Main theorem: The probability of selecting a matching pair is 67/1176 -/
theorem probability_of_pair_after_removal (d : Deck) : 
  ProbabilityOfPair d = 67 / 1176 := by
  sorry

#eval (67 : ℕ) + 1176  -- Should output 1243

end probability_of_pair_after_removal_l1857_185759


namespace number_of_arrangements_l1857_185735

/-- The number of foreign guests -/
def num_foreign_guests : ℕ := 4

/-- The number of security officers -/
def num_security_officers : ℕ := 2

/-- The total number of individuals -/
def total_individuals : ℕ := num_foreign_guests + num_security_officers

/-- The number of foreign guests that must be together -/
def num_guests_together : ℕ := 2

/-- The function to calculate the number of possible arrangements -/
def calculate_arrangements (n_foreign : ℕ) (n_security : ℕ) (n_together : ℕ) : ℕ :=
  sorry

/-- The theorem stating the number of possible arrangements -/
theorem number_of_arrangements :
  calculate_arrangements num_foreign_guests num_security_officers num_guests_together = 24 :=
sorry

end number_of_arrangements_l1857_185735


namespace least_integer_square_98_more_than_double_l1857_185777

theorem least_integer_square_98_more_than_double : 
  ∃ x : ℤ, x^2 = 2*x + 98 ∧ ∀ y : ℤ, y^2 = 2*y + 98 → x ≤ y :=
by
  -- The proof goes here
  sorry

end least_integer_square_98_more_than_double_l1857_185777


namespace min_value_fraction_min_value_fraction_equality_l1857_185794

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x - 2*y + 3*z = 0) : 
  (y^2 / (x*z)) ≥ 3 := by
sorry

theorem min_value_fraction_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x - 2*y + 3*z = 0) : 
  (y^2 / (x*z) = 3) ↔ (x = 3*z) := by
sorry

end min_value_fraction_min_value_fraction_equality_l1857_185794


namespace unique_two_digit_sum_reverse_prime_l1857_185756

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The main theorem -/
theorem unique_two_digit_sum_reverse_prime :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ isPrime (n + reverseDigits n) :=
sorry

end unique_two_digit_sum_reverse_prime_l1857_185756


namespace power_four_remainder_l1857_185791

theorem power_four_remainder (a : ℕ) (h1 : a > 0) (h2 : 2 ∣ a) : 4^a % 10 = 6 := by
  sorry

end power_four_remainder_l1857_185791


namespace tan_alpha_value_l1857_185763

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α + Real.cos α) = -1) : 
  Real.tan α = -1/3 := by
  sorry

end tan_alpha_value_l1857_185763


namespace cube_sum_divisibility_l1857_185774

theorem cube_sum_divisibility (a : ℤ) (h1 : a > 1) 
  (h2 : ∃ (k : ℤ), (a - 1)^3 + a^3 + (a + 1)^3 = k^3) : 
  4 ∣ a := by
  sorry

end cube_sum_divisibility_l1857_185774


namespace grass_seed_cost_l1857_185751

/-- Represents the cost and weight of a bag of grass seed -/
structure BagInfo where
  weight : ℕ
  cost : ℚ

/-- Calculates the total cost of a given number of bags -/
def totalCost (bag : BagInfo) (count : ℕ) : ℚ :=
  bag.cost * count

/-- Calculates the total weight of a given number of bags -/
def totalWeight (bag : BagInfo) (count : ℕ) : ℕ :=
  bag.weight * count

theorem grass_seed_cost
  (bag5 : BagInfo)
  (bag10 : BagInfo)
  (bag25 : BagInfo)
  (h1 : bag5.weight = 5)
  (h2 : bag10.weight = 10)
  (h3 : bag10.cost = 20.43)
  (h4 : bag25.weight = 25)
  (h5 : bag25.cost = 32.25)
  (h6 : ∃ (c5 c10 c25 : ℕ), 
    65 ≤ totalWeight bag5 c5 + totalWeight bag10 c10 + totalWeight bag25 c25 ∧
    totalWeight bag5 c5 + totalWeight bag10 c10 + totalWeight bag25 c25 ≤ 80 ∧
    totalCost bag5 c5 + totalCost bag10 c10 + totalCost bag25 c25 = 98.75 ∧
    ∀ (d5 d10 d25 : ℕ),
      65 ≤ totalWeight bag5 d5 + totalWeight bag10 d10 + totalWeight bag25 d25 →
      totalWeight bag5 d5 + totalWeight bag10 d10 + totalWeight bag25 d25 ≤ 80 →
      totalCost bag5 d5 + totalCost bag10 d10 + totalCost bag25 d25 ≥ 98.75) :
  bag5.cost = 13.82 := by
sorry

end grass_seed_cost_l1857_185751


namespace product_in_base7_l1857_185757

/-- Converts a base-7 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 7) ((m % 7) :: acc)
    go n []

/-- The product of 325₇ and 6₇ in base 7 is 2624₇ --/
theorem product_in_base7 :
  toBase7 (toBase10 [5, 2, 3] * toBase10 [6]) = [4, 2, 6, 2] := by
  sorry

end product_in_base7_l1857_185757


namespace count_triplets_eq_30_l1857_185787

/-- Count of ordered triplets (a, b, c) of positive integers satisfying 30a + 50b + 70c ≤ 343 -/
def count_triplets : ℕ :=
  (Finset.filter (fun (t : ℕ × ℕ × ℕ) =>
    let (a, b, c) := t
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 30 * a + 50 * b + 70 * c ≤ 343)
    (Finset.product (Finset.range 12) (Finset.product (Finset.range 7) (Finset.range 5)))).card

theorem count_triplets_eq_30 : count_triplets = 30 := by
  sorry

end count_triplets_eq_30_l1857_185787


namespace cricket_average_score_l1857_185755

theorem cricket_average_score (score1 score2 : ℝ) (n1 n2 : ℕ) (h1 : score1 = 27) (h2 : score2 = 32) (h3 : n1 = 2) (h4 : n2 = 3) :
  (score1 * n1 + score2 * n2) / (n1 + n2) = 30 := by
  sorry

end cricket_average_score_l1857_185755


namespace vector_decomposition_l1857_185743

def x : ℝ × ℝ × ℝ := (-13, 2, 18)
def p : ℝ × ℝ × ℝ := (1, 1, 4)
def q : ℝ × ℝ × ℝ := (-3, 0, 2)
def r : ℝ × ℝ × ℝ := (1, 2, -1)

theorem vector_decomposition :
  x = (2 : ℝ) • p + (5 : ℝ) • q := by sorry

end vector_decomposition_l1857_185743


namespace circle_symmetry_l1857_185725

-- Define the original circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y = 2

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y x' y' : ℝ),
    circle_C x y →
    line_l ((x + x') / 2) ((y + y') / 2) →
    symmetric_circle x' y' :=
by sorry

end circle_symmetry_l1857_185725


namespace max_consecutive_integers_sum_thirty_one_is_max_max_consecutive_integers_sum_is_31_l1857_185752

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by
  sorry

theorem thirty_one_is_max : ∀ m : ℕ, m > 31 → m * (m + 1) > 1000 := by
  sorry

theorem max_consecutive_integers_sum_is_31 :
  (∃ n : ℕ, n * (n + 1) ≤ 1000 ∧ ∀ m : ℕ, m > n → m * (m + 1) > 1000) ∧
  (∀ n : ℕ, n * (n + 1) ≤ 1000 ∧ (∀ m : ℕ, m > n → m * (m + 1) > 1000) → n = 31) := by
  sorry

end max_consecutive_integers_sum_thirty_one_is_max_max_consecutive_integers_sum_is_31_l1857_185752
