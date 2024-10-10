import Mathlib

namespace fresh_fruit_amount_l1236_123604

-- Define the total amount of fruit sold
def total_fruit : ℕ := 9792

-- Define the amount of frozen fruit sold
def frozen_fruit : ℕ := 3513

-- Define the amount of fresh fruit sold
def fresh_fruit : ℕ := total_fruit - frozen_fruit

-- Theorem to prove
theorem fresh_fruit_amount : fresh_fruit = 6279 := by
  sorry

end fresh_fruit_amount_l1236_123604


namespace village_chief_assistants_l1236_123608

theorem village_chief_assistants (n : ℕ) (k : ℕ) (a b c : Fin n) (h1 : n = 10) (h2 : k = 3) :
  let total_combinations := Nat.choose n k
  let combinations_without_ab := Nat.choose (n - 2) k
  total_combinations - combinations_without_ab = 49 :=
sorry

end village_chief_assistants_l1236_123608


namespace songs_leftover_l1236_123683

theorem songs_leftover (total_songs : ℕ) (num_playlists : ℕ) (h1 : total_songs = 372) (h2 : num_playlists = 9) :
  total_songs % num_playlists = 3 := by
  sorry

end songs_leftover_l1236_123683


namespace ellipse_m_value_l1236_123621

/-- Represents an ellipse with equation x^2 + my^2 = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ x y : ℝ, x^2 + m * y^2 = 1

/-- Indicates that the foci of the ellipse are on the y-axis -/
def foci_on_y_axis (e : Ellipse m) : Prop :=
  ∃ c : ℝ, c^2 = 1/m - 1 ∧ c ≥ 0

/-- The length of the major axis is twice the length of the minor axis -/
def major_axis_twice_minor (e : Ellipse m) : Prop :=
  2 * (1 : ℝ) = Real.sqrt (1/m)

/-- Theorem stating that m = 1/4 for the given ellipse properties -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m)
  (h1 : foci_on_y_axis e)
  (h2 : major_axis_twice_minor e) :
  m = 1/4 := by
  sorry

end ellipse_m_value_l1236_123621


namespace distinct_prime_factors_count_l1236_123633

/-- The product of the given numbers -/
def product : ℕ := 101 * 103 * 105 * 107

/-- The set of prime factors of the product -/
def prime_factors : Finset ℕ := sorry

theorem distinct_prime_factors_count :
  Finset.card prime_factors = 6 := by sorry

end distinct_prime_factors_count_l1236_123633


namespace max_value_under_constraints_l1236_123619

theorem max_value_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 12) : 
  2 * x + y ≤ 39 / 11 := by
  sorry

end max_value_under_constraints_l1236_123619


namespace inequality_solution_l1236_123661

theorem inequality_solution (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 8) ≥ 1 / 2 ↔ x ∈ Set.Ioc (-8) (-4) ∪ Set.Icc (-2) 2 :=
sorry

end inequality_solution_l1236_123661


namespace exists_b_for_234_quadrants_l1236_123693

-- Define the linear function
def f (b : ℝ) (x : ℝ) : ℝ := -2 * x + b

-- Define the property of passing through the second, third, and fourth quadrants
def passes_through_234_quadrants (b : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ,
    (x₁ < 0 ∧ f b x₁ > 0) ∧  -- Second quadrant
    (x₂ < 0 ∧ f b x₂ < 0) ∧  -- Third quadrant
    (x₃ > 0 ∧ f b x₃ < 0)    -- Fourth quadrant

-- Theorem statement
theorem exists_b_for_234_quadrants :
  ∃ b : ℝ, b < 0 ∧ passes_through_234_quadrants b :=
sorry

end exists_b_for_234_quadrants_l1236_123693


namespace angle_sum_quadrilateral_l1236_123677

theorem angle_sum_quadrilateral (a b : ℝ) : 
  36 + b + 44 + 52 = 180 → b = 48 := by
  sorry

end angle_sum_quadrilateral_l1236_123677


namespace initial_red_marbles_l1236_123616

/-- Represents the number of marbles in a bag -/
structure MarbleBag where
  red : ℚ
  green : ℚ

/-- The initial ratio of red to green marbles is 7:3 -/
def initial_ratio (bag : MarbleBag) : Prop :=
  bag.red / bag.green = 7 / 3

/-- After removing 14 red marbles and adding 30 green marbles, the new ratio is 1:4 -/
def new_ratio (bag : MarbleBag) : Prop :=
  (bag.red - 14) / (bag.green + 30) = 1 / 4

/-- Theorem stating that the initial number of red marbles is 24 -/
theorem initial_red_marbles (bag : MarbleBag) :
  initial_ratio bag → new_ratio bag → bag.red = 24 := by
  sorry

end initial_red_marbles_l1236_123616


namespace toy_store_shelves_l1236_123686

/-- Calculates the number of shelves needed to store bears in a toy store. -/
def shelves_needed (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  (initial_stock + new_shipment) / bears_per_shelf

/-- Proves that given the specific conditions, the number of shelves needed is 5. -/
theorem toy_store_shelves : shelves_needed 15 45 12 = 5 := by
  sorry

end toy_store_shelves_l1236_123686


namespace composite_rectangle_theorem_l1236_123692

/-- The side length of square S2 in the composite rectangle. -/
def side_length_S2 : ℕ := 775

/-- The width of the composite rectangle. -/
def total_width : ℕ := 4000

/-- The height of the composite rectangle. -/
def total_height : ℕ := 2450

/-- The shorter side length of rectangles R1 and R2. -/
def shorter_side_R : ℕ := (total_height - side_length_S2) / 2

theorem composite_rectangle_theorem :
  (2 * shorter_side_R + side_length_S2 = total_height) ∧
  (2 * shorter_side_R + 3 * side_length_S2 = total_width) := by
  sorry

#check composite_rectangle_theorem

end composite_rectangle_theorem_l1236_123692


namespace min_m_intersection_nonempty_l1236_123669

def set_B (m : ℝ) : Set (ℝ × ℝ) := {p | 3 * p.1 + 2 * p.2 - m = 0}

theorem min_m_intersection_nonempty (A : Set (ℝ × ℝ)) (h : ∃ m : ℝ, (A ∩ set_B m).Nonempty) :
  ∃ m_min : ℝ, m_min = 0 ∧ (A ∩ set_B m_min).Nonempty ∧ ∀ m : ℝ, (A ∩ set_B m).Nonempty → m ≥ m_min :=
by
  sorry

end min_m_intersection_nonempty_l1236_123669


namespace expression_evaluation_l1236_123624

theorem expression_evaluation (x : ℝ) (hx : x^2 - 2*x - 3 = 0) (hx_neq : x ≠ 3) :
  (2 / (x - 3) - 1 / x) * ((x^2 - 3*x) / (x^2 + 6*x + 9)) = 1 / 2 := by
  sorry

end expression_evaluation_l1236_123624


namespace min_pet_owners_l1236_123674

/-- Represents the number of people who own only dogs -/
def only_dogs : Nat := 15

/-- Represents the number of people who own only cats -/
def only_cats : Nat := 10

/-- Represents the number of people who own only cats and dogs -/
def cats_and_dogs : Nat := 5

/-- Represents the number of people who own cats, dogs, and snakes -/
def cats_dogs_snakes : Nat := 3

/-- Represents the total number of snakes -/
def total_snakes : Nat := 59

/-- Theorem stating that the minimum number of pet owners is 33 -/
theorem min_pet_owners : 
  only_dogs + only_cats + cats_and_dogs + cats_dogs_snakes = 33 := by
  sorry

#check min_pet_owners

end min_pet_owners_l1236_123674


namespace square_equation_solution_l1236_123622

theorem square_equation_solution :
  ∃ x : ℝ, (3000 + x)^2 = x^2 ∧ x = -1500 := by sorry

end square_equation_solution_l1236_123622


namespace murtha_pebbles_l1236_123610

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Murtha's pebble collection problem -/
theorem murtha_pebbles : arithmetic_sum 3 3 18 = 513 := by
  sorry

end murtha_pebbles_l1236_123610


namespace age_problem_solution_l1236_123605

/-- The ages of the king and queen satisfy the given conditions -/
def age_problem (king_age queen_age : ℕ) : Prop :=
  ∃ (t : ℕ),
    -- The king's current age is twice the queen's age when the king was as old as the queen is now
    king_age = 2 * (queen_age - t) ∧
    -- When the queen is as old as the king is now, their combined ages will be 63 years
    king_age + (king_age + t) = 63 ∧
    -- The age difference
    king_age - queen_age = t

/-- The solution to the age problem -/
theorem age_problem_solution :
  ∃ (king_age queen_age : ℕ), age_problem king_age queen_age ∧ king_age = 28 ∧ queen_age = 21 :=
sorry

end age_problem_solution_l1236_123605


namespace sequential_structure_essential_l1236_123644

/-- Represents the different types of algorithm structures -/
inductive AlgorithmStructure
  | Logical
  | Selection
  | Loop
  | Sequential

/-- Represents an algorithm -/
structure Algorithm where
  structures : List AlgorithmStructure

/-- Defines what it means for a structure to be essential for all algorithms -/
def isEssentialStructure (s : AlgorithmStructure) : Prop :=
  ∀ (a : Algorithm), s ∈ a.structures

/-- States that an algorithm can exist without Logical, Selection, or Loop structures -/
axiom non_essential_structures :
  ∃ (a : Algorithm),
    AlgorithmStructure.Logical ∉ a.structures ∧
    AlgorithmStructure.Selection ∉ a.structures ∧
    AlgorithmStructure.Loop ∉ a.structures

/-- The main theorem: Sequential structure is the only essential structure -/
theorem sequential_structure_essential :
  isEssentialStructure AlgorithmStructure.Sequential ∧
  (∀ s : AlgorithmStructure, s ≠ AlgorithmStructure.Sequential → ¬isEssentialStructure s) :=
sorry

end sequential_structure_essential_l1236_123644


namespace sport_participation_l1236_123646

theorem sport_participation (total : ℕ) (football : ℕ) (basketball : ℕ) (baseball : ℕ) (all_three : ℕ)
  (h1 : total = 427)
  (h2 : football = 128)
  (h3 : basketball = 291)
  (h4 : baseball = 318)
  (h5 : all_three = 36)
  (h6 : total = football + basketball + baseball - (football_basketball + football_baseball + basketball_baseball) + all_three)
  : football_basketball + football_baseball + basketball_baseball - 3 * all_three = 274 :=
by sorry

end sport_participation_l1236_123646


namespace scooter_gain_percent_l1236_123653

/-- Calculate the gain percent for a scooter sale -/
theorem scooter_gain_percent (purchase_price repair_costs selling_price : ℝ) 
  (h1 : purchase_price = 4700)
  (h2 : repair_costs = 800)
  (h3 : selling_price = 6000) : 
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 9.09 :=
by sorry

end scooter_gain_percent_l1236_123653


namespace son_age_proof_l1236_123639

theorem son_age_proof (father_age son_age : ℕ) : 
  father_age = son_age + 29 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 27 := by
sorry

end son_age_proof_l1236_123639


namespace johns_allowance_l1236_123688

/-- John's weekly allowance problem -/
theorem johns_allowance :
  ∀ (A : ℚ),
  (A > 0) →
  (3 / 5 * A + 1 / 3 * (2 / 5 * A) + 0.4 = A) →
  A = 1.5 := by
  sorry

end johns_allowance_l1236_123688


namespace line_equation_through_points_l1236_123614

/-- The equation of a line passing through two given points -/
theorem line_equation_through_points (x y : ℝ) : 
  (2 * x - y - 2 = 0) ↔ 
  ((x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = -2) ∨ 
   (∃ t : ℝ, x = 1 - t ∧ y = 0 + 2*t)) :=
sorry

end line_equation_through_points_l1236_123614


namespace triangle_internal_point_theorem_l1236_123663

/-- Triangle with sides a, b, c and internal point P --/
structure TriangleWithInternalPoint where
  a : ℝ
  b : ℝ
  c : ℝ
  P : ℝ × ℝ

/-- Parallel segments through P have equal length d --/
def parallelSegmentsEqual (T : TriangleWithInternalPoint) (d : ℝ) : Prop :=
  ∃ (x y z : ℝ), x + y + z = T.a ∧ x + y + z = T.b ∧ x + y + z = T.c ∧ x = y ∧ y = z ∧ z = d

theorem triangle_internal_point_theorem (T : TriangleWithInternalPoint) 
    (h1 : T.a = 550) (h2 : T.b = 580) (h3 : T.c = 620) :
    ∃ (d : ℝ), parallelSegmentsEqual T d ∧ d = 342 := by
  sorry

end triangle_internal_point_theorem_l1236_123663


namespace function_identity_l1236_123694

theorem function_identity (f : ℕ → ℕ) (h : ∀ n, f (n + 1) > f (f n)) : ∀ n, f n = n := by
  sorry

end function_identity_l1236_123694


namespace point_on_line_l1236_123654

/-- Given five points O, A, B, C, D on a straight line and a point P between B and C,
    prove that OP = 1 + 4√3 under the given conditions. -/
theorem point_on_line (O A B C D P : ℝ) : 
  O < A ∧ A < B ∧ B < C ∧ C < D ∧  -- Points are in order on the line
  A - O = 1 ∧                      -- OA = 1
  B - O = 3 ∧                      -- OB = 3
  C - O = 5 ∧                      -- OC = 5
  D - O = 7 ∧                      -- OD = 7
  B < P ∧ P < C ∧                  -- P is between B and C
  (P - A) / (D - P) = 2 * (P - B) / (C - P)  -- AP : PD = 2(BP : PC)
  → P - O = 1 + 4 * Real.sqrt 3 := by
sorry

end point_on_line_l1236_123654


namespace roots_polynomial_sum_l1236_123662

theorem roots_polynomial_sum (a b c : ℝ) (s : ℝ) : 
  (∀ x, x^3 - 9*x^2 + 11*x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  s = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  s^4 - 18*s^2 - 8*s = -37 := by
sorry

end roots_polynomial_sum_l1236_123662


namespace paint_remaining_l1236_123612

theorem paint_remaining (initial_paint : ℚ) : 
  initial_paint = 1 →
  let remaining_after_day1 := initial_paint - (3/8 * initial_paint)
  let remaining_after_day2 := remaining_after_day1 - (1/4 * remaining_after_day1)
  remaining_after_day2 = 15/32 := by
  sorry

end paint_remaining_l1236_123612


namespace student_pet_difference_l1236_123602

/-- Represents a fourth-grade classroom -/
structure Classroom where
  students : ℕ
  rabbits : ℕ
  birds : ℕ

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- A fourth-grade classroom at Green Park Elementary -/
def green_park_classroom : Classroom := {
  students := 22,
  rabbits := 3,
  birds := 2
}

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * green_park_classroom.students

/-- The total number of pets (rabbits and birds) in all classrooms -/
def total_pets : ℕ := num_classrooms * (green_park_classroom.rabbits + green_park_classroom.birds)

/-- Theorem: The difference between the total number of students and the total number of pets is 85 -/
theorem student_pet_difference : total_students - total_pets = 85 := by
  sorry

end student_pet_difference_l1236_123602


namespace bug_position_after_2010_jumps_l1236_123676

/-- Represents the six points on the circle -/
inductive Point
| one
| two
| three
| four
| five
| six

/-- Determines if a point is odd-numbered -/
def is_odd (p : Point) : Bool :=
  match p with
  | Point.one => true
  | Point.three => true
  | Point.five => true
  | _ => false

/-- Performs one jump based on the current point -/
def jump (p : Point) : Point :=
  if is_odd p then
    match p with
    | Point.one => Point.two
    | Point.three => Point.four
    | Point.five => Point.six
    | _ => p  -- This case should never occur
  else
    match p with
    | Point.two => Point.five
    | Point.four => Point.one
    | Point.six => Point.three
    | _ => p  -- This case should never occur

/-- Performs n jumps starting from a given point -/
def multi_jump (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | Nat.succ m => jump (multi_jump start m)

theorem bug_position_after_2010_jumps : 
  multi_jump Point.six 2010 = Point.two :=
by sorry

end bug_position_after_2010_jumps_l1236_123676


namespace largest_multiple_of_seven_below_negative_fifty_l1236_123664

theorem largest_multiple_of_seven_below_negative_fifty :
  ∀ n : ℤ, n * 7 < -50 → n * 7 ≤ -56 :=
by
  sorry

end largest_multiple_of_seven_below_negative_fifty_l1236_123664


namespace product_remainder_l1236_123667

theorem product_remainder (x y : ℤ) 
  (hx : x % 315 = 53) 
  (hy : y % 385 = 41) : 
  (x * y) % 21 = 10 := by
  sorry

end product_remainder_l1236_123667


namespace mr_a_net_gain_l1236_123603

def initial_value : ℚ := 12000
def first_sale_profit : ℚ := 20 / 100
def second_sale_loss : ℚ := 15 / 100
def third_sale_profit : ℚ := 10 / 100

theorem mr_a_net_gain : 
  let first_sale := initial_value * (1 + first_sale_profit)
  let second_sale := first_sale * (1 - second_sale_loss)
  let third_sale := second_sale * (1 + third_sale_profit)
  first_sale - second_sale + third_sale - initial_value = 3384 := by
sorry

end mr_a_net_gain_l1236_123603


namespace parallel_angles_theorem_l1236_123659

/-- Given two angles A and B where the sides of A are parallel to the sides of B, 
    prove that if B = 3A - 60°, then B is either 30° or 120° -/
theorem parallel_angles_theorem (A B : ℝ) : 
  (B = 3 * A - 60) → (B = 30 ∨ B = 120) := by
  sorry

end parallel_angles_theorem_l1236_123659


namespace max_distance_to_upper_vertex_l1236_123647

def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

def upper_vertex (B : ℝ × ℝ) : Prop :=
  B.1 = 0 ∧ B.2 = 1 ∧ ellipse B.1 B.2

theorem max_distance_to_upper_vertex :
  ∃ (B : ℝ × ℝ), upper_vertex B ∧
  ∀ (P : ℝ × ℝ), ellipse P.1 P.2 →
  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) ≤ 5/2 :=
sorry

end max_distance_to_upper_vertex_l1236_123647


namespace floor_of_expression_equals_32_l1236_123666

theorem floor_of_expression_equals_32 :
  ⌊(1 + (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 4) / 
     (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 8 + 4))^10⌋ = 32 := by
  sorry

end floor_of_expression_equals_32_l1236_123666


namespace sum_of_coefficients_l1236_123672

noncomputable def g (x : ℝ) (A B C : ℤ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem sum_of_coefficients 
  (A B C : ℤ) 
  (h1 : ∀ x > 5, g x A B C > 0.5)
  (h2 : (A : ℝ) * (-3)^2 + B * (-3) + C = 0)
  (h3 : (A : ℝ) * 4^2 + B * 4 + C = 0) :
  A + B + C = -24 := by sorry

end sum_of_coefficients_l1236_123672


namespace bruces_shopping_money_l1236_123681

theorem bruces_shopping_money (initial_amount : ℕ) (shirt_cost : ℕ) (num_shirts : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 71 →
  shirt_cost = 5 →
  num_shirts = 5 →
  pants_cost = 26 →
  remaining_amount = initial_amount - (num_shirts * shirt_cost + pants_cost) →
  remaining_amount = 20 := by
sorry

end bruces_shopping_money_l1236_123681


namespace quadratic_real_roots_l1236_123600

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + k^2 - 1 = 0) ↔ 
  (-2 / Real.sqrt 3 ≤ k ∧ k ≤ 2 / Real.sqrt 3 ∧ k ≠ 0) :=
by sorry

end quadratic_real_roots_l1236_123600


namespace zero_is_rational_l1236_123684

/-- A number is rational if it can be expressed as the quotient of two integers with a non-zero denominator -/
def IsRational (x : ℚ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

/-- Theorem: Zero is a rational number -/
theorem zero_is_rational : IsRational 0 := by
  sorry

end zero_is_rational_l1236_123684


namespace smallest_square_l1236_123631

theorem smallest_square (a b : ℕ+) 
  (h1 : ∃ r : ℕ+, (15 * a + 16 * b : ℕ) = r ^ 2)
  (h2 : ∃ s : ℕ+, (16 * a - 15 * b : ℕ) = s ^ 2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 481 ^ 2 ∧
  ∃ (a₀ b₀ : ℕ+), (15 * a₀ + 16 * b₀ : ℕ) = 481 ^ 2 ∧ (16 * a₀ - 15 * b₀ : ℕ) = 481 ^ 2 := by
  sorry

end smallest_square_l1236_123631


namespace janet_savings_l1236_123638

theorem janet_savings (monthly_rent : ℕ) (advance_months : ℕ) (deposit : ℕ) (additional_needed : ℕ) : 
  monthly_rent = 1250 →
  advance_months = 2 →
  deposit = 500 →
  additional_needed = 775 →
  monthly_rent * advance_months + deposit - additional_needed = 2225 :=
by sorry

end janet_savings_l1236_123638


namespace sum_of_squares_l1236_123651

theorem sum_of_squares (a b : ℝ) : (a^2 + b^2) * (a^2 + b^2 - 4) - 5 = 0 → a^2 + b^2 = 5 := by
  sorry

end sum_of_squares_l1236_123651


namespace reduced_rates_fraction_l1236_123650

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the number of hours in a day
def hours_in_day : ℕ := 24

-- Define the number of weekdays (Monday to Friday)
def weekdays : ℕ := 5

-- Define the number of weekend days (Saturday and Sunday)
def weekend_days : ℕ := 2

-- Define the number of hours with reduced rates on weekdays (8 p.m. to 8 a.m.)
def reduced_hours_weekday : ℕ := 12

-- Define the number of hours with reduced rates on weekend days (24 hours)
def reduced_hours_weekend : ℕ := 24

-- Theorem stating that the fraction of a week with reduced rates is 9/14
theorem reduced_rates_fraction :
  (weekdays * reduced_hours_weekday + weekend_days * reduced_hours_weekend) / 
  (days_in_week * hours_in_day) = 9 / 14 := by
  sorry

end reduced_rates_fraction_l1236_123650


namespace zeroes_of_f_range_of_a_l1236_123620

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + b - 1

-- Theorem for the zeroes of f(x) when a = 1 and b = -2
theorem zeroes_of_f : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f 1 (-2) x₁ = 0 ∧ f 1 (-2) x₂ = 0 ∧ x₁ = 3 ∧ x₂ = -1 :=
sorry

-- Theorem for the range of a when f(x) always has two distinct zeroes
theorem range_of_a (a : ℝ) : 
  (a ≠ 0 ∧ ∀ b : ℝ, ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0) ↔ 0 < a ∧ a < 1 :=
sorry

end zeroes_of_f_range_of_a_l1236_123620


namespace circular_arrangement_theorem_l1236_123636

/-- Represents a circular seating arrangement of men and women -/
structure CircularArrangement where
  total_people : ℕ
  women : ℕ
  men : ℕ
  women_left_of_women : ℕ
  men_left_of_women : ℕ
  women_right_of_men_ratio : ℚ

/-- The properties of the circular arrangement in the problem -/
def problem_arrangement : CircularArrangement where
  total_people := 35
  women := 19
  men := 16
  women_left_of_women := 7
  men_left_of_women := 12
  women_right_of_men_ratio := 3/4

theorem circular_arrangement_theorem (arr : CircularArrangement) :
  arr.women_left_of_women = 7 ∧
  arr.men_left_of_women = 12 ∧
  arr.women_right_of_men_ratio = 3/4 →
  arr.total_people = 35 ∧
  arr.women = 19 ∧
  arr.men = 16 := by
  sorry

#check circular_arrangement_theorem problem_arrangement

end circular_arrangement_theorem_l1236_123636


namespace calculus_class_mean_l1236_123678

/-- Calculates the class mean given the number of students and average scores for three groups -/
def class_mean (total_students : ℕ) (group1_students : ℕ) (group1_avg : ℚ) 
               (group2_students : ℕ) (group2_avg : ℚ)
               (group3_students : ℕ) (group3_avg : ℚ) : ℚ :=
  (group1_students * group1_avg + group2_students * group2_avg + group3_students * group3_avg) / total_students

theorem calculus_class_mean :
  let total_students : ℕ := 60
  let group1_students : ℕ := 40
  let group1_avg : ℚ := 68 / 100
  let group2_students : ℕ := 15
  let group2_avg : ℚ := 74 / 100
  let group3_students : ℕ := 5
  let group3_avg : ℚ := 88 / 100
  class_mean total_students group1_students group1_avg group2_students group2_avg group3_students group3_avg = 4270 / 60 :=
by sorry

end calculus_class_mean_l1236_123678


namespace school_trip_student_count_l1236_123628

theorem school_trip_student_count :
  let num_buses : ℕ := 95
  let max_seats_per_bus : ℕ := 118
  let bus_capacity_percentage : ℚ := 9/10
  let attendance_percentage : ℚ := 4/5
  let total_students : ℕ := 12588
  (↑num_buses * ↑max_seats_per_bus * bus_capacity_percentage).floor = 
    (↑total_students * attendance_percentage).floor := by
  sorry

end school_trip_student_count_l1236_123628


namespace midpoint_distance_theorem_l1236_123698

theorem midpoint_distance_theorem (t : ℝ) : 
  let P : ℝ × ℝ := (t - 5, -2)
  let Q : ℝ × ℝ := (-3, t + 4)
  let midpoint : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let midpoint_to_endpoint_sq := ((midpoint.1 - P.1)^2 + (midpoint.2 - P.2)^2)
  midpoint_to_endpoint_sq = t^2 / 3 →
  t = -12 - 2 * Real.sqrt 21 ∨ t = -12 + 2 * Real.sqrt 21 :=
by sorry

end midpoint_distance_theorem_l1236_123698


namespace book_arrangement_and_distribution_l1236_123640

/-- The number of ways to arrange 5 books, including 2 mathematics books, in a row such that
    the mathematics books are not adjacent and not placed at both ends simultaneously. -/
def arrange_books : ℕ := 60

/-- The number of ways to distribute 5 books, including 2 mathematics books, to 3 students,
    with each student receiving at least 1 book. -/
def distribute_books : ℕ := 150

/-- Theorem stating the correct number of arrangements and distributions -/
theorem book_arrangement_and_distribution :
  arrange_books = 60 ∧ distribute_books = 150 := by
  sorry

end book_arrangement_and_distribution_l1236_123640


namespace intersection_M_N_l1236_123671

def M : Set (ℝ × ℝ) := {p | (p.1^2 / 9) + (p.2^2 / 4) = 1}

def N : Set (ℝ × ℝ) := {p | (p.1 / 3) + (p.2 / 2) = 1}

theorem intersection_M_N : M ∩ N = {(3, 0), (0, 2)} := by sorry

end intersection_M_N_l1236_123671


namespace dividing_line_theorem_l1236_123623

/-- Represents a disk in 2D space -/
structure Disk where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the configuration of five disks -/
structure DiskConfiguration where
  disks : Fin 5 → Disk
  square_vertices : Fin 4 → ℝ × ℝ
  aligned_centers : Fin 3 → ℝ × ℝ

/-- Represents a line in 2D space -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The center of a square given its vertices -/
def square_center (vertices : Fin 4 → ℝ × ℝ) : ℝ × ℝ := sorry

/-- Calculates the area of the figure formed by the disks on one side of a line -/
def area_on_side (config : DiskConfiguration) (line : Line) : ℝ := sorry

/-- States that the line passing through the square center and the fifth disk's center
    divides the total area of the five disks into two equal parts -/
theorem dividing_line_theorem (config : DiskConfiguration) :
  let square_center := square_center config.square_vertices
  let fifth_disk_center := (config.disks 4).center
  let dividing_line := Line.mk square_center fifth_disk_center
  area_on_side config dividing_line = (area_on_side config dividing_line) / 2 := by sorry

end dividing_line_theorem_l1236_123623


namespace polynomial_simplification_l1236_123657

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  2*x^3 - x^2 + 23*x - 3 := by
sorry

end polynomial_simplification_l1236_123657


namespace tangent_points_tangent_parallel_points_l1236_123630

/-- The function f(x) = x³ + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_points :
  ∀ x : ℝ, (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
by sorry

theorem tangent_parallel_points :
  ∀ x : ℝ, (f' x = 4) → (x = 1 ∧ f x = 0) ∨ (x = -1 ∧ f x = -4) :=
by sorry

end tangent_points_tangent_parallel_points_l1236_123630


namespace coffee_payment_dimes_l1236_123609

/-- Represents the number of coins of each type used in the payment -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The total value of the coins in cents -/
def totalValue (c : CoinCount) : ℕ :=
  c.pennies + 5 * c.nickels + 10 * c.dimes

/-- The total number of coins -/
def totalCoins (c : CoinCount) : ℕ :=
  c.pennies + c.nickels + c.dimes

theorem coffee_payment_dimes :
  ∃ (c : CoinCount),
    totalValue c = 200 ∧
    totalCoins c = 50 ∧
    c.dimes = 14 :=
by sorry

end coffee_payment_dimes_l1236_123609


namespace initial_charge_is_3_5_l1236_123613

/-- A taxi company's pricing model -/
structure TaxiCompany where
  initialCharge : ℝ  -- Initial charge for the first 1/5 mile
  additionalCharge : ℝ  -- Charge for each additional 1/5 mile
  totalCharge : ℝ  -- Total charge for a specific ride
  rideLength : ℝ  -- Length of the ride in miles

/-- The initial charge for the first 1/5 mile is $3.5 -/
theorem initial_charge_is_3_5 (t : TaxiCompany) 
    (h1 : t.additionalCharge = 0.4)
    (h2 : t.totalCharge = 19.1)
    (h3 : t.rideLength = 8) : 
    t.initialCharge = 3.5 := by
  sorry

#check initial_charge_is_3_5

end initial_charge_is_3_5_l1236_123613


namespace prob_at_most_one_white_ball_l1236_123625

/-- The number of black balls in the box -/
def black_balls : ℕ := 10

/-- The number of red balls in the box -/
def red_balls : ℕ := 12

/-- The number of white balls in the box -/
def white_balls : ℕ := 4

/-- The total number of balls in the box -/
def total_balls : ℕ := black_balls + red_balls + white_balls

/-- The number of balls drawn -/
def drawn_balls : ℕ := 2

/-- X represents the number of white balls drawn -/
def X : Fin (drawn_balls + 1) → ℕ := sorry

/-- The probability of drawing at most one white ball -/
def P_X_le_1 : ℚ := sorry

/-- The main theorem to prove -/
theorem prob_at_most_one_white_ball :
  P_X_le_1 = (Nat.choose (total_balls - white_balls) 1 * Nat.choose white_balls 1 + 
              Nat.choose (total_balls - white_balls) 2) / 
             Nat.choose total_balls 2 :=
sorry

end prob_at_most_one_white_ball_l1236_123625


namespace field_area_in_square_yards_l1236_123675

/-- Conversion rate from feet to yards -/
def feet_to_yard : ℝ := 3

/-- Length of the field in feet -/
def field_length_feet : ℝ := 12

/-- Width of the field in feet -/
def field_width_feet : ℝ := 9

/-- Theorem stating that the area of the field in square yards is 12 -/
theorem field_area_in_square_yards :
  (field_length_feet / feet_to_yard) * (field_width_feet / feet_to_yard) = 12 :=
by sorry

end field_area_in_square_yards_l1236_123675


namespace four_of_a_kind_probability_l1236_123691

-- Define a standard deck of cards
def standardDeck : ℕ := 52

-- Define the number of cards drawn
def cardsDrawn : ℕ := 6

-- Define the number of different card values (ranks)
def cardValues : ℕ := 13

-- Define the number of cards of each value
def cardsPerValue : ℕ := 4

-- Function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem four_of_a_kind_probability :
  (cardValues * binomial (standardDeck - cardsPerValue) (cardsDrawn - cardsPerValue)) /
  (binomial standardDeck cardsDrawn) = 3 / 4165 :=
sorry

end four_of_a_kind_probability_l1236_123691


namespace parallelepiped_diagonal_squared_l1236_123626

/-- The square of the diagonal of a rectangular parallelepiped is equal to the sum of squares of its dimensions -/
theorem parallelepiped_diagonal_squared (p q r : ℝ) :
  let diagonal_squared := p^2 + q^2 + r^2
  diagonal_squared = p^2 + q^2 + r^2 := by sorry

end parallelepiped_diagonal_squared_l1236_123626


namespace unique_intersection_point_l1236_123634

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 5*x^2 + 12*x + 20

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-5, -5) := by
  sorry

end unique_intersection_point_l1236_123634


namespace sqrt_two_times_two_minus_sqrt_two_sqrt_six_div_sqrt_three_times_sqrt_twentyfour_sum_of_square_roots_squared_difference_minus_product_l1236_123680

-- Problem 1
theorem sqrt_two_times_two_minus_sqrt_two :
  Real.sqrt 2 * (2 - Real.sqrt 2) = 2 * Real.sqrt 2 - 2 := by sorry

-- Problem 2
theorem sqrt_six_div_sqrt_three_times_sqrt_twentyfour :
  Real.sqrt 6 / Real.sqrt 3 * Real.sqrt 24 = 4 * Real.sqrt 3 := by sorry

-- Problem 3
theorem sum_of_square_roots :
  Real.sqrt 54 + Real.sqrt 24 - Real.sqrt 18 + 2 * Real.sqrt (1/2) = 5 * Real.sqrt 6 - 2 * Real.sqrt 2 := by sorry

-- Problem 4
theorem squared_difference_minus_product :
  (Real.sqrt 2 - 1)^2 - (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) = 2 - 2 * Real.sqrt 2 := by sorry

end sqrt_two_times_two_minus_sqrt_two_sqrt_six_div_sqrt_three_times_sqrt_twentyfour_sum_of_square_roots_squared_difference_minus_product_l1236_123680


namespace quadratic_inequality_solution_set_l1236_123665

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x < -1 ∨ x > 3/2} := by sorry

end quadratic_inequality_solution_set_l1236_123665


namespace max_value_trig_expression_l1236_123611

theorem max_value_trig_expression :
  ∀ x y z : ℝ, 
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) * 
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 
  (9 : ℝ) / 2 := by
sorry

end max_value_trig_expression_l1236_123611


namespace olaf_initial_cars_l1236_123635

/-- The number of toy cars Olaf's uncle gave him -/
def uncle_cars : ℕ := 5

/-- The number of toy cars Olaf's grandpa gave him -/
def grandpa_cars : ℕ := 2 * uncle_cars

/-- The number of toy cars Olaf's dad gave him -/
def dad_cars : ℕ := 10

/-- The number of toy cars Olaf's mum gave him -/
def mum_cars : ℕ := dad_cars + 5

/-- The number of toy cars Olaf's auntie gave him -/
def auntie_cars : ℕ := 6

/-- The total number of toy cars Olaf has after receiving gifts -/
def total_cars : ℕ := 196

/-- The number of toy cars Olaf had initially -/
def initial_cars : ℕ := total_cars - (grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars)

theorem olaf_initial_cars : initial_cars = 150 := by
  sorry

end olaf_initial_cars_l1236_123635


namespace intersection_M_N_l1236_123642

def M : Set ℝ := {x | 4 < x ∧ x < 8}
def N : Set ℝ := {x | x^2 - 6*x < 0}

theorem intersection_M_N : M ∩ N = {x | 4 < x ∧ x < 6} := by sorry

end intersection_M_N_l1236_123642


namespace alice_purse_value_l1236_123641

-- Define the values of coins in cents
def penny : ℕ := 1
def dime : ℕ := 10
def quarter : ℕ := 25
def half_dollar : ℕ := 50

-- Define the total value of coins in Alice's purse
def purse_value : ℕ := penny + dime + quarter + half_dollar

-- Define one dollar in cents
def one_dollar : ℕ := 100

-- Theorem statement
theorem alice_purse_value :
  (purse_value : ℚ) / one_dollar = 86 / 100 := by sorry

end alice_purse_value_l1236_123641


namespace triangle_angle_C_l1236_123601

noncomputable def f (x φ : Real) : Real :=
  2 * Real.sin x * (Real.cos (φ / 2))^2 + Real.cos x * Real.sin φ - Real.sin x

theorem triangle_angle_C (φ A B C : Real) (a b c : Real) :
  0 < φ ∧ φ < Real.pi ∧
  (∀ x, f x φ ≥ f Real.pi φ) ∧
  Real.cos (2 * C) - Real.cos (2 * A) = 2 * Real.sin (Real.pi / 3 + C) * Real.sin (Real.pi / 3 - C) ∧
  a = 1 ∧
  b = Real.sqrt 2 ∧
  f A φ = Real.sqrt 3 / 2 ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C
  →
  C = 7 * Real.pi / 12 ∨ C = Real.pi / 12 :=
sorry

end triangle_angle_C_l1236_123601


namespace silverware_probability_l1236_123699

/-- The probability of selecting one fork, one spoon, and one knife when
    randomly removing three pieces of silverware from a drawer. -/
theorem silverware_probability (forks spoons knives : ℕ) 
  (h1 : forks = 6)
  (h2 : spoons = 8)
  (h3 : knives = 6) :
  (forks * spoons * knives : ℚ) / (Nat.choose (forks + spoons + knives) 3) = 24 / 95 :=
by sorry

end silverware_probability_l1236_123699


namespace hyperbola_iff_ab_neg_l1236_123648

/-- A curve in the xy-plane -/
structure Curve where
  equation : ℝ → ℝ → Prop

/-- Definition of a hyperbola -/
def is_hyperbola (c : Curve) : Prop := sorry

/-- The specific curve ax^2 + by^2 = 1 -/
def quadratic_curve (a b : ℝ) : Curve where
  equation := fun x y => a * x^2 + b * y^2 = 1

/-- Theorem stating that ab < 0 is both necessary and sufficient for the curve to be a hyperbola -/
theorem hyperbola_iff_ab_neg (a b : ℝ) :
  is_hyperbola (quadratic_curve a b) ↔ a * b < 0 := by sorry

end hyperbola_iff_ab_neg_l1236_123648


namespace shape_perimeter_l1236_123656

theorem shape_perimeter (total_area : ℝ) (num_squares : ℕ) (h1 : total_area = 196) (h2 : num_squares = 4) :
  let side_length := Real.sqrt (total_area / num_squares)
  let perimeter := (num_squares + 1) * side_length + 2 * num_squares * side_length
  perimeter = 91 := by
sorry

end shape_perimeter_l1236_123656


namespace muscovy_duck_count_muscovy_duck_count_proof_l1236_123697

theorem muscovy_duck_count : ℕ → ℕ → ℕ → Prop :=
  fun muscovy cayuga khaki =>
    muscovy = cayuga + 4 ∧
    muscovy = 2 * cayuga + khaki + 3 ∧
    muscovy + cayuga + khaki = 90 →
    muscovy = 89

-- The proof is omitted
theorem muscovy_duck_count_proof : muscovy_duck_count 89 85 6 :=
  sorry

end muscovy_duck_count_muscovy_duck_count_proof_l1236_123697


namespace quadratic_equation_solution_l1236_123632

theorem quadratic_equation_solution :
  ∃ x : ℝ, 4 * x^2 - 12 * x + 9 = 0 ∧ x = 3/2 := by sorry

end quadratic_equation_solution_l1236_123632


namespace loss_percentage_calculation_l1236_123629

theorem loss_percentage_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1500 → 
  selling_price = 1275 → 
  (cost_price - selling_price) / cost_price * 100 = 15 := by
sorry

end loss_percentage_calculation_l1236_123629


namespace function_intersects_x_axis_l1236_123627

/-- A function f(x) = kx² - 2x - 1 intersects the x-axis if and only if k ≥ -1 -/
theorem function_intersects_x_axis (k : ℝ) :
  (∃ x, k * x^2 - 2*x - 1 = 0) ↔ k ≥ -1 := by
sorry

end function_intersects_x_axis_l1236_123627


namespace correct_ages_l1236_123689

/-- Represents the ages of family members -/
structure FamilyAges where
  man : ℕ
  son : ℕ
  sibling : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  (ages.man = ages.son + 30) ∧
  (ages.man + 2 = 2 * (ages.son + 2)) ∧
  (ages.sibling + 2 = (ages.son + 2) / 2)

/-- Theorem stating that the ages 58, 28, and 13 satisfy the conditions -/
theorem correct_ages : 
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ ages.son = 28 ∧ ages.sibling = 13 :=
by
  sorry


end correct_ages_l1236_123689


namespace function_inequality_and_sum_inequality_l1236_123673

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x + 2|

-- Define the theorem
theorem function_inequality_and_sum_inequality :
  (∀ x m : ℝ, f x ≥ |m + 1|) →
  (∃ M : ℝ, M = 4 ∧
    (∀ m : ℝ, (∀ x : ℝ, f x ≥ |m + 1|) → m ≤ M) ∧
    (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + 2*b + c = M →
      1 / (a + b) + 1 / (b + c) ≥ 1)) :=
by sorry

end function_inequality_and_sum_inequality_l1236_123673


namespace power_of_seven_mod_ten_thousand_l1236_123690

theorem power_of_seven_mod_ten_thousand :
  7^2045 % 10000 = 6807 := by
  sorry

end power_of_seven_mod_ten_thousand_l1236_123690


namespace kangaroo_jump_theorem_l1236_123607

theorem kangaroo_jump_theorem :
  ∃ (a b c d : ℕ),
    a + b + c + d = 30 ∧
    7 * a + 5 * b + 3 * c - 3 * d = 200 ∧
    (a = 25 ∧ c = 5 ∧ b = 0 ∧ d = 0) ∨
    (a = 26 ∧ b = 3 ∧ c = 1 ∧ d = 0) ∨
    (a = 27 ∧ b = 1 ∧ c = 2 ∧ d = 0) ∨
    (a = 29 ∧ d = 1 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end kangaroo_jump_theorem_l1236_123607


namespace gemstones_for_four_sets_l1236_123670

/-- The number of gemstones needed for a given number of earring sets -/
def gemstones_needed (num_sets : ℕ) : ℕ :=
  let magnets_per_earring : ℕ := 2
  let buttons_per_earring : ℕ := magnets_per_earring / 2
  let gemstones_per_earring : ℕ := buttons_per_earring * 3
  let earrings_per_set : ℕ := 2
  num_sets * earrings_per_set * gemstones_per_earring

/-- Theorem: 4 sets of earrings require 24 gemstones -/
theorem gemstones_for_four_sets : gemstones_needed 4 = 24 := by
  sorry

end gemstones_for_four_sets_l1236_123670


namespace spatial_relationships_l1236_123695

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Define the perpendicular relationship between lines
variable (perpendicular_line : Line → Line → Prop)

theorem spatial_relationships 
  (m n : Line) (α β γ : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (perpendicular m α ∧ parallel_line_plane n α → perpendicular_line m n) ∧
  (parallel_plane α β ∧ parallel_plane β γ ∧ perpendicular m α → perpendicular m γ) :=
sorry

end spatial_relationships_l1236_123695


namespace rightmost_three_digits_of_5_pow_1993_l1236_123687

/-- The rightmost three digits of 5^1993 are 125 -/
theorem rightmost_three_digits_of_5_pow_1993 : 5^1993 % 1000 = 125 := by
  sorry

end rightmost_three_digits_of_5_pow_1993_l1236_123687


namespace furniture_markup_l1236_123606

/-- Given a selling price and a cost price, calculate the percentage markup -/
def percentageMarkup (sellingPrice costPrice : ℕ) : ℚ :=
  ((sellingPrice - costPrice : ℚ) / costPrice) * 100

theorem furniture_markup :
  percentageMarkup 5750 5000 = 15 := by sorry

end furniture_markup_l1236_123606


namespace simplify_expression_l1236_123668

theorem simplify_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b < 0) (hc : c < 0) 
  (hab : abs a > abs b) (hca : abs c > abs a) : 
  abs (a + c) - abs (b + c) - abs (a + b) = -2 * a := by
  sorry

end simplify_expression_l1236_123668


namespace quadratic_inequality_l1236_123617

theorem quadratic_inequality (x : ℝ) : x^2 + 5*x + 6 > 0 ↔ x < -3 ∨ x > -2 := by
  sorry

end quadratic_inequality_l1236_123617


namespace polygon_diagonals_integer_l1236_123685

theorem polygon_diagonals_integer (n : ℕ) (h : n > 0) : ∃ k : ℤ, (n * (n - 3) : ℤ) / 2 = k := by
  sorry

end polygon_diagonals_integer_l1236_123685


namespace number_of_gigs_played_l1236_123637

-- Define the earnings for each band member
def lead_singer_earnings : ℕ := 30
def guitarist_earnings : ℕ := 25
def bassist_earnings : ℕ := 20
def drummer_earnings : ℕ := 25
def keyboardist_earnings : ℕ := 20
def backup_singer_earnings : ℕ := 15

-- Define the total earnings per gig
def earnings_per_gig : ℕ := lead_singer_earnings + guitarist_earnings + bassist_earnings + 
                            drummer_earnings + keyboardist_earnings + backup_singer_earnings

-- Define the total earnings from all gigs
def total_earnings : ℕ := 2055

-- Theorem: The number of gigs played is 15
theorem number_of_gigs_played : 
  ⌊(total_earnings : ℚ) / (earnings_per_gig : ℚ)⌋ = 15 := by sorry

end number_of_gigs_played_l1236_123637


namespace unique_two_digit_integer_l1236_123655

theorem unique_two_digit_integer (s : ℕ) : s ≥ 10 ∧ s < 100 ∧ (13 * s) % 100 = 42 ↔ s = 34 := by
  sorry

end unique_two_digit_integer_l1236_123655


namespace fair_coin_four_tosses_l1236_123615

/-- A fair coin is a coin with equal probability of landing on either side -/
def fairCoin (p : ℝ) : Prop := p = 1/2

/-- The probability of n consecutive tosses landing on the same side -/
def consecutiveSameSide (p : ℝ) (n : ℕ) : ℝ := p^(n-1)

/-- Theorem: The probability of a fair coin landing on the same side 4 times in a row is 1/8 -/
theorem fair_coin_four_tosses (p : ℝ) (h : fairCoin p) : consecutiveSameSide p 4 = 1/8 := by
  sorry


end fair_coin_four_tosses_l1236_123615


namespace workshop_inspection_problem_l1236_123618

-- Define the number of products produced each day
variable (n : ℕ)

-- Define the probability of passing inspection on the first day
def prob_pass_first_day : ℚ := 3/5

-- Define the probability of passing inspection on the second day
def prob_pass_second_day (n : ℕ) : ℚ := (n - 2).choose 4 / n.choose 4

-- Define the probability of passing inspection on both days
def prob_pass_both_days (n : ℕ) : ℚ := prob_pass_first_day * prob_pass_second_day n

-- Define the probability of passing inspection on at least one day
def prob_pass_at_least_one_day (n : ℕ) : ℚ := 1 - (1 - prob_pass_first_day) * (1 - prob_pass_second_day n)

-- Theorem statement
theorem workshop_inspection_problem (n : ℕ) :
  (prob_pass_first_day = (n - 1).choose 4 / n.choose 4) →
  (n = 10) ∧
  (prob_pass_both_days n = 1/5) ∧
  (prob_pass_at_least_one_day n = 11/15) := by
  sorry


end workshop_inspection_problem_l1236_123618


namespace carries_money_from_mom_l1236_123660

def sweater_cost : ℕ := 24
def tshirt_cost : ℕ := 6
def shoes_cost : ℕ := 11
def money_left : ℕ := 50

theorem carries_money_from_mom : 
  sweater_cost + tshirt_cost + shoes_cost + money_left = 91 := by
  sorry

end carries_money_from_mom_l1236_123660


namespace systematic_sampling_probability_l1236_123643

/-- Represents a batch of parts with different classes -/
structure Batch :=
  (total : ℕ)
  (first_class : ℕ)
  (second_class : ℕ)
  (third_class : ℕ)

/-- Represents a sampling process -/
structure Sampling :=
  (batch : Batch)
  (sample_size : ℕ)

/-- The probability of selecting an individual part in systematic sampling -/
def selection_probability (s : Sampling) : ℚ :=
  s.sample_size / s.batch.total

/-- Theorem stating the probability of selecting each part in the given scenario -/
theorem systematic_sampling_probability (b : Batch) (s : Sampling) :
  b.total = 120 →
  b.first_class = 24 →
  b.second_class = 36 →
  b.third_class = 60 →
  s.batch = b →
  s.sample_size = 20 →
  selection_probability s = 1 / 6 := by
  sorry


end systematic_sampling_probability_l1236_123643


namespace complex_product_equals_33_l1236_123649

theorem complex_product_equals_33 (x : ℂ) (h : x = Complex.exp (2 * π * I / 9)) :
  (2 * x + x^2) * (2 * x^2 + x^4) * (2 * x^3 + x^6) * (2 * x^4 + x^8) = 33 := by
  sorry

end complex_product_equals_33_l1236_123649


namespace consecutive_integer_average_l1236_123696

theorem consecutive_integer_average (c d : ℤ) : 
  (∀ i : ℕ, i < 7 → c + i > 0) →
  d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 : ℚ) = c + 6 := by
  sorry

end consecutive_integer_average_l1236_123696


namespace cookies_per_bag_l1236_123658

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : total_cookies = 75)
  (h2 : num_bags = 25)
  (h3 : total_cookies = num_bags * cookies_per_bag) :
  cookies_per_bag = 3 := by
  sorry

end cookies_per_bag_l1236_123658


namespace complex_number_coordinate_l1236_123679

theorem complex_number_coordinate : 
  let z : ℂ := 1 + (1 / Complex.I)
  (z.re = 1 ∧ z.im = -1) := by sorry

end complex_number_coordinate_l1236_123679


namespace class_average_weight_l1236_123652

theorem class_average_weight (n₁ : ℕ) (n₂ : ℕ) (w₁ : ℝ) (w_total : ℝ) :
  n₁ = 16 →
  n₂ = 8 →
  w₁ = 50.25 →
  w_total = 48.55 →
  ((n₁ * w₁ + n₂ * ((n₁ + n₂) * w_total - n₁ * w₁) / n₂) / (n₁ + n₂) = w_total) →
  ((n₁ + n₂) * w_total - n₁ * w₁) / n₂ = 45.15 :=
by sorry

end class_average_weight_l1236_123652


namespace expected_digits_icosahedral_die_l1236_123682

def icosahedralDie : Finset ℕ := Finset.range 20

theorem expected_digits_icosahedral_die :
  let E := (icosahedralDie.filter (λ n => n < 10)).card / 20 +
           2 * (icosahedralDie.filter (λ n => n ≥ 10)).card / 20
  E = 31 / 20 := by sorry

end expected_digits_icosahedral_die_l1236_123682


namespace problem_solution_l1236_123645

theorem problem_solution (a b c x : ℝ) 
  (h1 : a + x^2 = 2015)
  (h2 : b + x^2 = 2016)
  (h3 : c + x^2 = 2017)
  (h4 : a * b * c = 24) :
  a / (b * c) + b / (a * c) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1 / 8 := by
  sorry

end problem_solution_l1236_123645
