import Mathlib

namespace NUMINAMATH_CALUDE_mouse_seeds_l607_60750

theorem mouse_seeds (mouse_seeds_per_burrow rabbit_seeds_per_burrow : ℕ)
  (mouse_burrows rabbit_burrows : ℕ) :
  mouse_seeds_per_burrow = 4 →
  rabbit_seeds_per_burrow = 6 →
  mouse_seeds_per_burrow * mouse_burrows = rabbit_seeds_per_burrow * rabbit_burrows →
  mouse_burrows = rabbit_burrows + 2 →
  mouse_seeds_per_burrow * mouse_burrows = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_mouse_seeds_l607_60750


namespace NUMINAMATH_CALUDE_mark_cereal_boxes_l607_60772

def soup_cost : ℕ := 6 * 2
def bread_cost : ℕ := 2 * 5
def milk_cost : ℕ := 2 * 4
def cereal_cost : ℕ := 3
def total_payment : ℕ := 4 * 10

def cereal_boxes : ℕ := (total_payment - (soup_cost + bread_cost + milk_cost)) / cereal_cost

theorem mark_cereal_boxes : cereal_boxes = 3 := by
  sorry

end NUMINAMATH_CALUDE_mark_cereal_boxes_l607_60772


namespace NUMINAMATH_CALUDE_clay_cost_calculation_l607_60795

/-- The price of clay in won per gram -/
def clay_price : ℝ := 17.25

/-- The weight of the first clay piece in grams -/
def clay_weight_1 : ℝ := 1000

/-- The weight of the second clay piece in grams -/
def clay_weight_2 : ℝ := 10

/-- The total cost of clay for Seungjun -/
def total_cost : ℝ := clay_price * (clay_weight_1 + clay_weight_2)

theorem clay_cost_calculation :
  total_cost = 17422.5 := by sorry

end NUMINAMATH_CALUDE_clay_cost_calculation_l607_60795


namespace NUMINAMATH_CALUDE_jerry_reaches_first_l607_60739

-- Define the points
variable (A B C D : Point)

-- Define the distances
variable (AB BD AC CD : ℝ)

-- Define the speeds
variable (speed_tom speed_jerry : ℝ)

-- Define the delay
variable (delay : ℝ)

-- Theorem statement
theorem jerry_reaches_first (h1 : AB = 32) (h2 : BD = 12) (h3 : AC = 13) (h4 : CD = 27)
  (h5 : speed_tom = 5) (h6 : speed_jerry = 4) (h7 : delay = 5) :
  (AB + BD) / speed_jerry < delay + (AC + CD) / speed_tom := by
  sorry

end NUMINAMATH_CALUDE_jerry_reaches_first_l607_60739


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l607_60745

theorem other_solution_quadratic (x : ℚ) :
  56 * (5/7)^2 + 27 = 89 * (5/7) - 8 →
  56 * (7/8)^2 + 27 = 89 * (7/8) - 8 :=
by sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l607_60745


namespace NUMINAMATH_CALUDE_acute_triangle_theorem_l607_60779

theorem acute_triangle_theorem (A B C : Real) (a b c : Real) :
  0 < A → A < π / 2 →
  0 < B → B < π / 2 →
  0 < C → C < π / 2 →
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  Real.sqrt 3 * b * Real.sin A - a * Real.cos B - a = 0 →
  (B = π / 3) ∧ 
  (3 * Real.sqrt 3 / 2 < Real.sin A + Real.sin C ∧ Real.sin A + Real.sin C ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_theorem_l607_60779


namespace NUMINAMATH_CALUDE_min_value_expression_l607_60705

theorem min_value_expression (x : ℝ) : 
  (15 - x) * (14 - x) * (15 + x) * (14 + x) ≥ -142.25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l607_60705


namespace NUMINAMATH_CALUDE_min_cos_plus_sin_l607_60798

theorem min_cos_plus_sin (A : Real) :
  let f := λ A : Real => Real.cos (A / 2) + Real.sin (A / 2)
  ∃ (min_value : Real), 
    (∀ A, f A ≥ min_value) ∧ 
    (min_value = -Real.sqrt 2) ∧
    (f (π / 2) = min_value) :=
by sorry

end NUMINAMATH_CALUDE_min_cos_plus_sin_l607_60798


namespace NUMINAMATH_CALUDE_flag_arrangement_theorem_l607_60748

/-- The number of distinguishable flagpoles -/
def num_poles : ℕ := 2

/-- The total number of flags -/
def total_flags : ℕ := 25

/-- The number of blue flags -/
def blue_flags : ℕ := 15

/-- The number of green flags -/
def green_flags : ℕ := 10

/-- Function to calculate the number of distinguishable arrangements -/
def calculate_arrangements (np gf bf : ℕ) : ℕ := sorry

/-- Theorem stating that the number of distinguishable arrangements,
    when divided by 1000, yields a remainder of 122 -/
theorem flag_arrangement_theorem :
  calculate_arrangements num_poles green_flags blue_flags % 1000 = 122 := by sorry

end NUMINAMATH_CALUDE_flag_arrangement_theorem_l607_60748


namespace NUMINAMATH_CALUDE_infinite_points_on_line_l607_60703

/-- A point on the line x + y = 4 with positive rational coordinates -/
structure PointOnLine where
  x : ℚ
  y : ℚ
  x_pos : 0 < x
  y_pos : 0 < y
  on_line : x + y = 4

/-- The set of all points on the line x + y = 4 with positive rational coordinates -/
def PointsOnLine : Set PointOnLine :=
  {p : PointOnLine | True}

/-- Theorem: There are infinitely many points on the line x + y = 4 with positive rational coordinates -/
theorem infinite_points_on_line : Set.Infinite PointsOnLine := by
  sorry

end NUMINAMATH_CALUDE_infinite_points_on_line_l607_60703


namespace NUMINAMATH_CALUDE_two_year_increase_l607_60702

/-- Calculates the final value after two years of percentage increases -/
def final_value (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial * (1 + rate1) * (1 + rate2)

/-- Theorem stating the final value after two years of specific increases -/
theorem two_year_increase (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) 
  (h1 : initial = 65000)
  (h2 : rate1 = 0.12)
  (h3 : rate2 = 0.08) :
  final_value initial rate1 rate2 = 78624 := by
  sorry

#eval final_value 65000 0.12 0.08

end NUMINAMATH_CALUDE_two_year_increase_l607_60702


namespace NUMINAMATH_CALUDE_complex_minimum_value_l607_60761

theorem complex_minimum_value (w : ℂ) (h : Complex.abs (w - (3 - 3•I)) = 4) :
  Complex.abs (w + (2 - I))^2 + Complex.abs (w - (7 - 2•I))^2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_complex_minimum_value_l607_60761


namespace NUMINAMATH_CALUDE_student_calculation_error_l607_60765

theorem student_calculation_error (N : ℚ) : 
  (N / (4/5)) = ((4/5) * N + 45) → N = 100 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_error_l607_60765


namespace NUMINAMATH_CALUDE_females_watch_count_l607_60774

/-- The number of people who watch WXLT -/
def total_watch : ℕ := 160

/-- The number of males who watch WXLT -/
def males_watch : ℕ := 85

/-- The number of females who don't watch WXLT -/
def females_dont_watch : ℕ := 120

/-- The total number of people who don't watch WXLT -/
def total_dont_watch : ℕ := 180

/-- The number of females who watch WXLT -/
def females_watch : ℕ := total_watch - males_watch

theorem females_watch_count : females_watch = 75 := by
  sorry

end NUMINAMATH_CALUDE_females_watch_count_l607_60774


namespace NUMINAMATH_CALUDE_coffee_price_correct_l607_60714

/-- The price of a cup of coffee satisfying the given conditions -/
def coffee_price : ℝ := 6

/-- The price of a piece of cheesecake -/
def cheesecake_price : ℝ := 10

/-- The discount rate applied to the set of coffee and cheesecake -/
def discount_rate : ℝ := 0.25

/-- The final price of the set (coffee + cheesecake) with discount -/
def discounted_set_price : ℝ := 12

/-- Theorem stating that the coffee price satisfies the given conditions -/
theorem coffee_price_correct :
  (1 - discount_rate) * (coffee_price + cheesecake_price) = discounted_set_price := by
  sorry

end NUMINAMATH_CALUDE_coffee_price_correct_l607_60714


namespace NUMINAMATH_CALUDE_vector_relations_l607_60758

/-- Given plane vectors a, b, and c, prove parallel and perpendicular conditions. -/
theorem vector_relations (a b c : ℝ × ℝ) (t : ℝ) 
  (ha : a = (-2, 1)) 
  (hb : b = (4, 2)) 
  (hc : c = (2, t)) : 
  (∃ (k : ℝ), a = k • c → t = -1) ∧ 
  (b.1 * c.1 + b.2 * c.2 = 0 → t = -4) := by
  sorry


end NUMINAMATH_CALUDE_vector_relations_l607_60758


namespace NUMINAMATH_CALUDE_sum_of_odds_15_to_51_l607_60799

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ := 
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem sum_of_odds_15_to_51 : 
  arithmetic_sum 15 51 2 = 627 := by sorry

end NUMINAMATH_CALUDE_sum_of_odds_15_to_51_l607_60799


namespace NUMINAMATH_CALUDE_min_value_z_l607_60712

theorem min_value_z (x y : ℝ) : x^2 + 3*y^2 + 8*x - 6*y + 30 ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l607_60712


namespace NUMINAMATH_CALUDE_dress_cost_equals_total_savings_l607_60764

/-- Calculates the cost of the dress based on initial savings, weekly allowance, weekly spending, and waiting period. -/
def dress_cost (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) (waiting_weeks : ℕ) : ℕ :=
  initial_savings + (weekly_allowance - weekly_spending) * waiting_weeks

/-- Proves that the dress cost is equal to Vanessa's total savings after the waiting period. -/
theorem dress_cost_equals_total_savings :
  dress_cost 20 30 10 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_dress_cost_equals_total_savings_l607_60764


namespace NUMINAMATH_CALUDE_unique_functional_equation_solution_l607_60784

theorem unique_functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x y : ℝ, f (x + y) = x * f x + y * f y) →
  (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_solution_l607_60784


namespace NUMINAMATH_CALUDE_complex_equation_solution_l607_60717

theorem complex_equation_solution (z : ℂ) : 
  (1 + Complex.I)^2 * z = 3 + 2 * Complex.I → z = 1 - (3/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l607_60717


namespace NUMINAMATH_CALUDE_fireworks_display_count_l607_60704

/-- The number of fireworks needed to display a single number. -/
def fireworks_per_number : ℕ := 6

/-- The number of fireworks needed to display a single letter. -/
def fireworks_per_letter : ℕ := 5

/-- The number of digits in the year display. -/
def year_digits : ℕ := 4

/-- The number of letters in "HAPPY NEW YEAR". -/
def phrase_letters : ℕ := 12

/-- The number of additional boxes of fireworks. -/
def additional_boxes : ℕ := 50

/-- The number of fireworks in each additional box. -/
def fireworks_per_box : ℕ := 8

/-- The total number of fireworks lit during the display. -/
def total_fireworks : ℕ := 
  year_digits * fireworks_per_number + 
  phrase_letters * fireworks_per_letter + 
  additional_boxes * fireworks_per_box

theorem fireworks_display_count : total_fireworks = 484 := by
  sorry

end NUMINAMATH_CALUDE_fireworks_display_count_l607_60704


namespace NUMINAMATH_CALUDE_pens_sold_in_garage_sale_l607_60721

/-- Given that Paul initially had 42 pens and after a garage sale had 19 pens left,
    prove that he sold 23 pens in the garage sale. -/
theorem pens_sold_in_garage_sale :
  let initial_pens : ℕ := 42
  let remaining_pens : ℕ := 19
  initial_pens - remaining_pens = 23 := by sorry

end NUMINAMATH_CALUDE_pens_sold_in_garage_sale_l607_60721


namespace NUMINAMATH_CALUDE_special_polynomial_q_count_l607_60794

/-- A polynomial of degree 4 with specific properties -/
structure SpecialPolynomial where
  o : ℤ
  p : ℤ
  q : ℤ
  roots_distinct : True  -- represents that the roots are distinct
  roots_positive : True  -- represents that the roots are positive
  one_integer_root : True  -- represents that exactly one root is an integer
  integer_root_sum : True  -- represents that the integer root is the sum of two other roots

/-- The number of possible values for q in the special polynomial -/
def count_q_values : ℕ := 1003001

/-- Theorem stating the number of possible q values -/
theorem special_polynomial_q_count :
  ∀ (poly : SpecialPolynomial), count_q_values = 1003001 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_q_count_l607_60794


namespace NUMINAMATH_CALUDE_subtraction_puzzle_sum_l607_60726

theorem subtraction_puzzle_sum :
  ∀ (P Q R S T : ℕ),
    P < 10 → Q < 10 → R < 10 → S < 10 → T < 10 →
    70000 + 1000 * Q + 200 + 10 * S + T - (10000 * P + 3000 + 100 * R + 90 + 6) = 22222 →
    P + Q + R + S + T = 29 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_puzzle_sum_l607_60726


namespace NUMINAMATH_CALUDE_units_digit_sum_l607_60787

theorem units_digit_sum (n m : ℕ) : (35^87 + 3^45) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_l607_60787


namespace NUMINAMATH_CALUDE_diagonals_of_adjacent_faces_perpendicular_l607_60710

/-- A cube is a three-dimensional shape with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- A face diagonal is a line segment connecting opposite corners of a face -/
structure FaceDiagonal (c : Cube) where
  -- We don't need to define the specifics of a face diagonal for this problem

/-- Two faces are adjacent if they share an edge -/
def adjacent_faces (c : Cube) (f1 f2 : FaceDiagonal c) : Prop :=
  sorry  -- Definition of adjacent faces

/-- The angle between two lines -/
def angle_between (l1 l2 : FaceDiagonal c) : ℝ :=
  sorry  -- Definition of angle between two lines

/-- Theorem: The angle between diagonals of adjacent faces of a cube is 90 degrees -/
theorem diagonals_of_adjacent_faces_perpendicular (c : Cube) (d1 d2 : FaceDiagonal c) 
  (h : adjacent_faces c d1 d2) : angle_between d1 d2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_of_adjacent_faces_perpendicular_l607_60710


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l607_60741

/-- Given a geometric sequence {a_n} with a_1 = 1/2 and a_4 = 4, 
    the common ratio q is equal to 2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q) 
  (h_a1 : a 1 = 1/2) 
  (h_a4 : a 4 = 4) :
  q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l607_60741


namespace NUMINAMATH_CALUDE_B_determines_xy_l607_60701

/-- Function B that determines x and y --/
def B (x y : ℕ) : ℕ := (x + y) * (x + y + 1) - y

/-- Theorem stating that B(x, y) uniquely determines x and y --/
theorem B_determines_xy (x y : ℕ) : 
  ∀ a b : ℕ, B x y = B a b → x = a ∧ y = b := by sorry

end NUMINAMATH_CALUDE_B_determines_xy_l607_60701


namespace NUMINAMATH_CALUDE_perfect_square_quotient_l607_60763

theorem perfect_square_quotient (a b : ℕ+) (h : (a * b + 1) ∣ (a ^ 2 + b ^ 2)) :
  ∃ k : ℕ, (a ^ 2 + b ^ 2) / (a * b + 1) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quotient_l607_60763


namespace NUMINAMATH_CALUDE_max_value_on_interval_max_value_attained_l607_60725

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_on_interval (x : ℝ) (h : x ∈ Set.Icc (-1) 1) : f x ≤ 2 := by
  sorry

theorem max_value_attained : ∃ x ∈ Set.Icc (-1) 1, f x = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_interval_max_value_attained_l607_60725


namespace NUMINAMATH_CALUDE_zach_lawn_mowing_pay_l607_60780

/-- Represents the financial situation for Zach's bike savings --/
structure BikeSavings where
  bikeCost : ℕ
  weeklyAllowance : ℕ
  currentSavings : ℕ
  babysittingPayRate : ℕ
  babysittingHours : ℕ
  additionalNeeded : ℕ

/-- Calculates the amount Zach's parent should pay him to mow the lawn --/
def lawnMowingPay (s : BikeSavings) : ℕ :=
  s.bikeCost - s.currentSavings - s.weeklyAllowance - s.babysittingPayRate * s.babysittingHours - s.additionalNeeded

/-- Theorem stating that the amount Zach's parent will pay him to mow the lawn is 10 --/
theorem zach_lawn_mowing_pay :
  let s : BikeSavings := {
    bikeCost := 100
    weeklyAllowance := 5
    currentSavings := 65
    babysittingPayRate := 7
    babysittingHours := 2
    additionalNeeded := 6
  }
  lawnMowingPay s = 10 := by sorry

end NUMINAMATH_CALUDE_zach_lawn_mowing_pay_l607_60780


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l607_60733

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := y - 2 = m * x + m

-- Theorem statement
theorem fixed_point_on_line :
  ∀ m : ℝ, line_equation (-1) 2 m :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l607_60733


namespace NUMINAMATH_CALUDE_solve_equation_l607_60727

theorem solve_equation : ∃ x : ℚ, 5 * (x - 4) = 3 * (6 - 3 * x) + 9 ∧ x = 47 / 14 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l607_60727


namespace NUMINAMATH_CALUDE_shopkeeper_articles_sold_l607_60723

/-- Proves that the number of articles sold is 30, given the selling price and profit conditions -/
theorem shopkeeper_articles_sold (C : ℝ) (C_pos : C > 0) : 
  ∃ N : ℕ, 
    (35 : ℝ) * C = (N : ℝ) * C + (1 / 6 : ℝ) * ((N : ℝ) * C) ∧ 
    N = 30 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_articles_sold_l607_60723


namespace NUMINAMATH_CALUDE_building_dimension_difference_l607_60771

-- Define the building structure
structure Building where
  floor1_length : ℝ
  floor1_width : ℝ
  floor2_length : ℝ
  floor2_width : ℝ

-- Define the conditions
def building_conditions (b : Building) : Prop :=
  b.floor1_width = (1/2) * b.floor1_length ∧
  b.floor1_length * b.floor1_width = 578 ∧
  b.floor2_width = (1/3) * b.floor2_length ∧
  b.floor2_length * b.floor2_width = 450

-- Define the combined length and width
def combined_length (b : Building) : ℝ := b.floor1_length + b.floor2_length
def combined_width (b : Building) : ℝ := b.floor1_width + b.floor2_width

-- Theorem statement
theorem building_dimension_difference (b : Building) 
  (h : building_conditions b) : 
  ∃ ε > 0, |combined_length b - combined_width b - 41.494| < ε :=
sorry

end NUMINAMATH_CALUDE_building_dimension_difference_l607_60771


namespace NUMINAMATH_CALUDE_salt_solution_dilution_l607_60788

theorem salt_solution_dilution (initial_volume : ℝ) (initial_salt_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 64 ∧ 
  initial_salt_percentage = 0.1 ∧ 
  added_water = 16 →
  let salt_amount := initial_volume * initial_salt_percentage
  let new_volume := initial_volume + added_water
  let final_salt_percentage := salt_amount / new_volume
  final_salt_percentage = 0.08 := by
sorry

end NUMINAMATH_CALUDE_salt_solution_dilution_l607_60788


namespace NUMINAMATH_CALUDE_square_properties_l607_60752

structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

def square : Square := {
  A := (0, 0),
  B := (-5, -3),
  C := (-4, -8),
  D := (1, -5)
}

theorem square_properties (s : Square) (h : s = square) :
  let side_length := Real.sqrt ((s.B.1 - s.A.1)^2 + (s.B.2 - s.A.2)^2)
  (side_length^2 = 34) ∧ (4 * side_length = 4 * Real.sqrt 34) := by
  sorry

#check square_properties

end NUMINAMATH_CALUDE_square_properties_l607_60752


namespace NUMINAMATH_CALUDE_jury_stabilization_jury_stabilization_30_l607_60767

/-- Represents a jury member -/
structure JuryMember where
  id : Nat

/-- Represents the state of the jury after a voting session -/
structure JuryState where
  members : List JuryMember
  sessionCount : Nat

/-- Represents a voting process -/
def votingProcess (state : JuryState) : JuryState :=
  sorry

/-- Theorem: For a jury with 2n members (n ≥ 2), the jury stabilizes after at most n sessions -/
theorem jury_stabilization (n : Nat) (h : n ≥ 2) :
  ∀ (initialState : JuryState),
    initialState.members.length = 2 * n →
    ∃ (finalState : JuryState),
      finalState = (votingProcess^[n]) initialState ∧
      finalState.members = ((votingProcess^[n + 1]) initialState).members :=
by
  sorry

/-- Corollary: A jury with 30 members stabilizes after at most 15 sessions -/
theorem jury_stabilization_30 :
  ∀ (initialState : JuryState),
    initialState.members.length = 30 →
    ∃ (finalState : JuryState),
      finalState = (votingProcess^[15]) initialState ∧
      finalState.members = ((votingProcess^[16]) initialState).members :=
by
  sorry

end NUMINAMATH_CALUDE_jury_stabilization_jury_stabilization_30_l607_60767


namespace NUMINAMATH_CALUDE_solution_to_equation_l607_60749

theorem solution_to_equation :
  ∃! (x y : ℝ), (x - y)^2 + (y - 2 * Real.sqrt x + 2)^2 = (1/2 : ℝ) ∧ x = 1 ∧ y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l607_60749


namespace NUMINAMATH_CALUDE_pipe_filling_time_l607_60766

theorem pipe_filling_time (pipe_a_rate pipe_b_rate total_time : ℚ) 
  (h1 : pipe_a_rate = 1 / 12)
  (h2 : pipe_b_rate = 1 / 20)
  (h3 : total_time = 10) :
  ∃ (x : ℚ), 
    x * (pipe_a_rate + pipe_b_rate) + (total_time - x) * pipe_b_rate = 1 ∧ 
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l607_60766


namespace NUMINAMATH_CALUDE_truck_distance_l607_60728

/-- Given a bike and a truck traveling for 8 hours, where the bike covers 136 miles
    and the truck's speed is 3 mph faster than the bike's, prove that the truck covers 160 miles. -/
theorem truck_distance (time : ℝ) (bike_distance : ℝ) (speed_difference : ℝ) :
  time = 8 ∧ bike_distance = 136 ∧ speed_difference = 3 →
  (bike_distance / time + speed_difference) * time = 160 :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_l607_60728


namespace NUMINAMATH_CALUDE_train_journey_l607_60762

theorem train_journey
  (average_speed : ℝ)
  (first_distance : ℝ)
  (first_time : ℝ)
  (second_time : ℝ)
  (h1 : average_speed = 70)
  (h2 : first_distance = 225)
  (h3 : first_time = 3.5)
  (h4 : second_time = 5)
  : (average_speed * (first_time + second_time) - first_distance) = 370 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_l607_60762


namespace NUMINAMATH_CALUDE_second_number_calculation_l607_60753

theorem second_number_calculation (A B : ℝ) : 
  A = 6400 → 
  0.05 * A = 0.20 * B + 190 → 
  B = 650 := by
sorry

end NUMINAMATH_CALUDE_second_number_calculation_l607_60753


namespace NUMINAMATH_CALUDE_total_peaches_l607_60783

theorem total_peaches (red : ℕ) (yellow : ℕ) (green : ℕ) 
  (h_red : red = 7) (h_yellow : yellow = 15) (h_green : green = 8) :
  red + yellow + green = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l607_60783


namespace NUMINAMATH_CALUDE_trig_problem_l607_60755

theorem trig_problem (α : Real) 
  (h1 : α ∈ Set.Ioo (5 * Real.pi / 4) (3 * Real.pi / 2))
  (h2 : Real.tan α + 1 / Real.tan α = 8) : 
  Real.sin α * Real.cos α = 1 / 8 ∧ Real.sin α - Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l607_60755


namespace NUMINAMATH_CALUDE_tracy_popped_half_balloons_l607_60782

theorem tracy_popped_half_balloons 
  (brooke_balloons : ℕ) 
  (tracy_initial_balloons : ℕ) 
  (total_balloons_after_popping : ℕ) 
  (tracy_popped_fraction : ℚ) :
  brooke_balloons = 20 →
  tracy_initial_balloons = 30 →
  total_balloons_after_popping = 35 →
  brooke_balloons + tracy_initial_balloons * (1 - tracy_popped_fraction) = total_balloons_after_popping →
  tracy_popped_fraction = 1/2 := by
sorry

end NUMINAMATH_CALUDE_tracy_popped_half_balloons_l607_60782


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l607_60775

theorem reciprocal_of_negative_two :
  ∃ x : ℚ, x * (-2) = 1 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l607_60775


namespace NUMINAMATH_CALUDE_third_circle_radius_l607_60719

/-- Given two internally tangent circles with radii R and r (R > r),
    the radius x of a third circle tangent to both circles and their common diameter
    is given by x = 4Rr / (R + r). -/
theorem third_circle_radius (R r : ℝ) (h : R > r) (h_pos_R : R > 0) (h_pos_r : r > 0) :
  ∃ x : ℝ, x > 0 ∧ x = (4 * R * r) / (R + r) ∧
    (∀ y : ℝ, y > 0 → y ≠ x →
      ¬(∃ p q : ℝ × ℝ,
        (p.1 - q.1)^2 + (p.2 - q.2)^2 = (R - r)^2 ∧
        (p.1 - 0)^2 + (p.2 - 0)^2 = R^2 ∧
        (q.1 - 0)^2 + (q.2 - 0)^2 = r^2 ∧
        ((p.1 + q.1)/2 - 0)^2 + ((p.2 + q.2)/2 - y)^2 = y^2)) :=
by sorry

end NUMINAMATH_CALUDE_third_circle_radius_l607_60719


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l607_60778

theorem expression_simplification_and_evaluation (m : ℤ) 
  (h1 : -2 ≤ m ∧ m ≤ 2) 
  (h2 : m ≠ -2 ∧ m ≠ 0 ∧ m ≠ 1 ∧ m ≠ 2) :
  (m / (m - 2) - 4 / (m^2 - 2*m)) / ((m + 2) / (m^2 - m)) = 
  ((m - 4) * (m + 1) * (m - 1)) / (m * (m - 2) * (m + 2)) ∧
  ((m - 4) * (m + 1) * (m - 1)) / (m * (m - 2) * (m + 2)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l607_60778


namespace NUMINAMATH_CALUDE_savings_calculation_l607_60740

/-- Given a person's income and expenditure ratio, and their income, calculate their savings -/
def calculate_savings (income_ratio : ℕ) (expenditure_ratio : ℕ) (income : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem stating that given the specific income-expenditure ratio and income, the savings are 3000 -/
theorem savings_calculation :
  let income_ratio : ℕ := 10
  let expenditure_ratio : ℕ := 7
  let income : ℕ := 10000
  calculate_savings income_ratio expenditure_ratio income = 3000 := by
  sorry

#eval calculate_savings 10 7 10000

end NUMINAMATH_CALUDE_savings_calculation_l607_60740


namespace NUMINAMATH_CALUDE_same_color_probability_l607_60734

def total_plates : ℕ := 13
def red_plates : ℕ := 7
def blue_plates : ℕ := 6
def plates_to_select : ℕ := 3

theorem same_color_probability :
  (Nat.choose red_plates plates_to_select + Nat.choose blue_plates plates_to_select) /
  Nat.choose total_plates plates_to_select = 55 / 286 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l607_60734


namespace NUMINAMATH_CALUDE_square_area_error_l607_60786

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l607_60786


namespace NUMINAMATH_CALUDE_second_day_speed_l607_60707

/-- Proves that given the climbing conditions, the speed on the second day is 4 km/h -/
theorem second_day_speed (total_time : ℝ) (speed_difference : ℝ) (time_difference : ℝ) (total_distance : ℝ)
  (h1 : total_time = 14)
  (h2 : speed_difference = 0.5)
  (h3 : time_difference = 2)
  (h4 : total_distance = 52) :
  let first_day_time := (total_time + time_difference) / 2
  let second_day_time := total_time - first_day_time
  let first_day_speed := (total_distance - speed_difference * second_day_time) / total_time
  let second_day_speed := first_day_speed + speed_difference
  second_day_speed = 4 := by sorry

end NUMINAMATH_CALUDE_second_day_speed_l607_60707


namespace NUMINAMATH_CALUDE_room_with_193_black_tiles_has_1089_total_tiles_l607_60777

/-- Represents a square room with tiled floor -/
structure TiledRoom where
  side_length : ℕ
  black_tile_count : ℕ

/-- Calculates the number of black tiles in a square room with given side length -/
def black_tiles (s : ℕ) : ℕ := 6 * s - 5

/-- Calculates the total number of tiles in a square room with given side length -/
def total_tiles (s : ℕ) : ℕ := s * s

/-- Theorem stating that a square room with 193 black tiles has 1089 total tiles -/
theorem room_with_193_black_tiles_has_1089_total_tiles :
  ∃ (room : TiledRoom), room.black_tile_count = 193 ∧ total_tiles room.side_length = 1089 :=
by sorry

end NUMINAMATH_CALUDE_room_with_193_black_tiles_has_1089_total_tiles_l607_60777


namespace NUMINAMATH_CALUDE_no_solution_to_system_l607_60743

theorem no_solution_to_system :
  ∀ x : ℝ, ¬(x^5 + 3*x^4 + 5*x^3 + 5*x^2 + 6*x + 2 = 0 ∧ x^3 + 3*x^2 + 4*x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l607_60743


namespace NUMINAMATH_CALUDE_smallest_number_minus_one_in_list_minus_one_is_smallest_l607_60757

def numbers : List ℚ := [3, 0, -1, -1/2]

theorem smallest_number (n : ℚ) (hn : n ∈ numbers) :
  -1 ≤ n := by sorry

theorem minus_one_in_list : -1 ∈ numbers := by sorry

theorem minus_one_is_smallest : ∀ n ∈ numbers, -1 ≤ n ∧ ∃ m ∈ numbers, -1 = m := by sorry

end NUMINAMATH_CALUDE_smallest_number_minus_one_in_list_minus_one_is_smallest_l607_60757


namespace NUMINAMATH_CALUDE_calculation_proof_l607_60713

theorem calculation_proof : 8 - (7.14 * (1/3) - 2 * (2/9) / 2.5) + 0.1 = 6.62 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l607_60713


namespace NUMINAMATH_CALUDE_max_sequence_length_l607_60768

theorem max_sequence_length (a : ℕ → ℤ) (n : ℕ) : 
  (∀ i : ℕ, i + 6 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) > 0)) →
  (∀ i : ℕ, i + 10 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) + a (i+7) + a (i+8) + a (i+9) + a (i+10) < 0)) →
  n ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_sequence_length_l607_60768


namespace NUMINAMATH_CALUDE_unique_value_at_three_l607_60735

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * g y + 2 * x) = 2 * x * y + g x

/-- The theorem stating that g(3) = 6 is the only possible value -/
theorem unique_value_at_three
  (g : ℝ → ℝ) (h : SatisfiesFunctionalEquation g) :
  g 3 = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_value_at_three_l607_60735


namespace NUMINAMATH_CALUDE_prime_power_divisors_l607_60792

theorem prime_power_divisors (p q : ℕ) (n : ℕ) : 
  Prime p → Prime q → (Nat.divisors (p^n * q^6)).card = 28 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_power_divisors_l607_60792


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l607_60711

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (5 * x + 11) = 14 → x = 37 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l607_60711


namespace NUMINAMATH_CALUDE_total_money_divided_l607_60744

/-- The total amount of money divided among A, B, and C is 120, given the specified conditions. -/
theorem total_money_divided (a b c : ℕ) : 
  b = 20 → a = b + 20 → c = a + 20 → a + b + c = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_money_divided_l607_60744


namespace NUMINAMATH_CALUDE_polynomial_degree_problem_l607_60790

theorem polynomial_degree_problem (m n : ℤ) : 
  (m + 1 + 2 = 6) →  -- Degree of the polynomial term x^(m+1)y^2 is 6
  (2*n + (5 - m) = 6) →  -- Degree of the monomial x^(2n)y^(5-m) is 6
  (-m)^3 + 2*n = -23 := by
sorry

end NUMINAMATH_CALUDE_polynomial_degree_problem_l607_60790


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l607_60724

theorem rectangular_prism_volume (x y z : ℝ) 
  (eq1 : 2*x + 2*y = 38)
  (eq2 : y + z = 14)
  (eq3 : x + z = 11) :
  x * y * z = 264 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l607_60724


namespace NUMINAMATH_CALUDE_donalds_oranges_l607_60751

theorem donalds_oranges (initial : ℕ) : initial + 5 = 9 → initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_donalds_oranges_l607_60751


namespace NUMINAMATH_CALUDE_carpet_shaded_area_carpet_specific_shaded_area_l607_60708

/-- Calculates the total shaded area of a rectangular carpet with specific dimensions and shaded areas. -/
theorem carpet_shaded_area (carpet_length carpet_width : ℝ) 
  (num_small_squares : ℕ) (ratio_long_to_R ratio_R_to_S : ℝ) : ℝ :=
  let R := carpet_length / ratio_long_to_R
  let S := R / ratio_R_to_S
  let area_R := R * R
  let area_S := S * S
  let total_area := area_R + (num_small_squares : ℝ) * area_S
  total_area

/-- Proves that the total shaded area of the carpet with given specifications is 141.75 square feet. -/
theorem carpet_specific_shaded_area : 
  carpet_shaded_area 18 12 12 2 4 = 141.75 := by
  sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_carpet_specific_shaded_area_l607_60708


namespace NUMINAMATH_CALUDE_estimate_cube_of_331_l607_60756

/-- Proves that (.331)^3 is approximately equal to 0.037, given that .331 is close to 1/3 -/
theorem estimate_cube_of_331 (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |0.331 - (1/3)| < δ → |0.331^3 - 0.037| < ε :=
sorry

end NUMINAMATH_CALUDE_estimate_cube_of_331_l607_60756


namespace NUMINAMATH_CALUDE_complement_of_union_is_five_l607_60720

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 3}

-- Define set B
def B : Set Nat := {2, 4}

-- Theorem statement
theorem complement_of_union_is_five :
  (U \ (A ∪ B)) = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_is_five_l607_60720


namespace NUMINAMATH_CALUDE_daltons_uncle_gift_l607_60770

/-- The amount of money Dalton's uncle gave him -/
def uncles_gift (jump_rope_cost board_game_cost ball_cost savings needed_more : ℕ) : ℕ :=
  (jump_rope_cost + board_game_cost + ball_cost) - savings - needed_more

/-- Proof that Dalton's uncle gave him $13 -/
theorem daltons_uncle_gift :
  uncles_gift 7 12 4 6 4 = 13 := by
  sorry

end NUMINAMATH_CALUDE_daltons_uncle_gift_l607_60770


namespace NUMINAMATH_CALUDE_sum_of_ages_l607_60709

theorem sum_of_ages (maria_age jose_age : ℕ) : 
  maria_age = 14 → 
  jose_age = maria_age + 12 → 
  maria_age + jose_age = 40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l607_60709


namespace NUMINAMATH_CALUDE_fractional_equation_solution_exists_l607_60797

theorem fractional_equation_solution_exists : ∃ m : ℝ, ∃ x : ℝ, x ≠ 1 ∧ (x + 2) / (x - 1) = m / (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_exists_l607_60797


namespace NUMINAMATH_CALUDE_handshake_count_is_correct_handshakes_per_person_is_correct_l607_60706

/-- Represents a social gathering with married couples -/
structure SocialGathering where
  couples : ℕ
  people : ℕ
  handshakes_per_person : ℕ

/-- Calculate the total number of unique handshakes in the gathering -/
def total_handshakes (g : SocialGathering) : ℕ :=
  g.people * g.handshakes_per_person / 2

/-- The specific social gathering described in the problem -/
def our_gathering : SocialGathering :=
  { couples := 8
  , people := 16
  , handshakes_per_person := 12 }

theorem handshake_count_is_correct :
  total_handshakes our_gathering = 96 := by
  sorry

/-- Prove that the number of handshakes per person is correct -/
theorem handshakes_per_person_is_correct (g : SocialGathering) :
  g.handshakes_per_person = g.people - 1 - 3 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_is_correct_handshakes_per_person_is_correct_l607_60706


namespace NUMINAMATH_CALUDE_fabric_cutting_l607_60742

theorem fabric_cutting (initial_length : ℚ) (cut_length : ℚ) (desired_length : ℚ) :
  initial_length = 2/3 →
  cut_length = 1/6 →
  desired_length = 1/2 →
  initial_length - cut_length = desired_length :=
by sorry

end NUMINAMATH_CALUDE_fabric_cutting_l607_60742


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fraction_zero_l607_60746

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem arithmetic_sequence_fraction_zero 
  (a₁ d : ℚ) (h₁ : a₁ ≠ 0) (h₂ : arithmetic_sequence a₁ d 9 = 0) :
  (arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 8 + 
   arithmetic_sequence a₁ d 11 + arithmetic_sequence a₁ d 16) / 
  (arithmetic_sequence a₁ d 7 + arithmetic_sequence a₁ d 8 + 
   arithmetic_sequence a₁ d 14) = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fraction_zero_l607_60746


namespace NUMINAMATH_CALUDE_notebooks_left_l607_60736

theorem notebooks_left (notebooks_per_bundle : ℕ) (num_bundles : ℕ) (num_groups : ℕ) (students_per_group : ℕ) : 
  notebooks_per_bundle = 25 →
  num_bundles = 5 →
  num_groups = 8 →
  students_per_group = 13 →
  num_bundles * notebooks_per_bundle - num_groups * students_per_group = 21 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_left_l607_60736


namespace NUMINAMATH_CALUDE_no_equilateral_right_triangle_l607_60730

theorem no_equilateral_right_triangle :
  ¬ ∃ (a b c : ℝ) (A B C : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
    A > 0 ∧ B > 0 ∧ C > 0 ∧  -- Positive angles
    a = b ∧ b = c ∧          -- Equilateral condition
    A = 90 ∧                 -- Right angle condition
    A + B + C = 180          -- Sum of angles in a triangle
    := by sorry

end NUMINAMATH_CALUDE_no_equilateral_right_triangle_l607_60730


namespace NUMINAMATH_CALUDE_sod_area_calculation_sod_area_is_9474_l607_60776

/-- Calculates the area of sod needed for Jill's front yard -/
theorem sod_area_calculation (front_yard_width front_yard_length sidewalk_width sidewalk_length
                              flowerbed1_depth flowerbed1_length flowerbed2_width flowerbed2_length
                              flowerbed3_width flowerbed3_length : ℕ) : ℕ :=
  let front_yard_area := front_yard_width * front_yard_length
  let sidewalk_area := sidewalk_width * sidewalk_length
  let flowerbed1_area := 2 * (flowerbed1_depth * flowerbed1_length)
  let flowerbed2_area := flowerbed2_width * flowerbed2_length
  let flowerbed3_area := flowerbed3_width * flowerbed3_length
  let total_subtract_area := sidewalk_area + flowerbed1_area + flowerbed2_area + flowerbed3_area
  front_yard_area - total_subtract_area

/-- Proves that the area of sod needed for Jill's front yard is 9,474 square feet -/
theorem sod_area_is_9474 :
  sod_area_calculation 200 50 3 50 4 25 10 12 7 8 = 9474 := by
  sorry

end NUMINAMATH_CALUDE_sod_area_calculation_sod_area_is_9474_l607_60776


namespace NUMINAMATH_CALUDE_storks_joined_l607_60715

theorem storks_joined (initial_birds initial_storks final_difference : ℕ) :
  initial_birds = 4 →
  initial_storks = 3 →
  final_difference = 5 →
  ∃ joined : ℕ, initial_storks + joined = initial_birds + final_difference ∧ joined = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_joined_l607_60715


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l607_60732

/-- An isosceles triangle with side lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 7 ∧ c = 7 →  -- Two sides are 7, one side is 3
  a + b + c = 17 :=        -- The perimeter is 17
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l607_60732


namespace NUMINAMATH_CALUDE_power_of_product_l607_60737

theorem power_of_product (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l607_60737


namespace NUMINAMATH_CALUDE_gcd_g_x_eq_six_l607_60759

def g (x : ℤ) : ℤ := (5*x+3)*(8*x+2)*(11*x+7)*(3*x+5)

theorem gcd_g_x_eq_six (x : ℤ) (h : 18432 ∣ x) : 
  Nat.gcd (g x).natAbs x.natAbs = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_eq_six_l607_60759


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l607_60716

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i) / i
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l607_60716


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l607_60791

theorem similar_triangles_leg_sum 
  (A₁ A₂ : ℝ) 
  (h_areas : A₁ = 12 ∧ A₂ = 192) 
  (h_similar : ∃ (k : ℝ), k > 0 ∧ A₂ = k^2 * A₁) 
  (a b : ℝ) 
  (h_right : a^2 + b^2 = 10^2) 
  (h_leg_ratio : a = 2*b) 
  (h_area_small : A₁ = 1/2 * a * b) : 
  ∃ (c d : ℝ), c^2 + d^2 = (4*10)^2 ∧ A₂ = 1/2 * c * d ∧ c + d = 24 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l607_60791


namespace NUMINAMATH_CALUDE_max_red_balls_l607_60718

theorem max_red_balls 
  (total : ℕ) 
  (green : ℕ) 
  (h1 : total = 28) 
  (h2 : green = 12) 
  (h3 : ∀ red : ℕ, red + green < 24) : 
  ∃ max_red : ℕ, max_red = 11 ∧ ∀ red : ℕ, red ≤ max_red := by
sorry

end NUMINAMATH_CALUDE_max_red_balls_l607_60718


namespace NUMINAMATH_CALUDE_division_of_decimals_l607_60781

theorem division_of_decimals : (0.05 : ℝ) / 0.01 = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l607_60781


namespace NUMINAMATH_CALUDE_total_amount_not_unique_l607_60731

/-- Represents the investment scenario with two different interest rates -/
structure Investment where
  x : ℝ  -- Amount invested at 10%
  y : ℝ  -- Amount invested at 8%
  T : ℝ  -- Total amount invested

/-- The conditions of the investment problem -/
def investment_conditions (inv : Investment) : Prop :=
  inv.x * 0.10 - inv.y * 0.08 = 65 ∧ inv.x + inv.y = inv.T

/-- Theorem stating that the total amount T cannot be uniquely determined -/
theorem total_amount_not_unique :
  ∃ (inv1 inv2 : Investment), 
    investment_conditions inv1 ∧ 
    investment_conditions inv2 ∧ 
    inv1.T ≠ inv2.T :=
sorry

#check total_amount_not_unique

end NUMINAMATH_CALUDE_total_amount_not_unique_l607_60731


namespace NUMINAMATH_CALUDE_row_5_seat_4_denotation_l607_60747

/-- Represents a seat in a theater -/
structure Seat where
  row : ℕ
  number : ℕ

/-- Converts a seat to its denotation as an ordered pair -/
def seat_denotation (s : Seat) : ℕ × ℕ := (s.row, s.number)

/-- Given condition: "Row 4, Seat 5" is denoted as (4, 5) -/
axiom example_seat : seat_denotation ⟨4, 5⟩ = (4, 5)

/-- Theorem: The denotation of "Row 5, Seat 4" is (5, 4) -/
theorem row_5_seat_4_denotation : seat_denotation ⟨5, 4⟩ = (5, 4) := by
  sorry

end NUMINAMATH_CALUDE_row_5_seat_4_denotation_l607_60747


namespace NUMINAMATH_CALUDE_work_completion_time_l607_60754

theorem work_completion_time (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (1/a + 1/b = 1/4) → (1/b = 1/6) → (1/a = 1/12) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l607_60754


namespace NUMINAMATH_CALUDE_circle_center_polar_coordinates_l607_60785

/-- Given a circle with polar coordinate equation ρ = 2(cosθ + sinθ), 
    the polar coordinates of its center are (√2, π/4) -/
theorem circle_center_polar_coordinates :
  ∀ ρ θ : ℝ, 
  ρ = 2 * (Real.cos θ + Real.sin θ) →
  ∃ r α : ℝ, 
    r = Real.sqrt 2 ∧ 
    α = π / 4 ∧ 
    (r * Real.cos α - 1)^2 + (r * Real.sin α - 1)^2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_circle_center_polar_coordinates_l607_60785


namespace NUMINAMATH_CALUDE_inequality_solution_set_l607_60789

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x^2 - 2 ≥ 2*x - a*x}
  (a = 0 → S = {x : ℝ | x ≤ -1}) ∧
  (a > 0 → S = {x : ℝ | x ≥ 2/a ∨ x ≤ -1}) ∧
  (-2 < a ∧ a < 0 → S = {x : ℝ | 2/a ≤ x ∧ x ≤ -1}) ∧
  (a = -2 → S = {x : ℝ | x = -1}) ∧
  (a < -2 → S = {x : ℝ | -1 ≤ x ∧ x ≤ 2/a}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l607_60789


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l607_60729

theorem largest_angle_in_triangle (y : ℝ) : 
  y + 60 + 70 = 180 → 
  max y (max 60 70) = 70 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l607_60729


namespace NUMINAMATH_CALUDE_unique_monic_polynomial_l607_60793

/-- A monic polynomial of degree 3 satisfying specific conditions -/
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

/-- Theorem stating that f is the unique monic polynomial of degree 3 satisfying given conditions -/
theorem unique_monic_polynomial :
  (∀ x, f x = x^3 + 2*x^2 + 3*x + 4) ∧
  f 0 = 4 ∧ f 1 = 10 ∧ f (-1) = 2 ∧
  (∀ g : ℝ → ℝ, (∃ a b c : ℝ, ∀ x, g x = x^3 + a*x^2 + b*x + c) →
    g 0 = 4 → g 1 = 10 → g (-1) = 2 → g = f) :=
by sorry

end NUMINAMATH_CALUDE_unique_monic_polynomial_l607_60793


namespace NUMINAMATH_CALUDE_milford_lake_algae_increase_l607_60738

/-- The increase in algae plants in Milford Lake -/
def algae_increase (original : ℕ) (current : ℕ) : ℕ :=
  current - original

/-- Theorem stating the increase in algae plants in Milford Lake -/
theorem milford_lake_algae_increase :
  algae_increase 809 3263 = 2454 := by
  sorry

end NUMINAMATH_CALUDE_milford_lake_algae_increase_l607_60738


namespace NUMINAMATH_CALUDE_f_symmetry_l607_60796

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l607_60796


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_l607_60722

/-- Represents the position of the cat -/
inductive CatPosition
  | TopLeft
  | TopRight
  | BottomRight
  | BottomLeft

/-- Represents the position of the mouse -/
inductive MousePosition
  | TopLeft
  | TopMiddle
  | TopRight
  | RightMiddle
  | BottomRight
  | BottomMiddle
  | BottomLeft
  | LeftMiddle

/-- The number of squares in the cat's cycle -/
def catCycleLength : Nat := 4

/-- The number of segments in the mouse's cycle -/
def mouseCycleLength : Nat := 8

/-- The total number of moves -/
def totalMoves : Nat := 317

/-- Function to determine the cat's position after a given number of moves -/
def catPositionAfterMoves (moves : Nat) : CatPosition :=
  match moves % catCycleLength with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | 3 => CatPosition.BottomRight
  | _ => CatPosition.TopLeft  -- This case should never occur due to the modulo operation

/-- Function to determine the mouse's position after a given number of moves -/
def mousePositionAfterMoves (moves : Nat) : MousePosition :=
  match moves % mouseCycleLength with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.LeftMiddle
  | 2 => MousePosition.BottomLeft
  | 3 => MousePosition.BottomMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.RightMiddle
  | 6 => MousePosition.TopRight
  | 7 => MousePosition.TopMiddle
  | _ => MousePosition.TopLeft  -- This case should never occur due to the modulo operation

theorem cat_and_mouse_positions :
  catPositionAfterMoves totalMoves = CatPosition.TopLeft ∧
  mousePositionAfterMoves totalMoves = MousePosition.BottomMiddle := by
  sorry

end NUMINAMATH_CALUDE_cat_and_mouse_positions_l607_60722


namespace NUMINAMATH_CALUDE_used_cd_price_l607_60769

theorem used_cd_price (n u : ℝ) 
  (eq1 : 6 * n + 2 * u = 127.92)
  (eq2 : 3 * n + 8 * u = 133.89) :
  u = 9.99 := by
sorry

end NUMINAMATH_CALUDE_used_cd_price_l607_60769


namespace NUMINAMATH_CALUDE_rhombus_sum_difference_l607_60700

theorem rhombus_sum_difference (a b c d : ℝ) (h : a + b + c + d = 2021) :
  ((a + 1) * (b + 1) + (b + 1) * (c + 1) + (c + 1) * (d + 1) + (d + 1) * (a + 1)) -
  (a * b + b * c + c * d + d * a) = 4046 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_sum_difference_l607_60700


namespace NUMINAMATH_CALUDE_box_filling_proof_l607_60773

theorem box_filling_proof (length width depth : ℕ) (num_cubes : ℕ) : 
  length = 49 → 
  width = 42 → 
  depth = 14 → 
  num_cubes = 84 → 
  ∃ (cube_side : ℕ), 
    cube_side > 0 ∧ 
    length % cube_side = 0 ∧ 
    width % cube_side = 0 ∧ 
    depth % cube_side = 0 ∧ 
    (length / cube_side) * (width / cube_side) * (depth / cube_side) = num_cubes :=
by
  sorry

#check box_filling_proof

end NUMINAMATH_CALUDE_box_filling_proof_l607_60773


namespace NUMINAMATH_CALUDE_five_dollar_four_equals_85_l607_60760

/-- Custom operation $\$$ defined as a $ b = a(2b + 1) + 2ab -/
def dollar_op (a b : ℕ) : ℕ := a * (2 * b + 1) + 2 * a * b

/-- Theorem stating that 5 $ 4 = 85 -/
theorem five_dollar_four_equals_85 : dollar_op 5 4 = 85 := by
  sorry

end NUMINAMATH_CALUDE_five_dollar_four_equals_85_l607_60760
