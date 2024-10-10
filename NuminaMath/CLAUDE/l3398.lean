import Mathlib

namespace largest_integer_times_eleven_less_than_150_l3398_339840

theorem largest_integer_times_eleven_less_than_150 :
  ∀ x : ℤ, x ≤ 13 ↔ 11 * x < 150 :=
by
  sorry

end largest_integer_times_eleven_less_than_150_l3398_339840


namespace square_number_ones_digit_l3398_339874

/-- A number is a square number if it's the square of an integer -/
def IsSquareNumber (a : ℕ) : Prop := ∃ x : ℕ, a = x^2

/-- Get the tens digit of a natural number -/
def TensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- Get the ones digit of a natural number -/
def OnesDigit (n : ℕ) : ℕ := n % 10

/-- A number is odd if it's not divisible by 2 -/
def IsOdd (n : ℕ) : Prop := n % 2 = 1

theorem square_number_ones_digit
  (a : ℕ)
  (h1 : IsSquareNumber a)
  (h2 : IsOdd (TensDigit a)) :
  OnesDigit a = 6 := by
  sorry

end square_number_ones_digit_l3398_339874


namespace villager_count_l3398_339832

theorem villager_count (milk bottles_of_milk : ℕ) (apples : ℕ) (bread : ℕ) 
  (milk_left : ℕ) (apples_left : ℕ) (bread_short : ℕ) :
  bottles_of_milk = 160 →
  apples = 197 →
  bread = 229 →
  milk_left = 4 →
  apples_left = 2 →
  bread_short = 5 →
  ∃ (villagers : ℕ),
    villagers > 0 ∧
    (bottles_of_milk - milk_left) % villagers = 0 ∧
    (apples - apples_left) % villagers = 0 ∧
    (bread + bread_short) % villagers = 0 ∧
    villagers = 39 := by
  sorry

end villager_count_l3398_339832


namespace problem_solution_l3398_339875

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
sorry

end problem_solution_l3398_339875


namespace vector_parallel_perpendicular_l3398_339813

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem vector_parallel_perpendicular (x : ℝ) :
  (∃ k : ℝ, a + 2 • b x = k • (2 • a - b x) → x = 1/2) ∧
  ((a + 2 • b x) • (2 • a - b x) = 0 → x = -2 ∨ x = 7/2) :=
sorry

end vector_parallel_perpendicular_l3398_339813


namespace point_C_coordinates_l3398_339893

def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (4, 9)

theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) →  -- C lies on segment AB
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 16 * ((B.1 - C.1)^2 + (B.2 - C.2)^2) →  -- AC = 4CB
  C = (8/5, 14/5) :=
by sorry

end point_C_coordinates_l3398_339893


namespace perfect_square_trinomial_l3398_339830

theorem perfect_square_trinomial (x : ℝ) : ∃ (a : ℝ), (x^2 + 4 + 4*x) = (x + a)^2 := by
  sorry

end perfect_square_trinomial_l3398_339830


namespace ellipse_line_intersection_range_l3398_339827

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope 1 and y-intercept m -/
structure Line where
  m : ℝ

/-- Represents the intersection of an ellipse and a line -/
def Intersection (e : Ellipse) (l : Line) :=
  {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1 ∧ p.2 = p.1 + l.m}

/-- Theorem stating the range of m for which the line intersects the ellipse at two distinct points forming an acute angle at the origin -/
theorem ellipse_line_intersection_range (e : Ellipse) (l : Line) 
  (h_minor : e.b = 1)
  (h_ecc : Real.sqrt (1 - e.b^2 / e.a^2) = Real.sqrt 3 / 2)
  (h_intersect : ∃ A B, A ∈ Intersection e l ∧ B ∈ Intersection e l ∧ A ≠ B)
  (h_acute : ∃ A B, A ∈ Intersection e l ∧ B ∈ Intersection e l ∧ A ≠ B ∧ 
    0 < Real.arccos ((A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2))) ∧
    Real.arccos ((A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2))) < π / 2) :
  (-Real.sqrt 5 < l.m ∧ l.m < -2 * Real.sqrt 10 / 5) ∨ (2 * Real.sqrt 10 / 5 < l.m ∧ l.m < Real.sqrt 5) := by
  sorry

end ellipse_line_intersection_range_l3398_339827


namespace smallest_price_with_tax_l3398_339887

theorem smallest_price_with_tax (n : ℕ) (x : ℕ) : n = 21 ↔ 
  n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬∃ y : ℕ, y > 0 ∧ 105 * y = 100 * m * 100) ∧
  x > 0 ∧ 
  105 * x = 100 * n * 100 :=
sorry

end smallest_price_with_tax_l3398_339887


namespace boys_in_class_l3398_339854

theorem boys_in_class (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 345 → diff = 69 → boys + (boys + diff) = total → boys = 138 := by
  sorry

end boys_in_class_l3398_339854


namespace find_a_l3398_339816

-- Define the solution set
def solutionSet (x : ℝ) : Prop :=
  (-3 < x ∧ x < -1) ∨ x > 2

-- Define the inequality
def inequality (a x : ℝ) : Prop :=
  (x + a) / (x^2 + 4*x + 3) > 0

theorem find_a :
  (∃ a : ℝ, ∀ x : ℝ, inequality a x ↔ solutionSet x) →
  (∃ a : ℝ, a = -2 ∧ ∀ x : ℝ, inequality a x ↔ solutionSet x) :=
by sorry

end find_a_l3398_339816


namespace triangle_perimeter_not_48_l3398_339804

theorem triangle_perimeter_not_48 (a b c : ℝ) : 
  a = 25 → b = 12 → a + b + c > a + b → a + c > b → b + c > a → a + b + c ≠ 48 := by
  sorry

end triangle_perimeter_not_48_l3398_339804


namespace repeated_root_condition_l3398_339806

theorem repeated_root_condition (m : ℝ) : 
  (∃! x : ℝ, (5 * x) / (x - 2) + 1 = m / (x - 2) ∧ x ≠ 2) ↔ m = 10 :=
by sorry

end repeated_root_condition_l3398_339806


namespace smallest_staircase_steps_l3398_339861

/-- The number of steps Cozy takes to climb the stairs -/
def cozy_jumps (n : ℕ) : ℕ := (n + 2) / 3

/-- The number of steps Dash takes to climb the stairs -/
def dash_jumps (n : ℕ) : ℕ := (n + 6) / 7

/-- Theorem stating the smallest number of steps in the staircase -/
theorem smallest_staircase_steps : 
  ∃ (n : ℕ), 
    n % 11 = 0 ∧ 
    cozy_jumps n - dash_jumps n = 13 ∧ 
    ∀ (m : ℕ), m < n → (m % 11 ≠ 0 ∨ cozy_jumps m - dash_jumps m ≠ 13) :=
by sorry

end smallest_staircase_steps_l3398_339861


namespace complement_of_M_in_U_l3398_339888

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : 
  (U \ M) = {3, 5} := by sorry

end complement_of_M_in_U_l3398_339888


namespace item_prices_l3398_339838

theorem item_prices (x y z : ℝ) 
  (eq1 : 3 * x + 5 * y + z = 32) 
  (eq2 : 4 * x + 7 * y + z = 40) : 
  x + y + z = 16 := by
  sorry

end item_prices_l3398_339838


namespace largest_vertex_sum_l3398_339860

/-- Represents a parabola passing through specific points -/
structure Parabola where
  a : ℤ
  T : ℤ
  h : T ≠ 0

/-- The sum of coordinates of the vertex of the parabola -/
def vertexSum (p : Parabola) : ℤ := p.T - p.a * p.T^2

/-- The parabola passes through the point (2T+1, 28) -/
def passesThroughC (p : Parabola) : Prop :=
  p.a * (2 * p.T + 1) = 28

theorem largest_vertex_sum :
  ∀ p : Parabola, passesThroughC p → vertexSum p ≤ 60 :=
sorry

end largest_vertex_sum_l3398_339860


namespace leon_order_total_l3398_339889

/-- Calculates the total amount Leon paid for his order, including discounts and delivery fee. -/
def total_paid (toy_organizer_price : ℚ) (toy_organizer_count : ℕ) 
                (gaming_chair_price : ℚ) (gaming_chair_count : ℕ)
                (desk_price : ℚ) (bookshelf_price : ℚ)
                (toy_organizer_discount : ℚ) (gaming_chair_discount : ℚ)
                (delivery_fee_rate : ℚ → ℚ) : ℚ :=
  let toy_organizer_total := toy_organizer_price * toy_organizer_count * (1 - toy_organizer_discount)
  let gaming_chair_total := gaming_chair_price * gaming_chair_count * (1 - gaming_chair_discount)
  let subtotal := toy_organizer_total + gaming_chair_total + desk_price + bookshelf_price
  let total_items := toy_organizer_count + gaming_chair_count + 2
  let delivery_fee := subtotal * delivery_fee_rate total_items
  subtotal + delivery_fee

/-- The statement to be proved -/
theorem leon_order_total :
  let toy_organizer_price : ℚ := 78
  let toy_organizer_count : ℕ := 3
  let gaming_chair_price : ℚ := 83
  let gaming_chair_count : ℕ := 2
  let desk_price : ℚ := 120
  let bookshelf_price : ℚ := 95
  let toy_organizer_discount : ℚ := 0.1
  let gaming_chair_discount : ℚ := 0.05
  let delivery_fee_rate (items : ℚ) : ℚ :=
    if items ≤ 3 then 0.04
    else if items ≤ 5 then 0.06
    else 0.08
  total_paid toy_organizer_price toy_organizer_count 
             gaming_chair_price gaming_chair_count
             desk_price bookshelf_price
             toy_organizer_discount gaming_chair_discount
             delivery_fee_rate = 629.96 := by
  sorry

end leon_order_total_l3398_339889


namespace amy_candy_difference_l3398_339835

/-- Amy's candy problem -/
theorem amy_candy_difference (initial : ℕ) (given_away : ℕ) (left : ℕ) 
  (h1 : given_away = 6)
  (h2 : left = 5)
  (h3 : initial = given_away + left) :
  given_away - left = 1 := by
  sorry

end amy_candy_difference_l3398_339835


namespace f_odd_and_decreasing_l3398_339898

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- Theorem statement
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end f_odd_and_decreasing_l3398_339898


namespace rice_division_l3398_339869

/-- Proves that dividing 25/4 pounds of rice equally among 4 containers results in 25 ounces per container. -/
theorem rice_division (total_weight : ℚ) (num_containers : ℕ) (pound_to_ounce : ℕ) :
  total_weight = 25 / 4 →
  num_containers = 4 →
  pound_to_ounce = 16 →
  (total_weight / num_containers) * pound_to_ounce = 25 := by
  sorry

end rice_division_l3398_339869


namespace andrew_sandwiches_l3398_339859

/-- The number of friends Andrew has coming over -/
def num_friends : ℕ := 4

/-- The number of sandwiches Andrew makes for each friend -/
def sandwiches_per_friend : ℕ := 3

/-- The total number of sandwiches Andrew made -/
def total_sandwiches : ℕ := num_friends * sandwiches_per_friend

theorem andrew_sandwiches : total_sandwiches = 12 := by
  sorry

end andrew_sandwiches_l3398_339859


namespace quadratic_properties_l3398_339879

/-- Quadratic function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_down : a < 0)
  (h_b : b < 0)
  (h_c : c > 0)
  (h_sym : ∀ x, f a b c (x - 1) = f a b c (-x - 1)) :
  abc > 0 ∧
  (∀ x, -3 < x ∧ x < 1 → f a b c x > 0) ∧
  f a b c (-4) = -10/3 ∧
  f a b c 2 = -10/3 ∧
  f a b c 1 = 0 ∧
  f a b c (-3/2) = 5/2 ∧
  f a b c (-1/2) = 5/2 := by
sorry

end quadratic_properties_l3398_339879


namespace multiple_of_nine_between_12_and_30_l3398_339842

theorem multiple_of_nine_between_12_and_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 144)
  (h3 : x < 30) :
  x = 18 ∨ x = 27 := by
  sorry

end multiple_of_nine_between_12_and_30_l3398_339842


namespace new_years_day_theorem_l3398_339890

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a year -/
structure Date where
  month : Nat
  day : Nat

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

theorem new_years_day_theorem 
  (february_has_29_days : Nat)
  (february_has_four_mondays : Nat)
  (february_has_five_sundays : Nat)
  (february_13_is_friday : DayOfWeek)
  : (february_has_29_days = 29) →
    (february_has_four_mondays = 4) →
    (february_has_five_sundays = 5) →
    (february_13_is_friday = DayOfWeek.Friday) →
    (∃ (new_years_day : DayOfWeek), 
      new_years_day = DayOfWeek.Thursday ∧ 
      advanceDay new_years_day 366 = DayOfWeek.Saturday) :=
by sorry


end new_years_day_theorem_l3398_339890


namespace volleyball_team_chemistry_l3398_339802

theorem volleyball_team_chemistry (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) :
  total_players = 30 →
  physics_players = 15 →
  both_subjects = 10 →
  total_players = (physics_players - both_subjects) + (total_players - (physics_players - both_subjects)) →
  (total_players - (physics_players - both_subjects)) = 25 :=
by sorry

end volleyball_team_chemistry_l3398_339802


namespace fraction_problem_l3398_339885

theorem fraction_problem (n : ℚ) (f : ℚ) (h1 : n = 120) (h2 : (1/2) * f * n = 36) : f = 3/5 := by
  sorry

end fraction_problem_l3398_339885


namespace fruit_cost_solution_l3398_339896

/-- Given a system of linear equations representing the cost of fruits,
    prove that the solution satisfies the given equations. -/
theorem fruit_cost_solution (x y z : ℝ) : 
  x + 2 * y = 8.9 ∧ 
  2 * z + 3 * y = 23 ∧ 
  3 * z + 4 * x = 30.1 →
  x = 2.5 ∧ y = 3.2 ∧ z = 6.7 := by
sorry

end fruit_cost_solution_l3398_339896


namespace perpendicular_line_through_P_l3398_339895

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2*x - y + 4 = 0

theorem perpendicular_line_through_P : 
  (∀ x y : ℝ, given_line x y → (∃ m : ℝ, m * (2*x - y) = -1)) ∧ 
  perpendicular_line point_P.1 point_P.2 :=
sorry

end perpendicular_line_through_P_l3398_339895


namespace prob_even_product_is_four_fifths_l3398_339883

/-- Represents a spinner with a given set of numbers -/
structure Spinner where
  numbers : Finset ℕ

/-- The probability of selecting an even number from a spinner -/
def prob_even (s : Spinner) : ℚ :=
  (s.numbers.filter Even).card / s.numbers.card

/-- The probability of selecting an odd number from a spinner -/
def prob_odd (s : Spinner) : ℚ :=
  1 - prob_even s

/-- Spinner A with numbers 1 to 5 -/
def spinner_A : Spinner :=
  ⟨Finset.range 5 ∪ {5}⟩

/-- Spinner B with numbers 1, 2, 4 -/
def spinner_B : Spinner :=
  ⟨{1, 2, 4}⟩

theorem prob_even_product_is_four_fifths :
  1 - (prob_odd spinner_A * prob_odd spinner_B) = 4/5 := by
  sorry

end prob_even_product_is_four_fifths_l3398_339883


namespace remainder_theorem_l3398_339822

theorem remainder_theorem : (7 * 10^20 + 2^20) % 11 = 8 := by
  sorry

end remainder_theorem_l3398_339822


namespace two_segment_journey_average_speed_l3398_339865

/-- Calculates the average speed of a two-segment journey -/
theorem two_segment_journey_average_speed 
  (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : distance1 = 20) (h2 : speed1 = 10) (h3 : distance2 = 30) (h4 : speed2 = 20) :
  (distance1 + distance2) / ((distance1 / speed1) + (distance2 / speed2)) = 50 / 3.5 := by
sorry

#eval (20 + 30) / ((20 / 10) + (30 / 20)) -- To verify the result

end two_segment_journey_average_speed_l3398_339865


namespace cymbal_strike_interval_l3398_339894

def beats_between_triangle_strikes : ℕ := 2
def lcm_cymbal_triangle_strikes : ℕ := 14

theorem cymbal_strike_interval :
  ∃ (c : ℕ), c > 0 ∧ Nat.lcm c beats_between_triangle_strikes = lcm_cymbal_triangle_strikes ∧ c = 14 := by
  sorry

end cymbal_strike_interval_l3398_339894


namespace parallel_sum_diff_l3398_339844

/-- Two vectors in ℝ² are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Given vectors a and b in ℝ², if a + b is parallel to a - b, then the second component of b is 1 -/
theorem parallel_sum_diff (x : ℝ) : 
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (2, x)
  are_parallel (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) → x = 1 := by
  sorry

end parallel_sum_diff_l3398_339844


namespace equation_solution_l3398_339820

theorem equation_solution : 
  ∃ x : ℚ, (24 - 4 = 3 * (1 + x)) ∧ (x = 17 / 3) :=
by sorry

end equation_solution_l3398_339820


namespace period_of_tan_transformed_l3398_339843

open Real

/-- The period of the tangent function with a transformed argument -/
theorem period_of_tan_transformed (f : ℝ → ℝ) :
  (∀ x, f x = tan (2 * x / 3)) →
  (∃ p > 0, ∀ x, f (x + p) = f x) ∧
  (∀ q > 0, (∀ x, f (x + q) = f x) → q ≥ 3 * π / 2) :=
sorry

end period_of_tan_transformed_l3398_339843


namespace cats_and_dogs_sum_l3398_339867

/-- Represents the number of individuals of each type on the ship --/
structure ShipPopulation where
  cats : ℕ
  parrots : ℕ
  dogs : ℕ
  sailors : ℕ
  cook : ℕ := 1
  captain : ℕ := 1

/-- The total number of heads on the ship --/
def totalHeads (p : ShipPopulation) : ℕ :=
  p.cats + p.parrots + p.dogs + p.sailors + p.cook + p.captain

/-- The total number of legs on the ship --/
def totalLegs (p : ShipPopulation) : ℕ :=
  4 * p.cats + 2 * p.parrots + 4 * p.dogs + 2 * p.sailors + 2 * p.cook + 1 * p.captain

/-- Theorem stating that the total number of cats and dogs is 14 --/
theorem cats_and_dogs_sum (p : ShipPopulation) 
    (h1 : totalHeads p = 38) 
    (h2 : totalLegs p = 103) : 
  p.cats + p.dogs = 14 := by
  sorry

end cats_and_dogs_sum_l3398_339867


namespace one_tangent_line_l3398_339810

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 26 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

-- Define a function to count the number of tangent lines
def count_tangent_lines (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- Theorem stating that there is exactly one tangent line
theorem one_tangent_line : count_tangent_lines C1 C2 = 1 := by sorry

end one_tangent_line_l3398_339810


namespace fraction_to_decimal_l3398_339876

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := by sorry

end fraction_to_decimal_l3398_339876


namespace integer_divisibility_problem_l3398_339809

theorem integer_divisibility_problem (n : ℤ) :
  (∃ k : ℤ, n - 4 = 6 * k) ∧ (∃ m : ℤ, n - 8 = 10 * m) →
  n ≡ 28 [ZMOD 30] := by
  sorry

end integer_divisibility_problem_l3398_339809


namespace complex_power_20_l3398_339892

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_20 : (1 + i) ^ 20 = -1024 := by
  sorry

end complex_power_20_l3398_339892


namespace expression_value_l3398_339873

theorem expression_value (x y : ℤ) (hx : x = -6) (hy : y = -3) :
  4 * (x - y)^2 - x * y = 18 := by sorry

end expression_value_l3398_339873


namespace ending_number_proof_l3398_339899

theorem ending_number_proof (n : ℕ) (h1 : n > 45) (h2 : ∃ (evens : List ℕ), 
  evens.length = 30 ∧ 
  (∀ m ∈ evens, Even m ∧ m > 45 ∧ m ≤ n) ∧
  (∀ m, 45 < m ∧ m ≤ n ∧ Even m → m ∈ evens)) : 
  n = 104 := by
sorry

end ending_number_proof_l3398_339899


namespace tigers_games_count_l3398_339866

theorem tigers_games_count :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (60 * initial_games) / 100 →
    ∀ (final_games : ℕ),
      final_games = initial_games + 11 →
      (initial_wins + 8) = (65 * final_games) / 100 →
      final_games = 28 := by
sorry

end tigers_games_count_l3398_339866


namespace residue_of_negative_1035_mod_37_l3398_339833

theorem residue_of_negative_1035_mod_37 :
  ∃ (r : ℤ), r ≥ 0 ∧ r < 37 ∧ -1035 ≡ r [ZMOD 37] ∧ r = 1 := by
  sorry

end residue_of_negative_1035_mod_37_l3398_339833


namespace composite_numbers_l3398_339814

theorem composite_numbers (n : ℕ) (h : n > 2) : 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 2*n^2 + 1 = a * b) ∧ 
  (∃ c d : ℕ, c > 1 ∧ d > 1 ∧ n^4 + n^2 + 1 = c * d) := by
  sorry

#check composite_numbers

end composite_numbers_l3398_339814


namespace cube_volume_in_box_l3398_339829

theorem cube_volume_in_box (box_length box_width box_height : ℝ)
  (num_cubes : ℕ) (cube_volume : ℝ) :
  box_length = 8 →
  box_width = 9 →
  box_height = 12 →
  num_cubes = 24 →
  cube_volume * num_cubes = box_length * box_width * box_height →
  cube_volume = 36 :=
by sorry

end cube_volume_in_box_l3398_339829


namespace train_crossing_time_l3398_339818

-- Define the given parameters
def train_speed_kmph : ℝ := 72
def train_speed_ms : ℝ := 20
def platform_length : ℝ := 260
def time_cross_platform : ℝ := 31

-- Define the theorem
theorem train_crossing_time (train_length : ℝ) 
  (h1 : train_length + platform_length = train_speed_ms * time_cross_platform)
  (h2 : train_speed_kmph * (1000 / 3600) = train_speed_ms) :
  train_length / train_speed_ms = 18 := by
  sorry

-- Note: The proof is omitted as per the instructions

end train_crossing_time_l3398_339818


namespace complex_number_in_first_quadrant_l3398_339801

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (i / (1 + i) : ℂ) = a + b * I :=
  sorry

end complex_number_in_first_quadrant_l3398_339801


namespace sum_a_d_equals_six_l3398_339825

theorem sum_a_d_equals_six (a b c d : ℝ) 
  (hab : a + b = 12) 
  (hbc : b + c = 9) 
  (hcd : c + d = 3) : 
  a + d = 6 := by
sorry

end sum_a_d_equals_six_l3398_339825


namespace other_items_tax_is_ten_percent_l3398_339828

/-- Represents the tax rates and spending percentages in Jill's shopping trip -/
structure ShoppingTax where
  clothing_spend : Rat
  food_spend : Rat
  other_spend : Rat
  clothing_tax : Rat
  food_tax : Rat
  total_tax : Rat

/-- The tax rate on other items given the shopping tax structure -/
def other_items_tax_rate (st : ShoppingTax) : Rat :=
  (st.total_tax - st.clothing_tax * st.clothing_spend) / st.other_spend

/-- Theorem stating that the tax rate on other items is 10% -/
theorem other_items_tax_is_ten_percent (st : ShoppingTax) 
  (h1 : st.clothing_spend = 1/2)
  (h2 : st.food_spend = 1/5)
  (h3 : st.other_spend = 3/10)
  (h4 : st.clothing_tax = 1/20)
  (h5 : st.food_tax = 0)
  (h6 : st.total_tax = 11/200) :
  other_items_tax_rate st = 1/10 := by
  sorry


end other_items_tax_is_ten_percent_l3398_339828


namespace work_rate_problem_l3398_339858

theorem work_rate_problem (A B C D : ℚ) :
  A = 1 / 4 →
  A + C = 1 / 2 →
  B + C = 1 / 3 →
  D = 1 / 5 →
  A + B + C + D = 1 →
  B = 13 / 60 :=
by sorry

end work_rate_problem_l3398_339858


namespace fifa_world_cup_2010_matches_l3398_339852

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the number of matches in a knockout tournament -/
def knockoutMatches (n : Nat) : Nat :=
  n - 1

theorem fifa_world_cup_2010_matches : 
  let totalTeams : Nat := 24
  let groups : Nat := 6
  let teamsPerGroup : Nat := 4
  let knockoutTeams : Nat := 16
  let firstRoundMatches := groups * roundRobinMatches teamsPerGroup
  let knockoutStageMatches := knockoutMatches knockoutTeams
  firstRoundMatches + knockoutStageMatches = 51 := by
  sorry

end fifa_world_cup_2010_matches_l3398_339852


namespace fraction_equality_l3398_339826

theorem fraction_equality (m n : ℝ) (h : n ≠ 0) (h1 : m / n = 2 / 3) :
  m / (m + n) = 2 / 5 := by
  sorry

end fraction_equality_l3398_339826


namespace students_per_table_l3398_339817

theorem students_per_table (total_tables : ℕ) (total_students : ℕ) 
  (h1 : total_tables = 34) (h2 : total_students = 204) : 
  total_students / total_tables = 6 := by
  sorry

end students_per_table_l3398_339817


namespace exactly_two_consecutive_sets_sum_18_l3398_339851

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)
  (h_length : length ≥ 2)

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a ConsecutiveSet that sums to 18 -/
def sums_to_18 (s : ConsecutiveSet) : Prop :=
  sum_consecutive s = 18

theorem exactly_two_consecutive_sets_sum_18 :
  ∃! (sets : Finset ConsecutiveSet), (∀ s ∈ sets, sums_to_18 s) ∧ sets.card = 2 :=
sorry

end exactly_two_consecutive_sets_sum_18_l3398_339851


namespace brick_wall_bottom_row_l3398_339821

/-- Represents a brick wall with a decreasing number of bricks per row from bottom to top -/
structure BrickWall where
  numRows : Nat
  totalBricks : Nat
  bottomRowBricks : Nat

/-- Calculates the total number of bricks in the wall given the number of bricks in the bottom row -/
def sumBricks (n : Nat) : Nat :=
  List.range n |> List.map (fun i => n - i) |> List.sum

/-- Theorem: A brick wall with 5 rows and 50 total bricks, where each row above the bottom
    has one less brick than the row below, has 12 bricks in the bottom row -/
theorem brick_wall_bottom_row : 
  ∀ (wall : BrickWall), 
    wall.numRows = 5 → 
    wall.totalBricks = 50 → 
    (sumBricks wall.bottomRowBricks = wall.totalBricks) → 
    wall.bottomRowBricks = 12 := by
  sorry

end brick_wall_bottom_row_l3398_339821


namespace bethany_current_age_l3398_339871

def bethany_age_problem (bethany_age : ℕ) (sister_age : ℕ) : Prop :=
  (bethany_age - 3 = 2 * (sister_age - 3)) ∧ (sister_age + 5 = 16)

theorem bethany_current_age :
  ∃ (bethany_age : ℕ) (sister_age : ℕ), bethany_age_problem bethany_age sister_age ∧ bethany_age = 19 :=
by
  sorry

end bethany_current_age_l3398_339871


namespace arithmetic_progression_sum_l3398_339848

/-- Given an arithmetic progression where the sum of the 4th and 12th terms is 10,
    prove that the sum of the first 15 terms is 75. -/
theorem arithmetic_progression_sum (a d : ℝ) : 
  (a + 3*d) + (a + 11*d) = 10 → 
  (15 : ℝ) / 2 * (2*a + 14*d) = 75 := by
sorry

end arithmetic_progression_sum_l3398_339848


namespace jims_weight_l3398_339862

theorem jims_weight (jim steve stan : ℕ) 
  (h1 : stan = steve + 5)
  (h2 : steve = jim - 8)
  (h3 : jim + steve + stan = 319) :
  jim = 110 := by
sorry

end jims_weight_l3398_339862


namespace john_needs_60_bags_l3398_339868

/-- Calculates the number of half-ton bags of horse food needed for a given number of horses, 
    feedings per day, pounds per feeding, and number of days. --/
def bags_needed (num_horses : ℕ) (feedings_per_day : ℕ) (pounds_per_feeding : ℕ) (days : ℕ) : ℕ :=
  let daily_food_per_horse := feedings_per_day * pounds_per_feeding
  let total_daily_food := daily_food_per_horse * num_horses
  let total_food := total_daily_food * days
  let bag_weight := 1000  -- half-ton in pounds
  total_food / bag_weight

/-- Theorem stating that John needs 60 bags of food for his horses over 60 days. --/
theorem john_needs_60_bags : 
  bags_needed 25 2 20 60 = 60 := by
  sorry

end john_needs_60_bags_l3398_339868


namespace unique_six_digit_number_l3398_339849

/-- A function that checks if a number is a 6-digit number beginning and ending with 2 -/
def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n % 10 = 2 ∧ n / 100000 = 2

/-- A function that checks if a number is the product of three consecutive even integers -/
def is_product_of_three_consecutive_even (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k) * (2*k + 2) * (2*k + 4)

theorem unique_six_digit_number : 
  ∀ n : ℕ, is_valid_number n ∧ is_product_of_three_consecutive_even n ↔ n = 287232 :=
sorry

end unique_six_digit_number_l3398_339849


namespace speeding_statistics_l3398_339831

structure SpeedingCategory where
  name : Char
  percentMotorists : ℝ
  ticketRate : ℝ

def categoryA : SpeedingCategory := ⟨'A', 0.14, 0.25⟩
def categoryB : SpeedingCategory := ⟨'B', 0.07, 0.55⟩
def categoryC : SpeedingCategory := ⟨'C', 0.04, 0.80⟩
def categoryD : SpeedingCategory := ⟨'D', 0.02, 0.95⟩

def categories : List SpeedingCategory := [categoryA, categoryB, categoryC, categoryD]

theorem speeding_statistics :
  (List.sum (categories.map (λ c => c.percentMotorists)) = 0.27) ∧
  (categoryA.percentMotorists * categoryA.ticketRate = 0.035) ∧
  (categoryB.percentMotorists * categoryB.ticketRate = 0.0385) ∧
  (categoryC.percentMotorists * categoryC.ticketRate = 0.032) ∧
  (categoryD.percentMotorists * categoryD.ticketRate = 0.019) :=
by sorry

end speeding_statistics_l3398_339831


namespace waitress_income_fraction_l3398_339803

theorem waitress_income_fraction (salary : ℚ) (tips : ℚ) (income : ℚ) :
  tips = (11 / 4) * salary →
  income = salary + tips →
  tips / income = 11 / 15 :=
by sorry

end waitress_income_fraction_l3398_339803


namespace second_group_size_l3398_339884

/-- Given a gym class with two groups of students, prove that the second group has 37 students. -/
theorem second_group_size (total : ℕ) (group1 : ℕ) (group2 : ℕ) : 
  total = 71 → group1 = 34 → total = group1 + group2 → group2 = 37 := by
  sorry

end second_group_size_l3398_339884


namespace adoption_cost_calculation_l3398_339808

/-- Calculates the total cost of preparing animals for adoption --/
def total_adoption_cost (cat_prep_cost adult_dog_prep_cost puppy_prep_cost : ℕ) 
                        (num_cats num_adult_dogs num_puppies : ℕ)
                        (additional_costs : List ℝ) : ℝ :=
  (cat_prep_cost * num_cats + 
   adult_dog_prep_cost * num_adult_dogs + 
   puppy_prep_cost * num_puppies : ℝ) + 
  additional_costs.sum

/-- Theorem stating the total cost for the given scenario --/
theorem adoption_cost_calculation 
  (cat_prep_cost : ℕ) (adult_dog_prep_cost : ℕ) (puppy_prep_cost : ℕ)
  (x1 x2 x3 x4 x5 x6 x7 : ℝ) :
  cat_prep_cost = 50 →
  adult_dog_prep_cost = 100 →
  puppy_prep_cost = 150 →
  total_adoption_cost cat_prep_cost adult_dog_prep_cost puppy_prep_cost 2 3 2 [x1, x2, x3, x4, x5, x6, x7] = 
    700 + x1 + x2 + x3 + x4 + x5 + x6 + x7 :=
by
  sorry

end adoption_cost_calculation_l3398_339808


namespace unique_tangent_line_l3398_339870

/-- The function f(x) = x^4 + 4x^3 - 26x^2 -/
def f (x : ℝ) : ℝ := x^4 + 4*x^3 - 26*x^2

/-- The line L(x) = 60x - 225 -/
def L (x : ℝ) : ℝ := 60*x - 225

theorem unique_tangent_line :
  ∃! (a b : ℝ), 
    (∀ x : ℝ, f x ≥ a*x + b ∨ f x ≤ a*x + b) ∧ 
    (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a*x₁ + b ∧ f x₂ = a*x₂ + b) ∧
    a = 60 ∧ b = -225 :=
sorry

end unique_tangent_line_l3398_339870


namespace octadecagon_diagonals_l3398_339891

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octadecagon has 18 sides -/
def octadecagon_sides : ℕ := 18

theorem octadecagon_diagonals :
  num_diagonals octadecagon_sides = 135 := by
  sorry

end octadecagon_diagonals_l3398_339891


namespace second_player_winning_strategy_l3398_339863

/-- Represents a character in the message -/
inductive Character
| Letter (c : Char)
| ExclamationMark

/-- Represents the state of the game board -/
def Board := List Character

/-- Represents a valid move in the game -/
inductive Move
| EraseSingle (c : Character)
| EraseMultiple (c : Char) (n : Nat)

/-- Applies a move to the board -/
def applyMove (board : Board) (move : Move) : Board :=
  sorry

/-- Checks if the game is over (no more characters to erase) -/
def isGameOver (board : Board) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := Board → Move

/-- Checks if a strategy is winning for the current player -/
def isWinningStrategy (strategy : Strategy) (board : Board) : Bool :=
  sorry

theorem second_player_winning_strategy 
  (initialBoard : Board) : 
  ∃ (strategy : Strategy), isWinningStrategy strategy initialBoard :=
sorry

end second_player_winning_strategy_l3398_339863


namespace salmon_population_increase_l3398_339824

theorem salmon_population_increase (initial_salmon : ℕ) (increase_factor : ℕ) : 
  initial_salmon = 500 → increase_factor = 10 → initial_salmon * increase_factor = 5000 := by
  sorry

end salmon_population_increase_l3398_339824


namespace quadratic_properties_l3398_339847

-- Define the quadratic function
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 4 * m * x + m - 2

-- Define the conditions
theorem quadratic_properties (m : ℝ) (h_m_neq_0 : m ≠ 0) 
  (h_distinct_roots : ∃ M N : ℝ, M ≠ N ∧ quadratic_function m M = 0 ∧ quadratic_function m N = 0)
  (h_passes_through_A : quadratic_function m 3 = 0) :
  -- 1. The value of m is -1
  m = -1 ∧
  -- 2. The vertex coordinates are (2, 1)
  (let vertex_x := 2; let vertex_y := 1;
   quadratic_function m vertex_x = vertex_y ∧
   ∀ x : ℝ, quadratic_function m x ≤ vertex_y) ∧
  -- 3. When m < 0 and MN ≤ 4, the range of m is m < 0
  (m < 0 → ∀ M N : ℝ, M ≠ N → quadratic_function m M = 0 → quadratic_function m N = 0 →
    (M - N)^2 ≤ 4^2 → m < 0) := by
  sorry

end quadratic_properties_l3398_339847


namespace circle_op_calculation_l3398_339834

-- Define the ⊗ operation
def circle_op (a b : ℚ) : ℚ := (a^2 + b^2) / (a + b)

-- State the theorem
theorem circle_op_calculation : circle_op (circle_op 5 2) 4 = 11375 / 2793 := by
  sorry

end circle_op_calculation_l3398_339834


namespace tensor_solution_l3398_339853

/-- Custom operation ⊗ -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

theorem tensor_solution :
  ∀ m : ℝ, m > 0 → tensor 1 m = 3 → m = 1 := by sorry

end tensor_solution_l3398_339853


namespace f_range_characterization_l3398_339837

def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem f_range_characterization :
  (∀ x : ℝ, f x > 2 ↔ x < (1/2) ∨ x > (5/2)) ∧
  (∀ x : ℝ, (∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ |a| * f x) ↔ (1/2) ≤ x ∧ x ≤ (5/2)) :=
by sorry

end f_range_characterization_l3398_339837


namespace function_value_at_log_third_l3398_339846

/-- Given a function f and a real number a, proves that f(ln(1/3)) = -1 -/
theorem function_value_at_log_third (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 2^x / (2^x + 1) + a * x
  f (Real.log 3) = 2 → f (Real.log (1/3)) = -1 := by
  sorry

end function_value_at_log_third_l3398_339846


namespace g_is_even_l3398_339855

noncomputable def g (x : ℝ) : ℝ := Real.log (x^2 + Real.sqrt (1 + x^4))

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by sorry

end g_is_even_l3398_339855


namespace smallest_positive_multiple_of_45_l3398_339881

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, k > 0 ∧ n = 45 * k) → n ≥ 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l3398_339881


namespace square_sum_geq_product_l3398_339872

theorem square_sum_geq_product {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : 2 * (a + b + c + d) ≥ a * b * c * d) :
  a^2 + b^2 + c^2 + d^2 ≥ a * b * c * d := by
  sorry

end square_sum_geq_product_l3398_339872


namespace f_leq_f_f_eq_abs_f_l3398_339839

/-- The function f(x) = x^2 + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*a

theorem f_leq_f'_iff_a_geq_three_halves (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) (-1), f a x ≤ f' a x) ↔ a ≥ 3/2 := by sorry

theorem f_eq_abs_f'_solutions (a : ℝ) :
  (∀ x : ℝ, f a x = |f' a x|) ↔
  ((a < -1 ∧ (x = -1 ∨ x = 1 - 2*a)) ∨
   (-1 ≤ a ∧ a ≤ 1 ∧ (x = 1 ∨ x = -1 ∨ x = 1 - 2*a ∨ x = -(1 + 2*a))) ∨
   (a > 1 ∧ (x = 1 ∨ x = -(1 + 2*a)))) := by sorry

end f_leq_f_f_eq_abs_f_l3398_339839


namespace hyperbola_locus_l3398_339856

/-- The locus of points P satisfying |PM| - |PN| = 4, where M(-3, 0) and N(3, 0) are fixed points -/
def rightBranchHyperbola : Set (ℝ × ℝ) :=
  {P | ‖P - (-3, 0)‖ - ‖P - (3, 0)‖ = 4 ∧ P.1 > 3}

/-- Theorem stating that the locus of points P satisfying |PM| - |PN| = 4 
    is the right branch of a hyperbola with foci M(-3, 0) and N(3, 0) -/
theorem hyperbola_locus :
  ∀ P : ℝ × ℝ, P ∈ rightBranchHyperbola ↔ 
    (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
      (P.1 / a)^2 - (P.2 / b)^2 = 1 ∧
      a^2 - b^2 = 9 ∧
      P.1 > 3) :=
by sorry

end hyperbola_locus_l3398_339856


namespace pants_and_belt_cost_l3398_339819

/-- The total cost of a pair of pants and a belt, given their prices -/
def total_cost (pants_price belt_price : ℝ) : ℝ := pants_price + belt_price

theorem pants_and_belt_cost :
  let pants_price : ℝ := 34.0
  let belt_price : ℝ := pants_price + 2.93
  total_cost pants_price belt_price = 70.93 := by
sorry

end pants_and_belt_cost_l3398_339819


namespace necessary_but_not_sufficient_l3398_339815

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) :=
sorry

end necessary_but_not_sufficient_l3398_339815


namespace point_on_terminal_side_l3398_339807

theorem point_on_terminal_side (x : ℝ) (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (x, 2) ∧ P.2 / Real.sqrt (P.1^2 + P.2^2) = 2/3) →
  x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end point_on_terminal_side_l3398_339807


namespace evaluate_expression_l3398_339886

theorem evaluate_expression : (4^4 - 4*(4-1)^4)^4 = 21381376 := by
  sorry

end evaluate_expression_l3398_339886


namespace conditional_probability_l3398_339877

-- Define the probability measure z
variable (z : Set α → ℝ)

-- Define events x and y
variable (x y : Set α)

-- State the theorem
theorem conditional_probability
  (hx : z x = 0.02)
  (hy : z y = 0.10)
  (hxy : z (x ∩ y) = 0.10)
  (h_prob : ∀ A, 0 ≤ z A ∧ z A ≤ 1)
  (h_add : ∀ A B, z (A ∪ B) = z A + z B - z (A ∩ B))
  : z x / z y = 1 :=
sorry

end conditional_probability_l3398_339877


namespace abc_inequality_l3398_339882

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end abc_inequality_l3398_339882


namespace parallel_line_equation_l3398_339823

/-- Given a line L1 with equation 2x + y - 5 = 0 and a point P (1, -3),
    prove that the line L2 passing through P and parallel to L1
    has the equation 2x + y + 1 = 0 -/
theorem parallel_line_equation (x y : ℝ) :
  let L1 : ℝ × ℝ → Prop := λ (x, y) ↦ 2 * x + y - 5 = 0
  let P : ℝ × ℝ := (1, -3)
  let L2 : ℝ × ℝ → Prop := λ (x, y) ↦ 2 * x + y + 1 = 0
  (∀ (x₁ y₁ x₂ y₂ : ℝ), L1 (x₁, y₁) ∧ L1 (x₂, y₂) → (y₂ - y₁) = -2 * (x₂ - x₁)) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), L2 (x₁, y₁) ∧ L2 (x₂, y₂) → (y₂ - y₁) = -2 * (x₂ - x₁)) →
  L2 P →
  ∀ (a b : ℝ), (∀ (x y : ℝ), a * x + b * y + 1 = 0 ↔ L2 (x, y)) →
  a = 2 ∧ b = 1 :=
by sorry

end parallel_line_equation_l3398_339823


namespace sum_of_k_values_for_distinct_integer_solutions_l3398_339897

theorem sum_of_k_values_for_distinct_integer_solutions : ∃ (S : Finset ℤ), 
  (∀ k ∈ S, ∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 12 = 0 ∧ 3 * y^2 - k * y + 12 = 0) ∧ 
  (∀ k : ℤ, (∃ x y : ℤ, x ≠ y ∧ 3 * x^2 - k * x + 12 = 0 ∧ 3 * y^2 - k * y + 12 = 0) → k ∈ S) ∧
  (Finset.sum S id = 0) := by
sorry

end sum_of_k_values_for_distinct_integer_solutions_l3398_339897


namespace square_difference_l3398_339864

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 := by
sorry

end square_difference_l3398_339864


namespace beverage_selection_probabilities_l3398_339850

/-- The number of cups of Beverage A -/
def num_a : ℕ := 3

/-- The number of cups of Beverage B -/
def num_b : ℕ := 2

/-- The total number of cups -/
def total_cups : ℕ := num_a + num_b

/-- The number of cups to be selected -/
def select_cups : ℕ := 3

/-- The probability of selecting all cups of Beverage A -/
def prob_excellent : ℚ := 1 / 10

/-- The probability of selecting at least 2 cups of Beverage A -/
def prob_good_or_above : ℚ := 7 / 10

theorem beverage_selection_probabilities :
  (Nat.choose total_cups select_cups : ℚ) * prob_excellent = Nat.choose num_a select_cups ∧
  (Nat.choose total_cups select_cups : ℚ) * prob_good_or_above = 
    Nat.choose num_a select_cups + Nat.choose num_a 2 * Nat.choose num_b 1 := by
  sorry

end beverage_selection_probabilities_l3398_339850


namespace hen_egg_production_l3398_339878

/-- Given the following conditions:
    - There are 10 hens
    - Eggs are sold for $3 per dozen
    - In 4 weeks, $120 worth of eggs were sold
    Prove that each hen lays 12 eggs per week. -/
theorem hen_egg_production 
  (num_hens : ℕ) 
  (price_per_dozen : ℚ) 
  (weeks : ℕ) 
  (total_sales : ℚ) 
  (h1 : num_hens = 10)
  (h2 : price_per_dozen = 3)
  (h3 : weeks = 4)
  (h4 : total_sales = 120) :
  (total_sales / price_per_dozen * 12 / weeks / num_hens : ℚ) = 12 := by
sorry


end hen_egg_production_l3398_339878


namespace sum_reciprocal_inequality_root_inequality_l3398_339812

-- Problem 1
theorem sum_reciprocal_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by sorry

-- Problem 2
theorem root_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 + y^2) > (x^3 + y^3)^(1/3) := by sorry

end sum_reciprocal_inequality_root_inequality_l3398_339812


namespace f_max_min_sum_l3398_339880

noncomputable def f (x : ℝ) : ℝ :=
  ((Real.sqrt 1008 * x + Real.sqrt 1009)^2 + Real.sin (2018 * x)) / (2016 * x^2 + 2018)

def has_max_min (f : ℝ → ℝ) (M m : ℝ) : Prop :=
  (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ (∀ x, m ≤ f x) ∧ (∃ x, f x = m)

theorem f_max_min_sum :
  ∃ M m : ℝ, has_max_min f M m ∧ M + m = 1 :=
sorry

end f_max_min_sum_l3398_339880


namespace factorization_ax2_minus_4a_l3398_339836

theorem factorization_ax2_minus_4a (a x : ℝ) : a * x^2 - 4 * a = a * (x + 2) * (x - 2) := by
  sorry

end factorization_ax2_minus_4a_l3398_339836


namespace opposite_of_2023_l3398_339800

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by
  sorry

end opposite_of_2023_l3398_339800


namespace divided_stick_properties_l3398_339845

/-- Represents a stick divided into segments by different colored lines -/
structure DividedStick where
  length : ℝ
  red_segments : ℕ
  blue_segments : ℕ
  black_segments : ℕ

/-- Calculates the total number of segments after cutting -/
def total_segments (stick : DividedStick) : ℕ := sorry

/-- Calculates the length of the shortest segment -/
def shortest_segment (stick : DividedStick) : ℝ := sorry

/-- Theorem stating the properties of a stick divided into 8, 12, and 18 segments -/
theorem divided_stick_properties (L : ℝ) (h : L > 0) :
  let stick := DividedStick.mk L 8 12 18
  total_segments stick = 28 ∧ shortest_segment stick = L / 72 := by sorry

end divided_stick_properties_l3398_339845


namespace negative_sixty_four_to_four_thirds_l3398_339841

theorem negative_sixty_four_to_four_thirds (x : ℝ) : x = (-64)^(4/3) → x = 256 := by
  sorry

end negative_sixty_four_to_four_thirds_l3398_339841


namespace kara_water_consumption_l3398_339805

/-- Amount of water Kara drinks with each medication dose -/
def water_per_dose (total_water : ℕ) (doses_per_day : ℕ) (total_days : ℕ) (missed_doses : ℕ) : ℚ :=
  total_water / (doses_per_day * total_days - missed_doses)

/-- Theorem stating that Kara drinks 4 ounces of water per medication dose -/
theorem kara_water_consumption :
  water_per_dose 160 3 14 2 = 4 := by
  sorry

end kara_water_consumption_l3398_339805


namespace min_value_of_function_equality_condition_l3398_339811

theorem min_value_of_function (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 :=
sorry

theorem equality_condition (x : ℝ) (h : x > 1) : 
  x + 1 / (x - 1) = 3 ↔ x = 2 :=
sorry

end min_value_of_function_equality_condition_l3398_339811


namespace allocation_schemes_count_l3398_339857

/-- The number of ways to divide 6 volunteers into 4 groups and assign them to venues -/
def allocationSchemes : ℕ :=
  let n := 6  -- number of volunteers
  let k := 4  -- number of groups/venues
  let g₂ := 2  -- number of groups with 2 people
  let g₁ := 2  -- number of groups with 1 person
  540

/-- Theorem stating that the number of allocation schemes is 540 -/
theorem allocation_schemes_count : allocationSchemes = 540 := by
  sorry

end allocation_schemes_count_l3398_339857
