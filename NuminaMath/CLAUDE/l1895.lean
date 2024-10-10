import Mathlib

namespace prime_sum_theorem_l1895_189569

theorem prime_sum_theorem (a b : ℕ) : 
  Prime a → Prime b → a^2 + b = 2003 → a + b = 2001 := by sorry

end prime_sum_theorem_l1895_189569


namespace zoo_ticket_cost_l1895_189542

theorem zoo_ticket_cost (adult_price : ℝ) : 
  (adult_price > 0) →
  (6 * adult_price + 5 * (adult_price / 2) + 3 * (adult_price - 1.5) = 40.5) →
  (10 * adult_price + 8 * (adult_price / 2) + 4 * (adult_price - 1.5) = 64.38) :=
by
  sorry

end zoo_ticket_cost_l1895_189542


namespace quadratic_equation_solution_l1895_189571

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + 2*d = 0) (h2 : d^2 + c*d + 2*d = 0) : 
  c = 2 ∧ d = -4 := by
  sorry

end quadratic_equation_solution_l1895_189571


namespace sqrt_equality_problem_l1895_189515

theorem sqrt_equality_problem : ∃ (a x : ℝ), 
  x > 0 ∧ 
  Real.sqrt x = 2 * a - 3 ∧ 
  Real.sqrt x = 5 - a ∧ 
  a = -2 ∧ 
  x = 49 := by
  sorry

end sqrt_equality_problem_l1895_189515


namespace negation_of_absolute_value_less_than_zero_l1895_189578

theorem negation_of_absolute_value_less_than_zero :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x₀ : ℝ, |x₀| ≥ 0) := by sorry

end negation_of_absolute_value_less_than_zero_l1895_189578


namespace complex_magnitude_l1895_189525

theorem complex_magnitude (z : ℂ) : (z + Complex.I) * (2 - Complex.I) = 11 + 7 * Complex.I → Complex.abs z = 5 := by
  sorry

end complex_magnitude_l1895_189525


namespace arithmetic_sequence_problem_l1895_189528

theorem arithmetic_sequence_problem (n : ℕ) (sum : ℝ) (min_term max_term : ℝ) :
  n = 300 ∧ 
  sum = 22500 ∧ 
  min_term = 5 ∧ 
  max_term = 150 →
  let avg : ℝ := sum / n
  let d : ℝ := min ((avg - min_term) / (n - 1)) ((max_term - avg) / (n - 1))
  let L : ℝ := avg - (75 - 1) * d
  let G : ℝ := avg + (75 - 1) * d
  G - L = 31500 / 299 := by sorry

end arithmetic_sequence_problem_l1895_189528


namespace parallel_vectors_x_value_l1895_189593

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (2, x)
  parallel a b → x = -6 := by sorry

end parallel_vectors_x_value_l1895_189593


namespace bus_speed_excluding_stoppages_l1895_189520

/-- Given a bus that travels at 45 km/hr including stoppages and stops for 10 minutes per hour,
    prove that its speed excluding stoppages is 54 km/hr. -/
theorem bus_speed_excluding_stoppages :
  let speed_with_stoppages : ℝ := 45
  let stop_time_per_hour : ℝ := 10 / 60
  let travel_time_per_hour : ℝ := 1 - stop_time_per_hour
  let speed_without_stoppages : ℝ := speed_with_stoppages / travel_time_per_hour
  speed_without_stoppages = 54 := by
  sorry

end bus_speed_excluding_stoppages_l1895_189520


namespace fraction_sum_theorem_l1895_189541

theorem fraction_sum_theorem (a b c : ℝ) (h : ((a - b) * (b - c) * (c - a)) / ((a + b) * (b + c) * (c + a)) = 2004 / 2005) :
  a / (a + b) + b / (b + c) + c / (c + a) = 4011 / 4010 := by
  sorry

end fraction_sum_theorem_l1895_189541


namespace min_value_sum_l1895_189577

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 19 / x + 98 / y = 1) :
  x + y ≥ 203 := by
  sorry

end min_value_sum_l1895_189577


namespace total_photos_l1895_189565

def friends_photos : ℕ := 63
def family_photos : ℕ := 23

theorem total_photos : friends_photos + family_photos = 86 := by
  sorry

end total_photos_l1895_189565


namespace power_equation_solution_l1895_189519

theorem power_equation_solution (x : ℝ) : (2^4 * 3^6 : ℝ) = 9 * 6^x → x = 4 := by
  sorry

end power_equation_solution_l1895_189519


namespace income_percentage_difference_l1895_189598

/-- Given the monthly incomes of A, B, and C, prove that B's income is 12% more than C's -/
theorem income_percentage_difference :
  ∀ (a b c : ℝ),
  -- A's and B's monthly incomes are in the ratio 5:2
  a / b = 5 / 2 →
  -- C's monthly income is 12000
  c = 12000 →
  -- A's annual income is 403200.0000000001
  12 * a = 403200.0000000001 →
  -- B's monthly income is 12% more than C's
  b = 1.12 * c := by
sorry


end income_percentage_difference_l1895_189598


namespace hamburger_combinations_l1895_189539

/-- Represents the number of available condiments -/
def num_condiments : Nat := 8

/-- Represents the number of choices for meat patties -/
def meat_patty_choices : Nat := 3

/-- Calculates the total number of hamburger combinations -/
def total_combinations : Nat := 2^num_condiments * meat_patty_choices

/-- Theorem: The total number of different hamburger combinations is 768 -/
theorem hamburger_combinations : total_combinations = 768 := by
  sorry

end hamburger_combinations_l1895_189539


namespace cube_root_property_l1895_189589

theorem cube_root_property (x : ℤ) (h : x^3 = 9261) : (x + 1) * (x - 1) = 440 := by
  sorry

end cube_root_property_l1895_189589


namespace investment_time_q_is_thirteen_l1895_189558

/-- Represents the investment and profit data for two partners -/
structure PartnershipData where
  investment_ratio_p : ℚ
  investment_ratio_q : ℚ
  profit_ratio_p : ℚ
  profit_ratio_q : ℚ
  investment_time_p : ℚ

/-- Calculates the investment time for partner Q given the partnership data -/
def calculate_investment_time_q (data : PartnershipData) : ℚ :=
  (data.profit_ratio_q * data.investment_ratio_p * data.investment_time_p) / 
  (data.profit_ratio_p * data.investment_ratio_q)

/-- Theorem stating that given the specified partnership data, Q's investment time is 13 months -/
theorem investment_time_q_is_thirteen : 
  let data : PartnershipData := {
    investment_ratio_p := 7,
    investment_ratio_q := 5,
    profit_ratio_p := 7,
    profit_ratio_q := 13,
    investment_time_p := 5
  }
  calculate_investment_time_q data = 13 := by sorry

end investment_time_q_is_thirteen_l1895_189558


namespace binomial_30_3_l1895_189588

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l1895_189588


namespace monotone_increasing_ln_plus_ax_l1895_189500

open Real

theorem monotone_increasing_ln_plus_ax (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, Monotone (λ x => Real.log x + a * x)) →
  a ≥ -1/2 := by
sorry

end monotone_increasing_ln_plus_ax_l1895_189500


namespace square_sum_difference_specific_square_sum_difference_l1895_189586

theorem square_sum_difference (n : ℕ) : 
  (2*n+1)^2 - (2*n-1)^2 + (2*n-3)^2 - (2*n-5)^2 + (2*n-7)^2 - (2*n-9)^2 + 
  (2*n-11)^2 - (2*n-13)^2 + (2*n-15)^2 - (2*n-17)^2 + (2*n-19)^2 - (2*n-21)^2 + (2*n-23)^2 = 
  4*n^2 + 1 :=
by sorry

theorem specific_square_sum_difference : 
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 337 :=
by sorry

end square_sum_difference_specific_square_sum_difference_l1895_189586


namespace smoking_health_negative_correlation_l1895_189516

-- Define the type for relationships
inductive Relationship
| ParentChildHeight
| SmokingHealth
| CropYieldFertilization
| MathPhysicsGrades

-- Define a function to determine if a relationship is negatively correlated
def is_negatively_correlated (r : Relationship) : Prop :=
  match r with
  | Relationship.SmokingHealth => True
  | _ => False

-- Theorem statement
theorem smoking_health_negative_correlation :
  ∀ r : Relationship, is_negatively_correlated r ↔ r = Relationship.SmokingHealth :=
by sorry

end smoking_health_negative_correlation_l1895_189516


namespace part_one_part_two_l1895_189530

/-- Set A defined as {x | -2 ≤ x ≤ 2} -/
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

/-- Set B defined as {x | 1-m ≤ x ≤ 2m-2} where m is a real number -/
def B (m : ℝ) : Set ℝ := {x | 1-m ≤ x ∧ x ≤ 2*m-2}

/-- Theorem for part (1) -/
theorem part_one (m : ℝ) : A ⊆ B m ∧ A ≠ B m → m ∈ Set.Ici 3 := by sorry

/-- Theorem for part (2) -/
theorem part_two (m : ℝ) : A ∩ B m = B m → m ∈ Set.Iic 2 := by sorry

end part_one_part_two_l1895_189530


namespace product_of_three_numbers_l1895_189575

theorem product_of_three_numbers (x y z : ℝ) : 
  x + y + z = 30 → 
  x = 3 * (y + z) → 
  y = 6 * z → 
  x * y * z = 7762.5 := by
sorry

end product_of_three_numbers_l1895_189575


namespace second_player_prevents_complete_2x2_l1895_189573

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the state of a square (colored by first player, second player, or uncolored) -/
inductive SquareState
  | FirstPlayer
  | SecondPlayer
  | Uncolored

/-- Represents the game state -/
def GameState := Square → SquareState

/-- Represents a 2x2 square on the board -/
structure Square2x2 where
  topLeft : Square

/-- The strategy function for the second player -/
def secondPlayerStrategy (gs : GameState) (lastMove : Square) : Square := sorry

/-- Checks if a 2x2 square is completely colored by the first player -/
def isComplete2x2FirstPlayer (gs : GameState) (s : Square2x2) : Bool := sorry

/-- The main theorem stating that the second player can always prevent
    the first player from coloring any 2x2 square completely -/
theorem second_player_prevents_complete_2x2 :
  ∀ (numMoves : Nat) (gs : GameState),
    (∀ (s : Square), gs s = SquareState.Uncolored) →
    ∀ (moves : Fin numMoves → Square),
      let finalState := sorry  -- Final game state after all moves
      ∀ (s : Square2x2), ¬(isComplete2x2FirstPlayer finalState s) :=
sorry

end second_player_prevents_complete_2x2_l1895_189573


namespace factory_output_decrease_l1895_189523

theorem factory_output_decrease (initial_output : ℝ) : 
  let increased_output := initial_output * 1.1
  let holiday_output := increased_output * 1.4
  let required_decrease := (holiday_output - initial_output) / holiday_output * 100
  abs (required_decrease - 35.06) < 0.01 := by
sorry

end factory_output_decrease_l1895_189523


namespace ice_cream_cost_l1895_189568

/-- The cost of ice cream problem -/
theorem ice_cream_cost (ice_cream_quantity : ℕ) (yogurt_quantity : ℕ) (yogurt_cost : ℕ) (price_difference : ℕ) :
  ice_cream_quantity = 20 →
  yogurt_quantity = 2 →
  yogurt_cost = 1 →
  price_difference = 118 →
  ∃ (ice_cream_cost : ℕ), 
    ice_cream_cost * ice_cream_quantity = yogurt_cost * yogurt_quantity + price_difference ∧
    ice_cream_cost = 6 :=
by sorry

end ice_cream_cost_l1895_189568


namespace peach_difference_l1895_189529

/-- Given information about peaches owned by Jake, Steven, and Jill -/
theorem peach_difference (jill steven jake : ℕ) 
  (h1 : jake = steven - 5)  -- Jake has 5 fewer peaches than Steven
  (h2 : steven = jill + 18) -- Steven has 18 more peaches than Jill
  (h3 : jill = 87)          -- Jill has 87 peaches
  : jake - jill = 13 :=     -- Prove that Jake has 13 more peaches than Jill
by sorry

end peach_difference_l1895_189529


namespace total_students_l1895_189506

/-- The number of students taking history -/
def H : ℕ := 36

/-- The number of students taking statistics -/
def S : ℕ := 32

/-- The number of students taking history or statistics or both -/
def H_or_S : ℕ := 57

/-- The number of students taking history but not statistics -/
def H_not_S : ℕ := 25

/-- The theorem stating that the total number of students in the group is 57 -/
theorem total_students : H_or_S = 57 := by sorry

end total_students_l1895_189506


namespace min_value_theorem_l1895_189504

theorem min_value_theorem (a b : ℝ) (h : 2 * a - 3 * b + 6 = 0) :
  ∃ (min_val : ℝ), min_val = (1 / 4 : ℝ) ∧ ∀ (x : ℝ), 4^a + (1 / 8^b) ≥ x → x ≥ min_val :=
sorry

end min_value_theorem_l1895_189504


namespace perimeter_quarter_circle_square_l1895_189560

/-- The perimeter of a region bounded by quarter circular arcs constructed on each side of a square with side length 4/π is equal to 8. -/
theorem perimeter_quarter_circle_square : 
  let side_length : ℝ := 4 / Real.pi
  let quarter_circle_arc_length : ℝ := (1/4) * (2 * Real.pi * side_length)
  let num_arcs : ℕ := 4
  let perimeter : ℝ := num_arcs * quarter_circle_arc_length
  perimeter = 8 := by sorry

end perimeter_quarter_circle_square_l1895_189560


namespace parking_methods_count_l1895_189527

/-- Represents the number of parking spaces -/
def n : ℕ := 6

/-- Represents the number of cars to be parked -/
def k : ℕ := 3

/-- Calculates the number of ways to park cars when they are not adjacent -/
def non_adjacent_ways : ℕ := (n - k + 1).choose k * 2^k

/-- Calculates the number of ways to park cars when two are adjacent -/
def two_adjacent_ways : ℕ := 2 * k.choose 2 * (n - k).choose 1 * 2^2

/-- Calculates the number of ways to park cars when all are adjacent -/
def all_adjacent_ways : ℕ := (n - k + 1) * 2

/-- The total number of parking methods -/
def total_parking_methods : ℕ := non_adjacent_ways + two_adjacent_ways + all_adjacent_ways

theorem parking_methods_count : total_parking_methods = 528 := by sorry

end parking_methods_count_l1895_189527


namespace right_triangle_hypotenuse_l1895_189554

theorem right_triangle_hypotenuse (m1 m2 : ℝ) (h1 : m1 = Real.sqrt 52) (h2 : m2 = Real.sqrt 73) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 = c^2 ∧
  m1^2 = (2 * b^2 + 2 * c^2 - a^2) / 4 ∧
  m2^2 = (2 * a^2 + 2 * c^2 - b^2) / 4 ∧
  c = 10 :=
by sorry

end right_triangle_hypotenuse_l1895_189554


namespace unique_divisor_with_remainders_l1895_189548

theorem unique_divisor_with_remainders :
  ∃! N : ℕ,
    10 ≤ N ∧ N < 100 ∧
    5655 % N = 11 ∧
    5879 % N = 14 :=
by
  -- The proof would go here
  sorry

end unique_divisor_with_remainders_l1895_189548


namespace triangle_inequality_l1895_189543

theorem triangle_inequality (a b c : ℝ) (n : ℕ) (S : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_S : 2 * S = a + b + c) : 
  a^n / (b + c) + b^n / (c + a) + c^n / (a + b) ≥ (2/3)^(n-2) * S^(n-1) := by
sorry

end triangle_inequality_l1895_189543


namespace repeating_decimal_equals_fraction_l1895_189531

/-- The repeating decimal 0.565656... -/
def repeating_decimal : ℚ :=
  0 + (56 / 100) * (1 / (1 - 1/100))

/-- The target fraction 56/99 -/
def target_fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_equals_fraction :
  repeating_decimal = target_fraction := by sorry

end repeating_decimal_equals_fraction_l1895_189531


namespace rhombus_all_sides_equal_rectangle_not_necessarily_l1895_189545

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ (i j : Fin 4), sides i = sides j

/-- A rectangle is a quadrilateral with four right angles and opposite sides equal. -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Theorem stating that all sides of a rhombus are equal, but not necessarily for a rectangle -/
theorem rhombus_all_sides_equal_rectangle_not_necessarily (r : Rhombus) (rect : Rectangle) :
  (∀ (i j : Fin 4), r.sides i = r.sides j) ∧
  ¬(∀ (rect : Rectangle), rect.width = rect.height) :=
sorry

end rhombus_all_sides_equal_rectangle_not_necessarily_l1895_189545


namespace min_value_of_sum_l1895_189583

theorem min_value_of_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_eq : a^2 + 2*a*b + 2*a*c + 4*b*c = 12) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + 2*x*y + 2*x*z + 4*y*z = 12 → 
  a + b + c ≤ x + y + z ∧ a + b + c = 2 * Real.sqrt 3 := by
  sorry

end min_value_of_sum_l1895_189583


namespace probability_even_distinct_digits_l1895_189579

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i j, i ≠ j → digits.get i ≠ digits.get j

def count_favorable_outcomes : ℕ := 7 * 8 * 7 * 5

theorem probability_even_distinct_digits :
  (count_favorable_outcomes : ℚ) / (9999 - 2000 + 1 : ℚ) = 49 / 200 := by
  sorry

end probability_even_distinct_digits_l1895_189579


namespace speed_in_still_water_l1895_189585

/-- The speed of a man rowing a boat in still water, given his downstream speed and the speed of the current. -/
theorem speed_in_still_water 
  (downstream_speed : ℝ) 
  (current_speed : ℝ) 
  (h1 : downstream_speed = 17.9997120913593) 
  (h2 : current_speed = 3) : 
  downstream_speed - current_speed = 14.9997120913593 := by
  sorry

#eval (17.9997120913593 : Float) - 3

end speed_in_still_water_l1895_189585


namespace book_reading_fraction_l1895_189512

theorem book_reading_fraction (total_pages remaining_pages : ℕ) 
  (h1 : total_pages = 468)
  (h2 : remaining_pages = 96)
  (h3 : (7 : ℚ) / 13 * total_pages + remaining_pages < total_pages) :
  let pages_read_first_week := (7 : ℚ) / 13 * total_pages
  let pages_remaining_after_first_week := total_pages - pages_read_first_week
  let pages_read_second_week := pages_remaining_after_first_week - remaining_pages
  pages_read_second_week / pages_remaining_after_first_week = 5 / 9 := by
sorry

end book_reading_fraction_l1895_189512


namespace concentric_circles_area_ratio_l1895_189591

theorem concentric_circles_area_ratio : 
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let A₁ : ℝ := π * r₁^2  -- area of smaller circle
  let A₂ : ℝ := π * r₂^2  -- area of larger circle
  (A₂ - A₁) / A₁ = 8
  := by sorry

end concentric_circles_area_ratio_l1895_189591


namespace largest_smallest_three_digit_div_six_with_seven_l1895_189502

/-- A function that checks if a number contains the digit 7 --/
def contains_seven (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ (a = 7 ∨ b = 7 ∨ c = 7)

/-- A function that checks if a number is a three-digit number --/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- The main theorem --/
theorem largest_smallest_three_digit_div_six_with_seven :
  (∀ n : ℕ, is_three_digit n → n % 6 = 0 → contains_seven n → n ≤ 978) ∧
  (∀ n : ℕ, is_three_digit n → n % 6 = 0 → contains_seven n → 174 ≤ n) ∧
  is_three_digit 978 ∧ 978 % 6 = 0 ∧ contains_seven 978 ∧
  is_three_digit 174 ∧ 174 % 6 = 0 ∧ contains_seven 174 :=
by sorry

end largest_smallest_three_digit_div_six_with_seven_l1895_189502


namespace min_sum_odd_days_l1895_189544

/-- A sequence of 5 non-negative integers representing fish caught each day --/
def FishSequence := (ℕ × ℕ × ℕ × ℕ × ℕ)

/-- Check if a sequence is non-increasing --/
def is_non_increasing (seq : FishSequence) : Prop :=
  let (a, b, c, d, e) := seq
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e

/-- Calculate the sum of all elements in the sequence --/
def sum_sequence (seq : FishSequence) : ℕ :=
  let (a, b, c, d, e) := seq
  a + b + c + d + e

/-- Calculate the sum of 1st, 3rd, and 5th elements --/
def sum_odd_days (seq : FishSequence) : ℕ :=
  let (a, _, c, _, e) := seq
  a + c + e

/-- The main theorem --/
theorem min_sum_odd_days (seq : FishSequence) :
  is_non_increasing seq →
  sum_sequence seq = 100 →
  sum_odd_days seq ≥ 50 := by
  sorry

end min_sum_odd_days_l1895_189544


namespace modified_geometric_structure_pieces_l1895_189587

/-- Calculates the sum of an arithmetic progression -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculates the nth triangular number -/
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The total number of pieces in the modified geometric structure -/
theorem modified_geometric_structure_pieces :
  let num_rows : ℕ := 10
  let first_rod_count : ℕ := 3
  let rod_difference : ℕ := 3
  let connector_rows : ℕ := num_rows + 1
  let rod_count := arithmetic_sum first_rod_count rod_difference num_rows
  let connector_count := triangular_number connector_rows
  rod_count + connector_count = 231 := by
  sorry

end modified_geometric_structure_pieces_l1895_189587


namespace square_difference_equals_double_product_problem_instance_l1895_189597

theorem square_difference_equals_double_product (a b : ℕ) :
  (a + b)^2 - (a^2 + b^2) = 2 * a * b :=
by sorry

-- Specific instance for the given problem
theorem problem_instance : (25 + 15)^2 - (25^2 + 15^2) = 750 :=
by sorry

end square_difference_equals_double_product_problem_instance_l1895_189597


namespace circle_center_l1895_189546

def is_circle (a : ℝ) : Prop :=
  ∃ (h : a^2 = a + 2 ∧ a^2 ≠ 0),
  ∀ (x y : ℝ), a^2*x^2 + (a+2)*y^2 + 4*x + 8*y + 5*a = 0 →
  ∃ (r : ℝ), (x + 2)^2 + (y + 4)^2 = r^2

theorem circle_center (a : ℝ) (h : is_circle a) :
  ∃ (x y : ℝ), a^2*x^2 + (a+2)*y^2 + 4*x + 8*y + 5*a = 0 ∧ x = -2 ∧ y = -4 :=
sorry

end circle_center_l1895_189546


namespace solution_set_equivalence_l1895_189567

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/3

-- Define the solution set of f(e^x) > 0
def solution_set_f_exp_pos (x : ℝ) : Prop := x < -Real.log 3

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (Real.exp x) > 0 ↔ solution_set_f_exp_pos x) :=
sorry

end solution_set_equivalence_l1895_189567


namespace complex_fraction_power_l1895_189564

theorem complex_fraction_power (i : ℂ) (a b : ℝ) :
  i * i = -1 →
  (1 : ℂ) / (1 + i) = a + b * i →
  a ^ b = Real.sqrt 2 := by
  sorry

end complex_fraction_power_l1895_189564


namespace yoongi_flowers_l1895_189582

def flowers_problem (initial : ℕ) (to_eunji : ℕ) (to_yuna : ℕ) : Prop :=
  initial - (to_eunji + to_yuna) = 12

theorem yoongi_flowers : flowers_problem 28 7 9 := by
  sorry

end yoongi_flowers_l1895_189582


namespace chip_drawing_probability_l1895_189536

/-- The number of tan chips in the bag -/
def tan_chips : ℕ := 4

/-- The number of pink chips in the bag -/
def pink_chips : ℕ := 3

/-- The number of violet chips in the bag -/
def violet_chips : ℕ := 5

/-- The number of green chips in the bag -/
def green_chips : ℕ := 2

/-- The total number of chips in the bag -/
def total_chips : ℕ := tan_chips + pink_chips + violet_chips + green_chips

/-- The probability of drawing the chips as specified -/
def probability : ℚ := 1 / 42000

theorem chip_drawing_probability :
  (tan_chips.factorial * pink_chips.factorial * violet_chips.factorial * (3 + green_chips).factorial) / total_chips.factorial = probability := by
  sorry

end chip_drawing_probability_l1895_189536


namespace f_properties_l1895_189559

def f (x : ℝ) := x^3 - 3*x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (f 1 = -2) ∧
  (∀ x, x = -1 ∨ x = 1 → deriv f x = 0) ∧
  (∀ x, f x ≤ f (-1)) :=
by sorry

end f_properties_l1895_189559


namespace plastic_bottles_count_l1895_189533

/-- The weight of a glass bottle in grams -/
def glass_bottle_weight : ℕ := 200

/-- The weight of a plastic bottle in grams -/
def plastic_bottle_weight : ℕ := 50

/-- The total weight of the second scenario in grams -/
def total_weight : ℕ := 1050

/-- The number of glass bottles in the second scenario -/
def num_glass_bottles : ℕ := 4

theorem plastic_bottles_count :
  ∃ (x : ℕ), 
    3 * glass_bottle_weight = 600 ∧
    glass_bottle_weight = plastic_bottle_weight + 150 ∧
    4 * glass_bottle_weight + x * plastic_bottle_weight = total_weight ∧
    x = 5 := by
  sorry

end plastic_bottles_count_l1895_189533


namespace gcd_special_powers_l1895_189553

theorem gcd_special_powers : Nat.gcd (2^1001 - 1) (2^1012 - 1) = 2^11 - 1 := by
  sorry

end gcd_special_powers_l1895_189553


namespace complex_number_location_l1895_189535

theorem complex_number_location :
  let z : ℂ := ((-1 : ℂ) + Complex.I) / ((1 : ℂ) + Complex.I) - 1
  z = -1 + Complex.I ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_number_location_l1895_189535


namespace m_intersect_n_equals_n_l1895_189562

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M as the domain of ln(1-x)
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -6 < x ∧ x < 1}

-- Theorem statement
theorem m_intersect_n_equals_n : M ∩ N = N := by
  sorry

end m_intersect_n_equals_n_l1895_189562


namespace min_value_of_expression_equality_condition_l1895_189552

theorem min_value_of_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem equality_condition :
  Real.sqrt ((4/3)^2 + (2 - 4/3)^2) + Real.sqrt ((2 - 4/3)^2 + (2 + 4/3)^2) = 2 * Real.sqrt 5 :=
by sorry

end min_value_of_expression_equality_condition_l1895_189552


namespace sum_of_roots_equation_l1895_189540

theorem sum_of_roots_equation (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 3*x^2 - 10*x - 7*(x + 2)
  (∃ a b : ℝ, (∀ x, f x = (x - a) * (x - b) * (x + 2))) →
  a + b = 5 :=
by sorry

end sum_of_roots_equation_l1895_189540


namespace line_inclination_angle_l1895_189501

theorem line_inclination_angle (a : ℝ) (h : a < 0) :
  let line := {(x, y) : ℝ × ℝ | x - a * y + 2 = 0}
  let slope := (1 : ℝ) / a
  let inclination_angle := Real.pi + Real.arctan slope
  ∀ (x y : ℝ), (x, y) ∈ line → inclination_angle ∈ Set.Icc 0 Real.pi ∧
    Real.tan inclination_angle = slope :=
by sorry

end line_inclination_angle_l1895_189501


namespace problem_solution_l1895_189576

def f (x : ℝ) : ℝ := |2*x + 2| + |x - 3|

theorem problem_solution :
  (∃ m : ℝ, m > 0 ∧ (∀ x : ℝ, f x ≥ m) ∧
    (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = m →
      1 / (a + b) + 1 / (b + c) + 1 / (a + c) ≥ 9 / (2 * m))) ∧
  {x : ℝ | f x ≤ 5} = {x : ℝ | -4/3 ≤ x ∧ x ≤ 0} := by
  sorry

end problem_solution_l1895_189576


namespace distance_P_to_y_axis_l1895_189510

/-- The distance from a point to the y-axis in a Cartesian coordinate system --/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- Point P in the Cartesian coordinate system --/
def P : ℝ × ℝ := (-3, 4)

/-- Theorem: The distance from P(-3,4) to the y-axis is 3 --/
theorem distance_P_to_y_axis :
  distance_to_y_axis P.1 P.2 = 3 := by
  sorry

end distance_P_to_y_axis_l1895_189510


namespace correct_age_order_l1895_189537

-- Define the set of friends
inductive Friend : Type
| David : Friend
| Emma : Friend
| Fiona : Friend
| George : Friend

-- Define a type for age comparisons
def AgeOrder := Friend → Friend → Prop

-- Define the problem conditions
def ProblemConditions (order : AgeOrder) : Prop :=
  -- All friends have different ages
  (∀ x y : Friend, x ≠ y → (order x y ∨ order y x)) ∧
  (∀ x y : Friend, order x y → ¬order y x) ∧
  -- Exactly one of the following statements is true
  (((order Friend.Emma Friend.David) ∧ 
    ¬(¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    ¬(∀ x : Friend, order Friend.George x) ∧
    ¬(∃ x : Friend, order x Friend.David)) ∨
   (¬(order Friend.Emma Friend.David) ∧ 
    (¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    ¬(∀ x : Friend, order Friend.George x) ∧
    ¬(∃ x : Friend, order x Friend.David)) ∨
   (¬(order Friend.Emma Friend.David) ∧ 
    ¬(¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    (∀ x : Friend, order Friend.George x) ∧
    ¬(∃ x : Friend, order x Friend.David)) ∨
   (¬(order Friend.Emma Friend.David) ∧ 
    ¬(¬(order Friend.Fiona Friend.Emma ∧ order Friend.Fiona Friend.David ∧ order Friend.Fiona Friend.George)) ∧
    ¬(∀ x : Friend, order Friend.George x) ∧
    (∃ x : Friend, order x Friend.David)))

-- State the theorem
theorem correct_age_order (order : AgeOrder) :
  ProblemConditions order →
  (order Friend.David Friend.Emma ∧
   order Friend.Emma Friend.George ∧
   order Friend.George Friend.Fiona) :=
by sorry

end correct_age_order_l1895_189537


namespace circle_area_ratio_when_diameter_tripled_l1895_189517

theorem circle_area_ratio_when_diameter_tripled :
  ∀ (r : ℝ), r > 0 →
  (π * r^2) / (π * (3*r)^2) = 1/9 := by
sorry

end circle_area_ratio_when_diameter_tripled_l1895_189517


namespace intersection_sum_l1895_189580

theorem intersection_sum (p q : ℝ) : 
  let M := {x : ℝ | x^2 - 5*x < 0}
  let N := {x : ℝ | p < x ∧ x < 6}
  ({x : ℝ | x ∈ M ∧ x ∈ N} = {x : ℝ | 2 < x ∧ x < q}) → p + q = 7 := by
  sorry

end intersection_sum_l1895_189580


namespace largest_angle_in_pentagon_l1895_189555

theorem largest_angle_in_pentagon (F G H I J : ℝ) : 
  F = 80 ∧ 
  G = 100 ∧ 
  H = I ∧ 
  J = 2 * H + 20 ∧ 
  F + G + H + I + J = 540 →
  max F (max G (max H (max I J))) = 190 :=
sorry

end largest_angle_in_pentagon_l1895_189555


namespace probability_penny_dime_same_different_dollar_l1895_189538

-- Define the coin types
inductive Coin
| Penny
| Nickel
| Dime
| Quarter
| Dollar

-- Define the possible outcomes for a coin flip
inductive FlipResult
| Heads
| Tails

-- Define a function to represent the result of flipping all coins
def CoinFlips := Coin → FlipResult

-- Define the condition for a successful outcome
def SuccessfulOutcome (flips : CoinFlips) : Prop :=
  (flips Coin.Penny = flips Coin.Dime) ∧ (flips Coin.Penny ≠ flips Coin.Dollar)

-- Define the total number of possible outcomes
def TotalOutcomes : ℕ := 2^5

-- Define the number of successful outcomes
def SuccessfulOutcomes : ℕ := 8

-- Theorem statement
theorem probability_penny_dime_same_different_dollar :
  (SuccessfulOutcomes : ℚ) / TotalOutcomes = 1 / 4 := by
  sorry

end probability_penny_dime_same_different_dollar_l1895_189538


namespace largest_common_divisor_fifteen_always_divides_l1895_189534

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def product (n : ℕ) : ℕ := n * (n+2) * (n+4) * (n+6) * (n+8)

theorem largest_common_divisor :
  ∀ (d : ℕ), d > 15 →
    ∃ (n : ℕ), is_odd n ∧ ¬(d ∣ product n) :=
sorry

theorem fifteen_always_divides :
  ∀ (n : ℕ), is_odd n → (15 ∣ product n) :=
sorry

end largest_common_divisor_fifteen_always_divides_l1895_189534


namespace parallel_lines_k_value_l1895_189551

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} : 
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 15 * x + 5 ↔ y = (5 * k) * x - 7) → k = 3 := by
  sorry

end parallel_lines_k_value_l1895_189551


namespace ocean_depth_for_specific_mountain_l1895_189566

/-- Represents a cone-shaped mountain partially submerged in water -/
structure SubmergedMountain where
  totalHeight : ℝ
  aboveWaterVolumeFraction : ℝ

/-- Calculates the depth of the ocean at the base of a submerged mountain -/
def oceanDepth (mountain : SubmergedMountain) : ℝ :=
  mountain.totalHeight * (1 - (1 - mountain.aboveWaterVolumeFraction) ^ (1/3))

/-- Theorem stating that for a specific mountain configuration, the ocean depth is 648 feet -/
theorem ocean_depth_for_specific_mountain :
  let mountain : SubmergedMountain := {
    totalHeight := 12000,
    aboveWaterVolumeFraction := 1/6
  }
  oceanDepth mountain = 648 := by sorry

end ocean_depth_for_specific_mountain_l1895_189566


namespace cartesian_angle_theorem_l1895_189514

/-- An angle in the Cartesian plane -/
structure CartesianAngle where
  -- The x-coordinate of the point on the terminal side
  x : ℝ
  -- The y-coordinate of the point on the terminal side
  y : ℝ
  -- The initial side is the non-negative half of the x-axis
  initial_side_positive_x : x > 0

/-- The theorem statement for the given problem -/
theorem cartesian_angle_theorem (α : CartesianAngle) 
  (h1 : α.x = 2) (h2 : α.y = 4) : 
  Real.tan (Real.arctan (α.y / α.x)) = 2 ∧ 
  (2 * Real.sin (Real.pi - Real.arctan (α.y / α.x)) + 
   2 * Real.cos (Real.arctan (α.y / α.x) / 2) ^ 2 - 1) / 
  (Real.sqrt 2 * Real.sin (Real.arctan (α.y / α.x) + Real.pi / 4)) = 2 * Real.sqrt 2 := by
  sorry

end cartesian_angle_theorem_l1895_189514


namespace factorial_1000_trailing_zeros_l1895_189550

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- Theorem: 1000! ends with 249 zeros -/
theorem factorial_1000_trailing_zeros :
  trailingZeros 1000 = 249 := by
  sorry

end factorial_1000_trailing_zeros_l1895_189550


namespace specific_coin_flip_probability_l1895_189557

/-- The probability of getting a specific sequence of heads and tails
    when flipping a fair coin multiple times. -/
def coin_flip_probability (n : ℕ) (k : ℕ) : ℚ :=
  (1 / 2) ^ n

theorem specific_coin_flip_probability :
  coin_flip_probability 5 2 = 1 / 32 := by
  sorry

end specific_coin_flip_probability_l1895_189557


namespace rectangle_area_with_perimeter_and_breadth_l1895_189590

/-- Theorem: Area of a rectangle with given perimeter and breadth -/
theorem rectangle_area_with_perimeter_and_breadth
  (perimeter : ℝ) (breadth : ℝ) (h_perimeter : perimeter = 900)
  (h_breadth : breadth = 190) :
  let length : ℝ := perimeter / 2 - breadth
  let area : ℝ := length * breadth
  area = 49400 := by sorry

end rectangle_area_with_perimeter_and_breadth_l1895_189590


namespace machines_for_hundred_books_l1895_189594

/-- The number of printing machines required to print a given number of books in a given number of days. -/
def machines_required (initial_machines : ℕ) (initial_books : ℕ) (initial_days : ℕ) 
                      (target_books : ℕ) (target_days : ℕ) : ℕ :=
  (target_books * initial_machines * initial_days) / (initial_books * target_days)

/-- Theorem stating that 5 machines are required to print 100 books in 100 days,
    given that 5 machines can print 5 books in 5 days. -/
theorem machines_for_hundred_books : 
  machines_required 5 5 5 100 100 = 5 := by
  sorry

#eval machines_required 5 5 5 100 100

end machines_for_hundred_books_l1895_189594


namespace function_properties_l1895_189509

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem function_properties (a : ℝ) (h : f a (π / 3) = 0) :
  (∃ T > 0, ∀ x, f a (x + T) = f a x ∧ ∀ S, 0 < S → S < T → ∃ y, f a (y + S) ≠ f a y) ∧
  (∀ y ∈ Set.Icc (π / 2) (3 * π / 2), -1 ≤ f a y ∧ f a y ≤ 2) ∧
  (∃ y₁ ∈ Set.Icc (π / 2) (3 * π / 2), f a y₁ = -1) ∧
  (∃ y₂ ∈ Set.Icc (π / 2) (3 * π / 2), f a y₂ = 2) :=
by sorry

end function_properties_l1895_189509


namespace complex_quadratic_roots_l1895_189511

theorem complex_quadratic_roots (z : ℂ) :
  z ^ 2 = -91 + 104 * I ∧ (7 + 10 * I) ^ 2 = -91 + 104 * I →
  z = 7 + 10 * I ∨ z = -7 - 10 * I :=
by sorry

end complex_quadratic_roots_l1895_189511


namespace no_such_function_l1895_189549

theorem no_such_function : ¬∃ f : ℤ → ℤ, ∀ m n : ℤ, f (m + f n) = f m - n := by
  sorry

end no_such_function_l1895_189549


namespace distance_difference_l1895_189584

def sprint_distance : ℝ := 0.88
def jog_distance : ℝ := 0.75

theorem distance_difference : sprint_distance - jog_distance = 0.13 := by
  sorry

end distance_difference_l1895_189584


namespace intersection_complement_theorem_l1895_189547

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | x^2 + 2*x < 0}

theorem intersection_complement_theorem :
  A ∩ (Set.univ \ B) = {-2, 0, 1, 2} := by sorry

end intersection_complement_theorem_l1895_189547


namespace calculation_proof_l1895_189505

theorem calculation_proof :
  (- (2^3 / 8) - (1/4 * (-2)^2) = -2) ∧
  ((-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21) := by
sorry

end calculation_proof_l1895_189505


namespace trigonometric_expression_simplification_l1895_189518

open Real

theorem trigonometric_expression_simplification (α : ℝ) :
  (sin (2 * π - α) * cos (π + α) * cos (π / 2 + α) * cos (11 * π / 2 - α)) /
  (cos (π - α) * sin (3 * π - α) * sin (-π - α) * sin (9 * π / 2 + α)) = -tan α :=
by sorry

end trigonometric_expression_simplification_l1895_189518


namespace xyz_divisible_by_55_l1895_189503

theorem xyz_divisible_by_55 (x y z a b c : ℤ) 
  (h1 : x^2 + y^2 = a^2) 
  (h2 : y^2 + z^2 = b^2) 
  (h3 : z^2 + x^2 = c^2) : 
  55 ∣ (x * y * z) := by
  sorry

end xyz_divisible_by_55_l1895_189503


namespace spinner_probability_l1895_189521

def spinner1 : Finset ℕ := {2, 4, 5, 7, 9}
def spinner2 : Finset ℕ := {3, 4, 6, 8, 10, 12}

def isEven (n : ℕ) : Bool := n % 2 = 0

def productIsEven (x : ℕ) (y : ℕ) : Bool := isEven (x * y)

def favorableOutcomes : ℕ := (spinner1.card * spinner2.card) - 
  (spinner1.filter (λ x => ¬isEven x)).card * (spinner2.filter (λ x => ¬isEven x)).card

theorem spinner_probability : 
  (favorableOutcomes : ℚ) / (spinner1.card * spinner2.card) = 9 / 10 := by
  sorry

end spinner_probability_l1895_189521


namespace right_triangle_median_to_hypotenuse_l1895_189513

theorem right_triangle_median_to_hypotenuse (DE DF EF : ℝ) :
  DE = 15 →
  DF = 9 →
  EF = 12 →
  DE^2 = DF^2 + EF^2 →
  (DE / 2 : ℝ) = 7.5 := by
  sorry

end right_triangle_median_to_hypotenuse_l1895_189513


namespace average_of_five_numbers_l1895_189581

variable (x : ℝ)

theorem average_of_five_numbers (x : ℝ) :
  let numbers := [-4*x, 0, 4*x, 12*x, 20*x]
  (numbers.sum / numbers.length : ℝ) = 6.4 * x :=
by sorry

end average_of_five_numbers_l1895_189581


namespace min_value_sum_reciprocals_l1895_189595

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 2) : 
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) ≥ 9 / 4 := by
sorry

end min_value_sum_reciprocals_l1895_189595


namespace binomial_coefficient_ratio_l1895_189599

theorem binomial_coefficient_ratio (n : ℕ) (k : ℕ) : 
  n = 14 ∧ k = 4 →
  (Nat.choose n k = 1001 ∧ 
   Nat.choose n (k+1) = 2002 ∧ 
   Nat.choose n (k+2) = 3003) ∧
  ∀ m : ℕ, m > 3 → 
    ¬(∃ j : ℕ, ∀ i : ℕ, i < m → 
      Nat.choose n (j+i+1) = (i+1) * Nat.choose n j) :=
by sorry

end binomial_coefficient_ratio_l1895_189599


namespace merged_class_size_and_rank_l1895_189532

/-- Represents a group of students with known positions from left and right -/
structure StudentGroup where
  leftPos : Nat
  rightPos : Nat

/-- Calculates the total number of students in a group -/
def groupSize (g : StudentGroup) : Nat :=
  g.leftPos + g.rightPos - 1

theorem merged_class_size_and_rank (groupA groupB groupC : StudentGroup)
  (hA : groupA = ⟨8, 13⟩)
  (hB : groupB = ⟨12, 10⟩)
  (hC : groupC = ⟨7, 6⟩) :
  let totalStudents := groupSize groupA + groupSize groupB + groupSize groupC
  let rankFromLeft := groupSize groupA + groupB.leftPos
  totalStudents = 53 ∧ rankFromLeft = 32 := by
  sorry

end merged_class_size_and_rank_l1895_189532


namespace divisible_by_six_l1895_189572

theorem divisible_by_six (n : ℤ) : ∃ k : ℤ, n^3 + 5*n = 6*k := by sorry

end divisible_by_six_l1895_189572


namespace least_b_value_l1895_189522

theorem least_b_value (a b : ℕ+) : 
  (∃ p : ℕ+, p.val.Prime ∧ p > 2 ∧ a = p^2) → -- a is the square of the next smallest prime after 2
  (Finset.card (Nat.divisors a) = 3) →        -- a has 3 factors
  (Finset.card (Nat.divisors b) = a) →        -- b has a factors
  (a ∣ b) →                                   -- b is divisible by a
  b ≥ 36 :=                                   -- the least possible value of b is 36
by sorry

end least_b_value_l1895_189522


namespace min_sphere_surface_area_l1895_189556

/-- Represents a cuboid with vertices on a sphere -/
structure CuboidOnSphere where
  -- The length of edge AB
  ab : ℝ
  -- The length of edge AD
  ad : ℝ
  -- The length of edge AA'
  aa' : ℝ
  -- The radius of the sphere
  r : ℝ
  -- All vertices are on the sphere
  vertices_on_sphere : ab ^ 2 + ad ^ 2 + aa' ^ 2 = (2 * r) ^ 2
  -- AB = 2
  ab_equals_two : ab = 2
  -- Volume of pyramid O-A'B'C'D' is 2
  pyramid_volume : (1 / 3) * ad * aa' = 2

/-- The minimum surface area of the sphere is 16π -/
theorem min_sphere_surface_area (c : CuboidOnSphere) : 
  ∃ (min_area : ℝ), min_area = 16 * π ∧ 
  ∀ (area : ℝ), area = 4 * π * c.r ^ 2 → area ≥ min_area := by
  sorry

end min_sphere_surface_area_l1895_189556


namespace chastity_initial_money_l1895_189570

def lollipop_cost : ℚ := 1.5
def gummies_cost : ℚ := 2
def lollipops_bought : ℕ := 4
def gummies_packs_bought : ℕ := 2
def money_left : ℚ := 5

def initial_money : ℚ := 15

theorem chastity_initial_money :
  initial_money = 
    (lollipop_cost * lollipops_bought + gummies_cost * gummies_packs_bought + money_left) :=
by sorry

end chastity_initial_money_l1895_189570


namespace arithmetic_sequence_third_term_l1895_189574

/-- 
An arithmetic sequence is a sequence where the difference between 
consecutive terms is constant.
-/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- 
Theorem: In an arithmetic sequence where the sum of the first and fifth terms is 10, 
the third term is equal to 5.
-/
theorem arithmetic_sequence_third_term 
  (a : ℕ → ℚ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 1 + a 5 = 10) : 
  a 3 = 5 := by
  sorry

end arithmetic_sequence_third_term_l1895_189574


namespace complex_equation_solution_l1895_189561

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 1 - Complex.I) : 
  z = -1 - Complex.I := by
  sorry

end complex_equation_solution_l1895_189561


namespace orange_ribbons_l1895_189524

theorem orange_ribbons (total : ℕ) (yellow purple orange silver : ℕ) : 
  yellow + purple + orange + silver = total →
  4 * yellow = total →
  3 * purple = total →
  6 * orange = total →
  silver = 40 →
  orange = 27 := by
sorry

end orange_ribbons_l1895_189524


namespace max_amount_C_is_correct_l1895_189592

/-- Represents the maximum amount of 11% saline solution (C) that can be used
    to prepare 100 kg of 7% saline solution, given 3% (A) and 8% (B) solutions
    are also available. -/
def maxAmountC : ℝ := 50

/-- The concentration of saline solution A -/
def concentrationA : ℝ := 0.03

/-- The concentration of saline solution B -/
def concentrationB : ℝ := 0.08

/-- The concentration of saline solution C -/
def concentrationC : ℝ := 0.11

/-- The target concentration of the final solution -/
def targetConcentration : ℝ := 0.07

/-- The total amount of the final solution -/
def totalAmount : ℝ := 100

theorem max_amount_C_is_correct :
  ∃ (y : ℝ),
    0 ≤ y ∧
    0 ≤ (totalAmount - maxAmountC - y) ∧
    concentrationC * maxAmountC + concentrationB * y +
      concentrationA * (totalAmount - maxAmountC - y) =
    targetConcentration * totalAmount ∧
    ∀ (x : ℝ),
      x > maxAmountC →
      ¬∃ (z : ℝ),
        0 ≤ z ∧
        0 ≤ (totalAmount - x - z) ∧
        concentrationC * x + concentrationB * z +
          concentrationA * (totalAmount - x - z) =
        targetConcentration * totalAmount :=
by sorry

end max_amount_C_is_correct_l1895_189592


namespace circles_externally_tangent_l1895_189526

/-- Two circles with radii R and r, where R and r are the roots of x^2 - 3x + 2 = 0,
    and whose centers are at a distance d = 3 apart, are externally tangent. -/
theorem circles_externally_tangent (R r : ℝ) (d : ℝ) : 
  (R^2 - 3*R + 2 = 0) → 
  (r^2 - 3*r + 2 = 0) → 
  (d = 3) → 
  (R + r = d) := by sorry

end circles_externally_tangent_l1895_189526


namespace fraction_of_girls_at_event_l1895_189563

theorem fraction_of_girls_at_event (maplewood_total : ℕ) (brookside_total : ℕ)
  (maplewood_boy_ratio maplewood_girl_ratio : ℕ)
  (brookside_boy_ratio brookside_girl_ratio : ℕ)
  (h1 : maplewood_total = 300)
  (h2 : brookside_total = 240)
  (h3 : maplewood_boy_ratio = 3)
  (h4 : maplewood_girl_ratio = 2)
  (h5 : brookside_boy_ratio = 2)
  (h6 : brookside_girl_ratio = 3) :
  (maplewood_total * maplewood_girl_ratio / (maplewood_boy_ratio + maplewood_girl_ratio) +
   brookside_total * brookside_girl_ratio / (brookside_boy_ratio + brookside_girl_ratio)) /
  (maplewood_total + brookside_total) = 22 / 45 := by
  sorry

end fraction_of_girls_at_event_l1895_189563


namespace palm_meadows_beds_l1895_189508

theorem palm_meadows_beds (total_rooms : ℕ) (rooms_with_fewer_beds : ℕ) (beds_in_other_rooms : ℕ) (total_beds : ℕ) :
  total_rooms = 13 →
  rooms_with_fewer_beds = 8 →
  total_rooms - rooms_with_fewer_beds = 5 →
  beds_in_other_rooms = 3 →
  total_beds = 31 →
  (rooms_with_fewer_beds * 2) + ((total_rooms - rooms_with_fewer_beds) * beds_in_other_rooms) = total_beds :=
by
  sorry

end palm_meadows_beds_l1895_189508


namespace x_squared_minus_y_squared_l1895_189596

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5 / 11) 
  (h2 : x - y = 1 / 101) : 
  x^2 - y^2 = 5 / 1111 := by
  sorry

end x_squared_minus_y_squared_l1895_189596


namespace profit_decrease_for_one_loom_l1895_189507

/-- Represents the profit decrease when one loom breaks down for a month -/
def profit_decrease (num_looms : ℕ) (total_sales : ℕ) (manufacturing_expenses : ℕ) (establishment_charges : ℕ) : ℕ :=
  let sales_per_loom := total_sales / num_looms
  let manufacturing_per_loom := manufacturing_expenses / num_looms
  let establishment_per_loom := establishment_charges / num_looms
  sales_per_loom - manufacturing_per_loom - establishment_per_loom

/-- Theorem stating the profit decrease when one loom breaks down for a month -/
theorem profit_decrease_for_one_loom :
  profit_decrease 125 500000 150000 75000 = 2200 := by
  sorry

#eval profit_decrease 125 500000 150000 75000

end profit_decrease_for_one_loom_l1895_189507
