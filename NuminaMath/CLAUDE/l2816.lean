import Mathlib

namespace max_candy_leftover_l2816_281631

theorem max_candy_leftover (x : ℕ) : ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
sorry

end max_candy_leftover_l2816_281631


namespace no_valid_seating_l2816_281606

/-- A seating arrangement of deputies around a circular table. -/
structure Seating :=
  (deputies : Fin 47 → Fin 12)

/-- The property that any 15 consecutive deputies include all 12 regions. -/
def hasAllRegionsIn15 (s : Seating) : Prop :=
  ∀ start : Fin 47, ∃ (f : Fin 12 → Fin 15), ∀ r : Fin 12,
    ∃ i : Fin 15, s.deputies ((start + i) % 47) = r

/-- Theorem stating that no valid seating arrangement exists. -/
theorem no_valid_seating : ¬ ∃ s : Seating, hasAllRegionsIn15 s := by
  sorry

end no_valid_seating_l2816_281606


namespace bryce_raisins_l2816_281694

theorem bryce_raisins (x : ℕ) : 
  (x - 6 = x / 2) → x = 12 := by
  sorry

end bryce_raisins_l2816_281694


namespace sqrt_square_eq_abs_sqrt_neg_seven_squared_l2816_281670

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by sorry

theorem sqrt_neg_seven_squared : Real.sqrt ((-7)^2) = 7 := by sorry

end sqrt_square_eq_abs_sqrt_neg_seven_squared_l2816_281670


namespace faster_train_length_l2816_281630

/-- Given two trains moving in the same direction, this theorem calculates the length of the faster train. -/
theorem faster_train_length (v_fast v_slow : ℝ) (t_cross : ℝ) (h1 : v_fast = 72) (h2 : v_slow = 36) (h3 : t_cross = 18) :
  (v_fast - v_slow) * (5 / 18) * t_cross = 180 :=
by sorry

end faster_train_length_l2816_281630


namespace problem_solution_l2816_281656

theorem problem_solution :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 7 ∧ x2 = 2 - Real.sqrt 7 ∧
    x1^2 - 4*x1 - 3 = 0 ∧ x2^2 - 4*x2 - 3 = 0) ∧
  (abs (-3) - 4 * Real.sin (π/4) + Real.sqrt 8 + (π - 3)^0 = 4) := by
  sorry

end problem_solution_l2816_281656


namespace max_value_fraction_l2816_281665

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (x * y) / (x + 8*y) ≤ 1/18 :=
by sorry

end max_value_fraction_l2816_281665


namespace problem_1_l2816_281676

theorem problem_1 : 
  Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 48 + Real.sqrt 54 = 4 + Real.sqrt 6 := by
  sorry

end problem_1_l2816_281676


namespace polynomial_with_no_integer_roots_but_modular_roots_l2816_281661

/-- Definition of the polynomial P(x) = (x³ + 3)(x² + 1)(x² + 2)(x² - 2) -/
def P (x : ℤ) : ℤ := (x^3 + 3) * (x^2 + 1) * (x^2 + 2) * (x^2 - 2)

/-- Theorem stating the existence of a polynomial with the required properties -/
theorem polynomial_with_no_integer_roots_but_modular_roots :
  (∀ x : ℤ, P x ≠ 0) ∧
  (∀ n : ℕ, n > 0 → ∃ x : ℤ, P x % n = 0) := by
  sorry

end polynomial_with_no_integer_roots_but_modular_roots_l2816_281661


namespace expression_evaluation_l2816_281645

theorem expression_evaluation : 3 * 3^4 - 9^20 / 9^18 + 5^3 = 287 := by
  sorry

end expression_evaluation_l2816_281645


namespace quadratic_equation_distinct_roots_quadratic_equation_distinct_roots_2_l2816_281620

theorem quadratic_equation_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + (a + 1) * x - 2 = 0 ∧ a * y^2 + (a + 1) * y - 2 = 0) ↔
  (a < -5 - 2 * Real.sqrt 6 ∨ (-5 + 2 * Real.sqrt 6 < a ∧ a < 0) ∨ a > 0) :=
sorry

theorem quadratic_equation_distinct_roots_2 (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (1 - a) * x^2 + (a + 1) * x - 2 = 0 ∧ (1 - a) * y^2 + (a + 1) * y - 2 = 0) ↔
  (a < 1 ∨ (1 < a ∧ a < 3) ∨ a > 3) :=
sorry

end quadratic_equation_distinct_roots_quadratic_equation_distinct_roots_2_l2816_281620


namespace multiplication_division_sum_l2816_281681

theorem multiplication_division_sum : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end multiplication_division_sum_l2816_281681


namespace magic_square_constant_l2816_281644

def MagicSquare (a b c d e f g h i : ℕ) : Prop :=
  a + b + c = d + e + f ∧
  d + e + f = g + h + i ∧
  a + d + g = b + e + h ∧
  b + e + h = c + f + i ∧
  a + e + i = c + e + g

theorem magic_square_constant (a b c d e f g h i : ℕ) :
  MagicSquare a b c d e f g h i →
  a = 12 → c = 4 → d = 7 → h = 1 →
  a + b + c = 15 :=
sorry

end magic_square_constant_l2816_281644


namespace no_natural_square_diff_2014_l2816_281658

theorem no_natural_square_diff_2014 : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end no_natural_square_diff_2014_l2816_281658


namespace remaining_investment_rate_l2816_281623

-- Define the investment amounts and rates
def total_investment : ℝ := 12000
def investment1_amount : ℝ := 5000
def investment1_rate : ℝ := 0.03
def investment2_amount : ℝ := 4000
def investment2_rate : ℝ := 0.045
def desired_income : ℝ := 600

-- Define the remaining investment amount
def remaining_investment : ℝ := total_investment - (investment1_amount + investment2_amount)

-- Define the income from the first two investments
def known_income : ℝ := investment1_amount * investment1_rate + investment2_amount * investment2_rate

-- Define the required income from the remaining investment
def required_income : ℝ := desired_income - known_income

-- Theorem to prove
theorem remaining_investment_rate : 
  (required_income / remaining_investment) = 0.09 := by sorry

end remaining_investment_rate_l2816_281623


namespace a_divides_b_l2816_281625

theorem a_divides_b (a b : ℕ) (h1 : a > 1) (h2 : b > 1)
  (r : ℕ → ℕ)
  (h3 : ∀ n : ℕ, n > 0 → r n = b^n % a^n)
  (h4 : ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (r n : ℚ) < 2^n / n) :
  a ∣ b :=
by sorry

end a_divides_b_l2816_281625


namespace area_inside_EFG_outside_AFD_l2816_281612

/-- Square ABCD with side length 36 -/
def square_side_length : ℝ := 36

/-- Point E is on side AB, 12 units from B -/
def distance_E_from_B : ℝ := 12

/-- Point F is the midpoint of side BC -/
def F_is_midpoint : Prop := True

/-- Point G is on side CD, 12 units from C -/
def distance_G_from_C : ℝ := 12

/-- The area of the region inside triangle EFG and outside triangle AFD -/
def area_difference : ℝ := 0

theorem area_inside_EFG_outside_AFD :
  square_side_length = 36 →
  distance_E_from_B = 12 →
  F_is_midpoint →
  distance_G_from_C = 12 →
  area_difference = 0 := by
  sorry

end area_inside_EFG_outside_AFD_l2816_281612


namespace sausage_problem_l2816_281641

/-- Represents the sausage problem --/
theorem sausage_problem (total_meat : ℕ) (total_links : ℕ) (remaining_meat : ℕ) 
  (h1 : total_meat = 10) 
  (h2 : total_links = 40)
  (h3 : remaining_meat = 112) : 
  (total_meat * 16 - remaining_meat) / (total_meat * 16 / total_links) = 12 := by
  sorry

#check sausage_problem

end sausage_problem_l2816_281641


namespace chicken_pieces_needed_l2816_281679

/-- Represents the number of pieces of chicken used in different orders -/
structure ChickenPieces where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- Represents the number of orders for each type of dish -/
structure Orders where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- Calculates the total number of chicken pieces needed for all orders -/
def totalChickenPieces (pieces : ChickenPieces) (orders : Orders) : ℕ :=
  pieces.pasta * orders.pasta +
  pieces.barbecue * orders.barbecue +
  pieces.friedDinner * orders.friedDinner

/-- Theorem stating that given the specific conditions, 37 pieces of chicken are needed -/
theorem chicken_pieces_needed :
  let pieces := ChickenPieces.mk 2 3 8
  let orders := Orders.mk 6 3 2
  totalChickenPieces pieces orders = 37 := by
  sorry


end chicken_pieces_needed_l2816_281679


namespace gcd_1515_600_l2816_281667

theorem gcd_1515_600 : Nat.gcd 1515 600 = 15 := by
  sorry

end gcd_1515_600_l2816_281667


namespace parallel_squares_theorem_l2816_281632

/-- Two squares with parallel sides -/
structure ParallelSquares where
  a : ℝ  -- Side length of the first square
  b : ℝ  -- Side length of the second square
  a_pos : 0 < a
  b_pos : 0 < b

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of an equilateral triangle -/
def is_equilateral (p q r : Point) : Prop :=
  (p.x - q.x)^2 + (p.y - q.y)^2 = (q.x - r.x)^2 + (q.y - r.y)^2 ∧
  (q.x - r.x)^2 + (q.y - r.y)^2 = (r.x - p.x)^2 + (r.y - p.y)^2

/-- The set of points M satisfying the condition -/
def valid_points (squares : ParallelSquares) : Set Point :=
  {m : Point | ∀ p : Point, p.x ∈ [-squares.a/2, squares.a/2] ∧ p.y ∈ [-squares.a/2, squares.a/2] →
    ∃ q : Point, q.x ∈ [-squares.b/2, squares.b/2] ∧ q.y ∈ [-squares.b/2, squares.b/2] ∧
    is_equilateral m p q}

/-- The main theorem -/
theorem parallel_squares_theorem (squares : ParallelSquares) :
  (valid_points squares).Nonempty ↔ squares.b ≥ (squares.a / 2) * (Real.sqrt 3 + 1) :=
sorry

end parallel_squares_theorem_l2816_281632


namespace geometric_sequence_sum_l2816_281653

/-- Given a geometric sequence {a_n} with common ratio q ≠ 1, where a₁a₂a₃ = -1/8
    and a₂, a₄, a₃ form an arithmetic sequence, the sum of the first 4 terms
    of the sequence {a_n} is equal to 5/8. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ, a (n + 1) = a n * q) →
  a 1 * a 2 * a 3 = -1/8 →
  2 * a 4 = a 2 + a 3 →
  (a 1 + a 2 + a 3 + a 4 : ℝ) = 5/8 := by
  sorry

end geometric_sequence_sum_l2816_281653


namespace max_value_of_f_l2816_281634

open Real

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt (x^4 - 3*x^2 - 6*x + 13) - Real.sqrt (x^4 - x^2 + 1)

theorem max_value_of_f : 
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = Real.sqrt 10 := by
  sorry

end max_value_of_f_l2816_281634


namespace right_triangle_area_l2816_281697

theorem right_triangle_area (a c : ℝ) (h1 : a = 30) (h2 : c = 34) : 
  let b := Real.sqrt (c^2 - a^2)
  (1/2) * a * b = 240 := by
  sorry

end right_triangle_area_l2816_281697


namespace first_valve_fills_in_two_hours_l2816_281614

/-- Represents the time in hours taken by the first valve to fill the pool -/
def first_valve_fill_time (pool_capacity : ℝ) (both_valves_fill_time : ℝ) (valve_difference : ℝ) : ℝ :=
  2

/-- Theorem stating that under given conditions, the first valve takes 2 hours to fill the pool -/
theorem first_valve_fills_in_two_hours 
  (pool_capacity : ℝ) 
  (both_valves_fill_time : ℝ) 
  (valve_difference : ℝ) 
  (h1 : pool_capacity = 12000)
  (h2 : both_valves_fill_time = 48 / 60) -- Convert 48 minutes to hours
  (h3 : valve_difference = 50) :
  first_valve_fill_time pool_capacity both_valves_fill_time valve_difference = 2 := by
  sorry

#check first_valve_fills_in_two_hours

end first_valve_fills_in_two_hours_l2816_281614


namespace clock_hands_angle_l2816_281663

-- Define the initial speeds of the hands
def initial_hour_hand_speed : ℝ := 0.5
def initial_minute_hand_speed : ℝ := 6

-- Define the swapped speeds
def swapped_hour_hand_speed : ℝ := initial_minute_hand_speed
def swapped_minute_hand_speed : ℝ := initial_hour_hand_speed

-- Define the starting position (3 PM)
def starting_hour_position : ℝ := 90
def starting_minute_position : ℝ := 0

-- Define the target position (4 o'clock)
def target_hour_position : ℝ := 120

-- Theorem statement
theorem clock_hands_angle :
  let time_to_target := (target_hour_position - starting_hour_position) / swapped_hour_hand_speed
  let final_minute_position := starting_minute_position + swapped_minute_hand_speed * time_to_target
  let angle := target_hour_position - final_minute_position
  min angle (360 - angle) = 117.5 := by sorry

end clock_hands_angle_l2816_281663


namespace gain_percentage_proof_l2816_281629

theorem gain_percentage_proof (C S : ℝ) (h : 80 * C = 25 * S) : 
  (S - C) / C * 100 = 220 := by
sorry

end gain_percentage_proof_l2816_281629


namespace expensive_rock_cost_l2816_281647

/-- Given a mixture of two types of rock, prove the cost of the more expensive rock -/
theorem expensive_rock_cost 
  (total_weight : ℝ) 
  (total_cost : ℝ) 
  (cheap_rock_cost : ℝ) 
  (cheap_rock_weight : ℝ) 
  (expensive_rock_weight : ℝ)
  (h1 : total_weight = 24)
  (h2 : total_cost = 800)
  (h3 : cheap_rock_cost = 30)
  (h4 : cheap_rock_weight = 8)
  (h5 : expensive_rock_weight = 8)
  : (total_cost - cheap_rock_cost * cheap_rock_weight) / (total_weight - cheap_rock_weight) = 35 := by
  sorry

end expensive_rock_cost_l2816_281647


namespace area_equality_iff_concyclic_l2816_281668

-- Define the triangle ABC
variable (A B C : Point)

-- Define the altitudes and their intersection
variable (U V W H : Point)

-- Define points X, Y, Z on the altitudes
variable (X Y Z : Point)

-- Define the property of being an acute-angled triangle
def is_acute_angled (A B C : Point) : Prop := sorry

-- Define the property of a point being on a line segment
def on_segment (P Q R : Point) : Prop := sorry

-- Define the property of points being different
def are_different (P Q : Point) : Prop := sorry

-- Define the property of points being concyclic
def are_concyclic (P Q R S : Point) : Prop := sorry

-- Define the area of a triangle
def area (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem area_equality_iff_concyclic :
  is_acute_angled A B C →
  on_segment A U H → on_segment B V H → on_segment C W H →
  on_segment A U X → on_segment B V Y → on_segment C W Z →
  are_different X H → are_different Y H → are_different Z H →
  (are_concyclic X Y Z H ↔ area A B C = area A B Z + area A Y C + area X B C) :=
by sorry

end area_equality_iff_concyclic_l2816_281668


namespace square_of_negative_two_l2816_281619

theorem square_of_negative_two : (-2)^2 = 4 := by
  sorry

end square_of_negative_two_l2816_281619


namespace number_in_interval_l2816_281640

theorem number_in_interval (y : ℝ) (h : y = (1/y) * (-y) + 5) : 2 < y ∧ y ≤ 4 := by
  sorry

end number_in_interval_l2816_281640


namespace inequality_system_no_solution_l2816_281666

/-- The inequality system has no solution if and only if a ≥ -1 -/
theorem inequality_system_no_solution (a : ℝ) : 
  (∀ x : ℝ, ¬(x < a - 3 ∧ x + 2 > 2 * a)) ↔ a ≥ -1 := by
  sorry

end inequality_system_no_solution_l2816_281666


namespace charity_book_donation_l2816_281615

theorem charity_book_donation (initial_books : ℕ) (donors : ℕ) (borrowed_books : ℕ) (final_books : ℕ)
  (h1 : initial_books = 300)
  (h2 : donors = 10)
  (h3 : borrowed_books = 140)
  (h4 : final_books = 210) :
  (final_books + borrowed_books - initial_books) / donors = 5 := by
  sorry

end charity_book_donation_l2816_281615


namespace basketball_tryouts_l2816_281649

theorem basketball_tryouts (girls : ℕ) (called_back : ℕ) (didnt_make_cut : ℕ) : girls = 39 → called_back = 26 → didnt_make_cut = 17 → girls + (called_back + didnt_make_cut - girls) = 43 := by
  sorry

end basketball_tryouts_l2816_281649


namespace units_digit_3_2009_l2816_281691

def units_digit (n : ℕ) : ℕ := n % 10

def power_3_units_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

theorem units_digit_3_2009 : units_digit (3^2009) = 3 := by
  sorry

end units_digit_3_2009_l2816_281691


namespace max_third_term_is_15_l2816_281674

/-- An arithmetic sequence of four positive integers with sum 46 -/
structure ArithmeticSequence where
  a : ℕ+  -- first term
  d : ℕ   -- common difference
  sum_eq_46 : a + (a + d) + (a + 2*d) + (a + 3*d) = 46

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℕ :=
  seq.a + 2 * seq.d

/-- Theorem: The maximum possible value of the third term is 15 -/
theorem max_third_term_is_15 :
  ∀ seq : ArithmeticSequence, third_term seq ≤ 15 ∧ ∃ seq : ArithmeticSequence, third_term seq = 15 :=
sorry

end max_third_term_is_15_l2816_281674


namespace average_value_iff_m_in_zero_two_l2816_281684

/-- A function f has an average value on [a, b] if there exists x₀ ∈ (a, b) such that
    f(x₀) = (f(b) - f(a)) / (b - a) -/
def has_average_value (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ : ℝ, a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

/-- The quadratic function f(x) = -x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x + 1

theorem average_value_iff_m_in_zero_two :
  ∀ m : ℝ, has_average_value (f m) (-1) 1 ↔ 0 < m ∧ m < 2 := by sorry

end average_value_iff_m_in_zero_two_l2816_281684


namespace hyperbola_other_asymptote_l2816_281660

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- X-coordinate of the foci -/
  foci_x : ℝ

/-- Given a hyperbola, returns its other asymptote -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ :=
  fun x ↦ 2 * x + 16

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x ↦ -2 * x + 4)
  (h2 : h.foci_x = -3) :
  other_asymptote h = fun x ↦ 2 * x + 16 := by
  sorry

end hyperbola_other_asymptote_l2816_281660


namespace pyramid_volume_l2816_281621

/-- The volume of a pyramid with a rectangular base and given edge length --/
theorem pyramid_volume (base_length base_width edge_length : ℝ) 
  (h_base_length : base_length = 6)
  (h_base_width : base_width = 8)
  (h_edge_length : edge_length = 13) : 
  (1 / 3 : ℝ) * base_length * base_width * 
    Real.sqrt (edge_length^2 - ((base_length^2 + base_width^2) / 4)) = 192 := by
  sorry

#check pyramid_volume

end pyramid_volume_l2816_281621


namespace cricket_team_average_age_l2816_281672

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (team_average : ℝ) 
  (captain_age_diff : ℝ) 
  (remaining_average_diff : ℝ) :
  team_size = 15 →
  team_average = 28 →
  captain_age_diff = 4 →
  remaining_average_diff = 2 →
  (team_size : ℝ) * team_average = 
    ((team_size - 2 : ℝ) * (team_average - remaining_average_diff)) + 
    (team_average + captain_age_diff) + 
    (team_average * team_size - ((team_size - 2 : ℝ) * (team_average - remaining_average_diff)) - 
    (team_average + captain_age_diff)) →
  team_average = 28 := by
sorry

end cricket_team_average_age_l2816_281672


namespace min_value_3x_plus_2y_min_value_attained_l2816_281669

theorem min_value_3x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x = 4 * x * y - 2 * y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a = 4 * a * b - 2 * b → 3 * x + 2 * y ≤ 3 * a + 2 * b :=
by sorry

theorem min_value_attained (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x = 4 * x * y - 2 * y) :
  3 * x + 2 * y = 2 + Real.sqrt 3 ↔ x = (3 + Real.sqrt 3) / 6 ∧ y = (Real.sqrt 3 + 1) / 4 :=
by sorry

end min_value_3x_plus_2y_min_value_attained_l2816_281669


namespace base_for_216_four_digits_l2816_281643

def has_exactly_four_digits (b : ℕ) (n : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

theorem base_for_216_four_digits :
  ∃! b : ℕ, b > 1 ∧ has_exactly_four_digits b 216 :=
by
  sorry

end base_for_216_four_digits_l2816_281643


namespace average_of_data_l2816_281652

def data : List ℕ := [5, 6, 5, 6, 4, 4]

theorem average_of_data : (data.sum : ℚ) / data.length = 5 := by
  sorry

end average_of_data_l2816_281652


namespace imaginary_part_z_2017_l2816_281683

theorem imaginary_part_z_2017 : 
  let i : ℂ := Complex.I
  let z : ℂ := (1 + i) / (1 - i)
  Complex.im (z^2017) = Complex.im i := by sorry

end imaginary_part_z_2017_l2816_281683


namespace unique_solution_fifth_root_equation_l2816_281699

theorem unique_solution_fifth_root_equation (x : ℝ) :
  (((x^3 + 2*x)^(1/5) = (x^5 - 2*x)^(1/3)) ↔ (x = 0)) := by
  sorry

end unique_solution_fifth_root_equation_l2816_281699


namespace foundation_digging_l2816_281605

/-- Represents the work rate for digging a foundation -/
def work_rate (men : ℕ) (days : ℝ) : ℝ := men * days

theorem foundation_digging 
  (men_first_half : ℕ) (days_first_half : ℝ) 
  (men_second_half : ℕ) :
  men_first_half = 10 →
  days_first_half = 6 →
  men_second_half = 20 →
  work_rate men_first_half days_first_half = work_rate men_second_half 3 :=
by sorry

end foundation_digging_l2816_281605


namespace three_number_problem_l2816_281692

theorem three_number_problem :
  ∃ (X Y Z : ℤ),
    (X = (35 * X) / 100 + 60) ∧
    (X = (70 * Y) / 200 + Y / 2) ∧
    (Y = 2 * Z^2) ∧
    (X = 92) ∧
    (Y = 108) ∧
    (Z = 7) := by
  sorry

end three_number_problem_l2816_281692


namespace total_hours_worked_l2816_281655

/-- Represents the hours worked by Thomas, Toby, and Rebecca in one week -/
structure WorkHours where
  thomas : ℕ
  toby : ℕ
  rebecca : ℕ

/-- Calculates the total hours worked by all three people -/
def totalHours (h : WorkHours) : ℕ :=
  h.thomas + h.toby + h.rebecca

/-- Theorem stating the total hours worked given the conditions -/
theorem total_hours_worked :
  ∀ h : WorkHours,
    (∃ x : ℕ, h.thomas = x ∧
              h.toby = 2 * x - 10 ∧
              h.rebecca = h.toby - 8 ∧
              h.rebecca = 56) →
    totalHours h = 157 := by
  sorry

end total_hours_worked_l2816_281655


namespace modulus_of_z_l2816_281673

theorem modulus_of_z (z : ℂ) (h : z + 3*I = 3 - I) : Complex.abs z = 5 := by
  sorry

end modulus_of_z_l2816_281673


namespace simplify_and_evaluate_l2816_281642

theorem simplify_and_evaluate (m : ℚ) (h : m = 2) : 
  ((2 * m + 1) / m - 1) / ((m^2 - 1) / m) = 1 := by
  sorry

end simplify_and_evaluate_l2816_281642


namespace sqrt_product_equality_l2816_281685

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l2816_281685


namespace problem_solution_l2816_281698

theorem problem_solution : ∃ x : ℝ, (6000 - (x / 21)) = 5995 ∧ x = 105 := by
  sorry

end problem_solution_l2816_281698


namespace triangle_projection_relation_l2816_281675

/-- In a triangle with sides a, b, and c, where a > b > c and a = 2(b - c),
    and p is the projection of side c onto a, the equation 4c + 8p = 3a holds. -/
theorem triangle_projection_relation (a b c p : ℝ) : 
  a > b → b > c → a = 2*(b - c) → 4*c + 8*p = 3*a := by
  sorry

end triangle_projection_relation_l2816_281675


namespace internet_rate_proof_l2816_281680

/-- The regular monthly internet rate without discount -/
def regular_rate : ℝ := 50

/-- The discounted rate as a fraction of the regular rate -/
def discount_rate : ℝ := 0.95

/-- The number of months -/
def num_months : ℕ := 4

/-- The total payment for the given number of months -/
def total_payment : ℝ := 190

theorem internet_rate_proof : 
  regular_rate * discount_rate * num_months = total_payment := by
  sorry

#check internet_rate_proof

end internet_rate_proof_l2816_281680


namespace inequality_system_solution_l2816_281662

theorem inequality_system_solution :
  (∀ x : ℝ, 2 - x ≥ (x - 1) / 3 - 1 ↔ x ≤ 2.5) ∧
  ¬∃ x : ℝ, (5 * x + 1 < 3 * (x - 1)) ∧ ((x + 8) / 5 < (2 * x - 5) / 3 - 1) :=
by sorry

end inequality_system_solution_l2816_281662


namespace fred_marble_count_l2816_281607

/-- The number of blue marbles Tim has -/
def tim_marbles : ℕ := 5

/-- The factor by which Fred has more marbles than Tim -/
def fred_factor : ℕ := 22

/-- The number of blue marbles Fred has -/
def fred_marbles : ℕ := tim_marbles * fred_factor

theorem fred_marble_count : fred_marbles = 110 := by
  sorry

end fred_marble_count_l2816_281607


namespace smallest_m_for_nth_root_in_T_l2816_281609

def T : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_root_in_T : 
  (∀ n : ℕ+, n ≥ 12 → ∃ z ∈ T, z ^ (n : ℕ) = 1) ∧ 
  (∀ m : ℕ+, m < 12 → ∃ n : ℕ+, n ≥ m ∧ ∀ z ∈ T, z ^ (n : ℕ) ≠ 1) :=
sorry

end smallest_m_for_nth_root_in_T_l2816_281609


namespace union_of_M_and_N_l2816_281696

def M : Set ℝ := {x | x^2 - 6*x + 5 = 0}
def N : Set ℝ := {x | x^2 - 5*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 5} := by sorry

end union_of_M_and_N_l2816_281696


namespace smaller_number_problem_l2816_281616

theorem smaller_number_problem (x y : ℝ) 
  (sum_eq : x + y = 18) 
  (diff_eq : x - y = 8) : 
  min x y = 5 := by sorry

end smaller_number_problem_l2816_281616


namespace star_associativity_l2816_281690

universe u

variable {U : Type u}

def star (X Y : Set U) : Set U := X ∩ Y

theorem star_associativity (X Y Z : Set U) : star (star X Y) Z = (X ∩ Y) ∩ Z := by
  sorry

end star_associativity_l2816_281690


namespace quadratic_equation_roots_l2816_281633

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*x₁ - 2*m + 5 = 0 ∧ 
    x₂^2 - 4*x₂ - 2*m + 5 = 0 ∧
    x₁*x₂ + x₁ + x₂ = m^2 + 6) →
  m = 1 ∧ m ≥ (1/2 : ℝ) :=
by sorry

end quadratic_equation_roots_l2816_281633


namespace tony_age_at_end_of_period_l2816_281693

/-- Represents Tony's age at the start of the period -/
def initial_age : ℕ := 14

/-- Represents Tony's age at the end of the period -/
def final_age : ℕ := initial_age + 1

/-- Represents the number of days Tony worked at his initial age -/
def days_at_initial_age : ℕ := 46

/-- Represents the number of days Tony worked at his final age -/
def days_at_final_age : ℕ := 100 - days_at_initial_age

/-- Represents Tony's daily earnings at a given age -/
def daily_earnings (age : ℕ) : ℚ := 1.9 * age

/-- Represents Tony's total earnings during the period -/
def total_earnings : ℚ := 3750

theorem tony_age_at_end_of_period :
  final_age = 15 ∧
  days_at_initial_age + days_at_final_age = 100 ∧
  daily_earnings initial_age * days_at_initial_age +
  daily_earnings final_age * days_at_final_age = total_earnings :=
sorry

end tony_age_at_end_of_period_l2816_281693


namespace perfect_square_condition_l2816_281626

theorem perfect_square_condition (n : ℕ+) :
  (∃ m : ℕ, n^4 + 2*n^3 + 5*n^2 + 12*n + 5 = m^2) ↔ (n = 1 ∨ n = 2) := by
  sorry

end perfect_square_condition_l2816_281626


namespace logarithm_sum_simplification_l2816_281603

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 9 / Real.log 18 + 1) = 7 / 4 := by
  sorry

end logarithm_sum_simplification_l2816_281603


namespace melanie_cats_count_l2816_281618

theorem melanie_cats_count (jacob_cats : ℕ) (annie_cats : ℕ) (melanie_cats : ℕ)
  (h1 : jacob_cats = 90)
  (h2 : annie_cats * 3 = jacob_cats)
  (h3 : melanie_cats = annie_cats * 2) :
  melanie_cats = 60 := by
  sorry

end melanie_cats_count_l2816_281618


namespace farm_area_calculation_l2816_281610

/-- Calculates the area of a rectangular farm given its length and width ratio. -/
def farm_area (length : ℝ) (width_ratio : ℝ) : ℝ :=
  length * (width_ratio * length)

/-- Theorem stating that a rectangular farm with length 0.6 km and width three times its length has an area of 1.08 km². -/
theorem farm_area_calculation :
  farm_area 0.6 3 = 1.08 := by
  sorry

end farm_area_calculation_l2816_281610


namespace band_problem_solution_l2816_281686

def band_problem (num_flutes num_clarinets num_trumpets num_total : ℕ) 
  (flute_ratio clarinet_ratio trumpet_ratio pianist_ratio : ℚ) : Prop :=
  let flutes_in := (num_flutes : ℚ) * flute_ratio
  let clarinets_in := (num_clarinets : ℚ) * clarinet_ratio
  let trumpets_in := (num_trumpets : ℚ) * trumpet_ratio
  let non_pianists_in := flutes_in + clarinets_in + trumpets_in
  let pianists_in := (num_total : ℚ) - non_pianists_in
  ∃ (num_pianists : ℕ), (num_pianists : ℚ) * pianist_ratio = pianists_in ∧ num_pianists = 20

theorem band_problem_solution :
  band_problem 20 30 60 53 (4/5) (1/2) (1/3) (1/10) :=
sorry

end band_problem_solution_l2816_281686


namespace constant_term_in_expansion_l2816_281635

theorem constant_term_in_expansion (n : ℕ) (h : n > 0) 
  (h_coeff : (n.choose 2) - (n.choose 1) = 44) : 
  let general_term (r : ℕ) := (n.choose r) * (33 - 11 * r) / 2
  ∃ (k : ℕ), k = 4 ∧ general_term (k - 1) = 0 :=
sorry

end constant_term_in_expansion_l2816_281635


namespace digital_sum_property_l2816_281639

/-- Digital sum of a natural number -/
def digitalSum (n : ℕ) : ℕ := sorry

/-- Proposition: M satisfies S(Mk) = S(M) for all 1 ≤ k ≤ M iff M = 10^l - 1 for some l -/
theorem digital_sum_property (M : ℕ) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ M → digitalSum (M * k) = digitalSum M) ↔
  ∃ l : ℕ, l > 0 ∧ M = 10^l - 1 :=
sorry

end digital_sum_property_l2816_281639


namespace function_is_2x_l2816_281617

def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x, f x = a * x + b

theorem function_is_2x (f : ℝ → ℝ) 
  (h₁ : f (-1) = -2)
  (h₂ : f 0 = 0)
  (h₃ : f 1 = 2)
  (h₄ : f 2 = 4) :
  ∀ x, f x = 2 * x := by
sorry

end function_is_2x_l2816_281617


namespace power_of_two_plus_one_l2816_281646

-- Define a relation for numbers with the same prime factors
def same_prime_factors (x y : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ x ↔ p ∣ y)

theorem power_of_two_plus_one (b m n : ℕ) 
  (hb : b > 1) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m ≠ n) 
  (h_same_factors : same_prime_factors (b^m - 1) (b^n - 1)) : 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end power_of_two_plus_one_l2816_281646


namespace shortest_path_length_l2816_281622

/-- A regular tetrahedron with unit edge length -/
structure UnitRegularTetrahedron where
  -- We don't need to define the structure explicitly for this problem

/-- The shortest path on the surface of a unit regular tetrahedron between midpoints of opposite edges -/
def shortest_path (t : UnitRegularTetrahedron) : ℝ :=
  sorry -- Definition of the shortest path

/-- Theorem: The shortest path on the surface of a unit regular tetrahedron 
    between the midpoints of its opposite edges has a length of 1 -/
theorem shortest_path_length (t : UnitRegularTetrahedron) : 
  shortest_path t = 1 := by
  sorry

end shortest_path_length_l2816_281622


namespace correct_average_l2816_281638

theorem correct_average (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 16 →
  incorrect_num = 25 →
  correct_num = 45 →
  (n : ℚ) * incorrect_avg = (n - 1 : ℚ) * incorrect_avg + incorrect_num →
  ((n : ℚ) * incorrect_avg - incorrect_num + correct_num) / n = 18 := by
  sorry

end correct_average_l2816_281638


namespace cylinder_max_volume_l2816_281671

/-- Given a cylinder with a constant cross-section perimeter of 4,
    prove that its maximum volume is 8π/27 -/
theorem cylinder_max_volume :
  ∀ (r h : ℝ), r > 0 → h > 0 →
  (4 * r + 2 * h = 4) →
  (π * r^2 * h ≤ 8 * π / 27) ∧
  (∃ (r₀ h₀ : ℝ), r₀ > 0 ∧ h₀ > 0 ∧ 4 * r₀ + 2 * h₀ = 4 ∧ π * r₀^2 * h₀ = 8 * π / 27) :=
by sorry

end cylinder_max_volume_l2816_281671


namespace average_xyz_is_five_sixths_l2816_281678

theorem average_xyz_is_five_sixths (x y z : ℚ) 
  (eq1 : 2003 * z - 4006 * x = 1002)
  (eq2 : 2003 * y + 6009 * x = 4004) :
  (x + y + z) / 3 = 5 / 6 := by
  sorry

end average_xyz_is_five_sixths_l2816_281678


namespace rectangle_length_l2816_281602

theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 12 →
  rect_width = 6 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 24 := by
sorry

end rectangle_length_l2816_281602


namespace parallel_lines_and_planes_l2816_281613

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Returns whether two lines are parallel -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Returns whether a line is parallel to a plane -/
def line_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Returns whether a line is a subset of a plane -/
def line_subset_of_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem parallel_lines_and_planes 
  (a b : Line3D) (α : Plane3D) 
  (h : line_subset_of_plane b α) :
  ¬(∀ (a b : Line3D) (α : Plane3D), are_parallel a b → line_parallel_to_plane a α) ∧
  ¬(∀ (a b : Line3D) (α : Plane3D), line_parallel_to_plane a α → are_parallel a b) :=
by
  sorry

end parallel_lines_and_planes_l2816_281613


namespace smallest_square_containing_circle_l2816_281687

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) : 
  (2 * r) ^ 2 = 100 := by
  sorry

end smallest_square_containing_circle_l2816_281687


namespace range_of_m_l2816_281689

def p (m : ℝ) : Prop := ∃ (x y : ℝ), x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop := ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ < 0 ∧ x₁^2 - x₁ + m - 4 = 0 ∧ x₂^2 - x₂ + m - 4 = 0

theorem range_of_m (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬p m) :
  m ≤ 1 - Real.sqrt 2 ∨ (1 + Real.sqrt 2 ≤ m ∧ m < 4) :=
sorry

end range_of_m_l2816_281689


namespace specific_cube_figure_surface_area_l2816_281604

/-- A three-dimensional figure composed of unit cubes -/
structure CubeFigure where
  num_cubes : ℕ
  edge_length : ℝ

/-- Calculate the surface area of a cube figure -/
def surface_area (figure : CubeFigure) : ℝ :=
  sorry

/-- Theorem: The surface area of a specific cube figure is 32 square units -/
theorem specific_cube_figure_surface_area :
  let figure : CubeFigure := { num_cubes := 9, edge_length := 1 }
  surface_area figure = 32 := by sorry

end specific_cube_figure_surface_area_l2816_281604


namespace simplify_expression_l2816_281648

theorem simplify_expression (x y : ℝ) (n : ℤ) :
  (4 * x^(n+1) * y^n)^2 / ((-x*y)^2)^n = 16 * x^2 := by
  sorry

end simplify_expression_l2816_281648


namespace abc_inequality_l2816_281677

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = a * b * c) : 
  max a (max b c) > 17/10 := by
sorry

end abc_inequality_l2816_281677


namespace derivative_x_ln_x_l2816_281627

noncomputable section

open Real

theorem derivative_x_ln_x (x : ℝ) (h : x > 0) :
  deriv (λ x => x * log x) x = 1 + log x :=
sorry

end derivative_x_ln_x_l2816_281627


namespace initial_cheerleaders_initial_cheerleaders_correct_l2816_281657

theorem initial_cheerleaders (initial_football_players : ℕ) 
                             (quit_football_players : ℕ) 
                             (quit_cheerleaders : ℕ) 
                             (remaining_total : ℕ) : ℕ :=
  let initial_cheerleaders := 16
  have h1 : initial_football_players = 13 := by sorry
  have h2 : quit_football_players = 10 := by sorry
  have h3 : quit_cheerleaders = 4 := by sorry
  have h4 : remaining_total = 15 := by sorry
  have h5 : initial_football_players - quit_football_players + 
            (initial_cheerleaders - quit_cheerleaders) = remaining_total := by sorry
  initial_cheerleaders

theorem initial_cheerleaders_correct : initial_cheerleaders 13 10 4 15 = 16 := by sorry

end initial_cheerleaders_initial_cheerleaders_correct_l2816_281657


namespace sum_of_sixth_powers_mod_7_l2816_281624

theorem sum_of_sixth_powers_mod_7 :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
sorry

end sum_of_sixth_powers_mod_7_l2816_281624


namespace min_sum_four_reals_l2816_281636

theorem min_sum_four_reals (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ + x₂ ≥ 12)
  (h2 : x₁ + x₃ ≥ 13)
  (h3 : x₁ + x₄ ≥ 14)
  (h4 : x₃ + x₄ ≥ 22)
  (h5 : x₂ + x₃ ≥ 23)
  (h6 : x₂ + x₄ ≥ 24) :
  x₁ + x₂ + x₃ + x₄ ≥ 37 ∧ ∃ a b c d : ℝ, a + b + c + d = 37 ∧ 
    a + b ≥ 12 ∧ a + c ≥ 13 ∧ a + d ≥ 14 ∧ c + d ≥ 22 ∧ b + c ≥ 23 ∧ b + d ≥ 24 :=
by sorry

end min_sum_four_reals_l2816_281636


namespace identity_implies_a_minus_b_zero_l2816_281628

theorem identity_implies_a_minus_b_zero 
  (x : ℚ) 
  (h_pos : x > 0) 
  (h_identity : ∀ x > 0, a / (2^x - 1) + b / (2^x + 2) = (2 * 2^x + 1) / ((2^x - 1) * (2^x + 2))) : 
  a - b = 0 :=
by sorry

end identity_implies_a_minus_b_zero_l2816_281628


namespace glitched_clock_correct_time_fraction_l2816_281650

/-- Represents a 24-hour digital clock with a glitch that displays '2' as '7' -/
structure GlitchedClock where
  /-- The number of hours in a day -/
  hours_per_day : ℕ := 24
  /-- The number of minutes per hour -/
  minutes_per_hour : ℕ := 60
  /-- The digit that is displayed incorrectly -/
  glitch_digit : ℕ := 2

/-- Calculates the fraction of the day the clock displays the correct time -/
def correct_time_fraction (clock : GlitchedClock) : ℚ :=
  let correct_hours := clock.hours_per_day - 6  -- Hours without '2'
  let correct_minutes_per_hour := clock.minutes_per_hour - 16  -- Minutes without '2' per hour
  (correct_hours : ℚ) / clock.hours_per_day * correct_minutes_per_hour / clock.minutes_per_hour

theorem glitched_clock_correct_time_fraction :
  ∀ (clock : GlitchedClock), correct_time_fraction clock = 11 / 20 := by
  sorry

end glitched_clock_correct_time_fraction_l2816_281650


namespace qualified_light_bulb_probability_l2816_281637

def market_probability (factory_A_share : ℝ) (factory_B_share : ℝ) 
                       (factory_A_qualification : ℝ) (factory_B_qualification : ℝ) : ℝ :=
  factory_A_share * factory_A_qualification + factory_B_share * factory_B_qualification

theorem qualified_light_bulb_probability :
  market_probability 0.7 0.3 0.9 0.8 = 0.87 := by
  sorry

end qualified_light_bulb_probability_l2816_281637


namespace rectangle_perimeter_l2816_281654

/-- Given a square with perimeter 160 units divided into 4 rectangles, where each rectangle
    has one side equal to half of the square's side length and the other side equal to the
    full side length of the square, the perimeter of one of these rectangles is 120 units. -/
theorem rectangle_perimeter (square_perimeter : ℝ) (rectangle_count : ℕ) 
  (h1 : square_perimeter = 160)
  (h2 : rectangle_count = 4)
  (h3 : ∀ r : ℝ, r > 0 → ∃ (s w : ℝ), s = r ∧ w = r / 2 ∧ 
       4 * r = square_perimeter ∧
       rectangle_count * (s * w) = r * r) :
  ∃ (rectangle_perimeter : ℝ), rectangle_perimeter = 120 := by
  sorry

end rectangle_perimeter_l2816_281654


namespace lcm_of_9_12_18_l2816_281695

theorem lcm_of_9_12_18 : Nat.lcm (Nat.lcm 9 12) 18 = 36 := by
  sorry

end lcm_of_9_12_18_l2816_281695


namespace product_difference_squared_l2816_281611

theorem product_difference_squared : 2012 * 2016 - 2014^2 = -4 := by
  sorry

end product_difference_squared_l2816_281611


namespace f_value_at_2_l2816_281601

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : f a b (-2) = 0 → f a b 2 = -16 := by
  sorry

end f_value_at_2_l2816_281601


namespace coefficient_x_cube_equals_neg_84_equal_coefficients_implies_n_7_l2816_281600

-- Part 1
def binomial_coefficient (n k : ℕ) : ℤ := sorry

def coefficient_x_cube (x : ℝ) : ℤ := 
  binomial_coefficient 9 3 * (-1)^3

theorem coefficient_x_cube_equals_neg_84 : 
  coefficient_x_cube = λ _ ↦ -84 := by sorry

-- Part 2
def nth_term_coefficient (n r : ℕ) : ℤ := 
  binomial_coefficient n r

theorem equal_coefficients_implies_n_7 (n : ℕ) : 
  nth_term_coefficient n 2 = nth_term_coefficient n 5 → n = 7 := by sorry

end coefficient_x_cube_equals_neg_84_equal_coefficients_implies_n_7_l2816_281600


namespace no_positive_subtraction_l2816_281608

theorem no_positive_subtraction (x : ℝ) : x > 0 → 24 - x ≠ 34 := by
  sorry

end no_positive_subtraction_l2816_281608


namespace banana_group_size_l2816_281682

/-- Given a collection of bananas organized into groups, this theorem proves
    the size of each group when the total number of bananas and groups are known. -/
theorem banana_group_size
  (total_bananas : ℕ)
  (num_groups : ℕ)
  (h1 : total_bananas = 203)
  (h2 : num_groups = 7)
  : total_bananas / num_groups = 29 := by
  sorry

#eval 203 / 7  -- This should evaluate to 29

end banana_group_size_l2816_281682


namespace petes_flag_problem_l2816_281651

theorem petes_flag_problem (us_stars : Nat) (us_stripes : Nat) (total_shapes : Nat) :
  us_stars = 50 →
  us_stripes = 13 →
  total_shapes = 54 →
  ∃ (circles squares : Nat),
    circles < us_stars / 2 ∧
    squares = 2 * us_stripes + 6 ∧
    circles + squares = total_shapes ∧
    us_stars / 2 - circles = 3 :=
by
  sorry

end petes_flag_problem_l2816_281651


namespace store_sales_total_l2816_281664

/-- Represents the number of DVDs and CDs sold in a store in one day. -/
structure StoreSales where
  dvds : ℕ
  cds : ℕ

/-- Given a store that sells 1.6 times as many DVDs as CDs and sells 168 DVDs in one day,
    the total number of DVDs and CDs sold is 273. -/
theorem store_sales_total (s : StoreSales) 
    (h1 : s.dvds = 168)
    (h2 : s.dvds = (1.6 : ℝ) * s.cds) : 
    s.dvds + s.cds = 273 := by
  sorry

end store_sales_total_l2816_281664


namespace floor_plus_self_unique_solution_l2816_281659

theorem floor_plus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ + s = 20.5 :=
by
  sorry

end floor_plus_self_unique_solution_l2816_281659


namespace cricket_team_age_difference_l2816_281688

theorem cricket_team_age_difference (team_size : ℕ) (avg_age : ℝ) (captain_age : ℝ) (keeper_age_diff : ℝ) :
  team_size = 11 →
  avg_age = 25 →
  captain_age = 28 →
  keeper_age_diff = 3 →
  let total_age := avg_age * team_size
  let keeper_age := captain_age + keeper_age_diff
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + keeper_age)
  let remaining_avg := remaining_age / remaining_players
  avg_age - remaining_avg = 1 := by sorry

end cricket_team_age_difference_l2816_281688
