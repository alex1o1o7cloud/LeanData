import Mathlib

namespace subtraction_result_l3857_385790

theorem subtraction_result : 6102 - 2016 = 4086 := by
  sorry

end subtraction_result_l3857_385790


namespace starting_lineup_count_l3857_385719

def total_players : ℕ := 20
def num_goalies : ℕ := 1
def num_forwards : ℕ := 6
def num_defenders : ℕ := 4

def starting_lineup_combinations : ℕ := 
  (total_players.choose num_goalies) * 
  ((total_players - num_goalies).choose num_forwards) * 
  ((total_players - num_goalies - num_forwards).choose num_defenders)

theorem starting_lineup_count : starting_lineup_combinations = 387889200 := by
  sorry

end starting_lineup_count_l3857_385719


namespace yellow_apples_count_l3857_385772

theorem yellow_apples_count (green red total : ℕ) 
  (h1 : green = 2) 
  (h2 : red = 3) 
  (h3 : total = 19) : 
  total - (green + red) = 14 := by
  sorry

end yellow_apples_count_l3857_385772


namespace six_students_adjacent_permutations_l3857_385712

/-- The number of permutations of n elements where two specific elements must be adjacent -/
def adjacent_permutations (n : ℕ) : ℕ :=
  2 * Nat.factorial (n - 1)

/-- Theorem: The number of permutations of 6 students where 2 specific students
    must stand next to each other is 240 -/
theorem six_students_adjacent_permutations :
  adjacent_permutations 6 = 240 := by
  sorry

#eval adjacent_permutations 6

end six_students_adjacent_permutations_l3857_385712


namespace expression_evaluation_l3857_385715

theorem expression_evaluation : (4^4 - 4*(4-1)^4)^4 = 21381376 := by sorry

end expression_evaluation_l3857_385715


namespace tank_emptying_time_l3857_385754

/-- Proves that a tank with given properties empties in 6 hours due to a leak alone -/
theorem tank_emptying_time (tank_capacity : ℝ) (inlet_rate : ℝ) (emptying_time_with_inlet : ℝ) :
  tank_capacity = 4320 →
  inlet_rate = 3 →
  emptying_time_with_inlet = 8 →
  ∃ (leak_rate : ℝ),
    leak_rate > 0 ∧
    (leak_rate - inlet_rate) * (emptying_time_with_inlet * 60) = tank_capacity ∧
    tank_capacity / leak_rate / 60 = 6 :=
by sorry

end tank_emptying_time_l3857_385754


namespace special_function_at_two_l3857_385736

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 0 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)

/-- Theorem stating that f(2) = 0 for any function satisfying the given conditions -/
theorem special_function_at_two (f : ℝ → ℝ) (h : special_function f) : f 2 = 0 := by
  sorry

end special_function_at_two_l3857_385736


namespace remainder_of_587421_div_6_l3857_385796

theorem remainder_of_587421_div_6 : 587421 % 6 = 3 := by
  sorry

end remainder_of_587421_div_6_l3857_385796


namespace age_sum_theorem_l3857_385739

theorem age_sum_theorem (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 72 → a + b + c = 14 := by
  sorry

end age_sum_theorem_l3857_385739


namespace point_in_second_quadrant_l3857_385734

theorem point_in_second_quadrant (m : ℝ) :
  (m - 1 < 0 ∧ 3 > 0) → m < 1 := by
  sorry

end point_in_second_quadrant_l3857_385734


namespace jacket_price_theorem_l3857_385722

theorem jacket_price_theorem (SRP : ℝ) (marked_discount : ℝ) (additional_discount : ℝ) :
  SRP = 120 →
  marked_discount = 0.4 →
  additional_discount = 0.2 →
  let marked_price := SRP * (1 - marked_discount)
  let final_price := marked_price * (1 - additional_discount)
  (final_price / SRP) * 100 = 48 := by
  sorry

end jacket_price_theorem_l3857_385722


namespace line_equations_correct_l3857_385793

/-- Triangle ABC with vertices A(4,0), B(8,10), and C(0,6) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Definition of the specific triangle in the problem -/
def triangle : Triangle :=
  { A := (4, 0),
    B := (8, 10),
    C := (0, 6) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The equation of the line passing through A and parallel to BC -/
def line_parallel_to_BC : LineEquation :=
  { a := 1,
    b := -1,
    c := -4 }

/-- The equation of the line containing the altitude on edge AC -/
def altitude_on_AC : LineEquation :=
  { a := 2,
    b := -3,
    c := -8 }

/-- Theorem stating the correctness of the line equations -/
theorem line_equations_correct (t : Triangle) :
  t = triangle →
  (line_parallel_to_BC.a * t.A.1 + line_parallel_to_BC.b * t.A.2 + line_parallel_to_BC.c = 0) ∧
  (altitude_on_AC.a * t.B.1 + altitude_on_AC.b * t.B.2 + altitude_on_AC.c = 0) :=
by sorry

end line_equations_correct_l3857_385793


namespace circle_equation_l3857_385760

/-- Given a circle with center (a, 5-3a) that passes through (0, 0) and (3, -1),
    prove that its equation is (x - 1)^2 + (y - 2)^2 = 5 -/
theorem circle_equation (a : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - (5 - 3*a))^2 = a^2 + (5 - 3*a)^2) →
  (a^2 + (5 - 3*a)^2 = 3^2 + (-1 - (5 - 3*a))^2) →
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 5) :=
by sorry

end circle_equation_l3857_385760


namespace min_values_ab_and_a_plus_2b_l3857_385762

theorem min_values_ab_and_a_plus_2b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a * b = 2 * a + b) :
  (∀ x y, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y → 8 ≤ x * y) ∧
  (∃ x y, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y ∧ x * y = 8) ∧
  (∀ x y, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y → 9 ≤ x + 2 * y) ∧
  (∃ x y, 0 < x ∧ 0 < y ∧ x * y = 2 * x + y ∧ x + 2 * y = 9) := by
sorry

end min_values_ab_and_a_plus_2b_l3857_385762


namespace sphere_radius_change_factor_l3857_385789

theorem sphere_radius_change_factor (initial_area new_area : ℝ) 
  (h1 : initial_area = 2464)
  (h2 : new_area = 9856) : 
  let factor := (new_area / initial_area).sqrt
  factor = 2 := by sorry

end sphere_radius_change_factor_l3857_385789


namespace total_boxes_theorem_l3857_385726

/-- Calculates the total number of boxes sold over three days given the conditions --/
def total_boxes_sold (friday_boxes : ℕ) : ℕ :=
  let saturday_boxes := friday_boxes + (friday_boxes * 50 / 100)
  let sunday_boxes := saturday_boxes - (saturday_boxes * 30 / 100)
  friday_boxes + saturday_boxes + sunday_boxes

/-- Proves that the total number of boxes sold over three days is 213 --/
theorem total_boxes_theorem : total_boxes_sold 60 = 213 := by
  sorry

#eval total_boxes_sold 60

end total_boxes_theorem_l3857_385726


namespace multiply_fractions_l3857_385765

theorem multiply_fractions : (12 : ℚ) * (1 / 17) * 34 = 24 := by
  sorry

end multiply_fractions_l3857_385765


namespace num_successful_sequences_l3857_385731

/-- Represents the number of cards in the game -/
def num_cards : ℕ := 13

/-- Represents the number of cards that need to be flipped for success -/
def cards_to_flip : ℕ := 12

/-- Represents the number of choices for each flip after the first -/
def choices_per_flip : ℕ := 2

/-- Represents the rules of the card flipping game -/
structure CardGame where
  cards : Fin num_cards → Bool
  is_valid_flip : Fin num_cards → Bool

/-- Theorem stating the number of successful flip sequences -/
theorem num_successful_sequences (game : CardGame) :
  (num_cards : ℕ) * (choices_per_flip ^ (cards_to_flip - 1) : ℕ) = 26624 := by
  sorry

end num_successful_sequences_l3857_385731


namespace policeman_hats_l3857_385733

theorem policeman_hats (simpson_hats : ℕ) (obrien_hats_after : ℕ) : 
  simpson_hats = 15 →
  obrien_hats_after = 34 →
  ∃ (obrien_hats_before : ℕ), 
    obrien_hats_before > 2 * simpson_hats ∧
    obrien_hats_before = obrien_hats_after + 1 ∧
    obrien_hats_before - 2 * simpson_hats = 5 :=
by sorry

end policeman_hats_l3857_385733


namespace cylinder_surface_area_l3857_385711

theorem cylinder_surface_area (r l : ℝ) : 
  r = 1 → l = 2*r → 2*π*r*(r + l) = 6*π := by
  sorry

end cylinder_surface_area_l3857_385711


namespace common_chord_length_l3857_385787

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 12 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := x - 2*y + 6 = 0

-- Theorem statement
theorem common_chord_length : 
  ∃ (a b c d : ℝ), 
    (circle1 a b ∧ circle1 c d ∧ common_chord a b ∧ common_chord c d) →
    ((a - c)^2 + (b - d)^2)^(1/2 : ℝ) = 4 * 2^(1/2 : ℝ) :=
sorry

end common_chord_length_l3857_385787


namespace six_rounds_maximize_configurations_optimal_rounds_is_six_l3857_385708

/-- The number of cities and days in the championship --/
def n : ℕ := 8

/-- The number of possible configurations for k rounds --/
def N (k : ℕ) : ℚ :=
  (Nat.factorial n * Nat.factorial n) / (Nat.factorial k * (Nat.factorial (n - k))^2)

/-- The theorem stating that 6 rounds maximizes the number of configurations --/
theorem six_rounds_maximize_configurations :
  ∀ k : ℕ, k ≠ 6 → k ≤ n → N k ≤ N 6 := by
  sorry

/-- The main theorem proving that 6 is the optimal number of rounds --/
theorem optimal_rounds_is_six :
  ∃ k : ℕ, k ≤ n ∧ (∀ j : ℕ, j ≤ n → N j ≤ N k) ∧ k = 6 := by
  sorry

end six_rounds_maximize_configurations_optimal_rounds_is_six_l3857_385708


namespace max_silver_tokens_l3857_385753

/-- Represents the number of tokens Kevin has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth -/
structure ExchangeBooth where
  input_color : String
  input_amount : ℕ
  output_silver : ℕ
  output_other_color : String
  output_other_amount : ℕ

/-- Function to perform a single exchange -/
def exchange (tokens : TokenCount) (booth : ExchangeBooth) : TokenCount :=
  sorry

/-- Function to check if an exchange is possible -/
def can_exchange (tokens : TokenCount) (booth : ExchangeBooth) : Bool :=
  sorry

/-- Function to perform all possible exchanges -/
def perform_all_exchanges (tokens : TokenCount) (booths : List ExchangeBooth) : TokenCount :=
  sorry

/-- The main theorem stating the maximum number of silver tokens Kevin can obtain -/
theorem max_silver_tokens : 
  let initial_tokens : TokenCount := ⟨100, 100, 0⟩
  let booth1 : ExchangeBooth := ⟨"red", 3, 1, "blue", 2⟩
  let booth2 : ExchangeBooth := ⟨"blue", 4, 1, "red", 2⟩
  let final_tokens := perform_all_exchanges initial_tokens [booth1, booth2]
  final_tokens.silver = 132 :=
sorry

end max_silver_tokens_l3857_385753


namespace ellipse_sum_specific_l3857_385775

/-- Represents an ellipse with center (h, k) and semi-axes lengths a and b -/
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The sum of h, k, a, and b for a specific ellipse -/
def ellipse_sum (e : Ellipse) : ℝ :=
  e.h + e.k + e.a + e.b

theorem ellipse_sum_specific : ∃ (e : Ellipse), 
  e.h = 3 ∧ 
  e.k = -1 ∧ 
  e.a = 6 ∧ 
  e.b = 4 ∧ 
  ellipse_sum e = 12 := by
  sorry

end ellipse_sum_specific_l3857_385775


namespace biff_break_even_time_biff_break_even_time_is_three_l3857_385720

/-- Calculates the break-even time for Biff's bus trip -/
theorem biff_break_even_time 
  (ticket_cost : ℝ) 
  (snacks_cost : ℝ) 
  (headphones_cost : ℝ) 
  (work_rate : ℝ) 
  (wifi_cost : ℝ) : ℝ :=
  let total_cost := ticket_cost + snacks_cost + headphones_cost
  let net_hourly_rate := work_rate - wifi_cost
  total_cost / net_hourly_rate

/-- Proves that Biff's break-even time is 3 hours given the specific costs and rates -/
theorem biff_break_even_time_is_three :
  biff_break_even_time 11 3 16 12 2 = 3 := by
  sorry

end biff_break_even_time_biff_break_even_time_is_three_l3857_385720


namespace younger_person_age_l3857_385791

/-- Proves that the younger person's age is 8 years, given the conditions of the problem. -/
theorem younger_person_age (y e : ℕ) : 
  e = y + 12 →  -- The elder person's age is 12 years more than the younger person's
  e - 5 = 5 * (y - 5) →  -- Five years ago, the elder was 5 times as old as the younger
  y = 8 :=  -- The younger person's present age is 8 years
by sorry

end younger_person_age_l3857_385791


namespace origin_outside_circle_l3857_385781

/-- The circle equation: x^2 + y^2 - 2ax - 2y + (a-1)^2 = 0 -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*y + (a-1)^2 = 0

/-- A point (x, y) is outside the circle if the left-hand side of the equation is positive -/
def is_outside_circle (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 2*y + (a-1)^2 > 0

theorem origin_outside_circle (a : ℝ) (h : a > 1) :
  is_outside_circle 0 0 a :=
sorry

end origin_outside_circle_l3857_385781


namespace first_walk_time_l3857_385742

/-- Represents a walk with its speed and distance -/
structure Walk where
  speed : ℝ
  distance : ℝ

/-- Proves that the time taken for the first walk is 2 hours given the problem conditions -/
theorem first_walk_time (first_walk : Walk) (second_walk : Walk) 
  (h1 : first_walk.speed = 3)
  (h2 : second_walk.speed = 4)
  (h3 : second_walk.distance = first_walk.distance + 2)
  (h4 : first_walk.distance / first_walk.speed + second_walk.distance / second_walk.speed = 4) :
  first_walk.distance / first_walk.speed = 2 := by
  sorry


end first_walk_time_l3857_385742


namespace solve_equation_l3857_385744

theorem solve_equation (x : ℚ) : x / 4 - x - 3 / 6 = 1 → x = -2 := by
  sorry

end solve_equation_l3857_385744


namespace solve_star_equation_l3857_385746

-- Define the custom operation ※
def star (a b : ℚ) : ℚ := a + b

-- State the theorem
theorem solve_star_equation :
  ∃ x : ℚ, star 4 (star x 3) = 1 ∧ x = -6 := by
  sorry

end solve_star_equation_l3857_385746


namespace condition_relationship_l3857_385768

theorem condition_relationship (A B : Prop) 
  (h : (¬A → ¬B) ∧ ¬(¬B → ¬A)) : 
  (B → A) ∧ ¬(A → B) := by
  sorry

end condition_relationship_l3857_385768


namespace no_real_roots_condition_l3857_385725

/-- A quadratic equation of the form ax^2 + bx + c = 0 has no real roots if and only if its discriminant is negative. -/
axiom no_real_roots_iff_neg_discriminant {a b c : ℝ} (ha : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≠ 0) ↔ b^2 - 4*a*c < 0

/-- For the quadratic equation 6x^2 - 5x + a = 0 to have no real roots, a must be greater than 25/24. -/
theorem no_real_roots_condition (a : ℝ) :
  (∀ x, 6 * x^2 - 5 * x + a ≠ 0) ↔ a > 25/24 := by
  sorry

end no_real_roots_condition_l3857_385725


namespace largest_number_in_ratio_l3857_385776

theorem largest_number_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (b / a = 4 / 3) →
  (c / a = 6 / 3) →
  a * b * c = 1944 →
  max a (max b c) = 18 := by
sorry

end largest_number_in_ratio_l3857_385776


namespace min_value_theorem_min_value_achievable_l3857_385798

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 19) / Real.sqrt (x^2 + 8) ≥ 2 * Real.sqrt 11 := by
  sorry

theorem min_value_achievable : 
  ∃ x : ℝ, (x^2 + 19) / Real.sqrt (x^2 + 8) = 2 * Real.sqrt 11 := by
  sorry

end min_value_theorem_min_value_achievable_l3857_385798


namespace min_value_theorem_l3857_385794

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b) * (b + 4/a) ≥ 9 := by
  sorry

end min_value_theorem_l3857_385794


namespace five_digit_reverse_multiply_nine_l3857_385723

theorem five_digit_reverse_multiply_nine :
  ∃! n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧
    (∃ a b c d e : ℕ,
      n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
      9 * n = 10000 * e + 1000 * d + 100 * c + 10 * b + a ∧
      a ≠ 0) ∧
    n = 10989 :=
by sorry

end five_digit_reverse_multiply_nine_l3857_385723


namespace triangle_inequality_l3857_385730

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l3857_385730


namespace nested_f_evaluation_l3857_385706

def f (x : ℝ) : ℝ := x^2 + 1

theorem nested_f_evaluation : f (f (f (-1))) = 26 := by sorry

end nested_f_evaluation_l3857_385706


namespace product_of_fractions_l3857_385774

theorem product_of_fractions : 
  (((3^4 - 1) / (3^4 + 1)) * ((4^4 - 1) / (4^4 + 1)) * ((5^4 - 1) / (5^4 + 1)) * 
   ((6^4 - 1) / (6^4 + 1)) * ((7^4 - 1) / (7^4 + 1))) = 25 / 210 := by
  sorry

end product_of_fractions_l3857_385774


namespace terminal_side_half_angle_l3857_385797

def is_in_first_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2

def is_in_first_or_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi < α ∧ α < k * Real.pi + Real.pi / 2

theorem terminal_side_half_angle (α : Real) :
  is_in_first_quadrant α → is_in_first_or_third_quadrant (α / 2) :=
by sorry

end terminal_side_half_angle_l3857_385797


namespace hyperbola_center_l3857_385756

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (3, 2) ∧ f2 = (11, 6) →
  center = (7, 4) := by sorry

end hyperbola_center_l3857_385756


namespace flour_cost_for_cheapest_pie_l3857_385710

/-- The cost of flour for the cheapest pie -/
def flour_cost : ℝ := 2

/-- The cost of sugar for both pies -/
def sugar_cost : ℝ := 1

/-- The cost of eggs and butter for both pies -/
def eggs_butter_cost : ℝ := 1.5

/-- The weight of blueberries needed for the blueberry pie in pounds -/
def blueberry_weight : ℝ := 3

/-- The weight of a container of blueberries in ounces -/
def blueberry_container_weight : ℝ := 8

/-- The cost of a container of blueberries -/
def blueberry_container_cost : ℝ := 2.25

/-- The weight of cherries needed for the cherry pie in pounds -/
def cherry_weight : ℝ := 4

/-- The cost of a four-pound bag of cherries -/
def cherry_bag_cost : ℝ := 14

/-- The total price to make the cheapest pie -/
def cheapest_pie_cost : ℝ := 18

theorem flour_cost_for_cheapest_pie :
  flour_cost = cheapest_pie_cost - min
    (sugar_cost + eggs_butter_cost + (blueberry_weight * 16 / blueberry_container_weight) * blueberry_container_cost)
    (sugar_cost + eggs_butter_cost + cherry_bag_cost) :=
by sorry

end flour_cost_for_cheapest_pie_l3857_385710


namespace min_value_of_expression_lower_bound_achievable_l3857_385773

theorem min_value_of_expression (x y : ℝ) : 
  (x^2 * y^2 - 1)^2 + (x^2 + y^2)^2 ≥ 1 := by
  sorry

theorem lower_bound_achievable : 
  ∃ x y : ℝ, (x^2 * y^2 - 1)^2 + (x^2 + y^2)^2 = 1 := by
  sorry

end min_value_of_expression_lower_bound_achievable_l3857_385773


namespace arrow_connections_theorem_l3857_385751

/-- The number of ways to connect 2n points on a circle with n arrows -/
def arrow_connections (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- Theorem statement for the arrow connection problem -/
theorem arrow_connections_theorem (n : ℕ) (h : n > 0) :
  arrow_connections n = Nat.choose (2 * n) n :=
by sorry

end arrow_connections_theorem_l3857_385751


namespace cone_base_radius_l3857_385764

/-- Given a cone with slant height 5 cm and lateral surface area 15π cm², 
    prove that the radius of its base is 3 cm. -/
theorem cone_base_radius (l : ℝ) (L : ℝ) (π : ℝ) (r : ℝ) : 
  l = 5 →
  L = 15 * π →
  L = π * r * l →
  r = 3 := by sorry

end cone_base_radius_l3857_385764


namespace min_value_of_expression_l3857_385761

theorem min_value_of_expression (x y k : ℝ) : 
  (x * y + k)^2 + (x - y)^2 ≥ 0 ∧ 
  ∃ (x y k : ℝ), (x * y + k)^2 + (x - y)^2 = 0 :=
by sorry

end min_value_of_expression_l3857_385761


namespace min_distance_point_l3857_385758

/-- Given a triangle ABC with vertices (x₁, y₁), (x₂, y₂), (x₃, y₃), 
    the point P that minimizes the sum of squares of distances from P to the three vertices 
    has coordinates ((x₁ + x₂ + x₃) / 3, (y₁ + y₂ + y₃) / 3) -/
theorem min_distance_point (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  let dist_sum_sq (x y : ℝ) := 
    (x - x₁)^2 + (y - y₁)^2 + (x - x₂)^2 + (y - y₂)^2 + (x - x₃)^2 + (y - y₃)^2
  ∃ (x y : ℝ), (∀ (u v : ℝ), dist_sum_sq x y ≤ dist_sum_sq u v) ∧ 
    x = (x₁ + x₂ + x₃) / 3 ∧ y = (y₁ + y₂ + y₃) / 3 := by
  sorry

end min_distance_point_l3857_385758


namespace smallest_b_value_l3857_385702

/-- Given real numbers a and b satisfying certain conditions, 
    the smallest possible value of b is 2. -/
theorem smallest_b_value (a b : ℝ) 
  (h1 : 2 < a) 
  (h2 : a < b) 
  (h3 : ¬ (2 + a > b ∧ 2 + b > a ∧ a + b > 2)) 
  (h4 : ¬ (1/b + 1/a > 2 ∧ 1/b + 2 > 1/a ∧ 1/a + 2 > 1/b)) : 
  ∀ ε > 0, b ≥ 2 - ε := by
sorry

end smallest_b_value_l3857_385702


namespace total_tickets_is_900_l3857_385783

/-- Represents the total number of tickets sold at a movie theater. -/
def total_tickets (adult_price child_price : ℕ) (total_revenue child_tickets : ℕ) : ℕ :=
  let adult_tickets := (total_revenue - child_price * child_tickets) / adult_price
  adult_tickets + child_tickets

/-- Theorem stating that the total number of tickets sold is 900. -/
theorem total_tickets_is_900 :
  total_tickets 7 4 5100 400 = 900 := by
  sorry

end total_tickets_is_900_l3857_385783


namespace polynomial_factorization_l3857_385777

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 100*x^4 = (-10*x^2 + x + 3) * (10*x^2 + x + 3) := by
  sorry

end polynomial_factorization_l3857_385777


namespace triangle_ratio_l3857_385770

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → b = 1 → c = 4 → 
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end triangle_ratio_l3857_385770


namespace modulus_of_complex_l3857_385743

theorem modulus_of_complex (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  a / (1 - i) = 1 - b * i →
  Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end modulus_of_complex_l3857_385743


namespace smallest_multiple_l3857_385763

theorem smallest_multiple : ∃ n : ℕ, 
  n > 0 ∧ 
  19 ∣ n ∧ 
  n % 97 = 3 ∧ 
  ∀ m : ℕ, m > 0 → 19 ∣ m → m % 97 = 3 → n ≤ m ∧ 
  n = 494 := by
  sorry

end smallest_multiple_l3857_385763


namespace pure_imaginary_ratio_l3857_385780

theorem pure_imaginary_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = y * Complex.I) : 
  p / q = -4 / 3 := by
sorry

end pure_imaginary_ratio_l3857_385780


namespace unique_solution_factorial_equation_l3857_385717

theorem unique_solution_factorial_equation :
  ∃! (a b : ℕ), a^2 + 2 = Nat.factorial b :=
by
  -- The proof goes here
  sorry

end unique_solution_factorial_equation_l3857_385717


namespace base_2_representation_of_84_l3857_385703

theorem base_2_representation_of_84 :
  ∃ (a b c d e f g : ℕ),
    (a = 1 ∧ b = 0 ∧ c = 1 ∧ d = 0 ∧ e = 1 ∧ f = 0 ∧ g = 0) ∧
    84 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end base_2_representation_of_84_l3857_385703


namespace number_divided_by_five_equals_number_plus_three_l3857_385792

theorem number_divided_by_five_equals_number_plus_three : 
  ∃ x : ℚ, x / 5 = x + 3 ∧ x = -15 / 4 := by
  sorry

end number_divided_by_five_equals_number_plus_three_l3857_385792


namespace max_value_g_and_range_of_a_l3857_385727

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x)

def g (a : ℝ) (x : ℝ) : ℝ := x^2 * f a x

def h (a : ℝ) (x : ℝ) : ℝ := x^2 / f a x - 1

theorem max_value_g_and_range_of_a :
  (∀ x > 0, g (-2) x ≤ Real.exp (-2)) ∧
  (∀ a : ℝ, (∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 16 ∧ h a x₁ = 0 ∧ h a x₂ = 0) →
    1/2 * Real.log 2 < a ∧ a < 2 / Real.exp 1) :=
by sorry

end max_value_g_and_range_of_a_l3857_385727


namespace discriminant_of_specific_quadratic_l3857_385757

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 9x + 1 is 61 -/
theorem discriminant_of_specific_quadratic : discriminant 5 (-9) 1 = 61 := by
  sorry

end discriminant_of_specific_quadratic_l3857_385757


namespace store_discount_difference_l3857_385700

theorem store_discount_difference :
  let initial_discount : ℝ := 0.25
  let additional_discount : ℝ := 0.10
  let claimed_discount : ℝ := 0.35
  let price_after_initial := 1 - initial_discount
  let price_after_both := price_after_initial * (1 - additional_discount)
  let true_discount := 1 - price_after_both
  claimed_discount - true_discount = 0.025 := by
sorry

end store_discount_difference_l3857_385700


namespace diamond_four_three_l3857_385741

def diamond (a b : ℝ) : ℝ := 4*a + 3*b - 2*a*b

theorem diamond_four_three : diamond 4 3 = 1 := by sorry

end diamond_four_three_l3857_385741


namespace friend_savings_rate_l3857_385735

/-- Proves that given the initial amounts and saving rates, after 25 weeks,
    both people will have the same amount of money if and only if the friend saves 5 dollars per week. -/
theorem friend_savings_rate (your_initial : ℕ) (friend_initial : ℕ) (your_weekly_savings : ℕ) (weeks : ℕ) :
  your_initial = 160 →
  friend_initial = 210 →
  your_weekly_savings = 7 →
  weeks = 25 →
  (your_initial + your_weekly_savings * weeks = friend_initial + 5 * weeks) :=
by sorry

end friend_savings_rate_l3857_385735


namespace crossed_out_digit_l3857_385705

theorem crossed_out_digit (N : Nat) (x : Nat) : 
  (N % 9 = 3) → 
  (x ≤ 9) →
  (∃ a b : Nat, N = a * 10 + x + b ∧ b < 10^9) →
  ((N - x) % 9 = 7) →
  x = 5 := by
sorry

end crossed_out_digit_l3857_385705


namespace thirty_two_team_tournament_games_l3857_385748

/-- Represents a single-elimination tournament -/
structure SingleEliminationTournament where
  num_teams : ℕ

/-- The number of games played in a single-elimination tournament -/
def games_played (t : SingleEliminationTournament) : ℕ :=
  t.num_teams - 1

theorem thirty_two_team_tournament_games :
  ∀ (t : SingleEliminationTournament),
    t.num_teams = 32 →
    games_played t = 31 := by
  sorry

end thirty_two_team_tournament_games_l3857_385748


namespace fraction_equality_l3857_385745

-- Define the @ operation
def at_op (a b : ℝ) : ℝ := a * b - b^3

-- Define the # operation
def hash_op (a b : ℝ) : ℝ := a + b - a * b^2

-- Theorem statement
theorem fraction_equality : 
  let a : ℝ := 3
  let b : ℝ := 2
  (at_op a b) / (hash_op a b) = 2 / 7 := by
  sorry

end fraction_equality_l3857_385745


namespace proposition_implication_l3857_385749

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 4) : 
  ¬ P 3 := by
  sorry

end proposition_implication_l3857_385749


namespace man_walking_speed_l3857_385766

/-- Calculates the walking speed of a man given the following conditions:
  * The man walks at a constant speed
  * He takes a 5-minute rest after every kilometer
  * He covers 5 kilometers in 50 minutes
-/
theorem man_walking_speed (total_time : ℝ) (total_distance : ℝ) (rest_time : ℝ) 
  (rest_frequency : ℝ) (h1 : total_time = 50) (h2 : total_distance = 5) 
  (h3 : rest_time = 5) (h4 : rest_frequency = 1) : 
  (total_distance / ((total_time - (rest_time * (total_distance - 1))) / 60)) = 10 := by
  sorry

#check man_walking_speed

end man_walking_speed_l3857_385766


namespace lineup_count_l3857_385788

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_members : ℕ) (offensive_linemen : ℕ) : ℕ :=
  offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem: The number of ways to choose a starting lineup for a team of 15 members
    with 5 offensive linemen is 109200 -/
theorem lineup_count :
  choose_lineup 15 5 = 109200 := by
  sorry

end lineup_count_l3857_385788


namespace salary_increase_after_four_years_l3857_385716

theorem salary_increase_after_four_years (annual_raise : ℝ) (h : annual_raise = 0.1) :
  (1 + annual_raise)^4 - 1 > 0.45 := by sorry

end salary_increase_after_four_years_l3857_385716


namespace area_ADBC_l3857_385738

/-- Given a triangle ABC in the xy-plane where:
    A is at the origin (0, 0)
    B lies on the positive x-axis
    C is in the upper right quadrant
    ∠A = 30°, ∠B = 60°, ∠C = 90°
    Length BC = 1
    D is the intersection of the angle bisector of ∠C with the y-axis

    The area of quadrilateral ADBC is (5√3 + 9) / 4 -/
theorem area_ADBC (A B C D : ℝ × ℝ) : 
  A = (0, 0) →
  B.1 > 0 ∧ B.2 = 0 →
  C.1 > 0 ∧ C.2 > 0 →
  Real.cos (π/6) * (C.1 - A.1) = Real.sin (π/6) * (C.2 - A.2) →
  Real.cos (π/3) * (C.1 - B.1) = Real.sin (π/3) * (C.2 - B.2) →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 1 →
  D.1 = 0 →
  (C.2 - D.2) / (C.1 - D.1) = (C.2 - A.2) / (C.1 - A.1) →
  let area := abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 +
               abs ((C.1 - A.1) * (D.2 - A.2) - (D.1 - A.1) * (C.2 - A.2)) / 2
  area = (5 * Real.sqrt 3 + 9) / 4 := by
  sorry


end area_ADBC_l3857_385738


namespace initial_chickens_is_300_l3857_385786

/-- Represents the initial state and conditions of the poultry farm problem --/
structure PoultryFarm where
  initial_turkeys : ℕ
  initial_guinea_fowls : ℕ
  daily_loss_chickens : ℕ
  daily_loss_turkeys : ℕ
  daily_loss_guinea_fowls : ℕ
  duration_days : ℕ
  total_remaining : ℕ

/-- Calculates the initial number of chickens given the farm conditions --/
def calculate_initial_chickens (farm : PoultryFarm) : ℕ :=
  let remaining_turkeys := farm.initial_turkeys - farm.daily_loss_turkeys * farm.duration_days
  let remaining_guinea_fowls := farm.initial_guinea_fowls - farm.daily_loss_guinea_fowls * farm.duration_days
  let remaining_chickens := farm.total_remaining - remaining_turkeys - remaining_guinea_fowls
  remaining_chickens + farm.daily_loss_chickens * farm.duration_days

/-- Theorem stating that the initial number of chickens is 300 --/
theorem initial_chickens_is_300 (farm : PoultryFarm)
  (h1 : farm.initial_turkeys = 200)
  (h2 : farm.initial_guinea_fowls = 80)
  (h3 : farm.daily_loss_chickens = 20)
  (h4 : farm.daily_loss_turkeys = 8)
  (h5 : farm.daily_loss_guinea_fowls = 5)
  (h6 : farm.duration_days = 7)
  (h7 : farm.total_remaining = 349) :
  calculate_initial_chickens farm = 300 := by
  sorry

end initial_chickens_is_300_l3857_385786


namespace slices_per_pie_is_four_l3857_385718

/-- The number of slices in a whole pie at a pie shop -/
def slices_per_pie : ℕ := sorry

/-- The price of a single slice of pie in dollars -/
def price_per_slice : ℕ := 5

/-- The number of whole pies sold -/
def pies_sold : ℕ := 9

/-- The total revenue in dollars from selling all pies -/
def total_revenue : ℕ := 180

/-- Theorem stating that the number of slices per pie is 4 -/
theorem slices_per_pie_is_four :
  slices_per_pie = 4 :=
by sorry

end slices_per_pie_is_four_l3857_385718


namespace binary_to_hex_l3857_385750

-- Define the binary number
def binary_num : ℕ := 1011001

-- Define the hexadecimal number
def hex_num : ℕ := 0x59

-- Theorem stating that the binary number is equal to the hexadecimal number
theorem binary_to_hex : binary_num = hex_num := by
  sorry

end binary_to_hex_l3857_385750


namespace speed_increase_percentage_l3857_385767

def initial_speed : ℝ := 80
def training_weeks : ℕ := 16
def speed_gain_per_week : ℝ := 1

def final_speed : ℝ := initial_speed + (speed_gain_per_week * training_weeks)

theorem speed_increase_percentage :
  (final_speed - initial_speed) / initial_speed * 100 = 20 := by
  sorry

end speed_increase_percentage_l3857_385767


namespace factorization_constant_l3857_385799

theorem factorization_constant (c : ℝ) : 
  (∀ x, x^2 - 4*x + c = (x - 1) * (x - 3)) → c = 3 := by
  sorry

end factorization_constant_l3857_385799


namespace parallel_vectors_tan_double_angle_l3857_385752

/-- Given two vectors a and b in R², where a is parallel to b, 
    prove that tan(2θ) = -4/3 -/
theorem parallel_vectors_tan_double_angle (θ : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (Real.sin θ, 2)) 
  (hb : b = (Real.cos θ, 1)) 
  (hparallel : ∃ (k : ℝ), a = k • b) : 
  Real.tan (2 * θ) = -4/3 := by
  sorry

end parallel_vectors_tan_double_angle_l3857_385752


namespace rationalize_denominator_l3857_385759

theorem rationalize_denominator :
  ∀ x : ℝ, x > 0 → (30 : ℝ) / (5 - Real.sqrt x) = -30 - 6 * Real.sqrt x → x = 30 :=
by sorry

end rationalize_denominator_l3857_385759


namespace no_unique_solution_implies_a_equals_four_l3857_385721

/-- Given two linear equations in two variables, this function determines if they have a unique solution. -/
def hasUniqueSolution (a k : ℝ) : Prop :=
  ∃! (x y : ℝ), a * (3 * x + 4 * y) = 36 ∧ k * x + 12 * y = 30

/-- The theorem states that when k = 9 and the equations don't have a unique solution, a must equal 4. -/
theorem no_unique_solution_implies_a_equals_four :
  ∀ (a : ℝ), (¬ hasUniqueSolution a 9) → a = 4 := by
  sorry

#check no_unique_solution_implies_a_equals_four

end no_unique_solution_implies_a_equals_four_l3857_385721


namespace w_squared_value_l3857_385795

theorem w_squared_value (w : ℝ) (h : (2*w + 10)^2 = (5*w + 15)*(w + 6)) : 
  w^2 = (90 + 10*Real.sqrt 65) / 4 := by
sorry

end w_squared_value_l3857_385795


namespace brown_mice_count_l3857_385707

theorem brown_mice_count (total : ℕ) (white : ℕ) : 
  (2 : ℚ) / 3 * total = white → white = 14 → total - white = 7 := by
  sorry

end brown_mice_count_l3857_385707


namespace sum_of_triangle_operations_l3857_385701

def triangle_operation (a b c : ℤ) : ℤ := 2*a + b - c

theorem sum_of_triangle_operations : 
  triangle_operation 1 2 3 + triangle_operation 4 6 5 + triangle_operation 2 7 1 = 20 := by
  sorry

end sum_of_triangle_operations_l3857_385701


namespace max_sum_given_constraints_l3857_385729

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end max_sum_given_constraints_l3857_385729


namespace pet_store_ratio_l3857_385732

theorem pet_store_ratio (num_cats : ℕ) (num_dogs : ℕ) : 
  (num_cats : ℚ) / num_dogs = 3 / 4 →
  num_cats = 18 →
  num_dogs = 24 := by
sorry

end pet_store_ratio_l3857_385732


namespace max_regions_is_nine_l3857_385713

/-- Represents a square in a 2D plane -/
structure Square where
  -- We don't need to define the internals of the square for this problem

/-- The number of regions created by two intersecting squares -/
def num_regions (s1 s2 : Square) : ℕ := sorry

/-- The maximum number of regions that can be created by two intersecting squares -/
def max_regions : ℕ := sorry

/-- Theorem: The maximum number of regions created by two intersecting squares is 9 -/
theorem max_regions_is_nine : max_regions = 9 := by sorry

end max_regions_is_nine_l3857_385713


namespace complex_expression_evaluation_l3857_385778

theorem complex_expression_evaluation :
  ∀ (a b : ℂ),
  a = 5 - 3*I →
  b = 2 + 4*I →
  3*a - 4*b = 7 - 25*I :=
by
  sorry

end complex_expression_evaluation_l3857_385778


namespace ascent_speed_l3857_385784

/-- 
Given a round trip journey with:
- Total time of 8 hours
- Ascent time of 5 hours
- Descent time of 3 hours
- Average speed for the entire journey of 3 km/h
Prove that the average speed during the ascent is 2.4 km/h
-/
theorem ascent_speed (total_time : ℝ) (ascent_time : ℝ) (descent_time : ℝ) (avg_speed : ℝ) :
  total_time = 8 →
  ascent_time = 5 →
  descent_time = 3 →
  avg_speed = 3 →
  (avg_speed * total_time / 2) / ascent_time = 2.4 := by
  sorry

end ascent_speed_l3857_385784


namespace seating_arrangements_l3857_385785

theorem seating_arrangements (n : ℕ) (h : n = 6) : Nat.factorial n = 720 := by
  sorry

end seating_arrangements_l3857_385785


namespace parabola_directrix_p_l3857_385755

/-- A parabola with equation y^2 = 2px and directrix x = -1 has p = 2 -/
theorem parabola_directrix_p (y x p : ℝ) : 
  (∀ y, y^2 = 2*p*x) →  -- Parabola equation
  (x = -1)             -- Directrix equation
  → p = 2 := by sorry

end parabola_directrix_p_l3857_385755


namespace zeros_after_one_in_factorial_power_is_2400_l3857_385709

/-- The number of zeros following the digit '1' in the decimal expansion of (100!)^100 -/
def zeros_after_one_in_factorial_power : ℕ :=
  let factors_of_five : ℕ := (100 / 5) + (100 / 25)
  let zeros_in_factorial : ℕ := factors_of_five
  zeros_in_factorial * 100

/-- Theorem stating that the number of zeros after '1' in (100!)^100 is 2400 -/
theorem zeros_after_one_in_factorial_power_is_2400 :
  zeros_after_one_in_factorial_power = 2400 := by
  sorry

end zeros_after_one_in_factorial_power_is_2400_l3857_385709


namespace terry_age_l3857_385747

/-- Given the following conditions:
    1. In 10 years, Terry will be 4 times Nora's current age.
    2. Nora is currently 10 years old.
    3. In 5 years, Nora will be half Sam's age.
    4. Sam is currently 6 years older than Terry.
    Prove that Terry is currently 19 years old. -/
theorem terry_age (nora_age : ℕ) (terry_future_age : ℕ → ℕ) (sam_age : ℕ → ℕ) :
  nora_age = 10 ∧
  terry_future_age 10 = 4 * nora_age ∧
  sam_age 5 = 2 * (nora_age + 5) ∧
  sam_age 0 = terry_future_age 0 + 6 →
  terry_future_age 0 = 19 := by
  sorry

end terry_age_l3857_385747


namespace f_monotonicity_and_a_range_l3857_385782

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) / Real.log x

theorem f_monotonicity_and_a_range :
  (∀ x₁ x₂, e < x₁ ∧ x₁ < x₂ → f 0 x₁ < f 0 x₂) ∧ 
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f 0 x₁ > f 0 x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < e → f 0 x₁ > f 0 x₂) ∧
  (∀ a, (∀ x, 1 < x → f a x > Real.sqrt x) → a ≤ 1) :=
sorry

end f_monotonicity_and_a_range_l3857_385782


namespace library_book_distribution_l3857_385724

/-- The number of ways to distribute n identical objects between two locations,
    with at least one object in each location. -/
def distributionWays (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

/-- The problem statement as a theorem -/
theorem library_book_distribution :
  distributionWays 8 = 7 := by
  sorry

end library_book_distribution_l3857_385724


namespace one_zero_quadratic_l3857_385714

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem one_zero_quadratic (a : ℝ) :
  (∃! x, f a x = 0) → (a = 0 ∨ a = 1) :=
by sorry

end one_zero_quadratic_l3857_385714


namespace rohan_entertainment_spending_l3857_385740

/-- Represents Rohan's monthly finances --/
structure RohanFinances where
  salary : ℝ
  food_percent : ℝ
  rent_percent : ℝ
  conveyance_percent : ℝ
  savings : ℝ

/-- The conditions of Rohan's finances --/
def rohan_finances : RohanFinances :=
  { salary := 12500
  , food_percent := 40
  , rent_percent := 20
  , conveyance_percent := 10
  , savings := 2500 }

/-- Theorem stating that Rohan spends 10% on entertainment --/
theorem rohan_entertainment_spending (rf : RohanFinances := rohan_finances) :
  let total_percent := rf.food_percent + rf.rent_percent + rf.conveyance_percent + (rf.savings / rf.salary * 100)
  let entertainment_percent := 100 - total_percent
  entertainment_percent = 10 := by sorry

end rohan_entertainment_spending_l3857_385740


namespace students_transferred_theorem_l3857_385704

/-- Calculates the number of students transferred to fifth grade -/
def students_transferred_to_fifth (initial_students : ℝ) (students_left : ℝ) (final_students : ℝ) : ℝ :=
  initial_students - students_left - final_students

/-- Proves that the number of students transferred to fifth grade is 10.0 -/
theorem students_transferred_theorem (initial_students : ℝ) (students_left : ℝ) (final_students : ℝ)
  (h1 : initial_students = 42.0)
  (h2 : students_left = 4.0)
  (h3 : final_students = 28.0) :
  students_transferred_to_fifth initial_students students_left final_students = 10.0 := by
  sorry

#eval students_transferred_to_fifth 42.0 4.0 28.0

end students_transferred_theorem_l3857_385704


namespace A_completes_in_20_days_l3857_385769

/-- The number of days B takes to complete the project alone -/
def B_days : ℝ := 30

/-- The number of days A and B work together -/
def together_days : ℝ := 8

/-- The number of days B works alone after A quits -/
def B_alone_days : ℝ := 10

/-- The total amount of work (100% of the project) -/
def total_work : ℝ := 1

/-- Theorem stating that A can complete the project alone in 20 days -/
theorem A_completes_in_20_days :
  ∃ A_days : ℝ,
    A_days = 20 ∧
    together_days * (1 / A_days + 1 / B_days) + B_alone_days * (1 / B_days) = total_work :=
by sorry

end A_completes_in_20_days_l3857_385769


namespace equation_solution_l3857_385779

theorem equation_solution : ∃ x : ℝ, (4 / (x - 1) + 1 / (1 - x) = 1) ∧ (x = 4) := by
  sorry

end equation_solution_l3857_385779


namespace normal_probability_theorem_l3857_385728

/-- The standard normal cumulative distribution function -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- Normal distribution probability density function -/
def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ := sorry

theorem normal_probability_theorem (ξ : ℝ → ℝ) (μ σ : ℝ) 
  (h_normal : ∀ x, normal_pdf μ σ x = sorry)  -- ξ follows N(μ, σ²)
  (h_mean : ∫ x, x * normal_pdf μ σ x = 3)    -- E[ξ] = 3
  (h_var : ∫ x, (x - μ)^2 * normal_pdf μ σ x = 1)  -- D[ξ] = 1
  : ∫ x in Set.Ioo (-1) 1, normal_pdf μ σ x = Φ (-4) - Φ (-2) :=
sorry

end normal_probability_theorem_l3857_385728


namespace interest_rate_difference_l3857_385737

/-- The difference between two simple interest rates given specific conditions -/
theorem interest_rate_difference (principal : ℝ) (time : ℝ) (interest_diff : ℝ) :
  principal = 2600 →
  time = 3 →
  interest_diff = 78 →
  ∃ (rate1 rate2 : ℝ), rate2 - rate1 = 0.01 ∧
    principal * time * (rate2 - rate1) / 100 = interest_diff :=
by sorry

end interest_rate_difference_l3857_385737


namespace optimal_z_maximizes_optimal_z_satisfies_condition_l3857_385771

open Complex

/-- The complex number that maximizes the given expression -/
def optimal_z : ℂ := -4 + I

theorem optimal_z_maximizes (z : ℂ) (h : arg (z + 3) = Real.pi * (3 / 4)) :
  1 / (abs (z + 6) + abs (z - 3 * I)) ≤ 1 / (abs (optimal_z + 6) + abs (optimal_z - 3 * I)) :=
by sorry

theorem optimal_z_satisfies_condition :
  arg (optimal_z + 3) = Real.pi * (3 / 4) :=
by sorry

end optimal_z_maximizes_optimal_z_satisfies_condition_l3857_385771
