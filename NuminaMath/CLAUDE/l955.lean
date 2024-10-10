import Mathlib

namespace quadratic_root_relation_l955_95589

theorem quadratic_root_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -c ∧ s₁ * s₂ = a) ∧
               (3*s₁ + 3*s₂ = -a ∧ 9*s₁*s₂ = b)) →
  b / c = 27 :=
by sorry

end quadratic_root_relation_l955_95589


namespace solve_candy_problem_l955_95517

def candy_problem (initial_candies : ℕ) (friend_multiplier : ℕ) (friend_eaten : ℕ) : Prop :=
  let friend_brought := initial_candies * friend_multiplier
  let total_candies := initial_candies + friend_brought
  let each_share := total_candies / 2
  let friend_final := each_share - friend_eaten
  friend_final = 65

theorem solve_candy_problem :
  candy_problem 50 2 10 := by
  sorry

end solve_candy_problem_l955_95517


namespace distance_between_vertices_l955_95521

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 2| = 5

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 2.5)
def vertex2 : ℝ × ℝ := (0, -1.5)

-- Theorem statement
theorem distance_between_vertices : 
  ∀ (v1 v2 : ℝ × ℝ), 
  (∀ x y, parabola_equation x y → (x = v1.1 ∧ y = v1.2) ∨ (x = v2.1 ∧ y = v2.2)) →
  v1 = vertex1 ∧ v2 = vertex2 →
  Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 4 := by
sorry

end distance_between_vertices_l955_95521


namespace point_P_conditions_l955_95511

def point_P (m : ℝ) : ℝ × ℝ := (3*m - 6, m + 1)

def point_A : ℝ × ℝ := (-1, 2)

theorem point_P_conditions (m : ℝ) :
  (∃ m, point_P m = (-9, 0) ∧ (point_P m).2 = 0) ∧
  (∃ m, point_P m = (-1, 8/3) ∧ (point_P m).1 = (point_A).1) :=
by sorry

end point_P_conditions_l955_95511


namespace arithmetic_sequence_problem_l955_95505

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 5 = 14)
  (h_prod : a 2 * a 6 = 33) :
  a 1 * a 7 = 13 := by
  sorry

end arithmetic_sequence_problem_l955_95505


namespace andy_tomato_plants_l955_95506

theorem andy_tomato_plants :
  ∀ (P : ℕ),
  (∃ (total_tomatoes dried_tomatoes sauce_tomatoes remaining_tomatoes : ℕ),
    total_tomatoes = 7 * P ∧
    dried_tomatoes = total_tomatoes / 2 ∧
    sauce_tomatoes = (total_tomatoes - dried_tomatoes) / 3 ∧
    remaining_tomatoes = total_tomatoes - dried_tomatoes - sauce_tomatoes ∧
    remaining_tomatoes = 42) →
  P = 18 :=
by sorry

end andy_tomato_plants_l955_95506


namespace angela_has_eight_more_l955_95548

/-- The number of marbles each person has -/
structure MarbleCount where
  albert : ℕ
  angela : ℕ
  allison : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : MarbleCount) : Prop :=
  m.albert = 3 * m.angela ∧
  m.angela > m.allison ∧
  m.allison = 28 ∧
  m.albert + m.allison = 136

/-- The theorem stating that Angela has 8 more marbles than Allison -/
theorem angela_has_eight_more (m : MarbleCount) 
  (h : marble_problem m) : m.angela - m.allison = 8 := by
  sorry

end angela_has_eight_more_l955_95548


namespace equipment_marked_price_marked_price_approx_58_82_l955_95514

/-- The marked price of equipment given specific buying and selling conditions --/
theorem equipment_marked_price (original_price : ℝ) (buying_discount : ℝ) 
  (desired_gain : ℝ) (selling_discount : ℝ) : ℝ :=
  let cost_price := original_price * (1 - buying_discount)
  let selling_price := cost_price * (1 + desired_gain)
  selling_price / (1 - selling_discount)

/-- The marked price of equipment is approximately 58.82 given the specific conditions --/
theorem marked_price_approx_58_82 : 
  ∃ ε > 0, |equipment_marked_price 50 0.2 0.25 0.15 - 58.82| < ε :=
sorry

end equipment_marked_price_marked_price_approx_58_82_l955_95514


namespace isosceles_triangle_from_wire_isosceles_triangle_with_side_6_l955_95573

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Represents an isosceles triangle -/
def IsoscelesTriangle (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

theorem isosceles_triangle_from_wire (wire_length : ℝ) 
  (h_wire : wire_length = 24) :
  ∃ (t : Triangle), IsoscelesTriangle t ∧ 
    t.a + t.b + t.c = wire_length ∧
    t.a = t.b ∧ t.a = 2 * t.c ∧
    t.a = 48 / 5 := by
  sorry

theorem isosceles_triangle_with_side_6 (wire_length : ℝ) 
  (h_wire : wire_length = 24) :
  ∃ (t : Triangle), IsoscelesTriangle t ∧ 
    t.a + t.b + t.c = wire_length ∧
    (t.a = 6 ∨ t.b = 6 ∨ t.c = 6) := by
  sorry

end isosceles_triangle_from_wire_isosceles_triangle_with_side_6_l955_95573


namespace circular_cross_section_shapes_l955_95520

-- Define the shapes
inductive Shape
  | Cone
  | Cylinder
  | Sphere
  | PentagonalPrism

-- Define a function to check if a shape can have a circular cross-section
def canHaveCircularCrossSection (s : Shape) : Prop :=
  match s with
  | Shape.Cone => true
  | Shape.Cylinder => true
  | Shape.Sphere => true
  | Shape.PentagonalPrism => false

-- Theorem statement
theorem circular_cross_section_shapes :
  ∀ s : Shape, canHaveCircularCrossSection s ↔ (s = Shape.Cone ∨ s = Shape.Cylinder ∨ s = Shape.Sphere) :=
by sorry

end circular_cross_section_shapes_l955_95520


namespace ratio_of_a_to_b_l955_95500

theorem ratio_of_a_to_b (a b : ℚ) (h : (6*a - 5*b) / (8*a - 3*b) = 2/7) : 
  a/b = 29/26 := by sorry

end ratio_of_a_to_b_l955_95500


namespace population_after_three_years_l955_95574

def population_growth (initial : ℕ) (rate : ℚ) (additional : ℕ) : ℕ :=
  ⌊(initial : ℚ) * (1 + rate) + additional⌋.toNat

def three_year_population (initial : ℕ) (rate1 rate2 rate3 : ℚ) (add1 add2 add3 : ℕ) : ℕ :=
  let year1 := population_growth initial rate1 add1
  let year2 := population_growth year1 rate2 add2
  population_growth year2 rate3 add3

theorem population_after_three_years :
  three_year_population 14000 (12/100) (8/100) (6/100) 150 100 500 = 18728 :=
by sorry

end population_after_three_years_l955_95574


namespace complement_of_S_in_U_l955_95549

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the set S
def S : Set Nat := {1, 2, 3, 4}

-- Theorem statement
theorem complement_of_S_in_U : 
  (U \ S) = {5} := by sorry

end complement_of_S_in_U_l955_95549


namespace biffs_drinks_and_snacks_cost_l955_95545

/-- Represents Biff's expenses and earnings during his bus trip -/
structure BusTrip where
  ticket_cost : ℝ
  headphones_cost : ℝ
  online_rate : ℝ
  wifi_rate : ℝ
  trip_duration : ℝ

/-- Calculates the amount Biff spent on drinks and snacks -/
def drinks_and_snacks_cost (trip : BusTrip) : ℝ :=
  (trip.online_rate - trip.wifi_rate) * trip.trip_duration - 
  (trip.ticket_cost + trip.headphones_cost)

/-- Theorem stating that Biff's expenses on drinks and snacks equal $3 -/
theorem biffs_drinks_and_snacks_cost :
  let trip := BusTrip.mk 11 16 12 2 3
  drinks_and_snacks_cost trip = 3 := by
  sorry

end biffs_drinks_and_snacks_cost_l955_95545


namespace cosine_largest_angle_triangle_l955_95580

theorem cosine_largest_angle_triangle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  cos_C = -(1/4) := by
  sorry

end cosine_largest_angle_triangle_l955_95580


namespace statement_contradiction_l955_95559

-- Define the possible types of speakers
inductive Speaker
| Knight
| Liar

-- Define the statement made by A
def statement (s : Speaker) : Prop :=
  match s with
  | Speaker.Knight => s = Speaker.Liar ∨ 2 + 2 = 5
  | Speaker.Liar => ¬(s = Speaker.Liar ∨ 2 + 2 = 5)

-- Theorem stating that the conditions lead to a contradiction
theorem statement_contradiction :
  ¬∃ (s : Speaker), statement s :=
by
  sorry


end statement_contradiction_l955_95559


namespace line_through_points_with_slope_one_l955_95575

/-- Given a line passing through points M(-2, a) and N(a, 4) with a slope of 1, prove that a = 1 -/
theorem line_through_points_with_slope_one (a : ℝ) : 
  (let M := (-2, a)
   let N := (a, 4)
   (4 - a) / (a - (-2)) = 1) → 
  a = 1 := by
  sorry

end line_through_points_with_slope_one_l955_95575


namespace selection_theorem_l955_95590

def num_boys : ℕ := 5
def num_girls : ℕ := 4
def total_people : ℕ := num_boys + num_girls
def num_selected : ℕ := 4

/-- The number of ways to select 4 people from 5 boys and 4 girls, 
    ensuring at least one of boy A and girl B participates, 
    and both boys and girls are present -/
def selection_ways : ℕ := sorry

theorem selection_theorem : 
  selection_ways = (total_people.choose num_selected) - 
                   (num_boys.choose num_selected) - 
                   (num_girls.choose num_selected) := by sorry

end selection_theorem_l955_95590


namespace f_compose_three_equals_43_l955_95593

-- Define the function f
def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 1

-- Theorem statement
theorem f_compose_three_equals_43 : f (f (f 3)) = 43 := by
  sorry

end f_compose_three_equals_43_l955_95593


namespace power_of_two_ge_square_l955_95570

theorem power_of_two_ge_square (n : ℕ) (h : n ≥ 4) : 2^n ≥ n^2 := by
  sorry

end power_of_two_ge_square_l955_95570


namespace harriett_quarters_l955_95523

/-- Represents the number of coins of each type found by Harriett --/
structure CoinCount where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in cents for a given coin count --/
def totalValue (coins : CoinCount) : ℕ :=
  coins.quarters * 25 + coins.dimes * 10 + coins.nickels * 5 + coins.pennies

/-- The coin count found by Harriett --/
def harriettCoins : CoinCount := {
  quarters := 10,  -- This is what we want to prove
  dimes := 3,
  nickels := 3,
  pennies := 5
}

theorem harriett_quarters : 
  harriettCoins.quarters = 10 ∧ totalValue harriettCoins = 300 := by
  sorry

end harriett_quarters_l955_95523


namespace late_fisherman_arrival_day_l955_95539

/-- Represents the day of the week when the late fisherman arrived -/
inductive ArrivalDay
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- Calculates the number of days the late fisherman fished -/
def daysLateArrivalFished (d : ArrivalDay) : Nat :=
  match d with
  | .Monday => 5
  | .Tuesday => 4
  | .Wednesday => 3
  | .Thursday => 2
  | .Friday => 1

theorem late_fisherman_arrival_day :
  ∃ (n : Nat) (d : ArrivalDay),
    n > 0 ∧
    50 * n + 10 * (daysLateArrivalFished d) = 370 ∧
    d = ArrivalDay.Thursday :=
by sorry

end late_fisherman_arrival_day_l955_95539


namespace company_workforce_after_hiring_l955_95527

theorem company_workforce_after_hiring 
  (initial_female_percentage : Real)
  (additional_male_workers : Nat)
  (new_female_percentage : Real) :
  initial_female_percentage = 0.60 →
  additional_male_workers = 22 →
  new_female_percentage = 0.55 →
  (initial_female_percentage * (264 - additional_male_workers)) / 264 = new_female_percentage :=
by sorry

end company_workforce_after_hiring_l955_95527


namespace max_eggs_per_basket_l955_95551

/-- The number of yellow Easter eggs -/
def yellow_eggs : ℕ := 16

/-- The number of green Easter eggs -/
def green_eggs : ℕ := 28

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 4

theorem max_eggs_per_basket :
  eggs_per_basket = 4 ∧
  yellow_eggs % eggs_per_basket = 0 ∧
  green_eggs % eggs_per_basket = 0 ∧
  eggs_per_basket ≥ 2 ∧
  ∀ n : ℕ, n > eggs_per_basket →
    (yellow_eggs % n ≠ 0 ∨ green_eggs % n ≠ 0 ∨ n < 2) :=
by sorry

end max_eggs_per_basket_l955_95551


namespace initials_count_l955_95534

/-- The number of letters available (A through H) -/
def num_letters : ℕ := 8

/-- The length of each set of initials -/
def set_length : ℕ := 4

/-- The number of different four-letter sets of initials possible using letters A through H -/
theorem initials_count : (num_letters ^ set_length : ℕ) = 4096 := by
  sorry

end initials_count_l955_95534


namespace age_ratio_after_two_years_l955_95507

/-- Proves that the ratio of a man's age to his student's age after two years is 2:1,
    given that the man is 26 years older than his 24-year-old student. -/
theorem age_ratio_after_two_years (student_age : ℕ) (man_age : ℕ) : 
  student_age = 24 →
  man_age = student_age + 26 →
  (man_age + 2) / (student_age + 2) = 2 := by
sorry

end age_ratio_after_two_years_l955_95507


namespace petrol_expense_l955_95504

def monthly_expenses (rent milk groceries education misc petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + misc + petrol

def savings_percentage : ℚ := 1/10

theorem petrol_expense (rent milk groceries education misc savings : ℕ) 
  (h1 : rent = 5000)
  (h2 : milk = 1500)
  (h3 : groceries = 4500)
  (h4 : education = 2500)
  (h5 : misc = 6100)
  (h6 : savings = 2400)
  : ∃ (petrol total_salary : ℕ),
    (savings_percentage * total_salary = savings) ∧
    (monthly_expenses rent milk groceries education misc petrol + savings = total_salary) ∧
    (petrol = 2000) := by
  sorry

end petrol_expense_l955_95504


namespace system_solution_transformation_l955_95552

theorem system_solution_transformation 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h : ∃ (x y : ℝ), x = 3 ∧ y = 4 ∧ a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂) :
  ∃ (x y : ℝ), x = 5 ∧ y = 5 ∧ 3 * a₁ * x + 4 * b₁ * y = 5 * c₁ ∧ 3 * a₂ * x + 4 * b₂ * y = 5 * c₂ :=
by sorry

end system_solution_transformation_l955_95552


namespace solve_star_equation_l955_95501

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- Theorem statement
theorem solve_star_equation (x : ℝ) : star 3 x = 15 → x = 3 := by
  sorry

end solve_star_equation_l955_95501


namespace sqrt_sum_equals_2sqrt6_l955_95596

theorem sqrt_sum_equals_2sqrt6 : 
  Real.sqrt (9 - 6 * Real.sqrt 2) + Real.sqrt (9 + 6 * Real.sqrt 2) = 2 * Real.sqrt 6 := by
  sorry

end sqrt_sum_equals_2sqrt6_l955_95596


namespace intersection_count_theorem_l955_95529

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters

/-- Represents the number of intersection points between two lines -/
def intersectionCount (l1 l2 : Line3D) : ℕ := sorry

/-- Represents if two lines are skew -/
def areSkew (l1 l2 : Line3D) : Prop := sorry

/-- Represents if two lines are parallel -/
def areParallel (l1 l2 : Line3D) : Prop := sorry

/-- Represents if a line is perpendicular to two other lines -/
def isCommonPerpendicular (l l1 l2 : Line3D) : Prop := sorry

theorem intersection_count_theorem 
  (a b EF l : Line3D) 
  (h1 : isCommonPerpendicular EF a b) 
  (h2 : areSkew a b) 
  (h3 : areParallel l EF) : 
  (intersectionCount l a + intersectionCount l b = 0) ∨ 
  (intersectionCount l a + intersectionCount l b = 1) := by
  sorry

end intersection_count_theorem_l955_95529


namespace trig_expression_value_l955_95544

/-- The value of the trigonometric expression is approximately 1.481 -/
theorem trig_expression_value : 
  let expr := (2 * Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
               Real.cos (160 * π / 180) * Real.cos (110 * π / 180)) /
              (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + 
               Real.cos (156 * π / 180) * Real.cos (94 * π / 180))
  ∃ ε > 0, |expr - 1.481| < ε :=
by sorry

end trig_expression_value_l955_95544


namespace max_a_for_real_roots_l955_95522

theorem max_a_for_real_roots : ∃ (a_max : ℤ), 
  (∀ a : ℤ, (∃ x : ℝ, (a + 1 : ℝ) * x^2 - 2*x + 3 = 0) → a ≤ a_max) ∧ 
  (∃ x : ℝ, (a_max + 1 : ℝ) * x^2 - 2*x + 3 = 0) ∧ 
  a_max = -2 :=
sorry

end max_a_for_real_roots_l955_95522


namespace factorial_inequality_l955_95597

theorem factorial_inequality (n p : ℕ) (h : 2 * p ≤ n) :
  (n - p).factorial / p.factorial ≤ ((n + 1) / 2 : ℚ) ^ (n - 2 * p) ∧
  ((n - p).factorial / p.factorial = ((n + 1) / 2 : ℚ) ^ (n - 2 * p) ↔ n = 2 * p ∨ n = 2 * p + 1) :=
by sorry

end factorial_inequality_l955_95597


namespace book_cost_price_l955_95594

theorem book_cost_price (selling_price : ℝ) (profit_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 260 ∧ profit_percentage = 20 → 
  selling_price = cost_price * (1 + profit_percentage / 100) →
  cost_price = 216.67 := by
  sorry

end book_cost_price_l955_95594


namespace point_on_circle_range_l955_95583

/-- Given two points A(a,0) and B(-a,0) where a > 0, and a circle with center (2√3, 2) and radius 3,
    if there exists a point P on the circle such that ∠APB = 90°, then 1 ≤ a ≤ 7. -/
theorem point_on_circle_range (a : ℝ) (h_a_pos : a > 0) :
  (∃ P : ℝ × ℝ, (P.1 - 2 * Real.sqrt 3)^2 + (P.2 - 2)^2 = 9 ∧ 
   (P.1 - a)^2 + P.2^2 + (P.1 + a)^2 + P.2^2 = ((P.1 - a)^2 + P.2^2) + ((P.1 + a)^2 + P.2^2)) →
  1 ≤ a ∧ a ≤ 7 :=
by sorry

end point_on_circle_range_l955_95583


namespace doughnuts_given_away_l955_95509

theorem doughnuts_given_away (total_doughnuts : ℕ) (small_boxes_sold : ℕ) (large_boxes_sold : ℕ)
  (h1 : total_doughnuts = 300)
  (h2 : small_boxes_sold = 20)
  (h3 : large_boxes_sold = 10) :
  total_doughnuts - (small_boxes_sold * 6 + large_boxes_sold * 12) = 60 := by
  sorry

end doughnuts_given_away_l955_95509


namespace quadratic_inequality_range_l955_95516

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a ∈ Set.Ioi 3 ∪ Set.Iio (-1) :=
sorry

end quadratic_inequality_range_l955_95516


namespace inverse_expression_equals_one_sixth_l955_95519

theorem inverse_expression_equals_one_sixth :
  (2 + 4 * (4 - 3)⁻¹)⁻¹ = (1 : ℚ) / 6 := by sorry

end inverse_expression_equals_one_sixth_l955_95519


namespace ajay_dal_transaction_gain_l955_95557

/-- Represents the transaction of buying and selling dal -/
structure DalTransaction where
  quantity1 : ℝ
  price1 : ℝ
  quantity2 : ℝ
  price2 : ℝ
  selling_price : ℝ

/-- Calculate the total gain from a dal transaction -/
def calculate_gain (t : DalTransaction) : ℝ :=
  let total_quantity := t.quantity1 + t.quantity2
  let total_cost := t.quantity1 * t.price1 + t.quantity2 * t.price2
  let total_revenue := total_quantity * t.selling_price
  total_revenue - total_cost

/-- Theorem stating that Ajay's total gain in the dal transaction is 27.50 rs -/
theorem ajay_dal_transaction_gain :
  let t : DalTransaction := {
    quantity1 := 15,
    price1 := 14.50,
    quantity2 := 10,
    price2 := 13,
    selling_price := 15
  }
  calculate_gain t = 27.50 := by
  sorry

end ajay_dal_transaction_gain_l955_95557


namespace last_digit_379_base_4_l955_95562

def last_digit_base_4 (n : ℕ) : ℕ := n % 4

theorem last_digit_379_base_4 :
  last_digit_base_4 379 = 3 := by
  sorry

end last_digit_379_base_4_l955_95562


namespace retailer_profit_percentage_l955_95564

/-- Calculates the profit percentage for a retailer given wholesale price, retail price, and discount percentage. -/
def profit_percentage (wholesale_price retail_price discount_percent : ℚ) : ℚ :=
  let discount := discount_percent * retail_price / 100
  let selling_price := retail_price - discount
  let profit := selling_price - wholesale_price
  (profit / wholesale_price) * 100

/-- Theorem stating that given the specific conditions, the profit percentage is 20%. -/
theorem retailer_profit_percentage :
  let wholesale_price : ℚ := 81
  let retail_price : ℚ := 108
  let discount_percent : ℚ := 10
  profit_percentage wholesale_price retail_price discount_percent = 20 := by
sorry

#eval profit_percentage 81 108 10

end retailer_profit_percentage_l955_95564


namespace units_digit_of_7_pow_3_pow_5_l955_95554

theorem units_digit_of_7_pow_3_pow_5 : 7^(3^5) % 10 = 3 := by
  sorry

end units_digit_of_7_pow_3_pow_5_l955_95554


namespace negation_of_universal_quadratic_inequality_l955_95525

theorem negation_of_universal_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end negation_of_universal_quadratic_inequality_l955_95525


namespace lcm_18_24_l955_95524

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end lcm_18_24_l955_95524


namespace inequality_proof_l955_95585

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l955_95585


namespace sum_odd_sequence_to_99_l955_95566

/-- Sum of arithmetic sequence -/
def sum_arithmetic_sequence (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: Sum of sequence 1+3+5+...+99 -/
theorem sum_odd_sequence_to_99 :
  sum_arithmetic_sequence 1 99 2 = 2500 := by
  sorry

end sum_odd_sequence_to_99_l955_95566


namespace total_area_is_200_l955_95587

/-- The total area of Yanni's paintings -/
def total_area : ℕ :=
  let painting1_count : ℕ := 3
  let painting1_width : ℕ := 5
  let painting1_height : ℕ := 5
  let painting2_width : ℕ := 10
  let painting2_height : ℕ := 8
  let painting3_width : ℕ := 9
  let painting3_height : ℕ := 5
  (painting1_count * painting1_width * painting1_height) +
  (painting2_width * painting2_height) +
  (painting3_width * painting3_height)

/-- Theorem stating that the total area of Yanni's paintings is 200 square feet -/
theorem total_area_is_200 : total_area = 200 := by
  sorry

end total_area_is_200_l955_95587


namespace mandy_pieces_l955_95503

def chocolate_distribution (total : Nat) (n : Nat) : Nat :=
  if n = 0 then
    total
  else
    chocolate_distribution (total / 2) (n - 1)

theorem mandy_pieces : chocolate_distribution 60 3 = 8 := by
  sorry

end mandy_pieces_l955_95503


namespace constant_function_inequality_l955_95565

theorem constant_function_inequality (f : ℝ → ℝ) :
  (∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2*y + 3*z)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by sorry

end constant_function_inequality_l955_95565


namespace equation_proof_l955_95578

theorem equation_proof : (49 : ℚ) / (7 - 3 / 4) = 196 / 25 := by
  sorry

end equation_proof_l955_95578


namespace cubic_equation_one_real_root_l955_95579

theorem cubic_equation_one_real_root :
  ∃! x : ℝ, x^3 - Real.sqrt 3 * x^2 + x - (1 + Real.sqrt 3 / 9) = 0 := by
  sorry

end cubic_equation_one_real_root_l955_95579


namespace right_triangle_with_special_point_l955_95571

theorem right_triangle_with_special_point (A B C P : ℝ × ℝ) 
  (h_right : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0)
  (h_AP : Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 2)
  (h_BP : Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 1)
  (h_CP : Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) = Real.sqrt 5)
  (h_inside : ∃ (t u : ℝ), t > 0 ∧ u > 0 ∧ t + u < 1 ∧ 
    P.1 = t * B.1 + u * C.1 + (1 - t - u) * A.1 ∧
    P.2 = t * B.2 + u * C.2 + (1 - t - u) * A.2) :
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 5 := by
  sorry

#check right_triangle_with_special_point

end right_triangle_with_special_point_l955_95571


namespace intersection_line_slope_l955_95558

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 40 = 0

-- Define the intersection points
def intersection_points (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle2 C.1 C.2 ∧
  circle1 D.1 D.2 ∧ circle2 D.1 D.2 ∧
  C ≠ D

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) 
  (h : intersection_points C D) : 
  (D.2 - C.2) / (D.1 - C.1) = 5/2 := by
  sorry

end intersection_line_slope_l955_95558


namespace coefficient_a2_l955_95535

theorem coefficient_a2 (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, a₀ + a₁ * (2 * x - 1) + a₂ * (2 * x - 1)^2 + a₃ * (2 * x - 1)^3 + a₄ * (2 * x - 1)^4 = x^4) →
  a₂ = 3/8 := by
  sorry

end coefficient_a2_l955_95535


namespace problem_solution_l955_95513

theorem problem_solution (x a : ℝ) :
  (a > 0) →
  (∀ x, (x^2 - 4*x + 3 < 0 ∧ x^2 - x - 12 ≤ 0 ∧ x^2 + 2*x - 8 > 0) → (2 < x ∧ x < 3)) ∧
  ((∀ x, (x^2 - 4*a*x + 3*a^2 ≥ 0) → (x^2 - x - 12 > 0 ∨ x^2 + 2*x - 8 ≤ 0)) ∧
   (∃ x, (x^2 - x - 12 > 0 ∨ x^2 + 2*x - 8 ≤ 0) ∧ x^2 - 4*a*x + 3*a^2 < 0) →
   (1 ≤ a ∧ a ≤ 2)) :=
by sorry

end problem_solution_l955_95513


namespace student_assistant_sequences_l955_95588

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 5

/-- The number of different sequences of student assistants possible in one week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

theorem student_assistant_sequences :
  num_sequences = 759375 :=
by sorry

end student_assistant_sequences_l955_95588


namespace library_visitors_average_l955_95526

/-- Calculates the average number of visitors per day in a 30-day month starting with a Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let numSundays := 4
  let numOtherDays := 26
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / 30

theorem library_visitors_average :
  averageVisitorsPerDay 500 140 = 188 := by
  sorry

end library_visitors_average_l955_95526


namespace product_simplification_l955_95543

theorem product_simplification :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end product_simplification_l955_95543


namespace hilton_marbles_l955_95599

/-- Calculates the final number of marbles Hilton has -/
def final_marbles (initial : ℝ) (found : ℝ) (lost : ℝ) (compensation_rate : ℝ) : ℝ :=
  initial + found - lost + compensation_rate * lost

/-- Proves that Hilton ends up with 44.5 marbles given the initial conditions -/
theorem hilton_marbles :
  final_marbles 30 8.5 12 1.5 = 44.5 := by
  sorry

end hilton_marbles_l955_95599


namespace cube_volume_puzzle_l955_95595

theorem cube_volume_puzzle (a : ℝ) : 
  a > 0 → 
  (a + 2) * (a - 2) * a = a^3 - 8 → 
  a^3 = 8 := by
sorry

end cube_volume_puzzle_l955_95595


namespace two_digit_number_system_l955_95538

theorem two_digit_number_system (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 →
  (10 * x + y) - 3 * (x + y) = 13 →
  (10 * x + y) % (x + y) = 6 →
  (10 * x + y) / (x + y) = 4 →
  (10 * x + y - 3 * (x + y) = 13 ∧ 10 * x + y - 6 = 4 * (x + y)) := by
  sorry

end two_digit_number_system_l955_95538


namespace cube_volume_from_painting_cost_l955_95561

/-- Given a cube where the cost of painting its entire surface area is Rs. 343.98
    at a rate of 13 paise per sq. cm, the volume of the cube is 9261 cubic cm. -/
theorem cube_volume_from_painting_cost (cost : ℚ) (rate : ℚ) (volume : ℚ) : 
  cost = 343.98 →
  rate = 13 / 100 →
  volume = (((cost * 100) / rate / 6).sqrt ^ 3) →
  volume = 9261 := by sorry

end cube_volume_from_painting_cost_l955_95561


namespace triangle_area_l955_95555

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  f (A / 2) = -Real.sqrt 3 / 2 →
  a = 3 →
  b + c = 2 * Real.sqrt 3 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 4 :=
by sorry

end triangle_area_l955_95555


namespace value_of_M_l955_95512

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1504) ∧ (M = 2105.6) := by sorry

end value_of_M_l955_95512


namespace function_difference_theorem_l955_95576

theorem function_difference_theorem (p q c : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  let f : ℝ → ℝ := λ x => p * x^6 + q * x^4 + 3 * x - Real.sqrt 2
  let d := f c - f (-c)
  d = 6 * c := by sorry

end function_difference_theorem_l955_95576


namespace sequence_constant_iff_perfect_square_l955_95591

/-- S(n) is defined as n minus the largest perfect square not exceeding n -/
def S (n : ℕ) : ℕ :=
  n - (Nat.sqrt n) ^ 2

/-- The sequence a_k is defined recursively -/
def a (A : ℕ) : ℕ → ℕ
  | 0 => A
  | k + 1 => a A k + S (a A k)

/-- A positive integer A makes the sequence eventually constant
    if and only if A is a perfect square -/
theorem sequence_constant_iff_perfect_square (A : ℕ) (h : A > 0) :
  (∃ N : ℕ, ∀ k ≥ N, a A k = a A N) ↔ ∃ m : ℕ, A = m^2 := by
  sorry

end sequence_constant_iff_perfect_square_l955_95591


namespace mean_of_combined_sets_l955_95528

theorem mean_of_combined_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 20 →
  (set1_count * set1_mean + set2_count * set2_mean) / (set1_count + set2_count) = 53 / 3 :=
by
  sorry

end mean_of_combined_sets_l955_95528


namespace handshake_theorem_l955_95568

def number_of_people : ℕ := 12
def handshakes_per_person : ℕ := 3

def handshake_arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

theorem handshake_theorem :
  let M := handshake_arrangements number_of_people handshakes_per_person
  M = 6100940 ∧ M % 1000 = 940 := by sorry

end handshake_theorem_l955_95568


namespace fraction_equality_l955_95536

theorem fraction_equality (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (1 / a + 1 / b = 4 / (a + b)) → (a / b + b / a = 2) := by
  sorry

end fraction_equality_l955_95536


namespace total_medicine_boxes_l955_95540

def vitamins : ℕ := 472
def supplements : ℕ := 288

theorem total_medicine_boxes : vitamins + supplements = 760 := by
  sorry

end total_medicine_boxes_l955_95540


namespace geometric_sequence_sum_l955_95560

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 + a 2 = 30) →
  (a 3 + a 4 = 120) →
  (a 5 + a 6 = 480) := by
  sorry

end geometric_sequence_sum_l955_95560


namespace circle_area_through_point_l955_95532

/-- The area of a circle with center P(2, 5) passing through point Q(6, -1) is 52π. -/
theorem circle_area_through_point (P Q : ℝ × ℝ) : P = (2, 5) → Q = (6, -1) → 
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 52 * π := by
  sorry

end circle_area_through_point_l955_95532


namespace min_value_quadratic_l955_95533

theorem min_value_quadratic (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 - 8*x₀ + 6*y₀ + 25 = 0 :=
by sorry

end min_value_quadratic_l955_95533


namespace paul_remaining_money_l955_95553

/-- The amount of money Paul had for shopping -/
def initial_money : ℕ := 15

/-- The cost of bread -/
def bread_cost : ℕ := 2

/-- The cost of butter -/
def butter_cost : ℕ := 3

/-- The cost of juice (twice the price of bread) -/
def juice_cost : ℕ := 2 * bread_cost

/-- The total cost of groceries -/
def total_cost : ℕ := bread_cost + butter_cost + juice_cost

/-- The remaining money after shopping -/
def remaining_money : ℕ := initial_money - total_cost

theorem paul_remaining_money :
  remaining_money = 6 :=
sorry

end paul_remaining_money_l955_95553


namespace blanket_collection_proof_l955_95537

/-- Calculates the total number of blankets collected over three days -/
def totalBlankets (teamSize : ℕ) (firstDayPerPerson : ℕ) (secondDayMultiplier : ℕ) (thirdDayTotal : ℕ) : ℕ :=
  let firstDay := teamSize * firstDayPerPerson
  let secondDay := firstDay * secondDayMultiplier
  firstDay + secondDay + thirdDayTotal

/-- Proves that the total number of blankets collected is 142 given the specific conditions -/
theorem blanket_collection_proof :
  totalBlankets 15 2 3 22 = 142 := by
  sorry

end blanket_collection_proof_l955_95537


namespace hyperbola_asymptote_l955_95530

/-- Given a hyperbola with equation x²/a² - y²/16 = 1 where a > 0,
    if one of its asymptotes has equation 2x - y = 0, then a = 2 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2 / a^2 - y^2 / 16 = 1 ∧ 2*x - y = 0) → a = 2 := by
  sorry

end hyperbola_asymptote_l955_95530


namespace tangent_sum_identity_l955_95541

theorem tangent_sum_identity : 
  Real.sqrt 3 * Real.tan (12 * π / 180) + 
  Real.sqrt 3 * Real.tan (18 * π / 180) + 
  Real.tan (12 * π / 180) * Real.tan (18 * π / 180) = 1 := by
sorry

end tangent_sum_identity_l955_95541


namespace rectangle_rotation_path_length_l955_95584

/-- The length of the path traveled by point A in a rectangle ABCD undergoing three 90° rotations -/
theorem rectangle_rotation_path_length (AB BC : ℝ) (hAB : AB = 3) (hBC : BC = 8) :
  let diagonal := Real.sqrt (AB^2 + BC^2)
  let first_rotation := (1/2) * π * diagonal
  let second_rotation := (3/2) * π
  let third_rotation := 4 * π
  first_rotation + second_rotation + third_rotation = ((1/2) * Real.sqrt 73 + 11/2) * π :=
sorry

end rectangle_rotation_path_length_l955_95584


namespace equal_angles_necessary_not_sufficient_l955_95515

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  sorry -- Definition of a square

-- Define the property of having four equal interior angles
def has_four_equal_angles (q : Quadrilateral) : Prop :=
  sorry -- Definition of having four equal interior angles

theorem equal_angles_necessary_not_sufficient :
  (∀ q : Quadrilateral, is_square q → has_four_equal_angles q) ∧
  (∃ q : Quadrilateral, has_four_equal_angles q ∧ ¬is_square q) :=
sorry

end equal_angles_necessary_not_sufficient_l955_95515


namespace noahs_age_ratio_l955_95510

theorem noahs_age_ratio (joe_age : ℕ) (noah_future_age : ℕ) (years_to_future : ℕ) :
  joe_age = 6 →
  noah_future_age = 22 →
  years_to_future = 10 →
  ∃ k : ℕ, k * joe_age = noah_future_age - years_to_future →
  (noah_future_age - years_to_future) / joe_age = 2 := by
sorry

end noahs_age_ratio_l955_95510


namespace arithmetic_geometric_mean_ratio_l955_95550

theorem arithmetic_geometric_mean_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_mean : (x + y) / 2 = 3 * Real.sqrt (x * y)) : 
  ∃ (n : ℤ), ∀ (m : ℤ), |x / y - n| ≤ |x / y - m| ∧ n = 34 :=
sorry

end arithmetic_geometric_mean_ratio_l955_95550


namespace chenny_friends_l955_95531

/-- The number of friends Chenny has -/
def num_friends (initial_candies : ℕ) (bought_candies : ℕ) (candies_per_friend : ℕ) : ℕ :=
  (initial_candies + bought_candies) / candies_per_friend

/-- Proof that Chenny has 7 friends -/
theorem chenny_friends : num_friends 10 4 2 = 7 := by
  sorry

end chenny_friends_l955_95531


namespace quadruple_primes_l955_95592

theorem quadruple_primes (p q r : ℕ) (n : ℕ+) : 
  (Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p^2 = q^2 + r^(n : ℕ)) ↔ 
  ((p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4)) :=
sorry

end quadruple_primes_l955_95592


namespace total_commute_time_l955_95586

def first_bus_duration : ℕ := 40
def first_wait_duration : ℕ := 10
def second_bus_duration : ℕ := 50
def second_wait_duration : ℕ := 15
def third_bus_duration : ℕ := 95

theorem total_commute_time :
  first_bus_duration + first_wait_duration + second_bus_duration +
  second_wait_duration + third_bus_duration = 210 := by
  sorry

end total_commute_time_l955_95586


namespace minimum_other_sales_l955_95567

/-- Represents the sales distribution of a stationery store -/
structure SalesDistribution where
  pens : ℝ
  pencils : ℝ
  other : ℝ

/-- The sales distribution meets the store's goals -/
def MeetsGoals (s : SalesDistribution) : Prop :=
  s.pens = 40 ∧
  s.pencils = 28 ∧
  s.other ≥ 20 ∧
  s.pens + s.pencils + s.other = 100

theorem minimum_other_sales (s : SalesDistribution) (h : MeetsGoals s) :
  s.other = 32 ∧ s.pens + s.pencils + s.other = 100 := by
  sorry

#check minimum_other_sales

end minimum_other_sales_l955_95567


namespace jack_ernie_income_ratio_l955_95547

theorem jack_ernie_income_ratio :
  ∀ (ernie_prev ernie_curr jack_curr : ℝ),
    ernie_curr = (4/5) * ernie_prev →
    ernie_curr + jack_curr = 16800 →
    ernie_prev = 6000 →
    jack_curr / ernie_prev = 2 := by
  sorry

end jack_ernie_income_ratio_l955_95547


namespace paint_time_problem_l955_95546

theorem paint_time_problem (anthony_time : ℝ) (combined_time : ℝ) (first_person_time : ℝ) : 
  anthony_time = 5 →
  combined_time = 20 / 7 →
  (1 / first_person_time + 1 / anthony_time) * combined_time = 2 →
  first_person_time = 2 := by
sorry

end paint_time_problem_l955_95546


namespace article_cost_l955_95569

/-- Calculates the final cost of an article after two years of inflation and price changes -/
def finalCost (originalCost : ℝ) (inflationRate : ℝ) 
  (year1Increase year1Decrease year2Increase year2Decrease : ℝ) : ℝ :=
  let adjustedCost1 := originalCost * (1 + inflationRate)
  let afterYear1 := adjustedCost1 * (1 + year1Increase) * (1 - year1Decrease)
  let adjustedCost2 := afterYear1 * (1 + inflationRate)
  adjustedCost2 * (1 + year2Increase) * (1 - year2Decrease)

/-- Theorem stating the final cost of the article after two years -/
theorem article_cost : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |finalCost 75 0.05 0.20 0.20 0.30 0.25 - 77.40| < ε :=
sorry

end article_cost_l955_95569


namespace smallest_k_with_remainder_l955_95542

theorem smallest_k_with_remainder (k : ℕ) : k = 534 ↔ 
  (k > 2) ∧ 
  (k % 19 = 2) ∧ 
  (k % 7 = 2) ∧ 
  (k % 4 = 2) ∧ 
  (∀ m : ℕ, m > 2 ∧ m % 19 = 2 ∧ m % 7 = 2 ∧ m % 4 = 2 → k ≤ m) :=
by sorry

end smallest_k_with_remainder_l955_95542


namespace problem_solution_l955_95581

theorem problem_solution (x : ℝ) : (400 * 7000 : ℝ) = 28000 * (100 ^ x) → x = 1 := by
  sorry

end problem_solution_l955_95581


namespace special_sum_equals_250_l955_95572

/-- The sum of two arithmetic sequences with 5 terms each, where the first sequence starts at 3 and increases by 10, and the second sequence starts at 7 and increases by 10 -/
def special_sum : ℕ := (3+13+23+33+43)+(7+17+27+37+47)

/-- Theorem stating that the special sum equals 250 -/
theorem special_sum_equals_250 : special_sum = 250 := by
  sorry

end special_sum_equals_250_l955_95572


namespace stock_transaction_profit_l955_95518

/-- Represents a stock transaction and calculates the profit -/
def stock_transaction (initial_shares : ℕ) (initial_price : ℚ) (sold_shares : ℕ) (selling_price : ℚ) : ℚ :=
  let initial_cost := initial_shares * initial_price
  let sale_revenue := sold_shares * selling_price
  let remaining_shares := initial_shares - sold_shares
  let final_value := sale_revenue + (remaining_shares * (2 * initial_price))
  final_value - initial_cost

/-- Proves that the profit from the given stock transaction is $40 -/
theorem stock_transaction_profit :
  stock_transaction 20 3 10 4 = 40 := by
  sorry

end stock_transaction_profit_l955_95518


namespace inequality_solution_set_l955_95563

theorem inequality_solution_set : 
  ∀ x : ℝ, abs (x - 4) + abs (3 - x) < 2 ↔ 2.5 < x ∧ x < 4.5 :=
by sorry

end inequality_solution_set_l955_95563


namespace janet_action_figures_l955_95556

/-- The number of action figures Janet initially owns -/
def initial_figures : ℕ := 10

/-- The number of new action figures Janet buys -/
def new_figures : ℕ := 4

/-- The total number of action figures Janet has at the end -/
def total_figures : ℕ := 24

/-- The number of action figures Janet sold -/
def sold_figures : ℕ := 6

theorem janet_action_figures :
  ∃ (x : ℕ),
    x = sold_figures ∧
    initial_figures - x + new_figures +
    2 * (initial_figures - x + new_figures) = total_figures :=
by sorry

end janet_action_figures_l955_95556


namespace three_numbers_sum_to_perfect_square_l955_95577

def numbers : List Nat := [4784887, 2494651, 8595087, 1385287, 9042451, 9406087]

theorem three_numbers_sum_to_perfect_square :
  ∃ (a b c : Nat), a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  ∃ (n : Nat), a + b + c = n * n :=
by
  sorry

end three_numbers_sum_to_perfect_square_l955_95577


namespace probability_same_heads_l955_95598

/-- Represents the outcome of tossing two coins -/
inductive CoinToss
| HH -- Two heads
| HT -- Head then tail
| TH -- Tail then head
| TT -- Two tails

/-- The sample space of all possible outcomes when two people each toss two coins -/
def sampleSpace : List (CoinToss × CoinToss) :=
  [(CoinToss.HH, CoinToss.HH), (CoinToss.HH, CoinToss.HT), (CoinToss.HH, CoinToss.TH), (CoinToss.HH, CoinToss.TT),
   (CoinToss.HT, CoinToss.HH), (CoinToss.HT, CoinToss.HT), (CoinToss.HT, CoinToss.TH), (CoinToss.HT, CoinToss.TT),
   (CoinToss.TH, CoinToss.HH), (CoinToss.TH, CoinToss.HT), (CoinToss.TH, CoinToss.TH), (CoinToss.TH, CoinToss.TT),
   (CoinToss.TT, CoinToss.HH), (CoinToss.TT, CoinToss.HT), (CoinToss.TT, CoinToss.TH), (CoinToss.TT, CoinToss.TT)]

/-- Counts the number of heads in a single coin toss -/
def countHeads : CoinToss → Nat
  | CoinToss.HH => 2
  | CoinToss.HT => 1
  | CoinToss.TH => 1
  | CoinToss.TT => 0

/-- Checks if two coin tosses have the same number of heads -/
def sameHeads : CoinToss × CoinToss → Bool
  | (t1, t2) => countHeads t1 = countHeads t2

/-- The probability of getting the same number of heads -/
theorem probability_same_heads :
  (sampleSpace.filter sameHeads).length / sampleSpace.length = 3 / 8 := by
  sorry


end probability_same_heads_l955_95598


namespace max_value_xyz_l955_95582

theorem max_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + 2*y + 3*z = 1) :
  x^3 * y^2 * z ≤ 2048 / 11^6 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ + 2*y₀ + 3*z₀ = 1 ∧ x₀^3 * y₀^2 * z₀ = 2048 / 11^6 :=
by sorry

end max_value_xyz_l955_95582


namespace pizza_theorem_l955_95508

/-- Represents a pizza with given topping distributions -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  olive_slices : ℕ
  all_toppings_slices : ℕ

/-- Conditions for a valid pizza configuration -/
def is_valid_pizza (p : Pizza) : Prop :=
  p.total_slices = 20 ∧
  p.pepperoni_slices = 12 ∧
  p.mushroom_slices = 14 ∧
  p.olive_slices = 12 ∧
  p.all_toppings_slices ≤ p.total_slices ∧
  p.all_toppings_slices ≤ p.pepperoni_slices ∧
  p.all_toppings_slices ≤ p.mushroom_slices ∧
  p.all_toppings_slices ≤ p.olive_slices

theorem pizza_theorem (p : Pizza) (h : is_valid_pizza p) : p.all_toppings_slices = 6 := by
  sorry

end pizza_theorem_l955_95508


namespace book_original_price_l955_95502

/-- Given a book sold for $78 with a 30% profit, prove that the original price was $60 -/
theorem book_original_price (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 78 → profit_percentage = 30 → 
  ∃ (original_price : ℝ), 
    original_price = 60 ∧ 
    selling_price = original_price * (1 + profit_percentage / 100) := by
  sorry

#check book_original_price

end book_original_price_l955_95502
