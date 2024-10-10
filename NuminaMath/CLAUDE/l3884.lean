import Mathlib

namespace M_mod_1000_l3884_388457

def M : ℕ := Nat.choose 14 8

theorem M_mod_1000 : M % 1000 = 3 := by
  sorry

end M_mod_1000_l3884_388457


namespace equilateral_triangle_area_l3884_388403

/-- An equilateral triangle with height 9 and perimeter 36 has area 54 -/
theorem equilateral_triangle_area (h : ℝ) (p : ℝ) :
  h = 9 → p = 36 → (1/2) * (p/3) * h = 54 := by
  sorry

end equilateral_triangle_area_l3884_388403


namespace no_bounded_ratio_interval_l3884_388406

theorem no_bounded_ratio_interval (a : ℝ) (ha : a > 0) :
  ¬∃ (b c : ℝ) (hbc : b < c),
    ∀ (x y : ℝ) (hx : b < x ∧ x < c) (hy : b < y ∧ y < c) (hxy : x ≠ y),
      |((x + y) / (x - y))| ≤ a :=
sorry

end no_bounded_ratio_interval_l3884_388406


namespace rationalize_sqrt_five_twelfths_l3884_388418

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end rationalize_sqrt_five_twelfths_l3884_388418


namespace self_inverse_fourth_power_congruence_l3884_388432

theorem self_inverse_fourth_power_congruence (n : ℕ+) (a : ℤ) 
  (h : a * a ≡ 1 [ZMOD n]) : 
  a^4 ≡ 1 [ZMOD n] := by
  sorry

end self_inverse_fourth_power_congruence_l3884_388432


namespace intersecting_line_equation_l3884_388411

/-- A line passing through a point and intersecting both axes -/
structure IntersectingLine where
  P : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  passes_through_P : true  -- Placeholder for the condition that the line passes through P
  intersects_x_axis : A.2 = 0
  intersects_y_axis : B.1 = 0
  P_is_midpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- The equation of the line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

theorem intersecting_line_equation (l : IntersectingLine) (eq : LineEquation) :
  l.P = (-4, 6) →
  eq.a = 3 ∧ eq.b = -2 ∧ eq.c = 24 :=
by sorry

end intersecting_line_equation_l3884_388411


namespace marathon_run_solution_l3884_388423

/-- Represents the marathon run problem -/
def marathon_run (x : ℝ) : Prop :=
  let total_distance : ℝ := 95
  let total_time : ℝ := 15
  let speed1 : ℝ := 8
  let speed2 : ℝ := 6
  let speed3 : ℝ := 5
  (speed1 * x + speed2 * x + speed3 * (total_time - 2 * x) = total_distance) ∧
  (x ≥ 0) ∧ (x ≤ total_time / 2)

/-- Proves that the only solution to the marathon run problem is 5 hours at each speed -/
theorem marathon_run_solution :
  ∃! x : ℝ, marathon_run x ∧ x = 5 := by sorry

end marathon_run_solution_l3884_388423


namespace circle_symmetry_l3884_388438

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 4*y + 19 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 5)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_circle x₁ y₁ →
    symmetric_circle x₂ y₂ →
    ∃ (x_mid y_mid : ℝ),
      symmetry_line x_mid y_mid ∧
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 :=
sorry

end circle_symmetry_l3884_388438


namespace ellipse_condition_l3884_388417

/-- The equation m(x^2 + y^2 + 2y + 1) = (x - 2y + 3)^2 represents an ellipse if and only if m ∈ (5, +∞) -/
theorem ellipse_condition (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) ↔ m > 5 :=
by sorry

end ellipse_condition_l3884_388417


namespace down_payment_percentage_l3884_388425

def house_price : ℝ := 100000
def parents_contribution_rate : ℝ := 0.30
def remaining_balance : ℝ := 56000

theorem down_payment_percentage :
  ∃ (down_payment_rate : ℝ),
    down_payment_rate * house_price +
    parents_contribution_rate * (house_price - down_payment_rate * house_price) +
    remaining_balance = house_price ∧
    down_payment_rate = 0.20 := by
  sorry

end down_payment_percentage_l3884_388425


namespace last_digit_tower3_5_l3884_388412

/-- The last digit of a number n -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Exponentiation modulo m -/
def powMod (base exp m : ℕ) : ℕ :=
  (base ^ exp) % m

/-- The tower of powers of 3 with height 5 -/
def tower3_5 : ℕ := 3^(3^(3^(3^3)))

/-- The last digit of the tower of powers of 3 with height 5 is 7 -/
theorem last_digit_tower3_5 : lastDigit tower3_5 = 7 := by
  sorry

end last_digit_tower3_5_l3884_388412


namespace quadratic_equation_two_distinct_roots_l3884_388490

theorem quadratic_equation_two_distinct_roots (c d : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁ + c) * (x₁ + d) - (2 * x₁ + c + d) = 0 ∧
  (x₂ + c) * (x₂ + d) - (2 * x₂ + c + d) = 0 :=
by sorry

end quadratic_equation_two_distinct_roots_l3884_388490


namespace expression_equals_zero_l3884_388453

theorem expression_equals_zero : (-3)^3 + (-3)^2 * 3^1 + 3^2 * (-3)^1 + 3^3 = 0 := by
  sorry

end expression_equals_zero_l3884_388453


namespace quadratic_one_solution_l3884_388455

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 - 6 * x + m = 0) → m = 3 := by
  sorry

end quadratic_one_solution_l3884_388455


namespace min_value_tangent_sum_l3884_388492

theorem min_value_tangent_sum (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π / 2) :
  3 * Real.tan B * Real.tan C + 2 * Real.tan A * Real.tan C + Real.tan A * Real.tan B ≥ 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 2 + 2 * Real.sqrt 6 :=
by sorry

end min_value_tangent_sum_l3884_388492


namespace tank_plastering_cost_l3884_388405

/-- Calculates the total cost of plastering a rectangular tank -/
def plasteringCost (length width depth rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

theorem tank_plastering_cost :
  let length : ℝ := 25
  let width : ℝ := 12
  let depth : ℝ := 6
  let rate : ℝ := 0.55  -- 55 paise converted to rupees
  plasteringCost length width depth rate = 409.2 := by
  sorry

end tank_plastering_cost_l3884_388405


namespace shortest_distance_to_start_l3884_388482

/-- Proof of the shortest distance between the third meeting point and the starting point on a circular track -/
theorem shortest_distance_to_start (track_length : ℝ) (time : ℝ) (speed_diff : ℝ) : 
  track_length = 400 →
  time = 8 * 60 →
  speed_diff = 0.1 →
  ∃ (speed_b : ℝ), 
    time * (speed_b + speed_b + speed_diff) = track_length * 3 ∧
    (time * speed_b) % track_length = 176 := by
  sorry

end shortest_distance_to_start_l3884_388482


namespace cost_of_dozen_pens_cost_of_dozen_pens_is_720_l3884_388415

/-- The cost of one dozen pens given the cost of 3 pens and 5 pencils and the ratio of pen to pencil cost -/
theorem cost_of_dozen_pens (total_cost : ℕ) (ratio_pen_pencil : ℕ) : ℕ :=
  let pen_cost := ratio_pen_pencil * (total_cost / (3 * ratio_pen_pencil + 5))
  12 * pen_cost

/-- Proof that the cost of one dozen pens is 720 given the conditions -/
theorem cost_of_dozen_pens_is_720 :
  cost_of_dozen_pens 240 5 = 720 := by
  sorry

end cost_of_dozen_pens_cost_of_dozen_pens_is_720_l3884_388415


namespace one_third_percentage_l3884_388437

-- Define the given numbers
def total : ℚ := 1206
def divisor : ℚ := 3
def base : ℚ := 134

-- Define one-third of the total
def one_third : ℚ := total / divisor

-- Define the percentage calculation
def percentage : ℚ := (one_third / base) * 100

-- Theorem to prove
theorem one_third_percentage : percentage = 300 := by
  sorry

end one_third_percentage_l3884_388437


namespace arithmetic_geometric_inequality_l3884_388400

theorem arithmetic_geometric_inequality 
  (a b c d h k : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_h : 0 < h) (pos_k : 0 < k)
  (arith_prog : ∃ t : ℝ, t > 0 ∧ a = d + 3*t ∧ b = d + 2*t ∧ c = d + t)
  (geom_prog : ∃ r : ℝ, r > 1 ∧ a = d * r^3 ∧ h = d * r^2 ∧ k = d * r) :
  b * c > h * k := by
  sorry

end arithmetic_geometric_inequality_l3884_388400


namespace questionnaire_responses_l3884_388414

theorem questionnaire_responses (response_rate : ℝ) (min_questionnaires : ℕ) (responses_needed : ℕ) : 
  response_rate = 0.60 → 
  min_questionnaires = 370 → 
  responses_needed = ⌊response_rate * min_questionnaires⌋ →
  responses_needed = 222 := by
sorry

end questionnaire_responses_l3884_388414


namespace quadratic_has_real_root_l3884_388497

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + a - 1 = 0 := by
  sorry

end quadratic_has_real_root_l3884_388497


namespace only_rational_root_l3884_388439

def polynomial (x : ℚ) : ℚ := 6 * x^4 - 5 * x^3 - 17 * x^2 + 7 * x + 3

theorem only_rational_root :
  ∀ x : ℚ, polynomial x = 0 ↔ x = 1/2 := by sorry

end only_rational_root_l3884_388439


namespace pizza_toppings_l3884_388484

theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (pepperoni_slices : ℕ)
  (h1 : total_slices = 16)
  (h2 : cheese_slices = 10)
  (h3 : pepperoni_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range cheese_slices ∨ slice ∈ Finset.range pepperoni_slices)) :
  cheese_slices + pepperoni_slices - total_slices = 6 :=
by sorry

end pizza_toppings_l3884_388484


namespace power_sum_equals_fourteen_l3884_388474

theorem power_sum_equals_fourteen : 2^3 + 2^2 + 2^1 = 14 := by
  sorry

end power_sum_equals_fourteen_l3884_388474


namespace arithmetic_calculation_l3884_388494

theorem arithmetic_calculation : 12 - 10 + 8 / 2 * 5 + 4 - 6 * 3 + 1 = 9 := by
  sorry

end arithmetic_calculation_l3884_388494


namespace max_y_value_l3884_388463

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -6) : y ≤ 3 := by
  sorry

end max_y_value_l3884_388463


namespace hamburger_cost_is_correct_l3884_388485

/-- The cost of a hamburger given the conditions of Robert and Teddy's snack purchase --/
def hamburger_cost : ℚ :=
  let pizza_box_cost : ℚ := 10
  let soft_drink_cost : ℚ := 2
  let robert_pizza_boxes : ℕ := 5
  let robert_soft_drinks : ℕ := 10
  let teddy_hamburgers : ℕ := 6
  let teddy_soft_drinks : ℕ := 10
  let total_spent : ℚ := 106

  let robert_spent : ℚ := pizza_box_cost * robert_pizza_boxes + soft_drink_cost * robert_soft_drinks
  let teddy_spent : ℚ := total_spent - robert_spent
  let teddy_hamburgers_cost : ℚ := teddy_spent - soft_drink_cost * teddy_soft_drinks

  teddy_hamburgers_cost / teddy_hamburgers

theorem hamburger_cost_is_correct :
  hamburger_cost = 267/100 := by
  sorry

end hamburger_cost_is_correct_l3884_388485


namespace two_thirds_to_tenth_bounds_l3884_388424

theorem two_thirds_to_tenth_bounds : 1/100 < (2/3)^10 ∧ (2/3)^10 < 2/100 := by
  sorry

end two_thirds_to_tenth_bounds_l3884_388424


namespace koala_fiber_intake_l3884_388466

/-- The absorption rate of fiber for koalas -/
def absorption_rate : ℝ := 0.35

/-- The amount of fiber absorbed on the first day -/
def fiber_absorbed_day1 : ℝ := 14.7

/-- The amount of fiber absorbed on the second day -/
def fiber_absorbed_day2 : ℝ := 9.8

/-- Theorem: The total amount of fiber eaten by the koala over two days is 70 ounces -/
theorem koala_fiber_intake :
  let fiber_eaten_day1 := fiber_absorbed_day1 / absorption_rate
  let fiber_eaten_day2 := fiber_absorbed_day2 / absorption_rate
  fiber_eaten_day1 + fiber_eaten_day2 = 70 := by sorry

end koala_fiber_intake_l3884_388466


namespace radish_basket_difference_l3884_388440

theorem radish_basket_difference (total : ℕ) (first_basket : ℕ) : total = 88 → first_basket = 37 → total - first_basket - first_basket = 14 := by
  sorry

end radish_basket_difference_l3884_388440


namespace two_ones_in_twelve_dice_l3884_388496

def probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem two_ones_in_twelve_dice :
  probability_two_ones 12 2 (1/6) = (66 * 5^10 : ℚ) / (36 * 6^10 : ℚ) := by
  sorry

end two_ones_in_twelve_dice_l3884_388496


namespace shaded_area_is_24_5_l3884_388495

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  bottomLeft : Point
  baseLength : ℝ

/-- Calculates the area of the shaded region -/
def shadedArea (square : Square) (triangle : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating the area of the shaded region -/
theorem shaded_area_is_24_5 (square : Square) (triangle : IsoscelesTriangle) :
  square.bottomLeft = Point.mk 0 0 →
  square.sideLength = 7 →
  triangle.bottomLeft = Point.mk 7 0 →
  triangle.baseLength = 7 →
  shadedArea square triangle = 24.5 :=
by
  sorry

end shaded_area_is_24_5_l3884_388495


namespace cat_finishes_food_on_sunday_l3884_388419

/-- Represents days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the amount of food eaten by the cat -/
def cat_food_eaten (d : Day) : Rat :=
  match d with
  | Day.Monday => 5/6
  | Day.Tuesday => 10/6
  | Day.Wednesday => 15/6
  | Day.Thursday => 20/6
  | Day.Friday => 25/6
  | Day.Saturday => 30/6
  | Day.Sunday => 35/6

theorem cat_finishes_food_on_sunday :
  ∀ d : Day, cat_food_eaten d ≤ 9 ∧
  (d = Day.Sunday → cat_food_eaten d > 54/6) :=
by sorry

#check cat_finishes_food_on_sunday

end cat_finishes_food_on_sunday_l3884_388419


namespace perpendicular_vectors_difference_magnitude_l3884_388465

/-- Given two vectors a and b in ℝ², where a = (2,1) and b = (1-y, 2+y),
    and a is perpendicular to b, prove that |a - b| = 5√2. -/
theorem perpendicular_vectors_difference_magnitude :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ → ℝ × ℝ := λ y ↦ (1 - y, 2 + y)
  ∃ y : ℝ, (a.1 * (b y).1 + a.2 * (b y).2 = 0) →
    Real.sqrt ((a.1 - (b y).1)^2 + (a.2 - (b y).2)^2) = 5 * Real.sqrt 2 :=
by sorry

end perpendicular_vectors_difference_magnitude_l3884_388465


namespace altons_weekly_profit_l3884_388498

/-- Calculates the weekly profit for a business owner given daily earnings and weekly rent. -/
def weekly_profit (daily_earnings : ℕ) (weekly_rent : ℕ) : ℕ :=
  daily_earnings * 7 - weekly_rent

/-- Theorem stating that given specific daily earnings and weekly rent, the weekly profit is 36. -/
theorem altons_weekly_profit :
  weekly_profit 8 20 = 36 := by
  sorry

end altons_weekly_profit_l3884_388498


namespace identity_proof_l3884_388475

theorem identity_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hsum : a + b + c ≠ 0) (h : 1/a + 1/b + 1/c = 1/(a+b+c)) : 
  1/a^1999 + 1/b^1999 + 1/c^1999 = 1/(a^1999 + b^1999 + c^1999) := by
  sorry

end identity_proof_l3884_388475


namespace product_of_sum_and_sum_of_squares_l3884_388493

theorem product_of_sum_and_sum_of_squares (m n : ℝ) 
  (h1 : m + n = 3) 
  (h2 : m^2 + n^2 = 3) : 
  m * n = 3 := by sorry

end product_of_sum_and_sum_of_squares_l3884_388493


namespace books_remaining_after_sale_l3884_388427

-- Define the initial number of books
def initial_books : Nat := 136

-- Define the number of books sold
def books_sold : Nat := 109

-- Theorem to prove
theorem books_remaining_after_sale : 
  initial_books - books_sold = 27 := by sorry

end books_remaining_after_sale_l3884_388427


namespace parallel_to_x_axis_symmetric_in_third_quadrant_equal_distance_to_axes_l3884_388421

-- Define point P
def P (x : ℝ) : ℝ × ℝ := (2*x - 3, 3 - x)

-- Define point A
def A : ℝ × ℝ := (-3, 4)

-- Theorem 1: If AP is parallel to x-axis, then P(-5, 4)
theorem parallel_to_x_axis :
  (∃ x : ℝ, P x = (-5, 4)) ↔ (∃ x : ℝ, (P x).2 = A.2) :=
sorry

-- Theorem 2: If symmetric point is in third quadrant, then x < 3/2
theorem symmetric_in_third_quadrant :
  (∃ x : ℝ, (2*x - 3 < 0 ∧ x - 3 < 0)) ↔ (∃ x : ℝ, x < 3/2) :=
sorry

-- Theorem 3: If distances to axes are equal, then P(1,1) or P(-3,3)
theorem equal_distance_to_axes :
  (∃ x : ℝ, |2*x - 3| = |3 - x|) ↔ (P 2 = (1, 1) ∨ P 0 = (-3, 3)) :=
sorry

end parallel_to_x_axis_symmetric_in_third_quadrant_equal_distance_to_axes_l3884_388421


namespace sqrt_of_2_4_3_6_5_2_l3884_388478

theorem sqrt_of_2_4_3_6_5_2 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := by
  sorry

end sqrt_of_2_4_3_6_5_2_l3884_388478


namespace max_value_of_a_l3884_388464

theorem max_value_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = |x - 5/2| + |x - a|) →
  (∀ x, f x ≥ a) →
  ∃ a_max : ℝ, a_max = 5/4 ∧ ∀ a' : ℝ, (∀ x, |x - 5/2| + |x - a'| ≥ a') → a' ≤ a_max :=
by sorry

end max_value_of_a_l3884_388464


namespace squats_on_fourth_day_l3884_388469

/-- Calculates the number of squats on a given day, given the initial number of squats and daily increase. -/
def squats_on_day (initial_squats : ℕ) (daily_increase : ℕ) (day : ℕ) : ℕ :=
  initial_squats + (day - 1) * daily_increase

/-- Theorem stating that given an initial number of 30 squats and a daily increase of 5,
    the number of squats on the fourth day will be 45. -/
theorem squats_on_fourth_day :
  squats_on_day 30 5 4 = 45 := by
  sorry

end squats_on_fourth_day_l3884_388469


namespace score_calculation_l3884_388442

/-- Proves that given the average score and difference between subjects, we can determine the individual scores -/
theorem score_calculation (average : ℝ) (difference : ℝ) 
  (h_average : average = 96) 
  (h_difference : difference = 8) : 
  ∃ (chinese : ℝ) (math : ℝ), 
    chinese + math = 2 * average ∧ 
    math = chinese + difference ∧
    chinese = 92 ∧ 
    math = 100 := by
  sorry

end score_calculation_l3884_388442


namespace circle_radius_l3884_388436

/-- The radius of a circle defined by the equation x^2 + 2x + y^2 = 0 is 1 -/
theorem circle_radius (x y : ℝ) : x^2 + 2*x + y^2 = 0 → ∃ (c : ℝ × ℝ), (x - c.1)^2 + (y - c.2)^2 = 1 := by
  sorry

end circle_radius_l3884_388436


namespace train_bridge_problem_l3884_388407

/-- Represents the problem of determining the carriage position of a person walking through a train on a bridge. -/
theorem train_bridge_problem
  (bridge_length : ℝ)
  (train_speed : ℝ)
  (person_speed : ℝ)
  (carriage_length : ℝ)
  (h_bridge : bridge_length = 1400)
  (h_train : train_speed = 54 * (1000 / 3600))
  (h_person : person_speed = 3.6 * (1000 / 3600))
  (h_carriage : carriage_length = 23)
  : ∃ (n : ℕ), 5 ≤ n ∧ n ≤ 6 ∧
    (n : ℝ) * carriage_length ≥
      person_speed * (bridge_length / (train_speed + person_speed)) ∧
    ((n + 1) : ℝ) * carriage_length >
      person_speed * (bridge_length / (train_speed + person_speed)) :=
by sorry

end train_bridge_problem_l3884_388407


namespace largest_m_for_quadratic_function_l3884_388410

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem largest_m_for_quadratic_function (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b c (x - 4) = f a b c (2 - x)) →
  (∀ x, f a b c x ≥ x) →
  (∀ x ∈ Set.Ioo 0 2, f a b c x ≤ ((x + 1) / 2)^2) →
  (∃ x, ∀ y, f a b c y ≥ f a b c x) →
  (∃ x, f a b c x = 0) →
  (∃ m > 1, ∃ t, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) →
  (∀ m > 9, ¬∃ t, ∀ x ∈ Set.Icc 1 m, f a b c (x + t) ≤ x) :=
by sorry

end largest_m_for_quadratic_function_l3884_388410


namespace pyramid_hemisphere_theorem_l3884_388447

/-- A triangular pyramid with an equilateral triangular base -/
structure TriangularPyramid where
  /-- The height of the pyramid -/
  height : ℝ
  /-- The side length of the equilateral triangular base -/
  base_side : ℝ

/-- A hemisphere placed inside the pyramid -/
structure Hemisphere where
  /-- The radius of the hemisphere -/
  radius : ℝ

/-- Predicate to check if the hemisphere is properly placed in the pyramid -/
def is_properly_placed (p : TriangularPyramid) (h : Hemisphere) : Prop :=
  h.radius = 3 ∧ 
  p.height = 9 ∧
  -- The hemisphere is tangent to all three faces and rests on the base
  -- (This condition is assumed to be true when the predicate is true)
  True

/-- The main theorem -/
theorem pyramid_hemisphere_theorem (p : TriangularPyramid) (h : Hemisphere) :
  is_properly_placed p h → p.base_side = 6 * Real.sqrt 3 := by
  sorry

end pyramid_hemisphere_theorem_l3884_388447


namespace fourTangentCircles_l3884_388479

-- Define the circles C₁ and C₂
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the plane containing the circles
def Plane : Type := ℝ × ℝ

-- Define tangency between circles
def areTangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

-- Define the given conditions
def givenConditions (c1 c2 : Circle) : Prop :=
  c1.radius = 2 ∧ c2.radius = 2 ∧ areTangent c1 c2

-- Define a function to count tangent circles
def countTangentCircles (c1 c2 : Circle) : ℕ :=
  sorry

-- Theorem statement
theorem fourTangentCircles (c1 c2 : Circle) :
  givenConditions c1 c2 → countTangentCircles c1 c2 = 4 := by
  sorry

end fourTangentCircles_l3884_388479


namespace scientific_notation_correct_l3884_388409

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 3050000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation := {
  coefficient := 3.05,
  exponent := 6,
  is_valid := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end scientific_notation_correct_l3884_388409


namespace james_balloons_l3884_388443

-- Define the number of balloons Amy has
def amy_balloons : ℕ := 513

-- Define the difference in balloons between James and Amy
def difference : ℕ := 208

-- Theorem statement
theorem james_balloons : amy_balloons + difference = 721 := by
  sorry

end james_balloons_l3884_388443


namespace eugene_pencils_l3884_388441

/-- The number of pencils Eugene has after receiving more from Joyce -/
def total_pencils (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Eugene has 57 pencils in total -/
theorem eugene_pencils : total_pencils 51 6 = 57 := by
  sorry

end eugene_pencils_l3884_388441


namespace parabola_vertex_on_x_axis_l3884_388487

/-- If the vertex of the parabola y = x^2 + 2x + c is on the x-axis, then c = 1 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + c = 0 ∧ ∀ t : ℝ, t^2 + 2*t + c ≥ x^2 + 2*x + c) → c = 1 := by
  sorry

end parabola_vertex_on_x_axis_l3884_388487


namespace quadratic_roots_existence_l3884_388431

theorem quadratic_roots_existence (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    3 * x₁^2 - 2*(a+b+c)*x₁ + a*b + b*c + a*c = 0 ∧
    3 * x₂^2 - 2*(a+b+c)*x₂ + a*b + b*c + a*c = 0 ∧
    a < x₁ ∧ x₁ < b ∧ b < x₂ ∧ x₂ < c :=
by sorry

end quadratic_roots_existence_l3884_388431


namespace scientific_notation_of_0_00003_l3884_388459

theorem scientific_notation_of_0_00003 :
  ∃ (a : ℝ) (n : ℤ), 0.00003 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3 ∧ n = -5 :=
by sorry

end scientific_notation_of_0_00003_l3884_388459


namespace cat_litter_cost_210_days_l3884_388472

/-- Calculates the cost of cat litter for a given number of days -/
def catLitterCost (containerSize : ℕ) (containerPrice : ℕ) (litterBoxCapacity : ℕ) (changeDays : ℕ) (totalDays : ℕ) : ℕ :=
  let changes := totalDays / changeDays
  let totalLitter := changes * litterBoxCapacity
  let containers := (totalLitter + containerSize - 1) / containerSize  -- Ceiling division
  containers * containerPrice

/-- The cost of cat litter for 210 days is $210 -/
theorem cat_litter_cost_210_days :
  catLitterCost 45 21 15 7 210 = 210 := by sorry

end cat_litter_cost_210_days_l3884_388472


namespace spotted_cats_ratio_l3884_388434

/-- Proves that the ratio of spotted cats to total cats is 1:3 -/
theorem spotted_cats_ratio (total_cats : ℕ) (spotted_fluffy : ℕ) :
  total_cats = 120 →
  spotted_fluffy = 10 →
  (4 : ℚ) * spotted_fluffy = total_spotted →
  (total_spotted : ℚ) / total_cats = 1 / 3 :=
by sorry

end spotted_cats_ratio_l3884_388434


namespace floor_times_x_equals_88_l3884_388428

theorem floor_times_x_equals_88 (x : ℝ) (h1 : x > 0) (h2 : ⌊x⌋ * x = 88) : x = 88 / 9 := by
  sorry

end floor_times_x_equals_88_l3884_388428


namespace average_apples_per_day_l3884_388480

def boxes : ℕ := 12
def apples_per_box : ℕ := 25
def days : ℕ := 4

def total_apples : ℕ := boxes * apples_per_box

theorem average_apples_per_day : total_apples / days = 75 := by
  sorry

end average_apples_per_day_l3884_388480


namespace distance_to_origin_is_sqrt2_l3884_388413

-- Define the ellipse parameters
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = 1/2

-- Define the right focus
def right_focus (a c : ℝ) : Prop := c^2 = a^2 / 4

-- Define the quadratic equation and its roots
def quadratic_roots (a b c x₁ x₂ : ℝ) : Prop :=
  a * x₁^2 + 2 * b * x₁ + c = 0 ∧
  a * x₂^2 + 2 * b * x₂ + c = 0

-- Theorem statement
theorem distance_to_origin_is_sqrt2
  (a b c x₁ x₂ : ℝ)
  (h_ellipse : ellipse a b x₁ x₂)
  (h_eccentricity : eccentricity (Real.sqrt (1 - b^2 / a^2)))
  (h_focus : right_focus a c)
  (h_roots : quadratic_roots a (Real.sqrt (a^2 - c^2)) c x₁ x₂) :
  Real.sqrt (x₁^2 + x₂^2) = Real.sqrt 2 := by
  sorry

end distance_to_origin_is_sqrt2_l3884_388413


namespace simple_interest_sum_l3884_388422

/-- Given a sum of money with simple interest, prove that it equals 1700 --/
theorem simple_interest_sum (P r : ℝ) 
  (h1 : P * (1 + r) = 1717)
  (h2 : P * (1 + 2 * r) = 1734) :
  P = 1700 := by sorry

end simple_interest_sum_l3884_388422


namespace b₁_value_l3884_388402

/-- The polynomial f(x) with 4 distinct real roots -/
def f (x : ℝ) : ℝ := 8 + 32*x - 12*x^2 - 4*x^3 + x^4

/-- The set of roots of f(x) -/
def roots_f : Set ℝ := {x | f x = 0}

/-- The polynomial g(x) with roots being squares of roots of f(x) -/
def g (b₀ b₁ b₂ b₃ : ℝ) (x : ℝ) : ℝ := b₀ + b₁*x + b₂*x^2 + b₃*x^3 + x^4

/-- The set of roots of g(x) -/
def roots_g (b₀ b₁ b₂ b₃ : ℝ) : Set ℝ := {x | g b₀ b₁ b₂ b₃ x = 0}

theorem b₁_value (b₀ b₁ b₂ b₃ : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
    roots_f = {x₁, x₂, x₃, x₄} ∧ 
    roots_g b₀ b₁ b₂ b₃ = {x₁^2, x₂^2, x₃^2, x₄^2}) →
  b₁ = -1216 := by
sorry

end b₁_value_l3884_388402


namespace sequence_prime_divisor_l3884_388446

/-- Given a positive integer n > 1, prove that for all k ≥ 1, the k-th term of the sequence
    a_k = n^(n^(k-1)) - 1 has a prime divisor that does not divide any of the previous terms. -/
theorem sequence_prime_divisor (n : ℕ) (hn : n > 1) :
  ∀ k : ℕ, k ≥ 1 →
    ∃ p : ℕ, Nat.Prime p ∧ p ∣ (n^(n^(k-1)) - 1) ∧
      ∀ i : ℕ, 1 ≤ i ∧ i < k → ¬(p ∣ (n^(n^(i-1)) - 1)) :=
by sorry

end sequence_prime_divisor_l3884_388446


namespace sin_x_squared_not_periodic_l3884_388467

theorem sin_x_squared_not_periodic : ¬∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), Real.sin ((x + p)^2) = Real.sin (x^2) := by
  sorry

end sin_x_squared_not_periodic_l3884_388467


namespace second_term_of_geometric_sequence_l3884_388461

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem second_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 1/5)
  (h_a3 : a 3 = 5) :
  a 2 = 1 ∨ a 2 = -1 :=
sorry

end second_term_of_geometric_sequence_l3884_388461


namespace min_draws_for_pair_of_each_color_l3884_388454

/-- Represents the number of items of a given color -/
structure ColorCount where
  count : Nat

/-- Represents the box containing hats and gloves -/
structure Box where
  red : ColorCount
  green : ColorCount
  orange : ColorCount

/-- Calculates the minimum number of draws required for a given color -/
def minDrawsForColor (c : ColorCount) : Nat :=
  c.count + 1

/-- Calculates the total minimum draws required for all colors -/
def totalMinDraws (box : Box) : Nat :=
  minDrawsForColor box.red + minDrawsForColor box.green + minDrawsForColor box.orange

/-- The main theorem to be proved -/
theorem min_draws_for_pair_of_each_color (box : Box) 
  (h_red : box.red.count = 41)
  (h_green : box.green.count = 23)
  (h_orange : box.orange.count = 11) :
  totalMinDraws box = 78 := by
  sorry

#eval totalMinDraws { red := { count := 41 }, green := { count := 23 }, orange := { count := 11 } }

end min_draws_for_pair_of_each_color_l3884_388454


namespace quadratic_equation_properties_l3884_388451

theorem quadratic_equation_properties (a b : ℝ) (ha : a > 0) (hab : a^2 = 4*b) :
  (a^2 - b^2 ≤ 4) ∧
  (a^2 + 1/b ≥ 4) ∧
  (∀ c : ℝ, ∀ x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x + b < c ↔ x₁ < x ∧ x < x₂) → |x₁ - x₂| = 4 → c = 4) :=
by sorry

end quadratic_equation_properties_l3884_388451


namespace simplify_trig_expression_l3884_388452

theorem simplify_trig_expression (x : ℝ) : 
  (Real.sqrt 3 / 2) * Real.sin x - (1 / 2) * Real.cos x = Real.sin (x - π / 6) := by
  sorry

end simplify_trig_expression_l3884_388452


namespace factorization_problem_1_factorization_problem_2_l3884_388426

-- Problem 1
theorem factorization_problem_1 (a b : ℝ) :
  12 * a^3 * b - 12 * a^2 * b + 3 * a * b = 3 * a * b * (2*a - 1)^2 := by
  sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  9 - x^2 + 2*x*y - y^2 = (3 + x - y) * (3 - x + y) := by
  sorry

end factorization_problem_1_factorization_problem_2_l3884_388426


namespace pretzels_theorem_l3884_388450

def pretzels_problem (initial_pretzels john_pretzels marcus_pretzels : ℕ) : Prop :=
  let alan_pretzels := initial_pretzels - john_pretzels - marcus_pretzels
  john_pretzels - alan_pretzels = 1

theorem pretzels_theorem :
  ∀ (initial_pretzels john_pretzels marcus_pretzels : ℕ),
    initial_pretzels = 95 →
    john_pretzels = 28 →
    marcus_pretzels = 40 →
    marcus_pretzels = john_pretzels + 12 →
    pretzels_problem initial_pretzels john_pretzels marcus_pretzels :=
by
  sorry

#check pretzels_theorem

end pretzels_theorem_l3884_388450


namespace triangle_side_length_l3884_388433

/-- Given a triangle ABC with the condition that cos(∠A - ∠B) + sin(∠A + ∠B) = 2 and AB = 4,
    prove that BC = 2√2 -/
theorem triangle_side_length (A B C : ℝ) (h1 : 0 < A ∧ A < π)
    (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π)
    (h5 : Real.cos (A - B) + Real.sin (A + B) = 2) (h6 : ∃ AB : ℝ, AB = 4) :
    ∃ BC : ℝ, BC = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_length_l3884_388433


namespace two_thousandth_point_l3884_388435

/-- Represents a point in the first quadrant with integer coordinates -/
structure Point where
  x : Nat
  y : Nat

/-- The spiral numbering function that assigns a natural number to each point -/
def spiralNumber : Point → Nat := sorry

/-- The inverse function that finds the point corresponding to a given number -/
def spiralPoint : Nat → Point := sorry

/-- Theorem stating that the 2000th point in the spiral has coordinates (44, 24) -/
theorem two_thousandth_point : spiralPoint 2000 = Point.mk 44 24 := by sorry

end two_thousandth_point_l3884_388435


namespace systematic_sampling_l3884_388491

/-- Systematic sampling problem -/
theorem systematic_sampling
  (total_population : ℕ)
  (sample_size : ℕ)
  (first_drawn : ℕ)
  (interval_start : ℕ)
  (interval_end : ℕ)
  (h1 : total_population = 960)
  (h2 : sample_size = 32)
  (h3 : first_drawn = 29)
  (h4 : interval_start = 200)
  (h5 : interval_end = 480) :
  (Finset.filter (fun n => interval_start ≤ (first_drawn + (total_population / sample_size) * (n - 1)) ∧
                           (first_drawn + (total_population / sample_size) * (n - 1)) ≤ interval_end)
                 (Finset.range sample_size)).card = 10 := by
  sorry


end systematic_sampling_l3884_388491


namespace quadratic_properties_l3884_388430

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  f : ℝ → ℝ
  quad : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  axis_sym : ∀ x, f (1 + x) = f (1 - x)
  vertex : f 1 = -4
  table_values : f (-2) = 5 ∧ f (-1) = 0 ∧ f 0 = -3 ∧ f 1 = -4 ∧ f 2 = -3 ∧ f 3 = 0

theorem quadratic_properties (f : QuadraticFunction) :
  (∃ a > 0, ∀ x, f.f x = a * x^2 + f.f 1 - a) ∧
  f.f 4 = 5 ∧
  f.f (-3) > f.f 2 ∧
  {x : ℝ | f.f x < 0} = {x : ℝ | -1 < x ∧ x < 3} ∧
  {x : ℝ | f.f x = 5} = {-2, 4} := by
  sorry


end quadratic_properties_l3884_388430


namespace school_population_l3884_388445

/-- Represents the total number of students in the school -/
def total_students : ℕ := 50

/-- Represents the number of students of 8 years of age -/
def students_8_years : ℕ := 24

/-- Represents the fraction of students below 8 years of age -/
def fraction_below_8 : ℚ := 1/5

/-- Represents the ratio of students above 8 years to students of 8 years -/
def ratio_above_to_8 : ℚ := 2/3

theorem school_population :
  (students_8_years : ℚ) + 
  (ratio_above_to_8 * students_8_years) + 
  (fraction_below_8 * total_students) = total_students := by sorry

end school_population_l3884_388445


namespace cubic_root_function_l3884_388481

/-- Given a function y = kx^(1/3) where y = 4√3 when x = 64, 
    prove that y = 2√3 when x = 8 -/
theorem cubic_root_function (k : ℝ) :
  (∀ x, x > 0 → k * x^(1/3) = 4 * Real.sqrt 3 → x = 64) →
  k * 8^(1/3) = 2 * Real.sqrt 3 := by
  sorry

end cubic_root_function_l3884_388481


namespace cone_cylinder_volume_ratio_l3884_388401

/-- Given a cylinder with height 15 cm and radius 5 cm, and a cone with the same radius
    and height one-third of the cylinder's, prove that the ratio of their volumes is 1/9. -/
theorem cone_cylinder_volume_ratio :
  let cylinder_height : ℝ := 15
  let cylinder_radius : ℝ := 5
  let cone_radius : ℝ := cylinder_radius
  let cone_height : ℝ := cylinder_height / 3
  let cylinder_volume := π * cylinder_radius^2 * cylinder_height
  let cone_volume := (1/3) * π * cone_radius^2 * cone_height
  cone_volume / cylinder_volume = 1/9 := by
sorry


end cone_cylinder_volume_ratio_l3884_388401


namespace probability_no_repetition_l3884_388470

def three_digit_numbers : ℕ := 3^3

def numbers_without_repetition : ℕ := 6

theorem probability_no_repetition :
  (numbers_without_repetition : ℚ) / three_digit_numbers = 2 / 9 := by
  sorry

end probability_no_repetition_l3884_388470


namespace tic_tac_toe_tie_probability_l3884_388462

theorem tic_tac_toe_tie_probability 
  (amy_win_prob : ℚ) 
  (lily_win_prob : ℚ) 
  (h1 : amy_win_prob = 4/9) 
  (h2 : lily_win_prob = 1/3) : 
  1 - (amy_win_prob + lily_win_prob) = 2/9 := by
  sorry

end tic_tac_toe_tie_probability_l3884_388462


namespace quadratic_inequality_l3884_388488

theorem quadratic_inequality (x : ℝ) : x^2 + x - 12 > 0 ↔ x > 3 ∨ x < -4 := by
  sorry

end quadratic_inequality_l3884_388488


namespace vector_computation_l3884_388456

theorem vector_computation :
  let v1 : Fin 3 → ℝ := ![3, -2, 5]
  let v2 : Fin 3 → ℝ := ![2, -3, 4]
  4 • v1 - 3 • v2 = ![6, 1, 8] :=
by
  sorry

end vector_computation_l3884_388456


namespace quadratic_coefficient_l3884_388477

theorem quadratic_coefficient (b : ℝ) :
  (b < 0) →
  (∃ m : ℝ, ∀ x : ℝ, x^2 + b*x + 1/4 = (x + m)^2 + 1/16) →
  b = -Real.sqrt 3 / 2 :=
by sorry

end quadratic_coefficient_l3884_388477


namespace b_formula_T_formula_l3884_388489

/-- An arithmetic sequence with first term 1 and common difference 1 -/
def arithmetic_sequence (n : ℕ) : ℕ := n

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sequence b_n defined as 1/S_n -/
def b (n : ℕ) : ℚ := 1 / (S n)

/-- Sum of the first n terms of the sequence b_n -/
def T (n : ℕ) : ℚ := sorry

theorem b_formula (n : ℕ) : b n = 2 / (n * (n + 1)) :=
  sorry

theorem T_formula (n : ℕ) : T n = 2 * n / (n + 1) :=
  sorry

end b_formula_T_formula_l3884_388489


namespace waldo_total_time_l3884_388416

/-- The number of "Where's Waldo?" books -/
def num_books : ℕ := 15

/-- The number of puzzles per book -/
def puzzles_per_book : ℕ := 30

/-- The average time (in minutes) to find Waldo in a puzzle -/
def time_per_puzzle : ℕ := 3

/-- The total time (in minutes) to find Waldo in all puzzles across all books -/
def total_time : ℕ := num_books * puzzles_per_book * time_per_puzzle

theorem waldo_total_time : total_time = 1350 := by
  sorry

end waldo_total_time_l3884_388416


namespace same_terminal_side_negative_420_and_660_l3884_388483

-- Define a function to represent angles with the same terminal side
def same_terminal_side (θ : ℝ) (φ : ℝ) : Prop :=
  ∃ n : ℤ, φ = θ + n * 360

-- Theorem statement
theorem same_terminal_side_negative_420_and_660 :
  same_terminal_side (-420) 660 := by
  sorry

end same_terminal_side_negative_420_and_660_l3884_388483


namespace intersection_triangle_area_l3884_388476

/-- Given a regular tetrahedron with side length 2, cut by a plane parallel to one face
    at height 1 from the base, the area of the intersection triangle is (2√3 - 3) / 4 -/
theorem intersection_triangle_area (side_length : ℝ) (cut_height : ℝ) : 
  side_length = 2 → cut_height = 1 → 
  ∃ (area : ℝ), area = (2 * Real.sqrt 3 - 3) / 4 := by
  sorry

end intersection_triangle_area_l3884_388476


namespace circumcircle_equation_incircle_equation_l3884_388473

-- Define the Triangle type
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the Circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def Triangle1 : Triangle := { A := (5, 1), B := (7, -3), C := (2, -8) }
def Triangle2 : Triangle := { A := (0, 0), B := (5, 0), C := (0, 12) }

def CircumcircleEquation (t : Triangle) (c : Circle) : Prop :=
  ∀ (x y : ℝ), (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔
    (x = t.A.1 ∧ y = t.A.2) ∨ (x = t.B.1 ∧ y = t.B.2) ∨ (x = t.C.1 ∧ y = t.C.2)

def IncircleEquation (t : Triangle) (c : Circle) : Prop :=
  ∀ (x y : ℝ), (x - c.center.1)^2 + (y - c.center.2)^2 ≤ c.radius^2 ↔
    (y ≥ 0 ∧ y ≤ 12 ∧ x ≥ 0 ∧ 5*y + 12*x ≤ 60)

theorem circumcircle_equation (t : Triangle) (h : t = Triangle1) :
  CircumcircleEquation t { center := (2, -3), radius := 5 } := by sorry

theorem incircle_equation (t : Triangle) (h : t = Triangle2) :
  IncircleEquation t { center := (2, 2), radius := 2 } := by sorry

end circumcircle_equation_incircle_equation_l3884_388473


namespace triangle_theorem_l3884_388449

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0)
  (h2 : t.a = 2)
  (h3 : 1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3) : 
  t.A = π/3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry


end triangle_theorem_l3884_388449


namespace function_inequality_l3884_388486

-- Define the functions f and g
def f (x b : ℝ) : ℝ := |x + b^2| - |-x + 1|
def g (x a b c : ℝ) : ℝ := |x + a^2 + c^2| + |x - 2*b^2|

-- State the theorem
theorem function_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a*b + b*c + a*c = 1) :
  (∀ x : ℝ, f x 1 ≥ 1 ↔ x ∈ Set.Ici (1/2)) ∧
  (∀ x : ℝ, f x b ≤ g x a b c) := by
  sorry

end function_inequality_l3884_388486


namespace negation_equivalence_l3884_388471

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ ≥ 0 ∧ 2 * x₀ = 3) ↔ (∀ x : ℝ, x ≥ 0 → 2 * x ≠ 3) := by
  sorry

end negation_equivalence_l3884_388471


namespace min_value_theorem_l3884_388408

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 3) :
  (2 / x + 1 / y) ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 3 ∧ 2 / x₀ + 1 / y₀ = 3 :=
by sorry

end min_value_theorem_l3884_388408


namespace smallest_undefined_value_l3884_388429

theorem smallest_undefined_value (x : ℝ) :
  let f := fun x => (x - 3) / (6 * x^2 - 47 * x + 7)
  let smallest_x := (47 - Real.sqrt 2041) / 12
  (∀ y < smallest_x, f y ≠ 0⁻¹) ∧
  (f smallest_x = 0⁻¹) :=
by sorry

end smallest_undefined_value_l3884_388429


namespace intersection_line_polar_equation_l3884_388420

/-- Given two circles in polar coordinates, find the polar equation of the line
    passing through their intersection points. -/
theorem intersection_line_polar_equation
  (O₁ : ℝ → ℝ → Prop) -- Circle O₁ in polar coordinates
  (O₂ : ℝ → ℝ → Prop) -- Circle O₂ in polar coordinates
  (h₁ : ∀ ρ θ, O₁ ρ θ ↔ ρ = 2)
  (h₂ : ∀ ρ θ, O₂ ρ θ ↔ ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos (θ - π/4) = 2) :
  ∀ ρ θ, (∃ θ₁ θ₂, O₁ ρ θ₁ ∧ O₂ ρ θ₂) →
    ρ * Real.sin (θ + π/4) = Real.sqrt 2 / 2 :=
by sorry

end intersection_line_polar_equation_l3884_388420


namespace no_preimage_set_l3884_388460

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Theorem statement
theorem no_preimage_set (k : ℝ) :
  (∀ x : ℝ, f x ≠ k) ↔ k > 1 :=
sorry

end no_preimage_set_l3884_388460


namespace quadratic_discriminant_positive_l3884_388404

theorem quadratic_discriminant_positive 
  (a b c : ℝ) 
  (h : (a + b + c) * c < 0) : 
  b^2 - 4*a*c > 0 := by
sorry

end quadratic_discriminant_positive_l3884_388404


namespace intersection_of_A_and_B_l3884_388448

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by
  sorry

end intersection_of_A_and_B_l3884_388448


namespace product_evaluation_l3884_388468

theorem product_evaluation : (6 - 5) * (6 - 4) * (6 - 3) * (6 - 2) * (6 - 1) * 6 = 720 := by
  sorry

end product_evaluation_l3884_388468


namespace book_reading_ratio_l3884_388499

theorem book_reading_ratio (total_pages : ℕ) (total_days : ℕ) (speed1 speed2 : ℕ) 
  (h1 : total_pages = 500)
  (h2 : total_days = 75)
  (h3 : speed1 = 10)
  (h4 : speed2 = 5)
  (h5 : ∃ x : ℕ, speed1 * x + speed2 * (total_days - x) = total_pages) :
  ∃ x : ℕ, (speed1 * x : ℚ) / total_pages = 1 / 2 := by
  sorry

end book_reading_ratio_l3884_388499


namespace max_tennis_court_area_l3884_388444

/-- Represents the dimensions of a rectangular tennis court --/
structure CourtDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular tennis court --/
def area (d : CourtDimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular tennis court --/
def perimeter (d : CourtDimensions) : ℝ := 2 * (d.length + d.width)

/-- Checks if the court dimensions meet the minimum requirements --/
def meetsMinimumRequirements (d : CourtDimensions) : Prop :=
  d.length ≥ 85 ∧ d.width ≥ 45

/-- Theorem stating the maximum area of the tennis court --/
theorem max_tennis_court_area :
  ∃ (d : CourtDimensions),
    perimeter d = 320 ∧
    meetsMinimumRequirements d ∧
    area d = 6375 ∧
    ∀ (d' : CourtDimensions),
      perimeter d' = 320 ∧ meetsMinimumRequirements d' → area d' ≤ area d :=
by sorry

end max_tennis_court_area_l3884_388444


namespace chord_equation_parabola_l3884_388458

/-- Given a parabola y² = 4x and a chord AB with midpoint P(1,1), 
    the equation of the line containing chord AB is 2x - y - 1 = 0 -/
theorem chord_equation_parabola (A B : ℝ × ℝ) :
  let parabola := fun (p : ℝ × ℝ) ↦ p.2^2 = 4 * p.1
  let midpoint := (1, 1)
  let on_parabola := fun (p : ℝ × ℝ) ↦ parabola p
  let is_midpoint := fun (m p1 p2 : ℝ × ℝ) ↦ 
    m.1 = (p1.1 + p2.1) / 2 ∧ m.2 = (p1.2 + p2.2) / 2
  on_parabola A ∧ on_parabola B ∧ is_midpoint midpoint A B →
  ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 ∧
                  a * B.1 + b * B.2 + c = 0 ∧
                  (a, b, c) = (2, -1, -1) :=
by sorry

end chord_equation_parabola_l3884_388458
