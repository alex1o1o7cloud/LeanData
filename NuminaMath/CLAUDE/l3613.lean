import Mathlib

namespace triangle_coloring_theorem_l3613_361307

-- Define the set of colors
inductive Color
| Blue
| Red
| Yellow

-- Define a point with a color
structure Point where
  color : Color

-- Define a triangle
structure Triangle where
  vertex1 : Point
  vertex2 : Point
  vertex3 : Point

-- Define the main theorem
theorem triangle_coloring_theorem 
  (K P S : Point)
  (A B C D E F : Point)
  (h1 : K.color = Color.Blue)
  (h2 : P.color = Color.Red)
  (h3 : S.color = Color.Yellow)
  (h4 : (A.color = K.color ∨ A.color = S.color) ∧
        (B.color = K.color ∨ B.color = S.color) ∧
        (C.color = K.color ∨ C.color = P.color) ∧
        (D.color = P.color ∨ D.color = S.color) ∧
        (E.color = P.color ∨ E.color = S.color) ∧
        (F.color = K.color ∨ F.color = P.color)) :
  ∃ (t : Triangle), t.vertex1.color ≠ t.vertex2.color ∧ 
                    t.vertex2.color ≠ t.vertex3.color ∧ 
                    t.vertex3.color ≠ t.vertex1.color :=
by sorry

end triangle_coloring_theorem_l3613_361307


namespace board_numbers_l3613_361393

theorem board_numbers (a b : ℕ+) : 
  (a.val - b.val)^2 = a.val^2 - b.val^2 - 4038 →
  ((a.val = 2020 ∧ b.val = 1) ∨
   (a.val = 2020 ∧ b.val = 2019) ∨
   (a.val = 676 ∧ b.val = 3) ∨
   (a.val = 676 ∧ b.val = 673)) :=
by sorry

end board_numbers_l3613_361393


namespace intersection_values_l3613_361399

/-- Definition of the circle M -/
def circle_M (x y : ℝ) : Prop := x^2 - 2*x + y^2 + 4*y - 10 = 0

/-- Definition of the intersecting line -/
def intersecting_line (x y : ℝ) (C : ℝ) : Prop := x + 3*y + C = 0

/-- Theorem stating the possible values of C -/
theorem intersection_values (C : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    circle_M A.1 A.2 ∧ 
    circle_M B.1 B.2 ∧
    intersecting_line A.1 A.2 C ∧
    intersecting_line B.1 B.2 C ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 20) →
  C = 15 ∨ C = -5 := by
sorry

end intersection_values_l3613_361399


namespace wally_initial_tickets_l3613_361395

/-- Proves that Wally had 400 tickets initially given the conditions of the problem -/
theorem wally_initial_tickets : 
  ∀ (total : ℕ) (jensen finley : ℕ),
  (3 : ℚ) / 4 * total = jensen + finley →
  jensen * 11 = finley * 4 →
  finley = 220 →
  total = 400 :=
by sorry

end wally_initial_tickets_l3613_361395


namespace quadratic_inequality_l3613_361317

/-- The quadratic function f(x) = -2(x+1)^2 + k -/
def f (k : ℝ) (x : ℝ) : ℝ := -2 * (x + 1)^2 + k

/-- Theorem stating the inequality between f(2), f(-3), and f(-0.5) -/
theorem quadratic_inequality (k : ℝ) : f k 2 < f k (-3) ∧ f k (-3) < f k (-0.5) := by
  sorry

end quadratic_inequality_l3613_361317


namespace slope_product_of_triple_angle_and_slope_l3613_361396

/-- Given two non-horizontal lines with slopes m and n, where one line forms
    three times as large an angle with the horizontal as the other and has
    three times the slope, prove that mn = 9/4 -/
theorem slope_product_of_triple_angle_and_slope
  (m n : ℝ) -- slopes of the lines
  (h₁ : m ≠ 0) -- L₁ is not horizontal
  (h₂ : n ≠ 0) -- L₂ is not horizontal
  (h₃ : ∃ θ₁ θ₂ : ℝ, θ₁ = 3 * θ₂ ∧ m = Real.tan θ₁ ∧ n = Real.tan θ₂) -- angle relation
  (h₄ : m = 3 * n) -- slope relation
  : m * n = 9 / 4 := by
  sorry

end slope_product_of_triple_angle_and_slope_l3613_361396


namespace school_chairs_problem_l3613_361366

theorem school_chairs_problem (initial_chairs : ℕ) : 
  initial_chairs < 35 →
  ∃ (k : ℕ), initial_chairs + 27 = 35 * k →
  initial_chairs = 8 := by
sorry

end school_chairs_problem_l3613_361366


namespace initial_interest_rate_l3613_361355

/-- Given interest conditions, prove the initial interest rate -/
theorem initial_interest_rate
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Initial interest rate in percentage
  (h1 : P * r / 100 = 202.50)  -- Interest at initial rate
  (h2 : P * (r + 5) / 100 = 225)  -- Interest at increased rate
  : r = 45 := by
  sorry

end initial_interest_rate_l3613_361355


namespace birthday_800th_day_l3613_361387

/-- Given a person born on a Tuesday, their 800th day of life will fall on a Thursday. -/
theorem birthday_800th_day (birth_day : Nat) (days_passed : Nat) : 
  birth_day = 2 → days_passed = 800 → (birth_day + days_passed) % 7 = 4 := by
  sorry

end birthday_800th_day_l3613_361387


namespace comparison_of_expressions_l3613_361367

theorem comparison_of_expressions (x : ℝ) (h : x ≠ 1) :
  (x > 1 → 1 + x > 1 / (1 - x)) ∧ (x < 1 → 1 + x < 1 / (1 - x)) := by
  sorry

end comparison_of_expressions_l3613_361367


namespace circle_intersection_theorem_l3613_361352

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the number of intersection points between three circles
def intersectionPoints (c1 c2 c3 : Circle) : ℕ := sorry

theorem circle_intersection_theorem :
  -- There exist three circles that intersect at exactly one point
  (∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 1) ∧
  -- There exist three circles that intersect at exactly two points
  (∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 2) ∧
  -- There do not exist three circles that intersect at exactly three points
  (¬ ∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 3) ∧
  -- There do not exist three circles that intersect at exactly four points
  (¬ ∃ c1 c2 c3 : Circle, intersectionPoints c1 c2 c3 = 4) :=
by
  sorry

end circle_intersection_theorem_l3613_361352


namespace y_percentage_more_than_z_l3613_361361

/-- Given that x gets 25% more than y, the total amount is 370, and z's share is 100,
    prove that y gets 20% more than z. -/
theorem y_percentage_more_than_z (x y z : ℝ) : 
  x = 1.25 * y →  -- x gets 25% more than y
  x + y + z = 370 →  -- total amount is 370
  z = 100 →  -- z's share is 100
  y = 1.2 * z  -- y gets 20% more than z
  := by sorry

end y_percentage_more_than_z_l3613_361361


namespace y_value_l3613_361302

theorem y_value (x y : ℝ) (h1 : x^(2*y) = 16) (h2 : x = 8) : y = 2/3 := by
  sorry

end y_value_l3613_361302


namespace circle_equation_proof_l3613_361332

/-- Given a point M on the line 2x + y - 1 = 0 and points (3,0) and (0,1) on a circle centered at M,
    prove that the equation of this circle is (x-1)² + (y+1)² = 5 -/
theorem circle_equation_proof (M : ℝ × ℝ) :
  (2 * M.1 + M.2 - 1 = 0) →
  ((3 : ℝ) - M.1)^2 + (0 - M.2)^2 = ((0 : ℝ) - M.1)^2 + (1 - M.2)^2 →
  ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 5 ↔ (x - M.1)^2 + (y - M.2)^2 = ((3 : ℝ) - M.1)^2 + (0 - M.2)^2 :=
by sorry

end circle_equation_proof_l3613_361332


namespace product_of_five_consecutive_integers_not_square_l3613_361331

theorem product_of_five_consecutive_integers_not_square (n : ℕ+) :
  ¬∃ k : ℕ, (n : ℕ) * (n + 1) * (n + 2) * (n + 3) * (n + 4) = k^2 :=
sorry

end product_of_five_consecutive_integers_not_square_l3613_361331


namespace largest_sum_of_digits_24hour_pm_l3613_361391

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours ≥ 0 ∧ hours ≤ 23
  minute_valid : minutes ≥ 0 ∧ minutes ≤ 59

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits in a Time24 -/
def sumOfDigitsTime24 (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- Checks if a Time24 is between 12:00 and 23:59 -/
def isBetween12And2359 (t : Time24) : Prop :=
  t.hours ≥ 12 ∧ t.hours ≤ 23

theorem largest_sum_of_digits_24hour_pm :
  ∃ (t : Time24), isBetween12And2359 t ∧
    ∀ (t' : Time24), isBetween12And2359 t' →
      sumOfDigitsTime24 t' ≤ sumOfDigitsTime24 t ∧
      sumOfDigitsTime24 t = 24 :=
sorry

end largest_sum_of_digits_24hour_pm_l3613_361391


namespace mild_curries_count_mild_curries_proof_l3613_361370

/-- The number of peppers needed for different curry types -/
def peppers_per_curry : List Nat := [3, 2, 1]

/-- The number of curries of each type previously bought -/
def previous_curries : List Nat := [30, 30, 10]

/-- The number of spicy curries now bought -/
def current_spicy_curries : Nat := 15

/-- The reduction in total peppers bought -/
def pepper_reduction : Nat := 40

/-- Calculate the total number of peppers previously bought -/
def previous_total_peppers : Nat :=
  List.sum (List.zipWith (· * ·) peppers_per_curry previous_curries)

/-- Calculate the current total number of peppers bought -/
def current_total_peppers : Nat := previous_total_peppers - pepper_reduction

/-- Calculate the number of peppers used for current spicy curries -/
def current_spicy_peppers : Nat := peppers_per_curry[1] * current_spicy_curries

theorem mild_curries_count : Nat :=
  current_total_peppers - current_spicy_peppers

theorem mild_curries_proof : mild_curries_count = 90 := by
  sorry

end mild_curries_count_mild_curries_proof_l3613_361370


namespace line_through_point_l3613_361315

/-- Given a line with equation 3kx - 2 = 4y passing through the point (-1/2, -5),
    prove that k = 12. -/
theorem line_through_point (k : ℝ) : 
  (3 * k * (-1/2) - 2 = 4 * (-5)) → k = 12 := by
  sorry

end line_through_point_l3613_361315


namespace f_derivative_l3613_361390

noncomputable def f (x : ℝ) : ℝ := (1 - x) / ((1 + x^2) * Real.cos x)

theorem f_derivative :
  deriv f = λ x => ((x^2 - 2*x - 1) * Real.cos x + (1 - x) * (1 + x^2) * Real.sin x) / ((1 + x^2)^2 * (Real.cos x)^2) :=
by sorry

end f_derivative_l3613_361390


namespace tuesday_pages_l3613_361343

/-- Represents the number of pages read on each day of the week --/
structure PagesRead where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Represents the reading plan for the week --/
def ReadingPlan (total_pages : ℕ) (pages : PagesRead) : Prop :=
  pages.monday = 23 ∧
  pages.wednesday = 61 ∧
  pages.thursday = 12 ∧
  pages.friday = 2 * pages.thursday ∧
  total_pages = pages.monday + pages.tuesday + pages.wednesday + pages.thursday + pages.friday

theorem tuesday_pages (total_pages : ℕ) (pages : PagesRead) 
  (h : ReadingPlan total_pages pages) (h_total : total_pages = 158) : 
  pages.tuesday = 38 := by
  sorry

#check tuesday_pages

end tuesday_pages_l3613_361343


namespace no_defective_products_exactly_two_defective_products_at_least_two_defective_products_l3613_361398

-- Define the total number of items
def total_items : ℕ := 100

-- Define the number of defective items
def defective_items : ℕ := 3

-- Define the number of items to be selected
def selected_items : ℕ := 5

-- Theorem for scenario (I): No defective product
theorem no_defective_products : 
  Nat.choose (total_items - defective_items) selected_items = 64446024 := by sorry

-- Theorem for scenario (II): Exactly two defective products
theorem exactly_two_defective_products :
  Nat.choose defective_items 2 * Nat.choose (total_items - defective_items) (selected_items - 2) = 442320 := by sorry

-- Theorem for scenario (III): At least two defective products
theorem at_least_two_defective_products :
  Nat.choose defective_items 2 * Nat.choose (total_items - defective_items) (selected_items - 2) +
  Nat.choose defective_items 3 * Nat.choose (total_items - defective_items) (selected_items - 3) = 446886 := by sorry

end no_defective_products_exactly_two_defective_products_at_least_two_defective_products_l3613_361398


namespace polynomial_solution_l3613_361386

theorem polynomial_solution (a : ℝ) (ha : a ≠ -1) 
  (h : a^5 + 5*a^4 + 10*a^3 + 3*a^2 - 9*a - 6 = 0) : 
  (a + 1)^3 = 7 := by
sorry

end polynomial_solution_l3613_361386


namespace product_of_real_parts_quadratic_complex_l3613_361376

theorem product_of_real_parts_quadratic_complex (x : ℂ) :
  x^2 + 3*x = -2 + 2*I →
  ∃ (s₁ s₂ : ℂ), (s₁^2 + 3*s₁ = -2 + 2*I) ∧ 
                 (s₂^2 + 3*s₂ = -2 + 2*I) ∧
                 (s₁.re * s₂.re = (5 - 2*Real.sqrt 5) / 4) :=
by sorry

end product_of_real_parts_quadratic_complex_l3613_361376


namespace chicken_rabbit_problem_l3613_361324

theorem chicken_rabbit_problem (c r : ℕ) : 
  c = r - 20 → 
  4 * r = 3 * (2 * c) + 10 → 
  c = 35 :=
by sorry

end chicken_rabbit_problem_l3613_361324


namespace ellipse_properties_l3613_361356

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  eccentricity : ℝ
  eccentricity_eq : eccentricity = Real.sqrt 6 / 3
  equation : ℝ → ℝ → Prop
  equation_def : equation = λ x y => x^2 / a^2 + y^2 / b^2 = 1
  focal_line_length : ℝ
  focal_line_length_eq : focal_line_length = 2 * Real.sqrt 3 / 3

/-- The main theorem about the ellipse -/
theorem ellipse_properties (e : Ellipse) :
  e.equation = λ x y => x^2 / 3 + y^2 = 1 ∧
  ∃ k : ℝ, k = 7 / 6 ∧
    ∀ C D : ℝ × ℝ,
      (e.equation C.1 C.2 ∧ e.equation D.1 D.2) →
      (C.2 = k * C.1 + 2 ∧ D.2 = k * D.1 + 2) →
      (C.1 - (-1))^2 + (C.2 - 0)^2 = (D.1 - (-1))^2 + (D.2 - 0)^2 :=
by sorry

end ellipse_properties_l3613_361356


namespace geometric_progression_cubic_roots_l3613_361328

theorem geometric_progression_cubic_roots (x y z r p q : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x ≠ y → y ≠ z → x ≠ z →
  y^2 = r * x^2 →
  z^2 = r * y^2 →
  x^3 - p*x^2 + q*x - r = 0 →
  y^3 - p*y^2 + q*y - r = 0 →
  z^3 - p*z^2 + q*z - r = 0 →
  r^2 = 1 := by
sorry

end geometric_progression_cubic_roots_l3613_361328


namespace girls_attending_event_l3613_361318

theorem girls_attending_event (total_students : ℕ) (total_attending : ℕ) 
  (girls : ℕ) (boys : ℕ) : 
  total_students = 1800 →
  total_attending = 1110 →
  girls + boys = total_students →
  (3 * girls) / 4 + (2 * boys) / 3 = total_attending →
  (3 * girls) / 4 = 690 :=
by sorry

end girls_attending_event_l3613_361318


namespace intersection_A_B_l3613_361365

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end intersection_A_B_l3613_361365


namespace number_of_spinsters_l3613_361397

-- Define the number of spinsters and cats
def spinsters : ℕ := sorry
def cats : ℕ := sorry

-- State the theorem
theorem number_of_spinsters :
  -- Condition 1: The ratio of spinsters to cats is 2:7
  (spinsters : ℚ) / cats = 2 / 7 →
  -- Condition 2: There are 55 more cats than spinsters
  cats = spinsters + 55 →
  -- Conclusion: The number of spinsters is 22
  spinsters = 22 := by
  sorry

end number_of_spinsters_l3613_361397


namespace system_solution_l3613_361346

theorem system_solution (x y : ℝ) (h1 : 2*x + y = 7) (h2 : x + 2*y = 10) : 
  (x + y) / 3 = 17/9 := by
sorry

end system_solution_l3613_361346


namespace phi_value_l3613_361345

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem phi_value (φ : ℝ) :
  -π ≤ φ ∧ φ < π →
  (∀ x, f (x - π/2) φ = Real.sin x * Real.cos x + (Real.sqrt 3 / 2) * Real.cos x) →
  |φ| = 5*π/6 := by
  sorry

end phi_value_l3613_361345


namespace diego_fruit_problem_l3613_361381

/-- Given a bag with capacity for fruit and some fruits already in the bag,
    calculate the remaining capacity for additional fruit. -/
def remaining_capacity (bag_capacity : ℕ) (occupied_capacity : ℕ) : ℕ :=
  bag_capacity - occupied_capacity

/-- Diego's fruit buying problem -/
theorem diego_fruit_problem (bag_capacity : ℕ) (watermelon_weight : ℕ) (grapes_weight : ℕ) (oranges_weight : ℕ) 
  (h1 : bag_capacity = 20)
  (h2 : watermelon_weight = 1)
  (h3 : grapes_weight = 1)
  (h4 : oranges_weight = 1) :
  remaining_capacity bag_capacity (watermelon_weight + grapes_weight + oranges_weight) = 17 :=
sorry

end diego_fruit_problem_l3613_361381


namespace stock_price_calculation_l3613_361389

/-- Calculates the price of a stock given the investment amount, stock percentage, and annual income. -/
theorem stock_price_calculation (investment : ℝ) (stock_percentage : ℝ) (annual_income : ℝ) :
  investment = 6800 ∧ 
  stock_percentage = 0.6 ∧ 
  annual_income = 3000 →
  ∃ (stock_price : ℝ), stock_price = 136 := by
  sorry

end stock_price_calculation_l3613_361389


namespace complex_number_properties_l3613_361303

theorem complex_number_properties (z : ℂ) (h : z = (2 * Complex.I) / (-1 - Complex.I)) : 
  z ^ 2 = 2 * Complex.I ∧ z.im = -1 := by sorry

end complex_number_properties_l3613_361303


namespace number_of_sodas_bought_l3613_361351

/-- Given the total cost, sandwich cost, and soda cost, calculate the number of sodas bought -/
theorem number_of_sodas_bought (total_cost sandwich_cost soda_cost : ℚ) 
  (h_total : total_cost = 8.36)
  (h_sandwich : sandwich_cost = 2.44)
  (h_soda : soda_cost = 0.87) :
  (total_cost - 2 * sandwich_cost) / soda_cost = 4 := by
sorry

end number_of_sodas_bought_l3613_361351


namespace least_common_denominator_l3613_361334

theorem least_common_denominator : 
  let denominators := [3, 4, 5, 6, 8, 9, 10]
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 3 4) 5) 6) 8) 9) 10 = 360 :=
by sorry

end least_common_denominator_l3613_361334


namespace x_value_proof_l3613_361322

theorem x_value_proof (x : ℝ) : (5 * x - 3)^3 = Real.sqrt 64 → x = 1 := by
  sorry

end x_value_proof_l3613_361322


namespace inequality_solution_l3613_361339

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) > 4 / x + 23 / 10) ↔ (x > -2 ∧ x < 0) := by
  sorry

end inequality_solution_l3613_361339


namespace factorization_equality_l3613_361340

theorem factorization_equality (x : ℝ) : 16 * x^3 + 8 * x^2 = 8 * x^2 * (2 * x + 1) := by
  sorry

end factorization_equality_l3613_361340


namespace vlads_height_in_feet_l3613_361358

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h_inches_lt_12 : inches < 12

/-- Converts a height to total inches -/
def Height.to_inches (h : Height) : ℕ :=
  h.feet * 12 + h.inches

/-- The height of Vlad's sister -/
def sister_height : Height :=
  { feet := 2, inches := 10, h_inches_lt_12 := by sorry }

/-- The difference in height between Vlad and his sister in inches -/
def height_difference : ℕ := 41

/-- Theorem: Vlad's height in feet is 6 -/
theorem vlads_height_in_feet :
  (Height.to_inches sister_height + height_difference) / 12 = 6 := by sorry

end vlads_height_in_feet_l3613_361358


namespace right_triangle_point_distance_l3613_361354

theorem right_triangle_point_distance (h d x : ℝ) : 
  h > 0 → d > 0 → x > 0 →
  x + Real.sqrt ((x + h)^2 + d^2) = h + d →
  x = h * d / (2 * h + d) := by
sorry

end right_triangle_point_distance_l3613_361354


namespace return_trip_time_l3613_361394

/-- Represents the flight scenario between two cities --/
structure FlightScenario where
  d : ℝ  -- distance between cities
  v : ℝ  -- speed of plane in still air
  u : ℝ  -- speed of wind
  outboundTime : ℝ  -- time from A to B against wind
  returnTimeDifference : ℝ  -- difference in return time compared to calm air

/-- Conditions for the flight scenario --/
def flightConditions (s : FlightScenario) : Prop :=
  s.v > 0 ∧ s.u > 0 ∧ s.d > 0 ∧
  s.outboundTime = 60 ∧
  s.returnTimeDifference = 10 ∧
  s.d = s.outboundTime * (s.v - s.u) ∧
  s.d / (s.v + s.u) = s.d / s.v - s.returnTimeDifference

/-- The theorem stating that the return trip takes 20 minutes --/
theorem return_trip_time (s : FlightScenario) 
  (h : flightConditions s) : s.d / (s.v + s.u) = 20 := by
  sorry


end return_trip_time_l3613_361394


namespace circle_and_line_properties_l3613_361313

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 1 = 0

-- Define the line l
def line_l (x y m : ℝ) : Prop := y = x + m

-- Define the center of a circle
def is_center (h k : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ r, ∀ x y, C x y ↔ (x - h)^2 + (y - k)^2 = r^2

-- Define tangency between a line and a circle
def is_tangent (l : ℝ → ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) (m : ℝ) : Prop :=
  ∃! p, (∃ x y, l x y m ∧ C x y ∧ p = (x, y))

-- Theorem statement
theorem circle_and_line_properties :
  (is_center 0 1 circle_C) ∧
  (∀ m, is_tangent line_l circle_C m ↔ (m = 3 ∨ m = -1)) :=
sorry

end circle_and_line_properties_l3613_361313


namespace trail_mix_weight_l3613_361341

/-- The weight of peanuts in pounds -/
def peanuts : ℝ := 0.16666666666666666

/-- The weight of chocolate chips in pounds -/
def chocolate_chips : ℝ := 0.16666666666666666

/-- The weight of raisins in pounds -/
def raisins : ℝ := 0.08333333333333333

/-- The total weight of trail mix in pounds -/
def total_weight : ℝ := peanuts + chocolate_chips + raisins

/-- Theorem stating that the total weight of trail mix is equal to 0.41666666666666663 pounds -/
theorem trail_mix_weight : total_weight = 0.41666666666666663 := by sorry

end trail_mix_weight_l3613_361341


namespace scientific_notation_2023_l3613_361320

/-- Scientific notation representation with a specified number of significant figures -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  sigFigs : ℕ

/-- Convert a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- Check if a ScientificNotation representation is valid -/
def isValidScientificNotation (sn : ScientificNotation) : Prop :=
  1 ≤ sn.coefficient ∧ sn.coefficient < 10 ∧ sn.sigFigs > 0

theorem scientific_notation_2023 :
  let sn := toScientificNotation 2023 2
  isValidScientificNotation sn ∧ sn.coefficient = 2.0 ∧ sn.exponent = 3 := by
  sorry

end scientific_notation_2023_l3613_361320


namespace local_maximum_at_two_l3613_361385

/-- The function f(x) defined as x(x-c)^2 --/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- The derivative of f(x) with respect to x --/
def f_derivative (c : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*c*x + c^2

/-- Theorem stating that the value of c for which f(x) has a local maximum at x=2 is 6 --/
theorem local_maximum_at_two (c : ℝ) : 
  (∀ x, x ≠ 2 → ∃ δ > 0, ∀ y, |y - 2| < δ → f c y ≤ f c 2) → c = 6 :=
sorry

end local_maximum_at_two_l3613_361385


namespace pythagorean_triple_with_24_and_7_l3613_361330

theorem pythagorean_triple_with_24_and_7 : 
  ∃ (x : ℕ), x > 0 ∧ x^2 + 7^2 = 24^2 ∨ x^2 = 24^2 + 7^2 → x = 25 :=
by sorry

end pythagorean_triple_with_24_and_7_l3613_361330


namespace sqrt_problem_l3613_361377

theorem sqrt_problem (h1 : Real.sqrt 15129 = 123) (h2 : Real.sqrt x = 0.123) : x = 0.015129 := by
  sorry

end sqrt_problem_l3613_361377


namespace min_bills_for_payment_l3613_361368

/-- Represents the available denominations of bills and coins --/
structure Denominations :=
  (ten_dollar : ℕ)
  (five_dollar : ℕ)
  (two_dollar : ℕ)
  (one_dollar : ℕ)
  (fifty_cent : ℕ)

/-- Calculates the minimum number of bills and coins needed to pay a given amount --/
def min_bills_and_coins (d : Denominations) (amount : ℚ) : ℕ :=
  sorry

/-- Tim's available bills and coins --/
def tims_denominations : Denominations :=
  { ten_dollar := 15
  , five_dollar := 7
  , two_dollar := 12
  , one_dollar := 20
  , fifty_cent := 10 }

/-- The theorem stating that Tim needs 17 bills and coins to pay $152.50 --/
theorem min_bills_for_payment :
  min_bills_and_coins tims_denominations (152.5 : ℚ) = 17 :=
sorry

end min_bills_for_payment_l3613_361368


namespace unique_partition_count_l3613_361335

/-- The number of ways to partition n into three distinct positive integers -/
def partition_count (n : ℕ) : ℕ :=
  (n - 1) * (n - 2) / 2 - 3 * ((n / 2) - 2) - 1

/-- Theorem stating that 18 is the only positive integer satisfying the condition -/
theorem unique_partition_count :
  ∀ n : ℕ, n > 0 → (partition_count n = n + 1 ↔ n = 18) := by sorry

end unique_partition_count_l3613_361335


namespace vector_addition_l3613_361360

/-- Given two vectors AB and BC in 2D space, prove that AC is their sum. -/
theorem vector_addition (AB BC : ℝ × ℝ) (h1 : AB = (2, 3)) (h2 : BC = (1, -4)) :
  AB.1 + BC.1 = 3 ∧ AB.2 + BC.2 = -1 := by
  sorry

end vector_addition_l3613_361360


namespace volume_of_special_prism_l3613_361349

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents a triangular prism -/
structure TriangularPrism where
  base : Set Point3D
  height : ℝ

/-- Given a cube, returns the midpoints of edges AB, AD, and AA₁ -/
def getMidpoints (c : Cube) : Set Point3D :=
  { Point3D.mk (c.edgeLength / 2) 0 0,
    Point3D.mk 0 (c.edgeLength / 2) 0,
    Point3D.mk 0 0 (c.edgeLength / 2) }

/-- Constructs a triangular prism from given midpoints -/
def constructPrism (midpoints : Set Point3D) (c : Cube) : TriangularPrism :=
  sorry

/-- Calculates the volume of a triangular prism -/
def prismVolume (p : TriangularPrism) : ℝ :=
  sorry

theorem volume_of_special_prism (c : Cube) :
  c.edgeLength = 1 →
  let midpoints := getMidpoints c
  let prism := constructPrism midpoints c
  prismVolume prism = 3/16 := by
  sorry

end volume_of_special_prism_l3613_361349


namespace equation_solution_l3613_361308

theorem equation_solution :
  ∃ y : ℚ, (1 : ℚ) / 3 + 1 / y = 7 / 9 ↔ y = 9 / 4 := by
  sorry

end equation_solution_l3613_361308


namespace sunglasses_and_hats_probability_l3613_361321

theorem sunglasses_and_hats_probability
  (total_sunglasses : ℕ)
  (total_hats : ℕ)
  (prob_hat_and_sunglasses : ℚ)
  (h1 : total_sunglasses = 80)
  (h2 : total_hats = 50)
  (h3 : prob_hat_and_sunglasses = 3/5) :
  (prob_hat_and_sunglasses * total_hats) / total_sunglasses = 3/8 := by
  sorry

end sunglasses_and_hats_probability_l3613_361321


namespace arithmetic_sequence_common_difference_l3613_361325

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 2 = 5 ∧
  a 3 + a 5 = 2

/-- The common difference of an arithmetic sequence with given conditions is -2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -2 := by
  sorry

end arithmetic_sequence_common_difference_l3613_361325


namespace min_value_sum_of_squares_l3613_361384

theorem min_value_sum_of_squares (x y z : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end min_value_sum_of_squares_l3613_361384


namespace tyler_saltwater_animals_l3613_361314

/-- The number of saltwater aquariums Tyler has -/
def saltwater_aquariums : ℕ := 56

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 39

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := saltwater_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_saltwater_animals = 2184 :=
by sorry

end tyler_saltwater_animals_l3613_361314


namespace cloth_cutting_l3613_361347

theorem cloth_cutting (S : ℝ) : 
  S / 2 + S / 4 = 75 → S = 100 := by
sorry

end cloth_cutting_l3613_361347


namespace sufficient_not_necessary_l3613_361371

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 9 → 1/a < 1/9) ∧ 
  (∃ a, 1/a < 1/9 ∧ ¬(a > 9)) :=
by sorry

end sufficient_not_necessary_l3613_361371


namespace neighborhood_cleanup_weight_l3613_361375

/-- The total weight of litter collected during a neighborhood clean-up. -/
def total_litter_weight (gina_bags : ℕ) (neighborhood_multiplier : ℕ) (bag_weight : ℕ) : ℕ :=
  (gina_bags + neighborhood_multiplier * gina_bags) * bag_weight

/-- Theorem stating that the total weight of litter collected is 664 pounds. -/
theorem neighborhood_cleanup_weight :
  total_litter_weight 2 82 4 = 664 := by
  sorry

end neighborhood_cleanup_weight_l3613_361375


namespace arithmetic_sequence_ninth_term_l3613_361383

/-- Given an arithmetic sequence where the 5th term is 23 and the 7th term is 37, 
    prove that the 9th term is 51. -/
theorem arithmetic_sequence_ninth_term 
  (a : ℤ) -- First term of the sequence
  (d : ℤ) -- Common difference
  (h1 : a + 4 * d = 23) -- 5th term is 23
  (h2 : a + 6 * d = 37) -- 7th term is 37
  : a + 8 * d = 51 := by -- 9th term is 51
sorry

end arithmetic_sequence_ninth_term_l3613_361383


namespace combined_tennis_percentage_l3613_361327

-- Define the given conditions
def north_students : ℕ := 1800
def north_tennis_percentage : ℚ := 25 / 100
def south_students : ℕ := 2700
def south_tennis_percentage : ℚ := 35 / 100

-- Define the theorem
theorem combined_tennis_percentage :
  let north_tennis := (north_students : ℚ) * north_tennis_percentage
  let south_tennis := (south_students : ℚ) * south_tennis_percentage
  let total_tennis := north_tennis + south_tennis
  let total_students := (north_students + south_students : ℚ)
  (total_tennis / total_students) * 100 = 31 := by
  sorry

end combined_tennis_percentage_l3613_361327


namespace base_6_addition_l3613_361301

def to_base_10 (n : ℕ) (base : ℕ) : ℕ := sorry

def from_base_10 (n : ℕ) (base : ℕ) : ℕ := sorry

theorem base_6_addition : 
  from_base_10 (to_base_10 5 6 + to_base_10 21 6) 6 = 30 := by sorry

end base_6_addition_l3613_361301


namespace binary_arithmetic_theorem_l3613_361380

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ ⟨i, bi⟩ acc => acc + if bi then 2^i else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

def binary_add (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a + binary_to_decimal b)

def binary_sub (a b : List Bool) : List Bool :=
  decimal_to_binary (binary_to_decimal a - binary_to_decimal b)

theorem binary_arithmetic_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [false, true, false, true] -- 1010₂
  let d := [true, false, false, true] -- 1001₂
  binary_add (binary_sub (binary_add a b) c) d = [true, false, false, false, true] -- 10001₂
  := by sorry

end binary_arithmetic_theorem_l3613_361380


namespace triangle_problem_l3613_361310

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) (h1 : t.a * Real.cos t.C - t.c * Real.sin t.A = 0)
    (h2 : t.b = 4) (h3 : (1/2) * t.a * t.b * Real.sin t.C = 6) :
    t.C = π/3 ∧ t.c = 2 * Real.sqrt 7 := by
  sorry


end triangle_problem_l3613_361310


namespace factorization_4a_squared_minus_2a_l3613_361373

-- Define what it means for an expression to be a factorization from left to right
def is_factorization_left_to_right (f g : ℝ → ℝ) : Prop :=
  ∃ (h k : ℝ → ℝ), (∀ x, f x = h x * k x) ∧ (∀ x, g x = h x * k x) ∧ (f ≠ g)

-- Define the left side of the equation
def left_side (a : ℝ) : ℝ := 4 * a^2 - 2 * a

-- Define the right side of the equation
def right_side (a : ℝ) : ℝ := 2 * a * (2 * a - 1)

-- Theorem statement
theorem factorization_4a_squared_minus_2a :
  is_factorization_left_to_right left_side right_side :=
sorry

end factorization_4a_squared_minus_2a_l3613_361373


namespace crayon_count_initial_crayon_count_l3613_361348

theorem crayon_count (crayons_taken : ℕ) (crayons_left : ℕ) : ℕ :=
  crayons_taken + crayons_left

theorem initial_crayon_count : crayon_count 3 4 = 7 := by
  sorry

end crayon_count_initial_crayon_count_l3613_361348


namespace circle_equation_standard_form_tangent_line_b_value_l3613_361319

open Real

/-- A line ax + by = c is tangent to a circle (x - h)^2 + (y - k)^2 = r^2 if and only if
    the distance from the center (h, k) to the line is equal to the radius r. -/
def is_tangent_line_to_circle (a b c h k r : ℝ) : Prop :=
  (|a * h + b * k - c| / sqrt (a^2 + b^2)) = r

/-- The equation of the circle x^2 + y^2 - 2x - 2y + 1 = 0 in standard form is (x - 1)^2 + (y - 1)^2 = 1 -/
theorem circle_equation_standard_form :
  ∀ x y : ℝ, x^2 + y^2 - 2*x - 2*y + 1 = 0 ↔ (x - 1)^2 + (y - 1)^2 = 1 :=
sorry

/-- The main theorem: If the line 3x + 4y = b is tangent to the circle x^2 + y^2 - 2x - 2y + 1 = 0,
    then b = 2 or b = 12 -/
theorem tangent_line_b_value :
  ∀ b : ℝ, (∀ x y : ℝ, is_tangent_line_to_circle 3 4 b 1 1 1) → (b = 2 ∨ b = 12) :=
sorry

end circle_equation_standard_form_tangent_line_b_value_l3613_361319


namespace horner_rule_evaluation_l3613_361305

/-- Horner's Rule evaluation for a polynomial --/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 + 35x - 8x² + 79x³ + 6x⁴ + 5x⁵ + 3x⁶ --/
def f : List ℤ := [12, 35, -8, 79, 6, 5, 3]

/-- Theorem: The value of f(-4) using Horner's Rule is 220 --/
theorem horner_rule_evaluation :
  horner_eval f (-4) = 220 := by
  sorry

end horner_rule_evaluation_l3613_361305


namespace solution_mixture_proof_l3613_361336

theorem solution_mixture_proof (x : ℝ) 
  (h1 : x + 20 = 100) -- First solution is x% carbonated water and 20% lemonade
  (h2 : 0.6799999999999997 * x + 0.32000000000000003 * 55 = 72) -- Mixture equation
  : x = 80 := by
  sorry

end solution_mixture_proof_l3613_361336


namespace sum_of_squares_theorem_l3613_361357

theorem sum_of_squares_theorem (a d : ℤ) : 
  ∃ (x y z w : ℤ), 
    a^2 + 2*(a+d)^2 + 3*(a+2*d)^2 + 4*(a+3*d)^2 = (x*a + y*d)^2 + (z*a + w*d)^2 := by
  sorry

end sum_of_squares_theorem_l3613_361357


namespace emilia_valentin_numbers_l3613_361359

theorem emilia_valentin_numbers (x : ℝ) : 
  (5 + 9) / 2 = 7 ∧ 
  (5 + x) / 2 = 10 ∧ 
  (x + 9) / 2 = 12 → 
  x = 15 := by
sorry

end emilia_valentin_numbers_l3613_361359


namespace terminal_side_angle_expression_l3613_361382

theorem terminal_side_angle_expression (α : Real) :
  let P : Real × Real := (1, 3)
  let r : Real := Real.sqrt (P.1^2 + P.2^2)
  (P.1 / r = Real.cos α) ∧ (P.2 / r = Real.sin α) →
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / (2 * Real.cos (α - 2 * π)) = 1 :=
by sorry

end terminal_side_angle_expression_l3613_361382


namespace area_of_enclosed_region_l3613_361326

/-- The curve defined by |x-1| + |y-1| = 1 -/
def curve (x y : ℝ) : Prop := |x - 1| + |y - 1| = 1

/-- The region enclosed by the curve -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve p.1 p.2}

/-- The area of the region enclosed by the curve is 2 -/
theorem area_of_enclosed_region : MeasureTheory.volume enclosed_region = 2 := by
  sorry

end area_of_enclosed_region_l3613_361326


namespace roses_kept_l3613_361344

theorem roses_kept (initial : ℕ) (mother grandmother sister : ℕ) 
  (h1 : initial = 20)
  (h2 : mother = 6)
  (h3 : grandmother = 9)
  (h4 : sister = 4) :
  initial - (mother + grandmother + sister) = 1 := by
  sorry

end roses_kept_l3613_361344


namespace theater_population_l3613_361338

theorem theater_population :
  ∀ (total : ℕ),
  (19 : ℕ) + (total / 2) + (total / 4) + 6 = total →
  total = 100 :=
by
  sorry

end theater_population_l3613_361338


namespace small_planter_capacity_l3613_361353

/-- Given the total number of seeds, the number and capacity of large planters,
    and the number of small planters, prove that each small planter can hold 4 seeds. -/
theorem small_planter_capacity
  (total_seeds : ℕ)
  (large_planters : ℕ)
  (large_planter_capacity : ℕ)
  (small_planters : ℕ)
  (h1 : total_seeds = 200)
  (h2 : large_planters = 4)
  (h3 : large_planter_capacity = 20)
  (h4 : small_planters = 30)
  : (total_seeds - large_planters * large_planter_capacity) / small_planters = 4 := by
  sorry

end small_planter_capacity_l3613_361353


namespace speedster_convertibles_l3613_361316

/-- The number of Speedster convertibles given the total number of vehicles,
    non-Speedsters, and the fraction of Speedsters that are convertibles. -/
theorem speedster_convertibles
  (total_vehicles : ℕ)
  (non_speedsters : ℕ)
  (speedster_convertible_fraction : ℚ)
  (h1 : total_vehicles = 80)
  (h2 : non_speedsters = 50)
  (h3 : speedster_convertible_fraction = 4/5) :
  (total_vehicles - non_speedsters) * speedster_convertible_fraction = 24 := by
  sorry

#eval (80 - 50) * (4/5 : ℚ)

end speedster_convertibles_l3613_361316


namespace max_expression_value_l3613_361342

def expression (a b c d : ℕ) : ℕ := c * a^b - d

theorem max_expression_value :
  ∃ (a b c d : ℕ),
    a ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    b ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    c ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    d ∈ ({1, 2, 3, 4} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    expression a b c d = 5 ∧
    ∀ (a' b' c' d' : ℕ),
      a' ∈ ({1, 2, 3, 4} : Set ℕ) →
      b' ∈ ({1, 2, 3, 4} : Set ℕ) →
      c' ∈ ({1, 2, 3, 4} : Set ℕ) →
      d' ∈ ({1, 2, 3, 4} : Set ℕ) →
      a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ b' ≠ c' ∧ b' ≠ d' ∧ c' ≠ d' →
      expression a' b' c' d' ≤ 5 :=
sorry

end max_expression_value_l3613_361342


namespace sum_of_roots_quadratic_l3613_361392

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ - 2 = 0) → 
  (x₂^2 - 4*x₂ - 2 = 0) → 
  (x₁ + x₂ = 4) := by
sorry

end sum_of_roots_quadratic_l3613_361392


namespace ladybug_leaves_l3613_361388

theorem ladybug_leaves (ladybugs_per_leaf : ℕ) (total_ladybugs : ℕ) (h1 : ladybugs_per_leaf = 139) (h2 : total_ladybugs = 11676) :
  total_ladybugs / ladybugs_per_leaf = 84 := by
sorry

end ladybug_leaves_l3613_361388


namespace product_repeating_decimal_one_third_and_eight_l3613_361363

def repeating_decimal_one_third : ℚ := 1/3

theorem product_repeating_decimal_one_third_and_eight :
  repeating_decimal_one_third * 8 = 8/3 := by sorry

end product_repeating_decimal_one_third_and_eight_l3613_361363


namespace max_sum_of_squares_exists_max_sum_of_squares_l3613_361304

/-- Given a quadruple (a, b, c, d) satisfying certain conditions, 
    the sum of their squares is at most 254. -/
theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 12 →
  a * b + c + d = 47 →
  a * d + b * c = 88 →
  c * d = 54 →
  a^2 + b^2 + c^2 + d^2 ≤ 254 := by
  sorry

/-- There exists a quadruple (a, b, c, d) satisfying the conditions 
    where the sum of their squares equals 254. -/
theorem exists_max_sum_of_squares : 
  ∃ (a b c d : ℝ), 
    a + b = 12 ∧
    a * b + c + d = 47 ∧
    a * d + b * c = 88 ∧
    c * d = 54 ∧
    a^2 + b^2 + c^2 + d^2 = 254 := by
  sorry

end max_sum_of_squares_exists_max_sum_of_squares_l3613_361304


namespace simplify_fraction_product_l3613_361369

theorem simplify_fraction_product : 8 * (15 / 9) * (-45 / 40) = -1 / 15 := by
  sorry

end simplify_fraction_product_l3613_361369


namespace score_difference_l3613_361306

theorem score_difference (chuck_team_score red_team_score : ℕ) 
  (h1 : chuck_team_score = 95) 
  (h2 : red_team_score = 76) : 
  chuck_team_score - red_team_score = 19 := by
sorry

end score_difference_l3613_361306


namespace work_completion_time_l3613_361350

/-- The efficiency of worker q -/
def q_efficiency : ℝ := 1

/-- The efficiency of worker p relative to q -/
def p_efficiency : ℝ := 1.6

/-- The efficiency of worker r relative to q -/
def r_efficiency : ℝ := 1.4

/-- The time taken by p alone to complete the work -/
def p_time : ℝ := 26

/-- The total amount of work to be done -/
def total_work : ℝ := p_efficiency * p_time

/-- The combined efficiency of p, q, and r -/
def combined_efficiency : ℝ := p_efficiency + q_efficiency + r_efficiency

/-- The theorem stating the time taken for p, q, and r to complete the work together -/
theorem work_completion_time : 
  total_work / combined_efficiency = 10.4 := by sorry

end work_completion_time_l3613_361350


namespace net_profit_is_107_70_l3613_361374

/-- Laundry shop rates and quantities for a three-day period --/
structure LaundryData where
  regular_rate : ℝ
  delicate_rate : ℝ
  business_rate : ℝ
  bulky_rate : ℝ
  discount_rate : ℝ
  day1_regular : ℝ
  day1_delicate : ℝ
  day1_business : ℝ
  day1_bulky : ℝ
  day2_regular : ℝ
  day2_delicate : ℝ
  day2_business : ℝ
  day2_bulky : ℝ
  day3_regular : ℝ
  day3_delicate : ℝ
  day3_business : ℝ
  day3_bulky : ℝ
  overhead_costs : ℝ

/-- Calculate the net profit for a three-day period in a laundry shop --/
def calculate_net_profit (data : LaundryData) : ℝ :=
  let day1_total := data.regular_rate * data.day1_regular +
                    data.delicate_rate * data.day1_delicate +
                    data.business_rate * data.day1_business +
                    data.bulky_rate * data.day1_bulky
  let day2_total := data.regular_rate * data.day2_regular +
                    data.delicate_rate * data.day2_delicate +
                    data.business_rate * data.day2_business +
                    data.bulky_rate * data.day2_bulky
  let day3_total := (data.regular_rate * data.day3_regular +
                    data.delicate_rate * data.day3_delicate +
                    data.business_rate * data.day3_business +
                    data.bulky_rate * data.day3_bulky) * (1 - data.discount_rate)
  day1_total + day2_total + day3_total - data.overhead_costs

/-- Theorem: The net profit for the given three-day period is $107.70 --/
theorem net_profit_is_107_70 :
  let data := LaundryData.mk 3 4 5 6 0.1 7 4 3 2 10 6 4 3 20 4 5 2 150
  calculate_net_profit data = 107.7 := by
  sorry

end net_profit_is_107_70_l3613_361374


namespace min_value_n_plus_32_over_n_squared_l3613_361329

theorem min_value_n_plus_32_over_n_squared (n : ℝ) (h : n > 0) :
  n + 32 / n^2 ≥ 6 ∧ ∃ n₀ > 0, n₀ + 32 / n₀^2 = 6 := by sorry

end min_value_n_plus_32_over_n_squared_l3613_361329


namespace similar_squares_side_length_l3613_361337

theorem similar_squares_side_length (s1 s2 : ℝ) (h1 : s1 > 0) (h2 : s2 > 0) : 
  (s1 ^ 2 : ℝ) / (s2 ^ 2) = 9 → s2 = 5 → s1 = 15 := by sorry

end similar_squares_side_length_l3613_361337


namespace unique_prime_103207_l3613_361372

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_prime_103207 :
  (¬ is_prime 103201) ∧
  (¬ is_prime 103202) ∧
  (¬ is_prime 103203) ∧
  (is_prime 103207) ∧
  (¬ is_prime 103209) :=
sorry

end unique_prime_103207_l3613_361372


namespace smallest_value_operation_l3613_361311

theorem smallest_value_operation (a b : ℤ) (h1 : a = -3) (h2 : b = -6) :
  a + b ≤ min (a - b) (min (a * b) (a / b)) := by sorry

end smallest_value_operation_l3613_361311


namespace inscribed_equilateral_triangle_side_length_l3613_361309

theorem inscribed_equilateral_triangle_side_length 
  (diameter : ℝ) (side_length : ℝ) 
  (h1 : diameter = 2000) 
  (h2 : side_length = 1732 + 1/20) : 
  side_length = diameter / 2 * Real.sqrt 3 := by
  sorry

end inscribed_equilateral_triangle_side_length_l3613_361309


namespace perpendicular_line_equation_l3613_361300

-- Define the given line
def given_line (x y : ℝ) (c : ℝ) : Prop := x - 2 * y + c = 0

-- Define the point through which the perpendicular line passes
def point : ℝ × ℝ := (-1, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  ∀ (c : ℝ),
  (perpendicular_line point.1 point.2) ∧
  (∀ (x y : ℝ), perpendicular_line x y →
    ∃ (m : ℝ), m * (x - point.1) = y - point.2 ∧
    m * (-1/2) = -1) ∧
  (∀ (x y : ℝ), given_line x y c →
    ∃ (m : ℝ), y = m * x + c / 2 ∧ m = 1/2) :=
by sorry

end perpendicular_line_equation_l3613_361300


namespace three_rug_overlap_l3613_361362

theorem three_rug_overlap (total_rug_area floor_area double_layer_area : ℝ) 
  (h1 : total_rug_area = 90)
  (h2 : floor_area = 60)
  (h3 : double_layer_area = 12) : 
  ∃ (triple_layer_area : ℝ),
    triple_layer_area = 9 ∧
    ∃ (single_layer_area : ℝ),
      single_layer_area + double_layer_area + triple_layer_area = floor_area ∧
      single_layer_area + 2 * double_layer_area + 3 * triple_layer_area = total_rug_area :=
by sorry

end three_rug_overlap_l3613_361362


namespace petes_flag_shapes_petes_flag_total_shapes_l3613_361323

/-- Calculates the total number of shapes on Pete's flag based on US flag specifications -/
theorem petes_flag_shapes (us_stars : Nat) (us_stripes : Nat) : Nat :=
  let circles := us_stars / 2 - 3
  let squares := us_stripes * 2 + 6
  circles + squares

/-- Proves that the total number of shapes on Pete's flag is 54 -/
theorem petes_flag_total_shapes : 
  petes_flag_shapes 50 13 = 54 := by
  sorry

end petes_flag_shapes_petes_flag_total_shapes_l3613_361323


namespace intersection_angle_l3613_361333

/-- A regular hexagonal pyramid with lateral faces at 45° to the base -/
structure RegularHexagonalPyramid :=
  (base : Set (ℝ × ℝ))
  (apex : ℝ × ℝ × ℝ)
  (lateral_angle : Real)
  (is_regular : Bool)
  (lateral_angle_eq : lateral_angle = Real.pi / 4)

/-- A plane intersecting the pyramid -/
structure IntersectingPlane :=
  (base_edge : Set (ℝ × ℝ))
  (intersections : Set (ℝ × ℝ × ℝ))
  (is_parallel : Bool)

/-- The theorem to be proved -/
theorem intersection_angle (p : RegularHexagonalPyramid) (s : IntersectingPlane) :
  p.is_regular ∧ s.is_parallel →
  ∃ α : Real, α = Real.arctan (1 / 2) := by
  sorry

end intersection_angle_l3613_361333


namespace probability_point_closer_to_center_l3613_361378

theorem probability_point_closer_to_center (R : ℝ) (r : ℝ) : R > 0 → r > 0 → R = 3 * r →
  (π * (2 * r)^2) / (π * R^2) = 4 / 9 := by
  sorry

end probability_point_closer_to_center_l3613_361378


namespace chess_tournament_games_l3613_361364

theorem chess_tournament_games (n : ℕ) (h : n = 25) : 
  n * (n - 1) = 600 → 2 * (n * (n - 1)) = 1200 := by
  sorry

#check chess_tournament_games

end chess_tournament_games_l3613_361364


namespace rectangular_to_polar_sqrt2_l3613_361312

theorem rectangular_to_polar_sqrt2 :
  ∃ (r : ℝ) (θ : ℝ), 
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r * Real.cos θ = Real.sqrt 2 ∧
    r * Real.sin θ = -Real.sqrt 2 ∧
    r = 2 ∧
    θ = 7 * Real.pi / 4 := by
  sorry

end rectangular_to_polar_sqrt2_l3613_361312


namespace decreasing_function_a_range_l3613_361379

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a
  else Real.log x / Real.log a

-- Theorem statement
theorem decreasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  (1 / 7 : ℝ) ≤ a ∧ a < (1 / 3 : ℝ) :=
by sorry

end decreasing_function_a_range_l3613_361379
