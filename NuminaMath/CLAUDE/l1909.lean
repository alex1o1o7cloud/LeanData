import Mathlib

namespace molecular_weight_not_affects_l1909_190932

-- Define plasma osmotic pressure
def plasma_osmotic_pressure : ℝ → ℝ := sorry

-- Define factors that affect plasma osmotic pressure
def protein_content : ℝ := sorry
def cl_content : ℝ := sorry
def na_content : ℝ := sorry
def molecular_weight_protein : ℝ := sorry

-- State that protein content affects plasma osmotic pressure
axiom protein_content_affects : ∃ (ε : ℝ), ε ≠ 0 ∧ 
  plasma_osmotic_pressure (protein_content + ε) ≠ plasma_osmotic_pressure protein_content

-- State that Cl- content affects plasma osmotic pressure
axiom cl_content_affects : ∃ (ε : ℝ), ε ≠ 0 ∧ 
  plasma_osmotic_pressure (cl_content + ε) ≠ plasma_osmotic_pressure cl_content

-- State that Na+ content affects plasma osmotic pressure
axiom na_content_affects : ∃ (ε : ℝ), ε ≠ 0 ∧ 
  plasma_osmotic_pressure (na_content + ε) ≠ plasma_osmotic_pressure na_content

-- Theorem: Molecular weight of plasma protein does not affect plasma osmotic pressure
theorem molecular_weight_not_affects : ∀ (ε : ℝ), ε ≠ 0 → 
  plasma_osmotic_pressure (molecular_weight_protein + ε) = plasma_osmotic_pressure molecular_weight_protein :=
sorry

end molecular_weight_not_affects_l1909_190932


namespace sum_prod_nonzero_digits_equals_46_pow_2009_l1909_190976

/-- The number of digits in the problem -/
def n : ℕ := 2009

/-- Calculate the product of non-zero digits for a given natural number -/
def prod_nonzero_digits (k : ℕ) : ℕ := sorry

/-- Sum of products of non-zero digits for integers from 1 to 10^n -/
def sum_prod_nonzero_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the sum of products of non-zero digits for integers from 1 to 10^2009 -/
theorem sum_prod_nonzero_digits_equals_46_pow_2009 :
  sum_prod_nonzero_digits n = 46^n := by sorry

end sum_prod_nonzero_digits_equals_46_pow_2009_l1909_190976


namespace polly_cooking_time_l1909_190961

/-- Represents the cooking time for a week -/
structure CookingTime where
  breakfast_daily : ℕ
  lunch_daily : ℕ
  dinner_four_days : ℕ
  total_week : ℕ

/-- Calculates the time spent cooking dinner on the remaining days -/
def remaining_dinner_time (c : CookingTime) : ℕ :=
  c.total_week - (7 * (c.breakfast_daily + c.lunch_daily) + 4 * c.dinner_four_days)

/-- Theorem stating that given the conditions, Polly spends 90 minutes cooking dinner on the remaining days -/
theorem polly_cooking_time :
  let c : CookingTime := {
    breakfast_daily := 20,
    lunch_daily := 5,
    dinner_four_days := 10,
    total_week := 305
  }
  remaining_dinner_time c = 90 := by sorry

end polly_cooking_time_l1909_190961


namespace sequence_terms_equal_twenty_l1909_190988

def a (n : ℕ) : ℤ := n^2 - 14*n + 65

theorem sequence_terms_equal_twenty :
  (∀ n : ℕ, a n = 20 ↔ n = 5 ∨ n = 9) :=
sorry

end sequence_terms_equal_twenty_l1909_190988


namespace no_solution_system_l1909_190992

theorem no_solution_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 8) ∧ (6 * x - 8 * y = 12) := by
sorry

end no_solution_system_l1909_190992


namespace breakfast_cost_theorem_l1909_190946

/-- The cost of each meal in Herman's breakfast purchases -/
def meal_cost (people : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_spent : ℕ) : ℚ :=
  total_spent / (people * days_per_week * weeks)

/-- Theorem stating that the meal cost is $4 given the problem conditions -/
theorem breakfast_cost_theorem :
  meal_cost 4 5 16 1280 = 4 := by
  sorry

end breakfast_cost_theorem_l1909_190946


namespace apple_count_theorem_l1909_190935

/-- The number of apples originally on the tree -/
def original_apples : ℕ := 9

/-- The number of apples picked from the tree -/
def picked_apples : ℕ := 2

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := 7

/-- Theorem stating that the original number of apples is equal to
    the sum of remaining and picked apples -/
theorem apple_count_theorem :
  original_apples = remaining_apples + picked_apples :=
by sorry

end apple_count_theorem_l1909_190935


namespace at_least_two_boundary_triangles_l1909_190955

/-- A polygon divided into triangles by non-intersecting diagonals -/
structure TriangulatedPolygon where
  /-- The number of sides of the polygon -/
  n : ℕ
  /-- The number of triangles with exactly i sides as sides of the polygon -/
  k : Fin 3 → ℕ
  /-- The total number of triangles is n - 2 -/
  total_triangles : k 0 + k 1 + k 2 = n - 2
  /-- The total number of polygon sides used in forming triangles is n -/
  total_sides : 2 * k 2 + k 1 = n

/-- 
In a polygon divided into triangles by non-intersecting diagonals, 
there are at least two triangles that have at least two sides 
coinciding with the sides of the original polygon.
-/
theorem at_least_two_boundary_triangles (p : TriangulatedPolygon) : 
  p.k 2 ≥ 2 := by
  sorry

end at_least_two_boundary_triangles_l1909_190955


namespace ufo_convention_attendees_l1909_190938

theorem ufo_convention_attendees :
  let total_attendees : ℕ := 450
  let male_female_difference : ℕ := 26
  let male_attendees : ℕ := (total_attendees + male_female_difference) / 2
  male_attendees = 238 :=
by sorry

end ufo_convention_attendees_l1909_190938


namespace line_l_equation_l1909_190954

-- Define the points and lines
def P : ℝ × ℝ := (-1, 1)
def l₁ : Set (ℝ × ℝ) := {(x, y) | x + 2*y - 5 = 0}
def l₂ : Set (ℝ × ℝ) := {(x, y) | x + 2*y - 3 = 0}
def l₃ : Set (ℝ × ℝ) := {(x, y) | x - y - 1 = 0}

-- Define the line l (we'll prove this is correct)
def l : Set (ℝ × ℝ) := {(x, y) | y = 1}

-- Define the properties of the problem
theorem line_l_equation (M : ℝ × ℝ) :
  P ∈ l ∧  -- l passes through P
  (∃ M₁ M₂, M₁ ∈ l ∩ l₁ ∧ M₂ ∈ l ∩ l₂ ∧  -- l intersects l₁ and l₂
    M = ((M₁.1 + M₂.1) / 2, (M₁.2 + M₂.2) / 2)) ∧  -- M is the midpoint of M₁M₂
  M ∈ l₃  -- M lies on l₃
  → l = {(x, y) | y = 1} :=
by sorry

end line_l_equation_l1909_190954


namespace quadratic_vertex_form_h_l1909_190951

/-- 
Given a quadratic expression 3x^2 + 9x + 20, when expressed in the form a(x - h)^2 + k,
h is equal to -3/2
-/
theorem quadratic_vertex_form_h (x : ℝ) : 
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k := by
  sorry

end quadratic_vertex_form_h_l1909_190951


namespace tank_dimension_l1909_190933

theorem tank_dimension (cost_per_sqft : ℝ) (total_cost : ℝ) (length : ℝ) (width : ℝ) :
  cost_per_sqft = 20 →
  total_cost = 1440 →
  length = 3 →
  width = 6 →
  ∃ height : ℝ, 
    height = 2 ∧ 
    total_cost = cost_per_sqft * (2 * (length * width + length * height + width * height)) :=
by sorry

end tank_dimension_l1909_190933


namespace double_time_double_discount_l1909_190928

/-- Represents the true discount calculation for a bill -/
def true_discount (face_value : ℝ) (discount : ℝ) (time : ℝ) : Prop :=
  ∃ (rate : ℝ),
    discount = (face_value - discount) * rate * time ∧
    rate > 0 ∧
    time > 0

/-- 
If the true discount on a bill of 110 is 10 for a certain time,
then the true discount on the same bill for double the time is 20.
-/
theorem double_time_double_discount :
  ∀ (time : ℝ),
    true_discount 110 10 time →
    true_discount 110 20 (2 * time) :=
by
  sorry

end double_time_double_discount_l1909_190928


namespace circle_equation_tangent_to_x_axis_l1909_190907

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is on a circle --/
def Circle.isOn (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a circle is tangent to the x-axis --/
def Circle.tangentToXAxis (c : Circle) : Prop :=
  c.center.2 = c.radius

theorem circle_equation_tangent_to_x_axis (x y : ℝ) :
  (x - 5)^2 + (y - 4)^2 = 16 ↔
  ∃ (c : Circle), c.center = (5, 4) ∧ c.radius = 4 ∧
  c.isOn (x, y) ∧ c.tangentToXAxis :=
sorry

end circle_equation_tangent_to_x_axis_l1909_190907


namespace perfect_squares_from_equation_l1909_190924

theorem perfect_squares_from_equation (x y : ℕ) (h : 2 * x^2 + x = 3 * y^2 + y) :
  ∃ (a b c : ℕ), (x - y = a^2) ∧ (2 * x + 2 * y + 1 = b^2) ∧ (3 * x + 3 * y + 1 = c^2) := by
  sorry

end perfect_squares_from_equation_l1909_190924


namespace cody_bill_is_99_l1909_190915

/-- Represents the cost and quantity of tickets for Cody's order -/
structure TicketOrder where
  childPrice : ℚ
  adultPrice : ℚ
  childCount : ℕ
  adultCount : ℕ

/-- Calculates the total bill for a given ticket order -/
def totalBill (order : TicketOrder) : ℚ :=
  order.childPrice * order.childCount + order.adultPrice * order.adultCount

/-- Theorem stating that Cody's total bill is $99.00 given the problem conditions -/
theorem cody_bill_is_99 : ∃ (order : TicketOrder),
  order.childPrice = 7.5 ∧
  order.adultPrice = 12 ∧
  order.childCount = order.adultCount + 8 ∧
  order.childCount + order.adultCount = 12 ∧
  totalBill order = 99 := by
  sorry

end cody_bill_is_99_l1909_190915


namespace min_value_expression_l1909_190937

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x + 2 * y)) + (y / x) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end min_value_expression_l1909_190937


namespace stratified_sample_under35_l1909_190999

/-- Represents the number of employees in each age group -/
structure EmployeeGroups where
  total : ℕ
  under35 : ℕ
  between35and49 : ℕ
  over50 : ℕ

/-- Calculates the number of employees to be drawn from a specific group in stratified sampling -/
def stratifiedSampleSize (groups : EmployeeGroups) (sampleTotal : ℕ) (groupSize : ℕ) : ℕ :=
  (groupSize * sampleTotal) / groups.total

/-- Theorem stating that in the given scenario, 25 employees under 35 should be drawn -/
theorem stratified_sample_under35 (groups : EmployeeGroups) (sampleTotal : ℕ) :
  groups.total = 500 →
  groups.under35 = 125 →
  groups.between35and49 = 280 →
  groups.over50 = 95 →
  sampleTotal = 100 →
  stratifiedSampleSize groups sampleTotal groups.under35 = 25 := by
  sorry

#check stratified_sample_under35

end stratified_sample_under35_l1909_190999


namespace f_increasing_f_two_zeros_l1909_190919

/-- The function f(x) = 2|x+1| + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * abs (x + 1) + a * x

/-- Theorem: f(x) is increasing on ℝ when a > 2 -/
theorem f_increasing (a : ℝ) (h : a > 2) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂ :=
sorry

/-- Theorem: f(x) has exactly two zeros if and only if a ∈ (0,2) -/
theorem f_two_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ ∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ↔
  0 < a ∧ a < 2 :=
sorry

end f_increasing_f_two_zeros_l1909_190919


namespace max_soap_boxes_in_carton_l1909_190925

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 30, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 5 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 360 := by
  sorry

#eval (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions)

end max_soap_boxes_in_carton_l1909_190925


namespace net_amount_calculation_l1909_190969

/-- Calculate the net amount received from selling puppies --/
def net_amount_from_puppies (luna_puppies stella_puppies : ℕ)
                            (luna_sold stella_sold : ℕ)
                            (luna_price stella_price : ℕ)
                            (luna_cost stella_cost : ℕ) : ℕ :=
  let luna_revenue := luna_sold * luna_price
  let stella_revenue := stella_sold * stella_price
  let luna_expenses := luna_puppies * luna_cost
  let stella_expenses := stella_puppies * stella_cost
  (luna_revenue + stella_revenue) - (luna_expenses + stella_expenses)

theorem net_amount_calculation :
  net_amount_from_puppies 10 14 8 10 200 250 80 90 = 2040 := by
  sorry

end net_amount_calculation_l1909_190969


namespace sabrina_cookies_l1909_190984

theorem sabrina_cookies (initial_cookies : ℕ) (final_cookies : ℕ) 
  (h1 : initial_cookies = 20) 
  (h2 : final_cookies = 5) : ℕ :=
  let cookies_to_brother := 10
  let cookies_from_mother := cookies_to_brother / 2
  let total_before_sister := initial_cookies - cookies_to_brother + cookies_from_mother
  let cookies_kept := total_before_sister / 3
  by
    have h3 : cookies_kept = final_cookies := by sorry
    have h4 : total_before_sister = cookies_kept * 3 := by sorry
    have h5 : initial_cookies - cookies_to_brother + cookies_from_mother = total_before_sister := by sorry
    exact cookies_to_brother

end sabrina_cookies_l1909_190984


namespace f_properties_l1909_190944

/-- The quadratic function f(x) = x^2 + ax + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- Theorem stating the properties of the function f --/
theorem f_properties (a : ℝ) :
  (∃ (s : Set ℝ), ∀ x, f a x > 0 ↔ x ∈ s) ∧
  (∀ x > 0, f a x ≥ 0) ↔ a ≥ -2 :=
sorry

end f_properties_l1909_190944


namespace probability_of_letter_selection_l1909_190972

theorem probability_of_letter_selection (total_letters : ℕ) (unique_letters : ℕ) 
  (h1 : total_letters = 26) (h2 : unique_letters = 8) :
  (unique_letters : ℚ) / total_letters = 4 / 13 := by
  sorry

end probability_of_letter_selection_l1909_190972


namespace f_is_even_f_increasing_on_nonneg_l1909_190942

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem for the parity of the function (even function)
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

-- Theorem for monotonic increase on [0, +∞)
theorem f_increasing_on_nonneg : ∀ x y : ℝ, 0 ≤ x → x < y → f x < f y := by sorry

end f_is_even_f_increasing_on_nonneg_l1909_190942


namespace upstairs_vacuuming_time_l1909_190930

/-- Represents the vacuuming problem with given conditions -/
def VacuumingProblem (downstairs upstairs total : ℕ) : Prop :=
  upstairs = 2 * downstairs + 5 ∧ 
  downstairs + upstairs = total ∧
  total = 38

/-- Proves that given the conditions, the time to vacuum upstairs is 27 minutes -/
theorem upstairs_vacuuming_time :
  ∀ downstairs upstairs total, 
  VacuumingProblem downstairs upstairs total → 
  upstairs = 27 := by
  sorry

end upstairs_vacuuming_time_l1909_190930


namespace area_of_triangle_ABC_l1909_190991

/-- A square surrounded by four identical regular triangles -/
structure SquareWithTriangles where
  /-- Side length of the square -/
  squareSide : ℝ
  /-- The square has side length 2 -/
  squareSideIs2 : squareSide = 2
  /-- Side length of the surrounding triangles that touches the square -/
  triangleSide : ℝ
  /-- The triangle side that touches the square is equal to the square side -/
  triangleSideEqSquareSide : triangleSide = squareSide
  /-- The surrounding triangles are regular -/
  trianglesAreRegular : True
  /-- The surrounding triangles are symmetrically placed -/
  trianglesAreSymmetric : True

/-- Triangle ABC formed by connecting midpoints of outer sides of surrounding triangles -/
def TriangleABC (swt : SquareWithTriangles) : Set (ℝ × ℝ) := sorry

/-- The area of Triangle ABC -/
def areaOfTriangleABC (swt : SquareWithTriangles) : ℝ := sorry

/-- Theorem stating that the area of Triangle ABC is √3/2 -/
theorem area_of_triangle_ABC (swt : SquareWithTriangles) : 
  areaOfTriangleABC swt = Real.sqrt 3 / 2 := by sorry

end area_of_triangle_ABC_l1909_190991


namespace ellipse_hyperbola_relation_l1909_190911

/-- Given an ellipse and a hyperbola with coincident foci, prove that the semi-major axis of the ellipse is greater than that of the hyperbola, and the product of their eccentricities is greater than 1. -/
theorem ellipse_hyperbola_relation (m n : ℝ) (e₁ e₂ : ℝ) : 
  m > 1 →
  n > 0 →
  (∀ x y : ℝ, x^2 / m^2 + y^2 = 1 ↔ x^2 / n^2 - y^2 = 1) →
  e₁^2 = (m^2 - 1) / m^2 →
  e₂^2 = (n^2 + 1) / n^2 →
  m > n ∧ e₁ * e₂ > 1 :=
by sorry

end ellipse_hyperbola_relation_l1909_190911


namespace parallel_lines_distance_l1909_190917

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y = 2
def line2 (x y : ℝ) : Prop := 3 * x + 4 * y = 7

theorem parallel_lines_distance : 
  let d := |2 - 7| / Real.sqrt (3^2 + 4^2)
  d = 5 := by sorry

end parallel_lines_distance_l1909_190917


namespace water_trough_problem_l1909_190904

/-- Calculates the remaining water volume in a trough after a given number of days,
    given an initial volume and a constant daily evaporation rate. -/
def remaining_water_volume (initial_volume : ℝ) (evaporation_rate : ℝ) (days : ℝ) : ℝ :=
  initial_volume - evaporation_rate * days

/-- Proves that given an initial water volume of 300 gallons, with a constant evaporation rate
    of 1 gallon per day over 45 days and no additional water added or removed,
    the final water volume will be 255 gallons. -/
theorem water_trough_problem :
  remaining_water_volume 300 1 45 = 255 := by
  sorry


end water_trough_problem_l1909_190904


namespace unique_nonzero_elements_in_rows_and_columns_l1909_190981

open Matrix

theorem unique_nonzero_elements_in_rows_and_columns
  (n : ℕ)
  (A : Matrix (Fin n) (Fin n) ℝ)
  (h_nonneg : ∀ i j, 0 ≤ A i j)
  (h_nonsingular : IsUnit (det A))
  (h_inv_nonneg : ∀ i j, 0 ≤ A⁻¹ i j) :
  (∀ i, ∃! j, A i j ≠ 0) ∧ (∀ j, ∃! i, A i j ≠ 0) := by
  sorry

end unique_nonzero_elements_in_rows_and_columns_l1909_190981


namespace train_length_l1909_190918

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 3 → ∃ (length : ℝ), abs (length - 50.01) < 0.01 := by
  sorry


end train_length_l1909_190918


namespace unique_two_digit_multiple_l1909_190901

theorem unique_two_digit_multiple : ∃! t : ℕ, 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 := by
  sorry

end unique_two_digit_multiple_l1909_190901


namespace prove_c_minus_d_equals_negative_three_l1909_190967

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- Define c and d
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

-- State the theorem
theorem prove_c_minus_d_equals_negative_three :
  Function.Injective g ∧ g c = d ∧ g d = 5 → c - d = -3 := by sorry

end prove_c_minus_d_equals_negative_three_l1909_190967


namespace probability_intersection_independent_events_l1909_190906

theorem probability_intersection_independent_events 
  (p : Set α → ℝ) (a b : Set α) 
  (ha : p a = 4/5) 
  (hb : p b = 2/5) 
  (hab_indep : p (a ∩ b) = p a * p b) : 
  p (a ∩ b) = 8/25 := by
sorry

end probability_intersection_independent_events_l1909_190906


namespace circle_line_symmetry_l1909_190956

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two points are symmetric with respect to a line -/
def symmetric_points (p q : ℝ × ℝ) (l : Line) : Prop :=
  sorry

theorem circle_line_symmetry (c : Circle) (l : Line) :
  c.center.1 = -1 ∧ c.center.2 = 3 ∧ c.radius = 3 ∧
  l.a = 1 ∧ l.c = 4 ∧
  ∃ (p q : ℝ × ℝ), (p.1 + 1)^2 + (p.2 - 3)^2 = 9 ∧
                   (q.1 + 1)^2 + (q.2 - 3)^2 = 9 ∧
                   symmetric_points p q l →
  l.b = -1 :=
sorry

end circle_line_symmetry_l1909_190956


namespace student_miscalculation_l1909_190994

theorem student_miscalculation (a : ℤ) : 
  (-16 - a = -12) → (-16 + a = -20) := by
  sorry

end student_miscalculation_l1909_190994


namespace sqrt_27_div_3_eq_sqrt_3_l1909_190923

theorem sqrt_27_div_3_eq_sqrt_3 : Real.sqrt 27 / 3 = Real.sqrt 3 := by
  sorry

end sqrt_27_div_3_eq_sqrt_3_l1909_190923


namespace log_2_3_in_terms_of_a_b_l1909_190997

theorem log_2_3_in_terms_of_a_b (a b : ℝ) (ha : a = Real.log 6) (hb : b = Real.log 20) :
  Real.log 3 / Real.log 2 = (a - b + 1) / (b - 1) := by
  sorry

end log_2_3_in_terms_of_a_b_l1909_190997


namespace bouquet_composition_l1909_190948

/-- Represents a bouquet of branches -/
structure Bouquet :=
  (white : ℕ)
  (blue : ℕ)

/-- The conditions for our specific bouquet -/
def ValidBouquet (b : Bouquet) : Prop :=
  b.white + b.blue = 7 ∧
  b.white ≥ 1 ∧
  ∀ (x y : ℕ), x < y → x < 7 → y < 7 → (x = b.white → y = b.blue)

/-- The theorem to be proved -/
theorem bouquet_composition (b : Bouquet) (h : ValidBouquet b) : b.white = 1 ∧ b.blue = 6 := by
  sorry


end bouquet_composition_l1909_190948


namespace barons_claim_l1909_190934

/-- Define the type of weight sets -/
def WeightSet := Fin 1000 → ℕ

/-- The condition that all weights are different -/
def all_different (w : WeightSet) : Prop :=
  ∀ i j, i ≠ j → w i ≠ w j

/-- The sum of one of each weight -/
def sum_of_weights (w : WeightSet) : ℕ :=
  Finset.sum Finset.univ (λ i => w i)

/-- The uniqueness of the sum -/
def unique_sum (w : WeightSet) : Prop :=
  ∀ s : Finset (Fin 1000), s.card < 1000 → Finset.sum s (λ i => w i) ≠ sum_of_weights w

/-- The main theorem -/
theorem barons_claim :
  ∃ w : WeightSet,
    all_different w ∧
    sum_of_weights w < 2^1010 ∧
    unique_sum w :=
  sorry

end barons_claim_l1909_190934


namespace new_ratio_after_adding_water_l1909_190922

/-- Given a mixture of alcohol and water with an initial ratio and known quantities,
    this theorem proves the new ratio after adding water. -/
theorem new_ratio_after_adding_water
  (initial_alcohol : ℝ)
  (initial_water : ℝ)
  (added_water : ℝ)
  (h1 : initial_alcohol / initial_water = 4 / 3)
  (h2 : initial_alcohol = 20)
  (h3 : added_water = 4) :
  initial_alcohol / (initial_water + added_water) = 20 / 19 := by
  sorry

end new_ratio_after_adding_water_l1909_190922


namespace second_number_value_l1909_190977

theorem second_number_value (A B : ℝ) (h1 : A = 200) (h2 : 0.3 * A = 0.6 * B + 30) : B = 50 := by
  sorry

end second_number_value_l1909_190977


namespace sqrt_less_than_linear_approx_l1909_190998

theorem sqrt_less_than_linear_approx (x : ℝ) (hx : x > 0) : 
  Real.sqrt (1 + x) < 1 + x / 2 := by
sorry

end sqrt_less_than_linear_approx_l1909_190998


namespace triangle_heights_existence_l1909_190910

/-- Check if a triangle with given heights exists -/
def triangle_exists (h₁ h₂ h₃ : ℝ) : Prop :=
  ∃ a b c : ℝ, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    h₁ * a = h₂ * b ∧ h₂ * b = h₃ * c

theorem triangle_heights_existence :
  (¬ triangle_exists 2 3 6) ∧ (triangle_exists 2 3 5) := by
  sorry


end triangle_heights_existence_l1909_190910


namespace orange_price_l1909_190949

theorem orange_price (apple_price : ℚ) (total_fruit : ℕ) (initial_avg : ℚ) 
  (oranges_removed : ℕ) (final_avg : ℚ) :
  apple_price = 40 / 100 →
  total_fruit = 10 →
  initial_avg = 54 / 100 →
  oranges_removed = 4 →
  final_avg = 50 / 100 →
  ∃ (orange_price : ℚ),
    orange_price = 60 / 100 ∧
    ∃ (apples oranges : ℕ),
      apples + oranges = total_fruit ∧
      (apple_price * apples + orange_price * oranges) / total_fruit = initial_avg ∧
      (apple_price * apples + orange_price * (oranges - oranges_removed)) / 
        (total_fruit - oranges_removed) = final_avg :=
by
  sorry

end orange_price_l1909_190949


namespace sequence_existence_and_boundedness_l1909_190902

theorem sequence_existence_and_boundedness (a : ℝ) (n : ℕ) (hn : n > 0) :
  ∃! x : Fin (n + 2) → ℝ,
    (x 0 = 0 ∧ x (Fin.last n) = 0) ∧
    (∀ i : Fin (n + 1), i.val > 0 →
      (x i + x (i + 1)) / 2 = x i + (x i)^3 - a^3) ∧
    (∀ i : Fin (n + 2), |x i| ≤ |a|) := by
  sorry

end sequence_existence_and_boundedness_l1909_190902


namespace bread_waste_savings_l1909_190953

/-- Represents the daily bread waste and cost parameters -/
structure BreadWaste where
  pieces_per_day : ℕ
  pieces_per_loaf : ℕ
  cost_per_loaf : ℕ

/-- Calculates the money saved over a given number of days -/
def money_saved (waste : BreadWaste) (days : ℕ) : ℕ :=
  (days * waste.pieces_per_day * waste.cost_per_loaf) / (2 * waste.pieces_per_loaf)

/-- Theorem stating the money saved in 20 and 60 days -/
theorem bread_waste_savings (waste : BreadWaste) 
  (h1 : waste.pieces_per_day = 7)
  (h2 : waste.pieces_per_loaf = 14)
  (h3 : waste.cost_per_loaf = 35) :
  money_saved waste 20 = 350 ∧ money_saved waste 60 = 1050 := by
  sorry

#eval money_saved ⟨7, 14, 35⟩ 20
#eval money_saved ⟨7, 14, 35⟩ 60

end bread_waste_savings_l1909_190953


namespace geometric_sequence_common_ratio_l1909_190905

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 1)
  (h_a4a6 : a 4 * a 6 = 64) :
  ∃ q : ℝ, is_geometric_sequence a ∧ q = 2 := by
sorry

end geometric_sequence_common_ratio_l1909_190905


namespace cube_roll_no_90_degree_rotation_l1909_190987

/-- Represents a cube on a plane -/
structure Cube where
  position : ℝ × ℝ × ℝ
  top_face : Fin 6
  orientation : ℕ

/-- Represents a sequence of cube rolls -/
def RollSequence := List (Fin 4)

/-- Applies a sequence of rolls to a cube -/
def apply_rolls (c : Cube) (rolls : RollSequence) : Cube :=
  sorry

/-- Checks if the cube is in its initial position -/
def is_initial_position (initial : Cube) (final : Cube) : Prop :=
  initial.position = final.position ∧ initial.top_face = final.top_face

/-- Theorem: A cube rolled back to its initial position cannot have its top face rotated by 90 degrees -/
theorem cube_roll_no_90_degree_rotation 
  (c : Cube) (rolls : RollSequence) : 
  let c' := apply_rolls c rolls
  is_initial_position c c' → c.orientation ≠ (c'.orientation + 1) % 4 :=
sorry

end cube_roll_no_90_degree_rotation_l1909_190987


namespace sat_score_improvement_l1909_190983

theorem sat_score_improvement (first_score second_score : ℕ) 
  (h1 : first_score = 1000) 
  (h2 : second_score = 1100) : 
  (second_score - first_score) / first_score * 100 = 10 := by
  sorry

end sat_score_improvement_l1909_190983


namespace journey_time_ratio_l1909_190916

theorem journey_time_ratio (speed_to_sf : ℝ) (avg_speed : ℝ) :
  speed_to_sf = 48 →
  avg_speed = 32 →
  (1 / avg_speed - 1 / speed_to_sf) / (1 / speed_to_sf) = 3 / 2 :=
by sorry

end journey_time_ratio_l1909_190916


namespace snow_probability_in_week_l1909_190909

theorem snow_probability_in_week (p1 p2 : ℝ) : 
  p1 = 1/2 → p2 = 1/3 → 
  (1 - (1 - p1)^4 * (1 - p2)^3) = 53/54 := by
  sorry

end snow_probability_in_week_l1909_190909


namespace cost_of_450_candies_l1909_190947

/-- The cost of buying a given number of chocolate candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (total_candies : ℕ) : ℚ :=
  (total_candies / candies_per_box) * cost_per_box

/-- Theorem: The cost of 450 chocolate candies is $112.50 -/
theorem cost_of_450_candies :
  cost_of_candies 30 (7.5 : ℚ) 450 = (112.5 : ℚ) := by
  sorry

end cost_of_450_candies_l1909_190947


namespace min_perimeter_triangle_l1909_190978

theorem min_perimeter_triangle (a b : ℝ) (h1 : 0 < b) (h2 : b < a) :
  let min_perimeter := Real.sqrt (2 * (a^2 + b^2))
  ∀ c d : ℝ, (c ≥ 0) → (d ≥ 0) → 
    Real.sqrt ((a - c)^2 + b^2) + Real.sqrt ((d - c)^2 + d^2) + Real.sqrt ((a - d)^2 + (b - d)^2) ≥ min_perimeter :=
by sorry


end min_perimeter_triangle_l1909_190978


namespace min_xy_value_l1909_190945

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (x + 1) * (y + 1) = 2 * x + 2 * y + 4) : 
  x * y ≥ 9 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x + 1) * (y + 1) = 2 * x + 2 * y + 4 ∧ x * y = 9 := by
  sorry

end min_xy_value_l1909_190945


namespace sufficient_not_necessary_l1909_190926

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a < -1 → a^2 - 5*a - 6 > 0) ∧
  (∃ a, a^2 - 5*a - 6 > 0 ∧ a ≥ -1) :=
by sorry

end sufficient_not_necessary_l1909_190926


namespace solve_dining_problem_l1909_190913

def dining_problem (total_bill : ℚ) (tip_percentage : ℚ) (individual_share : ℚ) : Prop :=
  let tip := total_bill * tip_percentage
  let total_with_tip := total_bill + tip
  let num_people := total_with_tip / individual_share
  num_people = 5

theorem solve_dining_problem :
  dining_problem 139 (1/10) (3058/100) := by
  sorry

end solve_dining_problem_l1909_190913


namespace r_squared_perfect_fit_l1909_190920

/-- Linear regression model with zero error -/
structure LinearRegressionModel where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  a : ℝ
  b : ℝ
  h : ∀ i, y i = b * x i + a

/-- Coefficient of determination (R-squared) -/
def r_squared (model : LinearRegressionModel) : ℝ :=
  sorry

/-- Theorem: R-squared equals 1 for a perfect fit linear regression model -/
theorem r_squared_perfect_fit (model : LinearRegressionModel) :
  r_squared model = 1 :=
sorry

end r_squared_perfect_fit_l1909_190920


namespace division_remainder_l1909_190963

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 181 → 
  divisor = 20 → 
  quotient = 9 → 
  remainder = dividend - (divisor * quotient) → 
  remainder = 1 := by
  sorry

end division_remainder_l1909_190963


namespace infinite_squares_in_progression_l1909_190974

/-- An arithmetic progression with positive integer members -/
structure ArithmeticProgression where
  a : ℕ+  -- First term
  d : ℕ+  -- Common difference

/-- Predicate to check if a number is in the arithmetic progression -/
def inProgression (ap : ArithmeticProgression) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = ap.a + k * ap.d

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem infinite_squares_in_progression (ap : ArithmeticProgression) :
  (∃ n : ℕ, inProgression ap n ∧ isPerfectSquare n) →
  (∀ N : ℕ, ∃ n : ℕ, n > N ∧ inProgression ap n ∧ isPerfectSquare n) :=
sorry

end infinite_squares_in_progression_l1909_190974


namespace restaurant_friends_l1909_190903

theorem restaurant_friends (initial_wings : ℕ) (cooked_wings : ℕ) (wings_per_person : ℕ) : 
  initial_wings = 9 →
  cooked_wings = 7 →
  wings_per_person = 4 →
  (initial_wings + cooked_wings) / wings_per_person = 4 := by
  sorry

end restaurant_friends_l1909_190903


namespace product_equals_32_l1909_190986

theorem product_equals_32 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 := by
  sorry

end product_equals_32_l1909_190986


namespace range_of_S_l1909_190939

theorem range_of_S (a b : ℝ) 
  (h : ∀ x ∈ Set.Icc 0 1, |a * x + b| ≤ 1) : 
  ∃ S : ℝ, S = (a + 1) * (b + 1) ∧ -2 ≤ S ∧ S ≤ 9/4 :=
sorry

end range_of_S_l1909_190939


namespace A_infinite_l1909_190929

/-- A function that represents z = n^4 + a -/
def z (n a : ℕ) : ℕ := n^4 + a

/-- The set of natural numbers a such that z(n, a) is composite for all n -/
def A : Set ℕ := {a : ℕ | ∀ n : ℕ, ¬ Nat.Prime (z n a)}

/-- Theorem stating that A is infinite -/
theorem A_infinite : Set.Infinite A := by sorry

end A_infinite_l1909_190929


namespace equation_transformation_l1909_190936

theorem equation_transformation (x y : ℝ) : x - 2 = y - 2 → x = y := by
  sorry

end equation_transformation_l1909_190936


namespace rectangle_area_l1909_190995

-- Define the rectangle
structure Rectangle where
  breadth : ℝ
  length : ℝ
  diagonal : ℝ

-- Define the conditions
def rectangleConditions (r : Rectangle) : Prop :=
  r.length = 3 * r.breadth ∧ r.diagonal = 20

-- Define the area function
def area (r : Rectangle) : ℝ :=
  r.length * r.breadth

-- Theorem statement
theorem rectangle_area (r : Rectangle) (h : rectangleConditions r) : area r = 120 := by
  sorry

end rectangle_area_l1909_190995


namespace waiter_tips_fraction_l1909_190971

/-- Given a waiter's salary and tips, where the tips are 7/4 of the salary,
    prove that the fraction of total income from tips is 7/11. -/
theorem waiter_tips_fraction (salary : ℚ) (tips : ℚ) (total_income : ℚ) : 
  tips = (7 : ℚ) / 4 * salary →
  total_income = salary + tips →
  tips / total_income = (7 : ℚ) / 11 := by
  sorry

end waiter_tips_fraction_l1909_190971


namespace sum_of_values_equals_three_l1909_190996

/-- A discrete random variable with two possible values -/
structure DiscreteRV (α : Type) where
  value : α
  prob : α → ℝ

/-- The expectation of a discrete random variable -/
def expectation {α : Type} (X : DiscreteRV α) : ℝ :=
  sorry

/-- The variance of a discrete random variable -/
def variance {α : Type} (X : DiscreteRV α) : ℝ :=
  sorry

theorem sum_of_values_equals_three
  (ξ : DiscreteRV ℝ)
  (a b : ℝ)
  (h_prob_a : ξ.prob a = 2/3)
  (h_prob_b : ξ.prob b = 1/3)
  (h_lt : a < b)
  (h_expect : expectation ξ = 4/3)
  (h_var : variance ξ = 2/9) :
  a + b = 3 :=
sorry

end sum_of_values_equals_three_l1909_190996


namespace max_value_of_expression_max_value_achievable_l1909_190921

theorem max_value_of_expression (x : ℝ) :
  x^6 / (x^12 + 4*x^9 - 6*x^6 + 16*x^3 + 64) ≤ 1/26 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, x^6 / (x^12 + 4*x^9 - 6*x^6 + 16*x^3 + 64) = 1/26 :=
by sorry

end max_value_of_expression_max_value_achievable_l1909_190921


namespace ducks_drinking_order_l1909_190958

theorem ducks_drinking_order (total_ducks : ℕ) (ducks_before_a : ℕ) (ducks_after_a : ℕ) :
  total_ducks = 20 →
  ducks_before_a = 11 →
  ducks_after_a = total_ducks - (ducks_before_a + 1) →
  ducks_after_a = 8 :=
by sorry

end ducks_drinking_order_l1909_190958


namespace side_c_value_l1909_190965

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions for a valid triangle if necessary
  true

-- State the theorem
theorem side_c_value 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_b : b = 3)
  (h_a : a = Real.sqrt 3)
  (h_A : A = 30 * π / 180) -- Convert 30° to radians
  : c = 2 * Real.sqrt 3 := by
  sorry

end side_c_value_l1909_190965


namespace rectangle_area_diagonal_l1909_190931

/-- Theorem: Area of a rectangle with length-to-width ratio 3:2 and diagonal d --/
theorem rectangle_area_diagonal (length width diagonal : ℝ) 
  (h_ratio : length / width = 3 / 2)
  (h_diagonal : length^2 + width^2 = diagonal^2) :
  length * width = (6/13) * diagonal^2 := by
  sorry

end rectangle_area_diagonal_l1909_190931


namespace min_value_of_product_l1909_190975

-- Define the quadratic function f
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem min_value_of_product (a b c : ℝ) (x₁ x₂ x₃ : ℝ) :
  a ≠ 0 →
  f a b c (-1) = 0 →
  (∀ x : ℝ, f a b c x ≥ x) →
  (∀ x : ℝ, 0 < x → x < 2 → f a b c x ≤ (x + 1)^2 / 4) →
  0 < x₁ → x₁ < 2 →
  0 < x₂ → x₂ < 2 →
  0 < x₃ → x₃ < 2 →
  1 / x₁ + 1 / x₂ + 1 / x₃ = 3 →
  ∃ (m : ℝ), m = 1 ∧ ∀ y₁ y₂ y₃ : ℝ,
    0 < y₁ → y₁ < 2 →
    0 < y₂ → y₂ < 2 →
    0 < y₃ → y₃ < 2 →
    1 / y₁ + 1 / y₂ + 1 / y₃ = 3 →
    m ≤ f a b c y₁ * f a b c y₂ * f a b c y₃ :=
by
  sorry

end min_value_of_product_l1909_190975


namespace complex_fraction_equals_two_l1909_190940

theorem complex_fraction_equals_two (z : ℂ) (h : z = 1 - I) : z^2 / (z - 1) = 2 := by
  sorry

end complex_fraction_equals_two_l1909_190940


namespace evaluate_expression_l1909_190912

theorem evaluate_expression : (1 / ((-7^3)^3)) * ((-7)^10) = -7 := by
  sorry

end evaluate_expression_l1909_190912


namespace sqrt_product_equality_l1909_190943

theorem sqrt_product_equality : (Real.sqrt 12 + 2) * (Real.sqrt 3 - 1) = 4 := by
  sorry

end sqrt_product_equality_l1909_190943


namespace circle_to_hyperbola_l1909_190908

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 5*y + 4 = 0

-- Define the intersection points of circle C with coordinate axes
def intersection_points (C : (ℝ → ℝ → Prop)) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), (x = 0 ∨ y = 0) ∧ C x y ∧ p = (x, y)}

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := (y - 1)^2 / 1 - x^2 / 15 = 1

-- Theorem statement
theorem circle_to_hyperbola :
  ∀ (focus vertex : ℝ × ℝ),
    focus ∈ intersection_points circle_C →
    vertex ∈ intersection_points circle_C →
    focus ≠ vertex →
    (∀ x y : ℝ, hyperbola_equation x y ↔
      ∃ (a b : ℝ),
        a > 0 ∧ b > 0 ∧
        (y - vertex.2)^2 / a^2 - (x - vertex.1)^2 / b^2 = 1 ∧
        (focus.1 - vertex.1)^2 + (focus.2 - vertex.2)^2 = a^2 + b^2) :=
sorry

end circle_to_hyperbola_l1909_190908


namespace frank_weed_eating_earnings_l1909_190982

/-- Calculates the amount Frank made weed eating given his lawn mowing earnings, weekly spending, and duration of savings. -/
def weed_eating_earnings (lawn_mowing_earnings weekly_spending duration_weeks : ℕ) : ℕ :=
  weekly_spending * duration_weeks - lawn_mowing_earnings

theorem frank_weed_eating_earnings :
  weed_eating_earnings 5 7 9 = 58 := by
  sorry

end frank_weed_eating_earnings_l1909_190982


namespace matrix_identities_l1909_190952

variable {n : ℕ} (hn : n ≥ 2)
variable (k : ℝ)
variable (A B C D : Matrix (Fin n) (Fin n) ℂ)

theorem matrix_identities 
  (h1 : A * C + k • (B * D) = 1)
  (h2 : A * D = B * C) :
  C * A + k • (D * B) = 1 ∧ D * A = C * B := by
sorry

end matrix_identities_l1909_190952


namespace area_of_triangle_DBG_l1909_190966

-- Define the triangle and squares
structure RightTriangle :=
  (A B C : ℝ × ℝ)
  (is_right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0)

def square_area (side : ℝ) : ℝ := side ^ 2

-- State the theorem
theorem area_of_triangle_DBG 
  (triangle : RightTriangle)
  (area_ABDE : square_area (Real.sqrt ((triangle.A.1 - triangle.B.1)^2 + (triangle.A.2 - triangle.B.2)^2)) = 8)
  (area_BCFG : square_area (Real.sqrt ((triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2)) = 26) :
  let D : ℝ × ℝ := (triangle.A.1 + (triangle.B.2 - triangle.A.2), triangle.A.2 - (triangle.B.1 - triangle.A.1))
  let G : ℝ × ℝ := (triangle.B.1 + (triangle.C.2 - triangle.B.2), triangle.B.2 - (triangle.C.1 - triangle.B.1))
  (1/2) * Real.sqrt ((D.1 - triangle.B.1)^2 + (D.2 - triangle.B.2)^2) * 
         Real.sqrt ((G.1 - triangle.B.1)^2 + (G.2 - triangle.B.2)^2) = 2 * Real.sqrt 13 :=
by sorry

end area_of_triangle_DBG_l1909_190966


namespace stating_regular_duck_price_is_correct_l1909_190985

/-- The price of a regular size rubber duck in the city's charity race. -/
def regular_duck_price : ℚ :=
  3

/-- The price of a large size rubber duck in the city's charity race. -/
def large_duck_price : ℚ :=
  5

/-- The number of regular size ducks sold in the charity race. -/
def regular_ducks_sold : ℕ :=
  221

/-- The number of large size ducks sold in the charity race. -/
def large_ducks_sold : ℕ :=
  185

/-- The total amount raised in the charity race. -/
def total_raised : ℚ :=
  1588

/-- 
Theorem stating that the regular duck price is correct given the conditions of the charity race.
-/
theorem regular_duck_price_is_correct :
  regular_duck_price * regular_ducks_sold + large_duck_price * large_ducks_sold = total_raised :=
by sorry

end stating_regular_duck_price_is_correct_l1909_190985


namespace fraction_subtraction_l1909_190962

theorem fraction_subtraction : 
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12 := by
  sorry

end fraction_subtraction_l1909_190962


namespace smallest_right_triangle_area_l1909_190964

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∀ (c : ℝ), c > 0 → a^2 + b^2 = c^2 → (1/2) * a * b ≤ 24 :=
by sorry

end smallest_right_triangle_area_l1909_190964


namespace find_t_value_l1909_190989

theorem find_t_value (s t : ℚ) 
  (eq1 : 12 * s + 7 * t = 154)
  (eq2 : s = 2 * t - 3) : 
  t = 190 / 31 := by
sorry

end find_t_value_l1909_190989


namespace unit_vector_parallel_to_a_unit_vector_perpendicular_to_a_rotated_vector_e_l1909_190900

-- Define the vector a
def a : ℝ × ℝ := (3, -4)

-- Theorem for the unit vector b parallel to a
theorem unit_vector_parallel_to_a :
  ∃ b : ℝ × ℝ, (b.1 = 3/5 ∧ b.2 = -4/5) ∨ (b.1 = -3/5 ∧ b.2 = 4/5) ∧
  (b.1 * a.1 + b.2 * a.2)^2 = (b.1^2 + b.2^2) * (a.1^2 + a.2^2) ∧
  b.1^2 + b.2^2 = 1 :=
sorry

-- Theorem for the unit vector c perpendicular to a
theorem unit_vector_perpendicular_to_a :
  ∃ c : ℝ × ℝ, (c.1 = 4/5 ∧ c.2 = 3/5) ∨ (c.1 = -4/5 ∧ c.2 = -3/5) ∧
  c.1 * a.1 + c.2 * a.2 = 0 ∧
  c.1^2 + c.2^2 = 1 :=
sorry

-- Theorem for the vector e obtained by rotating a 45° counterclockwise
theorem rotated_vector_e :
  ∃ e : ℝ × ℝ, e.1 = 7 * Real.sqrt 2 / 2 ∧ e.2 = - Real.sqrt 2 / 2 ∧
  e.1^2 + e.2^2 = a.1^2 + a.2^2 ∧
  e.1 * a.1 + e.2 * a.2 = Real.sqrt ((a.1^2 + a.2^2)^2 / 2) :=
sorry

end unit_vector_parallel_to_a_unit_vector_perpendicular_to_a_rotated_vector_e_l1909_190900


namespace stratified_sampling_medium_supermarkets_l1909_190993

theorem stratified_sampling_medium_supermarkets :
  let total_supermarkets : ℕ := 200 + 400 + 1400
  let medium_supermarkets : ℕ := 400
  let sample_size : ℕ := 100
  (medium_supermarkets * sample_size) / total_supermarkets = 20 := by
  sorry

end stratified_sampling_medium_supermarkets_l1909_190993


namespace gcd_1908_4187_l1909_190941

theorem gcd_1908_4187 : Nat.gcd 1908 4187 = 53 := by
  sorry

end gcd_1908_4187_l1909_190941


namespace prisoner_release_time_l1909_190959

def prisoner_age : ℕ := 25
def warden_age : ℕ := 54

theorem prisoner_release_time : 
  ∃ (years : ℕ), warden_age + years = 2 * (prisoner_age + years) ∧ years = 4 :=
by sorry

end prisoner_release_time_l1909_190959


namespace smallest_class_size_l1909_190979

/-- Represents the number of students in a physical education class. -/
def class_size (x : ℕ) : ℕ := 5 * x + 3

/-- Theorem stating the smallest possible class size satisfying the given conditions. -/
theorem smallest_class_size :
  ∀ n : ℕ, class_size n > 50 → class_size 10 ≤ class_size n :=
by
  sorry

#eval class_size 10  -- Should output 53

end smallest_class_size_l1909_190979


namespace benny_pumpkin_pies_l1909_190990

/-- Represents the number of pumpkin pies Benny plans to make -/
def num_pumpkin_pies : ℕ := sorry

/-- The cost to make one pumpkin pie -/
def pumpkin_pie_cost : ℕ := 3

/-- The cost to make one cherry pie -/
def cherry_pie_cost : ℕ := 5

/-- The number of cherry pies Benny plans to make -/
def num_cherry_pies : ℕ := 12

/-- The profit Benny wants to make -/
def desired_profit : ℕ := 20

/-- The price Benny charges for each pie -/
def pie_price : ℕ := 5

/-- Theorem stating that the number of pumpkin pies Benny plans to make is 10 -/
theorem benny_pumpkin_pies : 
  num_pumpkin_pies = 10 := by
  sorry

end benny_pumpkin_pies_l1909_190990


namespace min_value_of_expression_lower_bound_achievable_l1909_190957

theorem min_value_of_expression (x : ℝ) : 
  (x + 1)^2 * (x + 2)^2 * (x + 3)^2 * (x + 4)^2 + 2025 ≥ 3625 :=
by sorry

theorem lower_bound_achievable : 
  ∃ x : ℝ, (x + 1)^2 * (x + 2)^2 * (x + 3)^2 * (x + 4)^2 + 2025 = 3625 :=
by sorry

end min_value_of_expression_lower_bound_achievable_l1909_190957


namespace parabola_equation_l1909_190973

/-- A parabola with focus F and point A on the curve, where |FA| is the radius of a circle
    intersecting the parabola's axis at B and C, forming an equilateral triangle FBC. -/
structure ParabolaWithTriangle where
  -- The parameter of the parabola
  p : ℝ
  -- The coordinates of points A, B, C, and F
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  F : ℝ × ℝ

/-- Properties of the parabola and associated triangle -/
def ParabolaProperties (P : ParabolaWithTriangle) : Prop :=
  -- A lies on the parabola y^2 = 2px
  P.A.2^2 = 2 * P.p * P.A.1 ∧
  -- F is the focus (p/2, 0)
  P.F = (P.p/2, 0) ∧
  -- B and C are on the x-axis
  P.B.2 = 0 ∧ P.C.2 = 0 ∧
  -- |FA| = |FB| = |FC|
  (P.A.1 - P.F.1)^2 + (P.A.2 - P.F.2)^2 = 
  (P.B.1 - P.F.1)^2 + (P.B.2 - P.F.2)^2 ∧
  (P.A.1 - P.F.1)^2 + (P.A.2 - P.F.2)^2 = 
  (P.C.1 - P.F.1)^2 + (P.C.2 - P.F.2)^2 ∧
  -- Area of triangle ABC is 128/3
  abs ((P.A.1 - P.C.1) * (P.B.2 - P.C.2) - (P.B.1 - P.C.1) * (P.A.2 - P.C.2)) / 2 = 128/3

theorem parabola_equation (P : ParabolaWithTriangle) 
  (h : ParabolaProperties P) : P.p = 8 ∧ ∀ (x y : ℝ), y^2 = 16*x ↔ y^2 = 2*P.p*x := by
  sorry

end parabola_equation_l1909_190973


namespace unique_number_l1909_190968

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n % 10 = 2 ∧
  200 + (n / 10) = n + 18

theorem unique_number : ∃! n : ℕ, is_valid_number n ∧ n = 202 :=
sorry

end unique_number_l1909_190968


namespace jeanine_pencils_proof_l1909_190960

/-- The number of pencils Jeanine bought initially -/
def jeanine_pencils : ℕ := 18

/-- The number of pencils Clare bought -/
def clare_pencils : ℕ := jeanine_pencils / 2

/-- The number of pencils Jeanine has after giving some to Abby -/
def jeanine_remaining_pencils : ℕ := (2 * jeanine_pencils) / 3

theorem jeanine_pencils_proof :
  (clare_pencils = jeanine_pencils / 2) ∧
  (jeanine_remaining_pencils = (2 * jeanine_pencils) / 3) ∧
  (jeanine_remaining_pencils = clare_pencils + 3) →
  jeanine_pencils = 18 := by
sorry

end jeanine_pencils_proof_l1909_190960


namespace cupcake_distribution_l1909_190950

/-- Represents the number of cupcakes in a pack --/
inductive PackSize
  | five : PackSize
  | ten : PackSize
  | fifteen : PackSize
  | twenty : PackSize

/-- Returns the number of cupcakes in a pack --/
def packSizeToInt (p : PackSize) : Nat :=
  match p with
  | PackSize.five => 5
  | PackSize.ten => 10
  | PackSize.fifteen => 15
  | PackSize.twenty => 20

/-- Calculates the total number of cupcakes from a given number of packs --/
def totalCupcakes (packSize : PackSize) (numPacks : Nat) : Nat :=
  (packSizeToInt packSize) * numPacks

/-- Represents Jean's initial purchase --/
def initialPurchase : Nat :=
  totalCupcakes PackSize.fifteen 4 + totalCupcakes PackSize.twenty 2

/-- The number of children in the orphanage --/
def numChildren : Nat := 220

/-- The theorem to prove --/
theorem cupcake_distribution :
  totalCupcakes PackSize.ten 8 + totalCupcakes PackSize.five 8 + initialPurchase = numChildren := by
  sorry

end cupcake_distribution_l1909_190950


namespace division_remainder_problem_l1909_190980

theorem division_remainder_problem (L S R : ℕ) : 
  L - S = 1365 → 
  L = 1631 → 
  L = 6 * S + R → 
  R = 35 := by
sorry

end division_remainder_problem_l1909_190980


namespace alices_favorite_number_l1909_190970

def is_favorite_number (n : ℕ) : Prop :=
  50 ≤ n ∧ n ≤ 100 ∧ 
  n % 11 = 0 ∧
  n % 2 ≠ 0 ∧
  (n / 10 + n % 10) % 5 = 0

theorem alices_favorite_number :
  ∃! n : ℕ, is_favorite_number n ∧ n = 55 := by
sorry

end alices_favorite_number_l1909_190970


namespace puzzle_solution_l1909_190927

-- Define the types of beings
inductive Being
| Human
| Monkey

-- Define the types of speakers
inductive Speaker
| Knight
| Liar

-- Define A and B as individuals
structure Individual where
  being : Being
  speaker : Speaker

-- Define the statements made by A and B
def statement_A (a b : Individual) : Prop :=
  a.being = Being.Monkey ∨ b.being = Being.Monkey

def statement_B (a b : Individual) : Prop :=
  a.speaker = Speaker.Liar ∨ b.speaker = Speaker.Liar

-- Theorem stating the conclusion
theorem puzzle_solution :
  ∃ (a b : Individual),
    (statement_A a b ↔ a.speaker = Speaker.Liar) ∧
    (statement_B a b ↔ b.speaker = Speaker.Knight) ∧
    a.being = Being.Human ∧
    b.being = Being.Human ∧
    a.speaker = Speaker.Liar ∧
    b.speaker = Speaker.Knight :=
sorry

end puzzle_solution_l1909_190927


namespace work_completion_time_l1909_190914

-- Define the work capacity ratio of A to B
def ratio_A_to_B : ℚ := 3 / 2

-- Define the time A takes to complete the work alone
def time_A_alone : ℕ := 45

-- Define the function to calculate the time taken when A and B work together
def time_together (ratio : ℚ) (time_alone : ℕ) : ℚ :=
  (ratio * time_alone) / (ratio + 1)

-- Theorem statement
theorem work_completion_time :
  time_together ratio_A_to_B time_A_alone = 27 := by
  sorry

end work_completion_time_l1909_190914
