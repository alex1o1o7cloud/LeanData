import Mathlib

namespace number_line_position_l2748_274826

theorem number_line_position : 
  ∀ (total_distance : ℝ) (num_steps : ℕ) (step_number : ℕ),
    total_distance = 32 →
    num_steps = 8 →
    step_number = 5 →
    (step_number : ℝ) * (total_distance / num_steps) = 20 :=
by sorry

end number_line_position_l2748_274826


namespace recorder_price_problem_l2748_274874

theorem recorder_price_problem (a b : ℕ) : 
  a < 10 → b < 10 →  -- Ensure a and b are single digits
  10 * b + a < 50 →  -- Old price less than 50
  (10 * a + b : ℚ) = 1.2 * (10 * b + a) →  -- 20% price increase
  a = 5 ∧ b = 4 := by sorry

end recorder_price_problem_l2748_274874


namespace price_decrease_l2748_274863

/-- The price of 6 packets last month in dollars -/
def last_month_price : ℚ := 7.5

/-- The number of packets in last month's offer -/
def last_month_packets : ℕ := 6

/-- The price of 10 packets this month in dollars -/
def this_month_price : ℚ := 11

/-- The number of packets in this month's offer -/
def this_month_packets : ℕ := 10

/-- The percent decrease in price per packet -/
def percent_decrease : ℚ := 12

theorem price_decrease :
  (last_month_price / last_month_packets - this_month_price / this_month_packets) /
  (last_month_price / last_month_packets) * 100 = percent_decrease := by
  sorry


end price_decrease_l2748_274863


namespace trapezoid_isosceles_and_diagonal_l2748_274806

-- Define the trapezoid
structure Trapezoid :=
  (AB CD AD BC : ℝ)
  (parallel : AB ≠ CD)
  (ab_length : AB = 25)
  (cd_length : CD = 13)
  (ad_length : AD = 15)
  (bc_length : BC = 17)

-- Define an isosceles trapezoid
def IsoscelesTrapezoid (t : Trapezoid) : Prop :=
  ∃ (h : ℝ), h > 0 ∧ 
  (t.AD)^2 = h^2 + ((t.AB - t.CD) / 2)^2 ∧
  (t.BC)^2 = h^2 + ((t.AB - t.CD) / 2)^2

-- State the theorem
theorem trapezoid_isosceles_and_diagonal (t : Trapezoid) : 
  IsoscelesTrapezoid t ∧ ∃ (AC : ℝ), AC = Real.sqrt 524 :=
by sorry

end trapezoid_isosceles_and_diagonal_l2748_274806


namespace triple_transformation_to_zero_l2748_274849

/-- Represents a transformation on a triple of integers -/
inductive Transform : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → Prop
  | xy : ∀ x y z, x ≤ y → y ≤ z → Transform (x, y, z) (min (2*x) (y-x), max (2*x) (y-x), z)
  | xz : ∀ x y z, x ≤ y → y ≤ z → Transform (x, y, z) (min (2*x) (z-x), y, max (2*x) (z-x))
  | yz : ∀ x y z, x ≤ y → y ≤ z → Transform (x, y, z) (x, min (2*y) (z-y), max (2*y) (z-y))

/-- Represents a sequence of transformations -/
def TransformSeq : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → Prop :=
  Relation.ReflTransGen Transform

/-- The main theorem to be proved -/
theorem triple_transformation_to_zero :
  ∀ x y z : ℕ, x ≤ y → y ≤ z → ∃ a b c : ℕ, TransformSeq (x, y, z) (a, b, c) ∧ (a = 0 ∨ b = 0 ∨ c = 0) :=
sorry

end triple_transformation_to_zero_l2748_274849


namespace odd_area_rectangles_count_l2748_274850

/-- Represents a 3x3 grid of rectangles with integer side lengths -/
structure Grid :=
  (horizontal_lengths : Fin 4 → ℕ)
  (vertical_lengths : Fin 4 → ℕ)

/-- Counts the number of rectangles with odd area in the grid -/
def count_odd_area_rectangles (g : Grid) : ℕ :=
  sorry

/-- Theorem stating that the number of rectangles with odd area is either 0 or 4 -/
theorem odd_area_rectangles_count (g : Grid) : 
  count_odd_area_rectangles g = 0 ∨ count_odd_area_rectangles g = 4 :=
sorry

end odd_area_rectangles_count_l2748_274850


namespace rectangular_prism_volume_l2748_274839

-- Define the rectangular prism
structure RectangularPrism :=
  (height : ℝ)
  (base_length : ℝ)
  (base_width : ℝ)

-- Define the slant representation
structure SlantRepresentation :=
  (angle : ℝ)
  (long_side : ℝ)
  (short_side : ℝ)

-- Define the theorem
theorem rectangular_prism_volume
  (prism : RectangularPrism)
  (slant : SlantRepresentation)
  (h1 : prism.height = 1)
  (h2 : slant.angle = 45)
  (h3 : slant.long_side = 2)
  (h4 : slant.long_side = 2 * slant.short_side)
  (h5 : prism.base_length = slant.long_side)
  (h6 : prism.base_width = slant.long_side) :
  prism.height * prism.base_length * prism.base_width = 4 :=
by sorry

end rectangular_prism_volume_l2748_274839


namespace real_roots_of_polynomial_l2748_274840

/-- The polynomial function we're considering -/
def f (x : ℝ) : ℝ := x^4 - 4*x^3 + 5*x^2 + 2*x - 8

/-- Theorem stating that 1 + √3 and 1 - √3 are the real roots of the polynomial -/
theorem real_roots_of_polynomial :
  (∃ (x : ℝ), f x = 0) ↔ (∃ (x : ℝ), x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) :=
sorry

end real_roots_of_polynomial_l2748_274840


namespace complex_on_imaginary_axis_l2748_274809

theorem complex_on_imaginary_axis (a b : ℝ) : 
  (Complex.I * Complex.I = -1) →
  (∃ (y : ℝ), (a + Complex.I) / (b - 3 * Complex.I) = Complex.I * y) →
  a * b = 3 := by
  sorry

end complex_on_imaginary_axis_l2748_274809


namespace notebook_cost_l2748_274832

theorem notebook_cost (initial_money : ℕ) (notebooks_bought : ℕ) (books_bought : ℕ) 
  (book_cost : ℕ) (money_left : ℕ) :
  initial_money = 56 →
  notebooks_bought = 7 →
  books_bought = 2 →
  book_cost = 7 →
  money_left = 14 →
  ∃ (notebook_cost : ℕ), 
    notebook_cost * notebooks_bought + book_cost * books_bought = initial_money - money_left ∧
    notebook_cost = 4 := by
  sorry

end notebook_cost_l2748_274832


namespace prescription_duration_l2748_274872

/-- The number of days a prescription lasts -/
def prescription_days : ℕ := 30

/-- The daily dose in pills -/
def daily_dose : ℕ := 2

/-- The number of pills remaining after 4/5 of the days -/
def remaining_pills : ℕ := 12

/-- Theorem stating that the prescription lasts for 30 days -/
theorem prescription_duration :
  prescription_days = 30 ∧
  remaining_pills = (1/5 : ℚ) * (prescription_days * daily_dose) :=
by sorry

end prescription_duration_l2748_274872


namespace is_circle_center_l2748_274870

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y - 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -1)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center : 
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 6 :=
by sorry

end is_circle_center_l2748_274870


namespace purple_ring_weight_l2748_274847

/-- The weight of the purple ring given the weights of other rings and the total weight -/
theorem purple_ring_weight 
  (orange_weight : ℝ) 
  (white_weight : ℝ) 
  (total_weight : ℝ) 
  (h_orange : orange_weight = 0.08333333333333333)
  (h_white : white_weight = 0.4166666666666667)
  (h_total : total_weight = 0.8333333333333334) :
  total_weight - (orange_weight + white_weight) = 0.3333333333333334 := by
sorry

end purple_ring_weight_l2748_274847


namespace decreasing_before_vertex_l2748_274899

/-- The quadratic function f(x) = (x - 4)² + 3 -/
def f (x : ℝ) : ℝ := (x - 4)^2 + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * (x - 4)

theorem decreasing_before_vertex :
  ∀ x : ℝ, x < 4 → f' x < 0 :=
by
  sorry

end decreasing_before_vertex_l2748_274899


namespace isosceles_triangle_perimeter_l2748_274898

/-- An isosceles triangle with side lengths 3 and 8 has a perimeter of 19. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  (a = b ∧ (c = 3 ∨ c = 8)) ∨ (a = c ∧ (b = 3 ∨ b = 8)) ∨ (b = c ∧ (a = 3 ∨ a = 8)) →
  a + b > c → b + c > a → a + c > b →
  a + b + c = 19 :=
by
  sorry

end isosceles_triangle_perimeter_l2748_274898


namespace product_of_eight_consecutive_integers_divisible_by_ten_l2748_274862

theorem product_of_eight_consecutive_integers_divisible_by_ten (n : ℕ) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7)) = 10 * k := by
  sorry

end product_of_eight_consecutive_integers_divisible_by_ten_l2748_274862


namespace parabola_equation_theorem_l2748_274876

/-- A parabola with vertex at the origin and symmetric about the y-axis. -/
structure Parabola where
  /-- The parameter of the parabola, which is half the length of the chord passing through the focus and perpendicular to the axis of symmetry. -/
  p : ℝ
  /-- The chord passing through the focus and perpendicular to the y-axis has length 16. -/
  chord_length : p * 2 = 16

/-- The equation of a parabola with vertex at the origin and symmetric about the y-axis. -/
def parabola_equation (par : Parabola) : Prop :=
  ∀ x y : ℝ, (x^2 = 4 * par.p * y) ∨ (x^2 = -4 * par.p * y)

/-- Theorem stating that the equation of the parabola is either x^2 = 32y or x^2 = -32y. -/
theorem parabola_equation_theorem (par : Parabola) : parabola_equation par :=
  sorry

end parabola_equation_theorem_l2748_274876


namespace money_conditions_l2748_274884

theorem money_conditions (a b : ℝ) 
  (h1 : 6 * a - b = 45)
  (h2 : 4 * a + b > 60) : 
  a > 10.5 ∧ b > 18 := by
sorry

end money_conditions_l2748_274884


namespace y_variation_l2748_274802

/-- A function that varies directly as x and inversely as the square of z -/
def y (x z : ℝ) : ℝ := sorry

theorem y_variation (k : ℝ) :
  (∀ x z, y x z = k * x / (z^2)) →
  y 5 1 = 10 →
  y (-10) 2 = -5 := by sorry

end y_variation_l2748_274802


namespace max_sum_arithmetic_sequence_l2748_274810

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

theorem max_sum_arithmetic_sequence :
  let a₁ : ℚ := 5
  let d : ℚ := -5/7
  let S : ℕ → ℚ := λ n => sum_arithmetic_sequence a₁ d n
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧
    (∀ m : ℕ, S m ≤ S n) ∧
    S n = 1075/14 :=
by sorry

end max_sum_arithmetic_sequence_l2748_274810


namespace horner_rule_v4_l2748_274853

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 79
  v3 * x - 8

theorem horner_rule_v4 :
  horner_v4 (-4) = 220 :=
by sorry

end horner_rule_v4_l2748_274853


namespace gain_percent_calculation_l2748_274891

/-- Calculates the gain percent from a purchase, repair, and sale of an item. -/
theorem gain_percent_calculation 
  (purchase_price : ℝ) 
  (repair_costs : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 800)
  (h2 : repair_costs = 200)
  (h3 : selling_price = 1400) : 
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 40 := by
  sorry

#check gain_percent_calculation

end gain_percent_calculation_l2748_274891


namespace total_cubes_after_distribution_l2748_274888

/-- Represents the number of cubes a person has -/
structure CubeCount where
  red : ℕ
  blue : ℕ

def Grady : CubeCount := { red := 20, blue := 15 }
def Gage : CubeCount := { red := 10, blue := 12 }
def Harper : CubeCount := { red := 8, blue := 10 }

def giveToGage (c : CubeCount) : CubeCount :=
  { red := c.red * 2 / 5, blue := c.blue / 3 }

def remainingAfterGage (initial : CubeCount) (given : CubeCount) : CubeCount :=
  { red := initial.red - given.red, blue := initial.blue - given.blue }

def giveToHarper (c : CubeCount) : CubeCount :=
  { red := c.red / 4, blue := c.blue / 2 }

def totalCubes (c : CubeCount) : ℕ := c.red + c.blue

theorem total_cubes_after_distribution :
  let gageGiven := giveToGage Grady
  let harperGiven := giveToHarper (remainingAfterGage Grady gageGiven)
  let finalGage := { red := Gage.red + gageGiven.red, blue := Gage.blue + gageGiven.blue }
  let finalHarper := { red := Harper.red + harperGiven.red, blue := Harper.blue + harperGiven.blue }
  totalCubes finalGage + totalCubes finalHarper = 61 := by
  sorry


end total_cubes_after_distribution_l2748_274888


namespace five_g_growth_equation_l2748_274836

theorem five_g_growth_equation (initial_users : ℕ) (target_users : ℕ) (x : ℝ) :
  initial_users = 30000 →
  target_users = 76800 →
  initial_users * (1 + x)^2 = target_users →
  3 * (1 + x)^2 = 7.68 :=
by sorry

end five_g_growth_equation_l2748_274836


namespace sum_positive_if_difference_abs_positive_l2748_274845

theorem sum_positive_if_difference_abs_positive (a b : ℝ) :
  a - |b| > 0 → b + a > 0 := by
  sorry

end sum_positive_if_difference_abs_positive_l2748_274845


namespace factor_and_divisor_properties_l2748_274852

theorem factor_and_divisor_properties :
  (∃ n : ℤ, 24 = 4 * n) ∧
  (∃ n : ℤ, 209 = 19 * n) ∧ ¬(∃ m : ℤ, 63 = 19 * m) ∧
  (∃ k : ℤ, 180 = 9 * k) := by
sorry

end factor_and_divisor_properties_l2748_274852


namespace total_wheels_in_lot_l2748_274801

/-- The number of wheels on a standard car -/
def wheels_per_car : ℕ := 4

/-- The number of cars in the parking lot -/
def cars_in_lot : ℕ := 17

/-- Theorem: The total number of car wheels in the parking lot is 68 -/
theorem total_wheels_in_lot : cars_in_lot * wheels_per_car = 68 := by
  sorry

end total_wheels_in_lot_l2748_274801


namespace b_hire_charges_l2748_274830

/-- The hire charges for person b given the total cost and usage hours -/
def hire_charges_b (total_cost : ℚ) (hours_a hours_b hours_c : ℚ) : ℚ :=
  total_cost * (hours_b / (hours_a + hours_b + hours_c))

/-- Theorem stating that b's hire charges are 225 Rs given the problem conditions -/
theorem b_hire_charges :
  hire_charges_b 720 9 10 13 = 225 := by
  sorry

end b_hire_charges_l2748_274830


namespace transportation_problem_l2748_274896

/-- Represents the capacity and cost of trucks -/
structure TruckInfo where
  a_capacity : ℝ
  b_capacity : ℝ
  a_cost : ℝ
  b_cost : ℝ

/-- Represents a transportation plan -/
structure TransportPlan where
  a_trucks : ℕ
  b_trucks : ℕ

/-- Theorem stating the properties of the transportation problem -/
theorem transportation_problem (info : TruckInfo) 
  (h1 : 3 * info.a_capacity + 2 * info.b_capacity = 90)
  (h2 : 5 * info.a_capacity + 4 * info.b_capacity = 160)
  (h3 : info.a_cost = 500)
  (h4 : info.b_cost = 400) :
  ∃ (plans : List TransportPlan),
    (info.a_capacity = 20 ∧ info.b_capacity = 15) ∧
    (plans.length = 3) ∧
    (∀ p ∈ plans, p.a_trucks * info.a_capacity + p.b_trucks * info.b_capacity = 190) ∧
    (∃ p ∈ plans, p.a_trucks = 8 ∧ p.b_trucks = 2 ∧
      ∀ p' ∈ plans, p'.a_trucks * info.a_cost + p'.b_trucks * info.b_cost ≥ 
                    p.a_trucks * info.a_cost + p.b_trucks * info.b_cost) := by
  sorry

end transportation_problem_l2748_274896


namespace interest_rate_is_five_percent_l2748_274808

/-- Calculates the interest rate given the principal, time, and interest amount -/
def calculate_interest_rate (principal : ℚ) (time : ℚ) (interest : ℚ) : ℚ :=
  interest / (principal * time)

theorem interest_rate_is_five_percent 
  (principal : ℚ) 
  (time : ℚ) 
  (interest : ℚ) 
  (h1 : principal = 6200)
  (h2 : time = 10)
  (h3 : interest = principal - 3100) :
  calculate_interest_rate principal time interest = 1/20 := by
  sorry

#eval (1/20 : ℚ) * 100 -- To show the result as a percentage

end interest_rate_is_five_percent_l2748_274808


namespace fraction_difference_l2748_274819

theorem fraction_difference (r s : ℕ+) : 
  (5 : ℚ) / 11 < (r : ℚ) / s ∧ 
  (r : ℚ) / s < 4 / 9 ∧ 
  (∀ (r' s' : ℕ+), (5 : ℚ) / 11 < (r' : ℚ) / s' ∧ (r' : ℚ) / s' < 4 / 9 → s ≤ s') →
  s - r = 11 := by
sorry

end fraction_difference_l2748_274819


namespace roots_product_l2748_274837

theorem roots_product (a b c d : ℝ) : 
  (a^2 + 68*a + 1 = 0) →
  (b^2 + 68*b + 1 = 0) →
  (c^2 - 86*c + 1 = 0) →
  (d^2 - 86*d + 1 = 0) →
  (a+c)*(b+c)*(a-d)*(b-d) = 2772 := by
  sorry

end roots_product_l2748_274837


namespace tangent_lines_theorem_point_P_theorem_l2748_274814

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define the tangent line with equal intercepts
def tangent_equal_intercepts (k : ℝ) : Prop :=
  k = 2 + Real.sqrt 6 ∨ k = 2 - Real.sqrt 6 ∨ 
  (∀ x y : ℝ, x + y + 1 = 0) ∨ (∀ x y : ℝ, x + y - 3 = 0)

-- Define the point P outside the circle
def point_P (x y : ℝ) : Prop :=
  ¬ circle_C x y ∧ 2*x - 4*y + 3 = 0 ∧ 2*x + y = 0

-- Theorem for the tangent lines
theorem tangent_lines_theorem :
  ∃ k : ℝ, tangent_equal_intercepts k := by sorry

-- Theorem for the point P
theorem point_P_theorem :
  ∃ x y : ℝ, point_P x y ∧ x = -3/10 ∧ y = 3/5 := by sorry

end tangent_lines_theorem_point_P_theorem_l2748_274814


namespace children_in_milburg_l2748_274843

/-- The number of grown-ups in Milburg -/
def grown_ups : ℕ := 5256

/-- The total population of Milburg -/
def total_population : ℕ := 8243

/-- Theorem: The number of children in Milburg is 2987 -/
theorem children_in_milburg : total_population - grown_ups = 2987 := by
  sorry

end children_in_milburg_l2748_274843


namespace number_1985_in_column_2_l2748_274873

/-- The column number (1-5) in which a given odd positive integer appears when arranged in 5 columns -/
def column_number (n : ℕ) : ℕ :=
  let row := (n - 1) / 5 + 1
  let pos_in_row := (n - 1) % 5 + 1
  if row % 2 = 1 then
    pos_in_row
  else
    6 - pos_in_row

theorem number_1985_in_column_2 : column_number 1985 = 2 := by
  sorry

end number_1985_in_column_2_l2748_274873


namespace solution_set_f_range_of_a_l2748_274881

-- Define the function f
def f (x : ℝ) : ℝ := |1 - 2*x| - |1 + x|

-- Theorem for the solution set of f(x) ≥ 4
theorem solution_set_f (x : ℝ) : f x ≥ 4 ↔ x ≤ -2 ∨ x ≥ 6 := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : 
  (∀ x, a^2 + 2*a + |1 + x| > f x) ↔ a < -3 ∨ a > 1 := by sorry

end solution_set_f_range_of_a_l2748_274881


namespace fraction_value_l2748_274861

theorem fraction_value (a b : ℝ) (h : 1/a - 1/(2*b) = 4) : 
  4*a*b / (a - 2*b) = -1/2 := by sorry

end fraction_value_l2748_274861


namespace aunt_gift_amount_l2748_274833

theorem aunt_gift_amount (jade_initial : ℕ) (julia_initial : ℕ) (total_after : ℕ) 
    (h1 : jade_initial = 38)
    (h2 : julia_initial = jade_initial / 2)
    (h3 : total_after = 97)
    (h4 : ∃ (gift : ℕ), total_after = jade_initial + julia_initial + 2 * gift) :
  ∃ (gift : ℕ), gift = 20 ∧ total_after = jade_initial + julia_initial + 2 * gift :=
by sorry

end aunt_gift_amount_l2748_274833


namespace solution_set_of_inequality_l2748_274838

theorem solution_set_of_inequality (x : ℝ) :
  (x - 2) / (1 - x) > 0 ↔ 1 < x ∧ x < 2 :=
by sorry

end solution_set_of_inequality_l2748_274838


namespace quadratic_roots_condition_l2748_274880

theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ x = (3 + Real.sqrt c) / 2 ∨ x = (3 - Real.sqrt c) / 2) → 
  c = 9/5 := by
sorry

end quadratic_roots_condition_l2748_274880


namespace remaining_amount_for_seat_and_tape_l2748_274846

def initial_amount : ℕ := 60
def frame_cost : ℕ := 15
def wheel_cost : ℕ := 25

theorem remaining_amount_for_seat_and_tape : 
  initial_amount - (frame_cost + wheel_cost) = 20 := by
sorry

end remaining_amount_for_seat_and_tape_l2748_274846


namespace matrix_multiplication_example_l2748_274815

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  let C : Matrix (Fin 2) (Fin 2) ℤ := !![23, -7; 24, -16]
  A * B = C := by sorry

end matrix_multiplication_example_l2748_274815


namespace circle_problem_l2748_274859

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a line is tangent to a circle -/
def Circle.tangentTo (c : Circle) (m a : ℝ) : Prop :=
  let d := |m * c.center.1 - c.center.2 + a| / Real.sqrt (m^2 + 1)
  d = c.radius

/-- The equation of the circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_problem :
  ∃ c : Circle,
    c.contains (2, -1) ∧
    c.tangentTo 1 1 ∧
    c.center.2 = -2 * c.center.1 ∧
    (∀ x y, c.equation x y ↔ ((x - 1)^2 + (y + 2)^2 = 2 ∨ (x - 9)^2 + (y + 18)^2 = 338)) :=
by sorry

end circle_problem_l2748_274859


namespace compound_interest_principal_l2748_274871

/-- Proves that given a final amount of 8400 after 1 year of compound interest
    at 5% per annum (compounded annually), the initial principal amount is 8000. -/
theorem compound_interest_principal (final_amount : ℝ) (interest_rate : ℝ) (time : ℝ) :
  final_amount = 8400 ∧
  interest_rate = 0.05 ∧
  time = 1 →
  ∃ initial_principal : ℝ,
    initial_principal = 8000 ∧
    final_amount = initial_principal * (1 + interest_rate) ^ time :=
by
  sorry

#check compound_interest_principal

end compound_interest_principal_l2748_274871


namespace rationalize_and_sum_l2748_274867

theorem rationalize_and_sum (a b c d e : ℤ) : 
  (∃ (A B C D E : ℤ),
    (3 : ℝ) / (4 * Real.sqrt 7 + 5 * Real.sqrt 2) = 
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = a ∧ B = b ∧ C = c ∧ D = d ∧ E = e) →
  a + b + c + d + e = 68 := by
  sorry

end rationalize_and_sum_l2748_274867


namespace mary_regular_rate_l2748_274820

/-- Represents Mary's work schedule and pay structure --/
structure MaryPayStructure where
  maxHours : ℕ
  regularHours : ℕ
  overtimeRate : ℚ
  maxEarnings : ℚ

/-- Calculates Mary's regular hourly rate --/
def regularHourlyRate (m : MaryPayStructure) : ℚ :=
  let totalRegularHours := m.regularHours
  let totalOvertimeHours := m.maxHours - m.regularHours
  let overtimeMultiplier := 1 + m.overtimeRate
  m.maxEarnings / (totalRegularHours + overtimeMultiplier * totalOvertimeHours)

/-- Theorem stating that Mary's regular hourly rate is $8 --/
theorem mary_regular_rate :
  let m : MaryPayStructure := {
    maxHours := 70,
    regularHours := 20,
    overtimeRate := 1/4,
    maxEarnings := 660
  }
  regularHourlyRate m = 8 := by sorry

end mary_regular_rate_l2748_274820


namespace simplify_complex_fraction_l2748_274866

theorem simplify_complex_fraction : 
  1 / ((1 / (Real.sqrt 3 + 2)) + (2 / (Real.sqrt 5 - 2))) = Real.sqrt 3 - 2 * Real.sqrt 5 - 2 := by
  sorry

end simplify_complex_fraction_l2748_274866


namespace decimal_to_fraction_l2748_274879

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end decimal_to_fraction_l2748_274879


namespace cubic_sum_minus_product_l2748_274860

theorem cubic_sum_minus_product (a b c : ℝ) 
  (h1 : a + b + c = 10) 
  (h2 : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 100 := by
sorry

end cubic_sum_minus_product_l2748_274860


namespace equal_domain_interval_exponential_l2748_274804

/-- Definition of an "equal domain interval" for a function --/
def is_equal_domain_interval (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧
  (∀ x y, m ≤ x ∧ x < y ∧ y ≤ n → f x < f y) ∧
  (∀ y, m ≤ y ∧ y ≤ n → ∃ x, m ≤ x ∧ x ≤ n ∧ f x = y)

/-- Theorem: If [1,2] is an "equal domain interval" for f(x) = a⋅2^x + b (a > 0), then a = 1/2 and b = 0 --/
theorem equal_domain_interval_exponential (a b : ℝ) (h : a > 0) :
  is_equal_domain_interval (λ x => a * 2^x + b) 1 2 → a = 1/2 ∧ b = 0 := by
  sorry


end equal_domain_interval_exponential_l2748_274804


namespace polynomial_division_remainder_l2748_274895

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 + 2 = (X^2 - 3*X + 2) * q + (15*X - 12) := by sorry

end polynomial_division_remainder_l2748_274895


namespace sally_initial_cards_l2748_274890

/-- The number of Pokemon cards Sally received from Dan -/
def cards_from_dan : ℕ := 41

/-- The number of Pokemon cards Sally bought -/
def cards_bought : ℕ := 20

/-- The total number of Pokemon cards Sally has now -/
def total_cards_now : ℕ := 88

/-- The number of Pokemon cards Sally had initially -/
def initial_cards : ℕ := total_cards_now - (cards_from_dan + cards_bought)

theorem sally_initial_cards : initial_cards = 27 := by
  sorry

end sally_initial_cards_l2748_274890


namespace x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l2748_274800

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∀ x : ℝ, x = 1 → x^2 = 1) ∧
  ¬(∀ x : ℝ, x^2 = 1 → x = 1) :=
by sorry

end x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l2748_274800


namespace sequence_periodicity_l2748_274864

theorem sequence_periodicity (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n) :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 9) = a n :=
sorry

end sequence_periodicity_l2748_274864


namespace square_area_from_diagonal_l2748_274825

/-- Theorem: The area of a square with a diagonal of 12 centimeters is 72 square centimeters. -/
theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) :
  diagonal = 12 →
  area = diagonal^2 / 2 →
  area = 72 := by
  sorry

end square_area_from_diagonal_l2748_274825


namespace negation_equivalence_l2748_274816

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 1 ≤ 2*x) ↔ (∀ x : ℝ, x^2 + 1 > 2*x) := by
  sorry

end negation_equivalence_l2748_274816


namespace circle_sum_problem_l2748_274865

theorem circle_sum_problem (a b c d X Y : ℤ) 
  (h1 : a + b + c + d = 40)
  (h2 : X + Y + c + b = 40)
  (h3 : a + b + X = 30)
  (h4 : c + d + Y = 30)
  (h5 : X = 9) :
  Y = 11 := by
  sorry

end circle_sum_problem_l2748_274865


namespace negative_power_fourth_l2748_274894

theorem negative_power_fourth (x : ℝ) : (-x^7)^4 = x^28 := by
  sorry

end negative_power_fourth_l2748_274894


namespace integral_sqrt_x_2_minus_x_l2748_274813

theorem integral_sqrt_x_2_minus_x (f : ℝ → ℝ) :
  (∀ x, f x = Real.sqrt (x * (2 - x))) →
  ∫ x in (0 : ℝ)..1, f x = π / 4 := by
  sorry

end integral_sqrt_x_2_minus_x_l2748_274813


namespace cube_surface_area_l2748_274869

/-- The surface area of a cube with edge length 8 cm is 384 square centimeters. -/
theorem cube_surface_area : 
  let edge_length : ℝ := 8
  let face_area : ℝ := edge_length ^ 2
  let surface_area : ℝ := 6 * face_area
  surface_area = 384 := by
  sorry

end cube_surface_area_l2748_274869


namespace wallpaper_removal_time_is_32_5_l2748_274835

/-- The time it takes to remove wallpaper from the remaining rooms -/
def total_wallpaper_removal_time (dining_remaining_walls : Nat) 
                                 (dining_time_per_wall : Real)
                                 (living_fast_walls : Nat) 
                                 (living_fast_time : Real)
                                 (living_slow_walls : Nat) 
                                 (living_slow_time : Real)
                                 (bedroom_walls : Nat) 
                                 (bedroom_time_per_wall : Real)
                                 (hallway_slow_wall : Nat) 
                                 (hallway_slow_time : Real)
                                 (hallway_fast_walls : Nat) 
                                 (hallway_fast_time : Real) : Real :=
  dining_remaining_walls * dining_time_per_wall +
  living_fast_walls * living_fast_time +
  living_slow_walls * living_slow_time +
  bedroom_walls * bedroom_time_per_wall +
  hallway_slow_wall * hallway_slow_time +
  hallway_fast_walls * hallway_fast_time

/-- Theorem stating that the total wallpaper removal time is 32.5 hours -/
theorem wallpaper_removal_time_is_32_5 : 
  total_wallpaper_removal_time 3 1.5 2 1 2 2.5 3 3 1 4 4 2 = 32.5 := by
  sorry


end wallpaper_removal_time_is_32_5_l2748_274835


namespace mans_speed_against_current_l2748_274887

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem: Given a man's speed with the current of 21 km/hr and a current speed of 4.3 km/hr,
    the man's speed against the current is 12.4 km/hr. -/
theorem mans_speed_against_current :
  speed_against_current 21 4.3 = 12.4 := by
  sorry

#eval speed_against_current 21 4.3

end mans_speed_against_current_l2748_274887


namespace ninth_minus_eighth_square_tiles_l2748_274848

/-- The side length of the nth square in the sequence -/
def square_side (n : ℕ) : ℕ := 2 * n - 1

/-- The number of tiles in the nth square -/
def square_tiles (n : ℕ) : ℕ := (square_side n) ^ 2

/-- The difference in tiles between the 9th and 8th squares -/
def tile_difference : ℕ := square_tiles 9 - square_tiles 8

theorem ninth_minus_eighth_square_tiles : tile_difference = 64 := by
  sorry

end ninth_minus_eighth_square_tiles_l2748_274848


namespace positive_abc_l2748_274893

theorem positive_abc (a b c : ℝ) 
  (sum_pos : a + b + c > 0) 
  (sum_prod_pos : a * b + b * c + c * a > 0) 
  (prod_pos : a * b * c > 0) : 
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end positive_abc_l2748_274893


namespace problem_1_l2748_274882

theorem problem_1 (a : ℝ) : a * (2 - a) + (a + 1) * (a - 1) = 2 * a - 1 := by
  sorry

end problem_1_l2748_274882


namespace james_club_expenditure_l2748_274858

/-- Calculate the total amount James spent at the club --/
theorem james_club_expenditure :
  let entry_fee : ℕ := 20
  let rounds_for_friends : ℕ := 2
  let num_friends : ℕ := 5
  let drinks_for_self : ℕ := 6
  let drink_cost : ℕ := 6
  let food_cost : ℕ := 14
  let tip_percentage : ℚ := 30 / 100

  let drinks_cost : ℕ := rounds_for_friends * num_friends * drink_cost + drinks_for_self * drink_cost
  let subtotal : ℕ := entry_fee + drinks_cost + food_cost
  let tip : ℕ := (tip_percentage * (drinks_cost + food_cost)).num.toNat
  let total_spent : ℕ := subtotal + tip

  total_spent = 163 := by sorry

end james_club_expenditure_l2748_274858


namespace absent_children_count_l2748_274855

/-- Given a school with a total of 700 children, where each child was supposed to get 2 bananas,
    but due to absences, each present child got 4 bananas instead,
    prove that the number of absent children is 350. -/
theorem absent_children_count (total_children : ℕ) (bananas_per_child_original : ℕ) 
    (bananas_per_child_actual : ℕ) (absent_children : ℕ) : 
    total_children = 700 → 
    bananas_per_child_original = 2 →
    bananas_per_child_actual = 4 →
    absent_children = total_children - (total_children * bananas_per_child_original) / bananas_per_child_actual →
    absent_children = 350 := by
  sorry

end absent_children_count_l2748_274855


namespace cuboid_properties_l2748_274883

/-- Represents a cuboid with given length, width, and height -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total length of edges of a cuboid -/
def totalEdgeLength (c : Cuboid) : ℝ :=
  4 * (c.length + c.width + c.height)

/-- Calculates the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.width * c.height + c.height * c.length)

/-- Theorem stating the correctness of totalEdgeLength and surfaceArea functions -/
theorem cuboid_properties (c : Cuboid) :
  (totalEdgeLength c = 4 * (c.length + c.width + c.height)) ∧
  (surfaceArea c = 2 * (c.length * c.width + c.width * c.height + c.height * c.length)) := by
  sorry

end cuboid_properties_l2748_274883


namespace boat_upstream_speed_l2748_274817

/-- Calculates the upstream speed of a boat given its still water speed and downstream speed. -/
def upstream_speed (still_water_speed downstream_speed : ℝ) : ℝ :=
  2 * still_water_speed - downstream_speed

/-- Proves that a boat with a still water speed of 11 km/hr and a downstream speed of 15 km/hr 
    has an upstream speed of 7 km/hr. -/
theorem boat_upstream_speed :
  upstream_speed 11 15 = 7 := by
  sorry

end boat_upstream_speed_l2748_274817


namespace ellipse_equation_l2748_274886

/-- An ellipse with a = 2b passing through point (2, 0) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  a_eq_2b : a = 2 * b
  passes_through_2_0 : (2 : ℝ)^2 / (a^2) + 0^2 / (b^2) = 1

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  ∀ (x y : ℝ), x^2 / 4 + y^2 = 1 ↔ x^2 / (e.a^2) + y^2 / (e.b^2) = 1

theorem ellipse_equation (e : Ellipse) : standard_equation e := by
  sorry

end ellipse_equation_l2748_274886


namespace no_solution_equations_l2748_274811

theorem no_solution_equations :
  (∀ x : ℝ, |4*x| + 7 ≠ 0) ∧
  (∀ x : ℝ, Real.sqrt (-3*x) + 1 ≠ 0) ∧
  (∃ x : ℝ, (x - 3)^2 = 0) ∧
  (∃ x : ℝ, Real.sqrt (2*x) - 5 = 0) ∧
  (∃ x : ℝ, |2*x| - 10 = 0) :=
by sorry

end no_solution_equations_l2748_274811


namespace bell_rings_count_l2748_274821

/-- Represents a school day with a given number of classes -/
structure SchoolDay where
  num_classes : ℕ
  current_class : ℕ

/-- Calculates the number of times the bell has rung -/
def bell_rings (day : SchoolDay) : ℕ :=
  2 * (day.current_class - 1) + 1

/-- Theorem stating that the bell has rung 15 times -/
theorem bell_rings_count (day : SchoolDay) 
  (h1 : day.num_classes = 8) 
  (h2 : day.current_class = day.num_classes) : 
  bell_rings day = 15 := by
  sorry

#check bell_rings_count

end bell_rings_count_l2748_274821


namespace time_after_1876_minutes_l2748_274851

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time and wraps around to the next day if necessary -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := (totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  { hours := newHours, minutes := newMinutes }

def startTime : Time := { hours := 15, minutes := 0 }  -- 3:00 PM
def minutesToAdd : Nat := 1876

theorem time_after_1876_minutes :
  addMinutes startTime minutesToAdd = { hours := 10, minutes := 16 } := by
  sorry

end time_after_1876_minutes_l2748_274851


namespace quadratic_equation_sum_l2748_274823

theorem quadratic_equation_sum (p q : ℝ) : 
  (∀ x, 9*x^2 - 36*x - 81 = 0 ↔ (x + p)^2 = q) → p + q = 11 := by
sorry

end quadratic_equation_sum_l2748_274823


namespace max_handshakes_25_20_l2748_274889

/-- Represents a meeting with a given number of people and a maximum number of handshakes per person. -/
structure Meeting where
  num_people : ℕ
  max_handshakes_per_person : ℕ

/-- Calculates the maximum number of handshakes in a meeting. -/
def max_handshakes (m : Meeting) : ℕ :=
  (m.num_people * m.max_handshakes_per_person) / 2

/-- Theorem stating that in a meeting of 25 people where each person shakes hands with at most 20 others,
    the maximum number of handshakes is 250. -/
theorem max_handshakes_25_20 :
  let m : Meeting := ⟨25, 20⟩
  max_handshakes m = 250 := by
  sorry

#eval max_handshakes ⟨25, 20⟩

end max_handshakes_25_20_l2748_274889


namespace remainder_theorem_application_l2748_274875

theorem remainder_theorem_application (D E F : ℝ) : 
  let q : ℝ → ℝ := λ x => D * x^4 + E * x^2 + F * x - 2
  (q 2 = 10) → (q (-2) = -2) := by
sorry

end remainder_theorem_application_l2748_274875


namespace purely_imaginary_complex_fraction_l2748_274822

/-- Given that z = (a - √2) + ai is a purely imaginary number where a ∈ ℝ,
    prove that (a + i⁷) / (1 + ai) = -i -/
theorem purely_imaginary_complex_fraction (a : ℝ) :
  (a - Real.sqrt 2 : ℂ) + a * I = (0 : ℂ) →
  (a + I^7) / (1 + a * I) = -I := by sorry

end purely_imaginary_complex_fraction_l2748_274822


namespace arithmetic_sequence_problem_l2748_274877

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 1 + a 3 = 2) : 
  a 2 = 1 := by
  sorry

end arithmetic_sequence_problem_l2748_274877


namespace trigonometric_expression_equals_two_l2748_274885

theorem trigonometric_expression_equals_two (α : ℝ) : 
  (Real.sin (π + α))^2 - Real.cos (π + α) * Real.cos (-α) + 1 = 2 := by
sorry

end trigonometric_expression_equals_two_l2748_274885


namespace break_room_seating_capacity_l2748_274868

/-- Given a break room with tables and total seating capacity, 
    calculate the number of people each table can seat. -/
def people_per_table (num_tables : ℕ) (total_capacity : ℕ) : ℕ :=
  total_capacity / num_tables

theorem break_room_seating_capacity 
  (num_tables : ℕ) (total_capacity : ℕ) 
  (h1 : num_tables = 4) 
  (h2 : total_capacity = 32) : 
  people_per_table num_tables total_capacity = 8 := by
  sorry

end break_room_seating_capacity_l2748_274868


namespace sum_of_numbers_l2748_274807

theorem sum_of_numbers (x y : ℝ) (h1 : x - y = 10) (h2 : x * y = 200) : x + y = 30 := by
  sorry

end sum_of_numbers_l2748_274807


namespace expected_gain_is_negative_three_halves_l2748_274897

/-- Represents the faces of the three-sided die -/
inductive DieFace
  | Heads
  | Tails
  | Edge

/-- The probability of rolling each face -/
def probability (face : DieFace) : ℚ :=
  match face with
  | DieFace.Heads => 1/4
  | DieFace.Tails => 1/4
  | DieFace.Edge => 1/2

/-- The gain (or loss) associated with each face -/
def gain (face : DieFace) : ℤ :=
  match face with
  | DieFace.Heads => 2
  | DieFace.Tails => 4
  | DieFace.Edge => -6

/-- The expected gain from rolling the die once -/
def expected_gain : ℚ :=
  (probability DieFace.Heads * gain DieFace.Heads) +
  (probability DieFace.Tails * gain DieFace.Tails) +
  (probability DieFace.Edge * gain DieFace.Edge)

theorem expected_gain_is_negative_three_halves :
  expected_gain = -3/2 := by sorry

end expected_gain_is_negative_three_halves_l2748_274897


namespace field_path_area_and_cost_l2748_274856

/-- Represents the dimensions of a rectangular field with a path around it -/
structure FieldWithPath where
  field_length : ℝ
  field_width : ℝ
  path_width : ℝ

/-- Calculates the area of the path around a rectangular field -/
def path_area (f : FieldWithPath) : ℝ :=
  (f.field_length + 2 * f.path_width) * (f.field_width + 2 * f.path_width) - f.field_length * f.field_width

/-- Calculates the cost of constructing the path given a cost per square meter -/
def path_cost (f : FieldWithPath) (cost_per_sqm : ℝ) : ℝ :=
  path_area f * cost_per_sqm

/-- Theorem stating the area of the path and its construction cost for the given field -/
theorem field_path_area_and_cost :
  let f : FieldWithPath := { field_length := 75, field_width := 40, path_width := 2.5 }
  path_area f = 600 ∧ path_cost f 2 = 1200 := by sorry

end field_path_area_and_cost_l2748_274856


namespace only_negative_four_has_no_sqrt_l2748_274834

-- Define the set of numbers we're considering
def numbers : Set ℝ := {-4, 0, 0.5, 2}

-- Define what it means for a real number to have a square root
def has_sqrt (x : ℝ) : Prop := ∃ y : ℝ, y^2 = x

-- State the theorem
theorem only_negative_four_has_no_sqrt :
  ∀ x ∈ numbers, ¬(has_sqrt x) ↔ x = -4 := by
  sorry

end only_negative_four_has_no_sqrt_l2748_274834


namespace no_rectangle_with_given_cuts_l2748_274824

theorem no_rectangle_with_given_cuts : ¬ ∃ (w h : ℕ), 
  (w * h = 37 + 135 * 3) ∧ 
  (w ≥ 2 ∧ h ≥ 2) ∧
  (w * h - 37 ≥ 135 * 3) :=
sorry

end no_rectangle_with_given_cuts_l2748_274824


namespace dinner_total_cost_l2748_274827

def food_cost : ℝ := 30
def sales_tax_rate : ℝ := 0.095
def tip_rate : ℝ := 0.10

theorem dinner_total_cost : 
  food_cost + (food_cost * sales_tax_rate) + (food_cost * tip_rate) = 35.85 := by
  sorry

end dinner_total_cost_l2748_274827


namespace tip_percentage_is_twenty_percent_l2748_274803

def lunch_cost : ℝ := 60.80
def total_spent : ℝ := 72.96

theorem tip_percentage_is_twenty_percent :
  (total_spent - lunch_cost) / lunch_cost * 100 = 20 := by sorry

end tip_percentage_is_twenty_percent_l2748_274803


namespace vector_problem_l2748_274831

def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def c : ℝ × ℝ := (1, -1)

def m (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

theorem vector_problem :
  (∃ k : ℝ, (m k).1 * (2 * a.1 - b.1) + (m k).2 * (2 * a.2 - b.2) = 0 ∧ k = 5/3) ∧
  (∃ k : ℝ, ∃ t : ℝ, t ≠ 0 ∧ (m k).1 = t * (k * b.1 + c.1) ∧ (m k).2 = t * (k * b.2 + c.2) ∧ k = -1/3) :=
by sorry

end vector_problem_l2748_274831


namespace intersection_A_B_when_m_1_sufficient_necessary_condition_l2748_274812

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 7*x + 6 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Part 1: Intersection of A and B when m = 1
theorem intersection_A_B_when_m_1 : A ∩ B 1 = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2: Condition for x ∈ A to be sufficient and necessary for x ∈ B
theorem sufficient_necessary_condition (m : ℝ) :
  (∀ x, x ∈ A ↔ x ∈ B m) ↔ m ≥ 5 := by sorry

end intersection_A_B_when_m_1_sufficient_necessary_condition_l2748_274812


namespace unshaded_area_square_with_circles_l2748_274878

/-- The area of the unshaded region in a square with three-quarter circles at corners -/
theorem unshaded_area_square_with_circles (side_length : ℝ) (h : side_length = 12) :
  let radius : ℝ := side_length / 4
  let square_area : ℝ := side_length ^ 2
  let circle_area : ℝ := π * radius ^ 2
  let total_circle_area : ℝ := 4 * (3 / 4) * circle_area
  square_area - total_circle_area = 144 - 27 * π :=
by sorry

end unshaded_area_square_with_circles_l2748_274878


namespace min_value_quadratic_l2748_274841

theorem min_value_quadratic (x y : ℝ) :
  2 * x^2 + 2 * y^2 - 8 * x + 6 * y + 25 ≥ 12.5 := by
  sorry

end min_value_quadratic_l2748_274841


namespace convention_handshakes_l2748_274842

/-- Represents the Annual Mischief Convention --/
structure Convention where
  num_gremlins : ℕ
  num_imps : ℕ
  num_antisocial_gremlins : ℕ

/-- Calculates the number of handshakes at the convention --/
def count_handshakes (c : Convention) : ℕ :=
  let social_gremlins := c.num_gremlins - c.num_antisocial_gremlins
  let gremlin_handshakes := social_gremlins * (social_gremlins - 1) / 2
  let imp_gremlin_handshakes := c.num_imps * c.num_gremlins
  gremlin_handshakes + imp_gremlin_handshakes

/-- The main theorem stating the number of handshakes at the convention --/
theorem convention_handshakes :
  let c := Convention.mk 25 18 5
  count_handshakes c = 640 := by
  sorry

end convention_handshakes_l2748_274842


namespace money_distribution_l2748_274892

theorem money_distribution (ram gopal krishan : ℚ) : 
  (ram / gopal = 7 / 17) →
  (gopal / krishan = 7 / 17) →
  (ram = 490) →
  (krishan = 2890) :=
by sorry

end money_distribution_l2748_274892


namespace range_of_a_l2748_274805

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + (a+2)*x + b

-- Define the theorem
theorem range_of_a (a b : ℝ) :
  (f a b (-1) = -2) →
  (∀ x, ∃ y, y = Real.log (f a b x + 3)) →
  -2 < a ∧ a < 2 := by
  sorry

end range_of_a_l2748_274805


namespace donut_selection_count_l2748_274829

theorem donut_selection_count :
  let n : ℕ := 6  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 84 := by
sorry

end donut_selection_count_l2748_274829


namespace calculate_expression_l2748_274857

theorem calculate_expression : (8^3 / 8^2) * 2^6 = 512 := by
  sorry

end calculate_expression_l2748_274857


namespace division_simplification_l2748_274828

theorem division_simplification : 
  (250 : ℚ) / (15 + 13 * 3^2) = 125 / 66 := by sorry

end division_simplification_l2748_274828


namespace statement_a_not_proposition_l2748_274854

-- Define what a proposition is
def is_proposition (s : String) : Prop := 
  (s = "true" ∨ s = "false") ∧ ¬(s = "true" ∧ s = "false")

-- Define the statement
def statement_a : String := "It may rain tomorrow"

-- Theorem to prove
theorem statement_a_not_proposition : ¬(is_proposition statement_a) := by
  sorry

end statement_a_not_proposition_l2748_274854


namespace ring_toss_losers_l2748_274818

theorem ring_toss_losers (winners : ℕ) (ratio : ℚ) (h1 : winners = 28) (h2 : ratio = 4/1) : 
  (winners : ℚ) / ratio = 7 := by
  sorry

end ring_toss_losers_l2748_274818


namespace second_shot_probability_l2748_274844

/-- Probability of scoring in the next shot if the previous shot was successful -/
def p_success : ℚ := 3/4

/-- Probability of scoring in the next shot if the previous shot was missed -/
def p_miss : ℚ := 1/4

/-- Probability of scoring in the first shot -/
def p_first : ℚ := 3/4

/-- The probability of scoring in the second shot -/
def p_second : ℚ := p_first * p_success + (1 - p_first) * p_miss

theorem second_shot_probability : p_second = 5/8 := by
  sorry

end second_shot_probability_l2748_274844
