import Mathlib

namespace NUMINAMATH_CALUDE_marys_remaining_money_l777_77713

/-- The amount of money Mary has left after her purchases -/
def money_left (p : ℝ) : ℝ :=
  50 - (4 * p + 2.5 * p + 2 * 4 * p)

/-- Theorem stating that Mary's remaining money is 50 - 14.5p dollars -/
theorem marys_remaining_money (p : ℝ) : money_left p = 50 - 14.5 * p := by
  sorry

end NUMINAMATH_CALUDE_marys_remaining_money_l777_77713


namespace NUMINAMATH_CALUDE_roast_cost_is_17_l777_77703

/-- Calculates the cost of a roast given initial money, vegetable cost, and remaining money --/
def roast_cost (initial_money : ℤ) (vegetable_cost : ℤ) (remaining_money : ℤ) : ℤ :=
  initial_money - vegetable_cost - remaining_money

/-- Proves that the roast cost €17 given the problem conditions --/
theorem roast_cost_is_17 :
  roast_cost 100 11 72 = 17 := by
  sorry

end NUMINAMATH_CALUDE_roast_cost_is_17_l777_77703


namespace NUMINAMATH_CALUDE_circus_tickets_cost_l777_77772

def adult_ticket_price : ℚ := 44
def child_ticket_price : ℚ := 28
def num_adults : ℕ := 2
def num_children : ℕ := 5
def discount_rate : ℚ := 0.1
def discount_threshold : ℕ := 6

def total_cost : ℚ :=
  let total_tickets := num_adults + num_children
  let subtotal := num_adults * adult_ticket_price + num_children * child_ticket_price
  if total_tickets > discount_threshold then
    subtotal * (1 - discount_rate)
  else
    subtotal

theorem circus_tickets_cost :
  total_cost = 205.2 := by sorry

end NUMINAMATH_CALUDE_circus_tickets_cost_l777_77772


namespace NUMINAMATH_CALUDE_real_roots_of_p_l777_77722

/-- The polynomial under consideration -/
def p (x : ℝ) : ℝ := x^5 - 3*x^4 - x^2 + 3*x

/-- The set of real roots of the polynomial -/
def root_set : Set ℝ := {0, 1, 3}

/-- Theorem stating that root_set contains exactly the real roots of p -/
theorem real_roots_of_p :
  ∀ x : ℝ, x ∈ root_set ↔ p x = 0 :=
sorry

end NUMINAMATH_CALUDE_real_roots_of_p_l777_77722


namespace NUMINAMATH_CALUDE_cos_75_degrees_l777_77785

theorem cos_75_degrees : 
  let cos_75 := Real.cos (75 * π / 180)
  let cos_60 := Real.cos (60 * π / 180)
  let sin_60 := Real.sin (60 * π / 180)
  let cos_15 := Real.cos (15 * π / 180)
  let sin_15 := Real.sin (15 * π / 180)
  cos_60 = 1/2 ∧ sin_60 = Real.sqrt 3 / 2 →
  cos_75 = cos_60 * cos_15 - sin_60 * sin_15 →
  cos_75 = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l777_77785


namespace NUMINAMATH_CALUDE_prob_of_three_l777_77715

/-- The decimal representation of 8/13 -/
def decimal_rep : ℚ := 8 / 13

/-- The length of the repeating block in the decimal representation -/
def block_length : ℕ := 6

/-- The count of digit 3 in one repeating block -/
def count_of_threes : ℕ := 1

/-- The probability of randomly selecting the digit 3 from the decimal representation of 8/13 -/
theorem prob_of_three (decimal_rep : ℚ) (block_length : ℕ) (count_of_threes : ℕ) :
  decimal_rep = 8 / 13 →
  block_length = 6 →
  count_of_threes = 1 →
  (count_of_threes : ℚ) / (block_length : ℚ) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_of_three_l777_77715


namespace NUMINAMATH_CALUDE_age_problem_l777_77776

/-- Given ages a, b, c, d, and their sum Y, prove b's age. -/
theorem age_problem (a b c d Y : ℚ) 
  (h1 : a = b + 2)           -- a is two years older than b
  (h2 : b = 2 * c)           -- b is twice as old as c
  (h3 : d = a / 2)           -- d is half the age of a
  (h4 : a + b + c + d = Y)   -- sum of ages is Y
  : b = Y / 3 - 1 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l777_77776


namespace NUMINAMATH_CALUDE_floor_subtraction_inequality_l777_77759

theorem floor_subtraction_inequality (x y : ℝ) : ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_subtraction_inequality_l777_77759


namespace NUMINAMATH_CALUDE_matilda_age_is_35_l777_77784

-- Define the ages as natural numbers
def louis_age : ℕ := 14
def jerica_age : ℕ := 2 * louis_age
def matilda_age : ℕ := jerica_age + 7

-- Theorem statement
theorem matilda_age_is_35 : matilda_age = 35 := by
  sorry

end NUMINAMATH_CALUDE_matilda_age_is_35_l777_77784


namespace NUMINAMATH_CALUDE_interest_calculation_l777_77740

def total_investment : ℝ := 33000
def rate1 : ℝ := 0.04
def rate2 : ℝ := 0.0225
def partial_investment : ℝ := 13000

theorem interest_calculation :
  ∃ (investment1 investment2 : ℝ),
    investment1 + investment2 = total_investment ∧
    (investment1 = partial_investment ∨ investment2 = partial_investment) ∧
    investment1 * rate1 + investment2 * rate2 = 970 :=
by sorry

end NUMINAMATH_CALUDE_interest_calculation_l777_77740


namespace NUMINAMATH_CALUDE_trig_values_of_α_l777_77779

-- Define the angle α and its properties
def α : Real := sorry

-- Define that the terminal side of α passes through (3, 4)
axiom terminal_point : ∃ (r : Real), r * Real.cos α = 3 ∧ r * Real.sin α = 4

-- Theorem to prove
theorem trig_values_of_α :
  Real.sin α = 4/5 ∧ Real.cos α = 3/5 ∧ Real.tan α = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_values_of_α_l777_77779


namespace NUMINAMATH_CALUDE_savings_calculation_l777_77786

theorem savings_calculation (income expenditure savings : ℕ) : 
  (income : ℚ) / expenditure = 8 / 7 →
  income = 40000 →
  savings = income - expenditure →
  savings = 5000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l777_77786


namespace NUMINAMATH_CALUDE_newspaper_conference_max_neither_l777_77705

theorem newspaper_conference_max_neither (total : ℕ) (writers : ℕ) (editors : ℕ) (x : ℕ) (N : ℕ) :
  total = 90 →
  writers = 45 →
  editors ≥ 39 →
  writers + editors - x + N = total →
  N = 2 * x →
  N ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_newspaper_conference_max_neither_l777_77705


namespace NUMINAMATH_CALUDE_total_pupils_l777_77796

def number_of_girls : ℕ := 542
def number_of_boys : ℕ := 387

theorem total_pupils : number_of_girls + number_of_boys = 929 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_l777_77796


namespace NUMINAMATH_CALUDE_pole_wire_length_l777_77767

def pole_problem (base_distance : ℝ) (short_pole_height : ℝ) (tall_pole_height : ℝ) (short_pole_elevation : ℝ) : Prop :=
  let effective_short_pole_height : ℝ := short_pole_height + short_pole_elevation
  let vertical_distance : ℝ := tall_pole_height - effective_short_pole_height
  let wire_length : ℝ := Real.sqrt (base_distance^2 + vertical_distance^2)
  wire_length = Real.sqrt 445

theorem pole_wire_length :
  pole_problem 18 6 20 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pole_wire_length_l777_77767


namespace NUMINAMATH_CALUDE_second_race_lead_l777_77782

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

theorem second_race_lead (h d : ℝ) (first_race second_race : Race) :
  h > 0 ∧ d > 0 ∧
  first_race.distance = h ∧
  second_race.distance = h ∧
  first_race.runner_a = second_race.runner_a ∧
  first_race.runner_b = second_race.runner_b ∧
  first_race.runner_a.speed * h = first_race.runner_b.speed * (h - 2 * d) →
  let finish_time := (h + 2 * d) / first_race.runner_a.speed
  finish_time * first_race.runner_a.speed - finish_time * first_race.runner_b.speed = 4 * d^2 / h :=
by sorry

end NUMINAMATH_CALUDE_second_race_lead_l777_77782


namespace NUMINAMATH_CALUDE_min_sum_distances_to_lines_l777_77744

/-- The minimum sum of distances from a point on the parabola y² = 4x to two specific lines -/
theorem min_sum_distances_to_lines : ∃ (a : ℝ),
  let P : ℝ × ℝ := (a^2, 2*a)
  let d₁ : ℝ := |4*a^2 - 6*a + 6| / 5  -- Distance to line 4x - 3y + 6 = 0
  let d₂ : ℝ := a^2                    -- Distance to line x = 0
  (∀ b : ℝ, d₁ + d₂ ≤ |4*b^2 - 6*b + 6| / 5 + b^2) ∧ 
  d₁ + d₂ = 1 :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_to_lines_l777_77744


namespace NUMINAMATH_CALUDE_age_difference_proof_l777_77751

theorem age_difference_proof (man_age son_age : ℕ) : 
  man_age > son_age →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 26 →
  man_age - son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l777_77751


namespace NUMINAMATH_CALUDE_total_stars_l777_77725

theorem total_stars (num_students : ℕ) (stars_per_student : ℕ) 
  (h1 : num_students = 124) 
  (h2 : stars_per_student = 3) : 
  num_students * stars_per_student = 372 := by
sorry

end NUMINAMATH_CALUDE_total_stars_l777_77725


namespace NUMINAMATH_CALUDE_carmen_fudge_delights_sales_l777_77729

/-- Represents the number of boxes sold for each cookie type -/
structure CookieSales where
  samoas : Nat
  thin_mints : Nat
  fudge_delights : Nat
  sugar_cookies : Nat

/-- Represents the price of each cookie type -/
structure CookiePrices where
  samoas : Rat
  thin_mints : Rat
  fudge_delights : Rat
  sugar_cookies : Rat

/-- Calculates the total revenue from cookie sales -/
def total_revenue (sales : CookieSales) (prices : CookiePrices) : Rat :=
  sales.samoas * prices.samoas +
  sales.thin_mints * prices.thin_mints +
  sales.fudge_delights * prices.fudge_delights +
  sales.sugar_cookies * prices.sugar_cookies

/-- The main theorem stating that Carmen sold 1 box of fudge delights -/
theorem carmen_fudge_delights_sales
  (sales : CookieSales)
  (prices : CookiePrices)
  (h1 : sales.samoas = 3)
  (h2 : sales.thin_mints = 2)
  (h3 : sales.sugar_cookies = 9)
  (h4 : prices.samoas = 4)
  (h5 : prices.thin_mints = 7/2)
  (h6 : prices.fudge_delights = 5)
  (h7 : prices.sugar_cookies = 2)
  (h8 : total_revenue sales prices = 42) :
  sales.fudge_delights = 1 := by
  sorry


end NUMINAMATH_CALUDE_carmen_fudge_delights_sales_l777_77729


namespace NUMINAMATH_CALUDE_tan_3x_domain_l777_77793

theorem tan_3x_domain (x : ℝ) : 
  ∃ y, y = Real.tan (3 * x) ↔ ∀ k : ℤ, x ≠ π / 6 + k * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_3x_domain_l777_77793


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_two_l777_77752

theorem sqrt_sum_eq_two (x₁ x₂ : ℝ) 
  (h1 : x₁ ≥ x₂) 
  (h2 : x₂ ≥ 0) 
  (h3 : x₁ + x₂ = 2) : 
  Real.sqrt (x₁ + Real.sqrt (x₁^2 - x₂^2)) + Real.sqrt (x₁ - Real.sqrt (x₁^2 - x₂^2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_two_l777_77752


namespace NUMINAMATH_CALUDE_pencils_across_diameter_l777_77780

theorem pencils_across_diameter (radius : ℝ) (pencil_length : ℝ) :
  radius = 14 →
  pencil_length = 0.5 →
  (2 * radius) / pencil_length = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_pencils_across_diameter_l777_77780


namespace NUMINAMATH_CALUDE_real_part_range_l777_77790

theorem real_part_range (z : ℂ) (ω : ℝ) (h1 : ω = z + z⁻¹) (h2 : -1 < ω) (h3 : ω < 2) :
  -1/2 < z.re ∧ z.re < 1 := by sorry

end NUMINAMATH_CALUDE_real_part_range_l777_77790


namespace NUMINAMATH_CALUDE_combined_salaries_l777_77736

/-- The combined salaries of A, C, D, and E given B's salary and the average salary of all five. -/
theorem combined_salaries 
  (salary_B : ℕ) 
  (average_salary : ℕ) 
  (num_individuals : ℕ) 
  (h1 : salary_B = 5000)
  (h2 : average_salary = 8400)
  (h3 : num_individuals = 5) :
  average_salary * num_individuals - salary_B = 37000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l777_77736


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_pi_sixth_l777_77766

theorem sin_2alpha_minus_pi_sixth (α : ℝ) 
  (h : Real.cos (α + π / 6) = Real.sqrt 3 / 3) : 
  Real.sin (2 * α - π / 6) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_pi_sixth_l777_77766


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l777_77747

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (3, 4) and (9, 18) is 17 -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 3
  let y1 : ℝ := 4
  let x2 : ℝ := 9
  let y2 : ℝ := 18
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 17 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l777_77747


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l777_77765

theorem stratified_sampling_sample_size 
  (total_population : ℕ) 
  (selection_probability : ℝ) 
  (sample_size : ℕ) :
  total_population = 1200 →
  selection_probability = 0.4 →
  (sample_size : ℝ) / total_population = selection_probability →
  sample_size = 480 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l777_77765


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l777_77721

theorem solve_exponential_equation (x : ℝ) :
  3^(x - 1) = (1 : ℝ) / 9 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l777_77721


namespace NUMINAMATH_CALUDE_pencil_cost_l777_77795

/-- Given Mrs. Hilt's initial amount and the amount left after buying a pencil,
    prove that the cost of the pencil is the difference between these two amounts. -/
theorem pencil_cost (initial_amount amount_left : ℕ) 
    (h1 : initial_amount = 15)
    (h2 : amount_left = 4) :
    initial_amount - amount_left = 11 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l777_77795


namespace NUMINAMATH_CALUDE_pen_cost_calculation_l777_77799

def notebook_cost (pen_cost : ℝ) : ℝ := 3 * pen_cost

theorem pen_cost_calculation (total_cost : ℝ) (num_notebooks : ℕ) 
  (h1 : total_cost = 18)
  (h2 : num_notebooks = 4) :
  ∃ (pen_cost : ℝ), 
    pen_cost = 1.5 ∧ 
    total_cost = num_notebooks * (notebook_cost pen_cost) :=
by
  sorry

end NUMINAMATH_CALUDE_pen_cost_calculation_l777_77799


namespace NUMINAMATH_CALUDE_regular_polygon_sides_is_ten_l777_77718

/-- The number of sides of a regular polygon with an interior angle of 144 degrees -/
def regular_polygon_sides : ℕ := by
  -- Define the interior angle
  let interior_angle : ℝ := 144

  -- Define the function for the sum of interior angles of an n-sided polygon
  let sum_of_angles (n : ℕ) : ℝ := 180 * (n - 2)

  -- Define the equation: sum of angles equals n times the interior angle
  let sides_equation (n : ℕ) : Prop := sum_of_angles n = n * interior_angle

  -- The number of sides is the solution to this equation
  exact sorry

theorem regular_polygon_sides_is_ten : regular_polygon_sides = 10 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_is_ten_l777_77718


namespace NUMINAMATH_CALUDE_parabola_cross_section_l777_77701

/-- Represents a cone --/
structure Cone where
  vertex_angle : ℝ

/-- Represents a cross-section of a cone --/
structure CrossSection where
  angle_with_axis : ℝ

/-- Represents the type of curve formed by a cross-section --/
inductive CurveType
  | Circle
  | Ellipse
  | Hyperbola
  | Parabola

/-- Determines the curve type of a cross-section for a given cone --/
def cross_section_curve_type (cone : Cone) (cs : CrossSection) : CurveType :=
  sorry

/-- Theorem stating that for a cone with 90° vertex angle and 45° cross-section angle, 
    the resulting curve is a parabola --/
theorem parabola_cross_section 
  (cone : Cone) 
  (cs : CrossSection) 
  (h1 : cone.vertex_angle = 90) 
  (h2 : cs.angle_with_axis = 45) : 
  cross_section_curve_type cone cs = CurveType.Parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_cross_section_l777_77701


namespace NUMINAMATH_CALUDE_f_properties_l777_77711

noncomputable def f (x : Real) : Real := Real.sqrt 3 * (Real.sin x) ^ 2 + Real.sin x * Real.cos x

theorem f_properties :
  let a := π / 2
  let b := π
  ∃ (max_value min_value : Real),
    (∀ x ∈ Set.Icc a b, f x ≤ max_value) ∧
    (∀ x ∈ Set.Icc a b, f x ≥ min_value) ∧
    (f (5 * π / 6) = 0) ∧
    (f π = 0) ∧
    (max_value = Real.sqrt 3) ∧
    (f (π / 2) = max_value) ∧
    (min_value = -1 + Real.sqrt 3 / 2) ∧
    (f (11 * π / 12) = min_value) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l777_77711


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l777_77743

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given that {a, b, c} form a basis for a space V, prove that {a + b, a - b, c} are not coplanar -/
theorem vectors_not_coplanar (a b c : V) 
  (h : LinearIndependent ℝ ![a, b, c]) :
  ¬ (∃ (x y z : ℝ), x • (a + b) + y • (a - b) + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l777_77743


namespace NUMINAMATH_CALUDE_cabinet_price_l777_77750

theorem cabinet_price (P : ℝ) (discount_rate : ℝ) (discounted_price : ℝ) : 
  discount_rate = 0.15 →
  discounted_price = 1020 →
  discounted_price = P * (1 - discount_rate) →
  P = 1200 := by
sorry

end NUMINAMATH_CALUDE_cabinet_price_l777_77750


namespace NUMINAMATH_CALUDE_digit_multiplication_theorem_l777_77700

/-- A function that checks if a number is a digit (0-9) -/
def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- A function that converts a three-digit number to its decimal representation -/
def three_digit_to_decimal (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- A function that converts a four-digit number to its decimal representation -/
def four_digit_to_decimal (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

theorem digit_multiplication_theorem (A B C D : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D)
  (h_multiplication : three_digit_to_decimal A B C * D = four_digit_to_decimal A B C D) :
  C + D = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_multiplication_theorem_l777_77700


namespace NUMINAMATH_CALUDE_linear_function_characterization_l777_77710

theorem linear_function_characterization (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) → 
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l777_77710


namespace NUMINAMATH_CALUDE_possible_values_of_a_l777_77770

theorem possible_values_of_a (a b x : ℤ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 27*x^3) (h3 : a - b = 2*x) :
  a = (7*x + 5*(6: ℤ).sqrt*x) / 6 ∨ a = (7*x - 5*(6: ℤ).sqrt*x) / 6 :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l777_77770


namespace NUMINAMATH_CALUDE_taxi_cost_proof_l777_77746

/-- The cost per mile for a taxi ride to the airport -/
def cost_per_mile : ℚ := 5 / 14

/-- Mike's distance in miles -/
def mike_distance : ℚ := 28

theorem taxi_cost_proof :
  ∀ (x : ℚ),
  (2.5 + x * mike_distance = 2.5 + 5 + x * 14) →
  x = cost_per_mile := by
  sorry

end NUMINAMATH_CALUDE_taxi_cost_proof_l777_77746


namespace NUMINAMATH_CALUDE_system_solution_proof_l777_77775

theorem system_solution_proof :
  ∃! (x y : ℝ), 
    (2 * x + Real.sqrt (2 * x + 3 * y) - 3 * y = 5) ∧ 
    (4 * x^2 + 2 * x + 3 * y - 9 * y^2 = 32) ∧
    (x = 17/4) ∧ (y = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l777_77775


namespace NUMINAMATH_CALUDE_min_a_value_l777_77755

-- Define the conditions
def p (x : ℝ) : Prop := |x + 1| ≤ 2
def q (x a : ℝ) : Prop := x ≤ a

-- Define what it means for p to be sufficient but not necessary for q
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ¬(∀ x, q x a → p x)

-- State the theorem
theorem min_a_value :
  ∀ a : ℝ, sufficient_not_necessary a → a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l777_77755


namespace NUMINAMATH_CALUDE_profit_maximization_l777_77719

/-- Profit function for computer sales --/
def profit_function (x : ℝ) : ℝ := -50 * x + 15000

/-- Constraint on the number of computers --/
def constraint (x : ℝ) : Prop := 100 / 3 ≤ x ∧ x ≤ 100 / 3

theorem profit_maximization (x : ℝ) :
  constraint x →
  ∀ y, constraint y → profit_function y ≤ profit_function x →
  x = 34 :=
sorry

end NUMINAMATH_CALUDE_profit_maximization_l777_77719


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l777_77749

theorem unique_number_with_three_prime_factors (x n : ℕ) :
  x = 9^n - 1 →
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 7 ∧ q ≠ 7 ∧
    (∀ r : ℕ, Nat.Prime r → r ∣ x → (r = p ∨ r = q ∨ r = 7))) →
  7 ∣ x →
  x = 728 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l777_77749


namespace NUMINAMATH_CALUDE_expression_simplification_l777_77797

theorem expression_simplification (x y : ℝ) (hx : x = -3) (hy : y = -1) :
  (-3 * x^2 - 4*y) - (2 * x^2 - 5*y + 6) + (x^2 - 5*y - 1) = -39 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l777_77797


namespace NUMINAMATH_CALUDE_all_lamps_on_iff_even_l777_77745

/-- Represents the state of a lamp (on or off) -/
inductive LampState
| On : LampState
| Off : LampState

/-- Represents a grid of lamps -/
def LampGrid (n : ℕ) := Fin n → Fin n → LampState

/-- Function to toggle a lamp state -/
def toggleLamp : LampState → LampState
| LampState.On => LampState.Off
| LampState.Off => LampState.On

/-- Function to press a switch at position (i, j) -/
def pressSwitch (grid : LampGrid n) (i j : Fin n) : LampGrid n :=
  fun x y => if x = i ∨ y = j then toggleLamp (grid x y) else grid x y

/-- Predicate to check if all lamps are on -/
def allLampsOn (grid : LampGrid n) : Prop :=
  ∀ i j, grid i j = LampState.On

/-- Main theorem: It's possible to achieve all lamps on iff n is even -/
theorem all_lamps_on_iff_even (n : ℕ) :
  (∀ (initialGrid : LampGrid n), ∃ (switches : List (Fin n × Fin n)),
    allLampsOn (switches.foldl (fun g (i, j) => pressSwitch g i j) initialGrid)) ↔
  Even n :=
sorry

end NUMINAMATH_CALUDE_all_lamps_on_iff_even_l777_77745


namespace NUMINAMATH_CALUDE_sum_of_x₁_and_x₂_l777_77763

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}
def B : Set ℝ := {x | ∃ (x₁ x₂ : ℝ), x₁ ≤ x ∧ x ≤ x₂}

-- Define the conditions for union and intersection
axiom union_condition : A ∪ B = {x | x > -2}
axiom intersection_condition : A ∩ B = {x | 1 < x ∧ x ≤ 3}

-- The theorem to prove
theorem sum_of_x₁_and_x₂ : 
  ∃ (x₁ x₂ : ℝ), (∀ x, x ∈ B ↔ x₁ ≤ x ∧ x ≤ x₂) ∧ x₁ + x₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x₁_and_x₂_l777_77763


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l777_77794

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 6) :
  Real.sqrt (x + 3) + Real.sqrt (y + 3) + Real.sqrt (z + 3) ≤ 3 * Real.sqrt 5 ∧
  ∃ a b c : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 6 ∧
    Real.sqrt (a + 3) + Real.sqrt (b + 3) + Real.sqrt (c + 3) = 3 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l777_77794


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l777_77730

theorem dining_bill_calculation (total : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
  (h_total : total = 184.80)
  (h_tax : tax_rate = 0.10)
  (h_tip : tip_rate = 0.20) : 
  ∃ (food_price : ℝ), 
    food_price * (1 + tax_rate) * (1 + tip_rate) = total ∧ 
    food_price = 140 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l777_77730


namespace NUMINAMATH_CALUDE_customer_payment_l777_77708

def cost_price : ℝ := 6425
def markup_percentage : ℝ := 24

theorem customer_payment (cost : ℝ) (markup : ℝ) :
  cost = cost_price →
  markup = markup_percentage →
  cost * (1 + markup / 100) = 7967 := by
  sorry

end NUMINAMATH_CALUDE_customer_payment_l777_77708


namespace NUMINAMATH_CALUDE_median_average_ratio_l777_77723

theorem median_average_ratio (a b c : ℤ) : 
  a < b → b < c → a = 0 → (a + b + c) / 3 = 4 * b → c / b = 11 := by
  sorry

end NUMINAMATH_CALUDE_median_average_ratio_l777_77723


namespace NUMINAMATH_CALUDE_ratio_equality_l777_77758

theorem ratio_equality (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a / 2 = b / 3) (h5 : b / 3 = c / 5) : (a + b) / (c - a) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l777_77758


namespace NUMINAMATH_CALUDE_split_2017_implies_45_l777_77737

-- Define the sum of consecutive integers from 2 to n
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2 - 1

-- Define the property that 2017 is in the split of m³
def split_contains_2017 (m : ℕ) : Prop :=
  m > 1 ∧ sum_to_n m ≥ 1008 ∧ sum_to_n (m - 1) < 1008

theorem split_2017_implies_45 :
  ∀ m : ℕ, split_contains_2017 m → m = 45 :=
by sorry

end NUMINAMATH_CALUDE_split_2017_implies_45_l777_77737


namespace NUMINAMATH_CALUDE_average_monthly_balance_l777_77727

def monthly_balances : List ℚ := [150, 250, 100, 200, 300]

theorem average_monthly_balance : 
  (monthly_balances.sum / monthly_balances.length : ℚ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l777_77727


namespace NUMINAMATH_CALUDE_wheel_radius_increase_l777_77728

-- Define constants
def inches_per_mile : ℝ := 63360

-- Define the theorem
theorem wheel_radius_increase 
  (D d₁ d₂ r : ℝ) 
  (h₁ : D > 0)
  (h₂ : d₁ > 0)
  (h₃ : d₂ > 0)
  (h₄ : r > 0)
  (h₅ : d₁ > d₂)
  (h₆ : D = d₁) :
  ∃ Δr : ℝ, Δr = (D * (30 * π / inches_per_mile) * inches_per_mile) / (2 * π * d₂) - r :=
by
  sorry

#check wheel_radius_increase

end NUMINAMATH_CALUDE_wheel_radius_increase_l777_77728


namespace NUMINAMATH_CALUDE_not_in_fourth_quadrant_l777_77748

-- Define the linear function
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem: The function f does not pass through the fourth quadrant
theorem not_in_fourth_quadrant :
  ¬ ∃ (x y : ℝ), f x = y ∧ x > 0 ∧ y < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_not_in_fourth_quadrant_l777_77748


namespace NUMINAMATH_CALUDE_estimate_fish_population_l777_77756

/-- Estimates the number of fish in a pond using the capture-recapture method. -/
theorem estimate_fish_population (initial_marked : ℕ) (second_sample : ℕ) (marked_in_second : ℕ) :
  initial_marked = 200 →
  second_sample = 100 →
  marked_in_second = 20 →
  (initial_marked * second_sample) / marked_in_second = 1000 :=
by
  sorry

#check estimate_fish_population

end NUMINAMATH_CALUDE_estimate_fish_population_l777_77756


namespace NUMINAMATH_CALUDE_pens_per_student_l777_77762

theorem pens_per_student (total_pens : ℕ) (total_pencils : ℕ) (max_students : ℕ) :
  total_pens = 100 →
  total_pencils = 50 →
  max_students = 50 →
  total_pens / max_students = 2 :=
by sorry

end NUMINAMATH_CALUDE_pens_per_student_l777_77762


namespace NUMINAMATH_CALUDE_rhombus_c_coordinate_sum_l777_77735

/-- A rhombus with vertices A, B, C, and D in 2D space -/
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The property that A and D are diagonally opposite in the rhombus -/
def diagonallyOpposite (r : Rhombus) : Prop :=
  (r.A.1 + r.D.1) / 2 = (r.B.1 + r.C.1) / 2 ∧
  (r.A.2 + r.D.2) / 2 = (r.B.2 + r.C.2) / 2

/-- The theorem stating that for a rhombus ABCD with given coordinates,
    the sum of coordinates of C is 9 -/
theorem rhombus_c_coordinate_sum :
  ∀ (r : Rhombus),
  r.A = (-3, -2) →
  r.B = (1, -5) →
  r.D = (9, 1) →
  diagonallyOpposite r →
  r.C.1 + r.C.2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_c_coordinate_sum_l777_77735


namespace NUMINAMATH_CALUDE_tan_600_l777_77774

-- Define the tangent function (simplified for this example)
noncomputable def tan (x : ℝ) : ℝ := sorry

-- State the periodicity of tangent
axiom tan_periodic (x : ℝ) : tan (x + 180) = tan x

-- State the value of tan 60°
axiom tan_60 : tan 60 = Real.sqrt 3

-- Theorem to prove
theorem tan_600 : tan 600 = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_600_l777_77774


namespace NUMINAMATH_CALUDE_least_number_divisible_by_multiple_l777_77717

theorem least_number_divisible_by_multiple (n : ℕ) : n = 856 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    (m + 8 = 24 * k₁) ∧ 
    (m + 8 = 32 * k₂) ∧ 
    (m + 8 = 36 * k₃) ∧ 
    (m + 8 = 54 * k₄))) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    (n + 8 = 24 * k₁) ∧ 
    (n + 8 = 32 * k₂) ∧ 
    (n + 8 = 36 * k₃) ∧ 
    (n + 8 = 54 * k₄)) := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_multiple_l777_77717


namespace NUMINAMATH_CALUDE_median_and_altitude_length_l777_77783

/-- An isosceles triangle DEF with DE = DF = 10 and EF = 12 -/
structure IsoscelesTriangle where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side DF -/
  df : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- DE equals DF -/
  de_eq_df : de = df
  /-- DE equals 10 -/
  de_eq_ten : de = 10
  /-- EF equals 12 -/
  ef_eq_twelve : ef = 12

/-- The median DM from vertex D to side EF in the isosceles triangle -/
def median (t : IsoscelesTriangle) : ℝ := sorry

/-- The altitude DH from vertex D to side EF in the isosceles triangle -/
def altitude (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem: Both the median and altitude have length 8 -/
theorem median_and_altitude_length (t : IsoscelesTriangle) : 
  median t = 8 ∧ altitude t = 8 := by sorry

end NUMINAMATH_CALUDE_median_and_altitude_length_l777_77783


namespace NUMINAMATH_CALUDE_equation_solution_l777_77753

theorem equation_solution (x : ℝ) :
  x ≠ 5 ∧ x ≠ 6 →
  ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)) /
  ((x - 5) * (x - 6) * (x - 5)) = 1 ↔ x = 1 ∨ x = 2 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l777_77753


namespace NUMINAMATH_CALUDE_max_y_value_l777_77760

theorem max_y_value (x y : ℤ) (h : 3*x*y + 7*x + 6*y = 20) : 
  y ≤ 16 ∧ ∃ (x' y' : ℤ), 3*x'*y' + 7*x' + 6*y' = 20 ∧ y' = 16 :=
sorry

end NUMINAMATH_CALUDE_max_y_value_l777_77760


namespace NUMINAMATH_CALUDE_equation_roots_l777_77787

theorem equation_roots : 
  let f (x : ℝ) := 18 / (x^2 - 9) - 3 / (x - 3) - 2
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -4.5 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_l777_77787


namespace NUMINAMATH_CALUDE_integral_equals_three_implies_k_equals_four_l777_77733

theorem integral_equals_three_implies_k_equals_four (k : ℝ) : 
  (∫ x in (0:ℝ)..(1:ℝ), 3 * x^2 + k * x) = 3 → k = 4 := by
sorry

end NUMINAMATH_CALUDE_integral_equals_three_implies_k_equals_four_l777_77733


namespace NUMINAMATH_CALUDE_simplify_expression_l777_77754

theorem simplify_expression (n : ℕ) : (3^(n+4) - 3*(3^n)) / (3*(3^(n+3))) = 26 / 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l777_77754


namespace NUMINAMATH_CALUDE_remainder_theorem_l777_77738

theorem remainder_theorem (n : ℤ) : 
  (∃ k : ℤ, 2 * n = 10 * k + 2) → 
  (∃ m : ℤ, n = 20 * m + 1) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l777_77738


namespace NUMINAMATH_CALUDE_zero_mxn_table_l777_77769

/-- Represents a move on the table -/
inductive Move
  | Row (i : Nat)
  | Column (j : Nat)
  | Diagonal (d : Int)

/-- Represents the state of the table -/
def Table (m n : Nat) := Fin m → Fin n → Int

/-- Applies a move to the table -/
def applyMove (t : Table m n) (move : Move) (delta : Int) : Table m n :=
  sorry

/-- Checks if all elements in the table are zero -/
def allZero (t : Table m n) : Prop :=
  sorry

/-- Checks if we can change all numbers to zero in a 3x3 table -/
def canZero3x3 : Prop :=
  ∀ (t : Table 3 3), ∃ (moves : List (Move × Int)), 
    allZero (moves.foldl (fun acc (m, d) => applyMove acc m d) t)

/-- Main theorem: If we can zero any 3x3 table, we can zero any mxn table -/
theorem zero_mxn_table (m n : Nat) (h : canZero3x3) : 
  ∀ (t : Table m n), ∃ (moves : List (Move × Int)), 
    allZero (moves.foldl (fun acc (m, d) => applyMove acc m d) t) :=
  sorry

end NUMINAMATH_CALUDE_zero_mxn_table_l777_77769


namespace NUMINAMATH_CALUDE_g_composition_of_three_l777_77764

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 5*n - 3

theorem g_composition_of_three : g (g (g 3)) = 232 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l777_77764


namespace NUMINAMATH_CALUDE_dice_invisible_dots_l777_77789

theorem dice_invisible_dots : 
  let total_dots_per_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6
  let total_dots : ℕ := 4 * total_dots_per_die
  let visible_numbers : List ℕ := [1, 1, 2, 3, 3, 4, 5, 6]
  let sum_visible : ℕ := visible_numbers.sum
  total_dots - sum_visible = 59 := by
sorry

end NUMINAMATH_CALUDE_dice_invisible_dots_l777_77789


namespace NUMINAMATH_CALUDE_complex_real_condition_l777_77732

theorem complex_real_condition (m : ℝ) : 
  (((m : ℂ) + Complex.I) / (1 - Complex.I)).im = 0 → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l777_77732


namespace NUMINAMATH_CALUDE_jonathan_social_media_time_l777_77781

/-- Calculates the total time spent on social media in a week -/
def social_media_time_per_week (daily_time : ℕ) (days_in_week : ℕ) : ℕ :=
  daily_time * days_in_week

/-- Proves that Jonathan spends 21 hours on social media in a week -/
theorem jonathan_social_media_time :
  social_media_time_per_week 3 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_jonathan_social_media_time_l777_77781


namespace NUMINAMATH_CALUDE_min_intersection_distance_l777_77707

/-- The minimum distance between intersection points of a line and a circle --/
theorem min_intersection_distance (k : ℝ) : 
  let l := {(x, y) : ℝ × ℝ | y = k * x + 1}
  let C := {(x, y) : ℝ × ℝ | x^2 + y^2 - 2*x - 3 = 0}
  ∃ (A B : ℝ × ℝ), A ∈ l ∧ A ∈ C ∧ B ∈ l ∧ B ∈ C ∧
    ∀ (P Q : ℝ × ℝ), P ∈ l ∧ P ∈ C ∧ Q ∈ l ∧ Q ∈ C →
      Real.sqrt 8 ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_intersection_distance_l777_77707


namespace NUMINAMATH_CALUDE_complex_power_abs_one_l777_77761

theorem complex_power_abs_one : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2))^8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_abs_one_l777_77761


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l777_77726

/-- Arithmetic sequence sum -/
def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The problem statement -/
theorem arithmetic_sequence_first_term
  (d : ℚ)
  (h_d : d = 5)
  (h_constant : ∃ (c : ℚ), ∀ (n : ℕ+),
    arithmetic_sum a d (3 * n) / arithmetic_sum a d n = c) :
  a = 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l777_77726


namespace NUMINAMATH_CALUDE_larger_number_problem_l777_77768

theorem larger_number_problem (L S : ℕ) (hL : L > S) : 
  L - S = 1365 → L = 6 * S + 10 → L = 1636 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l777_77768


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l777_77714

/-- A function f(x) = kx - ln x is monotonically increasing on (1/2, +∞) if and only if k ≥ 2 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > (1/2 : ℝ), Monotone (λ x => k * x - Real.log x)) ↔ k ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l777_77714


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l777_77716

theorem fractional_equation_solution :
  ∃ (x : ℚ), (1 - x) / (2 - x) - 1 = (3 * x - 4) / (x - 2) ∧ x = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l777_77716


namespace NUMINAMATH_CALUDE_cost_of_tax_free_items_l777_77778

/-- Given a total cost, sales tax, and tax rate, calculate the cost of tax-free items -/
theorem cost_of_tax_free_items
  (total_cost : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h1 : total_cost = 25)
  (h2 : sales_tax = 0.30)
  (h3 : tax_rate = 0.05)
  : ∃ (tax_free_cost : ℝ), tax_free_cost = 19 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_tax_free_items_l777_77778


namespace NUMINAMATH_CALUDE_marching_band_weight_l777_77712

/-- The total weight carried by the Oprah Winfrey High School marching band -/
def total_weight : ℕ :=
  let trumpet_weight := 5 + 3
  let clarinet_weight := 5 + 3
  let trombone_weight := 10 + 4
  let tuba_weight := 20 + 5
  let drummer_weight := 15 + 6
  let percussionist_weight := 8 + 3
  let trumpet_count := 6
  let clarinet_count := 9
  let trombone_count := 8
  let tuba_count := 3
  let drummer_count := 2
  let percussionist_count := 4
  trumpet_count * trumpet_weight +
  clarinet_count * clarinet_weight +
  trombone_count * trombone_weight +
  tuba_count * tuba_weight +
  drummer_count * drummer_weight +
  percussionist_count * percussionist_weight

theorem marching_band_weight : total_weight = 393 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_weight_l777_77712


namespace NUMINAMATH_CALUDE_trivia_team_members_l777_77709

/-- Represents a trivia team with its total members and points scored. -/
structure TriviaTeam where
  totalMembers : ℕ
  absentMembers : ℕ
  pointsPerMember : ℕ
  totalPoints : ℕ

/-- Theorem stating the total members in the trivia team -/
theorem trivia_team_members (team : TriviaTeam)
  (h1 : team.absentMembers = 4)
  (h2 : team.pointsPerMember = 8)
  (h3 : team.totalPoints = 64)
  : team.totalMembers = 12 := by
  sorry

#check trivia_team_members

end NUMINAMATH_CALUDE_trivia_team_members_l777_77709


namespace NUMINAMATH_CALUDE_natural_number_divisibility_l777_77734

theorem natural_number_divisibility (a b n : ℕ) 
  (h : ∀ (k : ℕ), k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by sorry

end NUMINAMATH_CALUDE_natural_number_divisibility_l777_77734


namespace NUMINAMATH_CALUDE_organization_size_after_five_years_l777_77788

def organization_growth (initial_members : ℕ) (leaders : ℕ) (recruitment : ℕ) (years : ℕ) : ℕ :=
  let rec growth (k : ℕ) (members : ℕ) : ℕ :=
    if k = 0 then
      members
    else
      growth (k - 1) (4 * members - 18)
  growth years initial_members

theorem organization_size_after_five_years :
  organization_growth 20 6 3 5 = 14382 :=
by sorry

end NUMINAMATH_CALUDE_organization_size_after_five_years_l777_77788


namespace NUMINAMATH_CALUDE_biology_enrollment_percentage_l777_77741

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) :
  total_students = 840 →
  not_enrolled = 546 →
  (((total_students - not_enrolled : ℚ) / total_students) * 100 : ℚ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_biology_enrollment_percentage_l777_77741


namespace NUMINAMATH_CALUDE_polar_curve_is_parabola_l777_77792

/-- The curve defined by the polar equation r = 1 / (1 - sin θ) is a parabola -/
theorem polar_curve_is_parabola :
  ∀ r θ x y : ℝ,
  (r = 1 / (1 - Real.sin θ)) →
  (x = r * Real.cos θ) →
  (y = r * Real.sin θ) →
  ∃ a b : ℝ, x^2 = a * y + b :=
by sorry

end NUMINAMATH_CALUDE_polar_curve_is_parabola_l777_77792


namespace NUMINAMATH_CALUDE_hyperbola_equation_l777_77798

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1, eccentricity e = 5/4, 
    and right focus F₂(5,0), prove that the equation of C is x²/16 - y²/9 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →  -- Equation of hyperbola C
  (a/b)^2 + 1 = (5/4)^2 →               -- Eccentricity e = 5/4
  5^2 = a^2 + b^2 →                     -- Right focus F₂(5,0)
  a^2 = 16 ∧ b^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l777_77798


namespace NUMINAMATH_CALUDE_pet_store_cages_l777_77777

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 45)
  (h2 : sold_puppies = 39)
  (h3 : puppies_per_cage = 2) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l777_77777


namespace NUMINAMATH_CALUDE_inequality_proof_l777_77720

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1 / 4) :
  (16 * a * c + a / (c^2 * b) + 16 * c / (a^2 * d) + 4 / (a * c)) * 
  (b * d + b / (256 * d^2 * c) + d / (b^2 * a) + 1 / (64 * b * d)) ≥ 81 / 4 ∧
  (16 * a * c + a / (c^2 * b) + 16 * c / (a^2 * d) + 4 / (a * c)) * 
  (b * d + b / (256 * d^2 * c) + d / (b^2 * a) + 1 / (64 * b * d)) = 81 / 4 ↔ 
  a = 2 ∧ b = 1 ∧ c = 1 / 2 ∧ d = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l777_77720


namespace NUMINAMATH_CALUDE_exponent_multiplication_l777_77731

theorem exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l777_77731


namespace NUMINAMATH_CALUDE_stratified_sampling_is_appropriate_l777_77742

/-- Represents a stratum in a population --/
structure Stratum where
  size : ℕ
  characteristic : ℝ

/-- Represents a population with three strata --/
structure Population where
  strata : Fin 3 → Stratum
  total_size : ℕ
  avg_characteristic : ℝ

/-- Represents a sampling method --/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Determines if a sampling method is appropriate for a given population and sample size --/
def is_appropriate_sampling_method (pop : Population) (sample_size : ℕ) (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.StratifiedSampling => true
  | _ => false

/-- Theorem stating that stratified sampling is the appropriate method for the given scenario --/
theorem stratified_sampling_is_appropriate (pop : Population) (sample_size : ℕ) :
  is_appropriate_sampling_method pop sample_size SamplingMethod.StratifiedSampling :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_is_appropriate_l777_77742


namespace NUMINAMATH_CALUDE_bike_tractor_speed_ratio_l777_77791

/-- The ratio of the speed of a bike to the speed of a tractor -/
theorem bike_tractor_speed_ratio :
  ∀ (speed_car speed_bike speed_tractor : ℝ),
  speed_car = (9/5) * speed_bike →
  speed_car = 450 / 5 →
  speed_tractor = 575 / 23 →
  ∃ (k : ℝ), speed_bike = k * speed_tractor →
  speed_bike / speed_tractor = 2 := by
sorry

end NUMINAMATH_CALUDE_bike_tractor_speed_ratio_l777_77791


namespace NUMINAMATH_CALUDE_sam_watermelons_l777_77704

def total_watermelons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

theorem sam_watermelons : 
  let initial := 4
  let additional := 3
  total_watermelons initial additional = 7 := by
  sorry

end NUMINAMATH_CALUDE_sam_watermelons_l777_77704


namespace NUMINAMATH_CALUDE_unique_perfect_square_sum_l777_77773

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def distinct_perfect_square_sum (a b c : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = 100

theorem unique_perfect_square_sum :
  ∃! (s : Finset (Finset ℕ)), s.card = 1 ∧
    ∀ t ∈ s, t.card = 3 ∧
      (∃ a b c, {a, b, c} = t ∧ distinct_perfect_square_sum a b c) :=
sorry

end NUMINAMATH_CALUDE_unique_perfect_square_sum_l777_77773


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l777_77702

theorem restaurant_bill_calculation (appetizer_cost : ℝ) (entree_cost : ℝ) (num_entrees : ℕ) (tip_percentage : ℝ) : 
  appetizer_cost = 10 ∧ 
  entree_cost = 20 ∧ 
  num_entrees = 4 ∧ 
  tip_percentage = 0.2 → 
  appetizer_cost + (entree_cost * num_entrees) + (appetizer_cost + entree_cost * num_entrees) * tip_percentage = 108 := by
  sorry

#check restaurant_bill_calculation

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l777_77702


namespace NUMINAMATH_CALUDE_jack_position_change_l777_77739

-- Define the constants from the problem
def flights_up : ℕ := 3
def flights_down : ℕ := 6
def steps_per_flight : ℕ := 12
def inches_per_step : ℕ := 8
def inches_per_foot : ℕ := 12

-- Define the function to calculate the net change in position
def net_position_change : ℚ :=
  (flights_down - flights_up) * steps_per_flight * inches_per_step / inches_per_foot

-- Theorem statement
theorem jack_position_change :
  net_position_change = 24 := by sorry

end NUMINAMATH_CALUDE_jack_position_change_l777_77739


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l777_77771

theorem sqrt_sum_equality : 
  (Real.sqrt 54 - Real.sqrt 27) + Real.sqrt 3 + 8 * Real.sqrt (1/2) = 
  3 * Real.sqrt 6 - 2 * Real.sqrt 3 + 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l777_77771


namespace NUMINAMATH_CALUDE_fundraiser_goal_is_750_l777_77724

/-- Represents the fundraiser goal calculation --/
def fundraiser_goal (bronze_families : ℕ) (silver_families : ℕ) (gold_families : ℕ) 
  (bronze_donation : ℕ) (silver_donation : ℕ) (gold_donation : ℕ) 
  (final_day_goal : ℕ) : ℕ :=
  bronze_families * bronze_donation + 
  silver_families * silver_donation + 
  gold_families * gold_donation + 
  final_day_goal

/-- Theorem stating that the fundraiser goal is $750 --/
theorem fundraiser_goal_is_750 : 
  fundraiser_goal 10 7 1 25 50 100 50 = 750 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_goal_is_750_l777_77724


namespace NUMINAMATH_CALUDE_prob_first_second_given_two_fail_l777_77706

-- Define the failure probabilities for each component
def p1 : ℝ := 0.2
def p2 : ℝ := 0.4
def p3 : ℝ := 0.3

-- Define the probability of two components failing
def prob_two_fail : ℝ := p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3

-- Define the probability of the first and second components failing
def prob_first_second_fail : ℝ := p1 * p2 * (1 - p3)

-- Theorem statement
theorem prob_first_second_given_two_fail : 
  prob_first_second_fail / prob_two_fail = 0.3 := by sorry

end NUMINAMATH_CALUDE_prob_first_second_given_two_fail_l777_77706


namespace NUMINAMATH_CALUDE_product_prs_is_27_l777_77757

theorem product_prs_is_27 (p r s : ℕ) 
  (eq1 : 4^p + 4^3 = 272)
  (eq2 : 3^r + 27 = 54)
  (eq3 : 2^(s+2) + 10 = 42) :
  p * r * s = 27 := by
  sorry

end NUMINAMATH_CALUDE_product_prs_is_27_l777_77757
