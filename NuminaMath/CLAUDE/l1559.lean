import Mathlib

namespace greatest_three_digit_multiple_of_17_l1559_155942

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l1559_155942


namespace total_cash_is_correct_l1559_155963

/-- Calculates the total cash realized from three stocks after accounting for brokerage fees. -/
def total_cash_realized (stock_a_proceeds stock_b_proceeds stock_c_proceeds : ℝ)
                        (stock_a_brokerage_rate stock_b_brokerage_rate stock_c_brokerage_rate : ℝ) : ℝ :=
  let stock_a_cash := stock_a_proceeds * (1 - stock_a_brokerage_rate)
  let stock_b_cash := stock_b_proceeds * (1 - stock_b_brokerage_rate)
  let stock_c_cash := stock_c_proceeds * (1 - stock_c_brokerage_rate)
  stock_a_cash + stock_b_cash + stock_c_cash

/-- Theorem stating that the total cash realized from the given stock sales is equal to 463.578625. -/
theorem total_cash_is_correct : 
  total_cash_realized 107.25 155.40 203.50 (1/400) (1/200) (3/400) = 463.578625 := by
  sorry

end total_cash_is_correct_l1559_155963


namespace fraction_value_l1559_155915

theorem fraction_value (x y : ℝ) (h : 2 * x = -y) :
  x * y / (x^2 - y^2) = 2 / 3 :=
by sorry

end fraction_value_l1559_155915


namespace barbara_candies_l1559_155923

/-- Calculates the remaining number of candies Barbara has -/
def remaining_candies (initial : ℝ) (used : ℝ) (received : ℝ) (eaten : ℝ) : ℝ :=
  initial - used + received - eaten

/-- Proves that Barbara has 18.4 candies left -/
theorem barbara_candies : remaining_candies 18.5 4.2 6.8 2.7 = 18.4 := by
  sorry

end barbara_candies_l1559_155923


namespace new_supervisor_salary_range_l1559_155990

theorem new_supervisor_salary_range (
  old_average : ℝ) 
  (old_supervisor_salary : ℝ) 
  (new_average : ℝ) 
  (min_worker_salary : ℝ) 
  (max_worker_salary : ℝ) 
  (min_supervisor_salary : ℝ) 
  (max_supervisor_salary : ℝ) :
  old_average = 430 →
  old_supervisor_salary = 870 →
  new_average = 410 →
  min_worker_salary = 300 →
  max_worker_salary = 500 →
  min_supervisor_salary = 800 →
  max_supervisor_salary = 1100 →
  ∃ (new_supervisor_salary : ℝ),
    min_supervisor_salary ≤ new_supervisor_salary ∧
    new_supervisor_salary ≤ max_supervisor_salary ∧
    (9 * new_average - 8 * old_average + old_supervisor_salary = new_supervisor_salary) :=
by sorry

end new_supervisor_salary_range_l1559_155990


namespace import_value_calculation_l1559_155934

theorem import_value_calculation (tax_free_limit : ℝ) (tax_rate : ℝ) (tax_paid : ℝ) : 
  tax_free_limit = 500 →
  tax_rate = 0.08 →
  tax_paid = 18.40 →
  ∃ total_value : ℝ, total_value = 730 ∧ tax_paid = tax_rate * (total_value - tax_free_limit) :=
by sorry

end import_value_calculation_l1559_155934


namespace fireworks_count_l1559_155961

/-- The number of fireworks Henry and his friend have now -/
def total_fireworks (henry_new : ℕ) (friend_new : ℕ) (last_year : ℕ) : ℕ :=
  henry_new + friend_new + last_year

/-- Proof that Henry and his friend have 11 fireworks in total -/
theorem fireworks_count : total_fireworks 2 3 6 = 11 := by
  sorry

end fireworks_count_l1559_155961


namespace correct_calculation_l1559_155991

theorem correct_calculation (a : ℝ) : 4 * a - (-7 * a) = 11 * a := by
  sorry

end correct_calculation_l1559_155991


namespace steve_final_marbles_l1559_155979

/- Define the initial number of marbles for each person -/
def sam_initial : ℕ := 14
def steve_initial : ℕ := 7
def sally_initial : ℕ := 9

/- Define the number of marbles Sam gives away -/
def marbles_given : ℕ := 3

/- Define Sam's final number of marbles -/
def sam_final : ℕ := 8

/- Theorem to prove -/
theorem steve_final_marbles :
  /- Conditions -/
  (sam_initial = 2 * steve_initial) →
  (sally_initial = sam_initial - 5) →
  (sam_final = sam_initial - 2 * marbles_given) →
  /- Conclusion -/
  (steve_initial + marbles_given = 10) := by
sorry

end steve_final_marbles_l1559_155979


namespace stream_speed_l1559_155940

/-- Given a boat that travels at 14 km/hr in still water and covers 72 km downstream in 3.6 hours,
    prove that the speed of the stream is 6 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 14 →
  distance = 72 →
  time = 3.6 →
  stream_speed = (distance / time) - boat_speed →
  stream_speed = 6 := by
sorry

end stream_speed_l1559_155940


namespace negative_sqrt_product_l1559_155967

theorem negative_sqrt_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  -Real.sqrt a * Real.sqrt b = -Real.sqrt (a * b) := by
  sorry

end negative_sqrt_product_l1559_155967


namespace third_card_value_l1559_155960

theorem third_card_value (a b c : ℕ) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h2 : 1 ≤ a ∧ a ≤ 13)
  (h3 : 1 ≤ b ∧ b ≤ 13)
  (h4 : 1 ≤ c ∧ c ≤ 13)
  (h5 : a + b = 25)
  (h6 : b + c = 13) :
  c = 1 := by
sorry

end third_card_value_l1559_155960


namespace sin_210_degrees_l1559_155993

theorem sin_210_degrees :
  Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l1559_155993


namespace horatio_sonnets_count_l1559_155959

/-- Represents the number of sonnets Horatio wrote -/
def total_sonnets : ℕ := 12

/-- Represents the number of lines in each sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of sonnets Horatio's lady fair heard -/
def sonnets_heard : ℕ := 7

/-- Represents the number of romantic lines that were never heard -/
def unheard_lines : ℕ := 70

/-- Theorem stating that the total number of sonnets Horatio wrote is correct -/
theorem horatio_sonnets_count :
  total_sonnets = sonnets_heard + (unheard_lines / lines_per_sonnet) := by
  sorry

end horatio_sonnets_count_l1559_155959


namespace arithmetic_proof_l1559_155975

theorem arithmetic_proof : -3 + 15 - (-8) = 20 := by
  sorry

end arithmetic_proof_l1559_155975


namespace complex_multiplication_sum_l1559_155939

theorem complex_multiplication_sum (z a b : ℂ) : 
  z = 3 + Complex.I ∧ Complex.I * z = a + b * Complex.I → a + b = 2 := by
  sorry

end complex_multiplication_sum_l1559_155939


namespace regular_square_pyramid_side_edge_l1559_155914

theorem regular_square_pyramid_side_edge 
  (base_edge : ℝ) 
  (volume : ℝ) 
  (h : base_edge = 4 * Real.sqrt 2) 
  (h' : volume = 32) : 
  ∃ (side_edge : ℝ), side_edge = 5 := by
sorry

end regular_square_pyramid_side_edge_l1559_155914


namespace ellipse_equation_l1559_155958

/-- An ellipse with center at the origin, foci on the x-axis, eccentricity 1/2, 
    and the perimeter of triangle PF₁F₂ equal to 12 -/
structure Ellipse where
  /-- The eccentricity of the ellipse -/
  e : ℝ
  /-- The perimeter of triangle PF₁F₂ -/
  perimeter : ℝ
  /-- The eccentricity is 1/2 -/
  h_e : e = 1/2
  /-- The perimeter is 12 -/
  h_perimeter : perimeter = 12

/-- The standard equation of the ellipse -/
def standardEquation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2/16 + y^2/12 = 1

/-- Theorem stating that the given ellipse satisfies the standard equation -/
theorem ellipse_equation (E : Ellipse) (x y : ℝ) : 
  standardEquation E x y := by
  sorry

end ellipse_equation_l1559_155958


namespace container_capacity_proof_l1559_155955

/-- The capacity of a container in liters -/
def container_capacity : ℝ := 100

/-- The initial fill level of the container as a percentage -/
def initial_fill : ℝ := 30

/-- The final fill level of the container as a percentage -/
def final_fill : ℝ := 75

/-- The amount of water added to the container in liters -/
def water_added : ℝ := 45

theorem container_capacity_proof :
  (final_fill / 100 * container_capacity) - (initial_fill / 100 * container_capacity) = water_added :=
sorry

end container_capacity_proof_l1559_155955


namespace solve_for_y_l1559_155946

theorem solve_for_y (x y : ℝ) (h : 2 * x - 7 * y = 8) : y = (2 * x - 8) / 7 := by
  sorry

end solve_for_y_l1559_155946


namespace line_plane_perpendicular_parallel_l1559_155964

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (not_subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel
  (m n l : Line) (α : Plane)
  (distinct : m ≠ n ∧ m ≠ l ∧ n ≠ l)
  (perp_lm : perpendicular l m)
  (not_in_plane : not_subset m α)
  (perp_lα : perpendicular_plane l α) :
  parallel_plane m α :=
sorry

end line_plane_perpendicular_parallel_l1559_155964


namespace difference_of_squares_l1559_155911

theorem difference_of_squares (a : ℝ) : a^2 - 100 = (a + 10) * (a - 10) := by
  sorry

end difference_of_squares_l1559_155911


namespace current_rate_calculation_l1559_155980

theorem current_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) :
  boat_speed = 21 →
  distance = 6.283333333333333 →
  time = 13 / 60 →
  ∃ current_rate : ℝ, 
    distance = (boat_speed + current_rate) * time ∧
    current_rate = 8 := by
  sorry

end current_rate_calculation_l1559_155980


namespace cloth_cost_price_l1559_155936

def cloth_problem (total_meters : ℝ) (total_price : ℝ) (loss_per_meter : ℝ) (discount_rate : ℝ) : Prop :=
  let selling_price_per_meter : ℝ := total_price / total_meters
  let discounted_price_per_meter : ℝ := selling_price_per_meter * (1 - discount_rate)
  let cost_price_per_meter : ℝ := discounted_price_per_meter + loss_per_meter
  cost_price_per_meter = 130

theorem cloth_cost_price :
  cloth_problem 450 45000 40 0.1 :=
by
  sorry

end cloth_cost_price_l1559_155936


namespace not_isosceles_l1559_155986

/-- A set of three distinct real numbers that can form the sides of a triangle -/
structure TriangleSet where
  a : ℝ
  b : ℝ
  c : ℝ
  distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The triangle formed by a TriangleSet cannot be isosceles -/
theorem not_isosceles (S : TriangleSet) : ¬(S.a = S.b ∨ S.b = S.c ∨ S.c = S.a) :=
sorry

end not_isosceles_l1559_155986


namespace algebraic_expression_value_l1559_155952

theorem algebraic_expression_value (a b : ℝ) (h : a - 3*b = -3) :
  5 - a + 3*b = 8 := by sorry

end algebraic_expression_value_l1559_155952


namespace linear_equation_solution_l1559_155909

/-- A linear function passing through (-4, 3) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 5

/-- The point (-4, 3) lies on the graph of f -/
def point_condition (a : ℝ) : Prop := f a (-4) = 3

/-- The equation ax - 5 = 3 -/
def equation (a x : ℝ) : Prop := a * x - 5 = 3

theorem linear_equation_solution (a : ℝ) (h : point_condition a) :
  ∃ x, equation a x ∧ x = -4 := by sorry

end linear_equation_solution_l1559_155909


namespace polynomial_coefficient_sum_difference_squares_l1559_155926

theorem polynomial_coefficient_sum_difference_squares (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x + Real.sqrt 3) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end polynomial_coefficient_sum_difference_squares_l1559_155926


namespace hidden_dots_count_l1559_155928

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The sum of dots on all faces of a standard die -/
def sumOfDots : ℕ := 21

/-- The number of dice in the stack -/
def numberOfDice : ℕ := 3

/-- The visible faces on the stack of dice -/
def visibleFaces : List ℕ := [1, 3, 4, 5, 6]

/-- The total number of faces on the stack of dice -/
def totalFaces : ℕ := 18

/-- The number of hidden faces on the stack of dice -/
def hiddenFaces : ℕ := 13

/-- Theorem stating that the total number of hidden dots is 44 -/
theorem hidden_dots_count :
  (numberOfDice * sumOfDots) - (visibleFaces.sum) = 44 := by sorry

end hidden_dots_count_l1559_155928


namespace binomial_coefficient_20_10_l1559_155930

theorem binomial_coefficient_20_10 (h1 : Nat.choose 17 7 = 19448)
                                   (h2 : Nat.choose 17 8 = 24310)
                                   (h3 : Nat.choose 17 9 = 24310) :
  Nat.choose 20 10 = 111826 := by
  sorry

end binomial_coefficient_20_10_l1559_155930


namespace range_of_m_solution_set_correct_l1559_155950

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (2*m + 1) * x + 2

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∃ x y, x > 1 ∧ y < 1 ∧ f m x = 0 ∧ f m y = 0) → -1 < m ∧ m < 0 :=
sorry

-- Define the solution set for f(x) ≤ 0
def solution_set (m : ℝ) : Set ℝ :=
  if m = 0 then { x | x ≤ -2 }
  else if m < 0 then { x | -2 ≤ x ∧ x ≤ -1/m }
  else if 0 < m ∧ m < 1/2 then { x | -1/m ≤ x ∧ x ≤ -2 }
  else if m = 1/2 then { -2 }
  else { x | -2 ≤ x ∧ x ≤ -1/m }

-- Theorem for the solution set
theorem solution_set_correct (m : ℝ) (x : ℝ) : 
  x ∈ solution_set m ↔ f m x ≤ 0 :=
sorry

end range_of_m_solution_set_correct_l1559_155950


namespace quadratic_function_max_min_difference_l1559_155925

-- Define the function f(x) = x^2 + bx + c
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- Define the theorem
theorem quadratic_function_max_min_difference (b c : ℝ) :
  (∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → f b c x ≤ max) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ f b c x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → min ≤ f b c x) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ f b c x = min) ∧
    max - min = 25) →
  b = -4 ∨ b = -12 :=
by sorry

end quadratic_function_max_min_difference_l1559_155925


namespace line_perp_plane_implies_parallel_l1559_155906

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpLine : Line → Line → Prop)
variable (perpLinePlane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the "contained in" relation
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_parallel
  (m n : Line) (α : Plane)
  (h1 : perpLinePlane m α)
  (h2 : perpLine n m)
  (h3 : ¬ containedIn n α) :
  parallel n α :=
sorry

end line_perp_plane_implies_parallel_l1559_155906


namespace largest_n_proof_l1559_155988

/-- Binary operation @ defined as n @ n = n - (n * 5) -/
def binary_op (n : ℤ) : ℤ := n - (n * 5)

/-- The largest positive integer n such that n @ n < 21 -/
def largest_n : ℕ := 1

theorem largest_n_proof :
  (∀ (m : ℕ), m > largest_n → binary_op m ≥ 21) ∧
  binary_op largest_n < 21 :=
sorry

end largest_n_proof_l1559_155988


namespace array_sum_theorem_l1559_155924

-- Define the array structure
def array_sum (p : ℕ) : ℚ := 3 * p^2 / ((3*p - 1) * (p - 1))

-- Define the result of (m+n) mod 2009
def result_mod_2009 (p : ℕ) : ℕ :=
  let m : ℕ := 3 * p^2
  let n : ℕ := (3*p - 1) * (p - 1)
  (m + n) % 2009

-- The main theorem
theorem array_sum_theorem :
  array_sum 2008 = 3 * 2008^2 / ((3*2008 - 1) * (2008 - 1)) ∧
  result_mod_2009 2008 = 1 := by sorry

end array_sum_theorem_l1559_155924


namespace rent_increase_is_thirty_percent_l1559_155929

/-- Calculates the percentage increase in rent given last year's expenses and this year's total increase --/
def rent_increase_percentage (last_year_rent : ℕ) (last_year_food : ℕ) (last_year_insurance : ℕ) (food_increase_percent : ℕ) (insurance_multiplier : ℕ) (total_yearly_increase : ℕ) : ℕ :=
  let last_year_monthly_total := last_year_rent + last_year_food + last_year_insurance
  let this_year_food := last_year_food + (last_year_food * food_increase_percent) / 100
  let this_year_insurance := last_year_insurance * insurance_multiplier
  let monthly_increase_without_rent := (this_year_food + this_year_insurance) - (last_year_food + last_year_insurance)
  let yearly_increase_without_rent := monthly_increase_without_rent * 12
  let rent_increase := total_yearly_increase - yearly_increase_without_rent
  (rent_increase * 100) / (last_year_rent * 12)

theorem rent_increase_is_thirty_percent :
  rent_increase_percentage 1000 200 100 50 3 7200 = 30 := by
  sorry

end rent_increase_is_thirty_percent_l1559_155929


namespace ranch_problem_l1559_155907

theorem ranch_problem (ponies horses : ℕ) (horseshoe_fraction : ℚ) :
  ponies + horses = 163 →
  horses = ponies + 3 →
  ∃ (iceland_ponies : ℕ), iceland_ponies = (5 : ℚ) / 8 * horseshoe_fraction * ponies →
  horseshoe_fraction = 1 / 10 :=
by sorry

end ranch_problem_l1559_155907


namespace f_monotone_increasing_l1559_155965

def f (x : ℝ) : ℝ := 3 * x + 1

theorem f_monotone_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end f_monotone_increasing_l1559_155965


namespace polygon_sides_count_l1559_155927

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: If 3n minus the number of diagonals equals 6, then n equals 6 -/
theorem polygon_sides_count (n : ℕ) (h : 3 * n - num_diagonals n = 6) : n = 6 := by
  sorry

end polygon_sides_count_l1559_155927


namespace choir_size_proof_choir_size_minimum_l1559_155962

theorem choir_size_proof : 
  ∀ n : ℕ, (n % 9 = 0 ∧ n % 11 = 0 ∧ n % 13 = 0 ∧ n % 10 = 0) → n ≥ 12870 :=
by
  sorry

theorem choir_size_minimum : 
  12870 % 9 = 0 ∧ 12870 % 11 = 0 ∧ 12870 % 13 = 0 ∧ 12870 % 10 = 0 :=
by
  sorry

end choir_size_proof_choir_size_minimum_l1559_155962


namespace distance_is_134_l1559_155910

/-- The distance between two girls walking in opposite directions after 12 hours -/
def distance_between_girls : ℝ :=
  let girl1_speed1 : ℝ := 7
  let girl1_time1 : ℝ := 6
  let girl1_speed2 : ℝ := 10
  let girl1_time2 : ℝ := 6
  let girl2_speed1 : ℝ := 3
  let girl2_time1 : ℝ := 8
  let girl2_speed2 : ℝ := 2
  let girl2_time2 : ℝ := 4
  let girl1_distance : ℝ := girl1_speed1 * girl1_time1 + girl1_speed2 * girl1_time2
  let girl2_distance : ℝ := girl2_speed1 * girl2_time1 + girl2_speed2 * girl2_time2
  girl1_distance + girl2_distance

/-- Theorem stating that the distance between the girls after 12 hours is 134 km -/
theorem distance_is_134 : distance_between_girls = 134 := by
  sorry

end distance_is_134_l1559_155910


namespace f_greater_than_one_range_l1559_155920

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

theorem f_greater_than_one_range :
  {x : ℝ | f x > 1} = {x : ℝ | x > 1 ∨ x < -1} := by sorry

end f_greater_than_one_range_l1559_155920


namespace class_height_ratio_l1559_155992

theorem class_height_ratio :
  ∀ (x y : ℕ),
  x > 0 → y > 0 →
  149 * x + 144 * y = 147 * (x + y) →
  (x : ℚ) / y = 3 / 2 :=
by
  sorry

end class_height_ratio_l1559_155992


namespace hyperbola_to_ellipse_l1559_155933

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 3 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_to_ellipse :
  ∀ (x y : ℝ),
  hyperbola x y →
  (∃ (a b c : ℝ),
    a = 2 * Real.sqrt 3 ∧
    c = 3 ∧
    b^2 = a^2 - c^2 ∧
    ellipse x y) :=
sorry

end hyperbola_to_ellipse_l1559_155933


namespace product_inspection_theorem_l1559_155937

/-- Represents a collection of products -/
structure ProductCollection where
  total : ℕ
  selected : ℕ

/-- Defines the concept of a population in statistics -/
def population (pc : ProductCollection) : ℕ := pc.total

/-- Defines the concept of a sample in statistics -/
def sample (pc : ProductCollection) : ℕ := pc.selected

/-- Defines the concept of sample size in statistics -/
def sampleSize (pc : ProductCollection) : ℕ := pc.selected

theorem product_inspection_theorem (pc : ProductCollection) 
  (h1 : pc.total = 80) 
  (h2 : pc.selected = 10) 
  (h3 : pc.selected ≤ pc.total) : 
  population pc = 80 ∧ 
  sampleSize pc = 10 ∧ 
  sample pc ≤ population pc := by
  sorry


end product_inspection_theorem_l1559_155937


namespace unique_solution_quadratic_system_l1559_155997

theorem unique_solution_quadratic_system (x : ℚ) :
  (6 * x^2 + 19 * x - 7 = 0) ∧ (18 * x^2 + 47 * x - 21 = 0) → x = 1/3 := by
  sorry

end unique_solution_quadratic_system_l1559_155997


namespace students_only_swimming_l1559_155954

/-- The number of students only participating in swimming in a sports day scenario --/
theorem students_only_swimming (total : ℕ) (swimming : ℕ) (track : ℕ) (ball : ℕ) 
  (swim_track : ℕ) (swim_ball : ℕ) : 
  total = 28 → 
  swimming = 15 → 
  track = 8 → 
  ball = 14 → 
  swim_track = 3 → 
  swim_ball = 3 → 
  swimming - (swim_track + swim_ball) = 9 := by
  sorry

#check students_only_swimming

end students_only_swimming_l1559_155954


namespace inequality_proof_l1559_155902

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) : 
  (x^4 + y^2*z^2)/(x^(5/2)*(y+z)) + 
  (y^4 + z^2*x^2)/(y^(5/2)*(z+x)) + 
  (z^4 + y^2*x^2)/(z^(5/2)*(y+x)) ≥ 1 := by
sorry

end inequality_proof_l1559_155902


namespace room_population_lower_limit_l1559_155948

theorem room_population_lower_limit (total : ℕ) (under_21 : ℕ) (over_65 : ℕ) : 
  under_21 = 30 →
  under_21 = (3 : ℚ) / 7 * total →
  over_65 = (5 : ℚ) / 10 * total →
  ∃ (upper : ℕ), total ∈ Set.Icc total upper →
  70 ≤ total :=
by sorry

end room_population_lower_limit_l1559_155948


namespace train_length_proof_l1559_155943

theorem train_length_proof (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ) 
  (h1 : bridge_length = 800)
  (h2 : bridge_time = 45)
  (h3 : post_time = 15) :
  ∃ train_length : ℝ, train_length = 400 ∧ 
  train_length / post_time = (train_length + bridge_length) / bridge_time :=
by sorry

end train_length_proof_l1559_155943


namespace largest_four_digit_sum_20_l1559_155978

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
sorry

end largest_four_digit_sum_20_l1559_155978


namespace rational_solutions_quadratic_l1559_155913

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 22 * x + k = 0) ↔ k = 11 := by
  sorry

end rational_solutions_quadratic_l1559_155913


namespace rogers_final_money_rogers_final_money_proof_l1559_155900

/-- Calculates Roger's final amount of money after various transactions -/
theorem rogers_final_money (initial_amount : ℝ) (birthday_money : ℝ) (found_money : ℝ) 
  (game_cost : ℝ) (gift_percentage : ℝ) : ℝ :=
  let total_before_spending := initial_amount + birthday_money + found_money
  let after_game_purchase := total_before_spending - game_cost
  let gift_cost := gift_percentage * after_game_purchase
  let final_amount := after_game_purchase - gift_cost
  final_amount

/-- Proves that Roger's final amount of money is $106.25 -/
theorem rogers_final_money_proof :
  rogers_final_money 84 56 20 35 0.15 = 106.25 := by
  sorry

end rogers_final_money_rogers_final_money_proof_l1559_155900


namespace randys_fathers_biscuits_l1559_155982

/-- Proves that Randy's father gave him 13 biscuits given the initial conditions and final result. -/
theorem randys_fathers_biscuits :
  ∀ (initial mother_gave brother_ate final father_gave : ℕ),
  initial = 32 →
  mother_gave = 15 →
  brother_ate = 20 →
  final = 40 →
  initial + mother_gave + father_gave - brother_ate = final →
  father_gave = 13 := by
  sorry

end randys_fathers_biscuits_l1559_155982


namespace max_value_A_l1559_155931

theorem max_value_A (x y z : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1) :
  (Real.sqrt (8 * x^4 + y) + Real.sqrt (8 * y^4 + z) + Real.sqrt (8 * z^4 + x) - 3) / (x + y + z) ≤ 2 :=
by sorry

end max_value_A_l1559_155931


namespace lines_skew_and_parallel_l1559_155996

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_skew_and_parallel (a b c : Line) 
  (h1 : skew a b) (h2 : parallel c a) : skew c b := by
  sorry

end lines_skew_and_parallel_l1559_155996


namespace matrix_inverse_proof_l1559_155971

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

def inverse_A : Matrix (Fin 2) (Fin 2) ℚ := !![9/46, -5/46; 1/23, 2/23]

theorem matrix_inverse_proof :
  (Matrix.det A ≠ 0 ∧ A * inverse_A = 1 ∧ inverse_A * A = 1) ∨
  (Matrix.det A = 0 ∧ inverse_A = 0) := by
  sorry

end matrix_inverse_proof_l1559_155971


namespace base_7_even_digits_403_l1559_155921

/-- Counts the number of even digits in a base-7 number -/
def countEvenDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 -/
def toBase7 (n : ℕ) : List ℕ := sorry

theorem base_7_even_digits_403 :
  let base7Repr := toBase7 403
  countEvenDigitsBase7 403 = 1 := by sorry

end base_7_even_digits_403_l1559_155921


namespace pizza_combinations_six_toppings_l1559_155917

/-- The number of different one- and two-topping pizzas that can be ordered from a pizza parlor with a given number of toppings. -/
def pizza_combinations (n : ℕ) : ℕ :=
  n + n * (n - 1) / 2

/-- Theorem: The number of different one- and two-topping pizzas that can be ordered from a pizza parlor with 6 toppings is equal to 21. -/
theorem pizza_combinations_six_toppings :
  pizza_combinations 6 = 21 := by
  sorry

end pizza_combinations_six_toppings_l1559_155917


namespace triangle_properties_l1559_155919

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  2 * c * Real.cos B = 2 * a - b →
  -- Prove C = π/3
  C = π / 3 ∧
  -- When c = 3, prove a + b is in the range (3, 6]
  (c = 3 → 3 < a + b ∧ a + b ≤ 6) :=
by sorry

end triangle_properties_l1559_155919


namespace tan_value_from_sin_cos_equation_l1559_155966

theorem tan_value_from_sin_cos_equation (α : Real) 
  (h : Real.sin α + Real.sqrt 2 * Real.cos α = Real.sqrt 3) : 
  Real.tan α = Real.sqrt 2 / 2 := by
  sorry

end tan_value_from_sin_cos_equation_l1559_155966


namespace total_players_is_77_l1559_155916

/-- The number of cricket players -/
def cricket_players : ℕ := 22

/-- The number of hockey players -/
def hockey_players : ℕ := 15

/-- The number of football players -/
def football_players : ℕ := 21

/-- The number of softball players -/
def softball_players : ℕ := 19

/-- Theorem stating that the total number of players is 77 -/
theorem total_players_is_77 : 
  cricket_players + hockey_players + football_players + softball_players = 77 := by
  sorry

end total_players_is_77_l1559_155916


namespace sum_of_reciprocals_squared_l1559_155918

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 →
  b = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 →
  c = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 →
  d = -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 →
  (1/a + 1/b + 1/c + 1/d)^2 = 1600 := by
  sorry

end sum_of_reciprocals_squared_l1559_155918


namespace find_k_value_l1559_155998

/-- Given two functions f and g, prove that k = -15.8 when f(5) - g(5) = 15 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 5 * x^2 - 3 * x + 6) → 
  (∀ x, g x = 2 * x^2 - k * x + 2) → 
  f 5 - g 5 = 15 → 
  k = -15.8 := by
sorry

end find_k_value_l1559_155998


namespace pure_imaginary_product_l1559_155987

def complex (a b : ℝ) : ℂ := Complex.mk a b

theorem pure_imaginary_product (m : ℝ) : 
  let z₁ : ℂ := complex 3 2
  let z₂ : ℂ := complex 1 m
  (z₁ * z₂).re = 0 → m = 3/2 := by
  sorry

end pure_imaginary_product_l1559_155987


namespace haley_tv_watching_l1559_155908

/-- Haley's TV watching problem -/
theorem haley_tv_watching (total_hours sunday_hours : ℕ) 
  (h1 : total_hours = 9)
  (h2 : sunday_hours = 3) :
  total_hours - sunday_hours = 6 := by
  sorry

end haley_tv_watching_l1559_155908


namespace adam_purchases_cost_l1559_155989

/-- Represents the cost of Adam's purchases -/
def total_cost (nuts_quantity : ℝ) (dried_fruits_quantity : ℝ) (nuts_price : ℝ) (dried_fruits_price : ℝ) : ℝ :=
  nuts_quantity * nuts_price + dried_fruits_quantity * dried_fruits_price

/-- Theorem stating that Adam's purchases cost $56 -/
theorem adam_purchases_cost :
  total_cost 3 2.5 12 8 = 56 := by
  sorry

end adam_purchases_cost_l1559_155989


namespace alex_exam_result_l1559_155932

/-- Represents the scoring system and result of a multiple-choice exam -/
structure ExamResult where
  total_questions : ℕ
  correct_points : ℕ
  blank_points : ℕ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions -/
def max_correct_answers (exam : ExamResult) : ℕ :=
  sorry

/-- Theorem stating that for the given exam conditions, the maximum number of correct answers is 38 -/
theorem alex_exam_result :
  let exam : ExamResult :=
    { total_questions := 60
      correct_points := 5
      blank_points := 0
      incorrect_points := -2
      total_score := 150 }
  max_correct_answers exam = 38 := by
  sorry

end alex_exam_result_l1559_155932


namespace scientific_notation_of_595_5_billion_yuan_l1559_155985

def billion : ℝ := 1000000000

theorem scientific_notation_of_595_5_billion_yuan :
  ∃ (a : ℝ) (n : ℤ), 
    595.5 * billion = a * (10 : ℝ) ^ n ∧ 
    1 ≤ |a| ∧ 
    |a| < 10 ∧
    a = 5.955 ∧
    n = 11 := by
  sorry

end scientific_notation_of_595_5_billion_yuan_l1559_155985


namespace triangle_third_side_l1559_155904

theorem triangle_third_side (a b c : ℕ) : 
  a = 3 → b = 8 → c % 2 = 0 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) → 
  c ≠ 12 := by
sorry

end triangle_third_side_l1559_155904


namespace square_value_l1559_155983

theorem square_value (a b : ℝ) (h : ∃ square, square * (3 * a * b) = 3 * a^2 * b) : 
  ∃ square, square = a := by sorry

end square_value_l1559_155983


namespace seventy_eighth_ball_is_green_l1559_155901

def ball_color (n : ℕ) : String :=
  match n % 5 with
  | 0 => "violet"
  | 1 => "red"
  | 2 => "yellow"
  | 3 => "green"
  | 4 => "blue"
  | _ => "invalid"  -- This case should never occur

theorem seventy_eighth_ball_is_green : ball_color 78 = "green" := by
  sorry

end seventy_eighth_ball_is_green_l1559_155901


namespace bruno_score_l1559_155972

/-- Given that Richard's score is 62 and Bruno's score is 14 points lower than Richard's,
    prove that Bruno's score is 48. -/
theorem bruno_score (richard_score : ℕ) (bruno_diff : ℕ) : 
  richard_score = 62 → 
  bruno_diff = 14 → 
  richard_score - bruno_diff = 48 := by
  sorry

end bruno_score_l1559_155972


namespace shopkeeper_profit_percentage_l1559_155999

/-- Given a shopkeeper who sells 15 articles at the cost price of 20 articles, 
    prove that the profit percentage is 1/3. -/
theorem shopkeeper_profit_percentage 
  (cost_price : ℝ) (cost_price_positive : cost_price > 0) : 
  let selling_price := 20 * cost_price
  let total_cost := 15 * cost_price
  let profit := selling_price - total_cost
  let profit_percentage := profit / total_cost
  profit_percentage = 1 / 3 := by sorry

end shopkeeper_profit_percentage_l1559_155999


namespace expression_minimizes_q_l1559_155905

/-- The function q in terms of x and the expression to be determined -/
def q (x : ℝ) (expression : ℝ → ℝ) : ℝ :=
  (expression x)^2 + (x + 1)^2 - 6

/-- The condition that y is least when x = 2 -/
axiom y_min_at_2 : ∀ (y : ℝ → ℝ), ∀ (x : ℝ), y 2 ≤ y x

/-- The relationship between q and y -/
axiom q_related_to_y : ∃ (y : ℝ → ℝ), ∀ (x : ℝ), q x (λ t => t - 2) = y x

/-- The theorem stating that (x - 2) minimizes q when x = 2 -/
theorem expression_minimizes_q :
  ∀ (x : ℝ), q 2 (λ t => t - 2) ≤ q x (λ t => t - 2) :=
by sorry

end expression_minimizes_q_l1559_155905


namespace abs_sum_lower_bound_l1559_155977

theorem abs_sum_lower_bound :
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ 3) ∧
  (∀ ε > 0, ∃ x : ℝ, |x - 1| + |x + 2| < 3 + ε) :=
by sorry

end abs_sum_lower_bound_l1559_155977


namespace correct_regression_coefficients_l1559_155995

-- Define the linear regression equation
def linear_regression (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Define positive correlation
def positively_correlated (a : ℝ) : Prop := a > 0

-- Define the sample means
def x_mean : ℝ := 3
def y_mean : ℝ := 3.5

-- Theorem statement
theorem correct_regression_coefficients (a b : ℝ) :
  positively_correlated a ∧
  linear_regression a b x_mean = y_mean →
  a = 0.4 ∧ b = 2.3 :=
by sorry

end correct_regression_coefficients_l1559_155995


namespace distance_rides_to_car_l1559_155922

/-- The distance Heather walked from the car to the entrance -/
def distance_car_to_entrance : ℝ := 0.3333333333333333

/-- The distance Heather walked from the entrance to the carnival rides -/
def distance_entrance_to_rides : ℝ := 0.3333333333333333

/-- The total distance Heather walked -/
def total_distance : ℝ := 0.75

/-- The theorem states that given the above distances, 
    the distance Heather walked from the carnival rides back to the car 
    is 0.08333333333333337 miles -/
theorem distance_rides_to_car : 
  total_distance - (distance_car_to_entrance + distance_entrance_to_rides) = 0.08333333333333337 := by
  sorry

end distance_rides_to_car_l1559_155922


namespace circle_condition_l1559_155984

/-- 
Theorem: The equation x^2 + y^2 + x + 2my + m = 0 represents a circle if and only if m ≠ 1/2.
-/
theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + x + 2*m*y + m = 0) ↔ m ≠ 1/2 := by
  sorry

end circle_condition_l1559_155984


namespace math_majors_consecutive_probability_math_majors_consecutive_probability_proof_l1559_155970

/-- The probability of all math majors sitting consecutively around a circular table -/
theorem math_majors_consecutive_probability : ℚ :=
  let total_people : ℕ := 12
  let math_majors : ℕ := 5
  let physics_majors : ℕ := 4
  let biology_majors : ℕ := 3
  1 / 330

/-- Proof that the probability of all math majors sitting consecutively is 1/330 -/
theorem math_majors_consecutive_probability_proof :
  math_majors_consecutive_probability = 1 / 330 := by
  sorry

end math_majors_consecutive_probability_math_majors_consecutive_probability_proof_l1559_155970


namespace good_set_properties_l1559_155938

def GoodSet (s : Set ℝ) : Prop :=
  ∀ a ∈ s, (8 - a) ∈ s

theorem good_set_properties :
  (¬ GoodSet {1, 2}) ∧
  (GoodSet {1, 4, 7}) ∧
  (GoodSet {4}) ∧
  (GoodSet {3, 4, 5}) ∧
  (GoodSet {2, 6}) ∧
  (GoodSet {1, 2, 4, 6, 7}) ∧
  (GoodSet {0, 8}) :=
by sorry

end good_set_properties_l1559_155938


namespace policy_support_percentage_l1559_155976

theorem policy_support_percentage
  (total_population : ℕ)
  (men_count : ℕ)
  (women_count : ℕ)
  (men_support_rate : ℚ)
  (women_support_rate : ℚ)
  (h1 : total_population = men_count + women_count)
  (h2 : total_population = 1000)
  (h3 : men_count = 200)
  (h4 : women_count = 800)
  (h5 : men_support_rate = 70 / 100)
  (h6 : women_support_rate = 75 / 100)
  : (men_count * men_support_rate + women_count * women_support_rate) / total_population = 74 / 100 := by
  sorry

end policy_support_percentage_l1559_155976


namespace possible_values_of_a_l1559_155945

def P : Set ℝ := {x | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem possible_values_of_a :
  {a : ℝ | Q a ⊆ P} = {0, 1/3, -1/2} := by
  sorry

end possible_values_of_a_l1559_155945


namespace function_lower_bound_l1559_155935

theorem function_lower_bound (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, a * (Real.exp x + a) - x > 2 * Real.log a + 3/2 := by
  sorry

end function_lower_bound_l1559_155935


namespace fraction_sum_l1559_155956

theorem fraction_sum (a b : ℕ+) (h1 : (a : ℚ) / b = 9 / 16) 
  (h2 : ∀ d : ℕ, d > 1 → d ∣ a → d ∣ b → False) : 
  (a : ℕ) + b = 25 := by
  sorry

end fraction_sum_l1559_155956


namespace sqrt_equation_solution_l1559_155947

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 7) = 9 → x = 74 := by
  sorry

end sqrt_equation_solution_l1559_155947


namespace not_p_false_range_p_necessary_not_sufficient_range_l1559_155974

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 - 2*x + a^2 + 3*a - 3

-- Define proposition p
def p (a : ℝ) : Prop := ∃ x, f x a < 0

-- Define proposition r
def r (a x : ℝ) : Prop := 1 - a ≤ x ∧ x ≤ 1 + a

-- Theorem for part (1)
theorem not_p_false_range (a : ℝ) : 
  ¬(¬(p a)) → a ∈ Set.Ioo (-4 : ℝ) 1 :=
sorry

-- Theorem for part (2)
theorem p_necessary_not_sufficient_range (a : ℝ) :
  (∀ x, r a x → p a) ∧ (∃ x, p a ∧ ¬r a x) → a ∈ Set.Ici 5 :=
sorry

end not_p_false_range_p_necessary_not_sufficient_range_l1559_155974


namespace min_value_theorem_l1559_155903

/-- The minimum value of a function given specific conditions -/
theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) (hmn : m * n > 0) :
  let f := fun x => a^(x - 1) + 1
  let line := fun x y => 2 * m * x + n * y - 4 = 0
  ∃ (x y : ℝ), f x = y ∧ line x y →
  (4 / m + 2 / n : ℝ) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_theorem_l1559_155903


namespace log_equation_solution_l1559_155968

theorem log_equation_solution (b x : ℝ) 
  (h1 : b > 0) 
  (h2 : b ≠ 1) 
  (h3 : x ≠ 1) 
  (h4 : Real.log x / Real.log (b^3) + Real.log b / Real.log (x^3) = 1) : 
  x = b^((3 + Real.sqrt 5) / 2) ∨ x = b^((3 - Real.sqrt 5) / 2) :=
sorry

end log_equation_solution_l1559_155968


namespace function_property_implies_odd_l1559_155969

theorem function_property_implies_odd (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y = f (x + y)) : 
  ∀ x : ℝ, f (-x) = -f x := by
sorry

end function_property_implies_odd_l1559_155969


namespace quadratic_equation_k_value_l1559_155944

theorem quadratic_equation_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 402*x₁ + k = 0 ∧ 
                x₂^2 - 402*x₂ + k = 0 ∧ 
                x₁ + 3 = 80 * x₂) → 
  k = 1985 := by
sorry

end quadratic_equation_k_value_l1559_155944


namespace right_triangle_area_right_triangle_area_proof_l1559_155994

/-- The area of a right triangle with base 15 and height 10 is 75 -/
theorem right_triangle_area : Real → Real → Real → Prop :=
  fun base height area =>
    base = 15 ∧ height = 10 ∧ area = (base * height) / 2 → area = 75

theorem right_triangle_area_proof : right_triangle_area 15 10 75 := by
  sorry

end right_triangle_area_right_triangle_area_proof_l1559_155994


namespace platform_length_l1559_155941

/-- Given a train of length 300 meters that crosses a platform in 45 seconds
    and crosses a signal pole in 18 seconds, prove that the platform length is 450 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 45)
  (h3 : pole_crossing_time = 18) :
  let train_speed := train_length / pole_crossing_time
  let total_distance := train_speed * platform_crossing_time
  train_length + (total_distance - train_length) = 450 :=
by sorry

end platform_length_l1559_155941


namespace stability_comparison_l1559_155951

/-- Represents a student's math exam scores -/
structure StudentScores where
  mean : ℝ
  variance : ℝ
  exam_count : ℕ

/-- Defines the concept of stability for exam scores -/
def more_stable (a b : StudentScores) : Prop :=
  a.variance < b.variance

theorem stability_comparison 
  (student_A student_B : StudentScores)
  (h1 : student_A.mean = student_B.mean)
  (h2 : student_A.exam_count = student_B.exam_count)
  (h3 : student_A.exam_count = 5)
  (h4 : student_A.mean = 102)
  (h5 : student_A.variance = 38)
  (h6 : student_B.variance = 15) :
  more_stable student_B student_A :=
sorry

end stability_comparison_l1559_155951


namespace davids_math_marks_l1559_155973

def english_marks : ℝ := 90
def physics_marks : ℝ := 85
def chemistry_marks : ℝ := 87
def biology_marks : ℝ := 85
def average_marks : ℝ := 87.8
def total_subjects : ℕ := 5

theorem davids_math_marks :
  ∃ (math_marks : ℝ),
    (english_marks + physics_marks + chemistry_marks + biology_marks + math_marks) / total_subjects = average_marks ∧
    math_marks = 92 := by
  sorry

end davids_math_marks_l1559_155973


namespace melissa_shoe_repair_time_l1559_155949

/-- The time Melissa spends repairing her shoes -/
theorem melissa_shoe_repair_time :
  ∀ (buckle_time heel_time : ℕ) (num_shoes : ℕ),
  buckle_time = 5 →
  heel_time = 10 →
  num_shoes = 2 →
  buckle_time * num_shoes + heel_time * num_shoes = 30 :=
by
  sorry

end melissa_shoe_repair_time_l1559_155949


namespace problem_statement_l1559_155953

theorem problem_statement (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hab : a + b = 1) :
  (∃ (min : ℝ), min = 2/3 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → a^2 + 2*b^2 ≥ min) ∧
  (a*x + b*y) * (a*y + b*x) ≥ x*y := by
  sorry

end problem_statement_l1559_155953


namespace min_value_reciprocal_sum_l1559_155912

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
sorry

end min_value_reciprocal_sum_l1559_155912


namespace evaluate_expression_l1559_155981

theorem evaluate_expression : (2^2003 * 3^2005) / 6^2004 = 3/2 := by
  sorry

end evaluate_expression_l1559_155981


namespace expression_value_l1559_155957

theorem expression_value (x y : ℝ) (h : x - 2*y = 3) : 1 - 2*x + 4*y = -5 := by
  sorry

end expression_value_l1559_155957
